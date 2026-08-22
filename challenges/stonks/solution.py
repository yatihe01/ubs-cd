r"""Time Travelling Stonks Man.

We start in 2037 with `capital` and `energy`, may jump between years at a cost of
one energy per year travelled, and must finish back in 2037 with as much cash as
possible.  Buying is capped by each year's `qty` (a stock bought in a given year
is permanently gone); selling is not capped - the sample sells 50 Apple in a year
listing only 10, so a year's quantity governs supply, not demand.

Distance is not the only thing worth buying with the battery
------------------------------------------------------------
Every year is <= 2037, so every trip is into the past and back.  Reaching year `m`
and returning costs at least 2 * (2037 - m), and walking straight down then
straight back up pays exactly that while passing through *every* year in between,
twice - jump cost telescopes, so the intermediate stops are free.

That makes it tempting to conclude the only decision is how deep to go, and to
spend the whole battery on depth.  It is wrong, and expensively so.  A straight
trip gives each year one chance to buy and one to sell, which is enough only if
the budget can absorb a year's entire supply in one go.  When it cannot, running a
*short stretch repeatedly* turns each lap's profit into the next lap's stake and
the money compounds, while a deeper trip just reaches more years it cannot afford.
On a two-year timeline with ten energy, one deep pass returns 20 and five laps of
the same pair return 320.

So the itinerary is searched rather than assumed.  Three families are simulated
and the money picks the winner:

  * straight - down to the deepest reachable year and back, for when the prize is
    simply far away.  Only the deepest is tried: it passes through every shallower
    year, so it can do anything a shallower trip could.
  * cycling - descend, then run laps over a short stretch, for when the prize is
    close but too expensive to take in one bite.
  * lap-then-dive - laps near home first, then one deeper trip, for when the
    starting capital is too small to exploit a distant bargain on arrival.

Whichever route is chosen, each year appears several times and shares one pool of
`qty` across every visit.

The trading problem
-------------------
A forward simulation over one itinerary's stops, holding cash and shares.  Every
candidate route is run through it and the one ending on the most cash wins.  At
each stop:

  1. Liquidate every holding the current year prices, into cash.
  2. Offer that cash a menu of opportunities, each a (price, available, sell price)
     triple where the sell price is the best price for that stock at any *later*
     stop.  Alongside the real ones sits a synthetic "repurchase" entry for each
     stock just liquidated, priced at today's price with no quantity cost.
  3. Allocate the cash across the menu, then execute only the entries belonging to
     this stop.

The repurchase entry is what makes step 1 safe: buying it back is exactly holding.
It falls out of the allocation for free that a holding is kept while its best
future price beats today's (ratio > 1), sold the moment today *is* its peak (ratio
<= 1, so it never enters the menu), and sold early when the cash is worth more
redeployed into a stronger trade.  Encoding "hold" as an option rather than a rule
means one mechanism decides all three.

Allocation is a bounded knapsack - maximise sum(shares * (sell - buy)) subject to
sum(shares * buy) <= cash - so ratio-greedy gives the LP optimum but can miss on
integrality (60-cost/70-profit crowds out two 50-cost/50-profit items at cash 100).
Re-running the greedy with each of the leading items forced first and keeping the
best recovers those cases at negligible cost.
"""

from __future__ import annotations

import heapq
import math
import time
from itertools import zip_longest
from typing import Any, Iterator, NamedTuple


# The year the machine departs from and must return to.
START_YEAR = 2037

# How many ratio-greedy restarts to try when allocating cash.  Each restart forces
# one leading opportunity to the front, which is what repairs the integrality gap;
# the items past this cut are too cheap for their rounding to matter.
RESTART_LIMIT = 12

# Restarts repair integer rounding, which only bites when one purchase is a large
# fraction of the budget - the case on small menus, where they are worth several
# percent.  On a broad menu the budget spreads over many small purchases and the
# gap all but vanishes: measured on a 100-year, 30-stock timeline, twelve restarts
# bought 0.0006% more profit for eight times the runtime.  So they are spent where
# they pay and skipped where they do not.
#
# The menu itself is never truncated.  Dropping the tail looks safe because cash is
# spent richest-return-first, but a large budget genuinely reaches deep into it -
# capping at 160 entries cost 75% of the profit on that same timeline.
BROAD_MENU = 48

# Laps compound the money, so the useful count is logarithmic in how much supply
# there is to work through: once the budget can take a year's whole stock in one
# go, another lap adds nothing.  The cap is generous because a lap that turns out
# barren costs no energy at all - rendering drops stops that never trade - so the
# only price of overshooting is simulation time, which the clock already governs.
MAX_LAPS = 160

# Wall-clock budget for a whole batch, split across its cases.  The search is
# anytime: candidate itineraries are tried best-first and the clock decides how
# many get run, so spending the budget buys breadth rather than risking a timeout.
# Held under ten minutes with room for the response itself.
TIME_BUDGET = 540.0

# Floor on per-case time, so a long batch still searches each case properly rather
# than slicing the budget into uselessly thin pieces.
MIN_CASE_SECONDS = 2.0

# Longest itinerary to simulate, by timeline size.  Simulation costs about
# O(stops^2 * stocks), so an over-long route buys one deep candidate at the price
# of many shallower ones - the clock is a better throttle than the route length.
STOP_BUDGETS = ((240, 600), (1200, 400), (4000, 260), (None, 180))

# How many of the strongest stretches to pair against each other.  Pairing is
# quadratic in this and each pair yields several lap splits, so it buys breadth
# quickly - the strongest stretches are the ones worth combining anyway.
STRETCH_PAIRING = 14


def _stop_budget(years: int, stocks: int) -> int:
    scale = years * max(stocks, 1)
    for limit, stops in STOP_BUDGETS:
        if limit is None or scale <= limit:
            return stops
    return STOP_BUDGETS[-1][1]


class Opportunity(NamedTuple):
    """A purchase that can be made now or later, and the best price it can be sold
    into afterwards.  `year` is None for a repurchase - keeping shares already held
    consumes none of any year's quantity.  `sell_stop` is when the money comes
    back, which is what lets one trade fund another."""

    ratio: float
    stop: int
    year: int | None
    stock: str
    price: int
    available: int
    sell_price: int
    sell_stop: int


def _coerce_int(value: Any) -> int | None:
    """Ints, int-valued floats and numeric strings, or None if it is not a usable
    whole number.  Booleans are rejected: `True` is an int in Python but never a
    price or a quantity."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == int(value) else None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except (ValueError, AttributeError):
            return None
    return None


def _coerce_positive_int(value: Any) -> int | None:
    number = _coerce_int(value)
    return number if number is not None and number > 0 else None


def _coerce_quantity(value: Any) -> int | None:
    """Quantities may legitimately be zero - the brief allows `0 <= qty`.  A
    zero-quantity listing still matters: it cannot be bought from, but selling is
    not capped by quantity, so its price is a real place to sell into."""
    number = _coerce_int(value)
    return number if number is not None and number >= 0 else None


def parse_timeline(raw: Any) -> dict[int, dict[str, tuple[int, int]]]:
    """`{year: {stock: (price, qty)}}` for the years worth visiting.

    Zero-quantity listings are kept: nothing can be bought from them, but they are
    still somewhere to sell into.  Unusable entries are dropped rather than
    rejected - a year after 2037 is unreachable going backwards, and a malformed
    listing should not cost the rest of the batch its answer.
    """
    markets: dict[int, dict[str, tuple[int, int]]] = {}
    if not isinstance(raw, dict):
        return markets

    for year_key, listings in raw.items():
        year = _coerce_positive_int(year_key)
        if year is None or year > START_YEAR or not isinstance(listings, dict):
            continue
        market: dict[str, tuple[int, int]] = {}
        for stock, listing in listings.items():
            if not isinstance(stock, str) or not isinstance(listing, dict):
                continue
            price = _coerce_positive_int(listing.get("price"))
            quantity = _coerce_quantity(listing.get("qty"))
            if price is None or quantity is None:
                continue
            market[stock] = (price, quantity)
        if market:
            markets[year] = market
    return markets


def _straight_route(years: list[int], deepest: int) -> list[int]:
    """Descend to `deepest`, then climb back.  The turning point appears once;
    every other year appears twice."""
    within = [year for year in years if year >= deepest]
    descending = sorted(within, reverse=True)
    ascending = [year for year in sorted(within) if year > deepest]
    return descending + ascending


def _cycling_route(
    years: list[int], deepest: int, peak: int, laps: int, max_stops: int
) -> list[int]:
    """Descend to `deepest`, run `laps` extra round trips up to `peak` and back,
    then climb home.

    A straight there-and-back visits each year once on the way down and once on
    the way up, which is one chance to buy and one to sell.  That is enough only
    when the budget can absorb a year's whole supply in a single go.  When it
    cannot, running the same stretch again turns the profit from one lap into the
    stake for the next, and the money compounds - which is worth far more than the
    extra years a deeper trip would have reached.

    A lap over the stretch costs `2 * (peak - deepest)`, so a short, steep stretch
    can be run many times on the same battery that one long descent would drain.
    """
    below = [year for year in years if year >= deepest]
    descending = sorted(below, reverse=True)
    inside = [year for year in sorted(below) if deepest <= year <= peak]
    up = [year for year in inside if year > deepest]
    down = sorted((year for year in inside if year < peak), reverse=True)

    home = [year for year in sorted(below) if year > deepest]
    lap = up + down
    if lap:
        room = (max_stops - len(descending) - len(home)) // len(lap)
        laps = max(0, min(laps, room))
    else:
        laps = 0
    return descending + lap * laps + home


def _lap_then_dive_route(
    years: list[int], lap_floor: int, deepest: int, laps: int, max_stops: int
) -> list[int]:
    """Run `laps` shallow round trips down to `lap_floor` first, then spend the
    winnings on one deeper trip to `deepest`.

    The mirror of `_cycling_route`, and it matters when the starting capital is too
    small to exploit the deep bargain on arrival.  Compounding close to home is
    cheap per lap, and the money it builds is what makes the expensive trip worth
    taking - diving first would reach the bargain with nothing to spend.
    """
    lap = _straight_route(years, lap_floor)
    dive = _straight_route(years, deepest)
    if not lap:
        return dive
    laps = max(0, min(laps, (max_stops - len(dive)) // len(lap)))
    return lap * laps + dive


def _two_stretch_route(
    years: list[int],
    near: tuple[int, int],
    near_laps: int,
    deep: tuple[int, int],
    deep_laps: int,
    max_stops: int,
) -> list[int]:
    """Lap one stretch, carry the winnings down, lap a second deeper one, go home.

    Neither single-stretch family can express this.  Cycling picks one stretch and
    stays on it; lap-then-dive can only spend the winnings on a single pass.  When
    the best engine is deep but the starting capital is too small to turn it, the
    winning shape is to compound cheaply near home *and then* compound again on the
    better stretch once there is enough money to work it.

    The travel is free relative to reaching the deeper floor at all: descending
    2037 -> near floor -> deep floor -> 2037 covers exactly 2 * (2037 - deep floor),
    the same as a straight trip to the deeper floor, so the whole cost of the near
    stretch is its own laps.
    """
    near_floor, near_peak = near
    deep_floor, deep_peak = deep

    def lap(floor: int, peak: int) -> list[int]:
        inside = [year for year in years if floor <= year <= peak]
        rising = [year for year in sorted(inside) if year > floor]
        falling = sorted((year for year in inside if year < peak), reverse=True)
        return rising + falling

    approach = sorted((y for y in years if y >= near_floor), reverse=True)
    descent = sorted(
        (y for y in years if deep_floor <= y < near_floor), reverse=True
    )
    home = [y for y in sorted(years) if y > deep_floor]
    near_lap, deep_lap = lap(*near), lap(*deep)

    room = max_stops - len(approach) - len(descent) - len(home)
    if near_lap:
        near_laps = max(0, min(near_laps, room // len(near_lap)))
        room -= near_laps * len(near_lap)
    else:
        near_laps = 0
    if deep_lap:
        deep_laps = max(0, min(deep_laps, room // len(deep_lap)))
    else:
        deep_laps = 0

    return (
        approach
        + near_lap * near_laps
        + descent
        + deep_lap * deep_laps
        + home
    )


def _energy_spent(actions: list[str]) -> int:
    return sum(
        abs(int(parts[2]) - int(parts[1]))
        for parts in (action.split("-") for action in actions)
        if parts[0] == "j"
    )


def _best_later_prices(
    route: list[int], markets: dict[int, dict[str, tuple[int, int]]]
) -> list[dict[str, tuple[int, int]]]:
    """`later[i][stock]` is `(best price, stop)` for that stock at any stop after
    `i` - what a share bought at stop `i` can be sold for, and when.

    Ties go to the earlier stop, so the cash comes back as soon as it can and is
    free to fund something else.
    """
    later: list[dict[str, tuple[int, int]]] = [{} for _ in route]
    for index in range(len(route) - 2, -1, -1):
        best = dict(later[index + 1])
        following = index + 1
        for stock, (price, _) in markets[route[following]].items():
            if price >= best.get(stock, (0, 0))[0]:
                best[stock] = (price, following)
        later[index] = best
    return later


def _menu_order(opportunity: Opportunity) -> tuple[float, int, str]:
    """Richest return per dollar first; ties to the earliest stop, so the money is
    free again sooner."""
    return (-opportunity.ratio, opportunity.stop, opportunity.stock)


def _build_menu(
    route: list[int],
    markets: dict[int, dict[str, tuple[int, int]]],
    later_prices: list[dict[str, tuple[int, int]]],
) -> list[tuple[float, int, int, str, int, int, int]]:
    """Every purchase the itinerary offers, in ratio order, as raw tuples.

    A year reached twice appears twice, deliberately.  Buying on the way down
    leaves more stops to sell into, but buying on the way back can be paid for by a
    trade that settles in between - which visit is better depends on the cash
    curve, so it is the allocator's call rather than a rule hardcoded here.  Both
    entries draw on that year's one pool.

    Only availability changes as trading proceeds, so everything here is computed
    once and filtered per stop.
    """
    menu: list[tuple[float, int, int, str, int, int, int]] = []
    for stop, year in enumerate(route):
        reachable = later_prices[stop]
        for stock, (price, _) in markets[year].items():
            sell_price, sell_stop = reachable.get(stock, (0, stop))
            if sell_price <= price:
                continue
            menu.append(
                (sell_price / price, stop, year, stock, price, sell_price, sell_stop)
            )
    menu.sort(key=lambda item: (-item[0], item[1], item[3]))
    return menu


def _greedy_allocation(
    cash: int,
    opportunities: list[Opportunity],
    stops: int,
    lead: int | None = None,
) -> tuple[dict[int, int], int]:
    """Spend `cash` down the menu, richest return per dollar first, optionally
    forcing one opportunity to the front.  Returns the shares bought per menu index
    and the profit that implies.

    Rather than draw everything from one running total, this tracks the cash curve
    across the remaining stops, because money recycles: a trade that sells before a
    later one buys hands its stake back, with profit, in time to fund it.  A single
    total gets this wrong in both directions - it holds cash back for a trade an
    earlier one was about to pay for, and it lets a trade spend money a later,
    better one was depending on.

    A trade only ties its stake up between buying and selling, so what it can
    afford is the *lowest* the cash curve gets over exactly that window.  Keeping
    the curve non-negative everywhere is what makes the plan actually payable.
    """
    order = list(range(len(opportunities)))
    if lead is not None:
        order.remove(lead)
        order.insert(0, lead)

    curve = [cash] * stops
    plan: dict[int, int] = {}
    # One year's shares are one pool no matter how many stops can reach it, so
    # allocating against a year draws that pool down for every other entry on it.
    pool: dict[tuple[int | None, str], int] = {}
    total_profit = 0
    for index in order:
        opportunity = opportunities[index]
        key = (opportunity.year, opportunity.stock)
        if key not in pool:
            pool[key] = opportunity.available
        stock_left = pool[key]
        if stock_left <= 0:
            continue
        held_from, held_until = opportunity.stop, opportunity.sell_stop
        if held_from >= held_until:
            continue
        # The window minimum can never exceed its first point, so this rejects
        # what the budget cannot reach without scanning the whole window - which is
        # most of a long menu once the cash is committed.
        if curve[held_from] < opportunity.price:
            continue
        affordable = min(curve[held_from:held_until])
        if affordable < opportunity.price:
            continue
        shares = min(stock_left, affordable // opportunity.price)
        if shares <= 0:
            continue
        pool[key] -= shares
        stake = shares * opportunity.price
        gain = shares * (opportunity.sell_price - opportunity.price)
        for stop in range(held_from, held_until):
            curve[stop] -= stake
        for stop in range(held_until, stops):
            curve[stop] += gain
        plan[index] = shares
        total_profit += gain
    return plan, total_profit


def _allocate(
    cash: int, opportunities: list[Opportunity], stops: int
) -> dict[int, int]:
    """Best allocation found across the plain ratio-greedy and the forced-lead
    restarts that repair its integrality gap."""
    best_plan, best_profit = _greedy_allocation(cash, opportunities, stops)
    if len(opportunities) > BROAD_MENU:
        return best_plan
    for lead in range(min(len(opportunities), RESTART_LIMIT)):
        plan, profit = _greedy_allocation(cash, opportunities, stops, lead)
        if profit > best_profit:
            best_plan, best_profit = plan, profit
    return best_plan


def _simulate(
    route: list[int],
    markets: dict[int, dict[str, tuple[int, int]]],
    capital: int,
) -> tuple[int, list[list[str]]]:
    """Trade the itinerary through and report the cash it ends on, with the trades
    made at each stop."""
    later_prices = _best_later_prices(route, markets)
    menu = _build_menu(route, markets, later_prices)

    supply = {
        (year, stock): quantity
        for year, market in markets.items()
        for stock, (_, quantity) in market.items()
    }
    cash = capital
    holdings: dict[str, int] = {}
    trades: list[list[str]] = [[] for _ in route]

    for index, year in enumerate(route):
        market = markets[year]
        later = later_prices[index]

        # 1. Liquidate everything this year can price, so the allocation below
        #    decides what to do with the proceeds rather than assuming.
        liquidated: dict[str, int] = {}
        for stock in sorted(holdings):
            held = holdings[stock]
            if held <= 0 or stock not in market:
                continue
            price = market[stock][0]
            cash += held * price
            liquidated[stock] = held
            holdings[stock] = 0

        # 2. Build the menu: every purchase still reachable, in ratio order, plus a
        #    repurchase entry for each stock just liquidated.
        #
        #    Everything but availability was settled up front, so this filters the
        #    pre-sorted menu rather than rebuilding and re-sorting it - the same
        #    answer without paying O(stops * stocks * log) at every stop.
        opportunities = [
            Opportunity(
                ratio=ratio,
                stop=stop,
                year=stop_year,
                stock=stock,
                price=price,
                available=supply[(stop_year, stock)],
                sell_price=sell_price,
                sell_stop=sell_stop,
            )
            for ratio, stop, stop_year, stock, price, sell_price, sell_stop in menu
            if stop >= index and supply[(stop_year, stock)] > 0
        ]

        repurchases: list[Opportunity] = []
        for stock, held in liquidated.items():
            price = market[stock][0]
            sell_price, sell_stop = later.get(stock, (0, index))
            if sell_price <= price:
                continue  # today is this stock's peak; let it stay sold
            repurchases.append(
                Opportunity(
                    ratio=sell_price / price,
                    stop=index,
                    year=None,
                    stock=stock,
                    price=price,
                    available=held,
                    sell_price=sell_price,
                    sell_stop=sell_stop,
                )
            )
        if repurchases:
            repurchases.sort(key=_menu_order)
            opportunities = list(
                heapq.merge(opportunities, repurchases, key=_menu_order)
            )

        plan = _allocate(cash, opportunities, len(route))

        # 3. Execute this stop's share of the plan.  Repurchases first: they only
        #    reduce how much of the liquidation is really sold.
        kept: dict[str, int] = {}
        purchases: list[tuple[str, int]] = []
        for menu_index, shares in plan.items():
            opportunity = opportunities[menu_index]
            if opportunity.stop != index:
                continue  # reserved for a later stop
            if opportunity.year is None:
                kept[opportunity.stock] = shares
            else:
                purchases.append((opportunity.stock, shares))

        # Every quantity below is re-capped against the cash actually in hand.  The
        # allocator reasons about cash arriving from trades it has only planned, so
        # it can promise more than this stop can pay for; the plan is a preference
        # order, and this is where it meets the balance.
        for stock, held in liquidated.items():
            price = market[stock][0]
            keeping = min(kept.get(stock, 0), held, cash // price)
            if keeping > 0:
                cash -= keeping * price
                holdings[stock] = holdings.get(stock, 0) + keeping
            sold = held - keeping
            if sold > 0:
                trades[index].append(f"s-{stock}-{sold}")

        for stock, shares in sorted(purchases):
            price = market[stock][0]
            shares = min(shares, supply.get((year, stock), 0), cash // price)
            if shares <= 0:
                continue
            cash -= shares * price
            holdings[stock] = holdings.get(stock, 0) + shares
            supply[(year, stock)] -= shares
            trades[index].append(f"b-{stock}-{shares}")

    return cash, trades


def _render(route: list[int], trades: list[list[str]]) -> list[str]:
    """Turn per-stop trades into the action list, jumping only to years that
    actually trade.  Skipping barren years costs nothing - jump distance
    telescopes - and keeps the itinerary within budget by construction, since the
    deepest traded year can only be shallower than the deepest reachable one."""
    actions: list[str] = []
    current = START_YEAR
    for index, year in enumerate(route):
        if not trades[index]:
            continue
        if year != current:
            actions.append(f"j-{current}-{year}")
            current = year
        actions.extend(trades[index])
    if not actions:
        return []
    if current != START_YEAR:
        actions.append(f"j-{current}-{START_YEAR}")
    return actions


def _candidate_routes(
    reachable: list[int],
    markets: dict[int, dict[str, tuple[int, int]]],
    energy: int,
) -> "Iterator[list[int]]":
    """Itineraries worth trying, best-looking first, for as long as the caller
    keeps asking.

    Three families.  Straight there-and-back covers a prize that is simply far
    away.  Cycling covers one that is close but too expensive to take in one bite,
    where the battery is better spent on laps than on distance.  Lap-then-dive
    covers a distant prize the starting capital cannot use on arrival.  Which wins
    depends on supply, prices and budget together and is not decidable up front,
    so they are simulated and the money decides.

    There are O(years^2) stretches in each cycling family, far too many to run, so
    they are scored first and yielded best-first.  A lap multiplies the money by
    roughly the best ratio inside the stretch, which makes `laps * log(ratio)` a
    ranking by the wealth it could compound to.  Nothing is truncated here - the
    caller stops when its clock runs out, so a small case gets an exhaustive search
    and a large one still spends its whole budget on the most promising stretches.
    """
    ascending = sorted(reachable)
    max_stops = _stop_budget(
        len(ascending), max((len(market) for market in markets.values()), default=1)
    )
    # Only the deepest straight trip is worth simulating: it passes through every
    # shallower year on the way, so it can do anything a shallower trip could, and
    # the battery it saves has nowhere else to go within this family.
    yield _straight_route(reachable, ascending[0])

    # Best buy price and best sell price for every stretch, swept in one pass so
    # scoring stays O(years^2 * stocks) rather than re-scanning per pair.
    cycles: list[tuple[float, int, int, int]] = []
    for low_index, deepest in enumerate(ascending):
        travel = 2 * (START_YEAR - deepest)
        if travel > energy:
            continue
        cheapest: dict[str, int] = {}
        dearest: dict[str, int] = {}
        for peak in ascending[low_index:]:
            for stock, (price, quantity) in markets[peak].items():
                if quantity > 0 and price < cheapest.get(stock, price + 1):
                    cheapest[stock] = price
                if price > dearest.get(stock, 0):
                    dearest[stock] = price
            span = peak - deepest
            if span <= 0:
                continue
            laps = (energy - travel) // (2 * span)
            if laps <= 0:
                continue
            ratio = max(
                (dearest[stock] / cheapest[stock] for stock in cheapest),
                default=1.0,
            )
            if ratio <= 1.0:
                continue
            laps = min(laps, MAX_LAPS)
            cycles.append((laps * math.log(ratio), deepest, peak, laps))
    cycles.sort(reverse=True)

    # Compound close to home first, then spend the winnings on one deeper trip.
    warmups: list[tuple[float, int, int, int]] = []
    best_ratio_above: dict[int, float] = {}
    for floor_index, lap_floor in enumerate(ascending):
        cheapest = {}
        dearest = {}
        for year in ascending[floor_index:]:
            for stock, (price, quantity) in markets[year].items():
                if quantity > 0 and price < cheapest.get(stock, price + 1):
                    cheapest[stock] = price
                if price > dearest.get(stock, 0):
                    dearest[stock] = price
        best_ratio_above[lap_floor] = max(
            (dearest[stock] / cheapest[stock] for stock in cheapest), default=1.0
        )
    for lap_floor in ascending:
        lap_travel = 2 * (START_YEAR - lap_floor)
        ratio = best_ratio_above[lap_floor]
        if lap_travel <= 0 or ratio <= 1.0:
            continue
        for deepest in ascending:
            dive_travel = 2 * (START_YEAR - deepest)
            if dive_travel <= lap_travel or dive_travel > energy:
                continue
            laps = min((energy - dive_travel) // lap_travel, MAX_LAPS)
            if laps <= 0:
                continue
            warmups.append((laps * math.log(ratio), lap_floor, deepest, laps))
    warmups.sort(reverse=True)

    # Pairs of stretches: compound cheaply near home, then again on a better but
    # deeper engine.  There are O(years^4) of these, so only the strongest few
    # stretches are paired up, and the laps between them are split a handful of
    # ways rather than optimised - the clock spends its time simulating, not
    # searching a split that the simulation itself will judge.
    ranked = sorted(
        {
            (deepest, peak): (ratio, span)
            for ratio, deepest, peak, span in (
                (math.exp(score / max(laps, 1)), lo, hi, hi - lo)
                for score, lo, hi, laps in cycles
            )
        }.items(),
        key=lambda item: -item[1][0],
    )[:STRETCH_PAIRING]

    pairs: list[tuple[float, tuple[int, int], int, tuple[int, int], int]] = []
    for (near_floor, near_peak), (near_ratio, near_span) in ranked:
        for (deep_floor, deep_peak), (deep_ratio, deep_span) in ranked:
            if deep_floor > near_floor or near_span <= 0 or deep_span <= 0:
                continue
            spare = energy - 2 * (START_YEAR - deep_floor)
            if spare <= 0:
                continue
            budget = spare // 2
            for numerator, denominator in ((0, 1), (1, 3), (1, 2), (2, 3), (1, 1)):
                near_laps = min(
                    MAX_LAPS, budget * numerator // denominator // near_span
                )
                left = budget - near_laps * near_span
                deep_laps = min(MAX_LAPS, left // deep_span)
                if near_laps <= 0 and deep_laps <= 0:
                    continue
                score = near_laps * math.log(near_ratio) + deep_laps * math.log(
                    deep_ratio
                )
                pairs.append(
                    (
                        score,
                        (near_floor, near_peak),
                        near_laps,
                        (deep_floor, deep_peak),
                        deep_laps,
                    )
                )
    pairs.sort(key=lambda item: -item[0])

    # Interleaved, so no family is starved if the clock stops early.
    for cycle, warmup, pair in zip_longest(cycles, warmups, pairs):
        if cycle is not None:
            _, deepest, peak, laps = cycle
            yield _cycling_route(reachable, deepest, peak, laps, max_stops)
        if warmup is not None:
            _, lap_floor, deepest, laps = warmup
            yield _lap_then_dive_route(
                reachable, lap_floor, deepest, laps, max_stops
            )
        if pair is not None:
            _, near, near_laps, deep, deep_laps = pair
            yield _two_stretch_route(
                reachable, near, near_laps, deep, deep_laps, max_stops
            )


def solve_case(case: Any, deadline: float | None = None) -> list[str]:
    """The action list for one test case, or `[]` when nothing is worth doing -
    staying in 2037 is always legal and always costs nothing.

    `deadline` is a `time.monotonic()` reading to stop searching at.  The first
    candidate is always simulated, so a deadline already in the past still yields a
    real answer rather than an empty one.
    """
    if not isinstance(case, dict):
        return []

    energy = _coerce_positive_int(case.get("energy")) or 0
    capital = _coerce_positive_int(case.get("capital")) or 0
    markets = parse_timeline(case.get("timeline"))
    if energy < 2 or capital <= 0 or not markets:
        return []

    # Half the battery goes on the return leg, so this is how far back we can get.
    earliest = START_YEAR - energy // 2
    reachable = [year for year in markets if year >= earliest]
    if not reachable:
        return []

    best_actions: list[str] = []
    best_cash = capital
    for tried, route in enumerate(_candidate_routes(reachable, markets, energy)):
        if tried and deadline is not None and time.monotonic() >= deadline:
            break
        cash, trades = _simulate(route, markets, capital)
        if cash <= best_cash:
            continue
        actions = _render(route, trades)
        # Barren stops are dropped by rendering, which can only shorten the trip,
        # so this should always hold - checked because an over-budget answer scores
        # nothing at all, and silently keeping one would be the worst outcome.
        if _energy_spent(actions) > energy:
            continue
        best_actions, best_cash = actions, cash
    return best_actions


def solve_batch(batch: Any) -> list[list[str]]:
    """One action list per test case, in the order received.

    The wall-clock budget is shared out as we go rather than divided up front, so a
    case that finishes early hands the rest of its slice to whatever follows.  Each
    case is guaranteed a floor of time, because slicing a long batch strictly
    evenly would leave every case too little to search with.
    """
    if not isinstance(batch, list):
        raise ValueError("request body must be a JSON array of test cases")

    finish_by = time.monotonic() + TIME_BUDGET
    results = []
    for index, case in enumerate(batch):
        remaining_cases = len(batch) - index
        share = max(
            MIN_CASE_SECONDS, (finish_by - time.monotonic()) / remaining_cases
        )
        results.append(solve_case(case, min(finish_by, time.monotonic() + share)))
    return results
