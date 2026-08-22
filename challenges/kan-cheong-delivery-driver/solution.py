import heapq
import json
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from fractions import Fraction

_MICRO = timedelta(microseconds=1)
MIDWAY_BLOCK = "stall"


def _parse_time(s):
    """Parse ISO-8601, keeping the original UTC offset (naive -> UTC)."""
    dt = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _tz_style(s):
    """How the input wrote its timezone, so the output can mirror it."""
    s = str(s).strip()
    if s.endswith(("Z", "z")):
        return "Z"
    return "offset" if ("+" in s[10:] or "-" in s[10:]) else "naive"


def _rel(dt, t0):
    """Exact seconds from t0 to dt; int when whole, Fraction otherwise."""
    micros = (dt - t0) // _MICRO
    return micros // 1000000 if micros % 1000000 == 0 else Fraction(micros, 1000000)


def _fmt_time(base_dt, offset, style="Z"):
    # datetime is microsecond-resolution; round once, rather than truncating the
    # fractional second and potentially disagreeing with total_duration_sec.
    micros = round(Fraction(offset) * 1000000)
    dt = base_dt + timedelta(microseconds=micros)
    stamp = dt.strftime("%Y-%m-%dT%H:%M:%S")
    if dt.microsecond:
        stamp += ("%.6f" % (dt.microsecond / 1000000))[1:].rstrip("0")
    if style == "Z":
        return stamp + "Z"
    if style == "naive":
        return stamp
    off = dt.utcoffset() or timedelta(0)
    sign = "-" if off < timedelta(0) else "+"
    mins = abs(int(off.total_seconds())) // 60
    return f"{stamp}{sign}{mins // 60:02d}:{mins % 60:02d}"


def _num(x):
    f = Fraction(x)
    return int(f) if f.denominator == 1 else float(f)


def _coord(c):
    """Validate the integer coordinate without losing precision via float."""
    if not isinstance(c, int) or isinstance(c, bool):
        raise ValueError(f"Invalid coordinate component: {c!r}")
    return c


def _key(pt):
    return (_coord(pt[0]), _coord(pt[1]))


def _dur(x):
    """Validate the duration range stated by the challenge."""
    if (
        not isinstance(x, int)
        or isinstance(x, bool)
        or not 0 <= x <= 999
    ):
        raise ValueError(f"Invalid base_duration_sec: {x!r}")
    return x


def _normalize_windows(windows):
    """Merge windows into disjoint segments using the slowest active factor.

    Events are processed with end-before-start semantics at the same timestamp,
    matching half-open windows [start, end).  A lazy min-heap makes the sweep
    O(k log k), after which traversal can locate the active segment by binary
    search instead of rescanning every obstruction at every boundary.
    """
    events = {}
    for start, end, factor in windows:
        starts, _ = events.setdefault(start, ([], []))
        starts.append(factor)
        _, ends = events.setdefault(end, ([], []))
        ends.append(factor)

    boundaries = sorted(events)
    active = {}
    factors = []
    normalized = []

    for index, start in enumerate(boundaries[:-1]):
        starts, ends = events[start]
        for factor in ends:
            remaining = active[factor] - 1
            if remaining:
                active[factor] = remaining
            else:
                del active[factor]
        for factor in starts:
            active[factor] = active.get(factor, 0) + 1
            heapq.heappush(factors, factor)
        while factors and factors[0] not in active:
            heapq.heappop(factors)

        end = boundaries[index + 1]
        if not factors:
            continue
        factor = factors[0]
        if normalized and normalized[-1][1] == start and normalized[-1][2] == factor:
            previous_start, _, _ = normalized[-1]
            normalized[-1] = (previous_start, end, factor)
        else:
            normalized.append((start, end, factor))

    return normalized




class Solver:

    def __init__(self, payload):
        self.t0 = _parse_time(payload["start_time"])
        self.tz_style = _tz_style(payload["start_time"])
        self.start = _key(payload["start_coordinate"])
        self.end = _key(payload["end_coordinate"])

        # node -> [(edge_id, neighbour, base_duration)]
        self.adj = {}
        for pt in payload.get("nodes") or []:
            self.adj.setdefault(_key(pt), [])
        for e in payload.get("edges") or []:
            a, b = _key(e["node1"]), _key(e["node2"])
            d = _dur(e["base_duration_sec"])
            eid = e["edge_id"]
            self.adj.setdefault(a, []).append((eid, b, d))
            if a != b:                                   # self-loop once only
                self.adj.setdefault(b, []).append((eid, a, d))

        # (edge_id, from, to) -> [(start, end, factor)] and window boundaries
        self.obs = {}
        self.bounds = {}
        self.obs_starts = {}
        self.fastest = {}          # (edge_id, from, to) -> max factor seen
        for o in payload.get("obstructions") or []:
            k = (o["edge_id"], _key(o["edge"]["from"]), _key(o["edge"]["to"]))
            s = _rel(_parse_time(o["start_time"]), self.t0)
            t = _rel(_parse_time(o["end_time"]), self.t0)
            if t < s:
                raise ValueError("Obstruction end_time precedes start_time")
            if t == s:
                continue                                 # empty window
            f = Fraction(o["speed_factor"])
            if f < 0:
                raise ValueError(f"Invalid speed_factor: {f!r}")
            self.obs.setdefault(k, []).append((s, t, f))
        for k, ws in self.obs.items():
            normalized = _normalize_windows(ws)
            self.obs[k] = normalized
            self.obs_starts[k] = [s for s, _, _ in normalized]
            self.bounds[k] = sorted(
                {boundary for s, t, _ in normalized for boundary in (s, t)}
            )
            self.fastest[k] = max(
                [Fraction(1)] + [f for _, _, f in normalized]
            )

        self.horizon = max([t for ws in self.obs.values() for _, t, _ in ws] + [0])
        # Hard blocks are the ONLY thing that breaks FIFO, so past the last one
        # the earliest arrival at a node dominates every later arrival and a
        # single label per node suffices.  With no blocks at all this is 0 and
        # the whole search degenerates to ordinary A*.
        self.block_horizon = max(
            [t for ws in self.obs.values() for _, t, f in ws if f == 0] + [0])


    def probe(self, edge_id, u, v, base, t):
        """Attempt the directed traversal (u -> v) entering at time t.

        ("arrive", arrival_time, base)   traversal completes
        ("block", stop_time, covered)    a hard block halts the driver `covered`
                                         base-seconds in (only in "uturn" mode;
                                         covered is always > 0)
        ("deny", None, 0)                the move is illegal outright
        """
        key = (edge_id, u, v)
        windows = self.obs.get(key)
        if not windows:
            return ("arrive", t + base, base)            # unobstructed direction
        bounds = self.bounds[key]
        starts = self.obs_starts[key]
        remaining = base
        now = t
        while True:
            segment = bisect_right(starts, now) - 1
            if segment >= 0 and now < windows[segment][1]:
                f = windows[segment][2]
            else:
                f = 1                                    # free-flowing
            if f == 0:                                   # blocked, not delayed
                if now == t:
                    return ("deny", None, 0)             # never enter a blocked edge
                if MIDWAY_BLOCK == "stall":
                    i = bisect_right(bounds, now)
                    if i == len(bounds):
                        return ("deny", None, 0)         # window never lifts
                    now = bounds[i]                      # stall it out
                    continue
                if MIDWAY_BLOCK == "uturn":
                    return ("block", now, base - remaining)
                return ("deny", None, 0)                 # "void"
            if remaining == 0:
                return ("arrive", now, base)
            i = bisect_right(bounds, now)
            if i == len(bounds):                         # no further changes
                return ("arrive", now + (remaining / f if f != 1 else remaining), base)
            nb = bounds[i]
            burn = (nb - now) * f                        # base-seconds burned
            if burn >= remaining:
                return ("arrive", now + (remaining / f if f != 1 else remaining), base)
            remaining -= burn
            now = nb

    def traverse(self, edge_id, u, v, base, t):
        """Arrival time entering (u -> v) at time t, or None if not possible."""
        kind, when, _ = self.probe(edge_id, u, v, base, t)
        return when if kind == "arrive" else None


    def _reverse_dijkstra(self, weight):
        """Backwards from the destination; returns dist and next-hop maps."""
        dist = {self.end: 0}
        nxt = {}
        pq = [(0, 0, self.end)]
        c = 0
        while pq:
            d, _, u = heapq.heappop(pq)
            if d > dist.get(u, d):
                continue
            for eid, v, w in self.adj.get(u, ()):
                nd = d + weight(eid, v, u, w)            # travelling v -> u
                if v not in dist or nd < dist[v]:
                    dist[v] = nd
                    nxt[v] = (eid, u)
                    c += 1
                    heapq.heappush(pq, (nd, c, v))
        return dist, nxt

    def _static_path(self, node, nxt):
        path = []
        while node != self.end:
            eid, node = nxt[node]
            path.append(eid)
        return path

    def _earliest_arrival(self):
        """One label per node, no cycling: a feasible route and upper bound."""
        dist = {self.start: 0}
        parent = {}
        pq = [(0, 0, self.start)]
        c = 0
        while pq:
            t, _, u = heapq.heappop(pq)
            if t > dist.get(u, t):
                continue
            if u == self.end:
                path = []
                while u != self.start:
                    eid, u = parent[u]
                    path.append(eid)
                path.reverse()
                return t, path
            for eid, v, base in self.adj.get(u, ()):
                arr = self.traverse(eid, u, v, base, t)
                if arr is None:
                    continue
                if v not in dist or arr < dist[v]:
                    dist[v] = arr
                    parent[v] = (eid, u)
                    c += 1
                    heapq.heappush(pq, (arr, c, v))
        return None, None


    def run(self, max_states=None, label_budget=None):
        """Fastest arrival and the edge list that achieves it.

        label_budget optionally caps how many times one node may be settled
        during an initial pass while hard blocks are live.  It defaults to None.
        Such a cap is not safe for a final answer: a route may need arbitrarily
        many cycles to burn time past a long block, so every truncated pass must
        be treated only as a speed optimization.

        Limits apply only to an initial, potentially cheaper pass.  If either
        limit truncates that pass, the search is rerun uncapped.  A feasible
        upper bound must never be returned as though its optimality were proved.
        """
        if self.start not in self.adj or self.end not in self.adj:
            return None, []
        if self.start == self.end:
            return 0, []

        sdist, snxt = self._reverse_dijkstra(
            lambda eid, frm, to, base: base
        )
        if self.start not in sdist:
            return None, []                              # not connected at all
        if self.horizon <= 0:                            # already fully static
            return sdist[self.start], self._static_path(self.start, snxt)

        # h(v): every edge priced at its fastest ever traversal -> lower bound
        def fastest_w(eid, frm, to, base):
            f = self.fastest.get((eid, frm, to))
            return base / f if f else base

        # When all effective factors are <= 1, static base distance is already
        # the strongest possible version of this heuristic.
        if all(f <= 1 for f in self.fastest.values()):
            h = sdist
        else:
            h, _ = self._reverse_dijkstra(fastest_w)

        best_t, best_path = self._earliest_arrival()     # feasible upper bound

        bh = self.block_horizon
        visited = set()          # (node, time) labels, below the block horizon
        settled = set()          # nodes, at or above it - one label is enough
        n_labels = {}            # node -> labels spent below the block horizon
        budget = float("inf") if label_budget is None else label_budget
        truncated = False
        parent = {}
        counter = 0
        pq = [(h[self.start], 0, 0, self.start, None, None)]
        states = 0

        def dominated(w, arr):
            nonlocal truncated
            if arr >= bh:
                return w in settled
            if (w, arr) in visited:
                return True
            if n_labels.get(w, 0) >= budget:
                truncated = True
                return True
            return False

        while pq:
            f, t, _, u, prev, eid = heapq.heappop(pq)
            if best_t is not None and f >= best_t:
                break                                    # nothing can improve
            state = (u, t)
            if dominated(u, t):
                continue
            # A* pops a given node in non-decreasing t (f = t + h(u)), so a node
            # settled above the horizon can never be reopened below it.
            if t >= bh:
                settled.add(u)
            else:
                visited.add(state)
                n_labels[u] = n_labels.get(u, 0) + 1
            parent[state] = (prev, eid)

            if u == self.end:                            # h is consistent
                return t, self._path(state, parent)

            if t >= self.horizon:                        # static from here on
                if u in sdist:
                    cand = t + sdist[u]
                    if best_t is None or cand < best_t:
                        best_t = cand
                        best_path = (self._path(state, parent)
                                     + self._static_path(u, snxt))
                continue

            states += 1
            if max_states is not None and states > max_states:
                truncated = True
                break

            def offer(w, arr, label):
                nonlocal counter
                if w not in h or dominated(w, arr):
                    return
                nf = arr + h[w]
                if best_t is not None and nf >= best_t:
                    return
                counter += 1
                heapq.heappush(pq, (nf, arr, counter, w, state, label))

            for e_id, v, base in self.adj.get(u, ()):
                kind, when, covered = self.probe(e_id, u, v, base, t)
                if kind == "arrive":
                    offer(v, when, (e_id,))
                elif kind == "block":                    # forced reversal
                    back, at, _ = self.probe(e_id, v, u, covered, when)
                    if back == "arrive":                 # reverse side passable
                        offer(u, at, (e_id, e_id))

        if truncated:
            # The previous best is only an upper bound, not a proved optimum.
            return self.run(max_states=None, label_budget=None)
        if best_t is None:
            return None, []
        return best_t, best_path

    @staticmethod
    def _path(state, parent):
        chunks = []
        while True:
            prev, label = parent[state]
            if prev is None:
                break
            chunks.append(label)                         # tuple of 1 or 2 ids
            state = prev
        chunks.reverse()
        return [eid for label in chunks for eid in label]



def solve_case(payload: dict) -> dict:
    solver = Solver(payload)
    total, path = solver.run()
    if total is None:
        return {"total_duration_sec": None, "arrival_time": None, "path": []}
    return {
        "total_duration_sec": _num(total),
        "arrival_time": _fmt_time(solver.t0, total, solver.tz_style),
        "path": path,
    }


def solve(data: str) -> str:
    payload = json.loads(data, parse_float=Decimal)
    return json.dumps(solve_case(payload))
