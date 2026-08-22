"""Deterministic tools for Tool Box Phase 3 working-life problems."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal
from urllib.parse import quote

import httpx


API_BASE_URL = "https://tool-box-2591eaa24fa3.herokuapp.com"
Day = Literal[
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
DAYS: tuple[str, ...] = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

_TIME_RE = re.compile(r"^(?:0[8-9]|1[0-9]|2[0-3]):00$")
_RESPONSE_RE = re.compile(
    r"^Response:\s*(ACCEPTED|DECLINED|TENTATIVE)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_WHEN_RE = re.compile(
    r"^When:\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
    r"((?:0[8-9]|1[0-9]|2[0-3]):00)-((?:0[8-9]|1[0-9]|2[0-3]):00)\s*$",
    re.MULTILINE | re.IGNORECASE,
)

_transport = httpx.HTTPTransport(retries=1)
_client = httpx.Client(
    base_url=API_BASE_URL,
    headers={"Accept": "application/json"},
    timeout=httpx.Timeout(6.0, connect=3.0),
    follow_redirects=True,
    transport=_transport,
)


@dataclass(frozen=True)
class CalendarEvent:
    day: str
    start: int
    end: int
    response: str


@dataclass(frozen=True)
class Venue:
    name: str
    x: int
    y: int
    available: tuple[tuple[int, int], ...]


def find_open_venues(day: Day, time: str) -> str:
    """Return all venue names open at a specified day and hour."""

    normalized_day = _validate_day(day)
    at = _parse_time(time)
    names = [
        venue.name
        for venue in _fetch_venues(normalized_day)
        if any(start <= at < end for start, end in venue.available)
    ]
    return ", ".join(names) if names else "No venues"


def find_group_meeting_time(
    day: Day,
    friends: list[str],
    window_start: str,
    window_end: str,
    duration_minutes: int = 60,
) -> str:
    """Return the exact best meeting window for the user and all named friends."""

    normalized_day = _validate_day(day)
    attendees = _validate_friends(friends)
    range_start, range_end, duration = _validate_window(
        window_start,
        window_end,
        duration_minutes,
    )
    schedules, events = _load_scheduling_inputs(attendees, normalized_day)
    start, end, _ = choose_meeting_window(
        range_start,
        range_end,
        duration,
        friend_busy=[interval for schedule in schedules for interval in schedule],
        calendar_events=events,
    )
    return f"{_format_time(start)}-{_format_time(end)}"


def find_minimum_travel_meeting_point(
    day: Day,
    your_x: int,
    your_y: int,
    friends: list[str],
) -> str:
    """Return a minimum-total-distance meeting point for everyone."""

    normalized_day = _validate_day(day)
    origin = _validate_coordinate(your_x, your_y)
    attendees = _validate_friends(friends)
    locations = _fetch_locations(attendees, normalized_day)
    point, _ = minimum_travel_point([origin, *locations])
    return f"[{point[0]},{point[1]}]"


def plan_group_outing(
    day: Day,
    your_x: int,
    your_y: int,
    friends: list[str],
    window_start: str,
    window_end: str,
    duration_minutes: int = 60,
) -> str:
    """Plan the required meeting window, travel-optimal point, and post-meeting venue."""

    normalized_day = _validate_day(day)
    origin = _validate_coordinate(your_x, your_y)
    attendees = _validate_friends(friends)
    range_start, range_end, duration = _validate_window(
        window_start,
        window_end,
        duration_minutes,
    )

    schedules, events, locations, venues = _load_outing_inputs(
        attendees,
        normalized_day,
    )
    meeting_start, meeting_end, _ = choose_meeting_window(
        range_start,
        range_end,
        duration,
        friend_busy=[interval for schedule in schedules for interval in schedule],
        calendar_events=events,
    )
    eligible_venues = [
        venue
        for venue in venues
        if any(
            available_start <= meeting_end and meeting_end + 60 <= available_end
            for available_start, available_end in venue.available
        )
    ]
    if not eligible_venues:
        raise ValueError("no venue is open for the full hour after the meeting")

    point, venue, _ = optimize_outing([origin, *locations], eligible_venues)
    return (
        f"{_format_time(meeting_start)}-{_format_time(meeting_end)}; "
        f"meeting point [{point[0]},{point[1]}]; venue {venue.name}"
    )


def parse_calendar_events(payload: Any) -> tuple[CalendarEvent, ...]:
    """Parse only anchored Response and When fields from the email inbox."""

    if not isinstance(payload, dict) or not isinstance(payload.get("emails"), list):
        raise ValueError("emails response must contain an emails list")

    events: list[CalendarEvent] = []
    for email in payload["emails"]:
        if not isinstance(email, dict) or not isinstance(email.get("body"), str):
            continue
        body = email["body"]
        response_match = _RESPONSE_RE.search(body)
        when_match = _WHEN_RE.search(body)
        if response_match is None or when_match is None:
            continue
        start = _parse_time(when_match.group(2))
        end = _parse_time(when_match.group(3))
        if end <= start:
            continue
        events.append(
            CalendarEvent(
                day=when_match.group(1).title(),
                start=start,
                end=end,
                response=response_match.group(1).upper(),
            )
        )
    return tuple(events)


def choose_meeting_window(
    window_start: int,
    window_end: int,
    duration_minutes: int,
    *,
    friend_busy: list[tuple[int, int]],
    calendar_events: tuple[CalendarEvent, ...] | list[CalendarEvent],
) -> tuple[int, int, bool]:
    """Choose earliest clean slot, using tentative time only as a fallback."""

    hard_busy = list(friend_busy) + [
        (event.start, event.end)
        for event in calendar_events
        if event.response == "ACCEPTED"
    ]
    tentative_busy = [
        (event.start, event.end)
        for event in calendar_events
        if event.response == "TENTATIVE"
    ]
    hard_free: list[tuple[int, int]] = []
    clean: list[tuple[int, int]] = []
    for start in range(window_start, window_end - duration_minutes + 1, 60):
        end = start + duration_minutes
        if any(_overlaps(start, end, busy_start, busy_end) for busy_start, busy_end in hard_busy):
            continue
        hard_free.append((start, end))
        if not any(
            _overlaps(start, end, busy_start, busy_end)
            for busy_start, busy_end in tentative_busy
        ):
            clean.append((start, end))

    if clean:
        return (*clean[0], False)
    if hard_free:
        return (*hard_free[0], True)
    raise ValueError("no meeting window avoids all hard conflicts")


def minimum_travel_point(
    positions: list[tuple[int, int]],
) -> tuple[tuple[int, int], int]:
    """Choose the lower coordinate-wise median and its total travel cost."""

    if not positions:
        raise ValueError("at least one starting position is required")
    for x, y in positions:
        _validate_coordinate(x, y)
    xs = sorted(x for x, _ in positions)
    ys = sorted(y for _, y in positions)
    point = (xs[(len(xs) - 1) // 2], ys[(len(ys) - 1) // 2])
    return point, sum(_distance(position, point) for position in positions)


def optimize_outing(
    positions: list[tuple[int, int]],
    venues: list[Venue],
) -> tuple[tuple[int, int], Venue, int]:
    """Jointly optimize the meeting cell and one onward trip to a venue."""

    if not positions:
        raise ValueError("at least one starting position is required")
    if not venues:
        raise ValueError("at least one eligible venue is required")

    candidates: list[tuple[int, str, int, int, Venue]] = []
    for venue in venues:
        for x in range(10):
            for y in range(10):
                point = (x, y)
                total = sum(_distance(position, point) for position in positions)
                total += _distance(point, (venue.x, venue.y))
                candidates.append((total, venue.name.casefold(), x, y, venue))
    total, _, x, y, venue = min(candidates)
    return (x, y), venue, total


def _load_scheduling_inputs(
    friends: tuple[str, ...],
    day: str,
) -> tuple[list[tuple[tuple[int, int], ...]], tuple[CalendarEvent, ...]]:
    with ThreadPoolExecutor(max_workers=max(2, len(friends) + 1)) as executor:
        schedule_futures = [executor.submit(_fetch_schedule, friend, day) for friend in friends]
        email_future = executor.submit(_fetch_calendar_events)
        schedules = [future.result() for future in schedule_futures]
        events = tuple(event for event in email_future.result() if event.day == day)
    return schedules, events


def _load_outing_inputs(
    friends: tuple[str, ...],
    day: str,
) -> tuple[
    list[tuple[tuple[int, int], ...]],
    tuple[CalendarEvent, ...],
    list[tuple[int, int]],
    tuple[Venue, ...],
]:
    with ThreadPoolExecutor(max_workers=max(4, len(friends) * 2 + 2)) as executor:
        schedule_futures = [executor.submit(_fetch_schedule, friend, day) for friend in friends]
        location_futures = [executor.submit(_fetch_location, friend, day) for friend in friends]
        email_future = executor.submit(_fetch_calendar_events)
        venues_future = executor.submit(_fetch_venues, day)
        schedules = [future.result() for future in schedule_futures]
        locations = [future.result() for future in location_futures]
        events = tuple(event for event in email_future.result() if event.day == day)
        venues = venues_future.result()
    return schedules, events, locations, venues


def _fetch_locations(friends: tuple[str, ...], day: str) -> list[tuple[int, int]]:
    with ThreadPoolExecutor(max_workers=max(1, len(friends))) as executor:
        return list(executor.map(lambda friend: _fetch_location(friend, day), friends))


@lru_cache(maxsize=128)
def _fetch_schedule(person: str, day: str) -> tuple[tuple[int, int], ...]:
    payload = _get_json(f"/schedule/{quote(person, safe='')}/{quote(day, safe='')}")
    if not isinstance(payload, dict) or not isinstance(payload.get("busy"), list):
        raise ValueError("schedule response must contain a busy list")
    return tuple(_parse_interval(interval, "busy interval") for interval in payload["busy"])


@lru_cache(maxsize=128)
def _fetch_location(person: str, day: str) -> tuple[int, int]:
    payload = _get_json(f"/location/{quote(person, safe='')}/{quote(day, safe='')}")
    if not isinstance(payload, dict):
        raise ValueError("location response must be an object")
    return _validate_coordinate(payload.get("x"), payload.get("y"))


@lru_cache(maxsize=7)
def _fetch_venues(day: str) -> tuple[Venue, ...]:
    payload = _get_json(f"/venues/{quote(day, safe='')}")
    if not isinstance(payload, dict) or not isinstance(payload.get("venues"), list):
        raise ValueError("venues response must contain a venues list")

    venues: list[Venue] = []
    for item in payload["venues"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("venue entry is invalid")
        name = item["name"].strip()
        if not name or not isinstance(item.get("available"), list):
            raise ValueError("venue entry is invalid")
        x, y = _validate_coordinate(item.get("x"), item.get("y"))
        available = tuple(
            _parse_interval(interval, "venue availability")
            for interval in item["available"]
        )
        venues.append(Venue(name=name, x=x, y=y, available=available))
    return tuple(venues)


@lru_cache(maxsize=1)
def _fetch_calendar_events() -> tuple[CalendarEvent, ...]:
    return parse_calendar_events(_get_json("/emails"))


def _get_json(path: str) -> Any:
    response = _client.get(path)
    response.raise_for_status()
    return response.json()


def _validate_day(day: Any) -> str:
    if not isinstance(day, str):
        raise ValueError("day must be Monday through Sunday")
    normalized = day.strip().title()
    if normalized not in DAYS:
        raise ValueError("day must be Monday through Sunday")
    return normalized


def _validate_friends(friends: Any) -> tuple[str, ...]:
    if not isinstance(friends, list) or not friends:
        raise ValueError("friends must be a non-empty list of names")
    normalized: list[str] = []
    for friend in friends:
        if not isinstance(friend, str) or not friend.strip():
            raise ValueError("every friend name must be a non-empty string")
        name = friend.strip().lower()
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def _validate_window(start: str, end: str, duration: Any) -> tuple[int, int, int]:
    range_start = _parse_time(start)
    range_end = _parse_time(end)
    if range_end <= range_start:
        raise ValueError("window_end must be later than window_start")
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError("duration_minutes must be an integer")
    if duration <= 0 or duration % 60:
        raise ValueError("duration_minutes must be a positive whole number of hours")
    if duration > range_end - range_start:
        raise ValueError("duration_minutes does not fit inside the requested window")
    return range_start, range_end, duration


def _parse_interval(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain a start and end time")
    start = _parse_time(value[0])
    end = _parse_time(value[1])
    if end <= start:
        raise ValueError(f"{label} end must be later than its start")
    return start, end


def _parse_time(value: Any) -> int:
    if not isinstance(value, str) or _TIME_RE.fullmatch(value) is None:
        raise ValueError("times must use an hourly HH:MM value from 08:00 to 23:00")
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _format_time(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _validate_coordinate(x: Any, y: Any) -> tuple[int, int]:
    if (
        isinstance(x, bool)
        or isinstance(y, bool)
        or not isinstance(x, int)
        or not isinstance(y, int)
        or not 0 <= x <= 9
        or not 0 <= y <= 9
    ):
        raise ValueError("coordinates must be integers from 0 through 9")
    return x, y


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end


def _distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])
