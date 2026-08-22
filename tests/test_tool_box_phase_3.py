import pytest

from challenges.tool_box import working_life
from challenges.tool_box.working_life import (
    CalendarEvent,
    Venue,
    choose_meeting_window,
    find_open_venues,
    minimum_travel_point,
    optimize_outing,
    parse_calendar_events,
)


def test_open_venues_returns_every_name_and_uses_half_open_availability(monkeypatch):
    monkeypatch.setattr(
        working_life,
        "_fetch_venues",
        lambda day: (
            Venue("Breakfast", 0, 0, ((8 * 60, 9 * 60),)),
            Venue("Brunch", 0, 0, ((9 * 60, 10 * 60),)),
            Venue("All Morning", 0, 0, ((8 * 60, 10 * 60),)),
        ),
    )

    assert find_open_venues("Monday", "09:00") == "Brunch, All Morning"


def test_email_parser_uses_anchored_current_fields_and_ignores_stale_prose():
    payload = {
        "emails": [
            {
                "body": (
                    "The old proposal was Monday 08:00-09:00.\n"
                    "Response: ACCEPTED\n"
                    "When: Tuesday 10:00-11:00\n"
                )
            },
            {
                "body": "Response: DECLINED\nWhen: Friday 14:00-15:00\n"
            },
            {"body": "A prose-only email mentioning Wednesday 12:00-13:00."},
        ]
    }

    events = parse_calendar_events(payload)

    assert events == (
        CalendarEvent("Tuesday", 600, 660, "ACCEPTED"),
        CalendarEvent("Friday", 840, 900, "DECLINED"),
    )


def test_meeting_prefers_later_clean_slot_over_earlier_tentative_overlap():
    result = choose_meeting_window(
        8 * 60,
        12 * 60,
        60,
        friend_busy=[(9 * 60, 10 * 60)],
        calendar_events=[CalendarEvent("Monday", 8 * 60, 9 * 60, "TENTATIVE")],
    )

    assert result == (10 * 60, 11 * 60, False)


def test_meeting_uses_earliest_tentative_overlap_only_when_no_clean_slot_exists():
    result = choose_meeting_window(
        8 * 60,
        10 * 60,
        60,
        friend_busy=[],
        calendar_events=[CalendarEvent("Monday", 8 * 60, 10 * 60, "TENTATIVE")],
    )

    assert result == (8 * 60, 9 * 60, True)


def test_meeting_treats_accepted_as_hard_and_declined_as_free():
    result = choose_meeting_window(
        8 * 60,
        11 * 60,
        60,
        friend_busy=[],
        calendar_events=[
            CalendarEvent("Monday", 8 * 60, 9 * 60, "ACCEPTED"),
            CalendarEvent("Monday", 9 * 60, 10 * 60, "DECLINED"),
        ],
    )

    assert result == (9 * 60, 10 * 60, False)


def test_meeting_intervals_are_half_open():
    result = choose_meeting_window(
        8 * 60,
        11 * 60,
        60,
        friend_busy=[(8 * 60, 9 * 60)],
        calendar_events=[],
    )

    assert result == (9 * 60, 10 * 60, False)


def test_meeting_raises_when_every_slot_has_a_hard_conflict():
    with pytest.raises(ValueError, match="no meeting window"):
        choose_meeting_window(
            8 * 60,
            10 * 60,
            60,
            friend_busy=[(8 * 60, 10 * 60)],
            calendar_events=[],
        )


def test_minimum_travel_point_includes_all_people_and_uses_lower_median():
    point, total = minimum_travel_point([(0, 9), (4, 1), (8, 5), (9, 0)])

    assert point == (4, 1)
    assert total == 26


def test_outing_jointly_optimizes_point_and_venue_not_people_median_alone():
    venues = [Venue("Far Cafe", 9, 0, ((8 * 60, 23 * 60),))]

    point, venue, total = optimize_outing([(0, 0), (2, 0)], venues)

    # With the venue as an additional L1 objective, any x from 2 through 9 is
    # optimal. Our deterministic tie break chooses x=2, not the lower people-only
    # median x=0.
    assert point == (2, 0)
    assert venue.name == "Far Cafe"
    assert total == 9


def test_outing_selects_the_globally_cheapest_venue_and_point():
    venues = [
        Venue("North", 0, 9, ((8 * 60, 23 * 60),)),
        Venue("East", 8, 0, ((8 * 60, 23 * 60),)),
    ]

    point, venue, total = optimize_outing([(7, 0), (9, 0)], venues)

    assert point == (8, 0)
    assert venue.name == "East"
    assert total == 2


def test_outing_requires_full_hour_of_venue_availability_after_meeting(monkeypatch):
    monkeypatch.setattr(
        working_life,
        "_load_outing_inputs",
        lambda friends, day: (
            [()],
            (),
            [(0, 0)],
            (
                Venue("Closes at Meeting End", 0, 0, ((8 * 60, 9 * 60),)),
                Venue("Open", 2, 0, ((9 * 60, 10 * 60),)),
            ),
        ),
    )

    result = working_life.plan_group_outing(
        "Monday",
        0,
        0,
        ["Ada"],
        "08:00",
        "10:00",
    )

    assert result == "08:00-09:00; meeting point [0,0]; venue Open"
