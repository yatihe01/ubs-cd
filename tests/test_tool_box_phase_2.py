import pytest

from challenges.tool_box.routing import find_cheapest_path
from challenges.tool_box.study_retrieval import (
    retrieve_study_passages,
    select_passages,
)


def test_retrieval_handles_paraphrased_calibration_question():
    documents = (
        """# Meridian Trench Research Station
## Calibration Procedures
The Kesterline acoustic array was recalibrated on 14 March after a routine drift review.

## Catering
The galley changed its breakfast menu on 2 April.
""",
        "# Other notes\n## Sensors\nA backup camera was inspected in June.",
    )

    passages = select_passages(
        "When was the sensor grid last brought back into alignment?",
        documents,
    )

    assert "14 March" in "\n".join(passages)


def test_retrieval_prefers_resident_staffing_over_simultaneous_engineers():
    documents = (
        """# Meridian Trench Research Station
## Staffing Roster
Crew levels at the station are tracked closely. The station maintains a resident
population of forty-one scientists and technicians across three rotating shifts.
During changeover periods occupancy can briefly spike to forty-six individuals.
""",
        """# Hollowlight Engine Technical Handbook
## Leadership and Team Structure
The core engine group maintains thirty-two engineers working simultaneously.
Temporary contractor support can swell the group to just under fifty individuals.
""",
    )

    passages = select_passages(
        "Roughly how many personnel live aboard the facility simultaneously?",
        documents,
    )

    assert "forty-one" in passages[0]
    assert "Staffing Roster" in passages[0]


def test_semantic_context_resolves_air_scrubber_failure_distractors():
    documents = (
        """# Meridian Trench Research Station
## Incident Reports
An oxygen scrubber failure occurred on 2 November, prompting an emergency
ventilation drill. Investigators traced the failure to a corroded valve seat.
""",
        """# Ashgrove Metropolitan Transit Authority
## Driver Shift Protocols
Driving time is capped before a mandatory break is required.
""",
        """# Thornmere Growers Cooperative Yearbook
## Incident Reports
A door-seal failure on 4 April let warm air into the Old Kiln store overnight.
""",
    )

    passages = select_passages(
        "On what date did the air-scrubbing equipment break down?",
        documents,
        semantic_context=(
            "oxygen scrubber failure ventilation malfunction breakdown date"
        ),
    )

    assert "2 November" in passages[0]
    assert "Meridian Trench Research Station — Incident Reports" in passages[0]


@pytest.mark.parametrize("semantic_context", ["", "   ", None])
def test_retrieval_tool_requires_semantic_context(semantic_context):
    with pytest.raises(ValueError, match="semantic_context"):
        retrieve_study_passages("What happened?", semantic_context)


def test_retrieval_finds_school_trip_stop_and_stays_within_budget():
    filler = " ".join(f"unrelated{i}" for i in range(2000))
    paragraph_lead = " ".join("maintenance timetable" for _ in range(45))
    documents = (
        f"# Transit\n## Stops\n{paragraph_lead} Verity Observatory is served by STOP_05."
        f"\n\n## Appendix\n{filler}",
    )

    passages = select_passages("Travel to the Verity Observatory", documents)
    assert "STOP_05" in "\n".join(passages)
    assert sum(len(passage.encode("utf-8")) for passage in passages) <= 3_300


def test_route_includes_entry_tolls_and_excludes_start_toll():
    adjacency = {
        "A": {"B": 1, "C": 3},
        "B": {"D": 1},
        "C": {"D": 3},
        "D": {},
    }
    tolls = {"A": 999, "B": 100, "C": 0, "D": 0}

    assert find_cheapest_path(adjacency, tolls, "A", "D") == ["A", "C", "D"]


def test_route_respects_direction_and_hop_allowance():
    adjacency = {
        "A": {"B": 1, "D": 10},
        "B": {"C": 1},
        "C": {"D": 1},
        "D": {},
    }
    tolls = {node: 0 for node in adjacency}

    assert find_cheapest_path(adjacency, tolls, "A", "D") == ["A", "B", "C", "D"]
    assert find_cheapest_path(adjacency, tolls, "A", "D", 2) == ["A", "D"]
    with pytest.raises(ValueError, match="no route"):
        find_cheapest_path(adjacency, tolls, "D", "A")


def test_route_rejects_exhausted_hop_allowance():
    with pytest.raises(ValueError, match="at least one"):
        find_cheapest_path({"A": {"B": 1}, "B": {}}, {"A": 0, "B": 0}, "A", "B", 0)
