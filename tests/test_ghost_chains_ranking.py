"""Cross-phase invariants for occurrence-first Ghost Chains ranking."""

from challenges.ghost_chains.ranking import occurrence_rank_score


def test_one_more_occurrence_always_outranks_any_lower_tier_weight():
    strongest_one = occurrence_rank_score(1, 1e12, weighted_squash=2.0)
    weakest_two = occurrence_rank_score(2, 1e-12, weighted_squash=2.0)
    assert weakest_two > strongest_one


def test_weight_is_the_tie_breaker_inside_one_occurrence_tier():
    weak = occurrence_rank_score(3, 0.01, weighted_squash=2.0)
    strong = occurrence_rank_score(3, 100.0, weighted_squash=2.0)
    assert strong > weak


def test_no_risky_occurrence_stays_on_the_floor():
    assert occurrence_rank_score(0, 100.0, weighted_squash=2.0) == 0.0

