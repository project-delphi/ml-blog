"""Tests for the diagnostics, and for every synthetic number the post quotes."""

import pytest
from diagnostics import (
    Cell,
    Item,
    Response,
    misconceptions,
    next_quiz,
    objective_cells,
    remediation,
    student_coverage,
    wilson,
)
from make_figures import quoted_numbers
from synthetic import simulate

ITEMS = {
    1: Item(1, "O1", "notes:p1:aa", 0, (None, "m1", "m2", "m3")),
    2: Item(2, "O1", "notes:p2:bb", 2, ("m1", "m2", None, "m3")),
    3: Item(3, "O2", "notes:p3:cc", 1, ("m4", None, "m5", "m6")),
}


def test_wilson_is_none_without_data_and_stays_in_bounds():
    assert wilson(0, 0) is None
    low, high = wilson(0, 10)
    assert low == 0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson(10, 10)
    assert high == 1 and low == pytest.approx(0.7225, abs=1e-4)


def test_a_class_of_forty_cannot_place_a_cell_near_the_bar():
    low, high = wilson(0.6 * 40, 40)
    assert (round(low, 2), round(high, 2)) == (0.45, 0.74)
    low, high = wilson(0.4 * 40, 40)
    assert high > 0.55  # even 0.40 does not clear a gap verdict by much


def test_students_count_once_and_late_answers_are_excluded():
    responses = [
        Response("w1", "ana", 1, 0, "on_time"),  # right
        Response("w1", "ana", 2, 2, "on_time"),  # right
        Response("w1", "ben", 1, 1, "on_time"),  # wrong
        Response("w1", "cai", 2, 2, "late_after_reveal"),  # saw the key first
        # dan's answers never arrived
    ]
    everyone = {"ana", "ben", "cai", "dan"}
    present = {("w1", 1): everyone, ("w1", 2): everyone}
    [cell] = objective_cells(responses, ITEMS, present)
    assert cell.students == 2  # ana and ben; cai's only answer is unusable
    assert cell.mastery == pytest.approx((1.0 + 0.0) / 2)
    assert cell.answer_rate == pytest.approx(3 / 8)
    assert cell.late_after_reveal == 1


def test_a_late_joiner_who_answers_counts_in_the_denominator_too():
    responses = [
        Response("w1", "ana", 1, 0, "on_time"),
        Response("w1", "eve", 1, 0, "on_time"),  # joined after question 1 opened
    ]
    [cell] = objective_cells(responses, ITEMS, {("w1", 1): {"ana", "ben"}})
    assert cell.answer_rate == pytest.approx(2 / 3)  # never above 1


def test_a_cell_nobody_answered_is_no_data_not_zero():
    [cell] = objective_cells([], ITEMS, {("w1", 3): {f"s{i}" for i in range(30)}})
    assert (cell.students, cell.mastery, cell.verdict) == (0, None, "no data")
    assert cell.answer_rate == 0


def test_coverage_lists_the_students_whose_evidence_is_thinnest():
    responses = [
        Response("w1", "ana", 1, 0, "on_time"),
        Response("w1", "ana", 2, 2, "on_time"),
        Response("w1", "ben", 1, 0, "late_before_reveal"),
        Response("w1", "ben", 2, 1, "late_after_reveal"),
    ]
    assert student_coverage(responses, {"ana": 2, "ben": 2, "cai": 2}) == [
        ("cai", 0.0),
        ("ben", 0.5),
        ("ana", 1.0),
    ]


def cell(low, high, session="w1", objective="O1"):
    return Cell(session, objective, 30, (low + high) / 2, low, high, 1.0, 0)


def test_verdicts_follow_the_interval_not_the_point_estimate():
    assert cell(0.30, 0.55).verdict == "gap"
    assert cell(0.45, 0.75).verdict == "unclear"
    assert cell(0.62, 0.90).verdict == "secure"


def test_misconceptions_and_remediation_rank_what_went_wrong():
    responses = [
        Response("w1", "a", 1, 1, "on_time"),  # m1 on notes:p1
        Response("w1", "b", 1, 1, "on_time"),  # m1 on notes:p1
        Response("w1", "c", 2, 0, "on_time"),  # m1 on notes:p2
        Response("w1", "d", 2, 3, "late_before_reveal"),  # m3, still usable
        Response("w1", "e", 2, 1, "late_after_reveal"),  # excluded
        Response("w1", "f", 1, 0, "on_time"),  # right
    ]
    assert misconceptions(responses, ITEMS, "O1") == [("m1", 3), ("m3", 1)]
    assert remediation(responses, ITEMS, "O1") == [
        ("notes:p1:aa", 2),
        ("notes:p2:bb", 2),
    ]


def test_only_a_confident_gap_earns_extra_questions():
    cells = [
        cell(0.30, 0.55, "w1", "O1"),  # a gap in week 1 ...
        cell(0.50, 0.80, "w2", "O1"),  # ... but unclear by week 2
        cell(0.20, 0.50, "w2", "O2"),
        cell(0.45, 0.75, "w2", "O3"),
    ]
    plan = next_quiz(cells, ["w1", "w2"], total=8)
    assert plan == {"O1": 2, "O2": 4, "O3": 2}


def test_a_session_with_no_answers_does_not_erase_a_gap():
    silent = Cell("w2", "O1", 0, None, None, None, 0.0, 12)  # all after the reveal
    cells = [cell(0.30, 0.55, "w1", "O1"), silent, cell(0.45, 0.75, "w2", "O3")]
    assert next_quiz(cells, ["w1", "w2"], total=6) == {"O1": 4, "O3": 2}
    assert next_quiz([], ["w1"], total=6) == {}


def test_simulation_is_reproducible_and_loses_answers_where_it_says():
    a, b = simulate(3), simulate(3)
    assert a.responses == b.responses
    delivered = {(r.player_id, r.question_id) for r in a.responses}
    lost = {(r.player_id, r.question_id) for r in a.lost}
    assert not delivered & lost
    assert len(delivered | lost) == 40 * len(a.items)
    weak_lost = sum(r.player_id in a.weak for r in a.lost)
    assert weak_lost > len(a.lost) / 2  # 10 weak students lose most answers


def test_the_numbers_the_post_quotes():
    q = quoted_numbers()
    assert -0.02 < q["mean shift"][0] and q["mean shift"][1] < 0.02
    assert max(abs(v) for v in q["worst cell shift"]) < 0.09
    assert 0.25 < q["interval width"][0] and q["interval width"][1] < 0.30
    assert 0.88 < q["answer rate"][0] and q["answer rate"][1] < 0.92
    assert 0.64 < q["usable share, weak"][0] and q["usable share, weak"][1] < 0.76
    assert q["usable share, good"][0] > 0.95
    assert 0.45 < q["unclear share"][0] and q["unclear share"][1] < 0.9
