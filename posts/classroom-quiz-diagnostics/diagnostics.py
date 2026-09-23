"""Per-objective evidence from live-session answers, with honest denominators."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

Z = 1.96  # 95% intervals
MASTERY = 0.6  # the bar a cell is judged against


@dataclass(frozen=True)
class Item:
    """A question as diagnostics needs it (from Part 1's questions table)."""

    id: int
    objective_id: str
    chunk_id: str
    key: int
    tags: tuple[str | None, ...]  # misconception per option; None for the key


@dataclass(frozen=True)
class Response:
    """One stored answer (from Part 2's responses table)."""

    session_id: str
    player_id: str
    question_id: int
    choice: int
    timing: str  # on_time | late_before_reveal | late_after_reveal


@dataclass(frozen=True)
class Cell:
    """One objective in one session."""

    session_id: str
    objective_id: str
    students: int  # students with at least one usable answer
    mastery: float | None  # mean over students of their share correct
    low: float | None
    high: float | None
    answer_rate: float  # usable answers / answers we could have had
    late_after_reveal: int

    @property
    def verdict(self) -> str:
        """Judge against the bar only as far as the interval allows."""
        if self.high is None:
            return "no data"
        if self.high < MASTERY:
            return "gap"  # weak even on the most generous reading
        if self.low >= MASTERY:
            return "secure"
        return "unclear"


def wilson(successes: float, n: int, z: float = Z) -> tuple[float, float] | None:
    """Wilson score interval; stays inside [0, 1] and behaves at small n."""
    if n == 0:
        return None
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def usable(responses: Iterable[Response]) -> list[Response]:
    """Answers given before the key was shown. Later ones are contaminated."""
    return [r for r in responses if r.timing != "late_after_reveal"]


def objective_cells(
    responses: list[Response],
    items: dict[int, Item],
    present: dict[tuple[str, int], set[str]],
) -> list[Cell]:
    """Build every (session, objective) cell.

    `present[(session, question)]` is who was in the room when that question
    opened. With anyone who answered it anyway (a late joiner), those are the
    answers we could have had: the denominator that makes missing ones visible.
    """
    seats = {(s, q, p) for (s, q), players in present.items() for p in players}
    seats |= {(r.session_id, r.question_id, r.player_id) for r in responses}
    expected = Counter((s, items[q].objective_id) for s, q, _ in seats)

    got: Counter = Counter()
    late: Counter = Counter()
    per_student: dict = defaultdict(list)
    for r in responses:
        key = (r.session_id, items[r.question_id].objective_id)
        if r.timing == "late_after_reveal":
            late[key] += 1
            continue
        got[key] += 1
        per_student[key + (r.player_id,)].append(r.choice == items[r.question_id].key)

    scores: dict = defaultdict(list)  # each student counts once per cell
    for (session, objective, _player), marks in per_student.items():
        scores[session, objective].append(sum(marks) / len(marks))

    cells = []
    for key in sorted(expected):
        student_scores = scores.get(key, [])
        n = len(student_scores)
        mean = sum(student_scores) / n if n else None
        bounds = wilson(sum(student_scores), n)
        cells.append(
            Cell(
                session_id=key[0],
                objective_id=key[1],
                students=n,
                mastery=mean,
                low=bounds[0] if bounds else None,
                high=bounds[1] if bounds else None,
                answer_rate=got[key] / expected[key],
                late_after_reveal=late[key],
            )
        )
    return cells


def student_coverage(
    responses: list[Response], asked: dict[str, int]
) -> list[tuple[str, float]]:
    """Share of each student's possible answers that arrived usable, lowest first.

    `asked[player]` is how many questions were asked while that student was in
    the room. A low share means that student's part of the picture is a guess.
    """
    got = Counter(r.player_id for r in usable(responses))
    shares = [(player, got[player] / n) for player, n in asked.items() if n]
    return sorted(shares, key=lambda pair: pair[1])


def misconceptions(
    responses: list[Response], items: dict[int, Item], objective_id: str
) -> list[tuple[str, int]]:
    """Which wrong answers students chose on one objective, most common first."""
    tags = Counter(
        items[r.question_id].tags[r.choice]
        for r in usable(responses)
        if items[r.question_id].objective_id == objective_id
        and r.choice != items[r.question_id].key
    )
    return tags.most_common()


def remediation(
    responses: list[Response], items: dict[int, Item], objective_id: str
) -> list[tuple[str, int]]:
    """The source passages behind the missed questions, most missed first."""
    misses = Counter(
        items[r.question_id].chunk_id
        for r in usable(responses)
        if items[r.question_id].objective_id == objective_id
        and r.choice != items[r.question_id].key
    )
    return misses.most_common()


def next_quiz(cells: list[Cell], sessions: list[str], total: int) -> dict[str, int]:
    """Share `total` questions over objectives, doubling the weight of gaps.

    Uses each objective's cell from the latest of `sessions` (oldest first)
    that produced any answers for it. Only a confident gap (the upper bound under the bar)
    earns extra questions; an unclear cell does not.
    """
    when = {s: i for i, s in enumerate(sessions)}
    latest = {}
    for c in sorted(cells, key=lambda c: when[c.session_id]):
        if c.verdict != "no data" or c.objective_id not in latest:
            latest[c.objective_id] = c  # a session with no answers changes nothing
    if not latest:
        return {}
    weights = {o: 2 if c.verdict == "gap" else 1 for o, c in latest.items()}
    scale = total / sum(weights.values())
    shares = {o: w * scale for o, w in weights.items()}
    plan = {o: math.floor(s) for o, s in shares.items()}
    by_remainder = sorted(shares, key=lambda o: shares[o] - plan[o], reverse=True)
    for o in by_remainder[: total - sum(plan.values())]:
        plan[o] += 1
    return plan
