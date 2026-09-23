"""A synthetic class for illustrating the diagnostics. No real students.

Generating process, per seed:

- 40 students; 10 of them answer over weak connections.
- 6 objectives. A student's mastery of an objective is drawn from that
  objective's Beta distribution; O4 is the hard one (mean 0.4).
- Weak-connection students' mastery is shifted down by `weak_shift`. This
  is the assumption under test, not a finding: it stands for any reason the
  students whose phones drop out also know the material less well.
- 8 weekly sessions, 2 questions per covered objective. Mastery rises 0.05
  per session after an objective is introduced; each question gets a small
  random difficulty offset.
- A wrong answer picks a distractor by objective-specific misconception
  weights; on O4 one misconception dominates.
- A weak connection delivers an answer on time with probability 0.55,
  after time is up but before the reveal 0.15, after the reveal 0.05, and
  never 0.25. A good connection loses 3% and delivers the rest on time.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from diagnostics import Item, Response

N_STUDENTS, N_WEAK = 40, 10
OBJECTIVES = {  # Beta(a, b) for mastery, misconception tags, their weights
    "O1": ((8, 3), ("t1a", "t1b", "t1c"), (0.4, 0.3, 0.3)),
    "O2": ((7, 3), ("t2a", "t2b", "t2c"), (0.4, 0.3, 0.3)),
    "O3": ((6, 4), ("t3a", "t3b", "t3c"), (0.4, 0.3, 0.3)),
    "O4": ((4, 6), ("t4a", "t4b", "t4c"), (0.7, 0.2, 0.1)),
    "O5": ((6, 3), ("t5a", "t5b", "t5c"), (0.4, 0.3, 0.3)),
    "O6": ((5, 4), ("t6a", "t6b", "t6c"), (0.4, 0.3, 0.3)),
}
SCHEDULE = [  # objectives each weekly session covers
    ["O1"],
    ["O1", "O2"],
    ["O1", "O2", "O3"],
    ["O2", "O3", "O4"],
    ["O3", "O4", "O5"],
    ["O4", "O5", "O6"],
    ["O4", "O5", "O6"],
    ["O3", "O4", "O6"],
]
WEAK_DELIVERY = (("on_time", 0.55), ("late_before_reveal", 0.15),
                 ("late_after_reveal", 0.05), (None, 0.25))  # fmt: skip
GOOD_DELIVERY = (("on_time", 0.97), (None, 0.03))


@dataclass
class SyntheticClass:
    """Everything the diagnostics read, plus the truth they cannot see."""

    sessions: list[str]
    items: dict[int, Item]
    responses: list[Response]
    present: dict[tuple[str, int], set[str]]
    weak: set[str]
    lost: list[Response]  # answers that never arrived, as they would have been


def simulate(seed: int, weak_shift: float = 0.10) -> SyntheticClass:
    """Draw one synthetic semester."""
    rng = random.Random(seed)
    students = [f"s{i:02d}" for i in range(N_STUDENTS)]
    weak = set(rng.sample(students, N_WEAK))
    mastery = {
        (s, o): rng.betavariate(*spec[0]) - (weak_shift if s in weak else 0.0)
        for s in students
        for o, spec in OBJECTIVES.items()
    }
    introduced = {}
    for week, covered in enumerate(SCHEDULE):
        for o in covered:
            introduced.setdefault(o, week)

    sessions, items, responses, present, lost = [], {}, [], {}, []
    for week, covered in enumerate(SCHEDULE):
        session = f"w{week + 1}"
        sessions.append(session)
        for o in covered:
            _, tags, weights = OBJECTIVES[o]
            growth = 0.05 * (week - introduced[o])
            for k in range(2):
                qid = len(items) + 1
                key = rng.randrange(4)
                option_tags = list(tags)
                option_tags.insert(key, None)
                items[qid] = Item(
                    qid, o, f"notes-{o}:p{k + 1}", key, tuple(option_tags)
                )
                present[session, qid] = set(students)
                offset = rng.gauss(0, 0.07)
                for s in students:
                    p = min(0.97, max(0.03, mastery[s, o] + growth + offset))
                    delivery = WEAK_DELIVERY if s in weak else GOOD_DELIVERY
                    timing = _pick(rng, delivery)
                    if rng.random() < p:
                        choice = key
                    else:
                        wrong = _pick(rng, tuple(zip(tags, weights, strict=True)))
                        choice = option_tags.index(wrong)
                    if timing is None:  # never reached the server
                        lost.append(Response(session, s, qid, choice, "on_time"))
                    else:
                        responses.append(Response(session, s, qid, choice, timing))
    return SyntheticClass(sessions, items, responses, present, weak, lost)


def _pick(rng: random.Random, weighted):
    """Draw one value from ((value, probability), ...)."""
    values, probs = zip(*weighted, strict=True)
    return rng.choices(values, weights=probs)[0]
