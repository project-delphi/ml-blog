"""The game as a pure state machine: the server owns the clock, the phase, the key."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

Phase = Literal["lobby", "question", "locked", "reveal", "leaderboard", "final"]
Timing = Literal["on_time", "late_before_reveal", "late_after_reveal"]

# Host commands and the phases they are allowed from.
TRANSITIONS: dict[str, tuple[tuple[Phase, ...], Phase]] = {
    "next": (("lobby", "leaderboard"), "question"),
    "lock": (("question",), "locked"),
    "reveal": (("locked",), "reveal"),
    "leaderboard": (("reveal",), "leaderboard"),
    "finish": (("leaderboard",), "final"),
}

CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I to misread


class NotApproved(ValueError):
    """A question that has not passed triage was offered to a room."""


class Rejected(ValueError):
    """A command or answer that the current state does not allow."""


@dataclass(frozen=True)
class Question:
    """An approved question as the game needs it."""

    id: int
    stem: str
    options: tuple[str, ...]
    answer: int
    explanation: str
    evidence: str
    source: str
    seconds: float = 20.0


def load_questions(rows: Iterable[dict]) -> list[Question]:
    """Build the question list, refusing anything the professor has not approved."""
    questions = []
    for row in rows:
        if row["status"] != "approved":
            raise NotApproved(f"question {row['id']} is {row['status']}, not approved")
        options = row["options"]
        questions.append(
            Question(
                id=row["id"],
                stem=row["stem"],
                options=tuple(o["text"] for o in options),
                answer=next(i for i, o in enumerate(options) if o["correct"]),
                explanation=row["explanation"],
                evidence=row["evidence"],
                source=row["source"],
                seconds=row.get("seconds", 20.0),
            )
        )
    return questions


@dataclass
class Answer:
    """One answer as the server received it."""

    answer_id: str
    player: str
    question_id: int
    choice: int
    timing: Timing
    points: int
    received_at: datetime


@dataclass
class Room:
    """One live game. Every mutation bumps `seq`; screens render the latest."""

    questions: list[Question]
    code: str = field(default_factory=lambda: new_code())
    epoch: str = field(default_factory=lambda: secrets.token_hex(4))
    clock: Callable[[], float] = time.monotonic
    wall: Callable[[], datetime] = lambda: datetime.now(UTC)
    phase: Phase = "lobby"
    index: int = -1
    seq: int = 0
    names: dict[str, str] = field(default_factory=dict)
    scores: dict[str, int] = field(default_factory=dict)
    answers: dict[str, Answer] = field(default_factory=dict)
    opened_at: float = 0.0

    # ------------------------------------------------------------ host side

    def join(self, name: str, player: str | None = None) -> str:
        """Add a player under the phone's own id (or a fresh one); return it."""
        if player is None or not 8 <= len(player) <= 64:
            player = secrets.token_urlsafe(8)
        self.names[player] = name
        self.scores[player] = 0
        self._changed()
        return player

    def command(self, name: str) -> None:
        """Apply a host command, if the current phase allows it."""
        allowed_from, target = TRANSITIONS[name]
        if self.phase not in allowed_from:
            raise Rejected(f"cannot {name} during {self.phase}")
        if name == "next" and self.index + 1 >= len(self.questions):
            raise Rejected("no questions left; use finish")
        if name == "next":
            self.index += 1
            self.opened_at = self.clock()
        self.phase = target
        self._changed()

    def tick(self) -> bool:
        """Lock the question once its time is up. Returns True if it locked."""
        if self.phase == "question" and self.remaining() <= 0:
            self.command("lock")
            return True
        return False

    def remaining(self) -> float:
        """Seconds left on the current question, never negative."""
        if self.phase != "question":
            return 0.0
        elapsed = self.clock() - self.opened_at
        return max(0.0, self.current.seconds - elapsed)

    @property
    def current(self) -> Question:
        """The question being asked or last asked."""
        return self.questions[self.index]

    # ---------------------------------------------------------- player side

    def submit(self, player: str, answer_id: str, question_id: int, choice: int):
        """Record an answer exactly once. Returns (answer, is_new)."""
        if not isinstance(question_id, int) or not isinstance(choice, int):
            raise Rejected("question_id and choice must be integers")
        claim = (player, question_id, choice)
        if answer_id in self.answers:
            prior = self.answers[answer_id]
            if (prior.player, prior.question_id, prior.choice) != claim:
                raise Rejected("that answer_id was already used for another answer")
            return prior, False  # a retry: same acknowledgement, never rescored
        if player not in self.names:
            raise Rejected("unknown player")
        asked = [q.id for q in self.questions[: self.index + 1]]
        if question_id not in asked:
            raise Rejected("that question has not been asked")
        if any(
            a.player == player and a.question_id == question_id
            for a in self.answers.values()
        ):
            raise Rejected("already answered that question")
        question = next(q for q in self.questions if q.id == question_id)
        if not 0 <= choice < len(question.options):
            raise Rejected("no such option")

        timing = self._timing(question_id)
        points = 0
        if timing == "on_time" and choice == question.answer:
            # Half the points for being right, half for being quick.
            points = 500 + round(500 * self.remaining() / question.seconds)
        answer = Answer(
            answer_id, player, question_id, choice, timing, points, self.wall()
        )
        self.answers[answer_id] = answer
        self.scores[player] += points
        return answer, True

    def _timing(self, question_id: int) -> Timing:
        """When an answer arrived, relative to the clock and to the reveal."""
        key_shown = self.phase in ("reveal", "leaderboard", "final")
        if question_id != self.current.id or key_shown:
            return "late_after_reveal"
        if self.phase == "question" and self.remaining() > 0:
            return "on_time"
        return "late_before_reveal"  # time is up, but nobody has seen the key

    # --------------------------------------------------------------- views

    def snapshot(self) -> dict:
        """The public state every screen receives. No key before the reveal."""
        view: dict = {
            "room": self.code,
            "epoch": self.epoch,
            "seq": self.seq,
            "phase": self.phase,
            "players": len(self.names),
        }
        if self.phase in ("question", "locked"):
            q = self.current
            view["question"] = {
                "id": q.id,
                "number": self.index + 1,
                "of": len(self.questions),
                "stem": q.stem,
                "options": list(q.options),
                "remaining_ms": round(self.remaining() * 1000),
            }
        if self.phase == "reveal":
            q = self.current
            counts = [0] * len(q.options)
            for a in self.answers.values():
                if a.question_id == q.id and a.timing != "late_after_reveal":
                    counts[a.choice] += 1
            view["question"] = {
                "id": q.id,
                "stem": q.stem,
                "options": list(q.options),
                "answer": q.answer,
                "counts": counts,
                "explanation": q.explanation,
                "evidence": q.evidence,
                "source": q.source,
            }
        if self.phase in ("leaderboard", "final"):
            top = sorted(self.scores.items(), key=lambda kv: -kv[1])[:5]
            view["leaderboard"] = [{"name": self.names[p], "score": s} for p, s in top]
        return view

    def private_view(self, player: str) -> dict:
        """What only this player's phone receives: their own answer and score."""
        view: dict = {"player": player, "score": self.scores.get(player, 0)}
        if self.index >= 0:
            mine = next(
                (
                    a
                    for a in self.answers.values()
                    if a.player == player and a.question_id == self.current.id
                ),
                None,
            )
            view["answered"] = mine is not None
            if mine and self.phase in ("reveal", "leaderboard", "final"):
                view["correct"] = mine.choice == self.current.answer
                view["points"] = mine.points
        return view

    def _changed(self) -> None:
        self.seq += 1


def new_code(length: int = 4) -> str:
    """A short room code that is easy to read off a projector."""
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(length))
