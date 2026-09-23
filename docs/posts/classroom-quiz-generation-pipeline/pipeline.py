"""Source-grounded quiz questions: chunks, the evidence contract, the review loop."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import IO, Literal, Protocol, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader

MODEL = "claude-opus-5"
MAX_ATTEMPTS = 3
MIN_QUOTE_WORDS = 8

# Multiple choice can test these four levels. "remember" is the trivia failure
# mode, and "create" needs an answer the student builds, not one they pick.
BloomLevel = Literal["understand", "apply", "analyze", "evaluate"]
AnyBloomLevel = Literal[
    "remember", "understand", "apply", "analyze", "evaluate", "create"
]

LEVEL_GUIDE = {
    "understand": "restate, classify or explain an idea from the passage in new words",
    "apply": "use a rule or procedure from the passage on a case it does not mention",
    "analyze": "take apart a mechanism or argument and say how the parts relate",
    "evaluate": "judge a claim or a choice against criteria the passage gives",
}

Language = Literal["es", "pt-BR", "en"]
LANGUAGES = {"es": "Spanish", "pt-BR": "Brazilian Portuguese", "en": "English"}


# ---------------------------------------------------------------- ingestion


@dataclass(frozen=True)
class Chunk:
    """A run of whole sentences from one page of one document."""

    id: str
    document_id: str
    page: int
    text: str


@dataclass(frozen=True)
class Objective:
    """A syllabus learning objective and its closed list of misconceptions."""

    id: str
    statement: str
    misconceptions: tuple[str, ...]


def extract_pages(pdf: str | IO[bytes]) -> list[str]:
    """Return the raw text of every page in a PDF, one string per page."""
    return [page.extract_text() or "" for page in PdfReader(pdf).pages]


def clean(text: str) -> str:
    """Undo PDF extraction artefacts: ligatures, soft hyphens, broken lines."""
    text = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split at line ends
    return re.sub(r"\s+", " ", text).strip()


def chunk_pages(
    document_id: str, pages: Sequence[str], max_words: int = 220
) -> list[Chunk]:
    """Pack whole sentences into chunks that never cross a page boundary."""
    chunks = []
    for page_no, raw in enumerate(pages, start=1):
        sentences = re.split(r"(?<=[.!?])\s+", clean(raw))
        buf: list[str] = []
        words = 0
        for sentence in filter(None, sentences):
            n = len(sentence.split())
            if buf and words + n > max_words:
                chunks.append(_make_chunk(document_id, page_no, buf))
                buf, words = [], 0
            buf.append(sentence)
            words += n
        if buf:
            chunks.append(_make_chunk(document_id, page_no, buf))
    return chunks


def _make_chunk(document_id: str, page: int, sentences: list[str]) -> Chunk:
    text = " ".join(sentences)
    digest = hashlib.sha256(text.encode()).hexdigest()[:10]
    return Chunk(f"{document_id}:p{page}:{digest}", document_id, page, text)


# ------------------------------------------------------- the evidence contract


class Option(BaseModel):
    """One answer choice; a distractor names the misconception it stands for."""

    text: str
    correct: bool
    misconception: str | None


class QuestionDraft(BaseModel):
    """What the generator must return. `evidence` is copied from the passage."""

    stem: str
    options: list[Option]
    bloom_level: BloomLevel
    evidence: str
    explanation: str


QUOTES = str.maketrans("\u2018\u2019\u201c\u201d", "''\"\"")
DASHES = re.compile(r"[-\u2010-\u2015]")


def match_key(text: str) -> str:
    """Normalise text so a faithful quote matches the passage it came from."""
    text = DASHES.sub("", clean(text).translate(QUOTES)).casefold()
    return re.sub(r"\s+", " ", text).strip(" \"'.,;:")


def contract_problems(
    draft: QuestionDraft, chunk: Chunk, objective: Objective, level: BloomLevel
) -> list[str]:
    """Deterministic checks that run before any reviewer sees the draft."""
    problems = []
    if len(draft.evidence.split()) < MIN_QUOTE_WORDS:
        problems.append(f"the evidence quote is under {MIN_QUOTE_WORDS} words")
    elif match_key(draft.evidence) not in match_key(chunk.text):
        problems.append("the evidence quote does not appear in the passage")
    if len(draft.options) != 4:
        problems.append(f"expected 4 options, got {len(draft.options)}")
    if sum(o.correct for o in draft.options) != 1:
        problems.append("expected exactly one correct option")
    if len({match_key(o.text) for o in draft.options}) != len(draft.options):
        problems.append("two options say the same thing")
    for i, option in enumerate(draft.options):
        if option.correct and option.misconception is not None:
            problems.append(f"option {i} is the key but carries a misconception tag")
        if not option.correct and option.misconception not in objective.misconceptions:
            problems.append(
                f"option {i} is tagged {option.misconception!r}, "
                f"which is not in {list(objective.misconceptions)}"
            )
    if draft.bloom_level != level:
        problems.append(f"asked for a {level} question, got {draft.bloom_level}")
    return problems


# ------------------------------------------------------------------ reviewers


class OptionJudgement(BaseModel):
    """What the passage says about one option."""

    option: int
    status: Literal["supported", "contradicted", "not_addressed"]
    note: str


class SourceCheck(BaseModel):
    """The source reviewer answers blind, then judges every option."""

    answer: int
    judgements: list[OptionJudgement]


class PedagogyCheck(BaseModel):
    """The pedagogy reviewer's reading of the question as a teaching item."""

    bloom_level: AnyBloomLevel
    one_defensible_answer: bool
    distractors_plausible: bool
    reasons: list[str]


def key_index(draft: QuestionDraft) -> int:
    """Index of the correct option."""
    return next(i for i, o in enumerate(draft.options) if o.correct)


def source_problems(check: SourceCheck, draft: QuestionDraft) -> list[str]:
    """Turn the source reviewer's observations into a verdict."""
    key = key_index(draft)
    judged = {j.option: j for j in check.judgements}
    problems = []
    if check.answer != key:
        problems.append(
            f"reading only the passage, the reviewer chose option {check.answer}, "
            f"but the key is option {key}"
        )
    for i in range(len(draft.options)):
        j = judged.get(i)
        if j is None:
            problems.append(f"the reviewer did not judge option {i}")
        elif i == key and j.status != "supported":
            problems.append(f"the passage does not support the key: {j.note}")
        elif i != key and j.status == "supported":
            problems.append(f"the passage also supports option {i}: {j.note}")
    return problems


def pedagogy_problems(check: PedagogyCheck, level: BloomLevel) -> list[str]:
    """Turn the pedagogy reviewer's observations into a verdict."""
    problems = []
    if check.bloom_level != level:
        problems.append(f"asked for {level}; it reads as {check.bloom_level}")
    if not check.one_defensible_answer:
        problems.append("more than one option can be defended")
    if not check.distractors_plausible:
        problems.append("some distractors are not plausible")
    return problems + check.reasons if problems else []


# ---------------------------------------------------------------- the models

T = TypeVar("T", bound=BaseModel)


class ModelOutputError(RuntimeError):
    """The model returned no usable structured output."""


class LLM(Protocol):
    """Anything that turns a prompt into a validated instance of `schema`."""

    def __call__(
        self, *, system: str, prompt: str, schema: type[T], effort: str
    ) -> T: ...


class ClaudeLLM:
    """The production LLM: Claude with structured outputs and fallbacks."""

    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL):
        """Create the wrapper, building a client from the environment if needed."""
        self.client = client or anthropic.Anthropic()
        self.model = model

    def __call__(self, *, system: str, prompt: str, schema: type[T], effort: str) -> T:
        """Ask once and validate the answer; raise ModelOutputError otherwise."""
        try:
            response = self._create(system, prompt, schema, effort)
        except anthropic.APIError as err:  # the SDK has already retried
            raise ModelOutputError(f"API error: {type(err).__name__}") from err
        if response.stop_reason in ("refusal", "max_tokens"):
            raise ModelOutputError(f"the model stopped with {response.stop_reason}")
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return schema.model_validate_json(text)
        except ValidationError as err:
            raise ModelOutputError(f"output did not match {schema.__name__}") from err

    def _create(self, system, prompt, schema, effort):
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": effort,
                "format": {
                    "type": "json_schema",
                    "schema": anthropic.transform_schema(schema),
                },
            },
            # If a safety classifier declines, the API re-runs the request on
            # Anthropic's recommended fallback model instead of refusing.
            fallbacks="default",
            betas=["server-side-fallback-2026-07-01"],
        )


# ------------------------------------------------------------------- prompts

GENERATOR = """You write one multiple-choice question for a university course.
It must test the given learning objective at the given Bloom level, and a student
who understood the passage must be able to answer it from the passage alone.
Write four options: one correct, three distractors. Tag every distractor with the
misconception it represents, chosen from the list given, and give the correct
option no tag. Copy `evidence` word for word from the passage, in the passage's
own language: one or more consecutive sentences that support the correct option,
with no ellipses. Write the stem, options and explanation in {language}."""

SOURCE_REVIEWER = """You check quiz questions against their source passage.
Use only the passage, not what you know about the subject. First choose the
option the passage supports. Then, for every option, say whether the passage
supports it, contradicts it, or does not address it, with a one-line note."""

PEDAGOGY_REVIEWER = """You review multiple-choice questions for a university
course. Say which Bloom level the question tests in practice (a question that
can be answered by recognising a phrase from the reading tests "remember",
whatever it claims), whether exactly one option can be defended, and whether a
student holding each tagged misconception would find that distractor tempting."""


def passage_block(chunk: Chunk) -> str:
    """The passage as every agent sees it, labelled with its address."""
    return f"<passage id={chunk.id!r} page={chunk.page}>\n{chunk.text}\n</passage>"


def options_block(draft: QuestionDraft, show_key: bool) -> str:
    """The question as a numbered list, with or without the answer key."""
    lines = [draft.stem]
    for i, option in enumerate(draft.options):
        mark = " (correct)" if show_key and option.correct else ""
        lines.append(f"{i}. {option.text}{mark}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the loop


@dataclass
class Attempt:
    """One pass through generation and review."""

    draft: QuestionDraft | None
    problems: list[str]


@dataclass
class Outcome:
    """Where a (chunk, objective, level) job ended up, with its full history."""

    status: Literal["verified", "rejected"]
    chunk: Chunk
    objective: Objective
    level: BloomLevel
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def draft(self) -> QuestionDraft | None:
        """The last draft produced, verified or not."""
        return self.attempts[-1].draft if self.attempts else None


def review_attempt(
    llm: LLM,
    chunk: Chunk,
    objective: Objective,
    level: BloomLevel,
    language: Language,
    previous: Attempt | None,
) -> Attempt:
    """Generate (or revise) one draft, then review it."""
    prompt = (
        f"Learning objective {objective.id}: {objective.statement}\n"
        f"Bloom level: {level} ({LEVEL_GUIDE[level]})\n"
        f"Misconception tags: {', '.join(objective.misconceptions)}\n\n"
        f"{passage_block(chunk)}"
    )
    if previous and previous.draft:
        prompt += (
            f"\n\nRevise this draft:\n{previous.draft.model_dump_json()}\n"
            "It failed review for these reasons:\n- " + "\n- ".join(previous.problems)
        )
    system = GENERATOR.format(language=LANGUAGES[language])
    try:
        draft = llm(system=system, prompt=prompt, schema=QuestionDraft, effort="high")
    except ModelOutputError as err:
        return Attempt(None, [f"generator: {err}"])
    return Attempt(draft, review_draft(llm, draft, chunk, objective, level))


def review_draft(
    llm: LLM,
    draft: QuestionDraft,
    chunk: Chunk,
    objective: Objective,
    level: BloomLevel,
) -> list[str]:
    """Run a draft past the contract and both reviewers, cheapest first.

    Also used on a question a professor edited: an edit throws away the old
    verification, so the edited text goes through the same three checks.
    """
    problems = contract_problems(draft, chunk, objective, level)
    if problems:
        return problems
    try:
        source = llm(
            system=SOURCE_REVIEWER,
            prompt=f"{passage_block(chunk)}\n\n{options_block(draft, show_key=False)}",
            schema=SourceCheck,
            effort="medium",
        )
        problems = source_problems(source, draft)
        if problems:
            return problems
        pedagogy = llm(
            system=PEDAGOGY_REVIEWER,
            prompt=(
                f"Learning objective: {objective.statement}\n"
                f"Intended Bloom level: {level}\n\n"
                f"{options_block(draft, show_key=True)}\n\n"
                "Distractor tags: "
                + "; ".join(
                    f"{i}={o.misconception}"
                    for i, o in enumerate(draft.options)
                    if not o.correct
                )
            ),
            schema=PedagogyCheck,
            effort="medium",
        )
    except ModelOutputError as err:
        return [f"reviewer: {err}"]
    return pedagogy_problems(pedagogy, level)


def generate_question(
    llm: LLM,
    chunk: Chunk,
    objective: Objective,
    level: BloomLevel,
    language: Language = "en",
    max_attempts: int = MAX_ATTEMPTS,
) -> Outcome:
    """Generate, check and revise until a draft passes or attempts run out."""
    outcome = Outcome("rejected", chunk, objective, level)
    previous = None
    for _ in range(max_attempts):
        previous = review_attempt(llm, chunk, objective, level, language, previous)
        outcome.attempts.append(previous)
        if previous.draft and not previous.problems:
            outcome.status = "verified"
            break
    return outcome


def outcome_payload(outcome: Outcome) -> dict:
    """The JSON document that `save_outcome` in schema.sql stores in one call."""
    return {
        "status": outcome.status,
        "chunk_id": outcome.chunk.id,
        "objective_id": outcome.objective.id,
        "bloom_level": outcome.level,
        "draft": outcome.draft.model_dump() if outcome.draft else None,
        "attempts": [
            {"problems": a.problems, "draft": a.draft.model_dump() if a.draft else None}
            for a in outcome.attempts
        ],
    }
