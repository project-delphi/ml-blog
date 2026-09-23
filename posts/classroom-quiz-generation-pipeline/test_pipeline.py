"""Offline tests; nothing here sends a request to Anthropic."""

import json

import anthropic
import httpx2
import pytest
from app import app, get_llm, get_store
from fastapi.testclient import TestClient
from pipeline import (
    Chunk,
    ClaudeLLM,
    ModelOutputError,
    Objective,
    Option,
    OptionJudgement,
    PedagogyCheck,
    QuestionDraft,
    SourceCheck,
    chunk_pages,
    contract_problems,
    generate_question,
    match_key,
    outcome_payload,
    pedagogy_problems,
    review_draft,
    source_problems,
)

# A passage written for these tests, not taken from any textbook.
PASSAGE = (
    "The standard deviation describes how spread out individual measurements are. "
    "The standard error describes something different: how much the sample mean "
    "would vary if we repeated the study with new samples of the same size. For "
    "independent measurements, the standard error of the mean equals the standard "
    "deviation divided by the square root of the sample size. Collecting more data "
    "therefore does not make individual measurements less variable; it makes the "
    "average more stable. Because of the square root, quadrupling the sample size "
    "only halves the standard error."
)

CHUNK = Chunk("stats-notes:p1:0000000000", "stats-notes", 1, PASSAGE)
OBJECTIVE = Objective(
    "STAT101-O4",
    "Explain why the standard error of a mean shrinks as the sample grows.",
    ("se-equals-sd", "linear-in-n", "more-data-less-spread"),
)


def good_draft(**changes) -> QuestionDraft:
    draft = QuestionDraft(
        stem=(
            "Reaction times in a lab have a standard deviation of 40 ms. With 25 "
            "participants the standard error of the mean is 8 ms. How many "
            "participants bring it down to 4 ms?"
        ),
        options=[
            Option(text="100", correct=True, misconception=None),
            Option(text="50", correct=False, misconception="linear-in-n"),
            Option(
                text="None: the standard error always equals the standard deviation",
                correct=False,
                misconception="se-equals-sd",
            ),
            Option(
                text="Any number above 25, because each measurement gets less noisy",
                correct=False,
                misconception="more-data-less-spread",
            ),
        ],
        bloom_level="apply",
        evidence=(
            "Because of the square root, quadrupling the sample size only halves "
            "the standard error."
        ),
        explanation="Halving the standard error takes four times the sample: 100.",
    )
    return draft.model_copy(update=changes)


def clean_source_check() -> SourceCheck:
    return SourceCheck(
        answer=0,
        judgements=[
            OptionJudgement(option=0, status="supported", note="4x n halves SE"),
            OptionJudgement(option=1, status="contradicted", note="not linear"),
            OptionJudgement(option=2, status="contradicted", note="SE = SD/sqrt n"),
            OptionJudgement(option=3, status="contradicted", note="spread unchanged"),
        ],
    )


def clean_pedagogy() -> PedagogyCheck:
    return PedagogyCheck(
        bloom_level="apply",
        one_defensible_answer=True,
        distractors_plausible=True,
        reasons=[],
    )


class ScriptedLLM:
    """Returns queued answers per schema and records every prompt it saw."""

    def __init__(self, script):
        self.script = {schema: list(items) for schema, items in script.items()}
        self.calls = []

    def __call__(self, *, system, prompt, schema, effort):
        self.calls.append({"schema": schema, "system": system, "prompt": prompt})
        item = self.script[schema].pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ----------------------------------------------------------------- chunking


def test_chunks_never_cross_pages_and_respect_the_word_budget():
    pages = [PASSAGE, "Short second page. It has two sentences."]
    chunks = chunk_pages("stats-notes", pages, max_words=40)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert chunks[-1].text == "Short second page. It has two sentences."
    assert " ".join(c.text for c in chunks if c.page == 1) == PASSAGE


def test_chunk_ids_are_deterministic_addresses():
    first = chunk_pages("stats-notes", [PASSAGE])
    again = chunk_pages("stats-notes", [PASSAGE])
    assert [c.id for c in first] == [c.id for c in again]
    assert first[0].id.startswith("stats-notes:p1:")


def test_cleaning_undoes_pdf_artefacts():
    [chunk] = chunk_pages("d", ["The e\ufb03cient esti-\nmator is  un\u00adbiased."])
    assert chunk.text == "The efficient estimator is unbiased."


def test_match_key_forgives_typography_but_not_words():
    assert match_key("the \u201cmean\u201d \u2013 not") == match_key('the "mean" - not')
    assert match_key("the mean") != match_key("the median")


# --------------------------------------------------------- the contract


def test_a_faithful_draft_passes_the_contract():
    assert contract_problems(good_draft(), CHUNK, OBJECTIVE, "apply") == []


def test_quote_wrapped_in_curly_quotes_still_matches():
    quote = "Collecting more data therefore does not make individual measurements"
    draft = good_draft(evidence=f"\u201c{quote}.\u201d")
    assert contract_problems(draft, CHUNK, OBJECTIVE, "apply") == []


def test_invented_evidence_is_caught_before_any_reviewer():
    draft = good_draft(evidence="Standard error falls in proportion to sample size.")
    assert contract_problems(draft, CHUNK, OBJECTIVE, "apply") == [
        "the evidence quote does not appear in the passage"
    ]


def test_tags_must_come_from_the_objectives_list():
    options = good_draft().options
    options[1] = Option(text="50", correct=False, misconception="bad-arithmetic")
    problems = contract_problems(good_draft(options=options), CHUNK, OBJECTIVE, "apply")
    assert len(problems) == 1 and "'bad-arithmetic'" in problems[0]


def test_two_keys_and_a_tagged_key_are_both_reported():
    options = good_draft().options
    options[1] = Option(text="50", correct=True, misconception="linear-in-n")
    problems = contract_problems(good_draft(options=options), CHUNK, OBJECTIVE, "apply")
    assert "expected exactly one correct option" in problems
    assert "option 1 is the key but carries a misconception tag" in problems


# ------------------------------------------------------------ reviewers


def test_blind_answer_that_disagrees_with_the_key_fails():
    check = clean_source_check().model_copy(update={"answer": 1})
    assert source_problems(check, good_draft())[0].startswith(
        "reading only the passage, the reviewer chose option 1"
    )


def test_a_distractor_the_passage_supports_fails():
    check = clean_source_check()
    check.judgements[3] = OptionJudgement(option=3, status="supported", note="!")
    assert source_problems(check, good_draft()) == [
        "the passage also supports option 3: !"
    ]


def test_recall_question_fails_pedagogy_with_the_reviewers_reasons():
    check = clean_pedagogy().model_copy(
        update={"bloom_level": "remember", "reasons": ["the stem repeats line 4"]}
    )
    assert pedagogy_problems(check, "apply") == [
        "asked for apply; it reads as remember",
        "the stem repeats line 4",
    ]


def test_passing_pedagogy_ignores_stray_reasons():
    check = clean_pedagogy().model_copy(update={"reasons": ["fine"]})
    assert pedagogy_problems(check, "apply") == []


# ----------------------------------------------------------------- the loop


def test_failed_draft_is_revised_with_its_reasons_then_verified():
    invented = good_draft(evidence="Standard error falls in proportion to sample size.")
    llm = ScriptedLLM(
        {
            QuestionDraft: [invented, good_draft()],
            SourceCheck: [clean_source_check()],
            PedagogyCheck: [clean_pedagogy()],
        }
    )
    outcome = generate_question(llm, CHUNK, OBJECTIVE, "apply")
    assert outcome.status == "verified"
    assert [a.problems for a in outcome.attempts] == [
        ["the evidence quote does not appear in the passage"],
        [],
    ]
    second_prompt = llm.calls[1]["prompt"]
    assert "Revise this draft" in second_prompt
    assert "does not appear in the passage" in second_prompt


def test_source_reviewer_never_sees_the_key():
    llm = ScriptedLLM(
        {
            QuestionDraft: [good_draft()],
            SourceCheck: [clean_source_check()],
            PedagogyCheck: [clean_pedagogy()],
        }
    )
    generate_question(llm, CHUNK, OBJECTIVE, "apply")
    prompts = {c["schema"]: c["prompt"] for c in llm.calls}
    source_prompt, pedagogy_prompt = prompts[SourceCheck], prompts[PedagogyCheck]
    assert "(correct)" not in source_prompt
    assert "(correct)" in pedagogy_prompt


def test_attempts_run_out_and_the_draft_is_kept_as_rejected():
    disagree = clean_source_check().model_copy(update={"answer": 2})
    llm = ScriptedLLM({QuestionDraft: [good_draft()] * 3, SourceCheck: [disagree] * 3})
    outcome = generate_question(llm, CHUNK, OBJECTIVE, "apply")
    assert outcome.status == "rejected"
    assert len(outcome.attempts) == 3
    assert outcome.draft == good_draft()
    assert not any(c["schema"] is PedagogyCheck for c in llm.calls)


def test_a_refusal_is_a_failed_attempt_not_a_crash():
    llm = ScriptedLLM(
        {
            QuestionDraft: [ModelOutputError("the model stopped with refusal")] * 3,
        }
    )
    outcome = generate_question(llm, CHUNK, OBJECTIVE, "apply")
    assert outcome.status == "rejected" and outcome.draft is None
    payload = outcome_payload(outcome)
    assert payload["draft"] is None
    assert payload["attempts"][0]["problems"] == [
        "generator: the model stopped with refusal"
    ]


def test_course_language_reaches_the_generator_but_not_the_quote():
    llm = ScriptedLLM({QuestionDraft: [ModelOutputError("x")]})
    generate_question(llm, CHUNK, OBJECTIVE, "apply", language="es", max_attempts=1)
    system = llm.calls[0]["system"]
    assert "Write the stem, options and explanation in Spanish." in system
    assert "in the passage's\nown language" in system


def test_an_edited_question_is_reviewed_without_being_rewritten():
    edited = good_draft(
        stem="With 25 participants the standard error is 8 ms. How "
        "many participants bring it down to 4 ms?"
    )
    llm = ScriptedLLM(
        {SourceCheck: [clean_source_check()], PedagogyCheck: [clean_pedagogy()]}
    )
    assert review_draft(llm, edited, CHUNK, OBJECTIVE, "apply") == []
    assert [c["schema"] for c in llm.calls] == [SourceCheck, PedagogyCheck]


# ---------------------------------------------------- the Claude contract


def claude_answering(content, stop_reason="end_turn", seen=None):
    def handler(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5",
                "content": content,
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(handler)
        ),
    )
    return ClaudeLLM(client)


def ask(llm):
    return llm(system="s", prompt="p", schema=SourceCheck, effort="medium")


def test_request_carries_schema_effort_and_fallbacks():
    seen = []
    body = clean_source_check().model_dump_json()
    ask(claude_answering([{"type": "text", "text": body}], seen=seen))
    [request] = seen
    sent = json.loads(request.content)
    assert sent["model"] == "claude-opus-5"
    assert sent["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in request.headers["anthropic-beta"]
    assert sent["output_config"]["effort"] == "medium"
    schema = sent["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"answer", "judgements"}


def test_output_after_a_fallback_block_is_still_parsed():
    fallback = {
        "type": "fallback",
        "from": {"model": "claude-opus-5"},
        "to": {"model": "claude-opus-4-8"},
        "trigger": {"type": "refusal", "category": "cyber"},
    }
    text = {"type": "text", "text": clean_source_check().model_dump_json()}
    assert ask(claude_answering([fallback, text])) == clean_source_check()


def test_an_api_failure_after_retries_is_a_failed_attempt():
    def handler(request):
        return httpx2.Response(
            529,
            json={
                "type": "error",
                "error": {"type": "overloaded_error", "message": "busy"},
            },
        )

    client = anthropic.Anthropic(
        api_key="test",
        max_retries=0,
        http_client=anthropic.DefaultHttpxClient(
            transport=httpx2.MockTransport(handler)
        ),
    )
    with pytest.raises(ModelOutputError, match="API error"):
        ask(ClaudeLLM(client))


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refusal_and_truncation_raise_before_parsing(stop_reason):
    llm = claude_answering(
        [{"type": "text", "text": '{"answer": 0, "judg'}], stop_reason
    )
    with pytest.raises(ModelOutputError, match=stop_reason):
        ask(llm)


def test_output_that_breaks_the_schema_raises():
    llm = claude_answering([{"type": "text", "text": '{"answer": "zero"}'}])
    with pytest.raises(ModelOutputError, match="SourceCheck"):
        ask(llm)


# ------------------------------------------------------------------ the app


class MemoryStore:
    def __init__(self):
        self.chunks, self.outcomes = {}, []
        self.objectives = {OBJECTIVE.id: OBJECTIVE}

    def add_document(self, document_id, course_id, title, chunks):
        self.chunks.update({c.id: c for c in chunks})

    def chunk(self, chunk_id):
        return self.chunks.get(chunk_id)

    def objective(self, objective_id):
        return self.objectives.get(objective_id)

    def save_outcome(self, payload):
        self.outcomes.append(payload)
        return len(self.outcomes)


def tiny_pdf(pages: list[str]) -> bytes:
    """A valid PDF with one line of Helvetica text per page."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        stream = f"BT /F1 9 Tf 20 700 Td ({text}) Tj ET"
        objects.append(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 4000 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offsets = b"%PDF-1.4\n", []
    for k, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{k} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode()
    return out


@pytest.fixture
def client_and_store():
    store = MemoryStore()
    llm = ScriptedLLM(
        {
            QuestionDraft: [good_draft()],
            SourceCheck: [clean_source_check()],
            PedagogyCheck: [clean_pedagogy()],
        }
    )
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm
    with TestClient(app) as client:
        yield client, store
    app.dependency_overrides.clear()


def test_upload_then_generate_stores_a_verified_question(client_and_store):
    client, store = client_and_store
    pdf = tiny_pdf([PASSAGE, "A second page about variance."])
    response = client.post(
        "/documents",
        files={"file": ("stats-notes.pdf", pdf, "application/pdf")},
        data={"course_id": "STAT101"},
    )
    assert response.status_code == 200
    chunk_ids = response.json()["chunks"]
    assert [c.split(":")[1] for c in chunk_ids] == ["p1", "p2"]
    assert chunk_ids[0].startswith("STAT101-stats-notes-")
    assert store.chunk(chunk_ids[0]).text == PASSAGE

    response = client.post(
        "/questions",
        json={"chunk_id": chunk_ids[0], "objective_id": OBJECTIVE.id, "level": "apply"},
    )
    assert response.status_code == 202
    [saved] = store.outcomes
    assert saved["status"] == "verified"
    assert saved["chunk_id"] == chunk_ids[0]
    assert saved["draft"]["options"][1]["misconception"] == "linear-in-n"


def test_same_name_different_course_or_content_gets_new_ids(client_and_store):
    client, _ = client_and_store

    def upload(course, pages):
        files = {"file": ("lecture1.pdf", tiny_pdf(pages), "application/pdf")}
        response = client.post("/documents", files=files, data={"course_id": course})
        return response.json()["document_id"]

    first = upload("STAT101", [PASSAGE])
    assert upload("STAT101", [PASSAGE]) == first  # same file again: same ids
    assert upload("BIO110", [PASSAGE]) != first
    assert upload("STAT101", ["A corrected passage."]) != first


def test_a_dash_in_a_course_code_cannot_forge_another_documents_id(client_and_store):
    client, _ = client_and_store

    def upload(course, filename):
        files = {"file": (filename, tiny_pdf([PASSAGE]), "application/pdf")}
        return client.post("/documents", files=files, data={"course_id": course})

    a = upload("a-b", "c.pdf").json()["document_id"]
    b = upload("a", "b-c.pdf").json()["document_id"]
    assert a != b


def test_unknown_chunk_is_a_404(client_and_store):
    client, _ = client_and_store
    response = client.post(
        "/questions",
        json={"chunk_id": "nope", "objective_id": OBJECTIVE.id, "level": "apply"},
    )
    assert response.status_code == 404
