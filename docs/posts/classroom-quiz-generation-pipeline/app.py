"""FastAPI service: upload a reading, then generate verified questions from it."""

from __future__ import annotations

import hashlib
import io
import os
from functools import lru_cache
from typing import Annotated, Protocol

from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, UploadFile
from pipeline import (
    LLM,
    BloomLevel,
    Chunk,
    ClaudeLLM,
    Language,
    Objective,
    chunk_pages,
    extract_pages,
    generate_question,
    outcome_payload,
)
from pydantic import BaseModel


class Store(Protocol):
    """Where chunks, objectives and outcomes live."""

    def add_document(
        self, document_id: str, course_id: str, title: str, chunks: list[Chunk]
    ) -> None: ...
    def chunk(self, chunk_id: str) -> Chunk | None: ...
    def objective(self, objective_id: str) -> Objective | None: ...
    def save_outcome(self, payload: dict) -> int: ...


class PgStore:
    """The Postgres store; all it knows about the schema is in schema.sql."""

    def __init__(self, dsn: str):
        """Remember where the database is; each call opens its own connection."""
        self.dsn = dsn

    def _connect(self):
        # One connection per call: endpoints and background jobs run on
        # different threads, and must not share a transaction.
        import psycopg

        return psycopg.connect(self.dsn)

    def add_document(self, document_id, course_id, title, chunks):
        """Insert a document and its chunks in one transaction.

        Ids come from the file's content, so uploading the same file again
        changes nothing and a corrected file gets new ids.
        """
        with self._connect() as conn:
            conn.execute(
                "insert into documents (id, course_id, title) values (%s, %s, %s)"
                " on conflict do nothing",
                [document_id, course_id, title],
            )
            conn.cursor().executemany(
                "insert into chunks (id, document_id, page, text)"
                " values (%s, %s, %s, %s) on conflict do nothing",
                [(c.id, c.document_id, c.page, c.text) for c in chunks],
            )

    def chunk(self, chunk_id):
        """Fetch one chunk by id."""
        with self._connect() as conn:
            row = conn.execute(
                "select id, document_id, page, text from chunks where id = %s",
                [chunk_id],
            ).fetchone()
        return Chunk(*row) if row else None

    def objective(self, objective_id):
        """Fetch one objective by id."""
        with self._connect() as conn:
            row = conn.execute(
                "select id, statement, misconceptions from objectives where id = %s",
                [objective_id],
            ).fetchone()
        return Objective(row[0], row[1], tuple(row[2])) if row else None

    def save_outcome(self, payload):
        """Store a question and its review history in one database call."""
        from psycopg.types.json import Jsonb

        with self._connect() as conn:
            return conn.execute("select save_outcome(%s)", [Jsonb(payload)]).fetchone()[
                0
            ]


@lru_cache
def get_store() -> Store:
    """The store for this process, from DATABASE_URL."""
    return PgStore(os.environ["DATABASE_URL"])


def get_llm() -> LLM:
    """The model the pipeline talks to."""
    return ClaudeLLM()


app = FastAPI()
StoreDep = Annotated[Store, Depends(get_store)]


@app.post("/documents")
def upload(file: UploadFile, course_id: Annotated[str, Form()], store: StoreDep):
    """Split an uploaded PDF into addressed chunks and store them."""
    data = file.file.read()
    stem = os.path.splitext(os.path.basename(file.filename or "upload"))[0]
    # Course, name and a hash of the bytes: two courses can both upload
    # lecture1.pdf, and a corrected file never overwrites the questions
    # already tied to the old one.
    document_id = f"{course_id}-{stem}-{hashlib.sha256(data).hexdigest()[:6]}"
    chunks = chunk_pages(document_id, extract_pages(io.BytesIO(data)))
    store.add_document(document_id, course_id, file.filename or document_id, chunks)
    return {"document_id": document_id, "chunks": [c.id for c in chunks]}


class GenerateRequest(BaseModel):
    """One generation job: a passage, the objective it serves, a Bloom level."""

    chunk_id: str
    objective_id: str
    level: BloomLevel
    language: Language = "en"


@app.post("/questions", status_code=202)
def queue_question(
    job: GenerateRequest,
    tasks: BackgroundTasks,
    store: StoreDep,
    llm: Annotated[LLM, Depends(get_llm)],
):
    """Accept a job and run the review loop after the response is sent."""
    chunk, objective = store.chunk(job.chunk_id), store.objective(job.objective_id)
    if chunk is None or objective is None:
        raise HTTPException(404, "unknown chunk or objective")
    tasks.add_task(run_job, llm, store, chunk, objective, job.level, job.language)
    return {"queued": job.chunk_id}


def run_job(llm, store, chunk, objective, level, language):
    """Run the loop and store whatever it ends with, verified or rejected."""
    outcome = generate_question(llm, chunk, objective, level, language)
    store.save_outcome(outcome_payload(outcome))
