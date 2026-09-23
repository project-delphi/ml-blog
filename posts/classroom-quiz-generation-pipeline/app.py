"""FastAPI service: upload a reading, then generate verified questions from it."""

from __future__ import annotations

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
        """Open one autocommitting connection."""
        import psycopg

        self.conn = psycopg.connect(dsn, autocommit=True)

    def add_document(self, document_id, course_id, title, chunks):
        """Insert a document and its chunks in one transaction."""
        with self.conn.transaction():
            self.conn.execute(
                "insert into documents (id, course_id, title) values (%s, %s, %s)",
                [document_id, course_id, title],
            )
            self.conn.cursor().executemany(
                "insert into chunks (id, document_id, page, text) "
                "values (%s, %s, %s, %s)",
                [(c.id, c.document_id, c.page, c.text) for c in chunks],
            )

    def chunk(self, chunk_id):
        """Fetch one chunk by id."""
        row = self.conn.execute(
            "select id, document_id, page, text from chunks where id = %s", [chunk_id]
        ).fetchone()
        return Chunk(*row) if row else None

    def objective(self, objective_id):
        """Fetch one objective by id."""
        row = self.conn.execute(
            "select id, statement, misconceptions from objectives where id = %s",
            [objective_id],
        ).fetchone()
        return Objective(row[0], row[1], tuple(row[2])) if row else None

    def save_outcome(self, payload):
        """Store a question and its review history in one database call."""
        from psycopg.types.json import Jsonb

        return self.conn.execute(
            "select save_outcome(%s)", [Jsonb(payload)]
        ).fetchone()[0]


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
    document_id = os.path.splitext(file.filename or "upload")[0]
    chunks = chunk_pages(document_id, extract_pages(file.file))
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
