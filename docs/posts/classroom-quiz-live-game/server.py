"""FastAPI game server: rooms over WebSockets, a QR code to join, answers saved."""

from __future__ import annotations

import asyncio
import os
import secrets
from datetime import datetime
from functools import lru_cache
from typing import Annotated, Protocol

import qrcode
import qrcode.image.svg
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from game import Answer, NotApproved, Rejected, Room, load_questions
from pydantic import BaseModel

PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://localhost:3000")


class QuestionSource(Protocol):
    """Reads questions, with their triage status, from Part 1's tables."""

    def rows(self, question_ids: list[int]) -> list[dict]: ...


class Triage(Protocol):
    """The professor's review queue over Part 1's questions table."""

    def queue(self, course_id: str) -> list[dict]: ...
    def approve(self, question_id: int) -> bool: ...
    def edit(self, question_id: int, stem: str, options: list[dict]) -> bool: ...


class Sink(Protocol):
    """Receives everything Part 3's diagnostics will need."""

    def session_started(self, session: str, course_id: str, questions: list[int]): ...
    def player_joined(self, session: str, player: str, at: datetime): ...
    def question_opened(self, session: str, question_id: int, at: datetime): ...
    def answer(self, session: str, answer: Answer): ...


class Postgres:
    """Both roles against the schema in schema.sql and sessions.sql."""

    def __init__(self, dsn: str):
        """Open one autocommitting connection."""
        import psycopg
        from psycopg.rows import dict_row

        self.conn = psycopg.connect(dsn, autocommit=True, row_factory=dict_row)

    def rows(self, question_ids):
        """Approved or not, with a readable source label for the reveal."""
        return self.conn.execute(
            "select q.id, q.status::text as status, q.stem, q.options, q.evidence,"
            " q.explanation, d.title || ', p. ' || c.page as source"
            " from questions q join chunks c on c.id = q.chunk_id"
            " join documents d on d.id = c.document_id"
            " where q.id = any(%s::bigint[]) order by array_position(%s::bigint[], q.id)",
            [question_ids, question_ids],
        ).fetchall()

    def queue(self, course_id):
        """Verified questions waiting for a decision, with source and history."""
        return self.conn.execute(
            "select q.id, q.stem, q.options, q.evidence, q.explanation, q.bloom_level,"
            " d.title || ', p. ' || c.page as source,"
            " (select count(*) from reviews r where r.question_id = q.id) as attempts"
            " from questions q"
            " join chunks c on c.id = q.chunk_id"
            " join documents d on d.id = c.document_id"
            " join objectives o on o.id = q.objective_id"
            " where o.course_id = %s and q.status = 'verified' order by q.id",
            [course_id],
        ).fetchall()

    def approve(self, question_id):
        """Approve a verified question. Anything else is left alone."""
        row = self.conn.execute(
            "update questions set status = 'approved'"
            " where id = %s and status = 'verified' returning id",
            [question_id],
        ).fetchone()
        return row is not None

    def edit(self, question_id, stem, options):
        """Save an edit. It voids the old verification, so it is a draft again."""
        from psycopg.types.json import Jsonb

        row = self.conn.execute(
            "update questions set stem = %s, options = %s, status = 'draft'"
            " where id = %s and status in ('verified', 'approved') returning id",
            [stem, Jsonb(options), question_id],
        ).fetchone()
        return row is not None

    def session_started(self, session, course_id, questions):
        """Record the session and its question order."""
        self.conn.execute(
            "insert into sessions (id, course_id, question_ids) values (%s, %s, %s)",
            [session, course_id, questions],
        )

    def player_joined(self, session, player, at):
        """Record who was in the room, and from when."""
        self.conn.execute(
            "insert into session_players (session_id, player_id, joined_at)"
            " values (%s, %s, %s) on conflict do nothing",
            [session, player, at],
        )

    def question_opened(self, session, question_id, at):
        """Record when a question went live."""
        self.conn.execute(
            "insert into session_questions (session_id, question_id, opened_at)"
            " values (%s, %s, %s) on conflict do nothing",
            [session, question_id, at],
        )

    def answer(self, session, answer):
        """Store an answer; a replayed answer_id is a no-op here too."""
        self.conn.execute(
            "insert into responses (answer_id, session_id, player_id, question_id,"
            " choice, timing, points, received_at)"
            " values (%s, %s, %s, %s, %s, %s, %s, %s) on conflict do nothing",
            [
                answer.answer_id,
                session,
                answer.player,
                answer.question_id,
                answer.choice,
                answer.timing,
                answer.points,
                answer.received_at,
            ],
        )


@lru_cache
def get_db() -> Postgres:
    """One connection per process, from DATABASE_URL."""
    return Postgres(os.environ["DATABASE_URL"])


def get_source() -> QuestionSource:
    """Where approved questions come from."""
    return get_db()


def get_sink() -> Sink:
    """Where session events go."""
    return get_db()


def get_triage() -> Triage:
    """Where the professor's decisions go."""
    return get_db()


app = FastAPI()
# Rooms live in this process's memory: run one worker, or route every
# connection for a room to the same worker.
rooms: dict[str, Room] = {}
host_keys: dict[str, str] = {}
sockets: dict[str, dict[WebSocket, str | None]] = {}
timers: set[asyncio.Task] = set()


def session_id(room: Room) -> str:
    """Room codes get reused; the epoch makes the session unique."""
    return f"{room.code}-{room.epoch}"


def join_url(code: str) -> str:
    """The page a phone opens. Built from config, not the request's host."""
    return f"{PUBLIC_URL}/join/{code}"


def qr_svg(url: str) -> str:
    """An SVG QR code; scales to any projector, needs no imaging library."""
    image = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, border=2)
    return image.to_string(encoding="unicode")


@app.get("/triage/{course_id}")
def triage_queue(course_id: str, triage: Annotated[Triage, Depends(get_triage)]):
    """What the dashboard lists: verified drafts beside their evidence."""
    return triage.queue(course_id)


@app.post("/questions/{question_id}/approve")
def approve(question_id: int, triage: Annotated[Triage, Depends(get_triage)]):
    """The only way a question becomes playable."""
    if not triage.approve(question_id):
        raise HTTPException(409, "only a verified question can be approved")
    return {"id": question_id, "status": "approved"}


class Edit(BaseModel):
    """A professor's rewrite of a question's wording or options."""

    stem: str
    options: list[dict]


@app.post("/questions/{question_id}/edit")
def edit(
    question_id: int, change: Edit, triage: Annotated[Triage, Depends(get_triage)]
):
    """Save an edit; the question must pass review again before approval."""
    if not triage.edit(question_id, change.stem, change.options):
        raise HTTPException(409, "only a verified or approved question can be edited")
    return {"id": question_id, "status": "draft"}


class NewRoom(BaseModel):
    """What the professor launches from the triage dashboard."""

    course_id: str
    question_ids: list[int]
    seconds: float = 20.0


@app.post("/rooms")
def create_room(
    spec: NewRoom,
    source: Annotated[QuestionSource, Depends(get_source)],
    sink: Annotated[Sink, Depends(get_sink)],
):
    """Open a room, but only for questions the professor approved."""
    try:
        rows = [
            row | {"seconds": spec.seconds} for row in source.rows(spec.question_ids)
        ]
        questions = load_questions(rows)
    except NotApproved as err:
        raise HTTPException(409, str(err)) from err
    if len(questions) != len(spec.question_ids):
        raise HTTPException(404, "unknown question id")
    room = Room(questions)
    while room.code in rooms:
        room = Room(questions)
    rooms[room.code], sockets[room.code] = room, {}
    host_keys[room.code] = secrets.token_urlsafe(16)
    sink.session_started(session_id(room), spec.course_id, spec.question_ids)
    return {
        "code": room.code,
        "join_url": join_url(room.code),
        "host_key": host_keys[room.code],
    }


@app.get("/rooms/{code}/qr.svg")
def room_qr(code: str):
    """The QR code the projector shows in the lobby."""
    if code not in rooms:
        raise HTTPException(404, "no such room")
    return Response(qr_svg(join_url(code)), media_type="image/svg+xml")


async def broadcast(room: Room) -> None:
    """Send every screen the new public state, plus each phone its own view."""
    public = room.snapshot()
    for ws, player in list(sockets[room.code].items()):
        you = room.private_view(player) if player else None
        try:
            await ws.send_json({"type": "state", "public": public, "you": you})
        except (WebSocketDisconnect, RuntimeError):
            sockets[room.code].pop(ws, None)


async def lock_when_due(room: Room) -> None:
    """The server, not any phone, decides when time is up."""
    question = room.current.id
    # asyncio can wake a timer a hair early, so keep going until the lock
    # happens, or the host has already moved on from this question.
    while room.phase == "question" and room.current.id == question:
        await asyncio.sleep(max(room.remaining(), 0.01))
        if room.tick():
            await broadcast(room)


@app.websocket("/ws/{code}")
async def play(
    ws: WebSocket,
    code: str,
    sink: Annotated[Sink, Depends(get_sink)],
    name: str | None = None,
    player: str | None = None,
    host: str | None = None,
):
    """One connection: a phone, the projector, or the professor's controls."""
    room = rooms.get(code)
    if room is None:
        await ws.close(code=4404)
        return
    await ws.accept()
    is_host = host is not None and secrets.compare_digest(host, host_keys[code])
    if player not in room.names:
        player = None
    if player is None and name and not is_host:
        player = room.join(name)
        await asyncio.to_thread(
            sink.player_joined, session_id(room), player, room.wall()
        )
        await ws.send_json({"type": "welcome", "player": player})
    sockets[code][ws] = player
    await broadcast(room)  # a rejoining phone catches up from this alone

    try:
        while True:
            try:
                message = await ws.receive_json()
                await handle(room, sink, ws, player, is_host, message)
            except (Rejected, KeyError, TypeError, ValueError) as err:
                await ws.send_json({"type": "error", "message": str(err)})
    except WebSocketDisconnect:
        pass
    finally:
        sockets[code].pop(ws, None)


async def handle(room, sink, ws, player, is_host, message) -> None:
    """Apply one message. Database writes run off the event loop."""
    if message["type"] == "command" and is_host:
        room.command(message["name"])
        if message["name"] == "next":
            opened = (session_id(room), room.current.id, room.wall())
            await asyncio.to_thread(sink.question_opened, *opened)
            task = asyncio.create_task(lock_when_due(room))
            timers.add(task)
            task.add_done_callback(timers.discard)
        await broadcast(room)
    elif message["type"] == "answer" and player:
        answer, _ = room.submit(
            player, message["answer_id"], message["question_id"], message["choice"]
        )
        # Write before acknowledging, on replays too: the insert ignores a
        # duplicate answer_id, so a write that failed last time is retried.
        await asyncio.to_thread(sink.answer, session_id(room), answer)
        await ws.send_json({"type": "ack", "answer_id": answer.answer_id})
    else:
        raise Rejected(f"{message['type']!r} is not allowed here")
