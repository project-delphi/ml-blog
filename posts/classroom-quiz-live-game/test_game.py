"""Offline tests for the state machine and the WebSocket server."""

import xml.etree.ElementTree as ET

import pytest
from fastapi.testclient import TestClient
from game import NotApproved, Rejected, Room, load_questions
from server import app, get_sink, get_source, get_triage

pytestmark = pytest.mark.timeout(10)


def row(qid, status="approved", answer=0):
    return {
        "id": qid,
        "status": status,
        "stem": f"Question {qid}?",
        "options": [
            {"text": t, "correct": i == answer, "misconception": None}
            for i, t in enumerate(["A", "B", "C", "D"])
        ],
        "explanation": "Because.",
        "evidence": "A sentence copied from the reading.",
        "source": "stats-notes.pdf, p. 1",
    }


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


@pytest.fixture
def room():
    clock = Clock()
    room = Room(load_questions([row(1), row(2, answer=2)]), clock=clock)
    room.test_clock = clock
    return room


def test_unapproved_questions_never_reach_a_room():
    with pytest.raises(NotApproved, match="question 2 is verified"):
        load_questions([row(1), row(2, status="verified")])


def test_phases_follow_host_commands_only_in_order(room):
    assert room.phase == "lobby"
    room.command("next")
    assert room.phase == "question"
    with pytest.raises(Rejected, match="cannot reveal during question"):
        room.command("reveal")
    for step in ["lock", "reveal", "leaderboard", "next"]:
        room.command(step)
    assert (room.phase, room.current.id) == ("question", 2)
    for step in ["lock", "reveal", "leaderboard"]:
        room.command(step)
    with pytest.raises(Rejected, match="no questions left"):
        room.command("next")
    room.command("finish")
    assert room.phase == "final"


def test_every_change_bumps_seq(room):
    seqs = [room.seq]
    room.join("Ana")
    seqs.append(room.seq)
    room.command("next")
    seqs.append(room.seq)
    assert seqs == sorted(set(seqs))


def test_the_server_clock_locks_the_question(room):
    room.command("next")
    room.test_clock.now += 19.9
    assert room.tick() is False and room.phase == "question"
    room.test_clock.now += 0.1
    assert room.tick() is True and room.phase == "locked"


def test_no_answer_key_before_the_reveal(room):
    room.command("next")
    for phase in ["question", "locked"]:
        assert room.phase == phase
        assert "answer" not in room.snapshot()["question"]
        room.command("lock" if phase == "question" else "reveal")
    assert room.snapshot()["question"]["answer"] == 0


def test_points_reward_being_right_and_being_quick(room):
    ana, ben = room.join("Ana"), room.join("Ben")
    room.command("next")
    room.test_clock.now += 5  # 15 of 20 seconds left
    right, _ = room.submit(ana, "a1", 1, 0)
    wrong, _ = room.submit(ben, "b1", 1, 3)
    assert (right.points, wrong.points) == (500 + 375, 0)
    assert right.timing == wrong.timing == "on_time"


def test_a_replayed_answer_is_acknowledged_but_never_rescored(room):
    ana = room.join("Ana")
    room.command("next")
    first, new = room.submit(ana, "a1", 1, 0)
    again, again_new = room.submit(ana, "a1", 1, 0)
    assert (new, again_new) == (True, False)
    assert again is first
    assert room.scores[ana] == first.points


def test_reusing_an_answer_id_or_answering_twice_is_rejected(room):
    ana = room.join("Ana")
    room.command("next")
    room.submit(ana, "a1", 1, 0)
    with pytest.raises(Rejected, match="already used"):
        room.submit(ana, "a1", 1, 2)
    with pytest.raises(Rejected, match="already answered"):
        room.submit(ana, "a2", 1, 2)


def test_answers_to_unasked_questions_are_rejected(room):
    ana = room.join("Ana")
    room.command("next")
    with pytest.raises(Rejected, match="has not been asked"):
        room.submit(ana, "a1", 2, 0)


def test_late_answers_score_nothing_and_only_pre_reveal_ones_count(room):
    ana, ben, cai = room.join("Ana"), room.join("Ben"), room.join("Cai")
    room.command("next")
    room.submit(ana, "a1", 1, 0)
    room.test_clock.now += 30
    room.tick()
    late, _ = room.submit(ben, "b1", 1, 0)  # time up, key not yet shown
    room.command("reveal")
    after, _ = room.submit(cai, "c1", 1, 0)  # arrives after the key is shown
    assert (late.timing, late.points) == ("late_before_reveal", 0)
    assert (after.timing, after.points) == ("late_after_reveal", 0)
    assert room.snapshot()["question"]["counts"] == [2, 0, 0, 0]


def test_a_queued_answer_to_an_earlier_question_is_after_its_reveal(room):
    ana = room.join("Ana")
    for step in ["next", "lock", "reveal", "leaderboard", "next"]:
        room.command(step)
    replayed, _ = room.submit(ana, "a1", 1, 0)
    assert replayed.timing == "late_after_reveal"


def test_a_phone_learns_if_it_was_right_only_at_the_reveal(room):
    ana = room.join("Ana")
    room.command("next")
    room.submit(ana, "a1", 1, 0)
    assert room.private_view(ana) == {"player": ana, "score": 1000, "answered": True}
    room.command("lock")
    room.command("reveal")
    assert room.private_view(ana)["correct"] is True


# -------------------------------------------------------------- the server


class FakeDB:
    def __init__(self, rows):
        self.by_id = {r["id"]: r for r in rows}
        self.events = []
        self.writes = 0

    def rows(self, ids):
        return [self.by_id[i] for i in ids if i in self.by_id]

    def queue(self, course_id):
        return [r for r in self.by_id.values() if r["status"] == "verified"]

    def approve(self, question_id):
        row = self.by_id.get(question_id)
        if row is None or row["status"] != "verified":
            return False
        row["status"] = "approved"
        return True

    def edit(self, question_id, stem, options):
        row = self.by_id.get(question_id)
        if row is None or row["status"] not in ("verified", "approved"):
            return False
        row.update(stem=stem, options=options, status="draft")
        return True

    def session_started(self, session, course_id, questions):
        self.events.append(("session", session))

    def player_joined(self, session, player, at):
        self.events.append(("join", player))

    def question_opened(self, session, question_id, at):
        self.events.append(("open", question_id))

    def answer(self, session, answer):
        # Like the real insert: a repeated answer_id is ignored.
        if ("answer", answer.answer_id) not in self.events:
            self.events.append(("answer", answer.answer_id))
        self.writes += 1


@pytest.fixture
def served():
    db = FakeDB(
        [row(1), row(2, answer=2), row(3, status="rejected"), row(4, "verified")]
    )
    app.dependency_overrides[get_source] = lambda: db
    app.dependency_overrides[get_sink] = lambda: db
    app.dependency_overrides[get_triage] = lambda: db
    with TestClient(app) as client:
        yield client, db
    app.dependency_overrides.clear()


def state(ws):
    while True:
        message = ws.receive_json()
        if message["type"] == "state":
            return message


def open_room(client, seconds=20.0):
    response = client.post(
        "/rooms",
        json={"course_id": "STAT101", "question_ids": [1, 2], "seconds": seconds},
    )
    assert response.status_code == 200
    return response.json()


def test_a_room_with_an_unapproved_question_is_refused(served):
    client, _ = served
    response = client.post(
        "/rooms", json={"course_id": "STAT101", "question_ids": [1, 3]}
    )
    assert response.status_code == 409


def test_triage_gates_the_room(served):
    client, _ = served
    launch = {"course_id": "STAT101", "question_ids": [1, 4]}
    assert [q["id"] for q in client.get("/triage/STAT101").json()] == [4]
    assert client.post("/rooms", json=launch).status_code == 409
    assert client.post("/questions/4/approve").json()["status"] == "approved"
    assert client.post("/rooms", json=launch).status_code == 200


def test_an_edit_sends_a_question_back_through_review(served):
    client, _ = served
    edited = {"stem": "Reworded?", "options": row(1)["options"]}
    assert client.post("/questions/1/edit", json=edited).json()["status"] == "draft"
    assert client.post("/questions/1/approve").status_code == 409
    launch = {"course_id": "STAT101", "question_ids": [1]}
    assert client.post("/rooms", json=launch).status_code == 409


def test_a_rejected_question_cannot_be_approved(served):
    client, _ = served
    assert client.post("/questions/3/approve").status_code == 409


def test_the_qr_code_encodes_the_join_url_as_scalable_svg(served):
    client, _ = served
    room = open_room(client)
    assert room["join_url"].endswith(f"/join/{room['code']}")
    response = client.get(f"/rooms/{room['code']}/qr.svg")
    assert response.headers["content-type"] == "image/svg+xml"
    svg = ET.fromstring(response.text)
    assert svg.attrib["viewBox"].startswith("0 0 ")


def test_a_full_question_over_websockets(served):
    client, db = served
    room = open_room(client)
    code, key = room["code"], room["host_key"]
    with client.websocket_connect(f"/ws/{code}?host={key}") as host:
        assert state(host)["public"]["phase"] == "lobby"
        with client.websocket_connect(f"/ws/{code}?name=Ana") as phone:
            player = phone.receive_json()["player"]
            assert state(phone)["public"]["players"] == 1
            state(host)

            host.send_json({"type": "command", "name": "next"})
            shown = state(phone)
            assert shown["public"]["phase"] == "question"
            assert "answer" not in shown["public"]["question"]
            state(host)

            answer = {"type": "answer", "answer_id": "x1", "question_id": 1}
            for _ in range(2):  # the second send is a retry after a dropped ack
                phone.send_json(answer | {"choice": 0})
                assert phone.receive_json() == {"type": "ack", "answer_id": "x1"}
            assert db.events.count(("answer", "x1")) == 1
            assert db.writes == 2  # the retry re-sent the write; storage kept one

            phone.send_json(answer | {"choice": "0"})
            assert phone.receive_json()["type"] == "error"
            phone.send_text("not json")
            assert phone.receive_json()["type"] == "error"

            phone.send_json({"type": "command", "name": "reveal"})
            assert phone.receive_json()["type"] == "error"

            host.send_json({"type": "command", "name": "lock"})
            state(host), state(phone)
            host.send_json({"type": "command", "name": "reveal"})
            revealed = state(phone)
            assert revealed["public"]["question"]["answer"] == 0
            assert revealed["you"]["correct"] is True
            state(host)

        with client.websocket_connect(f"/ws/{code}?player={player}") as again:
            caught_up = state(again)
            assert caught_up["public"]["phase"] == "reveal"
            assert caught_up["you"]["player"] == player
    assert db.events.count(("join", player)) == 1


def test_the_server_timer_locks_without_the_host(served):
    client, _ = served
    room = open_room(client, seconds=0.2)
    with client.websocket_connect(
        f"/ws/{room['code']}?host={room['host_key']}"
    ) as host:
        state(host)
        host.send_json({"type": "command", "name": "next"})
        assert state(host)["public"]["phase"] == "question"
        assert state(host)["public"]["phase"] == "locked"


class FlakyDB(FakeDB):
    """Fails the first answer write, like a dropped database connection."""

    def answer(self, session, answer):
        if self.writes == 0:
            self.writes += 1
            raise ConnectionError("database went away")
        super().answer(session, answer)


def test_an_answer_whose_write_failed_is_stored_on_replay():
    db = FlakyDB([row(1), row(2, answer=2)])
    app.dependency_overrides[get_source] = lambda: db
    app.dependency_overrides[get_sink] = lambda: db
    try:
        with TestClient(app) as client:
            room = open_room(client)
            code, key = room["code"], room["host_key"]
            with client.websocket_connect(f"/ws/{code}?host={key}") as host:
                state(host)
                answer = {"type": "answer", "answer_id": "y1", "question_id": 1}
                with (
                    pytest.raises(ConnectionError),  # the handler dies, no ack
                    client.websocket_connect(f"/ws/{code}?name=Ana") as phone,
                ):
                    player = phone.receive_json()["player"]
                    state(phone), state(host)
                    host.send_json({"type": "command", "name": "next"})
                    state(phone), state(host)
                    phone.send_json(answer | {"choice": 0})
                    phone.receive_json()
                with client.websocket_connect(f"/ws/{code}?player={player}") as again:
                    state(again)
                    again.send_json(answer | {"choice": 0})
                    assert again.receive_json() == {"type": "ack", "answer_id": "y1"}
        assert db.events.count(("answer", "y1")) == 1
    finally:
        app.dependency_overrides.clear()


def test_the_timer_keeps_trying_if_it_wakes_early(room):
    import asyncio

    import server

    room.command("next")
    server.sockets.setdefault(room.code, {})
    calls = []

    async def fake_sleep(seconds):
        calls.append(seconds)
        room.test_clock.now += seconds - 0.001 if len(calls) == 1 else seconds

    original = asyncio.sleep
    asyncio.sleep = fake_sleep
    try:
        asyncio.run(server.lock_when_due(room))
    finally:
        asyncio.sleep = original
    assert room.phase == "locked" and len(calls) == 2


def test_a_phone_that_names_itself_rejoins_as_itself(served):
    client, db = served
    room = open_room(client)
    code, key = room["code"], room["host_key"]
    me = "3f1c2a9e-5d4b-4c8e-9a71-0b6d2e4f8c13"
    with client.websocket_connect(f"/ws/{code}?host={key}") as host:
        state(host)
        # The first connection drops before the phone reads `welcome` ...
        with client.websocket_connect(f"/ws/{code}?player={me}&name=Ana"):
            pass
        # ... and the reconnect, same id, is the same player, not a second one.
        with client.websocket_connect(f"/ws/{code}?player={me}&name=Ana") as phone:
            caught_up = state(phone)
            assert caught_up["public"]["players"] == 1
            assert caught_up["you"]["player"] == me
    assert [e for e in db.events if e[0] == "join"] == [("join", me)]


def test_answers_from_an_earlier_game_are_refused_by_id(served):
    client, _ = served
    room = open_room(client)
    code, key = room["code"], room["host_key"]
    me = "7a2b9c4d-1e3f-4a5b-8c6d-9e0f1a2b3c4d"
    with (
        client.websocket_connect(f"/ws/{code}?host={key}") as host,
        client.websocket_connect(f"/ws/{code}?player={me}&name=Ana") as phone,
    ):
        assert phone.receive_json() == {"type": "welcome", "player": me}
        state(phone)
        host.send_json({"type": "command", "name": "next"})
        state(phone)
        stale = {"type": "answer", "answer_id": "old1", "question_id": 1}
        phone.send_json(stale | {"choice": 0, "epoch": "not-this-game"})
        error = phone.receive_json()
        assert (error["type"], error["answer_id"]) == ("error", "old1")
        phone.send_json(stale | {"answer_id": "new1", "choice": 0})
        assert phone.receive_json() == {"type": "ack", "answer_id": "new1"}
        phone.send_json(stale | {"answer_id": "new2", "choice": 1})
        second = phone.receive_json()  # a double tap: refused, and says which
        assert (second["type"], second["answer_id"]) == ("error", "new2")


def test_a_screen_with_no_name_only_watches(served):
    client, _ = served
    room = open_room(client)
    with client.websocket_connect(f"/ws/{room['code']}?player=unknown-id-123") as tv:
        assert state(tv)["you"] is None
        tv.send_json(
            {"type": "answer", "answer_id": "t1", "question_id": 1, "choice": 0}
        )
        assert tv.receive_json()["type"] == "error"
