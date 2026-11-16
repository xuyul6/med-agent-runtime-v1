"""Forgetting a demo user: their sessions, summaries, bookings and in-process pointers go; nobody else's do."""
from datetime import date

import pytest
from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.core.config import Settings
from medagent.memory import SessionMemory, owner_key, scope_key
from support.appointments import Clock, booking_store
from support.harness import fixture, run
from support.multiturn import ContextModel, ask

SLOT = "demo-general:neurology:2026-11-04:am"


class Vectors:
    async def encode(self, texts): return [[1.0, 0.0] for _ in texts]


def chroma_collection(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings as ChromaSettings
    client = chromadb.PersistentClient(path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False))
    return client.get_or_create_collection("episodes_forget", embedding_function=None)


def test_forget_deletes_one_users_sessions_and_summaries_in_real_chroma(tmp_path):
    episodes = chroma_collection(tmp_path)
    memory = SessionMemory(episodes=episodes, vectors=Vectors(), episode_ttl=1000)
    memory.context_clock = lambda: 500.0

    async def scenario():
        for user, session, subject in [("alice", "s1", "self"), ("alice", "s2", "mother"), ("bob", "s1", "self")]:
            await memory.track(user, session, subject)
            await memory.remember(scope_key(user, session, subject), "问题", "回答")
        episodes.add(ids=["a-self", "a-mother", "b-self"], documents=["甲本人摘要", "甲母亲摘要", "乙本人摘要"],
                     embeddings=[[1.0, 0.0]]*3,
                     metadatas=[{"scope":scope_key("alice", "s1", "self"), "owner":owner_key("alice", "self"), "created_at":400.0},
                                {"scope":scope_key("alice", "s2", "mother"), "owner":owner_key("alice", "mother"), "created_at":400.0},
                                {"scope":scope_key("bob", "s1", "self"), "owner":owner_key("bob", "self"), "created_at":400.0}])

        forgotten = await memory.forget("alice")

        assert sorted(forgotten.sessions) == [("s1", "self"), ("s2", "mother")]
        assert forgotten.summaries == 2
        assert (await memory.read(scope_key("alice", "s1", "self")))["messages"] == []
        assert (await memory.read(scope_key("alice", "s2", "mother")))["messages"] == []
        assert len((await memory.read(scope_key("bob", "s1", "self")))["messages"]) == 2
        assert episodes.get()["ids"] == ["b-self"]
        assert await memory.recall(scope_key("alice", "s3", "self"), "就医", owner_key("alice", "self")) == []
        assert (await memory.forget("alice")).summaries == 0
    run(scenario())


def test_forget_finds_summaries_of_well_known_subjects_without_a_session_index(tmp_path):
    episodes = chroma_collection(tmp_path)
    memory = SessionMemory(episodes=episodes, vectors=Vectors(), episode_ttl=1000)
    episodes.add(ids=["old"], documents=["索引过期后的摘要"], embeddings=[[1.0, 0.0]],
                 metadatas=[{"scope":"x", "owner":owner_key("alice", "father"), "created_at":400.0}])
    assert run(memory.forget("alice")).summaries == 1
    assert episodes.get()["ids"] == []


def test_store_forget_removes_only_that_users_records_and_frees_capacity():
    store = booking_store(Clock())
    for user in ("alice", "bob"):
        draft, token = store.propose_create(user=user, session="s", subject="self", slot_id=SLOT)
        store.confirm(user=user, draft_id=draft.draft_id, token=token)
    store.propose_create(user="alice", session="s", subject="self", slot_id=SLOT.replace(":am", ":pm"))
    remaining_before = dict((slot.slot_id, left) for slot, left in store.search("demo-general", "neurology", date(2026, 11, 4), "am"))[SLOT]

    assert store.forget("alice") == 1

    assert [row[1] for row in store.booking_rows()] == ["bob"]
    assert store.upcoming("alice") == []
    remaining_after = dict((slot.slot_id, left) for slot, left in store.search("demo-general", "neurology", date(2026, 11, 4), "am"))[SLOT]
    assert remaining_after == remaining_before+1
    assert store.forget("alice") == 0


def test_runtime_forget_resets_subject_pointer_and_history_for_that_user_only():
    async def scenario():
        runtime = await fixture(ContextModel())
        assert (await ask(runtime, "现在给我妈妈问布洛芬资料", user="alice", session="s")).subject == "mother"
        assert (await ask(runtime, "现在给我妈妈问布洛芬资料", user="bob", session="s")).subject == "mother"

        result = await runtime.forget("alice")

        assert result["sessions"] == 1
        follow = await ask(runtime, "这个药呢？", user="alice", session="s")
        assert follow.subject == "self"
        assert "medication_name" not in follow.intent.entities
        assert (await ask(runtime, "这个药呢？", user="bob", session="s")).subject == "mother"
    run(scenario())


def test_forget_clears_a_recent_urgent_hint():
    async def scenario():
        runtime = await fixture(ContextModel())
        assert (await ask(runtime, "我现在胸痛", user="alice", session="s")).risk.level == "CRITICAL"
        await runtime.forget("alice")
        assert (await ask(runtime, "还是这样，越来越疼", user="alice", session="s")).risk.level == "ROUTINE"
    run(scenario())


def test_delete_user_endpoint_reports_counts_and_validates_the_id():
    store = booking_store(Clock())
    runtime = run(fixture(appointments=store))
    draft, token = store.propose_create(user="alice", session="s", subject="self", slot_id=SLOT)
    store.confirm(user="alice", draft_id=draft.draft_id, token=token)
    with TestClient(create_app(Settings(preview=True), runtime)) as client:
        response = client.delete("/users/alice")
        assert response.status_code == 200
        assert response.json() == {"user":"alice", "sessions":0, "summaries":0, "bookings":1}
        assert client.get("/appointments", params={"user":"alice"}).json() == []
        assert client.delete("/users/" + "x"*101).status_code == 422


def test_delete_user_reports_storage_failure_without_touching_bookings():
    store = booking_store(Clock())
    runtime = run(fixture(appointments=store))
    draft, token = store.propose_create(user="alice", session="s", subject="self", slot_id=SLOT)
    store.confirm(user="alice", draft_id=draft.draft_id, token=token)

    async def broken(account):
        raise ConnectionError("redis down")
    runtime.memory.forget = broken
    with TestClient(create_app(Settings(preview=True), runtime)) as client:
        response = client.delete("/users/alice")
        assert response.status_code == 503 and response.json()["detail"] == "memory_unavailable"
        assert [b["department"] for b in client.get("/appointments", params={"user":"alice"}).json()] == ["神经内科"]
