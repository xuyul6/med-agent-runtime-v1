import pytest
from support.harness import run
from medagent.memory import PURGE_INTERVAL_SECONDS, SessionMemory, owner_key, scope_key
from medagent.core.contracts import Question
from support.harness import fixture


class Vectors:
    async def encode(self, texts): return [[1.0, 0.0] for _ in texts]


def chroma_collection(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection("episodes_retention", embedding_function=None)


def test_recall_skips_expired_and_other_scope_summaries_in_real_chroma(tmp_path):
    episodes = chroma_collection(tmp_path)
    own, other = scope_key("alice", "s", "self"), scope_key("bob", "s", "self")
    episodes.add(ids=["fresh", "old", "legacy", "other"], documents=["近期摘要", "过期摘要", "旧版摘要", "他人摘要"],
                 embeddings=[[1.0, 0.0]]*4,
                 metadatas=[{"scope":own, "created_at":950.0}, {"scope":own, "created_at":10.0},
                            {"scope":own}, {"scope":other, "created_at":990.0}])
    memory = SessionMemory(episodes=episodes, vectors=Vectors(), episode_ttl=100)
    memory.context_clock = lambda: 1000.0
    assert run(memory.recall(own, "就医准备")) == ["历史摘要（1970-01-01）：近期摘要"]
    assert run(memory.purge_expired_episodes()) == 2
    assert sorted(episodes.get()["ids"]) == ["fresh", "other"]


def test_compression_stamps_creation_time_and_purges_at_most_hourly():
    class Episodes:
        def __init__(self): self.rows, self.gets = {}, 0
        def add(self, ids, documents, embeddings, metadatas): self.rows.update(zip(ids, metadatas))
        def get(self, include):
            self.gets += 1
            return {"ids":list(self.rows), "metadatas":list(self.rows.values())}
        def delete(self, ids):
            for identifier in ids: self.rows.pop(identifier)

    class Model:
        async def json(self, stage, system, content, budget): return {"summary":"用户在整理就医资料。"}

    async def scenario():
        episodes, now = Episodes(), [5000.0]
        memory = SessionMemory(episodes=episodes, vectors=Vectors(), model=Model(), episode_ttl=1000)
        memory.context_clock = lambda: now[0]
        episodes.rows["stale"] = {"scope":"x", "created_at":1.0}
        scope = scope_key("alice", "s", "self")
        for turn in range(6):
            await memory.remember(scope, f"问题{turn}", f"回答{turn}")
        assert [meta["created_at"] for meta in episodes.rows.values()] == [5000.0]
        assert "stale" not in episodes.rows and episodes.gets == 1
        for turn in range(4):
            await memory.remember(scope, f"追问{turn}", "回答")
        assert episodes.gets == 1
        now[0] += PURGE_INTERVAL_SECONDS
        for turn in range(4):
            await memory.remember(scope, f"再问{turn}", "回答")
        assert episodes.gets == 2
    run(scenario())


def test_purge_failure_is_recorded_without_failing_the_write():
    class Broken:
        def add(self, **kwargs): pass
        def get(self, **kwargs): raise ConnectionError("chroma_down")

    class Model:
        async def json(self, *args): return {"summary":"摘要"}

    async def scenario():
        memory = SessionMemory(episodes=Broken(), vectors=Vectors(), model=Model())
        for turn in range(6):
            await memory.remember(scope_key("a", "s", "self"), f"问{turn}", "答")
        assert (await memory.read(scope_key("a", "s", "self")))["summary"] == "摘要"
    run(scenario())


def test_invalid_retention_is_rejected():
    with pytest.raises(ValueError, match="invalid_episode_ttl"):
        SessionMemory(episode_ttl=0)


def test_new_session_recalls_earlier_sessions_of_the_same_user_and_subject_in_real_chroma(tmp_path):
    episodes = chroma_collection(tmp_path)
    alice, mother, bob = owner_key("alice", "self"), owner_key("alice", "mother"), owner_key("bob", "self")
    earlier, current = scope_key("alice", "s1", "self"), scope_key("alice", "s2", "self")
    rows = [("earlier", "上次的摘要", {"scope":earlier, "owner":alice, "created_at":900.0}),
            ("current", "本会话当前摘要", {"scope":current, "owner":alice, "created_at":990.0}),
            ("mother", "母亲的摘要", {"scope":scope_key("alice", "s1", "mother"), "owner":mother, "created_at":990.0}),
            ("bob", "他人摘要", {"scope":scope_key("bob", "s1", "self"), "owner":bob, "created_at":990.0}),
            ("expired", "过期摘要", {"scope":earlier, "owner":alice, "created_at":10.0}),
            ("ownerless", "无主摘要", {"scope":current, "created_at":950.0})]
    episodes.add(ids=[row[0] for row in rows], documents=[row[1] for row in rows], embeddings=[[1.0, 0.0]]*len(rows),
                 metadatas=[row[2] for row in rows])
    memory = SessionMemory(episodes=episodes, vectors=Vectors(), episode_ttl=200, timezone="UTC")
    memory.context_clock = lambda: 1000.0
    recalled = run(memory.recall(current, "就医准备", alice, exclude="本会话当前摘要"))
    # With an owner, a summary stored without one is not recalled, even in the same session.
    assert recalled == ["历史摘要（1970-01-01）：上次的摘要"]
    # A brand-new session of the same user and subject, with nothing of its own yet.
    fresh = run(memory.recall(scope_key("alice", "s3", "self"), "就医准备", alice))
    assert len(fresh) == 2 and all("母亲" not in item and "他人" not in item and "无主" not in item for item in fresh)


def test_compression_stamps_the_owner_only_when_one_is_given():
    class Episodes:
        def __init__(self): self.rows = []
        def add(self, ids, documents, embeddings, metadatas): self.rows += metadatas
        def get(self, include): return {"ids":[], "metadatas":[]}

    class Model:
        async def json(self, *args): return {"summary":"摘要"}

    async def scenario():
        episodes = Episodes()
        memory = SessionMemory(episodes=episodes, vectors=Vectors(), model=Model())
        for turn in range(6):
            await memory.remember(scope_key("a", "s", "self"), f"问{turn}", "答", owner=owner_key("a", "self"))
        for turn in range(6):
            await memory.remember(scope_key("a", "t", "self"), f"问{turn}", "答")
        assert episodes.rows[0]["owner"] == owner_key("a", "self")
        assert "owner" not in episodes.rows[1]
    run(scenario())


def test_runtime_recalls_on_the_first_turn_of_a_new_session_and_survives_recall_failure():
    async def scenario():
        runtime = await fixture()
        calls = []
        async def recall(scope, query, owner=None, exclude=""):
            calls.append((scope, owner, exclude))
            return [{"written_on":"2026-11-05", "text":"上次的摘要"}, {"written_on":"2026-11-05", "text":"更早的摘要"}]
        runtime.memory.recall_episodes = recall
        reply = await runtime.chat(Question(text="看医生前要准备什么？", user="alice", session="fresh"))
        assert calls == [(scope_key("alice", "fresh", "self"), owner_key("alice", "self"), "")]
        assert not reply.degraded and reply.recalled_summaries == ["2026-11-05"]
        recalled = [item for item in runtime.monitor.traces[-1]["events"] if item["event"] == "memory_recalled"]
        assert recalled == [{"event":"memory_recalled", "at_ms":recalled[0]["at_ms"], "count":2, "written_on":["2026-11-05"]}]
        assert "上次的摘要" not in str(runtime.monitor.traces[-1])
        # The model context carries the labelled summaries; the citations do not.
        assert all(reference.key for reference in reply.references)

        async def nothing(*args, **kwargs): return []
        runtime.memory.recall_episodes = nothing
        reply = await runtime.chat(Question(text="看医生前要准备什么？", user="alice", session="fresh"))
        assert reply.recalled_summaries == []

        async def broken(*args, **kwargs): raise ConnectionError("chroma_down")
        runtime.memory.recall_episodes = broken
        reply = await runtime.chat(Question(text="看医生前要准备什么？", user="alice", session="fresh"))
        assert reply.degraded and reply.storage == "degraded" and reply.recalled_summaries == []
    run(scenario())
