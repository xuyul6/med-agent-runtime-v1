import asyncio
import hashlib
import json
import time
import uuid
import weakref
from .core.observability import event

_UNCHANGED = object()
EPISODE_TTL_SECONDS = 7*86400
PURGE_INTERVAL_SECONDS = 3600

def scope_key(user: str, session: str, subject: str) -> str:
    return hashlib.sha256(json.dumps([user, session, subject], ensure_ascii=False).encode()).hexdigest()


class SessionMemory:
    def __init__(self, redis=None, episodes=None, vectors=None, model=None, episode_ttl: float = EPISODE_TTL_SECONDS):
        if not episode_ttl > 0:
            raise ValueError("invalid_episode_ttl")
        self.redis, self.episodes, self.vectors, self.model = redis, episodes, vectors, model
        self.episode_ttl, self.purged_at = episode_ttl, None
        self.local = {}
        self.locks = weakref.WeakValueDictionary()
        self.available = True
        self.context_clock = time.time

    async def read(self, scope: str) -> dict:
        if self.redis is None:
            return dict(self.local.get(scope, {"messages": [], "summary": ""}))
        raw = await self.redis.get("mar2:session:"+scope)
        state = json.loads(raw) if raw else {"messages": [], "summary": ""}
        if not isinstance(state, dict) or not isinstance(state.get("messages"), list) or not isinstance(state.get("summary"), str):
            raise ValueError("invalid_session_state")
        if not isinstance(state.get("entities", {}), dict) or type(state.get("turn_counter", 0)) is not int or state.get("turn_counter", 0) < 0:
            raise ValueError("invalid_session_metadata")
        if not isinstance(state.get("last_intent", ""), str) or not isinstance(state.get("last_missing_fields", []), list):
            raise ValueError("invalid_session_context")
        if any(not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}
               or not isinstance(message.get("content"), str) for message in state["messages"]):
            raise ValueError("invalid_session_messages")
        if any("turn_index" in message and (type(message["turn_index"]) is not int or message["turn_index"] < 0)
               or "turn_id" in message and not isinstance(message["turn_id"], str) for message in state["messages"]):
            raise ValueError("invalid_session_turn_metadata")
        return state

    async def remember(self, scope: str, user: str, assistant: str, understanding=None,
                       pending_consultation=_UNCHANGED, expected_pending_turn=_UNCHANGED) -> None:
        async with self.locks.setdefault(scope, asyncio.Lock()):
            state = await self.read(scope)
            pending_current = expected_pending_turn is _UNCHANGED or state.get("last_intent_turn_id") == expected_pending_turn
            turn_id = understanding.turn_id if understanding is not None else str(uuid.uuid4())
            turn_index = state.get("turn_counter", len(state["messages"])//2)+1
            state["turn_counter"] = turn_index
            state["last_user_at"] = self.context_clock()
            state["last_intent_turn_id"] = turn_id if understanding is not None else None
            state["messages"] = state["messages"] + [{"role":"user", "content":user, "turn_id":turn_id, "turn_index":turn_index},
                                                        {"role":"assistant", "content":assistant, "turn_id":turn_id, "turn_index":turn_index}]
            if understanding is not None:
                from .understanding.conversation import merge_fields
                updates = [fact.model_copy(update={"source":fact.source.model_copy(update={"turn_index":turn_index})})
                           if fact.source.turn_id == turn_id else fact for fact in understanding.updates]
                fields = merge_fields(state.get("entities", {}), updates, understanding.subject)
                state["entities"] = {name:fact.model_dump(mode="json") for name,fact in fields.items()}
                state["last_intent"] = understanding.primary.value
                state["last_missing_fields"] = understanding.missing_fields
            if pending_consultation is not _UNCHANGED and pending_current:
                if pending_consultation is None:
                    state.pop("pending_consultation", None)
                else:
                    state["pending_consultation"] = {**pending_consultation, "last_turn_id":turn_id}
            if len(state["messages"]) >= 12 and self.model:
                older = state["messages"][:-4]
                result = await self.model.json("memory_summary", '将历史压缩为用户已提供的信息、未解决的问题及咨询对象。不增加病情判断。返回{"summary":"不超过300字"}。',
                    json.dumps({"previous": state["summary"], "messages": older}, ensure_ascii=False), 750)
                summary = result.get("summary")
                if not isinstance(summary, str) or not summary.strip():
                    raise ValueError("empty_memory_summary")
                state["summary"] = summary[:1200]
                state["messages"] = state["messages"][-4:]
                if self.episodes is not None:
                    encoded = await self.vectors.encode([state["summary"]])
                    await asyncio.to_thread(self.episodes.add, ids=[str(uuid.uuid4())],
                        documents=[state["summary"]], embeddings=encoded, metadatas=[self.episode_metadata(scope)])
                    await self._purge_if_due()
                event("memory_compressed", kept=len(state["messages"]))
            if self.redis is None:
                self.local[scope] = state
                if len(self.local) > 128:
                    self.local.pop(next(iter(self.local)))
            else:
                await self.redis.set("mar2:session:"+scope, json.dumps(state, ensure_ascii=False), ex=86400)

    def episode_metadata(self, scope: str) -> dict:
        return {"scope": scope, "created_at": float(self.context_clock())}

    async def recall(self, scope: str, query: str) -> list[str]:
        if self.episodes is None:
            return []
        count = await asyncio.to_thread(self.episodes.count)
        if not count:
            return []
        encoded = await self.vectors.encode([query])
        cutoff = self.context_clock()-self.episode_ttl
        result = await asyncio.to_thread(self.episodes.query, query_embeddings=encoded, n_results=min(2,count),
                                         where={"$and":[{"scope":scope}, {"created_at":{"$gte":cutoff}}]})
        return result["documents"][0]

    async def purge_expired_episodes(self) -> int:
        """Delete summaries past the retention window, including older ones stored without a creation time."""
        if self.episodes is None:
            return 0
        now = self.context_clock()
        rows = await asyncio.to_thread(self.episodes.get, include=["metadatas"])
        expired = [identifier for identifier, metadata in zip(rows["ids"], rows.get("metadatas") or [])
                   if not isinstance((metadata or {}).get("created_at"), (int, float))
                   or metadata["created_at"] < now-self.episode_ttl]
        if expired:
            await asyncio.to_thread(self.episodes.delete, ids=expired)
            event("memory_episodes_purged", count=len(expired))
        self.purged_at = now
        return len(expired)

    async def _purge_if_due(self) -> None:
        # Cleanup is housekeeping: a failure is recorded but never fails the conversation turn.
        if self.purged_at is not None and self.context_clock()-self.purged_at < PURGE_INTERVAL_SECONDS:
            return
        try:
            await self.purge_expired_episodes()
        except Exception as exc:
            event("memory_error", stage="purge", error=type(exc).__name__)
