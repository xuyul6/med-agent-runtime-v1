import asyncio
import hashlib
import json
import uuid
import weakref
from .observability import event


def scope_key(user: str, session: str, subject: str) -> str:
    return hashlib.sha256(json.dumps([user, session, subject], ensure_ascii=False).encode()).hexdigest()


class SessionMemory:
    def __init__(self, redis=None, episodes=None, vectors=None, model=None):
        self.redis, self.episodes, self.vectors, self.model = redis, episodes, vectors, model
        self.local = {}
        self.locks = weakref.WeakValueDictionary()
        self.available = True

    async def read(self, scope: str) -> dict:
        if self.redis is None:
            return dict(self.local.get(scope, {"messages": [], "summary": ""}))
        raw = await self.redis.get("mar2:session:"+scope)
        return json.loads(raw) if raw else {"messages": [], "summary": ""}

    async def remember(self, scope: str, user: str, assistant: str) -> None:
        async with self.locks.setdefault(scope, asyncio.Lock()):
            state = await self.read(scope)
            state["messages"] = state["messages"] + [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
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
                        documents=[state["summary"]], embeddings=encoded, metadatas=[{"scope": scope}])
                event("memory_compressed", kept=len(state["messages"]))
            if self.redis is None:
                self.local[scope] = state
                if len(self.local) > 128:
                    self.local.pop(next(iter(self.local)))
            else:
                await self.redis.set("mar2:session:"+scope, json.dumps(state, ensure_ascii=False), ex=86400)

    async def recall(self, scope: str, query: str) -> list[str]:
        if self.episodes is None:
            return []
        count = await asyncio.to_thread(self.episodes.count)
        if not count:
            return []
        encoded = await self.vectors.encode([query])
        result = await asyncio.to_thread(self.episodes.query, query_embeddings=encoded, n_results=min(2,count), where={"scope":scope})
        return result["documents"][0]
