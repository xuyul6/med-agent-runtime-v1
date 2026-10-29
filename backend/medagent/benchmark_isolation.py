"""Read-only run snapshot; independently restored conversation branches."""
import asyncio
from copy import deepcopy
from dataclasses import dataclass
import json
import time
from .evaluation_budget import BudgetedModel
from .evaluation_isolation import EvaluationIsolation, NamespacedRedis, SnapshotSkills
from .evaluation_metrics import digest
from .memory import SessionMemory, scope_key
from .retrieval import ReferenceIndex, Vectorizer
from .runtime import MedicalRuntime
from .tools import ToolRegistry


@dataclass(frozen=True)
class ConversationSnapshot:
    user: str
    session: str
    subject: str
    state_json: str
    episodes_json: str
    risk_seconds: float
    transcript_json: str = "[]"
    context_time: float = 0.0

    @property
    def payload(self):
        return {"state":json.loads(self.state_json), "episodes":json.loads(self.episodes_json),
                "subject":self.subject, "recent_risk":self.risk_seconds > 0,
                "original_transcript":json.loads(self.transcript_json), "context_time":self.context_time}

    @property
    def sha256(self): return digest(self.payload)

    @property
    def history(self): return json.loads(self.transcript_json)


class PairedIsolation(EvaluationIsolation):
    def __init__(self, source, run_id, budget):
        super().__init__(source,run_id,budget)
        self.settings, self.vector_mode = source.settings, source.router.vectors.mode
        self.memory_available, self.serial = source.memory.available, 0

    async def branch(self, phase, snapshot=None, cache=()):
        self.serial += 1
        name = f"bench-episodes-{self.run_id}-{self.serial}"
        self.created.append(name)
        episodes = await asyncio.to_thread(self.client.create_collection,name,embedding_function=None)
        vectors = Vectorizer(self.vector_mode)
        model = BudgetedModel(self.base_model,self.budget,phase) if self.base_model else None
        index = ReferenceIndex(vectors,model,self.index.collection)
        index.documents = {ref.key:ref.model_copy(deep=True) for ref in self.records}
        for key, identifiers in cache:
            index.cache[key] = (time.monotonic()+90,[index.documents[identifier] for identifier in identifiers])
        redis = NamespacedRedis(self.redis,f"branch-{self.serial}") if self.redis is not None else None
        memory = SessionMemory(redis,episodes,vectors,model)
        if snapshot and snapshot.context_time:
            memory.context_clock = lambda: snapshot.context_time
        memory.available = self.memory_available
        runtime = MedicalRuntime(self.settings,model,vectors,index,memory,profiles=self.profiles,skills=SnapshotSkills(self.skills))
        runtime.tools.registry = runtime.planner.tools = ToolRegistry(self.specs)
        self.runtime = runtime
        if snapshot:
            state = json.loads(snapshot.state_json)
            scope = scope_key(snapshot.user,snapshot.session,snapshot.subject)
            if redis is None: memory.local[scope] = deepcopy(state)
            else: await redis.set("mar2:session:"+scope,json.dumps(state,ensure_ascii=False),ex=3600)
            rows = json.loads(snapshot.episodes_json)
            if rows:
                await asyncio.to_thread(episodes.add,ids=[row["id"] for row in rows],
                    documents=[row["document"] for row in rows],embeddings=[row["embedding"] for row in rows],
                    metadatas=[{"scope":scope} for row in rows])
            if snapshot.risk_seconds > 0:
                runtime.risk.hints[scope] = runtime.risk.clock()+snapshot.risk_seconds
        return runtime

    async def capture(self, runtime, user, session, subject, transcript=None):
        scope = scope_key(user,session,subject)
        state = deepcopy(await runtime.memory.read(scope))
        rows = await asyncio.to_thread(runtime.memory.episodes.get,where={"scope":scope},include=["documents","embeddings"])
        episodes = sorted([{"id":identifier,"document":document,"embedding":[float(value) for value in embedding]}
            for identifier,document,embedding in zip(rows["ids"],rows["documents"],rows["embeddings"] if rows["ids"] else [])],key=lambda row:row["id"])
        return ConversationSnapshot(user,session,subject,json.dumps(state,ensure_ascii=False,sort_keys=True),
            json.dumps(episodes,ensure_ascii=False,sort_keys=True),max(0,runtime.risk.hints.get(scope,0)-runtime.risk.clock()),
            json.dumps(state["messages"] if transcript is None else transcript,ensure_ascii=False,sort_keys=True),
            runtime.memory.context_clock())

    def cache_snapshot(self, runtime):
        return tuple((key,tuple(ref.key for ref in refs)) for key,(expires,refs) in sorted(runtime.index.cache.items())
                     if expires > time.monotonic() and all(ref == runtime.index.documents.get(ref.key) for ref in refs))
