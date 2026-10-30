"""Independent runtime state with namespaced Redis and disposable Chroma collections."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import tempfile
from .budget import BudgetedModel
from ..memory import SessionMemory
from ..core.observability import event
from ..knowledge.retrieval import ReferenceIndex, Vectorizer
from ..runtime import MedicalRuntime
from ..tools.registry import ToolRegistry


class ReferenceSnapshotError(ValueError):
    pass


class NamespacedRedis:
    def __init__(self, parent, identity):
        self.parent, self.prefix, self.written = parent,"mar2:eval:"+identity+":",set()

    async def get(self, key):
        return await self.parent.get(self.prefix+key)

    async def set(self, key, value, ex):
        full = self.prefix+key
        self.written.add(full)
        return await self.parent.set(full,value,ex=min(ex,3600))

    async def cleanup(self):
        if self.written: await self.parent.delete(*sorted(self.written))


class SnapshotSkills:
    def __init__(self, snapshot): self.fixed = snapshot
    def snapshot(self): return self.fixed


class EvaluationIsolation:
    def __init__(self, source, run_id, budget):
        self.source,self.run_id,self.budget = source,run_id,budget
        self.records = [ref.model_copy(deep=True) for ref in source.index.documents.values()]
        self.skills = source.skills.snapshot()
        self.profiles = list(source.registry.profiles.values())
        self.specs = list(source.tools.registry.specs.values())
        self.base_model = source.model
        self.redis,self.client,self.runtime = None,None,None
        self.directory,self.creating = None,None
        self.created, self.cleanup_errors = [],[]

    async def create(self):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        # Never reset a client or touch the production collection; names are generated internally.
        self.directory = tempfile.TemporaryDirectory(prefix="medagent-eval-")
        self.creating = asyncio.create_task(asyncio.to_thread(chromadb.PersistentClient,path=self.directory.name,
            settings=ChromaSettings(anonymized_telemetry=False)))
        self.client = await asyncio.shield(self.creating)
        refs_name,episodes_name = "eval-refs-"+self.run_id,"eval-episodes-"+self.run_id
        self.created.append(refs_name)
        refs = await asyncio.to_thread(self.client.create_collection,refs_name,embedding_function=None)
        self.created.append(episodes_name)
        episodes = await asyncio.to_thread(self.client.create_collection,episodes_name,embedding_function=None)
        vectors = Vectorizer(self.source.vectors.mode)
        model = BudgetedModel(self.base_model,self.budget) if self.base_model else None
        index = ReferenceIndex(vectors,model,refs)
        await index.upsert(self.records)
        if self.source.memory.redis is not None:
            self.redis = NamespacedRedis(self.source.memory.redis,self.run_id)
        memory = SessionMemory(self.redis,episodes,vectors,model)
        memory.available = self.source.memory.available
        self.model,self.vectors,self.index,self.memory = model,vectors,index,memory
        return self.new_case()

    def new_case(self):
        self.index.cache.clear()
        self.runtime = MedicalRuntime(self.source.settings,self.model,self.vectors,self.index,self.memory,
            profiles=self.profiles,skills=SnapshotSkills(self.skills))
        self.runtime.tools.registry = ToolRegistry(self.specs)
        self.runtime.planner.tools = self.runtime.tools.registry
        return self.runtime

    def select_case_references(self, keys):
        """Apply the preflight-validated frozen view to both lexical and Chroma recall."""
        selected = set(keys) if keys is not None else None
        self.index.documents = {ref.key:ref.model_copy(deep=True) for ref in self.records
                                if selected is None or ref.key in selected}
        self.index.reference_keys_filter = selected
        self.index.cache.clear()

    async def cleanup(self):
        if self.client is None and self.creating is not None:
            try: self.client = await asyncio.shield(self.creating)
            except Exception as exc: self.cleanup_errors.append("chroma_initialization:"+type(exc).__name__)
        if self.redis:
            try: await self.redis.cleanup()
            except Exception as exc: self.cleanup_errors.append("redis:"+type(exc).__name__)
        if self.runtime: await self.runtime.monitor.stop()
        if self.client:
            for name in self.created:
                try: await asyncio.to_thread(self.client.delete_collection,name)
                except Exception as exc: self.cleanup_errors.append("chroma:"+type(exc).__name__)
            # Chroma 0.5.23 has no public close(). Only release the system belonging to our unique directory.
            from chromadb.api.shared_system_client import SharedSystemClient
            identifier = self.client._identifier
            if self.directory and identifier == self.directory.name:
                system = SharedSystemClient._identifier_to_system.get(identifier)
                if system is not None:
                    try:
                        await asyncio.to_thread(system.stop)
                        if SharedSystemClient._identifier_to_system.get(identifier) is system:
                            SharedSystemClient._identifier_to_system.pop(identifier)
                    except Exception as exc: self.cleanup_errors.append("chroma_close:"+type(exc).__name__)
        if self.directory:
            try: self.directory.cleanup()
            except Exception as exc: self.cleanup_errors.append("temporary_directory:"+type(exc).__name__)


class FaultModel:
    def __init__(self, base, kinds): self.base,self.kinds = base,kinds
    async def json(self, stage, *args, **kwargs):
        fault = {"synthesis":"synthesis_failure", "memory_summary":"summary_failure",
                 "understanding":"language_unavailable"}.get(stage)
        if fault in self.kinds:
            event("synthetic_fault_triggered", fault=fault, stage=stage)
            raise RuntimeError("synthetic_model_failure")
        return await self.base.json(stage,*args,**kwargs)
    async def message(self, stage, *args, **kwargs):
        if stage.startswith("agent_") and "agent_invalid_output" in self.kinds:
            event("synthetic_fault_triggered", fault="agent_invalid_output", role=stage.removeprefix("agent_"))
            return {"content":[{"type":"text","text":json.dumps({"role":"invalid-role"})}]}
        return await self.base.message(stage,*args,**kwargs)


@asynccontextmanager
async def injected_faults(runtime, faults, turn):
    kinds = {fault.kind for fault in faults if fault.turn is None or fault.turn == turn}
    available,collection,registry = runtime.memory.available,runtime.index.collection,runtime.tools.registry
    base_model,agent_models = runtime.model,[(agent,agent.model) for agents in runtime.agents.values() for agent in agents]
    memory_model,index_model,router_model = runtime.memory.model,runtime.index.model,runtime.router.model
    try:
        if "memory_unavailable" in kinds: runtime.memory.available = False
        if "index_unavailable" in kinds: runtime.index.collection = None
        if "tool_timeout" in kinds or "tool_invalid_result" in kinds:
            async def failed_lookup(context,args):
                fault = "tool_timeout" if "tool_timeout" in kinds else "tool_invalid_result"
                event("synthetic_fault_triggered", fault=fault, role=context.profile.role, stage="lookup_reference")
                if fault == "tool_timeout": raise TimeoutError("synthetic_tool_timeout")
                return {"documents":[{"invalid":"synthetic"}]}
            runtime.tools.registry = ToolRegistry([replace(spec,handler=failed_lookup) if name == "lookup_reference" else spec
                                                   for name,spec in registry.specs.items()])
            runtime.planner.tools = runtime.tools.registry
        if base_model is not None:
            wrapper = FaultModel(base_model,kinds)
            runtime.model = runtime.memory.model = runtime.index.model = runtime.router.model = wrapper
            for agent,_ in agent_models: agent.model = wrapper
        yield kinds
    finally:
        runtime.memory.available,runtime.index.collection = available,collection
        runtime.tools.registry = runtime.planner.tools = registry
        runtime.model, runtime.memory.model, runtime.index.model,runtime.router.model = base_model,memory_model,index_model,router_model
        for agent,model in agent_models: agent.model = model
