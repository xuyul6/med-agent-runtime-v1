import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Response
from redis.asyncio import Redis
from .config import ROOT, Settings
from .contracts import DocumentBatch, Question, Reference, Reply
from .evaluation import evaluate
from .memory import SessionMemory
from .provider import ClaudeGateway
from .retrieval import ReferenceIndex, Vectorizer
from .runtime import MedicalRuntime


def create_app(settings: Settings | None=None, runtime: MedicalRuntime | None=None) -> FastAPI:
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app):
        if runtime is not None:
            app.state.runtime = runtime
            app.state.services = {"mode":"injected"}
            yield
            return
        gateway = None if settings.preview else ClaudeGateway(settings)
        vectors = Vectorizer("hash" if settings.preview else settings.embeddings)
        redis, client, collection, episodes = None, None, None, None
        services = {"redis":"preview","chroma":"preview"}
        if not settings.preview:
            try:
                redis = Redis.from_url(settings.redis_url, decode_responses=True, socket_timeout=3, socket_connect_timeout=3)
                await redis.ping()
                services["redis"] = "available"
            except Exception:
                if redis:
                    await redis.aclose()
                redis = None
                services["redis"] = "unavailable"
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                client = await asyncio.to_thread(chromadb.HttpClient, host=settings.chroma_host, port=settings.chroma_port,
                                                settings=ChromaSettings(anonymized_telemetry=False))
                collection = await asyncio.to_thread(client.get_or_create_collection, "mar2_refs_"+vectors.identity)
                episodes = await asyncio.to_thread(client.get_or_create_collection, "mar2_episodes_"+vectors.identity)
                services["chroma"] = "available"
            except Exception:
                services["chroma"] = "unavailable"
        index = ReferenceIndex(vectors, gateway, collection)
        await index.restore()
        seed = [Reference.model_validate(record) for record in json.loads((ROOT/"resources"/"references.json").read_text())]
        missing = [record for record in seed if record.key not in index.documents]
        # Same-version catalog metadata may gain aliases without replacing edited content.
        missing += [index.documents[record.key].model_copy(update={"entities":record.entities}) for record in seed
                    if record.key in index.documents and record.entities and not index.documents[record.key].entities
                    and record.version == index.documents[record.key].version]
        if missing:
            await index.upsert(missing)
        memory = SessionMemory(redis, episodes, vectors, gateway)
        memory.available = settings.preview or redis is not None
        app.state.runtime = MedicalRuntime(settings, gateway, vectors, index, memory)
        app.state.services = services
        yield
        if gateway:
            await gateway.close()
        if redis:
            await redis.aclose()

    app = FastAPI(title="MedAgent Runtime", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status":"degraded" if "unavailable" in app.state.services.values() else "ok",
                "implementation":"independent-v1","preview":settings.preview,"model":settings.model,
                "embedding":app.state.runtime.router.vectors.identity,"services":app.state.services}

    @app.post("/chat", response_model=Reply)
    async def chat(question: Question):
        return await app.state.runtime.chat(question)

    @app.get("/runtime")
    async def status():
        return app.state.runtime.monitor.snapshot()

    @app.get("/traces/{request_id}")
    async def trace(request_id: str):
        found = next((trace for trace in app.state.runtime.monitor.traces if trace["request_id"] == request_id), None)
        if found is None:
            raise HTTPException(404, "trace_not_found")
        return found

    @app.get("/knowledge")
    async def knowledge():
        return {"documents":[r.model_dump(mode="json") for r in app.state.runtime.index.documents.values()]}

    @app.post("/knowledge")
    async def add_knowledge(batch: DocumentBatch):
        if settings.preview:
            raise HTTPException(409, "preview_is_read_only")
        await app.state.runtime.index.upsert(batch.documents)
        return {"saved":len(batch.documents)}

    @app.get("/skills")
    async def skills():
        return {"roles":sorted(app.state.runtime.skills.content)}

    @app.post("/skills/reload")
    async def reload_skills():
        return app.state.runtime.skills.reload()

    @app.post("/eval/run")
    async def run_evaluation():
        if settings.preview:
            raise HTTPException(409,"online_evaluation_requires_model")
        return await evaluate(app.state.runtime)

    @app.get("/metrics")
    async def metrics():
        snapshot = app.state.runtime.monitor.snapshot()
        return Response(f'mar_requests_total {snapshot["requests"]}\nmar_degraded_total {snapshot["degraded"]}\n',
                        media_type="text/plain; version=0.0.4")

    return app


app = create_app()
