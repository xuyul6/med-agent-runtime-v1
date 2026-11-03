import asyncio
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Response
from redis.asyncio import Redis
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
from .appointments.resolution import booking_view
from .appointments.store import AppointmentError
from .tools.contracts import BookingView
from .core.config import ROOT, Settings
from .core.contracts import DocumentBatch, Question, Reference, Reply
from .benchmark.contracts import BenchmarkRequest
from .benchmark.runner import run_benchmark
from .comparison.contracts import ComparisonRequest
from .comparison.runner import run_comparison
from .evaluation.runner import evaluate
from .evaluation.isolation import ReferenceSnapshotError
from .evaluation.contracts import EvaluationRequest
from .evaluation.metrics import BaselineError, BaselineStore
from .memory import SessionMemory
from .core.provider import ClaudeGateway, fallback_gateway, judge_gateway
from .knowledge.retrieval import ReferenceIndex, Vectorizer, catalog_updates
from .runtime import MedicalRuntime
from .agents.skills import SkillValidationError


class ConfirmRequest(BaseModel):
    user: str = Field(min_length=1, max_length=100)
    draft_id: str = Field(min_length=1, max_length=64)
    token: str = Field(min_length=1, max_length=128)


class ConfirmResult(BaseModel):
    status: Literal["booked", "cancelled"]
    appointment: BookingView


CONFIRM_STATUS = {"draft_not_found":404, "appointment_not_found":404, "slot_not_found":404, "user_mismatch":403, "token_invalid":403,
                  "draft_expired":410, "slot_out_of_window":410, "draft_not_pending":409, "slot_full":409, "already_booked":409}


def create_app(settings: Settings | None=None, runtime: MedicalRuntime | None=None) -> FastAPI:
    settings = settings or Settings.load()

    @asynccontextmanager
    async def lifespan(app):
        if runtime is not None:
            app.state.runtime = runtime
            app.state.services = {"mode":"injected"}
            runtime.monitor.start()
            try:
                yield
            finally:
                await runtime.monitor.stop()
            return
        gateway = None if settings.preview else ClaudeGateway(settings)
        standby = None if settings.preview else fallback_gateway(settings)
        judge = None if settings.preview else judge_gateway(settings)
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
        missing = catalog_updates(seed, index.documents)
        if missing:
            await index.upsert(missing)
        memory = SessionMemory(redis, episodes, vectors, gateway, episode_ttl=settings.episode_ttl)
        memory.available = settings.preview or redis is not None
        try:
            await memory.purge_expired_episodes()
        except Exception:
            pass  # retried after the next summary write
        app.state.runtime = MedicalRuntime(settings, gateway, vectors, index, memory, fallback_model=standby, judge_model=judge)
        app.state.services = services
        app.state.runtime.monitor.start()
        try:
            yield
        finally:
            await app.state.runtime.monitor.stop()
            if gateway:
                await gateway.close()
            if standby:
                await standby.close()
            if judge:
                await judge.close()
            if redis:
                await redis.aclose()

    app = FastAPI(title="MedAgent Runtime", version="1.0.0", lifespan=lifespan)
    app.state.evaluation_lock = asyncio.Lock()
    app.state.baseline_store = BaselineStore()

    @app.get("/health")
    async def health():
        return {"status":"degraded" if "unavailable" in app.state.services.values() else "ok",
                "implementation":"independent-v1","preview":settings.preview,"model":settings.model,
                "standby_model":settings.fallback_model if settings.fallback_api_key and not settings.preview else None,
                "judge_model":(settings.judge_model if settings.fallback_api_key else "") or settings.model,
                "embedding":app.state.runtime.vectors.identity,"services":app.state.services}

    @app.post("/chat", response_model=Reply)
    async def chat(question: Question):
        return await app.state.runtime.chat(question)

    @app.post("/appointments/confirm", response_model=ConfirmResult)
    async def confirm_appointment(request: ConfirmRequest):
        # The only write into the mock hospital: a user's button press carrying the one-time token.
        try:
            booking = app.state.runtime.appointments.confirm(user=request.user, draft_id=request.draft_id, token=request.token)
        except AppointmentError as exc:
            raise HTTPException(CONFIRM_STATUS.get(exc.code, 409), exc.code) from None
        return ConfirmResult(status=booking.status, appointment=booking_view(booking))

    @app.get("/appointments", response_model=list[BookingView])
    async def appointments(user: str = Query(min_length=1, max_length=100)):
        return [booking_view(booking) for booking in app.state.runtime.appointments.upcoming(user)]

    @app.get("/runtime")
    @app.get("/monitor")
    async def status():
        return app.state.runtime.monitor.snapshot()

    @app.get("/agents")
    async def agent_profiles():
        return {"agents":[profile.describe() for profile in app.state.runtime.registry.profiles.values()]}

    @app.get("/tools")
    async def tool_catalog():
        return {"tools":[spec.describe() for spec in app.state.runtime.tools.registry.specs.values()]}

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
        return app.state.runtime.skills.snapshot().describe()

    @app.post("/skills/reload")
    async def reload_skills():
        try:
            return await asyncio.to_thread(app.state.runtime.skills.reload)
        except SkillValidationError as exc:
            raise HTTPException(422, {"code":"invalid_skills", "reason":str(exc),
                                      "active_snapshot_id":app.state.runtime.skills.snapshot().snapshot_id}) from exc

    @app.post("/eval/run")
    async def run_evaluation(request: EvaluationRequest | None = None):
        request = request or EvaluationRequest()
        if settings.preview and request.judge:
            raise HTTPException(409,"online_evaluation_requires_model")
        if app.state.evaluation_lock.locked(): raise HTTPException(409,"evaluation_already_running")
        async with app.state.evaluation_lock:
            try:
                return await evaluate(app.state.runtime,request,baseline_store=app.state.baseline_store)
            except ReferenceSnapshotError as exc:
                raise HTTPException(409,{"code":"reference_snapshot_rejected","reason":str(exc)}) from exc
            except BaselineError as exc:
                raise HTTPException(409,{"code":"baseline_rejected","reason":str(exc)}) from exc
            except ValidationError as exc:
                raise HTTPException(422,{"code":"invalid_evaluation_cases",
                    "errors":exc.errors(include_input=False,include_context=False,include_url=False)}) from exc
            except ValueError as exc:
                raise HTTPException(422,{"code":"invalid_evaluation_cases","reason":type(exc).__name__}) from exc

    @app.post("/benchmark/run")
    async def run_context_benchmark(request: BenchmarkRequest | None = None):
        request = request or BenchmarkRequest()
        if settings.preview and request.judge:
            raise HTTPException(409,"online_benchmark_requires_model")
        if app.state.evaluation_lock.locked(): raise HTTPException(409,"evaluation_already_running")
        async with app.state.evaluation_lock:
            return await run_benchmark(app.state.runtime,request)

    @app.post("/comparison/run")
    async def run_architecture_comparison(request: ComparisonRequest | None = None):
        request = request or ComparisonRequest()
        if settings.preview and request.judge:
            raise HTTPException(409,"online_comparison_requires_model")
        if app.state.evaluation_lock.locked(): raise HTTPException(409,"evaluation_already_running")
        async with app.state.evaluation_lock:
            return await run_comparison(app.state.runtime,request)

    @app.get("/metrics")
    async def metrics():
        return Response(app.state.runtime.monitor.prometheus(),
                        media_type="text/plain; version=0.0.4")

    return app


app = create_app()
