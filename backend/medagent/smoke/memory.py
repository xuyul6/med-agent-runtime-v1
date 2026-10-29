"""Synthetic real-storage smoke test; never use actual health records."""
import argparse
import asyncio
import json
import uuid
from pathlib import Path
import httpx
from redis.asyncio import Redis
from ..core.config import Settings
from ..memory import SessionMemory, scope_key
from ..knowledge.retrieval import Vectorizer


async def verify(url, output):
    settings = Settings.load()
    user, session = "synthetic-memory-"+str(uuid.uuid4()), str(uuid.uuid4())
    scope = scope_key(user,session,"self")
    redis = Redis.from_url(settings.redis_url,decode_responses=True)
    messages = []
    for index in range(5):
        messages += [{"role":"user","content":f"虚构准备记录{index}：我在整理就医资料。"},
                     {"role":"assistant","content":"可以整理已有资料和要向医生确认的问题。"}]
    try:
        await redis.set("mar2:session:"+scope,json.dumps({"messages":messages,"summary":""},ensure_ascii=False),ex=1800)
        async with httpx.AsyncClient(timeout=150) as http:
            result = await http.post(url.rstrip("/")+"/chat",json={"text":"看医生前还要记录哪些信息？","user":user,"session":session})
            result.raise_for_status()
            reply = result.json()
            trace = (await http.get(url.rstrip("/")+"/traces/"+reply["request_id"])).json()
        state = json.loads(await redis.get("mar2:session:"+scope))
        import chromadb
        from chromadb.config import Settings as CS
        client = chromadb.HttpClient(host=settings.chroma_host,port=settings.chroma_port,settings=CS(anonymized_telemetry=False))
        vectors = Vectorizer(settings.embeddings)
        collection = client.get_collection("mar2_episodes_"+vectors.identity)
        recalled = await SessionMemory(redis,collection,vectors).recall(scope,"就医准备")
        checks = {"runtime_healthy":not reply["degraded"] and reply["storage"]=="available",
                  "recent_messages":len(state["messages"])==4,"summary_present":bool(state["summary"]),
                  "episodic_recall":bool(recalled),"summary_model_call":any(event.get("stage")=="memory_summary" for event in trace["events"])}
        report={"implementation":"independent-v1","synthetic_history_messages":10,"checks":checks,"reply":reply,"trace":trace}
        target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps(checks));print(target)
        if not all(checks.values()):raise RuntimeError("memory_smoke_failed")
    finally:
        await redis.aclose()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",default="artifacts/memory-check.json")
    args=parser.parse_args()
    asyncio.run(verify(args.url,args.output))
