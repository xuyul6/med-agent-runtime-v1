"""Explicit CLI for bounded evaluation; refuses to overwrite previous results."""
import argparse
import asyncio
import json
from pathlib import Path
import httpx
from .smoke.common import require_developer_tools


async def run(url, output, request=None):
    target = Path(output)
    if target.exists(): raise FileExistsError("preserve_existing_evaluation_report")
    async with httpx.AsyncClient(timeout=620) as client:
        require_developer_tools((await client.get(url.rstrip("/")+"/health")).raise_for_status().json())
        response = await client.post(url.rstrip("/")+"/eval/run",json=request or {})
        response.raise_for_status()
        result = response.json()
        result["environment"] = (await client.get(url.rstrip("/")+"/health")).raise_for_status().json()
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(f'{result["passed"]}/{result["total"]} passed; {result["executed"]} executed; complete={result["complete"]}. Saved: {target}')
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",default="artifacts/evaluation.json")
    parser.add_argument("--cases",help="JSON EvaluationRequest file; omitted arrays use built-ins")
    parser.add_argument("--no-judge",action="store_true")
    parser.add_argument("--compare-to")
    parser.add_argument("--save-baseline")
    args = parser.parse_args()
    body = json.loads(Path(args.cases).read_text()) if args.cases else {}
    if args.no_judge: body["judge"] = False
    if args.compare_to or args.save_baseline:
        body["baseline"] = {**body.get("baseline",{}),**({"compare_to":args.compare_to} if args.compare_to else {}),
                            **({"save_as":args.save_baseline} if args.save_baseline else {})}
    asyncio.run(run(args.url,args.output,body))
