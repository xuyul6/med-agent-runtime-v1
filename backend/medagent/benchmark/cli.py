"""Review a bounded paired protocol; --execute explicitly calls the isolated API."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import httpx
from .contracts import BenchmarkRequest, PROTOCOL, resolve_benchmark
from .runner import schedule
from ..evaluation.metrics import digest

def build_plan(request: BenchmarkRequest):
    version,cases = resolve_benchmark(request)
    pairs = schedule(request,cases)
    body = request.model_dump(mode="json")
    body["cases"] = [case.model_dump(mode="json") for case in cases]
    return {"schema_version":"paired-benchmark-plan-1","online_status":"not_run","dataset":version,
            "protocol":deepcopy(PROTOCOL),"protocol_hash":digest(PROTOCOL),"request":body,
            "planned_pairs":len(pairs),"planned_targets":2*len(pairs),"schedule":pairs,
            "notice":"Only --execute sends requests; no fixed percentage or clinical equivalence is assumed."}


def write_report(output, report):
    target=Path(output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


async def compare(url, output, request=None):
    if Path(output).exists(): raise FileExistsError("preserve_existing_benchmark_report")
    plan = build_plan(request or BenchmarkRequest())
    async with httpx.AsyncClient(timeout=plan["request"]["limits"]["timeout_seconds"]+30) as client:
        response = await client.post(url.rstrip("/")+"/benchmark/run",json=plan["request"])
        response.raise_for_status()
        report = response.json()
    report["client_plan"] = {key:value for key,value in plan.items() if key not in {"request","schedule","online_status"}}
    metadata = report.get("metadata",{})
    mismatches = [name for name,actual,expected in [
        ("protocol_hash",metadata.get("protocol_hash"),plan["protocol_hash"]),
        ("request_hash",metadata.get("request_hash"),digest(plan["request"]))] if actual != expected]
    report["client_plan_mismatch"] = mismatches
    if mismatches:
        report["quality_noninferiority"] = "not_established"
        report["supported_input_reduction"] = {cache:None for cache in plan["request"]["cache_conditions"]}
        report.setdefault("claim_blockers",[]).append("client_server_protocol_or_request_mismatch")
    write_report(output,report)
    print(f'complete={report["complete"]}; noninferiority={report["quality_noninferiority"]}. Saved: {output}')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",default="artifacts/context.json")
    parser.add_argument("--cases",help="JSON BenchmarkRequest; protocol limits and selected cases are reviewed before execution")
    parser.add_argument("--execute",action="store_true",help="Run against /benchmark/run; may incur model charges")
    parser.add_argument("--no-judge",action="store_true",help="Disable Judge only; runtime models still run with --execute")
    args=parser.parse_args()
    body = json.loads(Path(args.cases).read_text()) if args.cases else {}
    if args.no_judge: body["judge"] = False
    request = BenchmarkRequest.model_validate(body)
    if Path(args.output).exists(): raise FileExistsError("preserve_existing_benchmark_report")
    if args.execute:
        asyncio.run(compare(args.url,args.output,request))
    else:
        plan = build_plan(request)
        write_report(args.output,plan)
        print(f'Plan only: {plan["planned_pairs"]} pairs / {plan["planned_targets"]} targets; no network or model calls. Saved: {args.output}')


if __name__ == "__main__":
    main()
