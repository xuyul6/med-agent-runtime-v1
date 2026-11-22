"""Review the architecture comparison plan; only --execute calls the isolated API and may incur model charges."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import httpx
from ..benchmark.cli import write_report
from ..evaluation.metrics import digest
from .contracts import LAYERS, ComparisonRequest, daily_reference_snapshot, protocol_for, resolve_comparison
from .runner import schedule


def build_plan(request: ComparisonRequest) -> dict:
    version, cases = resolve_comparison(request)
    pairs = schedule(request, cases)
    body = request.model_dump(mode="json", exclude_unset=True)
    if request.layer == "daily" and request.cases is None:
        body["daily_version"] = version
    elif request.cases is None and not request.multi_vs_single:
        body["cases"] = [case.model_dump(mode="json", exclude_unset=True) for case in cases]
    protocol = protocol_for(request.layer, multi_vs_single=request.multi_vs_single)
    return {"schema_version":"architecture-comparison-plan-1", "online_status":"not_run", "dataset":version,
            "cases_hash":digest([case.model_dump(mode="json") for case in cases]), "reference_snapshot":daily_reference_snapshot(version),
            "protocol":deepcopy(protocol), "protocol_hash":digest(protocol), "request":body,
            "cases_per_category":{category:sum(case.category == category for case in cases) for category in {c.category for c in cases}},
            "planned_pairs":len(pairs), "planned_targets":len(request.arm_names)*len(pairs), "schedule":pairs,
            "notice":"Only --execute sends requests; no arm is assumed to be better."}


async def compare(url: str, output: str, request: ComparisonRequest) -> dict:
    plan = build_plan(request)
    async with httpx.AsyncClient(timeout=plan["request"]["limits"]["timeout_seconds"]+30) as client:
        response = await client.post(url.rstrip("/")+"/comparison/run", json=plan["request"])
        response.raise_for_status()
        report = response.json()
    if report.get("metadata", {}).get("protocol_hash") != plan["protocol_hash"]:
        report["conclusion"] = "not_established"
        report.setdefault("claim_blockers", []).append("client_server_protocol_mismatch")
    if request.multi_vs_single and report.get("metadata", {}).get("cases_hash") != plan["cases_hash"]:
        report["conclusion"] = "not_established"
        report.setdefault("claim_blockers", []).append("client_server_cases_mismatch")
    if plan["reference_snapshot"] and (
        report["metadata"].get("cases_hash") != plan["cases_hash"]
        or report["metadata"].get("references_hash") != plan["reference_snapshot"]["sha256"]
    ):
        report["conclusion"] = "not_established"
        report.setdefault("claim_blockers", []).append("client_server_daily_snapshot_mismatch")
    write_report(output, report)
    print(f'complete={report["complete"]}; conclusion={report["conclusion"]}. Saved: {output}')
    return report


def main():
    parser = argparse.ArgumentParser(description="Multi-agent / primary-only / single-agent comparison")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", help="JSON ComparisonRequest reviewed before execution")
    parser.add_argument("--execute", action="store_true", help="Run against /comparison/run; may incur model charges")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--layer", choices=sorted(LAYERS))
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    body = json.loads(Path(args.cases).read_text()) if args.cases else {}
    for name in ("layer", "repetitions"):
        value = getattr(args, name)
        if value is None: continue
        if body.get(name) not in (None, value): raise SystemExit(f"--{name} conflicts with the reviewed request file")
        body[name] = value
    if args.no_judge: body["judge"] = False
    request = ComparisonRequest.model_validate(body)
    if Path(args.output).exists(): raise FileExistsError("preserve_existing_comparison_report")
    if args.execute:
        asyncio.run(compare(args.url, args.output, request))
    else:
        plan = build_plan(request)
        write_report(args.output, plan)
        print(f'Plan only: {plan["planned_pairs"]} pairs / {plan["planned_targets"]} targets; no network or model calls. Saved: {args.output}')
