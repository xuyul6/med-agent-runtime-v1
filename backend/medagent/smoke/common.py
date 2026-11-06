"""Shared loop for online smoke checks: healthy runtime, /chat plus trace, incremental report, early stop."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import httpx

STOP_AFTER_FAILURES = 2
DEVELOPER_TOOLS_REQUIRED = "developer_tools_required: start the backend with MAR_DEVELOPER_TOOLS=true"


def require_developer_tools(health: dict) -> None:
    """These scripts read traces or run evaluations, which a user deployment answers with 404."""
    if health.get("developer_tools") is not True:
        raise ValueError(DEVELOPER_TOOLS_REQUIRED)


def new_report(dataset: str, **fields) -> dict:
    return {"dataset":dataset, "started_at":datetime.now(timezone.utc).isoformat(), "synthetic":True, "judge":False,
            "cases":[], **fields}


class OnlineSmoke:
    """Runs against an online, healthy backend and never overwrites an earlier report."""

    def __init__(self, url: str, output: str, report: dict, planned: list[str] | None = None):
        self.url, self.target, self.report, self.planned = url.rstrip("/"), Path(output), report, planned
        self.failures = 0

    async def __aenter__(self):
        if self.target.exists():
            raise FileExistsError("preserve_existing_smoke_report:"+str(self.target))
        self.client = httpx.AsyncClient(timeout=240)
        try:
            response = await self.client.get(self.url+"/health")
            response.raise_for_status()
            health = response.json()
            if health.get("preview") is not False or health.get("status") != "ok":
                raise ValueError("healthy_online_runtime_required")
            require_developer_tools(health)
        except BaseException:
            await self.client.aclose()
            raise
        self.report["environment"] = health
        return self

    async def __aexit__(self, *_exc):
        await self.client.aclose()

    async def attempt(self, row: dict, body: dict, checks) -> dict:
        """Send one turn; record reply, trace and named checks, or only the exception type."""
        row["passed"] = False
        try:
            response = await self.client.post(self.url+"/chat", json=body)
            response.raise_for_status()
            row["reply"] = response.json()
            traced = await self.client.get(self.url+"/traces/"+row["reply"]["request_id"])
            traced.raise_for_status()
            row["trace"] = traced.json()
            row["checks"] = checks(row["reply"], row["trace"])
            row["passed"] = all(row["checks"].values())
        except Exception as exc:
            row["error"] = type(exc).__name__
        return row

    def record(self, row: dict, label: dict) -> bool:
        """Persist progress after every turn; False once consecutive failures reach the stop limit."""
        cases = self.report["cases"]
        cases.append(row)
        self.report.update(executed=len(cases), passed=sum(item["passed"] for item in cases))
        if self.planned is not None:
            self.report["not_run"] = self.planned[len(cases):]
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.target.write_text(json.dumps(self.report, ensure_ascii=False, indent=2)+"\n")
        reply = row.get("reply", {})
        print(json.dumps({**label, "passed":row["passed"],
                          "failed_checks":[key for key, value in row.get("checks", {}).items() if not value],
                          "error":row.get("error"), "usage":reply.get("usage"), "elapsed_ms":reply.get("elapsed_ms")},
                         ensure_ascii=False), flush=True)
        self.failures = 0 if row["passed"] else self.failures+1
        if self.failures >= STOP_AFTER_FAILURES:
            print("Stopped after two consecutive failures; recorded results preserved.", flush=True)
            return False
        return True


def arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--output", required=True)
    return parser
