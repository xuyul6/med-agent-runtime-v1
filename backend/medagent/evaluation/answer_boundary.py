"""Offline red-team measurement of the deterministic answer boundary check; no model, storage or network."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from ..agents.checks import check_boundaries
from ..agents.output_validation import AgentOutputError
from ..core.config import ROOT
from .runner import CODE_HASH

DATASET = ROOT/"resources"/"answer-boundary-redteam-v1.json"


def blocked_by(text: str) -> str | None:
    try:
        check_boundaries({"text":text})
    except AgentOutputError as exc:
        return exc.code
    return None


def rate(hits: int, total: int) -> float | None:
    return round(hits/total, 4) if total else None


def summarize(rows: list[dict]) -> dict:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    block = [row for row in rows if row["expected"] == "block"]
    allow = [row for row in rows if row["expected"] == "allow"]
    caught = sum(row["code"] is not None for row in block)
    wrongly = [row for row in allow if row["code"] is not None]
    return {
        "cases":len(rows),
        "block_recall":{"total":len(block), "blocked":caught, "rate":rate(caught, len(block))},
        "false_block":{"total":len(allow), "blocked":len(wrongly), "rate":rate(len(wrongly), len(allow))},
        "by_category":{name:{"expected":items[0]["expected"], "cases":len(items), "passed":sum(row["passed"] for row in items)}
                       for name, items in sorted(by_category.items())},
        "missed":[row["id"] for row in block if row["code"] is None],
        "false_blocks":[row["id"] for row in wrongly],
    }


def run(dataset: Path = DATASET) -> dict:
    raw = dataset.read_bytes()
    body = json.loads(raw)
    ids = Counter(case["id"] for case in body["cases"])
    if duplicates := [key for key, count in ids.items() if count > 1]:
        raise ValueError("duplicate_case_ids:"+",".join(duplicates))
    rows = []
    for case in body["cases"]:
        if case["expected"] not in {"block", "allow"}:
            raise ValueError("unknown_expected:"+case["id"])
        code = blocked_by(case["text"])
        rows.append({**case, "code":code, "passed":(code is not None) == (case["expected"] == "block")})
    return {"kind":"offline_answer_boundary_redteam", "dataset":body["version"], "synthetic":body["synthetic"],
            "dataset_sha256":hashlib.sha256(raw).hexdigest(), "code_sha256":CODE_HASH,
            "labels":"engineering labels, not clinically reviewed", "paid_model_calls":0,
            "generated_at":datetime.now(timezone.utc).isoformat(), "summary":summarize(rows), "cases":rows}


def main():
    parser = argparse.ArgumentParser(description="Offline red-team check of answer boundary rules; no model calls")
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("preserve_existing_boundary_report:"+str(target))
    report = run(Path(args.dataset))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    summary = report["summary"]
    print(f'blocked {summary["block_recall"]["blocked"]}/{summary["block_recall"]["total"]}; '
          f'false blocks {summary["false_block"]["blocked"]}/{summary["false_block"]["total"]}. Saved: {target}')


if __name__ == "__main__":
    main()
