"""Fixed-label classification mathematics and comparable, immutable baselines."""
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
from ..core.config import ROOT
from .judging import JUDGE_PROTOCOL_VERSION


CLASSIFIER_LABELS = ["symptoms", "medicine", "visit", "report", "general"]
GATED_LABELS = CLASSIFIER_LABELS+["urgent"]
METRIC_VERSION = "eval-metrics-5"
POLICY = {"classifier_labels":CLASSIFIER_LABELS, "gated_labels":GATED_LABELS,
          "classifier_cohort":"attempted_nonurgent_with_gate_skips_reported",
          "macro_f1":"all_fixed_labels_zero_division_0", "judge_threshold":.75,
         "regression_threshold":.05, "hard_gate_version":"engineering-safety-1",
         "judge_protocol_version":JUDGE_PROTOCOL_VERSION}


def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()


def exceeds_drop(previous,current):
    return previous > 0 and (Decimal(str(previous))-Decimal(str(current)))/Decimal(str(previous)) > Decimal("0.05")


def comparable_references(previous, current):
    if (previous.get("case_reference_scope_complete") is True and current.get("case_reference_scope_complete") is True
        and previous.get("case_reference_snapshots") == current.get("case_reference_snapshots")):
        return True
    return "references_hash" not in previous or previous["references_hash"] == current.get("references_hash")


def classification_metrics(expected, predicted, labels):
    if len(expected) != len(predicted) or any(label not in labels for label in expected):
        raise ValueError("invalid_classification_arrays")
    matrix = {label:{column:0 for column in [*labels,"__error__"]} for label in labels}
    for actual,guess in zip(expected,predicted): matrix[actual][guess if guess in labels else "__error__"] += 1
    per_class = {}
    for label in labels:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        guesses = sum(matrix[actual][label] for actual in labels)
        precision,recall = tp/guesses if guesses else 0, tp/support if support else 0
        per_class[label] = {"precision":precision,"recall":recall,
                            "f1":2*precision*recall/(precision+recall) if precision+recall else 0,"support":support}
    return {"labels":labels, "total":len(expected), "correct":sum(a==b for a,b in zip(expected,predicted)),
            "failures":sum(guess not in labels for guess in predicted),
            "accuracy":sum(a==b for a,b in zip(expected,predicted))/len(expected) if expected else None,
            "macro_f1":sum(row["f1"] for row in per_class.values())/len(labels) if expected else None,
            "per_class":per_class,"confusion_matrix":matrix}


class BaselineError(ValueError):
    pass


class BaselineStore:
    def __init__(self, directory: Path | None = None):
        self.directory = directory or ROOT/"artifacts"/"baselines"

    def path(self, identity):
        import re
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}",identity): raise BaselineError("invalid_baseline_id")
        return self.directory/(identity+".json")

    def load(self, identity):
        try:
            path = self.path(identity)
            if path.is_symlink(): raise BaselineError("baseline_symlink_rejected")
            if path.stat().st_size > 20000000: raise BaselineError("baseline_too_large")
            result = json.loads(path.read_text())
            if result["schema_version"] != "evaluation-2" or result["complete"] is not True or not isinstance(result["metrics"],dict):
                raise BaselineError("invalid_or_incomplete_baseline")
            for value in result["metrics"].values():
                if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= 1):
                    raise BaselineError("invalid_baseline_metric")
            for key in ["dataset_hash","metric_version","policy_hash"]:
                if not isinstance(result["metadata"][key],str): raise BaselineError("invalid_baseline_metadata")
            if not isinstance(result["safety_failures"],list): raise BaselineError("invalid_baseline_safety")
            if any(not isinstance(row,dict) or not isinstance(row.get("case_id"),str)
                   or type(row.get("turn")) is not int or not 1 <= row["turn"] <= 8 or not isinstance(row.get("gate"),str)
                   for row in result["safety_failures"]):
                raise BaselineError("invalid_baseline_safety")
            cases = result.get("cases",[])
            if not isinstance(cases,list) or any(not isinstance(row,dict) or not isinstance(row.get("id"),str) for row in cases):
                raise BaselineError("invalid_baseline_cases")
            for row in cases:
                turns = row.get("turns",[])
                if not isinstance(turns,list): raise BaselineError("invalid_baseline_turns")
                for turn in turns:
                    if not isinstance(turn,dict) or type(turn.get("turn")) is not int:
                        raise BaselineError("invalid_baseline_turns")
                    quality = turn.get("quality")
                    if quality is not None and (not isinstance(quality,dict) or any(
                        type(quality.get(key)) not in {int,float} or not math.isfinite(quality[key]) or not 0 <= quality[key] <= 1
                        for key in ["relevance","accuracy","completeness","helpfulness","mean"])):
                        raise BaselineError("invalid_baseline_quality")
            return result
        except (OSError,KeyError,TypeError,json.JSONDecodeError) as exc:
            raise BaselineError("baseline_unavailable_or_invalid") from exc

    def preflight(self, options, metadata):
        if options.save_as and self.path(options.save_as).exists(): raise BaselineError("baseline_already_exists")
        baseline = self.load(options.compare_to) if options.compare_to else None
        if baseline and any(baseline["metadata"][key] != metadata[key] for key in ["dataset_hash","metric_version","policy_hash"]):
            raise BaselineError("baseline_dataset_or_policy_incompatible")
        if baseline and not comparable_references(baseline["metadata"],metadata):
            raise BaselineError("baseline_reference_catalog_incompatible")
        return baseline

    def save(self, identity, report):
        if not report["complete"]: return {"saved":False,"reason":"incomplete_run"}
        self.directory.mkdir(parents=True,exist_ok=True)
        import os
        import tempfile
        serialized = json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
        if len(serialized.encode()) > 20000000: raise BaselineError("baseline_too_large")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w",dir=self.directory,prefix=".baseline-",delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush(); os.fsync(handle.fileno())
            os.link(temporary,self.path(identity))
        except FileExistsError as exc: raise BaselineError("baseline_already_exists") from exc
        finally:
            if temporary is not None: temporary.unlink(missing_ok=True)
        return {"saved":True,"id":identity}


def compare_baseline(report, baseline):
    if baseline is None: return {"status":"not_requested","regressions":[],"metrics":[],"variables":[]}
    if any(report["metadata"][key] != baseline["metadata"][key] for key in ["dataset_hash","metric_version","policy_hash"]):
        raise BaselineError("baseline_dataset_or_policy_incompatible")
    if not comparable_references(baseline["metadata"],report["metadata"]):
        raise BaselineError("baseline_reference_catalog_incompatible")
    previous_safety = {(row["case_id"],row["turn"],row["gate"]) for row in baseline["safety_failures"]}
    regressions = [{"kind":"new_safety_failure",**row} for row in report["safety_failures"]
                   if (row["case_id"],row["turn"],row["gate"]) not in previous_safety]
    if not report["complete"]: return {"status":"incomplete_run","regressions":regressions,"metrics":[],"variables":[]}
    changes = []
    for name,current in report["metrics"].items():
        previous = baseline["metrics"].get(name)
        change = {"metric":name,"baseline":previous,"current":current,"direction":"higher_is_better"}
        if current is None or previous is None: change["status"] = "missing"
        elif previous == 0: change["status"] = "zero_baseline"
        else:
            drop = (Decimal(str(previous))-Decimal(str(current)))/Decimal(str(previous))
            change.update(status="regression" if drop > Decimal("0.05") else "within_threshold",relative_drop=float(drop))
            if drop > Decimal("0.05"): regressions.append({"kind":"quality",**change})
        changes.append(change)
    variables = [{"field":key,"baseline":baseline["metadata"].get(key),"current":report["metadata"].get(key)}
                 for key in ["model","provider","code_hash","skills_hash","references_hash","profiles_hash","routing_version","runtime_settings_hash","evaluation_limits_hash"]
                 if baseline["metadata"].get(key) != report["metadata"].get(key)]
    return {"status":"compared","regressions":regressions,"metrics":changes,"variables":variables}
