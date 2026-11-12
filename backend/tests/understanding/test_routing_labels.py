import json
import re
import pytest
from support.harness import fixture, run as run_async
from support.multiturn import ask
from medagent.core.config import ROOT
from medagent.evaluation.routing_labels import DATASET, run, summarize
from medagent.understanding.conversation import VISIT_EVENT, VISIT_PREPARATION_REQUEST

HOLDOUT = ROOT/"resources"/"routing-labels-holdout-v1.json"
# Holdout misses after the 2026-10-30 rule change: vocabulary outside the keyword lists and one
# unrecognized negation. Fixing them against this holdout would spend it; they need a new one.
KNOWN_HOLDOUT_MISSES = {"h01", "h02", "h06", "h08", "h22"}
# This offline run has no language channel. Since 2026-11-11 no rule rewrites the fused primary intent, so a
# background drug mention outweighs a visit request in keywords alone; the model channel decides it online.
KNOWN_TUNING_MISSES = {"t40"}


def test_label_sets_are_unique_and_primary_is_inside_the_role_set():
    for path in (DATASET, HOLDOUT):
        cases = json.loads(path.read_text())["cases"]
        assert len({case["id"] for case in cases}) == len(cases)
        assert all(set(case["primary"]) <= set(case["roles"]) for case in cases)


def test_tuning_set_routes_every_case():
    report = run()
    assert report["paid_model_calls"] == 0 and report["language_channel"] == "not_used_offline"
    assert set(report["summary"]["mismatches"]) <= KNOWN_TUNING_MISSES


def test_holdout_does_not_regress_beyond_known_misses():
    assert set(run(HOLDOUT)["summary"]["mismatches"]) <= KNOWN_HOLDOUT_MISSES


def test_summary_counts_support_only_when_the_primary_is_right():
    rows = [
        {"id":"a", "category":"compound", "expected_primary":["symptom_guide", "drug_reference"],
         "expected_roles":["drug_reference", "symptom_guide"], "actual_primary":"symptom_guide",
         "actual_roles":["drug_reference", "symptom_guide"], "primary_ok":True, "roles_ok":True},
        {"id":"b", "category":"single", "expected_primary":["visit_guide"], "expected_roles":["visit_guide"],
         "actual_primary":"general", "actual_roles":["general"], "primary_ok":False, "roles_ok":False},
    ]
    summary = summarize(rows)
    assert summary["support_recall"] == {"expected":1, "admitted":1, "rate":1.0}
    assert summary["extra_roles"] == 1 and summary["mismatches"] == ["b"]
    assert summary["single_label_confusion"]["visit_guide"]["general"] == 1


@pytest.mark.parametrize("text", ["下周复诊需要准备什么", "复查之前要整理哪些资料", "去门诊之前应该准备些什么",
                                  "整理去医院要带的东西"])
def test_visit_preparation_covers_follow_up_visits(text):
    assert re.search(VISIT_PREPARATION_REQUEST, text)


@pytest.mark.parametrize("text", ["我不去医院，只想查一下布洛芬的禁忌", "没去医院，想记录症状", "别去医院了"])
def test_negated_hospital_trip_is_not_a_visit(text):
    assert not re.search(VISIT_EVENT, text)


@pytest.mark.parametrize("text,roles", [
    ("我在吃二甲双胍，想整理一下最近头晕的症状", ["symptom_guide"]),
    ("报告我已经看懂了，现在想查一下阿莫西林的说明书", ["drug_reference"]),
    ("我不去医院，只想查一下布洛芬的禁忌", ["drug_reference"]),
    ("布洛芬能止痛吗", ["drug_reference"]),
    ("帮我解释一下化验报告里的参考范围，再查一下布洛芬的禁忌", ["drug_reference", "report_explainer"]),
])
def test_mentions_do_not_add_roles_but_requests_do(text, roles):
    async def scenario():
        reply = await ask(await fixture(), text, user="labels")
        assert sorted([reply.primary_agent, *reply.supporting_agents]) == roles
    run_async(scenario())


def test_keyword_tasks_are_marked_and_can_still_replace_a_general_primary():
    async def scenario():
        reply = await ask(await fixture(), "我在吃二甲双胍，想整理一下最近头晕的症状", user="source")
        assert [task.source for task in reply.intent.tasks] == ["classifier", "keywords"]
        replaced = await ask(await fixture(), "华法林和阿司匹林能一起吃吗", user="general-primary")
        assert replaced.primary_agent == "drug_reference"
    run_async(scenario())


class StageLog:
    """Language channel stand-in that records every model stage it is asked for."""
    def __init__(self):
        self.stages = []

    async def json(self, stage, system, content, budget=900):
        self.stages.append(stage)
        return {"scores":{"general":1}, "tasks":[], "updates":[], "topic_relation":"uncertain"}

    async def message(self, stage, system, messages, tools=None, budget=1800):
        self.stages.append(stage)
        raise RuntimeError("roles_must_stay_in_preview")


def test_online_channel_calls_the_model_only_for_understanding_and_keeps_roles_in_preview():
    from medagent.evaluation.routing_labels import score
    model = StageLog()
    report = run_async(score(DATASET, model=model, model_name="fake"))
    assert set(model.stages) == {"understanding"} and report["kind"] == "online_routing_labels"
    assert report["channels"][-1] == "language:fake" and report["budget_stop"] is None
    assert report["paid_model_calls"] == len(model.stages) == report["usage"]["attempted_calls"]
    # Urgent turns stop at the risk gate before understanding; every other scored turn used the language channel.
    assert report["language_channel"] == "scored_cases_without_language:0"


def test_online_budget_cap_is_reported_and_cases_without_language_stay_scored():
    from medagent.evaluation.contracts import EvalLimits
    from medagent.evaluation.routing_labels import score
    model = StageLog()
    report = run_async(score(DATASET, model=model, limits=EvalLimits(max_model_calls=3, max_output_tokens=2700, per_call_output_tokens=900)))
    assert len(model.stages) == 3 and report["budget_stop"] == "max_model_calls"
    assert report["summary"]["cases"] == 58 and int(report["language_channel"].rsplit(":", 1)[1]) > 0


def test_online_cli_requires_an_explicit_call_cap(monkeypatch, tmp_path):
    from medagent.evaluation import routing_labels
    monkeypatch.setattr("sys.argv", ["routing_labels", "--online", "--output", str(tmp_path/"x.json")])
    with pytest.raises(SystemExit):
        routing_labels.main()
    assert not (tmp_path/"x.json").exists()
