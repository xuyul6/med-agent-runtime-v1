import json
from dataclasses import fields, replace
import httpx
from pathlib import Path
from types import SimpleNamespace
import pytest
from support.comparison import RehearsalModel
from support.harness import fixture, run
from medagent.comparison.contracts import ComparisonCase, ComparisonRequest
from medagent.comparison.runner import run_comparison
from medagent.core.config import Settings, UNHASHED_SETTINGS
from medagent.core.provider import ClaudeGateway, json_schema, judge_gateway
from medagent.evaluation.budget import judge_model_name, judge_source
from medagent.evaluation.contracts import EvalLimits
from medagent.evaluation.judging import JUDGE_INSTRUCTIONS, JUDGE_PROTOCOL_VERSION
from medagent.evaluation.rejudge import score, summarize
from medagent.runtime import MedicalRuntime

SAVED = Path("artifacts/phase4-comparison-formal-1e5c3ab-2026-10-30.json")
LIMITS = EvalLimits(max_model_calls=100, max_output_tokens=95000, per_call_output_tokens=950)


class Judge(RehearsalModel):
    settings = SimpleNamespace(model="judge-model")


def test_judge_needs_both_key_and_model():
    assert judge_gateway(Settings(judge_model="deepseek-v4-pro")) is None
    assert judge_gateway(Settings(fallback_api_key="k")) is None


def test_judge_gateway_uses_the_second_vendor():
    settings = judge_gateway(Settings(api_key="primary", model="claude", fallback_api_key="k", judge_model="deepseek-v4-pro")).settings
    assert settings.model == "deepseek-v4-pro" and settings.api_key == "k"
    assert settings.api_url == "https://api.deepseek.com/anthropic" and settings.structured_output is False


def test_secrets_stay_out_of_the_settings_hash():
    names = {field.name for field in fields(Settings)}
    assert {"api_key", "fallback_api_key", "fallback_api_url"} <= UNHASHED_SETTINGS & names
    assert "judge_model" not in UNHASHED_SETTINGS


def test_judge_defaults_to_the_runtime_model():
    source = SimpleNamespace(model="primary", judge_model=None, settings=SimpleNamespace(model="claude"))
    assert judge_source(source) == "primary" and judge_model_name(source) == "claude"
    judge = Judge()
    assert judge_model_name(SimpleNamespace(model="primary", judge_model=judge, settings=source.settings)) == "judge-model"


def test_comparison_scores_with_the_judge_model_only():
    primary, judge = RehearsalModel(), Judge()
    async def scenario():
        base = await fixture(primary)
        source = MedicalRuntime(replace(base.settings, preview=False), primary, base.vectors, base.index, base.memory, judge_model=judge)
        return await run_comparison(source, ComparisonRequest(cases=[ComparisonCase(id="g", category="general", target={"message":"你好，你能做什么？"})]))
    report = run(scenario())
    assert "evaluation_judge" not in primary.stages and judge.stages.count("evaluation_judge") == 9
    assert report["metadata"]["judge_model"] == "judge-model"


def test_summary_counts_errors_and_repeat_spread():
    row = lambda pair, old, new: {"pair":pair, "arm":"a", "original_mean":old, "errors":[], "elapsed_ms":[1],
        "original":{key:old for key in ("relevance", "accuracy", "completeness", "helpfulness")},
        "scores":[{"mean":value, **{key:value for key in ("relevance", "accuracy", "completeness", "helpfulness")}} for value in new]}
    rows = [row("a", .8, [.7, .9]), row("b", .6, [.5, .5]), row("c", .9, [.95, .95]), {**row("d", .7, []), "errors":["ValueError"]}]
    summary = summarize(rows)
    assert summary["answers"] == 4 and summary["fully_scored"] == 3 and summary["errors"] == 1
    assert summary["max_repeat_spread"] == pytest.approx(.2) and summary["rank_correlation"] == 1


def test_instructions_cap_the_defects_a_judge_may_excuse():
    assert all(text in JUDGE_INSTRUCTIONS for text in ("评分锚点", "完整性不超过 0.5", "准确性不超过 0.6", "有用性不超过 0.8"))
    assert JUDGE_PROTOCOL_VERSION == "judge-context-5"


def test_reason_is_written_before_the_scores():
    assert list(json_schema("evaluation_judge")["properties"])[0] == "reason"
    assert JUDGE_INSTRUCTIONS.index("先写 reason") < JUDGE_INSTRUCTIONS.index("四个0到1数值")


def test_thinking_is_off_unless_requested():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content)["thinking"])
        return httpx.Response(200, json={"model":"m", "stop_reason":"end_turn", "usage":{"input_tokens":1, "output_tokens":1},
            "content":[{"type":"thinking", "thinking":"分析"}, {"type":"text", "text":'{"reason":"r"}'}]})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert run(ClaudeGateway(Settings(api_key="k", model="m"), client).json("evaluation_judge", "s", "c")) == {"reason":"r"}
    assert run(ClaudeGateway(Settings(api_key="k", model="m", thinking=True), client).json("evaluation_judge", "s", "c")) == {"reason":"r"}
    assert seen == [{"type":"disabled"}, {"type":"enabled"}]


def test_all_failures_report_no_scores():
    assert summarize([{"scores":[], "errors":["X"]}]) == {"answers":1, "fully_scored":0}


@pytest.mark.skipif(not SAVED.exists(), reason="local report")
def test_rebuilt_input_matches_what_the_original_judge_saw():
    class Fake:
        calls = 0
        async def json(self, stage, system, content, budget=950):
            Fake.calls += 1
            return {"relevance":.8, "accuracy":.8, "completeness":.7, "helpfulness":.8, "reason":"假模型"}
    report = run(score(json.loads(SAVED.read_text()), Fake(), LIMITS, per_category=6, repeats=2, seed=20261030, model_name="fake"))
    assert report["skipped_input_mismatch"] == 0 and report["summary"]["answers"] == 24 and Fake.calls == 48
    assert report["summary"]["errors"] == 0
