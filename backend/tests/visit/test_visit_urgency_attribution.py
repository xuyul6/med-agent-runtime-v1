"""Do not remove urgent advice while distinguishing it from a cited preparation template."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from medagent.app import create_app
from medagent.agents.collaboration import sections_answer
from medagent.core.config import Settings
from medagent.core.contracts import Answer, Intent, Question, Reference
from medagent.agents.checks import checked_role_output
from medagent.agents.profiles import DEFAULT_REGISTRY
from support.harness import Model, fixture, role_output, run
from support.routing import fixed_understanding, understanding


WARNING = "如出现剧烈头痛、伴随肢体无力或意识改变，请立即就近急诊或拨打急救电话。"
NOTICE = "部分具体风险提示未能与本次引用资料逐句核对；如情况紧急，请立即联系当地急救或就近急诊。"


def source(text, key="visit-synthetic"):
    return Reference(key=key,title="就医准备模板",text=text,domain=Intent.VISIT,
                     url="https://example.org/visit",version="test",scope="合成资料",approved=True)


def output(text, key="visit-synthetic"):
    data = role_output("visit_guide",[key])
    data["answer"]["text"] = text
    return data


class WarningModel(Model):
    async def message(self,stage,system,messages,tools=None,budget=1800):
        if stage == "agent_visit_guide":
            data = output("请整理资料。"+WARNING,"consultation-record")
            return {"content":[{"type":"text", "text":json.dumps(data,ensure_ascii=False)}]}
        return await super().message(stage,system,messages,tools,budget)


def test_specific_warning_stays_visible_with_unmatched_source_notice():
    ref = source("请整理既往资料和就医问题。")
    data = output("请整理资料。"+WARNING)
    result = checked_role_output(DEFAULT_REGISTRY["visit_guide"],data,{ref.key:ref})
    assert WARNING in result.answer.text and NOTICE in result.answer.limitations
    assert result.answer.reference_keys == [ref.key]


def test_exact_cited_warning_does_not_get_unmatched_notice():
    ref = source(WARNING)
    result = checked_role_output(DEFAULT_REGISTRY["visit_guide"],output(WARNING),{ref.key:ref})
    assert result.answer.text == WARNING and NOTICE not in result.answer.limitations


def test_matching_uncited_source_does_not_remove_notice():
    ref = source("请整理既往资料和就医问题。")
    uncited = source(WARNING, "other-visit")
    result = checked_role_output(DEFAULT_REGISTRY["visit_guide"],output(WARNING),{ref.key:ref,uncited.key:uncited})
    assert WARNING in result.answer.text and NOTICE in result.answer.limitations


def test_negated_source_sentence_is_not_treated_as_matching_warning():
    ref = source("不要认为"+WARNING)
    result = checked_role_output(DEFAULT_REGISTRY["visit_guide"],output(WARNING),{ref.key:ref})
    assert WARNING in result.answer.text and NOTICE in result.answer.limitations


def test_unrelated_or_generic_advice_is_not_labeled_as_specific_warning():
    ref = source("请整理既往资料和就医问题。")
    text = "请整理既往资料。如有紧急情况请立即联系当地急救。"
    result = checked_role_output(DEFAULT_REGISTRY["visit_guide"],output(text),{ref.key:ref})
    assert result.answer.text == text and NOTICE not in result.answer.limitations


def test_next_step_warning_and_existing_six_limitations_are_preserved():
    ref = source("请整理既往资料和就医问题。")
    data = output("请整理资料。")
    data["answer"]["next_steps"] = [WARNING]
    data["answer"]["limitations"] = [f"原限制{i}" for i in range(6)]
    result = checked_role_output(DEFAULT_REGISTRY["visit_guide"],data,{ref.key:ref})
    assert result.answer.next_steps == [WARNING] and len(result.answer.limitations) <= 6
    joined = "\n".join(result.answer.limitations)
    assert NOTICE in joined and all(f"原限制{i}" in joined for i in range(6))


def test_support_notice_survives_collaboration_with_six_primary_limitations():
    primary = SimpleNamespace(execution=SimpleNamespace(role="symptom_guide", task_id="task:"+"symptom_guide", status="answered", missing_fields=[]),
        answer=Answer(text="症状记录",limitations=[f"主限制{i}" for i in range(6)]))
    support = SimpleNamespace(execution=SimpleNamespace(role="visit_guide", task_id="task:"+"visit_guide", status="answered", missing_fields=[]),
        answer=Answer(text=WARNING,limitations=[NOTICE,"辅助限制"]))
    answer = sections_answer([primary,support])
    joined = "\n".join(answer.limitations)
    assert WARNING in answer.text and NOTICE in joined and "辅助限制" in joined
    assert all(f"主限制{i}" in joined for i in range(6)) and len(answer.limitations) <= 6


def test_generated_warning_is_labeled_in_real_chat_without_logging_raw_text():
    async def scenario():
        runtime = await fixture(WarningModel())
        reply = await runtime.chat(Question(text="第一次看医生，我应该准备什么资料？"))
        assert WARNING in reply.answer.text and NOTICE in "\n".join(reply.answer.limitations)
        assert reply.answer.reference_keys == ["consultation-record"]
        events = runtime.monitor.traces[-1]["events"]
        markers = [e for e in events if e["event"] == "evidence_attribution"]
        assert len(markers) == 1 and markers[0]["status"] == "not_verbatim"
        assert WARNING not in json.dumps(events,ensure_ascii=False)

    run(scenario())


def test_chat_http_response_exposes_both_warning_and_attribution_limit():
    runtime = run(fixture(WarningModel()))
    with TestClient(create_app(Settings(preview=True,embeddings="hash"),runtime)) as client:
        response = client.post("/chat",json={"text":"第一次看医生，我应该准备什么资料？"})
    assert response.status_code == 200
    answer = response.json()["answer"]
    assert WARNING in answer["text"] and NOTICE in "\n".join(answer["limitations"])
    assert answer["reference_keys"] == ["consultation-record"]


def test_support_warning_and_all_limitations_survive_actual_collaboration():
    class CombinedModel(Model):
        async def message(self,stage,system,messages,tools=None,budget=1800):
            if stage in {"agent_visit_guide", "agent_symptom_guide"}:
                role = stage.removeprefix("agent_")
                data = role_output(role,["consultation-record"] if role == "visit_guide" else [])
                data["answer"]["text"] = WARNING if role == "visit_guide" else "症状记录。"
                data["answer"]["limitations"] = [f"{role}限制{i}" for i in range(6)]
                return {"content":[{"type":"text", "text":json.dumps(data,ensure_ascii=False)}]}
            return await super().message(stage,system,messages,tools,budget)

    async def scenario():
        runtime = await fixture(CombinedModel())
        fixed_understanding(runtime,understanding(primary=Intent.SYMPTOMS,kinds=(Intent.SYMPTOMS,Intent.VISIT),
            scores={"symptoms":.8,"visit":.2}))
        reply = await runtime.chat(Question(text="咳嗽三天，想去看医生，需要整理资料。"))
        assert reply.primary_agent == "symptom_guide" and reply.supporting_agents == ["visit_guide"]
        combined = "\n".join(reply.answer.limitations)
        assert WARNING in reply.answer.text and NOTICE in combined
        assert all(f"{role}限制{i}" in combined for role in ("symptom_guide","visit_guide") for i in range(6))
        assert len(reply.answer.limitations) <= 6

    run(scenario())


def test_critical_risk_short_circuits_without_waiting_for_citation_checks():
    async def scenario():
        runtime = await fixture(WarningModelThatMustNotRun())
        reply = await runtime.chat(Question(text="我现在突然呼吸困难，越来越严重。"))
        assert reply.risk.level == "CRITICAL" and reply.primary_agent == "risk_handoff"
        assert reply.usage["calls"] == 0 and reply.storage == "skipped"

    run(scenario())


class WarningModelThatMustNotRun:
    def __getattr__(self, _name):
        raise AssertionError("critical_path_called_model")
