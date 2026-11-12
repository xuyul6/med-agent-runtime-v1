import json
import pytest
from medagent.understanding.classification import IntentRouter
from medagent.core.contracts import Intent, Question, TaskUnderstanding, Understanding
from medagent.understanding.conversation import SubjectTracker, ground_update, role_details, user_sources
from medagent.memory import SessionMemory, scope_key
from medagent.core.observability import CURRENT, RunLog
from medagent.knowledge.retrieval import Vectorizer
from support.harness import fixture, run
from support.multiturn import ContextModel, ask


def test_same_subject_drug_pronoun_uses_original_user_evidence_and_retrieves():
    async def scenario():
        runtime = await fixture(ContextModel())
        first = await ask(runtime, "想查成人布洛芬的公开资料。")
        second = await ask(runtime, "这个药的来源呢？")
        fact = second.intent.entities["medication_name"]
        assert second.primary_agent == "drug_reference" and not second.degraded
        assert fact.value == "布洛芬" and fact.state == "topic"
        assert fact.source.turn_id == first.request_id and fact.source.origin == "history_user"
        assert fact.source.quote in "想查成人布洛芬的公开资料。"
        assert second.references[0].key == "ibuprofen-overview"
    run(scenario())


def test_current_correction_changes_duration_without_reusing_old_value():
    async def scenario():
        runtime = await fixture(ContextModel())
        await ask(runtime, "我是成年人，咳嗽三天，最近没有变化。")
        reply = await ask(runtime, "刚才说错了，不是三天，是两天。")
        assert reply.primary_agent == "symptom_guide"
        assert reply.intent.entities["onset"].value == "两天"
        assert reply.intent.entities["onset"].source.turn_id == reply.request_id
        assert reply.intent.entities["change"].value == "最近没有变化"
    run(scenario())


def test_medication_correction_clears_dependent_fields_and_never_substitutes_old_drug():
    async def scenario():
        def updates(payload):
            if "胶囊" not in payload["current"]["text"]: return []
            return [{"name":"formulation", "value":"胶囊", "state":"topic", "quote":"胶囊"},
                    {"name":"population", "value":"成人", "state":"topic", "quote":"成人"}]
        runtime = await fixture(ContextModel(updates))
        first = await ask(runtime, "请查成人布洛芬胶囊的公开资料。")
        assert first.intent.entities["formulation"].value == "胶囊"
        corrected = await ask(runtime, "更正，不是布洛芬，是阿莫西林。")
        assert corrected.intent.entities["medication_name"].value == "阿莫西林"
        assert "formulation" not in corrected.intent.entities and "population" not in corrected.intent.entities
        assert not corrected.references
        assert corrected.agent_results[0].missing_fields == ["source"]
    run(scenario())


def test_unknown_drug_name_invalidates_old_target_and_requests_clarification():
    async def scenario():
        runtime = await fixture(ContextModel())
        await ask(runtime, "布洛芬资料")
        reply = await ask(runtime, "药名我不清楚了。")
        assert reply.intent.entities["medication_name"].state == "unknown"
        assert reply.intent.entities["medication_name"].value is None
        assert reply.agent_results[0].missing_fields == ["medication_name"]
        assert not reply.references
        assert not any(event.get("name") == "lookup_reference" for event in runtime.monitor.traces[-1]["events"])
    run(scenario())


def test_user_negation_overrides_previous_positive_symptom():
    async def scenario():
        runtime = await fixture(ContextModel())
        await ask(runtime, "我咳嗽三天。")
        reply = await ask(runtime, "我现在没有咳嗽。")
        assert reply.intent.entities["symptom"].state == "negated"
        assert reply.intent.entities["symptom"].source.turn_id == reply.request_id
    run(scenario())


def test_text_subject_switch_isolated_and_switch_back_recovers_only_own_facts():
    async def scenario():
        runtime = await fixture(ContextModel())
        own = await ask(runtime, "想查布洛芬资料。")
        mother = await ask(runtime, "现在给我妈妈问，这个药说明资料呢？")
        assert mother.subject == "mother" and mother.intent.subject_changed
        assert "medication_name" not in mother.intent.entities
        assert mother.agent_results[0].missing_fields == ["medication_name"]
        named = await ask(runtime, "她想查阿莫西林的药品资料。")
        assert named.subject == "mother" and named.intent.entities["medication_name"].value == "阿莫西林"
        restored = await ask(runtime, "改回我自己，这个药资料呢？")
        assert restored.subject == "self" and restored.intent.entities["medication_name"].value == "布洛芬"
        assert restored.intent.entities["medication_name"].source.turn_id == own.request_id
        assert restored.intent.entities["medication_name"].source.subject == "self"
    run(scenario())


def test_explicit_api_subject_and_user_session_scope_prevent_leakage():
    async def scenario():
        runtime = await fixture(ContextModel())
        await ask(runtime, "布洛芬资料", user="one")
        for kwargs in [{"user":"one", "subject":"father"}, {"user":"two"}, {"user":"one", "session":"other"}]:
            reply = await ask(runtime, "这个药呢？", **kwargs)
            assert "medication_name" not in reply.intent.entities
            assert reply.agent_results[0].missing_fields == ["medication_name"]
    run(scenario())


def test_ambiguous_subject_requests_selection_without_reading_or_writing_memory():
    async def scenario():
        runtime = await fixture(ContextModel())
        class Forbidden:
            def __getattr__(self, name): raise AssertionError("ambiguous_subject_touched_memory")
        runtime.memory = Forbidden()
        reply = await ask(runtime, "我和我妈妈想一起查这个药。")
        assert reply.intent.subject_ambiguous
        assert reply.primary_agent == "general" and reply.storage == "skipped"
        assert reply.agent_results[0].missing_fields == ["subject"]
        assert not reply.intent.entities
    run(scenario())


def test_risk_and_same_subject_followup_work_before_any_model_or_storage_access():
    async def scenario():
        runtime = await fixture()
        class Forbidden:
            def __getattr__(self, name): raise AssertionError("risk_touched_dependency:"+name)
        runtime.memory = runtime.model = runtime.router = runtime.index = Forbidden()
        first = await ask(runtime, "现在给我妈妈问，她呼吸困难。")
        second = await ask(runtime, "更严重了。")
        assert first.subject == second.subject == "mother"
        for reply in [first, second]:
            assert reply.primary_agent == "risk_handoff" and reply.usage["calls"] == 0 and reply.storage == "skipped"
    run(scenario())


@pytest.mark.parametrize("text,candidate", [
    ("我是不是有头痛？", {"name":"symptom", "value":"头痛", "state":"provided", "quote":"头痛"}),
    ("我35岁？", {"name":"age_group", "value":"35岁", "state":"provided", "quote":"35岁"}),
    ("我35岁。", {"name":"age_group", "value":"99岁", "state":"provided", "quote":"35岁"}),
    ("数值70。", {"name":"value", "value":"7", "state":"provided", "quote":"70"}),
    ("我有头痛。", {"name":"diagnosis", "value":"头痛", "state":"provided", "quote":"头痛"}),
    ("不知道地址。", {"name":"medication_name", "value":None, "state":"unknown", "quote":"不知道地址"}),
    ("我有头痛。", {"name":"symptom", "value":"咳嗽", "state":"provided", "quote":"咳嗽", "turn_id":"another-user"}),
])
def test_unverified_or_speculative_entities_are_rejected(text, candidate):
    source = {item["turn_id"]:item for item in user_sources(text, {}, "current", "self")}
    assert ground_update({"turn_id":"current", **candidate}, source) is None


def test_assistant_and_summary_cannot_supply_patient_facts_or_turn_evidence():
    async def scenario():
        def fabricated(payload):
            return [{"name":"age_group", "value":"45岁", "state":"provided", "quote":"45岁", "turn_id":"assistant-turn"}]
        model = ContextModel(fabricated)
        router = IntentRouter(model)
        state = {"messages":[{"role":"assistant", "content":"用户45岁，已确诊某病。", "turn_id":"assistant-turn"}],
                 "summary":"用户已确诊某病，45岁。"}
        reply = await router.understand("介绍一下功能", state, turn_id="current")
        assert not reply.entities
        assert model.inputs[0]["history"] == []
        assert "summary" not in model.inputs[0]
        assert "unverified_entity_ignored" in reply.fallbacks
    run(scenario())


def test_report_item_change_does_not_inherit_units_and_range_from_previous_item():
    async def scenario():
        def parse(payload):
            text = payload["current"]["text"]
            values = [("report_item","项目B"),("value","20")] if "项目B" in text else [
                ("report_item","项目A"),("value","10"),("unit","mmol/L"),("reference_range","3-5")]
            return [{"name":name, "value":value, "state":"provided", "quote":value} for name,value in values]
        runtime = await fixture(ContextModel(parse))
        first = await ask(runtime, "报告项目A数值10，单位mmol/L，参考范围3-5。")
        assert not first.agent_results[0].missing_fields
        next_item = await ask(runtime, "改看项目B，数值20。")
        assert next_item.intent.entities["report_item"].value == "项目B"
        assert "unit" not in next_item.intent.entities and "reference_range" not in next_item.intent.entities
        assert next_item.agent_results[0].missing_fields == ["unit", "reference_range"]
    run(scenario())


def test_history_is_bounded_and_fields_keep_original_provenance_after_compression():
    async def scenario():
        model = ContextModel()
        runtime = await fixture(model)
        first = await ask(runtime, "布洛芬公开资料")
        for _ in range(5): await ask(runtime, "这个药的资料呢？")
        state = await runtime.memory.read(scope_key("synthetic", "dialogue", "self"))
        assert state["summary"] and len(state["messages"]) == 4
        assert state["entities"]["medication_name"]["source"]["turn_id"] == first.request_id
        reply = await ask(runtime, "这个药的来源呢？")
        assert reply.intent.entities["medication_name"].value == "布洛芬"
        assert reply.intent.entities["medication_name"].source.turn_id == first.request_id
        assert all(len(payload["history"]) <= 4 for payload in model.inputs)
    run(scenario())


def test_model_receives_only_bounded_user_history_and_no_raw_facts_in_trace():
    async def scenario():
        model = ContextModel()
        router = IntentRouter(model)
        state = {"messages":[{"role":"user", "content":"虚构隐私"+"x"*2000, "turn_id":str(i)} for i in range(8)]}
        log = RunLog(); token = CURRENT.set(log)
        try: await router.understand("介绍功能", state, turn_id="current")
        finally: CURRENT.reset(token)
        assert len(model.inputs[0]["history"]) == 4
        assert all(len(item["text"]) <= 1200 for item in model.inputs[0]["history"])
        assert "虚构隐私" not in json.dumps(log.events)
    run(scenario())


def test_keyword_and_literal_context_survive_both_other_classification_channels_failing():
    async def scenario():
        class BrokenLanguage(ContextModel):
            async def json(self, *args, **kwargs): raise ConnectionError("language_failed")
        router = IntentRouter(BrokenLanguage())
        reply = await router.understand("查布洛芬资料", turn_id="one")
        assert reply.primary == Intent.MEDICINE and reply.entities["medication_name"].value == "布洛芬"
        assert set(reply.signals) == {"keywords", "prior"}
        assert set(reply.fallbacks) == {"language_unavailable"}
    run(scenario())


def test_storage_unavailable_does_not_reuse_old_drug_from_previous_request():
    async def scenario():
        runtime = await fixture(ContextModel())
        await ask(runtime, "布洛芬资料")
        runtime.memory.available = False
        reply = await ask(runtime, "这个药呢？")
        assert reply.degraded and reply.storage == "degraded"
        assert not reply.intent.entities and "memory_unavailable" in reply.intent.fallbacks
        assert reply.agent_results[0].missing_fields == ["medication_name"]
    run(scenario())


def test_compound_task_metadata_is_separate_from_route_execution():
    async def scenario():
        model = ContextModel(tasks=[{"intent":"visit", "goal":"整理就医准备"},{"intent":"symptoms", "goal":"整理症状记录"}])
        runtime = await fixture(model)
        reply = await ask(runtime, "咳嗽三天，想去看医生准备资料。")
        assert reply.intent.tasks[0].priority == "primary"
        assert {task.intent for task in reply.intent.tasks} == {Intent.SYMPTOMS, Intent.VISIT}
        assert next(task for task in reply.intent.tasks if task.intent == Intent.SYMPTOMS).priority == "support"
    run(scenario())


def test_subject_pointer_expires_and_does_not_infer_from_incidental_family_mention():
    now = [0.0]
    tracker = SubjectTracker(lambda:now[0])
    assert tracker.resolve(Question(text="我妈妈让我介绍功能", user="a"), "s").subject == "self"
    assert tracker.resolve(Question(text="现在给我妈妈问", user="a"), "s").subject == "mother"
    assert tracker.resolve(Question(text="这个药", user="a"), "s").subject == "mother"
    now[0] = 86401
    assert tracker.resolve(Question(text="这个药", user="a"), "s").subject == "self"


@pytest.mark.parametrize("text", ["我吃布洛芬，我妈妈咳嗽三天。", "之前我服用布洛芬，现在给我妈妈问这个药。", "换个对象，现在问朋友这个药。"])
def test_mixed_or_unidentified_subject_switch_cannot_assign_facts_to_previous_person(text):
    tracker = SubjectTracker()
    assert tracker.resolve(Question(text=text), "session").ambiguous


def test_redis_fields_survive_summary_and_runtime_restart_with_explicit_subject():
    async def scenario():
        class Redis:
            def __init__(self): self.rows, self.ttls = {}, []
            async def get(self, key): return self.rows.get(key)
            async def set(self, key, value, ex): self.rows[key] = value; self.ttls.append(ex)
        redis, model = Redis(), ContextModel()
        runtime = await fixture(model)
        runtime.memory = SessionMemory(redis=redis, model=model)
        first = await ask(runtime, "布洛芬公开资料", subject="mother")
        for _ in range(5): await ask(runtime, "这个药资料呢", subject="mother")
        assert all(ttl == 86400 for ttl in redis.ttls)
        restarted = await fixture(ContextModel())
        restarted.memory = SessionMemory(redis=redis, model=ContextModel())
        reply = await ask(restarted, "这个药的资料呢", subject="mother")
        assert reply.subject == "mother" and reply.intent.entities["medication_name"].value == "布洛芬"
        assert reply.intent.entities["medication_name"].source.turn_id == first.request_id
        assert not (await restarted.memory.read(scope_key("synthetic", "dialogue", "self")))["messages"]
    run(scenario())


def test_corrupt_storage_state_is_degraded_to_current_turn_not_http_failure():
    async def scenario():
        runtime = await fixture(ContextModel())
        class BrokenMemory:
            available = True
            async def read(self, scope): return ["not-a-session"]
            async def remember(self, *args): raise ConnectionError()
        runtime.memory = BrokenMemory()
        reply = await ask(runtime, "这个药是什么")
        assert reply.degraded and reply.intent.entities == {}
        assert reply.agent_results[0].missing_fields == ["medication_name"]
    run(scenario())


def test_unknown_response_can_only_clear_a_field_the_previous_turn_requested():
    source = user_sources("我不知道", {"last_missing_fields":["medication_name"]}, "current", "self")[-1]
    candidate = {"name":"medication_name", "value":None, "state":"unknown", "quote":"我不知道", "turn_id":"current"}
    assert ground_update(candidate, {"current":source}).state == "unknown"
    assert ground_update({**candidate, "name":"age_group"}, {"current":source}) is None


def test_long_valid_evidence_is_stored_as_bounded_literal_window():
    text = "背景"*700+"我35岁，想整理信息。"
    source = user_sources(text, {}, "current", "self")[-1]
    fact = ground_update({"name":"age_group", "value":"35岁", "state":"provided", "quote":text[:-1], "turn_id":"current"}, {"current":source})
    assert fact is not None and len(fact.source.quote) <= 600
    assert fact.source.quote in text and "35岁" in fact.source.quote


@pytest.mark.parametrize("intent,role,text", [
    (Intent.MEDICINE, "drug_reference", "不需要具体制剂，只介绍公开概览"),
    (Intent.VISIT, "visit_guide", "不查这家医院，只整理通用准备清单"),
])
def test_negated_task_scope_does_not_create_unnecessary_required_fields(intent, role, text):
    understanding = Understanding(primary=intent, tasks=[TaskUnderstanding(intent=intent, goal=text)])
    details = role_details(understanding, role)
    assert "query_type" not in details and "institution_specific" not in details


@pytest.mark.parametrize("field,value", [("turn_counter", "bad"), ("last_intent", {}), ("last_missing_fields", 1)])
def test_invalid_persisted_context_metadata_is_rejected(field, value):
    async def scenario():
        class Redis:
            async def get(self, key): return json.dumps({"messages":[], "summary":"", field:value})
        with pytest.raises(ValueError, match="invalid_session"):
            await SessionMemory(redis=Redis()).read("scope")
    run(scenario())


@pytest.mark.parametrize("text,subject", [("我母亲的药品资料", "mother"), ("我父亲的报告", "father")])
def test_explicit_family_possession_selects_its_own_subject(text, subject):
    assert SubjectTracker().resolve(Question(text=text), "session").subject == subject


@pytest.mark.parametrize("text", ["请介绍症状模板如何记录咳嗽三天", "比如我有头痛"])
def test_general_examples_cannot_be_promoted_to_confirmed_symptoms(text):
    async def scenario():
        reply = await IntentRouter(ContextModel()).understand(text, turn_id="test")
        assert "symptom" not in reply.entities and "onset" not in reply.entities
    run(scenario())


def test_person_age_does_not_block_general_symptom_information_without_complaint():
    async def scenario():
        understanding = await IntentRouter(ContextModel()).understand("我是成年人，请介绍症状模板。", turn_id="test")
        assert understanding.entities["age_group"].value == "成年人"
        assert "age_group" not in role_details(understanding, "symptom_guide")
    run(scenario())


def test_a_stored_user_message_without_a_turn_id_is_not_a_source():
    state = {"messages":[{"role":"user", "content":"我咳嗽三天"},
                         {"role":"user", "content":"昨天开始发烧", "turn_id":"t1", "turn_index":1}]}
    sources = user_sources("还有点头痛", state, "t2", "self")
    assert [source["turn_id"] for source in sources] == ["t1", "t2"]
