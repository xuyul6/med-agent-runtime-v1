"""Engineering regression only: no network, Judge or medical quality estimate."""
from dataclasses import replace
import json
import pytest
from pydantic import ValidationError
from medagent.core.contracts import Intent, Reference
from medagent.evaluation.runner import evaluate, resolve_cases
from medagent.evaluation.contracts import EvaluationRequest
from medagent.knowledge.medication_evidence import rejection_reason, validate_output_identity
from medagent.agents.profiles import DEFAULT_REGISTRY
from medagent.knowledge.retrieval import catalog_updates
from medagent.tools.registry import DEFAULT_TOOLS, ToolBudget, ToolContext, ToolExecutor, ToolRegistry, ToolResultError
from medagent.core.config import ROOT
from support.harness import fixture, run
from support.medication import EvidenceModel, LABEL_KEY, SCOPED, references, request


def test_catalog_provenance_keeps_overview_and_label_distinct():
    refs = references()
    overview, label = refs["ibuprofen-overview"], refs[LABEL_KEY]
    assert overview.provenance.document_type == "public_overview"
    assert label.provenance.document_type == "product_label"
    assert str(label.provenance.updated_on) == "2026-03-12"
    assert str(label.provenance.published_on) == "2026-03-16"
    assert str(label.provenance.checked_on) == "2026-10-27"
    assert label.medication_scope.region == "US" and label.medication_scope.strength == "200mg"
    assert "12岁" in label.medication_scope.label_population
    assert "非完整" in label.scope and "原文权利" in label.provenance.use_conditions


@pytest.mark.parametrize("field,value", [("region","中国"), ("population","儿童"), ("population","孕妇"),
    ("formulation","缓释胶囊"), ("strength","400mg"), ("product_name","Nurofen"), ("medication_name","阿莫西林")])
def test_wrong_scope_is_excluded(field, value):
    assert rejection_reason(references()[LABEL_KEY], request(**{field:value})) is not None


@pytest.mark.parametrize("field", ["region","population","formulation","strength","product_name"])
def test_unknown_scope_never_inherits_label_identity(field):
    assert rejection_reason(references()[LABEL_KEY], request(**{field:None})) == "missing_"+field


def test_scope_accepts_only_document_match_not_individual_suitability():
    assert rejection_reason(references()[LABEL_KEY], request()) is None
    assert rejection_reason(references()[LABEL_KEY], request(strength="200毫克")) is None
    assert rejection_reason(references()["ibuprofen-overview"], request()) == "overview_is_not_product_label"


@pytest.mark.parametrize("source_region,query_region", [
    ("CN", "中国"), ("中国", "CN"), ("cn", "中国大陆"), ("中国大陆", "cn"),
])
def test_china_region_alias_matches_in_both_directions(source_region, query_region):
    source = references()[LABEL_KEY].model_copy(deep=True)
    source.medication_scope.region = source_region
    assert rejection_reason(source, request(region=query_region)) is None


@pytest.mark.parametrize("source_region,query_region", [("CN", "美国"), ("中国", "US"), ("US", "中国")])
def test_china_region_alias_still_rejects_other_regions(source_region, query_region):
    source = references()[LABEL_KEY].model_copy(deep=True)
    source.medication_scope.region = source_region
    assert rejection_reason(source, request(region=query_region)) == "region_mismatch"


def test_untyped_source_and_incomplete_label_are_not_product_evidence():
    raw = references()[LABEL_KEY].model_dump()
    raw["provenance"] = raw["medication_scope"] = None
    assert rejection_reason(Reference.model_validate(raw), request()) == "unverified_source_scope"
    raw = references()[LABEL_KEY].model_dump()
    raw["medication_scope"]["strength"] = None
    with pytest.raises(ValidationError): Reference.model_validate(raw)


def test_keyword_rewrite_and_tool_arguments_cannot_change_scope_or_poison_cache():
    async def scenario():
        runtime = await fixture(EvidenceModel())
        for changes, expected in [({},[LABEL_KEY]), ({"region":"中国"},[]), ({},[LABEL_KEY])]:
            context = ToolContext(DEFAULT_REGISTRY["drug_reference"], request(**changes), runtime.index)
            result = await runtime.tools.call("lookup_reference", {"query":"布洛芬 美国 Advil 200mg 成人", "domain":"medicine"}, context, ToolBudget())
            assert [ref.key for ref in result.documents] == expected
            scope = await runtime.tools.call("check_medication_scope", {}, context, ToolBudget())
            assert scope.applicability == ("needs_verification" if expected else "no_source")
            if expected: assert scope.sources[0].provenance.source_version.endswith("version 38")
        assert len(runtime.index.cache) == 2
    run(scenario())


def test_native_tool_result_validation_rejects_bypassed_scope_filter():
    async def malicious(context, arguments): return {"documents":[references()[LABEL_KEY]]}
    async def scenario():
        specs = [replace(spec, handler=malicious) if spec.name == "lookup_reference" else spec for spec in DEFAULT_TOOLS.specs.values()]
        executor = ToolExecutor(ToolRegistry(specs))
        context = ToolContext(DEFAULT_REGISTRY["drug_reference"], request(region="中国"))
        with pytest.raises(ToolResultError):
            await executor.call("lookup_reference", {"query":"布洛芬", "domain":"medicine"}, context, ToolBudget())
        assert not context.references
    run(scenario())


@pytest.mark.parametrize("mode",["selective","full"])
def test_actual_agent_receives_only_matching_evidence_and_consistent_citations(mode):
    async def scenario():
        model = EvidenceModel()
        runtime = await fixture(model)
        result = await runtime.agents["drug_reference"][0].run("核对布洛芬标签",Intent.MEDICINE,{},mode,details=SCOPED)
        assert result.execution.status == "answered"
        assert [ref.key for ref in result.references] == [LABEL_KEY] == result.answer.reference_keys
        assert [ref["key"] for ref in model.contexts[-1]["references"]] == [LABEL_KEY]
        assert result.execution.payload["medication_identity"]["region"] == "美国"
    run(scenario())


@pytest.mark.parametrize("field",["product_name","strength","region"])
def test_model_cannot_fill_unknown_identity_from_source(field):
    with pytest.raises(ValueError,match="unconfirmed_medication_identity"):
        validate_output_identity({"payload":{"medication_identity":{field:SCOPED[field]}}}, request(**{field:None}))


def test_missing_or_inapplicable_source_returns_clarification_without_generation():
    async def scenario():
        model = EvidenceModel()
        runtime = await fixture(model)
        agent = runtime.agents["drug_reference"][0]
        for details, gaps in [(SCOPED | {"region":"中国"},["source"]), (SCOPED | {"strength":None},["strength"])]:
            result = await agent.run("核对布洛芬标签",Intent.MEDICINE,{},"full",details=details)
            assert result.execution.status == "needs_clarification" and result.execution.missing_fields == gaps
            assert not result.answer.reference_keys and not result.references
        assert not model.contexts
    run(scenario())


def test_catalog_enrichment_preserves_user_edits_and_changed_versions():
    refs = references()
    ref = refs["ibuprofen-overview"]
    legacy = ref.model_copy(update={"provenance":None,"medication_scope":None})
    assert catalog_updates([ref],{ref.key:legacy}) == [ref]
    for patch in [{"text":"用户审核内容"},{"version":"user-version"},{"approved":False}]:
        assert catalog_updates([ref],{ref.key:legacy.model_copy(update=patch)}) == []
    assert catalog_updates([ref],{ref.key:ref}) == []


def test_a_stored_record_with_other_text_is_never_replaced_automatically():
    ref = references()["ibuprofen-overview"]
    old = ref.model_copy(update={"version":"summary-2026-10-26", "provenance":None,"medication_scope":None,
        "text":ref.text.replace("本条目仅保存该概览的简短导读，不包含具体品牌制剂的完整禁忌或相互作用资料。",
                                "本索引仅保存简短导读，没有收录具体地区或品牌制剂说明书，也没有完整禁忌和相互作用数据库。")})
    assert catalog_updates([ref],{old.key:old}) == []
    assert catalog_updates([ref],{old.key:old.model_copy(update={"approved":False})}) == []
    assert catalog_updates([ref],{old.key:old.model_copy(update={"text":old.text+"用户注释"})}) == []


def test_original_failure_and_quality_threshold_are_preserved_in_new_suite():
    origin = json.loads((ROOT/"resources/medication_regression_origin.json").read_text())
    req = EvaluationRequest.model_validate_json((ROOT/"resources/medication-regression.json").read_text())
    assert req.dialog_cases[0].turns[0].message == origin["question"] == "查询布洛芬药品资料"
    assert origin["quality_policy"]["pass_threshold"] == .75
    assert origin["historical_runs"][-1]["scores_as_recorded"]["overall"] == .7
    assert origin["historical_runs"][0]["judge_failed"] is True
    assert origin["current_online_status"] == "not_run"
    _, _, cases = resolve_cases(EvaluationRequest())
    assert any(case.id == req.dialog_cases[0].id and case.turns[0].message == origin["question"] for case in cases)


def test_original_case_and_scoped_followup_use_production_chain_offline():
    async def scenario():
        runtime = await fixture(EvidenceModel())
        data = json.loads((ROOT/"resources/medication-regression.json").read_text())
        data["judge"] = False
        report = await evaluate(runtime, EvaluationRequest.model_validate(data))
        assert report["complete"] and report["executed"] == report["passed"] == 4
        assert all(case["quality"] is None for case in report["cases"])
        assert report["cases"][0]["turns"][0]["reply"]["agent_results"][0]["status"] == "needs_clarification"
        assert report["cases"][1]["turns"][1]["reply"]["answer"]["reference_keys"] == [LABEL_KEY]
    run(scenario())


async def offline_report():
    runtime = await fixture(EvidenceModel())
    data = json.loads((ROOT/"resources/medication-regression.json").read_text())
    data["judge"] = False
    report = await evaluate(runtime, EvaluationRequest.model_validate(data))
    report["verification"] = {
        "kind":"offline_engineering_regression", "model":"EvidenceModel (deterministic fake)",
        "paid_model_calls":0, "online_regression":"not_run", "original_quality_failure":"unresolved_pending_online_regression",
        "original_threshold":.75, "historical_valid_score":.7,
        "checks":{
            "all_requested_groups_executed":report["executed"] == 4 and report["complete"],
            "all_engineering_rules_pass":report["passed"] == 4,
            "original_question_unchanged":data["dialog_cases"][0]["turns"][0]["message"] == "查询布洛芬药品资料",
            "quality_not_fabricated":all(case["quality"] is None for case in report["cases"]),
            "unknown_formulation_clarified":report["cases"][2]["turns"][0]["reply"]["agent_results"][0]["status"] == "needs_clarification",
            "wrong_region_has_no_citations":not report["cases"][3]["turns"][0]["reply"]["answer"]["reference_keys"],
            "scoped_label_cited":report["cases"][1]["turns"][1]["reply"]["answer"]["reference_keys"] == [LABEL_KEY],
        },
        "source_records":[ref.model_dump(mode="json",exclude={"text"}) for ref in runtime.index.documents.values() if ref.domain == Intent.MEDICINE],
    }
    return report


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Offline medication engineering regression; never uses configured API keys")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists(): raise FileExistsError("preserve_existing_evaluation_report")
    report = run(offline_report())
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        handle.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report["verification"]["checks"],ensure_ascii=False))
    if not all(report["verification"]["checks"].values()): raise SystemExit(1)
