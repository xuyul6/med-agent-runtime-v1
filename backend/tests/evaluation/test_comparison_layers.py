"""A layer picks its case set and arms; cases carry the rules that score them without a model."""
import pytest

from medagent.comparison.contracts import ARMS, LAYERS, ComparisonCase, ComparisonRequest
from medagent.comparison.cli import build_plan
from medagent.comparison.runner import schedule


def case(**values):
    return ComparisonCase.model_validate({"id":"x", "category":"compound", "target":{"message":"问题"}, **values})


def stability_case(message="带什么，再查布洛芬", fill_pattern="身份证", gap_pattern="没能", category="stability_always",
                   fault=None, **values):
    return case(category=category, target={"message":message}, gap_pattern=gap_pattern, fill_pattern=fill_pattern,
                faults=[fault or {"kind":"generation_failure_always", "domain":"visit"}], **values)


def outage_case():
    return case(category="stability_outage", faults=[{"kind":"generation_failure_always"}])


def test_every_layer_names_its_set_and_arms():
    assert set(LAYERS) == {"daily", "compound", "stability", "safety", "permission"}
    assert LAYERS["compound"]["arms"] == ARMS
    # The open single agent exists only in the permission layer.
    assert all(set(layer["arms"]) <= set(ARMS) | {"single_agent_open"} for layer in LAYERS.values())
    assert LAYERS["permission"]["arms"] == ("multi_agent", "single_agent", "single_agent_open")


def test_no_layer_keeps_the_old_arms():
    assert ComparisonRequest(repetitions=3, cases=[case()]).arm_names == ARMS


def test_layer_selects_its_arms_and_rotates_them():
    request = ComparisonRequest(layer="daily", repetitions=4, cases=[case()])
    assert request.arm_names == ("multi_agent", "single_agent")
    rows = schedule(request, request.cases)
    assert {row["order"][0] for row in rows} == set(request.arm_names)
    assert all(set(row["arms"]) == set(request.arm_names) for row in rows)


def test_plan_counts_only_the_layer_arms():
    request = ComparisonRequest(layer="daily", repetitions=4, cases=[case()])
    plan = build_plan(request)
    assert plan["planned_targets"] == 8 and plan["protocol"]["version"] == "architecture-comparison-4"


def test_repetitions_must_rotate_the_layer_arms():
    with pytest.raises(ValueError):
        ComparisonRequest(layer="daily", repetitions=3, cases=[case()])


def test_safety_layer_runs_without_judge():
    with pytest.raises(ValueError):
        ComparisonRequest(layer="safety", repetitions=4, judge=True, cases=[case()])
    assert ComparisonRequest(layer="safety", repetitions=4, judge=False, cases=[case()])


def test_fill_pattern_cannot_match_the_question():
    with pytest.raises(ValueError):
        stability_case(message="要带身份证吗", fill_pattern="身份证")


def test_a_pattern_has_to_compile():
    with pytest.raises(ValueError):
        stability_case(gap_pattern="(")


def test_a_stability_case_needs_both_patterns():
    with pytest.raises(ValueError):
        case(category="stability_always", faults=[{"kind":"generation_failure_always", "domain":"visit"}])


@pytest.mark.parametrize("category", ["boundary", "injection"])
def test_a_safety_case_needs_its_forbidden_pattern(category):
    with pytest.raises(ValueError):
        case(category=category)
    assert case(category=category, forbidden_pattern="每次")


def test_generation_fault_only_in_stability_categories():
    with pytest.raises(ValueError):
        case(category="compound_parallel", faults=[{"kind":"generation_failure_once", "domain":"visit"}])


def test_a_category_takes_only_its_own_fault():
    with pytest.raises(ValueError):
        stability_case(category="stability_once", fault={"kind":"generation_failure_always", "domain":"visit"})
    with pytest.raises(ValueError):
        stability_case(category="stability_tool", fault={"kind":"generation_failure_once", "domain":"visit"})
    assert stability_case(category="stability_tool", fault={"kind":"tool_timeout", "domain":"visit"})


def test_outage_is_the_only_fault_without_a_domain():
    assert outage_case()
    with pytest.raises(ValueError):
        stability_case(category="stability_once", fault={"kind":"generation_failure_once"})
    with pytest.raises(ValueError):
        case(category="stability_outage", faults=[{"kind":"generation_failure_always", "domain":"visit"}])


def test_a_faulted_domain_cannot_be_required_coverage():
    with pytest.raises(ValueError):
        stability_case(reference_domains=["visit"])
