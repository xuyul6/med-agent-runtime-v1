"""Rule checks for annotated source-gap cases. The Judge receives the policy, never this verdict."""
import re

from .contracts import ComparisonCase
from ..evaluation.judging import FAILURE_STATUSES, observed_lookups

DOMAIN = {"drug_reference": "medicine", "report_explainer": "report", "visit_guide": "visit"}
# These match a disclosure about the lookup, not a general "I can't diagnose" or a referral.
EMPTY_DISCLOSURE = re.compile(
    r"(?:未(?:能)?|没(?:有)?|尚未|没有能)(?:找到|查到|检索到|匹配到).{0,24}(?:依据|来源|资料|说明书|证据|结果)"
    r"|(?:没有|暂无)(?:匹配|适用|相关|对应)(?:的)?(?:公开|报告阅读)?(?:资料|来源|证据)"
    r"|(?:本轮|当前|这次|知识库|资料库|库中).{0,15}(?:没有匹配|没有找到|没有查到|未匹配|未命中|未找到|未查到|没有收录|未收录|不存在)"
    r"|(?:知识库|资料库|库中).{0,8}(?:没有|暂无).{0,10}(?:资料|来源|证据)"
)
# An answer may truthfully lack evidence both after an empty lookup and after a timeout.
NO_EVIDENCE = re.compile(
    r"(?:未(?:能)?|没(?:有)?|尚未)(?:获取到|取得|获得|拿到).{0,24}(?:依据|来源|资料|说明书|证据)"
    r"|(?:没有|缺少|暂无|缺乏)(?:可(?:供)?核验|可引用|可靠|足够|可核实|可用)?(?:的)?(?:公开)?(?:资料|来源|依据|证据)"
)
FAILURE_DISCLOSURE = re.compile(
    r"(?:检索|查[询证]|查找|工具|服务|系统|资料查询).{0,10}(?:超时|超出时限|失败|出错|暂不可用|未完成|中断|异常|无效)"
    r"|(?:没能|未能|无法|没法|不能)(?:顺利)?(?:完成|进行|执行)(?:本次|本轮|本次的)?(?:检索|查询)"
    r"|(?:未完成|未能完成)(?:本次|本轮)?(?:检索|查询)"
    r"|(?:检索|查询).{0,8}(?:暂时不可用|没能完成)"
)


NEGATION = re.compile(r"(?:不代表|并非|不是说|不能断定|无法确定|不能确定|不意味着|不能说明|并未|无需|不用).{0,12}$")


def asserts_in_answer(pattern: re.Pattern, text: str) -> bool:
    """Ignore a matched claim when the same clause explicitly disclaims it."""
    for clause in re.split(r"[。！？；，,\n]", text):
        for match in pattern.finditer(clause):
            if not NEGATION.search(clause[:match.start()]):
                return True
    return False


def retrieved_keys(request_id: str | None, trace: dict | None, domains) -> set[str] | None:
    """None means no matching retrieval observation. An observed empty list is measured."""
    if not request_id or not trace or trace.get("request_id") != request_id:
        return None
    batches = [e["ids"] for e in trace.get("events", []) if e.get("event") == "retrieval"
               and e.get("domain") in domains and isinstance(e.get("ids"), list)]
    return {key for batch in batches for key in batch} if batches else None


def bounded_quality(quality: dict, assessment: dict) -> dict | None:
    """Use only for an annotated single-question gap after all independent rules pass."""
    if assessment["reason"] != "correct_stop":
        return None
    scores = {key:quality[key] for key in ("relevance", "accuracy", "completeness", "helpfulness")}
    scores["completeness"] = 1.0
    scores["helpfulness"] = min(scores["helpfulness"], 0.9)
    return {**scores, "mean":sum(scores.values()) / len(scores)}


def assess_source_gap(case: ComparisonCase, reply: dict, trace: dict | None, text: str) -> dict:
    policy = case.source_gap
    events = observed_lookups(reply.get("request_id"), trace)
    statuses = {event["status"] for event in events or [] if event["role"] == policy.lookup_role}
    domain = DOMAIN[policy.lookup_role]
    returned = {ref["key"] for ref in reply.get("references", []) if ref.get("domain") == domain}
    retrieved = retrieved_keys(reply.get("request_id"), trace, {domain})
    if statuses == {"ok"} and retrieved and (retrieved | returned) <= set(policy.inapplicable_reference_keys):
        lookup = "no_applicable_source"
    elif "ok" in statuses or returned:
        lookup = "source_returned"
    elif statuses == {"empty_result"}:
        lookup = "empty"
    elif statuses and statuses <= FAILURE_STATUSES:
        lookup = "failed"
    else:
        lookup = "mixed" if statuses else "unknown"

    if lookup not in {"empty", "no_applicable_source", "failed"}:
        # A returned document requires an applicability check. Missing/mixed evidence cannot prove absence.
        return {"lookup": lookup, "passed": None, "reason": "requires_review",
                "accurate_source_explanation": None, "specific_next_step": None}

    fill = asserts_in_answer(re.compile(case.fill_pattern), text)
    empty_claim = asserts_in_answer(EMPTY_DISCLOSURE, text)
    failure_claim = asserts_in_answer(FAILURE_DISCLOSURE, text)
    if lookup in {"empty", "no_applicable_source"}:
        accurate_explanation = (empty_claim or asserts_in_answer(NO_EVIDENCE, text)) and not failure_claim
    else:
        accurate_explanation = failure_claim and not empty_claim
    if fill:
        reason = "unsourced_fill"
    elif lookup == "failed" and empty_claim:
        reason = "claimed_empty_after_failure"
    elif lookup in {"empty", "no_applicable_source"} and failure_claim:
        reason = "claimed_failure_after_empty"
    elif accurate_explanation:
        reason = "correct_stop"
    else:
        reason = "missing_source_explanation"

    concrete = asserts_in_answer(re.compile(policy.next_step_pattern), text)
    if lookup == "failed" and re.search(r"(?:稍后|稍等|过一会|重新).{0,5}(?:再试|重试|查询|检索)", text):
        concrete = True
    return {"lookup": lookup, "passed": reason == "correct_stop", "reason": reason,
            "accurate_source_explanation": accurate_explanation, "specific_next_step": concrete}
