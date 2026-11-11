"""Report role gates: when a personal report lacks its item, and quoting the hash-pinned range summary meanwhile."""
import hashlib
import json
import re
from ..core.contracts import Intent
from ..understanding.conversation import clauses, positive_request
from .checks import check_boundaries

# The deterministic mixed-report path quotes only this reviewed summary, unchanged.
REPORT_RANGE_SOURCE_HASH = "bda2d03ef8df0772268958a0aedb9e1689a9b94da1e6312576380db194b40481"
REPORT_RANGE_QUESTION = r"参考范围(?:的)?(?:有什么意义|有何意义|是什么意思|是什么|什么含义|含义|定义)"


def mixed_personal_report_without_item(text: str) -> bool:
    """Recognize a narrow mixed question without converting user text into a report item."""
    if not positive_request(text, REPORT_RANGE_QUESTION):
        return False
    return personal_report_without_item(text, require_judgment=False)


def personal_report_without_item(text: str, *, require_judgment: bool = True) -> bool:
    """Identify a bounded request to interpret an unspecified abnormal personal result."""
    if require_judgment and not re.search(r"判断|诊断|生病|有病|什么病|严重|怎么办|怎么看|意味|是不是|解读|分析", text):
        return False
    if re.search(r"(?:不要|不需要|不用|无需|不必|不想).{0,12}(?:解读|分析|判断|看|解释).{0,8}(?:个人|指标|报告|结果|它)", text):
        return False
    for clause in clauses(text):
        if re.search(r"如果|假如|假设|例如|举例|科普|并非|不是|没有|未出现|未发现", clause):
            continue
        if re.search(r"(?:报告|体检|化验|检验).{0,24}(?:指标|项目|结果).{0,12}(?:偏高|偏低|异常|升高|降低|超出)", clause):
            return True
    return False


REPORT_FIELDS = ("report_item", "value", "unit", "reference_range")
# Not hash-pinned: its sentences reach an answer through checks.insert_range_sentences, like any approved report source.
RANGE_DEFINITION_KEY = "lab-reference-range-definition"


def reviewed_source(references: dict, key: str, expected_hash: str):
    """The approved report summary under `key`, only if it is exactly the reviewed version."""
    reference = references.get(key)
    if reference is None or not reference.approved or reference.domain != Intent.REPORT:
        return None
    digest = hashlib.sha256(json.dumps([reference.model_dump(mode="json")], ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return reference if digest == expected_hash else None


def missing_item_reason(current: str, memory: dict) -> str | None:
    """Why a report request without a parsed item has to ask for the item first."""
    if mixed_personal_report_without_item(current):
        return "mixed_personal_report_without_item"
    if personal_report_without_item(current):
        return "personal_report_without_item"
    grounded = memory.get("grounded_fields", {})
    unknown = {name for name in REPORT_FIELDS
        if isinstance(grounded, dict) and isinstance(grounded.get(name), dict)
        and grounded[name].get("state") == "unknown"
        and isinstance(grounded[name].get("source"), dict)
        and grounded[name]["source"].get("origin") == "current_user"}
    if "report_item" in unknown and len(unknown) >= 2:
        return "explicit_unknown_report_fields"
    return None


def asks_range_meaning(current: str) -> bool:
    return positive_request(current, REPORT_RANGE_QUESTION)


async def range_meaning_answer(agent, workflow, context, budget, missing: list[str], report_payload: dict):
    """Explain the generic term from the reviewed summary while personal fields stay missing."""
    await agent.tools.call("lookup_reference", {"query":"参考范围", "domain":"report"}, context, budget)
    reference = reviewed_source(context.references, "lab-context", REPORT_RANGE_SOURCE_HASH)
    workflow.complete("prepare_evidence", "evidence", reference_count=1 if reference is not None else 0)
    if reference is None:
        return agent.limited_result(workflow, "needs_clarification", [*missing, "source"],
            "个人报告的原始字段尚不完整；当前没有可用于通用参考范围说明的匹配资料。", {}, payload=report_payload)
    # Only quote the reviewed source: no personal result or model inference is added.
    excerpt = reference.text
    check_boundaries(excerpt)
    payload = {**report_payload, "term_explanations":[{"text":excerpt, "reference_keys":[reference.key]}]}
    return agent.limited_result(workflow, "needs_clarification", missing,
        "以下是当前资料对报告阅读的通用说明："+excerpt+"\n个人报告的原始字段仍需核对。",
        {reference.key:reference}, payload=payload, reference_keys=[reference.key])


def unrequested_range_definition(current: str, memory: dict, references: dict) -> bool:
    """A first-turn request to organize a report does not need the range definition when other sources apply."""
    has_prior_user = any(isinstance(row, dict) and row.get("role") == "user"
                         for row in memory.get("messages", []))
    asks_range = re.search(r"参考|范围|正常值|区间|上下限|上界|下界|界限|百分位|阈值|统计|分布|\brange\b|reference\s*[-_]?\s*(?:range|interval)", current, re.I)
    # Remove only an explicit refusal of personal analysis from the gating
    # text. The original request remains intact in the model context.
    screening_text = re.sub(r"(?:不要|不需要|无需|不想)(?:你|做|进行)?(?:分析|判断|解读|解释|诊断)(?:我的|个人)?病情", "", current)
    workflow_only = (re.search(r"整理|对应|配对|表格|记录", current)
        and not re.search(r"分析|判断|诊断|确诊|疾病|生病|患病|病情|异常|偏高|偏低|正常|治疗|开药|用药|(?:是否|会不会|是不是).{0,6}(?:病|异常)|如何(?:确定|计算)|(?:统计|计算).{0,8}(?:依据|方法)", screening_text))
    other_report_source = any(key != RANGE_DEFINITION_KEY and ref.approved and ref.domain == Intent.REPORT
                              for key, ref in references.items())
    return bool(not has_prior_user and not asks_range and workflow_only and other_report_source)
