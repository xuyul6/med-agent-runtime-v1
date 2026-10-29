"""Report role shortcuts that quote hash-pinned reviewed summaries directly, without the model."""
import hashlib
import json
import re
from ..core.contracts import Intent
from ..core.observability import event
from ..understanding.conversation import clauses, positive_request
from .checks import check_boundaries

# The deterministic mixed-report path quotes only this reviewed summary, unchanged.
REPORT_RANGE_SOURCE_HASH = "bda2d03ef8df0772268958a0aedb9e1689a9b94da1e6312576380db194b40481"
REPORT_RANGE_DEFINITION_HASH = "f506943119c77e64b1f33de9bf953309203f9516a666fca18b427d8324475f06"
REPORT_UNIT_SOURCE_HASH = "2f54021e88977383e66d6c25af8df122085d16381c842bd13104d0f4eb116758"
REPORT_STRUCTURE_SOURCE_HASH = "2be3b39f9eed112c4abb3293d3fbeab0559e704ad2907c86756af028adf33043"
REPORT_RANGE_QUESTION = r"参考范围(?:的)?(?:有什么意义|有何意义|是什么意思|是什么|什么含义|含义|定义)"


def general_report_source_request(text: str) -> str | None:
    """Recognize only complete, generic requests covered by the reviewed summary."""
    value = re.sub(r"\s+", "", text).rstrip("。！？!?")
    prefix = r"(?:(?:我)?(?:想|在)|请|麻烦|能否|可以)?(?:解释|说明|介绍|了解|说说|学习)?"
    report = r"(?:(?:化验|检验|体检|检查)?报告(?:里面的|里的|中的|上的|的|里|中|上)?)?"
    meaning = r"参考范围(?:的)?(?:有什么意义|有何意义|是什么意思|是什么|什么含义|含义|定义|代表什么)"
    followup = (r"(?:[,，](?:以及|和|并且)?(?:请)?(?:解释)?(?:为什么|为何)"
                r"(?:单个|单项|一个)(?:结果|指标)(?:不能|无法)(?:独立|单独|直接)?"
                r"(?:确定|判断|说明|反映)(?:健康状态|健康状况|健康情况|健康|病情))?")
    if re.fullmatch(prefix+report+meaning+followup, value): return "reference_range"
    combined = (r"(?:(?:请|我想|我希望)(?:说明|介绍|了解)?)?"
                r"(?:化验|检验|体检|检查)?报告(?:的)?(?:结果|项目|数值)?(?:如何|怎么)(?:整理|记录)"
                r"[？?，,；;。]+"
                r"(?:(?:不要|不需要|无需)(?:你|做|进行)?(?:分析|判断|解读|解释|诊断)(?:我的|个人)?病情[，,；;。]+)?"
                r"(?:也|再|并|同时)?(?:请)?(?:解释|说明|介绍)(?:一下)?参考范围(?:的含义|的意义|是什么意思)?")
    if re.fullmatch(combined, value): return "report_structure_and_range"
    mention = (r"(?:我)?(?:还|也)?(?:看到|注意到|发现)(?:了)?"
               r"(?:化验|检验|体检|检查)?报告(?:里面|里|中|上)?"
               r"(?:列有|列出了?|有|包含)(?:了)?参考范围")
    if re.fullmatch(mention, value): return "reference_range_mention"
    limitations = (r"(?:(?:我)?(?:没有要你|不需要你|不想让你)(?:判断|分析|解读)(?:个人病情|我的病情|个人结果)"
                   r"[,，](?:只|仅)(?:是)?(?:希望|想|要))?"
                   r"(?:请|我想|我希望)?(?:理解|了解|解释|说明)(?:化验|检验|体检|检查)?"
                   r"报告(?:阅读|解读)(?:的)?(?:限制|局限|局限性)")
    if re.fullmatch(limitations, value): return "report_reading_limitations"
    structure = (r"(?:我)?(?:知道|了解|注意到)(?:了)?(?:化验|检验|体检|检查)?报告"
                 r"(?:通常|一般|会|也会|都)?(?:列出|列有|包含|有)(?:了)?(?:项目名称|检查项目|检验项目)")
    if re.fullmatch(structure, value): return "report_structure_mention"
    unit_followup = (r"(?:我)?(?:知道|了解|注意到)(?:了)?(?:报告(?:上的|中的|里的)?)?"
                     r"(?:原始)?(?:结果|数值)(?:的)?(?:旁边|后面|附近)(?:还|也)?(?:有|写有|标有)(?:了)?单位")
    if re.fullmatch(unit_followup, value): return "report_unit_followup"
    learning = (r"(?:我在|我想|我)?(?:学习|了解)(?:阅读|理解|查看)?(?:化验|检验|体检|检查)?报告"
                r"(?:[,，](?:只|仅)(?:想)?(?:了解|学习)(?:通用|一般|基础)(?:概念|知识))?")
    if re.fullmatch(learning, value): return "report_learning"
    return None


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
DIRECT_SOURCES = {
    "report_unit_followup": ("lab-unit-overview", REPORT_UNIT_SOURCE_HASH),
    "report_structure_mention": ("lab-structure-overview", REPORT_STRUCTURE_SOURCE_HASH),
    "report_reading_limitations": ("lab-context", REPORT_RANGE_SOURCE_HASH),
}
RANGE_DEFINITION = ("lab-reference-range-definition", REPORT_RANGE_DEFINITION_HASH)


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
    other_report_source = any(key != RANGE_DEFINITION[0] and ref.approved and ref.domain == Intent.REPORT
                              for key, ref in references.items())
    return bool(not has_prior_user and not asks_range and workflow_only and not general_report_source_request(current)
                and other_report_source)


def direct_answer(agent, workflow, current: str, memory: dict, subject_changed: bool, references: dict, report_payload: dict):
    """Quote a reviewed summary for a complete generic request; None means the model path continues."""
    category = general_report_source_request(current)
    if category == "report_structure_and_range":
        selected = [reviewed_source(references, "lab-context", REPORT_RANGE_SOURCE_HASH), reviewed_source(references, *RANGE_DEFINITION)]
        if all(ref is not None for ref in selected):
            event("report_source_direct", role=agent.role, category=category, source_count=2)
            organization = ("可以按项目名称、结果数值、单位和本报告参考范围逐项记录；"
                            "如需解释个人结果，可交由医护人员结合实际情况核对。")
            payload = {**report_payload, "term_explanations":[
                {"text":organization, "reference_keys":[selected[0].key]},
                {"text":selected[1].text, "reference_keys":[selected[1].key]}]}
            result = agent.limited_result(workflow, "answered", [], organization+"\n\n"+selected[1].text,
                {ref.key:ref for ref in selected}, payload=payload, reference_keys=[ref.key for ref in selected])
            result.answer.limitations.append("仅说明通用报告整理和参考范围，不判断个人结果。")
            result.model_dependency_exercised = False
            return result
        category = None
    if category == "report_unit_followup":
        recent_users = [row.get("content", "") for row in memory.get("messages", [])[-4:]
                        if isinstance(row, dict) and row.get("role") == "user"]
        if subject_changed or not any(isinstance(value, str) and "报告" in value for value in recent_users):
            category = None
    if category is None:
        return None
    reference = reviewed_source(references, *DIRECT_SOURCES.get(category, RANGE_DEFINITION))
    if reference is None:
        return None
    event("report_source_direct", role=agent.role, source_key=reference.key, category=category)
    if category == "report_structure_mention":
        excerpt = ("是的，报告中可以看到检验项目名称及相应结果。"
                   "阅读时可以先找到想了解的项目，再核对该项目在报告中的结果；"
                   "不同报告的排版可能不同。")
    elif category == "report_reading_limitations":
        excerpt = "。".join(reference.text.split("。")[:2])+"。"
    else:
        excerpt = reference.text
    payload = {**report_payload, "term_explanations":[{"text":excerpt, "reference_keys":[reference.key]}]}
    result = agent.limited_result(workflow, "answered", [], excerpt, {reference.key:reference},
                                  payload=payload, reference_keys=[reference.key])
    result.model_dependency_exercised = False
    if category == "reference_range_mention":
        result.answer.limitations.append("本次仅提供所提及参考范围的通用说明，如有其他问题请补充。")
    elif category in {"report_structure_mention", "report_unit_followup"}:
        result.answer.limitations.append("这是通用整理提示，无需现在提供个人报告信息。")
        if category == "report_structure_mention":
            result.answer.questions.append("接下来你更想了解某个项目名称，还是学习如何查看报告？")
    elif category == "report_learning":
        result.answer.limitations.append("本次仅介绍参考范围，不覆盖报告中的其他术语。")
    return result
