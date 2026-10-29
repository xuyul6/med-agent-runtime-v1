"""Deterministic answer, boundary and source checks applied to every role draft."""
import json
import re
from pydantic import ValidationError
from .contracts import AgentOutput, FIELD_QUESTIONS, ReportInput
from ..core.contracts import Answer, Intent
from ..understanding.conversation import clauses, positive_request
from ..core.observability import event
from ..knowledge.medication_evidence import validate_output_identity
from .output_validation import AgentOutputError, schema_error
from .profiles import AgentProfile, DEFAULT_REGISTRY
from ..core.provider import object_from_text


ROLE_DOMAINS = {role:profile.knowledge_domains for role, profile in DEFAULT_REGISTRY.profiles.items()}
ROLE_FOR_INTENT = DEFAULT_REGISTRY.role_for_intent
SOURCE_GAP_QUESTIONS = {
    "drug_reference":"是否方便自行查找与药名和地区匹配的公开药品概览或具体产品说明书，并向药师核对？",
    "report_explainer":"是否方便带原报告向医护人员核对相关问题？",
    "visit_guide":"是否方便通过目标机构的官方渠道自行核对就医流程？",
}

# The deterministic mixed-report path quotes only this reviewed summary, unchanged.
REPORT_RANGE_SOURCE_HASH = "bda2d03ef8df0772268958a0aedb9e1689a9b94da1e6312576380db194b40481"
REPORT_RANGE_DEFINITION_HASH = "f506943119c77e64b1f33de9bf953309203f9516a666fca18b427d8324475f06"
REPORT_UNIT_SOURCE_HASH = "2f54021e88977383e66d6c25af8df122085d16381c842bd13104d0f4eb116758"
REPORT_STRUCTURE_SOURCE_HASH = "2be3b39f9eed112c4abb3293d3fbeab0559e704ad2907c86756af028adf33043"
REPORT_RANGE_QUESTION = r"参考范围(?:的)?(?:有什么意义|有何意义|是什么意思|是什么|什么含义|含义|定义)"
VISIT_URGENCY_ATTRIBUTION_NOTICE = "部分具体风险提示未能与本次引用资料逐句核对；如情况紧急，请立即联系当地急救或就近急诊。"


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


def explicit_reference_request(text: str) -> bool:
    for clause in re.split(r"[，。！？；\n]", text):
        for match in re.finditer(r"查阅|检索|引用|来源", clause):
            prefix = clause[max(0, match.start()-8):match.start()]
            if not re.search(r"(?:不要|不用|不必|无需|不需要).{0,3}$", prefix):
                return True
    return False


def checked_answer(data: dict, known: dict | list[str]) -> Answer:
    answer = Answer.model_validate(data)
    if set(answer.reference_keys)-set(known):
        raise AgentOutputError("unsupported_reference",["reference_keys"])
    check_boundaries(answer.model_dump())
    return answer


def check_boundaries(data) -> None:
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
    visible = "\n".join(strings(data))
    if re.search(r"(?:确诊为|诊断为|你患有|给你开处方|已为你(?:预约|挂号)|已转接人工)", visible):
        raise AgentOutputError("medical_boundary_violation")
    if re.search(r"(?:每次|每日|一天|一次).{0,10}\d+\s*(?:毫克|mg|片|粒)", visible, re.I):
        raise AgentOutputError("individual_dosing_not_supported")


def check_report_range_claims(text: str, keys: list[str], references: dict, field: str) -> None:
    """Conservatively require exact cited sentences for a narrow range-definition claim."""
    def sentences(value):
        def normalize(part):
            compact = re.sub(r"\s+", "", part)
            compact = re.sub(r"^(?:[-•*]|[0-9]{1,2}[.、)])", "", compact, count=1)
            wrappers = {"“":"”", "‘":"’", '"':'"', "「":"」", "『":"』"}
            if len(compact) >= 2 and compact[0] in wrappers and compact[-1] == wrappers[compact[0]]:
                compact = compact[1:-1]
            return compact
        return [normalize(part) for part in re.split(r"[。！？!?；;\n]+", value) if part.strip()]

    cited = {sentence for key in keys for sentence in sentences(references[key].text)}
    relevant = lambda value: bool(re.search(r"参考范围.*(?:统计|区间)|(?:统计|区间).*参考范围", value))
    for sentence in sentences(text):
        if relevant(sentence) and sentence not in cited:
            uncited = {part for key,ref in references.items() if key not in keys and ref.approved and ref.domain == Intent.REPORT
                       for part in sentences(ref.text)}
            reason = ("no_sources_cited" if not keys else
                      "exact_sentence_in_uncited_source" if sentence in uncited else
                      "not_exact_cited_sentence" if any(relevant(part) for part in cited) else
                      "no_range_sentence_in_cited_sources")
            raise AgentOutputError("report_range_claim_without_exact_source", [field], evidence_reason=reason)


def report_range_source_quotes(references: dict) -> list[dict[str, str]]:
    """Offer bounded whole source sentences already retrieved for this report role."""
    quotes = []
    for reference in references.values():
        if not reference.approved or reference.domain != Intent.REPORT:
            continue
        for sentence in re.split(r"[。！？!?；;\n]+", reference.text):
            sentence = sentence.strip()
            if 0 < len(sentence) <= 300 and re.search(r"参考范围.*(?:统计|区间)|(?:统计|区间).*参考范围", sentence):
                quotes.append({"reference_key":reference.key, "sentence":sentence})
                if len(quotes) == 3: return quotes
    return quotes


def unmatched_visit_urgency_warnings(output: AgentOutput, references: dict) -> int:
    """Count a narrow class of specific urgent advice not found verbatim in cited sources."""
    def sentences(value):
        return [re.sub(r"\s+", "", part) for part in re.split(r"[。！？!?；;\n]+", value) if part.strip()]

    cited = {sentence for key in output.answer.reference_keys for sentence in sentences(references[key].text)}
    symptom = r"胸痛|呼吸困难|喘不过气|意识(?:改变|不清|丧失)|昏迷|(?:剧烈|突然).{0,4}头痛|肢体无力|一侧无力|说话不清|口齿不清|视力(?:突然)?(?:变化|下降|模糊)|颈痛|颈部僵硬"
    urgency = r"急诊|急救|120|(?:立即|立刻|马上|尽快|及时).{0,12}(?:就医|求医|求助)"
    texts = [output.answer.text, *output.answer.questions, *output.answer.next_steps, *output.answer.limitations,
             *output.payload.preparation_checklist, *output.payload.needs_verification]
    unmatched = {sentence for text in texts for sentence in sentences(text)
                 if re.search(symptom, sentence) and re.search(urgency, sentence) and sentence not in cited}
    return len(unmatched)


def checked_role_output(profile: AgentProfile, data: dict, references: dict) -> AgentOutput:
    try:
        output = profile.output_contract.model_validate(data, strict=True)
    except ValidationError as exc:
        raise schema_error(exc) from None
    checked_answer(output.answer.model_dump(), references)
    check_boundaries(output.payload.model_dump())
    def reference_keys(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name == "reference_keys":
                    yield from item
                else:
                    yield from reference_keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from reference_keys(item)
    keys = set(reference_keys(output.payload.model_dump()))
    if not keys <= set(output.answer.reference_keys):
        raise AgentOutputError("payload_references_not_in_answer",["reference_keys"])
    if any(not references[key].approved or references[key].domain not in profile.knowledge_domains
           for key in output.answer.reference_keys):
        raise AgentOutputError("reference_outside_role_scope",["reference_keys"])
    if profile.role == "report_explainer":
        check_report_range_claims(output.answer.text, output.answer.reference_keys, references, "text")
        for item in output.payload.term_explanations:
            check_report_range_claims(item.text, item.reference_keys, references, "term_explanations")
    if profile.role == "visit_guide" and output.payload.institution_facts:
        raise AgentOutputError("institution_source_not_supported",["institution_facts"])
    if profile.role == "visit_guide" and unmatched_visit_urgency_warnings(output, references):
        limitations = output.answer.limitations
        if not any(VISIT_URGENCY_ATTRIBUTION_NOTICE in item for item in limitations):
            if len(limitations) < 6:
                limitations.insert(0, VISIT_URGENCY_ATTRIBUTION_NOTICE)
            else:
                limitations[0] = VISIT_URGENCY_ATTRIBUTION_NOTICE+"\n"+limitations[0]
    if output.status == "needs_clarification":
        output.answer.questions = [FIELD_QUESTIONS[name] for name in dict.fromkeys(output.missing_fields)]
        if profile.role in SOURCE_GAP_QUESTIONS and "source" in output.missing_fields and not references:
            output.answer.questions = [
                SOURCE_GAP_QUESTIONS[profile.role] if name == "source" else FIELD_QUESTIONS[name]
                for name in dict.fromkeys(output.missing_fields)
            ]
    return output


def validate_draft(profile, text: str, request, references: dict, evidence_requested: bool) -> dict:
    try:
        data = object_from_text(text)
    except json.JSONDecodeError:
        check_boundaries(text)
        raise AgentOutputError("invalid_output_json") from None
    except ValueError:
        raise AgentOutputError("invalid_output_shape") from None
    # Shape, evidence and clinical boundaries precede identity normalization/repair.
    output = checked_role_output(profile,data,references)
    if profile.role == "visit_guide":
        count = unmatched_visit_urgency_warnings(output, references)
        if count:
            event("evidence_attribution", role=profile.role, category="specific_urgent_advice", status="not_verbatim", count=count)
    data = output.model_dump()
    if request.intent == Intent.MEDICINE:
        changed = validate_output_identity(data,request,references.values())
        if changed: event("agent_output_normalized",role=profile.role,fields=changed,normalization="query_identity")
    if evidence_requested and output.status == "answered" and not output.answer.reference_keys:
        raise AgentOutputError("requested_evidence_not_returned",["reference_keys"])
    return data


def safe_report_status_repair(profile, text: str, request, references: dict) -> bool:
    """Allow a status-only retry after independently checking all other constraints."""
    try:
        data = object_from_text(text)
        if data.get("status") != "answered" or not data.get("missing_fields"):
            return False
        if not isinstance(request, ReportInput) or request.items or data.get("payload", {}).get("report_items"):
            return False
        checked_role_output(profile, {**data, "status":"needs_clarification"}, references)
        return True
    except (ValueError, TypeError, AttributeError, ValidationError):
        return False
