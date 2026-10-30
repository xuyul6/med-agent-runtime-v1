"""Deterministic answer, boundary and source checks applied to every role draft."""
import json
import re
from pydantic import ValidationError
from .contracts import AgentOutput, FIELD_QUESTIONS, ReportInput
from ..core.contracts import Answer, Intent
from ..core.observability import event
from ..knowledge.medication_evidence import validate_output_identity
from .output_validation import AgentOutputError, schema_error
from .profiles import AgentProfile
from ..core.provider import object_from_text
from ..understanding.medications import NAMES


SOURCE_GAP_QUESTIONS = {
    "drug_reference":"是否方便自行查找与药名和地区匹配的公开药品概览或具体产品说明书，并向药师核对？",
    "report_explainer":"是否方便带原报告向医护人员核对相关问题？",
    "visit_guide":"是否方便通过目标机构的官方渠道自行核对就医流程？",
}

VISIT_URGENCY_ATTRIBUTION_NOTICE = "部分具体风险提示未能与本次引用资料逐句核对；如情况紧急，请立即联系当地急救或就近急诊。"


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


NUMBER = r"(?:\d+(?:\.\d+)?|[一二两三四五六七八九十半]+)"
DOSE_UNIT = r"(?:片|粒|颗|袋|包|支|滴|毫克|mg|克|g|毫升|ml)"
# Intake wording only: a strength ("每片含200 mg", "规格是 200 mg") has no intake verb before the number.
DOSE_AMOUNT = (r"(?:吃|服|服用|口服|含服|用|加)\s*" + NUMBER + r"(?:\s*(?:到|至|-|~)\s*" + NUMBER + r")?\s*"
               + DOSE_UNIT + r"(?![a-z])")
DOSE_FREQUENCY = (r"(?:每(?:隔)?\s*" + NUMBER + r"?\s*(?:次|天|日|小时|周|顿|餐|晚|早)|一次|一天|一日|早晚)[^，。；！？]{0,10}?"
                  + NUMBER + r"\s*(?:次|" + DOSE_UNIT + r")")
REFUSAL_BEFORE = (r"不能|无法|不会|不可以|不应|不建议|不要|别|勿|未|没有|是否|能否|可否|不表示|不代表|并非|不等于"
                  r"|需要(?:由)?(?:医生|医师|药师)|请(?:向|咨询)|咨询|核实|核对")
# A conditional excuses only a claimed action ("如果已经联系了医生"); "如果没效果可以多吃一点" is still advice.
CONDITION_BEFORE = REFUSAL_BEFORE + r"|如果|假如|若"
# Payload summaries restate the user in second or third person ("用户自述每次吃一片").
USER_QUOTE = r"(?:你|您)(?:提到|说过|说(?!得)|描述|自述)|用户(?:提到|说|描述|自述|要求|以为|询问)"
# (code, rule, pattern, words that excuse the match when they come earlier in the clause). Dose amounts are
# never excused by a hedge or a question; a clause quoting the user ("你提到一次吃了两片") is not advice.
BOUNDARIES = (
    ("individual_dosing_not_supported", "dose_amount", DOSE_AMOUNT, None),
    ("individual_dosing_not_supported", "dose_frequency", DOSE_FREQUENCY, None),
    ("individual_dosing_not_supported", "dose_relative",
     r"多吃(?:一点|一些|点|些)?(?=[。，；！？,;!?]|$)|(?:剂量|药量|用量)(?:可以|可)?(?:加倍|减半|增加|减少|加大)"
     r"|加倍|减半|加量|减量|再加(?:一|两|半)(?:片|粒|颗)", REFUSAL_BEFORE),
    ("medical_boundary_violation", "medication_decision",
     r"停(?:药|用|服|一停)|换成(?:" + NAMES + r"|别的药|其他药)|(?:不用|不必|别)再(?:吃|用|服)", REFUSAL_BEFORE),
    ("medical_boundary_violation", "safety_guarantee",
     r"(?:肯定|一定|完全|绝对)(?:没问题|没事|安全|没有(?:风险|副作用))|放心(?:吃|用|服用)|(?:很|非常)安全"
     r"|没有(?:任何)?(?:风险|副作用)", REFUSAL_BEFORE),
    ("medical_boundary_violation", "diagnosis",
     r"确诊为|诊断为|你患有|患有|(?:很|极|非常|大)可能是|看起来像|应该是|考虑(?:是|为)|大概率|多半是|你得的是|这是典型的",
     REFUSAL_BEFORE),
    ("medical_boundary_violation", "prescription", r"(?:给你|为你|帮你)开(?:药|处方|点|些)?", REFUSAL_BEFORE),
    ("medical_boundary_violation", "action_claim",
     r"已(?:经)?(?:帮你|为你|替你|给你)?(?:预约|挂号|挂好|联系|通知|转接|安排|呼叫)|帮你(?:预约|挂号|挂好)"
     r"|(?:救护车|医生)(?:正在路上|稍后会|会联系你|会回电)", CONDITION_BEFORE),
)


def check_boundaries(data) -> None:
    """Reject dosing, medication decisions, safety guarantees, diagnoses and claimed actions in any visible text."""
    def strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
    for text in strings(data):
        for clause in re.findall(r"[^。！？!?；;\n，,]+[。！？!?；;\n，,]?", text):
            if re.search(USER_QUOTE, clause):
                continue
            question = re.search(r"(?:吗|？|\?)\s*[。！？!?；;，,]?$", clause)
            for code, rule, pattern, excused_by in BOUNDARIES:
                for match in re.finditer(pattern, clause, re.I):
                    before = clause[:match.start()]
                    # Wording inside quotation marks is mentioned ("不能说“一定安全”"), not asserted.
                    quoted = before.count("“") > before.count("”") or before.count("「") > before.count("」")
                    if excused_by and (question or quoted or re.search(excused_by, before)):
                        continue
                    raise AgentOutputError(code, rules=(rule,))


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


def unmatched_urgency_sentences(texts: list[str], keys: list[str], references: dict) -> int:
    """Count a narrow class of specific urgent advice not found verbatim in cited sources."""
    def sentences(value):
        return [re.sub(r"\s+", "", part) for part in re.split(r"[。！？!?；;\n]+", value) if part.strip()]

    cited = {sentence for key in keys for sentence in sentences(references[key].text)}
    symptom = r"胸痛|呼吸困难|喘不过气|意识(?:改变|不清|丧失)|昏迷|(?:剧烈|突然).{0,4}头痛|肢体无力|一侧无力|说话不清|口齿不清|视力(?:突然)?(?:变化|下降|模糊)|颈痛|颈部僵硬"
    urgency = r"急诊|急救|120|(?:立即|立刻|马上|尽快|及时).{0,12}(?:就医|求医|求助)"
    unmatched = {sentence for text in texts for sentence in sentences(text)
                 if re.search(symptom, sentence) and re.search(urgency, sentence) and sentence not in cited}
    return len(unmatched)


def unmatched_visit_urgency_warnings(output: AgentOutput, references: dict) -> int:
    return unmatched_urgency_sentences([output.answer.text, *output.answer.questions, *output.answer.next_steps,
                                        *output.answer.limitations, *output.payload.preparation_checklist,
                                        *output.payload.needs_verification], output.answer.reference_keys, references)


def add_urgency_notice(limitations: list[str]) -> None:
    if not any(VISIT_URGENCY_ATTRIBUTION_NOTICE in item for item in limitations):
        if len(limitations) < 6:
            limitations.insert(0, VISIT_URGENCY_ATTRIBUTION_NOTICE)
        else:
            limitations[0] = VISIT_URGENCY_ATTRIBUTION_NOTICE+"\n"+limitations[0]


def checked_single_answer(data: dict, references: dict, *, visit_task: bool, evidence_requested: bool) -> Answer:
    """The role path's answer-level evidence checks; its payload checks have no counterpart in a plain answer."""
    answer = checked_answer(data, references)
    if any(not references[key].approved for key in answer.reference_keys):
        raise AgentOutputError("reference_outside_role_scope",["reference_keys"])
    check_report_range_claims(answer.text, answer.reference_keys, references, "text")
    if visit_task and unmatched_urgency_sentences([answer.text, *answer.questions, *answer.next_steps, *answer.limitations],
                                                  answer.reference_keys, references):
        add_urgency_notice(answer.limitations)
    # Nothing retrieved: saying so honestly is the correct answer, as in the role path's source gap.
    if evidence_requested and references and not answer.reference_keys:
        raise AgentOutputError("requested_evidence_not_returned",["reference_keys"])
    return answer


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
        add_urgency_notice(output.answer.limitations)
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
