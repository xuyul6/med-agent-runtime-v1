"""Subject isolation and literal user evidence; summaries never establish facts."""
from dataclasses import dataclass
import hashlib
import math
import re
import time
from typing import get_args
from pydantic import ValidationError
from ..core.contracts import FieldName, FieldSource, GroundedField, Intent, Question
from ..memory import scope_key
from ..core.observability import event
from .medications import NAME_PATTERN, canonical


MAX_HISTORY_USERS = 4
MAX_HISTORY_CHARACTERS = 1200
FIELD_NAMES = get_args(FieldName)
SYMPTOMS = ("头痛", "咳嗽", "发热", "发烧", "腹痛", "恶心")
NEGATION = r"(?:没有|否认|不伴|并非|不是|未出现|不再|不用|不查|不问|不要|无).{0,3}$"
UNCERTAIN = r"是不是|是否|会不会|可能|假如|如果|科普|什么是|[？?]|吗"
UNKNOWN = r"不知道|不清楚|不记得|没记住|忘了|无法确定|不确定|未知"
MEDICATION_FIELDS = {"medication_name", "formulation", "population", "product_name", "strength", "region"}
FIELD_LABELS = {"medication_name":r"药名|什么药", "formulation":r"剂型", "population":r"人群|适用", "product_name":r"品牌|产品", "strength":r"规格|含量", "symptom":r"症状|不适",
                "age_group":r"年龄|几岁", "onset":r"开始|时间|多久", "change":r"变化|加重", "report_item":r"项目|指标",
                "value":r"数值|结果", "unit":r"单位", "reference_range":r"范围", "report_date":r"日期|报告时间",
                "institution":r"医院|机构", "region":r"地区|城市"}


@dataclass(frozen=True)
class SubjectResolution:
    subject: str
    method: str
    changed: bool
    ambiguous: bool = False


class SubjectTracker:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.active = {}

    def resolve(self, question: Question, session: str) -> SubjectResolution:
        now = self.clock()
        self.active = {key:value for key,value in self.active.items() if value[1] > now}
        key = scope_key(question.user, session, "__subject_pointer__")
        previous = self.active.get(key, ("self", 0))[0]
        selected = question.subject if "subject" in question.model_fields_set else previous
        method = "request" if "subject" in question.model_fields_set else "session" if key in self.active else "default"
        text = question.text
        aliases = {"mother":r"我妈妈|我母亲", "father":r"我爸爸|我父亲", "child":r"我孩子|我儿子|我女儿"}
        matches = []
        for subject, pattern in aliases.items():
            if re.search(r"(?:现在|这次|换成|改为|替|帮|给|为).{0,10}(?:"+pattern+r")", text) or re.match(r"(?:"+pattern+r")(?:[，,：:]|今年|是|有|正在|最近|想|要|的(?:报告|体检|药|症状|不适|用药))", text):
                matches.append(subject)
            elif re.search(r"(?:^|[，,。；])(?:"+pattern+r")(?:有|咳嗽|今年|正在|的报告|的体检)", text):
                matches.append(subject)
        if re.search(r"(?:换回|改回|现在问|这次问).{0,5}(?:我自己|我本人|本人)", text):
            matches.append("self")
        self_statement = bool(re.search(r"(?:^|[，,。；])(?:之前|以前|刚才)?我(?:自己|本人)?(?:曾经|以前|之前)?(?:有|正在|服用|吃|咳嗽|头痛|今年|是\d)", text))
        unknown_switch = not matches and bool(re.search(r"换个对象|换一位|换个人|(?:换成|改为).{0,5}(?:对象|患者|朋友|家属)|(?:现在|这次).{0,3}(?:给|替|帮).{1,8}(?:问|咨询)", text))
        ambiguous = unknown_switch or len(set(matches)) > 1 or self_statement and any(subject != "self" for subject in matches) or bool(re.search(r"我和(?:我妈妈|我母亲|我爸爸|我父亲|孩子)|(?:我妈妈|我母亲).{0,4}(?:和|以及)(?:我爸爸|我父亲)", text))
        if ambiguous:
            return SubjectResolution(selected, "ambiguous", False, True)
        if matches:
            selected, method = matches[0], "explicit_text"
        self.active[key] = (selected, now+86400)
        if len(self.active) > 256:
            self.active.pop(next(iter(self.active)))
        return SubjectResolution(selected, method, selected != previous)


def user_sources(text: str, state: dict, turn_id: str, subject: str) -> list[dict]:
    messages = [message for message in state.get("messages", [])[-8:] if isinstance(message, dict) and message.get("role") == "user"][-MAX_HISTORY_USERS:]
    sources = []
    for index, message in enumerate(messages):
        content = str(message.get("content", ""))[:MAX_HISTORY_CHARACTERS]
        identity = message.get("turn_id") or "legacy-"+hashlib.sha256(content.encode()).hexdigest()[:16]
        sources.append({"turn_id":identity, "turn_index":message.get("turn_index", index+1),
                        "subject":subject, "text":content, "origin":"history_user"})
    counter = max([state.get("turn_counter", 0), *[item["turn_index"] for item in sources]])
    sources.append({"turn_id":turn_id, "turn_index":counter+1, "subject":subject, "text":text, "origin":"current_user",
                    "expected_unknown_fields":state.get("last_missing_fields", [])})
    return sources


def clauses(text: str):
    yield from (match.group() for match in re.finditer(r"[^，,。！？；\n]+[！？]?", text))


# A planned visit; a directly preceding 不/没/别 ("不去医院") is not a plan.
VISIT_EVENT = r"看医生|就医|就诊|复诊|复查|门诊|(?<![不没别])(?:去|到)医院"
VISIT_PREPARATION_REQUEST = r"(?:" + VISIT_EVENT + r").*(?:准备|带|整理|记录|资料)|挂号.*(?:材料|流程)"


def visit_query_has_context(text: str) -> bool:
    """Recognize limited visit wording; this is not a semantic or identity check."""
    return bool(re.search(r"就医|就诊|医院|诊所|医疗机构|卫生院|社区卫生|挂号|门诊|复诊|问诊|"
                          r"医生|医师|医护|医务|体检|急诊|住院|看诊|"
                          r"看病(?=前|时|之前|需要|要|准备|去|就诊|[，。！？?\s]|$)", text))


@dataclass(frozen=True)
class VisitFollowupEvidence:
    current: str
    prior_clause: str
    prior_turn_id: str
    age_seconds: float


def recover_visit_followup(text: str, state: dict, *, subject_changed: bool, now: float) -> VisitFollowupEvidence | None:
    """Accept only a recent, immediately preceding user-authored general-visit task."""
    if subject_changed or state.get("last_intent") != Intent.VISIT.value or len(text) > 60:
        return None
    tokens = (r"接下来|然后|那么|请问|继续|还需要|还要|需要|应该|可以|是否|怎么|如何|"
              r"哪些|什么|这些|这个|其他|另外|准备|整理|列出|列一下|带上|携带|"
              r"资料|材料|清单|记录|事项|问题|接着|哪些|那|还|再|也|要|带|的|呢|吗|呀|啊|[，,。！？?\s]")
    if re.sub(tokens, "", text) or not re.search(r"准备|整理|带|携带|资料|材料|清单|记录|事项", text):
        return None
    messages = state.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return None
    user, assistant = messages[-2:]
    if not isinstance(user, dict) or not isinstance(assistant, dict) or user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    turn_id, turn_index, recorded = user.get("turn_id"), user.get("turn_index"), state.get("last_user_at")
    if (not isinstance(turn_id, str) or not turn_id or state.get("last_intent_turn_id") != turn_id
        or type(turn_index) is not int or turn_index != state.get("turn_counter")
        or assistant.get("turn_id") != turn_id or assistant.get("turn_index") != turn_index
        or type(recorded) not in {int, float} or not math.isfinite(recorded)
        or type(now) not in {int, float} or not math.isfinite(now)):
        return None
    age = now-recorded
    if not 0 <= age <= 300:
        return None
    prior = user.get("content")
    if not isinstance(prior, str) or len(prior) > MAX_HISTORY_CHARACTERS or institution_request(prior):
        return None
    # Avoid inferring continuity across negated or mixed non-medical tasks.
    if re.search(r"不(?:去|想|看|要|需|用|问|查|再|打算|咨询|是|了)|没(?:有|去|必要)|别(?:去|看|再|问|查|准备|整理|带)|无需|取消|停止|暂停|先别|暂不|"
                 r"旅行|旅游|出国|入职|搬家|登机|宠物|兽医|用药|药品|处方|药量|剂量|禁忌|相互作用|诊断|确诊|报告解读|化验|指标", prior):
        return None
    for clause in clauses(prior):
        if len(clause) <= 300 and visit_query_has_context(clause) and re.search(r"准备|整理|带|记录|资料|材料|清单|事项", clause):
            return VisitFollowupEvidence(text, clause, turn_id, age)
    return None


def ground_update(candidate: dict, sources: dict[str, dict]) -> GroundedField | None:
    try:
        source = sources[candidate["turn_id"]]
        name, value, quote, state = candidate["name"], candidate.get("value"), candidate["quote"], candidate["state"]
        if name not in FIELD_NAMES or not isinstance(quote, str) or quote not in source["text"]:
            raise ValueError("unverified_quote")
        if state == "unknown":
            if value is not None or not re.search(UNKNOWN, quote): raise ValueError("unverified_unknown")
            if not re.search(FIELD_LABELS[name], quote) and name not in source.get("expected_unknown_fields", []):
                raise ValueError("unknown_field_not_identified")
        elif not isinstance(value, str) or not value.strip() or value not in quote:
            raise ValueError("unverified_value")
        if name == "population" and value and re.fullmatch(
            r"(?:人群|适用人群|哪些人群|什么人群|哪些人|什么人|人|年龄|年龄段)",
            re.sub(r"\s+", "", value),
        ):
            # A question about a population does not specify that population.
            raise ValueError("population_question_is_not_a_population")
        if name == "value" and value and re.fullmatch(r"[\d.]+", value) and not re.search(r"(?<![\d.])"+re.escape(value)+r"(?![\d.])", quote):
            raise ValueError("partial_numeric_value")
        enclosing = next((part for part in clauses(source["text"]) if quote in part), quote)
        if name not in MEDICATION_FIELDS | {"institution"} and state in {"provided", "negated"} and re.search(r"科普|模板|知识|介绍|解释|比如|例如|假设|想了解|学习", enclosing):
            raise ValueError("general_information_is_not_patient_fact")
        if state == "provided" and name in MEDICATION_FIELDS | {"institution"} and re.search(UNCERTAIN, enclosing):
            state = "topic"
        elif state in {"provided", "negated"} and re.search(UNCERTAIN, enclosing):
            raise ValueError("question_is_not_patient_fact")
        if state == "topic" and name not in MEDICATION_FIELDS | {"institution"}:
            raise ValueError("topic_is_not_patient_fact")
        if value and state != "unknown":
            offset = enclosing.find(value)
            negative = bool(re.search(NEGATION, enclosing[max(0, offset-8):offset]))
            if state == "negated" and not negative: raise ValueError("unverified_negation")
            if state in {"provided", "topic"} and negative: raise ValueError("negated_assertion")
        if len(quote) > 600:
            anchor = quote.find(value) if value else re.search(UNKNOWN, quote).start()
            start = max(0, anchor-120)
            quote = quote[start:start+600]
        return GroundedField(name=name, value=value, state=state,
            source=FieldSource(turn_id=source["turn_id"], turn_index=source["turn_index"], subject=source["subject"],
                               quote=quote, origin=source["origin"]))
    except (KeyError, TypeError, ValueError, ValidationError):
        event("entity_rejected", reason="unsupported_user_evidence")
        return None


def medication_mentions(text: str) -> list[tuple[str, str, str]]:
    """(clause, written name, topic|negated) for every lexicon drug name in the text."""
    found = []
    for clause in clauses(text):
        for match in NAME_PATTERN.finditer(clause):
            negative = re.search(r"(?:不是|不查|不要|不问).{0,3}$", clause[max(0,match.start()-8):match.start()])
            found.append((clause, match.group(), "negated" if negative else "topic"))
    return found


def topic_medications(text: str) -> list[str]:
    """Distinct drugs the turn asks about, negated mentions excluded, in order of appearance."""
    return list(dict.fromkeys(canonical(name) for _, name, state in medication_mentions(text) if state == "topic"))


def local_updates(source: dict, prior_intent: str = "") -> list[dict]:
    result = []
    mentions = medication_mentions(source["text"])
    # Two different drugs in one turn (e.g. an interaction question) cannot be reduced to one lookup target.
    single_topic = len(topic_medications(source["text"])) <= 1
    for clause in clauses(source["text"]):
        def add(name, value, state="provided"):
            result.append({"name":name, "value":value, "state":state, "quote":clause, "turn_id":source["turn_id"]})
        for mentioned, name, state in mentions:
            if mentioned == clause and (state == "negated" or single_topic):
                add("medication_name", name, state)
        if prior_intent == "medicine" or NAME_PATTERN.search(source["text"]) or re.search(r"Advil|药品|制剂|说明书", source["text"], re.I):
            for name, pattern in {"product_name":r"\bAdvil\b", "strength":r"(?<![\d.])\d+(?:\.\d+)?\s*(?:mg|毫克)(?![a-z])",
                                  "formulation":r"口服包衣片|包衣片|缓释胶囊|胶囊|混悬液|凝胶|片剂",
                                  "region":r"美国|英国|中国大陆|中国", "population":r"18岁及以上|成年人|成人|儿童|婴儿|孕妇|哺乳期"}.items():
                matches = list(re.finditer(pattern, clause, re.I))
                if len(matches) == 1:
                    match = matches[0]
                    negative = re.search(NEGATION, clause[max(0, match.start()-8):match.start()])
                    if not re.search(UNKNOWN, clause):
                        add(name, match.group(), "negated" if negative else "topic")
        for symptom in SYMPTOMS:
            if symptom in clause and not re.search(UNCERTAIN, clause):
                prefix = clause[:clause.index(symptom)]
                add("symptom", symptom, "negated" if re.search(NEGATION, prefix[-8:]) else "provided")
        if not re.search(UNCERTAIN, clause):
            duration = re.search(r"(?:[一二三四五六七八九十两半\d]+)(?:小时|天|周|个月|月|年)", clause)
            if duration and (re.search("|".join(SYMPTOMS)+r"|持续|开始|已经|改为|说错", clause) or prior_intent == "symptoms" and re.match(r"\s*(?:不是|而是|是|其实|更正|改为|大概|约)?[一二三四五六七八九十两半\d]+(?:小时|天|周|个月|月|年)", clause)):
                prefix = clause[:duration.start()]
                add("onset", duration.group(), "negated" if re.search(NEGATION, prefix[-8:]) else "provided")
            age = re.search(r"(?:\d{1,3}岁|成年人|成年|儿童|婴儿)", clause)
            if age and not re.search(r"药|资料|概览|适用", clause): add("age_group", age.group())
            change = re.search(r"(?:近期|最近)?(?:没有变化|无变化|没有加重|已缓解|已经缓解|已恢复)", clause)
            if change: add("change", change.group())
        for name, label in {"medication_name":"药名", "age_group":"年龄", "onset":"开始时间", "formulation":"剂型", "population":"人群", "product_name":"品牌|产品", "strength":"规格|含量", "region":"地区", "report_item":"项目名", "value":"数值", "unit":"单位", "reference_range":"参考范围"}.items():
            if re.search(label, clause) and re.search(UNKNOWN, clause): add(name, None, "unknown")
    return result


def merge_fields(previous: dict, updates: list[GroundedField], subject: str) -> dict[str, GroundedField]:
    result = {}
    for key, raw in (previous.items() if isinstance(previous, dict) else []):
        try:
            fact = raw if isinstance(raw, GroundedField) else GroundedField.model_validate(raw)
            if fact.source.subject == subject:
                result[fact.name] = fact
        except (ValidationError, TypeError):
            continue
    for fact in sorted(updates, key=lambda item:(item.source.turn_index, item.name not in {"medication_name", "report_item"})):
        old = result.get(fact.name)
        if old and old.source.turn_index > fact.source.turn_index: continue
        if fact.state == "negated" and old and old.value != fact.value and old.state in {"provided", "topic"}:
            continue
        if fact.name in {"medication_name", "report_item"} and old and (fact.value != old.value or fact.state in {"unknown", "negated"}):
            dependent = ("formulation", "population", "product_name", "strength", "region") if fact.name == "medication_name" else ("value", "unit", "reference_range", "report_date")
            for key in dependent: result.pop(key, None)
        result[fact.name] = fact
    return result


def positive_request(text: str, pattern: str) -> bool:
    for clause in clauses(text):
        for match in re.finditer(pattern, clause):
            if not re.search(r"(?:不需要|不用|不要|无需|不查|不问|不必).{0,4}$", clause[max(0,match.start()-12):match.start()]):
                return True
    return False


INSTITUTION_REQUEST = (
    r"(?:这家|那家|该|具体|目标).{0,5}(?:医院|诊所|机构)|号源|收费|证件要求|"
    r"(?:医院|诊所|医疗机构).{0,24}(?:挂号|预约|入口|流程|费用|材料|开放时间|医保|报销|地址|位置|接诊|出诊|排班|营业时间)|"
    r"(?:预约|挂号).{0,16}(?:医院|诊所|医疗机构)"
)


def institution_request(text: str) -> bool:
    """Identify explicit institution-specific questions without assuming a verified source."""
    for clause in clauses(text):
        for match in re.finditer(INSTITUTION_REQUEST, clause):
            prefix = clause[max(0, match.start()-32):match.start()]
            if not re.search(r"(?:不需要|不用|不要|无需|不查|不问|不必|别查|没有问|不查询).{0,24}$", prefix):
                return True
    return False


def role_details(understanding, role: str) -> dict:
    entities = understanding.entities
    fields = {name:fact.value if fact.state in {"provided", "topic"} else None for name,fact in entities.items()}
    if role == "general": return {"subject_ambiguous":True} if understanding.subject_ambiguous else {}
    if role == "symptom_guide":
        details = {key:fields[key] for key in ["onset", "age_group", "symptom", "change"] if key in fields}
        if "symptom" not in details and "onset" not in details:
            details.pop("age_group", None)
        return details
    if role == "drug_reference":
        details = {key:fields[key] for key in ["medication_name", "formulation", "population", "product_name", "strength", "region"] if key in fields}
        if (understanding.tasks and positive_request(understanding.tasks[0].goal, r"具体制剂|这(?:个|种)规格|产品说明书")) or any(key in details for key in ("product_name", "strength", "formulation")):
            details["query_type"] = "product_information"
        if "medication_name" in understanding.missing_fields and "medication_name" not in details:
            details["medication_name"] = None
        return details
    if role == "report_explainer" and fields.get("report_item"):
        return {"items":[{"name":fields["report_item"], **{key:fields.get(key) for key in ["value", "unit", "reference_range"]}}]}
    if role == "visit_guide":
        details = {key:fields[key] for key in ["institution", "region"] if key in fields}
        if understanding.tasks and institution_request(understanding.tasks[0].goal):
            details["institution_specific"] = True
        return details
    return {}
