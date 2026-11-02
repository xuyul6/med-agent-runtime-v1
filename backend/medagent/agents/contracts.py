"""Role inputs and outputs; medical facts are never inferred by these schemas."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError
from ..core.contracts import Answer, Intent, Risk


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentInput(Contract):
    text: str = Field(min_length=1, max_length=6000)
    intent: Intent
    subject: str = Field(default="self", min_length=1, max_length=40)
    risk: Risk = Field(default_factory=Risk)

    def information_gaps(self) -> list[str]:
        return []


class GeneralInput(AgentInput):
    intent: Literal[Intent.GENERAL]
    subject_ambiguous: bool = False

    def information_gaps(self) -> list[str]:
        return ["subject"] if self.subject_ambiguous else []


class SymptomInput(AgentInput):
    intent: Literal[Intent.SYMPTOMS]
    onset: str | None = None
    age_group: str | None = None
    symptom: str | None = None
    change: str | None = None
    consultation_need: Literal["clarify", "clarify_followup", "food", "medicine"] | None = None

    def information_gaps(self) -> list[str]:
        if not self.model_fields_set & {"onset", "age_group"}:
            return []
        return ["onset"] if "age_group" in self.model_fields_set and not self.onset else []


class MedicationInput(AgentInput):
    intent: Literal[Intent.MEDICINE]
    medication_name: str | None = None
    formulation: str | None = None
    population: str | None = None
    product_name: str | None = None
    strength: str | None = None
    region: str | None = None
    query_type: Literal["overview", "product_information"] = "overview"

    def information_gaps(self) -> list[str]:
        if not self.model_fields_set & {"medication_name", "formulation", "population", "query_type"}:
            return []
        names = ["medication_name", "formulation", "population", "product_name", "strength", "region"] if self.query_type == "product_information" else ["medication_name"]
        return [name for name in names if not getattr(self, name)]


class ReportItem(Contract):
    name: str = Field(min_length=1)
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None


class ReportInput(AgentInput):
    intent: Literal[Intent.REPORT]
    items: list[ReportItem] = Field(default_factory=list)

    def information_gaps(self) -> list[str]:
        return [name for name in ["value", "unit", "reference_range"] if any(not getattr(item, name) for item in self.items)]


class VisitInput(AgentInput):
    intent: Literal[Intent.VISIT]
    institution: str | None = None
    region: str | None = None
    institution_specific: bool = False
    # The understanding model's quoted judgment that the turn is about a visit; None when no model judged it.
    visit_context: bool | None = None


class RiskInput(AgentInput):
    intent: Literal[Intent.URGENT]

    @model_validator(mode="after")
    def require_critical(self):
        if self.risk.level != "CRITICAL":
            raise ValueError("critical_risk_required")
        return self


class EvidenceItem(Contract):
    text: str = Field(min_length=1, max_length=1200)
    reference_keys: list[str] = Field(min_length=1, max_length=8)


class GeneralPayload(Contract):
    task_summary: str = ""
    suggested_role: Literal["symptom_guide", "drug_reference", "report_explainer", "visit_guide"] | None = None


class SymptomPayload(Contract):
    symptom_summary: str = ""
    timeline: list[str] = Field(default_factory=list)
    care_information: list[EvidenceItem] = Field(default_factory=list)


class MedicationIdentity(Contract):
    name: str | None = Field(default=None, description="查询身份：只复用已验证的 medication_name，不从资料推定新药名。")
    formulation: str | None = Field(default=None, description="用户明确给出的查询剂型；未提供时为 null，不从来源范围补齐。")
    population: str | None = Field(default=None, description="用户明确给出的查询人群；不是资料适用人群，未提供时为 null。")
    product_name: str | None = Field(default=None, description="用户明确给出的查询品牌或产品名；未提供时为 null。")
    strength: str | None = Field(default=None, description="用户明确给出的包装规格；未提供时为 null，不是推荐剂量。")
    region: str | None = Field(default=None, description="用户明确给出的查询地区；不是来源机构地区，未提供时为 null。")


class MedicationPayload(Contract):
    medication_identity: MedicationIdentity = Field(default_factory=MedicationIdentity)
    source_scope: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)


class ReportPayload(Contract):
    report_items: list[ReportItem] = Field(default_factory=list,
        description="只填用户实际提供的报告项目；通用术语问题使用空数组，未知值、单位和参考范围保留 null。")
    term_explanations: list[EvidenceItem] = Field(default_factory=list,
        description="来源支持的通用解释对象数组，每项含 text 和本次实际来源 reference_keys；不是字符串数组。通用术语 answered 时不可为空，正文不能代替此载荷。")
    questions_for_clinician: list[str] = Field(default_factory=list,
        description="待向医护核实的问题字符串数组，不是已经向医护发送的消息。")


class VisitPayload(Contract):
    preparation_checklist: list[str] = Field(default_factory=list)
    institution_facts: list[EvidenceItem] = Field(default_factory=list)
    needs_verification: list[str] = Field(default_factory=list)


class RiskPayload(Contract):
    risk_notice: str = Field(min_length=1)
    known_facts: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    handoff_connected: Literal[False] = False


class ContractAnswer(Answer):
    model_config = ConfigDict(extra="forbid")


class AgentOutput(Contract):
    role: str
    status: Literal["answered", "needs_clarification", "handoff_required"]
    answer: ContractAnswer
    missing_fields: list[str] = Field(default_factory=list, max_length=6)
    payload: Contract

    @model_validator(mode="after")
    def validate_state(self):
        if self.status == "needs_clarification" and not self.missing_fields:
            raise PydanticCustomError("clarification_requires_missing_fields","clarification_requires_missing_fields")
        if self.missing_fields and self.status == "answered":
            raise PydanticCustomError("missing_fields_require_clarification","missing_fields_require_clarification")
        if self.status == "answered":
            payload = self.payload
            if isinstance(payload, GeneralPayload) and not payload.task_summary.strip():
                raise PydanticCustomError("task_summary_required","task_summary_required")
            if isinstance(payload, SymptomPayload) and not payload.symptom_summary.strip():
                raise PydanticCustomError("symptom_summary_required","symptom_summary_required")
            if isinstance(payload, MedicationPayload) and not (
                payload.medication_identity.name and payload.source_scope and payload.evidence_items
            ):
                raise PydanticCustomError("medication_evidence_required","medication_evidence_required")
            if isinstance(payload, ReportPayload) and not (payload.report_items or payload.term_explanations):
                raise PydanticCustomError("report_information_required","report_information_required")
            if isinstance(payload, VisitPayload) and not payload.preparation_checklist:
                raise PydanticCustomError("visit_checklist_required","visit_checklist_required")
        return self


class GeneralOutput(AgentOutput):
    role: Literal["general"]
    missing_fields: list[Literal["consultation_purpose", "subject"]] = Field(default_factory=list, max_length=6)
    payload: GeneralPayload


class SymptomOutput(AgentOutput):
    role: Literal["symptom_guide"]
    missing_fields: list[Literal["symptom", "onset", "change", "age_group", "consultation_focus"]] = Field(default_factory=list, max_length=6)
    payload: SymptomPayload


class MedicationOutput(AgentOutput):
    role: Literal["drug_reference"]
    missing_fields: list[Literal["medication_name", "formulation", "population", "product_name", "strength", "region", "other_medications", "source"]] = Field(default_factory=list, max_length=6)
    payload: MedicationPayload


class ReportOutput(AgentOutput):
    role: Literal["report_explainer"]
    missing_fields: list[Literal["report_item", "value", "unit", "reference_range", "report_date", "source"]] = Field(default_factory=list, max_length=6)
    payload: ReportPayload


class VisitOutput(AgentOutput):
    role: Literal["visit_guide"]
    missing_fields: list[Literal["visit_purpose", "institution", "region", "source"]] = Field(default_factory=list, max_length=6)
    payload: VisitPayload


class RiskOutput(AgentOutput):
    role: Literal["risk_handoff"]
    status: Literal["handoff_required"] = "handoff_required"
    missing_fields: list[str] = Field(default_factory=list, max_length=0)
    payload: RiskPayload


FIELD_QUESTIONS = {
    "subject":"这次希望咨询哪一位对象？请一次选择一位，以免混用不同人的信息。",
    "consultation_purpose": "你希望整理症状、查询药品资料、了解报告文字，还是准备就医？",
    "consultation_focus": "你想了解发烧时的饮食，还是药品信息？",
    "symptom": "请描述希望整理的主要不适。",
    "onset": "这些情况从什么时候开始？",
    "change": "这些情况最近有什么变化？",
    "age_group": "咨询对象大致属于哪个年龄段？",
    "medication_name": "请提供希望查询的药品名称。",
    "formulation": "包装上标注的是哪种剂型？不清楚时可以保留未知。",
    "population": "希望查询哪类人群适用的资料？",
    "product_name": "包装上的产品或品牌名称是什么？不确定时请保留未知。",
    "strength": "包装标注的每单位药品规格是什么？这不是建议服用剂量。",
    "other_medications": "是否还有需要一起向药师核实的其他药品？",
    "report_item": "请提供希望了解的报告项目名称。",
    "value": "该项目在报告上的原始结果是什么？",
    "unit": "该项目的原始单位是什么？",
    "reference_range": "报告为该项目列出的参考范围是什么？",
    "report_date": "这份报告是什么时间的？",
    "visit_purpose": "你希望为哪类就医问题准备资料？",
    "institution": "希望了解哪家机构的流程？未确定时可以只整理通用准备事项。",
    "region": "希望核对哪个地区的资料？不同地区的药品标签或就医流程可能不同。",
    "source": "能否提供需要核对的公开资料或原始文字？当前尚缺适用来源。",
}
