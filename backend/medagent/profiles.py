"""Immutable role contracts and their runtime workflow, independent of Skills."""
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Literal
from .agent_contracts import (
    AgentInput, AgentOutput, GeneralInput, GeneralOutput, SymptomInput, SymptomOutput,
    MedicationInput, MedicationOutput, ReportInput, ReportOutput, VisitInput, VisitOutput,
    RiskInput, RiskOutput,
)
from .contracts import Intent
from .observability import event
from .tools import DEFAULT_TOOLS


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    purpose: str
    requires: tuple[str, ...]
    produces: str


@dataclass(frozen=True)
class HandoffCondition:
    when: Literal["critical_risk", "human_requested", "missing_information", "professional_review"]
    target: Literal["risk_handoff", "offline_professional", "user"]


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: tuple[WorkflowStep, ...]
    input_contract: type[AgentInput]
    output_contract: type[AgentOutput]
    handoff_conditions: tuple[HandoffCondition, ...]
    tool_scope: tuple[str, ...]
    knowledge_domains: frozenset[Intent]
    version: str = "role-contracts-3"
    deterministic: bool = False

    def describe(self) -> dict:
        return {"role":self.role, "mission":self.mission, "version":self.version,
                "workflow":[asdict(step) for step in self.workflow],
                "input_contract":self.input_contract.model_json_schema(),
                "output_contract":self.output_contract.model_json_schema(),
                "handoff_conditions":[asdict(condition) for condition in self.handoff_conditions],
                "tool_scope":list(self.tool_scope),
                "knowledge_domains":sorted(domain.value for domain in self.knowledge_domains),
                "deterministic":self.deterministic}


ROLE_CONTRACTS = {
    "general": (Intent.GENERAL, GeneralInput, GeneralOutput),
    "symptom_guide": (Intent.SYMPTOMS, SymptomInput, SymptomOutput),
    "drug_reference": (Intent.MEDICINE, MedicationInput, MedicationOutput),
    "report_explainer": (Intent.REPORT, ReportInput, ReportOutput),
    "visit_guide": (Intent.VISIT, VisitInput, VisitOutput),
    "risk_handoff": (Intent.URGENT, RiskInput, RiskOutput),
}
STEP_NAMES = ("validate_input", "check_handoff", "prepare_evidence", "compose_answer", "validate_output")
STEP_ARTIFACTS = ("input", "handoff_checked", "evidence", "draft", "validated_output")


def workflow(purposes: tuple[str, ...]) -> tuple[WorkflowStep, ...]:
    return tuple(WorkflowStep(name, purpose, (STEP_ARTIFACTS[i-1],) if i else (), STEP_ARTIFACTS[i])
                 for i, (name, purpose) in enumerate(zip(STEP_NAMES, purposes)))


def default_profiles() -> list[AgentProfile]:
    descriptions = {
        "general": ("回答普通生活问题，澄清咨询目的并指明合适的医疗角色。", "核对咨询目的", "检查风险或人工请求", "确认能力边界", "回答普通问题或整理转交建议", "校验任务摘要和角色建议"),
        "symptom_guide": ("整理症状及信息缺口；对发烧吃什么的歧义确认饮食或药品方向。", "核对原始症状信息", "检查风险与信息交接", "检索适用信息或标记不足", "整理症状、时间线和待补字段", "校验症状产物及引用"),
        "drug_reference": ("核对药品与资料适用范围，区分证据、未知信息和需药师核实事项。", "核对药品资料查询输入", "检查风险与专业交接", "检索本药品适用来源", "整理身份、来源范围和证据", "校验药品产物与引用"),
        "report_explainer": ("整理用户提供的报告文字，解释有来源的通用术语并保留缺失字段。", "核对报告原始字段", "检查风险与信息缺口", "检索报告术语资料", "整理项目、术语和就诊问题", "校验报告产物与引用"),
        "visit_guide": ("生成通用就医准备清单，区分机构已证实事实与待核实事项。", "核对就医目的与机构信息", "检查风险与实际连接边界", "检索公开流程或沟通模板", "整理准备清单与待核实事项", "校验流程来源及未执行预约"),
        "risk_handoff": ("立即输出确定性风险提示与有限交接信息，明确没有实际连接医护。", "核对确定性风险结果", "选择立即线下求助路径", "跳过检索与存储", "构造风险提示与下一步", "校验未实际转接的风险产物"),
    }
    return [AgentProfile(
        role=role, mission=descriptions[role][0], workflow=workflow(descriptions[role][1:]),
        input_contract=input_type, output_contract=output_type,
        handoff_conditions=(HandoffCondition("critical_risk", "risk_handoff"),
                            HandoffCondition("human_requested", "offline_professional"),
                            HandoffCondition("missing_information", "user"),
                            HandoffCondition("professional_review", "offline_professional")),
        tool_scope=DEFAULT_TOOLS.scope(role),
        knowledge_domains=frozenset() if role == "risk_handoff" else frozenset({intent}),
        version={"general":"role-contracts-4", "symptom_guide":"role-contracts-4", "drug_reference":"role-contracts-5","report_explainer":"role-contracts-6"}.get(role,"role-contracts-3"),
        deterministic=role == "risk_handoff",
    ) for role, (intent, input_type, output_type) in ROLE_CONTRACTS.items()]


class ProfileRegistry:
    def __init__(self, profiles: list[AgentProfile]):
        registered = {}
        for profile in profiles:
            role = profile.role
            if role in registered:
                raise ValueError(f"duplicate_agent_role:{role}")
            if role not in ROLE_CONTRACTS:
                raise ValueError(f"unknown_agent_role:{role}")
            intent, input_type, output_type = ROLE_CONTRACTS[role]
            if profile.input_contract is not input_type or profile.output_contract is not output_type:
                raise ValueError(f"invalid_agent_contract:{role}")
            expected_domains = frozenset() if role == "risk_handoff" else frozenset({intent})
            expected_tools = DEFAULT_TOOLS.scope(role)
            if profile.knowledge_domains != expected_domains or profile.tool_scope != expected_tools:
                raise ValueError(f"invalid_agent_scope:{role}")
            if not profile.mission.strip() or not profile.version.strip() or profile.deterministic != (role == "risk_handoff"):
                raise ValueError(f"invalid_agent_profile:{role}")
            if tuple(step.name for step in profile.workflow) != STEP_NAMES:
                raise ValueError(f"invalid_agent_workflow:{role}")
            for i, step in enumerate(profile.workflow):
                if not step.purpose.strip() or step.produces != STEP_ARTIFACTS[i] or step.requires != ((STEP_ARTIFACTS[i-1],) if i else ()):
                    raise ValueError(f"invalid_workflow_dependencies:{role}")
            expected_handoffs = {
                "critical_risk": "risk_handoff", "human_requested": "offline_professional", "missing_information": "user",
                "professional_review": "offline_professional",
            }
            if len(profile.handoff_conditions) != len(expected_handoffs) or {c.when:c.target for c in profile.handoff_conditions} != expected_handoffs:
                raise ValueError(f"invalid_handoff_conditions:{role}")
            registered[role] = profile
        missing = set(ROLE_CONTRACTS)-set(registered)
        if missing:
            raise ValueError("missing_agent_roles:"+",".join(sorted(missing)))
        self.profiles = MappingProxyType(registered)
        self.role_for_intent = MappingProxyType({intent:role for role, (intent, _, _) in ROLE_CONTRACTS.items()})

    def __getitem__(self, role: str) -> AgentProfile:
        return self.profiles[role]


class WorkflowRun:
    def __init__(self, profile: AgentProfile, instance: str):
        self.profile, self.instance = profile, instance
        self.completed: list[str] = []
        self.artifacts: set[str] = set()

    def complete(self, name: str, artifact: str, **metadata) -> None:
        if len(self.completed) >= len(self.profile.workflow):
            raise ValueError("workflow_already_finished")
        step = self.profile.workflow[len(self.completed)]
        if step.name != name or step.produces != artifact or not set(step.requires) <= self.artifacts:
            raise ValueError("workflow_precondition_failed")
        self.artifacts.add(artifact)
        self.completed.append(name)
        event("agent_workflow", role=self.profile.role, instance=self.instance, profile_version=self.profile.version,
              step=name, status="completed", **metadata)


DEFAULT_REGISTRY = ProfileRegistry(default_profiles())
