"""Domain admission, scored role selection and a fixed per-request execution plan."""
import math
import re
from .classification import LEXICON
from .medications import NAMES
from ..core.contracts import (AgentAssignment, InstanceChoice, Intent, RoutingCandidate, RoutingDecision,
                        RoutingFallback, Understanding)
from .conversation import clauses, institution_request, positive_request


ROLE_FIELDS = {
    "general": (),
    "symptom_guide": ("symptom", "onset", "change", "age_group"),
    "drug_reference": ("medication_name", "formulation", "population", "product_name", "strength", "region"),
    "report_explainer": ("report_item", "value", "unit", "reference_range", "report_date"),
    "visit_guide": ("institution", "region"),
}
GOALS = {
    "general": "回答普通生活问题，介绍服务范围或澄清本轮咨询目的。",
    "symptom_guide": "整理症状记录和信息缺口；发烧吃什么有歧义时确认饮食或药品方向。",
    "drug_reference": "仅核对指定药品的公开资料、适用范围和未解决问题。",
    "report_explainer": "仅整理报告文字、原始字段、通用术语与信息缺口。",
    "visit_guide": "仅整理本次就医准备事项和需向机构核实的问题。",
}
LABELS = {"general":"需求澄清", "symptom_guide":"症状记录", "drug_reference":"药品资料",
          "report_explainer":"报告整理", "visit_guide":"就医准备", "risk_handoff":"风险提示"}
TASK_EVIDENCE = {
    Intent.SYMPTOMS: LEXICON[Intent.SYMPTOMS],
    Intent.MEDICINE: r"(?:查|核对|了解|介绍|解释).{0,12}(?:药|" + NAMES + r")|药.{0,8}(?:资料|查询)|说明书|禁忌|相互作用|用量",
    Intent.REPORT: r"(?:整理|解释|解读|了解|查看|核对).{0,12}(?:报告|化验|指标|参考范围)|报告.{0,12}(?:含义|解释|整理)|参考范围.*(?:含义|术语)",
    Intent.VISIT: LEXICON[Intent.VISIT],
}


def cross_clause_visit_preparation(text: str) -> bool:
    """Recognize a narrow current-turn visit request split by a comma."""
    parts = re.split(r"([，,。！？；\n])", text)
    visit = r"(?:想|准备|计划|打算|要|需要|将|即将|马上|去).{0,8}(?:看医生|就医|就诊)|(?:看医生|就医|就诊)(?:前|之前)"
    preparation = r"(?:准备|带|整理|记录|携带).{0,18}(?:信息|资料|材料|清单|问题|报告|记录|情况)"
    negatives = r"不(?:去|想|要|需|用|打算|看|再|是)|没(?:有|去|打算|计划|想|必要)|无需|不要|取消|停止|暂停|暂不|先别|别(?:去|看|整理|准备|带)"
    unrelated = r"旅行|旅游|出国|登机|签证|行李|护照|入职|简历|工作|搬家|宠物|兽医"
    for index in range(1, len(parts)-1, 2):
        left, separator, right = parts[index-1:index+2]
        if separator not in {"，", ","} or len(left) > 80 or len(right) > 80:
            continue
        if (re.search(negatives + "|以前|过去|之前(?:去|看|就医)|上次|去年|曾经", left)
            or re.search(negatives, right) or re.search(unrelated, left+right)):
            continue
        if positive_request(left, visit) and positive_request(right, preparation):
            return True
    return False


def bounded_score(value: float) -> float:
    return max(0.0, min(1.0, value)) if math.isfinite(value) else 0.0


class RoutingPlanner:
    def __init__(self, registry, tools, supporting_limit: int = 2):
        if not 0 <= supporting_limit <= 3:
            raise ValueError("invalid_supporting_agent_limit")
        self.registry, self.tools, self.supporting_limit = registry, tools, supporting_limit

    def plan(self, text: str, understanding: Understanding, agents: dict, monitor, *, consultation_need: str | None = None) -> RoutingDecision:
        if understanding.primary == Intent.URGENT:
            return RoutingDecision(primary_agent="risk_handoff", routing_reason="critical_risk_override", routing_confidence=1,
                assignments=[AgentAssignment(task_id="task:urgent", role="risk_handoff", intent=Intent.URGENT,
                    goal="立即给出确定性风险提示。", input_text=text,
                    instances=[InstanceChoice(instance="risk_handoff.deterministic", health=1, samples=0)])])
        domains = {understanding.primary}
        capability_question = understanding.primary == Intent.GENERAL and bool(re.search(r"支持.*(?:能力|功能)|能做什么|介绍.*(?:能力|功能)", text))
        if not understanding.subject_ambiguous and not capability_question and not consultation_need:
            # Support must have current task evidence; an old entity alone cannot activate another role.
            for task in understanding.tasks:
                if task.intent not in TASK_EVIDENCE or task.intent == understanding.primary:
                    continue
                pattern = LEXICON[task.intent]
                negative_only = re.search(pattern, text) and not positive_request(text, pattern)
                if not negative_only and (task.confidence >= .3 or task.confidence >= .12 and positive_request(text, TASK_EVIDENCE[task.intent])):
                    domains.add(task.intent)
            domains.update(kind for kind, pattern in TASK_EVIDENCE.items() if positive_request(text, pattern)
                           and not (re.search(LEXICON[kind], text) and not positive_request(text, LEXICON[kind])))
            if institution_request(text):
                domains.add(Intent.VISIT)
        domains = {kind for kind in domains if kind == Intent.GENERAL or kind == Intent.VISIT and institution_request(text)
                   or not (re.search(LEXICON[kind], text) and not positive_request(text, LEXICON[kind]))}
        if not domains:
            domains = {Intent.GENERAL}
        if len(domains) > 1:
            domains.discard(Intent.GENERAL)
        candidates, instance_fallbacks = [], []
        for role, profile in self.registry.profiles.items():
            if profile.deterministic:
                continue
            kind = next(iter(profile.knowledge_domains))
            reasons = []
            if kind not in domains:
                reasons.append("no_admitted_task")
            if any(name not in self.tools.specs or role not in self.tools.specs[name].allowed_roles
                   or not profile.knowledge_domains <= self.tools.specs[name].domains for name in profile.tool_scope):
                reasons.append("required_tool_unavailable_or_out_of_scope")
            choices, probes, monitor_blocked = [], [], []
            for agent in agents.get(role, []):
                if agent.role != role or agent.profile.knowledge_domains != profile.knowledge_domains:
                    continue
                if agent.instance in {choice.instance for choice in choices}:
                    continue
                state = monitor.routing_state(agent.instance)
                choice = InstanceChoice(instance=agent.instance, health=bounded_score(state["health"]),
                    samples=len(monitor._rows(agent.instance)), monitor_penalty=state["monitor_penalty"], monitor_probe=state["probe_available"])
                if state["excluded"]:
                    monitor_blocked.append(agent.instance)
                    if state["probe_available"]: probes.append(choice)
                else:
                    choices.append(choice)
            # At most one recovery probe per role; it reuses a normal request and has a shorter timeout.
            choices += sorted(probes, key=lambda choice:choice.instance)[:1]
            choices.sort(key=lambda choice:(not choice.monitor_probe, -choice.health, choice.instance))
            if not choices:
                reasons.append("no_qualified_instance")
                if monitor_blocked: reasons.append("monitor_penalty_threshold")
            elif kind in domains:
                instance_fallbacks += [RoutingFallback(role=role, instance=name, reason="monitor_instance_excluded", action="use_qualified_same_role")
                                       for name in monitor_blocked if name not in {choice.instance for choice in choices}]
                if probes:
                    instance_fallbacks.append(RoutingFallback(role=role, reason="monitor_recovery_probe", action="bounded_probe_then_same_role"))
            fields = [understanding.entities[name] for name in ROLE_FIELDS[role] if name in understanding.entities]
            known = sum(fact.state in {"provided", "topic"} for fact in fields)
            components = {
                "intent":bounded_score(understanding.scores.get(kind.value, understanding.confidence if kind == understanding.primary else 0)),
                "task_fit":1.0 if kind == understanding.primary else .85,
                "entity_fit":min(1.0, known / (2 if role in {"symptom_guide", "report_explainer"} else 1)),
                "business_rule":1.0 if role == "visit_guide" and (
                    positive_request(text, r"(?:看医生|就医|就诊).*(?:准备|带|整理|记录|资料)|挂号.*(?:材料|流程)")
                    or cross_clause_visit_preparation(text)) else .5,
                "health":max(choice.health for choice in choices) if choices else 0,
            }
            score = sum(components[key]*weight for key,weight in {"intent":.4, "task_fit":.2, "entity_fit":.15, "business_rule":.1, "health":.15}.items())
            candidates.append(RoutingCandidate(role=role, intent=kind, eligible=not reasons, reasons=reasons,
                components=components, score=round(score, 6) if not reasons else 0, instances=choices))
        eligible = sorted([item for item in candidates if item.eligible],
                          key=lambda item:(-item.score, item.intent != understanding.primary, item.role))
        requested = self.registry.role_for_intent[understanding.primary]
        monitor_fallback = [item.role for item in candidates if item.intent in domains and "monitor_penalty_threshold" in item.reasons]
        monitor_reason = ";monitor_fallback=["+",".join(monitor_fallback)+"]" if monitor_fallback else ""
        if not eligible:
            return RoutingDecision(primary_agent=requested, routing_reason="no_eligible_role"+monitor_reason, routing_confidence=0,
                candidates=candidates, collaboration_status="failed",
                monitor_fallback=monitor_fallback,
                fallbacks=[RoutingFallback(role=requested, reason="monitor_penalty_threshold" if monitor_fallback else "no_qualified_role", action="transparent_guidance")])
        chosen, deferred = eligible[:1+self.supporting_limit], eligible[1+self.supporting_limit:]
        unavailable = [item.role for item in candidates if item.intent in domains and not item.eligible]
        assignments = [self._assignment(text, understanding, item) for item in chosen]
        return RoutingDecision(primary_agent=chosen[0].role, supporting_agents=[item.role for item in chosen[1:]],
            routing_reason="domain_and_tool_admission_then_weighted_score"+monitor_reason, routing_confidence=chosen[0].score,
            monitor_fallback=monitor_fallback,
            candidates=candidates, assignments=assignments, deferred_roles=[item.role for item in deferred]+unavailable,
            collaboration_status="single" if len(chosen) == 1 else "planned",
            fallbacks=instance_fallbacks+[RoutingFallback(role=item.role, reason="support_capacity_limit", action="report_unfinished_task") for item in deferred]
                      +[RoutingFallback(role=item.role, reason=";".join(item.reasons), action="report_unfinished_task")
                        for item in candidates if item.role in unavailable])

    def _assignment(self, text: str, understanding: Understanding, candidate: RoutingCandidate) -> AgentAssignment:
        fields = [name for name in ROLE_FIELDS[candidate.role] if name in understanding.entities]
        selected = [part for part in clauses(text) if re.search(LEXICON[candidate.intent], part)
                    or candidate.intent == Intent.VISIT and institution_request(part)]
        selected += [understanding.entities[name].source.quote for name in fields
                     if understanding.entities[name].source.turn_id == understanding.turn_id]
        excerpt = "\n".join(dict.fromkeys(selected)) or text
        return AgentAssignment(task_id="task:"+candidate.intent.value, role=candidate.role, intent=candidate.intent,
            goal=GOALS[candidate.role], input_text=(GOALS[candidate.role]+"\n用户相关原文：\n"+excerpt)[:6000],
            context_fields=fields, instances=candidate.instances)

    @staticmethod
    def trace(decision: RoutingDecision) -> dict:
        # Task text, goals and field values never enter the request log.
        return {"primary":decision.primary_agent, "support":decision.supporting_agents,
                "reason":decision.routing_reason, "confidence":decision.routing_confidence,
                "monitor_fallback":decision.monitor_fallback,
                "confidence_kind":decision.confidence_kind,
                "candidates":[candidate.model_dump(mode="json") for candidate in decision.candidates],
                "tasks":[{"task_id":assignment.task_id, "role":assignment.role, "context_fields":assignment.context_fields,
                          "instances":[choice.instance for choice in assignment.instances]} for assignment in decision.assignments],
                "deferred_roles":decision.deferred_roles}
