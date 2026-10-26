import asyncio
import json
import time
import uuid
from .classification import IntentRouter, LEXICON
from .contracts import Answer, Intent, Question, Reply, Understanding
from .execution import ROLE_FOR_INTENT, RoleAgent, Skills, ToolExecutor, checked_answer
from .memory import scope_key
from .observability import CURRENT, RunLog, Monitor, event
from .risk import RiskGate
import re


class MedicalRuntime:
    def __init__(self, settings, model, vectors, index, memory):
        self.settings, self.model, self.index, self.memory = settings, model, index, memory
        self.monitor, self.risk = Monitor(), RiskGate()
        self.router = IntentRouter(vectors, model)
        self.skills = Skills(settings.skill_directory)
        self.tools = ToolExecutor(timeout=settings.tool_timeout)
        self.agents = {role:[RoleAgent(role, f"{role}.{slot}", model, index, self.skills, self.tools) for slot in [1,2]]
                       for role in ["symptom_guide","drug_reference","visit_guide","general"]}

    async def _execute(self, role, question, understanding, context):
        instances = sorted(self.agents[role], key=lambda agent:self.monitor.health(agent.instance), reverse=True)
        domain = understanding.primary
        if ROLE_FOR_INTENT[domain] != role:
            domain = Intent.VISIT if role == "visit_guide" else Intent.SYMPTOMS
        for agent in instances:
            started = time.perf_counter()
            try:
                result = await asyncio.wait_for(agent.run(question.text, domain, context, question.context), self.settings.agent_timeout)
                self.monitor.record(agent.instance, True, (time.perf_counter()-started)*1000)
                event("agent", instance=agent.instance, status="ok")
                return result
            except Exception as exc:
                self.monitor.record(agent.instance, False, (time.perf_counter()-started)*1000)
                event("agent", instance=agent.instance, status="failed", error=type(exc).__name__)
        return None

    async def chat(self, question: Question) -> Reply:
        log = RunLog()
        token = CURRENT.set(log)
        degraded, storage = False, "skipped"
        session = question.session or str(uuid.uuid4())
        scope = scope_key(question.user, session, question.subject)
        try:
            risk = self.risk.screen(question.text, scope)
            event("risk", level=risk.level, reasons=risk.reasons, version=risk.rule_version)
            if risk.level == "CRITICAL":
                answer = Answer(text=f"你描述的信息包含需要立即线下评估的风险信号。请立即联系{self.settings.emergency_phone}或就近急诊，不要等待在线问答。",
                    next_steps=["向医护说明症状开始时间、当前状态及已知用药。"],
                    limitations=["本服务没有接入急救调度或人工队列；未实际转接。"])
                understanding = Understanding(primary=Intent.URGENT, confidence=1)
                primary, support, refs = "risk_handoff", [], []
            else:
                state = {"messages":[],"summary":""}
                try:
                    if not self.memory.available:
                        raise ConnectionError("memory_unavailable")
                    state = await asyncio.wait_for(self.memory.read(scope), timeout=3)
                    if state.get("summary"):
                        state["episodes"] = await asyncio.wait_for(self.memory.recall(scope, question.text), timeout=3)
                    storage = "memory_preview" if self.settings.preview else "available"
                except Exception as exc:
                    degraded, storage = True, "degraded"
                    event("memory_error", stage="read", error=type(exc).__name__)
                if not self.settings.preview and self.index.collection is None:
                    degraded = True
                    event("storage_fallback", service="reference_index")
                understanding = await self.router.understand(question.text)
                primary = ROLE_FOR_INTENT[understanding.primary]
                candidates = []
                for role in self.agents:
                    matching = sum(score for key,score in understanding.scores.items() if ROLE_FOR_INTENT[Intent(key)] == role)
                    health = max(self.monitor.health(a.instance) for a in self.agents[role])
                    candidates.append({"role":role,"domain_score":round(matching,4),"health":round(health,4),"score":round(matching*health,4)})
                support = []
                # Assistance is constrained by explicit task evidence, not health-based cross-domain fallback.
                if primary == "visit_guide" and re.search(LEXICON[Intent.SYMPTOMS], question.text):
                    support = ["symptom_guide"]
                elif primary in {"symptom_guide","drug_reference"} and re.search(LEXICON[Intent.VISIT], question.text):
                    support = ["visit_guide"]
                event("routing", primary=primary, support=support, candidates=candidates, reason="intent_scope_and_instance_health")
                outputs = await asyncio.gather(*(self._execute(role, question, understanding, state) for role in [primary]+support))
                valid = [x for x in outputs if x is not None]
                degraded = degraded or len(valid) != len(outputs)
                refs_by_key = {ref.key:ref for output in valid for ref in output.references}
                if not outputs[0]:
                    degraded = True
                    answer = Answer(text="当前未能完成可靠回答。请稍后重试，或携带问题和相关资料咨询医护人员。",
                                    limitations=["模型、工具或输出校验未完成，未生成诊断或预约结果。"])
                else:
                    answer = outputs[0].answer
                if outputs[0] is not None and len(valid) > 1 and self.model:
                    try:
                        merged = await self.model.json("synthesis", "合并各角色的信息，保留不确定性与适用范围，不增加医学事实。输出Answer JSON字段text,questions,next_steps,limitations,reference_keys。",
                            json.dumps({"answers":[x.answer.model_dump() for x in valid],"allowed_references":list(refs_by_key)},ensure_ascii=False), 1700)
                        answer = checked_answer(merged, refs_by_key)
                    except Exception as exc:
                        degraded = True
                        answer.limitations.append("辅助结果未通过汇总校验，当前显示主角色结果。")
                        event("synthesis_error", error=type(exc).__name__)
                refs = [refs_by_key[key] for key in answer.reference_keys]
                if risk.level == "CAUTION":
                    answer.limitations = (answer.limitations+["涉及特殊人群或用药风险，需医师或药师结合具体情况核实。"])[:6]
                try:
                    if not self.memory.available:
                        raise ConnectionError("memory_unavailable")
                    await asyncio.wait_for(self.memory.remember(scope, question.text, answer.text), timeout=self.settings.memory_timeout)
                except Exception as exc:
                    degraded, storage = True, "degraded"
                    event("memory_error", stage="write", error=type(exc).__name__)
            reply = Reply(request_id=log.request_id, session=session, intent=understanding, risk=risk,
                          primary_agent=primary, supporting_agents=support, answer=answer, references=refs,
                          degraded=degraded, storage=storage, usage=log.usage(), elapsed_ms=round(log.elapsed,1))
            self.monitor.finish(log, degraded)
            return reply
        finally:
            CURRENT.reset(token)
