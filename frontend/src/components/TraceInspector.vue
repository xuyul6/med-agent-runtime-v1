<script setup>
import { computed } from 'vue'
import {
  ENTITY_STATES, SUBJECT_RESOLUTIONS, FIELD_LABELS, INTENT_LABELS, RISK_LEVELS, ROLE_LABELS, STATUS_LABELS, STORAGE_LABELS,
  label, milliseconds, percent, riskReason, sortedScores, tokens
} from '../lib/format.js'

const props = defineProps({
  reply: { type: Object, required: true },
  subjectLabels: { type: Object, default: () => ({}) }
})
defineEmits(['close'])

const intent = computed(() => props.reply.intent || {})
const routing = computed(() => props.reply.routing_decision || null)
const risk = computed(() => RISK_LEVELS[props.reply.risk && props.reply.risk.level] || RISK_LEVELS.ROUTINE)
const scores = computed(() => sortedScores(intent.value.scores))
const channels = computed(() => Object.keys(intent.value.signals || {}))
const entities = computed(() => Object.values(intent.value.entities || {}))
const candidates = computed(() => [...((routing.value && routing.value.candidates) || [])].sort((a, b) => b.score - a.score))
const assignments = computed(() => (routing.value && routing.value.assignments) || [])
const results = computed(() => Object.fromEntries((props.reply.agent_results || []).map(item => [item.task_id || item.role, item])))
const fallbacks = computed(() => [
  ...(intent.value.fallbacks || []),
  ...((routing.value && routing.value.fallbacks) || []).map(item => `${label(ROLE_LABELS, item.role)}：${item.reason} → ${item.action}`),
  ...((routing.value && routing.value.monitor_fallback) || [])
])
const usage = computed(() => props.reply.usage || {})

const stages = computed(() => {
  const failed = assignments.value.some(item => item.status === 'failed')
  return [
    { name: '筛查', value: risk.value.label, tone: risk.value.tone === 'routine' ? 'ok' : risk.value.tone },
    { name: '理解', value: `${label(INTENT_LABELS, intent.value.primary)} ${percent(intent.value.confidence)}`, tone: 'ok' },
    { name: '路由', value: routing.value ? label(STATUS_LABELS, routing.value.collaboration_status) : '直接处理', tone: 'ok' },
    { name: '执行', value: assignments.value.length ? `${assignments.value.length} 个任务` : label(ROLE_LABELS, props.reply.primary_agent), tone: failed ? 'critical' : 'ok' },
    { name: '合成', value: props.reply.degraded ? '已降级' : label(STORAGE_LABELS, props.reply.storage), tone: props.reply.degraded ? 'caution' : 'ok' }
  ]
})

function attemptTone (status) {
  return status === 'ok' ? 'ok' : status === 'cancelled' ? '' : 'bad'
}
</script>

<template>
  <aside class="inspector" aria-label="执行过程">
    <header class="head">
      <div>
        <p class="eyebrow">执行过程</p>
        <p class="rid mono" :title="reply.request_id">{{ reply.request_id.slice(0, 8) }} · {{ milliseconds(reply.elapsed_ms) }}</p>
      </div>
      <button type="button" class="btn ghost icon" aria-label="关闭" @click="$emit('close')">
        <svg viewBox="0 0 24 24" width="16" height="16"><path d="M6 6l12 12M18 6 6 18" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </button>
    </header>

    <div class="scroll">
      <ol class="pipeline">
        <li v-for="stage in stages" :key="stage.name">
          <span class="node"><span class="dot" :class="stage.tone"></span></span>
          <span class="stage-name">{{ stage.name }}</span>
          <span class="stage-value">{{ stage.value }}</span>
        </li>
      </ol>

      <section class="block">
        <h3>急症筛查</h3>
        <p class="kv"><span>结论</span><span class="chip" :class="risk.tone === 'routine' ? '' : risk.tone">{{ risk.label }}</span></p>
        <p v-if="reply.risk && reply.risk.reasons.length" class="kv"><span>依据</span><span>{{ reply.risk.reasons.map(riskReason).join('、') }}</span></p>
        <p class="kv"><span>规则</span><span class="mono">{{ reply.risk && reply.risk.rule_version }}</span></p>
      </section>

      <section class="block">
        <h3>意图理解</h3>
        <div class="scores">
          <div v-for="[key, value] in scores" :key="key" class="score" :class="{ top: key === intent.primary }">
            <span>{{ label(INTENT_LABELS, key) }}</span>
            <div class="bar" :class="{ dim: key !== intent.primary }"><span :style="{ width: Math.round(value * 100) + '%' }"></span></div>
            <span class="mono">{{ percent(value) }}</span>
          </div>
        </div>
        <p class="kv"><span>参与通道</span><span>{{ channels.join(' · ') || '—' }}</span></p>
        <p class="kv"><span>咨询对象</span><span>{{ label(subjectLabels, intent.subject) }} <span class="muted">（{{ label(SUBJECT_RESOLUTIONS, intent.subject_resolution) }}）</span></span></p>
        <div v-if="entities.length" class="entities">
          <div v-for="item in entities" :key="item.name" class="entity">
            <div class="entity-top">
              <span>{{ label(FIELD_LABELS, item.name) }}</span>
              <strong>{{ item.value }}</strong>
              <span class="chip">{{ label(ENTITY_STATES, item.state) }}</span>
            </div>
            <p v-if="item.source && item.source.quote" class="quote">“{{ item.source.quote }}”</p>
          </div>
        </div>
        <p v-if="intent.missing_fields && intent.missing_fields.length" class="kv">
          <span>缺少字段</span><span>{{ intent.missing_fields.map(name => label(FIELD_LABELS, name)).join('、') }}</span>
        </p>
      </section>

      <section v-if="routing" class="block">
        <h3>路由</h3>
        <p class="kv"><span>主角色</span><span>{{ label(ROLE_LABELS, routing.primary_agent) }}</span></p>
        <p v-if="routing.supporting_agents.length" class="kv"><span>协作角色</span><span>{{ routing.supporting_agents.map(role => label(ROLE_LABELS, role)).join('、') }}</span></p>
        <p class="kv"><span>工程分</span><span class="mono">{{ percent(routing.routing_confidence) }}</span></p>
        <div class="candidates">
          <div v-for="item in candidates" :key="item.role" class="candidate" :class="{ out: !item.eligible }">
            <span class="cand-role">{{ label(ROLE_LABELS, item.role) }}</span>
            <div class="bar" :class="{ dim: !item.eligible }"><span :style="{ width: Math.round(item.score * 100) + '%' }"></span></div>
            <span class="mono">{{ item.eligible ? percent(item.score) : '未准入' }}</span>
          </div>
        </div>
        <p v-if="routing.deferred_roles.length" class="kv"><span>延后</span><span>{{ routing.deferred_roles.map(role => label(ROLE_LABELS, role)).join('、') }}</span></p>
        <p v-if="routing.conflicts.length" class="kv"><span>冲突</span><span>{{ routing.conflicts.length }} 处</span></p>
      </section>

      <section v-if="assignments.length" class="block">
        <h3>角色执行</h3>
        <div v-for="item in assignments" :key="item.task_id" class="task">
          <div class="task-top">
            <strong>{{ label(ROLE_LABELS, item.role) }}</strong>
            <span class="chip" :class="item.status === 'failed' ? 'critical' : item.status === 'answered' ? 'accent' : ''">{{ label(STATUS_LABELS, item.status) }}</span>
          </div>
          <p class="goal">{{ item.goal }}</p>
          <div class="attempts">
            <span v-for="(attempt, index) in item.attempts" :key="index" class="attempt">
              <span class="dot" :class="attemptTone(attempt.status)"></span>
              <span class="mono">{{ attempt.instance }}</span>
              <span class="muted">{{ label(STATUS_LABELS, attempt.status) }}{{ attempt.error_code ? ' · ' + attempt.error_code : '' }}</span>
            </span>
            <span v-if="!item.attempts.length" class="muted">未执行</span>
          </div>
          <ol v-if="results[item.task_id]" class="workflow">
            <li v-for="step in results[item.task_id].workflow" :key="step">{{ step }}</li>
          </ol>
        </div>
      </section>

      <section v-if="fallbacks.length" class="block">
        <h3>回退记录</h3>
        <ul class="plain">
          <li v-for="(item, index) in fallbacks" :key="index" class="mono">{{ item }}</li>
        </ul>
      </section>

      <section class="block">
        <h3>用量与存储</h3>
        <p class="kv"><span>模型调用</span><span class="mono">{{ usage.calls ?? '未知' }}</span></p>
        <p class="kv"><span>输入 / 输出 Token</span><span class="mono">{{ tokens(usage) }}</span></p>
        <p class="kv"><span>会话记录</span><span>{{ label(STORAGE_LABELS, reply.storage) }}</span></p>
        <p v-if="reply.recalled_summaries && reply.recalled_summaries.length" class="kv">
          <span>参考了往期摘要</span><span>{{ reply.recalled_summaries.join('、') }}</span>
        </p>
        <p class="kv"><span>人工 / 急救连接</span><span>未连接</span></p>
      </section>
    </div>
  </aside>
</template>

<style scoped>
.inspector {
  min-height: 0; display: flex; flex-direction: column;
  border-left: 1px solid var(--line); background: var(--surface);
}
.head { flex: none; display: flex; justify-content: space-between; align-items: flex-start; padding: 18px 20px 12px; border-bottom: 1px solid var(--line); }
.head .eyebrow { margin: 0; }
.rid { margin: 2px 0 0; color: var(--ink-2); }
.scroll { flex: 1; overflow-y: auto; padding: 8px 20px 32px; }

.pipeline { list-style: none; margin: 12px 0 8px; padding: 0; position: relative; }
.pipeline::before { content: ''; position: absolute; left: 7px; top: 12px; bottom: 12px; width: 1px; background: var(--line-strong); }
.pipeline li { position: relative; display: grid; grid-template-columns: 16px 44px 1fr; align-items: center; gap: 10px; padding: 5px 0; font-size: 13.5px; }
.node { display: grid; place-items: center; width: 16px; height: 16px; border-radius: 50%; background: var(--surface); z-index: 1; }
.stage-name { color: var(--ink-3); }
.stage-value { color: var(--ink); }

.block { padding: 16px 0; border-top: 1px solid var(--line); }
.block:first-of-type { margin-top: 8px; }
.block h3 { margin: 0 0 10px; font-size: 13px; font-weight: 600; }

.kv { display: grid; grid-template-columns: 96px 1fr; gap: 8px; margin: 6px 0; font-size: 13px; align-items: baseline; }
.kv > span:first-child { color: var(--ink-3); }

.scores, .candidates { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
.score, .candidate { display: grid; grid-template-columns: 64px 1fr 52px; align-items: center; gap: 10px; font-size: 13px; }
.score .mono, .candidate .mono { text-align: right; color: var(--ink-2); }
.score:not(.top) > span:first-child { color: var(--ink-2); }
.candidate { grid-template-columns: 72px 1fr 52px; }
.candidate.out .cand-role, .candidate.out .mono { color: var(--ink-3); }

.entities { display: flex; flex-direction: column; gap: 8px; margin: 10px 0; }
.entity { padding: 8px 10px; border-radius: var(--radius-sm); background: var(--sunken); font-size: 13px; }
.entity-top { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.entity-top > span:first-child { color: var(--ink-3); }
.quote { margin: 4px 0 0; color: var(--ink-2); font-size: 12.5px; }

.task { padding: 10px 12px; border: 1px solid var(--line); border-radius: var(--radius-sm); margin-bottom: 8px; }
.task-top { display: flex; justify-content: space-between; align-items: center; font-size: 14px; }
.goal { margin: 4px 0 8px; font-size: 12.5px; color: var(--ink-2); }
.attempts { display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; }
.attempt { display: inline-flex; align-items: center; gap: 8px; }
.workflow { list-style: none; display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0 0; padding: 0; }
.workflow li { font-family: var(--mono); font-size: 11px; padding: 1px 6px; border-radius: 4px; background: var(--sunken); color: var(--ink-2); }
.workflow li + li::before { content: '→ '; color: var(--ink-3); }

.plain { margin: 0; padding-left: 16px; font-size: 12.5px; color: var(--ink-2); }

@media (max-width: 1100px) {
  .inspector {
    position: fixed; z-index: 20; top: 0; right: 0; bottom: 0; width: min(420px, 100vw);
    box-shadow: -12px 0 32px rgba(0, 0, 0, .12);
  }
}
</style>
