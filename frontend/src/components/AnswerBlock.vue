<script setup>
import { computed } from 'vue'
import {
  INTENT_LABELS, RISK_LEVELS, ROLE_LABELS, citedReferences, dayLabel, label, milliseconds, paragraphs, percent, riskReason, safeUrl
} from '../lib/format.js'
import ConfirmCard from './ConfirmCard.vue'

const props = defineProps({
  reply: { type: Object, required: true },
  user: { type: String, required: true },
  subjectLabels: { type: Object, default: () => ({}) },
  active: { type: Boolean, default: false }
})
defineEmits(['ask', 'inspect'])

const answer = computed(() => props.reply.answer || {})
const risk = computed(() => RISK_LEVELS[props.reply.risk && props.reply.risk.level] || RISK_LEVELS.ROUTINE)
const reasons = computed(() => ((props.reply.risk && props.reply.risk.reasons) || []).map(riskReason))
const critical = computed(() => risk.value.tone === 'critical')
const references = computed(() => citedReferences(props.reply).map(item => ({ ...item, href: safeUrl(item.url) })))
const roles = computed(() => [props.reply.primary_agent, ...(props.reply.supporting_agents || [])].filter(Boolean))
const intent = computed(() => props.reply.intent || {})
</script>

<template>
  <article class="answer" :class="risk.tone">
    <div v-if="critical" class="alarm" role="alert">
      <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true"><path d="M12 3 2 20h20L12 3Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/><path d="M12 10v4.5M12 17.2v.3" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>
      <div>
        <strong>可能需要立即就医</strong>
        <span v-if="reasons.length"> · {{ reasons.join('、') }}</span>
        <p>本服务没有接入急救或人工，请直接拨打急救电话或前往最近的急诊。</p>
      </div>
    </div>

    <p v-if="reply.recalled_summaries && reply.recalled_summaries.length" class="recall">
      <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      参考了你 {{ reply.recalled_summaries.map(dayLabel).join('、') }} 的咨询摘要（只作背景，不作为资料来源）
    </p>

    <div class="body">
      <p v-for="(line, index) in paragraphs(answer.text)" :key="index">{{ line }}</p>
    </div>

    <section v-if="answer.questions && answer.questions.length" class="part">
      <h3 class="eyebrow">还需要你补充</h3>
      <div class="asks">
        <button v-for="(item, index) in answer.questions" :key="index" type="button" class="ask-chip" @click="$emit('ask')">
          {{ item }}
        </button>
      </div>
    </section>

    <section v-if="answer.next_steps && answer.next_steps.length" class="part">
      <h3 class="eyebrow">接下来可以</h3>
      <ol class="steps">
        <li v-for="(item, index) in answer.next_steps" :key="index">{{ item }}</li>
      </ol>
    </section>

    <ConfirmCard v-for="action in reply.pending_actions || []" :key="action.draft_id" :action="action" :user="user" />

    <section v-if="references.length" class="part">
      <h3 class="eyebrow">资料来源</h3>
      <ol class="refs">
        <li v-for="item in references" :key="item.key">
          <a v-if="item.href" :href="item.href" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
          <span v-else>{{ item.title }}</span>
          <span class="ref-meta">{{ item.provenance && item.provenance.organization ? item.provenance.organization + ' · ' : '' }}{{ item.version }}</span>
        </li>
      </ol>
    </section>

    <ul v-if="answer.limitations && answer.limitations.length" class="limits">
      <li v-for="(item, index) in answer.limitations" :key="index">{{ item }}</li>
    </ul>

    <button type="button" class="trace" :class="{ on: active }" :aria-expanded="active" @click="$emit('inspect')">
      <span class="trace-item"><span class="dot" :class="risk.tone"></span>{{ risk.label }}</span>
      <span class="sep"></span>
      <span class="trace-item">{{ label(INTENT_LABELS, intent.primary) }} <span class="mono muted">{{ percent(intent.confidence) }}</span></span>
      <span class="sep"></span>
      <span class="trace-item">{{ roles.map(role => label(ROLE_LABELS, role)).join(' + ') }}</span>
      <template v-if="reply.subject && reply.subject !== 'self'">
        <span class="sep"></span>
        <span class="trace-item">为{{ label(subjectLabels, reply.subject) }}</span>
      </template>
      <span class="sep"></span>
      <span class="trace-item mono">{{ milliseconds(reply.elapsed_ms) }}</span>
      <span v-if="reply.degraded" class="chip caution">已降级</span>
      <span class="trace-open">{{ active ? '收起过程' : '执行过程' }}
        <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true"><path d="m9 6 6 6-6 6" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round"/></svg>
      </span>
    </button>
  </article>
</template>

<style scoped>
.answer { position: relative; padding-left: 20px; border-left: 2px solid var(--line-strong); }
.answer.caution { border-left-color: var(--caution); }
.answer.critical { border-left-color: var(--critical); }

.alarm {
  display: flex; gap: 12px; padding: 14px 16px; margin-bottom: 16px; border-radius: var(--radius);
  background: var(--critical-soft); color: var(--critical);
}
.alarm svg { flex: none; margin-top: 3px; }
.alarm p { margin: 4px 0 0; color: var(--ink); font-size: 14px; }

.recall {
  display: inline-flex; align-items: center; gap: 6px; margin: 0 0 10px; padding: 3px 10px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent); font-size: 12.5px;
}
.body p { margin: 0 0 12px; white-space: pre-wrap; word-break: break-word; }
.body p:first-child { font-size: 16px; }

.part { margin-top: 18px; }
.part h3 { margin: 0 0 8px; font-weight: 500; }

.asks { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
.ask-chip {
  text-align: left; padding: 7px 14px; border-radius: 12px; font-size: 14px; line-height: 1.5;
  border: 1px dashed var(--line-strong); background: transparent; color: var(--ink-2); cursor: text;
}
.ask-chip:hover { border-color: var(--accent); color: var(--ink); }

.steps { margin: 0; padding: 0; list-style: none; counter-reset: step; }
.steps li { position: relative; padding: 4px 0 4px 32px; counter-increment: step; }
.steps li::before {
  content: counter(step); position: absolute; left: 0; top: 7px; width: 20px; height: 20px; border-radius: 50%;
  display: grid; place-items: center; font-size: 11.5px; font-family: var(--mono);
  background: var(--accent-soft); color: var(--accent);
}

.refs { margin: 0; padding-left: 20px; font-size: 14px; }
.refs li { padding: 3px 0; }
.refs li::marker { color: var(--ink-3); font-family: var(--mono); font-size: 12px; }
.ref-meta { display: block; font-size: 12.5px; color: var(--ink-3); }

.limits { margin: 18px 0 0; padding: 0; list-style: none; font-size: 13px; color: var(--ink-3); }
.limits li { padding-left: 14px; position: relative; }
.limits li::before { content: '·'; position: absolute; left: 2px; }

.trace {
  margin-top: 16px; display: flex; flex-wrap: wrap; align-items: center; gap: 6px 10px; width: 100%;
  padding: 8px 12px; border-radius: var(--radius-sm); border: 1px solid var(--line); background: var(--surface);
  font-size: 12.5px; color: var(--ink-2); text-align: left; transition: border-color .15s;
}
.trace:hover, .trace.on { border-color: var(--accent); }
.trace-item { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.sep { width: 1px; height: 12px; background: var(--line-strong); }
.trace-open { margin-left: auto; display: inline-flex; align-items: center; gap: 2px; color: var(--accent); }
.trace.on .trace-open svg { transform: rotate(180deg); }
</style>
