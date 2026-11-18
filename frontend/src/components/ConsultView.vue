<script setup>
import { computed, nextTick, ref } from 'vue'
import { api } from '../api/client.js'
import { errorMessage, label } from '../lib/format.js'
import AnswerBlock from './AnswerBlock.vue'
import TraceInspector from './TraceInspector.vue'

const props = defineProps({
  user: { type: String, required: true },
  subject: { type: String, default: 'self' },
  subjects: { type: Array, default: () => [] },
  preview: { type: Boolean, default: false }
})

const STARTERS = [
  { tag: '症状', text: '这两天咳嗽，晚上更厉害，没有发烧，需要注意什么？' },
  { tag: '用药', text: '成年人用布洛芬，一般适用于哪些情况？' },
  { tag: '报告', text: '化验单上的参考范围是什么意思？' },
  { tag: '预约', text: '帮我约明天上午演示总医院心血管内科的号' }
]

const turns = ref([])
const draft = ref('')
const busy = ref(false)
const session = ref('')
const inspected = ref(null)
const scroller = ref(null)
const input = ref(null)
let counter = 0

const subjectLabels = computed(() => Object.fromEntries(props.subjects.map(item => [item.id, item.label])))
const inspectedReply = computed(() => {
  const turn = turns.value.find(item => item.id === inspected.value)
  return turn && turn.reply ? turn.reply : null
})

async function scrollDown () {
  await nextTick()
  const el = scroller.value
  if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
}

function resize () {
  const el = input.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'
}

async function send (text = draft.value) {
  const question = text.trim()
  if (!question || busy.value) return
  busy.value = true
  draft.value = ''
  await nextTick()
  resize()
  turns.value.push({ id: ++counter, kind: 'user', text: question, subject: props.subject })
  const answer = { id: ++counter, kind: 'assistant', reply: null, error: '' }
  turns.value.push(answer)
  scrollDown()
  try {
    const reply = await api.chat({ text: question, user: props.user, session: session.value, subject: props.subject })
    session.value = reply.session || session.value
    turns.value = turns.value.map(item => item.id === answer.id ? { ...item, reply } : item)
  } catch (error) {
    turns.value = turns.value.map(item => item.id === answer.id ? { ...item, error: errorMessage(error) } : item)
  } finally {
    busy.value = false
    scrollDown()
    input.value && input.value.focus()
  }
}

function onKey (event) {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault()
    send()
  }
}

function focusInput () {
  nextTick(() => input.value && input.value.focus())
}

function restart () {
  turns.value = []
  session.value = ''
  inspected.value = null
}

function toggleInspect (id) {
  inspected.value = inspected.value === id ? null : id
}
</script>

<template>
  <div class="consult" :class="{ inspecting: inspectedReply }" @keydown.esc="inspected = null">
    <section class="thread">
      <div ref="scroller" class="scroll">
        <div class="column">
          <div v-if="!turns.length" class="welcome">
            <p class="eyebrow">合成数据演示 · 不构成诊断</p>
            <p class="lede">描述症状、药品、检查报告或就医安排。每个问题先经过急症筛查，再分派给对应的专职角色；回答附带资料来源和完整的执行过程。</p>
            <div class="starters">
              <button v-for="item in STARTERS" :key="item.tag" type="button" class="starter" @click="send(item.text)">
                <span class="starter-tag">{{ item.tag }}</span>
                <span>{{ item.text }}</span>
              </button>
            </div>
            <p v-if="preview" class="preview-note">当前是离线预览：能看到筛查、分派和执行结构，但不会生成医疗回答。</p>
          </div>

          <template v-for="turn in turns" :key="turn.id">
            <div v-if="turn.kind === 'user'" class="ask">
              <span v-if="turn.subject !== 'self'" class="ask-for">为{{ label(subjectLabels, turn.subject) }}</span>
              <p>{{ turn.text }}</p>
            </div>
            <div v-else class="reply">
              <div v-if="!turn.reply && !turn.error" class="thinking" role="status">
                <span class="pulse"></span><span class="pulse"></span><span class="pulse"></span>
                <span class="muted">正在筛查、分派并核对资料…</span>
              </div>
              <p v-else-if="turn.error" class="failed">{{ turn.error }}</p>
              <AnswerBlock v-else :reply="turn.reply" :user="user" :subject-labels="subjectLabels"
                           :active="inspected === turn.id" @ask="focusInput" @inspect="toggleInspect(turn.id)" />
            </div>
          </template>
        </div>
      </div>

      <form class="composer" @submit.prevent="send()">
        <div class="box">
          <textarea ref="input" v-model="draft" rows="1" maxlength="6000" placeholder="描述你的情况（Enter 发送，Shift + Enter 换行）"
                    aria-label="输入问题" @input="resize" @keydown="onKey"></textarea>
          <div class="box-foot">
            <button v-if="turns.length" type="button" class="btn ghost small" :disabled="busy" @click="restart">新对话</button>
            <span v-else></span>
            <button type="submit" class="send" :disabled="busy || !draft.trim()" aria-label="发送">
              <svg viewBox="0 0 24 24" width="18" height="18"><path d="M12 19V5M5 12l7-7 7 7" stroke="currentColor" stroke-width="2.2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </button>
          </div>
        </div>
        <p class="fine">本服务只整理公开资料，不提供诊断，也不会联系医生或急救。紧急情况请直接拨打急救电话。</p>
      </form>
    </section>

    <div v-if="inspectedReply" class="scrim" aria-hidden="true" @click="inspected = null"></div>
    <TraceInspector v-if="inspectedReply" :reply="inspectedReply" :subject-labels="subjectLabels" @close="inspected = null" />
  </div>
</template>

<style scoped>
.consult { height: 100%; display: grid; grid-template-columns: minmax(0, 1fr); }
.consult.inspecting { grid-template-columns: minmax(0, 1fr) 420px; }

.thread { min-height: 0; display: flex; flex-direction: column; }
.scroll { flex: 1; overflow-y: auto; scrollbar-gutter: stable; }
.column { max-width: 760px; margin: 0 auto; padding: 40px 24px 24px; }

.welcome { padding-top: 8vh; }
.lede { color: var(--ink-2); max-width: 600px; margin: 8px 0 28px; font-size: 16px; }
.starters { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.starter {
  display: flex; flex-direction: column; align-items: flex-start; gap: 6px; text-align: left;
  padding: 14px 16px; border-radius: var(--radius); border: 1px solid var(--line); background: var(--surface);
  color: var(--ink); font-size: 14px; line-height: 1.6; transition: border-color .15s, transform .15s;
}
.starter:hover { border-color: var(--accent); transform: translateY(-1px); }
.starter-tag { font-size: 12px; color: var(--accent); letter-spacing: .06em; }
.preview-note {
  margin-top: 20px; padding: 10px 14px; border-radius: var(--radius-sm);
  background: var(--caution-soft); color: var(--caution); font-size: 13px;
}

.ask { display: flex; flex-direction: column; align-items: flex-end; margin: 28px 0 18px; }
.ask p {
  margin: 0; max-width: 82%; padding: 10px 16px; border-radius: 18px 18px 4px 18px;
  background: var(--accent-soft); color: var(--ink); white-space: pre-wrap; word-break: break-word;
}
.ask-for { font-size: 12px; color: var(--ink-3); margin-bottom: 4px; }

.reply { margin-bottom: 12px; }
.thinking { display: flex; align-items: center; gap: 6px; padding: 8px 0; font-size: 14px; }
.thinking .muted { margin-left: 6px; }
.pulse { width: 6px; height: 6px; border-radius: 50%; background: var(--accent); animation: pulse 1.2s infinite ease-in-out; }
.pulse:nth-child(2) { animation-delay: .15s; }
.pulse:nth-child(3) { animation-delay: .3s; }
@keyframes pulse { 0%, 80%, 100% { opacity: .25; transform: scale(.8); } 40% { opacity: 1; transform: scale(1); } }
.failed { color: var(--critical); background: var(--critical-soft); padding: 10px 14px; border-radius: var(--radius-sm); margin: 0; }

.composer { flex: none; padding: 8px 24px 14px; }
.box {
  max-width: 760px; margin: 0 auto; padding: 12px 12px 8px 16px;
  border-radius: 20px; border: 1px solid var(--line-strong); background: var(--surface); box-shadow: var(--shadow);
}
.box:focus-within { border-color: var(--accent); }
.box textarea {
  width: 100%; border: 0; outline: none; resize: none; background: transparent;
  line-height: 1.6; max-height: 200px; display: block;
}
.box-foot { display: flex; justify-content: space-between; align-items: center; margin-top: 4px; }
.send {
  width: 34px; height: 34px; border-radius: 50%; border: 0; display: grid; place-items: center;
  background: var(--accent); color: var(--accent-ink);
}
.send:disabled { background: var(--line-strong); color: var(--surface); opacity: 1; }
.fine { max-width: 760px; margin: 8px auto 0; text-align: center; font-size: 12px; color: var(--ink-3); }

.scrim { display: none; }

@media (max-width: 1100px) {
  .consult.inspecting { grid-template-columns: minmax(0, 1fr); }
  .scrim { display: block; position: fixed; inset: 0; z-index: 19; background: rgba(20, 22, 21, .28); }
}
@media (max-width: 640px) {
  .column { padding: 24px 16px 16px; }
  .starters { grid-template-columns: 1fr; }
  .composer { padding: 8px 16px 12px; }
  .ask p { max-width: 92%; }
}
</style>
