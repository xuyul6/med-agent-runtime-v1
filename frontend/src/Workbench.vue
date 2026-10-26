<script setup>
import { computed, onMounted, ref } from 'vue'
import { makeClient, safeLink, tokenLabel } from './client.js'

const request = makeClient()
const view = ref('chat')
const status = ref(null)
const metrics = ref(null)
const error = ref('')
const busy = ref(false)
const text = ref('')
const conversation = ref([])
const session = ref('')
const user = crypto.randomUUID()
const subject = ref('self')
const context = ref('selective')
const last = ref(null)
const trace = ref(null)
const documents = ref([])
const evaluation = ref(null)
const draft = ref({ key: '', title: '', domain: 'medicine', text: '', url: '', version: '', scope: '', approved: false })
const drugAliases = ref('')
const roleNames = { symptom_guide: '症状与报告', drug_reference: '药品资料', visit_guide: '就医准备', general: '咨询接待', risk_handoff: '风险提示' }
const connection = computed(() => !status.value ? '未连接' : status.value.preview ? '离线预览' : status.value.status === 'ok' ? '在线模型' : '依赖降级')
const route = computed(() => trace.value?.events.find(event => event.event === 'routing'))

async function refresh() {
  try {
    const [health, runtime] = await Promise.all([request('/health'), request('/runtime')])
    status.value = health
    metrics.value = runtime
    error.value = ''
  } catch (failure) { error.value = `连接失败：${failure.message}` }
}

function newSession() { session.value = ''; conversation.value = []; last.value = null; trace.value = null }

async function send() {
  if (!text.value.trim() || busy.value) return
  const message = text.value.trim()
  text.value = ''
  busy.value = true
  error.value = ''
  conversation.value.push({ type: 'question', text: message })
  try {
    const reply = await request('/chat', { text: message, user, session: session.value, subject: subject.value, context: context.value })
    session.value = reply.session
    conversation.value.push({ type: 'reply', ...reply })
    last.value = reply
    trace.value = await request(`/traces/${encodeURIComponent(reply.request_id)}`)
    await refresh()
  } catch (failure) { error.value = `本次请求未完成：${failure.message}` }
  finally { busy.value = false }
}

async function show(next) {
  view.value = next
  error.value = ''
  if (next === 'knowledge') {
    try { documents.value = (await request('/knowledge')).documents }
    catch (failure) { error.value = failure.message }
  }
}

async function saveDocument() {
  busy.value = true
  try {
    await request('/knowledge', { documents: [{ ...draft.value, entities: drugAliases.value.split(/[,，]/).map(value => value.trim()).filter(Boolean) }] })
    await show('knowledge')
    draft.value = { key: '', title: '', domain: 'medicine', text: '', url: '', version: '', scope: '', approved: false }
    drugAliases.value = ''
  } catch (failure) { error.value = `资料未保存：${failure.message}` }
  finally { busy.value = false }
}

async function runEvaluation() {
  busy.value = true
  error.value = ''
  try { evaluation.value = await request('/eval/run', {}) }
  catch (failure) { error.value = `评测未完成：${failure.message}` }
  finally { busy.value = false; await refresh() }
}

async function reloadSkills() {
  try { await request('/skills/reload', {}); error.value = '角色规则已重载。' }
  catch (failure) { error.value = failure.message }
}

onMounted(refresh)
</script>

<template>
  <div class="workbench">
    <header class="header">
      <a href="#" class="identity" @click.prevent="show('chat')"><span class="symbol">m<span>+</span></span><span>MedAgent<small>INFORMATION RUNTIME</small></span></a>
      <nav aria-label="工作区"><button v-for="item in [['chat', '咨询'], ['knowledge', '参考资料'], ['evaluation', '评测']]" :key="item[0]" :class="{ selected: view === item[0] }" @click="show(item[0])">{{ item[1] }}</button></nav>
      <button class="connection" @click="refresh"><i :class="{ connected: status?.status === 'ok' }"></i>{{ connection }}</button>
    </header>

    <section class="introduction"><p class="eyebrow">OBSERVABLE MULTI-AGENT · INDEPENDENT V1</p><h1>把问题整理清楚，<br class="mobile-break">让就医准备更从容。</h1><p>记录症状、查阅公开资料、理解报告文字。每个回答都能查看来源和执行过程。</p><span class="boundary">信息参考与工程演示 · 未接入医院或实际预约</span></section>
    <p v-if="error" class="notice" role="status">{{ error }}</p>

    <div class="layout">
      <main class="main-panel">
        <section v-if="view === 'chat'" class="card consultation">
          <div class="panel-heading"><div><span class="eyebrow">CONSULTATION</span><h2>咨询记录</h2></div><button class="quiet" :disabled="busy" @click="newSession">新会话</button></div>
          <div v-if="!conversation.length" class="empty"><span class="empty-mark">＋</span><h3>从一个具体问题开始</h3><p>请使用虚构资料演示，避免输入身份信息。</p><div class="examples"><button v-for="example in ['看医生之前应该记录什么、准备哪些已有资料？', '成人布洛芬的公开资料适用于哪些人群？', '报告的参考范围有什么意义？']" :key="example" @click="text = example">{{ example }} <span>↗</span></button></div></div>
          <div v-else class="messages" aria-live="polite"><article v-for="(item, index) in conversation" :key="index" :class="['message', item.type]">
            <template v-if="item.type === 'question'"><span class="message-label">你的问题</span><p>{{ item.text }}</p></template>
            <template v-else>
              <div class="answer-heading"><span>{{ roleNames[item.primary_agent] }}</span><span>{{ Math.round(item.elapsed_ms) }} ms</span></div>
              <div v-if="item.risk.level === 'CRITICAL'" class="urgent">需要立即线下帮助 · 未实际连接急救或人工服务</div>
              <p class="answer-text">{{ item.answer.text }}</p>
              <div v-if="item.answer.questions.length"><h4>需要补充</h4><ul><li v-for="question in item.answer.questions" :key="question">{{ question }}</li></ul></div>
              <div v-if="item.answer.next_steps.length"><h4>下一步</h4><ul><li v-for="step in item.answer.next_steps" :key="step">{{ step }}</li></ul></div>
              <div v-if="item.references.length" class="references"><h4>本次引用</h4><a v-for="source in item.references" :key="source.key" :href="safeLink(source.url)" target="_blank" rel="noopener noreferrer"><span>{{ source.title }} ↗</span><small>{{ source.version }} · {{ source.scope }}</small></a></div>
              <p v-for="limit in item.answer.limitations" :key="limit" class="limitation">{{ limit }}</p>
              <p v-if="item.degraded" class="degraded">本次存在运行降级，请查看执行详情。</p>
            </template>
          </article></div>
          <form class="composer" @submit.prevent="send"><label class="sr-only" for="question">咨询问题</label><textarea id="question" v-model="text" maxlength="6000" rows="3" placeholder="输入问题，或选择上方的示例…" @keydown.ctrl.enter.prevent="send" @keydown.meta.enter.prevent="send"></textarea><div class="composer-footer"><small>⌘ / Ctrl + Enter</small><button class="primary" :disabled="busy || !text.trim()">{{ busy ? '执行中…' : '发送问题 →' }}</button></div></form>
        </section>

        <section v-if="view === 'knowledge'" class="card resource-panel"><div class="panel-heading"><div><span class="eyebrow">REFERENCE LIBRARY</span><h2>公开参考资料</h2></div><span>{{ documents.length }} 份</span></div><p class="hint">允许引用仅表示项目内可检索，不代表临床审核。个人报告不导入公共库。</p><article v-for="doc in documents" :key="doc.key" class="document"><div><span class="tag">{{ doc.domain }}</span><span class="tag">{{ doc.approved ? '允许检索' : '草稿' }}</span></div><h3><a :href="safeLink(doc.url)" target="_blank" rel="noopener noreferrer">{{ doc.title }} ↗</a></h3><p>{{ doc.text }}</p><small>{{ doc.version }} · {{ doc.scope }}</small></article>
          <form class="document-form" @submit.prevent="saveDocument"><h3>添加或替换公共资料</h3><div class="field-grid"><label>稳定标识<input v-model="draft.key" required maxlength="100"></label><label>领域<select v-model="draft.domain"><option value="medicine">药品资料</option><option value="report">报告文字</option><option value="visit">就医准备</option><option value="symptoms">症状记录</option></select></label><label>标题<input v-model="draft.title" required></label><label>来源 URL<input v-model="draft.url" type="url" required></label><label>版本<input v-model="draft.version" required></label><label>适用范围<input v-model="draft.scope" required></label></div><label v-if="draft.domain === 'medicine'">药名或别名（逗号分隔；无匹配药名的资料不用于药品回答）<input v-model="drugAliases" required></label><label>资料正文<textarea v-model="draft.text" required rows="5"></textarea></label><label class="checkbox"><input v-model="draft.approved" type="checkbox">已核对来源与适用范围，允许检索</label><button class="primary" :disabled="busy || status?.preview">保存资料</button></form>
        </section>

        <section v-if="view === 'evaluation'" class="card evaluation"><div class="panel-heading"><div><span class="eyebrow">QUALITY & REGRESSION</span><h2>固定案例评测</h2></div></div><p>通过同一条 /chat 生产链路运行虚构案例，分别记录确定性检查与模型评分。评分不代表临床准确率。</p><p class="hint">在线运行会产生模型费用；每次结果独立保存，不沿用其他实现的成绩。</p><button class="primary" :disabled="busy || !status || status.preview" @click="runEvaluation">{{ busy ? '评测执行中…' : '运行真实评测' }}</button><template v-if="evaluation"><div class="evaluation-summary"><strong>{{ evaluation.passed }} / {{ evaluation.total }}</strong><span>综合通过 · 规则通过 {{ evaluation.rule_passed }} 项</span></div><article v-for="result in evaluation.cases" :key="result.id" class="evaluation-case"><div><strong>{{ result.id }}</strong><span :class="result.passed ? 'good' : 'degraded'">{{ result.passed ? '通过' : '未通过' }}</span></div><p>{{ result.quality?.reason || result.judge_error || '没有有效 Judge 评分' }}</p><small>质量均分 {{ result.quality?.mean?.toFixed(2) ?? '未计量' }} · {{ Object.entries(result.checks).filter(([, passed]) => !passed).map(([key]) => key).join('、') || '规则检查通过' }}</small></article></template></section>
      </main>

      <aside class="sidebar"><section class="card session"><span class="eyebrow">SESSION</span><h2>咨询设置</h2><label>咨询对象<select v-model="subject" :disabled="busy" @change="newSession"><option value="self">本人</option><option value="parent">长辈</option><option value="child">孩子</option><option value="other">其他人</option></select></label><label>上下文策略<select v-model="context" :disabled="busy"><option value="selective">按需装配</option><option value="full">全量可用上下文（实验）</option></select></label><dl><dt>模型</dt><dd>{{ status?.model || '未配置 / 预览' }}</dd><dt>会话</dt><dd>{{ session ? session.slice(0, 8) : '发送后建立' }}</dd></dl></section>
        <section class="card execution"><span class="eyebrow">EXECUTION TRACE</span><h2>最近一次执行</h2><p v-if="!last" class="hint">发送问题后展示路由、来源及真实用量。</p><template v-else><dl><dt>主角色</dt><dd>{{ roleNames[last.primary_agent] }}</dd><dt>辅助角色</dt><dd>{{ last.supporting_agents.map(role => roleNames[role]).join('、') || '无' }}</dd><dt>意图</dt><dd>{{ last.intent.primary }}</dd><dt>融合置信度</dt><dd>{{ (last.intent.confidence * 100).toFixed(1) }}%</dd><dt>风险</dt><dd>{{ last.risk.level }}</dd><dt>存储</dt><dd>{{ last.storage }}</dd><dt>输入 Token</dt><dd>{{ tokenLabel(last.usage) }}</dd></dl><div v-if="route" class="routing"><div v-for="candidate in route.candidates" :key="candidate.role"><span>{{ roleNames[candidate.role] }}</span><progress :value="candidate.score" max="1"></progress></div></div><details v-if="trace"><summary>查看脱敏执行详情</summary><pre>{{ JSON.stringify(trace, null, 2) }}</pre></details></template></section>
        <section class="card runtime"><span class="eyebrow">RUNTIME HEALTH</span><h2>运行概览</h2><div class="counters"><div><strong>{{ metrics?.requests ?? '—' }}</strong><span>已处理请求</span></div><div><strong>{{ metrics?.degraded ?? '—' }}</strong><span>降级请求</span></div></div><button class="quiet" @click="reloadSkills">重载角色规则</button></section>
      </aside>
    </div><footer>MedAgent Runtime · 独立实现 v1 · 使用虚构资料演示</footer>
  </div>
</template>
