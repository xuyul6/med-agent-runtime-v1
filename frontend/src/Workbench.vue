<script setup>
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { makeClient, safeLink, tokenLabel } from './api/client.js'
import { buildKnowledgeDocument, emptyMedicationMetadata } from './utils/knowledge.js'
import { reportItemsForDisplay } from './utils/report.js'
import { answerTextForDisplay } from './utils/answer.js'
import MonitorDetails from './components/MonitorDetails.vue'
import RoleCatalog from './components/RoleCatalog.vue'
import { loadRoleCatalog } from './utils/roles.js'
import ExecutionDetails from './components/ExecutionDetails.vue'
import { fetchRequestTrace } from './utils/execution.js'
import ToolTrace from './components/ToolTrace.vue'
import EvaluationReport from './components/EvaluationReport.vue'
import { loadEvaluation } from './utils/evaluation.js'

const request = makeClient()
const view = ref('chat')
const status = ref(null)
const metrics = ref(null)
const monitorFetchedAt = ref('')
const error = ref('')
const busy = ref(false)
const text = ref('')
const conversation = ref([])
const messages = ref(null)
const followMessages = ref(true)
const session = ref('')
const user = crypto.randomUUID()
const subject = ref('self')
const context = ref('selective')
const last = ref(null)
const trace = ref(null)
const traceError = ref('')
const traceLoading = ref(false)
const documents = ref([])
const evaluation = ref(null)
const evaluationError = ref('')
const evaluationSettings = ref({ max_model_calls: 60, max_output_tokens: 75000, timeout_seconds: 300, compare_to: '', save_as: '' })
const roleCatalog = ref({ profiles: null, skills: null })
const roleLoading = ref(false)
const skillsReloading = ref(false)
const draft = ref({ key: '', title: '', domain: 'medicine', text: '', url: '', version: '', scope: '', approved: false })
const drugAliases = ref('')
const medicationMetadata = ref(emptyMedicationMetadata())
const visitPurpose = ref('')
const roleNames = { symptom_guide: '症状整理', report_explainer: '报告文字', drug_reference: '药品资料', visit_guide: '就医准备', general: '咨询接待', risk_handoff: '风险提示' }
const connection = computed(() => !status.value ? '未连接' : status.value.preview ? '离线预览' : status.value.status === 'ok' ? '在线模型' : '依赖降级')

async function refresh() {
  try {
    const [health, runtime] = await Promise.all([request('/health'), request('/runtime')])
    status.value = health
    metrics.value = runtime
    monitorFetchedAt.value = new Date().toLocaleString()
    error.value = ''
  } catch (failure) { error.value = `连接失败：${failure.message}` }
}

function newSession() { session.value = ''; conversation.value = []; followMessages.value = true; last.value = null; trace.value = null; traceError.value = ''; traceLoading.value = false }

function updateMessageFollow() {
  const panel = messages.value
  if (panel) followMessages.value = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 80
}

watch(() => conversation.value.length, async () => {
  if (!followMessages.value) return
  await nextTick()
  if (messages.value) messages.value.scrollTop = messages.value.scrollHeight
})

async function send() {
  if (!text.value.trim() || busy.value) return
  const message = text.value.trim()
  followMessages.value = true
  text.value = ''
  busy.value = true
  error.value = ''
  conversation.value.push({ type: 'question', text: message })
  try {
    const reply = await request('/chat', { text: message, user, session: session.value, subject: subject.value, context: context.value })
    session.value = reply.session
    subject.value = reply.subject || subject.value
    conversation.value.push({ type: 'reply', ...reply, question: message, reportItems: reportItemsForDisplay(reply) })
    last.value = reply
    trace.value = null
    traceError.value = ''
    traceLoading.value = true
    const result = await fetchRequestTrace(request, reply.request_id)
    trace.value = result.trace
    traceError.value = result.error
    traceLoading.value = false
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
  if (next === 'roles') await refreshRoles()
}

async function refreshRoles() {
  if (roleLoading.value || skillsReloading.value) return
  roleLoading.value = true
  try { roleCatalog.value = await loadRoleCatalog(request) }
  finally { roleLoading.value = false }
}

async function saveDocument() {
  busy.value = true
  error.value = ''
  try {
    const document = buildKnowledgeDocument(draft.value, drugAliases.value, medicationMetadata.value, visitPurpose.value)
    await request('/knowledge', { documents: [document] })
    await show('knowledge')
    draft.value = { key: '', title: '', domain: 'medicine', text: '', url: '', version: '', scope: '', approved: false }
    drugAliases.value = ''
    medicationMetadata.value = emptyMedicationMetadata()
    visitPurpose.value = ''
  } catch (failure) { error.value = `资料未保存：${failure.message}` }
  finally { busy.value = false }
}

async function runEvaluation() {
  if (busy.value || !status.value || status.value.preview) return
  busy.value = true
  error.value = ''
  try { await loadEvaluation(request, { report: evaluation, error: evaluationError }, evaluationSettings.value) }
  finally { busy.value = false; await refresh() }
}

async function reloadSkills() {
  skillsReloading.value = true
  try {
    const snapshot = await request('/skills/reload', {})
    roleCatalog.value = { ...roleCatalog.value, skills: snapshot }
    error.value = '角色规则已重载。'
  }
  catch (failure) { error.value = failure.message }
  finally {
    skillsReloading.value = false
    if (view.value === 'roles') await refreshRoles()
  }
}

onMounted(refresh)
</script>

<template>
  <div class="workbench">
    <header class="header">
      <a href="#" class="identity" @click.prevent="show('chat')"><span class="symbol">m<span>+</span></span><span>MedAgent<small>INFORMATION RUNTIME</small></span></a>
      <nav aria-label="工作区"><button v-for="item in [['chat', '咨询'], ['knowledge', '参考资料'], ['roles', '角色契约'], ['evaluation', '评测']]" :key="item[0]" :class="{ selected: view === item[0] }" @click="show(item[0])">{{ item[1] }}</button></nav>
      <button class="connection" @click="refresh"><i :class="{ connected: status?.status === 'ok' }"></i>{{ connection }}</button>
    </header>

    <section class="introduction"><p class="eyebrow">OBSERVABLE MULTI-AGENT · INDEPENDENT V1</p><h1>把问题整理清楚，<br class="mobile-break">让就医准备更从容。</h1><p>记录症状、查阅公开资料、理解报告文字。每个回答都能查看来源和执行过程。</p><span class="boundary">信息参考与工程演示 · 未接入医院或实际预约</span></section>
    <p v-if="error" class="notice" role="status">{{ error }}</p>

    <div class="layout">
      <main class="main-panel">
        <section v-if="view === 'chat'" class="card consultation">
          <div class="panel-heading"><div><span class="eyebrow">CONSULTATION</span><h2>咨询记录</h2></div><button class="quiet" :disabled="busy" @click="newSession">新会话</button></div>
          <div v-if="!conversation.length" class="empty"><h3>从一个具体问题开始</h3><p>请使用虚构资料演示，避免输入身份信息。</p><div class="examples"><button v-for="example in ['你好，能帮我做什么？', '看医生之前应该记录什么、准备哪些已有资料？', '成人布洛芬的公开资料适用于哪些人群？']" :key="example" @click="text = example">{{ example }} <span>↗</span></button></div></div>
          <div v-else ref="messages" class="messages" role="region" aria-label="咨询对话" aria-live="polite" tabindex="0" @scroll="updateMessageFollow"><article v-for="(item, index) in conversation" :key="index" :class="['message', item.type]">
            <template v-if="item.type === 'question'"><span class="message-label">你的问题</span><p>{{ item.text }}</p></template>
            <template v-else>
              <div class="answer-heading"><span>{{ roleNames[item.primary_agent] }}</span><span>{{ Math.round(item.elapsed_ms) }} ms</span></div>
              <div v-if="item.risk.level === 'CRITICAL'" class="urgent">需要立即线下帮助 · 未实际连接急救或人工服务</div>
              <p class="answer-text">{{ answerTextForDisplay(item) }}</p>
              <section v-if="item.reportItems?.length" class="report-items" aria-label="用户提供的报告字段">
                <h4>你提供的报告字段</h4><p>仅根据聊天原文整理，未经核验；请以原报告为准。</p>
                <dl v-for="(report, reportIndex) in item.reportItems" :key="reportIndex">
                  <dt>项目</dt><dd>{{ report.name }}</dd>
                  <dt>数值</dt><dd>{{ report.value ?? '未提供或未确认' }}</dd>
                  <dt>单位</dt><dd>{{ report.unit ?? '未提供或未确认' }}</dd>
                  <dt>参考范围</dt><dd>{{ report.reference_range ?? '未提供或未确认' }}</dd>
                </dl>
              </section>
              <div v-if="item.answer.questions.length"><h4>需要补充</h4><ul><li v-for="question in item.answer.questions" :key="question">{{ question }}</li></ul></div>
              <div v-if="item.answer.next_steps.length"><h4>下一步</h4><ul><li v-for="step in item.answer.next_steps" :key="step">{{ step }}</li></ul></div>
              <div v-if="item.references.length" class="references"><h4>本次引用</h4><a v-for="source in item.references" :key="source.key" :href="safeLink(source.url)" target="_blank" rel="noopener noreferrer"><span>{{ source.title }} ↗</span><small>{{ source.version }} · {{ source.scope }}</small><small v-if="source.provenance?.updated_on || source.provenance?.checked_on">{{ source.provenance?.updated_on ? `资料更新：${source.provenance.updated_on}` : '' }}{{ source.provenance?.updated_on && source.provenance?.checked_on ? ' · ' : '' }}{{ source.provenance?.checked_on ? `项目核对：${source.provenance.checked_on}` : '' }}</small></a></div>
              <p v-for="limit in item.answer.limitations" :key="limit" class="limitation">{{ limit }}</p>
              <p v-if="item.degraded" class="degraded">本次存在运行降级，请查看执行详情。</p>
            </template>
          </article></div>
          <form class="composer" @submit.prevent="send"><label class="sr-only" for="question">咨询问题</label><textarea id="question" v-model="text" maxlength="6000" rows="3" placeholder="输入问题，或选择上方的示例…" @keydown.ctrl.enter.prevent="send" @keydown.meta.enter.prevent="send"></textarea><div class="composer-footer"><small>⌘ / Ctrl + Enter</small><button class="primary" :disabled="busy || !text.trim()">{{ busy ? '执行中…' : '发送问题 →' }}</button></div></form>
        </section>

        <section v-if="view === 'knowledge'" class="card resource-panel"><div class="panel-heading"><div><span class="eyebrow">REFERENCE LIBRARY</span><h2>公开参考资料</h2></div><span>{{ documents.length }} 份</span></div><p class="hint">允许检索仅控制项目内准入，不代表系统已核验或临床审核。个人报告不导入公共库。</p><article v-for="doc in documents" :key="doc.key" class="document"><div><span class="tag">{{ doc.domain }}</span><span class="tag">{{ !doc.approved ? '草稿' : doc.domain === 'visit' && doc.visit_scope?.purpose !== 'general_preparation' ? '暂不用于回答' : '允许检索' }}</span></div><h3><a :href="safeLink(doc.url)" target="_blank" rel="noopener noreferrer">{{ doc.title }} ↗</a></h3><p>{{ doc.text }}</p><small>{{ doc.version }} · {{ doc.scope }}</small><small v-if="doc.domain === 'medicine' && doc.provenance && doc.medication_scope"> · {{ doc.provenance.document_type === 'product_label' ? '具体制剂标签' : '公开概览' }} · {{ doc.medication_scope.region }} · 核对日期 {{ doc.provenance.checked_on }}</small><small v-if="doc.domain === 'visit'"> · {{ doc.visit_scope?.purpose === 'general_preparation' ? '通用就医准备' : doc.visit_scope?.purpose === 'institution_specific' ? '机构专属资料' : '用途未标注' }}</small></article>
          <form class="document-form" @submit.prevent="saveDocument">
            <h3>添加或替换公共资料</h3>
            <p class="hint">相同稳定标识会替换已有资料。请先自行核对原文、适用范围和使用条件；系统不会自动验证来源。</p>
            <div class="field-grid"><label>稳定标识<input v-model="draft.key" required maxlength="100"></label><label>领域<select v-model="draft.domain"><option value="medicine">药品资料</option><option value="report">报告文字</option><option value="visit">就医准备</option><option value="symptoms">症状记录</option></select></label><label>标题<input v-model="draft.title" required maxlength="200"></label><label>来源 URL<input v-model="draft.url" type="url" required></label><label>项目资料版本<input v-model="draft.version" required maxlength="100"></label><label>适用范围说明<input v-model="draft.scope" required maxlength="300"></label></div>
            <template v-if="draft.domain === 'visit'"><label>资料用途<select v-model="visitPurpose" required><option value="" disabled>请选择</option><option value="general_preparation">通用就医准备</option><option value="institution_specific">机构专属资料</option></select></label><p class="hint">请人工核对用途；机构专属资料当前只保存，不能用于回答，也不代表来源已核验。</p></template>
            <label v-if="draft.domain === 'medicine'">药名或别名（逗号分隔；无匹配药名的资料不用于药品回答）<input v-model="drugAliases" required></label>
            <template v-if="draft.domain === 'medicine'">
              <h4>来源与药品范围</h4><p class="hint">必须根据来源原文填写。当前药品检索的人群范围仅支持 18 岁及以上成人，不能代表个人适用性；具体制剂还要求查询条件与资料范围一致。</p>
              <div class="field-grid">
                <label>资料类型<select v-model="medicationMetadata.document_type" required><option value="" disabled>请选择</option><option value="public_overview">公开概览</option><option value="product_label">具体制剂标签</option></select></label>
                <label>来源机构<input v-model="medicationMetadata.organization" required></label>
                <label>来源版本<input v-model="medicationMetadata.source_version" required></label>
                <label>来源更新日期<input v-model="medicationMetadata.updated_on" type="date" required></label>
                <label>来源发布日期（可选）<input v-model="medicationMetadata.published_on" type="date"></label>
                <label>人工核对日期<input v-model="medicationMetadata.checked_on" type="date" required></label>
                <label>资料地区<input v-model="medicationMetadata.region" placeholder="例如 UK、US" required></label>
                <label>项目覆盖人群<select v-model="medicationMetadata.population" required><option value="" disabled>请选择</option><option value="adults_18_plus">18 岁及以上成人</option></select></label>
              </div>
              <label>原文人群范围及项目限制<input v-model="medicationMetadata.label_population" required></label>
              <div v-if="medicationMetadata.document_type === 'product_label'" class="field-grid"><label>产品名称<input v-model="medicationMetadata.product_name" required></label><label>剂型<input v-model="medicationMetadata.formulation" required></label><label>包装规格<input v-model="medicationMetadata.strength" required></label></div>
              <label>已核对的原文章节（每行一项）<textarea v-model="medicationMetadata.sections" rows="3" required></textarea></label>
              <label>来源使用条件<textarea v-model="medicationMetadata.use_conditions" rows="3" required></textarea></label>
              <label>使用条款 URL<input v-model="medicationMetadata.terms_url" type="url" required></label>
            </template>
            <label>资料正文<textarea v-model="draft.text" required rows="5" maxlength="20000"></textarea></label>
            <label class="checkbox"><input v-model="draft.approved" type="checkbox">我已自行核对来源与适用范围，允许项目内检索</label>
            <button class="primary" :disabled="busy || status?.preview">保存资料</button>
          </form>
        </section>

        <section v-if="view === 'evaluation'" class="card evaluation"><div class="panel-heading"><div><span class="eyebrow">QUALITY & REGRESSION</span><h2>端到端评测</h2></div></div><p>在独立运行状态下评测合成分类与多轮咨询案例，分别记录规则检查和模型评分。评分不代表临床准确率。</p><p class="hint">在线评测会产生模型费用。以下为请求额度上限，输入 Token 与实际费用不受其限制；默认额度可能不足以完成全部内置案例，达到上限会保留未完成结果。</p><form class="evaluation-settings" @submit.prevent="runEvaluation"><div class="field-grid"><label>最多模型调用次数<input v-model="evaluationSettings.max_model_calls" type="number" min="1" max="400" step="1" required :disabled="busy"></label><label>请求输出 Token 额度<input v-model="evaluationSettings.max_output_tokens" type="number" min="1" max="400000" step="1" required :disabled="busy"></label><label>运行时限（秒）<input v-model="evaluationSettings.timeout_seconds" type="number" min="1" max="600" step="1" required :disabled="busy"></label></div><details><summary>基线选项（可选）</summary><p class="hint">填写已保存的基线名称可比较；保存需使用新名称。数据集、口径和资料条件不兼容时后端会拒绝比较，旧版本不能直接混比。</p><div class="field-grid"><label>对比基线名称<input v-model="evaluationSettings.compare_to" maxlength="64" :disabled="busy" autocomplete="off"></label><label>保存为新基线<input v-model="evaluationSettings.save_as" maxlength="64" :disabled="busy" autocomplete="off"></label></div></details><button class="primary" :disabled="busy || !status || status.preview">{{ busy ? '评测执行中…' : '运行内置评测' }}</button></form><p v-if="evaluationError" class="degraded" role="status">{{ evaluationError }}</p><EvaluationReport v-if="evaluation" :report="evaluation" /></section>
        <section v-if="view === 'roles'" class="card roles"><RoleCatalog :profiles="roleCatalog.profiles" :skills="roleCatalog.skills" :loading="roleLoading || skillsReloading" :role-names="roleNames" @refresh="refreshRoles" /></section>
      </main>

      <aside class="sidebar"><section class="card session"><span class="eyebrow">SESSION</span><h2>咨询设置</h2><label>咨询对象<select v-model="subject" :disabled="busy" @change="newSession"><option value="self">本人</option><option value="mother">母亲</option><option value="father">父亲</option><option value="parent">其他长辈</option><option value="child">孩子</option><option value="other">其他人</option></select></label><label>上下文策略<select v-model="context" :disabled="busy"><option value="selective">按需装配</option><option value="full">全量可用上下文（实验）</option></select></label><dl><dt>模型</dt><dd>{{ status?.model || '未配置 / 预览' }}</dd><dt>会话</dt><dd>{{ session ? session.slice(0, 8) : '发送后建立' }}</dd></dl></section>
        <section class="card execution"><span class="eyebrow">EXECUTION TRACE</span><h2>最近一次执行</h2><p v-if="!last" class="hint">发送问题后展示路由、来源及真实用量。</p><template v-else><dl><dt>主角色</dt><dd>{{ roleNames[last.primary_agent] || last.primary_agent }}</dd><dt>辅助角色</dt><dd>{{ last.supporting_agents.map(role => roleNames[role] || role).join('、') || '无' }}</dd><dt>意图</dt><dd>{{ last.intent.primary }}</dd><dt>融合置信度</dt><dd>{{ (last.intent.confidence * 100).toFixed(1) }}%</dd><dt>风险</dt><dd>{{ last.risk.level }}</dd><dt>存储</dt><dd>{{ last.storage }}</dd><dt>输入 Token</dt><dd>{{ tokenLabel(last.usage) }}</dd></dl><ExecutionDetails :decision="last.routing_decision" :role-names="roleNames" /><ToolTrace :trace="trace" :request-id="last.request_id" :loading="traceLoading" :role-names="roleNames" /><p v-if="traceError" class="hint" role="status">{{ traceError }}</p><details v-if="trace"><summary>查看脱敏执行详情</summary><pre>{{ JSON.stringify(trace, null, 2) }}</pre></details></template></section>
        <section class="card runtime"><span class="eyebrow">RUNTIME HEALTH</span><h2>运行概览</h2><div class="counters"><div><strong>{{ metrics?.requests ?? '—' }}</strong><span>已处理请求</span></div><div><strong>{{ metrics?.degraded ?? '—' }}</strong><span>降级请求</span></div></div><button class="quiet" @click="refresh">刷新监控快照</button><MonitorDetails :snapshot="metrics" :fetched-at="monitorFetchedAt" :role-names="roleNames" /><button class="quiet monitor-reload" :disabled="skillsReloading || roleLoading" @click="reloadSkills">{{ skillsReloading ? '重载中…' : '重载角色规则' }}</button></section>
      </aside>
    </div><footer>MedAgent Runtime · 独立实现 v1 · 使用虚构资料演示</footer>
  </div>
</template>
