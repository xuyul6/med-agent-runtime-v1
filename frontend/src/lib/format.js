// Pure display helpers. No network, no DOM; covered by tests/format.test.js.

export const ROLE_LABELS = {
  general: '综合咨询',
  symptom_guide: '症状整理',
  drug_reference: '药品资料',
  report_explainer: '报告解读',
  visit_guide: '就医准备',
  appointment: '预约挂号',
  risk_handoff: '急症转介'
}

export const INTENT_LABELS = {
  symptoms: '症状',
  medicine: '用药',
  visit: '就医',
  report: '报告',
  general: '综合',
  urgent: '急症',
  appointment: '预约'
}

export const RISK_LEVELS = {
  ROUTINE: { tone: 'routine', label: '常规' },
  CAUTION: { tone: 'caution', label: '需专业核实' },
  CRITICAL: { tone: 'critical', label: '可能紧急' }
}

export const RISK_REASONS = {
  abdominal: '剧烈腹痛',
  anaphylaxis: '严重过敏表现',
  bleeding: '出血',
  breathing: '呼吸困难',
  chest: '胸痛',
  consciousness: '意识改变',
  generic_distress: '情况紧急',
  heat: '中暑表现',
  model_urgent_signal: '语义判断为紧急',
  neurological: '神经系统症状',
  poisoning: '中毒或误服',
  professional_review_needed: '需医师或药师核实',
  recent_urgent_followup: '延续上一轮的紧急情况',
  seizure: '抽搐',
  self_harm: '自伤风险',
  trauma: '外伤'
}

export const STORAGE_LABELS = {
  available: '已保存',
  degraded: '保存失败',
  skipped: '未保存',
  memory_preview: '预览内存'
}

export const STATUS_LABELS = {
  answered: '已回答',
  needs_clarification: '需补充',
  handoff_required: '需转交',
  planned: '已计划',
  failed: '失败',
  cancelled: '已取消',
  ok: '成功',
  timeout: '超时',
  single: '单角色',
  merged: '已合并',
  partial: '部分完成',
  conflict: '有冲突'
}

export const FIELD_LABELS = {
  medication_name: '药品名称',
  formulation: '剂型',
  population: '人群',
  product_name: '产品名',
  strength: '规格',
  symptom: '症状',
  onset: '开始时间',
  change: '变化',
  age_group: '年龄段',
  report_item: '检查项目',
  value: '数值',
  unit: '单位',
  reference_range: '参考范围',
  report_date: '报告日期',
  institution: '医院',
  region: '地区',
  department: '科室',
  appointment_date: '就诊日期',
  time_window: '时段'
}

export const ENTITY_STATES = {
  provided: '已提供',
  topic: '提及',
  negated: '否认',
  unknown: '不确定'
}

export const SUBJECT_RESOLUTIONS = {
  request: '按所选对象',
  explicit_text: '原话指明',
  session: '沿用本次会话',
  history_user: '沿用前文',
  ambiguous: '无法确定'
}

const APPOINTMENT_ERRORS = {
  draft_not_found: '这份预约草稿不存在了，请重新发起。',
  appointment_not_found: '没有找到这条预约。',
  slot_not_found: '这个号源不存在。',
  user_mismatch: '这份草稿不属于当前用户。',
  token_invalid: '确认凭证无效，请重新发起。',
  draft_expired: '草稿已过期，请重新发起。',
  slot_out_of_window: '这个日期不在可预约范围内。',
  draft_not_pending: '这份草稿已经处理过了。',
  slot_full: '这个时段已经约满。',
  already_booked: '这个时段你已经约过了。',
  department_not_found: '没有找到这个科室。',
  memory_unavailable: '会话存储暂时不可用，什么都没有删除，请稍后重试。',
  network_error: '连不上服务，请检查后端是否启动。',
  timeout: '服务响应超时，请稍后再试。'
}

export function label (table, key) {
  if (key === null || key === undefined || key === '') return '—'
  return Object.prototype.hasOwnProperty.call(table, key) ? table[key] : String(key)
}

const CALLERS = { code: '代码', model: '模型' }

// Who may call a registered tool: the code at a fixed step, the role model as a tool it picks, or both.
export function toolCaller (invokedBy) {
  if (!Array.isArray(invokedBy) || !invokedBy.length) return '—'
  if (!invokedBy.every(item => Object.prototype.hasOwnProperty.call(CALLERS, item))) return '—'
  return invokedBy.map(item => CALLERS[item]).join('+')
}

export function riskReason (code) {
  const text = String(code)
  if (text.startsWith('urgent_hint_cleared')) return '此前的紧急提示已解除'
  return label(RISK_REASONS, text)
}

export function errorMessage (error) {
  const code = error && error.code
  if (code && APPOINTMENT_ERRORS[code]) return APPOINTMENT_ERRORS[code]
  if (error && error.status >= 500) return '服务出错了，请稍后再试。'
  return code ? `请求失败（${code}）` : '请求失败'
}

// Only plain web links are rendered as anchors; anything else stays text.
export function safeUrl (value) {
  if (typeof value !== 'string') return null
  try {
    const url = new URL(value)
    return url.protocol === 'https:' || url.protocol === 'http:' ? url.href : null
  } catch {
    return null
  }
}

// The answer cites references by key; keys the reply does not carry are dropped, never invented.
export function citedReferences (reply) {
  const references = (reply && reply.references) || []
  const keys = (reply && reply.answer && reply.answer.reference_keys) || []
  const byKey = new Map(references.map(reference => [reference.key, reference]))
  const cited = []
  for (const key of keys) {
    if (byKey.has(key) && !cited.includes(byKey.get(key))) cited.push(byKey.get(key))
  }
  return cited
}

export function paragraphs (text) {
  return String(text || '').split(/\n{1,}/).map(line => line.trim()).filter(Boolean)
}

export function percent (value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  return `${Math.round(value * 100)}%`
}

export function milliseconds (value) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—'
  if (value >= 1000) return `${(value / 1000).toFixed(1)} s`
  return `${Math.round(value)} ms`
}

// Unknown usage stays unknown; it is never shown as zero.
export function tokens (usage) {
  if (!usage) return '未知'
  const input = usage.input_tokens
  const output = usage.output_tokens
  if (typeof input !== 'number' || typeof output !== 'number') return '未知'
  return `${input.toLocaleString('en-US')} / ${output.toLocaleString('en-US')}`
}

export function remaining (expiresAt, now = Date.now()) {
  const end = Date.parse(expiresAt)
  if (Number.isNaN(end)) return { expired: false, text: '' }
  const left = end - now
  if (left <= 0) return { expired: true, text: '已过期' }
  const minutes = Math.floor(left / 60000)
  if (minutes >= 60) return { expired: false, text: `${Math.floor(minutes / 60)} 小时 ${minutes % 60} 分钟内有效` }
  if (minutes >= 1) return { expired: false, text: `${minutes} 分钟内有效` }
  return { expired: false, text: `${Math.ceil(left / 1000)} 秒内有效` }
}

export function sortedScores (scores) {
  return Object.entries(scores || {})
    .filter(([, value]) => typeof value === 'number')
    .sort((a, b) => b[1] - a[1])
}

export function dayLabel (iso) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(iso || ''))
  if (!match) return String(iso || '')
  const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]))
  return `${Number(match[2])}月${Number(match[3])}日 周${'日一二三四五六'[date.getDay()]}`
}

export function windowLabel (window) {
  return window === 'am' ? '上午' : window === 'pm' ? '下午' : '全天'
}
