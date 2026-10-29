const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value)

export function buildEvaluationRequest(settings) {
  const limits = {}
  for (const [key, label, maximum] of [
    ['max_model_calls', '模型调用次数', 400],
    ['max_output_tokens', '请求输出 Token 额度', 400000],
    ['timeout_seconds', '运行时限', 600]
  ]) {
    const raw = String(settings?.[key] ?? '').trim()
    const value = Number(raw)
    if (!/^\d+$/.test(raw) || !Number.isSafeInteger(value) || value < 1 || value > maximum) {
      throw new Error(`${label}须为 1 到 ${maximum} 的整数`)
    }
    limits[key] = value
  }
  const baseline = {}
  for (const [key, label] of [['compare_to', '对比基线'], ['save_as', '保存基线']]) {
    const value = String(settings?.[key] ?? '').trim()
    if (value && !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(value)) {
      throw new Error(`${label}名称须为 1–64 位字母、数字、下划线或连字符，且以字母或数字开头`)
    }
    if (value) baseline[key] = value
  }
  if (baseline.compare_to && baseline.compare_to === baseline.save_as) throw new Error('保存基线须使用新的名称')
  return { limits, ...(Object.keys(baseline).length ? { baseline } : {}) }
}

export async function requestEvaluation(request, body = {}) {
  const report = await request('/eval/run', body)
  if (!isObject(report) || !Array.isArray(report.cases) || typeof report.complete !== 'boolean') {
    throw new Error('评测报告格式无效')
  }
  return report
}

export async function loadEvaluation(request, state, settings) {
  state.report.value = null
  state.error.value = ''
  try {
    const body = settings ? buildEvaluationRequest(settings) : {}
    state.report.value = await requestEvaluation(request, body)
  } catch (failure) {
    const detail = failure.message === 'HTTP 409'
      ? 'HTTP 409（评测可能已在运行，或基线不可用、已存在或不兼容；请核对后端记录）'
      : failure.message
    state.error.value = `评测未完成：${detail}`
  }
}

export function countLabel(value) {
  return Number.isSafeInteger(value) && value >= 0 ? String(value) : '未提供'
}

export function scoreLabel(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value.toFixed(3) : '未计量'
}

export function checkSummary(checks) {
  if (!isObject(checks)) return '检查结果不可用'
  const entries = Object.entries(checks)
  if (!entries.length) return '无检查记录'
  const failures = entries.filter(([, value]) => value === false).map(([name]) => name)
  const unknown = entries.filter(([, value]) => value !== true && value !== false).map(([name]) => name)
  return [failures.length ? `未通过：${failures.join('、')}` : '', unknown.length ? `结果未知：${unknown.join('、')}` : ''].filter(Boolean).join('；') || '已记录的检查均通过'
}

export function comparisonLabel(comparison) {
  if (!isObject(comparison)) return '基线比较状态不可用'
  if (comparison.status === 'not_requested') return '未请求基线比较'
  if (comparison.status === 'incomplete_run') return '运行未完成，普通指标未比较；新增安全失败仍可能列出'
  if (comparison.status === 'compared') return '已执行基线比较'
  return '基线比较状态不可用'
}

export function usageLabel(usage, field) {
  if (!isObject(usage) || !['provider', 'not_used'].includes(usage.measurement)) return '未计量'
  const value = usage[field]
  return Number.isSafeInteger(value) && value >= 0 ? String(value) : '未计量'
}
