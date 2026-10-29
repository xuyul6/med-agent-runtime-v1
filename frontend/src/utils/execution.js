const object = value => value && typeof value === 'object' && !Array.isArray(value)

export async function fetchRequestTrace(request, requestId) {
  if (typeof requestId !== 'string' || !requestId) return { trace: null, error: '请求标识不可用，无法获取执行 Trace。' }
  try {
    const trace = await request(`/traces/${encodeURIComponent(requestId)}`)
    if (!object(trace) || trace.request_id !== requestId || !Array.isArray(trace.events)) {
      return { trace: null, error: '执行 Trace 与当前请求不匹配或格式无效。' }
    }
    return { trace, error: '' }
  } catch {
    return { trace: null, error: '本次回复已返回，但执行 Trace 暂不可用。' }
  }
}

export function scoreLabel(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? value.toFixed(3) : '未提供'
}

export function list(value) {
  return Array.isArray(value) ? value : null
}

export function label(mapping, value) {
  return typeof value === 'string' && Object.hasOwn(mapping, value) ? mapping[value] : typeof value === 'string' && value ? value : '未提供'
}

export function toolEventsForRequest(trace, requestId) {
  if (!object(trace) || typeof requestId !== 'string' || !requestId || trace.request_id !== requestId || !Array.isArray(trace.events)) return null
  return trace.events.filter(event => object(event) && (event.event === 'tool' || event.event === 'tool_batch'))
    .map(({ event, name, role, status, handler_invoked, elapsed_ms, at_ms, budget_used, budget_limit, error, call_count }) =>
      ({ event, name, role, status, handler_invoked, elapsed_ms, at_ms, budget_used, budget_limit, error, call_count }))
}

export function toolTimeLabel(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? `${value.toFixed(1)} ms` : '未提供'
}

export function toolCountLabel(value) {
  return Number.isSafeInteger(value) && value >= 0 ? String(value) : '未提供'
}
