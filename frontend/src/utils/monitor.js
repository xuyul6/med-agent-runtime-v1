const record = value => value && typeof value === 'object' && !Array.isArray(value) ? value : null
const finite = value => typeof value === 'number' && Number.isFinite(value)

export function countLabel(value) {
  return Number.isSafeInteger(value) && value >= 0 ? String(value) : '未计量'
}

export function percentLabel(value) {
  return finite(value) && value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : '未计量'
}

export function latencyLabel(value) {
  return finite(value) && value >= 0 ? `${Math.round(value)} ms` : '未计量'
}

export function circuitLabel(value) {
  const labels = { closed: '关闭', open: '开启', half_open: '半开' }
  return Object.hasOwn(labels, value) ? labels[value] : '未知'
}

export function monitorRows(snapshot, key) {
  const entries = record(record(snapshot)?.[key])
  if (!entries) return null
  return Object.entries(entries).filter(([, value]) => record(value)).map(([name, value]) => ({ ...value, name }))
}

export function monitorAlerts(snapshot) {
  const alerts = record(snapshot)?.active_alerts
  return Array.isArray(alerts) ? alerts.filter(alert => record(alert) && typeof alert.name === 'string') : null
}

export function alertReason(value) {
  const labels = {
    low_success_rate: '成功率偏低', high_latency: '平均延迟偏高',
    low_nonempty_validated_result_rate: '非空有效结果比例偏低', circuit_open: '熔断器开启',
    awaiting_recovery_evidence: '等待恢复证据', awaiting_recovery_confirmation: '等待恢复确认'
  }
  return Object.hasOwn(labels, value) ? labels[value] : '其他监控原因'
}
