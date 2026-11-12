import test from 'node:test'
import assert from 'node:assert/strict'
import { alertReason, circuitLabel, countLabel, latencyLabel, monitorAlerts, monitorRows, percentLabel } from '../src/utils/monitor.js'

test('unknown values remain unknown and genuine zero stays visible', () => {
  assert.equal(percentLabel(null), '未计量')
  assert.equal(percentLabel(0), '0.0%')
  assert.equal(percentLabel(0.875), '87.5%')
  assert.equal(percentLabel(NaN), '未计量')
  assert.equal(percentLabel(1.2), '未计量')
  assert.equal(countLabel(null), '未计量')
  assert.equal(countLabel(0), '0')
  assert.equal(countLabel(-1), '未计量')
  assert.equal(latencyLabel(null), '未计量')
  assert.equal(latencyLabel(0), '0 ms')
  assert.equal(latencyLabel(80.6), '81 ms')
})

test('tool execution success is distinct from its nonempty valid result rate', () => {
  const [tool] = monitorRows({ tool_stats: { lookup_reference: { success_rate: 1, quality_success_rate: 0, circuit_state: 'closed' } } }, 'tool_stats')
  assert.equal(percentLabel(tool.success_rate), '100.0%')
  assert.equal(percentLabel(tool.quality_success_rate), '0.0%')
  assert.equal(circuitLabel(tool.circuit_state), '关闭')
  assert.equal(circuitLabel('open'), '开启')
  assert.equal(circuitLabel('half_open'), '半开')
  assert.equal(circuitLabel('toString'), '未知')
})

test('missing snapshot, empty registration and actual alerts stay distinct', () => {
  assert.equal(monitorRows(null, 'agent_stats'), null)
  assert.equal(monitorRows({ agent_stats: [] }, 'agent_stats'), null)
  assert.deepEqual(monitorRows({ agent_stats: {} }, 'agent_stats'), [])
  assert.deepEqual(monitorRows({ agent_stats: { 'visit.1': { name: 'spoof', excluded: false }, broken: null } }, 'agent_stats'), [{ name: 'visit.1', excluded: false }])
  assert.equal(monitorAlerts({}), null)
  assert.deepEqual(monitorAlerts({ active_alerts: [] }), [])
  assert.deepEqual(monitorAlerts({ active_alerts: [{ name: 'visit.1', reasons: ['low_success_rate'] }, null] }), [{ name: 'visit.1', reasons: ['low_success_rate'] }])
  assert.equal(alertReason('low_success_rate'), '成功率偏低')
  assert.equal(alertReason('toString'), '其他监控原因')
})
