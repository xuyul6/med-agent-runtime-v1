import test from 'node:test'
import assert from 'node:assert/strict'
import { fetchRequestTrace, label, list, scoreLabel, toolCountLabel, toolEventsForRequest, toolTimeLabel } from '../src/utils/execution.js'

test('trace is attached only to the matching request', async () => {
  let path
  const expected = { request_id: 'req/1', events: [{ event: 'routing' }] }
  const result = await fetchRequestTrace(async value => { path = value; return expected }, 'req/1')
  assert.equal(path, '/traces/req%2F1')
  assert.equal(result.trace, expected)
  assert.equal(result.error, '')
  const mismatch = await fetchRequestTrace(async () => ({ request_id: 'previous', events: [] }), 'current')
  assert.equal(mismatch.trace, null)
  assert.match(mismatch.error, /不匹配/)
  const malformed = await fetchRequestTrace(async () => ({ request_id: 'current' }), 'current')
  assert.equal(malformed.trace, null)
  assert.match(malformed.error, /格式无效/)
})

test('trace failure preserves the successful chat and does not invent a trace', async () => {
  const result = await fetchRequestTrace(async () => { throw new Error('HTTP 404') }, 'current')
  assert.equal(result.trace, null)
  assert.match(result.error, /回复已返回/)
  const noId = await fetchRequestTrace(() => { throw new Error('should not run') }, '')
  assert.equal(noId.trace, null)
  assert.match(noId.error, /请求标识不可用/)
})

test('engineering scores, unknown lists, and unknown labels stay distinguishable', () => {
  assert.equal(scoreLabel(0), '0.000')
  assert.equal(scoreLabel(0.836), '0.836')
  assert.equal(scoreLabel(null), '未提供')
  assert.equal(scoreLabel(1.1), '未提供')
  assert.equal(scoreLabel(NaN), '未提供')
  assert.equal(list(null), null)
  assert.deepEqual(list([]), [])
  assert.equal(label({ failed: '失败' }, 'failed'), '失败')
  assert.equal(label({}, 'toString'), 'toString')
  assert.equal(label({}, null), '未提供')
})

test('tool events belong to this request and omit arguments or result payloads', () => {
  const trace = { request_id: 'this', events: [
    { event: 'routing' },
    { event: 'tool', name: 'lookup_reference', role: 'visit_guide', status: 'empty_result', handler_invoked: true, elapsed_ms: 1.2, at_ms: 20, budget_used: 2, budget_limit: 12, arguments: { query: 'private' }, result: 'private' },
    { event: 'tool', name: 'lookup_reference', role: 'visit_guide', status: 'circuit_open', handler_invoked: false },
    { event: 'tool_batch', role: 'visit_guide', status: 'round_or_batch_limit', call_count: 4 },
    null
  ] }
  const events = toolEventsForRequest(trace, 'this')
  assert.equal(events.length, 3)
  assert.equal(events[0].status, 'empty_result')
  assert.equal(events[0].handler_invoked, true)
  assert.equal(events[1].status, 'circuit_open')
  assert.equal(events[1].handler_invoked, false)
  assert.equal(events[2].event, 'tool_batch')
  assert.equal(events[2].call_count, 4)
  assert.equal('arguments' in events[0], false)
  assert.equal('result' in events[0], false)
  assert.equal(toolEventsForRequest(trace, 'different'), null)
  assert.equal(toolEventsForRequest(null, 'this'), null)
  assert.deepEqual(toolEventsForRequest({ request_id: 'this', events: [] }, 'this'), [])
})

test('tool timings and budget counts preserve zero and unavailable values', () => {
  assert.equal(toolTimeLabel(0), '0.0 ms')
  assert.equal(toolTimeLabel(1.26), '1.3 ms')
  assert.equal(toolTimeLabel(null), '未提供')
  assert.equal(toolTimeLabel(Infinity), '未提供')
  assert.equal(toolCountLabel(0), '0')
  assert.equal(toolCountLabel(null), '未提供')
  assert.equal(toolCountLabel(-1), '未提供')
})
