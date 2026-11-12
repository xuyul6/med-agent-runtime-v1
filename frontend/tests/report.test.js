import test from 'node:test'
import assert from 'node:assert/strict'
import { reportItemsForDisplay } from '../src/utils/report.js'

function fact(name, value, options = {}) {
  const { subject = 'self', origin = 'current_user', quote = value, state = 'provided' } = options
  return { name, value, state, source: { subject, origin, quote, turn_id: 'synthetic-turn' } }
}

function reply() {
  return { subject: 'self', risk: { level: 'ROUTINE' }, intent: { subject: 'self', entities: {
    report_item: fact('report_item', '项目A'), value: fact('value', '10'),
    unit: fact('unit', 'mmol/L'), reference_range: fact('reference_range', '3-5')
  } }, agent_results: [{ role: 'report_explainer', status: 'needs_clarification', missing_fields: ['source'],
    payload: { report_items: [{ name: '项目A', value: '10', unit: 'mmol/L', reference_range: '3-5' }] } }] }
}

test('shows only user-grounded report values for the active subject', () => {
  assert.deepEqual(reportItemsForDisplay(reply()), [{ name: '项目A', value: '10', unit: 'mmol/L', reference_range: '3-5' }])
  const previousTurn = reply()
  previousTurn.intent.entities.value = fact('value', '10', { origin: 'history_user', quote: '用户原文：10' })
  assert.equal(reportItemsForDisplay(previousTurn)[0].value, '10')
})

test('keeps missing or unconfirmed values empty without guessing', () => {
  const partial = reply()
  partial.agent_results[0].payload.report_items[0].unit = null
  partial.intent.entities.reference_range = fact('reference_range', null, { state: 'unknown', quote: '范围不清楚' })
  assert.deepEqual(reportItemsForDisplay(partial), [{ name: '项目A', value: '10', unit: null, reference_range: null }])
})

test('filters invented values, wrong subjects, unsupported quotes and partial numbers', () => {
  const wrong = reply()
  wrong.intent.entities.value = fact('value', '10', { subject: 'mother' })
  wrong.intent.entities.unit = fact('unit', 'mmol/L', { origin: 'assistant' })
  wrong.intent.entities.reference_range = fact('reference_range', '3-5', { quote: '未给出范围' })
  assert.deepEqual(reportItemsForDisplay(wrong), [{ name: '项目A', value: null, unit: null, reference_range: null }])
  const partialNumber = reply()
  partialNumber.agent_results[0].payload.report_items[0].value = '0'
  partialNumber.intent.entities.value = fact('value', '0', { quote: '数值是10' })
  assert.equal(reportItemsForDisplay(partialNumber)[0].value, null)
})

test('rejects ungrounded names, subject ambiguity, critical risk and malformed responses', () => {
  const wrongName = reply()
  wrongName.agent_results[0].payload.report_items[0].name = '项目B'
  assert.deepEqual(reportItemsForDisplay(wrongName), [])
  const wrongSubject = reply()
  wrongSubject.intent.subject = 'mother'
  assert.deepEqual(reportItemsForDisplay(wrongSubject), [])
  const ambiguous = reply()
  ambiguous.intent.subject_ambiguous = true
  assert.deepEqual(reportItemsForDisplay(ambiguous), [])
  const critical = reply()
  critical.risk.level = 'CRITICAL'
  assert.deepEqual(reportItemsForDisplay(critical), [])
  assert.deepEqual(reportItemsForDisplay(null), [])
  assert.deepEqual(reportItemsForDisplay({ subject: 'self', intent: { subject: 'self' } }), [])
})

test('only presents clarification results from the report role, including a supporting role', () => {
  const supported = reply()
  supported.primary_agent = 'visit_guide'
  supported.agent_results.unshift({ role: 'visit_guide', status: 'answered', payload: {} })
  assert.equal(reportItemsForDisplay(supported)[0].name, '项目A')
  supported.agent_results[1].status = 'answered'
  assert.deepEqual(reportItemsForDisplay(supported), [])
})
