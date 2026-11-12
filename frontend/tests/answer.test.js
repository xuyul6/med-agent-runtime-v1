import test from 'node:test'
import assert from 'node:assert/strict'
import { answerTextForDisplay, recalledNote } from '../src/utils/answer.js'

function reply() {
  return {
    primary_agent: 'drug_reference', question: '布洛芬适合什么？',
    answer: { text: '已核对布洛芬的公开概览来源（NHS 英国成人公开概览，版本 summary-2026-10-27，页面核对日期 2025-08-27）。该资料面向成人，不代表个人适用。', reference_keys: ['ibuprofen'] },
    references: [{ key: 'ibuprofen', version: 'summary-2026-10-27', provenance: { updated_on: '2025-08-27', checked_on: '2026-10-27' } }]
  }
}

test('omits only the standalone cited metadata header and keeps the answer', () => {
  const item = reply()
  assert.equal(answerTextForDisplay(item), '该资料面向成人，不代表个人适用。')
  assert.match(item.answer.text, /^已核对/) // The original API answer is not changed.
})

test('keeps explicitly requested provenance and non-medication answers', () => {
  const item = reply()
  item.question = '这个药的来源和核对日期是什么？'
  assert.equal(answerTextForDisplay(item), item.answer.text)
  item.question = '布洛芬适合什么？'
  item.primary_agent = 'report_explainer'
  assert.equal(answerTextForDisplay(item), item.answer.text)
})

test('keeps the original when reference, version or date is not verified', () => {
  for (const change of [
    item => { item.references = [] },
    item => { item.answer.reference_keys = [] },
    item => { item.references[0].version = 'different' },
    item => { item.references[0].provenance.updated_on = '2020-01-01' }
  ]) {
    const item = reply()
    change(item)
    assert.equal(answerTextForDisplay(item), item.answer.text)
  }
})

test('does not discard safety statements or the only answer', () => {
  for (const text of [
    '已核对孕妇不适用的公开概览来源（NHS 英国成人公开概览，版本 summary-2026-10-27）。请咨询药师。',
    '已核对布洛芬的公开概览来源（NHS 英国成人公开概览，版本 summary-2026-10-27，提示过敏风险）。请咨询药师。',
    '已核对布洛芬的公开概览来源（NHS 英国成人公开概览，版本 summary-2026-10-27）。'
  ]) {
    const item = reply()
    item.answer.text = text
    assert.equal(answerTextForDisplay(item), text)
  }
})

test('recalled summaries are described by day and kept apart from sources', () => {
  assert.equal(recalledNote([]), '')
  assert.equal(recalledNote(undefined), '')
  assert.equal(recalledNote(['not-a-day']), '')
  const note = recalledNote(['2026-11-05', '2026-10-30', '2026-11-05'])
  assert.match(note, /^本次带上了你以往的咨询摘要（10 月 30 日、11 月 5 日）/)
  assert.match(note, /不是资料来源/)
})
