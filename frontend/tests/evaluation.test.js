import test from 'node:test'
import assert from 'node:assert/strict'
import { createServer } from 'vite'
import { createSSRApp, h } from 'vue'
import { renderToString } from '@vue/server-renderer'
import { buildEvaluationRequest, checkSummary, comparisonLabel, countLabel, loadEvaluation, requestEvaluation, scoreLabel, usageLabel } from '../src/utils/evaluation.js'

const settings = { max_model_calls: 60, max_output_tokens: 75000, timeout_seconds: 300, compare_to: '', save_as: '' }

test('baseline names and explicit bounded limits map to the backend contract', () => {
  assert.deepEqual(buildEvaluationRequest(settings), { limits: { max_model_calls: 60, max_output_tokens: 75000, timeout_seconds: 300 } })
  assert.deepEqual(buildEvaluationRequest({ ...settings, compare_to: ' old-v4 ', save_as: 'new_v4' }), {
    limits: { max_model_calls: 60, max_output_tokens: 75000, timeout_seconds: 300 },
    baseline: { compare_to: 'old-v4', save_as: 'new_v4' }
  })
})

test('invalid limits and unsafe, duplicate or overlong baseline names fail before sending', async () => {
  const bad = [
    { max_model_calls: 0 }, { max_model_calls: 401 }, { max_model_calls: '1.5' },
    { max_output_tokens: '' }, { max_output_tokens: 400001 }, { timeout_seconds: 0 },
    { timeout_seconds: 601 }, { compare_to: '../baseline' }, { compare_to: '_bad' },
    { save_as: 'a'.repeat(65) }, { compare_to: 'same', save_as: 'same' }
  ]
  for (const change of bad) {
    let calls = 0
    const state = { report: { value: { cases: [] } }, error: { value: '' } }
    await loadEvaluation(async () => { calls++; return { complete: true, cases: [] } }, state, { ...settings, ...change })
    assert.equal(calls, 0)
    assert.equal(state.report.value, null)
    assert.match(state.error.value, /评测未完成/)
  }
})

test('valid settings reach the request without sending any implicit custom cases', async () => {
  let body
  const state = { report: { value: null }, error: { value: '' } }
  await loadEvaluation(async (path, payload) => { body = payload; return { complete: true, cases: [] } }, state,
    { ...settings, compare_to: 'baseline1', save_as: 'baseline2' })
  assert.equal(body.baseline.compare_to, 'baseline1')
  assert.equal(body.baseline.save_as, 'baseline2')
  assert.equal(body.limits.max_model_calls, 60)
  assert.equal('dialog_cases' in body, false)
  assert.equal('intent_cases' in body, false)
  assert.equal(state.error.value, '')
})

test('evaluation requests the actual endpoint and rejects malformed or failed reports', async () => {
  const report = { complete: false, cases: [], not_run: ['case-1'] }
  let sent
  const response = await requestEvaluation(async (path, body) => { sent = { path, body }; return report })
  assert.equal(response, report)
  assert.deepEqual(sent, { path: '/eval/run', body: {} })
  await assert.rejects(requestEvaluation(async () => ({ complete: true })), /格式无效/)
  await assert.rejects(requestEvaluation(async () => { throw new Error('HTTP 503') }), /HTTP 503/)
})

test('a new run immediately clears the old report and preserves its own failure', async () => {
  const state = { report: { value: { cases: [{ id: 'old' }] } }, error: { value: 'old error' } }
  let fail
  const pending = loadEvaluation(() => new Promise((resolve, reject) => { fail = reject }), state)
  assert.equal(state.report.value, null)
  assert.equal(state.error.value, '')
  fail(new Error('HTTP 503'))
  await pending
  assert.equal(state.report.value, null)
  assert.equal(state.error.value, '评测未完成：HTTP 503')
  await loadEvaluation(async () => ({ complete: true, cases: [] }), state)
  assert.equal(state.error.value, '')
  assert.deepEqual(state.report.value, { complete: true, cases: [] })
  await loadEvaluation(async () => { throw new Error('HTTP 409') }, state)
  assert.match(state.error.value, /可能已在运行/)
  assert.match(state.error.value, /基线不可用、已存在或不兼容/)
  assert.equal(state.report.value, null)
})

test('checks distinguish missing, empty, failed, unknown and passed', () => {
  assert.equal(checkSummary(null), '检查结果不可用')
  assert.equal(checkSummary({}), '无检查记录')
  assert.equal(checkSummary({ role: true, 'fault:tool_timeout_observed': false, risk: null }), '未通过：fault:tool_timeout_observed；结果未知：risk')
  assert.equal(checkSummary({ risk: true }), '已记录的检查均通过')
})

test('comparison never equates absence of comparison with no regression', () => {
  assert.equal(comparisonLabel({ status: 'not_requested', regressions: [] }), '未请求基线比较')
  assert.match(comparisonLabel({ status: 'incomplete_run' }), /未比较/)
  assert.equal(comparisonLabel({ status: 'compared' }), '已执行基线比较')
  assert.equal(comparisonLabel(null), '基线比较状态不可用')
})

test('zero is distinct from missing for counts, scores and usage', () => {
  assert.equal(countLabel(0), '0')
  assert.equal(countLabel(null), '未提供')
  assert.equal(scoreLabel(0), '0.000')
  assert.equal(scoreLabel(null), '未计量')
  assert.equal(usageLabel({ measurement: 'not_used', input_tokens: 0 }, 'input_tokens'), '0')
  assert.equal(usageLabel({ measurement: 'provider', input_tokens: 10 }, 'input_tokens'), '10')
  assert.equal(usageLabel({ measurement: 'unavailable', input_tokens: 0 }, 'input_tokens'), '未计量')
  assert.equal(usageLabel({ measurement: 'provider', input_tokens: null }, 'input_tokens'), '未计量')
})

test('report renders the actual failed turn, missing score and unrequested comparison', async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'silent' })
  try {
    const { default: EvaluationReport } = await server.ssrLoadModule('/src/components/EvaluationReport.vue')
    const report = { complete: false, passed: 0, total: 2, executed: 1, rule_passed: 0,
      not_run: ['unrun'], judge: { enabled: true, scored: 0, total: 1, missing: 1, errors: 1 },
      comparison: { status: 'not_requested', regressions: [] },
      cases: [{ id: 'case-a', passed: false, kind: 'dialog', not_run_turns: [2], turns: [{ turn: 1,
        passed: false, checks: { 'fault:tool_timeout_observed': false }, safety_checks: { citations: true },
        judge_error: 'RuntimeError', quality: null, trace: { request_id: 'trace-1' } }] }] }
    const html = await renderToString(createSSRApp({ render: () => h(EvaluationReport, { report }) }))
    for (const text of ['未完整运行', '未运行案例：unrun', '未运行轮次：2', '未请求基线比较', '第 1 轮',
      'fault:tool_timeout_observed', 'Judge 错误：RuntimeError', '无有效分数', 'trace-1', '未计量']) {
      assert.ok(html.includes(text), `missing ${text}`)
    }
    assert.ok(!html.includes('未发现已报告的回归'))
  } finally {
    await server.close()
  }
})

test('compared report distinguishes saved baselines, missing scores and changed variables', async () => {
  const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', logLevel: 'silent' })
  try {
    const { default: EvaluationReport } = await server.ssrLoadModule('/src/components/EvaluationReport.vue')
    const report = { complete: true, cases: [], baseline: { saved: false, reason: 'incomplete_run' },
      comparison: { status: 'compared', regressions: [], metrics: [
        { metric: 'quality_mean', baseline: 0.8, current: null, status: 'missing' }], variables: [{ field: 'model' }] } }
    const html = await renderToString(createSSRApp({ render: () => h(EvaluationReport, { report }) }))
    for (const text of ['已执行基线比较', '基线未保存：incomplete_run', 'quality_mean', '0.800', '未计量', 'missing', '实验变量发生变化：model']) {
      assert.ok(html.includes(text), `missing ${text}`)
    }
  } finally { await server.close() }
})
