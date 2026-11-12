import test from 'node:test'
import assert from 'node:assert/strict'
import { connectionLabel, degradedNotice, developerMode, visibleView, workspaceViews } from '../src/utils/developer.js'

test('only an explicit true from /health turns developer mode on', () => {
  assert.equal(developerMode({ developer_tools: true }), true)
  assert.equal(developerMode({ developer_tools: false }), false)
  assert.equal(developerMode({ developer_tools: 'true' }), false)
  assert.equal(developerMode({}), false)
  assert.equal(developerMode(null), false)
})

test('users see consultation and bookings only', () => {
  assert.deepEqual(workspaceViews(false).map(([key]) => key), ['chat', 'appointments'])
  assert.deepEqual(workspaceViews(true).map(([key]) => key), ['chat', 'appointments', 'knowledge', 'roles', 'evaluation'])
})

test('a developer view falls back to consultation when the mode is off', () => {
  assert.equal(visibleView('evaluation', false), 'chat')
  assert.equal(visibleView('roles', false), 'chat')
  assert.equal(visibleView('appointments', false), 'appointments')
  assert.equal(visibleView('evaluation', true), 'evaluation')
  assert.equal(visibleView('unknown', true), 'chat')
})

test('users get plain service wording, developers keep runtime wording', () => {
  assert.equal(connectionLabel(null, false), '未连接')
  assert.equal(connectionLabel({ preview: true, status: 'ok' }, false), '离线预览')
  assert.equal(connectionLabel({ status: 'ok' }, false), '服务正常')
  assert.equal(connectionLabel({ status: 'degraded' }, false), '部分服务不可用')
  assert.equal(connectionLabel({ status: 'ok' }, true), '在线模型')
  assert.equal(connectionLabel({ status: 'degraded' }, true), '依赖降级')
  assert.doesNotMatch(degradedNotice(false), /执行详情/)
  assert.match(degradedNotice(true), /执行详情/)
})
