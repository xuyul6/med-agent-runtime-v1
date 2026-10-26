import test from 'node:test'
import assert from 'node:assert/strict'
import { makeClient, safeLink, tokenLabel } from '../src/client.js'

test('unknown usage stays unknown instead of displaying zero', () => {
  assert.equal(tokenLabel({ measurement: 'unavailable', input_tokens: null }), '未计量')
  assert.equal(tokenLabel({ measurement: 'provider', input_tokens: 0 }), '0')
})

test('untrusted reference links cannot execute script or inline documents', () => {
  assert.equal(safeLink('javascript:alert(1)'), '')
  assert.equal(safeLink('data:text/html,hi'), '')
  assert.equal(safeLink('https://example.org/docs'), 'https://example.org/docs')
})

test('chat serializes the subject and preserves the API response', async () => {
  let sent
  const request = makeClient(async (url, options) => { sent = { url, options }; return { ok: true, json: async () => ({ session: 'new' }) } })
  assert.deepEqual(await request('/chat', { text: '虚构问题', subject: 'parent' }), { session: 'new' })
  assert.equal(sent.url, '/api/chat')
  assert.equal(JSON.parse(sent.options.body).subject, 'parent')
})

test('HTTP failures are surfaced without parsing an arbitrary error body', async () => {
  const request = makeClient(async () => ({ ok: false, status: 503, json: () => { throw new Error('must not parse') } }))
  await assert.rejects(request('/health'), /HTTP 503/)
})
