import { test } from 'node:test'
import assert from 'node:assert/strict'
import { ApiError, createClient, errorCode } from '../src/api/client.js'

function recorder (responses) {
  const calls = []
  const fetchImpl = async (url, options) => {
    calls.push({ url, options })
    const next = responses.shift()
    if (next instanceof Error) throw next
    return {
      ok: next.status >= 200 && next.status < 300,
      status: next.status,
      json: async () => {
        if (next.body === undefined) throw new SyntaxError('no body')
        return next.body
      }
    }
  }
  return { calls, fetchImpl }
}

test('chat posts the question with session and subject defaults', async () => {
  const { calls, fetchImpl } = recorder([{ status: 200, body: { request_id: 'r1' } }])
  const client = createClient({ fetchImpl })
  const reply = await client.chat({ text: '咳嗽', user: 'demo-abc123' })
  assert.equal(reply.request_id, 'r1')
  assert.equal(calls[0].url, '/api/chat')
  assert.equal(calls[0].options.method, 'POST')
  assert.deepEqual(JSON.parse(calls[0].options.body), { text: '咳嗽', user: 'demo-abc123', session: '', subject: 'self' })
})

test('slot search drops empty query values', async () => {
  const { calls, fetchImpl } = recorder([{ status: 200, body: [] }])
  const client = createClient({ fetchImpl })
  await client.slots({ hospitalId: 'demo-general', departmentId: 'cardio', day: '2026-11-17', window: undefined })
  assert.equal(calls[0].url, '/api/appointments/slots?hospital_id=demo-general&department_id=cardio&day=2026-11-17')
})

test('drafts send exactly one target for each action', async () => {
  const { calls, fetchImpl } = recorder([{ status: 200, body: {} }, { status: 200, body: {} }])
  const client = createClient({ fetchImpl })
  await client.draftCreate({ user: 'u', subject: 'self', slotId: 's1' })
  await client.draftCancel({ user: 'u', subject: 'mother', appointmentId: 'a1' })
  assert.deepEqual(JSON.parse(calls[0].options.body), { user: 'u', subject: 'self', action: 'create', slot_id: 's1' })
  assert.deepEqual(JSON.parse(calls[1].options.body), { user: 'u', subject: 'mother', action: 'cancel', appointment_id: 'a1' })
})

test('confirm carries the token only in the request body', async () => {
  const { calls, fetchImpl } = recorder([{ status: 200, body: { status: 'booked' } }])
  const client = createClient({ fetchImpl })
  await client.confirm({ user: 'u', draftId: 'draft-1', token: 'secret-token' })
  assert.equal(calls[0].url, '/api/appointments/confirm')
  assert.ok(!calls[0].url.includes('secret-token'))
  assert.deepEqual(JSON.parse(calls[0].options.body), { user: 'u', draft_id: 'draft-1', token: 'secret-token' })
})

test('failed responses become ApiError with the backend code', async () => {
  const { fetchImpl } = recorder([
    { status: 409, body: { detail: 'slot_full' } },
    { status: 422, body: { detail: { code: 'invalid_skills', reason: 'x' } } },
    { status: 502 }
  ])
  const client = createClient({ fetchImpl })
  await assert.rejects(client.confirm({ user: 'u', draftId: 'd', token: 't' }), error => error instanceof ApiError && error.status === 409 && error.code === 'slot_full')
  await assert.rejects(client.health(), error => error.code === 'invalid_skills')
  await assert.rejects(client.health(), error => error.status === 502 && error.code === 'http_502')
})

test('network failures and timeouts are distinguished', async () => {
  const abort = new Error('aborted')
  abort.name = 'AbortError'
  const { fetchImpl } = recorder([new TypeError('failed to fetch'), abort])
  const client = createClient({ fetchImpl })
  await assert.rejects(client.health(), error => error.status === 0 && error.code === 'network_error')
  await assert.rejects(client.health(), error => error.status === 0 && error.code === 'timeout')
})

test('errorCode falls back by status when the body has no detail', () => {
  assert.equal(errorCode(404, null), 'not_found')
  assert.equal(errorCode(422, { detail: [{ loc: ['body'] }] }), 'invalid_request')
  assert.equal(errorCode(500, {}), 'http_500')
})

test('forgetUser deletes by an encoded path', async () => {
  const { calls, fetchImpl } = recorder([{ status: 200, body: { user: '小王/1', sessions: 1, summaries: 2, bookings: 0 } }])
  const client = createClient({ fetchImpl })
  const result = await client.forgetUser('小王/1')
  assert.equal(calls[0].url, '/api/users/%E5%B0%8F%E7%8E%8B%2F1')
  assert.equal(calls[0].options.method, 'DELETE')
  assert.equal(calls[0].options.body, undefined)
  assert.equal(result.summaries, 2)
})
