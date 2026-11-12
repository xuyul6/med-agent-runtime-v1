import test from 'node:test'
import assert from 'node:assert/strict'
import { activeUser, confirmationError, confirmPending, createDraft, currentLogin, listAppointments, loadUser, loginUser, logout, normalizeLogin, pendingCards, resetUser, searchSlots } from '../src/utils/appointments.js'

test('pendingCards maps reply actions', () => {
  const cards = pendingCards({ pending_actions: [{ kind: 'appointment_cancel', draft_id: 'd1', token: 't', summary: 's', expires_at: 'x' }] })
  assert.deepEqual(cards.map(card => [card.draftId, card.title, card.summary, card.expiresAt]), [['d1', '确认取消预约', 's', 'x']])
  assert.equal(pendingCards({ pending_actions: [{ kind: 'appointment_create', draft_id: 'd2', token: 't' }] })[0].title, '确认预约')
  assert.deepEqual(pendingCards({}), [])
  assert.deepEqual(pendingCards(null), [])
})

test('confirmPending posts draft and token', async () => {
  const calls = []
  await confirmPending((path, body) => { calls.push([path, body]); return Promise.resolve({ status: 'booked' }) }, 'u1', { draftId: 'd1', token: 't' })
  assert.deepEqual(calls, [['/appointments/confirm', { user: 'u1', draft_id: 'd1', token: 't' }]])
})

test('confirmationError maps status codes', () => {
  assert.equal(confirmationError(new Error('HTTP 410')), '确认已过期或号源日期已不可约，请重新发起')
  assert.equal(confirmationError(new Error('HTTP 409')), '这张卡片已经处理过，或号源已满')
  assert.equal(confirmationError(new Error('HTTP 403')), '确认凭证无效')
  assert.equal(confirmationError(new Error('HTTP 500')), '确认失败，请稍后重试')
  assert.equal(confirmationError('boom'), '确认失败，请稍后重试')
})

test('simulated user persists and resets', () => {
  const store = new Map()
  const storage = { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value) }
  let next = 0
  const makeId = () => `id-${++next}`
  assert.equal(loadUser(storage, makeId), 'id-1')
  assert.equal(loadUser(storage, makeId), 'id-1')
  assert.equal(resetUser(storage, makeId), 'id-2')
  assert.equal(loadUser(storage, makeId), 'id-2')
  const broken = { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') } }
  assert.equal(loadUser(broken, makeId), 'id-3')
  assert.equal(resetUser(broken, makeId), 'id-4')
  assert.equal(loadUser(null, makeId), 'id-5')
})

test('a draft lost on the server asks to start again', () => {
  assert.equal(confirmationError(new Error('HTTP 404')), '草稿已失效，请重新发起')
})

function memoryStorage() {
  const store = new Map()
  return { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value), removeItem: key => store.delete(key) }
}

test('demo login normalizes names and survives broken storage', () => {
  assert.equal(normalizeLogin(' 小明 '), '小明')
  assert.equal(normalizeLogin('user_1-a'), 'user_1-a')
  for (const bad of ['a b', '', '   ', 'x'.repeat(33), '<script>', null]) assert.equal(normalizeLogin(bad), null)
  const storage = memoryStorage()
  assert.equal(loginUser(storage, ' 小明 '), 'demo:小明')
  assert.equal(currentLogin(storage), '小明')
  assert.equal(loginUser(storage, 'a b'), null)
  assert.equal(currentLogin(storage), '小明')
  logout(storage)
  assert.equal(currentLogin(storage), null)
  const broken = { getItem: () => { throw new Error('blocked') }, setItem: () => { throw new Error('blocked') }, removeItem: () => { throw new Error('blocked') } }
  assert.equal(currentLogin(broken), null)
  assert.equal(loginUser(broken, '小明'), 'demo:小明')
  assert.doesNotThrow(() => logout(broken))
})

test('activeUser prefers the login over the guest id', () => {
  const storage = memoryStorage()
  const makeId = () => 'guest-1'
  assert.deepEqual(activeUser(storage, makeId), { user: 'guest-1', name: null })
  loginUser(storage, '小红')
  assert.deepEqual(activeUser(storage, makeId), { user: 'demo:小红', name: '小红' })
})

test('createDraft posts the manual draft and returns a card', async () => {
  const calls = []
  const request = (path, body) => { calls.push([path, body]); return Promise.resolve({ kind: 'appointment_create', draft_id: 'd9', token: 't', summary: 's', expires_at: 'x' }) }
  const card = await createDraft(request, { user: 'demo:小红', subject: 'self', action: 'create', slot_id: 'h:d:2026-11-05:am' })
  assert.deepEqual(calls, [['/appointments/drafts', { user: 'demo:小红', subject: 'self', action: 'create', slot_id: 'h:d:2026-11-05:am' }]])
  assert.equal(card.draftId, 'd9')
  assert.equal(card.title, '确认预约')
})

test('searchSlots and listAppointments build encoded queries', async () => {
  const paths = []
  const request = path => { paths.push(path); return Promise.resolve([]) }
  await searchSlots(request, { hospital_id: 'demo-general', department_id: 'neurology', day: '2026-11-05', window: 'am' })
  await searchSlots(request, { hospital_id: 'demo-general', department_id: 'neurology', day: '2026-11-05', window: '' })
  await listAppointments(request, 'demo:小红')
  assert.deepEqual(paths, ['/appointments/slots?hospital_id=demo-general&department_id=neurology&day=2026-11-05&window=am',
    '/appointments/slots?hospital_id=demo-general&department_id=neurology&day=2026-11-05',
    '/appointments?user=demo%3A%E5%B0%8F%E7%BA%A2'])
})
