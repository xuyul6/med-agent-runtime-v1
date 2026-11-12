import test from 'node:test'
import assert from 'node:assert/strict'
import { SESSIONS_KEY, SESSION_LIMIT, forgetSession, rememberSession, savedSession } from '../src/utils/sessions.js'

function memoryStorage(initial = {}) {
  const data = { ...initial }
  return {
    getItem: key => (key in data ? data[key] : null),
    setItem: (key, value) => { data[key] = String(value) },
    removeItem: key => { delete data[key] },
    data
  }
}

test('a session is kept per user and subject and survives logging out', () => {
  const storage = memoryStorage()
  rememberSession(storage, 'demo:alice', 'self', 'session-a')
  rememberSession(storage, 'demo:alice', 'mother', 'session-m')
  rememberSession(storage, 'demo:bob', 'self', 'session-b')
  assert.equal(savedSession(storage, 'demo:alice', 'self'), 'session-a')
  assert.equal(savedSession(storage, 'demo:alice', 'mother'), 'session-m')
  assert.equal(savedSession(storage, 'demo:bob', 'self'), 'session-b')
  assert.equal(savedSession(storage, 'demo:carol', 'self'), '')
})

test('a later session replaces the earlier one and 新会话 forgets it', () => {
  const storage = memoryStorage()
  rememberSession(storage, 'demo:alice', 'self', 'first')
  rememberSession(storage, 'demo:alice', 'self', 'second')
  assert.equal(savedSession(storage, 'demo:alice', 'self'), 'second')
  forgetSession(storage, 'demo:alice', 'self')
  assert.equal(savedSession(storage, 'demo:alice', 'self'), '')
})

test('names cannot collide by joining user and subject', () => {
  const storage = memoryStorage()
  rememberSession(storage, 'demo:a|self', 'x', 'one')
  assert.equal(savedSession(storage, 'demo:a', 'self|x'), '')
})

test('only the most recent entries are kept', () => {
  const storage = memoryStorage()
  for (let index = 0; index <= SESSION_LIMIT; index += 1) rememberSession(storage, `user-${index}`, 'self', `s-${index}`)
  assert.equal(savedSession(storage, 'user-0', 'self'), '')
  assert.equal(savedSession(storage, `user-${SESSION_LIMIT}`, 'self'), `s-${SESSION_LIMIT}`)
  assert.equal(Object.keys(JSON.parse(storage.data[SESSIONS_KEY])).length, SESSION_LIMIT)
})

test('corrupt or blocked storage falls back to a fresh session', () => {
  assert.equal(savedSession(memoryStorage({ [SESSIONS_KEY]: 'not json' }), 'u', 'self'), '')
  assert.equal(savedSession(memoryStorage({ [SESSIONS_KEY]: '["array"]' }), 'u', 'self'), '')
  assert.equal(savedSession(memoryStorage({ [SESSIONS_KEY]: JSON.stringify({ '["u","self"]': 42 }) }), 'u', 'self'), '')
  const blocked = { getItem() { throw new Error('blocked') }, setItem() { throw new Error('blocked') } }
  assert.doesNotThrow(() => rememberSession(blocked, 'u', 'self', 's'))
  assert.equal(savedSession(blocked, 'u', 'self'), '')
  assert.equal(savedSession(null, 'u', 'self'), '')
  rememberSession(memoryStorage(), 'u', 'self', '')
})
