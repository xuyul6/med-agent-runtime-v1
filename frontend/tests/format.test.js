import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
  ROLE_LABELS, citedReferences, dayLabel, errorMessage, label, milliseconds, paragraphs, percent, remaining,
  riskReason, safeUrl, sortedScores, tokens, toolCaller
} from '../src/lib/format.js'
import { demoUser, isValidUserId, knownUsers, newUserId, removeUser, useUser } from '../src/lib/identity.js'

test('safeUrl keeps only http and https links', () => {
  assert.equal(safeUrl('https://www.who.int/news'), 'https://www.who.int/news')
  assert.equal(safeUrl('http://example.org/a'), 'http://example.org/a')
  assert.equal(safeUrl('javascript:alert(1)'), null)
  assert.equal(safeUrl('data:text/html,<b>x</b>'), null)
  assert.equal(safeUrl('/relative'), null)
  assert.equal(safeUrl(undefined), null)
})

test('citedReferences follows answer keys and never invents a reference', () => {
  const reply = {
    answer: { reference_keys: ['b', 'missing', 'a', 'b'] },
    references: [{ key: 'a', title: 'A' }, { key: 'b', title: 'B' }, { key: 'c', title: 'C' }]
  }
  assert.deepEqual(citedReferences(reply).map(item => item.key), ['b', 'a'])
  assert.deepEqual(citedReferences({ answer: {}, references: [{ key: 'a' }] }), [])
  assert.deepEqual(citedReferences(null), [])
})

test('labels fall back to the raw code', () => {
  assert.equal(label(ROLE_LABELS, 'drug_reference'), '药品资料')
  assert.equal(label(ROLE_LABELS, 'new_role'), 'new_role')
  assert.equal(label(ROLE_LABELS, ''), '—')
  assert.equal(label(ROLE_LABELS, 'toString'), 'toString')
  assert.equal(riskReason('breathing'), '呼吸困难')
  assert.equal(riskReason('urgent_hint_cleared:quote'), '此前的紧急提示已解除')
})

test('unknown token usage is shown as unknown, not zero', () => {
  assert.equal(tokens({ calls: 0, input_tokens: null, output_tokens: null }), '未知')
  assert.equal(tokens(undefined), '未知')
  assert.equal(tokens({ input_tokens: 1200, output_tokens: 340 }), '1,200 / 340')
})

test('number formatting', () => {
  assert.equal(percent(0.709), '71%')
  assert.equal(percent(null), '—')
  assert.equal(milliseconds(23.6), '24 ms')
  assert.equal(milliseconds(4210), '4.2 s')
  assert.deepEqual(sortedScores({ a: 0.1, b: 0.7, c: null }), [['b', 0.7], ['a', 0.1]])
})

test('paragraphs splits on newlines and drops blanks', () => {
  assert.deepEqual(paragraphs('第一段\n\n第二段\n  \n第三段 '), ['第一段', '第二段', '第三段'])
  assert.deepEqual(paragraphs(undefined), [])
})

test('remaining counts down and marks expiry', () => {
  const now = Date.parse('2026-11-16T10:00:00+08:00')
  assert.deepEqual(remaining('2026-11-16T10:12:30+08:00', now), { expired: false, text: '12 分钟内有效' })
  assert.deepEqual(remaining('2026-11-16T10:00:20+08:00', now), { expired: false, text: '20 秒内有效' })
  assert.deepEqual(remaining('2026-11-16T09:59:00+08:00', now), { expired: true, text: '已过期' })
})

test('dayLabel names the weekday', () => {
  assert.equal(dayLabel('2026-11-17'), '11月17日 周二')
  assert.equal(dayLabel('not-a-date'), 'not-a-date')
})

test('errorMessage explains appointment conflicts', () => {
  assert.equal(errorMessage({ status: 409, code: 'slot_full' }), '这个时段已经约满。')
  assert.equal(errorMessage({ status: 503, code: 'http_503' }), '服务出错了，请稍后再试。')
  assert.equal(errorMessage({ status: 400, code: 'odd' }), '请求失败（odd）')
})

test('demoUser reuses a saved id and survives broken storage', () => {
  const store = new Map()
  const storage = { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, value) }
  const first = demoUser(storage, () => 0.5)
  assert.match(first, /^demo-[a-z0-9]{6}$/)
  assert.equal(demoUser(storage, () => 0.1), first)
  const broken = { getItem () { throw new Error('denied') }, setItem () { throw new Error('denied') } }
  assert.match(demoUser(broken, () => 0.25), /^demo-[a-z0-9]{6}$/)
})

function memoryStorage () {
  const store = new Map()
  return { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, String(value)) }
}

test('switching users keeps a most-recent-first list without duplicates', () => {
  const storage = memoryStorage()
  const first = demoUser(storage, () => 0.1)
  useUser('demo-bbbbbb', storage)
  useUser(first, storage)
  assert.deepEqual(knownUsers(storage), [first, 'demo-bbbbbb'])
  assert.equal(demoUser(storage), first)
})

test('self-chosen ids are accepted within the allowed characters', () => {
  const storage = memoryStorage()
  assert.equal(useUser('  tester_01 ', storage), 'tester_01')
  assert.equal(demoUser(storage), 'tester_01')
  assert.equal(useUser('小王-2', storage), '小王-2')
  assert.equal(isValidUserId('Alice'), true)
  assert.equal(isValidUserId('a'), false)
  assert.equal(isValidUserId('a'.repeat(33)), false)
  assert.equal(isValidUserId('alice bob'), false)
  assert.equal(isValidUserId('<script>'), false)
  assert.throws(() => useUser('', storage), /invalid_demo_user/)
  assert.match(newUserId(() => 0), /^demo-0{6}$/)
  storage.setItem('medagent.users', '["demo-aaaaaa","<script>","demo-aaaaaa"]')
  assert.deepEqual(knownUsers(storage), ['demo-aaaaaa'])
  storage.setItem('medagent.users', '{broken')
  assert.deepEqual(knownUsers(storage), [])
})

test('removing a user drops it from the list and hands over the current seat', () => {
  const storage = memoryStorage()
  useUser('alice', storage)
  useUser('bob', storage)
  assert.equal(removeUser('alice', storage), 'bob')
  assert.deepEqual(knownUsers(storage), ['bob'])
  assert.equal(removeUser('bob', storage, () => 0), 'demo-000000')
  assert.deepEqual(knownUsers(storage), ['demo-000000'])
})

test('toolCaller names who may call a tool, and reads an unknown list as unknown', () => {
  assert.equal(toolCaller(['code']), '代码')
  assert.equal(toolCaller(['model']), '模型')
  assert.equal(toolCaller(['code', 'model']), '代码+模型')
  assert.equal(toolCaller([]), '—')
  assert.equal(toolCaller(undefined), '—')
  assert.equal(toolCaller(['code', 'shell']), '—')
})
