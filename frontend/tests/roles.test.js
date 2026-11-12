import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { jsonSchema, loadRoleCatalog, roleNames, roleSkill, stringList } from '../src/utils/roles.js'

test('loads real profile and skill responses independently and matches by role', async () => {
  const replies = {
    '/agents': { agents: [{ role: 'general', mission: '澄清目的', input_contract: { type: 'object' } }, { role: 'visit_guide' }] },
    '/skills': { snapshot_id: 'snapshot', skills: [{ role: 'visit_guide', version: '1.2.0' }, { role: 'general', version: '1.0.0' }] }
  }
  const result = await loadRoleCatalog(async path => replies[path])
  assert.equal(result.profiles.length, 2)
  assert.equal(roleSkill(result.skills, 'general').version, '1.0.0')
  assert.equal(roleSkill(result.skills, 'unknown'), null)
  assert.match(jsonSchema(result.profiles[0].input_contract), /"type": "object"/)
})

test('one failed endpoint does not hide the other and malformed data is unavailable', async () => {
  const profilesOnly = await loadRoleCatalog(async path => {
    if (path === '/skills') throw new Error('unavailable')
    return { agents: [{ role: 'general' }] }
  })
  assert.equal(profilesOnly.profiles.length, 1)
  assert.equal(profilesOnly.skills, null)
  const skillsOnly = await loadRoleCatalog(async path => path === '/skills' ? { skills: [] } : { agents: [null] })
  assert.equal(skillsOnly.profiles, null)
  assert.deepEqual(skillsOnly.skills.skills, [])
  const empty = await loadRoleCatalog(async path => path === '/agents' ? { agents: [] } : { skills: [] })
  assert.deepEqual(empty.profiles, [])
  assert.equal(roleSkill(empty.skills, 'general'), null)
})

test('schema and chapter rendering preserve unavailable and empty states', () => {
  assert.equal(jsonSchema(null), null)
  assert.equal(jsonSchema([]), null)
  assert.deepEqual(stringList(null), [])
  assert.deepEqual(stringList(['角色使命', 1, '交接条件']), ['角色使命', '交接条件'])
  assert.equal(roleSkill({ skills: [{ role: 'general' }, null] }, 'general'), null)
})

test('role names are the section titles the backend writes into merged answers', () => {
  const source = readFileSync(new URL('../../backend/medagent/understanding/routing.py', import.meta.url), 'utf8')
  const block = source.match(/^LABELS = \{([^}]*)\}/m)[1]
  const labels = Object.fromEntries([...block.matchAll(/"(\w+)":"([^"]+)"/g)].map(match => [match[1], match[2]]))
  assert.equal(Object.keys(labels).length, 7)
  assert.deepEqual(roleNames, labels)
})
