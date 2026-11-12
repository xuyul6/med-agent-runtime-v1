import test from 'node:test'
import assert from 'node:assert/strict'
import { buildKnowledgeDocument, emptyMedicationMetadata } from '../src/utils/knowledge.js'
import { makeClient } from '../src/api/client.js'

const draft = { key: 'synthetic-key', title: '虚构药品概览', domain: 'medicine', text: '虚构资料',
  url: 'https://example.org/source', version: 'v1', scope: '虚构范围', approved: false }
const metadata = { ...emptyMedicationMetadata(), organization: '虚构机构', document_type: 'public_overview',
  source_version: 'v1', updated_on: '2026-01-02', checked_on: '2026-01-03', sections: ' 第一章\n 第二章 ',
  use_conditions: '仅供虚构测试', terms_url: 'https://example.org/terms', region: 'UK',
  population: 'adults_18_plus', label_population: '只覆盖 18 岁及以上' }

test('medicine overview contains provenance and an explicit adult scope', () => {
  const doc = buildKnowledgeDocument(draft, '虚构药， demo ', metadata)
  assert.deepEqual(doc.entities, ['虚构药', 'demo'])
  assert.deepEqual(doc.provenance.sections, ['第一章', '第二章'])
  assert.equal(doc.provenance.published_on, null)
  assert.equal(doc.medication_scope.population, 'adults_18_plus')
  assert.equal(doc.medication_scope.product_name, null)
  assert.equal(doc.approved, false)
})

test('product label requires product identity and preserves the entered values', () => {
  const label = { ...metadata, document_type: 'product_label', product_name: ' 示例品牌 ', formulation: '片剂', strength: '200mg' }
  const doc = buildKnowledgeDocument({ ...draft, approved: true }, '虚构药', label)
  assert.equal(doc.approved, true)
  assert.equal(doc.medication_scope.product_name, '示例品牌')
  assert.equal(doc.medication_scope.formulation, '片剂')
  assert.equal(doc.medication_scope.strength, '200mg')
  for (const key of ['product_name', 'formulation', 'strength']) {
    assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...label, [key]: ' ' }))
  }
})

test('rejects missing, malformed and unsafe medication metadata', () => {
  assert.throws(() => buildKnowledgeDocument(draft, '', metadata), /药名或别名/)
  assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...metadata, sections: ' \n ' }), /章节/)
  assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...metadata, organization: ' ' }), /来源机构/)
  assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...metadata, population: '' }), /人群范围/)
  assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...metadata, document_type: 'unexpected' }), /资料类型/)
  assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...metadata, terms_url: 'javascript:alert(1)' }), /HTTP 或 HTTPS/)
  assert.throws(() => buildKnowledgeDocument(draft, '虚构药', { ...metadata, checked_on: '2026-02-30' }), /核对日期/)
})

test('saving a non-medication document does not send stale medication metadata', async () => {
  const report = { ...draft, domain: 'report', entities: ['old'], medication_scope: { region: 'UK' }, provenance: { organization: 'old' } }
  const doc = buildKnowledgeDocument(report, '虚构药', metadata)
  let payload
  const request = makeClient(async (_url, options) => {
    payload = JSON.parse(options.body)
    return { ok: true, json: async () => ({ saved: 1 }) }
  })
  assert.deepEqual(await request('/knowledge', { documents: [doc] }), { saved: 1 })
  assert.deepEqual(payload.documents[0].entities, [])
  assert.equal('medication_scope' in payload.documents[0], false)
  assert.equal('provenance' in payload.documents[0], false)
})

test('visit document requires an explicit use and keeps it separate from approval', () => {
  const visit = { ...draft, domain: 'visit', approved: true }
  assert.throws(() => buildKnowledgeDocument(visit, '', metadata), /就医资料用途/)
  assert.throws(() => buildKnowledgeDocument(visit, '', metadata, 'unverified'), /就医资料用途/)
  for (const purpose of ['general_preparation', 'institution_specific']) {
    const doc = buildKnowledgeDocument(visit, '', metadata, purpose)
    assert.deepEqual(doc.visit_scope, { purpose })
    assert.equal(doc.approved, true)
  }
})

test('switching domains does not leak a previously selected visit use', () => {
  const oldVisit = { purpose: 'general_preparation' }
  const report = buildKnowledgeDocument({ ...draft, domain: 'report', visit_scope: oldVisit }, '', metadata, 'general_preparation')
  const medicine = buildKnowledgeDocument({ ...draft, visit_scope: oldVisit }, '虚构药', metadata, 'institution_specific')
  assert.equal('visit_scope' in report, false)
  assert.equal('visit_scope' in medicine, false)
})
