import { safeLink } from '../api/client.js'

export function emptyMedicationMetadata() {
  return { organization: '', document_type: '', source_version: '', updated_on: '', published_on: '', checked_on: '',
    sections: '', use_conditions: '', terms_url: '', region: '', population: '', label_population: '',
    product_name: '', formulation: '', strength: '' }
}

function required(value, label) {
  const text = typeof value === 'string' ? value.trim() : ''
  if (!text) throw new Error(`请填写${label}`)
  return text
}

function dateValue(value, label, optional = false) {
  const text = typeof value === 'string' ? value.trim() : ''
  if (!text && optional) return null
  if (!/^\d{4}-\d{2}-\d{2}$/.test(text)) throw new Error(`请填写有效的${label}`)
  const date = new Date(`${text}T00:00:00Z`)
  if (!Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== text) throw new Error(`请填写有效的${label}`)
  return text
}

export function buildKnowledgeDocument(draft, drugAliases, metadata, visitPurpose = '') {
  const { entities: ignoredEntities, provenance: ignoredProvenance, medication_scope: ignoredScope, visit_scope: ignoredVisitScope, ...base } = draft
  const document = { ...base, entities: [] }
  if (draft.domain === 'visit') {
    if (!['general_preparation', 'institution_specific'].includes(visitPurpose)) throw new Error('请选择就医资料用途')
    return { ...document, visit_scope: { purpose: visitPurpose } }
  }
  if (draft.domain !== 'medicine') return document
  const aliases = drugAliases.split(/[,，]/).map(value => value.trim()).filter(Boolean)
  if (!aliases.length || aliases.length > 20) throw new Error('请填写 1 至 20 个药名或别名')
  const type = required(metadata.document_type, '资料类型')
  if (!['public_overview', 'product_label'].includes(type)) throw new Error('请选择有效的资料类型')
  const termsUrl = required(metadata.terms_url, '使用条款网址')
  if (!safeLink(termsUrl)) throw new Error('使用条款网址须为 HTTP 或 HTTPS')
  const sections = metadata.sections.split(/\r?\n/).map(value => value.trim()).filter(Boolean)
  if (!sections.length) throw new Error('请填写至少一个核对章节')
  if (metadata.population !== 'adults_18_plus') throw new Error('请选择当前支持的资料人群范围')
  const product = type === 'product_label'
    ? { product_name: required(metadata.product_name, '产品名称'), formulation: required(metadata.formulation, '剂型'), strength: required(metadata.strength, '包装规格') }
    : { product_name: null, formulation: null, strength: null }
  return { ...document, entities: aliases,
    provenance: { organization: required(metadata.organization, '来源机构'), document_type: type,
      source_version: required(metadata.source_version, '来源版本'), updated_on: dateValue(metadata.updated_on, '来源更新日期'),
      published_on: dateValue(metadata.published_on, '来源发布日期', true), checked_on: dateValue(metadata.checked_on, '核对日期'),
      sections, use_conditions: required(metadata.use_conditions, '使用条件'), terms_url: termsUrl },
    medication_scope: { region: required(metadata.region, '资料地区'), population: metadata.population,
      label_population: required(metadata.label_population, '原文人群范围'), ...product } }
}
