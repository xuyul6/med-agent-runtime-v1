// Remove only a standalone, duplicated provenance header; never rewrite medical content.
export function answerTextForDisplay(reply) {
  const text = reply?.answer?.text
  if (typeof text !== 'string' || reply?.primary_agent !== 'drug_reference' ||
      !Array.isArray(reply.references) || !Array.isArray(reply.answer.reference_keys) ||
      /出处|来源|参考文献|哪个机构|哪家机构|版本|核对日期|更新日期/.test(reply.question || '')) return text

  const match = text.match(/^(已核对[^。！？；，,（）\n]{1,40}(?:公开概览来源|公开概览资料|公开资料来源))（([^，,（）\n]+)[，,]\s*版本\s*([A-Za-z0-9._-]+)(?:[，,]\s*(?:页面)?核对日期\s*(\d{4}-\d{2}-\d{2}))?）[。]\s*/)
  if (!match || !text.slice(match[0].length).trim() ||
      /不|禁|风险|剂量|服用|过敏|孕|儿童|治疗|相互作用|适用|不可|建议|需要/.test(match[1] + match[2])) return text
  const source = reply.references.find(item => item?.version === match[3] && reply.answer.reference_keys.includes(item.key))
  if (!source) return text
  if (match[4] && ![source.provenance?.updated_on, source.provenance?.checked_on, source.provenance?.published_on].includes(match[4])) return text
  return text.slice(match[0].length).trimStart()
}

// Earlier summaries are context the model was given, not sources: say so beside the answer, apart from citations.
export function recalledNote(days) {
  const valid = (Array.isArray(days) ? days : []).filter(day => /^\d{4}-\d{2}-\d{2}$/.test(day))
  if (!valid.length) return ''
  const labels = [...new Set(valid)].sort().map(day => `${Number(day.slice(5, 7))} 月 ${Number(day.slice(8, 10))} 日`)
  return `本次带上了你以往的咨询摘要（${labels.join('、')}），回答可能参考了其中内容。摘要是对你之前说过的话的整理，不是资料来源。`
}
