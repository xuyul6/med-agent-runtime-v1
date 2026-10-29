function quoteContainsValue(quote, value, field) {
  if (field !== 'value' || !/^[\d.]+$/.test(value)) return quote.includes(value)
  for (let index = quote.indexOf(value); index !== -1; index = quote.indexOf(value, index + 1)) {
    const before = quote[index - 1] || ''
    const after = quote[index + value.length] || ''
    if (!/[\d.]/.test(before) && !/[\d.]/.test(after)) return true
  }
  return false
}

function matchesUserField(fact, field, value, subject) {
  return fact?.name === field && fact.state === 'provided' && typeof value === 'string' && value.trim().length > 0 &&
    fact.value === value && fact.source?.subject === subject &&
    ['current_user', 'history_user'].includes(fact.source?.origin) &&
    typeof fact.source?.turn_id === 'string' && fact.source.turn_id.length > 0 &&
    typeof fact.source?.quote === 'string' && quoteContainsValue(fact.source.quote, value, field)
}

export function reportItemsForDisplay(reply) {
  if (!reply || typeof reply.subject !== 'string' || !reply.subject || reply.intent?.subject !== reply.subject ||
      reply.intent?.subject_ambiguous || !['ROUTINE', 'CAUTION'].includes(reply.risk?.level) ||
      !Array.isArray(reply.agent_results)) return []
  const facts = reply.intent?.entities || {}
  const result = reply.agent_results.find(entry => entry?.role === 'report_explainer' && entry.status === 'needs_clarification')
  if (!Array.isArray(result?.payload?.report_items)) return []
  return result.payload.report_items.flatMap(item => {
    if (!item || !matchesUserField(facts.report_item, 'report_item', item.name, reply.subject)) return []
    const fields = Object.fromEntries(['value', 'unit', 'reference_range'].map(field => [field,
      matchesUserField(facts[field], field, item[field], reply.subject) ? item[field] : null]))
    return [{ name: item.name, ...fields }]
  })
}
