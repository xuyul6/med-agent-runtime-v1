const object = value => value && typeof value === 'object' && !Array.isArray(value)
const validProfiles = value => object(value) && Array.isArray(value.agents) && value.agents.every(item => object(item) && typeof item.role === 'string')
const validSkills = value => object(value) && Array.isArray(value.skills) && value.skills.every(item => object(item) && typeof item.role === 'string')

// The same names the backend puts in the section titles of a merged answer (LABELS in understanding/routing.py).
export const roleNames = { general: '需求澄清', symptom_guide: '症状记录', drug_reference: '药品资料', report_explainer: '报告整理', visit_guide: '就医准备', appointment: '预约', risk_handoff: '风险提示' }

export async function loadRoleCatalog(request) {
  const [profileResult, skillResult] = await Promise.allSettled([request('/agents'), request('/skills')])
  const profiles = profileResult.status === 'fulfilled' && validProfiles(profileResult.value) ? profileResult.value.agents : null
  const skills = skillResult.status === 'fulfilled' && validSkills(skillResult.value) ? skillResult.value : null
  return { profiles, skills }
}

export function roleSkill(snapshot, role) {
  if (!validSkills(snapshot)) return null
  return snapshot.skills.find(item => item.role === role) || null
}

export function stringList(value) {
  return Array.isArray(value) ? value.filter(item => typeof item === 'string') : []
}

export function jsonSchema(value) {
  return object(value) ? JSON.stringify(value, null, 2) : null
}
