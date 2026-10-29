const object = value => value && typeof value === 'object' && !Array.isArray(value)
const validProfiles = value => object(value) && Array.isArray(value.agents) && value.agents.every(item => object(item) && typeof item.role === 'string')
const validSkills = value => object(value) && Array.isArray(value.skills) && value.skills.every(item => object(item) && typeof item.role === 'string')

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
