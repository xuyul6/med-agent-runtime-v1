export const USER_KEY = 'medagent.simulated-user'

const STATUS_TEXT = {
  404: '草稿已失效，请重新发起',
  409: '这张卡片已经处理过，或号源已满',
  410: '确认已过期或号源日期已不可约，请重新发起',
  403: '确认凭证无效'
}

export function pendingCards(reply) {
  const actions = Array.isArray(reply?.pending_actions) ? reply.pending_actions : []
  return actions.map(action => ({
    draftId: action.draft_id,
    token: action.token,
    summary: action.summary,
    expiresAt: action.expires_at,
    kind: action.kind,
    title: action.kind === 'appointment_cancel' ? '确认取消预约' : '确认预约'
  }))
}

// The only write into the mock hospital: the user's button press with the one-time token.
export function confirmPending(request, user, card) {
  return request('/appointments/confirm', { user, draft_id: card.draftId, token: card.token })
}

export function confirmationError(error) {
  const status = Number(String(error?.message ?? '').match(/HTTP (\d{3})/)?.[1])
  return STATUS_TEXT[status] || '确认失败，请稍后重试'
}

// A synthetic identity kept across reloads so bookings stay visible; storage failures fall back to a fresh one.
export function loadUser(storage, makeId) {
  try {
    const saved = storage?.getItem(USER_KEY)
    if (saved) return saved
  } catch { /* storage blocked: use a fresh identity */ }
  return resetUser(storage, makeId)
}

export function resetUser(storage, makeId) {
  const user = makeId()
  try { storage?.setItem(USER_KEY, user) } catch { /* storage blocked: the identity lasts for this page only */ }
  return user
}

export const LOGIN_KEY = 'medagent.demo-login'
const LOGIN_PATTERN = /^[\p{Script=Han}A-Za-z0-9_-]{1,32}$/u

// Demo login: a name, no password. The same name is the same user; this is not authentication.
export function normalizeLogin(name) {
  const value = typeof name === 'string' ? name.trim() : ''
  return LOGIN_PATTERN.test(value) ? value : null
}

export function loginUser(storage, name) {
  const value = normalizeLogin(name)
  if (!value) return null
  try { storage?.setItem(LOGIN_KEY, value) } catch { /* storage blocked: the login lasts for this page only */ }
  return 'demo:' + value
}

export function currentLogin(storage) {
  try { return normalizeLogin(storage?.getItem(LOGIN_KEY)) } catch { return null }
}

export function logout(storage) {
  try { storage?.removeItem(LOGIN_KEY) } catch { /* nothing stored */ }
}

export function activeUser(storage, makeId) {
  const name = currentLogin(storage)
  return name ? { user: 'demo:' + name, name } : { user: loadUser(storage, makeId), name: null }
}

export function listAppointments(request, user) {
  return request('/appointments?' + new URLSearchParams({ user }))
}

export function loadCatalog(request) {
  return request('/appointments/catalog')
}

export function searchSlots(request, query) {
  const params = Object.fromEntries(Object.entries(query).filter(([, value]) => value))
  return request('/appointments/slots?' + new URLSearchParams(params))
}

export async function createDraft(request, body) {
  return pendingCards({ pending_actions: [await request('/appointments/drafts', body)] })[0]
}
