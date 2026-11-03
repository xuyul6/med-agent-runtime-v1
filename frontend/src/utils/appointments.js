export const USER_KEY = 'medagent.simulated-user'

const STATUS_TEXT = {
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
