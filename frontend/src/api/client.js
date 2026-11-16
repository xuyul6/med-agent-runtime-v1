// Every HTTP call the workbench makes goes through this file.

export class ApiError extends Error {
  constructor (status, code, message) {
    super(message || code)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

export function errorCode (status, body) {
  const detail = body && body.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && !Array.isArray(detail) && typeof detail.code === 'string') return detail.code
  if (status === 422) return 'invalid_request'
  if (status === 404) return 'not_found'
  return `http_${status}`
}

export function createClient ({ base = '/api', fetchImpl = globalThis.fetch, timeoutMs = 120000 } = {}) {
  async function request (path, { method = 'GET', query, body, timeout = timeoutMs } = {}) {
    let url = base + path
    if (query) {
      const params = new URLSearchParams()
      for (const [key, value] of Object.entries(query)) {
        if (value !== undefined && value !== null && value !== '') params.set(key, String(value))
      }
      const text = params.toString()
      if (text) url += '?' + text
    }
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), timeout)
    let response
    try {
      response = await fetchImpl(url, {
        method,
        headers: body === undefined ? { accept: 'application/json' } : { accept: 'application/json', 'content-type': 'application/json' },
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal
      })
    } catch (error) {
      throw new ApiError(0, error && error.name === 'AbortError' ? 'timeout' : 'network_error')
    } finally {
      clearTimeout(timer)
    }
    let data = null
    try {
      data = await response.json()
    } catch {
      data = null
    }
    if (!response.ok) throw new ApiError(response.status, errorCode(response.status, data))
    return data
  }

  return {
    health: () => request('/health', { timeout: 8000 }),
    chat: ({ text, user, session, subject }) => request('/chat', { method: 'POST', body: { text, user, session: session || '', subject: subject || 'self' } }),
    catalog: () => request('/appointments/catalog'),
    slots: ({ hospitalId, departmentId, day, window }) => request('/appointments/slots', {
      query: { hospital_id: hospitalId, department_id: departmentId, day, window }
    }),
    bookings: (user) => request('/appointments', { query: { user } }),
    draftCreate: ({ user, subject, slotId }) => request('/appointments/drafts', {
      method: 'POST', body: { user, subject, action: 'create', slot_id: slotId }
    }),
    draftCancel: ({ user, subject, appointmentId }) => request('/appointments/drafts', {
      method: 'POST', body: { user, subject, action: 'cancel', appointment_id: appointmentId }
    }),
    confirm: ({ user, draftId, token }) => request('/appointments/confirm', {
      method: 'POST', body: { user, draft_id: draftId, token }
    }),
    forgetUser: (user) => request('/users/' + encodeURIComponent(user), { method: 'DELETE' }),
    monitor: () => request('/monitor'),
    agents: () => request('/agents'),
    tools: () => request('/tools')
  }
}

export const api = createClient()
