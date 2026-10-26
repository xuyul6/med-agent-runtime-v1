export function safeLink(value) {
  try {
    const url = new URL(value)
    return ['https:', 'http:'].includes(url.protocol) ? url.href : ''
  } catch { return '' }
}

export function makeClient(fetcher = globalThis.fetch, prefix = '/api') {
  return async function request(path, body) {
    const response = await fetcher(prefix + path, {
      method: body === undefined ? 'GET' : 'POST',
      headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
      ...(body === undefined ? {} : { body: JSON.stringify(body) })
    })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return response.json()
  }
}

export function tokenLabel(usage) {
  return usage?.measurement === 'provider' && Number.isInteger(usage.input_tokens) ? String(usage.input_tokens) : '未计量'
}
