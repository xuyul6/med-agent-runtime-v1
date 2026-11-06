// The last session per user and consultation subject, kept in this browser so logging out and back in,
// reloading or switching subjects continues it. "新会话" forgets it. Storage failures mean a fresh session.
export const SESSIONS_KEY = 'medagent.sessions'
export const SESSION_LIMIT = 50

const entryKey = (user, subject) => JSON.stringify([user, subject])

function readAll(storage) {
  try {
    const value = JSON.parse(storage?.getItem(SESSIONS_KEY) || '{}')
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {}
  } catch { return {} }
}

function writeAll(storage, entries) {
  try { storage?.setItem(SESSIONS_KEY, JSON.stringify(entries)) } catch { /* storage blocked: nothing is kept */ }
}

export function savedSession(storage, user, subject) {
  const value = readAll(storage)[entryKey(user, subject)]
  return typeof value === 'string' && value.length > 0 && value.length <= 100 ? value : ''
}

export function rememberSession(storage, user, subject, session) {
  if (!user || !subject || !session) return
  const entries = readAll(storage)
  const key = entryKey(user, subject)
  // Re-inserting moves the entry to the end, so the oldest entries are the first dropped.
  delete entries[key]
  entries[key] = session
  const keys = Object.keys(entries)
  for (const old of keys.slice(0, Math.max(0, keys.length - SESSION_LIMIT))) delete entries[old]
  writeAll(storage, entries)
}

export function forgetSession(storage, user, subject) {
  const entries = readAll(storage)
  delete entries[entryKey(user, subject)]
  writeAll(storage, entries)
}
