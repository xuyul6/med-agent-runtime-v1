// Demo identities kept in this browser. There is no authentication: an id only separates one demo user's
// sessions, summaries and bookings from another's. Storage may be unavailable, so every access is guarded.

const CURRENT = 'medagent.user'
const KNOWN = 'medagent.users'
// Letters (any script), digits, '_' and '-', 2 to 32 characters. The backend accepts up to 100.
const PATTERN = /^[\p{L}\p{N}_-]{2,32}$/u
const LIMIT = 8

function read (storage, key) {
  try {
    return storage ? storage.getItem(key) : null
  } catch {
    return null
  }
}

function write (storage, key, value) {
  try {
    if (storage) storage.setItem(key, value)
  } catch {}
}

export function newUserId (random = () => Math.random()) {
  return 'demo-' + Math.floor(random() * 36 ** 6).toString(36).padStart(6, '0').slice(-6)
}

export function knownUsers (storage = globalThis.localStorage) {
  try {
    const list = JSON.parse(read(storage, KNOWN) || '[]')
    return Array.isArray(list) ? [...new Set(list.filter(item => typeof item === 'string' && PATTERN.test(item)))].slice(0, LIMIT) : []
  } catch {
    return []
  }
}

export function normalizeUserId (value) {
  return String(value ?? '').trim()
}

// Makes the id the current user and moves it to the front of the remembered list.
export function useUser (value, storage = globalThis.localStorage) {
  const id = normalizeUserId(value)
  if (!PATTERN.test(id)) throw new Error('invalid_demo_user')
  write(storage, CURRENT, id)
  write(storage, KNOWN, JSON.stringify([id, ...knownUsers(storage).filter(item => item !== id)].slice(0, LIMIT)))
  return id
}

export function demoUser (storage = globalThis.localStorage, random = () => Math.random()) {
  const saved = read(storage, CURRENT)
  return useUser(saved && PATTERN.test(saved) ? saved : newUserId(random), storage)
}

export function isValidUserId (value) {
  return PATTERN.test(normalizeUserId(value))
}

// Drops the id from this browser's list. When it was the current user, the next remembered id (or a new one) takes over.
export function removeUser (value, storage = globalThis.localStorage, random = () => Math.random()) {
  const id = normalizeUserId(value)
  const rest = knownUsers(storage).filter(item => item !== id)
  write(storage, KNOWN, JSON.stringify(rest))
  const current = read(storage, CURRENT)
  if (current !== id) return current && PATTERN.test(current) ? current : useUser(rest[0] || newUserId(random), storage)
  return useUser(rest[0] || newUserId(random), storage)
}
