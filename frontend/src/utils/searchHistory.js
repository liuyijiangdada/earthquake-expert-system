const KEY = 'biyelunwen.searchHistory.v1'
const MAX_ITEMS = 30

function safeJsonParse(s, fallback) {
  try {
    const v = JSON.parse(s)
    return v ?? fallback
  } catch {
    return fallback
  }
}

function normalizeText(t) {
  return String(t || '').trim()
}

export function loadSearchHistory() {
  if (typeof window === 'undefined') return []
  const raw = window.localStorage?.getItem?.(KEY)
  const arr = Array.isArray(raw ? safeJsonParse(raw, []) : []) ? safeJsonParse(raw, []) : []
  return arr
    .filter((x) => x && typeof x.text === 'string' && x.text.trim() !== '')
    .map((x) => ({
      text: normalizeText(x.text),
      ts: Number(x.ts) || Date.now(),
    }))
    .slice(0, MAX_ITEMS)
}

export function saveSearchHistory(items) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage?.setItem?.(KEY, JSON.stringify(items.slice(0, MAX_ITEMS)))
  } catch {
    // ignore quota / disabled storage
  }
}

export function addSearchHistory(text) {
  const t = normalizeText(text)
  if (!t) return loadSearchHistory()
  const prev = loadSearchHistory()
  const next = [{ text: t, ts: Date.now() }, ...prev.filter((x) => normalizeText(x.text) !== t)].slice(0, MAX_ITEMS)
  saveSearchHistory(next)
  return next
}

export function removeSearchHistory(text) {
  const t = normalizeText(text)
  const prev = loadSearchHistory()
  const next = prev.filter((x) => normalizeText(x.text) !== t)
  saveSearchHistory(next)
  return next
}

export function clearSearchHistory() {
  if (typeof window === 'undefined') return
  try {
    window.localStorage?.removeItem?.(KEY)
  } catch {
    // ignore
  }
}

