export async function queryLlm(input, history = []) {
  const r = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query_type: 'llm',
      params: { input, history },
    }),
  })
  const data = await r.json().catch(() => ({}))
  return { ok: r.ok, data }
}

export async function queryKgAll() {
  const r = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_type: 'kg', params: { type: 'all' } }),
  })
  return r.json().catch(() => ({}))
}

export async function updateEarthquakeData() {
  const r = await fetch('/api/update-data', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  return r.json().catch(() => ({}))
}

export async function classifyPhase(text) {
  const r = await fetch('/api/phase-classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  return r.json().catch(() => ({}))
}

export async function submitFeedback(payload) {
  const r = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  const data = await r.json().catch(() => ({}))
  return { ok: r.ok, data }
}

export async function login(username, password) {
  const r = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  const data = await r.json().catch(() => ({}))
  return { ok: r.ok, data }
}

export async function logout() {
  const r = await fetch('/api/auth/logout', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  const data = await r.json().catch(() => ({}))
  return { ok: r.ok, data }
}
