export async function queryLlm(input) {
  const r = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query_type: 'llm', params: { input } }),
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
