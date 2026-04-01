export function escapeHtml(text) {
  const div = document.createElement('div')
  div.textContent = text == null ? '' : String(text)
  return div.innerHTML
}

export function formatBotHtml(text) {
  let t = escapeHtml(text || '')
  t = t.replace(/\n/g, '<br>')
  t = t.replace(/\*\*/g, '')
  return t
}

export function nowTimeStr() {
  return new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
}
