/** 从界面消息列表提取最近 N 轮对话，供后端注入上下文 */

const MAX_CONTENT_LEN = 500

/**
 * @param {Array} messages App.vue 中的 messages
 * @param {number} maxRounds 轮数（1 轮 = 用户 + 助手）
 * @returns {{ role: string, content: string }[]}
 */
export function buildChatHistory(messages, maxRounds = 3) {
  const turns = []
  let pendingUser = null

  for (const m of messages || []) {
    if (m.role === 'user') {
      pendingUser = m
      continue
    }
    if (m.role !== 'bot') continue

    // 欢迎语、纯系统提示等不参与记忆
    if (m.feedback === false) {
      pendingUser = null
      continue
    }

    if (!pendingUser) continue

    let userText = (pendingUser.text || '').trim()
    if (pendingUser.imageUrl) {
      userText = userText ? `[用户上传了图片] ${userText}` : '[用户上传了图片]'
    }
    const botText = (m.text || '').trim()
    if (!userText && !botText) {
      pendingUser = null
      continue
    }

    if (userText) {
      turns.push({
        role: 'user',
        content: userText.slice(0, MAX_CONTENT_LEN),
      })
    }
    if (botText) {
      turns.push({
        role: 'assistant',
        content: botText.slice(0, MAX_CONTENT_LEN),
      })
    }
    pendingUser = null
  }

  return turns.slice(-maxRounds * 2)
}
