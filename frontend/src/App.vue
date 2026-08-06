<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import Login from '@/components/Login.vue'
import QuickPanel from '@/components/QuickPanel.vue'
import ChatMessages from '@/components/ChatMessages.vue'
import Composer from '@/components/Composer.vue'
import StatsBar from '@/components/StatsBar.vue'
import InsightChart from '@/components/InsightChart.vue'
import { WELCOME } from '@/constants.js'
import { nowTimeStr } from '@/utils/format.js'
import { queryLlm, queryKgAll, updateEarthquakeData, submitFeedback, logout as apiLogout } from '@/api.js'
import { compressImageDataUrl } from '@/utils/imageCompress.js'
import { buildChatHistory } from '@/utils/chatHistory.js'

const AUTH_KEY = 'eq_auth_user'

function readAuth() {
  try {
    const raw = localStorage.getItem(AUTH_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    if (parsed?.username) return parsed
  } catch {
    /* ignore */
  }
  return null
}

const authUser = ref(readAuth())
const isAuthenticated = computed(() => !!authUser.value?.username)

const userInput = ref('')
const messages = ref([])
const sending = ref(false)
const showTyping = ref(false)
const loadingBadge = ref(false)

const earthquakes = ref([])
const totalEarthquakes = ref('—')
const maxMagnitude = ref('—')
const regionCount = ref('—')
const depthShallow = ref('—')
const depthMid = ref('—')
const depthDeep = ref('—')

const badgeHtml = computed(() =>
  loadingBadge.value
    ? '<i class="fas fa-circle-notch fa-spin me-1"></i>推理中…'
    : '<i class="fas fa-link me-1"></i>服务已连接',
)

function pushMessage(partial) {
  messages.value.push({
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
    time: nowTimeStr(),
    ...partial,
  })
}

function mapMetaToMessage(meta) {
  if (!meta) return {}
  return {
    phase: meta.phase || '通用',
    urgency: meta.urgency || 0,
    staticConfidence: meta.static_confidence,
    dynamicAvailability: meta.dynamic_availability,
    reliabilityHint: meta.reliability_hint,
    mediaResources: meta.media_resources?.length ? meta.media_resources : undefined,
  }
}

function updateStats(eq) {
  if (!eq.length) {
    totalEarthquakes.value = '0'
    maxMagnitude.value = '—'
    regionCount.value = '0'
    depthShallow.value = '—'
    depthMid.value = '—'
    depthDeep.value = '—'
    return
  }
  totalEarthquakes.value = String(eq.length)
  const magnitudes = eq.map((e) => e.magnitude)
  maxMagnitude.value = Math.max(...magnitudes).toFixed(1)
  const regions = {}
  eq.forEach((e) => {
    regions[e.location] = true
  })
  regionCount.value = String(Object.keys(regions).length)

  let shallow = 0
  let mid = 0
  let deep = 0
  eq.forEach((e) => {
    const d = parseFloat(e.depth)
    if (Number.isNaN(d)) return
    if (d < 70) shallow++
    else if (d <= 300) mid++
    else deep++
  })
  depthShallow.value = String(shallow)
  depthMid.value = String(mid)
  depthDeep.value = String(deep)
}

async function loadEarthquakeData() {
  try {
    const data = await queryKgAll()
    if (!data.results) return
    earthquakes.value = data.results
    updateStats(data.results)
  } catch {
    totalEarthquakes.value = '—'
  }
}

function onLoginSuccess({ username }) {
  const payload = { username, loggedInAt: new Date().toISOString() }
  localStorage.setItem(AUTH_KEY, JSON.stringify(payload))
  authUser.value = payload
  if (!messages.value.length) {
    pushMessage({ role: 'bot', text: WELCOME, feedback: false })
  }
  loadEarthquakeData()
}

async function onLogout() {
  try {
    await apiLogout()
  } catch {
    /* ignore */
  }
  localStorage.removeItem(AUTH_KEY)
  authUser.value = null
}

let pollTimer
onMounted(() => {
  if (!isAuthenticated.value) return
  pushMessage({ role: 'bot', text: WELCOME, feedback: false })
  loadEarthquakeData()
  pollTimer = setInterval(loadEarthquakeData, 5 * 60 * 1000)
})

watch(isAuthenticated, (ok) => {
  if (ok) {
    if (!pollTimer) {
      pollTimer = setInterval(loadEarthquakeData, 5 * 60 * 1000)
    }
  } else if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
})

async function sendMessage() {
  const text = userInput.value.trim()
  if (!text || sending.value) return

  if (text.length > 2000) {
    pushMessage({ role: 'bot', text: `输入内容过长（${text.length}字），请控制在2000字以内。`, feedback: false })
    return
  }

  const history = buildChatHistory(messages.value, 3)

  pushMessage({ role: 'user', text })
  userInput.value = ''
  showTyping.value = true
  sending.value = true
  loadingBadge.value = true

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 60000)

    const { ok, data } = await queryLlm(text, history)
    clearTimeout(timeoutId)
    showTyping.value = false
    if (!ok) {
      if (data.error) {
        pushMessage({ role: 'bot', text: `请求失败：${data.error}`, feedback: false })
      } else {
        pushMessage({ role: 'bot', text: '服务暂时不可用，请稍后重试。', feedback: false })
      }
      return
    }
    if (data.error) {
      pushMessage({ role: 'bot', text: `系统提示：${data.error}`, feedback: false })
      return
    }
    if (typeof data.response !== 'string' || !data.response.trim()) {
      pushMessage({ role: 'bot', text: '未收到有效回答，请换种问法或稍后重试。' })
      return
    }
    const msgExtra = mapMetaToMessage(data.meta ?? data.debug)
    pushMessage({ role: 'bot', text: data.response, ...msgExtra })
  } catch (err) {
    showTyping.value = false
    if (err.name === 'AbortError') {
      pushMessage({ role: 'bot', text: '请求超时（60秒），模型推理可能较慢，请稍后重试。', feedback: false })
    } else if (err instanceof TypeError && err.message.includes('fetch')) {
      pushMessage({ role: 'bot', text: '无法连接服务器，请确认后端服务已启动。', feedback: false })
    } else {
      pushMessage({ role: 'bot', text: '网络异常，请检查连接后重试。', feedback: false })
    }
  } finally {
    sending.value = false
    loadingBadge.value = false
  }
}

function clearChat() {
  messages.value = []
  pushMessage({ role: 'bot', text: WELCOME, feedback: false })
}

async function refreshData() {
  try {
    const data = await updateEarthquakeData()
    if (data.status === 'success') {
      pushMessage({ role: 'bot', text: '数据已更新：' + (data.message || ''), feedback: false })
      loadEarthquakeData()
    } else {
      pushMessage({ role: 'bot', text: '更新失败：' + (data.message || '未知错误'), feedback: false })
    }
  } catch {
    pushMessage({ role: 'bot', text: '更新请求失败，请稍后重试。', feedback: false })
  }
}

async function sendImageMessage({ dataUrl, name, text }) {
  if (sending.value) return

  if (!text || !text.trim()) {
    text = '请分析这张与地震相关的图片'
  }

  const history = buildChatHistory(messages.value, 3)

  pushMessage({ role: 'user', text, imageUrl: dataUrl })
  showTyping.value = true
  sending.value = true
  loadingBadge.value = true

  try {
    const compressed = await compressImageDataUrl(dataUrl)
    const blob = compressed.blob

    if (blob.size > 10 * 1024 * 1024) {
      showTyping.value = false
      pushMessage({ role: 'bot', text: `图片文件过大（${(blob.size / 1024 / 1024).toFixed(1)}MB），最大支持10MB。`, feedback: false })
      return
    }

    const formData = new FormData()
    formData.append('image', blob, compressed.name || name || 'image.jpg')
    formData.append('input', text)
    if (history.length) {
      formData.append('history', JSON.stringify(history))
    }

    const r = await fetch('/api/multimodal-query', {
      method: 'POST',
      body: formData,
    })
    const data = await r.json().catch(() => ({}))
    showTyping.value = false

    if (!r.ok) {
      pushMessage({ role: 'bot', text: data.error || `图片分析请求失败（HTTP ${r.status}），请稍后重试。`, feedback: false })
      return
    }

    if (data.error) {
      pushMessage({ role: 'bot', text: `图片分析失败：${data.error}`, feedback: false })
      return
    }

    const msgExtra = mapMetaToMessage(data.meta ?? data.debug)
    pushMessage({ role: 'bot', text: data.response, ...msgExtra })
  } catch (err) {
    showTyping.value = false
    if (err.name === 'AbortError') {
      pushMessage({ role: 'bot', text: '图片分析请求超时，请稍后重试。', feedback: false })
    } else if (err instanceof TypeError && err.message.includes('fetch')) {
      pushMessage({ role: 'bot', text: '无法连接服务器，请确认后端服务已启动。', feedback: false })
    } else {
      pushMessage({ role: 'bot', text: '图片上传失败，请检查网络后重试。', feedback: false })
    }
  } finally {
    sending.value = false
    loadingBadge.value = false
  }
}

async function onFeedback({ id, type }) {
  const idx = messages.value.findIndex((x) => x.id === id)
  const m = idx >= 0 ? messages.value[idx] : null
  if (!m || m.role !== 'bot' || m.feedbackDone) return

  let question = ''
  for (let i = idx - 1; i >= 0; i -= 1) {
    if (messages.value[i].role === 'user') {
      question = messages.value[i].text || ''
      break
    }
  }

  m.feedbackDone = true
  m.feedbackNote = '正在提交反馈…'
  try {
    const { ok, data } = await submitFeedback({
      type,
      message_id: id,
      question,
      answer: m.text || '',
      phase: m.phase || '',
      static_confidence: m.staticConfidence,
      reliability_hint: m.reliabilityHint || '',
    })
    if (!ok) {
      m.feedbackDone = false
      m.feedbackNote = data.error || '反馈提交失败，请稍后重试。'
      return
    }
    m.feedbackNote =
      type === 'satisfied' ? '已记录：回答有用，感谢反馈。' : '已记录：需改进，感谢反馈。'
  } catch {
    m.feedbackDone = false
    m.feedbackNote = '反馈提交失败，请检查网络后重试。'
  }
}

function onSubmitQuick(q) {
  userInput.value = q
  nextTick(() => sendMessage())
}

const chatRoot = ref(null)
watch(
  () => [messages.value.length, showTyping.value],
  () => {
    nextTick(() => {
      const el = chatRoot.value?.querySelector?.('.chat-scroll')
      if (el) el.scrollTop = el.scrollHeight
    })
  },
)
</script>

<template>
  <div class="page-bg" aria-hidden="true"></div>

  <Login v-if="!isAuthenticated" @success="onLoginSuccess" />

  <template v-else>
    <AppHeader
      :badge-html="badgeHtml"
      :badge-warn="loadingBadge"
      :username="authUser.username"
      @logout="onLogout"
    />

    <main class="container py-4">
      <div class="row g-4">
        <div class="col-lg-7">
          <div class="card-dark h-100">
            <div class="card-h text-white">
              <i class="fas fa-comments text-info"></i>
              应急问答对话
            </div>
            <div class="card-b" ref="chatRoot">
              <QuickPanel v-model="userInput" @submit-quick="onSubmitQuick" />
              <ChatMessages :messages="messages" :show-typing="showTyping" @feedback="onFeedback" />
              <Composer
                v-model="userInput"
                :disabled="sending"
                @send="sendMessage"
                @send-image="sendImageMessage"
                @clear="clearChat"
                @refresh-data="refreshData"
              />
            </div>
          </div>
        </div>

        <div class="col-lg-5">
          <StatsBar
            :total-earthquakes="totalEarthquakes"
            :max-magnitude="maxMagnitude"
            :region-count="regionCount"
            :depth-shallow="depthShallow"
            :depth-mid="depthMid"
            :depth-deep="depthDeep"
          >
            <InsightChart :earthquakes="earthquakes" />
          </StatsBar>
        </div>
      </div>
    </main>
  </template>
</template>
