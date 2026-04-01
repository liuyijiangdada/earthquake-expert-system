<script setup>
import { ref, computed, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
import AppHeader from '@/components/AppHeader.vue'
import QuickPanel from '@/components/QuickPanel.vue'
import ChatMessages from '@/components/ChatMessages.vue'
import Composer from '@/components/Composer.vue'
import StatsBar from '@/components/StatsBar.vue'
import InsightChart from '@/components/InsightChart.vue'
import { WELCOME } from '@/constants.js'
import { nowTimeStr } from '@/utils/format.js'
import { queryLlm, queryKgAll, updateEarthquakeData } from '@/api.js'

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

let pollTimer
onMounted(() => {
  pushMessage({ role: 'bot', text: WELCOME, feedback: false })
  loadEarthquakeData()
  pollTimer = setInterval(loadEarthquakeData, 5 * 60 * 1000)
})

onBeforeUnmount(() => {
  if (pollTimer) clearInterval(pollTimer)
})

async function sendMessage() {
  const text = userInput.value.trim()
  if (!text || sending.value) return

  pushMessage({ role: 'user', text })
  userInput.value = ''
  showTyping.value = true
  sending.value = true
  loadingBadge.value = true

  try {
    const { ok, data } = await queryLlm(text)
    showTyping.value = false
    if (!ok || data.error) {
      pushMessage({
        role: 'bot',
        text: data.error || data.message || '服务暂时不可用，请稍后重试。',
        feedback: false,
      })
      return
    }
    if (typeof data.response !== 'string' || !data.response.trim()) {
      pushMessage({ role: 'bot', text: '未收到有效回答，请换种问法或稍后重试。' })
      return
    }
    pushMessage({ role: 'bot', text: data.response })
  } catch {
    showTyping.value = false
    pushMessage({ role: 'bot', text: '网络异常，请检查连接后重试。', feedback: false })
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

function onFeedback({ id, type }) {
  const m = messages.value.find((x) => x.id === id)
  if (!m || m.role !== 'bot') return
  m.feedbackDone = true
  m.feedbackNote = type === 'satisfied' ? '感谢反馈。' : '感谢反馈，我们会持续优化。'
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

  <AppHeader :badge-html="badgeHtml" :badge-warn="loadingBadge" />

  <main class="container py-4">
    <div class="row g-4">
      <div class="col-lg-7">
        <div class="card-dark h-100">
          <div class="card-h text-white">
            <i class="fas fa-comments text-info"></i>
            对话区
          </div>
          <div class="card-b" ref="chatRoot">
            <QuickPanel v-model="userInput" @submit-quick="onSubmitQuick" />
            <ChatMessages :messages="messages" :show-typing="showTyping" @feedback="onFeedback" />
            <Composer
              v-model="userInput"
              :disabled="sending"
              @send="sendMessage"
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
