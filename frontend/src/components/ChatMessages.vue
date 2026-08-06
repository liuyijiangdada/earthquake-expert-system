<script setup>
import { ref, watch } from 'vue'
import { formatBotHtml } from '@/utils/format.js'
import { PHASE_META } from '@/constants.js'

const brokenMediaIds = ref(new Set())
const showMetaMetrics = import.meta.env.DEV && import.meta.env.VITE_SHOW_META === 'true'

const props = defineProps({
  messages: { type: Array, default: () => [] },
  showTyping: { type: Boolean, default: false },
})

watch(
  () => props.messages.length,
  () => {
    brokenMediaIds.value = new Set()
  },
)

function onMediaImageError(resId) {
  brokenMediaIds.value = new Set([...brokenMediaIds.value, resId])
}

function hasVisibleMedia(resources) {
  if (!resources?.length) return false
  return resources.some((res) => res.type !== 'image' || !brokenMediaIds.value.has(res.id))
}

const emit = defineEmits(['feedback'])

function onFeedback(msgId, type) {
  emit('feedback', { id: msgId, type })
}

function phaseMeta(phase) {
  return PHASE_META[phase] || PHASE_META['通用']
}

function formatMetaMetrics(m) {
  if (!showMetaMetrics) return ''
  const parts = []
  if (m.staticConfidence != null) parts.push(`静态置信 ${m.staticConfidence}`)
  if (m.dynamicAvailability != null) parts.push(`动态可用 ${m.dynamicAvailability}`)
  if (m.reliabilityHint) parts.push(m.reliabilityHint)
  return parts.join(' · ')
}

function reliabilityNotice(m) {
  const hasImage = (m.mediaResources || []).some((r) => r.type === 'image' && r.url)
  if (m.reliabilityHint) {
    if (!hasImage && m.reliabilityHint.includes('示意图')) {
      return m.reliabilityHint
        .replace('，可参考下方示意图。', '')
        .replace('，可参考下方示意图', '')
        .replace('可参考下方示意图。', '')
        .replace('可参考下方示意图', '')
        .replace('，建议结合下方示意图阅读', '')
        .replace('建议结合下方示意图阅读。', '')
        .replace('建议结合下方示意图阅读', '')
        .replace(/[，。]+$/, '。')
    }
    return m.reliabilityHint
  }
  if (m.staticConfidence != null && m.staticConfidence < 0.7) {
    return hasImage
      ? '已依据本地知识库生成回答，可参考下方示意图。'
      : '已依据本地知识库生成回答。'
  }
  return ''
}
</script>

<template>
  <div class="chat-wrap">
    <div class="chat-scroll" role="log" aria-live="polite" aria-relevant="additions">
      <template v-for="m in messages" :key="m.id">
        <div class="msg-block" :class="m.role">
          <div class="msg-row" :class="m.role">
            <div class="msg-avatar" :class="m.role">
              <i
                :class="
                  m.role === 'user'
                    ? 'fas fa-user'
                    : m.phase === '震中'
                      ? 'fas fa-bolt'
                      : 'fas fa-robot'
                "
              ></i>
            </div>
            <div class="msg-bubble">
              <template v-if="m.role === 'user'">
                <p class="mb-0">{{ m.text }}</p>
                <div v-if="m.imageUrl" class="user-image-preview mt-2">
                  <img
                    :src="m.imageUrl"
                    alt="用户上传的图片"
                    class="img-fluid rounded"
                    style="max-height: 200px"
                  />
                </div>
              </template>
              <template v-else>
                <div v-if="m.phase" class="phase-badge-row mb-2">
                  <span class="phase-pill" :class="phaseMeta(m.phase).class">
                    <i :class="'fas ' + phaseMeta(m.phase).icon"></i>
                    {{ m.phase }} · {{ phaseMeta(m.phase).hint }}
                  </span>
                  <span v-if="m.urgency > 0.5" class="phase-pill urgency-pill">
                    <i class="fas fa-exclamation-triangle"></i> 高紧急度
                  </span>
                </div>
                <p class="mb-0" v-html="formatBotHtml(m.text)"></p>

                <p v-if="reliabilityNotice(m)" class="reliability-notice mb-0">
                  <i class="fas fa-info-circle me-1"></i>{{ reliabilityNotice(m) }}
                </p>

                <p v-if="formatMetaMetrics(m)" class="meta-strip mb-0">{{ formatMetaMetrics(m) }}</p>

                <div v-if="hasVisibleMedia(m.mediaResources)" class="media-section">
                  <p class="media-section-title">
                    <i class="fas fa-layer-group me-1"></i>相关资源（图谱 / 示意图 / 地图链接）
                  </p>
                  <template v-for="res in m.mediaResources" :key="res.id">
                    <div
                      v-if="res.type !== 'image' || !brokenMediaIds.has(res.id)"
                      class="media-item"
                    >
                      <template v-if="res.type === 'image'">
                        <div class="media-image-card">
                          <img
                            :src="res.url"
                            :alt="res.caption || '示意图'"
                            class="img-fluid rounded"
                            loading="lazy"
                            @error="onMediaImageError(res.id)"
                          />
                        </div>
                      </template>
                    <template v-else-if="res.type === 'link'">
                      <a
                        :href="res.url"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="media-link"
                      >
                        <i class="fas fa-external-link-alt"></i>{{ res.caption }}
                      </a>
                      <div v-if="res.source" class="media-source mt-1">{{ res.source }}</div>
                    </template>
                    <template v-else-if="res.type === 'video'">
                      <a
                        :href="res.url"
                        target="_blank"
                        rel="noopener noreferrer"
                        class="media-link"
                      >
                        <i class="fas fa-video"></i>{{ res.caption }}
                      </a>
                    </template>
                  </div>
                  </template>
                </div>

                <div v-if="m.feedback !== false" class="feedback-row">
                  <button
                    type="button"
                    class="btn btn-sm btn-outline-success"
                    :disabled="m.feedbackDone"
                    @click="onFeedback(m.id, 'satisfied')"
                  >
                    <i class="fas fa-thumbs-up"></i> 有用
                  </button>
                  <button
                    type="button"
                    class="btn btn-sm btn-outline-danger"
                    :disabled="m.feedbackDone"
                    @click="onFeedback(m.id, 'unsatisfied')"
                  >
                    <i class="fas fa-thumbs-down"></i> 需改进
                  </button>
                  <span v-if="m.feedbackNote" class="text-muted small">{{ m.feedbackNote }}</span>
                </div>
 
              </template>
            </div>
          </div>
          <div class="msg-meta">{{ m.time }}</div>
        </div>
      </template>

      <div v-if="showTyping" class="msg-block bot">
        <div class="msg-row bot">
          <div class="msg-avatar bot"><i class="fas fa-spinner fa-spin"></i></div>
          <div class="msg-bubble">
            <div class="typing-dots"><span></span><span></span><span></span></div>
            <p class="text-muted small mb-0 mt-1">正在协同检索知识源并推理…</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
