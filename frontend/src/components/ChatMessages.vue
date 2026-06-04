<script setup>
import { formatBotHtml } from '@/utils/format.js'

defineProps({
  messages: { type: Array, default: () => [] },
  showTyping: { type: Boolean, default: false },
})

const emit = defineEmits(['feedback'])

function onFeedback(msgId, type) {
  emit('feedback', { id: msgId, type })
}

const phaseColors = {
  '震前': 'bg-info',
  '震中': 'bg-danger',
  '震后': 'bg-success',
  '通用': 'bg-secondary',
}

function phaseClass(phase) {
  return phaseColors[phase] || 'bg-secondary'
}
</script>

<template>
  <div class="chat-wrap">
    <div class="chat-scroll" role="log" aria-live="polite" aria-relevant="additions">
      <template v-for="m in messages" :key="m.id">
        <div class="msg-block" :class="m.role">
          <div class="msg-row" :class="m.role">
            <div class="msg-avatar" :class="m.role">
              <i :class="m.role === 'user' ? 'fas fa-user' : 'fas fa-robot'"></i>
            </div>
            <div class="msg-bubble">
              <template v-if="m.role === 'user'">
                <p class="mb-0">{{ m.text }}</p>
                <div v-if="m.imageUrl" class="user-image-preview mt-2">
                  <img :src="m.imageUrl" alt="用户上传的图片" class="img-fluid rounded" style="max-height: 200px;" />
                </div>
              </template>
              <template v-else>
                <div v-if="m.phase && m.phase !== '通用'" class="phase-badge-row mb-1">
                  <span class="badge" :class="phaseClass(m.phase)">{{ m.phase }}阶段</span>
                  <span v-if="m.urgency > 0.5" class="badge bg-warning text-dark ms-1">
                    <i class="fas fa-exclamation-triangle me-1"></i>紧急
                  </span>
                </div>
                <p class="mb-0" v-html="formatBotHtml(m.text)"></p>

                <div v-if="m.mediaResources && m.mediaResources.length" class="media-section mt-2">
                  <div v-for="res in m.mediaResources" :key="res.id" class="media-item">
                    <template v-if="res.type === 'image'">
                      <div class="media-image-card">
                        <img :src="res.url" :alt="res.caption" class="img-fluid rounded" loading="lazy" />
                        <p class="media-caption mb-0">{{ res.caption }}</p>
                      </div>
                    </template>
                    <template v-else-if="res.type === 'link'">
                      <a :href="res.url" target="_blank" rel="noopener noreferrer" class="media-link">
                        <i class="fas fa-external-link-alt me-1"></i>{{ res.caption }}
                      </a>
                    </template>
                    <template v-else-if="res.type === 'video'">
                      <a :href="res.url" target="_blank" rel="noopener noreferrer" class="media-link">
                        <i class="fas fa-video me-1"></i>{{ res.caption }}
                      </a>
                    </template>
                  </div>
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
                <p v-if="m.feedback !== false" class="disclaimer mb-0">
                  以上内容由模型结合知识库生成，仅供科普与参考，不能替代官方预警与应急指引。
                </p>
              </template>
            </div>
          </div>
          <div class="msg-meta">{{ m.time }}</div>
        </div>
      </template>

      <div v-if="showTyping" class="msg-block bot">
        <div class="msg-row bot">
          <div class="msg-avatar bot"><i class="fas fa-robot"></i></div>
          <div class="msg-bubble">
            <div class="typing-dots"><span></span><span></span><span></span></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
