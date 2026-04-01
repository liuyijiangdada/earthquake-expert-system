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
              </template>
              <template v-else>
                <p class="mb-0" v-html="formatBotHtml(m.text)"></p>
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
