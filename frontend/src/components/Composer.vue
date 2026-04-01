<script setup>
import { ref, watch, nextTick, onMounted } from 'vue'

const props = defineProps({
  modelValue: { type: String, default: '' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'send', 'clear', 'refreshData'])

const ta = ref(null)

function autoResize() {
  const el = ta.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 140) + 'px'
}

watch(
  () => props.modelValue,
  () => nextTick(autoResize),
)

function onInput(e) {
  emit('update:modelValue', e.target.value)
  autoResize()
}

function onKeydown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    emit('send')
  }
}

onMounted(() => nextTick(autoResize))
</script>

<template>
  <div class="composer">
    <label for="userInput" class="visually-hidden">输入问题</label>
    <textarea
      id="userInput"
      ref="ta"
      class="form-control mb-2"
      rows="2"
      placeholder="例如：云南近期地震活动有什么特点？"
      :value="modelValue"
      :disabled="disabled"
      @input="onInput"
      @keydown="onKeydown"
    />
    <div class="d-flex flex-wrap gap-2 align-items-center">
      <button type="button" class="btn btn-send" :disabled="disabled" @click="emit('send')">
        <i class="fas fa-paper-plane me-1"></i>发送
      </button>
      <button type="button" class="btn btn-ghost btn-sm" title="清空对话区" @click="emit('clear')">
        <i class="fas fa-eraser me-1"></i>清空对话
      </button>
      <button
        type="button"
        class="btn btn-ghost btn-sm"
        title="从数据源刷新知识图谱"
        @click="emit('refreshData')"
      >
        <i class="fas fa-sync-alt me-1"></i>刷新地震数据
      </button>
      <span class="hint-bar ms-auto d-none d-md-inline"
        ><kbd>Enter</kbd> 发送 · <kbd>Shift</kbd>+<kbd>Enter</kbd> 换行</span
      >
    </div>
  </div>
</template>
