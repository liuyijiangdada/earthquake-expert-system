<script setup>
import { ref, watch, nextTick, onMounted } from 'vue'
import { compressImageFile } from '@/utils/imageCompress.js'

const props = defineProps({
  modelValue: { type: String, default: '' },
  disabled: { type: Boolean, default: false },
})

const emit = defineEmits(['update:modelValue', 'send', 'clear', 'refreshData', 'sendImage'])

const ta = ref(null)
const fileInput = ref(null)
const previewSrc = ref('')
const previewName = ref('')

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

function triggerFileInput() {
  fileInput.value?.click()
}

async function onFileChange(e) {
  const file = e.target.files?.[0]
  if (!file) return
  if (!file.type.startsWith('image/')) {
    alert('请选择图片文件（支持 JPEG、PNG、GIF、WebP）')
    clearPreview()
    return
  }
  if (file.size > 10 * 1024 * 1024) {
    alert(`图片文件过大（${(file.size / 1024 / 1024).toFixed(1)}MB），最大支持10MB`)
    clearPreview()
    return
  }
  try {
    const compressed = await compressImageFile(file)
    previewName.value = compressed.name
    previewSrc.value = compressed.dataUrl
    if (file.size > compressed.size * 1.1) {
      console.info(
        `图片已压缩：${(file.size / 1024).toFixed(0)}KB → ${(compressed.size / 1024).toFixed(0)}KB`,
      )
    }
  } catch {
    alert('图片处理失败，请换一张较小的图片重试')
    clearPreview()
  }
}

function clearPreview() {
  previewSrc.value = ''
  previewName.value = ''
  if (fileInput.value) fileInput.value.value = ''
}

function sendWithImage() {
  if (!previewSrc.value) return
  emit('sendImage', { dataUrl: previewSrc.value, name: previewName.value, text: props.modelValue })
  clearPreview()
}

onMounted(() => nextTick(autoResize))
</script>

<template>
  <div class="composer">
    <input
      ref="fileInput"
      type="file"
      accept="image/*"
      class="d-none"
      @change="onFileChange"
    />
    <div v-if="previewSrc" class="image-preview-bar">
      <img :src="previewSrc" :alt="previewName" class="preview-thumb" />
      <span class="preview-name">{{ previewName }}</span>
      <button type="button" class="btn btn-sm btn-ghost" @click="clearPreview">
        <i class="fas fa-times"></i>
      </button>
      <button
        type="button"
        class="btn btn-sm btn-send ms-2"
        :disabled="disabled"
        @click="sendWithImage"
      >
        <i class="fas fa-paper-plane me-1"></i>发送图片
      </button>
    </div>
    <label for="userInput" class="visually-hidden">输入问题</label>
    <textarea
      id="userInput"
      ref="ta"
      class="form-control mb-2"
      rows="2"
      placeholder="请输入地震应急相关问题，如震前准备、震中避险、震后自救或灾情查询等"
      :value="modelValue"
      :disabled="disabled"
      @input="onInput"
      @keydown="onKeydown"
    />
    <div class="d-flex flex-wrap gap-2 align-items-center">
      <button type="button" class="btn btn-send" :disabled="disabled" @click="emit('send')">
        <i class="fas fa-paper-plane me-1"></i>发送
      </button>
      <button
        type="button"
        class="btn btn-ghost btn-sm"
        title="上传图片进行多模态问答"
        @click="triggerFileInput"
      >
        <i class="fas fa-image me-1"></i>上传图片
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
