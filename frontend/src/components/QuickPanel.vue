<script setup>
import { QUICK_GROUPS } from '@/constants.js'

const props = defineProps({
  modelValue: { type: String, default: '' },
})

const emit = defineEmits(['update:modelValue', 'submitQuick'])

function isActive(q) {
  const v = props.modelValue.trim()
  return v !== '' && v === q
}

function onPanelCapture(e) {
  const chip = e.target.closest?.('.chip')
  if (!chip || !e.currentTarget.contains(chip)) return
  if (e.metaKey || e.ctrlKey) {
    e.preventDefault()
    e.stopPropagation()
    const q = chip.dataset.q
    if (q) emit('submitQuick', q)
  }
}

function onChipClick(e, q) {
  if (e.metaKey || e.ctrlKey) return
  emit('update:modelValue', q)
}

function onChipDblclick(e, q) {
  e.preventDefault()
  emit('submitQuick', q)
}
</script>

<template>
  <section class="quick-panel" aria-label="快捷提问" @click.capture="onPanelCapture">
    <div class="quick-panel-top">
      <h2 class="title"><i class="fas fa-bolt" aria-hidden="true"></i>快捷提问</h2>
      <p class="quick-hint">
        单击填入输入框；双击立即发送。<span class="d-none d-sm-inline"
          >按住 <kbd>⌘</kbd>/<kbd>Ctrl</kbd> 单击也可直接发送。</span
        >
      </p>
    </div>
    <div class="quick-groups">
      <div v-for="(g, gi) in QUICK_GROUPS" :key="gi">
        <p class="quick-group-title">{{ g.title }}</p>
        <div class="quick-chips" role="group" :aria-label="g.ariaLabel">
          <button
            v-for="(item, ii) in g.items"
            :key="ii"
            type="button"
            class="chip"
            :class="{ 'is-active': isActive(item.q) }"
            :title="'单击填入，双击立即发送'"
            :data-q="item.q"
            :aria-pressed="isActive(item.q) ? 'true' : 'false'"
            @click="onChipClick($event, item.q)"
            @dblclick="onChipDblclick($event, item.q)"
          >
            <i :class="'fas ' + item.icon" aria-hidden="true"></i>{{ item.label }}
          </button>
        </div>
      </div>
    </div>
  </section>
</template>
