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

function phaseDotClass(phase) {
  if (phase === '震前') return 'pre'
  if (phase === '震中') return 'during'
  if (phase === '震后') return 'post'
  return ''
}

function chipPhaseClass(phase) {
  if (phase === '震前') return 'chip-pre'
  if (phase === '震中') return 'chip-during'
  if (phase === '震后') return 'chip-post'
  return ''
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
  <section class="quick-panel" aria-label="场景化快捷提问" @click.capture="onPanelCapture">
    <div class="capability-strip" aria-hidden="true">
      <span class="cap-tag"><i class="fas fa-diagram-project"></i>KG</span>
      <span class="cap-tag"><i class="fas fa-magnifying-glass"></i>RAG</span>
      <span class="cap-tag"><i class="fas fa-satellite"></i>USGS</span>
      <span class="cap-tag"><i class="fas fa-map"></i>高德</span>
      <span class="cap-tag"><i class="fas fa-image"></i>多模态</span>
    </div>
    <div class="quick-panel-top">
      <h2 class="title"><i class="fas fa-bolt" aria-hidden="true"></i>场景快捷提问</h2>
      <p class="quick-hint">
        按应急阶段分类；单击填入，双击发送。<span class="d-none d-sm-inline"
          ><kbd>⌘</kbd>/<kbd>Ctrl</kbd>+单击亦可直接发送。</span
        >
      </p>
    </div>
    <div class="quick-groups">
      <div v-for="(g, gi) in QUICK_GROUPS" :key="gi">
        <p class="quick-group-title">
          <span v-if="g.phase" class="phase-dot" :class="phaseDotClass(g.phase)"></span>
          {{ g.title }}
        </p>
        <div class="quick-chips" role="group" :aria-label="g.ariaLabel">
          <button
            v-for="(item, ii) in g.items"
            :key="ii"
            type="button"
            class="chip"
            :class="[chipPhaseClass(g.phase), { 'is-active': isActive(item.q) }]"
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
