<script setup>
defineProps({
  badgeHtml: { type: String, default: '' },
  badgeWarn: { type: Boolean, default: false },
})
</script>

<template>
  <header class="app-header">
    <div class="container">
      <div class="d-flex flex-wrap align-items-center justify-content-between gap-2">
        <div>
          <h1><i class="fas fa-wave-square text-info me-2"></i>地震智能问答工作台</h1>
          <p class="subtitle">知识图谱检索 + 大模型推理 · 面向公众科普与应急认知</p>
        </div>
        <div class="d-flex align-items-center gap-2">
          <span
            class="badge-pill"
            :style="
              badgeWarn
                ? { borderColor: 'rgba(251, 191, 36, 0.5)', color: 'var(--warn)' }
                : { borderColor: 'rgba(52, 211, 153, 0.35)', color: 'var(--safe)' }
            "
            v-html="badgeHtml"
          />
        </div>
      </div>
    </div>
  </header>
</template>
