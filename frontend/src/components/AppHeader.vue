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
          <h1>
            <i class="fas fa-house-crack text-info me-2" aria-hidden="true"></i>
            地震应急智能问答
          </h1>
          <p class="subtitle">
            动态-静态知识协同 · 知识图谱 + RAG + 实时震情 · 震前 / 震中 / 震后全周期
          </p>
        </div>
        <div class="d-flex align-items-center gap-2">
          <span
            class="badge-pill"
            :style="
              badgeWarn
                ? { borderColor: 'rgba(251, 191, 36, 0.5)', color: 'var(--warn)', background: 'rgba(251,191,36,0.12)' }
                : {}
            "
            v-html="badgeHtml"
          />
        </div>
      </div>
    </div>
  </header>
</template>
