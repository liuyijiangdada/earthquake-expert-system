<script setup>
defineProps({
  items: { type: Array, default: () => [] },
})

const emit = defineEmits(['select', 'remove', 'clear'])
</script>

<template>
  <section class="history-panel" aria-label="搜索历史">
    <div class="history-top">
      <h2 class="title"><i class="fas fa-history" aria-hidden="true"></i>搜索历史</h2>
      <button type="button" class="btn btn-ghost btn-sm ms-auto" :disabled="!items.length" @click="emit('clear')">
        <i class="fas fa-trash-alt me-1"></i>清空
      </button>
    </div>
    <p v-if="!items.length" class="text-muted small mb-2">暂无历史记录。发送一次问题后会自动出现在这里。</p>
    <div v-else class="history-chips" role="group" aria-label="历史条目">
      <div v-for="(it, i) in items" :key="it.text + '-' + i" class="history-chip">
        <button type="button" class="chip chip-history" :title="'点击回填，双击发送'" @click="emit('select', it.text)">
          {{ it.text }}
        </button>
        <button
          type="button"
          class="btn btn-x"
          title="删除这条"
          aria-label="删除这条"
          @click="emit('remove', it.text)"
        >
          <i class="fas fa-times" aria-hidden="true"></i>
        </button>
      </div>
    </div>
  </section>
</template>

