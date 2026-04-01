<script setup>
import { ref, watch, onMounted, onBeforeUnmount, nextTick } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  earthquakes: { type: Array, default: () => [] },
})

const elRef = ref(null)
let chart

function buildOption(earthquakes) {
  const magLabels = ['<3 微震', '3–4 小震', '4–5 中震', '5–6 强震', '≥6 大震']
  const magBins = [0, 0, 0, 0, 0]
  earthquakes.forEach((eq) => {
    const m = parseFloat(eq.magnitude)
    if (Number.isNaN(m)) return
    if (m < 3) magBins[0]++
    else if (m < 4) magBins[1]++
    else if (m < 5) magBins[2]++
    else if (m < 6) magBins[3]++
    else magBins[4]++
  })
  const pieData = magLabels.map((name, i) => ({ name, value: magBins[i] }))
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(15, 23, 42, 0.95)',
      borderColor: 'rgba(255,255,255,0.1)',
      textStyle: { color: '#e8ecf4' },
      formatter(p) {
        return p.name + '<br/>条数：' + p.value + (p.percent != null ? '（' + p.percent + '%）' : '')
      },
    },
    legend: {
      orient: 'horizontal',
      bottom: 0,
      left: 'center',
      textStyle: { color: '#8b95a8', fontSize: 10 },
    },
    series: [
      {
        name: '震级分级',
        type: 'pie',
        radius: ['42%', '68%'],
        center: ['50%', '44%'],
        avoidLabelOverlap: true,
        itemStyle: {
          borderRadius: 6,
          borderColor: '#121a2e',
          borderWidth: 2,
        },
        label: { color: '#c4cad6', fontSize: 11, formatter: '{b}\n{c}条' },
        data: pieData,
        color: ['#34d399', '#3b82f6', '#fbbf24', '#f97316', '#ef4444'],
        emphasis: {
          itemStyle: { shadowBlur: 14, shadowColor: 'rgba(0,0,0,0.5)' },
        },
      },
    ],
  }
}

function ensureChart() {
  if (!elRef.value) return
  if (!chart) chart = echarts.init(elRef.value)
  chart.setOption(buildOption(props.earthquakes))
}

onMounted(() => {
  nextTick(ensureChart)
  window.addEventListener('resize', onResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', onResize)
  chart?.dispose()
  chart = null
})

function onResize() {
  chart?.resize()
}

watch(
  () => props.earthquakes,
  () => {
    nextTick(ensureChart)
  },
  { deep: true },
)

defineExpose({ resize: onResize })
</script>

<template>
  <div ref="elRef" class="w-100 h-100" style="min-height: 280px" />
</template>
