<script setup>
defineProps({
  totalEarthquakes: { type: String, default: '—' },
  maxMagnitude: { type: String, default: '—' },
  regionCount: { type: String, default: '—' },
  depthShallow: { type: String, default: '—' },
  depthMid: { type: String, default: '—' },
  depthDeep: { type: String, default: '—' },
})
</script>

<template>
  <div>
    <div class="stat-grid mb-3">
      <div class="stat-tile">
        <div class="val">{{ totalEarthquakes }}</div>
        <div class="lbl">知识库地震条数</div>
      </div>
      <div class="stat-tile">
        <div class="val">{{ maxMagnitude }}</div>
        <div class="lbl">最大震级</div>
      </div>
      <div class="stat-tile">
        <div class="val">{{ regionCount }}</div>
        <div class="lbl">地点种类</div>
      </div>
    </div>

    <div class="card-dark">
      <div class="card-h text-white">
        <i class="fas fa-bolt text-warning"></i>
        震级分级与震源深度
      </div>
      <div class="card-b">
        <p class="insight-hint">
          按震级区间统计当前知识库条数；深度按浅源(&lt;70km)、中源(70–300km)、深源(&gt;300km)划分，便于观察能量释放深度特征。
        </p>
        <div class="chart-box tall"><slot /></div>
        <div class="depth-strip text-center">
          浅源 <strong>{{ depthShallow }}</strong> 条 · 中源 <strong>{{ depthMid }}</strong> 条 · 深源
          <strong>{{ depthDeep }}</strong> 条
        </div>
      </div>
    </div>
  </div>
</template>
