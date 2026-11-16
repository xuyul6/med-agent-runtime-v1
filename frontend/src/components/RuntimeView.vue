<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { api } from '../api/client.js'
import { ROLE_LABELS, errorMessage, label, milliseconds, percent } from '../lib/format.js'

const monitor = ref(null)
const agents = ref([])
const tools = ref([])
const error = ref('')
const loading = ref(false)
let timer = null

async function load () {
  loading.value = true
  error.value = ''
  try {
    const [snapshot, agentList, toolList] = await Promise.all([api.monitor(), api.agents(), api.tools()])
    monitor.value = snapshot
    agents.value = agentList.agents || []
    tools.value = toolList.tools || []
  } catch (failure) {
    error.value = errorMessage(failure)
  } finally {
    loading.value = false
  }
}

const tiles = computed(() => {
  const stats = (monitor.value && monitor.value.request_stats) || {}
  return [
    { name: '请求数', value: monitor.value ? String(monitor.value.requests) : '—' },
    { name: '平均耗时', value: milliseconds(stats.avg_ms) },
    { name: 'P95 耗时', value: milliseconds(stats.p95_ms) },
    { name: '降级回答', value: monitor.value ? String(monitor.value.degraded) : '—' },
    { name: '监控回退', value: monitor.value ? String(monitor.value.monitor_fallbacks) : '—' }
  ]
})

const instances = computed(() => Object.entries((monitor.value && monitor.value.agent_stats) || {}).map(([name, stats]) => ({ name, ...stats })))
const alerts = computed(() => (monitor.value && monitor.value.active_alerts) || [])
const toolNames = computed(() => tools.value.map(item => item.name))
const sideEffects = computed(() => Object.fromEntries(tools.value.map(item => [item.name, item.side_effect])))

function alertText (alert) {
  if (typeof alert === 'string') return alert
  return alert.message || [alert.kind, alert.target].filter(Boolean).join(' · ') || JSON.stringify(alert)
}

function healthTone (item) {
  if (item.excluded) return 'bad'
  if (item.health < 0.6) return 'bad'
  if (item.health < 0.9 || item.monitor_penalty > 0) return 'caution'
  return 'ok'
}

onMounted(() => {
  load()
  timer = setInterval(load, 15000)
})
onBeforeUnmount(() => clearInterval(timer))
</script>

<template>
  <div class="page">
    <header class="intro">
      <div>
        <p class="eyebrow">开发者视图 · 单进程内存数据</p>
        <h1 class="serif">运行状态</h1>
      </div>
      <button type="button" class="btn small" :disabled="loading" @click="load">{{ loading ? '刷新中…' : '刷新' }}</button>
    </header>

    <p v-if="error" class="error">{{ error }}</p>

    <div class="tiles">
      <div v-for="tile in tiles" :key="tile.name" class="tile card">
        <span class="muted">{{ tile.name }}</span>
        <strong class="mono">{{ tile.value }}</strong>
      </div>
    </div>

    <section class="card panel">
      <h2>告警</h2>
      <p v-if="!alerts.length" class="muted calm"><span class="dot ok"></span>没有活动告警</p>
      <ul v-else class="alerts">
        <li v-for="(alert, index) in alerts" :key="index"><span class="dot bad"></span>{{ alertText(alert) }}</li>
      </ul>
    </section>

    <section class="card panel">
      <h2>角色实例</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>实例</th><th>角色</th><th>健康度</th><th>成功率</th><th>P95</th><th>样本</th><th>状态</th></tr>
          </thead>
          <tbody>
            <tr v-for="item in instances" :key="item.name">
              <td class="mono">{{ item.name }}</td>
              <td>{{ label(ROLE_LABELS, item.role) }}</td>
              <td>
                <div class="health">
                  <div class="bar"><span :style="{ width: Math.round((item.health ?? 0) * 100) + '%' }"></span></div>
                  <span class="mono">{{ percent(item.health) }}</span>
                </div>
              </td>
              <td class="mono">{{ percent(item.success_rate) }}</td>
              <td class="mono">{{ milliseconds(item.p95_ms) }}</td>
              <td class="mono">{{ item.samples }}</td>
              <td><span class="dot" :class="healthTone(item)"></span><span class="state">{{ item.excluded ? '已隔离' : item.probe_inflight ? '探测中' : '正常' }}</span></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="card panel">
      <h2>角色与工具权限</h2>
      <p class="muted sub">每个角色只能调用自己范围内的工具；写操作工具只会生成草稿。</p>
      <div class="table-wrap">
        <table class="matrix">
          <thead>
            <tr>
              <th>角色</th>
              <th v-for="name in toolNames" :key="name" class="rot"><span :title="sideEffects[name]">{{ name }}</span></th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="agent in agents" :key="agent.role">
              <td>{{ label(ROLE_LABELS, agent.role) }}</td>
              <td v-for="name in toolNames" :key="name" class="cell">
                <span v-if="agent.tool_scope.includes(name)" class="has" :class="sideEffects[name]" :title="sideEffects[name]"></span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div class="legend">
        <span><span class="has compute"></span>本地计算</span>
        <span><span class="has read_external"></span>读取资料或号源</span>
        <span><span class="has propose_write"></span>生成待确认草稿</span>
      </div>
    </section>
  </div>
</template>

<style scoped>
.page { height: 100%; overflow-y: auto; padding: 40px 24px 64px; }
.page > * { max-width: 1080px; margin-left: auto; margin-right: auto; }
.intro { display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 24px; }
.intro h1 { font-size: 30px; margin: 4px 0 0; }
.error { color: var(--critical); }

.tiles { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 20px; }
.tile { padding: 14px 16px; display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; }
.tile strong { font-size: 22px; font-weight: 500; color: var(--ink); }

.panel { padding: 20px 22px; margin-bottom: 20px; }
.panel h2 { margin: 0 0 12px; font-size: 16px; font-weight: 600; }
.sub { font-size: 13px; margin: -6px 0 12px; }
.calm { display: flex; align-items: center; gap: 8px; margin: 0; font-size: 14px; }
.alerts { margin: 0; padding: 0; list-style: none; }
.alerts li { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 14px; }

.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 500; color: var(--ink-3); padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 8px 10px; border-bottom: 1px solid var(--line); white-space: nowrap; }
tbody tr:last-child td { border-bottom: 0; }
.health { display: flex; align-items: center; gap: 8px; min-width: 120px; }
.health .bar { flex: 1; }
.state { margin-left: 6px; }
td .dot { display: inline-block; }

.matrix th.rot { height: 150px; vertical-align: bottom; padding: 0 0 8px; width: 30px; }
.matrix th.rot span { display: inline-block; writing-mode: vertical-rl; transform: rotate(180deg); font-family: var(--mono); font-size: 11.5px; }
.matrix { width: auto; }
.matrix .cell { text-align: center; padding: 8px 0; width: 30px; }
.matrix td:first-child, .matrix th:first-child { min-width: 96px; padding-right: 24px; }
.has { display: inline-block; width: 12px; height: 12px; border-radius: 3px; background: var(--accent); }
.has.read_external { background: transparent; box-shadow: inset 0 0 0 2px var(--accent); }
.has.propose_write { background: var(--caution); }
.legend { display: flex; gap: 16px; font-size: 12.5px; color: var(--ink-2); margin-top: 12px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }

@media (max-width: 860px) {
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 520px) {
  .page { padding: 24px 16px 48px; }
}
</style>
