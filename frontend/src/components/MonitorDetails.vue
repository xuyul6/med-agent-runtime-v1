<script setup>
import { computed } from 'vue'
import { alertReason, circuitLabel, countLabel, latencyLabel, monitorAlerts, monitorRows, percentLabel } from '../utils/monitor.js'

const props = defineProps({ snapshot: Object, fetchedAt: String, roleNames: Object })
const agents = computed(() => monitorRows(props.snapshot, 'agent_stats'))
const tools = computed(() => monitorRows(props.snapshot, 'tool_stats'))
const alerts = computed(() => monitorAlerts(props.snapshot))
</script>

<template>
  <p class="monitor-note">{{ fetchedAt ? `快照获取于 ${fetchedAt}` : '监控快照尚不可用' }} · 单进程，重启后清空</p>
  <p class="monitor-note">页面按需刷新；成功率和延迟按近期执行样本计算。</p>
  <details class="monitor-details">
    <summary>Agent 指标（{{ agents ? agents.length : '未计量' }}）</summary>
    <p v-if="!agents" class="monitor-note">Agent 指标不可用。</p>
    <p v-else-if="!agents.length" class="monitor-note">暂无已登记实例。</p>
    <article v-for="agent in agents || []" :key="agent.name" class="monitor-row">
      <h3>{{ roleNames?.[agent.role] || agent.role || '未知角色' }} <small>{{ agent.name }}</small></h3>
      <dl><dt>累计 / 窗口 / 执行</dt><dd>{{ countLabel(agent.total) }} / {{ countLabel(agent.samples) }} / {{ countLabel(agent.execution_samples) }}</dd>
        <dt>成功率 / 平均延迟</dt><dd>{{ percentLabel(agent.success_rate) }} / {{ latencyLabel(agent.avg_ms) }}</dd>
        <dt>降权 / 普通准入</dt><dd>{{ percentLabel(agent.monitor_penalty) }} / {{ agent.excluded === true ? '已排除' : agent.excluded === false ? '可准入' : '未知' }}</dd></dl>
    </article>
  </details>
  <details class="monitor-details">
    <summary>工具指标（{{ tools ? tools.length : '未计量' }}）</summary>
    <p v-if="!tools" class="monitor-note">工具指标不可用。</p>
    <p v-else-if="!tools.length" class="monitor-note">暂无已登记工具。</p>
    <article v-for="tool in tools || []" :key="tool.name" class="monitor-row">
      <h3>{{ tool.name }}</h3>
      <dl><dt>累计 / 窗口 / 执行</dt><dd>{{ countLabel(tool.total) }} / {{ countLabel(tool.samples) }} / {{ countLabel(tool.execution_samples) }}</dd>
        <dt>成功率 / 平均延迟</dt><dd>{{ percentLabel(tool.success_rate) }} / {{ latencyLabel(tool.avg_ms) }}</dd>
        <dt>非空有效结果率</dt><dd>{{ percentLabel(tool.quality_success_rate) }}</dd>
        <dt>熔断 / 连续失败</dt><dd>{{ circuitLabel(tool.circuit_state) }} / {{ countLabel(tool.consecutive_fails) }}</dd></dl>
    </article>
    <p class="monitor-note">空检索可计为执行成功，但不是可用证据；结果率不代表医学正确率。</p>
  </details>
  <details class="monitor-details">
    <summary>活跃告警（{{ alerts ? alerts.length : '未计量' }}）</summary>
    <p v-if="!alerts" class="monitor-note">告警数据不可用。</p>
    <p v-else-if="!alerts.length" class="monitor-note">当前无活跃告警，不代表已完成健康验证。</p>
    <article v-for="(alert, index) in alerts || []" :key="alert.id || index" class="monitor-row">
      <h3>{{ alert.kind === 'agent' ? 'Agent' : alert.kind === 'tool' ? '工具' : '未知类型' }} · {{ alert.name }}</h3>
      <p class="monitor-note">{{ Array.isArray(alert.reasons) && alert.reasons.length ? alert.reasons.map(alertReason).join('、') : '原因未提供' }}</p>
    </article>
    <p class="monitor-note">告警不等于实际回退；请查看最近一次执行详情。</p>
  </details>
</template>
