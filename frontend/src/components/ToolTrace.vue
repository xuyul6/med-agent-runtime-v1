<script setup>
import { computed } from 'vue'
import { label, toolCountLabel, toolEventsForRequest, toolTimeLabel } from '../utils/execution.js'

const props = defineProps({ trace: Object, requestId: String, loading: Boolean, roleNames: Object })
const events = computed(() => toolEventsForRequest(props.trace, props.requestId))
const statuses = { ok: '执行成功', empty_result: '执行成功，结果为空', denied: '权限或范围拒绝', invalid_request: '参数无效', invalid_result: '结果校验失败', unknown_tool: '未知工具', budget_exhausted: '预算不足', circuit_open: '熔断拦截', timeout: '超时', cancelled: '已取消', failed: '执行失败', round_or_batch_limit: '轮次或批量上限，整批拒绝' }
</script>

<template>
  <details class="execution-details">
    <summary>本次工具事件（{{ events ? events.length : '未计量' }}）</summary>
    <p v-if="loading" class="monitor-note">正在获取本次 Trace。</p>
    <p v-else-if="!events" class="monitor-note">Trace 不可用，无法确认工具事件。</p>
    <p v-else-if="!events.length" class="monitor-note">本次 Trace 没有记录工具事件。</p>
    <template v-else>
      <article v-for="(entry, index) in events" :key="index" class="execution-row">
        <template v-if="entry.event === 'tool'">
          <strong>{{ entry.name || '未知工具' }}</strong>
          <p>角色：{{ label(roleNames || {}, entry.role) }} · 状态：{{ label(statuses, entry.status) }}</p>
          <p>进入处理器：{{ entry.handler_invoked === true ? '是' : entry.handler_invoked === false ? '否' : '未提供' }} · 耗时：{{ toolTimeLabel(entry.elapsed_ms) }}</p>
          <p>记录时点：{{ toolTimeLabel(entry.at_ms) }} · 预算已用 / 上限：{{ toolCountLabel(entry.budget_used) }} / {{ toolCountLabel(entry.budget_limit) }}</p>
          <p v-if="entry.error">异常类型：{{ entry.error }}</p>
        </template>
        <template v-else>
          <strong>工具批次</strong>
          <p>角色：{{ label(roleNames || {}, entry.role) }} · 状态：{{ label(statuses, entry.status) }}</p>
          <p>模型请求工具数：{{ toolCountLabel(entry.call_count) }} · 记录时点：{{ toolTimeLabel(entry.at_ms) }}</p>
        </template>
      </article>
      <p class="monitor-note">结果为空不等于有可用资料；批次事件不计为单次工具调用。记录时点相对请求开始，工具耗时可能与其他阶段嵌套，不能相加计算请求总耗时。</p>
    </template>
  </details>
</template>
