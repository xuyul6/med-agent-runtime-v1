<script setup>
import { label, list, scoreLabel } from '../utils/execution.js'

defineProps({ decision: Object, roleNames: Object })
const statuses = { single: '单角色', planned: '已规划', merged: '已合并', partial: '部分完成', conflict: '存在冲突', failed: '失败', answered: '已回答', needs_clarification: '需要澄清', handoff_required: '需要交接提示', cancelled: '已取消', ok: '成功', timeout: '超时' }
const components = { intent: '意图', task_fit: '任务适配', entity_fit: '实体适配', business_rule: '业务规则', health: '运行健康度' }
</script>

<template>
  <details class="execution-details">
    <summary>查看路由与执行</summary>
    <p v-if="!decision" class="monitor-note">本次路由决策不可用。</p>
    <template v-else>
      <p class="monitor-note">版本：{{ decision.version || '未提供' }} · 协作：{{ label(statuses, decision.collaboration_status) }}</p>
      <p class="monitor-note">路由分数：{{ scoreLabel(decision.routing_confidence) }}（工程评分，不是概率）</p>
      <p class="monitor-note">路由原因：{{ decision.routing_reason || '未提供' }}</p>
      <h3>候选角色</h3>
      <p class="monitor-note">健康度无样本时使用默认初值，不代表可靠性已验证。</p>
      <p v-if="!list(decision.candidates)" class="monitor-note">候选数据不可用。</p>
      <p v-else-if="!decision.candidates.length" class="monitor-note">无候选评分；风险短路等路径可直接指定角色。</p>
      <article v-for="(candidate, index) in list(decision.candidates) || []" :key="index" class="execution-row">
        <strong>{{ label(roleNames || {}, candidate?.role) }}</strong>
        <p>准入：{{ candidate?.eligible === true ? '是' : candidate?.eligible === false ? '否' : '未知' }} · 总分：{{ scoreLabel(candidate?.score) }}</p>
        <p v-if="Array.isArray(candidate?.reasons) && candidate.reasons.length">原因：{{ candidate.reasons.join('、') }}</p>
        <p v-if="candidate?.components && typeof candidate.components === 'object'">分项：{{ Object.entries(candidate.components).map(([key, value]) => `${label(components, key)} ${scoreLabel(value)}`).join(' · ') || '未提供' }}</p>
      </article>
      <h3>任务执行</h3>
      <p v-if="!list(decision.assignments)" class="monitor-note">任务数据不可用。</p>
      <p v-else-if="!decision.assignments.length" class="monitor-note">没有分配可执行任务。</p>
      <article v-for="(assignment, index) in list(decision.assignments) || []" :key="index" class="execution-row">
        <strong>{{ label(roleNames || {}, assignment?.role) }}</strong><p>状态：{{ label(statuses, assignment?.status) }} · 选中实例：{{ assignment?.selected_instance === null ? '无' : assignment?.selected_instance || '未提供' }}</p>
        <p v-if="Array.isArray(assignment?.attempts) && assignment.attempts.length">尝试：{{ assignment.attempts.map(attempt => `${attempt?.instance || '未知实例'} ${label(statuses, attempt?.status)}${attempt?.error_code ? ` (${attempt.error_code})` : attempt?.error ? ` (${attempt.error})` : ''}`).join('；') }}</p>
        <p v-else>尝试：{{ Array.isArray(assignment?.attempts) ? '无' : '未提供' }}</p>
      </article>
      <h3>未完成与回退记录</h3>
      <p class="monitor-note">监控回退角色：{{ list(decision.monitor_fallback) ? decision.monitor_fallback.map(role => label(roleNames || {}, role)).join('、') || '无' : '未提供' }}</p>
      <p class="monitor-note">延后或未准入角色：{{ list(decision.deferred_roles) ? decision.deferred_roles.map(role => label(roleNames || {}, role)).join('、') || '无' : '未提供' }}</p>
      <p v-if="!list(decision.fallbacks)" class="monitor-note">回退记录不可用。</p>
      <p v-else-if="!decision.fallbacks.length" class="monitor-note">没有回退记录。</p>
      <article v-for="(fallback, index) in list(decision.fallbacks) || []" :key="index" class="execution-row"><strong>{{ label(roleNames || {}, fallback?.role) }}</strong><p>{{ fallback?.instance || '角色级' }} · 原因：{{ fallback?.reason || '未提供' }} · 动作：{{ fallback?.action || '未提供' }}</p></article>
      <p class="monitor-note">回退记录包括规划和执行阶段；具体实例是否尝试以任务执行记录为准。</p>
    </template>
  </details>
</template>
