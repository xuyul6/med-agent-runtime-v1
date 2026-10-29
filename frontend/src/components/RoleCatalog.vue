<script setup>
import { jsonSchema, roleSkill, stringList } from '../utils/roles.js'

defineProps({ profiles: Array, skills: Object, loading: Boolean, roleNames: Object })
defineEmits(['refresh'])
const whenNames = { critical_risk: '发现紧急风险', human_requested: '用户要求人工协助', missing_information: '缺少必要信息', professional_review: '需要专业人员核实' }
const targetNames = { risk_handoff: '风险提示', offline_professional: '提示线下专业人员', user: '向用户澄清' }
</script>

<template>
  <div class="panel-heading"><div><span class="eyebrow">ROLE CONTRACTS & SKILLS</span><h2>角色契约</h2></div><button class="quiet" :disabled="loading" @click="$emit('refresh')">{{ loading ? '获取中…' : '刷新配置' }}</button></div>
  <p class="hint">展示当前配置；某次咨询实际使用的角色与规则快照，请查看该次执行详情。工具范围是允许使用的范围，不代表每次都会调用。</p>
  <p v-if="loading" class="hint" role="status">正在获取角色和规则配置…</p>
  <template v-else>
    <div class="catalog-snapshot">
      <h3>当前规则快照</h3>
      <template v-if="skills"><p>加载时间：{{ skills.loaded_at || '未知' }}</p><p>快照标识：<code>{{ skills.snapshot_id || '未知' }}</code></p><p>共享规则哈希：<code>{{ skills.common_hash || '未知' }}</code></p></template>
      <p v-else role="status">规则快照不可用，请刷新后重试。</p>
    </div>
    <p v-if="!profiles" class="hint" role="status">角色契约不可用，请刷新后重试。</p>
    <p v-else-if="!profiles.length" class="hint">当前没有已登记角色。</p>
    <article v-for="profile in profiles || []" :key="profile.role" class="role-card">
      <div class="role-heading"><div><h3>{{ roleNames?.[profile.role] || profile.role }}</h3><small>{{ profile.role }} · 契约 {{ profile.version || '未知版本' }}</small></div><span v-if="profile.deterministic === true" class="tag">确定性执行</span></div>
      <p>{{ profile.mission || '未提供角色使命' }}</p>
      <p class="role-label">规则版本</p>
      <template v-if="roleSkill(skills, profile.role)"><p>{{ roleSkill(skills, profile.role).version || '未知版本' }} · {{ roleSkill(skills, profile.role).description || '未提供说明' }}</p><p class="role-meta">文件哈希：<code>{{ roleSkill(skills, profile.role).sha256 || '未知' }}</code></p><p class="role-meta">章节：{{ stringList(roleSkill(skills, profile.role).sections).join('、') || '未提供' }}</p></template>
      <p v-else class="hint">{{ skills ? '该角色的规则元数据未提供。' : '规则快照不可用。' }}</p>
      <details><summary>工作流、工具与交接</summary>
        <h4>工作流</h4><ol v-if="Array.isArray(profile.workflow) && profile.workflow.length"><li v-for="(step, index) in profile.workflow" :key="index">{{ step?.purpose || '未提供说明' }} <small>（{{ step?.name || '未知步骤' }}）</small></li></ol><p v-else class="hint">未提供工作流。</p>
        <h4>工具白名单</h4><p>{{ Array.isArray(profile.tool_scope) ? stringList(profile.tool_scope).join('、') || '无' : '未提供' }}</p>
        <h4>知识域</h4><p>{{ Array.isArray(profile.knowledge_domains) ? stringList(profile.knowledge_domains).join('、') || '无' : '未提供' }}</p>
        <h4>交接条件</h4><ul v-if="Array.isArray(profile.handoff_conditions) && profile.handoff_conditions.length"><li v-for="(condition, index) in profile.handoff_conditions" :key="index">{{ whenNames[condition?.when] || condition?.when || '未知条件' }} → {{ targetNames[condition?.target] || condition?.target || '未知目标' }}</li></ul><p v-else class="hint">未提供交接条件。</p>
        <p class="hint">线下专业人员和风险提示均不代表系统已实际接通医护或急救服务。</p>
      </details>
      <details><summary>输入 Schema</summary><pre v-if="jsonSchema(profile.input_contract)">{{ jsonSchema(profile.input_contract) }}</pre><p v-else class="hint">输入 Schema 不可用。</p></details>
      <details><summary>输出 Schema</summary><pre v-if="jsonSchema(profile.output_contract)">{{ jsonSchema(profile.output_contract) }}</pre><p v-else class="hint">输出 Schema 不可用。</p></details>
    </article>
  </template>
</template>
