<script setup>
import { checkSummary, comparisonLabel, countLabel, scoreLabel, usageLabel } from '../utils/evaluation.js'

defineProps({ report: { type: Object, required: true } })
</script>

<template>
  <div class="evaluation-summary">
    <strong>{{ countLabel(report.passed) }} / {{ countLabel(report.total) }}</strong>
    <span>综合通过 · 规则通过 {{ countLabel(report.rule_passed) }} 项 · 已执行 {{ countLabel(report.executed) }} 项</span>
  </div>
  <p v-if="!report.complete" class="hint">本次未完整运行：{{ report.stop_reason || '原因未提供或资源清理未确认' }}。</p>
  <p v-if="Array.isArray(report.cleanup_errors) && report.cleanup_errors.length" class="degraded">清理错误：{{ report.cleanup_errors.join('、') }}</p>
  <p v-if="Array.isArray(report.not_run) && report.not_run.length" class="hint">未运行案例：{{ report.not_run.join('、') }}</p>
  <p class="hint">指标版本：{{ report.metadata?.metric_version || '未提供' }} · 数据集：{{ report.dataset || '未提供' }} · 模式：{{ report.mode || '未提供' }}</p>
  <p v-if="report.classification" class="hint">分类器 Accuracy {{ scoreLabel(report.classification.classifier?.accuracy) }} · Macro-F1 {{ scoreLabel(report.classification.classifier?.macro_f1) }} · 覆盖率 {{ scoreLabel(report.classification.classifier?.coverage) }}；急症筛选控后 Accuracy {{ scoreLabel(report.classification.risk_gated?.accuracy) }} · Macro-F1 {{ scoreLabel(report.classification.risk_gated?.macro_f1) }}。</p>
  <p v-else class="hint">分类指标不可用。</p>
  <p class="hint">{{ comparisonLabel(report.comparison) }}</p>
  <p v-if="report.baseline?.saved === true" class="hint">基线已保存：{{ report.baseline.id }}</p>
  <p v-else-if="report.baseline?.saved === false" class="degraded">基线未保存：{{ report.baseline.reason || '原因未提供' }}</p>
  <template v-if="Array.isArray(report.comparison?.regressions)">
    <p v-if="report.comparison.status === 'compared' && !report.comparison.regressions.length" class="hint">未发现已报告的回归；缺测指标不代表通过。</p>
    <ul v-if="report.comparison.regressions.length" class="evaluation-regressions">
      <li v-for="(change, index) in report.comparison.regressions" :key="index">{{ change?.kind || '未知类型' }} · {{ change?.metric || change?.gate || '未提供指标' }}<template v-if="change?.case_id"> · {{ change.case_id }} · 第 {{ countLabel(change.turn) }} 轮</template></li>
    </ul>
  </template>
  <details v-if="report.comparison?.status === 'compared' && Array.isArray(report.comparison.metrics)" class="evaluation-turns">
    <summary>查看比较指标与实验变量</summary>
    <p v-if="!report.comparison.metrics.length">没有指标比较记录。</p>
    <p v-for="(metric, index) in report.comparison.metrics" :key="index">{{ metric?.metric || '未知指标' }}：{{ scoreLabel(metric?.baseline) }} → {{ scoreLabel(metric?.current) }} · {{ metric?.status || '状态未知' }}</p>
    <p v-if="Array.isArray(report.comparison.variables) && !report.comparison.variables.length">未报告实验变量变化。</p>
    <p v-else-if="Array.isArray(report.comparison.variables)">实验变量发生变化：{{ report.comparison.variables.map(item => item?.field || '未知字段').join('、') }}</p>
    <p v-else>实验变量信息不可用。</p>
  </details>
  <p v-if="report.judge?.enabled === true" class="hint">Judge 有效 {{ countLabel(report.judge.scored) }} / {{ countLabel(report.judge.total) }} 轮 · 缺测 {{ countLabel(report.judge.missing) }} · 错误 {{ countLabel(report.judge.errors) }}；有效均分不代表所有轮次通过。</p>
  <p v-else-if="report.judge?.enabled === false" class="hint">本次未启用 Judge；无质量评分。</p>
  <p v-else class="hint">Judge 状态不可用。</p>
  <p class="hint">模型用量（输入 / 输出 Token）：运行时 {{ usageLabel(report.usage?.runtime, 'input_tokens') }} / {{ usageLabel(report.usage?.runtime, 'output_tokens') }}；Judge {{ usageLabel(report.usage?.judge, 'input_tokens') }} / {{ usageLabel(report.usage?.judge, 'output_tokens') }}。</p>
  <p class="hint">预算：已尝试 {{ countLabel(report.budget?.attempted_calls) }} / {{ countLabel(report.limits?.max_model_calls) }} 次调用；已预留输出额度 {{ countLabel(report.budget?.reserved_max_output_tokens) }} / {{ countLabel(report.limits?.max_output_tokens) }} Token（预留额不是实际输出量）。</p>
  <p v-if="!report.cases.length" class="hint">没有已执行案例。</p>
  <article v-for="(result, index) in report.cases" :key="index" class="evaluation-case">
    <div><strong>{{ result?.id || '未知案例' }}</strong><span :class="result?.passed === true ? 'good' : 'degraded'">{{ result?.passed === true ? '通过' : result?.passed === false ? '未通过' : '状态未知' }}</span></div>
    <p v-if="result?.error">执行错误：{{ result.error }}</p>
    <p v-if="result?.kind === 'intent'">意图检查：{{ result.expected || '未提供' }} → {{ result.predicted || '未提供' }}；不使用 Judge。</p>
    <p v-if="result?.kind === 'intent'">行为：{{ checkSummary(result.checks) }} · 安全：{{ checkSummary(result.safety_checks) }}</p>
    <p v-if="Array.isArray(result?.not_run_turns) && result.not_run_turns.length">未运行轮次：{{ result.not_run_turns.join('、') }}</p>
    <details v-if="Array.isArray(result?.turns) && result.turns.length" class="evaluation-turns">
      <summary>查看逐轮结果（{{ result.turns.length }} 轮）</summary>
      <div v-for="(turn, turnIndex) in result.turns" :key="turnIndex" class="evaluation-turn">
        <strong>第 {{ countLabel(turn?.turn) }} 轮 · {{ turn?.passed === true ? '通过' : turn?.passed === false ? '未通过' : '状态未知' }}</strong>
        <p>行为：{{ checkSummary(turn?.checks) }}</p>
        <p>安全：{{ checkSummary(turn?.safety_checks) }}</p>
        <p v-if="turn?.error">执行错误：{{ turn.error }}</p>
        <p v-if="turn?.judge_error">Judge 错误：{{ turn.judge_error }}；无有效分数。</p>
        <p v-else-if="report.judge?.enabled === false">Judge 未启用。</p>
        <p v-else>Judge 均分：{{ scoreLabel(turn?.quality?.mean) }}<template v-if="turn?.quality?.reason"> · {{ turn.quality.reason }}</template></p>
        <p>Trace：{{ turn?.trace?.request_id || '未提供' }}</p>
      </div>
    </details>
  </article>
</template>
