<script setup>
import { computed, onBeforeUnmount, ref } from 'vue'
import { api } from '../api/client.js'
import { errorMessage, remaining } from '../lib/format.js'

const props = defineProps({
  action: { type: Object, required: true },
  user: { type: String, required: true }
})
const emit = defineEmits(['settled', 'dismiss'])

const now = ref(Date.now())
const timer = setInterval(() => { now.value = Date.now() }, 1000)
onBeforeUnmount(() => clearInterval(timer))

const phase = ref('idle')
const message = ref('')
const cancel = computed(() => props.action.kind === 'appointment_cancel')
const clock = computed(() => remaining(props.action.expires_at, now.value))

async function confirm () {
  if (phase.value === 'working' || clock.value.expired) return
  phase.value = 'working'
  try {
    const result = await api.confirm({ user: props.user, draftId: props.action.draft_id, token: props.action.token })
    phase.value = 'done'
    message.value = result.status === 'cancelled' ? '已取消这次预约。' : '预约成功，可以在「预约」里查看。'
    emit('settled', result)
  } catch (error) {
    phase.value = 'error'
    message.value = errorMessage(error)
  }
}

function dismiss () {
  phase.value = 'dismissed'
  emit('dismiss')
}
</script>

<template>
  <div class="confirm" :class="[phase, { cancel }]">
    <div class="head">
      <span class="kind">{{ cancel ? '取消预约' : '预约草稿' }}</span>
      <span v-if="phase === 'idle' || phase === 'error'" class="clock mono" :class="{ expired: clock.expired }">{{ clock.text }}</span>
    </div>
    <p class="summary">{{ action.summary }}</p>

    <p v-if="phase === 'done'" class="result ok">{{ message }}</p>
    <p v-else-if="phase === 'dismissed'" class="result muted">已放弃，未做任何更改。</p>
    <template v-else>
      <p v-if="phase === 'error'" class="result bad">{{ message }}</p>
      <div class="actions">
        <button type="button" class="btn small" :class="cancel ? 'danger' : 'primary'" :disabled="phase === 'working' || clock.expired" @click="confirm">
          {{ phase === 'working' ? '提交中…' : cancel ? '确认取消' : '确认预约' }}
        </button>
        <button type="button" class="btn ghost small" :disabled="phase === 'working'" @click="dismiss">不用了</button>
      </div>
      <p class="note">只有你按下确认才会写入（模拟）医院系统，模型无法自行提交。</p>
    </template>
  </div>
</template>

<style scoped>
.confirm {
  margin-top: 18px; padding: 16px 18px; border-radius: var(--radius);
  border: 1px solid var(--line-strong); background: var(--surface); box-shadow: var(--shadow);
}
.confirm.done { border-color: var(--good); box-shadow: none; }
.confirm.dismissed { box-shadow: none; opacity: .75; }
.head { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
.kind { font-size: 12px; letter-spacing: .08em; color: var(--accent); }
.cancel .kind { color: var(--critical); }
.clock { color: var(--ink-3); font-size: 12px; }
.clock.expired { color: var(--critical); }
.summary { margin: 6px 0 12px; font-size: 16px; font-weight: 500; }
.actions { display: flex; gap: 8px; }
.btn.danger { background: var(--critical); border-color: var(--critical); color: #fff; }
.note { margin: 10px 0 0; font-size: 12px; color: var(--ink-3); }
.result { margin: 0 0 10px; font-size: 14px; }
.result.ok { color: var(--good); margin: 0; }
.result.bad { color: var(--critical); }
.result.muted { margin: 0; }
</style>
