<script setup>
import { computed, onMounted, ref } from 'vue'
import { api } from './api/client.js'
import { demoUser, useUser } from './lib/identity.js'
import ConsultView from './components/ConsultView.vue'
import AppointmentsView from './components/AppointmentsView.vue'
import RuntimeView from './components/RuntimeView.vue'
import UserSwitcher from './components/UserSwitcher.vue'

const user = ref(demoUser())

function switchUser (id) {
  user.value = useUser(id)
}
const health = ref(null)
const healthFailed = ref(false)
const catalog = ref(null)
const subject = ref('self')
const tab = ref('consult')

const subjects = computed(() => (catalog.value && catalog.value.subjects) || [{ id: 'self', label: '本人' }])
const developer = computed(() => Boolean(health.value && health.value.developer_tools))
const tabs = computed(() => [
  { id: 'consult', label: '咨询' },
  { id: 'appointments', label: '预约' },
  ...(developer.value ? [{ id: 'runtime', label: '运行状态' }] : [])
])

const status = computed(() => {
  if (healthFailed.value) return { tone: 'bad', text: '服务未连接' }
  if (!health.value) return { tone: '', text: '连接中' }
  if (health.value.preview) return { tone: 'caution', text: '离线预览' }
  if (health.value.status === 'degraded') return { tone: 'caution', text: '部分服务降级' }
  return { tone: 'ok', text: health.value.model || '在线' }
})

onMounted(async () => {
  try {
    health.value = await api.health()
  } catch {
    healthFailed.value = true
  }
  try {
    catalog.value = await api.catalog()
  } catch {}
})
</script>

<template>
  <div class="shell">
    <header class="top">
      <div class="brand">
        <span class="mark" aria-hidden="true">
          <svg viewBox="0 0 24 24" width="22" height="22"><path d="M12 3v18M3 12h18" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" fill="none"/><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="1.2" fill="none" opacity=".35"/></svg>
        </span>
        <span class="serif name">MedAgent</span>
        <span class="tagline">健康信息助手</span>
      </div>

      <nav class="tabs" aria-label="视图">
        <button v-for="item in tabs" :key="item.id" type="button" :class="{ on: tab === item.id }"
                :aria-current="tab === item.id ? 'page' : undefined" @click="tab = item.id">{{ item.label }}</button>
      </nav>

      <div class="right">
        <UserSwitcher :user="user" @switch="switchUser" />
        <label class="who">
          <span class="muted">咨询对象</span>
          <select v-model="subject" class="who-select" aria-label="咨询对象">
            <option v-for="item in subjects" :key="item.id" :value="item.id">{{ item.label }}</option>
          </select>
        </label>
        <span class="status" :title="health && health.services ? JSON.stringify(health.services) : ''">
          <span class="dot" :class="status.tone"></span>{{ status.text }}
        </span>
      </div>
    </header>

    <main class="stage">
      <ConsultView v-show="tab === 'consult'" :key="user" :user="user" :subject="subject" :subjects="subjects" :preview="Boolean(health && health.preview)" />
      <AppointmentsView v-if="tab === 'appointments'" :key="user" :user="user" :subject="subject" :catalog="catalog" />
      <RuntimeView v-if="tab === 'runtime' && developer" />
    </main>
  </div>
</template>

<style scoped>
.shell { height: 100%; display: flex; flex-direction: column; }

.top {
  flex: none; height: 60px; padding: 0 24px;
  display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 16px;
  border-bottom: 1px solid var(--line); background: var(--paper);
}
.brand { display: flex; align-items: baseline; gap: 10px; min-width: 0; }
.mark { color: var(--accent); align-self: center; display: inline-flex; }
.name { font-size: 19px; }
.tagline { font-size: 13px; color: var(--ink-3); white-space: nowrap; }

.tabs { display: flex; gap: 4px; padding: 3px; border-radius: 999px; background: var(--sunken); border: 1px solid var(--line); }
.tabs button {
  border: 0; background: transparent; height: 30px; padding: 0 16px; border-radius: 999px;
  color: var(--ink-2); font-size: 14px;
}
.tabs button.on { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgba(0, 0, 0, .08); }

.right { display: flex; align-items: center; justify-content: flex-end; gap: 12px; min-width: 0; }
.who { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.who-select {
  height: 30px; padding: 0 28px 0 12px; border-radius: 999px; border: 1px solid var(--line-strong);
  background: var(--surface); font-size: 13px; appearance: none;
  background-image: linear-gradient(45deg, transparent 50%, var(--ink-3) 50%), linear-gradient(135deg, var(--ink-3) 50%, transparent 50%);
  background-position: calc(100% - 15px) 13px, calc(100% - 10px) 13px; background-size: 5px 5px; background-repeat: no-repeat;
}
.status { display: inline-flex; align-items: center; gap: 8px; font-size: 13px; color: var(--ink-2); white-space: nowrap; }

.stage { flex: 1; min-height: 0; }

@media (max-width: 820px) {
  .top { grid-template-columns: 1fr auto; height: auto; padding: 10px 16px; row-gap: 10px; }
  .tagline, .status, .who > .muted { display: none; }
  .tabs { grid-column: 1 / -1; justify-self: start; }
  .right { grid-row: 1; grid-column: 2; }
}
</style>
