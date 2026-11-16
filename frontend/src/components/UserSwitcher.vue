<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { api } from '../api/client.js'
import { errorMessage } from '../lib/format.js'
import { isValidUserId, knownUsers, newUserId, normalizeUserId, removeUser } from '../lib/identity.js'

const props = defineProps({ user: { type: String, required: true } })
const emit = defineEmits(['switch'])

const open = ref(false)
const users = ref([])
const typed = ref('')
const root = ref(null)
const pending = ref('')
const deleting = ref(false)
const notice = ref(null)

function toggle () {
  open.value = !open.value
  if (open.value) {
    users.value = knownUsers()
    typed.value = ''
    pending.value = ''
    notice.value = null
  }
}

function askDelete (id) {
  pending.value = pending.value === id ? '' : id
  notice.value = null
}

async function confirmDelete () {
  const id = pending.value
  if (!id || deleting.value) return
  deleting.value = true
  try {
    const result = await api.forgetUser(id)
    const next = removeUser(id)
    users.value = knownUsers()
    notice.value = { tone: 'ok', text: `已删除 ${id}：会话 ${result.sessions} 个、历史摘要 ${result.summaries} 条、预约 ${result.bookings} 个。` }
    if (id === props.user) emit('switch', next)
  } catch (error) {
    notice.value = { tone: 'bad', text: errorMessage(error) }
  } finally {
    deleting.value = false
    pending.value = ''
  }
}

function pick (id) {
  open.value = false
  if (id !== props.user) emit('switch', id)
}

function enter () {
  if (isValidUserId(typed.value)) pick(normalizeUserId(typed.value))
}

function outside (event) {
  if (open.value && root.value && !root.value.contains(event.target)) open.value = false
}

onMounted(() => document.addEventListener('pointerdown', outside))
onBeforeUnmount(() => document.removeEventListener('pointerdown', outside))
</script>

<template>
  <div ref="root" class="switcher" @keydown.esc="open = false">
    <button type="button" class="pill" :aria-expanded="open" aria-haspopup="dialog" @click="toggle">
      <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><circle cx="12" cy="8.5" r="3.6" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M5 19.5c1.2-3.4 4-5 7-5s5.8 1.6 7 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>
      <span class="mono">{{ user }}</span>
    </button>

    <div v-if="open" class="pop card" role="dialog" aria-label="演示身份">
      <p class="eyebrow">演示身份</p>
      <p class="note">没有登录认证。ID 只用来区分不同用户的会话、历史摘要和预约；换回同一个 ID，就能拿回它的历史摘要。别人输入同一个 ID 也能看到这些内容，请不要用真实姓名。</p>

      <ul class="list">
        <li v-for="id in users" :key="id">
          <div class="row" :class="{ current: id === user }">
            <button type="button" class="name" @click="pick(id)">
              <span class="mono">{{ id }}</span>
              <span v-if="id === user" class="chip accent">当前</span>
            </button>
            <button type="button" class="trash" :aria-label="'删除用户 ' + id" :aria-expanded="pending === id" @click="askDelete(id)">
              <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path d="M4 7h16M10 11v6M14 11v6M6 7l1 12h10l1-12M9 7V4h6v3" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
            </button>
          </div>
          <div v-if="pending === id" class="confirm-delete" role="alert">
            <p>删除 <span class="mono">{{ id }}</span> 在服务器上的会话、历史摘要和预约？删除后无法恢复，再用这个 ID 会从头开始。</p>
            <div class="confirm-actions">
              <button type="button" class="btn small danger" :disabled="deleting" @click="confirmDelete">{{ deleting ? '删除中…' : '确认删除' }}</button>
              <button type="button" class="btn ghost small" :disabled="deleting" @click="pending = ''">取消</button>
            </div>
          </div>
        </li>
      </ul>
      <p v-if="notice" class="notice" :class="notice.tone" role="status">{{ notice.text }}</p>

      <form class="enter" @submit.prevent="enter">
        <input v-model="typed" class="field mono" placeholder="自定义 ID，如 tester_01" aria-label="输入用户 ID" maxlength="32" spellcheck="false" autocomplete="off">
        <button type="submit" class="btn small" :disabled="!isValidUserId(typed)">进入</button>
      </form>
      <p class="hint" :class="{ bad: typed.trim() && !isValidUserId(typed) }">2–32 个字符，可用中英文、数字、下划线和连字符。输入已有 ID 即切回，新的 ID 即新建。</p>
      <button type="button" class="btn small fresh" @click="pick(newUserId())">随机生成一个</button>
    </div>
  </div>
</template>

<style scoped>
.switcher { position: relative; }
.pill {
  display: inline-flex; align-items: center; gap: 6px; height: 30px; padding: 0 12px; border-radius: 999px;
  border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); font-size: 13px;
}
.pill:hover { border-color: var(--ink-3); }
.pop {
  position: absolute; right: 0; top: calc(100% + 8px); z-index: 30; width: min(300px, calc(100vw - 32px)); padding: 16px;
  max-height: calc(100vh - 80px); overflow-y: auto;
  box-shadow: var(--shadow);
}
.pop .eyebrow { margin: 0; }
.note { margin: 6px 0 12px; font-size: 12.5px; line-height: 1.6; color: var(--ink-2); }
.list { list-style: none; margin: 0 0 12px; padding: 0; display: flex; flex-direction: column; gap: 2px; }
.row { display: flex; align-items: center; gap: 2px; border-radius: var(--radius-sm); }
.row:hover { background: var(--sunken); }
.row.current { background: var(--accent-soft); }
.name {
  flex: 1; min-width: 0; display: flex; justify-content: space-between; align-items: center; gap: 8px; padding: 6px 10px;
  border: 0; background: transparent; text-align: left;
}
.name .mono { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.trash {
  flex: none; width: 30px; height: 30px; display: grid; place-items: center; border: 0; border-radius: 6px;
  background: transparent; color: var(--ink-3);
}
.trash:hover, .trash[aria-expanded='true'] { color: var(--critical); background: var(--critical-soft); }
.confirm-delete { margin: 4px 0 8px; padding: 10px 12px; border-radius: var(--radius-sm); background: var(--critical-soft); }
.confirm-delete p { margin: 0 0 8px; font-size: 12.5px; line-height: 1.6; color: var(--ink); }
.confirm-actions { display: flex; gap: 6px; }
.btn.danger { background: var(--critical); border-color: var(--critical); color: #fff; }
.notice { margin: 0 0 10px; font-size: 12.5px; }
.notice.ok { color: var(--good); }
.notice.bad { color: var(--critical); }
.enter { display: flex; gap: 8px; margin-bottom: 8px; }
.enter .field { flex: 1; min-width: 0; height: 30px; font-size: 13px; }
.fresh { width: 100%; }
.hint { margin: 0 0 10px; font-size: 12px; color: var(--ink-3); }
.hint.bad { color: var(--critical); }
</style>
