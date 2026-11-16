<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api } from '../api/client.js'
import { dayLabel, errorMessage, windowLabel } from '../lib/format.js'
import ConfirmCard from './ConfirmCard.vue'

const props = defineProps({
  user: { type: String, required: true },
  subject: { type: String, default: 'self' },
  catalog: { type: Object, default: null }
})

const bookings = ref([])
const loadingBookings = ref(true)
const bookingsError = ref('')
const cancelDraft = ref(null)
const notice = ref('')

const hospitalId = ref('')
const departmentId = ref('')
const day = ref('')
const timeWindow = ref('')
const slots = ref(null)
const searching = ref(false)
const searchError = ref('')
const createDraft = ref(null)
const drafting = ref('')

const hospitals = computed(() => (props.catalog && props.catalog.hospitals) || [])
const departments = computed(() => (hospitals.value.find(item => item.id === hospitalId.value) || { departments: [] }).departments)
const subjectLabel = computed(() => ((props.catalog && props.catalog.subjects) || []).find(item => item.id === props.subject)?.label || '本人')

function seedForm () {
  if (!props.catalog) return
  if (!hospitalId.value && hospitals.value.length) hospitalId.value = hospitals.value[0].id
  if (!day.value) day.value = props.catalog.first_day
}

watch(() => props.catalog, seedForm, { immediate: true })
watch(hospitalId, () => {
  if (!departments.value.some(item => item.id === departmentId.value)) departmentId.value = departments.value[0]?.id || ''
  slots.value = null
}, { immediate: true })
watch([departmentId, day, timeWindow], () => { slots.value = null; createDraft.value = null })

async function loadBookings () {
  loadingBookings.value = true
  bookingsError.value = ''
  try {
    bookings.value = await api.bookings(props.user)
  } catch (error) {
    bookingsError.value = errorMessage(error)
  } finally {
    loadingBookings.value = false
  }
}

async function search ({ keepDraft = false } = {}) {
  if (!hospitalId.value || !departmentId.value || !day.value) return
  searching.value = true
  searchError.value = ''
  if (!keepDraft) createDraft.value = null
  try {
    slots.value = await api.slots({ hospitalId: hospitalId.value, departmentId: departmentId.value, day: day.value, window: timeWindow.value || undefined })
  } catch (error) {
    slots.value = null
    searchError.value = errorMessage(error)
  } finally {
    searching.value = false
  }
}

async function choose (slot) {
  drafting.value = slot.slot_id
  searchError.value = ''
  try {
    createDraft.value = await api.draftCreate({ user: props.user, subject: props.subject, slotId: slot.slot_id })
  } catch (error) {
    searchError.value = errorMessage(error)
  } finally {
    drafting.value = ''
  }
}

async function startCancel (booking) {
  bookingsError.value = ''
  notice.value = ''
  try {
    cancelDraft.value = { booking: booking.appointment_id, action: await api.draftCancel({ user: props.user, subject: booking.subject, appointmentId: booking.appointment_id }) }
  } catch (error) {
    bookingsError.value = errorMessage(error)
  }
}

function cancelled (result) {
  const item = result.appointment
  notice.value = `已取消 ${dayLabel(item.day)} ${windowLabel(item.window)} · ${item.department}`
  cancelDraft.value = null
  settled()
}

function settled () {
  loadBookings()
  if (slots.value) search({ keepDraft: true })
}

onMounted(loadBookings)
</script>

<template>
  <div class="page">
    <header class="intro">
      <p class="eyebrow">模拟医院 · 合成号源</p>
      <h1 class="serif">预约</h1>
      <p class="muted">号源和医院都是演示数据。每次预约或取消都先生成草稿，需要你亲自确认才会写入。</p>
    </header>

    <div class="grid">
      <section class="card panel">
        <div class="panel-head">
          <h2>我的预约</h2>
          <button type="button" class="btn ghost small" :disabled="loadingBookings" @click="loadBookings">刷新</button>
        </div>
        <p v-if="bookingsError" class="error">{{ bookingsError }}</p>
        <p v-if="notice" class="notice">{{ notice }}</p>
        <p v-if="loadingBookings && !bookings.length" class="muted empty">加载中…</p>
        <p v-else-if="!bookings.length" class="muted empty">还没有即将到来的预约。</p>
        <ul v-else class="bookings">
          <li v-for="item in bookings" :key="item.appointment_id">
            <div class="date">
              <span class="serif day">{{ dayLabel(item.day).split(' ')[0] }}</span>
              <span class="muted">{{ dayLabel(item.day).split(' ')[1] }} · {{ windowLabel(item.window) }}</span>
            </div>
            <div class="where">
              <strong>{{ item.department }}</strong>
              <span class="muted">{{ item.hospital }} · 就诊人 {{ item.subject_label }}</span>
            </div>
            <button type="button" class="btn small" @click="startCancel(item)">取消</button>
            <ConfirmCard v-if="cancelDraft && cancelDraft.booking === item.appointment_id" class="inline" :action="cancelDraft.action" :user="user"
                         @settled="cancelled" @dismiss="cancelDraft = null" />
          </li>
        </ul>
      </section>

      <section class="card panel">
        <div class="panel-head">
          <h2>预约新号</h2>
          <span class="chip accent">就诊人：{{ subjectLabel }}</span>
        </div>
        <p v-if="!catalog" class="muted empty">号源目录加载中…</p>
        <form v-else class="form" @submit.prevent="search()">
          <label>
            <span>医院</span>
            <select v-model="hospitalId" class="field">
              <option v-for="item in hospitals" :key="item.id" :value="item.id">{{ item.name }}</option>
            </select>
          </label>
          <label>
            <span>科室</span>
            <select v-model="departmentId" class="field">
              <option v-for="item in departments" :key="item.id" :value="item.id">{{ item.name }}</option>
            </select>
          </label>
          <label>
            <span>日期</span>
            <input v-model="day" type="date" class="field" :min="catalog.first_day" :max="catalog.last_day">
          </label>
          <div class="seg-wrap">
            <span>时段</span>
            <div class="seg" role="radiogroup" aria-label="时段">
              <button v-for="option in [['', '全天'], ['am', '上午'], ['pm', '下午']]" :key="option[0]" type="button" role="radio"
                      :aria-checked="timeWindow === option[0]" :class="{ on: timeWindow === option[0] }" @click="timeWindow = option[0]">{{ option[1] }}</button>
            </div>
          </div>
          <button type="submit" class="btn primary" :disabled="searching || !departmentId || !day">{{ searching ? '查询中…' : '查询号源' }}</button>
        </form>

        <p v-if="searchError" class="error">{{ searchError }}</p>
        <div v-if="slots" class="slots">
          <p v-if="!slots.length" class="muted empty">这一天没有可约的号。</p>
          <div v-for="slot in slots" :key="slot.slot_id" class="slot">
            <div>
              <strong>{{ dayLabel(slot.day) }} · {{ windowLabel(slot.window) }}</strong>
              <span class="muted">{{ slot.hospital }} · {{ slot.department }}</span>
            </div>
            <span class="left" :class="{ none: slot.remaining <= 0 }">余 <span class="mono">{{ slot.remaining }}</span></span>
            <button type="button" class="btn small" :disabled="slot.remaining <= 0 || drafting === slot.slot_id" @click="choose(slot)">
              {{ drafting === slot.slot_id ? '生成中…' : '选这个' }}
            </button>
          </div>
        </div>
        <ConfirmCard v-if="createDraft" :key="createDraft.draft_id" :action="createDraft" :user="user" @settled="settled" @dismiss="createDraft = null" />
      </section>
    </div>
  </div>
</template>

<style scoped>
.page { height: 100%; overflow-y: auto; padding: 40px 24px 64px; }
.intro, .grid { max-width: 1080px; margin: 0 auto; }
.intro h1 { font-size: 30px; margin: 4px 0 6px; }
.intro p.muted { margin: 0 0 28px; font-size: 14px; }
.grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; align-items: start; }

.panel { padding: 20px 22px; }
.panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.panel-head h2 { margin: 0; font-size: 16px; font-weight: 600; }
.empty { font-size: 14px; padding: 24px 0; text-align: center; margin: 0; }
.error { color: var(--critical); font-size: 13.5px; margin: 8px 0; }
.notice { color: var(--good); font-size: 13.5px; margin: 0 0 8px; }

.bookings { list-style: none; margin: 0; padding: 0; }
.bookings li { display: grid; grid-template-columns: 112px 1fr auto; align-items: center; gap: 14px; padding: 14px 0; border-top: 1px solid var(--line); }
.bookings li:first-child { border-top: 0; }
.date, .where { display: flex; flex-direction: column; font-size: 13px; line-height: 1.5; }
.day { font-size: 18px; }
.where strong { font-size: 15px; font-weight: 500; }
.inline { grid-column: 1 / -1; margin-top: 0; }

.form { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 14px; }
.form label, .seg-wrap { display: flex; flex-direction: column; gap: 4px; font-size: 12.5px; color: var(--ink-3); }
.form .field { color: var(--ink); font-size: 14px; }
.form .btn.primary { grid-column: 1 / -1; height: 40px; }
.seg { display: flex; padding: 3px; border-radius: var(--radius-sm); background: var(--sunken); border: 1px solid var(--line); height: 38px; }
.seg button { flex: 1; border: 0; background: transparent; border-radius: 6px; font-size: 13.5px; color: var(--ink-2); }
.seg button.on { background: var(--surface); color: var(--ink); box-shadow: 0 1px 2px rgba(0, 0, 0, .08); }

.slots { margin-top: 16px; }
.slot { display: grid; grid-template-columns: 1fr auto auto; gap: 14px; align-items: center; padding: 12px 0; border-top: 1px solid var(--line); }
.slot > div { display: flex; flex-direction: column; font-size: 13px; }
.slot strong { font-size: 14.5px; font-weight: 500; }
.left { font-size: 13px; color: var(--good); }
.left.none { color: var(--ink-3); }

@media (max-width: 860px) {
  .grid { grid-template-columns: 1fr; }
}
@media (max-width: 520px) {
  .page { padding: 24px 16px 48px; }
  .form { grid-template-columns: 1fr; }
  .bookings li { grid-template-columns: 1fr auto; }
  .bookings .date { grid-column: 1 / -1; flex-direction: row; gap: 8px; align-items: baseline; }
}
</style>
