<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import AppointmentCard from './AppointmentCard.vue'
import { createDraft, listAppointments, loadCatalog, searchSlots } from '../utils/appointments.js'

const props = defineProps({ user: String, request: Function, cardState: Function, version: Number })
const emit = defineEmits(['confirm', 'dismiss'])
const windowNames = { am: '上午', pm: '下午' }

const catalog = ref(null)
const bookings = ref([])
const loading = ref(false)
const error = ref('')
const form = ref({ hospital_id: '', department_id: '', subject: 'self', day: '', window: '' })
const slots = ref(null)
const selected = ref('')
const manualCard = ref(null)
const cancelCards = ref({})

const departments = computed(() => catalog.value?.hospitals.find(item => item.id === form.value.hospital_id)?.departments || [])
const subjectLabel = computed(() => Object.fromEntries((catalog.value?.subjects || []).map(item => [item.id, item.label])))

async function refresh() {
  loading.value = true
  error.value = ''
  try { bookings.value = await listAppointments(props.request, props.user) }
  catch (failure) { error.value = `预约列表读取失败：${failure.message}` }
  finally { loading.value = false }
}

async function prepareForm() {
  try {
    catalog.value = await loadCatalog(props.request)
    form.value.day = form.value.day || catalog.value.first_day
  } catch (failure) { error.value = `预约目录读取失败：${failure.message}` }
}

async function findSlots() {
  error.value = ''
  selected.value = ''
  manualCard.value = null
  try { slots.value = await searchSlots(props.request, { hospital_id: form.value.hospital_id, department_id: form.value.department_id, day: form.value.day, window: form.value.window }) }
  catch (failure) { slots.value = null; error.value = failure.message === 'HTTP 410' ? '这一天不在可约范围内。' : `号源查询失败：${failure.message}` }
}

async function draftCreate() {
  error.value = ''
  try { manualCard.value = await createDraft(props.request, { user: props.user, subject: form.value.subject, action: 'create', slot_id: selected.value }) }
  catch (failure) { error.value = failure.message === 'HTTP 409' ? '号源已满或已经预约过这个号。' : `生成草稿失败：${failure.message}` }
}

async function draftCancel(booking) {
  error.value = ''
  try {
    const card = await createDraft(props.request, { user: props.user, subject: booking.subject, action: 'cancel', appointment_id: booking.appointment_id })
    cancelCards.value = { ...cancelCards.value, [booking.appointment_id]: card }
  } catch (failure) { error.value = failure.message === 'HTTP 410' ? '当天及已过去的预约不能在这里取消。' : `生成取消草稿失败：${failure.message}` }
}

function reset() {
  bookings.value = []
  slots.value = null
  selected.value = ''
  manualCard.value = null
  cancelCards.value = {}
}

watch(() => form.value.hospital_id, () => { form.value.department_id = ''; slots.value = null })
watch(() => props.user, () => { reset(); refresh() })
watch(() => props.version, refresh)
onMounted(() => { prepareForm(); refresh() })
</script>

<template>
  <div class="appointments">
    <div class="panel-heading"><div><span class="eyebrow">MOCK HOSPITAL</span><h2>我的预约</h2></div><button class="quiet" :disabled="loading" @click="refresh">刷新</button></div>
    <p class="hint">只在本地模拟医院系统中进行，不接入真实医院。过了预约时间的记录不再显示。</p>
    <p v-if="error" class="degraded" role="status">{{ error }}</p>
    <p v-if="!loading && !bookings.length" class="hint">暂无未过期的预约。</p>
    <article v-for="booking in bookings" :key="booking.appointment_id" class="booking-row">
      <div>
        <strong>{{ booking.hospital }} · {{ booking.department }}</strong>
        <p>{{ booking.day }} {{ windowNames[booking.window] }} · 就诊人：{{ booking.subject_label }}</p>
      </div>
      <button v-if="!cancelCards[booking.appointment_id]" class="quiet" @click="draftCancel(booking)">取消</button>
      <AppointmentCard v-else :card="cancelCards[booking.appointment_id]" :state="cardState(cancelCards[booking.appointment_id]).state"
        :message="cardState(cancelCards[booking.appointment_id]).message" @confirm="emit('confirm', cancelCards[booking.appointment_id])"
        @dismiss="emit('dismiss', cancelCards[booking.appointment_id])" />
    </article>

    <h3>手动预约</h3>
    <form v-if="catalog" class="booking-form" @submit.prevent="findSlots">
      <div class="field-grid">
        <label>医院<select v-model="form.hospital_id" required><option value="" disabled>请选择</option><option v-for="hospital in catalog.hospitals" :key="hospital.id" :value="hospital.id">{{ hospital.name }}</option></select></label>
        <label>科室<select v-model="form.department_id" required :disabled="!form.hospital_id"><option value="" disabled>请选择</option><option v-for="dept in departments" :key="dept.id" :value="dept.id">{{ dept.name }}</option></select></label>
        <label>就诊人<select v-model="form.subject"><option v-for="item in catalog.subjects" :key="item.id" :value="item.id">{{ item.label }}</option></select></label>
        <label>日期<input v-model="form.day" type="date" :min="catalog.first_day" :max="catalog.last_day" required></label>
        <label>时段<select v-model="form.window"><option value="">不限</option><option value="am">上午</option><option value="pm">下午</option></select></label>
      </div>
      <button class="primary" :disabled="!form.hospital_id || !form.department_id || !form.day">查询号源</button>
    </form>
    <p v-if="slots && !slots.length" class="hint">这个时段没有号了，可以换一天或换个时段。</p>
    <div v-if="slots?.length" class="slot-list" role="radiogroup" aria-label="可约号源">
      <label v-for="slot in slots" :key="slot.slot_id"><input v-model="selected" type="radio" name="slot" :value="slot.slot_id"> {{ slot.day }} {{ windowNames[slot.window] }} · 剩 {{ slot.remaining }} 个</label>
      <button class="primary" :disabled="!selected" @click="draftCreate">生成预约草稿</button>
      <small>就诊人：{{ subjectLabel[form.subject] }}</small>
    </div>
    <AppointmentCard v-if="manualCard" :card="manualCard" :state="cardState(manualCard).state" :message="cardState(manualCard).message"
      @confirm="emit('confirm', manualCard)" @dismiss="emit('dismiss', manualCard)" />
  </div>
</template>
