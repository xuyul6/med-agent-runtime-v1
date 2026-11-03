<script setup>
defineProps({ card: Object, state: { type: String, default: 'pending' }, message: { type: String, default: '' } })
defineEmits(['confirm', 'dismiss'])
</script>

<template>
  <section v-if="state !== 'dismissed'" class="appointment-card" :aria-label="card.title">
    <h4>{{ card.title }}</h4>
    <p class="appointment-summary">{{ card.summary }}</p>
    <p class="appointment-note">模拟医院系统 · 草稿有效至 {{ card.expiresAt?.slice(11, 16) }}，点击确认后才会提交</p>
    <p v-if="state === 'done'" class="appointment-done" role="status">已提交到模拟医院系统</p>
    <p v-else-if="state === 'error'" class="degraded" role="status">{{ message }}</p>
    <div v-if="state === 'pending' || state === 'confirming' || state === 'error'" class="appointment-actions">
      <button class="primary" :disabled="state === 'confirming'" @click="$emit('confirm')">{{ state === 'confirming' ? '提交中…' : '确认' }}</button>
      <button class="quiet" :disabled="state === 'confirming'" @click="$emit('dismiss')">放弃</button>
    </div>
  </section>
</template>
