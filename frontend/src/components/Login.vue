<script setup>
import { ref } from 'vue'
import { login as apiLogin } from '@/api.js'

const emit = defineEmits(['success'])

const username = ref('admin')
const password = ref('')
const error = ref('')
const loading = ref(false)

async function onSubmit() {
  error.value = ''
  const u = username.value.trim()
  const p = password.value
  if (!u || !p) {
    error.value = '请输入用户名和密码'
    return
  }
  loading.value = true
  try {
    const { ok, data } = await apiLogin(u, p)
    if (!ok) {
      error.value = data.error || '登录失败，请稍后重试'
      return
    }
    emit('success', { username: data.username || u })
  } catch {
    error.value = '无法连接服务器，请确认后端已启动'
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-brand">
        <i class="fas fa-house-crack" aria-hidden="true"></i>
        <h1>地震应急智能问答</h1>
        <p>请登录后继续使用系统</p>
      </div>
      <form class="login-form" @submit.prevent="onSubmit">
        <label class="login-label">
          用户名
          <input
            v-model="username"
            class="login-input"
            type="text"
            autocomplete="username"
            placeholder="admin"
            :disabled="loading"
          />
        </label>
        <label class="login-label">
          密码
          <input
            v-model="password"
            class="login-input"
            type="password"
            autocomplete="current-password"
            placeholder="请输入密码"
            :disabled="loading"
          />
        </label>
        <p v-if="error" class="login-error" role="alert">
          <i class="fas fa-exclamation-circle me-1"></i>{{ error }}
        </p>
        <button class="login-btn" type="submit" :disabled="loading">
          <i v-if="loading" class="fas fa-circle-notch fa-spin me-1"></i>
          {{ loading ? '登录中…' : '登录' }}
        </button>
      </form>
    </div>
  </div>
</template>
