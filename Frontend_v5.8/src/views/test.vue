<template>
    <el-popconfirm
      width="220"
      :icon="InfoFilled"
      icon-color="#626AEF"
      title="Are you sure to delete this?"
      @cancel="onCancel"
    >
      <template #reference>
        <el-button>Delete</el-button>
      </template>
      <template #actions="{ confirm, cancel }">
        <el-row>
            <el-col :span="12"><el-button size="small" @click="cancel">取消</el-button></el-col>
            <el-col :span="12">
                <el-button
                type="primary"
                size="small"
                @click="confirm"
                >
                确定
                </el-button>
            </el-col>
        </el-row>
      </template>
    </el-popconfirm>
  </template>
  
  <script setup lang="ts">
  import { ref } from 'vue'
  import { InfoFilled } from '@element-plus/icons-vue'
  
  const clicked = ref(false)
  function onCancel() {
    clicked.value = true
  }
  </script>