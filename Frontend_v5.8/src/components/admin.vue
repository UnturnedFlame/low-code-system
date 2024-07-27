<template>
  <!-- 外层容器 -->
  <div style="display: flex; width: 100%; height: 100%;">
    <!-- 左侧导航栏 -->
    <div style=" height: 50px; background-color: lightgray;" :width="width" >
      <div class="text-header">
        <h3>后台管理</h3>
      </div>
      <div class="user-info-container" id="userInfo" style="position: absolute; right: 50px; top: 20px; color: white;">
          <span style="margin-right: 10px;">欢迎！ {{ username }}</span>
          <span @click="logout" class="clickable">退出登录</span>
      </div>
      <!-- 使用v-for遍历menus数组渲染菜单项 -->
      <el-menu class="el-menu" :collapse="isCollapse" :collapse-transition="false">
        <el-menu-item v-for="menu in menus" :key="menu.key" @click="handleMenuClick(menu)">
          <el-icon><component :is="menu.icon" /></el-icon>
          <span>{{ menu.label }}</span>
        </el-menu-item>
      </el-menu>
    </div>

    <!-- 右侧区域 -->
    <div class="right">
      <!-- 右侧上部标题 -->
      <div style="height: 50px; display: flex; align-items: center; ">
        <div class="left-content">
         <el-button icon="menu" style="width: 20px;" @click="handleCollapse()"></el-button>
          <el-breadcrumb separator="/" class="navigation" >
            <el-breadcrumb-item :to="{ path: '/admin' }" >首页</el-breadcrumb-item>
            <el-breadcrumb-item v-if="current" :to="current.path">
              {{current.label}}
            </el-breadcrumb-item>
          </el-breadcrumb>
        </div>
        <div class="right-content">
          <el-dropdown>
              <span class="el-dropdown-link">
                <img src="../assets/logo.png" style="width: 40px; height: 40px; border-radius: 50%;"/>
              </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>个人中心</el-dropdown-item>
                <el-dropdown-item>退出</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
      </div>

      <!-- 右侧下部内容区域 -->
      <div style="flex-grow: 1; background-color: white;">
        <router-view></router-view>
      </div>
    </div>
  </div>
</template>

<script setup>
import {computed, ref} from "vue";
import {useAllDataStore} from '../stores'
import {useRouter} from "vue-router";
import addUser from "./addUser.vue";

const store = useAllDataStore()

const username = window.localStorage.getItem('username')

const isCollapse = computed(() =>store.state.isCollapse)
const width = computed(() =>store.state.isCollapse ? '54px' : '150px')


const router = useRouter();
//处理菜单收缩
const handleCollapse =() => {
  store.state.isCollapse = !store.state.isCollapse
}
// 存储组件引用的字典


// 定义菜单数组
const menus = ref([
  { key: 'Usermanage', label: '用户管理', icon: 'UserFilled', path: '/admin/userManage' },
  { key: 'Modelmanage', label: '模型管理', icon: 'OfficeBuilding', path: '/admin/modelManage' },
  { key: 'dataManage', label: '数据管理', icon: 'UploadFilled', path: '/admin/dataManage' }
]);

// 活动索引
const current = computed(() =>store.state.currentMenu)
// 菜单点击处理函数
const handleMenuClick = (menu) => {

  const path = menu.path;
  store.state.currentMenu = menu;
  // 使用 Vue Router 的 router 实例进行跳转
  router.push({ path }); // 使用路径进行跳转
};

const logout = () => {
  router.push('/')
}
</script>


<style scoped>
.left-content {
  display: flex;
  flex-direction:row;
  float: left; /* 使元素靠左对齐 */
  margin-left: 10px;
  margin-right: 10px;
  .navigation{
    display: flex;
    margin-left: 10px;
    margin-right: 10px;
    align-items: center;
    justify-content: center;
  }
}
.right {
  flex-grow: 1;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: #4ca1c1;
  justify-content: space-between;
  overflow: auto; /* 清除浮动 */
}
.right-content {
  float: right;
  margin-left: auto; /* 将元素推向右侧 */
  margin-right: 10px;
}
.text-header {
  display: flex;
  height: 50px;
  text-align: center;
  margin: 2px;
  align-items: center;
  justify-content: center;
  font-size: 13px;

}

.el-menu {
  border-right: none;

}

.clickable:hover {
  cursor: pointer;
  color: #007BFF;
}
</style>