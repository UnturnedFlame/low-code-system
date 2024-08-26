<template>
  <div style="display: flex; width: 100%; height: 100vh;">
    <!-- 左侧导航栏 -->
    <div style="background-color: #262a2d; width: 200px; height: 100%;">
      <div class="left-header">
        <!-- <h3>后台管理</h3> -->
        <img src="../assets/system-logo.png" alt="" style="width: 100px; height: auto; margin-right: 80px; color: white">
      </div>
      
      <!-- 使用v-for遍历menus数组渲染菜单项 -->
      <div class="menu-container">
        <el-menu class="el-menu" :collapse="isCollapse" :collapse-transition="false">
          <el-menu-item v-for="menu in menus" :key="menu.key" @click="handleMenuClick(menu)">
            <el-icon><component :is="menu.icon" /></el-icon>
            <span style="font-size: 18px;">{{ menu.label }}</span>
          </el-menu-item>
        </el-menu>
      </div>
      
    </div>

    <!-- 右侧区域 -->
    <div class="right">
      <!-- 右侧上部标题 -->
      <div class="right-header">
        <div class="left-content">
         <!-- <el-button icon="menu" style="width: 20px;" @click="handleCollapse()"></el-button> -->
          <el-breadcrumb separator="/" class="navigation" >
            <!-- <el-breadcrumb-item :to="{ path: '/admin' }" >首页</el-breadcrumb-item> -->
            <!-- <el-breadcrumb-item v-if="current" :to="current.path">
              {{current.label}}
            </el-breadcrumb-item> -->
          </el-breadcrumb>
        </div>
        <div class="right-content">
          <el-dropdown>
              <span class="el-dropdown-link">
                <!-- <img src="../assets/system-logo.png" style="width: 40px; height: auto;"/> -->
              </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item>个人中心</el-dropdown-item>
                <el-dropdown-item>退出</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
        <div class="user-info-container" id="userInfo" style="position: absolute; right: 50px; top: 20px; color: black;">
          <span style="margin-right: 10px;">欢迎！ {{ username }}</span>
          <span @click="logout" class="clickable">退出登录</span>
        </div>
      </div>

      <!-- 右侧下部内容区域 -->
      <div style="flex-grow: 1; background-color: #f3f3f3; height: 100vh;">
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
const width = computed(() =>store.state.isCollapse ? '70px' : '150px')


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
  /* .navigation{
    display: flex;
    margin-left: 10px;
    margin-right: 10px;
    align-items: center;
    justify-content: center;
  } */
  
}

.right-header {
  display: flex;
  height: 40px;
  width: auto;
  justify-content: space-between;
  align-items: center;
  padding: 10px;
  background-color: white;
  border: 1px solid #e7e7e7; /* 添加淡淡的浅灰色边框 */
}

.right {
  flex-grow: 1;
  height: 100%;
  display: flex;
  flex-direction: column;
  background-color: white;
  justify-content: space-between;
  overflow: auto; /* 清除浮动 */
}
.right-content {
  float: right;
  margin-left: auto; /* 将元素推向右侧 */
  margin-right: 10px;
}
.left-header {
  display: flex;
  height: 60px;
  text-align: center;
  align-items: center;
  justify-content: center;
  background-color: white;
  font-size: 18px;
}

.el-menu {
  border-right: none;
  width: 200px;
  height: 800px;
  padding-top: 20px;
  background-color: #2e3439;

}
.clickable:hover {
  cursor: pointer;
  color: #007BFF;
}


.el-menu .el-menu-item,
.el-menu .el-submenu__title {
  font-family: 'Arial', sans-serif; /* 更改字体 */
  font-size: 14px; /* 设置字体大小 */
  line-height: 24px; /* 设置行高 */
  color: #faf7f7; /* 设置字体颜色 */
}

.el-menu-item.is-active {
  background-color: #c4bdbd; /* 激活状态下的背景颜色 */

}

.el-menu-item:hover,
.el-submenu__title:hover {
  background-color: #ccc; /* 鼠标悬停时的背景颜色 */
}

</style>