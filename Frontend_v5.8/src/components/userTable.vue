<template>
  <div class="roleManage">
    <div class="tableTool" style="position:relative; right: 5px;">
      <el-button @click="goToAddUserPage" style="width: 60px; border: none;">
        新增用户
      </el-button>
    </div>
  </div>

  <el-table :data="tableData" style="width: 100%;" >
    <el-table-column prop="id" label="ID" />
    <el-table-column prop="jobNumber" label="工号"  />
    <el-table-column prop="username" label="姓名"  />
    <el-table-column prop="password" label="密码" />
    <el-table-column prop="email" label="邮箱"  />
    <el-table-column label="操作" >
      <template #default="scope" >
        <div  style="display: flex; justify-content: space-between;">
          <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row)" style="width: 100px;">
            删除
          </el-button>
          <el-button size="small" type="primary" @click="handleResetPassword(scope.$index, scope.row)" style="width: 100px;">
            重置密码
          </el-button>
        </div>

      </template>
    </el-table-column>
  </el-table>

</template>

<script setup>
import {onMounted, reactive, ref} from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import api from "../utils/api.js"
const router = useRouter();


//模拟数据，对接时把tableData替换成真实数据
const tableData = ref([
  {
    id: 1,
    username: '张三',
    jobNumber: '78239',
    mailbox: '341763799@qq.com',
    password: '123456'
  },
  {
    id: 2,
    name: '李四',
    password: 'abcdef',
    jobNumber: '782',
    mailbox: '341763666@qq.com',
  },
  // 更多行数据...
]);

const handleDelete = (index, row) => {
  console.log('删除行数据：', index, row);
  // 向服务器发送删除请求
  api.get('/delete_user/?jobNumber=' + row.jobNumber)
      .then(response => {
        if (response.data.message === 'user deleted success'){
          console.log('删除成功：', response.data);
          
          // 从前端数据中删除该行
          tableData.value.splice(index, 1);
          ElMessage({
            message: '用户删除成功',
            type: 'success',
          })
        }else{
          console.log('删除失败：', response.data.message);
        }
        
      })
      .catch(error => {
        console.error('删除失败：', error);
        // 显示错误提示
        alert('删除失败，请稍后重试');
      });
};

const handleResetPassword = (index, row) => {
  // 向服务器发送删除请求
  api.get('/administration/reset_user_password/?jobNumber=' + row.jobNumber)
      .then(response => {
        if (response.data.code === 200){
          console.log('重置成功：', response.data);

          // 刷新展示数据
          fetchTableData();
          ElMessage({
            showClose: true,
            message: response.data.message,
            type: 'success',
            size: 'large',
          })
        }else{
          console.log('重置失败：', response.data.message);
        }

      })
      .catch(error => {
        console.error('重置失败：', error);
        // 显示错误提示
        alert('重置失败，请稍后重试');
      });
};
const goToAddUserPage =() =>{
  router.push('/admin/addUser');
}
const fetchTableData = async () => {
  try {
    // 发起 GET 请求获取数据
    const response = await api.get('/admin_fetch_users/');
    // 将响应数据赋值给 tableData
    tableData.value = response.data;
  } catch (error) {
    // 错误处理，例如显示一个错误消息
    console.error('Failed to fetch table data:', error);
    // 这里可以添加更多的错误处理逻辑，如用户提示等
  }
};

// 组件挂载后获取数据
onMounted(() => {
  fetchTableData();
});
</script>

<style>
.roleManage {
  .tableTool {
    padding: 10px 0;
    display: flex;
    justify-content: flex-end;
    align-items: center;
  }
  :deep(.el-table thead .el-table__cell) {
    text-align: center;
  }
}

.button-container {

  display: flex;
  justify-content: flex-end; /* 可以根据需要调整对齐方式 */
  flex-direction:row;
  align-items: center;
  gap: 10px; /* 可以根据需要调整间隔大小 */
}
.buttons-right-bottom {
  display: flex;
  justify-content: flex-end; /* 可以根据需要调整对齐方式 */
  flex-direction:row;
  align-items: center;
  gap: 10px; /* 可以根据需要调整间隔大小 */
}
.button-container button {
  margin-left: 10px; /* 给按钮之间添加间隔 */
}
</style>