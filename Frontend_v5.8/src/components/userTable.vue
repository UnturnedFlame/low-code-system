<template>
 
  <div class="shadow-border title-container">用户账号管理</div>
  <div class="roleManage">
    <div class="tableTool" style="position:relative; right: 150px;">
      <el-button @click="dialogFormVisible = true" style="width: 120px; border: none;" icon="Plus" color="green">
        新增用户
      </el-button>
    </div>
  </div>
  <div class="table-container">
    <el-table :data="tableData" style="width: 100%;" height="500px" :stripe="true" :header-cell-style="{ backgroundColor: '#f5f7fa', color: '#606266' }" border>
      <el-table-column prop="id" label="ID" />
      <el-table-column prop="jobNumber" label="工号"  />
      <el-table-column prop="username" label="姓名"  />
      <!-- <el-table-column prop="password" label="密码" /> -->
      <el-table-column prop="email" label="邮箱"  />
      <el-table-column label="操作" >
        <template #default="scope" >
          <div  style="display: flex; justify-content: space-between;">
            <el-popconfirm title="你确定要删除该用户吗?" @confirm="handleDelete(scope.$index, scope.row)" width="100px">
              <template #reference>
                <el-button size="small" type="danger" style="width: 100px;">删除</el-button>
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
            <!-- <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row)" style="width: 100px;">
              删除
            </el-button> -->
            <el-button size="small" type="primary" @click="handleResetPassword(scope.$index, scope.row)" style="width: 100px;">
              重置密码
            </el-button>
          </div>

        </template>
      </el-table-column>
    </el-table>
  </div>

  <el-dialog v-model="dialogFormVisible" title="新增用户" width="500" @close="() => {form.username = ''; form.password = ''}"
             :show-close="false" :close-on-click-modal="false" center >
    <el-form :model="form" :rules="rules" ref="formRef" style="margin-right: 70px">
      <el-form-item prop="jobNumber" label="工号" :label-width="formLabelWidth">
        <el-input v-model="form.jobNumber" autocomplete="off" />
      </el-form-item>
      <el-form-item prop="username" label="用户名" :label-width="formLabelWidth">
        <el-input v-model="form.username" autocomplete="off" />
      </el-form-item>
      <el-form-item prop="password" label="密码" :label-width="formLabelWidth">
        <el-input v-model="form.password" autocomplete="off" placeholder="请输入八位数且包含数字、大小写字母">
        </el-input>
      </el-form-item>
      <el-form-item prop="email" label="邮箱" :label-width="formLabelWidth">
        <el-input v-model="form.email" autocomplete="off" placeholder="请输入邮箱">
        </el-input>
      </el-form-item>
      <el-form-item prop="role" label="角色权限" :label-width="formLabelWidth">
        <el-radio-group v-model="form.role" style="padding-left: 10px">
          <el-radio label="user" size="large">用户</el-radio>
          <el-radio label="admin" size="large">管理员</el-radio>
        </el-radio-group>
      </el-form-item>
      
    </el-form>

    <template #footer>
      <div class="button-container" >
        <div class="buttons-right-bottom">
          <el-button @click="handlecancel(formRef)">取消</el-button>
          <el-button type="primary" @click="handleConfirm()">确定</el-button>
        </div>
      </div>

    </template>
  </el-dialog>


</template>

<script setup>
import {onMounted, reactive, ref} from 'vue';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import api from "../utils/api.js"
const router = useRouter();


//模拟数据，对接时把tableData替换成真实数据
const tableData = ref([]);

const handleDelete = (index, row) => {
  // 向服务器发送删除请求
  api.get('administration/delete_user/?jobNumber=' + row.jobNumber)
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
  // 向服务器发送重置用户密码的请求
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
    const response = await api.get('/administration/fetch_users_info/');
    // 将响应数据赋值给 tableData
    tableData.value = response.data;
  } catch (error) {
    // 错误处理，例如显示一个错误消息
    console.error('Failed to fetch table data:', error);
    // 这里可以添加更多的错误处理逻辑，如用户提示等
  }
};


// 添加用户操作
const formRef = ref(null)   
const formLabelWidth = '140px'
const dialogFormVisible = ref(false);

// 用户信息表单
const form = reactive({
  username: '',
  password: '',
  role:'user',
  jobNumber: '',
  email: ''
})

// 添加用户的信息表单校验规则
const rules = {
  username: [
    { required: true, message: '用户姓名不能为空', trigger: 'blur' },
    { min: 3, max: 10, message: '用户姓名长度必须在3到10个字符之间', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '密码不能为空', trigger: 'blur' },
    { min: 8, message: '密码长度必须至少为8位', trigger: 'blur' },
    { pattern: /^(?=.*[A-Z])(?=.*[a-z])(?=.*\d).+$/, message: '密码必须包含数字、大小写字母', trigger: 'blur' }
  ],
  jobNumber: [
    { required: true, message: '工号不能为空', trigger: 'blur' },
    { min: 5, max: 10, message: '工号长度必须在5到10个字符之间', trigger: 'blur' },
    // 可以添加正则表达式来验证工号的格式，例如只允许数字
    { pattern: /^\d{5,10}$/, message: '工号必须为5到10位数字', trigger: 'blur' }
  ],
  email: [
    { required: true, message: '邮箱不能为空', trigger: 'blur' },
    // 使用正则表达式来验证邮箱格式
    { pattern: /^[A-Za-z0-9\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$/, message: '邮箱格式不正确', trigger: 'blur' }
  ]
}


const handlecancel = (formEl) => {
  if (!formEl) return
  formEl.resetFields()
  dialogFormVisible.value = false
  router.push('/admin/userManage');
}


// 确认添加新用户
const handleConfirm = async () => {
  try {
    // 触发表单验证
    const isFormValid = await formRef.value.validate();
    if (!isFormValid) {
      console.error('表单验证失败');
      return;
    }
    else {
      let json_form = JSON.stringify(form);
      api.post('/administration/add_user/', json_form)
      .then(response => {
        if (response.data.message === 'add user success') {
          ElMessage({
            message: '用户创建成功',
            type: 'success',
          })
          // 重置表单和关闭对话框
          formRef.value.resetFields();
          dialogFormVisible.value = false;
          // 可能需要重新获取表格数据或其他逻辑
          // router.push('/admin/userManage');
          fetchTableData()
        }else {
          ElMessage({
            message: '用户创建失败,' + response.data.message,
            type: 'error',
          })
        }
    })
    .catch(error => {
      console.error('请求错误：', error);
      // 显示错误消息
      ElMessage.error('服务器出错，请稍微重试');
    })
  }
  } catch (error) {
    console.error('创建用户失败', error);
    // 显示错误提示
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
  justify-content: flex-end; 
  flex-direction:row;
  align-items: center;
  gap: 10px; 
}
.buttons-right-bottom {
  display: flex;
  justify-content: flex-end; 
  flex-direction:row;
  align-items: center;
  gap: 10px; 
}
.button-container button {
  margin-left: 10px; 
}

.shadow-border {
  width: 200px;
  height: 200px;
  /* border: 1px solid #888; */
  box-shadow: 4px 4px 8px 0 rgba(136, 136, 136, 0.5); /* 水平偏移, 垂直偏移, 模糊距离, 扩展距离, 颜色 */
}

.title-container {
  display: flex;        
  align-items: center;  
  justify-content: flex-start;
  background-color: white;
  width: 89%; 
  height: 100px; 
  font-weight: 8px; 
  font-size: 30px; 
  margin-bottom: 10px;
  margin-left: 10px;
  margin-top: 20px;
  padding-left: 20px;
  border-radius: 5px;
  font-family: '微软雅黑', sans-serif;
}

.table-container {
  width: 86%;
  height: 510px;
  padding: 20px;
  background-color: white;
  border-radius: 5px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
  margin-left: 30px;
}
</style>