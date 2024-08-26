<template>
  <div class="shadow-border title-container" style="">
    用户数据管理
  </div>

  <div class="table-container">
    <el-table :data="tableData" height="500px" :stripe="true" border :header-cell-style="{ backgroundColor: '#f5f7fa', color: '#606266' }">
      <el-table-column prop="id" label="文件ID"  />
      <el-table-column prop="dataset_name" label="文件名"  />
      <el-table-column prop="description" label="文件描述" />
      <el-table-column prop="owner" label="所有者" />
      <el-table-column label="操作" >
        <template #default="scope">
          <!-- <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row)" style="width: 100px;">
            删除
          </el-button> -->
          <el-popconfirm title="你确定要删除该用户数据吗?" @confirm="handleDelete(scope.$index, scope.row)" width="200px"
          >
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
        </template>
      </el-table-column>
    </el-table>
  </div>
  
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { useRouter } from 'vue-router';
import { ElMessage } from 'element-plus';
import api from '../utils/api.js';


const router = useRouter();


// 初始化 tableData 为 ref，且初始值为空数组
const tableData = ref([]);


const fetchTableData = async () => {
  try {
    // 发起 GET 请求获取数据
    const response = await api.get('administration/fetch_users_datafiles/');
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

// 处理删除操作的函数
const handleDelete = async (index, row) => {
  // if (!window.confirm('确定要删除这条记录吗?')) { // 确认删除
  //   return;
  // }
  try {
    api.get('administration/delete_user_datafile/?datafile_id=' + row.id)
    .then(response=>{
      if(response.data.code == 200){
        ElMessage({
          message: response.data.message,
          type: 'success'
        })
        tableData.value.splice(index, 1);
      }else{
        ElMessage({
          message: '文件删除失败,'+response.data.message,
          type: 'error'
        })
      }
    })
    
  } catch (error) {
    // 错误处理
    console.error('Failed to delete data:', error);
    // 可以在这里添加一些用户提示，例如:
    alert('删除失败，请稍后重试');
  }
};
</script>


<style scoped>
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