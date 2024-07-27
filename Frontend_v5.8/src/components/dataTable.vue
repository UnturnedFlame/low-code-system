<template>
  <el-table :data="tableData" style="width: 100%;" >
    <el-table-column prop="dataNumber" label="文件ID"  />
    <el-table-column prop="dataName" label="文件名"  />
    <el-table-column prop="dataDetail" label="文件描述" />
    <el-table-column label="操作" >
      <template #default="scope">
        <el-button size="small" type="danger" @click="handleDelete(scope.$index, scope.row)" style="width: 100px;">
          删除
        </el-button>
      </template>
    </el-table-column>
  </el-table>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import axios from 'axios';
import { useRouter } from 'vue-router';
const router = useRouter();


// 初始化 tableData 为 ref，且初始值为空数组
const tableData = ref([]);

const fetchTableData = async () => {
  try {
    // 发起 GET 请求获取数据
    const response = await axios.get('/api/users');
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
  if (!window.confirm('确定要删除这条记录吗?')) { // 确认删除
    return;
  }
  try {
    // 发起 DELETE 请求删除数据
    await axios.delete(`/api/data/${row.dataNumber}`); // 假设后端 API 使用文件ID作为路径参数
    // 从前端数据中删除对应的行
    tableData.value.splice(index, 1);
  } catch (error) {
    // 错误处理
    console.error('Failed to delete data:', error);
    // 可以在这里添加一些用户提示，例如:
    alert('删除失败，请稍后重试');
  }
};
</script>


<style scoped>

</style>