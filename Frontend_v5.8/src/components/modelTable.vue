<template>
  <div style="display: flex; flex-direction: column; ">
    <div class="shadow-border title-container" style="">
      用户模型管理
    </div>

    <div  class="table-container">
      <el-table :data="fetchedModelsInfo" stripe style="width: 100%; " height="500px" :stripe="true" border :header-cell-style="{ backgroundColor: '#f5f7fa', color: '#606266' }">
        <el-popover placement="bottom-start" title="模型信息" :width="400" trigger="hover" content="这是模型信息">
        </el-popover>
        <el-table-column  property="id" label="序号" />
        <el-table-column  property="model_name" label="模型名称" />
        <el-table-column  property="author" label="作者" />
        <el-table-column  property="jobNumber" label="工号" />
        
        <el-table-column  label="操作">
          <template #default="scope">

            <el-button size="small" type="danger" style="width: 50px;"
                        @click="delete_model(scope.$index, scope.row)">
              删除
            </el-button>
            <template></template>
            <el-popover placement="bottom" :width='500' trigger="click">
              <el-descriptions :title="model_name" :column="3" :size="size" direction="vertical"
                                >
                <el-descriptions-item label="使用模块" :span="3">
                  <el-tag size="small" v-for="algorithm in model_algorithms">{{ algorithm }}</el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="算法参数" :span="3">
                  <div v-for="item in model_params">{{ item.模块名 }}: {{ item.算法 }}</div>
                </el-descriptions-item>
              </el-descriptions>
              <template #reference>
                <el-button size="small" type="info" style="width: 80px" @click="show_model_info(scope.row)">
                  查看模型
                </el-button>
              </template>
            </el-popover>
          </template>
        </el-table-column>
      </el-table>
    </div>
    

    <el-dialog v-model="delete_model_confirm_visible" title="提示" width="500">
      <span style="font-size: 20px;">确定删除该模型吗？</span>
      <template #footer>
        <el-button style=" width: 150px;" @click="delete_model_confirm_visible = false">取消</el-button>
        <el-button style="width: 150px; margin-right: 70px;" type="primary"
                    @click="delete_model_confirm">确定</el-button>
      </template>
    </el-dialog>


  </div>

</template>

<script setup>
import {onMounted, ref} from "vue";
import axios from "axios";
import { ElMessage } from "element-plus";
import api from "../utils/api.js";
import {labelsForAlgorithms} from "./constant.ts";

const fetchedModelsInfo = ref([])

onMounted(() => {
  fetch_models();
});
const fetch_models = () => {
  // let url = 'http://127.0.0.1:8000/administration/fetch_users_models/'
  api.get('/administration/fetch_users_models/').then((response) => {
    let modelsInfo = response.data
    console.log('modelsInfo: ', modelsInfo)
    fetchedModelsInfo.value.length = 0
    for (let item of modelsInfo) {
      fetchedModelsInfo.value.push(item)
    }
  })
}

let index = 0
let row = 0
const delete_model_confirm_visible = ref(false)
const delete_model = (index_in, row_in) => {
  index = index_in
  row = row_in
  delete_model_confirm_visible.value = true
}
// 查看模型信息
const model_name = ref('')
const model_algorithms = ref([])
const model_params = ref([])  // {'模块名': xx, '算法': xx, '参数': xx}

const show_model_info = (row) => {
  let objects = JSON.parse(row.model_info)
  let node_list = objects.nodeList         // 模型节点信息
  let connection = objects.connection     // 模型连接顺序

  model_name.value = row.model_name
  model_algorithms.value = connection
  model_params.value.length = 0
  node_list.forEach(element => {
    let item = { '模块名': '', '算法': '' }
    item.模块名 = element.label
    item.算法 = labelsForAlgorithms[element.use_algorithm]
    model_params.value.push(item)
  });
}

const delete_model_confirm = () => {

  // 发送删除请求到后端，row 是要删除的数据行
  // let url = 'http://127.0.0.1:8000/administration/delete_user_model/?row_id=' + row.id
  api.get('/administration/delete_user_model/?row_id=' + row.id).then((response) => {
    if (response.data.message === 'delete user model success') {
      if (index !== -1) {
        // 删除前端表中数据
        fetchedModelsInfo.value.splice(index, 1)
        delete_model_confirm_visible.value = false
        ElMessage({
          message: '模型删除成功',
          type: 'success',
        })
      }
    }
  }).catch(error => {
    // 处理错误，例如显示一个错误消息
    ElMessage({
      message: '删除失败，请稍后重试',
      type: 'error',
    })
    console.error(error);
  });

}
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