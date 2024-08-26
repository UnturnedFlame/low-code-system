<template>
  <div class="clearfix" style="width: 250px">
    <a-radio-group v-model:value="loadingDataModel">
      <a-radio :value="1" style="margin-right: 50px">本地文件</a-radio>
      <a-radio :value="2">服务器文件</a-radio>
    </a-radio-group>
    <a-row>
      <a-col :span="24" v-if="loadingDataModel == 1">
        <a-upload
          :file-list="fileList"
          :max-count="1"
          @remove="handleRemove"
          :before-upload="beforeUpload"
        >
          <a-button style="margin-top: 16px; width: 160px; font-size: 16px; background-color: #2082F9; color: white"
          :icon="h(FolderOpenOutlined)">
            选择本地文件
          </a-button>
        </a-upload>
        <a-button
          type="primary"
          :disabled="fileList.length === 0"
          :loading="uploading"
          style="margin-top: 5px; width: 160px; font-size: 16px"
          @click="handleUpload"
        >
          <UploadOutlined />
          {{ uploading ? "正在上传" : "上传至服务器" }}
        </a-button>
      </a-col>
      <a-col :span="24" v-if="loadingDataModel == 2">
        <a-button
          type="default"
          style="margin-top: 35px; margin-left: 2pxs; width: 160px; font-size: 16px;  background-color: #2082F9; color: white"
          @click="switchDrawer"
          :icon="h(FolderOutlined)"
          >查看历史文件</a-button
        >
      </a-col>
      <div>
        <a-modal
          v-model:open="uploadConfirmDialog"
          title="提交所保存文件信息"
          :confirm-loading="confirmLoading"
          @ok="handleOk"
          okText="确定"
          cancelText="取消"
          :maskClosable="false"
        >
          <a-space direction="vertical">
            <a-form :model="formState" :rules="rules" ref="formRef">
              <a-form-item label="文件名称" name="filename">
                <a-input v-model:value="formState.filename" placeholder="请输入文件名" />
              </a-form-item>
              <a-form-item label="文件描述" name="description">
                <a-input
                  v-model:value="formState.description"
                  autofocus
                  placeholder="请输入文件描述"
                />
              </a-form-item>
            </a-form>
          </a-space>
        </a-modal>
      </div>
    </a-row>
  </div>
</template>
<script lang="ts" setup>
import { ref } from "vue";
import { UploadOutlined } from "@ant-design/icons-vue";
import { message } from "ant-design-vue";
import type { UploadProps } from "ant-design-vue";
import { Rule } from "ant-design-vue/es/form";
import { ElMessage } from "element-plus";
import { h } from 'vue';
import { FolderOutlined, FolderOpenOutlined } from '@ant-design/icons-vue';


const confirmLoading = ref<boolean>(false);
const uploadConfirmDialog = ref<boolean>(false);

const formState = ref({
  filename: "",
  description: "",
});
const formRef = ref();
const rules: Record<string, Rule[]> = {
  filename: [
    { required: true, message: "请输入文件名", trigger: "blur" },
    // { pattern: /[<>:"\/\\|?*]/, message: '文件名中包含非法字符', trigger: 'blur' }
    { pattern:/^[\u4e00-\u9fa5_a-zA-Z0-9]+$/, message: '请输入中英文/数字/下划线', trigger: 'blur' },
  ],
  description: [
    { required: true, message: "请输入文件描述", trigger: "blur" },
  ],
}


// 确认上传文件
const handleOk = () => {

  formRef.value.validate().then(() => {
  confirmLoading.value = true;
  const formData = new FormData();
  formData.append("file", fileList.value[0]);
  formData.append("filename", formState.value.filename);
  formData.append("description", formState.value.description);
  uploading.value = true;

  props.api
    .post("/user/upload_datafile/", formData)
    .then((response: any) => {
        if(response.data.message == 'save data success'){
            fileList.value = [];
            uploading.value = false;
            message.success("数据文件上传成功");

            confirmLoading.value = false;
            uploadConfirmDialog.value = false;
        }else{
            uploading.value = false;
            message.error("文件上传失败, "+response.data.message);
            confirmLoading.value = false;
        }

    })
    .catch((error:any) => {
      uploading.value = false;
      confirmLoading.value = false;
      message.error("上传失败, "+error);
    });
  })
}



const fileList = ref<UploadProps["fileList"]>([]);  // 文件列表
const uploading = ref<boolean>(false);
const loadingDataModel = ref<number>(1)
const props = defineProps({
  api: {
    type: Object,
    required: true,
  },
});


// 移除文件列表中的文件
const handleRemove: UploadProps["onRemove"] = (file) => {
  const index = fileList.value.indexOf(file);
  const newFileList = fileList.value.slice();
  newFileList.splice(index, 1);
  fileList.value = newFileList;
};


const beforeUpload: UploadProps["beforeUpload"] = (file) => {
  fileList.value.length = 0;
  fileList.value = [...(fileList.value || []), file];
  return false;
};


// 文件类型检查，只允许mat或是npy格式的文件
const handleUpload = () => {
  let file = fileList.value[0]
  let filename = file.name
  const ext = filename.split('.').pop().toLowerCase();  
  if (ext != 'mat' && ext != 'npy') {
    ElMessage({
      message: '文件格式错误，请上传mat或npy文件',
      type: 'error',
    })
    return  
  }
  uploadConfirmDialog.value = true

};


// 子组件向父组件发送数据
const emit = defineEmits(["switchDrawer"]);
const switchDrawer = () => {
    let url = 'user/fetch_datafiles/'
    let fetchedDatasetsInfo: Object[] = []
    props.api.get(url)
    .then((response: any) => {
      let datasetInfo = response.data

      for (let item of datasetInfo){
        fetchedDatasetsInfo.push(item)
      }
      emit("switchDrawer", fetchedDatasetsInfo);
    })

};
</script>
