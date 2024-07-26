<template>
  <div class="clearfix" style="width: 250px">
    <a-radio-group v-model:value="loadingDataModel">
      <a-radio :value="1" style="margin-right: 50px;">本地文件</a-radio>
      <a-radio :value="2">服务器文件</a-radio>
    </a-radio-group>
    <a-row>
      <a-col :span="12">
        <a-upload
          :file-list="fileList"
          :max-count="1"
          @remove="handleRemove"
          :before-upload="beforeUpload"
        >
          <a-button style="margin-top: 16px;" :disabled="loadingDataModel == 2">
            <upload-outlined></upload-outlined>
            选择文件
          </a-button>
        </a-upload>
        <a-button
          type="primary"
          :disabled="fileList.length === 0 || loadingDataModel == 2"
          :loading="uploading"
          style="margin-top: 20px"
          @click="handleUpload"
          
        >
          {{ uploading ? "正在上传" : "上传至服务器" }}
        </a-button>
      </a-col>
      <a-col :span="12">
        
        
        <a-button type="default" style="margin-top: 16px; margin-left: 2px" @click="switchDrawer" :disabled="loadingDataModel == 1"
          >查看历史文件</a-button
        >
      </a-col>
      <div>
       
        <a-modal v-model:open="uploadConfirmDialog" title="提交所保存文件信息" :confirm-loading="confirmLoading" @ok="handleOk">
          <a-space direction="vertical">
            <a-input v-model:value="filename" placeholder="请输入文件名" />
            <a-input v-model:value="description" autofocus placeholder="请输入文件描述" />
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
// import { defineEmits } from "vue";
// import { defineProps } from "vue";



const confirmLoading = ref<boolean>(false);
const uploadConfirmDialog = ref<boolean>(false);
const filename = ref<string>("");
const description = ref<string>("");
const handleOk = () => {
  
  confirmLoading.value = true;
  const formData = new FormData();
  //   fileList.value.forEach((file: UploadProps["fileList"][number]) => {
  //     formData.append("file", file as any);
  //   });
  formData.append("file", fileList.value[0]);
  formData.append("filename", filename.value);
  formData.append("description", description.value);
  uploading.value = true;

  props.api
    .post("http://localhost:8000/upload_datafile/", formData)
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
  // setTimeout(() => {
  //   uploadConfirmDialog.value = false;
  //   confirmLoading.value = false;
  // }, 2000);
};

const fileList = ref<UploadProps["fileList"]>([]);
const uploading = ref<boolean>(false);
const loadingDataModel = ref<number>(1)
const props = defineProps({
  api: {
    type: Object,
    required: true,
  },
});

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

const handleUpload = () => {

  uploadConfirmDialog.value = true

};

const emit = defineEmits(["switchDrawer"]);
const switchDrawer = () => {
    let url = 'http://127.0.0.1:8000/user_fetch_datafiles/'
    let fetchedDatasetsInfo: any[] = []
    props.api.request({
      method: 'GET',
      url: url
    })
    .then((response: any) => {
      let datasetInfo = response.data
      fetchedDatasetsInfo.length = 0
      for (let item of datasetInfo){
        fetchedDatasetsInfo.push(item)
      }
    })
    emit("switchDrawer", fetchedDatasetsInfo);
};
</script>
