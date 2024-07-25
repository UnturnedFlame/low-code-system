<template>
  <div class="clearfix" style="width: 250px">
    <a-radio-group v-model:value="loadingDataModel">
      <a-radio :value="1">本地文件</a-radio>
      <a-radio :value="2">服务器文件</a-radio>
    </a-radio-group>
    <a-row :gutter="16">
      <a-col :span="12">
        <a-upload
          :file-list="fileList"
          :max-count="1"
          @remove="handleRemove"
          :before-upload="beforeUpload"
        >
          <a-button>
            <upload-outlined></upload-outlined>
            选择文件
          </a-button>
        </a-upload>
        <a-button
          type="primary"
          :disabled="fileList.length === 0 || loadingDataModel == 2"
          :loading="uploading"
          style="margin-top: 16px"
          @click="handleUpload"
          
        >
          {{ uploading ? "Uploading" : "上传至服务器" }}
        </a-button>
      </a-col>
      <a-col :span="12">
        
        
        <a-button type="default" style="margin-top: 16px; margin-left: 8px" @click="switchDrawer" :disabled="loadingDataModel == 1"
          >查看历史文件</a-button
        >
      </a-col>
    </a-row>
  </div>
</template>
<script lang="ts" setup>
import { ref } from "vue";
import { UploadOutlined } from "@ant-design/icons-vue";
import { message } from "ant-design-vue";
import type { UploadProps } from "ant-design-vue";
import { defineEmits } from "vue";
import { defineProps } from "vue";

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
  const formData = new FormData();
  //   fileList.value.forEach((file: UploadProps["fileList"][number]) => {
  //     formData.append("file", file as any);
  //   });
  formData.append("file", fileList.value[0]);
  uploading.value = true;

  props.api
    .post("http://localhost:8000/upload_datafile", JSON.stringify(formData))
    .then((response: any) => {
        if(response.data.message == 'save data success'){
            fileList.value = [];
            uploading.value = false;
            message.success("数据文件上传成功");
        }else{
            message.error("文件上传失败, "+response.data.message);
        }
      
    })
    .catch((error:any) => {
      uploading.value = false;
      message.error("上传失败, "+error);
    });
};

let drawerState = false;
const emit = defineEmits(["switchDrawer"]);
const switchDrawer = () => {
    drawerState = !drawerState;
    emit("switchDrawer", drawerState);
};
</script>
