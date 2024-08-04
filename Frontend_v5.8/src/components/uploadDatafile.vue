<template>
  <div class="clearfix" style="width: 250px">
    <a-radio-group v-model:value="loadingDataModel">
      <a-radio :value="1" style="margin-right: 50px">本地文件</a-radio>
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
          <a-button style="margin-top: 16px" :disabled="loadingDataModel == 2">
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
        <a-button
          type="default"
          style="margin-top: 16px; margin-left: 2px"
          @click="switchDrawer"
          :disabled="loadingDataModel == 1"
          >查看历史文件</a-button
        >
      </a-col>
      <div>
        <a-modal
          v-model:open="uploadConfirmDialog"
          title="提交所保存文件信息"
          :confirm-loading="confirmLoading"
          @ok="handleOk"
        >
          <a-space direction="vertical">
            <a-form :model="formState" :rules="rules" ref="formRef">
              <a-form-item label="文件名" name="filename">
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
import { stringType } from "ant-design-vue/es/_util/type";
import { ElMessage } from "element-plus";


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
  ],
  description: [
    { required: true, message: "请输入文件描述", trigger: "blur" },
  ],
}
const handleOk = () => {

  formRef.value.validate().then(() => {
    confirmLoading.value = true;
  const formData = new FormData();
  //   fileList.value.forEach((file: UploadProps["fileList"][number]) => {
  //     formData.append("file", file as any);
  //   });
  formData.append("file", fileList.value[0]);
  formData.append("filename", formState.value.filename);
  formData.append("description", formState.value.description);
  uploading.value = true;

  props.api
    .post("/upload_datafile/", formData)
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

  // setTimeout(() => {
  //   uploadConfirmDialog.value = false;
  //   confirmLoading.value = false;
  // }, 2000);


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
