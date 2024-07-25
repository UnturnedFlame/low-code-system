<template>
  <el-upload ref="uploadRef" class="upload-demo" action="" v-model:file-list="uploads.file_list" :auto-upload="false"
             :on-change="onChangeHandler">
    <template #trigger>
      <el-button type="primary">选择文件</el-button>
    </template>

    <span style="margin: 10px;">
            <el-button class="ml-3" type="success" @click="submitUpload">
                点击上传
            </el-button>
        </span>


    <template #tip>
      <div class="el-upload__tip">
        请上传wav/audio类型的音频文件
      </div>
    </template>
  </el-upload>
</template>
<script lang="ts" setup>
import { ref, reactive } from 'vue'
import { ElMessage, ElUpload, ElButton, ElMessageBox } from 'element-plus'
import type { UploadInstance } from 'element-plus'
import axios from 'axios'

const uploads = reactive<any>({ file_list: [] })


const uploadRef = ref<UploadInstance>()



const submitUpload = () => {

  // 发送文件到后端
  if (uploads.file_list[0] == '')
    return ElMessage.error('文件不能为空！')
  let datafile = uploads.file_list[0].raw
  let model_params = {
    'modules': ['添加噪声', '音频分离'],
    'algorithms': { '添加噪声': 'WhiteGaussianNoise' , '音频分离': 'sepformer' },
    'parameters': { '添加噪声': { 'SNR': -10} , '音频分离': { 'param1': 10, 'param2': 20, 'param3': 100 } },
    'schedule': ['添加噪声', '音频分离']
  };
  const data = new FormData()
  data.append('file', datafile)
  data.append('params', JSON.stringify(model_params))
  console.log(data.getAll)

  axios.post('http://127.0.0.1:8000/homepage/', data,
      {
        headers: { "Content-Type": 'multipart/form-data' }
      }
  ).then((response) => {
    if (response.statusText == 'OK') {
      ElMessage.success('文件上传成功！')
    }
    else {
      ElMessage.error('文件上传失败，请重新上传！')
    }
  })

}


// const beforeUPload = (file: any) => {
//     console.log('调用了beforeUPload')
//     console.log(file.type)
//     const isWav =
//         file.type === 'audio/wav'
//     if (!isWav)
//         ElMessageBox({
//             title: '温馨提示',
//             message: '上传文件只能是 wav 格式！',
//             type: 'warning',
//         });
//     return isWav;
// }

const onChangeHandler = (file: any) => {
  if (file.status != 'ready') {
    return
  }
  let suffName = file.name.substring(file.name.lastIndexOf('.') + 1)
  // console.log('调用了onChangeHandler')
  // console.log(suffName)
  const isWav =
      (suffName === 'wav' || suffName ==='audio')
  if (!isWav){
    ElMessageBox({
      title: '温馨提示',
      message: '上传文件只能是 wav/audio 格式！',
      type: 'warning',
    });
    uploadRef.value!.clearFiles()
  }

  return isWav;
}
</script>