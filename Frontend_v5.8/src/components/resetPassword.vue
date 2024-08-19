<template>
  <div class="resetPassword">
    <div class="container" v-if="active==0 || active==1 || active==2">
      <el-steps :active="active" :space="200" finish-status="success"  align-center>
        <el-step title="验证用户名和邮箱" icon="EditPen"></el-step>
        <el-step title="输入验证码" icon="Promotion"></el-step>
        <el-step title="设置新密码" icon="Key"></el-step>
      </el-steps>
      <div v-if="active === 0" class="common_div">
        <el-form :model="Form"  :rules="rules"  ref="formRef" class="user-container" label-position="left" label-width="70px" size="default">
          <el-form-item  prop="username" style="float: right; width: 80%" label="用户名">
            <el-input type="text" v-model="Form.username" autofocus ref="username" auto-complete="off"
                      placeholder="请输入要找回密码的用户名" prefix-icon="User" spellcheck="false" >
            </el-input>
          </el-form-item>
          <el-form-item prop="email" style="float: right; width: 80%" label="邮箱号">
            <el-input type="text" v-model="Form.email" autofocus ref="email" auto-complete="off"
                      placeholder="请输入用来找回密码的邮箱" prefix-icon="Message" spellcheck="false" >
            </el-input>
          </el-form-item>
        </el-form>
      </div>
      <div v-if="active === 1" class="common_div">
        <el-form :model="codeForm"  class="user-container" label-position="left" label-width="60px" size="default">
          <el-form-item  style="float: right; width: 80%" label="验证码">
            <el-input type="text" v-model="codeForm.code" autofocus ref="code" auto-complete="off"
                      placeholder="请输入邮箱验证码" prefix-icon="el-icon-s-promotion" spellcheck="false">
            </el-input>
          </el-form-item>
        </el-form>
      </div>
      <div v-if="active === 2" class="common_div">
        <el-form :model="passwordForm" :rules="rules" class="user-container" label-position="left" label-width="90px" size="default">
          <el-form-item prop="password" style="float: right; width: 80%" label="新密码">
            <el-input type="password" v-model="passwordForm.password" autofocus ref="password"
                      auto-complete="off" placeholder="请输入新密码" prefix-icon="el-icon-key" >
            </el-input>
          </el-form-item>
          <el-form-item style="float: right; width: 80%" label="确认新密码">
            <el-input type="password" v-model="passwordForm.confirmPassword" ref="confirmPassword"
                      auto-complete="off" placeholder="请再次输入新密码" prefix-icon="el-icon-key">
            </el-input>
          </el-form-item>
        </el-form>
      </div>
      <div class="common_div">
        <el-button  @click="nextStep(formRef)" :disabled="false" class="action_button">下一步</el-button>
      </div>

    </div>
    <div class="container" v-if="active==3">
      <div style="height: 100px; width: 500px; font-size: 20px; text-align: center;">密码修改成功！</div>
      <el-button @click="backLogin" type="success" style="text-align: center; width: 30%; margin-left: 170px; font-size: large; color: white">返回登录界面</el-button>
    </div>
  </div>
</template>

<script setup>
import { ElMessage,ElNotification } from 'element-plus';
import {onMounted, reactive, ref} from "vue"; // 假设使用Element UI提供的消息提示
import {useRouter} from "vue-router";
import api from '../utils/api.js'

const formRef = ref(null);
// 定义常量
const STEP_0 = 0;
const STEP_1 = 1;
const STEP_2 = 2;
const STEP_3 = 3;

const router = useRouter();

// 初始化状态
const active = ref(STEP_0);
const Form = reactive({
  username: '',
  email: '',
});
const codeForm = reactive({
  code: '',
});
const passwordForm = reactive({
  password: '',
  confirmPassword: '',
});


const  rules = {
  password: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 8, max: 20, message: '长度在 8 到 20 个字符', trigger: 'blur' },
    { pattern: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d]{8,20}$/, message: '密码必须包含大小写字母和数字', trigger: 'blur' }
  ],
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 15, message: '长度在 3 到 15 个字符', trigger: 'blur' }
  ],
  email: [
    { required: true, message: '请输入邮箱地址', trigger: 'blur' },
    { type: 'email', message: '请输入正确的邮箱地址', trigger: ['blur', 'change'] },
    // 使用正则表达式来验证邮箱格式
    { pattern: /^[A-Za-z0-9\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+$/, message: '邮箱格式不正确', trigger: 'blur' }
  ]
}
const disabled = ref(false);

// 在组件挂载时进行初始化检查
onMounted(() => {
});


// 表单验证
const validateForm = () => {
  if (Form.username === '' || Form.email === '') {
    if (Form.username === '') {
      ElMessage.warning('用户名未输入');
    } else {
      ElMessage.warning('邮箱未输入');
    }
  }
  return true;
};

let username
let email
// 下一步操作
const nextStep = (formRef) => {
  if (active.value === STEP_0) {
    if (validateForm()) {
      formRef.validate().then(() => {
        disabled.value = true;
        let formData = new FormData();
        
        formData.append('username', Form.username)
        formData.append('email', Form.email)
        
        api.post('/resetPassword/send_captcha/', 
          formData
        ).then(successResponse => {
          if (successResponse.data.code === 200) {
            active.value++;
            
            ElNotification({
              title: '成功',
              message: '验证码发送成功！',
              type: 'success',
            });
          } else if (successResponse.data.code === 400) {
            ElMessage.error(successResponse.data.message);
          }
        });
        ElNotification.info({
          title: '提示',
          message: '数据正确发送，请耐心等待，勿重复操作！',
        });
})
    }
  } else if (active.value === STEP_1) {
    if (codeForm.code === '') {
      ElMessage.warning('验证码未输入');
      return;
    }
    let formData = new FormData();
    formData.append('captcha', codeForm.code)
    formData.append('email', Form.email)
    api.post('/resetPassword/check_captcha/', 
      formData
    ).then(successResponse => {
      if (successResponse.data.code === 200) {
        active.value++;
        ElNotification({
          title: '成功',
          message: '验证码匹配正确！',
          type: 'success',
        });
      } else if (successResponse.data.code === 400) {
        ElMessage.error(successResponse.data.message);
      }
    }).catch(error => {
      ElMessage.error('请求失败，请重试');
    });
  } else if (active.value === STEP_2) {
    if (passwordForm.password === '') {
      ElMessage.warning('新密码未输入');
      return;
    }
    if (passwordForm.confirmPassword !== passwordForm.password){
      ElMessage.warning('两次密码不一致');
      return;
    }
    let formData = new FormData();
    formData.append('email', Form.email)
    formData.append('password', passwordForm.password)
    api.post('/resetPassword/reset_password/', 
      formData
    ).then(successResponse => {
      if (successResponse.data.code === 200) {
        ElNotification({
          title: '成功',
          message: '该账号密码修改正确！',
          type: 'success',
        });
        active.value++;
        // let path = window.location.search.includes('redirect') ? decodeURIComponent(window.location.search.split('redirect=')[1]) : '/login';
        // window.location.href = path;
      } else if (successResponse.data.code === 400) {
        ElMessage.error(successResponse.data.message);
      }
    }).catch(error => {
      ElMessage.error('请求失败，请重试');
    });
  }
};

const backLogin = () => {
  router.push('/')
};
</script>

<style scoped>
.resetPassword{
  background-image: url("../assets/login-background.jpg");
  background-position: center;
  height: 100%;
  width: 100%;
  background-size: cover;
  position: fixed;
}
.container{
  border-radius: 15px;
  background-clip: padding-box;
  margin: 10% auto;
  width: 30%;
  padding: 25px 30px;
  background: #fff;
  border: 1px solid #eaeaea;
  box-shadow: 0 0 25px #cac6c6;
  opacity: 0.7;
}
.common_div{
  margin-top: 5%;
}
.user-container {
  width: 80%;
  background: #fff;

}
.action_button {
  width: 20%;
  margin-top: 3%;
  text-align: center;
}

</style>

