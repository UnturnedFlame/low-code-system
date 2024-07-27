import axios from 'axios'



const api = axios.create({
    baseURL: 'http://127.0.0.1:8000/',
});

// 拦截器
// 拦截请求并将登录时从服务器获得的token添加到Authorization头部
api.interceptors.request.use(function (config) {
    // 从localStorage获取JWT
    let token = window.localStorage.getItem('jwt');
    // console.log('the token is: ', token)

    // 将JWT添加到请求的Authorization头部
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;

}, function (error) {
    // 请求错误处理
    return Promise.reject(error);
})
// 可以自请求发送前对请求做一些处理
// 比如统一加token，对请求参数统一加密


// 网关拦截器
// 可以在接口响应后统一处理结果


export default api
