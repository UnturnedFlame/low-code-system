import { createRouter, createWebHistory } from 'vue-router';

//设置路由规则
const routes = [
    {
        path: '/',
        name: 'Home',
        component: () =>import("../components/home.vue")
    },
    {
        path: '/UserPlatform',
        name: 'UserPlatform',
        component: () =>import('../components/userPlatform.vue')
    },
    {
        path: '/admin',
        name: 'admin',
        component: () =>import('../components/admin.vue'),
        children:[
            {
                path: '/admin/userManage',
                name: 'userManage',
                component: () =>import('../components/userTable.vue'),
            },
            {
                path: '/admin/modelManage',
                name: 'modelManage',
                component: () =>import('../components/modelTable.vue'),
            },
            {
                path: '/admin/dataManage',
                name: 'dataManage',
                component: () =>import('../components/dataTable.vue'),
            },
            {
                path: '/admin/addUser',
                name: 'addUser',
                component: () =>import("../components/addUser.vue")
            },
        ]

    },
    {
        path: '/test',
        name: 'test',
        component: () =>import("../components/uploadDatafile.vue")
    },

];

const router = createRouter({
    //设置路由模式
    history: createWebHistory(),
    routes
});

export default router;