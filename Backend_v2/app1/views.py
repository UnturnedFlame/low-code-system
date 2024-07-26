import json
import os
import time

import numpy as np
from django.contrib import messages
from django.contrib.auth.models import Group
from django.contrib.auth import authenticate, get_user_model
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from datetime import datetime, timedelta
from django.core.mail import send_mail
import random
import string
import jwt

from Backend_v2 import settings
# from app1.module_management.algorithms.functions.speech_processing import extract_speaker_feature
# from app1.module_management.algorithms.functions.load_model import load_model_with_pytorch_lightning
from app1 import models
from app1 import forms
from app1.models import CaptchaModel
from app1.module_management.gradioAppDemo import Reactor

CORS_ALLOW_METHODS = (
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
)

data_file_root_dir = './app1/recv_file/examples'

User = get_user_model()
# def homepage(request):
#     if request.method == "GET":
#         return render(request, "homepage.html")
#     '''可视化操作，包括：数据预处理、模型选择、结果预测'''
#     # 上传文件并进行数据预处理
#     if request.method == 'POST':
#         data_file = request.FILES.get("loadFile", None)
#         if data_file is not None:
#             # 打开特定的文件进行二进制的写操作
#             # print(os.path.exists('/recv_file/'))
#             with open("./app1/recv_file/%s" % data_file.name, 'wb+') as f:
#                 # 分块写入文件
#                 for chunk in data_file.chunks():
#                     f.write(chunk)
#             # 进行数据的预处理
#             data_csv = pd.read_csv("./app1/recv_file/%s" % data_file.name)
#             data = np.array(data_csv)
#             # data为二维数组
#             features = []
#             # 特征提取，包括时域和频域特征
#             for row in data:
#                 t_features = feature_extraction.time_domain_extraction(row)
#                 f_features = feature_extraction.fre_domain_extraction(row)  # 默认采样率为1000fs
#                 features.append(dict(t_features, **f_features))
#             # data = dataset.TrainDataset(train_csv_path=f"./app1/recv_file/{data_file.name}", second=3)
#             # 将提取的特征存入数据库
#             for f_row in features:
#                 models.PreprocessedData.objects.create(**{k: v for k, v in f_row.items()})
#         else:
#             pass
#
#     return redirect("/homepage")


def run_with_local_datafile(request):
    if request.method == 'POST':
        datafile = request.FILES.get('file', None)
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)
            if datafile is None:
                return JsonResponse({'message': 'datafile is required'}, status=400)
            else:
                # 打开特定的文件进行二进制的写操作
                # print(os.path.exists('/recv_file/'))
                save_path = f"./app1/recv_file/examples/{username}/{datafile.name}"
                # save_path = f"./app1/recv_file/examples/{datafile.name}"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(save_path, 'wb+') as f:
                    # 分块写入文件
                    for chunk in datafile.chunks():
                        f.write(chunk)

                # 保存文件路径到数据库
                # if not models.SavedModelFromUser.objects.filter(owner=user, dataset_name=datafile.name).exists():
                #     saved_data = models.SavedDatasetsFromUser.objects.create(owner=user, dataset_name=datafile.name,
                #                                                              file_path=save_path)
                #     saved_data.save()

                params = json.loads(request.POST.get('params'))
                print(params)

                module_list = params['modules']
                print(module_list)
                algorithm_dict = params['algorithms']
                print(algorithm_dict)
                params_dict = params['parameters']
                print(params_dict)
                schedule = params['schedule']
                print(schedule)

                demo_app = Reactor()
                demo_app.init(schedule, algorithm_dict, params_dict)
                demo_app.start(datafile='./app1/recv_file/examples/' + datafile.name)
                use_tabs_dict = demo_app.module_configuration
                results = demo_app.results_to_response
                # print('results:', results)
                np.save('app1/module_management/use_tabs_dict.npy', use_tabs_dict, allow_pickle=True)

                # speech_processing_results = processing(module_list, algorithm_dict, params_dict)
                # 构建用于结果展示的页面

                return JsonResponse({'status': 'success', 'results': results})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=401)
    if request.method == 'GET':
        display = request.GET.get('display')
        display_list = [display]
        if not display:
            return JsonResponse({'status': 'shutdown'})
        demo_app = Reactor()
        demo_app.module_configuration = np.load('app1/module_management/use_tabs_dict.npy', allow_pickle=True).item()
        port = demo_app.display(display_list=display_list, no_thread_lock=True)
        time.sleep(4.5)
        demo_app.shutdown()
        return JsonResponse({'status': 'success', 'port': port})


def register_speaker(request):
    if request.method == 'POST':
        datafile = request.FILES.get('file', None)
        params = json.loads(request.POST.get('params', None))
        print(params)
        # lightning_model = load_model_with_pytorch_lightning()
        if datafile is not None:
            # 打开特定的文件进行二进制的写操作
            with open("./app1/recv_file/speakers/%s" % datafile.name, 'wb+') as f:
                # 分块写入文件
                for chunk in datafile.chunks():
                    f.write(chunk)
            # feature_vector = extract_speaker_feature(audio_path='./app1/recv_file/speakers/' + datafile.name,
            #                                          lightning_model=lightning_model)
            # print(feature_vector)
            # print(f'length of feature vector: {len(feature_vector)}')
            # models.RegisteredSpeakers.objects.create(name=params['speaker'], feature_vector=feature_vector)
            res = models.RegisteredSpeakers.objects.all()
            print(res[0].feature_vector)
            vector = np.frombuffer(res[0].feature_vector, dtype=np.float32)
            print(vector.reshape((1, -1)))

        return JsonResponse({'status': 'success'})


def shut(request):
    if request.method == 'GET':
        print('请求shutdown')
        return JsonResponse({'status': 'shutdown'})


def save_model(request):
    if request.method == 'POST':
        print('请求保存模型')
        model_name = request.POST.get('model_name')
        params_json = request.POST.get('model_info')
        print('模型名称：' + model_name, '\n', '模型信息：' + params_json)
        # 保存模型
        if not models.SavedModels.objects.filter(name=model_name).exists():
            models.SavedModels.objects.create(name=model_name, model_info=params_json)
            return JsonResponse({'message': 'save_model_successful'})
        return JsonResponse({'message': 'save_model_failed'})


# 用户保存模型
def user_save_model(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        model_name = request.POST.get('model_name')
        params_json = request.POST.get('model_info')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)

            if not models.SavedModelFromUser.objects.filter(model_name=model_name, author=user).exists():

                models.SavedModelFromUser.objects.create(author=user, model_name=model_name, model_info=params_json)

                return JsonResponse({'message': 'save model success', 'code': 200})
            return JsonResponse({'message': 'save_model_failed, model already exists!', 'code': 400})
        except jwt.ExpiredSignatureError:
            return JsonResponse({'message': 'signature expired!'}, status=401)


# 获取模型操作
def fetch_models(request):
    if request.method == 'GET':
        objects = models.SavedModels.objects.all()
        posts = objects.values()

        return JsonResponse(list(posts), safe=False)


# 通过用户名获取模型
def user_fetch_models(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)
            objects = models.SavedModelFromUser.objects.filter(author=user)
            posts = objects.values()

            return JsonResponse(list(posts), safe=False)
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=401)


def admin_fetch_models(request):
    if request.method == 'GET':
        all_user_models = models.SavedModelFromUser.objects.all()

        models_list = [
            {
                'id': model.id,
                'model_name': model.model_name,
                'model_info': model.model_info,
                'jobNumber': model.author.jobNumber
            } for model in all_user_models
        ]

        return JsonResponse(models_list, safe=False)


def admin_fetch_user_info(request):
    if request.method == 'GET':
        all_users = User.objects.all().values()
        return JsonResponse(list(all_users), safe=False)


# 删除模型操作
def delete_model(request):
    row_id = request.GET.get('row_id')
    models.SavedModels.objects.filter(id=row_id).delete()

    return JsonResponse({'message': 'deleteSuccessful'})


def admin_reset_user_password(request):
    if request.method == 'GET':
        jobNumber = request.GET.get('jobNumber')

        user = User.objects.filter(jobNumber=jobNumber).first()

        if user:
            length = 8
            characters = string.ascii_letters + string.digits  # 大小写字母和数字
            new_password = ''.join(random.choice(characters) for _ in range(length))
            user.set_password(new_password)
            user.save()

            return JsonResponse({'message': f'密码重置成功，新密码为{new_password}', 'code': 200})
        else:
            return JsonResponse({'message': '重置密码失败, 未找到该用户', 'code': 400})


# 管理员删除用户模型
def admin_delete_model(request):
    if request.method == 'GET':
        row_id = request.GET.get('row_id')
        try:
            models.SavedModelFromUser.objects.filter(id=row_id).delete()
            return JsonResponse({'message': 'delete user model success', 'code': 200})
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=400)


# 用户删除模型
def user_delete_model(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        row_id = request.GET.get('row_id')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            user = User.objects.get(username=payload.get('username'))
            models.SavedModelFromUser.objects.filter(author=user, id=row_id).delete()
            return JsonResponse({'message': 'deleteSuccessful'})
        except Exception as e:
            return JsonResponse({'message': str(e)}, status=401)


# 用户上才传文件
def upload_datafile(request):
    if request.method == 'POST':
        datafile = request.FILES.get('file', None)
        filename = request.POST.get('filename')
        description = request.POST.get('description')
        print("datafile: ", datafile)
        print("filename: ", filename)
        print("description: ", description)
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            if datafile is not None:
                user_save_dir = f"./app1/recv_file/examples/{username}"
                if not os.path.exists(user_save_dir):
                    os.makedirs(user_save_dir)
                save_path = os.path.join(user_save_dir, datafile.name)
                with open(save_path, 'wb+') as f:
                    # 分块写入文件
                    for chunk in datafile.chunks():
                        f.write(chunk)
                if not models.SavedDatasetsFromUser.objects.filter(owner=username,
                                                                   dataset_name=datafile.name).exists():
                    user = User.objects.get(username=username)
                    saved_data = models.SavedDatasetsFromUser.objects.create(owner=user,
                                                                             dataset_name=datafile.name,
                                                                             file_path=save_path)
                    saved_data.save()
                    return JsonResponse({'message': 'save data success', 'code': 200})
                else:
                    return JsonResponse({'message': '同名文件已存在', 'code': 400})
            return JsonResponse({'message': '无效的文件路径', 'code': 400})

        except jwt.ExpiredSignatureError:
            return JsonResponse({'message': '用户签名过期'})


def your_view_function(request):
    if request.method == 'POST':
        try:
            datafile = request.FILES.get('file', None)
            filename = request.POST.get('filename')
            description = request.POST.get('description')

            # 打印接收到的数据
            print("datafile: ", datafile)
            print("filename: ", filename)
            print("description: ", description)

            token = extract_jwt_from_request(request)
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            if datafile is not None:
                user_save_dir = f"./app1/recv_file/examples/{username}"
                if not os.path.exists(user_save_dir):
                    os.makedirs(user_save_dir)

                    # 检查目录权限（可选）
                if not os.access(user_save_dir, os.W_OK):
                    raise PermissionError(f"无法写入目录 {user_save_dir}")

                save_path = os.path.join(user_save_dir, datafile.name)

                try:
                    with open(save_path, 'wb+') as f:
                        # 分块写入文件
                        for chunk in datafile.chunks():
                            f.write(chunk)
                except Exception as e:
                    # 捕获并打印文件写入时的任何异常
                    print(f"写入文件时发生错误: {e}")
                    raise  # 可选：重新抛出异常以便上层处理

        except Exception as e:
            # 捕获并打印处理文件时的任何异常
            print(f"处理文件时发生错误: {e}")


# 用户获取上传的文件数据
def fetch_datafiles(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)

        try:
            payload = jwt.decode(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)
            objects = models.SavedDatasetsFromUser.objects.filter(owner=user)

            posts = [
                {
                    'dataset_name': obj.dataset_name,
                    'owner': obj.owner.username,

                } for obj in objects
            ]

            return JsonResponse(posts, safe=False)
        except jwt.ExpiredSignatureError:
            return JsonResponse({'message': 'signature expired'}, status=401)


# 用户删除数据文件
def delete_datafile(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        filename = request.GET.get('filename')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)
            objects = models.SavedDatasetsFromUser.objects.filter(owner=user, dataset_name=filename).first()
            if objects:
                objects.delete()
                return JsonResponse({'message': 'deleted successfully', 'code': 200})
            else:
                return JsonResponse({'message': 'file not found', 'code': 400})
        except Exception as e:
            return JsonResponse({'message': str(e), 'code': 401})


# 用户概览数据文件
def view_datafiles(request):
    if request.method == 'GET':
        username = request.GET.get('username')
        filename = request.GET.get('filename')


# 通过服务器中保存的用户数据文件运行
def run_with_datafile_on_cloud(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        file_path = request.POST.get('file_path')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            params = json.loads(request.POST.get('params'))

            algorithm_dict = params['algorithms']
            params_dict = params['parameters']
            schedule = params['schedule']

            demo_app = Reactor()
            demo_app.init(schedule, algorithm_dict, params_dict)
            demo_app.start(datafile=file_path)
            results = demo_app.results_to_response

            return JsonResponse({'message': 'success', 'results': results, 'code': 200})
        except jwt.ExpiredSignatureError:
            return JsonResponse({'message': 'signature expired', 'code': 401})




# def login_user(request):
#     if request.method == 'POST':
#         username = request.GET.get('username')
#         password = request.GET.get('password')
#
#         user = User.objects.filter(username=username).first()
#         if not user:
#             return JsonResponse({'message': 'user not exists'})
#         # 获取用户角色
#         user_groups = user.groups.all()
#         # 检查用户角色
#         specific_group_name = 'User'
#         user_is_in_specific_group = any(group.name == specific_group_name for group in user_groups)
#         if not user_is_in_specific_group:
#             return JsonResponse({'message': 'user not exists'})
#
#         # 检查登录密码
#         if password == user.password:
#             return JsonResponse({'message': 'login success'})
#         else:
#             return JsonResponse({'message': 'invalid password'})


def login(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        username = data.get('username')
        password = data.get('password')
        role = data.get('role')

        print('username: ', username)
        print('password: ', password)
        print('role: ', role)

        user = User.objects.filter(username=username).first()
        if not user:
            return JsonResponse({'message': 'user not exists'})
        # 获取用户角色
        user_groups = user.groups.all()
        print(user_groups)
        # 检查用户角色
        specific_group_name = role
        user_is_in_specific_group = any(group.name == specific_group_name for group in user_groups)
        if not user_is_in_specific_group:
            print('不存在该角色的用户')
            return JsonResponse({'message': 'user not exists'})
        print('user.password: ', user.password)
        # 检查登录密码
        if authenticate(request, username=username, password=password):
            payload = {
                'user_id': user.id,
                'username': user.username,
                'exp': datetime.utcnow() + timedelta(hours=24),  # 设置过期时间为24小时后
                'iat': datetime.utcnow(),  # 签发时间
            }
            token = jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')
            print('登陆成功')
            # 返回JWT给客户端
            return JsonResponse({
                'token': token.decode('utf-8'),
                'message': 'login success'
            })

        else:
            print('密码错误')
            return JsonResponse({'message': 'invalid password'})


# 从http请求中提取token
def extract_jwt_from_request(request) -> str:
    """
    从HTTP请求中提取JWT。
    规定JWT通过Authorization头部以"Bearer <token>"的形式发送。
    """
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    prefix = 'Bearer '
    if auth_header.startswith(prefix):
        return auth_header[len(prefix):]
    return ''


# token验证
def verify_jwt(token: str, secret_key: str) -> dict:
    """
    验证JWT并返回其payload。
    如果JWT无效（如签名不匹配、已过期等），则抛出异常。
    """
    try:
        # 设置JWT的验证选项，包括验证过期时间
        options = {
            'verify_signature': True,
            'verify_exp': True,
            'verify_nbf': True,
            'verify_iat': True,
            'verify_aud': False,  # 根据需要设置
            'verify_iss': False,  # 根据需要设置
            'require_exp': True,
            'require_iat': True,
            'require_nbf': False,  # 根据需要设置
            'algorithms': ['HS256'],  # 使用的算法
            'leeway': 0  # 允许的时间误差（秒）
        }

        # 解析JWT
        payload = jwt.decode(token, secret_key, **options)
        return payload
    except jwt.ExpiredSignatureError:
        # 处理JWT过期的情况
        raise Exception('JWT已过期')
    except jwt.InvalidTokenError:
        # 处理JWT无效的情况（如签名不匹配）
        raise Exception('无效的JWT')


# 管理员获取用户列表
def fetch_users(request):
    if request.method == 'GET':
        users = User.objects.all()
        posts = users.values()

        return JsonResponse(list(posts), safe=False)


# 管理员添加用户操作
def add_user(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        username = data.get('username')
        password = data.get('password')
        email = data.get('email')
        jobNumber = data.get('jobNumber')

        if User.objects.filter(jobNumber=jobNumber).exists():
            return JsonResponse({'message': 'failed, jobNumber has been registered', 'code': 400})
        elif User.objects.filter(username=username).exists():
            return JsonResponse({'message': 'username already exists', 'code': 400})
        else:
            user = User.objects.create_user(username=username, password=password, email=email, jobNumber=jobNumber)
            user_group = Group.objects.get(name='user')
            user.groups.add(user_group)
            user.save()
            # 登录用户（可选）
            # from django.contrib.auth import login
            # login(request, user)

            return JsonResponse({'message': 'add user success', 'code': 200})


# 管理员删除用户操作
def delete_user(request):
    if request.method == 'GET':
        jobNumber = request.GET.get('jobNumber')
        # 使用get_object_or_404来尝试获取用户对象
        # 如果找不到对应的用户，将返回404错误页面
        user = User.objects.get(jobNumber=jobNumber)

        if user:
            user.delete()
            return JsonResponse({'message': 'user deleted success', 'code': 200})
        else:
            return JsonResponse({'message': 'user not found', 'code': 404})


# 发送验证码到邮箱
def send_email_captcha(request):
    if request.method == 'GET':
        email = request.GET.get('email')
        if not email:
            return JsonResponse({'code': 400, 'message': 'email is required'})
        captcha = ''.join(random.sample(string.digits, k=4))
        # 保存验证码
        models.CaptchaModel.objects.update_or_create(email=email, defaults={'captcha': captcha})
        send_mail(subject="车轮状态分析与健康评估平台验证码",
                  message=f"您的验证码为{captcha}，请勿泄露给他人，并注意使用时限", recipient_list=[email],
                  from_email=None)
        return JsonResponse({"code": 200, "message": "captcha has been send to the email"})


def authenticate_register(request):
    form_data = json.loads(request.body)
    email = form_data['email']
    # username = form_data['username']
    captcha = form_data['captcha']

    exists = User.objects.filter(email=email).exists()
    if exists:
        return 'email exists'
    else:
        captcha_model = CaptchaModel.objects.filter(email=email, captcha=captcha).first()
        if not captcha_model:
            return 'captcha does not match the email'
        else:
            captcha_model.delete()
            return 'auth success'


# 使用邮箱注册用户
def register(request):
    if request.method == 'POST':
        auth = authenticate_register(request)
        # form表单需包括用户名、邮箱、验证码、密码
        form_data = json.loads(request.body)
        if auth == 'auth success':
            email = form_data.get('email')
            password = form_data.get('password')
            username = form_data.get('username')
            User.objects.create_user(email=email, username=username, password=password)
            return JsonResponse({'code': 200, 'message': 'register success'})
        else:
            print('auth error: ', auth)
            return JsonResponse({'code': 400, 'message': auth})
