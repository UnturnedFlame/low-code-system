"""
URL configuration for Demo1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from app1 import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView
)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("user/run_with_local_datafile/", views.run_with_local_datafile),
    path("user/run_with_datafile_on_cloud/", views.run_with_datafile_on_cloud),
    path("register_speaker/", views.register_speaker),
    path("shut/", views.shut),
    path("save_model/", views.save_model),
    path("user_save_model/", views.user_save_model),
    path("upload_datafile/", views.upload_datafile),
    path("delete_datafile/", views.delete_datafile),
    path("fetch_models/", views.fetch_models),
    path("user_fetch_models/", views.user_fetch_models),
    path("user_delete_model/", views.user_delete_model),
    path("user/fetch_datafiles/", views.user_fetch_datafiles),
    path("delete_model/", views.delete_model),
    path("resetPassword/send_captcha/", views.send_email_captcha),
    path("resetPassword/check_captcha/", views.check_captcha),
    path("resetPassword/reset_password/", views.user_reset_password),

    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('login/', views.login),
    path('captcha/', views.send_email_captcha),
    path('register/', views.register),
    path('administration/fetch_users_models/', views.admin_fetch_users_models),
    path('administration/delete_user_model/', views.admin_delete_model),
    path('administration/fetch_users_info/', views.admin_fetch_user_info),
    path('administration/fetch_users_datafiles/', views.admin_fetch_users_datafiles),
    path('administration/delete_user_datafile/', views.admin_delete_users_files),
    path('administration/reset_user_password/', views.admin_reset_user_password),
    path('add_user/', views.add_user),
    path('delete_user/', views.delete_user),
]
