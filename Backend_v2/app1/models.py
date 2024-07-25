from datetime import datetime
from django.contrib.auth.models import AbstractUser
# from django.contrib.auth import get_user_model
from django.db import models
from django.conf import settings


# 预处理后的数据
class PreprocessedData(models.Model):
    # 时域特征
    T_max = models.FloatField(verbose_name="时域最大值", null=False, blank=False, default=0)
    T_min = models.FloatField(verbose_name="时域最小值", null=False, blank=False, default=0)
    T_mean = models.FloatField(verbose_name="时域平均值", null=False, blank=False, default=0)
    T_std = models.FloatField(verbose_name="时域标准差", null=False, blank=False, default=0)
    T_median = models.FloatField(verbose_name="时域中位数", null=False, blank=False, default=0)
    T_peak_peak = models.FloatField(verbose_name="时域峰峰值", null=False, blank=False, default=0)
    T_variance = models.FloatField(verbose_name="时域方差", null=False, blank=False, default=0)
    T_kurtosis = models.FloatField(verbose_name="时域峰度", null=False, blank=False, default=0)
    T_skewness = models.FloatField(verbose_name="时域偏度", null=False, blank=False, default=0)
    T_root_mean_squared = models.FloatField(verbose_name="时域均方根值", null=False, blank=False, default=0)
    T_waveform_factor = models.FloatField(verbose_name="波形因子", null=False, blank=False, default=0)
    T_peak_factor = models.FloatField(verbose_name="峰值因子", null=False, blank=False, default=0)
    T_pulse_factor = models.FloatField(verbose_name="脉冲因子", null=False, blank=False, default=0)
    T_margin_factor = models.FloatField(verbose_name="裕度因子", null=False, blank=False, default=0)
    T_root_amplitude = models.FloatField(verbose_name="方根幅值", null=False, blank=False, default=0)
    T_commutation_mean = models.FloatField(verbose_name="整流平均值", null=False, blank=False, default=0)
    T_fourth_order_accumulation = models.FloatField(verbose_name="四阶累积量", null=False, blank=False, default=0)
    T_sixth_order_accumulation = models.FloatField(verbose_name="六阶累积量", null=False, blank=False, default=0)

    # 频域特征
    F_centroid_frequency = models.FloatField(verbose_name="重心频率", null=False, blank=False, default=0)
    F_mean_squared_frequency = models.FloatField(verbose_name="均方频率", null=False, blank=False, default=0)
    F_rmsf = models.FloatField(verbose_name="均方根频率", null=False, blank=False, default=0)
    F_vf = models.FloatField(verbose_name="频率方差", null=False, blank=False, default=0)
    F_rvf = models.FloatField(verbose_name="频率标准差", null=False, blank=False, default=0)
    F_sk_mean = models.FloatField(verbose_name="谱峭度均值", null=False, blank=False, default=0)
    F_sk_std = models.FloatField(verbose_name="谱峭度标准差", null=False, blank=False, default=0)
    F_sk_skewness = models.FloatField(verbose_name="谱峭度偏度", null=False, blank=False, default=0)
    F_sk_kurtosis = models.FloatField(verbose_name="谱峭度峰度", null=False, blank=False, default=0)


# 中间结果
class IntermediateResult(models.Model):
    inference = models.CharField(verbose_name="预测结果", max_length=6, null=False, blank=False, default='NA')


# 已注册的说话人
class RegisteredSpeakers(models.Model):
    name = models.CharField(verbose_name="说话人姓名", max_length=32, null=False, blank=False, default='NA')
    feature_vector = models.BinaryField(verbose_name='说话人音频', null=False, blank=False)


class SavedModels(models.Model):
    name = models.CharField(verbose_name="模型名称", max_length=32, null=False, blank=False, default='unknown')
    # date = models.DateField(verbose_name="创建时间", null=False, blank=False, default=datetime)
    model_info = models.JSONField(verbose_name="模型信息", null=False, blank=False)


# 用户建立的模型
class SavedModelFromUser(models.Model):
    author = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="模型建立者")
    model_name = models.CharField(verbose_name="模型名称", max_length=32, null=False, blank=False, default='unknown')
    model_info = models.JSONField(verbose_name="模型信息", null=False, blank=False)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['author', 'model_name'], name='unique_author_modelName')
        ]


# 用户所有的数据，以文件路径的形式保存
class SavedDatasetsFromUser(models.Model):
    owner = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="数据所有者")
    dataset_name = models.CharField(verbose_name="数据文件名", max_length=32, null=False, blank=False, default='')
    file_path = models.FileField(verbose_name="文件存放路径", max_length=255, null=False, blank=False, default='')
    description = models.TextField(verbose_name="文件描述", null=False, blank=False, default='无')

    # 联合去重
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['owner', 'dataset_name'], name='unique_owner_dataset')
        ]


class CustomUser(AbstractUser):
    jobNumber = models.CharField(verbose_name="工号", max_length=32, null=False, blank=False, unique=True, default='unknown')


# 用于保存用户邮箱验证码
class CaptchaModel(models.Model):
    email = models.EmailField(verbose_name="邮箱地址", unique=True)
    captcha = models.CharField(max_length=4, verbose_name="验证码")
    create_time = models.DateTimeField(verbose_name="发送时间", auto_now_add=True)

