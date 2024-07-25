"""
主要实现对单个信号的时频域特征提取，共27个特征
时域特征：均值、方差、标准差、峰度、偏度、四阶累积量、六阶累积量、最大值、最小值、中位数、峰峰值、整流平均值、均方根、方根幅度、波形因子、峰值因子、脉冲因子、裕度因子
频域特征：重心频率、均方频率、均方根频率、频率方差、频率标准差、谱峭度的均值、谱峭度的标准差、谱峭度的峰度、谱峭度的偏度
时域特征提取函数：timeDomain_extraction(signal)
频域特征提取函数：FreDomain_extraction(signal,sample_rate)
输入信号signal为numpy数组（一维），采样率sample_rate为double类型
输出为字典类型
"""
import numpy as np
import scipy.io
import scipy.stats as st
from scipy.stats import kurtosis, skew
from scipy.signal import welch

'''主要完成信号的时域特征提取, 其中各个函数的传入参数为数组形式的一维离散信号'''


# 最大值
def max(signal):
    max_v = np.max(signal)
    return max_v


# 最小值
def min(signal):
    min_v = np.min(signal)
    return min_v


# 中位数
def median(signal):
    median = np.median(signal)
    return median


# 峰峰值
def peak_peak(signal):
    pk_pk = np.max(signal) - np.min(signal)
    return pk_pk


# 均值
def mean(signal):
    mean = np.mean(signal)
    return mean


# 方差
def variance(signal):
    var = np.var(signal, ddof=1)
    return var


# 标准差
def std(signal):
    std = np.std(signal, ddof=1)
    return std


# 峰度（峭度）
def kurtosis(signal):
    kurt = st.kurtosis(signal, fisher=False)
    return kurt


# 偏度
def skew(signal):
    skew = st.skew(signal)
    return skew


# 均方根值
def root_mean_square(signal):
    rms = np.sqrt((np.mean(signal ** 2)))
    return rms


# 波形因子，信号均方根值和整流平均值的比值
def waveform_factor(signal):
    rms = root_mean_square(signal)  # 计算RMS有效值，即均方根值
    ff = rms / commutation_mean(signal)
    return ff


# 峰值因子，信号峰值与RMS有效值的比值
def peak_factor(signal):
    peak_max = np.max(np.abs(signal))  # 计算峰值
    rms = root_mean_square(signal)  # 计算RMS
    pf = peak_max / rms  # 峰值因子
    return pf


# 脉冲因子，信号峰值与整流平均值的比值
def pulse_factor(signal):
    peak_max = np.max(np.abs(signal))  # 计算峰值
    pf = peak_max / np.mean(np.abs(signal))
    return pf


# 裕度因子，信号峰值和方根幅值的比值
def margin_factor(signal):
    peak_max = np.max(np.abs(signal))  # 计算峰值
    rampl = root_amplitude(signal)  # 计算方根幅值
    margin_factor = peak_max / rampl  # 裕度因子
    return margin_factor


# 方根幅值
def root_amplitude(signal):
    rampl = ((np.mean(np.sqrt(np.abs(signal))))) ** 2
    return rampl


# 整流平均值，信号绝对值的平均值
def commutation_mean(signal):
    cm = np.mean(np.abs(signal))
    return cm


# 四阶累积量
def fourth_order_cumulant(signal):
    # 计算一阶、二阶中心距
    mean = np.mean(signal)
    variance = np.var(signal)

    # 计算四阶累积量
    # fourth_order_cumulant = np.mean((signal - np.mean(signal))**4)
    fourth_order_cumulant = np.mean((signal - mean) ** 4) - 3 * variance ** 2
    return fourth_order_cumulant


# 六阶累积量
def sixth_order_cumulant(signal):
    # 计算一阶、二阶中心距
    mean = np.mean(signal)
    variance = np.var(signal)

    # 计算六阶累积量
    sixth_order_cumulant = np.mean((signal - mean) ** 6) - 15 * variance * np.mean(
        (signal - mean) ** 4) + 30 * variance ** 3
    # data_centered = signal - np.mean(signal)
    # sixth_order_cumulant = np.mean(data_centered**6) - 15 * np.mean(data_centered**2) * np.mean(data_centered**4) + 30 * np.mean(signal)**2 * np.mean(data_centered**4)
    return sixth_order_cumulant


'''传入一维信号'''


def time_domain_extraction(signal):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 2:
        signal = signal.reshape(-1)

    time_domain_feature = {}
    time_domain_feature['T_max'] = max(signal)  # 最大值
    time_domain_feature['T_min'] = min(signal)  # 最小值
    time_domain_feature['T_median'] = median(signal)  # 中位数
    time_domain_feature['T_peak_peak'] = peak_peak(signal)  # 峰峰值
    time_domain_feature['T_mean'] = mean(signal)  # 均值
    time_domain_feature['T_variance'] = variance(signal)  # 方差
    time_domain_feature['T_std'] = std(signal)  # 标准差
    time_domain_feature['T_kurtosis'] = kurtosis(signal)  # 峰度
    time_domain_feature['T_skewness'] = skew(signal)  # 偏度
    time_domain_feature['T_commutation_mean'] = commutation_mean(signal)  # 整流平均值
    time_domain_feature['T_root_mean_squared'] = root_mean_square(signal)  # 均方根
    time_domain_feature['T_root_amplitude'] = root_amplitude(signal)  # 方根幅值
    time_domain_feature['T_waveform_factor'] = waveform_factor(signal)  # 波形因子
    time_domain_feature['T_peak_factor'] = peak_factor(signal)  # 峰值因子
    time_domain_feature['T_pulse_factor'] = pulse_factor(signal)  # 脉冲因子
    time_domain_feature['T_margin_factor'] = margin_factor(signal)  # 裕度因子
    time_domain_feature['T_fourth_order_accumulation'] = fourth_order_cumulant(signal)  # 四阶累积量
    time_domain_feature['T_sixth_order_accumulation'] = sixth_order_cumulant(signal)  # 六阶累积量

    return time_domain_feature


'''主要完成信号的时域特征提取, 其中各个函数的传入参数为数组形式的一维离散信号'''


def extract_spectrum(signal, sample_rate):
    '''
    frequencies：估计功率谱的频率数组
    power_spectrum：对应于频率的功率谱数值数组。（功率谱密度）
    SK: 信号的谱峭度
    注：在使用welch函数计算频率和功率谱密度时，参数是其默认参数
    '''
    # frequencies, power_spectrum = periodogram(signals, sample_rate, window=None, nfft=None, detrend='constant', scaling='density')
    # frequencies, power_spectrum = welch(signals, fs=sample_rate, nperseg=256)
    frequencies, t, power_spectrum = scipy.signal.spectrogram(signal, sample_rate, window='hamming', nperseg=256,
                                                              noverlap=128)
    return frequencies, power_spectrum


# 频率相关指标

# 重心频率（Centroid Frequency）
def CF(frequencies, power_spectrum):
    cf = (np.sum(frequencies * power_spectrum)) / (np.sum(power_spectrum))
    return cf


# 均方频率
def MSF(frequencies, power_spectrum):
    msf = np.sum((frequencies ** 2) * power_spectrum) / (np.sum(power_spectrum))
    return msf


# 均方根频率
def RMSF(frequencies, power_spectrum):
    rmsf = np.sqrt((np.sum((frequencies ** 2) * power_spectrum)) / (np.sum(power_spectrum)))
    return rmsf


# 频率方差
def VF(frequencies, power_spectrum):
    cf = CF(frequencies, power_spectrum)
    vf = np.sum(((frequencies - cf) ** 2) * power_spectrum) / (np.sum(power_spectrum))
    return vf


# 频率标准差
def RVF(frequencies, power_spectrum):
    vf = VF(frequencies, power_spectrum)
    rvf = np.sqrt(vf)
    return rvf


# 谱峭度相关指标
def SKone(frequencies, power_spectrum):
    u1 = np.sum(frequencies * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0)
    u2 = np.sqrt(np.sum((frequencies - u1) ** 2 * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0))
    sk = np.sum((frequencies - u1) ** 4 * power_spectrum, axis=0) / ((u2 ** 4) * np.sum(power_spectrum, axis=0))
    return sk


def SKMean(SK):  # 谱峭度的均值
    skMean = np.mean(SK)
    return skMean


def SKStd(SK):  # 谱峭度的标准差
    skStd = np.std(SK)
    return skStd


def SKSkewness(SK):  # 谱峭度的偏值
    skSkewness = skew(SK)
    return skSkewness


def SKKurtosis(SK):  # 谱峭度的峰度
    skKurtosis = kurtosis(SK)
    # skKurtosis = np.mean(power_spectrum ** 4) / (np.mean(power_spectrum ** 2)) ** 2 - 3
    return skKurtosis


def fre_domain_extraction(signal, sample_rate=1000):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 2:
        signal = signal.reshape(-1)

    frequencies, t, power_spectrum = scipy.signal.spectrogram(signal, sample_rate, window='hamming', nperseg=256,
                                                              noverlap=128)
    frequencies = frequencies.reshape(-1, 1)
    SK = SKone(frequencies, power_spectrum)
    fre_domain_feature = {}
    fre_domain_feature['F_centroid_frequency'] = CF(frequencies, power_spectrum)  # 重心频率
    fre_domain_feature['F_mean_squared_frequency'] = MSF(frequencies, power_spectrum)  # 均方频率
    fre_domain_feature['F_rmsf'] = RMSF(frequencies, power_spectrum)  # 均方根频率
    fre_domain_feature['F_vf'] = VF(frequencies, power_spectrum)  # 频率方差
    fre_domain_feature['F_rvf'] = RVF(frequencies, power_spectrum)  # 频率标准差
    fre_domain_feature['F_sk_mean'] = SKMean(SK)  # 谱峭度的均值
    fre_domain_feature['F_sk_std'] = SKStd(SK)  # 谱峭度的标准差
    fre_domain_feature['F_sk_skewness'] = SKSkewness(SK)  # 谱峭度的偏值
    fre_domain_feature['F_sk_kurtosis'] = SKKurtosis(SK)  # 谱峭度的峰度

    return fre_domain_feature
