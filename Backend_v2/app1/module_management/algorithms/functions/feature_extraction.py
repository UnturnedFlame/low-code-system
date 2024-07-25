"""
主要实现对单个信号的时频域特征提取，共27个特征
时域特征：均值、方差、标准差、峰度、偏度、四阶累积量、六阶累积量、最大值、最小值、中位数、峰峰值、整流平均值、均方根、方根幅值、波形因子、峰值因子、脉冲因子、裕度因子
频域特征：重心频率、均方频率、均方根频率、频率方差、频率标准差、谱峭度的均值、谱峭度的标准差、谱峭度的峰度、谱峭度的偏度

时域特征提取函数：timeDomain_extraction(signal)
频域特征提取函数：FreDomain_extraction(signal,sample_rate)
输入信号signal为numpy数组（一维），采样率sample_rate为double类型
输出为字典类型


"""
import numpy as np
import scipy.io
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from scipy.fft import fft

'''
主要完成信号的时域特征提取, 其中各个函数的传入参数为数组形式的一维离散信号
'''


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
# def kurtosis(signal):
#     kurt = st.kurtosis(signal, fisher=False)
#     return kurt


# 偏度
# def time_skew(signal):
#     skew = st.skew(signal)
#     return skew


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


def timeDomain_extraction(signal, features_to_extract):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 2:
        signal = signal.reshape(-1)

    # all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度', '整流平均值', '均方根',
    #                      '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子', '四阶累积量', '六阶累积量']

    timeDomain_feature = {}
    for feature in features_to_extract:
        match feature:
            case '最大值':
                timeDomain_feature['最大值'] = max(signal)  # 最大值
            case '最小值':
                timeDomain_feature['最小值'] = min(signal)  # 最小值
            case '中位数':
                timeDomain_feature['中位数'] = median(signal)  # 中位数
            case '峰峰值':
                timeDomain_feature['峰峰值'] = peak_peak(signal)  # 峰峰值
            case '均值':
                timeDomain_feature['均值'] = mean(signal)  # 均值
            case '方差':
                timeDomain_feature['方差'] = variance(signal)  # 方差
            case '标准差':
                timeDomain_feature['标准差'] = std(signal)  # 标准差
            case '峰度':
                timeDomain_feature['峰度'] = kurtosis(signal)  # 峰度
            case '偏度':
                timeDomain_feature['偏度'] = skew(signal)  # 偏度
            case '整流平均值':
                timeDomain_feature['整流平均值'] = commutation_mean(signal)  # 整流平均值
            case '均方根':
                timeDomain_feature['均方根'] = root_mean_square(signal)  # 均方根
            case '方根幅值':
                timeDomain_feature['方根幅值'] = root_amplitude(signal)  # 方根幅值
            case '波形因子':
                timeDomain_feature['波形因子'] = waveform_factor(signal)  # 波形因子
            case '峰值因子':
                timeDomain_feature['峰值因子'] = peak_factor(signal)  # 峰值因子
            case '脉冲因子':
                timeDomain_feature['脉冲因子'] = pulse_factor(signal)  # 脉冲因子
            case '裕度因子':
                timeDomain_feature['裕度因子'] = margin_factor(signal)  # 裕度因子
            case '四阶累积量':
                timeDomain_feature['四阶累积量'] = fourth_order_cumulant(signal)  # 四阶累积量
            case '六阶累积量':
                timeDomain_feature['六阶累积量'] = sixth_order_cumulant(signal)  # 六阶累积量

    # timeDomain_feature['最大值'] = max(signal)  # 最大值
    # timeDomain_feature['最小值'] = min(signal)  # 最小值
    # timeDomain_feature['中位数'] = median(signal)  # 中位数
    # timeDomain_feature['峰峰值'] = peak_peak(signal)  # 峰峰值
    # timeDomain_feature['均值'] = mean(signal)  # 均值
    # timeDomain_feature['方差'] = variance(signal)  # 方差
    # timeDomain_feature['标准差'] = std(signal)  # 标准差
    # timeDomain_feature['峰度'] = kurtosis(signal)  # 峰度
    # timeDomain_feature['偏度'] = skew(signal)  # 偏度
    # timeDomain_feature['整流平均值'] = commutation_mean(signal)  # 整流平均值
    # timeDomain_feature['均方根'] = root_mean_square(signal)  # 均方根
    # timeDomain_feature['方根幅值'] = root_amplitude(signal)  # 方根幅值
    # timeDomain_feature['波形因子'] = waveform_factor(signal)  # 波形因子
    # timeDomain_feature['峰值因子'] = peak_factor(signal)  # 峰值因子
    # timeDomain_feature['脉冲因子'] = pulse_factor(signal)  # 脉冲因子
    # timeDomain_feature['裕度因子'] = margin_factor(signal)  # 裕度因子
    # timeDomain_feature['四阶累积量'] = fourth_order_cumulant(signal)  # 四阶累积量
    # timeDomain_feature['六阶累积量'] = sixth_order_cumulant(signal)  # 六阶累积量

    return timeDomain_feature


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


def FreDomain_extraction(signal, features_to_extract, sample_rate=25600):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if signal.ndim == 2:
        signal = signal.reshape(-1)

    fft_vals = np.abs(fft(signal))

    frequencies, t, power_spectrum = scipy.signal.spectrogram(signal, sample_rate, window='hamming', nperseg=256,
                                                              noverlap=128)
    frequencies = frequencies.reshape(-1, 1)
    # SK = SKone(frequencies, power_spectrum)
    SK = kurtosis(fft_vals)
    FreDomain_feature = {}
    for feature in features_to_extract:
        match feature:
            case '重心频率':
                FreDomain_feature['重心频率'] = CF(frequencies, power_spectrum)  # 重心频率
            case '均方频率':
                FreDomain_feature['均方频率'] = MSF(frequencies, power_spectrum)  # 均方频率
            case '均方根频率':
                FreDomain_feature['均方根频率'] = RMSF(frequencies, power_spectrum)  # 均方根频率
            case '频率方差':
                FreDomain_feature['频率方差'] = VF(frequencies, power_spectrum)  # 频率方差
            case '频率标准差':
                FreDomain_feature['频率标准差'] = RVF(frequencies, power_spectrum)  # 频率标准差
            case '谱峭度的均值':
                # FreDomain_feature['谱峭度的均值'] = SKMean(SK)  # 谱峭度的均值
                FreDomain_feature['谱峭度的均值'] = np.mean(SK)
            case '谱峭度的标准差':
                # FreDomain_feature['谱峭度的标准差'] = SKStd(SK)  # 谱峭度的标准差
                FreDomain_feature['谱峭度的标准差'] = np.std(SK)
            case '谱峭度的偏度':
                # FreDomain_feature['谱峭度的偏值'] = SKSkewness(SK)  # 谱峭度的偏值
                FreDomain_feature['谱峭度的偏度'] = skew(SK)
            case '谱峭度的峰度':
                # FreDomain_feature['谱峭度的峰度'] = SKKurtosis(SK)  # 谱峭度的峰度
                FreDomain_feature['谱峭度的峰度'] = np.max(SK)

    # FreDomain_feature['重心频率'] = CF(frequencies, power_spectrum)  # 重心频率
    # FreDomain_feature['均方频率'] = MSF(frequencies, power_spectrum)  # 均方频率
    # FreDomain_feature['均方根频率'] = RMSF(frequencies, power_spectrum)  # 均方根频率
    # FreDomain_feature['频率方差'] = VF(frequencies, power_spectrum)  # 频率方差
    # FreDomain_feature['频率标准差'] = RVF(frequencies, power_spectrum)  # 频率标准差
    # FreDomain_feature['谱峭度的均值'] = SKMean(SK)  # 谱峭度的均值
    # FreDomain_feature['谱峭度的标准差'] = SKStd(SK)  # 谱峭度的标准差
    # FreDomain_feature['谱峭度的偏值'] = SKSkewness(SK)  # 谱峭度的偏值
    # FreDomain_feature['谱峭度的峰度'] = SKKurtosis(SK)  # 谱峭度的峰度

    return FreDomain_feature


def fre_dict2dict(data):
    all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
                              '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
    fre_feature = []
    for row in range(data.shape[0]):
        tmp_data = data[row, :]
        tmp_fre_feature = FreDomain_extraction(tmp_data, all_frequency_features)
        fre_feature.append(tmp_fre_feature)
    values_list = []
    for d in fre_feature:
        # 将当前字典的所有值按顺序添加到二维列表中
        values_list.append(np.array(list(d.values())))
    fre_features_array = np.array(values_list)
    fre_features = {}
    fre_features['重心频率'] = fre_features_array[:, 0]  # 重心频率
    fre_features['均方频率'] = fre_features_array[:, 1]  # 均方频率
    fre_features['均方根频率'] = fre_features_array[:, 2]  # 均方根频率
    fre_features['频率方差'] = fre_features_array[:, 3]  # 频率方差
    fre_features['频率标准差'] = fre_features_array[:, 4]  # 频率标准差
    fre_features['谱峭度的均值'] = fre_features_array[:, 5]  # 谱峭度的均值
    fre_features['谱峭度的标准差'] = fre_features_array[:, 6]  # 谱峭度的标准差
    fre_features['谱峭度的偏值'] = fre_features_array[:, 7]  # 谱峭度的偏值
    fre_features['谱峭度的峰度'] = fre_features_array[:, 8]  # 谱峭度的峰度
    return fre_features


def Get_feature(data_path, time_key_list, fre_key_list):
    all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度',
                         '整流平均值', '均方根', '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子',
                         '四阶累积量', '六阶累积量']
    data_all = scipy.io.loadmat(data_path)
    data_1 = data_all['stage_1']
    data_2 = data_all['stage_2']
    data_3 = data_all['stage_3']
    data_4 = data_all['stage_4']

    time_fea = {}
    time_fea[f'Stage_{1}'] = timeDomain_extraction(data_1, all_time_features)
    time_fea[f'Stage_{2}'] = timeDomain_extraction(data_2, all_time_features)
    time_fea[f'Stage_{3}'] = timeDomain_extraction(data_3, all_time_features)
    time_fea[f'Stage_{4}'] = timeDomain_extraction(data_4, all_time_features)

    fre_fea = {}
    fre_fea[f'Stage_{1}'] = fre_dict2dict(data_1)
    fre_fea[f'Stage_{2}'] = fre_dict2dict(data_2)
    fre_fea[f'Stage_{3}'] = fre_dict2dict(data_3)
    fre_fea[f'Stage_{4}'] = fre_dict2dict(data_4)

    Allstage_array_list = []
    for k in range(4):
        fre_tmp_data = fre_fea[f'Stage_{k + 1}']
        time_tmp_data = time_fea[f'Stage_{k + 1}']
        Perstage_array_list = []
        Perstage_array_list.extend([time_tmp_data[key] for key in time_key_list])
        Perstage_array_list.extend([fre_tmp_data[key] for key in fre_key_list])
        Perstage_array = np.array(Perstage_array_list).T
        Allstage_array_list.append(Perstage_array)
    Allstage_array = np.array(Allstage_array_list)

    return Allstage_array


def GetTest(test_path, time_key_list, fre_key_list):
    data = scipy.io.loadmat(test_path)
    data = data['test_data']
    time_dict = timeDomain_extraction(data, time_key_list)
    fre_dict = FreDomain_extraction(data, fre_key_list)
    Perstage_array_list = []
    Perstage_array_list.extend([time_dict[key] for key in time_key_list])
    Perstage_array_list.extend([fre_dict[key] for key in fre_key_list])

    Perstage_array = np.array(Perstage_array_list).T.astype(np.float64)
    return Perstage_array


def all_feature_extraction(signal):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    # if signal.ndim == 2:
    #     signal = signal.reshape(-1)

    extr_feature = {}
    extr_feature['最大值'] = max(signal)  # 最大值
    extr_feature['最小值'] = min(signal)  # 最小值
    extr_feature['中位数'] = median(signal)  # 中位数
    extr_feature['峰峰值'] = peak_peak(signal)  # 峰峰值
    extr_feature['均值'] = mean(signal)  # 均值
    extr_feature['方差'] = variance(signal)  # 方差
    extr_feature['标准差'] = std(signal)  # 标准差
    extr_feature['峰度'] = kurtosis(signal, axis=1)  # 峰度
    extr_feature['偏度'] = skew(signal, axis=1)  # 偏度
    extr_feature['整流平均值'] = commutation_mean(signal)  # 整流平均值
    extr_feature['均方根'] = root_mean_square(signal)  # 均方根
    extr_feature['方根幅值'] = root_amplitude(signal)  # 方根幅值
    extr_feature['波形因子'] = waveform_factor(signal)  # 波形因子
    extr_feature['峰值因子'] = peak_factor(signal)  # 峰值因子
    extr_feature['脉冲因子'] = pulse_factor(signal)  # 脉冲因子
    extr_feature['裕度因子'] = margin_factor(signal)  # 裕度因子
    extr_feature['四阶累积量'] = fourth_order_cumulant(signal)  # 四阶累积量
    extr_feature['六阶累积量'] = sixth_order_cumulant(signal)  # 六阶累积量

    def fre_domain_extraction(signal, sample_rate=25600):
        if not isinstance(signal, np.ndarray):
            signal = np.array(signal)

        # if signal.ndim == 2:
        #     signal = signal.reshape(-1)

        frequencies, t, power_spectrum = scipy.signal.spectrogram(signal, sample_rate, window='hamming', nperseg=256,
                                                                  noverlap=128)
        frequencies = frequencies.reshape(-1, 1)

        SK = SKone(frequencies, power_spectrum)
        FreDomain_feature = {}
        FreDomain_feature['重心频率'] = CF(frequencies, power_spectrum)  # 重心频率
        FreDomain_feature['均方频率'] = MSF(frequencies, power_spectrum)  # 均方频率
        FreDomain_feature['均方根频率'] = RMSF(frequencies, power_spectrum)  # 均方根频率
        FreDomain_feature['频率方差'] = VF(frequencies, power_spectrum)  # 频率方差
        FreDomain_feature['频率标准差'] = RVF(frequencies, power_spectrum)  # 频率标准差
        FreDomain_feature['谱峭度的均值'] = SKMean(SK)  # 谱峭度的均值
        FreDomain_feature['谱峭度的标准差'] = SKStd(SK)  # 谱峭度的标准差
        FreDomain_feature['谱峭度的偏值'] = SKSkewness(SK)  # 谱峭度的偏值
        FreDomain_feature['谱峭度的峰度'] = SKKurtosis(SK)  # 谱峭度的峰度

        return FreDomain_feature
    fre_feature = []
    for row in range(signal.shape[0]):
        tmp_data = signal[row, :]
        tmp_fre_feature = fre_domain_extraction(tmp_data)
        fre_feature.append(tmp_fre_feature)
    values_list = []
    for d in fre_feature:
        # 将当前字典的所有值按顺序添加到二维列表中
        values_list.append(np.array(list(d.values())))
    fre_features_array = np.array(values_list)
    extr_feature['重心频率'] = fre_features_array[:, 0]  # 重心频率
    extr_feature['均方频率'] = fre_features_array[:, 1]  # 均方频率
    extr_feature['均方根频率'] = fre_features_array[:, 2]  # 均方根频率
    extr_feature['频率方差'] = fre_features_array[:, 3]  # 频率方差
    extr_feature['频率标准差'] = fre_features_array[:, 4]  # 频率标准差
    extr_feature['谱峭度的均值'] = fre_features_array[:, 5]  # 谱峭度的均值
    extr_feature['谱峭度的标准差'] = fre_features_array[:, 6]  # 谱峭度的标准差
    extr_feature['谱峭度的偏值'] = fre_features_array[:, 7]  # 谱峭度的偏值
    extr_feature['谱峭度的峰度'] = fre_features_array[:, 8]  # 谱峭度的峰度
    return extr_feature


# 用于多传感器健康评估获取输入特征数据
def get_test_multiple_sensor(data_all, sensor1_key_list, sensor2_key_list, sensor3_key_list):

    print(data_all.shape)
    data_sensor1 = data_all[:, 0].reshape(1, -1)
    data_sensor2 = data_all[:, 3].reshape(1, -1)
    data_sensor3 = data_all[:, 6].reshape(1, -1)

    sensor1_dict = all_feature_extraction(data_sensor1)
    sensor2_dict = all_feature_extraction(data_sensor2)
    sensor3_dict = all_feature_extraction(data_sensor3)
    Perstage_array_list = []
    Perstage_array_list.extend([sensor1_dict[key] for key in sensor1_key_list])
    Perstage_array_list.extend([sensor2_dict[key] for key in sensor2_key_list])
    Perstage_array_list.extend([sensor3_dict[key] for key in sensor3_key_list])

    for index, item in enumerate(Perstage_array_list):
        if not isinstance(item, np.ndarray):
            new_item = np.array(list((item, )))
            Perstage_array_list[index] = new_item
    print(Perstage_array_list)
    Perstage_array = np.array(Perstage_array_list).astype(np.float64)

    return Perstage_array
