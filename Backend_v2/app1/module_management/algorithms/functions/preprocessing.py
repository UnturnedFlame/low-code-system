import os
import random

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import joblib
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from scipy.interpolate import PchipInterpolator, interp1d, lagrange, CubicSpline
from scipy.io import savemat

from app1.module_management.algorithms.functions.feature_extraction import timeDomain_extraction, FreDomain_extraction

output_root_dir = r'app1/module_management/algorithms/functions/preprocessing_results'


# 小波变换
def wavelet_denoise(data, wavelet='db1', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    uthresh = sigma * np.sqrt(2 * np.log(len(data)))

    denoised_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]

    denoised_data = pywt.waverec(denoised_coeffs, wavelet)

    return denoised_data


def wavelet_denoise_signal(raw_data, filename):
    denoised_data = wavelet_denoise(raw_data)

    if '.png' not in filename:
        filename += '.png'
    save_path = os.path.join(r'app1/module_management/algorithms/functions/'
                            r'preprocessing_results/wavelet_trans/single_signal/', filename)
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 12

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(raw_data.flatten())
    plt.title(f'原始数据')

    plt.subplot(2, 1, 2)
    plt.plot(denoised_data.flatten())
    plt.title(f'去噪后数据')

    plt.tight_layout()
    plt.savefig(save_path)

    results = {'figure_path': save_path, 'denoised_data': denoised_data}

    return results


# 针对四个阶段时序数据的小波变换
def wavelet_denoise_four_stages(mat_data, filename):
    stages = ['stage_1', 'stage_2', 'stage_3', 'stage_4']
    dir_path = os.path.join(r'app1/module_management/algorithms/functions/'
                            r'preprocessing_results/wavelet_trans/four_stages/', filename)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    results = {'all_save_paths': {}, 'denoised_datas': {}}

    # Process and plot each stage
    for stage in stages:
        save_path = os.path.join(dir_path, stage + '.png')
        results['all_save_paths'][stage] = save_path
        example_data = mat_data.get(stage)
        if example_data is not None:
            example_data = example_data.flatten()
        denoised_data = wavelet_denoise(example_data)
        results['denoised_datas'][stage] = denoised_data
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(example_data)
        plt.title(f'原始 {stage} 数据')

        plt.subplot(2, 1, 2)
        plt.plot(denoised_data)
        plt.title(f'去噪后 {stage} 数据')

        plt.tight_layout()
        plt.savefig(save_path)

    return results


def get_filetype(datafile):
    if datafile is not None and len(datafile) > 0:
        return os.path.basename(datafile).split('.')[1]


# 制造有缺失值的数据
def make_data(data: np.ndarray):
    while (True):
        missing_value_start = random.choice(range(data.shape[1]))
        if missing_value_start < data.shape[1] - 10:
            break
    missing_value_end = missing_value_start + 10

    new_data = data.copy()
    new_data[0, missing_value_start: missing_value_end] = np.nan

    return new_data, missing_value_start, missing_value_end


# 输入信号为一维数组size：(length,)
def plot_interpolation(raw_data: np.ndarray, interpolated_data: np.ndarray, save_path):
    if len(raw_data.shape) == 2:
        raw_data = raw_data[0, :]
    if len(interpolated_data.shape) == 2:
        interpolated_data = interpolated_data[0, :]

    # 设置全局字体属性，这里以SimHei为例
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['font.size'] = 30

    # 设置label字体大小
    font = FontProperties(size=25)

    # 图例展示区间
    # 使用numpy.isnan找到缺失值的布尔数组
    is_nan = np.isnan(raw_data)

    # 找到第一个缺失值的下标
    start_missing = np.where(is_nan)[0][0] if np.any(is_nan) else None

    # 找到最后一个缺失值的下标
    end_missing = np.where(is_nan)[0][-1] if np.any(is_nan) else None

    start = start_missing - 50
    end = end_missing + 50

    raw_data_display = raw_data[start:end]
    interpolated_data_display = interpolated_data[start:end]

    plt.figure(figsize=(20, 10))
    # 绘制测试样本散点图
    plt.scatter(np.array(range(start, end)), raw_data_display, c='blue', marker='o', label='原始信号样本')

    # 绘制模拟样本散点图，跳过缺失值
    valid_indices = ~np.isnan(interpolated_data_display)
    plt.scatter(np.array(range(start, end))[valid_indices], interpolated_data_display[valid_indices], c='red',
                marker='x',
                label='插补后信号样本')

    # 标注缺失值的区间
    # plt.axvspan(start_missing, end_missing, alpha=0.3, color='yellow', label='缺失值')

    # 添加图例
    plt.legend(prop=font)

    # 设置x轴和y轴的标签
    plt.xlabel('时间点')
    plt.ylabel('采样值')

    # 显示图形
    plt.savefig(save_path)


# 邻近值插补
def neighboring_values_interpolation_for_signal(data: np.ndarray, filename):
    """

    :return: .mat file with single signal and  a .png file
    """
    """临近值插补"""
    nan_indices = np.isnan(data)

    # 使用前向临近值插补
    array_filled_forward = data.copy()
    for i in range(1, len(data)):
        if np.isnan(array_filled_forward[i]):
            array_filled_forward[i] = array_filled_forward[i - 1]  # 用前一个值填补

    # 使用后向临近值插补
    array_filled_backward = data.copy()
    for i in range(len(data) - 2, -1, -1):
        if np.isnan(array_filled_backward[i]):
            array_filled_backward[i] = array_filled_backward[i + 1]  # 用后一个值填补

    # 将前向插补和后向插补的结果结合起来
    array_filled = np.where(np.isnan(array_filled_forward), array_filled_backward, array_filled_forward)

    # 指定保存路径
    save_path = os.path.join(output_root_dir, 'neighboring_values_interpolation', filename + '_interpolated.mat')
    figure_path = os.path.join(output_root_dir, 'neighboring_values_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, array_filled, figure_path)

    # 保存插补后的MAT文件
    savemat(save_path, {'data': array_filled}, format='4')

    return save_path, figure_path


def fill_missing_with_pchip(column, index):
    """
    :param column:
    :param index:
    :return:
    """
    # 仅对数值型数据应用np.isfinite
    if pd.api.types.is_numeric_dtype(column):
        finite_values = column[np.isfinite(column)]
        if len(finite_values) > 1:  # 确保至少有两个有限数值以进行插值
            # 创建PCHIP插值器
            pchip = PchipInterpolator(index[np.isfinite(column)], finite_values)
            # 填充缺失值
            return pchip(index)
        else:
            print("Not enough valid data points to interpolate.")
            return column  # 返回原始列，因为没有足够的数据进行插值
    else:
        print("Column contains non-numeric data, skipping interpolation.")
        return column  # 返回原始列，因为包含非数值数据


def polynomial_interpolation(input_file):
    """
    多项式插值算法
    :param input_file: input file path
    :return: output file path
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)
    # 对第二列之后的每一列进行插补
    for column_name in df.columns[1:]:  # 从第二列开始
        df[column_name] = fill_missing_with_pchip(df[column_name], df.index)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


# 对信号的多项式插值
def polynomial_interpolation_for_signal(data: np.ndarray, filename):
    """
    对信号的多项式插值
    :param data:
    :param filename:
    :return:
    """
    nan_indices = np.isnan(data)
    non_nan_indices = ~nan_indices

    x_non_nan = np.arange(len(data))[non_nan_indices]
    y_non_nan = data[non_nan_indices]

    interpolator = interp1d(x_non_nan, y_non_nan, kind='linear', fill_value='extrapolate')
    array_filled = data.copy()
    array_filled[nan_indices] = interpolator(np.arange(len(data))[nan_indices])

    # 指定保存路径
    save_path = os.path.join(output_root_dir, 'polynomial_interpolation', filename + '_interpolated.mat')
    figure_path = os.path.join(output_root_dir, 'polynomial_interpolation', filename + '_interpolated.png')

    savemat(save_path, {'data': array_filled}, format='4')
    plot_interpolation(data, array_filled, figure_path)

    return save_path, figure_path


# 对信号的拉格朗日插值
def lagrange_interpolation_for_signal(data: np.ndarray, filename):
    # 定义插值函数
    def lagrange_insert(arr):
        n = len(arr)
        for i in range(n):
            if np.isnan(arr[i]):
                left, right = i - 1, i + 1
                while np.isnan(arr[left]) and left >= 0:
                    left -= 1
                while np.isnan(arr[right]) and right < n:
                    right += 1
                if left < 0 or right >= n:
                    continue
                arr[i] = lagrange([left, right], [arr[left], arr[right]])(i)
        return arr

    interpolated_data = data.copy()
    # 对数据进行插值处理
    for i in range(interpolated_data.shape[0]):
        interpolated_data[i] = lagrange_insert(interpolated_data[i])

    save_path = os.path.join(output_root_dir, 'lagrange_interpolation', filename + '_interpolated.mat')
    savemat(save_path, {'data': data}, format='4')

    figure_path = os.path.join(output_root_dir, 'lagrange_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, interpolated_data, figure_path)

    return save_path, figure_path


# 对信号的双三次插值
def bicubic_interpolation_for_signal(data: np.ndarray, filename):
    nan_indices = np.isnan(data)
    non_nan_indices = ~nan_indices

    x_non_nan = np.arange(len(data))[non_nan_indices]
    y_non_nan = data[non_nan_indices]

    cs = CubicSpline(x_non_nan, y_non_nan, bc_type='natural')
    array_filled = data.copy()
    array_filled[nan_indices] = cs(np.arange(len(data))[nan_indices])

    save_path = os.path.join(output_root_dir, 'bicubic_interpolation', filename + '_interpolated.mat')
    figure_path = os.path.join(output_root_dir, 'bicubic_interpolation', filename + '_interpolated.png')

    savemat(save_path, {'data': array_filled}, format='4')
    plot_interpolation(data, array_filled, figure_path)

    return save_path, figure_path


# 定义一个函数，用于对指定列进行双三次插值
def cubic_spline_interpolation(df, column):
    # 检查列是否为数值型
    if pd.api.types.is_numeric_dtype(df[column]):
        # 获取数值型列的非NaN值
        valid_mask = ~df[column].isnull()
        x = np.arange(len(df))  # 创建一个与DataFrame长度相同的索引数组
        y = df[column][valid_mask].values  # 获取有效的y值

        # 如果有效y值的数量大于1，则进行插值
        if len(y) > 1:
            # 创建插值函数
            interp_func = interp1d(x[valid_mask], y, kind='cubic',
                                   bounds_error=False, fill_value="extrapolate")
            # 应用插值函数
            df[column] = interp_func(x)
        else:
            print(f"Not enough valid data points to interpolate for column '{column}'.")
    else:
        print(f"Column '{column}' is not numeric, skipping interpolation.")


def bicubic_interpolation(input_file):
    """
    双三次插值算法
    :param input_file: input file path
    :return: output file path
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)

    # 从第二列开始遍历DataFrame的每一列
    for column in df.columns[1:]:
        cubic_spline_interpolation(df, column)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


def lagrange_insert(s, n, k=3):
    y = s.reindex(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


def lagrange_interpolation(input_file):
    """
    拉格朗日插值算法
    :param input_file: input file path
    :return: output file path
    """
    df = pd.read_excel(input_file)
    for i in df.columns:
        for j in range(len(df)):
            if (df[i].isnull())[j]:
                df[i][j] = lagrange_insert(df[i], j)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


def newton_insert(s, n, k=4):
    y = s.reindex(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))
    # df['时间'] = df['时间'].dt.strftime('%Y-%m-%d %H:%M')
    x = pd.Series(list(range(n - k, n)) + list(range(n + 1, n + 1 + k)))

    y = y[y.notnull()]
    print("-------------------------------------------------------------------------")
    print(y)
    print(x)

    # 现在, 温度和对应的序号都有了
    # 差分
    def divided_diff(x, y):
        if len(y) == 1:
            return y.iloc[0]
        else:
            return (divided_diff(x[1:], y[1:]) - divided_diff(x[:-1], y[:-1])) / (x.iloc[-1] - x.iloc[0])

    # 构造插值多项式
    def newton_polynomial(x, y):
        if len(y) == 1:
            return y.iloc[0]
        else:
            return newton_polynomial(x[1:], y[:-1]) + divided_diff(x, y) * np.prod(np.array(x.iloc[0]) - np.array(x))

    interpolated_value = newton_polynomial(x, y)
    # return interpolated_value
    return round(interpolated_value, 6)


def newton_interpolation(input_file):
    """
    牛顿插值算法
    :param input_file: input file path
    :return: output file path
    """
    df = pd.read_excel(input_file)
    for i in df.columns:
        for j in range(len(df)):
            if (df[i].isnull())[j]:
                df[i][j] = newton_insert(df[i], j)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


def newton_interpolation_for_signal(data: np.ndarray, filename):
    """
    对于一维信号的牛顿插值
    :param filename:
    :param data:
    :return: save_path, figure_path
    """

    # 定义牛顿插值函数
    def NewtonInsert(arr):
        n = len(arr)
        for i in range(n):
            if np.isnan(arr[i]):
                left, right = i - 1, i + 1
                while np.isnan(arr[left]) and left >= 0:
                    left -= 1
                while np.isnan(arr[right]) and right < n:
                    right += 1
                if left < 0 or right >= n:
                    continue

                # 差分
                def divided_diff(x, y):
                    if len(y) == 1:
                        return y[0]
                    else:
                        return (divided_diff(x[1:], y[1:]) - divided_diff(x[:-1], y[:-1])) / (x[-1] - x[0])

                # 构造插值多项式
                def newton_polynomial(x, y):
                    if len(y) == 1:
                        return y[0]
                    else:
                        return newton_polynomial(x[1:], y[:-1]) + divided_diff(x, y) * np.prod(
                            np.array(x[0]) - np.array(x))

                # 获取插值结果
                interpolated_value = newton_polynomial(np.array([left, right]), np.array([arr[left], arr[right]]))
                arr[i] = interpolated_value

        return arr

    interpolated_data = data.copy()
    # 对数据进行插值处理
    for i in range(interpolated_data.shape[0]):
        interpolated_data[i] = NewtonInsert(interpolated_data[i])

    save_path = os.path.join(output_root_dir, 'newton_interpolation', filename + '_interpolated.mat')
    savemat(save_path, {'data': data}, format='4')

    figure_path = os.path.join(output_root_dir, 'newton_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, interpolated_data, figure_path)

    return save_path, figure_path


def linear_insert(s, n, k=1):
    # 获取相邻点的值
    y = s.reindex(range(n - k, n + k + 1))
    y = y.dropna()  # 去除空值
    if y.empty:  # 处理边界情况
        return np.nan

    x = y.index
    # 计算插值结果
    interpolated_value = np.interp(n, x, y)
    return interpolated_value


def linear_interpolation(input_file):
    """
    线性插值
    :param input_file: input file path
    :return: output file path
    """
    df = pd.read_excel(input_file)
    # 遍历数据框中的每一列，对缺失值进行插值
    for column in df.columns:
        for index, value in df[column].items():
            if pd.isnull(value):
                df.at[index, column] = linear_insert(df[column], index)

    basename = os.path.basename(input_file)
    filename = basename.split('.')[0]
    save_dir = os.path.join(output_root_dir, 'linear_interpolation', filename)
    save_raw_dir, save_result_dir = os.path.join(save_dir, 'raw_'), os.path.join(save_dir, 'result_')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_filename = os.path.join(save_dir, basename)

    # 将插补后的数据保存到新的Excel文件
    df.to_excel(output_filename, index=False)
    # 绘制波形图并保存
    waveform_save_path = {'原始数据': waveform_drawing(input_file, save_raw_dir),
                          '结果数据': waveform_drawing(output_filename, save_result_dir)}

    return waveform_save_path, output_filename


def linear_interpolation_for_signal(data: np.ndarray, filename):
    """
    对于一维信号的线性插值
    :param filename:
    :param data:
    :return:
    """
    interpolated_data = data.copy()
    # 对于每一个数据向量，进行插值处理
    for key in interpolated_data:
        if not key.startswith('__'):
            array = interpolated_data[key].flatten()  # 将数据展开成一维数组

            # 获取当前行的数据
            y = array
            x = np.arange(len(y))

            # 找到非NaN值和NaN值的索引
            nan_idx = np.isnan(y)
            not_nan_idx = ~nan_idx

            # 如果行中有NaN值且非NaN值数量大于等于2，才进行插值
            if nan_idx.any() and np.sum(not_nan_idx) >= 2:
                # 使用非NaN值进行线性插值
                interp_func = interp1d(x[not_nan_idx], y[not_nan_idx], kind='linear', fill_value="extrapolate")

                # 用插值结果替换NaN值
                y[nan_idx] = interp_func(x[nan_idx])
                array = y

            # 更新数据
            interpolated_data[key] = array.reshape(1, -1)  # 确保数据仍然是一行
    save_path = os.path.join(output_root_dir, 'linear_interpolation', filename + '_interpolated.mat')
    savemat(save_path, {'data': interpolated_data}, format='4')

    figure_path = os.path.join(output_root_dir, 'linear_interpolation', filename + '_interpolated.png')
    plot_interpolation(data, interpolated_data, figure_path)

    return save_path, figure_path


def waveform_drawing(datafile, output_dir):
    # 设置全局字体属性，这里以SimHei为例
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['font.size'] = 30

    # 设置label字体大小
    font = FontProperties(size=25)

    # 读取Excel文件
    data = pd.read_excel(datafile, parse_dates=['时间'])
    data['时间'] = pd.to_datetime(data['时间'])

    # data = pd.read_excel('data/Insert.xlsx')
    data = data.tail(1000)

    # sampled_data = data.groupby(pd.Grouper(key='时间', freq='M')).apply(
    #     lambda x: x.sample(n=20, replace=False)).reset_index(drop=True)

    # 提取时间点和各项性能指标
    time = data.iloc[:, 1]
    performance_indicators = (data.iloc[:, 2: 5], data.iloc[:, 5: 8], data.iloc[:, 8: 11], data.iloc[:, 11:])
    labels = ('发电机温度随时间变化折线图', '电网电压随时间变化折线图', '电网电流随时间变化折线图',
              '其他性能指标随时间变化折线图')
    out_feature_name = ('发电机温度', '电网电压', '电网电流', '其他性能指标')
    out_features = (output_dir + 'temperature.png', output_dir + 'grid_voltage.png', output_dir + 'current.png',
                    output_dir + 'other.png')
    save_path = {k: v for (k, v) in zip(out_feature_name, out_features)}

    # print(time)
    # print(performance_indicators)

    # 绘制折线图

    for (performance, label, path) in zip(performance_indicators, labels, out_features):
        plt.figure(figsize=(32, 8))
        for column in performance.columns:
            plt.plot(time, performance[column], label=column)

        # 使用AutoDateLocator自动选择最佳的时间间隔
        locator = mdates.AutoDateLocator()
        plt.gca().xaxis.set_major_locator(locator)

        plt.xlabel('时间点')
        plt.ylabel('性能指标')
        plt.title(label)
        plt.legend(prop=font)
        plt.grid(True)
        plt.savefig(path)
    plt.cla()
    return save_path


def extract_time_domain_for_three_dims(datafile):
    """
    读取三维数据文件并进行人工时域特征提取
    :param datafile:
    :return: filepath of extracted time domain features
    """
    file_type = get_filetype(datafile)
    if file_type == 'csv':
        data = pd.read_csv(datafile)
    elif file_type == 'npy':
        data = np.load(datafile)
    else:
        return 'Invalid file'

    features = []
    columns = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            time_domain_feature = timeDomain_extraction(data[i, j, :])
            features.append(time_domain_feature.values())
            if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                columns = list(time_domain_feature.keys())

    df = pd.DataFrame(data=features, columns=columns)
    out_filename = 'time_' + os.path.basename(datafile).split('.')[0] + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_domain',
        out_filename)
    df.to_csv(output_file, index=False)
    return output_file


def extract_frequency_domain_for_three_dims(datafile):
    """
    读取三维数据文件并进行人工频域特征提取
    :param datafile:
    :return: a dataframe with features include freDomain of each row of the datafile
    """
    file_type = get_filetype(datafile)
    if file_type == 'csv':
        data = pd.read_csv(datafile)
    elif file_type == 'npy':
        data = np.load(datafile)
    else:
        return 'Invalid file'

    features = []
    columns = []

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            fre_domain_feature = FreDomain_extraction(data[i, j, :], sample_rate=4000)
            features.append(fre_domain_feature.values())
            if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                columns = list(fre_domain_feature.keys())

    df = pd.DataFrame(data=features, columns=columns)
    out_filename = 'freq_' + os.path.basename(datafile).split('.')[0] + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_domain',
        out_filename)
    df.to_csv(output_file, index=False)
    return output_file


def extract_features_for_three_dims(datafile, features_to_extract):
    """
    读取三维数据文件并进行人工时频域特征提取
    :param features_to_extract: {time_domain: ['', '', ...], frequency_domain: ['', '', ...]}
    :param datafile:
    :return:
    """
    file_type = get_filetype(datafile)
    if file_type == 'csv':
        data = pd.read_csv(datafile)
    elif file_type == 'npy':
        data = np.load(datafile)
    else:
        return 'Invalid file'

    # all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度', '整流平均值',
    #                      '均方根',
    #                      '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子', '四阶累积量', '六阶累积量']
    # all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
    #                           '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']

    def extract_frequency_domain_features(data, frequency_domain_features):
        features = []
        columns = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                fre_domain_feature = FreDomain_extraction(data[i, j, :], frequency_domain_features)
                features.append(fre_domain_feature.values())
                if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                    columns = list(fre_domain_feature.keys())

        return pd.DataFrame(data=features, columns=columns)

    def extract_time_domain_features(data, time_domain_features):
        features = []
        columns = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                time_domain_feature = timeDomain_extraction(data[i, j, :], time_domain_features)
                features.append(time_domain_feature.values())
                if i == data.shape[0] - 1 and j == data.shape[1] - 1:
                    columns = list(time_domain_feature.keys())
        return pd.DataFrame(data=features, columns=columns)

    time_domain_features = None
    frequency_domain_features = None

    for domain, features in features_to_extract.items():
        if features:
            if domain == 'time_domain':
                time_domain_features = extract_time_domain_features(data, features)
            else:
                frequency_domain_features = extract_frequency_domain_features(data, features)

    features = pd.concat([time_domain_features, frequency_domain_features], axis=1)
    out_filename = os.path.basename(datafile).split('.')[0] + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_frequency_domain',
        out_filename)
    features.to_csv(output_file, index=False)
    return output_file


def extract_signal_features(input_data: np.ndarray, features_to_extract, filename=None, save=False):
    """
    读取三维数据文件并进行人工时频域特征提取
    :param save: save the features or not
    :param filename: the filename of the input_data
    :param input_data: input data
    :param features_to_extract: {time_domain: feature list, frequency_domain: feature list}
    :return:
    """

    # all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度', '整流平均值',
    #                      '均方根',
    #                      '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子', '四阶累积量', '六阶累积量']
    # all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
    #                           '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']

    time_domain_features = None
    frequency_domain_features = None

    all_features = []
    for _, features in features_to_extract.items():
        all_features.extend(features)

        if len(input_data.shape) == 2:
            new_data = input_data[0, :].copy()
        else:
            new_data = input_data.copy()

        for domain, features in features_to_extract.items():
            if features:
                if domain == 'time_domain':
                    time_domain_features = timeDomain_extraction(new_data, features)
                    values = [list(time_domain_features.values())]
                    time_domain_features = pd.DataFrame(data=values, columns=features)
                else:
                    frequency_domain_features = FreDomain_extraction(new_data, features)
                    values = [list(frequency_domain_features.values())]
                    frequency_domain_features = pd.DataFrame(data=values, columns=features)
    features_extracted = pd.concat([time_domain_features, frequency_domain_features], axis=1)
    features_extracted_group_by_sensor = {'sensor_1': features_extracted.iloc[0].tolist()}

    # print('features_extracted: ', features_extracted)
    if save and filename is not None:
        out_filename = filename + '.csv'
        output_file = os.path.join(
            'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_frequency_domain',
            out_filename)
        features_extracted.to_csv(output_file, index=False)
        return output_file, {'features_extracted_group_by_sensor': features_extracted_group_by_sensor,
                             'features_name': all_features}
    else:
        return features_extracted.iloc[0].tolist()


def extract_features_with_multiple_sensors(input_data: np.ndarray, features_to_extract, filename):
    """
    提取多传感器信号的特征
    :param input_data: input data
    :param features_to_extract: features to extract
    :param filename: the filename of the raw signal data
    :return:
    """

    # Parameters for frame extraction
    frame_length = 2048
    step_size = 8

    # Feature columns time_feature_columns = ['mean', 'var', 'std', 'skewness', 'kurtosis', 'cumulant_4th',
    # 'cumulant_6th', 'max', 'min', 'median', 'peak_to_peak', 'rectified_mean', 'rms', 'root_amplitude',
    # 'waveform_factor', 'peak_factor', 'impulse_factor', 'margin_factor'] freq_feature_columns = ['centroid_freq',
    # 'msf', 'rms_freq', 'freq_variance', 'freq_std', 'spectral_kurt_mean', 'spectral_kurt_peak']
    time_feature_columns = ['均值', '方差', '标准差', '偏度', '峰度', '四阶累积量', '六阶累积量', '最大值', '最小值',
                            '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值', '波形因子', '峰值因子', '脉冲因子',
                            '裕度因子']
    freq_feature_columns = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
                            '谱峭度的峰度']

    all_features = []
    for _, features in features_to_extract.items():
        all_features.extend(features)

    # Extract features for each sensor
    sensor_names = ['X维力(N)', 'Y维力(N)', 'Z维力(N)', 'X维振动(g)', 'Y维振动(g)', 'Z维振动(g)', 'AE-RMS (V)']

    extracted_features_group_by_sensor = {sensor_name: [] for sensor_name in sensor_names}

    combined_columns = []
    for sensor in sensor_names:
        combined_columns.extend([f"{sensor}_{feature}" for feature in all_features])

    # DataFrame to store the combined features
    combined_features_df = pd.DataFrame(columns=combined_columns)
    # Extract and combine features for each frame
    num_frames = (input_data.shape[0] - frame_length) // step_size + 1
    for frame_idx in range(num_frames):
        combined_features = []
        for i, sensor_name in enumerate(sensor_names):
            frame = input_data[frame_idx * step_size:frame_idx * step_size + frame_length, i]
            # time_features = compute_time_domain_features(frame)
            # freq_features = compute_frequency_domain_features(frame, frame_length)
            features = extract_signal_features(frame, features_to_extract, save=False)
            extracted_features_group_by_sensor[sensor_name].extend(features)
            combined_features.extend(features)
        # combined_features.append(1)
        combined_features_df.loc[frame_idx] = combined_features
    out_filename = filename + '.csv'
    output_file = os.path.join(
        'app1/module_management/algorithms/functions/preprocessing_results/feature_extraction/time_frequency_domain',
        out_filename)
    combined_features_df.to_csv(output_file, index=False)
    return output_file, {'features_extracted_group_by_sensor': extracted_features_group_by_sensor,
                         'features_name': all_features}
    # Save the DataFrame to a CSV file
    # output_path = 'mutli_features.csv'
    # combined_features_df['label'] = lable_value
    # return combined_features_df


def plot_signal(example, filename, multiple_sensor=False):
    # example, filename = load_data(example_filepath)
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.figure(figsize=(20, 10))

    if not multiple_sensor:
        if len(example.shape) == 2:
            example = example[0, :]
        else:
            example = example
        plt.plot(example)
        plt.title('信号波形图')

        plt.xlabel('采样点', fontsize=18)
        plt.ylabel('信号值', fontsize=18)
    else:

        # 创建图形和子图
        # plt.figure(figsize=(20, 10))  # 设置图形的大小
        num_sensors = example.shape[1]  # 获取传感器的数量
        for i in range(num_sensors):
            plt.subplot(num_sensors, 1, i + 1)  # 创建子图，num_sensors 行 1 列，当前是第 i+1 个子图
            plt.plot(example[:, i])  # 绘制第 i 个传感器的信号
            plt.title(f'Sensor {i + 1}')  # 设置子图的标题
            plt.xlabel('采样点', fontsize=18)  # 设置 x 轴标签
            plt.ylabel('信号值', fontsize=18)  # 设置 y 轴标签

        # 调整子图之间的间距
        plt.title('信号波形图')
        plt.tight_layout()
    save_path = 'app1/module_management/algorithms/functions/fault_diagnosis/' + filename + '.png'

    plt.savefig(save_path)

    return save_path


choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', '峰峰值', '六阶累积量', '均值', '四阶累积量',
                   '最小值']

choose_features_multiple = ['X维力(N)_六阶累积量', 'X维力(N)_峰峰值', 'X维力(N)_重心频率', 'X维力(N)_最大值', 'X维力(N)_四阶累积量',
                            'X维力(N)_方差', 'X维力(N)_裕度因子', 'X维力(N)_标准差', 'X维力(N)_均方根', 'X维力(N)_方根幅值']


# 无量纲化, 主要用于SVM等需要标准化的机器学习算法的预处理
def scaler(input_data: pd.DataFrame, features_group_by_sensor: dict, option=None, multiple_sensor=False):
    """
    :param features_group_by_sensor:
    :param input_data:
    :param option:
    :return: 标准化后的特征数据
    """
    data_scaled = input_data.copy()  # 实际使用的特征
    data_scaled_display = features_group_by_sensor.copy()  # 用于展示在前端页面中的特征
    if multiple_sensor is None:
        multiple_sensor = False
    if option == 'max_min':
        data_scaler = MinMaxScaler()
    elif option == 'z-score':
        # data_scaler = StandardScaler()

        if not multiple_sensor:
            data_scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/scaler_2.pkl')
            data_scaled[choose_features] = data_scaler.transform(data_scaled[choose_features])

        else:
            data_scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_scaler.pkl')
            data_scaled[choose_features_multiple] = data_scaler.transform(data_scaled[choose_features_multiple])
            index_start = 0
            try:
                for k, v in data_scaled_display.items():
                    features_num = len(v)
                    data_scaled_display[k] = data_scaled.iloc[:, index_start:features_num].to_list()
                    index_start += features_num
            except Exception as e:
                print(str(e))
        return data_scaled, data_scaled_display
    elif option == 'max_abs_scaler':
        data_scaler = MaxAbsScaler()
    else:
        data_scaler = RobustScaler()
    print('data_scaler[data_scaled.columns]: ', data_scaled[data_scaled.columns])
    data_scaled[data_scaled.columns] = data_scaler.fit_transform(data_scaled[data_scaled.columns])
    print('data_scaled: ', data_scaled)

    return data_scaled, data_scaled_display
