import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat

from app1.module_management.algorithms.functions.load_data import load_data

train_data = pd.read_csv('app1/module_management/algorithms/functions/datas/vibration_features_with_labels.csv')

# choose_features = ['std', 'rms', 'var', 'rectified_mean', 'root_amplitude', 'peak_to_peak', 'cumulant_6th', 'mean',
#                    'cumulant_4th', 'min']


choose_features = ['标准差', '均方根', '方差', '整流平均值', '方根幅值', '峰峰值', '六阶累积量', '均值',
                   '四阶累积量', '最小值']
choose_features_multiple = ['X维力(N)_六阶累积量', 'X维力(N)_峰峰值', 'X维力(N)_重心频率', 'X维力(N)_最大值', 'X维力(N)_四阶累积量',
                            'X维力(N)_方差', 'X维力(N)_裕度因子', 'X维力(N)_标准差', 'X维力(N)_均方根', 'X维力(N)_方根幅值']


def replace_features_name(data: pd.DataFrame) -> pd.DataFrame:
    # features_in_english = ['mean', 'var', 'std', 'skewness', 'kurtosis', 'cumulant_4th', 'cumulant_6th', 'max',
    # 'min', 'median', 'peak_to_peak', 'rectified_mean', 'rms', 'root_amplitude', 'waveform_factor', 'peak_factor',
    # 'impulse_factor', 'margin_factor'] + ['centroid_freq', 'msf', 'rms_freq', 'freq_variance', 'freq_std',
    # 'spectral_kurt_mean', 'spectral_kurt_std', 'spectral_kurt_peak', 'spectral_kurt_skew']
    names_mapping = {'均值': 'mean', '方差': 'var', '标准差': 'std', '偏度': 'skewness', '峰度': 'kurtosis',
                     '四阶累积量': 'cumulant_4th', '六阶累积量': 'cumulant_6th', '最大值': 'max', '最小值': 'min',
                     '中位数': 'median', '峰峰值': 'peak_to_peak', '整流平均值': 'rectified_mean', '均方根': 'rms',
                     '方根幅值': 'root_amplitude', '波形因子': 'waveform_factor', '峰值因子': 'peak_factor',
                     '脉冲因子': 'impulse_factor', '裕度因子': 'margin_factor', '重心频率': 'centroid_freq',
                     '均方频率': 'msf', '均方根频率': 'rms_freq', '频率方差': 'freq_variance',
                     '频率标准差': 'freq_std', '谱峭度的均值': 'spectral_kurt_mean',
                     '谱峭度的标准值': 'spectral_kurt_std',
                     '谱峭度的峰度': 'spectral_kurt_peak',
                     '谱峭度的偏度': 'spectral_kurt_skew'}

    data.rename(columns=names_mapping, inplace=True)
    return data


def diagnose_with_svc_model(data_with_selected_features, multiple_sensor=False):
    """
    :param multiple_sensor:
    :param data_with_selected_features: 输入样本以及选择的特征
    :return: “0”代表无故障，“1”代表有故障
    """
    example = data_with_selected_features.get('data').copy()
    example_filepath = data_with_selected_features.get('filepath')
    # example = replace_features_name(example)
    if not multiple_sensor:
        # 树模型没有标准化，svc（支持向量机）有
        scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/scaler_2.pkl')
        # 使用训练阶段保存的 StandardScaler 对测试数据进行同样的变换
        # train_data[choose_features] = scaler.transform(train_data[choose_features])
        example[choose_features] = scaler.transform(example[choose_features])
        # 预测结果为“0”代表无故障，“1”代表有故障
        svc_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/svc_model_2.pkl')
        svc_predictions = svc_model.predict(example[choose_features][0:1])[0]
    else:
        scaler = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_scaler.pkl')
        example[choose_features_multiple] = scaler.transform(example[choose_features_multiple])

        svc_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/svc/mutli_svc_model.pkl')
        svc_predictions = svc_model.predict(example[choose_features_multiple][0:1])[0]

    figure_path = plot_diagnosis(example_filepath, multiple_sensor)

    return svc_predictions, figure_path


def diagnose_with_random_forest_model(data_with_selected_features, multiple_sensor=False):
    """
    :return: “0”代表无故障，“1”代表有故障
    """
    example = data_with_selected_features.get('data').copy()
    example_filepath = data_with_selected_features.get('filepath')
    # example = replace_features_name(example)
    if not multiple_sensor:
        random_forest_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/random_forest'
                                          '/random_forest_model_2.pkl')
        random_forest_predictions = random_forest_model.predict(example[choose_features][0:1])
    else:
        random_forest_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/random_forest'
                                          '/mutli_random_forest_model.pkl')
        random_forest_predictions = random_forest_model.predict(example[choose_features_multiple][0:1])
    figure_path = plot_diagnosis(example_filepath, multiple_sensor)

    return random_forest_predictions[0], figure_path


def time_regression(data_with_selected_features, have_fault=0, multiple_sensor=False):
    example = data_with_selected_features.get('data').copy()
    raw_data_filepath = data_with_selected_features.get('filepath')
    # 没有故障预测故障时间
    if have_fault == 0:
        if multiple_sensor:
            # 多传感器预测
            reg_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/regression'
                                    '/mutli_time_reg.pkl')
            reg_predictions = reg_model.predict(example[choose_features_multiple][0:1])
        else:
            # 单传感器预测
            example = replace_features_name(example)
            reg_model = joblib.load('app1/module_management/algorithms/models/fault_diagnosis/regression'
                                    '/single_time_reg.pkl')
            reg_predictions = reg_model.predict(example[choose_features][0:1])
    # 有故障不用预测
    else:
        reg_predictions = [0]

    # 设置字体以支持中文显示
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    # 绘制信号图像
    figure_path = plot_diagnosis(raw_data_filepath, multiple_sensor)

    return reg_predictions[0], figure_path


# 定义GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hx=None):
        # 如果提供了隐藏状态，则使用它；否则自动初始化
        out, h_n = self.gru(x, hx)
        # 取最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :])
        return out, h_n


# 定义LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hx=None):
        # 如果提供了隐藏状态，则使用它；否则自动初始化
        out, (h_n, c_n) = self.lstm(x, hx)
        # 取最后一个时间步的输出用于分类
        out = self.fc(out[:, -1, :])
        return out, (h_n, c_n)


# 模型基本参数
input_size = 2048  # 特征维度，假设为2048
hidden_size = 128  # 隐藏层大小
num_layers = 2  # GRU层数
num_classes = 2  # 类别数
LSTM_weights_path = 'app1/module_management/algorithms/models/fault_diagnosis/lstm/LSTM_model.pth'
GRU_weights_path = 'app1/module_management/algorithms/models/fault_diagnosis/gru/GRU_model.pth'

# 模型初始化
LSTM_model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes)
GRU_model = GRUClassifier(input_size, hidden_size, num_layers, num_classes)

if torch.cuda.is_available():
    state_dict1 = torch.load(GRU_weights_path)
    state_dict2 = torch.load(LSTM_weights_path)
else:
    state_dict1 = torch.load(GRU_weights_path, map_location=torch.device('cpu'))
    state_dict2 = torch.load(LSTM_weights_path, map_location=torch.device('cpu'))

# 使用模型实例加载状态字典
GRU_model.load_state_dict(state_dict1)
LSTM_model.load_state_dict(state_dict2)
# 将模型移动到适当的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRU_model = GRU_model.to(device)
LSTM_model = LSTM_model.to(device)

# 设置为评估模式
GRU_model.eval()
LSTM_model.eval()


# 加载.mat数据
# def load_mat_data(file_path, stage):
#     file_path = file_path  # MATLAB文件路径
#     mat_data = loadmat(file_path)
#     input_data = mat_data[stage]
#     return input_data
def load_mat_data(file_path):
    # file_path = file_path  # MATLAB文件路径
    global test_data
    mat_data = loadmat(file_path)
    for key, value in mat_data.items():
        if isinstance(value, np.ndarray):
            test_data = value
    return test_data


# 选用模型
def diagnose_with_gru_model(data_with_selected_features, multiple_sensor=False):
    """
    以gru模型进行故障诊断
    :return: integer "0" or "1"
    """
    example = data_with_selected_features.get('data')
    example_filepath = data_with_selected_features.get('filepath')
    # 单传感器算法
    dims = len(example.shape)
    with torch.no_grad():
        if dims == 2:
            outputs, h_n = GRU_model(torch.tensor(example[0, :]).reshape(1, 1, -1).to(device))
        else:
            outputs, h_n = GRU_model(torch.tensor(example).reshape(1, 1, -1).to(device))
        _, predicted = torch.max(outputs, 1)

    figure_path = plot_diagnosis(example_filepath, multiple_sensor)
    return predicted.tolist()[0], figure_path


def diagnose_with_lstm_model(data_with_selected_features, multiple_sensor=False):
    """
    以lstm模型进行故障诊断
    :param example:
    :return: integer "0" or "1"
    """
    example = data_with_selected_features.get('data')
    example_filepath = data_with_selected_features.get('filepath')
    # 单传感器算法
    dims = len(example.shape)
    with torch.no_grad():
        if dims == 2:
            outputs, h_n = LSTM_model(torch.tensor(example[0, :]).reshape(1, 1, -1).to(device))
        else:
            outputs, h_n = LSTM_model(torch.tensor(example).reshape(1, 1, -1).to(device))
        _, predicted = torch.max(outputs, 1)
    figure_path = plot_diagnosis(example_filepath, multiple_sensor)
    return predicted.tolist()[0], figure_path


def plot_diagnosis(example_filepath, multiple_sensor=False):

    example, filename = load_data(example_filepath)
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 15

    if not multiple_sensor:
        plt.figure(figsize=(16, 8))
        if len(example.shape) == 2:
            example = example[0, :]
        else:
            example = example
        plt.plot(example)
        plt.title('信号波形图')

        plt.xlabel('采样点')
        plt.ylabel('信号值')
    else:
        plt.figure(figsize=(20, 15))
        # 创建图形和子图  
        # plt.figure(figsize=(20, 10))  # 设置图形的大小
        sensor_names = ['X维力(N)', 'Y维力(N)', 'Z维力(N)', 'X维振动(g)', 'Y维振动(g)', 'Z维振动(g)', 'AE-RMS (V)']
        # 创建图形和子图

        num_sensors = example.shape[1]  # 获取传感器的数量
        for i, sensor_name in enumerate(sensor_names):
            plt.subplot(num_sensors, 1, i + 1)  # 创建子图，num_sensors 行 1 列，当前是第 i+1 个子图
            plt.plot(example[:, i])  # 绘制第 i 个传感器的信号
            plt.title(f'传感器{i + 1}-{sensor_name}')  # 设置子图的标题
            plt.xlabel('时间点', )  # 设置 x 轴标签
            plt.ylabel('信号值', )  # 设置 y 轴标签

        # 调整子图之间的间距
        plt.title('信号波形图')
        plt.tight_layout()
    save_path = 'app1/module_management/algorithms/functions/fault_diagnosis/' + filename + '.png'
        
    plt.savefig(save_path)

    return save_path


if __name__ == '__main__':
    # 训练
    # train_Rf_model()
    # train_SVC()
    # 测试
    # diagnose_with_svc_model()
    # 测试
    file_path = './已划分数据_4阶段.mat'  # MATLAB文件路径
    stage = 'stage_3'

    # predicted = predict_with_model(input_data, GRU_model)
    # print(predicted)
