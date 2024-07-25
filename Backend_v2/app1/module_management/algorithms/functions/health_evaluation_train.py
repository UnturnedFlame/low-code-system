import numpy as np
import skfuzzy as fuzz
import pandas as pd
from sklearn.preprocessing import normalize
import warnings
import os
import matplotlib.pyplot as plt
from app1.module_management.algorithms.functions import feature_extraction
import pickle

criteria = np.array([[1, 3], [1/3, 1]])  # 一级指标权重
b1 = np.array([[1, 3, 1, 5], [1 / 3, 1, 1 / 3, 2], [1, 3, 1, 3], [1 / 5, 1 / 2, 1 / 3, 1]])
b2 = np.array([[1, 4, 2, 5], [1 / 4, 1, 1 / 3, 2], [1 / 2, 3, 1, 3], [1 / 5, 1 / 2, 1 / 3, 1]])  # 二级指标权重
b = [b1, b2]
data_path = 'datas/已划分数据_4阶段.mat'
test_path = 'datas/test.mat'
save_path = 'results'
primary_key_list = ['时域指标', '频域指标']
time_key_list = ['均方根', '峰峰值', '峰度', '偏度']
fre_key_list = ['重心频率', '均方频率', '均方根频率', '频率标准差']
primary_key_list_multiple_sensor = ['力传感器', '振动传感器', '声发射传感器']
sensor1_key_list = ['均方根', '峰峰值', '峰度', '偏度']
sensor2_key_list = ['方差', '方根幅值', '重心频率', '频率方差']
sensor3_key_list = ['均值', '最大值', '最小值', '标准差']
status_names = ['正常', '轻微退化', '严重退化', '失效']
suggestion_dict = {'正常': '设备当前处于正常工作状态，但为了保持其长期稳定运行，建议定期进行全面检查和清洁保养，尤其注意关键部位的润滑维护。同时，操作人员进行培训，确保其了解正确的操作流程和应急措施。',
                   '轻微退化': '设备已经表现出轻微的退化迹象，建议立即进行详细检查，确定退化的具体部位和原因。针对发现的问题进行局部维修或更换磨损零部件，同时加密检查频率，以防止进一步退化并恢复设备的正常运行状态。',
                   '严重退化': '设备出现严重退化，建议立即停机进行全面检查和评估。制定详细的维修计划，包括更换损坏或老化的关键零部件，并检查设备的所有连接和系统。维修后进行全面测试，确保设备恢复到正常工作状态，同时考虑加强未来的预防性维护措施。',
                   '失效': '设备已完全失效，建议立即停机并进行全面诊断，确定失效原因和范围。与设备制造商或专业维修团队合作，制定和实施详细的修复计划，包括更换损坏的核心部件和系统。修复后进行严格的测试和验收，确保设备完全恢复功能，并制定改进的维护计划以防止类似问题再次发生。'}
U = np.arange(0, 10000, 0.01)


def create_functions(U, num_second_level, data_path, time_key, fre_key, status_names):
    # 创建存储函数的字典
    data_all = feature_extraction.Get_feature(data_path, time_key, fre_key)
    num_status = len(status_names)
    functions = {}
    # 循环创建函数B
    for j in range(num_status):
        for k in range(num_second_level):
            def func_template(j=j, k=k):  # j状态，k指标
                return fuzz.trimf(U, [np.min(data_all[j, :, k]), np.mean(data_all[j, :, k]), np.max(data_all[j, :, k])])

            # 使用格式化字符串创建唯一的函数名称
            func_name = f'func_{j}_{k}'
            # 将函数存储到字典中
            functions[func_name] = func_template(j, k)
    return functions, num_status, status_names


# 多传感器
def create_functions_multiple_sensor(U, num_second_level, data_path, sensor1_key, sensor2_key, sensor3_key, status_names):
    # 创建存储函数的字典
    data_all = feature_extraction.Get_feature(data_path, sensor1_key, sensor2_key, sensor3_key)
    num_status = len(status_names)
    functions = {}
    # 循环创建函数B
    for j in range(num_status):
        for k in range(num_second_level):
            def func_template(j=j, k=k):  # j状态，k指标
                return fuzz.trimf(U, [np.min(data_all[j, :, k]), np.mean(data_all[j, :, k]), np.max(data_all[j, :, k])])

            # 使用格式化字符串创建唯一的函数名称
            func_name = f'func_{j}_{k}'
            # 将函数存储到字典中
            functions[func_name] = func_template(j, k)
    return functions, num_status, status_names



def split_rows(second_metric, matrix):
    sub_matrices = []
    start_row = 0
    for rows in second_metric:
        end_row = start_row + rows
        sub_matrix = matrix[start_row:end_row, :]
        sub_matrices.append(sub_matrix)
        start_row = end_row
    return sub_matrices


def getLevel(func_dict, U, test_Data, num_second_level, num_status):
    eval_matrix = np.zeros((num_second_level, num_status))
    for j in range(num_status):
        for k in range(num_second_level):
            eval_matrix[k, j] = fuzz.interp_membership(U, func_dict['func_{}_{}'.format(j, k)], test_Data[k])
    return normalize(eval_matrix, axis=1, norm='l1')


# 单传感器的权重柱状图
def weights_Barplot(data, save_path, time_key_list, fre_key_list, primary_key_list):
    namelist = [time_key_list, fre_key_list]
    num_groups = len(data)  # 组的数量（横坐标的数量）
    max_num_bars = max(len(group) for group in data)  # 每组中的最大柱子数量
    indices = np.arange(num_groups)  # 横坐标的范围
    bar_width = 0.8 / max_num_bars  # 每个柱子的宽度
    fig, ax = plt.subplots(figsize=(20, 10))

    for i in range(max_num_bars):
        # 提取每组中第 i 个柱子的值，如果该组中没有第 i 个值，则设置为0
        values = [U[i] if i < len(U) else 0 for U in data]
        # 计算每个柱子的位置
        positions = indices + i * bar_width
        # 绘制柱子
        bars = ax.bar(positions, values, bar_width, label=f'二级指标 {i + 1}')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height/2, f'{value:.2f}', ha='center', va='center', fontsize=20)

        for i, (group, names) in enumerate(zip(data, namelist)):
            for j, (value, name) in enumerate(zip(group, names)):
                x_pos = i + j * bar_width
                ax.text(x_pos, value, name, ha='center', va='bottom', fontsize=20, color='blue')
    # 添加标签和图例
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 20})
    ax.set_xlabel('指标', fontsize=20)
    ax.set_ylabel('权重', fontsize=20)
    ax.set_title('权重柱状图', fontsize=20)
    ax.set_xticks(indices + bar_width * (max_num_bars - 1) / 2)
    ax.set_xticklabels(primary_key_list, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(save_path + '/Weights_barPlot.png')
    return save_path + '/Weights_barPlot.png'


# 多传感器的权重柱状图
def weights_Barplot_multiple_sensor(data, save_path, sensor1_key_list, sensor2_key_list, sensor3_key_list, primary_key_list):
    namelist = [sensor1_key_list, sensor2_key_list, sensor3_key_list]
    num_groups = len(data)  # 组的数量（横坐标的数量）
    max_num_bars = max(len(group) for group in data)  # 每组中的最大柱子数量
    indices = np.arange(num_groups)  # 横坐标的范围
    bar_width = 0.8 / max_num_bars  # 每个柱子的宽度
    fig, ax = plt.subplots(figsize=(20, 10))

    for i in range(max_num_bars):
        # 提取每组中第 i 个柱子的值，如果该组中没有第 i 个值，则设置为0
        values = [U[i] if i < len(U) else 0 for U in data]
        # 计算每个柱子的位置
        positions = indices + i * bar_width
        # 绘制柱子
        bars = ax.bar(positions, values, bar_width, label=f'二级指标 {i + 1}')

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height/2, f'{value:.2f}', ha='center', va='center', fontsize=20)

        for i, (group, names) in enumerate(zip(data, namelist)):
            for j, (value, name) in enumerate(zip(group, names)):
                x_pos = i + j * bar_width
                ax.text(x_pos, value, name, ha='center', va='bottom', fontsize=20, color='blue')
    # 添加标签和图例
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 20})
    ax.set_xlabel('指标', fontsize=20)
    ax.set_ylabel('权重', fontsize=20)
    ax.set_title('权重柱状图', fontsize=20)
    ax.set_xticks(indices + bar_width * (max_num_bars - 1) / 2)
    ax.set_xticklabels(primary_key_list, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.savefig(save_path + '/Weights_barPlot.png')

    return save_path + '/Weights_barPlot.png'


def getW(criteria, b, fw, max_second_metric):
    W, eigen_list = AHP(criteria, b, max_second_metric).run(fw)
    return W, eigen_list


class AHP:
    def __init__(self, criteria, b, max_second_metric):
        self.RI = (0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49)
        self.criteria = criteria
        self.b = b
        self.num_criteria = criteria.shape[0]
        self.num_project = max_second_metric

    def cal_weights(self, input_matrix):
        criteria = np.array(input_matrix)
        n, n1 = criteria.shape
        assert n == n1, '"准则重要性矩阵"不是一个方阵'
        for i in range(n):
            for j in range(n):
                if np.abs(criteria[i, j] * criteria[j, i] - 1) > 1e-7:
                    raise ValueError('"准则重要性矩阵"不是反互对称矩阵，请检查位置 ({},{})'.format(i, j))

        eigenvalues, eigenvectors = np.linalg.eig(criteria)

        max_idx = np.argmax(eigenvalues)
        max_eigen = eigenvalues[max_idx].real
        eigen = eigenvectors[:, max_idx].real
        eigen = eigen / eigen.sum()

        if n > 9:
            CR = None
            warnings.warn('无法准确判断一致性')
        else:
            CI = (max_eigen - n) / (n - 1)
            CR = CI / self.RI[n - 1] if self.RI[n - 1] != 0 else 0
        return max_eigen, CR, eigen

    def run(self, fw):
        max_eigen, CR, criteria_eigen = self.cal_weights(self.criteria)
        fw.write('=' * 25 + '一级指标' + '=' * 25 + os.linesep)
        fw.write('最大特征值{:<5f},CR={:<5f},检验{}通过'.format(max_eigen, CR, '' if CR < 0.1 else '不') + os.linesep)
        fw.write('一级指标权重 = {}\n'.format(criteria_eigen) + os.linesep)

        max_eigen_list, CR_list, eigen_list = [], [], []
        for i in self.b:
            max_eigen, CR, eigen = self.cal_weights(i)
            max_eigen_list.append(max_eigen)
            CR_list.append(CR)
            eigen_list.append(eigen)

        pd_print = pd.DataFrame(eigen_list,
                                index=['一级指标U' + str(i + 1) for i in range(self.num_criteria)],
                                columns=['二级指标' + str(i + 1) for i in range(self.num_project)],
                                )
        pd_print.loc[:, '最大特征值'] = max_eigen_list
        pd_print.loc[:, 'CR'] = CR_list
        pd_print.loc[:, '一致性检验'] = pd_print.loc[:, 'CR'] < 0.1
        fw.write('=' * 25 + '二级指标' + '=' * 25 + os.linesep)
        table_str = pd_print.to_string()
        fw.write(table_str + os.linesep + os.linesep)

        return criteria_eigen, eigen_list


if __name__ == '__main__':
    second_metric = []
    for i in range(len(b)):
        second_metric.append(b[i].shape[0])  # 每个一级指标下的二级指标个数
    func_dict, status_nums, status_names = create_functions(U, sum(second_metric), data_path, time_key_list,
                                                            fre_key_list, status_names)
    test_data = feature_extraction.GetTest(test_path, time_key_list, fre_key_list)
    level_matrix = getLevel(func_dict, U, test_data, sum(second_metric), status_nums)
    sub_matrices = split_rows(second_metric, level_matrix)
    fw = open(save_path + "/Process_weights.txt", 'w', encoding='gbk')
    W, W_array = getW(criteria, b, fw, max(second_metric))
    # weights_Barplot(W_array, save_path, time_key_list, fre_key_list, primary_key_list)
    B = []
    for i in range(len(sub_matrices)):
        B.append(np.dot(W_array[i], sub_matrices[i]))
    result = np.dot(W, np.array(B))

    with open('models/model_1.pkl', 'wb') as file:
        pickle.dump({'function_dict': func_dict, 'Weight_1': W, 'Weight_2': W_array, 'U': U,
                     'num_second_level': sum(second_metric), 'status_nums': status_nums,
                     'time_key_list': time_key_list, 'fre_key_list': fre_key_list,
                     'status_names': status_names, 'suggestion_dict': suggestion_dict,
                     'primary_key_list': primary_key_list}, file)
