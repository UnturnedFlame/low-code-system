import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from app1.module_management.algorithms.functions.health_evaluation_train import getLevel, weights_Barplot, split_rows,\
    weights_Barplot_multiple_sensor
from app1.module_management.algorithms.functions.feature_extraction import get_test_multiple_sensor

# test_path = 'datas/test.mat'
# model_path = 'models/model_1.pkl'
# save_path = 'results'


def plot_tree(list1, list2_time, list2_freq, save_path):
    # 创建一个有向图
    G = nx.DiGraph()

    # 添加根节点
    root = '状态评估'
    G.add_node(root, level=0)

    # 添加二级节点
    for node in list1:
        G.add_node(node, level=1)
        G.add_edge(root, node)

    # 添加时域指标下的三级节点
    for subnode in list2_time:
        G.add_node(subnode, level=2)
        G.add_edge(list1[0], subnode)

    # 添加频域指标下的三级节点
    for subnode in list2_freq:
        G.add_node(subnode, level=2)
        G.add_edge(list1[1], subnode)

    # 获取每个节点的层次
    levels = nx.get_node_attributes(G, 'level')
    pos = nx.multipartite_layout(G, subset_key="level")

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    # 画图
    plt.figure(figsize=(20, 10))
    node_sizes = [6000 if G.nodes[node]['level'] == 0 else 4000 if G.nodes[node]['level'] == 1 else 3000 for node in
                  G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", font_size=18, font_color="black",
            font_weight="bold", edge_color="gray")

    # 调整箭头样式
    edge_labels = {edge: '' for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(save_path + '/TreePlot.png')
    plt.close()

    return save_path + '/TreePlot.png'


def plot_tree_multiple_sensor(list1, list2_sensor1, list2_sensor2, list2_sensor3, save_path):
    # 创建一个有向图
    G = nx.DiGraph()

    # 添加根节点
    root = '状态评估'
    G.add_node(root, level=0)

    # 添加二级节点
    for node in list1:
        G.add_node(node, level=1)
        G.add_edge(root, node)

    # 添加时域指标下的三级节点
    for subnode in list2_sensor1:
        G.add_node(subnode, level=2)
        G.add_edge(list1[0], subnode)

    # 添加频域指标下的三级节点
    for subnode in list2_sensor2:
        G.add_node(subnode, level=2)
        G.add_edge(list1[1], subnode)

    for subnode in list2_sensor3:
        G.add_node(subnode, level=2)
        G.add_edge(list1[2], subnode)

    # 获取每个节点的层次
    levels = nx.get_node_attributes(G, 'level')
    pos = nx.multipartite_layout(G, subset_key="level")

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    # 画图
    plt.figure(figsize=(12, 8))
    node_sizes = [6000 if G.nodes[node]['level'] == 0 else 4000 if G.nodes[node]['level'] == 1 else 3000 for node in
                  G.nodes()]
    nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="skyblue", font_size=18, font_color="black",
            font_weight="bold", edge_color="gray")

    # 调整箭头样式
    edge_labels = {edge: '' for edge in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.savefig(save_path + '/TreePlot.png')
    plt.close()

    return save_path + '/TreePlot.png'


def model_eval(test_data: pd.DataFrame, model_path, save_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    func_dict = data['function_dict']
    W = data['Weight_1']
    W_array = data['Weight_2']
    U = data['U']
    num_second_level = data['num_second_level']
    status_nums = data['status_nums']
    time_key_list = data['time_key_list']
    fre_key_list = data['fre_key_list']
    status_names = data['status_names']
    suggestion_dict = data['suggestion_dict']
    primary_key_list = data['primary_key_list']
    # test_data = feature_extraction.GetTest(test_path, time_key_list, fre_key_list)
    features_list = time_key_list + fre_key_list
    test_data_selected = test_data[features_list].iloc[0, :].to_list()
    test_data_selected = np.array(test_data_selected).T.astype(np.float64)

    level_matrix = getLevel(func_dict, U, test_data_selected, num_second_level, status_nums)
    weights_bar = weights_Barplot(W_array, save_path, time_key_list, fre_key_list, primary_key_list)
    second_metric = [len(vector) for vector in W_array]
    sub_matrices = split_rows(second_metric, level_matrix)
    B = []
    for i in range(len(sub_matrices)):
        B.append(np.dot(W_array[i], sub_matrices[i]))
    result = np.dot(W, np.array(B))
    suggestion = save_path + "/suggestion.txt"
    fw = open(suggestion, 'w', encoding='gbk')
    fw.write(suggestion_dict[status_names[np.argmax(result)]])

    plot_y = list(result)
    plot_x = [m for m in status_names]
    plt.figure(figsize=(20, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 20})
    plt.title("评估结果(状态隶属度)")
    plt.grid(ls=" ", alpha=0.5)
    bars = plt.bar(plot_x, plot_y)
    for bar in bars:
        plt.setp(bar, color=plt.get_cmap('cividis')(bar.get_height() / max(plot_y)))
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=20)
    result_vector = save_path + '/result.npy'
    result_bar = save_path + '/barPlot.png'
    np.save(result_vector, result)
    plt.savefig(result_bar)
    plt.close()
    tree = plot_tree(primary_key_list, time_key_list, fre_key_list, save_path)
    return {'weights_bar': weights_bar, 'result_vector': result_vector, 'result_bar': result_bar,
            'suggestion': suggestion, 'tree': tree}


def model_eval_multiple_sensor(data_all, model_path, save_path):
    with open(model_path, 'rb') as file:
        data = pickle.load(file)
    func_dict = data['function_dict']
    W = data['Weight_1']
    W_array = data['Weight_2']
    U = data['U']
    num_second_level = data['num_second_level']
    status_nums = data['status_nums']
    sensor1_key_list = data['sensor1_key_list']
    sensor2_key_list = data['sensor2_key_list']
    sensor3_key_list = data['sensor3_key_list']
    status_names = data['status_names']
    suggestion_dict = data['suggestion_dict']
    primary_key_list = data['primary_key_list']
    test_data = get_test_multiple_sensor(data_all, sensor1_key_list, sensor2_key_list, sensor3_key_list)

    level_matrix = getLevel(func_dict, U, test_data, num_second_level, status_nums)
    weights_bar = weights_Barplot_multiple_sensor(W_array, save_path, sensor1_key_list, sensor2_key_list, sensor3_key_list, primary_key_list)
    second_metric = [len(vector) for vector in W_array]
    sub_matrices = split_rows(second_metric, level_matrix)
    B = []
    for i in range(len(sub_matrices)):
        B.append(np.dot(W_array[i], sub_matrices[i]))
    result = np.dot(W, np.array(B))
    result = normalize(result.reshape(1, -1), norm='l1').squeeze()
    suggestion = save_path + "/suggestion.txt"
    fw = open(suggestion, 'w', encoding='gbk')
    fw.write(suggestion_dict[status_names[np.argmax(result)]])

    plot_y = list(result)
    plot_x = [m for m in status_names]
    plt.figure(figsize=(20, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams.update({'font.size': 20})
    plt.title("评估结果(状态隶属度)")
    plt.grid(ls=" ", alpha=0.5)
    bars = plt.bar(plot_x, plot_y)
    for bar in bars:
        plt.setp(bar, color=plt.get_cmap('cividis')(bar.get_height() / max(plot_y)))
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center', fontsize=20)
    # plt.savefig(save_path + '/barPlot.png')
    tree = plot_tree_multiple_sensor(primary_key_list, sensor1_key_list, sensor2_key_list, sensor3_key_list, save_path)
    result_vector = save_path + '/result.npy'
    result_bar = save_path + '/barPlot.png'
    np.save(result_vector, result)
    plt.savefig(result_bar)
    plt.close()
    return {'weights_bar': weights_bar, 'result_vector': result_vector, 'result_bar': result_bar,
            'suggestion': suggestion, 'tree': tree}
