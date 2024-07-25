import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

# 加载数据
file_path = 'app1/module_management/algorithms/functions/datas/vibration_features_with_labels.csv'  # 请确保文件路径正确
file_path_2 = 'app1/module_management/algorithms/functions/datas/multi_sensor_features.csv'
save_path = 'app1/module_management/algorithms/functions/feature_selection_results'


def feature_imp(multiple_sensor=False):

    if not multiple_sensor:
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path_2)
    all_columns = data.columns
    empty_columns = [col for col in all_columns if ('谱峭度的偏度' in col or '谱峭度的标准差' in col)]
    # 删除全空的列
    # data_cleaned = data.drop(columns=['谱峭度的偏度', '谱峭度的标准差'])
    data_cleaned = data.drop(columns=empty_columns)
    # 分离特征和标签
    X_cleaned = data_cleaned.drop(columns=['label'])
    y_cleaned = data_cleaned['label']

    # 用列的均值填充缺失值
    imputer = SimpleImputer(strategy='mean')
    X_imputed_cleaned = imputer.fit_transform(X_cleaned)

    # 训练随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_imputed_cleaned, y_cleaned)

    # 获取特征重要性
    importances_cleaned = clf.feature_importances_
    indices_cleaned = np.argsort(importances_cleaned)[::-1]

    # 打印每个特征的重要性
    # feature_importances_cleaned = {X_cleaned.columns[indices_cleaned[i]]: importances_cleaned[indices_cleaned[i]] for i
    #                                in range(X_cleaned.shape[1])}
    # print("Feature Importances:")
    # for feature, importance in feature_importances_cleaned.items():
    #     print(f"{feature}: {importance}")
    # top10的名字和值
    top10_features = {X_cleaned.columns[indices_cleaned[i]]: importances_cleaned[indices_cleaned[i]] for i in range(10)}

    # 可视化特征重要性
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    plt.figure(figsize=(20, 10))

    plt.title("特征重要性", fontsize=20)
    if not multiple_sensor:
        bars = plt.bar(range(len(importances_cleaned)), importances_cleaned[indices_cleaned], align="center")
        plt.xticks(range(len(importances_cleaned)), X_cleaned.columns[indices_cleaned], rotation=90, fontsize=20)
    else:
        if len(X_cleaned.columns) > 20:
            end_of_view = 20
        else:
            end_of_view = len(X_cleaned.columns)
        bars = plt.bar(range(end_of_view), importances_cleaned[indices_cleaned][0:end_of_view], align="center")
        plt.xticks(range(end_of_view), X_cleaned.columns[indices_cleaned][0:end_of_view], rotation=90, fontsize=20)

    # 使用不同颜色区分前五个特征
    for i in range(10):
        bars[i].set_color('r')

    # plt.xticks(range(len(importances_cleaned)), X_cleaned.columns[indices_cleaned], rotation=90, fontsize=20)
    # plt.xlim([-1, len(importances_cleaned)])
    plt.tight_layout()
    plt.savefig(save_path + "/feature_imp" + "/feature_imp.png")
    plt.close()

    return save_path + "/feature_imp" + "/feature_imp.png", list(top10_features.keys())


def plot_importance(importance_dict, title, multiple_sensor=False):
    sorted_importance = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
    features = [item[0] for item in sorted_importance]
    scores = [item[1] for item in sorted_importance]

    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题

    plt.figure(figsize=(20, 10))
    plt.title(title, fontsize=20)
    if not multiple_sensor:
        bars = plt.bar(range(len(scores)), scores, align="center")
        plt.xticks(range(len(scores)), features, rotation=90, fontsize=20)
    else:
        bars = plt.bar(range(20), scores[0:20], align="center")
        plt.xticks(range(20), features[0:20], rotation=90, fontsize=20)

    # 使用不同颜色区分前10个特征
    for i in range(10):
        bars[i].set_color('r')

    plt.tight_layout()
    if title == "互信息重要性":
        filename = "mutual_information_importance.png"
    else:
        filename = "相关系数重要性.png"
    plt.savefig(save_path + "/feature_self/" + filename)
    plt.close()

    return save_path + "/feature_self/" + filename, features[0:10]


def mutual_information_importance(multiple_sensor=False):
    """
    互信息重要性
    :return:
    """
    if not multiple_sensor:
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path_2)
    all_columns = data.columns
    empty_columns = [col for col in all_columns if ('谱峭度的偏度' in col or '谱峭度的标准差' in col)]
    # 删除全空的列
    # data_cleaned = data.drop(columns=['谱峭度的偏度', '谱峭度的标准差'])
    data_cleaned = data.drop(columns=empty_columns)
    # 分离特征和标签
    X_cleaned = data_cleaned.drop(columns=['label'])
    y_cleaned = data_cleaned['label']

    # 互信息法
    mi = mutual_info_classif(X_cleaned, y_cleaned, discrete_features='auto')
    mi_importance = {X_cleaned.columns[i]: mi[i] for i in range(len(mi))}

    # 可视化互信息重要性
    figure_path, features = plot_importance(mi_importance, "互信息重要性", multiple_sensor=multiple_sensor)

    return figure_path, features


def correlation_coefficient_importance(multiple_sensor=False):
    """
    :return:
    """
    if not multiple_sensor:
        data = pd.read_csv(file_path)
    else:
        data = pd.read_csv(file_path_2)

    all_columns = data.columns
    empty_columns = [col for col in all_columns if ('谱峭度的偏度' in col or '谱峭度的标准差' in col)]
    # 删除全空的列
    # data_cleaned = data.drop(columns=['谱峭度的偏度', '谱峭度的标准差'])
    data_cleaned = data.drop(columns=empty_columns)
    # 分离特征和标签
    X_cleaned = data_cleaned.drop(columns=['label'])
    y_cleaned = data_cleaned['label']

    # 相关系数法
    correlations = [pearsonr(X_cleaned[col], y_cleaned)[0] for col in X_cleaned.columns]
    cor_importance = {X_cleaned.columns[i]: abs(correlations[i]) for i in range(len(correlations))}

    # 可视化相关系数重要性
    figure_path, features = plot_importance(cor_importance, "相关系数重要性", multiple_sensor=multiple_sensor)

    return figure_path, features
