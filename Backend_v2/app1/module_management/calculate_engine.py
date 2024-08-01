import base64
import os.path

import gradio as gr
import pandas as pd

from app1.module_management.algorithms.functions.fault_diagnosis import diagnose_with_svc_model, \
    diagnose_with_random_forest_model, time_regression, \
    diagnose_with_gru_model, diagnose_with_lstm_model
from app1.module_management.algorithms.functions.feature_selection import feature_imp, mutual_information_importance, \
    correlation_coefficient_importance
from app1.module_management.algorithms.functions.health_evaluation import model_eval, model_eval_multiple_sensor
from app1.module_management.algorithms.functions.load_data import load_data
# from app1.module_management.algorithms.functions.speech_processing import mel_spectrogram, audio_sep, add_noise, \
#     verify_speaker
# from app1.module_management.algorithms.functions.load_model import load_model_with_pytorch_lightning
from app1.module_management.algorithms.functions.preprocessing import bicubic_interpolation, polynomial_interpolation, \
    newton_interpolation, linear_interpolation, lagrange_interpolation, extract_signal_features, \
    bicubic_interpolation_for_signal, polynomial_interpolation_for_signal, newton_interpolation_for_signal, \
    linear_interpolation_for_signal, lagrange_interpolation_for_signal, neighboring_values_interpolation_for_signal, \
    extract_features_with_multiple_sensors, wavelet_denoise_four_stages, scaler

# 算法介绍文本
algorithm_introduction = {
    'polynomial_interpolation': {
        'text': '多项式补插法是一种分段三次Hermite插值方法，它在每个数据段上使用三次多项式来逼近数据点，并且在连接处保持一阶导数的连续性。与双三次插值不同，PCHIP'
                '插值在每个段上使用不同的三次多项式，并且尝试保持二阶导数的变号，从而生成一个形状类似于原始数据的曲线。',
        'image': 'D:/研究生工作/项目相关工作/interpolation/算法介绍公式.png'},
    'bicubic_interpolation': {'text': '双三次插值是一种平滑插值方法，它通过三次多项式段来逼近数据点，并且在每个数据段的连接处保持一阶导数和二阶导数的连续性。'
                                      '这种方法可以生成一个平滑的曲线，通过数据点，并且在数据点处具有连续的一阶和二阶导数。'},
    'newton_interpolation': {'text': '在牛顿插值法中，首先利用一组已知的数据点计算差商，再将差商带入插值公式f(x)。将所提供数据中的数据各属性值作为y，而将索引号定义为x，'
                                     '对于所给数据中每一行每一列空白的位置，取空白位置上下4个相邻的值作为输入依据，'
                                     '并计算差商再反向带入包含差商的插值公式，替换原来的空白值。'},
    'linear_interpolation': {'text': '在线性插值算法中，首先遍历数据中的每一行每一列，找到空值位置并获取相邻点的值，'
                                     '然后去除相邻的空值，并处理边界情况，计算插值结果，替换原来的空白值。'},
    'lagrange_interpolation': {'text': '对于所给数据中每一行每一列空白的位置，取空白位置上下3个相邻的值作为输入依据，'
                                       '根据拉格朗日算法构建一个多项式函数，使得该多项式函数在取得的这些点上的值都为零，'
                                       '将空白位置的行值作为输入，计算出y值，替换原来的空白值，从而达到插值的效果。'},

}


def add_user_dir(example_with_selected_features: dict):
    if not example_with_selected_features.get('user_dir') and example_with_selected_features.get('filepath'):
        split_path = example_with_selected_features['filepath'].split('/')
        example_with_selected_features['user_dir'] = os.path.join(split_path[-2], split_path[-1])


def encode_image_to_base64(image_path):
    """
    将PNG文件编码为Base64字符串。

    参数:
    image_path (str): PNG文件的路径。

    返回:
    str: PNG文件的Base64编码字符串。
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        # Base64编码返回的是bytes，所以需要将其解码为str
        return encoded_string.decode('utf-8')


class Reactor:
    def __init__(self):

        self.module_configuration = {n: {'usage': False, 'algorithm': '', 'params': {}, 'result': {},
                                         'introduction': ''} for n in
                                     ['音频分离', '声纹识别', '说话人注册', '添加噪声', '插值处理', '特征提取',
                                      '层次分析模糊综合评估', '小波变换', '特征选择', '故障诊断', '趋势预测',
                                      '无量纲化']}
        self.results_to_response = {n: {} for n in
                                    ['音频分离', '声纹识别', '说话人注册', '添加噪声', '插值处理', '特征提取',
                                     '层次分析模糊综合评估', '小波变换', '特征选择', '故障诊断', '趋势预测',
                                     '无量纲化']}
        self.data_stream = {'filepath': None, 'raw_data': None, 'extracted_features': None,
                            'filename': None, 'user_dir': None, 'features_name': None, 'diagnosis_result': None}
        self.gradio_app = None
        self.schedule = None
        self.lightning_model = None

    # 构造数据流
    def construct_data_stream(self, filepath=None, raw_data=None, extracted_features=None,
                              filename=None, user_dir=None, features_name=None, diagnosis_result=None):
        lst = [filepath, raw_data, extracted_features, filename, user_dir, features_name, diagnosis_result]
        index = 0
        for k in self.data_stream.keys():
            if lst[index] is not None:
                self.data_stream[k] = lst[index]
            index = index + 1
            if index > len(lst) - 1:
                break
        add_user_dir(self.data_stream)

    def init(self, schedule, algorithm_dict, params_dict):
        """
        模型的初始化操作
        :param schedule: (list)how to organize the modules selected to accomplish the mission
        :param params_dict: parameters of the algorithm of each module
        :param algorithm_dict: (dictionary) the algorithms used by the modules

        :return: No return value
        """
        self.schedule = schedule
        for module in self.schedule:
            self.module_configuration[module]['algorithm'] = algorithm_dict[module]
            self.module_configuration[module]['params'] = params_dict[algorithm_dict[module]]
            params_str = ''
            for param, value in params_dict[algorithm_dict[module]].items():
                params_str += param + '=' + str(value) + ' '
            self.module_configuration[module]['introduction'] = ''.join(
                '使用算法：' + algorithm_dict[module] + '\n' + f'算法参数：{params_str}')
            # self.use_tabs_dict[module]['result'] = results_dict[module]

        # 加载进行声纹识别的模型
        # self.lightning_model = load_model_with_pytorch_lightning()

    # 无量纲化
    def scaler(self, data_with_selected_features, multiple_sensor=False):
        data: pd.DataFrame = data_with_selected_features.get('extracted_features')

        use_algorithm = self.module_configuration['无量纲化']['algorithm']
        scaled_data: pd.DataFrame = scaler(data, option=use_algorithm, multiple_sensor=multiple_sensor)

        self.module_configuration['无量纲化']['result']['raw_data'] = data
        self.module_configuration['无量纲化']['result']['scaled_data'] = scaled_data

        self.results_to_response['无量纲化']['raw_data'] = [list(row) for index, row in data.iterrows()]
        self.results_to_response['无量纲化']['scaled_data'] = [list(row) for index, row in scaled_data.iterrows()]
        self.results_to_response['无量纲化']['features_name'] = data_with_selected_features.get('feature_name')

        return data_with_selected_features

    # 插值处理
    def interpolation_v1(self, datafile):
        self.module_configuration['插值处理']['result']['原始数据'] = pd.read_excel(datafile)
        use_algorithm = self.module_configuration['插值处理']['algorithm']
        if use_algorithm == 'polynomial_interpolation':
            waveform, result_data = polynomial_interpolation(datafile)
        elif use_algorithm == 'bicubic_interpolation':
            waveform, result_data = bicubic_interpolation(datafile)
        elif use_algorithm == 'newton_interpolation':
            waveform, result_data = newton_interpolation(datafile)
        elif use_algorithm == 'linear_interpolation':
            waveform, result_data = linear_interpolation(datafile)
        elif use_algorithm == 'lagrange_interpolation':
            waveform, result_data = lagrange_interpolation(datafile)
        else:
            waveform, result_data = '', ''
        self.module_configuration['插值处理']['result']['原始数据波形图'] = waveform.get('原始数据')
        self.module_configuration['插值处理']['result']['结果数据波形图'] = waveform.get('结果数据')
        self.module_configuration['插值处理']['result']['结果数据'] = pd.read_excel(result_data)

        return result_data

    def wavelet_transform_denoise(self, datafile):
        """
        小波变换去噪
        :param datafile:
        :return: (raw_data, denoised_data, filename)
        """
        global denoise_datas
        data_mat, filename = load_data(datafile, multiple_data=True)
        results = wavelet_denoise_four_stages(data_mat, filename)
        if self.module_configuration['小波变换']['algorithm'] == 'wavelet_trans_denoise':
            all_save_paths, denoise_datas = results.get('all_save_paths'), results.get('denoised_datas')
            self.module_configuration['小波变换']['result']['figure_paths'] = {}
            self.module_configuration['小波变换']['result']['denoised_datas'] = {}
            for key, value in all_save_paths.items():
                self.module_configuration['小波变换']['result']['figure_paths'][key] = value
            for key, value in denoise_datas.items():
                self.module_configuration['小波变换']['result']['denoised_datas'][key] = value

        return data_mat, denoise_datas, filename

    def interpolation_v2(self, datafile):
        """
        对信号的插值处理
        :param datafile: data to be interpolated
        :return: (interpolated_data_mat)  -- file_path
        """
        raw_data, filename = load_data(datafile)
        use_algorithm = self.module_configuration['插值处理']['algorithm']
        if use_algorithm == 'polynomial_interpolation':
            result, result_figure = polynomial_interpolation_for_signal(raw_data, filename)
        elif use_algorithm == 'bicubic_interpolation':
            result, result_figure = bicubic_interpolation_for_signal(raw_data, filename)
        elif use_algorithm == 'newton_interpolation':
            result, result_figure = newton_interpolation_for_signal(raw_data, filename)
        elif use_algorithm == 'linear_interpolation':
            result, result_figure = linear_interpolation_for_signal(raw_data, filename)
        elif use_algorithm == 'lagrange_interpolation':
            result, result_figure = lagrange_interpolation_for_signal(raw_data, filename)
        elif use_algorithm == 'neighboring_values_interpolation':
            result, result_figure = neighboring_values_interpolation_for_signal(raw_data, filename)
        else:
            result, result_figure = '', ''

        self.module_configuration['插值处理']['result']['result_data'] = result
        self.module_configuration['插值处理']['result']['result_figure'] = result_figure
        self.results_to_response['插值处理']['figure_Base64'] = result_figure

        return result

    # 特征提取
    def feature_extraction(self, datafile, multiple_sensor=False):
        all_time_features = ['最大值', '最小值', '中位数', '峰峰值', '均值', '方差', '标准差', '峰度', '偏度',
                             '整流平均值', '均方根', '方根幅值', '波形因子', '峰值因子', '脉冲因子', '裕度因子',
                             '四阶累积量', '六阶累积量']
        all_frequency_features = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值',
                                  '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
        features = {'time_domain': [], 'frequency_domain': []}
        use_algorithm = self.module_configuration['特征提取']['algorithm']
        if '_multiple' in use_algorithm:
            multiple_sensor = True
        for key, value in self.module_configuration['特征提取']['params'].items():
            if value:
                if key in all_time_features:
                    features['time_domain'].append(key)
                elif key in all_frequency_features:
                    features['frequency_domain'].append(key)
        data, filename = load_data(datafile)
        # 单传感器特征提取
        if not multiple_sensor:
            features_save_path, features_with_name = extract_signal_features(data, features, filename, save=True)
            features_extracted, filename = load_data(features_save_path)
            # if self.module_configuration['特征提取']['algorithm'] == 'time_domain_features':
            #     features_save_path = extract_time_domain_for_three_dims(datafile, features)
            #     self.module_configuration['特征提取']['result'][
            #         'time_domain_features'] = pd.read_csv(features_save_path)
            # elif self.module_configuration['特征提取']['algorithm'] == 'time_frequency_domain_features':
            #     features_save_path = extract_features_for_three_dims(datafile)
            #     self.module_configuration['特征提取']['results'][
            #         'time_frequency_domain_features'] = pd.read_csv(features_save_path)[features]
            # else:
            #     features_save_path = extract_frequency_domain_for_three_dims(datafile)
            #     self.module_configuration['特征提取']['result'][
            #         'frequency_domain_features'] = pd.read_csv(features_save_path)[features]
        # 多传感器特征提取
        else:
            features_save_path, features_with_name = extract_features_with_multiple_sensors(data, features, filename)
            features_extracted, filename = load_data(features_save_path)
        self.module_configuration['特征提取']['result'] = features_extracted
        self.results_to_response['特征提取']['features_with_name'] = features_with_name
        # self.data_stream['filename'] = filename
        # self.data_stream['filepath'] = datafile
        # self.data_stream['extracted_features'] = features_extracted
        # self.data_stream['raw_data'] = data
        # self.data_stream['features_name'] = features
        # add_user_dir(self.data_stream)
        self.construct_data_stream(filepath=datafile, raw_data=data, extracted_features=features_extracted,
                                   filename=filename, features_name=features)

        # return {'data': features_extracted, 'features': features, 'filename': filename, 'filepath': datafile}
        return self.data_stream

    # 层次分析模糊综合评估法
    def ahp_f(self, data_with_selected_features, multiple_sensor=False):
        """
        层次分析模糊综合评估法
        :return:
        """
        example = data_with_selected_features.get('extracted_features')
        data_path = data_with_selected_features.get('filepath')

        raw_data = data_with_selected_features.get('raw_data')

        filename = data_with_selected_features.get('filename')
        model_path = 'app1/module_management/algorithms/models/health_evaluation/model_1.pkl'
        model_path_multiple_sensor = 'app1/module_management/algorithms/models/health_evaluation/model_2.pkl'
        save_path = f'app1/module_management/algorithms/functions/health_evaluation_results/{filename}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if '_multiple' in self.module_configuration['层次分析模糊综合评估']['algorithm']:
            multiple_sensor = True
        if not multiple_sensor:
            results_path = model_eval(example, model_path, save_path)
        else:
            results_path = model_eval_multiple_sensor(raw_data, model_path_multiple_sensor, save_path)
        with open(results_path.get('suggestion'), 'r', encoding='gbk') as file:
            suggestion = file.read()
        self.module_configuration['层次分析模糊综合评估']['result']['评估建议'] = suggestion
        self.module_configuration['层次分析模糊综合评估']['result']['二级指标权重柱状图'] = results_path.get(
            'weights_bar')
        self.module_configuration['层次分析模糊综合评估']['result']['评估结果柱状图'] = results_path.get('result_bar')
        self.module_configuration['层次分析模糊综合评估']['result']['评估结果向量'] = results_path.get('result_vector')
        self.module_configuration['层次分析模糊综合评估']['result']['层级有效指标'] = results_path.get('tree')

        self.results_to_response['层次分析模糊综合评估']['评估建议'] = suggestion
        self.results_to_response['层次分析模糊综合评估']['二级指标权重柱状图_Base64'] = results_path.get(
                'weights_bar')
        self.results_to_response['层次分析模糊综合评估']['评估结果柱状图_Base64'] = results_path.get('result_bar')
        self.results_to_response['层次分析模糊综合评估']['层级有效指标_Base64'] = results_path.get('tree')

        return results_path

    # 特征选择
    def feature_selection(self, data_with_selected_features, multiple_sensor=False):

        use_algorithm = self.module_configuration['特征选择']['algorithm']
        num_features = self.module_configuration['特征选择']['params']['num_features']

        if '_multiple' in use_algorithm:
            multiple_sensor = True
        # if self.module_configuration['特征选择']['algorithm'] == 'feature_imp':
        #     figure_path, features = feature_imp(multiple_sensor)
        # elif self.module_configuration['特征选择']['algorithm'] == 'mutual_information_importance':
        #     figure_path, features = mutual_information_importance(multiple_sensor)
        # else:
        #     figure_path, features = correlation_coefficient_importance(multiple_sensor)
        if 'feature_imp' in use_algorithm:
            figure_path, features = feature_imp(multiple_sensor, int(num_features))
        elif 'mutual_information_importance' in use_algorithm:
            figure_path, features = mutual_information_importance(multiple_sensor, int(num_features))
        else:
            figure_path, features = correlation_coefficient_importance(multiple_sensor, int(num_features))
        self.module_configuration['特征选择']['result']['features'] = features
        self.module_configuration['特征选择']['result']['figure'] = figure_path

        self.results_to_response['特征选择']['figure_Base64'] = figure_path
        self.results_to_response['特征选择']['selected_features'] = features
        self.construct_data_stream(features_name=features)

        return self.data_stream

    # 故障诊断
    def fault_diagnosis(self, input_data, multiple_sensor=False):

        use_algorithm = self.module_configuration['故障诊断']['algorithm']
        if '_multiple' in use_algorithm:
            multiple_sensor = True
        if 'random_forest' in use_algorithm:
            diagnosis_result, figure_path = diagnose_with_random_forest_model(input_data,
                                                                              multiple_sensor)
            self.module_configuration['故障诊断']['result']['信号波形图'] = figure_path
        elif 'svc' in use_algorithm:
            diagnosis_result, figure_path = diagnose_with_svc_model(input_data, multiple_sensor)
            self.module_configuration['故障诊断']['result']['信号波形图'] = figure_path
        elif 'gru' in use_algorithm:
            raw_data, _ = load_data(input_data)
            self.construct_data_stream(raw_data=raw_data, filepath=input_data)

            diagnosis_result, figure_path = diagnose_with_gru_model(self.data_stream, multiple_sensor)

        elif 'lstm' in use_algorithm:
            raw_data, _ = load_data(input_data)
            self.construct_data_stream(raw_data=raw_data, filepath=input_data)

            diagnosis_result, figure_path = diagnose_with_lstm_model(self.data_stream, multiple_sensor)

        else:
            diagnosis_result = None
            figure_path = ''
        if figure_path:
            self.results_to_response['故障诊断']['figure_Base64'] = figure_path
        if diagnosis_result is not None:
            self.construct_data_stream(diagnosis_result=diagnosis_result)
            # self.results_to_response['故障诊断']['diagnosis_result'] = diagnosis_result
            if diagnosis_result == 0:
                self.module_configuration['故障诊断']['result'][
                    '诊断结果'] = '### 由输入的振动信号，根据故障诊断算法得知该部件<span style=\"color: red\">无故障</span>'
                # self.results_to_response['故障诊断']['diagnosis_result'] = '### 由输入的振动信号，根据故障诊断算法得知该部件<span style=\"color: red\">无故障</span>'
            else:
                self.module_configuration['故障诊断']['result'][
                    '诊断结果'] = '### 由输入的振动信号，根据故障诊断算法得知该部件<span style=\"color: red\">存在故障</span>'
            self.results_to_response['故障诊断']['diagnosis_result'] = str(diagnosis_result)
        else:
            self.module_configuration['故障诊断']['result']['诊断结果'] = '### 发生错误，建议重新检查模型构建过程'

        return self.data_stream

    # 趋势预测
    def fault_regression(self, example_with_selected_features, multiple_sensor=False):

        # 换算时间函数
        def format_seconds(seconds):
            days = seconds // (24 * 3600)
            seconds %= (24 * 3600)
            hours = seconds // 3600
            seconds %= 3600
            minutes = seconds // 60
            seconds %= 60

            # 使用格式化字符串来组合结果
            return f"{days}天{hours}小时{minutes}分钟{seconds:.1f}秒"

        diagnosis_result = example_with_selected_features.get('diagnosis_result')
        use_algorithm = self.module_configuration['趋势预测']['algorithm']
        if '_multiple' in use_algorithm:
            multiple_sensor = True
        time_to_fault, figure_path = time_regression(example_with_selected_features, diagnosis_result, multiple_sensor)
        if time_to_fault == 0:
            time_to_fault_str = ''
            self.module_configuration['趋势预测']['result'][
                'evaluation'] = '经算法预测，目前该部件<span style=\"color: red\">已经故障</span>'
            self.module_configuration['趋势预测']['result']['figure_path'] = figure_path
        else:
            time_to_fault_str = format_seconds(time_to_fault // 100)
            self.module_configuration['趋势预测']['result'][
                'evaluation'] = f'目前该部件<span style=\"color: red\">还未出现故障</span>，预测<span style=\"color: red\">{time_to_fault_str}</span>后会出现故障'
            self.module_configuration['趋势预测']['result']['figure_path'] = figure_path
        self.results_to_response['趋势预测']['figure_Base64'] = figure_path
        self.results_to_response['趋势预测']['time_to_fault'] = str(time_to_fault)
        self.results_to_response['趋势预测']['time_to_fault_str'] = time_to_fault_str

        return example_with_selected_features

    def start(self, datafile, queue):
        """
        依据建立的模型进行业务处理
        :param datafile: data to be precessed(type:filepath)
        :return: no return value
        """
        input_data = datafile
        file_type = datafile.split('.')[-1]
        outcome = 'xxx'
        for module in self.schedule:
            if outcome == '':
                break
            # if module == '音频分离':
            #     outcome = self.speech_separation(input_data)
            # elif module == '添加噪声':
            #     outcome = self.add_noise(input_data)
            # elif module == '声纹识别':
            #     outcome = self.voiceprint_recognition(input_data)
            if module == '插值处理':
                if file_type == 'csv':
                    outcome = self.interpolation_v1(input_data)
                elif file_type == 'mat' or file_type == 'npy':
                    outcome = self.interpolation_v2(input_data)
            elif module == '特征提取':
                outcome = self.feature_extraction(input_data)
            elif module == '层次分析模糊综合评估':
                outcome = self.ahp_f(input_data)
            elif module == '特征选择':
                outcome = self.feature_selection(input_data)
            elif module == '无量纲化':
                outcome = self.scaler(input_data)
            elif module == '故障诊断':
                outcome = self.fault_diagnosis(input_data)
            elif module == '小波变换':
                outcome = self.wavelet_transform_denoise(input_data)
            elif module == '趋势预测':
                outcome = self.fault_regression(input_data)
            else:
                outcome = ''
            input_data = outcome
        print(self.results_to_response)
        print(queue)
        try:
            queue.put(self.results_to_response)
        except Exception as e:
            print(str(e))

    def display(self, display_list, no_thread_lock):
        for module in self.module_configuration.keys():
            if module in display_list:
                self.module_configuration[module]['usage'] = True
            else:
                self.module_configuration[module]['usage'] = False
        self.gradio_app = gradio_template(self.module_configuration)
        try:
            self.gradio_app.launch(prevent_thread_lock=no_thread_lock, show_api=False, server_name='127.0.0.1',
                                   )
            return '7860'
        except Exception as e:
            self.gradio_app.launch(prevent_thread_lock=no_thread_lock, show_api=False, server_name='127.0.0.1',
                                   server_port=7861)
            return '7861'

    def shutdown(self):
        if self.gradio_app is not None:
            self.gradio_app.close()


'''
创建gradio App, 对所处理业务的中间过程进行数据可视化
'''


def gradio_template(use_tabs):
    """
    :param use_tabs: 需要进行可视化的模块
    :return: (gradio.Blocks) Gradio Web App
    """
    with gr.Blocks(
            title="demo1",
            theme=gr.themes.Base(
                primary_hue=gr.themes.colors.blue,
                secondary_hue=gr.themes.colors.amber,
                font=gr.themes.GoogleFont(name='Noto Serif Simplified Chinese', weights=(400, 600)),
                font_mono=['JetBrains mono', "Consolas", 'Courier New'],
                radius_size='sm',
                spacing_size=gr.themes.sizes.spacing_sm
            ),
            # theme=gr.themes.Soft,
            css=" footer {visibility: hidden}",
    ) as gradio_app:
        with gr.Tabs():
            if use_tabs['添加噪声']['usage']:
                with gr.TabItem(label='添加噪声'):
                    gr.Markdown(value='**<center><font size=4>添加噪声</font></center>**')
                    with gr.Row():
                        with gr.Column():
                            gr.Audio(label='原始音频', interactive=False,
                                     value=use_tabs['添加噪声']['result']['原始音频'])
                        with gr.Column():
                            gr.Audio(label='加入噪声后的音频', interactive=False,
                                     value=use_tabs['添加噪声']['result']['加入噪声后的音频'])
            if use_tabs['音频分离']['usage']:
                tabs = ['原始音频', '纯净音频', '噪声音频']
                for tab in tabs:
                    with gr.TabItem(tab):
                        with gr.Row():
                            gr.Audio(label=tab, interactive=False,
                                     value=use_tabs['音频分离']['result'][tab])
                            gr.Image(label=tab + "频谱图", interactive=False, type="filepath",
                                     value=use_tabs['音频分离']['result'][
                                         tab + '频谱图'], show_download_button=False)

            if use_tabs['声纹识别']['usage']:
                with gr.TabItem("声纹识别"):
                    gr.Markdown(value='**<center><font size=4>声纹识别</font></center>**')
                    # 说话人识别
                    with gr.Row(variant="panel"):
                        # with gr.Column(1):
                        #     gr.Markdown(value='**<center><font size=4>算法简介</font></center>**')
                        #     gr.Textbox(label='算法介绍', interactive=False)
                        with gr.Column(scale=2):
                            gr.Markdown(value="""**<center><font size=4>说话人人声频谱分析</font></center>**""")
                            with gr.Row():
                                # audio_input1 = gr.Audio(label="选择音频1", type="filepath")
                                # audio_wave1 = gr.Video(value=gr.make_waveform(audio=audio_input1))

                                gr.Audio(label="说话人音频", interactive=False,
                                         value=use_tabs['声纹识别']['result']['说话人音频'])

                                gr.Image(label="说话人音频频谱图", interactive=False, type="filepath",
                                         value=use_tabs['声纹识别']['result']['说话人音频频谱图'])
                            with gr.Row():
                                gr.Markdown(value='**<center><font size=4></font>识别说话人</center>**')
                                gr.Textbox(label="说话人信息", interactive=False,
                                           value=use_tabs['声纹识别']['result']['识别信息'])
            if use_tabs['说话人注册']['usage']:
                with gr.TabItem("注册"):
                    gr.Markdown(value="""
                                    说话人注册系统
                                    """)
                    # 说话人注册
                    with gr.Row(variant="panel"):
                        with gr.Column():
                            gr.Markdown(value="""<font size=2> 说话人注册</font>""")
                            with gr.TabItem("注册") as speaker_reg_upload:
                                with gr.Row():
                                    audio_regist = gr.Audio(label="选择音频", type="filepath")
                                    sp_name = gr.Textbox(label="说话人姓名")

                        with gr.Column():
                            gr.Markdown(value="""
                                            <font size=3>左侧信息输入完毕后，点击“注册说话人”进行注册：</font>
                                            """)
                            gr.Button(value="注册说话人", variant="primary")
                            gr.Textbox(label="注册信息反馈")

                # sp_submit.click(mel_show, [audio_input1, audio_input2],
                #                 [audio_mel1, audio_mel2, audio_wave1, audio_wave2, sid_mel])
                # sp_submit2.click(verify_speaker, [audio_input1, audio_input2, sid_mel], [sp_output])
                # sp_regist.click(regist_speaker, [audio_regist, sp_name], [output_regist])
            if use_tabs['插值处理']['usage']:
                with gr.Column():
                    gr.Markdown(value="**<center><font size=4>插值处理</font></center>**")
                    image_path = use_tabs['插值处理']['result']['result_figure']
                    gr.Image(value=image_path, interactive=False, show_download_button=False)
                # gr.Markdown(value="""**<center><font size=4>插值处理</font></center>**""")
                # feature_name = use_tabs['插值处理']['result']['原始数据波形图'].keys()
                # raw_features = use_tabs['插值处理']['result']['原始数据波形图']
                # result_features = use_tabs['插值处理']['result']['结果数据波形图']
                # for feature in feature_name:
                #     with gr.TabItem(f"{feature}"):
                #         with gr.Row(variant="panel"):
                #             with gr.Column():
                #                 # gr.Markdown(value='**<center><font size=1>原始数据波形图</font></center>**')
                #                 gr.Image(value=raw_features[feature], show_download_button=False,
                #                          label='原始数据波形图')
                #
                #                 # gr.Markdown(value='**<center><font size=1>结果数据波形图</font></center>**')
                #                 gr.Image(value=result_features[feature], show_download_button=False,
                #                          label='结果数据波形图')
                # with gr.TabItem("插值处理"):
                #     gr.Markdown(value="""**<center><font size=4>插值处理</font></center>**""")
                #         raw_data = use_tabs['插值处理']['result']['原始数据']
                #         gr.Markdown(value='**<center><font size=4>原始数据波形图</font></center>**')
                #         for waveform in use_tabs['插值处理']['result']['原始数据波形图']:
                #             gr.Image(waveform, show_download_button=False, show_label=False)
                #     with gr.Column(variant="panel"):
                #         algorithm = use_tabs['插值处理']['algorithm']
                #         gr.Markdown(value=f"""**<center><font size=4>{algorithm}</font></center>**""")
                #         # gr.Textbox(interactive=False, label='算法介绍',
                #         #            value=algorithm_introduction.get(use_tabs['插值处理']['algorithm']).get('text'))
                #         raw_data = use_tabs['插值处理']['result']['原始数据']
                #         output_data = use_tabs['插值处理']['result']['结果数据']
                #         gr.Markdown(value='**<center><font size=4>原始数据波形图</font></center>**')
                #         for waveform in use_tabs['插值处理']['result']['原始数据波形图']:
                #             gr.Image(waveform, show_download_button=False, show_label=False)
                #         gr.Markdown(value='**<center><font size=4>结果数据波形图</font></center>**')
                #         for waveform in use_tabs['插值处理']['result']['结果数据波形图']:
                #             gr.Image(waveform, show_download_button=False, show_label=False)
                #         gr.Markdown(value='**<center><font size=4>附录数据表单</font></center>**')
                #         gr.Dataframe(label='原始数据', interactive=False, value=raw_data)
                #         gr.Dataframe(label='处理结果', interactive=False, value=output_data)
            if use_tabs['小波变换']['usage']:
                for stage, figure_path in use_tabs['小波变换']['result']['figure_paths'].items():
                    with gr.TabItem(stage):
                        gr.Markdown(value="**<center><font size=4>小波降噪</font></center>**")
                        gr.Image(value=figure_path, show_download_button=False, interactive=False)
            if use_tabs['特征提取']['usage']:
                with gr.TabItem('特征提取'):
                    gr.Markdown(value="""**<center><font size=4>特征提取</font></center>**""")
                    with gr.Column(variant='panel'):
                        if use_tabs['特征提取']['algorithm'] == 'time_domain_features':
                            gr.Markdown(value="""**<center><font size=4>时域特征</font></center>**""")
                        elif use_tabs['特征提取']['algorithm'] == 'time_frequency_domain_features':
                            gr.Markdown(value="""**<center><font size=4>时频域特征</font></center>**""")
                        else:
                            gr.Markdown(value="""**<center><font size=4>频域特征</font></center>**""")
                        # features_type = use_tabs['特征提取']['algorithm']
                        features = use_tabs['特征提取']['result']
                        gr.Dataframe(interactive=False, value=features)
            if use_tabs['无量纲化']['usage']:
                with gr.TabItem('无量纲化'):
                    with gr.Column(variant='panel'):
                        gr.Markdown(value="**<center><font size=4>原始特征数据</font></center>**")
                        gr.Dataframe(value=use_tabs['无量纲化']['result']['raw_data'])
                        gr.Markdown(value="**<center><font size=4>标准化特征数据</font></center>**")
                        gr.Dataframe(value=use_tabs['无量纲化']['result']['scaled_data'])
            if use_tabs['特征选择']['usage']:
                with gr.TabItem('特征选择'):
                    with gr.Column():
                        gr.Image(value=use_tabs['特征选择']['result']['figure'], interactive=False,
                                 show_download_button=False)
                        gr.Markdown(
                            value=f"#### 根据所选特征选择算法，最终选择特征：{use_tabs['特征选择']['result']['features']}")
            if use_tabs['故障诊断']['usage']:
                with gr.TabItem('故障诊断'):
                    gr.Markdown(value=use_tabs['故障诊断']['result']['诊断结果'])
                    gr.Image(value=use_tabs['故障诊断']['result']['信号波形图'], interactive=False,
                             show_download_button=False, label='信号波形图')
            if use_tabs['趋势预测']['usage']:
                with gr.TabItem('趋势预测'):
                    gr.Markdown(value=f"#### 预测结果{use_tabs['趋势预测']['result']['evaluation']}")
                    gr.Image(value=use_tabs['趋势预测']['result']['figure_path'], interactive=False,
                             show_download_button=False, label='信号波形图')
            if use_tabs['层次分析模糊综合评估']['usage']:
                with gr.TabItem('层级有效指标'):
                    with gr.Column():
                        gr.Image(label='层级有效指标', value=use_tabs['层次分析模糊综合评估']['result']['层级有效指标'],
                                 interactive=False, show_download_button=False)
                with gr.TabItem('指标权重'):
                    gr.Image(label='指标权重柱状图',
                             value=use_tabs['层次分析模糊综合评估']['result']['二级指标权重柱状图'], interactive=False,
                             show_download_button=False)
                with gr.TabItem('评估结果'):
                    with gr.Column():
                        gr.Image(label='评估结果', value=use_tabs['层次分析模糊综合评估']['result']['评估结果柱状图'],
                                 interactive=False, show_download_button=False)
                        gr.Markdown(value=f"### {use_tabs['层次分析模糊综合评估']['result']['评估建议']}")

    return gradio_app


