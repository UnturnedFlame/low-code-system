# import math
# import os
# import torch
# from app1.module_management.algorithms.models.sepformer import Sepformer
# from app1.module_management.algorithms.models.conformer import ConformerCSS
# from app1.module_management.algorithms.models.utils import remove_pad
# import soundfile
# import librosa
# from app1.module_management.algorithms.functions.utils import MelSpectrogram
# from app1.module_management.algorithms.functions.dataset import load_audio
# # from elasticsearch import Elasticsearch
# from app1.module_management.algorithms.functions.utils import cosine_similarity
# # from app1.module_management.algorithms.functions.load_model import load_model_with_pytorch_lightning
# import numpy as np
# from app1 import models
#
# """
# 该模块可选择使用不同的模型对语音音频进行分离。
# """
#
#
# def audio_sep(audio_path, model):
#     """
#     :param audio_path: 需要分离的音频文件路径
#     :param model: 使用的分离模型
#     :return: seperated_audios(dict), spk_mel(filepath: Image), noise_mel(filepath: Image)
#     """
#     # from scipy.io import wavfile
#     if model == 'sepformer':
#         model = Sepformer.load_model("app1/module_management/algorithms/models/checkpoints/sepformer.pth.tar")
#     elif model == 'conformer':
#         model = ConformerCSS.load_model("app1/module_management/algorithms/models/checkpoints/conformer.pth.tar")
#
#     model.eval()
#     if torch.cuda.is_available():
#         model.cuda()
#
#     def write_wav(inputs, filename, sr=16000):
#         soundfile.write(filename, inputs, sr)  # norm=True)
#
#     # filename = os.path.join("audio_test/mix", 'mix.wav')
#     # write_wav(audio_path, filename)
#     sig, sr = librosa.load(audio_path, sr=16000)
#
#     with torch.no_grad():
#         mix_lengths = len(sig)
#         mixture = torch.tensor(sig)
#         mix_lengths = torch.tensor(mix_lengths)
#         mixture = mixture.view((1, -1))
#         mix_lengths = mix_lengths.view(-1)
#         if torch.cuda.is_available():
#             mixture = mixture.cuda()
#
#             mix_lengths = mix_lengths.cuda()
#
#         estimate_source = model(mixture)  # 将数据放入模型
#
#         # Remove padding and flat
#         flat_estimate = remove_pad(estimate_source, mix_lengths)
#
#         mixture = remove_pad(mixture, mix_lengths)
#         seperated_audios = {}
#         for c in range(2):
#             if c == 0:
#                 filename = os.path.join(
#                     "app1/module_management/algorithms/functions/speech_processing_results/seperated_audio",
#                     "noise.wav")
#                 write_wav(flat_estimate[0][c], filename, sr)
#                 seperated_audios['noise'] = filename
#                 # filename = os.path.join("static", a1)
#                 # write_wav(flat_estimate[0][c], filename)
#             elif c == 1:
#                 filename = os.path.join(
#                     "app1/module_management/algorithms/functions/speech_processing_results/seperated_audio",
#                     "spk.wav")
#                 write_wav(flat_estimate[0][c], filename, sr)
#                 seperated_audios['speaker'] = filename
#
#         # 频谱图生成
#         from scipy.io import wavfile
#         import matplotlib.pyplot as plt
#         from torchvision import transforms as transforms
#         sample_rate, sig = wavfile.read(seperated_audios['speaker'])
#         sig = torch.FloatTensor(sig.copy())
#         sig = sig.repeat(10, 1)
#         spec = MelSpectrogram(sample_rate=16000)
#         out = spec(sig)
#         # trans = transforms.RandomResizedCrop((80, 300))
#         # out = trans(out)
#         plt.imshow(out[0][0])
#         plt.xticks([])
#         plt.yticks([])
#         plt.axis('off')
#
#         import uuid
#         spk_mel = os.path.join("app1/module_management/algorithms/functions/speech_processing_results/seperated_audio",
#                                "spk" + "_" + str(uuid.uuid4()) + ".png")
#         plt.savefig(spk_mel, bbox_inches='tight', pad_inches=0.0)
#
#         sample_rate, sig = wavfile.read(seperated_audios['noise'])
#         sig = torch.FloatTensor(sig.copy())
#         sig = sig.repeat(10, 1)
#         spec = MelSpectrogram(sample_rate=16000)
#         out = spec(sig)
#         # trans = transforms.RandomResizedCrop((80, 300))
#         # out = trans(out)
#         plt.imshow(out[0][0])
#         plt.xticks([])
#         plt.yticks([])
#         plt.axis('off')
#
#         noise_mel = os.path.join(
#             "app1/module_management/algorithms/functions/speech_processing_results/seperated_audio",
#             "noise" + "_" + str(uuid.uuid4()) + ".png")
#         plt.savefig(noise_mel, bbox_inches='tight', pad_inches=0.0)
#     return seperated_audios, spk_mel, noise_mel
#     # return gr.make_waveform(audio=filenames[1], bars_color=('#1E90FF'), animate=False), spk_mel, gr.make_waveform(
#     #     audio=filenames[0], bars_color=('#1E90FF'), animate=False), noise_mel, "分离噪声成功！"
#
#
# def mel_spectrogram(audio_path):
#     if audio_path is None:
#         return None, None
#     # 频谱图生成
#     from scipy.io import wavfile
#     import matplotlib.pyplot as plt
#     # from torchvision import transforms as transforms
#     sample_rate, sig = wavfile.read(audio_path)
#     sig = torch.FloatTensor(sig.copy())
#     sig = sig.repeat(10, 1)
#     spec = MelSpectrogram(sample_rate=48000)
#     out = spec(sig)
#     # trans = transforms.RandomResizedCrop((80, 300))
#     # out = trans(out)
#     plt.imshow(out[0][0])
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#
#     import uuid
#     # 构建保存文件的路径，并保存到results文件夹内
#     if not os.path.exists("mels_results"):
#         os.makedirs("mels_results")
#     output_mel = os.path.join("mels_results", "raw" + "_" + str(uuid.uuid4()) + ".png")
#     plt.savefig(output_mel, bbox_inches='tight', pad_inches=0.0)
#
#     return output_mel
#     # gr.make_waveform(audio=audio_path, bars_color=('#1E90FF'), animate=False)
#
#
# def verify_speaker(audio_path, lightning_model):
#     # if sid_mel is None:
#     #     return "请选择识别音频！"
#     # elif audio_path1 is None or audio_path2 is None:
#     #     return "请上传识别音频！"
#     # elif sid_mel == "音频1":
#     #     audio_path = audio_path1
#     # elif sid_mel == "音频2":
#     #     audio_path = audio_path2
#     # else:
#     #     return "分析系统异常！"
#     # lightning_model = load_model_with_pytorch_lightning()
#     # 说话人识别
#     score = -1
#     speaker = ""
#     wav = load_audio(audio_path, -1)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     emb_wav = lightning_model(torch.FloatTensor(wav).unsqueeze(0).to(device))
#     np_emb = emb_wav.cpu().detach().numpy()
#     # 启动elasticsearch
#     # es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
#     threshold = 0.75
#     # try:
#     #     response = es.search(index="speakers", doc_type='_doc')
#     # except Exception as e:
#     #     print(f'捕获到异常{e}')
#     #     return None, "请先注册说话人！"
#     # for hit in response['hits']['hits']:
#     #     ts = cosine_similarity(hit['_source']['speaker_embedding'], np_emb)
#     #     if ts > score:
#     #         speaker = hit['_source']['speaker_name']
#     #         score = ts
#     speakers = models.RegisteredSpeakers.objects.all()
#     for speaker in speakers:
#         feature_vector = np.frombuffer(speaker.feature_vector, dtype=np.float32)
#         temp_score = cosine_similarity(feature_vector, np_emb)
#         if temp_score > score:
#             speaker = speaker.name
#             score = temp_score
#     if score > threshold:
#         sp_inform = str("说话人姓名： " + speaker + "  |  匹配度得分： {:.2f}".format(score))
#     else:
#         # sp_inform = str("未知说话人！"+ "     " + "最大匹配度说话人姓名： " + speaker + "|| 匹配度得分： {:.2f}".format(score))
#         sp_inform = str("未知说话人！")
#
#     return sp_inform
#
#
# # 添加高斯噪声
# def add_noise(audio_path, outcome_filename, noise_type=None, SNR=-5, sr=16000):
#     """
#
#     :param noise_type: 需要加入的噪声类型
#     :param audio_path: 要加入噪声的音频文件
#     :param outcome_filename: 输出结果的文件名(.wav)
#     :param SNR: 信噪比, default -5
#     :param sr: 采样率, default 16000
#     :return: 输出结果的文件路径
#     """
#     # 读取语音文件data和fs
#     src, sr = librosa.core.load(audio_path, sr=sr)
#
#     if noise_type == "WhiteGaussianNoise":
#         random_values = np.random.rand(len(src))
#         # 计算语音信号功率Ps和噪声功率Pn1
#         Ps = np.sum(src ** 2) / len(src)
#         Pn1 = np.sum(random_values ** 2) / len(random_values)
#
#         # 计算k值
#         k = math.sqrt(Ps / (10 ** (SNR / 10) * Pn1))
#         # 将噪声数据乘以k,
#         random_values_we_need = random_values * k
#         # 将噪声数据叠加到纯净音频上去
#         outcome_data = src + random_values_we_need
#     elif noise_type is None:
#         outcome_data = src
#     # 计算新的噪声数据的功率
#     # Pn = np.sum(random_values_we_need ** 2) / len(random_values_we_need)
#     # 以下开始计算信噪比
#     # snr = 10 * math.log10(Ps / Pn)
#
#     # 单独将噪音数据写入文件
#     # soundfile.write(noise_path, random_values_we_need, sr)
#
#
#     # 将叠加噪声的数据写入文件
#     out_path = 'app1/module_management/algorithms/functions/speech_processing_results/audios_with_noise/' + outcome_filename
#     soundfile.write(out_path, outcome_data, sr)
#
#     return out_path
#
#
# def extract_speaker_feature(audio_path, lightning_model):
#     # lightning_model = load_model_with_pytorch_lightning()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     wav = load_audio(audio_path, -1)
#     emb_wav = lightning_model(torch.FloatTensor(wav).unsqueeze(0).to(device))
#     np_emb = emb_wav.cpu().detach().numpy()
#     data_in_bytes = np_emb.tobytes()
#
#     # e = {
#     #     "speaker_name": name,
#     #     "speaker_embedding": np_emb
#     # }
#     # res = es.index(index='speakers', doc_type='_doc', body=e)
#     # return str("系统发生异常，说话人注册失败！")
#     return data_in_bytes
#     # return str("恭喜您！ 注册说话人" + name + "成功！")
#
#
# if __name__ == '__main__':
#     model = load_model_with_pytorch_lightning()
#     audio = 'app1/recv_file/examples/spk.wav'
#     # result = verify_speaker(audio_path, model)
#     # print(result)
