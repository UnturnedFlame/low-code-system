import torch
from utils import MelSpectrogram
import os
import gradio as gr


def mel_spectrogram(audio_path):
    if audio_path is None:
        return None, None
    # 频谱图生成
    from scipy.io import wavfile
    import matplotlib.pyplot as plt
    # from torchvision import transforms as transforms
    sample_rate, sig = wavfile.read(audio_path)
    sig = torch.FloatTensor(sig.copy())
    sig = sig.repeat(10, 1)
    spec = MelSpectrogram(sample_rate=48000)
    out = spec(sig)
    # trans = transforms.RandomResizedCrop((80, 300))
    # out = trans(out)
    plt.imshow(out[0][0])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')

    import uuid
    # 构建保存文件的路径，并保存到results文件夹内
    if not os.path.exists("mels_results"):
        os.makedirs("mels_results")
    output_mel = os.path.join("mels_results", "raw" + "_" + str(uuid.uuid4()) + ".png")
    plt.savefig(output_mel, bbox_inches='tight', pad_inches=0.0)

    return output_mel, gr.make_waveform(audio=audio_path, bars_color=('#1E90FF'), animate=False)
