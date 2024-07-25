import numpy as np
import pandas as pd

from scipy.io import wavfile
from scipy import signal
import soundfile


def compute_dB(waveform):
    """
    Args:
        x (numpy.array): Input waveform (#length).
    Returns:
        numpy.array: Output array (#length).
    """
    val = max(0.0, np.mean(np.power(waveform, 2)))
    dB = 10 * np.log10(val + 1e-4)
    return dB


class WavAugment(object):
    def __init__(self, noise_csv_path="data/noise.csv", rir_csv_path="data/rir.csv"):
        self.noise_paths = pd.read_csv(noise_csv_path)["utt_paths"].values
        self.noise_names = pd.read_csv(noise_csv_path)["speaker_name"].values
        self.rir_paths = pd.read_csv(rir_csv_path)["utt_paths"].values

    def __call__(self, waveform):
        idx = np.random.randint(0, 5)
        if idx == 0:
            waveform = waveform

        if idx == 1:
            waveform = self.reverberate(waveform)

        if idx == 2:
            waveform = self.add_real_noise(waveform, 'speech')

        if idx == 3:
            waveform = self.add_real_noise(waveform, 'music')

        if idx == 4:
            waveform = self.add_real_noise(waveform, 'noise')

        if idx == 5:
            waveform = self.add_real_noise(waveform, 'speech')
            waveform = self.add_real_noise(waveform, 'music')

        return waveform

    def add_real_noise(self, waveform, mtype):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        clean_dB = compute_dB(waveform)

        stat = np.where(self.noise_names == mtype)[0][0]
        end = np.where(self.noise_names == mtype)[-1][-1]

        idx = np.random.randint(stat, end)
        sample_rate, noise = wavfile.read(self.noise_paths[idx])
        noise = noise.astype(np.float64)

        snr = np.random.uniform(15, 25)

        noise_length = len(noise)
        audio_length = len(waveform)

        if audio_length >= noise_length:
            shortage = audio_length - noise_length
            noise = np.pad(noise, (0, shortage), 'wrap')
        else:
            start = np.random.randint(0, (noise_length - audio_length))
            noise = noise[start:start + audio_length]

        noise_dB = compute_dB(noise)
        noise = np.sqrt(10 ** ((clean_dB - noise_dB - snr) / 10)) * noise
        waveform = (waveform + noise)
        return waveform

    def reverberate(self, waveform):
        """
        Args:
            x (numpy.array): Input length (#length).
        Returns:
            numpy.array: Output waveform array (#length).
        """
        audio_length = len(waveform)
        idx = np.random.randint(0, len(self.rir_paths))

        path = self.rir_paths[idx]
        rir, sample_rate = soundfile.read(path)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        # 没有采用real_rirs_isotropic_noises
        waveform = signal.convolve(waveform, rir, mode='full')

        return waveform[:audio_length]
