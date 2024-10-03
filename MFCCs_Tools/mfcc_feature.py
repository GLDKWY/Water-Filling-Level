import numpy
import librosa
import numpy as np
import Onset_Dection
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

audio_folder = os.path.join('', 'test_audio')
audio_feature_path = os.path.join('', 'feature_mfcc_test_0dB')
ndb = 0

cut_nums = []
audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
label_id = 1
audio_feature_id = 1
sr = 44100  # frame rate
pre_emphasis = 0.97
frame_t = 0.025  # frame length 25ms
hop_length_t = 0.010  # step 10ms
win_length = int(frame_t * sr)  # 1102
hop_length = int(hop_length_t * sr)  # 441
n_fft = int(2 ** np.ceil(np.log2(win_length)))


def add_noise(s, d, SNR):
    """
    Adds noise to the clean waveform at a specific SNR value. A random section
    of the noise waveform is used.

    Inputs:
        s - clean waveform.
        d - noise waveform.
        SNR - SNR level.

    Outputs:
        x - noisy speech waveform.
        d - truncated and scaled noise waveform.
    """

    s_len = s.shape[0]  # length, channel
    d_len = d.shape[0]
    i = torch.randint(int(d_len - s_len) + 1, (1,))
    d = d[i:i + s_len]  #

    a = 10 ** (0.05 * SNR)

    d = (d / np.linalg.norm(d, axis=0)) * (np.linalg.norm(s, axis=0) / a)  # multichannel
    x = s + d.reshape(s_len, -1) if s.shape != d.shape else s + d

    return x, d


for i, path in enumerate(audio_paths):
    waveform, _ = librosa.core.load(path, sr=sr, mono=False)

    waveform = waveform.transpose(1, 0)
    waveform = (waveform[:, 0] + waveform[:, 1]) / 2
    wave_file_time, wave_file_data_l, wave_file_data_r, wave_file_framerate = Onset_Dection.get_audio_data(path)
    audio_cutframes = Onset_Dection.cut_signal(wave_file_data_l, 512, 128)
    audio_points = Onset_Dection.point_check(audio_cutframes)
    waveform = waveform[audio_points[0] * 128: audio_points[-1] * 128]

    noise = np.random.randn(len(waveform))
    waveform, d = add_noise(waveform, noise, ndb)
    '''
    MAX = np.max(waveform)
    MIN = np.min(waveform)
    # MEAN = np.mean(waveform)
    # STD = np.std(waveform)
    waveform = (waveform - MIN) / (MAX - MIN)  # normalize to (0,1)
    # waveform = (waveform - 0.5) * 2
    # waveform = (waveform - MEAN) / STD  # standardize
    '''
    step_time = 4.0  # clip length
    small_step_time = 0.5  # sample length
    step_points = int(step_time * sr)  # clip length (sample rate)
    small_step_points = int(small_step_time * sr)  # sample length (sample rate)
    cut_num = int(len(waveform) // (sr * step_time))
    waveform = waveform[:cut_num * step_points]  # padding and discard clips shorter than 4s
    # print(waveform.shape)
    # waveform = move_data_to_device(waveform, device)
    # annotation_pth = os.path.join('Train_Annotations', '{0:03d}.csv'.format(i + 1))
    print(path)

    for j in range(cut_num):
        all_signal = []

        # tmp_waveform = waveform[(j*step_points)+(l*small_step_points) : (j*step_points)+(l+1)*small_step_points]
        tmp_waveform = waveform[j * step_points:(j + 1) * step_points]
        tmp_waveform = tmp_waveform[None, :]  # (1, audio_length)

        emphasized_signal = numpy.append(tmp_waveform[0], tmp_waveform[1:] - pre_emphasis * tmp_waveform[:-1])
        signal = librosa.feature.mfcc(y=emphasized_signal, sr=sr, S=None, n_fft=n_fft, win_length=win_length,
                                      hop_length=hop_length,
                                      dct_type=2, n_mels=128, n_mfcc=128)
        signal_deta = librosa.feature.delta(signal)  # delta
        signal_deta2 = librosa.feature.delta(signal, order=2)  # delta-delta
        signal = np.concatenate([signal, signal_deta, signal_deta2], axis=0)  # mix
        all_signal.append(signal)

        print(np.shape(signal))
        np.save(os.path.join(audio_feature_path, "{0:05d}".format(audio_feature_id)), signal)
        audio_feature_id += 1
