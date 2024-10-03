import os
import time
import seaborn as sns
import cv2
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import soundfile
import pyaudio
import wave

# FILENAME = './train_audio/005.wav'
#
#
# def add_noise(s, d, SNR):
#     """
#     Adds noise to the clean waveform at a specific SNR value. A random section
#     of the noise waveform is used.
#
#     Inputs:
#         s - clean waveform.
#         d - noise waveform.
#         SNR - SNR level.
#
#     Outputs:
#         x - noisy speech waveform.
#         d - truncated and scaled noise waveform.
#     """
#
#     s_len = s.shape[0]  # length, channel
#     d_len = d.shape[0]
#     i = torch.randint(int(d_len - s_len) + 1, (1,))
#     d = d[i:i + s_len]  #
#
#     a = 10 ** (0.05 * SNR)
#
#     d = (d / np.linalg.norm(d, axis=0)) * (np.linalg.norm(s, axis=0) / a)  # multichannel
#     x = s + d.reshape(s_len, -1) if s.shape != d.shape else s + d
#
#     return x, d
#
#
# (waveform, _) = librosa.core.load(FILENAME, sr=44100, mono=True)
# plt.plot(waveform)
# plt.show()
# print(len(waveform))
# noise = np.random.randn(len(waveform))
# noisy_waveform, d = add_noise(waveform, noise, -20)
# print(len(noisy_waveform))
# soundfile.write('tmp.wav', noisy_waveform, 44100)


# loadData = np.load('PANNS_Audio_Feature/test_audio_features/00023.npy', allow_pickle=True)
#
# print("----type----")
# print(type(loadData))
# print("----shape----")
# print(loadData.shape)
# print("----data----")
# print(loadData)
# print(len(loadData))

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# import numpy as np
#
# a = [x for x in range(100)]
# a = np.array(a)
# a = np.expand_dims(a, axis=-1)
# print(a.shape)
# # 构造一个含有噪声的正弦波
# time = np.arange(0, 100, 1)
# sin_waves = np.sin(time)
# sin_waves = np.expand_dims(sin_waves, axis=-1)
# noise = np.concatenate((sin_waves + 1, sin_waves - 1), axis=1)
# print('noise shape: ', noise.shape)  # (63, 10)
# data = sin_waves + noise
# data_mean = np.mean(data, axis=1)
# data_std = np.std(data, axis=1)
# data_var = np.var(data, axis=1)
# data_max = np.max(data, axis=1)
# data_min = np.min(data, axis=1)
#
#
# # 将 time 扩展为 10 列一样的数据
# time_array = time
# for i in range(noise.shape[1] - 1):
#     time_array = np.column_stack((time_array, time))
#
# # 将 time 和 signal 平铺为两列数据，且一一对应
# time_array = time_array.flatten()  # (630,)
# data = data.flatten()  # (630,)
# data = np.column_stack((time_array, data))  # (630,2)
# df = pd.DataFrame(data, columns=['time', 'signal'])
#
# # 绘图
# sns.set(style='darkgrid', font_scale=1.1)
# g = sns.relplot(x='time', y='signal', data=df, kind='line', markers=True, color='black')
# g.set_axis_labels("Filling level (%)", "MAE Error (%)")
# g.set(ylim=(-4, 4))
# plt.legend(labels=["Average", "Boundary"])
# plt.show()

# import os
# from moviepy.editor import AudioFileClip
#
# video_folder = 'test_tactile'
# video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
# for i, path in enumerate(video_paths):
#     # print(path)
#     # print(i + 1)
#     id = i + 1
#     my_audio_clip = AudioFileClip(path)
#     audio_name = os.path.join('test_audio', "{0:03d}.wav".format(id))
#     my_audio_clip.write_audiofile(audio_name)


# import wave
#
# with wave.open('./train_audio/016.wav', "rb") as f:
#     f = wave.open('./train_audio/016.wav')
#     print(f.getparams())
# (waveform, _) = librosa.core.load('./train_audio/011.wav', sr=44100, mono=True)
# step_time = 0.25  # 一小段音频为0.25s
# step_points = int(step_time * 44100)  # 0.25s中包括多少音频幅度值
# cut_nums = int(len(waveform) // step_points)  # 整段音频中能有几个0.25s的小音频
# print(type(waveform))


# import librosa
#
#
# def get_duration_mp3_and_wav(file_path):
#     """
#     获取mp3/wav音频文件时长
#     :param file_path:
#     :return:
#     """
#     duration = librosa.get_duration(filename=file_path)
#     return duration
#
#
# durations = []
# # audio_folder = 'train_audio'
# # audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
# # for i, path in enumerate(audio_paths):
# #     tmp = get_duration_mp3_and_wav(path)
# #     durations.append(tmp)
#
# audio_folder = 'test_audio'
# audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
# for i, path in enumerate(audio_paths):
#     print(path)
#     tmp = get_duration_mp3_and_wav(path)
#     durations.append(tmp)
#
# print(len(durations))
# print('sum', np.sum(durations))
# print('min', np.min(durations))
# print('max', np.max(durations))


# df = pd.read_csv('results.csv')
# sns.set(style='whitegrid', font_scale=1.1)
# # sns.set_style("darkgrid")
# g = sns.catplot(
#     data=df, x="SNR(dB)", hue="Type", y="Accuracy (%)", col="Tolerance (%)",
#     kind="bar", height=4, aspect=.6, palette=sns.color_palette("colorblind"))
# g.set_axis_labels("SNR (dB)", "Accuracy (%)")
# # g.set_xticklabels(["MFCCs", "PANNs", "VGGish", "MobileNetv2"])
# g.set_xticklabels(["-20", "-10", "0"])
# g.set_titles("{col_name} {col_var}")
# g.set(ylim=(0, 100))
# # g.despine(left=True)
# plt.show()

'''批量提取音频'''
# import os
# from moviepy.editor import AudioFileClip
#
# video_folder = 'train_tactile'
# video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
# for i, path in enumerate(video_paths):
#     id = i
#     my_audio_clip = AudioFileClip(path)
#     audio_name = os.path.join('train_audio', "{0:03d}.wav".format(id + 1))
#     my_audio_clip.write_audiofile(audio_name)


# annotation_paths = [os.path.join('labels/train_labels_4s_8_32000', path) for path in
#                                 sorted(os.listdir('labels/train_labels_4s_8_32000'))]
# for i, path in enumerate(annotation_paths):
#     tmp_audio_annotation = np.load(path)
#     print(tmp_audio_annotation.shape)
#
# min_percentage = 101
# annotations_folder = 'Train_Annotations'
# audio_paths = [os.path.join(annotations_folder, path) for path in sorted(os.listdir(annotations_folder))]
# for i, path in enumerate(audio_paths):
#     print(path)
#     df = pd.read_csv(path)
#     col_array = df['percentage'].to_numpy()
#     if col_array[-1] < min_percentage:
#         min_percentage = col_array[-1]
#
# print(min_percentage)
index = 0
target_paths = [os.path.join("labels/test_labels_4s_8_32000", path) for path
                in sorted(os.listdir("labels/test_labels_4s_8_32000"))]
for i, path in enumerate(target_paths):
    data = np.load(path)
    print(data)
    if data[0] < 10:
        index += 1
    if index == 8:
        print("desired", data)
        print("desired: ", path)
