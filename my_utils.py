import numpy as np
import os

import pandas as pd
import scipy
from numpy.ma import copy
from tqdm.notebook import tqdm
import time
import torch
import json
import cv2
import scipy.io.wavfile
# from audioset import vggish_input, vggish_postprocess


def voting(audio_folder, voting_dir, model_pretrained, device, save_size=64):
    print('Start voting for task2...')
    mfcc_MAX_VALUE = 194.19187653405487
    mfcc_MIN_VALUE = -313.07119549054045

    t2_MAX_VALUE = 57.464638
    t2_MIN_VALUE = -1.1948369
    start = time.time()

    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_data = {}
    data_num = 0
    filling_type_list = []
    for i, path in enumerate(tqdm(audio_paths)):
        count_pred = [0] * 4
        pred_list = []
        sample_rate, signal = scipy.io.wavfile.read(path)
        ap = AudioProcessing(sample_rate, signal)
        mfcc = ap.calc_MFCC()
        mfcc_length = mfcc.shape[0]
        f_step = int(mfcc.shape[1] * 0.25)
        f_length = mfcc.shape[1]
        save_mfcc_num = int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
        for i in range(save_mfcc_num):
            tmp_mfcc = mfcc[i * f_step:save_size + i * f_step, :, :]
            tmp_mfcc = (tmp_mfcc - mfcc_MIN_VALUE) / (mfcc_MAX_VALUE - mfcc_MIN_VALUE)
            tmp_mfcc = tmp_mfcc.transpose(2, 0, 1)
            audio = torch.from_numpy(tmp_mfcc.astype(np.float32))
            audio = torch.unsqueeze(audio, 0)
            audio = audio.to(device)
            with torch.no_grad():
                pred_T2 = model_pretrained.forward(audio)
                _, pred_T2 = torch.max(pred_T2, 1)
                count_pred[pred_T2.item()] += 1
                pred_list.append(pred_T2.item())
        if count_pred[1] >= 5 or count_pred[2] >= 5 or count_pred[3] >= 5:
            final_pred_T2 = count_pred[1:4].index(max(count_pred[1:4])) + 1
        else:
            final_pred_T2 = 0

        filling_type_list.append(final_pred_T2)

        file_name = path.split(os.path.sep)[-1].replace('.wav', '')
        # print("sequence:{}, frequency:{}".format(file_name, count_pred))
        to_save_data = {"data_num": data_num,
                        "file": file_name,
                        "count_pred": count_pred,
                        "final_pred": final_pred_T2,
                        'pred': pred_list}
        save_data["{}".format(file_name)] = to_save_data
        data_num += 1

    with open(os.path.join(voting_dir, "voting.json"), 'w') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
        elapsed_time = time.time() - start
        print("elapsed_time:{}".format(elapsed_time) + "sec")

    return filling_type_list


def voting_t1(model, testloader, device):
    model.eval()
    loss_test = 0
    correct_test = 0
    num_val = len(testloader)
    pred_list = []
    with torch.no_grad():
        for batch_idx, (audio, lbl) in enumerate(tqdm(testloader)):
            audio = audio.to(device)
            outputs = model.forward(audio)
            _, preds = torch.max(outputs, 1)
            pred_list.append(preds.item())

    return pred_list


def audioPreprocessing_t1(audio_folder, T2_mid_dir, my_model, device):
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_size = 64
    ratio_step = 0.25
    count = 0
    MAX_VALUE = 194.19187653405487
    MIN_VALUE = -313.07119549054045

    pbar = tqdm(total=len(audio_paths))
    id = 1
    for i, path in enumerate(audio_paths):
        datalist = []
        all_means = []
        sample_rate, signal = scipy.io.wavfile.read(path)
        # ap = AudioProcessing(sample_rate, signal, nfilt=save_size)
        # mfcc = ap.calc_MFCC()
        # mfcc_length = mfcc.shape[0]
        # f_step = int(mfcc.shape[0] * ratio_step)
        # f_length = mfcc.shape[1]
        # save_signal_num = int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
        annotation_pth = os.path.join('Annotations', '{0:03d}.csv'.format(i + 1))
        print(annotation_pth)
        audio = vggish_input.waveform_to_examples(signal, sample_rate)
        data = pd.read_csv(annotation_pth, sep=',', header='infer')
        all_percentages = data['percentage'].to_numpy()
        num = audio.shape[0]
        step = int(len(all_percentages) / num)
        for m in range(num):
            tmp_percentages = all_percentages[m * step: (m + 1) * step]
            mean = np.mean(tmp_percentages)
            all_means.append(mean)
        audio = torch.from_numpy(audio).unsqueeze(dim=1)
        audio = audio.float().to(device)
        with torch.no_grad():  # 后面加上来的
            feature = my_model(audio)
            datalist.append(feature.to('cpu').detach().numpy().copy())
        datalist = np.squeeze(np.array(datalist))
        for j in range(audio.shape[0]):
            tmp_datalist = datalist[j]
            tmp_datalist = list(tmp_datalist)
            tmp_datalist.append(all_means[j])
            tmp_datalist = np.array(tmp_datalist)
            # print(tmp_datalist.shape)
            # print(tmp_datalist[-1])  # 平均液体体积
            np.save(os.path.join(T2_mid_dir, "{0:04d}".format(id)), tmp_datalist)
            id += 1

        # if signal_length < save_size:
        #     print("file {} is too short".format(id + 1))
        # else:
        #     f_step = int(signal.shape[0] * ratio_step)
        #     f_length = mfcc.shape[1]
        #     save_signal_num = int(np.ceil(float(np.abs(signal_length - save_size)) / f_step))
        #     print(f_step)
        #     print(save_signal_num)
        #
        #     for i in range(save_signal_num):
        #         tmp_signal = signal[i * f_step:save_size + i * f_step, :]
        #         print(tmp_signal.shape)
        #         # tmp_signal = (tmp_signal - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
        #         # tmp_signal = tmp_signal.transpose(2, 0, 1)  # 变换轴
        #         # audio = torch.from_numpy(tmp_signal.astype(np.float32))
        #         # audio = torch.unsqueeze(audio, 0)
        #         audio = vggish_input.waveform_to_examples(tmp_signal, 44100)
        #         print(audio.shape)
        #         audio = torch.from_numpy(audio).unsqueeze(dim=1)
        #         audio = audio.float().to(device)
        #         with torch.no_grad():  # 后面加上来的
        #             feature = my_model(audio)
        #             datalist.append(feature.to('cpu').detach().numpy().copy())
        #     datalist = np.squeeze(np.array(datalist))
        #     np.save(os.path.join(T2_mid_dir, "{0:03d}".format(id)), datalist)

        pbar.update()


def audioPreprocessing(audio_folder, gt, base_path, mfcc_path):
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_size = 64
    ratio_step = 0.25
    count = 0
    pouring_or_shaking_list = []
    file_idx_list = []
    filling_type_list = []
    pbar = tqdm(total=len(audio_paths))

    for i, path in enumerate(audio_paths):
        id = i
        start_time = gt[gt.id == id]['start'].item()
        end_time = gt[gt.id == id]['end'].item()
        filling_type = gt[gt.id == id]['filling_type'].item()
        sample_rate, signal = scipy.io.wavfile.read(path)
        ap = AudioProcessing(sample_rate, signal, nfilt=save_size)
        mfcc = ap.calc_MFCC()
        raw_frames = ap.cal_frames()
        mfcc_length = mfcc.shape[0]

        if mfcc_length < save_size:
            print("file {} is too short".format(id))
        else:
            f_step = int(mfcc.shape[1] * ratio_step)
            f_length = mfcc.shape[1]
            save_mfcc_num = int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))

            for i in range(save_mfcc_num):
                count += 1
                tmp_mfcc = mfcc[i * f_step:save_size + i * f_step, :, :]
                if start_time == -1:
                    pouring_or_shaking_list.append(0)
                elif start_time / ap.signal_length_t * mfcc_length < i * f_step + f_length * 0.75 and end_time / ap.signal_length_t * mfcc_length > i * f_step + f_length * 0.25:
                    pouring_or_shaking_list.append(1)
                else:
                    pouring_or_shaking_list.append(0)

                filling_type_list.append(filling_type)
                file_idx_list.append(id)

                np.save(os.path.join(mfcc_path, "{0:06d}".format(count)), tmp_mfcc)
        pbar.update()

    np.save(os.path.join(base_path, 'audios', 'pouring_or_shaking'), np.array(pouring_or_shaking_list))
    np.save(os.path.join(base_path, 'audios', 'filling_type'), np.array(filling_type_list))


class AudioProcessing():
    def __init__(self, sample_rate, signal, frame_length_t=0.025, frame_stride_t=0.01, nfilt=64):

        self.sample_rate = sample_rate
        self.signal = signal
        self.frame_length_t = frame_length_t
        self.frame_stride_t = frame_stride_t
        self.signal_length_t = float(signal.shape[0] / sample_rate)
        self.frame_length = int(round(frame_length_t * sample_rate))  # number of samples
        self.frame_step = int(round(frame_stride_t * sample_rate))
        self.signal_length = signal.shape[0]
        self.nfilt = nfilt
        self.num_frames = int(np.ceil(float(np.abs(self.signal_length - self.frame_length)) / self.frame_step))
        self.pad_signal_length = self.num_frames * self.frame_step + self.frame_length
        self.NFFT = 512

    def cal_frames(self):
        z = np.zeros([self.pad_signal_length - self.signal_length, 8])
        pad_signal = np.concatenate([self.signal, z], 0)
        indices = np.tile(np.arange(0, self.frame_length), (self.num_frames, 1)) + np.tile(
            np.arange(0, self.num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        return frames

    def calc_MFCC(self):
        # 291576
        pre_emphasis = 0.97
        # (n,2)
        emphasized_signal = np.concatenate(
            [self.signal[0, :].reshape([1, -1]), self.signal[1:, :] - pre_emphasis * self.signal[:-1, :]], 0)
        z = np.zeros([self.pad_signal_length - self.signal_length, 2])
        pad_signal = np.concatenate([emphasized_signal, z], 0)
        indices = np.tile(np.arange(0, self.frame_length), (self.num_frames, 1)) + np.tile(
            np.arange(0, self.num_frames * self.frame_step, self.frame_step), (self.frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames = frames * np.hamming(self.frame_length).reshape(1, -1, 1)
        frames = frames.transpose(0, 2, 1)
        mag_frames = np.absolute(np.fft.rfft(frames, self.NFFT))
        pow_frames = ((1.0 / self.NFFT) * ((mag_frames) ** 2))
        filter_banks = np.dot(pow_frames, self.cal_fbank().T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)  # dB
        filter_banks = filter_banks.transpose(0, 2, 1)

        return filter_banks

    def cal_fbank(self):

        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.sample_rate / 2) / 700))
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.nfilt + 2)
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))
        bin = np.floor((self.NFFT + 1) * hz_points / self.sample_rate)
        fbank = np.zeros((self.nfilt, int(np.floor(self.NFFT / 2 + 1))))
        for m in range(1, self.nfilt + 1):
            f_m_minus = int(bin[m - 1])  # left
            f_m = int(bin[m])  # center
            f_m_plus = int(bin[m + 1])  # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        return fbank


def videoPreprocessing_t1(audio_folder, video_folder):
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    save_size = 64
    ratio_step = 0.25
    count = 0
    MAX_VALUE = 194.19187653405487
    MIN_VALUE = -313.07119549054045
    pbar = tqdm(total=len(audio_paths))
    id = 1
    for i, path in enumerate(tqdm(audio_paths)):
        sample_rate, signal = scipy.io.wavfile.read(path)
        ap = AudioProcessing(sample_rate, signal, nfilt=save_size)
        mfcc = ap.calc_MFCC()
        mfcc_length = mfcc.shape[0]
        f_step = int(mfcc.shape[0] * ratio_step)
        f_length = mfcc.shape[1]
        save_signal_num = int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
        '''根据VGGish网络的逻辑，将音频分成每0.96s一段'''
        audio = vggish_input.waveform_to_examples(signal, 44100)
        # print(audio.shape)  # audio.shape[0]就是一个视频被分段的数目

        video_name = os.path.join(video_folder, '{:03d}.mp4'.format(i + 1))  # modify the video name here
        print(video_name)
        '''下面的函数本来是要将视频切割后再padding的，但是目前只是读取tactile的部分（每一帧图像做个一裁剪）'''
        video_frame_list = find_corres_video_frame_new(video_name, mfcc_length, ap.frame_step, ap.frame_length,
                                                       signal.shape[0])
        print(video_frame_list.shape)
        if mfcc_length < save_size:
            print("file {} is too short".format(i))
        else:
            # f_step = int(mfcc.shape[1] * ratio_step)
            f_step = int(len(video_frame_list) / audio.shape[0])
            f_length = mfcc.shape[1]
            # save_mfcc_num = int(np.ceil(float(np.abs(mfcc_length - save_size)) / f_step))
            save_mfcc_num = audio.shape[0]
            for j in range(save_mfcc_num):
                framelist = []
                tmp_frame = video_frame_list[j * f_step: (j + 1) * f_step, :, :, :]
                # print(tmp_frame.shape)
                framelist.append(tmp_frame)
                framelist.append([save_mfcc_num])  # 一个视频被分成几段
                framelist.append([j + 1])  # 当前文件是第几段
                np.save(os.path.join('train_tactile_frames', "{0:04d}".format(id)), framelist)
                id += 1
        pbar.update()


def videoPreprocessing_feature(video_folder, model, device):
    video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
    data_list = []
    for i, path in enumerate(tqdm(video_paths)):
        id = i
        data_list = []
        print(path)
        data = np.load(path, allow_pickle=True)
        video_data = data[0]
        for j in range(len(video_data)):
            tmp_data = video_data[j]
            tmp_data = tmp_data.transpose(2, 0, 1)
            tmp_data = torch.from_numpy(tmp_data.astype(np.float32))
            tmp_data = torch.unsqueeze(tmp_data, 0)
            tmp_data = tmp_data.to(device)
            with torch.no_grad():
                feature = model(tmp_data)
                # print(feature.shape)
                data_list.extend(feature.to('cpu').detach().numpy().copy())
        data_list = np.squeeze(np.array(data_list))
        np.save(os.path.join('train_tactile_features', "{0:04d}".format(id + 1)), data_list)


def find_corres_video_frame_new(video_name, mfcc_length, frame_step, frame_length, signal_length):
    capture = cv2.VideoCapture(video_name)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for i in range(int(total_frame)):
        success, image = capture.read()
        image = image[:, 1334:]  # 根据具体视频分辨率修改
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)

    video_frame_list = []
    # n = np.linspace(0, mfcc_length - 1, mfcc_length)
    # mid_point = total_frame * (frame_length / 2 + frame_step * n) / signal_length
    # extract_frame_num = np.round(mid_point).astype(np.int32)
    # extract_frame_num[extract_frame_num >= len(frames)] = -1
    # video_frame_list = np.array(frames)[extract_frame_num]
    video_frame_list = frames

    return np.array(video_frame_list)


def find_corres_video_frame(video_name, mfcc_length, frame_step, frame_length, signal_length):
    '''
    Args:
        video name
        mfcc_length : mfcc.shape[0]
        frame_step : 441 if stride_t = 0.01s
        frame_length : 1102 if frame_length_t = 0.25s
        signal_length: signal.shape[0]
    '''
    capture = cv2.VideoCapture(video_name)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    video_frame_list = []
    for n in range(mfcc_length):
        mid_point = total_frame * (frame_length / 2 + frame_step * n) / signal_length
        video_frame_num = round(mid_point)
        capture.set(1, video_frame_num)
        success, image = capture.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        video_frame_list.append(image)
    return np.array(video_frame_list)


def extract_frames(video_name):
    assert (os.path.exists(video_name))
    capture = cv2.VideoCapture(video_name)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = []
    num_list = []
    for i in range(total_frames - 1):
        success, image = capture.read()
        frame_list.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        num_list.append(i)
    return frame_list, num_list


def extract_depths(name, nums, path):
    depths_list = []
    for i in nums:
        depth_frame_name = os.path.join(path, name, '{:04d}.png'.format(i))
        depth_frame = cv2.imread(depth_frame_name)
        depths_list.append(depth_frame)
    return depths_list


def extract_depths_all(name, nums, path):
    depths_list = []
    depths = os.listdir(os.path.join(path, name))
    depths.sort()
    for depth in depths:
        depth_frame_name = os.path.join(path, name, depth)
        depth_frame = cv2.imread(depth_frame_name)
        depths_list.append(depth_frame)
    return depths_list


def extract_frames_sample(video_name, sample_number=10):
    """
    Args:
        video_name
        sample_number: how many frames wish to extract
    
    return:
        frames : list of all the frames in BGR format
        frame_numbers :list of the index of each extracted frames
    """
    assert (os.path.exists(video_name))
    capture = cv2.VideoCapture(video_name)
    frames = []

    total_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_nums = list(range(0, int(total_frames), int(round(total_frames / sample_number))))
    frame_nums_success = []

    capture.set(1, 0)
    success, image = capture.read()
    frames.append(image)
    frame_nums_success.append(0)

    capture.set(1, total_frames - 1)
    success, image = capture.read()
    frames.append(image)
    frame_nums_success.append(total_frames - 1)

    for i in frame_nums:
        capture.set(1, i)
        success, image = capture.read()
        if success:
            frames.append(image)
            frame_nums_success.append(i)

    return frames, frame_nums_success


def crop_depth_image(image, xmin, ymin, xmax, ymax, margin=0.05):
    """
    corp the depth image
    """
    image = image[int(ymin * (1. - margin)):int(ymax * (1. + margin)),
            int(xmin * (1. - margin)):int(xmax * (1. + margin))]
    image = image[:, :, 0]
    return image


def crop_rgb_image(image, xmin, ymin, xmax, ymax, margin=0.05):
    """
    corp the depth image
    """
    image = image[int(ymin * (1. - margin)):int(ymax * (1. + margin)),
            int(xmin * (1. - margin)):int(xmax * (1. + margin))]

    return image


def crop_images(results, frames, depths, object_list):
    # results = results.pandas().xyxy
    cropped_rgb_list = []
    cropped_depth_list = []

    for i in range(len(results)):
        p = results[i]
        p = p[(p.ymax - p.ymin) > 100]
        p = p[p['name'].isin(object_list)]
        p = p.reset_index(drop=True)

        if len(p) == 1:  # only detect one
            cropped_rgb = crop_rgb_image(frames[i], p.xmin.item(), p.ymin.item(), p.xmax.item(), p.ymax.item())
            cropped__depth = crop_depth_image(depths[i], p.xmin.item(), p.ymin.item(), p.xmax.item(), p.ymax.item())
            cropped_rgb_list.append(cropped_rgb)
            cropped_depth_list.append(cropped__depth)

        elif len(p) == 2 and p['name'][0] == 'cup' and p['name'][1] == 'cup':  # two cups

            if p.xmax[0] < p.xmax[1]:
                cropped_rgb = crop_rgb_image(frames[i], p.xmin[1], p.ymin[1], p.xmax[1], p.ymax[1])
                cropped_depth = crop_depth_image(depths[i], p.xmin[1], p.ymin[1], p.xmax[1], p.ymax[1])
                cropped_rgb_list.append(cropped_rgb)
                cropped_depth_list.append(cropped_depth)

            else:  # 0>1
                cropped_rgb = crop_rgb_image(frames[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
                cropped_depth = crop_depth_image(depths[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
                cropped_rgb_list.append(cropped_rgb)
                cropped_depth_list.append(cropped_depth)

        elif (p['name'] == 'wine glass').any():  # one wine glass and others
            p = p[p['name'] == 'wine glass']
            p = p.reset_index(drop=True)
            cropped_rgb = crop_rgb_image(frames[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
            cropped_depth = crop_depth_image(depths[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
            cropped_rgb_list.append(cropped_rgb)
            cropped_depth_list.append(cropped_depth)

        elif (p['name'] == 'cup').any():  # one cup and others
            p = p[p['name'] == 'cup']
            p = p.reset_index(drop=True)
            cropped_rgb = crop_rgb_image(frames[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
            cropped_depth = crop_depth_image(depths[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
            cropped_rgb_list.append(cropped_rgb)
            cropped_depth_list.append(cropped_depth)

        elif len(p) != 0:
            cropped_rgb = crop_rgb_image(frames[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
            cropped_depth = crop_depth_image(depths[i], p.xmin[0], p.ymin[0], p.xmax[0], p.ymax[0])
            cropped_rgb_list.append(cropped_rgb)
            cropped_depth_list.append(cropped_depth)

    return cropped_rgb_list, cropped_depth_list


def depth2xyz(depth_map, depth_cam_matrix, flatten=True, depth_scale=1000):
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz


def sample_pointcloud(input):
    '''
    input : [npoints,3]
    '''


def get_annotation(id, input, split):
    """
    Args:
        image id (int)
        one of the lables to get :
            'id' 
            'container capacity'
            'width at the top'
            'width at the bottom'
            'height'
            'container mass'
            'filling type'
            'filling level'
        path to label folder
    
    return:
        label
    """
    anno_path = f'datasets/corsmal_mini/{split}/labels'
    anno = np.load(os.path.join(anno_path, '{:06d}.npy'.format(id)), allow_pickle=True).item()
    return anno.get(input)


def computeScoreType1(gt, _est):  # capacity  mass
    est = copy.deepcopy(_est)
    est = est.squeeze(1)

    assert (len(gt) == len(est))

    if all(x == -1 for x in est):  # check if there's -1 in est
        return 0

    indicator_f = est > -1

    ec = np.exp(-(np.abs(gt - est) / gt)) * indicator_f

    score = np.sum(ec) / len(gt)

    return score


def computeScoreType2(gt, _est):  # width top, width bottom, height

    est = copy.deepcopy(_est)

    assert (len(gt) == len(est))

    if all(x == -1 for x in est):
        return 0

    indicator_f = est > -1

    ec = np.zeros(len(est))

    err_abs = np.abs(est - gt)

    ec[err_abs < gt] = 1 - err_abs[err_abs < gt] / gt[err_abs < gt]
    ec[err_abs >= gt] = 0

    ec[(est == 0) * (gt == 0)] = 1

    score = np.sum(ec * indicator_f) / len(gt)

    return score
