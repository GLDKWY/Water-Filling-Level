import os
import scipy.io.wavfile
import cv2
import torch
import torch.nn as nn
import numpy as np
import math
import Onset_Dection

snr = -20
sr = 44100
feature_folder = 'test_audio_features_-20dB'
os.makedirs(feature_folder, exist_ok=True)
audio_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'test_audio')
video_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'test_tactile')


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


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # coordinate attention
                CoordAtt(hidden_dim, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        y = self.conv(x)
        if self.identity:
            return x + y
        else:
            return y


class MBV2_CA(nn.Module):
    def __init__(self, in_c=4, num_classes=1, width_mult=1.):
        super(MBV2_CA, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(in_c, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(output_channel, num_classes)
        )

        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def extract(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        feature = x.view(x.size(0), -1)
        x = self.classifier(feature)

        return feature, x

    def _initialize_weights(self):
        for m in self.modules():
            # print(m)
            if isinstance(m, nn.Conv2d):
                # print(m.weight.size())
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


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

        # (n,8)
        emphasized_signal = np.concatenate(
            [self.signal[0, :].reshape([1, -1]), self.signal[1:, :] - pre_emphasis * self.signal[:-1, :]], 0)
        z = np.zeros([self.pad_signal_length - self.signal_length, 8])
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


cut_nums = []


def read_tactile_frame(video_name):
    capture = cv2.VideoCapture(video_name)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for i in range(int(total_frame)):
        success, image = capture.read()
        image = image[:, 1334:]  # select the tactile part
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    return np.array(frames), total_frame - 1


def video_feature(video_folder, model, device):
    global cut_nums
    video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
    id = 1
    f_step = 96
    for i, path in enumerate(video_paths):
        # video_name = os.path.join(video_folder, '{:03d}.mp4'.format(i + 1))  # modify the video name here
        print(path)
        video_frame_list, total_frame = read_tactile_frame(path)
        # print(video_frame_list.shape)
        cut_num = int(total_frame // f_step)  # 96 frames a clip (4s)
        cut_nums.append(cut_num)
        # video_frame_list = video_frame_list[:int(cut_num * f_step)]  # padding
        # print(video_frame_list.shape)
        # for j in range(cut_num):
        #     framelist = []
        #     tmp_frame = video_frame_list[j * f_step: (j + 1) * f_step, :, :, :]  # (6, 1080, 586, 3)
        #     # framelist.append(tmp_frame)
        #     # framelist.append([sample_num])  # 一个视频被分成几段
        #     # framelist.append([j + 1])  # 当前文件是第几段
        #     # np.save(os.path.join('train_tactile_frames', "{0:04d}".format(id)), framelist)
        #     data_list = []
        #     for j in range(len(tmp_frame)):
        #         tmp_data = tmp_frame[j]  # (1080, 586, 3)
        #         tmp_data = tmp_data.transpose(2, 0, 1)
        #         tmp_data = torch.from_numpy(tmp_data.astype(np.float32))
        #         tmp_data = torch.unsqueeze(tmp_data, 0)
        #         tmp_data = tmp_data.to(device)
        #         with torch.no_grad():
        #             feature = model(tmp_data)
        #             # print(feature.shape)
        #             data_list.extend(feature.to('cpu').detach().numpy().copy())
        #     data_list = np.squeeze(np.array(data_list))  # (36, 256)
        #     print(data_list.shape)
        # np.save(os.path.join('train_tactile_features', "{0:05d}".format(id)), data_list)
        # id += 1
        print(sum(cut_nums))


def audio_Preprocessing(audio_folder, T2_mid_dir, my_model, device):
    global cut_nums
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    MAX_VALUE = 194.19187653405487
    MIN_VALUE = -313.07119549054045
    id = 1
    for i, path in enumerate(audio_paths):
        print(path)
        sample_rate, signal = scipy.io.wavfile.read(path)
        mono_signal = (signal[:, 0] + signal[:, 1]) / 2
        noise = np.random.randn(len(mono_signal))
        mono_signal, d = add_noise(mono_signal, noise, SNR=snr)
        waveform = np.zeros((len(mono_signal), 8))  # convert on channel audio to eight channels by simply copy
        for j in range(8):
            waveform[:, j] = mono_signal
        # print(waveform.shape)
        wave_file_time, wave_file_data_l, wave_file_data_r, wave_file_framerate = Onset_Dection.get_audio_data(path)
        audio_cutframes = Onset_Dection.cut_signal(wave_file_data_l, 512, 128)
        audio_points = Onset_Dection.point_check(audio_cutframes)
        signal = waveform[audio_points[0] * 128: audio_points[-1] * 128, :]
        cut_num = len(signal) // (44100 * 4)
        step_time = 4.0  # clip length
        small_step_time = 0.5  # sample length
        step_points = int(step_time * sr)  # clip length (sample rate)
        small_step_points = int(small_step_time * sr)  # sample length (sample rate)
        for j in range(cut_num):
            datalist = []
            for l in range(8):
                audio = signal[(j * step_points) + (l * small_step_points): (j * step_points) + (
                        l + 1) * small_step_points, :]
                ap = AudioProcessing(sample_rate, audio, nfilt=64)
                mfcc = ap.calc_MFCC()  # (48, 8, 64)
                mfcc = (mfcc - MIN_VALUE) / (MAX_VALUE - MIN_VALUE)
                mfcc = mfcc.transpose(2, 0, 1)
                mfcc = torch.from_numpy(mfcc.astype(np.float32))
                mfcc = torch.unsqueeze(mfcc, 0)
                mfcc = mfcc.to(device)
                with torch.no_grad():
                    feature, x = my_model.extract(mfcc)
                    datalist.append(feature.to('cpu').detach().numpy().copy())
            datalist = np.squeeze(np.array(datalist))  # (8, 48, 128)
            print(datalist.shape)
            np.save(os.path.join(T2_mid_dir, "{0:05d}".format(id)), datalist)
            id += 1
        # data = pd.read_csv(annotation_pth, sep=',', header='infer')
        # all_percentages = data['percentage'].to_numpy()
        # num = audio.shape[0] - (audio.shape[0] % 4)
        # print('asd', cut_nums[i])
        # step = int(len(all_percentages) / num)
        # for m in range(num):
        #     tmp_percentages = all_percentages[m * step: (m + 1) * step]
        #     mean = np.mean(tmp_percentages)
        #     all_means.append(mean)


model = MBV2_CA(in_c=8, num_classes=4)
model.load_state_dict(torch.load('task1_ae.pth'))
# state_dict = torch.load(r'E:/Orignal_F/PyCharm/Squids/weights/task1_ae.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)
model.to(device)
model.eval()

# videoPreprocessing_t1(audio_folder, video_folder)
# print('Video Cut Done!')

# 使用ResNet50提取视频特征
# weights = detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
# resnet50_feature_extractor = detection.maskrcnn_resnet50_fpn(weights=weights)
# resnet50_feature_extractor = models.resnet50(pretrained=True)
# resnet50_feature_extractor.fc = nn.Linear(2048, 512)
# torch.nn.init.eye(resnet50_feature_extractor.fc.weight)  # 用单位矩阵填充二维输入Tensor。保留 Linear 层中输入的标识，其中尽可能多地保留输入。
# resnet50_feature_extractor.to(device)
# resnet50_feature_extractor.eval()
# video_feature(video_folder, resnet50_feature_extractor, device)
# print('Tactile Features Extraction Done!')

audio_Preprocessing(audio_folder, feature_folder, model, device)
print('Audio Features Extraction Done!')
