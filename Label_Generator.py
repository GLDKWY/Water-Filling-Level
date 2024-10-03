import librosa
import math
from my_utils import *
import Onset_Dection

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# # 设置随机数种子
# setup_seed(20)

sr = 44100


def read_tactile_frame(video_name):
    capture = cv2.VideoCapture(video_name)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for i in range(int(total_frame)):
        success, image = capture.read()
        image = image[:, 1334:]  # 根据具体视频分辨率修改
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    return np.array(frames), total_frame - 1


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


def video_feature(video_folder, model, device):
    global cut_nums
    video_paths = [os.path.join(video_folder, path) for path in sorted(os.listdir(video_folder))]
    id = 1
    f_step = 96
    for i, path in enumerate(tqdm(video_paths)):
        # video_name = os.path.join(video_folder, '{:03d}.mp4'.format(i + 1))  # modify the video name here
        print(path)
        video_frame_list, total_frame = read_tactile_frame(path)
        # print(video_frame_list.shape)
        cut_num = int(total_frame // f_step)  # 24帧一个sample，一个sample 1.00s
        cut_nums.append(cut_num)
        # video_frame_list = video_frame_list[:int(cut_num * f_step)]  # 简单padding，截去尾部多余的部分
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
        # print(sum(cut_nums))


def audio_feature(audio_folder, label_path, device):
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    label_id = 1
    for i, path in enumerate(audio_paths):
        print(path)
        sample_rate, signal = scipy.io.wavfile.read(path)
        waveform = (signal[:, 0] + signal[:, 1]) / 2
        # waveform, sr = librosa.core.load(path, sr=44100, mono=True)
        # print(waveform.shape)
        # 音频起始、结束时刻裁剪
        # wave_file_time, wave_file_data_l, wave_file_data_r, wave_file_framerate = Onset_Dection.get_audio_data(path)
        # audio_cutframes = Onset_Dection.cut_signal(wave_file_data_l, 512, 128)
        # audio_points = Onset_Dection.point_check(audio_cutframes)
        # waveform = waveform[audio_points[0] * 128: audio_points[-1] * 128]
        # waveform = librosa.resample(waveform, sample_rate, 32000)
        step_time = 4.0  # 一小段音频为4.0s
        small_step_time = 0.5
        step_points = int(step_time * sr)  # 4.0s中包括多少音频幅度值
        small_step_points = int(small_step_time * sr)  # 0.5s中包括多少音频幅度值
        cut_num = len(waveform) // (sr * 4)
        waveform = waveform[:cut_num * step_points]  # 做padding，截去最后不满足4.0s分段的部分
        # print(waveform.shape)
        annotation_pth = os.path.join('Sand_Annotations', '{0:03d}.csv'.format(i + 1))
        print(annotation_pth)
        data = pd.read_csv(annotation_pth, sep=',', header='infer')
        all_percentages = data['percentage'].to_numpy()
        # all_percentages = all_percentages[int(audio_points[0] * 128 / sr * 24):
        #                                   int(audio_points[-1] * 128 / sr * 24)]
        all_percentages = all_percentages[:cut_num * 96]
        for m in range(cut_num):
            all_means = []
            for n in range(8):  # 每4.0s再被分成8小份，一份0.5s
                tmp_percentages = all_percentages[(m * 96) + (n * 12): (m * 96) + (n + 1) * 12]
                mean = np.mean(tmp_percentages)
                if math.isnan(mean):
                    print('nan', mean, m, annotation_pth)
                all_means.append(mean)
            np.save(os.path.join(label_path, "{0:05d}".format(label_id)), all_means)
            label_id += 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
audio_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'Sand_Audio')


label_path = r'E:/Orignal_F/PyCharm/pythonProject3/labels/Sand_labels'
os.makedirs(label_path, exist_ok=True)
audio_feature(audio_folder, label_path, device)
print('Audio Features Extraction Done!')
