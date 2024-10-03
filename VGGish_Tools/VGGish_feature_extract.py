from my_utils import *
from vggish import VGGish
import Onset_Dection
from audioset import vggish_input

cut_nums = []
snr = -20
sr = 44100


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


def read_tactile_frame(video_name):
    capture = cv2.VideoCapture(video_name)
    total_frame = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frames = []
    for i in range(int(total_frame)):
        success, image = capture.read()
        image = image[:, 1334:]  # select tactile part
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames.append(image)
    return np.array(frames), total_frame - 1


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

    pbar = tqdm(total=len(audio_paths))
    id = 1
    for i, path in enumerate(audio_paths):
        print(path)
        sample_rate, signal = scipy.io.wavfile.read(path)
        waveform = (signal[:, 0] + signal[:, 1]) / 2
        noise = np.random.randn(len(waveform))
        waveform, d = add_noise(waveform, noise, snr)
        wave_file_time, wave_file_data_l, wave_file_data_r, wave_file_framerate = Onset_Dection.get_audio_data(path)
        audio_cutframes = Onset_Dection.cut_signal(wave_file_data_l, 512, 128)
        audio_points = Onset_Dection.point_check(audio_cutframes)
        waveform = waveform[audio_points[0] * 128: audio_points[-1] * 128]
        MAX = np.max(waveform)
        MIN = np.min(waveform)
        # MEAN = np.mean(waveform)
        # STD = np.std(waveform)
        signal = (waveform - MIN) / (MAX - MIN)  # normalize to (0,1)
        # annotation_pth = os.path.join('Annotations', '{0:03d}.csv'.format(i + 1))
        # print(annotation_pth)
        # cut_num = cut_nums[i]
        cut_num = len(signal) // (44100 * 4)
        step_time = 4.0  # clip length
        small_step_time = 0.5  # sample length
        step_points = int(step_time * sr)  # clip length (sample rate)
        small_step_points = int(small_step_time * sr)  # sample length (sample rate)
        for j in range(cut_num):
            datalist = []
            for l in range(8):
                tmp_waveform = signal[(j * step_points) + (l * small_step_points): (j * step_points) + (
                            l + 1) * small_step_points]
                audio = vggish_input.waveform_to_examples(tmp_waveform, sample_rate)
                audio = torch.from_numpy(audio).unsqueeze(dim=1)  # (, 1, 48, 64)
                audio = audio.float().to(device)
                with torch.no_grad():
                    feature = my_model(audio)  # (1, 6144)
                    datalist.append(feature.to('cpu').detach().numpy().copy())
            datalist = np.squeeze(np.array(datalist))  # (8, 6144)
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
        pbar.update()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


audio_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'test_audio')
video_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'test_tactile')
# videoPreprocessing_t1(audio_folder, video_folder)
# print('Video Cut Done!')

# 使用ResNet50提取视频特征
# resnet50_feature_extractor = models.resnet50(pretrained=True)
# resnet50_feature_extractor.fc = nn.Linear(2048, 512)
# torch.nn.init.eye(resnet50_feature_extractor.fc.weight)
# resnet50_feature_extractor.to(device)
# resnet50_feature_extractor.eval()
# video_feature(video_folder, resnet50_feature_extractor, device)
# print('Tactile Features Extraction Done!')

os.makedirs('test_audio_features_-20dB', exist_ok=True)
feature_folder = 'test_audio_features_-20dB'
vggish_feature_extractor = VGGish()
pretrain_dict = torch.load('pytorch_vggish.pth', map_location=device)
state_dict = vggish_feature_extractor.state_dict()
model_dict = {}
for k, v in pretrain_dict.items():
    if k in state_dict:
        # print(k)
        model_dict[k] = v
state_dict.update(model_dict)
vggish_feature_extractor.load_state_dict(state_dict)
vggish_feature_extractor.to(device)
vggish_feature_extractor.eval()
audio_Preprocessing(audio_folder, feature_folder, vggish_feature_extractor, device)
print('Audio Features Extraction Done!')
