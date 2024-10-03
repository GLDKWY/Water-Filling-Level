import Onset_Dection
from my_utils import *
from PANNS_utils import *
from PANNS_models import *
import math


RAND_SEED = 0
snr = 0
sr = 32000
base_path = 'Different_Content_audio_features'
os.makedirs(base_path, exist_ok=True)
audio_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'Different_Content_origin/audio')
video_folder = os.path.join('E:/Orignal_F/PyCharm/pythonProject3', 'train_tactile')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(RAND_SEED)

cut_nums = []


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


def audio_feature(audio_folder, audio_feature_path, my_model, device):
    global cut_nums
    audio_paths = [os.path.join(audio_folder, path) for path in sorted(os.listdir(audio_folder))]
    label_id = 1
    audio_feature_id = 1
    for i, path in enumerate(audio_paths):
        print(path)
        # (waveform, _) = librosa.core.load(path, sr=44100, mono=True)
        sample_rate, signal = scipy.io.wavfile.read(path)
        waveform = (signal[:, 0] + signal[:, 1]) / 2
        # noise = np.random.randn(len(waveform))
        # waveform, d = add_noise(waveform, noise, snr)
        wave_file_time, wave_file_data_l, wave_file_data_r, wave_file_framerate = Onset_Dection.get_audio_data(path)
        audio_cutframes = Onset_Dection.cut_signal(wave_file_data_l, 512, 128)
        audio_points = Onset_Dection.point_check(audio_cutframes)
        waveform = waveform[audio_points[0] * 128: audio_points[-1] * 128]
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=32000)
        # plt.plot(waveform)
        # plt.show()
        MAX = np.max(waveform)
        MIN = np.min(waveform)
        # MEAN = np.mean(waveform)
        # STD = np.std(waveform)
        waveform = (waveform - MIN) / (MAX - MIN)  # normalize to (0,1)
        # waveform = (waveform - 0.5) * 2
        # waveform = (waveform - MEAN) / STD  # standardize
        step_time = 4.0  # clip length
        small_step_time = 0.5  # sample length
        step_points = int(step_time * sr)  # clip length (sample rate)
        cut_num = len(waveform) // (32000 * 4)
        print('cut_num', cut_num)
        waveform = waveform[:cut_num * step_points]  # padding and discard clips shorter than 4s
        # print(waveform.shape)
        waveform = move_data_to_device(waveform, device)
        # annotation_pth = os.path.join(r'E:\Orignal_F\PyCharm\pythonProject3\021', '{0:03d}.csv'.format(120))
        # print(annotation_pth)
        # data = pd.read_csv(annotation_pth, sep=',', header='infer')
        # all_percentages = data['percentage'].to_numpy()
        # all_percentages = all_percentages[audio_points[0] * 128 // sr * 24: audio_points[-1] * 128 // sr * 24]
        # all_percentages = all_percentages[: cut_num * 96]
        # for m in range(cut_num):
        #     all_means = []
        #     for n in range(8):
        #         tmp_percentages = all_percentages[(m * 96) + (n * 12): (m * 96) + (n + 1) * 12]
        #         mean = np.mean(tmp_percentages)
        #         if math.isnan(mean):
        #             print('nan', mean, m, annotation_pth)
        #         all_means.append(mean)
        #     np.save(os.path.join('', "{0:05d}".format(label_id)), all_means)
        #     label_id += 1

        for j in range(cut_num):
            tmp_waveform = waveform[j * step_points: ((j + 1) * step_points)]
            tmp_waveform = tmp_waveform[None, :]  # (1, audio_length)
            with torch.no_grad():
                Wavegram, Log_mel_spectrogram = my_model(tmp_waveform, None)
                Wavegram = Wavegram.data.cpu().numpy()
                Wavegram = Wavegram.squeeze()
                Log_mel_spectrogram = Log_mel_spectrogram.data.cpu().numpy()
                Log_mel_spectrogram = Log_mel_spectrogram.squeeze()
            embedding = np.concatenate((Wavegram, Log_mel_spectrogram), axis=0)
            # print(embedding.shape)
            np.save(os.path.join(audio_feature_path, "{0:05d}".format(audio_feature_id)), embedding)
            audio_feature_id += 1


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# resnet50_feature_extractor = models.resnet50(pretrained=True)
# resnet50_feature_extractor.fc = nn.Linear(2048, 256)
# torch.nn.init.eye(resnet50_feature_extractor.fc.weight)
# resnet50_feature_extractor.to(device)
# resnet50_feature_extractor.eval()
# os.makedirs('test_tactile_features', exist_ok=True)
# video_feature(video_folder, resnet50_feature_extractor, device)
# print('Tactile Features Extraction Done!')

panns_feature_extractor = Wavegram_Logmel_Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=20,
                                                fmax=10000, classes_num=527)

pretrain_dict = torch.load('Wavegram_Logmel_Cnn14_mAP=0.439.pth', map_location=device)
state_dict = panns_feature_extractor.state_dict()
model_dict = {}
for k, v in pretrain_dict['model'].items():
    if k in state_dict:
        # print(k)
        model_dict[k] = v
state_dict.update(model_dict)
panns_feature_extractor.load_state_dict(state_dict)
panns_feature_extractor.to(device)
panns_feature_extractor.eval()
audio_feature(audio_folder, base_path, panns_feature_extractor, device)
print('Audio Features Extraction Done!')
