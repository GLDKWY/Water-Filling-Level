import wave
import numpy as np
import matplotlib.pyplot as plt


def get_audio_data(file_path):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    # print(params)
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()

    # print(sampwidth)
    # print("str_data", len(str_data))
    audio_data = np.frombuffer(str_data, dtype=np.short)
    # print("audio_data", len(audio_data))
    audio_data.shape = -1, 2
    audio_data = audio_data.T
    # print(len(audio_data_r))
    audio_data_l = audio_data[0] * 1.0 / (max(abs(audio_data[0])))  # normalize amplitude and keep the left channel
    audio_data_r = audio_data[1] * 1.0 / (max(abs(audio_data[0])))  # normalize amplitude and keep the right channel
    # time = np.arange(0, nframes) * (1.0 / framerate)
    time = np.arange(0, nframes)
    # print("nframes", nframes)
    # print("time", len(time))
    # print("audio_data_r", len(audio_data_l))
    return time, audio_data_l, audio_data_r, framerate


def cut_signal(sig, nw=512, inc=128):  # nw: frame length，inc: frame step length
    signal_length = len(sig)
    if signal_length <= nw:
        nf = 1
    else:
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))  # split frames and discard the last frame
    pad_length = int((nf - 1) * inc + nw)
    pad_signal = np.pad(sig, (0, pad_length - signal_length), 'constant')  # padding with 0
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (
        nw, 1)).T

    # print(indices)
    indices = np.array(indices, dtype=np.int32)
    # print(indices)
    frames = pad_signal[indices]
    # print(frames)
    win = np.tile(np.hamming(nw), (nf, 1))
    # print(win)
    return frames * win


def compute_zcr(cut_sig):
    # cross zero rate(CZR)
    frames = cut_sig.shape[0]
    frame_size = cut_sig.shape[1]
    _zcr = np.zeros((frames, 1))
    np.set_printoptions(threshold=2000)

    for i in range(frames):
        frame = cut_sig[i]
        tmp1 = frame[:frame_size - 1]
        # print(tmp1)
        tmp2 = frame[1:frame_size]
        # print(tmp2)
        signs = tmp1 * tmp2.T < 0
        # print(signs)
        diff = np.abs((tmp1 - tmp2)) > 0.02
        _zcr[i] = np.sum(signs.T * diff)
    return _zcr


def compute_amp(cut_sig):
    # short time energy(STE)
    frames = cut_sig.shape[0]
    cut_en = np.zeros((frames, 1))
    for i in range(frames):
        frame = cut_sig[i]
        # print(cut_sig[i])
        sigs = cut_sig[i] * cut_sig[i]
        # print(sigs)
        cut_en[i] = np.sum(sigs)

    return cut_en


def point_check(cut_sig):
    zcr = compute_zcr(cut_sig)
    amp = compute_amp(cut_sig)
    # np.set_printoptions(threshold=np.inf)

    zcr_low = max(np.round(np.mean(zcr) * 0.1), 3)  # ZCR thresh hold
    zcr_high = max(np.round(max(zcr) * 0.1), 5)
    amp_low = min([np.mean(amp) * 0.02, max(amp) * 0.1])  # STE thresh hold
    amp_high = max([min(amp) * 10, np.mean(amp) * 0.2, max(amp) * 0.1])
    # print(zcr_low, zcr_high, amp_low, amp_high)
    max_slice = 30  # 12
    min_audio = 1  # 20
    sig_length = len(zcr)
    status = 0
    hold_time = 0
    slice_time = 0
    start_point = 0
    points = []
    for i in range(len(zcr)):
        if status == 0 or status == 1:
            if amp[i] > amp_high or zcr[i] > zcr_high:
                start_point = i - hold_time
                status = 2
                hold_time += 1
                slice_time = 0
            elif amp[i] > amp_low or zcr[i] > zcr_low:
                status = 1
                hold_time += 1
            else:
                status = 0
                hold_time = 0
        elif status == 2:
            if amp[i] > amp_low or zcr[i] > zcr_low:
                hold_time += 1
            else:
                slice_time += 1
                if slice_time < max_slice and i < sig_length - 1:
                    hold_time += 1
                elif (hold_time - slice_time) < min_audio:
                    status = 0
                    hold_time = 0
                    slice_time = 0
                else:
                    points.append(start_point)
                    hold_time = hold_time - slice_time
                    end_point = start_point + hold_time
                    points.append(end_point)
                    status = 0
    return points  #


def plt_audio_file():
    plt.figure(1, figsize=(10, 10))
    # plt.subplot(3, 1, 1)
    time = wave_file_time
    y = wave_file_data_l
    plt.plot(time, y)
    # plt.title('max_slice = 40')
    # for p in audio_points:
    #     plt.axvline(p * 128 / wave_file_framerate, c='r')
    plt.axvline(audio_points[0] * 128, c='r')
    plt.axvline(audio_points[-1] * 128, c='r')
    # time = np.arange(0, len(audio_en), 1)
    # plt.subplot(3, 1, 2)
    # y = audio_en
    # plt.plot(time * (1.0 / wave_file_framerate) * 128, y, c="red")
    # plt.subplot(3, 1, 3)
    # y = audio_zcr
    # plt.plot(time * (1.0 / wave_file_framerate) * 128, y, c="green")
    plt.show()


def plt_audio(time, y, points=[]):
    plt.flag()
    plt.plot(time, y)
    # for p in points:
    #     plt.axvline(p * 128 / wave_file_framerate, c='r')
    plt.show()


if __name__ == '__main__':
    wave_file_time, wave_file_data_l, wave_file_data_r, wave_file_framerate = get_audio_data(
        "test_audio/009.wav")

    audio_cutframes = cut_signal(wave_file_data_l, 512, 128)
    print("audio_cutframes", len(audio_cutframes))

    audio_zcr = compute_zcr(audio_cutframes)
    print("audio_zcr", len(audio_zcr))

    audio_en = compute_amp(audio_cutframes)
    print("audio_en", len(audio_en))

    audio_points = point_check(audio_cutframes)
    plt_audio_file()
    print(len(audio_points))
