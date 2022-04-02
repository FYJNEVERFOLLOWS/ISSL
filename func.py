import torch
import apkit
import numpy as np

_FREQ_MAX = 8000
_FREQ_MIN = 100
SEG_LEN = 8192
SEG_HOP = 4096

def angular_distance(a1, a2):
    return 180 - abs(abs(a1 - a2) - 180)

def encode(y):
    def gaussian_func(gt_angle):
        # sigma = beam_width
        # sigma = 3.15
        sigma = 8
        angles = np.arange(360)
        out = np.array(np.exp(-1 * np.square(angles - 180) / sigma ** 2))
        out = np.roll(out, gt_angle - 180) # 向右 roll gt_angle - 180 / 向左 roll 180 - gt_angle
        return out

    mat_out = []
    for gt_angle in y:
        if not np.isnan(gt_angle):
            mat_out.append(gaussian_func(gt_angle))
    if not mat_out:
        return np.full(360, 0)
    # mat_out对360个角度分别取max
    mat_out = np.asarray(mat_out)
    mat_out = mat_out.transpose()
    mat_out = np.max(mat_out, axis=1)
    return mat_out

def extract_stft_real_image(sig, fs, feat_seg_idx):
    # calculate the complex spectrogram stft
    tf = apkit.stft(sig[:, feat_seg_idx * SEG_HOP : feat_seg_idx * SEG_HOP + SEG_LEN], apkit.cola_hamming, 2048, 1024, last_sample=True)
    # tf.shape: [C, num_frames, win_size] tf.dtype: complex128
    nch, nframe, _ = tf.shape # num_frames=sig_len/hop_len - 1
    # tf.shape:(4, num_frames, 2048) num_frames=7 when len_segment=8192 and win_size=2048
    # why not Nyquist 1 + n_fft/ 2?

    # trim freq bins
    max_fbin = int(_FREQ_MAX * 2048 / fs)            # 100-8kHz
    min_fbin = int(_FREQ_MIN * 2048 / fs)            # 100-8kHz
    tf = tf[:, :, min_fbin:max_fbin]
    # tf.shape: (C, num_frames, 337)

    # calculate the magnitude of the spectrogram
    mag_spectrogram = np.abs(tf)
    # print(f'mag_spectrogram.shape {mag_spectrogram.shape} mag_spectrogram.dtype {mag_spectrogram.dtype}')
    # mag_spectrogram.shape: (C, num_frames, 337) mag_spectrogram.dtype: float64

    # calculate the phase of the spectrogram
    phase_spectrogram = np.angle(tf)
    # print(f'phase_spectrogram.shape {phase_spectrogram.shape} phase_spectrogram.dtype {phase_spectrogram.dtype}')
    # imaginary_spectrogram.shape: (C, num_frames, 337) imaginary_spectrogram.dtype: float64

    # combine these two parts by the channel axis
    stft_seg_level = np.concatenate((mag_spectrogram, phase_spectrogram), axis=0)
    # print(f'stft_seg_level.shape {stft_seg_level.shape} stft_seg_level.dtype {stft_seg_level.dtype}')
    # stft_seg_level.shape: (C*2, num_frames, 337) stft_seg_level.dtype: float64

    return stft_seg_level

def get_top2_doa(output):
    fst_val, fst = torch.max(output, 0)
    temp = torch.roll(output, int(180 - fst), 0)
    temp[180 - 15 : 180 + 15] = 0
    sec_val, sec = torch.max(torch.roll(temp, int(fst - 180)), 0)
    # 返回第二个 peak 的值及两个 peak 对应的角度
    return sec_val, torch.tensor([fst.item(), sec.item()], dtype=torch.long, device=output.device)

def get_top3_doa(output):
    fst_val, fst = torch.max(output, 0)
    temp = torch.roll(output, int(180 - fst), 0)
    temp[180 - 15 : 180 + 15] = 0
    new_sps = torch.roll(temp, int(fst - 180))
    sec_val, sec = torch.max(new_sps, 0)
    temp = torch.roll(new_sps, int(180 - sec), 0)
    temp[180 - 15 : 180 + 15] = 0
    thr_val, thr = torch.max(torch.roll(temp, int(sec - 180)), 0)

    # 返回第三个 peak 的值及三个 peak 对应的角度
    return thr_val, torch.tensor([fst.item(), sec.item(), thr.item()], dtype=torch.long, device=output.device)

def get_topk_doa(output, k):
    peaks = []
    val, doa = torch.max(output, 0)
    temp = torch.roll(output, int(180 - doa), 0)
    temp[180 - 15 : 180 + 15] = 0
    peaks.append(doa)
    while k - 1 > 0:
        new_sps = torch.roll(temp, int(doa - 180))
        val, doa = torch.max(new_sps, 0)
        temp = torch.roll(new_sps, int(180 - doa), 0)
        temp[180 - 15 : 180 + 15] = 0
        peaks.append(doa)
        k -= 1
    return peaks

# 返回高于阈值的 k 个未知声源所在角度
def get_topk_doa_threshold(output, threshold):
    peaks = []
    val, doa = torch.max(output, 0)
    temp = torch.roll(output, int(180 - doa), 0)
    temp[180 - 15 : 180 + 15] = 0
    peaks.append(doa)
    while val > threshold:
        new_sps = torch.roll(temp, int(doa - 180))
        val, doa = torch.max(new_sps, 0)
        temp = torch.roll(new_sps, int(180 - doa), 0)
        temp[180 - 15 : 180 + 15] = 0
        if val > threshold:
            peaks.append(doa)

    return peaks

def pop_peak(sps):
    # remove and return the highest peak
    val, doa = torch.max(sps, 0)
    temp = torch.roll(sps, int(180 - doa), 0)
    temp[180 - 15 : 180 + 15] = 0
    new_sps = torch.roll(temp, int(doa - 180))

    return doa, new_sps
