"""
可控 AEC 场景数据生成工具。

本文件是项目级通用工具模块，不绑定某个具体实验。

主要用途：
    1. 生成可控线性回声场景，用于验证 LMS / FDAF 是否正确；
    2. 生成可控非线性回声场景，用于验证 WL-LMS / GH-WL-LMS-Fast 残差补偿；
    3. 支持自制 RIR，也支持加载公开 RIR 数据；
    4. 支持绘制 RIR，并返回 RIR 长度、主峰延迟等信息。

典型链路：
    farend x(n)
        ↓
    可选扬声器/功放非线性
        ↓
    RIR 卷积
        ↓
    echo(n)
        ↓
    可选加入 nearend / noise
        ↓
    mic(n)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


# ============================================================
# 基础工具函数
# ============================================================

def to_1d_float64(x) -> np.ndarray:
    """
    将输入转换为一维 float64 数组。

    如果输入是多通道音频，则默认取第一个通道。
    """
    x = np.asarray(x)

    if x.ndim == 0:
        x = x.reshape(1)

    if x.ndim > 1:
        x = x[:, 0]

    x = x.astype(np.float64, copy=False)
    x = np.nan_to_num(x)

    return x


def normalize_peak(x, peak: float = 0.99, eps: float = 1e-12) -> np.ndarray:
    """
    峰值归一化。

    参数：
        x:
            输入信号。

        peak:
            归一化后的最大绝对幅值。

    返回：
        归一化后的信号。
    """
    x = to_1d_float64(x)

    max_abs = max(float(np.max(np.abs(x))) if len(x) else 0.0, float(eps))

    return x * (float(peak) / max_abs)


def normalize_rms(x, target_rms: float = 0.1, eps: float = 1e-12) -> np.ndarray:
    """
    RMS 归一化。

    参数：
        x:
            输入信号。

        target_rms:
            目标 RMS。

    返回：
        RMS 调整后的信号。
    """
    x = to_1d_float64(x)

    rms = np.sqrt(np.mean(x ** 2)) + float(eps)

    return x * (float(target_rms) / rms)


def remove_dc(x) -> np.ndarray:
    """去除直流分量。"""
    x = to_1d_float64(x)
    return x - np.mean(x)


def crop_or_pad_signal(
    x,
    n_samples: int,
    mode: str = "start",
    seed: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    将信号裁剪或补零到指定长度。

    参数：
        x:
            输入信号。

        n_samples:
            目标长度。

        mode:
            "start"  ：从开头截取；
            "random" ：随机截取；
            "active" ：选取能量最高的连续片段。

        seed:
            随机种子。

    返回：
        y:
            长度为 n_samples 的信号。

        start:
            裁剪起点。如果原信号不足并补零，则 start=0。
    """
    x = to_1d_float64(x)
    n_samples = int(n_samples)

    if n_samples <= 0:
        raise ValueError("n_samples 必须为正数。")

    if len(x) == n_samples:
        return x.copy(), 0

    if len(x) < n_samples:
        y = np.zeros(n_samples, dtype=np.float64)
        y[:len(x)] = x
        return y, 0

    max_start = len(x) - n_samples

    if mode == "start":
        start = 0

    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        start = int(rng.integers(0, max_start + 1))

    elif mode == "active":
        # 用 0.5 秒左右的步长搜索能量较高的片段。
        # 这里不知道采样率，因此用目标长度的 1/20 作为保守步长。
        hop = max(1, n_samples // 20)

        best_start = 0
        best_power = -1.0

        for s in range(0, max_start + 1, hop):
            seg = x[s:s + n_samples]
            power = float(np.mean(seg ** 2))

            if power > best_power:
                best_power = power
                best_start = s

        start = best_start

    else:
        raise ValueError(f"未知裁剪模式：{mode}")

    return x[start:start + n_samples].copy(), int(start)


# ============================================================
# 音频读取与重采样
# ============================================================

def load_audio_mono(path: Union[str, Path], target_fs: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """
    读取音频文件，并转换为单通道 float64。

    优先使用 soundfile；
    如果没有 soundfile，则回退到 scipy.io.wavfile。

    参数：
        path:
            音频路径。

        target_fs:
            如果不为 None，则重采样到目标采样率。

    返回：
        audio:
            单通道音频。

        fs:
            采样率。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在：{path}")

    try:
        import soundfile as sf
        audio, fs = sf.read(str(path), always_2d=False)
        audio = np.asarray(audio)
    except Exception:
        from scipy.io import wavfile

        fs, audio = wavfile.read(str(path))
        audio = np.asarray(audio)

        if np.issubdtype(audio.dtype, np.integer):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float64) / float(max_val)
        else:
            audio = audio.astype(np.float64)

    audio = to_1d_float64(audio)

    if target_fs is not None and int(target_fs) != int(fs):
        audio = resample_if_needed(audio, fs, int(target_fs))
        fs = int(target_fs)

    return audio, int(fs)


def resample_if_needed(x, fs: int, target_fs: int) -> np.ndarray:
    """
    如果采样率不同，则重采样到目标采样率。

    使用 scipy.signal.resample_poly。
    """
    x = to_1d_float64(x)

    fs = int(fs)
    target_fs = int(target_fs)

    if fs == target_fs:
        return x

    from math import gcd
    from scipy.signal import resample_poly

    g = gcd(fs, target_fs)
    up = target_fs // g
    down = fs // g

    y = resample_poly(x, up, down)

    return to_1d_float64(y)


def find_audio_by_fileid_strict(folder: Union[str, Path], fileid: int, prefix: str = "") -> Path:
    """
    根据 fileid 严格匹配音频文件。

    注意：
        这里不能使用 *199.wav 这种宽松匹配，
        否则 fileid=199 可能误匹配到 fileid=1199。

    常见文件名：
        farend_speech_fileid_199.wav
        echo_fileid_199.wav
        fileid_199.wav
        199.wav
    """
    import re

    folder = Path(folder)
    fileid = int(fileid)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    candidates = []

    if prefix:
        candidates.extend([
            folder / f"{prefix}_fileid_{fileid}.wav",
            folder / f"{prefix}_{fileid}.wav",
        ])

    candidates.extend([
        folder / f"fileid_{fileid}.wav",
        folder / f"clean_fileid_{fileid}.wav",
        folder / f"{fileid}.wav",
    ])

    for p in candidates:
        if p.exists():
            return p

    pattern = re.compile(r"fileid_(\d+)\.wav$", re.IGNORECASE)

    hits = []
    for p in folder.glob("*.wav"):
        m = pattern.search(p.name)
        if m is None:
            continue

        fid = int(m.group(1))
        if fid == fileid:
            hits.append(p)

    hits = sorted(hits)

    if len(hits) == 1:
        return hits[0]

    if len(hits) > 1:
        raise RuntimeError(f"fileid={fileid} 在 {folder} 中匹配到多个文件：{hits}")

    raise FileNotFoundError(f"无法在 {folder} 中严格匹配 fileid={fileid} 的 wav 文件")


# ============================================================
# 远端信号生成
# ============================================================

def generate_white_noise(
    n_samples: int,
    rms: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """
    生成白噪声远端输入。

    白噪声适合用于最基础的系统辨识验证，
    因为它激励频带更均匀，方便检查 FDAF / LMS 是否正确。
    """
    rng = np.random.default_rng(int(seed))

    x = rng.standard_normal(int(n_samples))
    x = normalize_rms(x, target_rms=float(rms))

    return x


def generate_two_tone(
    n_samples: int,
    fs: int,
    f1: float = 1000.0,
    f2: float = 2700.0,
    amp: float = 0.2,
) -> np.ndarray:
    """
    生成双音信号。

    可用于测试互调失真、谐波失真等非线性现象。
    """
    n_samples = int(n_samples)
    fs = int(fs)

    t = np.arange(n_samples, dtype=np.float64) / float(fs)

    x = float(amp) * (
        np.sin(2.0 * np.pi * float(f1) * t)
        + np.sin(2.0 * np.pi * float(f2) * t)
    )

    return x.astype(np.float64)


def load_farend_from_file(
    path: Union[str, Path],
    target_fs: int,
    n_samples: int,
    crop_mode: str = "active",
    remove_dc_flag: bool = True,
    peak_normalize: bool = True,
    seed: int = 0,
) -> Dict:
    """
    从音频文件读取远端语音，并裁剪到指定长度。

    返回字典，包含：
        signal
        fs
        path
        crop_start
    """
    x, fs = load_audio_mono(path, target_fs=target_fs)

    if remove_dc_flag:
        x = remove_dc(x)

    x, start = crop_or_pad_signal(
        x,
        n_samples=int(n_samples),
        mode=crop_mode,
        seed=seed,
    )

    if peak_normalize:
        x = normalize_peak(x, peak=0.99)

    return dict(
        signal=x,
        fs=int(fs),
        path=str(path),
        crop_start=int(start),
    )


def load_farend_by_fileid(
    folder: Union[str, Path],
    fileid: int,
    target_fs: int,
    n_samples: int,
    prefix: str = "farend_speech",
    crop_mode: str = "active",
    remove_dc_flag: bool = True,
    peak_normalize: bool = True,
    seed: int = 0,
) -> Dict:
    """
    根据 fileid 从 farend_speech 文件夹读取远端语音。
    """
    path = find_audio_by_fileid_strict(folder, fileid=fileid, prefix=prefix)

    return load_farend_from_file(
        path=path,
        target_fs=target_fs,
        n_samples=n_samples,
        crop_mode=crop_mode,
        remove_dc_flag=remove_dc_flag,
        peak_normalize=peak_normalize,
        seed=seed,
    )


# ============================================================
# RIR 生成、加载与绘图
# ============================================================

def generate_exponential_rir(
    fs: int,
    rir_length_ms: float = 64.0,
    direct_delay_ms: float = 8.0,
    rt60_ms: float = 200.0,
    direct_gain: float = 1.0,
    reverb_gain: float = 0.3,
    normalize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """
    生成可控指数衰减 RIR。

    RIR 结构：
        1. 在 direct_delay_ms 位置加入直接声；
        2. 从直接声之后加入指数衰减混响尾巴；
        3. 混响尾巴使用随机符号，模拟多径反射。

    参数：
        fs:
            采样率。

        rir_length_ms:
            RIR 总长度。

        direct_delay_ms:
            直接声延迟。

        rt60_ms:
            衰减到 -60 dB 的时间常数，用于控制混响尾巴衰减速度。

        direct_gain:
            直接声增益。

        reverb_gain:
            混响尾巴整体增益。

        normalize:
            是否将 RIR 峰值归一化。

        seed:
            随机种子。

    返回：
        h:
            RIR，shape=(rir_length_samples,)
    """
    fs = int(fs)
    rng = np.random.default_rng(int(seed))

    rir_len = max(1, int(round(float(rir_length_ms) * 1e-3 * fs)))
    direct_delay = int(round(float(direct_delay_ms) * 1e-3 * fs))
    direct_delay = int(np.clip(direct_delay, 0, rir_len - 1))

    rt60_samples = max(1.0, float(rt60_ms) * 1e-3 * fs)

    h = np.zeros(rir_len, dtype=np.float64)

    # 直接声。
    h[direct_delay] = float(direct_gain)

    # 混响尾巴。
    tail_len = rir_len - direct_delay - 1

    if tail_len > 0 and float(reverb_gain) != 0.0:
        n = np.arange(tail_len, dtype=np.float64)

        # -60 dB 对应幅度 0.001。
        # envelope = exp(-n / tau)，使 n=rt60_samples 时约为 0.001。
        tau = rt60_samples / np.log(1000.0)
        envelope = np.exp(-n / tau)

        random_reflections = rng.standard_normal(tail_len)
        random_reflections = random_reflections / (np.std(random_reflections) + 1e-12)

        tail = float(reverb_gain) * envelope * random_reflections

        h[direct_delay + 1:] += tail

    if normalize:
        max_abs = np.max(np.abs(h)) + 1e-12
        h = h / max_abs

    return h.astype(np.float64)


def load_rir(
    path: Union[str, Path],
    target_fs: Optional[int] = None,
    normalize: bool = True,
) -> Tuple[np.ndarray, Optional[int]]:
    """
    加载公开 RIR 文件。

    支持：
        .wav
        .npy
        .npz

    对 wav：
        返回 rir 和采样率。

    对 npy / npz：
        如果文件中没有采样率，则 fs 返回 None。
        npz 中优先查找键：
            "rir", "h", "data", "fs", "sr", "sample_rate"

    参数：
        path:
            RIR 文件路径。

        target_fs:
            如果 RIR 是 wav，并且需要重采样，则传入目标采样率。
            对 npy/npz，若文件里有 fs，也会尝试重采样。

        normalize:
            是否峰值归一化。

    返回：
        rir:
            RIR 数组。

        fs:
            RIR 采样率。如果无法知道，则返回 None。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"RIR 文件不存在：{path}")

    suffix = path.suffix.lower()

    fs = None

    if suffix == ".wav":
        rir, fs = load_audio_mono(path, target_fs=None)

    elif suffix == ".npy":
        rir = np.load(path)
        rir = to_1d_float64(rir)

    elif suffix == ".npz":
        data = np.load(path)

        rir_key = None
        for k in ["rir", "h", "data"]:
            if k in data:
                rir_key = k
                break

        if rir_key is None:
            raise KeyError(f"{path} 中没有找到 rir/h/data 键。")

        rir = to_1d_float64(data[rir_key])

        for fs_key in ["fs", "sr", "sample_rate"]:
            if fs_key in data:
                fs = int(data[fs_key])
                break

    else:
        raise ValueError(f"不支持的 RIR 文件格式：{suffix}")

    if target_fs is not None and fs is not None and int(fs) != int(target_fs):
        rir = resample_if_needed(rir, fs, int(target_fs))
        fs = int(target_fs)

    if normalize:
        rir = normalize_peak(rir, peak=1.0)

    return rir.astype(np.float64), fs


def get_rir_info(rir, fs: Optional[int] = None, threshold_db: float = -60.0) -> Dict:
    """
    获取 RIR 基本信息。

    返回：
        length_samples:
            RIR 总长度，单位 samples。

        length_ms:
            RIR 总长度，单位 ms。如果 fs=None，则为 None。

        peak_delay_samples:
            RIR 绝对值最大点位置。

        peak_delay_ms:
            主峰延迟，单位 ms。如果 fs=None，则为 None。

        effective_length_samples:
            基于阈值的有效长度。
            从第一个超过阈值的位置到最后一个超过阈值的位置。

        effective_length_ms:
            有效长度，单位 ms。如果 fs=None，则为 None。
    """
    h = to_1d_float64(rir)

    length_samples = int(len(h))

    if length_samples == 0:
        raise ValueError("RIR 为空。")

    abs_h = np.abs(h)
    peak_idx = int(np.argmax(abs_h))
    peak_val = float(abs_h[peak_idx]) + 1e-12

    threshold = peak_val * (10.0 ** (float(threshold_db) / 20.0))
    active = np.where(abs_h >= threshold)[0]

    if len(active) > 0:
        eff_start = int(active[0])
        eff_end = int(active[-1])
        eff_len = int(eff_end - eff_start + 1)
    else:
        eff_start = 0
        eff_end = 0
        eff_len = 0

    if fs is not None:
        fs = int(fs)
        length_ms = 1000.0 * length_samples / float(fs)
        peak_delay_ms = 1000.0 * peak_idx / float(fs)
        effective_length_ms = 1000.0 * eff_len / float(fs)
    else:
        length_ms = None
        peak_delay_ms = None
        effective_length_ms = None

    return dict(
        length_samples=length_samples,
        length_ms=length_ms,
        peak_delay_samples=peak_idx,
        peak_delay_ms=peak_delay_ms,
        effective_start_samples=eff_start,
        effective_end_samples=eff_end,
        effective_length_samples=eff_len,
        effective_length_ms=effective_length_ms,
        threshold_db=float(threshold_db),
    )


def plot_rir(
    rir,
    fs: Optional[int] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Room Impulse Response",
    show: bool = False,
    threshold_db: float = -60.0,
) -> Dict:
    """
    绘制 RIR，并返回 RIR 长度信息。

    这个函数既可以用于自制 RIR，也可以用于公开 RIR 数据集加载后的 RIR。

    参数：
        rir:
            RIR 数组。

        fs:
            采样率。如果提供，则横轴单位为 ms；否则横轴单位为 samples。

        save_path:
            如果不为 None，则保存图片。

        title:
            图标题。

        show:
            是否调用 plt.show()。

        threshold_db:
            计算有效长度时使用的阈值，默认 -60 dB。

    返回：
        info:
            RIR 信息字典，包括：
                length_samples
                length_ms
                peak_delay_samples
                peak_delay_ms
                effective_length_samples
                effective_length_ms
    """
    import matplotlib.pyplot as plt

    h = to_1d_float64(rir)
    info = get_rir_info(h, fs=fs, threshold_db=threshold_db)

    if fs is not None:
        xs = 1000.0 * np.arange(len(h)) / float(fs)
        xlabel = "Time (ms)"
    else:
        xs = np.arange(len(h))
        xlabel = "Samples"

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(xs, h, linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)

    if fs is not None:
        peak_x = info["peak_delay_ms"]
        length_text = f"Length = {info['length_ms']:.2f} ms / {info['length_samples']} samples"
        peak_text = f"Peak delay = {info['peak_delay_ms']:.2f} ms"
    else:
        peak_x = info["peak_delay_samples"]
        length_text = f"Length = {info['length_samples']} samples"
        peak_text = f"Peak delay = {info['peak_delay_samples']} samples"

    ax.axvline(peak_x, linestyle="--", linewidth=1.0, label=peak_text)
    ax.legend(loc="best")

    ax.text(
        0.01,
        0.98,
        length_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)

    return info


# ============================================================
# 扬声器 / 设备非线性
# ============================================================

def speaker_nonlinearity_soft_clip(x, drive_gain: float = 1.0) -> np.ndarray:
    """
    软限幅非线性。

    公式：
        y = r_max * r / sqrt(r_max^2 + |r|^2)

    其中：
        r = drive_gain * x
    """
    x = to_1d_float64(x)

    r = float(drive_gain) * x
    r_max = max(float(np.max(np.abs(r))), 1e-12)

    y = r_max * r / np.sqrt(r_max * r_max + np.abs(r) ** 2)

    return y.astype(np.float64)


def speaker_nonlinearity_soft_sigmoid(
    x,
    delta_pos: float = 3.0,
    delta_neg: float = 3.0,
    drive_gain: float = 1.0,
) -> np.ndarray:
    """
    软限幅 + sigmoid 非线性。

    该形式对应之前 D3 中使用的文献式扬声器/功放非线性：

        r_soft = r_max * r / sqrt(r_max^2 + |r|^2)

        zeta = 3/2 * r_soft - 3/10 * r_soft^2

        y = 1 / (1 + exp(-delta * zeta)) - 1/2

    其中：
        delta = delta_pos, zeta > 0
        delta = delta_neg, zeta <= 0
    """
    x = to_1d_float64(x)

    r = float(drive_gain) * x

    r_max = max(float(np.max(np.abs(r))), 1e-12)

    r_soft = r_max * r / np.sqrt(r_max * r_max + np.abs(r) ** 2)

    zeta = 1.5 * r_soft - 0.3 * (r_soft ** 2)

    delta = np.where(zeta > 0.0, float(delta_pos), float(delta_neg))

    y = 1.0 / (1.0 + np.exp(-delta * zeta)) - 0.5

    return y.astype(np.float64)


def speaker_nonlinearity_tanh(x, drive_gain: float = 1.0) -> np.ndarray:
    """
    tanh 饱和非线性。
    """
    x = to_1d_float64(x)

    y = np.tanh(float(drive_gain) * x)

    return y.astype(np.float64)


def speaker_nonlinearity_poly_clip(
    x,
    drive_gain: float = 1.0,
    clip_value: float = 1.0,
) -> np.ndarray:
    """
    多项式失真 + 限幅。

    y = r - 0.2 r^2 + 0.1 r^3
    """
    x = to_1d_float64(x)

    r = float(drive_gain) * x

    y = r - 0.2 * (r ** 2) + 0.1 * (r ** 3)
    y = np.clip(y, -float(clip_value), float(clip_value))

    return y.astype(np.float64)


def apply_speaker_nonlinearity(
    x,
    mode: str = "identity",
    drive_gain: float = 1.0,
    delta_pos: float = 3.0,
    delta_neg: float = 3.0,
    clip_value: float = 1.0,
    center_output: bool = False,
    match_input_rms: bool = False,
) -> np.ndarray:
    """
    对远端信号施加扬声器/设备非线性。

    参数：
        mode:
            "identity"     ：无非线性；
            "soft_clip"    ：软限幅；
            "soft_sigmoid" ：软限幅 + sigmoid；
            "tanh"         ：tanh 饱和；
            "poly_clip"    ：多项式失真 + 限幅。

        drive_gain:
            非线性前级增益。越大越容易进入饱和区。

        center_output:
            是否去除输出直流分量。

        match_input_rms:
            是否让输出 RMS 与输入 RMS 一致。
            用于比较算法时避免整体能量差异过大。

    返回：
        y:
            非线性输出。
    """
    x = to_1d_float64(x)

    mode = str(mode).lower()

    if mode == "identity":
        y = x.copy()

    elif mode == "soft_clip":
        y = speaker_nonlinearity_soft_clip(
            x,
            drive_gain=drive_gain,
        )

    elif mode == "soft_sigmoid":
        y = speaker_nonlinearity_soft_sigmoid(
            x,
            delta_pos=delta_pos,
            delta_neg=delta_neg,
            drive_gain=drive_gain,
        )

    elif mode == "tanh":
        y = speaker_nonlinearity_tanh(
            x,
            drive_gain=drive_gain,
        )

    elif mode == "poly_clip":
        y = speaker_nonlinearity_poly_clip(
            x,
            drive_gain=drive_gain,
            clip_value=clip_value,
        )

    else:
        raise ValueError(f"未知非线性模式：{mode}")

    if center_output:
        y = remove_dc(y)

    if match_input_rms:
        x_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        y = normalize_rms(y, target_rms=x_rms)

    return y.astype(np.float64)


# ============================================================
# Echo 生成与混合
# ============================================================

def convolve_rir(
    x,
    rir,
    mode: str = "same_length",
) -> np.ndarray:
    """
    使用 RIR 对信号进行卷积。

    参数：
        x:
            输入信号。

        rir:
            房间冲激响应。

        mode:
            "full"        ：返回完整卷积；
            "same_length" ：返回与 x 等长的前 len(x) 个样本。

    返回：
        y:
            卷积结果。
    """
    x = to_1d_float64(x)
    h = to_1d_float64(rir)

    y_full = np.convolve(x, h, mode="full")

    if mode == "full":
        return y_full.astype(np.float64)

    if mode == "same_length":
        return y_full[:len(x)].astype(np.float64)

    raise ValueError(f"未知卷积模式：{mode}")


def generate_echo_from_rir(
    farend,
    rir,
    nonlinear_mode: str = "identity",
    nonlinear_before_rir: bool = True,
    drive_gain: float = 1.0,
    delta_pos: float = 3.0,
    delta_neg: float = 3.0,
    clip_value: float = 1.0,
    center_nonlinear_output: bool = False,
    match_nonlinear_rms: bool = False,
    echo_peak_normalize: bool = False,
) -> Dict:
    """
    根据 farend 和 RIR 生成 echo。

    支持两种链路：

    链路 A，默认：
        farend
            ↓
        speaker nonlinearity
            ↓
        RIR convolution
            ↓
        echo

    链路 B：
        farend
            ↓
        RIR convolution
            ↓
        speaker nonlinearity
            ↓
        echo

    参数：
        nonlinear_before_rir:
            True  ：非线性在 RIR 之前；
            False ：非线性在 RIR 之后。

    返回：
        {
            "farend_nonlinear": ...,
            "echo": ...
        }
    """
    x = to_1d_float64(farend)
    h = to_1d_float64(rir)

    if nonlinear_before_rir:
        x_nl = apply_speaker_nonlinearity(
            x,
            mode=nonlinear_mode,
            drive_gain=drive_gain,
            delta_pos=delta_pos,
            delta_neg=delta_neg,
            clip_value=clip_value,
            center_output=center_nonlinear_output,
            match_input_rms=match_nonlinear_rms,
        )

        echo = convolve_rir(x_nl, h, mode="same_length")

    else:
        x_lin_echo = convolve_rir(x, h, mode="same_length")

        echo = apply_speaker_nonlinearity(
            x_lin_echo,
            mode=nonlinear_mode,
            drive_gain=drive_gain,
            delta_pos=delta_pos,
            delta_neg=delta_neg,
            clip_value=clip_value,
            center_output=center_nonlinear_output,
            match_input_rms=match_nonlinear_rms,
        )

        x_nl = x.copy()

    if echo_peak_normalize:
        echo = normalize_peak(echo, peak=0.99)

    return dict(
        farend_nonlinear=x_nl.astype(np.float64),
        echo=echo.astype(np.float64),
    )


def add_noise_by_snr(
    clean,
    snr_db: Optional[float],
    seed: int = 0,
) -> np.ndarray:
    """
    按指定 SNR 给 clean 加白噪声。

    如果 snr_db=None，则返回全零噪声。
    """
    clean = to_1d_float64(clean)

    if snr_db is None:
        return np.zeros_like(clean)

    rng = np.random.default_rng(int(seed))

    noise = rng.standard_normal(len(clean))
    noise = noise - np.mean(noise)

    clean_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12

    target_noise_power = clean_power / (10.0 ** (float(snr_db) / 10.0))

    noise = noise * np.sqrt(target_noise_power / noise_power)

    return noise.astype(np.float64)


def mix_nearend_by_ser(
    echo,
    nearend,
    ser_db: Optional[float],
) -> np.ndarray:
    """
    按 SER 调整近端语音能量。

    SER 定义：
        SER = 10log10(P_nearend / P_echo)

    如果 ser_db=None，则直接返回 nearend 原始幅度。
    """
    echo = to_1d_float64(echo)
    nearend = to_1d_float64(nearend)

    n = min(len(echo), len(nearend))
    echo = echo[:n]
    nearend = nearend[:n]

    if ser_db is None:
        return nearend.astype(np.float64)

    echo_power = np.mean(echo ** 2) + 1e-12
    near_power = np.mean(nearend ** 2) + 1e-12

    target_near_power = echo_power * (10.0 ** (float(ser_db) / 10.0))
    nearend_scaled = nearend * np.sqrt(target_near_power / near_power)

    return nearend_scaled.astype(np.float64)


# ============================================================
# 统一样本生成入口
# ============================================================

def generate_controlled_aec_sample(
    config: Dict,
    seed: int = 0,
) -> Dict:
    """
    生成一个可控 AEC 样本。

    config 示例：

    {
        "fs": 16000,
        "duration_s": 10.0,

        "farend": {
            "type": "white_noise",      # "white_noise" / "two_tone" / "file" / "fileid"
            "rms": 0.1,

            # type="file" 时使用：
            "path": "...",

            # type="fileid" 时使用：
            "folder": "Dataset/farend_speech",
            "fileid": 123,
            "prefix": "farend_speech",

            "crop_mode": "active",
            "remove_dc": True,
            "peak_normalize": True,
        },

        "rir": {
            "type": "exponential",      # "exponential" / "file"
            "rir_length_ms": 64,
            "direct_delay_ms": 8,
            "rt60_ms": 200,
            "direct_gain": 1.0,
            "reverb_gain": 0.3,

            # type="file" 时使用：
            "path": "xxx.wav",
        },

        "nonlinearity": {
            "mode": "identity",         # "identity" / "soft_sigmoid" / ...
            "before_rir": True,
            "drive_gain": 1.0,
            "delta_pos": 3.0,
            "delta_neg": 3.0,
            "center_output": False,
            "match_input_rms": False,
        },

        "nearend": {
            "type": "none",             # "none" / "file"
            "path": "...",
            "ser_db": 0.0,
        },

        "noise": {
            "snr_db": None,
        },

        "normalize": {
            "mic_peak_normalize": False,
            "echo_peak_normalize": False,
        }
    }

    返回：
        {
            "farend": x,
            "farend_nonlinear": x_nl,
            "rir": h,
            "echo": echo,
            "nearend": nearend,
            "noise": noise,
            "mic": mic,
            "fs": fs,
            "metadata": {...}
        }
    """
    seed = int(seed)

    fs = int(config.get("fs", 16000))
    duration_s = float(config.get("duration_s", 10.0))
    n_samples = int(round(duration_s * fs))

    # ------------------------------------------------------------
    # 1. 生成 / 加载 farend
    # ------------------------------------------------------------

    far_cfg = dict(config.get("farend", {}))
    far_type = str(far_cfg.get("type", "white_noise")).lower()

    if far_type == "white_noise":
        farend = generate_white_noise(
            n_samples=n_samples,
            rms=float(far_cfg.get("rms", 0.1)),
            seed=seed,
        )
        farend_info = dict(type="white_noise")

    elif far_type == "two_tone":
        farend = generate_two_tone(
            n_samples=n_samples,
            fs=fs,
            f1=float(far_cfg.get("f1", 1000.0)),
            f2=float(far_cfg.get("f2", 2700.0)),
            amp=float(far_cfg.get("amp", 0.2)),
        )
        farend_info = dict(type="two_tone")

    elif far_type == "file":
        item = load_farend_from_file(
            path=far_cfg["path"],
            target_fs=fs,
            n_samples=n_samples,
            crop_mode=str(far_cfg.get("crop_mode", "active")),
            remove_dc_flag=bool(far_cfg.get("remove_dc", True)),
            peak_normalize=bool(far_cfg.get("peak_normalize", True)),
            seed=seed,
        )
        farend = item["signal"]
        farend_info = dict(
            type="file",
            path=item["path"],
            crop_start=item["crop_start"],
        )

    elif far_type == "fileid":
        item = load_farend_by_fileid(
            folder=far_cfg["folder"],
            fileid=int(far_cfg["fileid"]),
            target_fs=fs,
            n_samples=n_samples,
            prefix=str(far_cfg.get("prefix", "farend_speech")),
            crop_mode=str(far_cfg.get("crop_mode", "active")),
            remove_dc_flag=bool(far_cfg.get("remove_dc", True)),
            peak_normalize=bool(far_cfg.get("peak_normalize", True)),
            seed=seed,
        )
        farend = item["signal"]
        farend_info = dict(
            type="fileid",
            fileid=int(far_cfg["fileid"]),
            path=item["path"],
            crop_start=item["crop_start"],
        )

    else:
        raise ValueError(f"未知 farend 类型：{far_type}")

    # ------------------------------------------------------------
    # 2. 生成 / 加载 RIR
    # ------------------------------------------------------------

    rir_cfg = dict(config.get("rir", {}))
    rir_type = str(rir_cfg.get("type", "exponential")).lower()

    if rir_type == "exponential":
        rir = generate_exponential_rir(
            fs=fs,
            rir_length_ms=float(rir_cfg.get("rir_length_ms", 64.0)),
            direct_delay_ms=float(rir_cfg.get("direct_delay_ms", 8.0)),
            rt60_ms=float(rir_cfg.get("rt60_ms", 200.0)),
            direct_gain=float(rir_cfg.get("direct_gain", 1.0)),
            reverb_gain=float(rir_cfg.get("reverb_gain", 0.3)),
            normalize=bool(rir_cfg.get("normalize", True)),
            seed=seed + 1000,
        )
        rir_info = dict(
            type="exponential",
            **get_rir_info(rir, fs=fs),
        )

    elif rir_type == "file":
        rir, rir_fs = load_rir(
            path=rir_cfg["path"],
            target_fs=fs,
            normalize=bool(rir_cfg.get("normalize", True)),
        )
        rir_info = dict(
            type="file",
            path=str(rir_cfg["path"]),
            original_fs=rir_fs,
            **get_rir_info(rir, fs=fs),
        )

    else:
        raise ValueError(f"未知 RIR 类型：{rir_type}")

    # ------------------------------------------------------------
    # 3. 生成 echo
    # ------------------------------------------------------------

    nl_cfg = dict(config.get("nonlinearity", {}))
    norm_cfg = dict(config.get("normalize", {}))

    echo_dict = generate_echo_from_rir(
        farend=farend,
        rir=rir,
        nonlinear_mode=str(nl_cfg.get("mode", "identity")),
        nonlinear_before_rir=bool(nl_cfg.get("before_rir", True)),
        drive_gain=float(nl_cfg.get("drive_gain", 1.0)),
        delta_pos=float(nl_cfg.get("delta_pos", 3.0)),
        delta_neg=float(nl_cfg.get("delta_neg", 3.0)),
        clip_value=float(nl_cfg.get("clip_value", 1.0)),
        center_nonlinear_output=bool(nl_cfg.get("center_output", False)),
        match_nonlinear_rms=bool(nl_cfg.get("match_input_rms", False)),
        echo_peak_normalize=bool(norm_cfg.get("echo_peak_normalize", False)),
    )

    farend_nonlinear = echo_dict["farend_nonlinear"]
    echo = echo_dict["echo"]

    # ------------------------------------------------------------
    # 4. 生成 nearend
    # ------------------------------------------------------------

    near_cfg = dict(config.get("nearend", {}))
    near_type = str(near_cfg.get("type", "none")).lower()

    if near_type == "none":
        nearend = np.zeros_like(echo)

    elif near_type == "file":
        near_item = load_farend_from_file(
            path=near_cfg["path"],
            target_fs=fs,
            n_samples=n_samples,
            crop_mode=str(near_cfg.get("crop_mode", "active")),
            remove_dc_flag=bool(near_cfg.get("remove_dc", True)),
            peak_normalize=bool(near_cfg.get("peak_normalize", True)),
            seed=seed + 2000,
        )
        nearend_raw = near_item["signal"]
        nearend = mix_nearend_by_ser(
            echo=echo,
            nearend=nearend_raw,
            ser_db=near_cfg.get("ser_db", None),
        )

    else:
        raise ValueError(f"未知 nearend 类型：{near_type}")

    nearend = crop_or_pad_signal(nearend, n_samples=n_samples, mode="start")[0]

    # ------------------------------------------------------------
    # 5. 生成 noise
    # ------------------------------------------------------------

    noise_cfg = dict(config.get("noise", {}))
    noise = add_noise_by_snr(
        clean=echo + nearend,
        snr_db=noise_cfg.get("snr_db", None),
        seed=seed + 3000,
    )

    # ------------------------------------------------------------
    # 6. 混合 mic
    # ------------------------------------------------------------

    mic = echo + nearend + noise

    if bool(norm_cfg.get("mic_peak_normalize", False)):
        # 对 farend/echo/nearend/noise/mic 采用公共缩放，避免破坏相对比例。
        peak = max(
            float(np.max(np.abs(farend))) if len(farend) else 0.0,
            float(np.max(np.abs(echo))) if len(echo) else 0.0,
            float(np.max(np.abs(nearend))) if len(nearend) else 0.0,
            float(np.max(np.abs(noise))) if len(noise) else 0.0,
            float(np.max(np.abs(mic))) if len(mic) else 0.0,
            1e-12,
        )

        scale = 0.99 / peak

        farend = farend * scale
        farend_nonlinear = farend_nonlinear * scale
        echo = echo * scale
        nearend = nearend * scale
        noise = noise * scale
        mic = mic * scale

    # ------------------------------------------------------------
    # 7. 返回统一结构
    # ------------------------------------------------------------

    metadata = dict(
        fs=fs,
        duration_s=duration_s,
        n_samples=n_samples,
        seed=seed,
        farend=farend_info,
        rir=rir_info,
        nonlinearity=nl_cfg,
        nearend=near_cfg,
        noise=noise_cfg,
        normalize=norm_cfg,
    )

    return dict(
        farend=farend.astype(np.float64),
        farend_nonlinear=farend_nonlinear.astype(np.float64),
        rir=rir.astype(np.float64),
        echo=echo.astype(np.float64),
        nearend=nearend.astype(np.float64),
        noise=noise.astype(np.float64),
        mic=mic.astype(np.float64),
        fs=fs,
        metadata=metadata,
    )


# ============================================================
# 简单自检示例
# ============================================================

if __name__ == "__main__":
    cfg = {
        "fs": 16000,
        "duration_s": 3.0,

        "farend": {
            "type": "white_noise",
            "rms": 0.1,
        },

        "rir": {
            "type": "exponential",
            "rir_length_ms": 64,
            "direct_delay_ms": 8,
            "rt60_ms": 200,
            "direct_gain": 1.0,
            "reverb_gain": 0.3,
        },

        "nonlinearity": {
            "mode": "identity",
            "before_rir": True,
            "drive_gain": 1.0,
        },

        "nearend": {
            "type": "none",
        },

        "noise": {
            "snr_db": None,
        },
    }

    sample = generate_controlled_aec_sample(cfg, seed=0)

    print("farend:", sample["farend"].shape)
    print("rir:", sample["rir"].shape)
    print("echo:", sample["echo"].shape)
    print("mic:", sample["mic"].shape)
    print("RIR info:", sample["metadata"]["rir"])

    info = plot_rir(
        sample["rir"],
        fs=sample["fs"],
        save_path="controlled_rir_debug.png",
        title="Controlled Exponential RIR",
        show=False,
    )

    print("plot_rir 返回信息:", info)