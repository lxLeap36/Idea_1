"""
AEC Challenge 合成数据集读取工具。

本文件只服务实验 D，不影响实验 A/B/C。

第一版实验目标：
    用 farend_speech 作为输入 x(n)
    用 echo_signal 作为目标 d(n)

也就是先做远端单讲下的非线性回声建模。
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


def load_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    """
    读取 wav 文件，并转为单通道 float64。

    优先使用 soundfile；
    如果没有 soundfile，则回退到 scipy.io.wavfile。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"音频文件不存在：{path}")

    try:
        import soundfile as sf
        data, fs = sf.read(str(path), always_2d=False)
    except Exception:
        from scipy.io import wavfile
        fs, data = wavfile.read(str(path))

        data = np.asarray(data)

        if np.issubdtype(data.dtype, np.integer):
            max_val = np.iinfo(data.dtype).max
            data = data.astype(np.float64) / float(max_val)
        else:
            data = data.astype(np.float64)

    data = np.asarray(data, dtype=np.float64)

    if data.ndim > 1:
        data = data[:, 0]

    data = np.nan_to_num(data)

    return data, int(fs)


def resample_if_needed(x: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    """
    如果采样率不同，则重采样到目标采样率。

    使用 scipy.signal.resample_poly，通常比直接 FFT resample 更稳。
    """
    if int(fs) == int(target_fs):
        return np.asarray(x, dtype=np.float64)

    from math import gcd
    from scipy.signal import resample_poly

    fs = int(fs)
    target_fs = int(target_fs)

    g = gcd(fs, target_fs)
    up = target_fs // g
    down = fs // g

    y = resample_poly(x, up, down)
    return np.asarray(y, dtype=np.float64)


def remove_dc(x: np.ndarray) -> np.ndarray:
    """去除直流分量。"""
    x = np.asarray(x, dtype=np.float64)
    return x - np.mean(x)


def common_peak_normalize(x: np.ndarray, d: np.ndarray, eps: float = 1e-12):
    """
    对输入和目标做公共峰值归一化。

    注意：
        这里不是分别归一化 x 和 d，
        而是用二者共同峰值归一化，
        这样可以保留 farend 与 echo 之间的相对幅度关系。
    """
    peak = max(float(np.max(np.abs(x))), float(np.max(np.abs(d))), eps)
    return x / peak, d / peak


def find_audio_by_fileid(folder: Path, fileid: int, prefix: str = "") -> Path:
    """
    根据 fileid 在指定文件夹中查找 wav 文件。

    注意：
        这里必须严格匹配 fileid，不能用 *199.wav 这种宽松匹配，
        否则 fileid=199 会误匹配到 fileid=1199。

    合法匹配示例：
        farend_speech_fileid_199.wav
        echo_fileid_199.wav
        fileid_199.wav
        clean_fileid_199.wav
    """
    import re

    folder = Path(folder)
    fileid = int(fileid)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    # 先尝试常见精确文件名。
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

    # 再对文件夹内 wav 做严格正则匹配。
    # 只允许文件名里出现完整的 fileid_<数字>，并且数字必须等于目标 fileid。
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
        raise RuntimeError(
            f"fileid={fileid} 在 {folder} 中匹配到多个文件：{hits}"
        )

    raise FileNotFoundError(
        f"无法在 {folder} 中严格匹配 fileid={fileid} 的 wav 文件"
    )

def collect_available_fileids(folder: Path) -> set:
    """
    扫描指定音频文件夹，收集实际存在的 fileid。

    例如：
        farend_speech_fileid_1199.wav -> 1199
        echo_fileid_1199.wav          -> 1199
    """
    import re

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在：{folder}")

    pattern = re.compile(r"fileid_(\d+)\.wav$", re.IGNORECASE)

    ids = set()

    for p in folder.glob("*.wav"):
        m = pattern.search(p.name)
        if m is not None:
            ids.add(int(m.group(1)))

    return ids

def filter_meta_by_available_audio(
    df: pd.DataFrame,
    farend_dir: Path,
    echo_dir: Path,
) -> pd.DataFrame:
    """
    只保留 farend_speech 和 echo_signal 文件夹中都实际存在的 fileid。

    这样可以避免：
        meta 中 fileid=199
        但本地只下载了 fileid=1199
        然后错误匹配或报错。
    """
    farend_ids = collect_available_fileids(farend_dir)
    echo_ids = collect_available_fileids(echo_dir)

    common_ids = farend_ids.intersection(echo_ids)

    out = df[df["fileid"].astype(int).isin(common_ids)].copy()
    out = out.reset_index(drop=True)

    print(f"[available] farend_speech 文件数：{len(farend_ids)}")
    print(f"[available] echo_signal 文件数：{len(echo_ids)}")
    print(f"[available] farend 和 echo 共同 fileid 数：{len(common_ids)}")
    print(f"[available] meta 过滤后可用样本数：{len(out)}")

    return out

def load_meta(meta_csv: Path) -> pd.DataFrame:
    """读取 meta.csv。"""
    meta_csv = Path(meta_csv)

    if not meta_csv.exists():
        raise FileNotFoundError(f"meta.csv 不存在：{meta_csv}")

    df = pd.read_csv(meta_csv)
    return df


def filter_meta_for_exp_d(
    df: pd.DataFrame,
    require_farend_nonlinear: int = 1,
    require_farend_noisy: int = 0,
    require_nearend_noisy: int = 0,
    split: Optional[str] = None,
) -> pd.DataFrame:
    """
    按实验 D 的第一版条件筛选样本。

    默认条件：
        is_farend_nonlinear == 1
        is_farend_noisy == 0
        is_nearend_noisy == 0
    """
    out = df.copy()

    out = out[out["is_farend_nonlinear"] == int(require_farend_nonlinear)]
    out = out[out["is_farend_noisy"] == int(require_farend_noisy)]
    out = out[out["is_nearend_noisy"] == int(require_nearend_noisy)]

    if split is not None:
        out = out[out["split"].astype(str) == str(split)]

    out = out.reset_index(drop=True)
    return out


def choose_active_segment(
    x: np.ndarray,
    d: np.ndarray,
    segment_samples: int,
    mode: str = "active",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    从音频中裁剪一个固定长度片段。

    mode="active"：
        选择远端语音能量最高的连续片段。
    mode="start"：
        从开头截取。
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = min(len(x), len(d))
    x = x[:n]
    d = d[:n]

    if n < segment_samples:
        raise ValueError(
            f"音频长度不足：当前 {n} 点，需要 {segment_samples} 点"
        )

    if mode == "start":
        start = 0
        return x[start:start + segment_samples], d[start:start + segment_samples], start

    if mode != "active":
        raise ValueError(f"未知 SEGMENT_MODE={mode}")

    # 用 1 秒窗口、0.5 秒步长搜索远端语音活跃区域。
    # 这里不知道采样率，因此用 segment_samples 的十分之一作为粗略窗口。
    frame = max(1024, segment_samples // 10)
    hop = max(512, frame // 2)

    best_start = 0
    best_power = -1.0

    max_start = n - segment_samples
    for s in range(0, max_start + 1, hop):
        seg = x[s:s + segment_samples]
        p = float(np.mean(seg ** 2))
        if p > best_power:
            best_power = p
            best_start = s

    return (
        x[best_start:best_start + segment_samples],
        d[best_start:best_start + segment_samples],
        best_start,
    )


def load_exp_d_pair(
    row: pd.Series,
    farend_dir: Path,
    echo_dir: Path,
    target_fs: int,
    segment_seconds: float,
    segment_mode: str = "active",
    remove_dc_flag: bool = True,
    peak_normalize: bool = True,
    peak_eps: float = 1e-12,
) -> Dict:
    """
    读取单条实验 D 样本。

    返回：
        x: 远端语音
        d: 回声信号
        fs: 目标采样率
        fileid: 样本编号
    """
    fileid = int(row["fileid"])

    farend_path = find_audio_by_fileid(farend_dir, fileid, prefix="farend_speech")
    echo_path = find_audio_by_fileid(echo_dir, fileid, prefix="echo_signal")

    x, fs_x = load_wav_mono(farend_path)
    d, fs_d = load_wav_mono(echo_path)

    x = resample_if_needed(x, fs_x, target_fs)
    d = resample_if_needed(d, fs_d, target_fs)

    n = min(len(x), len(d))
    x = x[:n]
    d = d[:n]

    if remove_dc_flag:
        x = remove_dc(x)
        d = remove_dc(d)

    segment_samples = int(round(float(segment_seconds) * int(target_fs)))

    x_seg, d_seg, start = choose_active_segment(
        x,
        d,
        segment_samples=segment_samples,
        mode=segment_mode,
    )

    if peak_normalize:
        x_seg, d_seg = common_peak_normalize(x_seg, d_seg, eps=peak_eps)

    return dict(
        x=x_seg,
        d=d_seg,
        fs=int(target_fs),
        fileid=fileid,
        farend_path=str(farend_path),
        echo_path=str(echo_path),
        segment_start_sample=int(start),
        segment_samples=int(segment_samples),
    )


def build_input_vector_stream(x: np.ndarray, filter_order: int):
    """
    流式生成自适应滤波器输入向量。

    对于每个时刻 k，生成：
        [x(k), x(k-1), ..., x(k-p+1)]

    这样可以避免一次性构造巨大的 X 矩阵。
    """
    x = np.asarray(x, dtype=np.float64)
    p = int(filter_order)

    buf = np.zeros(p, dtype=np.float64)

    for sample in x:
        buf[1:] = buf[:-1]
        buf[0] = sample
        yield buf.copy()