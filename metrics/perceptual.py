"""
语音感知评价指标工具函数。

本文件提供 PESQ 和 STOI 的统一接口，供实验 D2/D3 或后续真实 AEC 实验复用。

依赖安装：
    pip install pesq pystoi

注意：
    1. PESQ 官方常用采样率为 8000 Hz 或 16000 Hz；
    2. 16000 Hz 对应 wide-band，mode="wb"；
    3. 8000 Hz 对应 narrow-band，mode="nb"；
    4. STOI 通常用于评价语音可懂度；
    5. PESQ/STOI 的输入应该是“参考干净语音”和“待评价语音”，长度需要一致。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import math
import numpy as np


def _to_1d_float64(x) -> np.ndarray:
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


def _match_length(ref: np.ndarray, deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将参考信号和待评价信号裁剪到相同长度。
    """
    n = min(len(ref), len(deg))

    if n <= 0:
        raise ValueError("参考信号或待评价信号为空，无法计算 PESQ/STOI。")

    return ref[:n], deg[:n]


def _safe_peak_normalize_pair(
    ref: np.ndarray,
    deg: np.ndarray,
    peak: float = 0.99,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对两路信号做公共峰值归一化。

    注意：
        这里使用共同峰值，而不是分别归一化。
        这样可以尽量保留参考信号和待评价信号之间的相对幅度关系。
    """
    max_abs = max(
        float(np.max(np.abs(ref))) if len(ref) else 0.0,
        float(np.max(np.abs(deg))) if len(deg) else 0.0,
        float(eps),
    )

    scale = float(peak) / max_abs

    ref_n = ref * scale
    deg_n = deg * scale

    ref_n = np.clip(ref_n, -1.0, 1.0)
    deg_n = np.clip(deg_n, -1.0, 1.0)

    return ref_n, deg_n


def _resample_if_needed(x: np.ndarray, fs: int, target_fs: int) -> np.ndarray:
    """
    如有需要，将信号重采样到目标采样率。

    PESQ 推荐使用 8000 或 16000 Hz。
    """
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
    return np.asarray(y, dtype=np.float64)


def compute_pesq(
    ref,
    deg,
    fs: int,
    mode: Optional[str] = None,
    normalize: bool = True,
    on_error: str = "nan",
) -> float:
    """
    计算 PESQ。

    参数：
        ref:
            参考信号，例如 clean speech 或目标信号。

        deg:
            待评价信号，例如增强后语音、残差信号或模型输出。

        fs:
            采样率。PESQ 常用 8000 或 16000。

        mode:
            PESQ 模式：
                "nb"：窄带，通常 fs=8000；
                "wb"：宽带，通常 fs=16000；
            如果为 None，则根据 fs 自动选择。

        normalize:
            是否对 ref 和 deg 做公共峰值归一化。

        on_error:
            "nan"   ：出错时返回 np.nan；
            "raise" ：出错时直接抛出异常。

    返回：
        PESQ 分数。
    """
    try:
        from pesq import pesq
    except Exception as ex:
        if on_error == "raise":
            raise ImportError("未安装 pesq，请先运行：pip install pesq") from ex
        return float("nan")

    ref = _to_1d_float64(ref)
    deg = _to_1d_float64(deg)
    ref, deg = _match_length(ref, deg)

    fs = int(fs)

    if fs not in (8000, 16000):
        # PESQ 常用 8k 或 16k，这里默认重采样到 16k。
        target_fs = 16000
        ref = _resample_if_needed(ref, fs, target_fs)
        deg = _resample_if_needed(deg, fs, target_fs)
        fs = target_fs
        ref, deg = _match_length(ref, deg)

    if mode is None:
        mode = "wb" if fs == 16000 else "nb"

    if normalize:
        ref, deg = _safe_peak_normalize_pair(ref, deg)

    try:
        score = pesq(fs, ref, deg, mode)
        return float(score)
    except Exception as ex:
        if on_error == "raise":
            raise
        return float("nan")


def compute_stoi(
    ref,
    deg,
    fs: int,
    extended: bool = False,
    normalize: bool = True,
    on_error: str = "nan",
) -> float:
    """
    计算 STOI 或 ESTOI。

    参数：
        ref:
            参考信号。

        deg:
            待评价信号。

        fs:
            采样率。

        extended:
            False：计算 STOI；
            True ：计算 ESTOI。

        normalize:
            是否对 ref 和 deg 做公共峰值归一化。

        on_error:
            "nan"   ：出错时返回 np.nan；
            "raise" ：出错时直接抛出异常。

    返回：
        STOI / ESTOI 分数，通常范围接近 [0, 1]。
    """
    try:
        from pystoi.stoi import stoi
    except Exception as ex:
        if on_error == "raise":
            raise ImportError("未安装 pystoi，请先运行：pip install pystoi") from ex
        return float("nan")

    ref = _to_1d_float64(ref)
    deg = _to_1d_float64(deg)
    ref, deg = _match_length(ref, deg)

    fs = int(fs)

    if normalize:
        ref, deg = _safe_peak_normalize_pair(ref, deg)

    try:
        score = stoi(ref, deg, fs, extended=bool(extended))
        return float(score)
    except Exception:
        if on_error == "raise":
            raise
        return float("nan")


def compute_pesq_stoi(
    ref,
    deg,
    fs: int,
    pesq_mode: Optional[str] = None,
    extended_stoi: bool = False,
    normalize: bool = True,
    on_error: str = "nan",
) -> Dict[str, float]:
    """
    同时计算 PESQ 和 STOI。

    返回：
        {
            "PESQ": ...,
            "STOI": ...
        }
    """
    pesq_score = compute_pesq(
        ref=ref,
        deg=deg,
        fs=fs,
        mode=pesq_mode,
        normalize=normalize,
        on_error=on_error,
    )

    stoi_score = compute_stoi(
        ref=ref,
        deg=deg,
        fs=fs,
        extended=extended_stoi,
        normalize=normalize,
        on_error=on_error,
    )

    return {
        "PESQ": float(pesq_score),
        "STOI": float(stoi_score),
    }