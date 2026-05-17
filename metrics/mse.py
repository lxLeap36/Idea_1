"""
评价指标模块
论文 Eq.(78): MSE(dB) = 10 * log10( (1/N) * sum((d_k - d_hat_k)^2) )
"""

import numpy as np


def mse_db(errors: np.ndarray) -> float:
    """
    计算 MSE (dB)
    errors: shape (N,)，估计误差序列 e_k = d_k - d_hat_k
    """
    mse_linear = np.mean(errors ** 2)
    mse_linear = max(mse_linear, 1e-20)   # 防止 log(0)
    return 10.0 * np.log10(mse_linear)


def mse_db_curve(errors: np.ndarray, window: int = 1) -> np.ndarray:
    """
    逐点（或滑动窗口）MSE(dB) 曲线，用于绘制学习曲线
    errors: shape (N,)
    window: 滑动平均窗口大小（1 = 逐点瞬时误差平方）
    返回  : shape (N,) 的 MSE(dB) 数组
    """
    sq = errors ** 2
    if window > 1:
        kernel = np.ones(window) / window
        sq = np.convolve(sq, kernel, mode='same')
    sq = np.maximum(sq, 1e-20)
    return 10.0 * np.log10(sq)


def steady_state_mse_db(errors: np.ndarray, last_n: int = 1000) -> float:
    """
    稳态 MSE(dB)：取最后 last_n 个误差的均值
    """
    tail = errors[-last_n:] if len(errors) >= last_n else errors
    return mse_db(tail)
