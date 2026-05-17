"""
非线性系统数据生成模块
论文 Eq.(80):
    u_k = (c1 - c2*exp(-u_{k-1}^2)) * u_{k-1}
          - u_{k-2} * (c3 + c4*exp(-u_{k-1}^2))
          + c5 * sin(u_{k-1} * pi)

用前 p 个样本预测当前值（默认 p=5）。
"""

import numpy as np
from typing import Tuple, List


def generate_nonlinear_sequence(
    n_samples: int,
    c: List[float],
    noise_var: float = 0.0,
    seed: int = 0,
    u_init: float = 0.1,
) -> np.ndarray:
    """
    生成非线性系统时间序列（干净 + 噪声）

    参数
    ----
    n_samples  : 总样本数
    c          : 系数向量 [c1, c2, c3, c4, c5]
    noise_var  : 加性高斯噪声方差（0 = 无噪声）
    seed       : 随机种子
    u_init     : 初始值 u1 = u2 = u_init

    返回
    ----
    u_clean : shape (n_samples,)  干净序列
    u_noisy : shape (n_samples,)  含噪序列
    """
    rng = np.random.default_rng(seed)
    c1, c2, c3, c4, c5 = c

    u = np.zeros(n_samples + 2)
    u[0] = u_init
    u[1] = u_init

    for k in range(2, n_samples + 2):
        uk1 = u[k - 1]
        uk2 = u[k - 2]
        exp_term = np.exp(-uk1 ** 2)
        u[k] = ((c1 - c2 * exp_term) * uk1
                - uk2 * (c3 + c4 * exp_term)
                + c5 * np.sin(uk1 * np.pi))

    u_clean = u[2:]   # shape (n_samples,)

    noise = rng.normal(0, np.sqrt(noise_var), size=n_samples) if noise_var > 0 else np.zeros(n_samples)
    u_noisy = u_clean + noise

    return u_clean, u_noisy


def build_dataset(
    u_clean: np.ndarray,
    u_noisy: np.ndarray,
    p: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    将时间序列转换为监督学习格式
    用前 p 个含噪样本作为输入，预测当前干净值

    返回
    ----
    X_noisy : shape (N-p, p)   含噪输入矩阵
    X_clean : shape (N-p, p)   干净输入矩阵（仅供参考）
    d_noisy : shape (N-p,)     含噪目标（训练用）
    d_clean : shape (N-p,)     干净目标（测试用）
    """
    N = len(u_clean)
    rows = N - p
    X_noisy = np.zeros((rows, p))
    X_clean = np.zeros((rows, p))
    for i in range(rows):
        X_noisy[i] = u_noisy[i:i + p]
        X_clean[i] = u_clean[i:i + p]
    d_noisy = u_noisy[p:]
    d_clean = u_clean[p:]
    return X_noisy, X_clean, d_noisy, d_clean


def normalize_01(X: np.ndarray, x_min=None, x_max=None):
    """
    将数据归一化到 [0, 1]（论文中平稳场景需要归一化）
    返回归一化后的数据以及 (x_min, x_max)，便于测试集复用
    """
    if x_min is None:
        x_min = X.min()
    if x_max is None:
        x_max = X.max()
    denom = x_max - x_min
    denom = denom if denom != 0 else 1.0
    return (X - x_min) / denom, x_min, x_max


# -----------------------------------------------------------------------
# 高层接口：直接生成实验三所需的所有数据集
# -----------------------------------------------------------------------

def get_stationary_dataset(
    noise_var: float,
    n_train: int = 2000,
    n_test: int = 200,
    p: int = 5,
    c: List[float] = None,
    seed: int = 42,
):
    """
    平稳非线性系统辨识数据集（论文实验三前半部分）
    训练集含噪，测试集干净，并归一化到 [0,1]

    返回
    ----
    X_train, d_train : 含噪训练输入 / 目标
    X_test,  d_test  : 干净测试输入 / 目标
    norm_params      : (x_min, x_max, d_min, d_max)，供后续反归一化
    """
    if c is None:
        c = [0.8, 0.5, 0.3, 0.9, 0.1]

    total = n_train + n_test + p + 2
    u_clean, u_noisy = generate_nonlinear_sequence(total, c, noise_var, seed)

    # 构建数据集
    X_noisy, X_clean, d_noisy, d_clean = build_dataset(u_clean, u_noisy, p)

    # 归一化（用训练集统计量）
    X_tr_raw = X_noisy[:n_train]
    d_tr_raw = d_noisy[:n_train]
    X_te_raw = X_clean[n_train:n_train + n_test]
    d_te_raw = d_clean[n_train:n_train + n_test]

    X_tr_norm, x_min, x_max = normalize_01(X_tr_raw)
    d_tr_norm, d_min, d_max = normalize_01(d_tr_raw)
    X_te_norm, _, _ = normalize_01(X_te_raw, x_min, x_max)
    d_te_norm, _, _ = normalize_01(d_te_raw, d_min, d_max)

    return X_tr_norm, d_tr_norm, X_te_norm, d_te_norm, (x_min, x_max, d_min, d_max)


def get_nonstationary_dataset(
    noise_var: float,
    n_train: int = 4000,
    n_test: int = 200,
    p: int = 5,
    c1: List[float] = None,
    c2_coef: List[float] = None,
    change_point: int = 2001,
    seed: int = 42,
):
    """
    非平稳非线性系统辨识数据集（论文实验三后半部分）
    k=change_point 处系数向量突变，不做归一化

    返回
    ----
    X_train, d_train : 含噪训练输入 / 目标（拼接两段）
    X_test,  d_test  : 干净测试输入 / 目标
    """
    if c1 is None:
        c1 = [0.8, 0.5, 0.3, 0.9, 0.1]
    if c2_coef is None:
        c2_coef = [0.4, 0.7, 0.6, 0.6, 0.2]

    seg1_len = change_point - 1            # ~2000 个训练样本（含噪）
    seg2_len = n_train - seg1_len + n_test + p + 2

    u_clean1, u_noisy1 = generate_nonlinear_sequence(seg1_len + p + 2, c1, noise_var, seed)
    u_clean2, u_noisy2 = generate_nonlinear_sequence(seg2_len, c2_coef, noise_var, seed + 1)

    u_clean_all = np.concatenate([u_clean1, u_clean2])
    u_noisy_all = np.concatenate([u_noisy1, u_noisy2])

    X_noisy, X_clean, d_noisy, d_clean = build_dataset(u_clean_all, u_noisy_all, p)

    X_train = X_noisy[:n_train]
    d_train = d_noisy[:n_train]
    X_test  = X_clean[n_train:n_train + n_test]
    d_test  = d_clean[n_train:n_train + n_test]

    return X_train, d_train, X_test, d_test
