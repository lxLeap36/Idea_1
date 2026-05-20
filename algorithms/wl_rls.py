"""
Weight-Learning-based RLS (WL-RLS) 算法
论文核心贡献之一，基于 Kolmogorov-Arnold 表示定理。

核心思想：在 WLNF 框架下，最小化加权过去误差平方和。

论文 Eq.(29-37):
    min_ω Σ_{i=0}^{k} λ^{k-i} (d_i - ω^T Z_i)² + λ^k ς/2 ||ω||²
    P_k = (λP_{k-1}^{-1} + Z_k Z_k^T)^{-1}   （矩阵求逆引理递推）
    ω_k = ω_{k-1} + e_k P_k Z_k
"""

import numpy as np
from metrics.mse import mse_db_curve


class WLRLS:
    """
    Weight-Learning-based Recursive Least Squares

    参数
    ----
    filter_order : 滤波器阶数 L（输入维度）
    M            : 每个输入维度的高斯基函数数量
    sigma        : 高斯核宽度 σ
    reg          : 正则化因子 ς（初始化 P_0 = I/ς）
    forgetting   : 遗忘因子 λ ∈ (0, 1]
    seed         : 随机种子（固定基函数中心）
    """

    def __init__(self, filter_order: int, M: int = 20,
                 sigma: float = 1.0, reg: float = 1e-3,
                 forgetting: float = 0.999, seed: int = 0):
        self.L = filter_order
        self.M = M
        self.sigma = sigma
        self.reg = reg
        self.lam = forgetting
        self.seed = seed

        # 固定高斯基函数中心，shape (L, M)
        rng = np.random.default_rng(seed)
        self.centers = rng.standard_normal(size=(filter_order, M))

        dim = filter_order * M
        self.omega = np.zeros(dim)
        self.P = np.eye(dim) / reg      # 初始逆相关矩阵

    def reset(self, reseed_centers: bool = False, seed: int = None):
        """
        Reset WL-RLS state.

        Parameters
        ----------
        reseed_centers : bool
            If True, regenerate Gaussian centers according to self.seed.
        seed : int or None
            If not None, update self.seed before regenerating centers.
        """
        if seed is not None:
            self.seed = int(seed)

        if reseed_centers:
            rng = np.random.default_rng(self.seed)
            self.centers = rng.standard_normal(size=(self.L, self.M))

        dim = self.L * self.M
        self.omega = np.zeros(dim)
        self.P = np.eye(dim) / self.reg

    def get_state(self) -> dict:
        return {
            'omega': self.omega.copy(),
            'P': self.P.copy(),
            'centers': self.centers.copy(),
            'sigma': float(self.sigma),
            'M': int(self.M),
            'seed': int(self.seed),
            'reg': float(self.reg),
            'forgetting': float(self.lam),
        }

    def set_state(self, state: dict):
        self.omega = state.get('omega', np.zeros(self.L * self.M)).copy()

        P = state.get('P', None)
        self.P = np.array(P, copy=True) if P is not None else np.eye(self.L * self.M) / self.reg

        centers = state.get('centers', None)
        if centers is not None:
            self.centers = np.array(centers, copy=True)

        if 'seed' in state:
            self.seed = int(state['seed'])

    def get_init_kwargs(self) -> dict:
        """Return kwargs to construct a WL-RLS instance with the same hyperparameters."""
        return {
            'filter_order': int(self.L),
            'M': int(self.M),
            'sigma': float(self.sigma),
            'reg': float(self.reg),
            'forgetting': float(self.lam),
            'seed': int(getattr(self, 'seed', 0)),
        }

    def _build_Zk(self, x: np.ndarray) -> np.ndarray:
        """
        Z_k = vec(X_k)，shape (L*M,)
        X_k[t, j] = exp(-(x_t - z_{t,j})² / σ²)
        """
        diff = x[:, np.newaxis] - self.centers      # (L, M)
        Xk = np.exp(-(diff ** 2) / (self.sigma ** 2))
        return Xk.ravel()

    def predict(self, x: np.ndarray) -> float:
        Zk = self._build_Zk(x)
        return float(self.omega @ Zk)

    def update(self, x: np.ndarray, d: float) -> float:
        """
        单步 RLS 更新（论文 Eq.36-37）
        P_k = λ^{-1} P_{k-1} - λ^{-1} P_{k-1} Z_k Z_k^T P_{k-1} / (λ + Z_k^T P_{k-1} Z_k)
        ω_k = ω_{k-1} + e_k P_k Z_k
        """
        Zk = self._build_Zk(x)
        y = float(self.omega @ Zk)
        e = d - y

        # 计算增益向量（Sherman-Morrison 公式，论文 Eq.36）
        Pz = self.P @ Zk                            # shape (L*M,)
        denom = self.lam + float(Zk @ Pz)
        # 更新逆相关矩阵
        self.P = (self.P - np.outer(Pz, Pz) / denom) / self.lam
        # 增益向量 k_k = P_k Z_k
        gain = self.P @ Zk
        # 更新权重向量
        self.omega = self.omega + e * gain
        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray = None, d_test: np.ndarray = None):
        """Training convenience: perform updates over X_train only."""
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        mse_curve = mse_db_curve(train_errors, window=1)
        return train_errors, None, mse_curve, None
