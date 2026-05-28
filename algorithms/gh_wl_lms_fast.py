"""
向量化快速版 GH-WL-LMS。

主要改进：
1. 原版 GHWLLMS 通常是逐 tap、逐阶数计算 Hermite 函数；
2. 本版本一次性对整个输入向量 x 计算 shape=(L, M) 的 Hermite 特征矩阵；
3. Python 循环只保留在 Hermite 阶数 M 上，不再对 filter_order L 做 Python 循环；
4. 对真实 AEC 长滤波器场景，例如 p=256、512，会明显更快。

模型形式：
    y_hat(k) = omega^T Z(k)

其中：
    Z(k) = vec(Phi(k))
    Phi[i, m] = psi_m(x_i / scale)

psi_m 使用物理学家 Hermite functions 的归一化递推形式。
"""

import math
import numpy as np
from metrics.mse import mse_db_curve


class GHWLLMSFast:
    """
    向量化快速版 Gauss-Hermite Weight-Learning LMS。

    参数
    ----
    filter_order : int
        滤波器长度 L。

    M : int
        每个 tap 使用的 Hermite basis 数量。

    scale : float
        输入归一化尺度，u = x / scale。

    step_size : float
        LMS / NLMS 更新步长。

    normalized : bool
        是否使用 NLMS 风格归一化更新。

    eps : float
        防止除零的小常数。

    seed : int
        保留该参数用于和原 GHWLLMS 接口兼容。
    """

    def __init__(
        self,
        filter_order: int,
        M: int = 8,
        scale: float = 1.0,
        step_size: float = 0.01,
        normalized: bool = True,
        eps: float = 1e-8,
        seed: int = 0,
    ):
        self.L = int(filter_order)
        self.M = int(M)
        self.scale = float(scale)
        self.eta = float(step_size)
        self.normalized = bool(normalized)
        self.eps = float(eps)
        self.seed = int(seed)

        if self.L <= 0:
            raise ValueError(f"filter_order 必须为正数，当前为 {self.L}")
        if self.M <= 0:
            raise ValueError(f"M 必须为正数，当前为 {self.M}")
        if self.scale <= 0:
            raise ValueError(f"scale 必须为正数，当前为 {self.scale}")

        self.n_features = self.L * self.M
        self.omega = np.zeros(self.n_features, dtype=np.float64)

        # 预计算 Hermite 递推系数，避免每个样本重复调用 sqrt。
        self._a = np.zeros(max(self.M, 2), dtype=np.float64)
        self._b = np.zeros(max(self.M, 2), dtype=np.float64)

        for n in range(1, self.M - 1):
            self._a[n] = math.sqrt(2.0 / (n + 1.0))
            self._b[n] = math.sqrt(n / (n + 1.0))

        self._pi_neg_quarter = math.pi ** (-0.25)
        self._sqrt2 = math.sqrt(2.0)

    # ============================================================
    # 状态管理接口
    # ============================================================

    def reset(self, seed: int = None, **kwargs):
        """
        重置权重。

        kwargs 用于兼容已有实验代码中可能出现的：
            reset(reseed_centers=True, seed=...)
        """
        if seed is not None:
            self.seed = int(seed)

        self.omega = np.zeros(self.n_features, dtype=np.float64)

    def get_state(self) -> dict:
        """返回当前状态，用于快照或复制模型。"""
        return {
            "omega": self.omega.copy(),
            "L": int(self.L),
            "M": int(self.M),
            "scale": float(self.scale),
            "eta": float(self.eta),
            "normalized": bool(self.normalized),
            "eps": float(self.eps),
            "seed": int(self.seed),
        }

    def set_state(self, state: dict):
        """恢复状态。"""
        if "omega" in state:
            self.omega = np.asarray(state["omega"], dtype=np.float64).copy()

        if "scale" in state:
            self.scale = float(state["scale"])
        if "eta" in state:
            self.eta = float(state["eta"])
        if "normalized" in state:
            self.normalized = bool(state["normalized"])
        if "eps" in state:
            self.eps = float(state["eps"])
        if "seed" in state:
            self.seed = int(state["seed"])

    def get_init_kwargs(self) -> dict:
        """
        返回初始化参数。

        有些实验脚本会用这个函数复制算法对象。
        """
        return {
            "filter_order": int(self.L),
            "M": int(self.M),
            "scale": float(self.scale),
            "step_size": float(self.eta),
            "normalized": bool(self.normalized),
            "eps": float(self.eps),
            "seed": int(self.seed),
        }

    # ============================================================
    # 向量化 Hermite 特征
    # ============================================================

    def _hermite_functions_matrix(self, x: np.ndarray) -> np.ndarray:
        """
        向量化计算整个输入向量的 Hermite functions。

        输入：
            x: shape=(L,)

        输出：
            Phi: shape=(L, M)

        原版慢的地方通常是：
            for i in range(L):
                for m in range(M):
                    ...
        本函数改成：
            for m in range(M):
                对所有 L 个 tap 一次性计算
        """
        x = np.asarray(x, dtype=np.float64)

        if x.shape[0] != self.L:
            raise ValueError(f"输入向量长度应为 {self.L}，当前为 {x.shape[0]}")

        u = x / self.scale

        Phi = np.empty((self.L, self.M), dtype=np.float64)

        # psi_0(u) = pi^{-1/4} exp(-u^2 / 2)
        Phi[:, 0] = self._pi_neg_quarter * np.exp(-0.5 * u * u)

        if self.M == 1:
            return Phi

        # psi_1(u) = sqrt(2) * u * psi_0(u)
        Phi[:, 1] = self._sqrt2 * u * Phi[:, 0]

        # psi_{n+1}(u)
        for n in range(1, self.M - 1):
            Phi[:, n + 1] = (
                self._a[n] * u * Phi[:, n]
                - self._b[n] * Phi[:, n - 1]
            )

        return Phi

    def _build_features(self, x: np.ndarray) -> np.ndarray:
        """
        构造一维特征向量 Z。

        Z 的顺序与原 GHWLLMS 保持一致：
            [tap0_order0, tap0_order1, ..., tap0_orderM-1,
             tap1_order0, tap1_order1, ..., tap1_orderM-1,
             ...]
        """
        Phi = self._hermite_functions_matrix(x)
        return Phi.ravel()

    # ============================================================
    # 预测与更新
    # ============================================================

    def predict(self, x: np.ndarray) -> float:
        """预测输出。"""
        Z = self._build_features(x)
        return float(self.omega @ Z)

    def update(self, x: np.ndarray, d: float) -> float:
        """
        单步在线更新。

        返回：
            e = d - y_hat
        """
        Z = self._build_features(x)

        y = float(self.omega @ Z)
        e = float(d - y)

        if self.normalized:
            denom = self.eps + float(Z @ Z)
            self.omega += (self.eta / denom) * e * Z
        else:
            self.omega += self.eta * e * Z

        return e

    def run(
        self,
        X_train: np.ndarray,
        d_train: np.ndarray,
        X_test: np.ndarray = None,
        d_test: np.ndarray = None,
    ):
        """
        批量运行接口，用于兼容已有实验脚本。

        注意：
            实验 D 主要用 update() 做流式训练；
            这里保留 run() 是为了不破坏其他调用方式。
        """
        self.reset()

        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train, dtype=np.float64)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], float(d_train[k]))

        mse_curve = mse_db_curve(train_errors, window=1)

        return train_errors, None, mse_curve, None