"""
Weight-Learning-based LMS (WL-LMS) 算法
论文核心贡献之一，基于 Kolmogorov-Arnold 表示定理。

核心思想：将滤波器权重视为可学习函数（高斯基函数的线性组合），
而非线性组合中的标量系数。

论文 Eq.(11-13, 28):
    F(x_k) = x_k^T ∘ Ω_k  （每维度独立的可学习函数）
    F_t(·) = Σ_{j=1}^{M} ϖ_{t,j} G_{t,j}(·)
    G_{t,j}(x) = exp(-(x - z_{t,j})² / σ²)
    ω_k = ω_{k-1} + η * e_k * Z_k   （SGD 更新）
"""

import numpy as np
from metrics.mse import mse_db_curve, steady_state_mse_db


class WLLMS:
    """
    Weight-Learning-based Least Mean Square

    参数
    ----
    filter_order : 滤波器阶数 L（输入维度）
    M            : 每个输入维度的高斯基函数数量（论文要求 M ≥ 2L+1）
    sigma        : 高斯核宽度 σ
    step_size    : 步长 η
    seed         : 随机种子（用于生成固定中心 z）
    """

    def __init__(self, filter_order: int, M: int = 40,
                 sigma: float = 1.0, step_size: float = 0.01,
                 seed: int = 0):
        self.L = filter_order
        self.M = M
        self.sigma = sigma
        self.eta = step_size

        # 从标准正态分布采样固定中心，训练过程中不更新
        rng = np.random.default_rng(seed)
        # centers[t, j] 是第 t 维输入对应的第 j 个基函数中心
        self.centers = rng.standard_normal(size=(filter_order, M))  # shape (L, M)

        # 等价权重向量 ω = vec(Υ)，shape (L*M,)
        self.omega = np.zeros(filter_order * M)

    def reset(self):
        self.omega = np.zeros(self.L * self.M)

    def _build_Xk(self, x: np.ndarray) -> np.ndarray:
        """
        构建非线性映射输入矩阵 X_k，shape (L, M)
        X_k[t, j] = G_{t,j}(x_t) = exp(-(x_t - z_{t,j})² / σ²)
        """
        # x: shape (L,)
        # centers: shape (L, M)
        diff = x[:, np.newaxis] - self.centers          # (L, M)
        Xk = np.exp(-(diff ** 2) / (self.sigma ** 2))  # (L, M)
        return Xk

    def _build_Zk(self, x: np.ndarray) -> np.ndarray:
        """
        Z_k = vec(X_k)，shape (L*M,)
        """
        return self._build_Xk(x).ravel()

    def predict(self, x: np.ndarray) -> float:
        """
        F(x_k) = ω^T Z_k
        """
        Zk = self._build_Zk(x)
        return float(self.omega @ Zk)

    def update(self, x: np.ndarray, d: float) -> float:
        """
        单步 SGD 更新（论文 Eq.28）
        ω_k = ω_{k-1} + η * e_k * Z_k
        """
        Zk = self._build_Zk(x)
        y = float(self.omega @ Zk)
        e = d - y
        self.omega = self.omega + self.eta * e * Zk
        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray, d_test: np.ndarray):
        """
        全量运行

        返回
        ----
        train_errors : shape (n_train,)  训练误差序列
        test_errors  : shape (n_test,)   测试误差序列
        mse_curve    : shape (n_train,)  MSE(dB) 学习曲线
        ss_mse_db    : float             稳态 MSE(dB)
        """
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        test_errors = np.array([d_test[k] - self.predict(X_test[k])
                                 for k in range(len(d_test))])

        mse_curve = mse_db_curve(train_errors, window=1)
        ss_mse = steady_state_mse_db(test_errors)
        return train_errors, test_errors, mse_curve, ss_mse
