"""
Kernel LMS (KLMS) 算法
参考：Liu et al., "The kernel least-mean-square algorithm," IEEE TSP 2008.

采用在线字典增长策略（每步新增一个核单元），不做稀疏化。
"""

import numpy as np
from utils.kernels import gaussian_kernel
from metrics.mse import mse_db_curve, steady_state_mse_db


class KLMS:
    """
    Kernel Least Mean Square

    参数
    ----
    step_size  : 步长 μ
    sigma      : 高斯核宽度
    """

    def __init__(self, step_size: float = 0.1, sigma: float = 1.0):
        self.mu = step_size
        self.sigma = sigma
        self._reset()

    def _reset(self):
        self.centers = []   # 存储输入向量（核中心）
        self.alphas = []    # 对应系数

    def reset(self):
        self._reset()

    def predict(self, x: np.ndarray) -> float:
        if len(self.centers) == 0:
            return 0.0
        k_vec = np.array([gaussian_kernel(x, c, self.sigma) for c in self.centers])
        return float(np.dot(self.alphas, k_vec))

    def update(self, x: np.ndarray, d: float) -> float:
        y = self.predict(x)
        e = d - y
        self.centers.append(x.copy())
        self.alphas.append(self.mu * e)
        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray, d_test: np.ndarray):
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
