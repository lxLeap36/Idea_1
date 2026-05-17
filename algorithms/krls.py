"""
Kernel Recursive Least Squares (KRLS) 算法
参考：Engel et al., "The kernel recursive least-squares algorithm," IEEE TSP 2004.

采用在线增长字典，数值稳定版本。
"""

import numpy as np
from utils.kernels import gaussian_kernel
from metrics.mse import mse_db_curve, steady_state_mse_db


class KRLS:
    def __init__(self, sigma: float = 1.0, reg: float = 1e-3,
                 forgetting: float = 0.999):
        self.sigma = sigma
        self.reg = reg
        self.lam = forgetting
        self._reset()

    def _reset(self):
        self.centers = []
        self.alpha = np.array([])
        self.P = None

    def reset(self):
        self._reset()

    def _kvec(self, x):
        return np.array([gaussian_kernel(x, c, self.sigma) for c in self.centers])

    def predict(self, x):
        if len(self.centers) == 0:
            return 0.0
        return float(self._kvec(x) @ self.alpha)

    def update(self, x, d):
        n = len(self.centers)
        k_self = 1.0  # gaussian_kernel(x,x) = 1

        if n == 0:
            denom = k_self + self.reg
            self.P = np.array([[1.0 / denom]])
            self.centers.append(x.copy())
            self.alpha = np.array([d / denom])
            return d

        kv = self._kvec(x)           # (n,)
        y  = float(kv @ self.alpha)
        e  = d - y

        Pk    = self.P @ kv          # (n,)
        denom = self.lam * (k_self + self.reg) + float(kv @ Pk)
        denom = max(denom, 1e-8)
        gamma = 1.0 / denom

        # 扩展 P -> (n+1)x(n+1)
        P_tl = (self.P - gamma * np.outer(Pk, Pk)) / self.lam
        P_new = np.zeros((n + 1, n + 1))
        P_new[:n, :n] = P_tl
        P_new[:n,  n] = -gamma * Pk
        P_new[ n, :n] = -gamma * Pk
        P_new[ n,  n] = gamma

        # 更新 alpha
        gain = np.append(P_tl @ kv - gamma * Pk, gamma)
        alpha_new = np.append(self.lam * self.alpha, 0.0) + e * gain

        if not np.all(np.isfinite(alpha_new)) or not np.all(np.isfinite(P_new)):
            return e   # 数值异常时跳过本步

        self.alpha = alpha_new
        self.P = P_new
        self.centers.append(x.copy())
        return e

    def run(self, X_train, d_train, X_test, d_test):
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)
        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])
        test_errors = np.array([d_test[k] - self.predict(X_test[k])
                                 for k in range(len(d_test))])
        train_errors = np.where(np.isfinite(train_errors), train_errors, 0.0)
        test_errors  = np.where(np.isfinite(test_errors),  test_errors,  0.0)
        mse_curve = mse_db_curve(train_errors, window=1)
        ss_mse    = steady_state_mse_db(test_errors)
        return train_errors, test_errors, mse_curve, ss_mse
