"""
Random Fourier Filter under Maximum Correntropy Criterion (RFFMC)
参考：Wang et al., "Random Fourier filters under maximum correntropy criterion," IEEE TCAS-I 2018.

用随机傅里叶特征近似高斯核，在最大相关熵准则下更新权重。
本实现使用 LMS 型更新以与论文对比保持结构可比性。
"""

import numpy as np
from metrics.mse import mse_db_curve


class RFFMC:
    """
    Random Fourier Filter under Maximum Correntropy

    参数
    ----
    filter_order : 输入维度 L
    d            : 随机傅里叶特征维数
    step_size    : 步长
    sigma        : 高斯核宽度（用于 RFF 近似）
    kernel_bw    : 相关熵核带宽（用于 correntropy 权重）
    seed         : 随机种子（固定 RFF 基）
    """

    def __init__(self, filter_order: int, d: int = 100,
                 step_size: float = 0.1, sigma: float = 1.0,
                 kernel_bw: float = 1.0, seed: int = 0):
        self.L = filter_order
        self.d = d
        self.eta = step_size
        self.sigma = sigma
        self.kernel_bw = kernel_bw

        # 预生成固定 RFF 投影矩阵
        rng = np.random.default_rng(seed)
        self.omega = rng.normal(0, 1.0 / sigma, size=(filter_order, d))
        self.b = rng.uniform(0, 2 * np.pi, size=d)

        self.w = np.zeros(d)

    def _map(self, x: np.ndarray) -> np.ndarray:
        """将输入 x 映射到 RFF 特征空间"""
        return np.sqrt(2.0 / self.d) * np.cos(x @ self.omega + self.b)

    def reset(self):
        self.w = np.zeros(self.d)

    def get_state(self) -> dict:
        """Return minimal state (weights w)."""
        return {
            'w': self.w.copy(),
            'omega': self.omega.copy(),
            'b': self.b.copy(),
            'd': int(self.d),
            'sigma': float(self.sigma),
        }

    def set_state(self, state: dict):
        w = state.get('w', None)
        if w is not None:
            self.w = np.atleast_1d(w).copy()
        omega = state.get('omega', None)
        b = state.get('b', None)
        if omega is not None and b is not None:
            self.omega = np.array(omega, copy=True)
            self.b = np.array(b, copy=True)
        # restore other meta if present
        if 'd' in state:
            try:
                self.d = int(state['d'])
            except Exception:
                pass
        if 'sigma' in state:
            try:
                self.sigma = float(state['sigma'])
            except Exception:
                pass

    def get_init_kwargs(self) -> dict:
        """Return kwargs to construct a compatible RFFMC instance."""
        return {'filter_order': int(self.L), 'd': int(self.d), 'step_size': float(self.eta),
                'sigma': float(self.sigma), 'kernel_bw': float(self.kernel_bw), 'seed': 0}

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ self._map(x))

    def update(self, x: np.ndarray, d: float) -> float:
        z = self._map(x)
        y = float(self.w @ z)
        e = d - y
        # 最大相关熵准则下的梯度：加入 correntropy 核权重
        corr_weight = np.exp(-e ** 2 / (2 * self.kernel_bw ** 2))
        self.w = self.w + self.eta * corr_weight * e * z
        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray = None, d_test: np.ndarray = None):
        """
        Training convenience. Does not evaluate test set. Returns training-related values.
        """
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        mse_curve = mse_db_curve(train_errors, window=1)
        return train_errors, None, mse_curve, None
