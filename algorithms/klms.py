"""
Kernel LMS (KLMS) 算法
参考：Liu et al., "The kernel least-mean-square algorithm," IEEE TSP 2008.

采用在线字典增长策略（每步新增一个核单元），不做稀疏化。
"""

import numpy as np
from utils.kernels import gaussian_kernel
from metrics.mse import mse_db_curve


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

    def get_state(self) -> dict:
        """Return minimal snapshot state for prediction: centers and alphas."""
        return {
            'centers': np.array(self.centers) if len(self.centers) > 0 else np.zeros((0,)),
            'alphas': np.array(self.alphas) if len(self.alphas) > 0 else np.zeros((0,)),
            'sigma': float(self.sigma),
            'mu': float(self.mu),
        }

    def get_init_kwargs(self) -> dict:
        """Return kwargs suitable to construct a fresh KLMS instance with same hyperparams."""
        return {'step_size': float(self.mu), 'sigma': float(self.sigma)}

    def set_state(self, state: dict):
        """Restore centers and alphas from state dict."""
        centers = state.get('centers', None)
        alphas = state.get('alphas', None)
        if centers is None or alphas is None:
            return
        # store as lists to be consistent with update() expectations
        self.centers = [c.copy() for c in np.atleast_2d(centers)]
        self.alphas = list(np.atleast_1d(alphas).astype(float))
        # restore hyperparameters if present
        if 'sigma' in state:
            try:
                self.sigma = float(state['sigma'])
            except Exception:
                pass
        if 'mu' in state:
            try:
                self.mu = float(state['mu'])
            except Exception:
                pass

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray = None, d_test: np.ndarray = None):
        """
        Training convenience: perform online updates over X_train.
        Does NOT evaluate test set (snapshot evaluation is done in scenario layer).

        Returns (train_errors, None, mse_curve, None) for backward compatibility.
        """
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        mse_curve = mse_db_curve(train_errors, window=1)
        return train_errors, None, mse_curve, None
