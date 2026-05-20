"""
标准 LMS 自适应滤波算法（线性框架）
"""

import numpy as np
from metrics.mse import mse_db_curve


class LMS:
    """
    Least Mean Square 算法

    参数
    ----
    filter_order : 滤波器阶数 L（= 输入维度）
    step_size    : 步长 η
    """

    def __init__(self, filter_order: int, step_size: float = 0.01):
        self.L = filter_order
        self.eta = step_size
        self.w = np.zeros(filter_order)

    def reset(self):
        self.w = np.zeros(self.L)

    def get_state(self) -> dict:
        """Return minimal state dict for snapshotting (weights)."""
        return {'w': self.w.copy(), 'L': int(self.L), 'eta': float(self.eta)}

    def set_state(self, state: dict):
        """Restore state from dict produced by get_state()."""
        self.w = state.get('w', np.zeros(self.L)).copy()

    def get_init_kwargs(self) -> dict:
        """Return kwargs suitable to construct a fresh LMS instance with same hyperparameters."""
        return {'filter_order': int(self.L), 'step_size': float(self.eta)}

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ x)

    def update(self, x: np.ndarray, d: float) -> float:
        """单步更新，返回估计误差"""
        y = self.predict(x)
        e = d - y
        self.w = self.w + self.eta * e * x
        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray = None, d_test: np.ndarray = None):
        """
        训练便利方法（仅在训练集上逐步更新）。

        为保持向后兼容，返回值仍保留 4 元组格式，但 test 相关的返回值为 None。

        返回
        ----
        train_errors : shape (n_train,)
        test_errors  : None
        mse_curve    : shape (n_train,) MSE(dB)（基于训练误差）
        ss_mse_db    : None
        """
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        # 返回训练误差相关量；不在此处对测试集做评估（场景层负责快照/评估）
        mse_curve = mse_db_curve(train_errors, window=1)
        return train_errors, None, mse_curve, None
