"""
标准 LMS 自适应滤波算法（线性框架）
"""

import numpy as np
from metrics.mse import mse_db_curve, steady_state_mse_db


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

    def predict(self, x: np.ndarray) -> float:
        return float(self.w @ x)

    def update(self, x: np.ndarray, d: float) -> float:
        """单步更新，返回估计误差"""
        y = self.predict(x)
        e = d - y
        self.w = self.w + self.eta * e * x
        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray, d_test: np.ndarray):
        """
        全量运行：先在训练集上在线学习，再在测试集上评估

        返回
        ----
        train_errors : shape (n_train,)
        test_errors  : shape (n_test,)
        mse_curve    : shape (n_train,) MSE(dB) 学习曲线
        ss_mse_db    : 稳态 MSE(dB)（基于测试集）
        """
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        # 测试
        test_errors = np.array([d_test[k] - self.predict(X_test[k])
                                 for k in range(len(d_test))])

        mse_curve = mse_db_curve(train_errors, window=1)
        ss_mse = steady_state_mse_db(test_errors)
        return train_errors, test_errors, mse_curve, ss_mse
