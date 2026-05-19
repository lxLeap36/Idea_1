"""
Nyström Kernel Recursive Generalized Maximum Correntropy (NKRGMC)
参考：Zhang & Wang, "Nyström kernel algorithm under generalized maximum correntropy criterion,"
      IEEE Signal Process. Lett. 2020.

用 Nyström 特征近似高斯核，在广义最大相关熵准则下用 RLS 型更新。
"""

import numpy as np
from metrics.mse import mse_db_curve
from utils.kernels import gaussian_kernel


class NKRGMC:
    """
    Nyström KRGMC

    参数
    ----
    filter_order : 输入维度 L
    d            : Nyström 锚点数（特征维数）
    sigma        : 高斯核宽度
    reg          : 正则化因子 ς
    forgetting   : 遗忘因子 λ
    kernel_bw    : 广义相关熵核带宽
    alpha_order  : 广义相关熵误差阶数 α
    seed         : 随机种子（用于锚点初始化）
    """

    def __init__(self, filter_order: int, d: int = 100,
                 sigma: float = 1.0, reg: float = 1e-3,
                 forgetting: float = 0.999, kernel_bw: float = 1.0,
                 alpha_order: float = 2.0, seed: int = 0):
        self.L = filter_order
        self.d = d
        self.sigma = sigma
        self.reg = reg
        self.lam = forgetting
        self.kernel_bw = kernel_bw
        self.alpha_order = alpha_order
        self.seed = seed

        self.anchors = None     # Nyström 锚点，训练开始时从数据采样
        self.w = np.zeros(d)
        self.P = np.eye(d) / reg

    def _nystrom_map(self, x: np.ndarray) -> np.ndarray:
        """将 x 映射到 Nyström 特征"""
        k_vec = np.array([gaussian_kernel(x, a, self.sigma) for a in self.anchors])
        return k_vec  # 未做归一化（与论文一致）

    def reset(self):
        self.anchors = None
        self.w = np.zeros(self.d)
        self.P = np.eye(self.d) / self.reg

    def get_state(self) -> dict:
        return {
            'anchors': np.array(self.anchors) if self.anchors is not None else None,
            'w': self.w.copy(),
            'P': self.P.copy(),
        }

    def set_state(self, state: dict):
        anchors = state.get('anchors', None)
        if anchors is not None:
            self.anchors = np.atleast_2d(anchors).copy()
        self.w = state.get('w', np.zeros(self.d)).copy()
        P = state.get('P', None)
        self.P = np.array(P, copy=True) if P is not None else np.eye(self.d) / self.reg

    def get_init_kwargs(self) -> dict:
        return {'filter_order': int(self.L), 'd': int(self.d), 'sigma': float(self.sigma),
                'reg': float(self.reg), 'forgetting': float(self.lam), 'kernel_bw': float(self.kernel_bw),
                'alpha_order': float(self.alpha_order), 'seed': int(self.seed)}

    def init_anchors(self, X_train: np.ndarray):
        """从训练集随机选取 d 个锚点"""
        rng = np.random.default_rng(self.seed)
        chosen = min(self.d, len(X_train))
        idx = rng.choice(len(X_train), size=chosen, replace=False)
        self.anchors = X_train[idx]

        # If we selected fewer anchors than configured self.d (e.g., small train set),
        # adjust internal dimensions (w, P) so that mapping size matches anchors count.
        actual_d = self.anchors.shape[0]
        if actual_d != self.d:
            self.d = actual_d
            # resize weight vector and P accordingly
            self.w = np.zeros(self.d)
            self.P = np.eye(self.d) / self.reg

    def predict(self, x: np.ndarray) -> float:
        z = self._nystrom_map(x)
        return float(self.w @ z)

    def update(self, x: np.ndarray, d: float) -> float:
        z = self._nystrom_map(x)
        y = float(self.w @ z)
        e = d - y

        # 广义相关熵权重
        corr_weight = np.exp(-np.abs(e) ** self.alpha_order / self.kernel_bw)

        # RLS 型更新（加权）
        Pz = self.P @ z
        denom = self.lam / corr_weight + z @ Pz
        gain = Pz / denom
        self.P = (self.P - np.outer(gain, z) @ self.P) / self.lam
        self.w = self.w + gain * e

        return e

    def run(self, X_train: np.ndarray, d_train: np.ndarray,
            X_test: np.ndarray = None, d_test: np.ndarray = None):
        """
        Training convenience. Does not evaluate test set; returns training results.
        """
        self.reset()
        self.init_anchors(X_train)

        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        mse_curve = mse_db_curve(train_errors, window=1)
        return train_errors, None, mse_curve, None
