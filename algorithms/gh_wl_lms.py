"""
Gauss-Hermite Weight-Learning LMS (GH-WL-LMS)

This algorithm keeps the WL-LMS outer framework:

    y_hat(k) = omega^T Z_k
    omega <- omega + eta * e(k) * Z_k

but replaces random Gaussian basis functions with normalized
Gauss-Hermite functions:

    psi_n(u) = H_n(u) exp(-u^2 / 2) / sqrt(2^n n! sqrt(pi))

where u = x / scale.

Compared with the original WL-LMS:
    original WL-LMS:
        G_j(x) = exp(-(x - z_j)^2 / sigma^2), random centers z_j

    GH-WL-LMS:
        psi_j(x / scale), no random centers, orthogonal basis orders
"""

import math
import numpy as np
from metrics.mse import mse_db_curve


class GHWLLMS:
    """
    Gauss-Hermite Weight-Learning-based Least Mean Square.

    Parameters
    ----------
    filter_order : int
        Input memory length L.

    M : int
        Number of Gauss-Hermite basis functions per input dimension.
        Basis orders are 0, 1, ..., M-1.

    scale : float
        Input scaling factor for u = x / scale.
        If input is normally scaled to [-1, 1], scale=1.0 is a good start.
        If input has larger dynamic range, increase scale.

    step_size : float
        LMS step size.

    normalized : bool
        If True, use normalized LMS update:
            omega <- omega + eta * e * Z / (eps + ||Z||^2)

        This is recommended because Hermite features may have different
        energy across orders.

    eps : float
        Small constant for normalized update.
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

        # seed is kept only for interface compatibility with existing MC code.
        # GH basis itself is deterministic and does not use random centers.
        self.seed = int(seed)

        self.omega = np.zeros(self.L * self.M, dtype=float)

        # Precompute normalization constants:
        # norm_n = sqrt(2^n n! sqrt(pi))
        self._norm = np.zeros(self.M, dtype=float)
        for n in range(self.M):
            self._norm[n] = math.sqrt((2.0 ** n) * math.factorial(n) * math.sqrt(math.pi))
            # 后续代码完全没有使用 self._norm。这不会影响正确性，因为递推本身就直接产生了归一化的结果。_norm 只是闲置的内存。

    def reset(self, seed: int = None, **kwargs):
        """
        Reset adaptive coefficients.

        Extra kwargs are accepted for compatibility with run_one_trial(),
        which may call reset(reseed_centers=True, seed=...).
        """
        if seed is not None:
            self.seed = int(seed)

        self.omega = np.zeros(self.L * self.M, dtype=float)

    def get_state(self) -> dict:
        return {
            'omega': self.omega.copy(),
            'M': int(self.M),
            'scale': float(self.scale),
            'eta': float(self.eta),
            'normalized': bool(self.normalized),
            'eps': float(self.eps),
            'seed': int(self.seed),
        }

    def set_state(self, state: dict):
        self.omega = state.get('omega', np.zeros(self.L * self.M)).copy()

        if 'scale' in state:
            self.scale = float(state['scale'])
        if 'eta' in state:
            self.eta = float(state['eta'])
        if 'normalized' in state:
            self.normalized = bool(state['normalized'])
        if 'eps' in state:
            self.eps = float(state['eps'])
        if 'seed' in state:
            self.seed = int(state['seed'])

    def get_init_kwargs(self) -> dict:
        return {
            'filter_order': int(self.L),
            'M': int(self.M),
            'scale': float(self.scale),
            'step_size': float(self.eta),
            'normalized': bool(self.normalized),
            'eps': float(self.eps),
            'seed': int(self.seed),
        }

    def _hermite_functions_1d(self, x_scalar: float) -> np.ndarray:
        """
        Compute normalized Hermite functions psi_0(u), ..., psi_{M-1}(u)
        using stable three-term recurrence.

        psi_0(u) = pi^{-1/4} exp(-u^2/2)
        psi_1(u) = sqrt(2) u psi_0(u)
        psi_{n+1}(u) = sqrt(2/(n+1)) u psi_n(u)
                       - sqrt(n/(n+1)) psi_{n-1}(u)
        """
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

        u = float(x_scalar) / self.scale

        psi = np.zeros(self.M, dtype=float)

        if self.M <= 0:
            return psi

        # psi_0
        psi[0] = (math.pi ** -0.25) * math.exp(-0.5 * u * u)

        if self.M == 1:
            return psi

        # psi_1
        psi[1] = math.sqrt(2.0) * u * psi[0]

        # recurrence
        for n in range(1, self.M - 1):
            psi[n + 1] = (
                math.sqrt(2.0 / (n + 1.0)) * u * psi[n]
                - math.sqrt(n / (n + 1.0)) * psi[n - 1]
            )

        return psi

    def _build_Xk(self, x: np.ndarray) -> np.ndarray:
        """
        Build feature matrix X_k, shape (L, M)

        X_k[t, m] = psi_m(x_t / scale)
        """
        x = np.asarray(x, dtype=float)

        if x.shape[0] != self.L:
            raise ValueError(f"Expected x length {self.L}, got {x.shape[0]}")

        Xk = np.zeros((self.L, self.M), dtype=float)

        for t in range(self.L):
            Xk[t, :] = self._hermite_functions_1d(x[t])

        return Xk

    def _build_Zk(self, x: np.ndarray) -> np.ndarray:
        return self._build_Xk(x).ravel()

    def predict(self, x: np.ndarray) -> float:
        Zk = self._build_Zk(x)
        return float(self.omega @ Zk)

    def update(self, x: np.ndarray, d: float) -> float:
        Zk = self._build_Zk(x)
        y = float(self.omega @ Zk)
        e = float(d - y)

        if self.normalized:
            denom = self.eps + float(Zk @ Zk)
            self.omega = self.omega + (self.eta / denom) * e * Zk
        else:
            self.omega = self.omega + self.eta * e * Zk

        return e

    def run(
        self,
        X_train: np.ndarray,
        d_train: np.ndarray,
        X_test: np.ndarray = None,
        d_test: np.ndarray = None,
    ):
        self.reset()
        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        for k in range(n_train):
            train_errors[k] = self.update(X_train[k], d_train[k])

        mse_curve = mse_db_curve(train_errors, window=1)
        return train_errors, None, mse_curve, None