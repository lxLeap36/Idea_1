"""
Sparse 2D Gauss-Hermite Weight-Learning LMS (GH2D-WL-LMS)

This algorithm extends GH-WL-LMS by adding sparse 2D cross terms:

    y_hat(k) = y_1d(k) + y_2d(k)

where

    y_1d(k) = omega_1d^T Z_1d(k)

and

    y_2d(k) = omega_2d^T Z_2d(k)

with

    Z_2d(k) = {
        psi_m(x_i / scale) * psi_n(x_j / scale)
        for (i, j) in cross_pairs
        for (m, n) in cross_orders
    }

Purpose
-------
The 2D terms are designed to explicitly model cross-tap nonlinear
interactions such as:

    x(k) * x(k-1)
    x(k)^2 * x(k-1)

while avoiding the full O(L^2 M^2) Volterra expansion.

Default design
--------------
For the current IMD target:

    d(k) = c2 * x(k) * x(k-1) + c3 * x(k)^2 * x(k-1)

the default sparse structure is:

    cross_pairs  = [(0, 1)]
    cross_orders = [(1, 1), (2, 1)]

This is intentionally small and physically matched.
"""

import math
import numpy as np
from metrics.mse import mse_db_curve


class GH2DWLLMS:
    """
    Sparse 2D Gauss-Hermite Weight-Learning LMS.

    Parameters
    ----------
    filter_order : int
        Input memory length L.

    M : int
        Number of 1D Gauss-Hermite basis functions per tap.
        Basis orders are 0, 1, ..., M-1.

    scale : float
        Input scaling factor for u = x / scale.

    step_size : float
        Default LMS step size. Used for both 1D and 2D branches if
        step_size_1d / step_size_2d are not provided.

    step_size_1d : float or None
        Step size for the 1D GH branch.

    step_size_2d : float or None
        Step size for the sparse 2D GH cross branch.
        Usually should be smaller than step_size_1d.

    normalized : bool
        If True, use NLMS-style normalization separately for 1D and 2D.

    eps : float
        Small constant for normalized update.

    include_1d : bool
        If True, include original 1D GH-WL-LMS branch.

    cross_pairs : list[tuple[int, int]] or None
        Sparse lag pairs. Example:
            [(0, 1)]
        means using x(k) and x(k-1).

    cross_orders : list[tuple[int, int]] or None
        Sparse GH order pairs. Example:
            [(1, 1), (2, 1)]
        means:
            psi_1(x_i) psi_1(x_j)
            psi_2(x_i) psi_1(x_j)

        Avoid including (0, 0), (1, 0), (0, 1) by default because they
        are usually redundant with constant / 1D terms.

    leakage_1d : float
        Optional leakage for 1D coefficients.
        Update uses:
            omega_1d <- (1 - leakage_1d) * omega_1d + update

    leakage_2d : float
        Optional leakage for 2D coefficients.
        This is useful to prevent 2D cross weights from drifting after bursts.

    seed : int
        Kept for interface compatibility. The GH basis itself is deterministic.
    """

    def __init__(
        self,
        filter_order: int,
        M: int = 8,
        scale: float = 1.0,
        step_size: float = 0.01,
        step_size_1d=None,
        step_size_2d=None,
        normalized: bool = True,
        eps: float = 1e-8,
        include_1d: bool = True,
        cross_pairs=None,
        cross_orders=None,
        leakage_1d: float = 0.0,
        leakage_2d: float = 0.0,
        seed: int = 0,
    ):
        self.L = int(filter_order)
        self.M = int(M)
        self.scale = float(scale)

        self.eta = float(step_size)
        self.eta_1d = float(step_size if step_size_1d is None else step_size_1d)
        self.eta_2d = float(step_size if step_size_2d is None else step_size_2d)

        self.normalized = bool(normalized)
        self.eps = float(eps)
        self.include_1d = bool(include_1d)

        self.leakage_1d = float(leakage_1d)
        self.leakage_2d = float(leakage_2d)
        self.seed = int(seed)

        if self.L <= 0:
            raise ValueError(f"filter_order must be positive, got {self.L}")
        if self.M <= 0:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not (0.0 <= self.leakage_1d < 1.0):
            raise ValueError(f"leakage_1d must be in [0, 1), got {self.leakage_1d}")
        if not (0.0 <= self.leakage_2d < 1.0):
            raise ValueError(f"leakage_2d must be in [0, 1), got {self.leakage_2d}")

        # Default pair is matched to current IMD target:
        # x(k) with x(k-1).
        if cross_pairs is None:
            cross_pairs = [(0, 1)]

        # Default orders are matched to:
        # x(k) * x(k-1) and x(k)^2 * x(k-1)-like interaction.
        if cross_orders is None:
            cross_orders = [(1, 1), (2, 1)]

        self.cross_pairs = self._validate_cross_pairs(cross_pairs)
        self.cross_orders = self._validate_cross_orders(cross_orders)

        self.n_1d = self.L * self.M if self.include_1d else 0
        self.n_2d = len(self.cross_pairs) * len(self.cross_orders)

        self.omega_1d = np.zeros(self.n_1d, dtype=float)
        self.omega_2d = np.zeros(self.n_2d, dtype=float)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_cross_pairs(self, pairs):
        out = []
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError(f"Each cross pair must have length 2, got {pair}")

            i, j = int(pair[0]), int(pair[1])

            if i < 0 or i >= self.L or j < 0 or j >= self.L:
                raise ValueError(
                    f"Invalid cross pair {(i, j)} for filter length L={self.L}"
                )

            if i == j:
                raise ValueError(
                    f"Invalid cross pair {(i, j)}: i == j. "
                    "Use 1D GH terms for same-tap nonlinearity."
                )

            out.append((i, j))

        if len(out) == 0:
            raise ValueError("cross_pairs must contain at least one pair")

        return out

    def _validate_cross_orders(self, orders):
        out = []
        for order in orders:
            if len(order) != 2:
                raise ValueError(f"Each cross order must have length 2, got {order}")

            m, n = int(order[0]), int(order[1])

            if m < 0 or m >= self.M or n < 0 or n >= self.M:
                raise ValueError(
                    f"Invalid cross order {(m, n)} for M={self.M}. "
                    f"Valid orders are 0 ... {self.M - 1}"
                )

            out.append((m, n))

        if len(out) == 0:
            raise ValueError("cross_orders must contain at least one order pair")

        return out

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def reset(self, seed: int = None, **kwargs):
        """
        Reset adaptive coefficients.

        Extra kwargs are accepted for compatibility with run_one_trial(),
        which may call reset(reseed_centers=True, seed=...).
        """
        if seed is not None:
            self.seed = int(seed)

        self.omega_1d = np.zeros(self.n_1d, dtype=float)
        self.omega_2d = np.zeros(self.n_2d, dtype=float)

    def get_state(self) -> dict:
        return {
            "omega_1d": self.omega_1d.copy(),
            "omega_2d": self.omega_2d.copy(),
            "M": int(self.M),
            "scale": float(self.scale),
            "eta": float(self.eta),
            "eta_1d": float(self.eta_1d),
            "eta_2d": float(self.eta_2d),
            "normalized": bool(self.normalized),
            "eps": float(self.eps),
            "include_1d": bool(self.include_1d),
            "cross_pairs": list(self.cross_pairs),
            "cross_orders": list(self.cross_orders),
            "leakage_1d": float(self.leakage_1d),
            "leakage_2d": float(self.leakage_2d),
            "seed": int(self.seed),
        }

    def set_state(self, state: dict):
        if "omega_1d" in state:
            self.omega_1d = np.asarray(state["omega_1d"], dtype=float).copy()
        if "omega_2d" in state:
            self.omega_2d = np.asarray(state["omega_2d"], dtype=float).copy()

        if "scale" in state:
            self.scale = float(state["scale"])
        if "eta" in state:
            self.eta = float(state["eta"])
        if "eta_1d" in state:
            self.eta_1d = float(state["eta_1d"])
        if "eta_2d" in state:
            self.eta_2d = float(state["eta_2d"])
        if "normalized" in state:
            self.normalized = bool(state["normalized"])
        if "eps" in state:
            self.eps = float(state["eps"])
        if "include_1d" in state:
            self.include_1d = bool(state["include_1d"])
        if "leakage_1d" in state:
            self.leakage_1d = float(state["leakage_1d"])
        if "leakage_2d" in state:
            self.leakage_2d = float(state["leakage_2d"])
        if "seed" in state:
            self.seed = int(state["seed"])

    def get_init_kwargs(self) -> dict:
        return {
            "filter_order": int(self.L),
            "M": int(self.M),
            "scale": float(self.scale),
            "step_size": float(self.eta),
            "step_size_1d": float(self.eta_1d),
            "step_size_2d": float(self.eta_2d),
            "normalized": bool(self.normalized),
            "eps": float(self.eps),
            "include_1d": bool(self.include_1d),
            "cross_pairs": list(self.cross_pairs),
            "cross_orders": list(self.cross_orders),
            "leakage_1d": float(self.leakage_1d),
            "leakage_2d": float(self.leakage_2d),
            "seed": int(self.seed),
        }

    # ------------------------------------------------------------------
    # Hermite basis
    # ------------------------------------------------------------------
    def _hermite_functions_1d(self, x_scalar: float) -> np.ndarray:
        """
        Compute normalized physicists' Hermite functions:
            psi_0(u), ..., psi_{M-1}(u)

        using the same recurrence style as algorithms/gh_wl_lms.py.

        psi_0(u) = pi^{-1/4} exp(-u^2/2)
        psi_1(u) = sqrt(2) u psi_0(u)
        psi_{n+1}(u) = sqrt(2/(n+1)) u psi_n(u)
                       - sqrt(n/(n+1)) psi_{n-1}(u)
        """
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")

        u = float(x_scalar) / self.scale

        psi = np.zeros(self.M, dtype=float)

        psi[0] = (math.pi ** -0.25) * math.exp(-0.5 * u * u)

        if self.M == 1:
            return psi

        psi[1] = math.sqrt(2.0) * u * psi[0]

        for n in range(1, self.M - 1):
            psi[n + 1] = (
                math.sqrt(2.0 / (n + 1.0)) * u * psi[n]
                - math.sqrt(n / (n + 1.0)) * psi[n - 1]
            )

        return psi

    def _build_Phi(self, x: np.ndarray) -> np.ndarray:
        """
        Build Hermite feature matrix Phi, shape (L, M).

            Phi[i, m] = psi_m(x_i / scale)
        """
        x = np.asarray(x, dtype=float)

        if x.shape[0] != self.L:
            raise ValueError(f"Expected x length {self.L}, got {x.shape[0]}")

        Phi = np.zeros((self.L, self.M), dtype=float)

        for i in range(self.L):
            Phi[i, :] = self._hermite_functions_1d(x[i])

        return Phi

    def _build_Z1d_from_Phi(self, Phi: np.ndarray) -> np.ndarray:
        if not self.include_1d:
            return np.zeros(0, dtype=float)
        return Phi.ravel()

    def _build_Z2d_from_Phi(self, Phi: np.ndarray) -> np.ndarray:
        """
        Build sparse 2D product features.

        Feature order:
            for pair in cross_pairs:
                for order in cross_orders:
                    append Phi[i, m] * Phi[j, n]
        """
        Z2 = np.zeros(self.n_2d, dtype=float)

        idx = 0
        for i, j in self.cross_pairs:
            for m, n in self.cross_orders:
                Z2[idx] = Phi[i, m] * Phi[j, n]
                idx += 1

        return Z2

    def _build_features(self, x: np.ndarray):
        Phi = self._build_Phi(x)
        Z1 = self._build_Z1d_from_Phi(Phi)
        Z2 = self._build_Z2d_from_Phi(Phi)
        return Z1, Z2

    # ------------------------------------------------------------------
    # Prediction and adaptation
    # ------------------------------------------------------------------
    def predict(self, x: np.ndarray) -> float:
        Z1, Z2 = self._build_features(x)

        y = 0.0

        if self.include_1d:
            y += float(self.omega_1d @ Z1)

        y += float(self.omega_2d @ Z2)

        return float(y)

    def update(self, x: np.ndarray, d: float) -> float:
        Z1, Z2 = self._build_features(x)

        y = 0.0
        if self.include_1d:
            y += float(self.omega_1d @ Z1)
        y += float(self.omega_2d @ Z2)

        e = float(d - y)

        # 1D branch update
        if self.include_1d and self.n_1d > 0:
            if self.leakage_1d > 0:
                self.omega_1d *= (1.0 - self.leakage_1d)

            if self.normalized:
                denom_1d = self.eps + float(Z1 @ Z1)
                self.omega_1d += (self.eta_1d / denom_1d) * e * Z1
            else:
                self.omega_1d += self.eta_1d * e * Z1

        # 2D branch update
        if self.n_2d > 0:
            if self.leakage_2d > 0:
                self.omega_2d *= (1.0 - self.leakage_2d)

            if self.normalized:
                denom_2d = self.eps + float(Z2 @ Z2)
                self.omega_2d += (self.eta_2d / denom_2d) * e * Z2
            else:
                self.omega_2d += self.eta_2d * e * Z2

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