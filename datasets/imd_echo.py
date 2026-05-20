"""
IMD nonlinear echo dataset generator
y(k) = c2 * x(k) * x(k-1) + c3 * x(k)**2 * x(k-1) + noise
Provides several input signal types: colored Gaussian, AR(1), sum of sines, speech waveform (optional).
"""

import numpy as np
from typing import Tuple, Optional


def gen_colored_gaussian(n_samples: int, rho: float = 0.9, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1.0, size=n_samples)
    x = np.zeros(n_samples)
    for k in range(1, n_samples):
        x[k] = rho * x[k - 1] + e[k]
    return x


def gen_ar1(n_samples: int, a: float = 0.8, noise_std: float = 1.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n_samples)
    for k in range(1, n_samples):
        x[k] = a * x[k - 1] + rng.normal(0, noise_std)
    return x


def gen_sines(n_samples: int, f1: float = 0.05, f2: float = 0.11, amp1: float = 1.0, amp2: float = 0.5, seed: int = 0) -> np.ndarray:
    t = np.arange(n_samples)
    return amp1 * np.sin(2 * np.pi * f1 * t) + amp2 * np.sin(2 * np.pi * f2 * t)


def load_speech(wav_path: str, n_samples: int, seed: int = 0) -> np.ndarray:
    """Try to load a speech waveform from wav_path and return first n_samples (mono)."""
    try:
        # prefer soundfile if available
        import soundfile as sf
        data, sr = sf.read(wav_path, always_2d=False)
        if data.ndim > 1:
            data = data[:, 0]
        if len(data) < n_samples:
            # tile if too short
            reps = int(np.ceil(n_samples / len(data)))
            data = np.tile(data, reps)
        return np.asarray(data[:n_samples], dtype=float)
    except Exception:
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(wav_path)
            data = data.astype(float)
            if data.ndim > 1:
                data = data[:, 0]
            if len(data) < n_samples:
                reps = int(np.ceil(n_samples / len(data)))
                data = np.tile(data, reps)
            return data[:n_samples]
        except Exception:
            raise RuntimeError(f"Failed to load speech file: {wav_path} — please install soundfile or scipy, or provide a longer file.")


def generate_imd_echo(
    n_samples: int,
    c2: float = 0.3,
    c3: float = 0.1,
    noise_var: float = 0.0,
    input_type: str = 'colored',
    input_params: Optional[dict] = None,
    speech_path: Optional[str] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate input x and echo y according to IMD equation.

    Returns (x, y) arrays of length n_samples.
    """
    if input_params is None:
        input_params = {}

    if input_type == 'colored':
        rho = input_params.get('rho', 0.9)
        x = gen_colored_gaussian(n_samples, rho=rho, seed=seed)
    elif input_type == 'ar1':
        a = input_params.get('a', 0.8)
        noise_std = input_params.get('noise_std', 1.0)
        x = gen_ar1(n_samples, a=a, noise_std=noise_std, seed=seed)
    elif input_type == 'sines':
        f1 = input_params.get('f1', 0.05)
        f2 = input_params.get('f2', 0.11)
        amp1 = input_params.get('amp1', 1.0)
        amp2 = input_params.get('amp2', 0.5)
        x = gen_sines(n_samples, f1=f1, f2=f2, amp1=amp1, amp2=amp2, seed=seed)
    elif input_type == 'speech':
        if speech_path is None:
            raise ValueError('speech input type requires speech_path')
        x = load_speech(speech_path, n_samples, seed)
    else:
        raise ValueError(f'Unknown input_type: {input_type}')

    # compute IMD echo y(k) = c2 * x(k) * x(k-1) + c3 * x(k)**2 * x(k-1) + noise
    y = compute_imd_y(x, c2, c3)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(noise_var), size=n_samples) if noise_var > 0 else np.zeros(n_samples)
    y_noisy = y + noise
    return x, y_noisy


def build_dataset_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    p: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised dataset for IMD echo identification.

    Target equation:
        y(k) = c2 * x(k) * x(k-1) + c3 * x(k)^2 * x(k-1)

    Therefore, when predicting y(k), the input vector must contain
    the current sample x(k) and recent history:

        X_row(k) = [x(k), x(k-1), ..., x(k-p+1)]

    This avoids the previous off-by-one problem where the model tried
    to predict y(k) without seeing x(k).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")

    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    N = len(x)
    if N < p:
        raise ValueError(f"signal length N={N} is shorter than p={p}")

    rows = N - p + 1
    X = np.zeros((rows, p), dtype=float)
    d = np.zeros(rows, dtype=float)

    for i in range(rows):
        # Current target index.
        k = i + p - 1

        # Use [x(k), x(k-1), ..., x(k-p+1)]
        X[i, :] = x[k - np.arange(p)]

        # Predict y(k)
        d[i] = y[k]

    return X, X.copy(), d, d

def compute_imd_y(x: np.ndarray, c2: float = 0.3, c3: float = 0.1) -> np.ndarray:
    y = np.zeros_like(x)
    for k in range(1, len(x)):
        y[k] = c2 * x[k] * x[k - 1] + c3 * (x[k] ** 2) * x[k - 1]
    return y
