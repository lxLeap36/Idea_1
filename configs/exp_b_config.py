"""
Experiment B configuration: IMD nonlinear echo identification
"""
import os

DATASET = dict(
    n_train = 2000,
    n_test = 100,
    p = 5,
    seed = 123,
    noise_var = 0.0,
    # input choices: 'colored', 'ar1', 'sines', 'speech'
    input_type = 'sines',
    # parameters for colored Gaussian (AR(1) on noise): {'rho': 0.9}
    input_params_colored = {'rho': 0.9},
    # parameters for AR(1) signal: {'a': 0.8, 'noise_std': 1.0}
    input_params_ar1 = {'a': 0.9, 'noise_std': 1.0},
    # parameters for sum-of-sines: {'f1':0.05, 'f2':0.11, 'amp1':1.0, 'amp2':0.5}
    input_params_sines = {'f1': 0.05, 'f2': 0.11, 'amp1': 1.0, 'amp2': 0.5},
    # speech input: provide path to wav file; if None, speech is unavailable
    speech_path = r"D:\pyProject\Idea_1\Data\248-130644-0002.wav",
    fs = 16000.0,
    # backward-compatible alias for older code expecting 'input_params'
    input_params = {'rho': 0.9},
)

ALGO_PARAMS = dict(
    LMS = dict(step_size=0.02),
    KLMS = dict(step_size=0.15, sigma=1.0),
    KRLS = dict(sigma=1.0, reg=1e-3, forgetting=0.999),
    RFFMC = dict(d=100, step_size=0.5, sigma=1.0, kernel_bw=1.0, seed=0),
    NKRGMC = dict(d=100, sigma=1.0, reg=1e-3, forgetting=0.999, kernel_bw=1.0, alpha_order=2.0, seed=0),
    WLLMS = dict(M=40, sigma=0.4, step_size=0.0005, seed=0),
    GHWLLMS = dict(M=40, scale=0.6, step_size=0.2, normalized=True, eps=1e-8, seed=0),
    GH2DWLLMS=dict(
        M=40,
        scale=0.6,
        step_size=0.01,
        step_size_1d=0.0,
        step_size_2d=0.002,
        normalized=True,
        eps=1e-8,
        include_1d=False,
        cross_pairs=[(0, 1)],
        cross_orders=[(1, 1), (2, 1)],
        leakage_1d=0.0,
        leakage_2d=0.0,
        seed=0,
    ),
    WLRLS = dict(M=20, sigma=1.0, reg=1e-3, forgetting=0.999, seed=0),
)

ALGO_LIST = ['LMS', 'WL-LMS', 'GH-WL-LMS', 'GH2D-WL-LMS']

# IMD coefficients
IMD = dict(c2=0.3, c3=0.1)

# Monte Carlo
MC_TRIALS = 5

# snapshot defaults (reuse exp3 defaults but override if needed)
SNAPSHOT = True
SNAPSHOT_EVERY = 1
SS_LAST_N = 1000

# plotting / results
PLOT = dict(smooth_window=80, y_lim=(-60, 10))

RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_b')

# Input peak amplitude: scale inputs so peak is within [-A, A]
PEAK_A = 1.0

# Spectral visualization: length of long, noiseless signal used for high-resolution FFT
# Set to 2000 or 4096 as desired. Also allow fixed seed for reproducibility.
SPEC_VIS_LEN = 4096
SPEC_VIS_SEED = 9999

# Whether to save per-algorithm short-test spectra (based on n_test). Set False to avoid clutter.
SAVE_SHORT_SPECTRA = False

# Noise generation mode: 'var' means noise_var is variance; 'snr' means noise_value is desired SNR in dB
# If 'snr', the scenario will compute noise variance from signal power and SNR
NOISE_MODE = 'var'  # 'var' or 'snr'
# If NOISE_MODE == 'var' then NOISE_VALUE is interpreted as variance; if 'snr' as dB
NOISE_VALUE = 30.0  # SNR in dB when NOISE_MODE == 'snr'
