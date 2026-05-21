"""
Experiment C configuration:
Test whether WLNF / WL-LMS becomes unstable when sudden large-amplitude input appears.

C-1: AR1 + amplitude burst
C-2: two-tone sine + amplitude burst
C-3: speech + amplitude burst

This config is independent from exp_b_config.py and will not affect Experiment B.
"""

import os


# Choose one:
#   'C1_AR1_BURST'
#   'C2_SINES_BURST'
#   'C3_SPEECH_BURST'
EXPERIMENT_CASE = 'C3_SPEECH_BURST'


COMMON_DATASET = dict(
    p=5,
    seed=123,
    noise_var=0.0,
    fs=16000.0,

    # Keep this for compatibility with older code, although Experiment C
    # will explicitly use input_params_ar1 / input_params_sines / speech_path.
    input_params={'rho': 0.9},
)


EXP_C_DATASETS = {
    # ------------------------------------------------------------
    # C-1: AR1 + amplitude burst
    # Purpose:
    #   Exclude speech nonstationarity and isolate large-signal shock.
    # ------------------------------------------------------------
    'C1_AR1_BURST': dict(
        **COMMON_DATASET,

        n_train=8000,
        n_test=200,

        input_type='ar1',
        input_params_colored={'rho': 0.9},
        input_params_ar1={'a': 0.9, 'noise_std': 1.0},
        input_params_sines={'f1': 0.05, 'f2': 0.11, 'amp1': 1.0, 'amp2': 0.5},
        speech_path=None,

        amplitude_burst=dict(
            enabled=True,
            domain='train_iter',
            start=4000,
            length=400,
            gain=3.0,
        ),
    ),

    # ------------------------------------------------------------
    # C-2: two-tone sine + amplitude burst
    # Purpose:
    #   Observe whether IMD frequency structure plus large signal
    #   causes residual / MSE spikes.
    # ------------------------------------------------------------
    'C2_SINES_BURST': dict(
        **COMMON_DATASET,

        n_train=8000,
        n_test=200,

        input_type='sines',
        input_params_colored={'rho': 0.9},
        input_params_ar1={'a': 0.9, 'noise_std': 1.0},
        input_params_sines={'f1': 0.05, 'f2': 0.11, 'amp1': 1.0, 'amp2': 0.5},
        speech_path=None,

        amplitude_burst=dict(
            enabled=True,
            domain='train_iter',
            start=4000,
            length=400,
            gain=3.0,
        ),
    ),

    # ------------------------------------------------------------
    # C-3: speech + amplitude burst
    # Purpose:
    #   Closer to AEC; observe WL-LMS robustness under broadband,
    #   nonstationary speech plus sudden large input.
    # ------------------------------------------------------------
    'C3_SPEECH_BURST': dict(
        **COMMON_DATASET,

        n_train=16000,
        n_test=8000,

        input_type='speech',
        input_params_colored={'rho': 0.9},
        input_params_ar1={'a': 0.9, 'noise_std': 1.0},
        input_params_sines={'f1': 0.05, 'f2': 0.11, 'amp1': 1.0, 'amp2': 0.5},

        speech_path=r"D:\pyProject\Idea_1\Data\248-130644-0002.wav",

        amplitude_burst=dict(
            enabled=True,
            domain='train_iter',
            start=6000,
            length=800,
            gain=3.0,
        ),
    ),
}


DATASET = EXP_C_DATASETS[EXPERIMENT_CASE]


ALGO_PARAMS = dict(
    LMS=dict(step_size=0.02),
    KLMS=dict(step_size=0.15, sigma=1.0),
    KRLS=dict(sigma=1.0, reg=1e-3, forgetting=0.999),
    RFFMC=dict(d=100, step_size=0.5, sigma=1.0, kernel_bw=1.0, seed=0),
    NKRGMC=dict(d=100, sigma=1.0, reg=1e-3, forgetting=0.999, kernel_bw=1.0, alpha_order=2.0, seed=0),
    WLLMS=dict(M=50, sigma=0.4, step_size=0.006, seed=0),
    WLRLS=dict(M=20, sigma=1.0, reg=1e-3, forgetting=0.999, seed=0),
)


ALGO_LIST = ['LMS', 'WL-LMS']


IMD = dict(c2=0.3, c3=0.1)


MC_TRIALS = 5


SNAPSHOT = True
SNAPSHOT_EVERY = 1
SS_LAST_N = 1000


# For Experiment C, smoothing should not be too large,
# otherwise the burst spike will be visually suppressed.
PLOT = dict(
    smooth_window=10,
    y_lim=(-80, 40),
)


RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp_c')


PEAK_A = 1.0


NOISE_MODE = 'var'
NOISE_VALUE = 30.0


# Spectrum visualization
SPEC_VIS_LEN = 4096
SPEC_VIS_SEED = 9999
SAVE_SHORT_SPECTRA = False