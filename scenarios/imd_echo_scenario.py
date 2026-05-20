"""
IMD echo identification scenario wrapper.
Reuses run_monte_carlo from nonlinear_id.py where possible.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.imd_echo import generate_imd_echo, build_dataset_from_xy, compute_imd_y
from configs.exp_b_config import PEAK_A, NOISE_MODE, NOISE_VALUE
from scenarios.nonlinear_id import build_algorithms, run_monte_carlo

# store last trial raw signals for downstream analysis (spectra)
LAST_TRIAL_SIGNALS = []


def build_imd_fn(dataset_cfg, imd_cfg, algo_list=None):
    def _build(trial):
        seed = dataset_cfg.get('seed', 0) + trial
        print(f"\n[Trial {trial+1}] building dataset (seed={seed})...")
        n_total = dataset_cfg['n_train'] + dataset_cfg['n_test'] + dataset_cfg.get('p', 5) + 2
        # generate raw x (noisy target handled below according to noise mode)
        input_type = dataset_cfg.get('input_type','colored')
        if input_type == 'colored':
            input_params = dataset_cfg.get('input_params_colored', dataset_cfg.get('input_params', {}))
            speech_path = None
        elif input_type == 'ar1':
            input_params = dataset_cfg.get('input_params_ar1', {})
            speech_path = None
        elif input_type == 'sines':
            input_params = dataset_cfg.get('input_params_sines', {})
            speech_path = None
        elif input_type == 'speech':
            input_params = None
            speech_path = dataset_cfg.get('speech_path', None)
        else:
            input_params = dataset_cfg.get('input_params', {})
            speech_path = dataset_cfg.get('speech_path', None)

        x, _ = generate_imd_echo(n_total,
                                 c2=imd_cfg['c2'], c3=imd_cfg['c3'],
                                 noise_var=0.0,
                                 input_type=input_type,
                                 input_params=input_params,
                                 speech_path=speech_path,
                                 seed=seed)

        # Scale x so peak amplitude <= PEAK_A. For speech, we'll scale based on training portion.
        A = PEAK_A
        # temporary compute indices for train/test split
        p = dataset_cfg.get('p', 5)
        n_train = dataset_cfg['n_train']
        n_test = dataset_cfg['n_test']
        # Determine scale factor over the whole signal, except for speech where we scale by train peak
        input_type = dataset_cfg.get('input_type','colored')
        print(f"  input_type: {input_type}")
        if input_type == 'speech':
            # scale based on training segment peak
            x_train_segment = x[:n_train + p]
            peak = float(np.max(np.abs(x_train_segment))) if x_train_segment.size > 0 else 1.0
            scale = A / peak if peak > 0 else 1.0
        else:
            peak = float(np.max(np.abs(x))) if x.size > 0 else 1.0
            scale = A / peak if peak > 0 else 1.0
        x = x * scale
        print(f"  peak_before={peak:.6f}, scale={scale:.6f}, peak_after={float(np.max(np.abs(x))):.6f}")

        # compute clean IMD y from scaled x
        y_clean = compute_imd_y(x, c2=imd_cfg['c2'], c3=imd_cfg['c3'])

        # determine noise variance
        if NOISE_MODE == 'var':
            noise_var = float(dataset_cfg.get('noise_var', 0.0))
        else:
            # NOISE_MODE == 'snr' -> NOISE_VALUE is desired SNR in dB
            desired_snr_db = float(NOISE_VALUE)
            # compute signal power from training portion of y
            y_tr = y_clean[:n_train + p]
            sig_pow = float(np.mean(y_tr ** 2)) if y_tr.size > 0 else 0.0
            if sig_pow <= 0:
                noise_var = 0.0
            else:
                noise_var = sig_pow / (10 ** (desired_snr_db / 10.0))
        print(f"  computed noise_var={noise_var:.6e} (mode={NOISE_MODE})")

        # add noise
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, np.sqrt(noise_var), size=n_total) if noise_var > 0 else np.zeros(n_total)
        y = y_clean + noise

        # save raw signals for later inspection (module-level list)
        # store copies to avoid accidental mutation
        LAST_TRIAL_SIGNALS.append((x.copy(), y.copy()))

        # build supervised matrices and split
        X_tr, X_tr_clean, d_tr, _ = build_dataset_from_xy(x, y, p)
        X_train = X_tr[:n_train]
        d_train = d_tr[:n_train]
        X_test = X_tr_clean[n_train:n_train + n_test]
        d_test = d_tr[n_train:n_train + n_test]
        # Determine which algorithms to instantiate: prefer explicit algo_list passed to builder,
        # otherwise fall back to dataset_cfg['algo_list'] if present.
        requested = algo_list if algo_list is not None else dataset_cfg.get('algo_list', None)
        algos = build_algorithms(dataset_cfg.get('p', 5), ALGO_PARAMS, requested)
        print(f"  instantiating algorithms: {list(algos.keys())}")
        return algos, X_train, d_train, X_test, d_test
    return _build

# Note: we import ALGO_PARAMS at runtime from configs in the runner

def run_imd_experiment(dataset_cfg, imd_cfg, algo_params, n_trials=10, snapshot=True, snapshot_every=1, ss_last_n=1000, verbose=True, algo_list=None):
    global ALGO_PARAMS
    ALGO_PARAMS = algo_params
    build_fn = build_imd_fn(dataset_cfg, imd_cfg, algo_list=algo_list)
    avg_curves, ss_mse, avg_time, last_trial_results = run_monte_carlo(build_fn, n_trials, verbose, snapshot=snapshot, snapshot_every=snapshot_every, ss_last_n=ss_last_n, return_last_trial=True)
    # retrieve last raw signals if available
    last_signals = LAST_TRIAL_SIGNALS[-1] if len(LAST_TRIAL_SIGNALS) > 0 else (None, None)
    return avg_curves, ss_mse, avg_time, last_trial_results, last_signals
