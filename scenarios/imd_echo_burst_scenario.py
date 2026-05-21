"""
Experiment C scenario:
IMD echo identification with sudden amplitude burst.

This scenario is independent from scenarios/imd_echo_scenario.py,
so Experiment B remains unchanged.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.imd_echo import generate_imd_echo, build_dataset_from_xy, compute_imd_y
from configs.exp_c_config import PEAK_A, NOISE_MODE, NOISE_VALUE
from scenarios.nonlinear_id import build_algorithms, run_monte_carlo


LAST_TRIAL_SIGNALS = []
LAST_TRIAL_BURST_INFO = []


def apply_amplitude_burst(
    x: np.ndarray,
    burst_cfg: dict,
    p: int,
    n_train: int,
    n_test: int,
):
    """
    Apply sudden amplitude burst to x.

    Important:
        This function should be called AFTER normal peak scaling
        and BEFORE compute_imd_y().

    Supported domain:
        'train_iter':
            burst start is interpreted as training iteration index.
            sample index = p - 1 + start

        'test_iter':
            burst start is interpreted as test iteration index.
            sample index = p - 1 + n_train + start

        'sample':
            burst start is interpreted as raw signal sample index.
    """
    if burst_cfg is None:
        return x, None

    if not bool(burst_cfg.get('enabled', False)):
        return x, None

    x = np.asarray(x, dtype=float).copy()

    domain = burst_cfg.get('domain', 'train_iter')
    start = int(burst_cfg.get('start', n_train // 2))
    length = int(burst_cfg.get('length', 200))
    gain = float(burst_cfg.get('gain', 3.0))

    if length <= 0:
        return x, None

    if domain == 'train_iter':
        start_sample = (p - 1) + start
    elif domain == 'test_iter':
        start_sample = (p - 1) + n_train + start
    elif domain == 'sample':
        start_sample = start
    else:
        raise ValueError(
            f"Unknown burst domain: {domain}. "
            "Use 'train_iter', 'test_iter', or 'sample'."
        )

    start_sample = max(0, min(start_sample, len(x)))
    end_sample = max(0, min(start_sample + length, len(x)))

    if end_sample <= start_sample:
        return x, None

    peak_before = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    rms_before = float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else 0.0

    x[start_sample:end_sample] *= gain

    peak_after = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    rms_after = float(np.sqrt(np.mean(x ** 2))) if x.size > 0 else 0.0

    info = dict(
        enabled=True,
        domain=domain,
        start=start,
        length=length,
        gain=gain,
        start_sample=start_sample,
        end_sample=end_sample,
        peak_before=peak_before,
        peak_after=peak_after,
        rms_before=rms_before,
        rms_after=rms_after,
    )

    return x, info


def build_imd_burst_fn(dataset_cfg, imd_cfg, algo_params, algo_list=None):
    def _build(trial):
        seed = dataset_cfg.get('seed', 0) + trial
        print(f"\n[Trial {trial + 1}] building Experiment C dataset (seed={seed})...")

        p = int(dataset_cfg.get('p', 5))
        n_train = int(dataset_cfg['n_train'])
        n_test = int(dataset_cfg['n_test'])
        n_total = n_train + n_test + p + 2

        input_type = dataset_cfg.get('input_type', 'ar1')

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

        x, _ = generate_imd_echo(
            n_total,
            c2=imd_cfg['c2'],
            c3=imd_cfg['c3'],
            noise_var=0.0,
            input_type=input_type,
            input_params=input_params,
            speech_path=speech_path,
            seed=seed,
        )

        print(f"  input_type: {input_type}")

        # Normal scaling first.
        A = PEAK_A
        if input_type == 'speech':
            x_train_segment = x[:n_train + p]
            peak = float(np.max(np.abs(x_train_segment))) if x_train_segment.size > 0 else 1.0
        else:
            peak = float(np.max(np.abs(x))) if x.size > 0 else 1.0

        scale = A / peak if peak > 0 else 1.0
        x = x * scale

        print(
            f"  normal scale: peak_before={peak:.6f}, "
            f"scale={scale:.6f}, peak_after={float(np.max(np.abs(x))):.6f}"
        )

        # Experiment C burst:
        # Apply burst after normal scaling and before computing y_clean.
        burst_cfg = dataset_cfg.get('amplitude_burst', None)
        x, burst_info = apply_amplitude_burst(
            x,
            burst_cfg=burst_cfg,
            p=p,
            n_train=n_train,
            n_test=n_test,
        )

        if burst_info is not None:
            print(
                "  amplitude_burst: "
                f"domain={burst_info['domain']}, "
                f"start={burst_info['start']}, "
                f"length={burst_info['length']}, "
                f"gain={burst_info['gain']}, "
                f"sample_range=[{burst_info['start_sample']}, {burst_info['end_sample']}), "
                f"peak_before={burst_info['peak_before']:.6f}, "
                f"peak_after={burst_info['peak_after']:.6f}"
            )

        # Compute IMD output from burst-modified x.
        y_clean = compute_imd_y(x, c2=imd_cfg['c2'], c3=imd_cfg['c3'])

        # Noise handling.
        if NOISE_MODE == 'var':
            noise_var = float(dataset_cfg.get('noise_var', 0.0))
        else:
            desired_snr_db = float(NOISE_VALUE)
            y_tr = y_clean[:n_train + p]
            sig_pow = float(np.mean(y_tr ** 2)) if y_tr.size > 0 else 0.0
            noise_var = 0.0 if sig_pow <= 0 else sig_pow / (10 ** (desired_snr_db / 10.0))

        print(f"  computed noise_var={noise_var:.6e} (mode={NOISE_MODE})")

        rng = np.random.default_rng(seed)
        noise = rng.normal(0, np.sqrt(noise_var), size=n_total) if noise_var > 0 else np.zeros(n_total)
        y = y_clean + noise

        LAST_TRIAL_SIGNALS.append((x.copy(), y.copy()))
        LAST_TRIAL_BURST_INFO.append(burst_info)

        X_all, X_clean, d_all, _ = build_dataset_from_xy(x, y, p)

        X_train = X_all[:n_train]
        d_train = d_all[:n_train]
        X_test = X_clean[n_train:n_train + n_test]
        d_test = d_all[n_train:n_train + n_test]

        requested = algo_list if algo_list is not None else dataset_cfg.get('algo_list', None)
        algos = build_algorithms(p, algo_params, requested)

        print(f"  instantiating algorithms: {list(algos.keys())}")

        return algos, X_train, d_train, X_test, d_test

    return _build


def run_imd_burst_experiment(
    dataset_cfg,
    imd_cfg,
    algo_params,
    n_trials=5,
    snapshot=True,
    snapshot_every=1,
    ss_last_n=1000,
    verbose=True,
    algo_list=None,
):
    build_fn = build_imd_burst_fn(
        dataset_cfg=dataset_cfg,
        imd_cfg=imd_cfg,
        algo_params=algo_params,
        algo_list=algo_list,
    )

    avg_curves, ss_mse, avg_time, last_trial_results = run_monte_carlo(
        build_fn,
        n_trials,
        verbose,
        snapshot=snapshot,
        snapshot_every=snapshot_every,
        ss_last_n=ss_last_n,
        return_last_trial=True,
    )

    last_signals = LAST_TRIAL_SIGNALS[-1] if len(LAST_TRIAL_SIGNALS) > 0 else (None, None)
    last_burst_info = LAST_TRIAL_BURST_INFO[-1] if len(LAST_TRIAL_BURST_INFO) > 0 else None

    return avg_curves, ss_mse, avg_time, last_trial_results, last_signals, last_burst_info