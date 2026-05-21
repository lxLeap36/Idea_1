"""
Experiment B runner: IMD echo identification
"""
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# import config
sys.path.insert(0, str(ROOT / 'configs'))
from configs.exp_b_config import DATASET, ALGO_PARAMS, MC_TRIALS, PLOT, RESULT_DIR, ALGO_LIST, IMD, SNAPSHOT, SNAPSHOT_EVERY, SS_LAST_N, PEAK_A, NOISE_MODE, NOISE_VALUE, SPEC_VIS_LEN, SPEC_VIS_SEED

from scenarios.imd_echo_scenario import run_imd_experiment
from utils.plotting import (
    plot_learning_curves,
    plot_spectra_comparison,
    plot_spectra_grid,
    plot_residuals_grid,
    plot_speech_spectra_grid,
    plot_speech_residual_spectra_grid,
)
from datasets.imd_echo import generate_imd_echo, build_dataset_from_xy, compute_imd_y
import numpy as np

# timestamped results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_PATH = Path(RESULT_DIR) / timestamp
RESULT_PATH.mkdir(parents=True, exist_ok=True)


def save_curves_csv(curves: dict, path: Path):
    names = list(curves.keys())
    n = max(len(v) for v in curves.values())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration'] + names)
        for i in range(n):
            row = [i + 1] + [f'{curves[name][i]:.6f}' if i < len(curves[name]) else '' for name in names]
            writer.writerow(row)
    print(f'[csv] saved: {path}')


def save_summary_csv(ss_mse: dict, avg_time: dict, path: Path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'SS_MSE(dB)', 'Time(s)'])
        for name in ss_mse:
            writer.writerow([name, f'{ss_mse[name]:.3f}', f'{avg_time[name]:.4f}'])
    print(f'[csv] saved: {path}')


def main():
    print('▶ Experiment B: IMD echo identification')
    print('results ->', RESULT_PATH)

    algo_list = list(ALGO_LIST)
    # Print configuration summary so user knows what's being run
    print('\n[Config]')
    print(f"  Algorithms: {algo_list}")
    print(f"  Dataset input_type: {DATASET.get('input_type')}, n_train={DATASET.get('n_train')}, n_test={DATASET.get('n_test')}, p={DATASET.get('p')}")
    print(f"  IMD coeffs: c2={IMD.get('c2')}, c3={IMD.get('c3')}")
    print(f"  Peak A: {PEAK_A}, Noise mode: {NOISE_MODE}, Noise value: {NOISE_VALUE}")

    t0 = time.time()
    avg_curves, ss_mse, avg_time, last_trial_results, last_signals = run_imd_experiment(DATASET, IMD, ALGO_PARAMS,
                                                                                       n_trials=MC_TRIALS,
                                                                                       snapshot=SNAPSHOT,
                                                                                       snapshot_every=SNAPSHOT_EVERY,
                                                                                       ss_last_n=SS_LAST_N,
                                                                                       verbose=True,
                                                                                       algo_list=ALGO_LIST)
    print(f'elapsed: {time.time()-t0:.1f}s')
    print('\n[Done] Experiment finished successfully.')

    save_curves_csv(avg_curves, RESULT_PATH / 'imd_curves.csv')
    save_summary_csv(ss_mse, avg_time, RESULT_PATH / 'imd_summary.csv')

    plot_learning_curves(results=avg_curves, title='IMD Echo (Testing MSE)', save_path=RESULT_PATH / 'fig_imd.png', smooth_window=PLOT['smooth_window'], y_lim=PLOT['y_lim'])

    # Broadband spectral visualizations for speech input.
    # Unlike sine input, speech has no fixed f1/f2 or discrete IMD lines,
    # so we plot broadband spectra and residual spectra.
    if DATASET.get('input_type') == 'speech' and last_signals is not None and last_trial_results is not None:
        x_full, y_full = last_signals

        p = int(DATASET.get('p', 5))
        n_train = int(DATASET.get('n_train'))
        n_test = int(DATASET.get('n_test'))

        # build_dataset_from_xy() uses rows:
        # X row k -> [x(k), x(k-1), ...], target d -> y(k)
        # After training samples, test target starts at original index:
        # k = p - 1 + n_train
        start = p - 1 + n_train
        end = start + n_test

        x_seg = x_full[start:end]
        y_seg = y_full[start:end]

        # Sampling rate for display only.
        # If your speech file is 16 kHz, keep this default.
        fs_speech = float(DATASET.get('fs', 16000.0))

        algo_signals = {}

        for name in list(ALGO_LIST):
            if name not in last_trial_results:
                continue

            preds = last_trial_results[name].get('final_preds', None)
            if preds is None:
                print(f"[speech spectra] no predictions for {name}, skipping")
                continue

            y_pred = np.asarray(preds, dtype=float)

            L = min(len(x_seg), len(y_seg), len(y_pred))
            if L <= 0:
                print(f"[speech spectra] empty signals for {name}, skipping")
                continue

            x_plot = np.asarray(x_seg[:L], dtype=float)
            y_plot = np.asarray(y_seg[:L], dtype=float)
            y_pred_plot = y_pred[:L]
            residual_plot = y_plot - y_pred_plot

            algo_signals[name] = (x_plot, y_plot, y_pred_plot, residual_plot)

        if len(algo_signals) > 0:
            plot_speech_spectra_grid(
                algo_signals,
                RESULT_PATH / 'speech_spectra_grid.png',
                fs=fs_speech,
                nfft=None,
            )

            plot_speech_residual_spectra_grid(
                algo_signals,
                RESULT_PATH / 'speech_residual_spectra_grid.png',
                fs=fs_speech,
                nfft=None,
            )

    # Additional spectral visualizations for 'sines' input: per-algorithm spectra of input x, target y, prediction, and residual
    if DATASET.get('input_type') == 'sines' and last_signals is not None and last_trial_results is not None:
        x_full, y_full = last_signals
        # compute test segment indices consistent with build_dataset_from_xy: d corresponds to y[p:]
        p = DATASET.get('p', 5)
        n_train = DATASET.get('n_train')
        n_test = DATASET.get('n_test')
        # d_test corresponds to y[p + n_train : p + n_train + n_test]
        start = p - 1 + n_train
        end = start + n_test
        x_seg = x_full[start:end]
        y_seg = y_full[start:end]

        # fundamental frequencies
        sines_params = DATASET.get('input_params_sines', {})
        f1 = sines_params.get('f1')
        f2 = sines_params.get('f2')

        for name in list(ALGO_LIST):
            if name not in last_trial_results:
                continue
            preds = last_trial_results[name].get('final_preds', None)
            if preds is None:
                print(f"[spectra] no predictions for {name}, skipping")
                continue

            # preds correspond to d_test length (n_test)
            y_pred = preds
            residual = y_seg - y_pred
            # save figure
            if DATASET.get('SAVE_SHORT_SPECTRA', False) or globals().get('SAVE_SHORT_SPECTRA', False):
                fig_name = RESULT_PATH / f'spectra_{name}.png'
                plot_spectra_comparison(x_seg, y_seg, y_pred, name, fig_name, fs=1.0, f1=f1, f2=f2)

    # High-resolution spectral visualizations using snapshot-reconstructed final model on a long noiseless signal
    if DATASET.get('input_type') == 'sines' and last_trial_results is not None:
        # prepare visualization params
        p = DATASET.get('p', 5)
        base_n_vis = int(SPEC_VIS_LEN) if SPEC_VIS_LEN is not None else 4096
        seed_vis = int(SPEC_VIS_SEED) if SPEC_VIS_SEED is not None else (DATASET.get('seed', 0) + 9999)
        sines_params = DATASET.get('input_params_sines', {})
        f1 = sines_params.get('f1')
        f2 = sines_params.get('f2')

        # choose N so that f1*N and f2*N are integers -> avoid spectral leakage
        N = base_n_vis
        try:
            from fractions import Fraction
            from math import gcd

            def lcm(a, b):
                return a * b // gcd(a, b)

            if f1 is not None and f2 is not None:
                frac1 = Fraction(f1).limit_denominator(10000)
                frac2 = Fraction(f2).limit_denominator(10000)
                den1 = frac1.denominator
                den2 = frac2.denominator
                base_period = lcm(den1, den2)
                # pick N as the smallest multiple of base_period >= base_n_vis
                mult = -(-base_n_vis // base_period)  # ceiling division
                N = int(base_period * mult)
                print(f"[spectra] f1 ~ {frac1} (den={den1}), f2 ~ {frac2} (den={den2}), base_period={base_period}, chosen N={N}")
        except Exception:
            # fallback to base_n_vis
            N = base_n_vis

        n_samples_vis = N + p

        # generate long noiseless input and clean output with length n_samples_vis
        x_long, _ = generate_imd_echo(n_samples_vis,
                                      c2=IMD.get('c2'), c3=IMD.get('c3'),
                                      noise_var=0.0,
                                      input_type='sines',
                                      input_params=sines_params,
                                      speech_path=None,
                                      seed=seed_vis)
        # scale to PEAK_A
        A = PEAK_A
        peak = float(np.max(np.abs(x_long))) if x_long.size > 0 else 1.0
        scale = A / peak if peak > 0 else 1.0
        x_long = x_long * scale
        y_long = compute_imd_y(x_long, c2=IMD.get('c2'), c3=IMD.get('c3'))

        X_vis, X_vis_clean, d_vis, _ = build_dataset_from_xy(x_long, y_long, p)

        # collect per-algo signals (use same x_plot/y_plot length N for all)
        algo_signals = {}
        for name in list(ALGO_LIST):
            if name not in last_trial_results:
                continue
            res = last_trial_results[name]
            # reconstruct model from final_snapshot
            model = None
            snapshot = res.get('final_snapshot', None)
            algo_class = res.get('algo_class', None)
            init_kwargs = res.get('algo_init_kwargs', None) or {}
            L = res.get('algo_filter_length', None)

            if snapshot is not None:
                tag, payload = snapshot
                if tag == 'state' and payload is not None and algo_class is not None:
                    try:
                        model = algo_class(**init_kwargs) if init_kwargs else (algo_class(L) if L is not None else algo_class())
                        if hasattr(model, 'set_state'):
                            model.set_state(payload)
                        else:
                            if isinstance(payload, dict):
                                for k, v in payload.items():
                                    try:
                                        setattr(model, k, v)
                                    except Exception:
                                        pass
                    except Exception:
                        model = None
                elif tag == 'copy' and payload is not None:
                    model = payload

            if model is None and algo_class is not None:
                try:
                    model = algo_class(**init_kwargs) if init_kwargs else (algo_class(L) if L is not None else algo_class())
                except Exception:
                    model = None

            if model is None:
                print(f"[spectra] cannot reconstruct model for {name}, skipping high-res plot")
                continue

            # predict on long rows
            try:
                y_pred_long = np.array([model.predict(xx) for xx in X_vis])
            except Exception:
                print(f"[spectra] model.predict failed for {name} on long signal, skipping")
                continue

            x_plot = x_long[p:p + N]
            y_plot = d_vis[:N]
            res_plot = y_plot - y_pred_long[:N]
            algo_signals[name] = (x_plot, y_plot, y_pred_long[:N], res_plot)

        # draw combined grid figure
        if len(algo_signals) > 0:
            fig_name1 = RESULT_PATH / 'spectra_grid_hr.png'
            fig_name2 = RESULT_PATH / 'residuals_grid_hr.png'
            plot_spectra_grid(algo_signals, fig_name1, fs=1.0, f1=f1, f2=f2, nfft=N)
            plot_residuals_grid(algo_signals, fig_name2, fs=1.0, f1=f1, f2=f2, nfft=N)

    print('\nDone. Results saved to', RESULT_PATH)

if __name__ == '__main__':
    main()
