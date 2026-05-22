"""
Experiment C runner:
WLNF / WL-LMS robustness under sudden large-amplitude input burst.

This runner is independent from run_exp_b.py and will not affect Experiment B.
"""

import sys
import time
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / 'configs'))

from configs.exp_c_config import (
    EXPERIMENT_CASE,
    DATASET,
    ALGO_PARAMS,
    MC_TRIALS,
    PLOT,
    RESULT_DIR,
    ALGO_LIST,
    IMD,
    SNAPSHOT,
    SNAPSHOT_EVERY,
    SS_LAST_N,
)

from scenarios.imd_echo_burst_scenario import run_imd_burst_experiment
from utils.plotting import ALGO_STYLES, smooth


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULT_PATH = Path(RESULT_DIR) / EXPERIMENT_CASE / timestamp
RESULT_PATH.mkdir(parents=True, exist_ok=True)


def save_curves_csv(curves: dict, path: Path):
    names = list(curves.keys())
    n = max(len(v) for v in curves.values())

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration'] + names)

        for i in range(n):
            row = [str(i + 1)]
            for name in names:
                if i < len(curves[name]):
                    row.append(f'{curves[name][i]:.6f}')
                else:
                    row.append('')
            writer.writerow(row)

    print(f'[csv] saved: {path}')


def save_summary_csv(ss_mse: dict, avg_time: dict, path: Path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'SS_MSE(dB)', 'Time(s)'])
        for name in ss_mse:
            writer.writerow([name, f'{ss_mse[name]:.3f}', f'{avg_time[name]:.4f}'])

    print(f'[csv] saved: {path}')


def compute_spike_metrics(curves: dict, burst_cfg: dict, ss_mse: dict):
    """
    Compute simple spike metrics from averaged testing MSE curves.

    The learning curve x-axis is training iteration.
    This is meaningful when burst_cfg['domain'] == 'train_iter'.
    """
    rows = []

    if burst_cfg is None or not bool(burst_cfg.get('enabled', False)):
        return rows

    if burst_cfg.get('domain', 'train_iter') != 'train_iter':
        return rows

    start = int(burst_cfg.get('start', 0))
    length = int(burst_cfg.get('length', 0))
    end = start + length

    for name, curve in curves.items():
        curve = np.asarray(curve, dtype=float)

        if curve.size == 0:
            continue

        s = max(0, min(start, curve.size))
        e = max(0, min(end, curve.size))

        if e <= s:
            burst_peak = np.nan
            burst_mean = np.nan
        else:
            burst_peak = float(np.max(curve[s:e]))
            burst_mean = float(np.mean(curve[s:e]))

        overall_peak = float(np.max(curve))
        ss = float(ss_mse.get(name, np.nan))

        rows.append(dict(
            Algorithm=name,
            Burst_Start=start,
            Burst_End=end,
            Burst_Peak_MSE_dB=burst_peak,
            Burst_Mean_MSE_dB=burst_mean,
            Overall_Peak_MSE_dB=overall_peak,
            SS_MSE_dB=ss,
            Spike_Over_SS_dB=burst_peak - ss if np.isfinite(burst_peak) and np.isfinite(ss) else np.nan,
        ))

    return rows


def save_spike_metrics_csv(rows: list, path: Path):
    if len(rows) == 0:
        print('[csv] no spike metrics to save')
        return

    headers = list(rows[0].keys())

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            writer.writerow([row[h] for h in headers])

    print(f'[csv] saved: {path}')


def plot_learning_curves_with_burst(
    results: dict,
    title: str,
    save_path: Path,
    smooth_window: int = 10,
    y_lim: tuple = None,
    burst_cfg: dict = None,
    algo_params: dict = None,
    param_keys: dict = None,
):
    """Plot learning curves and optionally annotate legend with main algo params.

    Parameters:
    - algo_params: dict from config mapping algo_key -> param dict (e.g. ALGO_PARAMS)
    - param_keys: optional dict mapping algo_key -> list of param names to display
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # helper: normalize name to match keys like 'WLLMS' or 'GHWLLMS'
    import re
    def _norm_key(s):
        return re.sub(r'[^0-9A-Za-z]', '', s)

    # sensible defaults for which params to show per algorithm key (normalized)
    default_param_keys = {
        'LMS': ['step_size'],
        'KLMS': ['step_size', 'sigma'],
        'WLLMS': ['M', 'sigma', 'step_size'],
        'GHWLLMS': ['M', 'scale', 'step_size'],
    }

    for name, curve in results.items():
        style = ALGO_STYLES.get(name, {'color': 'black', 'linestyle': '-', 'label': name})
        smoothed = smooth(np.asarray(curve), smooth_window)

        # build legend label possibly augmented with key params
        legend_label = style.get('label', name)
        if algo_params:
            cfg_key = None
            # try direct match first
            if name in algo_params:
                cfg_key = name
            else:
                nk = _norm_key(name)
                # try matching normalized keys
                for k in algo_params.keys():
                    if _norm_key(k).lower() == nk.lower():
                        cfg_key = k
                        break

            if cfg_key is not None:
                params = algo_params.get(cfg_key, {}) or {}
                # choose keys to show: priority param_keys -> default_param_keys
                keys_to_show = None
                if param_keys and cfg_key in param_keys:
                    keys_to_show = param_keys[cfg_key]
                else:
                    keys_to_show = default_param_keys.get(_norm_key(cfg_key), None)

                # fallback: first two params
                if not keys_to_show:
                    try:
                        ks = list(params.keys())
                        keys_to_show = ks[:2]
                    except Exception:
                        keys_to_show = []

                if keys_to_show:
                    parts = []
                    for kk in keys_to_show:
                        if kk in params:
                            v = params[kk]
                            if isinstance(v, float):
                                vs = f"{v:.4g}"
                            else:
                                vs = str(v)
                            parts.append(f"{kk}={vs}")
                    if parts:
                        legend_label = f"{legend_label} ({', '.join(parts)})"

        ax.plot(smoothed, color=style['color'], linestyle=style['linestyle'], label=legend_label)

    if burst_cfg is not None and bool(burst_cfg.get('enabled', False)):
        domain = burst_cfg.get('domain', 'train_iter')

        if domain == 'train_iter':
            start = int(burst_cfg.get('start', 0))
            length = int(burst_cfg.get('length', 0))
            end = start + length

            ax.axvline(
                start,
                color='tab:purple',
                linestyle='--',
                linewidth=1.2,
                alpha=0.85,
                label='burst start',
            )
            ax.axvline(
                end,
                color='tab:purple',
                linestyle=':',
                linewidth=1.2,
                alpha=0.85,
                label='burst end',
            )

            ymin, ymax = ax.get_ylim()
            ax.text(
                start,
                ymax,
                'burst start',
                rotation=90,
                va='top',
                ha='right',
                fontsize=8,
                color='tab:purple',
            )
            ax.text(
                end,
                ymax,
                'burst end',
                rotation=90,
                va='top',
                ha='right',
                fontsize=8,
                color='tab:purple',
            )

    ax.set_xlabel('Iteration k')
    ax.set_ylabel('MSE (dB)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f'[plot] saved: {save_path}')


def main():
    print('▶ Experiment C: sudden amplitude burst robustness')
    print('case ->', EXPERIMENT_CASE)
    print('results ->', RESULT_PATH)

    print('\n[Config]')
    print(f"  Algorithms: {list(ALGO_LIST)}")
    print(
        f"  Dataset input_type: {DATASET.get('input_type')}, "
        f"n_train={DATASET.get('n_train')}, "
        f"n_test={DATASET.get('n_test')}, "
        f"p={DATASET.get('p')}"
    )
    print(f"  IMD coeffs: c2={IMD.get('c2')}, c3={IMD.get('c3')}")
    print(f"  amplitude_burst: {DATASET.get('amplitude_burst', None)}")

    t0 = time.time()

    avg_curves, ss_mse, avg_time, last_trial_results, last_signals, last_burst_info = run_imd_burst_experiment(
        DATASET,
        IMD,
        ALGO_PARAMS,
        n_trials=MC_TRIALS,
        snapshot=SNAPSHOT,
        snapshot_every=SNAPSHOT_EVERY,
        ss_last_n=SS_LAST_N,
        verbose=True,
        algo_list=ALGO_LIST,
    )

    print(f'elapsed: {time.time() - t0:.1f}s')
    print('\n[Done] Experiment C finished successfully.')

    save_curves_csv(avg_curves, RESULT_PATH / 'exp_c_curves.csv')
    save_summary_csv(ss_mse, avg_time, RESULT_PATH / 'exp_c_summary.csv')

    spike_rows = compute_spike_metrics(
        curves=avg_curves,
        burst_cfg=DATASET.get('amplitude_burst', None),
        ss_mse=ss_mse,
    )
    save_spike_metrics_csv(spike_rows, RESULT_PATH / 'exp_c_spike_metrics.csv')

    plot_learning_curves_with_burst(
        results=avg_curves,
        title=f'Experiment C — {EXPERIMENT_CASE}',
        save_path=RESULT_PATH / 'fig_exp_c_burst.png',
        smooth_window=PLOT.get('smooth_window', 10),
        y_lim=PLOT.get('y_lim', None),
        burst_cfg=DATASET.get('amplitude_burst', None),
        algo_params=ALGO_PARAMS,
    )

    print('\n[Summary]')
    for name in ss_mse:
        print(f"  {name}: SS_MSE={ss_mse[name]:.3f} dB, time={avg_time[name]:.4f}s")

    if len(spike_rows) > 0:
        print('\n[Spike metrics]')
        for row in spike_rows:
            print(
                f"  {row['Algorithm']}: "
                f"burst_peak={row['Burst_Peak_MSE_dB']:.3f} dB, "
                f"SS={row['SS_MSE_dB']:.3f} dB, "
                f"spike_over_SS={row['Spike_Over_SS_dB']:.3f} dB"
            )

    print('\nDone. Results saved to', RESULT_PATH)


if __name__ == '__main__':
    main()