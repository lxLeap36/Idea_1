"""
Grid search for Experiment C hyperparameters.

This script automatically tests combinations of:
    - GH-WL-LMS M
    - GH-WL-LMS scale
    - GH-WL-LMS step_size

It saves one CSV row per algorithm per parameter combination.

Recommended usage:

    python experiments/run_exp_c_grid.py ^
        --case C2_SINES_BURST ^
        --Ms 6,8,10,12 ^
        --scales 0.6,0.8,1.0,1.2,2.0,3.0 ^
        --steps 0.01,0.02,0.05 ^
        --trials 5

For speech C-3, use a smaller grid first:

    python experiments/run_exp_c_grid.py ^
        --case C3_SPEECH_BURST ^
        --Ms 8,10 ^
        --scales 1.0,1.5,2.0 ^
        --steps 0.02,0.05 ^
        --trials 5
"""

import sys
import csv
import time
import copy
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.exp_c_config import (
    EXP_C_DATASETS,
    ALGO_PARAMS,
    IMD,
    SNAPSHOT,
    SNAPSHOT_EVERY,
    SS_LAST_N,
    RESULT_DIR,
    MC_TRIALS,
)

import scenarios.imd_echo_burst_scenario as burst_scenario
from scenarios.imd_echo_burst_scenario import run_imd_burst_experiment


def parse_float_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [float(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_int_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in str(s).split(",") if x.strip() != ""]


def parse_str_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [x.strip() for x in str(s).split(",") if x.strip() != ""]


def compute_spike_metrics(curves: dict, burst_cfg: dict, ss_mse: dict):
    """
    Compute spike metrics for averaged testing MSE curves.

    Spike_Over_SS_dB = Burst_Peak_MSE_dB - SS_MSE_dB

    This is meaningful when burst_cfg['domain'] == 'train_iter',
    which is the current Experiment C design.
    """
    rows = []

    if burst_cfg is None or not bool(burst_cfg.get("enabled", False)):
        return rows

    if burst_cfg.get("domain", "train_iter") != "train_iter":
        return rows

    start = int(burst_cfg.get("start", 0))
    length = int(burst_cfg.get("length", 0))
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

        rows.append(
            dict(
                Algorithm=name,
                Burst_Start=start,
                Burst_End=end,
                Burst_Peak_MSE_dB=burst_peak,
                Burst_Mean_MSE_dB=burst_mean,
                Overall_Peak_MSE_dB=overall_peak,
                SS_MSE_dB=ss,
                Spike_Over_SS_dB=burst_peak - ss
                if np.isfinite(burst_peak) and np.isfinite(ss)
                else np.nan,
            )
        )

    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"[csv] no rows to save: {path}")
        return

    # Union of all keys, preserving first-row order as much as possible.
    headers = list(rows[0].keys())
    for row in rows:
        for k in row.keys():
            if k not in headers:
                headers.append(k)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[csv] saved: {path}")


def make_pivot_rows(long_rows, target_algorithm="GH-WL-LMS"):
    """
    Make one compact row per hyperparameter combination.

    It keeps GH-WL-LMS metrics as main columns and also stores
    LMS / WL-LMS SS_MSE for reference.
    """
    grouped = {}

    key_fields = [
        "Case",
        "GH_M",
        "WL_M",
        "Match_WL_M",
        "GH_scale",
        "GH_step_size",
        "GH_normalized",
        "n_train",
        "n_test",
        "MC_TRIALS",
    ]

    for row in long_rows:
        key = tuple(row.get(k) for k in key_fields)
        grouped.setdefault(key, []).append(row)

    pivot_rows = []

    for key, rows in grouped.items():
        base = {k: v for k, v in zip(key_fields, key)}

        by_alg = {r.get("Algorithm"): r for r in rows}

        for alg in ["LMS", "WL-LMS", target_algorithm]:
            r = by_alg.get(alg)
            if r is None:
                continue

            prefix = alg.replace("-", "_").replace(" ", "_")

            base[f"{prefix}_SS_MSE_dB"] = r.get("SS_MSE_dB")
            base[f"{prefix}_Time_s"] = r.get("AvgTime_s")
            base[f"{prefix}_Burst_Peak_MSE_dB"] = r.get("Burst_Peak_MSE_dB")
            base[f"{prefix}_Burst_Mean_MSE_dB"] = r.get("Burst_Mean_MSE_dB")
            base[f"{prefix}_Spike_Over_SS_dB"] = r.get("Spike_Over_SS_dB")
            base[f"{prefix}_Overall_Peak_MSE_dB"] = r.get("Overall_Peak_MSE_dB")

        gh = by_alg.get(target_algorithm)
        if gh is not None:
            base["RankMetric_GH_SS_MSE_dB"] = gh.get("SS_MSE_dB")
            base["RankMetric_GH_Spike_Over_SS_dB"] = gh.get("Spike_Over_SS_dB")
            base["RankMetric_GH_Burst_Peak_MSE_dB"] = gh.get("Burst_Peak_MSE_dB")
            base["RankMetric_GH_Burst_Mean_MSE_dB"] = gh.get("Burst_Mean_MSE_dB")

        pivot_rows.append(base)

    return pivot_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--case",
        type=str,
        default="C2_SINES_BURST",
        help="Experiment C case: C1_AR1_BURST, C2_SINES_BURST, or C3_SPEECH_BURST",
    )

    parser.add_argument(
        "--Ms",
        type=str,
        default="6,8,10,12",
        help="Comma-separated GH-WL-LMS M values, e.g. 6,8,10,12",
    )

    parser.add_argument(
        "--scales",
        type=str,
        default="0.8,1.0,1.2,2.0",
        help="Comma-separated GH-WL-LMS scale values",
    )

    parser.add_argument(
        "--steps",
        type=str,
        default="0.01,0.02,0.05",
        help="Comma-separated GH-WL-LMS step_size values",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=MC_TRIALS,
        help="Monte Carlo trials",
    )

    parser.add_argument(
        "--algos",
        type=str,
        default="LMS,WL-LMS,GH-WL-LMS",
        help="Comma-separated algorithm list",
    )

    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Override n_train. Leave unset to use config.",
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Override n_test. Leave unset to use config.",
    )

    parser.add_argument(
        "--gh-normalized",
        type=int,
        default=1,
        help="1 for normalized update, 0 for standard LMS update",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Epsilon for normalized update",
    )

    parser.add_argument(
        "--match-wl-m",
        type=int,
        default=1,
        help=(
            "If 1, set WL-LMS M equal to GH-WL-LMS M in every grid job. "
            "This enables matched-M fair comparison."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Leave unset to use results/exp_c_grid.",
    )

    args = parser.parse_args()

    Ms = parse_int_list(args.Ms)
    scales = parse_float_list(args.scales)
    steps = parse_float_list(args.steps)
    algo_list = parse_str_list(args.algos)

    if args.case not in EXP_C_DATASETS:
        raise ValueError(
            f"Unknown case {args.case}. Available cases: {list(EXP_C_DATASETS.keys())}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.out_dir is None:
        result_path = Path(RESULT_DIR).parent / "exp_c_grid" / args.case / timestamp
    else:
        result_path = Path(args.out_dir)

    result_path.mkdir(parents=True, exist_ok=True)

    print("▶ Experiment C grid search")
    print(f"  case        = {args.case}")
    print(f"  Ms          = {Ms}")
    print(f"  scales      = {scales}")
    print(f"  steps       = {steps}")
    print(f"  trials      = {args.trials}")
    print(f"  algorithms  = {algo_list}")
    print(f"  result_path = {result_path}")

    long_rows = []
    total_jobs = len(Ms) * len(scales) * len(steps)
    job_id = 0

    t_all = time.time()

    for M in Ms:
        for scale in scales:
            for step_size in steps:
                job_id += 1

                print(
                    "\n"
                    + "=" * 80
                    + f"\n[{job_id}/{total_jobs}] "
                    f"GH-WL-LMS: M={M}, scale={scale}, step_size={step_size}\n"
                    + "=" * 80
                )

                dataset_cfg = copy.deepcopy(EXP_C_DATASETS[args.case])
                if args.n_train is not None:
                    dataset_cfg["n_train"] = int(args.n_train)
                if args.n_test is not None:
                    dataset_cfg["n_test"] = int(args.n_test)

                algo_params = copy.deepcopy(ALGO_PARAMS)

                # ------------------------------------------------------------
                # Matched-M option:
                # If enabled, WL-LMS uses the same M as GH-WL-LMS.
                # This makes the comparison fair in terms of feature dimension:
                #
                #     dim(Z_k) = L * M
                #
                # Without this, GH-WL-LMS may use a larger M than WL-LMS,
                # and the performance gain may partly come from having more parameters.
                # ------------------------------------------------------------
                if int(args.match_wl_m) == 1:
                    if "WLLMS" not in algo_params:
                        raise KeyError("ALGO_PARAMS does not contain 'WLLMS'. Cannot match WL-LMS M.")

                    algo_params["WLLMS"]["M"] = int(M)

                # Overwrite GH-WL-LMS settings for this grid job.
                algo_params["GHWLLMS"] = dict(
                    M=int(M),
                    scale=float(scale),
                    step_size=float(step_size),
                    normalized=bool(args.gh_normalized),
                    eps=float(args.eps),
                    seed=0,
                )

                # Clear cached last-trial signals to avoid memory growth in large grids.
                if hasattr(burst_scenario, "LAST_TRIAL_SIGNALS"):
                    burst_scenario.LAST_TRIAL_SIGNALS.clear()
                if hasattr(burst_scenario, "LAST_TRIAL_BURST_INFO"):
                    burst_scenario.LAST_TRIAL_BURST_INFO.clear()

                t0 = time.time()

                try:
                    (
                        avg_curves,
                        ss_mse,
                        avg_time,
                        last_trial_results,
                        last_signals,
                        last_burst_info,
                    ) = run_imd_burst_experiment(
                        dataset_cfg,
                        IMD,
                        algo_params,
                        n_trials=args.trials,
                        snapshot=SNAPSHOT,
                        snapshot_every=SNAPSHOT_EVERY,
                        ss_last_n=SS_LAST_N,
                        verbose=False,
                        algo_list=algo_list,
                    )

                    elapsed = time.time() - t0

                    spike_rows = compute_spike_metrics(
                        curves=avg_curves,
                        burst_cfg=dataset_cfg.get("amplitude_burst", None),
                        ss_mse=ss_mse,
                    )
                    spike_by_alg = {r["Algorithm"]: r for r in spike_rows}

                    for alg in algo_list:
                        sp = spike_by_alg.get(alg, {})

                        row = dict(
                            Case=args.case,
                            GH_M=int(M),
                            WL_M=int(algo_params["WLLMS"].get("M", -1)),
                            Match_WL_M=bool(args.match_wl_m),
                            GH_scale=float(scale),
                            GH_step_size=float(step_size),
                            GH_normalized=bool(args.gh_normalized),
                            n_train=int(dataset_cfg.get("n_train")),
                            n_test=int(dataset_cfg.get("n_test")),
                            MC_TRIALS=int(args.trials),
                            Algorithm=alg,
                            SS_MSE_dB=float(ss_mse.get(alg, np.nan)),
                            AvgTime_s=float(avg_time.get(alg, np.nan)),
                            Burst_Start=sp.get("Burst_Start", np.nan),
                            Burst_End=sp.get("Burst_End", np.nan),
                            Burst_Peak_MSE_dB=sp.get("Burst_Peak_MSE_dB", np.nan),
                            Burst_Mean_MSE_dB=sp.get("Burst_Mean_MSE_dB", np.nan),
                            Overall_Peak_MSE_dB=sp.get("Overall_Peak_MSE_dB", np.nan),
                            Spike_Over_SS_dB=sp.get("Spike_Over_SS_dB", np.nan),
                            JobElapsed_s=float(elapsed),
                            Status="ok",
                            Error="",
                        )

                        long_rows.append(row)

                    print(
                        f"[ok] elapsed={elapsed:.2f}s | "
                        f"GH SS={ss_mse.get('GH-WL-LMS', np.nan):.3f} dB | "
                        f"GH spike={spike_by_alg.get('GH-WL-LMS', {}).get('Spike_Over_SS_dB', np.nan):.3f} dB | "
                        f"GH peak={spike_by_alg.get('GH-WL-LMS', {}).get('Burst_Peak_MSE_dB', np.nan):.3f} dB"
                    )

                except Exception as ex:
                    elapsed = time.time() - t0

                    row = dict(
                        Case=args.case,
                        GH_M=int(M),
                        WL_M=int(algo_params.get("WLLMS", {}).get("M", -1)),
                        Match_WL_M=bool(args.match_wl_m),
                        GH_scale=float(scale),
                        GH_step_size=float(step_size),
                        GH_normalized=bool(args.gh_normalized),
                        n_train=int(dataset_cfg.get("n_train")),
                        n_test=int(dataset_cfg.get("n_test")),
                        MC_TRIALS=int(args.trials),
                        Algorithm="ERROR",
                        SS_MSE_dB=np.nan,
                        AvgTime_s=np.nan,
                        Burst_Start=np.nan,
                        Burst_End=np.nan,
                        Burst_Peak_MSE_dB=np.nan,
                        Burst_Mean_MSE_dB=np.nan,
                        Overall_Peak_MSE_dB=np.nan,
                        Spike_Over_SS_dB=np.nan,
                        JobElapsed_s=float(elapsed),
                        Status="error",
                        Error=str(ex),
                    )
                    long_rows.append(row)

                    print(f"[error] {ex}")

                # Save after every job, so partial results are preserved.
                write_csv(long_rows, result_path / "exp_c_grid_long.csv")

                pivot_rows = make_pivot_rows(long_rows, target_algorithm="GH-WL-LMS")
                write_csv(pivot_rows, result_path / "exp_c_grid_pivot.csv")

    # Final sorted summaries.
    pivot_rows = make_pivot_rows(long_rows, target_algorithm="GH-WL-LMS")

    # Sort by GH steady-state MSE: more negative is better.
    by_ss = sorted(
        pivot_rows,
        key=lambda r: float(r.get("RankMetric_GH_SS_MSE_dB", np.inf)),
    )
    write_csv(by_ss, result_path / "exp_c_grid_best_by_ss.csv")

    # Sort by GH spike over SS: smaller is better.
    by_spike = sorted(
        pivot_rows,
        key=lambda r: float(r.get("RankMetric_GH_Spike_Over_SS_dB", np.inf)),
    )
    write_csv(by_spike, result_path / "exp_c_grid_best_by_spike.csv")

    # Sort by GH burst peak: more negative is better.
    by_peak = sorted(
        pivot_rows,
        key=lambda r: float(r.get("RankMetric_GH_Burst_Peak_MSE_dB", np.inf)),
    )
    write_csv(by_peak, result_path / "exp_c_grid_best_by_burst_peak.csv")

    print("\nDone.")
    print(f"Total elapsed: {time.time() - t_all:.2f}s")
    print(f"Results saved to: {result_path}")


if __name__ == "__main__":
    main()