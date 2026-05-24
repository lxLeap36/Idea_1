"""
Tuned sanity search for WL-LMS vs GH-WL-LMS in Experiment C.

Purpose:
    This script performs a wider but still limited hyperparameter search.

    WL-LMS:
        M, sigma, step_size

    GH-WL-LMS:
        M, scale, step_size

    It is intended as a sanity check before moving to 2D-GH-WL-LMS.

Right-click friendly:
    All important settings are defined at the top of this file.
    You can run this file directly without command-line arguments.
"""

import sys
import csv
import time
import copy
from pathlib import Path
from datetime import datetime

import numpy as np


# ============================================================
# Project path
# ============================================================
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# Import your existing Experiment C code
# ============================================================
from configs.exp_c_config import (
    EXP_C_DATASETS,
    ALGO_PARAMS,
    IMD,
    SNAPSHOT,
    SNAPSHOT_EVERY,
    SS_LAST_N,
    RESULT_DIR,
)

import scenarios.imd_echo_burst_scenario as burst_scenario
from scenarios.imd_echo_burst_scenario import run_imd_burst_experiment


# ============================================================
# Manual settings for right-click running
# ============================================================

CASE_NAME = "C2_SINES_BURST"

# For quick screening, use 1 or 2.
# After finding good candidates, rerun top candidates with 5.
MC_TRIALS_DEFAULT = 5

# Keep C2 quick at first.
# Leave as None to use config file default.
OVERRIDE_N_TRAIN = None
OVERRIDE_N_TEST = 200

# Search ranges recommended for current stage
M_LIST = [8, 12, 20, 40]

WL_SIGMAS = [0.2, 0.4, 0.8, 1.5, 3.0, 5.0]
WL_STEPS = [0.0005, 0.001, 0.003, 0.006, 0.01, 0.02]

GH_SCALES = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
GH_STEPS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]

GH_NORMALIZED = True
GH_EPS = 1e-8

SEED = 0

# If True, also run LMS once for reference.
RUN_LMS_BASELINE = True

# Candidate selection rule:
# For balanced candidate:
#   within BEST_SS_TOL_DB of the best steady-state SS_MSE,
#   choose the smallest spike.
BEST_SS_TOL_DB = 2.0


# ============================================================
# Utility functions
# ============================================================

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[csv] no rows to save: {path}")
        return

    headers = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in headers:
                headers.append(key)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] saved: {path}")


def compute_spike_metrics(curves: dict, burst_cfg: dict, ss_mse: dict):
    """
    Compute spike metrics from averaged testing MSE curves.

    Definitions:
        Burst_Peak_MSE_dB:
            maximum MSE inside burst interval.

        Burst_Mean_MSE_dB:
            average MSE inside burst interval.

        Overall_Peak_MSE_dB:
            maximum MSE over the whole curve.

        Spike_Over_SS_dB:
            Burst_Peak_MSE_dB - SS_MSE_dB.
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
            burst_peak = float(np.nanmax(curve[s:e]))
            burst_mean = float(np.nanmean(curve[s:e]))

        overall_peak = float(np.nanmax(curve))
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


def run_one_job(
    dataset_cfg,
    algo_params,
    algo_list,
    n_trials,
    ss_last_n,
):
    """
    Wrapper for your existing experiment runner.
    """

    if hasattr(burst_scenario, "LAST_TRIAL_SIGNALS"):
        burst_scenario.LAST_TRIAL_SIGNALS.clear()
    if hasattr(burst_scenario, "LAST_TRIAL_BURST_INFO"):
        burst_scenario.LAST_TRIAL_BURST_INFO.clear()

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
        n_trials=n_trials,
        snapshot=SNAPSHOT,
        snapshot_every=SNAPSHOT_EVERY,
        ss_last_n=ss_last_n,
        verbose=False,
        algo_list=algo_list,
    )

    spike_rows = compute_spike_metrics(
        curves=avg_curves,
        burst_cfg=dataset_cfg.get("amplitude_burst", None),
        ss_mse=ss_mse,
    )

    spike_by_alg = {r["Algorithm"]: r for r in spike_rows}

    rows = []

    for alg in algo_list:
        sp = spike_by_alg.get(alg, {})

        row = dict(
            Algorithm=alg,
            SS_MSE_dB=safe_float(ss_mse.get(alg, np.nan)),
            AvgTime_s=safe_float(avg_time.get(alg, np.nan)),
            Burst_Start=sp.get("Burst_Start", np.nan),
            Burst_End=sp.get("Burst_End", np.nan),
            Burst_Peak_MSE_dB=sp.get("Burst_Peak_MSE_dB", np.nan),
            Burst_Mean_MSE_dB=sp.get("Burst_Mean_MSE_dB", np.nan),
            Overall_Peak_MSE_dB=sp.get("Overall_Peak_MSE_dB", np.nan),
            Spike_Over_SS_dB=sp.get("Spike_Over_SS_dB", np.nan),
        )
        rows.append(row)

    return rows


def add_common_fields(row, case_name, n_train, n_test, n_trials, status, error, elapsed):
    row["Case"] = case_name
    row["n_train"] = n_train
    row["n_test"] = n_test
    row["MC_TRIALS"] = n_trials
    row["Status"] = status
    row["Error"] = error
    row["JobElapsed_s"] = elapsed
    return row


def rank_rows(rows, algorithm_family, metric, ascending=True):
    valid = [
        r for r in rows
        if r.get("Algorithm_Family") == algorithm_family
        and np.isfinite(safe_float(r.get(metric, np.nan)))
        and r.get("Status") == "ok"
    ]

    return sorted(
        valid,
        key=lambda r: safe_float(r.get(metric, np.nan)),
        reverse=not ascending,
    )


def select_balanced_rows(rows, algorithm_family, best_ss_tol_db=2.0):
    """
    Select balanced candidates.

    Rule:
        1. Find the best SS_MSE_dB, smaller is better.
        2. Keep candidates with SS_MSE_dB <= best_ss + tolerance.
        3. Among them, sort by Spike_Over_SS_dB, smaller is better.
    """
    valid = [
        r for r in rows
        if r.get("Algorithm_Family") == algorithm_family
        and r.get("Status") == "ok"
        and np.isfinite(safe_float(r.get("SS_MSE_dB", np.nan)))
        and np.isfinite(safe_float(r.get("Spike_Over_SS_dB", np.nan)))
    ]

    if not valid:
        return []

    best_ss = min(safe_float(r["SS_MSE_dB"]) for r in valid)
    threshold = best_ss + best_ss_tol_db

    kept = [
        r for r in valid
        if safe_float(r["SS_MSE_dB"]) <= threshold
    ]

    return sorted(
        kept,
        key=lambda r: safe_float(r["Spike_Over_SS_dB"]),
    )


def make_best_summary(rows):
    """
    Create a compact summary table.
    """
    summary = []

    for family in ["WL-LMS", "GH-WL-LMS"]:
        by_ss = rank_rows(rows, family, "SS_MSE_dB", ascending=True)
        by_spike = rank_rows(rows, family, "Spike_Over_SS_dB", ascending=True)
        by_burst_peak = rank_rows(rows, family, "Burst_Peak_MSE_dB", ascending=True)
        by_balanced = select_balanced_rows(rows, family, BEST_SS_TOL_DB)

        if by_ss:
            r = copy.deepcopy(by_ss[0])
            r["Selection_Type"] = "best_by_ss"
            summary.append(r)

        if by_spike:
            r = copy.deepcopy(by_spike[0])
            r["Selection_Type"] = "best_by_spike"
            summary.append(r)

        if by_burst_peak:
            r = copy.deepcopy(by_burst_peak[0])
            r["Selection_Type"] = "best_by_burst_peak"
            summary.append(r)

        if by_balanced:
            r = copy.deepcopy(by_balanced[0])
            r["Selection_Type"] = f"balanced_ss_within_{BEST_SS_TOL_DB}dB_then_min_spike"
            summary.append(r)

    return summary


# ============================================================
# Main search
# ============================================================

def main():
    if CASE_NAME not in EXP_C_DATASETS:
        raise ValueError(
            f"Unknown CASE_NAME={CASE_NAME}. "
            f"Available cases: {list(EXP_C_DATASETS.keys())}"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(RESULT_DIR).parent / "exp_c_tuned_sanity" / CASE_NAME / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg_base = copy.deepcopy(EXP_C_DATASETS[CASE_NAME])

    if OVERRIDE_N_TRAIN is not None:
        dataset_cfg_base["n_train"] = int(OVERRIDE_N_TRAIN)
    if OVERRIDE_N_TEST is not None:
        dataset_cfg_base["n_test"] = int(OVERRIDE_N_TEST)

    n_train = int(dataset_cfg_base.get("n_train"))
    n_test = int(dataset_cfg_base.get("n_test"))

    print("=" * 80)
    print("Experiment C tuned sanity search")
    print("=" * 80)
    print(f"CASE_NAME        = {CASE_NAME}")
    print(f"MC_TRIALS        = {MC_TRIALS_DEFAULT}")
    print(f"n_train          = {n_train}")
    print(f"n_test           = {n_test}")
    print(f"M_LIST           = {M_LIST}")
    print(f"WL_SIGMAS        = {WL_SIGMAS}")
    print(f"WL_STEPS         = {WL_STEPS}")
    print(f"GH_SCALES        = {GH_SCALES}")
    print(f"GH_STEPS         = {GH_STEPS}")
    print(f"out_dir          = {out_dir}")
    print("=" * 80)

    all_rows = []

    total_wl_jobs = len(M_LIST) * len(WL_SIGMAS) * len(WL_STEPS)
    total_gh_jobs = len(M_LIST) * len(GH_SCALES) * len(GH_STEPS)
    total_jobs = total_wl_jobs + total_gh_jobs + (1 if RUN_LMS_BASELINE else 0)

    job_id = 0
    t_all = time.time()

    # ------------------------------------------------------------
    # Optional LMS baseline
    # ------------------------------------------------------------
    if RUN_LMS_BASELINE:
        job_id += 1
        print(f"\n[{job_id}/{total_jobs}] Running LMS baseline")

        dataset_cfg = copy.deepcopy(dataset_cfg_base)
        algo_params = copy.deepcopy(ALGO_PARAMS)

        t0 = time.time()
        try:
            rows = run_one_job(
                dataset_cfg=dataset_cfg,
                algo_params=algo_params,
                algo_list=["LMS"],
                n_trials=MC_TRIALS_DEFAULT,
                ss_last_n=SS_LAST_N,
            )
            elapsed = time.time() - t0

            for r in rows:
                r["Algorithm_Family"] = "LMS"
                r["M"] = np.nan
                r["WL_sigma"] = np.nan
                r["WL_step_size"] = np.nan
                r["GH_scale"] = np.nan
                r["GH_step_size"] = np.nan
                r["GH_normalized"] = np.nan
                add_common_fields(
                    r,
                    CASE_NAME,
                    n_train,
                    n_test,
                    MC_TRIALS_DEFAULT,
                    "ok",
                    "",
                    elapsed,
                )
                all_rows.append(r)

            print(
                f"[ok] LMS SS={rows[0]['SS_MSE_dB']:.3f} dB | "
                f"Spike={rows[0]['Spike_Over_SS_dB']:.3f} dB"
            )

        except Exception as ex:
            elapsed = time.time() - t0
            all_rows.append(
                add_common_fields(
                    dict(
                        Algorithm="LMS",
                        Algorithm_Family="LMS",
                        M=np.nan,
                        WL_sigma=np.nan,
                        WL_step_size=np.nan,
                        GH_scale=np.nan,
                        GH_step_size=np.nan,
                        GH_normalized=np.nan,
                        SS_MSE_dB=np.nan,
                        AvgTime_s=np.nan,
                        Burst_Start=np.nan,
                        Burst_End=np.nan,
                        Burst_Peak_MSE_dB=np.nan,
                        Burst_Mean_MSE_dB=np.nan,
                        Overall_Peak_MSE_dB=np.nan,
                        Spike_Over_SS_dB=np.nan,
                    ),
                    CASE_NAME,
                    n_train,
                    n_test,
                    MC_TRIALS_DEFAULT,
                    "error",
                    str(ex),
                    elapsed,
                )
            )
            print(f"[error] LMS baseline failed: {ex}")

        write_csv(all_rows, out_dir / "tuned_sanity_long.csv")

    # ------------------------------------------------------------
    # WL-LMS tuned search
    # ------------------------------------------------------------
    for M in M_LIST:
        for sigma in WL_SIGMAS:
            for step_size in WL_STEPS:
                job_id += 1

                print(
                    f"\n[{job_id}/{total_jobs}] "
                    f"WL-LMS: M={M}, sigma={sigma}, step_size={step_size}"
                )

                dataset_cfg = copy.deepcopy(dataset_cfg_base)
                algo_params = copy.deepcopy(ALGO_PARAMS)

                algo_params["WLLMS"] = dict(
                    M=int(M),
                    sigma=float(sigma),
                    step_size=float(step_size),
                    seed=SEED,
                )

                t0 = time.time()

                try:
                    rows = run_one_job(
                        dataset_cfg=dataset_cfg,
                        algo_params=algo_params,
                        algo_list=["WL-LMS"],
                        n_trials=MC_TRIALS_DEFAULT,
                        ss_last_n=SS_LAST_N,
                    )
                    elapsed = time.time() - t0

                    for r in rows:
                        r["Algorithm_Family"] = "WL-LMS"
                        r["M"] = int(M)
                        r["WL_sigma"] = float(sigma)
                        r["WL_step_size"] = float(step_size)
                        r["GH_scale"] = np.nan
                        r["GH_step_size"] = np.nan
                        r["GH_normalized"] = np.nan
                        add_common_fields(
                            r,
                            CASE_NAME,
                            n_train,
                            n_test,
                            MC_TRIALS_DEFAULT,
                            "ok",
                            "",
                            elapsed,
                        )
                        all_rows.append(r)

                    print(
                        f"[ok] WL SS={rows[0]['SS_MSE_dB']:.3f} dB | "
                        f"Peak={rows[0]['Burst_Peak_MSE_dB']:.3f} dB | "
                        f"Spike={rows[0]['Spike_Over_SS_dB']:.3f} dB | "
                        f"time={rows[0]['AvgTime_s']:.4f}s"
                    )

                except Exception as ex:
                    elapsed = time.time() - t0

                    all_rows.append(
                        add_common_fields(
                            dict(
                                Algorithm="WL-LMS",
                                Algorithm_Family="WL-LMS",
                                M=int(M),
                                WL_sigma=float(sigma),
                                WL_step_size=float(step_size),
                                GH_scale=np.nan,
                                GH_step_size=np.nan,
                                GH_normalized=np.nan,
                                SS_MSE_dB=np.nan,
                                AvgTime_s=np.nan,
                                Burst_Start=np.nan,
                                Burst_End=np.nan,
                                Burst_Peak_MSE_dB=np.nan,
                                Burst_Mean_MSE_dB=np.nan,
                                Overall_Peak_MSE_dB=np.nan,
                                Spike_Over_SS_dB=np.nan,
                            ),
                            CASE_NAME,
                            n_train,
                            n_test,
                            MC_TRIALS_DEFAULT,
                            "error",
                            str(ex),
                            elapsed,
                        )
                    )

                    print(f"[error] WL-LMS failed: {ex}")

                write_csv(all_rows, out_dir / "tuned_sanity_long.csv")

    # ------------------------------------------------------------
    # GH-WL-LMS tuned search
    # ------------------------------------------------------------
    for M in M_LIST:
        for scale in GH_SCALES:
            for step_size in GH_STEPS:
                job_id += 1

                print(
                    f"\n[{job_id}/{total_jobs}] "
                    f"GH-WL-LMS: M={M}, scale={scale}, step_size={step_size}, "
                    f"normalized={GH_NORMALIZED}"
                )

                dataset_cfg = copy.deepcopy(dataset_cfg_base)
                algo_params = copy.deepcopy(ALGO_PARAMS)

                algo_params["GHWLLMS"] = dict(
                    M=int(M),
                    scale=float(scale),
                    step_size=float(step_size),
                    normalized=bool(GH_NORMALIZED),
                    eps=float(GH_EPS),
                    seed=SEED,
                )

                t0 = time.time()

                try:
                    rows = run_one_job(
                        dataset_cfg=dataset_cfg,
                        algo_params=algo_params,
                        algo_list=["GH-WL-LMS"],
                        n_trials=MC_TRIALS_DEFAULT,
                        ss_last_n=SS_LAST_N,
                    )
                    elapsed = time.time() - t0

                    for r in rows:
                        r["Algorithm_Family"] = "GH-WL-LMS"
                        r["M"] = int(M)
                        r["WL_sigma"] = np.nan
                        r["WL_step_size"] = np.nan
                        r["GH_scale"] = float(scale)
                        r["GH_step_size"] = float(step_size)
                        r["GH_normalized"] = bool(GH_NORMALIZED)
                        add_common_fields(
                            r,
                            CASE_NAME,
                            n_train,
                            n_test,
                            MC_TRIALS_DEFAULT,
                            "ok",
                            "",
                            elapsed,
                        )
                        all_rows.append(r)

                    print(
                        f"[ok] GH SS={rows[0]['SS_MSE_dB']:.3f} dB | "
                        f"Peak={rows[0]['Burst_Peak_MSE_dB']:.3f} dB | "
                        f"Spike={rows[0]['Spike_Over_SS_dB']:.3f} dB | "
                        f"time={rows[0]['AvgTime_s']:.4f}s"
                    )

                except Exception as ex:
                    elapsed = time.time() - t0

                    all_rows.append(
                        add_common_fields(
                            dict(
                                Algorithm="GH-WL-LMS",
                                Algorithm_Family="GH-WL-LMS",
                                M=int(M),
                                WL_sigma=np.nan,
                                WL_step_size=np.nan,
                                GH_scale=float(scale),
                                GH_step_size=float(step_size),
                                GH_normalized=bool(GH_NORMALIZED),
                                SS_MSE_dB=np.nan,
                                AvgTime_s=np.nan,
                                Burst_Start=np.nan,
                                Burst_End=np.nan,
                                Burst_Peak_MSE_dB=np.nan,
                                Burst_Mean_MSE_dB=np.nan,
                                Overall_Peak_MSE_dB=np.nan,
                                Spike_Over_SS_dB=np.nan,
                            ),
                            CASE_NAME,
                            n_train,
                            n_test,
                            MC_TRIALS_DEFAULT,
                            "error",
                            str(ex),
                            elapsed,
                        )
                    )

                    print(f"[error] GH-WL-LMS failed: {ex}")

                write_csv(all_rows, out_dir / "tuned_sanity_long.csv")

    # ------------------------------------------------------------
    # Final summaries
    # ------------------------------------------------------------
    write_csv(all_rows, out_dir / "tuned_sanity_long.csv")

    wl_by_ss = rank_rows(all_rows, "WL-LMS", "SS_MSE_dB", ascending=True)
    wl_by_spike = rank_rows(all_rows, "WL-LMS", "Spike_Over_SS_dB", ascending=True)
    wl_by_peak = rank_rows(all_rows, "WL-LMS", "Burst_Peak_MSE_dB", ascending=True)
    wl_balanced = select_balanced_rows(all_rows, "WL-LMS", BEST_SS_TOL_DB)

    gh_by_ss = rank_rows(all_rows, "GH-WL-LMS", "SS_MSE_dB", ascending=True)
    gh_by_spike = rank_rows(all_rows, "GH-WL-LMS", "Spike_Over_SS_dB", ascending=True)
    gh_by_peak = rank_rows(all_rows, "GH-WL-LMS", "Burst_Peak_MSE_dB", ascending=True)
    gh_balanced = select_balanced_rows(all_rows, "GH-WL-LMS", BEST_SS_TOL_DB)

    write_csv(wl_by_ss, out_dir / "WL_best_by_ss.csv")
    write_csv(wl_by_spike, out_dir / "WL_best_by_spike.csv")
    write_csv(wl_by_peak, out_dir / "WL_best_by_burst_peak.csv")
    write_csv(wl_balanced, out_dir / "WL_balanced_candidates.csv")

    write_csv(gh_by_ss, out_dir / "GH_best_by_ss.csv")
    write_csv(gh_by_spike, out_dir / "GH_best_by_spike.csv")
    write_csv(gh_by_peak, out_dir / "GH_best_by_burst_peak.csv")
    write_csv(gh_balanced, out_dir / "GH_balanced_candidates.csv")

    summary_rows = make_best_summary(all_rows)
    write_csv(summary_rows, out_dir / "tuned_sanity_best_summary.csv")

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Total elapsed: {time.time() - t_all:.2f}s")
    print(f"Results saved to: {out_dir}")
    print("=" * 80)

    if summary_rows:
        print("\nBest summary:")
        for r in summary_rows:
            print(
                f"{r['Selection_Type']:>40s} | "
                f"{r['Algorithm_Family']:>10s} | "
                f"M={r.get('M')} | "
                f"SS={safe_float(r.get('SS_MSE_dB')):.3f} | "
                f"Peak={safe_float(r.get('Burst_Peak_MSE_dB')):.3f} | "
                f"Spike={safe_float(r.get('Spike_Over_SS_dB')):.3f}"
            )


if __name__ == "__main__":
    main()