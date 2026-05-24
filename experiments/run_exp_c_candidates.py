"""
Run selected WL-LMS and GH-WL-LMS candidate parameters on C1/C3.

Purpose:
    After C2 tuned sanity search, we do NOT search parameters again on C1/C3.
    Instead, we fix representative candidate parameters selected from C2 and
    test whether they generalize to AR1 input and speech input.

Right-click friendly:
    All settings are defined at the top of this file.
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
# Import existing Experiment C code
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

# Main purpose: test whether C2-selected candidates generalize to C1 and C3.
# You can also include C2 here for reference, but it will cost more time.
CASE_LIST = [
    "C1_AR1_BURST",
    "C3_SPEECH_BURST",
    # "C2_SINES_BURST",
]

MC_TRIALS_DEFAULT = 5
# ============================================================
# Evaluation mode
# ============================================================
# train_online:
#     no snapshot; use online training error e(k)^2 as the curve.
#     fastest and suitable for candidate generalization testing.
#
# test_snapshot:
#     original expensive snapshot-based test curve.
EVAL_CURVE_MODE = "train_online"

# Leave as None to use each case's config value.
OVERRIDE_N_TRAIN = None
OVERRIDE_N_TEST = None

# If C3 speech is slow, you can temporarily set:
#   OVERRIDE_N_TEST = 2000
# But for final verification, keep None.
SEED = 0


# ============================================================
# Candidate parameters selected from C2 tuned sanity search
# ============================================================

WL_CANDIDATES = [
    dict(
        Candidate_Name="WL_1_best_ss",
        Description="WL-LMS extreme steady-state candidate from C2",
        params=dict(M=40, sigma=0.4, step_size=0.0005, seed=SEED),
    ),
    dict(
        Candidate_Name="WL_2_best_spike",
        Description="WL-LMS robust/spike candidate from C2",
        params=dict(M=8, sigma=0.2, step_size=0.0005, seed=SEED),
    ),
    dict(
        Candidate_Name="WL_3_balanced_ss31",
        Description="WL-LMS balanced candidate around SS=-31 dB from C2",
        params=dict(M=40, sigma=0.2, step_size=0.0005, seed=SEED),
    ),
]

GH_CANDIDATES = [
    dict(
        Candidate_Name="GH_1_best_ss",
        Description="GH-WL-LMS extreme steady-state candidate from C2",
        params=dict(M=40, scale=0.6, step_size=0.2, normalized=True, eps=1e-8, seed=SEED),
    ),
    dict(
        Candidate_Name="GH_2_best_spike",
        Description="GH-WL-LMS robust/spike candidate from C2",
        params=dict(M=40, scale=0.6, step_size=0.005, normalized=True, eps=1e-8, seed=SEED),
    ),
    dict(
        Candidate_Name="GH_3_balanced_ss31",
        Description="GH-WL-LMS balanced candidate around SS=-31 dB from C2",
        params=dict(M=40, scale=0.8, step_size=0.005, normalized=True, eps=1e-8, seed=SEED),
    ),
    dict(
        Candidate_Name="GH_4_balanced_ss33",
        Description="GH-WL-LMS stronger balanced candidate around SS=-33 dB from C2",
        params=dict(M=40, scale=0.6, step_size=0.01, normalized=True, eps=1e-8, seed=SEED),
    ),
    dict(
        Candidate_Name="GH_5_light_robust",
        Description="GH-WL-LMS lighter robust candidate, lower cost than M=40",
        params=dict(M=20, scale=0.6, step_size=0.005, normalized=True, eps=1e-8, seed=SEED),
    ),
]

RUN_LMS_BASELINE = True


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
        for k in row.keys():
            if k not in headers:
                headers.append(k)

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] saved: {path}")


def compute_spike_metrics(curves: dict, burst_cfg: dict, ss_mse: dict):
    """
    Compute burst-related metrics from averaged testing MSE curves.
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


def clear_last_trial_cache():
    if hasattr(burst_scenario, "LAST_TRIAL_SIGNALS"):
        burst_scenario.LAST_TRIAL_SIGNALS.clear()

    if hasattr(burst_scenario, "LAST_TRIAL_BURST_INFO"):
        burst_scenario.LAST_TRIAL_BURST_INFO.clear()


def make_dataset_cfg(case_name):
    if case_name not in EXP_C_DATASETS:
        raise ValueError(
            f"Unknown case: {case_name}. "
            f"Available cases: {list(EXP_C_DATASETS.keys())}"
        )

    dataset_cfg = copy.deepcopy(EXP_C_DATASETS[case_name])

    if OVERRIDE_N_TRAIN is not None:
        dataset_cfg["n_train"] = int(OVERRIDE_N_TRAIN)

    if OVERRIDE_N_TEST is not None:
        dataset_cfg["n_test"] = int(OVERRIDE_N_TEST)

    return dataset_cfg


def run_single_candidate(case_name, algorithm_family, candidate_name, description, params):
    """
    Run one candidate on one Experiment C case.
    """
    dataset_cfg = make_dataset_cfg(case_name)
    algo_params = copy.deepcopy(ALGO_PARAMS)

    if algorithm_family == "LMS":
        algo_list = ["LMS"]

    elif algorithm_family == "WL-LMS":
        algo_params["WLLMS"] = copy.deepcopy(params)
        algo_list = ["WL-LMS"]

    elif algorithm_family == "GH-WL-LMS":
        algo_params["GHWLLMS"] = copy.deepcopy(params)
        algo_list = ["GH-WL-LMS"]

    else:
        raise ValueError(f"Unsupported algorithm_family: {algorithm_family}")

    clear_last_trial_cache()

    t0 = time.time()

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
        n_trials=MC_TRIALS_DEFAULT,

        # Important:
        # For candidate generalization testing, do not use expensive snapshots.
        snapshot=False,
        snapshot_every=1,

        ss_last_n=SS_LAST_N,
        verbose=False,
        algo_list=algo_list,
        curve_mode=EVAL_CURVE_MODE,
    )

    elapsed = time.time() - t0

    spike_rows = compute_spike_metrics(
        curves=avg_curves,
        burst_cfg=dataset_cfg.get("amplitude_burst", None),
        ss_mse=ss_mse,
    )

    spike_by_alg = {r["Algorithm"]: r for r in spike_rows}

    alg = algo_list[0]
    sp = spike_by_alg.get(alg, {})
    final_test_mse_mc_db = np.nan
    if last_trial_results is not None and alg in last_trial_results:
        final_test_mse_mc_db = last_trial_results[alg].get(
            "final_test_mse_mc_db",
            np.nan
        )

    row = dict(
        Case=case_name,
        Input_Type=dataset_cfg.get("input_type", ""),
        Evaluation_Mode=EVAL_CURVE_MODE,
        n_train=int(dataset_cfg.get("n_train")),
        n_test=int(dataset_cfg.get("n_test")),
        MC_TRIALS=int(MC_TRIALS_DEFAULT),

        Candidate_Name=candidate_name,
        Description=description,
        Algorithm_Family=algorithm_family,
        Algorithm=alg,

        M=params.get("M", np.nan) if isinstance(params, dict) else np.nan,

        WL_sigma=params.get("sigma", np.nan) if algorithm_family == "WL-LMS" else np.nan,
        WL_step_size=params.get("step_size", np.nan) if algorithm_family == "WL-LMS" else np.nan,

        GH_scale=params.get("scale", np.nan) if algorithm_family == "GH-WL-LMS" else np.nan,
        GH_step_size=params.get("step_size", np.nan) if algorithm_family == "GH-WL-LMS" else np.nan,
        GH_normalized=params.get("normalized", np.nan) if algorithm_family == "GH-WL-LMS" else np.nan,

        # In train_online mode, SS_MSE_dB means the tail average of
        # online training error curve, not snapshot test MSE.
        SS_MSE_dB=safe_float(ss_mse.get(alg, np.nan)),
        Online_Train_SS_MSE_dB=safe_float(ss_mse.get(alg, np.nan)),

        # Final test MSE is evaluated once after training and averaged over MC trials.
        Final_Test_MSE_dB=safe_float(final_test_mse_mc_db),

        AvgTime_s=safe_float(avg_time.get(alg, np.nan)),

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

    return row


def make_pivot_rows(long_rows):
    """
    Make a compact pivot-style table:
        one row per Case,
        candidate metrics become columns.
    """
    cases = sorted(set(r["Case"] for r in long_rows))
    pivot_rows = []

    metric_names = [
        "SS_MSE_dB",
        "Burst_Peak_MSE_dB",
        "Burst_Mean_MSE_dB",
        "Spike_Over_SS_dB",
        "Overall_Peak_MSE_dB",
        "AvgTime_s",
    ]

    for case in cases:
        rows_case = [r for r in long_rows if r["Case"] == case and r["Status"] == "ok"]

        base = dict(Case=case)

        if rows_case:
            base["Input_Type"] = rows_case[0].get("Input_Type", "")
            base["n_train"] = rows_case[0].get("n_train", "")
            base["n_test"] = rows_case[0].get("n_test", "")
            base["MC_TRIALS"] = rows_case[0].get("MC_TRIALS", "")

        for r in rows_case:
            cname = r["Candidate_Name"]
            for m in metric_names:
                base[f"{cname}_{m}"] = r.get(m, np.nan)

        pivot_rows.append(base)

    return pivot_rows


def make_cross_case_summary(long_rows):
    """
    Summarize each candidate across all cases:
        average and worst-case metrics.
    """
    candidates = sorted(set(r["Candidate_Name"] for r in long_rows if r["Status"] == "ok"))
    summary = []

    for cname in candidates:
        rows = [r for r in long_rows if r["Candidate_Name"] == cname and r["Status"] == "ok"]

        if not rows:
            continue

        ss_vals = np.array([safe_float(r["SS_MSE_dB"]) for r in rows], dtype=float)
        peak_vals = np.array([safe_float(r["Burst_Peak_MSE_dB"]) for r in rows], dtype=float)
        mean_vals = np.array([safe_float(r["Burst_Mean_MSE_dB"]) for r in rows], dtype=float)
        spike_vals = np.array([safe_float(r["Spike_Over_SS_dB"]) for r in rows], dtype=float)
        time_vals = np.array([safe_float(r["AvgTime_s"]) for r in rows], dtype=float)

        first = rows[0]

        # For MSE in dB, larger is worse. Therefore worst-case = max.
        row = dict(
            Candidate_Name=cname,
            Algorithm_Family=first.get("Algorithm_Family", ""),
            Description=first.get("Description", ""),
            M=first.get("M", np.nan),

            WL_sigma=first.get("WL_sigma", np.nan),
            WL_step_size=first.get("WL_step_size", np.nan),
            GH_scale=first.get("GH_scale", np.nan),
            GH_step_size=first.get("GH_step_size", np.nan),
            GH_normalized=first.get("GH_normalized", np.nan),

            Num_Cases=len(rows),

            Avg_SS_MSE_dB=float(np.nanmean(ss_vals)),
            Worst_SS_MSE_dB=float(np.nanmax(ss_vals)),

            Avg_Burst_Peak_MSE_dB=float(np.nanmean(peak_vals)),
            Worst_Burst_Peak_MSE_dB=float(np.nanmax(peak_vals)),

            Avg_Burst_Mean_MSE_dB=float(np.nanmean(mean_vals)),
            Worst_Burst_Mean_MSE_dB=float(np.nanmax(mean_vals)),

            Avg_Spike_Over_SS_dB=float(np.nanmean(spike_vals)),
            Worst_Spike_Over_SS_dB=float(np.nanmax(spike_vals)),

            AvgTime_s=float(np.nanmean(time_vals)),
        )

        summary.append(row)

    return summary


def sort_summary_rows(summary_rows):
    """
    Sort by a practical balanced criterion:
        1. lower Worst_Burst_Peak_MSE_dB is better
        2. lower Worst_Spike_Over_SS_dB is better
        3. lower Worst_SS_MSE_dB is better

    Since all metrics are in dB and lower is better,
    ascending order is used.
    """
    return sorted(
        summary_rows,
        key=lambda r: (
            safe_float(r.get("Worst_Burst_Peak_MSE_dB", np.inf)),
            safe_float(r.get("Worst_Spike_Over_SS_dB", np.inf)),
            safe_float(r.get("Worst_SS_MSE_dB", np.inf)),
        ),
    )


# ============================================================
# Main
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(RESULT_DIR).parent / "exp_c_candidates" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Experiment C candidate generalization test")
    print("=" * 80)
    print(f"CASE_LIST         = {CASE_LIST}")
    print(f"MC_TRIALS_DEFAULT = {MC_TRIALS_DEFAULT}")
    print(f"OVERRIDE_N_TRAIN  = {OVERRIDE_N_TRAIN}")
    print(f"OVERRIDE_N_TEST   = {OVERRIDE_N_TEST}")
    print(f"out_dir           = {out_dir}")
    print("=" * 80)

    all_jobs = []

    if RUN_LMS_BASELINE:
        for case_name in CASE_LIST:
            all_jobs.append(
                dict(
                    case_name=case_name,
                    algorithm_family="LMS",
                    candidate_name="LMS_baseline",
                    description="Linear LMS baseline",
                    params=dict(),
                )
            )

    for case_name in CASE_LIST:
        for cand in WL_CANDIDATES:
            all_jobs.append(
                dict(
                    case_name=case_name,
                    algorithm_family="WL-LMS",
                    candidate_name=cand["Candidate_Name"],
                    description=cand["Description"],
                    params=cand["params"],
                )
            )

        for cand in GH_CANDIDATES:
            all_jobs.append(
                dict(
                    case_name=case_name,
                    algorithm_family="GH-WL-LMS",
                    candidate_name=cand["Candidate_Name"],
                    description=cand["Description"],
                    params=cand["params"],
                )
            )

    long_rows = []
    t_all = time.time()

    for job_id, job in enumerate(all_jobs, start=1):
        print(
            "\n"
            + "-" * 80
            + f"\n[{job_id}/{len(all_jobs)}] "
            f"{job['case_name']} | {job['candidate_name']} | {job['algorithm_family']}\n"
            + "-" * 80
        )

        try:
            row = run_single_candidate(
                case_name=job["case_name"],
                algorithm_family=job["algorithm_family"],
                candidate_name=job["candidate_name"],
                description=job["description"],
                params=job["params"],
            )

            print(
                f"[ok] SS={row['SS_MSE_dB']:.3f} dB | "
                f"Peak={row['Burst_Peak_MSE_dB']:.3f} dB | "
                f"Mean={row['Burst_Mean_MSE_dB']:.3f} dB | "
                f"Spike={row['Spike_Over_SS_dB']:.3f} dB | "
                f"Time={row['AvgTime_s']:.4f}s"
            )

        except Exception as ex:
            row = dict(
                Case=job["case_name"],
                Candidate_Name=job["candidate_name"],
                Description=job["description"],
                Algorithm_Family=job["algorithm_family"],
                Algorithm=job["algorithm_family"],
                Status="error",
                Error=str(ex),
                SS_MSE_dB=np.nan,
                AvgTime_s=np.nan,
                Burst_Start=np.nan,
                Burst_End=np.nan,
                Burst_Peak_MSE_dB=np.nan,
                Burst_Mean_MSE_dB=np.nan,
                Overall_Peak_MSE_dB=np.nan,
                Spike_Over_SS_dB=np.nan,
                JobElapsed_s=np.nan,
            )

            print(f"[error] {ex}")

        long_rows.append(row)

        # Save after every job to avoid losing partial results.
        write_csv(long_rows, out_dir / "candidate_generalization_long.csv")

    pivot_rows = make_pivot_rows(long_rows)
    write_csv(pivot_rows, out_dir / "candidate_generalization_pivot.csv")

    summary_rows = make_cross_case_summary(long_rows)
    write_csv(summary_rows, out_dir / "candidate_generalization_summary.csv")

    sorted_summary = sort_summary_rows(summary_rows)
    write_csv(sorted_summary, out_dir / "candidate_generalization_summary_sorted.csv")

    print("\n" + "=" * 80)
    print("Done.")
    print(f"Total elapsed: {time.time() - t_all:.2f}s")
    print(f"Results saved to: {out_dir}")
    print("=" * 80)

    print("\nTop candidates by cross-case robust criterion:")
    for r in sorted_summary[:10]:
        print(
            f"{r['Candidate_Name']:>22s} | "
            f"{r['Algorithm_Family']:>10s} | "
            f"AvgSS={r['Avg_SS_MSE_dB']:.3f} | "
            f"WorstSS={r['Worst_SS_MSE_dB']:.3f} | "
            f"WorstPeak={r['Worst_Burst_Peak_MSE_dB']:.3f} | "
            f"WorstSpike={r['Worst_Spike_Over_SS_dB']:.3f} | "
            f"AvgTime={r['AvgTime_s']:.4f}s"
        )


if __name__ == "__main__":
    main()