"""
实验 D：
AEC Challenge 合成数据集上的远端单讲非线性回声建模实验。

目标：
    比较 LMS、WL-LMS、GH-WL-LMS 在真实语音 AEC 合成数据上的效果。

第一版实验设置：
    x(n) = farend_speech
    d(n) = echo_signal

注意：
    本实验使用在线训练误差绘制学习曲线，不使用快照评估法。
"""

import sys
import csv
import time
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Ensure Chinese glyphs render on systems without DejaVu Sans CJK:
# prefer common Windows fonts and fall back to SimHei / DejaVu Sans
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


# ============================================================
# 项目路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# 导入配置与模块
# ============================================================

from configs.exp_d_config import (
    META_CSV,
    FAREND_DIR,
    ECHO_DIR,
    REQUIRE_FAREND_NONLINEAR,
    REQUIRE_FAREND_NOISY,
    REQUIRE_NEAREND_NOISY,
    SPLIT,
    TARGET_FS,
    SEGMENT_SECONDS,
    SEGMENT_MODE,
    SKIP_SHORT_CLIPS,
    REMOVE_DC,
    PEAK_NORMALIZE,
    PEAK_EPS,
    FILTER_ORDERS,
    MC_TRIALS,
    SEED,
    CURVE_WINDOW,
    SS_LAST_RATIO,
    ALGO_LIST,
    ALGO_PARAMS,
    PLOT,
    RESULT_DIR,
)

from datasets.aec_synthetic import (
    load_meta,
    filter_meta_for_exp_d,
    load_exp_d_pair,
    build_input_vector_stream,
)

from algorithms import LMS, WLLMS, GHWLLMS, GHWLLMSFast


# ============================================================
# 工具函数
# ============================================================

def safe_db(x, floor=1e-20):
    """将线性能量转换为 dB。"""
    x = max(float(x), float(floor))
    return 10.0 * np.log10(x)


def moving_average(x, window: int):
    """简单滑动平均，用于平滑绘图。"""
    x = np.asarray(x, dtype=float)

    if window is None or int(window) <= 1:
        return x

    window = int(window)
    if len(x) < window:
        return x

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def write_csv(rows, path: Path):
    """保存字典列表为 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"[csv] 无数据可保存：{path}")
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

    print(f"[csv] 已保存：{path}")


def save_curves_csv(curves: dict, path: Path):
    """
    保存平均学习曲线。

    curves:
        {
            "LMS_MSE_dB": array,
            "LMS_ERLE_dB": array,
            ...
        }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = list(curves.keys())
    n = max(len(v) for v in curves.values())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["window_index"] + names)

        for i in range(n):
            row = [i]
            for name in names:
                v = curves[name]
                if i < len(v):
                    row.append(f"{float(v[i]):.6f}")
                else:
                    row.append("")
            writer.writerow(row)

    print(f"[csv] 已保存：{path}")


def build_algorithm(name: str, filter_order: int, params: dict):
    """根据算法名称构造算法实例。"""
    if name == "LMS":
        return LMS(filter_order, **params["LMS"])

    if name == "WL-LMS":
        return WLLMS(filter_order, **params["WLLMS"])

    if name == "GH-WL-LMS":
        return GHWLLMS(filter_order, **params["GHWLLMS"])

    if name == "GH-WL-LMS-Fast":
        return GHWLLMSFast(filter_order, **params["GHWLLMSFast"])

    raise ValueError(f"未知算法：{name}")


def reset_algorithm(algo, seed: int):
    """重置算法状态，并兼容已有算法的 reset 接口。"""
    if hasattr(algo, "seed"):
        try:
            algo.seed = int(seed)
        except Exception:
            pass

    try:
        algo.reset(reseed_centers=True, seed=seed)
    except TypeError:
        try:
            algo.reset(reseed_features=True, seed=seed)
        except TypeError:
            try:
                algo.reset(seed=seed)
            except TypeError:
                algo.reset()


def run_one_algorithm_online(
    algo,
    x: np.ndarray,
    d: np.ndarray,
    filter_order: int,
    curve_window: int,
):
    """
    对单个算法进行在线训练，并返回窗口级 MSE 与 ERLE 曲线。

    这里不使用快照法。
    每个窗口统计：
        residual_mse = mean(e^2)
        erle = 10log10(mean(d^2) / mean(e^2))
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = min(len(x), len(d))
    x = x[:n]
    d = d[:n]

    mse_curve = []
    erle_curve = []

    e2_sum = 0.0
    d2_sum = 0.0
    count = 0

    t0 = time.perf_counter()

    for k, x_vec in enumerate(build_input_vector_stream(x, filter_order)):
        e = algo.update(x_vec, float(d[k]))

        e2_sum += float(e * e)
        d2_sum += float(d[k] * d[k])
        count += 1

        if count >= int(curve_window):
            mse_lin = e2_sum / max(count, 1)
            d_pow = d2_sum / max(count, 1)

            mse_curve.append(safe_db(mse_lin))
            erle_curve.append(safe_db(d_pow / max(mse_lin, 1e-20)))

            e2_sum = 0.0
            d2_sum = 0.0
            count = 0

    # 处理最后不足一个窗口的部分。
    if count > 0:
        mse_lin = e2_sum / max(count, 1)
        d_pow = d2_sum / max(count, 1)

        mse_curve.append(safe_db(mse_lin))
        erle_curve.append(safe_db(d_pow / max(mse_lin, 1e-20)))

    elapsed = time.perf_counter() - t0

    mse_curve = np.asarray(mse_curve, dtype=np.float64)
    erle_curve = np.asarray(erle_curve, dtype=np.float64)

    ss_n = max(1, int(np.ceil(len(mse_curve) * float(SS_LAST_RATIO))))

    final_mse_db = float(np.mean(mse_curve[-ss_n:]))
    final_erle_db = float(np.mean(erle_curve[-ss_n:]))

    return dict(
        mse_curve=mse_curve,
        erle_curve=erle_curve,
        final_mse_db=final_mse_db,
        final_erle_db=final_erle_db,
        time_s=float(elapsed),
    )


def sample_trials(df, mc_trials: int, seed: int):
    """
    从筛选后的 meta 表中抽取 Monte Carlo 样本。

    每个 trial 对应一个不同 fileid。
    """
    if len(df) == 0:
        raise RuntimeError("筛选后没有可用样本，请检查 meta.csv 和筛选条件。")

    rng = np.random.default_rng(int(seed))
    replace = len(df) < int(mc_trials)

    idx = rng.choice(len(df), size=int(mc_trials), replace=replace)
    return df.iloc[idx].reset_index(drop=True)


def average_curves(curve_list):
    """
    对多条曲线求平均和标准差。

    若长度略有不同，则截断到最短长度。
    """
    if len(curve_list) == 0:
        return np.array([]), np.array([])

    min_len = min(len(c) for c in curve_list)
    arr = np.stack([np.asarray(c[:min_len], dtype=float) for c in curve_list], axis=0)

    return np.mean(arr, axis=0), np.std(arr, axis=0)


def plot_mean_curve_with_std(
    mean_dict: dict,
    std_dict: dict,
    title: str,
    ylabel: str,
    save_path: Path,
    smooth_window: int = 1,
    y_lim=None,
):
    """绘制平均曲线和标准差阴影。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for name, mean_curve in mean_dict.items():
        mean_curve = np.asarray(mean_curve, dtype=float)
        std_curve = np.asarray(std_dict[name], dtype=float)

        mean_s = moving_average(mean_curve, smooth_window)
        std_s = moving_average(std_curve, smooth_window)

        xs = np.arange(len(mean_s))

        ax.plot(xs, mean_s, label=name)
        ax.fill_between(xs, mean_s - std_s, mean_s + std_s, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel(f"窗口索引，每个窗口 {CURVE_WINDOW} 个样本")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    if y_lim is not None:
        ax.set_ylim(y_lim)

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[plot] 已保存：{save_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(RESULT_DIR) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("实验 D：AEC Challenge 合成数据集远端单讲非线性回声建模")
    print("=" * 80)
    print(f"META_CSV          = {META_CSV}")
    print(f"FAREND_DIR        = {FAREND_DIR}")
    print(f"ECHO_DIR          = {ECHO_DIR}")
    print(f"TARGET_FS         = {TARGET_FS}")
    print(f"SEGMENT_SECONDS   = {SEGMENT_SECONDS}")
    print(f"FILTER_ORDERS     = {FILTER_ORDERS}")
    print(f"MC_TRIALS         = {MC_TRIALS}")
    print(f"ALGO_LIST         = {ALGO_LIST}")
    print(f"RESULT_DIR        = {out_dir}")
    print("=" * 80)

    df = load_meta(META_CSV)

    df_sel = filter_meta_for_exp_d(
        df,
        require_farend_nonlinear=REQUIRE_FAREND_NONLINEAR,
        require_farend_noisy=REQUIRE_FAREND_NOISY,
        require_nearend_noisy=REQUIRE_NEAREND_NOISY,
        split=SPLIT,
    )

    print(f"[meta] 原始样本数：{len(df)}")
    print(f"[meta] 筛选后样本数：{len(df_sel)}")

    if len(df_sel) == 0:
        raise RuntimeError("筛选后没有样本，请检查筛选条件。")

    trial_rows_all = []
    summary_rows_all = []

    for filter_order in FILTER_ORDERS:
        print("\n" + "=" * 80)
        print(f"开始 filter_order = {filter_order}")
        print("=" * 80)

        trials = sample_trials(df_sel, MC_TRIALS, seed=SEED + int(filter_order))

        # 保存本次实际使用的 fileid，方便复现实验。
        selected_rows = []
        for _, r in trials.iterrows():
            selected_rows.append({
                "fileid": int(r["fileid"]),
                "split": r.get("split", ""),
                "ser": r.get("ser", ""),
                "is_farend_nonlinear": int(r["is_farend_nonlinear"]),
                "is_farend_noisy": int(r["is_farend_noisy"]),
                "is_nearend_noisy": int(r["is_nearend_noisy"]),
            })

        write_csv(
            selected_rows,
            out_dir / f"exp_d_selected_fileids_p{filter_order}.csv",
        )

        curves_mse_by_algo = {name: [] for name in ALGO_LIST}
        curves_erle_by_algo = {name: [] for name in ALGO_LIST}
        final_mse_by_algo = {name: [] for name in ALGO_LIST}
        final_erle_by_algo = {name: [] for name in ALGO_LIST}
        time_by_algo = {name: [] for name in ALGO_LIST}

        for trial_id, row in trials.iterrows():
            fileid = int(row["fileid"])

            print("\n" + "-" * 80)
            print(f"[trial {trial_id + 1}/{len(trials)}] fileid={fileid}")
            print("-" * 80)

            try:
                pair = load_exp_d_pair(
                    row=row,
                    farend_dir=FAREND_DIR,
                    echo_dir=ECHO_DIR,
                    target_fs=TARGET_FS,
                    segment_seconds=SEGMENT_SECONDS,
                    segment_mode=SEGMENT_MODE,
                    remove_dc_flag=REMOVE_DC,
                    peak_normalize=PEAK_NORMALIZE,
                    peak_eps=PEAK_EPS,
                )
            except Exception as ex:
                if SKIP_SHORT_CLIPS:
                    print(f"[skip] fileid={fileid}, 原因：{ex}")
                    continue
                raise

            x = pair["x"]
            d = pair["d"]

            print(
                f"  片段长度：{len(x)} samples, "
                f"fs={pair['fs']}, "
                f"start={pair['segment_start_sample']}"
            )

            for algo_name in ALGO_LIST:
                algo = build_algorithm(
                    name=algo_name,
                    filter_order=filter_order,
                    params=ALGO_PARAMS,
                )

                reset_algorithm(algo, seed=SEED + trial_id)

                result = run_one_algorithm_online(
                    algo=algo,
                    x=x,
                    d=d,
                    filter_order=filter_order,
                    curve_window=CURVE_WINDOW,
                )

                curves_mse_by_algo[algo_name].append(result["mse_curve"])
                curves_erle_by_algo[algo_name].append(result["erle_curve"])
                final_mse_by_algo[algo_name].append(result["final_mse_db"])
                final_erle_by_algo[algo_name].append(result["final_erle_db"])
                time_by_algo[algo_name].append(result["time_s"])

                trial_row = dict(
                    Filter_Order=int(filter_order),
                    Trial=int(trial_id),
                    FileID=int(fileid),
                    Algorithm=algo_name,
                    Final_MSE_dB=float(result["final_mse_db"]),
                    Final_ERLE_dB=float(result["final_erle_db"]),
                    Time_s=float(result["time_s"]),
                    Segment_Start_Sample=int(pair["segment_start_sample"]),
                    Segment_Samples=int(pair["segment_samples"]),
                    Farend_Path=pair["farend_path"],
                    Echo_Path=pair["echo_path"],
                )
                trial_rows_all.append(trial_row)

                print(
                    f"  {algo_name:10s} | "
                    f"MSE={result['final_mse_db']:.3f} dB | "
                    f"ERLE={result['final_erle_db']:.3f} dB | "
                    f"time={result['time_s']:.3f}s"
                )

            # 每个 trial 后保存一次，避免中途出错丢失结果。
            write_csv(
                trial_rows_all,
                out_dir / "exp_d_trial_results_long.csv",
            )

        # 统计当前 filter_order 的平均曲线。
        mean_mse = {}
        std_mse = {}
        mean_erle = {}
        std_erle = {}

        for algo_name in ALGO_LIST:
            mean_mse[algo_name], std_mse[algo_name] = average_curves(curves_mse_by_algo[algo_name])
            mean_erle[algo_name], std_erle[algo_name] = average_curves(curves_erle_by_algo[algo_name])

            mse_vals = np.asarray(final_mse_by_algo[algo_name], dtype=float)
            erle_vals = np.asarray(final_erle_by_algo[algo_name], dtype=float)
            time_vals = np.asarray(time_by_algo[algo_name], dtype=float)

            summary_rows_all.append(dict(
                Filter_Order=int(filter_order),
                Algorithm=algo_name,
                Num_Trials=int(len(mse_vals)),
                Mean_Final_MSE_dB=float(np.nanmean(mse_vals)) if len(mse_vals) else np.nan,
                Std_Final_MSE_dB=float(np.nanstd(mse_vals)) if len(mse_vals) else np.nan,
                Median_Final_MSE_dB=float(np.nanmedian(mse_vals)) if len(mse_vals) else np.nan,
                Mean_Final_ERLE_dB=float(np.nanmean(erle_vals)) if len(erle_vals) else np.nan,
                Std_Final_ERLE_dB=float(np.nanstd(erle_vals)) if len(erle_vals) else np.nan,
                Median_Final_ERLE_dB=float(np.nanmedian(erle_vals)) if len(erle_vals) else np.nan,
                Mean_Time_s=float(np.nanmean(time_vals)) if len(time_vals) else np.nan,
            ))

        # 保存平均曲线。
        curves_to_save = {}
        for algo_name in ALGO_LIST:
            curves_to_save[f"{algo_name}_MSE_dB"] = mean_mse[algo_name]
            curves_to_save[f"{algo_name}_ERLE_dB"] = mean_erle[algo_name]

        save_curves_csv(
            curves_to_save,
            out_dir / f"exp_d_mean_curves_p{filter_order}.csv",
        )

        # 绘制 MSE 曲线。
        plot_mean_curve_with_std(
            mean_dict=mean_mse,
            std_dict=std_mse,
            title=f"实验 D：非线性回声建模 MSE，p={filter_order}",
            ylabel="Residual MSE (dB)",
            save_path=out_dir / f"fig_exp_d_mse_p{filter_order}.png",
            smooth_window=PLOT.get("smooth_window", 3),
            y_lim=PLOT.get("mse_ylim", None),
        )

        # 绘制 ERLE 曲线。
        plot_mean_curve_with_std(
            mean_dict=mean_erle,
            std_dict=std_erle,
            title=f"实验 D：非线性回声建模 ERLE，p={filter_order}",
            ylabel="ERLE (dB)",
            save_path=out_dir / f"fig_exp_d_erle_p{filter_order}.png",
            smooth_window=PLOT.get("smooth_window", 3),
            y_lim=PLOT.get("erle_ylim", None),
        )

        write_csv(
            summary_rows_all,
            out_dir / "exp_d_summary.csv",
        )

    print("\n" + "=" * 80)
    print("实验 D 完成")
    print(f"结果保存到：{out_dir}")
    print("=" * 80)

    print("\n[Summary]")
    for row in summary_rows_all:
        print(
            f"p={row['Filter_Order']:4d} | "
            f"{row['Algorithm']:10s} | "
            f"MSE={row['Mean_Final_MSE_dB']:.3f}±{row['Std_Final_MSE_dB']:.3f} dB | "
            f"ERLE={row['Mean_Final_ERLE_dB']:.3f}±{row['Std_Final_ERLE_dB']:.3f} dB | "
            f"time={row['Mean_Time_s']:.3f}s"
        )


if __name__ == "__main__":
    main()