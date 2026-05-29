"""
实验 D2：
标准 FDAF 无重叠帧 + 非线性残差分支级联实验。

本脚本同时完成：
    1. 级联效果评估；
    2. 流式逐帧处理；
    3. 实时性评估。

算法对比：
    LMS
    FDAF
    FDAF+WL-LMS
    FDAF+GH-WL-LMS-Fast

注意：
    当前是远端单讲场景，理想消除后输出是静音，
    因此本实验不使用 PESQ / STOI。
"""

import sys
import csv
import time
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
# 导入配置
# ============================================================

from configs.exp_d2_cascade_config import (
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
    FRAME_LENGTH,
    HOP_LENGTH,
    MC_TRIALS,
    SEED,
    SS_LAST_RATIO,
    ALGO_LIST,
    LMS_PARAMS,
    FDAF_PARAMS,
    NONLINEAR_FILTER_ORDER,
    WLLMS_PARAMS,
    GHWLLMSFAST_PARAMS,
    PLOT,
    RESULT_DIR,
)

from datasets.aec_synthetic import (
    load_meta,
    filter_meta_for_exp_d,
    filter_meta_by_available_audio,
    load_exp_d_pair,
)

from algorithms import (
    LMS,
    WLLMS,
    GHWLLMSFast,
    FDAF,
    FDAFNonlinearCascade,
)


# ============================================================
# 基础工具函数
# ============================================================

def safe_db(x, floor=1e-20):
    """线性能量转 dB。"""
    x = max(float(x), float(floor))
    return 10.0 * np.log10(x)


def moving_average(x, window: int):
    """滑动平均，用于平滑绘图。"""
    x = np.asarray(x, dtype=np.float64)

    if window is None or int(window) <= 1:
        return x

    window = int(window)
    if len(x) < window:
        return x

    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(x, kernel, mode="same")


def write_csv(rows, path: Path):
    """保存字典列表为 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
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
    """保存平均曲线 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    names = list(curves.keys())
    n = max(len(v) for v in curves.values())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index"] + names)

        for i in range(n):
            row = [i]
            for name in names:
                v = curves[name]
                row.append(f"{float(v[i]):.6f}" if i < len(v) else "")
            writer.writerow(row)

    print(f"[csv] 已保存：{path}")


def average_curves(curve_list):
    """对多条曲线求均值和标准差，长度不同则截断到最短。"""
    if len(curve_list) == 0:
        return np.array([]), np.array([])

    min_len = min(len(c) for c in curve_list)
    arr = np.stack([np.asarray(c[:min_len], dtype=np.float64) for c in curve_list], axis=0)

    return np.mean(arr, axis=0), np.std(arr, axis=0)


def sample_trials(df, mc_trials: int, seed: int):
    """从筛选后的 meta 中抽取 trial。"""
    if len(df) == 0:
        raise RuntimeError("筛选后没有可用样本。")

    rng = np.random.default_rng(int(seed))
    replace = len(df) < int(mc_trials)
    idx = rng.choice(len(df), size=int(mc_trials), replace=replace)

    return df.iloc[idx].reset_index(drop=True)


def split_into_blocks(x, d, block_size: int):
    """
    将信号切分为无重叠 block。

    末尾不足 block_size 的部分直接丢弃，避免补零影响实时统计和 ERLE。
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = min(len(x), len(d))
    n_blocks = n // int(block_size)
    n_used = n_blocks * int(block_size)

    x = x[:n_used]
    d = d[:n_used]

    for i in range(n_blocks):
        s = i * int(block_size)
        e = s + int(block_size)
        yield i, x[s:e], d[s:e]


# ============================================================
# 模型构造
# ============================================================

def reset_algorithm(algo, seed: int):
    """重置已有 LMS/WL/GH 算法对象。"""
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


def build_model(algo_name: str, seed: int):
    """
    根据算法名称构造模型。

    LMS:
        时域 sample-by-sample 线性 LMS baseline。

    FDAF:
        只有线性 FDAF。

    FDAF+WL-LMS:
        FDAF + WL-LMS 非线性残差分支。

    FDAF+GH-WL-LMS-Fast:
        FDAF + GH-WL-LMS-Fast 非线性残差分支。
    """
    if algo_name == "LMS":
        lms = LMS(
            LMS_PARAMS["filter_order"],
            step_size=LMS_PARAMS["step_size"],
        )
        reset_algorithm(lms, seed=seed)
        return dict(type="LMS", model=lms)

    if algo_name == "FDAF":
        fdaf = FDAF(**FDAF_PARAMS)
        cascade = FDAFNonlinearCascade(
            fdaf=fdaf,
            nonlinear_filter=None,
            nonlinear_order=NONLINEAR_FILTER_ORDER,
        )
        return dict(type="CASCADE", model=cascade)

    if algo_name == "FDAF+WL-LMS":
        fdaf = FDAF(**FDAF_PARAMS)
        wl = WLLMS(
            NONLINEAR_FILTER_ORDER,
            **WLLMS_PARAMS,
        )
        reset_algorithm(wl, seed=seed)

        cascade = FDAFNonlinearCascade(
            fdaf=fdaf,
            nonlinear_filter=wl,
            nonlinear_order=NONLINEAR_FILTER_ORDER,
        )
        return dict(type="CASCADE", model=cascade)

    if algo_name == "FDAF+GH-WL-LMS-Fast":
        fdaf = FDAF(**FDAF_PARAMS)
        gh = GHWLLMSFast(
            NONLINEAR_FILTER_ORDER,
            **GHWLLMSFAST_PARAMS,
        )
        reset_algorithm(gh, seed=seed)

        cascade = FDAFNonlinearCascade(
            fdaf=fdaf,
            nonlinear_filter=gh,
            nonlinear_order=NONLINEAR_FILTER_ORDER,
        )
        return dict(type="CASCADE", model=cascade)

    raise ValueError(f"未知算法：{algo_name}")


# ============================================================
# LMS block 处理
# ============================================================

def process_lms_block(lms, x_block, d_block, x_buf):
    """
    用时域 LMS 处理一个 block。

    LMS 仍然是逐样本更新；
    这里只是为了和流式 block 框架统一。
    """
    x_block = np.asarray(x_block, dtype=np.float64)
    d_block = np.asarray(d_block, dtype=np.float64)

    y_hat = np.zeros_like(d_block)
    e_block = np.zeros_like(d_block)

    for k, xk in enumerate(x_block):
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = float(xk)

        if hasattr(lms, "predict"):
            y = float(lms.predict(x_buf))
            e = float(d_block[k] - y)
            lms.update(x_buf, float(d_block[k]))
        else:
            e = float(lms.update(x_buf, float(d_block[k])))
            y = float(d_block[k] - e)

        y_hat[k] = y
        e_block[k] = e

    return y_hat, e_block, x_buf


# ============================================================
# 单条 trial 运行
# ============================================================

def run_one_trial_one_algo(
    algo_name: str,
    x: np.ndarray,
    d: np.ndarray,
    fs: int,
    seed: int,
):
    """
    运行一个 trial 的一个算法。

    返回：
        每帧 MSE / ERLE 曲线；
        每帧耗时；
        稳态指标和实时性统计。
    """
    model_info = build_model(algo_name, seed=seed)
    model_type = model_info["type"]
    model = model_info["model"]

    frame_budget_ms = 1000.0 * float(FRAME_LENGTH) / float(fs)

    mse_curve = []
    erle_curve = []
    frame_times_ms = []
    frame_rows = []

    if model_type == "LMS":
        x_buf = np.zeros(int(LMS_PARAMS["filter_order"]), dtype=np.float64)
    else:
        x_buf = None

    y_all = []
    e_all = []
    d_all = []

    t_total0 = time.perf_counter()

    for frame_idx, x_block, d_block in split_into_blocks(x, d, FRAME_LENGTH):
        t0 = time.perf_counter()

        if model_type == "LMS":
            y_hat, e_block, x_buf = process_lms_block(
                model,
                x_block,
                d_block,
                x_buf,
            )
        else:
            out = model.process_block(x_block, d_block)
            y_hat = out["y_hat"]
            e_block = out["e_final"]

        elapsed_ms = 1000.0 * (time.perf_counter() - t0)

        d_power = float(np.mean(d_block ** 2))
        e_power = float(np.mean(e_block ** 2))

        mse_db = safe_db(e_power)
        erle_db = safe_db(d_power / max(e_power, 1e-20))

        mse_curve.append(mse_db)
        erle_curve.append(erle_db)
        frame_times_ms.append(elapsed_ms)

        load = elapsed_ms / max(frame_budget_ms, 1e-12)
        overrun = bool(load > 1.0)

        frame_rows.append(dict(
            Frame_Index=int(frame_idx),
            Frame_Time_ms=float(elapsed_ms),
            Frame_Budget_ms=float(frame_budget_ms),
            Load=float(load),
            Overrun=int(overrun),
            MSE_dB=float(mse_db),
            ERLE_dB=float(erle_db),
        ))

        y_all.append(y_hat)
        e_all.append(e_block)
        d_all.append(d_block)

    total_time_s = time.perf_counter() - t_total0

    mse_curve = np.asarray(mse_curve, dtype=np.float64)
    erle_curve = np.asarray(erle_curve, dtype=np.float64)
    frame_times_ms = np.asarray(frame_times_ms, dtype=np.float64)

    audio_duration_s = len(mse_curve) * float(FRAME_LENGTH) / float(fs)

    ss_n = max(1, int(np.ceil(len(mse_curve) * float(SS_LAST_RATIO))))

    final_mse_db = float(np.mean(mse_curve[-ss_n:]))
    final_erle_db = float(np.mean(erle_curve[-ss_n:]))

    rtf = float(total_time_s / max(audio_duration_s, 1e-12))

    loads = frame_times_ms / max(frame_budget_ms, 1e-12)

    realtime = dict(
        Audio_Duration_s=float(audio_duration_s),
        Total_Processing_Time_s=float(total_time_s),
        RTF=float(rtf),
        Frame_Budget_ms=float(frame_budget_ms),
        Mean_Frame_Time_ms=float(np.mean(frame_times_ms)),
        Median_Frame_Time_ms=float(np.median(frame_times_ms)),
        P95_Frame_Time_ms=float(np.percentile(frame_times_ms, 95)),
        P99_Frame_Time_ms=float(np.percentile(frame_times_ms, 99)),
        Max_Frame_Time_ms=float(np.max(frame_times_ms)),
        Mean_Load=float(np.mean(loads)),
        P95_Load=float(np.percentile(loads, 95)),
        P99_Load=float(np.percentile(loads, 99)),
        Max_Load=float(np.max(loads)),
        Overrun_Rate=float(np.mean(loads > 1.0)),
    )

    return dict(
        mse_curve=mse_curve,
        erle_curve=erle_curve,
        frame_rows=frame_rows,
        final_mse_db=final_mse_db,
        final_erle_db=final_erle_db,
        realtime=realtime,
    )


# ============================================================
# 绘图函数
# ============================================================

def plot_mean_curve_with_std(
    mean_dict,
    std_dict,
    title,
    ylabel,
    save_path,
    smooth_window=1,
    y_lim=None,
):
    """绘制均值曲线和标准差阴影。"""
    fig, ax = plt.subplots(figsize=(8, 4.8))

    for name, mean_curve in mean_dict.items():
        mean_curve = np.asarray(mean_curve, dtype=np.float64)
        std_curve = np.asarray(std_dict[name], dtype=np.float64)

        mean_s = moving_average(mean_curve, smooth_window)
        std_s = moving_average(std_curve, smooth_window)

        xs = np.arange(len(mean_s))

        ax.plot(xs, mean_s, label=name)
        ax.fill_between(xs, mean_s - std_s, mean_s + std_s, alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel(f"帧索引，每帧 {FRAME_LENGTH} 个样本")
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


def plot_realtime_bar(summary_rows, save_path):
    """绘制不同算法的平均 RTF 柱状图。"""
    algos = [r["Algorithm"] for r in summary_rows]
    rtfs = [r["Mean_RTF"] for r in summary_rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(algos, rtfs)
    ax.axhline(1.0, linestyle="--", linewidth=1.2, label="实时阈值 RTF=1")

    ax.set_title("实验 D2：实时性 RTF 对比")
    ax.set_ylabel("RTF = 处理时间 / 音频时长")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    plt.xticks(rotation=20)

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
    print("实验 D2：FDAF + 非线性残差分支级联，含流式实时性评估")
    print("=" * 80)
    print(f"META_CSV            = {META_CSV}")
    print(f"FAREND_DIR          = {FAREND_DIR}")
    print(f"ECHO_DIR            = {ECHO_DIR}")
    print(f"TARGET_FS           = {TARGET_FS}")
    print(f"SEGMENT_SECONDS     = {SEGMENT_SECONDS}")
    print(f"FRAME_LENGTH        = {FRAME_LENGTH}")
    print(f"HOP_LENGTH          = {HOP_LENGTH}")
    print(f"MC_TRIALS           = {MC_TRIALS}")
    print(f"ALGO_LIST           = {ALGO_LIST}")
    print(f"RESULT_DIR          = {out_dir}")
    print("=" * 80)

    if int(FRAME_LENGTH) != int(HOP_LENGTH):
        raise RuntimeError(
            "当前 D2 第一版只实现无重叠帧，要求 FRAME_LENGTH == HOP_LENGTH。"
        )

    df = load_meta(META_CSV)

    df_sel = filter_meta_for_exp_d(
        df,
        require_farend_nonlinear=REQUIRE_FAREND_NONLINEAR,
        require_farend_noisy=REQUIRE_FAREND_NOISY,
        require_nearend_noisy=REQUIRE_NEAREND_NOISY,
        split=SPLIT,
    )

    df_sel = filter_meta_by_available_audio(
        df_sel,
        farend_dir=FAREND_DIR,
        echo_dir=ECHO_DIR,
    )

    print(f"[meta] 原始样本数：{len(df)}")
    print(f"[meta] 筛选后可用样本数：{len(df_sel)}")

    trials = sample_trials(df_sel, MC_TRIALS, seed=SEED)

    selected_rows = []
    for _, r in trials.iterrows():
        selected_rows.append(dict(
            fileid=int(r["fileid"]),
            split=r.get("split", ""),
            is_farend_nonlinear=r.get("is_farend_nonlinear", ""),
            is_farend_noisy=r.get("is_farend_noisy", ""),
            is_nearend_noisy=r.get("is_nearend_noisy", ""),
        ))

    write_csv(selected_rows, out_dir / "exp_d2_selected_fileids.csv")

    trial_rows_all = []
    frame_rows_all = []

    curves_mse_by_algo = {name: [] for name in ALGO_LIST}
    curves_erle_by_algo = {name: [] for name in ALGO_LIST}

    final_mse_by_algo = {name: [] for name in ALGO_LIST}
    final_erle_by_algo = {name: [] for name in ALGO_LIST}
    realtime_by_algo = {name: [] for name in ALGO_LIST}

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
        fs = int(pair["fs"])

        print(
            f"  片段长度：{len(x)} samples, fs={fs}, "
            f"start={pair['segment_start_sample']}"
        )

        for algo_name in ALGO_LIST:
            result = run_one_trial_one_algo(
                algo_name=algo_name,
                x=x,
                d=d,
                fs=fs,
                seed=SEED + trial_id,
            )

            curves_mse_by_algo[algo_name].append(result["mse_curve"])
            curves_erle_by_algo[algo_name].append(result["erle_curve"])

            final_mse_by_algo[algo_name].append(result["final_mse_db"])
            final_erle_by_algo[algo_name].append(result["final_erle_db"])
            realtime_by_algo[algo_name].append(result["realtime"])

            trial_rows_all.append(dict(
                Experiment="D2",
                Trial=int(trial_id),
                FileID=int(fileid),
                Algorithm=algo_name,
                Final_MSE_dB=float(result["final_mse_db"]),
                Final_ERLE_dB=float(result["final_erle_db"]),
                Audio_Duration_s=float(result["realtime"]["Audio_Duration_s"]),
                Total_Processing_Time_s=float(result["realtime"]["Total_Processing_Time_s"]),
                RTF=float(result["realtime"]["RTF"]),
                Mean_Frame_Time_ms=float(result["realtime"]["Mean_Frame_Time_ms"]),
                P95_Frame_Time_ms=float(result["realtime"]["P95_Frame_Time_ms"]),
                P99_Frame_Time_ms=float(result["realtime"]["P99_Frame_Time_ms"]),
                Max_Frame_Time_ms=float(result["realtime"]["Max_Frame_Time_ms"]),
                Overrun_Rate=float(result["realtime"]["Overrun_Rate"]),
                Frame_Budget_ms=float(result["realtime"]["Frame_Budget_ms"]),
                Segment_Start_Sample=int(pair["segment_start_sample"]),
                Segment_Samples=int(pair["segment_samples"]),
                Farend_Path=pair["farend_path"],
                Echo_Path=pair["echo_path"],
            ))

            for fr in result["frame_rows"]:
                rr = dict(fr)
                rr.update(dict(
                    Trial=int(trial_id),
                    FileID=int(fileid),
                    Algorithm=algo_name,
                ))
                frame_rows_all.append(rr)

            print(
                f"  {algo_name:24s} | "
                f"MSE={result['final_mse_db']:.3f} dB | "
                f"ERLE={result['final_erle_db']:.3f} dB | "
                f"RTF={result['realtime']['RTF']:.3f} | "
                f"P95={result['realtime']['P95_Frame_Time_ms']:.3f} ms | "
                f"overrun={100.0 * result['realtime']['Overrun_Rate']:.2f}%"
            )

        write_csv(trial_rows_all, out_dir / "exp_d2_trial_results_long.csv")
        write_csv(frame_rows_all, out_dir / "exp_d2_frame_time_long.csv")

    mean_mse = {}
    std_mse = {}
    mean_erle = {}
    std_erle = {}

    summary_rows = []

    for algo_name in ALGO_LIST:
        mean_mse[algo_name], std_mse[algo_name] = average_curves(
            curves_mse_by_algo[algo_name]
        )
        mean_erle[algo_name], std_erle[algo_name] = average_curves(
            curves_erle_by_algo[algo_name]
        )

        mse_vals = np.asarray(final_mse_by_algo[algo_name], dtype=np.float64)
        erle_vals = np.asarray(final_erle_by_algo[algo_name], dtype=np.float64)

        rtf_vals = np.asarray([r["RTF"] for r in realtime_by_algo[algo_name]], dtype=np.float64)
        mean_frame_vals = np.asarray([r["Mean_Frame_Time_ms"] for r in realtime_by_algo[algo_name]], dtype=np.float64)
        p95_vals = np.asarray([r["P95_Frame_Time_ms"] for r in realtime_by_algo[algo_name]], dtype=np.float64)
        p99_vals = np.asarray([r["P99_Frame_Time_ms"] for r in realtime_by_algo[algo_name]], dtype=np.float64)
        overrun_vals = np.asarray([r["Overrun_Rate"] for r in realtime_by_algo[algo_name]], dtype=np.float64)

        summary_rows.append(dict(
            Experiment="D2",
            Algorithm=algo_name,
            Num_Trials=int(len(mse_vals)),
            Mean_Final_MSE_dB=float(np.nanmean(mse_vals)),
            Std_Final_MSE_dB=float(np.nanstd(mse_vals)),
            Median_Final_MSE_dB=float(np.nanmedian(mse_vals)),
            Mean_Final_ERLE_dB=float(np.nanmean(erle_vals)),
            Std_Final_ERLE_dB=float(np.nanstd(erle_vals)),
            Median_Final_ERLE_dB=float(np.nanmedian(erle_vals)),
            Mean_RTF=float(np.nanmean(rtf_vals)),
            Std_RTF=float(np.nanstd(rtf_vals)),
            Mean_Frame_Time_ms=float(np.nanmean(mean_frame_vals)),
            Mean_P95_Frame_Time_ms=float(np.nanmean(p95_vals)),
            Mean_P99_Frame_Time_ms=float(np.nanmean(p99_vals)),
            Mean_Overrun_Rate=float(np.nanmean(overrun_vals)),
            Frame_Budget_ms=float(1000.0 * FRAME_LENGTH / TARGET_FS),
            Frame_Length=int(FRAME_LENGTH),
            Hop_Length=int(HOP_LENGTH),
        ))

    write_csv(summary_rows, out_dir / "exp_d2_summary.csv")

    curves_to_save = {}
    for algo_name in ALGO_LIST:
        curves_to_save[f"{algo_name}_MSE_dB"] = mean_mse[algo_name]
        curves_to_save[f"{algo_name}_ERLE_dB"] = mean_erle[algo_name]

    save_curves_csv(
        curves_to_save,
        out_dir / "exp_d2_mean_curves.csv",
    )

    plot_mean_curve_with_std(
        mean_dict=mean_mse,
        std_dict=std_mse,
        title="实验 D2：FDAF + 非线性残差级联 MSE",
        ylabel="Residual MSE (dB)",
        save_path=out_dir / "fig_exp_d2_mse.png",
        smooth_window=PLOT.get("smooth_window", 3),
        y_lim=PLOT.get("mse_ylim", None),
    )

    plot_mean_curve_with_std(
        mean_dict=mean_erle,
        std_dict=std_erle,
        title="实验 D2：FDAF + 非线性残差级联 ERLE",
        ylabel="ERLE (dB)",
        save_path=out_dir / "fig_exp_d2_erle.png",
        smooth_window=PLOT.get("smooth_window", 3),
        y_lim=PLOT.get("erle_ylim", None),
    )

    plot_realtime_bar(
        summary_rows,
        save_path=out_dir / "fig_exp_d2_rtf_bar.png",
    )

    print("\n" + "=" * 80)
    print("实验 D2 完成")
    print(f"结果保存到：{out_dir}")
    print("=" * 80)

    print("\n[Summary]")
    for row in summary_rows:
        print(
            f"{row['Algorithm']:24s} | "
            f"MSE={row['Mean_Final_MSE_dB']:.3f}±{row['Std_Final_MSE_dB']:.3f} dB | "
            f"ERLE={row['Mean_Final_ERLE_dB']:.3f}±{row['Std_Final_ERLE_dB']:.3f} dB | "
            f"RTF={row['Mean_RTF']:.3f} | "
            f"P95={row['Mean_P95_Frame_Time_ms']:.3f} ms | "
            f"Overrun={100.0 * row['Mean_Overrun_Rate']:.2f}%"
        )


if __name__ == "__main__":
    main()