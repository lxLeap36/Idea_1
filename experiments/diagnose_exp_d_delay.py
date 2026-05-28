"""
实验 D 延迟诊断脚本：
估计 AEC Challenge 合成数据中 farend_speech 到 echo_signal 的主延迟。

目的：
    在正式比较 LMS / WL-LMS / GH-WL-LMS 之前，
    先判断当前 filter_order 是否覆盖了主要回声延迟。

核心诊断：
    如果估计延迟 delay_samples 明显大于 p，
    那么使用 p taps 的自适应滤波器很可能学不到主要回声成分。

输出：
    results/exp_d_aec_synthetic_delay/<timestamp>/
        exp_d_delay_diagnostics.csv
        fig_delay_hist.png
        fig_delay_vs_score.png
"""

import sys
import csv
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
# 导入实验 D 配置和数据工具
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
    REMOVE_DC,
    PEAK_NORMALIZE,
    PEAK_EPS,
    FILTER_ORDERS,
    MC_TRIALS,
    SEED,
)

from datasets.aec_synthetic import (
    load_meta,
    filter_meta_for_exp_d,
    filter_meta_by_available_audio,
    load_exp_d_pair,
)


# ============================================================
# 本脚本专用参数
# ============================================================

# 搜索最大延迟范围。
# 200 ms 对 16 kHz 是 3200 samples。
# 如果后面发现很多样本卡在最大值附近，可以改成 500 ms。
MAX_DELAY_MS = 200.0

# 诊断样本数量。
# None 表示使用 MC_TRIALS。
# 如果想全量诊断，可以设为 "all"。
NUM_DIAG_TRIALS = MC_TRIALS

# 是否同时计算包络/绝对值相关。
# 对于强非线性回声，原始波形相关可能偏弱，
# abs 相关可以作为辅助判断。
USE_ABS_XCORR = True

# 结果输出目录。
RESULT_DIR = ROOT / "results" / "exp_d_aec_synthetic_delay"


# ============================================================
# 工具函数
# ============================================================

def write_csv(rows, path: Path):
    """保存字典列表为 CSV。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"[csv] 无结果可保存：{path}")
        return

    headers = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] 已保存：{path}")


def sample_trials(df, num_trials, seed: int):
    """
    从筛选后的 meta 中抽取诊断样本。

    num_trials:
        - "all"：使用全部样本
        - None ：使用 MC_TRIALS
        - int  ：使用指定数量
    """
    if len(df) == 0:
        raise RuntimeError("筛选后没有可用样本，请检查 meta.csv 和筛选条件。")

    if num_trials == "all":
        return df.reset_index(drop=True)

    if num_trials is None:
        num_trials = MC_TRIALS

    num_trials = int(num_trials)

    rng = np.random.default_rng(int(seed))
    replace = len(df) < num_trials

    idx = rng.choice(len(df), size=num_trials, replace=replace)
    return df.iloc[idx].reset_index(drop=True)


def _preprocess_for_xcorr(x: np.ndarray, mode: str = "raw") -> np.ndarray:
    """
    为互相关估计做预处理。

    mode:
        raw：直接使用去均值后的波形
        abs：使用绝对值包络，再去均值
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x)

    if mode == "raw":
        y = x.copy()
    elif mode == "abs":
        y = np.abs(x)
    else:
        raise ValueError(f"未知互相关模式：{mode}")

    y = y - np.mean(y)

    std = np.std(y)
    if std > 1e-12:
        y = y / std

    return y


def estimate_delay_by_xcorr(
    x: np.ndarray,
    d: np.ndarray,
    fs: int,
    max_delay_ms: float = 200.0,
    mode: str = "raw",
):
    """
    使用归一化互相关估计 farend_speech 到 echo_signal 的主延迟。

    定义：
        delay >= 0 表示 echo_signal 相对 farend_speech 滞后 delay 个采样点。

    计算：
        对每个 delay，比较：
            x[0 : N-delay]
            d[delay : N]

    返回：
        best_delay_samples
        best_delay_ms
        best_score
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = min(len(x), len(d))
    x = x[:n]
    d = d[:n]

    x0 = _preprocess_for_xcorr(x, mode=mode)
    d0 = _preprocess_for_xcorr(d, mode=mode)

    max_delay_samples = int(round(float(max_delay_ms) * 1e-3 * int(fs)))
    max_delay_samples = min(max_delay_samples, n - 2)

    if max_delay_samples <= 0:
        raise ValueError("max_delay_samples 太小，无法估计延迟。")

    scores = np.zeros(max_delay_samples + 1, dtype=np.float64)

    # 用累积能量做每个 lag 的归一化。
    x2_cumsum = np.concatenate([[0.0], np.cumsum(x0 * x0)])
    d2_cumsum = np.concatenate([[0.0], np.cumsum(d0 * d0)])

    for delay in range(max_delay_samples + 1):
        length = n - delay

        x_seg = x0[:length]
        d_seg = d0[delay:delay + length]

        numerator = float(np.dot(x_seg, d_seg))

        x_energy = float(x2_cumsum[length] - x2_cumsum[0])
        d_energy = float(d2_cumsum[delay + length] - d2_cumsum[delay])

        denom = np.sqrt(x_energy * d_energy) + 1e-12
        scores[delay] = numerator / denom

    # 这里用绝对值找峰值。
    # 如果你只关心同相相关，也可以改成 np.argmax(scores)。
    best_delay = int(np.argmax(np.abs(scores)))
    best_score = float(scores[best_delay])
    best_delay_ms = 1000.0 * best_delay / float(fs)

    return best_delay, best_delay_ms, best_score, scores


def plot_delay_hist(rows, save_path: Path):
    """绘制延迟分布直方图。"""
    delays = np.asarray([r["Delay_Raw_ms"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(delays, bins=30, alpha=0.8)

    ax.set_title("实验 D：farend_speech 到 echo_signal 的主延迟分布")
    ax.set_xlabel("估计延迟 delay (ms)")
    ax.set_ylabel("样本数量")
    ax.grid(True, alpha=0.3)

    # 标注当前配置中的 filter_order 覆盖时长。
    for p in FILTER_ORDERS:
        p_ms = 1000.0 * int(p) / float(TARGET_FS)
        ax.axvline(p_ms, linestyle="--", linewidth=1.2, label=f"p={p} 覆盖 {p_ms:.1f} ms")

    ax.legend(loc="best")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[plot] 已保存：{save_path}")


def plot_delay_vs_score(rows, save_path: Path):
    """绘制延迟和相关峰值分数的散点图。"""
    delays = np.asarray([r["Delay_Raw_ms"] for r in rows], dtype=float)
    scores = np.asarray([abs(r["Score_Raw"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter(delays, scores, alpha=0.75)

    ax.set_title("实验 D：延迟估计与互相关峰值强度")
    ax.set_xlabel("估计延迟 delay (ms)")
    ax.set_ylabel("|归一化互相关峰值|")
    ax.grid(True, alpha=0.3)

    for p in FILTER_ORDERS:
        p_ms = 1000.0 * int(p) / float(TARGET_FS)
        ax.axvline(p_ms, linestyle="--", linewidth=1.2, label=f"p={p}")

    ax.legend(loc="best")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[plot] 已保存：{save_path}")


def summarize_delays(rows):
    """在控制台打印延迟统计。"""
    delays = np.asarray([r["Delay_Raw_samples"] for r in rows], dtype=float)
    delays_ms = np.asarray([r["Delay_Raw_ms"] for r in rows], dtype=float)
    scores = np.asarray([abs(r["Score_Raw"]) for r in rows], dtype=float)

    print("\n[延迟统计：raw xcorr]")
    print(f"  样本数                 : {len(rows)}")
    print(f"  delay mean             : {np.mean(delays):.1f} samples / {np.mean(delays_ms):.2f} ms")
    print(f"  delay median           : {np.median(delays):.1f} samples / {np.median(delays_ms):.2f} ms")
    print(f"  delay min              : {np.min(delays):.1f} samples / {np.min(delays_ms):.2f} ms")
    print(f"  delay max              : {np.max(delays):.1f} samples / {np.max(delays_ms):.2f} ms")
    print(f"  |score| mean           : {np.mean(scores):.4f}")
    print(f"  |score| median         : {np.median(scores):.4f}")

    for p in FILTER_ORDERS:
        p = int(p)
        p_ms = 1000.0 * p / float(TARGET_FS)
        ratio_over = float(np.mean(delays > p))
        print(
            f"  超过 p={p:<5d} ({p_ms:6.2f} ms) 的比例 : "
            f"{100.0 * ratio_over:6.2f}%"
        )


# ============================================================
# 主流程
# ============================================================

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(RESULT_DIR) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("实验 D 延迟诊断：farend_speech -> echo_signal")
    print("=" * 80)
    print(f"META_CSV              = {META_CSV}")
    print(f"FAREND_DIR            = {FAREND_DIR}")
    print(f"ECHO_DIR              = {ECHO_DIR}")
    print(f"TARGET_FS             = {TARGET_FS}")
    print(f"SEGMENT_SECONDS       = {SEGMENT_SECONDS}")
    print(f"SEGMENT_MODE          = {SEGMENT_MODE}")
    print(f"MAX_DELAY_MS          = {MAX_DELAY_MS}")
    print(f"FILTER_ORDERS         = {FILTER_ORDERS}")
    print(f"NUM_DIAG_TRIALS       = {NUM_DIAG_TRIALS}")
    print(f"RESULT_DIR            = {out_dir}")
    print("=" * 80)

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
    print(f"[meta] 筛选后样本数：{len(df_sel)}")

    if len(df_sel) == 0:
        raise RuntimeError("筛选后没有样本，请检查筛选条件。")

    trials = sample_trials(df_sel, NUM_DIAG_TRIALS, seed=SEED)

    rows = []

    for trial_id, row in trials.iterrows():
        fileid = int(row["fileid"])

        print("\n" + "-" * 80)
        print(f"[diag {trial_id + 1}/{len(trials)}] fileid={fileid}")
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
            print(f"[skip] fileid={fileid}, 原因：{ex}")
            continue

        x = pair["x"]
        d = pair["d"]
        fs = int(pair["fs"])

        delay_raw, delay_raw_ms, score_raw, _ = estimate_delay_by_xcorr(
            x=x,
            d=d,
            fs=fs,
            max_delay_ms=MAX_DELAY_MS,
            mode="raw",
        )

        if USE_ABS_XCORR:
            delay_abs, delay_abs_ms, score_abs, _ = estimate_delay_by_xcorr(
                x=x,
                d=d,
                fs=fs,
                max_delay_ms=MAX_DELAY_MS,
                mode="abs",
            )
        else:
            delay_abs = np.nan
            delay_abs_ms = np.nan
            score_abs = np.nan

        result_row = dict(
            Trial=int(trial_id),
            FileID=int(fileid),
            FS=int(fs),
            Segment_Start_Sample=int(pair["segment_start_sample"]),
            Segment_Samples=int(pair["segment_samples"]),
            Segment_Seconds=float(pair["segment_samples"] / fs),

            Delay_Raw_samples=int(delay_raw),
            Delay_Raw_ms=float(delay_raw_ms),
            Score_Raw=float(score_raw),

            Delay_Abs_samples=int(delay_abs) if np.isfinite(delay_abs) else np.nan,
            Delay_Abs_ms=float(delay_abs_ms) if np.isfinite(delay_abs_ms) else np.nan,
            Score_Abs=float(score_abs) if np.isfinite(score_abs) else np.nan,

            Farend_Path=pair["farend_path"],
            Echo_Path=pair["echo_path"],
        )

        rows.append(result_row)

        print(
            f"  raw delay = {delay_raw:5d} samples "
            f"({delay_raw_ms:8.2f} ms), "
            f"score={score_raw:+.4f}"
        )

        if USE_ABS_XCORR:
            print(
                f"  abs delay = {delay_abs:5d} samples "
                f"({delay_abs_ms:8.2f} ms), "
                f"score={score_abs:+.4f}"
            )

        # 每条样本后保存一次，避免中途出错丢失结果。
        write_csv(rows, out_dir / "exp_d_delay_diagnostics.csv")

    if len(rows) == 0:
        print("[done] 没有成功诊断的样本。")
        return

    summarize_delays(rows)

    write_csv(rows, out_dir / "exp_d_delay_diagnostics.csv")

    plot_delay_hist(
        rows,
        save_path=out_dir / "fig_delay_hist.png",
    )

    plot_delay_vs_score(
        rows,
        save_path=out_dir / "fig_delay_vs_score.png",
    )

    print("\n" + "=" * 80)
    print("实验 D 延迟诊断完成")
    print(f"结果保存到：{out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()