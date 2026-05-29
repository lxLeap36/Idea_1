"""
实验 D3：
真实远端语音上的模拟扬声器/功放非线性建模实验。

本实验只使用 AEC Challenge synthetic 数据集中的 farend_speech。
我们手动对 farend_speech 施加 memoryless 非线性，得到目标信号 d(n)。

实验目的：
    排除长线性回声路径的干扰，单独验证：
        LMS
        WL-LMS
        GH-WL-LMS
    对扬声器饱和 / 设备失真这类非线性映射的建模能力。

注意：
    这里不是完整 AEC 实验。
    这里的任务是：
        x(n) -> f_nonlinear(x(n))

单讲场景下，PESQ/STOI 可以作为辅助的建模相似度指标，但它不代表 AEC 消除质量，
    因为这里没有近端语音，用的是 y_hat 与 y=f_nonlinear(x(n)) 的比较。
"""

import sys
import csv
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
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
from metrics import compute_pesq_stoi

from configs.exp_d3_config import (
    META_CSV,
    FAREND_DIR,
    REQUIRE_FAREND_NOISY,
    SPLIT,
    TARGET_FS,
    SEGMENT_SECONDS,
    SEGMENT_MODE,
    SKIP_SHORT_CLIPS,
    REMOVE_DC,
    PEAK_NORMALIZE_INPUT,
    PEAK_EPS,
    NONLINEARITY_TYPE,
    DELTA_POS,
    DELTA_NEG,
    DRIVE_GAIN,
    CENTER_TARGET,
    MATCH_TARGET_RMS_TO_INPUT,
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
    load_wav_mono,
    resample_if_needed,
    remove_dc,
    find_audio_by_fileid,
    build_input_vector_stream,
)

from algorithms import LMS, WLLMS, GHWLLMSFast


# ============================================================
# 基础工具函数
# ============================================================

def safe_db(x, floor=1e-20):
    """将线性能量转换为 dB。"""
    x = max(float(x), float(floor))
    return 10.0 * np.log10(x)


def moving_average(x, window: int):
    """对曲线做简单滑动平均，方便绘图观察趋势。"""
    x = np.asarray(x, dtype=float)

    if window is None or int(window) <= 1:
        return x

    window = int(window)
    if len(x) < window:
        return x

    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def write_csv(rows, path: Path):
    """保存字典列表为 CSV 文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"[csv] 无数据可保存：{path}")
        return

    headers = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[csv] 已保存：{path}")


def save_curves_csv(curves: dict, path: Path):
    """保存平均学习曲线。"""
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


def average_curves(curve_list):
    """
    对多条曲线求均值和标准差。

    如果不同 trial 曲线长度略有差异，则截断到最短长度。
    """
    if len(curve_list) == 0:
        return np.array([]), np.array([])

    min_len = min(len(c) for c in curve_list)
    arr = np.stack([np.asarray(c[:min_len], dtype=float) for c in curve_list], axis=0)

    return np.mean(arr, axis=0), np.std(arr, axis=0)


# ============================================================
# 数据集筛选与读取
# ============================================================

def collect_available_fileids(folder: Path) -> set:
    """
    扫描 farend_speech 文件夹中实际存在的 fileid。

    这样可以避免 meta.csv 中有记录，但本地没有下载对应音频。
    """
    import re

    folder = Path(folder)
    pattern = re.compile(r"fileid_(\d+)\.wav$", re.IGNORECASE)

    ids = set()
    for p in folder.glob("*.wav"):
        m = pattern.search(p.name)
        if m is not None:
            ids.add(int(m.group(1)))

    return ids


def load_and_filter_meta_for_d3():
    """
    读取 meta.csv，并筛选 D3 可用样本。

    D3 只需要 farend_speech，因此只检查：
        1. 是否满足 is_farend_noisy 条件；
        2. 本地 farend_speech 文件是否实际存在。
    """
    if not Path(META_CSV).exists():
        raise FileNotFoundError(f"meta.csv 不存在：{META_CSV}")

    df = pd.read_csv(META_CSV)

    if "fileid" not in df.columns:
        raise RuntimeError("meta.csv 中没有 fileid 列。")

    out = df.copy()

    if "is_farend_noisy" in out.columns:
        out = out[out["is_farend_noisy"] == int(REQUIRE_FAREND_NOISY)]

    if SPLIT is not None and "split" in out.columns:
        out = out[out["split"].astype(str) == str(SPLIT)]

    available_ids = collect_available_fileids(FAREND_DIR)
    out = out[out["fileid"].astype(int).isin(available_ids)]
    out = out.reset_index(drop=True)

    print(f"[meta] 原始样本数：{len(df)}")
    print(f"[meta] farend_speech 实际存在 fileid 数：{len(available_ids)}")
    print(f"[meta] D3 筛选后样本数：{len(out)}")

    if len(out) == 0:
        raise RuntimeError("D3 筛选后没有可用样本，请检查 Dataset/farend_speech 和 meta.csv。")

    return out


def choose_active_segment(x: np.ndarray, segment_samples: int, mode: str):
    """
    从远端语音中裁剪指定长度片段。

    mode="active"：
        选择平均能量最高的连续片段。
    mode="start"：
        从开头截取。
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n < segment_samples:
        raise ValueError(f"音频长度不足：当前 {n} 点，需要 {segment_samples} 点。")

    if mode == "start":
        start = 0
        return x[start:start + segment_samples], start

    if mode != "active":
        raise ValueError(f"未知 SEGMENT_MODE={mode}")

    # 使用约 0.5 秒的步长搜索能量较高的片段。
    hop = max(1, int(0.5 * TARGET_FS))

    best_start = 0
    best_power = -1.0

    max_start = n - segment_samples
    for s in range(0, max_start + 1, hop):
        seg = x[s:s + segment_samples]
        power = float(np.mean(seg ** 2))

        if power > best_power:
            best_power = power
            best_start = s

    return x[best_start:best_start + segment_samples], best_start


def load_farend_segment(row):
    """
    根据 meta 中的 fileid 读取 farend_speech，并裁剪出实验片段。
    """
    fileid = int(row["fileid"])

    farend_path = find_audio_by_fileid(
        FAREND_DIR,
        fileid=fileid,
        prefix="farend_speech",
    )

    x, fs = load_wav_mono(farend_path)
    x = resample_if_needed(x, fs, TARGET_FS)

    if REMOVE_DC:
        x = remove_dc(x)

    segment_samples = int(round(float(SEGMENT_SECONDS) * int(TARGET_FS)))

    x_seg, start = choose_active_segment(
        x=x,
        segment_samples=segment_samples,
        mode=SEGMENT_MODE,
    )

    if PEAK_NORMALIZE_INPUT:
        peak = max(float(np.max(np.abs(x_seg))), float(PEAK_EPS))
        x_seg = x_seg / peak

    return dict(
        x=x_seg.astype(np.float64),
        fs=int(TARGET_FS),
        fileid=fileid,
        farend_path=str(farend_path),
        segment_start_sample=int(start),
        segment_samples=int(segment_samples),
    )


def sample_trials(df, mc_trials: int, seed: int):
    """
    从可用样本中抽取 Monte Carlo trial。

    每个 trial 对应一个不同 fileid。
    如果本地样本数少于 MC_TRIALS，则允许有放回抽样。
    """
    rng = np.random.default_rng(int(seed))
    replace = len(df) < int(mc_trials)

    idx = rng.choice(len(df), size=int(mc_trials), replace=replace)
    return df.iloc[idx].reset_index(drop=True)


# ============================================================
# 非线性系统定义
# ============================================================

def speaker_nonlinearity_soft_sigmoid(
    x: np.ndarray,
    delta_pos: float,
    delta_neg: float,
    drive_gain: float,
):
    """
    软限幅 + sigmoid 非线性。

    这个函数模拟功放/扬声器的 memoryless saturation-type 非线性。

    步骤：
        1. 对输入加 drive_gain，让信号更容易进入饱和区；
        2. 软限幅：
              r_soft = r_max * r / sqrt(r_max^2 + |r|^2)
        3. 构造 zeta：
              zeta = 3/2 * r_soft - 3/10 * r_soft^2
        4. 根据 zeta 正负选择 delta；
        5. sigmoid 非线性：
              y = 1 / (1 + exp(-delta * zeta)) - 1/2
    """
    x = np.asarray(x, dtype=np.float64)

    r = float(drive_gain) * x

    r_max = max(float(np.max(np.abs(r))), 1e-12)

    r_soft = r_max * r / np.sqrt(r_max * r_max + np.abs(r) ** 2)

    zeta = 1.5 * r_soft - 0.3 * (r_soft ** 2)

    delta = np.where(zeta > 0.0, float(delta_pos), float(delta_neg))

    y = 1.0 / (1.0 + np.exp(-delta * zeta)) - 0.5

    return y.astype(np.float64)


def speaker_nonlinearity_soft_clip(x: np.ndarray, drive_gain: float):
    """
    只使用软限幅非线性。

    这个版本比 soft_sigmoid 更简单，适合作为消融测试。
    """
    x = np.asarray(x, dtype=np.float64)

    r = float(drive_gain) * x
    r_max = max(float(np.max(np.abs(r))), 1e-12)

    y = r_max * r / np.sqrt(r_max * r_max + np.abs(r) ** 2)

    return y.astype(np.float64)


def speaker_nonlinearity_tanh(x: np.ndarray, drive_gain: float):
    """
    使用 tanh 饱和非线性。

    tanh 是常见的平滑饱和函数，便于检查算法是否能学习基本饱和曲线。
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.tanh(float(drive_gain) * x)

    return y.astype(np.float64)


def speaker_nonlinearity_poly_clip(x: np.ndarray, drive_gain: float):
    """
    多项式失真 + 限幅。

    y = r - 0.2 r^2 + 0.1 r^3
    然后进行幅值裁剪，避免输出过大。
    """
    x = np.asarray(x, dtype=np.float64)

    r = float(drive_gain) * x
    y = r - 0.2 * (r ** 2) + 0.1 * (r ** 3)

    y = np.clip(y, -1.0, 1.0)

    return y.astype(np.float64)


def generate_nonlinear_target(x: np.ndarray):
    """
    根据配置生成非线性目标信号 d(n)。

    返回：
        d: 非线性目标
    """
    if NONLINEARITY_TYPE == "soft_sigmoid":
        d = speaker_nonlinearity_soft_sigmoid(
            x,
            delta_pos=DELTA_POS,
            delta_neg=DELTA_NEG,
            drive_gain=DRIVE_GAIN,
        )

    elif NONLINEARITY_TYPE == "soft_clip":
        d = speaker_nonlinearity_soft_clip(
            x,
            drive_gain=DRIVE_GAIN,
        )

    elif NONLINEARITY_TYPE == "tanh":
        d = speaker_nonlinearity_tanh(
            x,
            drive_gain=DRIVE_GAIN,
        )

    elif NONLINEARITY_TYPE == "poly_clip":
        d = speaker_nonlinearity_poly_clip(
            x,
            drive_gain=DRIVE_GAIN,
        )

    else:
        raise ValueError(f"未知 NONLINEARITY_TYPE={NONLINEARITY_TYPE}")

    d = np.asarray(d, dtype=np.float64)

    if CENTER_TARGET:
        d = d - np.mean(d)

    if MATCH_TARGET_RMS_TO_INPUT:
        x_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        d_rms = np.sqrt(np.mean(d ** 2)) + 1e-12
        d = d * (x_rms / d_rms)

    return d.astype(np.float64)


# ============================================================
# 算法构造与运行
# ============================================================

def build_algorithm(name: str, filter_order: int):
    """根据算法名称构造算法实例。"""
    if name == "LMS":
        return LMS(filter_order, **ALGO_PARAMS["LMS"])

    if name == "WL-LMS":
        return WLLMS(filter_order, **ALGO_PARAMS["WLLMS"])

    if name == "GH-WL-LMS-Fast":
        # 实验 D3 默认使用向量化快速版。
        # 算法数学形式仍然是 GH-WL-LMS，只是计算 Hermite 特征时进行了向量化。
        return GHWLLMSFast(filter_order, **ALGO_PARAMS["GHWLLMSFast"])

    raise ValueError(f"未知算法：{name}")


def reset_algorithm(algo, seed: int):
    """重置算法状态，并兼容不同算法类的 reset 接口。"""
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
    return_output: bool = True,
):
    """
    对单个算法进行在线训练。

    统计：
        1. Residual MSE
        2. Modeling Gain / ERLE
        3. 可选返回模型输出 y_hat，用于计算 PESQ / STOI

    注意：
        对 D3 来说，d 是模拟非线性目标；
        y_hat 是算法对该非线性目标的估计。
    """
    x = np.asarray(x, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    n = min(len(x), len(d))
    x = x[:n]
    d = d[:n]

    mse_curve = []
    erle_curve = []

    if return_output:
        y_hat = np.zeros(n, dtype=np.float64)
    else:
        y_hat = None

    e2_sum = 0.0
    d2_sum = 0.0
    count = 0

    t0 = time.perf_counter()

    for k, x_vec in enumerate(build_input_vector_stream(x, filter_order)):
        # 先预测，再更新。
        # 这样 y_hat[k] 表示更新前模型对当前样本的在线输出。
        if hasattr(algo, "predict"):
            y = float(algo.predict(x_vec))
            e = float(d[k] - y)
            algo.update(x_vec, float(d[k]))
        else:
            # 如果某些算法没有 predict 接口，则退化为使用 update 返回的误差。
            e = float(algo.update(x_vec, float(d[k])))
            y = float(d[k] - e)

        if return_output:
            y_hat[k] = y

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
        y_hat=y_hat,
    )


# ============================================================
# 绘图函数
# ============================================================

def plot_mean_curve_with_std(
    mean_dict: dict,
    std_dict: dict,
    title: str,
    ylabel: str,
    save_path: Path,
    smooth_window: int = 1,
    y_lim=None,
):
    """绘制均值曲线和标准差阴影。"""
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


def plot_nonlinearity_curve(save_path: Path):
    """
    绘制当前非线性函数的输入输出曲线，方便检查公式是否合理。
    """
    xs = np.linspace(-1.0, 1.0, 1000)
    ys = generate_nonlinear_target(xs)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))

    ax.plot(xs, xs, linestyle="--", label="线性参考 y=x")
    ax.plot(xs, ys, label=f"非线性：{NONLINEARITY_TYPE}")

    ax.set_title("实验 D3：模拟扬声器/功放非线性曲线")
    ax.set_xlabel("输入 x")
    ax.set_ylabel("目标 d=f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

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
    print("实验 D3：真实远端语音上的模拟扬声器/功放非线性建模")
    print("=" * 80)
    print(f"META_CSV              = {META_CSV}")
    print(f"FAREND_DIR            = {FAREND_DIR}")
    print(f"TARGET_FS             = {TARGET_FS}")
    print(f"SEGMENT_SECONDS       = {SEGMENT_SECONDS}")
    print(f"SEGMENT_MODE          = {SEGMENT_MODE}")
    print(f"NONLINEARITY_TYPE     = {NONLINEARITY_TYPE}")
    print(f"DRIVE_GAIN            = {DRIVE_GAIN}")
    print(f"DELTA_POS/NEG         = {DELTA_POS}, {DELTA_NEG}")
    print(f"FILTER_ORDERS         = {FILTER_ORDERS}")
    print(f"MC_TRIALS             = {MC_TRIALS}")
    print(f"ALGO_LIST             = {ALGO_LIST}")
    print(f"RESULT_DIR            = {out_dir}")
    print("=" * 80)

    plot_nonlinearity_curve(out_dir / "fig_exp_d3_nonlinearity_curve.png")

    df = load_and_filter_meta_for_d3()
    trials = sample_trials(df, MC_TRIALS, seed=SEED)

    selected_rows = []
    for _, r in trials.iterrows():
        selected_rows.append({
            "fileid": int(r["fileid"]),
            "split": r.get("split", ""),
            "is_farend_noisy": r.get("is_farend_noisy", ""),
            "is_farend_nonlinear": r.get("is_farend_nonlinear", ""),
            "is_nearend_noisy": r.get("is_nearend_noisy", ""),
        })

    write_csv(selected_rows, out_dir / "exp_d3_selected_fileids.csv")

    trial_rows_all = []
    summary_rows_all = []

    for filter_order in FILTER_ORDERS:
        print("\n" + "=" * 80)
        print(f"开始 D3，filter_order = {filter_order}")
        print("=" * 80)

        curves_mse_by_algo = {name: [] for name in ALGO_LIST}
        curves_erle_by_algo = {name: [] for name in ALGO_LIST}
        final_mse_by_algo = {name: [] for name in ALGO_LIST}
        final_erle_by_algo = {name: [] for name in ALGO_LIST}
        time_by_algo = {name: [] for name in ALGO_LIST}
        pesq_by_algo = {name: [] for name in ALGO_LIST}
        stoi_by_algo = {name: [] for name in ALGO_LIST}

        for trial_id, row in trials.iterrows():
            fileid = int(row["fileid"])

            print("\n" + "-" * 80)
            print(f"[trial {trial_id + 1}/{len(trials)}] fileid={fileid}")
            print("-" * 80)

            try:
                item = load_farend_segment(row)
            except Exception as ex:
                if SKIP_SHORT_CLIPS:
                    print(f"[skip] fileid={fileid}, 原因：{ex}")
                    continue
                raise

            x = item["x"]
            d = generate_nonlinear_target(x)

            print(
                f"  片段长度：{len(x)} samples, "
                f"fs={item['fs']}, "
                f"start={item['segment_start_sample']}, "
                f"x_rms={np.sqrt(np.mean(x ** 2)):.6f}, "
                f"d_rms={np.sqrt(np.mean(d ** 2)):.6f}"
            )

            for algo_name in ALGO_LIST:
                algo = build_algorithm(
                    name=algo_name,
                    filter_order=filter_order,
                )

                reset_algorithm(algo, seed=SEED + trial_id)

                result = run_one_algorithm_online(
                    algo=algo,
                    x=x,
                    d=d,
                    filter_order=filter_order,
                    curve_window=CURVE_WINDOW,
                )

                perceptual_scores = compute_pesq_stoi(
                    ref=d,
                    deg=result["y_hat"],
                    fs=item["fs"],
                    pesq_mode="wb",
                    extended_stoi=False,
                    normalize=True,
                    on_error="nan",
                )
                curves_mse_by_algo[algo_name].append(result["mse_curve"])
                curves_erle_by_algo[algo_name].append(result["erle_curve"])
                final_mse_by_algo[algo_name].append(result["final_mse_db"])
                final_erle_by_algo[algo_name].append(result["final_erle_db"])
                time_by_algo[algo_name].append(result["time_s"])
                pesq_by_algo[algo_name].append(perceptual_scores["PESQ"])
                stoi_by_algo[algo_name].append(perceptual_scores["STOI"])

                trial_row = dict(
                    Experiment="D3",
                    Filter_Order=int(filter_order),
                    Trial=int(trial_id),
                    FileID=int(fileid),
                    Algorithm=algo_name,
                    Nonlinearity=NONLINEARITY_TYPE,
                    Drive_Gain=float(DRIVE_GAIN),
                    Delta_Pos=float(DELTA_POS),
                    Delta_Neg=float(DELTA_NEG),
                    Final_MSE_dB=float(result["final_mse_db"]),
                    Final_ERLE_dB=float(result["final_erle_db"]),
                    Time_s=float(result["time_s"]),
                    Segment_Start_Sample=int(item["segment_start_sample"]),
                    Segment_Samples=int(item["segment_samples"]),
                    Farend_Path=item["farend_path"],
                    PESQ=float(perceptual_scores["PESQ"]),
                    STOI=float(perceptual_scores["STOI"]),
                )
                trial_rows_all.append(trial_row)

                print(
                    f"  {algo_name:14s} | "
                    f"MSE={result['final_mse_db']:.3f} dB | "
                    f"Gain={result['final_erle_db']:.3f} dB | "
                    f"PESQ={perceptual_scores['PESQ']:.3f} | "
                    f"STOI={perceptual_scores['STOI']:.3f} | "
                    f"time={result['time_s']:.3f}s"
                )

            write_csv(
                trial_rows_all,
                out_dir / "exp_d3_trial_results_long.csv",
            )

        mean_mse = {}
        std_mse = {}
        mean_erle = {}
        std_erle = {}

        for algo_name in ALGO_LIST:
            mean_mse[algo_name], std_mse[algo_name] = average_curves(
                curves_mse_by_algo[algo_name]
            )
            mean_erle[algo_name], std_erle[algo_name] = average_curves(
                curves_erle_by_algo[algo_name]
            )

            mse_vals = np.asarray(final_mse_by_algo[algo_name], dtype=float)
            erle_vals = np.asarray(final_erle_by_algo[algo_name], dtype=float)
            time_vals = np.asarray(time_by_algo[algo_name], dtype=float)
            pesq_vals = np.asarray(pesq_by_algo[algo_name], dtype=float)
            stoi_vals = np.asarray(stoi_by_algo[algo_name], dtype=float)

            summary_rows_all.append(dict(
                Experiment="D3",
                Filter_Order=int(filter_order),
                Algorithm=algo_name,
                Nonlinearity=NONLINEARITY_TYPE,
                Drive_Gain=float(DRIVE_GAIN),
                Delta_Pos=float(DELTA_POS),
                Delta_Neg=float(DELTA_NEG),
                Num_Trials=int(len(mse_vals)),
                Mean_Final_MSE_dB=float(np.nanmean(mse_vals)) if len(mse_vals) else np.nan,
                Std_Final_MSE_dB=float(np.nanstd(mse_vals)) if len(mse_vals) else np.nan,
                Median_Final_MSE_dB=float(np.nanmedian(mse_vals)) if len(mse_vals) else np.nan,
                Mean_Final_ERLE_dB=float(np.nanmean(erle_vals)) if len(erle_vals) else np.nan,
                Std_Final_ERLE_dB=float(np.nanstd(erle_vals)) if len(erle_vals) else np.nan,
                Median_Final_ERLE_dB=float(np.nanmedian(erle_vals)) if len(erle_vals) else np.nan,
                Mean_Time_s=float(np.nanmean(time_vals)) if len(time_vals) else np.nan,
                Mean_PESQ=float(np.nanmean(pesq_vals)) if len(pesq_vals) else np.nan,
                Std_PESQ=float(np.nanstd(pesq_vals)) if len(pesq_vals) else np.nan,
                Median_PESQ=float(np.nanmedian(pesq_vals)) if len(pesq_vals) else np.nan,
                Mean_STOI=float(np.nanmean(stoi_vals)) if len(stoi_vals) else np.nan,
                Std_STOI=float(np.nanstd(stoi_vals)) if len(stoi_vals) else np.nan,
                Median_STOI=float(np.nanmedian(stoi_vals)) if len(stoi_vals) else np.nan,
            ))

        curves_to_save = {}
        for algo_name in ALGO_LIST:
            curves_to_save[f"{algo_name}_MSE_dB"] = mean_mse[algo_name]
            curves_to_save[f"{algo_name}_ERLE_dB"] = mean_erle[algo_name]

        save_curves_csv(
            curves_to_save,
            out_dir / f"exp_d3_mean_curves_p{filter_order}.csv",
        )

        plot_mean_curve_with_std(
            mean_dict=mean_mse,
            std_dict=std_mse,
            title=f"实验 D3：模拟扬声器非线性建模 MSE，p={filter_order}",
            ylabel="Residual MSE (dB)",
            save_path=out_dir / f"fig_exp_d3_mse_p{filter_order}.png",
            smooth_window=PLOT.get("smooth_window", 3),
            y_lim=PLOT.get("mse_ylim", None),
        )

        plot_mean_curve_with_std(
            mean_dict=mean_erle,
            std_dict=std_erle,
            title=f"实验 D3：模拟扬声器非线性建模增益，p={filter_order}",
            ylabel="Modeling Gain / ERLE (dB)",
            save_path=out_dir / f"fig_exp_d3_erle_p{filter_order}.png",
            smooth_window=PLOT.get("smooth_window", 3),
            y_lim=PLOT.get("erle_ylim", None),
        )

        write_csv(
            summary_rows_all,
            out_dir / "exp_d3_summary.csv",
        )

    print("\n" + "=" * 80)
    print("实验 D3 完成")
    print(f"结果保存到：{out_dir}")
    print("=" * 80)

    print("\n[Summary]")
    for row in summary_rows_all:
        print(
            f"p={row['Filter_Order']:4d} | "
            f"{row['Algorithm']:12s} | "
            f"MSE={row['Mean_Final_MSE_dB']:.3f}±{row['Std_Final_MSE_dB']:.3f} dB | "
            f"Gain={row['Mean_Final_ERLE_dB']:.3f}±{row['Std_Final_ERLE_dB']:.3f} dB | "
            f"time={row['Mean_Time_s']:.3f}s"
        )


if __name__ == "__main__":
    main()