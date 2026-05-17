"""
实验三：非线性系统辨识 主入口
对应论文 Section V-C，复现图 14、15、16 以及表 III

运行方式：
    cd wlnf_project
    python experiments/run_exp3.py

可选参数（修改 configs/exp3_config.py 中的 MC_TRIALS 等）
"""

import sys
import csv
import time
from pathlib import Path
from datetime import datetime

# ── 路径设置 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── 导入配置 ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / 'configs'))
from configs.exp3_config import DATASET, ALGO_PARAMS, MC_TRIALS, PLOT, RESULT_DIR, ALGO_LIST

from scenarios.nonlinear_id import run_stationary, run_nonstationary
from utils.plotting import plot_learning_curves, plot_table

# Use a timestamped subdirectory inside the configured RESULT_DIR so multiple
# runs don't overwrite each other. Example: results/exp3/20260517_142530
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = Path(RESULT_DIR) / timestamp
RESULT_PATH.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：保存 CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_curves_csv(curves: dict, path: Path):
    """将多算法学习曲线保存为 CSV（列 = 算法名）"""
    names = list(curves.keys())
    n = max(len(v) for v in curves.values())
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['iteration'] + names)
        for i in range(n):
            row = [i + 1] + [
                f'{curves[name][i]:.6f}' if i < len(curves[name]) else ''
                for name in names
            ]
            writer.writerow(row)
    print(f"[csv]  已保存: {path}")


def save_summary_csv(ss_mse: dict, avg_time: dict, path: Path):
    """将稳态 MSE 和运行时间保存为 CSV"""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'SS_MSE(dB)', 'Time(s)'])
        for name in ss_mse:
            writer.writerow([name, f'{ss_mse[name]:.3f}', f'{avg_time[name]:.4f}'])
    print(f"[csv]  已保存: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 子实验 A：平稳场景（σ²=0.0036 和 σ²=0.01）
# ─────────────────────────────────────────────────────────────────────────────

def run_stationary_experiments():
    print("\n" + "=" * 60)
    print("子实验 A：平稳非线性系统辨识")
    print("=" * 60)

    # 收集两组噪声方差的稳态 MSE，用于打印汇总表
    summary_rows_mse  = []   # 每行：{Algorithm, MSE(σ²=0.0036), MSE(σ²=0.01)}
    summary_rows_time = []

    # 允许在配置中指定要比较的算法顺序与子集
    algo_names = list(ALGO_LIST)
    ss_mse_all  = {n: {} for n in algo_names}
    avg_time_all= {n: {} for n in algo_names}

    for nv in DATASET['noise_vars']:
        label = f"sigma2_{str(nv).replace('.', '')}"
        print(f"\n─── 噪声方差 σ² = {nv} ───")

        t_start = time.time()
        avg_curves, ss_mse, avg_time = run_stationary(
            noise_var   = nv,
            params      = ALGO_PARAMS,
            dataset_cfg = DATASET,
            n_trials    = MC_TRIALS,
            verbose     = True,
            algo_list   = algo_names,
        )
        print(f"    耗时 {time.time() - t_start:.1f}s")

        # 保存曲线
        save_curves_csv(
            avg_curves,
            RESULT_PATH / f'stationary_{label}_curves.csv'
        )
        save_summary_csv(
            ss_mse, avg_time,
            RESULT_PATH / f'stationary_{label}_summary.csv'
        )

        # 绘制学习曲线
        plot_learning_curves(
            results      = avg_curves,
            title        = f'Nonlinear System ID (stationary, σ²={nv})',
            save_path    = RESULT_PATH / f'fig_stationary_{label}.png',
            smooth_window= PLOT['smooth_window'],
            y_lim        = PLOT['y_lim_stationary'],
        )

        # 记录稳态 MSE 与时间
        for name in algo_names:
            ss_mse_all[name][nv]   = ss_mse.get(name, float('nan'))
            avg_time_all[name][nv] = avg_time.get(name, float('nan'))

        # 打印本轮结果
        print(f"\n  {'Algorithm':<12} {'SS MSE (dB)':>14} {'Time (s)':>12}")
        print(f"  {'-'*40}")
        for name in algo_names:
            print(f"  {name:<12} {ss_mse.get(name, float('nan')):>14.3f} "
                  f"{avg_time.get(name, float('nan')):>12.4f}")

    # ── 打印对应论文表 III 格式的汇总 ──
    # 打印论文表格样式的汇总（若只选了一个噪声方差则只显示那个）
    noise_vars = DATASET['noise_vars']
    if len(noise_vars) == 0:
        print("No noise vars configured.")
    elif len(noise_vars) == 1:
        nv1 = noise_vars[0]
        print(f"\n  {'Algorithm':<12} {'MSE(σ²={:.4f})'.format(nv1):>20} {'Time':>8}")
        print(f"  {'-'*40}")
        for name in algo_names:
            print(f"  {name:<12} {ss_mse_all[name][nv1]:>20.3f} {avg_time_all[name][nv1]:>8.4f}")
    else:
        nv1, nv2 = noise_vars[:2]
        print(f"\n  {'Algorithm':<12} {'MSE(σ²={:.4f})'.format(nv1):>20} "
              f"{'Time':>8}  {'MSE(σ²={:.2f})'.format(nv2):>18} {'Time':>8}")
        print(f"  {'-'*72}")
        for name in algo_names:
            print(f"  {name:<12} "
                  f"{ss_mse_all[name][nv1]:>20.3f} "
                  f"{avg_time_all[name][nv1]:>8.4f}  "
                  f"{ss_mse_all[name][nv2]:>18.3f} "
                  f"{avg_time_all[name][nv2]:>8.4f}")

    # 保存表格图片
    # 为表格准备行（根据配置的噪声方差数量调整列）
    table_rows = []
    if len(noise_vars) <= 1:
        nv1 = noise_vars[0]
        for name in algo_names:
            table_rows.append({
                'Algorithm': name,
                f'MSE(σ²={nv1})': f"{ss_mse_all[name][nv1]:.3f}",
                f'Time(s)': f"{avg_time_all[name][nv1]:.4f}",
            })
    else:
        nv1, nv2 = noise_vars[:2]
        for name in algo_names:
            table_rows.append({
                'Algorithm': name,
                f'MSE(σ²={nv1})': f"{ss_mse_all[name][nv1]:.3f}",
                f'Time₁(s)': f"{avg_time_all[name][nv1]:.4f}",
                f'MSE(σ²={nv2})': f"{ss_mse_all[name][nv2]:.3f}",
                f'Time₂(s)': f"{avg_time_all[name][nv2]:.4f}",
            })
    plot_table(table_rows, RESULT_PATH / 'table_stationary.png')

    return ss_mse_all, avg_time_all


# ─────────────────────────────────────────────────────────────────────────────
# 子实验 B：非平稳场景（系数突变）
# ─────────────────────────────────────────────────────────────────────────────

def run_nonstationary_experiment():
    print("\n" + "=" * 60)
    print("子实验 B：非平稳非线性系统辨识（系数在 k=2001 突变）")
    print("=" * 60)

    nv = DATASET['noise_var_ns']
    print(f"  噪声方差 σ² = {nv}\n")

    # 使用配置中指定的算法子集
    algo_names = list(ALGO_LIST)

    t_start = time.time()
    avg_curves, ss_mse, avg_time = run_nonstationary(
        noise_var   = nv,
        params      = ALGO_PARAMS,
        dataset_cfg = DATASET,
        n_trials    = MC_TRIALS,
        verbose     = True,
        algo_list   = algo_names,
    )
    print(f"  耗时 {time.time() - t_start:.1f}s")

    label = f"sigma2_{str(nv).replace('.', '')}"

    save_curves_csv(avg_curves, RESULT_PATH / f'nonstationary_{label}_curves.csv')
    save_summary_csv(ss_mse, avg_time, RESULT_PATH / f'nonstationary_{label}_summary.csv')

    plot_learning_curves(
        results      = avg_curves,
        title        = f'Nonlinear System ID (non-stationary, σ²={nv})',
        save_path    = RESULT_PATH / f'fig_nonstationary_{label}.png',
        smooth_window= PLOT['smooth_window'],
        y_lim        = PLOT['y_lim_ns'],
    )

    algo_names = list(ALGO_LIST)
    print(f"\n  {'Algorithm':<12} {'SS MSE (dB)':>14} {'Time (s)':>12}")
    print(f"  {'-'*40}")
    for name in algo_names:
        print(f"  {name:<12} {ss_mse.get(name, float('nan')):>14.3f} "
              f"{avg_time.get(name, float('nan')):>12.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("▶  实验三：非线性系统辨识")
    print(f"   结果保存路径: {RESULT_PATH.resolve()}")
    print(f"   Monte Carlo 次数: {MC_TRIALS}")

    # 子实验 A：平稳场景（对应论文图 14、15 和表 III）
    run_stationary_experiments()

    # 子实验 B：非平稳场景（对应论文图 16）
    # run_nonstationary_experiment()

    print("\n✓  实验三全部完成，结果已保存至:", RESULT_PATH.resolve())


if __name__ == '__main__':
    main()
