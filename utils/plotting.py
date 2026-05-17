"""
绘图工具模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
})

# 每个算法固定颜色 / 线型，方便多图对比
ALGO_STYLES = {
    'LMS':      {'color': 'gray',     'linestyle': '-',  'label': 'LMS'},
    'KLMS':     {'color': 'green',    'linestyle': '--', 'label': 'KLMS'},
    'KRLS':     {'color': 'blue',     'linestyle': '-.',  'label': 'KRLS'},
    'RFFMC':    {'color': 'orange',   'linestyle': ':',  'label': 'RFFMC'},
    'NKRGMC':   {'color': 'purple',   'linestyle': '--', 'label': 'NKRGMC'},
    'WL-LMS':   {'color': 'red',      'linestyle': '-',  'label': 'WL-LMS'},
    'WL-RLS':   {'color': 'black',    'linestyle': '-',  'label': 'WL-RLS'},
}


def smooth(curve: np.ndarray, window: int = 30) -> np.ndarray:
    """简单滑动平均平滑（用于可视化）"""
    if window <= 1:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode='same')


def plot_learning_curves(
    results: dict,          # {algo_name: mse_db_array}
    title: str,
    save_path: Path,
    smooth_window: int = 50,
    y_lim: tuple = None,
):
    """
    绘制多算法学习曲线（MSE dB vs iteration）
    results : dict，key 为算法名，value 为 shape (N,) 的 MSE(dB) 数组
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name, curve in results.items():
        style = ALGO_STYLES.get(name, {'color': 'black', 'linestyle': '-', 'label': name})
        smoothed = smooth(curve, smooth_window)
        ax.plot(smoothed, color=style['color'],
                linestyle=style['linestyle'], label=style['label'])

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
    print(f"[plot] 已保存: {save_path}")


def plot_table(
    rows: list,             # list of dict: {algo, mse_1, time_1, mse_2, time_2}
    save_path: Path,
    col_headers: list = None,
):
    """
    将稳态 MSE / 运行时间汇总为表格图片
    rows 示例:
        [{'Algorithm': 'LMS', 'MSE(σ²=0.0036)': '-27.1', 'Time': '0.002', ...}]
    """
    if col_headers is None:
        col_headers = list(rows[0].keys())

    cell_text = [[str(row[c]) for c in col_headers] for row in rows]

    fig, ax = plt.subplots(figsize=(len(col_headers) * 1.8, len(rows) * 0.5 + 0.8))
    ax.axis('off')
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_headers,
        cellLoc='center',
        loc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] 已保存: {save_path}")
