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


# 新代码（utils/plotting.py 中的 smooth 函数）
def smooth(curve, window=30):
    if window <= 1:
        return curve
    pad = window // 2
    padded = np.pad(curve, pad_width=pad, mode='edge')  # 用边缘值填充，而非补零
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(curve)]


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


# 新增：绘制频谱比较（input, target, pred, residual）
def _compute_imd_freqs(f1, f2):
    """Return a sorted list of relevant frequencies (normalized cycles/sample) to mark.
    Includes fundamentals, difference, sum, and 3rd-order IMD: 2f1-f2, 2f2-f1.
    """
    freqs = [f1, f2, abs(f2 - f1), f1 + f2, 2 * f1 - f2, 2 * f2 - f1]
    # keep unique and within [0, 0.5]
    uniq = sorted(set([round(float(f), 8) for f in freqs if 0.0 <= f <= 0.5]))
    return uniq


def plot_spectra_comparison(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    algo_name: str,
    save_path: Path,
    fs: float = 1.0,
    f1: float = None,
    f2: float = None,
    nfft: int = None,
    show: bool = False,
):
    """
    Plot four spectra (input x, target y, predicted y, residual) in a 2x2 grid and save to file.

    Parameters
    - x, y, y_pred: 1-D time series (should be same length)
    - algo_name: name used to style the plots
    - save_path: Path to save the figure
    - fs: sampling frequency (default 1.0 -> normalized frequency per sample)
    - f1, f2: optional fundamental frequencies (normalized) to mark IMD positions
    - nfft: FFT length (defaults to next power of two >= len(x))
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    x = _np.asarray(x)
    y = _np.asarray(y)
    y_pred = _np.asarray(y_pred)
    if x.size == 0 or y.size == 0 or y_pred.size == 0:
        print(f"[plot] empty signals, skipping spectra for {algo_name}")
        return

    N = x.size
    if nfft is None:
        # choose nfft as next power of two for decent resolution
        nfft = 1 << (N - 1).bit_length()
    # compute spectra (rfft)
    Xf = _np.fft.rfft(x, n=nfft)
    Yf = _np.fft.rfft(y, n=nfft)
    Pf = _np.fft.rfft(y - y_pred, n=nfft)
    Ypf = _np.fft.rfft(y_pred, n=nfft)

    freqs = _np.fft.rfftfreq(nfft, d=1.0 / fs)

    # magnitude in dB (20*log10)
    eps = 1e-20
    Xdb = 20.0 * _np.log10(_np.abs(Xf) + eps)
    Ydb = 20.0 * _np.log10(_np.abs(Yf) + eps)
    Ypdb = 20.0 * _np.log10(_np.abs(Ypf) + eps)
    Rdb = 20.0 * _np.log10(_np.abs(Pf) + eps)

    style = ALGO_STYLES.get(algo_name, {'color': 'black', 'linestyle': '-', 'label': algo_name})

    fig, axes = _plt.subplots(2, 2, figsize=(9, 6))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # input x: gray thin solid + semi-transparent
    ax1.plot(freqs, Xdb, color='gray', linestyle='-', linewidth=0.8, alpha=0.4, label='input x')
    ax1.set_title('Input x spectrum')
    ax1.set_xlabel('Freq (cycles/sample)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True, alpha=0.3)

    # target y: black solid standard width
    ax2.plot(freqs, Ydb, color='k', linestyle='-', linewidth=1.4, label='target y')
    ax2.set_title('Target y spectrum')
    ax2.set_xlabel('Freq (cycles/sample)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.grid(True, alpha=0.3)

    # predicted y: bright orange thick dashed
    ax3.plot(freqs, Ypdb, color='tab:orange', linestyle='--', linewidth=2.0, label='predicted y')
    ax3.set_title('Predicted y spectrum')
    ax3.set_xlabel('Freq (cycles/sample)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.grid(True, alpha=0.3)

    # residual: red thin dotted
    ax4.plot(freqs, Rdb, color='tab:red', linestyle=':', linewidth=0.8, alpha=0.9, label='residual')
    ax4.set_title('Residual (y - y_pred) spectrum')
    ax4.set_xlabel('Freq (cycles/sample)')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.grid(True, alpha=0.3)

    # Mark IMD frequencies if provided
    if f1 is not None and f2 is not None:
        imd_freqs = _compute_imd_freqs(f1, f2)
        for ax in (ax1, ax2, ax3, ax4):
            for ff in imd_freqs:
                ax.axvline(ff, color='k', linestyle='--', linewidth=1.0, alpha=0.8)
            # annotate fundamentals
            ax.set_xlim(0, min(0.5, freqs.max()))

    # Show legend in a compact manner on the top-right subplot
    ax3.legend(loc='upper right')

    fig.suptitle(f'Spectra comparison — {algo_name}')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    # If show True, display interactive window first to allow zooming
    if show:
        _plt.show()

    fig.savefig(save_path, dpi=150)
    _plt.close(fig)
    print(f"[plot] saved spectra: {save_path}")


def plot_spectra_grid(
    algo_signals: dict,
    save_path: Path,
    fs: float = 1.0,
    f1: float = None,
    f2: float = None,
    nfft: int = None,
    figsize_per_plot: tuple = (4.5, 3.2),
    show: bool = False,
):
    """
    Plot multiple algorithms in a grid where each subplot shows four spectra:
    input x, target y, predicted y, residual (y - y_pred).

    Parameters
    - algo_signals: dict mapping algo_name -> (x, y, y_pred, residual) (1D arrays)
    - save_path: Path to save the combined figure
    - fs: sampling frequency (default 1.0 -> normalized cycles/sample)
    - f1, f2: optional fundamentals to mark
    - nfft: FFT length (if None, chosen per signal length)
    - figsize_per_plot: size per subplot (width, height)
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    names = list(algo_signals.keys())
    n = len(names)
    if n == 0:
        print("[plot] no algorithms to plot")
        return

    # grid layout: try square-ish
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig_w = figsize_per_plot[0] * cols
    fig_h = figsize_per_plot[1] * rows
    fig, axes = _plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.reshape(-1) if isinstance(axes, _np.ndarray) else [axes]

    # IMD freqs
    imd_freqs = _compute_imd_freqs(f1, f2) if (f1 is not None and f2 is not None) else []

    for idx, name in enumerate(names):
        ax = axes[idx]
        x, y, y_pred, res = algo_signals[name]
        # choose nfft if provided else next pow2 of length
        L = len(x)
        if nfft is None:
            nfft_use = 1 << (L - 1).bit_length()
        else:
            nfft_use = nfft

        Xf = _np.fft.rfft(x, n=nfft_use)
        Yf = _np.fft.rfft(y, n=nfft_use)
        Ypf = _np.fft.rfft(y_pred, n=nfft_use)
        Pf = _np.fft.rfft(res, n=nfft_use)
        freqs = _np.fft.rfftfreq(nfft_use, d=1.0 / fs)
        eps = 1e-20
        Xdb = 20.0 * _np.log10(_np.abs(Xf) + eps)
        Ydb = 20.0 * _np.log10(_np.abs(Yf) + eps)
        Ypdb = 20.0 * _np.log10(_np.abs(Ypf) + eps)
        Rdb = 20.0 * _np.log10(_np.abs(Pf) + eps)
        # clamp floor for plotting
        floor = -120.0
        Xdb = _np.maximum(Xdb, floor)
        Ydb = _np.maximum(Ydb, floor)
        Ypdb = _np.maximum(Ypdb, floor)
        Rdb = _np.maximum(Rdb, floor)

        # apply requested styles: input grey thin semi-transparent; target black solid; predicted orange thick dashed; residual red thin dotted
        ax.plot(freqs, Xdb, label='input x', color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
        ax.plot(freqs, Ydb, label='target y', color='k', linestyle='-', linewidth=1.4)
        ax.plot(freqs, Ypdb, label='predicted y', color='tab:orange', linestyle='--', linewidth=2.0)
        ax.plot(freqs, Rdb, label='residual', color='tab:red', linestyle=':', linewidth=0.8, alpha=0.9)

        # mark fundamental and IMD bins with small markers for easy identification
        mark_freqs = []
        if f1 is not None:
            mark_freqs.append(f1)
        if f2 is not None:
            mark_freqs.append(f2)
        mark_freqs += imd_freqs
        for ff in mark_freqs:
            if 0.0 <= ff <= freqs.max():
                # find nearest bin
                idx_bin = int(_np.argmin(_np.abs(freqs - ff)))
                ax.plot(freqs[idx_bin], Xdb[idx_bin], 'o', color='tab:blue', markersize=4, alpha=0.8)
                ax.plot(freqs[idx_bin], Ydb[idx_bin], 's', color='k', markersize=3, alpha=0.8)
                ax.plot(freqs[idx_bin], Ypdb[idx_bin], 's', color='tab:orange', markersize=3, alpha=0.8)
                ax.plot(freqs[idx_bin], Rdb[idx_bin], 'o', color='tab:red', markersize=4, alpha=0.8)

        ax.set_title(name)
        ax.set_xlim(0, min(0.5, freqs.max()))
        ax.set_xlabel('Freq (cycles/sample)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, alpha=0.25)
        ax.legend(loc='upper right', fontsize=8)

    # hide any unused axes
    for j in range(len(names), rows * cols):
        ax = axes[j]
        ax.axis('off')

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        _plt.show()
    _plt.close(fig)
    print(f"[plot] saved combined spectra figure: {save_path}")


def plot_residuals_grid(
    algo_residuals: dict,
    save_path: Path,
    fs: float = 1.0,
    f1: float = None,
    f2: float = None,
    nfft: int = None,
    figsize_per_plot: tuple = (4.5, 3.2),
    show: bool = False,
):
    """
    Plot residual spectra for multiple algorithms in a grid (one subplot per algorithm).

    Parameters
    - algo_residuals: dict mapping algo_name -> residual_signal (1D array) or algo_name -> (x,y,y_pred,res)
    - save_path: Path to save the combined figure
    - fs: sampling frequency (default 1.0 -> normalized cycles/sample)
    - f1, f2: optional fundamentals to mark IMD positions
    - nfft: FFT length; if None uses length of each signal (or next pow2)
    - figsize_per_plot: size per subplot
    - show: if True, display interactive window before saving
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    names = list(algo_residuals.keys())
    n = len(names)
    if n == 0:
        print('[plot] no residuals to plot')
        return

    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig_w = figsize_per_plot[0] * cols
    fig_h = figsize_per_plot[1] * rows
    fig, axes = _plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.reshape(-1) if isinstance(axes, _np.ndarray) else [axes]

    imd_freqs = _compute_imd_freqs(f1, f2) if (f1 is not None and f2 is not None) else []

    floor = -120.0
    for idx, name in enumerate(names):
        ax = axes[idx]
        val = algo_residuals[name]
        # support value either residual array or tuple (x,y,y_pred,res)
        if isinstance(val, (list, tuple)) and len(val) >= 4:
            res = _np.asarray(val[3])
        else:
            res = _np.asarray(val)

        L = res.size
        if L == 0:
            print(f"[plot] empty residual for {name}, skipping")
            ax.axis('off')
            continue

        nfft_use = nfft if nfft is not None else (1 << (L - 1).bit_length())
        Rf = _np.fft.rfft(res, n=nfft_use)
        freqs = _np.fft.rfftfreq(nfft_use, d=1.0 / fs)
        Rdb = 20.0 * _np.log10(_np.abs(Rf) + 1e-20)
        Rdb = _np.maximum(Rdb, floor)

        # plot residual spectrum with thin dotted red (as requested)
        ax.plot(freqs, Rdb, color='tab:red', linestyle=':', linewidth=0.9, alpha=0.95)
        ax.set_title(name)
        ax.set_xlim(0, min(0.5, freqs.max()))
        ax.set_xlabel('Freq (cycles/sample)')
        ax.set_ylabel('Magnitude (dB)')
        ax.grid(True, alpha=0.25)

        # mark IMD freqs
        for ff in imd_freqs:
            if 0.0 <= ff <= freqs.max():
                idx_bin = int(_np.argmin(_np.abs(freqs - ff)))
                ax.axvline(ff, color='k', linestyle='--', linewidth=1.0, alpha=0.7)
                # small marker at residual magnitude
                ax.plot(freqs[idx_bin], Rdb[idx_bin], 'o', color='tab:red', markersize=4, alpha=0.8)

    # hide unused axes
    for j in range(len(names), rows * cols):
        axes[j].axis('off')

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if show:
        _plt.show()

    fig.savefig(save_path, dpi=150)
    _plt.close(fig)
    print(f"[plot] saved residuals grid figure: {save_path}")


def plot_speech_spectra_grid(
    algo_signals: dict,
    save_path: Path,
    fs: float = 16000.0,
    nfft: int = None,
    floor_db: float = -120.0,
    figsize_per_plot: tuple = (5.0, 3.4),
    show: bool = False,
):
    """
    Plot broadband spectra for speech input.

    Each subplot corresponds to one algorithm and shows:
        input x, target y, predicted y, residual y - y_pred

    This is different from the sine-input spectrum plot:
    speech has no fixed f1/f2 or discrete IMD lines, so we do not mark IMD frequencies.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    names = list(algo_signals.keys())
    n_algos = len(names)
    if n_algos == 0:
        print("[plot] no algorithms to plot for speech spectra")
        return

    cols = min(3, n_algos)
    rows = (n_algos + cols - 1) // cols

    fig_w = figsize_per_plot[0] * cols
    fig_h = figsize_per_plot[1] * rows
    fig, axes = _plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.reshape(-1) if isinstance(axes, _np.ndarray) else [axes]

    for idx, name in enumerate(names):
        ax = axes[idx]

        x, y, y_pred, residual = algo_signals[name]
        x = _np.asarray(x, dtype=float)
        y = _np.asarray(y, dtype=float)
        y_pred = _np.asarray(y_pred, dtype=float)
        residual = _np.asarray(residual, dtype=float)

        L = min(len(x), len(y), len(y_pred), len(residual))
        if L <= 0:
            ax.axis("off")
            print(f"[plot] empty speech signals for {name}, skipping")
            continue

        x = x[:L]
        y = y[:L]
        y_pred = y_pred[:L]
        residual = residual[:L]

        # Remove DC to avoid a huge zero-frequency spike.
        x = x - _np.mean(x)
        y = y - _np.mean(y)
        y_pred = y_pred - _np.mean(y_pred)
        residual = residual - _np.mean(residual)

        if nfft is None:
            nfft_use = 1 << (L - 1).bit_length()
        else:
            nfft_use = int(nfft)

        # Hann window is important for speech because it is not periodic in the selected segment.
        win = _np.hanning(L)
        win_norm = _np.sum(win) + 1e-12

        def spectrum_db(sig):
            spec = _np.fft.rfft(sig * win, n=nfft_use)
            mag = _np.abs(spec) / win_norm
            db = 20.0 * _np.log10(mag + 1e-20)
            return _np.maximum(db, floor_db)

        Xdb = spectrum_db(x)
        Ydb = spectrum_db(y)
        Ypdb = spectrum_db(y_pred)
        Rdb = spectrum_db(residual)

        freqs = _np.fft.rfftfreq(nfft_use, d=1.0 / fs)

        ax.plot(freqs, Xdb, label="input x", color="gray", linestyle="-", linewidth=0.8, alpha=0.45)
        ax.plot(freqs, Ydb, label="target y", color="k", linestyle="-", linewidth=1.4)
        ax.plot(freqs, Ypdb, label="predicted y", color="tab:orange", linestyle="--", linewidth=2.0)
        ax.plot(freqs, Rdb, label="residual", color="tab:red", linestyle=":", linewidth=0.9, alpha=0.95)

        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

        # Speech at 16 kHz normally has useful content below 8 kHz.
        ax.set_xlim(0, fs / 2)

    for j in range(n_algos, rows * cols):
        axes[j].axis("off")

    fig.suptitle("Speech input spectra", y=1.02)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if show:
        _plt.show()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    _plt.close(fig)
    print(f"[plot] saved speech spectra grid: {save_path}")


def plot_speech_residual_spectra_grid(
    algo_signals: dict,
    save_path: Path,
    fs: float = 16000.0,
    nfft: int = None,
    floor_db: float = -120.0,
    figsize_per_plot: tuple = (5.0, 3.4),
    show: bool = False,
):
    """
    Plot only residual spectra for speech input.

    This figure is often clearer than plotting x/y/y_pred/residual together.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt

    names = list(algo_signals.keys())
    n_algos = len(names)
    if n_algos == 0:
        print("[plot] no algorithms to plot for speech residual spectra")
        return

    cols = min(3, n_algos)
    rows = (n_algos + cols - 1) // cols

    fig_w = figsize_per_plot[0] * cols
    fig_h = figsize_per_plot[1] * rows
    fig, axes = _plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = axes.reshape(-1) if isinstance(axes, _np.ndarray) else [axes]

    for idx, name in enumerate(names):
        ax = axes[idx]

        _, _, _, residual = algo_signals[name]
        residual = _np.asarray(residual, dtype=float)

        L = len(residual)
        if L <= 0:
            ax.axis("off")
            print(f"[plot] empty residual for {name}, skipping")
            continue

        residual = residual - _np.mean(residual)

        if nfft is None:
            nfft_use = 1 << (L - 1).bit_length()
        else:
            nfft_use = int(nfft)

        win = _np.hanning(L)
        win_norm = _np.sum(win) + 1e-12

        spec = _np.fft.rfft(residual * win, n=nfft_use)
        mag = _np.abs(spec) / win_norm
        Rdb = 20.0 * _np.log10(mag + 1e-20)
        Rdb = _np.maximum(Rdb, floor_db)

        freqs = _np.fft.rfftfreq(nfft_use, d=1.0 / fs)

        ax.plot(freqs, Rdb, color="tab:red", linestyle=":", linewidth=0.9, alpha=0.95)
        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Residual magnitude (dB)")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, fs / 2)

    for j in range(n_algos, rows * cols):
        axes[j].axis("off")

    fig.suptitle("Speech residual spectra", y=1.02)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if show:
        _plt.show()

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    _plt.close(fig)
    print(f"[plot] saved speech residual spectra grid: {save_path}")
