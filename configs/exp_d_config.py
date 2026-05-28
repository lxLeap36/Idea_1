"""
实验 D 配置文件：
AEC Challenge 合成数据集上的真实语音非线性回声建模实验。

本实验不修改实验 A/B/C 的任何配置。
第一版只做远端单讲场景：
    输入 x(n)：farend_speech
    目标 d(n)：echo_signal

筛选条件：
    is_farend_nonlinear = 1
    is_farend_noisy = 0
    is_nearend_noisy = 0
"""

import os
from pathlib import Path


# ============================================================
# 项目路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

# 默认数据目录（项目内 Data/）
# 如果你的数据存放在项目外（例如 D:/Datasets/...），请修改为绝对路径。
# 当前工作空间用户数据位于:
#   D:\Datasets\AEC-Challenge\datasets\synthetic
DATA_ROOT = Path(r"D:\Datasets\AEC-Challenge\datasets\synthetic")

META_CSV = DATA_ROOT / "meta.csv"
FAREND_DIR = DATA_ROOT / "farend_speech"
ECHO_DIR = DATA_ROOT / "echo_signal"
NEAREND_DIR = DATA_ROOT / "nearend_speech"
MIC_DIR = DATA_ROOT / "nearend_mic_signal"


# ============================================================
# 数据筛选条件
# ============================================================

# 只选含非线性远端回声的样本
REQUIRE_FAREND_NONLINEAR = 1

# 不使用远端噪声
REQUIRE_FAREND_NOISY = 0

# 不使用近端噪声
REQUIRE_NEAREND_NOISY = 0

# 是否限定 split。
# 可以设为 "train" / "test" / None。
# 如果你不确定官方 split 怎么划分，先用 None。
SPLIT = None


# ============================================================
# 音频与片段设置
# ============================================================

# 第一版建议先统一到 16 kHz，计算量比 48 kHz 小很多。
TARGET_FS = 16000

# 每条 trial 使用 10 秒。
SEGMENT_SECONDS = 3.0

# 每条音频内部如何裁剪片段：
#   "active"：选择远端语音能量较高的连续片段
#   "start" ：从开头截取
SEGMENT_MODE = "active"

# 如果音频比目标长度短，是否跳过。
SKIP_SHORT_CLIPS = True

# 是否去直流分量。
REMOVE_DC = True

# 是否做公共幅度归一化。
# 注意：这里用 farend 和 echo 的共同峰值归一化，避免破坏二者相对比例。
PEAK_NORMALIZE = True

# 防止过小峰值导致数值问题。
PEAK_EPS = 1e-12


# ============================================================
# 自适应滤波器设置
# ============================================================

# 第一版先跑 256 taps。
# 后续正式实验可以改成 [256, 512]。
FILTER_ORDERS = [256]

# 蒙特卡洛 trial 数。
# 调试时可以先设 3 或 5，正式实验建议 20 或 30。
MC_TRIALS = 3

# 随机种子。
SEED = 0

# 学习曲线统计窗口长度，单位为样本点。
# 16 kHz 下 1024 点约 64 ms。
CURVE_WINDOW = 1

# 稳态指标取最后多少比例的窗口。
SS_LAST_RATIO = 0.1


# ============================================================
# 算法列表
# ============================================================

ALGO_LIST = [
    "LMS",
    "WL-LMS",
    "GH-WL-LMS",
]


# ============================================================
# 算法参数
# ============================================================

# 真实 AEC 语音数据比实验 B/C 难，建议先用保守步长。
# 如果收敛太慢，再逐步增大 step_size。
ALGO_PARAMS = dict(
    LMS=dict(
        step_size=0.001,
    ),

    WLLMS=dict(
        M=20,
        sigma=0.4,
        step_size=1e-4,
        seed=0,
    ),

    GHWLLMS=dict(
        M=20,
        scale=0.6,
        step_size=0.02,
        normalized=True,
        eps=1e-8,
        seed=0,
    ),
)


# ============================================================
# 绘图与结果保存
# ============================================================

PLOT = dict(
    smooth_window=3,
    erle_ylim=None,
    mse_ylim=None,
)

RESULT_DIR = ROOT / "results" / "exp_d_aec_synthetic"