"""
实验 D2 配置：
标准 FDAF 无重叠帧 + 非线性残差分支级联实验。

本实验同时完成：
    1. 级联效果评估：MSE / ERLE；
    2. 流式逐帧处理；
    3. 实时性评估：每帧耗时、RTF、超时率。

任务：
    远端单讲 echo modeling / echo cancellation 诊断。

信号：
    x(n) = farend_speech
    d(n) = echo_signal

算法对比：
    LMS
    FDAF
    FDAF+WL-LMS
    FDAF+GH-WL-LMS-Fast
"""

from pathlib import Path


# ============================================================
# 路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(r"D:\Datasets\AEC-Challenge\datasets\synthetic")
META_CSV = DATA_ROOT / "meta.csv"
FAREND_DIR = DATA_ROOT / "farend_speech"
ECHO_DIR = DATA_ROOT / "echo_signal"


# ============================================================
# 数据筛选
# ============================================================

REQUIRE_FAREND_NONLINEAR = 1
REQUIRE_FAREND_NOISY = 0
REQUIRE_NEAREND_NOISY = 0

SPLIT = None


# ============================================================
# 音频设置
# ============================================================

TARGET_FS = 16000

SEGMENT_SECONDS = 9.0

SEGMENT_MODE = "active"

SKIP_SHORT_CLIPS = True

REMOVE_DC = True

PEAK_NORMALIZE = True

PEAK_EPS = 1e-12


# ============================================================
# 流式帧设置
# ============================================================

# 第一版采用标准无重叠帧。
# 因此 FRAME_LENGTH = HOP_LENGTH。
FRAME_LENGTH = 2048
HOP_LENGTH = 2048

# 实时预算 = FRAME_LENGTH / TARGET_FS。
# 16 kHz 下 256 点对应 16 ms。


# ============================================================
# Monte Carlo 设置
# ============================================================

MC_TRIALS = 10

SEED = 0

SS_LAST_RATIO = 0.1


# ============================================================
# 算法列表
# ============================================================

ALGO_LIST = [
    "LMS",
    "FDAF",
    "FDAF+WL-LMS",
    "FDAF+GH-WL-LMS-Fast",
]


# ============================================================
# LMS baseline 参数
# ============================================================

# 为了和第一版单分区 FDAF 公平，默认 LMS 阶数也设为 FRAME_LENGTH。
# 如果后续想复现 D1 中 p=1024 的长 LMS，可单独改成 1024。
LMS_PARAMS = dict(
    filter_order=FRAME_LENGTH,
    step_size=0.001,
)


# ============================================================
# FDAF 参数
# ============================================================

FDAF_PARAMS = dict(
    filter_length=FRAME_LENGTH,
    block_size=FRAME_LENGTH,
    step_size=0.5,
    eps=1e-6,
    leakage=0.0,
)


# ============================================================
# 非线性残差分支参数
# ============================================================

NONLINEAR_FILTER_ORDER = 5

WLLMS_PARAMS = dict(
    M=40,
    sigma=0.4,
    step_size=0.0005,
    seed=0,
)

GHWLLMSFAST_PARAMS = dict(
    M=40,
    scale=0.6,
    step_size=0.1,
    normalized=True,
    eps=1e-8,
    seed=0,
)


# ============================================================
# 绘图
# ============================================================

PLOT = dict(
    smooth_window=3,
    mse_ylim=None,
    erle_ylim=None,
)


# ============================================================
# 结果目录
# ============================================================

RESULT_DIR = ROOT / "results" / "exp_d2_fdaf_nonlinear_cascade"