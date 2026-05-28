"""
实验 D3 配置文件：
真实远端语音上的模拟扬声器/功放非线性建模实验。

实验目的：
    只验证 LMS / WL-LMS / GH-WL-LMS 对“纯非线性映射”的建模能力。
    这里暂时不加入房间回声路径，不加入 echo_signal，不加入 nearend_mic_signal。

信号定义：
    x(n) = farend_speech
    d(n) = speaker_nonlinearity(x(n))

因此，本实验不是完整 AEC，而是非线性设备失真建模实验。
"""

from pathlib import Path


# ============================================================
# 项目路径
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(r"D:\Datasets\AEC-Challenge\datasets\synthetic")
META_CSV = DATA_ROOT / "meta.csv"
FAREND_DIR = DATA_ROOT / "farend_speech"


# ============================================================
# 数据筛选设置
# ============================================================

# D3 只使用 farend_speech。
# 因为我们自己施加非线性，所以不要求 is_farend_nonlinear=1。
# 这里优先选无远端噪声的远端语音，避免噪声影响非线性建模判断。
REQUIRE_FAREND_NOISY = 0

# 是否限定 split。
# 可选："train" / "test" / None。
SPLIT = None


# ============================================================
# 音频设置
# ============================================================

TARGET_FS = 16000

# 每条 trial 使用的片段长度。
# AEC synthetic 每条通常约 10 秒，因此这里可以直接用 10 秒。
SEGMENT_SECONDS = 9.0

# 片段裁剪方式：
#   "active"：选择远端语音能量较高的连续片段；
#   "start" ：从开头截取。
SEGMENT_MODE = "active"

# 如果音频不足目标长度，是否跳过。
SKIP_SHORT_CLIPS = True

# 是否去除输入直流分量。
REMOVE_DC = True

# 是否对输入做峰值归一化。
# 注意：这里只归一化 farend_speech，因为目标是由归一化后的 x 生成的。
PEAK_NORMALIZE_INPUT = True

PEAK_EPS = 1e-12


# ============================================================
# 模拟非线性设置
# ============================================================

# 非线性类型：
#   "soft_sigmoid"：使用软限幅 + sigmoid 非线性；
#   "soft_clip"   ：只使用软限幅；
#   "tanh"        ：使用 tanh 饱和；
#   "poly_clip"   ：多项式 + 限幅。
NONLINEARITY_TYPE = "soft_sigmoid"

# sigmoid 非线性中的 delta 候选。
# 文献截图中提到 delta pair 可从 (1,1)、(2,2)、(3,3)、(4,4) 中选。
# 这里先用 (3,3) 作为中等偏强非线性。
DELTA_POS = 3.0
DELTA_NEG = 3.0

# 非线性强度增益。
# 增大该值会让输入更容易进入饱和区。
DRIVE_GAIN = 1.8

# 是否对非线性输出去均值。
# 如果设为 True，LMS 不会因为 DC 偏置被额外惩罚；
# 如果设为 False，可以检验非线性模型是否能学习偏置项。
CENTER_TARGET = True

# 是否让目标和输入具有相近 RMS。
# 设为 True 后，MSE 数值更容易比较；非线性形状仍然保留。
MATCH_TARGET_RMS_TO_INPUT = True


# ============================================================
# 自适应滤波器设置
# ============================================================

# 由于 D3 是 memoryless 非线性映射，理论上 p=1 就足够。
# 也可以增加 p=4 或 p=8，观察语音相关性是否让 LMS 借助历史样本变好。
FILTER_ORDERS = [1]

MC_TRIALS = 20
SEED = 0

# 学习曲线窗口。
# 16 kHz 下 1024 点约 64 ms。
CURVE_WINDOW = 512

# 稳态指标使用最后多少比例的窗口。
SS_LAST_RATIO = 0.1


# ============================================================
# 算法列表
# ============================================================

ALGO_LIST = [
    "LMS",
    "WL-LMS",
    "GH-WL-LMS-Fast",
]


# ============================================================
# 算法参数
# ============================================================

# D3 是纯非线性映射，p 很短，所以可以适当提高非线性算法步长。
# 如果发现曲线震荡，就降低对应 step_size。
ALGO_PARAMS = dict(
    LMS=dict(
        step_size=0.02,
    ),

    WLLMS=dict(
        M=40,
        sigma=0.4,
        step_size=0.005,
        seed=0,
    ),

    GHWLLMSFast=dict(
        M=40,
        scale=0.6,
        step_size=0.2,
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
    mse_ylim=None,
    erle_ylim=None,
)

RESULT_DIR = ROOT / "results" / "exp_d3_speaker_nonlinearity"