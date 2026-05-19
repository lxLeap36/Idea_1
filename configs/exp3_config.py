"""
实验三：非线性系统辨识 超参数配置
对应论文 Section V-C
"""

# ─── 数据集配置 ────────────────────────────────────────────────────────────────
DATASET = dict(
    p          = 5,        # 输入阶数（用前 5 个样本预测当前值）
    n_train    = 2000,     # 平稳场景训练样本数
    n_test     = 200,      # 测试样本数
    c_stationary   = [0.8, 0.5, 0.3, 0.9, 0.1],   # 初始系数向量
    c_nonstationary= [0.4, 0.7, 0.6, 0.6, 0.2],   # 突变后系数向量
    change_point   = 2001,                          # 非平稳场景突变时刻
    n_train_ns = 4000,     # 非平稳场景训练样本数
    seed       = 42,
    # 实验噪声方差（平稳场景跑两组）
    noise_vars = [0.0036, 0.01],
    # 非平稳场景噪声方差（论文图16只跑 σ²=0.0036）
    noise_var_ns = 0.0036,
)

# ─── Monte Carlo 配置 ──────────────────────────────────────────────────────────
MC_TRIALS = 1          # Monte Carlo 平均次数

# ─── 各算法超参数（尽量复现论文最优性能） ──────────────────────────────────────

ALGO_PARAMS = dict(

    LMS = dict(
        step_size    = 0.02,
    ),

    KLMS = dict(
        step_size    = 0.15,
        sigma        = 1.0,
    ),

    KRLS = dict(
        sigma        = 1.0,
        reg          = 1e-3,
        forgetting   = 0.999,
    ),

    RFFMC = dict(
        d            = 100,        # 随机傅里叶特征维数
        step_size    = 0.14,
        sigma        = 2.0,
        kernel_bw    = 1.0,
        seed         = 0,
    ),

    NKRGMC = dict(
        d            = 100,        # Nyström 锚点数
        sigma        = 2.0,
        reg          = 1e-3,
        forgetting   = 0.999,
        kernel_bw    = 1.0,
        alpha_order  = 2.5,
        seed         = 0,
    ),

    # 论文实验三平稳场景：WL-LMS M=50，WL-RLS M=20（图14/15注释推断）
    WLLMS = dict(
        M            = 50,
        sigma        = 0.4,
        step_size    = 0.006,
        seed         = 0,
    ),

    WLRLS = dict(
        M            = 20,
        sigma        = 0.5,
        reg          = 1e-3,
        forgetting   = 0.999,
        seed         = 0,
    ),
)

# ─── 可配置要运行的算法列表（默认包含全部 7 种） 'LMS', 'KLMS', 'KRLS', 'RFFMC', 'NKRGMC', 'WL-LMS', 'WL-RLS'
ALGO_LIST = [
    'LMS', 'KLMS', 'WL-LMS'
]

# ─── 快照评估配置 ─────────────────────────────────────────────────────────────
# 是否启用快照评估（每步或按间隔在训练过程中保存模型状态用于后续在测试集上评估）
SNAPSHOT = True
# 每隔多少步保存一次快照（1 表示每步）
SNAPSHOT_EVERY = 1
# 用于稳态 MSE 的尾部长度（与 steady_state_mse_db 原始默认一致）
SS_LAST_N = 500

# ─── 绘图配置 ──────────────────────────────────────────────────────────────────
PLOT = dict(
    smooth_window = 80,      # 学习曲线滑动平均窗口
    y_lim_stationary = (-55, 5),   # MSE(dB) y 轴范围
    y_lim_ns         = (-60, 10),
)

# ─── 结果输出路径 ──────────────────────────────────────────────────────────────
import os
RESULT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'exp3')
