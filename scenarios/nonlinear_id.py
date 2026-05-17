"""
非线性系统辨识场景封装（实验三）
将数据集生成、算法实例化、运行、结果收集全部封装在此模块。
外部只需调用 run_stationary() 或 run_nonstationary()。
"""

import sys
import time
import numpy as np
from pathlib import Path

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.nonlinear_system import get_stationary_dataset, get_nonstationary_dataset
from algorithms import LMS, KLMS, KRLS, RFFMC, NKRGMC, WLLMS, WLRLS


# ─── 算法工厂 ──────────────────────────────────────────────────────────────────

def build_algorithms(filter_order: int, params: dict, algo_list: list = None) -> dict:
    """
    根据配置字典实例化所需算法，返回 {name: algo_instance}。

    Parameters
    - filter_order: int
    - params: dict 中包含每个算法的超参数
    - algo_list: 可选 list，只实例化并返回其中列出的算法（按名字），
                 若为 None 则实例化全部算法。
    """
    p = filter_order
    all_algos = {
        'LMS':    LMS(p, **params['LMS']),
        'KLMS':   KLMS(**params['KLMS']),
        'KRLS':   KRLS(**params['KRLS']),
        'RFFMC':  RFFMC(p, **params['RFFMC']),
        'NKRGMC': NKRGMC(p, **params['NKRGMC']),
        'WL-LMS': WLLMS(p, **params['WLLMS']),
        'WL-RLS': WLRLS(p, **params['WLRLS']),
    }

    if algo_list is None:
        return all_algos

    # 只返回在请求列表中的算法（保留请求顺序）
    selected = {}
    for name in algo_list:
        if name in all_algos:
            selected[name] = all_algos[name]
        else:
            raise ValueError(f"Unknown algorithm requested: {name}")
    return selected


# ─── 单次 trial 运行 ────────────────────────────────────────────────────────────

def run_one_trial(algos: dict,
                  X_train, d_train, X_test, d_test,
                  trial_seed: int = 0):
    """
    对所有算法各跑一次，返回
    {name: {'mse_curve': np.ndarray, 'test_errors': np.ndarray, 'time': float}}
    """
    results = {}
    for name, algo in algos.items():
        # 使用 trial_seed 重置算法随机状态（仅对有 seed 属性的算法生效）
        if hasattr(algo, 'seed'):
            algo.seed = trial_seed
        t0 = time.perf_counter()
        _, test_errors, mse_curve, _ = algo.run(X_train, d_train, X_test, d_test)
        elapsed = time.perf_counter() - t0
        results[name] = {
            'mse_curve':   mse_curve,
            'test_errors': test_errors,
            'time':        elapsed,
        }
    return results


# ─── Monte Carlo 平均 ──────────────────────────────────────────────────────────

def run_monte_carlo(
    build_fn,           # callable() -> (algos, X_train, d_train, X_test, d_test)
    n_trials: int = 50,
    verbose: bool = True,
):
    """
    对同一实验配置重复 n_trials 次，对学习曲线和测试误差取均值。

    返回
    ----
    avg_curves : {name: np.ndarray (n_train,)}  平均 MSE(dB) 学习曲线
    ss_mse     : {name: float}                  平均稳态 MSE(dB)
    avg_time   : {name: float}                  平均运行时间 (s)
    """
    all_curves = {}
    all_test_sq = {}
    all_times = {}

    for trial in range(n_trials):
        if verbose:
            print(f"  Trial {trial + 1}/{n_trials} ...", end='\r')

        algos, X_train, d_train, X_test, d_test = build_fn(trial)
        trial_results = run_one_trial(algos, X_train, d_train, X_test, d_test, trial)

        for name, res in trial_results.items():
            if name not in all_curves:
                all_curves[name]   = []
                all_test_sq[name]  = []
                all_times[name]    = []
            all_curves[name].append(res['mse_curve'])
            all_test_sq[name].append(res['test_errors'] ** 2)
            all_times[name].append(res['time'])

    if verbose:
        print()

    avg_curves = {}
    ss_mse = {}
    avg_time = {}

    for name in all_curves:
        # 学习曲线：对各 trial 的 mse_curve 取对数域均值（先线性域平均再转 dB）
        curves_linear = [10 ** (c / 10) for c in all_curves[name]]
        avg_linear = np.mean(curves_linear, axis=0)
        avg_curves[name] = 10 * np.log10(np.maximum(avg_linear, 1e-20))

        # 稳态 MSE：用测试集误差平方均值
        mean_sq = np.mean([np.mean(sq) for sq in all_test_sq[name]])
        ss_mse[name] = 10 * np.log10(max(mean_sq, 1e-20))

        avg_time[name] = float(np.mean(all_times[name]))

    return avg_curves, ss_mse, avg_time


# ─── 平稳场景 ──────────────────────────────────────────────────────────────────

def run_stationary(noise_var: float, params: dict, dataset_cfg: dict,
                   n_trials: int = 50, verbose: bool = True, algo_list: list = None):
    """
    平稳非线性系统辨识实验

    返回
    ----
    avg_curves, ss_mse, avg_time
    """
    p = dataset_cfg['p']

    def build_fn(trial):
        seed = dataset_cfg['seed'] + trial
        X_tr, d_tr, X_te, d_te, _ = get_stationary_dataset(
            noise_var    = noise_var,
            n_train      = dataset_cfg['n_train'],
            n_test       = dataset_cfg['n_test'],
            p            = p,
            c            = dataset_cfg['c_stationary'],
            seed         = seed,
        )
        algos = build_algorithms(p, params, algo_list)
        return algos, X_tr, d_tr, X_te, d_te

    return run_monte_carlo(build_fn, n_trials, verbose)


# ─── 非平稳场景 ────────────────────────────────────────────────────────────────

def run_nonstationary(noise_var: float, params: dict, dataset_cfg: dict,
                      n_trials: int = 50, verbose: bool = True, algo_list: list = None):
    """
    非平稳非线性系统辨识实验（系数突变）

    返回
    ----
    avg_curves, ss_mse, avg_time
    """
    p = dataset_cfg['p']

    def build_fn(trial):
        seed = dataset_cfg['seed'] + trial
        X_tr, d_tr, X_te, d_te = get_nonstationary_dataset(
            noise_var    = noise_var,
            n_train      = dataset_cfg['n_train_ns'],
            n_test       = dataset_cfg['n_test'],
            p            = p,
            c1           = dataset_cfg['c_stationary'],
            c2_coef      = dataset_cfg['c_nonstationary'],
            change_point = dataset_cfg['change_point'],
            seed         = seed,
        )
        algos = build_algorithms(p, params, algo_list)
        return algos, X_tr, d_tr, X_te, d_te

    return run_monte_carlo(build_fn, n_trials, verbose)
