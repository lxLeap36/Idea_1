"""
非线性系统辨识场景封装（实验三）
将数据集生成、算法实例化、运行、结果收集全部封装在此模块。
外部只需调用 run_stationary() 或 run_nonstationary()。
"""

import sys
import time
import numpy as np
from pathlib import Path
import copy

# 确保项目根目录在 sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.nonlinear_system import get_stationary_dataset, get_nonstationary_dataset
from algorithms import LMS, KLMS, KRLS, RFFMC, NKRGMC, WLLMS, WLRLS
from configs.exp3_config import SNAPSHOT, SNAPSHOT_EVERY, SS_LAST_N


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
                  trial_seed: int = 0,
                  snapshot: bool = None,
                  snapshot_every: int = None):
    """
    对所有算法各跑一次，返回
    {name: {'mse_curve': np.ndarray, 'test_errors': np.ndarray, 'time': float}}
    """
    results = {}
    for name, algo in algos.items():
        # 使用 trial_seed 重置算法随机状态（仅对有 seed 属性的算法生效）
        if hasattr(algo, 'seed'):
            try:
                algo.seed = trial_seed
            except Exception:
                pass

        # reset algorithm and perform any algorithm-specific initialization
        algo.reset()
        if hasattr(algo, 'init_anchors'):
            try:
                # NKRGMC needs anchors initialized from training set
                algo.init_anchors(X_train)
            except Exception:
                pass

        n_train = X_train.shape[0]
        train_errors = np.zeros(n_train)

        # Decide whether to snapshot and how often per config (allow overrides)
        if snapshot is None:
            snapshot = SNAPSHOT
        if snapshot_every is None:
            snapshot_every = SNAPSHOT_EVERY
        snapshots = []
        t0 = time.perf_counter()
        for k in range(n_train):
            train_errors[k] = algo.update(X_train[k], d_train[k])
            if snapshot and ((k % snapshot_every) == 0):
                # Prefer lightweight state dict if algorithm provides it
                if hasattr(algo, 'get_state'):
                    try:
                        snapshots.append(('state', algo.get_state()))
                    except Exception:
                        snapshots.append(('copy', copy.copy(algo)))
                else:
                    # fallback to shallow copy of object
                    try:
                        snapshots.append(('copy', copy.copy(algo)))
                    except Exception:
                        # last resort: store None to keep indexing consistent
                        snapshots.append(('none', None))

        elapsed = time.perf_counter() - t0

        # Build test_mse_curve aligned to training iterations. If snapshot_every >1,
        # we fill non-snapshot iterations by repeating the most recent snapshot's result.
        test_mse_curve = np.zeros(n_train)
        last_val = None
        snap_idx = 0
        for k in range(n_train):
            if snapshot and ((k % snapshot_every) == 0) and snap_idx < len(snapshots):
                tag, payload = snapshots[snap_idx]
                snap_idx += 1
                if tag == 'state' and payload is not None:
                    # restore a fresh algorithm instance of same class with state
                    tmp = None
                    try:
                        # Prefer constructing with algorithm-provided init kwargs
                        if hasattr(algo, 'get_init_kwargs'):
                            kwargs = algo.get_init_kwargs() or {}
                            tmp = algo.__class__(**kwargs)
                        else:
                            # try basic constructor with filter length if available
                            try:
                                tmp = algo.__class__(*([algo.L] if hasattr(algo, 'L') else []))
                            except Exception:
                                tmp = copy.copy(algo)
                    except Exception:
                        tmp = copy.copy(algo)
                    try:
                        if hasattr(tmp, 'set_state'):
                            tmp.set_state(payload)
                        else:
                            # fallback: attempt to assign attributes
                            for k, v in payload.items():
                                try:
                                    setattr(tmp, k, v)
                                except Exception:
                                    pass
                    except Exception:
                        pass
                elif tag == 'copy' and payload is not None:
                    tmp = payload
                else:
                    tmp = None

                if tmp is not None:
                    preds = np.array([tmp.predict(x) for x in X_test])
                    test_errs = d_test - preds
                    mse_lin = float(np.mean(test_errs ** 2))
                    mse_lin = max(mse_lin, 1e-20)
                    last_val = 10.0 * np.log10(mse_lin)
                # else last_val remains as previous
            test_mse_curve[k] = last_val if last_val is not None else 10.0 * np.log10(1e-20)

        # compute final_test_errors using final model state (algo has been trained)
        final_preds = np.array([algo.predict(x) for x in X_test])
        final_test_errors = d_test - final_preds

        # capture the final model state in a portable form so callers can reconstruct the trained model
        try:
            if hasattr(algo, 'get_state'):
                final_snapshot = ('state', algo.get_state())
            else:
                final_snapshot = ('copy', copy.copy(algo))
        except Exception:
            final_snapshot = snapshots[-1] if len(snapshots) > 0 else None

        # try to capture algo init kwargs if algorithm provides helper
        init_kwargs = None
        try:
            if hasattr(algo, 'get_init_kwargs'):
                init_kwargs = algo.get_init_kwargs() or None
        except Exception:
            init_kwargs = None

        results[name] = {
            'mse_curve':   test_mse_curve,
            'test_errors': final_test_errors,
            'time':        elapsed,
            'final_preds': final_preds,
            'final_snapshot': final_snapshot,
            'algo_class': algo.__class__,
            'algo_init_kwargs': init_kwargs,
            'algo_filter_length': getattr(algo, 'L', None),
        }
    return results


# ─── Monte Carlo 平均 ──────────────────────────────────────────────────────────

def run_monte_carlo(
    build_fn,           # callable() -> (algos, X_train, d_train, X_test, d_test)
    n_trials: int = 50,
    verbose: bool = True,
    snapshot: bool = None,
    snapshot_every: int = None,
    ss_last_n: int = None,
    return_last_trial: bool = False,
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
    all_times = {}

    last_trial_results = None
    for trial in range(n_trials):
        if verbose:
            print(f"  Trial {trial + 1}/{n_trials} ...", end='\r')

        algos, X_train, d_train, X_test, d_test = build_fn(trial)
        trial_results = run_one_trial(algos, X_train, d_train, X_test, d_test, trial,
                                      snapshot=snapshot, snapshot_every=snapshot_every)

        for name, res in trial_results.items():
            if name not in all_curves:
                all_curves[name] = []
                all_times[name]  = []
            all_curves[name].append(res['mse_curve'])
            all_times[name].append(res['time'])
        # keep the last trial's detailed results for downstream analysis (spectra etc.)
        last_trial_results = trial_results

    if verbose:
        print()

    avg_curves = {}
    ss_mse = {}
    avg_time = {}

    if ss_last_n is None:
        ss_last_n = SS_LAST_N

    for name in all_curves:
        # 学习曲线：对各 trial 的 mse_curve 取对数域均值（先线性域平均再转 dB）
        curves_linear = [10 ** (c / 10) for c in all_curves[name]]
        avg_linear = np.mean(curves_linear, axis=0)
        avg_curves[name] = 10 * np.log10(np.maximum(avg_linear, 1e-20))

        # 稳态 MSE：取平均测试曲线的最后若干步的线性均值，再转为 dB
        last_n = ss_last_n
        tail = avg_linear[-last_n:] if len(avg_linear) >= last_n else avg_linear
        ss_lin = float(np.mean(tail)) if len(tail) > 0 else 1e-20
        ss_mse[name] = 10 * np.log10(max(ss_lin, 1e-20))

        avg_time[name] = float(np.mean(all_times[name]))

    if return_last_trial:
        return avg_curves, ss_mse, avg_time, last_trial_results
    return avg_curves, ss_mse, avg_time


# ─── 平稳场景 ──────────────────────────────────────────────────────────────────

def run_stationary(noise_var: float, params: dict, dataset_cfg: dict,
                   n_trials: int = 50, verbose: bool = True, algo_list: list = None,
                   snapshot: bool = None, snapshot_every: int = None, ss_last_n: int = None):
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

    return run_monte_carlo(build_fn, n_trials, verbose,
                           snapshot=snapshot, snapshot_every=snapshot_every, ss_last_n=ss_last_n)


# ─── 非平稳场景 ────────────────────────────────────────────────────────────────

def run_nonstationary(noise_var: float, params: dict, dataset_cfg: dict,
                      n_trials: int = 50, verbose: bool = True, algo_list: list = None,
                      snapshot: bool = None, snapshot_every: int = None, ss_last_n: int = None):
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

    return run_monte_carlo(build_fn, n_trials, verbose,
                           snapshot=snapshot, snapshot_every=snapshot_every, ss_last_n=ss_last_n)
