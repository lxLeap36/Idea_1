# smoke_snapshot_test.py
# Save this file in the project root (same folder that contains 'configs' and 'scenarios').

from pathlib import Path
import sys
import pprint
import time

# Ensure project root is on sys.path (script is expected to live in project root)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imports from the project
from configs.exp3_config import ALGO_PARAMS, DATASET
from scenarios.nonlinear_id import run_stationary

def main():
    print("Smoke test: snapshot evaluation (LMS only, small sizes)")

    # small dataset to keep runtime tiny
    ds = dict(DATASET)
    ds['n_train'] = 10
    ds['n_test']  = 5

    algo_list = ['LMS','KLMS','KRLS','RFFMC','NKRGMC','WL-LMS','WL-RLS']

    t0 = time.time()
    avg_curves, ss_mse, avg_time = run_stationary(
        noise_var   = 0.0036,
        params      = ALGO_PARAMS,
        dataset_cfg = ds,
        n_trials    = 1,
        verbose     = True,
        algo_list   = algo_list,
        snapshot    = True,
        snapshot_every = 1,
        ss_last_n   = 5,   # small tail for smoke test
    )
    dt = time.time() - t0

    print("\n=== Smoke test results ===")
    print("algorithms:", list(avg_curves.keys()))
    for name, curve in avg_curves.items():
        print(f"\nAlgorithm: {name}")
        print(f"  curve length: {len(curve)}")
        print(f"  curve (Testing MSE(dB)) per iteration:")
        pprint.pprint(curve.tolist())
    print("\nsteady-state MSE (ss_mse):")
    pprint.pprint(ss_mse)
    print("\navg_time per algo (s):")
    pprint.pprint(avg_time)
    print(f"\nTotal run_stationary() wall time: {dt:.3f}s")

if __name__ == '__main__':
    main()
