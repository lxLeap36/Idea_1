import numpy as np
from datasets.imd_echo import compute_imd_y, build_dataset_from_xy

x = np.arange(10, dtype=float)
y = compute_imd_y(x, c2=0.3, c3=0.1)

X, _, d, _ = build_dataset_from_xy(x, y, p=3)

print("X[0] =", X[0])
print("d[0] =", d[0])
print("should equal y[2] =", y[2])

print("X[1] =", X[1])
print("d[1] =", d[1])
print("should equal y[3] =", y[3])