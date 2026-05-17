"""
核函数工具模块
"""

import numpy as np


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    """
    高斯核函数（标量或向量输入）
    κ(x, y) = exp(-||x - y||² / σ²)
    """
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (sigma ** 2))


def gaussian_kernel_matrix(X: np.ndarray, sigma: float) -> np.ndarray:
    """
    计算 Gram 矩阵 K[i,j] = κ(x_i, x_j)
    X: shape (n_samples, n_features)
    """
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k = gaussian_kernel(X[i], X[j], sigma)
            K[i, j] = k
            K[j, i] = k
    return K


def gaussian_basis_1d(x: float, centers: np.ndarray, sigma: float) -> np.ndarray:
    """
    一维高斯基函数向量
    G_j(x) = exp(-(x - z_j)² / σ²)，j = 1,...,M
    x      : 标量输入
    centers: shape (M,)，基函数中心
    返回   : shape (M,)
    """
    return np.exp(-((x - centers) ** 2) / (sigma ** 2))


def random_fourier_features(X: np.ndarray, d: int, sigma: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Random Fourier Features（近似高斯核）
    X  : shape (n_samples, n_features)
    d  : 特征维数
    返回: shape (n_samples, d)
    """
    n_features = X.shape[1]
    omega = rng.normal(0, 1.0 / sigma, size=(n_features, d))
    b = rng.uniform(0, 2 * np.pi, size=d)
    Z = np.sqrt(2.0 / d) * np.cos(X @ omega + b)
    return Z


def nystrom_features(X_train: np.ndarray, X: np.ndarray, d: int,
                     sigma: float) -> np.ndarray:
    """
    Nyström 特征近似
    X_train: 随机选取的 d 个锚点，shape (d, n_features)
    X      : 待映射数据，shape (n_samples, n_features)
    返回   : shape (n_samples, d)
    """
    n = X.shape[0]
    K_mn = np.zeros((n, d))
    for i in range(n):
        for j in range(d):
            K_mn[i, j] = gaussian_kernel(X[i], X_train[j], sigma)

    K_mm = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            K_mm[i, j] = gaussian_kernel(X_train[i], X_train[j], sigma)

    # 对 K_mm 做特征分解
    eigvals, eigvecs = np.linalg.eigh(K_mm)
    eigvals = np.maximum(eigvals, 1e-10)
    K_mm_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    return K_mn @ K_mm_inv_sqrt
