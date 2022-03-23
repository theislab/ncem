import numpy as np


def ols_fit(x_, y_):
    """beta = (XT * X)^-1 XT y"""
    X = np.matmul(
        np.linalg.pinv(np.matmul(x_.T, x_)),
        x_.T
    )
    return np.array([
        np.matmul(
            X, y_[:, [i]]
        )
        for i in range(y_.shape[1])
    ])