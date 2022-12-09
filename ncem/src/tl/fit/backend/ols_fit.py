import numpy as np
import scipy.sparse


def ols_fit(x_, y_):
    """beta = (XT * X)^-1 XT y"""
    x = np.matmul(
        np.linalg.pinv(np.matmul(x_.T, x_)),
        x_.T
    )
    # Densify only per gene in loop over gene fits:
    if isinstance(y_, scipy.sparse.spmatrix):
        return np.array([
            np.matmul(
                x, y_[:, [i]].todense()
            )
            for i in range(y_.shape[1])
        ])
    else:
        return np.array([
            np.matmul(
                x, y_[:, [i]]
            )
            for i in range(y_.shape[1])
        ])
