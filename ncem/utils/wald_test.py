import numpy as np


def get_fim_inv(x: np.array, y: np.array):
    var = np.var(y, axis=0)
    fim = np.expand_dims(np.matmul(x.T, x), axis=0) / np.expand_dims(var, axis=[1, 2])

    fim_inv = np.array([
        np.linalg.pinv(fim[i, :, :])
        for i in range(fim.shape[0])
    ])
    return fim_inv


def _get_p_value(a_var: np.array, fisher_inv: np.array, coef_loc_totest: int):
    """Return the p-value for differential expression for each gene.

    Parameters
    ----------
    a_var : np.array
        a var matrix.
    fisher_inv : np.array
        Fisher inv matrix.
    coef_loc_totest : int
        Coefficient location to test.

    Returns
    -------
    wald_test
    """
    from diffxpy.stats.stats import wald_test

    theta_mle = a_var[coef_loc_totest]
    theta_sd = fisher_inv[:, coef_loc_totest, coef_loc_totest]
    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    theta_sd = np.sqrt(theta_sd)
    return wald_test(theta_mle=theta_mle, theta_sd=theta_sd, theta0=0)


def wald_test(
    params,
    fisher_inv: list,
    significance_threshold: float = 0.01,
):
    """Compute wald test.

    Parameters
    ----------
    params
        Parameters.
    fisher_inv : list
        List of fisher inv matrix.
    significance_threshold : float
        Significance threshold for corrected p-values.

    Returns
    -------
    bool_res, res
    """
    from diffxpy.testing.correction import correct
    significance = []
    qvalues = []
    pvalues = []
    for idx in range(params.T.shape[0]):
        pvals = _get_p_value(params.T, fisher_inv, idx)
        pvalues.append(pvals)
        
    pvalues = np.concatenate(pvalues)
    qvalues = correct(pvalues)
    pvalues = np.reshape(pvalues, (-1,params.T.shape[1])) 
    qvalues = np.reshape(qvalues, (-1,params.T.shape[1]))  
    significance = qvalues < significance_threshold

    return significance, pvalues, qvalues
