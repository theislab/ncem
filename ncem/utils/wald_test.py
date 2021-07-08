import numpy as np


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

    res = []
    bool_res = []
    for i, fi in enumerate(fisher_inv):
        sig = []
        bool_sig = []
        a_var = params[:, i, :].T
        for idx in range(a_var.shape[0]):
            p_val = _get_p_value(a_var, fi, idx)
            significance = correct(p_val)
            bool_significance = significance < significance_threshold
            sig.append(significance)
            bool_sig.append(bool_significance)
        res.append(np.array(sig))  # produces (target, sources, genes) array
        bool_res.append(np.array(bool_sig))
    return np.array(bool_res), np.array(res)
