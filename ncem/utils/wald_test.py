import numpy as np


def _get_p_value(a_var: np.ndarray, fisher_inv: np.ndarray, coef_loc_totest: int):
    """
    Returns the p-value for differential expression for each gene
    :return: np.ndarray
    """
    from diffxpy.stats.stats import wald_test

    theta_mle = a_var[coef_loc_totest]
    theta_sd = fisher_inv[:, coef_loc_totest, coef_loc_totest]
    theta_sd = np.nextafter(0, np.inf, out=theta_sd, where=theta_sd < np.nextafter(0, np.inf))
    theta_sd = np.sqrt(theta_sd)
    return wald_test(theta_mle=theta_mle, theta_sd=theta_sd, theta0=0)


def wald_test(
    parameters,
    fisher_inv: list,
    significance_threshold: float = 0.01,
):
    from diffxpy.testing.correction import correct

    res = []
    bool_res = []
    for i, fi in enumerate(fisher_inv):
        sig = []
        bool_sig = []
        a_var = parameters[:, i, :].T
        for idx in range(a_var.shape[0]):
            p_val = _get_p_value(a_var, fi, idx)
            significance = correct(p_val)
            bool_significance = significance < significance_threshold
            sig.append(significance)
            bool_sig.append(bool_significance)
        res.append(np.array(sig))  # produces (target, sources, genes) array
        bool_res.append(np.array(bool_sig))
    return np.array(bool_res), np.array(res)
