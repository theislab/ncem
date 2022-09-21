# Overall design matrix used for NCEM:
OBSM_KEY_DMAT = "ncem_dmat"
# Design matrix representation of niche of each observation, subset of OBSM_KEY_DMAT:
OBSM_KEY_DMAT_NICHE = "ncem_niche"
# All fitted parameters:
VARM_KEY_PARAMS = "ncem_params"
# Coefficient values of tested parameters of NCEM:
VARM_KEY_TESTED_PARAMS = "ncem_tested_params"
# P-values of pair-wise type interactions (intercept):
VARM_KEY_PVALS = "ncem_pvals"
# FDR correction of VARM_KEY_PVALS:
VARM_KEY_FDR_PVALS = "ncem_fdr_pvals"
# Coefficient values of tested parameters of differential NCEM:
VARM_KEY_TESTED_PARAMS_DIFFERENTIAL = "differential_ncem_tested_params"
# P-values of pair-wise type interactions (interaction of condition to type pair interactions):
VARM_KEY_PVALS_DIFFERENTIAL = "differential_ncem_pvals"
# FDR correction of VARM_KEY_PVALS_DIFFERENTIAL:
VARM_KEY_FDR_PVALS_DIFFERENTIAL = "differential_ncem_fdr_pvals"
# P-values of spline coefficients which indicate if spline fit is non-constant:
VARM_KEY_PVALS_SPLINE = "ncem_spline_pvals"
# FDR correction of VARM_KEY_PVALS_SPLINE:
VARM_KEY_FDR_PVALS_SPLINE = "ncem_spline_fdr_pvals"
# .uns key under which list of modelled cell types is saved:
UNS_KEY_CELL_TYPES = "cell_types"
# .uns key under which list of modelled conditions is saved (used for differential mode):
UNS_KEY_CONDITIONS = "conditions"
# .uns key under which all NCEM entries in .uns are stored
UNS_KEY_NCEM = "ncem"
# .uns key under which fit mode (by index type or globally) is saved:
UNS_KEY_PER_INDEX = "per_index_type"
# List of coefficient names that correspond to spline, organised by cell type into a dictionary:
UNS_KEY_SPLINE_COEFS = "spline_coefs"
# Number of degrees of freedom of spline:
UNS_KEY_SPLINE_DF = "spline_df"
# Type of spline:
UNS_KEY_SPLINE_FAMILY = "spline_family"
# .obs key of 1D spatial coordinate:
UNS_KEY_SPLINE_KEY_1D = "spline_1d_coord"


PREFIX_INDEX = "index_"
PREFIX_NEIGHBOR = "neighbor_"
