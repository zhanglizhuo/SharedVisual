"""Shared statistical helpers (rank transformation, partial correlation,
bootstrap confidence intervals, Mantel test).

These were duplicated across analysis/ and analysis/ with subtly
different implementations (partial_corr had 4 variants, rankmean had 3).
This module is the canonical version.
"""
import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr


def rank_z(x):
    """Tie-corrected average ranks, z-scored (mean 0, std 1).

    This is the rank transformation used throughout the paper for
    Spearman-based mediation analysis.
    """
    r = rankdata(x, method="average")
    s = r.std(ddof=0)
    return (r - r.mean()) / (s if s > 0 else 1.0)


def mediation_paths(x, m, y):
    """Baron-Kenny product-of-coefficients mediation on z-scored ranks.

    Closed-form standardized multiple regression (X, M -> Y):
        beta_c' = (r_xy - r_xm*r_my) / (1 - r_xm^2)   [direct effect X->Y|M]
        beta_b  = (r_my - r_xm*r_xy) / (1 - r_xm^2)   [M->Y|X]
        a       = r_xm                                 [X->M]
        c       = r_xy                                 [total]
        indirect = a * beta_b   (equals c - c' exactly for standardized OLS)

    Returns dict with keys a, b, c, cprime, indirect, prop_mediated.
    prop_mediated = (c - cprime) / c (NaN if c ~= 0).
    Returns NaNs if X and M are near-collinear.
    """
    rx, rm, ry = rank_z(x), rank_z(m), rank_z(y)
    r_xm, _ = pearsonr(rx, rm)
    r_xy, _ = pearsonr(rx, ry)
    r_my, _ = pearsonr(rm, ry)
    denom = 1.0 - r_xm ** 2
    if abs(denom) < 1e-8 or abs(r_xm) > 0.999:
        return dict(a=r_xm, b=float("nan"), c=r_xy, cprime=float("nan"),
                    indirect=float("nan"), prop_mediated=float("nan"))
    beta_c = (r_xy - r_xm * r_my) / denom
    beta_b = (r_my - r_xm * r_xy) / denom
    a = r_xm
    c = r_xy
    indirect = a * beta_b
    prop = (c - beta_c) / c if abs(c) > 1e-9 else float("nan")
    return dict(a=a, b=beta_b, c=c, cprime=beta_c, indirect=indirect,
                prop_mediated=prop)


def partial_corr(x, y, covars):
    """Spearman partial correlation of x and y controlling for covars.

    Computed via residualization (OLS on ranks). covars is a 1D or 2D array.
    Returns the Pearson correlation of the residuals.
    """
    from numpy.linalg import lstsq
    rx, ry = rank_z(x), rank_z(y)
    covars = np.atleast_2d(covars).T if np.ndim(covars) == 1 else covars
    # Residualize x and y against [intercept, covars]
    design = np.column_stack([np.ones(len(rx)), rank_z(covars)])
    rx_res = rx - design @ lstsq(design, rx, rcond=None)[0]
    ry_res = ry - design @ lstsq(design, ry, rcond=None)[0]
    r, _ = pearsonr(rx_res, ry_res)
    return r


def bootstrap_ci(statistic_fn, n_boot, seed, ci=95):
    """Percentile bootstrap confidence interval.

    Args:
        statistic_fn: callable() -> float, draws a bootstrap sample and returns
            the statistic. Must close over the data and its own RNG.
        n_boot: number of bootstrap replicates.
        seed: random seed (used by the caller's RNG, not here).
        ci: confidence level (default 95 -> 2.5/97.5 percentiles).

    Returns:
        (lo, hi, samples) where samples is the array of valid bootstrap stats.
    """
    samples = []
    for _ in range(n_boot):
        try:
            v = statistic_fn()
        except Exception:
            continue
        if np.isfinite(v):
            samples.append(v)
    samples = np.array(samples)
    alpha = (100 - ci) / 2
    lo, hi = np.percentile(samples, [alpha, 100 - alpha])
    return lo, hi, samples


def mantel_rho(A, B):
    """Mantel statistic: Spearman correlation of the upper triangles of A, B."""
    n = A.shape[0]
    iu = np.triu_indices(n, k=1)
    r, _ = spearmanr(A[iu], B[iu])
    return r


def permute_matrix(mat, perm):
    """Simultaneously permute rows and columns of a square matrix."""
    return mat[np.ix_(perm, perm)]
