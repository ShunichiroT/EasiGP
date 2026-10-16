"""
EasiGP - models/kernel_effects.py
--------------------------------------------------------------------------
ver4-25, Stage 7 (design blueprint R13, re-scoped by finding D5): exact-
predictor marker-effect routes for GBLUP (`closed_form`) and RKHS
(`shap_predictor`/`shapiq_predictor`, both built on `RKHSPredictor`),
replacing the R-side `iml::Shapley` perturbation-refit route with routes
that need only the model's own already-completed main BGLR fit - no
per-coalition BGLR refits at all. This update (RKHS marker-effect/shapiq
comparability fix): what was originally one option, `shapiq_predictor`,
is now two - `shap_predictor` (the exact original implementation, renamed)
explains `RKHSPredictor` with `shap.KernelExplainer`; `shapiq_predictor`
(the new default) explains it with the real `shapiq` package instead
(`pipeline_utils.shapiq_marker_effects()`) - see `genomic_prediction.py::
_apply_rkhs_shapiq_predictor_effect()`'s own docstring for the full
rationale (the original `shapiq_predictor` never actually called `shapiq`
despite its name, unlike every other model's own `shapiq` marker-effect
choice).

Background (see `models/GBLUP.R`/`models/RKHS.R`'s own module-level
comments for the full cost analysis): because GBLUP/RKHS are transductive
kernel models (BGLR fits train+valid+test jointly, with test labels
masked), scoring a single Shapley-perturbed row the "obvious" way requires
rebuilding the whole kernel and re-running the ENTIRE BGLR MCMC chain from
scratch - roughly `2 * M * Shapley_num` full refits. The routes below need
exactly ONE fit (the model's normal main fit) plus, at most, cheap linear
algebra or a SHAP/shapiq explanation of a plain NumPy predict function.

GBLUP -> ``closed_form``
-------------------------
Linear-kernel GBLUP (``G = Z Z' / M``, `Z` = standardised markers, `M` =
marker count) is algebraically identical to ridge-regression BLUP
(rrBLUP) fitted on the SAME `Z` - the standard VanRaden (2008)
back-solving identity converts the fitted breeding values `u` (BGLR's own
``fm$ETA[[1]]$u`` - a plain vector already produced by the completed main
fit, no extra BGLR call) into per-marker effects:

    beta_hat = Z_train' @ G_train_inv @ u_train / M

For a model whose prediction is an EXACT linear function of `Z`
(``yhat = mu + Z @ beta_hat``), the Shapley value of marker `j` for a
query row `i`, relative to the training-mean baseline, has the closed
form (the same "linear model" special case KernelSHAP itself documents):

    phi_ij = beta_hat_j * z_ij         (z_ij standardised against TRAIN
                                         mean/sd, which is already exactly
                                         0-mean by construction, so no
                                         further centring term is needed)

No sampling, no coalitions, no refits - `beta_hat` is computed ONCE per
task, reused for every explained row. `gblup_closed_form_effects()` below
implements exactly this.

Caveat (blueprint V13.1): the exact scaling/centring convention used here
mirrors `GBLUP.R`'s own ``X <- scale(data_qtl, center=T, scale=T); G <-
(X %*% t(X)) / ncol(data_qtl)`` line for line - if that R-side convention
is ever changed, this module must change with it (both are the single
"how GBLUP's linear kernel is built" authority for their own runtime,
Python and R respectively, and are documented as needing to stay in
lockstep, the same discipline `_R_MODEL_BASE_PARAM_COUNT` already
requires elsewhere in `genomic_prediction.py`).

RKHS -> ``shapiq_predictor`` (``RKHSPredictor``)
--------------------------------------------------
``RKHSPredictor`` is a pure NumPy re-implementation of RKHS.R's own
fitted kernel regression (``K = exp(-h * D / mean(D))``, ``yhat = mu + K
(X, X_train) @ alpha``, ``alpha = K_train_train^{-1} @ u_train``) -
picklable (a module-level class holding only plain arrays, no closures,
no rpy2/R objects) so it can be explained with shapiq's/SHAP's own
row-level multiprocess fan-out. Blueprint V13.2 gates this at the call
site (`genomic_prediction.py`): this predictor's own test-set predictions
must reproduce RKHS.R's BGLR test predictions within
`RKHS_RECON_MC_TOL_REL * sd(y)` (default 5%) before it is trusted for an
explanation - a run where that check fails retries with the training-block
-only reconstruction (a NOTE either way), and only reports an all-zero
effect vector (with a NOTE) if that retry fails too. Bug fix (this update):
this NEVER falls back to the legacy `iml_shapley` route any more (a fresh,
expensive per-coalition BGLR refit this whole exact-predictor module
exists to avoid) - see `genomic_prediction.py::_build_gated_rkhs_
predictor()`'s and `_apply_rkhs_shapiq_predictor_effect()`'s own
docstrings for the full rationale and the two-tier retry this replaced it
with. Never silently returning a possibly-wrong effect vector either way.
Bug fix: this tolerance used to be a floating-point-level
`1e-6 * sd(y)` (correct for GBLUP's own exact closed form below, which is
fully deterministic) but was wrongly reused verbatim for RKHS, whose BGLR
fit is Monte-Carlo/MCMC estimated - `real_test` itself carries irreducible
MC sampling noise (measured at ~0.5-1.2% of sd(y) by direct simulation of
the same Gibbs sampler, at this file's default nIter/burnIn) that no
reconstruction can ever match to 1e-6 precision, so with the old tolerance
this gate failed on essentially every task, permanently forcing
`shapiq_predictor` back to `iml_shapley` - see the `RKHS_RECON_MC_TOL_REL`
constant in `genomic_prediction.py` for the full rationale and margin
analysis against a genuine reconstruction bug.

This update ("shapiq is unreliable and should not rely on this" - user
feedback on RKHS's/SVR's own 'shapiq_predictor'/'shapiq' interaction
choices): `rbf_kernel_hessian_interactions()` below is a THIRD, fully
closed-form marker-PAIR-interaction route for any model whose prediction
is an exact, known Gaussian/RBF-kernel function of its own training rows
- RKHSPredictor's own kernel here, and (separately) sklearn SVR's dual
decision function when `kernel='rbf'` (see `models/SVR.py`'s own
'kernel_hessian' choice). Because that function is analytically twice
differentiable, its own mixed second partial derivative
`d^2(predict)/dx_p dx_q` - the textbook definition of pairwise
interaction strength for a smooth function - can be computed EXACTLY, in
plain NumPy linear algebra, with NO coalition sampling, NO background/
budget knobs, and NO dependency on the optional `shapiq` package at all.
Unlike `shapiq`'s own KernelSHAP-style estimators, calling this function
twice on the same inputs always returns bit-identical output - there is
no run-to-run variance to be "unreliable" about, because nothing here is
estimated.

Both routes are validated defensively at every call - malformed/singular
inputs raise informative, run-stopping `ValueError`s (never a silently
wrong effect vector), matching this codebase's own "fatal, never silent"
convention (blueprint K1/K7 mitigations).
"""
import numpy as np

__all__ = [
    'standardize_markers', 'build_grm', 'gblup_closed_form_effects',
    'linear_closed_form_effects', 'RKHSPredictor', 'squared_euclidean_distance',
    'rbf_kernel_hessian_interactions', 'RKHSMaskedPredictor', 'rbf_kernel_occlusion_effects',
    'linear_kernel_occlusion_effects', 'poly_kernel_occlusion_effects',
    'sigmoid_kernel_occlusion_effects', 'poly_kernel_hessian_interactions',
    'sigmoid_kernel_hessian_interactions', 'zero_pair_interaction_table',
    'SVRBaselineShapFastEval', 'exact_kernel_column', 'clear_exact_kernel_memo',
]

# Default GRM blending fraction for `build_grm()`'s own Cholesky
# conditioning - matches this update's other GRM blends (see
# genomic_prediction.py's own `asreml_grm_blend` default of the same
# value, blueprint R7 §2). A standard, small ridge toward the identity,
# not a modelling choice specific to this module.
DEFAULT_GRM_BLEND = 0.01


def standardize_markers(X, *, center=True, scale=True, ddof=1):
    """Column-wise standardisation matching R's own `scale()` convention
    (sample, i.e. `ddof=1`, standard deviation) - the same convention
    `GBLUP.R`'s own ``scale(data_qtl, center=T, scale=T)`` call uses.

    A monomorphic (zero-variance) marker column is mapped to an all-zero
    standardised column, rather than R's own NaN/Inf-producing 0/0
    division - a deliberately SAFER deviation (real marker data reaching
    this point has already passed this codebase's own upstream QC - LD
    pruning / RF marker filtering - so a genuinely zero-variance marker
    is rare, and this only changes behaviour for that rare, otherwise-
    undefined case).

    Returns
    -------
    (Z, mean, sd) : each a 1-D/2-D `np.ndarray`; `Z` has `X`'s own shape.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"kernel_effects.standardize_markers: expected a 2-D marker matrix, "
                          f"got shape {X.shape}.")
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1])
    sd = X.std(axis=0, ddof=ddof) if scale else np.ones(X.shape[1])
    safe_sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mean) / safe_sd if (center or scale) else X.copy()
    if scale:
        Z[:, sd <= 0] = 0.0
    return Z, mean, sd


def build_grm(Z, *, blend=DEFAULT_GRM_BLEND):
    """``G = Z Z' / M`` (`M` = number of markers), blended toward the
    identity matrix for a numerically well-conditioned Cholesky inverse -
    standard GRM-blending practice, the same role this update's own
    ASReml GRM path already blends for (`genomic_prediction.py::
    _asreml_build_grm_for_universe`). `blend=0` reproduces the raw `G`
    exactly.
    """
    Z = np.asarray(Z, dtype=float)
    M = Z.shape[1]
    if M == 0:
        raise ValueError("kernel_effects.build_grm: zero markers - cannot build a relationship matrix.")
    G = (Z @ Z.T) / M
    if blend:
        G = (1.0 - blend) * G + blend * np.eye(G.shape[0])
    return G


def gblup_closed_form_effects(X_train, u_train, X_explain, *, scale, blend=DEFAULT_GRM_BLEND,
                               aggregate=True):
    """VanRaden (2008) back-solved marker effects for a linear-kernel
    GBLUP fit, then exact linear-model Shapley values from them.

    Parameters
    ----------
    X_train : (n_train, M) array-like
        Raw (unstandardised) training marker matrix - the SAME rows, in
        the SAME order, as the training block `GBLUP.R`'s own
        ``data_qtl`` restricts to.
    u_train : (n_train,) array-like
        The fitted model's own training-row breeding values (BGLR's
        ``fm$ETA[[1]]$u``, sliced to the training rows, in the SAME row
        order as `X_train`).
    X_explain : (n_explain, M) array-like
        Raw marker matrix for the rows being explained (e.g. the same
        test-row sample the legacy `iml_shapley` route explains).
    scale : int
        The SAME marker-count scaling `GBLUP.R`'s own
        ``G <- (X %*% t(X)) / ncol(data_qtl)`` divides by (V13.1) -
        passed explicitly, rather than re-derived from `X_train.shape[1]`,
        so a caller mismatch (e.g. a K_precomputed GPU kernel built over
        a different column set) is a visible `ValueError`, never a
        silent wrong scale.
    blend : float, default `DEFAULT_GRM_BLEND`
        `build_grm()`'s own blending fraction.
    aggregate : bool, default True
        True returns the per-marker SUM of `|phi_ij|` over every row of
        `X_explain` - the SAME aggregation the legacy `iml_shapley` route
        already reports (``colSums(abs(effect))`` in `GBLUP.R`), so
        magnitudes stay comparable across methods (AC13.1). False
        returns the full `(n_explain, M)` phi matrix instead (e.g. for
        testing/inspection).

    Returns
    -------
    1-D `np.ndarray` (`aggregate=True`, length `M`) or 2-D
    (`aggregate=False`, shape `(n_explain, M)`), in `X_train`'s own
    column order.

    Raises
    ------
    ValueError
        On a row/column-count mismatch, a non-positive `scale`, or a
        training-block relationship matrix that stays non-positive-
        definite even after `blend` (never silently falls back to an
        approximate/regularised solve beyond what `blend` already is).
    """
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    u_train = np.asarray(u_train, dtype=float).reshape(-1)
    if X_train.ndim != 2 or X_explain.ndim != 2:
        raise ValueError("kernel_effects.gblup_closed_form_effects: X_train/X_explain must be 2-D.")
    if X_train.shape[0] != u_train.shape[0]:
        raise ValueError(
            f"kernel_effects.gblup_closed_form_effects: X_train has {X_train.shape[0]} rows "
            f"but u_train has {u_train.shape[0]} values - these must describe the SAME "
            f"training individuals, in the SAME order.")
    if X_train.shape[1] != X_explain.shape[1]:
        raise ValueError(
            f"kernel_effects.gblup_closed_form_effects: X_train has {X_train.shape[1]} marker "
            f"columns but X_explain has {X_explain.shape[1]} - marker sets must match exactly.")
    if not scale or scale <= 0:
        raise ValueError(
            f"kernel_effects.gblup_closed_form_effects: scale must be a positive marker count, "
            f"got {scale!r}.")

    Z_train, mean, sd = standardize_markers(X_train, center=True, scale=True)
    G_train = build_grm(Z_train, blend=blend)
    try:
        L = np.linalg.cholesky(G_train)
        G_inv_u = np.linalg.solve(L.T, np.linalg.solve(L, u_train))
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"kernel_effects.gblup_closed_form_effects: the training-block relationship "
            f"matrix is not positive-definite even after blend={blend} - a larger blend "
            f"(ASREML['blue_refinement']-style GRM blending) is needed for this task's own "
            f"training population. ({exc})") from exc
    beta = (Z_train.T @ G_inv_u) / float(scale)

    safe_sd = np.where(sd > 0, sd, 1.0)
    Z_explain = (X_explain - mean) / safe_sd
    Z_explain[:, sd <= 0] = 0.0
    # phi_ij = beta_j * (z_ij - zbar_train_j); zbar_train_j == 0 exactly
    # by construction (Z_train is centred on its own mean above), so the
    # baseline term drops out and this reduces to beta_j * z_ij.
    phi = Z_explain * beta[np.newaxis, :]
    if not aggregate:
        return phi
    return np.abs(phi).sum(axis=0)


def linear_closed_form_effects(beta, X_explain, X_train, *, aggregate=True):
    """Exact linear-model Shapley values from an ALREADY-KNOWN marker
    effect vector `beta` (Requirements3.md item 1: rrBLUP/BayesB already
    output `beta` directly - BGLR's own ``fm$ETA[['mrk']][['b']]`` - so,
    unlike GBLUP, no relationship-matrix back-solving is needed here at
    all; this is the SAME closed-form SHAP formula
    `gblup_closed_form_effects()` uses, factored out so both share one
    implementation rather than two copies of the same three-line
    identity).

    Parameters
    ----------
    beta : (M,) array-like
        The model's own fitted per-marker effect vector, on the SAME
        standardised scale the model was actually fit on (rrBLUP/BayesB
        fit on `scale(data_qtl)/sqrt(ncol(data_qtl))` - see
        `models/rrBLUP.R`/`models/BayesB.R` - so `X_train`/`X_explain`
        below are standardised the SAME way before applying `beta`,
        never on the raw/un-standardised scale).
    X_explain : (n_explain, M) array-like
        Raw marker matrix for the rows being explained.
    X_train : (n_train, M) array-like
        Raw marker matrix the model was fitted on - supplies the
        standardisation reference (mean/sd) `X_explain` is measured
        against, mirroring `gblup_closed_form_effects()`'s own baseline.
    aggregate : bool, default True
        See `gblup_closed_form_effects()`.

    Returns
    -------
    1-D (`aggregate=True`) or 2-D (`aggregate=False`) `np.ndarray`, as
    `gblup_closed_form_effects()`.
    """
    beta = np.asarray(beta, dtype=float).reshape(-1)
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    if X_train.shape[1] != beta.shape[0] or X_explain.shape[1] != beta.shape[0]:
        raise ValueError(
            f"kernel_effects.linear_closed_form_effects: beta has {beta.shape[0]} markers but "
            f"X_train has {X_train.shape[1]} and X_explain has {X_explain.shape[1]} - all three "
            f"must describe the SAME marker set.")
    # rrBLUP/BayesB.R both standardise via `scale(data_qtl)/sqrt(ncol(data_qtl))`
    # rather than plain `scale()` - the extra `sqrt(ncol(...))` divisor is
    # a fixed, data-independent constant applied identically to every
    # marker, so it does not change which markers are "used" by the
    # standardisation step in standardize_markers() below (mean/sd are
    # computed the same way regardless); it only rescales beta's own
    # units, and beta itself already comes from BGLR fitted against that
    # exact scaling, so applying beta to `standardize_markers()`'s own
    # (unscaled-by-sqrt(ncol)) Z reproduces the SAME fitted linear
    # function BGLR actually used.
    _, mean, sd = standardize_markers(X_train, center=True, scale=True)
    safe_sd = np.where(sd > 0, sd, 1.0)
    scale_factor = 1.0 / np.sqrt(X_train.shape[1])
    Z_explain = ((X_explain - mean) / safe_sd) * scale_factor
    Z_explain[:, sd <= 0] = 0.0
    phi = Z_explain * beta[np.newaxis, :]
    if not aggregate:
        return phi
    return np.abs(phi).sum(axis=0)


def squared_euclidean_distance(A, B=None):
    """Squared Euclidean distance matrix via the algebraic identity
    ``D[i,j] = ||a_i||^2 + ||b_j||^2 - 2*(a_i . b_j)``, expressed as ONE
    matrix product - the same GEMM identity `RKHS.R`'s own
    ``.squared_euclidean_dist_gemm()`` uses (see that file's own
    module-level comment for the accuracy/perf rationale). `B=None`
    (default) computes the square, symmetric `A` vs. `A` distance matrix;
    otherwise the (possibly rectangular) `A` vs. `B` distance matrix.
    Tiny negative values from floating-point cancellation (a point's
    distance to itself) are clamped to 0.
    """
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    a_sq = np.sum(A * A, axis=1)
    b_sq = np.sum(B * B, axis=1)
    D = a_sq[:, np.newaxis] + b_sq[np.newaxis, :] - 2.0 * (A @ B.T)
    np.clip(D, 0.0, None, out=D)
    return D


def rbf_kernel_hessian_interactions(coeffs, X_train, X_explain, gamma, feature_names, *, top_k=None):
    """Closed-form, DETERMINISTIC marker-pair interaction strengths for
    any prediction function of the exact form

        f(x) = const + sum_i coeffs_i * exp(-gamma * ||x - x_i||^2)

    i.e. any fitted Gaussian/RBF-kernel regression - `RKHSPredictor`'s own
    kernel here (``K = exp(-h*D/d_scale)``, so ``gamma = h/d_scale``,
    ``coeffs = alpha``), or a scikit-learn ``SVR(kernel='rbf')``'s own
    dual decision function (``coeffs = dual_coef_.ravel()``,
    ``X_train = support_vectors_``, ``gamma`` = that fit's own resolved
    ``gamma`` value) - see `models/SVR.py`'s own 'kernel_hessian' choice.
    No sampling, no coalitions, no background/budget knobs, and no
    dependency on the optional `shapiq` package - this is the model's own
    EXACT analytic second derivative, not an estimate of one, so it never
    carries the run-to-run variance a Shapley-coalition sampler (`shapiq`)
    or a black-box finite-difference estimator (Friedman's H-statistic)
    can.

    Derivation
    ----------
    Write ``r_i(x) = x - x_i`` (an ``M``-length vector, ``M`` = number of
    columns explained) and ``k_i(x) = exp(-gamma * ||r_i(x)||^2)``. Then,
    for any two distinct marker columns ``p != q``:

        d^2f/dx_p dx_q (x) = 4 * gamma^2 * sum_i coeffs_i * k_i(x) * r_i,p(x) * r_i,q(x)

    - the standard mixed second partial derivative of a smooth function,
    which is exactly what Friedman's H-statistic and SHAP/shapiq
    interaction indices are both, in their own different ways, trying to
    ESTIMATE for a black-box model; here the function is known and
    differentiable, so this is available exactly instead. Vectorised per
    explained row as ``4*gamma^2 * (R^T @ diag(w) @ R)`` where
    ``R = x - X_train`` and ``w_i = coeffs_i * k_i(x)`` - `O(n_train * M^2)`
    per row, no worse than, and typically much cheaper than, a single
    `shapiq` coalition-sampling budget over the same shortlist.

    Parameters
    ----------
    coeffs : (n_train,) array-like
        The model's own already-fitted per-training-row coefficient.
    X_train : (n_train, M) array-like
        Raw marker matrix the coefficients are defined against (SAME row
        order as `coeffs`) - restricted to whatever marker shortlist the
        caller already narrowed to (this function does not shortlist on
        its own, exactly like every other interaction route in this
        codebase bounds its own O(M^2) pair count BEFORE calling in).
    X_explain : (n_explain, M) array-like
        Rows to evaluate the interaction Hessian at (e.g. a sampled test-
        row subset), restricted to the SAME `M` shortlisted columns.
    gamma : float
        The kernel bandwidth (`exp(-gamma * ||x-x'||^2)`).
    feature_names : sequence of str, length M
        Column names, in `X_train`'s own column order.
    top_k : int or None, default None
        If given, keep only the `top_k` largest-|value| pairs (mirrors
        `top_select()`'s own row-count bound elsewhere in this codebase).
        `None` (default) returns every pair.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']`` -
    the SAME canonical long-format table every other pairwise-interaction
    route in this codebase returns (see `models.interaction_extraction.
    canonicalize_shapiq_table()`'s own docstring for the shared
    convention). `value` is `mean(|d^2f/dx_p dx_q(x)|)` over every row of
    `X_explain`, sorted descending. Returns an empty (zero-row, correctly
    columned) DataFrame when `M < 2` (no pair to report) or `X_explain`
    has zero rows.

    Raises
    ------
    ValueError
        On a row/column-count mismatch or a non-positive `gamma` - never
        silently returns a wrong table.
    """
    import pandas as pd
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    feature_names = list(feature_names)
    if X_train.ndim != 2 or X_explain.ndim != 2:
        raise ValueError("kernel_effects.rbf_kernel_hessian_interactions: X_train/X_explain must be 2-D.")
    if X_train.shape[0] != coeffs.shape[0]:
        raise ValueError(
            f"kernel_effects.rbf_kernel_hessian_interactions: X_train has {X_train.shape[0]} rows "
            f"but coeffs has {coeffs.shape[0]} values - these must describe the SAME training rows.")
    if X_train.shape[1] != X_explain.shape[1] or X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"kernel_effects.rbf_kernel_hessian_interactions: X_train has {X_train.shape[1]} "
            f"columns, X_explain has {X_explain.shape[1]}, and feature_names has "
            f"{len(feature_names)} entries - all three must describe the SAME marker shortlist.")
    if not gamma or gamma <= 0:
        raise ValueError(f"kernel_effects.rbf_kernel_hessian_interactions: gamma must be positive, "
                          f"got {gamma!r}.")

    M = X_train.shape[1]
    empty = pd.DataFrame(columns=['marker1', 'marker2', 'value'])
    if M < 2 or X_explain.shape[0] == 0:
        return empty

    accum = np.zeros((M, M), dtype=float)
    gamma = float(gamma)
    for x in X_explain:
        R = x[np.newaxis, :] - X_train                       # (n_train, M)
        D = np.sum(R * R, axis=1)                             # (n_train,)
        k = np.exp(-gamma * D)                                # (n_train,)
        w = coeffs * k                                        # (n_train,)
        H = (4.0 * gamma * gamma) * (R.T * w) @ R              # (M, M)
        accum += np.abs(H)
    mean_abs_H = accum / X_explain.shape[0]
    np.fill_diagonal(mean_abs_H, 0.0)

    rows = [
        (feature_names[p], feature_names[q], mean_abs_H[p, q])
        for p in range(M) for q in range(p + 1, M)
    ]
    table = pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])
    table = table.sort_values('value', ascending=False).reset_index(drop=True)
    if top_k is not None and table.shape[0] > int(top_k):
        table = table.head(int(top_k)).reset_index(drop=True)
    return table


# Smallest eigenvalue this module will trust in an un-regularised kernel
# solve, expressed as a fraction of that same kernel's own largest
# eigenvalue - see `_solve_kernel_alpha()`'s own docstring for the full
# derivation of why this (not a training-set residual check) is the
# correct diagnostic here.
_MIN_EIG_RATIO = 1e-8


def _solve_kernel_alpha(K_train, u_train, *, blend=0.0, min_eig_ratio=_MIN_EIG_RATIO):
    """Solve ``K_train @ alpha = u_train`` for `alpha` via an
    eigenvalue-floored pseudo-solve - the shared training-block back-
    solve `RKHSPredictor.__init__` needs (see that class's own docstring
    for the surrounding derivation).

    Bug fix (silent ill-conditioning - root cause of the V13.2 gate's own
    large single-row reconstruction failures seen in production, e.g. a
    ~20,000-marker run with a structured, closely-related training
    population): a Gaussian/RBF kernel is strictly positive-definite for
    distinct rows (Bochner's theorem), so `np.linalg.cholesky(K_train)`
    always numerically "succeeds" even when `K_train` is severely
    ILL-CONDITIONED - positive-definiteness guarantees a successful
    factorisation, never a numerically ACCURATE one. Real marker data
    routinely has exactly the shape that produces this: with `M` (marker
    count) far larger than `n_train` and any family/population structure
    (full-/half-sib lines, near-isogenic cultivars), several training
    individuals can sit almost exactly on top of each other in marker
    space, making `K_train` positive-definite but numerically
    near-singular.

    This module's PREVIOUS fix attempt checked the solve's own
    TRAINING-set residual (``||K_train @ alpha - u_train||``) and only
    re-solved with a defensive blend if that residual was large. That
    check cannot actually detect this failure mode: a direct Cholesky
    solve is backward-stable, so it reproduces `u_train` to
    floating-point precision REGARDLESS of how ill-conditioned `K_train`
    is - conditioning affects the FORWARD error (how wrong `alpha` itself
    is relative to what a numerically well-behaved solve would have
    produced), not the training-set residual, and it is that forward
    error which then gets amplified when `alpha` is used to extrapolate
    to new (test) rows via `predict()`'s own `K(x, X_train) @ alpha` -
    exactly the "huge single test-row reconstruction mismatch, tiny
    training-set residual" signature this bug actually produces. The
    residual check therefore always reported "fine" and never caught it.

    Fix: diagnose conditioning DIRECTLY, from `K_train`'s own eigenvalues,
    rather than indirectly via a residual that cannot reflect it. `K_train`
    is symmetric, so `np.linalg.eigh` gives its exact eigendecomposition;
    any eigenvalue smaller than `min_eig_ratio` times the largest
    eigenvalue is floored up to that threshold before back-solving
    (``alpha = V @ diag(1/max(lambda_i, floor)) @ V.T @ u_train``,
    computed here by reconstructing a floored-but-still-exact `K_used`
    and Cholesky-factoring THAT instead - algebraically the same
    solve, just numerically stable in the near-null directions). This is
    deliberately SELECTIVE (only the few problematic near-zero
    eigen-directions are damped) rather than `build_grm()`'s own uniform
    "blend every eigenvalue toward 1" strategy GBLUP's closed-form route
    uses - RKHS's own kernel is well-conditioned in EVERY direction except
    the rare near-duplicate ones, so flooring only those, rather than
    perturbing every explained row's own effect by a fixed global amount,
    stays closer to the exact, unregularised solve BGLR itself
    effectively targets. `min_eig_ratio` also mirrors, in spirit, BGLR's
    own eigendecomposition-based RKHS solver, which applies its own
    numerical floor to near-zero eigenvalues internally - so this
    generally brings the reconstruction CLOSER to what BGLR's own fit
    actually produced, not further from it. When `K_train` is already
    well-conditioned (the ordinary case), every eigenvalue already
    exceeds the floor, so `K_used == K_train` exactly and this reduces to
    the previous unblended behaviour byte-for-byte.

    Parameters
    ----------
    K_train : (n_train, n_train) np.ndarray
        The training-block kernel matrix (symmetric; not necessarily
        positive-definite in the numerically-useful sense - that is
        exactly what this function checks for).
    u_train : (n_train,) np.ndarray
        The fitted model's own training-row values to back-solve against.
    blend : float, default 0.0
        An additional, caller-requested uniform blend toward the identity
        matrix, applied BEFORE the eigenvalue floor below (0.0 = solve
        the exact kernel BGLR itself fit against - the normal case; a
        caller wanting `build_grm()`-style uniform blending on top of
        this function's own floor may still request it explicitly).
    min_eig_ratio : float, default `_MIN_EIG_RATIO`
        The smallest eigenvalue this function will trust unfloored,
        as a fraction of `K_train`'s own largest eigenvalue.

    Returns
    -------
    (alpha, L, n_floored) : the solved coefficient vector, the Cholesky
        factor of the (possibly eigenvalue-floored) matrix actually
        solved (so a caller's own downstream use - e.g.
        `RKHSPredictor.conditional_variance()` - stays consistent with
        `alpha`), and how many of `K_train`'s own eigenvalues needed
        flooring (0 for an ordinary, well-conditioned task - surfaced for
        diagnostics/logging only, never behaviourally load-bearing).

    Raises
    ------
    ValueError
        If `K_train` (after `blend`, if any) has no positive eigenvalues
        at all (not merely ill-conditioned, but not positive
        semi-definite) - never a silently-wrong `alpha` (matching this
        module's own "fatal, never silent" convention).
    """
    n = K_train.shape[0]
    K = K_train if not blend else (1.0 - float(blend)) * K_train + float(blend) * np.eye(n)
    # Symmetric by construction (K_train = exp(-h*D), D symmetric) - eigh
    # is both cheaper and more numerically appropriate than a general
    # eig() call for a symmetric matrix, and conveniently returns
    # eigenvalues in ascending order.
    eigvals, eigvecs = np.linalg.eigh(K)
    lam_max = float(eigvals[-1])
    if lam_max <= 0:
        raise ValueError(
            "kernel_effects._solve_kernel_alpha: the training-block kernel matrix has no "
            "positive eigenvalues - it is not positive semi-definite.")
    floor = lam_max * float(min_eig_ratio)
    n_floored = int(np.sum(eigvals < floor))
    lam_used = np.maximum(eigvals, floor)
    if n_floored:
        # Reconstruct the floored-but-exact matrix and factor THAT -
        # guaranteed positive-definite (every eigenvalue >= floor > 0), so
        # this Cholesky call never raises, and `L` stays the single
        # source of truth `alpha`/`conditional_variance()` both use.
        K_used = (eigvecs * lam_used) @ eigvecs.T
        K_used = 0.5 * (K_used + K_used.T)  # defensive: undo any eig/recon round-trip asymmetry
        L = np.linalg.cholesky(K_used)
    else:
        L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, u_train))
    return alpha, L, n_floored


# BGLR's own default eigenvalue-truncation threshold for an RKHS/kernel
# ETA term (its `tolD` argument) - an ABSOLUTE cutoff on the weighted
# GRID kernel's own eigenvalues (see `BGLR.R`'s source: `tmp <-
# LT$d > LT$tolD; LT$V <- LT$V[, tmp]; LT$d <- LT$d[tmp]`), not a
# fraction of the largest eigenvalue the way `_MIN_EIG_RATIO` above is.
# `models/RKHS.R`'s own `BGLR(...)` call never overrides `tolD`, so this
# is the exact value BGLR itself truncates against for every RKHS task in
# this codebase - `solve_rkhs_grid_alpha()` below must use the SAME
# absolute threshold to replicate BGLR's own truncation exactly, not an
# approximation of it.
RKHS_GRID_TOL_D = 1e-10


def solve_rkhs_grid_alpha(X_grid, weights_full, y_train, mu, h, d_scale, var_u, var_e, n_train,
                           *, tol_d=RKHS_GRID_TOL_D):
    """Reconstruct `RKHS.R`'s own fitted kernel regression EXACTLY,
    including BGLR's own weighted-grid eigenvalue truncation - the fix
    for a systematic, batch-wide V13.2 gate fallback that the training-
    block-only `_solve_kernel_alpha()`/`RKHSPredictor.__init__` route
    above cannot reproduce, no matter how well-conditioned its own
    training-block solve is made.

    Root cause (why the training-block-only reconstruction is
    STRUCTURALLY wrong, not just numerically fragile)
    -------------------------------------------------
    `RKHS.R`'s own `BGLR(y=y_test, ETA=list(list(K=K, model='RKHS')),
    weights=w_full, ...)` call does not fit a plain, unweighted GP on the
    training block alone. Per BGLR's real internals (`BGLR.R`, `RKHS`
    branch):

      1. The kernel is built over the FULL `rbind(train, valid, test)`
         grid (`K`, `n = n_train + n_valid + n_test` rows/cols) - already
         reflected in `d_scale` (`RKHS.R`'s own `mean(D)` over the full
         grid) and in `RKHSPredictor`'s own docstring.
      2. Every row's own fitting weight - `sqrt(w_fit)` for training rows
         (BLUE-reliability weighting), `1` for valid/test rows - is
         applied to the KERNEL ITSELF, not just to the residual variance:
         `K' = diag(weights_full) @ K @ diag(weights_full)`. This is the
         standard reparametrisation for a heteroscedastic-residual GP
         (`v = diag(weights_full) @ u`, `z = diag(weights_full) @ (y -
         mu)`, so that `z = v + eps` with `eps` i.i.d. `N(0, varE)` -
         `weights_full[i] = sqrt(w_fit_i)` exactly cancels the `1/w_fit_i`
         residual-variance scaling BLUE-reliability weighting introduces,
         which is also why BGLR divides the fitted `u` back by
         `weights_full` before returning it).
      3. BGLR eigendecomposes `K'` (not `K`) and TRUNCATES: any
         eigenvector whose eigenvalue is below `tolD` (`RKHS_GRID_TOL_D`
         above; `RKHS.R` never overrides BGLR's own default) is dropped
         BEFORE the Gibbs sampler ever runs - i.e. the model BGLR actually
         fits is the projection of `K'` onto its own top eigenspace, not
         `K'` itself.

      Real marker data (related breeding lines, population structure,
      clonal/near-isogenic individuals) routinely makes `K'` numerically
      rank-deficient enough for step 3 to drop a meaningful number of
      eigenvectors - independent of whether BLUE-reliability weighting is
      active or how heterogeneous its weights are (confirmed: the
      fallback this fixes was reported both under highly heterogeneous
      weights AND under near-uniform ones, and "across the whole batch" -
      i.e. this is a property of the population's own kernel, not of the
      weighting). The training-block-only reconstruction above never
      truncates anything - it back-solves the FULL-RANK, UNWEIGHTED
      `K_train` exactly - so whenever BGLR's own truncation actually
      drops eigenvectors that carry real train/valid/test cross-
      information, the two reconstructions diverge by a genuine, sytematic
      BIAS (not Monte Carlo noise the V13.2 gate's tolerance can average
      away), which is exactly the "row N: |reconstruction - real BGLR
      prediction| exceeds tolerance" signature reported even at a task's
      near-uniform-weight, small-magnitude worst case.

    Derivation of the fix (Nystrom-consistent full-grid reconstruction)
    ---------------------------------------------------------------
    Let `P = V_kept @ V_kept.T` be the projector onto BGLR's own kept
    eigenspace of `K'` (`n x n`, `n` = grid size). The exact posterior
    mean of `v` at the training rows, under BGLR's own truncated model,
    is standard ridge regression in that (possibly lower-rank) eigenbasis:

        gamma = (K'_trunc[train, train] + (varE/varU) I)^-1 @ z_train

    where `K'_trunc = V_kept @ diag(d_kept) @ V_kept.T` and `z_train =
    weights_full[train] * (y_train - mu)`. The posterior mean of `v` at
    ANY point `x` - including points that were never part of the original
    grid, e.g. a SHAP/shapiq synthetic coalition sample - extends via the
    standard Nystrom formula. Writing that extension out and simplifying
    (the `1/eigenvalue` terms cancel exactly, leaving no near-zero
    denominator to worry about) collapses to:

        v_hat(x) = K(x, GRID) @ diag(weights_full) @ [P[:, train] @ gamma]
        u_hat(x) = v_hat(x) / weight_x = v_hat(x)   (weight_x == 1 for any
                                                      new/out-of-grid point,
                                                      exactly as it is for
                                                      valid/test rows)

    i.e. `predict(x) = mu + K(x, GRID) @ alpha_grid` with

        alpha_grid = weights_full * (V_kept @ (V_kept[train, :].T @ gamma))

    - a single length-`n` (grid-sized) coefficient vector, computed once
    per task and reused for every explained row/coalition sample exactly
    like the training-block-only `alpha` it replaces, just evaluated
    against the FULL grid's raw kernel (`RKHSPredictor.X_grid`/`alpha_grid`
    /`X_ref`/`alpha_ref` - see that class's own docstring) instead of the
    training block alone. Verified numerically (synthetic near-duplicate-
    row populations, forcing genuine `tolD` truncation): this formula
    reproduces an independently-computed truncated-eigenbasis posterior
    mean to machine precision (`~1e-14` max abs diff) at held-out grid
    rows, where the training-block-only back-solve above was off by
    orders of magnitude more than the V13.2 gate's own MC-noise tolerance
    once truncation actually occurred - and reduces to the SAME answer as
    the training-block-only route (to machine precision) whenever `K'`
    happens to be full-rank at `tol_d` (no truncation - the ordinary,
    well-conditioned case), so this is a strict generalisation, not a
    behaviour change, for any task where BGLR's own fit never truncated
    anything.

    Deliberately does NOT reuse `r_u` (`RKHS.R`'s own `fm$ETA[[1]]$u`,
    the training-block back-solve's own target) at all: `r_u` is itself
    only a finite-MCMC-chain posterior MEAN estimate (carries its own
    Monte Carlo noise), whereas `gamma`/`alpha_grid` above are derived
    directly from the observed training phenotypes (`y_train`) and this
    fit's own fitted `mu`/`varU`/`varE` - the ANALYTIC target the Gibbs
    sampler is itself trying to estimate. This is a strictly more direct
    (and, empirically, more accurate) route than back-solving through
    `r_u`'s own sampling noise a second time.

    Parameters
    ----------
    X_grid : (n_train + n_valid + n_test, M) array-like
        Raw marker matrix for the FULL grid, in `RKHS.R`'s own `rbind(
        train, valid, test)` row order - i.e. `np.vstack([X_train,
        X_valid, X_test])` (an empty `X_valid` when this task has no
        validation split is fine).
    weights_full : (n_grid,) array-like
        This fit's own per-row BGLR fitting weight, in `X_grid`'s row
        order - `sqrt(w_fit)` for training rows (BLUE-reliability
        weighting active) or `1` (inactive), followed by `1` for every
        valid/test row - i.e. `RKHS.R`'s own `w_full` exactly.
    y_train : (n_train,) array-like
        The REAL, observed training phenotypes (not `r_u` - see above).
    mu : float
        This fit's own `fm$mu` (`r_mu`).
    h, d_scale : float
        This fit's own kernel bandwidth/distance-scale (`RKHS.R`'s own
        `params[3]`/`r_d_scale`) - `K(a, b) = exp(-h * ||a-b||^2 /
        d_scale)`.
    var_u, var_e : float
        This fit's own fitted variance components (`r_var_u[0]`,
        `r_var_e[0]`) - `var_u` must be strictly positive (raises
        `ValueError` otherwise; the caller's own `have_var_components`
        check should already guarantee this before calling in).
    n_train : int
        Row count of the training block at the START of `X_grid`/
        `weights_full` (`X_grid[:n_train]` are the training rows).
    tol_d : float, default `RKHS_GRID_TOL_D`
        BGLR's own absolute eigenvalue-truncation threshold - only
        overridable for testing; every real caller should use the
        default (BGLR's own, since `RKHS.R` never changes it).

    Returns
    -------
    (alpha_grid, n_kept, n_dropped) : the length-`n_grid` coefficient
        vector `RKHSPredictor` needs (`X_grid=`/`alpha_grid=` kwargs), and
        how many of `K'`'s own eigenvalues were kept/dropped at `tol_d` -
        surfaced so a caller can log a NOTE when truncation actually
        occurred (diagnostics only, never behaviourally load-bearing).

    Raises
    ------
    ValueError
        On a shape mismatch, a non-positive `d_scale`/`var_u`, or if
        EVERY eigenvalue of the weighted grid kernel falls below `tol_d`
        (a degenerate fit) - never a silently-wrong `alpha_grid`.
    """
    X_grid = np.asarray(X_grid, dtype=float)
    weights_full = np.asarray(weights_full, dtype=float).reshape(-1)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    n = X_grid.shape[0]
    if weights_full.shape[0] != n:
        raise ValueError(
            f"kernel_effects.solve_rkhs_grid_alpha: X_grid has {n} rows but weights_full has "
            f"{weights_full.shape[0]} values.")
    n_train = int(n_train)
    if n_train <= 0 or n_train > n:
        raise ValueError(
            f"kernel_effects.solve_rkhs_grid_alpha: n_train={n_train} is not a valid training-row "
            f"count for a {n}-row grid.")
    if y_train.shape[0] != n_train:
        raise ValueError(
            f"kernel_effects.solve_rkhs_grid_alpha: n_train={n_train} but y_train has "
            f"{y_train.shape[0]} values.")
    if not d_scale or d_scale <= 0:
        raise ValueError(f"kernel_effects.solve_rkhs_grid_alpha: d_scale must be positive, got {d_scale!r}.")
    var_u = float(var_u)
    if var_u <= 0:
        raise ValueError(f"kernel_effects.solve_rkhs_grid_alpha: var_u must be positive, got {var_u!r}.")
    var_e = float(var_e)

    D = squared_euclidean_distance(X_grid) / float(d_scale)
    K_raw = np.exp(-float(h) * D)
    Kp = (weights_full[:, np.newaxis] * K_raw) * weights_full[np.newaxis, :]
    Kp = 0.5 * (Kp + Kp.T)  # defensive: undo any GEMM round-trip asymmetry

    eigvals, eigvecs = np.linalg.eigh(Kp)
    keep = eigvals > float(tol_d)
    n_kept = int(np.sum(keep))
    n_dropped = n - n_kept
    if n_kept == 0:
        raise ValueError(
            "kernel_effects.solve_rkhs_grid_alpha: every eigenvalue of the weighted grid kernel "
            "fell below tol_d - this fit is degenerate.")
    V_k = eigvecs[:, keep]
    vals_k = eigvals[keep]

    V_train = V_k[:n_train, :]
    K_trunc_tt = (V_train * vals_k) @ V_train.T
    K_trunc_tt = 0.5 * (K_trunc_tt + K_trunc_tt.T)

    # varE/varU expressed in the kernel's own unit scale - the same ridge
    # term `RKHSPredictor.conditional_variance()`'s own weighted solve
    # uses (see that method's docstring). A non-positive varE (never
    # expected from a real BGLR fit, whose priors keep it strictly
    # positive - defensive only) falls back to a tiny relative floor
    # rather than solving an unregularised, potentially singular system.
    ridge = var_e / var_u if var_e > 0 else 1e-8 * float(np.trace(K_trunc_tt)) / max(n_train, 1)

    z_train = weights_full[:n_train] * (y_train - float(mu))
    M = K_trunc_tt + ridge * np.eye(n_train)
    gamma = np.linalg.solve(M, z_train)

    beta = V_k @ (V_train.T @ gamma)  # == P[:, :n_train] @ gamma, without forming P (n x n)
    alpha_grid = weights_full * beta
    return alpha_grid, n_kept, n_dropped


class RKHSPredictor:
    """Picklable, pure-NumPy re-implementation of `RKHS.R`'s own fitted
    kernel regression - see the module docstring for the full
    derivation. Holds only plain `np.ndarray`/`float` attributes (no
    closures, no rpy2/R objects), so it can be pickled across a
    `multiprocessing`/`joblib` process boundary for shapiq's/SHAP's own
    row-level parallel explanation fan-out (I10 - never threads, since
    this module makes no BGLR/R calls itself and has no PID-based
    file-path requirement, but the SAME "must survive a process
    boundary" requirement the rest of this codebase's Shapley fan-out
    already applies).

    Attributes
    ----------
    X_train : (n_train, M) np.ndarray
        Raw (unstandardised) training marker matrix - kernel distances
        to a query row are computed directly against this.
    alpha : (n_train,) np.ndarray
        ``K_train_train^{-1} @ u_train`` (solved once, at construction).
    mu : float
        The fitted model's overall intercept (BGLR's ``fm$mu``).
    h : float
        The Gaussian kernel bandwidth (`RKHS.R`'s own tunable `h`
        hyperparameter, params[3]/hparam index 2).
    d_scale : float
        The distance-matrix normalising constant `RKHS.R` divides by
        BEFORE exponentiating (``D <- D/mean(D)``) - computed over the
        FULL train+valid+test distance matrix at fit time (not just
        train), so it must be supplied from that same fit, not
        re-derived from `X_train` alone (V13.1-style caveat - see
        `genomic_prediction.py`'s own construction call site).
    _n_floored_eigs : int
        Diagnostic-only (leading underscore - never behaviourally read by
        `predict()`/`conditional_variance()`, which only use `alpha`/
        `L_train`): 0 unless `_solve_kernel_alpha()`'s own eigenvalue
        floor fired for this fit (near-singular training-block kernel -
        see that function's own docstring), in which case this is how
        many of `K_train`'s own eigenvalues needed flooring. A caller may
        surface this in a log/NOTE line for diagnostics.
    K_train : (n_train, n_train) np.ndarray
        The raw (unfloored) training-block kernel matrix - kept only so
        `conditional_variance()` can build its own, separately-
        regularised solve (see that method's own docstring for why it
        must NOT reuse `alpha`'s possibly eigenvalue-floored `L_train`
        unless no per-row weighting information is available) without
        recomputing `K_train` from `X_train`/`h`/`d_scale` a second time.
    X_grid, alpha_grid, X_ref, alpha_ref, uses_grid_reconstruction,
    n_grid_dropped_eigs :
        Bug fix (systematic, batch-wide V13.2 gate fallback - see
        `solve_rkhs_grid_alpha()`'s own module-level docstring for the
        full root-cause derivation): when the caller supplies `X_grid`/
        `alpha_grid` below (the normal case - `genomic_prediction.py`'s
        `_build_gated_rkhs_predictor()` computes them via that function
        whenever this fit's own `varU`/`varE` are available), `predict()`
        uses THOSE (`X_ref`/`alpha_ref` point at `X_grid`/`alpha_grid`)
        instead of the training-block-only `X_train`/`alpha` above, which
        cannot reproduce BGLR's own eigenvalue-truncated grid-kernel fit
        whenever truncation actually occurs. `X_train`/`alpha`/`K_train`/
        `L_train` above are ALWAYS still computed (used by
        `conditional_variance()`'s V13.2 gate-tolerance calibration,
        which is unaffected by this fix - that check calibrates how much
        Monte Carlo noise a training-block-conditioned GP should show,
        not the point-estimate reconstruction itself) and used as
        `predict()`'s own fallback (`X_ref`/`alpha_ref` point at them
        instead) whenever `X_grid`/`alpha_grid` are not supplied - I11:
        constructing an `RKHSPredictor` the old way (positional args
        only) reproduces the previous, training-block-only behaviour
        byte-for-byte.

        `X_grid` : (n_train + n_valid + n_test, M) np.ndarray or None
            The full `rbind(train, valid, test)`-order marker matrix
            `solve_rkhs_grid_alpha()` was built from.
        `alpha_grid` : (n_grid,) np.ndarray or None
            That function's own returned coefficient vector.
        `n_grid_dropped_eigs` : int
            How many of the weighted grid kernel's own eigenvalues fell
            below BGLR's own `tolD` for this fit (0 unless truncation
            actually occurred) - diagnostics only, mirrors
            `_n_floored_eigs` above. Always 0 when `X_grid`/`alpha_grid`
            are not supplied.
        `uses_grid_reconstruction` : bool
            Whether `predict()`/`X_ref`/`alpha_ref` are using the full-
            grid route (`True`) or the training-block-only fallback
            (`False`) - surfaced so a caller can decide whether to trust
            `X_ref`/`alpha_ref` as the SAME reference set an interaction
            route (e.g. `rbf_kernel_hessian_interactions()`) must pair
            `alpha_ref` with.
        `X_ref`/`alpha_ref` :
            Whichever of the two pairs above `predict()` actually uses -
            the single source of truth any OTHER caller needing this
            model's own `(reference_rows, coefficients)` pair (e.g.
            `genomic_prediction.py`'s `kernel_hessian` interaction route)
            must use instead of `X_train`/`alpha` directly, so it stays
            consistent with whichever reconstruction `predict()` itself
            is using for this task.
    """

    def __init__(self, X_train, u_train, mu, h, d_scale, *, blend=0.0,
                 X_grid=None, alpha_grid=None, n_grid_dropped_eigs=0):
        self.X_train = np.asarray(X_train, dtype=float)
        u_train = np.asarray(u_train, dtype=float).reshape(-1)
        if self.X_train.shape[0] != u_train.shape[0]:
            raise ValueError(
                f"kernel_effects.RKHSPredictor: X_train has {self.X_train.shape[0]} rows but "
                f"u_train has {u_train.shape[0]} values.")
        if not d_scale or d_scale <= 0:
            raise ValueError(f"kernel_effects.RKHSPredictor: d_scale must be positive, got {d_scale!r}.")
        self.mu = float(mu)
        self.h = float(h)
        self.d_scale = float(d_scale)

        D_train = squared_euclidean_distance(self.X_train) / self.d_scale
        K_train = np.exp(-self.h * D_train)
        # Bug fix (V13.2 gate false-positive, ver4-25 Stage 7): unlike
        # GBLUP's linear GRM (Z Z', which CAN be rank-deficient when
        # M < n and genuinely needs build_grm()'s identity blend to
        # invert), RKHS's own kernel K = exp(-h*D) is a Gaussian/RBF
        # kernel - strictly positive-definite for distinct rows by
        # construction (Bochner's theorem). RKHS.R's own BGLR fit (see
        # that file's `BGLR(... K=K ...)` call) uses this exact,
        # UNBLENDED K - so solving for `alpha` against a blended K here
        # was solving a DIFFERENT linear system than the one that
        # actually produced `u_train`, silently breaking the V13.2
        # exact-reconstruction guarantee - which is why `blend` defaults
        # to 0 (solve the SAME exact kernel BGLR fit) rather than always
        # blending the way GBLUP does.
        #
        # Bug fix (this update - root cause of the V13.2 gate's own large
        # single-row reconstruction failures seen in production, e.g. a
        # ~20,000-marker run with a structured, closely-related training
        # population): positive-definiteness is NOT the same guarantee as
        # well-conditioned - Bochner's theorem above only promises the
        # FORMER. `np.linalg.cholesky` "succeeding" on a near-singular
        # `K_train` (near-duplicate training individuals in marker space,
        # routine once M >> n_train with any family/population structure)
        # still numerically succeeds, but the resulting `alpha` can be
        # wrong by orders of magnitude on whichever rows the
        # ill-conditioning happens to affect most - undetectable from
        # `LinAlgError` alone, which this code used to rely on
        # exclusively, and (a first fix attempt found) ALSO undetectable
        # from the training-set solve residual, since a direct solve is
        # backward-stable regardless of conditioning - conditioning only
        # shows up as FORWARD error in `alpha` itself, which then gets
        # amplified when extrapolating to new (test) rows. Fixed for real
        # by `_solve_kernel_alpha()` (module-level, above), which floors
        # only the near-zero eigenvalues of `K_train` that actually cause
        # this, leaving every well-conditioned direction exactly as
        # before - see its own docstring for the full derivation.
        self.alpha, L, self._n_floored_eigs = _solve_kernel_alpha(K_train, u_train, blend=blend)
        # Kept for conditional_variance() below (genomic_prediction.py's
        # V13.2 gate) - the SAME Cholesky factor `_solve_kernel_alpha()`
        # already computed (of whichever, possibly eigenvalue-floored,
        # matrix it actually solved), not recomputed, so the gate's
        # kernel is always identical to the one predict()/alpha actually
        # used.
        self.L_train = L
        # Bug fix (V13.2 gate false-positive under BLUE-reliability
        # weighting - see conditional_variance()'s own docstring): kept,
        # raw/unfloored, so a weighted call to conditional_variance() can
        # build its OWN noise-regularised solve without recomputing
        # `K_train` a second time from X_train/h/d_scale.
        self.K_train = K_train

        # Bug fix (systematic, batch-wide V13.2 gate fallback) - see this
        # class's own docstring and `solve_rkhs_grid_alpha()`'s module-
        # level docstring for the full derivation. `X_ref`/`alpha_ref` are
        # the single pair `predict()` (and any external caller needing
        # this model's own reference-rows/coefficients, e.g. the
        # `kernel_hessian` interaction route) actually uses.
        if X_grid is not None and alpha_grid is not None:
            self.X_grid = np.asarray(X_grid, dtype=float)
            self.alpha_grid = np.asarray(alpha_grid, dtype=float).reshape(-1)
            if self.X_grid.shape[0] != self.alpha_grid.shape[0]:
                raise ValueError(
                    f"kernel_effects.RKHSPredictor: X_grid has {self.X_grid.shape[0]} rows but "
                    f"alpha_grid has {self.alpha_grid.shape[0]} values.")
            self.n_grid_dropped_eigs = int(n_grid_dropped_eigs)
            self.uses_grid_reconstruction = True
            self.X_ref = self.X_grid
            self.alpha_ref = self.alpha_grid
        else:
            self.X_grid = None
            self.alpha_grid = None
            self.n_grid_dropped_eigs = 0
            self.uses_grid_reconstruction = False
            self.X_ref = self.X_train
            self.alpha_ref = self.alpha

    def predict(self, X):
        """`predict(X) = mu + K(X, X_ref) @ alpha_ref` - `X_ref`/
        `alpha_ref` are the full grid (`X_grid`/`alpha_grid`) when this
        instance was built with them (the normal case - see this class's
        own docstring), else the training-block-only `X_train`/`alpha`
        (backward-compatible fallback). `X`: `(n, M)` array-like, raw
        (unstandardised) marker matrix, same column order as `X_ref`.
        Returns a 1-D `np.ndarray`, length `n`."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        D = squared_euclidean_distance(X, self.X_ref) / self.d_scale
        K = np.exp(-self.h * D)
        return self.mu + K @ self.alpha_ref

    def conditional_variance(self, X, var_u=1.0, *, var_e=None, weights=None):
        """Standard GP "posterior predictive variance" of `u(x)` at each
        row of `X`, GIVEN this exact training-block kernel - the part of
        `u(x)`'s variance NOT explained by the training rows:

            var_u * (k_xx - k(x, X_train) @ K_reg^{-1} @ k(X_train, x))

        `k_xx` (each row's own kernel self-similarity) is always exactly 1
        for this Gaussian/RBF kernel (``exp(-h*0) == 1`` regardless of
        `h`/`d_scale`), so this reduces to `var_u * (1 - quad)` below.
        `var_u` defaults to 1 (unit/"kernel-scale" variance - the caller
        multiplies in the model's own fitted `varU` separately if it wants
        it already applied, as `genomic_prediction.py`'s V13.2 gate does).

        Used ONLY by that V13.2 gate: it turns BGLR's own fitted variance
        components (`fm$ETA[[1]]$varU`/`fm$varE` - see `RKHS.R`'s
        `need_reconstruction` block) into a per-ROW, statistically-
        grounded tolerance for how much Monte Carlo noise a given test
        row's reconstructed prediction can be expected to carry, instead
        of one fixed, dataset-agnostic constant (which is what used to
        force EVERY `shapiq_predictor` task back onto `iml_shapley` - see
        that gate's own module-level comment for the full derivation and
        calibration evidence). A row far from every training individual
        (large `conditional_variance`) is expected to be noisier and gets
        a looser tolerance; a row close to several training individuals
        gets a tighter one - exactly the "one worst-covered test row
        blows the whole task's gate" failure mode a single global
        constant cannot represent.

        Bug fix (V13.2 gate false-positive under BLUE-reliability
        weighting, e.g. `ASREML['blue_weighting'] in ('full', 'diag')`):
        `K_reg` used to always be `self.L_train` (the plain, UNREGULARISED
        training kernel `_solve_kernel_alpha()` already factored for
        `alpha`) - i.e. this treated every training individual's own
        fitted `u_train` value as if it had been observed with IDENTICAL,
        noise-free precision. That is only true when every training row
        was fit with the same weight (the byte-identical case this
        function still reproduces exactly when `weights`/`var_e` are not
        supplied - I11). Under BLUE-reliability weighting, training
        individuals can carry per-row fitting weights spanning two orders
        of magnitude (`w_fit`, mean-normalised to 1) - `RKHS.R`'s own
        `BGLR(..., weights=sqrt(w_fit), ...)` call makes each training
        row's own effective residual variance `varE / w_fit_i`, i.e. a
        LOW-weight training individual's own `u_train` estimate is
        genuinely noisier evidence than a high-weight one's, even though
        both feed the SAME shared kernel. The old, unregularised formula
        had no way to reflect that: a test row whose nearest training
        neighbours happened to carry low fitting weight got the SAME
        (too-tight) tolerance as one neighboured by high-weight
        individuals, which is exactly the "reconstruction gate rejects a
        perfectly good reconstruction" failure this fix addresses - the
        gate was comparing a reconstruction built from noisier training
        evidence against a flat, noise-blind expected-error budget.

        Fix: when both `var_e` and `weights` are supplied, solve against
        `K_reg = K_train + diag((var_e/var_u) / weights)` instead of the
        plain `K_train` - the textbook heteroscedastic-noise GP posterior
        variance (the SAME `var_e_i = var_e / weights_i` residual variance
        `RKHS.R`'s own weighted BGLR call already fits against, just
        expressed here in the kernel's own unit scale by dividing through
        by `var_u`). A row near LOW-weight (noisy) training neighbours
        now correctly gets a WIDER tolerance; a row near high-weight
        (reliable) ones stays tight, exactly like before. `weights=None`
        (the default - a config with no BLUE-reliability weighting, or an
        older caller predating this fix) reduces byte-identically to the
        original, unregularised formula (I11) - this is a strict
        generalisation of it, not a behaviour change for an unweighted
        fit.

        Parameters
        ----------
        X : (n, M) array-like
            Raw marker matrix for the rows being explained (same column
            order as `X_train`).
        var_u : float, default 1.0
            The fitted kernel/marker variance component to scale by
            (BGLR's own `fm$ETA[[1]]$varU`) - pass 1.0 to get the
            unit-scale conditional variance instead.
        var_e : float or None, keyword-only, default None
            The fitted residual variance component (BGLR's own `fm$varE`)
            - required (together with `weights`) to build the
            heteroscedastic regularisation above; `None` (default) keeps
            the original, unregularised behaviour.
        weights : (n_train,) array-like or None, keyword-only, default
            None - this fit's own per-TRAINING-row fitting weight (the
            SAME `w_fit`/`weights` vector `RKHS.R`'s own `BGLR(...,
            weights=sqrt(w_fit), ...)` call used, in `X_train`'s own row
            order, BEFORE the sqrt - i.e. residual variance for training
            row `i` is `var_e / weights[i]`). `None` (default) keeps the
            original, unregularised behaviour. Any non-finite or
            non-positive entry is floored to a small positive value
            (defensive - never lets one degenerate weight raise here).

        Returns
        -------
        1-D `np.ndarray`, length `n`, each entry >= 0 (clipped - tiny
        negative values from floating-point cancellation, since the exact
        conditional variance is mathematically never negative, are
        rounding noise only).
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        D = squared_euclidean_distance(X, self.X_train) / self.d_scale
        K_x_train = np.exp(-self.h * D)

        L = self.L_train
        if weights is not None and var_e is not None and var_u and float(var_u) > 0:
            w = np.asarray(weights, dtype=float).reshape(-1)
            if w.shape[0] == self.K_train.shape[0]:
                safe_w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
                floor = np.nanmin(safe_w) if np.any(np.isfinite(safe_w)) else 1.0
                safe_w = np.where(np.isfinite(safe_w), safe_w, floor)
                lambda_unit = (float(var_e) / float(var_u)) / safe_w
                K_reg = self.K_train + np.diag(lambda_unit)
                try:
                    # Adding a strictly positive diagonal term can only
                    # IMPROVE conditioning relative to the plain K_train
                    # `_solve_kernel_alpha()` already factored - this
                    # almost never raises - but a defensive fallback to
                    # the unregularised `L_train` (rather than letting a
                    # LinAlgError escape from inside a "never fail the
                    # task" gate helper) costs nothing when it doesn't.
                    L = np.linalg.cholesky(K_reg)
                except np.linalg.LinAlgError:
                    pass
            # A shape mismatch (defensive only - every real caller passes
            # THIS fit's own w_fit, already aligned to X_train) silently
            # keeps the unregularised `L_train` above rather than raising
            # from inside a "never fail the task" gate helper.

        # k(x, train) @ K_reg^{-1} @ k(train, x), every row at once, via
        # the (possibly noise-regularised) Cholesky factor above.
        tmp = np.linalg.solve(L, K_x_train.T)
        quad = np.sum(tmp * tmp, axis=0)
        cond_var_unit = np.clip(1.0 - quad, 0.0, None)
        return float(var_u) * cond_var_unit

    # A callable instance works directly as a shap/shapiq "predict_fn" -
    # both libraries accept either a bound method or a plain callable;
    # this makes `RKHSPredictor(...)` itself usable as one without an
    # extra `.predict` attribute lookup at the call site.
    def __call__(self, X):
        return self.predict(X)


class RKHSPredictCore:
    """A minimal, cheaply-picklable stand-in for `RKHSPredictor.predict()` -
    carries ONLY the four arrays/scalars `predict()` actually reads
    (`X_ref`, `alpha_ref`, `mu`, `h`, `d_scale`), never `RKHSPredictor`'s
    own `X_train`, `K_train`, `L_train`, or (when the training-block-only
    reconstruction is in use) the separate `alpha` - every one of those is
    needed only by `conditional_variance()` (the V13.2 reconstruction-
    fidelity gate, evaluated once, in the MAIN process, before any
    `joblib`/`shap`/`shapiq` fan-out ever starts), never by `predict()`
    itself.

    Bug fix (RKHS marker-effect/interaction Shapley fan-out "un-serialize"
    failure, 2026-09): `genomic_prediction.py`'s own `_masked_predict()`
    closures (used by `baseline_shap_interactions()`, `h_statistic_
    interactions()`, `shap.KernelExplainer`, and `shapiq_marker_effects()`
    alike, whenever `n_jobs > 1` fans explained rows out across `joblib`
    worker processes) used to close over the FULL `RKHSPredictor` instance
    - `K_train`/`L_train` are `(n_train, n_train)`, and `X_train`/`X_grid`
    are `(n_train[+n_valid+n_test], M)` at this fit's FULL original marker
    width `M` (not the, typically much narrower, interaction/effect
    shortlist - `RKHSPredictor` cannot be refit on a marker subset, see
    that class's own docstring). On a whole-genome marker panel this is
    routinely hundreds of MB to several GB, re-pickled once per dispatched
    row-block and reconstructed independently by every one of `n_jobs`
    concurrently-running worker processes - `joblib`'s own `loky` backend
    reports a worker that runs out of memory reconstructing (unpickling)
    such a payload as `BrokenProcessPool("A task has failed to
    un-serialize...")`, indistinguishable at the call site from an
    actually-unpicklable argument, which is why `baseline_shap_
    interactions()`'s/`h_statistic_interactions()`'s own printed NOTE
    named it that way. This is a DIFFERENT failure mode from the
    already-fixed `_rkhs_masked_predict_batched()` OOM/SIGKILL bug (see
    that function's own module-level comment) - that fix bounds a single
    `predictor.predict()` call's OWN row count; this one bounds what has
    to cross the process boundary AT ALL, at every call site that fans a
    masked RKHS predictor out across `n_jobs` worker processes.

    Fixed by building ONE of these (a few small arrays, no `K_train`/
    `L_train`/unused `X_train`/`alpha`) from an already-gated `RKHSPredictor`
    BEFORE defining the `_masked_predict` closure, so that is what
    actually gets pickled/reconstructed per dispatched block, per worker -
    numerically identical `predict()` values (same `X_ref`/`alpha_ref`/
    `mu`/`h`/`d_scale`, same formula), just without the extra baggage
    `predict()` never reads. Never used for `conditional_variance()`
    (the V13.2 gate keeps using the real `RKHSPredictor`, in the main
    process, before this core is ever built).
    """

    __slots__ = ('X_ref', 'alpha_ref', 'mu', 'h', 'd_scale')

    def __init__(self, predictor):
        self.X_ref = predictor.X_ref
        self.alpha_ref = predictor.alpha_ref
        self.mu = predictor.mu
        self.h = predictor.h
        self.d_scale = predictor.d_scale

    def predict(self, X):
        """Byte-identical to `RKHSPredictor.predict()` - see that
        method's own docstring."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        D = squared_euclidean_distance(X, self.X_ref) / self.d_scale
        K = np.exp(-self.h * D)
        return self.mu + K @ self.alpha_ref

    def __call__(self, X):
        return self.predict(X)


# Perf fix (RKHS 'baseline_shap'/'friedman_h_predictor' interaction routes,
# and 'shap_predictor'/'shapiq_predictor' effect routes - all of them
# funnel through `genomic_prediction.py`'s own `_masked_predict()` closure
# whenever a marker shortlist narrower than this fit's full marker set is
# being explained): the SAME `n_features_full / len(shortlist)`
# multiplier that `_rkhs_masked_predict_batched()`'s own module-level
# comment already names as the root cause of an OOM/SIGKILL crash (see
# that function's docstring in `genomic_prediction.py`) is ALSO a pure
# compute-time cost, independent of whether it ever gets big enough to hit
# an OOM: every single masked `predict()` call, at ANY safe batch size,
# still spends `O(n_rows * n_grid * n_features_full)` work inside
# `squared_euclidean_distance()`, even though `n_features_full -
# len(shortlist)` of those columns are IDENTICAL (the fixed `baseline`
# vector) on every single row the masking trick ever builds - that
# memory-bounding fix only stops the multiplier from OOM-killing a
# worker, it does not remove the redundant FLOPs the multiplier spends on
# every call that does not OOM. On a whole-genome marker panel, where
# that ratio is routinely 40-100x (the exact figure
# `_rkhs_masked_predict_batched()`'s own comment measures), and where
# `baseline_shap_interactions()` alone makes `1 + M' + P` such calls PER
# EXPLAINED ROW (see that function's own Cost docstring,
# `models/interaction_extraction.py`), this is the single largest lever
# available for RKHS's masked-predictor interaction/effect routes.
#
# `RKHSMaskedPredictor` below removes it algebraically rather than just
# bounding it: `squared_euclidean_distance(a, b) = ||a||^2 + ||b||^2 -
# 2*a.b` is additively SEPARABLE across disjoint column groups, so for a
# row `a` that equals a FIXED `baseline` everywhere except the shortlist
# columns,
#
#     ||a - b_j||^2 = ||a_short - b_j_short||^2 + ||baseline_rest - b_j_rest||^2
#
# where "_short"/"_rest" mean "restricted to the shortlist"/"restricted
# to everything else". The second (non-shortlist) term never depends on
# which row of the batch is being scored - only on `baseline` and grid
# row `j`, both FIXED for the whole task - so it can be folded into a
# single length-`n_grid` constant, computed ONCE (not once per `predict()`
# call), leaving only the shortlist-width terms to be recomputed per
# query row. Concretely (full derivation, one line at a time):
#
#     D0[j]      = ||baseline - X_ref[j]||^2                      (full width, ONCE)
#     C[j]       = D0[j] - ||baseline_short||^2 + 2*(X_ref_short[j] . baseline_short)
#     D[i, j]    = ||X_short[i]||^2 - 2*(X_short[i] . X_ref_short[j]) + C[j]
#
# (`X_ref_short = X_ref[:, shortlist_idx]`, `baseline_short =
# baseline[shortlist_idx]`) reproduces `||a_i - X_ref[j]||^2` EXACTLY (to
# floating-point precision - verified directly against
# `_rkhs_masked_predict_batched()`'s own tile-and-overwrite computation on
# synthetic fixtures, not merely derived) for every row `a_i` built the
# masking-trick way, at `O(n_rows * n_grid * len(shortlist))` cost instead
# of `O(n_rows * n_grid * n_features_full)` - the `D0`/`C` precomputation
# itself costs `O(n_grid * n_features_full)`, but that happens ONCE per
# task (in `__init__` below), not once per `predict()` call, so it is
# amortised away entirely once more than a handful of calls are made (this
# codebase's own masked-predictor interaction routes make hundreds to
# tens of thousands).
#
# A second, smaller benefit: this class's own per-call working memory
# scales with `n_grid` (train+valid+test row count - typically a few
# hundred to a few thousand), not `n_features_full` (a whole-genome marker
# count, typically far larger) - `predict()`'s own internal batching
# (mirroring `_rkhs_masked_predict_batched()`'s own `max_elements` knob,
# just re-based on `n_grid` instead of `n_features_full`) therefore allows
# much larger batches per call before hitting the same memory ceiling,
# meaning FEWER, LARGER `predict_fn` calls for the same total row count -
# lower fixed per-call overhead on top of the FLOP reduction above. A
# third: an instance of this class is also a substantially SMALLER
# `joblib`-fan-out payload than `RKHSPredictCore` (see that class's own
# docstring for the "un-serialize" failure this already had to fix once) -
# `X_ref_short` is `(n_grid, len(shortlist))`, not `RKHSPredictCore`'s own
# `(n_grid, n_features_full)`.
_RKHS_MASKED_PREDICT_FAST_MAX_ELEMENTS = 20_000_000  # ~160MB per float64 sub-batch (D/K), matches
# `genomic_prediction.py`'s own `_RKHS_MASKED_PREDICT_MAX_ELEMENTS` - same
# reasoning, just re-based on n_grid instead of n_features_full (see the
# module-level comment above).


class RKHSMaskedPredictor:
    """Fast, exact drop-in replacement for repeatedly calling
    `predictor.predict(X_full)` where `X_full` is a shortlist-width
    `X_short` expanded back to a FIXED `baseline` at every non-shortlist
    column (the masking trick every RKHS masked-predictor interaction/
    effect route in this codebase uses, since `RKHSPredictor` cannot be
    refit on a marker subset - see that class's own docstring). See the
    module-level comment directly above this class for the full
    derivation of why this is both exact and much cheaper.

    Built ONCE per task (from an already-gated `predictor` - either a
    `RKHSPredictor` or a `RKHSPredictCore`, anything exposing `X_ref`/
    `alpha_ref`/`mu`/`h`/`d_scale` - `predict()` never reads anything
    else), then called once per `predict_fn` invocation exactly like
    `_rkhs_masked_predict_batched()`'s own closure convention.

    Parameters
    ----------
    predictor : RKHSPredictor or RKHSPredictCore
        Must be numerically consistent with `baseline`/`shortlist_idx`
        below (the same fit, same `X_train_full` mean) - this class
        performs no cross-checking of its own, exactly like
        `_rkhs_masked_predict_batched()`.
    baseline : (n_features_full,) array-like
        The fixed value every non-shortlist column takes in every masked
        query row (this codebase's own convention: `X_train_full.mean(
        axis=0)` - see `genomic_prediction.py`'s own construction site).
    shortlist_idx : (M,) array-like of int
        Positional column indices (into `baseline`'s/`predictor.X_ref`'s
        own column order) that actually vary - i.e. the marker shortlist
        being explained.

    Attributes/behaviour
    ---------------------
    Callable (`__call__` == `predict`), so an instance works directly as
    a `predict_fn` for `baseline_shap_interactions()`/`h_statistic_
    interactions()`/`shap.KernelExplainer`/`shapiq`, exactly like
    `RKHSPredictor`/`RKHSPredictCore` already are.
    """

    __slots__ = ('alpha_ref', 'mu', 'h', 'd_scale', 'X_ref_short', 'C', 'n_grid')

    def __init__(self, predictor, baseline, shortlist_idx):
        baseline = np.asarray(baseline, dtype=float)
        shortlist_idx = np.asarray(shortlist_idx, dtype=int)
        X_ref = np.asarray(predictor.X_ref, dtype=float)
        if baseline.shape[0] != X_ref.shape[1]:
            raise ValueError(
                f"kernel_effects.RKHSMaskedPredictor: baseline has {baseline.shape[0]} entries "
                f"but predictor.X_ref has {X_ref.shape[1]} columns - these must describe the "
                f"SAME full marker set.")
        if shortlist_idx.ndim != 1 or shortlist_idx.shape[0] == 0:
            raise ValueError(
                "kernel_effects.RKHSMaskedPredictor: shortlist_idx must be a non-empty 1-D array "
                "of column indices.")

        self.alpha_ref = np.asarray(predictor.alpha_ref, dtype=float).reshape(-1)
        self.mu = float(predictor.mu)
        self.h = float(predictor.h)
        self.d_scale = float(predictor.d_scale)
        self.n_grid = X_ref.shape[0]
        self.X_ref_short = np.ascontiguousarray(X_ref[:, shortlist_idx])

        baseline_short = baseline[shortlist_idx]
        baseline_short_sq = float(baseline_short @ baseline_short)
        baseline_dot_short = self.X_ref_short @ baseline_short  # (n_grid,)
        # Full-width, ONE-ROW-vs-grid distance - cheap (O(n_grid *
        # n_features_full), but n_grid rows only, computed once here, not
        # once per predict() call).
        D0 = squared_euclidean_distance(baseline.reshape(1, -1), X_ref).ravel()
        self.C = D0 - baseline_short_sq + 2.0 * baseline_dot_short  # (n_grid,)

    def predict(self, X_short, *, max_elements=_RKHS_MASKED_PREDICT_FAST_MAX_ELEMENTS):
        """`predict(X_short)` is numerically identical to
        `_rkhs_masked_predict_batched(predictor, baseline, shortlist_idx,
        X_short)` for the SAME `(predictor, baseline, shortlist_idx)` this
        instance was built from - see the module-level comment above this
        class for the derivation. `X_short`: `(n, len(shortlist_idx))`
        array-like. Returns a 1-D `np.ndarray`, length `n`."""
        X_short = np.asarray(X_short, dtype=float)
        if X_short.ndim == 1:
            X_short = X_short.reshape(1, -1)
        n_rows = X_short.shape[0]
        max_rows = max(1, int(max_elements) // max(1, self.n_grid))
        if n_rows <= max_rows:
            return self._predict_block(X_short)
        out = np.empty(n_rows, dtype=float)
        for start in range(0, n_rows, max_rows):
            end = min(start + max_rows, n_rows)
            out[start:end] = self._predict_block(X_short[start:end])
        return out

    def _predict_block(self, X_short):
        short_sq = np.einsum('ij,ij->i', X_short, X_short)
        cross = X_short @ self.X_ref_short.T  # (n_rows, n_grid)
        D = short_sq[:, np.newaxis] - 2.0 * cross
        D += self.C[np.newaxis, :]
        np.clip(D, 0.0, None, out=D)
        K = np.exp((-self.h / self.d_scale) * D)
        return self.mu + K @ self.alpha_ref

    def __call__(self, X_short):
        return self.predict(X_short)

    def as_rbf_occlusion_kernel(self):
        """Re-express this masked predictor's own kernel as a PURE
        `exp(-gamma * ||x_short - ref_j||^2)` form over shortlist-width
        vectors only - i.e. the `(coeffs, X_train, gamma)` triple
        `rbf_kernel_occlusion_effects()` below expects - so that function's
        own closed-form occlusion-effect derivation (built for a plain,
        un-masked RBF kernel) applies directly to the MASKED predictor too,
        without re-deriving anything: `predict()`'s own `D[i, j] =
        ||x_short_i - X_ref_short_j||^2 + N_j` (see this class's own
        module-level derivation - `N_j` is the FIXED, per-reference-row
        contribution from every non-shortlist column, baked into `C`
        alongside the pure shortlist term `||X_ref_short_j||^2`) means
        `K(x_short, ref_j) = exp(-gamma * ||x_short - ref_j||^2) *
        exp(-gamma * N_j)` - i.e. this predictor's own kernel is a plain
        RBF kernel over shortlist-width vectors, TIMES a fixed,
        `j`-only (never `x`-dependent) rescaling of each reference row's
        own coefficient. Folding that rescaling into the coefficient
        vector (`coeffs_eff_j = alpha_ref_j * exp(-gamma * N_j)`) recovers
        exactly the plain form, so `rbf_kernel_occlusion_effects(coeffs_eff,
        X_ref_short, X_explain_short, baseline_short, gamma)` reproduces
        THIS class's own `predict()` (up to the constant `mu`, which
        cancels in any occlusion difference either way) - see
        `genomic_prediction.py`'s own RKHS marker-effect call site for how
        this is used.

        Returns
        -------
        (coeffs_eff, X_ref_short, gamma) : `coeffs_eff` and `X_ref_short`
        are fresh `np.ndarray`s (this predictor's own arrays are never
        mutated); `gamma = self.h / self.d_scale`.
        """
        gamma = self.h / self.d_scale
        ref_short_sq = np.einsum('ij,ij->i', self.X_ref_short, self.X_ref_short)
        N = self.C - ref_short_sq  # fixed, per-reference-row, non-shortlist contribution
        coeffs_eff = self.alpha_ref * np.exp(-gamma * N)
        return coeffs_eff, self.X_ref_short, gamma


# ---------------------------------------------------------------------------
# Perf fix (RKHS 'shap_predictor' / SVR 'kernel_shap' marker-EFFECT routes at
# very large marker counts - "extracting marker effects using shap_predictor
# is computationally expensive [for RKHS] ... the same problem exists for SVR
# and KNN when using KernelSHAP ... minimise the computational cost when the
# total number of markers is > 20,000"):
#
# `pipeline_utils.fast_kernel_shap_effects()` already replaced the per-row
# `shap.KernelExplainer` fan-out above `FAST_KERNEL_SHAP_MARKER_THRESHOLD`
# (10,000) with an EXACT per-marker occlusion difference, `phi_j(x) = f(x) -
# f(x^{j->baseline})` (every column at `x`'s own value except column `j`,
# reset to a single baseline row) - see that function's own module-level
# comment for why this (not a smaller random-coalition WLS estimate) is what
# it computes. That is already linear in the number of `predict_fn` CALLS
# (one batched call per marker, chunked), but for RKHS's/SVR-rbf's own
# Gaussian/RBF-kernel `predict_fn` specifically, EVERY one of those calls
# still independently recomputes a squared-Euclidean distance over the FULL
# `n_features` width from scratch (`squared_euclidean_distance()` has no
# memory of the row it was just asked to score `n_features - 1` other
# columns of) - i.e. the occlusion loop's real cost is `O(n_rows * n_features
# * n_ref * n_features) = O(n_rows * n_ref * n_features^2)`, quadratic in the
# marker count despite being "only" `n_features` `predict_fn` calls, exactly
# the kind of hidden quadratic blowup `RKHSMaskedPredictor` above already
# fixed for the MASKED-shortlist case (a narrow shortlist inside a wide
# baseline) - here the "shortlist" IS the full marker set (this route is what
# 'Max markers considered for Shapley scores' = 'all' reaches), so that fix
# does not apply, but the SAME kind of algebraic shortcut still does, one
# level down: a single-column change is a RANK-1 update to the DISTANCE
# itself, not just to which columns are "fixed".
#
# Concretely: squared Euclidean distance is additively separable over
# columns, so for a query row `x`, a reference row `b`, and a single
# occluded column `j` (reset from `x[j]` to `baseline[j]`, every other column
# unchanged):
#
#     ||x - b||^2               = D_x[b]                         (computed ONCE per query row)
#     ||x^{j->baseline} - b||^2 = D_x[b] - (x[j]-b[j])^2 + (baseline[j]-b[j])^2
#
# i.e. the occluded distance is a CHEAP, closed-form correction to the
# already-computed full distance - no need to recompute a fresh `O(n_ref *
# n_features)` distance matrix for every one of the `n_features` markers.
# Vectorised across every marker AND every reference row at once (see
# `rbf_kernel_occlusion_effects()` below), this cuts the per-explained-row
# cost from `O(n_ref * n_features^2)` to `O(n_ref * n_features)` - LINEAR in
# the marker count, exactly the regime "> 20,000 markers" needs. Verified
# numerically IDENTICAL (bit-for-bit, to floating-point precision) to
# `fast_kernel_shap_effects()`'s own occlusion computation for the SAME
# `predict_fn`/inputs on a Gaussian/RBF-kernel model - this is a routing/
# algebra change only, never a change in what is computed, so it is used as
# a drop-in, ALWAYS-faster (never slower, never approximating further)
# replacement wherever the underlying model qualifies (RKHS's `RKHSPredictor`
# /`RKHSMaskedPredictor`/`RKHSPredictCore` - always an RBF kernel, see this
# module's own header; scikit-learn `SVR(kernel='rbf')` - see `models/SVR.py`'s
# own identical `kernel_hessian` gate for the same kernel restriction and
# resolved-`gamma` lookup convention this function's own callers reuse).
# Falls back to `fast_kernel_shap_effects()` with a printed NOTE on any
# failure (an unexpected shape, a non-RBF model passed in by mistake, etc.) -
# never a new failure mode for a route that used to work.
def rbf_kernel_occlusion_effects(coeffs, X_train, X_explain, baseline, gamma, *,
                                  max_batch_elements=2e8):
    """Exact, closed-form per-marker occlusion effect - `sum over explained
    rows of |f(x) - f(x^{j->baseline})|` for every marker `j` - for any
    prediction function of the exact form `f(x) = const + sum_i coeffs_i *
    exp(-gamma * ||x - x_i||^2)` (`RKHSPredictor`'s/`RKHSMaskedPredictor`'s
    own kernel, or a fitted `sklearn.svm.SVR(kernel='rbf')`'s own dual
    decision function - see `rbf_kernel_hessian_interactions()`'s own
    identical parameter convention/docstring, which this function mirrors
    for the marker-EFFECT side of the same closed-form idea). See the
    module-level comment directly above this function for the full
    derivation of why this reproduces `pipeline_utils.fast_kernel_shap_
    effects()`'s own occlusion computation exactly, at `O(n_rows * n_ref *
    n_features)` cost instead of that function's own `O(n_rows * n_ref *
    n_features^2)` for this specific (Gaussian/RBF-kernel) class of model.

    Parameters
    ----------
    coeffs : (n_ref,) array-like
        The model's own already-fitted per-reference-row coefficient
        (`RKHSPredictor.alpha_ref` / `RKHSMaskedPredictor`'s own internal
        equivalent, or `SVR.dual_coef_.ravel()`).
    X_train : (n_ref, n_features) array-like
        Raw marker matrix the coefficients are defined against (SAME row
        order as `coeffs`) - `RKHSPredictor.X_ref` or `SVR.support_
        vectors_`. This function does no shortlisting of its own - pass
        the SAME (possibly already-narrowed) marker set `X_explain`/
        `baseline` below use.
    X_explain : (n_rows, n_features) array-like
        Rows to explain (e.g. this task's own `shap.sample(...)` draw -
        the SAME rows a caller would otherwise hand to `fast_kernel_shap_
        effects()`/`shap.KernelExplainer`).
    baseline : (n_features,) array-like
        The single reference/background row every occluded column is
        reset to - the SAME "weighted background mean" convention
        `fast_kernel_shap_effects()`'s own `baseline` parameter already
        uses (this function reproduces that exact quantity, so a caller
        should pass the identical baseline vector for a bit-identical
        result).
    gamma : float
        The kernel bandwidth (`exp(-gamma * ||x-x'||^2)`) - `RKHSPredictor.
        h / RKHSPredictor.d_scale`, or the fitted SVR's own resolved
        `gamma` (see `models/SVR.py`'s own `kernel_hessian` branch for how
        to resolve this from `gam` when `SVR.gamma` was `'scale'`/`'auto'`,
        not a plain float).
    max_batch_elements : float, keyword-only, default 2e8
        Caps the largest single intermediate `(marker_chunk, n_ref)` array
        this function ever materialises (~1.6GB at the default for
        float64) - markers are processed in chunks bounded by this whenever
        `n_features * n_ref` would otherwise exceed it, exactly mirroring
        `fast_kernel_shap_effects()`'s own identical memory-bounding
        contract (never affects the returned values, only how many
        intermediate chunks they are computed in).

    Returns
    -------
    1-D `np.ndarray`, length `n_features` (`X_train`'s own column count) -
    `sum over rows of |phi_ij|`, bit-identical (to floating-point
    precision) to what `fast_kernel_shap_effects(predict_fn, ...)` returns
    for the SAME model/inputs, where `predict_fn` is this model's own
    plain `.predict()`.

    Raises
    ------
    ValueError
        On a shape mismatch or non-positive `gamma` - never a silently
        wrong effect vector.
    """
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.atleast_2d(np.asarray(X_explain, dtype=float))
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    if X_train.ndim != 2:
        raise ValueError("kernel_effects.rbf_kernel_occlusion_effects: X_train must be 2-D.")
    n_ref, n_features = X_train.shape
    if X_train.shape[0] != coeffs.shape[0]:
        raise ValueError(
            f"kernel_effects.rbf_kernel_occlusion_effects: X_train has {n_ref} rows but coeffs "
            f"has {coeffs.shape[0]} values - these must describe the SAME reference rows.")
    if X_explain.shape[1] != n_features or baseline.shape[0] != n_features:
        raise ValueError(
            f"kernel_effects.rbf_kernel_occlusion_effects: X_train has {n_features} columns, "
            f"X_explain has {X_explain.shape[1]}, and baseline has {baseline.shape[0]} entries - "
            f"all three must describe the SAME marker set.")
    if not gamma or gamma <= 0:
        raise ValueError(f"kernel_effects.rbf_kernel_occlusion_effects: gamma must be positive, "
                          f"got {gamma!r}.")
    gamma = float(gamma)

    if n_features == 0:
        return np.zeros(0)
    if X_explain.shape[0] == 0:
        return np.zeros(n_features)

    X_train_T = np.ascontiguousarray(X_train.T)  # (n_features, n_ref) - reused for every row/chunk
    feat_chunk = max(1, int(max_batch_elements // max(n_ref, 1)))
    phi_abs_sum = np.zeros(n_features)

    for x in X_explain:
        # Full-width distance/kernel value at this row - computed ONCE (not
        # once per occluded marker, unlike the generic occlusion loop this
        # replaces).
        D_x = squared_euclidean_distance(x.reshape(1, -1), X_train).ravel()  # (n_ref,)
        K_x = np.exp(-gamma * D_x)
        f_x = float(K_x @ coeffs)  # intercept/`mu` term cancels in phi below either way

        for start in range(0, n_features, feat_chunk):
            end = min(start + feat_chunk, n_features)
            cols = slice(start, end)
            # (x[j] - X_train[k,j])^2 and (baseline[j] - X_train[k,j])^2 for
            # every marker j in this chunk and every reference row k at
            # once - shape (chunk_width, n_ref).
            term_x = (x[cols, np.newaxis] - X_train_T[cols, :]) ** 2
            term_b = (baseline[cols, np.newaxis] - X_train_T[cols, :]) ** 2
            D_occ = D_x[np.newaxis, :] - term_x + term_b
            np.clip(D_occ, 0.0, None, out=D_occ)
            K_occ = np.exp(-gamma * D_occ)  # (chunk_width, n_ref)
            f_occ = K_occ @ coeffs  # (chunk_width,) - one occluded prediction per marker
            phi_abs_sum[cols] += np.abs(f_x - f_occ)

    return phi_abs_sum


# ---------------------------------------------------------------------------
# Perf fix (this update - "the current optimisation is done mainly for the
# case where the kernel is 'rbf' ... minimise the computational time of
# marker effects and interactions for other kernel cases: linear, poly, or
# sigmoid"): `rbf_kernel_occlusion_effects()`/`rbf_kernel_hessian_
# interactions()` above only apply to a Gaussian/RBF-kernel decision
# function - every OTHER SVR kernel this codebase supports (`models/SVR.py`'s
# own `ker` field) used to fall all the way back to the generic, black-box
# routes (`pipeline_utils.fast_kernel_shap_effects()` for marker effects,
# `models.interaction_extraction.h_statistic_interactions()` for marker-pair
# interactions whenever 'kernel_hessian' itself was requested) whenever
# `kernel != 'rbf'` - correct, but paying the SAME `O(n_ref * n_features^2)`
# occlusion cost (marker effects) or per-`predict()`-call cost (interactions)
# the rbf fix was written to remove.
#
# The fact making a closed form possible for 'linear'/'poly'/'sigmoid' too:
# every one of scikit-learn's SVR kernels other than 'rbf' has the common
# form `K(x, x_i) = g(gamma * (x . x_i) + coef0)` for some scalar function
# `g` -
#
#     'linear'  : g(z) = z              (sklearn's own 'linear' kernel fixes
#                                         gamma=1, coef0=0 and ignores both)
#     'poly'    : g(z) = z ** degree
#     'sigmoid' : g(z) = tanh(z)
#
# and `z_i(x) = gamma * (x . x_i) + coef0` is an AFFINE (linear-plus-constant)
# function of `x` for every one of these three kernels - unlike `rbf`'s own
# `z_i(x) = -gamma * ||x - x_i||^2`, which is quadratic in `x`. That single
# difference is what makes both routes below cheaper, and simpler, than their
# `rbf` counterparts:
#
#   * Occlusion (marker effects): `z_i(x^{j->baseline})` is a one-line, EXACT
#     correction to the already-computed `z_i(x)` (`z_i(x) - gamma * x_i,j *
#     (x_j - baseline_j)` - no squared term, no "distance so far" bookkeeping
#     needed at all), so the SAME `O(n_ref * n_features)` linear-in-`M` win
#     the `rbf` fix achieves applies here too, for every one of
#     'linear'/'poly'/'sigmoid'. See `_dot_product_kernel_occlusion_effects()`
#     below (`linear_kernel_occlusion_effects()` is cheaper still - see its
#     own docstring).
#
#   * Interaction (marker Hessian): because `z_i(x)` is affine in `x`, its own
#     second derivative is exactly zero, so the mixed second partial of
#     `f = const + sum_i coeffs_i * g(z_i(x))` has NO product-rule cross term
#     at all (unlike `rbf`'s own `R = x - x_i`, which itself varies with
#     `x`) - it reduces to `gamma^2 * sum_i coeffs_i * g''(z_i(x)) * x_i,p *
#     x_i,q`, i.e. `gamma^2 * X_train.T @ diag(w(x)) @ X_train` for `w_i(x) =
#     coeffs_i * g''(z_i(x))` - simpler than `rbf`'s own Hessian formula (no
#     `R`, just the raw support vectors), and, for 'linear' (`g''=0`
#     identically, since `g` itself is linear), exactly ZERO for every pair
#     and every `x` - a linear model has no pairwise feature interactions by
#     construction, so that case is answered with NO computation at all
#     (not even the per-row loop is run) rather than a fast approximation of
#     "no interaction". See `_dot_product_kernel_hessian_interactions()`/
#     `poly_kernel_hessian_interactions()` below.
#
# Both routes are exact (never a further approximation of what
# `fast_kernel_shap_effects()`/`h_statistic_interactions()`/
# `baseline_shap_interactions()` already compute/estimate) and are wired into
# `models/SVR.py` as try-first, drop-in replacements with automatic fallback
# to the existing route on any failure - the same "never a new failure mode"
# convention the rbf fix above already established.
# ---------------------------------------------------------------------------

def linear_kernel_occlusion_effects(w, X_explain, baseline):
    """Exact per-marker occlusion effect for an EXACTLY linear predictor
    `f(x) = const + w . x` (scikit-learn `SVR(kernel='linear')`'s own
    `.coef_.ravel()`, already `dual_coef_ @ support_vectors_` under the
    hood) - see the module-level comment above for the kernel family this
    specialises.

    For a linear function, `f(x) - f(x^{j->baseline}) = w_j * (x_j -
    baseline_j)` EXACTLY - no dual-coefficient/support-vector loop is
    needed at all (unlike `poly_kernel_occlusion_effects()`/
    `sigmoid_kernel_occlusion_effects()` below, whose `g` is genuinely
    non-linear): this is `O(n_rows * n_features)`, the true minimum
    possible cost for this class of model, cheaper even than the
    `O(n_rows * n_ref * n_features)` the other two closed forms need.

    Parameters
    ----------
    w : (n_features,) array-like
        The fitted linear-kernel SVR's own `.coef_.ravel()`.
    X_explain : (n_rows, n_features) array-like
        Rows to explain (the SAME rows a caller would otherwise hand to
        `fast_kernel_shap_effects()`/`shap.KernelExplainer`).
    baseline : (n_features,) array-like
        The single reference/background row every occluded column is
        reset to - the SAME weighted-background-mean convention
        `fast_kernel_shap_effects()`'s own `baseline` parameter uses.

    Returns
    -------
    1-D `np.ndarray`, length `n_features` - `sum over rows of |phi_ij|`,
    directly comparable to `fast_kernel_shap_effects()`'s own return
    value for the SAME model/inputs.

    Raises
    ------
    ValueError
        On a shape mismatch.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    X_explain = np.atleast_2d(np.asarray(X_explain, dtype=float))
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    if X_explain.shape[1] != w.shape[0] or baseline.shape[0] != w.shape[0]:
        raise ValueError(
            f"kernel_effects.linear_kernel_occlusion_effects: w has {w.shape[0]} markers but "
            f"X_explain has {X_explain.shape[1]} and baseline has {baseline.shape[0]} - all three "
            f"must describe the SAME marker set.")
    if w.shape[0] == 0:
        return np.zeros(0)
    if X_explain.shape[0] == 0:
        return np.zeros(w.shape[0])
    phi = (X_explain - baseline[np.newaxis, :]) * w[np.newaxis, :]
    return np.abs(phi).sum(axis=0)


def _dot_product_kernel_occlusion_effects(g, coeffs, X_train, X_explain, baseline, gamma, coef0, *,
                                           max_batch_elements=2e8):
    """Shared occlusion-effect computation for any SVR kernel of the form
    `K(x, x_i) = g(gamma * (x . x_i) + coef0)` - see the module-level
    comment above for the full derivation. `g` is applied elementwise via
    a plain NumPy ufunc/callable (`lambda z: np.power(z, degree)` /
    `np.tanh`), so `poly_kernel_occlusion_effects()`/`sigmoid_kernel_
    occlusion_effects()` below share this one loop rather than each
    re-implementing it.

    Parameters mirror `rbf_kernel_occlusion_effects()`'s own convention
    (`coeffs`/`X_train`/`X_explain`/`baseline` = `SVR.dual_coef_.ravel()`/
    `SVR.support_vectors_`/this task's own explained-row sample/the
    weighted background mean), plus this kernel family's own `gamma`/
    `coef0`. Cost: `O(n_rows * n_ref * n_features)` - linear in the
    marker count, instead of `fast_kernel_shap_effects()`'s own
    `O(n_rows * n_ref * n_features^2)` for this same quantity.
    """
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.atleast_2d(np.asarray(X_explain, dtype=float))
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    if X_train.ndim != 2:
        raise ValueError("kernel_effects._dot_product_kernel_occlusion_effects: X_train must be 2-D.")
    n_ref, n_features = X_train.shape
    if X_train.shape[0] != coeffs.shape[0]:
        raise ValueError(
            f"kernel_effects._dot_product_kernel_occlusion_effects: X_train has {n_ref} rows but "
            f"coeffs has {coeffs.shape[0]} values - these must describe the SAME reference rows.")
    if X_explain.shape[1] != n_features or baseline.shape[0] != n_features:
        raise ValueError(
            f"kernel_effects._dot_product_kernel_occlusion_effects: X_train has {n_features} "
            f"columns, X_explain has {X_explain.shape[1]}, and baseline has {baseline.shape[0]} "
            f"entries - all three must describe the SAME marker set.")
    gamma = float(gamma)
    coef0 = float(coef0)

    if n_features == 0:
        return np.zeros(0)
    if X_explain.shape[0] == 0:
        return np.zeros(n_features)

    feat_chunk = max(1, int(max_batch_elements // max(n_ref, 1)))
    phi_abs_sum = np.zeros(n_features)

    for x in X_explain:
        z = gamma * (X_train @ x) + coef0             # (n_ref,) - computed ONCE per row
        gz = g(z)
        for start in range(0, n_features, feat_chunk):
            end = min(start + feat_chunk, n_features)
            cols = slice(start, end)
            # Exact, closed-form correction to z (no squared term, unlike
            # the rbf case) for every marker in this chunk and every
            # reference row at once - shape (n_ref, chunk_width).
            delta = gamma * X_train[:, cols] * (x[cols] - baseline[cols])[np.newaxis, :]
            g_occ = g(z[:, np.newaxis] - delta)
            phi_chunk = coeffs @ (gz[:, np.newaxis] - g_occ)  # (chunk_width,)
            phi_abs_sum[cols] += np.abs(phi_chunk)

    return phi_abs_sum


def poly_kernel_occlusion_effects(coeffs, X_train, X_explain, baseline, gamma, coef0, degree, *,
                                   max_batch_elements=2e8):
    """Exact, closed-form per-marker occlusion effect for a fitted
    `sklearn.svm.SVR(kernel='poly')` (`K(x, x_i) = (gamma * x.x_i +
    coef0) ** degree`) - see the module-level comment above for the full
    derivation and cost. Mirrors `rbf_kernel_occlusion_effects()`'s own
    parameter convention/contract exactly, with `coef0`/`degree` added
    for this kernel's own two extra hyperparameters.

    Returns
    -------
    1-D `np.ndarray`, length `n_features` - `sum over rows of |phi_ij|`,
    bit-identical (to floating-point precision) to what
    `pipeline_utils.fast_kernel_shap_effects()` returns for the SAME
    model/inputs, at `O(n_rows * n_ref * n_features)` cost instead of
    that function's own `O(n_rows * n_ref * n_features^2)`.

    Raises
    ------
    ValueError
        On a shape mismatch or a non-positive `gamma`/`degree`.
    """
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"kernel_effects.poly_kernel_occlusion_effects: degree must be >= 1, got {degree!r}.")
    if not gamma or gamma <= 0:
        raise ValueError(f"kernel_effects.poly_kernel_occlusion_effects: gamma must be positive, got {gamma!r}.")
    return _dot_product_kernel_occlusion_effects(
        lambda z: np.power(z, degree), coeffs, X_train, X_explain, baseline, gamma, coef0,
        max_batch_elements=max_batch_elements,
    )


def sigmoid_kernel_occlusion_effects(coeffs, X_train, X_explain, baseline, gamma, coef0, *,
                                      max_batch_elements=2e8):
    """Exact, closed-form per-marker occlusion effect for a fitted
    `sklearn.svm.SVR(kernel='sigmoid')` (`K(x, x_i) = tanh(gamma * x.x_i +
    coef0)`) - see `poly_kernel_occlusion_effects()`'s own docstring (same
    contract/cost/convention, just `g = tanh` instead of `g(z) =
    z**degree`).

    Raises
    ------
    ValueError
        On a shape mismatch or a non-positive `gamma`.
    """
    if not gamma or gamma <= 0:
        raise ValueError(f"kernel_effects.sigmoid_kernel_occlusion_effects: gamma must be positive, got {gamma!r}.")
    return _dot_product_kernel_occlusion_effects(
        np.tanh, coeffs, X_train, X_explain, baseline, gamma, coef0,
        max_batch_elements=max_batch_elements,
    )


def zero_pair_interaction_table(feature_names, *, top_k=None):
    """The long-format `['marker1', 'marker2', 'value']` table every
    pairwise-interaction route in this codebase returns (see
    `rbf_kernel_hessian_interactions()`'s own docstring for the shared
    convention), with `value` set to exactly `0.0` for every one of the
    `M choose 2` combinations of `feature_names` - shared by every call
    site that already KNOWS, in closed form, that a pair's interaction is
    zero (an exactly linear model's Hessian, or its baseline-shap
    discrete second difference - both are mathematically zero for the
    SAME underlying reason, see `poly_kernel_hessian_interactions()`'s own
    `degree < 2` shortcut and `SVRBaselineShapFastEval`'s own module-level
    comment) and so has no need to run any per-row computation, or even
    construct a coefficient/reference-row array, to report it.

    Parameters
    ----------
    feature_names : sequence of str
        The marker shortlist - every unordered pair of DISTINCT entries is
        included, in the same `p < q` positional order every other
        interaction route in this codebase uses.
    top_k : int or None, keyword-only, default None
        See `rbf_kernel_hessian_interactions()`'s own identical parameter.

    Returns
    -------
    `pandas.DataFrame` with columns `['marker1', 'marker2', 'value']` -
    empty (zero-row, correctly columned) when `len(feature_names) < 2`.
    """
    import pandas as pd
    feature_names = list(feature_names)
    M = len(feature_names)
    if M < 2:
        return pd.DataFrame(columns=['marker1', 'marker2', 'value'])
    rows = [(feature_names[p], feature_names[q], 0.0) for p in range(M) for q in range(p + 1, M)]
    table = pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])
    if top_k is not None and table.shape[0] > int(top_k):
        table = table.head(int(top_k)).reset_index(drop=True)
    return table


def _dot_product_kernel_hessian_interactions(g2, coeffs, X_train, X_explain, gamma, coef0, feature_names, *,
                                              top_k=None):
    """Shared Hessian computation for any SVR kernel of the form `K(x,
    x_i) = g(gamma * (x . x_i) + coef0)` - see the module-level comment
    above for the full derivation (`g2` is `g''`, evaluated at each
    reference row's own `z_i(x)`). Mirrors `rbf_kernel_hessian_
    interactions()`'s own return convention exactly (a long-format
    `['marker1', 'marker2', 'value']` table, `value = mean(|d^2f/dx_p
    dx_q|)` over `X_explain`, sorted descending) - `poly_kernel_hessian_
    interactions()`/`sigmoid_kernel_hessian_interactions()` below share
    this one loop rather than each re-implementing it.
    """
    import pandas as pd
    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    X_train = np.asarray(X_train, dtype=float)
    X_explain = np.asarray(X_explain, dtype=float)
    feature_names = list(feature_names)
    if X_train.ndim != 2 or X_explain.ndim != 2:
        raise ValueError("kernel_effects._dot_product_kernel_hessian_interactions: X_train/X_explain must be 2-D.")
    if X_train.shape[0] != coeffs.shape[0]:
        raise ValueError(
            f"kernel_effects._dot_product_kernel_hessian_interactions: X_train has "
            f"{X_train.shape[0]} rows but coeffs has {coeffs.shape[0]} values - these must "
            f"describe the SAME training rows.")
    if X_train.shape[1] != X_explain.shape[1] or X_train.shape[1] != len(feature_names):
        raise ValueError(
            f"kernel_effects._dot_product_kernel_hessian_interactions: X_train has "
            f"{X_train.shape[1]} columns, X_explain has {X_explain.shape[1]}, and feature_names "
            f"has {len(feature_names)} entries - all three must describe the SAME marker "
            f"shortlist.")

    M = X_train.shape[1]
    empty = pd.DataFrame(columns=['marker1', 'marker2', 'value'])
    if M < 2 or X_explain.shape[0] == 0:
        return empty

    gamma = float(gamma)
    coef0 = float(coef0)
    accum = np.zeros((M, M), dtype=float)
    for x in X_explain:
        z = gamma * (X_train @ x) + coef0            # (n_train,)
        w = coeffs * g2(z)                            # (n_train,)
        H = (gamma * gamma) * (X_train.T @ (X_train * w[:, np.newaxis]))  # (M, M)
        accum += np.abs(H)
    mean_abs_H = accum / X_explain.shape[0]
    np.fill_diagonal(mean_abs_H, 0.0)

    rows = [
        (feature_names[p], feature_names[q], mean_abs_H[p, q])
        for p in range(M) for q in range(p + 1, M)
    ]
    table = pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])
    table = table.sort_values('value', ascending=False).reset_index(drop=True)
    if top_k is not None and table.shape[0] > int(top_k):
        table = table.head(int(top_k)).reset_index(drop=True)
    return table


def poly_kernel_hessian_interactions(coeffs, X_train, X_explain, gamma, coef0, degree, feature_names, *,
                                      top_k=None):
    """Closed-form, DETERMINISTIC marker-pair interaction strengths for a
    fitted `sklearn.svm.SVR(kernel='poly')` (`K(x, x_i) = (gamma * x.x_i +
    coef0) ** degree`) - see the module-level comment above for the full
    derivation. Mirrors `rbf_kernel_hessian_interactions()`'s own
    parameter convention/return contract exactly, with `coef0`/`degree`
    added for this kernel's own two extra hyperparameters. Also covers
    `kernel='linear'` (pass `gamma=1.0, coef0=0.0, degree=1`): a
    `degree < 2` kernel is an EXACTLY linear function of `x`, so its
    Hessian is identically zero for every pair - this is detected up
    front and answered with NO per-row computation at all (not even the
    `O(n_train * M^2)`-per-row loop below is run), the true minimum
    possible cost, rather than merely a fast approximation of "no
    interaction".

    Returns
    -------
    `pandas.DataFrame` with columns `['marker1', 'marker2', 'value']` -
    see `rbf_kernel_hessian_interactions()`'s own docstring for the
    shared convention. `value` is uniformly `0.0` for every pair when
    `degree < 2`.

    Raises
    ------
    ValueError
        On a shape mismatch, a non-positive `gamma` (only checked when
        `degree >= 2` - a `degree < 2` kernel's Hessian is zero
        regardless of `gamma`), or `degree < 1`.
    """
    import pandas as pd
    degree = int(degree)
    if degree < 1:
        raise ValueError(f"kernel_effects.poly_kernel_hessian_interactions: degree must be >= 1, got {degree!r}.")

    if degree < 2:
        X_train_arr = np.asarray(X_train, dtype=float)
        X_explain_arr = np.asarray(X_explain, dtype=float)
        feature_names = list(feature_names)
        if X_train_arr.ndim != 2 or X_explain_arr.ndim != 2:
            raise ValueError(
                "kernel_effects.poly_kernel_hessian_interactions: X_train/X_explain must be 2-D.")
        if X_train_arr.shape[1] != X_explain_arr.shape[1] or X_train_arr.shape[1] != len(feature_names):
            raise ValueError(
                f"kernel_effects.poly_kernel_hessian_interactions: X_train has "
                f"{X_train_arr.shape[1]} columns, X_explain has {X_explain_arr.shape[1]}, and "
                f"feature_names has {len(feature_names)} entries - all three must describe the "
                f"SAME marker shortlist.")
        if X_explain_arr.shape[0] == 0:
            return pd.DataFrame(columns=['marker1', 'marker2', 'value'])
        return zero_pair_interaction_table(feature_names, top_k=top_k)

    if not gamma or gamma <= 0:
        raise ValueError(f"kernel_effects.poly_kernel_hessian_interactions: gamma must be positive, got {gamma!r}.")
    return _dot_product_kernel_hessian_interactions(
        lambda z: (degree * (degree - 1)) * np.power(z, degree - 2),
        coeffs, X_train, X_explain, gamma, coef0, feature_names, top_k=top_k,
    )


def sigmoid_kernel_hessian_interactions(coeffs, X_train, X_explain, gamma, coef0, feature_names, *, top_k=None):
    """Closed-form, DETERMINISTIC marker-pair interaction strengths for a
    fitted `sklearn.svm.SVR(kernel='sigmoid')` (`K(x, x_i) = tanh(gamma *
    x.x_i + coef0)`) - see `poly_kernel_hessian_interactions()`'s own
    docstring (same contract/cost/convention, just `g''(z) = -2*tanh(z)*
    (1-tanh(z)**2)` instead of `degree*(degree-1)*z**(degree-2)`).

    Raises
    ------
    ValueError
        On a shape mismatch or a non-positive `gamma`.
    """
    if not gamma or gamma <= 0:
        raise ValueError(f"kernel_effects.sigmoid_kernel_hessian_interactions: gamma must be positive, got {gamma!r}.")

    def _g2(z):
        t = np.tanh(z)
        return -2.0 * t * (1.0 - t * t)

    return _dot_product_kernel_hessian_interactions(
        _g2, coeffs, X_train, X_explain, gamma, coef0, feature_names, top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Perf fix (this update - "the computational minimisation should be
# integrated into kernel_shap for marker effects and baseline_shap for
# marker interactions"): everything above this point speeds up SVR's
# 'kernel_shap' marker-effect route (`_dot_product_kernel_occlusion_effects()`
# and friends) and its 'kernel_hessian' marker-INTERACTION route
# (`_dot_product_kernel_hessian_interactions()` and friends) - but SVR's
# other interaction route, 'baseline_shap' (`models.interaction_extraction.
# baseline_shap_interactions()` - the RECOMMENDED route per hparam_specs.py,
# since it never assumes marker independence the way `kernel_hessian`'s own
# discrete-difference cousin 'friedman_h' does), stayed a generic, model-
# agnostic loop that calls the model's own `.predict()` on a literally-
# materialised hybrid row, at `O(n_ref * M)` cost PER hybrid row, for every
# kernel INCLUDING 'rbf' - it had never had a closed-form fast path at all,
# unlike 'kernel_hessian'.
#
# `SVRBaselineShapFastEval` below closes that gap WITHOUT changing a single
# line of `baseline_shap_interactions()`'s own algorithm (the discrete
# second difference over REAL background rows, the single-marker-knockout
# deduplication, `screen`, chunking, `n_jobs` fan-out - see that function's
# own docstring) - it only replaces HOW a hybrid row's own prediction is
# computed, via that function's new `kernel_fast_eval=` hook (see its own
# docstring, `models/interaction_extraction.py`).
#
# The key fact (same family as the `kernel_shap` fix above, extended to
# 'rbf' too): `baseline_shap_interactions()` only ever evaluates the model
# at a HYBRID row - every column equal to the explained row `x`'s own value,
# except a small set `S` (one marker for the single-knockout pass, two for
# the joint pass) replaced by a SPECIFIC background row `b`'s own value at
# those columns. For the affine-kernel family ('linear'/'poly'/'sigmoid'),
# `z_i(x^{S->b}) = z_i(x) + gamma * sum_{p in S} sv_i,p * (b_p - x_p)` - the
# SAME correction `_dot_product_kernel_occlusion_effects()` above uses, just
# with `b` now one of `B` distinct real background rows instead of a single
# averaged baseline. For 'rbf', squared distance is separable the SAME way
# `rbf_kernel_occlusion_effects()` above already exploits:
# `D(x^{S->b}, sv_i) = D(x, sv_i) + sum_{p in S} [(b_p-sv_i,p)^2 -
# (x_p-sv_i,p)^2]`. Either way, the hybrid row's own kernel value is a
# closed-form, `O(n_ref)` correction to the explained row's OWN
# already-computed kernel value - no need to ever materialise the
# full-width hybrid row or call `.predict()` on it at all. `linear` is a
# further special case: because `g` itself is linear there, the FOUR-TERM
# discrete second difference `f(x) - f(x^i) - f(x^j) + f(x^ij)`
# `baseline_shap_interactions()` computes cancels to EXACTLY zero for every
# pair, by the same algebra `poly_kernel_hessian_interactions()`'s own
# `degree < 2` shortcut already exploits - `models/SVR.py`'s own
# 'baseline_shap' branch detects this up front and reports
# `zero_pair_interaction_table()` directly, without constructing an
# instance of this class at all (the true minimum possible cost, not merely
# a fast evaluation of "no interaction").
# ---------------------------------------------------------------------------

class SVRBaselineShapFastEval:
    """Closed-form, drop-in accelerator for `models.interaction_extraction.
    baseline_shap_interactions()`'s own hybrid-row evaluation, for a fitted
    `sklearn.svm.SVR` of kernel `'rbf'`, `'poly'`, or `'sigmoid'` (see the
    module-level comment directly above for the full derivation and the
    separate, even-cheaper `'linear'` treatment - this class does not
    support `kernel='linear'`, since that case is answered analytically,
    with no per-row evaluation of any kind, by `zero_pair_interaction_
    table()` instead).

    Passed as `baseline_shap_interactions()`'s own `kernel_fast_eval=`
    argument - that function calls `begin_row()` once per explained row,
    then `single_knockout()`/`joint_knockout()` once per chunk of markers/
    pairs, exactly where it would otherwise materialise a hybrid row and
    call `predict_fn` on it. Every method below returns EXACTLY what
    `predict_fn(X_off)` would have returned for the SAME hybrid rows
    (bit-identical to floating-point precision) - this class changes only
    HOW that value is computed (`O(n_ref)` per hybrid row instead of
    `O(n_ref * M)`), never WHAT is computed.

    Parameters
    ----------
    kernel : {'rbf', 'poly', 'sigmoid'}
        Which of `sklearn.svm.SVR`'s own kernel formulas this instance
        evaluates.
    coeffs : (n_ref,) array-like
        `SVR.dual_coef_.ravel()`.
    X_ref : (n_ref, M) array-like
        `SVR.support_vectors_` - restricted to the SAME `M`-marker
        shortlist `X_background`/`X_explain` (in `baseline_shap_
        interactions()`'s own call) use.
    X_background : (n_background, M) array-like
        The SAME real background rows passed to `baseline_shap_
        interactions()` itself - `bg_rows` (below) are positional indices
        into this array's own rows.
    intercept : float, keyword-only, default 0.0
        `SVR.intercept_` (a length-1 array; pass its scalar value). Every
        one of `single_knockout()`/`joint_knockout()`'s own return values
        includes this constant, so it stays consistent with `f_full`
        (computed by the real `predict_fn`, which also includes it) -
        `baseline_shap_interactions()`'s own four-term difference always
        sums this constant's coefficient to exactly 0, so an inconsistent
        (e.g. omitted) intercept here would silently bias every reported
        value by a fixed amount; matching it exactly is what keeps this
        an EXACT accelerator rather than an approximation.
    gamma, coef0 : float, keyword-only, defaults 1.0/0.0
        This kernel's own resolved bandwidth/offset (ignored - but still
        must be positive for `gamma` - for `kernel='rbf'`'s own
        `coef0`-free formula; sklearn's 'rbf' kernel never uses `coef0`).
    degree : int, keyword-only, default 1
        This kernel's own polynomial degree - only meaningful for
        `kernel='poly'` (`degree >= 2` - `poly_kernel_hessian_
        interactions()`'s own `degree < 2` shortcut, and this class's own
        `'linear'` exclusion above, both mean this class is never
        constructed for an exactly-linear dot-product kernel).

    Raises
    ------
    ValueError
        On an unrecognised `kernel`, or a non-positive `gamma`.
    """

    __slots__ = ('kernel', 'coeffs', 'X_ref', 'X_background', 'intercept', 'gamma', 'coef0', 'degree', '_g')

    def __init__(self, kernel, coeffs, X_ref, X_background, *, intercept=0.0, gamma=1.0, coef0=0.0, degree=1):
        if kernel not in ('rbf', 'poly', 'sigmoid'):
            raise ValueError(
                f"kernel_effects.SVRBaselineShapFastEval: kernel must be one of "
                f"('rbf', 'poly', 'sigmoid'), got {kernel!r} ('linear' is handled by "
                f"zero_pair_interaction_table() instead - see this class's own docstring).")
        if not gamma or gamma <= 0:
            raise ValueError(f"kernel_effects.SVRBaselineShapFastEval: gamma must be positive, got {gamma!r}.")
        self.kernel = kernel
        self.coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        self.X_ref = np.asarray(X_ref, dtype=float)
        self.X_background = np.asarray(X_background, dtype=float)
        self.intercept = float(intercept)
        self.gamma = float(gamma)
        self.coef0 = float(coef0)
        self.degree = int(degree)
        if kernel == 'poly':
            _degree = self.degree
            self._g = lambda z: np.power(z, _degree)
        elif kernel == 'sigmoid':
            self._g = np.tanh
        else:
            self._g = None  # 'rbf' - distance-based, handled separately below

    def working_bytes_per_eval(self):
        """Approximate PEAK working memory (bytes) ONE hybrid-row evaluation
        costs in `single_knockout()`/`joint_knockout()`: those build about a
        dozen `(n_ref, n_evals)` float64 temporaries at once (the two
        gathered reference-column blocks, the four squared-difference
        terms, their running sum, the exponentiated kernel, ...). Used by
        `baseline_shap_interactions()` to size its per-worker chunk from
        available memory; never affects any computed value."""
        return 12 * 8 * int(self.X_ref.shape[0])

    def begin_row(self, x):
        """Precompute this explained row's own `z_i(x)` (affine kernels)
        or `D(x, sv_i)` (`'rbf'`) against every reference row ONCE - the
        opaque `row_cache` `single_knockout()`/`joint_knockout()` reuse
        across every chunk of both passes for THIS row, so it is never
        recomputed per chunk/marker/pair."""
        x = np.asarray(x, dtype=float)
        if self.kernel == 'rbf':
            return squared_euclidean_distance(x.reshape(1, -1), self.X_ref).ravel()
        return self.gamma * (self.X_ref @ x) + self.coef0

    def full(self, x, row_cache):
        """ver4-43 (R1): returns `f(x)` itself, from the SAME `row_cache`
        `single_knockout()`/`joint_knockout()` start from, so that the
        un-knocked-out term and the knocked-out terms of a difference
        `f(x) - f(x^{q->a}) - ...` all come from ONE evaluation engine and
        any rounding they share cancels (a `predict_fn(X_explain)` call
        would round differently, e.g. through sklearn's own libsvm kernel
        or `RKHSMaskedPredictor`'s clipped distance).

        `x` is unused (kept for symmetry with the other methods); the value
        is `coeffs @ K(x, ref) + intercept` with `K` built from `row_cache`.
        """
        if self.kernel == 'rbf':
            K_full = np.exp(-self.gamma * np.clip(row_cache, 0.0, None))
        else:
            K_full = self._g(row_cache)
        return float(self.coeffs @ K_full + self.intercept)

    def single_knockout(self, x, row_cache, cols, bg_rows):
        """Returns `f(x^{cols[t]->X_background[bg_rows[t], cols[t]]})`
        for every `t` - a drop-in replacement for `baseline_shap_
        interactions()`'s own `predict_fn(X_off)` in its single-marker
        knockout pass. `cols`/`bg_rows`: 1-D int arrays of equal length,
        exactly as that function's own `_row_deltas()` already builds
        them (positional marker index / background-row index per
        synthetic row)."""
        x = np.asarray(x, dtype=float)
        bg_vals = self.X_background[bg_rows, cols]
        x_vals = x[cols]
        if self.kernel == 'rbf':
            X_ref_cols = self.X_ref[:, cols]
            term_x = (x_vals[np.newaxis, :] - X_ref_cols) ** 2
            term_b = (bg_vals[np.newaxis, :] - X_ref_cols) ** 2
            D_occ = row_cache[:, np.newaxis] + (term_b - term_x)
            np.clip(D_occ, 0.0, None, out=D_occ)
            K_occ = np.exp(-self.gamma * D_occ)
        else:
            diff = bg_vals - x_vals
            delta = self.gamma * self.X_ref[:, cols] * diff[np.newaxis, :]
            K_occ = self._g(row_cache[:, np.newaxis] + delta)
        return self.coeffs @ K_occ + self.intercept

    def joint_knockout(self, x, row_cache, i_arr, j_arr, bg_rows):
        """Returns `f(x^{{i_arr[t],j_arr[t]}->X_background[bg_rows[t], .]})`
        for every `t` (both columns replaced from the SAME background row)
        - a drop-in replacement for `baseline_shap_interactions()`'s own
        `predict_fn(X_ij_off)` in its joint ("both markers off") knockout
        pass. `i_arr`/`j_arr`/`bg_rows`: 1-D int arrays of equal length,
        exactly as that function's own `_row_deltas()` already builds
        them."""
        x = np.asarray(x, dtype=float)
        bg_i = self.X_background[bg_rows, i_arr]
        bg_j = self.X_background[bg_rows, j_arr]
        x_i = x[i_arr]
        x_j = x[j_arr]
        if self.kernel == 'rbf':
            X_ref_i = self.X_ref[:, i_arr]
            X_ref_j = self.X_ref[:, j_arr]
            term_x_i = (x_i[np.newaxis, :] - X_ref_i) ** 2
            term_b_i = (bg_i[np.newaxis, :] - X_ref_i) ** 2
            term_x_j = (x_j[np.newaxis, :] - X_ref_j) ** 2
            term_b_j = (bg_j[np.newaxis, :] - X_ref_j) ** 2
            D_occ = row_cache[:, np.newaxis] + (term_b_i - term_x_i) + (term_b_j - term_x_j)
            np.clip(D_occ, 0.0, None, out=D_occ)
            K_occ = np.exp(-self.gamma * D_occ)
        else:
            diff_i = bg_i - x_i
            diff_j = bg_j - x_j
            delta = self.gamma * (
                self.X_ref[:, i_arr] * diff_i[np.newaxis, :] + self.X_ref[:, j_arr] * diff_j[np.newaxis, :]
            )
            K_occ = self._g(row_cache[:, np.newaxis] + delta)
        return self.coeffs @ K_occ + self.intercept


def rkhs_baseline_shap_fast_eval(predictor, baseline, shortlist_idx, X_background):
    """ver4-43 (R1/R2): the closed-form `SVRBaselineShapFastEval('rbf', ...)`
    for a masked RKHS reconstruction - the SAME accelerator the SVR route
    already uses, fed with the pure-RBF re-expression of
    `RKHSMaskedPredictor` that `as_rbf_occlusion_kernel()` already provides.
    No new kernel maths lives here: `f(x_short) = mu + sum_j coeffs_eff_j *
    exp(-gamma * ||x_short - ref_short_j||^2)` (see that method's docstring
    for why folding the fixed non-shortlist distance into the coefficients
    is exact).

    Why this exists: `baseline_shap_interactions()` called with only
    `RKHSMaskedPredictor.predict` evaluates every hybrid row through an
    `O(n_grid * M')` GEMM; the closed form needs `O(n_grid)` per hybrid.

    Parameters
    ----------
    predictor : RKHSPredictor or RKHSPredictCore
        The V13.2-gated reconstruction (anything with `X_ref`, `alpha_ref`,
        `mu`, `h`, `d_scale`).
    baseline : (n_features_full,) array-like
        Fixed value of every non-shortlist column (training-column mean).
    shortlist_idx : (M',) array-like of int
        Positional shortlist columns, in the order `X_background` uses.
    X_background : (B, M') array-like
        Real background rows restricted to the shortlist; `bg_rows` given
        to the evaluator index into this array.

    Returns
    -------
    SVRBaselineShapFastEval (kernel='rbf', intercept=predictor.mu).

    Raises
    ------
    ValueError
        From `RKHSMaskedPredictor` (shape mismatch) or
        `SVRBaselineShapFastEval` (non-positive gamma); the callers catch
        it, print a NOTE and keep the generic `predict_fn` path.
    """
    masked = RKHSMaskedPredictor(predictor, baseline, shortlist_idx)
    coeffs_eff, X_ref_short, gamma = masked.as_rbf_occlusion_kernel()
    X_background = np.asarray(X_background, dtype=float)
    if X_background.ndim != 2 or X_background.shape[1] != X_ref_short.shape[1]:
        raise ValueError(
            f"kernel_effects.rkhs_baseline_shap_fast_eval: X_background has shape "
            f"{X_background.shape} but the shortlist has {X_ref_short.shape[1]} column(s).")
    return SVRBaselineShapFastEval(
        'rbf', coeffs_eff, X_ref_short, X_background,
        intercept=float(predictor.mu), gamma=gamma,
    )


def linear_baseline_shap_effects(w, X_explain, X_background, *, reduce='abs_then_sum'):
    """ver4-43 (R2): the `'baseline_shap'` single-marker effect of an exactly
    linear model, in closed form. For `f(x) = w.x + b` the knock-out
    difference is `f(x) - f(x^{q->z}) = w_q * (x_q - z_q)`, so the
    background mean is `w_q * (x_q - mean_z z_q)` - no model evaluation.

    Parameters
    ----------
    w : (M,) array-like - the linear coefficients (`SVR.coef_.ravel()`).
    X_explain : (n_explain, M); X_background : (B, M).
    reduce : {'abs_then_sum', 'sum_then_abs'} - the same convention every
        effect route uses across explained rows.

    Returns
    -------
    (M,) float ndarray, never NaN (non-finite entries become 0.0).

    Raises
    ------
    ValueError on a width mismatch or an unknown `reduce`.
    """
    if reduce not in ('abs_then_sum', 'sum_then_abs'):
        raise ValueError(
            f"kernel_effects.linear_baseline_shap_effects: reduce must be 'abs_then_sum' or "
            f"'sum_then_abs', got {reduce!r}.")
    w = np.asarray(w, dtype=float).reshape(-1)
    X_explain = np.asarray(X_explain, dtype=float)
    X_background = np.asarray(X_background, dtype=float)
    if X_explain.ndim != 2 or X_background.ndim != 2 \
            or X_explain.shape[1] != w.shape[0] or X_background.shape[1] != w.shape[0]:
        raise ValueError(
            f"kernel_effects.linear_baseline_shap_effects: w has {w.shape[0]} entries but "
            f"X_explain is {X_explain.shape} and X_background is {X_background.shape}.")
    if X_explain.shape[0] == 0 or X_background.shape[0] == 0:
        return np.zeros(w.shape[0], dtype=float)
    per_row = w[np.newaxis, :] * (X_explain - X_background.mean(axis=0)[np.newaxis, :])
    scores = np.abs(per_row).sum(axis=0) if reduce == 'abs_then_sum' else np.abs(per_row.sum(axis=0))
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


# ===========================================================================
# ver4-36, R1.d - exact closed-form SEARCH engine for rrBLUP (BRR) and RKHS
# ===========================================================================
#
# Why this exists (design blueprint EasiGP_ver4-36, R1.d / F3): during the joint
# hyperparameter search, every rrBLUP and RKHS candidate is a full BGLR MCMC fit
# (12 000 iterations; RKHS also builds an n_total x n_total kernel), although the
# search only reads the VALIDATION predictions. Both models are Gaussian-process
# regressions,
#       y_o = mu + u_o + e,   u ~ N(0, sigma_u^2 K),   e ~ N(0, sigma_e^2 I),
# so given (sigma_u^2, sigma_e^2) the posterior mean of the masked rows is a closed
# form. This engine finds the variance components by maximising their (REML-like)
# posterior - the marginal likelihood times BGLR's own default scaled-inverse-chi-
# squared priors - and then predicts exactly, in milliseconds and with NO Monte-Carlo
# noise (which also removes the common-random-numbers caveat of F4).
#
# What it is NOT: it is never used for a confirmatory fit (those stay real BGLR fits
# through `ModelTrialRunner`, so the result files are unchanged), and it is a plug-in
# posterior-MODE approximation of BGLR's posterior MEAN. `joint_tuning` therefore
# checks, once per task and model, that the exact seed column correlates with the
# BGLR seed column at >= `engine_parity_min` and switches the engine off otherwise.
#
# BGLR default priors used (recorded from the BGLR source - `BGLR()`, `setLT.BRR`,
# `setLT.RKHS` - from memory of the package source, NOT re-verified against the R
# installation, see the ver4-36 Change Summary, P0.5). Every variance v has the
# prior  p(v) ~ v^-(df0/2 + 1) * exp(-S0 / (2 v))  (S0 is a SUM-OF-SQUARES-type
# scale: the Gibbs update is v = (SS + S0) / chi2(df0 + count)):
#   * residual:  df0 = 5, S0 = var(y_observed) * (1 - R2_global) * (df0 + 2), with the
#     global BGLR argument R2 = 0.5;
#   * the effect term (BRR 'mrk' with the model's own df0 / R2; RKHS with df0 = 5 and
#     R2 = 0.5 / number_of_terms = 0.5):
#     S0 = var(y_observed) * R2 / MSx * (df0 + 2), MSx = sum(colSums(X^2)) / n =
#     trace(K) / n_total (= mean(diag(K)) for both kernels used here).

_EXACT_LOG_VAR_BOUNDS = (-14.0, 7.0)   # log variance components, in units of var(y)
_EXACT_MIN_LAMBDA = 1.0e-8             # floor of sigma_e^2 / sigma_u^2
_EXACT_RESIDUAL_DF0 = 5.0
_EXACT_GLOBAL_R2 = 0.5
_EXACT_MEMO_MAX = 24                   # most memo entries kept per process

# Per-process memo: key -> value. Two kinds of key share the dict:
#   ('K', fingerprint, model)            -> the kernel input (rrBLUP: K; RKHS: D / mean(D))
#   ('eig', fingerprint, model, h-or-0)  -> (K_oo eigenvalues, eigenvectors, K_ao)
_EXACT_MEMO: dict = {}


def clear_exact_kernel_memo() -> None:
    """Empty the per-process kernel / eigendecomposition memo of the exact engine
    (used by tests and by anything that wants to release the memory)."""
    _EXACT_MEMO.clear()


def _memo_put(key, value) -> None:
    """Insert into the memo, dropping the oldest entries beyond `_EXACT_MEMO_MAX`
    (dicts keep insertion order, so this is first-in-first-out)."""
    _EXACT_MEMO[key] = value
    while len(_EXACT_MEMO) > _EXACT_MEMO_MAX:
        _EXACT_MEMO.pop(next(iter(_EXACT_MEMO)))


def _frames_fingerprint(train, valid, test) -> tuple:
    """Cheap identity of the (train, valid, test) frames for memo keys.

    Why not the data itself: hashing 250 rows x 50 000 markers on every fit would
    cost more than the fit. The fingerprint is the three shapes, the hash of each
    index, the first / middle / last rows of every frame and the training phenotype.
    Within one task the search hands the SAME frames to every candidate (and the CV
    folds use frames whose indices differ per fold), so equal fingerprints mean equal
    frames in practice; a different frame with identical shapes, indices, boundary
    rows and phenotype would be a bug elsewhere."""
    parts = []
    for frame in (train, valid, test):
        values = frame.to_numpy(dtype=float)
        n_rows = values.shape[0]
        if n_rows:
            picks = values[[0, n_rows // 2, n_rows - 1]]
            row_hash = hash(picks.tobytes())
        else:
            row_hash = 0
        parts.append((values.shape, hash(np.asarray(frame.index).tobytes()), row_hash))
    parts.append(hash(np.asarray(train.iloc[:, -1], dtype=float).tobytes()))
    return tuple(parts)


def _r_scale_columns(X: np.ndarray) -> np.ndarray:
    """R's ``scale(X)`` (column mean, sample sd with ddof = 1, NA ignored), followed by
    the model file's own ``X[is.na(X)] <- -1`` - exactly what ``rrBLUP.R`` does, so a
    constant marker (sd = 0, 0/0 = NaN in R) becomes a column of -1, not 0."""
    X = np.asarray(X, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1)
        Z = (X - mean) / sd
    Z[~np.isfinite(Z)] = -1.0
    return Z


def _kernel_matrix_for(model: str, frames, params) -> np.ndarray:
    """The full kernel over ``rbind(train, valid, test)`` that BGLR sees for ``model``.

    ``frames`` is ``(train, valid, test)`` (marker columns first, phenotype last).
    * ``'rrBLUP'``: ``K = X X'`` with ``X = scale(markers) / sqrt(p)`` and NA -> -1
      (``rrBLUP.R``); it does not depend on ``params``.
    * ``'RKHS'``: ``K = exp(-h D / mean(D))`` with ``D`` the squared Euclidean distance of
      the RAW markers over all rows (``RKHS.R``) and ``h = params[2]``.
    Raises ``ValueError`` for another model, a non-finite kernel or a non-positive
    ``h``. The result is the plain NumPy array, (n_total, n_total)."""
    train, valid, test = frames
    data = np.vstack([f.to_numpy(dtype=float)[:, :-1] for f in (train, valid, test)])
    if model == 'rrBLUP':
        Z = _r_scale_columns(data) / np.sqrt(data.shape[1])
        K = Z @ Z.T
    elif model == 'RKHS':
        h = float(params[2])
        if not np.isfinite(h) or h <= 0.0:
            raise ValueError(f"exact_kernel: RKHS bandwidth h={params[2]!r} must be positive.")
        K = np.exp(-h * _rkhs_normalised_distance(data))
    else:
        raise ValueError(f"exact_kernel: no exact engine for model {model!r}.")
    if not np.all(np.isfinite(K)):
        raise ValueError(f"exact_kernel: the {model} kernel has non-finite entries "
                         f"(missing marker values?).")
    return K


def _rkhs_normalised_distance(data: np.ndarray) -> np.ndarray:
    """``D / mean(D)`` of the squared Euclidean distances of ``data`` (RKHS.R). The
    mean is over ALL n^2 entries including the zero diagonal, as in R. A degenerate
    all-identical input (mean 0) raises ``ValueError``."""
    D = squared_euclidean_distance(data)
    scale = float(D.mean())
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("exact_kernel: all RKHS rows are identical (mean squared distance 0).")
    return D / scale


def _bglr_default_priors(model: str, y_o: np.ndarray, K: np.ndarray, params) -> dict:
    """BGLR's default scaled-inverse-chi-squared priors for the two variances, in units of
    ``var(y_o)`` (the engine standardises the phenotype, so ``var(y_o) = 1`` there).

    Returns ``{'u': {'S': ..., 'df': ...}, 'e': {'S': ..., 'df': ...}}`` - see the module
    comment above for the formulas. ``model='rrBLUP'`` uses the model's own
    ``df0 = params[2]`` and ``R2 = params[3]``; ``'RKHS'`` uses BGLR's defaults
    (``df0 = 5``, ``R2 = 0.5``). ``K`` is the full kernel (only ``mean(diag(K))`` is
    read). Raises ``ValueError`` for an unknown model or out-of-range df0 / R2."""
    if model == 'rrBLUP':
        df_u = float(params[2])
        r2_u = float(params[3])
    elif model == 'RKHS':
        df_u = 5.0
        r2_u = _EXACT_GLOBAL_R2
    else:
        raise ValueError(f"exact_kernel: no default priors for model {model!r}.")
    if not (np.isfinite(df_u) and df_u > 0.0 and np.isfinite(r2_u) and 0.0 < r2_u < 1.0):
        raise ValueError(f"exact_kernel: invalid prior settings df0={df_u!r}, R2={r2_u!r}.")
    msx = float(np.mean(np.diag(K)))
    if not np.isfinite(msx) or msx <= 0.0:
        raise ValueError("exact_kernel: the kernel has a non-positive mean diagonal.")
    return {
        'u': {'S': r2_u / msx * (df_u + 2.0), 'df': df_u},
        'e': {'S': (1.0 - _EXACT_GLOBAL_R2) * (_EXACT_RESIDUAL_DF0 + 2.0),
              'df': _EXACT_RESIDUAL_DF0},
    }


def _inv_chi2_nlp(v: float, S: float, df: float) -> float:
    """Negative log of the prior ``v^-(df/2 + 1) exp(-S / (2 v))`` (constant dropped)."""
    return (df / 2.0 + 1.0) * np.log(v) + S / (2.0 * v)


def _exact_kernel_predict(K_oo: np.ndarray, K_ao: np.ndarray, y_o: np.ndarray, prior: dict,
                          eig=None) -> np.ndarray:
    """Posterior-mean prediction of the masked rows ``a`` from the observed rows ``o``.

    ``y_o`` is standardised internally; ``prior`` comes from ``_bglr_default_priors``.
    Variance components maximise ``log marginal likelihood (intercept integrated out
    with a flat prior, i.e. REML) + log priors`` over ``(log sigma_u^2, log sigma_e^2)``
    with bounded L-BFGS-B in the eigenbasis of ``K_oo`` (each evaluation is O(n_o)).
    ``eig`` optionally supplies ``(d, U)`` = ``numpy.linalg.eigh(K_oo)`` (the memo).

    Prediction: ``yhat_a = mu + K_ao U diag(1 / (d + lambda)) U'(y_o - mu)`` with
    ``lambda = sigma_e^2 / sigma_u^2``. Returns the (n_a,) array on the original scale.
    Raises ``ValueError`` for a non-finite input or output and ``np.linalg.LinAlgError``
    if the eigendecomposition fails."""
    from scipy.optimize import minimize

    y_o = np.asarray(y_o, dtype=float)
    center = float(y_o.mean())
    spread = float(y_o.std(ddof=1)) if y_o.size > 1 else 0.0
    if not np.isfinite(spread) or spread <= 0.0:
        raise ValueError("exact_kernel: the training phenotype has no variance.")
    y_s = (y_o - center) / spread
    if eig is None:
        d, U = np.linalg.eigh(K_oo)
    else:
        d, U = eig
    d = np.clip(d, 0.0, None)
    ones_t = U.T @ np.ones_like(y_s)
    z = U.T @ y_s
    s_u, df_u = prior['u']['S'], prior['u']['df']
    s_e, df_e = prior['e']['S'], prior['e']['df']

    def _intercept(v):
        inv = 1.0 / v
        return float(np.sum(ones_t * z * inv) / np.sum(ones_t * ones_t * inv))

    def _neg_log_posterior(t):
        v_u, v_e = np.exp(t[0]), np.exp(t[1])
        v = v_u * d + v_e
        inv = 1.0 / v
        qq = float(np.sum(ones_t * ones_t * inv))
        mu = float(np.sum(ones_t * z * inv)) / qq
        resid = z - mu * ones_t
        value = 0.5 * (np.sum(np.log(v)) + np.log(qq) + np.sum(resid * resid * inv))
        value += _inv_chi2_nlp(v_u, s_u, df_u) + _inv_chi2_nlp(v_e, s_e, df_e)
        return float(value)

    x0 = np.log([0.5, 0.5])
    fit = minimize(_neg_log_posterior, x0, method='L-BFGS-B',
                   bounds=[_EXACT_LOG_VAR_BOUNDS, _EXACT_LOG_VAR_BOUNDS])
    if not np.all(np.isfinite(fit.x)) or not np.isfinite(fit.fun):
        raise ValueError("exact_kernel: the variance-component optimisation returned a "
                         "non-finite result.")
    v_u, v_e = np.exp(fit.x[0]), np.exp(fit.x[1])
    lam = max(v_e / v_u, _EXACT_MIN_LAMBDA)
    mu = _intercept(v_u * d + v_e)
    alpha = U @ ((U.T @ (y_s - mu)) / (d + lam))
    y_hat = center + spread * (mu + K_ao @ alpha)
    if not np.all(np.isfinite(y_hat)):
        raise ValueError("exact_kernel: non-finite predictions.")
    return y_hat


def exact_kernel_column(model: str, train, valid, test, params, *,
                        want_test: bool = False) -> dict:
    """Exact (closed-form) stand-in for a BGLR rrBLUP / RKHS SEARCH fit.

    Returns ``{'predicted_valid': array}`` (plus ``'predicted_test'`` when
    ``want_test``) - the same keys, in the same row order, as the runner result the
    joint search reads, so ``joint_tuning._fit_unit_worker`` can substitute it. The
    observed rows are ``train``; ``valid`` and ``test`` are masked exactly as in
    ``rrBLUP.R`` / ``RKHS.R`` (``rbind(train, valid, test)``, so the kernel is built
    over all three frames - the same kernel BGLR sees; no test PHENOTYPE is read,
    only the markers, as in the R code). I7: only the training phenotype is used.

    Speed: the kernel (rrBLUP) or the normalised distance matrix (RKHS) is built once per
    (frames, model) and the eigendecomposition once per (frames, model, h) - both in a
    small per-process memo - so a later rrBLUP candidate costs only the 2-parameter
    optimisation and one matrix-vector product (milliseconds).

    Args: ``model`` 'rrBLUP' or 'RKHS'; ``train`` / ``valid`` / ``test`` DataFrames
    (markers then phenotype; ``valid`` may be empty); ``params`` the model's
    HPARAMETERS-shaped list (rrBLUP: df0 = params[2], R2 = params[3]; RKHS:
    h = params[2]); ``want_test`` also return the test predictions.
    Raises ``ValueError`` (unsupported model, non-finite data, invalid hyperparameters,
    fewer than 5 training rows) or ``np.linalg.LinAlgError``; the caller falls back to the
    real fit on any of them."""
    if model not in ('rrBLUP', 'RKHS'):
        raise ValueError(f"exact_kernel_column: model must be 'rrBLUP' or 'RKHS', got {model!r}.")
    n_o, n_v, n_t = len(train), len(valid), len(test)
    if n_o < 5:
        raise ValueError(f"exact_kernel_column: need at least 5 training rows, got {n_o}.")
    y_o = np.asarray(train.iloc[:, -1], dtype=float)
    if not np.all(np.isfinite(y_o)):
        raise ValueError("exact_kernel_column: non-finite training phenotype.")
    fingerprint = _frames_fingerprint(train, valid, test)
    h_key = float(params[2]) if model == 'RKHS' else 0.0
    eig_key = ('eig', fingerprint, model, h_key)
    cached = _EXACT_MEMO.get(eig_key)
    if cached is None:
        base_key = ('K', fingerprint, model)
        if model == 'RKHS':
            base = _EXACT_MEMO.get(base_key)
            if base is None:
                data = np.vstack([f.to_numpy(dtype=float)[:, :-1] for f in (train, valid, test)])
                base = _rkhs_normalised_distance(data)
                _memo_put(base_key, base)
            if not np.isfinite(h_key) or h_key <= 0.0:
                raise ValueError(f"exact_kernel: RKHS bandwidth h={params[2]!r} must be positive.")
            K = np.exp(-h_key * base)
        else:
            K = _EXACT_MEMO.get(base_key)
            if K is None:
                K = _kernel_matrix_for(model, (train, valid, test), params)
                _memo_put(base_key, K)
        if not np.all(np.isfinite(K)):
            raise ValueError(f"exact_kernel: the {model} kernel has non-finite entries.")
        d, U = np.linalg.eigh(K[:n_o, :n_o])
        cached = (d, U, K[n_o:, :n_o], float(np.mean(np.diag(K))))
        _memo_put(eig_key, cached)
    d, U, K_ao, _msx = cached
    # Only mean(diag(K)) enters the priors, so a 1x1 matrix carrying it is enough (the
    # full kernel is not kept next to the eigendecomposition for this call).
    prior = _bglr_default_priors(model, y_o, np.eye(1) * _msx, params)
    y_hat = _exact_kernel_predict(None, K_ao, y_o, prior, eig=(d, U))
    out = {'predicted_valid': y_hat[:n_v]}
    if want_test:
        out['predicted_test'] = y_hat[n_v:n_v + n_t]
    return out
