"""
ensemble_regularization.py
============================

Phase 2, Requirements 4 & 5 - shared objective-regularisation helpers for
the weighted-ensemble weight-optimisation methods (``Nelder_Mead.py``,
``Bayesian_optimisation.py``, ``Linear_transformation.py``).

Background (see ``EasiGP_Phase2_Design_Blueprint.md`` §4/§5 for the full
derivation): the Diversity Prediction Theorem objective published as
Eq.2/Eq.3 (and faithfully implemented in ``Nelder_Mead.py``/
``Bayesian_optimisation.py``) reduces algebraically to a
**linear-fractional function of the weight vector `w`**
(``sum(wi*ci) / sum(wi)``, with each ``ci`` a validation-set constant
independent of `w`). Linear-fractional objectives over a box-constrained
domain are *always* optimised at a vertex of the feasible region - i.e.
one model driven to the boundary, the rest to the opposite boundary. This
is a structural property of the published objective, not an
implementation bug, and it directly explains both reported symptoms:
near-one-hot weight collapse, and validation-noise overfitting (a vertex
solution is really a high-variance single-model *selection*, not a
genuine diversity-weighted blend).

This module centralises the fix (a strictly-convex regulariser added to
the bracket before optimising) and the optional alternate, naturally
interior-optimised objective, so the two forms are implemented ONCE and
shared by every weight-optimisation method, rather than duplicated
per-file.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
# Already a project dependency (used directly, top-level, by
# models/hyperparameter_tuning.py) - no lazy-import guard needed (I12: not
# a NEW dependency).
from scipy.optimize import minimize, nnls


def _weighted_row_mean(values: "np.ndarray", *, w: "Optional[np.ndarray]" = None,
                        W: "Optional[np.ndarray]" = None) -> float:
    """ver4-25, R8 (design blueprint SS2, Stage 6): `m_W(values)` over
    the ROW (individual) axis - `1'W values / 1'W 1` (full mode) or
    `sum(w*values)/sum(w)` (diagonal mode) - blueprint SS2's own
    weighted-mean definition, applied here to a per-row bracket term
    (e.g. the DPT `second_term - third_term` vector) rather than to a
    prediction/actual pair (`blue_weights.weighted_pearson`/
    `weighted_mse` already cover that case). `w`/`W` both `None` (every
    pre-ver4-25 caller) is a byte-identical `float(np.mean(values))` -
    I11.

    Shared by `Nelder_Mead.py`/`Bayesian_optimisation.py`'s own literal
    (non-vectorised-objective-mode) DPT-ratio bracket, so the SAME
    weighted-mean definition is used whether or not
    `W_OPT_VECTORISED_OBJECTIVE` is set - see each caller's own
    integration for the vectorised form (`values = (E-D) @ weights`,
    reduced with this same helper)."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float('nan')
    if W is not None:
        W = np.asarray(W, dtype=float)
        ones = np.ones(values.shape[0])
        denom = float(ones @ W @ ones)
        if not np.isfinite(denom) or denom == 0:
            return float(np.mean(values))
        return float(ones @ W @ values) / denom
    if w is not None:
        w = np.asarray(w, dtype=float)
        w_sum = float(np.sum(w))
        if not np.isfinite(w_sum) or w_sum == 0:
            return float(np.mean(values))
        return float(np.sum(w * values)) / w_sum
    return float(np.mean(values))


def diversity_regularizer(
    weights: Sequence[float],
    w_sum: float,
    diversity_penalty: float,
    method: str = 'ridge_to_uniform',
) -> float:
    """Strictly-convex penalty term, evaluated on the NORMALISED weight
    shares (``wi / w_sum``), that breaks the linear-fractional DPT-ratio
    objective's vertex-seeking shape once added to it (Requirement 5,
    §4.3a). Always ``>= 0``, and exactly ``0`` at equal weighting - so a
    weight vector identical to naive/equal weighting is never penalised
    relative to itself, only candidates that diverge further from it.

    Parameters
    ----------
    weights : this candidate's raw (pre-normalisation) weight vector.
    w_sum : ``sum(weights)`` - passed in rather than recomputed, since
        every caller already has it (and has already guarded it against
        zero/non-finite before calling this).
    diversity_penalty : the regularisation strength ``lambda``. ``0``
        (or falsy) disables the penalty entirely - returns ``0.0``
        unconditionally, so this is a strict no-op / fully backward
        compatible default.
    method : ``'ridge_to_uniform'`` (default) - ``lambda * sum((wi/w_sum -
        1/N)^2)``, a simple ridge-to-equal-weighting penalty. ``'entropy'``
        - ``lambda * (log(N) - entropy(w/w_sum))``, i.e. the (non-negative)
        entropy DEFICIT relative to the maximum-entropy (uniform)
        distribution, on the same "0 at uniform, growing away from it"
        convention.

    Returns
    -------
    A single non-negative float to ADD to a to-be-MINIMISED objective (or
    to SUBTRACT from a to-be-MAXIMISED reciprocal target before
    reciprocating - see each caller's own integration, since Nelder-Mead
    minimises this bracket directly while Bayesian optimisation maximises
    its reciprocal).
    """
    if not diversity_penalty:
        return 0.0
    n = len(weights)
    if n == 0 or not np.isfinite(w_sum) or w_sum == 0:
        return 0.0
    shares = np.asarray(weights, dtype=float) / w_sum
    if method == 'entropy':
        shares_pos = np.clip(shares, 1e-12, None)
        shares_pos = shares_pos / shares_pos.sum()
        entropy = -float(np.sum(shares_pos * np.log(shares_pos)))
        max_entropy = float(np.log(n)) if n > 1 else 0.0
        return float(diversity_penalty) * max(0.0, max_entropy - entropy)
    # 'ridge_to_uniform' (default)
    uniform = 1.0 / n
    return float(diversity_penalty) * float(np.sum((shares - uniform) ** 2))


def ensemble_mse_objective(
    weights: Sequence[float],
    w_sum: float,
    data_valid: "pd.DataFrame",
    model_selected: List[str],
    pred_matrix: "Optional[np.ndarray]" = None,
    actual: "Optional[np.ndarray]" = None,
    *,
    w: "Optional[np.ndarray]" = None,
    W: "Optional[np.ndarray]" = None,
) -> float:
    """The optional alternate ``OBJECTIVE_MODE='ensemble_mse'`` objective
    (§4.3b): the ensemble prediction's OWN realised MSE against the actual
    validation target, ``mean_s( (sum_i wi*Mi(s) - V(s))^2 )``, evaluated
    on the normalised weight shares (``wi/w_sum``) so this stays scale-
    invariant exactly like the DPT-ratio objective it replaces.

    This is quadratic in `w` (the inner sum is linear in `w`, then
    squared) - a well-posed, generically strictly-convex objective with a
    genuine interior optimum, the same reason ordinary least-squares
    stacking regression doesn't collapse to one-hot weights. Departs from
    the literal DPT Eq.2/3 text but not from the theorem's intent
    (minimising many-model error).

    Always MINIMISED (both callers, Nelder-Mead and Bayesian optimisation,
    adapt their own direction around this consistently - see each file's
    own integration).

    Parameters
    ----------
    pred_matrix, actual : ver4-5 R1.d (blueprint §3.4) - an optional
        PRECOMPUTED form: ``pred_matrix`` is the (N, K) array of
        validation predictions (``data_valid[model_selected].to_numpy()``)
        and ``actual`` is the (N,) array of true validation targets
        (``data_valid['actual'].to_numpy()``), both built ONCE by the
        caller (they depend only on the validation split, never on the
        candidate weight vector) rather than re-derived from
        ``data_valid``/``model_selected`` on every single call - the same
        precompute-once discipline ``Bayesian_optimisation.py``'s own
        vectorised DPT-ratio path uses. When either is ``None`` (the
        default - every pre-ver4-5 caller), this falls back to building
        the combined prediction from ``data_valid``/``model_selected``
        exactly as before, byte-for-byte - so this remains a strict,
        backward-compatible ADDITION, not a breaking signature change.
    w, W : ver4-25, R8 (design blueprint SS2, Stage 6) - optional
        BLUE-reliability scoring weights for the SAME validation
        individuals (diagonal `w`/full `W` - see `blue_weights.
        scoring_for()`). Both `None` (every pre-ver4-25 caller) is a
        byte-identical `mean((combined-actual)**2)` - I11. Mirrors
        `pipeline_utils.safe_regression_metrics()`'s own "separate
        branch, not a W=I unification" discipline.
    """
    n = len(weights)
    if n == 0 or not np.isfinite(w_sum) or w_sum == 0:
        return float('inf')
    shares = np.asarray(weights, dtype=float) / w_sum

    if pred_matrix is not None and actual is not None:
        if pred_matrix.shape[0] == 0:
            return float('inf')
        combined = pred_matrix @ shares
        actual_local = actual
    else:
        combined = np.zeros(data_valid.shape[0], dtype=float)
        for i, m in enumerate(model_selected):
            combined = combined + data_valid[m].to_numpy(dtype=float) * shares[i]
        actual_local = data_valid['actual'].to_numpy(dtype=float)
        if combined.shape[0] == 0:
            return float('inf')

    resid = combined - actual_local
    if W is not None:
        W = np.asarray(W, dtype=float)
        trW = float(np.trace(W))
        if not np.isfinite(trW) or trW == 0:
            return float('inf')
        val = float(resid @ W @ resid) / trW
        return val if np.isfinite(val) else float('inf')
    if w is not None:
        w = np.asarray(w, dtype=float)
        w_total = float(np.sum(w))
        if not np.isfinite(w_total) or w_total == 0:
            return float('inf')
        val = float(np.sum(w * resid * resid)) / w_total
        return val if np.isfinite(val) else float('inf')
    return float(np.mean(resid ** 2))


def apply_naive_shrinkage(weights: Iterable[float], alpha) -> List[float]:
    """Requirement 5 (secondary mitigation): blend the search's own
    (already sum-to-1-normalised) weight vector toward equal weighting,
    ``w_final = alpha*w_optimized + (1-alpha)*(1/N)`` - a cheap additional
    safety margin against residual test-set variance, applied on top of
    (not instead of) the diversity regulariser above.

    ``alpha`` is clamped to ``[0, 1]``; ``alpha=1.0`` (or ``None`` /
    falsy-but-not-0, i.e. unset) is a strict no-op - the fully backward-
    compatible default for every run predating this option.
    """
    weights = list(weights)
    if alpha is None:
        return weights
    alpha = float(np.clip(float(alpha), 0.0, 1.0))
    if alpha >= 1.0:
        return weights
    n = len(weights)
    if n == 0:
        return weights
    uniform = 1.0 / n
    return [alpha * w + (1.0 - alpha) * uniform for w in weights]


def read_regularization_settings(hyperparameters_opt: Sequence, start_index: int) -> dict:
    """Read the (optional, APPENDED-ONLY - never inserted mid-list, so
    this never reorders/breaks any existing positional index a saved
    config already relies on) diversity-regularisation settings from a
    weight-optimisation method's own flat ``HYPERPARAMETERS_OPT`` list,
    starting at ``start_index`` (i.e. the first index PAST that method's
    own pre-existing, original parameters).

    Update ID ver4-6, R5.2a: the ``objective_mode`` FALLBACK (used only
    when this entry is absent from ``hyperparameters_opt`` altogether -
    e.g. a config saved before this field existed) changed from
    ``'dpt_ratio'`` to ``'ensemble_mse'``. The published Diversity-
    Prediction-Theorem ratio objective is linear in the normalised weight
    shares (a convex combination of validation-set constants), so its
    minimum is ALWAYS a simplex vertex - i.e. a single-model selection,
    never a genuine blend (measured: realised validation MSE 43% WORSE
    than equal weighting at that vertex - blueprint E3). ``'ensemble_mse'``
    (the ensemble's own realised validation MSE, quadratic and interior-
    optimised) does not have this defect. ``'dpt_ratio'`` remains fully
    selectable - explicitly setting it in a config is unaffected by this
    change - and is retained for reproducing published DPT results, not
    recommended for prediction accuracy. Every other entry's fallback is
    unchanged from before this update: ``diversity_penalty=0.0``
    (regulariser off), ``alpha=1.0`` (naive-shrinkage blend off),
    ``diversity_method='ridge_to_uniform'``.
    """
    def _get(offset, default):
        idx = start_index + offset
        return hyperparameters_opt[idx] if len(hyperparameters_opt) > idx else default

    return {
        'diversity_penalty': float(_get(0, 0.0) or 0.0),
        'objective_mode': _get(1, 'ensemble_mse') or 'ensemble_mse',
        'alpha': float(_get(2, 1.0) if _get(2, 1.0) is not None else 1.0),
        'diversity_method': _get(3, 'ridge_to_uniform') or 'ridge_to_uniform',
    }


# --------------------------------------------------------------------------- #
# ver4-6 R6 - analytic simplex least-squares weights
# --------------------------------------------------------------------------- #

def analytic_simplex_weights(pred_matrix: "np.ndarray", actual: "np.ndarray", *,
                              ridge: float = 1e-6,
                              w: "Optional[np.ndarray]" = None,
                              W: "Optional[np.ndarray]" = None) -> "np.ndarray":
    """Non-negative, sum-to-1 least-squares combination weights (ver4-6
    R6.2).

    Solves ``min ||P s - y||^2  s.t.  s >= 0, sum(s) = 1`` via
    ``scipy.optimize.minimize(method='SLSQP')``, started from equal
    weighting, with a small ridge term added to the Gram matrix for
    conditioning when base-model predictions are near-collinear (routine:
    models trained on the same markers correlate strongly). ``K <= ~20``
    dimensions here, so this reliably converges in single-digit
    milliseconds (AC-R6.4) - deterministically, unlike a stochastic
    black-box search over the same, already-convex problem.

    Never raises: returns equal weighting (never ``None``) if
    ``pred_matrix``/``actual`` are empty, mismatched, or the solve fails
    to converge / returns a non-finite vector - one ``[GP] NOTE:`` is
    printed in that case, but the caller is always handed something
    usable (R6.6 failure modes).

    Parameters
    ----------
    pred_matrix : (N, K) array of validation predictions, one column per
        base model - see ``ensemble_mse_objective()``'s own docstring for
        the precomputed-once discipline this mirrors.
    actual : (N,) array of true validation targets.
    ridge : Tikhonov regularisation strength added to the diagonal of the
        (K, K) Gram matrix ``P.T @ P`` before solving - conditions the
        problem when K base models are highly correlated (e.g. several
        GAT variants on the same markers), never large enough to bias a
        well-conditioned solve away from the true least-squares optimum.
    w, W : ver4-25, R8 (design blueprint SS2, Stage 6, AC8.7) - optional
        BLUE-reliability weights for these SAME validation individuals
        (diagonal `w`, a 1-D array; full `W`, a dense PSD matrix - see
        `blue_weights.scoring_for()`). When given, this solves the GLS
        form ``min (Ps-y)'W(Ps-y) s.t. s>=0, sum(s)=1`` instead (diagonal
        mode uses `W=diag(w)` implicitly, never materialised densely -
        matching `blue_weights.py`'s own "no dense matrix built for
        diagonal mode" discipline). Both `None` (every pre-ver4-25
        caller) reproduces the ORIGINAL, ordinary (unweighted)
        least-squares solve byte-for-byte - a strictly separate branch,
        not a `W=I` special case (I11 - mirrors
        `pipeline_utils.safe_regression_metrics()`'s own discipline).
    """
    pred_matrix = np.asarray(pred_matrix, dtype=float)
    actual = np.asarray(actual, dtype=float)
    k = pred_matrix.shape[1] if pred_matrix.ndim == 2 else 0
    k = max(k, 1)
    equal = np.full(k, 1.0 / k)

    if (pred_matrix.ndim != 2 or pred_matrix.shape[0] == 0 or pred_matrix.shape[1] == 0
            or actual.shape[0] != pred_matrix.shape[0]):
        print(f"[ensemble_regularization] NOTE: analytic_simplex_weights received "
              f"empty or mismatched inputs (pred_matrix shape "
              f"{getattr(pred_matrix, 'shape', None)}, actual shape "
              f"{getattr(actual, 'shape', None)}) - returning equal weighting "
              f"instead. Never fatal.")
        return equal

    try:
        n_rows = pred_matrix.shape[0]
        if W is not None:
            # ver4-25, R8: dense GLS form. `gram`/`target` are the
            # weighted Gram matrix/target vector; `scale` is `tr(W)`,
            # the SAME normaliser `blue_weights.weighted_mse()` divides
            # by, so this loss is directly comparable to every other
            # weighted MSE this run reports.
            W = np.asarray(W, dtype=float)
            WP = W @ pred_matrix
            gram = pred_matrix.T @ WP + ridge * np.eye(k)
            target = pred_matrix.T @ (W @ actual)
            scale = float(np.trace(W))
            scale = scale if (np.isfinite(scale) and scale > 0) else float(n_rows)

            def _loss(s):
                resid = pred_matrix @ s - actual
                return float(resid @ W @ resid) / scale

            def _grad(s):
                return (2.0 / scale) * (gram @ s - target) - (2.0 / scale) * (ridge * s)
        elif w is not None:
            # ver4-25, R8: diagonal mode - `W=diag(w)` implicit, never
            # materialised as a dense (N,N) matrix (cheap even for a
            # large validation split).
            w = np.asarray(w, dtype=float)
            wP = pred_matrix * w[:, None]
            gram = pred_matrix.T @ wP + ridge * np.eye(k)
            target = wP.T @ actual
            scale = float(np.sum(w))
            scale = scale if (np.isfinite(scale) and scale > 0) else float(n_rows)

            def _loss(s):
                resid = pred_matrix @ s - actual
                return float(np.sum(w * resid * resid)) / scale

            def _grad(s):
                return (2.0 / scale) * (gram @ s - target) - (2.0 / scale) * (ridge * s)
        else:
            gram = pred_matrix.T @ pred_matrix + ridge * np.eye(k)
            target = pred_matrix.T @ actual

            def _loss(s):
                resid = pred_matrix @ s - actual
                return float(np.mean(resid ** 2))

            def _grad(s):
                return (2.0 / n_rows) * (gram @ s - target) - (2.0 / n_rows) * (ridge * s)

        constraints = ({'type': 'eq', 'fun': lambda s: float(np.sum(s) - 1.0)},)
        bounds = [(0.0, 1.0)] * k
        res = minimize(_loss, equal, jac=_grad, method='SLSQP', bounds=bounds,
                        constraints=constraints, options={'maxiter': 200, 'ftol': 1e-10})
        if not res.success or not np.all(np.isfinite(res.x)):
            print(f"[ensemble_regularization] NOTE: analytic_simplex_weights SLSQP "
                  f"solve did not converge ({getattr(res, 'message', 'no message')}) "
                  f"- returning equal weighting instead. Never fatal.")
            return equal
        s = np.clip(res.x, 0.0, None)
        total = float(s.sum())
        if not np.isfinite(total) or total <= 0:
            return equal
        return s / total
    except Exception as exc:
        print(f"[ensemble_regularization] NOTE: analytic_simplex_weights failed "
              f"({type(exc).__name__}: {exc}) - returning equal weighting instead. "
              f"Never fatal.")
        return equal


# --------------------------------------------------------------------------- #
# ver4-6 R5.2b - validation-graded floor, shared by every weight method
# --------------------------------------------------------------------------- #

def select_weights_with_floor(
    w_candidate: Sequence[float],
    pred_matrix: "np.ndarray",
    actual: "np.ndarray",
    *,
    shrinkage_alpha: float = 1.0,
    extra_candidates: "Optional[List]" = None,
    tol: float = 0.0,
    w: "Optional[np.ndarray]" = None,
    W: "Optional[np.ndarray]" = None,
) -> "Tuple[List[float], str, dict]":
    """Return ``(w_final, chosen_label, diagnostics)`` (ver4-6 R5.2b).

    Every candidate - the search's own winner (``w_candidate``), equal
    weighting, and anything in ``extra_candidates`` (e.g. R6's
    ``analytic_simplex_weights(...)`` result) - is normalised to the
    simplex and graded on its OWN realised validation MSE,
    ``mean((pred_matrix @ share - actual)**2)``, NOT on whatever objective
    the search used to pick ``w_candidate``. Equal weighting is the
    starting ("current best") candidate, so any later candidate must
    beat it - and every subsequent one it displaces - by more than
    ``tol`` to win; a candidate that only TIES the current best never
    displaces it. This makes "weighted ensembles are never worse than the
    naive ensemble on the validation split" a construction-level
    guarantee for every caller, independent of which objective the search
    itself optimised (``dpt_ratio`` or ``ensemble_mse``) and independent
    of whether that objective is even convex.

    Naive shrinkage (``apply_naive_shrinkage``) is then applied to the
    winner, and the shrunk vector is graded once more against equal
    weighting before being accepted - so shrinkage can never make things
    worse than equal weighting either.

    Parameters
    ----------
    w_candidate : the search's own winning weight vector (raw - i.e.
        possibly not yet non-negative or sum-to-1; normalised here).
    pred_matrix, actual : (N, K) validation predictions and (N,) actual
        validation targets - see ``ensemble_mse_objective()``'s own
        docstring for the precomputed-once discipline this mirrors.
    shrinkage_alpha : forwarded to ``apply_naive_shrinkage`` unchanged;
        ``1.0`` (default) is a no-op.
    extra_candidates : additional raw weight vectors to grade alongside
        ``w_candidate`` and equal weighting. Each entry is either a plain
        vector (auto-labelled ``'extra_0'``, ``'extra_1'``, ...) or an
        explicit ``(label, vector)`` pair (e.g. ``('analytic', ...)``).
    tol : a candidate must beat the running-best MSE by MORE than this to
        become the new best - ``0.0`` (default) means any strict
        improvement wins; a small positive value additionally protects
        against choosing a "win" that is really floating-point noise.
    w, W : ver4-25, R8 (design blueprint SS2, Stage 6) - optional
        BLUE-reliability weights for these SAME validation individuals
        (see `analytic_simplex_weights()`'s identical parameters). Every
        candidate - including equal weighting - is graded on `mse_W`
        instead of the plain mean when given, so "never worse than the
        naive ensemble" becomes "never worse ON THE SAME WEIGHTED METRIC
        every model and ensemble this run reports" (blueprint SS2 R8
        Fairness Principle P8). Both `None` (default) is byte-identical
        to before this parameter existed (I11).

    Returns
    -------
    ``w_final`` : plain list of floats, non-negative, summing to 1 (never
        raises, never returns ``None`` - degenerates to equal weighting on
        any failure, per R5.7/R6.6).
    ``chosen_label`` : ``'search'``, ``'equal'``, or an ``extra_candidates``
        label (e.g. ``'analytic'``) - which raw candidate the final,
        possibly-shrunk vector was derived from.
    ``diagnostics`` : ``{'<label>_valid_mse': float, ...}`` for every
        candidate considered, plus ``'shrunk_valid_mse'`` - for reporting
        (e.g. into ``Result/<n>/wopt_method_comparison.csv``, R8) or
        logging.
    """
    pred_matrix = np.asarray(pred_matrix, dtype=float)
    actual = np.asarray(actual, dtype=float)
    k = pred_matrix.shape[1] if pred_matrix.ndim == 2 and pred_matrix.shape[1] else len(list(w_candidate))
    k = max(k, 1)
    equal = np.full(k, 1.0 / k)

    def _normalise(w):
        try:
            w = np.asarray(w, dtype=float)
        except Exception:
            return None
        if w.size != k or not np.all(np.isfinite(w)):
            return None
        w = np.clip(w, 0.0, None)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0:
            return None
        return w / total

    def _score(w_norm):
        if w_norm is None or pred_matrix.shape[0] == 0:
            return float('inf')
        combined = pred_matrix @ w_norm
        resid = combined - actual
        if W is not None:
            trW = float(np.trace(W))
            if not np.isfinite(trW) or trW == 0:
                return float('inf')
            val = float(resid @ W @ resid) / trW
            return val if np.isfinite(val) else float('inf')
        if w is not None:
            w_total = float(np.sum(w))
            if not np.isfinite(w_total) or w_total == 0:
                return float('inf')
            val = float(np.sum(w * resid * resid)) / w_total
            return val if np.isfinite(val) else float('inf')
        return float(np.mean(resid ** 2))

    diagnostics = {}
    equal_mse = _score(equal)
    diagnostics['equal_valid_mse'] = equal_mse
    best_label, best_w, best_mse = 'equal', equal, equal_mse

    candidates = [('search', w_candidate)]
    if extra_candidates:
        for i, cand in enumerate(extra_candidates):
            if isinstance(cand, tuple) and len(cand) == 2 and isinstance(cand[0], str):
                candidates.append(cand)
            else:
                candidates.append((f'extra_{i}', cand))

    for label, cand in candidates:
        w_norm = _normalise(cand)
        mse = _score(w_norm)
        diagnostics[f'{label}_valid_mse'] = mse
        if w_norm is not None and mse < best_mse - tol:
            best_label, best_w, best_mse = label, w_norm, mse

    # Naive shrinkage applied to the winner, then re-graded against equal
    # weighting once more so shrinkage can never make things worse either
    # (R5.2d - shrinkage is a *test*-set variance mitigation; it must not
    # be allowed to erode the validation-side guarantee established
    # above).
    shrunk = np.asarray(apply_naive_shrinkage(best_w.tolist(), shrinkage_alpha), dtype=float)
    shrunk_norm = _normalise(shrunk)
    shrunk_mse = _score(shrunk_norm)
    diagnostics['shrunk_valid_mse'] = shrunk_mse

    if shrunk_norm is not None and shrunk_mse < equal_mse - tol:
        w_final, final_label = shrunk_norm, best_label
    elif best_mse <= equal_mse:
        w_final, final_label = best_w, best_label
    else:
        w_final, final_label = equal, 'equal'

    return w_final.tolist(), final_label, diagnostics


# --------------------------------------------------------------------------- #
# ver4-6 R7.2(b) - stick-breaking simplex parameterisation
# --------------------------------------------------------------------------- #

def stick_breaking_to_simplex(u: Sequence[float]) -> "np.ndarray":
    """Map a point on the ``(K-1)``-dimensional unit cube onto the
    ``K``-dimensional probability simplex (ver4-6 R7.2(b)): symmetric
    (every vertex of the simplex is reachable), unit-cube-scaled (composes
    with R1's unit-cube encoding philosophy), gated by
    ``W_OPT_SIMPLEX_SEARCH`` in every caller.

    ``shares[i] = remaining * (1 - u[i]**(1/(K-i-1)))`` for
    ``i = 0 .. K-2``, consuming a fraction of whatever "stick" of total
    mass 1 remains at each step; ``shares[K-1]`` takes whatever mass is
    left. This is the standard construction for a UNIFORM distribution
    over the simplex when every ``u[i]`` is drawn ``Uniform(0, 1)``
    independently (see e.g. Devroye, *Non-Uniform Random Variate
    Generation*, ch. XI.4) - so a Bayesian search's own uniform
    ``init_points`` draws explore the simplex evenly, not concentrated
    toward any particular region. Every coordinate ``u[i]=0`` takes the
    ENTIRE remaining stick immediately (reaching the corresponding
    vertex); every coordinate ``u[i]=1`` for ``i < K-1`` defers the whole
    stick to the final share (reaching the LAST vertex) - both extremes,
    and everything between them, are reachable (AC-R7.1).

    Always returns a valid share vector: non-negative, summing to 1
    (renormalised defensively against floating-point drift) - never
    raises, never returns a vector with a negative entry (R7.2(c): this
    is exactly why negative weights are UNREACHABLE under simplex search
    - see each caller's own `[GP] NOTE:` when ``Minimum boundary < 0`` is
    configured alongside this).
    """
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    k_minus_1 = u.shape[0]
    k = k_minus_1 + 1
    shares = np.empty(k, dtype=float)
    remaining = 1.0
    for i in range(k_minus_1):
        exponent = 1.0 / (k - i - 1)
        take = remaining * (1.0 - u[i] ** exponent)
        take = min(max(take, 0.0), remaining)
        shares[i] = take
        remaining -= take
    shares[k_minus_1] = max(remaining, 0.0)
    total = shares.sum()
    if total > 0 and np.isfinite(total):
        shares = shares / total
    else:
        shares = np.full(k, 1.0 / k)
    return shares


def simplex_to_stick_breaking(shares: Sequence[float]) -> "np.ndarray":
    """Approximate inverse of ``stick_breaking_to_simplex()``: given a
    share vector (assumed non-negative, summing to ~1), returns the
    ``u`` in ``[0, 1]^(K-1)`` that reproduces it. Used to seed a stick-
    breaking search with a KNOWN share vector (equal weighting, or R6's
    analytic solution) as an explicit probe point / restart origin -
    exactly the way the pre-ver4-6, un-normalised K-dimensional box
    search already seeds with the equal-weight value directly.

    Degenerate inputs (all-zero remaining mass partway through) map to
    ``u=0.5`` for every remaining coordinate rather than raising or
    dividing by zero - never fatal, only affects where the resulting
    probe point lands, never whether it is a valid seed."""
    shares = np.clip(np.asarray(shares, dtype=float), 0.0, None)
    k = shares.shape[0]
    total = float(shares.sum())
    if k <= 1:
        return np.array([])
    if total <= 0 or not np.isfinite(total):
        return np.full(k - 1, 0.5)
    shares = shares / total
    u = np.empty(k - 1, dtype=float)
    remaining = 1.0
    for i in range(k - 1):
        # Inverse of the forward map's `take = remaining * (1 - u**exponent)`
        # with `exponent = 1/(k-i-1)`: solving for u gives
        # `u = (1 - take/remaining) ** (k-i-1)` - the RECIPROCAL power of
        # the forward exponent, not the same one.
        take = float(shares[i])
        if remaining <= 1e-15:
            u[i] = 0.5
            continue
        frac = min(max(take / remaining, 0.0), 1.0)
        u[i] = (1.0 - frac) ** (k - i - 1)
        remaining -= take
    return u


# --------------------------------------------------------------------------- #
# Joint-optimisation fix (Round 10) - cross-fitted (out-of-fold) weights
# --------------------------------------------------------------------------- #
#
# WHY THIS EXISTS. ``analytic_simplex_weights()`` minimises the MSE of the
# combined prediction on the very rows it is handed, and
# ``select_weights_with_floor()`` then "grades" the result on those SAME rows.
# That grading is vacuous: the analytic solution is, by construction, the
# minimiser of that MSE over the whole simplex, so equal weighting can never
# beat it in-sample - the "never worse than equal weighting" floor can never
# fire. On the small validation split a typical GP() task has (tens of rows,
# a handful of strongly correlated base models) the true gain of optimal over
# equal weighting is usually SMALLER than the estimation noise of the K-1
# free weights - the well-known "forecast combination puzzle" - so in-sample
# optimal weights systematically lose to a plain average on the TEST split.
#
# The fix is to grade weights on rows they were NOT fitted on. The two helpers
# below do that purely on already-computed validation predictions (no model is
# refit; each solve is a few milliseconds):
#
#   * ``cross_fitted_combination()``  - out-of-fold combined prediction for a
#     fixed shrinkage level; used to give the joint search an honest,
#     non-optimistic per-candidate score.
#   * ``cv_shrunk_simplex_weights()`` - picks the shrinkage level between equal
#     weighting (0) and the full analytic solution (1) by K-fold CV over the
#     validation rows, with a paired one-standard-error rule that prefers the
#     most equal-like level that is statistically indistinguishable from the
#     best - so equal weighting is the default unless analytic weights
#     demonstrably help out of sample.

def _fold_indices(n: int, n_folds: int, rng: "np.random.Generator") -> List["np.ndarray"]:
    order = rng.permutation(n)
    return [f for f in np.array_split(order, n_folds) if f.size]


def _cv_min_rows(n_folds: int) -> int:
    """Below this many rows there is no meaningful way to cross-fit K-1 free
    weights (each training fold would hold only a handful of rows)."""
    return max(10, 2 * int(n_folds))


def cross_fitted_combination(pred_matrix: "np.ndarray", actual: "np.ndarray", alpha: float, *,
                             n_folds: int = 5, seed: int = 0,
                             ridge: float = 1e-6,
                             weight_fn: "Optional[Callable[[np.ndarray, np.ndarray], Sequence[float]]]" = None,
                             return_folds: bool = False):
    """Out-of-fold combined prediction of ``pred_matrix`` (N, K).

    For every fold, the analytic simplex weights are solved on the OTHER
    folds only, shrunk toward equal weighting by ``alpha`` (``w = alpha *
    analytic + (1 - alpha) / K``; ``alpha=1`` -> pure analytic, ``0`` -> equal
    weighting), and applied to the held-out fold. Every row's combined value
    therefore comes from weights that never saw that row. Falls back to equal
    weighting for every row when there are too few rows to cross-fit. Never
    raises.

    ver4-34, R2.a (both new keywords default to the previous behaviour, so every
    existing caller is unchanged):

    ``weight_fn`` - when given, ``weight_fn(P_train_folds, y_train_folds)`` is
    called in place of ``analytic_simplex_weights`` for every training fold and
    must return K non-negative weights summing to 1 (the robust bagged
    estimator, ``robust_weights_only``). ``alpha`` is still applied around it
    (the joint search passes ``alpha=1.0``, i.e. no shrinkage). A ``weight_fn``
    that returns a wrongly shaped or non-finite vector for one fold is replaced
    by equal weighting for that fold only (and reported once), so a single bad
    fold never aborts a search.

    ``return_folds`` - when True the return value is ``(combined, folds)``,
    where ``folds`` is the list of held-out POSITIONAL index arrays that were
    used, in the order they were scored (``[]`` when there were too few rows,
    or ``alpha <= 0``, to cross-fit and equal weighting was used for every
    row). Why: the CV objective (``objective_data='cv_train_valid'``) needs
    per-fold scores of the cross-fitted combination, so the paired fresh gate
    can compare a candidate with the defaults fold by fold. The partition is
    ``_fold_indices(n, n_folds, np.random.default_rng(seed))``, i.e. identical
    to the one ``joint_tuning._cv_partition`` draws for the same ``(n, folds,
    seed)``."""
    P = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(actual, dtype=float)
    n, k = P.shape if P.ndim == 2 else (0, 0)
    if n == 0 or k == 0:
        return (np.zeros(n), []) if return_folds else np.zeros(n)
    equal = np.full(k, 1.0 / k)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    n_folds = int(max(2, min(n_folds, n)))
    if alpha <= 0.0 or n < _cv_min_rows(n_folds):
        return (P @ equal, []) if return_folds else P @ equal
    out = np.empty(n, dtype=float)
    rng = np.random.default_rng(seed)
    folds = _fold_indices(n, n_folds, rng)
    bad_fold_reported = False
    for held in folds:
        mask = np.ones(n, dtype=bool)
        mask[held] = False
        if weight_fn is None:
            w = analytic_simplex_weights(P[mask], y[mask], ridge=ridge)
        else:
            w = np.asarray(weight_fn(P[mask], y[mask]), dtype=float)
            if w.shape != (k,) or not np.all(np.isfinite(w)):
                if not bad_fold_reported:
                    print(f"[ensemble_regularization] NOTE: cross_fitted_combination's weight_fn "
                          f"returned shape {w.shape} / non-finite values for a {k}-column fold - "
                          f"using equal weighting for that fold instead. Never fatal.")
                    bad_fold_reported = True
                w = equal
        out[held] = P[held] @ (alpha * w + (1.0 - alpha) * equal)
    return (out, folds) if return_folds else out


def _valid_prior(prior, k: int) -> "Optional[np.ndarray]":
    """ver4-94 (R3): `prior` as a float vector of length `k` summing to 1, or ``None`` when it
    is unusable (wrong length, non-finite or negative entry, non-positive sum). Never raises."""
    try:
        vec = np.asarray(prior, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if vec.shape[0] != int(k) or not np.all(np.isfinite(vec)) or np.any(vec < 0.0):
        return None
    total = float(vec.sum())
    if total <= 0.0:
        return None
    return vec / total


def cv_shrunk_simplex_weights(pred_matrix: "np.ndarray", actual: "np.ndarray", *,
                              alpha_cap: float = 1.0,
                              alphas: "Optional[Sequence[float]]" = None,
                              n_folds: int = 5, n_repeats: int = 4, seed: int = 0,
                              one_se: bool = True, ridge: float = 1e-6,
                              return_fold_weights: bool = False,
                              prior: "Optional[np.ndarray]" = None
                              ) -> "Tuple[List[float], float, dict]":
    """Choose the ensemble weights by cross-validating the shrinkage level.

    Candidates are ``alpha * analytic + (1 - alpha) * equal`` for ``alpha`` in
    ``alphas`` (default 0, .25, .5, .75, 1, truncated at ``alpha_cap`` - the
    largest trust the caller allows the fitted weights). Each alpha is scored
    by repeated K-fold CV *over the rows of* ``pred_matrix`` (weights solved
    on the training folds, MSE measured on the held-out fold), so the choice
    reflects how the weights generalise, not how well they fit. With
    ``one_se`` the smallest alpha whose paired CV loss is within one standard
    error of the best alpha wins - i.e. the weights only move away from equal
    weighting when that is out-of-sample-significant.

    Returns ``(weights, alpha_star, diagnostics)`` where ``weights`` is a
    non-negative, sum-to-1 list built from the analytic solution on ALL rows
    shrunk by ``alpha_star``. With too few rows to cross-validate it returns
    equal weighting (``alpha_star = 0``). Never raises.

    ver4-44, R3: ``return_fold_weights=True`` additionally stores
    ``diagnostics['fold_weights']`` - the analytic simplex vector of EVERY
    (repeat, training-fold) fit of the loop, in the order they were solved, as a
    list of K-lists. Why: these vectors are already computed for the alpha
    choice; their mean is a lower-variance estimate of the same target than one
    refit on all rows (``shrinkage_path_weights(final='bagged')``), at zero extra
    fits. The default (False) changes nothing, byte for byte.

    ver4-94 (R3, ``prior``): the shrinkage target. ``None`` (the default) is equal
    weighting, byte for byte as before. A vector of K non-negative numbers (it is
    renormalised to sum 1) replaces ``equal`` everywhere: the candidates become
    ``alpha * analytic + (1 - alpha) * prior``, and so do the fallbacks. Why: the
    population-pooled Super learner adds columns fitted with other populations'
    lines. Its prior is equal over the member's OWN columns and 0 on the pooled
    columns, so alpha = 0 reproduces the own-lines-only equal-weight row and a
    pooled column gets weight only when cross-validation supports it. A prior of
    the wrong length, with a negative or non-finite entry, or summing to 0 is
    ignored (equal weighting) and ``diag['note']`` says so."""
    P = np.asarray(pred_matrix, dtype=float)
    y = np.asarray(actual, dtype=float)
    n, k = P.shape if P.ndim == 2 else (0, max(len(np.atleast_1d(actual)), 1))
    k = max(k, 1)
    equal = np.full(k, 1.0 / k)
    diag: dict = {'n_rows': int(n), 'n_folds': int(n_folds), 'n_repeats': int(n_repeats)}
    if prior is not None:
        # ver4-94 (R3): from here on `equal` is the shrinkage TARGET (the prior), not
        # necessarily uniform; every use below (fallbacks, loss loop, final vector) follows.
        prior_vec = _valid_prior(prior, k)
        if prior_vec is None:
            diag['note'] = (f"prior ignored: it must be {k} finite, non-negative numbers with a "
                            f"positive sum - equal weighting is the shrinkage target")
        else:
            equal = prior_vec
            diag['prior'] = prior_vec.tolist()
    if k == 1 or n == 0:
        diag['note'] = 'single model or no rows'
        return equal.tolist(), 0.0, diag
    n_folds = int(max(2, min(n_folds, n)))
    if n < _cv_min_rows(n_folds) or not np.all(np.isfinite(P)) or not np.all(np.isfinite(y)):
        diag['note'] = (f'too few (or non-finite) validation rows to cross-validate '
                        f'({n} < {_cv_min_rows(n_folds)}) - equal weighting')
        return equal.tolist(), 0.0, diag

    cap = float(np.clip(alpha_cap if alpha_cap is not None else 1.0, 0.0, 1.0))
    if alphas is None:
        alphas = [a for a in (0.0, 0.25, 0.5, 0.75, 1.0) if a < cap] + [cap]
    alphas = sorted({float(np.clip(a, 0.0, cap)) for a in alphas} | {0.0})

    # Per-row out-of-fold squared error, averaged over the repeats, per alpha.
    loss = np.zeros((len(alphas), n), dtype=float)
    rng = np.random.default_rng(seed)
    fold_weight_list: List[List[float]] = []
    for _ in range(max(1, int(n_repeats))):
        for held in _fold_indices(n, n_folds, rng):
            mask = np.ones(n, dtype=bool)
            mask[held] = False
            w = analytic_simplex_weights(P[mask], y[mask], ridge=ridge)
            if return_fold_weights:
                fold_weight_list.append(np.asarray(w, dtype=float).tolist())
            for ai, a in enumerate(alphas):
                pred = P[held] @ (a * w + (1.0 - a) * equal)
                loss[ai, held] += (pred - y[held]) ** 2
    loss /= max(1, int(n_repeats))
    mean_loss = loss.mean(axis=1)
    best_i = int(np.argmin(mean_loss))
    chosen_i = best_i
    if one_se:
        for ai in range(len(alphas)):                      # ascending alpha
            d = loss[ai] - loss[best_i]
            se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            if float(d.mean()) <= se:
                chosen_i = ai
                break
    alpha_star = alphas[chosen_i]
    w_full = analytic_simplex_weights(P, y, ridge=ridge)
    w_final = alpha_star * w_full + (1.0 - alpha_star) * equal
    w_final = np.clip(w_final, 0.0, None)
    w_final = w_final / w_final.sum()
    diag.update({
        'alphas': alphas,
        'cv_mse': {f'{a:g}': float(m) for a, m in zip(alphas, mean_loss)},
        'best_alpha': alphas[best_i],
        'chosen_alpha': alpha_star,
        'in_sample_mse_final': float(np.mean((P @ w_final - y) ** 2)),
    })
    if return_fold_weights:
        diag['fold_weights'] = fold_weight_list
    return w_final.tolist(), float(alpha_star), diag


# --------------------------------------------------------------------------- #
# ver4-34, R2.a - robust (bagged, correlation-clustered, exact-NNLS) simplex weights
# --------------------------------------------------------------------------- #
#
# Why this section exists (design blueprint EasiGP_ver4-34, SS0.1 H1 / R2.a):
# the joint search estimated K-1 free weights from ~50 validation rows over
# near-collinear base models (three BGLR columns that differ only in their
# prior), graded them on the very rows they were fitted to, and relied on one
# fixed 0.85 shrinkage as the only regulariser. The result is a high-variance
# vertex solution (one model gets almost all the weight). Three variance
# reductions, all data-driven (no equal-weight floor and no selection of the
# naive ensemble - requirement R2):
#   (1) correlation clusters - near-duplicate columns share ONE weight, split
#       equally inside the cluster, so collinearity cannot make the solution
#       jump between them;
#   (2) bagging - the mean of the simplex solutions over bootstrap resamples of
#       the rows, which smooths the discontinuity of a vertex solution;
#   (3) an exact, fast solver - non-negative least squares on an augmented
#       system (about 60 microseconds per solve), so 100 bags are affordable.

# Weight of the augmented "sum(s) = 1" row, relative to the data scale (see
# `nnls_simplex_weights`). The blueprint's snippet uses 1e3; the offline parity
# test of A-2a.1 (1 002 random, signal and collinear problems against the
# SLSQP solver) gave a worst loss excess of 1.2e-10 * ||y||^2 at 1e3 - inside
# the 1e-9 acceptance bound but with only an 8x margin - and 1.7e-15 at 1e5,
# at identical speed. 1e5 is therefore used (recorded in the Change Summary).
_NNLS_SUM_ROW_SCALE = 1.0e5
# A solution whose raw weight sum is further than this from 1 is not trusted:
# the caller falls back to `analytic_simplex_weights` (blueprint R2.a).
_NNLS_SUM_TOLERANCE = 1.0e-6
# Fewer rows than this cannot support a bootstrap of a simplex fit.
_ROBUST_MIN_ROWS = 10


def nnls_simplex_weights(P: "np.ndarray", y: "np.ndarray", *,
                         ridge: float = 0.0) -> "Optional[np.ndarray]":
    """Exact non-negative, sum-to-1 least-squares weights via ``scipy.optimize.nnls``.

    Solves ``min ||P s - y||^2 + ridge ||s||^2  s.t.  s >= 0, sum(s) = 1``.
    The equality constraint is imposed as one heavily weighted extra row
    (``big * 1's = big``) of an ordinary non-negative least-squares problem; the
    solution is then divided by its own sum, so the returned vector sums to 1 to
    machine precision. ``big`` scales with the data (``_NNLS_SUM_ROW_SCALE`` x
    the root-mean-square size of ``P``, never below ``_NNLS_SUM_ROW_SCALE``), so
    the row dominates whatever the units of the predictions are.

    Why not ``analytic_simplex_weights`` (SLSQP): it takes milliseconds, and
    the bagged estimator needs ~100 solves per candidate. D3 of the blueprint
    also records that its documented ridge is a no-op, so this function states
    its own ridge explicitly (``ridge=0.0`` by default: no hidden shrinkage).

    Args: ``P`` (N, K) base-model predictions; ``y`` (N,) targets; ``ridge``
    Tikhonov strength (>= 0).

    Returns the (K,) weight vector, or ``None`` when the solve cannot be trusted
    (empty, mismatched or non-finite input; ``nnls`` did not converge; or the
    raw weights did not sum to 1 within ``_NNLS_SUM_TOLERANCE``). A ``None`` is
    the signal for the caller to fall back to ``analytic_simplex_weights``.
    Never raises."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    if (P.ndim != 2 or P.shape[0] == 0 or P.shape[1] == 0 or y.ndim != 1
            or y.shape[0] != P.shape[0]):
        return None
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(y))):
        return None
    n, k = P.shape
    if k == 1:
        return np.ones(1)
    ridge = max(0.0, float(ridge or 0.0))
    big = _NNLS_SUM_ROW_SCALE * max(1.0, float(np.linalg.norm(P)) / np.sqrt(n))
    blocks = [P]
    targets = [y]
    if ridge > 0.0:
        blocks.append(np.sqrt(ridge) * np.eye(k))
        targets.append(np.zeros(k))
    blocks.append(big * np.ones((1, k)))
    targets.append(np.array([big]))
    try:
        s, _residual = nnls(np.vstack(blocks), np.concatenate(targets))
    except (RuntimeError, ValueError, np.linalg.LinAlgError):
        return None
    total = float(np.sum(s))
    if (not np.all(np.isfinite(s)) or not np.isfinite(total) or total <= 0.0
            or abs(total - 1.0) > _NNLS_SUM_TOLERANCE):
        return None
    return s / total


def correlation_clusters(P: "np.ndarray", threshold: float) -> "List[List[int]]":
    """Group the columns of ``P`` whose pairwise Pearson correlation is >= ``threshold``.

    Greedy complete linkage, deterministic in column order: each column joins the
    FIRST existing cluster in which it correlates >= ``threshold`` with EVERY
    member, otherwise it starts a new cluster. A constant column (undefined,
    NaN correlation) therefore always forms its own cluster.

    Args: ``P`` (N, K) predictions; ``threshold`` e.g. 0.97 (a value above 1
    gives one cluster per column). Returns a list of lists of column indices,
    covering every column exactly once, ordered by first member. Never raises
    for a 2-D array; a 1-D or empty input returns one cluster per column."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[1] == 0:
        return []
    k = P.shape[1]
    if P.shape[0] < 2 or k == 1:
        return [[j] for j in range(k)]
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(P.T)
    clusters: List[List[int]] = []
    for j in range(k):
        for members in clusters:
            if all(corr[j, i] >= threshold for i in members):
                members.append(j)
                break
        else:
            clusters.append([j])
    return clusters


def _robust_core(P: "np.ndarray", y: "np.ndarray", *, n_bags: int, cluster_corr: float,
                 seed: int, ridge: float) -> "Tuple[np.ndarray, dict]":
    """Shared body of ``robust_simplex_weights`` / ``robust_weights_only``.

    Returns ``(weights (K,), diag)`` and never prints: ``diag['note']`` carries a
    message when the inputs forced the analytic fallback, and the public
    wrappers decide whether to print it (the cross-fitted scoring path calls
    this thousands of times and must stay quiet). ``diag`` keys: ``clusters``,
    ``bag_weights`` ((B, K) expanded to the original columns), ``n_rows``,
    ``fallbacks`` (bags that needed ``analytic_simplex_weights``), and
    ``note`` (only on the fallback path)."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(P.shape[0]) if P.ndim == 2 else 0
    k = int(P.shape[1]) if P.ndim == 2 else 0
    diag = {'clusters': [[j] for j in range(k)], 'bag_weights': None, 'n_rows': n,
            'fallbacks': 0}
    usable = (P.ndim == 2 and n >= _ROBUST_MIN_ROWS and k >= 1 and y.ndim == 1
              and y.shape[0] == n and np.all(np.isfinite(P)) and np.all(np.isfinite(y)))
    if not usable:
        w = analytic_simplex_weights(P, y) if (P.ndim == 2 and k >= 1) else np.zeros(0)
        diag['note'] = (f"robust_simplex_weights needs >= {_ROBUST_MIN_ROWS} finite rows and a "
                        f"matching target (got {n} row(s), {k} column(s)) - using "
                        f"analytic_simplex_weights instead")
        diag['bag_weights'] = np.asarray(w, dtype=float).reshape(1, -1)
        return np.asarray(w, dtype=float), diag

    clusters = correlation_clusters(P, cluster_corr)
    diag['clusters'] = [list(c) for c in clusters]
    super_cols = np.column_stack([P[:, c].mean(axis=1) for c in clusters])
    n_bags = max(1, int(n_bags))
    rng = np.random.default_rng(seed)
    bag_weights = np.empty((n_bags, k), dtype=float)
    for b in range(n_bags):
        rows = rng.integers(0, n, size=n)
        v = nnls_simplex_weights(super_cols[rows], y[rows], ridge=ridge)
        if v is None:
            diag['fallbacks'] += 1
            v = analytic_simplex_weights(super_cols[rows], y[rows])
        expanded = np.zeros(k, dtype=float)
        for share, members in zip(v, clusters):
            expanded[members] = share / len(members)
        bag_weights[b] = expanded
    diag['bag_weights'] = bag_weights
    weights = bag_weights.mean(axis=0)
    total = float(weights.sum())
    weights = weights / total if np.isfinite(total) and total > 0 else np.full(k, 1.0 / k)
    return weights, diag


def robust_simplex_weights(P: "np.ndarray", y: "np.ndarray", *, n_bags: int = 100,
                           cluster_corr: float = 0.97, seed: int = 0,
                           ridge: float = 0.0) -> "Tuple[List[float], dict]":
    """Bagged, correlation-clustered simplex weights (ver4-34, R2.a) - the
    REPORTED-weights estimator of ``weight_estimator='bagged_cluster'``.

    Steps: (1) columns of ``P`` correlated >= ``cluster_corr`` are merged into
    equal-weight super-columns (``correlation_clusters``); (2) ``n_bags``
    bootstrap resamples of the rows are drawn from
    ``np.random.default_rng(seed)``; (3) each is solved exactly by
    ``nnls_simplex_weights`` (``analytic_simplex_weights`` if that returns
    ``None``); (4) the bag solutions are averaged and every cluster's share is
    split equally among its members (``s_k = v_c / |c|``). The result is
    non-negative, sums to 1, is deterministic for a fixed ``seed``, and has NO
    equal-weight floor and no shrinkage: it is whatever the data support.

    Args: ``P`` (N, K) predictions, ``y`` (N,) targets, ``n_bags`` bootstrap
    count, ``cluster_corr`` correlation threshold, ``seed`` rng seed, ``ridge``
    forwarded to the solver (default 0.0).

    Returns ``(weights, diag)``; ``weights`` is a plain list of K floats and
    ``diag`` has ``clusters`` (list of column-index lists), ``bag_weights`` (a
    (B, K) array, used by the weight-aware freezing of R1.d), ``n_rows`` and
    ``fallbacks``. Fewer than 10 rows, or any non-finite value, prints one NOTE
    and returns the ``analytic_simplex_weights`` solution (``diag['note']``
    set). Never raises."""
    weights, diag = _robust_core(P, y, n_bags=n_bags, cluster_corr=cluster_corr,
                                 seed=seed, ridge=ridge)
    if diag.get('note'):
        print(f"[ensemble_regularization] NOTE: {diag['note']}. Never fatal.")
    return weights.tolist(), diag


def robust_weights_only(P: "np.ndarray", y: "np.ndarray", *, n_bags: int = 10,
                        cluster_corr: float = 0.97, seed: int = 0,
                        ridge: float = 0.0) -> "np.ndarray":
    """Quiet, weights-only form of ``robust_simplex_weights`` (ver4-34, R2.a).

    Why: it is the ``weight_fn`` that ``cross_fitted_combination`` calls once per
    training fold when the joint search SCORES a candidate; it must return just
    the (K,) weight array and must not print (it runs for every candidate and
    fold). Same arguments and maths as ``robust_simplex_weights`` (the scoring
    path uses a smaller ``n_bags``). Never raises."""
    weights, _diag = _robust_core(P, y, n_bags=n_bags, cluster_corr=cluster_corr,
                                  seed=seed, ridge=ridge)
    return weights


# --------------------------------------------------------------------------- #
# ver4-36, R1.b / R2.b - bootstrap out-of-bag helpers for the zero-fit gate
# --------------------------------------------------------------------------- #
#
# Why this section exists (design blueprint EasiGP_ver4-36, R1.b/R2.b): the joint
# search's "did tuning help?" gate used to REFIT the winner and the defaults on
# fresh inner splits (60-90 fits). Given the columns already fitted during the
# search, the same question can be asked at zero fits by bootstrap-bias-corrected
# selection (Tsamardinos et al., 2018, Mach. Learn. 107:1895): resample the
# scoring rows with replacement, select on the in-bag rows, score on the
# out-of-bag rows. These two helpers are the row-level machinery; the selection
# itself (which library column per model) lives in `models/joint_tuning.py`.

def bootstrap_oob_indices(n: int, n_boot: int, seed: int, min_oob: int = 3
                          ) -> "List[Tuple[np.ndarray, np.ndarray]]":
    """Draw ``n_boot`` bootstrap resamples of ``range(n)`` as ``(in_bag, out_of_bag)``.

    ``in_bag`` has ``n`` positions drawn WITH replacement (duplicates kept - it is
    meant to be used as ``P[in_bag]``); ``out_of_bag`` holds the positions that
    were never drawn, sorted. A draw with fewer than ``min_oob`` out-of-bag rows is
    skipped (a score on 1-2 rows is noise), so the list can be shorter than
    ``n_boot``; with n >= 20 this is rare (P(oob < 3) is tiny).

    Why a seeded ``default_rng``: the gate decision must be identical across
    processes and across runs for a fixed seed (blueprint A-1b.2, invariant I9).

    Args: ``n`` row count (>= 1); ``n_boot`` requested resamples; ``seed`` any int;
    ``min_oob`` minimum out-of-bag size. Returns a list of index-array pairs
    (positions, dtype int). ``n < 1`` or ``n_boot < 1`` returns ``[]``. Never raises."""
    n = int(n)
    n_boot = int(n_boot)
    if n < 1 or n_boot < 1:
        return []
    rng = np.random.default_rng(int(seed))
    every = np.arange(n)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_boot):
        in_bag = rng.integers(0, n, size=n)
        oob = np.setdiff1d(every, in_bag)
        if oob.size < int(min_oob):
            continue
        out.append((in_bag, oob))
    return out


def fast_oob_score(P: "np.ndarray", y: "np.ndarray", fit_rows: "np.ndarray",
                   score_rows: "np.ndarray", *, shrink_alpha: "Optional[float]" = None
                   ) -> "Optional[float]":
    """Ensemble score of ``P`` on ``score_rows`` at weights fitted on ``fit_rows``.

    Steps: exact simplex weights by ``nnls_simplex_weights`` on ``P[fit_rows]`` (the
    ~50-microsecond solver; ``analytic_simplex_weights`` if it returns ``None``),
    optionally shrunk toward equal weighting with ``apply_naive_shrinkage`` (the
    single source of that shrinkage), then ``pearson - mse / var`` of
    ``P[score_rows] @ w`` against ``y[score_rows]`` - the same 'ensemble_score'
    scale as ``joint_tuning._ensemble_score_of`` (re-stated here because this module
    cannot import ``joint_tuning``).

    With ``fit_rows == score_rows`` it is the in-sample score (used inside a
    bootstrap to SELECT); with ``fit_rows = in_bag`` and ``score_rows = out_of_bag``
    it is the out-of-bag score (used to JUDGE).

    Args: ``P`` (N, K); ``y`` (N,); ``fit_rows`` / ``score_rows`` integer positions
    (``fit_rows`` may contain duplicates); ``shrink_alpha`` None = no shrinkage,
    else the ``apply_naive_shrinkage`` alpha. Returns a float, or ``None`` when the
    score cannot be formed (fewer than 2 score rows, no variance in the targets or
    in the combination, non-finite values). Never raises for well-shaped input."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    fit_rows = np.asarray(fit_rows, dtype=int)
    score_rows = np.asarray(score_rows, dtype=int)
    if P.ndim != 2 or score_rows.size < 2 or fit_rows.size < 1:
        return None
    w = nnls_simplex_weights(P[fit_rows], y[fit_rows])
    if w is None:
        w = np.asarray(analytic_simplex_weights(P[fit_rows], y[fit_rows]), dtype=float)
    if shrink_alpha is not None:
        w = np.asarray(apply_naive_shrinkage(w, shrink_alpha), dtype=float)
    combined = P[score_rows] @ w
    actual = y[score_rows]
    var = float(np.var(actual))
    sd = float(np.std(combined)) * float(np.std(actual))
    if not np.isfinite(var) or var <= 0.0 or not np.isfinite(sd) or sd <= 0.0:
        return None
    mse = float(np.mean((combined - actual) ** 2))
    pearson = float(np.mean((combined - combined.mean()) * (actual - actual.mean())) / sd)
    value = pearson - mse / var
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- #
# ver4-36, R2.c - bagged ensemble selection (Caruana et al. 2004 / 2006)
# --------------------------------------------------------------------------- #
#
# Why this section exists (blueprint E3 / R2.c): exact simplex weights on ~50 rows
# of near-collinear columns jump to simplex vertices (the joint row collapsed onto
# the RKHS column). Greedy forward selection WITH replacement is a Frank-Wolfe
# path on the simplex with step 1/t, so stopping after T steps regularises the
# fit; bagging rows and models then smooths what is left. No equal-weight floor,
# no shrinkage and no selection of the naive ensemble (requirement R2).

def _es_bags(P: "np.ndarray", y: "np.ndarray", rows: "np.ndarray", mask: "np.ndarray",
             steps: int) -> "np.ndarray":
    """All ensemble-selection bags at once: a (B, K) array of per-bag weight vectors.

    Why vectorised across bags: the estimator runs inside every candidate score (once
    per cross-fit fold), so a per-bag Python loop of ``steps`` numpy calls dominated
    the cost. Here every step is a handful of numpy operations on (B, n, K) arrays.

    Per bag: greedy forward selection WITH replacement over ``steps`` steps. Sorted
    initialisation: the first pick is the column allowed by ``mask`` with the lowest
    MSE on the bag's rows. Each later step t adds the allowed column that minimises
    the MSE of the running average ``(s * (t-1) + P[:, k]) / t``. Ties go to the
    lowest column position (``argmin``), so the result is deterministic. The weight of
    a column is its pick count / ``steps``, so every row of the result sums to 1.

    Args: ``P`` (N, K); ``y`` (N,); ``rows`` (B, n) integer row positions per bag
    (duplicates allowed); ``mask`` (B, K) bool - the models each bag may use (at least
    one True per row); ``steps`` >= 1."""
    n_bags, n_rows = rows.shape
    k_total = P.shape[1]
    Pb = P[rows]                                   # (B, n, K)
    yb = y[rows]                                   # (B, n)
    bag_ids = np.arange(n_bags)
    counts = np.zeros((n_bags, k_total), dtype=float)
    mse = np.where(mask, np.mean((Pb - yb[:, :, None]) ** 2, axis=1), np.inf)
    pick = np.argmin(mse, axis=1)
    counts[bag_ids, pick] = 1.0
    running = Pb[bag_ids, :, pick]                 # (B, n)
    for t in range(2, int(steps) + 1):
        candidate = (running[:, :, None] * (t - 1) + Pb) / t
        mse = np.where(mask, np.mean((candidate - yb[:, :, None]) ** 2, axis=1), np.inf)
        pick = np.argmin(mse, axis=1)
        counts[bag_ids, pick] += 1.0
        running = (running * (t - 1) + Pb[bag_ids, :, pick]) / t
    return counts / float(steps)


def _es_core(P: "np.ndarray", y: "np.ndarray", *, n_bags: int, steps: int,
             model_fraction: float, seed: int) -> "Tuple[np.ndarray, dict]":
    """Shared body of ``ensemble_selection_weights`` / ``..._only`` (never prints).

    Returns ``(weights (K,), diag)``. ``diag`` keys: ``bag_weights`` ((B, K)),
    ``steps`` (the T actually used), ``n_rows``, ``fallbacks`` (0 or 1: whether the
    analytic solver replaced the estimator because the input was unusable) and
    ``note`` (only on that fallback)."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    n = int(P.shape[0]) if P.ndim == 2 else 0
    k = int(P.shape[1]) if P.ndim == 2 else 0
    diag = {'bag_weights': None, 'steps': 0, 'n_rows': n, 'fallbacks': 0}
    usable = (P.ndim == 2 and n >= _ROBUST_MIN_ROWS and k >= 1 and y.ndim == 1
              and y.shape[0] == n and np.all(np.isfinite(P)) and np.all(np.isfinite(y)))
    if not usable:
        w = analytic_simplex_weights(P, y) if (P.ndim == 2 and k >= 1) else np.zeros(0)
        diag['note'] = (f"ensemble_selection_weights needs >= {_ROBUST_MIN_ROWS} finite rows and a "
                        f"matching target (got {n} row(s), {k} column(s)) - using "
                        f"analytic_simplex_weights instead")
        diag['fallbacks'] = 1
        diag['bag_weights'] = np.asarray(w, dtype=float).reshape(1, -1)
        return np.asarray(w, dtype=float), diag
    n_bags = max(1, int(n_bags))
    steps = int(steps) if int(steps or 0) > 0 else 4 * k
    steps = max(steps, 1)
    n_cols = k if k <= 2 else int(min(k, max(2, round(float(model_fraction) * k))))
    rng = np.random.default_rng(int(seed))
    rows = rng.integers(0, n, size=(n_bags, n))
    # Each bag may use `n_cols` randomly chosen models: the n_cols smallest of k
    # uniform draws per bag.
    ranks = np.argsort(np.argsort(rng.random((n_bags, k)), axis=1), axis=1)
    mask = ranks < n_cols
    bag_weights = _es_bags(P, y, rows, mask, steps)
    weights = bag_weights.mean(axis=0)
    total = float(weights.sum())
    weights = weights / total if np.isfinite(total) and total > 0 else np.full(k, 1.0 / k)
    diag['bag_weights'] = bag_weights
    diag['steps'] = steps
    return weights, diag


def ensemble_selection_weights(P: "np.ndarray", y: "np.ndarray", *, n_bags: int = 50,
                               steps: int = 0, model_fraction: float = 0.5, seed: int = 0,
                               cluster_corr: float = 0.97, ridge: float = 0.0
                               ) -> "Tuple[List[float], dict]":
    """Bagged ensemble-selection weights (ver4-36, R2.c) - the REPORTED-weights form of
    ``weight_estimator='ensemble_selection'``.

    Each of ``n_bags`` bags resamples the rows with replacement and draws a random
    subset of the models (``model_fraction`` of them, at least 2); inside the bag the
    weights are the pick counts of greedy forward selection with replacement over
    ``steps`` steps (``steps <= 0`` means ``4 * K``), starting from the best single
    model (``_es_bag``). The returned weights are the mean over bags, so they are
    non-negative and sum to 1; every bag's weight is a multiple of ``1 / steps``.
    Deterministic for a fixed ``seed``. No equal-weight floor and no shrinkage.

    ``cluster_corr`` and ``ridge`` are accepted only so that this function and
    ``robust_simplex_weights`` can be called through the same registry
    (``ROBUST_WEIGHT_ESTIMATORS``) with the same keywords; they are ignored here.

    Args: ``P`` (N, K) predictions; ``y`` (N,) targets; the rest as above.
    Returns ``(weights, diag)`` - ``weights`` a list of K floats, ``diag`` as in
    ``_es_core`` (``bag_weights`` is what the weight-aware freezing of R1.d reads).
    Fewer than 10 rows (or non-finite values) prints one NOTE and returns the
    ``analytic_simplex_weights`` solution. Never raises."""
    weights, diag = _es_core(P, y, n_bags=n_bags, steps=steps,
                             model_fraction=model_fraction, seed=seed)
    if diag.get('note'):
        print(f"[ensemble_regularization] NOTE: {diag['note']}. Never fatal.")
    return weights.tolist(), diag


def ensemble_selection_weights_only(P: "np.ndarray", y: "np.ndarray", *, n_bags: int = 10,
                                    steps: int = 0, model_fraction: float = 0.5, seed: int = 0,
                                    cluster_corr: float = 0.97, ridge: float = 0.0
                                    ) -> "np.ndarray":
    """Quiet, weights-only form of ``ensemble_selection_weights`` (ver4-36, R2.c): the
    ``weight_fn`` that ``cross_fitted_combination`` calls once per training fold when the
    joint search SCORES a candidate. Returns just the (K,) array and never prints.
    Same arguments and maths (the scoring path uses a smaller ``n_bags``). Never raises."""
    weights, _diag = _es_core(P, y, n_bags=n_bags, steps=steps,
                              model_fraction=model_fraction, seed=seed)
    return weights


#: ver4-36, R2.c: ONE registry of the robust weight estimators the joint search
#: knows (I3). ``name -> (full function returning (weights, diag), quiet function
#: returning weights)``. Both functions of an entry take
#: ``(P, y, *, n_bags, cluster_corr, seed, ...)`` keywords. ``joint_tuning`` asks
#: ``is_robust_estimator(name)`` instead of comparing against one string.
ROBUST_WEIGHT_ESTIMATORS = {
    'bagged_cluster': (robust_simplex_weights, robust_weights_only),
    'ensemble_selection': (ensemble_selection_weights, ensemble_selection_weights_only),
}


def is_robust_estimator(name) -> bool:
    """True when ``name`` is a key of ``ROBUST_WEIGHT_ESTIMATORS`` (a bagged,
    cross-fitted estimator that replaces the legacy weight-collapse protection).
    ``'legacy'``, ``None`` and unknown names are False. Never raises."""
    return name in ROBUST_WEIGHT_ESTIMATORS


# --------------------------------------------------------------------------- #
# ver4-40, R2 - candidate weight vectors for the weight-first joint search
# --------------------------------------------------------------------------- #

def weight_candidates(P: "np.ndarray", y: "np.ndarray", *, rules, es_settings=None,
                      dedup_tol: float = 0.05, max_n: int = 6, seed: int = 0, dead=()
                      ) -> "List[Tuple[str, np.ndarray]]":
    """A small, de-duplicated set of FIXED ensemble weight vectors, built once from the
    prediction columns ``P`` of a task (the weight-first joint search of ver4-40, R2).

    Why: today every hyperparameter candidate re-solves K-1 continuous weights on 50-150
    scoring rows, so the best of hundreds of candidates is partly the best weight fit to
    noise (a winner's curse). Fixing a handful of weight vectors first leaves the later
    search only log(C) bits of selection freedom per candidate vector instead of K-1
    continuous ones. This function is the single place (I3) that turns a rule list into
    those vectors; scoring stays in ``joint_tuning._combine_score``.

    Rule grammar (one entry of ``rules`` each, in priority order):
      ``'uniform'``      equal weights;
      ``'analytic'``     ``analytic_simplex_weights(P, y)`` (non-negative, sum 1);
      ``'shrunk:<a>'``   ``apply_naive_shrinkage(analytic, a)`` with ``0 <= a <= 1``;
      ``'es'``           bagged ensemble-selection weights (``es_settings`` keys
                         ``n_bags``, ``steps``, ``model_fraction``, ``cluster_corr``);
      a sequence of K floats   an explicit vector, named ``'custom:<i>'`` (``i`` = its
                         position in ``rules``); it is clipped at 0 and renormalised.

    ``dead`` lists the positions of columns that carry no information (constant); they
    get weight 0 in every vector and the rest is renormalised, so a dead model can never
    receive weight by construction. A vector whose L1 distance to an EARLIER kept vector
    is below ``dedup_tol`` is dropped (rule order decides which of two near-identical
    vectors survives); at most ``max_n`` vectors are kept.

    Args: ``P`` (n, K) scoring-row predictions; ``y`` (n,) targets; ``rules`` the list
    above; ``es_settings`` optional dict; ``dedup_tol`` L1 threshold (0 keeps everything);
    ``max_n`` cap (>= 1); ``seed`` seeds the ES bags; ``dead`` positions to zero.
    Returns ``[(name, w), ...]`` with every ``w`` a float array of length K, ``w >= 0``,
    ``sum(w) == 1``; ``[]`` when every column is dead or ``P`` has no rows. Raises
    ``ValueError`` for a malformed rule (a configuration error, reported with the grammar)
    and never for numerical trouble - a failing estimator falls back to the uniform
    vector with one NOTE."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float)
    if P.ndim != 2 or P.shape[0] < 1 or P.shape[1] < 1:
        return []
    K = P.shape[1]
    dead_set = {int(i) for i in dead if 0 <= int(i) < K}
    live = [j for j in range(K) if j not in dead_set]
    if not live:
        return []
    es = dict(es_settings or {})
    grammar = ("'uniform', 'analytic', 'shrunk:<alpha in [0,1]>', 'es', or a list of "
               f"{K} floats")

    def _embed(w_live) -> "np.ndarray":
        w = np.zeros(K, dtype=float)
        w[live] = np.clip(np.asarray(w_live, dtype=float), 0.0, None)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            w[:] = 0.0
            w[live] = 1.0 / len(live)
            return w
        return w / total

    P_live, uniform = P[:, live], np.full(len(live), 1.0 / len(live))
    analytic = None

    def _analytic():
        nonlocal analytic
        if analytic is None:
            if len(live) == 1 or P.shape[0] < 3:
                analytic = uniform
            else:
                try:
                    analytic = np.asarray(analytic_simplex_weights(P_live, y), dtype=float)
                except Exception as exc:
                    print(f"[ensemble_regularization] NOTE: weight_candidates: the analytic "
                          f"weights failed ({type(exc).__name__}: {exc}) - the uniform vector "
                          f"is used in their place.")
                    analytic = uniform
        return analytic

    out: "List[Tuple[str, np.ndarray]]" = []
    for pos, rule in enumerate(rules):
        if isinstance(rule, str):
            token = rule.strip().lower()
            if token == 'uniform':
                name, w_live = 'uniform', uniform
            elif token == 'analytic':
                name, w_live = 'analytic', _analytic()
            elif token.startswith('shrunk:'):
                try:
                    alpha = float(token.split(':', 1)[1])
                except ValueError:
                    raise ValueError(f"weight_candidates: bad rule {rule!r}; expected {grammar}.")
                if not (0.0 <= alpha <= 1.0):
                    raise ValueError(f"weight_candidates: shrinkage in {rule!r} is outside "
                                     f"[0, 1]; expected {grammar}.")
                name = f"shrunk:{alpha:g}"
                w_live = np.asarray(apply_naive_shrinkage(_analytic(), alpha), dtype=float)
            elif token == 'es':
                name = 'es'
                try:
                    w_live = np.asarray(ensemble_selection_weights_only(
                        P_live, y, n_bags=int(es.get('n_bags', 10)),
                        steps=int(es.get('steps', 0) or 0),
                        model_fraction=float(es.get('model_fraction', 0.5)),
                        seed=int(seed), cluster_corr=float(es.get('cluster_corr', 0.97))),
                        dtype=float)
                except Exception as exc:
                    print(f"[ensemble_regularization] NOTE: weight_candidates: the "
                          f"ensemble-selection weights failed ({type(exc).__name__}: {exc}) "
                          f"- the uniform vector is used in their place.")
                    w_live = uniform
            else:
                raise ValueError(f"weight_candidates: unknown rule {rule!r}; expected {grammar}.")
        else:
            try:
                vec = np.asarray(list(rule), dtype=float)
            except (TypeError, ValueError):
                raise ValueError(f"weight_candidates: bad rule {rule!r}; expected {grammar}.")
            if vec.shape != (K,) or not np.all(np.isfinite(vec)):
                raise ValueError(f"weight_candidates: custom rule at position {pos} has "
                                 f"{vec.size} value(s) (or a non-finite one); expected {K} "
                                 f"finite floats.")
            name, w_live = f"custom:{pos}", vec[live]
        w = _embed(w_live)
        if any(float(np.abs(w - kept).sum()) < float(dedup_tol) for _n, kept in out):
            continue
        out.append((name, w))
        if len(out) >= max(1, int(max_n)):
            break
    return out


# --------------------------------------------------------------------------- #
# ver4-42, R1 - cross-validated shrinkage path (weights for `weight_first_cv`)
# --------------------------------------------------------------------------- #

def shrinkage_path_weights(P: "np.ndarray", y: "np.ndarray", *, direction: str = 'analytic',
                           alphas: "Optional[Sequence[float]]" = None, rule: str = 'min',
                           n_folds: int = 5, n_repeats: int = 4, seed: int = 0,
                           dead: Iterable[int] = (), es_settings: Optional[dict] = None,
                           final: str = 'refit', prior: "Optional[np.ndarray]" = None
                           ) -> "Tuple[np.ndarray, float, dict]":
    """Pick ONE fixed ensemble weight vector from the uniform -> ``direction`` shrinkage path
    by repeated K-fold cross-validation over the rows of ``P`` (ver4-42, R1, stage S1 of
    ``search_strategy='weight_first_cv'``).

    Why: the weight-first search of ver4-40/41 estimated the K-1 free weights from the
    ~30-80 validation rows only (blueprint RC1). This function is called on the
    out-of-fold columns of ALL train+valid rows (4-6x more rows) and answers one question:
    how far from equal weighting do the fitted weights still generalise? The level
    ``alpha`` of ``w(alpha) = (1 - alpha) * uniform + alpha * direction_w`` is chosen on
    held-out squared error, so ``alpha* = 0`` (equal weights) is returned only when fitted
    weights do not generalise - never by a parsimony margin on top of noise (RC3).

    Args:
      P: (n, K) out-of-fold prediction columns; y: (n,) targets (same rows).
      direction: ``'analytic'`` (``analytic_simplex_weights`` - the loop is
        ``cv_shrunk_simplex_weights``, one authority, I3) or ``'es'`` (bagged ensemble
        selection, same CV loop with ``ensemble_selection_weights_only`` in the fold).
      alphas: candidate shrinkage levels in [0, 1] (0 is always added); default
        0, .1, ... 1.
      rule: ``'min'`` = the alpha with the lowest CV loss; ``'one_se'`` = the smallest alpha
        within one standard error of the best (``'analytic'`` only; ``'es'`` uses its own
        paired-SE loop with the same meaning).
      n_folds, n_repeats, seed: the CV loop (the seed makes it deterministic, I9).
      dead: positions of constant columns; they get weight 0 and are left out of the fit.
      es_settings: optional dict (``n_bags``, ``steps``, ``model_fraction``,
        ``cluster_corr``) for ``direction='es'``.
      final: ver4-44, R3 - how the DEPLOYED direction vector is formed. ``'refit'``
        (default, ver4-42 behaviour exactly) fits the direction once on all rows.
        ``'bagged'`` uses the mean of the in-fold direction vectors that the CV loop
        above already computes for every (repeat, training fold) - zero extra fits. It
        averages estimates of the same target, so it lowers the variance of the vector
        without pulling it towards equal weights (alpha, chosen on held-out loss, is the
        only thing that does that). An unknown value is reported in ``diag['note']`` and
        treated as ``'refit'``.
      prior: ver4-94, R3 - the shrinkage target over the K columns (``None`` = uniform, byte
        for byte the earlier behaviour). Used with ``direction='analytic'`` only (it is
        ignored, with a note, for ``'es'``). The weights of ``dead`` columns are removed and
        the rest renormalised. Every fallback below (too few rows, a failed path, ...)
        returns this target, not equal weights over all K columns: for the pooled Super
        learner that is "own columns equal, pooled columns 0". An unusable prior is
        ignored with a note (see ``cv_shrunk_simplex_weights``).

    Returns ``(w, alpha_star, diag)``: ``w`` is a float array of length K, ``w >= 0``,
    ``sum(w) == 1``; ``diag`` has ``alphas``, ``cv_mse`` ({alpha: loss}), ``best_alpha``,
    ``chosen_alpha``, ``direction_w`` (K floats; with ``final='bagged'`` the bagged
    direction), ``n_rows``, ``note`` (only when a fallback ran) and, once the CV loop ran
    (ver4-44), ``final`` (the form actually used), ``fold_spread`` (mean L1 distance of the
    in-fold direction vectors to the deployed direction - a per-task instability measure,
    0 = every fold agrees) and ``fold_direction_mean`` (K floats, the bagged direction;
    always reported). With too few rows (or non-finite input) it returns uniform weights,
    alpha 0 and a note. Never raises."""
    P = np.asarray(P, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = (P.shape if P.ndim == 2 else (0, 1))
    k = max(int(k), 1)
    dead_set = {int(i) for i in dead if 0 <= int(i) < k}
    live = [j for j in range(k) if j not in dead_set]
    diag: dict = {'n_rows': int(n), 'direction': str(direction), 'rule': str(rule),
                  'alphas': [], 'cv_mse': {}, 'best_alpha': 0.0, 'chosen_alpha': 0.0}

    def _embed(w_live) -> "np.ndarray":
        w = np.zeros(k, dtype=float)
        if live:
            w[live] = np.clip(np.asarray(w_live, dtype=float), 0.0, None)
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0.0:
            w[:] = 0.0
            w[live if live else slice(None)] = 1.0 / (len(live) if live else k)
            return w
        return w / total

    prior_live = np.ones(len(live), dtype=float) / max(len(live), 1)
    if prior is not None:
        prior_vec = _valid_prior(prior, k)
        if direction != 'analytic':
            diag['note'] = "prior ignored: it applies to direction='analytic' only"
        elif prior_vec is None:
            diag['note'] = (f"prior ignored: it must be {k} finite, non-negative numbers with a "
                            f"positive sum - uniform is the shrinkage target")
        elif live and float(prior_vec[live].sum()) > 0.0:
            prior_live = prior_vec[live] / float(prior_vec[live].sum())
            diag['prior'] = prior_vec.tolist()
        else:
            diag['note'] = "prior ignored: it has no weight on any live column"
    # without a prior the original expressions are kept, so the result stays bitwise identical
    uniform = _embed(prior_live if 'prior' in diag else np.ones(len(live)))
    diag['direction_w'] = uniform.tolist()
    if P.ndim != 2 or n == 0 or y.size != n:
        diag['note'] = 'no rows (or row mismatch) - uniform weights'
        return uniform, 0.0, diag
    if len(live) <= 1:
        diag['note'] = 'fewer than two live columns - nothing to weight'
        return uniform, 0.0, diag
    if direction not in ('analytic', 'es') or rule not in ('min', 'one_se'):
        diag['note'] = (f"unknown direction {direction!r} or rule {rule!r} (expected "
                        f"'analytic'|'es' and 'min'|'one_se') - uniform weights")
        return uniform, 0.0, diag
    n_folds = int(max(2, min(int(n_folds), n)))
    if not (np.all(np.isfinite(P[:, live])) and np.all(np.isfinite(y))):
        diag['note'] = 'non-finite values in the columns or targets - uniform weights'
        return uniform, 0.0, diag
    if n < _cv_min_rows(n_folds):
        diag['note'] = (f'too few rows to cross-validate ({n} < {_cv_min_rows(n_folds)}) - '
                        f'uniform weights')
        return uniform, 0.0, diag
    P_live = P[:, live]
    grid = ([round(0.1 * i, 10) for i in range(11)] if alphas is None
            else [float(a) for a in alphas])
    final = 'refit' if final is None else str(final)
    if final not in ('refit', 'bagged'):
        diag['note'] = (f"unknown final {final!r} (expected 'refit'|'bagged') - the direction "
                        f"is refitted on all rows")
        final = 'refit'
    fold_vecs: List[np.ndarray] = []     # in-fold direction vectors (live space), one per fit
    try:
        if direction == 'analytic':
            w_live, a_star, d = cv_shrunk_simplex_weights(
                P_live, y, alphas=grid, n_folds=n_folds, n_repeats=int(n_repeats),
                seed=int(seed), one_se=(rule == 'one_se'), return_fold_weights=True,
                prior=(prior_live if 'prior' in diag else None))
            dir_live = analytic_simplex_weights(P_live, y)
            fold_vecs = [np.asarray(v, dtype=float) for v in d.get('fold_weights', [])]
            diag.update({'alphas': list(d.get('alphas', [])), 'cv_mse': dict(d.get('cv_mse', {})),
                         'best_alpha': float(d.get('best_alpha', a_star)),
                         'chosen_alpha': float(a_star)})
            if 'note' in d:
                diag['note'] = d['note']
        else:
            es = dict(es_settings or {})

            def _es(Pm, ym, s):
                return ensemble_selection_weights_only(
                    Pm, ym, n_bags=int(es.get('n_bags', 10)), steps=int(es.get('steps', 0) or 0),
                    model_fraction=float(es.get('model_fraction', 0.5)), seed=int(s),
                    cluster_corr=float(es.get('cluster_corr', 0.97)))

            alist = sorted({float(np.clip(a, 0.0, 1.0)) for a in grid} | {0.0})
            equal = np.full(len(live), 1.0 / len(live))
            loss = np.zeros((len(alist), n), dtype=float)
            rng = np.random.default_rng(int(seed))
            reps = max(1, int(n_repeats))
            for rep in range(reps):
                for held in _fold_indices(n, n_folds, rng):
                    mask = np.ones(n, dtype=bool)
                    mask[held] = False
                    wf = np.asarray(_es(P_live[mask], y[mask], int(seed) + rep), dtype=float)
                    fold_vecs.append(wf)
                    for ai, a in enumerate(alist):
                        pred = P_live[held] @ (a * wf + (1.0 - a) * equal)
                        loss[ai, held] += (pred - y[held]) ** 2
            loss /= reps
            mean_loss = loss.mean(axis=1)
            best_i = int(np.argmin(mean_loss))
            chosen_i = best_i
            if rule == 'one_se':
                for ai in range(len(alist)):
                    dd = loss[ai] - loss[best_i]
                    se = float(dd.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    if float(dd.mean()) <= se:
                        chosen_i = ai
                        break
            a_star = alist[chosen_i]
            dir_live = np.asarray(_es(P_live, y, int(seed)), dtype=float)
            w_live = a_star * dir_live + (1.0 - a_star) * equal
            diag.update({'alphas': alist,
                         'cv_mse': {f'{a:g}': float(m) for a, m in zip(alist, mean_loss)},
                         'best_alpha': alist[best_i], 'chosen_alpha': float(a_star)})
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        diag['note'] = (f"the shrinkage path failed ({type(exc).__name__}: {exc}) - uniform "
                        f"weights")
        diag['chosen_alpha'] = 0.0
        return uniform, 0.0, diag
    if fold_vecs:
        # ver4-44, R3: the bagged direction (mean of the in-fold vectors, renormalised to the
        # simplex) is always reported; it is DEPLOYED only with final='bagged'. The deployed
        # shrunk vector is rebuilt around it with the SAME alpha* (chosen above on held-out loss).
        fold_mat = np.vstack(fold_vecs)
        bag_dir = np.clip(fold_mat.mean(axis=0), 0.0, None)
        bag_total = float(bag_dir.sum())
        bag_dir = (bag_dir / bag_total) if (np.isfinite(bag_total) and bag_total > 0.0) \
            else np.full(len(live), 1.0 / len(live))
        if final == 'bagged':
            dir_live = bag_dir
            w_live = float(a_star) * bag_dir + (1.0 - float(a_star)) * (
                prior_live if 'prior' in diag else np.full(len(live), 1.0 / len(live)))
        spread = float(np.mean(np.abs(fold_mat - np.asarray(dir_live, dtype=float)).sum(axis=1)))
        diag.update({'final': final, 'fold_spread': spread,
                     'fold_direction_mean': _embed(bag_dir).tolist()})
    elif final == 'bagged':
        diag['note'] = ("final='bagged' needs the in-fold vectors of the CV loop, which were not "
                        "produced - the direction is refitted on all rows")
        diag['final'] = 'refit'
    diag['direction_w'] = _embed(dir_live).tolist()
    return _embed(w_live), float(a_star), diag
