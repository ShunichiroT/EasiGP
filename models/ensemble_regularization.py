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

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
# Already a project dependency (used directly, top-level, by
# models/hyperparameter_tuning.py) - no lazy-import guard needed (I12: not
# a NEW dependency).
from scipy.optimize import minimize


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
    """
    n = len(weights)
    if n == 0 or not np.isfinite(w_sum) or w_sum == 0:
        return float('inf')
    shares = np.asarray(weights, dtype=float) / w_sum

    if pred_matrix is not None and actual is not None:
        if pred_matrix.shape[0] == 0:
            return float('inf')
        combined = pred_matrix @ shares
        return float(np.mean((combined - actual) ** 2))

    combined = np.zeros(data_valid.shape[0], dtype=float)
    for i, m in enumerate(model_selected):
        combined = combined + data_valid[m].to_numpy(dtype=float) * shares[i]
    actual_local = data_valid['actual'].to_numpy(dtype=float)
    if combined.shape[0] == 0:
        return float('inf')
    return float(np.mean((combined - actual_local) ** 2))


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
                              ridge: float = 1e-6) -> "np.ndarray":
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
        gram = pred_matrix.T @ pred_matrix + ridge * np.eye(k)
        target = pred_matrix.T @ actual
        n_rows = pred_matrix.shape[0]

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
        return float(np.mean((combined - actual) ** 2))

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
