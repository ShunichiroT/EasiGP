"""
blue_weights.py
================
ver4-25, R8 (design blueprint SS2, Stage 6) - a Layer 3d shared authority,
zero project imports (NumPy/SciPy only), mirroring the discipline
`model_registry.py`/`circos_geometry.py`/`asreml_syntax.py` already follow.

Governing principle P8 (blueprint SS2, R8): for a fair comparison, every
model and every ensemble uses the SAME BLUE response vector and the SAME
weighting information. This module is the single place that:

  1. turns a task's per-individual standard errors (`diagonal` mode) or a
     first-stage BLUE variance-covariance matrix (`full` mode) into one
     shared FITTING weight vector (`w_fit`, aligned to `id_train`) handed
     identically to every model this run selects;
  2. turns the same information into per-split SCORING weights (a plain
     vector `w` for `diagonal` mode, a dense matrix `W` for `full` mode)
     used by every consumer of `pearson`/`mse` - Metric.csv, tuning valid
     scores, ensemble-weight fitting, and the DPT summary.

Weighted-metric definitions (blueprint `jkae250.pdf` p.3 method, SS2 R8):

    m_W(a)      = 1'W a / 1'W 1
    cov_W(a,b)  = (a - m_W(a))' W (b - m_W(b)) / tr(W)
    pearson_W   = cov_W(y, yhat) / sqrt(cov_W(y,y) * cov_W(yhat,yhat))
    mse_W       = (y - yhat)' W (y - yhat) / tr(W)

`diagonal` mode sets `W = diag(w)` (reduces to the ordinary weighted
Pearson/weighted MSE); `W = I` recovers the unweighted metric exactly -
this module is only ever CALLED when weighting is on (mode != 'none');
`pipeline_utils.safe_regression_metrics()` keeps its own untouched,
byte-identical unweighted path for that case (I11).

This module never touches ASReml/rpy2/R - it consumes only plain NumPy
arrays and a plain dense covariance matrix (however that matrix reached
Python-side memory - `genomic_prediction.py`'s own Step 6b block decides
that). Imported (lazily, to avoid any import-order risk) by
`pipeline_utils.py`, `genomic_prediction.py`, the 17 model files, the
four weighted-ensemble methods + `models/ensemble.py`/
`ensemble_regularization.py`, and `diversity_summary.py`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np

VALID_MODES = ('diagonal', 'full')

# ver4-25, R8 (design blueprint SS2, Stage 6): the three columns
# `genomic_prediction.py`'s own central weighted-scoring recompute
# appends to a per-model `Metric.csv` row (SS6.3) whenever weighting is
# on for this run - single-authority names, so every caller that needs
# to either WRITE them (the four weighted-ensemble methods, mirroring
# that same central-recompute pattern for their own final test-score
# row) or EXCLUDE them (those same methods' own `weight_sample`, which
# is built by dropping known metric columns from a `record` row - see
# each of `models/{Linear_transformation,Nelder_Mead,Bayesian_
# optimisation,Analytic_least_squares}.py`'s own `weight_sample`
# construction) agrees on the exact strings, rather than re-typing them.
METRIC_WEIGHTING_EXTRA_COLUMNS = (
    'Pearson correlation (unweighted)', 'MSE (unweighted)', 'metric_weighting',
)


@dataclass(frozen=True)
class BlueWeightContext:
    """Everything every model/ensemble needs to apply the SAME BLUE-
    reliability weighting for one task (one population x phenotype x
    ratio x replicate scenario) - see module docstring, principle P8.

    Attributes
    ----------
    mode : `'diagonal'` or `'full'` (never `'none'` - a caller with
        weighting off never constructs one of these at all; see
        `genomic_prediction.py`'s own Step 6b call site).
    ids_train/ids_valid/ids_test : this task's own split membership, in
        the SAME row order as `train`/`valid`/`test` (architecture doc
        S4.2 - sample identity is carried positionally, outside the
        model frames) - `w_fit`/`w_valid`/`w_test` are aligned to these
        tuples element-for-element.
    w_fit : shared FITTING weight vector, aligned to `ids_train` - handed
        identically to every model's own `sample_weight=`/`weights=`
        argument (diagonal: `1/se_i^2`, full: Smith et al. 2001
        `[V_bar^-1]_ii`; both mean-normalised to 1 over the train split).
    w_valid/w_test : diagonal-mode SCORING weights for the validation/
        test split (`None` in `full` mode, and `None` for `w_valid`
        whenever `ids_valid` is empty).
    W_valid/W_test : full-mode SCORING matrices (`V_bar^-1` of that
        split) - `None` in `diagonal` mode, and `None` for `W_valid`
        whenever `ids_valid` is empty.
    w_fit_cv : optional FITTING weight vector over the task's train+valid rows
        (`ids_train` then `ids_valid`, the row order of `joint_tuning._cv_frame()`),
        built only when `build_blue_weight_context(..., with_cv_weights=True)` (ver4-52:
        the cross-validated joint objective fits fold models on train+valid rows, and a
        validation row has no entry in `w_fit`). `None` otherwise - then the CV objective
        is not available under BLUE weighting. Use `cv_weight_view()` to get the context a
        CV fold fit should see.
    fingerprint : sha1 of `w_fit` (rounded to 10 significant digits before
        hashing, so it is stable across an equivalent re-derivation) -
        logged once per model call (AC8.6), so two tasks/runs that
        happened to fit the SAME weights are visibly identifiable as such
        in the run log without printing the whole vector.
    diagnostics : free-form dict - jitter actually used (full mode),
        condition number of the jittered covariance blocks, per-split n,
        min/max raw (pre-normalisation) weight, and how many individuals
        had a missing/non-finite SE that was imputed to the split's own
        mean weight (diagonal mode) - purely for logging, never
        consumed programmatically by any other module.
    """
    mode: str
    ids_train: Tuple[str, ...]
    ids_valid: Tuple[str, ...]
    ids_test: Tuple[str, ...]
    w_fit: "np.ndarray"
    w_valid: "Optional[np.ndarray]" = None
    w_test: "Optional[np.ndarray]" = None
    W_valid: "Optional[np.ndarray]" = None
    W_test: "Optional[np.ndarray]" = None
    fingerprint: str = ''
    diagnostics: dict = field(default_factory=dict)
    w_fit_cv: "Optional[np.ndarray]" = None


def _fingerprint(w: "np.ndarray") -> str:
    try:
        rounded = np.round(np.asarray(w, dtype=float), 10)
        return hashlib.sha1(rounded.tobytes()).hexdigest()[:12]
    except Exception:
        return 'unavailable'


def _diagonal_weights_for_split(
    ids: Sequence[str], se_by_id: Mapping[str, float], *, log_prefix: str, power: float = 1.0,
) -> Tuple["np.ndarray", dict]:
    """`w_i = 1/se_i^2`, mean-normalised to 1 over `ids`. A missing or
    non-finite/non-positive SE is imputed to the mean of the OTHER,
    resolvable raw weights in this same split (a neutral, "average
    uncertainty" stand-in - never fabricated as a preferential value) -
    or, if every SE in the split is unusable, falls back to uniform
    weighting (all 1s) for that split alone, with a diagnostic count so
    this is never silent (architecture doc SS18 discipline).

    `power` (ver4-25, R8 bugfix follow-up #2 - see `_soften_raw_weights()`
    docstring for the full rationale) is applied to the raw `1/se_i^2`
    values, AFTER imputation and BEFORE the mean-normalisation above, so
    an imputed entry is softened by the exact same rule as a resolved
    one. `power=1.0` (the default, and every pre-existing caller) is a
    strict no-op (I11) - `_soften_raw_weights()` special-cases it to skip
    the `np.power()` call entirely rather than relying on `x**1.0`
    returning `x` bit-for-bit."""
    n = len(ids)
    if n == 0:
        return np.zeros(0, dtype=float), {'n': 0, 'n_imputed': 0}
    raw = np.full(n, np.nan, dtype=float)
    for idx, _id in enumerate(ids):
        se = se_by_id.get(str(_id))
        try:
            se = float(se)
        except (TypeError, ValueError):
            se = float('nan')
        if np.isfinite(se) and se > 0:
            raw[idx] = 1.0 / (se * se)
    n_missing = int(np.sum(~np.isfinite(raw)))
    if n_missing == n:
        print(f"{log_prefix} NOTE: no usable standard error for any of {n} individual(s) in "
              f"this split - falling back to UNIFORM weighting for this split only.")
        raw[:] = 1.0
    elif n_missing > 0:
        fill = float(np.nanmean(raw))
        raw[~np.isfinite(raw)] = fill
        print(f"{log_prefix} NOTE: {n_missing}/{n} individual(s) in this split had no usable "
              f"standard error - imputed to this split's own mean raw weight ({fill:.4g}).")
    raw = _soften_raw_weights(raw, power, log_prefix=log_prefix)
    mean_raw = float(np.mean(raw))
    w = raw / mean_raw if mean_raw > 0 and np.isfinite(mean_raw) else np.ones(n, dtype=float)
    return w, {'n': n, 'n_imputed': n_missing, 'min_weight': float(np.min(w)) if n else None,
               'max_weight': float(np.max(w)) if n else None,
               'effective_n': _kish_effective_n(w)}


def _soften_raw_weights(raw: "np.ndarray", power: float, *, log_prefix: str) -> "np.ndarray":
    """ver4-25, R8 bugfix follow-up #2 - addresses the still-open question
    the ver4-25 Change Summary Part 3/Part 4 both raised and left
    unresolved: "is a lower reported accuracy under weighting expected or
    an error?" A second, independent review of this exact module (this
    session) confirmed the Part 4 diagnosis with a concrete mechanism -
    when raw `1/se^2` precision spans orders of magnitude (a common
    consequence of unbalanced replication across a trial), the weight
    MASS concentrates on a handful of individuals (`_kish_effective_n()`
    collapses toward 1), which:

      (a) starves rigid, parametric models (rrBLUP/BayesB/GBLUP/RKHS - all
          four fit via a single shared residual-variance rescaling, never a
          flexible per-split adjustment) of effective information from the
          rest of the split, since BGLR's `weights=` argument rescales
          each row's OWN residual variance rather than resampling rows
          the way a tree ensemble's `sample_weight` can absorb more
          gracefully;
      (b) makes the 'full'-mode dense GLS scoring matrix (see
          `_block_weights()` below) an even higher-variance quantity, since
          its own eigenvalue spread inherits the same concentration.

    This is a genuine, principled statistical remedy for that concentration
    - a variance-stabilising power transform of the raw precision weights,
    a standard technique for taming an extreme weight ratio in weighted
    regression (e.g. `power=0.5` uses `1/se` instead of `1/se^2` - inverse-
    SE weighting is itself a recognised, if gentler, alternative to inverse-
    variance weighting in the WLS literature) - NOT a change to what the
    weights mean (still "more reliable BLUEs count for more"), only to how
    extreme that counting is allowed to get. `power=1.0` (default) leaves
    every pre-existing call byte-identical (I11); a caller opts in via
    `ASREML['blue_weighting_smoothing_power']` (see
    `genomic_prediction.py`'s Step 6b call site) with a recommended range
    of `(0, 1]` - values above 1 would SHARPEN the concentration instead
    of softening it, which is never the intent of this parameter, so they
    are rejected rather than silently doing the opposite of what a caller
    almost certainly meant.

    Applied identically to `w_fit`/`w_valid`/`w_test` (this function is
    called from every `_diagonal_weights_for_split()` invocation) - the
    SAME softened precision is what every model fits AND what every
    consumer scores against, preserving principle P8 (one shared weighting
    story per task, not a fitting-only patch that would leave the reported
    metric's own instability untouched)."""
    if power == 1.0:
        return raw
    if not (isinstance(power, (int, float)) and np.isfinite(power) and 0.0 < power <= 1.0):
        raise ValueError(
            f"{log_prefix} blue_weighting_smoothing_power={power!r} must be a finite number in "
            f"(0, 1] (1.0 = no softening; e.g. 0.5 rescales 1/se^2 to 1/se) - a value above 1 "
            f"would sharpen weight concentration rather than soften it."
        )
    return np.power(raw, power)


def _kish_effective_n(w: "np.ndarray") -> "Optional[float]":
    """Kish's effective sample size, `(sum w)^2 / sum(w^2)` - a standard
    diagnostic for how much a set of weights concentrates on a handful of
    individuals (bugfix follow-up to the ver4-25 Change Summary Part 3's
    own unresolved question, "is a lower reported accuracy under
    weighting expected or an error?"). Equal to `n` for uniform weights,
    and shrinks toward 1 as the weight mass concentrates on fewer
    individuals - a SMALL `effective_n` relative to `n` (e.g. 7
    individuals with an effective_n of 2) is the concrete, actionable
    signal that a weighted metric computed over this split is likely to
    be noisy/unstable, independent of whether the weights themselves are
    "correct". `None` for an empty split (nothing to compute)."""
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        return None
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if not np.isfinite(s1) or not np.isfinite(s2) or s2 <= 0:
        return None
    return (s1 * s1) / s2


def _kish_effective_n_matrix(W: "np.ndarray") -> "Optional[float]":
    """The `W`-bilinear-form analogue of `_kish_effective_n()`:
    `trace(W)^2 / trace(W @ W)` - the same "effective number of
    observations" quantity applied to `W`'s own eigenvalues (equal to
    `_kish_effective_n()` of the eigenvalues, since both `trace(W)` and
    `trace(W @ W)` are eigenvalue-basis-independent) rather than to a
    plain weight vector - the 'full'-mode analogue needed because a
    dense GLS scoring matrix has no single per-individual weight to run
    the vector version on directly. `None` on any non-finite result
    (e.g. a degenerate all-zero matrix)."""
    W = np.asarray(W, dtype=float)
    if W.size == 0:
        return None
    trW = float(np.trace(W))
    trWW = float(np.trace(W @ W))
    if not np.isfinite(trW) or not np.isfinite(trWW) or trWW <= 0:
        return None
    return (trW * trW) / trWW


def _cholesky_inverse_with_jitter(
    V: "np.ndarray", *, log_prefix: str = '[blue_weights]',
) -> Tuple["np.ndarray", float]:
    """Dense inverse of a (scale-free, symmetric, nominally PSD)
    covariance block, via a jittered Cholesky feasibility check
    (blueprint SS2 R8: "Cholesky with jitter 1e-8*mean(diag) escalating
    x10 to 1e-4, logged; failure fatal"). Returns `(V_inv, jitter_used)`.

    Raises ValueError (never silently degrades - a covariance-derived
    weight matrix that isn't actually PSD after this much jitter is a
    genuine data problem, not something to paper over) if even the
    largest jitter fails."""
    n = V.shape[0]
    diag_mean = float(np.mean(np.diag(V)))
    base = diag_mean if diag_mean > 0 else 1.0
    last_exc = None
    for mult in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
        jitter = base * mult
        try:
            np.linalg.cholesky(V + jitter * np.eye(n))
            V_inv = np.linalg.inv(V + jitter * np.eye(n))
            return V_inv, jitter
        except np.linalg.LinAlgError as exc:
            last_exc = exc
            continue
    raise ValueError(
        f"{log_prefix} covariance block ({n}x{n}) is not positive-definite even after jitter "
        f"up to {base * 1e-4:.3g} (mean diag {diag_mean:.3g}) - refusing to invert it."
    ) from last_exc


def _submatrix(vcov: "np.ndarray", vcov_ids: Sequence[str], ids: Sequence[str]) -> "np.ndarray":
    """Row/column slice of the ONE full ASReml `vcov` matrix (aligned to
    `vcov_ids`) restricted to `ids`, in `ids`' own order - every id in
    `ids` must appear in `vcov_ids` (fatal otherwise: a split individual
    with no covariance entry means the first-stage fit and this weighting
    step disagree about the universe, which must never pass silently)."""
    index = {str(v): i for i, v in enumerate(vcov_ids)}
    try:
        positions = [index[str(_id)] for _id in ids]
    except KeyError as exc:
        raise ValueError(
            f"[blue_weights] id {exc.args[0]!r} has no entry in the supplied vcov matrix - "
            f"cannot build 'full' weights for this split."
        ) from exc
    if not positions:
        return np.zeros((0, 0), dtype=float)
    return vcov[np.ix_(positions, positions)]


def build_blue_weight_context(
    mode: str,
    ids_train: Sequence[str], ids_valid: Sequence[str], ids_test: Sequence[str],
    se_by_id: Optional[Mapping[str, float]] = None,
    vcov: "Optional[np.ndarray]" = None, vcov_ids: Optional[Sequence[str]] = None,
    *, max_dense_n: int = 20000, log_prefix: str = '[blue_weights]',
    smoothing_power: float = 1.0, full_shrinkage: float = 0.0,
    with_cv_weights: bool = False,
) -> "BlueWeightContext":
    """Build one task's `BlueWeightContext` - see that class's own
    docstring for every field. `mode` must be `'diagonal'` or `'full'`
    (the caller never calls this at all for `'none'`).

    `diagonal` mode consumes only `se_by_id` (`{id: standard_error}`,
    covering every id in `ids_train`/`ids_valid`/`ids_test` - a missing
    entry degrades per-split, see `_diagonal_weights_for_split()`).

    `full` mode consumes `vcov`/`vcov_ids` - the ONE dense first-stage
    BLUE covariance matrix ASReml's own `predict(..., vcov=TRUE)`
    returned for this task's whole universe (`id_train UNION id_valid
    UNION id_test`), aligned to `vcov_ids`. Each split's own block is
    sliced out, scaled to `V_bar = V / mean(diag(V))` (scale-free, per
    blueprint SS2 R8), then inverted (with jitter - see
    `_cholesky_inverse_with_jitter()`). Fitting weights are
    `diag(V_bar_tr^-1)`, mean-normalised; scoring matrices are
    `V_bar_va^-1`/`V_bar_te^-1` directly (used as GLS `W` - see
    `weighted_pearson()`/`weighted_mse()`).

    Raises ValueError for any block whose size exceeds `max_dense_n`
    (dense NxN inversion beyond that is not attempted at all - the
    caller is expected to catch this and disable weighting for that one
    task with a NOTE, not crash the whole run - see
    `genomic_prediction.py`'s own Step 6b call site) or for a genuinely
    non-invertible block (see `_cholesky_inverse_with_jitter()`).

    `smoothing_power`/`full_shrinkage` (ver4-25, R8 bugfix follow-up #2 -
    see `_soften_raw_weights()`/the `full_shrinkage` note on
    `_block_weights()` below for the full rationale) are the two opt-in
    remedies for weight-mass concentration - `smoothing_power` softens
    `diagonal` mode's raw `1/se^2` values, `full_shrinkage` blends `full`
    mode's covariance block toward its own diagonal before inverting.
    Both default to their I11 no-op value (`1.0`/`0.0`), so every
    pre-existing caller (including every caller that never heard of
    either parameter) reproduces its previous output exactly.

    `with_cv_weights` (ver4-52, default False = every earlier caller's exact output): also
    build `BlueWeightContext.w_fit_cv`, the fitting weights of the train+valid rows, for
    the cross-validated joint objective (`diagonal`: `1/se^2` of every train+valid
    individual; `full`: `diag` of the inverse of the train+valid covariance block, scale-free
    and mean-normalised exactly like `w_fit`). Never raises for this part: when the block
    cannot be built (no validation rows, larger than `max_dense_n`, not invertible)
    `w_fit_cv` stays `None` and the reason goes into `diagnostics['cv_rows']`, so the caller
    simply reports the CV objective as unavailable.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"{log_prefix} mode must be one of {VALID_MODES}, got {mode!r}.")
    if not (0.0 <= full_shrinkage <= 1.0):
        raise ValueError(
            f"{log_prefix} blue_weighting_full_shrinkage={full_shrinkage!r} must be in [0, 1] "
            f"(0.0 = no shrinkage; 1.0 = fully diagonal, i.e. equivalent to 'diagonal' mode's "
            f"own per-individual precision)."
        )

    ids_train = tuple(str(v) for v in ids_train)
    ids_valid = tuple(str(v) for v in ids_valid)
    ids_test = tuple(str(v) for v in ids_test)
    diagnostics: Dict[str, object] = {'mode': mode, 'smoothing_power': smoothing_power,
                                       'full_shrinkage': full_shrinkage}

    if mode == 'diagonal':
        se_by_id = se_by_id or {}
        w_fit, d_fit = _diagonal_weights_for_split(ids_train, se_by_id, log_prefix=log_prefix,
                                                    power=smoothing_power)
        w_valid, d_valid = (_diagonal_weights_for_split(ids_valid, se_by_id, log_prefix=log_prefix,
                                                         power=smoothing_power)
                            if ids_valid else (None, {'n': 0}))
        w_test, d_test = _diagonal_weights_for_split(ids_test, se_by_id, log_prefix=log_prefix,
                                                       power=smoothing_power)
        diagnostics.update({'train': d_fit, 'valid': d_valid, 'test': d_test})
        w_fit_cv = None
        if with_cv_weights:
            if ids_valid:
                w_fit_cv, d_cv = _diagonal_weights_for_split(
                    ids_train + ids_valid, se_by_id, log_prefix=f"{log_prefix} [cv rows]",
                    power=smoothing_power)
                diagnostics['cv_rows'] = d_cv
            else:
                diagnostics['cv_rows'] = {'n': 0, 'reason': 'no validation rows'}
        return BlueWeightContext(
            mode='diagonal', ids_train=ids_train, ids_valid=ids_valid, ids_test=ids_test,
            w_fit=w_fit, w_valid=w_valid, w_test=w_test, fingerprint=_fingerprint(w_fit),
            diagnostics=diagnostics, w_fit_cv=w_fit_cv,
        )

    # mode == 'full'
    if vcov is None or vcov_ids is None:
        raise ValueError(f"{log_prefix} mode='full' requires both vcov and vcov_ids.")
    for _name, _ids in (('train', ids_train), ('valid', ids_valid), ('test', ids_test)):
        if len(_ids) > max_dense_n:
            raise ValueError(
                f"{log_prefix} the {_name} split has {len(_ids)} individuals, exceeding "
                f"max_dense_n={max_dense_n} - dense 'full' weighting is not attempted at this "
                f"scale (see ASREML['blue_weighting_max_dense_n'])."
            )
    vcov = np.asarray(vcov, dtype=float)

    def _block_weights(ids):
        if not ids:
            return None, None, {'n': 0}
        V = _submatrix(vcov, vcov_ids, ids)
        # ver4-25, R8 bugfix follow-up #2: a Ledoit-Wolf-style convex
        # shrinkage of this split's own covariance block toward its
        # diagonal, BEFORE the scale-free normalisation/inversion below -
        # the direct analogue, for a dense GLS matrix, of what
        # `_soften_raw_weights()` does for a plain precision vector. A
        # small-sample covariance block's off-diagonal entries are its
        # noisiest components (each is estimated from the same handful of
        # individuals as the diagonal, but there are O(n^2) of them versus
        # O(n) diagonal entries), and it is exactly that off-diagonal
        # structure which concentrates `V_inv`'s eigenvalues onto a few
        # directions - shrinking it toward `diag(diag(V))` pulls
        # `_kish_effective_n_matrix(V_inv)` back up toward this split's own
        # `n` (full_shrinkage=1.0 recovers a purely diagonal - i.e.
        # 'diagonal'-mode-equivalent - precision structure; 0.0, the
        # default, is a strict I11 no-op).
        if full_shrinkage:
            V = (1.0 - full_shrinkage) * V + full_shrinkage * np.diag(np.diag(V))
        diag_mean = float(np.mean(np.diag(V)))
        if not np.isfinite(diag_mean) or diag_mean <= 0:
            raise ValueError(f"{log_prefix} covariance block has non-positive mean diagonal "
                              f"({diag_mean!r}) - cannot scale-free normalise it.")
        V_bar = V / diag_mean
        V_inv, jitter = _cholesky_inverse_with_jitter(V_bar, log_prefix=log_prefix)
        try:
            cond = float(np.linalg.cond(V_bar))
        except Exception:
            cond = float('nan')
        # ver4-25, R8 bugfix follow-up: the `V_bar_inv` computed here for
        # 'valid'/'test' is used DIRECTLY as the dense GLS scoring matrix
        # (see `scoring_for()`/`weighted_pearson()`/`weighted_mse()`) -
        # its own `_kish_effective_n_matrix()` is therefore the single
        # most useful stability diagnostic for THIS split specifically,
        # not just for the training weight vector below. A SMALL
        # effective_n here (e.g. 2 out of 7 test individuals) means the
        # weighted test-score is dominated by a couple of individuals -
        # exactly the condition under which a dense, test-only covariance
        # sub-block (inherently a small-sample, high-variance quantity)
        # can swing the weighted Pearson/MSE well away from the
        # unweighted one for reasons having nothing to do with whether
        # the underlying weighting is "correct".
        return V_bar, V_inv, {'n': len(ids), 'jitter': jitter, 'condition_number': cond,
                               'effective_n': _kish_effective_n_matrix(V_inv)}

    _, W_tr_inv, d_fit = _block_weights(ids_train)
    _, W_va_inv, d_valid = _block_weights(ids_valid)
    _, W_te_inv, d_test = _block_weights(ids_test)

    raw_fit = np.diag(W_tr_inv) if W_tr_inv is not None else np.zeros(0)
    mean_raw = float(np.mean(raw_fit)) if raw_fit.size else 1.0
    w_fit = raw_fit / mean_raw if mean_raw > 0 and np.isfinite(mean_raw) else raw_fit
    d_fit['min_weight'] = float(np.min(w_fit)) if w_fit.size else None
    d_fit['max_weight'] = float(np.max(w_fit)) if w_fit.size else None
    diagnostics.update({'train': d_fit, 'valid': d_valid, 'test': d_test})

    w_fit_cv = None
    if with_cv_weights:
        ids_tv = ids_train + ids_valid
        if not ids_valid:
            diagnostics['cv_rows'] = {'n': 0, 'reason': 'no validation rows'}
        elif len(ids_tv) > max_dense_n:
            diagnostics['cv_rows'] = {'n': len(ids_tv), 'reason': (
                f"train+valid block of {len(ids_tv)} individuals exceeds max_dense_n="
                f"{max_dense_n}")}
        else:
            try:
                _, V_inv_tv, d_cv = _block_weights(ids_tv)
                raw_cv = np.diag(V_inv_tv)
                m_cv = float(np.mean(raw_cv))
                w_fit_cv = raw_cv / m_cv if m_cv > 0 and np.isfinite(m_cv) else raw_cv
                if not np.all(np.isfinite(w_fit_cv)) or np.any(w_fit_cv <= 0):
                    d_cv['reason'] = 'non-finite or non-positive diagonal of the inverse'
                    w_fit_cv = None
                else:
                    d_cv['min_weight'] = float(np.min(w_fit_cv))
                    d_cv['max_weight'] = float(np.max(w_fit_cv))
                diagnostics['cv_rows'] = d_cv
            except ValueError as exc:
                diagnostics['cv_rows'] = {'n': len(ids_tv), 'reason': str(exc)}
                w_fit_cv = None

    return BlueWeightContext(
        mode='full', ids_train=ids_train, ids_valid=ids_valid, ids_test=ids_test,
        w_fit=w_fit, W_valid=W_va_inv, W_test=W_te_inv, fingerprint=_fingerprint(w_fit),
        diagnostics=diagnostics, w_fit_cv=w_fit_cv,
    )


def _restrict_to_finite(y: "np.ndarray", yhat: "np.ndarray",
                         w: "Optional[np.ndarray]" = None, W: "Optional[np.ndarray]" = None):
    mask = np.isfinite(y) & np.isfinite(yhat)
    if w is not None:
        mask = mask & np.isfinite(w)
    if not np.all(mask):
        y, yhat = y[mask], yhat[mask]
        if w is not None:
            w = w[mask]
        if W is not None:
            W = W[np.ix_(mask, mask)]
    return y, yhat, w, W


def weighted_pearson(y: Sequence[float], yhat: Sequence[float], *,
                      w: "Optional[np.ndarray]" = None, W: "Optional[np.ndarray]" = None) -> float:
    """`pearson_W` - see module docstring for the exact formula.
    Exactly one of `w` (diagonal, a 1-D array) or `W` (dense, a 2-D PSD
    matrix) should be given; `NaN` (never raises) on empty/degenerate
    input, mirroring `pipeline_utils.safe_regression_metrics()`'s own
    unweighted failure-mode contract."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape[0] != yhat.shape[0] or y.shape[0] < 2:
        return float('nan')
    y, yhat, w, W = _restrict_to_finite(y, yhat, w, W)
    if y.shape[0] < 2:
        return float('nan')
    try:
        if W is not None:
            ones = np.ones(y.shape[0])
            denom1 = float(ones @ W @ ones)
            if not np.isfinite(denom1) or denom1 == 0:
                return float('nan')
            trW = float(np.trace(W))
            if not np.isfinite(trW) or trW == 0:
                return float('nan')
            m_y = float(ones @ W @ y) / denom1
            m_yhat = float(ones @ W @ yhat) / denom1
            dy, dyhat = y - m_y, yhat - m_yhat
            cov_yy = float(dy @ W @ dy) / trW
            cov_hh = float(dyhat @ W @ dyhat) / trW
            cov_yh = float(dy @ W @ dyhat) / trW
        else:
            if w is None:
                w = np.ones_like(y)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum == 0:
                return float('nan')
            m_y = float(np.sum(w * y)) / w_sum
            m_yhat = float(np.sum(w * yhat)) / w_sum
            dy, dyhat = y - m_y, yhat - m_yhat
            cov_yy = float(np.sum(w * dy * dy)) / w_sum
            cov_hh = float(np.sum(w * dyhat * dyhat)) / w_sum
            cov_yh = float(np.sum(w * dy * dyhat)) / w_sum
        if not (np.isfinite(cov_yy) and np.isfinite(cov_hh)) or cov_yy <= 0 or cov_hh <= 0:
            return float('nan')
        r = cov_yh / np.sqrt(cov_yy * cov_hh)
        return float(r) if np.isfinite(r) else float('nan')
    except Exception:
        return float('nan')


def weighted_mse(y: Sequence[float], yhat: Sequence[float], *,
                  w: "Optional[np.ndarray]" = None, W: "Optional[np.ndarray]" = None) -> float:
    """`mse_W` - see module docstring for the exact formula. `NaN` (never
    raises) on empty/degenerate input."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.shape[0] != yhat.shape[0] or y.shape[0] == 0:
        return float('nan')
    y, yhat, w, W = _restrict_to_finite(y, yhat, w, W)
    if y.shape[0] == 0:
        return float('nan')
    try:
        resid = y - yhat
        if W is not None:
            trW = float(np.trace(W))
            if not np.isfinite(trW) or trW == 0:
                return float('nan')
            mse = float(resid @ W @ resid) / trW
        else:
            if w is None:
                w = np.ones_like(y)
            w_sum = float(np.sum(w))
            if not np.isfinite(w_sum) or w_sum == 0:
                return float('nan')
            mse = float(np.sum(w * resid * resid)) / w_sum
        return mse if np.isfinite(mse) else float('nan')
    except Exception:
        return float('nan')



def align_fit_weights(w_fit, train, *, renormalise: bool = True):
    """Return the fitting weight vector aligned to the rows of `train`.

    `BlueWeightContext.w_fit` is aligned to the task's FULL train split
    (positional order, `train.reset_index(drop=True)` in GP()). Several
    callers legitimately fit on a ROW SUBSET of that split - the repeated
    inner 75/25 resamples of `hyperparameter_tuning.make_objective()`
    (HP_TUNE_VALID_REPEATS > 1), its multi-fidelity screening subsample,
    and joint_tuning's inner/OOF folds. Those subsets keep the parent's
    row labels (`train.loc[idx]`), so the labels ARE the positions into
    `w_fit`.

    * `w_fit is None` -> `None` (weighting off; byte-identical, I11).
    * `len(train) == len(w_fit)` -> `w_fit` returned UNCHANGED (the
      object itself - the main per-task dispatch path, I11).
    * otherwise -> `w_fit[train.index]`, re-normalised to mean 1 over the
      subset (the same convention `w_fit` itself follows over its own
      train split) when `renormalise` is True.

    Raises ValueError (never silently mis-aligns) if the labels are not
    unique non-negative integers inside `range(len(w_fit))`.
    """
    if w_fit is None:
        return None
    w = np.asarray(w_fit, dtype=float).ravel()
    n = len(train)
    if n == w.shape[0]:
        return w_fit
    idx = np.asarray(train.index)
    if (idx.size == 0 or idx.dtype.kind not in 'iu'
            or idx.min() < 0 or idx.max() >= w.shape[0]
            or np.unique(idx).size != idx.size):
        raise ValueError(
            f"[blue_weights] cannot align the fitting weights (len={w.shape[0]}) to a "
            f"{n}-row train subset: its row labels are not unique integer positions "
            f"inside the task's full train split.")
    sub = w[idx.astype(np.int64)]
    if not np.all(np.isfinite(sub)):
        raise ValueError("[blue_weights] aligned fitting weights contain non-finite values.")
    if renormalise:
        m = float(sub.mean())
        if m > 0:
            sub = sub / m
    return sub


def cv_weight_view(ctx: "BlueWeightContext") -> "BlueWeightContext":
    """ver4-52: the context a CROSS-VALIDATION fold fit must see - `w_fit` is the
    train+valid fitting vector (`ctx.w_fit_cv`), so `align_fit_weights(w_fit, fold_train)`
    resolves the fold-train rows' positions in `joint_tuning._cv_frame()` (labels
    `0..n_tv-1`, train rows first) to their own weights, re-normalised to mean 1 over the
    fold. The validation-scoring weights are dropped (`w_valid`/`W_valid` = None): they
    belong to the task's validation split, not to a fold's held-out rows, and a fold fit's
    own `pearson_valid`/`mse_valid` are never read by the CV objective (it scores the
    out-of-fold column itself).

    Raises ValueError when `ctx.w_fit_cv` is None (the caller must have checked)."""
    if ctx is None:
        raise ValueError("[blue_weights] cv_weight_view() needs a context, got None.")
    if ctx.w_fit_cv is None:
        raise ValueError(
            "[blue_weights] this context carries no train+valid fitting weights (w_fit_cv is "
            "None) - the cross-validated objective cannot fit fold models under BLUE weighting.")
    from dataclasses import replace
    w = np.asarray(ctx.w_fit_cv, dtype=float)
    return replace(ctx, w_fit=w, w_valid=None, W_valid=None, fingerprint=_fingerprint(w))


def scoring_for(ctx: "Optional[BlueWeightContext]", split: str) -> dict:
    """`{}` (unweighted - the caller should fall back to plain,
    untouched `safe_regression_metrics(actual, predicted)`) when `ctx`
    is `None`; otherwise `{'w': ...}` (diagonal mode) or `{'W': ...}`
    (full mode) for `split` in `('train', 'valid', 'test')` - ready to
    splat straight into `safe_regression_metrics()`/`weighted_pearson()`/
    `weighted_mse()`. Returns `{}` (not a `None`-valued key) whenever
    this split has no weight available (e.g. an empty validation split),
    so a caller's own `**scoring_for(ctx, 'valid')` degrades to the
    unweighted computation rather than passing `w=None, W=None`
    explicitly (equivalent, but `{}` is the cleaner/cheaper no-op)."""
    if ctx is None:
        return {}
    if ctx.mode == 'diagonal':
        w = {'train': ctx.w_fit, 'valid': ctx.w_valid, 'test': ctx.w_test}.get(split)
        return {} if w is None else {'w': w}
    if ctx.mode == 'full':
        if split == 'train':
            return {}  # no dense W_train is ever built - fitting uses w_fit (a vector), not GLS
        W = {'valid': ctx.W_valid, 'test': ctx.W_test}.get(split)
        return {} if W is None else {'W': W}
    return {}


def weight_matrix_from_frame(frame, key: Tuple, ids: Sequence[str]) -> "Optional[np.ndarray]":
    """Reconstruct a dense, symmetric weight matrix from the persisted
    long upper-triangle `ASReml_test_weight_matrix.csv[.gz]` format
    (`population, phenotype, ratio, sample, id1, id2, value`), aligned to
    `ids`' own order - the loader `diversity_summary.py`/the naive-
    ensemble finalisation path use for a resumed/merged 'full'-mode run
    (see `genomic_prediction.py`'s own Step 6b/finalisation notes).

    `key` is `(population, phenotype, ratio, sample)` (already
    stringified the same way `asreml_accuracy_metrics.scenario_key()`
    does). Returns `None` (never raises) if `frame` is `None`/empty, the
    key has no rows, or any requested id is missing from the persisted
    rows - a missing/incomplete matrix must never be silently treated as
    identity; the caller falls back to unweighted scoring in that case
    and should log why.
    """
    if frame is None or frame.shape[0] == 0:
        return None
    pop, phen, ratio, sample = (str(v) for v in key)
    sub = frame[
        (frame['population'].astype(str) == pop) & (frame['phenotype'].astype(str) == phen)
        & (frame['ratio'].astype(str) == ratio) & (frame['sample'].astype(str) == sample)
    ]
    if sub.shape[0] == 0:
        return None
    n = len(ids)
    index = {str(v): i for i, v in enumerate(ids)}
    W = np.full((n, n), np.nan, dtype=float)
    for _, row in sub.iterrows():
        i1, i2 = index.get(str(row['id1'])), index.get(str(row['id2']))
        if i1 is None or i2 is None:
            continue
        W[i1, i2] = float(row['value'])
        W[i2, i1] = float(row['value'])
    if not np.all(np.isfinite(W)):
        return None
    return W
