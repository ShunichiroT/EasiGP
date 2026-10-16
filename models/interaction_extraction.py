"""
models/interaction_extraction.py
=================================
Update ID ver4-5, R2 (blueprint §4.2, invariant I6).

Before this module existed, the identical pairwise-TreeSHAP-matrix recipe
(fit/explain -> abs()/sum() reduction -> upper-triangle selection -> melt
into a long ['marker1', 'marker2', 'value'] table) was implemented three
times, independently:

    models/RF.py                                (abs-then-sum reduction)
    models/GAT_prior_knowledge.py                (sum-then-abs reduction)
    Preprocess/data_driven_prior_network.py      (sum-then-abs reduction)

`tree_shap_interactions()` below is the one shared implementation all
three now call. The two reduction orders are numerically DIFFERENT
whenever a pair's sign varies across the explained rows
(`abs(x).sum(axis=0) != abs(x.sum(axis=0))` in general) - this was flagged,
not unified, by the ver4-4 Change Summary 6 §8, and is preserved here as an
explicit `reduce=` argument rather than silently picking one: each caller
passes its OWN pre-existing value, so redirecting a caller through this
module is a numeric no-op for that caller (verified empirically for the
RF.py vs. GAT_prior_knowledge.py triangle-selection difference too - see
this function's own docstring below).

`top_select()` is the separate, second consolidation: the top-N%/top-M
selection that decides which of the (already-computed) pairs are worth
keeping - RF.py's own historical quantile-filter line, moved here
unchanged, plus a 'count' mode mirroring circos_plot._select_top_
interactions()'s own 'percentage'/'count' semantics (added in the previous
update) so the model-side cap and the plot-side cap speak the same
language.

This module has no project-specific imports beyond `pipeline_utils`
(already a shared, dependency-free helper module every model file already
imports) and third-party packages already pinned by the project
(pandas, numpy, shap, scikit-learn) - I12 is unaffected.

Update ID ver4-5, R2 Stages 8-10 (blueprint §4.2 Layer 2, implementation
sequence stages 8/9/10): four more extractors were added on top of the
Stage 1-3 delivery's `tree_shap_interactions()`/`top_select()`, to reach
the models SHAP's tree explainer cannot: ``shortlist_markers()`` (a small
shared shortlist-by-importance helper every new extractor below uses, so
"take the top-K markers by some importance score" is defined once, not
once per caller), ``h_statistic_interactions()`` (Friedman's pairwise
H^2 - SVR/KNN's route, since ``shap.KernelExplainer`` has no interaction
API), ``nid_interactions()`` (Neural Interaction Detection - MLP's
default, free-from-weights route), and ``surrogate_tree_interactions()``
(the only route that reaches the four R/BGLR models, whose own
explanation machinery cannot be extended pairwise at any tolerable cost
- see that function's own docstring for why its output is APPROXIMATE,
never presented as equivalent to a model's own exact computation, per
risk RK-8).

Two extractors from the blueprint's own Layer 2 spec are NOT implemented
in this delivery - see EasiGP_ver4-5_Change_Summary_3.md §7/§8 for the
disclosed scope decision:
  - ``hessian_interactions()`` (MLP's OPT-IN second method - NID is
    MLP's default and satisfies R2 acceptance criteria 7/8 on its own).
  - ``epistasis_scan_interactions()`` (a classical reference/baseline
    ring, explicitly NOT tied to any single model's own interaction
    output in the blueprint's own framing - "offered as a
    baseline/reference ring ... never as a model's own interaction
    output").
Both remain straightforward additions to this module later - neither is
required by any model's Layer 3 wiring in this delivery.

``ebm_pairwise_interactions()`` is an ADDITION beyond the blueprint's own
enumerated Layer 2 function list, needed for the ``EBM`` Tier 2 model
(blueprint §4.2 Layer 3: "native pairwise terms are the model, not a
post-hoc explanation") - EBM has no post-hoc explainer step to
consolidate (there was never a duplicated inline implementation of this
one, since no model emitted it before), it is simply the natural home
for it alongside every other extractor.

Update (Requirement_patch3.md item 1 - ensemble/weighted-ensemble
interaction rings): ``normalize_task_interactions()``,
``naive_ensemble_interactions()`` and ``weighted_ensemble_interactions()``
(bottom of this module) are the shared implementation behind
Requirement_patch3.md's own item 1 - EasiGP could already return
per-model marker-pair interaction circos rings (every extractor above),
but never an 'ensemble' or weighted-ensemble ('Linear transformation'/
'Nelder Mead'/'Bayesian optimisation') ring combining them. These three
functions are called from models/ensemble.py::ensemble() (naive, once
per run at finalisation) and directly from genomic_prediction.py's own
per-task weighted-ensemble block (once per task, reusing that method's
own already-computed per-model weight row) respectively - see each
function's own docstring for the full contract. Living here rather than
being duplicated across models/ensemble.py/Linear_transformation.py/
Nelder_Mead.py/Bayesian_optimisation.py (the way e.g. `_safe_row_
normalize()` already is, four times, for marker EFFECTS) is a deliberate
departure from that duplication precedent: unlike a five-line row
normaliser, correctly combining a SPARSE, variable-marker-pair-set,
long-format table across models is involved enough that keeping ONE
implementation is worth the small extra indirection - exactly the same
reasoning this module's own docstring already gives for
`tree_shap_interactions()`/`top_select()` above.

Update (RKHS H-statistic interactions): ``surrogate_h_statistic_
interactions()`` is RKHS's OWN route to marker-pair interactions,
replacing ``surrogate_tree_interactions()`` for that one model only -
rrBLUP/BayesB/GBLUP are unaffected and keep calling
``surrogate_tree_interactions()`` exactly as before. It reuses the same
"fit a RandomForestRegressor surrogate on the model's own predictions"
bridge (RKHS's kernel machinery cannot be explained pairwise directly at
any tolerable cost, identically to the other three R models), but
explains that surrogate with Friedman's H-statistic
(``h_statistic_interactions()``) instead of exact pairwise TreeSHAP -
this is what makes a GPU-fitted surrogate viable here (see
``_gpu_random_forest_cls()``'s own docstring for why that is safe for
this extractor but not for ``surrogate_tree_interactions()``'s). See
``genomic_prediction.py::_r_model_surrogate_interaction()`` for the
per-model dispatch this feeds, and ``main_app.py``'s own
``HIDDEN_HPARAM_FIELDS`` for the GUI-visibility change (RKHS's own
interaction toggle is unhidden again; rrBLUP/BayesB/GBLUP's stay
hidden) that accompanies it.

Update (Requirements.md items 1/2 - single-source ensemble interaction
rings): ``naive_ensemble_interactions()`` and
``weighted_ensemble_interactions()`` now each skip any task whose
interaction data comes from FEWER than two distinct models, rather than
producing a combined ring anyway. Previously, a run with several
prediction models selected but only ONE of them actually emitting
interactions (e.g. only RF's own `get_interaction` opt-in enabled) still
produced a separate 'ensemble'/'Linear transformation'/'Nelder Mead'/
'Bayesian optimisation'/'Analytic least-squares' circos interaction ring
- each one just that single model's own ring, relabelled, with nothing
genuinely combined. Worse, those relabelled copies were not guaranteed to
agree with each other: the naive ensemble is built once, across every
task in the run, while each weighted ensemble is only ever built for
tasks with a validation split (W_OPT's own precondition) - so whenever a
run mixed validation and non-validation tasks, the naive ring and the
weighted rings silently averaged over different task subsets and could
render visibly different link patterns despite sharing the exact same
(single) underlying model. Gating both functions on "this task has 2+
contributing models" means a single-source run now produces NO extra
ensemble rings at all (only that one model's own, genuine ring), and any
run that DOES have 2+ contributing models is unaffected - see each
function's own per-task check below for the full rationale.

Bugfix (QTL blind spot in `h_statistic_interactions()`): found from a
side-by-side circos comparison of the same run's 'RF_Shapley' (exact
pairwise TreeSHAP) and 'RF_H-index' (Friedman's H^2, this module's
`h_statistic_interactions()`) interaction rings - the H-index ring was
visibly missing/weak precisely at known QTL positions that the Shapley
ring showed clear, strong interaction links for. Root cause: the
per-marker GRID and the "is this marker degenerate/monomorphic" test
were both resolved from `background_values` - the small, further-
subsampled draw used to average out the other markers (`n_background`,
which for RF's own default of 30 "samples for marker effect
interactions" can be far smaller than the interaction test split it is
drawn from) - rather than from every row this function was actually
given (`X`). A marker with a real but skewed allele frequency (exactly
the profile of a large-effect / selected QTL, which is often close to
fixation in the very panel used to detect it) can easily have its whole
minor genotype class absent from one small random draw purely by
chance, even though the marker is clearly polymorphic overall. That
made such a marker look monomorphic to this function, which short-
circuits to H^2 = 0.0 for EVERY pair touching it with no `predict_fn`
call at all (see `_is_degenerate()`) - not a random loss of precision,
but a systematic one that hit real QTL markers hardest, since they are
disproportionately likely to be skewed this way. Fix: both the grid
cache and the degeneracy test now read from `X` in full (`full_values`
below); `background_values` is unchanged and still the only thing the
partial-dependence AVERAGES themselves are computed over, so runtime
cost is unaffected. Whenever `X` was never subsampled to begin with
(`X.shape[0] <= n_background`) this is a byte-for-byte no-op. This also
transparently fixes the identical exposure in SVR/KNN's own calls into
this same function - see this module's own docstring above for why a
shared implementation is kept in exactly one place.

Update (Requirements.md items 1/2 - H-index precision and cross-model
result diversity): three further, separate findings from the same
investigation that produced the QTL-blind-spot bugfix above.

  1. RF's own `friedman_h` route was reusing `shapley_num` (the
     'pairwise_shap' route's own exact-explanation sample count,
     default 30) as its H-statistic BACKGROUND size, while SVR/KNN/RKHS
     already used a properly-sized ~100. A 30-row background gives a
     visibly noisier H^2 ranking than the same statistic computed on a
     100-row background for another model with every OTHER setting
     equal - a real, fixable contributor to "the number of extracted
     interactions among the prediction models are quite diverse even
     under the same configuration". Fix: RF gained its own dedicated
     `interaction_h_background` field (default 100, see
     hparam_specs.py), read defensively so older configs keep
     `shapley_num`'s previous reuse behaviour unchanged.

  2. `surrogate_tree_interactions()`/`surrogate_h_statistic_
     interactions()` (rrBLUP/BayesB/GBLUP/RKHS's own surrogate route)
     always fit AND explained on the same rows (`train`), while RF/SVR/
     KNN's own DIRECT routes always explain on `test`. A train-explained
     surrogate ring and a test-explained direct ring are drawn from
     different populations (different sample size, different realised
     allele frequencies) - genuinely incomparable "under the same
     configuration". Fix: both functions gained an `explain_pool`
     parameter (see each one's own docstring) - the surrogate still
     FITS on the larger/richer `train` split, but is now EXPLAINED on
     `test` when `genomic_prediction.py::_r_model_surrogate_
     interaction()` provides one (every one of its 8 call sites now
     does). `explain_pool=None` (or no `test` split available) preserves
     the previous train-explained behaviour exactly.

  3. Side-by-side circos rings for the SAME task can show one model
     (e.g. SVR) with only a handful of links and another (e.g. KNN) with
     a dense mesh, even at an identical shortlist size/background size/
     top-N%. This is NOT a residual routing bug - it is what Friedman's
     H^2 is EXPECTED to do when hitting a locally-adaptive, non-smooth
     predict_fn: H^2 assumes a reasonably smooth prediction surface, and
     a nearest-neighbour model's predictions can jump discontinuously as
     the neighbour set changes, making almost EVERY pair look
     "interacting" - not a handful of genuine ones. Measured directly:
     on a synthetic 50-marker panel with one real interacting pair,
     SVR's H^2 had a median of ~0.002 (0.1% of pairs above 0.3); KNN's
     had a median of ~0.058 (36% of pairs above 0.1) - under otherwise
     identical settings. `top_select()`'s nonzero-cap (see its own "Bug
     fix" note below) then leaves the smooth model's ring sparse (few
     genuinely nonzero pairs to rank) while the non-smooth model's ring
     fills out the full requested top-N% (almost nothing is exactly/
     near-zero to cap against) - this is the mechanism behind the
     "few vs. many links" contrast. `h_statistic_interactions()` now
     prints a diagnostic NOTE (`_WIDESPREAD_NONADDITIVITY_MEDIAN`) when
     a call's own median H^2 across live pairs is unusually high,
     naming this exact cause, and KNN's own GUI help text (hparam_
     specs.py) now discloses it - the actionable fix is user awareness
     (H^2 magnitude is not comparable across model types) plus, for
     KNN specifically, a larger neighbour count and/or background size
     to partially smooth the effect, not a change to the statistic
     itself, which is behaving exactly as defined.
"""

from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np
import pandas as pd
# Startup-time optimisation: `shap` (which pulls in numba/llvmlite) and scikit-learn's
# RandomForestRegressor are imported INSIDE the functions that use them (below), not here. This
# module is imported by the GUI only for one tiny helper, and importing both at module scope made
# every EasiGP launch pay their full cold-import cost (very slow on a shared HPC filesystem such as
# NCI Gadi) even when no model was ever run. Behaviour is unchanged: each function imports the same
# library the first time it is called (later calls hit Python's module cache).

from pipeline_utils import (
    parallel_shap_values, shapiq_pairwise_interactions, shapiq_tree_interactions,
    resolve_shapiq_explain_row_count,
    available_memory_bytes, resilient_block_fanout, get_active_compute_resources,
    cuml_gpu_usable,
)
from models.hyperparameter_tuning import base_of

_VALID_REDUCE = ('abs_then_sum', 'sum_then_abs')

#: ver4-35, R1.h2: largest raw TreeSHAP interaction tensor one streamed chunk
#: may hold (rows x M x M float64). Above this the rows are processed in
#: chunks and folded into one (M, M) matrix as they arrive.
_SHAP_STREAM_CHUNK_BYTES = 256 * 1024 * 1024

# Requirements.md item 2: see h_statistic_interactions()'s own inline
# comment (near where this is used) for the full rationale. Chosen well
# above the near-zero median a smooth, mostly-additive model typically
# shows on real marker data, and well below what a genuinely non-smooth
# predict_fn routinely produces on this codebase's own 0/1/2 genotype
# grid (measured informally at roughly 10-20x a smooth kernel model's
# own median on the same data) - a conservative trigger, not a precise
# statistical test.
_WIDESPREAD_NONADDITIVITY_MEDIAN = 0.05


def _xgboost_native_interaction_matrices(fitted_model, X_explain):
    """Per-row pairwise SHAP interaction matrices for an XGBoost model,
    computed via XGBoost's OWN native ``pred_interactions=True``
    prediction path - completely BYPASSING ``shap.TreeExplainer`` for
    XGBoost models.

    Why this exists (do not "simplify" this back to
    ``shap.TreeExplainer(xgb_model)``)
    ------------------------------------------------------------------
    ``shap.TreeExplainer``'s own XGBoost loader (``XGBTreeModelLoader``
    in ``shap/explainers/_tree.py``) reads the booster's serialised
    ``learner_model_param['base_score']`` and, on some ``shap``
    versions, parses it with a bare ``float(...)`` call. Modern XGBoost
    (>=1.6) always serialises ``base_score`` as a bracketed
    single-element JSON-array STRING, e.g. ``'[7.400324E1]'`` - this is
    XGBoost's normal, current-day serialisation format (it stores
    ``base_score`` as a vector to support multi-output models), not a
    corruption of any kind. On a ``shap`` version whose loader lacks a
    fallback for that bracketed form, this raises
    ``ValueError: could not convert string to float: '[7.400324E1]'``
    before the explainer can even be constructed.

    An earlier attempt to fix this by rewriting that field via
    ``booster.save_config()`` -> edit -> ``booster.load_config()`` was
    tried and DOES NOT WORK: ``base_score`` is a derived/output value
    in XGBoost's C++ core, not a settable input, so ``load_config()``
    silently ignores an edited ``base_score`` and the booster keeps
    serving its own original value on the next ``save_config()``/
    ``save_raw()`` call - confirmed directly, not assumed. There is no
    supported way to change how a given ``shap`` version parses that
    field from the XGBoost side, so patching the model is a dead end;
    the only robust fix is to avoid asking ``shap.TreeExplainer`` to
    parse it at all for XGBoost models.

    XGBoost has always been able to compute exact pairwise SHAP
    interaction values itself, natively, via
    ``Booster.predict(dmatrix, pred_interactions=True)`` - this is
    exactly the "native pred_interactions" computation this module's
    own header docstring already describes as what XGBoost does UNDER
    THE HOOD to answer a TreeSHAP interaction query; calling it
    directly here (rather than through ``shap.TreeExplainer``) never
    touches the ``base_score`` string-parsing code path at all, so
    the value's bracketed format is simply irrelevant to this route.
    Verified numerically IDENTICAL (within float tolerance) to
    ``shap.TreeExplainer(...).shap_interaction_values(...)`` for the
    same fitted model and rows, on a `shap` version where the latter
    happens to work, so this is a routing change only - not a change
    in what is computed.

    Parameters
    ----------
    fitted_model : xgboost.XGBRegressor
        Already-fitted (sklearn API) - as every caller in this codebase
        passes.
    X_explain : pandas.DataFrame
        The already-sampled rows to explain - same contract as
        ``tree_shap_interactions()``'s own ``X_explain`` parameter.

    Returns
    -------
    numpy.ndarray, shape (n_rows, n_features, n_features) - the SAME
    shape/semantics ``shap.TreeExplainer(...).shap_interaction_values()``
    returns for a single-output regressor. XGBoost's own native output
    additionally includes one extra bias row/column (shape
    ``(n_rows, n_features+1, n_features+1)``); that bias row/column is
    stripped here before returning, so every downstream caller of this
    function is unaffected by the extra XGBoost-only dimension.
    """
    import xgboost as xgb  # deferred - see models/XGBoost.py's own import-deferral rationale

    booster = fitted_model.get_booster()
    dmatrix = xgb.DMatrix(X_explain, feature_names=list(X_explain.columns))
    raw = np.asarray(booster.predict(dmatrix, pred_interactions=True))
    return raw[:, :-1, :-1]  # drop the trailing bias row/column


def _xgboost_native_shap_values(fitted_model, X_explain):
    """Per-row, per-marker TreeSHAP main-effect (additive attribution)
    values for an XGBoost model, computed via XGBoost's OWN native
    ``pred_contribs=True`` prediction path - the main-effect sibling of
    ``_xgboost_native_interaction_matrices()`` above, needed for exactly
    the SAME reason (see that function's own extensive docstring):
    ``shap.TreeExplainer(xgb_model)`` fails at CONSTRUCTION time on
    modern XGBoost's bracketed ``base_score`` serialisation format, for
    EVERY explanation it is asked for afterwards, not only
    ``shap_interaction_values()`` - so ``tree_shap_main_effects()`` below
    must route XGBoost around ``shap.TreeExplainer`` entirely too, never
    only its own interaction call.

    Parameters
    ----------
    fitted_model : xgboost.XGBRegressor
        Already-fitted (sklearn API).
    X_explain : pandas.DataFrame
        The already-sampled rows to explain.

    Returns
    -------
    numpy.ndarray, shape (n_rows, n_features) - the SAME shape/semantics
    ``shap.TreeExplainer(...).shap_values()`` returns for a single-output
    regressor. XGBoost's own native output additionally includes one
    extra trailing bias column (shape ``(n_rows, n_features+1)``); that
    bias column is stripped here before returning.
    """
    import xgboost as xgb  # deferred - see models/XGBoost.py's own import-deferral rationale

    booster = fitted_model.get_booster()
    dmatrix = xgb.DMatrix(X_explain, feature_names=list(X_explain.columns))
    raw = np.asarray(booster.predict(dmatrix, pred_contribs=True))
    return raw[:, :-1]  # drop the trailing bias column


def _gpu_random_forest_cls(use_gpu):
    """Return (RandomForestRegressor class, is_gpu) for a SURROGATE fit -
    cuML's GPU-backed RandomForestRegressor when `use_gpu` is True AND
    cuML is importable, else scikit-learn's own CPU implementation.
    Never raises - falls back to CPU on any import/compatibility failure,
    identical in spirit to ``models/RF.py``'s own
    ``_random_forest_regressor_cls()`` (not imported from there, to keep
    this module's own "no project-specific imports beyond
    ``pipeline_utils``" invariant - see the module docstring; ``cuml`` is
    a third-party package, exactly like scikit-learn/shap already are).

    Unlike ``models/RF.py``'s own use of this same class-selection idea,
    a GPU-fitted surrogate here is NOT restricted the way that file's own
    comment documents (``shap.TreeExplainer`` cannot introspect a cuML
    forest's tree structure at all) - ``surrogate_h_statistic_
    interactions()`` below never calls ``shap.TreeExplainer`` on the
    surrogate, only its plain ``.predict()`` (via
    ``h_statistic_interactions()``, which needs nothing but a
    ``predict_fn`` callable - see that function's own docstring). cuML
    mirrors its input's own array type by default (a numpy in, numpy out
    contract), so a GPU-fitted surrogate's predictions are consumed here
    exactly like a CPU one's - this is what makes a GPU surrogate safe
    for THIS extractor when it would not be for the TreeSHAP one.
    """
    if use_gpu and cuml_gpu_usable():
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
            return CumlRF, True
        except Exception as exc:
            print(f"[interaction_extraction] USE_GPU_SKLEARN requested for the surrogate "
                  f"H-statistic interaction step but cuML is unavailable ({exc}) - falling "
                  f"back to scikit-learn's CPU RandomForestRegressor.")
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor, False


def _shap_row_chunk(row_chunk, n_rows, n_markers, itemsize=8):
    """Rows per streamed chunk of :func:`tree_shap_interactions`, or `n_rows`
    when the whole tensor is built at once (the legacy behaviour).

    ``row_chunk=None``: automatic - stream only when
    ``ACCELERATION['SHAP_STREAM_REDUCE']`` is on (default) AND the raw
    ``rows x M x M`` tensor would exceed ``_SHAP_STREAM_CHUNK_BYTES``; then
    the chunk is the largest row count that fits. ``row_chunk=0``: never
    stream. ``row_chunk=k > 0``: chunks of `k` rows (used by tests)."""
    if n_rows <= 0:
        return 1
    if row_chunk is not None:
        chunk = int(row_chunk)
        return n_rows if chunk <= 0 else max(1, min(chunk, n_rows))
    if not get_active_compute_resources().get('accel_shap_stream_reduce', True):
        return n_rows
    per_row = max(1, int(n_markers) * int(n_markers) * int(itemsize))
    return max(1, min(n_rows, _SHAP_STREAM_CHUNK_BYTES // per_row))


def _fold_interaction_rows(accumulator, raw_chunk, reduce):
    """Add the per-row ``(M, M)`` matrices of `raw_chunk` to `accumulator`
    ONE ROW AT A TIME, in row order, and return it.

    Why row by row: ``np.abs(raw).sum(axis=0)`` adds the rows one after the
    other, so adding each row to the running matrix performs the very same
    floating-point additions in the very same order and the streamed result
    is bit-identical to the all-at-once one. Adding a whole chunk's sum at
    once would change the association and the last bits. (The blueprint's
    chunk-sum sketch has this flaw - see the change summary.)"""
    for row in raw_chunk:
        term = np.abs(row) if reduce == 'abs_then_sum' else row
        if accumulator is None:
            accumulator = np.array(term, copy=True)
        else:
            np.add(accumulator, term, out=accumulator)
    return accumulator


def tree_shap_interactions(fitted_model, X_explain, marker_names, *,
                            n_jobs=1, reduce='abs_then_sum', row_chunk=None):
    """Exact pairwise TreeSHAP interaction values for a fitted tree
    ensemble, reduced to one scalar strength per marker pair and returned
    as a canonical long-format table.

    Parameters
    ----------
    fitted_model
        A scikit-learn-compatible fitted tree ensemble (whatever
        ``shap.TreeExplainer`` accepts) - the caller decides which forest
        this is (e.g. a lightweight top-importance refit, as RF.py already
        does) and fits it BEFORE calling this function; this function
        never fits anything itself.
    X_explain : pandas.DataFrame
        The already-sampled rows to explain (e.g. via ``shap.sample(...)``
        upstream) - this function does no sampling of its own, so a
        caller's existing sample-size/row-selection convention (RF.py
        explains TEST rows, GAT_prior_knowledge.py explains TRAIN rows -
        both pre-existing and deliberately preserved, see the ver4-5
        blueprint §4.5) is completely unaffected by this consolidation.
    marker_names : sequence of str
        Column names, in the SAME order as ``X_explain``'s columns -
        attached to the returned table's ``marker1``/``marker2`` values.
        Must have the same length as ``X_explain.shape[1]``.
    n_jobs : int, default 1
        Forwarded to ``pipeline_utils.parallel_shap_values`` for the
        per-row fan-out (see that function's own docstring - ``<=1``
        skips the fan-out entirely).
    row_chunk : int or None, default None
        ver4-35, R1.h2. Rows explained per step. The raw TreeSHAP result is
        ``rows x M x M`` float64 (30 rows x 5,000 markers is 6 GB); with
        streaming each chunk's rows are folded into one ``M x M`` matrix and
        released, so memory is ``chunk x M x M`` instead. ``None`` decides
        automatically (``ACCELERATION['SHAP_STREAM_REDUCE']``, default on:
        stream only when the full tensor would exceed 256 MB); ``0`` never
        streams; ``k > 0`` forces chunks of `k` rows. The returned table is
        bit-identical in every case (see ``_fold_interaction_rows``).
    reduce : {'abs_then_sum', 'sum_then_abs'}, default 'abs_then_sum'
        How the per-row interaction matrices are collapsed into one
        matrix before triangle-selection:

          - ``'abs_then_sum'`` - ``abs(matrices).sum(axis=0)`` (RF.py's
            own, pre-existing order).
          - ``'sum_then_abs'`` - ``abs(matrices.sum(axis=0))``
            (GAT_prior_knowledge.py's and data_driven_prior_network.py's
            own, pre-existing, shared order).

        Each caller passes its OWN existing value, so redirecting a
        caller through this shared function changes no caller's numeric
        output for this reason.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']`` -
    one row per unordered marker pair with a nonzero interaction value
    (the diagonal, i.e. a marker "interacting with itself", is always
    excluded). ``marker1``/``marker2`` are real marker NAMES, never
    positional indices.

    Notes on the triangle convention
    ---------------------------------
    This function always selects the upper triangle (RF.py's own historical
    mechanism: ``numpy.triu`` mask -> NaN diagonal -> ``.stack().dropna()``)
    rather than the lower-triangle-via-cartesian-product mechanism
    GAT_prior_knowledge.py/data_driven_prior_network.py used previously.
    Because a pairwise interaction matrix is always symmetric
    (``M[i, j] == M[j, i]`` - true of both reduction orders above), this
    is verified to be a numeric no-op: for a symmetric matrix, the set of
    ``(marker1, marker2, value)`` triples the two mechanisms produce is
    IDENTICAL (confirmed directly, not assumed, against a synthetic
    symmetric fixture during Phase 2 preparation - see
    EasiGP_ver4-5_Change_Summary.md §10). Only which of a pair's two
    markers is reported as ``marker1`` vs. ``marker2`` can differ from the
    previous per-caller implementation for a caller that used the
    lower-triangle mechanism; since an interaction is an unordered
    relationship between two markers, this carries no information change
    (see the Change Summary's disclosed-deviation note for the exact
    callers this applies to and why it is safe).
    """
    if reduce not in _VALID_REDUCE:
        raise ValueError(
            f"tree_shap_interactions: reduce must be one of {_VALID_REDUCE}, got {reduce!r}."
        )
    marker_names = list(marker_names)
    if X_explain.shape[1] != len(marker_names):
        raise ValueError(
            f"tree_shap_interactions: X_explain has {X_explain.shape[1]} columns but "
            f"{len(marker_names)} marker_names were given - these must match 1:1."
        )

    is_xgboost = hasattr(fitted_model, 'get_booster')
    # XGBoost model (duck-typed - only xgboost.XGBRegressor/XGBClassifier
    # have get_booster among this function's callers) - routed through
    # XGBoost's own native pred_interactions computation instead of
    # shap.TreeExplainer. See _xgboost_native_interaction_matrices()'s
    # own docstring for why: shap.TreeExplainer's XGBoost loader can
    # fail on modern XGBoost's normal base_score serialisation format,
    # and that failure cannot be worked around from the model side.
    if not is_xgboost:
        import shap
    explainer = None if is_xgboost else shap.TreeExplainer(fitted_model)

    def _interaction_tensor(rows):
        if is_xgboost:
            return _xgboost_native_interaction_matrices(fitted_model, rows)
        return parallel_shap_values(explainer, rows, None, n_jobs, interaction=True)

    n_rows = X_explain.shape[0]
    chunk = _shap_row_chunk(row_chunk, n_rows, len(marker_names))
    if chunk >= n_rows:
        raw = _interaction_tensor(X_explain)
        if reduce == 'abs_then_sum':
            matrix = np.abs(raw).sum(axis=0)
        else:
            matrix = np.abs(raw.sum(axis=0))
    else:
        print(f"[tree_shap_interactions] streaming the TreeSHAP interaction reduction: "
              f"{n_rows} rows x {len(marker_names)} markers in chunks of {chunk} row(s) "
              f"(the full tensor would take about "
              f"{n_rows * len(marker_names) ** 2 * 8 / 2 ** 30:.1f} GB); the result is "
              f"bit-identical to the all-at-once reduction.")
        matrix = None
        for first in range(0, n_rows, chunk):
            matrix = _fold_interaction_rows(
                matrix, _interaction_tensor(X_explain.iloc[first:first + chunk]), reduce)
        if reduce == 'sum_then_abs':
            matrix = np.abs(matrix)

    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df.where(np.triu(np.ones(matrix_df.shape)).astype(bool))
    writable_array = matrix_df.to_numpy(copy=True)
    np.fill_diagonal(writable_array, np.nan)
    matrix_df.iloc[:, :] = writable_array
    matrix_df.index = matrix_df.columns = marker_names

    result = matrix_df.stack().dropna().reset_index(drop=False)
    result.columns = ['marker1', 'marker2', 'value']
    return result


# ---------------------------------------------------------------------------
# ver4-43 (R1/R2): exact background-pattern deduplication for the
# `baseline_shap` explanations.
#
# A hybrid row `x^{S->z}` (explained row `x`, markers S taken from background
# row `z`) depends on `z` ONLY through `z_S`. Two consequences the legacy loop
# (`baseline_shap_interactions::_row_deltas`) does not exploit:
#   1. every distinct `z_S` pattern is re-evaluated once per background row
#      that carries it (up to B times);
#   2. every hybrid with `z_S == x_S` IS `x`, so its term of the difference is
#      exactly 0, yet it is still evaluated.
# Evaluating each informative distinct pattern once, weighted by how many
# background rows carry it, gives the SAME mean over z (algebraically exact,
# floating-point rounding aside) for any `predict_fn` that is row-independent
# and deterministic. See the ver4-43 design blueprint, record R1.
# ---------------------------------------------------------------------------

#: Largest number of distinct values a marker column may have for the
#: 'levels' mode (0/1/2 genotypes give 3; mean-imputed PLINK columns add one).
_BASELINE_SHAP_MAX_LEVELS = 8

#: Largest (pair, level-combination) table the 'levels' mode will build. At
#: ~12 bytes per entry this is ~0.5 GB; above it the caller falls back to the
#: legacy loop with a NOTE rather than risking the node.
_BASELINE_SHAP_MAX_TABLE_COMBOS = 40_000_000


class _BackgroundLevels(NamedTuple):
    """Per-column level structure of a background matrix.

    mode : 'levels' (few distinct values per column) or 'rows' (every
        background row is its own level - used when a column has more than
        `max_levels` values, when `(K-1)**2 > B`, or when the matrix holds
        non-finite values; never more work than the legacy loop).
    values : (M, K) float - sorted distinct values per column, NaN-padded.
    codes : (B, M) int - level index of every background cell.
    n1 : (M, K) int32 - how many background rows carry each level (0 = pad).
    rep1 : (M, K) int32 - first background row carrying each level (-1 = pad).
    """
    mode: str
    values: np.ndarray
    codes: np.ndarray
    n1: np.ndarray
    rep1: np.ndarray


def _background_levels(X_background, *, max_levels=_BASELINE_SHAP_MAX_LEVELS):
    """Distinct-value structure of `X_background` (B, M) - see
    `_BackgroundLevels`. Representatives are the FIRST background row with
    each level, so the choice is deterministic (I9).

    Raises
    ------
    ValueError
        If `X_background` is not a non-empty 2-D array.
    """
    Z = np.asarray(X_background, dtype=float)
    if Z.ndim != 2 or Z.shape[0] == 0:
        raise ValueError(
            f"interaction_extraction._background_levels: X_background must be a non-empty 2-D "
            f"array, got shape {Z.shape}.")
    B, M = Z.shape
    per_col = None
    n_levels = 0
    if np.isfinite(Z).all():
        per_col = [
            np.unique(Z[:, q], return_index=True, return_inverse=True, return_counts=True)
            for q in range(M)
        ]
        n_levels = max((len(entry[0]) for entry in per_col), default=1)
    if per_col is not None and n_levels <= int(max_levels) and (n_levels - 1) ** 2 <= B:
        values = np.full((M, n_levels), np.nan, dtype=float)
        codes = np.empty((B, M), dtype=np.int16)
        n1 = np.zeros((M, n_levels), dtype=np.int32)
        rep1 = np.full((M, n_levels), -1, dtype=np.int32)
        for q, (uniq, first, inverse, counts) in enumerate(per_col):
            k = len(uniq)
            values[q, :k] = uniq
            codes[:, q] = inverse
            n1[q, :k] = counts
            rep1[q, :k] = first
        return _BackgroundLevels('levels', values, codes, n1, rep1)
    return _BackgroundLevels(
        'rows', np.ascontiguousarray(Z.T),
        np.repeat(np.arange(B, dtype=np.int32)[:, np.newaxis], M, axis=1),
        np.ones((M, B), dtype=np.int32),
        np.tile(np.arange(B, dtype=np.int32), (M, 1)),
    )


def resolve_baseline_shap_dedup(dedup):
    """Whether the deduplicated `baseline_shap` engine is on.

    `dedup=None` reads this run's `ACCELERATION['baseline_shap_engine']`
    (resolved by `pipeline_utils.resolve_compute_resources`; 'legacy' when the
    key or the whole `ACCELERATION` dict is absent - I11). An explicit bool
    wins (used by the offline bench and the parity tests)."""
    if dedup is None:
        return get_active_compute_resources().get('accel_baseline_shap_engine', 'legacy') == 'fast'
    return bool(dedup)


class _JointTable(NamedTuple):
    """Distinct (level-of-i, level-of-j) combinations that occur in the
    background, for every scored pair, sorted by pair index. `off[p]:off[p+1]`
    is pair p's slice; `z` is a representative background row of the combo."""
    p: np.ndarray      # (T,) int32 pair index
    a: np.ndarray      # (T,) int8 level index of the pair's first marker
    b: np.ndarray      # (T,) int8 level index of the pair's second marker
    cnt: np.ndarray    # (T,) int32 background rows carrying the combo
    z: np.ndarray      # (T,) int32 first such background row
    off: np.ndarray    # (P+1,) int64


def _build_joint_table(lv, I, J, *, block_pairs=2000, max_combos=_BASELINE_SHAP_MAX_TABLE_COMBOS):
    """The `_JointTable` of a 'levels'-mode `_BackgroundLevels` for the scored
    pairs `(I[p], J[p])`. Independent of the explained row, so it is built once
    per call and shared by every row.

    Raises
    ------
    RuntimeError
        If the table would exceed `max_combos` entries.
    """
    n_rows, _ = lv.codes.shape
    n_levels = lv.values.shape[1]
    n_combo = n_levels * n_levels
    n_pairs = int(I.shape[0])
    codes = lv.codes.astype(np.int32)
    parts = []
    total = 0
    for lo in range(0, n_pairs, block_pairs):
        hi = min(n_pairs, lo + block_pairs)
        width = hi - lo
        pattern = codes[:, I[lo:hi]] * n_levels + codes[:, J[lo:hi]]            # (B, width)
        flat = (np.arange(width, dtype=np.int64)[np.newaxis, :] * n_combo + pattern).ravel()
        uniq, first, counts = np.unique(flat, return_index=True, return_counts=True)
        total += uniq.shape[0]
        if total > max_combos:
            raise RuntimeError(
                f"level-combination table would exceed {max_combos:,} entries "
                f"({n_pairs:,} pairs x up to {n_combo} combos) - too large for the deduplicated "
                f"engine on this node")
        pair_local = uniq // n_combo
        combo = uniq % n_combo
        parts.append((
            (pair_local + lo).astype(np.int32),
            (combo // n_levels).astype(np.int8),
            (combo % n_levels).astype(np.int8),
            counts.astype(np.int32),
            (first // width).astype(np.int32),
        ))
    if parts:
        p, a, b, cnt, z = (np.concatenate([part[k] for part in parts]) for k in range(5))
    else:
        p = np.empty(0, np.int32); a = np.empty(0, np.int8); b = np.empty(0, np.int8)
        cnt = np.empty(0, np.int32); z = np.empty(0, np.int32)
    off = np.zeros(n_pairs + 1, dtype=np.int64)
    np.cumsum(np.bincount(p, minlength=n_pairs), out=off[1:])
    return _JointTable(p, a, b, cnt, z, off)


class _DedupContext:
    """Everything the deduplicated row loop needs, built once per call and
    shared (pickled) by every worker. A plain container of arrays and the
    caller's own picklable `predict_fn` / `kernel_fast_eval`."""
    __slots__ = ('lv', 'I', 'J', 'table', 'predict_fn', 'kernel_fast_eval', 'X_background',
                 'n_bg')

    def __init__(self, lv, I, J, table, predict_fn, kernel_fast_eval, X_background):
        self.lv = lv
        self.I = I
        self.J = J
        self.table = table
        self.predict_fn = predict_fn
        self.kernel_fast_eval = kernel_fast_eval
        self.X_background = X_background
        self.n_bg = int(X_background.shape[0])

    def combos(self, lo, hi):
        """`(p, a, b, cnt, z)` for pairs `lo:hi` - the table slice in 'levels'
        mode; in 'rows' mode every background row is its own combination."""
        if self.table is not None:
            s, e = int(self.table.off[lo]), int(self.table.off[hi])
            return (self.table.p[s:e], self.table.a[s:e].astype(np.intp),
                    self.table.b[s:e].astype(np.intp), self.table.cnt[s:e], self.table.z[s:e])
        n_bg = self.n_bg
        p = np.repeat(np.arange(lo, hi, dtype=np.int64), n_bg)
        z = np.tile(np.arange(n_bg, dtype=np.intp), hi - lo)
        return p, z, z, np.ones(p.shape[0], dtype=np.int32), z


def _eval_single_hybrids(x, cache, cols, zrows, ctx, max_evals):
    """`f(x^{cols[t] -> X_background[zrows[t], cols[t]]})` for every t, through
    the closed-form evaluator when there is one, else by materialising the
    hybrid rows exactly like the legacy pass. At most `max_evals` rows per call
    (the legacy per-call memory bound)."""
    n_eval = int(cols.shape[0])
    out = np.empty(n_eval, dtype=float)
    fast = ctx.kernel_fast_eval
    for start in range(0, n_eval, max_evals):
        stop = min(n_eval, start + max_evals)
        c = cols[start:stop]
        zr = zrows[start:stop]
        if fast is not None:
            out[start:stop] = fast.single_knockout(x, cache, c, zr)
        else:
            X_off = np.tile(x, (stop - start, 1))
            X_off[np.arange(stop - start), c] = ctx.X_background[zr, c]
            out[start:stop] = np.asarray(ctx.predict_fn(X_off), dtype=float).ravel()
    return out


def _eval_joint_hybrids(x, cache, i_arr, j_arr, zrows, ctx, max_evals):
    """`f(x^{i,j -> background row})` - the two-column analogue of
    `_eval_single_hybrids`."""
    n_eval = int(i_arr.shape[0])
    out = np.empty(n_eval, dtype=float)
    fast = ctx.kernel_fast_eval
    for start in range(0, n_eval, max_evals):
        stop = min(n_eval, start + max_evals)
        i_c = i_arr[start:stop]
        j_c = j_arr[start:stop]
        zr = zrows[start:stop]
        if fast is not None:
            out[start:stop] = fast.joint_knockout(x, cache, i_c, j_c, zr)
        else:
            X_off = np.tile(x, (stop - start, 1))
            flat_rows = np.arange(stop - start)
            X_off[flat_rows, i_c] = ctx.X_background[zr, i_c]
            X_off[flat_rows, j_c] = ctx.X_background[zr, j_c]
            out[start:stop] = np.asarray(ctx.predict_fn(X_off), dtype=float).ravel()
    return out


def _dedup_full_value(x, cache, f_full, ctx):
    """`f(x)` from the same engine as the knockouts when the closed-form
    evaluator offers `full()` (so shared rounding cancels), else the caller's
    own batched `predict_fn(X_explain)` value."""
    fast = ctx.kernel_fast_eval
    if fast is not None and hasattr(fast, 'full'):
        return float(fast.full(x, cache))
    return float(f_full)


def _dedup_row_deltas(x, f_full, ctx, max_evals):
    """Signed `delta(x; i, j)` for every scored pair, averaged over the
    background, for ONE explained row - the deduplicated counterpart of
    `baseline_shap_interactions::_row_deltas` (same return: shape (P,)).

        delta = (1/B) sum_{a != x_i} sum_{b != x_j} n_ij(a, b) *
                [f(x) - f(x^{i->a}) - f(x^{j->b}) + f(x^{ij->a,b})]
    """
    lv = ctx.lv
    n_pairs = int(ctx.I.shape[0])
    inv_b = 1.0 / ctx.n_bg
    cache = ctx.kernel_fast_eval.begin_row(x) if ctx.kernel_fast_eval is not None else None
    f_x = _dedup_full_value(x, cache, f_full, ctx)

    informative = (lv.values != x[:, np.newaxis]) & (lv.n1 > 0)       # (M, K)
    q_idx, a_idx = np.nonzero(informative)
    f_single = np.full(lv.values.shape, np.nan, dtype=float)
    if q_idx.size:
        f_single[q_idx, a_idx] = _eval_single_hybrids(
            x, cache, q_idx, lv.rep1[q_idx, a_idx], ctx, max_evals)

    n_levels = lv.values.shape[1]
    block_pairs = max(1, max_evals // (n_levels * n_levels if ctx.table is not None else ctx.n_bg))
    out = np.zeros(n_pairs, dtype=float)
    for lo in range(0, n_pairs, block_pairs):
        hi = min(n_pairs, lo + block_pairs)
        p, a, b, cnt, z = ctx.combos(lo, hi)
        live = informative[ctx.I[p], a] & informative[ctx.J[p], b]
        if not live.any():
            continue
        p, a, b, cnt, z = p[live], a[live], b[live], cnt[live], z[live]
        i_arr = ctx.I[p]
        j_arr = ctx.J[p]
        f_joint = _eval_joint_hybrids(x, cache, i_arr, j_arr, z, ctx, max_evals)
        terms = cnt * inv_b * (f_x - f_single[i_arr, a] - f_single[j_arr, b] + f_joint)
        out[lo:hi] += np.bincount(p - lo, weights=terms, minlength=hi - lo)
    return out


def _dedup_cost_summary(ctx, X_explain, n_markers_touched, max_rows=16):
    """One log line: mode, K, mean single / joint evaluations per explained row
    under the dedup engine against the legacy loop, and the mean number of
    live level combinations per pair (c-bar). Evaluated on at most `max_rows`
    evenly spaced explained rows."""
    lv = ctx.lv
    n_pairs = int(ctx.I.shape[0])
    n_rows = int(X_explain.shape[0])
    picks = np.unique(np.linspace(0, n_rows - 1, num=min(n_rows, max_rows)).astype(int))
    singles = []
    joints = []
    for r in picks:
        x = X_explain[r]
        informative = (lv.values != x[:, np.newaxis]) & (lv.n1 > 0)
        singles.append(int(informative.sum()))
        if ctx.table is not None:
            t = ctx.table
            joints.append(int((informative[ctx.I[t.p], t.a] & informative[ctx.J[t.p], t.b]).sum()))
        else:
            changed = informative.T                                            # (B, M)
            total = 0
            for lo in range(0, n_pairs, 2000):
                hi = min(n_pairs, lo + 2000)
                total += int((changed[:, ctx.I[lo:hi]] & changed[:, ctx.J[lo:hi]]).sum())
            joints.append(total)
    mean_single = float(np.mean(singles))
    mean_joint = float(np.mean(joints))
    legacy_per_row = float(n_markers_touched + n_pairs) * ctx.n_bg
    dedup_per_row = max(1.0, mean_single + mean_joint)
    n_levels = lv.values.shape[1]
    return (f"[interaction_extraction] baseline_shap dedup ENGINE='fast': level mode "
            f"{lv.mode!r} (K={n_levels}), evaluator "
            f"{'closed-form' if ctx.kernel_fast_eval is not None else 'predict_fn'}; per explained "
            f"row ~{mean_single:,.0f} single + ~{mean_joint:,.0f} joint hybrid evaluation(s) "
            f"(mean live combos per pair c-bar={mean_joint / max(1, n_pairs):.2f}) instead of "
            f"~{legacy_per_row:,.0f} in the legacy loop -> ~{legacy_per_row / dedup_per_row:,.0f}x "
            f"fewer (estimated from {len(picks)} of {n_rows} explained row(s)).")


def baseline_shap_interactions(predict_fn, X_explain, X_background, marker_names, *, pairs,
                                n_jobs=1, reduce='abs_then_sum', pair_chunk_size=2000,
                                screen=None, screen_keep=0.02, cost_log=True,
                                kernel_fast_eval=None, eval_bytes_hint=None, dedup=None):
    """A model-agnostic pairwise interaction value, DIRECTLY COMPARABLE to
    `tree_shap_interactions()`'s own numbers (same units, same reduction
    convention) - built specifically to replace `h_statistic_interactions()`
    (Friedman's H) and the `shapiq`-based routes for KNN/SVR/RKHS, per two
    pieces of user feedback on this codebase's own real experimental
    results: (1) Friedman's H "did not capture correct interactions", and
    (2) `shapiq` "is too computationally expensive" (and, separately, its
    own coalition-sampling variance was flagged as "unreliable").

    Root causes this specifically targets
    --------------------------------------
    (1) H-statistic's own failure mode: `h_statistic_interactions()`
    evaluates the model at a CARTESIAN-PRODUCT grid of marker i's and
    marker j's own resolved values (see that function's own Method
    section) - i.e. it evaluates joint genotype combinations that may
    never actually co-occur in a real, linkage-disequilibrium-structured
    population, silently assuming the two markers are independent. On
    real genomic data (where LD is the norm, not the exception), that
    assumption is routinely violated, and the model is being asked to
    extrapolate into a region it was never trained on - exactly the kind
    of artifact that would look like "does not capture correct
    interactions".

    (2) shapiq's own failure mode: it is a genuine Shapley-value sampling
    estimator over ALL `M` markers, so getting a low-variance answer
    needs a coalition budget that grows with the marker shortlist size -
    "expensive to be reliable, fast to be unreliable" is inherent to the
    method, not a tuning problem.

    This function avoids BOTH: every model evaluation it makes copies
    entire columns verbatim from either the row being explained (`x`) or
    a REAL background row (`z`) - it never synthesises a joint genotype
    combination for the pair being scored that did not actually occur in
    a real individual, so linkage disequilibrium between the two focal
    markers (and between them and every OTHER marker, which is never
    touched) is preserved exactly, by construction - no independence
    assumption anywhere. And it needs no coalition-sampling budget at
    all: for a FIXED pair of markers there are only 2x2=4 "on/off"
    states to evaluate (see Method below), so cost is fixed and
    predictable (see Cost below) rather than a budget/variance trade-off.

    Method (a background-referenced discrete second difference)
    --------------------------------------------------------------
    For an explained row `x`, a background row `z`, and a candidate pair
    of markers `(i, j)`, define four "hybrid" rows built by copying WHOLE
    COLUMNS from `x` or `z` (never any other combination):

        x        - every column from `x` (the row actually being explained)
        x^{i->z} - every column from `x`, EXCEPT column i, taken from `z`
        x^{j->z} - every column from `x`, EXCEPT column j, taken from `z`
        x^{ij->z}- every column from `x`, EXCEPT columns i AND j, taken from `z`

    The pair's interaction value at this (x, z) is the standard discrete
    mixed second difference:

        delta(x, z) = f(x) - f(x^{i->z}) - f(x^{j->z}) + f(x^{ij->z})

    - the same "does the joint change beat the sum of the two individual
    changes" quantity Friedman's H-statistic and SHAP interaction values
    are both, in their own different ways, trying to capture; here it is
    computed by direct evaluation at REAL hybrid rows rather than a
    synthetic independence-assuming grid (H-statistic) or coalition
    sampling (shapiq). `delta(x, z)` is averaged over every row of
    `X_background` (a background SAMPLE of real individuals, exactly the
    same background-averaging convention `h_statistic_interactions()`'s
    own `n_background` already uses) to get one value per explained row,
    then reduced across explained rows via the SAME `reduce=` convention
    `tree_shap_interactions()` uses (`'abs_then_sum'` - the default
    everywhere this codebase calls that function today - or
    `'sum_then_abs'`), so the two functions' output tables are on the SAME
    scale (model-output units, e.g. phenotype units) and can be ranked,
    thresholded, or plotted side by side without any unit conversion.

    Honesty note (what this is NOT): this is a "single-group-vs-rest"
    reduction of the full Shapley-interaction definition (which, in
    general, also marginalises over subsets of every OTHER marker) - it
    holds every marker outside the pair fixed at `x`'s own real values
    rather than also averaging over their omission, exactly like this
    codebase's own `_masked_predict` background-masking convention
    already does elsewhere (e.g. `genomic_prediction.py`'s RKHS routes).
    It is therefore NOT claimed to be bit-identical to TreeSHAP's own
    fully-marginalised value - the comparability claim is about UNITS,
    SIGN CONVENTION, and REDUCTION (so the two tables sit on the same
    scale for ranking/plotting), not a proof of numerical identity. This
    is the same disclosed-approximation class every other non-exact
    route in this codebase already carries (surrogate routes, shapiq's
    own approximators, etc.).

    Cost
    ----
    For `P` scored pairs touching `M'` distinct markers (`M' <= min(2*P,
    M)` - typically `M' << P` once a marker shortlist has more than a
    handful of markers, since `P = C(M, 2)` grows quadratically while
    `M'` can never exceed `M`), `n_explain` explained rows and `B`
    background rows: exactly `n_explain * (1 + M' + P)` batched-and-
    averaged `predict_fn` evaluations worth of work.

    Perf note (single-marker knockout deduplication): the two "one marker
    off" terms `f(x^{i->z})`/`f(x^{j->z})` in the delta formula above
    depend ONLY on `i` (resp. `j`) - never on the pair's other member - so
    `f(x^{i->z})` is IDENTICAL for every pair that includes marker `i`
    (same input rows, same `predict_fn`, hence the same output, up to
    `predict_fn`'s own determinism - already required, see `predict_fn`'s
    own Parameters entry below). Earlier versions of this function
    recomputed both terms once PER PAIR (`2 * P` single-marker
    evaluations - the naive count a formula written "per pair" suggests),
    which is pure redundant work whenever a marker appears in more than
    one scored pair (the normal case for anything but a hand-picked,
    disjoint pair list): `models/RF.py`'s/`models/SVR.py`'s/`models/
    KNN.py`'s own all-pairs shortlist calls, in particular, touch every
    one of `M` markers in `M - 1` pairs each. Computing every distinct
    single-marker term exactly ONCE (`M'` evaluations, not `2 * P`) and
    reusing it for every pair that needs it removes that redundancy
    without changing a single returned value - each term is the SAME
    `predict_fn` call on the SAME rows either way, just made once instead
    of (up to `M - 1`) times, so this is an EXACT refactor, not an
    approximation (I11 is unaffected: no existing caller's numbers
    change). The joint "both markers off" term `f(x^{ij->z})` genuinely
    depends on the pair (it varies both columns of the SAME hybrid row
    together, which is exactly what makes it the interaction term rather
    than a main effect), so it has no analogous shortcut and stays at one
    evaluation per scored pair - the `+ P` above.

    On a `500`-marker all-pairs shortlist (`P = 124,750`, `M' = M = 500`)
    this cuts the total from `1 + 3*124,750 = 374,251` down to
    `1 + 500 + 124,750 = 125,251` `predict_fn`-equivalent row-batches per
    explained row - a ~3x reduction, growing toward exactly 3x as `M`
    (and so `P`) grows further, with NO change to `screen`'s own,
    separate `P`-reducing effect (the two stack: `screen` first shrinks
    `P` itself, then this deduplication removes the redundant single-
    marker work over whatever `scored_pairs` remains).

    Every `predict_fn` call is BATCHED (every live pair's/marker's own
    hybrid rows, up to `pair_chunk_size` pairs/markers at a time, stacked
    into one array and passed to `predict_fn` in a single call) rather
    than looped one row/pair at a time. `n_jobs > 1` fans the explained
    rows out across worker processes via `joblib.Parallel`. The chunk is
    sized from the memory actually available, and if workers are still
    killed (OOM / SIGKILL) the finished rows are kept and only the rest
    are re-dispatched in parallel under a smaller plan - it NEVER falls
    back to a single serial pass (see `pair_chunk_size` below).

    Parameters
    ----------
    predict_fn : callable(numpy.ndarray of shape (n_rows, n_features)) -> array-like of shape (n_rows,)
        A plain prediction function, exactly like `h_statistic_
        interactions()`'s own `predict_fn` (MUST be row-independent - see
        that function's own docstring for the same requirement).
    X_explain : (n_explain, M) array-like
        Rows to explain (e.g. an already-sampled test-row subset) -
        restricted to the SAME `M` marker shortlist as `X_background`/
        `marker_names`/`pairs` (this function does no sampling or
        shortlisting of its own).
    X_background : (n_background, M) array-like
        REAL background rows (e.g. a sample of training individuals) -
        supplies the "off" reference for whichever pair is being scored.
        Never synthesised, never a grid - always an actual observed
        marker vector, so every hybrid row this function builds carries
        genuinely-observed linkage disequilibrium in every column it
        copies.
    marker_names : sequence of str, length M
        Column names, in `X_explain`'s/`X_background`'s own column order.
    pairs : iterable of (int, int)
        POSITIONAL index pairs (into `marker_names`) to evaluate -
        REQUIRED, no default, exactly like `h_statistic_interactions()`'s
        own `pairs` (an explicit shortlist, never every possible pair).
    n_jobs : int, default 1
        Fans the explained-row loop out across `n_jobs` worker processes
        via `joblib.Parallel` (mirrors `h_statistic_interactions()`'s own
        `n_jobs`, applied to explained rows here rather than pair blocks
        since every row's own work is already internally pair-batched).
    reduce : {'abs_then_sum', 'sum_then_abs'}, default 'abs_then_sum'
        See `tree_shap_interactions()`'s own identical parameter - pass
        the SAME value used for the RF/TreeSHAP comparison this route is
        meant to sit alongside, for like-for-like tables.
    eval_bytes_hint : int or None, keyword-only, default None
        Approximate PEAK working memory, in bytes, one hybrid-row
        evaluation costs inside `predict_fn` (e.g. a kernel model's
        `n_reference_rows * 8 * <temporaries>`). Only used to size the
        per-worker chunk (see `pair_chunk_size`); `None` uses
        `kernel_fast_eval.working_bytes_per_eval()` when that exists, else
        a conservative generic figure. A wrong value never changes any
        returned number - a too-small one is corrected automatically by
        the worker-death retry ladder described under `pair_chunk_size`.
    pair_chunk_size : int, default 2000
        MAXIMUM chunk size. The chunk actually used is never larger than
        this, and is additionally capped so that `n_workers * chunk *
        n_background * bytes_per_eval` fits in a safe fraction of the
        memory really available (cgroup / node - see `pipeline_utils.
        available_memory_bytes()`); the resolved plan is printed. If
        workers are nonetheless killed (SIGKILL / OOM), the rows that had
        already finished are KEPT and only the unfinished rows are
        re-dispatched, IN PARALLEL, under a smaller plan (smaller chunk,
        then fewer workers) - this function NEVER falls back to a single
        serial pass; if even the smallest parallel plan is killed it
        raises a `RuntimeError` naming the plan instead.
        How many pairs' (or, for the single-marker knockout pass -
        see Cost above - markers') hybrid rows are stacked into one
        `predict_fn` call at a time (bounds peak memory to roughly
        `pair_chunk_size * n_background` rows regardless of how many
        pairs/markers were requested - down from `3 * pair_chunk_size *
        n_background` before the single-marker deduplication above,
        since the single-marker and joint-knockout passes below each
        build only ONE stacked array per chunk now, not three) - an
        implementation/memory knob, not a statistical one; the returned
        values do not depend on it.
    screen : {None, 'marginal_pd', 'surrogate_shap'}, default None
        Performance knob (mirrors `h_statistic_interactions()`'s own
        identical parameter/semantics, reusing the SAME `_screen_pairs()`
        helper - never a second, parallel implementation). `None` (the
        default) evaluates every requested pair at full cost -
        byte-identical to this parameter not existing (I11), so no
        existing caller's numbers change. When set, only the top
        `screen_keep` LIVE pairs by a cheap proxy ranking are actually
        run through the expensive `_row_deltas` computation; every
        screened-out pair is still returned, at `value=0.0` (same "never
        fewer rows than requested" contract as the unscreened path - see
        Returns below), and a log line names 'APPROXIMATE' and the mode,
        exactly like every other screened route in this module. This is
        the single biggest lever on this function's own O(n_explain *
        B * P) cost (see Cost above) for a large, unscreened marker
        shortlist, since it cuts `P` directly - e.g. `screen_keep=0.02`
        on a 500-marker shortlist (P=124,750 pairs) cuts the expensive
        stage to ~2,495 pairs, a ~50x reduction in `predict_fn` work.
    screen_keep : float or int, default 0.02
        See `h_statistic_interactions()`'s own identical parameter -
        fraction (0, 1] of live pairs to keep when `screen` is set and
        this is a `float`; an exact pair COUNT when an `int`. Unused
        when `screen is None`.
    cost_log : bool, default True
        Print an unconditional one-line pre-flight cost estimate
        (`n_explain`, pairs, background size -> projected `predict_fn`-
        equivalent row count) before doing any work, mirroring
        `h_statistic_interactions()`'s own Layer B convention - purely
        diagnostic (never changes what is computed), so this stays
        `True` by default without affecting I11.
    kernel_fast_eval : object or None, keyword-only, default None
        This update ("minimise the computational time of marker effects
        and interactions for other kernel cases" - previously only
        `models/SVR.py`'s own 'kernel_hessian' route had a closed form;
        `baseline_shap` itself stayed a generic, black-box `predict_fn`
        loop for every caller/kernel): an OPTIONAL closed-form
        accelerator for the actual hybrid-row evaluation inside
        `_row_deltas()` below - `models.kernel_effects.
        SVRBaselineShapFastEval`'s own instance, when the caller's model
        is a fitted `sklearn.svm.SVR` (see that class's own docstring for
        the derivation: every SVR kernel this codebase supports lets a
        hybrid row's own kernel value be computed as a closed-form
        correction to the explained row's own already-computed kernel
        value, in `O(n_ref)` instead of `O(n_ref * M)`, without ever
        materialising the full-width hybrid row `predict_fn` would
        otherwise need). `None` (the default) reproduces this function's
        own pre-existing behaviour EXACTLY (I11) - `predict_fn` is called
        on a literally-materialised hybrid row exactly as before; every
        RKHS/KNN caller, and any SVR caller that does not supply this,
        is completely unaffected. When supplied, this function's own
        RETURNED VALUES are unchanged (bit-identical, to floating-point
        precision, to what `predict_fn` would have returned for the SAME
        hybrid rows) - this parameter changes ONLY how those values are
        computed, never what is computed, `screen`/`reduce`/chunking/
        `n_jobs` fan-out are all unaffected and still apply exactly as
        documented above. Must implement `begin_row(x)` (returns an
        opaque per-explained-row cache), `single_knockout(x, row_cache,
        cols, bg_rows)`, and `joint_knockout(x, row_cache, i_arr, j_arr,
        bg_rows)` - see `SVRBaselineShapFastEval`'s own docstring for the
        exact contract each must satisfy.

    dedup : bool or None, keyword-only, default None
        ver4-43 (R1): evaluate each informative DISTINCT background pattern
        once, weighted by how many background rows carry it, instead of every
        hybrid row for every background row (see the module comment above
        `_BackgroundLevels`). The mean over background rows is algebraically
        identical, so values agree with the legacy loop to floating-point
        rounding - for any row-independent, deterministic `predict_fn` and any
        `kernel_fast_eval` (which may also expose `full(x, row_cache)` so that
        `f(x)` comes from the same engine as the knockouts). `None` (default)
        reads `ACCELERATION['baseline_shap_engine']`
        (`resolve_baseline_shap_dedup`): 'legacy' when absent, i.e. the
        untouched loop below (I11). If the deduplicated path raises, a NOTE is
        printed and the legacy loop finishes the remaining rows.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']`` -
    the SAME canonical long-format table `tree_shap_interactions()`
    returns, one row per requested pair (never fewer - unlike a
    sampling-based route, every requested pair always gets a value, so
    there is nothing to "pad" for row-count parity - screened-out pairs
    are reported at `value=0.0`, exactly like `h_statistic_
    interactions()`'s own `screen=` convention).

    Raises
    ------
    ValueError
        On a column-count mismatch between `X_explain`/`X_background`/
        `marker_names`, or an unrecognised `reduce` - never silently
        returns a wrong table.
    """
    if reduce not in _VALID_REDUCE:
        raise ValueError(
            f"baseline_shap_interactions: reduce must be one of {_VALID_REDUCE}, got {reduce!r}."
        )
    if screen not in _VALID_SCREEN_MODES:
        raise ValueError(
            f"baseline_shap_interactions: screen must be one of {_VALID_SCREEN_MODES}, got {screen!r}."
        )
    marker_names = list(marker_names)
    M = len(marker_names)
    X_explain = np.asarray(X_explain, dtype=float)
    X_background = np.asarray(X_background, dtype=float)
    if X_explain.ndim != 2 or X_background.ndim != 2:
        raise ValueError("baseline_shap_interactions: X_explain/X_background must be 2-D.")
    if X_explain.shape[1] != M or X_background.shape[1] != M:
        raise ValueError(
            f"baseline_shap_interactions: X_explain has {X_explain.shape[1]} columns, "
            f"X_background has {X_background.shape[1]}, and marker_names has {M} entries - "
            f"all three must describe the SAME marker shortlist."
        )
    pairs = list(pairs)
    if not pairs:
        return pd.DataFrame(columns=['marker1', 'marker2', 'value'])
    if X_explain.shape[0] == 0 or X_background.shape[0] == 0:
        # Nothing to explain/reference against - every requested pair is
        # reported at 0.0 rather than silently dropped (I5-style "never
        # fewer rows than requested" contract, see this function's own
        # Returns section).
        rows = [(marker_names[i], marker_names[j], 0.0) for i, j in pairs]
        return pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])

    B = X_background.shape[0]
    n_explain = X_explain.shape[0]

    # Performance knob (see this function's own `screen` docstring above):
    # rank `pairs` by a cheap proxy via the SAME `_screen_pairs()` helper
    # `h_statistic_interactions()` already uses, and only run the
    # expensive per-pair hybrid-row computation below on the kept
    # shortlist - every screened-out pair is still reported, at
    # value=0.0 (never fewer rows than requested). `screen is None` (the
    # default) skips this entirely: `scored_pairs is pairs`, so the rest
    # of this function runs exactly as before (I11).
    scored_pairs = pairs
    if screen is not None:
        _involved = sorted({c for pr in pairs for c in pr})
        _grids = (
            {c: _resolve_marker_grid(X_background[:, c])[0] for c in _involved}
            if screen == 'marginal_pd' else None
        )
        scored_pairs, _screen_provenance = _screen_pairs(
            screen, pairs, predict_fn=predict_fn, background_values=X_background,
            marker_names=marker_names, grids=_grids, n_bg=B, screen_keep=screen_keep,
        )
        _n_screened_out = len(pairs) - len(scored_pairs)
        print(f"[interaction_extraction] APPROXIMATE: baseline_shap screen={_screen_provenance!r} "
              f"kept {len(scored_pairs)}/{len(pairs)} live pair(s) for the exact hybrid-row "
              f"computation ({_n_screened_out} unscreened pair(s) emitted at value=0.0).")

    P = len(scored_pairs)
    # Distinct markers actually touched by `scored_pairs` - see this
    # function's own Cost docstring above for why this (not `2 * P`) is
    # the real single-marker-knockout evaluation count.
    _involved_markers = sorted({c for pr in scored_pairs for c in pr})
    Mp = len(_involved_markers)

    if cost_log:
        _projected_rows = n_explain * (1 + Mp + P) * B
        _screen_note = (
            f" ({len(pairs) - P} of {len(pairs)} pair(s) screened out at no extra cost)"
            if screen is not None else ""
        )
        print(f"[interaction_extraction] baseline_shap: {n_explain} explained row(s) x {P} "
              f"scored pair(s) ({Mp} distinct marker(s)) x {B} background row(s) -> "
              f"~{_projected_rows:,} predict_fn-equivalent row(s) of work{_screen_note} "
              f"(single-marker knockout terms deduplicated across pairs - was "
              f"~{n_explain * (1 + 3 * P) * B:,} before that, a ~"
              f"{((1 + 3 * P) / max(1, 1 + Mp + P)):.1f}x reduction). If this is still too "
              f"slow, the biggest remaining levers (in order of effect) are: turn on "
              f"`screen` (cuts P directly, e.g. screen_keep=0.02 is a ~50x cut on a "
              f"500-marker shortlist), lower `max_interaction_features` (cuts P "
              f"quadratically), lower `interaction_background` (cuts B linearly), or lower "
              f"the marker-effect/Shapley row count feeding `X_explain` (cuts n_explain "
              f"linearly).")

    # Precompute f(x) for EVERY explained row in ONE batched predict_fn
    # call (previously one separate single-row predict_fn call PER
    # explained row, made from inside `_row_deltas` below) - removes
    # `n_explain - 1` redundant predict_fn calls' worth of per-call
    # overhead; the returned values are numerically unaffected.
    f_full_all = np.asarray(predict_fn(X_explain), dtype=float).ravel()

    # Vectorised marker-index -> f_single-position lookup (an array, not a
    # dict, so `_marker_pos[i_arr]` below resolves every pair's own
    # position in ONE numpy fancy-index call instead of a Python-level
    # per-pair dict lookup loop).
    _marker_pos = np.full(M, -1, dtype=int)
    _marker_pos[_involved_markers] = np.arange(Mp)

    def _row_deltas(x, f_full, pair_chunk_size):
        """Signed delta(x, z) for every SCORED pair, averaged over every
        row of X_background - shape (P,). See this function's own outer
        docstring (Cost, above) for the single-marker-knockout
        deduplication this implements: `f(x^{i->z})` is computed exactly
        ONCE per distinct marker `i` touched by `scored_pairs` (the
        `_single` pass below), not once per pair, then reused for every
        pair that needs it; only the genuinely pair-specific joint term
        `f(x^{ij->z})` (the `_joint` pass below) is still computed once
        per scored pair."""
        # Pass 1: single-marker knockout terms, one predict_fn-equivalent
        # evaluation per distinct marker touched by scored_pairs (not per
        # pair) - see the module-level Cost note above for why this is an
        # exact, not approximate, reduction. The hybrid-row assignment
        # itself is done with ONE vectorised fancy-index write per chunk
        # (not a per-marker Python loop) - purely a constant-factor
        # implementation detail (identical values, just written via numpy's
        # own C-level indexing instead of a Python `for`), never a change
        # to what is computed.
        #
        # This update (`kernel_fast_eval` - see this function's own
        # docstring): when supplied, the hybrid row is never materialised
        # at all - `kernel_fast_eval.single_knockout()`/`.joint_knockout()`
        # below return the SAME values `predict_fn(X_off)` would, via a
        # closed-form correction to this row's own `begin_row()` cache
        # (`_row_cache`, computed ONCE per explained row and reused across
        # every chunk of both passes). `kernel_fast_eval is None` (the
        # default) takes the exact pre-existing branch, byte-for-byte.
        _bg_cycle = np.arange(B)
        _row_cache = kernel_fast_eval.begin_row(x) if kernel_fast_eval is not None else None
        f_single = np.empty(Mp, dtype=float)
        for start in range(0, Mp, pair_chunk_size):
            block = _involved_markers[start:start + pair_chunk_size]
            n_blk = len(block)
            cols = np.repeat(block, B)
            bg_rows = np.tile(_bg_cycle, n_blk)
            if kernel_fast_eval is not None:
                f_vals = kernel_fast_eval.single_knockout(x, _row_cache, cols, bg_rows)
            else:
                X_off = np.tile(x, (n_blk * B, 1))
                flat_rows = np.arange(n_blk * B)
                X_off[flat_rows, cols] = X_background[bg_rows, cols]
                f_vals = np.asarray(predict_fn(X_off), dtype=float)
            f_single[start:start + n_blk] = f_vals.reshape(n_blk, B).mean(axis=1)

        # Pass 2: joint ("both markers off") knockout term - irreducibly
        # pair-specific (see this function's own docstring), one
        # predict_fn-equivalent evaluation per scored pair, exactly as
        # before (same vectorised-assignment treatment as Pass 1 above,
        # and the same `kernel_fast_eval` shortcut when supplied).
        out = np.empty(P, dtype=float)
        for start in range(0, P, pair_chunk_size):
            block = scored_pairs[start:start + pair_chunk_size]
            n_blk = len(block)
            i_arr = np.fromiter((i for i, _ in block), dtype=int, count=n_blk)
            j_arr = np.fromiter((j for _, j in block), dtype=int, count=n_blk)
            bg_rows = np.tile(_bg_cycle, n_blk)
            i_rep = np.repeat(i_arr, B)
            j_rep = np.repeat(j_arr, B)
            if kernel_fast_eval is not None:
                f_ij_off = kernel_fast_eval.joint_knockout(x, _row_cache, i_rep, j_rep, bg_rows)
            else:
                X_ij_off = np.tile(x, (n_blk * B, 1))
                flat_rows = np.arange(n_blk * B)
                X_ij_off[flat_rows, i_rep] = X_background[bg_rows, i_rep]
                X_ij_off[flat_rows, j_rep] = X_background[bg_rows, j_rep]
                f_ij_off = np.asarray(predict_fn(X_ij_off), dtype=float)
            f_ij_off_mean = f_ij_off.reshape(n_blk, B).mean(axis=1)
            out[start:start + n_blk] = (
                f_full - f_single[_marker_pos[i_arr]] - f_single[_marker_pos[j_arr]] + f_ij_off_mean
            )
        return out

    # ver4-43 (R1): the deduplicated engine. Everything it needs is built once
    # here (never per row); any failure to build it leaves `_dedup_ctx` None and
    # the untouched legacy `_row_deltas` above does all the work.
    _dedup_ctx = None
    if resolve_baseline_shap_dedup(dedup):
        try:
            _i_arr = np.fromiter((i for i, _ in scored_pairs), dtype=np.intp, count=P)
            _j_arr = np.fromiter((j for _, j in scored_pairs), dtype=np.intp, count=P)
            _lv = _background_levels(X_background)
            _table = _build_joint_table(_lv, _i_arr, _j_arr) if _lv.mode == 'levels' else None
            _dedup_ctx = _DedupContext(_lv, _i_arr, _j_arr, _table, predict_fn,
                                       kernel_fast_eval, X_background)
            try:
                print(_dedup_cost_summary(_dedup_ctx, X_explain, Mp))
            except Exception as exc:
                print(f"[interaction_extraction] NOTE: baseline_shap dedup cost summary could "
                      f"not be computed ({exc!r}); the deduplicated engine is still used.")
        except Exception as exc:
            _dedup_ctx = None
            print(f"[interaction_extraction] NOTE: baseline_shap_engine='fast' could not be set "
                  f"up ({exc!r}) - using the legacy hybrid-row loop for this call instead "
                  f"(slower, same estimator).")

    def _row_deltas_dedup(x, f_full, pair_chunk_size):
        """Deduplicated `_row_deltas` - same inputs, same (P,) output, see
        `_dedup_row_deltas`. At most `pair_chunk_size * B` hybrid rows go
        through one evaluation call, the legacy per-call memory bound."""
        return _dedup_row_deltas(x, f_full, _dedup_ctx, pair_chunk_size * B)

    def _make_row_task(chunk):
        """Per-block function (picklable closure) for a given chunk size."""
        def _process_rows(row_indices):
            if _dedup_ctx is None:
                return [_row_deltas(X_explain[r], f_full_all[r], chunk) for r in row_indices]
            rows_out = []
            dedup_ok = True
            for r in row_indices:
                if dedup_ok:
                    try:
                        rows_out.append(_row_deltas_dedup(X_explain[r], f_full_all[r], chunk))
                        continue
                    except Exception as exc:
                        dedup_ok = False
                        print(f"[interaction_extraction] NOTE: deduplicated baseline_shap failed "
                              f"on explained row {r} ({exc!r}) - this worker finishes its "
                              f"remaining row(s) with the legacy hybrid-row loop.")
                rows_out.append(_row_deltas(X_explain[r], f_full_all[r], chunk))
            return rows_out
        return _process_rows

    # ------------------------------------------------------------------
    # Memory-aware plan (fix: "The exit codes of the workers are
    # {SIGKILL(-9), ...}) - falling back to a single serial pass").
    #
    # Peak memory per worker is dominated by ONE predict/knockout call on
    # `chunk * B` stacked hybrid rows: for the SVR closed-form path that is
    # ~12 temporaries of shape (n_support_vectors, chunk * B) float64 (a
    # 2000-pair chunk at B=100 and ~750 support vectors is ~9 GB PER
    # WORKER - times `n_jobs` workers, far beyond a node's RAM, so the
    # kernel OOM-killer SIGKILLed the pool). The chunk size only controls
    # memory/scheduling - it never changes a returned value - so it is
    # sized from the memory really available instead of a fixed 2000.
    # ------------------------------------------------------------------
    _requested_chunk = max(1, int(pair_chunk_size))
    _MIN_COMFY_CHUNK, _FLOOR_CHUNK = 64, 8
    _WORKER_OVERHEAD = 1024 ** 3          # interpreter + numpy/sklearn/shap imports per worker
    if eval_bytes_hint is not None:
        _bytes_per_eval = float(eval_bytes_hint)
    elif kernel_fast_eval is not None and hasattr(kernel_fast_eval, 'working_bytes_per_eval'):
        _bytes_per_eval = float(kernel_fast_eval.working_bytes_per_eval())
    else:
        # X_off tile + fancy-assignment copy + predict_fn's input copy, plus a
        # cushion for a kernel/distance matrix against ~2k reference rows.
        _bytes_per_eval = 8.0 * (3 * M + 2048)
    _bytes_per_pair = max(1.0, _bytes_per_eval * B)   # one pair's stack of B hybrid rows

    if n_jobs in (None, 0, 1) or n_explain <= 1:
        _want_workers = 1
    elif int(n_jobs) < 0:
        import os as _os
        _want_workers = max(1, (_os.cpu_count() or 1) + 1 + int(n_jobs))
    else:
        _want_workers = int(n_jobs)
    _want_workers = max(1, min(_want_workers, n_explain))
    _min_workers = min(2, _want_workers)          # never below 2 if parallel was asked for
    _plan_history = []

    def _plan(attempt):
        avail = available_memory_bytes()
        frac = 0.6 * (0.5 ** attempt)
        workers = _want_workers
        if avail is None:
            chunk = max(_FLOOR_CHUNK, _requested_chunk // (2 ** attempt))
            mem_note = "available memory unknown"
        else:
            budget = float(avail) * frac
            def _fit(w):
                return int((budget / w - _WORKER_OVERHEAD) // _bytes_per_pair)
            chunk = _fit(workers)
            while chunk < _MIN_COMFY_CHUNK and workers > _min_workers:
                workers = max(_min_workers, workers // 2)
                chunk = _fit(workers)
            mem_note = (f"memory budget {budget / 1024 ** 3:.1f} GiB "
                        f"({frac:.0%} of {avail / 1024 ** 3:.1f} GiB available), "
                        f"~{_bytes_per_pair / 1024 ** 2:.1f} MiB per pair")
        chunk = max(_FLOOR_CHUNK, min(_requested_chunk, chunk))
        if _plan_history:                      # a retry is never LARGER than the plan that failed
            workers = min(workers, _plan_history[-1][0])
            chunk = min(chunk, _plan_history[-1][1])
        key = (workers, chunk)
        if attempt > 0 and _plan_history and key == _plan_history[-1]:
            # No smaller configuration than the one that just got killed:
            # force strictly smaller ones until the floors are reached.
            if chunk > _FLOOR_CHUNK:
                chunk = max(_FLOOR_CHUNK, chunk // 2)
            elif workers > _min_workers:
                workers = max(_min_workers, workers // 2)
            else:
                return None
            key = (workers, chunk)
        _plan_history.append(key)
        return workers, chunk, (f"pair_chunk={chunk} (max {_requested_chunk}), {mem_note}")

    row_indices = list(range(n_explain))
    if _want_workers == 1:
        # The caller asked for a single process (n_jobs=1) or there is only
        # one explained row - this is that request, not a fallback. Chunk is
        # still memory-capped.
        _p = _plan(0)
        raw_rows = _make_row_task(_p[1])(row_indices)
    else:
        n_blocks = max(1, min(n_explain, 4 * _want_workers))
        block_size = max(1, -(-n_explain // n_blocks))  # ceil division
        blocks = [row_indices[k:k + block_size] for k in range(0, n_explain, block_size)]
        block_results = resilient_block_fanout(
            _make_row_task, blocks,
            lambda attempt: (lambda p: None if p is None else (p[0], p[1], p[2]))(_plan(attempt)),
            label="baseline_shap_interactions", log_prefix="[interaction_extraction]",
        )
        raw_rows = [row for block in block_results for row in block]

    raw = np.vstack(raw_rows)  # (n_explain, P), signed
    if reduce == 'abs_then_sum':
        scored_values = np.abs(raw).sum(axis=0)
    else:
        scored_values = np.abs(raw.sum(axis=0))

    # Screened-out pairs (if any) are reported at value=0.0 - same
    # "never fewer rows than requested" contract as every other early
    # return above, and the SAME convention `h_statistic_interactions()`
    # already uses for its own `screen=` pairs.
    value_by_pair = dict(zip(scored_pairs, scored_values))
    values = [value_by_pair.get(pr, 0.0) for pr in pairs]

    result = pd.DataFrame(
        {'marker1': [marker_names[i] for i, _ in pairs],
         'marker2': [marker_names[j] for _, j in pairs],
         'value': values},
    )
    return result


def baseline_shap_effects(predict_fn, X_explain, X_background, *, kernel_fast_eval=None, n_jobs=1,
                          dedup=True, reduce='abs_then_sum', chunk_size=2000, cost_log=True,
                          log_prefix='[interaction_extraction]'):
    """ver4-43 (R2): per-marker `baseline_shap` EFFECT for every marker - the
    single-marker term of `baseline_shap_interactions()`'s estimator.

        phi_q(x) = (1/B) * sum_{z in background} [ f(x) - f(x^{q->z}) ]

    where `x^{q->z}` is `x` with column q taken from the REAL background row
    z. It is reported with the same cross-row reduction every effect route in
    this codebase uses (`reduce='abs_then_sum'`: sum over explained rows of
    |phi_q(x)|). Because the background rows are real training individuals,
    linkage disequilibrium among the markers that are NOT being knocked out is
    preserved, and every marker gets a deterministic value at any M (no
    marker-count threshold, no coalition sampling).

    Cost (`dedup=True`): `n_explain * M * (K - 1)` hybrid evaluations, K =
    distinct values per column (3 for 0/1/2 genotypes) - versus `n_explain * M
    * B` in the literal definition (`dedup=False`, kept only as the reference
    the parity check compares against). A hybrid depends on z only through
    `z_q`, so each informative distinct value is evaluated once and weighted by
    the number of background rows that carry it (see the module comment above
    `_BackgroundLevels`).

    Parameters
    ----------
    predict_fn : callable(ndarray (n, M)) -> array-like (n,)
        Row-independent, deterministic prediction function.
    X_explain : (n_explain, M); X_background : (B, M) - REAL rows, same columns.
    kernel_fast_eval : object or None, keyword-only
        Closed-form evaluator exposing `begin_row`, `single_knockout` and,
        optionally, `full` (`SVRBaselineShapFastEval`). `None` evaluates
        `predict_fn` on materialised hybrid rows.
    n_jobs : int - worker processes over explained rows (`resilient_block_fanout`).
    dedup : bool - the exact deduplicated evaluation (True) or the literal
        every-(q, z) evaluation (False).
    reduce : {'abs_then_sum', 'sum_then_abs'}.
    chunk_size : int - at most this many hybrid rows per evaluation call
        (further capped by the memory actually available).
    cost_log : bool - print the one-line cost estimate.
    log_prefix : str - prefix of the log lines (e.g. '[SVR]').

    Returns
    -------
    (M,) float ndarray of effect scores, never NaN (a non-finite score is
    reported as 0.0).

    Raises
    ------
    ValueError
        On a bad `reduce`, non-2-D inputs or a column-count mismatch.
    RuntimeError
        If worker processes keep being killed under every smaller plan.
    """
    if reduce not in _VALID_REDUCE:
        raise ValueError(
            f"baseline_shap_effects: reduce must be one of {_VALID_REDUCE}, got {reduce!r}.")
    X_explain = np.asarray(X_explain, dtype=float)
    X_background = np.asarray(X_background, dtype=float)
    if X_explain.ndim != 2 or X_background.ndim != 2:
        raise ValueError("baseline_shap_effects: X_explain/X_background must be 2-D.")
    n_explain, n_markers = X_explain.shape
    if X_background.shape[1] != n_markers:
        raise ValueError(
            f"baseline_shap_effects: X_explain has {n_markers} column(s) but X_background has "
            f"{X_background.shape[1]} - both must describe the SAME marker set.")
    if n_explain == 0 or X_background.shape[0] == 0 or n_markers == 0:
        return np.zeros(n_markers, dtype=float)
    n_bg = X_background.shape[0]

    lv = _background_levels(X_background) if dedup else None
    f_full_all = np.asarray(predict_fn(X_explain), dtype=float).ravel()
    ctx = _DedupContext(lv, np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), None,
                        predict_fn, kernel_fast_eval, X_background)

    def _row_effects(x, f_full, max_evals):
        """Signed phi_q(x) for every marker q, shape (M,)."""
        cache = kernel_fast_eval.begin_row(x) if kernel_fast_eval is not None else None
        f_x = _dedup_full_value(x, cache, f_full, ctx)
        if lv is not None:
            informative = (lv.values != x[:, np.newaxis]) & (lv.n1 > 0)
            q_idx, a_idx = np.nonzero(informative)
            if q_idx.size == 0:
                return np.zeros(n_markers, dtype=float)
            f_single = _eval_single_hybrids(x, cache, q_idx, lv.rep1[q_idx, a_idx], ctx, max_evals)
            return np.bincount(q_idx, weights=lv.n1[q_idx, a_idx] * (f_x - f_single) / n_bg,
                               minlength=n_markers)
        phi = np.empty(n_markers, dtype=float)
        per_block = max(1, max_evals // n_bg)
        for start in range(0, n_markers, per_block):
            block = np.arange(start, min(n_markers, start + per_block))
            cols = np.repeat(block, n_bg)
            zrows = np.tile(np.arange(n_bg), block.shape[0])
            f_single = _eval_single_hybrids(x, cache, cols, zrows, ctx, max_evals)
            phi[block] = (f_x - f_single.reshape(block.shape[0], n_bg)).mean(axis=1)
        return phi

    def _make_row_task(max_evals):
        def _process_rows(row_indices):
            return [_row_effects(X_explain[r], f_full_all[r], max_evals) for r in row_indices]
        return _process_rows

    if cost_log:
        per_row = (n_markers * n_bg) if lv is None else None
        print(f"{log_prefix} baseline_shap effect: {n_explain} explained row(s) x {n_markers} "
              f"marker(s) x {n_bg} background row(s); "
              + (f"evaluating every (marker, background row) hybrid (dedup off, ~{n_explain * per_row:,} "
                 f"hybrid row(s))." if lv is None else
                 f"deduplicated, level mode {lv.mode!r} (K={lv.values.shape[1]}) - at most "
                 f"~{n_explain * n_markers * (lv.values.shape[1] - 1 if lv.mode == 'levels' else n_bg):,} "
                 f"hybrid row(s), evaluator "
                 f"{'closed-form' if kernel_fast_eval is not None else 'predict_fn'}."))

    bytes_per_eval = (float(kernel_fast_eval.working_bytes_per_eval())
                      if kernel_fast_eval is not None and hasattr(kernel_fast_eval, 'working_bytes_per_eval')
                      else 8.0 * (3 * n_markers + 2048))
    requested = max(1, int(chunk_size))
    if n_jobs in (None, 0, 1) or n_explain <= 1:
        want_workers = 1
    elif int(n_jobs) < 0:
        import os as _os
        want_workers = max(1, (_os.cpu_count() or 1) + 1 + int(n_jobs))
    else:
        want_workers = int(n_jobs)
    want_workers = max(1, min(want_workers, n_explain))
    history = []

    def _plan(attempt):
        avail = available_memory_bytes()
        workers = want_workers
        if avail is None:
            max_evals = max(8, requested // (2 ** attempt))
            note = "available memory unknown"
        else:
            budget = float(avail) * 0.6 * (0.5 ** attempt)
            max_evals = int((budget / workers - 1024 ** 3) // bytes_per_eval)
            note = f"memory budget {budget / 1024 ** 3:.1f} GiB, ~{bytes_per_eval / 1024:.1f} KiB per hybrid"
        max_evals = max(8, min(requested, max_evals))
        if history:
            workers = min(workers, history[-1][0])
            max_evals = min(max_evals, history[-1][1])
            if attempt > 0 and (workers, max_evals) == history[-1]:
                if max_evals > 8:
                    max_evals = max(8, max_evals // 2)
                elif workers > min(2, want_workers):
                    workers = max(min(2, want_workers), workers // 2)
                else:
                    return None
        history.append((workers, max_evals))
        return workers, max_evals, f"hybrids per call={max_evals} (max {requested}), {note}"

    row_indices = list(range(n_explain))
    if want_workers == 1:
        raw_rows = _make_row_task(_plan(0)[1])(row_indices)
    else:
        n_blocks = max(1, min(n_explain, 4 * want_workers))
        block_size = max(1, -(-n_explain // n_blocks))
        blocks = [row_indices[k:k + block_size] for k in range(0, n_explain, block_size)]
        block_results = resilient_block_fanout(
            _make_row_task, blocks, _plan, label="baseline_shap_effects", log_prefix=log_prefix,
        )
        raw_rows = [row for block in block_results for row in block]

    raw = np.vstack(raw_rows)                                   # (n_explain, M), signed
    scores = np.abs(raw).sum(axis=0) if reduce == 'abs_then_sum' else np.abs(raw.sum(axis=0))
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def tree_shapiq_interactions(fitted_model, X_explain, marker_names, *, orders=(2,), index='k-SII',
                              max_seconds=None, n_jobs: int = 1,
                              X_train_background=None, budget=4096,
                              approximator: str = 'auto', candidate_narrowing: bool = True):
    """`models/RF.py`'s own 'shapiq_tree' 'Marker interaction method'
    (ver4-25, Stage 8, design blueprint R11): exact Shapley-interaction
    values from the fitted forest itself via `shapiq.TreeExplainer`
    (`pipeline_utils.shapiq_tree_interactions()`), reduced to this
    module's own canonical long-format table - the SAME shape
    `tree_shap_interactions()` above returns.

    Unlike `surrogate_shapiq_interactions()`/RKHS's own 'surrogate_
    shapiq' route (which explains a lightweight SURROGATE fitted to
    another model's predictions), this explains the FITTED FOREST
    directly and exactly, mirroring `tree_shap_interactions()`'s own
    "no surrogate, no sampling" contract - `shapiq`'s TreeSHAP-IQ
    algorithm walks the tree structure itself, the same way `shap.
    TreeExplainer` does for the pairwise (order-2-only) route.

    `shapiq` natively supports interaction orders above 2 (`orders` may
    list any subset of ``{2, 3, ...}``) - EVERY requested order's own
    rows are returned (ver4-25, Stage 9, design blueprint R12 - see
    Returns below); `models/genomic_prediction.py`'s own Step 10 is what
    actually routes order==2 rows into Interaction.csv and order>=3 rows
    into Interaction_higher_order.csv.

    ver4-30: exact TreeSHAP-IQ's own published complexity, O(T*L*D**k),
    is EXPONENTIAL in the requested order k. `models/RF.py`'s own
    order-aware depth/estimator/row-count adaptive shrinking (design
    blueprint R2, ver4-27, ver4-28) attacks the INPUTS to that exponent,
    but was confirmed in practice to still leave a genuine k>2 search
    "extremely large"/"cannot be used" on a realistically-sized forest
    and marker shortlist - shrinking D alone cannot fully offset raising
    k, and T/marker-count both also feed the same exponent. Rather than
    continue shrinking an EXACT, tree-structure-bound algorithm, this
    function now offers a second, BUDGET-bounded strategy for `max(orders)
    > 2`: explain `fitted_model`'s own ``.predict`` through shapiq's
    model-AGNOSTIC `TabularExplainer` (`pipeline_utils.
    shapiq_pairwise_interactions()` - the SAME bounded-cost mechanism
    already used by `surrogate_shapiq_interactions()` below and by
    RKHS's/SVR's/KNN's own 'shapiq' routes), whose wall-clock cost is
    governed by `budget` (a fixed ceiling), not by tree depth, tree
    count, or order. This is opt-in via `X_train_background` (below) -
    a caller that omits it (every pre-ver4-30 call site) is completely
    unaffected at any order (I11); a caller that supplies it still only
    takes this new path when `max(orders) > 2`, since order-2 was never
    the reported problem and the exact route remains fast there.

    Parameters
    ----------
    fitted_model, X_explain, marker_names : see `tree_shap_
    interactions()`'s own identical parameters.
    orders : sequence of int, default ``(2,)``
        Forwarded to `pipeline_utils.shapiq_tree_interactions()` (exact
        route) or `pipeline_utils.shapiq_pairwise_interactions()`
        (budgeted route), depending on which is taken - see above.
    index : str, default 'k-SII'
        Forwarded to whichever underlying route is taken.
    max_seconds : float or None, default None
        Requirements.md item 2 (performance): forwarded to
        `pipeline_utils.shapiq_tree_interactions()`'s own identical
        parameter - a wall-clock safety net on top of `models/RF.py`'s
        own depth-capped refit for the EXACT route. `None` (the default)
        preserves previous behaviour exactly (I11). Unused by the new
        budgeted route (below) - `budget` bounds that route's own cost
        directly, with no separate wall-clock parameter needed.
    n_jobs : int, keyword-only, default 1
        Design blueprint R1: forwarded to whichever underlying route is
        taken, fanning the per-row `shapiq` explain loop out across
        `n_jobs` worker processes. ``1`` (the default) preserves previous
        behaviour exactly (I11).
    X_train_background : pandas.DataFrame or None, keyword-only, default None
        ver4-30: background/imputation reference rows for the new
        budgeted route - forwarded to `pipeline_utils.
        shapiq_pairwise_interactions()`'s own `X_train_background`.
        `None` (the default) means this function behaves EXACTLY as
        before this parameter existed (I11): the exact TreeExplainer
        route is always taken, regardless of `orders`. Supplying it
        opts in to the budgeted route, but ONLY actually takes effect
        when `max(orders) > 2` (see above) - a `None`-vs-supplied
        difference at `max(orders) <= 2` is therefore also a no-op.
    budget : int, keyword-only, default 4096
        ver4-30: forwarded to `pipeline_utils.shapiq_pairwise_
        interactions()`'s own `budget` - the fixed coalition-sample
        ceiling that bounds the budgeted route's cost. Unused when the
        exact route is taken (`X_train_background is None` or
        `max(orders) <= 2`).
    approximator : str, keyword-only, default 'auto'
        ver4-30: forwarded to `pipeline_utils.shapiq_pairwise_
        interactions()`'s own identical parameter. Unused by the exact
        route.
    candidate_narrowing : bool, keyword-only, default True
        ver4-30: forwarded to `pipeline_utils.shapiq_pairwise_
        interactions()`'s own identical parameter. Unused by the exact
        route.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'markers',
    'order', 'value']`` - see `canonicalize_shapiq_table()`'s own
    docstring for the exact per-row convention.

    Raises
    ------
    Exception
        Any failure inside `shapiq` itself - deliberately NOT caught
        here; `models/RF.py`'s own dispatch wraps this call in its own
        `try`/`except` and falls back to the proven 'pairwise_shap'
        route (`tree_shap_interactions()`) with a printed NOTE.
    """
    marker_names = list(marker_names)
    max_order = max(orders) if orders else 2
    if max_order > 2 and X_train_background is not None:
        # ver4-30: the budgeted, model-agnostic route - see this
        # function's own module-level note above. `fitted_model.predict`
        # is called directly (no surrogate refit needed - a tree
        # ensemble's own predict is already cheap), so this explains the
        # SAME real, unmodified forest `models/RF.py` fit for prediction.
        raw = shapiq_pairwise_interactions(
            fitted_model.predict, X_train_background, X_explain, marker_names,
            orders=orders, budget=budget, index=index, n_jobs=n_jobs,
            candidate_narrowing=candidate_narrowing, approximator=approximator,
        )
        return canonicalize_shapiq_table(raw, marker_names=marker_names)

    raw = shapiq_tree_interactions(fitted_model, X_explain, marker_names, orders=orders, index=index,
                                    max_seconds=max_seconds, n_jobs=n_jobs)
    return canonicalize_shapiq_table(raw, marker_names=marker_names)


def tree_shap_main_effects(fitted_model, X_explain, marker_names):
    """Exact per-marker TreeSHAP main-effect (additive attribution)
    values for a fitted tree ensemble, aggregated across ``X_explain``'s
    own rows into one ``sum(|phi|)`` value per marker - the SAME
    aggregation convention ``models/KNN.py``'s/``models/SVR.py``'s own
    Shapley marker-effect routes already use.

    Requirements3.md item 2: the shared implementation behind
    RF.py/ExtraTrees.py/XGBoost.py's new selectable 'shap' Marker effect
    method (an alternative to their existing, default
    ``feature_importances_``-based effect) - one implementation, three
    call sites, exactly like ``tree_shap_interactions()`` above already
    is for their pairwise-interaction route.

    Deliberately does NOT route through
    ``pipeline_utils.parallel_shap_values()`` the way
    ``tree_shap_interactions()`` does: TreeSHAP's own main-effect
    computation is already an efficient, polynomial-time EXACT algorithm
    (no coalition sampling), so a single batched ``.shap_values(
    X_explain)`` call computes every row's attribution directly, with no
    ``nsamples`` parameter to plumb through - and
    ``parallel_shap_values()``'s own ``interaction=False`` branch
    specifically assumes a KernelExplainer-style ``nsamples`` kwarg
    (see that function's own docstring/Parameters), which
    ``shap.TreeExplainer.shap_values()`` does not accept, so reusing it
    here would risk a ``TypeError`` for no benefit.

    XGBoost models (duck-typed via ``get_booster``) are routed through
    ``_xgboost_native_shap_values()`` (native ``pred_contribs=True``)
    instead of ``shap.TreeExplainer`` - see that function's own
    docstring for why (the identical ``base_score``-parsing failure
    ``_xgboost_native_interaction_matrices()`` documents, which fires at
    ``TreeExplainer.__init__`` time regardless of which explanation is
    requested afterwards).

    Parameters
    ----------
    fitted_model
        A scikit-learn-compatible fitted tree ensemble (whatever
        ``shap.TreeExplainer`` accepts), or a fitted
        ``xgboost.XGBRegressor``.
    X_explain : pandas.DataFrame
        The already-sampled rows to explain (e.g. via ``shap.sample(...)``
        upstream, exactly like ``tree_shap_interactions()``'s own
        ``X_explain``).
    marker_names : sequence of str
        Column names, in the SAME order as ``X_explain``'s columns.

    Returns
    -------
    1-D ``numpy.ndarray``, length ``len(marker_names)`` - the per-marker
    ``sum(|phi|, axis=0)`` aggregate, in ``marker_names``'s own order.
    An empty (0-row) ``X_explain`` returns an all-zero vector rather than
    raising, matching this module's own "0 explainable rows" convention
    elsewhere (e.g. ``GBLUP.R``'s/``RKHS.R``'s identical edge case).
    """
    marker_names = list(marker_names)
    if X_explain.shape[1] != len(marker_names):
        raise ValueError(
            f"tree_shap_main_effects: X_explain has {X_explain.shape[1]} columns but "
            f"{len(marker_names)} marker_names were given - these must match 1:1."
        )
    if X_explain.shape[0] == 0:
        return np.zeros(len(marker_names))

    if hasattr(fitted_model, 'get_booster'):
        raw = _xgboost_native_shap_values(fitted_model, X_explain)
    else:
        import shap
        explainer = shap.TreeExplainer(fitted_model)
        raw = np.asarray(explainer.shap_values(X_explain))

    return np.abs(raw).sum(axis=0)


def top_select(df, mode, value):
    """Keep only the strongest marker-pair rows of `df` (which must have a
    ``'value'`` column, higher = stronger).

    Parameters
    ----------
    df : pandas.DataFrame
    mode : {'percentage', 'count'}
        - ``'percentage'`` - keeps a RANK-based top ``value``% of `df`
          (``round(df.shape[0] * value / 100)`` rows, by ``.nlargest``),
          restricted to rows with a genuinely nonzero ``'value'`` (see
          "Bug fix" below for why the restriction matters). This
          REPLACES RF.py's original threshold-filter line (``df[df['value']
          > df['value'].quantile(1 - value/100)]``) - see the note below.
        - ``'count'`` - keeps exactly the top ``value`` rows by
          ``.nlargest('value')`` (fewer if ``df`` has fewer rows to begin
          with) - mirrors circos_plot._select_top_interactions()'s own
          ``'count'`` mode exactly, so the model-side cap and the
          plot-side cap speak the same language. Unaffected by this
          function's own ``'percentage'`` bug fix below.
    value : float, int, or 'all'
        The percentage (0-100] or count. ``'all'`` (or ``None``) returns
        `df` unchanged - RF.py's own pre-existing "no filtering" sentinel.

    Returns
    -------
    A filtered, index-reset copy of `df` - never mutates the input.

    Bug fix (quantile-threshold collapse under tied/zero values)
    --------------------------------------------------------------
    The original ``'percentage'`` implementation (``df[df['value'] >
    df['value'].quantile(1 - value/100)]``) silently breaks whenever a
    large share of `df['value']` is TIED at (or near) the computed
    quantile - most commonly at EXACTLY ``0.0``. This is not a rare edge
    case for this module's own callers: every extractor here writes
    ``value=0.0`` for every marker pair it did NOT genuinely evaluate -
    a degenerate (near-constant) marker in ``h_statistic_interactions()``
    (`skip_degenerate`), and, far more consequentially, EVERY unscreened
    pair whenever that same function's own `screen=` pre-filter is
    enabled (`screen_keep` defaults to a mere 2% - see that function's
    own docstring: "every unscreened pair is still emitted, at
    value=0.0"). Confirmed directly (not assumed): with `screen_keep`
    at its own 2% default, `top_select(df, 'percentage', pct)` returned
    the exact same 143-row result for `pct` in {5, 10, 20, 50} - the
    requested percentage was completely ignored the moment it exceeded
    the pre-screen's own live fraction, because ``quantile(1 - pct/100)``
    lands inside (or below) the ~98%-large tied-at-zero block for every
    one of those `pct` values, collapsing the comparison to "value > 0"
    regardless of which `pct` was actually asked for. The circos-side
    twin of this function (``circos_plot._select_top_interactions()``,
    which used ``>=`` instead of ``>``) fails in the OPPOSITE, far more
    visible direction under the exact same condition: once the quantile
    itself resolves to ``0.0``, ``value >= 0.0`` is true for the ENTIRE
    table (every value here is non-negative by construction - see
    ``normalize_task_interactions()``'s own docstring) - confirmed
    directly to jump from a correctly-thresholded selection at `pct=1`
    (below the 2% live fraction) to keeping 100% of the table (all
    7,140/7,140 rows in the reproduction) at `pct` in {5, 10, 50}. This
    is exactly the symptom reported against real runs: models/settings
    that lean on ``h_statistic_interactions(screen=...)`` (or that
    simply happen to have many exactly-additive, i.e. exactly-zero,
    marker pairs even WITHOUT pre-screening) render a wildly different
    - and, for the circos side, wildly LARGER - interaction-link count
    than a TreeSHAP-based extractor given the identical "top N%" setting,
    even though both are meant to obey the same percentage.

    The fix ranks by value instead of thresholding by quantile - `m =
    round(df.shape[0] * value / 100)` rows, taken by ``.nlargest``, is
    ALWAYS approximately `value`% of `df` regardless of how many rows
    tie at any given value (nlargest has no "everything past this point
    is tied" failure mode). It is further capped to `df`'s own count of
    rows with `value > 0`: a pair this module never actually scored (a
    screened-out or degenerate pair, ``value == 0.0`` by construction)
    would otherwise be manufactured into the "top N%" purely to pad the
    requested count out to `m` once the genuinely-scored rows run out -
    drawing an interaction link with NO evidence behind it is worse than
    returning fewer rows than requested. In that situation (screening,
    or an extractor whose true interaction signal is genuinely sparser
    than `value`%, on a data set), this now transparently returns every
    row that WAS scored, capped at `value`% of the full candidate count
    - never more, and never padded with zeros - which also means a
    smaller `value`% than what pre-screening already kept is still
    honoured exactly (unaffected by this fix - the previous behaviour
    was already correct in that direction; only `value`% settings AT OR
    ABOVE the live/nonzero fraction were the ones silently ignored).

    Bug fix (round-to-zero starvation on small candidate tables)
    --------------------------------------------------------------
    ``m = round(df.shape[0] * value / 100)`` rounds DOWN to ``0`` whenever
    `df` has fewer than roughly ``50 / value`` rows - e.g. under 50,000
    rows at the GUI's own circos default of ``value=0.001`` - even when
    `value` is genuinely positive and `nonzero` contains real, scored
    interactions. This was invisible for the exhaustive
    `pairwise_shap`/`friedman_h` routes, whose shortlist-scan candidate
    tables are large enough (~124,750 rows at the default 500-marker
    shortlist) to clear that threshold, but a shapiq-based route
    (`shapiq_tree`, `shapiq`, `surrogate_shapiq`) is inherently
    budget-bounded or depth-capped and routinely returns only tens to a
    few hundred rows per order - reliably landing below the threshold and
    silently returning ZERO rows despite genuinely nonzero interactions
    having been found. `top_select_by_order()`'s per-order splitting
    (shapiq-only, since only shapiq routes carry an ``'order'`` column)
    shrinks the per-group row count further, making this specifically
    and disproportionately a shapiq-route failure. The fix floors `m` at
    1 whenever a positive `value` was requested and at least one
    genuinely-scored (nonzero) row exists, so "some positive percentage
    of a nonempty, real result" never rounds down to "nothing shown".
    """
    if value == 'all' or value is None:
        return df.reset_index(drop=True)
    if mode == 'count':
        m = max(0, min(int(value), df.shape[0]))
        return df.nlargest(m, 'value').reset_index(drop=True)
    if mode == 'percentage':
        # Requirements3.md item 3: this filter was accidentally left
        # commented out (`nonzero = df` instead of `df[df['value'] > 0]`),
        # silently re-enabling exactly the "manufactured top-N% padding"
        # failure mode the docstring above describes as fixed - once the
        # genuinely-scored (nonzero) rows run out, `.nlargest(m, 'value')`
        # below pads the result out to `m` rows with arbitrarily-chosen,
        # never-actually-evaluated `value == 0.0` rows instead of returning
        # fewer rows than requested. This is what let a shapiq-based route's
        # (or a heavily pre-screened Friedman's-H route's) interaction table
        # get swamped by manufactured zero-value pairs with no evidence
        # behind them - restoring the filter here is this bug's actual fix.
        nonzero = df[df['value'] > 0]
        m = int(round(df.shape[0] * (value / 100)))
        if m == 0 and value > 0 and nonzero.shape[0] > 0:
            # Bug fix (round-to-zero starvation) - see docstring above.
            m = 1
        m = max(0, min(m, nonzero.shape[0]))
        return nonzero.nlargest(m, 'value').reset_index(drop=True)
    raise ValueError(f"top_select: mode must be 'percentage' or 'count', got {mode!r}.")


def top_select_by_order(df, mode, value):
    """ver4-25, Stage 9 (design blueprint R12): the SAME top-N%/top-count
    filter as `top_select()` above, applied PER ORDER when `df` carries
    an ``'order'`` column (a shapiq-based route that searched more than
    one interaction order - see `canonicalize_shapiq_table()`), rather than
    once across every order mixed together.

    This matters because k-SII (and every other shapiq interaction
    index) magnitudes SHRINK with order - a shared top-N% threshold
    computed across a mixed order-2/order-3/... table would keep almost
    exclusively order-2 rows and drop nearly every higher-order set,
    even when the higher-order sets are exactly as significant relative
    to their OWN order's distribution (design blueprint R12's own
    stated rationale for per-order selection). Filtering each order's
    own rows against `top_select()` independently, then concatenating,
    keeps every order comparable to itself.

    A `df` with no ``'order'`` column (every non-shapiq route, and a
    shapiq route that only ever searched order 2 - `orders=(2,)`, the
    default) is passed straight through to `top_select()` unchanged -
    byte-identical to this function not existing at all (I11).

    Parameters
    ----------
    df, mode, value : see `top_select()`'s own identical parameters.

    Returns
    -------
    A filtered, index-reset copy of `df` - never mutates the input.
    """
    if 'order' not in df.columns:
        return top_select(df, mode, value)
    if df.empty:
        return df.reset_index(drop=True)
    parts = [
        top_select(group, mode, value)
        for _, group in df.groupby('order', sort=True)
    ]
    if not parts:
        return df.iloc[:0].reset_index(drop=True)
    return pd.concat(parts, ignore_index=True)


def pad_shapiq_pairs_to_shap_parity(interaction_sample, marker_names, threshold, *, random_state=0):
    """ver4-27: pad a `shapiq`-based order-2 interaction table with
    explicit ``value=0.0`` rows for candidate marker pairs this task did
    NOT end up selecting, so its row count becomes directly comparable to
    what the SAME task's exact 'pairwise_shap' route would report at the
    IDENTICAL `threshold`% setting - by explicit request (a `shapiq`
    route was found to report a much smaller "total number of extracted
    interactions" than 'pairwise_shap' on the SAME fitted forest, and a
    row-for-row/count-for-count comparison between the two routes' own
    ``Interaction.csv`` output was wanted more than a strictly minimal
    file).

    Why this is safe alongside `top_select()`'s own "no manufactured
    zero-value padding" fix
    --------------------------------------------------------------------
    `top_select()`'s own docstring (above) documents a real, previously-
    fixed bug: `.nlargest(m, 'value')` silently padding a "top N%"
    SELECTION out to `m` rows with never-actually-scored, `value == 0.0`
    pairs once the genuinely-scored ones ran out - manufacturing evidence
    for interactions no route ever actually found. This function does
    NOT reopen that bug: it is called strictly AFTER `top_select_by_
    order()` has already run (see `models/RF.py`'s own 'shapiq_tree' call
    site), so every row already in `interaction_sample` when this
    function is called is a row `top_select()` itself judged worth
    keeping - nothing here changes what counts as "selected" for ranking/
    circos purposes. The rows THIS function adds are appended afterward,
    explicitly at `value=0.0`, and are trivially distinguishable from a
    genuine finding by that same `value > 0` test - this is a transparent
    "this pair was not considered/selected" marker for comparability, not
    a second, hidden selection pass.

    Parameters
    ----------
    interaction_sample : pandas.DataFrame
        The already `top_select_by_order()`-filtered table (columns
        include at least ``'marker1'``, ``'marker2'``, ``'markers'``,
        ``'order'``, ``'value'`` - `canonicalize_shapiq_table()`'s own
        shape). Rows of ``order != 2`` (a higher-order `shapiq_tree`
        search) are passed through UNCHANGED - Interaction.csv only ever
        takes `order == 2` rows (`genomic_prediction.py`'s own Step 10
        split); there is no 'pairwise_shap' higher-order counterpart to
        be comparable WITH, so padding those would have no reference
        point and is deliberately out of scope here.
    marker_names : sequence of str
        The SAME top-importance marker shortlist 'pairwise_shap' itself
        searches for this task (`interaction_test_x.columns` at this
        function's own RF.py call site) - the candidate pair universe
        both routes are being made comparable against.
    threshold : float, 'all', or None
        This task's own configured top-N% setting - the SAME value
        already passed to `top_select_by_order()` above. The padding
        target is computed with the IDENTICAL formula `top_select()`
        applies to a fully-dense candidate table (`round(C(M,2) *
        threshold/100)`), which is what 'pairwise_shap' effectively
        selects from its own (generically all-nonzero) dense output at
        this same threshold.
    random_state : int, keyword-only, default 0
        Seeds which of the (usually far more numerous) un-selected
        candidate pairs are drawn for padding - deterministic per-task,
        so re-running an unchanged task reproduces an identical padded
        file (I11-style reproducibility, matching this codebase's other
        seeded sampling steps).

    Returns
    -------
    pandas.DataFrame
        `interaction_sample` with additional ``value=0.0``, ``order=2``
        rows appended (marker1/marker2 canonicalised ascending, matching
        `canonicalize_shapiq_table()`'s own convention) until its
        `order == 2` row count reaches the target above, or until the
        full candidate pool is exhausted - whichever comes first (never
        pads past ``C(len(marker_names), 2)`` rows, and never removes or
        reorders any existing row). Unchanged (a no-op copy) if
        `interaction_sample` already has at least as many `order == 2`
        rows as the target, or if `marker_names` has fewer than 2 names.
    """
    marker_names = list(marker_names)
    m = len(marker_names)
    total_pairs = m * (m - 1) // 2
    if total_pairs == 0 or interaction_sample is None:
        return interaction_sample

    has_order_col = 'order' in interaction_sample.columns
    order2 = interaction_sample[interaction_sample['order'] == 2] if has_order_col else interaction_sample
    other = interaction_sample[interaction_sample['order'] != 2] if has_order_col else interaction_sample.iloc[:0]

    if threshold == 'all' or threshold is None:
        target = total_pairs
    else:
        target = max(0, min(int(round(total_pairs * (float(threshold) / 100.0))), total_pairs))

    n_present = order2.shape[0]
    if n_present >= target:
        return interaction_sample

    present_pairs = set(zip(order2['marker1'], order2['marker2'])) if n_present else set()
    idx_i, idx_j = np.triu_indices(m, k=1)
    names_arr = np.array(marker_names)
    # Canonicalised ascending (marker1 <= marker2), matching
    # `_canonicalize_marker_pairs()`'s own convention - so a padded row
    # and a genuinely-computed row for the SAME pair can never both
    # appear under swapped marker1/marker2 labels.
    all_pairs = [tuple(sorted((a, b))) for a, b in zip(names_arr[idx_i], names_arr[idx_j])]
    candidates = [p for p in all_pairs if p not in present_pairs]

    rng = np.random.RandomState(random_state)
    rng.shuffle(candidates)
    pad_pairs = candidates[:target - n_present]
    if not pad_pairs:
        return interaction_sample

    pad_df = pd.DataFrame({
        'marker1': [p[0] for p in pad_pairs],
        'marker2': [p[1] for p in pad_pairs],
        'markers': [tuple(p) for p in pad_pairs],
        'order': 2,
        'value': 0.0,
    })
    padded_order2 = pd.concat([order2, pad_df], ignore_index=True)
    if has_order_col:
        return pd.concat([padded_order2, other], ignore_index=True)
    return padded_order2


def shortlist_markers(importance, k):
    """Positional indices of the top-`k` markers by |importance|, sorted
    back into their ORIGINAL position order (not by rank) - every caller
    below uses this to decide which markers are even eligible to appear
    in a pairwise interaction search, exactly as ``models/RF.py`` (and,
    before it, ``models/SVR.py``/``models/KNN.py`` for their own Shapley
    *effect* shortlists) already narrows to a top-importance shortlist
    before doing anything O(M^2) - this is the shared version of that
    same idea, usable for interactions too.

    Parameters
    ----------
    importance : array-like, shape (n_markers,)
        Any per-marker importance score - absolute correlation with the
        trait, feature_importances_, etc. Only the ABSOLUTE VALUE is
        used for ranking (a large negative correlation is just as
        informative a shortlist candidate as a large positive one).
    k : int, or 'all'/None
        How many markers to keep. ``'all'`` or ``None`` (or a `k` that
        is not smaller than the number of markers) keeps every marker,
        in original order - the "no shortlisting" sentinel every other
        model in this codebase already uses.

    Returns
    -------
    numpy.ndarray of int - positional indices into `importance` (and,
    by construction, into whatever DataFrame's columns `importance` was
    computed from), sorted ascending.
    """
    importance = np.asarray(importance, dtype=float).ravel()
    n_markers = importance.shape[0]
    if k == 'all' or k is None:
        return np.arange(n_markers)
    k = int(k)
    if k >= n_markers:
        return np.arange(n_markers)
    if k <= 0:
        return np.array([], dtype=int)
    order = np.argsort(np.abs(importance))[::-1][:k]
    return np.sort(order)


def should_use_effect_for_interaction_shortlist(use_effect_flag, max_interaction_features,
                                                  effect_marker_cap):
    """Single-authority decision for the 'Use marker effects to select
    markers for interaction search?' checkbox (RF/ExtraTrees/XGBoost/SVR/
    KNN/RKHS/MLP): should this task's interaction-marker shortlist be taken
    from this model's own already-computed marker-EFFECT values instead of
    recomputing a fresh correlation-with-trait (or model-importance)
    ranking purely for the interaction step?

    This is deliberately NOT just "the checkbox is checked" - reusing the
    effect values is only actually possible when they were computed widely
    enough to cover the interaction shortlist's own size. Every caller is
    expected to hide the checkbox's own widget in the GUI whenever this
    function would return False purely because of the count comparison
    (see main_app.py) - the checkbox is offered to the user only when
    ticking it would do something - but this function is the one place
    that comparison is actually made, so the GUI's decision to show/hide
    the widget and the engine's decision to honour it can never disagree.

    Parameters
    ----------
    use_effect_flag : bool
        The checkbox's own current value.
    max_interaction_features : int, or 'all'
        This task's own 'Max markers considered for interaction search'
        setting.
    effect_marker_cap : int, 'all', or None
        How many markers this task's own marker-EFFECT step actually
        covers - 'all' for an uncapped/native effect method (feature_
        importance, beta_direct, closed_form, deep_shap, kernel_shap with
        no shortlist, etc.), an int for a capped sampling-based method
        (shapiq, or a correlation-shortlisted Shapley effect), or None
        when marker effects were not computed at all this task (the
        model's own 'Return marker effect?' toggle is off) - in which
        case there is nothing to reuse regardless of the checkbox.

    Returns
    -------
    bool
    """
    if not use_effect_flag or effect_marker_cap is None:
        return False
    if effect_marker_cap == 'all':
        return True
    if max_interaction_features == 'all':
        # Interaction search wants every marker, but the effect step only
        # covers a capped subset - not wide enough to substitute for 'all'.
        return False
    return int(max_interaction_features) <= int(effect_marker_cap)


def _resolve_marker_grid(col, *, resolution=3, discrete_max=10, percentile_range=(0.0, 1.0)):
    """Update ID ver4-6, R1b (blueprint §2.10.2) - the single authority
    every grid this module builds (the exact H^2 search's own grid
    cache below, AND ``_marginal_partial_dependences()``'s screen route)
    resolves a marker's own grid THROUGH, so the exact stage and the
    approximate screen can never disagree about what a marker's grid is.

    ROOT CAUSE THIS REPLACES (blueprint §2.10.1): the pre-ver4-6 grid was
    ``np.unique(col)[:3]`` - the three numerically SMALLEST distinct
    values, which silently drops a genuine genotype class whenever any
    other value (e.g. a single mean-imputed fractional call - see
    ``genomic_prediction.py``'s own missing-genotype imputation) sorts
    below it. Measured impact (blueprint §2.10.1): a single mean-imputed
    cell in an otherwise clean 0/1/2 column already drops genotype class
    2 entirely; a RIL/dosage panel's shipped grid can span under 1% of a
    marker's own true range, corrupting the H^2 ranking circos draws
    links from.

    Four branches, evaluated in order:

      1. ``n_unique <= resolution`` - use the OBSERVED unique values
         directly. BIT-IDENTICAL to ver4-5's own ``np.unique(col)[:3]``
         whenever ``n_unique <= 3`` (the only regime ver4-5 was ever
         correct for, by construction) - this is what preserves AC1.1.
      2. ``n_unique <= discrete_max`` - the ``resolution`` MOST FREQUENT
         distinct values, sorted ascending. This is what recovers
         ``{0, 1, 2}`` through mean-imputation contamination: a
         hard-called column with a handful of stray fractional cells
         still has 0/1/2 as its three most frequent values by a wide
         margin, so ranking by FREQUENCY (not by numeric order) finds
         the real genotype classes instead of whatever three values
         happen to sort lowest.
      3. Otherwise - a quantile grid,
         ``unique(quantile(col, linspace(p_lo, p_hi, resolution)))`` -
         genuinely continuous/dosage data, where the grid should follow
         the column's own observed distribution.
      4. If branch 3 collapses to fewer than 2 distinct points (possible
         under heavy skew, e.g. a MAF-skewed dosage column) - a plain
         linearly-spaced grid over ``[col.min(), col.max()]``, which
         cannot collapse the same way.

    Parameters
    ----------
    col : numpy.ndarray, 1-D
        One marker's own BACKGROUND column values (the already-sampled
        background rows ``h_statistic_interactions()`` itself uses, not
        the whole training set).
    resolution : int, default 3
        Target grid size - ``G = resolution_i * resolution_j`` grid
        cells per pair, matching this codebase's own 0/1/2 genotype
        coding (3 classes) by default. Clamped to >= 2 by the caller
        (a 1-point grid makes H^2 undefined).
    discrete_max : int, default 10
        Branch 2's own cardinality ceiling.
    percentile_range : (float, float), default (0.0, 1.0)
        Branch 3's own quantile span.

    Returns
    -------
    (grid, branch) - ``grid`` a numpy.ndarray of up to `resolution`
    ascending distinct values; ``branch`` an int in {1, 2, 3, 4} naming
    which rule fired, used only for the coding-regime log line
    (``_classify_marker_coding()``, Layer B).
    """
    unique_vals, counts = np.unique(col, return_counts=True)
    n_unique = unique_vals.size

    if n_unique <= resolution:
        return unique_vals, 1

    if n_unique <= discrete_max:
        order = np.argsort(counts)[::-1][:resolution]
        grid = np.sort(unique_vals[order])
        return grid, 2

    p_lo, p_hi = percentile_range
    quantiles = np.linspace(p_lo, p_hi, resolution)
    grid = np.unique(np.quantile(col, quantiles))
    if grid.size >= 2:
        return grid, 3

    grid = np.unique(np.linspace(float(col.min()), float(col.max()), resolution))
    return grid, 4


def _classify_marker_coding(background_values, involved_cols, *, resolution=3,
                             discrete_max=10, percentile_range=(0.0, 1.0)):
    """Per-branch marker counts across `involved_cols` (Update ID
    ver4-6, R1b) - Layer B's own unconditional coding-regime log line
    uses this so a person can tell, on every run, whether their data is
    genuinely hard-called (branch 1 only - `grid='auto'` is then a
    byte-for-byte no-op) or has some contamination/dosage regime
    `grid='auto'` is actively correcting for (branches 2-4present).
    Never raises; a marker whose grid can't be resolved for any reason
    is simply not counted (defensive - this is a diagnostic line, not a
    contract)."""
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for c in involved_cols:
        try:
            _, branch = _resolve_marker_grid(
                background_values[:, c], resolution=resolution, discrete_max=discrete_max,
                percentile_range=percentile_range,
            )
            counts[branch] += 1
        except Exception:
            continue
    return counts


_VALID_GRID_MODES = ('auto', 'observed')
_VALID_SCREEN_MODES = (None, 'marginal_pd', 'surrogate_shap')


def _marginal_partial_dependences(predict_fn, background_values, marker_names, grids,
                                   involved_cols, n_bg):
    """Update ID ver4-6, R1 Layer C (blueprint §2.2): per-marker CENTRED
    marginal partial dependence, one batched ``predict_fn`` call PER
    MARKER (``grid_c.size * n_bg`` rows each - ``~3 * n_bg`` rows at the
    default `grid_resolution`, matching the blueprint's own ``3 * B * K``
    total-cost accounting), built through the SAME
    ``_resolve_marker_grid()`` resolver the exact H^2 stage uses - the
    screen must rank pairs on the SAME grid the exact stage would
    evaluate them on, or it ranks candidates on a different (possibly
    broken, pre-R1b-style) sliver than what actually gets computed
    (blueprint §2.10.2: "the Layer C screen inherits this").

    Used only by ``_screen_pairs(mode='marginal_pd')`` below - never
    called when screening is off (the default), so this adds no cost to
    the exact-only path.

    Returns
    -------
    dict {column index: centred marginal PD array (length grids[c].size)}
    """
    pd_by_col = {}
    for c in involved_cols:
        g = grids[c]
        if g.size < 1:
            pd_by_col[c] = np.zeros(1)
            continue
        buf = np.tile(background_values, (g.size, 1))
        buf[:, c] = np.repeat(g, n_bg)
        stacked_df = pd.DataFrame(buf, columns=marker_names, copy=False)
        preds = np.asarray(predict_fn(stacked_df), dtype=float).ravel()
        f_c = preds.reshape(g.size, n_bg).mean(axis=1)
        pd_by_col[c] = f_c - f_c.mean()
    return pd_by_col


def _screen_pairs(mode, pairs, *, predict_fn, background_values, marker_names, grids,
                   n_bg, screen_keep):
    """Update ID ver4-6, R1 Layer C (blueprint §2.2, RK-4): rank `pairs`
    by a CHEAP proxy and keep only the top `screen_keep`, so the
    EXPENSIVE exact H^2 stage only ever runs on a small, high-signal
    shortlist instead of every candidate pair - this is where the
    order-of-magnitude gain over Layer A's exact micro-optimisations
    (~1.5x, measured) comes from (blueprint §1: "the order-of-magnitude
    gain comes from an opt-in, RK-8-disclosed two-stage screen").

    APPROXIMATE, OFF BY DEFAULT (RK-4): screened output must never be
    mistaken for an exhaustive interaction table - every caller of
    ``h_statistic_interactions(screen=...)`` prints a log line naming
    'APPROXIMATE' and the screen mode, and the GUI help text for this
    field names the ``marginal_pd`` blind spot explicitly (see
    ``hparam_specs.py``'s own new field for this run's model).

    Parameters
    ----------
    mode : {'marginal_pd', 'surrogate_shap'}
        - ``'marginal_pd'`` (cheap, the default when screening is
          enabled) - ``score(i, j) = ptp(centre(f_i)) * ptp(centre(f_j))``
          from ``_marginal_partial_dependences()``. Known blind spot
          (disclosed in every caller's log line/GUI help, same
          convention as ``surrogate_h_statistic_interactions()``'s own
          RK-8 disclosure): a pure-XOR pair (both marginals flat, strong
          JOINT effect) ranks last and is missed.
        - ``'surrogate_shap'`` (costlier, no marginal-only blind spot) -
          fits a small ``RandomForestRegressor`` to `predict_fn`'s own
          outputs over the background rows, then ranks pairs by THIS
          module's own ``tree_shap_interactions()`` on that surrogate
          (reused, never re-implemented - the same "one shared
          implementation" discipline this module's own docstring
          describes for every other extractor here).
    pairs : list of (int, int)
        The LIVE (non-degenerate) candidate pairs to rank.
    screen_keep : float or int
        Fraction (0, 1] of `pairs` to keep when a `float`, or an exact
        count when an `int`. Clamped to at least 1 pair, with a log line
        naming the clamp, if it would otherwise resolve to 0.

    Returns
    -------
    (kept_pairs, provenance) - `kept_pairs` a list of (int, int), a
    subset of `pairs`; `provenance` a short string naming `mode`, for
    the caller's own log line.

    Raises
    ------
    ValueError
        If `mode` is not one of `_VALID_SCREEN_MODES` (minus `None`,
        which never reaches this function - the caller skips screening
        entirely when `screen is None`).
    """
    if mode not in ('marginal_pd', 'surrogate_shap'):
        raise ValueError(
            f"h_statistic_interactions: screen must be one of {_VALID_SCREEN_MODES}, got {mode!r}."
        )

    involved = sorted({c for pr in pairs for c in pr})

    if mode == 'marginal_pd':
        pd_by_col = _marginal_partial_dependences(
            predict_fn, background_values, marker_names, grids, involved, n_bg,
        )
        scores = {
            (i, j): float(np.ptp(pd_by_col[i]) * np.ptp(pd_by_col[j])) for (i, j) in pairs
        }
    else:  # 'surrogate_shap'
        sub_names = [marker_names[c] for c in involved]
        col_to_local = {c: k for k, c in enumerate(involved)}
        X_explain = pd.DataFrame(background_values[:, involved], columns=sub_names)
        y_bg = np.asarray(predict_fn(
            pd.DataFrame(background_values, columns=marker_names)
        ), dtype=float).ravel()
        from sklearn.ensemble import RandomForestRegressor
        surrogate = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=0)
        surrogate.fit(X_explain, y_bg)
        shap_table = tree_shap_interactions(surrogate, X_explain, sub_names, reduce='abs_then_sum')
        name_to_col = {name: col_to_local[c] for c, name in zip(involved, sub_names)}
        score_lookup = {}
        for _, row in shap_table.iterrows():
            a = name_to_col.get(row['marker1'])
            b = name_to_col.get(row['marker2'])
            if a is not None and b is not None:
                score_lookup[frozenset((involved[a], involved[b]))] = float(row['value'])
        scores = {(i, j): score_lookup.get(frozenset((i, j)), 0.0) for (i, j) in pairs}

    n_pairs = len(pairs)
    if isinstance(screen_keep, float):
        keep_n = int(round(n_pairs * screen_keep))
    else:
        keep_n = int(screen_keep)
    if keep_n < 1:
        print(f"[interaction_extraction] NOTE: screen_keep resolved to 0 pairs out of {n_pairs} - "
              f"clamped to 1 (keeping at least the single strongest-screened pair).")
        keep_n = 1
    keep_n = min(keep_n, n_pairs)

    ranked = sorted(pairs, key=lambda pr: scores.get(pr, 0.0), reverse=True)
    return ranked[:keep_n], mode


def h_statistic_interactions(predict_fn, X, marker_names, *, pairs,
                              n_background=100, grid='auto', grid_resolution=3,
                              grid_discrete_max=10, grid_percentile_range=(0.0, 1.0),
                              degenerate_atol=0.0, n_jobs=1,
                              batch_bytes=32 * 1024 ** 2, skip_degenerate=True,
                              screen=None, screen_keep=0.02, cost_log=True,
                              pair_predict_fn=None):
    """Friedman's pairwise H^2 statistic (Friedman & Popescu, 2008) -
    fully model-agnostic (needs only a ``predict_fn`` callable), which is
    what makes it the route for SVR/KNN, where ``shap.KernelExplainer``
    offers no ``shap_interaction_values`` equivalent (confirmed by
    introspection - see the ver4-5 blueprint's Appendix B2).

    Update ID ver4-6, R1/R1b (blueprint §2): this call now runs through
    four layers, in order, every one of them keyword-only and defaulted
    so an UNCHANGED call site (``grid``/`batch_bytes`/`skip_degenerate`
    at their new defaults) reproduces ver4-5's numbers exactly wherever
    ver4-5 was CORRECT (I11, AC1.1) - see each layer's own note below
    for what, if anything, is an intentional exception.

      - Layer A (exact, always on) - a per-marker grid CACHE (`K` calls
        to the grid resolver, not `2P`), a DEGENERATE-pair skip (a pair
        with a near-constant marker needs no predict call at all - H^2
        is PROVABLY zero, blueprint §2.1 F4), and BYTE-BUDGETED,
        shape-bucketed BATCHING (one ``predict_fn`` call per batch of
        same-shape pairs, not one per pair - blueprint §2.1 F2/F5/F6).
        Bit-identical to the pre-ver4-6 per-pair loop whenever
        `grid='observed'` (or `grid='auto'` resolves every involved
        marker through branch 1 - see `_resolve_marker_grid()`).
      - Layer B (exact, always on) - an unconditional pre-flight cost
        line (pair/degenerate counts, projected predicted rows) and a
        measured-throughput line once this call's own work has run -
        informs, never changes what is computed.
      - Layer C (approximate, OFF by default) - `screen` restricts the
        EXPENSIVE exact stage to a cheaply-ranked shortlist; every
        unscreened pair is still emitted, at `value=0.0`, so the
        returned table's row count/schema never changes.
      - Layer D (`pair_predict_fn`) - accepted for forward
        compatibility with an optional model-aware fast-path hook
        (blueprint §2.2 Layer D, an `[INVESTIGATE]` item) but NOT YET
        IMPLEMENTED in this delivery - the go/no-go spike this would
        require (agreement `<=1e-10` relative AND `>=5x` on SVR at
        K=500) was not run, so shipping an unvalidated fast path here
        would risk exactly the silent-wrong-number failure mode this
        whole module exists to avoid. Passing a non-``None`` value logs
        a NOTE that it is currently ignored, rather than silently doing
        nothing with no explanation.

    R1b (non-integer marker coding, blueprint §2.10): `grid='auto'`
    (the new default) resolves every marker's own grid through
    `_resolve_marker_grid()`'s four-branch rule instead of the pre-ver4-6
    ``np.unique(col)[:3]`` (three numerically SMALLEST values - silently
    drops a genotype class whenever any other value sorts below it, e.g.
    a single mean-imputed fractional call). `grid='observed'` reproduces
    ver4-5's exact behaviour, including that defect, for anyone who needs
    to regenerate a historical result verbatim (AC1b.7).

    Unusually cheap on this codebase's data: markers are coded 0/1/2, so
    a pair's joint partial-dependence grid has at most 9 cells (3x3),
    not a continuous quadrature - each pair costs exactly ONE batched
    ``predict_fn`` call over ``9 * len(background)`` rows, rather than a
    Python loop per grid cell.

    Method (restricted to the resolved grid, not a continuous integral -
    this is what keeps the cost bounded on this data):

      1. For marker pair (i, j), build the Cartesian-product grid of
         marker i's and marker j's own resolved grid values (see `grid`
         above - up to `grid_resolution` values each).
      2. For every grid cell ``(v_i, v_j)``, compute the joint partial
         dependence ``F_ij(v_i, v_j)`` as the MEAN of
         ``predict_fn(background rows with columns i, j overridden to
         v_i, v_j)`` - batched across every live pair sharing that
         grid's shape (Layer A), not one ``predict_fn`` call per pair.
      3. The single-marker partial dependences ``F_i(v_i)``/``F_j(v_j)``
         are read off the SAME joint grid (averaged over the other
         marker's grid values), rather than computed separately - this
         keeps the grid density identical between the joint and marginal
         terms, which the H^2 formula requires to be well-defined.
      4. All three functions are CENTRED (their own grid mean
         subtracted) before combining - a partial dependence function is
         only defined up to an additive constant, so centring is
         required for H^2 to be well-defined and bounded; this is
         standard practice, not a simplification unique to this
         implementation.
      5. ``H^2 = sum_grid[(F_ij - F_i - F_j)^2] / sum_grid[F_ij^2]``
         (both terms computed on the centred functions from step 4).

    Parameters
    ----------
    predict_fn : callable(numpy.ndarray of shape (n_rows, n_features)) -> array-like of shape (n_rows,)
        A plain prediction function - e.g. a fitted ``SVR``/
        ``KNeighborsRegressor``'s own ``.predict`` bound method. Never
        fits anything itself. MUST BE ROW-INDEPENDENT (Update ID
        ver4-6, R1, documented contract addition): the prediction for a
        row must not depend on which other rows are in the same call -
        every one of this module's four call sites (scikit-learn/cuML
        ``predict``) already satisfies this; a future caller that does
        not must pass `batch_bytes` small enough to force one pair per
        call.
    X : pandas.DataFrame
        BACKGROUND rows used to average out every marker other than the
        pair currently being evaluated. This function does no sampling
        of its own - pass an already-sampled background (mirroring
        every other extractor in this module).
    marker_names : sequence of str
        Column names, in the SAME order as `X`'s columns.
    pairs : iterable of (int, int)
        POSITIONAL index pairs (into `marker_names`/`X`'s columns) to
        evaluate - REQUIRED, no default. Every extractor in this module
        requires an explicit shortlist rather than defaulting to every
        possible pair, since that is O(M^2) predict-function calls
        (mirrors ``models/RF.py``'s own long-standing rationale comment
        for the exact same reason).
    n_background : int, default 100
        How many rows of `X` to actually use for averaging (a random
        sample, ``random_state=0``, taken once if `X` has more rows than
        this - the SAME `n_background` background is reused for every
        pair, so results are directly comparable to one another).
    grid : {'auto', 'observed'}, default 'auto'
        Update ID ver4-6, R1b - CHANGED default (was `'observed'`,
        the ONLY value ver4-5 supported). `'auto'` resolves every
        marker's grid through `_resolve_marker_grid()`'s four-branch
        rule; `'observed'` reproduces ver4-5's ``np.unique(col)[:3]``
        verbatim, including its defect on non-integer-coded data
        (AC1b.7 - the deliberate escape hatch for regenerating a
        historical result).
    grid_resolution : int, default 3
        Target points per marker (``G = resolution_i * resolution_j``
        grid cells per pair) - `3` reproduces today's exact row count.
        Clamped to `>= 2` (logged once if clamped - a 1-point grid makes
        H^2 undefined, zero denominator).
    grid_discrete_max : int, default 10
        `_resolve_marker_grid()`'s own branch-2 cardinality threshold.
    grid_percentile_range : (float, float), default (0.0, 1.0)
        `_resolve_marker_grid()`'s own branch-3 quantile span.
    degenerate_atol : float, default 0.0
        A pair is skipped (H^2 = 0, no predict call) whenever either
        marker's own BACKGROUND column has ``np.ptp(col) <=
        degenerate_atol`` - `0.0` is EXACTLY equivalent to the pre-ver4-6
        ``grid.size < 2`` test for hard-called data (``ptp == 0`` iff
        monomorphic), while also catching a near-constant DOSAGE column
        `grid.size < 2` would miss (blueprint §2.10.2). Only applied
        when `skip_degenerate` is True.
    n_jobs : int, default 1
        Fans the LIVE-pair workload out across ``n_jobs`` worker
        PROCESSES via ``joblib.Parallel``, dispatching approximately
        ``4 * n_jobs`` pair BLOCKS (Update ID ver4-6, R1 Layer A F5 -
        CHANGED from one joblib task per PAIR, which at `P=124,750`
        made dispatch/serialisation overhead a first-order cost on its
        own), not `P` individual tasks. ``<=1`` (or ``0``/``None``), or
        fewer than 2 live pairs, skips the fan-out entirely and
        evaluates every pair serially in the calling process - a
        byte-for-byte no-op relative to the serial path either way.
        Falls back to the same serial evaluation on ANY exception
        during the parallel fan-out - never raises for this reason;
        correctness never depends on which path actually ran.
    batch_bytes : int, default 33554432 (32 MiB)
        Update ID ver4-6, R1 Layer A - the byte budget for one batched
        ``predict_fn`` call within a single (marker-count, grid-shape)
        bucket. Clamped to `[1 MiB, 256 MiB]`. MUST be expressed in
        BYTES, never rows (blueprint §2.2's own "Phase 2 trap": a
        row-count budget silently scales with the marker shortlist size
        and, measured at `K=500`, produced an 800 MB buffer that ran
        3x SLOWER than the unbatched per-pair path it was meant to
        speed up).
    skip_degenerate : bool, default True
        Update ID ver4-6, R1 Layer A F4 - whether to apply the
        `degenerate_atol` skip above. Value-preserving PROVIDED
        degeneracy is judged against every row this call was given
        (`X`) rather than the `n_background`-sized draw from it used for
        PD averaging - a marker genuinely monomorphic across `X` makes
        H^2 exactly zero either way (blueprint §2.1 F4's proof), so
        leaving this on changes no correct output for such a marker,
        only how many `predict_fn` calls happen. Bugfix note: an earlier
        revision judged degeneracy (and resolved the grid) from
        `background_values` alone, which - for RF's own default of only
        30 background rows - could and did misclassify a real,
        skewed-allele-frequency QTL marker as "monomorphic" purely
        because its minor genotype class was absent from that one small
        draw, forcing H^2 = 0.0 for every pair touching it regardless of
        true effect size. Both the grid cache and this check are now
        resolved from `X` in full; `background_values` is used only for
        the PD averaging itself. `skip_degenerate=False` exists only for
        A/B diagnosis of the (now-correct) claim above.
    screen : {None, 'marginal_pd', 'surrogate_shap'}, default None
        Update ID ver4-6, R1 Layer C - when not `None`, restricts the
        EXACT stage to the top `screen_keep` LIVE pairs by a cheap proxy
        ranking (see `_screen_pairs()`); every other live pair is
        emitted at `value=0.0`. APPROXIMATE (RK-4) - OFF by default;
        every caller's own log line names 'APPROXIMATE' and the mode
        when this is enabled (AC1.6).
    screen_keep : float or int, default 0.02
        Fraction (0, 1] of live pairs to keep when `screen` is set and
        this is a `float`; an exact pair COUNT when an `int`.
    cost_log : bool, default True
        Update ID ver4-6, R1 Layer B - print the unconditional pre-flight
        cost line and the post-hoc measured-throughput line. `False`
        silences both (e.g. for a caller running this inside a tight
        hyperparameter-search loop that already logs its own summary).
    pair_predict_fn : callable or None, default None
        Reserved for the optional Layer D model-aware fast path
        (blueprint §2.2 Layer D) - NOT YET IMPLEMENTED (see the Layer D
        note above). Accepted so a future caller's signature does not
        need to change again once that spike is run; a non-``None``
        value currently only produces a log NOTE, with no effect on
        what is computed.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']`` -
    one row per pair in `pairs`, UNCHANGED schema/row-count regardless of
    `screen` (H^2, unbounded above in principle on a coarse grid, but in
    practice a relative RANKING statistic - callers apply their own
    ``top_select()`` afterwards exactly as every other extractor's
    caller does).

    Raises
    ------
    ValueError
        If `grid` is not one of `_VALID_GRID_MODES`, `screen` is not one
        of `_VALID_SCREEN_MODES`, or `marker_names` does not match `X`'s
        column count.
    """
    if grid not in _VALID_GRID_MODES:
        raise ValueError(f"h_statistic_interactions: grid must be one of {_VALID_GRID_MODES}, got {grid!r}.")
    if screen not in _VALID_SCREEN_MODES:
        raise ValueError(
            f"h_statistic_interactions: screen must be one of {_VALID_SCREEN_MODES}, got {screen!r}."
        )
    marker_names = list(marker_names)
    if X.shape[1] != len(marker_names):
        raise ValueError(
            f"h_statistic_interactions: X has {X.shape[1]} columns but "
            f"{len(marker_names)} marker_names were given - these must match 1:1."
        )
    if pair_predict_fn is not None:
        print(f"[interaction_extraction] NOTE: pair_predict_fn was supplied but the Layer D "
              f"model-aware fast path is not implemented in this delivery (the go/no-go "
              f"accuracy/speed spike required before shipping it was not run) - it is currently "
              f"IGNORED and every pair is still evaluated via the exact batched predict_fn path.")

    if grid_resolution < 2:
        print(f"[interaction_extraction] NOTE: grid_resolution clamped from {grid_resolution} to 2 "
              f"- a 1-point grid makes H^2 undefined (zero denominator).")
        grid_resolution = 2
    batch_bytes = int(max(1 * 1024 ** 2, min(256 * 1024 ** 2, batch_bytes)))

    background = X if X.shape[0] <= n_background else X.sample(n=n_background, random_state=0)
    background_values = background.to_numpy(dtype=float, copy=True)
    n_bg = background_values.shape[0]
    K = background_values.shape[1]
    pairs = list(pairs)
    n_pairs_total = len(pairs)

    # Bugfix (QTL-blind-spot): a marker's GRID and its degenerate/
    # monomorphic status must be resolved from every row this call was
    # actually given (`X` - e.g. the whole interaction test split), never
    # from `background_values` alone. `background_values` is frequently a
    # MUCH smaller random draw of it (`n_background` defaults to 30 for
    # RF's own 'Number of samples for marker effect interactions', see
    # hparam_specs.py) - small enough that a marker with a skewed allele
    # frequency can easily have its minor genotype class absent from that
    # one draw by chance alone, even though the marker is clearly
    # polymorphic overall. Genuine large-effect / selected QTLs are
    # disproportionately likely to BE skewed this way, so the bug was not
    # random noise: it was a systematic blind spot that hit real QTL
    # markers hardest, forcing H^2 = 0.0 (no predict call at all - see
    # `_is_degenerate()` below) for every pair touching them regardless of
    # true effect size, while pairwise-SHAP (which has no such shortcut)
    # kept reporting the real signal there. Confirmed by a side-by-side
    # circos comparison: 'RF_H-index' rings showed materially weaker
    # interaction links at known QTL positions than 'RF_Shapley' rings on
    # the identical run. `full_values` below is ONLY used to decide each
    # marker's grid values and whether it is degenerate; `background_values`
    # continues to be the (possibly much smaller) sample every partial-
    # dependence AVERAGE is computed over, exactly as before - so cost
    # scales the same way it always did. Whenever no subsampling happens
    # at all (`X.shape[0] <= n_background`), `full_values` and
    # `background_values` are identical and this is a byte-for-byte no-op.
    full_values = X.to_numpy(dtype=float, copy=True)

    # --- Layer A, F1: per-marker grid cache (K calls, not 2P) ---------
    involved_cols = sorted({c for pr in pairs for c in pr})
    grids = {}
    for c in involved_cols:
        col = full_values[:, c]
        if grid == 'observed':
            grids[c] = np.unique(col)[:3]
        else:
            grids[c], _ = _resolve_marker_grid(
                col, resolution=grid_resolution, discrete_max=grid_discrete_max,
                percentile_range=grid_percentile_range,
            )

    if cost_log:
        _branch_counts = (
            _classify_marker_coding(
                full_values, involved_cols, resolution=grid_resolution,
                discrete_max=grid_discrete_max, percentile_range=grid_percentile_range,
            ) if grid == 'auto' else None
        )
        if _branch_counts is not None:
            _coding_note = (
                f"grid='auto' coding regimes across {len(involved_cols)} involved marker(s): "
                f"{_branch_counts[1]} clean/hard-called (branch 1, bit-identical), "
                f"{_branch_counts[2]} discrete-with-contamination (branch 2, recovered), "
                f"{_branch_counts[3]} continuous/dosage (branch 3, quantile grid), "
                f"{_branch_counts[4]} skew-collapsed (branch 4, linear fallback)."
            )
        else:
            _coding_note = f"grid='observed' (legacy coding, [:3] of the numeric minimum)."
        print(f"[interaction_extraction] {_coding_note}")

    # --- Layer A, F4 / R1b: degenerate-pair skip -----------------------
    def _is_degenerate(c):
        if grids[c].size < 1:
            return True
        if not skip_degenerate:
            return False
        # Bugfix (see `full_values` note above): degeneracy is a claim
        # about the MARKER, not about one small random draw of it - it
        # must be checked against every row available (`full_values`),
        # not `background_values`.
        return bool(np.ptp(full_values[:, c]) <= degenerate_atol)

    live_pairs = []
    degenerate_rows = []
    for (i, j) in pairs:
        if _is_degenerate(i) or _is_degenerate(j):
            degenerate_rows.append((marker_names[i], marker_names[j], 0.0))
        else:
            live_pairs.append((i, j))
    n_degenerate = len(degenerate_rows)

    if cost_log:
        _max_grid_cells = max((grids[c].size for c in involved_cols), default=0) ** 2
        _projected_rows = sum(grids[i].size * grids[j].size * n_bg for (i, j) in live_pairs)
        print(f"[interaction_extraction] H-statistic: {len(involved_cols)}-marker shortlist -> "
              f"{n_pairs_total} pair(s) ({n_degenerate} skipped as degenerate), grid cells<="
              f"{_max_grid_cells}, background={n_bg} -> {_projected_rows} predicted rows.")

    # --- Layer C: optional approximate pre-screen of the LIVE pairs ---
    screened_zero_rows = []
    exact_pairs = live_pairs
    _screen_provenance = None
    if screen is not None and live_pairs:
        exact_pairs, _screen_provenance = _screen_pairs(
            screen, live_pairs, predict_fn=predict_fn, background_values=background_values,
            marker_names=marker_names, grids=grids, n_bg=n_bg, screen_keep=screen_keep,
        )
        _kept = set(exact_pairs)
        for (i, j) in live_pairs:
            if (i, j) not in _kept:
                screened_zero_rows.append((marker_names[i], marker_names[j], 0.0))
        print(f"[interaction_extraction] APPROXIMATE: screen={_screen_provenance!r} kept "
              f"{len(exact_pairs)}/{len(live_pairs)} live pair(s) for exact H^2 evaluation "
              f"({len(screened_zero_rows)} unscreened pair(s) emitted at value=0.0).")

    # --- Layer A, F2/F5/F6: bucket by grid shape, batch within budget -
    def _one_bucket(gi, gj, pair_list, bucket_batch_bytes):
        rows_per_pair = gi * gj * n_bg
        bytes_per_row = K * 8
        out_rows = []
        chunk_start = 0
        while chunk_start < len(pair_list):
            per_batch = max(1, bucket_batch_bytes // max(1, rows_per_pair * bytes_per_row))
            chunk = pair_list[chunk_start: chunk_start + per_batch]
            try:
                buf = np.tile(background_values, (len(chunk) * gi * gj, 1))
                for t, (i, j) in enumerate(chunk):
                    grid_i, grid_j = grids[i], grids[j]
                    row_start = t * rows_per_pair
                    row_end = row_start + rows_per_pair
                    buf[row_start:row_end, i] = np.repeat(grid_i, gj * n_bg)
                    buf[row_start:row_end, j] = np.tile(np.repeat(grid_j, n_bg), gi)
                # F6: one DataFrame / one predict_fn call per BATCH, not per pair.
                stacked_df = pd.DataFrame(buf, columns=marker_names, copy=False)
                preds = np.asarray(predict_fn(stacked_df), dtype=float).ravel()
            except MemoryError as exc:
                # Blueprint §2.8 failure mode: halve the byte budget and
                # retry THIS SAME chunk range (never advance past it) -
                # down to one pair per batch, at which point a further
                # MemoryError is genuinely fatal (nothing left to shrink).
                new_batch_bytes = max(1, bucket_batch_bytes // 2)
                if new_batch_bytes == bucket_batch_bytes:
                    raise
                print(f"[interaction_extraction] NOTE: MemoryError during a batched H-statistic "
                      f"predict call at batch_bytes={bucket_batch_bytes} ({exc!r}) - halving to "
                      f"{new_batch_bytes} and retrying this batch.")
                bucket_batch_bytes = new_batch_bytes
                continue
            for t, (i, j) in enumerate(chunk):
                row_start = t * rows_per_pair
                row_end = row_start + rows_per_pair
                f_ij = preds[row_start:row_end].reshape(gi, gj, n_bg).mean(axis=2)
                f_ij_c = f_ij - f_ij.mean()
                f_i = f_ij.mean(axis=1)
                f_j = f_ij.mean(axis=0)
                f_i_c = f_i - f_i.mean()
                f_j_c = f_j - f_j.mean()
                numerator = np.sum((f_ij_c - f_i_c[:, None] - f_j_c[None, :]) ** 2)
                denominator = np.sum(f_ij_c ** 2)
                h2 = float(numerator / denominator) if denominator > 0 else 0.0
                out_rows.append((marker_names[i], marker_names[j], h2))
            chunk_start += len(chunk)
        return out_rows, bucket_batch_bytes

    def _process_pairs(pair_list):
        """Group an arbitrary list of pairs by (gi, gj) shape, then
        batch-process each shape group - the unit of work a joblib
        BLOCK (below) runs, so a block spanning more than one grid
        shape is still handled correctly. Each bucket starts from the
        call's own `batch_bytes` and keeps whatever smaller size a
        MemoryError halving converged to for the REST of this pair
        list's own buckets, rather than retrying the already-known-too-
        large size from scratch for every subsequent bucket."""
        sub_buckets = {}
        for (i, j) in pair_list:
            sub_buckets.setdefault((grids[i].size, grids[j].size), []).append((i, j))
        out_rows = []
        _local_batch_bytes = batch_bytes
        for (gi, gj), plist in sub_buckets.items():
            bucket_rows, _local_batch_bytes = _one_bucket(gi, gj, plist, _local_batch_bytes)
            out_rows.extend(bucket_rows)
        return out_rows

    _t0 = time.perf_counter()
    if not exact_pairs:
        exact_rows = []
    elif n_jobs in (None, 0, 1) or len(exact_pairs) <= 1:
        exact_rows = _process_pairs(exact_pairs)
    else:
        # Update ID ver4-6, R1 Layer A F5: dispatch ~4 * n_jobs pair
        # BLOCKS, not one joblib task per pair (which, at P=124,750,
        # made cloudpickle/dispatch overhead a first-order cost on its
        # own - blueprint §2.1 F5).
        n_blocks = max(1, min(len(exact_pairs), 4 * int(n_jobs)))
        block_size = max(1, -(-len(exact_pairs) // n_blocks))  # ceil division
        blocks = [exact_pairs[k:k + block_size] for k in range(0, len(exact_pairs), block_size)]
        try:
            from joblib import Parallel, delayed
            block_results = Parallel(n_jobs=n_jobs)(delayed(_process_pairs)(block) for block in blocks)
            exact_rows = [row for block in block_results for row in block]
        except Exception as exc:
            print(f"[interaction_extraction] NOTE: h_statistic_interactions fan-out failed "
                  f"({exc}) - falling back to a single serial pass over all {len(exact_pairs)} "
                  f"live pair(s).")
            exact_rows = _process_pairs(exact_pairs)
    _elapsed = time.perf_counter() - _t0

    if cost_log and exact_pairs:
        _rows_computed = sum(
            grids[i].size * grids[j].size * n_bg for (i, j) in exact_pairs
        )
        _rate = _rows_computed / _elapsed if _elapsed > 0 else float('inf')
        print(f"[interaction_extraction] measured {_rate:,.0f} rows/s over {_elapsed:.2f}s "
              f"(n_jobs={n_jobs}) for {len(exact_pairs)} exactly-evaluated pair(s) this call. "
              f"Lower 'Max markers considered for interaction search', or enable the interaction "
              f"pre-screen, to reduce this on a future, larger call.")

    # Update (Requirements.md item 2 - "KNN returns many strong
    # interactions while SVR returns only a few under the same
    # configuration"): Friedman's H^2 implicitly assumes a reasonably
    # SMOOTH prediction surface - the numerator measures how much the
    # joint partial dependence departs from the two markers' own
    # additive main effects. A locally-adaptive, NON-smooth predict_fn
    # (nearest-neighbour models are the textbook example: which
    # training rows are "nearest" changes discontinuously the moment a
    # grid override crosses a neighbour boundary) is not well
    # approximated by ANY additive model, so H^2 comes out large for
    # ALMOST EVERY pair regardless of true epistasis - not a random
    # scattering of false positives, but a systematic, WIDESPREAD
    # inflation. A smooth kernel model (e.g. SVR-rbf) has no such
    # discontinuity, so its H^2 stays near zero except at genuine
    # interactions. This is exactly what makes one model's ring a dense
    # mesh of "strong" links and another's a sparse handful under
    # otherwise-identical settings (shortlist size, background size,
    # top-N%): the SPARSE ring's own `top_select()`/circos-side
    # `_select_top_interactions()` calls are capped by a genuinely
    # small NONZERO pool, while the DENSE ring's calls are not, because
    # non-smoothness leaves almost nothing exactly (or near-)additive to
    # cap against. Flagged here, once, as a diagnostic rather than
    # silently producing a visually dramatic ring - the fix is
    # methodological awareness, not a code change to the statistic
    # itself (H^2 is behaving exactly as defined; the SURFACE it is
    # measuring is what differs). `_WIDESPREAD_NONADDITIVITY_MEDIAN`
    # is a conservative threshold chosen well above the near-zero median
    # a smooth, mostly-additive genomic-prediction model typically shows
    # on real marker data, and well below what a genuinely non-smooth
    # predict_fn (as measured on this codebase's own 0/1/2 genotype
    # grid) routinely produces.
    if cost_log and exact_rows:
        _exact_values = np.array([_v for (_, _, _v) in exact_rows], dtype=float)
        _median_h2 = float(np.median(_exact_values))
        if _median_h2 > _WIDESPREAD_NONADDITIVITY_MEDIAN:
            _frac_high = float((_exact_values > _WIDESPREAD_NONADDITIVITY_MEDIAN).mean())
            print(f"[interaction_extraction] NOTE: the median H-statistic across this call's "
                  f"{_exact_values.size} exactly-evaluated pair(s) is {_median_h2:.3f} "
                  f"({_frac_high:.0%} exceed {_WIDESPREAD_NONADDITIVITY_MEDIAN}) - unusually "
                  f"high and WIDESPREAD for genuinely sparse epistasis. This pattern is typical "
                  f"of a locally-adaptive, non-smooth predict_fn (e.g. a nearest-neighbour "
                  f"model, especially with few neighbours) rather than a real, broad signal of "
                  f"interaction: H^2 assumes a reasonably smooth prediction surface, and a "
                  f"model whose predictions jump discontinuously as neighbouring training rows "
                  f"change will show large apparent 'interaction' almost everywhere. H^2 "
                  f"MAGNITUDE is therefore NOT directly comparable across different model "
                  f"types - only the RELATIVE ranking within one model's own H^2 column is "
                  f"meaningful. Increasing this model's own background sample size and/or (for "
                  f"a nearest-neighbour model) its own neighbour count can reduce, but may not "
                  f"eliminate, this effect.")

    rows = degenerate_rows + screened_zero_rows + exact_rows
    return pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])



def nid_interactions(first_layer_weight, downstream_influence, marker_names, *, pairs=None):
    """Neural Interaction Detection (Tsang, Cheng & Liu, 2018): pairwise
    strength for markers (i, j) aggregated as
    ``sum_h[ min(|W1[h, i]|, |W1[h, j]|) * downstream_influence[h] ]``
    over every hidden unit ``h`` of the network's FIRST layer. No
    forward passes at all - essentially free, which is why this is
    MLP's DEFAULT interaction method (see ``models/MLP.py``).

    The intuition: two markers can only jointly influence hidden unit
    h's output if BOTH have a non-negligible weight into it (hence the
    ``min`` - a pair is only as strong as its weaker connection into
    that unit), and that unit's contribution to the interaction is
    scaled by how much it actually influences the network's FINAL
    output (``downstream_influence`` - the aggregate absolute weight
    path from that hidden unit through every subsequent layer to the
    scalar output; the caller, ``models/MLP.py``, computes this from its
    own architecture, since it depends on whether the optional second
    hidden layer is in use).

    Parameters
    ----------
    first_layer_weight : array-like, shape (n_hidden_units, n_markers)
        The network's first ``Linear`` layer's weight matrix (e.g.
        ``model.hidden1.weight.detach().cpu().numpy()``).
    downstream_influence : array-like, shape (n_hidden_units,)
        Per-hidden-unit aggregate absolute influence on the scalar
        output - see ``models/MLP.py``'s own helper for how this is
        built for both the single- and two-hidden-layer architectures.
    marker_names : sequence of str
        Column names, in the SAME order as `first_layer_weight`'s
        columns (i.e. the model's own input marker order).
    pairs : iterable of (int, int), optional
        POSITIONAL index pairs to evaluate. ``None`` (the default)
        evaluates EVERY pair - unlike ``h_statistic_interactions()``/
        ``surrogate_tree_interactions()``, this is safe to leave
        unrestricted by default, since NID has no per-pair model-
        fitting or forward-pass cost at all (it is pure matrix algebra
        over already-fitted weights) - callers with very large marker
        counts may still pass an explicit shortlist to bound the output
        TABLE size (not the compute cost), via ``shortlist_markers()``.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']``.
    """
    w1 = np.abs(np.asarray(first_layer_weight, dtype=float))  # (H, M)
    influence = np.asarray(downstream_influence, dtype=float).ravel()  # (H,)
    n_hidden, n_markers = w1.shape
    if influence.shape[0] != n_hidden:
        raise ValueError(
            f"nid_interactions: first_layer_weight has {n_hidden} hidden units but "
            f"downstream_influence has {influence.shape[0]} entries - these must match."
        )
    marker_names = list(marker_names)
    if len(marker_names) != n_markers:
        raise ValueError(
            f"nid_interactions: first_layer_weight has {n_markers} input columns but "
            f"{len(marker_names)} marker_names were given - these must match 1:1."
        )
    if pairs is None:
        pairs = [(i, j) for i in range(n_markers) for j in range(i + 1, n_markers)]

    rows = []
    for (i, j) in pairs:
        strength = float(np.sum(np.minimum(w1[:, i], w1[:, j]) * influence))
        rows.append((marker_names[i], marker_names[j], strength))
    return pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])


def surrogate_tree_interactions(X, y_model_pred, marker_names, *, surrogate_cfg=None, n_jobs=1,
                                 explain_pool=None):
    """Fit a surrogate ``RandomForestRegressor`` to the MODEL'S OWN
    PREDICTIONS (never the true phenotype - see the "Raises" note
    below), then run ``tree_shap_interactions()`` on the SURROGATE. This
    is the only route this codebase offers that reaches
    rrBLUP/BayesB/GBLUP/RKHS, whose R-side explanation machinery
    (``iml::Shapley``, ~``2*M`` full BGLR refits per explained sample
    for MAIN effects alone - see ``models/GBLUP.R``'s own header) cannot
    be extended PAIRWISE at any tolerable cost (pairwise would be
    O(M^2) on top of that).

    Update (Requirements.md item 2 - cross-model H-index/TreeSHAP
    population mismatch): this function used to ALWAYS fit AND explain
    the surrogate on the same rows (`X` - this model's own TRAIN split,
    since that is what ``predicted_train`` pairs with). RF/SVR/KNN's own
    DIRECT (non-surrogate) interaction routes, by contrast, always
    explain on TEST rows (``models/RF.py``/``models/SVR.py``/
    ``models/KNN.py`` all sample from `test_x`, never `train_x`, for
    their own interaction search). Comparing a train-explained
    surrogate ring (rrBLUP/BayesB/GBLUP) against a test-explained direct
    ring (RF/SVR/KNN) is an apples-to-oranges population mismatch -
    different sample size, different realised allele frequencies -
    that shows up as unexplained cross-model "diversity" even when
    every OTHER setting (shortlist size, background/explained-sample
    size, top-N%) is configured identically. `explain_pool`, when given
    (``genomic_prediction.py::_r_model_surrogate_interaction()`` now
    always passes this task's own TEST split here, shortlisted to the
    same `marker_names`), is what the FITTED surrogate is actually
    explained on; fitting still uses the (usually larger, more
    informative) TRAIN split `X`/`y_model_pred` - only the explanation
    population changes, so the surrogate's own quality is unaffected.
    `None` (the default) reproduces the previous train-explained
    behaviour exactly, so any other/older caller of this function is
    unaffected.

    THIS IS APPROXIMATE, NOT THE MODEL'S OWN EXACT COMPUTATION (risk
    RK-8) - the values describe how well a tree ensemble's OWN pairwise
    structure explains that ensemble's attempt to reproduce the R
    model's predictions, not the R model's own interaction structure
    directly. Every caller of this function is responsible for
    disclosing this in its own log line and GUI help text (R2 acceptance
    criterion 9) - this function itself only prints one NOTE line, since
    it has no access to a caller-specific model name to phrase a fuller
    message with.

    Parameters
    ----------
    X : pandas.DataFrame
        The marker matrix to fit the surrogate on (and, by default, to
        explain - see `surrogate_cfg['explain_sample_size']` below).
    y_model_pred : array-like, shape (X.shape[0],)
        The ORIGINAL model's own predictions for these exact rows (e.g.
        an R model's ``predicted_train`` output) - the surrogate's
        target. MUST NOT be the true phenotype (see Raises below) -
        the whole point is to explain what the ORIGINAL model computed,
        not to fit an independent competing model.
    marker_names : sequence of str
        Column names, in the SAME order as `X`'s columns.
    surrogate_cfg : dict or None
        Optional overrides: ``'n_estimators'`` (default 200),
        ``'max_depth'`` (default None), ``'random_state'`` (default 0),
        ``'explain_sample_size'`` (default ``min(50, X.shape[0])`` (or
        ``min(50, explain_pool.shape[0])`` when `explain_pool` is given)
        - how many rows, sampled once at `random_state`, the surrogate
        is THEN explained on via ``tree_shap_interactions()`` - mirrors
        every other extractor's own explained-sample-size knob). ``None``
        (the default) uses every default above, reproducing today's
        (i.e. this feature's own first) behaviour.
    n_jobs : int, default 1
        Forwarded to the surrogate's own ``n_jobs`` (sklearn's usual
        meaning) AND to ``tree_shap_interactions()``'s row fan-out.
    explain_pool : pandas.DataFrame or None, default None
        Update (Requirements.md item 2) - the rows to explain the
        FITTED surrogate on, in place of `X` itself. Must carry the
        same columns as `X` (reindexed onto `X`'s own column order
        internally, so column order need not match). ``None`` (the
        default) explains on `X` - the original, pre-existing
        behaviour, unaffected.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']``.

    Raises
    ------
    ValueError
        If `X` and `y_model_pred` have a mismatched row count.
    """
    surrogate_cfg = dict(surrogate_cfg or {})
    n_estimators = surrogate_cfg.get('n_estimators', 200)
    max_depth = surrogate_cfg.get('max_depth', None)
    random_state = surrogate_cfg.get('random_state', 0)

    y_model_pred = np.asarray(y_model_pred, dtype=float).ravel()
    if y_model_pred.shape[0] != X.shape[0]:
        raise ValueError(
            f"surrogate_tree_interactions: X has {X.shape[0]} rows but y_model_pred has "
            f"{y_model_pred.shape[0]} values - a surrogate can only be fitted on the model's "
            f"OWN predictions for these SAME rows."
        )

    from sklearn.ensemble import RandomForestRegressor
    surrogate = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
        n_jobs=(n_jobs if n_jobs and n_jobs > 0 else None),
    )
    surrogate.fit(X, y_model_pred)

    # Requirements.md item 2: explain on `explain_pool` (this task's own
    # TEST split, when the caller provides one) instead of `X` (the
    # TRAIN split the surrogate was just fitted on) - see this
    # function's own module-level update note above for the full
    # rationale. Reindexed onto `X`'s own column order so a caller that
    # passes columns in a different order still lines up correctly.
    explain_source = X if explain_pool is None else explain_pool[list(X.columns)]
    explain_sample_size = surrogate_cfg.get('explain_sample_size', min(50, explain_source.shape[0]))

    explain_n = min(int(explain_sample_size), explain_source.shape[0])
    X_explain = (
        explain_source if explain_n >= explain_source.shape[0]
        else explain_source.sample(n=explain_n, random_state=random_state)
    )

    print(f"[interaction_extraction] NOTE: surrogate_tree_interactions() is an APPROXIMATE "
          f"method - it explains a {n_estimators}-tree surrogate fitted to this model's OWN "
          f"predictions, not the model's own internal computation. Treat these interaction "
          f"values as suggestive, not exact.")

    return tree_shap_interactions(surrogate, X_explain, marker_names, n_jobs=n_jobs,
                                   reduce='abs_then_sum')


def surrogate_h_statistic_interactions(X, y_model_pred, marker_names, *, pairs=None,
                                        surrogate_cfg=None, n_jobs=1, use_gpu=False,
                                        explain_pool=None):
    """Fit a surrogate ``RandomForestRegressor`` to the MODEL'S OWN
    PREDICTIONS (never the true phenotype - see the "Raises" note
    below, identical contract to ``surrogate_tree_interactions()``
    above), then run Friedman's pairwise H^2 statistic
    (``h_statistic_interactions()``) on the SURROGATE's own
    ``.predict()`` - RKHS's own route to marker-pair interactions.

    This is the SAME "explain a surrogate of this model's own
    predictions" bridge ``surrogate_tree_interactions()`` already uses
    for rrBLUP/BayesB/GBLUP (RKHS's R-side explanation machinery cannot
    be extended pairwise at any tolerable cost either - see that
    function's own docstring for the full reasoning, unchanged here).
    Only the INTERACTION-EXTRACTION algorithm applied to the surrogate
    differs: Friedman's H^2 (fully model-agnostic - needs only a
    ``predict_fn`` callable, see ``h_statistic_interactions()``'s own
    docstring) instead of exact pairwise TreeSHAP. This is RKHS's OWN
    route, not a general replacement - rrBLUP/BayesB/GBLUP keep calling
    ``surrogate_tree_interactions()`` unchanged; see
    ``genomic_prediction.py::_r_model_surrogate_interaction()`` for the
    per-model dispatch and the disclosed reasoning for singling RKHS
    out (its kernel already captures non-additive structure, unlike the
    three purely-additive/variable-selection models).

    THIS IS APPROXIMATE, NOT THE MODEL'S OWN EXACT COMPUTATION (risk
    RK-8) - identical caveat to ``surrogate_tree_interactions()``: the
    values describe how a tree-ensemble surrogate's own partial-
    dependence structure explains that surrogate's attempt to reproduce
    RKHS's predictions, not RKHS's own kernel structure directly. Every
    caller of this function is responsible for disclosing this in its
    own log line and GUI help text (R2 acceptance criterion 9), exactly
    as ``surrogate_tree_interactions()`` already requires of its own
    callers.

    Parameters
    ----------
    X : pandas.DataFrame
        The marker matrix to fit the surrogate on - also used, by
        default (whenever `explain_pool` is not given), as
        ``h_statistic_interactions()``'s own BACKGROUND set (see
        ``surrogate_cfg['n_background']`` below). Unlike
        ``surrogate_tree_interactions()``'s ``X_explain`` (a ONE-TIME
        row subsample explained once), H-statistic's background is
        resampled internally by ``h_statistic_interactions()`` itself
        (``random_state=0``) whenever the population it is given has
        more rows than `n_background` - this function does no sampling
        of its own beyond passing that population and `n_background`
        straight through.
    y_model_pred : array-like, shape (X.shape[0],)
        The ORIGINAL model's (RKHS's) own predictions for these exact
        rows (e.g. its ``predicted_train`` output) - the surrogate's
        fit target. MUST NOT be the true phenotype (see Raises below).
    marker_names : sequence of str
        Column names, in the SAME order as `X`'s columns.
    pairs : iterable of (int, int), optional
        POSITIONAL index pairs to evaluate. ``None`` (the default)
        evaluates EVERY pair among `marker_names` - safe here because
        the caller (``genomic_prediction.py::_r_model_surrogate_
        interaction()``) has ALREADY shortlisted `X`'s columns down to
        this model's own `max_interaction_features` before calling this
        function, exactly as ``models/SVR.py``/``models/KNN.py`` do
        before their own ``h_statistic_interactions()`` call - "every
        pair" here means every pair within an already-bounded
        shortlist, not an unbounded M^2.
    surrogate_cfg : dict or None
        Optional overrides: ``'n_estimators'`` (default 200),
        ``'max_depth'`` (default None), ``'random_state'`` (default 0),
        ``'n_background'`` (default ``min(100, X.shape[0])`` - forwarded
        directly to ``h_statistic_interactions()``'s own
        ``n_background``). ``None`` (the default) uses every default
        above. Update ID ver4-6, R1/R1b: also accepts ``'grid'``
        (default ``'auto'``), ``'grid_resolution'`` (default 3),
        ``'screen'``/``'screen_keep'`` (default ``None``/``0.02``) and
        ``'batch_bytes'``/``'cost_log'`` (default 32 MiB / True) -
        forwarded straight through to ``h_statistic_interactions()``,
        which see for what each one does. `genomic_prediction.py::
        _r_model_interaction_fields()`/`_r_model_surrogate_interaction()`
        populate these from RKHS's own appended ``HPARAMETERS`` fields.
    n_jobs : int, default 1
        Forwarded to the surrogate's own ``n_jobs`` (sklearn's usual
        meaning - ignored by a GPU/cuML surrogate, which has its own
        internal parallelism) AND to ``h_statistic_interactions()``'s
        own per-pair fan-out (true joblib multiprocessing across
        worker PROCESSES - see that function's own ``n_jobs``
        docstring). Both uses mirror ``surrogate_tree_interactions()``'s
        own identical "forwarded to both the fit and the explanation
        step" convention.
    use_gpu : bool, default False
        Whether to attempt a cuML GPU-backed surrogate fit (see
        ``_gpu_random_forest_cls()`` above) instead of scikit-learn's
        CPU ``RandomForestRegressor``. Falls back to CPU on any
        import/compatibility failure - never raises for this reason.
        Unlike ``surrogate_tree_interactions()`` (whose surrogate must
        stay CPU-side because ``shap.TreeExplainer`` cannot introspect
        a cuML forest), a GPU surrogate is safe here because this
        function only ever calls the surrogate's plain ``.predict()``.
    explain_pool : pandas.DataFrame or None, default None
        Update (Requirements.md item 2 - cross-model H-index population
        mismatch): the population to run ``h_statistic_interactions()``
        over in place of `X` itself - see ``surrogate_tree_
        interactions()``'s own identical parameter for the full
        rationale (this model's surrogate still FITS on `X`/
        `y_model_pred`; only the explained/background population
        changes). Must carry the same columns as `X` (reindexed onto
        `X`'s own column order internally). ``None`` (the default)
        explains on `X` - the original, pre-existing behaviour,
        unaffected.

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']``.

    Raises
    ------
    ValueError
        If `X` and `y_model_pred` have a mismatched row count.
    """
    surrogate_cfg = dict(surrogate_cfg or {})
    n_estimators = surrogate_cfg.get('n_estimators', 200)
    max_depth = surrogate_cfg.get('max_depth', None)
    random_state = surrogate_cfg.get('random_state', 0)
    # Requirements.md item 2: n_background's own default is now resolved
    # against the population actually explained (`explain_pool` when
    # given, else `X`) - see this function's own `explain_pool` note
    # above.
    explain_source = X if explain_pool is None else explain_pool[list(X.columns)]
    n_background = surrogate_cfg.get('n_background', min(100, explain_source.shape[0]))
    # Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): forwarded straight
    # through to h_statistic_interactions() below, at THAT function's own
    # defaults when absent from `surrogate_cfg` - so an existing caller
    # that never sets any of these keeps getting ver4-5-shaped behaviour
    # (grid='auto' is still the new, correctness-restoring default here
    # too - RKHS's surrogate reads PLINK-imputed / dosage-coded columns
    # exactly like SVR/KNN/RF's own shortlists do, so R1b's fix applies
    # here identically, not just at the three direct call sites).
    grid = surrogate_cfg.get('grid', 'auto')
    grid_resolution = surrogate_cfg.get('grid_resolution', 3)
    screen = surrogate_cfg.get('screen', None)
    screen_keep = surrogate_cfg.get('screen_keep', 0.02)
    batch_bytes = surrogate_cfg.get('batch_bytes', 32 * 1024 ** 2)
    cost_log = surrogate_cfg.get('cost_log', True)

    y_model_pred = np.asarray(y_model_pred, dtype=float).ravel()
    if y_model_pred.shape[0] != X.shape[0]:
        raise ValueError(
            f"surrogate_h_statistic_interactions: X has {X.shape[0]} rows but y_model_pred has "
            f"{y_model_pred.shape[0]} values - a surrogate can only be fitted on the model's "
            f"OWN predictions for these SAME rows."
        )

    marker_names = list(marker_names)
    if pairs is None:
        pairs = [(i, j) for i in range(len(marker_names)) for j in range(i + 1, len(marker_names))]

    surrogate_cls, is_gpu = _gpu_random_forest_cls(use_gpu)
    if is_gpu:
        surrogate = surrogate_cls(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state)
    else:
        surrogate = surrogate_cls(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            n_jobs=(n_jobs if n_jobs and n_jobs > 0 else None),
        )
    surrogate.fit(X, y_model_pred)

    print(f"[interaction_extraction] NOTE: surrogate_h_statistic_interactions() is an "
          f"APPROXIMATE method - it explains a {n_estimators}-tree{' GPU' if is_gpu else ''} "
          f"surrogate fitted to this model's OWN predictions via Friedman's H-statistic, not "
          f"the model's own internal computation. Treat these interaction values as "
          f"suggestive, not exact.")

    return h_statistic_interactions(surrogate.predict, explain_source, marker_names, pairs=pairs,
                                     n_background=n_background, n_jobs=n_jobs,
                                     grid=grid, grid_resolution=grid_resolution,
                                     screen=screen, screen_keep=screen_keep,
                                     batch_bytes=batch_bytes, cost_log=cost_log)


def canonicalize_shapiq_table(raw, marker_names=None):
    """ver4-25, Stage 9 (design blueprint R12): convert a raw shapiq
    long-format table (columns ``'markers'``/``'order'``/``'value'`` -
    ``'markers'`` always a tuple of marker NAMES, ``'order'`` its length
    - the shape `pipeline_utils.shapiq_pairwise_interactions()`/
    `shapiq_tree_interactions()` both return) into the shape every
    shapiq-based interaction route in this module now returns to its
    OWN caller (`surrogate_shapiq_interactions()`, `tree_shapiq_
    interactions()`, and `models/SVR.py`'s/`models/KNN.py`'s own inline
    'shapiq' interaction blocks).

    Every row keeps its own ``'order'`` and ``'markers'``, so a caller
    (ultimately `genomic_prediction.py`'s own Step 10) can split k=2 from
    k>=3 without this function needing to know or care which orders were
    actually requested; ``'marker1'``/``'marker2'`` are populated ONLY
    for order-2 rows (``None`` otherwise) purely so an order-2 row is
    ALSO immediately usable wherever this codebase's existing, pre-R12
    'marker1'/'marker2'/'value' convention is expected, with no separate
    conversion step for the common (orders=(2,)) case.

    Parameters
    ----------
    raw : pandas.DataFrame
        Columns ``['markers', 'order', 'value']`` (may be empty).

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'markers',
    'order', 'value']`` - empty (same columns, zero rows) if `raw` is
    empty.
    """
    if raw.empty:
        return pd.DataFrame(columns=['marker1', 'marker2', 'markers', 'order', 'value'])
    out = raw.copy()

    # Requirements2.md/Requirements3.md item 2/3 bugfix: `marker1`/
    # `marker2` must be populated ONLY for genuine order-2 rows, exactly
    # as this function's own docstring above already promises ("populated
    # ONLY for order-2 rows (None otherwise)"). The previous `len(m) >= 2`
    # test also matched every order>=3 tuple, silently TRUNCATING a 3+-way
    # interaction down to just its first two constituent markers and
    # reporting it as if it were an ordinary 2-way pair - a real
    # correctness bug (a spurious "pair" with no such interaction was ever
    # computed) that could also collide/fan-out against a genuine order-2
    # row for that same (marker1, marker2) pair once combined below.
    out['marker1'] = out['markers'].apply(lambda m: m[0] if isinstance(m, tuple) and len(m) == 2 else None)
    out['marker2'] = out['markers'].apply(lambda m: m[1] if isinstance(m, tuple) and len(m) == 2 else None)

    # Canonicalise ONLY the genuine order-2 rows (ascending marker1<=marker2)
    # - the order>=3 rows have marker1/marker2 == None and are left as-is
    # here (comparing None to a marker name would be meaningless, and
    # _canonicalize_marker_pairs' own >, .loc[...] assignment logic is only
    # meant for the order-2 shape anyway).
    _order2_mask = out['marker1'].notna()
    if _order2_mask.any():
        out = pd.concat(
            [_canonicalize_marker_pairs(out[_order2_mask]), out[~_order2_mask]],
            ignore_index=True,
        )

    # Requirements.md item 2 / Requirements3.md item 3 bugfix (performance
    # AND correctness): this function used to also densify `out` against
    # every possible C(len(marker_names), 2) marker pair - a full O(M^2)
    # outer-join grid - fabricating an explicit value=0.0/order=2 row for
    # every pair shapiq's own BUDGETED (sparse-by-design) search did not
    # happen to sample. For RF's own 'shapiq_tree' route (the caller most
    # affected - the ONLY interaction route that reaches this branch with
    # a large `marker_names`, and exactly why Requirements.md item 2 calls
    # out RF specifically), that is ~124,750 fabricated rows at this
    # model's own 500-marker shortlist default alone, growing to the tens
    # of millions for a larger/'all' shortlist - precisely the O(M^2)
    # blow-up every OTHER extractor in this module (`tree_shap_
    # interactions`, `h_statistic_interactions`) is carefully shortlisted/
    # triangle-selected/streamed specifically to AVOID. It was also
    # statistically wrong: a coalition shapiq's budget did not sample is
    # "not yet evaluated", not "confirmed zero interaction" - treating the
    # two as interchangeable buried genuine (sparse) signal under a wall
    # of manufactured zeros, which then (combined with `top_select()`'s
    # own now-fixed zero-row-padding bug above) could end up selected
    # into the "top N%" output directly. `marker_names` is therefore no
    # longer used to densify this table - only the pairs/sets shapiq
    # actually returned are kept, exactly like every other extractor here
    # (the parameter is kept, unused, purely so existing call sites -
    # `tree_shapiq_interactions()`, `surrogate_shapiq_interactions()` -
    # need no signature change).
    return out[['marker1', 'marker2', 'markers', 'order', 'value']].reset_index(drop=True)


def surrogate_shapiq_interactions(X, y_model_pred, marker_names, *, surrogate_cfg=None, n_jobs=1,
                                   use_gpu=False, explain_pool=None, orders=(2,), budget=4096,
                                   index='k-SII', candidate_narrowing: bool = True,
                                   adaptive_rows: bool = True):
    """Fit a surrogate ``RandomForestRegressor`` to the MODEL'S OWN
    PREDICTIONS (never the true phenotype - identical contract to
    ``surrogate_h_statistic_interactions()``/``surrogate_tree_
    interactions()`` above), then explain it with the optional `shapiq`
    package's own Shapley-interaction indices instead of Friedman's
    H-statistic - RKHS's own 'surrogate_shapiq' 'Marker interaction
    method' choice (design blueprint R11, Stage 8).

    This ROUTES THROUGH A SURROGATE, the SAME "explain a lightweight
    model fitted to this model's own predictions" bridge every other
    R-model interaction route in this file already uses - it does NOT
    explain an exact reconstruction of RKHS's own fitted kernel
    directly. This is a disclosed, deliberate scope decision (not an
    oversight): explaining the exact kernel reconstruction directly
    would need RKHS's own transductive fit machinery threaded through
    this call in a way this codebase's R/Python boundary cannot yet do
    safely without a live `shapiq`/R environment to validate it against
    - see the ver4-25 Change Summary. The hparam field was deliberately
    named 'surrogate_shapiq', not 'shapiq_predictor' (RKHS's OWN,
    separate, genuinely-exact 'Marker effect method' choice, which
    reconstructs this fit's own kernel regression exactly via
    ``models/kernel_effects.py::RKHSPredictor`` - see ``genomic_
    prediction.py::_apply_rkhs_shapiq_predictor_effect()``), precisely
    so this surrogate-based interaction route is never confused with
    that exact one. A future update MAY replace this surrogate route
    with a true exact one without changing this function's own public
    contract (still 'APPROXIMATE, not the model's own internal
    computation' either way, per the print() line below).

    THIS IS APPROXIMATE, NOT THE MODEL'S OWN EXACT COMPUTATION (risk
    RK-8) - identical caveat to ``surrogate_h_statistic_interactions()``.

    Unlike Friedman's H-statistic (pairs only), `shapiq` natively
    supports interaction orders above 2 - `orders` may list any subset
    of ``{2, 3, ...}``, and EVERY requested order's own rows are
    returned by this function (ver4-25, Stage 9, design blueprint R12 -
    see Returns below). Order-2 rows flow to ``Interaction.csv``
    unchanged (I11); order >= 3 rows flow to ``Interaction_higher_
    order.csv`` and are drawn on the circos plot as hub-and-spoke
    glyphs (`circos_plot.py::higher_order_interaction()`/
    `_draw_higher_order_glyphs()`) - see `genomic_prediction.py`'s own
    Step 10 split, described under Returns below.

    Parameters
    ----------
    X, y_model_pred, marker_names, surrogate_cfg (``'n_estimators'``/
    ``'max_depth'``/``'random_state'`` only - `shapiq`-specific settings
    are their own keyword arguments below, not read from this dict),
    n_jobs, use_gpu, explain_pool : see ``surrogate_h_statistic_
    interactions()``'s own identical parameters.
    orders : sequence of int, default ``(2,)``
        Forwarded to ``pipeline_utils.shapiq_pairwise_interactions()`` -
        see that function's own docstring. EVERY requested order's own
        rows are returned (ver4-25, Stage 9 - see Returns below).
    budget : int, default 4096
    index : str, default 'k-SII'
        Forwarded to ``pipeline_utils.shapiq_pairwise_interactions()``.
    candidate_narrowing : bool, keyword-only, default True
        Design blueprint R3b: forwarded to ``pipeline_utils.
        shapiq_pairwise_interactions()``'s own identical parameter.
        `True` (the default) narrows the order>=3 search to a
        calibration-ranked marker subset when `max(orders) >= 3`; `False`
        reproduces this function's pre-R3 single-pass, full-shortlist
        search exactly (I11).
    adaptive_rows : bool, keyword-only, default True
        ver4-28 (this update): shrinks the number of rows explained
        (below, previously a flat ``min(30, ...)``) as ``max(orders)``
        grows, via ``pipeline_utils.resolve_shapiq_explain_row_count()`` -
        the SAME "shrink cost as order grows" policy `models/RF.py`'s own
        depth/tree-count cap (design blueprint R2) already applies,
        extended here to the row axis (see that resolver's own
        docstring). `True` (the default) is a no-op at `max(orders) <= 2`
        (I11 - the common case). `False` restores the flat `min(30, ...)`
        row count at every order. Exposed as a function-level default
        (like `candidate_narrowing` above) rather than threaded through
        `genomic_prediction.py`'s own 7 call sites of
        `_r_model_surrogate_interaction()`, to keep this update's blast
        radius contained to this one file, matching `candidate_narrowing`'s
        own existing precedent.

    Design blueprint R1 note
    -------------------------
    `n_jobs` above was, before this update, used ONLY for this
    function's own surrogate ``RandomForestRegressor`` fit - the
    subsequent ``shapiq_pairwise_interactions()`` call silently dropped
    it, always explaining rows serially regardless of `n_jobs`. This was
    an omission (no comment anywhere disclaimed it), not a deliberate
    choice - fixed here by forwarding the SAME `n_jobs` value into that
    call's own new `n_jobs` parameter too (I11: `n_jobs<=1`, the common
    single-core case, is unaffected either way, since it was already the
    value used for the surrogate fit).

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'markers',
    'order', 'value']`` (ver4-25, Stage 9, design blueprint R12) -
    ``'marker1'``/``'marker2'`` are populated only for order-2 rows
    (``None`` otherwise); ``'markers'`` is always the full tuple of
    marker NAMES for every row, any order; ``'order'`` is
    ``len(markers)``. `genomic_prediction.py`'s own Step 10 splits this
    table by ``'order'``: ``==2`` rows (via ``marker1``/``marker2``) go
    to ``Interaction.csv`` (schema UNCHANGED, I11); ``>=3`` rows (via
    ``markers``) go to ``Interaction_higher_order.csv``. See
    ``canonicalize_shapiq_table()``'s own docstring for the exact
    conversion.

    Raises
    ------
    ValueError
        If `X` and `y_model_pred` have a mismatched row count.
    Exception
        Any failure inside `shapiq` itself (package missing, an API
        mismatch against the version actually installed) - deliberately
        NOT caught here; ``genomic_prediction.py::_r_model_surrogate_
        interaction()`` wraps this call in its own ``try``/``except`` and
        falls back to ``surrogate_h_statistic_interactions()`` with a
        printed NOTE, mirroring ``pipeline_utils.shapiq_marker_
        effects()``'s/``shapiq_pairwise_interactions()``'s own disclosed
        "written but not runtime-verified against a real `shapiq`
        install" caveat (see that module's own header comment).
    """
    surrogate_cfg = dict(surrogate_cfg or {})
    n_estimators = surrogate_cfg.get('n_estimators', 200)
    max_depth = surrogate_cfg.get('max_depth', None)
    random_state = surrogate_cfg.get('random_state', 0)
    explain_source = X if explain_pool is None else explain_pool[list(X.columns)]
    background_size = surrogate_cfg.get('n_background', min(100, explain_source.shape[0]))

    y_model_pred = np.asarray(y_model_pred, dtype=float).ravel()
    if y_model_pred.shape[0] != X.shape[0]:
        raise ValueError(
            f"surrogate_shapiq_interactions: X has {X.shape[0]} rows but y_model_pred has "
            f"{y_model_pred.shape[0]} values - a surrogate can only be fitted on the model's "
            f"OWN predictions for these SAME rows."
        )

    marker_names = list(marker_names)
    surrogate_cls, is_gpu = _gpu_random_forest_cls(use_gpu)
    if is_gpu:
        surrogate = surrogate_cls(n_estimators=n_estimators, max_depth=max_depth,
                                   random_state=random_state)
    else:
        surrogate = surrogate_cls(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
            n_jobs=(n_jobs if n_jobs and n_jobs > 0 else None),
        )
    surrogate.fit(X, y_model_pred)

    _base_explain_n = min(30, explain_source.shape[0])
    # ver4-28: order-aware shrink of the row axis (see this function's own
    # `adaptive_rows` docstring above) - a no-op at `max(orders) <= 2`
    # (I11), since `resolve_shapiq_explain_row_count()` itself returns
    # `_base_explain_n` unchanged in that case.
    explain_n = (
        resolve_shapiq_explain_row_count(max(orders) if orders else 2, _base_explain_n)
        if adaptive_rows else _base_explain_n
    )
    explain_n = min(explain_n, explain_source.shape[0])
    X_explain = (
        explain_source if explain_n >= explain_source.shape[0]
        else explain_source.sample(n=explain_n, random_state=random_state)
    )

    print(f"[interaction_extraction] NOTE: surrogate_shapiq_interactions() is an APPROXIMATE "
          f"method - it explains a {n_estimators}-tree{' GPU' if is_gpu else ''} surrogate fitted "
          f"to this model's OWN predictions via shapiq's own Shapley interaction indices, not "
          f"the model's own internal computation. Treat these interaction values as suggestive, "
          f"not exact.")

    raw = shapiq_pairwise_interactions(
        surrogate.predict, X, X_explain, marker_names, orders=orders,
        background_size=background_size, budget=budget, index=index,
        n_jobs=n_jobs, candidate_narrowing=candidate_narrowing,
    )
    # ver4-25, Stage 9 (design blueprint R12): every order shapiq computed
    # is now returned (previously anything above order 2 was dropped with
    # a NOTE, disclosed at the time as "until Interaction_higher_order.csv
    # storage lands" - it has now landed, see canonicalize_shapiq_table()'s
    # own docstring for the exact shape). genomic_prediction.py's own
    # Step 10 is what actually routes order==2 rows into Interaction.csv
    # and order>=3 rows into Interaction_higher_order.csv - this function
    # itself stays agnostic to that split.
    return canonicalize_shapiq_table(raw, marker_names=marker_names)


def ebm_pairwise_interactions(fitted_ebm, marker_names):
    """Read off an already-fitted ``interpret.glassbox.
    ExplainableBoostingRegressor``'s own NATIVE pairwise interaction
    terms - unlike every other extractor in this module, EBM's pairwise
    terms ARE part of the model itself (GA2M: "generalized additive
    model plus pairwise interactions"), not a post-hoc explanation of a
    separately-fitted model (blueprint §4.2 Layer 3) - there is no
    fit/explain step here, only a read of attributes the model already
    computed during its own ``.fit()``.

    Parameters
    ----------
    fitted_ebm : interpret.glassbox.ExplainableBoostingRegressor
        Already fitted. Reads its ``term_features_`` (a tuple of
        1-tuples for main effects and 2-tuples for pairwise terms - main
        effects are skipped here entirely, since they belong in this
        model's own marker-EFFECT output, not its interaction output)
        and ``term_importances()`` (one importance score per term,
        aligned with ``term_features_`` - verified directly against
        ``interpret`` 0.7.8 during Phase 2 preparation, not assumed).
    marker_names : sequence of str
        Column names in the SAME order the EBM was fitted on (i.e. its
        own ``feature_names_in_`` - the caller is responsible for this
        matching, exactly as ``tree_shap_interactions()``'s own
        `marker_names` argument requires).

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']`` -
    one row per pairwise term the EBM chose to fit (EBM's own
    ``interactions`` hyperparameter, not this function, controls HOW
    MANY pairwise terms exist in the first place - see
    ``models/EBM.py``).
    """
    marker_names = list(marker_names)
    term_features = fitted_ebm.term_features_
    importances = fitted_ebm.term_importances()
    rows = []
    for term, importance in zip(term_features, importances):
        if len(term) == 2:
            i, j = term
            rows.append((marker_names[i], marker_names[j], float(importance)))
    return pd.DataFrame(rows, columns=['marker1', 'marker2', 'value'])


# ---------------------------------------------------------------------------
# Requirement_patch3.md item 1 - ensemble / weighted-ensemble interaction
# rings. See the module docstring's own "Update (Requirement_patch3.md
# item 1 ...)" paragraph for why this lives here rather than being
# duplicated per caller.
# ---------------------------------------------------------------------------

_INTERACTION_ENSEMBLE_SCHEMA = [
    'population', 'phenotype', 'model', 'ratio', 'sample', 'marker1', 'marker2', 'value',
]


def normalize_task_interactions(interactions_df):
    """L1-normalise the ``'value'`` column within each (population,
    phenotype, model, ratio, sample) group - i.e. within ONE model's own
    interaction output for ONE task - so different interaction-inference
    methods' naturally different value scales (Friedman's H^2 in [0, 1] -
    ``h_statistic_interactions()``; abs()-summed pairwise TreeSHAP
    magnitudes - ``tree_shap_interactions()``/
    ``surrogate_tree_interactions()``; GATv2Conv attention weights;
    EBM's own native pairwise term importances -
    ``ebm_pairwise_interactions()``; Neural Interaction Detection
    magnitudes - ``nid_interactions()``) become directly comparable
    PROPORTIONS before being combined across models into one ensemble
    ring (Requirement_patch3.md item 1: "normalise interaction values so
    that we can ignore the difference in their scales attributed to
    different interaction inference methods").

    This is the marker-PAIR-interaction analogue of
    ``models/ensemble.py``'s own ``_safe_row_normalize()`` (used there
    for marker EFFECT rows), generalised for this long-format, SPARSE
    table, where different models can report entirely different sets of
    pairs for the same task and therefore cannot be treated as fixed-
    width rows the way marker effects are (a plain per-row L1 divide
    would silently assume every model reported every pair). Every
    extractor in this module already produces non-negative 'value'
    columns by construction (see the list above) - ``.abs()`` is still
    applied defensively before summing, mirroring ``_safe_row_
    normalize()``'s own posture for the one column in this codebase
    (marker effects) that genuinely can be signed.

    A (task, model) group whose values sum to exactly 0 is left as
    all-zero rather than becoming all-NaN via an unguarded 0/0 division -
    the same zero-sum guard ``_safe_row_normalize()`` already uses.

    Parameters
    ----------
    interactions_df : pandas.DataFrame
        Long-format table with (at least) ``['population', 'phenotype',
        'model', 'ratio', 'sample', 'value']`` columns - ``Interaction.
        csv``'s own in-memory shape.

    Returns
    -------
    A copy of `interactions_df` with `'value'` replaced by its
    per-(task, model)-group L1-normalised version. Never mutates the
    input.
    """
    if interactions_df.shape[0] == 0:
        return interactions_df.copy()
    out = interactions_df.copy()
    out['value'] = out['value'].astype(float).abs()
    group_cols = ['population', 'phenotype', 'model', 'ratio', 'sample']
    group_sums = out.groupby(group_cols)['value'].transform('sum').replace(0, np.nan)
    out['value'] = (out['value'] / group_sums).fillna(0.0)
    return out


def _canonicalize_marker_pairs(df):
    """Return a copy of `df` with `'marker1'`/`'marker2'` swapped wherever
    needed so `marker1 <= marker2` lexicographically for every row -
    canonicalising each (unordered) marker pair to ONE consistent
    representation before it is combined across models.

    Every extractor in this module (and every model-side caller, e.g.
    `models/RF.py`'s SHAP-triangle selection) reports `marker1`/
    `marker2` in whatever POSITIONAL order that particular call's own
    ``marker_names`` happened to be in - `tree_shap_interactions()`'s
    own docstring already notes this is a numeric no-op for a single
    extractor's own internal (upper-triangle vs. lower-triangle)
    consistency, since an interaction is an unordered relationship
    between two markers. It is NOT, however, guaranteed to be consistent
    ACROSS two DIFFERENT models: LD pruning/RF filtering/a top-K
    importance shortlist can each leave a different model with a
    different surviving marker SUBSET in a different column order, so
    one model's `('snp_5', 'snp_9')` and another's `('snp_9', 'snp_5')`
    can both describe the exact same pair. Without canonicalising first,
    `_combine_weighted_interactions()`'s own merge-on-`(marker1,
    marker2)` would silently treat those as two DIFFERENT pairs, and
    split what should be one combined ring entry into two under-counted
    ones instead.
    """
    if df.shape[0] == 0:
        return df.copy()
    out = df.copy()
    # Bug fix: `.to_numpy()` on this column dtype can return a VIEW, not
    # a copy - `copy=True` is required here, or the FIRST `.loc[swap,
    # ...] = ...` assignment below mutates `m1`/`m2` in place before the
    # SECOND assignment reads from them, silently corrupting swapped rows
    # (confirmed directly: without `copy=True`, a swapped row ends up
    # with the SAME marker name in both `marker1` and `marker2`, rather
    # than the two markers correctly exchanged).
    m1 = out['marker1'].to_numpy(copy=True)
    m2 = out['marker2'].to_numpy(copy=True)
    swap = m1 > m2
    if swap.any():
        out.loc[swap, 'marker1'] = m2[swap]
        out.loc[swap, 'marker2'] = m1[swap]
    return out


def _combine_weighted_interactions(task_df_normalized, weights):
    """Combine one task's already-normalised (see
    ``normalize_task_interactions``), possibly multi-model, marker-pair
    interaction rows into ONE table, via a weighted average over
    `weights` - the shared core both ``naive_ensemble_interactions()``
    (equal weights) and ``weighted_ensemble_interactions()`` (a task's
    own per-model weight vector) below delegate to.

    `task_df_normalized` MUST already be restricted to a SINGLE task
    (one population/phenotype/ratio/sample combination) - this function
    does no task-level filtering or grouping itself, only model-level
    combination.

    A marker pair only some of the contributing models reported is
    treated as an implicit 0 contribution from every model that didn't
    report it (i.e. a normal weighted-average denominator, not a
    per-pair one) - exactly how an arithmetic mean over models naturally
    behaves when not every term is present, directly analogous to every
    weighted-ensemble method's own ``effect_weighted`` computation in
    this codebase silently contributing nothing for a model with no
    effect row for a given task.

    Renormalises `weights` across only the models actually PRESENT in
    `task_df_normalized` (i.e. models this task genuinely reported an
    interaction for) - a model this task simply didn't compute
    interactions for (interactions are opt-in per model, per task,
    unlike marker effects which are essentially always returned)
    silently drops out rather than being counted as a hard-0 contributor
    that would otherwise deflate the combined total.

    Parameters
    ----------
    task_df_normalized : pandas.DataFrame
        ``['model', 'marker1', 'marker2', 'value']`` (plus whatever task
        columns happen to still be present - ignored).
    weights : dict {model_name: float}
        This task's own per-model combining weight - any positive scale
        (renormalised internally) and need not cover every model in
        `task_df_normalized` (a model absent from `weights` is simply
        excluded, exactly like one with 0 weight).

    Returns
    -------
    pandas.DataFrame with columns ``['marker1', 'marker2', 'value']`` -
    empty (but correctly columned) if no model both appears in
    `task_df_normalized` and has a usable (finite, nonzero-total) weight.
    """
    empty = pd.DataFrame(columns=['marker1', 'marker2', 'value'])
    if task_df_normalized.shape[0] == 0:
        return empty

    models_present = [m for m in task_df_normalized['model'].unique() if m in weights]
    if not models_present:
        return empty

    # Requirement_patch3.md item 1: interaction 'value' is a non-negative
    # STRENGTH measure (see normalize_task_interactions()'s own
    # docstring), not a signed effect - so, unlike effect_weighted's own
    # re-use of a weighted-ensemble method's raw combining weights
    # (which CAN be negative for Nelder_Mead/Bayesian_optimisation's
    # unconstrained weight bounds - see those files' own module notes),
    # the ABSOLUTE value of each model's weight is used here: a model's
    # magnitude of trust/influence for this task, regardless of sign, is
    # what should scale how much its interaction signal contributes to
    # the combined ring - a negative raw weight applied directly to a
    # strength measure would produce a negative "combined interaction
    # strength" with no valid interpretation for a circos ring.
    w = {m: abs(float(weights[m])) for m in models_present}
    w_sum = sum(w.values())
    if w_sum <= 0 or not np.isfinite(w_sum):
        return empty

    combined = None
    for m in models_present:
        if w[m] == 0:
            continue
        sub = task_df_normalized.loc[task_df_normalized['model'] == m, ['marker1', 'marker2', 'value']]
        sub = sub.groupby(['marker1', 'marker2'], as_index=False)['value'].sum()
        sub['value'] = sub['value'] * (w[m] / w_sum)
        if combined is None:
            combined = sub
        else:
            combined = pd.merge(combined, sub, on=['marker1', 'marker2'], how='outer', suffixes=('', '_add'))
            combined['value'] = combined['value'].fillna(0) + combined.pop('value_add').fillna(0)

    if combined is None or combined.shape[0] == 0:
        return empty
    return combined.reset_index(drop=True)


def _n_distinct_interaction_sources(task_df):
    """Requirements.md item 2 (follow-up): how many DISTINCT *prediction
    models* (not raw `MODEL_RUN` labels) actually contributed the
    interaction rows in `task_df['model']`.

    `task_df['model']` holds `MODEL_RUN`-style entries - which, for a
    single model tuned with more than one hyperparameter-tuning algorithm
    (``HP_TUNE[m]['algorithms']`` has 2+ entries - see
    ``models.hyperparameter_tuning.tuned_model_name()``), are suffixed
    per algorithm: e.g. ``'RF__Grid'`` and ``'RF__Bayesian'`` for ONE
    base model, ``RF``, tuned two ways. A first version of the "two or
    more models" gate below (Requirements.md item 1) counted
    `task_df['model'].nunique()` directly - which correctly gates a
    genuine two-different-models case, but WRONGLY let a single model
    tuned with two algorithms through as if it were "two prediction
    models": `get_interaction`/`emit_interaction` is not itself a tunable
    field (architecture doc S10 point 1), so BOTH tuning-algorithm
    variants of that one model independently reported interactions,
    landing in `task_df` under two different string labels despite being
    the exact same underlying model choice the user made. Each weighted-
    ensemble method (Linear transformation/Nelder Mead/Bayesian
    optimisation/Analytic least-squares) then computed its OWN validation-
    based weight split between the 'RF__Grid' variant and the
    'RF__Bayesian' variant - genuinely different splits per method, since
    each method's own optimisation naturally lands differently - producing
    genuinely DIFFERENT combined interaction rings between the weighted-
    ensemble methods even though the person configured only one model
    (RF) as an interaction source. That is exactly the "weighted ensembles
    also return different interaction patterns... even when only RF
    provides marker interactions" symptom: Requirement 1's task-level gate
    correctly suppressed the SIMPLE single-model case (RF untuned, or
    tuned with a single algorithm), which is why it looked "fixed" - but
    it left this one still open, merely making it harder to trigger
    (RF needs 2+ tuning algorithms configured) rather than resolving it.

    `models.hyperparameter_tuning.base_of()` - not `schema_key_of()` - is
    the correct inverse here: `base_of('RF__Bayesian') == 'RF'` collapses
    tuning-algorithm variants of the same model together (what this
    function wants), while leaving two independent
    `GAT_biological_prior_knowledge_<n>` bio-prior instances (each its
    own gene network, a genuinely different interaction source) distinct
    from one another - `schema_key_of()` would incorrectly collapse THOSE
    together too, since its own job is the opposite one (mapping every
    bio-prior instance to one shared HPARAM_SPECS/registry key).
    """
    return task_df['model'].map(base_of).nunique()


def naive_ensemble_interactions(interactions_df, model_selected):
    """Naive (equal-weight) marker-pair interaction ensemble - the
    interaction-table analogue of ``models/ensemble.py::ensemble()``'s
    own arithmetic-mean marker-EFFECT ensemble (Requirement_patch3.md
    item 1: "For the naive ensemble, the mean interaction values with
    the same weights should be calculated").

    Unlike the weighted variant below (necessarily called once PER TASK,
    from inside GP()'s own per-task loop, since each weighted-ensemble
    method's weight vector is itself task-specific), this runs over
    EVERY task present in `interactions_df` at once, in the same place
    and the same way ``models/ensemble.py``'s own effect/prediction
    ensemble already does - at finalisation, once, after every task has
    already completed.

    Parameters
    ----------
    interactions_df : pandas.DataFrame
        The FULL accumulated ``Interaction.csv``-shaped table (every
        task, every model) - filtered internally to `model_selected` and
        to whichever tasks those models actually reported interactions
        for.
    model_selected : list of str
        Which models' rows to combine - 'ensemble' itself is stripped if
        present (mirrors every ensemble function's own `model_selected`
        convention in this codebase).

    Notes
    -----
    Requirements.md items 1/2: a task only contributes a combined row here
    when it has interaction data from TWO OR MORE distinct models - a task
    where exactly one selected model reported an interaction (e.g. only RF
    has its own `get_interaction` opt-in enabled, even though several
    OTHER prediction models were selected for the run) is skipped
    entirely, since "combining" one model's ring with itself has nothing
    to show and previously produced a redundant, relabelled copy of that
    model's own ring - see the per-task check inside the loop below for
    the full rationale, and `weighted_ensemble_interactions()`'s matching
    check for why the two needed to agree.

    Returns
    -------
    pandas.DataFrame, columns ``['population', 'phenotype', 'model',
    'ratio', 'sample', 'marker1', 'marker2', 'value']`` - one row per
    (task, marker pair) the ensemble has a combined value for; `'model'`
    is always the literal string ``'ensemble'`` (the caller -
    genomic_prediction.py's own finalisation block - is responsible for
    the SAME `ensemble__<tuning-group>` relabelling it already applies
    to `sample_effect`/`sample_record` for a multi-tuning-algorithm run,
    exactly mirroring that existing rename). Empty (but correctly
    columned) if no selected model reported any interaction anywhere, OR
    if every task that did only ever had a single contributing model.
    """
    model_selected = [m for m in model_selected if m != 'ensemble']
    empty = pd.DataFrame(columns=_INTERACTION_ENSEMBLE_SCHEMA)
    if interactions_df.shape[0] == 0 or not model_selected:
        return empty

    selected = interactions_df[interactions_df['model'].isin(model_selected)]
    if selected.shape[0] == 0:
        return empty

    # Canonicalise BEFORE normalising/grouping - see
    # _canonicalize_marker_pairs()'s own docstring for why this must
    # happen before any across-model pair merge.
    selected = _canonicalize_marker_pairs(selected)
    normalized = normalize_task_interactions(selected)
    task_cols = ['population', 'phenotype', 'ratio', 'sample']

    out_frames = []
    for task_key, task_df in normalized.groupby(task_cols, dropna=False, sort=False):
        # Requirements.md items 1/2: a naive-ensemble interaction ring is
        # only meaningful when THIS task actually has interaction rows
        # from two or more DISTINCT models - with exactly one
        # contributing model (e.g. only RF has `get_interaction` enabled,
        # even though several OTHER prediction models were selected for
        # the run), an 'ensemble' ring here would just be that one
        # model's own ring under a different label, with no combination
        # to show. Before this check existed, that relabelled copy WAS
        # produced (satisfying `combined.shape[0] != 0` below just as a
        # genuine 2+-model combination would), which is exactly why a
        # single-source run still emitted a separate
        # 'circos_..._interaction_ensemble.png' - a plot whose only
        # possible "pattern difference" from RF's own ring was ever an
        # ARTEFACT of which tasks each ensemble happened to include, not
        # a genuine one (see weighted_ensemble_interactions()'s matching
        # check below for why the weighted rings could disagree with each
        # other in exactly that way despite sharing the same single
        # source). Skipping the task here means a single-interaction-
        # source run now never contributes ANY task to the 'ensemble'
        # ring, so `naive_ensemble_interactions()` returns fully empty in
        # that case and no such plot is produced at all - see
        # models/ensemble.py's own caller, which already treats an empty
        # `interaction_ensemble` as a normal no-op.
        #
        # Requirements.md item 2 (follow-up): counted via
        # `_n_distinct_interaction_sources()`, NOT a plain
        # `task_df['model'].nunique()` - see that helper's own docstring.
        # A plain nunique() over-counts a single base model tuned with
        # several hyperparameter-tuning algorithms (e.g. 'RF__Grid' AND
        # 'RF__Bayesian' both reporting interactions) as if it were two
        # independent prediction models, which let this exact single-
        # source-in-spirit case straight through the item-1 gate above.
        if _n_distinct_interaction_sources(task_df) < 2:
            continue
        weights = {m: 1.0 for m in task_df['model'].unique()}
        combined = _combine_weighted_interactions(task_df, weights)
        if combined.shape[0] == 0:
            continue
        combined = combined.copy()
        combined['population'], combined['phenotype'], combined['ratio'], combined['sample'] = task_key
        out_frames.append(combined)

    if not out_frames:
        return empty
    result = pd.concat(out_frames, ignore_index=True)
    result['model'] = 'ensemble'
    return result[_INTERACTION_ENSEMBLE_SCHEMA]


def weighted_ensemble_interactions(interactions_df, model_selected, model_weights,
                                    task_population, task_phenotype, task_ratio, task_sample,
                                    method_label):
    """Weighted marker-pair interaction ensemble for ONE task - the
    interaction-table analogue of Linear_transformation.py/
    Nelder_Mead.py/Bayesian_optimisation.py's own per-task
    ``effect_weighted`` combination (Requirement_patch3.md item 1:
    "weighted average should be applied to the weighted ensemble, as
    done for representing ensemble rings in circos plots").

    Called once per task, immediately after one of those three methods'
    own per-model weight vector for THIS task has already been computed
    and appended to the `weight` accumulator - reusing that SAME weight
    row (before or after any `<label>__<tuning-group>` relabelling; only
    the per-model weight VALUES matter here, not the row's own 'model'
    label) is what keeps this ensemble's interaction ring weighted
    consistently with that method's own marker-effect ring.

    Parameters
    ----------
    interactions_df : pandas.DataFrame
        The full accumulated interactions table (every task, every model
        so far) - filtered internally to this ONE task and to
        `model_selected`.
    model_selected : list of str
        This weighted ensemble's own model group (e.g. `_group_models`) -
        'ensemble' itself is stripped if present.
    model_weights : dict {model_name: float}
        This task's own already-computed per-model weight (e.g. straight
        from the newly-appended `weight` row's own model_selected
        columns) - any positive scale, renormalised internally (see
        `_combine_weighted_interactions()`).
    task_population, task_phenotype, task_ratio, task_sample
        This task's own identifiers, exactly as stored in
        ``sample.loc[i, ...]`` inside GP(). `task_ratio` is stringified
        internally to match `interactions_df['ratio']`'s own storage
        convention (every row written there is stringified at write
        time - see GP()'s own per-model dispatch loop - since a 3-tuple
        'between'/W_OPT ratio cannot itself be a clean DataFrame cell
        value).
    method_label : str
        The label to write into the returned rows' 'model' column - the
        SAME (possibly tuning-group-suffixed) label this method's own
        effect/record rows already carry for this task.

    Notes
    -----
    Requirements.md items 1/2: returns empty whenever this task has
    interaction data from FEWER than two distinct models (see the
    in-function check below) - a single contributing model has nothing
    to weight-combine, and previously produced a redundant, relabelled
    copy of that one model's own ring, which was not guaranteed to cover
    the same set of tasks `naive_ensemble_interactions()` does (this
    function only ever runs for tasks with a validation split) and could
    therefore render as a visibly DIFFERENT pattern from the naive
    ensemble's - the exact bug Requirements.md item 2 describes.

    Returns
    -------
    pandas.DataFrame, same schema as `naive_ensemble_interactions()` -
    empty (but correctly columned) if this task has no usable
    interaction data to combine, or has data from only one model.
    """
    model_selected = [m for m in model_selected if m != 'ensemble']
    empty = pd.DataFrame(columns=_INTERACTION_ENSEMBLE_SCHEMA)
    if interactions_df.shape[0] == 0 or not model_selected:
        return empty

    task_ratio_str = str(task_ratio)
    task_df = interactions_df[
        interactions_df['model'].isin(model_selected)
        & (interactions_df['population'] == task_population)
        & (interactions_df['phenotype'] == task_phenotype)
        & (interactions_df['ratio'] == task_ratio_str)
        & (interactions_df['sample'] == task_sample)
    ]
    if task_df.shape[0] == 0:
        return empty

    # Requirements.md items 1/2: same task-level "two or more
    # CONTRIBUTING models" gate as naive_ensemble_interactions() above -
    # see that function's matching check for the full rationale. Without
    # this, a single-interaction-source task (e.g. only RF has
    # `get_interaction` enabled) still produced a 'Linear transformation'/
    # 'Nelder Mead'/'Bayesian optimisation'/'Analytic least-squares' ring
    # here that was just RF's own ring relabelled - and, critically, NOT
    # necessarily over the SAME set of tasks naive_ensemble_interactions()
    # includes: this function only ever runs for tasks with a validation
    # split (W_OPT's own precondition in genomic_prediction.py), while the
    # naive ensemble runs over every task regardless. Two rings that are
    # each "just RF, relabelled" but averaged over two DIFFERENT task
    # subsets are not guaranteed to render identically once circos_plot.
    # py's own interaction() groups-and-means each label's rows separately
    # per phenotype - which is exactly the "different interaction patterns
    # for each ensemble model" bug Requirements.md item 2 describes. Gating
    # here (identically to the naive side) means neither ring is produced
    # at all whenever there is only one real source, so there is nothing
    # left that could disagree.
    #
    # Requirements.md item 2 (follow-up): counted via
    # `_n_distinct_interaction_sources()`, NOT a plain
    # `task_df['model'].nunique()` - a plain count over-counts a single
    # base model tuned with several hyperparameter-tuning algorithms
    # (e.g. 'RF__Grid' AND 'RF__Bayesian' both reporting interactions) as
    # if it were two independent prediction models. Left uncorrected, THIS
    # is exactly what let "different interaction patterns between the
    # weighted-ensemble methods themselves" through even after the item-1
    # gate above: each weighted method computes its own validation-based
    # weight split between the two tuning-algorithm variants of the SAME
    # model, so 'Linear transformation'/'Nelder Mead'/'Bayesian
    # optimisation' each combined 'RF__Grid'+'RF__Bayesian' differently -
    # genuinely different rings, despite the person having configured only
    # one model (RF) as their interaction source. See
    # `_n_distinct_interaction_sources()`'s own docstring for the full
    # account and why `base_of()` (not `schema_key_of()`) is the correct
    # collapse to use here.
    if _n_distinct_interaction_sources(task_df) < 2:
        return empty

    # Canonicalise BEFORE normalising/combining - see
    # _canonicalize_marker_pairs()'s own docstring for why this must
    # happen before any across-model pair merge.
    task_df = _canonicalize_marker_pairs(task_df)
    normalized = normalize_task_interactions(task_df)
    combined = _combine_weighted_interactions(normalized, model_weights)
    if combined.shape[0] == 0:
        return empty

    combined = combined.copy()
    combined['population'] = task_population
    combined['phenotype'] = task_phenotype
    combined['ratio'] = task_ratio_str
    combined['sample'] = task_sample
    combined['model'] = method_label
    return combined[_INTERACTION_ENSEMBLE_SCHEMA]
