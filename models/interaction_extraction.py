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

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor

from pipeline_utils import parallel_shap_values
from models.hyperparameter_tuning import base_of

_VALID_REDUCE = ('abs_then_sum', 'sum_then_abs')

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
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
            return CumlRF, True
        except Exception as exc:
            print(f"[interaction_extraction] USE_GPU_SKLEARN requested for the surrogate "
                  f"H-statistic interaction step but cuML is unavailable ({exc}) - falling "
                  f"back to scikit-learn's CPU RandomForestRegressor.")
    return RandomForestRegressor, False


def tree_shap_interactions(fitted_model, X_explain, marker_names, *,
                            n_jobs=1, reduce='abs_then_sum'):
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

    if hasattr(fitted_model, 'get_booster'):
        # XGBoost model (duck-typed - only xgboost.XGBRegressor/XGBClassifier
        # have this method among this function's callers) - routed through
        # XGBoost's own native pred_interactions computation instead of
        # shap.TreeExplainer. See _xgboost_native_interaction_matrices()'s
        # own docstring for why: shap.TreeExplainer's XGBoost loader can
        # fail on modern XGBoost's normal base_score serialisation format,
        # and that failure cannot be worked around from the model side.
        raw = _xgboost_native_interaction_matrices(fitted_model, X_explain)
    else:
        explainer = shap.TreeExplainer(fitted_model)
        raw = parallel_shap_values(explainer, X_explain, None, n_jobs, interaction=True)

    if reduce == 'abs_then_sum':
        matrix = np.abs(raw).sum(axis=0)
    else:
        matrix = np.abs(raw.sum(axis=0))

    matrix_df = pd.DataFrame(matrix)
    matrix_df = matrix_df.where(np.triu(np.ones(matrix_df.shape)).astype(bool))
    writable_array = matrix_df.to_numpy(copy=True)
    np.fill_diagonal(writable_array, np.nan)
    matrix_df.iloc[:, :] = writable_array
    matrix_df.index = matrix_df.columns = marker_names

    result = matrix_df.stack().dropna().reset_index(drop=False)
    result.columns = ['marker1', 'marker2', 'value']
    return result


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
    """
    if value == 'all' or value is None:
        return df.reset_index(drop=True)
    if mode == 'count':
        m = max(0, min(int(value), df.shape[0]))
        return df.nlargest(m, 'value').reset_index(drop=True)
    if mode == 'percentage':
        nonzero = df[df['value'] > 0]
        m = int(round(df.shape[0] * (value / 100)))
        m = max(0, min(m, nonzero.shape[0]))
        return nonzero.nlargest(m, 'value').reset_index(drop=True)
    raise ValueError(f"top_select: mode must be 'percentage' or 'count', got {mode!r}.")


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
            _coding_note = f"grid='observed' (ver4-5 verbatim, [:3] of the numeric minimum)."
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
