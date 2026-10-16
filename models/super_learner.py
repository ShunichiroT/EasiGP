"""
models/super_learner.py
=======================
ver4-77 (design blueprint ``EasiGP_ver4-77_Design_Blueprint.md``, R2): the opt-in
**Super learner**.

What it does
------------
After every model has been fitted once (Step 9) and the validation split has done its job
(tuning and the W_OPT weights), the Super learner puts those validation phenotypes to work a
second time:

1. every member is fitted OUT OF FOLD over ``train ∪ valid`` (``folds`` positional folds, one
   partition shared by all members), at the member's own Step 9 hyperparameters;
2. the K out-of-fold columns give one weight vector: the cross-validated shrinkage path of
   ``models.ensemble_regularization.shrinkage_path_weights`` (``(1 - a) * equal + a * simplex
   least squares``, ``a`` chosen on held-out squared error);
3. every member is refitted ONCE on ``train ∪ valid`` (``deploy='refit'``), or the mean of its
   fold models is used (``deploy='cross_fit'``, no refit), and the test prediction is the
   weighted sum of those columns.

Why this design (blueprint R2, simulation ``ice_sim22``)
--------------------------------------------------------
* The gain is information: members refitted on 33 % more phenotyped lines are better members.
  Re-weighting the same Step 9 columns is capped by ``ORK - base`` (+0.002 ... +0.006 r at
  600 lines) and never gets close to that; this route measured +0.012 ... +0.047 r over today's
  ``base`` in 6 of 6 cells.
* The weights must be cross-validated over ALL labelled rows. Weights fitted on the validation
  rows alone are too noisy (60 rows), and equal weights lose when one model class dominates.
* 5 folds are enough, and the Step 9 hyperparameters are good enough for the refit: re-selecting
  them inside the Super learner gained <= +0.003 r for 2.6x the fits.
* With no validation split the Super learner uses the training rows only (``deploy='step9'``):
  the test column is then the Step 9 columns at the cross-validated weights, at ``K * folds``
  fits.
* ver4-79 (``ice_sim24`` / ``ice_sim25``): the estimator was NOT changed. With a validation
  split the Super learner beat the best single model in 6 of 6 cells (+0.014 ... +0.048 r);
  without one (``deploy='step9'``) it only matches it (-0.014 ... +0.008 r), because members'
  errors are highly correlated and re-weighting the same columns cannot recover the oracle
  headroom. Twelve other combiners and two hyperparameter-library variants did not improve on
  it, and a fallback to the single best model was rejected by design. ``diag['information_gain']``
  and the ``deploy='step9'`` NOTE make this visible.

* ver4-85 (``ice_sim27``, float ratio, 6 cells): with ``oof_tuning`` every tuned member is scored
  out of fold over the training lines. Out-of-fold Bayesian tuning raised the Super learner by
  +0.0135 r over untuned members (t 3.1-6.3 in 4 cells, never worse). Drawing the tuning folds
  independently of the Super learner's partition (``sl_tuning_folds``) halved the excess optimism
  of the out-of-fold weights (+0.019 -> +0.009 r) and gained +0.0019 r over reusing the winners'
  own out-of-fold columns (L-polygenic +0.0024, t 4.3; no harmful cell), for 1 extra fit per
  member and fold.

Weight methods (ver4-80, R1)
----------------------------
Step 2 above is one of four weight methods, not the only one. ``run_super_learner(...,
weight_methods=[...])`` builds the out-of-fold matrix and the test columns ONCE and then lets
each requested W_OPT method determine the member weights on that matrix: the cross-validated
analytic least squares described above (``'Analytic least-squares'``, the legacy row and the
default), or the user's Nelder Mead / Bayesian optimisation / Linear transformation function,
called with the out-of-fold predictions of ALL labelled rows where it normally sees only the
validation split (:func:`_optimiser_variant`; each keeps its own hyperparameters, validation
floor and shrinkage). One :class:`SLVariant` per method goes into ``SLResult.variants``; the
caller (``genomic_prediction`` Step 11d) writes one row each. The four combinations are
therefore {out-of-fold predictions} x {the four weight optimisers}; the pre-ver4-80 Super
learner is the analytic one.

I7 (amended in ver4-77): members may be refitted on ``train ∪ valid`` AFTER all tuning and
weighting. Test rows never enter a fit, a weight or a selection: every unit is scored against its
own copy of the test frame only to obtain the prediction column, and no decision reads it.

Contract
--------
* Pure Python / NumPy / pandas. Settings are a plain dict (``pipeline_utils._SL_DEFAULTS``, I1).
  Imports only ``models.*``, ``model_registry`` and ``pipeline_utils`` helpers, never
  ``genomic_prediction`` (I2). Every fit goes through the caller-supplied ``runner`` (a
  ``ModelTrialRunner``) and ``models.joint_tuning._run_units`` in ONE parallel call (I3, I10);
  each unit carries its own test frame (``_FitUnit.test_frame``).
* Row selection is positional (``.iloc``, I4): the train+valid frame is ``_cv_frame(train,
  valid)`` (train rows first, fresh RangeIndex); an empty validation split gives ``train`` alone.
* I6: bio-prior members (``IterMember.marker_channel is False``) are never fitted out of fold
  (their runner ignores the frames it is given), so they are left out with a NOTE. Every other
  member is fitted on its OWN frames, so marker-source routing is kept.
* I9: the fold partition comes from ``seed + _SL_SEED_OFFSET`` and is shared by all members;
  nothing here keeps state between tasks.
* Degrades, never raises, on a failed or malformed fit (a NOTE says what happened); raises
  ``ValueError`` only on a contract violation (fewer than 2 members passed, or frames and
  predictions of different row counts).

ver4-94 (R0 / R3): the population-POOLED variant
------------------------------------------------
With ``settings['pop_sources'] == ['pooled']`` and a ``'pooled'`` frame in ``extra_sources`` (the
other populations' labelled lines, ``models.relatedness.build_population_source``), every member
also gets columns fitted with those lines added to its training set: out of fold on (train+valid
minus the fold) + source, and one refit on train+valid + source for the test column. A second
weight vector is then fitted over ``[own columns | pooled columns]`` by the same cross-validated
shrinkage path, starting not from equal weights but from "own columns equal, pooled columns 0"
(``shrinkage_path_weights(prior=...)``): a pooled column gets weight only when cross-validation
supports it, and alpha = 0 reproduces the own-lines equal-weight row. The result is an EXTRA row
(``SLResult.tagged_variants['pooled']``) next to the unchanged one. The unchanged variants are
computed first and never read a pooled column, so switching the feature on cannot move them.
Source columns never see the task's own validation or test lines (guaranteed by the source
builder). Evidence: simulation gate ``ice_sim40_p1_pooled.py`` (blueprint ver4-94 R3).

ver4-96: with ``pop_member_rows`` on, each member that has complete pooled columns also gets a row
of its own, ``SLResult.tagged_variants['pooled_member'][<member name>]``: its pooled refit's test
prediction and its pooled out-of-fold train / valid columns, unweighted. Next to the plain member's
row this is the like-for-like measure of what the borrowed lines do to that model alone (the
pooled Super-learner row mixes that with the combiner). No extra fit is made.

pool-fix: with ``pop_member_rows`` on, each such member ALSO gets a GATED row,
``SLResult.tagged_variants['pooled_member_gated'][<member name>]``. The ungated row above has no
safeguard: whatever the other populations do to that model, the row shows it, while the pooled
Super-learner row starts from "own lines only" and lets cross-validation decide how much of the
pooled columns to use. Comparing the two is therefore not like for like. The gated row gives one
model the same safeguard: weights over ``[own column | pooled column]`` of that model, from the
same cross-validated shrinkage path with the prior "own column 1, pooled column 0" (alpha = 0 is
the plain model). Use it against the pooled Super-learner row; use the ungated row only to see
what pooling does to the model with no safeguard. No extra fit is made.

pool-speed: the cost of the pooled columns. Each pooled fit trains on the own lines plus every
borrowed line, so this stage dominates the run time. ``pop_deploy='cross_fit'`` (default) takes the
pooled test column from the mean of the pooled fold models and makes no refit (K fits fewer);
``pop_folds`` sets the pooled columns' own fold count (0 = ``folds``). Evidence and the rejected
levers: ``ice_sim41_pooled_speed.py`` and the comment in ``pipeline_utils._SL_DEFAULTS``.
'refit' with ``pop_folds=0`` is the ver4-94..99 behaviour exactly.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from models.ensemble_regularization import shrinkage_path_weights
from models.hyperparameter_tuning import _disable_explainability
from models.iterative_ensemble import (IterMember, _columns_from_result, _fit_stats, _oof_vector,
                                       _positional_folds)
from models.joint_tuning import _FitUnit, _LazyTrainFrame, _cv_frame, _run_units
from pipeline_utils import _SL_DEFAULTS, _SL_OPTIONAL_DROP_ORDER, SL_ANALYTIC_METHOD

# I9: a prime that is not used by iterative_ensemble._OOF_SEED_OFFSET (104729) or by the joint
# search's out-of-fold offset, so the Super learner's partition is its own.
_SL_SEED_OFFSET = 130363
# ver4-80 (R1): the hold-out of the Linear-transformation combiner is its own partition (a prime
# not used by any offset above).
_SL_WOPT_OFFSET = 15485863
# ver4-80 (R1): share of the out-of-fold rows the Linear-transformation combiner keeps back for
# its own early stopping (it is the only weight method that fits on one part and stops on another).
_SL_LT_HOLDOUT = 0.2
# ver4-85: out-of-fold hyperparameter tuning (SUPER_LEARNER['oof_tuning']). The tuning folds are
# drawn from their OWN seed (a prime not used above) plus the replicate, i.e. reshuffled relative to
# the Super learner's partition, so the weights are fitted on fresh out-of-fold predictions rather
# than on the very predictions that won the hyperparameter search (ice_sim27, see sl_tuning_folds).
_SL_TUNE_SEED_OFFSET = 179424673
_SL_TUNE_RESHUFFLE = True
# ver4-94 (R3, I9): the seed of the capped random draw of source lines is ``seed + _SL_POP_OFFSET +
# 7919 * replicate`` (computed by the caller); a prime not used by any offset above.
_SL_POP_OFFSET = 1299709


@dataclass
class SLResult:
    """Outcome of :func:`run_super_learner`.

    ``pred`` maps 'train' / 'valid' / 'test' to the weighted prediction of the Super learner; it
    is EMPTY (``{}``) when no row should be written (see ``notes``). ``pred['train']`` and
    ``pred['valid']`` are OUT-OF-FOLD predictions (``P[rows] @ w``), not in-sample ones;
    ``pred['valid']`` has no entries when the task has no validation split. ``members`` are the
    member labels that carry weight, ``weights`` their weights (non-negative, sum 1), ``alpha``
    the chosen shrinkage level. ``deploy`` is how the test column was formed: 'refit',
    'cross_fit' or 'step9' (the Step 9 columns, when there is no validation split to refit on).
    ``effect`` holds each member's refit |effect| vector (only with ``explain_refit`` and
    ``deploy='refit'``; None otherwise). ``n_fits`` is the number of model fits made, ``notes`` the
    NOTE messages the caller should print, ``diag`` the per-group facts the caller logs
    (``oof_r`` Pearson per member, ``alpha``, ``n_rows``, ``folds``, ``seconds``, ...)."""
    pred: Dict[str, np.ndarray]
    members: List[str]
    weights: Dict[str, float]
    alpha: float
    deploy: str
    effect: Optional[Dict[str, np.ndarray]]
    n_fits: int
    notes: List[str]
    diag: dict = field(default_factory=dict)
    # ver4-80 (R1): one entry per weight method that produced usable weights, keyed by the
    # W_OPT method name ('Analytic least-squares' = the legacy cross-validated row). ``pred`` /
    # ``weights`` / ``alpha`` above are those of the FIRST variant in the requested order, so a
    # caller that ignores ``variants`` gets exactly the pre-ver4-80 result. Empty with ``pred``.
    variants: Dict[str, "SLVariant"] = field(default_factory=dict)
    # ver4-94 (R0): optional variants written as EXTRA rows next to the ones in ``variants``, keyed
    # by tag ('pooled') and then by weight method. Empty unless a feature that adds one is on and
    # produced a usable row; the untagged fields above never depend on it.
    tagged_variants: Dict[str, Dict[str, "SLVariant"]] = field(default_factory=dict)


@dataclass
class SLVariant:
    """One Super learner row's numbers (ver4-80, R1): the out-of-fold matrix combined by ONE
    weight method.

    ``method`` is the W_OPT method name; ``weights`` the member weights it chose (sum 1; for
    'Linear transformation' they are the explanatory weights of its network, which is not a
    weighted sum); ``pred`` the 'train' / 'valid' / 'test' predictions as in
    :class:`SLResult` (train and valid are out of fold for the weighted-sum methods); ``alpha``
    the cross-validated shrinkage level of the analytic method (NaN for the others, whose
    shrinkage is their own hyperparameter); ``seconds`` the time spent on this method's weights."""
    method: str
    weights: Dict[str, float]
    pred: Dict[str, np.ndarray]
    alpha: float
    seconds: float


def sl_unit_counts(n_members: int, settings: dict, has_valid: bool,
                   n_pooled: int = 0) -> Tuple[int, int]:
    """``(oof, refit)``: how many out-of-fold and refit fit units the Super learner plans for
    `n_members` marker-channel members. ``oof = n_members * folds``; ``refit = n_members`` only
    with ``deploy='refit'`` and a validation split (`has_valid`), else 0. Pure; the GUI caption
    and ``pipeline_utils.sl_fit_estimate`` agree with it.

    ver4-94 (R3): `n_pooled` members that also get pooled columns add ``n_pooled * folds``
    out-of-fold units and ``n_pooled`` refits (the pooled refit is needed whatever ``deploy`` is,
    because the pooled test column has no Step 9 counterpart).

    pool-speed: the pooled columns use ``pop_folds`` folds (0 = ``folds``) and, with
    ``pop_deploy='cross_fit'``, make no refit."""
    cfg = dict(_SL_DEFAULTS)
    if isinstance(settings, dict):
        cfg.update(settings)
    n = max(0, int(n_members))
    pooled = max(0, int(n_pooled))
    pop_folds = int(cfg.get('pop_folds', 0) or 0) or int(cfg['folds'])
    oof = n * int(cfg['folds']) + pooled * pop_folds
    refit = ((n if (has_valid and cfg['deploy'] == 'refit') else 0)
             + (0 if str(cfg.get('pop_deploy', 'cross_fit')) == 'cross_fit' else pooled))
    return oof, refit


def sl_tuning_folds(n_rows: int, settings: dict, replicate: int = 0) -> List[np.ndarray]:
    """ver4-85: the out-of-fold partition used to SCORE hyperparameter candidates when
    ``settings['oof_tuning']`` is on - ``settings['folds']`` disjoint arrays of row POSITIONS
    0..n_rows-1 of the task's training split (sorted within each fold), passed to
    ``models.hyperparameter_tuning.tune_model_hyperparameters(oof_folds=...)``.

    Why its own seed: with ``_SL_TUNE_RESHUFFLE`` (the default, chosen on ice_sim27 evidence) the
    partition is reshuffled relative to the Super learner's own one (``seed + _SL_SEED_OFFSET``),
    so the member weights are fitted on out-of-fold predictions that did not take part in choosing
    the hyperparameters. The replicate enters the seed, so replicates draw different partitions;
    nothing depends on sharding (I9). Raises ValueError when there are fewer than
    ``10 * folds`` rows (the caller checks first and leaves the model untuned with a NOTE)."""
    cfg = dict(_SL_DEFAULTS)
    if isinstance(settings, dict):
        cfg.update(settings)
    n, k = int(n_rows), int(cfg['folds'])
    if n < 10 * k:
        raise ValueError(f"out-of-fold tuning needs at least 10 x folds ({10 * k}) training rows, "
                         f"got {n}")
    if _SL_TUNE_RESHUFFLE:
        seed = int(cfg['seed']) + _SL_TUNE_SEED_OFFSET + 7919 * max(int(replicate), 0)
    else:
        seed = int(cfg['seed']) + _SL_SEED_OFFSET
    return [np.sort(f) for f in _positional_folds(n, k, seed)]


def _empty_result(notes: List[str], deploy: str, t_start: float, **diag) -> SLResult:
    """The 'write no row' result: every field empty, `notes` kept for the caller to print."""
    diag['seconds'] = time.time() - t_start
    return SLResult(pred={}, members=[], weights={}, alpha=float('nan'), deploy=deploy,
                    effect=None, n_fits=0, notes=notes, diag=diag)


def _tv_frame(m: IterMember) -> pd.DataFrame:
    """The member's train+valid frame (train rows first, fresh RangeIndex). An empty validation
    split (a float-ratio task, or a 'between' task without validation) gives the training rows
    alone: concatenating the empty placeholder frame would only change dtypes (I4)."""
    if m.valid.shape[0] == 0:
        return m.train.reset_index(drop=True)
    return _cv_frame(m.train, m.valid)


def _check_shapes(members: Sequence[IterMember]) -> Tuple[int, int, int]:
    """``(n_train, n_valid, n_test)`` shared by all `members`; raises ValueError (with the
    offending names and counts) when a member's frames disagree with the first member's or a
    prediction vector does not match its own frame."""
    first = members[0]
    shape = (first.train.shape[0], first.valid.shape[0], first.test.shape[0])
    for m in members:
        got = (m.train.shape[0], m.valid.shape[0], m.test.shape[0])
        if got != shape:
            raise ValueError(
                f"super learner: member {m.name!r} has frames of {got[0]}/{got[1]}/{got[2]} "
                f"train/valid/test rows, expected {shape[0]}/{shape[1]}/{shape[2]}")
        for split, frame in (('train', m.train), ('valid', m.valid), ('test', m.test)):
            arr = m.pred.get(split)
            n_pred = 0 if arr is None else len(arr)
            if n_pred != frame.shape[0]:
                raise ValueError(
                    f"super learner: member {m.name!r} has {n_pred} {split} prediction(s) for a "
                    f"{split} frame of {frame.shape[0]} row(s)")
    return shape


def _shared_target_group(members: Sequence[IterMember], tvs: Sequence[pd.DataFrame]
                         ) -> Tuple[List[int], np.ndarray, List[str]]:
    """Positions of the members whose train+valid target vector is the same (``np.allclose``,
    NaN equal), the target itself and the NOTE lines for the members left out.

    Why: the weights are fitted on ONE target vector, so every member's out-of-fold column must
    be a prediction of the same rows. The largest group of equal targets wins (ties: the group
    met first), so one odd member cannot knock out all the others."""
    targets = [tv.iloc[:, -1].to_numpy(dtype=float) for tv in tvs]
    groups: List[List[int]] = []
    for j, y in enumerate(targets):
        for group in groups:
            if np.allclose(targets[group[0]], y, equal_nan=True):
                group.append(j)
                break
        else:
            groups.append([j])
    best = max(groups, key=len)   # max() keeps the first of equally long groups
    notes = [f"member {members[j].name!r} is left out: its train+valid target differs from the "
             f"one shared by the other members" for j in range(len(members)) if j not in best]
    return best, targets[best[0]], notes


def _wopt_stub_tables(names: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(record, effect, weight)`` placeholders for a call of a weight-optimisation function.

    Those functions were written for ``GP()``'s accumulators: each duplicates the last row of
    ``record`` and ``effect`` and appends a row to ``weight``. Only the weight row is wanted
    here, so they get one-row stand-ins (one marker column, one effect row per member) whose
    other outputs the caller discards."""
    k = len(names)
    record = pd.DataFrame({'model': ['sl_stub'], 'Pearson correlation': [np.nan],
                           'MSE': [np.nan]})
    effect = pd.DataFrame({'population': ['sl'] * k, 'phenotype': ['sl'] * k,
                           'ratio': ['sl'] * k, 'sample': [0] * k, 'model': list(names),
                           'stub_marker': [1.0] * k})
    return record, effect, pd.DataFrame()


def _optimiser_variant(method: str, P_k: np.ndarray, y_tv: np.ndarray, test_mat: np.ndarray,
                       names: Sequence[str], n_train: int, hparams, seed: int,
                       notes: List[str]) -> Optional[SLVariant]:
    """The Super learner's weights from ONE weight-optimisation method, run on the out-of-fold
    matrix (ver4-80, R1). Returns ``None`` (and appends a NOTE) when the method cannot run.

    Why: the Super learner used to combine its out-of-fold columns with one fixed rule, whatever
    weight optimiser the user ticked. Here the method's OWN function
    (``Nelder_Mead``, ``Bayesian``, ``Linear_transformation``) searches on the out-of-fold
    predictions of every labelled row instead of on the validation split, with the method's own
    hyperparameters, validation floor and shrinkage (`hparams` is its ``HYPERPARAMETERS_OPT``
    entry). BLUE weights are not applied to this search (the same limit as the analytic path).

    Nelder Mead / Bayesian optimisation: the function is called with the out-of-fold frame as its
    validation split; its weight row is read back and applied as a weighted sum, so
    ``pred['train']`` / ``['valid']`` are out of fold, as for the analytic row.
    Linear transformation: the combiner is a small network, not a weighted sum. It is fitted on
    a seeded 80 % of the out-of-fold rows and stops early on the other 20 %; ONE call predicts
    the deployed test columns and, appended after them, the out-of-fold rows (train / valid
    predictions are therefore the network's output on out-of-fold inputs, fitted on those same
    rows - not out of fold for the combiner). Its member weights are its SHAP importances."""
    t0 = time.time()
    k = len(names)
    n_tv = P_k.shape[0]
    n_test = test_mat.shape[0]
    if not hparams:
        notes.append(f"weight method {method!r} is skipped: its hyperparameters are not "
                     f"available to the Super learner (no row for it)")
        return None
    names = list(names)
    try:
        record, effect, weight = _wopt_stub_tables(names)
        oof = pd.DataFrame(P_k, columns=names)
        oof['actual'] = y_tv
        oof['id'] = np.arange(n_tv)
        if method == 'Nelder Mead':
            from models.Nelder_Mead import Nelder_Mead
            _r, _e, _te, _va, _tr, w_out = Nelder_Mead(
                oof.iloc[:n_train].copy(), oof.copy(), oof.copy(), record, effect, weight, names,
                list(hparams), asreml_accuracy_ctx=None, sample_weight_ctx=None)
        elif method == 'Bayesian optimisation':
            from models.Bayesian_optimisation import Bayesian
            _r, _e, _te, _va, _tr, w_out = Bayesian(
                oof.iloc[:n_train].copy(), oof.copy(), oof.copy(), record, effect, weight, names,
                list(hparams), asreml_accuracy_ctx=None, sample_weight_ctx=None)
        elif method == 'Linear transformation':
            from models.Linear_transformation import Linear_transformation
            order = np.random.default_rng(int(seed) + _SL_WOPT_OFFSET).permutation(n_tv)
            n_hold = min(max(1, int(round(_SL_LT_HOLDOUT * n_tv))), n_tv - 1)
            hold, fit = np.sort(order[:n_hold]), np.sort(order[n_hold:])
            test_part = pd.DataFrame(test_mat, columns=names)
            test_part['actual'] = 0.0      # placeholder: no test phenotype enters (I7)
            test_part['id'] = -1
            stacked = pd.concat([test_part, oof], ignore_index=True)
            _r, _e, stacked, _va, _tr, w_out = Linear_transformation(
                oof.iloc[fit].reset_index(drop=True), oof.iloc[hold].reset_index(drop=True),
                stacked, record, effect, weight, names, list(hparams),
                asreml_accuracy_ctx=None, sample_weight_ctx=None)
        else:
            notes.append(f"weight method {method!r} is not known to the Super learner")
            return None
        w = np.asarray(w_out.iloc[-1][names].astype(float).to_numpy(), dtype=float)
        if w.shape[0] != k or not np.all(np.isfinite(w)) or w.sum() == 0.0:
            raise ValueError(f"unusable member weights {w.tolist()!r}")
        w = w / w.sum()
        if method == 'Linear transformation':
            column = np.asarray(stacked['Linear transformation'], dtype=float)
            if column.shape[0] != n_test + n_tv or not np.all(np.isfinite(column)):
                raise ValueError("the network returned missing or non-finite predictions")
            test_pred, tv_pred = column[:n_test], column[n_test:]
        else:
            test_pred, tv_pred = test_mat @ w, P_k @ w
    except Exception as err:   # an optimiser can fail in many ways; the other rows must survive
        notes.append(f"weight method {method!r} failed - no row for it "
                     f"({type(err).__name__}: {err})")
        return None
    return SLVariant(
        method=method, weights={n: float(x) for n, x in zip(names, w)},
        pred={'train': tv_pred[:n_train], 'valid': tv_pred[n_train:], 'test': test_pred},
        alpha=float('nan'), seconds=time.time() - t0)


def _source_frame(m: IterMember, src: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], str]:
    """The source lines as the frame one member trains on: ``src`` restricted to the member's own
    columns (``m.train.columns``: its marker view, phenotype last, I4 / I6) with a fresh
    RangeIndex. Returns ``(frame, '')``, or ``(None, <first missing column>)`` when the source
    lacks a column the member uses (the member then gets no pooled columns)."""
    cols = list(m.train.columns)
    missing = [c for c in cols if c not in src.columns]
    if missing:
        return None, str(missing[0])
    return src[cols].reset_index(drop=True), ''


def _pooled_units(usable: Sequence[IterMember], tvs: Sequence[pd.DataFrame],
                  srcs: Dict[int, pd.DataFrame], folds: Sequence[np.ndarray], n_tv: int,
                  cfg: dict, crossfit: bool = False) -> Tuple[List[_FitUnit], List[tuple]]:
    """The fit units of the pooled columns (ver4-94, R3), with their tags.

    Per member ``j`` in `srcs`: for each fold, a fit on (train+valid minus the fold) + its source
    lines, predicting the fold (out of fold: no fold line and no own validation / test line is in
    that training set); then ONE refit on train+valid + source, which gives the pooled test
    column. Tags are ``('poof', j, held)`` and ``('prefit', j, None)``. Pooled units never use BLUE
    fitting weights (the caller refuses the feature under BLUE weighting), the test stub is used
    for the out-of-fold units where the model allows it, and no explainability is computed.

    ver4-97: each unit's training frame is a :class:`_LazyTrainFrame` (references only) that the
    worker concatenates when the unit runs - the frames are 20x+ the task's own size when many
    populations are borrowed, and ``K * (folds + 1)`` eager copies were held at once.

    pool-speed: `folds` is the partition of the POOLED columns (``pop_folds``; it need not be the own
    columns' partition). With `crossfit` (``pop_deploy='cross_fit'``) the fold units also return
    their test predictions (no test stub) and no refit unit is made: the pooled test column is the
    mean of the fold models' test predictions (``_pooled_collect``)."""
    from model_registry import exact_kernel_search_capable, search_test_stub_safe
    units: List[_FitUnit] = []
    tags: List[tuple] = []
    per_fold_centre = bool(cfg.get('pop_center_per_fold', False)) and bool(cfg.get('pop_center', True))
    for f, held in enumerate(folds):
        keep = np.setdiff1d(np.arange(n_tv), held)
        for j in sorted(srcs):
            m = usable[j]
            shift = 0.0
            if per_fold_centre:
                # ver4-100: the source was centred on the mean of ALL own train+valid lines; this
                # fold's model must not know the held-out lines, so move it to the mean of the
                # kept lines (the final refit / cross-fit test column keep the original centring
                # in 'refit' mode).
                y_own = tvs[j].iloc[:, -1].to_numpy(dtype=float)
                shift = float(np.mean(y_own[keep]) - np.mean(y_own))
            exact = cfg['oof_engine'] == 'exact' and exact_kernel_search_capable(m.base)
            units.append(_FitUnit(
                model=m.base, params=_disable_explainability(m.base, m.params),
                split_tag=f'sl_pf{f}',
                train=_LazyTrainFrame(tvs[j], keep, srcs[j], shift),
                valid=tvs[j].iloc[held], fidelity='oof', keep_full=False,
                engine='exact_kernel' if exact else 'runner',
                test_stub=(not crossfit and bool(search_test_stub_safe(m.base))),
                keep_test=bool(crossfit), cv_weights=False, test_frame=m.test))
            tags.append(('poof', j, held))
    for j in ([] if crossfit else sorted(srcs)):
        m = usable[j]
        units.append(_FitUnit(
            model=m.base, params=_disable_explainability(m.base, m.params),
            split_tag='sl_prefit', train=_LazyTrainFrame(tvs[j], None, srcs[j]),
            valid=m.valid, fidelity='full', keep_full=True, cv_weights=False,
            test_frame=m.test))
        tags.append(('prefit', j, None))
    return units, tags


def _pooled_collect(ptags: Sequence[tuple], punits: Sequence[_FitUnit], presults: Sequence,
                    usable: Sequence[IterMember], n_tv: int, notes: List[str],
                    crossfit: bool = False
                    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """``(oof_cols, test_cols)``: per member index, the complete pooled out-of-fold column
    (length n_tv) and the pooled test column. A member with any failed, missing, wrong-length or
    non-finite pooled unit is left out of both (a NOTE names it); nothing here raises.

    pool-speed: with `crossfit` there is no refit unit; the test column is the mean of the member's
    fold units' ``predicted_test`` (every fold must have returned a finite column of the test
    length, else the member is left out)."""
    oof: Dict[int, np.ndarray] = {}
    fold_tests: Dict[int, List[np.ndarray]] = {}
    for j in sorted({t[1] for t in ptags}):
        oof[j] = np.full(n_tv, np.nan)
    test_cols: Dict[int, np.ndarray] = {}
    failed: Dict[int, str] = {}
    for (kind, j, held), unit, res in zip(ptags, punits, presults):
        if kind == 'poof':
            vec = _oof_vector(res, held.shape[0])
            if vec is None:
                failed.setdefault(j, f"a pooled out-of-fold fit ({unit.split_tag}) failed or "
                                     f"returned missing, wrong-length or non-finite values")
                continue
            oof[j][held] = vec
            if crossfit:
                p_test = None if not isinstance(res, dict) else res.get('predicted_test')
                if p_test is not None:
                    p_test = np.asarray(p_test, dtype=float).ravel()
                if (p_test is None or p_test.shape[0] != usable[j].test.shape[0]
                        or not np.all(np.isfinite(p_test))):
                    failed.setdefault(j, f"a pooled fold fit ({unit.split_tag}) returned no "
                                         f"finite test predictions (cross-fit)")
                    continue
                fold_tests.setdefault(j, []).append(p_test)
        else:
            cols = _columns_from_result(res, unit, usable[j].test)
            if cols is None:
                failed.setdefault(j, "the pooled refit failed or returned columns of the wrong "
                                     "length or non-finite values")
                continue
            test_cols[j] = cols['test']
    if crossfit:
        n_f = {j: sum(1 for t in ptags if t[0] == 'poof' and t[1] == j) for j in oof}
        for j, lst in fold_tests.items():
            if j not in failed and len(lst) == n_f.get(j, -1):
                test_cols[j] = np.mean(lst, axis=0)
    good_oof: Dict[int, np.ndarray] = {}
    good_test: Dict[int, np.ndarray] = {}
    for j in sorted(oof):
        if j in failed:
            notes.append(f"pooled columns of member {usable[j].name!r} are left out: {failed[j]}")
        elif j not in test_cols or not np.all(np.isfinite(oof[j])):
            notes.append(f"pooled columns of member {usable[j].name!r} are left out: incomplete")
        else:
            good_oof[j] = oof[j]
            good_test[j] = test_cols[j]
    return good_oof, good_test


def _pooled_variant(P_k: np.ndarray, test_mat: np.ndarray, names: Sequence[str],
                    kept: Sequence[int], pooled_oof: Dict[int, np.ndarray],
                    pooled_test: Dict[int, np.ndarray], y_tv: np.ndarray, n_train: int,
                    cfg: dict, seed: int, n_source: int, notes: List[str]
                    ) -> Optional[Tuple[SLVariant, dict]]:
    """The pooled Super-learner row (ver4-94, R3): weights over ``[own | pooled]`` columns.

    `kept` are the member indices of the own columns (`P_k` / `test_mat`, in `names` order).
    Only members that also have a pooled column contribute one. The weights come from the same
    cross-validated shrinkage path as the own row, with the prior "own columns equal, pooled
    columns 0" (so alpha = 0 is the own-lines equal-weight row, and an unusable result falls back
    to that, never to equal weights over every column). Member weights in the returned variant
    sum a member's own and pooled weight (the ``Weight.csv`` schema stays the member names); the
    source share is in the returned diag. Returns None, with a NOTE, when no pooled column is
    usable."""
    t0 = time.time()
    pj = [j for j in kept if j in pooled_oof and j in pooled_test]
    if not pj:
        notes.append("no member has complete pooled columns - no pooled row")
        return None
    k, kp = len(kept), len(pj)
    P_aug = np.column_stack([P_k] + [pooled_oof[j] for j in pj])
    T_aug = np.column_stack([test_mat] + [pooled_test[j] for j in pj])
    prior = np.concatenate([np.full(k, 1.0 / k), np.zeros(kp)])
    w_raw, alpha, wdiag = shrinkage_path_weights(
        P_aug, y_tv, direction='analytic', rule=str(cfg['shrink_rule']), seed=seed, prior=prior)
    w = np.clip(np.asarray(w_raw, dtype=float).ravel(), 0.0, None)
    if w.shape[0] != k + kp or not np.all(np.isfinite(w)) or w.sum() <= 0.0:
        notes.append("the pooled cross-validated weights were unusable - the own columns get "
                     "equal weights and the pooled columns none")
        w, alpha = prior.copy(), float('nan')
    w = w / w.sum()
    if wdiag.get('note'):
        notes.append(f"pooled member weights - {wdiag['note']}")
    member_w = {}
    for c, name in enumerate(names):
        extra = float(w[k + pj.index(kept[c])]) if kept[c] in pj else 0.0
        member_w[name] = float(w[c]) + extra
    tv_pred = P_aug @ w
    variant = SLVariant(
        method=SL_ANALYTIC_METHOD, weights=member_w,
        pred={'train': tv_pred[:n_train], 'valid': tv_pred[n_train:], 'test': T_aug @ w},
        alpha=float(alpha), seconds=time.time() - t0)
    pooled_names = [names[kept.index(j)] for j in pj]
    info = dict(
        n_source=int(n_source), source_share=float(w[k:].sum()), members=pooled_names,
        pooled_oof_r={n: _fit_stats(y_tv, pooled_oof[j])[0] for n, j in zip(pooled_names, pj)},
        source_weight={n: float(w[k + i]) for i, n in enumerate(pooled_names)},
        alpha=float(alpha))
    return variant, info


def _pooled_member_variants(names: Sequence[str], kept: Sequence[int],
                            pooled_oof: Dict[int, np.ndarray], pooled_test: Dict[int, np.ndarray],
                            n_train: int) -> Dict[str, SLVariant]:
    """One :class:`SLVariant` per member that has complete pooled columns (ver4-96): that
    member's OWN pooled prediction, no weighting.

    ``pred['test']`` is the member refitted on train+valid plus the source lines (the pooled test
    column the pooled Super learner already uses), ``pred['train']`` / ``pred['valid']`` its pooled
    OUT-OF-FOLD columns over the train / valid rows. Next to the plain member's row it is the
    like-for-like test of the borrowed lines (same model, same hyperparameters, same split; only
    the other populations' lines differ). It costs no extra fit. The ``weights`` field is 1 on
    this member and 0 on the others (so every consumer that normalises it keeps working) and is
    never written (the caller writes metric and prediction rows only: a
    member has no weight, effect or interaction row of its own). Pure."""
    out: Dict[str, SLVariant] = {}
    for c, name in enumerate(names):
        j = kept[c]
        if j not in pooled_oof or j not in pooled_test:
            continue
        oof = np.asarray(pooled_oof[j], dtype=float).ravel()
        out[name] = SLVariant(
            method=name, weights={n: (1.0 if n == name else 0.0) for n in names},
            pred={'train': oof[:n_train], 'valid': oof[n_train:],
                  'test': np.asarray(pooled_test[j], dtype=float).ravel()},
            alpha=float('nan'), seconds=0.0)
    return out


def _pooled_gated_member_variants(P_k: np.ndarray, test_mat: np.ndarray, names: Sequence[str],
                                  kept: Sequence[int], pooled_oof: Dict[int, np.ndarray],
                                  pooled_test: Dict[int, np.ndarray], y_tv: np.ndarray,
                                  n_train: int, cfg: dict, seed: int, notes: List[str]
                                  ) -> Tuple[Dict[str, SLVariant], Dict[str, float]]:
    """One GATED :class:`SLVariant` per member that has complete pooled columns (pool-fix), and
    ``{member name: weight on its pooled column}``.

    The like-for-like single-model counterpart of the pooled Super-learner row: for member ``c``
    the two columns ``[own | pooled]`` (own = the member's own out-of-fold column ``P_k[:, c]`` and
    its test column ``test_mat[:, c]``; pooled = its pooled out-of-fold and test columns) are
    combined by the same cross-validated shrinkage path as the pooled row
    (``shrinkage_path_weights``, analytic direction, same rule and seed) with the prior ``[1, 0]``:
    alpha = 0 is the plain model, and the pooled column enters only as far as cross-validation
    supports it. An unusable result falls back to the prior (the plain model). Same rows, columns
    and fits as the ungated member rows, so it costs no extra fit. The ``weights`` field is 1 on
    this member and 0 on the others (like the ungated member row; never written). Pure."""
    out: Dict[str, SLVariant] = {}
    share: Dict[str, float] = {}
    prior = np.array([1.0, 0.0])
    for c, name in enumerate(names):
        j = kept[c]
        if j not in pooled_oof or j not in pooled_test:
            continue
        t0 = time.time()
        P2 = np.column_stack([P_k[:, c], np.asarray(pooled_oof[j], dtype=float).ravel()])
        T2 = np.column_stack([test_mat[:, c], np.asarray(pooled_test[j], dtype=float).ravel()])
        try:
            w_raw, alpha, wdiag = shrinkage_path_weights(
                P2, y_tv, direction='analytic', rule=str(cfg['shrink_rule']), seed=seed,
                prior=prior)
            w = np.clip(np.asarray(w_raw, dtype=float).ravel(), 0.0, None)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
            notes.append(f"gated pooled row of member {name!r} uses the plain model "
                         f"({type(err).__name__}: {err})")
            w, alpha = prior.copy(), float('nan')
        if w.shape[0] != 2 or not np.all(np.isfinite(w)) or w.sum() <= 0.0:
            notes.append(f"the gated pooled weights of member {name!r} were unusable - the plain "
                         f"model is used")
            w, alpha = prior.copy(), float('nan')
        w = w / w.sum()
        tv_pred = P2 @ w
        out[name] = SLVariant(
            method=name, weights={n: (1.0 if n == name else 0.0) for n in names},
            pred={'train': tv_pred[:n_train], 'valid': tv_pred[n_train:], 'test': T2 @ w},
            alpha=float(alpha), seconds=time.time() - t0)
        share[name] = float(w[1])
    return out, share


def _pooled_method_variant(method: str, own_var: "SLVariant", P_k: np.ndarray,
                           test_mat: np.ndarray, names: Sequence[str], kept: Sequence[int],
                           pooled_oof: Dict[int, np.ndarray], pooled_test: Dict[int, np.ndarray],
                           y_tv: np.ndarray, n_train: int, hparams, cfg: dict, seed: int,
                           notes: List[str]) -> Optional[Tuple["SLVariant", dict]]:
    """The pooled row of a NON-analytic weight method (pool-methods): a two-stage gate.

    Why two stages: Nelder Mead, Bayesian optimisation and Linear transformation have no prior, so
    they cannot start from "own columns only" the way the analytic pooled row does. Instead the
    method's own weighting is run twice with the same settings and seed, then a gate chooses:

    1. the method on the own out-of-fold columns (``own_var``, the row that already exists);
    2. the same method on the POOLED out-of-fold columns (a member without a complete pooled column
       keeps its own column there, so every member is in both runs);
    3. the cross-validated analytic shrinkage path over the two resulting predictions
       ``[own | pooled]`` with the prior ``[1, 0]``: alpha = 0 is exactly the method's own row, and
       the pooled prediction gets weight only as far as cross-validation supports it.

    The returned member weights are ``(1 - s) * own + s * pooled`` (they sum to 1, so the
    ``Weight.csv`` schema is unchanged); ``s`` is in the returned diag. No model fit is added; the
    method runs once more on the pooled columns. Returns ``None`` (with a NOTE) when the method
    cannot run on them, in which case no pooled row is written for this method.

    Caveat (documented, not hidden): for Nelder Mead / Bayesian the weights are fitted on the same
    out-of-fold rows the gate then reads, and for Linear transformation the network's rows are
    in-sample for the combiner, so both stage predictions are slightly optimistic. The two stages
    are optimistic by a similar amount, which is why the gate compares them, but the effect is
    largest for Linear transformation."""
    t0 = time.time()
    members = [c for c, j in enumerate(kept) if j in pooled_oof and j in pooled_test]
    if not members:
        notes.append(f"no member has complete pooled columns - no pooled {method!r} row")
        return None
    P_pool, T_pool = P_k.copy(), test_mat.copy()
    for c in members:
        P_pool[:, c] = np.asarray(pooled_oof[kept[c]], dtype=float).ravel()
        T_pool[:, c] = np.asarray(pooled_test[kept[c]], dtype=float).ravel()
    pool_var = _optimiser_variant(method, P_pool, y_tv, T_pool, names, n_train, hparams, seed,
                                  notes)
    if pool_var is None:
        notes.append(f"the pooled {method!r} row is not written: the method could not weight the "
                     f"pooled columns")
        return None
    own_tv = np.concatenate([np.asarray(own_var.pred['train'], dtype=float).ravel(),
                             np.asarray(own_var.pred['valid'], dtype=float).ravel()])
    pool_tv = np.concatenate([np.asarray(pool_var.pred['train'], dtype=float).ravel(),
                              np.asarray(pool_var.pred['valid'], dtype=float).ravel()])
    P2 = np.column_stack([own_tv, pool_tv])
    T2 = np.column_stack([np.asarray(own_var.pred['test'], dtype=float).ravel(),
                          np.asarray(pool_var.pred['test'], dtype=float).ravel()])
    prior = np.array([1.0, 0.0])
    try:
        w_raw, alpha, wdiag = shrinkage_path_weights(
            P2, y_tv, direction='analytic', rule=str(cfg['shrink_rule']), seed=seed, prior=prior)
        g = np.clip(np.asarray(w_raw, dtype=float).ravel(), 0.0, None)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
        notes.append(f"the pooled {method!r} gate uses the own row ({type(err).__name__}: {err})")
        g, alpha = prior.copy(), float('nan')
    if g.shape[0] != 2 or not np.all(np.isfinite(g)) or g.sum() <= 0.0:
        notes.append(f"the pooled {method!r} gate weights were unusable - the own row is used")
        g, alpha = prior.copy(), float('nan')
    g = g / g.sum()
    share = float(g[1])
    tv_pred = P2 @ g
    member_w = {n: float(g[0] * own_var.weights.get(n, 0.0) + g[1] * pool_var.weights.get(n, 0.0))
                for n in names}
    total = sum(member_w.values())
    if total > 0.0:
        member_w = {n: v / total for n, v in member_w.items()}
    variant = SLVariant(
        method=method, weights=member_w,
        pred={'train': tv_pred[:n_train], 'valid': tv_pred[n_train:], 'test': T2 @ g},
        alpha=float(alpha), seconds=time.time() - t0 + float(own_var.seconds or 0.0))
    info = dict(
        source_share=share, alpha=float(alpha),
        source_weight={names[c]: share * float(pool_var.weights.get(names[c], 0.0))
                       for c in members},
        pooled_oof_r=_fit_stats(y_tv, pool_tv)[0], own_oof_r=_fit_stats(y_tv, own_tv)[0])
    return variant, info


def _pooled_progress_printer(label: str):
    """ver4-97: a ``_run_units`` progress callback that prints one line per finished pooled fit
    (model, unit, own seconds, elapsed, a rough time left). The pooled fits train on every
    borrowed line, so they are many times slower than the task's own fits and, without these
    lines, the log is silent for as long as the stage runs (it looks frozen)."""
    t0 = time.time()

    def _print(done, total, unit, res, secs):
        elapsed = time.time() - t0
        eta = elapsed / max(done, 1) * max(total - done, 0)
        n_rows = unit.train.shape[0] if hasattr(unit.train, 'shape') else '?'
        took = f"{secs:.0f}s" if secs is not None else "n/a"
        state = "ok" if isinstance(res, dict) else "FAILED"
        print(f"[super_learner] {label}: fit {done}/{total} {state} ({unit.model}, "
              f"{unit.split_tag}, {n_rows} training rows, {took}); elapsed {elapsed / 60:.1f} min, "
              f"about {eta / 60:.1f} min left.", flush=True)
    return _print


def run_super_learner(members: Sequence[IterMember], *, settings: dict, runner, RESULT_NAME: str,
                      runner_ctx: dict, use_cv_weights: bool,
                      weight_methods: Optional[Sequence[str]] = None,
                      weight_hparams: Optional[Dict[str, Sequence]] = None,
                      extra_sources: Optional[Dict[str, pd.DataFrame]] = None) -> SLResult:
    """The Super learner for ONE task and ONE ensemble group.

    `members` are the group's Step 9 fits (at least 2); `settings` a resolved SUPER_LEARNER dict
    (``pipeline_utils.resolve_super_learner``); `runner` a ``ModelTrialRunner``; `runner_ctx` the
    keyword arguments of ``joint_tuning._run_units`` (``n_jobs``, ``worker_init``,
    ``worker_init_args``); `use_cv_weights` True when the task runs under BLUE weighting and its
    context carries the train+valid fitting weights (every unit then fits through
    ``runner.with_cv_weights()``, as the joint search's CV units do).

    ver4-80 (R1): `weight_methods` lists the W_OPT method names to build a row for (default
    ``['Analytic least-squares']`` = the single legacy cross-validated row, so a caller that does
    not pass it gets the pre-ver4-80 result); `weight_hparams` maps a method name to its
    ``HYPERPARAMETERS_OPT`` entry (needed for every method but the analytic one). The fits are
    made ONCE; each method only re-weights the same out-of-fold matrix. ``SLResult.variants``
    holds one :class:`SLVariant` per method that worked; ``pred`` / ``weights`` / ``alpha`` are
    the first one's.

    Algorithm (``K`` members, ``F`` = ``folds``, ``TV`` = train rows then valid rows)::

        folds = F disjoint positional folds of TV  (seed + _SL_SEED_OFFSET, shared by all members)
        OOF   : for member m, fold f: fit on TV minus f at m's Step 9 hyperparameters, predict f
                -> P (|TV| x K)                                     K * F fits, explainability off
        w, a  = shrinkage_path_weights(P, y_TV, 'analytic', shrink_rule)   (w >= 0, sum 1)
        test  = 'refit'    : refit of m on TV (K fits)               column_m
                'cross_fit': mean of m's F fold models on the test rows
                'step9'    : m's Step 9 column (no validation split: nothing to add)
        pred  = column matrix @ w ;  train / valid columns = rows of P @ w (out of fold)

    All units run in ONE ``_run_units`` call (``stop_on_failure=False``). Out-of-fold units use
    the parity-guarded 1-row test stub of the BGLR models unless the test columns are needed
    (``deploy='cross_fit'``), and with ``oof_engine='exact'`` the closed-form kernel engine for
    rrBLUP / RKHS. Early-stopping models see the real phenotype of the held-out rows, the same
    precedent as ``iterative_ensemble``'s out-of-fold units.

    Degradation, never an exception (a NOTE is appended and, if fewer than 2 members remain, no
    row is written): a bio-prior member or one without resolved hyperparameters is left out
    (I6); a member whose train+valid target differs from the shared one is left out; fewer than
    ``10 * folds`` rows, or a plan above ``max_fits``, writes no row before anything is fitted;
    a failed or non-finite out-of-fold unit drops that member; a failed refit deploys the
    member's cross-fit mean or, failing that, its Step 9 test column. BLUE sample weights are
    used for the fits but not for the weight estimate (``shrinkage_path_weights`` is
    unweighted) - a NOTE says so.

    ver4-94 (R3): `extra_sources` may hold ``'pooled'``: the other populations' labelled lines
    (``models.relatedness.build_population_source``; markers, then phenotype last). Every member
    whose columns it covers then also gets pooled columns, and ``SLResult.tagged_variants['pooled']``
    holds one extra row (analytic combiner only) - see the module docstring. It is skipped, with
    a NOTE and no effect on the other rows, under BLUE weighting, when the analytic method is
    not among `weight_methods`, and when the plan would exceed ``max_fits`` (the pooled units are
    dropped first, ``_SL_OPTIONAL_DROP_ORDER``). The unchanged rows never read a pooled column.

    Raises ValueError only on a contract violation: fewer than 2 `members` passed, or members
    whose frames or predictions have different row counts."""
    t_start = time.time()
    cfg = dict(_SL_DEFAULTS)
    cfg.update(settings if isinstance(settings, dict) else {})
    members = list(members)
    if len(members) < 2:
        raise ValueError(f"super learner needs at least 2 members, got {len(members)}")
    n_train, n_valid, n_test = _check_shapes(members)
    n_folds = int(cfg['folds'])
    seed = int(cfg['seed'])
    fit_cap = int(cfg['max_fits'] or 0)
    has_valid = n_valid > 0
    notes: List[str] = []

    from model_registry import exact_kernel_search_capable, search_test_stub_safe

    # ---- which members take part ------------------------------------------------------------
    usable: List[IterMember] = []
    for m in members:
        if not m.marker_channel:
            notes.append(f"member {m.name!r} is left out: it is a bio-prior model (never fitted "
                         f"out of fold, I6)")
        elif not m.params:
            notes.append(f"member {m.name!r} is left out: its Step 9 hyperparameters could not "
                         f"be resolved")
        else:
            usable.append(m)

    deploy = str(cfg['deploy'])
    if deploy == 'refit' and not has_valid:
        deploy = 'step9'
        notes.append("the task has no validation split, so there is nothing to refit on - the "
                     "test column is the Step 9 columns at the cross-validated weights "
                     "(deploy='step9'); without new phenotypes the Super learner can only "
                     "re-weight the models; expect it to match the best model, not to exceed "
                     "it.")
    if len(usable) < 2:
        notes.append(f"fewer than 2 usable members ({[m.name for m in usable]!r})")
        return _empty_result(notes, deploy, t_start)

    tvs = [_tv_frame(m) for m in usable]
    keep_pos, y_tv, target_notes = _shared_target_group(usable, tvs)
    notes.extend(target_notes)
    usable = [usable[j] for j in keep_pos]
    tvs = [tvs[j] for j in keep_pos]
    n_tv = n_train + n_valid
    if len(usable) < 2:
        notes.append(f"fewer than 2 members share one train+valid target "
                     f"({[m.name for m in usable]!r})")
        return _empty_result(notes, deploy, t_start)
    if n_tv < 10 * n_folds:
        notes.append(f"the train+valid split has {n_tv} row(s), fewer than 10 x folds "
                     f"({10 * n_folds})")
        return _empty_result(notes, deploy, t_start)

    n_oof, n_refit = sl_unit_counts(len(usable), cfg, has_valid)
    n_plan = n_oof + n_refit
    if fit_cap > 0 and n_plan > fit_cap:
        notes.append(f"the plan needs {n_plan} fit(s) ({n_oof} out-of-fold + {n_refit} refit) - "
                     f"above max_fits={fit_cap}")
        return _empty_result(notes, deploy, t_start, n_planned=n_plan)
    if use_cv_weights:
        notes.append("BLUE sample weights are used for the member fits but not for the "
                     "member weights (shrinkage_path_weights is unweighted) - the weights are "
                     "unweighted estimates")

    # ---- ver4-94 (R3): which members get pooled columns ------------------------------------
    pooled_src: Dict[int, pd.DataFrame] = {}
    pooled_frame = (extra_sources or {}).get('pooled')
    # pool-speed: the pooled columns' own fold count and deploy mode (0 / 'refit' = ver4-98)
    pop_folds = int(cfg.get('pop_folds', 0) or 0) or n_folds
    pop_crossfit = str(cfg.get('pop_deploy', 'cross_fit')) == 'cross_fit'
    if pooled_frame is not None:
        if use_cv_weights:
            notes.append("pooled columns are skipped: BLUE weighting is on and the source lines "
                         "have no fitting weights")
        elif pooled_frame.shape[0] == 0:
            notes.append("pooled columns are skipped: the source has no lines")
        else:
            for j, m in enumerate(usable):
                frame_j, missing = _source_frame(m, pooled_frame)
                if frame_j is None:
                    notes.append(f"member {m.name!r} gets no pooled columns: the source lacks "
                                 f"its column {missing!r}")
                else:
                    pooled_src[j] = frame_j
        if pooled_src:
            n_pooled_units = len(pooled_src) * (pop_folds + (0 if pop_crossfit else 1))
            for optional in _SL_OPTIONAL_DROP_ORDER:
                if optional == 'pop_sources' and fit_cap > 0 and n_plan + n_pooled_units > fit_cap:
                    notes.append(f"pooled columns are dropped: they would add {n_pooled_units} "
                                 f"fit(s) to the plan of {n_plan}, above max_fits={fit_cap}")
                    pooled_src = {}

    # ---- the units: ONE parallel call -------------------------------------------------------
    folds = [np.sort(f) for f in _positional_folds(n_tv, n_folds, seed + _SL_SEED_OFFSET)]
    # a fit under BLUE weighting aligns its weights to the frame it is given: the train+valid
    # vector needs the cv view, the train-only vector (no validation split) the plain one.
    unit_cv_weights = bool(use_cv_weights) and has_valid
    units: List[_FitUnit] = []
    tags: List[tuple] = []      # ('oof', member index, held positions) | ('refit', member index)
    for f, held in enumerate(folds):
        keep = np.setdiff1d(np.arange(n_tv), held)
        for j, m in enumerate(usable):
            exact = cfg['oof_engine'] == 'exact' and exact_kernel_search_capable(m.base)
            units.append(_FitUnit(
                model=m.base, params=_disable_explainability(m.base, m.params),
                split_tag=f'sl_f{f}', train=tvs[j].iloc[keep], valid=tvs[j].iloc[held],
                fidelity='oof', keep_full=False, engine='exact_kernel' if exact else 'runner',
                test_stub=(deploy != 'cross_fit' and bool(search_test_stub_safe(m.base))),
                keep_test=(deploy == 'cross_fit'), cv_weights=unit_cv_weights,
                test_frame=m.test))
            tags.append(('oof', j, held))
    if deploy == 'refit':
        for j, m in enumerate(usable):
            params = list(m.params) if cfg['explain_refit'] else _disable_explainability(
                m.base, m.params)
            units.append(_FitUnit(
                model=m.base, params=params, split_tag='sl_refit', train=tvs[j], valid=m.valid,
                fidelity='full', keep_full=True, cv_weights=unit_cv_weights, test_frame=m.test))
            tags.append(('refit', j, None))

    n_own_units = len(units)
    ptags: List[tuple] = []
    punits: List[_FitUnit] = []
    if pooled_src:
        pfolds = folds if pop_folds == n_folds else [
            np.sort(f) for f in _positional_folds(n_tv, pop_folds, seed + _SL_SEED_OFFSET)]
        punits, ptags = _pooled_units(usable, tvs, pooled_src, pfolds, n_tv, cfg, pop_crossfit)

    # The member's own units run first and alone, so the unchanged rows never depend on the
    # pooled stage (ver4-97: the pooled units used to share this call; they are now a second
    # call with a per-fit progress line, and a failure of that call costs only the pooled row).
    results = _run_units(units, runner=runner, RESULT_NAME=RESULT_NAME, test=usable[0].test,
                         stop_on_failure=False, **(runner_ctx or {}))
    presults: List[Optional[dict]] = []
    if punits:
        n_src_rows = int(next(iter(pooled_src.values())).shape[0])
        n_own_fold = n_tv - n_tv // max(pop_folds, 1)
        _rows_txt = (f"{n_own_fold + n_src_rows} (out of fold; the test column is the mean of the "
                     f"{pop_folds} fold models, no refit)" if pop_crossfit else
                     f"{n_own_fold + n_src_rows} (out of fold) or {n_tv + n_src_rows} (refit)")
        print(f"[super_learner] pooled stage: {len(punits)} fit(s) ({pop_folds} fold(s) per member, "
              f"pop_deploy={cfg.get('pop_deploy', 'cross_fit')!r}), each on {_rows_txt} training rows: "
              f"the task's own {n_own_fold}-{n_tv} rows plus {n_src_rows} borrowed line(s), against "
              f"{n_tv - n_tv // max(n_folds, 1)} rows in the member's own out-of-fold fits. One "
              f"line per finished fit follows.", flush=True)
        try:
            presults = _run_units(punits, runner=runner, RESULT_NAME=RESULT_NAME,
                                  test=usable[0].test, stop_on_failure=False,
                                  progress=_pooled_progress_printer('pooled stage'),
                                  **(runner_ctx or {}))
        except Exception as err:        # never raises: the unchanged rows are already complete
            notes.append(f"the pooled stage failed - no pooled row is written "
                         f"({type(err).__name__}: {err})")
            ptags, punits, presults = [], [], []
    units = units + punits
    results = list(results) + list(presults)

    # ---- collect ----------------------------------------------------------------------------
    k_all = len(usable)
    P = np.full((n_tv, k_all), np.nan)
    failed: Dict[int, str] = {}
    fold_tests: Dict[int, List[np.ndarray]] = {j: [] for j in range(k_all)}
    refit_cols: Dict[int, np.ndarray] = {}
    refit_effects: Dict[int, np.ndarray] = {}
    for (kind, j, held), unit, res in zip(tags, units[:n_own_units], results[:n_own_units]):
        m = usable[j]
        if kind == 'oof':
            vec = _oof_vector(res, held.shape[0])
            if vec is None:
                failed.setdefault(j, f"an out-of-fold fit ({unit.split_tag}) failed or returned "
                                     f"missing, wrong-length or non-finite values")
                continue
            P[held, j] = vec
            if deploy == 'cross_fit':
                p_test = None if not isinstance(res, dict) else res.get('predicted_test')
                if p_test is not None:
                    p_test = np.asarray(p_test, dtype=float).ravel()
                if p_test is not None and p_test.shape[0] == n_test and np.all(np.isfinite(p_test)):
                    fold_tests[j].append(p_test)
        else:
            cols = _columns_from_result(res, unit, m.test)
            if cols is None:
                notes.append(f"the refit of {m.name!r} failed or returned columns of the wrong "
                             f"length or non-finite values")
                continue
            refit_cols[j] = cols['test']
            if cfg['explain_refit']:
                eff = res.get('effect') if isinstance(res, dict) else None
                if eff is not None:
                    vec = np.abs(np.asarray(eff, dtype=float).ravel())
                    if vec.shape[0] == m.train.shape[1] - 1 and np.all(np.isfinite(vec)):
                        refit_effects[j] = vec

    kept = [j for j in range(k_all) if j not in failed and np.all(np.isfinite(P[:, j]))]
    for j in range(k_all):
        if j in failed:
            notes.append(f"member {usable[j].name!r} is left out: {failed[j]}")
        elif j not in kept:
            notes.append(f"member {usable[j].name!r} is left out: its out-of-fold column has "
                         f"unfilled rows")
    if len(kept) < 2:
        notes.append(f"fewer than 2 members have a complete out-of-fold column "
                     f"({[usable[j].name for j in kept]!r})")
        return _empty_result(notes, deploy, t_start, n_fits=len(units))

    names = [usable[j].name for j in kept]
    P_k = P[:, kept]

    # ---- weights ----------------------------------------------------------------------------
    # ver4-80 (R1): one row per requested weight method; the legacy analytic one (the default)
    # is computed exactly as before, the others after the test columns are known.
    want = list(dict.fromkeys(str(m) for m in (weight_methods or [SL_ANALYTIC_METHOD])))
    w = None
    alpha = float('nan')
    wdiag: dict = {}
    t_analytic = time.time()
    if SL_ANALYTIC_METHOD in want:
        w_raw, alpha, wdiag = shrinkage_path_weights(
            P_k, y_tv, direction='analytic', rule=str(cfg['shrink_rule']), seed=seed)
        w = np.asarray(w_raw, dtype=float).ravel()
        if w.shape[0] != len(kept) or not np.all(np.isfinite(w)) or w.sum() <= 0.0:
            notes.append("the cross-validated weights were unusable - equal member weights are "
                         "used")
            w = np.full(len(kept), 1.0 / len(kept))
            alpha = float('nan')
        w = np.clip(w, 0.0, None)
        w = w / w.sum()
        if wdiag.get('note'):
            notes.append(f"member weights - {wdiag['note']}")
    t_analytic = time.time() - t_analytic

    # ---- the test column of every kept member -----------------------------------------------
    test_cols: List[np.ndarray] = []
    deployed_step9: List[str] = []
    for j in kept:
        m = usable[j]
        if deploy == 'refit' and j in refit_cols:
            test_cols.append(refit_cols[j])
        elif deploy == 'cross_fit' and len(fold_tests[j]) == n_folds:
            test_cols.append(np.mean(fold_tests[j], axis=0))
        else:
            test_cols.append(np.asarray(m.pred['test'], dtype=float).ravel())
            if deploy != 'step9':
                deployed_step9.append(m.name)
    if deployed_step9:
        notes.append(f"deployed Step 9 fit for {deployed_step9!r} (its "
                     f"{'refit' if deploy == 'refit' else 'cross-fit'} test column was not "
                     f"available)")
    test_mat = np.column_stack(test_cols)

    variants: Dict[str, SLVariant] = {}
    for method in want:
        if method == SL_ANALYTIC_METHOD:
            oof_pred = P_k @ w
            variants[method] = SLVariant(
                method=method, weights={n: float(x) for n, x in zip(names, w)},
                pred={'train': oof_pred[:n_train], 'valid': oof_pred[n_train:],
                      'test': test_mat @ w},
                alpha=float(alpha), seconds=t_analytic)
        else:
            variant = _optimiser_variant(
                method, P_k, y_tv, test_mat, names, n_train,
                (weight_hparams or {}).get(method), seed, notes)
            if variant is not None:
                variants[method] = variant
    if not variants:
        notes.append(f"no weight method gave usable weights ({want!r}) - no row")
        return _empty_result(notes, deploy, t_start, n_fits=len(units))
    primary = variants[next(m for m in want if m in variants)]
    pred = primary.pred
    alpha = primary.alpha

    # ---- ver4-94 (R3): the pooled row (computed AFTER the rows above, which never read it) ----
    tagged: Dict[str, Dict[str, SLVariant]] = {}
    pooled_info: Optional[dict] = None
    if ptags:
        try:
            p_oof, p_test = _pooled_collect(ptags, units[n_own_units:], results[n_own_units:],
                                            usable, n_tv, notes, pop_crossfit)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
            notes.append(f"the pooled rows failed - none are written ({type(err).__name__}: {err})")
            p_oof, p_test = {}, {}
        pj_all = [j for j in kept if j in p_oof and j in p_test]
        by_method: Dict[str, dict] = {}
        built = None
        # the analytic pooled row (ver4-94): own + pooled columns, one cross-validated combiner
        if SL_ANALYTIC_METHOD in variants and pj_all:
            try:
                built = _pooled_variant(P_k, test_mat, names, kept, p_oof, p_test, y_tv, n_train,
                                        cfg, seed, int(pooled_frame.shape[0]), notes)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
                notes.append(f"the pooled row failed - none is written "
                             f"({type(err).__name__}: {err})")
                built = None
        if built is not None:
            tagged['pooled'] = {SL_ANALYTIC_METHOD: built[0]}
            pooled_info = built[1]
            by_method[SL_ANALYTIC_METHOD] = dict(
                source_share=pooled_info['source_share'], alpha=pooled_info['alpha'],
                source_weight=dict(pooled_info['source_weight']))
        # pool-methods: the other weight methods, each as a two-stage gate (own row vs the
        # method's weighting of the pooled columns); never read by, and never changes, the rows
        # above. A failure leaves that method without a pooled row.
        if pj_all:
            for method in [m for m in want if m in variants and m != SL_ANALYTIC_METHOD]:
                try:
                    two = _pooled_method_variant(
                        method, variants[method], P_k, test_mat, names, kept, p_oof, p_test, y_tv,
                        n_train, (weight_hparams or {}).get(method), cfg, seed, notes)
                except Exception as err:   # an optimiser can fail in many ways
                    notes.append(f"the pooled {method!r} row failed - none is written "
                                 f"({type(err).__name__}: {err})")
                    two = None
                if two is not None:
                    tagged.setdefault('pooled', {})[method] = two[0]
                    by_method[method] = two[1]
        if pooled_info is None and by_method:
            first = by_method[next(iter(by_method))]
            pooled_info = dict(
                n_source=int(pooled_frame.shape[0]), source_share=first['source_share'],
                members=[names[kept.index(j)] for j in pj_all],
                pooled_oof_r={names[kept.index(j)]: _fit_stats(y_tv, p_oof[j])[0] for j in pj_all},
                source_weight=dict(first['source_weight']), alpha=first['alpha'])
        if pooled_info is not None:
            pooled_info['by_method'] = by_method
            # ver4-96: the optional per-member rows (the members' own pooled predictions)
            if cfg.get('pop_member_rows'):
                member_rows = _pooled_member_variants(names, kept, p_oof, p_test, n_train)
                if member_rows:
                    tagged['pooled_member'] = member_rows
                    pooled_info['member_rows'] = list(member_rows)
                # pool-fix: the gated per-member rows (the like-for-like counterpart of the pooled
                # Super-learner row: own column + pooled column, shrunk towards the own column)
                try:
                    gated_rows, gated_share = _pooled_gated_member_variants(
                        P_k, test_mat, names, kept, p_oof, p_test, y_tv, n_train, cfg, seed, notes)
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as err:
                    notes.append(f"the gated member rows failed - none are written "
                                 f"({type(err).__name__}: {err})")
                    gated_rows, gated_share = {}, {}
                if gated_rows:
                    tagged['pooled_member_gated'] = gated_rows
                    pooled_info['member_gated_share'] = gated_share

    effect: Optional[Dict[str, np.ndarray]] = None
    if deploy == 'refit' and cfg['explain_refit']:
        effect = {usable[j].name: refit_effects[j] for j in kept if j in refit_effects}

    diag = dict(
        deploy=deploy, folds=n_folds, n_rows=n_tv, alpha=float(alpha), seconds=time.time() - t_start,
        oof_r={n: _fit_stats(y_tv, P_k[:, c])[0] for c, n in enumerate(names)},
        cv_mse={float(a): float(v) for a, v in dict(wdiag.get('cv_mse', {})).items()},
        shrink_rule=str(cfg['shrink_rule']), oof_engine=str(cfg['oof_engine']),
        n_oof_units=sum(1 for t in tags if t[0] == 'oof'),
        n_refit_units=sum(1 for t in tags if t[0] == 'refit'),
        step9_fallback=list(deployed_step9),
        # ver4-79 (R1): internal flag - True only when new phenotypes reach the member fits.
        information_gain=(deploy != 'step9'),
        # ver4-80 (R1): the weight method(s) the rows were built with, in row order
        weight_methods=list(variants))
    if pooled_info is not None:
        pooled_info['n_units'] = len(ptags)
        diag['pooled'] = pooled_info
    return SLResult(pred=pred, members=names, weights=dict(primary.weights),
                    alpha=float(alpha), deploy=deploy, effect=effect, n_fits=len(units),
                    notes=notes, diag=diag, variants=variants, tagged_variants=tagged)
