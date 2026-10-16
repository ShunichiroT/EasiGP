"""
models/joint_tuning.py
=======================
ver4-26, R2/R3 (design blueprint `EasiGP_ver4-26_Design_Blueprint.md`):
joint hyperparameter + ensemble-weight optimisation.

Owns the ONE new piece of math this update adds: a single Bayesian search
over a space built ONLY from every group model's own tunable hyperparameter
dimensions (`models.hyperparameter_tuning.build_param_specs`, reused
unmodified) - NOT the ensemble weights (see Round 6 change note below).
Every candidate's SCORE requires refitting every group model at that
candidate's own decoded hyperparameters via the caller-supplied
`run_model_fn` (the SAME `ModelTrialRunner` single-model tuning already
uses - genomic_prediction.py's own single-authority "how to call a model"
implementation, invariant I3), then combining their validation predictions
with that candidate's own analytically-optimal ensemble weights.

This module deliberately composes existing single-authority pieces rather
than re-deriving any of them (design record R2): `build_param_specs`,
`_encode_bounds`-equivalent unit-cube encoding, `_decode`, `_disable_
explainability`, `_cheapen_mcmc`, `_cheapen_fidelity` (all imported from
`models.hyperparameter_tuning`), `analytic_simplex_weights` /
`ensemble_mse_objective` (from `models.ensemble_regularization`), and
`models.parallel_search.batch_bayesian_maximise` (the SAME constant-liar
batch-suggestion machinery `models/Bayesian_optimisation.py`'s own weight
search already uses).

BUGFIX/REDESIGN (joint-bayesian-optimisation-underperformance-fix, Round 6,
2026-09-29, on request to make the joint search's own hyperparameter tuning
independently earn any improvement, rather than rely on the opt-in Round 5
`untuned_baseline_floor` safety net to merely avoid losing): rounds 1-5 all
diagnosed and patched *symptoms* of one underlying design choice - the
search's own unit-cube vector used to encode BOTH each group model's
hyperparameters AND `K-1` stick-breaking ensemble-weight coordinates, so one
single black-box, noisy, sample-inefficient Bayesian search had to locate a
good weight vector by trial and error, in exactly the same way it had to
locate good hyperparameters. But the ensemble-weight sub-problem is NOT a
black-box search problem: given any fixed set of validation predictions, the
best non-negative, sum-to-1 combination of them is a convex, exactly-solvable
problem, and this codebase already has the solver for it -
`ensemble_regularization.analytic_simplex_weights()` (the SAME function the
standalone `Analytic least-squares` W_OPT method already calls). Round 6
removes the `K-1` weight dimensions from the searched unit-cube entirely and
instead calls `analytic_simplex_weights()` INSIDE `_score_candidate()`, once
per hyperparameter candidate (and once per inner resample fold), immediately
after that candidate's group models are refit and their validation
predictions are known - so the outer Bayesian search now only ever explores
hyperparameters, and every candidate it considers (including the untuned
`seed_point` baseline used by the default-vs-winner margin/1-SE-rule checks
below) is scored at its OWN best-achievable ensemble weighting, never at a
weighting the same noisy search still had to also discover. This closes the
Round 5 gap unconditionally, at every candidate the search ever evaluates,
rather than only as a final, opt-in, post-hoc floor check: if the search's
winning hyperparameters still do not beat the untuned baseline once BOTH are
judged at their own optimal weighting, that is now a genuine statement about
whether tuning these hyperparameters jointly helps on this data, not an
artefact of the weight sub-search doing a worse job for one candidate than
another. It also removes weight-dimension sampling noise from the search
entirely (the analytic solve is deterministic given a set of predictions),
so the `1-SE rule` above only ever has to account for hyperparameter-side
noise, not weight-side noise too. `select_weights_with_floor()`,
`diversity_regularizer()`, naive shrinkage and the opt-in
`untuned_baseline_floor` safeguard are all left wired in exactly as before -
harmless, and still a genuine defense-in-depth layer for the ONE remaining
gap this fix does not close (the search-phase `_score_candidate()` calls run
at cheapened, `search_fidelity=True` fitting; the full-fidelity confirmatory
fit could still occasionally disagree) - but none of them should need to
override this search's own result nearly as often as before, since the
fundamental confound they were compensating for no longer exists upstream.

ROUND 10 (joint-bayesian-optimisation-underperformance-fix, 2026-09-29, on
the report that the joint approach still did not beat the naive ensemble even
though, in that run's Metric.csv, the 'ensemble' row and the 'Joint Bayesian
optimisation' row are built from the SAME three jointly-tuned prediction
vectors - so the gap between them is decided entirely by the ensemble
weights). Findings (see the project doc for the simulation evidence; only the
fresh-resample gate and the exact-baseline fix are enabled by default):

 1. Weights were fitted and graded on the same rows. The analytic
    (MSE-optimal) weights are, by definition, the minimiser of validation MSE,
    so `select_weights_with_floor()`'s "never worse than equal weighting on
    validation MSE" floor could never fire, and the search scored every
    candidate at those in-sample-optimal weights. On the small validation
    split a task typically has (tens of rows) and strongly correlated base
    models, estimating K-1 free weights costs more variance than optimal
    weighting gains over a plain average whenever the models are of similar
    quality (the "forecast combination puzzle") - so equal weighting wins on
    the TEST split while looking worse on validation. Fix ('weight_cv',
    default OFF): candidates are scored with cross-fitted (out-of-fold)
    weights, and the final weights come from cross-validating the shrinkage
    level between equal weighting and the analytic solution
    (`ensemble_regularization.cv_shrunk_simplex_weights`, one-standard-error
    rule -> equal weighting unless the fitted weights demonstrably help on
    held-out rows).
 2. The accept/reject decision for the tuned hyperparameters compared the
    search's maximum score (winner's curse) with a re-score of the defaults on
    the same folds; the 'fresh' re-evaluation of the winner used for the 1-SE
    rule was computed and discarded. Fix ('fresh_gate', default ON): winner and
    defaults are re-scored on fresh inner resamples the search never saw, as a
    paired comparison.
 3. Multi-fidelity screening (opt-in) registered reduced-fidelity scores into
    the search as if they were full-fidelity ones, so the reported "winner"
    could be a candidate that had only ever been screened - on as few as ~8
    inner-validation rows when `valid_repeats>1`. Fix: discarded candidates
    are registered no better than the worst promoted one, the winner is chosen
    only among genuinely full-fidelity evaluations, and screening is skipped
    for tasks too small for it to be trustworthy or worthwhile.

WHAT THIS MODULE DOES NOT DO (R2's own contract, and see Change Summary
`EasiGP_ver4-26_Change_Summary.md` limitations section for the full,
disclosed scope): it never writes to `record`/`effect`/`weight`/any
result-file accumulator - it only DECIDES hyperparameters and weights and
hands them back to the caller. What happens to that decision next (a full-
fidelity refit of every group model whose results populate that model's
own ordinary per-task Metric.csv/Marker_effect.csv/etc. row, and one
combined 'Joint Bayesian optimisation' ensemble row) is
`genomic_prediction.py`'s own job (design record R5) - kept there
deliberately, since it is the same "Step 9/10/11 dispatch" concern every
other model and every other weighted-ensemble method already lives in, not
a new concern this module should own a second copy of.

SCOPE NOTE (superseded - ver4-28 R1, kept for history): this module's candidate
evaluation used to be SERIAL only. Since ver4-28 every fit unit this search
issues can fan out across loky workers via `_run_units()` when
`HYPERPARAMETERS_JOINT_OPT['parallel_trials']` is on and `n_jobs>1`
(`worker_init`/`worker_init_args` restore the run's own compute resources
inside each worker). ver4-32 R2 extends that to the block-coordinate search:
the q candidates of one block round are fitted as ONE parallel batch
(`_fit_columns_batch()`).

ver4-32 (design blueprint `EasiGP_ver4-32_joint-speed-accuracy_Design_Blueprint.md`),
six opt-in keys, every one defaulting to the ver4-31 behaviour (I11):
 * 'freeze_fidelity_fields' (R1) - run-length fields (`floor_at_baseline`) leave
   the search space; they are cheapened during the search and run at the
   configured value in the confirmatory fit.
 * 'search_patience' (R2) - stop a batch search after this many evaluations
   without improvement (`_patience_stop()`); 0 = off.
 * 'gate_fit_cache' (R3) - the fresh-resample gate reuses fits whose settings
   are identical for the defaults and the candidate.
 * 'score_weight_rule' (R4) - 'shrunk' scores candidates at the SAME shrunk
   weights that are reported and drops the diversity penalty.
 * 'warm_start' / 'block_warm_n_iter' (R5) - block strategy only: tune each
   block on its single-model score first, re-score the cached columns under
   the ensemble score at zero extra fits, and let the gate choose between the
   final and the warm candidate (`_paired_fresh_gate()`).
Unconditional ver4-32 change: the block-search seed uses `zlib.crc32(model)`
instead of the process-dependent `hash(model)` (a reproducibility bug fix).

ver4-36 (design blueprint `EasiGP_ver4-36_Design_Blueprint.md`): fast and genuinely
better joint optimisation. Fifteen more keys inside `HYPERPARAMETERS_JOINT_OPT`, each at
its ver4-35-preserving default (I11); with all of them absent a run is identical to
ver4-35. Speed levers: 'gate_mode'='bbc' (a zero-fit bootstrap gate over the fitted
library, R1.b), 'search_test_stub' (search fits of rrBLUP/GBLUP/BayesB/RKHS get a 1-row
test frame, R1.c), 'search_engine'='exact_kernel' (closed-form rrBLUP/RKHS search fits,
R1.d), `search_strategy='async_library'` (one pipeline per model, no global barrier,
R1.e) and the cost/ETA accounting (R1.a). Accuracy levers: 'noise_aware_surrogate'
(Matern + WhiteKernel surrogate, posterior-mean incumbent, R2.a), 'selection_rule'='bbc'
(bootstrap-validated pick from the library, R2.b), `weight_estimator='ensemble_selection'`
(bagged Caruana selection, R2.c) and 'library_dump' (write-only diagnostic of the fitted
library, R2.d). The confirmatory fit is ALWAYS a real BGLR/runner fit on the full `test`
frame; the naive/W_OPT result is never substituted, floored or selected; final models are
never fitted on train+valid.

ver4-34 (design blueprint `EasiGP_ver4-34_Design_Blueprint.md`): faster and more
accurate joint optimisation. Every key below lives inside
`HYPERPARAMETERS_JOINT_OPT` and defaults to the ver4-32 behaviour (I11); with all
of them absent a run is identical to ver4-32. Evidence (blueprint SS0.1): the
joint row of a six-model run was almost identical to its RKHS column, i.e. the
analytic simplex weights collapsed onto one model because they were estimated
from ~50 validation rows over near-collinear columns and graded on those same
rows.
 * R1.a - the per-task `fits:` log line also reports seconds and fits per model
   (`_fit_unit_worker()` times each fit; `_run_units()` accumulates).
 * R1.b - `estimate_joint_fit_count()` (pure) is the single authority for "how many
   fits will this configuration run"; the GUI shows it and the engine logs it.
 * R1.c - `search_strategy='parallel_library'` (`_parallel_library_search()`): every
   model proposes candidates in the same round, all of them are fitted in ONE
   parallel batch, every fitted column is kept in a per-model library, and the
   combination is chosen from the library with zero extra fits
   (`_coordinate_select()`). Sequential fit batches drop from ~30 to
   `library_rounds + 1`.
 * R1.d - `freeze_low_weight` stops TUNING a model whose bagged weight cannot
   matter (it stays in the ensemble); `time_budget_s` is a wall-clock bound.
 * R2.a - `weight_estimator='bagged_cluster'`: correlation-clustered, bagged,
   exact-NNLS simplex weights (`ensemble_regularization.robust_simplex_weights`),
   used consistently to SCORE candidates (cross-fitted) and to REPORT the weights.
   There is deliberately no equal-weight floor and no selection of the naive
   ensemble (requirement R2).
 * R2.b - `objective_data='cv_train_valid'`: a cross-validated super-learner
   objective over train+valid (OOF columns), which also gives ~3x more rows for
   the final weights. Final models are still fitted on TRAIN only.
 * R2.c - `search_space='priority'`: only fields marked
   `HPARAM_SPECS[..]['tunable']['priority']` are searched (less winner's curse).

ver4-40, R2 (blueprint S7-S9) - `search_strategy='weight_first'`
(`_weight_first_search`): the ensemble weights are chosen FIRST. The seed columns
are fitted once, `ensemble_regularization.weight_candidates` turns them into a few
distinct fixed weight vectors, and a successive-halving race over those candidates
tunes each model only where its candidate weight is material. Every candidate is
then re-scored over the whole fitted library with ZERO extra fits (the score is a
pure function of the prediction columns). The final weights are the chosen fixed
vector, not a re-estimate, and the pick (bootstrap OOB mean, or plain max) replaces
the fresh-refit gate/BBC. All rung requests go through one `_fit_requests_batch`
call per rung, so results do not depend on `n_jobs`. Fewer than two distinct
candidates falls back to `async_library`. Every other strategy is byte-identical.

ver4-41 - weight-first pick fix (`_wf_bbc`, `_wf_simplest`): on near-equal-quality models and ~50
scoring rows the ver4-40 pick often chose a one-hot / noisy-tuned finalist and the joint row fell
BELOW the equal-weight blend (the reported symptom: `Weight.csv` one-hot on a single model, joint
metrics identical to that model). Three changes, all zero-fit: (1) the in-bag re-selection of a
tuned finalist starts from the UNTUNED columns (it started from the full-row selection, which had
seen the out-of-bag rows); (2) `wf_pick='bbc'` now applies the one-standard-error parsimony rule
(the simplest finalist within `wf_k_se` SD of the best paired out-of-bag score), `'bbc_max'` is
the plain argmax; (3) with too few rows for the bootstrap the simplest finalist is taken, not the
best in-sample score. Every other strategy is byte-identical.

ver4-42, R1 (blueprint EasiGP_ver4-42_Design_Blueprint.md) - `search_strategy='weight_first_cv'`
(`_weight_first_cv_search`): a cross-fitted ("super learner") weight-first search. Why: the ver4-40/41
search estimated the weights from the ~30-80 validation rows only, chose them JOINTLY with noisy
tuning arms, and its parsimony margin (a per-draw SD, not a standard error) then almost always
returned the untuned equal-weight blend - the joint row tied the naive ensemble by construction.
Stages: R0 the untuned search-fidelity columns; S0 out-of-fold untuned columns over train+valid
(`wf_cv_weight_fidelity` 'full' refits them at the final fidelity, +K*V fits in the 'oof' stage);
S1 `ensemble_regularization.shrinkage_path_weights` fixes the weights ONCE
(`w = (1 - alpha) * uniform + alpha * NNLS-type direction`, alpha by repeated CV over the
out-of-fold rows); S2 (`wf_cv_tune`) `_async_library_search` tunes under those fixed weights
through `_FixedWeightLibraryScorer`; S3 `_wf_bbc` decides tuned vs untuned columns at the SAME
weights with the plain out-of-bag mean (`wf_cv_accept`; no parsimony margin on top of the
out-of-bag correction). The result uses the same `stats['wf']` schema as `weight_first`, so the
fixed-vector branches of this function apply it unchanged. The strategy forces
`objective_data='cv_train_valid'` (when not eligible, S1 runs on the validation rows and says so);
final models are still fitted on TRAIN only. Every other strategy is byte-identical.

ver4-48, R2 (blueprint EasiGP_ver4-48_Design_Blueprint.md) - `search_strategy='neural_stack'`
(`_neural_stack_search`, `models/hyper_stack.py`): the hyperparameter choice of every model and the
ensemble weights are the trainable weights of one small network. Why: the earlier strategies pick
the argmax of noisy CV scores (winner's curse) and estimate weights on columns other than the
deployed ones, and none of the 17 models is differentiable in its hyperparameters (BGLR/MCMC,
trees, libsvm SVR, integer-k KNN), so the literal "gradients through every fit" design is
infeasible (blueprint R1). The feasible form: a library of L configurations per model (entry 0 =
the configured parameters, the rest Latin-hypercube points) is fitted out of fold over train+valid
in ONE batch; `W = softmax(a)` weights the models and `q_m = softmax(b_m)` gates the configurations
of model m; the loss is `-ensemble_score + lambda * Omega` (L-BFGS, float64, CPU), and lambda comes
from repeated CV along a path whose lambda = infinity end is EXACTLY the untuned naive ensemble.
The result uses the `stats['wf']` fixed-vector schema, so the weight-first deployment branches of
this function apply it unchanged (no gate re-estimates it); `ns_deploy='mixture'` deploys the heavy
library columns through the ver4-38 member machinery. The strategy forces
`objective_data='cv_train_valid'`; final models are still fitted on TRAIN only. With
`safety_nets=False` the lambda = infinity candidate is removed and the rule is 'min'. Every other
strategy is byte-identical.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .hyperparameter_tuning import (
    build_param_specs, _decode, _disable_explainability, _cheapen_mcmc, _cheapen_fidelity,
    _split_train_rows, _encode_from_params, _enforce_mcmc_burnin_invariant, _evaluate_batch,
    FAILURE_FLOOR, FAILURE_MARGIN, SCORE_CAP, schema_key_of, MCMC_LENGTH_FIELDS,
    # ver4-40 R1: the fidelity/priority predicates and the filter itself now live in
    # hyperparameter_tuning (single authority, I3); re-imported under their old names.
    _is_fidelity_field, _is_priority_field, filter_param_specs,
)
from .ensemble_regularization import (
    ensemble_mse_objective, diversity_regularizer, select_weights_with_floor,
    analytic_simplex_weights, cross_fitted_combination, cv_shrunk_simplex_weights,
    apply_naive_shrinkage, robust_simplex_weights, robust_weights_only, _fold_indices,
    # ver4-36: bootstrap-validated gate/selection (R1.b, R2.b) and the estimator registry (R2.c)
    bootstrap_oob_indices, fast_oob_score, ROBUST_WEIGHT_ESTIMATORS, is_robust_estimator,
    # ver4-40 R2: the fixed candidate weight vectors of the weight-first search
    weight_candidates,
    # ver4-42 R1: the cross-validated shrinkage path of `search_strategy='weight_first_cv'`
    shrinkage_path_weights,
)
# Round 6: `stick_breaking_to_simplex`/`simplex_to_stick_breaking` are no
# longer imported here - the joint search no longer allocates unit-cube
# dimensions for ensemble weights at all (see this module's own top-level
# Round 6 note), so nothing in this file decodes/encodes a stick-breaking
# weight vector any more. Both remain in use elsewhere (e.g.
# `models/Bayesian_optimisation.py`'s own standalone weight search) -
# unaffected.
from .parallel_search import (
    batch_bayesian_maximise, suggest_batch, posterior_means, robust_incumbent,
    _install_noise_aware_surrogate, patience_stop,   # ver4-40 R1: patience_stop moved here
)
# ver4-36 R1.e: one reusable loky executor for the asynchronous library search.
from .hyperparameter_tuning import _loky_executor

try:
    from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
except ImportError:  # pragma: no cover - already a project dependency
    BayesianOptimization = None
    SequentialDomainReductionTransformer = None

try:
    from bayes_opt.acquisition import ExpectedImprovement
except ImportError:  # pragma: no cover - already a project dependency
    ExpectedImprovement = None


# R3 lever 4 (dimensionality guard) - provisional, flagged for calibration
# alongside resource_profiles.py's other provisional constants (architecture
# doc SS19.3) - informational only, never blocks the run.
JOINT_SEARCH_DIM_WARN_THRESHOLD = 40

# Fixed batch width for the joint search's own constant-liar suggestion
# rounds - matches HP_TUNE_BAYES_BATCH_MAX's own default (R3 lever 1),
# not a separately configurable key (mirrors models/Bayesian_optimisation.
# py's own W_OPT_BAYES_RESTARTS' fixed q=4 batch width for the same reason:
# each evaluation here is expensive - up to K model refits - so the batch
# width is a fixed, conservative constant rather than a user-facing knob).
_JOINT_BATCH_Q = 4

# BUGFIX (weight-collapse report): the joint search's own per-candidate
# score (`_score_candidate` below) originally had NO diversity protection
# at all - every candidate's raw validation-split score (however noisy on
# what is often a small validation split) was accepted at face value, so
# the search reliably converges to whichever single group model's
# validation performance happens to look best that task, not a genuine
# diversity-weighted blend. This is the SAME structural failure mode
# ensemble_regularization.py's own module docstring diagnoses for
# Nelder_Mead.py/Bayesian_optimisation.py/Linear_transformation.py's
# weight searches - just reached here without any of that module's
# mitigations ever being wired in for the joint path. Fixed below with
# the SAME two mitigations already proven for those three methods: a
# strictly-convex ridge-to-uniform penalty added to every candidate's
# score DURING the search (steers the surrogate away from noise-driven
# vertices while still letting genuinely one-sided evidence win), and a
# post-search validation-MSE floor (`select_weights_with_floor()`) that
# grades the search's own winning weights against equal weighting - and,
# optionally, the closed-form analytic least-squares solution - on the
# REALISED validation MSE of the confirmatory, full-fidelity fit, so the
# final result is never worse than equal weighting by construction, plus
# naive shrinkage as a residual test-set-variance safety margin.
#
# Unlike those three pre-existing methods - where every one of these
# mitigations ships OFF by default (diversity_penalty=0.0,
# W_OPT_ANALYTIC_SEED=False, W_OPT_VALIDATION_FLOOR=False) purely to keep
# an already-shipped default behaviour byte-identical for existing users -
# this joint-search module is new (ver4-26) and its only shipped default
# is the collapse this fix replaces, so there is no back-compat reason to
# leave the same protection off here. Every knob below stays fully
# user-configurable (and can be dialled back to 0/1.0 to reproduce the
# old, unprotected behaviour exactly) via the optional 'diversity_penalty'
# / 'diversity_method' / 'shrinkage_alpha' / 'analytic_seed' keys inside
# HYPERPARAMETERS_JOINT_OPT (main_app.py's JOINT_HYPERPARAM_OPT_SPECS).
_JOINT_DIVERSITY_PENALTY_DEFAULT = 0.15
_JOINT_DIVERSITY_METHOD_DEFAULT = 'ridge_to_uniform'
_JOINT_SHRINKAGE_ALPHA_DEFAULT = 0.85
_JOINT_ANALYTIC_SEED_DEFAULT = True

# BUGFIX (joint-optimisation-underperformance-fix, Round 5): unlike the
# four knobs just above, this one ships OFF by default, on request (Shun,
# 2026-09-29) - it is opt-in, not opt-out. Two reasons this one is
# different: (1) cost - unlike diversity_penalty/shrinkage_alpha/
# analytic_seed (each a cheap, purely arithmetic post-processing step on
# predictions the search already computed), this safeguard pays for ONE
# MORE full-fidelity confirmatory fit of every group model at its own
# untuned baseline - a real, potentially significant wall-clock cost on
# top of the search, only paid when the search's own winner survived
# every other safeguard - that a user may not want to pay on every run
# without opting in first. (2) unlike the weight-collapse protection
# above (which fixes a failure mode with no legitimate use case - a
# collapsed, near-one-hot weight vector is never desirable), THIS
# safeguard trades away part of joint tuning's own theoretical upside:
# it can fall back to untuned hyperparameters even on a task where the
# search's own tuned hyperparameters were a genuine (if noisy-looking)
# improvement, which is a real, if bounded (see select_weights_with_
# floor()'s own strict-improvement, tol=0.0 comparison - never a near
# tie), trade-off a user may prefer to make deliberately rather than
# have applied unconditionally. See `tune_joint_hyperparameters_and_
# weights()`'s own docstring for the full mechanism this key controls.
_JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT = False

# FEATURE (Round 7, requested follow-on to Round 6): multi-fidelity
# screening for the BATCH search path only - see `_screen_subsample_train()`
# and `tune_joint_hyperparameters_and_weights()`'s own Round 7 docstring
# paragraph for the full mechanism. OFF by default for the same reason
# `_JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT` is: it is a genuine trade-off
# (a screening-fidelity score, fit on fewer training rows, is registered
# into the search's own surrogate for a candidate that is never truly
# evaluated at full fidelity - noisier and systematically a little
# pessimistic relative to a full-training-rows fit, in exchange for
# spending less wall-clock per round), not an unconditional correctness
# fix - a user opts into this deliberately via the 'multi_fidelity' key
# inside HYPERPARAMETERS_JOINT_OPT (main_app.py's own
# JOINT_HYPERPARAM_OPT_SPECS checkbox).
_JOINT_MULTI_FIDELITY_DEFAULT = False
# Fraction of the task's own TRAINING rows (never `valid`/`test` - see
# `_screen_subsample_train()`'s own docstring) used for the cheap
# screening-rung fit. Conservative (not tiny): the screening score still
# has to rank candidates within a batch sensibly relative to each other,
# which gets less reliable the smaller this fraction is.
_JOINT_MULTI_FIDELITY_TRAIN_FRACTION = 0.35
# Never subsample a task's training split below this many rows for
# screening - below this, a fit's own validation score becomes dominated
# by which particular rows were drawn rather than by the candidate's own
# hyperparameters, defeating the point of using it to RANK candidates.
#
# ROUND 10: raised from 30 to 100. With `valid_repeats>1` the screening fit is
# itself split 75/25 into an inner train/validation pair, so a 30-35 row
# screening subsample left ~8-9 rows to score a candidate on (a Pearson r and
# K in-sample ensemble weights on 8-9 points is almost pure noise) and, worse,
# those scores were registered into the search as if they were full-fidelity
# scores. See `_screen_subsample_train()` and `objective_batch()` below.
_JOINT_MULTI_FIDELITY_MIN_ROWS = 100
# ROUND 10: screening is only worth doing when it actually saves something -
# if the screening subsample would already be more than this fraction of the
# task's training rows (small populations), it is skipped for that task.
_JOINT_MULTI_FIDELITY_MAX_TRAIN_FRACTION = 0.7
# Fraction of each round's batch promoted from the cheap screening rung to
# a real, full-training-rows evaluation - the rest keep their screening
# score. 0.5 (top half of the batch) is the standard successive-halving
# "eta=2" split; never promotes fewer than 1 candidate per round.
_JOINT_MULTI_FIDELITY_KEEP_FRACTION = 0.5

# ROUND 10 (joint-bayesian-optimisation-underperformance-fix): two correctness
# fixes that, unlike the opt-in trade-offs above, default ON. See the Round 10
# note in this module's docstring and `tune_joint_hyperparameters_and_weights()`.
#
#  * 'weight_cv'  - OPT-IN (default OFF): ensemble weights are chosen and scored
#    OUT-OF-SAMPLE (cross-validation over the validation rows). Simulation
#    showed the existing in-sample analytic weights (0.85 shrinkage) were as
#    good as or better than every CV variant, so this is NOT enabled by default.
#  * 'fresh_gate' - the "did tuning genuinely beat the untuned defaults?"
#    decision is taken on FRESH inner resamples that played no part in the
#    search, instead of on the search's own (winner's-curse-inflated) score.
_JOINT_WEIGHT_CV_DEFAULT = False
_JOINT_FRESH_GATE_DEFAULT = True
_JOINT_FRESH_GATE_REPEATS = 5
# Added to `inner_split_seed_base` so the gate's resamples never coincide with
# the ones the search itself scored every candidate on.
_JOINT_FRESH_GATE_SEED_OFFSET = 7919

# ver4-28 (design blueprint EasiGP_ver4-28_Design_Blueprint.md) - speed and
# accuracy levers. Every one of these ships OFF/at its legacy-preserving
# default (I11): with every key below left unset, `tune_joint_hyperparameters_
# and_weights()`'s own runner-call sequence, search result and return shape
# are unaffected - see the blueprint's own SS8 Rollback table.
#
# R1 - fit units fan out across loky workers via the existing
# `hyperparameter_tuning._evaluate_batch()` machinery (see `_run_units()`).
_JOINT_PARALLEL_TRIALS_DEFAULT = False
# R2 - the dispatch loop in genomic_prediction.py reuses this search's own
# confirmatory fit instead of refitting every group model a second time.
_JOINT_REUSE_CONFIRMATORY_FIT_DEFAULT = False
# R3 - 'monolithic' (existing single Bayesian search over the whole
# concatenated space) or 'block_coordinate' (one model's own hyperparameter
# block at a time - see `_block_coordinate_search()`).
_JOINT_SEARCH_STRATEGY_DEFAULT = 'monolithic'
_JOINT_BLOCK_CYCLES_DEFAULT = 2
_JOINT_BLOCK_INIT_POINTS_DEFAULT = 3
_JOINT_BLOCK_N_ITER_DEFAULT = 0  # 0 => auto max(8, 2*d_m) - see _block_coordinate_search().
_JOINT_BLOCK_REFINE_ITER_DEFAULT = 4
_JOINT_BLOCK_TOL_DEFAULT = 1e-3
# R4 - final ensemble weights fitted on pooled out-of-fold-train + validation
# rows instead of validation rows alone (see `_oof_pooled_matrix()`).
_JOINT_WEIGHT_SOURCE_DEFAULT = 'valid'
_JOINT_OOF_FOLDS_DEFAULT = 5
# Added to `seed` so the OOF fold assignment never coincides with any other
# seeded resample this search draws.
_JOINT_OOF_SEED_OFFSET = 104729

# ver4-32 (design blueprint EasiGP_ver4-32_joint-speed-accuracy_Design_Blueprint.md)
# - six opt-in keys, every one at its ver4-31-preserving default (I11).
# R1 - drop `floor_at_baseline` run-length fields from the search space.
_JOINT_FREEZE_FIDELITY_FIELDS_DEFAULT = False
# R5 - 'none' (start from the configured defaults) or 'individual' (tune each
# block on its single-model score first; block_coordinate strategy only).
_JOINT_WARM_START_DEFAULT = 'none'
_JOINT_BLOCK_WARM_N_ITER_DEFAULT = 0  # 0 => auto max(10, 3*d_m) - see _block_coordinate_search().
# R4 - 'analytic_penalised' (ver4-31: raw analytic weights minus the diversity
# penalty) or 'shrunk' (the shrunk weights that are reported, no penalty).
_JOINT_SCORE_WEIGHT_RULE_DEFAULT = 'analytic_penalised'
# R2 - stop a batch search after this many evaluations without improvement; 0 = off.
_JOINT_SEARCH_PATIENCE_DEFAULT = 0
# Minimum improvement (score units) that resets the patience counter.
_JOINT_PATIENCE_TOL = 1e-3
# R3 - the fresh gate reuses identical fits (defaults vs candidate).
_JOINT_GATE_FIT_CACHE_DEFAULT = False
# Added to the per-model block seed for the individual warm-start pass, so its
# random initial points differ from the ensemble cycles' own.
_JOINT_WARM_SEED_OFFSET = 31337
# Most recent (x, column) pairs per block re-scored under the ensemble score.
_JOINT_COL_HIST_CAP = 50

# ver4-34 (design blueprint EasiGP_ver4-34_Design_Blueprint.md) - every key ships at
# its ver4-32-preserving default (I11); see the module docstring for what each does.
# R2.a - 'legacy' (analytic weights, 0.85 shrinkage) or 'bagged_cluster'.
_JOINT_WEIGHT_ESTIMATOR_DEFAULT = 'legacy'
_JOINT_WEIGHT_BAGS_DEFAULT = 100       # bootstrap bags for the REPORTED weights
_JOINT_CLUSTER_CORR_DEFAULT = 0.97     # columns correlated >= this share one weight
_JOINT_SCORE_BAGS_DEFAULT = 10         # bags per cross-fit fold when SCORING (engine-only)
# R2.b - 'valid' (score on the validation rows) or 'cv_train_valid' (OOF over train+valid).
_JOINT_OBJECTIVE_DATA_DEFAULT = 'valid'
_JOINT_CV_FOLDS_DEFAULT = 5
# Added to `inner_split_seed_base` so the CV partition never coincides with any other
# seeded resample of this search (the fresh gate adds its own offset on top, which is
# what gives the gate a FRESH partition).
_JOINT_CV_SEED_OFFSET = 15485863
# R2.c - 'all' (every tunable field) or 'priority' (only fields marked priority).
_JOINT_SEARCH_SPACE_DEFAULT = 'all'
# R1.d - 0.0 = never freeze a model; 0 = no wall-clock budget.
_JOINT_FREEZE_LOW_WEIGHT_DEFAULT = 0.0
_JOINT_TIME_BUDGET_DEFAULT = 0
# R1.c - rounds of the parallel-library search and candidates per model per round.
_JOINT_LIBRARY_ROUNDS_DEFAULT = 4
_JOINT_LIBRARY_Q_DEFAULT = 3
# Most columns of one model's library re-scored per round (zero fits, but each
# re-score solves the weights, so the cost is bounded).
_JOINT_LIBRARY_RESCORE_CAP = 60
# At most this many greedy passes in `_coordinate_select()`.
_JOINT_LIBRARY_MAX_PASSES = 3
# Estimated BGLR-backed model names are read from `MCMC_LENGTH_FIELDS` (the single
# source for "this model is an MCMC model"), never from a second list here.
_JOINT_SEARCH_STRATEGIES = ('monolithic', 'block_coordinate', 'parallel_library',
                            'async_library', 'weight_first', 'weight_first_cv',
                            'neural_stack')
# ver4-42: the strategies whose result is ONE fixed weight vector (`stats['wf']`); every site that
# used to test `== 'weight_first'` for "the final weights are a fixed vector" tests this tuple.
# ver4-48: 'neural_stack' also ends in ONE fixed weight vector (`stats['wf']`, the per-model sums of the
# trained network), so it joins the tuple; the site audit is in the Change Summary.
_WF_STRATEGIES = ('weight_first', 'weight_first_cv', 'neural_stack')

# ver4-36 (design blueprint EasiGP_ver4-36_Design_Blueprint.md) - every key ships at its
# ver4-35-preserving default (I11); see the module docstring for what each does.
# R1.b - 'fresh_refit' (the paired fresh-resample gate, 60-90 fits) or 'bbc' (zero fits).
_JOINT_GATE_MODE_DEFAULT = 'fresh_refit'
_JOINT_BBC_BOOTSTRAPS_DEFAULT = 200
# Accept the tuned procedure iff this lower quantile of the bootstrap gap beats the margin
# (0.16 is the one-standard-error analogue of a normal lower quantile).
_JOINT_BBC_GATE_QUANTILE_DEFAULT = 0.16
# Fewer valid bootstraps than this, or fewer rows than `_JOINT_BBC_MIN_ROWS`, and the BBC
# gate is not trusted (the fresh-resample gate runs instead, with a NOTE).
_JOINT_BBC_MIN_VALID = 20
_JOINT_BBC_MIN_ROWS = 20
# Added to `seed + inner_split_seed_base` so the bootstrap draws never coincide with any
# other seeded resample of this search (I9).
_JOINT_BBC_SEED_OFFSET = 7727
# R1.c - search fits pass a 1-row `test` frame to models whose search never reads it.
_JOINT_SEARCH_TEST_STUB_DEFAULT = False
_JOINT_STUB_PARITY_MIN_DEFAULT = 0.995
# R1.d - 'bglr' (every search fit is a real fit) or 'exact_kernel' (closed-form rrBLUP / RKHS).
_JOINT_SEARCH_ENGINE_DEFAULT = 'bglr'
_JOINT_ENGINE_PARITY_MIN_DEFAULT = 0.98
# R1.e - budget of the asynchronous library search: columns of a model with d tunable
# dimensions = clip(1 + library_evals_per_dim * d, 4, library_max_evals); `refine_rounds`
# further barrier rounds after Phase 1.
_JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT = 6
_JOINT_LIBRARY_MAX_EVALS_DEFAULT = 30
_JOINT_LIBRARY_MIN_EVALS = 4
_JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT = 1
# R2.a - noise-aware surrogate (Matern + WhiteKernel) with a posterior-mean incumbent.
_JOINT_NOISE_AWARE_DEFAULT = False
# R2.b - 'greedy' (`_coordinate_select`) or 'bbc' (bootstrap-validated pick from the library).
_JOINT_SELECTION_RULE_DEFAULT = 'greedy'
# R2.c - ensemble-selection estimator: 0 steps = 4 * K; fraction of the models per bag.
_JOINT_ES_STEPS_DEFAULT = 0
_JOINT_ES_MODEL_FRACTION_DEFAULT = 0.5
# R2.d - write the fitted library (scoring AND test columns) to <result_dir>/Joint_library.
_JOINT_LIBRARY_DUMP_DEFAULT = False

# ver4-38 (design blueprint EasiGP_ver4-38_Design_Blueprint.md) - every key ships at its
# ver4-37-preserving default (I11).
# R1.b - score a proposal on `racing_folds` CV folds first and promote only competitive ones
# to the full fold set (async_library + CV objective only).
_JOINT_CV_RACING_DEFAULT = False
_JOINT_RACING_FOLDS_DEFAULT = 2
# A raced proposal is promoted iff its partial score >= the incumbent's partial score - margin.
_JOINT_RACING_MARGIN_DEFAULT = 0.01
# Subtracted from the (offset-calibrated) race score of a REJECTED proposal before it is shown
# to the surrogate, so a rejected point never looks better than it was measured to be.
_JOINT_RACING_PENALTY_DEFAULT = 0.0
# Racing needs at least this many scoring rows in the raced folds, else it promotes everything.
_JOINT_RACING_MIN_ROWS = 10
_JOINT_RACING_PROMOTION_NOMINAL = 1.0 / 3.0   # share of raced proposals completed (estimator only)
# R1.c - 'off' or 'value' (async_library: stop a model's pipeline when more columns would not pay).
_JOINT_LIBRARY_STOP_RULE_DEFAULT = 'off'
# After this many consecutive rounds whose best proposal gains less than `block_tol`.
_JOINT_STOP_RULE_PATIENCE = 2
# R1.d - 'full' (final OOF refit at full MCMC length) or 'search' (reuse the library columns).
_JOINT_OOF_FIDELITY_DEFAULT = 'full'
# R2.b1 - a (near-)constant column scores as a failure and a model whose untuned column is
# constant leaves the group for this task.
_JOINT_DEAD_COLUMN_GUARD_DEFAULT = False
# A column is dead when std(col) <= this * std(target) (or when it is not finite).
_JOINT_DEAD_COLUMN_REL_TOL = 1e-8
# R2.b2 - keep several jointly found configurations per model as ensemble members.
_JOINT_LIBRARY_ENSEMBLE_DEFAULT = False
_JOINT_LIBRARY_MEMBERS_MAX_DEFAULT = 12
_JOINT_LIBRARY_MEMBERS_PER_MODEL_DEFAULT = 3
# Members with an ES weight below this are dropped before renormalising.
_JOINT_LIBRARY_MEMBER_MIN_WEIGHT = 0.02
# Added to `seed + inner_split_seed_base` for the member-selection ES bags (I9).
_JOINT_LIBRARY_MEMBER_SEED_OFFSET = 6151
# Version of the Joint_library .npz layout (2 adds `oof__k`; tools read 1 and 2).
_JOINT_DUMP_VERSION = 2

# ver4-40 (design blueprint EasiGP_ver4-40_Design_Blueprint.md, R2) - read ONLY when
# `search_strategy='weight_first'`; every key ships at its ver4-39-preserving default (I11).
# Candidate weight vectors, in priority order (`ensemble_regularization.weight_candidates` grammar).
_JOINT_WF_RULES_DEFAULT = ['uniform', 'analytic', 'shrunk:0.25', 'shrunk:0.5', 'shrunk:0.75', 'es']
_JOINT_WF_MAX_CANDIDATES_DEFAULT = 6
_JOINT_WF_DEDUP_TOL_DEFAULT = 0.05      # L1 distance below which two vectors count as one
_JOINT_WF_RUNGS_DEFAULT = 3             # successive-halving rungs over the weight candidates
_JOINT_WF_ETA_DEFAULT = 2               # keep the best ceil(|A| / eta) candidates per rung
# A model gets proposals under a weight vector only when its weight there is at least this.
_JOINT_WF_MIN_TUNE_WEIGHT_DEFAULT = 0.05
# D1: the untuned (naive-ensemble-like) arms are finalists too (False = the strict ver4-34 policy).
_JOINT_WF_INCLUDE_UNTUNED_DEFAULT = True
# 'bbc' (ver4-41: honest out-of-bag pick + one-standard-error parsimony), 'bbc_max' (honest
# out-of-bag pick, plain highest mean - the ver4-40 rule minus its leak) or 'max' (best full-row
# score; in-sample, for comparison only).
_JOINT_WF_PICK_DEFAULT = 'bbc'
_JOINT_WF_PICK_CHOICES = ('bbc', 'bbc_max', 'max')
# ver4-41: a finalist is 'as good as the best' when the mean paired out-of-bag gap to the best is
# at most this many standard deviations of that gap over the bootstrap draws.
_JOINT_WF_K_SE_DEFAULT = 1.0
_JOINT_WF_CANDIDATE_ROWS_DEFAULT = 'scoring'   # 'scoring' or 'oof_train' (+K*folds fits)

# ver4-42 (design blueprint EasiGP_ver4-42_Design_Blueprint.md, R1) - read ONLY when
# `search_strategy='weight_first_cv'`; every key ships at a value that changes nothing for any
# other strategy (I11).
# S0 fidelity of the untuned out-of-fold columns the weights are estimated on: 'full' (+K*V fits;
# the columns of the final models' fidelity) or 'search' (library entry 0, 0 extra fits).
_JOINT_WFCV_WEIGHT_FIDELITY_DEFAULT = 'full'
# Direction of the shrinkage path: 'analytic' (NNLS-type weights) or 'es' (ensemble selection).
_JOINT_WFCV_DIRECTION_DEFAULT = 'analytic'
# Candidate shrinkage levels alpha of w(alpha) = (1 - alpha) * uniform + alpha * direction.
_JOINT_WFCV_ALPHAS_DEFAULT = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# 'min' = lowest cross-validated loss, 'one_se' = smallest alpha within one standard error of it.
_JOINT_WFCV_ALPHA_RULE_DEFAULT = 'min'
_JOINT_WFCV_REPEATS_DEFAULT = 4          # CV repeats of the alpha choice
_JOINT_WFCV_TUNE_DEFAULT = True          # False = weights only ("WFCV-W"): no S2 tuning
# S3 decision between the tuned and the untuned columns at the SAME weights: 'bbc_mean' (plain
# out-of-bag mean), 'bbc_one_se' (+ parsimony), 'always' (the tuned columns).
_JOINT_WFCV_ACCEPT_DEFAULT = 'bbc_mean'
_JOINT_WFCV_ACCEPT_CHOICES = ('bbc_mean', 'bbc_one_se', 'always')

# ver4-44 (design blueprint EasiGP_ver4-44_Design_Blueprint.md, R1 / R3).
# R1 `safety_nets` (default True = every protection of ver4-43 stays on, byte-identical):
# False = "pure joint evaluation" - the joint row is exactly what the optimiser produced. It
# switches off every post-hoc comparison that REPLACES the optimiser's answer by untuned
# columns or equal-ish weights (class A: the fresh and BBC gates, the legacy margin check, the
# untuned-baseline floor, the equal-weight floor, the weight-first untuned arm and silent
# strategy fallbacks) and every pull towards equal weights (class B: the alpha path, one-SE,
# ridge-to-uniform, shrinkage_alpha, weight_cv, the 'uniform' candidate). Crash and degenerate-
# data fallbacks (class C) stay and are RECORDED per task (`details['fallback_events']`).
_JOINT_SAFETY_NETS_DEFAULT = True
# R3 `wf_cv_final` ('weight_first_cv' only): 'refit' = the direction is fitted once on all rows
# (ver4-42); 'bagged' = the mean of the in-fold direction vectors the CV loop already computes
# (zero extra fits, no pull towards equal weights).
_JOINT_WFCV_FINAL_DEFAULT = 'refit'
_JOINT_WFCV_FINAL_CHOICES = ('refit', 'bagged')
# ver4-47, R5 `wf_cv_reweight` ('weight_first_cv' only): stage S4, after S2/S3. The weights w* of
# S1 were fitted on the UNTUNED out-of-fold columns but are deployed on the TUNED ones - once the
# columns change (a tuned kernel/ridge model is more accurate and often more correlated with its
# neighbours) the untuned-column weights are stale and the 'joint' row can end up worse than the
# plain tuned-column blend or a single tuned model. 'off' (default) = the legacy deployed weights
# w*. 'refit' = the SAME direction estimator (wf_cv_direction / wf_cv_final) is re-run on a FRESH
# out-of-fold partition of the DEPLOYED (tuned) columns and its vector is deployed instead of w*.
# 'avg' = the mean of w* and that vector (pools two estimates made on different columns and
# partitions; lowers estimator variance). Neither mode compares against a baseline or pulls
# towards equal weights (so it is not a safety net and applies in the safety_nets=False regime);
# it costs K*V extra full-fidelity fits ('oof' stage) and only runs when the tuned columns were
# kept (S3 'tuned' arm), the CV objective is active and S2 changed at least one column.
_JOINT_WFCV_REWEIGHT_DEFAULT = 'off'
_JOINT_WFCV_REWEIGHT_CHOICES = ('off', 'refit', 'avg')
_JOINT_REWEIGHT_SEED_OFFSET = 2750159
# ver4-47, R6 `wf_cv_members` ('weight_first_cv' + `library_ensemble` only): the deployed joint row
# averages the top `library_members_per_model` jointly found configurations of every model (equal
# split of that model's FIXED weight w[m]; blueprint R4) instead of ONE winning configuration.
# A single winner picked among tens of noisy CV scores carries the winner's curse; the mean of the
# top-q is a within-model variance reduction. It is not a safety net (no baseline comparison, no
# pull towards equal weights): the per-model weight sums are exactly the deployed vector. Costs at
# most K*(q-1) extra train-only confirm fits. Default False = the legacy single-configuration row.
_JOINT_WFCV_MEMBERS_DEFAULT = False

# ver4-48 (design blueprint EasiGP_ver4-48_Design_Blueprint.md, R2) - read ONLY when
# `search_strategy='neural_stack'` (`_neural_stack_search`, `models/hyper_stack.py`); every key ships
# at a value that changes nothing for any other strategy (I11).
# Library size per model: entry 0 = the configured parameters + (L - 1) Latin-hypercube points. A
# model that has no tunable dimension always has L = 1; MCMC (BGLR) models use the smaller figure
# because every column costs `cv_folds` full-length chains.
_JOINT_NS_CONFIGS_DEFAULT = 6
_JOINT_NS_CONFIGS_MCMC_DEFAULT = 3
# Regularisation path of the network (lambda = infinity, the untuned naive ensemble, is added
# unless `safety_nets=False`). A list, so `[]` means "prior only" and is NOT replaced by a default.
_JOINT_NS_LAMBDAS_DEFAULT = [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]
_JOINT_NS_LAMBDA_RULE_DEFAULT = 'one_se'     # forced to 'min' under safety_nets=False
_JOINT_NS_LAMBDA_RULE_CHOICES = ('min', 'one_se')
_JOINT_NS_REPEATS_DEFAULT = 4                # CV repeats of the lambda choice
# ver4-49: the fitted network's weights are averaged with the untuned naive ensemble,
# W' = (1 - s) W + s / K (and the same for the column weights), before deployment, and the one-SE
# rule uses the Nadeau-Bengio-corrected standard error. 0 = ver4-48 weights. Read with
# `budget_kwargs.get('ns_weight_shrink', ...)`; a GUI field since ver4-50 (R4).
_JOINT_NS_WEIGHT_SHRINK_DEFAULT = 0.5
# ver4-50 (R2): what the network is trained on. 'ensemble_score' = pearson - mse/var (ver4-48/49,
# the single score of every other strategy); 'pearson' = ranking only, invariant to the scale of
# the CV rows' predictions (blueprint R1 hypothesis H1). `_NS_SCORE_FNS` (below
# `_pearson_score_of`) maps each name to its numpy score; `hyper_stack._TORCH_SCORES` to its twin.
_JOINT_NS_OBJECTIVE_DEFAULT = 'ensemble_score'
_JOINT_NS_OBJECTIVE_CHOICES = ('ensemble_score', 'pearson')
# ver4-50 (R3): how a hyperparameter switch (a model's gate leaving entry 0) is admitted. 'gate' =
# ver4-49 (the argmax of the fitted gate, no test); 'paired_se' = the swap needs a paired CV gain
# above its Nadeau-Bengio standard error (`hyper_stack.swap_test`), otherwise the model keeps
# entry 0.
_JOINT_NS_SWAP_RULE_DEFAULT = 'gate'
# ver4-51 (R1): 'paired_refit' = the swap is judged against the network REFITTED with that model's
# gate clamped to entry 0 (a fair counterfactual) on a fresh partition, not against the fitted
# network with one gate overwritten (`hyper_stack.refit_swap_test`).
_JOINT_NS_SWAP_RULE_CHOICES = ('gate', 'paired_se', 'paired_refit')
# ver4-51: seeds are added to `ns_seed` so no partition of the neural stack coincides with another
# (I9): the fresh partition of the `paired_refit` swap test and the outer partition of the gate.
_JOINT_NS_SWAP_SEED_OFFSET = 7919
_JOINT_NS_OUTER_SEED_OFFSET = 104729
# ver4-51 (R2): an outer cross-validation of the WHOLE procedure (lambda path -> swap rule ->
# shrink -> deployment, `hyper_stack.nested_gate`) decides whether the fitted network is deployed
# at all: 'nested' deploys it only when its out-of-sample gain over the equal-weight ensemble beats
# `_JOINT_NS_OUTER_Z` Nadeau-Bengio standard errors, otherwise the untuned equal-weight ensemble
# (decision 'ns_outer_prior'). 'off' = ver4-50. The gate is not run under safety_nets=False.
_JOINT_NS_OUTER_GATE_DEFAULT = 'off'
_JOINT_NS_OUTER_GATE_CHOICES = ('off', 'nested')
_JOINT_NS_OUTER_FOLDS_DEFAULT = 5
_JOINT_NS_OUTER_REPEATS_DEFAULT = 2
_JOINT_NS_OUTER_Z = 1.0
_JOINT_NS_DEPLOY_DEFAULT = 'mixture'         # members = the heavy library columns
_JOINT_NS_DEPLOY_CHOICES = ('mixture', 'argmax')
_JOINT_NS_FIDELITY_DEFAULT = 'full'          # fidelity of the library's out-of-fold fits
_JOINT_NS_FIDELITY_CHOICES = ('full', 'search')
# Added to `seed + inner_split_seed_base` (+ crc32(model) for the library design) so no seeded
# draw of the neural stack coincides with another one of this search (I9).
_JOINT_NS_SEED_OFFSET = 9973
# Engine-only constants of the L-BFGS fit (not config keys).
_JOINT_NS_MAX_ITER = 200
_JOINT_NS_TOL = 1e-10

# ver4-36: strategy-stats entries that are data, not a summary (printed never).
_STATS_NOT_PRINTED = ('warm_x', 'lib', 'lib_cur', 'cur', 'wf', 'ns')

# ver4-28, R6: every key `_v428_nondefault()` below checks, paired with its
# own legacy-preserving default - kept as ONE table so a new ver4-28 key
# can never be added to `HYPERPARAMETERS_JOINT_OPT` without also being
# checked here (an omission would silently under-report "non-default" and
# skip the extra `hp_record` columns genomic_prediction.py adds when this
# returns True).
_JOINT_V428_KEY_DEFAULTS = {
    'parallel_trials': _JOINT_PARALLEL_TRIALS_DEFAULT,
    'reuse_confirmatory_fit': _JOINT_REUSE_CONFIRMATORY_FIT_DEFAULT,
    'search_strategy': _JOINT_SEARCH_STRATEGY_DEFAULT,
    'block_cycles': _JOINT_BLOCK_CYCLES_DEFAULT,
    'block_init_points': _JOINT_BLOCK_INIT_POINTS_DEFAULT,
    'block_n_iter': _JOINT_BLOCK_N_ITER_DEFAULT,
    'block_refine_iter': _JOINT_BLOCK_REFINE_ITER_DEFAULT,
    'block_tol': _JOINT_BLOCK_TOL_DEFAULT,
    'weight_source': _JOINT_WEIGHT_SOURCE_DEFAULT,
    'oof_folds': _JOINT_OOF_FOLDS_DEFAULT,
    # ver4-32
    'freeze_fidelity_fields': _JOINT_FREEZE_FIDELITY_FIELDS_DEFAULT,
    'warm_start': _JOINT_WARM_START_DEFAULT,
    'block_warm_n_iter': _JOINT_BLOCK_WARM_N_ITER_DEFAULT,
    'score_weight_rule': _JOINT_SCORE_WEIGHT_RULE_DEFAULT,
    'search_patience': _JOINT_SEARCH_PATIENCE_DEFAULT,
    'gate_fit_cache': _JOINT_GATE_FIT_CACHE_DEFAULT,
    # ver4-34
    'weight_estimator': _JOINT_WEIGHT_ESTIMATOR_DEFAULT,
    'weight_bags': _JOINT_WEIGHT_BAGS_DEFAULT,
    'cluster_corr': _JOINT_CLUSTER_CORR_DEFAULT,
    'objective_data': _JOINT_OBJECTIVE_DATA_DEFAULT,
    'cv_folds': _JOINT_CV_FOLDS_DEFAULT,
    'search_space': _JOINT_SEARCH_SPACE_DEFAULT,
    'freeze_low_weight': _JOINT_FREEZE_LOW_WEIGHT_DEFAULT,
    'time_budget_s': _JOINT_TIME_BUDGET_DEFAULT,
    'library_rounds': _JOINT_LIBRARY_ROUNDS_DEFAULT,
    'library_q': _JOINT_LIBRARY_Q_DEFAULT,
    'score_bags': _JOINT_SCORE_BAGS_DEFAULT,
    # ver4-36
    'gate_mode': _JOINT_GATE_MODE_DEFAULT,
    'bbc_bootstraps': _JOINT_BBC_BOOTSTRAPS_DEFAULT,
    'bbc_gate_quantile': _JOINT_BBC_GATE_QUANTILE_DEFAULT,
    'search_test_stub': _JOINT_SEARCH_TEST_STUB_DEFAULT,
    'stub_parity_min': _JOINT_STUB_PARITY_MIN_DEFAULT,
    'search_engine': _JOINT_SEARCH_ENGINE_DEFAULT,
    'engine_parity_min': _JOINT_ENGINE_PARITY_MIN_DEFAULT,
    'library_evals_per_dim': _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT,
    'library_max_evals': _JOINT_LIBRARY_MAX_EVALS_DEFAULT,
    'library_refine_rounds': _JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT,
    'noise_aware_surrogate': _JOINT_NOISE_AWARE_DEFAULT,
    'selection_rule': _JOINT_SELECTION_RULE_DEFAULT,
    'es_steps': _JOINT_ES_STEPS_DEFAULT,
    'es_model_fraction': _JOINT_ES_MODEL_FRACTION_DEFAULT,
    'library_dump': _JOINT_LIBRARY_DUMP_DEFAULT,
    # ver4-38
    'cv_racing': _JOINT_CV_RACING_DEFAULT,
    'racing_folds': _JOINT_RACING_FOLDS_DEFAULT,
    'racing_margin': _JOINT_RACING_MARGIN_DEFAULT,
    'racing_penalty': _JOINT_RACING_PENALTY_DEFAULT,
    'library_stop_rule': _JOINT_LIBRARY_STOP_RULE_DEFAULT,
    'oof_fidelity': _JOINT_OOF_FIDELITY_DEFAULT,
    'dead_column_guard': _JOINT_DEAD_COLUMN_GUARD_DEFAULT,
    'library_ensemble': _JOINT_LIBRARY_ENSEMBLE_DEFAULT,
    'library_members_max': _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT,
    'library_members_per_model': _JOINT_LIBRARY_MEMBERS_PER_MODEL_DEFAULT,
    # ver4-40 (R2: read only under search_strategy='weight_first')
    'wf_rules': _JOINT_WF_RULES_DEFAULT,
    'wf_max_candidates': _JOINT_WF_MAX_CANDIDATES_DEFAULT,
    'wf_dedup_tol': _JOINT_WF_DEDUP_TOL_DEFAULT,
    'wf_rungs': _JOINT_WF_RUNGS_DEFAULT,
    'wf_eta': _JOINT_WF_ETA_DEFAULT,
    'wf_min_tune_weight': _JOINT_WF_MIN_TUNE_WEIGHT_DEFAULT,
    'wf_include_untuned': _JOINT_WF_INCLUDE_UNTUNED_DEFAULT,
    'wf_pick': _JOINT_WF_PICK_DEFAULT,
    'wf_candidate_rows': _JOINT_WF_CANDIDATE_ROWS_DEFAULT,
    # ver4-42 (R1: read only under search_strategy='weight_first_cv')
    'wf_cv_weight_fidelity': _JOINT_WFCV_WEIGHT_FIDELITY_DEFAULT,
    'wf_cv_direction': _JOINT_WFCV_DIRECTION_DEFAULT,
    'wf_cv_alphas': _JOINT_WFCV_ALPHAS_DEFAULT,
    'wf_cv_alpha_rule': _JOINT_WFCV_ALPHA_RULE_DEFAULT,
    'wf_cv_repeats': _JOINT_WFCV_REPEATS_DEFAULT,
    'wf_cv_tune': _JOINT_WFCV_TUNE_DEFAULT,
    'wf_cv_accept': _JOINT_WFCV_ACCEPT_DEFAULT,
    # ver4-44 (R1 / R3)
    'safety_nets': _JOINT_SAFETY_NETS_DEFAULT,
    'wf_cv_final': _JOINT_WFCV_FINAL_DEFAULT,
    # ver4-47 (R5)
    'wf_cv_reweight': _JOINT_WFCV_REWEIGHT_DEFAULT,
    'wf_cv_members': _JOINT_WFCV_MEMBERS_DEFAULT,
    # ver4-48 (R2: read only under search_strategy='neural_stack')
    'ns_configs': _JOINT_NS_CONFIGS_DEFAULT,
    'ns_configs_mcmc': _JOINT_NS_CONFIGS_MCMC_DEFAULT,
    'ns_lambdas': _JOINT_NS_LAMBDAS_DEFAULT,
    'ns_lambda_rule': _JOINT_NS_LAMBDA_RULE_DEFAULT,
    'ns_repeats': _JOINT_NS_REPEATS_DEFAULT,
    'ns_deploy': _JOINT_NS_DEPLOY_DEFAULT,
    'ns_fidelity': _JOINT_NS_FIDELITY_DEFAULT,
    # ver4-50 (R2, R3: read only under search_strategy='neural_stack')
    'ns_objective': _JOINT_NS_OBJECTIVE_DEFAULT,
    'ns_swap_rule': _JOINT_NS_SWAP_RULE_DEFAULT,
    # ver4-51 (R2: read only under search_strategy='neural_stack')
    'ns_outer_gate': _JOINT_NS_OUTER_GATE_DEFAULT,
    'ns_outer_folds': _JOINT_NS_OUTER_FOLDS_DEFAULT,
    'ns_outer_repeats': _JOINT_NS_OUTER_REPEATS_DEFAULT,
}


def _v428_nondefault(budget_kwargs: dict) -> bool:
    """ver4-28, R6 (extended by ver4-32, ver4-34, ver4-36 and ver4-38): True when at least one of
    the fifty-two ver4-28/ver4-32/ver4-34/ver4-36/ver4-38 keys inside
    `HYPERPARAMETERS_JOINT_OPT` (`budget_kwargs`) is set away from its own
    legacy-preserving default (`_JOINT_V428_KEY_DEFAULTS` above) - used by
    `genomic_prediction.py` to decide whether this task's own `hp_record`
    row gains the three additional `joint_strategy`/`joint_gate_decision`/
    `joint_n_fits` columns (I8/P3 - every OTHER task's row stays column-
    for-column identical to a pre-ver4-28 run, avoiding a `run_step2_
    assemble.py` column-unification surprise on a run that never touches
    any of these keys).
    """
    return any(
        budget_kwargs.get(key, default) != default
        for key, default in _JOINT_V428_KEY_DEFAULTS.items()
    )


# ver4-44, R1 - the safety nets that `safety_nets=False` switches off, as the log names them.
# The N-numbers are the blueprint's inventory (R1 design record); N1 (silent strategy fallbacks)
# is enforced where the fallback happens, the others by `_pure_budget_overrides()`.
_SAFETY_NET_LABELS = (
    "N1 silent strategy fallbacks",
    "N2 fresh-resample gate",
    "N3 BBC gate",
    "N4 untuned-baseline floor",
    "N5 equal-weight floor",
    "N6 weight_cv",
    "N7 diversity penalty / shrinkage_alpha / 'shrunk' rule",
    "N8 shrinkage-path alpha (alpha=1, the fitted direction is deployed)",
    "N9 weight-first-cv S3 untuned-vs-tuned check",
    "N10 'uniform' candidate / untuned finalists / parsimony pick",
)


class _PureModeError(RuntimeError):
    """Raised (ver4-44, R1, N1) when a search that `safety_nets=False` must not replace by the
    monolithic search cannot complete. A distinct type lets the strategy dispatch re-raise it
    unwrapped."""


def _resolve_safety_nets(budget_kwargs: dict) -> Tuple[bool, List[str]]:
    """Read the `safety_nets` key once per task (I11: absent -> True, the ver4-43 behaviour).

    Returns `(on, disabled)`: `on` is the resolved boolean; `disabled` is the list of safety nets
    that are switched off (empty when `on`), worded for the log line the caller prints. Never
    raises (a non-boolean value is read with `bool()`)."""
    on = bool(budget_kwargs.get('safety_nets', _JOINT_SAFETY_NETS_DEFAULT))
    return on, ([] if on else list(_SAFETY_NET_LABELS))


def _pure_budget_overrides(budget_kwargs: dict) -> dict:
    """The `budget_kwargs` of a `safety_nets=False` task: a COPY of the caller's dict in which
    every key that feeds a safety net (blueprint R1, N2-N10) is set to its off value.

    Why a copy of the dict and not scattered `if` tests: every later read in
    `tune_joint_hyperparameters_and_weights()` is `budget_kwargs.get(key, default)`, so the nets
    cannot be re-enabled by a read that was overlooked, and the resolved values are the ones the
    settings log lines print. The caller's own dict is never modified.

    Off values: fresh_gate False (N2), gate_mode 'fresh_refit' (N3: the BBC GATE; the BBC
    `selection_rule`, which only chooses among library entries, is left alone), untuned_baseline_
    floor False (N4), weight_cv False (N6), diversity_penalty 0 / shrinkage_alpha 1 / score_
    weight_rule 'analytic_penalised' (N7), wf_cv_accept 'always' (N9), and for `weight_first` (N10)
    wf_rules without 'uniform', wf_include_untuned False, wf_pick 'bbc' -> 'bbc_max' (no
    parsimony). N5 (equal-weight floor) and N8 (alpha path) act at their own call sites, which
    read `safety_nets` directly."""
    out = dict(budget_kwargs)
    out.update({
        'fresh_gate': False, 'gate_mode': 'fresh_refit', 'untuned_baseline_floor': False,
        'weight_cv': False, 'diversity_penalty': 0.0, 'shrinkage_alpha': 1.0,
        'score_weight_rule': 'analytic_penalised', 'wf_cv_accept': 'always',
        'wf_include_untuned': False,
    })
    rules = out.get('wf_rules', _JOINT_WF_RULES_DEFAULT)
    if isinstance(rules, str):
        rules = [t.strip() for t in rules.split(',') if t.strip()]
    kept = [r for r in (rules or []) if str(r).strip().lower() != 'uniform']
    out['wf_rules'] = kept or [r for r in _JOINT_WF_RULES_DEFAULT if r != 'uniform']
    if (out.get('wf_pick', _JOINT_WF_PICK_DEFAULT) or _JOINT_WF_PICK_DEFAULT) == 'bbc':
        out['wf_pick'] = 'bbc_max'
    return out


def validate_search_strategy(budget_kwargs: Optional[dict]) -> str:
    """Return the `search_strategy` of a joint-optimisation config, or raise.

    ver4-44, R0: an unknown value used to print a NOTE and run 'monolithic', which is how a
    run that was meant to test 'weight_first_cv' tested something else without anyone noticing
    (blueprint S5). Absent / empty -> the default ('monolithic'). Anything else must be one of
    `_JOINT_SEARCH_STRATEGIES`.

    Called twice on purpose: by `GP()`'s pre-flight, so the run stops before task 1 (the
    per-task call site in genomic_prediction.py catches exceptions raised inside the search and
    carries on), and again by `tune_joint_hyperparameters_and_weights()` for direct callers.

    Raises ValueError naming the value, the valid values and the engine version."""
    value = ((budget_kwargs or {}).get('search_strategy', _JOINT_SEARCH_STRATEGY_DEFAULT)
             or _JOINT_SEARCH_STRATEGY_DEFAULT)
    if value in _JOINT_SEARCH_STRATEGIES:
        return value
    try:
        from pipeline_utils import ENGINE_VERSION
    except ImportError:
        ENGINE_VERSION = 'unknown'
    raise ValueError(
        f"[joint_tuning] search_strategy={value!r} is not recognised by this engine "
        f"(EasiGP {ENGINE_VERSION}); valid values: {list(_JOINT_SEARCH_STRATEGIES)!r}. The run is "
        f"stopped instead of silently running 'monolithic'. If the GUI offers a strategy this "
        f"engine does not know, the engine in memory is older than the GUI - restart EasiGP.")


def build_joint_param_specs(group_models: Sequence[str], hparam_specs: dict,
                             base_params_by_model: Optional[dict] = None, *,
                             freeze_fidelity: bool = False,
                             priority_only: bool = False
                             ) -> Tuple[Dict[str, list], int]:
    """Concatenate every group model's own tunable `ParamSpec` list (R2) -
    `build_param_specs()` is reused UNMODIFIED per model, in `group_models`
    order.

    Returns `(specs_by_model, n_weight_dims)`. `specs_by_model[m]` is `[]`
    for a model with nothing tunable in `HPARAM_SPECS` - a valid,
    zero-width contribution to the concatenated space, not an error: the
    joint search still optimises every OTHER group model's hyperparameters.

    `freeze_fidelity` (ver4-32, R1; default False = ver4-31 behaviour):
    when True, every spec whose field is a run-length knob
    (`_is_fidelity_field()`) is dropped from `specs_by_model[m]`. Those
    fields then run cheapened during the search and at the configured value
    in the confirmatory fit. A model whose fields are ALL frozen gets `[]`
    - a valid zero-width block (`_block_coordinate_search()` skips it).

    `priority_only` (ver4-34, R2.c; default False = ver4-32 behaviour): when True,
    a model keeps only its fields marked `priority` (`_is_priority_field()`).
    A model with NO marked field keeps all of its tunable fields (safe default:
    an unannotated model is never silently removed from the search). The marks are
    looked up on the model's FULL tunable list, i.e. before `freeze_fidelity`
    removes anything, so the two filters compose in either order: a field is
    searched only if neither filter removes it. Fields left out run at their
    configured value in the confirmatory fit (and, for run-length fields, cheapened
    during the search by `_decode_model_params()`, exactly like frozen ones).

    `n_weight_dims` is always `0` as of Round 6 (joint-bayesian-optimisation-
    underperformance-fix) - kept as a return value only for call-site
    compatibility (`total_dims = sum(...) + n_weight_dims` still reads
    correctly at every existing call site as a no-op `+0`). The ensemble
    weights are no longer unit-cube search dimensions at all: every
    candidate's weights are solved analytically, per candidate, inside
    `_score_candidate()` - see this module's own top-level Round 6 note.
    """
    base_params_by_model = base_params_by_model or {}
    specs_by_model = {
        m: build_param_specs(m, hparam_specs, base_params=base_params_by_model.get(m))
        for m in group_models
    }
    # ver4-40 R1: both filters now live in hyperparameter_tuning.filter_param_specs (I3);
    # the semantics (marks read on the full tunable list; unannotated model keeps all) are
    # unchanged.
    specs_by_model = {
        m: filter_param_specs(m, specs, hparam_specs, freeze_fidelity=freeze_fidelity,
                              priority_only=priority_only)
        for m, specs in specs_by_model.items()
    }
    n_weight_dims = 0
    return specs_by_model, n_weight_dims


class _LazyTrainFrame:
    """ver4-97: a training frame that is concatenated only when its fit unit runs.

    Why: the Super learner's pooled units (ver4-94) train on ``(train+valid rows) + every source
    line`` - thousands of rows x every marker. Building one ``pd.concat`` copy per unit up front
    held ``K * (folds + 1)`` full copies in memory at once, before the first fit started. This
    object keeps only references (the small own frame, the fold's kept positions, the shared
    source frame) and builds the one frame a unit needs inside ``_fit_unit_worker``, so a serial
    run holds one copy at a time and a parallel run one per live worker.

    ``shape`` is exposed so that code which only needs the row count of a unit's training frame
    (``iterative_ensemble._columns_from_result``) works without building it. Plain data, so it
    pickles into a loky worker like the frames it stands for."""

    def __init__(self, own, keep, extra, shift=0.0):
        self.own = own                    # DataFrame (train+valid rows, member's columns)
        self.keep = keep                  # None (all rows) or positional row indices of `own`
        self.extra = extra                # DataFrame with the same columns, appended below
        # ver4-100 (pop_center_per_fold): added to the LAST column (the phenotype) of `extra` when
        # the frame is built, so the borrowed lines are centred on this fold's own training mean
        # without one shifted copy of the source per fold being held. 0.0 = unchanged.
        self.shift = float(shift)

    @property
    def shape(self):
        n_own = self.own.shape[0] if self.keep is None else int(len(self.keep))
        return (n_own + self.extra.shape[0], self.own.shape[1])

    def materialise(self):
        base = self.own if self.keep is None else self.own.iloc[self.keep]
        out = pd.concat([base, self.extra], ignore_index=True)
        if self.shift != 0.0:
            last = out.columns[-1]
            n_own = base.shape[0]
            out.loc[out.index[n_own:], last] = out.loc[out.index[n_own:], last].astype(float) + self.shift
        return out


@dataclass
class _FitUnit:
    """ver4-28, R1: one model-refit request - `(model, params, split_tag,
    train_rows, valid_rows)` per the blueprint, plus `fidelity` (used only
    to build a `_FitCache` key, R3) and `keep_full` (R2: return the whole
    runner dict instead of trimming it down to `predicted_valid`).

    ver4-36 (all three default to the ver4-35 behaviour): `engine` ('runner' = the
    caller-supplied `run_model_fn`, or 'exact_kernel' = the closed-form rrBLUP / RKHS
    stand-in of `kernel_effects.exact_kernel_column`, R1.d), `test_stub` (pass a 1-row
    `test` frame to the runner, R1.c) and `keep_test` (also return `predicted_test`, used by
    the library dump, R2.d). `keep_full` units (confirmatory fits that hand their whole
    result to Step 9) ignore `engine` and `test_stub`: they are always a real fit on the full
    `test` frame.

    ver4-52: `cv_weights` (default False = every earlier behaviour) marks a unit whose frames are
    cut from the train+valid CV frame (`_cv_fold_frames()`); the worker then fits it through
    `runner.with_cv_weights()`, so under BLUE weighting the fold's training rows (validation rows
    included) get their own fitting weights instead of the task's train-split vector.

    ver4-68 (R2): `test_frame` (default None = every earlier behaviour: the unit is scored on the
    ONE `test` frame `_run_units()` was given) lets a unit carry its own test frame. The iterative
    ensemble's two-phase mode needs it because its members are refitted on DIFFERENT marker
    masks, so one shared `test` cannot serve a whole wave of units in a single parallel call.
    When set it replaces `test` before anything else (the test stub and the exact engine then
    act on it). Appended as the LAST field so positional construction elsewhere is unchanged."""
    model: str
    params: list
    split_tag: str
    train: object
    valid: object
    fidelity: str = 'search'
    keep_full: bool = False
    engine: str = 'runner'
    test_stub: bool = False
    keep_test: bool = False
    cv_weights: bool = False
    test_frame: object = None


# Messages about the exact engine are printed once per process and (model, reason): a
# fallback can repeat for every unit and would flood the log.
_ENGINE_NOTED: set = set()


def _fit_unit_worker(runner, RESULT_NAME, test, unit: '_FitUnit'):
    """ver4-28, R1: module-level (I10 - safe to pickle into a loky worker)
    fit-unit worker. Refits `unit.model` on `unit.train`/`unit.valid` at
    `unit.params` via the caller-supplied `runner` (the SAME `run_model_fn`/
    `ModelTrialRunner` adapter every other call site in this module already
    uses), then trims the result down to `{'predicted_valid': ...}` unless
    `unit.keep_full` - a worker only ever ships back what the parent needs
    to SCORE a candidate, except when R2's `reuse_confirmatory_fit` needs
    the caller to hand the FULL runner dict straight to genomic_prediction.
    py's own dispatch instead of refitting. Never raises: a fit failure
    (numerical divergence, an R MCMC failure, ...) is logged and returns
    `None`, exactly like today's in-process failure handling.

    ver4-34, R1.a: the seconds the fit took, measured HERE (inside the worker, so a
    loky process reports its own compute time, not queueing), are returned under the
    private key `'_secs'`. `_run_units()` pops it before anything else sees the
    result, so no consumer of the result dicts ever observes the key.

    ver4-36: (R1.d) with `unit.engine == 'exact_kernel'` (never for a `keep_full` unit) the
    column comes from `kernel_effects.exact_kernel_column` - the same rows, the same BGLR
    default priors, no MCMC; any exception there is reported once per process and the unit
    falls back to the runner, so an engine failure costs time, never a result. (R1.c) with
    `unit.test_stub` the runner receives `test.iloc[:1]` (BGLR masks valid+test rows and runs
    over all of them, so dropping unread test rows is the only saving; the guard in
    `_parity_guards()` verified the columns agree). (R2.d) with `unit.keep_test` the result
    also carries `predicted_test` from the SAME fit. The private key `'_engine'` names the
    path that produced the column ('exact_kernel' or 'runner'); `_account_result()` pops it.

    ver4-68 (R2): a unit with its own `test_frame` is scored on that frame instead of the shared
    `test` argument (see `_FitUnit`); with `test_frame=None` nothing changes."""
    test = unit.test_frame if unit.test_frame is not None else test
    t_fit = time.perf_counter()
    result = None
    engine_used = 'runner'
    # ver4-97: a lazily built training frame (the Super learner's pooled units) is made here, once.
    train = unit.train.materialise() if isinstance(unit.train, _LazyTrainFrame) else unit.train
    if unit.engine == 'exact_kernel' and not unit.keep_full:
        try:
            from .kernel_effects import exact_kernel_column
            result = exact_kernel_column(unit.model, train, unit.valid, test, unit.params,
                                         want_test=bool(unit.keep_test))
            engine_used = 'exact_kernel'
        except Exception as exc:
            _tag = (unit.model, type(exc).__name__)
            if _tag not in _ENGINE_NOTED:
                _ENGINE_NOTED.add(_tag)
                print(f"[joint_tuning] NOTE: the exact kernel engine failed for {unit.model!r} "
                      f"({type(exc).__name__}: {exc}) - these units fall back to the real "
                      f"model fit (slower, same result; shown once per process).")
            result = None
    if result is None:
        _stub = unit.test_stub and not unit.keep_full and not unit.keep_test
        _test_arg = test.iloc[:1] if _stub else test
        try:
            _runner = runner
            if unit.cv_weights:
                # ver4-52: a CV fold fit under BLUE weighting - see `_FitUnit`. A runner without
                # the hook (a plain callable, weighting off) is used unchanged.
                _swap = getattr(runner, 'with_cv_weights', None)
                if callable(_swap):
                    _runner = _swap()
            result = _runner(unit.model, train, unit.valid, _test_arg, unit.params,
                             RESULT_NAME)
        except Exception as exc:
            print(f"[joint_tuning] NOTE: fit unit failed refitting {unit.model!r} "
                  f"(split={unit.split_tag!r}, fidelity={unit.fidelity!r}) "
                  f"({type(exc).__name__}: {exc}) - this unit scores as a failure.")
            return None
    secs = time.perf_counter() - t_fit
    if unit.keep_full:
        if isinstance(result, dict):
            result['_secs'] = secs
        return result
    out = {'predicted_valid': result.get('predicted_valid'), '_secs': secs,
           '_engine': engine_used}
    if unit.keep_test:
        out['predicted_test'] = result.get('predicted_test')
    return out


def _apply_fit_flags(units: List['_FitUnit'], flags: Optional[dict], stage: Optional[str]) -> None:
    """ver4-36, R1.c/R1.d/R2.d: switch the per-unit speed flags on, in place.

    Why here: every fit in this module passes through `_run_units()`, and the stage tag
    (`search` / `gate` / `oof` / `confirm`) is the one fact that decides eligibility, so
    one function applying `flags` by stage keeps the rule in a single place (I3) instead of
    in every unit-building call site.

    `flags` keys (all optional): `engine_models` / `stub_models` (sets of model names for
    which the exact engine / the test stub is allowed - decided once per task by
    `_parity_guards()`), `engine_stages` (default `('search', 'gate')`), `stub_stages`
    (default `('search', 'gate', 'oof')`) and `keep_test` (return `predicted_test` in the
    `search` stage; it also switches the stub off, because the dump needs real test columns).
    A `keep_full` unit and every unit of the `confirm` stage are never touched. `None` or
    an empty dict is a no-op. Never raises."""
    if not flags:
        return
    engine_models = flags.get('engine_models') or ()
    stub_models = flags.get('stub_models') or ()
    engine_stages = flags.get('engine_stages', ('search', 'gate'))
    stub_stages = flags.get('stub_stages', ('search', 'gate', 'oof'))
    keep_test = bool(flags.get('keep_test')) and stage == 'search'
    for unit in units:
        if unit.keep_full:
            continue
        if keep_test:
            unit.keep_test = True
        if stage in engine_stages and unit.model in engine_models:
            unit.engine = 'exact_kernel'
        if stage in stub_stages and unit.model in stub_models and not keep_test:
            unit.test_stub = True


def _unit_fidelity_key(unit: '_FitUnit') -> str:
    """ver4-36: the fidelity component of a `_FitCache` key. A plain runner unit keeps its
    ver4-35 key (so legacy runs hit the same entries); a unit that used the exact engine or
    the test stub gets `'<fidelity>:<engine>:<stub>'`, because its column is not the same
    number as a full runner fit's."""
    if unit.engine == 'runner' and not unit.test_stub:
        return unit.fidelity
    return f"{unit.fidelity}:{unit.engine}:{int(unit.test_stub)}"


def _account_result(counter: Optional[dict], unit: '_FitUnit', res) -> Optional[float]:
    """ver4-34 R1.a / ver4-36: strip the worker's private `'_secs'` and `'_engine'` from `res`
    and add this fit to the per-model provenance tables of `counter` (when it has them).

    `fits_by_model[model]` and `secs_by_model[model]` as before; ver4-36 adds the nested
    `engine_fits` dict (`exact_kernel`: fits answered by the exact engine, `stub`: fits run
    with the 1-row test frame, `runner`: the rest) so the log and the GUI can say how many
    fits were MCMC. Returns the worker-measured seconds (None for a failed fit or an unknown
    time). Never raises."""
    secs = None
    engine = None
    if isinstance(res, dict):
        secs = res.pop('_secs', None)
        engine = res.pop('_engine', None)
    if counter is None:
        return secs
    fits_by_model = counter.get('fits_by_model')
    secs_by_model = counter.get('secs_by_model')
    if isinstance(fits_by_model, dict):
        fits_by_model[unit.model] = fits_by_model.get(unit.model, 0) + 1
    if isinstance(secs_by_model, dict) and secs is not None:
        secs_by_model[unit.model] = secs_by_model.get(unit.model, 0.0) + float(secs)
    engine_fits = counter.get('engine_fits')
    if isinstance(engine_fits, dict) and res is not None:
        if engine == 'exact_kernel':
            engine_fits['exact_kernel'] = engine_fits.get('exact_kernel', 0) + 1
        elif unit.test_stub and not unit.keep_full:
            engine_fits['stub'] = engine_fits.get('stub', 0) + 1
        else:
            engine_fits['runner'] = engine_fits.get('runner', 0) + 1
    return secs


class _FitCache:
    """ver4-28, R3: in-process memo of fit-unit results, one instance per
    `_block_coordinate_search()` call, keyed on a model's own DECODED
    parameters (never the raw unit-cube coordinate - two different `x`
    can decode to the same params for an untouched model's block and
    should hit the same cache entry). This is what makes re-scoring an
    untouched model's cached prediction column against a new candidate
    column free (0 fits) - see the blueprint's own R3 flow. `hits` feeds
    the per-task `fits:` provenance log line (R6)."""

    def __init__(self):
        self._store: dict = {}
        self.hits = 0

    @staticmethod
    def key(model, params, fidelity, split_tag):
        return (model, repr(params), fidelity, split_tag)

    def get(self, key):
        val = self._store.get(key)
        if val is not None:
            self.hits += 1
        return val

    def put(self, key, value):
        self._store[key] = value


def _report_progress(progress, done, total, unit, res, secs) -> None:
    """ver4-97: call a `_run_units` progress callback; never lets it raise."""
    if progress is None:
        return
    try:
        progress(done, total, unit, res, secs)
    except Exception:
        pass


def _run_units(units: List['_FitUnit'], *, runner, RESULT_NAME, test, n_jobs: int = 1,
                worker_init=None, worker_init_args=(), cache: Optional['_FitCache'] = None,
                stop_on_failure: bool = True, counter: Optional[dict] = None,
                stage: Optional[str] = None, fit_flags: Optional[dict] = None,
                progress: Optional[Callable] = None
                ) -> List[Optional[dict]]:
    """ver4-28, R1: evaluate a list of fit units, returning one
    `{'predicted_valid': ...}` (or full runner dict, `unit.keep_full`)
    per unit, in `units` order, or `None` for a unit that failed to fit.

    Serial (`n_jobs<=1` or a single unit) preserves the LEGACY early-abort
    order: units are fit one at a time and remaining units are left `None`
    the moment one fails (`stop_on_failure`, default True) - the exact
    control flow `_score_candidate()`'s own per-fold/per-model loops used
    before this refactor. Parallel (`n_jobs>1`, more than one unit) fans
    every not-yet-cached unit out through `hyperparameter_tuning.
    _evaluate_batch()` in one round (that function's own serial fallback on
    a pool failure applies unchanged here too) - there is then nothing to
    gain from aborting early, so every unit is always fit.

    `cache`, when supplied (R3, block-coordinate search only - never
    instantiated for the monolithic strategy), is consulted before a unit
    is fit and populated with every fit that is NOT a cache hit; a cache
    hit costs 0 fits. `counter`/`stage` (R6), when supplied, receive a
    running per-stage fit count (`counter[stage] += 1` per unit actually
    fit) and cache-hit count (`counter['cache_hits'] += 1`) - the shared
    `_fit_counter` dict `tune_joint_hyperparameters_and_weights()` reports
    in its own per-task `fits:` log line. ver4-32 R3: when `counter` also
    holds a nested `'secs'` dict, the wall-clock seconds spent in this call
    are added to `counter['secs'][stage]`.

    ver4-34, R1.a: when `counter` also holds nested `'fits_by_model'` and/or
    `'secs_by_model'` dicts, every fit that is actually run (a cache hit costs 0)
    adds 1 to `fits_by_model[model]` and the worker-measured seconds to
    `secs_by_model[model]` (a failed fit has no timing and adds only the count).
    The worker's private `'_secs'` key is always removed from the result here.

    ver4-36: `fit_flags` (default None = ver4-35 behaviour) is handed to
    `_apply_fit_flags()` together with `stage` before anything is fitted - it switches the
    exact engine / the test stub on for the stages and models that are eligible. The
    `_FitCache` key then carries the engine and stub flag (`_unit_fidelity_key()`).

    ver4-97: `progress` (default None = nothing changes) is called as ``progress(done, total,
    unit, result, seconds)`` each time a unit is finished - after every unit on the serial path
    (`seconds` = that unit's own wall time), once per unit after the whole round on the parallel
    path (`seconds` = the worker-measured fit time). A cache hit is not reported. An exception
    raised by `progress` is swallowed: a log callback must never lose a fit.
    """
    if not units:
        return []
    if fit_flags:
        _apply_fit_flags(units, fit_flags, stage)
    results: List[Optional[dict]] = [None] * len(units)
    serial = n_jobs is None or n_jobs <= 1 or len(units) <= 1
    # ver4-32, R3: wall-clock seconds per stage, accumulated in
    # `counter['secs'][stage]` when the caller supplied that nested dict
    # (`tune_joint_hyperparameters_and_weights()` does) - so the per-task
    # `fits:` log line can report where the time went.
    _t_start = time.perf_counter()

    def _add_secs():
        if counter is not None and stage is not None and isinstance(counter.get('secs'), dict):
            counter['secs'][stage] = (counter['secs'].get(stage, 0.0)
                                      + (time.perf_counter() - _t_start))

    def _cache_key(unit):
        return (cache.key(unit.model, unit.params, _unit_fidelity_key(unit), unit.split_tag)
                if cache is not None else None)

    def _account(unit, res):
        """ver4-34, R1.a: strip the worker's private keys from `res` and add this fit to the
        per-model provenance tables of `counter` (`_account_result()`)."""
        _account_result(counter, unit, res)

    if serial:
        for i, unit in enumerate(units):
            key = _cache_key(unit)
            cached = cache.get(key) if cache is not None else None
            if cached is not None:
                results[i] = cached
                if counter is not None:
                    counter['cache_hits'] = counter.get('cache_hits', 0) + 1
                continue
            _t_unit = time.perf_counter()
            res = _fit_unit_worker(runner, RESULT_NAME, test, unit)
            _account(unit, res)
            results[i] = res
            if counter is not None and stage is not None:
                counter[stage] = counter.get(stage, 0) + 1
            if cache is not None and res is not None:
                cache.put(key, res)
            _report_progress(progress, i + 1, len(units), unit, res,
                             time.perf_counter() - _t_unit)
            if stop_on_failure and res is None:
                break
        _add_secs()
        return results

    to_run, to_run_idx, to_run_keys = [], [], []
    for i, unit in enumerate(units):
        key = _cache_key(unit)
        cached = cache.get(key) if cache is not None else None
        if cached is not None:
            results[i] = cached
            if counter is not None:
                counter['cache_hits'] = counter.get('cache_hits', 0) + 1
            continue
        to_run.append(unit)
        to_run_idx.append(i)
        to_run_keys.append(key)
    if to_run:
        fitted = _evaluate_batch(partial(_fit_unit_worker, runner, RESULT_NAME, test),
                                  to_run, n_jobs, worker_init, worker_init_args)
        for n_done, (idx, unit, key, res) in enumerate(
                zip(to_run_idx, to_run, to_run_keys, fitted), start=1):
            _secs_unit = res.get('_secs') if isinstance(res, dict) else None
            _account(unit, res)
            results[idx] = res
            if counter is not None and stage is not None:
                counter[stage] = counter.get(stage, 0) + 1
            if cache is not None and res is not None:
                cache.put(key, res)
            _report_progress(progress, n_done, len(to_run), unit, res, _secs_unit)
    _add_secs()
    return results


def _is_dead_column(col, y, partition=None) -> bool:
    """ver4-38, R2.b1 / R1.c: True when the prediction column `col` carries no information.

    Why: a model whose column is (near-)constant - e.g. an SVR whose epsilon exceeds the
    spread of the trait, so it has no support vectors - still lowers the MSE of a
    sum-to-one ensemble by shrinking the combination toward the mean. The ensemble
    objective (`pearson - mse/var`) can therefore PREFER the dead column and the model's
    genetic signal silently disappears from the joint row. This is the single definition of
    "dead" (I3): `_ColumnScorer`, the selection filters and the async stop rule all call it.

    A column is dead when it is not a finite 1-D array, or when
    `std(col) <= _JOINT_DEAD_COLUMN_REL_TOL * std(y)`; for a constant `y` (std 0) the test
    degrades to `std(col) <= 0`. Never raises: anything that cannot be converted to floats
    is reported dead.

    `partition` (optional, ver4-38 extension of the blueprint's definition): the held-out
    positions of each cross-validation fold when `col` is an OUT-OF-FOLD vector. A dead model
    predicts its training mean (or intercept) in every fold, and that constant differs from
    fold to fold, so the whole OOF vector is not constant although every fold's block is. With
    a partition the column is also dead when EVERY block of at least 2 rows is constant to the
    same tolerance (and at least one such block exists). `partition=None` is the blueprint's
    definition exactly."""
    try:
        values = np.asarray(col, dtype=float).ravel()
    except (TypeError, ValueError):
        return True
    if values.size == 0 or not np.all(np.isfinite(values)):
        return True
    try:
        y_sd = float(np.std(np.asarray(y, dtype=float)))
    except (TypeError, ValueError):
        y_sd = 0.0
    if not np.isfinite(y_sd):
        y_sd = 0.0
    tol = _JOINT_DEAD_COLUMN_REL_TOL * y_sd
    if float(np.std(values)) <= tol:
        return True
    if partition is None:
        return False
    seen_block = False
    for held in partition:
        held = np.asarray(held, dtype=int)
        if held.size < 2 or held.max(initial=-1) >= values.size:
            continue
        seen_block = True
        if float(np.std(values[held])) > tol:
            return False
    return seen_block


def _failure_score(state):
    """BUGFIX (joint search reverting to a serial search under batch
    Bayesian optimisation, reported via a `[GP] NOTE: batch joint
    hyperparameter+weight search failed ... falling back to a serial joint
    search for this task` log line, root-caused to
    `ValueError: Input y contains infinity or a value too large for
    dtype('float64')` raised from inside `bayes_opt`'s Gaussian-process
    surrogate refit):

    Every degenerate/failed candidate in this module used to be scored as
    a literal `float('-inf')` (`_combine_score()` and several early-return
    sites inside `_score_candidate()` below). `models/parallel_search.py::
    batch_bayesian_maximise()` registers whatever `objective_batch()`
    returns directly as a `BayesianOptimization` target
    (`optimizer.register(params=..., target=y)`), and `-inf` is a value
    `sklearn`'s `GaussianProcessRegressor.fit()` (bayes_opt's own
    surrogate) cannot fit against at all - the moment two or more
    candidates have been registered and `.suggest()` triggers the next GP
    refit, it raises exactly the reported `ValueError`, which
    `batch_bayesian_maximise()`'s own `except Exception` wraps into the
    `RuntimeError` this module's own batch-search call site catches and
    silently downgrades to a full serial re-run of the ENTIRE search
    (see `tune_joint_hyperparameters_and_weights()`'s own `except Exception
    as exc:` block around its `batch_bayesian_maximise()` call) - not a
    cosmetic issue, a hard crash of the batch path (and the wall-clock
    saving it exists to provide) triggered by the very first degenerate
    candidate the search happens to draw, which is common: an early
    random-init draw is often a poor hyperparameter combination for at
    least one group model, and `_combine_score()`'s own Round 5 fix
    (scoring a constant/degenerate combined prediction as a failure rather
    than a neutral `0.0`) made this MORE likely to be hit, not less - it
    turned a previously-neutral score into literal `-inf` for exactly the
    kind of early candidate a random search draws often.

    This mirrors `models/hyperparameter_tuning.py`'s own established
    `FAILURE_FLOOR`/`_failure_score()` mechanism byte-for-byte (see that
    module's own comment above its `FAILURE_FLOOR` constant for the full
    derivation - the old `-SCORE_CAP`/`-inf`-style fixed sentinel there
    caused the identical GP-surrogate-flattening failure mode, blueprint
    E2, and was fixed there long before this joint module existed; the fix
    was simply never ported here). A failing/degenerate candidate now
    scores strictly worse than every FINITE score actually observed so far
    THIS SEARCH (`state['worst_finite']`), or `FAILURE_FLOOR` (-10.0, the
    same constant `hyperparameter_tuning.py` uses - both modules' default
    `objective_mode='ensemble_score'` scoring is the identical `pearson -
    mse/var` scale) when nothing has succeeded yet - never a literal
    `-inf`, so its magnitude always stays on the same order as real
    observations and is something both bayes_opt's serial
    `normalize_y=True` target standardisation AND the batch path's own GP
    refit can always fit against.

    `state` is an explicit, shared, mutable `{'worst_finite': ...}` dict
    threaded through every `_score_candidate()`/`_combine_score()` call for
    one `tune_joint_hyperparameters_and_weights()` search (created once,
    at the top of that function) - a free function here, not a closure
    the way `hyperparameter_tuning.py::make_objective()` is, since
    `_score_candidate()` is called from several distinct places in this
    module (the search's own repeated candidate evaluations, the
    confirmatory fit, the 1-SE-rule's extra evaluations, the opt-in
    untuned-baseline-floor check) that all need to observe the same
    running worst-finite-score floor. `state=None` (e.g. a future direct
    caller that doesn't participate in shared tracking) falls back to the
    absolute `FAILURE_FLOOR` unconditionally, same as the very first
    failure of a real search."""
    if state is None:
        return FAILURE_FLOOR
    w = state.get('worst_finite')
    if w is None:
        return FAILURE_FLOOR
    return w - max(FAILURE_MARGIN, 0.10 * abs(w))


def _ensemble_score_of(combined, actual):
    """`pearson - mse/var` of one combined prediction vector against its targets - the
    'ensemble_score' objective of `_combine_score()` for a SUBSET of rows (ver4-34,
    R2.b: per-fold scores for the paired gate). Returns `None` for a degenerate
    vector (no variance in the targets or in the combination, or a non-finite value),
    which `_combine_score()` scores as a failure."""
    combined = np.asarray(combined, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if combined.size < 2 or combined.size != actual.size:
        return None
    var = float(np.var(actual))
    sd = float(np.std(combined)) * float(np.std(actual))
    if not np.isfinite(var) or var <= 0 or not np.isfinite(sd) or sd <= 0:
        return None
    mse = float(np.mean((combined - actual) ** 2))
    pearson = float(np.mean((combined - combined.mean()) * (actual - actual.mean())) / sd)
    value = pearson - mse / var
    return value if np.isfinite(value) else None


def _pearson_score_of(combined, actual):
    """ver4-50, R2: the Pearson correlation of one combined prediction vector against its targets
    - the numpy definition of the neural stack's `'pearson'` objective (twin of
    `hyper_stack.torch_pearson_score`, parity-tested).

    Why: `pearson - mse/var` depends on the SCALE of the predictions (the mse term), and the
    out-of-fold rows are more predictable than the test rows, so the network was rewarded for
    widening its predictions by concentrating weight on the most dispersed column (blueprint R1,
    hypothesis H1). Pearson is invariant to ``c * combined + d`` (a monotone function of the
    ensemble score profiled over an affine calibration). Same `None` contract as
    `_ensemble_score_of`: fewer than 2 rows, a size mismatch, no variance in the targets or in
    the combination, or a non-finite value return `None`."""
    combined = np.asarray(combined, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if combined.size < 2 or combined.size != actual.size:
        return None
    var = float(np.var(actual))
    sd = float(np.std(combined)) * float(np.std(actual))
    if not np.isfinite(var) or var <= 0 or not np.isfinite(sd) or sd <= 0:
        return None
    value = float(np.mean((combined - combined.mean()) * (actual - actual.mean())) / sd)
    return value if np.isfinite(value) else None


# ver4-50 (R2): the single authority for the neural stack's numpy score, keyed by `ns_objective`
# (`_ensemble_score_of` itself stays the score of every other strategy and is not touched).
_NS_SCORE_FNS = {'ensemble_score': _ensemble_score_of, 'pearson': _pearson_score_of}


def _robust_kwargs(name: str, settings: Optional[dict], *, n_bags: int, seed: int) -> dict:
    """ver4-36, R2.c: the keyword arguments of the robust estimator `name` (a key of
    `ROBUST_WEIGHT_ESTIMATORS`) taken from `settings` (an `estimator_settings` dict):
    `cluster_corr` for both, plus `es_steps` / `es_model_fraction` for 'ensemble_selection'.
    For 'bagged_cluster' this is exactly `n_bags`, `cluster_corr`, `seed` - the call the
    ver4-34 code made."""
    settings = settings or {}
    kwargs = dict(n_bags=int(n_bags),
                  cluster_corr=float(settings.get('cluster_corr', _JOINT_CLUSTER_CORR_DEFAULT)),
                  seed=int(seed))
    if name == 'ensemble_selection':
        kwargs['steps'] = int(settings.get('es_steps', _JOINT_ES_STEPS_DEFAULT) or 0)
        kwargs['model_fraction'] = float(settings.get('es_model_fraction',
                                                      _JOINT_ES_MODEL_FRACTION_DEFAULT))
    return kwargs


def _robust_weight_fn(name: str, settings: Optional[dict], *, n_bags: int, seed: int):
    """ver4-36, R2.c: the QUIET weights-only function of estimator `name` with its keywords
    bound - the `weight_fn` that `cross_fitted_combination()` calls once per training fold.
    Dispatches through `ROBUST_WEIGHT_ESTIMATORS` (the single registry, I3). Raises KeyError
    for an unregistered name (callers check `is_robust_estimator()` first)."""
    return partial(ROBUST_WEIGHT_ESTIMATORS[name][1],
                   **_robust_kwargs(name, settings, n_bags=n_bags, seed=seed))


def _robust_weights_full(name: str, P, y, settings: Optional[dict], *, n_bags: int, seed: int):
    """ver4-36, R2.c: the FULL function of estimator `name` (returns `(weights, diag)`),
    used for the REPORTED weights and for `freeze_low_weight`. Same dispatch as
    `_robust_weight_fn()`."""
    return ROBUST_WEIGHT_ESTIMATORS[name][0](
        P, y, **_robust_kwargs(name, settings, n_bags=n_bags, seed=seed))


def _combine_score(shares, pred_matrix, actual_valid, group_models, objective_mode,
                    diversity_penalty, diversity_method, state=None, oof_alpha=None,
                    oof_seed=0, weight_rule='analytic_penalised', shrinkage_alpha=1.0,
                    weight_estimator='legacy', estimator_settings=None,
                    return_folds=False):
    """Score one (weight-shares, prediction-matrix) pair on whichever
    `objective_mode` was requested, then apply the weight-collapse
    diversity penalty. Factored out of `_score_candidate()` (ver4-27, R2)
    so the SAME scoring math can be reused for a candidate's single
    real-validation-split evaluation AND for each inner resample fold of
    the repeated-resample path below, without duplicating it.

    BUGFIX (batch joint search reverting to serial - see `_failure_score()`
    above for the full root cause): a non-finite result is no longer
    returned as a literal `float('-inf')` - it is scored via
    `_failure_score(state)` instead, strictly below every finite score
    this search has actually observed so far. A genuinely finite score is
    also clipped to `[-SCORE_CAP, SCORE_CAP]` (the SAME finite-value-only
    clip `hyperparameter_tuning.py`'s own `objective()` already applies,
    for the same reason: an extreme but technically-finite outlier score
    can distort the GP surrogate's target standardisation almost as badly
    as a non-finite one) and, when finite, updates
    `state['worst_finite']` so later failures in this same search are
    graded relative to it. Never raises.

    ROUND 10 (`oof_alpha`): when not `None` (and `objective_mode` is
    'ensemble_score'), the combined prediction that is scored is the
    CROSS-FITTED one - `cross_fitted_combination()` solves the analytic
    weights on part of the validation rows and applies them, shrunk toward
    equal weighting by `oof_alpha`, to the rows held out of that solve - so a
    candidate is graded on how well its predictions combine with weights that
    have NOT seen the rows being scored. Previously every candidate was scored
    at the weights that minimise MSE on the very rows being scored (an
    optimistic, in-sample number whose optimism differs between candidates and
    grows with prediction diversity/noise), which is the score the search then
    maximised. `oof_alpha=None` reproduces the previous in-sample scoring
    exactly. `shares` (the full-sample weights) still drive the diversity
    penalty and the returned weights.

    ver4-32 R4 (`weight_rule`, default 'analytic_penalised' = ver4-31
    behaviour exactly): with 'shrunk' the candidate is scored at the weights
    that will actually be REPORTED - `apply_naive_shrinkage(shares,
    shrinkage_alpha)` (the single source for that shrinkage, also used by
    `select_weights_with_floor()`) - and the diversity penalty is NOT
    subtracted. Why: the reported weights are shrunk toward equal weighting,
    so scoring raw analytic weights minus a penalty (up to ~0.125 for one-hot
    weights at K=6, about the size of the score gaps between candidates)
    optimised a different combination from the one reported and marked down
    settings that make the strongest model stronger. Shrinkage already pulls
    toward uniform without penalising accuracy. The `oof_alpha` path is
    unchanged (already shrinkage-consistent) but, under 'shrunk', also skips
    the penalty. `shares` returned to callers stay the raw analytic weights
    (`_grade_weights()` shrinks them once).

    ver4-34 R2.a (`weight_estimator`, default 'legacy' = ver4-32 behaviour exactly):
    with 'bagged_cluster' (and `objective_mode='ensemble_score'`; 'dpt_ratio' keeps
    the legacy math, the caller says so) the candidate is scored on the CROSS-FITTED
    combination whose per-fold weights come from the robust bagged estimator
    (`robust_weights_only`, `estimator_settings['score_bags']` bags, correlation
    threshold `estimator_settings['cluster_corr']`, `estimator_settings['n_folds']`
    cross-fit folds - exactly the combination the REPORTED weights are meant to
    generalise to). No diversity penalty, no shrinkage and no equal-weight floor are
    applied: the weights are whatever the data support. `shares` is then unused and
    may be `None` (the caller skips the analytic solve). `weight_rule` /
    `shrinkage_alpha` / `oof_alpha` are ignored.
    `estimator_settings['folds']`, when given, is the positional row partition whose
    per-fold scores `return_folds=True` reports for the non-cross-fitted paths (the
    cross-fitted path reports its own, identical, partition). `return_folds=True`
    returns `(score, fold_scores)` where `fold_scores` is a list of per-fold
    ensemble scores, or `None` when they cannot be formed (a degenerate fold, the
    'dpt_ratio' objective, or no partition); a failed candidate returns
    `(failure_score, None)`."""
    _bagged = (is_robust_estimator(weight_estimator) and objective_mode != 'dpt_ratio')
    _est = estimator_settings or {}
    _folds_used = None
    combined = None
    _shrunk = (weight_rule == 'shrunk') and not _bagged
    if _shrunk:
        shares = np.asarray(apply_naive_shrinkage(shares, shrinkage_alpha), dtype=float)
    if objective_mode == 'dpt_ratio':
        # ver4-26, R2: reuses ensemble_mse_objective() - the "Many-Model
        # error" (DPT's own first_term) evaluated at these weights -
        # minimising it directly rewards diversity, since DPT's own
        # identity (diversity_summary.py SS16.3) is
        # first_term = second_term - third_term: a larger third_term
        # (diversity) at fixed second_term (average model error) makes
        # first_term SMALLER, so minimising first_term is what "chosen
        # for the ensemble's diversity benefit" (Requirement 2) means in
        # THIS objective's own terms, not the literal published DPT
        # ratio (which R6's own read_regularization_settings() docstring
        # already documents as vertex-collapsing when used as a search
        # objective - see that function's own note).
        mse = ensemble_mse_objective(shares, 1.0, None, group_models,
                                      pred_matrix=pred_matrix, actual=actual_valid)
        score = -mse if np.isfinite(mse) else None
    else:
        # 'ensemble_score' (default) - pearson_valid - mse_valid/var(actual)
        # on the WEIGHTED-COMBINED validation prediction, the same scale/
        # sign convention models/hyperparameter_tuning.py's own per-model
        # objective already uses (maximised, both here and there).
        if _bagged:
            _weight_fn = _robust_weight_fn(
                weight_estimator, _est,
                n_bags=int(_est.get('score_bags', _JOINT_SCORE_BAGS_DEFAULT)),
                seed=int(oof_seed))
            combined, _folds_used = cross_fitted_combination(
                pred_matrix, actual_valid, 1.0, n_folds=int(_est.get('n_folds', 5)),
                seed=oof_seed, weight_fn=_weight_fn, return_folds=True)
        elif oof_alpha is None:
            combined = pred_matrix @ shares
        else:
            combined = cross_fitted_combination(pred_matrix, actual_valid, oof_alpha, seed=oof_seed)
        resid = combined - actual_valid
        mse = float(np.mean(resid ** 2))
        var = float(np.var(actual_valid))
        if var <= 0 or not np.isfinite(var):
            score = None
        else:
            sd = float(np.std(combined)) * float(np.std(actual_valid))
            if sd <= 0 or not np.isfinite(sd):
                # BUGFIX (joint-optimisation-underperformance-fix, Round 5):
                # `var > 0` was already confirmed above (actual_valid has
                # genuine spread), so reaching here means std(combined)==0 -
                # this candidate's WEIGHTED-COMBINED prediction collapsed to
                # a single constant value across every validation row (a
                # degenerate fit, not merely "uncorrelated"). The previous
                # `pearson = 0.0` treated that as a NEUTRAL outcome - on
                # this branch's own scale (`score = pearson - mse/var`), a
                # constant-prediction candidate with middling MSE could
                # therefore score close to, or even above, a candidate that
                # produces genuine, if imperfect, correlated variation,
                # letting the search actively drift toward degenerate
                # weight/hyperparameter combinations instead of away from
                # them. models/hyperparameter_tuning.py's own objective()
                # (this module's own explicit design precedent - see this
                # module's top-level docstring) never scores a degenerate/
                # non-finite Pearson r as neutral; it always scores it as
                # the worst possible outcome. Matched here: a degenerate
                # combined prediction is scored as an outright failure,
                # never as "no correlation".
                score = None
            else:
                pearson = float(np.mean((combined - combined.mean()) * (actual_valid - actual_valid.mean())) / sd)
                score = pearson - mse / var

    # BUGFIX (weight-collapse report): subtract the diversity regulariser
    # from whichever branch above produced a finite score - see this
    # module's top-level comment for the full rationale. `shares` already
    # sums to 1 (stick-breaking), so `w_sum=1.0` is passed directly rather
    # than recomputing it.
    if (score is not None and np.isfinite(score) and diversity_penalty and not _shrunk
            and not _bagged):
        score = score - diversity_regularizer(shares, 1.0, diversity_penalty, diversity_method)

    # BUGFIX (batch joint search reverting to serial - see `_failure_score()`
    # above): a non-finite result is scored relative to this search's own
    # worst-finite-score-so-far, never as a literal `float('-inf')` -
    # `bayes_opt`'s Gaussian-process surrogate cannot fit against `-inf`
    # at all (root cause of the reported "Input y contains infinity"
    # crash). A genuinely finite score is clipped to `[-SCORE_CAP,
    # SCORE_CAP]` (mirrors `hyperparameter_tuning.py`'s own `objective()`)
    # and used to advance `state['worst_finite']` so later failures in
    # this same search stay correctly ordered below it.
    if score is None or not np.isfinite(score):
        _failed = _failure_score(state)
        return (_failed, None) if return_folds else _failed

    score = float(np.clip(score, -SCORE_CAP, SCORE_CAP))
    if state is not None:
        state['worst_finite'] = (
            score if state.get('worst_finite') is None else min(state['worst_finite'], score)
        )
    if not return_folds:
        return score
    # ver4-34, R2.b: per-fold scores of the SAME combined vector (never re-fitted).
    fold_scores = None
    _partition = _folds_used if _folds_used else _est.get('folds')
    if combined is not None and _partition:
        _per_fold = [_ensemble_score_of(combined[f], np.asarray(actual_valid)[f])
                     for f in _partition]
        if all(v is not None for v in _per_fold):
            fold_scores = [float(v) for v in _per_fold]
    return score, fold_scores


def _cv_partition(n: int, folds: int, seed: int) -> List[np.ndarray]:
    """ver4-34, R2.b: the POSITIONAL row partition of the cross-validated objective -
    `folds` disjoint index arrays that together cover `range(n)` exactly once.

    Why: the CV objective builds each model's column as an out-of-fold (OOF) vector
    over train+valid and then scores the cross-fitted combination on the SAME
    partition (per-fold scores feed the paired fresh gate). The partition is
    `ensemble_regularization._fold_indices(n, folds, default_rng(seed))` - the exact
    draw `cross_fitted_combination()` makes for the same `(n, folds, seed)` - so the fit
    folds and the weight-cross-fitting folds are one and the same, and a different
    `seed` (the fresh gate adds its own offset to `inner_split_seed_base`) gives a fresh
    partition automatically (A-2b.2).

    Args: `n` number of rows (>= 2); `folds` requested fold count (clipped to
    [2, n]); `seed` any int. Returns the list of held-out position arrays (not sorted
    inside a fold; use them with `.iloc`). Never raises for n >= 2."""
    n_folds = int(max(2, min(int(folds), int(n))))
    return _fold_indices(int(n), n_folds, np.random.default_rng(int(seed)))


def _cv_frame(train, valid):
    """ver4-34, R2.b: the train+valid frame the CV objective folds over - `train` rows
    first, then `valid` rows, with a fresh RangeIndex.

    Why `ignore_index=True`: the task splits all keep their own `reset_index(drop=True)`
    labels, so label 0 exists in train, valid AND test; every later selection is
    therefore POSITIONAL (`.iloc`), never by label (invariant I4). `test` is never an
    argument here, so by construction no test row can enter the frame (A-2b.1).

    Raises `AssertionError` when the row count of the result is not
    `len(train) + len(valid)`."""
    tv = pd.concat([train, valid], ignore_index=True)
    if len(tv) != len(train) + len(valid):
        raise AssertionError(
            f"[joint_tuning] internal error: train+valid frame has {len(tv)} rows, expected "
            f"{len(train)} + {len(valid)} = {len(train) + len(valid)}.")
    return tv


def _cv_fold_frames(tv, partition: Sequence[np.ndarray]) -> list:
    """ver4-34 R2.b (extracted unchanged in ver4-36 so the asynchronous library search can
    build the fold frames once): `(held_positions, train_part, held_part)` per fold of
    `partition` over the train+valid frame `tv`. Selection is POSITIONAL (`.iloc`, I4)."""
    n = len(tv)
    fold_frames = []
    for held in partition:
        held = np.asarray(held, dtype=int)
        keep = np.ones(n, dtype=bool)
        keep[held] = False
        fold_frames.append((held, tv.iloc[np.flatnonzero(keep)], tv.iloc[held]))
    return fold_frames


def _test_sink_key(model: str, params) -> tuple:
    """ver4-36, R2.d: the key a library entry's test column is stored under - the model and
    its DECODED parameters (the same pair that identifies a fit; never the unit-cube point)."""
    return (model, repr(params))


def _record_test_columns(sink: dict, request: Tuple[str, list], results: Sequence[Optional[dict]]
                         ) -> None:
    """ver4-36, R2.d (library dump only): remember the TEST-row prediction of the fits that
    produced one library column. One result (objective 'valid') is stored as is; several
    (one per CV fold) are averaged - the test column of a CV entry is the mean of its fold
    models' test predictions. A missing or wrong-length prediction stores nothing (the dump
    then writes NaN for that entry). Write-only: nothing in the search reads `sink` (I7)."""
    preds = []
    for res in results:
        pt = None if res is None else res.get('predicted_test')
        if pt is None:
            return
        preds.append(np.asarray(pt, dtype=float))
    if not preds or any(p.shape != preds[0].shape for p in preds):
        return
    sink[_test_sink_key(request[0], request[1])] = np.mean(np.vstack(preds), axis=0)


def _cv_fit_columns(requests: Sequence[Tuple[str, list]], *, tv, partition: Sequence[np.ndarray],
                     seed: int, test, runner, RESULT_NAME, runner_ctx: Optional[dict] = None,
                     stop_on_failure: bool = True, fidelity: str = 'cv',
                     fold_subset: Optional[Sequence[int]] = None
                     ) -> List[Optional[np.ndarray]]:
    """ver4-34, R2.b: out-of-fold (OOF) prediction column over `tv` for every
    `(model, params)` request - `len(partition)` fit units per request, ALL handed to
    ONE `_run_units()` call.

    Why: the CV objective scores a candidate on how its columns combine over the whole
    train+valid frame (~150 rows) instead of ~50 validation rows. For fold `f` the
    model is fitted on `tv` minus the held-out rows and predicts the held-out rows;
    the predictions are written back by POSITION (`oof[held] = pv`), so the column is
    aligned with `tv` (I4). Fold units are tagged `cv{f}@{seed}` with fidelity `fidelity`
    so a `_FitCache` can tell partitions apart (cache keys carry no data).

    Args: `requests` list of `(model, params)`; `tv` frame from `_cv_frame()`;
    `partition` from `_cv_partition()`; `seed` the partition seed (only used in the
    unit tags); `test` passed through to the runner untouched; `runner_ctx` the usual
    n_jobs/worker_init/worker_init_args/cache/counter/stage dict;
    `stop_on_failure` the `_run_units()` early-abort switch (True keeps the serial
    abort-on-first-failure order, False fits everything for a parallel batch).

    ver4-38, R1.b: `fold_subset` (default None = every fold, ver4-37 behaviour) restricts the
    fits to the listed fold numbers - the "race" stage of CV racing. The returned column is
    NaN outside the held-out rows of those folds, is `None` when a listed fold failed, and
    nothing is written to the test sink (a partial column is never a library entry). Fold
    units keep their `cv{f}@{seed}` tags, so a later full call hits the fit cache for the
    folds already fitted. Raises `ValueError` for a fold number outside the partition.

    Returns one `np.ndarray` of length `len(tv)` per request, or `None` for a request
    with a failed fold, a wrong-length prediction or a non-finite value (the worker
    already logged a fit failure). Raises `AssertionError` when `partition` does not
    cover each row of `tv` exactly once (a contract violation)."""
    n = len(tv)
    covered = np.concatenate([np.asarray(h, dtype=int) for h in partition]) if len(partition) else np.array([], dtype=int)
    if covered.size != n or np.unique(covered).size != n or (n and (covered.min() < 0 or covered.max() >= n)):
        raise AssertionError(
            f"[joint_tuning] internal error: the CV partition covers {covered.size} position(s) "
            f"({np.unique(covered).size} distinct) for {n} train+valid row(s) - every row must be "
            f"held out exactly once.")
    rc = runner_ctx or {}
    fold_frames = _cv_fold_frames(tv, partition)
    if fold_subset is None:
        used_folds = list(range(len(fold_frames)))
    else:
        used_folds = sorted({int(f) for f in fold_subset})
        bad = [f for f in used_folds if f < 0 or f >= len(fold_frames)]
        if bad:
            raise ValueError(
                f"[joint_tuning] fold_subset {bad!r} is outside the {len(fold_frames)}-fold "
                f"partition (valid fold numbers are 0..{len(fold_frames) - 1}).")
    units = []
    for model, params in requests:
        for f in used_folds:
            _held, tr_f, va_f = fold_frames[f]
            units.append(_FitUnit(model=model, params=params, split_tag=f'cv{f}@{int(seed)}',
                                  train=tr_f, valid=va_f, fidelity=fidelity, cv_weights=True))
    fitted = _run_units(
        units, runner=runner, RESULT_NAME=RESULT_NAME, test=test, n_jobs=rc.get('n_jobs', 1),
        worker_init=rc.get('worker_init'), worker_init_args=rc.get('worker_init_args', ()),
        cache=rc.get('cache'), stop_on_failure=stop_on_failure, counter=rc.get('counter'),
        stage=rc.get('stage'), fit_flags=rc.get('fit_flags'))
    n_f = len(used_folds)
    sink = rc.get('test_sink')
    columns: List[Optional[np.ndarray]] = []
    for r in range(len(requests)):
        col = np.full(n, np.nan, dtype=float)
        ok = True
        covered_rows = []
        for j, f in enumerate(used_folds):
            held = fold_frames[f][0]
            res = fitted[r * n_f + j]
            pv = None if res is None else res.get('predicted_valid')
            if pv is None or len(pv) != len(held):
                ok = False
                break
            col[held] = np.asarray(pv, dtype=float)
            covered_rows.append(held)
        if fold_subset is None:
            finite = bool(np.all(np.isfinite(col)))
        else:
            rows_done = (np.concatenate(covered_rows).astype(int) if covered_rows
                         else np.array([], dtype=int))
            finite = bool(rows_done.size) and bool(np.all(np.isfinite(col[rows_done])))
        columns.append(col if (ok and finite) else None)
        if sink is not None and ok and fold_subset is None:
            _record_test_columns(sink, requests[r], fitted[r * n_f:(r + 1) * n_f])
    return columns


def _screen_subsample_train(train, fraction, min_rows, seed):
    """FEATURE (Round 7): a random ROW subsample of `train` ONLY - never
    `valid`/`test` (the SAME invariant `_split_train_rows()` above already
    keeps - the task's real validation split is never touched by anything
    other than the ONE confirmatory evaluation). Used as this module's own
    multi-fidelity SCREENING rung: a candidate is scored on a small
    fraction of the task's own training rows, through the exact SAME
    scoring pipeline (`_score_candidate()` via `joint_objective()`) and
    against the exact SAME real validation targets as a full evaluation -
    never a different metric, never a proxy score, just fewer training
    rows to fit each group model on. Training-row count is what drives
    most of a model's own fit cost regardless of model type (unlike
    `models/hyperparameter_tuning.py`'s own `_cheapen_mcmc()`/
    `_cheapen_fidelity()`, which only cheapen specific per-model fields
    for models that happen to have an entry in `MCMC_LENGTH_FIELDS`/
    `FIDELITY_FIELDS`), so this applies uniformly to any `group_models`
    combination.

    Returns `None` when `train` is too small to subsample meaningfully
    below `min_rows` - the caller skips screening entirely for that task
    rather than force an unreasonably tiny, unrepresentative fit.
    """
    n = len(train)
    n_screen = max(int(min_rows), int(round(n * fraction)))
    # ROUND 10: also skip when the subsample would still be most of the
    # training split - there is then nothing worth saving, and the noisier
    # screening score would be bought for (almost) no speed-up.
    if n_screen >= n or n_screen > _JOINT_MULTI_FIDELITY_MAX_TRAIN_FRACTION * n:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.choice(train.index.to_numpy(), size=n_screen, replace=False)
    return train.loc[idx]


def _decode_model_params(m: str, chunk, *, specs: list, base_params: list,
                          search_fidelity: bool, skip_decode: bool = False) -> list:
    """ver4-32, R2: decode ONE model's slice `chunk` of a unit-cube point into
    its final hyperparameter list. Extracted unchanged from the per-model loop
    of `_score_candidate()` so the block search can build single-model fit
    units without scoring a whole candidate (`_fit_columns_batch()`); both
    callers therefore share one implementation.

    Args: `m` model name; `chunk` this model's coordinates (length
    `len(specs)`); `specs` its `ParamSpec` list; `base_params` the model's
    configured hyperparameters (never mutated); `search_fidelity` apply the
    search-phase cheapening (explainability off, MCMC length, fidelity) to
    every field that is NOT in `specs`; `skip_decode` use `base_params`
    exactly instead of `_decode(_encode(base))` (lossy for a categorical
    value outside its `choices`). Returns the new list. Raises whatever
    `_decode()` raises for a malformed chunk."""
    base = list(base_params)
    if len(specs) and not skip_decode:
        decoded = _decode(specs, chunk)
        for idx, val in decoded.items():
            base[idx] = val
    # Round 4: burnIn < nIter enforced on search AND confirmatory fits.
    base = _enforce_mcmc_burnin_invariant(m, base)
    if search_fidelity:
        tuned_indices = {sp.index for sp in specs}
        base = _disable_explainability(m, base)
        base = _cheapen_mcmc(m, base, base_params, tuned_indices)
        base = _cheapen_fidelity(m, base, base_params, tuned_indices)
    return base


def _score_candidate(x, *, group_models, base_params_by_model, specs_by_model,
                      train, valid, test, run_model_fn, RESULT_NAME,
                      objective_mode, search_fidelity,
                      diversity_penalty=0.0, diversity_method='ridge_to_uniform',
                      valid_repeats=1, inner_split_seed_base=0, state=None,
                      weight_cv_alpha=None, use_base_params=False,
                      keep_results: bool = False, use_base_params_for: Optional[Set[str]] = None,
                      runner_ctx: Optional[dict] = None, weight_rule: str = 'analytic_penalised',
                      shrinkage_alpha: float = 1.0, weight_estimator: str = 'legacy',
                      estimator_settings: Optional[dict] = None,
                      objective_data: str = 'valid', cv_folds: int = _JOINT_CV_FOLDS_DEFAULT):
    """Decode one unit-cube point `x` into per-model hyperparameters, refit
    every group model at those hyperparameters via `run_model_fn`, then
    solve for THIS candidate's own best-achievable ensemble weights
    analytically (`ensemble_regularization.analytic_simplex_weights()`,
    Round 6 - see this module's own top-level Round 6 note) from its
    resulting validation predictions, and score. `x` spans ONLY the
    concatenated per-model hyperparameter dimensions as of Round 6 - it no
    longer carries any trailing stick-breaking weight coordinates, so the
    ensemble weights are never a free variable the outer search has to
    locate; they are the exact solution to a convex sub-problem, recomputed
    fresh for every candidate (and, in the repeated-resample branch below,
    fresh for every inner fold of every candidate).

    `search_fidelity=True` (every candidate evaluated DURING the search)
    applies the SAME search-phase cheapening single-model tuning already
    uses (`_disable_explainability`/`_cheapen_mcmc`/`_cheapen_fidelity` -
    R3 lever 2) to every group model, skipping any field this candidate
    itself is searching over (`tuned_indices`). `search_fidelity=False`
    (the ONE final confirmatory evaluation, at the search's own winning
    candidate) never cheapens - full explainability/MCMC-length/iteration
    settings, mirroring `tune_model_hyperparameters()`'s own search/confirm
    split exactly.

    `diversity_penalty`/`diversity_method` (BUGFIX, weight-collapse report -
    see this module's own top-level comment above `_JOINT_DIVERSITY_
    PENALTY_DEFAULT`): a strictly-convex `ensemble_regularization.
    diversity_regularizer()` penalty, evaluated on this candidate's own
    analytically-solved weight shares and SUBTRACTED from the score (both
    objective modes are maximised here, so subtracting - not adding - is
    what penalises). Pre-Round-6 this steered the search AWAY from a
    near-one-hot weight vector that only looked good because of
    validation-split noise in the weight search itself; post-Round-6 the
    weights are no longer searched (they are this candidate's own exact
    analytic optimum, given its predictions), so this penalty now instead
    shapes which HYPERPARAMETER regions the outer search prefers - it
    still discourages hyperparameter choices whose resulting predictions
    are so collinear/dominated that the analytic solve degenerates toward
    a single model, without changing what weights are actually reported
    for a given candidate's predictions (the analytic solve itself is
    never penalised - only the score used to rank candidates is). `0.0`
    (falsy) is a strict no-op, reproducing this function's exact pre-fix
    behaviour.

    Never raises: a candidate any group model fails to fit at (a numerical
    divergence, an R MCMC failure, ...) scores as a failure via
    `_failure_score(state)` (BUGFIX, batch-joint-search-reverting-to-serial
    report - see `_failure_score()`'s own docstring for why this is no
    longer a literal `float('-inf')`) and returns `(failure_score, None)` -
    mirrors `Bayesian_optimisation.py`'s own batch-search failure fallback
    (R2's own disclosed failure mode), now on the SAME never-return-a-
    literal-`-inf` footing `models/hyperparameter_tuning.py`'s own
    `_failure_score()` already established. `state`, when supplied, is
    forwarded to every `_combine_score()` call inside this function so a
    real, finite score updates it and a later failure within the SAME
    search is graded relative to it - see `tune_joint_hyperparameters_and_
    weights()`'s own single, shared `state` dict for the whole search.

    ver4-32: `weight_rule`/`shrinkage_alpha` (R4) are forwarded to
    `_combine_score()` - 'shrunk' scores the candidate at the shrunk weights
    that are reported and drops the diversity penalty; the payload `weights`
    stay the raw analytic ones. Defaults reproduce ver4-31 exactly. The
    per-model decoding/cheapening of the loop below lives in
    `_decode_model_params()` (shared with the block search). Repeated-resample
    fold fits are labelled `fold{b}@{seed}` so a fit cache can tell resamples
    of different seeds apart.

    ver4-34 (all four new keywords default to the ver4-32 behaviour exactly):
    `weight_estimator`/`estimator_settings` (R2.a) are forwarded to `_combine_score()`;
    with 'bagged_cluster' a SEARCH-fidelity evaluation skips the analytic solve (the
    score uses the robust cross-fitted weights, so `shares` is not needed) and the
    confirmatory evaluation still returns the analytic `weights` in its payload (they
    are only a diagnostic then - `_grade_weights()` computes the reported weights).
    `objective_data='cv_train_valid'` (R2.b, search fidelity only - the confirmatory
    fit is always train -> valid, so the final models stay train-only): the candidate
    is scored on out-of-fold columns over train+valid (`_cv_fit_columns()`, `cv_folds`
    folds drawn with seed `inner_split_seed_base + _JOINT_CV_SEED_OFFSET`), `valid_repeats`
    is ignored, and the payload has the same `{'se', 'n_folds', 'fold_scores'}` shape as
    the repeated-resample branch, with the fold scores of the cross-fitted combination, so
    `_paired_fresh_gate()` works unchanged. The gate passes a different
    `inner_split_seed_base`, which yields a fresh partition (A-2b.2).

    Returns `(score, payload)` where `payload` is `None` on any failure,
    else `{'final_params_by_model': {...}, 'weights': {...},
    'tuned_values_by_model': {...}, 'pred_matrix': ..., 'actual_valid': ...}`
    - the exact per-model hyperparameter list and ensemble weight this
    candidate used, keyed by model name (not a positional array - see this
    module's own Change Summary note on why a dict is used here instead of
    the blueprint's own illustrative `np.ndarray` sketch: every caller in
    `genomic_prediction.py` needs to look a specific model's own value up
    by name, which a dict does directly and a positional array would
    require `group_models.index(m)` for at every call site instead).
    `tuned_values_by_model[m]` is `{field_label: final_value}` for every
    tunable field of model `m` - the SAME shape `tune_model_hyperparameters
    ()`'s own `tuned_values` already has for single-model HP_TUNE (BUGFIX,
    hyperparameter.csv recording report: this was never returned at all
    before, so genomic_prediction.py's own joint hp_record row had no
    tuned-value columns). `pred_matrix`/`actual_valid` (this candidate's
    own (N, K) validation-prediction matrix and (N,) actual targets) are
    returned so the caller's post-search validation-MSE floor (see
    `tune_joint_hyperparameters_and_weights()` below) can grade candidate
    weight vectors on the CONFIRMATORY fit's own realised predictions
    without refitting anything a second time.
    """
    x = np.asarray(x, dtype=float)
    # ver4-28, R1/R3: `runner_ctx` (n_jobs/worker_init/worker_init_args/
    # cache/counter/stage) drives every `_run_units()` call below - absent
    # (`None`, every pre-ver4-28 caller) reproduces the previous in-process,
    # serial, uncached fitting behaviour exactly (`n_jobs=1`, no cache).
    runner_ctx = runner_ctx or {}
    _ctx_n_jobs = runner_ctx.get('n_jobs', 1)
    _ctx_worker_init = runner_ctx.get('worker_init')
    _ctx_worker_init_args = runner_ctx.get('worker_init_args', ())
    _ctx_cache = runner_ctx.get('cache')
    _ctx_counter = runner_ctx.get('counter')
    _ctx_stage = runner_ctx.get('stage')
    offset = 0
    final_params_by_model = {}
    for m in group_models:
        specs = specs_by_model[m]
        k = len(specs)
        # ver4-28, R3: `use_base_params_for` (a set of model names) fixes the
        # lossy encode/decode round trip for a model whose own block hasn't
        # been touched yet by a block-coordinate search cycle - the SAME
        # exact-baseline mechanism `use_base_params` already provides for
        # EVERY model (Round 10), scoped to just one model here.
        # ROUND 10 (`use_base_params`): evaluate the user's own configured
        # hyperparameters EXACTLY, instead of `_decode(_encode(base))`. The
        # encode/decode round trip is lossy for a categorical field whose
        # configured value is not one of its tunable `choices` (encoded as
        # choice 0) and for float/step rounding, so "fall back to the untuned
        # defaults" could silently apply slightly DIFFERENT hyperparameters.
        _skip_decode = use_base_params or (use_base_params_for is not None and m in use_base_params_for)
        # Bug fix (Round 4) and search-phase cheapening live in
        # `_decode_model_params()` (ver4-32 refactor, behaviour unchanged).
        final_params_by_model[m] = _decode_model_params(
            m, x[offset:offset + k], specs=specs, base_params=base_params_by_model[m],
            search_fidelity=search_fidelity, skip_decode=_skip_decode,
        )
        offset += k

    # Round 6 safety check: pre-Round-6, any `x` coordinates left over after
    # decoding every model's hyperparameters were consumed as stick-breaking
    # weight coordinates, so a search-space sizing mismatch here would have
    # silently been absorbed as "extra weight dimensions" rather than
    # surfaced. Now that there is no weight portion left to absorb it, a
    # leftover coordinate can only mean `total_dims`/`specs_by_model` and the
    # `x` this function was actually called with disagree - a search-space
    # construction bug, not an expected extra weight coordinate - so this is
    # raised loudly rather than silently dropped.
    if len(x) != offset:
        raise AssertionError(
            f"[joint_tuning] internal error: candidate vector length ({len(x)}) does not "
            f"match the concatenated hyperparameter dimension count ({offset}) across "
            f"{list(group_models)!r} - the joint search space no longer allocates any "
            f"ensemble-weight dimensions (Round 6), so every element of `x` must be "
            f"accounted for by a model's own tunable hyperparameters."
        )

    # Round 6: `x` no longer carries any trailing weight coordinates (see
    # this function's own docstring) - `offset` has consumed every
    # dimension of `x` by this point. Ensemble weights are solved
    # analytically below, per evaluation (once per fold in the
    # repeated-resample branch, once here otherwise), from whichever
    # validation predictions that evaluation actually produced - never
    # decoded from `x`.

    _bagged = (is_robust_estimator(weight_estimator) and objective_mode != 'dpt_ratio')

    # ver4-34, R2.b: cross-validated super-learner objective (search-phase only).
    if objective_data == 'cv_train_valid' and search_fidelity:
        if valid is None or valid.shape[0] == 0:
            return _failure_score(state), None
        cv_seed = int(inner_split_seed_base) + _JOINT_CV_SEED_OFFSET
        tv = _cv_frame(train, valid)
        partition = _cv_partition(len(tv), cv_folds, cv_seed)
        cv_cols = _cv_fit_columns(
            [(m, final_params_by_model[m]) for m in group_models], tv=tv, partition=partition,
            seed=cv_seed, test=test, runner=run_model_fn, RESULT_NAME=RESULT_NAME,
            runner_ctx=runner_ctx)
        cv_failed = [m for m, col in zip(group_models, cv_cols) if col is None]
        if cv_failed:
            print(f"[joint_tuning] NOTE: cross-validated candidate evaluation failed for "
                  f"{cv_failed!r} (a fold did not fit or returned an unusable prediction) - this "
                  f"candidate scores as a failure and is discarded; the search continues.")
            return _failure_score(state), None
        cv_matrix = np.column_stack(cv_cols)
        cv_actual = np.asarray(tv.iloc[:, -1], dtype=float)
        cv_est = dict(estimator_settings or {})
        cv_est['n_folds'] = len(partition)
        cv_est['folds'] = partition
        cv_shares = None if _bagged else analytic_simplex_weights(cv_matrix, cv_actual)
        cv_score, cv_fold_scores = _combine_score(
            cv_shares, cv_matrix, cv_actual, group_models, objective_mode, diversity_penalty,
            diversity_method, state=state, oof_alpha=weight_cv_alpha, oof_seed=cv_seed,
            weight_rule=weight_rule, shrinkage_alpha=shrinkage_alpha,
            weight_estimator=weight_estimator, estimator_settings=cv_est, return_folds=True)
        if not np.isfinite(cv_score):
            return _failure_score(state), None
        cv_arr = np.asarray(cv_fold_scores if cv_fold_scores is not None else [], dtype=float)
        cv_se = (float(cv_arr.std(ddof=1) / math.sqrt(len(cv_arr))) if len(cv_arr) > 1 else 0.0)
        return float(cv_score), {'se': cv_se, 'n_folds': int(len(cv_arr)),
                                 'fold_scores': cv_arr.tolist()}

    # ver4-27, R2: repeated-resample search-phase scoring - see this
    # branch's own comment below for the full rationale. Never active for
    # the ONE confirmatory (search_fidelity=False) evaluation, regardless
    # of valid_repeats, and never active when valid_repeats<=1 (I11) -
    # both reproduce this function's exact pre-ver4-27 behaviour bit for
    # bit.
    effective_repeats = max(1, int(valid_repeats)) if search_fidelity else 1
    if effective_repeats > 1 and len(train) < 4 * effective_repeats:
        print(f"[joint_tuning] NOTE: this task's training split ({len(train)} rows) is too "
              f"small to draw {effective_repeats} independent inner resamples from (need at "
              f"least {4 * effective_repeats}) - falling back to a single evaluation on this "
              f"task's own validation split for this candidate (joint valid_repeats effectively "
              f"1 for this task only).")
        effective_repeats = 1

    if effective_repeats > 1:
        # Mirrors models/hyperparameter_tuning.py's own HP_TUNE_VALID_REPEATS
        # mechanism (make_objective()'s _evaluate_detailed()), never
        # previously wired in for the joint search despite
        # inner_split_seed_base already being accepted as a parameter here.
        # Scores this SEARCH-PHASE candidate as the MEAN over
        # `effective_repeats` independent inner train/validation resamples
        # drawn from the current task's own TRAINING split only - the
        # task's real `valid` frame is never touched by this branch, and
        # stays reserved for the ONE confirmatory evaluation below, exactly
        # as single-model tuning already reserves it (invariant I7).
        # Reduces how much a candidate's score - and therefore which
        # hyperparameters+weights the search converges on - is driven by
        # whichever single validation split happened to be drawn, which
        # matters more here than for single-model tuning: the joint
        # search's own combined space is typically much larger (see
        # JOINT_SEARCH_DIM_WARN_THRESHOLD), so it is correspondingly noisier
        # to navigate on a single noisy per-candidate score. The payload
        # returned here is always None: joint_objective() only ever uses
        # this function's SCORE for a search-phase candidate anyway (see
        # its own docstring), and the caller's `weights`/`final_params_
        # by_model`/`pred_matrix` payload is always rebuilt from the ONE
        # confirmatory, full-fidelity, real-validation-split evaluation of
        # the search's own winning `x` (tune_joint_hyperparameters_and_
        # weights() below), never from a search-phase candidate.
        fold_scores = []
        for b in range(effective_repeats):
            tr_b, va_b = _split_train_rows(train, inner_split_seed_base + b)
            if tr_b is None:
                # BUGFIX (batch joint search reverting to serial - see
                # `_failure_score()`'s own docstring): every early-return
                # site in this branch used to hand back a literal
                # `float('-inf')` for the WHOLE candidate the instant any
                # one fold failed - `bayes_opt`'s GP surrogate cannot fit
                # against `-inf` at all, which is the root cause of the
                # reported batch-search crash. Control flow is unchanged
                # (a failed fold still aborts this candidate immediately,
                # exactly as before); only the numeric value returned in
                # place of `-inf` changed, to `_failure_score(state)` -
                # strictly worse than every finite score this search has
                # actually observed so far, never a non-finite sentinel.
                return _failure_score(state), None
            # ver4-28, R1: this fold's own K model refits are now fit UNITS
            # run through `_run_units()` - serial (byte-identical order/
            # abort-on-first-failure to the previous direct loop) unless
            # `runner_ctx` requests parallel fan-out.
            units_b = [
                _FitUnit(model=m, params=final_params_by_model[m],
                         split_tag=f'fold{b}@{inner_split_seed_base + b}',
                         train=tr_b, valid=va_b, fidelity='search')
                for m in group_models
            ]
            fitted_b = _run_units(
                units_b, runner=run_model_fn, RESULT_NAME=RESULT_NAME, test=test,
                n_jobs=_ctx_n_jobs, worker_init=_ctx_worker_init,
                worker_init_args=_ctx_worker_init_args, cache=_ctx_cache,
                counter=_ctx_counter, stage=_ctx_stage, fit_flags=runner_ctx.get('fit_flags'),
            )
            preds_b = []
            _fold_failed = False
            for m, result in zip(group_models, fitted_b):
                if result is None:
                    _fold_failed = True
                    break
                _pv = result.get('predicted_valid')
                if _pv is None or len(_pv) == 0:
                    _fold_failed = True
                    break
                preds_b.append(np.asarray(_pv, dtype=float))
            if _fold_failed:
                return _failure_score(state), None
            try:
                pred_matrix_b = np.column_stack(preds_b)
            except Exception:
                return _failure_score(state), None
            actual_valid_b = np.asarray(va_b.iloc[:, -1], dtype=float)
            if pred_matrix_b.shape[0] != actual_valid_b.shape[0]:
                return _failure_score(state), None
            # Round 6: this fold's own best-achievable ensemble weights,
            # solved analytically from THIS fold's own predictions - never
            # the same fixed weight vector reused across every fold (there
            # is no such vector any more; each fold gets its own optimum,
            # exactly like each fold already gets its own refit models).
            shares_b = None if _bagged else analytic_simplex_weights(pred_matrix_b, actual_valid_b)
            fold_score = _combine_score(shares_b, pred_matrix_b, actual_valid_b, group_models,
                                         objective_mode, diversity_penalty, diversity_method,
                                         state=state, oof_alpha=weight_cv_alpha,
                                         oof_seed=inner_split_seed_base + b,
                                         weight_rule=weight_rule, shrinkage_alpha=shrinkage_alpha,
                                         weight_estimator=weight_estimator,
                                         estimator_settings=estimator_settings)
            # `_combine_score()` itself can no longer return a non-finite
            # value at all as of this same fix (see its own BUGFIX
            # comment) - this check is a pure defensive backstop, not a
            # path that should ever be reached.
            if not np.isfinite(fold_score):
                return _failure_score(state), None
            fold_scores.append(fold_score)
        # BUGFIX (joint search still underperforming non-joint alternatives
        # despite the default-vs-winner margin below): this candidate's own
        # score UNCERTAINTY across the `effective_repeats` inner folds -
        # the standard error of the mean - was computed nowhere before this
        # fix; the payload slot here was always `None` and the caller's
        # safeguard had no uncertainty to consult, only a fixed
        # `selection_margin`. Returned as a dict rather than a bare float
        # so this branch's payload stays distinguishable from the "real
        # confirmatory fit" payload shape used elsewhere in this function
        # (a dict with 'final_params_by_model' etc.) - callers that only
        # want the score (`joint_objective`) already discard this value
        # unconditionally, so this is purely additive (I11).
        arr = np.asarray(fold_scores, dtype=float)
        n = len(arr)
        se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        # ROUND 10: the per-fold scores are returned too (additive) so the
        # fresh-resample gate can form a PAIRED winner-vs-baseline difference
        # (same folds for both) rather than combining two unpaired SEs.
        return float(arr.mean()), {'se': se, 'n_folds': n, 'fold_scores': arr.tolist()}

    if valid is None or valid.shape[0] == 0:
        return _failure_score(state), None

    # ver4-28, R1/R2: the K group-model refits for this ONE evaluation are
    # fit UNITS run through `_run_units()`. `keep_results` (R2) asks every
    # unit to keep its FULL runner dict, not just `predicted_valid`, so the
    # caller can hand a confirmatory fit straight to genomic_prediction.py's
    # own dispatch instead of refitting it a second time.
    _fidelity_tag = 'search' if search_fidelity else 'confirm'
    units_valid = [
        _FitUnit(model=m, params=final_params_by_model[m], split_tag='valid',
                 train=train, valid=valid, fidelity=_fidelity_tag, keep_full=keep_results)
        for m in group_models
    ]
    fitted_valid = _run_units(
        units_valid, runner=run_model_fn, RESULT_NAME=RESULT_NAME, test=test,
        n_jobs=_ctx_n_jobs, worker_init=_ctx_worker_init, worker_init_args=_ctx_worker_init_args,
        cache=_ctx_cache, counter=_ctx_counter, stage=_ctx_stage,
        fit_flags=runner_ctx.get('fit_flags'),
    )
    preds_valid = []
    confirm_results: Optional[dict] = {} if keep_results else None
    for m, result in zip(group_models, fitted_valid):
        if result is None:
            print(f"[joint_tuning] NOTE: {'search' if search_fidelity else 'confirmatory'} "
                  f"candidate evaluation failed refitting {m!r} - "
                  + ("this candidate scores as a failure and is discarded; the search continues."
                     if search_fidelity else
                     "the joint search's own winning candidate could not be reproduced at full "
                     "fidelity for this model."))
            return _failure_score(state), None
        _pv = result.get('predicted_valid')
        if _pv is None or len(_pv) == 0:
            return _failure_score(state), None
        preds_valid.append(np.asarray(_pv, dtype=float))
        if keep_results:
            confirm_results[m] = result

    try:
        pred_matrix = np.column_stack(preds_valid)
    except Exception:
        return _failure_score(state), None
    actual_valid = np.asarray(valid.iloc[:, -1], dtype=float)
    if pred_matrix.shape[0] != actual_valid.shape[0]:
        return _failure_score(state), None

    # Round 6: this candidate's own best-achievable ensemble weights,
    # solved analytically (`ensemble_regularization.analytic_simplex_
    # weights()` - the SAME closed-form, non-negative, sum-to-1
    # least-squares solver the standalone `Analytic least-squares` W_OPT
    # method already uses) from THIS candidate's own realised validation
    # predictions - never a free variable the outer Bayesian search has to
    # locate. Applies identically whether this is a cheapened search-phase
    # evaluation or the one full-fidelity confirmatory evaluation.
    # ver4-34, R2.a: a bagged-estimator SEARCH evaluation never uses `shares` (the score
    # is cross-fitted), so the analytic solve is skipped there; the payload then carries
    # equal weights, which only the confirmatory evaluation's payload is ever read for.
    if _bagged and search_fidelity:
        shares = None
    else:
        shares = analytic_simplex_weights(pred_matrix, actual_valid)

    score = _combine_score(shares, pred_matrix, actual_valid, group_models, objective_mode,
                            diversity_penalty, diversity_method, state=state,
                            oof_alpha=weight_cv_alpha, oof_seed=inner_split_seed_base,
                            weight_rule=weight_rule, shrinkage_alpha=shrinkage_alpha,
                            weight_estimator=weight_estimator,
                            estimator_settings=estimator_settings)
    # `_combine_score()` itself can no longer return a non-finite value at
    # all as of this fix (see its own BUGFIX comment) - this check is a
    # pure defensive backstop, not a path that should ever be reached.
    if not np.isfinite(score):
        return _failure_score(state), None

    _payload_shares = (shares if shares is not None
                       else np.full(len(group_models), 1.0 / len(group_models)))
    weights = {m: float(w) for m, w in zip(group_models, _payload_shares.tolist())}
    tuned_values_by_model = {
        m: {s.name: final_params_by_model[m][s.index] for s in specs_by_model[m]}
        for m in group_models
    }

    return float(score), {
        'final_params_by_model': final_params_by_model,
        'weights': weights,
        'tuned_values_by_model': tuned_values_by_model,
        'pred_matrix': pred_matrix,
        'actual_valid': actual_valid,
        'confirm_results': confirm_results,
    }


def joint_objective(x, *, group_models, base_params_by_model, specs_by_model,
                     train, valid, test, run_model_fn, RESULT_NAME,
                     objective_mode='ensemble_score', search_fidelity=True,
                     diversity_penalty=0.0, diversity_method='ridge_to_uniform',
                     valid_repeats=1, inner_split_seed_base=0, state=None,
                     weight_cv_alpha=None, runner_ctx: Optional[dict] = None,
                     weight_rule: str = 'analytic_penalised',
                     shrinkage_alpha: float = 1.0, weight_estimator: str = 'legacy',
                     estimator_settings: Optional[dict] = None,
                     objective_data: str = 'valid',
                     cv_folds: int = _JOINT_CV_FOLDS_DEFAULT) -> float:
    """Thin, score-only wrapper around `_score_candidate()` - the shape
    `models/parallel_search.py::batch_bayesian_maximise()` and
    `bayes_opt.BayesianOptimization` both require for an objective
    function (a single float, no side payload). The full per-model
    hyperparameters/weights a candidate produced are recovered separately,
    ONCE, by re-evaluating the search's own winning `x` at
    `search_fidelity=False` (see `tune_joint_hyperparameters_and_weights()`
    below) - never tracked through this function's own return value, so
    this function's signature can stay exactly what both search back-ends
    expect. `valid_repeats`/`inner_split_seed_base` (ver4-27, R2) are
    forwarded straight through to `_score_candidate()`, which is the only
    place that decides whether/how they apply (never for a
    `search_fidelity=False` confirmatory call - see that function's own
    comment). `state` (BUGFIX, batch joint search reverting to serial) is
    forwarded straight through to `_score_candidate()`/`_combine_score()` so
    this candidate's own score - success or failure - is tracked against
    the SAME shared worst-finite-score floor every other candidate this
    search evaluates uses; a failing/degenerate candidate can therefore
    never come back as the literal, GP-surrogate-breaking `-inf` this
    function used to be able to return - see `_failure_score()`'s own
    docstring for the full root-cause explanation.

    ver4-34: `weight_estimator`/`estimator_settings` (R2.a) and
    `objective_data`/`cv_folds` (R2.b) are forwarded unchanged to
    `_score_candidate()`, which documents them; the defaults are the ver4-32
    behaviour."""
    score, _ = _score_candidate(
        x, group_models=group_models, base_params_by_model=base_params_by_model,
        specs_by_model=specs_by_model, train=train, valid=valid, test=test,
        run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME,
        objective_mode=objective_mode, search_fidelity=search_fidelity,
        diversity_penalty=diversity_penalty, diversity_method=diversity_method,
        valid_repeats=valid_repeats, inner_split_seed_base=inner_split_seed_base,
        state=state, weight_cv_alpha=weight_cv_alpha, runner_ctx=runner_ctx,
        weight_rule=weight_rule, shrinkage_alpha=shrinkage_alpha,
        weight_estimator=weight_estimator, estimator_settings=estimator_settings,
        objective_data=objective_data, cv_folds=cv_folds,
    )
    return score


def _patience_stop(patience: int, tol: float = _JOINT_PATIENCE_TOL
                   ) -> Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]:
    """ver4-32, R2: build the `should_stop` callable for
    `parallel_search.batch_bayesian_maximise()`.

    Why: neither joint search had a stopping rule, so a search whose score had
    plateaued still spent its whole budget (each evaluation refits up to K
    models). The rule is deliberately simple: stop once at least `patience`
    evaluations exist and the best of the LAST `patience` scores does not
    beat the best of all EARLIER scores by more than `tol`.

    Args: `patience` evaluations without improvement (<= 0 disables the rule
    and returns None, which `batch_bayesian_maximise()` treats as "never
    stop"); `tol` minimum improvement that counts (default
    `_JOINT_PATIENCE_TOL`). The callable receives the `(x, score)` history of
    the current search call, in evaluation order, and returns True to stop.
    It is checked once per round, so a search can overshoot by up to q-1
    evaluations. Never raises for a well-formed history."""
    # ver4-40, R1: the body moved to `parallel_search.patience_stop` (single authority,
    # I3) because single-model Bayesian tuning now uses the same rule. Same arguments,
    # same returned callable, same numbers.
    return patience_stop(patience, tol)


def _fit_requests_batch(requests: Sequence[Tuple[str, np.ndarray]], *, ctx: dict
                         ) -> List[Optional[np.ndarray]]:
    """ver4-34, R1.c/R2.b: fit EVERY `(model, coordinate chunk)` request - possibly for
    several different models - as ONE `_run_units()` batch and return one prediction
    column per request (`None` for a failed one), in `requests` order.

    Why: the parallel-library search proposes candidates for all models in the same
    round and must hand them to the pool together (`_fit_columns_batch()` is the
    single-model special case, and the block search still uses it).
    `stop_on_failure=False` because the requests are independent.

    The column is what the objective scores: with `ctx['objective_data'] ==
    'valid'` (default) it is the model's prediction of the validation rows after a fit
    on train; with 'cv_train_valid' it is the out-of-fold vector over train+valid
    (`_cv_fit_columns()`, `ctx['tv']`, `ctx['partition']`, `ctx['cv_seed']`).

    `ctx` keys: group_models, specs_by_model, base_params_by_model, train, valid, test,
    runner, RESULT_NAME and optionally runner_ctx (n_jobs, worker_init,
    worker_init_args, cache, counter, stage), objective_data, tv, partition, cv_seed.
    Raises whatever `_decode_model_params()` raises for a malformed chunk; a fit failure
    is reported as `None`, never raised."""
    requests = list(requests)
    rc = ctx.get('runner_ctx') or {}
    cv_mode = ctx.get('objective_data', _JOINT_OBJECTIVE_DATA_DEFAULT) == 'cv_train_valid'
    valid = ctx['valid']
    if not cv_mode and (valid is None or len(valid) == 0):
        return [None] * len(requests)
    decoded = []
    for m, chunk in requests:
        decoded.append((m, _decode_model_params(
            m, chunk, specs=ctx['specs_by_model'][m], base_params=ctx['base_params_by_model'][m],
            search_fidelity=True)))
    if cv_mode:
        return _cv_fit_columns(
            decoded, tv=ctx['tv'], partition=ctx['partition'], seed=ctx['cv_seed'],
            test=ctx['test'], runner=ctx['runner'], RESULT_NAME=ctx['RESULT_NAME'],
            runner_ctx=rc, stop_on_failure=False)
    units = [_FitUnit(model=m, params=params, split_tag='valid', train=ctx['train'],
                      valid=valid, fidelity='search') for m, params in decoded]
    fitted = _run_units(
        units, runner=ctx['runner'], RESULT_NAME=ctx['RESULT_NAME'], test=ctx['test'],
        n_jobs=rc.get('n_jobs', 1), worker_init=rc.get('worker_init'),
        worker_init_args=rc.get('worker_init_args', ()), cache=rc.get('cache'),
        stop_on_failure=False, counter=rc.get('counter'), stage=rc.get('stage'),
        fit_flags=rc.get('fit_flags'),
    )
    columns: List[Optional[np.ndarray]] = []
    sink = rc.get('test_sink')
    for (_m, _params), result in zip(decoded, fitted):
        pv = None if result is None else result.get('predicted_valid')
        if pv is None or len(pv) == 0 or len(pv) != len(valid):
            columns.append(None)
        else:
            columns.append(np.asarray(pv, dtype=float))
            if sink is not None:
                _record_test_columns(sink, (_m, _params), [result])
    return columns


def _fit_columns_batch(m: str, chunks: Sequence, *, ctx: dict) -> List[Optional[np.ndarray]]:
    """ver4-32, R2: fit model `m` ALONE at each coordinate chunk in `chunks`
    and return its prediction column per chunk (`None` for a chunk whose fit
    failed), in `chunks` order.

    Why: the block search used to fit its q candidates one at a time, each
    through a single-unit `_run_units()` call, so `parallel_trials` never had
    more than one unit to fan out. All q single-model `_FitUnit`s are now
    handed to ONE `_run_units(..., stop_on_failure=False)` call
    (`stop_on_failure=False` because the candidates are independent). With
    `n_jobs=1` that is the same fits in the same order, and the columns are
    identical to what `_score_candidate()` produced for model m before.

    ver4-34: this is `_fit_requests_batch()` for one model; with
    `ctx['objective_data'] == 'cv_train_valid'` the columns are out-of-fold vectors
    over train+valid instead of validation predictions (see that function). `ctx`
    keys as there. Raises whatever `_decode_model_params()` raises for a malformed
    chunk; a fit failure is reported as `None` (the worker already logged it),
    never raised."""
    return _fit_requests_batch([(m, chunk) for chunk in chunks], ctx=ctx)


def _make_column_context(*, group_models, specs_by_model, base_params_by_model, train, valid,
                          test, runner, RESULT_NAME, runner_ctx: Optional[dict],
                          objective_data: str = _JOINT_OBJECTIVE_DATA_DEFAULT,
                          cv_folds: int = _JOINT_CV_FOLDS_DEFAULT,
                          inner_split_seed_base: int = 0) -> dict:
    """ver4-34, R2.b: the one `ctx` dict the column searches (`_fit_requests_batch()`,
    `_block_coordinate_search()`, `_parallel_library_search()`) share, plus the target
    vector `actual` that their columns are scored against.

    Why: both searches used to read the targets straight from `valid`; with the CV
    objective the columns are out-of-fold vectors over train+valid, so the target
    (`y_tv`), the partition and the frame must travel with the fit context. Building it
    here keeps that in one place (I3).

    `objective_data='valid'` (default): `actual` is the last column of `valid`, no other
    keys are added - exactly the pre-ver4-34 context. 'cv_train_valid': adds `tv`,
    `partition` (`_cv_partition()` with seed `cv_seed = inner_split_seed_base +
    _JOINT_CV_SEED_OFFSET`), `cv_seed`, `cv_folds`, and `actual = y_tv`.
    Returns the dict; raises `AssertionError` from `_cv_frame()` on a row-count
    mismatch."""
    ctx = {
        'group_models': group_models, 'specs_by_model': specs_by_model,
        'base_params_by_model': base_params_by_model, 'train': train, 'valid': valid,
        'test': test, 'runner': runner, 'RESULT_NAME': RESULT_NAME, 'runner_ctx': runner_ctx,
        'objective_data': objective_data,
    }
    if objective_data == 'cv_train_valid':
        cv_seed = int(inner_split_seed_base) + _JOINT_CV_SEED_OFFSET
        tv = _cv_frame(train, valid)
        ctx.update({
            'tv': tv, 'partition': _cv_partition(len(tv), cv_folds, cv_seed),
            'cv_seed': cv_seed, 'cv_folds': int(cv_folds),
            'actual': np.asarray(tv.iloc[:, -1], dtype=float),
        })
    else:
        ctx['actual'] = np.asarray(valid.iloc[:, -1], dtype=float)
    return ctx


class _ColumnScorer:
    """ver4-34, R1.c/R2.b: score a `{model: prediction column}` dict through the SAME
    `_combine_score()` every other evaluation in this module uses.

    Why a class: the block search and the library search both re-score fixed sets of
    already-fitted columns thousands of times at zero fits, and each needs the same
    settings (objective, penalty, weight rule, estimator, partition seed). They used to
    be closed over inside `_block_coordinate_search()`; a shared object keeps the two
    searches on one definition of "the score of these columns" (I3).

    `settings` keys (all optional): objective_mode, diversity_penalty, diversity_method,
    weight_cv_alpha, inner_split_seed_base, weight_rule, shrinkage_alpha,
    weight_estimator, estimator_settings. `ctx` supplies `actual` and, for the CV
    objective, `partition`/`cv_seed` (the cross-fit partition then equals the fit
    partition). `state` is the search's shared `_failure_score` state.

    `score(pred_by_model)` returns the ensemble score; `score_with_folds()` also returns
    the per-fold scores (or `None`). Neither raises.

    ver4-38: `settings['dead_column_guard']` (default False = ver4-37) makes a candidate with
    a dead column (`_is_dead_column()`) score as `_failure_score(state)` instead of its
    shrink-to-the-mean ensemble value, and `drop_dead_seed_models()` lets a search take a
    model whose UNTUNED column is dead out of the scored group (`inactive`); `matrix()` and
    `_evaluate()` then use `active_models` only. `score_rows()` scores a row subset (CV
    racing). With the guard off and `inactive` empty every method equals its ver4-37 self."""

    def __init__(self, group_models: Sequence[str], ctx: dict, settings: dict, state):
        self.group_models = list(group_models)
        self.actual = np.asarray(ctx['actual'], dtype=float)
        self.state = state
        self.dead_guard = bool(settings.get('dead_column_guard', _JOINT_DEAD_COLUMN_GUARD_DEFAULT))
        self.inactive: set = set()
        self.partition = (list(ctx['partition']) if ctx.get('objective_data') == 'cv_train_valid'
                          and ctx.get('partition') is not None else None)
        self.objective_mode = settings.get('objective_mode', 'ensemble_score')
        self.diversity_penalty = settings.get('diversity_penalty', 0.0)
        self.diversity_method = settings.get('diversity_method', 'ridge_to_uniform')
        self.weight_cv_alpha = settings.get('weight_cv_alpha')
        self.weight_rule = settings.get('weight_rule', _JOINT_SCORE_WEIGHT_RULE_DEFAULT)
        self.shrinkage_alpha = settings.get('shrinkage_alpha', _JOINT_SHRINKAGE_ALPHA_DEFAULT)
        self.weight_estimator = settings.get('weight_estimator', _JOINT_WEIGHT_ESTIMATOR_DEFAULT)
        est = dict(settings.get('estimator_settings') or {})
        if ctx.get('objective_data') == 'cv_train_valid':
            self.oof_seed = int(ctx['cv_seed'])
            est['n_folds'] = len(ctx['partition'])
            est['folds'] = ctx['partition']
        else:
            self.oof_seed = int(settings.get('inner_split_seed_base', 0))
        self.estimator_settings = est
        self.bagged = (is_robust_estimator(self.weight_estimator)
                       and self.objective_mode != 'dpt_ratio')

    @property
    def active_models(self) -> List[str]:
        """ver4-38: the models that are scored - `group_models` minus `inactive` (a model whose
        untuned column was dead, see `drop_dead_seed_models()`)."""
        return [m for m in self.group_models if m not in self.inactive]

    def matrix(self, pred_by_model: Dict[str, np.ndarray]) -> np.ndarray:
        """`(N, K_active)` prediction matrix in `group_models` order (inactive models left out;
        with none inactive this is the ver4-37 `(N, K)` matrix)."""
        return np.column_stack([pred_by_model[m] for m in self.active_models])

    def drop_dead_seed_models(self, cols_by_model: Dict[str, np.ndarray], *,
                              where: str = 'search') -> List[str]:
        """ver4-38, R2.b1: with `dead_column_guard`, take every model whose UNTUNED column in
        `cols_by_model` is dead out of the scored group and return their names (empty when the
        guard is off or nothing is dead). Why: a dead reference column would make every
        candidate fail (the guard scores any dead column as a failure) or, unguarded, let the
        ensemble use it as a shrinkage knob; neither tells the search anything about the
        OTHER models. The search then runs on K-1 columns and the caller reports weight 0
        for the dropped model. Logs a WARN naming the model. If EVERY model is dead nothing is
        dropped (a NOTE says so) - there would be no ensemble left to tune. Never raises."""
        if not self.dead_guard:
            return []
        dead = [m for m in self.active_models
                if _is_dead_column(cols_by_model[m], self.actual, self.partition)]
        if not dead:
            return []
        if len(dead) >= len(self.active_models):
            print(f"[GP] NOTE: Joint Bayesian optimisation dead_column_guard: the untuned "
                  f"column of EVERY model {dead!r} is constant ({where}) - nothing can be "
                  f"dropped, so the search runs unguarded for this task.")
            return []
        self.inactive.update(dead)
        print(f"[GP] WARN: Joint Bayesian optimisation dead_column_guard: the untuned column of "
              f"{dead!r} is CONSTANT (no genetic signal; for SVR usually epsilon larger than the "
              f"trait SD - see 'Scale epsilon and C by the trait SD') - {dead!r} leaves the "
              f"ensemble for this task ({where}); the joint row uses the remaining "
              f"{len(self.active_models)} model(s) and its weight is 0.")
        return dead

    def score_rows(self, pred_by_model: Dict[str, np.ndarray], rows,
                   sub_partition: Sequence[np.ndarray]) -> float:
        """ver4-38, R1.b: the ensemble score of `pred_by_model` restricted to the scoring rows
        `rows` (positional, any order) - the same estimator, objective and failure state as
        `score()`, applied to the sub-matrix. Used by CV racing to compare a proposal that has
        only been fitted on some folds with the incumbent on EXACTLY those rows.

        Args: `rows` positions into the scoring vector (every column must be finite there);
        `sub_partition` the raced folds' held-out positions; they only set the number of
        cross-fit folds (the cross-fit partition itself is drawn by `cross_fitted_combination`
        from `oof_seed`, deterministic). Returns a float; a non-finite or too-small sub-matrix
        scores as `_failure_score(state)`. Never raises."""
        rows = np.asarray(rows, dtype=int)
        models = self.active_models
        try:
            P = np.column_stack([np.asarray(pred_by_model[m], dtype=float)[rows] for m in models])
        except (IndexError, ValueError):
            return _failure_score(self.state)
        y = self.actual[rows]
        if rows.size < 3 or not np.all(np.isfinite(P)):
            return _failure_score(self.state)
        if self.dead_guard and any(_is_dead_column(P[:, i], self.actual)
                                   for i in range(P.shape[1])):
            return _failure_score(self.state)
        est = dict(self.estimator_settings)
        est['n_folds'] = int(max(2, len(sub_partition)))
        est['folds'] = list(sub_partition)
        shares = None if self.bagged else analytic_simplex_weights(P, y)
        return _combine_score(
            shares, P, y, models, self.objective_mode, self.diversity_penalty,
            self.diversity_method, state=self.state, oof_alpha=self.weight_cv_alpha,
            oof_seed=self.oof_seed, weight_rule=self.weight_rule,
            shrinkage_alpha=self.shrinkage_alpha, weight_estimator=self.weight_estimator,
            estimator_settings=est, return_folds=False)

    def _evaluate(self, pred_by_model: Dict[str, np.ndarray], return_folds: bool):
        models = self.active_models
        if self.dead_guard and any(_is_dead_column(pred_by_model[m], self.actual, self.partition)
                                   for m in models):
            failed = _failure_score(self.state)
            return (failed, None) if return_folds else failed
        pred_matrix = self.matrix(pred_by_model)
        shares = None if self.bagged else analytic_simplex_weights(pred_matrix, self.actual)
        return _combine_score(
            shares, pred_matrix, self.actual, models, self.objective_mode,
            self.diversity_penalty, self.diversity_method, state=self.state,
            oof_alpha=self.weight_cv_alpha, oof_seed=self.oof_seed,
            weight_rule=self.weight_rule, shrinkage_alpha=self.shrinkage_alpha,
            weight_estimator=self.weight_estimator, estimator_settings=self.estimator_settings,
            return_folds=return_folds)

    def clone(self, state) -> '_ColumnScorer':
        """ver4-36, R1.e: a shallow copy that reports failures against its OWN `state`
        dict. The asynchronous library search gives every model's pipeline one, so a
        degenerate candidate's failure score never depends on which OTHER model's scores
        happened to be computed first (completion order), keeping the search deterministic."""
        other = copy.copy(self)
        other.state = state
        other.inactive = set(self.inactive)
        return other

    def score_with_folds(self, pred_by_model: Dict[str, np.ndarray]
                         ) -> Tuple[float, Optional[List[float]]]:
        return self._evaluate(pred_by_model, True)

    def score(self, pred_by_model: Dict[str, np.ndarray]) -> float:
        return self._evaluate(pred_by_model, False)

    __call__ = score


class _FixedWeightScorer:
    """ver4-40, R2: score a `{model: prediction column}` dict at a FIXED weight vector `w`.

    Why: the weight-first search (and its zero-fit replay in `tools/joint_replay.py`) fixes
    a small set of weight vectors first and only then asks which library column each model
    should contribute. With `w` fixed the joint score is a plain linear functional of the
    columns - no weight is re-estimated on 50-150 scoring rows, which is where the
    winner's-curse of the monolithic search comes from. Scoring goes through
    `_combine_score()` (I3: one definition of "the score of these columns"):
    `shares=w`, `weight_rule='shrunk'` with `shrinkage_alpha=1.0` (a strict no-op shrink,
    so `w` is used exactly as given and the diversity penalty is not applied), which for the
    default 'ensemble_score' objective equals `_ensemble_score_of(P @ w, y)` exactly
    (verified by pre-check P3). The cross-validated objective needs no cross-fitting here:
    the columns are already out-of-fold and `w` is fixed.

    `w` has one entry per `group_models` entry (a model with weight 0 still has a column
    but cannot influence the score). `dead_column_guard` (settings key, default
    `_JOINT_DEAD_COLUMN_GUARD_DEFAULT`) makes a candidate whose column is dead
    (`_is_dead_column`) fail ONLY when that model carries weight > 0 - a dead model that
    `weight_candidates(dead=...)` already zeroed must not fail every candidate.

    `score(pred_by_model)` returns a float and never raises; `clone(state)` returns a copy
    with its own failure state (the same contract as `_ColumnScorer`, so the asynchronous
    search stays deterministic)."""

    def __init__(self, group_models: Sequence[str], w, ctx: dict, settings: dict, state):
        self.group_models = list(group_models)
        self.w = np.asarray(w, dtype=float)
        if self.w.shape != (len(self.group_models),):
            raise ValueError(
                f"_FixedWeightScorer: {self.w.size} weight(s) for {len(self.group_models)} "
                f"model(s) {self.group_models!r}.")
        self.actual = np.asarray(ctx['actual'], dtype=float)
        self.state = state
        self.dead_guard = bool(settings.get('dead_column_guard', _JOINT_DEAD_COLUMN_GUARD_DEFAULT))
        self.partition = (list(ctx['partition']) if ctx.get('objective_data') == 'cv_train_valid'
                          and ctx.get('partition') is not None else None)
        self.objective_mode = settings.get('objective_mode', 'ensemble_score')

    def matrix(self, pred_by_model: Dict[str, np.ndarray]) -> np.ndarray:
        """`(N, K)` prediction matrix in `group_models` order."""
        return np.column_stack([pred_by_model[m] for m in self.group_models])

    def score(self, pred_by_model: Dict[str, np.ndarray]) -> float:
        if self.dead_guard and any(
                self.w[i] > 0 and _is_dead_column(pred_by_model[m], self.actual, self.partition)
                for i, m in enumerate(self.group_models)):
            return _failure_score(self.state)
        return _combine_score(
            self.w, self.matrix(pred_by_model), self.actual, self.group_models,
            self.objective_mode, 0.0, 'ridge_to_uniform', state=self.state,
            weight_rule='shrunk', shrinkage_alpha=1.0)

    __call__ = score

    def clone(self, state) -> '_FixedWeightScorer':
        """A shallow copy that reports failures against its OWN `state` dict."""
        other = copy.copy(self)
        other.state = state
        return other


class _FixedWeightLibraryScorer(_FixedWeightScorer):
    """ver4-42, R1: the fixed-weight scorer, given the surface `_async_library_search()` expects
    of a `_ColumnScorer` (stage S2 of `search_strategy='weight_first_cv'`).

    Why a subclass: the async library search tunes every model against FIXED reference columns
    and asks its scorer for `score`, `__call__`, `clone`, `score_rows`, `inactive`,
    `active_models`, `drop_dead_seed_models` and `bagged` (pre-check P4). With the weights fixed
    by stage S1 none of these has to estimate anything, so each one is a thin wrapper around the
    ver4-40 `_FixedWeightScorer.score()` (I3: one definition of "the score of these columns").

    IMPORTANT - `inactive` means something different here than in `_ColumnScorer`: there it is
    "left out of the score", here it is only "NOT TUNED" (its weight is below
    `wf_min_tune_weight`, or it is dead). `score()` always uses all K entries of the fixed
    vector, and a model in `inactive` keeps its weight. Do not reuse `_ColumnScorer.matrix()`
    with this class.

    Args: `group_models`, `w` (K floats, `w >= 0`, sum 1; dead models 0), `ctx`, `settings`,
    `state` as in `_FixedWeightScorer`; `inactive` the models that are not tuned; `dead_models`
    the models whose untuned column is constant (weight 0, reported through
    `drop_dead_seed_models`). Never raises for a well-formed call."""

    bagged = False

    def __init__(self, group_models: Sequence[str], w, ctx: dict, settings: dict, state, *,
                 inactive: Sequence[str] = (), dead_models: Sequence[str] = ()):
        super().__init__(group_models, w, ctx, settings, state)
        self.fixed_w = self.w
        self.inactive: set = set(inactive)
        self.dead_models: List[str] = list(dead_models)

    @property
    def active_models(self) -> List[str]:
        """Every model with a positive weight, in `group_models` order. Used for the visiting
        order and for reporting only - the score itself always uses all K entries of `w`."""
        return [m for i, m in enumerate(self.group_models) if self.w[i] > 0]

    def drop_dead_seed_models(self, cols_by_model: Dict[str, np.ndarray], *,
                              where: str = 'search') -> List[str]:
        """The models stage S1 found dead (constant untuned column; weight 0 and not tuned).
        Nothing is dropped here - the weights were fixed before the search started."""
        return list(self.dead_models)

    def score_rows(self, pred_by_model: Dict[str, np.ndarray], rows,
                   sub_partition: Sequence[np.ndarray]) -> float:
        """The score of `pred_by_model` restricted to the scoring rows `rows` (positional), at
        the same fixed weights - used by CV racing. A too-small or non-finite sub-matrix scores
        as `_failure_score(state)`. Never raises."""
        rows = np.asarray(rows, dtype=int)
        try:
            P = np.column_stack([np.asarray(pred_by_model[m], dtype=float)[rows]
                                 for m in self.group_models])
        except (IndexError, ValueError):
            return _failure_score(self.state)
        if rows.size < 3 or not np.all(np.isfinite(P)):
            return _failure_score(self.state)
        y = self.actual[rows]
        if self.dead_guard and any(self.w[i] > 0 and _is_dead_column(P[:, i], self.actual)
                                   for i in range(P.shape[1])):
            return _failure_score(self.state)
        return _combine_score(
            self.w, P, y, self.group_models, self.objective_mode, 0.0, 'ridge_to_uniform',
            state=self.state, weight_rule='shrunk', shrinkage_alpha=1.0)

    def clone(self, state) -> '_FixedWeightLibraryScorer':
        """A shallow copy with its OWN failure `state` and its own `inactive` set."""
        other = copy.copy(self)
        other.state = state
        other.inactive = set(self.inactive)
        return other


def _time_budget_stop(t0: float, budget_s: float
                      ) -> Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]:
    """ver4-34, R1.d: build a `should_stop` callable that turns True once `budget_s`
    seconds have passed since `t0` (a `time.time()` value taken at the start of Step 8.5).

    Why: users on shared HPC need a hard bound on the search. The callable is checked
    once per round by `batch_bayesian_maximise()` (and before every block / library
    round by the strategy loops), so a search can overshoot by at most the longest
    single batch (A-1d.1). Gate and confirmatory fits always run afterwards - that is
    disclosed in the settings log line.

    Args: `t0` start time; `budget_s` seconds (<= 0 disables the rule and returns
    None, which `batch_bayesian_maximise()` treats as "never stop"). The returned
    callable ignores its history argument. Never raises."""
    budget_s = float(budget_s or 0)
    if budget_s <= 0:
        return None

    def _should_stop(_history) -> bool:
        return (time.time() - float(t0)) >= budget_s

    return _should_stop


def _compose_stops(*fns) -> Optional[Callable[[List[Tuple[np.ndarray, float]]], bool]]:
    """ver4-34, R1.d: combine several `should_stop` callables (or `None`) into one that
    is True when ANY of them is. Returns `None` when every argument is `None`, so the
    result can be handed to `batch_bayesian_maximise(should_stop=...)` unchanged."""
    active = [fn for fn in fns if fn is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _should_stop(history) -> bool:
        return any(fn(history) for fn in active)

    return _should_stop


def _frozen_models(pred_by_model: Dict[str, np.ndarray], group_models: Sequence[str],
                   actual: np.ndarray, *, low: float, n_bags: int, cluster_corr: float,
                   seed: int, estimator: str = 'bagged_cluster',
                   estimator_settings: Optional[dict] = None) -> Dict[str, float]:
    """ver4-34, R1.d: the models whose TUNING should stop - those whose 90th-percentile
    bagged weight is below `low` - as `{model: P90 weight}`.

    Why: a model whose weight is ~0 in (almost) every bootstrap bag cannot change the
    ensemble whatever its hyperparameters are, so proposals for it are wasted fits. The
    P90 (not the mean) is used so a model that is important in a tenth of the bags keeps
    being tuned. A frozen model STAYS in the ensemble at its current column; the caller
    recomputes this after every round/cycle, so a model whose P90 rises above `low` is
    unfrozen automatically. The model with the highest P90 is never frozen (a
    `low` above 1/K must not freeze everyone).

    Args: `pred_by_model` current column per model; `actual` target vector; `low`
    threshold (<= 0 returns `{}`); `n_bags`/`cluster_corr`/`seed` forwarded to the full
    function of `estimator` (ver4-36: any key of `ROBUST_WEIGHT_ESTIMATORS`, default
    'bagged_cluster' = `robust_simplex_weights()`, whose diagnostics carry `bag_weights`;
    `estimator_settings` supplies the ES keywords). Returns `{}` when the bag weights cannot
    be formed. Never raises."""
    if low is None or float(low) <= 0.0 or len(group_models) < 2:
        return {}
    pred_matrix = np.column_stack([pred_by_model[m] for m in group_models])
    try:
        _settings = dict(estimator_settings or {})
        _settings['cluster_corr'] = cluster_corr
        _w, diag = _robust_weights_full(estimator, pred_matrix, actual, _settings,
                                        n_bags=n_bags, seed=seed)
        bag = np.asarray(diag.get('bag_weights'), dtype=float)
    except Exception as exc:
        print(f"[joint_tuning] NOTE: freeze_low_weight could not compute bagged weights "
              f"({type(exc).__name__}: {exc}) - no model is frozen this round.")
        return {}
    if bag.ndim != 2 or bag.shape[1] != len(group_models) or not np.all(np.isfinite(bag)):
        return {}
    p90 = np.percentile(bag, 90, axis=0)
    keep = int(np.argmax(p90))
    return {m: float(p90[i]) for i, m in enumerate(group_models)
            if i != keep and p90[i] < float(low)}


def _coordinate_select(lib: Dict[str, list], cur: Dict[str, int],
                       score_fn: Callable[[Dict[str, np.ndarray]], float],
                       order: Sequence[str], max_passes: int = _JOINT_LIBRARY_MAX_PASSES,
                       tol: float = _JOINT_BLOCK_TOL_DEFAULT) -> Tuple[Dict[str, int], float, int]:
    """ver4-34, R1.c: zero-fit greedy selection of one library column per model.

    Why: with analytic/bagged weights the ensemble score is a pure function of the K
    prediction columns (Levesque 2016), so which already-fitted column each model
    should use can be decided WITHOUT any fit: swap one model's column for another
    library entry, re-score, keep the swap if the score rises by more than `tol`.

    Args: `lib[m]` list of `(x_m, column)` entries; `cur[m]` index of the model's current
    entry; `score_fn({model: column}) -> float`; `order` the models in the order they are
    visited (the caller passes descending current weight); `max_passes` greedy passes
    (stops early when a pass changes nothing); `tol` the minimum improvement that counts.
    Returns `(cur, best_score, swaps)` - a NEW dict (the input is not mutated), the
    score of that selection and the number of swaps made. Raises KeyError for a model
    missing from `lib`/`cur`."""
    cur = dict(cur)
    best = score_fn({m: lib[m][cur[m]][1] for m in order})
    swaps = 0
    for _ in range(int(max_passes)):
        moved = False
        for m in order:
            for j in range(len(lib[m])):
                if j == cur[m]:
                    continue
                trial = dict(cur)
                trial[m] = j
                s = score_fn({k: lib[k][trial[k]][1] for k in order})
                if s > best + tol:
                    best, cur, moved = s, trial, True
                    swaps += 1
        if not moved:
            break
    return cur, float(best), swaps


def _paired_fresh_gate(candidates: List[Tuple[str, np.ndarray]], seed_point, *, repeats: int,
                        seed: int, margin: float, score_kwargs: dict,
                        runner_ctx: Optional[dict] = None, cache: Optional['_FitCache'] = None
                        ) -> Tuple[Optional[str], Optional[np.ndarray], float, dict]:
    """ver4-32, R3/R5: the fresh-resample gate for one OR SEVERAL candidates.

    Every candidate and the untuned defaults (`seed_point`, scored at exactly
    the configured hyperparameters) are scored on the SAME `repeats` fresh
    inner resamples (seeds `seed + b`) that the search never saw, so each
    comparison is PAIRED. A candidate passes when its mean per-fold gap to the
    defaults exceeds `max(margin, paired SE)`; among those that pass the
    largest gap wins, and a tie goes to the EARLIER entry of `candidates`
    (the caller lists the final candidate first). With one candidate, no
    cache and the caller's `seed`, this reproduces the ver4-31 gate exactly:
    same fits, same order (candidate, then defaults), same numbers.

    `cache` (a gate-only `_FitCache`, opt-in `gate_fit_cache`) lets fits whose
    decoded settings are identical for a candidate and the defaults run once.
    For that to hit, a model whose coordinates equal the defaults' is fitted
    at its exact configured values (`use_base_params_for`) - otherwise
    `_decode(_encode(base))` rounding could make the settings differ by a
    hair and defeat the cache. Fold labels carry the resample seed
    (`fold{b}@{seed+b}`), so gate entries never collide with search folds.

    Args: `candidates` list of `(label, x)`; `seed_point` the defaults' point;
    `repeats`/`seed` fresh-resample count and base seed; `margin` the minimum
    gap (`selection_margin`); `score_kwargs` every `_score_candidate()` keyword
    that is common to all evaluations (group_models, specs, data, runner,
    objective and weight-rule settings - NOT valid_repeats,
    inner_split_seed_base, use_base_params, use_base_params_for, runner_ctx);
    `runner_ctx` fit settings (stage 'gate'); `cache` optional gate cache.

    Returns `(label, x, mean_score, info)`. `label` is a candidate label, or
    'seed' when none passed (then `x` is `seed_point`). If the paired
    comparison cannot be formed (defaults or every candidate lack >= 2 fold
    scores, e.g. a fit failed or the training split is too small) the result is
    `(None, None, nan, info)` with `info['ok'] == False` and the caller falls
    back to the legacy comparison. `info['candidates'][label]` holds
    `valid, mean, gap, se, required, accepted, n_folds`; `info['seed_mean']`
    the defaults' mean. Never raises beyond `_score_candidate()`'s own
    AssertionError for a malformed search space."""
    group_models = score_kwargs['group_models']
    specs_by_model = score_kwargs['specs_by_model']
    seed_point = np.asarray(seed_point, dtype=float)
    spans = {}
    _off = 0
    for m in group_models:
        spans[m] = (_off, _off + len(specs_by_model[m]))
        _off += len(specs_by_model[m])
    ctx = dict(runner_ctx or {})
    ctx['cache'] = cache

    def _eval(x, exact_base):
        untouched = None
        if cache is not None and not exact_base:
            untouched = {m for m in group_models
                         if np.array_equal(x[spans[m][0]:spans[m][1]],
                                           seed_point[spans[m][0]:spans[m][1]])}
        return _score_candidate(
            x, valid_repeats=repeats, inner_split_seed_base=seed, use_base_params=exact_base,
            use_base_params_for=untouched, runner_ctx=ctx, **score_kwargs)

    scored = []
    for label, x in candidates:
        x = np.asarray(x, dtype=float)
        mean, extra = _eval(x, False)
        scored.append((label, x, mean, extra))
    seed_mean, seed_extra = _eval(seed_point, True)

    def _folds(extra):
        if isinstance(extra, dict) and len(extra.get('fold_scores', ())) > 1:
            return np.asarray(extra['fold_scores'], dtype=float)
        return None

    info = {'ok': False, 'seed_mean': float(seed_mean), 'candidates': {}}
    seed_folds = _folds(seed_extra)
    if seed_folds is None:
        return None, None, math.nan, info

    best = None
    for label, x, mean, extra in scored:
        folds = _folds(extra)
        if folds is None or len(folds) != len(seed_folds):
            info['candidates'][label] = {'valid': False}
            continue
        diff = folds - seed_folds
        gap = float(diff.mean())
        se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
        required = max(margin, se)
        accepted = gap > required
        info['candidates'][label] = {'valid': True, 'mean': float(mean), 'gap': gap, 'se': se,
                                     'required': required, 'accepted': accepted,
                                     'n_folds': int(len(diff))}
        if accepted and (best is None or gap > best[0]):
            best = (gap, label, x, float(mean))
    if not any(v['valid'] for v in info['candidates'].values()):
        return None, None, math.nan, info
    info['ok'] = True
    if best is None:
        return 'seed', seed_point, float(seed_mean), info
    return best[1], best[2], best[3], info


def _posterior_gain(hist: Sequence[Tuple[np.ndarray, float]],
                    trials: Sequence[Tuple[np.ndarray, float]], x_new: np.ndarray,
                    x_cur: np.ndarray, seed: int, observed_gain: float) -> float:
    """ver4-36, R2.a: gain of `x_new` over `x_cur` in the noise-aware surrogate's POSTERIOR
    MEAN, from every observation of one block round (`hist` = zero-fit re-scores handed in as
    prior observations, `trials` = the evaluations of the round).

    Why: with `noise_aware_surrogate` the block's incumbent is the point with the highest
    posterior mean; comparing that point's single observed score with the current score
    would still let a lucky draw clear `block_tol`. If the surrogate cannot be fitted (fewer
    than 5 points, non-finite score, fit failure - `posterior_means()` prints one NOTE) or
    one of the two points is not among the observations, the OBSERVED gain is returned
    unchanged. Never raises."""
    xs = [np.asarray(x, dtype=float) for x, _ in hist] + [np.asarray(x, dtype=float)
                                                         for x, _ in trials]
    ys = [float(y) for _, y in hist] + [float(y) for _, y in trials]
    means = posterior_means(xs, ys, seed=seed)
    if means is None:
        return observed_gain
    keys = [x.tobytes() for x in xs]
    try:
        i_new = keys.index(np.asarray(x_new, dtype=float).tobytes())
        i_cur = keys.index(np.asarray(x_cur, dtype=float).tobytes())
    except ValueError:
        return observed_gain
    return float(means[i_new] - means[i_cur])


def _block_coordinate_search(*, group_models, specs_by_model, base_params_by_model, seed_point,
                              train, valid, test, runner, RESULT_NAME, settings, state, cache,
                              n_jobs, worker_init, worker_init_args, seed
                              ) -> Tuple[np.ndarray, float, dict]:
    """ver4-28, R3: coordinate ascent over each group model's own
    hyperparameter block, one block at a time, holding every OTHER
    model's validation-prediction column fixed. With analytic ensemble
    weights (Round 6), the joint score is a pure function of the K
    prediction columns, so re-scoring a fixed column set against a new
    candidate column for model m costs zero extra fits for every other
    model (Levesque et al., 2016, 'Bayesian hyperparameter optimization
    for ensemble learning') - see this module's own top-level R3 design
    note. `cache` (a `_FitCache`) additionally makes re-scoring a
    PREVIOUSLY-TRIED point for model m free too (`_fit_columns_batch()`
    routes through `_run_units()`'s cache).

    `seed_point`/the return value share the SAME concatenated unit-cube
    layout `_score_candidate()` expects elsewhere in this module, so the
    caller's fresh gate, confirmatory fit and `_grade_weights()` are
    reused completely unchanged.

    `settings` is a plain dict carrying `objective_mode`, `diversity_
    penalty`, `diversity_method`, `weight_cv_alpha`, `inner_split_seed_
    base`, `block_cycles`, `block_init_points`, `block_n_iter`, `block_
    refine_iter`, `block_tol` and, optionally, `fit_counter`/`fit_stage`
    (R6 provenance - forwarded into every `_run_units()` call this
    function makes, so a block search's own fits are counted in the same
    per-task `fits:` log line as every other stage). ver4-32 adds the
    optional keys `weight_rule`/`shrinkage_alpha` (R4, forwarded to
    `_combine_score()`), `search_patience` (R2, `_patience_stop()`),
    `warm_start`/`block_warm_n_iter` (R5, below).

    ver4-32 R2: the q candidates of one block round are fitted as ONE batch
    (`_fit_columns_batch()`), so `parallel_trials` now acts in block mode,
    and the per-model seed is `seed + crc32(model) % 10000` (the previous
    `hash(model)` changed between processes - PYTHONHASHSEED - so a block
    search was not reproducible).

    ver4-32 R5 (`settings['warm_start'] == 'individual'`): pass 0 optimises
    each block on the SINGLE-MODEL score (`_combine_score` with one column,
    weight 1, no penalty - identical to the HP_TUNE objective, so no new
    maths), keeping every (x_m, column) pair it fits in `col_hist[m]`. The
    ensemble cycles then start from whichever of {seed, warm} scores higher
    on the ensemble score; before each block's ensemble search the cached
    columns of `col_hist[m]` (the 50 most recent) are re-scored against the
    current other columns and handed over as `prior_observations` - zero extra
    fits. Why: under the ensemble score a gain in model m is scaled by its
    weight, so the per-model signal is weak; the individual pass gives each
    block the full signal first.

    ver4-34: `settings` may also carry `objective_data`/`cv_folds` (R2.b: the columns
    are then out-of-fold vectors over train+valid and the target is y(train+valid) -
    `_make_column_context()`), `weight_estimator`/`estimator_settings` (R2.a, scored
    through `_ColumnScorer`), `freeze_low_weight`/`weight_bags`/`cluster_corr` (R1.d: a
    model whose P90 bagged weight is below the threshold gets no further proposals, it
    keeps its current column; recomputed after the untuned columns and after every
    cycle) and `time_budget_s`/`time_budget_t0` (R1.d: no new block or cycle starts, and
    `batch_bayesian_maximise()` stops after its current round, once the budget has
    elapsed; the best point so far is returned and `stats['time_stopped']` is set).
    `stats['frozen']` is added only when freezing is active.

    Returns `(best_x, best_score, stats)` - `stats` is `{'cycles_run',
    'block_evals', 'blocks_updated', 'warm_x', 'seed_ens', 'warm_ens'}`
    (`warm_x`/`warm_ens` are None without a warm start; `seed_ens` is the
    ensemble score of the untuned columns). Returns `(seed_point, -inf,
    stats)` when an untuned column cannot be fitted."""
    if cache is None:
        cache = _FitCache()
    dims_by_model = {m: len(specs_by_model[m]) for m in group_models}
    offsets = {}
    _off = 0
    for m in group_models:
        offsets[m] = _off
        _off += dims_by_model[m]
    total_dims = _off

    def _chunk(x, m):
        return np.asarray(x, dtype=float)[offsets[m]:offsets[m] + dims_by_model[m]]

    def _assemble(chunks_by_model):
        return np.concatenate([chunks_by_model[m] for m in group_models]) if total_dims else np.array([])

    _runner_ctx = {
        'n_jobs': n_jobs, 'worker_init': worker_init, 'worker_init_args': worker_init_args,
        'cache': cache, 'counter': settings.get('fit_counter'), 'stage': settings.get('fit_stage', 'search'),
        # ver4-36: the per-model shortcut flags and the test-column sink (both None = off)
        'fit_flags': settings.get('fit_flags'), 'test_sink': settings.get('test_sink'),
    }
    # ver4-34, R2.b: the column context also carries the target vector (`actual`), which is
    # the validation targets by default and y(train+valid) under the CV objective.
    _fit_ctx = _make_column_context(
        group_models=group_models, specs_by_model=specs_by_model,
        base_params_by_model=base_params_by_model, train=train, valid=valid, test=test,
        runner=runner, RESULT_NAME=RESULT_NAME, runner_ctx=_runner_ctx,
        objective_data=settings.get('objective_data', _JOINT_OBJECTIVE_DATA_DEFAULT),
        cv_folds=settings.get('cv_folds', _JOINT_CV_FOLDS_DEFAULT),
        inner_split_seed_base=settings.get('inner_split_seed_base', 0))
    actual_valid = _fit_ctx['actual']
    _weight_rule = settings.get('weight_rule', _JOINT_SCORE_WEIGHT_RULE_DEFAULT)
    _shrinkage = settings.get('shrinkage_alpha', _JOINT_SHRINKAGE_ALPHA_DEFAULT)
    scorer = _ColumnScorer(group_models, _fit_ctx, settings, state)

    # ver4-36 R1.b/R2.b: every column this search fits is also recorded, per model and in
    # fitting order (the first entry is the untuned column), so the bootstrap gate and the
    # bootstrap selection can run over the fitted library at zero fits. Recording changes
    # nothing the search does (the fit cache has no coordinates, so it cannot serve this).
    lib_rec = {m: {} for m in group_models}

    def _fit_batch_recorded(m, chunks):
        cols = _fit_columns_batch(m, chunks, ctx=_fit_ctx)
        for x_m, col in zip(chunks, cols):
            if col is not None:
                x_arr = np.array(x_m, dtype=float)
                lib_rec[m].setdefault(x_arr.tobytes(), (x_arr, col))
        return cols

    def _fit_column(m, chunk):
        """Refit model `m` ALONE at `chunk`'s decoded params and return its
        prediction column (validation predictions, or the out-of-fold vector under the
        CV objective), or `None` on failure."""
        return _fit_batch_recorded(m, [chunk])[0]

    def _score_columns(pred_by_model):
        return scorer.score(pred_by_model)

    def _model_seed(m, extra=0):
        return seed + (zlib.crc32(m.encode('utf-8')) % 10_000) + extra

    theta = {m: _chunk(seed_point, m) for m in group_models}
    pred_col = {}
    stats = {'cycles_run': 0, 'block_evals': 0, 'blocks_updated': 0,
             'warm_x': None, 'seed_ens': None, 'warm_ens': None}
    _snap_seed = _counter_snapshot(settings.get('fit_counter'), group_models)
    for m in group_models:
        col = _fit_column(m, theta[m])
        stats['block_evals'] += 1
        if col is None:
            print(f"[GP] NOTE: Joint Bayesian optimisation block-coordinate search could not fit "
                  f"the initial (untuned) column for {m!r} - falling back to the untuned baseline "
                  f"for every model in this search.")
            return seed_point, -math.inf, stats
        pred_col[m] = col

    _log_eta('block-coordinate', group_models, settings.get('eta_plan'),
             _seed_times(settings.get('fit_counter'), _snap_seed, group_models,
                         settings.get('t_seed')), n_jobs)
    # ver4-38, R2.b1: a model whose untuned column is constant leaves the scored group (no-op
    # unless `dead_column_guard`); the cycles below then skip its block.
    stats['dead_models'] = scorer.drop_dead_seed_models(pred_col, where='block-coordinate R0')
    current_score = _score_columns(pred_col)
    stats['seed_ens'] = float(current_score)
    block_tol = float(settings.get('block_tol', _JOINT_BLOCK_TOL_DEFAULT))
    block_cycles = max(1, int(settings.get('block_cycles', _JOINT_BLOCK_CYCLES_DEFAULT)))
    block_init_points = int(settings.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT))
    block_n_iter_config = int(settings.get('block_n_iter', _JOINT_BLOCK_N_ITER_DEFAULT) or 0)
    block_refine_iter = int(settings.get('block_refine_iter', _JOINT_BLOCK_REFINE_ITER_DEFAULT))
    patience = int(settings.get('search_patience', _JOINT_SEARCH_PATIENCE_DEFAULT) or 0)
    warm = settings.get('warm_start', _JOINT_WARM_START_DEFAULT) == 'individual'
    q_m = min(max(2, n_jobs if n_jobs and n_jobs > 1 else _JOINT_BATCH_Q), _JOINT_BATCH_Q)
    # ver4-36 R2.a: noise-aware surrogate + posterior-mean incumbent (default off = ver4-35).
    _noise_aware = bool(settings.get('noise_aware_surrogate', _JOINT_NOISE_AWARE_DEFAULT))
    _incumbent = 'posterior_mean' if _noise_aware else 'max'

    # ver4-34, R1.d: weight-aware freezing and the wall-clock budget. Both default to off,
    # in which case `_stop_hook` is exactly `_patience_stop(patience)` and `frozen` stays
    # empty - the ver4-32 control flow.
    _freeze_low = float(settings.get('freeze_low_weight', _JOINT_FREEZE_LOW_WEIGHT_DEFAULT) or 0.0)
    _budget_s = float(settings.get('time_budget_s', _JOINT_TIME_BUDGET_DEFAULT) or 0.0)
    _budget_stop = (_time_budget_stop(settings['time_budget_t0'], _budget_s)
                    if settings.get('time_budget_t0') is not None else None)
    _stop_hook = _compose_stops(_patience_stop(patience), _budget_stop)

    def _over_budget():
        return _budget_stop is not None and bool(_budget_stop([]))

    def _refresh_frozen(tag):
        """Models whose P90 bagged weight is below `freeze_low_weight` right now (empty
        when freezing is off or the estimator is not the bagged one)."""
        if not (_freeze_low > 0.0 and scorer.bagged):
            return {}
        fz = _frozen_models(
            pred_col, scorer.active_models, actual_valid, low=_freeze_low,
            n_bags=int(settings.get('weight_bags', _JOINT_WEIGHT_BAGS_DEFAULT)),
            cluster_corr=float(settings.get('cluster_corr', _JOINT_CLUSTER_CORR_DEFAULT)),
            seed=int(seed),
            estimator=settings.get('weight_estimator', 'bagged_cluster'),
            estimator_settings=settings.get('estimator_settings'))
        stats.setdefault('frozen', {})[tag] = sorted(fz)
        print(f"[GP] Joint Bayesian optimisation block search ({tag}): frozen models (P90 bagged "
              f"weight < {_freeze_low:g}) = {fz if fz else 'none'}; a frozen model stays in the "
              f"ensemble at its current column, only its tuning stops.")
        return fz

    frozen = _refresh_frozen('after the untuned columns')

    # col_hist[m]: {x bytes: (x_m, column)} for every point fitted for block m
    # (insertion order = chronological). Only kept for a warm start - without
    # it the ensemble cycles seed each block with its current point only,
    # exactly as in ver4-31.
    col_hist = {}
    for m in group_models:
        _x0 = np.array(theta[m], dtype=float)
        col_hist[m] = {_x0.tobytes(): (_x0, pred_col[m])}

    if warm:
        # ---- pass 0: every block on its own single-model score -----------
        indiv_state = {'worst_finite': None}  # own failure-score scale

        def _indiv_score(col):
            return _combine_score(np.array([1.0]), np.asarray(col, dtype=float)[:, None],
                                   actual_valid, ['single'], settings['objective_mode'], 0.0,
                                   settings['diversity_method'], state=indiv_state)

        warm_theta = dict(theta)
        warm_col = dict(pred_col)
        for m in group_models:
            d_m = dims_by_model[m]
            if d_m == 0 or m in frozen or m in scorer.inactive:
                continue
            if _over_budget():
                stats['time_stopped'] = True
                break
            n_warm = int(settings.get('block_warm_n_iter', _JOINT_BLOCK_WARM_N_ITER_DEFAULT) or 0) \
                or max(10, 3 * d_m)

            def _warm_objective_batch(xs, _m=m):
                cols = _fit_batch_recorded(_m, xs)
                stats['block_evals'] += len(xs)
                scores = []
                for x_m, col in zip(xs, cols):
                    if col is None:
                        scores.append(_failure_score(indiv_state))
                        continue
                    x_arr = np.array(x_m, dtype=float)
                    col_hist[_m].setdefault(x_arr.tobytes(), (x_arr, col))
                    scores.append(_indiv_score(col))
                return scores

            try:
                if BayesianOptimization is not None:
                    batch_bayesian_maximise(
                        _warm_objective_batch, [(0.0, 1.0)] * d_m,
                        n_iter=max(0, n_warm - 1 - block_init_points),
                        init_points=block_init_points,
                        seed=_model_seed(m, _JOINT_WARM_SEED_OFFSET), q=q_m,
                        probe_points=[theta[m]],
                        prior_observations=[(theta[m], _indiv_score(pred_col[m]))],
                        should_stop=_stop_hook, noise_aware=_noise_aware,
                    )
            except Exception as exc:
                print(f"[GP] NOTE: Joint Bayesian optimisation individual warm-start pass failed "
                      f"for model {m!r}'s own block ({type(exc).__name__}: {exc}) - using the "
                      f"best point fitted so far (or the untuned one) for this block.")
            # Best point fitted for this block by its OWN score; ties keep the
            # earliest (the untuned point comes first).
            entries = list(col_hist[m].values())
            best_i = int(np.argmax([_indiv_score(col) for _x, col in entries]))
            warm_theta[m], warm_col[m] = entries[best_i]

        warm_ens = _score_columns(warm_col)
        stats['warm_x'] = _assemble(warm_theta)
        stats['warm_ens'] = float(warm_ens)
        if warm_ens > current_score:
            start_label = 'warm'
            theta, pred_col, current_score = dict(warm_theta), dict(warm_col), warm_ens
        else:
            start_label = 'seed'
        print(f"[GP] Joint Bayesian optimisation individual warm start: ensemble score of the "
              f"untuned settings={stats['seed_ens']:.4f}, of the individually tuned settings="
              f"{warm_ens:.4f} - the ensemble cycles start from the {start_label!r} settings.")

    # ---- ensemble cycles ---------------------------------------------------
    _stopped = bool(stats.get('time_stopped'))
    for cycle in range(1, block_cycles + 1):
        if _stopped:
            break
        cycle_gain = 0.0
        for m in group_models:
            d_m = dims_by_model[m]
            if d_m == 0 or m in frozen or m in scorer.inactive:
                continue
            if _over_budget():
                stats['time_stopped'] = True
                _stopped = True
                break
            if cycle == 1:
                n_iter_m = block_n_iter_config or (max(4, d_m) if warm else max(8, 2 * d_m))
            else:
                n_iter_m = block_refine_iter
            if warm:
                # Zero-fit re-score of every cached column of this block
                # against the CURRENT other columns (most recent 50).
                hist = []
                recent_keys = set()
                for key_h, (x_h, col_h) in list(col_hist[m].items())[-_JOINT_COL_HIST_CAP:]:
                    trial_pred = dict(pred_col)
                    trial_pred[m] = col_h
                    hist.append((x_h, _score_columns(trial_pred)))
                    recent_keys.add(key_h)
                if np.asarray(theta[m], dtype=float).tobytes() not in recent_keys:
                    hist.append((np.array(theta[m], dtype=float), current_score))
            else:
                # Zero-fit re-score of this block's own CURRENT point at its
                # already-known score, seeded into this round's history.
                hist = [(theta[m], current_score)]

            def _block_objective_batch(xs, _m=m):
                cols = _fit_batch_recorded(_m, xs)
                stats['block_evals'] += len(xs)
                scores = []
                for x_m, col in zip(xs, cols):
                    if col is None:
                        scores.append(_failure_score(state))
                        continue
                    if warm:
                        x_arr = np.array(x_m, dtype=float)
                        col_hist[_m].setdefault(x_arr.tobytes(), (x_arr, col))
                    trial_pred = dict(pred_col)
                    trial_pred[_m] = col
                    scores.append(_score_columns(trial_pred))
                return scores

            x_star, score_star = theta[m], current_score
            if d_m >= 1 and n_iter_m > 0 and BayesianOptimization is not None:
                try:
                    x_star, score_star, _trials = batch_bayesian_maximise(
                        _block_objective_batch, [(0.0, 1.0)] * d_m,
                        n_iter=max(0, n_iter_m - 1 - block_init_points), init_points=block_init_points,
                        seed=_model_seed(m), q=q_m, probe_points=[theta[m]],
                        prior_observations=hist, should_stop=_stop_hook,
                        noise_aware=_noise_aware, incumbent=_incumbent,
                    )
                except Exception as exc:
                    print(f"[GP] NOTE: Joint Bayesian optimisation block-coordinate search failed for "
                          f"model {m!r}'s own block ({type(exc).__name__}: {exc}) - keeping this "
                          f"block at its current value for this cycle.")
                    x_star, score_star = theta[m], current_score

            gain = score_star - current_score
            if _noise_aware and (np.asarray(x_star, dtype=float).tobytes()
                                 != np.asarray(theta[m], dtype=float).tobytes()):
                # ver4-36 R2.a: with a posterior-mean incumbent an update needs a gain in the
                # surrogate's posterior MEAN, not in one noisy observed score (the observed
                # score of a lucky point would otherwise pass `block_tol` on noise alone).
                gain = _posterior_gain(hist, _trials, x_star, theta[m], _model_seed(m), gain)
            if gain > block_tol:
                col = _fit_column(m, x_star)
                stats['block_evals'] += 1
                if col is not None:
                    theta[m] = x_star
                    pred_col[m] = col
                    current_score = _score_columns(pred_col)
                    cycle_gain += gain
                    stats['blocks_updated'] += 1
        stats['cycles_run'] = cycle
        if _stopped:
            break
        frozen = _refresh_frozen(f'after cycle {cycle}')
        if cycle_gain <= block_tol:
            break

    # ver4-36 R1.b/R2.b: the recorded library and the index of each model's chosen entry
    # (printed neither in the search summary nor anywhere else - `tune_...` pops both).
    stats['lib'] = {m: list(lib_rec[m].values()) for m in group_models}
    _bytes_index = {m: {k: i for i, k in enumerate(lib_rec[m])} for m in group_models}
    stats['lib_cur'] = {m: _bytes_index[m].get(np.asarray(theta[m], dtype=float).tobytes(), 0)
                        for m in group_models}
    return _assemble(theta), current_score, stats


def _parallel_library_search(*, group_models, specs_by_model, base_params_by_model, seed_point,
                              fit_ctx: dict, score_fn: '_ColumnScorer', settings: dict, state,
                              seed: int) -> Tuple[np.ndarray, float, dict]:
    """ver4-34, R1.c: `search_strategy='parallel_library'` - every model proposes
    candidates in the SAME round, all of them are fitted in ONE parallel batch, every
    fitted column is kept in a per-model library, and the combination is chosen from
    the libraries with zero extra fits.

    Why: with analytic/bagged weights the joint score is a pure function of the K
    prediction columns (see `_block_coordinate_search()`, Levesque 2016), so blocks can
    propose CONCURRENTLY against the same fixed "other" columns (Jacobi style, not
    Gauss-Seidel) and any already-fitted column can be recombined at zero fits. The
    sequential-batch count drops from ~30 (block search) to `library_rounds + 1`
    (A-1c.1) and each batch has `sum(q_m)` independent units for `parallel_trials`.

    Flow: R0 fits the untuned column of every model (one batch). Each round r: every
    active model m proposes `X_m` (round 1: `block_init_points` random points; later: a
    constant-liar GP batch of `library_q` points from `parallel_search.suggest_batch()`
    conditioned on the zero-fit re-scores of m's library against the CURRENT other
    columns); all proposals are fitted in one `_run_units(stop_on_failure=False)` call
    (`_fit_requests_batch()`); successful columns join `lib[m]` (a failed one is dropped);
    `_coordinate_select()` re-selects one column per model, restarted once from each
    model's single-model best - the better result wins. Stops after `library_rounds`
    rounds, on `search_patience` (evaluations without improvement), when the wall-clock
    budget has elapsed, or - from round 2 on, round 1 being random exploration - when a
    round's gain is <= `block_tol`.

    R1.d: with `freeze_low_weight > 0` and the bagged estimator, a model whose P90 bagged
    weight is below the threshold gets no proposals (`_frozen_models()`, recomputed after
    every round, so it can be unfrozen); it stays in the ensemble at its selected column.

    Args: `group_models`, `specs_by_model`, `base_params_by_model` as everywhere in this
    module; `seed_point` the untuned point (concatenated unit-cube layout, the same one
    `_block_coordinate_search()` returns, so the gate/confirm/weights code is reused);
    `fit_ctx` from `_make_column_context()` (its `runner_ctx` carries the cache, counter
    and `n_jobs`); `score_fn` a `_ColumnScorer`; `settings` keys: library_rounds,
    library_q, block_init_points, block_tol, search_patience, freeze_low_weight,
    weight_bags, cluster_corr, estimator_settings, objective_mode, diversity_method,
    time_budget_s, time_budget_t0; `state` the shared failure-score state; `seed` the
    search seed (per-model proposal seeds are `seed + crc32(model) % 10000 + round`, I9).

    Returns `(best_x, best_score, stats)`. `stats` = `rounds` (run), `lib_sizes`,
    `frozen` ({tag: [models]}), `swaps`, `batches` (fit batches, incl. R0), `seed_ens`
    and `stop` (why it ended). Returns `(seed_point, -inf, stats)` when an untuned
    column cannot be fitted (the caller falls back to the monolithic search). Never
    raises for a failed fit; raises whatever `_decode_model_params()` raises for a
    malformed point."""
    dims = {m: len(specs_by_model[m]) for m in group_models}
    offsets = {}
    _off = 0
    for m in group_models:
        offsets[m] = _off
        _off += dims[m]
    seed_point = np.asarray(seed_point, dtype=float)
    actual = np.asarray(fit_ctx['actual'], dtype=float)
    rounds = max(1, int(settings.get('library_rounds', _JOINT_LIBRARY_ROUNDS_DEFAULT)))
    q = max(1, int(settings.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)))
    init_points = max(1, int(settings.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)))
    tol = float(settings.get('block_tol', _JOINT_BLOCK_TOL_DEFAULT))
    patience = int(settings.get('search_patience', _JOINT_SEARCH_PATIENCE_DEFAULT) or 0)
    freeze_low = float(settings.get('freeze_low_weight', _JOINT_FREEZE_LOW_WEIGHT_DEFAULT) or 0.0)
    est = dict(settings.get('estimator_settings') or {})
    bagged = bool(getattr(score_fn, 'bagged', False))
    patience_hook = _patience_stop(patience)
    budget_s = float(settings.get('time_budget_s', _JOINT_TIME_BUDGET_DEFAULT) or 0.0)
    budget_hook = (_time_budget_stop(settings['time_budget_t0'], budget_s)
                   if settings.get('time_budget_t0') is not None else None)
    acquisition = ((lambda: ExpectedImprovement(xi=0.01)) if ExpectedImprovement is not None
                   else None)
    noise_aware = bool(settings.get('noise_aware_surrogate', _JOINT_NOISE_AWARE_DEFAULT))
    stats = {'rounds': 0, 'lib_sizes': {}, 'frozen': {}, 'swaps': 0, 'batches': 0,
             'seed_ens': None, 'stop': ''}

    def _lib_seed(m, r):
        return int(seed) + (zlib.crc32(m.encode('utf-8')) % 10_000) + int(r)

    def _assemble(cur_idx):
        if sum(dims.values()) == 0:
            return np.array([], dtype=float)
        return np.concatenate([np.asarray(lib[m][cur_idx[m]][0], dtype=float)
                               for m in group_models])

    # ---- R0: the untuned column of every model, one batch --------------------------------
    chunks0 = {m: seed_point[offsets[m]:offsets[m] + dims[m]] for m in group_models}
    _rc0 = fit_ctx.get('runner_ctx') or {}
    _snap0 = _counter_snapshot(_rc0.get('counter'), group_models)
    cols0 = _fit_requests_batch([(m, chunks0[m]) for m in group_models], ctx=fit_ctx)
    stats['batches'] = 1
    _log_eta('parallel-library', group_models, settings.get('eta_plan'),
             _seed_times(_rc0.get('counter'), _snap0, group_models, settings.get('t_seed')),
             int(_rc0.get('n_jobs', 1) or 1))
    failed0 = [m for m, col in zip(group_models, cols0) if col is None]
    if failed0:
        print(f"[GP] NOTE: Joint Bayesian optimisation parallel-library search could not fit the "
              f"initial (untuned) column for {failed0!r} - falling back to the monolithic search.")
        return seed_point, -math.inf, stats
    lib = {m: [(np.array(chunks0[m], dtype=float), col)] for m, col in zip(group_models, cols0)}
    seen = {m: {np.array(chunks0[m], dtype=float).tobytes()} for m in group_models}
    cur = {m: 0 for m in group_models}

    def _cur_cols():
        return {m: lib[m][cur[m]][1] for m in group_models}

    # ver4-38, R2.b1: a model whose untuned column is constant leaves the scored group (no-op
    # unless `dead_column_guard`); it gets no proposals below.
    stats['dead_models'] = score_fn.drop_dead_seed_models(_cur_cols(), where='parallel-library R0')
    cur_score = float(score_fn(_cur_cols()))
    stats['seed_ens'] = cur_score
    indiv_state = {'worst_finite': None}  # the single-model scores keep their own failure scale

    def _single_score(col):
        return _combine_score(np.array([1.0]), np.asarray(col, dtype=float)[:, None], actual,
                              ['single'], settings.get('objective_mode', 'ensemble_score'), 0.0,
                              settings.get('diversity_method', _JOINT_DIVERSITY_METHOD_DEFAULT),
                              state=indiv_state)

    def _weight_order():
        scored = score_fn.active_models      # ver4-38: models dropped by the dead-column guard
        matrix = np.column_stack([lib[m][cur[m]][1] for m in scored])
        if bagged:
            w = _robust_weight_fn(
                settings.get('weight_estimator', 'bagged_cluster'), est,
                n_bags=int(est.get('score_bags', _JOINT_SCORE_BAGS_DEFAULT)),
                seed=int(seed))(matrix, actual)
        else:
            w = analytic_simplex_weights(matrix, actual)
        w = np.asarray(w, dtype=float)
        return [scored[i] for i in sorted(range(len(scored)), key=lambda i: (-w[i], i))]

    def _refresh_frozen(tag):
        if not (freeze_low > 0.0 and bagged):
            return {}
        fz = _frozen_models(
            _cur_cols(), score_fn.active_models, actual, low=freeze_low,
            n_bags=int(settings.get('weight_bags', _JOINT_WEIGHT_BAGS_DEFAULT)),
            cluster_corr=float(settings.get('cluster_corr', _JOINT_CLUSTER_CORR_DEFAULT)),
            seed=int(seed),
            estimator=settings.get('weight_estimator', 'bagged_cluster'),
            estimator_settings=settings.get('estimator_settings'))
        stats['frozen'][tag] = sorted(fz)
        print(f"[GP] Joint Bayesian optimisation library search ({tag}): frozen models (P90 bagged "
              f"weight < {freeze_low:g}) = {fz if fz else 'none'}; a frozen model stays in the "
              f"ensemble at its selected column, only its tuning stops.")
        return fz

    frozen = _refresh_frozen('after the untuned columns')
    history: List[Tuple[np.ndarray, float]] = []

    # ---- rounds ----------------------------------------------------------------------------
    for r in range(1, rounds + 1):
        if r > 1 and budget_hook is not None and budget_hook([]):
            stats['stop'] = 'time budget'
            break
        active = [m for m in group_models
                  if dims[m] > 0 and m not in frozen and m not in score_fn.inactive]
        if not active:
            stats['stop'] = 'no tunable, unfrozen model'
            break
        requests = []
        for m in active:
            bounds = [(0.0, 1.0)] * dims[m]
            if r == 1:
                points = suggest_batch(bounds, [], q=init_points, seed=_lib_seed(m, r))
            else:
                cols_now = _cur_cols()
                observations = []
                for x_e, col_e in lib[m][-_JOINT_LIBRARY_RESCORE_CAP:]:
                    trial = dict(cols_now)
                    trial[m] = col_e
                    sc = float(score_fn(trial))
                    if np.isfinite(sc):
                        observations.append((x_e, sc))
                try:
                    points = suggest_batch(bounds, observations, q=q, seed=_lib_seed(m, r),
                                           liar='max', acquisition_factory=acquisition,
                                           noise_aware=noise_aware)
                except (ImportError, RuntimeError) as exc:
                    print(f"[GP] NOTE: Joint Bayesian optimisation library search could not build a "
                          f"surrogate suggestion for {m!r} in round {r} ({type(exc).__name__}: "
                          f"{exc}) - proposing random points for it this round.")
                    points = suggest_batch(bounds, [], q=q, seed=_lib_seed(m, r))
            for x in points:
                x = np.array(x, dtype=float)
                key = x.tobytes()
                if key in seen[m]:
                    continue
                seen[m].add(key)
                requests.append((m, x))
        if not requests:
            stats['stop'] = 'no new proposals'
            break
        cols = _fit_requests_batch([(m, x) for m, x in requests], ctx=fit_ctx)
        stats['batches'] += 1
        dropped = 0
        cols_now = _cur_cols()
        for (m, x), col in zip(requests, cols):
            if col is None:
                dropped += 1
                continue
            lib[m].append((x, col))
            trial = dict(cols_now)
            trial[m] = col
            history.append((x, float(score_fn(trial))))
        if dropped:
            print(f"[GP] NOTE: Joint Bayesian optimisation library search round {r}: {dropped} of "
                  f"{len(requests)} proposed column(s) failed to fit and were dropped from the "
                  f"library.")

        # Zero-fit selection: from the current columns, and restarted once from each model's
        # single-model best; the better of the two wins.
        order = _weight_order()
        round_start = cur_score
        cur_a, score_a, swaps_a = _coordinate_select(
            lib, cur, score_fn, order, max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
        single_best = {m: int(np.argmax([_single_score(col) for _x, col in lib[m]]))
                       for m in group_models}
        cur_b, score_b, swaps_b = _coordinate_select(
            lib, single_best, score_fn, order, max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
        if score_b > score_a:
            cur, cur_score, swaps = cur_b, float(score_b), swaps_b
        else:
            cur, cur_score, swaps = cur_a, float(score_a), swaps_a
        stats['swaps'] += swaps
        stats['rounds'] = r
        stats['lib_sizes'] = {m: len(lib[m]) for m in group_models}
        print(f"[GP] Joint Bayesian optimisation library search round {r}/{rounds}: fitted "
              f"{len(cols) - dropped} column(s) in one batch, library sizes={stats['lib_sizes']}, "
              f"ensemble score {round_start:.4f} -> {cur_score:.4f} ({swaps} swap(s)).")

        gain = cur_score - round_start
        if r >= 2 and gain <= tol:
            stats['stop'] = 'round gain <= block_tol'
            break
        if patience_hook is not None and patience_hook(list(history)):
            stats['stop'] = 'search_patience'
            break
        if budget_hook is not None and budget_hook([]):
            stats['stop'] = 'time budget'
            break
        frozen = _refresh_frozen(f'after round {r}')
    else:
        stats['stop'] = 'library_rounds reached'
    stats['lib_sizes'] = {m: len(lib[m]) for m in group_models}
    # ver4-36 R1.b/R2.b: the fitted library and the chosen entries, for the bootstrap gate /
    # selection and the dump (popped from the printed summary by `_STATS_NOT_PRINTED`).
    stats['lib'] = lib
    stats['lib_cur'] = dict(cur)
    return _assemble(cur), float(cur_score), stats


# ==========================================================================================
# ver4-36 (design blueprint EasiGP_ver4-36_Design_Blueprint.md): cost accounting (R1.a),
# bootstrap gate and selection (R1.b / R2.b), parity guards for the test stub and the exact
# engine (R1.c / R1.d), the asynchronous library search (R1.e) and the library dump (R2.d).
# ==========================================================================================

def _counter_snapshot(counter: Optional[dict], group_models: Sequence[str]) -> dict:
    """ver4-36, R1.a: copy of the per-model fit counts and seconds of `counter` (used to
    measure how long the seed fits of a search took by differencing two snapshots)."""
    counter = counter or {}
    return {'fits': dict(counter.get('fits_by_model') or {}),
            'secs': dict(counter.get('secs_by_model') or {})}


def _seed_times(counter: Optional[dict], before: dict, group_models: Sequence[str],
                fallback: Optional[dict] = None) -> Dict[str, float]:
    """ver4-36, R1.a: measured seconds PER FIT of each model since the snapshot `before`.

    Why: the ETA line extrapolates from the seed columns, the first fits of the search.
    A model whose seed fit was a cache hit (no fit since the snapshot) uses `fallback[m]`
    (the parity guard's measurement) and then 0.0 if nothing was measured. Never raises."""
    counter = counter or {}
    fits_now = counter.get('fits_by_model') or {}
    secs_now = counter.get('secs_by_model') or {}
    fallback = fallback or {}
    out = {}
    for m in group_models:
        d_fits = fits_now.get(m, 0) - before.get('fits', {}).get(m, 0)
        d_secs = secs_now.get(m, 0.0) - before.get('secs', {}).get(m, 0.0)
        if d_fits > 0 and d_secs > 0.0:
            out[m] = float(d_secs) / d_fits
        else:
            out[m] = float(fallback.get(m, 0.0))
    return out


def _log_eta(label: str, group_models: Sequence[str], plan: Optional[dict],
             times: Dict[str, float], width: int) -> None:
    """ver4-36, R1.a: print ONE line with the expected search time, from the measured
    seed-fit seconds `times[m]` and the planned work `plan` = `{'fits_by_model',
    'rounds_by_model'}` (from `_estimate_joint_fits()`).

    CPU time = sum_m fits_m * t_m, divided by `width` (the fit parallelism); critical path =
    max_m rounds_m * t_m (the longest chain of sequential fit rounds, which no amount of
    parallelism shortens); the expected wall time is the larger of the two. Only the SEARCH
    stage is estimated - the gate, the confirmatory and any out-of-fold fits come on top.
    `plan=None` prints nothing. Never raises."""
    if not plan:
        return
    fits = plan.get('fits_by_model') or {}
    rounds = plan.get('rounds_by_model') or {}
    width = max(1, int(width))
    cpu = sum(fits.get(m, 0) * times.get(m, 0.0) for m in group_models)
    chain = {m: rounds.get(m, 0) * times.get(m, 0.0) for m in group_models}
    crit_model = max(chain, key=chain.get) if chain else None
    crit = chain[crit_model] if crit_model is not None else 0.0
    wall = max(cpu / width, crit)
    per_model = ', '.join(f"{m}={times.get(m, 0.0):.2f}s" for m in group_models)
    print(f"[GP] Joint Bayesian optimisation ETA ({label}; search stage only - the gate, the "
          f"confirmatory fit and any out-of-fold fits come on top): "
          f"{sum(fits.get(m, 0) for m in group_models)} planned fit(s), CPU time ~{cpu:.0f}s / "
          f"width {width} = ~{cpu / width:.0f}s; critical path ~{crit:.0f}s"
          + (f" ({crit_model}: {rounds.get(crit_model, 0)} sequential round(s))"
             if crit_model is not None else "")
          + f" => expected ~{wall:.0f}s. Measured seconds per seed fit: {per_model}.")


# ------------------------------------------------------------------------------------------
# R1.b / R2.b - bootstrap-bias-corrected (BBC) gate and selection over the fitted library
# ------------------------------------------------------------------------------------------

def _pick_key(pick: Dict[str, int], group_models: Sequence[str]) -> tuple:
    """Hashable form of a library selection `{model: entry index}`."""
    return tuple(int(pick[m]) for m in group_models)


def _bbc_analyse(lib: Dict[str, list], group_models: Sequence[str], actual: np.ndarray, *,
                 c_full: Dict[str, int], n_boot: int, seed: int,
                 shrink_alpha: Optional[float] = None, tol: float = _JOINT_BLOCK_TOL_DEFAULT,
                 quantile: float = _JOINT_BBC_GATE_QUANTILE_DEFAULT, margin: float = 0.0,
                 dead_guard: bool = False, partition=None) -> dict:
    """ver4-36, R1.b + R2.b: bootstrap-bias-corrected analysis of a fitted library
    (Tsamardinos et al., 2018, Machine Learning 107:1895) - ZERO model fits.

    The joint score is a pure function of the K prediction columns, so the whole
    "choose one library column per model" procedure can be repeated on resampled rows.
    For each of `n_boot` bootstrap draws (in-bag rows with replacement, out-of-bag rows
    never drawn, at least 3):

      1. the in-bag selection `c_b` is `_coordinate_select()` started from the untuned
         columns (entry 0 of every model), scoring with `fast_oob_score(P, y, ib, ib)`: exact
         simplex (NNLS) weights fitted on the in-bag rows, shrunk with `shrink_alpha` when
         given, scored on the in-bag rows;
      2. `gap_b = OOB(c_b) - OOB(untuned)`, where OOB(c) is the same fast score with
         weights fitted in-bag and scored on the out-of-bag rows.

    GATE: the tuned procedure is accepted iff the `quantile` (default 0.16, the
    one-standard-error analogue) of the gaps exceeds `margin`; the gap distribution measures
    how much of the in-sample gain of the selection survives on rows it did not see.
    SELECTION: the candidate set is {`c_full` (the strategy's greedy choice on all rows),
    the untuned selection} U {c_b}; the pick is the candidate with the highest MEAN
    out-of-bag score over the draws; ties go to `c_full`, then to the untuned selection.

    Disclosure: BBC resamples scoring rows, not training sets - and with the CV objective the
    columns already carry the training-fold variability. The weights of the resampled
    procedure are the fast NNLS weights for every estimator (the configured estimator
    only decides the reported weights), a deliberate simplification: running a bagged
    estimator 200 x |library| times would cost more than the fits it saves.

    Args: `lib[m]` list of `(x_m, column)` with entry 0 the untuned column; `actual` the
    scoring targets (len = column length); `c_full` `{model: index}`; `n_boot` requested
    draws; `seed` any int (fixed seed => identical result across processes, A-1b.2);
    `shrink_alpha` the shrinkage of the legacy 'shrunk' score rule or None; `tol` the
    greedy swap tolerance; `quantile`, `margin` the gate parameters; ver4-38 `dead_guard`
    (default False = ver4-37) makes the in-bag selection refuse any library entry that
    `_is_dead_column()` calls dead (it scores -inf, so it is never swapped in); `partition`
    (the CV folds of an out-of-fold library, or None) is handed to that test.

    Returns a dict: `ok` (False with a `reason` when the analysis cannot be formed - fewer
    than 20 rows or 20 valid draws), `n_valid`, `gaps` (array), `gate_quantile`
    (the quantile value), `accepted`, `selected` (`{model: index}`), `selected_label`
    ('full' | 'untuned' | 'bootstrap'), `mean_oob` ({label: mean OOB score of c_full,
    untuned, selected}). Never raises for a well-formed library."""
    y = np.asarray(actual, dtype=float)
    n = int(y.size)
    out = {'ok': False, 'reason': '', 'n_valid': 0, 'gaps': np.array([]), 'gate_quantile': math.nan,
           'accepted': False, 'selected': dict(c_full), 'selected_label': 'full', 'mean_oob': {}}
    if n < _JOINT_BBC_MIN_ROWS:
        out['reason'] = f"only {n} scoring row(s) (need >= {_JOINT_BBC_MIN_ROWS})"
        return out
    draws = bootstrap_oob_indices(n, int(n_boot), int(seed))
    if len(draws) < _JOINT_BBC_MIN_VALID:
        out['reason'] = (f"only {len(draws)} usable bootstrap draw(s) (need >= "
                         f"{_JOINT_BBC_MIN_VALID})")
        return out
    stacks = {m: np.column_stack([np.asarray(col, dtype=float) for _x, col in lib[m]])
              for m in group_models}
    seed_pick = {m: 0 for m in group_models}
    order = list(group_models)

    def _matrix(pick):
        return np.column_stack([stacks[m][:, pick[m]] for m in group_models])

    gaps: List[float] = []
    kept: List[Tuple[Dict[str, int], np.ndarray, np.ndarray]] = []
    seed_matrix = _matrix(seed_pick)
    for ib, ob in draws:
        def _in_bag(cols_by_model, _ib=ib):
            if dead_guard and any(_is_dead_column(cols_by_model[m], y, partition)
                                  for m in group_models):
                return -math.inf
            P = np.column_stack([cols_by_model[m] for m in group_models])
            v = fast_oob_score(P, y, _ib, _ib, shrink_alpha=shrink_alpha)
            return -math.inf if v is None else float(v)

        c_b, _score_b, _swaps = _coordinate_select(
            lib, seed_pick, _in_bag, order, max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
        oob_c = fast_oob_score(_matrix(c_b), y, ib, ob, shrink_alpha=shrink_alpha)
        oob_s = fast_oob_score(seed_matrix, y, ib, ob, shrink_alpha=shrink_alpha)
        if oob_c is None or oob_s is None:
            continue
        gaps.append(float(oob_c) - float(oob_s))
        kept.append((c_b, ib, ob))
    out['n_valid'] = len(gaps)
    if len(gaps) < _JOINT_BBC_MIN_VALID:
        out['reason'] = (f"only {len(gaps)} bootstrap draw(s) gave a usable out-of-bag score "
                         f"(need >= {_JOINT_BBC_MIN_VALID})")
        return out
    gaps_arr = np.asarray(gaps, dtype=float)
    out['gaps'] = gaps_arr
    out['gate_quantile'] = float(np.quantile(gaps_arr, float(quantile)))
    out['accepted'] = bool(out['gate_quantile'] > float(margin))

    # ---- selection: highest mean out-of-bag score among {c_full, untuned} U {c_b} --------
    candidates: List[Tuple[str, Dict[str, int]]] = [('full', dict(c_full)),
                                                    ('untuned', dict(seed_pick))]
    candidates.extend(('bootstrap', c_b) for c_b, _ib, _ob in kept)
    seen_keys = set()
    best_label, best_pick, best_mean = None, None, -math.inf
    means: Dict[str, float] = {}
    for label, pick in candidates:
        key = _pick_key(pick, group_models)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        P = _matrix(pick)
        vals = []
        for _c_b, ib, ob in kept:
            v = fast_oob_score(P, y, ib, ob, shrink_alpha=shrink_alpha)
            vals.append(math.nan if v is None else float(v))
        arr = np.asarray(vals, dtype=float)
        mean = float(np.nanmean(arr)) if np.isfinite(arr).any() else -math.inf
        if label in ('full', 'untuned'):
            means[label] = mean
        if mean > best_mean:
            best_label, best_pick, best_mean = label, dict(pick), mean
    out['ok'] = True
    out['selected'] = best_pick if best_pick is not None else dict(c_full)
    out['selected_label'] = best_label or 'full'
    means['selected'] = best_mean
    out['mean_oob'] = means
    return out


def _wf_rule_specs(cands: Sequence[Tuple[str, np.ndarray]]) -> list:
    """ver4-40, R2: the `weight_candidates` rule spec that reproduces each candidate of
    `cands` (the `[(name, w), ...]` list it returned) - the names are canonical grammar
    tokens ('uniform', 'analytic', 'shrunk:0.5', 'es'), a 'custom:<i>' vector is passed back
    as an explicit list of floats. `_wf_bbc()` needs them to re-run each rule on a bootstrap
    sample; one function keeps the mapping in a single place (I3)."""
    return [list(map(float, w)) if str(name).startswith('custom:') else str(name)
            for name, w in cands]


def _wf_complexity_key(finalists: Sequence[dict], i: int, live: Sequence[int]) -> tuple:
    """ver4-41: how much freedom finalist `i` used - `(L1 distance of its weight vector from the
    equal-weight vector over the LIVE models (rounded to 0.01), 1 if it is a tuned arm, i)`.
    Smaller sorts first: the untuned equal-weight blend is the simplest finalist, a vertex
    vector the most complex, and between two equally weighted finalists the untuned one wins.
    One function so the pick (`_wf_bbc`) and its fallback (`_wf_simplest`) cannot disagree."""
    w = np.asarray(finalists[i]['w'], dtype=float)
    k = max(1, len(live))
    uniform = np.zeros_like(w)
    for j in live:
        uniform[j] = 1.0 / k
    return (round(float(np.abs(w - uniform).sum()), 2),
            1 if finalists[i]['arm'] == 'tuned' else 0, int(i))


def _wf_simplest(finalists: Sequence[dict], dead: Sequence[int] = ()) -> int:
    """ver4-41: index of the simplest finalist (`_wf_complexity_key`) - the choice made when the
    bootstrap evidence cannot be formed (too few scoring rows). Replaces the ver4-40 fallback to
    the best FULL-ROW score, which is the finalist that fitted those rows best by construction
    (a tuned arm of the in-sample optimal 'analytic' vector, i.e. a one-hot weight vector)."""
    dead_set = {int(d) for d in dead}
    K = len(finalists[0]['w']) if finalists else 0
    live = [j for j in range(K) if j not in dead_set]
    return min(range(len(finalists)), key=lambda i: _wf_complexity_key(finalists, i, live))


def _wf_bbc(lib: Dict[str, list], finalists: Sequence[dict], rules: Sequence, actual: np.ndarray,
            *, n_boot: int, seed: int, partition=None, es_settings: Optional[dict] = None,
            tol: float = _JOINT_BLOCK_TOL_DEFAULT, parsimony: bool = True,
            k_se: float = _JOINT_WF_K_SE_DEFAULT,
            max_passes: int = _JOINT_LIBRARY_MAX_PASSES) -> dict:
    """ver4-40, R2 (revised ver4-41): bootstrap-bias-corrected PICK among the (weight vector,
    column set) finalists of the weight-first search - ZERO model fits.

    Why: the finalists differ in how much freedom they used (a tuned arm has searched library
    columns under its weight vector; an untuned arm uses the configured columns; a data-fitted
    weight vector such as 'analytic' has K-1 free weights), so comparing their full-row scores
    favours the finalist that fitted the scoring rows best. The honest comparison scores every
    finalist on rows it did not see (Tsamardinos et al., 2018, Machine Learning 107:1895). For
    each of `n_boot` draws (`bootstrap_oob_indices`, in-bag rows with replacement, out-of-bag
    rows never drawn):

      1. every finalist's weight RULE is re-run IN-BAG on the untuned (entry 0) columns -
         `weight_candidates(..., rules=[rule], dedup_tol=0, max_n=1)` - so a data-dependent
         vector (analytic, shrunk, es) is re-estimated exactly as the full-row one was;
      2. a 'tuned' finalist re-selects its library columns IN-BAG, STARTING FROM THE UNTUNED
         COLUMNS, with up to `max_passes` `_coordinate_select()` passes (scored in-bag at the
         in-bag weights) - the same procedure that produced its full-row selection;
      3. the finalist is scored on the out-of-bag rows at the in-bag weights
         (`_ensemble_score_of`, the 'ensemble_score' objective).

    ver4-41 fixes (blueprint-free; evidence: a replay on simulated near-equal-quality libraries,
    ~50 scoring rows - the regime of a typical within-population task):
      * LEAK: ver4-40 started a tuned finalist's in-bag re-selection at its FULL-ROW selection,
        which was chosen using the out-of-bag rows too, so tuned arms scored optimistically.
        The re-selection now starts from the untuned columns (as `_bbc_analyse` does).
      * WINNER'S CURSE: the plain argmax over ~2C noisy mean out-of-bag scores picked a
        vertex / tuned finalist whose edge was noise (mean test r 0.009 BELOW the equal-weight
        blend). With `parsimony=True` (default) the pick is the SIMPLEST finalist
        (`_wf_complexity_key`) whose mean out-of-bag score is within `k_se` standard
        deviations of the paired out-of-bag difference to the best finalist (Breiman's
        one-standard-error rule). A complex finalist is therefore chosen only when the
        out-of-bag evidence for it is strong; otherwise the equal-weight blend (or the simplest
        adequate finalist) is. `parsimony=False` is the plain argmax (ties go to a tuned arm over
        an untuned one, then to the earlier finalist).

    Speed: each library column's dead/constant flag is computed once, not on every score.

    Args: `lib[m]` list of `(x_m, column)` entries, entry 0 = untuned; `finalists` a list of
    dicts with `c` (index of the finalist's rule in `rules`), `arm` ('tuned' | 'untuned'),
    `pick` ({model: library index}, the full-row selection; all zeros for 'untuned') and
    `w` (the full-row weight vector); `rules` the rule list that built the candidates
    (`weight_candidates` grammar), indexed by `c`; `actual` the scoring targets; `n_boot`
    requested draws; `seed` any int (fixed seed => identical result in every process, I9);
    `partition` the CV folds of an out-of-fold library or None (only used by the dead-column
    test); `es_settings` the ES keywords of `weight_candidates`; `tol` the greedy swap
    tolerance; `parsimony` / `k_se` the one-standard-error rule above; `max_passes` the greedy
    passes of a tuned finalist's in-bag re-selection.

    Returns `{'ok': bool, 'reason': str, 'n_valid': int, 'selected': index into `finalists`
    (0 when not ok), 'best': index of the highest-mean finalist, 'mean_oob': list of floats
    (NaN where a finalist had no valid draw), 'gap': list (mean paired difference to the best),
    'sd_gap': list (its standard deviation over the draws), 'one_se': bool (the parsimony rule
    chose a finalist other than the best)}`. `ok` is False (and the caller falls back to
    `_wf_simplest`) when fewer than `_JOINT_BBC_MIN_ROWS` scoring rows or
    `_JOINT_BBC_MIN_VALID` usable draws exist. Never raises for a well-formed library."""
    y = np.asarray(actual, dtype=float)
    n = int(y.size)
    nan_list = [math.nan] * len(finalists)
    out = {'ok': False, 'reason': '', 'n_valid': 0, 'selected': 0, 'best': 0,
           'mean_oob': list(nan_list), 'gap': list(nan_list), 'sd_gap': list(nan_list),
           'one_se': False}
    if not finalists:
        out['reason'] = 'no finalist'
        return out
    if n < _JOINT_BBC_MIN_ROWS:
        out['reason'] = f"only {n} scoring row(s) (need >= {_JOINT_BBC_MIN_ROWS})"
        return out
    draws = bootstrap_oob_indices(n, int(n_boot), int(seed))
    if len(draws) < _JOINT_BBC_MIN_VALID:
        out['reason'] = (f"only {len(draws)} usable bootstrap draw(s) (need >= "
                         f"{_JOINT_BBC_MIN_VALID})")
        return out
    models = list(lib.keys())
    stacks = {m: np.column_stack([np.asarray(col, dtype=float) for _x, col in lib[m]])
              for m in models}
    # one dead/constant test per library column (the ver4-40 code repeated it on every score)
    dead_by_id = {id(col): _is_dead_column(col, y, partition) for m in models for _x, col in lib[m]}
    zero = {m: 0 for m in models}
    dead_seed = [i for i, m in enumerate(models) if dead_by_id[id(lib[m][0][1])]]
    live = [i for i in range(len(models)) if i not in dead_seed]

    def _matrix(pick):
        return np.column_stack([stacks[m][:, pick[m]] for m in models])

    seed_matrix = _matrix(zero)
    F, D = len(finalists), len(draws)
    S = np.full((F, D), np.nan)                 # out-of-bag score of finalist f on draw d
    for d_i, (ib, ob) in enumerate(draws):
        weights_by_rule: Dict[int, Optional[np.ndarray]] = {}
        for fin in finalists:
            c = int(fin['c'])
            if c in weights_by_rule:
                continue
            cand = weight_candidates(seed_matrix[ib], y[ib], rules=[rules[c]], dedup_tol=0.0,
                                     max_n=1, seed=int(seed) + d_i, dead=dead_seed,
                                     es_settings=es_settings)
            weights_by_rule[c] = cand[0][1] if cand else None
        for f_i, fin in enumerate(finalists):
            w_b = weights_by_rule.get(int(fin['c']))
            if w_b is None:
                continue
            pick = dict(fin['pick'])
            if fin['arm'] == 'tuned':
                pick = dict(zero)               # ver4-41: start from the UNTUNED columns (no leak)

                def _in_bag(cols_by_model, _ib=ib, _w=w_b):
                    for i, m in enumerate(models):
                        if _w[i] > 0 and dead_by_id.get(id(cols_by_model[m]), False):
                            return -math.inf
                    v = _ensemble_score_of(
                        np.column_stack([cols_by_model[m] for m in models])[_ib] @ _w, y[_ib])
                    return -math.inf if v is None else float(v)
                pick, _score_b, _swaps = _coordinate_select(
                    lib, pick, _in_bag, models, max_passes=int(max_passes), tol=tol)
            v = _ensemble_score_of(_matrix(pick)[ob] @ w_b, y[ob])
            if v is not None:
                S[f_i, d_i] = float(v)
    n_valid = int(np.isfinite(S).any(axis=0).sum())
    out['n_valid'] = n_valid
    if n_valid < _JOINT_BBC_MIN_VALID:
        out['reason'] = (f"only {n_valid} bootstrap draw(s) gave a usable out-of-bag score "
                         f"(need >= {_JOINT_BBC_MIN_VALID})")
        return out
    with np.errstate(all='ignore'):
        means = np.array([float(np.nanmean(S[i])) if np.isfinite(S[i]).any() else math.nan
                          for i in range(F)])
    out['mean_oob'] = [float(x) for x in means]
    finite = [i for i in range(F) if np.isfinite(means[i])]
    if not finite:
        out['reason'] = 'no finalist had a finite out-of-bag score'
        return out
    top = max(means[i] for i in finite)
    tied = [i for i in finite if means[i] >= top - 1e-12]
    best = min(tied, key=lambda i: (0 if finalists[i]['arm'] == 'tuned' else 1, i))
    out['best'] = int(best)
    admissible = [best]
    for i in finite:
        diff = S[best] - S[i]
        ok = np.isfinite(diff)
        if int(ok.sum()) < 5:
            continue
        gap, sd = float(np.mean(diff[ok])), float(np.std(diff[ok]))
        out['gap'][i], out['sd_gap'][i] = gap, sd
        if i != best and gap <= float(k_se) * sd:
            admissible.append(i)
    if parsimony:
        chosen = min(admissible, key=lambda i: _wf_complexity_key(finalists, i, live))
    else:
        chosen = best
    out['one_se'] = bool(chosen != best)
    out['selected'] = int(chosen)
    out['ok'] = True
    return out


def _safe_name(text: str) -> str:
    """File-system / npz-key safe form of `text` (letters, digits, '.', '_', '-')."""
    out = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text)).strip('_')
    return out or 'x'


def _write_library_dump(path: str, *, group_models: Sequence[str], lib: Dict[str, list],
                        specs_by_model: dict, base_params_by_model: dict, test_sink: Optional[dict],
                        actual: np.ndarray, y_test: Optional[np.ndarray],
                        partition: Optional[Sequence[np.ndarray]], best_x: np.ndarray,
                        meta: dict, oof_cols: Optional[Sequence[np.ndarray]] = None,
                        wf_weights: Optional[Sequence[float]] = None,
                        entry_params: Optional[Dict[str, list]] = None) -> str:
    """ver4-36, R2.d: write the fitted library of one task to `path` (an `.npz`), atomically.

    Why: the joint score is a pure function of the K prediction columns, so a saved library
    lets `tools/joint_replay.py` re-run every selection / gate / weight-estimator design at ZERO
    fits (the blueprint's offline replay harness, A-2d). Nothing in the search reads the dump
    (I7), and `y_test` / the test columns are only ever written, never scored here.

    Layout (every array is plain numeric or a string array, loadable with
    `np.load(path, allow_pickle=False)`):
      `models` (K names), `keys` (K npz-safe names, same order), `y_score` (the scoring
      targets, length n), `y_test` (test targets or empty), `fold` (fold id of every scoring
      row, or -1 everywhere for the validation objective), `meta` (one JSON string: the
      scoring settings, the gate decision, the seed ...), and per model with key `k`:
      `x__k` (entries x dims), `cols__k` (n x entries scoring columns, entry 0 = the untuned
      column), `test__k` (n_test x entries test columns; NaN where none was recorded),
      `params__k` (entries JSON strings of the decoded parameters), `selected__k` (index of
      the entry equal to `best_x`, -1 when `best_x` is not in the library).
    ver4-38 (dump version 2, `meta['dump_version']`): when the final out-of-fold matrix at the
    selected parameters exists (`oof_cols`, one column per group model in `group_models` order,
    scoring-row order), it is stored as `oof__k` (length n) - the column the final weights were
    estimated on, so `tools/joint_replay.py` can measure the weight change between the search
    columns and the final-fidelity columns (blueprint R2.a, measure "delta weights"). A version-1
    dump has no `oof__k`; readers must treat it as optional. `oof_cols` entries of the wrong
    length are skipped with no error.
    The file is written to `path + '.tmp'` and moved with `os.replace`, so a reader never sees
    a partial file. Returns `path`. Raises OSError / ValueError to the caller (which turns it
    into a NOTE: a failed dump never fails a task).
    ver4-40 (dump version stays 2): `wf_weights` (K floats, `group_models` order) is stored as the
    OPTIONAL `wf_weights` array - the fixed weight vector the weight-first search chose; tools
    must treat it as absent for every other strategy.
    ver4-48: `entry_params` (default None = every earlier call, unchanged) maps a model to the
    decoded parameter list of EACH of its library entries, in entry order. The neural stack passes
    it because its entry 0 is fitted with the configured parameters exactly and its entries may
    be decoded without the search cheapening, so re-decoding `x` here would key the test column
    differently and write NaN for it (found in the ver4-48 dump test: SVR / RF entry 0)."""
    arrays: Dict[str, np.ndarray] = {}
    keys = []
    offsets, off = {}, 0
    for m in group_models:
        offsets[m] = off
        off += len(specs_by_model[m])
    best_x = np.asarray(best_x, dtype=float)
    n_test = None
    for m in group_models:
        k = _safe_name(m)
        keys.append(k)
        entries = lib[m]
        xs = [np.asarray(x_m, dtype=float) for x_m, _col in entries]
        d = len(specs_by_model[m])
        arrays[f'x__{k}'] = (np.vstack(xs) if d > 0 else np.zeros((len(xs), 0), dtype=float))
        arrays[f'cols__{k}'] = np.column_stack([np.asarray(col, dtype=float)
                                                for _x, col in entries])
        if entry_params is not None and len(entry_params.get(m) or []) == len(xs):
            params = [list(p) for p in entry_params[m]]
        else:
            params = [_decode_model_params(m, x_m, specs=specs_by_model[m],
                                           base_params=base_params_by_model[m],
                                           search_fidelity=True) for x_m in xs]
        arrays[f'params__{k}'] = np.array([json.dumps(p, default=str) for p in params])
        tests = []
        for p in params:
            col = (test_sink or {}).get(_test_sink_key(m, p))
            tests.append(None if col is None else np.asarray(col, dtype=float))
        lens = {len(t) for t in tests if t is not None}
        n_t = (lens.pop() if len(lens) == 1 else (len(y_test) if y_test is not None else 0))
        n_test = n_t if n_test is None else n_test
        arrays[f'test__{k}'] = np.column_stack(
            [t if (t is not None and len(t) == n_t) else np.full(n_t, np.nan) for t in tests]
        ) if n_t > 0 else np.zeros((0, len(tests)), dtype=float)
        chunk = best_x[offsets[m]:offsets[m] + d]
        sel = -1
        for i, x_m in enumerate(xs):
            if x_m.shape == chunk.shape and np.array_equal(x_m, chunk):
                sel = i
                break
        arrays[f'selected__{k}'] = np.array(sel, dtype=int)
    n = len(np.asarray(actual))
    if oof_cols is not None and len(oof_cols) == len(group_models):
        for m, col in zip(group_models, oof_cols):
            if col is not None and np.asarray(col).ndim == 1 and len(col) == n:
                arrays[f'oof__{_safe_name(m)}'] = np.asarray(col, dtype=float)
    fold = np.full(n, -1, dtype=int)
    if partition is not None:
        for f, held in enumerate(partition):
            fold[np.asarray(held, dtype=int)] = f
    if wf_weights is not None and len(wf_weights) == len(group_models):
        arrays['wf_weights'] = np.asarray(wf_weights, dtype=float)
    arrays['models'] = np.array(list(group_models))
    arrays['keys'] = np.array(keys)
    arrays['y_score'] = np.asarray(actual, dtype=float)
    arrays['y_test'] = (np.asarray(y_test, dtype=float) if y_test is not None
                        else np.zeros(0, dtype=float))
    arrays['fold'] = fold
    arrays['meta'] = np.array(json.dumps(meta, default=str))
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------------------
# R2.b2 - library ensemble (several jointly found configurations per model)
# ------------------------------------------------------------------------------------------

def _selected_entries(group_models: Sequence[str], lib: Dict[str, list], specs_by_model: dict,
                      best_x: np.ndarray) -> Dict[str, int]:
    """ver4-38, R2.b2: for every group model the index of the library entry whose coordinates
    equal that model's slice of `best_x`, or -1 when none does. Pure.

    Why: the library ensemble keeps the selected entry of each model as its PRIMARY (the model
    Step 9 refits), so it must know which library entry that is. `best_x` is a library
    concatenation by construction for the library strategies; -1 is only seen after a gate
    fallback to an unlisted point and makes the caller skip the library ensemble."""
    best_x = np.asarray(best_x, dtype=float)
    out: Dict[str, int] = {}
    off = 0
    for m in group_models:
        d = len(specs_by_model[m])
        chunk = best_x[off:off + d]
        off += d
        out[m] = -1
        for i, (x_m, _col) in enumerate(lib.get(m) or []):
            x_arr = np.asarray(x_m, dtype=float)
            if x_arr.shape == chunk.shape and np.array_equal(x_arr, chunk):
                out[m] = i
                break
    return out


def _prune_library_candidates(group_models: Sequence[str], lib: Dict[str, list],
                              selected: Dict[str, int], y: np.ndarray, *, per_model: int,
                              dead_guard: bool, partition=None) -> List[Tuple[str, int]]:
    """ver4-38, R2.b2: the pruned library as `[(model, entry index), ...]` (models in
    `group_models` order, entries ascending within a model). Pure.

    Per model the `per_model` entries with the lowest single-model out-of-fold MSE against `y`
    are kept, plus the SELECTED entry (`selected[model]`, -1 = none). An entry whose column is
    not finite, has the wrong length, or (with `dead_guard`) is constant is never a candidate -
    the selected entry is added even when it ranks lower, unless it is itself unusable. One
    authority for the pruning: `_library_ensemble_members()` weights this set and
    `tools/joint_replay.py` uses it for the library ceiling CL."""
    y = np.asarray(y, dtype=float)
    cand: List[Tuple[str, int]] = []
    for m in group_models:
        entries = lib.get(m) or []
        scored = []
        for i, (_x, col) in enumerate(entries):
            col = np.asarray(col, dtype=float)
            if col.shape != y.shape or not np.all(np.isfinite(col)):
                continue
            if dead_guard and _is_dead_column(col, y, partition):
                continue
            scored.append((float(np.mean((col - y) ** 2)), i))
        scored.sort()
        keep = [i for _mse, i in scored[:max(1, int(per_model))]]
        sel = int(selected.get(m, -1))
        if sel >= 0 and sel not in keep and sel < len(entries):
            col = np.asarray(entries[sel][1], dtype=float)
            if col.shape == y.shape and np.all(np.isfinite(col)):
                keep.append(sel)
        cand.extend((m, i) for i in sorted(keep))
    return cand


def _library_ensemble_members(group_models: Sequence[str], lib: Dict[str, list],
                              selected: Dict[str, int], y: np.ndarray, weight_fn, *,
                              per_model: int, members_max: int, min_weight: float,
                              dead_guard: bool, partition=None) -> Optional[dict]:
    """ver4-38, R2.b2: choose the members of the library ensemble. Pure (no fits).

    Why: the joint search fits hundreds of out-of-fold columns and keeps one per model; the
    others are a by-product that an ensemble can use. Steps (blueprint R2.b2): (1) the pruned
    library of `_prune_library_candidates()` (the `per_model` best entries per model plus the
    SELECTED entry); (2) `weight_fn(P, y)` (the ensemble-selection estimator) weights the pruned
    matrix `P`; (3) keep at most `members_max` members with weight >= `min_weight`, heaviest
    first, and renormalise to sum 1. `partition` (the CV folds of the out-of-fold columns, or
    None) is handed to the dead-column test.

    Args: `lib` {model: [(x, column), ...]}; `selected` {model: index of the selected entry or
    -1} from `_selected_entries()`; `y` the scoring target of the columns; `weight_fn`
    callable returning a weight vector (one per column of `P`). Returns None when the pruned
    matrix has fewer than 2 columns or no member survives; else `{'members': [{'model',
    'entry', 'weight', 'primary'}...] (heaviest first), 'n_pruned': int, 'weights_raw': list}`.
    Raises whatever `weight_fn` raises."""
    y = np.asarray(y, dtype=float)
    cand = _prune_library_candidates(group_models, lib, selected, y, per_model=per_model,
                                     dead_guard=dead_guard, partition=partition)
    if len(cand) < 2:
        return None
    P = np.column_stack([np.asarray(lib[m][i][1], dtype=float) for m, i in cand])
    w = np.asarray(weight_fn(P, y), dtype=float).ravel()
    if w.shape[0] != len(cand) or not np.all(np.isfinite(w)):
        return None
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 0.0:
        return None
    w = w / total
    order = sorted(range(len(cand)), key=lambda j: (-w[j], j))
    chosen = [j for j in order if w[j] >= float(min_weight)][:max(1, int(members_max))]
    if not chosen:
        return None
    kept = np.array([w[j] for j in chosen], dtype=float)
    kept = kept / float(kept.sum())
    members = []
    for j, wj in zip(chosen, kept):
        m, i = cand[j]
        members.append({'model': m, 'entry': int(i), 'weight': float(wj),
                        'primary': bool(int(selected.get(m, -1)) == int(i))})
    return {'members': members, 'n_pruned': len(cand), 'weights_raw': w.tolist()}


def _wf_split_members(group_models: Sequence[str], lib: Dict[str, list],
                      selected: Dict[str, int], y: np.ndarray, w_fixed: Dict[str, float], *,
                      per_model: int, dead_guard: bool, partition=None) -> Optional[dict]:
    """ver4-47, R6: the members of the weight-first library ensemble. Pure (no fits).

    Same pruned library as `_library_ensemble_members()` (the `per_model` lowest-MSE entries of a
    model plus its SELECTED entry), but nothing is re-estimated: model `m` keeps its FIXED
    weight `w_fixed[m]`, split equally over its kept entries. A model with weight 0 contributes
    no member. Returns the schema of `_library_ensemble_members()` (`members` heaviest first,
    `n_pruned`, `weights_raw`), or None when no member can be formed (the caller then keeps the
    ordinary one-configuration-per-model result)."""
    y = np.asarray(y, dtype=float)
    cand = _prune_library_candidates(group_models, lib, selected, y, per_model=per_model,
                                     dead_guard=dead_guard, partition=partition)
    members = []
    for m in group_models:
        wm = float(w_fixed.get(m, 0.0))
        if not np.isfinite(wm) or wm <= 0.0:
            continue
        idx = [i for (mm, i) in cand if mm == m]
        if not idx:
            return None
        for i in idx:
            members.append({'model': m, 'entry': int(i), 'weight': wm / len(idx),
                            'primary': bool(int(selected.get(m, -1)) == int(i))})
    if not members:
        return None
    members.sort(key=lambda d: (-d['weight'], d['model'], d['entry']))
    tot = float(sum(d['weight'] for d in members))
    for d in members:
        d['weight'] = float(d['weight']) / tot
    return {'members': members, 'n_pruned': len(cand),
            'weights_raw': [float(d['weight']) for d in members]}


# ------------------------------------------------------------------------------------------
# R1.c / R1.d - parity guards (once per task, per model)
# ------------------------------------------------------------------------------------------

def _column_corr(a, b) -> float:
    """Pearson correlation of two prediction columns; NaN for a constant or non-finite
    column (a parity guard then fails, which is the safe answer)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 3 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        return math.nan
    if float(np.std(a)) <= 0.0 or float(np.std(b)) <= 0.0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _parity_guards(*, group_models: Sequence[str], specs_by_model: dict,
                   base_params_by_model: dict, seed_point: np.ndarray, train, valid, test,
                   runner, RESULT_NAME, runner_ctx: dict, want_stub: bool, want_engine: bool,
                   stub_min: float, engine_min: float) -> dict:
    """ver4-36, R1.c / R1.d: decide, once per task and per model, whether the 1-row test stub
    and the exact kernel engine may be used for the SEARCH fits.

    Why: both shortcuts claim to leave the column the search reads unchanged (the stub drops
    test rows the search never reads; the engine replaces MCMC by the closed-form posterior
    of the same model). The claim is cheap to check on the one column every task fits anyway,
    the untuned seed column, so it is checked: the reference is the seed column of a normal
    fit (`train` -> `valid`, full `test`, search-fidelity parameters, the runner); the stub
    column is the same fit with `test.iloc[:1]`; the engine column is
    `kernel_effects.exact_kernel_column()` at the same parameters. A shortcut is kept for a
    model iff corr(reference, shortcut) >= its minimum (`stub_min`, default 0.995;
    `engine_min`, default 0.98); a failed fit, a non-finite or constant column, or an
    engine exception turns that shortcut off for that model with a NOTE (never a task
    failure).

    Cost: one reference fit and one stub fit per eligible model (counted in the `search`
    stage; the stub fit is stored in the fit cache under its own key, so a cached search's
    untuned column is then free) and one exact fit per engine model. Models are fitted in
    one parallel batch each when `runner_ctx['n_jobs'] > 1`.

    Args: the usual task data and fit settings; `runner_ctx` the dict `_run_units()` takes
    (n_jobs, worker_init, worker_init_args, cache, counter, optional `keep_test` = the library
    dump needs the test prediction of the reference fit) WITHOUT `fit_flags` and `stage`;
    `want_stub` / `want_engine` which shortcuts the user asked for; `stub_min` /
    `engine_min` the thresholds. Models are judged on their own: a model outside
    `model_registry.search_test_stub_safe` / `exact_kernel_search_capable` is never eligible.

    Returns `{'stub_models': set, 'engine_models': set, 't_seed': {model: seconds per
    fit of the shortcut path}, 'corr': {model: {'stub': c, 'engine': c}}}`. Never raises for
    a failed fit (the worker already printed why)."""
    from model_registry import search_test_stub_safe, exact_kernel_search_capable

    result = {'stub_models': set(), 'engine_models': set(), 't_seed': {}, 'corr': {}}
    stub_candidates = [m for m in group_models if want_stub and search_test_stub_safe(m)]
    engine_candidates = [m for m in group_models if want_engine and exact_kernel_search_capable(m)]
    for m in group_models:
        if want_stub and m not in stub_candidates:
            print(f"[joint_tuning] NOTE: search_test_stub does not apply to {m!r} (its search "
                  f"fit is not one that masks test rows) - it keeps the full test frame.")
        if want_engine and m not in engine_candidates:
            print(f"[joint_tuning] NOTE: search_engine='exact_kernel' does not apply to {m!r} "
                  f"(no closed-form engine for it) - its search fits stay real fits.")
    need_ref = [m for m in group_models if m in stub_candidates or m in engine_candidates]
    if not need_ref:
        return result
    if valid is None or len(valid) == 0:
        print("[joint_tuning] NOTE: this task has no validation rows, so the stub / exact-engine "
              "parity check cannot run - both are off for this task.")
        return result
    offsets, _off = {}, 0
    for m in group_models:
        offsets[m] = _off
        _off += len(specs_by_model[m])
    seed_point = np.asarray(seed_point, dtype=float)
    params = {m: _decode_model_params(
        m, seed_point[offsets[m]:offsets[m] + len(specs_by_model[m])],
        specs=specs_by_model[m], base_params=base_params_by_model[m], search_fidelity=True)
        for m in need_ref}
    counter = runner_ctx.get('counter')

    def _run(units):
        snap = _counter_snapshot(counter, need_ref)
        fitted = _run_units(
            units, runner=runner, RESULT_NAME=RESULT_NAME, test=test,
            n_jobs=runner_ctx.get('n_jobs', 1), worker_init=runner_ctx.get('worker_init'),
            worker_init_args=runner_ctx.get('worker_init_args', ()),
            cache=runner_ctx.get('cache'), stop_on_failure=False, counter=counter, stage='search')
        return fitted, _seed_times(counter, snap, need_ref)

    ref_units = [_FitUnit(model=m, params=params[m], split_tag='valid', train=train, valid=valid,
                          fidelity='search') for m in need_ref]
    if runner_ctx.get('keep_test'):
        # library dump: this reference fit is cached and later serves as the seed column of
        # the library search, so it must carry the test prediction like any other search fit
        for _u in ref_units:
            _u.keep_test = True
    ref_res, t_ref = _run(ref_units)
    ref_col = {}
    for m, res in zip(need_ref, ref_res):
        pv = None if res is None else res.get('predicted_valid')
        if pv is not None and len(pv) == len(valid):
            ref_col[m] = np.asarray(pv, dtype=float)
    stub_list = [m for m in stub_candidates if m in ref_col]
    stub_col, t_stub = {}, {}
    if stub_list:
        stub_units = [_FitUnit(model=m, params=params[m], split_tag='valid', train=train,
                               valid=valid, fidelity='search', test_stub=True)
                      for m in stub_list]
        stub_res, t_stub = _run(stub_units)
        for m, res in zip(stub_list, stub_res):
            pv = None if res is None else res.get('predicted_valid')
            if pv is not None and len(pv) == len(valid):
                stub_col[m] = np.asarray(pv, dtype=float)
    for m in need_ref:
        result['corr'][m] = {}
        if m not in ref_col:
            print(f"[joint_tuning] NOTE: the reference seed fit of {m!r} failed, so its stub / "
                  f"exact-engine parity cannot be checked - both are off for it.")
            continue
        if m in stub_candidates:
            c = _column_corr(ref_col[m], stub_col[m]) if m in stub_col else math.nan
            result['corr'][m]['stub'] = c
            if np.isfinite(c) and c >= float(stub_min):
                result['stub_models'].add(m)
                result['t_seed'][m] = float(t_stub.get(m, 0.0))
                print(f"[joint_tuning] stub parity {m}: corr(full test, 1-row test)={c:.5f} >= "
                      f"{stub_min:g} - the 1-row test stub is ON for its search fits.")
            else:
                print(f"[joint_tuning] NOTE: stub parity {m}: corr(full test, 1-row test)="
                      f"{c:.5f} < {stub_min:g} (or not computable) - the 1-row test stub is OFF "
                      f"for {m!r}; its search fits keep the full test frame.")
        if m in engine_candidates:
            t0_exact = time.perf_counter()
            col_exact = None
            try:
                from .kernel_effects import exact_kernel_column
                col_exact = np.asarray(
                    exact_kernel_column(m, train, valid, test, params[m])['predicted_valid'],
                    dtype=float)
            except Exception as exc:
                print(f"[joint_tuning] NOTE: the exact kernel engine failed on the seed fit of "
                      f"{m!r} ({type(exc).__name__}: {exc}) - search_engine='exact_kernel' is "
                      f"OFF for it.")
            secs_exact = time.perf_counter() - t0_exact
            if col_exact is not None:
                _acct_unit = _FitUnit(model=m, params=params[m], split_tag='valid', train=train,
                                      valid=valid, fidelity='search', engine='exact_kernel')
                _account_result(counter, _acct_unit, {'predicted_valid': col_exact,
                                                       '_secs': secs_exact,
                                                       '_engine': 'exact_kernel'})
                if counter is not None:
                    counter['search'] = counter.get('search', 0) + 1
            c = _column_corr(ref_col[m], col_exact) if col_exact is not None else math.nan
            result['corr'][m]['engine'] = c
            if np.isfinite(c) and c >= float(engine_min):
                result['engine_models'].add(m)
                result['t_seed'][m] = float(secs_exact)
                print(f"[joint_tuning] engine parity {m}: corr(BGLR seed column, exact seed "
                      f"column)={c:.5f} >= {engine_min:g} ({secs_exact * 1000:.0f} ms) - the "
                      f"exact kernel engine is ON for its search fits.")
            elif col_exact is not None:
                print(f"[joint_tuning] NOTE: engine parity {m}: corr(BGLR seed column, exact seed "
                      f"column)={c:.5f} < {engine_min:g} (or not computable) - the exact kernel "
                      f"engine is OFF for {m!r}; its search fits stay real fits.")
    return result


# ------------------------------------------------------------------------------------------
# R1.e - asynchronous per-model library search
# ------------------------------------------------------------------------------------------

def _async_budget(d_m: int, per_dim: int, max_evals: int) -> int:
    """ver4-36, R1.e: number of library COLUMNS (seed column included) the asynchronous
    search builds for a model with `d_m` tunable dimensions:
    `clip(1 + library_evals_per_dim * d_m, 4, library_max_evals)`, at least 2 and 1 for a
    model with nothing tunable (it only keeps its seed column)."""
    if d_m <= 0:
        return 1
    return int(max(2, min(int(max_evals), max(_JOINT_LIBRARY_MIN_EVALS, 1 + int(per_dim) * int(d_m)))))


def _async_round_plan(proposals: int, init_points: int, q: int) -> List[int]:
    """ver4-36, R1.e: sizes of the successive proposal rounds of one model's Phase-1
    pipeline for `proposals` new columns - a first round of `init_points` random points, then
    rounds of `q` guided points, the last one truncated. `[]` for `proposals <= 0`."""
    sizes: List[int] = []
    remaining = int(proposals)
    first = True
    while remaining > 0:
        take = min(int(init_points) if first else int(q), remaining)
        sizes.append(take)
        remaining -= take
        first = False
    return sizes


def _async_library_search(*, group_models, specs_by_model, base_params_by_model, seed_point,
                          fit_ctx: dict, score_fn: '_ColumnScorer', settings: dict, state,
                          seed: int) -> Tuple[np.ndarray, float, dict]:
    """ver4-36, R1.e: `search_strategy='async_library'` - one independent search pipeline per
    model, no global barrier between models.

    Why: in `parallel_library` every round waits for its slowest unit (BayesB), so the cheap
    models idle the CPUs while it runs, and a block search is serial across models. A model's
    column depends only on its OWN hyperparameters (F2) and the ensemble score is a pure
    function of the K columns, so a model can propose, fit and re-score its own candidates
    against FIXED reference columns (the untuned seed columns) without knowing what any other
    model is doing - the pipelines never talk to each other and a pipeline that finishes early
    simply frees workers for the others.

    Flow
      R0       the untuned (seed) column of every model, one batch (`_fit_requests_batch()`).
      Phase 1  per model m with d_m > 0: `_async_budget()` columns in rounds of
               `_async_round_plan()` - round 1 random (`block_init_points`), later rounds a
               constant-liar GP batch of `library_q` points from `suggest_batch()` conditioned
               on m's OWN library re-scored against the seed columns of the other models. All
               pipelines run at once on one reusable loky executor (`_loky_executor()`; inline
               when `n_jobs <= 1` or no executor can be made - identical results); the results
               of a model are processed in submission order.
      Phase 2  the library selection (the better of `_coordinate_select()` from the current
               columns and from each model's single-model best) becomes the new reference;
               `library_refine_rounds` barrier rounds follow, each model proposing `library_q`
               points conditioned on the new reference; one more selection.
      Final    the caller applies `selection_rule` ('greedy' = the selection returned here,
               'bbc' = `_bbc_analyse()`) and the gate.

    Determinism (A-1e.1): the proposals of model m depend only on m's own ordered history
    (Phase 1) or on the Phase-1 selection (Phase 2); every pipeline scores with its own
    `_ColumnScorer.clone()` and failure state; per-model seeds are `seed + crc32(m) % 10000 +
    round` (I9). `n_jobs = 1` and `n_jobs = N` therefore return the same library and the same
    `best_x`. NOT deterministic, and disclosed as for every other strategy: the wall-clock
    budget (`time_budget_s`), which stops starting new rounds; `search_patience` and
    `freeze_low_weight` are not used by this strategy (the per-model budget replaces them).

    Cost of crossing processes: a unit carries its frames (a loky executor has no shared
    memory map), which is negligible next to an MCMC fit; units answered by the exact engine
    are therefore run in this process, where they cost milliseconds.

    Args: `group_models`, `specs_by_model`, `base_params_by_model`, `seed_point` as in
    `_parallel_library_search()`; `fit_ctx` from `_make_column_context()` (its `runner_ctx`
    carries cache, counter, `n_jobs`, `fit_flags`, `test_sink`); `score_fn` a
    `_ColumnScorer`; `settings` keys: library_q, block_init_points, block_tol,
    library_evals_per_dim, library_max_evals, library_refine_rounds, weight_estimator,
    estimator_settings, objective_mode, diversity_method, noise_aware_surrogate,
    time_budget_s, time_budget_t0, eta_plan, t_seed; `state` the shared failure state; `seed`
    the search seed.

    Returns `(best_x, best_score, stats)`; `stats` = `evals_by_model`, `rounds_by_model`,
    `lib_sizes`, `refine` (rounds run), `critical_path_s` (longest per-model chain of fit
    seconds plus the Phase-2 barriers), `busy_s` / `wall_s` / `utilisation` (worker-busy
    seconds over wall seconds x n_jobs, A-1e.2), `stop`, `seed_ens`, `swaps`, and `lib` /
    `lib_cur` for the caller. Returns `(seed_point, -inf, stats)` when an untuned column cannot
    be fitted (the caller falls back to the monolithic search). Never raises for a failed
    fit."""
    t_start = time.perf_counter()
    dims = {m: len(specs_by_model[m]) for m in group_models}
    offsets, _off = {}, 0
    for m in group_models:
        offsets[m] = _off
        _off += dims[m]
    seed_point = np.asarray(seed_point, dtype=float)
    actual = np.asarray(fit_ctx['actual'], dtype=float)
    rc = dict(fit_ctx.get('runner_ctx') or {})
    counter = rc.get('counter')
    cache = rc.get('cache')
    n_jobs = max(1, int(rc.get('n_jobs') or 1))
    flags = rc.get('fit_flags')
    sink = rc.get('test_sink')
    runner, RESULT_NAME, test = fit_ctx['runner'], fit_ctx['RESULT_NAME'], fit_ctx['test']
    cv_mode = fit_ctx.get('objective_data') == 'cv_train_valid'
    fold_frames = _cv_fold_frames(fit_ctx['tv'], fit_ctx['partition']) if cv_mode else None
    _dead_part = list(fit_ctx['partition']) if cv_mode else None   # ver4-38: fold-wise dead test
    n_rows = int(len(actual))
    q = max(1, int(settings.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)))
    init_points = max(1, int(settings.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)))
    tol = float(settings.get('block_tol', _JOINT_BLOCK_TOL_DEFAULT))
    per_dim = int(settings.get('library_evals_per_dim', _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT))
    max_evals = int(settings.get('library_max_evals', _JOINT_LIBRARY_MAX_EVALS_DEFAULT))
    refine_rounds = max(0, int(settings.get('library_refine_rounds',
                                            _JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT)))
    est = dict(settings.get('estimator_settings') or {})
    est_name = settings.get('weight_estimator', _JOINT_WEIGHT_ESTIMATOR_DEFAULT)
    bagged = bool(getattr(score_fn, 'bagged', False))
    noise_aware = bool(settings.get('noise_aware_surrogate', _JOINT_NOISE_AWARE_DEFAULT))
    budget_s = float(settings.get('time_budget_s', _JOINT_TIME_BUDGET_DEFAULT) or 0.0)
    budget_hook = (_time_budget_stop(settings['time_budget_t0'], budget_s)
                   if settings.get('time_budget_t0') is not None else None)
    acquisition = ((lambda: ExpectedImprovement(xi=0.01)) if ExpectedImprovement is not None
                   else None)
    # ver4-38 (all three default to the ver4-37 behaviour): R1.c the per-model stop rule,
    # R1.b CV fold racing. `fold_ids_all` are the fold numbers of the CV objective.
    stop_rule = settings.get('library_stop_rule', _JOINT_LIBRARY_STOP_RULE_DEFAULT)
    freeze_low = float(settings.get('freeze_low_weight', _JOINT_FREEZE_LOW_WEIGHT_DEFAULT) or 0.0)
    fold_ids_all = list(range(len(fold_frames))) if cv_mode else [0]
    race_idx: List[int] = []
    race_rows = np.array([], dtype=int)
    race_sub: List[np.ndarray] = []
    race_margin = float(settings.get('racing_margin', _JOINT_RACING_MARGIN_DEFAULT))
    race_penalty = float(settings.get('racing_penalty', _JOINT_RACING_PENALTY_DEFAULT))
    if settings.get('cv_racing', _JOINT_CV_RACING_DEFAULT):
        n_race = int(settings.get('racing_folds', _JOINT_RACING_FOLDS_DEFAULT))
        if not cv_mode:
            print("[GP] NOTE: Joint Bayesian optimisation cv_racing needs objective_data="
                  "'cv_train_valid' - racing is off for this task.")
        elif n_race < 1 or n_race >= len(fold_frames):
            print(f"[GP] NOTE: Joint Bayesian optimisation racing_folds={n_race} must be between "
                  f"1 and cv_folds-1={len(fold_frames) - 1} - racing is off for this task.")
        else:
            race_idx = list(range(n_race))
            race_sub = [np.asarray(fold_frames[f][0], dtype=int) for f in race_idx]
            race_rows = np.sort(np.concatenate(race_sub))
            if race_rows.size < _JOINT_RACING_MIN_ROWS:
                print(f"[GP] NOTE: Joint Bayesian optimisation cv_racing: the {n_race} raced "
                      f"fold(s) hold only {race_rows.size} scoring row(s) (need >= "
                      f"{_JOINT_RACING_MIN_ROWS}) - every proposal is promoted, i.e. racing is "
                      f"off for this task.")
                race_idx = []
    racing = bool(race_idx)
    stats = {'evals_by_model': {m: 0 for m in group_models},
             'rounds_by_model': {m: 0 for m in group_models}, 'lib_sizes': {}, 'refine': 0,
             'critical_path_s': 0.0, 'busy_s': 0.0, 'wall_s': 0.0, 'utilisation': 0.0,
             'stop': '', 'seed_ens': None, 'swaps': 0}
    if racing:
        stats['raced'] = {m: {'raced': 0, 'promoted': 0, 'rejected': 0} for m in group_models}
    if stop_rule == 'value':
        stats['stopped'] = {}
    race_obs: Dict[str, list] = {m: [] for m in group_models}

    def _lib_seed(m, r):
        return int(seed) + (zlib.crc32(m.encode('utf-8')) % 10_000) + int(r)

    # ---- R0: the untuned column of every model, one batch ------------------------------
    chunks0 = {m: seed_point[offsets[m]:offsets[m] + dims[m]] for m in group_models}
    snap0 = _counter_snapshot(counter, group_models)
    cols0 = _fit_requests_batch([(m, chunks0[m]) for m in group_models], ctx=fit_ctx)
    failed0 = [m for m, col in zip(group_models, cols0) if col is None]
    if failed0:
        print(f"[GP] NOTE: Joint Bayesian optimisation async-library search could not fit the "
              f"initial (untuned) column for {failed0!r} - falling back to the monolithic search.")
        return seed_point, -math.inf, stats
    lib = {m: [(np.array(chunks0[m], dtype=float), col)] for m, col in zip(group_models, cols0)}
    seen = {m: {np.array(chunks0[m], dtype=float).tobytes()} for m in group_models}
    cur = {m: 0 for m in group_models}
    ref_cols = {m: lib[m][0][1] for m in group_models}
    # ver4-38, R2.b1: a model whose untuned column is constant leaves the scored group (no-op
    # unless `dead_column_guard`); its pipeline is not started and the caller gives it weight 0.
    stats['dead_models'] = score_fn.drop_dead_seed_models(ref_cols, where='async-library R0')
    cur_score = float(score_fn(dict(ref_cols)))
    stats['seed_ens'] = cur_score
    plan = {m: _async_round_plan(_async_budget(dims[m], per_dim, max_evals) - 1, init_points, q)
            for m in group_models}
    _log_eta('async_library', group_models,
             {'fits_by_model': {m: (1 + sum(plan[m]) + (refine_rounds * q if dims[m] else 0))
                                * (len(fold_frames) if cv_mode else 1) for m in group_models},
              'rounds_by_model': {m: 1 + len(plan[m]) + (refine_rounds if dims[m] else 0)
                                  for m in group_models}},
             _seed_times(counter, snap0, group_models, settings.get('t_seed')), n_jobs)

    # ---- execution machinery -----------------------------------------------------------
    executor_box = [None]
    if n_jobs > 1:
        try:
            executor_box[0] = _loky_executor(n_jobs, rc.get('worker_init'),
                                             rc.get('worker_init_args', ()))
        except Exception as exc:
            print(f"[GP] NOTE: Joint Bayesian optimisation async-library search could not start "
                  f"its worker pool ({type(exc).__name__}: {exc}) - the same pipelines run "
                  f"serially in this process (identical results, only slower).")

    def _executor_failed(exc):
        if executor_box[0] is not None:
            print(f"[GP] NOTE: Joint Bayesian optimisation async-library search lost its worker "
                  f"pool ({type(exc).__name__}: {exc}) - the remaining units run serially in "
                  f"this process (identical results, only slower).")
        executor_box[0] = None

    def _submit(unit):
        """Cache hit -> resolved future; else the fit, inline or on the pool. Returns
        `(future, cache_key, was_cached)`."""
        key = (cache.key(unit.model, unit.params, _unit_fidelity_key(unit), unit.split_tag)
               if cache is not None else None)
        cached = cache.get(key) if cache is not None else None
        fut = Future()
        if cached is not None:
            fut.set_result(cached)
            return fut, key, True
        if executor_box[0] is None or unit.engine == 'exact_kernel':
            fut.set_result(_fit_unit_worker(runner, RESULT_NAME, test, unit))
            return fut, key, False
        test_arg = test.iloc[:1] if (unit.test_stub and not unit.keep_test) else test
        try:
            return executor_box[0].submit(_fit_unit_worker, runner, RESULT_NAME, test_arg,
                                          unit), key, False
        except Exception as exc:
            _executor_failed(exc)
            fut.set_result(_fit_unit_worker(runner, RESULT_NAME, test, unit))
            return fut, key, False

    def _resolve(part):
        """`(result, seconds)` of one submitted unit; a failed pool re-runs it inline."""
        fut, unit, key, was_cached = part
        try:
            res = fut.result()
        except Exception as exc:
            _executor_failed(exc)
            res = _fit_unit_worker(runner, RESULT_NAME, test, unit)
        if was_cached:
            return res, 0.0
        secs = _account_result(counter, unit, res)
        if counter is not None:
            counter['search'] = counter.get('search', 0) + 1
        if cache is not None and res is not None:
            cache.put(key, res)
        return res, float(secs or 0.0)

    def _open_request(m, x, folds=None):
        """Decode `x` for model `m`, build its fit unit(s) and submit them. `folds` (CV objective
        only; default = every fold) lists the fold numbers to fit - the race stage of CV racing
        passes the raced folds; a later full request for the same `x` then hits the fit cache for
        them (same model, params, tag, fidelity key)."""
        params = _decode_model_params(
            m, x, specs=specs_by_model[m], base_params=base_params_by_model[m],
            search_fidelity=True)
        fold_ids = list(fold_ids_all) if folds is None else list(folds)
        if cv_mode:
            units = [_FitUnit(model=m, params=params,
                              split_tag=f"cv{f}@{int(fit_ctx['cv_seed'])}",
                              train=fold_frames[f][1], valid=fold_frames[f][2], fidelity='cv',
                              cv_weights=True)
                     for f in fold_ids]
        else:
            units = [_FitUnit(model=m, params=params, split_tag='valid', train=fit_ctx['train'],
                              valid=fit_ctx['valid'], fidelity='search')]
        _apply_fit_flags(units, flags, 'search')
        return {'model': m, 'x': x, 'params': params, 'units': units, 'fold_ids': fold_ids,
                'parts': [_part(u) for u in units]}

    def _part(unit):
        fut, key, was_cached = _submit(unit)
        return fut, unit, key, was_cached

    def _close_request(req, partial=False):
        """Collect a request's units in submission order -> `(column | None, max secs,
        total secs)`; the column of a CV request is the out-of-fold vector. With `partial`
        (a race-stage request) the column is NaN outside the raced folds, finite on them, and
        nothing is written to the test sink (a partial column is never a library entry)."""
        results, secs_list = [], []
        for part in req['parts']:
            res, secs = _resolve(part)
            results.append(res)
            secs_list.append(secs)
        longest, total = (max(secs_list) if secs_list else 0.0), float(sum(secs_list))
        if any(r is None for r in results):
            return None, longest, total
        if cv_mode:
            col = np.full(n_rows, np.nan, dtype=float)
            done_rows = []
            for f, res in zip(req['fold_ids'], results):
                held = fold_frames[f][0]
                pv = res.get('predicted_valid')
                if pv is None or len(pv) != len(held):
                    return None, longest, total
                col[held] = np.asarray(pv, dtype=float)
                done_rows.append(np.asarray(held, dtype=int))
            check = col if not partial else col[np.concatenate(done_rows)]
            if not np.all(np.isfinite(check)):
                return None, longest, total
        else:
            pv = results[0].get('predicted_valid')
            if pv is None or len(pv) == 0 or len(pv) != n_rows:
                return None, longest, total
            col = np.asarray(pv, dtype=float)
        if sink is not None and not partial:
            _record_test_columns(sink, (req['model'], req['params']), results)
        return col, longest, total

    def _single_score_factory():
        indiv_state = {'worst_finite': None}

        def _single_score(col):
            return _combine_score(np.array([1.0]), np.asarray(col, dtype=float)[:, None], actual,
                                  ['single'], settings.get('objective_mode', 'ensemble_score'),
                                  0.0, settings.get('diversity_method',
                                                    _JOINT_DIVERSITY_METHOD_DEFAULT),
                                  state=indiv_state)
        return _single_score

    def _propose(m, r, scorer, base_cols, n_points, guided):
        """`n_points` unseen unit-cube points for model `m` (random when not `guided`)."""
        bounds = [(0.0, 1.0)] * dims[m]
        if not guided:
            points = suggest_batch(bounds, [], q=n_points, seed=_lib_seed(m, r))
        else:
            observations = []
            for x_e, col_e in lib[m][-_JOINT_LIBRARY_RESCORE_CAP:]:
                trial = dict(base_cols)
                trial[m] = col_e
                sc = float(scorer.score(trial))
                if np.isfinite(sc):
                    observations.append((x_e, sc))
            # ver4-38, R1.b: proposals that lost the race are shown to the surrogate at their
            # offset-calibrated race score (empty unless cv_racing).
            for x_r, sc_r in race_obs[m][-_JOINT_LIBRARY_RESCORE_CAP:]:
                observations.append((x_r, sc_r))
            try:
                points = suggest_batch(bounds, observations, q=n_points, seed=_lib_seed(m, r),
                                       liar='max', acquisition_factory=acquisition,
                                       noise_aware=noise_aware)
            except (ImportError, RuntimeError) as exc:
                print(f"[GP] NOTE: Joint Bayesian optimisation async-library search could not "
                      f"build a surrogate suggestion for {m!r} in round {r} "
                      f"({type(exc).__name__}: {exc}) - proposing random points for it.")
                points = suggest_batch(bounds, [], q=n_points, seed=_lib_seed(m, r))
        fresh = []
        for x in points:
            x = np.array(x, dtype=float)
            key = x.tobytes()
            if key in seen[m]:
                continue
            seen[m].add(key)
            fresh.append(x)
        return fresh

    # ---- Phase 1: independent per-model pipelines ---------------------------------------
    # Longest chain first: the pool is a FIFO queue, so the slowest model's fits must be
    # submitted before the cheap models' fits or its sequential rounds start late and set the
    # wall time (measured with a sleep proxy: BayesB queued behind 9 cheap fits finished its
    # last round 3 s after the cheap models were done). Ordering only changes WHEN a pipeline
    # starts - every pipeline's points, seeds and columns are independent of the others -
    # so results are identical at any order; ties keep the group order.
    _t_fit = _seed_times(counter, snap0, group_models, settings.get('t_seed'))
    _start_order = sorted((m for m in group_models
                           if dims[m] and plan[m] and m not in score_fn.inactive),
                          key=lambda m: (-_t_fit.get(m, 0.0) * len(plan[m]),
                                         group_models.index(m)))
    pipelines = {}
    for m in _start_order:
        pipelines[m] = {'m': m, 'scorer': score_fn.clone({'worst_finite': None}), 'round': 0,
                        'pending': [], 'done': False, 'crit': 0.0, 'dropped': 0,
                        'sizes': list(plan[m]), 'stage': 'full'}
        if stop_rule == 'value':
            # ver4-38, R1.c: the incumbent of model m is its best column scored against the FIXED
            # reference columns; every test below uses only m's own ordered history (I9).
            sc0 = float(pipelines[m]['scorer'].score(dict(ref_cols)))
            pipelines[m].update(best_prev=sc0, seed_score=sc0, best_col=ref_cols[m],
                                best_x=lib[m][0][0], low_rounds=0)

    def _race_decide(m, race_results, scorer, base_cols):
        """ver4-38, R1.b: decide which raced proposals of model `m` are promoted to the full fold
        set. `race_results` = `[(x, partial column | None)]` (None = a raced fit failed).
        A proposal is promoted iff its score on the raced rows is >= the best score any column
        already in `lib[m]` reaches on THE SAME rows, minus `racing_margin` (both scored by
        `scorer.score_rows`, the same estimator). A rejected proposal is shown to the surrogate
        at `race score + (incumbent full-row score - incumbent race-row score) - racing_penalty`,
        i.e. calibrated to the full-row scale through the incumbent. Returns
        `(promoted x list in submission order, number of failed fits)`."""
        s_inc_race, s_inc_full = -math.inf, -math.inf
        for _x_e, col_e in lib[m]:
            trial = dict(base_cols)
            trial[m] = col_e
            s_inc_race = max(s_inc_race, float(scorer.score_rows(trial, race_rows, race_sub)))
            s_inc_full = max(s_inc_full, float(scorer.score(trial)))
        usable = bool(np.isfinite(s_inc_race) and np.isfinite(s_inc_full))
        promoted, n_failed = [], 0
        stats['raced'][m]['raced'] += len(race_results)
        for x, col_p in race_results:
            if col_p is None:
                n_failed += 1
                continue
            trial = dict(base_cols)
            trial[m] = col_p
            s_p = float(scorer.score_rows(trial, race_rows, race_sub))
            if (not usable) or s_p >= s_inc_race - race_margin:
                promoted.append(x)
            else:
                race_obs[m].append((x, float(s_p + (s_inc_full - s_inc_race) - race_penalty)))
                stats['raced'][m]['rejected'] += 1
        stats['raced'][m]['promoted'] += len(promoted)
        return promoted, n_failed

    def _stop_check(pl, added):
        """ver4-38, R1.c: why model `pl['m']`'s pipeline should stop now, or '' to go on.
        (a) every column of the model is dead; (b) the best proposal of each of the last
        `_JOINT_STOP_RULE_PATIENCE` rounds gained < `block_tol` over the incumbent (the
        observed gain, or `_posterior_gain` with the noise-aware surrogate); (c) with
        `freeze_low_weight` > 0 and a bagged estimator: the model's bagged weight among the
        reference columns plus its own best is < `freeze_low_weight` and its best gain over its
        untuned column is < `block_tol`. Pure in m's own history and the fixed reference
        columns, so the result does not depend on `n_jobs` (A-1c.3)."""
        m, scorer = pl['m'], pl['scorer']
        if all(_is_dead_column(col, actual, _dead_part) for _x_e, col in lib[m]):
            return (f"every one of its {len(lib[m])} column(s) is constant (dead) - it carries "
                    f"no signal, so further proposals are wasted")
        if not added:
            return ''
        prev_entries = [(x_e, col_e) for x_e, col_e in lib[m] if all(
            x_e is not x_a for x_a, _c in added)]
        new_scores = []
        for x_a, col_a in added:
            trial = dict(ref_cols)
            trial[m] = col_a
            new_scores.append(float(scorer.score(trial)))
        i_best = int(np.argmax(new_scores))
        gain = new_scores[i_best] - pl['best_prev']
        if noise_aware and prev_entries:
            hist = []
            for x_e, col_e in prev_entries[-_JOINT_LIBRARY_RESCORE_CAP:]:
                trial = dict(ref_cols)
                trial[m] = col_e
                hist.append((x_e, float(scorer.score(trial))))
            trials = [(x_a, sc) for (x_a, _c), sc in zip(added, new_scores)]
            gain = _posterior_gain(hist, trials, added[i_best][0], pl['best_x'],
                                   _lib_seed(m, 2000 + pl['round']), gain)
        if new_scores[i_best] > pl['best_prev']:
            pl['best_prev'], pl['best_col'], pl['best_x'] = (new_scores[i_best],
                                                             added[i_best][1], added[i_best][0])
        pl['low_rounds'] = pl['low_rounds'] + 1 if (np.isfinite(gain) and gain < tol) else 0
        if pl['low_rounds'] >= _JOINT_STOP_RULE_PATIENCE:
            return (f"its best proposal gained < block_tol={tol:g} over the incumbent in "
                    f"{pl['low_rounds']} consecutive round(s)")
        if freeze_low > 0.0 and bagged and (pl['best_prev'] - pl['seed_score']) < tol:
            trial = dict(ref_cols)
            trial[m] = pl['best_col']
            active = scorer.active_models
            if m in active:
                w = np.asarray(_robust_weight_fn(
                    est_name, est, n_bags=int(est.get('score_bags', _JOINT_SCORE_BAGS_DEFAULT)),
                    seed=int(seed))(np.column_stack([trial[a] for a in active]), actual),
                    dtype=float)
                w_m = float(w[active.index(m)])
                if w_m < freeze_low:
                    return (f"its bagged weight is {w_m:.3f} < freeze_low_weight="
                            f"{freeze_low:g} and tuning gained < block_tol={tol:g} over its "
                            f"untuned column")
        return ''

    def _start_round(pl):
        m = pl['m']
        if pl['round'] >= len(pl['sizes']):
            pl['done'] = True
            return
        if pl['round'] > 0 and budget_hook is not None and budget_hook([]):
            pl['done'] = True
            stats['stop'] = 'time budget'
            return
        r = pl['round'] + 1
        points = _propose(m, r, pl['scorer'], ref_cols, pl['sizes'][pl['round']], guided=(r > 1))
        if not points:
            pl['done'] = True
            return
        pl['round'] = r
        pl['stage'] = 'race' if racing else 'full'
        pl['pending'] = [_open_request(m, x, folds=(race_idx if racing else None))
                         for x in points]

    def _finish_race(pl):
        """Close a race-stage round: decide the promotions and submit the promoted proposals at
        the full fold set (the raced folds are fit-cache hits). Returns the longest race fit."""
        m = pl['m']
        race_results, longest_r = [], 0.0
        for req in pl['pending']:
            col, longest, total = _close_request(req, partial=True)
            longest_r = max(longest_r, longest)
            stats['busy_s'] += total
            race_results.append((req['x'], col))
        promoted, n_failed = _race_decide(m, race_results, pl['scorer'], ref_cols)
        stats['evals_by_model'][m] += len(race_results) - len(promoted)
        pl['dropped'] += n_failed
        pl['pending'] = [_open_request(m, x) for x in promoted]
        pl['stage'] = 'finish'
        return longest_r

    def _complete_round(pl, added):
        m = pl['m']
        stats['rounds_by_model'][m] = pl['round']
        pl['pending'] = []
        pl['stage'] = 'full'
        reason = _stop_check(pl, added) if stop_rule == 'value' else ''
        if reason:
            pl['done'] = True
            stats['stopped'][m] = f"round {pl['round']}: {reason}"
            print(f"[GP] {'WARN' if reason.startswith('every one') else 'NOTE'}: Joint Bayesian "
                  f"optimisation async-library stop rule: {m!r} stops proposing after round "
                  f"{pl['round']} with {len(lib[m])} column(s) - {reason}; the workers it frees "
                  f"go to the other models and its library stays available for the selection.")
        else:
            _start_round(pl)
        if pl['done']:
            print(f"[GP] Joint Bayesian optimisation async-library search: {m!r} finished "
                  f"phase 1 with {len(lib[m])} column(s) in {pl['round']} round(s)"
                  + (f" ({pl['dropped']} failed fit(s) dropped)" if pl['dropped'] else "")
                  + " - waiting for the other models before the selection.")

    for pl in pipelines.values():
        _start_round(pl)
    while any(not pl['done'] for pl in pipelines.values()):
        outstanding = [part[0] for pl in pipelines.values() for req in pl['pending']
                       for part in req['parts'] if not part[0].done()]
        if outstanding:
            wait(outstanding, return_when=FIRST_COMPLETED)
        for pl in pipelines.values():
            if pl['done'] or not pl['pending']:
                continue
            if not all(part[0].done() for req in pl['pending'] for part in req['parts']):
                continue
            m = pl['m']
            if pl['stage'] == 'race':
                pl['crit'] += _finish_race(pl)
                if pl['pending']:
                    continue                 # wait for the promoted proposals' remaining folds
                _complete_round(pl, [])     # every proposal lost the race: the round is over
                continue
            round_longest = 0.0
            added = []
            for req in pl['pending']:
                col, longest, total = _close_request(req)
                round_longest = max(round_longest, longest)
                stats['busy_s'] += total
                stats['evals_by_model'][m] += 1
                if col is None:
                    pl['dropped'] += 1
                    continue
                lib[m].append((req['x'], col))
                added.append((req['x'], col))
            pl['crit'] += round_longest
            _complete_round(pl, added)
    stats['critical_path_s'] = max([pl['crit'] for pl in pipelines.values()] or [0.0])

    # ---- selection (also reused after the refine rounds) --------------------------------
    _single_score = _single_score_factory()

    def _order_by_weight(cols):
        scored = score_fn.active_models      # ver4-38: models dropped by the dead-column guard
        fixed_w = getattr(score_fn, 'fixed_w', None)
        if fixed_w is not None:
            # ver4-42: stage S2 of 'weight_first_cv' - the weights are FIXED (stage S1), so the
            # visiting order follows them instead of re-estimating weights on the columns.
            w = np.asarray([fixed_w[group_models.index(m)] for m in scored], dtype=float)
        else:
            matrix = np.column_stack([cols[m] for m in scored])
            if bagged:
                w = _robust_weight_fn(est_name, est, n_bags=int(est.get('score_bags',
                                                                        _JOINT_SCORE_BAGS_DEFAULT)),
                                      seed=int(seed))(matrix, actual)
            else:
                w = analytic_simplex_weights(matrix, actual)
            w = np.asarray(w, dtype=float)
        return [scored[i] for i in sorted(range(len(scored)), key=lambda i: (-w[i], i))]

    def _select(cur_now):
        cols_now = {m: lib[m][cur_now[m]][1] for m in group_models}
        order = _order_by_weight(cols_now)
        cur_a, score_a, swaps_a = _coordinate_select(
            lib, cur_now, score_fn, order, max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
        single_best = {m: int(np.argmax([_single_score(col) for _x, col in lib[m]]))
                       for m in group_models}
        cur_b, score_b, swaps_b = _coordinate_select(
            lib, single_best, score_fn, order, max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
        if score_b > score_a:
            return cur_b, float(score_b), swaps_b
        return cur_a, float(score_a), swaps_a

    before = cur_score
    cur, cur_score, swaps = _select(cur)
    stats['swaps'] += swaps
    print(f"[GP] Joint Bayesian optimisation async-library search: phase 1 done, library sizes="
          f"{ {m: len(lib[m]) for m in group_models} }, ensemble score {before:.4f} -> "
          f"{cur_score:.4f} ({swaps} swap(s)) - next: {refine_rounds} refine round(s) against "
          f"this selection.")

    # ---- Phase 2: barrier rounds against the selected columns -----------------------------
    for rr in range(1, refine_rounds + 1):
        if budget_hook is not None and budget_hook([]):
            stats['stop'] = 'time budget'
            break
        base_cols = {m: lib[m][cur[m]][1] for m in group_models}
        requests = []
        for m in group_models:
            if dims[m] == 0 or m in score_fn.inactive:
                continue
            pts = _propose(m, 1000 + rr, score_fn, base_cols, q, guided=True)
            requests.extend(_open_request(m, x, folds=(race_idx if racing else None))
                            for x in pts)
        if not requests:
            stats['stop'] = stats['stop'] or 'no new proposals'
            break
        longest_round = 0.0
        if racing:
            # ver4-38, R1.b: race the barrier round's proposals on the raced folds first, then
            # fit only the promoted ones on the remaining folds (decisions in group order, I9).
            raced_by_model: Dict[str, list] = {}
            for req in requests:
                col, longest, total = _close_request(req, partial=True)
                longest_round = max(longest_round, longest)
                stats['busy_s'] += total
                raced_by_model.setdefault(req['model'], []).append((req['x'], col))
            requests = []
            for m in group_models:
                if m not in raced_by_model:
                    continue
                promoted, _n_failed = _race_decide(m, raced_by_model[m], score_fn, base_cols)
                stats['evals_by_model'][m] += len(raced_by_model[m]) - len(promoted)
                requests.extend(_open_request(m, x) for x in promoted)
        for req in requests:
            col, longest, total = _close_request(req)
            longest_round = max(longest_round, longest)
            stats['busy_s'] += total
            stats['evals_by_model'][req['model']] += 1
            if col is not None:
                lib[req['model']].append((req['x'], col))
        stats['critical_path_s'] += longest_round
        before = cur_score
        cur, cur_score, swaps = _select(cur)
        stats['swaps'] += swaps
        stats['refine'] = rr
        print(f"[GP] Joint Bayesian optimisation async-library refine round {rr}/{refine_rounds}: "
              f"{len(requests)} column(s) fitted in one batch, ensemble score {before:.4f} -> "
              f"{cur_score:.4f} ({swaps} swap(s)).")
    if not stats['stop']:
        stats['stop'] = 'budget reached'
    stats['wall_s'] = time.perf_counter() - t_start
    stats['utilisation'] = (stats['busy_s'] / (stats['wall_s'] * n_jobs)
                            if stats['wall_s'] > 0 else 0.0)
    if counter is not None and isinstance(counter.get('secs'), dict):
        counter['secs']['search'] = counter['secs'].get('search', 0.0) + stats['wall_s']
    stats['lib_sizes'] = {m: len(lib[m]) for m in group_models}
    stats['lib'] = lib
    stats['lib_cur'] = dict(cur)
    best_x = (np.concatenate([np.asarray(lib[m][cur[m]][0], dtype=float) for m in group_models])
              if sum(dims.values()) else np.array([], dtype=float))
    return best_x, float(cur_score), stats


# ------------------------------------------------------------------------------------------
# ver4-40, R2 - weight-first joint search
# ------------------------------------------------------------------------------------------

def _wf_active_sizes(n_cands: int, rungs: int, eta: int) -> List[int]:
    """ver4-40, R2: the size of the active weight-candidate set |A_r| at every rung - `n_cands`
    in rung 0, then ceil(|A| / eta) (never below 1). The ONE place this halving is defined: the
    engine and the fit-count estimator both call it (I3)."""
    sizes: List[int] = []
    a = max(1, int(n_cands))
    for _ in range(max(1, int(rungs))):
        sizes.append(a)
        a = max(1, int(math.ceil(a / float(max(2, int(eta))))))
    return sizes


def _wf_rung_counts(budget_new: int, active: Sequence[int], init_points: int, cap: int
                    ) -> List[int]:
    """ver4-40, R2: how many NEW library columns one model fits in each rung (an upper bound:
    the engine proposes for fewer candidates when a weight is below `wf_min_tune_weight`).

    Rung 0 is ONE random block shared by every weight candidate (a column does not depend on
    the weight vector): `min(init_points, budget_new // R)`, at least 1. Rung r >= 1 proposes
    `max(1, budget_new // (R * |A_r|))` columns for each active candidate. The running total is
    capped at `cap` (the model's budget plus one library_q of slack), so the total never exceeds
    what `async_library` would fit for the same `library_evals_per_dim`, give or take that slack.
    Args: `budget_new` columns after the seed column (`_async_budget() - 1`), `active` from
    `_wf_active_sizes()`, `init_points`, `cap`. Returns one int per rung."""
    rungs = len(active)
    counts: List[int] = []
    left = max(0, int(cap))
    for r in range(rungs):
        if r == 0:
            n = max(1, min(int(init_points), max(1, int(budget_new) // rungs)))
        else:
            n = max(1, int(budget_new) // (rungs * int(active[r]))) * int(active[r])
        n = min(n, left)
        counts.append(int(n))
        left -= n
    return counts


def _weight_first_search(*, group_models, specs_by_model, base_params_by_model, seed_point,
                         fit_ctx: dict, settings: dict, state, seed: int
                         ) -> Tuple[np.ndarray, float, dict]:
    """ver4-40, R2: `search_strategy='weight_first'` - fix a small set of candidate weight vectors
    FIRST, search the hyperparameters under each fixed vector, and pick the best (weight vector,
    library columns) pair.

    Why: the monolithic and library searches re-estimate K-1 continuous weights on 50-150
    scoring rows for EVERY candidate, so the winner is partly the candidate whose weights fitted
    the noise (a winner's curse). With the weights fixed the score of a candidate is a plain
    linear functional of its columns - far less selection freedom (log C bits instead of K-1
    continuous weights) and a smoother search surface. The score is still a pure function of the
    K prediction columns, so every (weight vector, column set) is re-scored with ZERO fits; fits
    are paid only for NEW columns, shared across weight candidates through the fit cache.

    Flow (`R` = `wf_rungs`, `W` = the weight candidates)
      R0     the untuned (seed) column of every model, one batch; `W = weight_candidates()` of the
             seed columns on the scoring rows (`wf_candidate_rows='oof_train'`: of the
             out-of-fold TRAIN predictions at the seed parameters, `+K * folds` fits). Fewer than 2
             distinct vectors -> `stats['fallback_async']` (the caller runs `async_library`).
      rung r for r in 0..R-1, active set `A_r` (`A_0 = W`, then the best ceil(|A|/eta) by score):
             per model m with tunable dimensions, per candidate c in `A_r` with `w_c[m] >=
             wf_min_tune_weight`: rung 0 one shared random block of columns, later rungs
             `suggest_batch()` proposals conditioned on m's library re-scored under (w_c, ref_c);
             ALL requests of the rung go to ONE `_fit_requests_batch()` (deduplicated by the fit
             cache); then every c in `A_r` re-selects its columns with `_coordinate_select()` under
             `_FixedWeightScorer(w_c)`; the barrier makes the result independent of `n_jobs` (I9).
      pick   finalists = every (c, selection, 'tuned') plus, with `wf_include_untuned`, every
             (c, untuned, 'untuned'); `wf_pick='bbc'` -> `_wf_bbc()` (mean out-of-bag score),
             falling back to `'max'` (best full-row score) with a NOTE.
    The final weights are the chosen FIXED vector (the caller applies them; they are not
    re-estimated). Everything that chooses reads scoring rows only (I7); the test frame is only
    ever passed through to the fits.

    Args: as `_async_library_search()` minus `score_fn`; `settings` keys: wf_rules,
    wf_max_candidates, wf_dedup_tol, wf_rungs, wf_eta, wf_min_tune_weight, wf_include_untuned,
    wf_pick, wf_candidate_rows, library_q, block_init_points, block_tol, library_evals_per_dim,
    library_max_evals, objective_mode, dead_column_guard, estimator_settings, es_settings,
    bbc_bootstraps, oof_folds, inner_split_seed_base, time_budget_s, time_budget_t0.

    Returns `(best_x, best_score, stats)`; `stats` has `lib`, `lib_cur` (the chosen selection),
    `dead_models`, `evals_by_model`, `rounds_by_model`, `lib_sizes`, `stop`, `wall_s` and
    `wf` = {name, w, arm, c, pick_how, n_candidates, rungs_run, table, rules}. Returns
    `(seed_point, -inf, stats)` when an untuned column cannot be fitted (the caller falls back to
    the monolithic search) or, with `stats['fallback_async']`, when fewer than two distinct weight
    vectors exist. Raises AssertionError if the scoring vector length is neither `len(valid)` nor
    `len(train) + len(valid)` (A2.5), ValueError for a malformed `wf_rules` entry. Never raises
    for a failed fit."""
    t_start = time.perf_counter()
    dims = {m: len(specs_by_model[m]) for m in group_models}
    offsets, _off = {}, 0
    for m in group_models:
        offsets[m] = _off
        _off += dims[m]
    K = len(group_models)
    seed_point = np.asarray(seed_point, dtype=float)
    actual = np.asarray(fit_ctx['actual'], dtype=float)
    n_rows = int(len(actual))
    cv_mode = fit_ctx.get('objective_data') == 'cv_train_valid'
    n_expected = (len(fit_ctx['train']) + len(fit_ctx['valid'])) if cv_mode else len(fit_ctx['valid'])
    if n_rows != int(n_expected):
        raise AssertionError(
            f"[joint_tuning] internal error: the weight-first scoring vector has {n_rows} row(s), "
            f"expected {int(n_expected)} ({'train+valid' if cv_mode else 'valid'}).")
    partition = list(fit_ctx['partition']) if cv_mode else None
    rc = dict(fit_ctx.get('runner_ctx') or {})
    counter = rc.get('counter')
    q = max(1, int(settings.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)))
    init_points = max(1, int(settings.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)))
    tol = float(settings.get('block_tol', _JOINT_BLOCK_TOL_DEFAULT))
    per_dim = int(settings.get('library_evals_per_dim', _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT))
    max_evals = int(settings.get('library_max_evals', _JOINT_LIBRARY_MAX_EVALS_DEFAULT))
    rungs = max(1, int(settings.get('wf_rungs', _JOINT_WF_RUNGS_DEFAULT)))
    eta = max(2, int(settings.get('wf_eta', _JOINT_WF_ETA_DEFAULT)))
    min_w = float(settings.get('wf_min_tune_weight', _JOINT_WF_MIN_TUNE_WEIGHT_DEFAULT))
    include_untuned = bool(settings.get('wf_include_untuned', _JOINT_WF_INCLUDE_UNTUNED_DEFAULT))
    pick_mode = settings.get('wf_pick', _JOINT_WF_PICK_DEFAULT)
    cand_rows = settings.get('wf_candidate_rows', _JOINT_WF_CANDIDATE_ROWS_DEFAULT)
    dead_guard = bool(settings.get('dead_column_guard', _JOINT_DEAD_COLUMN_GUARD_DEFAULT))
    es_settings = dict(settings.get('es_settings') or {})
    inner_seed = int(settings.get('inner_split_seed_base', 0))
    budget_s = float(settings.get('time_budget_s', _JOINT_TIME_BUDGET_DEFAULT) or 0.0)
    budget_hook = (_time_budget_stop(settings['time_budget_t0'], budget_s)
                   if settings.get('time_budget_t0') is not None else None)
    acquisition = ((lambda: ExpectedImprovement(xi=0.01)) if ExpectedImprovement is not None
                   else None)
    noise_aware = bool(settings.get('noise_aware_surrogate', _JOINT_NOISE_AWARE_DEFAULT))
    stats = {'evals_by_model': {m: 0 for m in group_models},
             'rounds_by_model': {m: 0 for m in group_models}, 'lib_sizes': {}, 'stop': '',
             'wall_s': 0.0, 'seed_ens': None, 'swaps': 0, 'dead_models': []}

    def _wf_seed(m, c, r):
        return int(seed) + (zlib.crc32(m.encode('utf-8')) % 10_000) + 97 * int(c) + int(r)

    # ---- R0: the untuned column of every model, one batch ------------------------------
    chunks0 = {m: seed_point[offsets[m]:offsets[m] + dims[m]] for m in group_models}
    cols0 = _fit_requests_batch([(m, chunks0[m]) for m in group_models], ctx=fit_ctx)
    failed0 = [m for m, col in zip(group_models, cols0) if col is None]
    if failed0:
        print(f"[GP] NOTE: Joint Bayesian optimisation weight-first search could not fit the "
              f"initial (untuned) column for {failed0!r} - falling back to the monolithic search.")
        return seed_point, -math.inf, stats
    lib = {m: [(np.array(chunks0[m], dtype=float), col)] for m, col in zip(group_models, cols0)}
    seen = {m: {np.array(chunks0[m], dtype=float).tobytes()} for m in group_models}
    seed_cols = {m: lib[m][0][1] for m in group_models}
    P0 = np.column_stack([seed_cols[m] for m in group_models])

    # ---- dead untuned columns (only with dead_column_guard): weight 0 in every vector ------
    dead_idx: List[int] = []
    if dead_guard:
        dead_idx = [i for i, m in enumerate(group_models)
                    if _is_dead_column(seed_cols[m], actual, partition)]
        if len(dead_idx) >= K:
            print(f"[GP] NOTE: Joint Bayesian optimisation weight-first search: the untuned column "
                  f"of EVERY model is constant - nothing can be dropped, so no model is "
                  f"zero-weighted for this task.")
            dead_idx = []
        elif dead_idx:
            print(f"[GP] WARN: Joint Bayesian optimisation weight-first search: the untuned column "
                  f"of {[group_models[i] for i in dead_idx]!r} is CONSTANT (no genetic signal) - "
                  f"those model(s) get weight 0 in every candidate vector and are not tuned.")
    dead_models = [group_models[i] for i in dead_idx]
    stats['dead_models'] = list(dead_models)

    # ---- the weight candidates -------------------------------------------------------------
    P_w, y_w, rows_note = P0, actual, 'the scoring rows'
    if cand_rows == 'oof_train':
        if cv_mode:
            print("[GP] NOTE: Joint Bayesian optimisation wf_candidate_rows='oof_train' is ignored "
                  "with objective_data='cv_train_valid' (the scoring rows are already out-of-fold "
                  "over train+valid) - the weight candidates use the scoring rows.")
        else:
            _n_folds = max(2, int(settings.get('oof_folds', _JOINT_OOF_FOLDS_DEFAULT)))
            oof = _oof_pooled_matrix(
                group_models, {m: list(base_params_by_model[m]) for m in group_models},
                fit_ctx['train'], fit_ctx['test'], fit_ctx['runner'], fit_ctx['RESULT_NAME'],
                n_folds=_n_folds, seed=int(seed), n_jobs=int(rc.get('n_jobs') or 1),
                worker_init=rc.get('worker_init'), worker_init_args=rc.get('worker_init_args', ()))
            if counter is not None:
                counter['oof'] = counter.get('oof', 0) + _n_folds * K
            if oof is None:
                print("[GP] NOTE: Joint Bayesian optimisation wf_candidate_rows='oof_train': the "
                      "out-of-fold training matrix could not be formed - the weight candidates "
                      "use the scoring rows.")
            else:
                P_w, y_w = np.asarray(oof[0], dtype=float), np.asarray(oof[1], dtype=float)
                rows_note = f"{len(y_w)} out-of-fold TRAIN row(s)"
    cands = weight_candidates(
        P_w, y_w, rules=list(settings.get('wf_rules', _JOINT_WF_RULES_DEFAULT)),
        es_settings=es_settings, dedup_tol=float(settings.get('wf_dedup_tol',
                                                              _JOINT_WF_DEDUP_TOL_DEFAULT)),
        max_n=int(settings.get('wf_max_candidates', _JOINT_WF_MAX_CANDIDATES_DEFAULT)),
        seed=int(seed) + inner_seed, dead=dead_idx)
    if len(cands) < 2:
        stats['fallback_async'] = (f"only {len(cands)} distinct weight candidate(s) after "
                                   f"de-duplication (need >= 2)")
        stats['lib'], stats['lib_cur'] = lib, {m: 0 for m in group_models}
        print(f"[GP] NOTE: Joint Bayesian optimisation weight-first search has "
              f"{stats['fallback_async']} - running the async-library search instead.")
        return seed_point, -math.inf, stats
    C = len(cands)
    W = [np.asarray(w, dtype=float) for _name, w in cands]
    cand_names = [name for name, _w in cands]
    bbc_rules = ([list(map(float, w)) for w in W] if rows_note != 'the scoring rows'
                 else _wf_rule_specs(cands))
    print(f"[GP] Joint Bayesian optimisation weight-first search: {C} weight candidate(s) from "
          f"{rows_note}: "
          + '; '.join(f"{n}=" + '/'.join(f"{x:.2f}" for x in w)
                      for n, w in zip(cand_names, W)) + f" (model order {list(group_models)!r}).")

    scorers = [_FixedWeightScorer(group_models, w, fit_ctx, settings, {'worst_finite': None})
               for w in W]
    orders = [[group_models[i] for i in sorted(range(K), key=lambda i, _w=w: (-_w[i], i))]
              for w in W]
    ref = [{m: 0 for m in group_models} for _ in range(C)]       # per-candidate selection
    s_ref = [float(scorers[c].score(dict(seed_cols))) for c in range(C)]
    stats['seed_ens'] = float(np.mean(s_ref))
    budget_new = {m: max(0, _async_budget(dims[m], per_dim, max_evals) - 1) if dims[m] else 0
                  for m in group_models}
    active = list(range(C))
    sizes = _wf_active_sizes(C, rungs, eta)
    new_cols = {m: 0 for m in group_models}
    caps = {m: budget_new[m] + q for m in group_models}
    tunable = [m for m in group_models if dims[m] > 0 and m not in dead_models]

    def _score_of(c, cols_by_model):
        return float(scorers[c].score(cols_by_model))

    # ---- successive halving over the weight candidates -------------------------------------
    rungs_run = 0
    for r in range(rungs):
        if r > 0 and budget_hook is not None and budget_hook([]):
            stats['stop'] = 'time budget'
            break
        requests = []                       # (model, x) in deterministic (model, candidate) order
        for m in tunable:
            relevant = [c for c in active if W[c][group_models.index(m)] >= min_w]
            if not relevant or new_cols[m] >= caps[m]:
                continue
            if r == 0:
                n0 = _wf_rung_counts(budget_new[m], sizes, init_points, caps[m])[0]
                pts = (suggest_batch([(0.0, 1.0)] * dims[m], [], q=n0, seed=_wf_seed(m, 0, 0))
                       if n0 > 0 else [])
                proposals = [pts]
            else:
                n_c = max(1, budget_new[m] // (rungs * sizes[r]))
                proposals = []
                for c in relevant:
                    observations = []
                    base_cols = {k: lib[k][ref[c][k]][1] for k in group_models}
                    for x_e, col_e in lib[m][-_JOINT_LIBRARY_RESCORE_CAP:]:
                        trial = dict(base_cols)
                        trial[m] = col_e
                        sc = _score_of(c, trial)
                        if np.isfinite(sc):
                            observations.append((x_e, sc))
                    try:
                        pts = suggest_batch([(0.0, 1.0)] * dims[m], observations, q=n_c,
                                            seed=_wf_seed(m, c, r), liar='max',
                                            acquisition_factory=acquisition,
                                            noise_aware=noise_aware)
                    except (ImportError, RuntimeError) as exc:
                        print(f"[GP] NOTE: Joint Bayesian optimisation weight-first search could "
                              f"not build a surrogate suggestion for {m!r} under candidate "
                              f"{cand_names[c]!r} in rung {r} ({type(exc).__name__}: {exc}) - "
                              f"proposing random points for it.")
                        pts = suggest_batch([(0.0, 1.0)] * dims[m], [], q=n_c,
                                            seed=_wf_seed(m, c, r))
                    proposals.append(pts)
            room = caps[m] - new_cols[m]
            for pts in proposals:
                for x in pts:
                    x = np.array(x, dtype=float)
                    key = x.tobytes()
                    if key in seen[m] or room <= 0:
                        continue
                    seen[m].add(key)
                    requests.append((m, x))
                    room -= 1
            new_cols[m] = caps[m] - room
        if requests:
            fitted = _fit_requests_batch(requests, ctx=fit_ctx)
            for (m, x), col in zip(requests, fitted):
                stats['evals_by_model'][m] += 1
                if col is not None:
                    lib[m].append((x, col))
            for m in {mm for mm, _x in requests}:
                stats['rounds_by_model'][m] = r + 1
        rungs_run += 1
        for c in active:
            ref[c], s_ref[c], sw = _coordinate_select(
                lib, ref[c], scorers[c].score, orders[c],
                max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
            stats['swaps'] += sw
        keep = max(1, int(math.ceil(len(active) / float(eta))))
        order_c = sorted(active, key=lambda c: (-s_ref[c], c))
        print(f"[GP] Joint Bayesian optimisation weight-first rung {r + 1}/{rungs}: "
              f"{len(requests)} new column(s) fitted in one batch (library sizes "
              f"{ {m: len(lib[m]) for m in group_models} }); candidate scores "
              + ', '.join(f"{cand_names[c]}={s_ref[c]:.4f}" for c in active)
              + (f" - keeping {[cand_names[c] for c in sorted(order_c[:keep])]!r}"
                 if r + 1 < rungs and keep < len(active) else "") + ".")
        if r + 1 < rungs:
            active = sorted(order_c[:keep])
    # every candidate re-selects over the COMPLETE library (zero fits), so the finalists are
    # compared on the same information
    for c in range(C):
        ref[c], s_ref[c], sw = _coordinate_select(
            lib, ref[c], scorers[c].score, orders[c],
            max_passes=_JOINT_LIBRARY_MAX_PASSES, tol=tol)
        stats['swaps'] += sw

    # ---- finalists and the pick ------------------------------------------------------------
    zero = {m: 0 for m in group_models}
    finalists, f_score = [], []
    for c in range(C):
        finalists.append({'c': c, 'arm': 'tuned', 'pick': dict(ref[c]), 'w': W[c]})
        f_score.append(float(s_ref[c]))
        if include_untuned:
            finalists.append({'c': c, 'arm': 'untuned', 'pick': dict(zero), 'w': W[c]})
            f_score.append(_score_of(c, seed_cols))
    chosen, how, mean_oob = None, 'max', [math.nan] * len(finalists)
    gap_sd = [math.nan] * len(finalists)
    if pick_mode in ('bbc', 'bbc_max'):
        res = _wf_bbc(lib, finalists, bbc_rules, actual,
                      n_boot=int(settings.get('bbc_bootstraps', _JOINT_BBC_BOOTSTRAPS_DEFAULT)),
                      seed=int(seed) + inner_seed + _JOINT_BBC_SEED_OFFSET, partition=partition,
                      es_settings=es_settings, tol=tol, parsimony=(pick_mode == 'bbc'),
                      k_se=float(settings.get('wf_k_se', _JOINT_WF_K_SE_DEFAULT)))
        if res['ok']:
            chosen, how, mean_oob = int(res['selected']), pick_mode, list(res['mean_oob'])
            gap_sd = list(res['sd_gap'])
            if res['one_se']:
                b = finalists[int(res['best'])]
                print(f"[GP] Joint Bayesian optimisation weight-first one-standard-error rule: the "
                      f"highest out-of-bag mean was {cand_names[b['c']]!r} ({b['arm']}), but "
                      f"{cand_names[finalists[chosen]['c']]!r} ({finalists[chosen]['arm']}) is "
                      f"within {float(settings.get('wf_k_se', _JOINT_WF_K_SE_DEFAULT)):g} "
                      f"standard deviation(s) of it and simpler - the simpler finalist is kept.")
        else:
            # ver4-41: with no out-of-bag evidence the in-sample best (a tuned arm of the vector
            # that fitted the scoring rows best - typically one-hot) must NOT be chosen; take the
            # simplest finalist (the equal-weight blend) instead.
            chosen, how = _wf_simplest(finalists, dead_idx), 'simplest'
            print(f"[GP] NOTE: Joint Bayesian optimisation weight-first BBC pick could not be "
                  f"formed ({res['reason']}) - using the simplest finalist "
                  f"({cand_names[finalists[chosen]['c']]!r}, {finalists[chosen]['arm']}) for this "
                  f"task (no out-of-bag evidence that a more complex one is better).")
    if chosen is None:
        top = max(f_score)
        tied = [i for i, v in enumerate(f_score) if v >= top - 1e-12]
        chosen = min(tied, key=lambda i: (0 if finalists[i]['arm'] == 'tuned' else 1, i))
    fin = finalists[chosen]
    pick = dict(fin['pick'])
    best_x = (np.concatenate([np.asarray(lib[m][pick[m]][0], dtype=float) for m in group_models])
              if sum(dims.values()) else np.array([], dtype=float))
    table = [{'c': f['c'], 'rule': cand_names[f['c']], 'arm': f['arm'],
              'score': float(f_score[i]), 'mean_oob': float(mean_oob[i]),
              'sd_gap': float(gap_sd[i]),
              'n_swaps': int(sum(f['pick'][m] != 0 for m in group_models))}
             for i, f in enumerate(finalists)]
    stats['wf'] = {'name': cand_names[fin['c']], 'w': [float(x) for x in W[fin['c']]],
                   'arm': fin['arm'], 'c': int(fin['c']), 'pick_how': how, 'n_candidates': C,
                   'rungs_run': rungs_run, 'table': table, 'rules': list(cand_names),
                   'rows': rows_note}
    print(f"[GP] Joint Bayesian optimisation weight-first pick ({how}): candidate "
          f"{cand_names[fin['c']]!r} ({fin['arm']}) weights="
          f"{ {m: round(float(x), 4) for m, x in zip(group_models, W[fin['c']])} }; "
          + '; '.join(f"{t['rule']}/{t['arm']}: score={t['score']:.4f}"
                      + (f" oob={t['mean_oob']:.4f}" if np.isfinite(t['mean_oob']) else '')
                      for t in table) + ".")
    if not stats['stop']:
        stats['stop'] = 'budget reached'
    stats['wall_s'] = time.perf_counter() - t_start
    if counter is not None and isinstance(counter.get('secs'), dict):
        counter['secs']['search'] = counter['secs'].get('search', 0.0) + stats['wall_s']
    stats['lib_sizes'] = {m: len(lib[m]) for m in group_models}
    stats['lib'] = lib
    stats['lib_cur'] = dict(pick)
    return best_x, float(f_score[chosen]), stats


def _weight_first_cv_search(*, group_models, specs_by_model, base_params_by_model, seed_point,
                            fit_ctx: dict, settings: dict, state, seed: int
                            ) -> Tuple[np.ndarray, float, dict]:
    """ver4-42, R1: `search_strategy='weight_first_cv'` - a cross-fitted ("super learner")
    weight-first search. The weights are decided ONCE, on 4-6x more rows than the validation
    split, BEFORE any hyperparameter is tuned; tuning then runs under those fixed weights.

    Why (blueprint RC1-RC4): the ver4-40/41 weight-first search chose its weights from the
    ~30-80 validation rows only (high-variance weights), picked them jointly with noisy tuning
    arms, and its parsimony rule then almost always returned the untuned equal-weight blend, so
    the joint row tied the naive ensemble by construction. Here a noisy tuning result can no
    longer throw away a good weight vector, and the reverse.

    Stages (`V` = number of CV folds; every row is a train+valid row, the test split is only
    ever passed through to the fits, I7)
      R0  the untuned (seed) column of every model at SEARCH fidelity, one batch (library entry 0).
      S0  `Z0` = out-of-fold untuned columns over train+valid. `wf_cv_weight_fidelity='full'`
          refits them at the final models' fidelity (+K*V fits, counted in the 'oof' stage; a
          model whose refit fails keeps its search-fidelity column), `'search'` reuses entry 0.
          Without the CV objective (not eligible: see the caller's NOTE) `Z0` is entry 0 over the
          validation rows and everything below runs on those rows (`rows='valid'`).
      S1  `ensemble_regularization.shrinkage_path_weights(Z0, y)`: `w(alpha) = (1 - alpha) *
          uniform + alpha * direction`, alpha chosen by repeated CV over the out-of-fold rows
          (`wf_cv_alpha_rule`). `w*` is final from here on. A constant (dead) column gets 0.
      S2  (`wf_cv_tune`) `_async_library_search()` with a `_FixedWeightLibraryScorer(w*)`: every
          model whose weight is at least `wf_min_tune_weight` is tuned against the fixed
          reference columns; the others keep their untuned column AND their weight.
      S3  zero fits: `_wf_bbc()` decides between the untuned and the tuned columns AT THE SAME
          weights (`wf_cv_accept`: 'bbc_mean' = plain out-of-bag mean, 'bbc_one_se' = with the
          parsimony rule, 'always' = the tuned columns). The plain mean is the default because
          the out-of-bag score already corrects the selection optimism; a margin on top would
          penalise twice (blueprint RC3).

    Args: as `_weight_first_search()`; `settings` keys: wf_cv_weight_fidelity, wf_cv_direction,
    wf_cv_alphas, wf_cv_alpha_rule, wf_cv_repeats, wf_cv_tune, wf_cv_accept, wf_min_tune_weight,
    wf_k_se, bbc_bootstraps, block_tol, dead_column_guard, es_settings, inner_split_seed_base
    plus everything `_async_library_search()` reads (library_*, cv_racing, racing_*,
    time_budget_*).

    Returns `(best_x, best_score, stats)` with the contract of `_weight_first_search()`:
    `stats['wf']` = {name ('cv_alpha=0.40'), w, arm, c (0), pick_how, n_candidates (1),
    rungs_run, table, rules, rows, alpha, cv_mse, direction, fidelity}; `best_x` is the chosen
    columns' unit-cube point and `best_score` their score at `w*`. Returns `(seed_point, -inf,
    stats)` when an untuned column cannot be fitted (the caller falls back to the monolithic
    search). Raises AssertionError if the scoring vector length is neither `len(valid)` nor
    `len(train) + len(valid)` (A1.4). Never raises for a failed fit or a failed stage S2 (S2
    then degrades to the untuned columns at `w*`, with a NOTE)."""
    t_start = time.perf_counter()
    dims = {m: len(specs_by_model[m]) for m in group_models}
    offsets, _off = {}, 0
    for m in group_models:
        offsets[m] = _off
        _off += dims[m]
    K = len(group_models)
    seed_point = np.asarray(seed_point, dtype=float)
    actual = np.asarray(fit_ctx['actual'], dtype=float)
    n_rows = int(len(actual))
    cv_mode = fit_ctx.get('objective_data') == 'cv_train_valid'
    n_expected = (len(fit_ctx['train']) + len(fit_ctx['valid'])) if cv_mode else len(fit_ctx['valid'])
    if n_rows != int(n_expected):
        raise AssertionError(
            f"[joint_tuning] internal error: the weight-first-cv scoring vector has {n_rows} "
            f"row(s), expected {int(n_expected)} ({'train+valid' if cv_mode else 'valid'}).")
    partition = list(fit_ctx['partition']) if cv_mode else None
    rc = dict(fit_ctx.get('runner_ctx') or {})
    fidelity = settings.get('wf_cv_weight_fidelity', _JOINT_WFCV_WEIGHT_FIDELITY_DEFAULT)
    direction = settings.get('wf_cv_direction', _JOINT_WFCV_DIRECTION_DEFAULT)
    alphas = list(settings.get('wf_cv_alphas', _JOINT_WFCV_ALPHAS_DEFAULT))
    alpha_rule = settings.get('wf_cv_alpha_rule', _JOINT_WFCV_ALPHA_RULE_DEFAULT)
    repeats = max(1, int(settings.get('wf_cv_repeats', _JOINT_WFCV_REPEATS_DEFAULT)))
    tune = bool(settings.get('wf_cv_tune', _JOINT_WFCV_TUNE_DEFAULT))
    accept = settings.get('wf_cv_accept', _JOINT_WFCV_ACCEPT_DEFAULT)
    min_w = float(settings.get('wf_min_tune_weight', _JOINT_WF_MIN_TUNE_WEIGHT_DEFAULT))
    dead_guard = bool(settings.get('dead_column_guard', _JOINT_DEAD_COLUMN_GUARD_DEFAULT))
    es_settings = dict(settings.get('es_settings') or {})
    inner_seed = int(settings.get('inner_split_seed_base', 0))
    tol = float(settings.get('block_tol', _JOINT_BLOCK_TOL_DEFAULT))
    rows_note = 'train+valid out-of-fold' if cv_mode else 'valid'
    # ver4-44: R1 `safety_nets` (False: alpha is 1 and S3 keeps the tuned columns; class C events
    # are recorded in the task's shared `fallback_events` list) and R3 `wf_cv_final`.
    safety = bool(settings.get('safety_nets', _JOINT_SAFETY_NETS_DEFAULT))
    final_mode = settings.get('wf_cv_final', _JOINT_WFCV_FINAL_DEFAULT)
    events = settings.get('fallback_events')
    if events is None:
        events = []
    if not safety:
        accept = 'always'
    stats = {'evals_by_model': {m: 0 for m in group_models},
             'rounds_by_model': {m: 0 for m in group_models}, 'lib_sizes': {}, 'stop': '',
             'wall_s': 0.0, 'seed_ens': None, 'swaps': 0, 'dead_models': []}

    # ---- R0: the untuned column of every model (search fidelity), one batch -----------------
    chunks0 = {m: seed_point[offsets[m]:offsets[m] + dims[m]] for m in group_models}
    cols0 = _fit_requests_batch([(m, chunks0[m]) for m in group_models], ctx=fit_ctx)
    failed0 = [m for m, col in zip(group_models, cols0) if col is None]
    if failed0:
        print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv search could not fit the "
              f"initial (untuned) column for {failed0!r} - falling back to the monolithic search.")
        return seed_point, -math.inf, stats
    lib0 = {m: [(np.array(chunks0[m], dtype=float), col)] for m, col in zip(group_models, cols0)}
    seed_cols = {m: lib0[m][0][1] for m in group_models}

    # ---- S0: the out-of-fold untuned columns the weights are estimated on -------------------
    z0 = dict(seed_cols)
    z0_label = 'search fidelity'
    if cv_mode and fidelity == 'full':
        rc_s0 = dict(rc)
        rc_s0['stage'] = 'oof'                   # full-fidelity refits are counted as 'oof' fits
        requests = [(m, _disable_explainability(m, _decode_model_params(
            m, chunks0[m], specs=specs_by_model[m], base_params=base_params_by_model[m],
            search_fidelity=False))) for m in group_models]
        full_cols = _cv_fit_columns(
            requests, tv=fit_ctx['tv'], partition=fit_ctx['partition'], seed=fit_ctx['cv_seed'],
            test=fit_ctx['test'], runner=fit_ctx['runner'], RESULT_NAME=fit_ctx['RESULT_NAME'],
            runner_ctx=rc_s0, stop_on_failure=False)
        failed_full = [m for m, col in zip(group_models, full_cols) if col is None]
        for m, col in zip(group_models, full_cols):
            if col is not None:
                z0[m] = col
        z0_label = 'full fidelity'
        if failed_full:
            print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S0: the full-fidelity "
                  f"out-of-fold fit failed for {failed_full!r} - their search-fidelity columns "
                  f"are used for the weights.")
            z0_label = f"full fidelity except {failed_full!r}"
            events.extend(f"N12:{m}" for m in failed_full)      # ver4-44, class C
    elif cv_mode:
        z0_label = "search fidelity (wf_cv_weight_fidelity='search')"
    Z0 = np.column_stack([z0[m] for m in group_models])
    if Z0.shape != (n_rows, K):
        raise AssertionError(
            f"[joint_tuning] internal error: the weight-first-cv matrix has shape {Z0.shape}, "
            f"expected ({n_rows}, {K}).")

    # ---- dead columns (only with dead_column_guard): weight 0, never tuned -------------------
    dead_idx: List[int] = []
    if dead_guard:
        dead_idx = [i for i, m in enumerate(group_models)
                    if _is_dead_column(seed_cols[m], actual, partition)
                    or _is_dead_column(z0[m], actual, partition)]
        if len(dead_idx) >= K:
            print("[GP] NOTE: Joint Bayesian optimisation weight-first-cv search: the untuned "
                  "column of EVERY model is constant - nothing can be dropped, so no model is "
                  "zero-weighted for this task.")
            dead_idx = []
        elif dead_idx:
            print(f"[GP] WARN: Joint Bayesian optimisation weight-first-cv search: the untuned "
                  f"column of {[group_models[i] for i in dead_idx]!r} is CONSTANT (no genetic "
                  f"signal) - those model(s) get weight 0 and are not tuned.")
    dead_models = [group_models[i] for i in dead_idx]
    stats['dead_models'] = list(dead_models)

    # ---- S1: the weights, decided once ------------------------------------------------------
    w_star, alpha_star, diag = shrinkage_path_weights(
        Z0, actual, direction=direction, alphas=alphas, rule=alpha_rule, n_repeats=repeats,
        seed=int(seed) + inner_seed, dead=dead_idx, es_settings=es_settings, final=final_mode)
    w_star = np.asarray(w_star, dtype=float)
    if diag.get('note'):
        print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S1: {diag['note']}.")
        events.append(f"N14:S1_{str(diag['note']).split(' - ')[0][:60]}")     # ver4-44, class C
    cv_curve = {k: round(float(v), 5) for k, v in (diag.get('cv_mse') or {}).items()}
    if not safety:
        # ver4-44, R1 (N8): no shrinkage towards equal weights. The fitted direction (the
        # refit, or with wf_cv_final='bagged' the mean of the in-fold vectors) IS w*, after
        # dead masking (shrinkage_path_weights() already zeroed `dead_idx` in it).
        w_star = np.asarray(diag['direction_w'], dtype=float)
        alpha_star = 1.0
        wf_name = 'cv_direction'
    else:
        wf_name = f"cv_alpha={alpha_star:.2f}"
    _spread = diag.get('fold_spread')
    _alpha_txt = ("alpha path off (safety_nets=False): w* = the fitted direction, alpha*=1.00"
                  if not safety else
                  f"alpha*={alpha_star:.2f} (best CV alpha "
                  f"{diag.get('best_alpha', alpha_star):.2f}), CV loss by alpha {cv_curve}")
    print(f"[GP] Joint Bayesian optimisation weight-first-cv S1: {n_rows} {rows_note} row(s) "
          f"({z0_label}), direction={direction!r}, rule={alpha_rule!r}, repeats={repeats}, "
          f"final={diag.get('final', final_mode)!r}: {_alpha_txt}, w*="
          f"{ {m: round(float(x), 4) for m, x in zip(group_models, w_star)} }"
          + (f", fold_spread={_spread:.4f}" if _spread is not None else "") + ".")

    # ---- the scorer at the fixed weights ------------------------------------------------------
    tunable = [m for i, m in enumerate(group_models)
               if dims[m] > 0 and m not in dead_models and w_star[i] >= min_w]
    inactive = [m for m in group_models if m not in tunable]
    scorer = _FixedWeightLibraryScorer(group_models, w_star, fit_ctx, settings, state,
                                       inactive=inactive, dead_models=dead_models)
    stats['seed_ens'] = float(scorer.score(dict(seed_cols)))
    lib, lib_cur = lib0, {m: 0 for m in group_models}
    astats: dict = {}

    # ---- S2: tune under the fixed weights -----------------------------------------------------
    s2_ran = False
    if not tune:
        print("[GP] Joint Bayesian optimisation weight-first-cv S2: wf_cv_tune=False - no "
              "hyperparameter is tuned; the joint row is the fixed weights w* on the untuned "
              "columns.")
    elif not tunable:
        print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S2 skipped: no model with "
              f"a tunable dimension has a weight >= wf_min_tune_weight={min_w:g} (weights "
              f"{ {m: round(float(x), 4) for m, x in zip(group_models, w_star)} }) - the joint "
              f"row is w* on the untuned columns.")
    else:
        print(f"[GP] Joint Bayesian optimisation weight-first-cv S2: tuning {tunable!r} under the "
              f"fixed weights; not tuned (weight < {min_w:g}, dead or nothing tunable): "
              f"{inactive!r}.")
        try:
            _bx, s2_score, astats = _async_library_search(
                group_models=group_models, specs_by_model=specs_by_model,
                base_params_by_model=base_params_by_model, seed_point=seed_point, fit_ctx=fit_ctx,
                score_fn=scorer, settings=settings, state=state, seed=seed)
            if not np.isfinite(s2_score):
                print("[GP] NOTE: Joint Bayesian optimisation weight-first-cv S2: the async "
                      "library search could not fit its untuned columns - the joint row is w* on "
                      "the untuned columns.")
                events.append('N15:S2_untuned_columns')       # ver4-44, class C
                astats = {}
            else:
                lib, lib_cur = astats['lib'], dict(astats['lib_cur'])
                s2_ran = True
        except Exception as exc:
            print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S2 failed "
                  f"({type(exc).__name__}: {exc}) - the joint row is w* on the untuned columns "
                  f"for this task.")
            events.append('N15:S2_failed')                    # ver4-44, class C
            astats = {}
            lib, lib_cur = lib0, {m: 0 for m in group_models}
    stats.update({k: v for k, v in astats.items() if k not in ('lib', 'lib_cur', 'dead_models')})
    stats['dead_models'] = list(dead_models)

    # ---- S3: untuned vs tuned columns at the SAME weights, zero fits ----------------------------
    zero = {m: 0 for m in group_models}
    tuned_pick = {m: int(lib_cur[m]) for m in group_models}
    chosen_arm, how = 'untuned', 'untuned_only'
    table = []
    finalists = [{'c': 0, 'arm': 'untuned', 'pick': dict(zero), 'w': w_star}]
    scores = [float(scorer.score({m: lib[m][0][1] for m in group_models}))]
    if s2_ran and tuned_pick != zero:
        finalists.append({'c': 0, 'arm': 'tuned', 'pick': dict(tuned_pick), 'w': w_star})
        scores.append(float(scorer.score({m: lib[m][tuned_pick[m]][1] for m in group_models})))
        mean_oob = [math.nan, math.nan]
        sd_gap = [math.nan, math.nan]
        if accept == 'always':
            chosen_arm, how = 'tuned', 'always'
        else:
            res = _wf_bbc(lib, finalists, [list(map(float, w_star))], actual,
                          n_boot=int(settings.get('bbc_bootstraps', _JOINT_BBC_BOOTSTRAPS_DEFAULT)),
                          seed=int(seed) + inner_seed + _JOINT_BBC_SEED_OFFSET,
                          partition=partition, es_settings=es_settings, tol=tol,
                          parsimony=(accept == 'bbc_one_se'),
                          k_se=float(settings.get('wf_k_se', _JOINT_WF_K_SE_DEFAULT)))
            if res['ok']:
                chosen_arm, how = finalists[int(res['selected'])]['arm'], accept
                mean_oob, sd_gap = list(res['mean_oob']), list(res['sd_gap'])
            else:
                print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S3: the "
                      f"out-of-bag decision could not be formed ({res['reason']}) - the untuned "
                      f"columns are kept at w* (no evidence that the tuned ones are better).")
                how = 'untuned_no_evidence'
                events.append('N16:S3_no_evidence')           # ver4-44, class C
        for i, f in enumerate(finalists):
            table.append({'c': 0, 'rule': wf_name, 'arm': f['arm'], 'score': scores[i],
                          'mean_oob': float(mean_oob[i]), 'sd_gap': float(sd_gap[i]),
                          'n_swaps': int(sum(f['pick'][m] != 0 for m in group_models))})
        print(f"[GP] Joint Bayesian optimisation weight-first-cv S3 ({accept}): "
              + '; '.join(f"{t['arm']}: score={t['score']:.4f}"
                          + (f" oob={t['mean_oob']:.4f}" if np.isfinite(t['mean_oob']) else '')
                          for t in table) + f" -> {chosen_arm}.")
    else:
        table.append({'c': 0, 'rule': wf_name, 'arm': 'untuned', 'score': scores[0],
                      'mean_oob': math.nan, 'sd_gap': math.nan, 'n_swaps': 0})
        if s2_ran:
            print("[GP] Joint Bayesian optimisation weight-first-cv S3: the tuned selection is "
                  "the untuned columns - nothing to decide.")
    pick = dict(zero) if chosen_arm == 'untuned' else dict(tuned_pick)
    best_x = (np.concatenate([np.asarray(lib[m][pick[m]][0], dtype=float) for m in group_models])
              if sum(dims.values()) else np.array([], dtype=float))
    best_score = float(scorer.score({m: lib[m][pick[m]][1] for m in group_models}))

    # ---- S4 (ver4-47, R5): re-weight on the DEPLOYED columns ----------------------------------
    # w* was estimated on the untuned columns; the tuned ones are what gets deployed. Re-run the
    # same direction estimator on a fresh out-of-fold partition of the deployed columns. Not a
    # safety net: no baseline comparison and no pull towards equal weights.
    reweight = settings.get('wf_cv_reweight', _JOINT_WFCV_REWEIGHT_DEFAULT)
    w_deploy = np.asarray(w_star, dtype=float)
    s4 = {'mode': reweight, 'ran': False, 'w_s1': [float(x) for x in w_star]}
    if reweight != 'off':
        if not (cv_mode and s2_ran and chosen_arm == 'tuned' and any(pick[m] != 0 for m in group_models)):
            print(f"[GP] Joint Bayesian optimisation weight-first-cv S4 (wf_cv_reweight="
                  f"{reweight!r}): skipped - no tuned column is deployed (or the CV objective is "
                  f"off), so w* is already the weight vector of the deployed columns.")
        else:
            try:
                seed4 = int(fit_ctx['cv_seed']) + _JOINT_REWEIGHT_SEED_OFFSET
                part4 = _cv_partition(len(fit_ctx['tv']), len(partition), seed4)
                rc4 = dict(rc)
                rc4['stage'] = 'oof'
                req4 = [(m, _disable_explainability(m, _decode_model_params(
                    m, best_x[offsets[m]:offsets[m] + dims[m]], specs=specs_by_model[m],
                    base_params=base_params_by_model[m], search_fidelity=False)))
                    for m in group_models]
                cols4 = _cv_fit_columns(
                    req4, tv=fit_ctx['tv'], partition=part4, seed=seed4, test=fit_ctx['test'],
                    runner=fit_ctx['runner'], RESULT_NAME=fit_ctx['RESULT_NAME'],
                    runner_ctx=rc4, stop_on_failure=False)
                bad4 = [m for m, col in zip(group_models, cols4) if col is None]
                if bad4:
                    print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S4: the "
                          f"out-of-fold refit of {bad4!r} failed - w* is kept.")
                    events.append('N17:S4_oof_failed')                  # class C
                else:
                    Z1 = np.column_stack(cols4)
                    w4, a4, d4 = shrinkage_path_weights(
                        Z1, actual, direction=direction, alphas=alphas, rule=alpha_rule,
                        n_repeats=repeats, seed=int(seed) + inner_seed + _JOINT_REWEIGHT_SEED_OFFSET,
                        dead=dead_idx, es_settings=es_settings, final=final_mode)
                    w4 = np.asarray(w4, dtype=float)
                    if not safety:
                        w4 = np.asarray(d4['direction_w'], dtype=float)
                    tot4 = float(w4.sum())
                    if not np.isfinite(tot4) or tot4 <= 0.0:
                        print("[GP] NOTE: Joint Bayesian optimisation weight-first-cv S4: the "
                              "re-estimated vector is degenerate - w* is kept.")
                        events.append('N17:S4_degenerate')              # class C
                    else:
                        w4 = w4 / tot4
                        w_new = w4 if reweight == 'refit' else 0.5 * (np.asarray(w_star, float) + w4)
                        w_new = np.asarray(w_new, dtype=float)
                        for _i in dead_idx:
                            w_new[_i] = 0.0
                        w_new = w_new / float(w_new.sum())
                        w_deploy = w_new
                        s4.update({'ran': True, 'w_s4': [float(x) for x in w4],
                                   'w_deploy': [float(x) for x in w_new],
                                   'fold_spread': d4.get('fold_spread')})
                        print(f"[GP] Joint Bayesian optimisation weight-first-cv S4 "
                              f"(wf_cv_reweight={reweight!r}): weights re-estimated on the "
                              f"deployed (tuned) out-of-fold columns of {n_rows} row(s): w*(S1)="
                              f"{ {m: round(float(x), 4) for m, x in zip(group_models, w_star)} } "
                              f"-> deployed="
                              f"{ {m: round(float(x), 4) for m, x in zip(group_models, w_deploy)} }.")
            except Exception as exc:
                print(f"[GP] NOTE: Joint Bayesian optimisation weight-first-cv S4 failed "
                      f"({type(exc).__name__}: {exc}) - w* is kept for this task.")
                events.append('N17:S4_failed')                          # class C
    if not stats['stop']:
        stats['stop'] = 'budget reached'
    stats['wf'] = {'name': wf_name, 'w': [float(x) for x in w_deploy], 'arm': chosen_arm, 'c': 0,
                   'pick_how': how, 'n_candidates': 1, 'rungs_run': 1 if s2_ran else 0,
                   'table': table, 'rules': [wf_name], 'rows': rows_note,
                   'alpha': float(alpha_star), 'cv_mse': cv_curve, 'direction': direction,
                   'fidelity': z0_label,
                   # ver4-44: the deployed form of the direction (R3) and its fold instability
                   'final': diag.get('final', final_mode), 'fold_spread': _spread,
                   'safety_nets': safety, 'reweight': s4}
    stats['lib_sizes'] = {m: len(lib[m]) for m in group_models}
    stats['lib'] = lib
    stats['lib_cur'] = dict(pick)
    stats['wall_s'] = time.perf_counter() - t_start
    print(f"[GP] Joint Bayesian optimisation weight-first-cv result: alpha*={alpha_star:.2f}, "
          f"arm={chosen_arm} ({how}), weights="
          f"{ {m: round(float(x), 4) for m, x in zip(group_models, w_star)} }, score at w*="
          f"{best_score:.4f}.")
    return best_x, best_score, stats


# ------------------------------------------------------------------------------------------
# ver4-48, R2 - search_strategy='neural_stack': hyperparameter choices and ensemble weights as
# the trainable weights of one small network (blueprint EasiGP_ver4-48_Design_Blueprint.md)
# ------------------------------------------------------------------------------------------

def _resolve_ns_settings(budget_kwargs: Optional[dict]) -> dict:
    """ver4-48, R2: read and validate the `neural_stack` keys of `budget_kwargs` (the seven of
    ver4-48, `ns_weight_shrink` of ver4-49, `ns_objective` and `ns_swap_rule` of ver4-50 and, ver4-51,
    `ns_outer_gate`, `ns_outer_folds` and `ns_outer_repeats`).

    Why one function: the engine reads them once per task (I11: every key is read with
    `budget_kwargs.get(KEY, <legacy default>)` and the RESOLVED value is logged by the caller),
    and `tools/joint_replay.py` / the GUI must agree on what a bad value means. An invalid value
    prints a NOTE and falls back to the default of that key only; nothing raises, so a typo in an
    option that a run does not even use can never stop a task.

    Keys (defaults in the `_JOINT_NS_*` constants): `ns_configs` (int >= 2, configurations per
    model, entry 0 included), `ns_configs_mcmc` (int >= 1, the same for the models in
    `MCMC_LENGTH_FIELDS`), `ns_lambdas` (finite floats >= 0, a list or a comma-separated string;
    an EMPTY list is valid and means "the lambda = infinity prior only"; a non-finite entry is
    dropped, because the lambda = infinity candidate is controlled by `safety_nets`),
    `ns_lambda_rule` ('min' | 'one_se'), `ns_repeats` (int >= 1), `ns_deploy` ('mixture' |
    'argmax'), `ns_fidelity` ('full' | 'search'), `ns_weight_shrink` (float in [0, 1]),
    `ns_objective` ('ensemble_score' | 'pearson', ver4-50), `ns_swap_rule` ('gate' | 'paired_se'
    | 'paired_refit', ver4-50/51), `ns_outer_gate` ('off' | 'nested', ver4-51), `ns_outer_folds`
    (int >= 2) and `ns_outer_repeats` (int >= 1).

    Returns the dict `{key: resolved value}` (`ns_lambdas` sorted ascending, duplicates removed)."""
    cfg = budget_kwargs or {}
    out: dict = {}

    def _int(key, default, low):
        raw = cfg.get(key, default)
        try:
            val = int(raw)
            if val < low:
                raise ValueError(f"must be >= {low}")
        except (TypeError, ValueError) as exc:
            print(f"[GP] NOTE: Joint Bayesian optimisation {key}={raw!r} is not valid ({exc}) - "
                  f"using {default} for this task.")
            val = default
        return val

    out['ns_configs'] = _int('ns_configs', _JOINT_NS_CONFIGS_DEFAULT, 2)
    out['ns_configs_mcmc'] = _int('ns_configs_mcmc', _JOINT_NS_CONFIGS_MCMC_DEFAULT, 1)
    out['ns_repeats'] = _int('ns_repeats', _JOINT_NS_REPEATS_DEFAULT, 1)
    _raw_s = cfg.get('ns_weight_shrink', _JOINT_NS_WEIGHT_SHRINK_DEFAULT)
    try:
        _shr = float(_raw_s)
        if not (0.0 <= _shr <= 1.0):
            raise ValueError("must be in [0, 1]")
    except (TypeError, ValueError) as exc:
        print(f"[GP] NOTE: Joint Bayesian optimisation ns_weight_shrink={_raw_s!r} is not valid "
              f"({exc}) - using {_JOINT_NS_WEIGHT_SHRINK_DEFAULT} for this task.")
        _shr = _JOINT_NS_WEIGHT_SHRINK_DEFAULT
    out['ns_weight_shrink'] = _shr
    raw_l = cfg.get('ns_lambdas', _JOINT_NS_LAMBDAS_DEFAULT)
    if raw_l is None:
        raw_l = _JOINT_NS_LAMBDAS_DEFAULT
    if isinstance(raw_l, str):
        raw_l = [t.strip() for t in raw_l.split(',') if t.strip()]
    try:
        vals = [float(v) for v in raw_l]
        if any(math.isnan(v) or v < 0.0 for v in vals):
            raise ValueError("every lambda must be a number >= 0")
        finite = sorted({v for v in vals if math.isfinite(v)})
        if len(finite) != len(set(vals)):
            print(f"[GP] NOTE: Joint Bayesian optimisation ns_lambdas={raw_l!r}: non-finite "
                  f"entries are dropped (the lambda=infinity prior is added by safety_nets=True "
                  f"and absent with safety_nets=False).")
    except (TypeError, ValueError) as exc:
        print(f"[GP] NOTE: Joint Bayesian optimisation ns_lambdas={raw_l!r} is not valid ({exc}) "
              f"- using the default {_JOINT_NS_LAMBDAS_DEFAULT!r} for this task.")
        finite = sorted({float(v) for v in _JOINT_NS_LAMBDAS_DEFAULT})
    out['ns_lambdas'] = finite

    def _choice(key, default, choices):
        raw = cfg.get(key, default) or default
        if raw not in choices:
            print(f"[GP] NOTE: Joint Bayesian optimisation {key}={raw!r} is not recognised "
                  f"(expected one of {list(choices)!r}) - falling back to {default!r} for this "
                  f"task.")
            raw = default
        return raw

    out['ns_lambda_rule'] = _choice('ns_lambda_rule', _JOINT_NS_LAMBDA_RULE_DEFAULT,
                                    _JOINT_NS_LAMBDA_RULE_CHOICES)
    out['ns_deploy'] = _choice('ns_deploy', _JOINT_NS_DEPLOY_DEFAULT, _JOINT_NS_DEPLOY_CHOICES)
    out['ns_fidelity'] = _choice('ns_fidelity', _JOINT_NS_FIDELITY_DEFAULT,
                                 _JOINT_NS_FIDELITY_CHOICES)
    out['ns_objective'] = _choice('ns_objective', _JOINT_NS_OBJECTIVE_DEFAULT,
                                  _JOINT_NS_OBJECTIVE_CHOICES)
    out['ns_swap_rule'] = _choice('ns_swap_rule', _JOINT_NS_SWAP_RULE_DEFAULT,
                                  _JOINT_NS_SWAP_RULE_CHOICES)
    # ver4-51
    out['ns_outer_gate'] = _choice('ns_outer_gate', _JOINT_NS_OUTER_GATE_DEFAULT,
                                   _JOINT_NS_OUTER_GATE_CHOICES)
    out['ns_outer_folds'] = _int('ns_outer_folds', _JOINT_NS_OUTER_FOLDS_DEFAULT, 2)
    out['ns_outer_repeats'] = _int('ns_outer_repeats', _JOINT_NS_OUTER_REPEATS_DEFAULT, 1)
    return out


def _ns_library_points(group_models, specs_by_model, base_params_by_model, *, n_cfg: int,
                       n_cfg_mcmc: int, seed: int, varied_specs: Optional[dict] = None,
                       search_fidelity: bool = False) -> Dict[str, List[Tuple[np.ndarray, list]]]:
    """ver4-48, R2 (stage N0): the configuration library of the neural stack. Pure (no fits).

    Why: the network chooses among a FIXED set of fitted configurations per model (a softmax gate
    over a library) instead of searching the continuous hyperparameter space, so the set must be
    decided before any fit. Per model `m`: entry 0 is the model's CONFIGURED parameters, decoded
    with `skip_decode=True` (exact, never the lossy `_decode(_encode(base))` round trip), so the
    lambda = infinity end of the path is exactly the untuned naive ensemble; entries 1..L-1 are
    Latin-hypercube points of the unit cube (`scipy.stats.qmc.LatinHypercube`, seeded by `seed +
    zlib.crc32(m) + _JOINT_NS_SEED_OFFSET`, I9) decoded by `_decode_model_params()` and with
    explainability off. `L` = `n_cfg_mcmc` for a model in `MCMC_LENGTH_FIELDS` (its fits are
    minutes long), else `n_cfg`; a model with no tunable dimension has `L = 1`. A point whose
    decoded parameters equal an earlier entry's (`_test_sink_key`) is dropped, so a model with a
    tiny space can end up with fewer than `L` entries.

    Args: `group_models`; `specs_by_model` {m: [ParamSpec]} - the dimensions of the unit-cube
    point every other part of the search uses (`best_x` is their concatenation);
    `base_params_by_model`; `n_cfg`, `n_cfg_mcmc`; `seed`; `varied_specs` {m: [ParamSpec]} - the
    dimensions the Latin hypercube may VARY (the caller passes the `freeze_fidelity=True` and
    `search_space` filtered lists, so a run-length field stays at its configured coordinate in
    every entry); `None` varies every dimension of `specs_by_model[m]`; `search_fidelity` decode
    the entries with the search-phase cheapening (`ns_fidelity='search'`).

    Returns `{m: [(x, params), ...]}` with `x` of length `len(specs_by_model[m])` (entry 0's `x`
    is `_encode_from_params(...)`, the same coordinates as the search's seed point) and `params`
    the decoded hyperparameter list of the fit. Raises whatever `_decode_model_params()` raises
    for a malformed chunk."""
    from scipy.stats import qmc

    out: Dict[str, List[Tuple[np.ndarray, list]]] = {}
    for m in group_models:
        specs = specs_by_model[m]
        d = len(specs)
        base = base_params_by_model[m]
        x0 = (np.asarray(_encode_from_params(specs, base), dtype=float) if d
              else np.array([], dtype=float))
        p0 = _disable_explainability(m, _decode_model_params(
            m, x0, specs=specs, base_params=base, search_fidelity=search_fidelity,
            skip_decode=True))
        entries: List[Tuple[np.ndarray, list]] = [(x0.copy(), p0)]
        n_entries = int(n_cfg_mcmc) if schema_key_of(m) in MCMC_LENGTH_FIELDS else int(n_cfg)
        varied = specs if varied_specs is None else varied_specs.get(m, [])
        varied_idx = sorted(i for i, sp in enumerate(specs)
                            if any(sp.index == v.index for v in varied))
        if d > 0 and varied_idx and n_entries > 1:
            sampler = qmc.LatinHypercube(
                d=len(varied_idx),
                seed=int(seed) + zlib.crc32(m.encode('utf-8')) + _JOINT_NS_SEED_OFFSET)
            points = sampler.random(n_entries - 1)
            seen = {_test_sink_key(m, p0)}
            for u in points:
                x = x0.copy()
                x[varied_idx] = np.asarray(u, dtype=float)
                params = _disable_explainability(m, _decode_model_params(
                    m, x, specs=specs, base_params=base, search_fidelity=search_fidelity))
                key = _test_sink_key(m, params)
                if key in seen:
                    continue
                seen.add(key)
                entries.append((x, params))
        out[m] = entries
    return out


def _neural_stack_search(*, group_models, specs_by_model, base_params_by_model, seed_point,
                         fit_ctx: dict, settings: dict, state, seed: int
                         ) -> Tuple[np.ndarray, float, dict]:
    """ver4-48, R2: `search_strategy='neural_stack'` - the hyperparameter choice of every model
    and the ensemble weights are the trainable weights of ONE network
    (`models/hyper_stack.py`), fitted jointly on out-of-fold columns over train+valid.

    Why (blueprint R1/R2): the earlier joint strategies pick the argmax of noisy scores (winner's
    curse) and estimate weights on columns other than the deployed ones. Here the library of
    fitted configurations is the network's input: `W = softmax(a)` weights the models,
    `q_m = softmax(b_m)` gates the configurations of model m, the column weight is `W_m * q_ml`,
    the loss is `-ensemble_score + lambda * Omega`, and `lambda` is chosen by repeated K-fold CV
    along a path whose lambda = infinity end is EXACTLY the untuned naive ensemble. No model has
    to be differentiable: the network sits on top of ordinary `ModelTrialRunner` fits, so effects
    and interactions still come from the real fitted models (I3).

    Stages (every row is a train+valid row; the test split is only passed through to the fits, I7)
      N0  `_ns_library_points()`: entry 0 = the configured parameters, L-1 Latin-hypercube points.
      N1  ONE `_cv_fit_columns()` call fits every (model, entry) out of fold (stage 'oof', or
          'search' with `ns_fidelity='search'`, which also makes the cheapened fits eligible for
          the exact engine / test stub). A failed non-entry-0 column is dropped (event
          `N18:<model>:<entry>`); a failed entry 0 (or a constant one, with `dead_column_guard`)
          makes the model dead (weight 0, event `N18:<model>:0`), or - `safety_nets=False` -
          raises `_PureModeError`.
      N2  `hyper_stack.lambda_path()`: lambda by repeated CV (`ns_lambdas` plus infinity unless
          `safety_nets=False`; rule `ns_lambda_rule`, forced to 'min' without the nets), score
          `_NS_SCORE_FNS[ns_objective]` (`_ensemble_score_of` by default - the single score
          authority, I3 - or, ver4-50, `_pearson_score_of`). With `ns_swap_rule='paired_se'` the
          fitted network of every CV item is kept and, for every model whose gate leaves entry 0,
          `hyper_stack.swap_test()` admits the switch only when its paired CV gain exceeds its
          Nadeau-Bengio standard error; a rejected switch sets that gate to delta_0 (event
          `N21:<model>:<entry>`; N20 is already the ver4-48 `N20:valid_rows` event) before the
          ver4-49 shrink. ver4-51: `ns_swap_rule='paired_refit'` judges the switch against the
          network REFITTED with that model's gate clamped to entry 0, on a fresh partition
          (`hyper_stack.refit_swap_test`), and N2 + N2b + the shrink + the deployed vector are ONE
          pure function, `hyper_stack.stack_procedure`. No model fit is added.
      N2c (ver4-51, `ns_outer_gate='nested'`, finite lambda* only) `hyper_stack.nested_gate()`: an
          outer repeated K-fold re-runs `stack_procedure` on the inner rows and scores the DEPLOYED
          weights on the outer rows against the equal-weight ensemble; the fitted network is kept
          only when the mean gain exceeds one Nadeau-Bengio standard error, otherwise the untuned
          equal-weight ensemble is deployed (`stats['wf']['name']='ns_outer_prior'`, event
          `N22:<gain>:<se>`, `N22:too_small` or `N22:error`). Not run under `safety_nets=False`.
      N3  deployment: the primary of a model is its most probable entry (argmax q, ties -> entry
          0). `ns_deploy='mixture'`: the members are the columns with weight >= 0.02 (at most
          `library_members_max`, heaviest first, renormalised) in the schema of
          `_library_ensemble_members()`; `'argmax'`: no members, the model weights `W_m`.
      N4  hand-back: `stats['wf']` (the fixed vector the existing weight-first deployment path
          uses, so no gate re-estimates it), `stats['lib']`, `stats['lib_cur']`, `stats['ns']`.

    Args: as `_weight_first_cv_search()`. `settings` keys: ns_configs, ns_configs_mcmc,
    ns_lambdas, ns_lambda_rule, ns_repeats, ns_deploy, ns_fidelity, ns_weight_shrink,
    ns_objective, ns_swap_rule, ns_outer_gate, ns_outer_folds, ns_outer_repeats, ns_varied_specs (see
    `_ns_library_points`), safety_nets, fallback_events, dead_column_guard,
    library_members_max, inner_split_seed_base, objective_mode. `state` is unused (kept for the
    shared signature).

    Returns `(best_x, cv_score_at_lambda_star, stats)`: `best_x` is the concatenation of the
    primaries' unit-cube points, the score is the repeated-CV mean of the chosen lambda (the
    in-sample score when it is missing), `stats['ns']` holds the details payload (`lambda`,
    `rule`, `lambdas`, `path`, `W`, `q`, `effective`, `members`, `n_cols`, `dropped`, `entropy`,
    `converged`, `fidelity`, `deploy`, since ver4-50 `ns_objective`, `swap_rule`, `swaps` and, since
    ver4-51, `ns_outer_gate`, `outer` (the nested gate's dict or None), `lambda_cv`) plus
    `lm` (the member set or None).

    Raises ValueError when the CV objective is not active (the network needs out-of-fold
    columns), when no model has a usable column, or when `hyper_stack` rejects its inputs, and
    `_PureModeError` for the `safety_nets=False` cases above (the caller turns a ValueError into
    the monolithic fallback, or into `_PureModeError` without the nets)."""
    from .hyper_stack import (HyperStackFitError, StackSpec, deployed_column_weights,
                              nested_gate, prior_fit, stack_procedure)

    t_start = time.perf_counter()
    K = len(group_models)
    dims = {m: len(specs_by_model[m]) for m in group_models}
    offsets, _off = {}, 0
    for m in group_models:
        offsets[m] = _off
        _off += dims[m]
    seed_point = np.asarray(seed_point, dtype=float)
    actual = np.asarray(fit_ctx['actual'], dtype=float)
    n_rows = int(len(actual))
    if fit_ctx.get('objective_data') != 'cv_train_valid':
        raise ValueError(
            "[joint_tuning] search_strategy='neural_stack' needs objective_data='cv_train_valid' "
            "(the network is fitted on out-of-fold columns over train+valid), but this task runs "
            "the validation objective.")
    n_expected = len(fit_ctx['train']) + len(fit_ctx['valid'])
    if n_rows != int(n_expected):
        raise AssertionError(
            f"[joint_tuning] internal error: the neural-stack scoring vector has {n_rows} "
            f"row(s), expected {int(n_expected)} (train+valid).")
    partition = list(fit_ctx['partition'])
    rc = dict(fit_ctx.get('runner_ctx') or {})
    safety = bool(settings.get('safety_nets', _JOINT_SAFETY_NETS_DEFAULT))
    events = settings.get('fallback_events')
    if events is None:
        events = []
    inner_seed = int(settings.get('inner_split_seed_base', 0))
    n_cfg = int(settings.get('ns_configs', _JOINT_NS_CONFIGS_DEFAULT))
    n_cfg_mcmc = int(settings.get('ns_configs_mcmc', _JOINT_NS_CONFIGS_MCMC_DEFAULT))
    lambdas = [float(v) for v in settings.get('ns_lambdas', _JOINT_NS_LAMBDAS_DEFAULT)]
    rule = settings.get('ns_lambda_rule', _JOINT_NS_LAMBDA_RULE_DEFAULT)
    if not safety:
        rule = 'min'                       # blueprint R2: no parsimony pull towards lambda = inf
    repeats = max(1, int(settings.get('ns_repeats', _JOINT_NS_REPEATS_DEFAULT)))
    weight_shrink = float(settings.get('ns_weight_shrink', _JOINT_NS_WEIGHT_SHRINK_DEFAULT))
    deploy = settings.get('ns_deploy', _JOINT_NS_DEPLOY_DEFAULT)
    fidelity = settings.get('ns_fidelity', _JOINT_NS_FIDELITY_DEFAULT)
    # ver4-50 (I11: legacy default, resolved value logged by the caller)
    objective = settings.get('ns_objective', _JOINT_NS_OBJECTIVE_DEFAULT)
    if objective not in _NS_SCORE_FNS:
        raise ValueError(
            f"[joint_tuning] neural_stack ns_objective={objective!r} is not one of "
            f"{list(_NS_SCORE_FNS)!r}")
    swap_rule = settings.get('ns_swap_rule', _JOINT_NS_SWAP_RULE_DEFAULT)
    if swap_rule not in _JOINT_NS_SWAP_RULE_CHOICES:
        raise ValueError(
            f"[joint_tuning] neural_stack ns_swap_rule={swap_rule!r} is not one of "
            f"{list(_JOINT_NS_SWAP_RULE_CHOICES)!r}")
    # ver4-51 (I11: legacy defaults; the resolved values are logged by the caller)
    outer_gate = settings.get('ns_outer_gate', _JOINT_NS_OUTER_GATE_DEFAULT)
    if outer_gate not in _JOINT_NS_OUTER_GATE_CHOICES:
        raise ValueError(
            f"[joint_tuning] neural_stack ns_outer_gate={outer_gate!r} is not one of "
            f"{list(_JOINT_NS_OUTER_GATE_CHOICES)!r}")
    outer_folds = max(2, int(settings.get('ns_outer_folds', _JOINT_NS_OUTER_FOLDS_DEFAULT)))
    outer_repeats = max(1, int(settings.get('ns_outer_repeats', _JOINT_NS_OUTER_REPEATS_DEFAULT)))
    ns_score = _NS_SCORE_FNS[objective]
    members_max = max(2, int(settings.get('library_members_max',
                                          _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT)))
    dead_guard = bool(settings.get('dead_column_guard', _JOINT_DEAD_COLUMN_GUARD_DEFAULT))
    folds = max(2, int(fit_ctx.get('cv_folds') or len(partition)))
    if settings.get('objective_mode') == 'dpt_ratio':
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='neural_stack' scores the "
              f"network with its own ns_objective={objective!r} ("
              f"{'pearson - mse/var' if objective == 'ensemble_score' else 'pearson'}); this "
              f"task's objective_mode='dpt_ratio' does not apply to it.")
    stats = {'evals_by_model': {m: 0 for m in group_models},
             'rounds_by_model': {m: 0 for m in group_models}, 'lib_sizes': {}, 'stop': '',
             'wall_s': 0.0, 'seed_ens': None, 'swaps': 0, 'dead_models': []}

    # ---- N0: the library design ----------------------------------------------------------------
    lib_pts = _ns_library_points(
        group_models, specs_by_model, base_params_by_model, n_cfg=n_cfg, n_cfg_mcmc=n_cfg_mcmc,
        seed=int(seed) + inner_seed, varied_specs=settings.get('ns_varied_specs'),
        search_fidelity=(fidelity == 'search'))

    # ---- N1: every (model, entry) out of fold, one batch -----------------------------------------
    requests: List[Tuple[str, list]] = []
    owners: List[Tuple[str, int]] = []
    for m in group_models:
        for l, (_x, params) in enumerate(lib_pts[m]):
            requests.append((m, params))
            owners.append((m, l))
    rc_n1 = dict(rc)
    rc_n1['stage'] = 'search' if fidelity == 'search' else 'oof'
    print(f"[GP] Joint Bayesian optimisation neural-stack N1: fitting {len(requests)} "
          f"configuration(s) ({ {m: len(lib_pts[m]) for m in group_models} }) x {len(partition)} "
          f"folds = {len(requests) * len(partition)} out-of-fold fits at "
          f"{'search' if fidelity == 'search' else 'full'} fidelity (stage "
          f"{rc_n1['stage']!r}).")
    cols = _cv_fit_columns(
        requests, tv=fit_ctx['tv'], partition=partition, seed=fit_ctx['cv_seed'],
        test=fit_ctx['test'], runner=fit_ctx['runner'], RESULT_NAME=fit_ctx['RESULT_NAME'],
        runner_ctx=rc_n1, stop_on_failure=False)
    kept: Dict[str, List[Tuple[int, np.ndarray, np.ndarray]]] = {m: [] for m in group_models}
    dropped: List[Tuple[str, int]] = []
    for (m, l), col in zip(owners, cols):
        if col is None:
            dropped.append((m, l))
        else:
            kept[m].append((l, lib_pts[m][l][0], np.asarray(col, dtype=float)))
    failed_entry0 = [m for m in group_models if not kept[m] or kept[m][0][0] != 0]
    if failed_entry0 and not safety:
        raise _PureModeError(
            f"[joint_tuning] safety_nets=False: the configured (entry 0) column of "
            f"{failed_entry0!r} could not be fitted out of fold, and a model cannot be dropped "
            f"from the neural stack without the safety nets - see the [GP] NOTE above for the "
            f"failed fit.")
    for m, l in dropped:
        events.append(f"N18:{m}:{l}")
    if dropped:
        print(f"[GP] NOTE: Joint Bayesian optimisation neural-stack N1: "
              f"{len(dropped)} column(s) could not be fitted and are dropped from the library: "
              f"{[f'{m}:{l}' for m, l in dropped]!r} (event N18).")
    dead_models = list(failed_entry0)
    const_dead: List[str] = []
    if dead_guard:
        const_dead = [m for m in group_models if m not in dead_models
                      and _is_dead_column(kept[m][0][2], actual, partition)]
        if len(const_dead) >= len([m for m in group_models if m not in dead_models]):
            if const_dead:
                print("[GP] NOTE: Joint Bayesian optimisation neural-stack: the configured column "
                      "of EVERY usable model is constant - nothing can be dropped, so no model "
                      "is zero-weighted for this task.")
            const_dead = []
        elif const_dead:
            print(f"[GP] WARN: Joint Bayesian optimisation neural-stack: the configured column of "
                  f"{const_dead!r} is CONSTANT (no genetic signal) - those model(s) get weight 0.")
    dead_models = [m for m in group_models if m in set(dead_models) | set(const_dead)]
    live = [m for m in group_models if m not in dead_models]
    if not live:
        raise ValueError(
            "[joint_tuning] search_strategy='neural_stack': no model has a usable configured "
            "column (every entry-0 fit failed).")
    if failed_entry0:
        print(f"[GP] NOTE: Joint Bayesian optimisation neural-stack N1: the configured column of "
              f"{failed_entry0!r} could not be fitted - those model(s) get weight 0 (event N18).")
    stats['dead_models'] = list(dead_models)

    # ---- the network's input -----------------------------------------------------------------------
    new_lib: Dict[str, list] = {}
    cols_live: List[np.ndarray] = []
    sizes: List[int] = []
    entry_orig: Dict[str, List[int]] = {}
    for m in group_models:
        if m in failed_entry0:
            new_lib[m] = [(lib_pts[m][0][0], np.full(n_rows, np.nan, dtype=float))]
            entry_orig[m] = [0]
            continue
        new_lib[m] = [(x, col) for (_l, x, col) in kept[m]]
        entry_orig[m] = [l for (l, _x, _c) in kept[m]]
        if m in live:
            sizes.append(len(kept[m]))
            cols_live.extend(col for (_l, _x, col) in kept[m])
    P = np.column_stack(cols_live)
    if P.shape != (n_rows, int(sum(sizes))):
        raise AssertionError(
            f"[joint_tuning] internal error: the neural-stack matrix has shape {P.shape}, "
            f"expected ({n_rows}, {int(sum(sizes))}).")
    stats['lib_sizes'] = {m: len(new_lib[m]) for m in group_models}
    stats['evals_by_model'] = {m: len(new_lib[m]) for m in group_models}
    p0_cols = [kept[m][0][2] for m in live]
    stats['seed_ens'] = ns_score(np.mean(np.column_stack(p0_cols), axis=1), actual)

    # ---- N2: lambda by repeated CV, the network refitted at lambda* ------------------------------------
    # ver4-51 (R2): stages N2, N2b and the shrink are ONE pure function, `hyper_stack.stack_procedure`,
    # so the nested outer gate below (N2c) runs the identical procedure on its inner rows.
    include_prior = bool(safety)
    ns_seed = int(seed) + inner_seed + _JOINT_NS_SEED_OFFSET
    spec = StackSpec(
        lambdas=tuple(lambdas), include_prior=include_prior, folds=int(folds),
        repeats=int(repeats), seed=int(ns_seed), rule=rule, objective=objective,
        nb_correction=True, swap_rule=swap_rule, shrink=float(weight_shrink), deploy=deploy,
        member_min_weight=float(_JOINT_LIBRARY_MEMBER_MIN_WEIGHT), members_max=int(members_max),
        max_iter=_JOINT_NS_MAX_ITER, tol=_JOINT_NS_TOL,
        swap_seed_offset=_JOINT_NS_SWAP_SEED_OFFSET)
    try:
        fit, _w_dep, proc = stack_procedure(P, actual, sizes, spec, ns_score)
    except HyperStackFitError as exc:
        # only reachable without the nets: with them `stack_procedure` falls back to the prior
        raise _PureModeError(
            f"[joint_tuning] safety_nets=False: no lambda of the neural stack could be fitted "
            f"({exc}) and falling back to the untuned naive ensemble is switched off.") from exc
    info = proc['lambda_info']
    lambda_note = proc['lambda_note']
    if safety and lambdas and int(info.get('n_finite_ok', 0)) == 0:
        events.append('N19:ns_prior_only')
        lambda_note = lambda_note or 'no finite lambda could be fitted in every fold'
    if lambda_note:
        print(f"[GP] NOTE: Joint Bayesian optimisation neural-stack N2: {lambda_note} - the "
              f"untuned naive ensemble (lambda=inf) is deployed for this task (event N19).")
    lam = float(fit.lam)
    lam_txt = 'inf' if not math.isfinite(lam) else f"{lam:g}"
    lambda_cv = lam                                  # lambda* of the cross-validation (before N2c)
    # ver4-50 (R3) / ver4-51 (R1): the swap test. Every model whose all-rows gate leaves entry 0 must
    # beat its counterfactual by more than the Nadeau-Bengio standard error, otherwise its gate was
    # reset to delta_0 inside `stack_procedure` (so 'argmax' and 'mixture' both deploy the
    # configured entry 0). 'paired_se': the fitted network with the gate overwritten; 'paired_refit':
    # the network REFITTED with the gate clamped, on a fresh partition. Zero model fits.
    swap_rows: List[dict] = []
    for row in proc['swaps']:
        m_name = live[row['m']]
        entry_orig_id = int(entry_orig[m_name][row['entry']])
        swap_row = {
            'model': m_name, 'entry': entry_orig_id, 'admitted': bool(row['admitted']),
            'gap': (None if not math.isfinite(row['gap']) else float(row['gap'])),
            'se': (None if not math.isfinite(row['se']) else float(row['se'])),
            'reason': row['reason']}
        if swap_rule == 'paired_refit':
            swap_row['gap_forced'] = (None if not math.isfinite(row['gap_forced'])
                                      else float(row['gap_forced']))
        swap_rows.append(swap_row)
        if not row['admitted']:
            events.append(f"N21:{m_name}:{entry_orig_id}")
    if swap_rows:
        _refit = (swap_rule == 'paired_refit')
        print("[GP] Joint Bayesian optimisation neural-stack N2b ("
              + ("refit swap test, " if _refit else "paired swap test, ")
              + objective + "): " + '; '.join(
                  f"{r['model']} entry {r['entry']}: gain="
                  + ('n/a' if r['gap'] is None else f"{r['gap']:+.4f}")
                  + ((" gain_forced=" + ('n/a' if r['gap_forced'] is None
                                         else f"{r['gap_forced']:+.4f}")) if _refit else "")
                  + " se=" + ('n/a' if r['se'] is None else f"{r['se']:.4f}") + " -> "
                  + ('admitted' if r['admitted'] else
                     'rejected' + (f" ({r['reason']})" if r['reason'] else ''))
                  for r in swap_rows)
              + " (a rejected switch keeps the configured entry 0; event N21).")
    elif proc['swap_skipped'] == 'lambda_inf':
        print(f"[GP] NOTE: Joint Bayesian optimisation neural-stack ns_swap_rule={swap_rule!r}: "
              "lambda=inf was chosen, so every model keeps its configured entry 0 and there is "
              "no switch to test.")
    # ver4-49: the fitted weights were pulled towards the untuned naive ensemble inside
    # `stack_procedure` (no-op at lambda = infinity); `W` before that shrink:
    w_model_raw = list(proc['w_model_raw'])

    # ---- N2c (ver4-51, R2): the nested outer gate ---------------------------------------------------
    # Does the WHOLE procedure above beat the equal-weight ensemble out of sample? Otherwise the
    # untuned equal-weight ensemble is deployed (decision 'ns_outer_prior'). Zero model fits.
    outer: Optional[dict] = None
    members = proc['members']
    if outer_gate == 'nested':
        if not safety:
            print("[GP] NOTE: Joint Bayesian optimisation neural-stack ns_outer_gate='nested' is "
                  "ignored under safety_nets=False (the gate's rejection deploys the untuned "
                  "equal-weight ensemble, which the pure mode never falls back to).")
        elif not math.isfinite(lam):
            outer = {'gain': float('nan'), 'se': float('nan'), 'n_items': 0, 'admitted': False,
                     'reason': 'lambda_inf', 'inner_prior_share': float('nan'), 'wall_s': 0.0,
                     'mode': 'none', 'workers': 0, 'decision': 'not_needed', 'replaced': False}
            print("[GP] NOTE: Joint Bayesian optimisation neural-stack ns_outer_gate='nested': "
                  "lambda=inf was chosen, so the untuned equal-weight ensemble is already "
                  "deployed and there is no fitted network to gate.")
        else:
            try:
                outer = dict(nested_gate(
                    P, actual, sizes, spec, ns_score, outer_folds=outer_folds,
                    outer_repeats=outer_repeats, seed=int(ns_seed) + _JOINT_NS_OUTER_SEED_OFFSET,
                    n_jobs=rc.get('n_jobs', 1), z=_JOINT_NS_OUTER_Z))
            except (ValueError, HyperStackFitError) as exc:
                print(f"[GP] WARN: Joint Bayesian optimisation neural-stack N2c: the nested gate "
                      f"failed ({type(exc).__name__}: {exc}) - the untuned equal-weight ensemble "
                      f"is deployed for this task (event N22).")
                outer = {'gain': float('nan'), 'se': float('nan'), 'n_items': 0,
                         'admitted': False, 'reason': 'error', 'inner_prior_share': float('nan'),
                         'wall_s': 0.0, 'mode': 'none', 'workers': 0}
            outer['decision'] = ('admitted' if outer['admitted'] else
                                 outer['reason'] if outer['reason'] in ('too_small', 'error')
                                 else 'rejected')
            outer['replaced'] = not bool(outer['admitted'])
            if outer['reason'] == 'too_small':
                print(f"[GP] NOTE: Joint Bayesian optimisation neural-stack N2c: too few rows for "
                      f"{outer_folds} outer folds x the inner {folds}-fold lambda path "
                      f"({n_rows} rows) - the gate is not run and the untuned equal-weight "
                      f"ensemble is deployed (event N22).")
                events.append('N22:too_small')
            elif outer['reason'] == 'error':
                events.append('N22:error')
            else:
                print(f"[GP] Joint Bayesian optimisation neural-stack N2c (nested gate, "
                      f"{outer_folds} folds x {outer_repeats} repeats, {outer['workers']} "
                      f"worker(s) {outer['mode']}): gain over equal weights="
                      + ('n/a' if not math.isfinite(outer['gain']) else f"{outer['gain']:+.4f}")
                      + " se=" + ('n/a' if not math.isfinite(outer['se'])
                                  else f"{outer['se']:.4f}")
                      + f" (z={_JOINT_NS_OUTER_Z:g}), inner lambda=inf share="
                      + ('n/a' if not math.isfinite(outer['inner_prior_share'])
                         else f"{outer['inner_prior_share']:.2f}")
                      + " -> " + ('admitted: the fitted network is deployed'
                                  if outer['admitted'] else
                                  "rejected: the untuned equal-weight ensemble is deployed "
                                  + (f"({outer['reason']}) " if outer['reason'] else "")
                                  + "(event N22)")
                      + f" ({outer['wall_s']:.1f} s).")
                if not outer['admitted']:
                    events.append(f"N22:{outer['gain']:+.4f}:{outer['se']:.4f}")
            if outer['replaced']:
                fit = prior_fit(sizes)
                lam = float(fit.lam)
                lam_txt = 'inf'
                _w_dep, members = deployed_column_weights(
                    fit, sizes, deploy, member_min_weight=float(_JOINT_LIBRARY_MEMBER_MIN_WEIGHT),
                    members_max=int(members_max))

    # ---- N3: deployment ---------------------------------------------------------------------------------
    live_pos = {m: i for i, m in enumerate(live)}
    slices, _o = {}, 0
    for m, sz in zip(live, sizes):
        slices[m] = slice(_o, _o + sz)
        _o += sz
    primary = {}
    for m in group_models:
        if m in live_pos:
            primary[m] = int(np.argmax(np.asarray(fit.q[live_pos[m]], dtype=float)))
        else:
            primary[m] = 0
    best_x = (np.concatenate([np.asarray(new_lib[m][primary[m]][0], dtype=float)
                              for m in group_models])
              if sum(dims.values()) else np.array([], dtype=float))
    if best_x.shape != seed_point.shape:
        raise AssertionError(
            f"[joint_tuning] internal error: the neural-stack point has {best_x.shape[0]} "
            f"coordinate(s), the search space has {seed_point.shape[0]}.")
    w_model = np.zeros(K, dtype=float)
    for m in live:
        w_model[group_models.index(m)] = float(fit.w_model[live_pos[m]])
    lm = None
    if deploy == 'mixture':
        if members is not None:
            # `deployed_column_weights` is the single authority of the member rule (>= the minimum
            # weight, heaviest first, at most `members_max`, renormalised); the raw weights are
            # the fit's own column weights.
            w_cols = np.asarray(fit.w, dtype=float)
            lm = {'members': [{'model': live[mp], 'entry': int(j), 'weight': float(wk),
                               'primary': bool(primary[live[mp]] == j),
                               'raw_weight': float(w_cols[slices[live[mp]]][j])}
                              for (mp, j, wk) in members],
                  'n_pruned': int(w_cols.size), 'weights_raw': [float(c) for c in w_cols]}
        else:
            print("[GP] NOTE: Joint Bayesian optimisation neural-stack N3: no library column has "
                  f"a weight >= {_JOINT_LIBRARY_MEMBER_MIN_WEIGHT:g} - the model weights W_m "
                  "weight the primaries (as ns_deploy='argmax').")
    arm = 'untuned' if all(primary[m] == 0 for m in group_models) else 'tuned'

    # ---- diagnostics ----------------------------------------------------------------------------------------
    q_by_model = {m: [float(v) for v in np.asarray(fit.q[live_pos[m]], dtype=float)]
                  if m in live_pos else [1.0] for m in group_models}
    effective: Dict[str, dict] = {}
    for m in live:
        qv = np.asarray(fit.q[live_pos[m]], dtype=float)
        eff = {}
        for sp in specs_by_model[m]:
            vals = [lib_pts[m][entry_orig[m][j]][1][sp.index] for j in range(len(qv))]
            if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                   for v in vals):
                eff[sp.name] = float(np.dot(qv, np.asarray(vals, dtype=float)))
            else:
                mass: Dict[str, float] = {}
                for v, p in zip(vals, qv):
                    mass[str(v)] = mass.get(str(v), 0.0) + float(p)
                eff[sp.name] = max(mass.items(), key=lambda kv: kv[1])[0]
        effective[m] = eff
    gated = [np.asarray(fit.q[live_pos[m]], dtype=float) for m in live if sizes[live_pos[m]] > 1]
    entropy = (float(np.mean([-np.sum(g[g > 0] * np.log(g[g > 0])) for g in gated]))
               if gated else 0.0)
    path_rows = [(row[0], row[1], row[2]) for row in info['path']]
    chosen_row = next((r for r in path_rows
                       if (math.isinf(r[0]) and math.isinf(lam)) or (r[0] == lam)), None)
    if chosen_row is not None and chosen_row[1] is not None:
        best_score = float(chosen_row[1])
    else:
        _ins = ns_score(P @ np.asarray(fit.w, dtype=float), actual)
        best_score = float(_ins) if _ins is not None else -math.inf
    w_names = {m: float(w_model[i]) for i, m in enumerate(group_models)}
    ns = {'lambda': lam, 'rule': info.get('rule', rule), 'lambdas': list(info['lambdas']),
          'path': path_rows, 'W': w_names, 'q': q_by_model, 'effective': effective,
          'members': (len(lm['members']) if lm is not None else None), 'n_cols': int(P.shape[1]),
          'dropped': [f"{m}:{l}" for m, l in dropped], 'entropy': entropy,
          'converged': bool(fit.converged), 'fidelity': fidelity, 'deploy': deploy,
          'objective': float(fit.objective), 'n_iter': int(fit.n_iter), 'lm': lm,
          'weight_shrink': (weight_shrink if math.isfinite(lam) else 0.0),
          'W_fitted': {m: w_model_raw[live_pos[m]] for m in live},
          # ver4-50
          # (`objective` above is the final LOSS value, so the training objective is `ns_objective`)
          'ns_objective': objective, 'swap_rule': swap_rule, 'swaps': swap_rows,
          # ver4-51: the nested outer gate (None when ns_outer_gate='off' or not run) and the
          # lambda* of the cross-validation before that gate (== 'lambda' unless it replaced it)
          'ns_outer_gate': outer_gate, 'outer': outer, 'lambda_cv': lambda_cv,
          'outer_folds': int(outer_folds), 'outer_repeats': int(outer_repeats),
          # the lambda-choice CV loop, so `tools/joint_replay.py` can reproduce this fit exactly
          'folds': int(folds), 'repeats': int(repeats)}
    _outer_replaced = bool(outer is not None and outer.get('replaced'))
    name = ('ns_outer_prior' if _outer_replaced
            else 'ns_prior' if not math.isfinite(lam) else f"ns_lambda={lam_txt}")
    stats['wf'] = {'name': name, 'w': [float(x) for x in w_model], 'arm': arm, 'c': 0,
                   'pick_how': ('nested_gate' if _outer_replaced
                                else 'cv_lambda_' + str(info.get('rule', rule))),
                   'n_candidates': len(info['lambdas']), 'rungs_run': 0, 'table': [],
                   'rules': [name], 'rows': 'oof', 'direction': 'neural_stack',
                   'fidelity': fidelity, 'safety_nets': safety}
    stats['ns'] = ns
    # the decoded parameters each retained library entry was ACTUALLY fitted with, so the library
    # dump can key the test columns exactly (entry 0 is fitted with the configured parameters, which
    # `_decode(_encode(base))` need not reproduce) - dump-only, never read by the search
    stats['ns_entry_params'] = {m: [lib_pts[m][l][1] for l in entry_orig[m]] for m in group_models}
    stats['lib'] = new_lib
    stats['lib_cur'] = dict(primary)
    stats['stop'] = 'complete'
    stats['wall_s'] = time.perf_counter() - t_start
    print(f"[GP] Joint Bayesian optimisation neural-stack N2: lambda path (rule {ns['rule']!r}, "
          f"{folds} folds x {repeats} repeats, mean CV score): "
          + ', '.join(f"{('inf' if math.isinf(r[0]) else format(r[0], 'g'))}="
                      + ('n/a' if r[1] is None else f"{r[1]:.4f}") for r in path_rows)
          + f" -> lambda*={'inf' if not math.isfinite(lambda_cv) else format(lambda_cv, 'g')} "
          f"(converged={bool(fit.converged)}, iterations={int(fit.n_iter)})"
          + (f"; the nested outer gate (N2c) replaced it by the untuned equal-weight ensemble "
             f"(lambda=inf)" if _outer_replaced else "") + ".")
    swap_txt = [r['model'] + ':' + str(r['entry']) + ':'
                + ('admitted' if r['admitted'] else 'rejected') for r in swap_rows]
    print(f"[GP] Joint Bayesian optimisation neural-stack N3: model weights W="
          f"{ {m: round(v, 4) for m, v in w_names.items()} } (fitted before the shrink "
          f"s={ns['weight_shrink']:g}: { {m: round(v, 4) for m, v in ns['W_fitted'].items()} }); most probable entry per model="
          f"{primary}; gate entropy={entropy:.3f} nats; {P.shape[1]} library column(s)"
          + (f"; {len(lm['members'])} member(s)" if lm is not None else
             f"; deploy={deploy!r} without members")
          + (f"; effective values={ {m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()} for m, e in effective.items()} }"
             if effective and math.isfinite(lam) else "")
          + (f"; swaps={swap_txt}" if swap_rows else "") + ".")
    return best_x, best_score, stats


def _oof_pooled_matrix(group_models, final_params_by_model, train, test, runner, RESULT_NAME, *,
                        n_folds: int = 5, seed: int, n_jobs: int = 1, worker_init=None,
                        worker_init_args=(), cache: Optional['_FitCache'] = None,
                        ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """ver4-28, R4: out-of-fold validation predictions for every group
    model at its own FINAL (already search-selected) hyperparameters,
    fit on TRAIN rows only (I7 - `test` is untouched, `valid` plays no
    part in this function at all). Explainability is disabled for the
    OOF fits only (`_disable_explainability()` - full MCMC length/
    iterations otherwise), avoiding `n_folds`x the reporting overhead a
    full-fidelity fit would otherwise pay for a fit whose only purpose is
    estimating out-of-sample prediction error.

    Returns `(oof_pred_matrix, oof_actual)` - shape `(N_train, K)` /
    `(N_train,)`, row order matching `train`'s own row order - or `None`
    when `train` is too small for `n_folds` folds of at least 10 rows
    each, or any OOF fit unit fails (the caller falls back to
    `weight_source='valid'` for this task either way).
    """
    n = len(train)
    if n < 10 * n_folds:
        print(f"[GP] NOTE: Joint Bayesian optimisation weight_source='oof_train_valid' requires at "
              f"least {10 * n_folds} training rows for {n_folds} OOF fold(s) (this task's train "
              f"split has {n}) - falling back to weight_source='valid' for this task.")
        return None

    rng = np.random.default_rng(seed + _JOINT_OOF_SEED_OFFSET)
    row_ids = train.index.to_numpy()
    order = rng.permutation(row_ids)
    folds = np.array_split(order, n_folds)
    idx_pos = {row_id: pos for pos, row_id in enumerate(row_ids)}
    oof_pred = {m: np.full(n, np.nan, dtype=float) for m in group_models}

    for fi, held_out in enumerate(folds):
        held_out = np.asarray(held_out)
        if len(held_out) == 0:
            continue
        tr_fold = train.drop(index=held_out)
        va_fold = train.loc[held_out]
        units = [
            _FitUnit(model=m, params=_disable_explainability(m, list(final_params_by_model[m])),
                     split_tag=f'oof{fi}', train=tr_fold, valid=va_fold, fidelity='oof')
            for m in group_models
        ]
        fitted = _run_units(units, runner=runner, RESULT_NAME=RESULT_NAME, test=test,
                             n_jobs=n_jobs, worker_init=worker_init, worker_init_args=worker_init_args,
                             cache=cache)
        for m, result in zip(group_models, fitted):
            if result is None:
                print(f"[GP] NOTE: Joint Bayesian optimisation OOF fold {fi} failed refitting "
                      f"{m!r} - falling back to weight_source='valid' for this task.")
                return None
            pv = result.get('predicted_valid')
            if pv is None or len(pv) != len(held_out):
                print(f"[GP] NOTE: Joint Bayesian optimisation OOF fold {fi} for {m!r} returned "
                      f"{0 if pv is None else len(pv)} prediction(s) for {len(held_out)} held-out "
                      f"row(s) - falling back to weight_source='valid' for this task.")
                return None
            pv = np.asarray(pv, dtype=float)
            for pos_in_fold, row_id in enumerate(held_out):
                oof_pred[m][idx_pos[row_id]] = pv[pos_in_fold]

    pred_matrix = np.column_stack([oof_pred[m] for m in group_models])
    if not np.all(np.isfinite(pred_matrix)):
        print(f"[GP] NOTE: Joint Bayesian optimisation OOF matrix has unfilled row(s) after "
              f"{n_folds} fold(s) - falling back to weight_source='valid' for this task.")
        return None
    actual = np.asarray(train.iloc[:, -1], dtype=float)
    return pred_matrix, actual


def _estimate_joint_fits(group_models: Sequence[str], hparam_specs: dict,
                         budget: Optional[dict]) -> dict:
    """ver4-34 R1.b / ver4-36 R1.a: the full fit-count estimate of one joint-optimisation
    task. `estimate_joint_fit_count()` is the public, integer-only view of this dict; the
    engine itself also reads the two per-model tables (`search_by_model`, `rounds_by_model`)
    for the ETA line (`_log_eta()`).

    Keys: `search`, `gate`, `confirm`, `oof`, `total` (their sum, exactly the `total` the
    engine's `fits:` line reports - cache hits excluded), `seq_batches`,
    `bglr_full_length_fits`, and the ver4-36 additions `mcmc_fits` (fits of MCMC models, i.e.
    those in `MCMC_LENGTH_FIELDS`, that are NOT answered by the exact engine - the ones that
    cost minutes), `exact_fits` (search/gate fits answered by the exact kernel engine, plus its
    parity fits) and `critical_path` (the longest chain of SEQUENTIAL fit rounds: search
    rounds + gate rounds + the confirmatory round + out-of-fold rounds - what no amount of
    parallelism shortens), plus `search_by_model` / `rounds_by_model` (dicts). The
    ver4-40: `search_strategy='weight_first'` counts R0 plus `wf_rungs` rungs
    (`_wf_rung_counts`), no gate fits, no final out-of-fold refit (unless `library_dump`) and
    `K * oof_folds` more fits with `wf_candidate_rows='oof_train'`.
    ver4-42: `search_strategy='weight_first_cv'` counts R0 + the `async_library` figure of
    its stage S2 (nothing with `wf_cv_tune=False`), no gate fits, `K * V` more fits in the
    'oof' stage for stage S0 when `wf_cv_weight_fidelity='full'` (an upper bound: a model with
    no fidelity field is answered from the fit cache) and no final out-of-fold refit (unless
    `library_dump`).
    ver4-48: `search_strategy='neural_stack'` counts `L_m * V` out-of-fold fits per model (`L_m` =
    `ns_configs`, `ns_configs_mcmc` for an MCMC model, 1 without a varied dimension; an upper bound:
    a duplicate or failed configuration costs nothing) in the 'oof' stage ('search' stage with
    `ns_fidelity='search'`), no search and no gate fits, `K` confirm fits and, with
    `ns_deploy='mixture'`, at most `min(library_members_max, C - K)` extra member fits, `C` = the
    number of library columns. The CV objective is counted as active.
    ver4-36 keys read from `budget`: `search_strategy` (now also 'async_library'),
    `gate_mode`, `search_test_stub`, `search_engine`, `library_evals_per_dim`,
    `library_max_evals`, `library_refine_rounds`; with the stub or engine requested the
    model-level eligibility comes from `model_registry` (a model outside the registry sets is
    counted as an ordinary MCMC model), and the one reference fit (plus stub / exact parity
    fit) of `_parity_guards()` is counted in the search stage. ver4-38 keys read from
    `budget`: `cv_racing` / `racing_folds` (async_library, CV objective: each candidate costs
    folds * (k/folds + p * (1 - k/folds)) fits with p = 1/3 nominal - an upper bound, the real
    promotion rate is logged), `oof_fidelity` ('search' makes the out-of-fold stage cost 0 fits
    for a library strategy: its fold fits are cache hits), `library_ensemble` and
    `library_members_max` (adds at most members_max - K extra confirm fits). The stop rule
    (`library_stop_rule`) cannot be predicted, so the search figures stay upper bounds. Raises
    nothing for a model without tunable fields."""
    from model_registry import search_test_stub_safe, exact_kernel_search_capable

    budget = dict(budget or {})
    models = list(group_models)
    k_models = len(models)
    freeze = bool(budget.get('freeze_fidelity_fields', _JOINT_FREEZE_FIDELITY_FIELDS_DEFAULT))
    priority_only = (budget.get('search_space', _JOINT_SEARCH_SPACE_DEFAULT) == 'priority')
    specs_by_model, _unused = build_joint_param_specs(
        models, hparam_specs, None, freeze_fidelity=freeze, priority_only=priority_only)
    dims = {m: len(specs_by_model[m]) for m in models}
    total_dims = sum(dims.values())
    n_iter = int(budget.get('n_iter', 30) or 30)
    init_points = int(budget.get('init_points', 5) or 5)
    strategy = budget.get('search_strategy', _JOINT_SEARCH_STRATEGY_DEFAULT) \
        or _JOINT_SEARCH_STRATEGY_DEFAULT
    if strategy not in _JOINT_SEARCH_STRATEGIES or total_dims == 0 or k_models < 2:
        strategy = 'monolithic'
    cv = budget.get('objective_data', _JOINT_OBJECTIVE_DATA_DEFAULT) == 'cv_train_valid'
    if strategy == 'neural_stack':
        cv = True                      # ver4-48: the network is fitted on out-of-fold columns
    folds = max(2, int(budget.get('cv_folds', _JOINT_CV_FOLDS_DEFAULT) or _JOINT_CV_FOLDS_DEFAULT))
    reps = max(1, int(budget.get('valid_repeats', 1) or 1))
    parallel = bool(budget.get('parallel_trials', _JOINT_PARALLEL_TRIALS_DEFAULT))
    q = _JOINT_BATCH_Q
    unit = folds if cv else 1          # fits per column (block / library searches)

    def _rounds(evals):
        return int(math.ceil(max(1, evals) / float(q)))

    search_by_model = {m: 0 for m in models}
    rounds_by_model = {m: 0 for m in models}
    ns_cols_by_model = {m: 1 for m in models}      # ver4-48 (neural_stack only)
    search_batches = 0
    if strategy == 'monolithic':
        evals = 1 + init_points + n_iter if total_dims > 0 else 1
        per_eval = folds if cv else reps
        for m in models:
            search_by_model[m] = evals * per_eval
            rounds_by_model[m] = evals * (1 if cv else reps)
        search_batches = evals * (1 if cv else reps)
    elif strategy == 'block_coordinate':
        warm = (budget.get('warm_start', _JOINT_WARM_START_DEFAULT) == 'individual')
        block_init = int(budget.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)
                         or _JOINT_BLOCK_INIT_POINTS_DEFAULT)
        cycles = max(1, int(budget.get('block_cycles', _JOINT_BLOCK_CYCLES_DEFAULT)
                            or _JOINT_BLOCK_CYCLES_DEFAULT))
        refine = int(budget.get('block_refine_iter', _JOINT_BLOCK_REFINE_ITER_DEFAULT)
                     or _JOINT_BLOCK_REFINE_ITER_DEFAULT)
        n_iter_cfg = int(budget.get('block_n_iter', _JOINT_BLOCK_N_ITER_DEFAULT) or 0)
        warm_cfg = int(budget.get('block_warm_n_iter', _JOINT_BLOCK_WARM_N_ITER_DEFAULT) or 0)
        search_batches = k_models                      # R0: one column per call
        for m in models:
            d = dims[m]
            fits = 1                                   # the untuned column
            chain = 1
            if d > 0:
                evals_list = []
                if warm:
                    evals_list.append(max(1 + block_init, warm_cfg or max(10, 3 * d)))
                first = n_iter_cfg or (max(4, d) if warm else max(8, 2 * d))
                evals_list.append(max(1 + block_init, first))
                evals_list.extend([max(1 + block_init, refine)] * (cycles - 1))
                for ev in evals_list:
                    fits += ev - 1                     # the probe point is a cache hit
                    search_batches += _rounds(ev)
                    chain += _rounds(ev)
            search_by_model[m] = fits * unit
            rounds_by_model[m] = chain
    elif strategy == 'parallel_library':
        rounds = max(1, int(budget.get('library_rounds', _JOINT_LIBRARY_ROUNDS_DEFAULT)
                            or _JOINT_LIBRARY_ROUNDS_DEFAULT))
        lq = max(1, int(budget.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)
                        or _JOINT_LIBRARY_Q_DEFAULT))
        block_init = max(1, int(budget.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)
                                or _JOINT_BLOCK_INIT_POINTS_DEFAULT))
        for m in models:
            extra = (block_init + (rounds - 1) * lq) if dims[m] > 0 else 0
            search_by_model[m] = (1 + extra) * unit
            rounds_by_model[m] = (1 + rounds) if dims[m] > 0 else 1
        search_batches = 1 + rounds
    elif strategy == 'weight_first':
        # ver4-40, R2: R0 (K seed columns) + `wf_rungs` barrier rungs; the new columns per model
        # are `_wf_rung_counts()` (the engine's own rule, at the largest candidate set), capped
        # at the model's `async_library` budget plus one `library_q` of slack, so the total stays
        # within the async_library figure. Nothing is raced and no fresh gate runs.
        lq = max(1, int(budget.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)
                        or _JOINT_LIBRARY_Q_DEFAULT))
        block_init = max(1, int(budget.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)
                                or _JOINT_BLOCK_INIT_POINTS_DEFAULT))
        per_dim = int(budget.get('library_evals_per_dim', _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT)
                      or _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT)
        max_evals = int(budget.get('library_max_evals', _JOINT_LIBRARY_MAX_EVALS_DEFAULT)
                        or _JOINT_LIBRARY_MAX_EVALS_DEFAULT)
        n_cands = max(2, int(budget.get('wf_max_candidates', _JOINT_WF_MAX_CANDIDATES_DEFAULT)
                             or _JOINT_WF_MAX_CANDIDATES_DEFAULT))
        n_rungs = max(1, int(budget.get('wf_rungs', _JOINT_WF_RUNGS_DEFAULT)
                             or _JOINT_WF_RUNGS_DEFAULT))
        wf_eta = max(2, int(budget.get('wf_eta', _JOINT_WF_ETA_DEFAULT) or _JOINT_WF_ETA_DEFAULT))
        sizes = _wf_active_sizes(n_cands, n_rungs, wf_eta)
        for m in models:
            if dims[m] > 0:
                b_new = _async_budget(dims[m], per_dim, max_evals) - 1
                counts = _wf_rung_counts(b_new, sizes, block_init, b_new + lq)
                search_by_model[m] = int(unit + sum(counts) * unit)
                rounds_by_model[m] = 1 + n_rungs
            else:
                search_by_model[m] = unit
                rounds_by_model[m] = 1
        search_batches = 1 + n_rungs
    elif strategy == 'neural_stack':
        # ver4-48: ONE out-of-fold batch of L_m configurations per model (an upper bound: a
        # duplicate or failed configuration costs nothing). The network and the lambda path are
        # in-process and cost no fit. `ns_fidelity='search'` runs the batch in the 'search'
        # stage, 'full' in the 'oof' stage (accounted below).
        ns_varied, _unused2 = build_joint_param_specs(
            models, hparam_specs, None, freeze_fidelity=True, priority_only=priority_only)
        ns_n_cfg = max(2, int(budget.get('ns_configs', _JOINT_NS_CONFIGS_DEFAULT)
                              or _JOINT_NS_CONFIGS_DEFAULT))
        ns_n_cfg_mcmc = max(1, int(budget.get('ns_configs_mcmc', _JOINT_NS_CONFIGS_MCMC_DEFAULT)
                                   or _JOINT_NS_CONFIGS_MCMC_DEFAULT))
        for m in models:
            ns_cols_by_model[m] = (
                (ns_n_cfg_mcmc if schema_key_of(m) in MCMC_LENGTH_FIELDS else ns_n_cfg)
                if (dims[m] > 0 and len(ns_varied[m]) > 0) else 1)
        ns_fid = budget.get('ns_fidelity', _JOINT_NS_FIDELITY_DEFAULT) or _JOINT_NS_FIDELITY_DEFAULT
        for m in models:
            rounds_by_model[m] = 1
            if ns_fid == 'search':
                search_by_model[m] = ns_cols_by_model[m] * folds
        search_batches = 1 if ns_fid == 'search' else 0
    else:  # async_library (ver4-42: also stage S2 of 'weight_first_cv', which runs it)
        lq = max(1, int(budget.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)
                        or _JOINT_LIBRARY_Q_DEFAULT))
        block_init = max(1, int(budget.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)
                                or _JOINT_BLOCK_INIT_POINTS_DEFAULT))
        per_dim = int(budget.get('library_evals_per_dim', _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT)
                      or _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT)
        max_evals = int(budget.get('library_max_evals', _JOINT_LIBRARY_MAX_EVALS_DEFAULT)
                        or _JOINT_LIBRARY_MAX_EVALS_DEFAULT)
        refine_n = budget.get('library_refine_rounds', _JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT)
        refine_n = _JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT if refine_n is None else max(0, int(refine_n))
        longest = 0
        # ver4-38, R1.b: fold racing. A candidate is first fitted on `k` of the `folds` folds
        # and only the promoted share `p` (nominal 1/3) is completed, so a candidate costs
        # folds * (k/folds + p * (1 - k/folds)) fits instead of `folds`. The untuned seed
        # column is never raced. An upper bound: a small validation set disables racing.
        race_factor = 1.0
        if (budget.get('cv_racing', _JOINT_CV_RACING_DEFAULT) and cv and folds > 2):
            k_race = int(budget.get('racing_folds', _JOINT_RACING_FOLDS_DEFAULT)
                         or _JOINT_RACING_FOLDS_DEFAULT)
            if 1 <= k_race < folds:
                race_factor = (k_race / float(folds)
                               + _JOINT_RACING_PROMOTION_NOMINAL * (1.0 - k_race / float(folds)))
        for m in models:
            if dims[m] > 0:
                plan = _async_round_plan(_async_budget(dims[m], per_dim, max_evals) - 1,
                                         block_init, lq)
                cand_fits = (sum(plan) + refine_n * lq) * unit * race_factor
                search_by_model[m] = int(math.ceil(unit + cand_fits))
                rounds_by_model[m] = 1 + len(plan) + refine_n
                longest = max(longest, len(plan))
            else:
                search_by_model[m] = unit
                rounds_by_model[m] = 1
        search_batches = 1 + longest + refine_n
        if strategy == 'weight_first_cv' and not budget.get('wf_cv_tune', _JOINT_WFCV_TUNE_DEFAULT):
            # ver4-42: weights only ("WFCV-W") - stage S2 does not run, only R0 is fitted
            for m in models:
                search_by_model[m] = unit
                rounds_by_model[m] = 1
            search_batches = 1

    gate_mode = budget.get('gate_mode', _JOINT_GATE_MODE_DEFAULT) or _JOINT_GATE_MODE_DEFAULT
    bbc_gate = ((gate_mode == 'bbc' or strategy in _WF_STRATEGIES) and strategy != 'monolithic'
                and total_dims > 0)
    gate_on = bool(budget.get('fresh_gate', _JOINT_FRESH_GATE_DEFAULT)) and total_dims > 0
    gate_repeats = max(2, int(budget.get('fresh_gate_repeats', _JOINT_FRESH_GATE_REPEATS)
                              or _JOINT_FRESH_GATE_REPEATS))
    gate_per_model = 0
    gate_batches = 0
    if bbc_gate:
        # ver4-36 R1.b: the bootstrap gate runs over the fitted library - zero fits.
        pass
    elif gate_on:
        # candidates (the final point, plus the individually tuned one with a block-search
        # warm start) and the untuned defaults are each scored once, on the same resamples
        n_scored = 1 + (2 if (strategy == 'block_coordinate'
                              and budget.get('warm_start', _JOINT_WARM_START_DEFAULT)
                              == 'individual') else 1)
        gate_per_model = n_scored * (folds if cv else gate_repeats)
        gate_batches = n_scored * (1 if cv else gate_repeats)
    if (not bbc_gate) and (not gate_on) and total_dims > 0:
        # The legacy margin check (fresh_gate off) scores the untuned defaults once more
        # (counted in the 'search' stage) and, with valid_repeats > 1, two more candidates
        # for the 1-SE rule (counted in 'gate'). Under the CV objective valid_repeats is ignored
        # and the defaults' columns are already in the fit cache (the seed point), so nothing
        # is added.
        per_default = 0 if cv else reps
        for m in models:
            search_by_model[m] += per_default
        if reps > 1 and not cv:
            gate_per_model = 2 * reps
            gate_batches = 2 * reps

    # ver4-36 R1.c / R1.d: parity fits of the stub / the exact engine (counted in 'search').
    stub_req = bool(budget.get('search_test_stub', _JOINT_SEARCH_TEST_STUB_DEFAULT))
    engine_req = (budget.get('search_engine', _JOINT_SEARCH_ENGINE_DEFAULT) == 'exact_kernel')
    reuse = (strategy != 'monolithic') and not cv
    stub_models = {m for m in models if stub_req and search_test_stub_safe(m)}
    engine_models = {m for m in models if engine_req and exact_kernel_search_capable(m)}
    parity_mcmc = {m: 0 for m in models}
    parity_exact = {m: 0 for m in models}
    for m in models:
        if m in stub_models or m in engine_models:
            parity_mcmc[m] = 1                                   # the reference fit
            if m in stub_models:
                # the stub fit replaces the untuned column when the cache serves it (it is not
                # needed at all when the exact engine answers the untuned column)
                parity_mcmc[m] += 0 if (reuse and m not in engine_models) else 1
            if m in engine_models:
                parity_exact[m] = 1
        search_by_model[m] += parity_mcmc[m] + parity_exact[m]

    oof_per_model = 0
    oof_batches = 0
    oof_by_model = None                             # ver4-48: per-model figure (neural_stack)
    oof_search = (cv and budget.get('oof_fidelity', _JOINT_OOF_FIDELITY_DEFAULT) == 'search'
                  and strategy != 'monolithic')
    if strategy == 'neural_stack':
        # ver4-48: the library's fits are the out-of-fold stage with ns_fidelity='full' (they are
        # 'search'-stage fits with 'search', counted above). The final out-of-fold refit of the
        # selected parameters (opt-in `library_dump`) is a fit-cache hit at full fidelity and K*V
        # new fits at search fidelity, whose decode differs from the confirmatory one.
        ns_fid = budget.get('ns_fidelity', _JOINT_NS_FIDELITY_DEFAULT) or _JOINT_NS_FIDELITY_DEFAULT
        dump_refit = bool(budget.get('library_dump', _JOINT_LIBRARY_DUMP_DEFAULT)) and ns_fid == 'search'
        oof_by_model = {m: ((ns_cols_by_model[m] * folds) if ns_fid == 'full' else 0)
                        + (folds if dump_refit else 0) for m in models}
        oof_batches = (1 if ns_fid == 'full' else 0) + (1 if dump_refit else 0)
    elif cv and strategy in _WF_STRATEGIES:
        # ver4-40: the final weights are the fixed vector, so the out-of-fold refit only runs
        # for the opt-in library dump (and is all cache hits with oof_fidelity='search').
        dump_refit = bool(budget.get('library_dump', _JOINT_LIBRARY_DUMP_DEFAULT)) and not oof_search
        oof_per_model, oof_batches = (folds, 1) if dump_refit else (0, 0)
        if (strategy == 'weight_first_cv' and budget.get(
                'wf_cv_weight_fidelity', _JOINT_WFCV_WEIGHT_FIDELITY_DEFAULT) == 'full'):
            # ver4-42 stage S0: the full-fidelity out-of-fold refit of the untuned columns
            # (an upper bound: a model with no fidelity field answers it from the fit cache)
            oof_per_model += folds
            oof_batches += 1
    elif cv:
        # ver4-38, R1.d: oof_fidelity='search' reuses the library's own fold fits (cache hits),
        # so the out-of-fold stage adds no fits for a library strategy.
        oof_per_model, oof_batches = (0, 0) if oof_search else (folds, 1)
    elif strategy == 'weight_first':
        # wf_candidate_rows='oof_train' builds the weight candidates from out-of-fold TRAIN
        # predictions at the configured parameters (K * oof_folds extra fits)
        if budget.get('wf_candidate_rows', _JOINT_WF_CANDIDATE_ROWS_DEFAULT) == 'oof_train':
            oof_per_model = int(budget.get('oof_folds', _JOINT_OOF_FOLDS_DEFAULT)
                                or _JOINT_OOF_FOLDS_DEFAULT)
            oof_batches = oof_per_model
    elif budget.get('weight_source', _JOINT_WEIGHT_SOURCE_DEFAULT) == 'oof_train_valid':
        oof_per_model = int(budget.get('oof_folds', _JOINT_OOF_FOLDS_DEFAULT)
                            or _JOINT_OOF_FOLDS_DEFAULT)
        oof_batches = oof_per_model
    search = sum(search_by_model.values())
    gate = gate_per_model * k_models
    confirm = k_models
    # ver4-38, R2.b2: at most `members_max - K` extra confirm fits (an upper bound).
    if (budget.get('library_ensemble', _JOINT_LIBRARY_ENSEMBLE_DEFAULT) and cv
            and strategy not in ('monolithic', 'weight_first', 'weight_first_cv', 'neural_stack')):
        members_max = int(budget.get('library_members_max', _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT)
                          or _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT)
        confirm += max(0, members_max - k_models)
    if (strategy == 'neural_stack' and (budget.get('ns_deploy', _JOINT_NS_DEPLOY_DEFAULT)
                                        or _JOINT_NS_DEPLOY_DEFAULT) == 'mixture'):
        # ver4-48: the mixture deployment fits at most min(library_members_max, C - K) extra
        # members (train-only confirm fits), C = the number of library columns (upper bound).
        members_max = int(budget.get('library_members_max', _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT)
                          or _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT)
        confirm += max(0, min(members_max, sum(ns_cols_by_model.values()) - k_models))
    if oof_by_model is not None:
        oof = sum(oof_by_model.values())
    else:
        oof = oof_per_model * k_models
    total = search + gate + confirm + oof
    critical = search_batches + gate_batches + 1 + oof_batches
    seq = critical if parallel else total
    bglr = 0
    mcmc = 0
    exact = 0
    for m in models:
        is_mcmc = schema_key_of(m) in MCMC_LENGTH_FIELDS
        m_oof = oof_by_model[m] if oof_by_model is not None else oof_per_model
        if is_mcmc:
            searched_len = (strategy != 'neural_stack' and (not freeze) and any(
                _is_fidelity_field(m, sp.index, hparam_specs) for sp in specs_by_model[m]))
            bglr += 1 + m_oof + ((search_by_model[m] + gate_per_model)
                                 if searched_len else 0)
        if m in engine_models:
            m_exact = (search_by_model[m] - parity_mcmc[m]) + gate_per_model
            exact += m_exact
            if is_mcmc:
                mcmc += parity_mcmc[m] + 1 + m_oof
        elif is_mcmc:
            mcmc += search_by_model[m] + gate_per_model + 1 + m_oof
    return {'search': int(search), 'gate': int(gate), 'confirm': int(confirm), 'oof': int(oof),
            'total': int(total), 'seq_batches': int(seq), 'bglr_full_length_fits': int(bglr),
            'mcmc_fits': int(mcmc), 'exact_fits': int(exact), 'critical_path': int(critical),
            'search_by_model': dict(search_by_model), 'rounds_by_model': dict(rounds_by_model)}


def estimate_joint_fit_count(group_models: Sequence[str], hparam_specs: dict,
                             budget: Optional[dict]) -> Dict[str, int]:
    """ver4-34, R1.b: how many model fits will one joint-optimisation task run?

    Why: the six-model default is ~400 fits per task (blueprint SS0.2), which users only
    discovered after submitting a job. This pure function is the single authority for
    that number: the GUI shows it next to the settings and the engine logs it at the
    start of every task (I3 - nobody re-derives the arithmetic). It never fits anything.

    Args: `group_models` the jointly tuned models; `hparam_specs` the HPARAM_SPECS dict
    (tunable dimensions per model, after the `freeze_fidelity_fields` and
    `search_space` filters); `budget` the `HYPERPARAMETERS_JOINT_OPT`-shaped dict - every
    key is read with the same default the engine uses (`n_iter` 30, `init_points` 5, ...).
    `None` is treated as `{}`.

    Returns a dict of ints: `search`, `gate`, `confirm`, `oof` (fits of those stages),
    `total` (their sum - exactly the `total` the engine's `fits:` line reports, cache hits
    excluded), `seq_batches` (sequential fit batches: with `parallel_trials` the number
    of `_run_units` fan-outs, otherwise every fit is its own batch, i.e. `total`) and
    `bglr_full_length_fits` (fits of the MCMC models - those in `MCMC_LENGTH_FIELDS` -
    that run at full MCMC length: confirm + OOF, plus the search and gate fits when the
    run-length fields are searched rather than frozen). ver4-36 adds `mcmc_fits` (fits of
    MCMC models not answered by the exact engine), `exact_fits` (fits answered by it) and
    `critical_path` (longest chain of sequential fit rounds) - see `_estimate_joint_fits()`
    for how the stub, the engine and the BBC gate enter the arithmetic.

    Notes: `total` does not include Step 9's per-model refit (K more fits unless
    `reuse_confirmatory_fit`), because the engine's counter does not see it. For
    `block_coordinate`, `parallel_library` and `async_library` the figures are UPPER bounds
    for the search stage (every cycle / round is assumed to run, every model active, no cache
    hit; the early stops cannot be predicted); the gate is counted for the largest candidate
    set (final point + warm point + defaults). `objective_data='cv_train_valid'` is counted as
    requested - a task that turns out ineligible runs the cheaper `'valid'` objective; a stub
    or engine request that turns out ineligible (BLUE weighting, a failed parity guard) runs
    more MCMC fits than `mcmc_fits` says. Raises nothing for a model without tunable fields
    (it simply contributes no search dimensions)."""
    full = _estimate_joint_fits(group_models, hparam_specs, budget)
    return {k: int(v) for k, v in full.items() if not isinstance(v, dict)}


def tune_joint_hyperparameters_and_weights(
        group_models: Sequence[str], train, valid, test, run_model_fn, RESULT_NAME: str,
        budget_kwargs: dict, hparam_specs: dict, base_params_by_model: dict,
        objective_mode: str = 'ensemble_score',
        bayes_batch: bool = True, bayes_batch_max: int = 4, bayes_liar: str = 'max',
        n_jobs: int = 1, selection_margin: float = 0.02, inner_split_seed_base: int = 0,
        worker_init=None, worker_init_args=(), return_details: bool = False,
        eligibility_ctx: Optional[dict] = None, diagnostics_ctx: Optional[dict] = None,
        ):
    """Search `group_models`'s combined hyperparameter space with ONE joint
    Bayesian optimisation (the ensemble weights are NOT searched - since
    Round 6 they are solved analytically per candidate, see this module's
    docstring), then perform ONE final
    confirmatory evaluation at the winning candidate with every group
    model's cheapening removed (full explainability, full MCMC length/
    iterations - R3 lever 2, mirroring `tune_model_hyperparameters()`'s
    own search/confirm split).

    Parameters
    ----------
    base_params_by_model : this task's own already-resolved (never a
        per-phenotype dict - the caller resolves that first, exactly as
        `genomic_prediction.py`'s existing single-model HP_TUNE dispatch
        already does for a bio-prior model) `HPARAMETERS[m]`-shaped list
        per group model - the search's own starting point / untuned
        fallback for every non-tunable field.
    budget_kwargs : `{'n_iter': int, 'init_points': int, 'seed': int}` -
        the SAME shape a single algorithm's own `HP_TUNE[model]['budget']
        [algorithm]` entry already uses; every key defaults exactly like
        `tune_model_hyperparameters()`'s own Bayesian budget does when
        absent. BUGFIX (weight-collapse report): also reads four optional
        keys governing this search's own weight-collapse protection -
        `'diversity_penalty'` (default `_JOINT_DIVERSITY_PENALTY_DEFAULT`),
        `'diversity_method'` (default `_JOINT_DIVERSITY_METHOD_DEFAULT`),
        `'shrinkage_alpha'` (default `_JOINT_SHRINKAGE_ALPHA_DEFAULT`) and
        `'analytic_seed'` (default `_JOINT_ANALYTIC_SEED_DEFAULT`) - see
        this module's own top-level comment for what each does and why
        they default ON here (unlike their opt-in-only counterparts for
        the three pre-existing weighted-ensemble methods). BUGFIX
        (joint-optimisation-underperformance-fix, Round 5): also reads
        `'untuned_baseline_floor'` (default
        `_JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT`, i.e. OFF) - when
        enabled, pays for one extra full-fidelity confirmatory fit of
        every group model's own untuned baseline hyperparameters
        whenever the search's own winner survives every other safeguard
        above, and falls back to the baseline (optimally weighted, via
        the same `select_weights_with_floor()` mechanism) if it beats
        the search's own result on realised validation MSE - guarantees
        the joint search is never worse than untuned models plus the
        best weighting this codebase already knows how to compute, at
        the cost of one more model-refit pass and of potentially
        discarding a genuine (if search-noise-adjacent) tuned
        improvement. See `_JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT`'s own
        comment for why this one knob defaults OFF unlike every other
        key here. FEATURE (Round 7): also reads `'multi_fidelity'`
        (default `_JOINT_MULTI_FIDELITY_DEFAULT`, i.e. OFF) - only takes
        effect on the BATCH search path (`bayes_batch=True` and `q>1`;
        see `use_batch` below). When enabled, each round's batch of
        candidates is first screened cheaply (every candidate refit on a
        small random subsample of this task's own TRAINING rows only -
        `_screen_subsample_train()` - scored through the exact same
        pipeline against the task's real validation split), and only the
        top `_JOINT_MULTI_FIDELITY_KEEP_FRACTION` of that round's batch is
        then truly re-evaluated at the normal search-phase cost (every
        training row, the SAME evaluation every candidate got before this
        feature existed); the rest keep their cheaper screening score.
        Trades a small amount of per-candidate score fidelity (a
        screening-fidelity score is registered into the search's own
        surrogate for a candidate never truly evaluated at full fidelity
        that round) for spending less wall-clock per round, letting the
        same configured `n_iter`/`init_points` budget cover more of the
        search space, or the same coverage finish sooner - never changes
        how many candidates are evaluated in total (`batch_bayesian_
        maximise()`'s own `1 + init_points + n_iter` budget-accounting
        contract, and every score it returns for that round's batch, are
        both completely unaffected - see this function's own Round 7
        comment beside `objective_batch` below for why this is safe to
        layer in without touching `models/parallel_search.py` at all).
        Has NO effect (an informational NOTE is printed once instead) when
        the batch path isn't in use for this task, e.g. `bayes_batch=
        False`, a single-dimension-or-less search, or `q<=1`. ROUND 10:
        also reads `'weight_cv'` (default OFF: out-of-sample candidate
        scoring + cross-validated final weights - see this module's Round 10
        note), `'fresh_gate'` (default ON: paired winner-vs-defaults check on
        fresh inner resamples) and `'fresh_gate_repeats'` (default 5). Both
        booleans set to False reproduce the pre-Round-10 search.
    n_jobs : ver4-28, R1 - now genuinely used: when `budget_kwargs['parallel_
        trials']` is True and `n_jobs>1`, every fit unit this search issues
        (search-phase candidates, the fresh gate, the confirmatory fit, the
        OOF pool) fans out across up to `n_jobs` loky worker processes via
        `hyperparameter_tuning._evaluate_batch()` (see `_run_units()`).
        `parallel_trials=False` (default) or `n_jobs<=1` keeps every fit
        in-process and serial, byte-identically to before ver4-28.
    worker_init, worker_init_args : ver4-28, R1 - forwarded straight to
        every `_run_units()` call (see that function's own docstring and
        `hyperparameter_tuning._evaluate_batch()`'s own parameter of the
        same name) - restores this run's own resolved compute resources/R
        environment/GPU semaphore inside a freshly spawned worker. `None`/
        `()` (default) reproduces the pre-ver4-28 in-process behaviour.
    return_details : ver4-28, R6 - when True, a 6th return value (`details`,
        a dict - see this function's own body for its exact keys) is
        appended. `False` (default) reproduces the pre-ver4-28 5-tuple
        return shape exactly (I11).

    Also reads (ver4-28; every key OFF/at its legacy-preserving default -
    see the design blueprint's own SS8 Rollback table): `'parallel_trials'`
    (R1), `'reuse_confirmatory_fit'` (R2), `'search_strategy'`/`'block_
    cycles'`/`'block_init_points'`/`'block_n_iter'`/`'block_refine_iter'`/
    `'block_tol'` (R3), `'weight_source'`/`'oof_folds'` (R4).

    Also reads (ver4-32; all six default to the ver4-31 behaviour, see the
    blueprint's SS8 Rollback table): `'freeze_fidelity_fields'` (R1: drop
    run-length fields from the search; they run at the configured value in
    the confirmatory fit), `'search_patience'` (R2: stop a batch search after
    this many evaluations without improvement, 0 = off), `'gate_fit_cache'`
    (R3: the fresh gate reuses identical fits), `'score_weight_rule'` (R4:
    'analytic_penalised' or 'shrunk'), `'warm_start'` (R5: 'none' or
    'individual', block_coordinate strategy only) and `'block_warm_n_iter'`
    (R5, 0 = auto).

    Also reads (ver4-34; every key defaults to the ver4-32 behaviour, see the blueprint's
    SS4/SS8): `'weight_estimator'` ('legacy' | 'bagged_cluster', R2.a) with `'weight_bags'`
    / `'cluster_corr'` / `'score_bags'`; `'objective_data'` ('valid' | 'cv_train_valid',
    R2.b) with `'cv_folds'`; `'search_space'` ('all' | 'priority', R2.c);
    `'freeze_low_weight'` and `'time_budget_s'` (R1.d); `'library_rounds'` / `'library_q'`
    and `search_strategy='parallel_library'` (R1.c).

    `eligibility_ctx` (ver4-34, R2.b; default None): facts about the task that decide
    whether `objective_data='cv_train_valid'` may run - `{'scenario': 'within' | 'between',
    'between_validation_source': str, 'blue_weighting': bool, 'has_bio_prior': bool}`
    (`genomic_prediction.py` Step 8.5 builds it). The CV objective is used only for
    `within`, or `between` with `between_validation_source='split_train'`, with no BLUE
    weighting - or, since ver4-52, with BLUE weighting when the caller also passes
    `'blue_cv_weights': True` (the BLUE context carries the train+valid fitting weights and
    every CV fold unit is fitted through `runner.with_cv_weights()`; without that key the
    runner would pass a full-length weight vector with a row SUBSET, D5) - no
    bio-prior model in the group (they ignore the frames they are handed, D5) and at least
    `10 * cv_folds` train+valid rows. Otherwise the task logs a NOTE and runs the
    `'valid'` objective. A missing context counts as "cannot verify" and also falls back.

    Also reads (ver4-36; every key defaults to the ver4-35 behaviour, see the blueprint's
    SS4/SS8): `'gate_mode'` ('fresh_refit' | 'bbc') with `'bbc_bootstraps'` /
    `'bbc_gate_quantile'` (R1.b); `'search_test_stub'` / `'stub_parity_min'` (R1.c);
    `'search_engine'` ('bglr' | 'exact_kernel') / `'engine_parity_min'` (R1.d);
    `search_strategy='async_library'` with `'library_evals_per_dim'` / `'library_max_evals'` /
    `'library_refine_rounds'` (R1.e); `'noise_aware_surrogate'` (R2.a); `'selection_rule'`
    ('greedy' | 'bbc', R2.b); `weight_estimator='ensemble_selection'` with `'es_steps'` /
    `'es_model_fraction'` (R2.c); `'library_dump'` (R2.d). The stub and the exact engine are
    used only when `eligibility_ctx` is given and says `blue_weighting` is False, and only
    for models that pass `_parity_guards()`; the confirmatory fit is always a real fit.

    Also reads (ver4-38; every key defaults to the ver4-37 behaviour, see the blueprint):
    `'cv_racing'` / `'racing_folds'` / `'racing_margin'` / `'racing_penalty'` (R1.b, async
    library + CV objective only); `'library_stop_rule'` ('off' | 'value', R1.c, async library
    only); `'oof_fidelity'` ('full' | 'search', R1.d, CV objective only; the library dump then
    carries `oof__<k>` and `meta.dump_version` 2); `'dead_column_guard'` (R2.b1) and
    `'library_ensemble'` / `'library_members_max'` / `'library_members_per_model'` (R2.b2;
    the extra members' columns are returned in `details['library_members']`). A key whose
    requirement is not met logs a NOTE and runs as if off.

    Also reads (ver4-48; every key defaults to the ver4-47 behaviour and acts only under
    `search_strategy='neural_stack'`): `'ns_configs'` / `'ns_configs_mcmc'` (configurations per
    model; per MCMC model), `'ns_lambdas'` (the regularisation path; lambda = infinity, the untuned
    naive ensemble, is added unless `safety_nets=False`), `'ns_lambda_rule'` ('min' | 'one_se';
    forced to 'min' without the nets), `'ns_repeats'` (CV repeats of the lambda choice),
    `'ns_deploy'` ('mixture' = the heavy library columns are the members, 'argmax' = one
    configuration per model) and `'ns_fidelity'` (fidelity of the library's out-of-fold fits).
    ver4-50 adds `'ns_weight_shrink'` (a GUI field since this version; default 0.5),
    `'ns_objective'` ('ensemble_score' = ver4-49 | 'pearson': the network is trained on ranking
    only) and `'ns_swap_rule'` ('gate' = ver4-49 | 'paired_se': a hyperparameter switch needs a
    paired CV gain above its standard error | 'paired_refit', ver4-51: the gain is measured against
    the network refitted with that gate clamped to entry 0). ver4-51 adds `'ns_outer_gate'` ('off'
    = ver4-50 | 'nested': an outer cross-validation of the whole procedure must show a gain over
    equal weights, else the equal-weight ensemble is deployed), `'ns_outer_folds'` (5) and
    `'ns_outer_repeats'` (2). `details['ns']` carries the fitted network (see
    `_neural_stack_search`).

    `diagnostics_ctx` (ver4-36, R2.d; default None): `{'task_tag': str, 'result_dir': str}` -
    where `library_dump` writes `<result_dir>/Joint_library/<task_tag>.npz` (the folder name
    comes from `checkpoint_utils.DIAGNOSTIC_DIR_NAMES`). Without it `library_dump` logs a NOTE
    and writes nothing.

    Also reads (ver4-44; every key defaults to the ver4-43 behaviour, byte-identical when
    absent - I11): `'safety_nets'` (default True) - False switches off every safety net that
    replaces the optimiser's answer with an untuned / equal-weight result (N1-N10: the silent
    strategy fallbacks, the fresh and BBC gates, the untuned and equal-weight floors, weight_cv,
    the diversity penalty / shrinkage / 'shrunk' rule, the shrinkage-path alpha, the
    weight-first-cv untuned-vs-tuned check, the 'uniform' candidate and untuned finalists), so
    the joint row is the search's own result; crash / degenerate-data fallbacks (class C) stay
    and are returned in `details['fallback_events']`. With nets off, an unusable library
    strategy or a failed search raises instead of falling back. `'wf_cv_final'` ('refit' =
    ver4-43, or 'bagged' = the mean of the cross-fitted fold weight vectors; only read under
    `search_strategy='weight_first_cv'`). An unrecognised `search_strategy` raises ValueError
    (`validate_search_strategy`, which `GP()` also calls before task 1) instead of running
    'monolithic'.

    Returns
    -------
    (final_params_by_model, weights, best_valid_score, elapsed,
    tuned_values_by_model) - `final_params_by_model`, `weights` and
    `tuned_values_by_model` are all dicts keyed by model name (see
    `_score_candidate()`'s own docstring for why a dict, not the design
    blueprint's own illustrative positional array). `weights` is the
    validation-MSE-floor-graded result (BUGFIX, weight-collapse report),
    not necessarily the raw search winner - see the floor step below.
    `tuned_values_by_model[m]` is `{field_label: final_value}` (BUGFIX,
    hyperparameter.csv recording report) - the caller writes this into
    that model's own `hyperparameter.csv` row exactly like a single-model
    HP_TUNE row already does with its own `tuned_values`. A 6th element,
    `details` (ver4-28, R6 - `{'strategy', 'gate_decision', 'n_fits',
    'confirm_results', 'weight_source', 'secs'}`; ver4-32 adds `'secs'` =
    wall-clock seconds per stage and the gate value 'warm_accepted'; ver4-44 adds
    `'safety_nets'` and `'fallback_events'`, and the gate value 'searched_unprotected'), is
    appended only when `return_details=True`.
    """
    t0 = time.time()
    base_params_by_model = {m: list(base_params_by_model[m]) for m in group_models}
    # ver4-32, R1: 'freeze_fidelity_fields' (default False = ver4-31). Read
    # before the search space is built because it changes the space.
    _freeze_fidelity = bool(budget_kwargs.get('freeze_fidelity_fields',
                                              _JOINT_FREEZE_FIDELITY_FIELDS_DEFAULT))
    # ver4-34, R2.c: 'search_space' ('all' = ver4-32, 'priority' = only fields marked
    # `priority` in HPARAM_SPECS). Also read here because it changes the space.
    _search_space = (budget_kwargs.get('search_space', _JOINT_SEARCH_SPACE_DEFAULT)
                     or _JOINT_SEARCH_SPACE_DEFAULT)
    if _search_space not in ('all', 'priority'):
        print(f"[GP] NOTE: Joint Bayesian optimisation search_space={_search_space!r} is not "
              f"recognised (expected 'all' or 'priority') - falling back to 'all' for this task.")
        _search_space = 'all'
    _priority_only = (_search_space == 'priority')
    specs_by_model, n_weight_dims = build_joint_param_specs(
        group_models, hparam_specs, base_params_by_model, freeze_fidelity=_freeze_fidelity,
        priority_only=_priority_only)
    total_dims = sum(len(s) for s in specs_by_model.values()) + n_weight_dims
    # Fields taken out of the search (run-length fields and/or non-priority fields):
    # {model: [(label, idx), ...]}. Their final values are added back to
    # `tuned_values_by_model` after the confirmatory fit so the hyperparameter.csv
    # columns do not change.
    _full_specs_by_model = specs_by_model
    _frozen_fields = {}
    _dims_before_freeze = total_dims
    if _freeze_fidelity or _priority_only:
        _full_specs_by_model, _ = build_joint_param_specs(
            group_models, hparam_specs, base_params_by_model, freeze_fidelity=False)
        _dims_before_freeze = sum(len(sp) for sp in _full_specs_by_model.values())
        for m in group_models:
            _kept = {sp.index for sp in specs_by_model[m]}
            _gone = [(sp.name, sp.index) for sp in _full_specs_by_model[m] if sp.index not in _kept]
            if _gone:
                _frozen_fields[m] = _gone
    if _freeze_fidelity and not _priority_only:
        print(f"[GP] Joint Bayesian optimisation: freeze_fidelity_fields=True removed "
              f"{_dims_before_freeze - total_dims} run-length field(s) from the search space "
              f"(dimensions {_dims_before_freeze} -> {total_dims}): "
              f"{ {m: [lab for lab, _i in g] for m, g in _frozen_fields.items()} }. They run "
              f"cheapened during the search and at the configured value in the confirmatory fit.")
    elif _freeze_fidelity:
        _fz_specs, _ = build_joint_param_specs(
            group_models, hparam_specs, base_params_by_model, freeze_fidelity=True)
        _fz_dims = sum(len(sp) for sp in _fz_specs.values())
        _fz_gone = {m: [sp.name for sp in _full_specs_by_model[m]
                        if sp.index not in {x.index for x in _fz_specs[m]}]
                    for m in group_models}
        print(f"[GP] Joint Bayesian optimisation: freeze_fidelity_fields=True removed "
              f"{_dims_before_freeze - _fz_dims} run-length field(s) from the search space "
              f"(dimensions {_dims_before_freeze} -> {_fz_dims}): "
              f"{ {m: g for m, g in _fz_gone.items() if g} }. They run cheapened during the "
              f"search and at the configured value in the confirmatory fit.")
    if _priority_only:
        print(f"[GP] Joint Bayesian optimisation: search_space='priority' reduced the search "
              f"space from {_dims_before_freeze} to {total_dims} dimension(s) "
              f"(dimensions {_dims_before_freeze} -> {total_dims}); searched fields per model: "
              f"{ {m: [sp.name for sp in specs_by_model[m]] for m in group_models} }. The other "
              f"fields run at their configured value (cheapened during the search when they are "
              f"run-length fields, full length in the confirmatory fit); hyperparameter.csv keeps "
              f"every column.")

    if all(len(s) == 0 for s in specs_by_model.values()):
        if _freeze_fidelity and _dims_before_freeze > 0:
            print(f"[GP] NOTE: Joint Bayesian optimisation - freeze_fidelity_fields=True left no "
                  f"tunable hyperparameter in {list(group_models)!r} (every tunable field is a "
                  f"run-length field); this joint search only optimises the ensemble weights, "
                  f"not any model's own hyperparameters.")
        else:
            print(f"[GP] NOTE: Joint Bayesian optimisation - none of {list(group_models)!r} has "
                  f"any tunable hyperparameter field in HPARAM_SPECS; this joint search only "
                  f"optimises the ensemble weights, not any model's own hyperparameters.")

    has_categorical = any(s.kind == 'categorical' for specs in specs_by_model.values() for s in specs)

    n_iter = int(budget_kwargs.get('n_iter', 30) or 30)
    init_points = int(budget_kwargs.get('init_points', 5) or 5)

    # ver4-27, R3: this NOTE now also states a concrete RECOMMENDED
    # iteration count and compares it against what's actually configured,
    # rather than only naming the dimensionality threshold - the
    # difference matters in practice: the same n_iter/init_points DEFAULTS
    # (50/5, main_app.py's own JOINT_HYPERPARAM_OPT_SPECS) are reasonable
    # for jointly tuning two modest models but are typically a severe
    # under-count once four to six models (each contributing 7-26 of
    # their own tunable fields - see hparam_specs.py) are all joint-tuned
    # together, since every evaluation here ALSO costs a refit of every
    # included model, discouraging simply cranking n_iter up blind. The
    # `8x total_dims` figure is a rough, commonly-used rule of thumb for
    # Gaussian-process Bayesian optimisation coverage (like
    # JOINT_SEARCH_DIM_WARN_THRESHOLD itself, a provisional constant, not
    # a guarantee) - informational only, never adjusts n_iter/init_points
    # on its own (silently spending more of a user's compute budget than
    # configured is exactly what this codebase's own I11 discipline
    # elsewhere refuses to do).
    if total_dims > JOINT_SEARCH_DIM_WARN_THRESHOLD:
        _dominant = sorted(specs_by_model.items(), key=lambda kv: -len(kv[1]))[:3]
        _dominant_txt = ', '.join(f"{m}={len(s)}" for m, s in _dominant if len(s) > 0)
        _recommended_n_iter = max(30, total_dims * 8)
        _configured = init_points + n_iter
        print(f"[GP] NOTE: Joint Bayesian optimisation search space is {total_dims}-dimensional "
              f"(> {JOINT_SEARCH_DIM_WARN_THRESHOLD}) across {list(group_models)!r} "
              f"(dominant contributor(s): {_dominant_txt}). This task's own configured budget is "
              f"{_configured} evaluations (init_points={init_points} + n_iter={n_iter}); a rough "
              f"rule of thumb for a space this size is at least ~{_recommended_n_iter} iterations "
              f"({'well above' if n_iter < _recommended_n_iter else 'already at or above'} what's "
              f"configured here). Each evaluation refits every one of the "
              f"{len(group_models)} included model(s), so raising n_iter this much multiplies "
              f"wall-clock cost accordingly - consider jointly tuning fewer, more deliberately "
              f"chosen models via HYPERPARAMETERS_JOINT_OPT['group_models'] instead if that cost "
              f"isn't affordable. This is informational only and never blocks the run or changes "
              f"n_iter/init_points on its own.")
    seed = int(budget_kwargs.get('seed', 1) or 1)

    # ver4-44, R1: `safety_nets` (default True = every protection of ver4-43, byte-identical).
    # False = "pure joint evaluation": from here on the task works on a COPY of `budget_kwargs`
    # in which every key that feeds a safety net is at its off value (see
    # `_pure_budget_overrides()`), so the joint row is what the optimiser itself produced.
    # Class C events (a crash or degenerate data that a fallback has to absorb) are NOT switched
    # off; each one is appended to `_fallback_events`, returned in `details['fallback_events']`
    # and written to hyperparameter.csv, so the evaluation can exclude those tasks.
    _safety_nets_on, _nets_disabled = _resolve_safety_nets(budget_kwargs)
    _fallback_events: List[str] = []
    if not _safety_nets_on:
        budget_kwargs = _pure_budget_overrides(budget_kwargs)
        print(f"[GP] Joint Bayesian optimisation safety_nets=False (pure joint evaluation): "
              f"disabled " + '; '.join(_nets_disabled) + ". Crash / degenerate-data fallbacks "
              f"stay and are recorded per task (fallback_events).")
    else:
        print("[GP] Joint Bayesian optimisation safety_nets=True: every safety net "
              "stays on.")

    # BUGFIX (weight-collapse report) - see this module's own top-level
    # comment and tune_joint_hyperparameters_and_weights()'s own docstring
    # above for what each setting does and why it defaults ON here.
    _diversity_penalty = budget_kwargs.get('diversity_penalty', _JOINT_DIVERSITY_PENALTY_DEFAULT)
    _diversity_penalty = float(_diversity_penalty) if _diversity_penalty is not None else 0.0
    _diversity_method = budget_kwargs.get('diversity_method', _JOINT_DIVERSITY_METHOD_DEFAULT) \
        or _JOINT_DIVERSITY_METHOD_DEFAULT
    _shrinkage_alpha = budget_kwargs.get('shrinkage_alpha', _JOINT_SHRINKAGE_ALPHA_DEFAULT)
    _shrinkage_alpha = float(_shrinkage_alpha) if _shrinkage_alpha is not None else _JOINT_SHRINKAGE_ALPHA_DEFAULT
    _analytic_seed = bool(budget_kwargs.get('analytic_seed', _JOINT_ANALYTIC_SEED_DEFAULT))
    if _diversity_penalty:
        print(f"[GP] Joint Bayesian optimisation: weight-collapse protection active - "
              f"diversity_penalty={_diversity_penalty} (method={_diversity_method!r}), "
              f"shrinkage_alpha={_shrinkage_alpha}, analytic_seed={_analytic_seed}.")

    # BUGFIX (joint-optimisation-underperformance-fix, Round 5): optional
    # 'untuned_baseline_floor' key inside budget_kwargs - OFF by default
    # (see _JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT's own comment above for
    # why this one, unlike every other knob in this function, defaults
    # off rather than on). Gates the untuned-baseline-vs-optimally-
    # weighted-ensemble comparison below entirely - when off, this
    # function's behaviour is byte-identical to before Round 5 (I11).
    _untuned_baseline_floor = bool(
        budget_kwargs.get('untuned_baseline_floor', _JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT)
    )
    if _untuned_baseline_floor:
        print(f"[GP] Joint Bayesian optimisation: untuned-baseline-vs-optimally-weighted-ensemble "
              f"floor is ACTIVE (untuned_baseline_floor=True) - the search's own winning candidate "
              f"will be compared against every group model's own configured hyperparameters, "
              f"optimally weighted, and replaced by it if that scores better on realised "
              f"validation MSE.")

    # ver4-27, R2: optional 'valid_repeats' key inside budget_kwargs (the
    # SAME shape a single HP_TUNE algorithm's own HP_TUNE_VALID_REPEATS
    # already uses - see _score_candidate()'s own comment for the full
    # rationale). Defaults to 1 (I11): every existing caller/config that
    # never sets this key gets this function's exact pre-ver4-27 behaviour,
    # a single evaluation per candidate on the task's own real validation
    # split, bit for bit.
    _valid_repeats = int(budget_kwargs.get('valid_repeats', 1) or 1)
    if _valid_repeats > 1:
        print(f"[GP] Joint Bayesian optimisation: scoring each search-phase candidate as the "
              f"mean over {_valid_repeats} inner training-split resamples (valid_repeats="
              f"{_valid_repeats}) - the confirmatory evaluation of this search's own winning "
              f"candidate always uses a single fit on the task's real validation split, "
              f"unaffected by this setting.")

    # FEATURE (Round 7): optional 'multi_fidelity' key inside budget_kwargs
    # - OFF by default (see _JOINT_MULTI_FIDELITY_DEFAULT's own comment
    # above for why, mirroring _JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT's own
    # rationale: a genuine cost/fidelity trade-off, not an unconditional
    # correctness fix). Only wired into the BATCH search path's own
    # `objective_batch()` closure below - whether that path is actually in
    # use for this task is only known once `use_batch` is computed further
    # down, so the "has no effect" NOTE for a batch-less task is printed
    # there instead, right where `use_batch` is decided.
    _multi_fidelity = bool(budget_kwargs.get('multi_fidelity', _JOINT_MULTI_FIDELITY_DEFAULT))

    # ROUND 10: 'weight_cv' (default OFF - opt-in; the constant
    # `_JOINT_WEIGHT_CV_DEFAULT` is False) / 'fresh_gate' (default ON -
    # correctness fix, see the Round 10 note in this module's docstring). With
    # 'weight_cv' the search scores every candidate at cross-fitted
    # (out-of-fold) weights shrunk by `shrinkage_alpha` instead of at the
    # in-sample MSE-optimal weights, and the FINAL weights are chosen by
    # cross-validating the shrinkage level (equal weighting <-> analytic)
    # over the validation rows. `_weight_cv_alpha` is the `oof_alpha` handed
    # to `_score_candidate()`; None reproduces the pre-Round-10 in-sample
    # scoring exactly.
    _weight_cv = bool(budget_kwargs.get('weight_cv', _JOINT_WEIGHT_CV_DEFAULT))
    _weight_cv_alpha = float(np.clip(_shrinkage_alpha, 0.0, 1.0)) if _weight_cv else None
    _fresh_gate = bool(budget_kwargs.get('fresh_gate', _JOINT_FRESH_GATE_DEFAULT))
    _fresh_gate_repeats = max(2, int(budget_kwargs.get('fresh_gate_repeats',
                                                        _JOINT_FRESH_GATE_REPEATS) or _JOINT_FRESH_GATE_REPEATS))

    # ver4-28 (design blueprint EasiGP_ver4-28_Design_Blueprint.md) - every
    # key below is read with `cfg.get('KEY', <legacy default>)` (I11) and
    # logged, resolved, unconditionally below, regardless of whether it
    # differs from its default.
    _parallel_trials = bool(budget_kwargs.get('parallel_trials', _JOINT_PARALLEL_TRIALS_DEFAULT))
    _effective_n_jobs = int(n_jobs) if (_parallel_trials and n_jobs and int(n_jobs) > 1) else 1
    _reuse_confirmatory_fit = bool(
        budget_kwargs.get('reuse_confirmatory_fit', _JOINT_REUSE_CONFIRMATORY_FIT_DEFAULT))
    # ver4-44, R0: an unknown strategy is an error, never a silent 'monolithic' (blueprint S5).
    _search_strategy = validate_search_strategy(budget_kwargs)
    _block_cycles = int(budget_kwargs.get('block_cycles', _JOINT_BLOCK_CYCLES_DEFAULT)
                         or _JOINT_BLOCK_CYCLES_DEFAULT)
    _block_init_points = int(budget_kwargs.get('block_init_points', _JOINT_BLOCK_INIT_POINTS_DEFAULT)
                              or _JOINT_BLOCK_INIT_POINTS_DEFAULT)
    _block_n_iter = int(budget_kwargs.get('block_n_iter', _JOINT_BLOCK_N_ITER_DEFAULT) or 0)
    _block_refine_iter = int(budget_kwargs.get('block_refine_iter', _JOINT_BLOCK_REFINE_ITER_DEFAULT)
                              or _JOINT_BLOCK_REFINE_ITER_DEFAULT)
    _block_tol = float(budget_kwargs.get('block_tol', _JOINT_BLOCK_TOL_DEFAULT) or _JOINT_BLOCK_TOL_DEFAULT)
    _weight_source = (budget_kwargs.get('weight_source', _JOINT_WEIGHT_SOURCE_DEFAULT)
                       or _JOINT_WEIGHT_SOURCE_DEFAULT)
    if _weight_source not in ('valid', 'oof_train_valid'):
        print(f"[GP] NOTE: Joint Bayesian optimisation weight_source={_weight_source!r} is not "
              f"recognised (expected 'valid' or 'oof_train_valid') - falling back to 'valid' for "
              f"this task.")
        _weight_source = 'valid'
    _oof_folds = int(budget_kwargs.get('oof_folds', _JOINT_OOF_FOLDS_DEFAULT) or _JOINT_OOF_FOLDS_DEFAULT)

    print(f"[GP] Joint Bayesian optimisation settings: search_strategy={_search_strategy!r} "
          f"(block_cycles={_block_cycles}, block_init_points={_block_init_points}, "
          f"block_n_iter={_block_n_iter or 'auto'}, block_refine_iter={_block_refine_iter}, "
          f"block_tol={_block_tol}), parallel_trials={_parallel_trials} "
          f"(effective_n_jobs={_effective_n_jobs}), reuse_confirmatory_fit={_reuse_confirmatory_fit}, "
          f"weight_source={_weight_source!r} (oof_folds={_oof_folds}).")

    # ver4-32 keys - read with `cfg.get('KEY', <legacy default>)` (I11) and
    # logged, resolved, unconditionally, whether or not they differ from the
    # default. R1's 'freeze_fidelity_fields' was read at the top of this function.
    _warm_start = (budget_kwargs.get('warm_start', _JOINT_WARM_START_DEFAULT)
                   or _JOINT_WARM_START_DEFAULT)
    if _warm_start not in ('none', 'individual'):
        print(f"[GP] NOTE: Joint Bayesian optimisation warm_start={_warm_start!r} is not "
              f"recognised (expected 'none' or 'individual') - falling back to 'none' for this "
              f"task.")
        _warm_start = 'none'
    _block_warm_n_iter = int(
        budget_kwargs.get('block_warm_n_iter', _JOINT_BLOCK_WARM_N_ITER_DEFAULT) or 0)
    _score_weight_rule = (budget_kwargs.get('score_weight_rule', _JOINT_SCORE_WEIGHT_RULE_DEFAULT)
                          or _JOINT_SCORE_WEIGHT_RULE_DEFAULT)
    if _score_weight_rule not in ('analytic_penalised', 'shrunk'):
        print(f"[GP] NOTE: Joint Bayesian optimisation score_weight_rule={_score_weight_rule!r} "
              f"is not recognised (expected 'analytic_penalised' or 'shrunk') - falling back "
              f"to 'analytic_penalised' for this task.")
        _score_weight_rule = 'analytic_penalised'
    _search_patience = int(
        budget_kwargs.get('search_patience', _JOINT_SEARCH_PATIENCE_DEFAULT) or 0)
    _gate_fit_cache = bool(budget_kwargs.get('gate_fit_cache', _JOINT_GATE_FIT_CACHE_DEFAULT))
    print(f"[GP] Joint Bayesian optimisation settings: "
          f"freeze_fidelity_fields={_freeze_fidelity}, "
          f"warm_start={_warm_start!r} (block_warm_n_iter={_block_warm_n_iter or 'auto'}), "
          f"score_weight_rule={_score_weight_rule!r} (shrinkage_alpha={_shrinkage_alpha}), "
          f"search_patience={_search_patience or 'off'}, gate_fit_cache={_gate_fit_cache}.")
    if _warm_start == 'individual' and _search_strategy != 'block_coordinate':
        print(f"[GP] NOTE: Joint Bayesian optimisation warm_start='individual' only applies to "
              f"search_strategy='block_coordinate' (this task uses {_search_strategy!r}) - it is "
              f"ignored for this task.")
    if _score_weight_rule == 'shrunk' and _weight_cv:
        print("[GP] NOTE: Joint Bayesian optimisation score_weight_rule='shrunk' with "
              "weight_cv=True: candidates are scored on cross-fitted weights (weight_cv), "
              "which already include the shrinkage; 'shrunk' then only removes the "
              "diversity penalty.")

    # ver4-34 keys - read with `cfg.get('KEY', <legacy default>)` (I11); the resolved
    # values are logged, unconditionally, in the `ver4-34 settings` line below.
    # R2.a: the weight estimator.
    _weight_estimator = (budget_kwargs.get('weight_estimator', _JOINT_WEIGHT_ESTIMATOR_DEFAULT)
                         or _JOINT_WEIGHT_ESTIMATOR_DEFAULT)
    if _weight_estimator != 'legacy' and not is_robust_estimator(_weight_estimator):
        print(f"[GP] NOTE: Joint Bayesian optimisation weight_estimator={_weight_estimator!r} is "
              f"not recognised (expected 'legacy' or one of "
              f"{sorted(ROBUST_WEIGHT_ESTIMATORS)!r}) - falling back to 'legacy' for this task.")
        _weight_estimator = 'legacy'
    _weight_bags = max(2, int(budget_kwargs.get('weight_bags', _JOINT_WEIGHT_BAGS_DEFAULT)
                               or _JOINT_WEIGHT_BAGS_DEFAULT))
    _cluster_corr = float(budget_kwargs.get('cluster_corr', _JOINT_CLUSTER_CORR_DEFAULT)
                          or _JOINT_CLUSTER_CORR_DEFAULT)
    if not (0.0 < _cluster_corr <= 1.0):
        print(f"[GP] NOTE: Joint Bayesian optimisation cluster_corr={_cluster_corr} is outside "
              f"(0, 1] - using {_JOINT_CLUSTER_CORR_DEFAULT}.")
        _cluster_corr = _JOINT_CLUSTER_CORR_DEFAULT
    _score_bags = max(1, int(budget_kwargs.get('score_bags', _JOINT_SCORE_BAGS_DEFAULT)
                              or _JOINT_SCORE_BAGS_DEFAULT))
    _bagged_final = (is_robust_estimator(_weight_estimator) and objective_mode != 'dpt_ratio')
    _es_steps = max(0, int(budget_kwargs.get('es_steps', _JOINT_ES_STEPS_DEFAULT) or 0))
    _es_model_fraction = float(budget_kwargs.get('es_model_fraction',
                                                 _JOINT_ES_MODEL_FRACTION_DEFAULT)
                               or _JOINT_ES_MODEL_FRACTION_DEFAULT)
    if not (0.0 < _es_model_fraction <= 1.0):
        print(f"[GP] NOTE: Joint Bayesian optimisation es_model_fraction={_es_model_fraction} is "
              f"outside (0, 1] - using {_JOINT_ES_MODEL_FRACTION_DEFAULT}.")
        _es_model_fraction = _JOINT_ES_MODEL_FRACTION_DEFAULT
    _estimator_settings = {'score_bags': _score_bags, 'cluster_corr': _cluster_corr,
                           'n_folds': _JOINT_CV_FOLDS_DEFAULT}
    if _weight_estimator == 'ensemble_selection':
        _estimator_settings.update({'es_steps': _es_steps, 'es_model_fraction': _es_model_fraction})
    if is_robust_estimator(_weight_estimator):
        if objective_mode == 'dpt_ratio':
            print(f"[GP] NOTE: Joint Bayesian optimisation weight_estimator={_weight_estimator!r} "
                  f"applies to the 'ensemble_score' objective only; this task uses 'dpt_ratio', "
                  f"so the legacy weights are used.")
        else:
            print(f"[GP] NOTE: Joint Bayesian optimisation weight_estimator={_weight_estimator!r} "
                  f"replaces the weight-collapse protection: diversity_penalty, shrinkage_alpha, "
                  f"analytic_seed, weight_cv and score_weight_rule are ignored, and there is no "
                  f"equal-weight floor - the weights are whatever the (cross-fitted) data "
                  f"support.")
            _weight_cv = False
            _weight_cv_alpha = None
    # R2.b: the objective's data, and whether this task may use the CV objective.
    _objective_data = (budget_kwargs.get('objective_data', _JOINT_OBJECTIVE_DATA_DEFAULT)
                       or _JOINT_OBJECTIVE_DATA_DEFAULT)
    if _objective_data not in ('valid', 'cv_train_valid'):
        print(f"[GP] NOTE: Joint Bayesian optimisation objective_data={_objective_data!r} is not "
              f"recognised (expected 'valid' or 'cv_train_valid') - falling back to 'valid' for "
              f"this task.")
        _objective_data = 'valid'
    if (_search_strategy == 'weight_first_cv' and _objective_data != 'cv_train_valid'
            and len(group_models) > 1):
        # ver4-42: the weights are estimated on the out-of-fold rows over train+valid, so the
        # strategy needs the CV objective; the eligibility block below still applies its own
        # fallback to 'valid' (and then says so).
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first_cv' uses "
              f"objective_data='cv_train_valid' (the weights are cross-fitted on the out-of-fold "
              f"rows over train+valid); the configured objective_data={_objective_data!r} is "
              f"overridden for this task.")
        _objective_data = 'cv_train_valid'
    if (_search_strategy == 'neural_stack' and _objective_data != 'cv_train_valid'
            and len(group_models) > 1):
        # ver4-48: the network is fitted on out-of-fold columns over train+valid (every row is
        # held out once), so the strategy needs the CV objective; the eligibility block below
        # still applies its own fallback to 'valid' (and then says so).
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='neural_stack' uses "
              f"objective_data='cv_train_valid' (the network is fitted on the out-of-fold "
              f"columns over train+valid); the configured objective_data={_objective_data!r} is "
              f"overridden for this task.")
        _objective_data = 'cv_train_valid'
    _cv_folds = max(2, int(budget_kwargs.get('cv_folds', _JOINT_CV_FOLDS_DEFAULT)
                            or _JOINT_CV_FOLDS_DEFAULT))
    if _objective_data == 'cv_train_valid':
        _n_tv = int(len(train)) + (0 if valid is None else int(len(valid)))
        _why = []
        if valid is None or len(valid) == 0:
            _why.append("this task has no validation rows")
        if eligibility_ctx is None:
            _why.append("the caller supplied no eligibility context, so the scenario cannot be "
                        "verified")
        else:
            _scenario = str(eligibility_ctx.get('scenario', '')).strip().lower()
            _source = str(eligibility_ctx.get('between_validation_source', '')).strip().lower()
            if _scenario == 'between':
                if _source != 'split_train':
                    _why.append(f"SCENARIO='between' with BETWEEN_VALIDATION_SOURCE={_source!r} "
                                f"(only 'split_train' draws its validation rows from the training "
                                f"population)")
            elif _scenario != 'within':
                _why.append(f"SCENARIO={_scenario!r} (only 'within', or 'between' with "
                            f"'split_train', is supported)")
            if eligibility_ctx.get('blue_weighting'):
                if eligibility_ctx.get('blue_cv_weights'):
                    # ver4-52: the context carries the train+valid fitting weights, and every CV
                    # fold unit is fitted through `runner.with_cv_weights()` (`_FitUnit.cv_weights`),
                    # so the fold fits are weighted by their own rows' BLUE reliabilities. The
                    # out-of-fold columns are still scored unweighted, exactly as the validation
                    # objective is.
                    print("[GP] NOTE: Joint Bayesian optimisation objective_data='cv_train_valid' "
                          "stays available under BLUE weighting: each fold model is fitted with "
                          "the BLUE fitting weights of its own training rows (train+valid), and "
                          "the out-of-fold columns are scored unweighted.")
                else:
                    _why.append("BLUE weighting is active and the caller supplied no train+valid "
                                "fitting weights for the folds (the runner would pass a "
                                "full-length weight vector with a row subset)")
            if eligibility_ctx.get('has_bio_prior'):
                _why.append("a bio-prior model is in the group (it ignores the frames it is "
                            "handed)")
        if _n_tv < 10 * _cv_folds:
            _why.append(f"only {_n_tv} train+valid rows (need >= {10 * _cv_folds} for "
                        f"{_cv_folds} folds)")
        if _why:
            print(f"[GP] NOTE: Joint Bayesian optimisation objective_data='cv_train_valid' is not "
                  f"available for this task - " + '; '.join(_why) + " - falling back to "
                  f"objective_data='valid' for this task.")
            _objective_data = 'valid'
            if _search_strategy == 'weight_first_cv' and len(group_models) > 1:
                # ver4-44, R1 (N13): the weights of this task come from the ~validation rows only
                _fallback_events.append('N13:valid_rows')
            if _search_strategy == 'neural_stack' and len(group_models) > 1:
                # ver4-48 (N20): the neural stack cannot run without the CV objective, so this
                # task runs the monolithic search (or, with safety_nets=False, fails)
                _fallback_events.append('N20:valid_rows')
    _cv_active = (_objective_data == 'cv_train_valid')
    if _cv_active:
        print(f"[GP] Joint Bayesian optimisation objective_data='cv_train_valid': candidates are "
              f"scored on out-of-fold columns over train+valid ({_cv_folds} folds, the test split "
              f"is never read); the FINAL models are still fitted on the train split only.")
        if _valid_repeats > 1:
            print(f"[GP] NOTE: Joint Bayesian optimisation valid_repeats={_valid_repeats} is "
                  f"ignored with objective_data='cv_train_valid' (the folds replace the inner "
                  f"resamples).")
            _valid_repeats = 1
        if _weight_source != 'valid':
            print(f"[GP] NOTE: Joint Bayesian optimisation weight_source={_weight_source!r} is "
                  f"ignored with objective_data='cv_train_valid' (the final weights already come "
                  f"from the out-of-fold matrix over train+valid).")
        if _multi_fidelity:
            print("[GP] NOTE: Joint Bayesian optimisation 'multi_fidelity' is ignored with "
                  "objective_data='cv_train_valid' (the fit cache is keyed by fold, not by "
                  "training subsample).")
            _multi_fidelity = False
    _cv_cache = _FitCache() if _cv_active else None
    # R1.d: freezing and the wall-clock budget.
    _freeze_low_weight = max(0.0, float(budget_kwargs.get('freeze_low_weight',
                                                          _JOINT_FREEZE_LOW_WEIGHT_DEFAULT) or 0.0))
    _time_budget_s = max(0, int(budget_kwargs.get('time_budget_s', _JOINT_TIME_BUDGET_DEFAULT)
                                 or 0))
    if _freeze_low_weight > 0 and not is_robust_estimator(_weight_estimator):
        print("[GP] NOTE: Joint Bayesian optimisation freeze_low_weight needs a bagged "
              "weight_estimator ('bagged_cluster' or 'ensemble_selection' - it uses the bagged "
              "weights) - it is ignored for this task.")
        _freeze_low_weight = 0.0
    if _freeze_low_weight >= 1.0 / max(1, len(group_models)):
        print(f"[GP] NOTE: Joint Bayesian optimisation freeze_low_weight={_freeze_low_weight:g} is "
              f"at or above the equal weight 1/{len(group_models)}="
              f"{1.0 / max(1, len(group_models)):.3f}, so almost every model can be frozen "
              f"(the model with the highest P90 weight is never frozen) - a value around 0.05 "
              f"is the usual choice.")
    if _freeze_low_weight > 0 and (_search_strategy == 'monolithic' or (
            _search_strategy == 'async_library'
            and budget_kwargs.get('library_stop_rule',
                                  _JOINT_LIBRARY_STOP_RULE_DEFAULT) != 'value')):
        print(f"[GP] NOTE: Joint Bayesian optimisation freeze_low_weight only applies to "
              f"search_strategy 'block_coordinate' and 'parallel_library' (and to 'async_library' "
              f"with library_stop_rule='value') - it has no effect on the {_search_strategy!r} "
              f"search with these settings.")
    # R1.c: the library search's own knobs.
    _library_rounds = max(1, int(budget_kwargs.get('library_rounds', _JOINT_LIBRARY_ROUNDS_DEFAULT)
                                  or _JOINT_LIBRARY_ROUNDS_DEFAULT))
    _library_q = max(1, int(budget_kwargs.get('library_q', _JOINT_LIBRARY_Q_DEFAULT)
                             or _JOINT_LIBRARY_Q_DEFAULT))
    _v434_kwargs = dict(weight_estimator=_weight_estimator, estimator_settings=_estimator_settings,
                        objective_data=_objective_data, cv_folds=_cv_folds)
    print(f"[GP] Joint Bayesian optimisation settings: weight_estimator="
          f"{_weight_estimator!r} (weight_bags={_weight_bags}, cluster_corr={_cluster_corr}, "
          f"score_bags={_score_bags}), objective_data={_objective_data!r} "
          f"(cv_folds={_cv_folds}), search_space={_search_space!r} ({total_dims} dimension(s) "
          f"searched), freeze_low_weight={_freeze_low_weight or 'off'}, "
          f"time_budget_s={_time_budget_s or 'off'}, library_rounds={_library_rounds}, "
          f"library_q={_library_q}, search_strategy={_search_strategy!r}."
          + (" The time budget bounds the SEARCH only: the fresh gate, the confirmatory fit and "
             "any out-of-fold fit always run after it." if _time_budget_s else ""))
    # ver4-36 keys - read with `cfg.get('KEY', <legacy default>)` (I11); the resolved values
    # are logged, unconditionally, in the `ver4-36 settings` line below.
    _gate_mode = (budget_kwargs.get('gate_mode', _JOINT_GATE_MODE_DEFAULT)
                  or _JOINT_GATE_MODE_DEFAULT)
    if _gate_mode not in ('fresh_refit', 'bbc'):
        print(f"[GP] NOTE: Joint Bayesian optimisation gate_mode={_gate_mode!r} is not recognised "
              f"(expected 'fresh_refit' or 'bbc') - falling back to 'fresh_refit' for this task.")
        _gate_mode = 'fresh_refit'
    _bbc_bootstraps = int(budget_kwargs.get('bbc_bootstraps', _JOINT_BBC_BOOTSTRAPS_DEFAULT)
                          or _JOINT_BBC_BOOTSTRAPS_DEFAULT)
    if _bbc_bootstraps < _JOINT_BBC_MIN_VALID:
        print(f"[GP] NOTE: Joint Bayesian optimisation bbc_bootstraps={_bbc_bootstraps} is below "
              f"the minimum of {_JOINT_BBC_MIN_VALID} - using {_JOINT_BBC_BOOTSTRAPS_DEFAULT}.")
        _bbc_bootstraps = _JOINT_BBC_BOOTSTRAPS_DEFAULT
    _bbc_gate_quantile = float(budget_kwargs.get('bbc_gate_quantile',
                                                 _JOINT_BBC_GATE_QUANTILE_DEFAULT)
                               or _JOINT_BBC_GATE_QUANTILE_DEFAULT)
    if not (0.0 < _bbc_gate_quantile < 1.0):
        print(f"[GP] NOTE: Joint Bayesian optimisation bbc_gate_quantile={_bbc_gate_quantile} is "
              f"outside (0, 1) - using {_JOINT_BBC_GATE_QUANTILE_DEFAULT}.")
        _bbc_gate_quantile = _JOINT_BBC_GATE_QUANTILE_DEFAULT
    _search_test_stub = bool(budget_kwargs.get('search_test_stub',
                                               _JOINT_SEARCH_TEST_STUB_DEFAULT))
    _stub_parity_min = float(budget_kwargs.get('stub_parity_min', _JOINT_STUB_PARITY_MIN_DEFAULT)
                             or _JOINT_STUB_PARITY_MIN_DEFAULT)
    _search_engine = (budget_kwargs.get('search_engine', _JOINT_SEARCH_ENGINE_DEFAULT)
                      or _JOINT_SEARCH_ENGINE_DEFAULT)
    if _search_engine not in ('bglr', 'exact_kernel'):
        print(f"[GP] NOTE: Joint Bayesian optimisation search_engine={_search_engine!r} is not "
              f"recognised (expected 'bglr' or 'exact_kernel') - falling back to 'bglr' for "
              f"this task.")
        _search_engine = 'bglr'
    _engine_parity_min = float(budget_kwargs.get('engine_parity_min',
                                                 _JOINT_ENGINE_PARITY_MIN_DEFAULT)
                               or _JOINT_ENGINE_PARITY_MIN_DEFAULT)
    _library_evals_per_dim = max(1, int(budget_kwargs.get(
        'library_evals_per_dim', _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT)
        or _JOINT_LIBRARY_EVALS_PER_DIM_DEFAULT))
    _library_max_evals = max(2, int(budget_kwargs.get('library_max_evals',
                                                      _JOINT_LIBRARY_MAX_EVALS_DEFAULT)
                                    or _JOINT_LIBRARY_MAX_EVALS_DEFAULT))
    _refine_cfg = budget_kwargs.get('library_refine_rounds', _JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT)
    _library_refine_rounds = (_JOINT_LIBRARY_REFINE_ROUNDS_DEFAULT if _refine_cfg is None
                              else max(0, int(_refine_cfg)))
    _noise_aware = bool(budget_kwargs.get('noise_aware_surrogate', _JOINT_NOISE_AWARE_DEFAULT))
    _selection_rule = (budget_kwargs.get('selection_rule', _JOINT_SELECTION_RULE_DEFAULT)
                       or _JOINT_SELECTION_RULE_DEFAULT)
    if _selection_rule not in ('greedy', 'bbc'):
        print(f"[GP] NOTE: Joint Bayesian optimisation selection_rule={_selection_rule!r} is not "
              f"recognised (expected 'greedy' or 'bbc') - falling back to 'greedy' for this "
              f"task.")
        _selection_rule = 'greedy'
    _library_dump = bool(budget_kwargs.get('library_dump', _JOINT_LIBRARY_DUMP_DEFAULT))
    _library_strategy = _search_strategy in ('block_coordinate', 'parallel_library',
                                             'async_library', 'weight_first', 'weight_first_cv',
                                             'neural_stack')
    # Whether a fitted library will exist (the search must run and have > 1 model and > 0
    # dimensions); decided again when the search has actually run.
    _library_expected = bool(_library_strategy and total_dims > 0 and len(group_models) > 1)
    if _library_strategy and not _library_expected and not _safety_nets_on:
        # ver4-44, R1 (N1): without the net, a strategy that cannot run is an error - the task
        # would otherwise be evaluated with a different search than the one that was asked for.
        raise ValueError(
            f"[joint_tuning] safety_nets=False: search_strategy={_search_strategy!r} needs at "
            f"least one tunable dimension and >= 2 models, but this task has {total_dims} "
            f"dimension(s) and {len(group_models)} model(s) - the monolithic search would run "
            f"instead, which safety_nets=False does not allow. Choose search_strategy="
            f"'monolithic' or change the model group / search space.")
    # ver4-38 keys - read with `cfg.get('KEY', <legacy default>)` (I11), validated, and logged
    # unconditionally in the `ver4-38 settings` line below. Each requested feature that cannot
    # run for this task says so (log negative states) and falls back to its ver4-37 behaviour.
    _cv_racing_req = bool(budget_kwargs.get('cv_racing', _JOINT_CV_RACING_DEFAULT))
    _racing_folds = int(budget_kwargs.get('racing_folds', _JOINT_RACING_FOLDS_DEFAULT)
                        or _JOINT_RACING_FOLDS_DEFAULT)
    _racing_margin_cfg = budget_kwargs.get('racing_margin', _JOINT_RACING_MARGIN_DEFAULT)
    _racing_margin = (_JOINT_RACING_MARGIN_DEFAULT if _racing_margin_cfg is None
                      else max(0.0, float(_racing_margin_cfg)))
    _racing_penalty_cfg = budget_kwargs.get('racing_penalty', _JOINT_RACING_PENALTY_DEFAULT)
    _racing_penalty = (_JOINT_RACING_PENALTY_DEFAULT if _racing_penalty_cfg is None
                       else max(0.0, float(_racing_penalty_cfg)))
    _stop_rule = (budget_kwargs.get('library_stop_rule', _JOINT_LIBRARY_STOP_RULE_DEFAULT)
                  or _JOINT_LIBRARY_STOP_RULE_DEFAULT)
    if _stop_rule not in ('off', 'value'):
        print(f"[GP] NOTE: Joint Bayesian optimisation library_stop_rule={_stop_rule!r} is not "
              f"recognised (expected 'off' or 'value') - falling back to 'off' for this task.")
        _stop_rule = 'off'
    _oof_fidelity = (budget_kwargs.get('oof_fidelity', _JOINT_OOF_FIDELITY_DEFAULT)
                     or _JOINT_OOF_FIDELITY_DEFAULT)
    if _oof_fidelity not in ('full', 'search'):
        print(f"[GP] NOTE: Joint Bayesian optimisation oof_fidelity={_oof_fidelity!r} is not "
              f"recognised (expected 'full' or 'search') - falling back to 'full' for this task.")
        _oof_fidelity = 'full'
    _dead_column_guard = bool(budget_kwargs.get('dead_column_guard',
                                                _JOINT_DEAD_COLUMN_GUARD_DEFAULT))
    _library_ensemble_req = bool(budget_kwargs.get('library_ensemble',
                                                   _JOINT_LIBRARY_ENSEMBLE_DEFAULT))
    _members_max = max(2, int(budget_kwargs.get('library_members_max',
                                                _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT)
                              or _JOINT_LIBRARY_MEMBERS_MAX_DEFAULT))
    _members_per_model = max(1, int(budget_kwargs.get('library_members_per_model',
                                                       _JOINT_LIBRARY_MEMBERS_PER_MODEL_DEFAULT)
                                    or _JOINT_LIBRARY_MEMBERS_PER_MODEL_DEFAULT))
    # ver4-48 keys (R2) - read with `cfg.get('KEY', <legacy default>)` (I11), validated by
    # `_resolve_ns_settings()` and logged unconditionally in the `ver4-48 settings` line below.
    # They only act under search_strategy='neural_stack'.
    _ns = _resolve_ns_settings(budget_kwargs)
    if _search_strategy == 'neural_stack':
        # The mixture deployment IS the library ensemble (the members are the heavy library
        # columns), so the switch follows `ns_deploy`; a configured `library_ensemble` that
        # disagrees is overridden, and the task says so.
        _ns_members_req = (_ns['ns_deploy'] == 'mixture')
        if _library_ensemble_req != _ns_members_req:
            print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='neural_stack' sets "
                  f"library_ensemble={_ns_members_req} (ns_deploy={_ns['ns_deploy']!r}); the "
                  f"configured library_ensemble={_library_ensemble_req} is overridden for this "
                  f"task.")
        _library_ensemble_req = _ns_members_req
    # ver4-40 keys (R2) - read with `cfg.get('KEY', <legacy default>)` (I11), validated, and
    # logged unconditionally in the `ver4-40 settings` line below. They only act under
    # search_strategy='weight_first'.
    _wf_rules = budget_kwargs.get('wf_rules', _JOINT_WF_RULES_DEFAULT)
    if isinstance(_wf_rules, str):
        _wf_rules = [t.strip() for t in _wf_rules.split(',') if t.strip()]
    _wf_rules = list(_wf_rules) if _wf_rules else list(_JOINT_WF_RULES_DEFAULT)
    if _search_strategy == 'weight_first':
        try:
            _rng_chk = np.random.default_rng(0)
            weight_candidates(
                _rng_chk.normal(size=(8, len(group_models))), _rng_chk.normal(size=8),
                rules=[r for r in _wf_rules if str(r).strip().lower() != 'es'], dedup_tol=0.0,
                max_n=len(_wf_rules) + 1)
        except ValueError as exc:
            print(f"[GP] NOTE: Joint Bayesian optimisation wf_rules={_wf_rules!r} is not valid "
                  f"({exc}) - using the default rules {_JOINT_WF_RULES_DEFAULT!r} for this task.")
            _wf_rules = list(_JOINT_WF_RULES_DEFAULT)
    _wf_max = max(2, int(budget_kwargs.get('wf_max_candidates', _JOINT_WF_MAX_CANDIDATES_DEFAULT)
                         or _JOINT_WF_MAX_CANDIDATES_DEFAULT))
    _wf_dedup_cfg = budget_kwargs.get('wf_dedup_tol', _JOINT_WF_DEDUP_TOL_DEFAULT)
    _wf_dedup = (_JOINT_WF_DEDUP_TOL_DEFAULT if _wf_dedup_cfg is None
                 else max(0.0, float(_wf_dedup_cfg)))
    _wf_rungs = max(1, int(budget_kwargs.get('wf_rungs', _JOINT_WF_RUNGS_DEFAULT)
                           or _JOINT_WF_RUNGS_DEFAULT))
    _wf_eta = max(2, int(budget_kwargs.get('wf_eta', _JOINT_WF_ETA_DEFAULT)
                         or _JOINT_WF_ETA_DEFAULT))
    _wf_minw_cfg = budget_kwargs.get('wf_min_tune_weight', _JOINT_WF_MIN_TUNE_WEIGHT_DEFAULT)
    _wf_min_w = (_JOINT_WF_MIN_TUNE_WEIGHT_DEFAULT if _wf_minw_cfg is None
                 else float(np.clip(float(_wf_minw_cfg), 0.0, 1.0)))
    _wf_include_untuned = bool(budget_kwargs.get('wf_include_untuned',
                                                 _JOINT_WF_INCLUDE_UNTUNED_DEFAULT))
    _wf_pick = (budget_kwargs.get('wf_pick', _JOINT_WF_PICK_DEFAULT) or _JOINT_WF_PICK_DEFAULT)
    if _wf_pick not in _JOINT_WF_PICK_CHOICES:
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_pick={_wf_pick!r} is not recognised "
              f"(expected one of {list(_JOINT_WF_PICK_CHOICES)!r}) - falling back to 'bbc' for "
              f"this task.")
        _wf_pick = 'bbc'
    _wf_cand_rows = (budget_kwargs.get('wf_candidate_rows', _JOINT_WF_CANDIDATE_ROWS_DEFAULT)
                     or _JOINT_WF_CANDIDATE_ROWS_DEFAULT)
    if _wf_cand_rows not in ('scoring', 'oof_train'):
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_candidate_rows={_wf_cand_rows!r} is not "
              f"recognised (expected 'scoring' or 'oof_train') - falling back to 'scoring' for "
              f"this task.")
        _wf_cand_rows = 'scoring'
    # ver4-42 keys (R1) - read with `cfg.get('KEY', <default>)` (I11), validated, and logged
    # unconditionally in the `ver4-42 settings` line below. They only act under
    # search_strategy='weight_first_cv'.
    _wfcv_fidelity = (budget_kwargs.get('wf_cv_weight_fidelity', _JOINT_WFCV_WEIGHT_FIDELITY_DEFAULT)
                      or _JOINT_WFCV_WEIGHT_FIDELITY_DEFAULT)
    if _wfcv_fidelity not in ('full', 'search'):
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_weight_fidelity={_wfcv_fidelity!r} is "
              f"not recognised (expected 'full' or 'search') - falling back to 'full' for this "
              f"task.")
        _wfcv_fidelity = 'full'
    _wfcv_direction = (budget_kwargs.get('wf_cv_direction', _JOINT_WFCV_DIRECTION_DEFAULT)
                       or _JOINT_WFCV_DIRECTION_DEFAULT)
    if _wfcv_direction not in ('analytic', 'es'):
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_direction={_wfcv_direction!r} is not "
              f"recognised (expected 'analytic' or 'es') - falling back to 'analytic' for this "
              f"task.")
        _wfcv_direction = 'analytic'
    _wfcv_alphas_cfg = budget_kwargs.get('wf_cv_alphas', _JOINT_WFCV_ALPHAS_DEFAULT)
    if isinstance(_wfcv_alphas_cfg, str):
        _wfcv_alphas_cfg = [t.strip() for t in _wfcv_alphas_cfg.split(',') if t.strip()]
    try:
        _wfcv_alphas = sorted({float(a) for a in (_wfcv_alphas_cfg or [])})
        if not _wfcv_alphas or min(_wfcv_alphas) < 0.0 or max(_wfcv_alphas) > 1.0:
            raise ValueError("every alpha must lie in [0, 1] and at least one is needed")
    except (TypeError, ValueError) as exc:
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_alphas={_wfcv_alphas_cfg!r} is not "
              f"valid ({exc}) - using the default {_JOINT_WFCV_ALPHAS_DEFAULT!r} for this task.")
        _wfcv_alphas = list(_JOINT_WFCV_ALPHAS_DEFAULT)
    _wfcv_alpha_rule = (budget_kwargs.get('wf_cv_alpha_rule', _JOINT_WFCV_ALPHA_RULE_DEFAULT)
                        or _JOINT_WFCV_ALPHA_RULE_DEFAULT)
    if _wfcv_alpha_rule not in ('min', 'one_se'):
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_alpha_rule={_wfcv_alpha_rule!r} is "
              f"not recognised (expected 'min' or 'one_se') - falling back to 'min' for this "
              f"task.")
        _wfcv_alpha_rule = 'min'
    _wfcv_repeats = max(1, int(budget_kwargs.get('wf_cv_repeats', _JOINT_WFCV_REPEATS_DEFAULT)
                               or _JOINT_WFCV_REPEATS_DEFAULT))
    _wfcv_tune = bool(budget_kwargs.get('wf_cv_tune', _JOINT_WFCV_TUNE_DEFAULT))
    _wfcv_accept = (budget_kwargs.get('wf_cv_accept', _JOINT_WFCV_ACCEPT_DEFAULT)
                    or _JOINT_WFCV_ACCEPT_DEFAULT)
    if _wfcv_accept not in _JOINT_WFCV_ACCEPT_CHOICES:
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_accept={_wfcv_accept!r} is not "
              f"recognised (expected one of {list(_JOINT_WFCV_ACCEPT_CHOICES)!r}) - falling back "
              f"to 'bbc_mean' for this task.")
        _wfcv_accept = 'bbc_mean'
    # ver4-44, R3: how the deployed weight direction is formed (weight_first_cv only).
    _wfcv_final = (budget_kwargs.get('wf_cv_final', _JOINT_WFCV_FINAL_DEFAULT)
                   or _JOINT_WFCV_FINAL_DEFAULT)
    if _wfcv_final not in _JOINT_WFCV_FINAL_CHOICES:
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_final={_wfcv_final!r} is not "
              f"recognised (expected one of {list(_JOINT_WFCV_FINAL_CHOICES)!r}) - falling back "
              f"to 'refit' for this task.")
        _wfcv_final = 'refit'
    # ver4-47, R5: re-estimate the deployed weights on the tuned columns (weight_first_cv only).
    _wfcv_reweight = (budget_kwargs.get('wf_cv_reweight', _JOINT_WFCV_REWEIGHT_DEFAULT)
                      or _JOINT_WFCV_REWEIGHT_DEFAULT)
    if _wfcv_reweight not in _JOINT_WFCV_REWEIGHT_CHOICES:
        print(f"[GP] NOTE: Joint Bayesian optimisation wf_cv_reweight={_wfcv_reweight!r} is not "
              f"recognised (expected one of {list(_JOINT_WFCV_REWEIGHT_CHOICES)!r}) - falling "
              f"back to 'off' for this task.")
        _wfcv_reweight = 'off'
    _wfcv_members = bool(budget_kwargs.get('wf_cv_members', _JOINT_WFCV_MEMBERS_DEFAULT))
    if not _safety_nets_on and _search_strategy == 'weight_first_cv' and (
            list(_wfcv_alphas) != list(_JOINT_WFCV_ALPHAS_DEFAULT)
            or _wfcv_alpha_rule != _JOINT_WFCV_ALPHA_RULE_DEFAULT):
        print(f"[GP] NOTE: Joint Bayesian optimisation safety_nets=False ignores wf_cv_alphas="
              f"{_wfcv_alphas!r} and wf_cv_alpha_rule={_wfcv_alpha_rule!r}: no shrinkage towards "
              f"equal weights is applied (alpha is 1).")
    print(f"[GP] Joint Bayesian optimisation settings: safety_nets={_safety_nets_on}, "
          f"wf_cv_final={_wfcv_final!r} (read only under 'weight_first_cv'), search_strategy="
          f"{_search_strategy!r}, gate_mode={_gate_mode!r}, fresh_gate={_fresh_gate}, "
          f"untuned_baseline_floor={_untuned_baseline_floor}, weight_cv={_weight_cv}, "
          f"diversity_penalty={_diversity_penalty:g}, shrinkage_alpha={_shrinkage_alpha:g}.")
    _racing_active = False
    if _cv_racing_req:
        _why_race = []
        if _search_strategy not in ('async_library', 'weight_first_cv') or not _library_expected:
            _why_race.append(f"search_strategy is {_search_strategy!r} (racing needs "
                             f"'async_library' or 'weight_first_cv' with a tunable dimension and "
                             f">= 2 models; the other strategies have shared barriers)")
        if not _cv_active:
            _why_race.append("objective_data is not 'cv_train_valid' (no folds to race on)")
        if _racing_folds < 1 or _racing_folds >= _cv_folds:
            _why_race.append(f"racing_folds={_racing_folds} is not between 1 and cv_folds-1="
                             f"{_cv_folds - 1}")
        if _why_race:
            print("[GP] NOTE: Joint Bayesian optimisation cv_racing is not available for this "
                  "task - " + '; '.join(_why_race) + " - every proposal is fitted on every fold.")
        else:
            _racing_active = True
    _stop_rule_active = False
    if _stop_rule == 'value':
        if _search_strategy != 'async_library' or not _library_expected:
            print(f"[GP] NOTE: Joint Bayesian optimisation library_stop_rule='value' needs "
                  f"search_strategy='async_library' with a tunable dimension and >= 2 models "
                  f"(this task uses {_search_strategy!r}) - no pipeline is stopped early.")
        else:
            _stop_rule_active = True
    _oof_search_active = False
    if _oof_fidelity == 'search':
        if not _cv_active:
            print("[GP] NOTE: Joint Bayesian optimisation oof_fidelity='search' needs "
                  "objective_data='cv_train_valid' (the final out-of-fold fits only exist for "
                  "that objective) - the final weights are fitted as before (oof_fidelity="
                  "'full').")
        else:
            _oof_search_active = True
    if _dead_column_guard and not _library_expected:
        print(f"[GP] NOTE: Joint Bayesian optimisation dead_column_guard acts inside the library "
              f"searches ('block_coordinate', 'parallel_library', 'async_library'; this task uses "
              f"{_search_strategy!r}); a constant column in the FINAL fits still gets weight 0.")
    _library_ensemble_active = False
    if _library_ensemble_req:
        _why_ens = []
        if not _library_expected:
            _why_ens.append(f"search_strategy {_search_strategy!r} builds no library")
        _wf_members_ok = bool((_wfcv_members and _search_strategy == 'weight_first_cv')
                              or _search_strategy == 'neural_stack')
        if _search_strategy in _WF_STRATEGIES and not _wf_members_ok:
            _why_ens.append(f"search_strategy {_search_strategy!r} fixes the ensemble weights (a "
                            f"member-weighted ensemble would re-estimate them)")
        if not _cv_active:
            _why_ens.append("objective_data is not 'cv_train_valid' (the member weights need "
                            "out-of-fold columns over train+valid)")
        if _weight_estimator != 'ensemble_selection' and not _wf_members_ok:
            _why_ens.append(f"weight_estimator is {_weight_estimator!r} (the member weights use "
                            f"'ensemble_selection')")
        if _why_ens:
            print("[GP] NOTE: Joint Bayesian optimisation library_ensemble is not available for "
                  "this task - " + '; '.join(_why_ens) + " - one configuration per model is used "
                  "(the joint row).")
        else:
            _library_ensemble_active = True
    # Shortcut eligibility (R1.c / R1.d, blueprint D4): BLUE weighting makes the runner pass a
    # full-length weight vector with a row subset, which neither shortcut reproduces; a missing
    # context means "cannot verify". The dump needs real test columns, so the stub is off.
    _shortcut_why = []
    if (_search_test_stub or _search_engine == 'exact_kernel') and total_dims > 0:
        if eligibility_ctx is None:
            _shortcut_why.append("the caller supplied no eligibility context, so BLUE weighting "
                                 "cannot be ruled out")
        elif eligibility_ctx.get('blue_weighting'):
            _shortcut_why.append("BLUE weighting is active (the runner passes a full-length "
                                 "weight vector with a row subset)")
    _stub_active = _search_test_stub and not _shortcut_why and total_dims > 0
    _engine_active = (_search_engine == 'exact_kernel') and not _shortcut_why and total_dims > 0
    if _shortcut_why:
        print(f"[GP] NOTE: Joint Bayesian optimisation search_test_stub / search_engine="
              f"'exact_kernel' are not available for this task - " + '; '.join(_shortcut_why)
              + " - every search fit is a real fit on the full test frame.")
    if _library_dump and _stub_active:
        print("[GP] NOTE: Joint Bayesian optimisation library_dump needs the TEST column of every "
              "fit, so search_test_stub is switched off for this task (the exact engine, if on, "
              "returns test columns from the same fit).")
        _stub_active = False
    if _library_dump and not _library_expected:
        print(f"[GP] NOTE: Joint Bayesian optimisation library_dump needs a library search "
              f"('block_coordinate', 'parallel_library' or 'async_library' with a tunable "
              f"dimension and >= 2 models); this task uses {_search_strategy!r} - nothing will "
              f"be written.")
    if _library_dump and diagnostics_ctx is None:
        print("[GP] NOTE: Joint Bayesian optimisation library_dump=True but the caller supplied no "
              "diagnostics_ctx (task tag and result folder) - nothing will be written.")
    if (_gate_mode == 'bbc' or _selection_rule == 'bbc') and not _library_expected:
        print(f"[GP] NOTE: Joint Bayesian optimisation gate_mode / selection_rule 'bbc' need the "
              f"fitted library of a library search (this task uses {_search_strategy!r}) - the "
              f"fresh-resample gate and the greedy selection are used instead.")
    if _noise_aware and total_dims == 0:
        print("[GP] NOTE: Joint Bayesian optimisation noise_aware_surrogate has nothing to act on "
              "(no tunable dimension) for this task.")
    if _search_strategy in ('async_library', 'weight_first', 'weight_first_cv', 'neural_stack') \
            and _search_patience > 0:
        print(f"[GP] NOTE: Joint Bayesian optimisation search_patience={_search_patience} is not "
              f"used by search_strategy={_search_strategy!r} (each model has its own column budget) "
              f"- it is ignored for this task.")
    print(f"[GP] Joint Bayesian optimisation settings: gate_mode={_gate_mode!r} "
          f"(bbc_bootstraps={_bbc_bootstraps}, bbc_gate_quantile={_bbc_gate_quantile:g}), "
          f"selection_rule={_selection_rule!r}, search_test_stub={_search_test_stub} "
          f"(active={_stub_active}, stub_parity_min={_stub_parity_min:g}), search_engine="
          f"{_search_engine!r} (active={_engine_active}, engine_parity_min="
          f"{_engine_parity_min:g}), library_evals_per_dim={_library_evals_per_dim}, "
          f"library_max_evals={_library_max_evals}, library_refine_rounds="
          f"{_library_refine_rounds}, noise_aware_surrogate={_noise_aware}, weight_estimator="
          f"{_weight_estimator!r} (es_steps={_es_steps or 'auto'}, es_model_fraction="
          f"{_es_model_fraction:g}), library_dump={_library_dump}.")
    print(f"[GP] Joint Bayesian optimisation settings: cv_racing={_cv_racing_req} "
          f"(active={_racing_active}, racing_folds={_racing_folds}, racing_margin="
          f"{_racing_margin:g}, racing_penalty={_racing_penalty:g}), library_stop_rule="
          f"{_stop_rule!r} (active={_stop_rule_active}), oof_fidelity={_oof_fidelity!r} "
          f"(active={_oof_search_active}), dead_column_guard={_dead_column_guard}, "
          f"library_ensemble={_library_ensemble_req} (active={_library_ensemble_active}, "
          f"library_members_max={_members_max}, library_members_per_model="
          f"{_members_per_model}).")
    _wf_requested = (_search_strategy == 'weight_first')
    if _wf_requested and not (total_dims > 0 and len(group_models) > 1):
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first' needs a "
              f"tunable dimension and >= 2 models (this task has {total_dims} dimension(s) and "
              f"{len(group_models)} model(s)) - the monolithic search runs instead.")
    if _wf_requested:
        _wf_ignored = []
        if _selection_rule != 'greedy':
            _wf_ignored.append(f"selection_rule={_selection_rule!r}")
        if _cv_racing_req:
            _wf_ignored.append("cv_racing")
        if _stop_rule != 'off':
            _wf_ignored.append(f"library_stop_rule={_stop_rule!r}")
        if _freeze_low_weight > 0:
            _wf_ignored.append("freeze_low_weight")
        if _library_ensemble_req:
            _wf_ignored.append("library_ensemble")
        if _untuned_baseline_floor:
            _wf_ignored.append("untuned_baseline_floor")
        if _weight_source != 'valid':
            _wf_ignored.append(f"weight_source={_weight_source!r}")
        if _wf_ignored:
            print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first' ignores "
                  f"{', '.join(_wf_ignored)} - the final weights are the chosen FIXED vector "
                  f"(never re-estimated), so the weight-estimator, equal-weight-floor and "
                  f"library-ensemble machinery has nothing to act on.")
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first' replaces "
              f"gate_mode={_gate_mode!r}: the pick among the (weight vector, columns) finalists "
              f"is made by wf_pick={_wf_pick!r}"
              + {'bbc': " (zero-fit bootstrap out-of-bag score with the one-standard-error "
                        "parsimony rule: the simplest finalist that is not clearly worse than "
                        "the best).",
                 'bbc_max': " (zero-fit bootstrap out-of-bag score, plain highest mean).",
                 'max': " (highest full-row score; no out-of-bag protection)."}[_wf_pick])
    print(f"[GP] Joint Bayesian optimisation settings: search_strategy={_search_strategy!r}, "
          f"wf_rules={_wf_rules!r}, wf_max_candidates={_wf_max}, wf_dedup_tol={_wf_dedup:g}, "
          f"wf_rungs={_wf_rungs}, wf_eta={_wf_eta}, wf_min_tune_weight={_wf_min_w:g}, "
          f"wf_include_untuned={_wf_include_untuned}, wf_pick={_wf_pick!r}, "
          f"wf_candidate_rows={_wf_cand_rows!r} (read only under 'weight_first').")
    if _search_strategy == 'weight_first_cv' and not (total_dims > 0 and len(group_models) > 1):
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first_cv' needs a "
              f"tunable dimension and >= 2 models (this task has {total_dims} dimension(s) and "
              f"{len(group_models)} model(s)) - the monolithic search runs instead.")
    if _search_strategy == 'weight_first_cv' and (_wfcv_reweight != 'off' or _wfcv_members):
        print(f"[GP] Joint Bayesian optimisation settings: wf_cv_reweight="
              f"{_wfcv_reweight!r} (S4: the deployed weights are re-estimated on the tuned "
              f"columns; 'off' = the S1 weights are final), wf_cv_members={_wfcv_members} "
              f"(the model weights are split over the top-{_members_per_model} tuned "
              f"configurations per model; needs library_ensemble, requested="
              f"{_library_ensemble_req}).")
    if _search_strategy == 'weight_first_cv':
        _wfcv_ignored = []
        if _selection_rule != 'greedy':
            _wfcv_ignored.append(f"selection_rule={_selection_rule!r}")
        if _freeze_low_weight > 0:
            _wfcv_ignored.append("freeze_low_weight")
        if _library_ensemble_req and not _wfcv_members:        # ver4-47: R6 uses it
            _wfcv_ignored.append("library_ensemble")
        if _untuned_baseline_floor:
            _wfcv_ignored.append("untuned_baseline_floor")
        if _weight_source != 'valid':
            _wfcv_ignored.append(f"weight_source={_weight_source!r}")
        if _weight_estimator != _JOINT_WEIGHT_ESTIMATOR_DEFAULT:
            _wfcv_ignored.append(f"weight_estimator={_weight_estimator!r}")
        if _wfcv_ignored:
            print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first_cv' "
                  f"ignores {', '.join(_wfcv_ignored)} - the final weights are the FIXED vector "
                  f"of stage S1 (never re-estimated), so the weight-estimator, equal-weight-floor "
                  f"and library-ensemble machinery has nothing to act on.")
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='weight_first_cv' replaces "
              f"gate_mode={_gate_mode!r}: stage S3 decides between the tuned and the untuned "
              f"columns at the same fixed weights (wf_cv_accept={_wfcv_accept!r}"
              + {'bbc_mean': ", zero-fit bootstrap out-of-bag mean, no parsimony margin).",
                 'bbc_one_se': ", zero-fit bootstrap out-of-bag mean with the one-standard-error "
                               "parsimony rule).",
                 'always': " - the tuned columns are always taken)."}[_wfcv_accept])
    print(f"[GP] Joint Bayesian optimisation settings: search_strategy="
          f"{_search_strategy!r}, ns_configs={_ns['ns_configs']} (ns_configs_mcmc="
          f"{_ns['ns_configs_mcmc']}), ns_lambdas={_ns['ns_lambdas']!r} (lambda=inf prior "
          f"{'added' if _safety_nets_on else 'absent: safety_nets=False'}), ns_lambda_rule="
          f"{(_ns['ns_lambda_rule'] if _safety_nets_on else 'min')!r}"
          f"{'' if _safety_nets_on else ' (forced: safety_nets=False)'}, ns_repeats="
          f"{_ns['ns_repeats']}, ns_deploy={_ns['ns_deploy']!r}, ns_fidelity="
          f"{_ns['ns_fidelity']!r}, ns_weight_shrink={_ns['ns_weight_shrink']:g}, "
          f"ns_objective={_ns['ns_objective']!r}, ns_swap_rule={_ns['ns_swap_rule']!r}, "
          f"ns_outer_gate={_ns['ns_outer_gate']!r}, ns_outer_folds={_ns['ns_outer_folds']}, "
          f"ns_outer_repeats={_ns['ns_outer_repeats']} (read only under 'neural_stack').")
    if _search_strategy == 'neural_stack' and not (total_dims > 0 and len(group_models) > 1):
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='neural_stack' needs a "
              f"tunable dimension and >= 2 models (this task has {total_dims} dimension(s) and "
              f"{len(group_models)} model(s)) - the monolithic search runs instead.")
    if _search_strategy == 'neural_stack':
        _ns_ignored = []
        if _selection_rule != 'greedy':
            _ns_ignored.append(f"selection_rule={_selection_rule!r}")
        if _cv_racing_req:
            _ns_ignored.append("cv_racing")
        if _stop_rule != 'off':
            _ns_ignored.append(f"library_stop_rule={_stop_rule!r}")
        if _freeze_low_weight > 0:
            _ns_ignored.append("freeze_low_weight")
        if _untuned_baseline_floor:
            _ns_ignored.append("untuned_baseline_floor")
        if _weight_source != 'valid':
            _ns_ignored.append(f"weight_source={_weight_source!r}")
        if _weight_estimator != _JOINT_WEIGHT_ESTIMATOR_DEFAULT:
            _ns_ignored.append(f"weight_estimator={_weight_estimator!r}")
        if _oof_fidelity != _JOINT_OOF_FIDELITY_DEFAULT:
            _ns_ignored.append(f"oof_fidelity={_oof_fidelity!r} (use ns_fidelity)")
        if _ns_ignored:
            print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='neural_stack' ignores "
                  f"{', '.join(_ns_ignored)} - the final weights are the FIXED vector of the "
                  f"network (never re-estimated), so the weight-estimator, equal-weight-floor "
                  f"and library-search machinery has nothing to act on.")
        print(f"[GP] NOTE: Joint Bayesian optimisation search_strategy='neural_stack' replaces "
              f"gate_mode={_gate_mode!r}: the lambda chosen by repeated cross-validation decides "
              f"between the untuned naive ensemble (lambda=inf) and the fitted network (decision "
              f"'ns_prior' / 'ns_fitted'); with safety_nets=False the lambda=inf candidate is "
              f"removed and the rule is 'min' (set ns_lambdas=[0] for no regularisation at all).")
    print(f"[GP] Joint Bayesian optimisation settings: search_strategy="
          f"{_search_strategy!r}, wf_cv_weight_fidelity={_wfcv_fidelity!r}, wf_cv_direction="
          f"{_wfcv_direction!r}, wf_cv_alphas={_wfcv_alphas!r}, wf_cv_alpha_rule="
          f"{_wfcv_alpha_rule!r}, wf_cv_repeats={_wfcv_repeats}, wf_cv_tune={_wfcv_tune}, "
          f"wf_cv_accept={_wfcv_accept!r} (read only under 'weight_first_cv').")
    _estimate_budget = dict(
        budget_kwargs, n_iter=n_iter, init_points=init_points, search_strategy=_search_strategy,
        wf_cv_weight_fidelity=_wfcv_fidelity, wf_cv_tune=_wfcv_tune,
        wf_rules=_wf_rules, wf_max_candidates=_wf_max, wf_rungs=_wf_rungs, wf_eta=_wf_eta,
        wf_candidate_rows=_wf_cand_rows,
        search_space=_search_space, freeze_fidelity_fields=_freeze_fidelity,
        objective_data=_objective_data, cv_folds=_cv_folds, valid_repeats=_valid_repeats,
        fresh_gate=_fresh_gate, fresh_gate_repeats=_fresh_gate_repeats,
        weight_source=('valid' if _cv_active else _weight_source), oof_folds=_oof_folds,
        library_rounds=_library_rounds, library_q=_library_q, gate_mode=_gate_mode,
        search_test_stub=_stub_active, search_engine=('exact_kernel' if _engine_active else 'bglr'),
        library_evals_per_dim=_library_evals_per_dim, library_max_evals=_library_max_evals,
        library_refine_rounds=_library_refine_rounds,
        cv_racing=_racing_active, racing_folds=_racing_folds,
        oof_fidelity=('search' if _oof_search_active else 'full'),
        library_ensemble=_library_ensemble_active, library_members_max=_members_max,
        library_stop_rule=('value' if _stop_rule_active else 'off'),
        **_ns)
    _fit_estimate_full = _estimate_joint_fits(group_models, hparam_specs, _estimate_budget)
    _fit_estimate = {k: v for k, v in _fit_estimate_full.items() if not isinstance(v, dict)}
    print(f"[GP] Joint Bayesian optimisation fit-count estimate for this task (cache hits and "
          f"early stops not predicted, so an upper bound for the block and library searches; "
          f"Step 9's own refit is not counted): search={_fit_estimate['search']} "
          f"gate={_fit_estimate['gate']} confirm={_fit_estimate['confirm']} "
          f"oof={_fit_estimate['oof']} total={_fit_estimate['total']} | sequential fit batches="
          f"{_fit_estimate['seq_batches']} | BGLR full-length fits="
          f"{_fit_estimate['bglr_full_length_fits']}"
          + (f" | MCMC fits={_fit_estimate['mcmc_fits']} exact-engine fits="
             f"{_fit_estimate['exact_fits']} critical path={_fit_estimate['critical_path']} "
             f"round(s)" if (_stub_active or _engine_active or _gate_mode == 'bbc'
                             or _search_strategy in ('async_library', 'weight_first',
                                                     'weight_first_cv', 'neural_stack')) else "")
          + ".")

    # ver4-28, R6: one shared, mutable per-stage fit counter for this entire
    # search's own `fits:` provenance log line - see `_run_units()`'s own
    # `counter`/`stage` parameters for how each stage's count is populated.
    # ver4-32 R3: the nested 'secs' dict holds wall-clock seconds per stage
    # (kept apart from the integer counts so `n_fits` stays all integers).
    # ver4-34 R1.a: the nested 'fits_by_model'/'secs_by_model' dicts hold the per-model
    # cost (also kept apart from the integer stage counts).
    _fit_counter = {'search': 0, 'gate': 0, 'confirm': 0, 'oof': 0, 'cache_hits': 0,
                    'secs': {'search': 0.0, 'gate': 0.0, 'confirm': 0.0, 'oof': 0.0},
                    'fits_by_model': {m: 0 for m in group_models},
                    'secs_by_model': {m: 0.0 for m in group_models},
                    # ver4-36: how many fits were answered by the exact engine / ran with the
                    # 1-row test stub / were ordinary runner fits (nested, so it stays out of
                    # the integer stage counts)
                    'engine_fits': {'exact_kernel': 0, 'stub': 0, 'runner': 0}}
    _gate_decision = 'not_run'
    # ver4-36: per-model shortcut flags (filled in by the parity guards below; empty sets =
    # every fit is a normal fit, byte-identical to ver4-35) and the test-column sink of the
    # library dump (None unless a dump will be written).
    _dump_on = bool(_library_dump and _library_expected and diagnostics_ctx is not None)
    _fit_flags = {'engine_models': set(), 'stub_models': set(), 'keep_test': _dump_on}
    if _oof_search_active:
        # ver4-38, R1.d: the final out-of-fold requests must hash like the library entries they
        # reuse; an exact-engine library column is therefore also served by the engine in the
        # 'oof' stage (the default `engine_stages` stop at 'gate').
        _fit_flags['engine_stages'] = ('search', 'gate', 'oof')
    _test_sink = {} if _dump_on else None

    # BUGFIX (joint search reverting to a serial search under batch Bayesian
    # optimisation, root-caused to `ValueError: Input y contains infinity or
    # a value too large for dtype('float64')` raised from inside bayes_opt's
    # GP surrogate refit - see `_failure_score()`'s own docstring for the
    # full mechanism): ONE shared, mutable `{'worst_finite': ...}` dict for
    # this entire search, threaded into every `_evaluate`/`_screen_score`/
    # `_evaluate_with_se`/confirmatory/untuned-baseline-floor evaluation
    # below, mirroring `models/hyperparameter_tuning.py::make_objective()`'s
    # own closure-local `state` - the free-function equivalent here, since
    # `_score_candidate()` is called from several distinct places in this
    # function rather than from inside one closure. Every failing/
    # degenerate candidate this search evaluates is now scored strictly
    # below every finite score actually observed so far (or `FAILURE_FLOOR`
    # before any candidate has succeeded), never as a literal, non-finite
    # `-inf` a Gaussian-process surrogate cannot fit against.
    state = {'worst_finite': None}

    # ver4-28, R1: every `_score_candidate()` call this function makes below
    # is handed a `runner_ctx` built by this factory - `stage` buckets the
    # fit into the right slot of `_fit_counter` for the R6 provenance log
    # line; `cache` is only ever non-None inside `_block_coordinate_search()`
    # itself (never for the monolithic strategy - see `_JOINT_SEARCH_
    # STRATEGY_DEFAULT`'s own note).
    def _runner_ctx(stage, cache=None):
        # ver4-34, R2.b: with the CV objective ONE cache (`_cv_cache`) serves the search
        # and the final out-of-fold fits, so a model whose search parameters equal its
        # final parameters is not fitted a second time. Without it `cache` is only ever
        # non-None inside the block/library searches and the gate, as before.
        if cache is None and _cv_cache is not None and stage in ('search', 'oof'):
            cache = _cv_cache
        return {
            'n_jobs': _effective_n_jobs, 'worker_init': worker_init,
            'worker_init_args': worker_init_args, 'cache': cache,
            'counter': _fit_counter, 'stage': stage,
            'fit_flags': _fit_flags, 'test_sink': _test_sink,
        }

    def _evaluate(x, *, search_fidelity):
        return joint_objective(
            x, group_models=group_models, base_params_by_model=base_params_by_model,
            specs_by_model=specs_by_model, train=train, valid=valid, test=test,
            run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME,
            objective_mode=objective_mode, search_fidelity=search_fidelity,
            diversity_penalty=_diversity_penalty, diversity_method=_diversity_method,
            valid_repeats=_valid_repeats, inner_split_seed_base=inner_split_seed_base,
            state=state, weight_cv_alpha=_weight_cv_alpha, runner_ctx=_runner_ctx('search'),
            weight_rule=_score_weight_rule, shrinkage_alpha=_shrinkage_alpha, **_v434_kwargs,
        )

    # ver4-32: the `_score_candidate()` keywords every gate evaluation shares
    # (see `_paired_fresh_gate()`).
    _gate_score_kwargs = dict(
        group_models=group_models, base_params_by_model=base_params_by_model,
        specs_by_model=specs_by_model, train=train, valid=valid, test=test,
        run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME, objective_mode=objective_mode,
        search_fidelity=True, diversity_penalty=_diversity_penalty,
        diversity_method=_diversity_method, state=state, weight_cv_alpha=_weight_cv_alpha,
        weight_rule=_score_weight_rule, shrinkage_alpha=_shrinkage_alpha, **_v434_kwargs,
    )

    def _make_acquisition():
        return ExpectedImprovement(xi=0.01) if ExpectedImprovement is not None else None

    def _make_bounds_transformer():
        # R3 lever 5: domain reduction is disabled whenever any dimension
        # is categorical (SequentialDomainReductionTransformer only
        # supports an all-continuous box) - the SAME rule
        # HP_TUNE_BAYES_DOMAIN_REDUCTION='auto' already applies for
        # single-model tuning, evaluated here over the JOINT spec instead
        # of one model's own.
        if has_categorical or SequentialDomainReductionTransformer is None or total_dims == 0:
            return None
        return SequentialDomainReductionTransformer(minimum_window=[0.02] * total_dims)

    # ver4-27, R3: seed the search with each group model's OWN current
    # hyperparameters (the SAME base_params_by_model[m] the search already
    # falls back to for every non-tunable field - the model's shipped
    # default, or whatever HPARAMETERS/HPARAMETERS_BASELINE already holds
    # for it, e.g. from an earlier independent HP_TUNE run) encoded into
    # this search's own unit-cube coordinates via _encode_from_params()
    # (hyperparameter_tuning.py's own exact inverse of _decode(), already
    # used there for single-model tuning's own default-vs-winner
    # safeguard), rather than the previous blind midpoint-everywhere
    # probe. This matters more as more models are jointly tuned: total_dims
    # grows with every included model's own tunable-field count, and a
    # plain midpoint carries no information about which region of an
    # increasingly large combined space is actually plausible, whereas
    # each model's own current hyperparameters are a concrete, known-
    # reasonable point to refine FROM rather than search a huge space
    # cold.
    #
    # SUPERSEDED BY ROUND 6 (joint-bayesian-optimisation-underperformance-
    # fix): rounds prior to this one carried an extended history here about
    # getting the WEIGHT portion of this seed point right (`stick_breaking_
    # to_simplex()`'s asymmetric coordinate mapping meant the bare unit-cube
    # midpoint was not actually equal weighting for K>2 group models, biasing
    # the very "no worse than not tuning at all" baseline every margin/1-SE
    # check below is graded against). Round 6 makes that entire history moot
    # rather than further patching it: there is no weight portion of this
    # seed - or of any candidate `x` - any more (see this module's own
    # top-level Round 6 note). `seed_point` now spans ONLY the concatenated
    # per-model hyperparameter dimensions, and whatever weighting the
    # baseline/winner/every other candidate gets is always this candidate's
    # own analytically-optimal weighting, solved fresh inside
    # `_score_candidate()` from its own realised predictions - never seeded,
    # never decoded, never capable of being skewed by a coordinate-mapping
    # asymmetry.
    # Exactly ONE probe point is passed to the search below, never two:
    # models/parallel_search.py::batch_bayesian_maximise()'s own
    # documented `1 + init_points + n_iter` budget-accounting contract
    # assumes exactly one probe, and passing a second would silently use
    # one evaluation more than the configured budget promises - so this
    # REPLACES the previous probe rather than adding to it.
    _seed_chunks = [
        _encode_from_params(specs_by_model[m], base_params_by_model[m])
        if specs_by_model[m] else np.array([], dtype=float)
        for m in group_models
    ]
    seed_point = (
        np.concatenate(_seed_chunks)
        if total_dims > 0 else np.array([], dtype=float)
    )

    best_x, best_score = None, None
    _warm_x = None  # ver4-32 R5: the individually tuned point, set by the block search

    # ver4-36, R1.c / R1.d: the parity guards decide ONCE per task and per model whether the
    # 1-row test stub / exact kernel engine may serve the search fits (the confirmatory fit is
    # never touched, see `_apply_fit_flags()`). They run before any strategy, so a
    # monolithic search benefits too. A guard failure never fails the task: both shortcuts then
    # simply stay off.
    _library_cache = ((_cv_cache if _cv_cache is not None else _FitCache())
                      if _library_strategy else None)
    _t_seed = {}
    _parity = None
    if (_stub_active or _engine_active) and total_dims > 0:
        try:
            _parity = _parity_guards(
                group_models=group_models, specs_by_model=specs_by_model,
                base_params_by_model=base_params_by_model, seed_point=seed_point, train=train,
                valid=valid, test=test, runner=run_model_fn, RESULT_NAME=RESULT_NAME,
                runner_ctx={'n_jobs': _effective_n_jobs, 'worker_init': worker_init,
                            'worker_init_args': worker_init_args,
                            'cache': (_library_cache if _library_cache is not None else _cv_cache),
                            'counter': _fit_counter, 'keep_test': _dump_on},
                want_stub=_stub_active, want_engine=_engine_active,
                stub_min=_stub_parity_min, engine_min=_engine_parity_min)
            _fit_flags['stub_models'].update(_parity['stub_models'])
            _fit_flags['engine_models'].update(_parity['engine_models'])
            _t_seed = dict(_parity['t_seed'])
        except Exception as exc:
            _fit_flags['stub_models'].clear()
            _fit_flags['engine_models'].clear()
            print(f"[GP] NOTE: Joint Bayesian optimisation parity guards failed "
                  f"({type(exc).__name__}: {exc}) - the test stub and the exact engine are OFF "
                  f"for this task; every search fit is a normal fit.")
    _eta_plan = {'fits_by_model': dict(_fit_estimate_full.get('search_by_model') or {}),
                 'rounds_by_model': dict(_fit_estimate_full.get('rounds_by_model') or {})}
    _block_stats = {}
    _block_settings = None
    _lib_after = None        # {model: [(x_m, column), ...]} of a library strategy that succeeded
    _lib_cur_after = None    # {model: index} the strategy's own greedy selection

    # ver4-28, R3: block-coordinate search REPLACES the monolithic search
    # below entirely when eligible and successful - `best_x` stays a
    # concrete array (never `None`) on success, so both the `use_batch`
    # gate and the `if best_x is None:` serial fallback below skip
    # themselves automatically; a failed/ineligible block search leaves
    # `best_x` at `None` and falls through to the monolithic path exactly
    # as if `search_strategy` were left at its 'monolithic' default.
    if _library_strategy and total_dims > 0 and len(group_models) > 1:
        _strategy_label = {'block_coordinate': 'block-coordinate',
                           'parallel_library': 'parallel-library',
                           'async_library': 'async-library',
                           'weight_first': 'weight-first',
                           'weight_first_cv': 'weight-first-cv',
                           'neural_stack': 'neural-stack'}[_search_strategy]
        # ver4-34, R2.b: under the CV objective the search shares `_cv_cache` with the
        # final out-of-fold fits (see `_runner_ctx()`).
        _block_cache = _library_cache
        _block_settings = {
            'objective_mode': objective_mode, 'diversity_penalty': _diversity_penalty,
            'diversity_method': _diversity_method, 'weight_cv_alpha': _weight_cv_alpha,
            'inner_split_seed_base': inner_split_seed_base, 'block_cycles': _block_cycles,
            'block_init_points': _block_init_points, 'block_n_iter': _block_n_iter,
            'block_refine_iter': _block_refine_iter, 'block_tol': _block_tol,
            'fit_counter': _fit_counter, 'fit_stage': 'search',
            # ver4-32
            'weight_rule': _score_weight_rule, 'shrinkage_alpha': _shrinkage_alpha,
            'search_patience': _search_patience, 'warm_start': _warm_start,
            'block_warm_n_iter': _block_warm_n_iter,
            # ver4-34
            'objective_data': _objective_data, 'cv_folds': _cv_folds,
            'weight_estimator': _weight_estimator, 'estimator_settings': _estimator_settings,
            'freeze_low_weight': _freeze_low_weight, 'weight_bags': _weight_bags,
            'cluster_corr': _cluster_corr, 'time_budget_s': _time_budget_s,
            'time_budget_t0': t0, 'library_rounds': _library_rounds, 'library_q': _library_q,
            # ver4-36
            'noise_aware_surrogate': _noise_aware,
            'library_evals_per_dim': _library_evals_per_dim,
            'library_max_evals': _library_max_evals,
            'library_refine_rounds': _library_refine_rounds,
            'fit_flags': _fit_flags, 'test_sink': _test_sink,
            'eta_plan': _eta_plan, 't_seed': _t_seed,
            # ver4-38
            'cv_racing': _racing_active, 'racing_folds': _racing_folds,
            'racing_margin': _racing_margin, 'racing_penalty': _racing_penalty,
            'library_stop_rule': ('value' if _stop_rule_active else 'off'),
            'dead_column_guard': _dead_column_guard,
            # ver4-42 (read only by the weight-first-cv search)
            'wf_cv_weight_fidelity': _wfcv_fidelity, 'wf_cv_direction': _wfcv_direction,
            'wf_cv_alphas': _wfcv_alphas, 'wf_cv_alpha_rule': _wfcv_alpha_rule,
            'wf_cv_repeats': _wfcv_repeats, 'wf_cv_tune': _wfcv_tune,
            'wf_cv_accept': _wfcv_accept,
            # ver4-44 (R1 / R3): `safety_nets` is read by the weight-first-cv search (N8, N9);
            # `fallback_events` is the task's shared list of recorded class C events.
            'safety_nets': _safety_nets_on, 'wf_cv_final': _wfcv_final,
            'wf_cv_reweight': _wfcv_reweight,
            'fallback_events': _fallback_events,
            # ver4-40 (read only by the weight-first search)
            'wf_rules': _wf_rules, 'wf_max_candidates': _wf_max, 'wf_dedup_tol': _wf_dedup,
            'wf_rungs': _wf_rungs, 'wf_eta': _wf_eta, 'wf_min_tune_weight': _wf_min_w,
            'wf_include_untuned': _wf_include_untuned, 'wf_pick': _wf_pick,
            'wf_candidate_rows': _wf_cand_rows, 'oof_folds': _oof_folds,
            'bbc_bootstraps': _bbc_bootstraps,
            'es_settings': {'n_bags': _score_bags, 'steps': _es_steps,
                            'model_fraction': _es_model_fraction, 'cluster_corr': _cluster_corr},
            # ver4-48 (read only by the neural-stack search)
            **_ns, 'library_members_max': _members_max,
        }
        if _search_strategy == 'neural_stack':
            # the dimensions the library's Latin hypercube may vary: run-length fields are
            # frozen (they stay at the configured coordinate in every entry) and the
            # `search_space` filter is respected
            _ns_varied_specs, _unused_nsv = build_joint_param_specs(
                group_models, hparam_specs, base_params_by_model, freeze_fidelity=True,
                priority_only=_priority_only)
            _block_settings['ns_varied_specs'] = _ns_varied_specs
        try:
            if _search_strategy in ('parallel_library', 'async_library', 'weight_first',
                                    'weight_first_cv', 'neural_stack'):
                _lib_ctx = _make_column_context(
                    group_models=group_models, specs_by_model=specs_by_model,
                    base_params_by_model=base_params_by_model, train=train, valid=valid,
                    test=test, runner=run_model_fn, RESULT_NAME=RESULT_NAME,
                    runner_ctx={'n_jobs': _effective_n_jobs, 'worker_init': worker_init,
                                'worker_init_args': worker_init_args, 'cache': _block_cache,
                                'counter': _fit_counter, 'stage': 'search',
                                'fit_flags': _fit_flags, 'test_sink': _test_sink},
                    objective_data=_objective_data, cv_folds=_cv_folds,
                    inner_split_seed_base=inner_split_seed_base)
                if _search_strategy == 'neural_stack':
                    best_x, best_score, _block_stats = _neural_stack_search(
                        group_models=group_models, specs_by_model=specs_by_model,
                        base_params_by_model=base_params_by_model, seed_point=seed_point,
                        fit_ctx=_lib_ctx, settings=_block_settings, state=state, seed=seed)
                elif _search_strategy == 'weight_first_cv':
                    best_x, best_score, _block_stats = _weight_first_cv_search(
                        group_models=group_models, specs_by_model=specs_by_model,
                        base_params_by_model=base_params_by_model, seed_point=seed_point,
                        fit_ctx=_lib_ctx, settings=_block_settings, state=state, seed=seed)
                elif _search_strategy == 'weight_first':
                    best_x, best_score, _block_stats = _weight_first_search(
                        group_models=group_models, specs_by_model=specs_by_model,
                        base_params_by_model=base_params_by_model, seed_point=seed_point,
                        fit_ctx=_lib_ctx, settings=_block_settings, state=state, seed=seed)
                    if _block_stats.get('fallback_async'):
                        # < 2 distinct weight vectors: there is nothing to fix first, so the
                        # closest strategy runs (its provenance reports what actually ran)
                        _fallback_events.append('N1:weight_first->async_library')
                        _search_strategy, _strategy_label = 'async_library', 'async-library'
                        best_x, best_score, _block_stats = _async_library_search(
                            group_models=group_models, specs_by_model=specs_by_model,
                            base_params_by_model=base_params_by_model, seed_point=seed_point,
                            fit_ctx=_lib_ctx,
                            score_fn=_ColumnScorer(group_models, _lib_ctx, _block_settings, state),
                            settings=_block_settings, state=state, seed=seed)
                else:
                    _lib_search_fn = (_parallel_library_search
                                      if _search_strategy == 'parallel_library'
                                      else _async_library_search)
                    best_x, best_score, _block_stats = _lib_search_fn(
                        group_models=group_models, specs_by_model=specs_by_model,
                        base_params_by_model=base_params_by_model, seed_point=seed_point,
                        fit_ctx=_lib_ctx,
                        score_fn=_ColumnScorer(group_models, _lib_ctx, _block_settings, state),
                        settings=_block_settings, state=state, seed=seed)
            else:
                best_x, best_score, _block_stats = _block_coordinate_search(
                    group_models=group_models, specs_by_model=specs_by_model,
                    base_params_by_model=base_params_by_model, seed_point=seed_point,
                    train=train, valid=valid, test=test, runner=run_model_fn,
                    RESULT_NAME=RESULT_NAME, settings=_block_settings, state=state,
                    cache=_block_cache, n_jobs=_effective_n_jobs, worker_init=worker_init,
                    worker_init_args=worker_init_args, seed=seed,
                )
            _fit_counter['cache_hits'] = _fit_counter.get('cache_hits', 0) + _block_cache.hits
            _warm_x = _block_stats.get('warm_x')
            print(f"[GP] Joint Bayesian optimisation {_strategy_label} search finished: "
                  f"{ {k: v for k, v in _block_stats.items() if k not in _STATS_NOT_PRINTED} } "
                  f"(cache hits={_block_cache.hits}).")
            if not np.isfinite(best_score):
                if not _safety_nets_on:
                    raise _PureModeError(
                        f"[joint_tuning] safety_nets=False: the {_strategy_label} search could "
                        f"not fit the initial (untuned) column of every model for this task, and "
                        f"falling back to the monolithic search is switched off - see the "
                        f"[GP] NOTE above for which model.")
                print(f"[GP] NOTE: {_strategy_label} search's own initial column fit failed for "
                      f"this task - falling back to the monolithic search.")
                best_x, best_score = None, None
                _warm_x = None
            else:
                _lib_after = _block_stats.get('lib')
                _lib_cur_after = _block_stats.get('lib_cur')
        except Exception as exc:
            if not _safety_nets_on:
                # ver4-44, R1 (N1): no silent change of strategy; the task fails with the cause.
                if isinstance(exc, _PureModeError):
                    raise
                raise _PureModeError(
                    f"[joint_tuning] safety_nets=False: the {_strategy_label} search failed "
                    f"({type(exc).__name__}: {exc}) and falling back to the monolithic search is "
                    f"switched off.") from exc
            print(f"[GP] NOTE: Joint Bayesian optimisation {_strategy_label} search failed "
                  f"({type(exc).__name__}: {exc}) - falling back to the monolithic search for this "
                  f"task. This search's own result is unaffected, only which strategy found it.")
            best_x, best_score = None, None
            _warm_x = None

    # ver4-40, R2: the weight-first search ran and left a (weight vector, columns) choice. Every
    # later stage that would re-estimate weights or gate the result is replaced / skipped.
    _wf_stats = (_block_stats or {}).get('wf') if (_search_strategy in _WF_STRATEGIES) else None
    _wf_active = bool(_wf_stats) and best_x is not None and bool(_lib_after)
    if not _safety_nets_on and _search_strategy in _WF_STRATEGIES and not _wf_active:
        raise _PureModeError(
            f"[joint_tuning] safety_nets=False: search_strategy={_search_strategy!r} left no "
            f"fixed weight vector for this task, and falling back to the monolithic search is "
            f"switched off.")

    # ver4-38, R2.b1: models whose UNTUNED column was constant and that the library search took
    # out of the scored group. They keep their configured hyperparameters (their block was never
    # searched) and receive weight 0 in the final weights below.
    _dead_models = set(_block_stats.get('dead_models') or []) if _block_stats else set()

    q = min(max(2, int(bayes_batch_max or _JOINT_BATCH_Q)), _JOINT_BATCH_Q)
    use_batch = (best_x is None and bool(bayes_batch) and total_dims > 0 and q > 1
                 and BayesianOptimization is not None)

    # FEATURE (Round 7): 'multi_fidelity' only has a mechanism to attach to
    # on the batch path (see this function's own docstring paragraph for
    # why) - a batch-less task gets an explicit NOTE rather than silently
    # ignoring the setting.
    if _multi_fidelity and not use_batch:
        print(f"[GP] NOTE: Joint Bayesian optimisation 'multi_fidelity' was requested but has no "
              f"effect for this task - it only applies to the batch search path (bayes_batch=True "
              f"with a >0-dimensional search and q>1), which isn't in use here (bayes_batch="
              f"{bool(bayes_batch)}, total_dims={total_dims}, q={q}). Every candidate is evaluated "
              f"at the normal search-phase fidelity, exactly as if 'multi_fidelity' were off.")

    # ROUND 10: every candidate that received a genuine FULL-training-rows
    # search-phase evaluation, keyed by its coordinates. `batch_bayesian_
    # maximise()` reports the arg-max over EVERY score `objective_batch()`
    # returned - with multi-fidelity screening on, that included the noisy,
    # reduced-training-rows screening scores of candidates that were never
    # promoted, so the "winner" could be a candidate that was only ever
    # screened, on a score that is optimistic by construction (a maximum over
    # many noisy draws). The winner is therefore re-selected below from this
    # record only.
    _full_fidelity = {}

    def _evaluate_recorded(x):
        _sc = _evaluate(x, search_fidelity=True)
        _full_fidelity[np.asarray(x, dtype=float).tobytes()] = (np.array(x, dtype=float), _sc)
        return _sc

    if use_batch:
        if _multi_fidelity:
            # Round 7: successive-halving-style screening rung layered onto
            # the existing constant-liar BATCH suggestion loop, without any
            # change to models/parallel_search.py::batch_bayesian_maximise()
            # itself - that function's own contract
            # (`objective_batch(xs) -> len(xs) scores, one per xs[i], in
            # order`) is honoured exactly the same way whether this branch
            # runs or not; only HOW each returned score was obtained
            # differs. `_mf_round` is a one-element mutable list (not a
            # plain int) purely so this closure can increment it across
            # calls without a `nonlocal` declaration - it exists only to
            # vary the screening subsample's own seed from round to round,
            # so every round doesn't happen to draw the identical subset of
            # training rows.
            _mf_round = [0]

            def objective_batch(xs):
                # A round of exactly one candidate (the last, possibly
                # partial round when `remaining < q` - see
                # batch_bayesian_maximise()'s own docstring) has nothing to
                # rank a candidate AGAINST within the round, so screening
                # buys nothing - falls back to the ordinary evaluation,
                # exactly as if multi_fidelity were off, for that round only.
                if len(xs) <= 1:
                    return [_evaluate_recorded(x) for x in xs]

                _mf_round[0] += 1
                screen_train = _screen_subsample_train(
                    train, _JOINT_MULTI_FIDELITY_TRAIN_FRACTION, _JOINT_MULTI_FIDELITY_MIN_ROWS,
                    seed=seed + 10_000 * _mf_round[0],
                )
                if screen_train is None:
                    # This task's own training split is too small to
                    # subsample meaningfully (see
                    # _screen_subsample_train()'s own docstring) - screening
                    # is skipped for THIS TASK only; every candidate gets
                    # the ordinary evaluation, exactly as if multi_fidelity
                    # were off.
                    return [_evaluate_recorded(x) for x in xs]

                def _screen_score(x):
                    # SAME scoring pipeline, SAME real validation split,
                    # only `train` differs (the small subsample above) -
                    # never a different metric or a fabricated placeholder,
                    # so a candidate that ends up NOT promoted below still
                    # gets returned a real, honest (if noisier) evaluation
                    # of itself, exactly matching objective_batch()'s own
                    # contract.
                    return joint_objective(
                        x, group_models=group_models, base_params_by_model=base_params_by_model,
                        specs_by_model=specs_by_model, train=screen_train, valid=valid, test=test,
                        run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME,
                        objective_mode=objective_mode, search_fidelity=True,
                        diversity_penalty=_diversity_penalty, diversity_method=_diversity_method,
                        valid_repeats=_valid_repeats, inner_split_seed_base=inner_split_seed_base,
                        state=state, weight_cv_alpha=_weight_cv_alpha, runner_ctx=_runner_ctx('search'),
                        weight_rule=_score_weight_rule, shrinkage_alpha=_shrinkage_alpha,
                        **_v434_kwargs,
                    )

                screen_scores = [_screen_score(x) for x in xs]
                n_keep = max(1, math.ceil(len(xs) * _JOINT_MULTI_FIDELITY_KEEP_FRACTION))
                order = sorted(range(len(xs)), key=lambda i: screen_scores[i], reverse=True)
                promoted = set(order[:n_keep])

                final_scores = list(screen_scores)
                _promoted_full = []
                for i in promoted:
                    # Only the promoted candidates pay for the normal,
                    # full-training-rows search-phase evaluation - the SAME
                    # evaluation every candidate in the batch got before
                    # this feature existed. A discarded candidate's score
                    # is left at its screening value (never re-evaluated,
                    # never discarded from the returned list) - this is the
                    # entire wall-clock saving this feature buys.
                    final_scores[i] = _evaluate_recorded(xs[i])
                    _promoted_full.append(final_scores[i])
                # ROUND 10: successive halving's own semantics are that the
                # eliminated candidates are known to be WORSE than the
                # survivors. The raw screening score is on a different
                # footing from a full-fidelity one (fewer training rows, and -
                # with valid_repeats>1 - scored on far fewer inner-validation
                # rows), so registering it as-is could hand the surrogate a
                # value ABOVE every genuinely evaluated candidate. Cap it at
                # the worst promoted (full-fidelity) score of this round:
                # rank-consistent, never optimistic.
                if _promoted_full:
                    _cap = min(_promoted_full)
                    for i in range(len(xs)):
                        if i not in promoted:
                            final_scores[i] = min(final_scores[i], _cap)
                return final_scores
        else:
            def objective_batch(xs):
                return [_evaluate_recorded(x) for x in xs]
        try:
            best_x, best_score, _trials = batch_bayesian_maximise(
                objective_batch, [(0.0, 1.0)] * total_dims,
                n_iter=n_iter, init_points=init_points, seed=seed, q=q,
                probe_points=[seed_point],
                acquisition_factory=_make_acquisition,
                bounds_transformer_factory=_make_bounds_transformer,
                liar=bayes_liar,
                should_stop=_compose_stops(_patience_stop(_search_patience),
                                           _time_budget_stop(t0, _time_budget_s)),
                # ver4-36 R2.a: noise-aware surrogate + posterior-mean incumbent (defaults
                # = ver4-35: interpolating Matern surrogate, max observed score)
                noise_aware=_noise_aware,
                incumbent='posterior_mean' if _noise_aware else 'max',
            )
            if _multi_fidelity and _full_fidelity:
                # ROUND 10: winner = best genuinely full-fidelity evaluation
                # (see `_full_fidelity` above). Insertion order + strict `>`
                # reproduces batch_bayesian_maximise()'s own tie-breaking.
                _bx, _bs = None, -math.inf
                if _noise_aware and len(_full_fidelity) > 1:
                    # ver4-36 R2.a: the posterior-mean incumbent, restricted to the full-
                    # fidelity points (the screening scores are on another footing); the
                    # reported score stays the observed one.
                    _ff = list(_full_fidelity.values())
                    _i_rob = robust_incumbent([f[0] for f in _ff], [f[1] for f in _ff],
                                              seed=int(seed))
                    _bx, _bs = _ff[_i_rob]
                else:
                  for _x, _sc in _full_fidelity.values():
                    if _sc > _bs:
                        _bx, _bs = _x, _sc
                if _bx is not None:
                    best_x, best_score = _bx, _bs
        except Exception as exc:
            print(f"[GP] NOTE: batch joint hyperparameter+weight search failed "
                  f"({type(exc).__name__}: {exc}) - falling back to a serial joint search for "
                  f"this task. This search's own result is unaffected, only wall-clock time.")
            best_x, best_score = None, None

    if best_x is None:
        if _search_patience > 0 and total_dims > 0:
            print(f"[GP] NOTE: Joint Bayesian optimisation search_patience={_search_patience} does "
                  f"not apply to the serial bayes_opt search used for this task (the batch path is "
                  f"not in use or failed) - the full evaluation budget is used.")
        if _time_budget_s > 0 and total_dims > 0:
            print(f"[GP] NOTE: Joint Bayesian optimisation time_budget_s={_time_budget_s} does "
                  f"not apply to the serial bayes_opt search used for this task (the batch path "
                  f"is not in use or failed) - the full evaluation budget is used.")
        if total_dims == 0:
            best_x, best_score = np.array([]), _evaluate(np.array([]), search_fidelity=True)
        else:
            if BayesianOptimization is None:
                raise ImportError(
                    "bayes_opt is required for Joint Bayesian optimisation (already a project "
                    "dependency via models/Bayesian_optimisation.py)."
                )
            names = [f'x{i}' for i in range(total_dims)]
            pbounds_dict = {n: (0.0, 1.0) for n in names}

            def wrapped(**kwargs):
                x = np.array([kwargs[n] for n in names])
                return _evaluate(x, search_fidelity=True)

            optimizer = BayesianOptimization(
                wrapped, pbounds_dict, random_state=seed, allow_duplicate_points=True,
                acquisition_function=_make_acquisition(), bounds_transformer=_make_bounds_transformer(),
            )
            try:
                optimizer._gp.set_params(alpha=1e-3)
            except Exception:
                pass
            if _noise_aware:
                # ver4-36 R2.a: the same noise-aware surrogate on the serial path
                try:
                    _install_noise_aware_surrogate(optimizer)
                except Exception as exc:
                    print(f"[GP] NOTE: Joint Bayesian optimisation noise_aware_surrogate could not "
                          f"be installed on the serial surrogate ({type(exc).__name__}: {exc}) - "
                          f"the default surrogate is used.")
            try:
                optimizer.probe(params=dict(zip(names, seed_point)), lazy=True)
            except Exception:
                pass
            optimizer.maximize(init_points=init_points, n_iter=n_iter)
            best_x = np.array([optimizer.max['params'][n] for n in names])
            best_score = optimizer.max['target']
            if _noise_aware:
                # ver4-36 R2.a: posterior-mean incumbent over the observed points
                try:
                    _res = list(optimizer.res)
                    _xs = [np.array([r['params'][n] for n in names]) for r in _res]
                    _ys = [float(r['target']) for r in _res]
                    _i_rob = robust_incumbent(_xs, _ys, seed=int(seed))
                    best_x, best_score = _xs[_i_rob], _ys[_i_rob]
                except Exception as exc:
                    print(f"[GP] NOTE: Joint Bayesian optimisation noise_aware_surrogate could not "
                          f"form the posterior-mean incumbent ({type(exc).__name__}: {exc}) - the "
                          f"max observed score is used.")

    # BUGFIX (joint-tuned models underperforming their own untuned
    # defaults): `selection_margin` was accepted as a parameter but never
    # actually consulted anywhere in this function - unlike
    # `models/hyperparameter_tuning.py::tune_model_hyperparameters()`'s
    # own, well-established "default-vs-winner" safeguard (see that
    # function's own docstring/comments for the full rationale, which
    # applies here verbatim), this joint search had NO comparison at all
    # between the search's own winning candidate and simply leaving every
    # group model at its own already-configured hyperparameters with
    # equal ensemble weighting. `seed_point` (this search's own probe
    # point, encoding exactly that baseline) was evaluated as one
    # candidate among many during the search, but never singled out as a
    # floor the winner must clear - so a candidate that merely overfit
    # this task's own (often small) validation split via sampling noise
    # was accepted unconditionally and refit into every group model's own
    # official Metric.csv/Marker_effect.csv/etc. row, even when it did not
    # genuinely outperform not tuning at all. This is the exact same
    # structural failure mode the single-model fix above already
    # diagnoses and fixes for rrBLUP/GBLUP/BayesB/RKHS/RF/ExtraTrees/SVR/
    # KNN - just never ported to this newer, joint-search module. Fixed
    # here the same way: re-score the untuned baseline through the exact
    # same search-phase objective every candidate was scored through, and
    # require the search's own winner to beat it by more than
    # `selection_margin` (in objective-score units) or fall back to the
    # baseline (every group model's own configured hyperparameters, equal
    # ensemble weighting) entirely. `selection_margin=0.0` reproduces the
    # bare `>` comparison; a `None`/falsy value is treated as `0.0` rather
    # than disabling the check, so this safeguard is never silently off.
    margin = max(0.0, float(selection_margin) if selection_margin is not None else 0.0)
    _fresh_done = False

    # ver4-36, R1.b / R2.b: the bootstrap-bias-corrected gate and / or selection over the
    # fitted library (ZERO fits). It replaces the fresh-resample gate when gate_mode='bbc' and
    # re-picks the library combination when selection_rule='bbc'. Any problem (library missing,
    # too few rows or draws) is a NOTE and the ver4-35 path (greedy selection, fresh gate)
    # runs instead - never a task failure.
    _bbc_summary = None
    if (best_x is not None and total_dims > 0 and _lib_after and _lib_cur_after
            and not _wf_active and (_gate_mode == 'bbc' or _selection_rule == 'bbc')):
        try:
            _bbc_ctx = _make_column_context(
                group_models=group_models, specs_by_model=specs_by_model,
                base_params_by_model=base_params_by_model, train=train, valid=valid, test=test,
                runner=run_model_fn, RESULT_NAME=RESULT_NAME, runner_ctx=None,
                objective_data=_objective_data, cv_folds=_cv_folds,
                inner_split_seed_base=inner_split_seed_base)
            _bbc_scorer = _ColumnScorer(group_models, _bbc_ctx, _block_settings, state)
            _bbc_scorer.inactive = set(_dead_models)     # ver4-38: same scored group as the search
            _bbc_models = [m for m in group_models if m not in _dead_models]

            def _x_of(_pick):
                return np.concatenate([np.asarray(_lib_after[m][_pick[m]][0], dtype=float)
                                       for m in group_models])

            def _cols_of(_pick):
                return {m: _lib_after[m][_pick[m]][1] for m in group_models}

            _c_full = {m: int(_lib_cur_after[m]) for m in group_models}
            if not np.array_equal(_x_of(_c_full), np.asarray(best_x, dtype=float)):
                raise ValueError("the recorded library does not contain the search's final point")
            _bbc_margin = max(0.0, float(selection_margin) if selection_margin is not None else 0.0)
            _bbc = _bbc_analyse(
                _lib_after, _bbc_models, _bbc_ctx['actual'],
                c_full={m: _c_full[m] for m in _bbc_models},
                n_boot=_bbc_bootstraps,
                seed=int(seed) + int(inner_split_seed_base) + _JOINT_BBC_SEED_OFFSET,
                shrink_alpha=(_shrinkage_alpha if _score_weight_rule == 'shrunk' else None),
                tol=_block_tol, quantile=_bbc_gate_quantile, margin=_bbc_margin,
                dead_guard=_dead_column_guard, partition=_bbc_ctx.get('partition'))
            if not _bbc['ok']:
                raise ValueError(_bbc['reason'])
            _zero_pick = {m: 0 for m in group_models}
            _pick = ({**_c_full, **_bbc['selected']} if _selection_rule == 'bbc'
                     else dict(_c_full))
            _seed_cols = _cols_of(_zero_pick)
            _seed_ens_bbc = float(_bbc_scorer.score(_seed_cols))
            _mo = _bbc['mean_oob']
            _bbc_summary = {
                'gate_mode': _gate_mode, 'selection_rule': _selection_rule,
                'n_valid': int(_bbc['n_valid']), 'gate_quantile': float(_bbc['gate_quantile']),
                'accepted': bool(_bbc['accepted']), 'selected_label': _bbc['selected_label'],
                'mean_oob': {k: float(v) for k, v in _mo.items()},
                'gap_mean': float(np.mean(_bbc['gaps'])), 'margin': _bbc_margin,
            }
            if _selection_rule == 'bbc':
                _changed = _pick != _c_full
                print(f"[GP] Joint Bayesian optimisation BBC selection: picked the "
                      f"{_bbc['selected_label']!r} library combination by the highest mean "
                      f"out-of-bag score over {_bbc['n_valid']} bootstrap draws (mean OOB: "
                      f"greedy={_mo.get('full', math.nan):.4f}, untuned="
                      f"{_mo.get('untuned', math.nan):.4f}, selected="
                      f"{_mo.get('selected', math.nan):.4f}) - "
                      + ("it differs from the greedy selection." if _changed
                         else "the same as the greedy selection."))
                if _changed:
                    best_x = _x_of(_pick)
                    best_score = float(_bbc_scorer.score(_cols_of(_pick)))
            if _gate_mode == 'bbc':
                _fresh_done = True
                _gq = _bbc['gate_quantile']
                _gdesc = (f"the {_bbc_gate_quantile:g} quantile of the out-of-bag gap (tuned - "
                          f"untuned) over {_bbc['n_valid']} bootstrap draws is {_gq:+.4f} "
                          f"(mean gap {_bbc_summary['gap_mean']:+.4f}; required > "
                          f"{_bbc_margin:.4f})")
                if _bbc['accepted']:
                    if _pick == _zero_pick:
                        best_x, best_score = seed_point, _seed_ens_bbc
                        _gate_decision = 'seed_won'
                        print(f"[GP] Joint Bayesian optimisation BBC gate: {_gdesc} - accepted, "
                              f"but the selected combination is the untuned one, so every group "
                              f"model keeps its own configured hyperparameters.")
                    else:
                        _gate_decision = 'bbc_accepted'
                        print(f"[GP] Joint Bayesian optimisation BBC gate: {_gdesc} - accepted "
                              f"as a genuine improvement.")
                else:
                    best_x, best_score = seed_point, _seed_ens_bbc
                    _gate_decision = 'bbc_fallback'
                    print(f"[GP] Joint Bayesian optimisation BBC gate: {_gdesc} - the search's "
                          f"advantage did not survive on out-of-bag rows, so this is treated as "
                          f"selection noise: falling back to every group model's own configured "
                          f"hyperparameters for this task.")
        except Exception as exc:
            _fresh_done = False
            print(f"[GP] NOTE: Joint Bayesian optimisation BBC gate / selection could not be "
                  f"formed ({type(exc).__name__}: {exc}) - the greedy selection and the "
                  f"fresh-resample gate are used for this task.")


    # ver4-40, R2: the weight-first pick (`_wf_bbc`, zero fits) IS the gate: an untuned finalist
    # winning means 'no tuning', a tuned one means the searched columns survived on out-of-bag
    # rows. Neither the fresh-resample gate nor the 1-SE margin runs (they would re-estimate
    # weights the strategy deliberately fixes).
    if _wf_active:
        _fresh_done = True
        _wf_untuned = (_wf_stats['arm'] == 'untuned' or np.array_equal(best_x, seed_point))
        _gate_decision = 'wf_untuned' if _wf_untuned else 'wf_tuned'
        if (_search_strategy == 'weight_first_cv' and _wf_untuned
                and float(_wf_stats.get('alpha', 1.0)) == 0.0):
            _gate_decision = 'wfcv_uniform'      # ver4-42: alpha* = 0 and untuned = the naive row
        if _search_strategy == 'neural_stack':
            # ver4-48: lambda* = infinity IS the untuned naive ensemble ('ns_prior'); anything
            # else is the fitted network ('ns_fitted'). No other gate runs; with the nets off
            # the lambda = infinity candidate was never in the path, so only 'ns_fitted' occurs.
            _gate_decision = ('ns_prior' if not math.isfinite(float(_block_stats['ns']['lambda']))
                              else 'ns_fitted')
            # ver4-51: the nested outer gate replaced a fitted network by the equal-weight ensemble
            _ns_outer = _block_stats['ns'].get('outer') or {}
            if _ns_outer.get('replaced'):
                _gate_decision = 'ns_outer_prior'
                print(f"[GP] Joint Bayesian optimisation neural-stack gate: lambda*="
                      f"{format(_block_stats['ns']['lambda_cv'], 'g')} was chosen by the "
                      f"cross-validated lambda path, but the nested outer gate (N2c, "
                      f"{_ns_outer.get('decision')}) did not show an out-of-sample gain over equal "
                      f"weights -> the untuned equal-weight ensemble (decision 'ns_outer_prior').")
            else:
                print(f"[GP] Joint Bayesian optimisation neural-stack gate: lambda*="
                      f"{'inf' if _gate_decision == 'ns_prior' else format(_block_stats['ns']['lambda'], 'g')} "
                      f"chosen by the {_wf_stats['pick_how']!r} rule over the cross-validated lambda "
                      f"path -> "
                      + ("the untuned naive ensemble (decision 'ns_prior')."
                         if _gate_decision == 'ns_prior' else
                         "the fitted network (decision 'ns_fitted')."))
        elif not _safety_nets_on:
            # ver4-44, R1: nothing was judged against the untuned columns; if the columns are
            # the untuned ones, it is because the search produced no tuned selection.
            _gate_decision = 'wf_no_tuned_columns' if _wf_untuned else 'wf_tuned'
            print(f"[GP] Joint Bayesian optimisation weight-first gate (safety_nets=False): no "
                  f"untuned-vs-tuned check; weight vector {_wf_stats['name']!r} with the "
                  + ("untuned columns, because the search produced no tuned selection "
                     "(decision 'wf_no_tuned_columns')." if _wf_untuned
                     else "tuned columns (decision 'wf_tuned')."))
        else:
            print(f"[GP] Joint Bayesian optimisation weight-first gate: the "
                  f"{_wf_stats['pick_how']!r} pick chose weight candidate {_wf_stats['name']!r} "
                  f"with the "
                  + ("UNTUNED columns - every group model keeps its own configured hyperparameters "
                     "(decision 'wf_untuned')." if _wf_untuned
                     else "TUNED columns (decision 'wf_tuned')."))
    elif not _safety_nets_on:
        # ver4-44, R1 (N2): no gate of any kind - not the BBC gate (switched off through
        # gate_mode), not the fresh-resample gate and not the legacy margin check below. The
        # search's winner is the joint result, even when it equals the untuned point.
        _fresh_done = True
        _gate_decision = 'searched_unprotected' if total_dims > 0 else 'not_run'
        print(f"[GP] Joint Bayesian optimisation gate: safety_nets=False - the search's winner is "
              f"kept without any comparison against the untuned defaults (decision "
              f"{_gate_decision!r}).")

    # ROUND 10 (joint-bayesian-optimisation-underperformance-fix) - FRESH-RESAMPLE
    # GATE. The check below ("did the search's winner genuinely beat the untuned
    # defaults?") compared `best_score` - the MAXIMUM of every score the search
    # produced, all measured on the SAME inner folds - with a re-evaluation of
    # the defaults on those same folds. Taking the arg-max of many noisy scores
    # measured on shared folds is optimistic by construction (the winner's
    # curse: the more candidates, and the noisier each score, the more likely
    # the winner just got lucky on THESE folds), and the fixed `selection_margin`
    # (or the 1-SE rule further down, whose `_best_mean` re-evaluation was
    # computed on the very same folds, returned the very same number, and was
    # then discarded) cannot correct for that. The remedy is standard: judge
    # the winner on data that played no part in choosing it. Here that is a set
    # of FRESH inner train/validation resamples (different seeds from every
    # fold the search saw), on which the winner and the untuned defaults are
    # scored on the SAME folds, so the comparison is PAIRED and its standard
    # error is the standard error of the per-fold differences. The winner is
    # accepted only if it beats the defaults by more than
    # max(selection_margin, that standard error); otherwise every group model
    # keeps its own configured hyperparameters. Costs 2 x `fresh_gate_repeats`
    # extra search-fidelity evaluations once per task (about 6% of a default
    # 50+5-candidate, 3-repeat search). Falls through to the previous
    # comparison (below) whenever the fresh evaluation cannot be formed
    # (training split too small for that many resamples, or a fit failed).
    # ver4-32 R3/R5: the gate is `_paired_fresh_gate()`. Candidates are the
    # search's final point and, with `warm_start='individual'`, the individually
    # tuned point; the legacy path (one candidate, no cache) reproduces the
    # ver4-31 gate exactly. `gate_fit_cache` additionally reuses fits whose
    # settings are identical for a candidate and the defaults.
    _gate_candidates = []
    if _fresh_done:
        pass  # ver4-36: the BBC gate already decided
    elif total_dims > 0 and _fresh_gate:
        if not np.array_equal(best_x, seed_point):
            _gate_candidates.append(('final', best_x))
        if (_warm_x is not None and not np.array_equal(_warm_x, seed_point)
                and not np.array_equal(_warm_x, best_x)):
            _gate_candidates.append(('warm', _warm_x))
    elif total_dims > 0 and _warm_x is not None and not np.array_equal(_warm_x, best_x):
        print("[GP] NOTE: Joint Bayesian optimisation warm_start='individual' produced a "
              "second candidate, but fresh_gate is off, so only the search's final point is "
              "checked (the warm point only decided where the ensemble cycles started).")
    if _gate_candidates:
        _fg_seed = inner_split_seed_base + _JOINT_FRESH_GATE_SEED_OFFSET
        _gate_cache = _FitCache() if _gate_fit_cache else None
        _fg_label, _fg_x, _fg_mean, _fg_info = _paired_fresh_gate(
            _gate_candidates, seed_point, repeats=_fresh_gate_repeats, seed=_fg_seed,
            margin=margin, score_kwargs=_gate_score_kwargs, runner_ctx=_runner_ctx('gate'),
            cache=_gate_cache)
        if _fg_info['ok']:
            _fresh_done = True
            _fg_c = _fg_info['candidates']
            if _fg_label == 'seed':
                if len(_gate_candidates) == 1:
                    _c = _fg_c['final']
                    print(f"[GP] Joint Bayesian optimisation fresh-resample gate: the search "
                          f"winner (search objective={best_score:.4f}) beat the untuned defaults "
                          f"by only "
                          f"{_c['gap']:+.4f} on {_c['n_folds']} FRESH inner resamples it was not "
                          f"selected on (required > {_c['required']:.4f} = max(selection_margin="
                          f"{margin:.4f}, paired SE={_c['se']:.4f})) - the search's advantage was "
                          f"not reproduced on new data, so this is treated as selection noise: "
                          f"falling back to every group model's own configured hyperparameters "
                          f"for this task.")
                else:
                    _txt = '; '.join(f"{k}: gap {v['gap']:+.4f} (required > {v['required']:.4f})"
                                     for k, v in _fg_c.items() if v['valid'])
                    print(f"[GP] Joint Bayesian optimisation fresh-resample gate: none of the "
                          f"{len(_gate_candidates)} candidates beat the untuned defaults on fresh "
                          f"inner resamples ({_txt}) - treated as selection noise: falling back to "
                          f"every group model's own configured hyperparameters for this task.")
                best_x, best_score = seed_point, float(_fg_mean)
                _gate_decision = 'gate_fallback'
            else:
                _c = _fg_c[_fg_label]
                print(f"[GP] Joint Bayesian optimisation fresh-resample gate: the {_fg_label!r} "
                      f"candidate beat the untuned defaults by {_c['gap']:+.4f} on {_c['n_folds']} "
                      f"fresh inner resamples (required > {_c['required']:.4f}) - accepted as a "
                      f"genuine improvement"
                      + (f" (chosen among {len(_gate_candidates)} candidates by the largest paired "
                         f"gap)." if len(_gate_candidates) > 1 else "."))
                best_x, best_score = _fg_x, float(_fg_mean)
                _gate_decision = 'searched_accepted' if _fg_label == 'final' else 'warm_accepted'
    elif total_dims > 0 and _fresh_gate and not _fresh_done:
        # The search's own winner IS the untuned baseline - nothing to gate.
        _fresh_done = True
        _gate_decision = 'seed_won'

    if not _fresh_done:
        default_score = _evaluate(seed_point, search_fidelity=True) if total_dims > 0 else best_score
        margin = max(0.0, float(selection_margin) if selection_margin is not None else 0.0)

        # BUGFIX (joint search still underperforming the naive ensemble AND
        # the non-joint, weight-only Bayesian optimisation method, even after
        # the default-vs-winner margin above): a FIXED `selection_margin`
        # (0.02, same units regardless of how noisy the winning candidate's
        # own score estimate is) does not account for the joint search's own
        # combined hyperparameter space typically being far higher-dimensional
        # (13-30+ tunable fields across every group model - Round 6 removed the
        # additional `K-1` weight dimensions this comment used to also count
        # here, but the remaining hyperparameter-only space is still large)
        # than any single model's own tunable space (7-18 fields) -
        # `models/hyperparameter_tuning.py::tune_model_
        # hyperparameters()`'s own "1-SE rule" (see that function's own
        # comment block for the full derivation, which applies here verbatim,
        # only more so) exists for exactly this reason: comparing one
        # (possibly noisy) default-baseline draw against the MAXIMUM of many
        # noisy candidate draws is a textbook winner's-curse comparison, and
        # the candidates most likely to win on noise alone are exactly the
        # ones a fixed margin under-protects against. `_score_candidate()`'s
        # own `effective_repeats>1` branch (see its own comment) already
        # computes each candidate's standard error across the
        # `valid_repeats` inner training-split resamples whenever that's
        # configured (`_valid_repeats>1`) - it was simply never propagated up
        # to this safeguard before this fix, so raising `valid_repeats` alone
        # reduced per-candidate noise but never widened the bar a noisy
        # winner had to clear. Two extra, bounded evaluations here (one for
        # `seed_point`, one for `best_x`, each `_valid_repeats` model refits)
        # - paid once, after the search has already finished, mirroring
        # `tune_model_hyperparameters()`'s own identical trade-off. Only
        # computable when `_valid_repeats>1` gave each candidate more than one
        # inner-fold score to estimate an uncertainty FROM; with
        # `_valid_repeats<=1` (the pre-ver4-27 default) `se_margin` stays 0.0
        # and this reproduces the plain `selection_margin` behaviour exactly,
        # byte for byte (I11).
        se_margin = 0.0
        if total_dims > 0 and _valid_repeats > 1:
            def _evaluate_with_se(x):
                _score, _extra = _score_candidate(
                    x, group_models=group_models, base_params_by_model=base_params_by_model,
                    specs_by_model=specs_by_model, train=train, valid=valid, test=test,
                    run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME,
                    objective_mode=objective_mode, search_fidelity=True,
                    diversity_penalty=_diversity_penalty, diversity_method=_diversity_method,
                    valid_repeats=_valid_repeats, inner_split_seed_base=inner_split_seed_base,
                    state=state, weight_cv_alpha=_weight_cv_alpha, runner_ctx=_runner_ctx('gate'),
                    weight_rule=_score_weight_rule, shrinkage_alpha=_shrinkage_alpha,
                    **_v434_kwargs,
                )
                return _score, (_extra['se'] if isinstance(_extra, dict) else 0.0)

            _default_mean, se_default = _evaluate_with_se(seed_point)
            _best_mean, se_best = _evaluate_with_se(best_x)
            # Standard error of an independent difference of two means.
            se_margin = math.sqrt(se_default ** 2 + se_best ** 2)
            if se_margin > margin:
                print(f"[GP] NOTE: Joint Bayesian optimisation 1-SE rule raised the required "
                      f"winning margin from {margin:.4f} to {se_margin:.4f} (se_default="
                      f"{se_default:.4f}, se_best={se_best:.4f}, valid_repeats={_valid_repeats}) - "
                      f"the search's own winning candidate carries enough score uncertainty across "
                      f"its inner resamples that it must beat this task's own untuned defaults by "
                      f"more than that to be trusted as a genuine improvement rather than sampling "
                      f"noise, mirroring models/hyperparameter_tuning.py's own 1-SE rule for "
                      f"single-model HP_TUNE.")
        margin = max(margin, se_margin)

        if total_dims > 0 and default_score >= best_score - margin:
            print(f"[GP] Joint Bayesian optimisation: search winner (valid objective="
                  f"{best_score:.4f}) did not beat this task's own untuned defaults (valid "
                  f"objective={default_score:.4f}) by the required margin ({margin:.4f}) - "
                  f"falling back to every group model's own configured hyperparameters and "
                  f"equal ensemble weighting for this task, the same 'no worse than not "
                  f"tuning at all' guarantee models/hyperparameter_tuning.py's own "
                  f"single-model HP_TUNE already provides.")
            best_x, best_score = seed_point, default_score
            _gate_decision = 'legacy_margin_fallback'
        elif total_dims > 0:
            _gate_decision = 'searched_accepted'
        else:
            _gate_decision = 'not_run'

    # ONE final confirmatory evaluation at the winning candidate, full
    # fidelity (search_fidelity=False) - R3 lever 2, and the fit whose
    # OWN output becomes final_params_by_model/tuned_values_by_model
    # returned here, and whose OWN (pred_matrix, actual_valid) is reused
    # below by the weight-collapse floor (never a second confirmatory
    # fit just to grade weights).
    # ver4-28, R2: `keep_results=True` (opt-in, `reuse_confirmatory_fit`)
    # asks this call to keep every group model's FULL runner dict, not
    # just `predicted_valid` - see `payload['confirm_results']` below and
    # the `details['confirm_results']` this function returns.
    # ver4-48: the neural stack's entry 0 is the model's CONFIGURED parameters, fitted exactly
    # (never `_decode(_encode(base))`), so a model whose primary is entry 0 is confirmed with
    # exactly those parameters (`use_base_params_for`; None = every other strategy, unchanged).
    _ns_base_models = None
    if _wf_active and _search_strategy == 'neural_stack':
        _ns_base_models = {m for m in group_models if int(_lib_cur_after.get(m, 0)) == 0}
    _confirm_score, payload = _score_candidate(
        best_x, group_models=group_models, base_params_by_model=base_params_by_model,
        specs_by_model=specs_by_model, train=train, valid=valid, test=test,
        run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME,
        objective_mode=objective_mode, search_fidelity=False,
        diversity_penalty=_diversity_penalty, diversity_method=_diversity_method,
        state=state, weight_cv_alpha=_weight_cv_alpha,
        use_base_params=bool(total_dims > 0 and np.array_equal(best_x, seed_point)),
        use_base_params_for=_ns_base_models,
        keep_results=_reuse_confirmatory_fit, runner_ctx=_runner_ctx('confirm'),
        weight_rule=_score_weight_rule, shrinkage_alpha=_shrinkage_alpha, **_v434_kwargs,
    )
    if payload is None:
        raise RuntimeError(
            f"Joint Bayesian optimisation: the confirmatory (full-fidelity) evaluation of this "
            f"search's own winning candidate failed for at least one of {list(group_models)!r} "
            f"- see the [joint_tuning] NOTE above for which model and why. This search cannot "
            f"produce a result for this task; the task will fail (see genomic_prediction.py's "
            f"own per-task rollback - no partial joint-mode rows are left behind)."
        )

    # BUGFIX (weight-collapse report): grade the search's own winning
    # weight vector - and, optionally, the closed-form analytic
    # least-squares solution - against plain equal weighting on the
    # REALISED validation MSE of this confirmatory, full-fidelity fit
    # (ensemble_regularization.select_weights_with_floor(), the SAME
    # mechanism Bayesian_optimisation.py's own W_OPT_VALIDATION_FLOOR
    # already uses), then apply naive shrinkage as a residual safety
    # margin. This makes "the joint search's own weights are never worse
    # than a plain average of the same group models, on validation MSE"
    # a construction-level guarantee, replacing whatever near-one-hot
    # vertex the raw search happened to land on with a graded, genuinely
    # diversity-aware choice whenever equal (or analytic) weighting
    # actually scores better.
    _dump_path = None

    _pred_matrix = payload['pred_matrix']
    _actual_valid = payload['actual_valid']
    _search_shares = np.array([payload['weights'][m] for m in group_models], dtype=float)

    def _grade_weights_core(pred_matrix, actual_valid, search_shares):
        """Final ensemble weights for one confirmatory fit's validation
        predictions. Returns `(shares, label, diagnostics)`.

        ROUND 10 (`weight_cv`, default OFF - opt-in): the weights are chosen by
        CROSS-VALIDATING the shrinkage level between equal weighting and the
        analytic (MSE-optimal) solution over the validation rows
        (`ensemble_regularization.cv_shrunk_simplex_weights`), so the choice
        reflects how the weights generalise to rows they were not fitted on.
        The previous grading (`select_weights_with_floor`, still used when
        `weight_cv` is off) scored every candidate on the SAME rows the
        analytic solution had just been fitted to - a comparison the analytic
        solution wins by construction - so its "never worse than equal
        weighting" floor could never fire, and the only protection left was
        a fixed 0.85 shrinkage.

        ver4-34, R2.a (`weight_estimator='bagged_cluster'`, 'ensemble_score' objective): the
        weights are the correlation-clustered, bagged NNLS estimate on ALL the rows given
        (`robust_simplex_weights`, `weight_bags` bags, seed `seed + inner_split_seed_base`)
        - the same estimator the search scored candidates with. No equal-weight floor, no
        shrinkage, no selection of another ensemble."""
        if _bagged_final and _weight_estimator == 'bagged_cluster':
            _w, _bdiag = robust_simplex_weights(
                pred_matrix, actual_valid, n_bags=_weight_bags, cluster_corr=_cluster_corr,
                seed=int(seed) + int(inner_split_seed_base))
            _label = (f"bagged_cluster(bags={_weight_bags}, "
                      f"clusters={len(_bdiag.get('clusters', []))})")
            return (np.asarray(_w, dtype=float), _label,
                    {'clusters': _bdiag.get('clusters'), 'n_rows': _bdiag.get('n_rows')})
        if _bagged_final:
            # ver4-36 R2.c: any other registered robust estimator (ensemble_selection) - the
            # full function of the registry, same bags / seed rule as bagged_cluster.
            _w, _bdiag = _robust_weights_full(
                _weight_estimator, pred_matrix, actual_valid, _estimator_settings,
                n_bags=_weight_bags, seed=int(seed) + int(inner_split_seed_base))
            _label = (f"{_weight_estimator}(bags={_weight_bags}, "
                      f"steps={_bdiag.get('steps')})")
            return (np.asarray(_w, dtype=float), _label,
                    {'steps': _bdiag.get('steps'), 'n_rows': _bdiag.get('n_rows'),
                     'fallbacks': _bdiag.get('fallbacks')})
        if not _safety_nets_on:
            # ver4-44, R1 (N5): no equal-weight floor and no shrinkage - the weights are the
            # search's own (analytic, or the robust estimator above when one is set), as they are.
            return (np.asarray(search_shares, dtype=float), 'search_shares(safety_nets=False)',
                    {'safety_nets': False})
        if _weight_cv:
            _w, _a_star, _diag = cv_shrunk_simplex_weights(
                pred_matrix, actual_valid, alpha_cap=_shrinkage_alpha, seed=seed)
            _label = f"cv-shrunk(alpha={_a_star:g})"
            return np.asarray(_w, dtype=float), _label, _diag
        _analytic_w = analytic_simplex_weights(pred_matrix, actual_valid) if _analytic_seed else None
        _w, _label, _diag = select_weights_with_floor(
            search_shares, pred_matrix, actual_valid,
            shrinkage_alpha=_shrinkage_alpha,
            extra_candidates=[('analytic', _analytic_w)] if _analytic_w is not None else None,
        )
        return np.asarray(_w, dtype=float), _label, _diag

    _dead_final: set = set()
    _oof_partition = None      # ver4-38: the CV folds of the final out-of-fold matrix (when used)

    def _grade_weights(pred_matrix, actual_valid, search_shares):
        """ver4-38, R2.b1: `_grade_weights_core()` with the dead models fixed at weight 0.

        Why: a model whose column is constant (searched out of the group, or constant in the
        matrix being weighted) must not take part in the weight estimate - as a shrink-to-the-
        mean knob it would otherwise absorb weight and remove that model's signal. With
        `dead_column_guard` off this is exactly `_grade_weights_core()`. The remaining models are
        weighted by the same estimator on their columns only (their `search_shares` are
        renormalised), and the returned vector has length K with zeros for the dead models.
        Returns `(shares, label, diagnostics)` like the core function."""
        pred_matrix = np.asarray(pred_matrix, dtype=float)
        masked = set(_dead_models)
        if _dead_column_guard:
            for i, m in enumerate(group_models):
                _part = (_oof_partition if (_oof_partition is not None
                                            and pred_matrix.shape[0] == len(actual_valid)
                                            and len(actual_valid) == sum(
                                                len(h) for h in _oof_partition)) else None)
                if m not in masked and _is_dead_column(pred_matrix[:, i], actual_valid, _part):
                    masked.add(m)
        new_dead = masked - _dead_final
        if new_dead and _dead_column_guard:
            _dead_final.update(new_dead)
            print(f"[GP] WARN: Joint Bayesian optimisation dead_column_guard: the column of "
                  f"{sorted(new_dead)!r} is constant in the matrix being weighted - its weight is "
                  f"fixed at 0 (it carries no signal).")
        keep = [i for i, m in enumerate(group_models) if m not in masked]
        if not masked or not keep:
            return _grade_weights_core(pred_matrix, actual_valid, search_shares)
        shares_k = np.asarray(search_shares, dtype=float)[keep]
        total = float(shares_k.sum())
        shares_k = (shares_k / total if total > 0 and np.isfinite(total)
                    else np.full(len(keep), 1.0 / len(keep)))
        w_k, label, diag = _grade_weights_core(pred_matrix[:, keep], actual_valid, shares_k)
        full = np.zeros(len(group_models), dtype=float)
        full[keep] = np.asarray(w_k, dtype=float)
        return full, f"{label}; dead model(s) {sorted(masked)!r} fixed at weight 0", diag

    # ver4-28, R4: pool out-of-fold TRAIN predictions with the confirmatory
    # fit's own validation predictions before the final weight grading -
    # opt-in (`weight_source='oof_train_valid'`), so the weights come from
    # roughly N_train + N_valid rows instead of N_valid alone (the
    # "forecast combination puzzle", Round 10 note 1). `weight_source='valid'`
    # (default) reproduces the pre-ver4-28 behaviour exactly.
    _pool_pred_matrix, _pool_actual_valid = _pred_matrix, _actual_valid
    if _weight_source == 'oof_train_valid' and not _cv_active and not _wf_active:
        _t_oof = time.perf_counter()
        _oof_result = _oof_pooled_matrix(
            group_models, payload['final_params_by_model'], train, test, run_model_fn, RESULT_NAME,
            n_folds=_oof_folds, seed=seed, n_jobs=_effective_n_jobs, worker_init=worker_init,
            worker_init_args=worker_init_args,
        )
        _fit_counter['secs']['oof'] += time.perf_counter() - _t_oof
        if _oof_result is not None:
            _oof_pred, _oof_actual = _oof_result
            _fit_counter['oof'] = _fit_counter.get('oof', 0) + _oof_folds * len(group_models)
            _pool_pred_matrix = np.vstack([_oof_pred, _pred_matrix])
            _pool_actual_valid = np.concatenate([_oof_actual, _actual_valid])
            print(f"[GP] Joint Bayesian optimisation: weight_source='oof_train_valid' pooled "
                  f"{len(_oof_actual)} OOF training row(s) with {len(_actual_valid)} validation "
                  f"row(s) ({len(_pool_actual_valid)} total) for the final weight fit.")

    # ver4-34, R2.b: with the CV objective the final weights are estimated on the
    # out-of-fold matrix over train+valid at the SELECTED parameters (stage 'oof',
    # explainability off), i.e. on ~3x more rows than the validation split alone. The
    # partition seed is the search's own, so a model whose search parameters equal its
    # final ones is a `_cv_cache` hit. The final MODELS are still the train-only
    # confirmatory fits above (Step 9/11 rows stay comparable). If any out-of-fold fit
    # fails, the validation rows are used instead and the task says so.
    _oof_dump_cols = None
    _oof_label = ''
    if _cv_active and _wf_active and not _dump_on:
        print("[GP] Joint Bayesian optimisation weight-first: the final weights are the fixed "
              "candidate vector, so the final out-of-fold refit is skipped for this task.")
        if _wfcv_members or _search_strategy == 'neural_stack':
            # ver4-47, R6: the member selection below only needs the frame and the partition
            # (the dead-column test); no fit is made here.
            _tv_frame = _cv_frame(train, valid)
            _oof_partition = _cv_partition(len(_tv_frame), _cv_folds,
                                           int(inner_split_seed_base) + _JOINT_CV_SEED_OFFSET)
    elif _cv_active:
        _cv_final_seed = int(inner_split_seed_base) + _JOINT_CV_SEED_OFFSET
        _tv_frame = _cv_frame(train, valid)
        if _oof_search_active:
            # ver4-38, R1.d: the OOF matrix is built from the SEARCH-fidelity decode of the
            # selected point - the same parameters the winning library entry was fitted with, so
            # every (model, fold) fit is a `_cv_cache` hit (0 new fits when the winner is a
            # library entry) and the exact engine / stub flags apply to any miss. The final
            # MODELS (confirmatory fits above) stay full fidelity.
            _oof_requests = []
            _off = 0
            for m in group_models:
                _d = len(specs_by_model[m])
                _oof_requests.append((m, _decode_model_params(
                    m, best_x[_off:_off + _d], specs=specs_by_model[m],
                    base_params=base_params_by_model[m], search_fidelity=True)))
                _off += _d
            _oof_label = " (oof_fidelity='search')"
        else:
            _oof_requests = [
                (m, _disable_explainability(m, list(payload['final_params_by_model'][m])))
                for m in group_models]
        _oof_partition = _cv_partition(len(_tv_frame), _cv_folds, _cv_final_seed)
        _cv_final_cols = _cv_fit_columns(
            _oof_requests,
            tv=_tv_frame, partition=_oof_partition,
            seed=_cv_final_seed, test=test, runner=run_model_fn, RESULT_NAME=RESULT_NAME,
            runner_ctx=_runner_ctx('oof'))
        if any(col is None for col in _cv_final_cols):
            print("[GP] NOTE: Joint Bayesian optimisation could not form the out-of-fold matrix "
                  "over train+valid at the selected parameters (a fold failed - see the "
                  "[joint_tuning] NOTE above) - the final weights use the validation rows only "
                  "for this task.")
        else:
            _oof_dump_cols = list(_cv_final_cols)
            _pool_pred_matrix = np.column_stack(_cv_final_cols)
            _pool_actual_valid = np.asarray(_tv_frame.iloc[:, -1], dtype=float)
            if _pool_pred_matrix.shape != (len(_tv_frame), len(group_models)):
                raise AssertionError(
                    f"[joint_tuning] internal error: out-of-fold matrix has shape "
                    f"{_pool_pred_matrix.shape}, expected ({len(_tv_frame)}, {len(group_models)}).")
            _search_shares = np.asarray(
                analytic_simplex_weights(_pool_pred_matrix, _pool_actual_valid), dtype=float)
            print(f"[GP] Joint Bayesian optimisation: final weights estimated on "
                  f"{len(_pool_actual_valid)} out-of-fold row(s) (train {len(train)} + valid "
                  f"{len(valid)}) instead of the {len(_actual_valid)} validation row(s)"
                  f"{_oof_label}.")
    elif _oof_search_active and not _cv_active:
        print("[GP] NOTE: Joint Bayesian optimisation oof_fidelity='search' only applies with "
              "objective_data='cv_train_valid' - this task uses the validation objective, so the "
              "setting has no effect.")

    # ver4-36, R2.d: the library dump (opt-in `library_dump`). After the confirmatory fit so the
    # file can record the final decision; a failure is a NOTE, never a task failure.
    # ver4-38: moved after the out-of-fold block so the dump (version 2) can carry `oof__k`.
    if _dump_on and _lib_after:
        try:
            from checkpoint_utils import DIAGNOSTIC_DIR_NAMES
            _dump_dir = os.path.join(str(diagnostics_ctx['result_dir']),
                                     DIAGNOSTIC_DIR_NAMES['joint_library'])
            _dump_path = os.path.join(
                _dump_dir, _safe_name(diagnostics_ctx.get('task_tag') or RESULT_NAME) + '.npz')
            _dump_ctx = _make_column_context(
                group_models=group_models, specs_by_model=specs_by_model,
                base_params_by_model=base_params_by_model, train=train, valid=valid, test=test,
                runner=run_model_fn, RESULT_NAME=RESULT_NAME, runner_ctx=None,
                objective_data=_objective_data, cv_folds=_cv_folds,
                inner_split_seed_base=inner_split_seed_base)
            _write_library_dump(
                _dump_path, group_models=group_models, lib=_lib_after,
                specs_by_model=specs_by_model, base_params_by_model=base_params_by_model,
                test_sink=_test_sink, actual=_dump_ctx['actual'],
                y_test=(np.asarray(test.iloc[:, -1], dtype=float) if test is not None else None),
                partition=_dump_ctx.get('partition'), best_x=best_x,
                oof_cols=_oof_dump_cols,
                wf_weights=(_wf_stats['w'] if _wf_active else None),
                entry_params=(_block_stats.get('ns_entry_params')
                              if (_wf_active and _search_strategy == 'neural_stack') else None),
                meta={'version': 'ver4-38', 'dump_version': _JOINT_DUMP_VERSION,
                      'oof_fidelity': ('search' if _oof_search_active else 'full'),
                      'seed': int(seed),
                      'inner_split_seed_base': int(inner_split_seed_base),
                      'search_strategy': _search_strategy, 'objective_mode': objective_mode,
                      'objective_data': _objective_data, 'cv_folds': int(_cv_folds),
                      'weight_estimator': _weight_estimator,
                      'estimator_settings': dict(_estimator_settings or {}),
                      'diversity_penalty': _diversity_penalty,
                      'diversity_method': _diversity_method,
                      'score_weight_rule': _score_weight_rule,
                      'shrinkage_alpha': _shrinkage_alpha, 'block_tol': _block_tol,
                      'selection_margin': margin, 'gate_mode': _gate_mode,
                      'selection_rule': _selection_rule, 'gate_decision': _gate_decision,
                      'bbc_bootstraps': _bbc_bootstraps, 'bbc_gate_quantile': _bbc_gate_quantile,
                      'confirm_score': float(_confirm_score),
                      'wf': ({k: v for k, v in _wf_stats.items() if k != 'table'}
                             if _wf_active else None),
                      # ver4-48: the key exists only in a neural-stack dump (other dumps are
                      # unchanged)
                      **({'ns': {k: v for k, v in _block_stats['ns'].items() if k != 'lm'}}
                         if (_wf_active and _search_strategy == 'neural_stack') else {}),
                      'task_tag': str(diagnostics_ctx.get('task_tag'))})
            print(f"[GP] Joint Bayesian optimisation library dump written: {_dump_path} "
                  f"({sum(len(v) for v in _lib_after.values())} columns over "
                  f"{len(group_models)} model(s)) - replay it with tools/joint_replay.py.")
        except Exception as exc:
            _dump_path = None
            print(f"[GP] NOTE: Joint Bayesian optimisation library dump could not be written "
                  f"({type(exc).__name__}: {exc}) - the task is unaffected.")

    if _wf_active:
        # ver4-40, R2: the chosen fixed vector, dead models zeroed (as `_grade_weights` does) and
        # the rest renormalised. Never re-estimated: that is the point of weight-first.
        _w_fix = np.asarray(_wf_stats['w'], dtype=float).copy()
        _wf_masked = set(_dead_models)
        for _i, _m in enumerate(group_models):
            if (_dead_column_guard and _m not in _wf_masked and _w_fix[_i] > 0
                    and _is_dead_column(_pred_matrix[:, _i], _actual_valid)):
                _wf_masked.add(_m)
                _dead_final.add(_m)
                print(f"[GP] WARN: Joint Bayesian optimisation dead_column_guard: the confirmatory "
                      f"column of {_m!r} is constant - its weight in the fixed weight-first vector "
                      f"is set to 0 and the rest renormalised.")
        for _i, _m in enumerate(group_models):
            if _m in _wf_masked:
                _w_fix[_i] = 0.0
        _tot = float(_w_fix.sum())
        if not np.isfinite(_tot) or _tot <= 0.0:
            _live = [i for i, m in enumerate(group_models) if m not in _wf_masked] or \
                list(range(len(group_models)))
            _w_fix = np.zeros(len(group_models))
            _w_fix[_live] = 1.0 / len(_live)
        else:
            _w_fix = _w_fix / _tot
        _final_shares = _w_fix
        _floor_label = (f"weight_first({_wf_stats['name']}, {_wf_stats['arm']}, "
                        f"pick={_wf_stats['pick_how']})")
        _floor_diag = {'fixed': True, 'masked': sorted(_wf_masked)}
    else:
        _final_shares, _floor_label, _floor_diag = _grade_weights(
            _pool_pred_matrix, _pool_actual_valid, _search_shares)
    weights = {m: float(w) for m, w in zip(group_models, _final_shares)}
    print(f"[GP] Joint Bayesian optimisation weight selection: selected {_floor_label!r} "
          f"(validation diagnostics - {_floor_diag}) -> weights={weights}{_oof_label}")

    # BUGFIX (joint-optimisation-underperformance-fix, Round 5 - the root
    # cause of this session's own reported regression, Metric.csv vs
    # Metric_original.csv): every safeguard above only ever grades the
    # search's own winning candidate against "every group model at its own
    # configured hyperparameters, EQUAL ensemble weighting" (the
    # default-vs-winner margin/1-SE rule compare search objective scores
    # against `seed_point`; `select_weights_with_floor()` just above grades
    # weight vectors against equal weighting on the WINNER'S OWN, already
    # hyperparameter-tuned, predictions). None of that ever compares
    # against "every group model at its own configured hyperparameters,
    # OPTIMALLY weighted" - which is exactly what the non-joint 'Bayesian
    # optimisation'/'Analytic least-squares' weight-only methods the user
    # is comparing this search's own results against actually produce.
    # A joint search whose own hyperparameter changes make one or more
    # group models' raw validation predictions individually WORSE - even
    # while its own ensemble weights are well-chosen for those (now worse)
    # predictions - can therefore still lose to simply leaving
    # hyperparameters untouched and weighting the UNTUNED models
    # optimally, and nothing above ever catches that case, since the
    # untuned baseline is only ever re-evaluated under EQUAL weighting
    # (`seed_point`'s own weight coordinates), never under its own best
    # achievable weighting. Theoretically the joint search should never
    # lose this comparison (it can always choose to leave hyperparameters
    # at the baseline and let its own weight dimensions converge on the
    # same optimum) - in practice, on a limited evaluation budget over a
    # combined space many times larger than either sub-problem alone (see
    # JOINT_SEARCH_DIM_WARN_THRESHOLD), it is entirely possible for the
    # search to spend its budget on hyperparameter regions that never let
    # it rediscover that optimum. Fixed here with the same one-more-
    # confirmatory-fit trade-off the 1-SE rule above already pays: whenever
    # the search's own winner survived every safeguard above (`best_x`
    # differs from `seed_point` - if it did not, this fallback already
    # reduces to the untuned baseline and this comparison is redundant),
    # pay for ONE more full-fidelity confirmatory fit of the untuned
    # baseline itself, grade an equal- and (when enabled) analytic-weighted
    # ensemble of THAT baseline's own realised validation predictions
    # through the SAME `select_weights_with_floor()` mechanism used above,
    # and fall back to the baseline entirely - untuned hyperparameters,
    # optimally-graded weights - whenever its realised validation MSE
    # beats the joint search's own floor-graded result. This makes "the
    # joint search is never worse than untuned models plus the best
    # ensemble weighting this codebase already knows how to compute" a
    # construction-level guarantee, closing the gap every safeguard above
    # leaves open.
    #
    # ROUND 6 ADDENDUM: the paragraph above ("the untuned baseline is only
    # ever re-evaluated under EQUAL weighting... never under its own best
    # achievable weighting") described `default_score`/`best_score` above
    # (the default-vs-winner margin and 1-SE-rule comparisons) as they stood
    # through Round 5 - true then, no longer true now. Round 6 (see this
    # module's own top-level Round 6 note) removed the ensemble weights from
    # the searched space entirely, so `default_score` (evaluating
    # `seed_point`, i.e. the untuned baseline) and `best_score` (evaluating
    # the search's own winning hyperparameters) are BOTH already computed via
    # `_score_candidate()`'s own per-candidate analytic weight solve, at
    # search-phase fidelity, for EVERY task - not only when this opt-in
    # floor fires. That margin/1-SE comparison is therefore already a fair
    # "baseline, optimally weighted, vs candidate, optimally weighted"
    # comparison by construction, which is what this block below still
    # re-derives independently, at full fidelity, as a second check. This
    # block is kept ON here rather than removed: it still catches the one
    # remaining gap Round 6 does not close on its own - a search-phase
    # (cheapened) evaluation and the full-fidelity confirmatory fit
    # occasionally disagreeing - but it should fire far less often than it
    # did through Round 5, since the fundamental confound it was built to
    # catch no longer exists upstream. Still opt-in/OFF by default (see
    # `_JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT`'s own comment): it remains a
    # genuine wall-clock trade-off (one more full-fidelity refit of every
    # group model), not a free correctness fix, so it stays a deliberate
    # choice rather than something applied unconditionally.
    #
    # OFF by default (`_untuned_baseline_floor`, see this key's own
    # `_JOINT_UNTUNED_BASELINE_FLOOR_DEFAULT` comment above for why this
    # one knob is opt-in rather than opt-out unlike every other Round-5-
    # and-earlier safeguard in this function) - a user who wants this
    # guarantee enables it explicitly via the 'untuned_baseline_floor'
    # key inside HYPERPARAMETERS_JOINT_OPT; with it left off (the
    # default), this whole block never runs and this function's
    # behaviour is unaffected by Round 5's primary fix, byte for byte.
    _floor_replaced = False
    if (_untuned_baseline_floor and total_dims > 0 and not _wf_active
            and not np.array_equal(best_x, seed_point)):
        _base_score, _base_payload = _score_candidate(
            seed_point, group_models=group_models, base_params_by_model=base_params_by_model,
            specs_by_model=specs_by_model, train=train, valid=valid, test=test,
            run_model_fn=run_model_fn, RESULT_NAME=RESULT_NAME,
            objective_mode=objective_mode, search_fidelity=False,
            diversity_penalty=_diversity_penalty, diversity_method=_diversity_method,
            state=state, weight_cv_alpha=_weight_cv_alpha, use_base_params=True,
            runner_ctx=_runner_ctx('confirm'),
            weight_rule=_score_weight_rule, shrinkage_alpha=_shrinkage_alpha, **_v434_kwargs,
        )
        if _base_payload is None:
            print(f"[GP] NOTE: Joint Bayesian optimisation could not re-evaluate the untuned "
                  f"baseline at full fidelity to compare against this search's own winner (see "
                  f"the [joint_tuning] NOTE above for why) - keeping this search's own "
                  f"floor-graded result for this task.")
        else:
            _base_pred_matrix = _base_payload['pred_matrix']
            _base_actual_valid = _base_payload['actual_valid']
            _base_equal_shares = np.full(len(group_models), 1.0 / len(group_models))
            _base_analytic_w = (
                analytic_simplex_weights(_base_pred_matrix, _base_actual_valid) if _analytic_seed else None
            )
            _base_final_shares, _base_floor_label, _base_floor_diag = _grade_weights(
                _base_pred_matrix, _base_actual_valid, _base_equal_shares)
            _base_final_shares = np.asarray(_base_final_shares, dtype=float)
            _winner_mse = float(np.mean((_pred_matrix @ np.asarray(_final_shares, dtype=float)
                                          - _actual_valid) ** 2))
            _baseline_mse = float(np.mean((_base_pred_matrix @ _base_final_shares - _base_actual_valid) ** 2))
            if _baseline_mse < _winner_mse:
                print(f"[GP] Joint Bayesian optimisation: untuned baseline + optimally-weighted "
                      f"ensemble ({_base_floor_label!r}, validation MSE={_baseline_mse:.4f}) beat "
                      f"this search's own tuned, floor-graded result (validation MSE="
                      f"{_winner_mse:.4f}) - falling back to every group model's own configured "
                      f"hyperparameters, weighted by {_base_floor_label!r} instead of this "
                      f"search's own tuned hyperparameters, so the joint search is never worse "
                      f"than not tuning hyperparameters at all plus the best ensemble weighting "
                      f"this codebase already knows how to compute.")
                payload = _base_payload
                weights = {m: float(w) for m, w in zip(group_models, _base_final_shares)}
                # ver4-32 R4: `_base_final_shares` are already the graded (shrunk)
                # weights, so under 'shrunk' they are scored as they are
                # (shrinkage_alpha=1.0 = no second shrinkage, and no penalty).
                best_score = _combine_score(_base_final_shares, _base_pred_matrix, _base_actual_valid,
                                             group_models, objective_mode, _diversity_penalty,
                                             _diversity_method, state=state,
                                             weight_rule=('shrunk' if _bagged_final
                                                          else _score_weight_rule),
                                             shrinkage_alpha=1.0)
                _gate_decision = 'gate_fallback'
                _floor_replaced = True

    # ver4-38, R2.b2: the library ensemble (opt-in `library_ensemble`). After the selection and
    # the untuned-baseline floor: the search's by-product (the other fitted configurations of
    # each model) is combined with the selected ones. Step 9 still refits/reuses the PRIMARY of
    # every model exactly as before; the EXTRAS get one train-only, full-fidelity,
    # explainability-off confirm fit each here, and `weights` becomes the per-model SUM of the
    # member weights (so Weight.csv keeps its columns). genomic_prediction.py combines primaries
    # plus extras from `details['library_members']`. Any failure leaves the ordinary result.
    _library_members = None
    if _library_ensemble_active:
        _lm_skip = None
        if _floor_replaced:
            _lm_skip = ("the untuned-baseline floor replaced the search's result (its columns are "
                        "not library entries)")
        elif not _lib_after or best_x is None:
            _lm_skip = "this search left no library"
        elif not _cv_active:
            _lm_skip = "the out-of-fold objective is not active for this task"
        if _lm_skip is None:
            _lm_sel = _selected_entries(group_models, _lib_after, specs_by_model, best_x)
            if any(v < 0 for k, v in _lm_sel.items() if k not in _dead_models):
                _lm_skip = (f"the selected point is not a library entry for "
                            f"{sorted(k for k, v in _lm_sel.items() if v < 0)!r}")
        if _lm_skip is None:
            _lm_y = np.asarray(_tv_frame.iloc[:, -1], dtype=float)
            _lm_live = [m for m in group_models if m not in _dead_models]
            _lm_seed = int(seed) + int(inner_split_seed_base) + _JOINT_LIBRARY_MEMBER_SEED_OFFSET

            def _lm_weight_fn(P, y):
                _w, _d = _robust_weights_full(
                    'ensemble_selection', P, y, _estimator_settings, n_bags=_weight_bags,
                    seed=_lm_seed)
                return _w

            try:
                _lm_sel_live = {m: _lm_sel[m] for m in _lm_live}
                if _wf_active and _search_strategy == 'neural_stack':
                    # ver4-48: the members were chosen by the search from the network's column
                    # weights W_m * q_ml (nothing is re-estimated); a model whose confirmatory
                    # column turned out dead has weight 0 in `weights` and loses its members.
                    _lm = copy.deepcopy(_block_stats['ns'].get('lm'))
                    if _lm is not None:
                        _lm_alive = [x for x in _lm['members']
                                     if float(weights.get(x['model'], 0.0)) > 0.0]
                        _lm_tot = float(sum(x['weight'] for x in _lm_alive))
                        if not _lm_alive or _lm_tot <= 0.0:
                            _lm = None
                        else:
                            for x in _lm_alive:
                                x['weight'] = float(x['weight']) / _lm_tot
                            _lm['members'] = _lm_alive
                elif _wf_active and _wfcv_members:
                    # ver4-47, R6: fixed per-model weights, split over the top-q configurations.
                    _lm = _wf_split_members(
                        _lm_live, {m: _lib_after[m] for m in _lm_live}, _lm_sel_live, _lm_y,
                        {m: float(weights[m]) for m in _lm_live},
                        per_model=_members_per_model, dead_guard=_dead_column_guard,
                        partition=_oof_partition)
                else:
                    _lm = _library_ensemble_members(
                        _lm_live, {m: _lib_after[m] for m in _lm_live}, _lm_sel_live, _lm_y,
                        _lm_weight_fn, per_model=_members_per_model, members_max=_members_max,
                        min_weight=_JOINT_LIBRARY_MEMBER_MIN_WEIGHT, dead_guard=_dead_column_guard,
                        partition=_oof_partition)
            except Exception as exc:
                _lm = None
                print(f"[GP] NOTE: Joint Bayesian optimisation library_ensemble could not choose "
                      f"members ({type(exc).__name__}: {exc}) - the ordinary one-configuration-"
                      f"per-model result is used for this task.")
            else:
                if _lm is None:
                    _lm_skip = "no member set with weight >= the minimum could be formed"
        if _lm_skip is not None:
            print(f"[GP] NOTE: Joint Bayesian optimisation library_ensemble is skipped for this "
                  f"task: {_lm_skip} - the ordinary one-configuration-per-model result is used.")
        elif _lm is not None:
            _lm_members = _lm['members']
            _lm_extras = [mem for mem in _lm_members if not mem['primary']]
            for mem in _lm_members:
                x_m = np.asarray(_lib_after[mem['model']][mem['entry']][0], dtype=float)
                if mem['primary']:
                    mem['params'] = list(payload['final_params_by_model'][mem['model']])
                    mem['pred'] = None
                else:
                    mem['params'] = _disable_explainability(mem['model'], _decode_model_params(
                        mem['model'], x_m, specs=specs_by_model[mem['model']],
                        base_params=base_params_by_model[mem['model']], search_fidelity=False))
                    mem['pred'] = None
            _lm_dropped = []
            if _lm_extras:
                _lm_rc = _runner_ctx('confirm')
                _lm_units = [_FitUnit(model=mem['model'], params=mem['params'],
                                      split_tag='confirm', train=train, valid=valid,
                                      fidelity='confirm', keep_full=True) for mem in _lm_extras]
                _lm_fitted = _run_units(
                    _lm_units, runner=run_model_fn, RESULT_NAME=RESULT_NAME, test=test,
                    n_jobs=_lm_rc['n_jobs'], worker_init=_lm_rc['worker_init'],
                    worker_init_args=_lm_rc['worker_init_args'], cache=None,
                    stop_on_failure=False, counter=_fit_counter, stage='confirm',
                    fit_flags=_lm_rc['fit_flags'])
                for mem, res in zip(_lm_extras, _lm_fitted):
                    arrays = {}
                    if isinstance(res, dict):
                        for _k, _src in (('train', 'predicted_train'), ('valid', 'predicted_valid'),
                                         ('test', 'predicted_test')):
                            _v = res.get(_src)
                            if _v is None:
                                break
                            _a = np.asarray(_v, dtype=float).ravel()
                            if _a.size == 0 or not np.all(np.isfinite(_a)):
                                break
                            arrays[_k] = _a
                    if len(arrays) == 3:
                        mem['pred'] = arrays
                    else:
                        _lm_dropped.append(mem)
                        print(f"[GP] NOTE: Joint Bayesian optimisation library_ensemble: the extra "
                              f"member {mem['model']!r} entry {mem['entry']} could not be fitted "
                              f"(or returned a non-finite / missing prediction) - it is dropped "
                              f"and the other member weights are renormalised.")
            _lm_final = [mem for mem in _lm_members if mem not in _lm_dropped]
            _lm_total = float(sum(mem['weight'] for mem in _lm_final))
            if not _lm_final or _lm_total <= 0.0:
                print("[GP] NOTE: Joint Bayesian optimisation library_ensemble: no member is left "
                      "after the extra fits - the ordinary result is used for this task.")
            else:
                for mem in _lm_final:
                    mem['weight'] = float(mem['weight']) / _lm_total
                _library_members = _lm_final
                _lm_sums = {m: 0.0 for m in group_models}
                for mem in _lm_final:
                    _lm_sums[mem['model']] += mem['weight']
                _old_weights = dict(weights)
                weights = {m: float(_lm_sums[m]) for m in group_models}
                print(f"[GP] Joint Bayesian optimisation library_ensemble: "
                      f"{len(_lm_final)} member(s) ({sum(1 for x in _lm_final if x['primary'])} "
                      f"primary, {sum(1 for x in _lm_final if not x['primary'])} extra) from "
                      f"{_lm['n_pruned']} pruned library column(s): "
                      + ', '.join(f"{x['model']}#{x['entry']}{'*' if x['primary'] else ''}="
                                  f"{x['weight']:.3f}" for x in _lm_final)
                      + f" ('*' = the selected, Step 9 configuration) -> per-model weight sums="
                      f"{ {m: round(v, 4) for m, v in weights.items()} } (was {_old_weights}); "
                      f"the joint effect row uses these sums x the primaries' effects "
                      f"(approximation).")

    # ver4-32 R1: put the frozen run-length fields back into
    # `tuned_values_by_model` (their value in the confirmatory fit, i.e. the
    # configured one) so hyperparameter.csv keeps the same columns, in the same
    # order, as when those fields were searched. Done after the untuned-baseline
    # floor, which may have replaced `payload`.
    if _frozen_fields:
        _tv = payload['tuned_values_by_model']
        _fp = payload['final_params_by_model']
        for m in _frozen_fields:
            _tv[m] = {sp.name: _fp[m][sp.index] for sp in _full_specs_by_model[m]}

    # ver4-44, R1 (N14): models whose column was constant and got weight 0 are class C events.
    for _m in sorted(set(_dead_models) | set(_dead_final)):
        _fallback_events.append(f"N14:{_m}")

    elapsed = time.time() - t0

    # ver4-48: the neural stack's details payload (None for every other strategy). `members` is
    # the number of members of the deployed mixture, or the number of models with a positive
    # weight when no member set was formed (ns_deploy='argmax' or no extra column was heavy).
    _ns_details = None
    if _wf_active and _search_strategy == 'neural_stack':
        _ns_details = {k: v for k, v in _block_stats['ns'].items() if k != 'lm'}
        _ns_details['members'] = (len(_library_members) if _library_members
                                  else int(sum(1 for _m in group_models
                                               if float(weights.get(_m, 0.0)) > 0.0)))
        _ns_details['weights'] = {m: float(w) for m, w in weights.items()}

    # ver4-28, R6: provenance - one line, every task, unconditionally (never
    # only on request), naming the search strategy, the accept/reject
    # decision and a per-stage fit-count breakdown, so "the search searched
    # X and accepted/rejected because Y" is auditable straight from the run
    # log without re-running anything. `hp_record` gains the matching
    # 'joint_strategy'/'joint_gate_decision'/'joint_n_fits' columns in
    # genomic_prediction.py, but only when at least one ver4-28 key is
    # non-default (`_v428_nondefault()` below) - this log line itself is
    # unconditional either way.
    _secs = dict(_fit_counter['secs'])
    _n_fits = {k: v for k, v in _fit_counter.items()
               if k not in ('secs', 'fits_by_model', 'secs_by_model', 'engine_fits')}
    _fit_total = sum(v for k, v in _n_fits.items() if k != 'cache_hits')
    print(f"[joint_tuning] fits: search={_n_fits['search']} gate={_n_fits['gate']} "
          f"confirm={_n_fits['confirm']} oof={_n_fits['oof']} "
          f"cache_hits={_n_fits['cache_hits']} total={_fit_total} | "
          f"secs: search={_secs['search']:.1f} gate={_secs['gate']:.1f} "
          f"confirm={_secs['confirm']:.1f} oof={_secs['oof']:.1f} | "
          f"gate={_gate_decision} | strategy={_search_strategy} | model_secs: "
          + ' '.join(f"{m}={_fit_counter['secs_by_model'].get(m, 0.0):.1f}"
                     for m in group_models)
          + f" | objective={_objective_data} estimator={_weight_estimator}"
          + (f" | engine_fits: exact_kernel={_fit_counter['engine_fits']['exact_kernel']} "
             f"stub={_fit_counter['engine_fits']['stub']} "
             f"runner={_fit_counter['engine_fits']['runner']}"
             if (_fit_flags['engine_models'] or _fit_flags['stub_models']) else "")
          + (f" | critical_path_s={_block_stats.get('critical_path_s', 0.0):.1f} "
             f"utilisation={_block_stats.get('utilisation', 0.0):.2f}"
             if _search_strategy == 'async_library' and _block_stats else "")
          + (f" | racing: " + ' '.join(
              f"{m}={v['promoted']}/{v['raced']}" for m, v in _block_stats['raced'].items())
             + " (promoted/raced)"
             if _racing_active and _block_stats and _block_stats.get('raced') else "")
          + (f" | stopped: {_block_stats['stopped']}"
             if _stop_rule_active and _block_stats and _block_stats.get('stopped') else "")
          + (f" | dead_models={sorted(_dead_models)}" if _dead_models else "")
          + (f" | wf: candidate={_wf_stats['name']!r} arm={_wf_stats['arm']} "
             f"pick={_wf_stats['pick_how']} n_candidates={_wf_stats['n_candidates']} "
             f"rungs={_wf_stats['rungs_run']}" if _wf_active else "")
          + (f" alpha={_wf_stats['alpha']:.2f} rows={_wf_stats['rows']!r}"
             if _wf_active and 'alpha' in _wf_stats else "")
          + (f" | ns: lambda={'inf' if not math.isfinite(_ns_details['lambda']) else format(_ns_details['lambda'], 'g')} "
             f"rule={_ns_details['rule']} deploy={_ns_details['deploy']} "
             f"members={_ns_details['members']} library_columns={_ns_details['n_cols']} "
             f"entropy={_ns_details['entropy']:.3f} converged={_ns_details['converged']}"
             if _ns_details is not None else "")
          + (f" | safety_nets=False fallbacks={_fallback_events if _fallback_events else 'none'}"
             if not _safety_nets_on else
             (f" | fallbacks={_fallback_events}" if _fallback_events else "")))

    details = {
        'strategy': _search_strategy,
        'gate_decision': _gate_decision,
        'n_fits': _n_fits,
        'secs': _secs,
        'confirm_results': payload.get('confirm_results') if _reuse_confirmatory_fit else None,
        'weight_source': _weight_source,
        # ver4-34
        'fits_by_model': dict(_fit_counter['fits_by_model']),
        'secs_by_model': dict(_fit_counter['secs_by_model']),
        'objective_data': _objective_data,
        'weight_estimator': _weight_estimator,
        'search_space': _search_space,
        'fit_estimate': _fit_estimate,
        # ver4-36
        'engine_fits': dict(_fit_counter['engine_fits']),
        'fit_flags': {'stub_models': sorted(_fit_flags['stub_models']),
                      'engine_models': sorted(_fit_flags['engine_models'])},
        'parity_corr': (_parity or {}).get('corr'),
        'bbc': _bbc_summary,
        'critical_path_s': _block_stats.get('critical_path_s'),
        'utilisation': _block_stats.get('utilisation'),
        'library_dump': _dump_path,
        'noise_aware_surrogate': bool(_noise_aware),
        # ver4-38
        'dead_models': sorted(set(_dead_models) | set(_dead_final)),
        'racing': _block_stats.get('raced') if _block_stats else None,
        'stopped': _block_stats.get('stopped') if _block_stats else None,
        'oof_fidelity': ('search' if _oof_search_active else 'full'),
        'library_members': _library_members,
        # ver4-40 (None unless search_strategy='weight_first' ran its own search)
        'wf': ({k: v for k, v in _wf_stats.items()} if _wf_active else None),
        'joint_weight_candidate': (_wf_stats['name'] if _wf_active else None),
        # ver4-48 (R2): {lambda, rule, lambdas, path, W, q, effective, members, n_cols, dropped,
        # entropy, converged, fidelity, deploy, objective, n_iter, weights}; None unless
        # search_strategy='neural_stack' ran
        'ns': _ns_details,
        # ver4-44 (R1): the resolved switch and the class C events of this task, e.g.
        # ['N12:SVR', 'N13:valid_rows'] ('N11' is added by genomic_prediction.py, which owns the
        # per-model confirmatory refit)
        'safety_nets': bool(_safety_nets_on),
        # ';' joins the events in hyperparameter.csv (`joint_fallbacks`), so none may contain one
        'fallback_events': [str(_e).replace(';', ',') for _e in _fallback_events],
    }

    if return_details:
        return (payload['final_params_by_model'], weights, float(best_score), elapsed,
                payload['tuned_values_by_model'], details)
    return (payload['final_params_by_model'], weights, float(best_score), elapsed,
            payload['tuned_values_by_model'])
