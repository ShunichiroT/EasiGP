"""joint_replay.py - ver4-36, R2.d (INVESTIGATE): offline replay of the joint search's library.

Why: the joint score is a pure function of the K prediction columns, so once a task's fitted
library has been dumped (`HYPERPARAMETERS_JOINT_OPT['library_dump'] = True` writes
`Result/<RESULT_NAME>/Joint_library/<task_tag>.npz` after the confirmatory fit) every
"which library column per model" rule and every ensemble-weight estimator can be re-run on it
in seconds, with ZERO model fits, and compared on the TEST rows the dump carries from the very
same fits. A real arm costs hours; this ranks the variants first, and the real-run benchmark
(tools/joint_benchmark.py, arms F1/A1/A2/A3) confirms the top two.

This script never fits a model, never writes into the engine's accumulators and is never
imported by the engine. It imports `models.joint_tuning` for the scoring maths so the replay
uses the engine's own definition of "the score of these columns" (one authority).

ver4-40, R2 (A2.0 gate): rule `weight_first` replays the weight-first search (fixed candidate
weight vectors -> zero-fit column selection under each -> bootstrap out-of-bag pick; see
`replay_weight_first`) once per dump and `replay_a20.csv` / the printed `A2.0 DECISION` apply the
go / no-go criteria of the blueprint (>= 20 task dumps; vs the naive mean of the untuned columns,
vs the best untuned single column, and vs the J arm + 0.005 with a paired bootstrap CI). Options
`--wf-rules`, `--wf-max-candidates`, `--wf-dedup-tol`, `--wf-no-untuned`, `--wf-pick`. Add
`--rules weight_first` alone (with `--estimators legacy`) for just that arm.

ver4-42, R1 (A1.0 gate): rule `weight_first_cv` replays stage S1 (and the S3 decision) of
`search_strategy='weight_first_cv'` on dumps written with `objective_data='cv_train_valid'`
(`replay_weight_first_cv`): the uniform -> analytic shrinkage level is chosen by repeated CV over
ALL out-of-fold rows of library entry 0 (`ensemble_regularization.shrinkage_path_weights`) and the
weights are applied to the entry-0 test columns. `replay_a10.csv` / the printed `A1.0 DECISION`
apply the go / no-go criteria of the ver4-42 blueprint. Options `--wfcv-direction`,
`--wfcv-alpha-rule`, `--wfcv-repeats`, `--n-valid-frac`. ver4-44, R2: `--evaluate --joint <dir> --reference <dir> [--exclude-fallbacks]` pairs a joint run
with an independent reference run (same config, JOINT_W_OPT off, W_OPT on) task by task with zero
fits and reports Delta_total = Delta_tune + Delta_weight, the weighting ceiling H, the deltas vs
every reference single model and W_OPT method, pooled statistics and a recorded verdict (see
`evaluate_runs`); outputs `evaluation_tasks.csv` and `evaluation_summary.csv`. It reads standard
result files only and never imports the engine's fitting code.

ver4-42, R2: `--dp-check` measures, with
zero fits, whether a dynamic-programming / grid optimiser could help (A2.1-A2.2) and writes
`replay_dp_check.csv`. Add `--rules weight_first_cv --estimators legacy` for just the A1.0 arm.

ver4-48, R1 (A-NS.0 gate): rule `neural_stack` replays the "neural stack" (`models/hyper_stack.py`:
hyperparameter choices and ensemble weights as the weights of one small network, lambda chosen by
repeated CV along a path whose lambda = infinity end is the untuned naive ensemble) on dumps written
with `objective_data='cv_train_valid'` (`replay_neural_stack`): the columns are ALL library entries
of the scoring rows, the weights come from `hyper_stack.lambda_path`, and the test prediction is
the weighted sum of the dump's test columns (a NaN test column is dropped and the rest
renormalised; the count is reported). `replay_ans0.csv` / the printed `A-NS.0 DECISION` apply the
go / no-go criteria G1-G4 of the ver4-48 blueprint (see `ans0_gate`). Options `--ns-lambdas`,
`--ns-rule`, `--ns-repeats`. Zero fits. Not part of the default `--rules` (add
`--rules neural_stack --estimators legacy` for just this arm).

ver4-50, R1 (A-NS.1 gate): the neural-stack replay now mirrors the ver4-49/50 engine (the replay
of ver4-48 lacked the Nadeau-Bengio one-SE rule and the weight shrink - blueprint disagreement D1)
and `--ns-variants V0,V1,V2,V3` replays four variants on every dump (V0 = the ver4-49 engine,
V1 = `ns_objective='pearson'`, V2 = V1 + `ns_swap_rule='paired_se'`, V3 = V2 + mixture
deployment): `replay_ans1.csv` / `replay_ans1_gate.csv` and the printed `A-NS.1 DECISION:` apply
G1-G4 of A-NS.0 plus G5 (no worse than V0); `replay_ns_diag.csv` and the printed H1-H3 lines
measure WHY (calibration share of the network's gain, over-dispersed heavy columns, losing gate
flips). `--ns-parity` checks that the replay reproduces the engine's own model weights from the
dump (< 1e-9). The lambda-choice folds use the engine's seed stored in each dump unless
`--ns-seed` is given. Zero fits. Example:
  python tools/joint_replay.py --dumps 'Result/*/Joint_library/*.npz' --out replay_out \
      --rules neural_stack --estimators legacy --ns-variants V0,V1,V2,V3 --ns-parity

Variants = estimator x selection rule (x gate):
  estimator  legacy (analytic simplex weights, shrunk toward equal) | bagged_cluster |
             ensemble_selection
  rule       greedy (`_coordinate_select` from the untuned columns, the engine's default) |
             bbc (bootstrap-validated selection, R2.b)
  gate       'off' = use the selection as is; 'on' = the BBC gate (R1.b) may send the task back to
             the untuned columns
Everything that CHOOSES (the selection, the gate, the weights) sees the scoring rows only; the
test rows are used once, to evaluate. Reported per task: test r and MSE of the weighted
ensemble, of the naive (equal-weight) mean of the SAME selected columns, of the untuned seed
columns' naive mean, and of the best single column (picked on the scoring rows).

Usage (run from the EasiGP source directory):

  python tools/joint_replay.py --dumps 'Result/*/Joint_library/*.npz' --out replay_out \
      [--bootstraps 200] [--estimators legacy,bagged_cluster,ensemble_selection] \
      [--rules greedy,bbc] [--weight-bags 100] [--seed 1]

Writes `replay_task_metrics.csv` (one row per task x variant), `replay_paired.csv` (paired
tables with a bootstrap 95 % CI and a one-sided Wilcoxon p), `Joint_replay_summary.csv` (the
ver4-38 measures above) and prints both decisions. `--skip-variants` computes the measures only
(seconds); `--members-per-model` sets the pruning of the library ceiling CL (default 3).
Reads library dumps of version 1 (ver4-36/37) and version 2 (ver4-38: adds `oof__k`).

ver4-38, R2.a: besides the variant replay the tool computes, per dump and with zero fits, the
measures that decide where the joint row is limited (written to `Joint_replay_summary.csv`, one
row per task plus a final `MEAN` row that carries the decision):
  (1) C6  ceiling: test r of simplex-NNLS weights fitted ON TEST to the K selected test columns
          (an in-sample upper bound - diagnostic only, never a result row)
  (2) CL  the same ceiling on the pruned library (the `per_model` best entries per model plus the
          selected one, `joint_tuning._prune_library_candidates`)
  (3) J_replay  ensemble-selection weights fitted on `y_score`, applied to the test columns
  (4) optimism  the scoring-row r minus the test r, for J and for each single model
  (5) effective models  exp(entropy(w)) of the J weights
  (6) dead columns  library columns that are constant (and how many selected ones are)
  (7) delta weights  ES weights on the library columns vs on the final-fidelity out-of-fold
          columns `oof__k` (dump version 2 only; NaN for a version-1 dump)
and the decision rule of the blueprint: ceiling gap g6 = C6 - max(F, S) (F = the single model with
the highest mean test r, S = the single model with the best scoring-row r per task); g6 < 0.01 =
column-limited (weighting cannot meet R2: invest in the SVR scale, the dead-column guard and the
library ensemble; the J-F criterion becomes non-inferiority); C6 - J >= 0.01 with J's optimism
above S's = weight / selection-limited (compare `selection_rule` and `weight_estimator` arms);
CL - C6 >= 0.005 = enable the library ensemble in the accuracy preset. The benchmark tool reads
`ceiling_gap_F` (C6 - F) from this file via `--ceiling-gap-f`.

Disclosure (blueprint R2.d): a replay cannot reproduce the proposal path - the Bayesian
optimisation was conditioned on the scores of the arm that wrote the dump - so a replay only
ranks selection and weighting rules on a FIXED library; the winners need real-run confirmation.
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ESTIMATORS = ('legacy', 'bagged_cluster', 'ensemble_selection')
RULES = ('greedy', 'bbc', 'weight_first', 'weight_first_cv')   # ver4-40 R2: 'weight_first' is the
# A2.0 gate arm; ver4-42 R1: 'weight_first_cv' is the A1.0 gate arm (zero fits, CV dumps only)
WF_DEFAULT_RULES = ('uniform', 'analytic', 'shrunk:0.25', 'shrunk:0.5', 'shrunk:0.75', 'es')
# ver4-48 R1: 'neural_stack' is the A-NS.0 gate arm. It is accepted by --rules but is NOT part of the
# default RULES, so a default replay run does exactly what it did before ver4-48.
RULE_CHOICES = RULES + ('neural_stack',)
ANS0_MIN_TASKS = 20                   # ver4-48 A-NS.0: the gate needs at least this many CV dumps
ANS0_MIN_GROUPS = 2                   # ... from at least this many (population, phenotype) groups
ANS0_G1_MARGIN = 0.005                # G1: mean delta r vs the naive mean of the entry-0 columns
ANS0_G1_LOWER_QUANTILE = 10.0         # G1: one-sided 90 % bootstrap lower bound (percentile)
ANS0_G2_NONINF = -0.002               # G2: mean delta r vs the weight_first_cv replay (non-inferior)
ANS0_G3_LOSS = -0.02                  # G3: a task "loses" when delta r vs naive is below this
ANS0_G3_MAX_SHARE = 0.15              # G3: at most this share of tasks may lose
ANS0_BOOTSTRAPS = 2000                # G1 bootstrap resamples (seed ANS0_BOOT_SEED)
ANS0_BOOT_SEED = 1
ANS0_COLUMNS = ['task', 'n_tv', 'n_test', 'C', 'lambda', 'r_ns', 'r_naive0', 'r_wfcv', 'r_best0',
                'mse_ns', 'mse_naive0', 'mse_wfcv', 'mse_best0', 'n_test_cols_dropped']
A20_MIN_TASKS = 20                    # ver4-40 A2.0: the gate needs at least this many task dumps
A20_MARGIN_VS_J = 0.005               # ver4-40 A2.0 (iii): weight_first must beat J by this much
DUMP_VERSION_KNOWN = 2      # ver4-38: dump version 2 adds `oof__k` and `meta['dump_version']`
A10_MIN_TASKS = 20                    # ver4-42 A1.0: the gate needs at least this many CV dumps
A10_MARGIN_VS_NAIVE = 0.005           # ver4-42 A1.0 (i): weight_first_cv must beat naive by this
A10_NONINF_COLUMN_LIMITED = -0.002    # ver4-42 A1.0 (i) when g6 < 0.01: non-inferiority bound
A10_VALID_FRACTION = 0.25             # n_valid / (n_train + n_valid) of a 60/20/20 split
DP_GRID_STEPS = 20                    # ver4-42 R2: the grid step is 1/DP_GRID_STEPS
DP_GRID_MAX_POINTS = 300000           # lower the steps when C(T+K-1, K-1) would exceed this
DP_GAP_MAX = 0.002                    # ver4-42 A2.1
DP_OPTIMISM_MIN = 0.01                # ver4-42 A2.2
CEILING_GAP_COLUMN_LIMITED = 0.01     # g6 below this: column-limited
CEILING_GAP_WEIGHT_LIMITED = 0.01     # C6 - J at or above this: weight / selection-limited
CEILING_GAP_LIBRARY = 0.005           # CL - C6 at or above this: enable the library ensemble


def load_dump(path):
    """Read one library dump (see `joint_tuning._write_library_dump` for the layout) into a
    plain dict: `models`, `keys`, `y_score`, `y_test`, `fold`, `meta` (dict), `lib`
    ({model: [(x, scoring column), ...]}), `test` ({model: (n_test, entries)}), `selected`
    ({model: index or -1}), `oof` ({model: final-fidelity out-of-fold column}; empty for a
    version-1 dump), `version` (1 or 2, from `meta['dump_version']`), `path`. No pickles are
    read (`allow_pickle=False`). Raises ValueError for a dump version newer than this tool
    knows."""
    z = np.load(path, allow_pickle=False)
    models = [str(m) for m in z['models']]
    keys = [str(k) for k in z['keys']]
    d = {'path': path, 'models': models, 'keys': keys, 'y_score': np.asarray(z['y_score'], float),
         'y_test': np.asarray(z['y_test'], float), 'fold': np.asarray(z['fold'], int),
         'meta': json.loads(str(z['meta'])), 'lib': {}, 'test': {}, 'selected': {}, 'oof': {}}
    d['version'] = int(d['meta'].get('dump_version', 1))
    if d['version'] > DUMP_VERSION_KNOWN:
        raise ValueError(f"library dump version {d['version']} is newer than this tool "
                         f"(knows up to {DUMP_VERSION_KNOWN}) - update tools/joint_replay.py")
    for m, k in zip(models, keys):
        xs, cols = z[f'x__{k}'], z[f'cols__{k}']
        d['lib'][m] = [(np.asarray(xs[i], float), np.asarray(cols[:, i], float))
                       for i in range(cols.shape[1])]
        d['test'][m] = np.asarray(z[f'test__{k}'], float)
        d['selected'][m] = int(z[f'selected__{k}'])
        if f'oof__{k}' in z.files:
            d['oof'][m] = np.asarray(z[f'oof__{k}'], float)
    return d


def _partition(fold):
    """CV partition (list of held-out position arrays) from the dump's `fold` vector, or None
    for the validation objective (all -1)."""
    if fold.size == 0 or int(fold.max()) < 0:
        return None
    return [np.flatnonzero(fold == f) for f in range(int(fold.max()) + 1)]


def _scorer(jt, dump, estimator, est_settings):
    """The engine's `_ColumnScorer` for this dump's scoring settings and `estimator`."""
    meta = dump['meta']
    part = _partition(dump['fold'])
    ctx = {'actual': dump['y_score']}
    if part is not None and meta.get('objective_data') == 'cv_train_valid':
        ctx.update({'objective_data': 'cv_train_valid', 'partition': part,
                    'cv_seed': int(meta['inner_split_seed_base']) + jt._JOINT_CV_SEED_OFFSET})
    settings = {
        'objective_mode': meta.get('objective_mode', 'ensemble_score'),
        'diversity_penalty': meta.get('diversity_penalty', 0.0),
        'diversity_method': meta.get('diversity_method', 'ridge_to_uniform'),
        'weight_rule': meta.get('score_weight_rule', 'analytic_penalised'),
        'shrinkage_alpha': meta.get('shrinkage_alpha', 0.85),
        'inner_split_seed_base': int(meta.get('inner_split_seed_base', 0)),
        'weight_estimator': estimator, 'estimator_settings': dict(est_settings),
    }
    return jt._ColumnScorer(dump['models'], ctx, settings, {'worst_finite': None})


def _final_weights(jt, estimator, P, y, est_settings, shrink_alpha, seed, n_bags):
    """Reported weights of `estimator` on the scoring columns `P`. legacy = the exact analytic
    simplex solution shrunk toward equal weights (the engine's default grading without its
    validation floor); the robust estimators = their own full function (no floor, no shrink)."""
    from models.ensemble_regularization import analytic_simplex_weights, apply_naive_shrinkage
    if jt.is_robust_estimator(estimator):
        w, _diag = jt._robust_weights_full(estimator, P, y, est_settings, n_bags=n_bags, seed=seed)
        return np.asarray(w, float)
    w = np.asarray(analytic_simplex_weights(P, y), float)
    return np.asarray(apply_naive_shrinkage(w, shrink_alpha), float)


def _r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or a.std() <= 0 or b.std() <= 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def replay_task(dump, estimator, rule, *, bootstraps=200, weight_bags=100, seed=1,
                est_extra=None):
    """Replay one dump for one (estimator, selection rule). Returns a list of two row dicts
    (gate 'off' and 'on'), or `[]` when the dump cannot be replayed (missing test columns,
    a single model). Pure function of the dump and the arguments (deterministic)."""
    from models import joint_tuning as jt
    from models.ensemble_regularization import analytic_simplex_weights
    models, lib = dump['models'], dump['lib']
    y, y_test = dump['y_score'], dump['y_test']
    if len(models) < 2 or y_test.size < 3:
        return []
    est_settings = {'cluster_corr': 0.97, 'score_bags': 10, 'weight_bags': int(weight_bags),
                    **(est_extra or {})}
    scorer = _scorer(jt, dump, estimator, est_settings)
    shrink = float(dump['meta'].get('shrinkage_alpha', 0.85))
    tol = float(dump['meta'].get('block_tol', jt._JOINT_BLOCK_TOL_DEFAULT))
    zero = {m: 0 for m in models}
    # --- greedy selection, visiting models in descending weight at the untuned columns ------
    P0 = scorer.matrix({m: lib[m][0][1] for m in models})
    w0 = np.asarray(analytic_simplex_weights(P0, y), float)
    order = [models[i] for i in sorted(range(len(models)), key=lambda i: (-w0[i], i))]
    greedy, _score, _swaps = jt._coordinate_select(
        lib, zero, scorer.score, order, max_passes=jt._JOINT_LIBRARY_MAX_PASSES, tol=tol)
    # --- BBC analysis (gate + selection) ----------------------------------------------------
    bbc = jt._bbc_analyse(
        lib, models, y, c_full=greedy, n_boot=int(bootstraps),
        seed=int(seed) + int(dump['meta'].get('inner_split_seed_base', 0)) + jt._JOINT_BBC_SEED_OFFSET,
        shrink_alpha=(shrink if dump['meta'].get('score_weight_rule') == 'shrunk' else None),
        tol=tol, quantile=float(dump['meta'].get('bbc_gate_quantile', 0.16)), margin=0.0)
    pick = dict(bbc['selected']) if (rule == 'bbc' and bbc['ok']) else dict(greedy)
    rows = []
    for gate in ('off', 'on'):
        use = dict(pick)
        accepted = True
        if gate == 'on':
            accepted = bool(bbc['ok'] and bbc['accepted'])
            if not accepted:
                use = dict(zero)
        cols = {m: lib[m][use[m]][1] for m in models}
        T = np.column_stack([dump['test'][m][:, use[m]] for m in models])
        if not np.all(np.isfinite(T)):
            continue
        P = scorer.matrix(cols)
        w = _final_weights(jt, estimator, P, y, est_settings, shrink, int(seed), int(weight_bags))
        pred = T @ w
        naive = T.mean(axis=1)
        seed_T = np.column_stack([dump['test'][m][:, 0] for m in models])
        singles_score_r = [_r(P[:, j], y) for j in range(len(models))]
        j_best = int(np.nanargmax(singles_score_r)) if np.isfinite(singles_score_r).any() else 0
        rows.append(dict(
            task=os.path.splitext(os.path.basename(dump['path']))[0], estimator=estimator,
            rule=rule, gate=gate, gate_accepted=accepted, n_swaps_vs_untuned=int(
                sum(use[m] != 0 for m in models)),
            r=_r(pred, y_test), mse=float(np.mean((pred - y_test) ** 2)),
            r_naive=_r(naive, y_test), mse_naive=float(np.mean((naive - y_test) ** 2)),
            r_naive_untuned=_r(seed_T.mean(axis=1), y_test),
            r_best_single=_r(T[:, j_best], y_test), best_single_model=models[j_best],
            weight_max=float(np.max(w)), weight_entropy=float(
                -np.sum(np.where(w > 0, w * np.log(np.where(w > 0, w, 1.0)), 0.0)))))
    return rows


def replay_weight_first(dump, *, rules=WF_DEFAULT_RULES, max_n=6, dedup_tol=0.05,
                        include_untuned=True, pick='bbc', bootstraps=200, weight_bags=100,
                        seed=1):
    """ver4-40, R2 step 1 (the A2.0 gate arm): replay the WEIGHT-FIRST rule on one dump, zero
    fits. Returns a list with ONE row dict (same columns as `replay_task`, `estimator='fixed'`,
    `rule='weight_first'`, `gate='off'`, plus `wf_choice` and `wf_untuned_won`), or `[]` when the
    dump cannot be replayed (a single model, no usable test columns, fewer than two distinct
    weight candidates after de-duplication - the engine would then run `async_library`).

    Procedure (everything that CHOOSES sees the scoring rows only, I7; the test rows evaluate
    once at the end):
      1. W = `weight_candidates()` of the UNTUNED (entry 0) columns on `y_score`; a dead untuned
         column gets weight 0 in every vector.
      2. for every vector w_c: `_coordinate_select` from the untuned columns under the FIXED
         weights (`_FixedWeightScorer`) -> its selection ref_c and score s_c. (The engine's
         successive halving over C and the per-candidate Bayesian proposals are NOT replayed:
         a replay only ranks selection and weighting on the fixed library.)
      3. finalists = every (c, ref_c, 'tuned') plus, with `include_untuned`, every
         (c, untuned, 'untuned').
      4. the pick: `pick='bbc'` -> `joint_tuning._wf_bbc` (out-of-bag mean score; falls back to
         'max' with a NOTE when too few rows), `pick='max'` -> the highest full-row score.
      5. test r / MSE of `T[:, pick] @ w_pick`.
    `r_best_single` here is the best UNTUNED single column picked on the scoring rows (the
    A2.0 (ii) comparator - the user's 'individual models')."""
    from models import joint_tuning as jt
    models, lib = dump['models'], dump['lib']
    y, y_test = dump['y_score'], dump['y_test']
    task = os.path.splitext(os.path.basename(dump['path']))[0]
    if len(models) < 2 or y_test.size < 3:
        return []
    meta = dump['meta']
    part = _partition(dump['fold'])
    est = dict(meta.get('estimator_settings') or {})
    es_settings = {'n_bags': int(est.get('score_bags', 10)),
                   'steps': int(est.get('es_steps', 0) or 0),
                   'model_fraction': float(est.get('es_model_fraction', 0.5)),
                   'cluster_corr': float(est.get('cluster_corr', 0.97))}
    inner_seed = int(meta.get('inner_split_seed_base', 0))
    tol = float(meta.get('block_tol', jt._JOINT_BLOCK_TOL_DEFAULT))
    P0 = np.column_stack([lib[m][0][1] for m in models])
    dead = [i for i in range(len(models)) if jt._is_dead_column(P0[:, i], y, part)]
    cands = jt.weight_candidates(P0, y, rules=list(rules), es_settings=es_settings,
                                 dedup_tol=float(dedup_tol), max_n=int(max_n),
                                 seed=int(seed) + inner_seed, dead=dead)
    if len(cands) < 2:
        print(f"[REPLAY] {task}: weight_first needs >= 2 distinct weight candidates, got "
              f"{len(cands)} - skipped (the engine would run async_library here).")
        return []
    ctx = {'actual': y}
    if part is not None and meta.get('objective_data') == 'cv_train_valid':
        ctx.update({'objective_data': 'cv_train_valid', 'partition': part})
    settings = {'objective_mode': meta.get('objective_mode', 'ensemble_score'),
                'dead_column_guard': True}
    zero = {m: 0 for m in models}
    finalists, scores = [], []
    for c, (_name, w) in enumerate(cands):
        scorer = jt._FixedWeightScorer(models, w, ctx, settings, {'worst_finite': None})
        order = [models[i] for i in sorted(range(len(models)), key=lambda i: (-w[i], i))]
        ref, s_c, _swaps = jt._coordinate_select(
            lib, zero, scorer.score, order, max_passes=jt._JOINT_LIBRARY_MAX_PASSES, tol=tol)
        finalists.append({'c': c, 'arm': 'tuned', 'pick': ref, 'w': w})
        scores.append(float(s_c))
        if include_untuned:
            finalists.append({'c': c, 'arm': 'untuned', 'pick': dict(zero), 'w': w})
            scores.append(float(scorer.score({m: lib[m][0][1] for m in models})))
    chosen, how = None, 'max'
    if pick in ('bbc', 'bbc_max'):
        # ver4-41: 'bbc' = honest out-of-bag pick + one-standard-error parsimony, 'bbc_max' = the
        # plain highest mean; with no out-of-bag evidence the SIMPLEST finalist is taken (the
        # same rule as the engine, `joint_tuning._wf_simplest`).
        res = jt._wf_bbc(lib, finalists, jt._wf_rule_specs(cands), y,
                         n_boot=int(bootstraps), seed=int(seed) + inner_seed
                         + jt._JOINT_BBC_SEED_OFFSET, partition=part, es_settings=es_settings,
                         tol=tol, parsimony=(pick == 'bbc'))
        if res['ok']:
            chosen, how = int(res['selected']), pick
        else:
            chosen, how = jt._wf_simplest(finalists, dead), 'simplest'
            print(f"[REPLAY] {task}: weight_first BBC pick not formed ({res['reason']}) - "
                  f"using the simplest finalist instead.")
    if chosen is None:
        chosen = int(np.argmax(scores))
    fin = finalists[chosen]
    use, w = fin['pick'], np.asarray(fin['w'], float)
    T = np.column_stack([dump['test'][m][:, use[m]] for m in models]) \
        if all(dump['test'][m].ndim == 2 and dump['test'][m].shape[1] > use[m] for m in models) \
        else None
    seed_T = np.column_stack([dump['test'][m][:, 0] for m in models]) \
        if all(dump['test'][m].ndim == 2 and dump['test'][m].shape[1] > 0 for m in models) \
        else None
    if (T is None or seed_T is None or T.shape[0] != y_test.size
            or not (np.all(np.isfinite(T)) and np.all(np.isfinite(seed_T)))):
        return []
    pred = T @ w
    naive = T.mean(axis=1)
    singles_score_r = [_r(P0[:, j], y) for j in range(len(models))]
    j_best = int(np.nanargmax(singles_score_r)) if np.isfinite(singles_score_r).any() else 0
    return [dict(
        task=task, estimator='fixed', rule='weight_first', gate='off', gate_accepted=True,
        n_swaps_vs_untuned=int(sum(use[m] != 0 for m in models)),
        r=_r(pred, y_test), mse=float(np.mean((pred - y_test) ** 2)),
        r_naive=_r(naive, y_test), mse_naive=float(np.mean((naive - y_test) ** 2)),
        r_naive_untuned=_r(seed_T.mean(axis=1), y_test),
        r_best_single=_r(seed_T[:, j_best], y_test), best_single_model=models[j_best],
        weight_max=float(np.max(w)),
        weight_entropy=float(-np.sum(np.where(w > 0, w * np.log(np.where(w > 0, w, 1.0)), 0.0))),
        wf_choice=f"{cands[fin['c']][0]}/{fin['arm']}", wf_untuned_won=bool(fin['arm'] == 'untuned'),
        wf_pick=how, wf_n_candidates=len(cands))]


def a20_gate(tm, j_estimator_by_task, rng, g6=None):
    """ver4-40, A2.0: the go / no-go table of the weight-first search from the replay rows `tm`
    (a DataFrame with the `weight_first` rows and, per task, the J arm = `rule='greedy'`,
    `gate='off'` row of that task's configured estimator, named in `j_estimator_by_task`).

    PASS needs ALL of: (i) mean test r of weight_first >= the naive mean of the untuned columns;
    (ii) >= the best untuned single column picked on the scoring rows; (iii) >= the J arm +
    `A20_MARGIN_VS_J` with the paired bootstrap 95 % CI lower bound > 0 - on at least
    `A20_MIN_TASKS` tasks. Fewer tasks give `decision='NOT DECIDED'` (never a pass). `g6` (the
    ceiling gap of `Joint_replay_summary.csv`, optional): below 0.01 the library is column-
    limited and no weighting rule can help - the decision then reads 'FAIL (column-limited)'.
    Returns `(table DataFrame, decision str)`; one row per criterion."""
    wf = tm[(tm['rule'] == 'weight_first')].set_index('task')
    j_rows = {}
    for task, est in j_estimator_by_task.items():
        hit = tm[(tm['task'] == task) & (tm['rule'] == 'greedy') & (tm['gate'] == 'off')
                 & (tm['estimator'] == est)]
        if len(hit):
            j_rows[task] = float(hit['r'].iloc[0])
    tasks = [t for t in wf.index if t in j_rows]
    n = len(tasks)
    rows = []
    if n:
        r_wf = wf.loc[tasks, 'r'].astype(float)
        for label, ref, margin in (
                ('(i) vs naive mean of untuned columns', wf.loc[tasks, 'r_naive_untuned'], 0.0),
                ('(ii) vs best untuned single column (scoring rows)', wf.loc[tasks, 'r_best_single'], 0.0),
                ('(iii) vs J arm (greedy + configured estimator)',
                 pd.Series([j_rows[t] for t in tasks], index=tasks), A20_MARGIN_VS_J)):
            res = paired(r_wf, ref.astype(float), rng)
            ok = bool(np.isfinite(res['mean']) and res['mean'] >= margin
                      and (margin == 0.0 or res['ci_lo'] > 0.0))
            rows.append(dict(criterion=label, n=res['n'], mean_delta_r=res['mean'],
                             ci_lo=res['ci_lo'], ci_hi=res['ci_hi'], required_min_delta=margin,
                             passed=ok))
    table = pd.DataFrame(rows)
    if n < A20_MIN_TASKS:
        decision = (f"NOT DECIDED: {n} task dump(s) with a weight_first row, A2.0 needs at least "
                    f"{A20_MIN_TASKS} (run the user's joint config with library_dump=True on more "
                    f"tasks).")
    elif g6 is not None and np.isfinite(g6) and g6 < CEILING_GAP_COLUMN_LIMITED:
        decision = (f"FAIL (column-limited): g6 = {g6:+.4f} < {CEILING_GAP_COLUMN_LIMITED} - no "
                    f"weighting of these columns can beat the best single model; record 'no "
                    f"change warranted' for R2.")
    elif len(table) == 3 and bool(table['passed'].all()):
        decision = "PASS: all three A2.0 criteria hold - implement the engine strategy (S7-S9)."
    else:
        decision = "FAIL: at least one A2.0 criterion does not hold - record the table; skip S7-S8."
    return table, decision


def simplex_nnls(P, y):
    """Exact non-negative, sum-to-one least squares: argmin ||P w - y||^2 s.t. w >= 0, sum w = 1.

    Why a local solver: the ceilings C6 / CL are UPPER BOUNDS and must not be biased by the
    1e-6 ridge that the engine's `analytic_simplex_weights` adds for conditioning. SLSQP with a
    tight tolerance, started at the engine's solution, gives the optimum to well below 1e-8 in
    the weights for the sizes met here (K <= ~30). Returns a length-K vector; falls back to
    the start point (a NOTE is not needed - it is still feasible) if the solver fails. Pure."""
    from scipy.optimize import minimize
    from models.ensemble_regularization import analytic_simplex_weights
    P = np.asarray(P, float)
    y = np.asarray(y, float)
    k = P.shape[1]
    if k == 1:
        return np.ones(1)
    w0 = np.asarray(analytic_simplex_weights(P, y), float)
    gram, rhs = P.T @ P, P.T @ y

    def fun(w):
        return float(0.5 * w @ gram @ w - rhs @ w)

    def jac(w):
        return gram @ w - rhs

    res = minimize(fun, w0, jac=jac, method='SLSQP', bounds=[(0.0, 1.0)] * k,
                   constraints=[{'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0),
                                 'jac': lambda w: np.ones(k)}],
                   options={'ftol': 1e-15, 'maxiter': 2000})
    w = np.clip(np.asarray(res.x, float), 0.0, None)
    total = float(w.sum())
    if not res.success and fun(w0) <= fun(w / total if total > 0 else w0):
        return w0
    return w / total if total > 0 and np.isfinite(total) else w0


def _entropy_effective(w):
    """exp(Shannon entropy in nats) of the normalised absolute weights: the effective number of
    models (1 = one-hot, K = equal weights)."""
    a = np.abs(np.asarray(w, float))
    if a.sum() <= 0:
        return np.nan
    a = a / a.sum()
    return float(np.exp(-np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0)), 0.0))))


def ceiling_measures(dump, *, per_model=3, weight_bags=100, weight_seed=None):
    """ver4-38, R2.a: the replay measures (1)-(7) for ONE dump, as a flat dict (one row of
    `Joint_replay_summary.csv`). Zero fits; everything that CHOOSES (the ES weights, the library
    pruning) sees the scoring rows only, the test rows only evaluate - except the two ceilings
    C6 / CL, which by definition fit weights on the test rows (diagnostics, never a result).

    Keys: `task`, `n_models`, `selected_in_library` (every model's selected entry was found),
    `r_J` / `mse_J` (J_replay: ES weights on `y_score` applied to the test columns of the selected
    entries), `r_C6`, `r_CL`, `n_pruned`, `r_S` / `S_model` (best scoring-row r), `r_O` / `O_model`
    (oracle, report only), `optimism_J`, `optimism_S`, `optimism_singles_mean` (scoring r - test
    r), `effective_models`, `dead_columns` / `dead_selected` / `library_columns`, `delta_w_l1`
    / `delta_w_max` (NaN without `oof__k`), and `r_test__<model>` per single model (for F).
    Test-dependent keys are NaN when the dump has no usable test columns (written with the
    stub). Never raises for missing optional parts."""
    from models import joint_tuning as jt
    meta = dump['meta']
    models, lib = dump['models'], dump['lib']
    y, y_test = dump['y_score'], dump['y_test']
    row = dict(task=os.path.splitext(os.path.basename(dump['path']))[0], n_models=len(models),
               dump_version=dump['version'])
    sel = {m: int(dump['selected'].get(m, -1)) for m in models}
    row['selected_in_library'] = bool(all(v >= 0 for v in sel.values()))
    use = {m: (sel[m] if sel[m] >= 0 else 0) for m in models}
    P = np.column_stack([lib[m][use[m]][1] for m in models])
    est_settings = {'cluster_corr': 0.97, 'score_bags': 10, **dict(meta.get('estimator_settings')
                                                                   or {}),
                    'weight_bags': int(weight_bags)}
    seed = (int(weight_seed) if weight_seed is not None
            else int(meta.get('seed', 0)) + int(meta.get('inner_split_seed_base', 0)))
    w, _diag = jt._robust_weights_full('ensemble_selection', P, y, est_settings,
                                       n_bags=int(weight_bags), seed=seed)
    w = np.asarray(w, float)
    row['effective_models'] = _entropy_effective(w)
    row['weight_max'] = float(np.max(w))
    # (6) dead columns (an out-of-fold column is dead when every fold block is constant)
    part = _partition(dump['fold'])
    flat = [(m, i) for m in models for i in range(len(lib[m]))]
    dead = [(m, i) for m, i in flat if jt._is_dead_column(lib[m][i][1], y, part)]
    row['library_columns'] = len(flat)
    row['dead_columns'] = len(dead)
    row['dead_selected'] = int(sum(jt._is_dead_column(lib[m][use[m]][1], y, part)
                                   for m in models))
    # (7) delta weights, dump version 2
    row['delta_w_l1'] = row['delta_w_max'] = np.nan
    if all(m in dump['oof'] and len(dump['oof'][m]) == len(y) for m in models):
        P_oof = np.column_stack([dump['oof'][m] for m in models])
        w_oof, _d2 = jt._robust_weights_full('ensemble_selection', P_oof, y, est_settings,
                                             n_bags=int(weight_bags), seed=seed)
        w_oof = np.asarray(w_oof, float)
        row['delta_w_l1'] = float(0.5 * np.abs(w - w_oof).sum())
        row['delta_w_max'] = float(np.abs(w - w_oof).max())
    # single models on the scoring rows (always available)
    r_score = [_r(P[:, j], y) for j in range(len(models))]
    finite_score = [v if np.isfinite(v) else -np.inf for v in r_score]
    j_s = int(np.argmax(finite_score))
    row['S_model'] = models[j_s]
    row['r_J_score'] = _r(P @ w, y)
    # test-dependent measures
    nan_keys = ('r_J', 'mse_J', 'r_C6', 'r_CL', 'n_pruned', 'r_S', 'r_O', 'optimism_J',
                'optimism_S', 'optimism_singles_mean')
    for k in nan_keys:
        row[k] = np.nan
    row['O_model'] = None
    T = np.column_stack([dump['test'][m][:, use[m]] for m in models]) \
        if all(dump['test'][m].ndim == 2 and dump['test'][m].shape[1] > use[m] for m in models) \
        else None
    if T is None or y_test.size < 3 or T.shape[0] != y_test.size or not np.all(np.isfinite(T)):
        return row
    r_test = [_r(T[:, j], y_test) for j in range(len(models))]
    for m, v in zip(models, r_test):
        row[f'r_test__{m}'] = v
    pred = T @ w
    row['r_J'] = _r(pred, y_test)
    row['mse_J'] = float(np.mean((pred - y_test) ** 2))
    row['r_S'] = r_test[j_s]
    finite_test = [v if np.isfinite(v) else -np.inf for v in r_test]
    j_o = int(np.argmax(finite_test))
    row['O_model'], row['r_O'] = models[j_o], r_test[j_o]
    row['optimism_J'] = row['r_J_score'] - row['r_J']
    row['optimism_S'] = r_score[j_s] - r_test[j_s]
    opt = [a - b for a, b in zip(r_score, r_test) if np.isfinite(a) and np.isfinite(b)]
    row['optimism_singles_mean'] = float(np.mean(opt)) if opt else np.nan
    # (1) C6: ceiling on the selected test columns
    row['r_C6'] = _r(T @ simplex_nnls(T, y_test), y_test)
    # (2) CL: ceiling on the pruned library (entries with a recorded test column)
    cand = jt._prune_library_candidates(models, lib, sel, y, per_model=int(per_model),
                                        dead_guard=True, partition=part)
    cols = []
    for m, i in cand:
        tc = dump['test'][m]
        if tc.ndim == 2 and tc.shape[1] > i and np.all(np.isfinite(tc[:, i])) \
                and tc.shape[0] == y_test.size:
            cols.append(tc[:, i])
    row['n_pruned'] = len(cols)
    if len(cols) >= 2:
        TL = np.column_stack(cols)
        row['r_CL'] = _r(TL @ simplex_nnls(TL, y_test), y_test)
    return row


def decision_rule(rows):
    """ver4-38, R2.a: aggregate the per-task measures (a DataFrame from `ceiling_measures`) into
    the MEAN row and apply the blueprint decision rule. Returns `(mean_row dict, [sentences])`.

    F = the single model with the highest MEAN test r over the tasks (best on average); S = the
    per-task single model with the best scoring-row r, scored on test. g6 = C6 - max(F, S)."""
    df = pd.DataFrame(rows)
    out = {'task': 'MEAN', 'n_tasks': int(len(df))}
    for k in ('r_J', 'mse_J', 'r_C6', 'r_CL', 'r_S', 'r_O', 'optimism_J', 'optimism_S',
              'optimism_singles_mean', 'effective_models', 'weight_max', 'dead_columns',
              'dead_selected', 'library_columns', 'delta_w_l1', 'delta_w_max', 'n_pruned'):
        out[k] = float(df[k].mean(skipna=True)) if k in df and df[k].notna().any() else np.nan
    test_cols = [c for c in df.columns if c.startswith('r_test__')]
    f_means = {c[len('r_test__'):]: float(df[c].mean(skipna=True)) for c in test_cols
               if df[c].notna().any()}
    f_model = max(f_means, key=f_means.get) if f_means else None
    out['F_model'] = f_model
    out['r_F'] = f_means[f_model] if f_model else np.nan
    notes = []
    if not np.isfinite(out['r_C6']) or f_model is None:
        notes.append("no dump carries test columns (written with the stub, or a single model) - "
                     "the ceilings cannot be computed; rerun the arm with library_dump=True.")
        out.update(ceiling_gap_F=np.nan, g6=np.nan, decision='; '.join(notes))
        return out, notes
    out['ceiling_gap_F'] = out['r_C6'] - out['r_F']
    out['g6'] = out['r_C6'] - max(out['r_F'], out['r_S'])
    out['gap_C6_J'] = out['r_C6'] - out['r_J']
    out['gap_CL_C6'] = (out['r_CL'] - out['r_C6']) if np.isfinite(out['r_CL']) else np.nan
    fired = []
    if out['g6'] < CEILING_GAP_COLUMN_LIMITED:
        fired.append('column-limited')
        notes.append(
            f"g6 = C6 - max(F, S) = {out['g6']:+.4f} < {CEILING_GAP_COLUMN_LIMITED}: COLUMN-"
            f"LIMITED - no weighting of these columns can meet R2. Invest in the SVR scale "
            f"(R3), the dead-column guard (R2.b1) and the library ensemble (R2.b2); the J-F "
            f"criterion becomes non-inferiority (pass --ceiling-gap-f {out['ceiling_gap_F']:.4f} "
            f"to tools/joint_benchmark.py).")
    if out['gap_C6_J'] >= CEILING_GAP_WEIGHT_LIMITED and out['optimism_J'] > out['optimism_S']:
        fired.append('weight/selection-limited')
        notes.append(
            f"C6 - J = {out['gap_C6_J']:+.4f} >= {CEILING_GAP_WEIGHT_LIMITED} and J's optimism "
            f"({out['optimism_J']:+.4f}) exceeds S's ({out['optimism_S']:+.4f}): WEIGHT / "
            f"SELECTION-LIMITED - compare the selection_rule and weight_estimator arms in the "
            f"replay variants (run without --skip-variants).")
    if np.isfinite(out['gap_CL_C6']) and out['gap_CL_C6'] >= CEILING_GAP_LIBRARY:
        fired.append('enable library_ensemble')
        notes.append(
            f"CL - C6 = {out['gap_CL_C6']:+.4f} >= {CEILING_GAP_LIBRARY}: the pruned library "
            f"holds more than the selected columns - ENABLE library_ensemble (R2.b2) in the "
            f"accuracy preset.")
    if not fired:
        notes.append(
            f"no rule fires (g6 = {out['g6']:+.4f}, C6 - J = {out['gap_C6_J']:+.4f}, CL - C6 = "
            f"{out['gap_CL_C6']:+.4f}): the measured gaps are below the thresholds - R2.b may "
            f"conclude 'no change warranted' for the component whose gap is < 0.005.")
    out['decision'] = '; '.join(fired) if fired else 'no rule fires'
    return out, notes


def write_summary(dumps_rows, out_dir):
    """Write `Joint_replay_summary.csv` (one row per task plus the final MEAN row), print the
    decision sentences and return `(summary DataFrame, notes)`. Empty input writes nothing."""
    if not dumps_rows:
        return pd.DataFrame(), []
    mean_row, notes = decision_rule(dumps_rows)
    df = pd.concat([pd.DataFrame(dumps_rows), pd.DataFrame([mean_row])], ignore_index=True)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'Joint_replay_summary.csv')
    df.to_csv(path, index=False)
    show = ['task', 'r_J', 'r_C6', 'r_CL', 'r_S', 'r_O', 'optimism_J', 'optimism_S',
            'effective_models', 'dead_columns', 'delta_w_l1']
    print(df[[c for c in show if c in df.columns]].to_string(
        index=False, float_format=lambda v: f'{v:.4f}'))
    for sentence in notes:
        print(f"[REPLAY] DECISION: {sentence}")
    print(f"[REPLAY] wrote {path}")
    return df, notes


def paired(a, b, rng, n_boot=2000):
    """Paired a - b: mean delta, bootstrap 95 % CI and one-sided Wilcoxon p (a > b)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return dict(n=int(d.size), mean=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_greater=np.nan)
    boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    try:
        import warnings
        from scipy.stats import wilcoxon
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = float(wilcoxon(d, alternative='greater').pvalue)
    except ValueError:  # all differences are zero
        p = float('nan')
    return dict(n=int(d.size), mean=float(d.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), p_greater=p)


def replay_weight_first_cv(dump, *, direction='analytic', alpha_rule='min', repeats=4,
                           alphas=None, n_valid_frac=A10_VALID_FRACTION, bootstraps=200,
                           seed=1):
    """ver4-42, R1 (the A1.0 gate arm): replay stage S1 (+ the S3 decision) of
    `search_strategy='weight_first_cv'` on one dump, zero fits. Returns a list with ONE row
    dict, or `[]` when the dump cannot be replayed (not a CV dump, a single model, no usable
    entry-0 test columns). Pure function of the dump and the arguments (deterministic).

    Procedure (everything that CHOOSES sees the scoring rows only, I7; the test rows evaluate
    once at the end):
      1. P0 = the entry-0 (untuned) out-of-fold columns over ALL scoring rows (train + valid).
      2. S1: `shrinkage_path_weights(P0, y_score)` -> (w*, alpha*). Dead untuned columns get 0.
      3. arm `r` / `mse`: test r / MSE of `T0 @ w*` (T0 = entry-0 test columns). Comparators on
         the same T0: the naive mean (`r_naive_untuned`), the single model with the best
         out-of-fold r (`r_best_single`) and the W_OPT analogue `r_wopt` (analytic weights fitted
         on the LAST n_valid scoring rows only - the rows a weighted-ensemble method sees).
         `n_valid` = `meta['n_valid']` when the dump has it, else `n_valid_frac` of the rows.
      4. S3 arm `r_s3` / `mse_s3`: finalists (untuned columns at w*) and (zero-fit
         `_coordinate_select` columns under w*), picked by `joint_tuning._wf_bbc` (plain mean,
         no parsimony) - what the engine's stage S3 would decide on this fixed library.
    Note: a CV library entry's test column is the AVERAGE of its V fold models' test
    predictions (pre-check P3); the bias is common to every arm of the table."""
    from models import joint_tuning as jt
    from models.ensemble_regularization import analytic_simplex_weights, shrinkage_path_weights
    models, lib = dump['models'], dump['lib']
    y, y_test = dump['y_score'], dump['y_test']
    task = os.path.splitext(os.path.basename(dump['path']))[0]
    meta = dump['meta']
    part = _partition(dump['fold'])
    if len(models) < 2 or y_test.size < 3:
        return []
    if part is None or meta.get('objective_data') != 'cv_train_valid':
        print(f"[REPLAY] {task}: weight_first_cv needs a dump written with objective_data="
              f"'cv_train_valid' - skipped.")
        return []
    T0 = (np.column_stack([dump['test'][m][:, 0] for m in models])
          if all(dump['test'][m].ndim == 2 and dump['test'][m].shape[1] > 0 for m in models)
          else None)
    if T0 is None or T0.shape[0] != y_test.size or not np.all(np.isfinite(T0)):
        print(f"[REPLAY] {task}: weight_first_cv needs finite entry-0 test columns - skipped.")
        return []
    inner_seed = int(meta.get('inner_split_seed_base', 0))
    est = dict(meta.get('estimator_settings') or {})
    es_settings = {'n_bags': int(est.get('score_bags', 10)),
                   'steps': int(est.get('es_steps', 0) or 0),
                   'model_fraction': float(est.get('es_model_fraction', 0.5)),
                   'cluster_corr': float(est.get('cluster_corr', 0.97))}
    P0 = np.column_stack([lib[m][0][1] for m in models])
    dead = [i for i in range(len(models)) if jt._is_dead_column(P0[:, i], y, part)]
    w, alpha, diag = shrinkage_path_weights(
        P0, y, direction=direction, alphas=alphas, rule=alpha_rule, n_repeats=int(repeats),
        seed=int(seed) + inner_seed, dead=dead, es_settings=es_settings)
    n = len(y)
    n_valid = int(meta.get('n_valid', round(float(n_valid_frac) * n)))
    n_valid = max(3, min(n_valid, n))
    live = [i for i in range(len(models)) if i not in dead]
    w_wopt = np.zeros(len(models))
    w_wopt[live] = analytic_simplex_weights(P0[-n_valid:, live], y[-n_valid:])
    naive = T0.mean(axis=1)
    r_oof = [_r(P0[:, j], y) for j in range(len(models))]
    j_best = int(np.nanargmax(r_oof)) if np.isfinite(r_oof).any() else 0
    # S3 arm: untuned@w* vs coordinate-select@w*, picked by the engine's BBC helper
    zero = {m: 0 for m in models}
    ctx = {'actual': y, 'objective_data': 'cv_train_valid', 'partition': part}
    scorer = jt._FixedWeightScorer(models, w, ctx, {'objective_mode': meta.get(
        'objective_mode', 'ensemble_score'), 'dead_column_guard': True}, {'worst_finite': None})
    order = [models[i] for i in sorted(range(len(models)), key=lambda i: (-w[i], i))]
    tol = float(meta.get('block_tol', jt._JOINT_BLOCK_TOL_DEFAULT))
    ref, _s, _sw = jt._coordinate_select(lib, zero, scorer.score, order,
                                         max_passes=jt._JOINT_LIBRARY_MAX_PASSES, tol=tol)
    finalists = [{'c': 0, 'arm': 'untuned', 'pick': dict(zero), 'w': w},
                 {'c': 0, 'arm': 'tuned', 'pick': ref, 'w': w}]
    s3_arm, s3_pick = 'untuned', dict(zero)
    res = jt._wf_bbc(lib, finalists, [list(w)], y, n_boot=int(bootstraps),
                     seed=int(seed) + inner_seed + jt._JOINT_BBC_SEED_OFFSET, partition=part,
                     es_settings=es_settings, tol=tol, parsimony=False)
    if res['ok']:
        s3_arm = finalists[int(res['selected'])]['arm']
        s3_pick = finalists[int(res['selected'])]['pick']
    T3 = (np.column_stack([dump['test'][m][:, s3_pick[m]] for m in models])
          if all(dump['test'][m].ndim == 2 and dump['test'][m].shape[1] > s3_pick[m]
                 for m in models) else T0)
    if not np.all(np.isfinite(T3)):
        T3, s3_arm = T0, 'untuned'
    pred, pred_w, pred_3 = T0 @ w, T0 @ w_wopt, T3 @ w
    return [dict(
        task=task, rule='weight_first_cv', direction=direction, alpha_rule=alpha_rule,
        n_rows=n, n_valid_used=n_valid, n_models=len(models), n_dead=len(dead),
        alpha=float(alpha), r=_r(pred, y_test), mse=float(np.mean((pred - y_test) ** 2)),
        r_naive_untuned=_r(naive, y_test), mse_naive=float(np.mean((naive - y_test) ** 2)),
        r_best_single=_r(T0[:, j_best], y_test), best_single_model=models[j_best],
        r_wopt=_r(pred_w, y_test), mse_wopt=float(np.mean((pred_w - y_test) ** 2)),
        r_s3=_r(pred_3, y_test), mse_s3=float(np.mean((pred_3 - y_test) ** 2)), s3_arm=s3_arm,
        weight_max=float(np.max(w)), effective_models=_entropy_effective(w),
        note=str(diag.get('note', '')))]


def a10_gate(tm, rng, g6=None, f_model_means=None):
    """ver4-42, A1.0: the go / no-go table of `weight_first_cv` from the `replay_weight_first_cv`
    rows `tm` (a DataFrame, one row per task). Zero fits.

    PASS needs ALL of, on at least `A10_MIN_TASKS` CV dumps (mean test r over tasks):
      (i)   r(S1) >= naive(entry 0) + `A10_MARGIN_VS_NAIVE`, paired bootstrap 95 % CI lower bound
            > 0 (when `g6` < 0.01, i.e. the library is column-limited, this becomes
            non-inferiority: mean delta >= `A10_NONINF_COLUMN_LIMITED`);
      (ii)  r(S1) >= the best single model chosen on the out-of-fold rows (mean delta >= 0);
            `f_model_means` ({model: mean test r}, optional) adds the diagnostic row for the
            single model that is best ON AVERAGE (not part of the decision);
      (iii) r(S1) >= the W_OPT analogue (analytic weights on the last n_valid rows; mean >= 0).
    Fewer tasks give `decision='NOT DECIDED'` (never a pass). Returns `(table, decision)`."""
    rows = []
    if len(tm):
        r = tm['r'].astype(float)
        column_limited = g6 is not None and np.isfinite(g6) and g6 < CEILING_GAP_COLUMN_LIMITED
        for label, ref, margin, need_ci in (
                ('(i) vs naive mean of the untuned columns', tm['r_naive_untuned'],
                 A10_NONINF_COLUMN_LIMITED if column_limited else A10_MARGIN_VS_NAIVE,
                 not column_limited),
                ('(ii) vs best single model chosen on the OOF rows', tm['r_best_single'], 0.0,
                 False),
                ('(iii) vs W_OPT analogue (analytic, last n_valid rows)', tm['r_wopt'], 0.0,
                 False)):
            res = paired(r, ref.astype(float), rng)
            ok = bool(np.isfinite(res['mean']) and res['mean'] >= margin
                      and (not need_ci or res['ci_lo'] > 0.0))
            rows.append(dict(criterion=label, n=res['n'], mean_delta_r=res['mean'],
                             ci_lo=res['ci_lo'], ci_hi=res['ci_hi'], p_greater=res['p_greater'],
                             required_min_delta=margin, passed=ok))
        mse_ok = float(tm['mse'].mean()) <= float(tm['mse_naive'].mean())
        rows.append(dict(criterion='(info) mean MSE no worse than naive', n=int(len(tm)),
                         mean_delta_r=float(tm['mse_naive'].mean() - tm['mse'].mean()),
                         ci_lo=np.nan, ci_hi=np.nan, p_greater=np.nan, required_min_delta=0.0,
                         passed=bool(mse_ok)))
        if f_model_means:
            best = max(f_model_means, key=f_model_means.get)
            rows.append(dict(criterion=f'(info) mean r(S1) - mean r of best-on-average model '
                             f'({best})', n=int(len(tm)),
                             mean_delta_r=float(r.mean() - f_model_means[best]), ci_lo=np.nan,
                             ci_hi=np.nan, p_greater=np.nan, required_min_delta=0.0,
                             passed=bool(r.mean() >= f_model_means[best])))
    table = pd.DataFrame(rows)
    n = len(tm)
    if n < A10_MIN_TASKS:
        decision = (f"NOT DECIDED: {n} CV task dump(s), A1.0 needs at least {A10_MIN_TASKS} (run "
                    f"the user's joint config with objective_data='cv_train_valid', "
                    f"library_dump=True, search_strategy='async_library', "
                    f"library_evals_per_dim=1 on more tasks).")
    elif len(table) >= 3 and bool(table['passed'].iloc[:3].all()):
        decision = ("PASS: all three A1.0 criteria hold - implement the engine strategy "
                    "(blueprint steps S4-S7).")
    else:
        decision = ("FAIL: at least one A1.0 criterion does not hold - record the table in the "
                    "Change Summary and skip S4-S7 (no engine code).")
    return table, decision


# ver4-50, R1/R2/R3: the four variants of gate A-NS.1 (blueprint SS2 R1 table). V0 is the ver4-49
# engine (the baseline every other variant is compared with); the order of `ANS1_ORDER` is the
# order in which the first GO variant is reported.
NS_VARIANTS = {
    'V0': dict(objective='ensemble_score', swap_rule='gate', shrink=0.5, deploy='argmax'),
    'V1': dict(objective='pearson', swap_rule='gate', shrink=0.5, deploy='argmax'),
    'V2': dict(objective='pearson', swap_rule='paired_se', shrink=0.5, deploy='argmax'),
    'V3': dict(objective='pearson', swap_rule='paired_se', shrink=0.5, deploy='mixture'),
    # ver4-51 (A-NS.2): V4 = V2 with the fair (refitted) swap test; V5 = V4 + the nested outer gate
    'V4': dict(objective='pearson', swap_rule='paired_refit', shrink=0.5, deploy='argmax'),
    'V5': dict(objective='pearson', swap_rule='paired_refit', shrink=0.5, deploy='argmax',
               outer_gate='nested'),
}
ANS1_ORDER = ('V2', 'V3', 'V1')
ANS2_ORDER = ('V5', 'V4')
ANS1_COLUMNS = ['variant'] + ANS0_COLUMNS
ANS2_COLUMNS = ANS1_COLUMNS + ['r_eq_dep', 'outer_decision']
NS_DIAG_COLUMNS = ['task', 'model', 'entry', 'w_V0', 'w_V1', 'oof_r', 'oof_slope', 'oof_sd_ratio',
                   'test_r', 'test_slope', 'test_sd_ratio', 'r_oof_naive0', 'r_test_naive0',
                   'gain_pearson', 'gain_mse', 'h1_mse_share', 'flips', 'flip_test_dr', 'row_type']
ANS1_H1_SHARE = 0.5                   # H1 is confirmed when the median MSE share is >= this


def _slope_and_ratio(p, y):
    """`(cov(y, p) / var(p), sd(p) / sd(y))` of one column against its target (NaN when either
    has no variance or fewer than 3 finite rows)."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    if int(ok.sum()) < 3:
        return np.nan, np.nan
    p, y = p[ok], y[ok]
    vp, vy = float(np.var(p)), float(np.var(y))
    if vp <= 0.0 or vy <= 0.0:
        return np.nan, np.nan
    cov = float(np.mean((p - p.mean()) * (y - y.mean())))
    return cov / vp, float(np.sqrt(vp / vy))


def _ns_prepare(dump):
    """The neural stack's input from one dump, shared by every variant: `P` (scoring rows) and `T`
    (test rows) hold every finite library entry of every model (models in dump order, entry 0
    first), `sizes`, `entry0_pos`, `col_model` / `col_entry` (the model name and the ORIGINAL entry
    index of each column) and `part`. Returns `None` (the reason printed) when the dump cannot be
    replayed: not a CV dump, fewer than two models or three test rows, a non-finite entry-0 scoring
    or test column, a test block that does not match the library."""
    models, lib = dump['models'], dump['lib']
    y, y_test = dump['y_score'], dump['y_test']
    meta = dump['meta']
    task = os.path.splitext(os.path.basename(dump['path']))[0]
    part = _partition(dump['fold'])
    if len(models) < 2 or y_test.size < 3:
        print(f"[REPLAY] {task}: neural_stack needs >= 2 models and >= 3 test rows - skipped.")
        return None
    if part is None or meta.get('objective_data') != 'cv_train_valid':
        print(f"[REPLAY] {task}: neural_stack needs a dump written with objective_data="
              f"'cv_train_valid' (every fold id >= 0) - skipped.")
        return None
    cols, test_cols, sizes, entry0_pos, col_model, col_entry = [], [], [], [], [], []
    for m in models:
        entries = lib[m]
        test_block = dump['test'][m]
        if (test_block.ndim != 2 or test_block.shape[0] != y_test.size
                or test_block.shape[1] != len(entries)):
            print(f"[REPLAY] {task}: neural_stack - the test block of {m!r} has shape "
                  f"{test_block.shape}, expected ({y_test.size}, {len(entries)}) - skipped.")
            return None
        keep = [i for i, (_x, col) in enumerate(entries)
                if col.size == y.size and bool(np.all(np.isfinite(col)))]
        if not keep or keep[0] != 0:
            print(f"[REPLAY] {task}: neural_stack - the untuned (entry 0) scoring column of {m!r} "
                  f"is missing or not finite - skipped.")
            return None
        if not bool(np.all(np.isfinite(test_block[:, 0]))):
            print(f"[REPLAY] {task}: neural_stack - the untuned (entry 0) test column of {m!r} is "
                  f"not finite - skipped.")
            return None
        entry0_pos.append(len(cols))
        sizes.append(len(keep))
        for i in keep:
            cols.append(entries[i][1])
            test_cols.append(test_block[:, i])
            col_model.append(m)
            col_entry.append(int(i))
    return {'task': task, 'P': np.column_stack(cols), 'T': np.column_stack(test_cols),
            'sizes': sizes, 'entry0_pos': entry0_pos, 'col_model': col_model,
            'col_entry': col_entry, 'part': part}


def _ns_dump_defaults(dump):
    """The neural-stack settings the ENGINE used for this dump (`meta['ns']`), as a dict with
    `lambdas` (finite floats), `include_prior`, `rule`, `repeats`, `folds`, `seed`, `inner_seed`;
    any value the dump does not carry falls back to the engine's own default (ver4-48 dumps have
    no `repeats` / `folds`)."""
    from models import joint_tuning as jt
    meta = dump['meta']
    ns = meta.get('ns') or {}
    raw = ns.get('lambdas')
    if raw:
        lambdas = [float(v) for v in raw if str(v) != 'inf']
        include_prior = any(str(v) == 'inf' for v in raw)
    else:
        lambdas, include_prior = list(jt._JOINT_NS_LAMBDAS_DEFAULT), True
    return {'lambdas': lambdas, 'include_prior': include_prior,
            'rule': str(ns.get('rule') or jt._JOINT_NS_LAMBDA_RULE_DEFAULT),
            'repeats': int(ns.get('repeats') or jt._JOINT_NS_REPEATS_DEFAULT),
            'folds': int(ns.get('folds') or meta.get('cv_folds') or 0),
            'seed': int(meta.get('seed', 1)),
            'inner_seed': int(meta.get('inner_split_seed_base', 0))}


def _ns_fit_variant(prep, y, *, lambdas, include_prior, rule, repeats, folds, seed, objective,
                    swap_rule, shrink, deploy, members_max, outer_gate='off', outer_folds=5,
                    outer_repeats=2, n_jobs=1):
    """One variant on the prepared dump. Since ver4-51 this CALLS the engine's own pure functions
    (`hyper_stack.stack_procedure`, and `hyper_stack.nested_gate` for ``outer_gate='nested'``)
    instead of mirroring them, so the replay cannot drift from `_neural_stack_search`. Returns a
    dict with `fit_raw` (the network at lambda*, before the swap rule and the shrink), `fit` (as
    deployed: after swap, shrink and - when the nested gate rejects - the equal-weight prior),
    `info` (the lambda path info), `swaps` (rows of the swap test, model positions), `w_cols` (the
    weight of every library column in the deployed prediction, `hyper_stack.
    deployed_column_weights`), `primary` and `outer` (the nested gate's dict or None). Zero fits.
    Raises ValueError / `HyperStackFitError` from the lambda path."""
    from models import joint_tuning as jt
    from models import hyper_stack as hs
    P, sizes = prep['P'], prep['sizes']
    score = jt._NS_SCORE_FNS[objective]
    ns_seed = int(seed) + int(jt._JOINT_NS_SEED_OFFSET)
    spec = hs.StackSpec(
        lambdas=tuple(float(v) for v in lambdas), include_prior=bool(include_prior),
        folds=int(folds), repeats=int(repeats), seed=ns_seed, rule=rule, objective=objective,
        nb_correction=True, swap_rule=swap_rule, shrink=float(shrink), deploy=deploy,
        member_min_weight=float(jt._JOINT_LIBRARY_MEMBER_MIN_WEIGHT), members_max=int(members_max),
        max_iter=jt._JOINT_NS_MAX_ITER, tol=jt._JOINT_NS_TOL,
        swap_seed_offset=jt._JOINT_NS_SWAP_SEED_OFFSET)
    fit, w_cols, proc = hs.stack_procedure(P, y, sizes, spec, score)
    info = proc['lambda_info']
    outer = None
    if outer_gate == 'nested' and include_prior and np.isfinite(float(fit.lam)):
        try:
            outer = dict(hs.nested_gate(
                P, y, sizes, spec, score, outer_folds=int(outer_folds),
                outer_repeats=int(outer_repeats), seed=ns_seed + jt._JOINT_NS_OUTER_SEED_OFFSET,
                n_jobs=n_jobs, z=jt._JOINT_NS_OUTER_Z))
        except (ValueError, hs.HyperStackFitError):
            outer = {'gain': float('nan'), 'se': float('nan'), 'admitted': False,
                     'reason': 'error'}
        if not outer['admitted']:
            fit = hs.prior_fit(sizes)
            w_cols, _members = hs.deployed_column_weights(
                fit, sizes, deploy, member_min_weight=float(jt._JOINT_LIBRARY_MEMBER_MIN_WEIGHT),
                members_max=int(members_max))
    primary = [int(np.argmax(np.asarray(g, float))) for g in fit.q]
    return {'fit_raw': proc['fit_lambda'], 'fit': fit, 'info': info, 'swaps': proc['swaps'],
            'w_cols': np.asarray(w_cols, dtype=float), 'primary': primary, 'outer': outer}


def ns_parity(dump):
    """ver4-50, AC1.1: replay the neural stack with the ENGINE's own settings for this dump (read
    from `meta['ns']`: objective, swap rule, shrink, lambdas, rule, repeats, folds, seed) and return
    `(max_abs_diff, per_model)` of the model weights `W` against the dump's `meta['ns']['W']`;
    `(nan, {})` when the dump has no neural-stack record or cannot be replayed. A difference above
    1e-9 means the replay no longer mirrors the engine. Zero fits."""
    from models import joint_tuning as jt
    meta = dump['meta']
    ns = meta.get('ns') or {}
    prep = _ns_prepare(dump)
    if prep is None or not ns.get('W'):
        return np.nan, {}
    d = _ns_dump_defaults(dump)
    shrink = float(ns.get('weight_shrink', jt._JOINT_NS_WEIGHT_SHRINK_DEFAULT))
    if not np.isfinite(float(ns.get('lambda', np.inf))):
        shrink = 0.0
    out = _ns_fit_variant(
        prep, dump['y_score'], lambdas=d['lambdas'], include_prior=d['include_prior'],
        rule=d['rule'], repeats=d['repeats'], folds=d['folds'] or len(prep['part']),
        seed=d['seed'] + d['inner_seed'],
        objective=str(ns.get('ns_objective', jt._JOINT_NS_OBJECTIVE_DEFAULT)),
        swap_rule=str(ns.get('swap_rule', jt._JOINT_NS_SWAP_RULE_DEFAULT)), shrink=shrink,
        deploy='argmax', members_max=jt._JOINT_LIBRARY_MEMBERS_MAX_DEFAULT,
        outer_gate=str(ns.get('ns_outer_gate', 'off')),
        outer_folds=int(ns.get('outer_folds', jt._JOINT_NS_OUTER_FOLDS_DEFAULT)),
        outer_repeats=int(ns.get('outer_repeats', jt._JOINT_NS_OUTER_REPEATS_DEFAULT)))
    per_model = {m: float(w) for m, w in zip(dump['models'], out['fit'].w_model)}
    diffs = [abs(per_model[m] - float(ns['W'][m])) for m in per_model if m in ns['W']]
    return (float(max(diffs)) if diffs else np.nan), per_model


def replay_neural_stack_variants(dump, variants, *, lambdas=None, rule=None, repeats=None,
                                 seed=None, wfcv_bootstraps=20, with_diag=False,
                                 outer_folds=5, outer_repeats=2, n_jobs=1):
    """ver4-50 (A-NS.1): replay several neural-stack variants (`{name: settings}`, see
    `NS_VARIANTS`) on one dump, zero fits. The comparators (`r_naive0`, `r_wfcv`, `r_best0`) are
    computed once. Returns `(rows, diag_rows)`: one row dict per variant (keys `ANS1_COLUMNS`) and,
    with `with_diag`, the `replay_ns_diag.csv` rows (`NS_DIAG_COLUMNS`; one per library column plus
    one task-level row, see `ns_diag_report`). `([], [])` when the dump cannot be replayed.

    `lambdas`, `rule`, `repeats` default to what the dump's own engine run used (`meta['ns']`) and
    otherwise to the engine defaults; `seed=None` uses the engine's seed of the dump (so the CV
    folds of the lambda choice are the engine's own), an int overrides it. Everything that CHOOSES
    sees the scoring rows only (I7); the test rows evaluate once.

    ver4-51: a variant may carry ``outer_gate='nested'`` (V5; ``outer_folds`` / ``outer_repeats`` /
    ``n_jobs`` as in `hyper_stack.nested_gate`) and every row also holds `r_eq_dep` (test r of the
    equal-weight mean of the DEPLOYED primary columns of that variant - the `ensemble` row of the
    engine) and `outer_decision` (the nested gate's verdict, '' without it)."""
    from models import joint_tuning as jt
    from models import hyper_stack as hs
    prep = _ns_prepare(dump)
    if prep is None:
        return [], []
    y, y_test = dump['y_score'], dump['y_test']
    P, T, sizes, entry0_pos = prep['P'], prep['T'], prep['sizes'], prep['entry0_pos']
    task = prep['task']
    d = _ns_dump_defaults(dump)
    lam_list = list(lambdas) if lambdas is not None else d['lambdas']
    rule_use = str(rule) if rule is not None else d['rule']
    reps = int(repeats) if repeats is not None else d['repeats']
    folds = d['folds'] or len(prep['part'])
    base_seed = (int(seed) if seed is not None else d['seed']) + d['inner_seed']
    members_max = int(dump['meta'].get('library_members_max',
                                       jt._JOINT_LIBRARY_MEMBERS_MAX_DEFAULT))
    finite_test = np.all(np.isfinite(T), axis=0)
    T0, P0 = T[:, entry0_pos], P[:, entry0_pos]
    naive0, naive0_oof = T0.mean(axis=1), P0.mean(axis=1)
    r_oof = [_r(P0[:, j], y) for j in range(P0.shape[1])]
    j_best = int(np.nanargmax(r_oof)) if np.isfinite(r_oof).any() else 0
    best0 = T0[:, j_best]
    wfcv = replay_weight_first_cv(dump, direction='analytic', alpha_rule='min', repeats=reps,
                                  bootstraps=int(wfcv_bootstraps), seed=(1 if seed is None
                                                                         else int(seed)))
    r_wfcv = wfcv[0]['r'] if wfcv else np.nan
    mse_wfcv = wfcv[0]['mse'] if wfcv else np.nan
    n_dropped = int(np.count_nonzero(~finite_test))
    rows, fits = [], {}
    for name, cfg in variants.items():
        try:
            out = _ns_fit_variant(
                prep, y, lambdas=lam_list, include_prior=True, rule=rule_use, repeats=reps,
                folds=folds, seed=base_seed, objective=cfg['objective'],
                swap_rule=cfg['swap_rule'], shrink=cfg['shrink'], deploy=cfg['deploy'],
                members_max=members_max, outer_gate=cfg.get('outer_gate', 'off'),
                outer_folds=outer_folds, outer_repeats=outer_repeats, n_jobs=n_jobs)
        except (ValueError, hs.HyperStackFitError) as exc:
            print(f"[REPLAY] {task}: neural_stack {name} - the network could not be fitted "
                  f"({type(exc).__name__}: {exc}) - skipped.")
            continue
        w_use = np.where(finite_test, out['w_cols'], 0.0)
        total = float(w_use.sum())
        if not np.isfinite(total) or total <= 0.0:
            print(f"[REPLAY] {task}: neural_stack {name} - every weighted test column is NaN - "
                  f"skipped.")
            continue
        pred = T[:, finite_test] @ (w_use[finite_test] / total)
        fits[name] = out
        # equal weights over the deployed primary columns (the engine's `ensemble` row)
        _starts = np.cumsum([0] + list(sizes))
        _prim = [int(_starts[m]) + int(pm) for m, pm in enumerate(out['primary'])
                 if float(out['fit'].w_model[m]) > 0.0 and bool(finite_test[int(_starts[m]) + int(pm)])]
        r_eq_dep = _r(T[:, _prim].mean(axis=1), y_test) if _prim else np.nan
        _outer = out.get('outer')
        outer_decision = ('' if _outer is None else
                          'admitted' if _outer['admitted'] else
                          (_outer['reason'] if _outer.get('reason') in ('too_small', 'error')
                           else 'rejected'))
        rows.append(dict(
            variant=name, task=task, n_tv=int(y.size), n_test=int(y_test.size), C=int(P.shape[1]),
            **{'lambda': float(out['info']['chosen'])},
            r_ns=_r(pred, y_test), r_naive0=_r(naive0, y_test), r_wfcv=r_wfcv,
            r_best0=_r(best0, y_test), mse_ns=float(np.mean((pred - y_test) ** 2)),
            mse_naive0=float(np.mean((naive0 - y_test) ** 2)), mse_wfcv=mse_wfcv,
            mse_best0=float(np.mean((best0 - y_test) ** 2)), n_test_cols_dropped=n_dropped,
            r_eq_dep=r_eq_dep, outer_decision=outer_decision))
        swaps = out['swaps']
        print(f"[REPLAY] {task}: neural_stack {name} lambda*={out['info']['chosen']:g} "
              f"({rule_use}), {len(sizes)} models / {P.shape[1]} columns, {n_dropped} NaN test "
              f"column(s) dropped, {out['info']['n_finite_ok']} finite lambda(s) eligible"
              + (f", swaps admitted {sum(1 for s in swaps if s['admitted'])}/{len(swaps)}"
                 if swaps else "") + ".")
    diag = []
    if with_diag and 'V0' in fits:
        diag = _ns_diag_rows(prep, y, y_test, fits, naive0, naive0_oof, finite_test)
    return rows, diag


def _ns_diag_rows(prep, y, y_test, fits, naive0, naive0_oof, finite_test):
    """The `replay_ns_diag.csv` rows of one task (hypotheses H1-H3 of the ver4-50 blueprint, R1):
    one row per library column (`row_type='column'`: network weight under V0 / V1, OOF and test
    r, slope `cov(y, p) / var(p)` and spread ratio `sd(p) / sd(y)`) and one task row
    (`row_type='task'`): `gain_pearson` / `gain_mse` = the in-sample gain of the V0 network over
    the naive mean of the entry-0 columns on the scoring rows, split into its Pearson and its
    `-mse/var` part (`h1_mse_share` = gain_mse / (gain_pearson + gain_mse), NaN when the total gain
    is not positive, e.g. lambda = infinity), `flips` = models whose V0 gate left entry 0 and
    `flip_test_dr` = mean over them of (test r of the chosen entry - test r of entry 0). Zero
    fits."""
    P, T, sizes = prep['P'], prep['T'], prep['sizes']
    task = prep['task']
    starts = np.cumsum([0] + list(sizes))
    f0 = fits['V0']['fit_raw']
    f1 = fits['V1']['fit_raw'] if 'V1' in fits else None
    out = []
    for c in range(P.shape[1]):
        o_slope, o_ratio = _slope_and_ratio(P[:, c], y)
        t_slope, t_ratio = _slope_and_ratio(T[:, c], y_test)
        out.append(dict(
            task=task, model=prep['col_model'][c], entry=prep['col_entry'][c],
            w_V0=float(f0.w[c]), w_V1=(float(f1.w[c]) if f1 is not None else np.nan),
            oof_r=_r(P[:, c], y), oof_slope=o_slope, oof_sd_ratio=o_ratio,
            test_r=_r(T[:, c], y_test) if bool(finite_test[c]) else np.nan,
            test_slope=t_slope, test_sd_ratio=t_ratio, row_type='column'))
    net = P @ np.asarray(f0.w, float)
    r_net, r_naive = _r(net, y), _r(naive0_oof, y)
    var_y = float(np.var(y))
    gain_p = r_net - r_naive if np.isfinite(r_net) and np.isfinite(r_naive) else np.nan
    gain_m = (float(np.mean((naive0_oof - y) ** 2)) - float(np.mean((net - y) ** 2))) / var_y \
        if var_y > 0 else np.nan
    total = gain_p + gain_m if np.isfinite(gain_p) and np.isfinite(gain_m) else np.nan
    share = float(gain_m / total) if np.isfinite(total) and total > 1e-12 else np.nan
    flip_dr = []
    for m, gate in enumerate(f0.q):
        l_star = int(np.argmax(np.asarray(gate, float)))
        if sizes[m] > 1 and l_star != 0:
            a, b = int(starts[m]) + l_star, int(starts[m])
            if bool(finite_test[a]) and bool(finite_test[b]):
                flip_dr.append(_r(T[:, a], y_test) - _r(T[:, b], y_test))
    n_flips = sum(1 for m, g in enumerate(f0.q)
                  if sizes[m] > 1 and int(np.argmax(np.asarray(g, float))) != 0)
    out.append(dict(
        task=task, model='*', entry=-1, r_oof_naive0=r_naive,
        r_test_naive0=_r(T[:, prep['entry0_pos']].mean(axis=1), y_test), gain_pearson=gain_p,
        gain_mse=gain_m, h1_mse_share=share, flips=int(n_flips),
        flip_test_dr=(float(np.nanmean(flip_dr)) if flip_dr else np.nan), row_type='task'))
    return out


def replay_neural_stack(dump, *, lambdas=None, rule=None, repeats=None, seed=None,
                        wfcv_bootstraps=20, objective='ensemble_score', swap_rule='gate',
                        shrink=0.5, deploy='argmax', outer_gate='off', outer_folds=5,
                        outer_repeats=2):
    """ver4-48, R1 (the A-NS.0 gate arm), ver4-50: replay ONE neural-stack variant on one dump,
    zero fits. Returns a list with ONE row dict (keys `ANS0_COLUMNS`), or `[]` (with the reason
    printed) when the dump cannot be replayed (see `_ns_prepare`).

    Since ver4-50 the replay mirrors the ver4-49 / ver4-50 engine (`_ns_fit_variant`): the
    Nadeau-Bengio one-SE rule, the optional paired swap test (`swap_rule`), the `shrink` towards
    equal weights and the deployment (`deploy`: 'argmax' = the model weight on each model's most
    probable entry, 'mixture' = the heavy library columns). The defaults are the ver4-49 engine
    with `ns_deploy='argmax'` (variant V0); ver4-48 replays (no shrink, no noise correction) are
    no longer produced - that was the parity defect D1 of the ver4-50 blueprint. `lambdas`, `rule`,
    `repeats` and `seed` default to the dump's own engine run (see `replay_neural_stack_variants`).
    Comparators on the same test rows: `r_naive0` (mean of the entry-0 columns), `r_wfcv` (the
    weight_first_cv replay) and `r_best0` (the entry-0 column with the best out-of-fold r)."""
    rows, _diag = replay_neural_stack_variants(
        dump, {'V': dict(objective=objective, swap_rule=swap_rule, shrink=shrink, deploy=deploy,
                         outer_gate=outer_gate)},
        lambdas=lambdas, rule=rule, repeats=repeats, seed=seed, wfcv_bootstraps=wfcv_bootstraps,
        outer_folds=outer_folds, outer_repeats=outer_repeats)
    return [{k: r[k] for k in ANS0_COLUMNS} for r in rows]


def _task_groups(tasks):
    """Distinct (population, phenotype) groups among dump names. The engine names a dump
    `<population>_<phenotype>_<ratio>_<sample>` (`genomic_prediction.py`, `diagnostics_ctx`), so
    dropping the last two '_' tokens leaves `<population>_<phenotype>`; a population or phenotype
    that itself contains '_' cannot be split further, which is why this counts groups (a proxy for
    the blueprint's 'phenotypes')."""
    groups = set()
    for t in tasks:
        parts = str(t).split('_')
        groups.add('_'.join(parts[:-2]) if len(parts) > 2 else str(t))
    return groups


def ans0_gate(tm):
    """ver4-48, A-NS.0: the go / no-go table of the neural stack from the `replay_neural_stack`
    rows `tm` (a DataFrame, one row per task). Zero fits. Delta is test Pearson r, paired by task.

    GO needs ALL of, on at least `ANS0_MIN_TASKS` dumps from at least `ANS0_MIN_GROUPS`
    (population, phenotype) groups:
      G1  mean delta(NS - naive mean of the entry-0 columns) >= `ANS0_G1_MARGIN` AND the one-sided
          90 % bootstrap lower bound (`ANS0_BOOTSTRAPS` resamples, seed `ANS0_BOOT_SEED`) > 0;
      G2  mean delta(NS - weight_first_cv replay) >= `ANS0_G2_NONINF`;
      G3  share of tasks with delta(NS - naive) < `ANS0_G3_LOSS` is <= `ANS0_G3_MAX_SHARE`;
      G4  mean delta(NS - best entry-0 single column picked on the scoring rows) >= 0.
    MSE deltas (positive = NS better) are reported as info rows and never gate. Fewer dumps or
    groups give 'NOT DECIDED' (never GO, and not a NO-GO either: the evidence is missing).
    Returns `(table, decision)`; `decision` starts with 'GO', 'NO-GO' or 'NOT DECIDED'."""
    rows = []
    n = len(tm)
    if n:
        def _delta(a, b):
            d = tm[a].astype(float).to_numpy() - tm[b].astype(float).to_numpy()
            return d[np.isfinite(d)]

        d1 = _delta('r_ns', 'r_naive0')
        boots = (np.random.default_rng(ANS0_BOOT_SEED).choice(
            d1, size=(ANS0_BOOTSTRAPS, d1.size), replace=True).mean(axis=1)
            if d1.size else np.array([np.nan]))
        lower = float(np.percentile(boots, ANS0_G1_LOWER_QUANTILE))
        mean1 = float(d1.mean()) if d1.size else np.nan
        g1 = bool(np.isfinite(mean1) and mean1 >= ANS0_G1_MARGIN and lower > 0.0)
        rows.append(dict(criterion='G1 NS - naive mean of entry-0 columns', n=int(d1.size),
                         value=mean1, bound=lower, required=ANS0_G1_MARGIN, passed=g1))
        d2 = _delta('r_ns', 'r_wfcv')
        mean2 = float(d2.mean()) if d2.size else np.nan
        rows.append(dict(criterion='G2 NS - weight_first_cv replay', n=int(d2.size), value=mean2,
                         bound=np.nan, required=ANS0_G2_NONINF,
                         passed=bool(np.isfinite(mean2) and mean2 >= ANS0_G2_NONINF)))
        share = float(np.mean(d1 < ANS0_G3_LOSS)) if d1.size else np.nan
        rows.append(dict(criterion=f'G3 share of tasks with NS - naive < {ANS0_G3_LOSS:g}',
                         n=int(d1.size), value=share, bound=np.nan, required=ANS0_G3_MAX_SHARE,
                         passed=bool(np.isfinite(share) and share <= ANS0_G3_MAX_SHARE)))
        d4 = _delta('r_ns', 'r_best0')
        mean4 = float(d4.mean()) if d4.size else np.nan
        rows.append(dict(criterion='G4 NS - best entry-0 column (picked on scoring rows)',
                         n=int(d4.size), value=mean4, bound=np.nan, required=0.0,
                         passed=bool(np.isfinite(mean4) and mean4 >= 0.0)))
        for label, ref in (('naive0', 'mse_naive0'), ('weight_first_cv', 'mse_wfcv'),
                           ('best0', 'mse_best0')):
            dm = tm[ref].astype(float).to_numpy() - tm['mse_ns'].astype(float).to_numpy()
            dm = dm[np.isfinite(dm)]
            rows.append(dict(criterion=f'(info) mean MSE reduction vs {label} (positive = NS better)',
                             n=int(dm.size), value=float(dm.mean()) if dm.size else np.nan,
                             bound=np.nan, required=np.nan, passed=np.nan))
    table = pd.DataFrame(rows)
    groups = _task_groups(tm['task']) if n else set()
    if n < ANS0_MIN_TASKS or len(groups) < ANS0_MIN_GROUPS:
        decision = (f"NOT DECIDED: {n} dump(s) from {len(groups)} (population, phenotype) group(s); "
                    f"A-NS.0 needs >= {ANS0_MIN_TASKS} dumps from >= {ANS0_MIN_GROUPS} groups "
                    f"written with objective_data='cv_train_valid' and test columns.")
    elif bool(table['passed'].iloc[:4].astype(bool).all()):
        g = table['value'].iloc[:4].tolist()
        decision = (f"GO: G1={g[0]:+.4f} G2={g[1]:+.4f} G3={g[2]:.3f} G4={g[3]:+.4f} - offer "
                    f"'neural_stack' in the GUI.")
    else:
        g = table['value'].iloc[:4].tolist()
        failed = [c.split(' ')[0] for c, ok in zip(table['criterion'].iloc[:4],
                                                    table['passed'].iloc[:4]) if not ok]
        decision = (f"NO-GO: {', '.join(failed)} failed (G1={g[0]:+.4f} G2={g[1]:+.4f} "
                    f"G3={g[2]:.3f} G4={g[3]:+.4f}) - hide 'neural_stack' from the GUI choices "
                    f"(one line, blueprint SS8).")
    return table, decision


def ans1_gate(rows):
    """ver4-50, gate A-NS.1: go / no-go of every neural-stack variant from the `replay_neural_stack_
    variants` rows `rows` (a DataFrame with a `variant` column, one row per task x variant). Zero
    fits. A variant is GO when the A-NS.0 criteria G1-G4 hold (`ans0_gate`, unchanged thresholds,
    on that variant's own rows) AND G5: the mean delta test Pearson r (variant - V0), paired by
    task, is >= 0. V0 (the ver4-49 engine) is the baseline: it gets G1-G4 and no G5, and is never
    the answer. MSE deltas are reported, never gating.

    Returns `(table, decisions, decision)`: `table` (one row per variant x criterion, columns
    `variant, criterion, n, value, bound, required, passed`), `decisions` ({variant: the
    `ans0_gate` text, with G5 appended}) and `decision`, the text after `A-NS.1 DECISION:` - the
    first GO variant in the order V2, V3, V1, `NONE` when the evidence suffices and no variant is
    GO, or `NOT DECIDED (...)` when `ans0_gate` lacks dumps or groups for the baseline."""
    rows = rows.copy()
    tables, decisions, go = [], {}, {}
    base = (rows[rows['variant'] == 'V0'].set_index('task') if 'V0' in set(rows['variant'])
            else None)
    undecided = None
    for name in [v for v in NS_VARIANTS if v in set(rows['variant'])]:
        tm = rows[rows['variant'] == name][ANS0_COLUMNS].reset_index(drop=True)
        table, text = ans0_gate(tm)
        g5_ok, g5_text = True, ''
        if name != 'V0':
            if base is None:
                g5_ok, g5_text = False, ' G5 not computable (no V0 rows).'
            else:
                mine = rows[rows['variant'] == name].set_index('task')
                common = mine.index.intersection(base.index)
                d5 = (mine.loc[common, 'r_ns'].astype(float).to_numpy()
                      - base.loc[common, 'r_ns'].astype(float).to_numpy())
                d5 = d5[np.isfinite(d5)]
                mean5 = float(d5.mean()) if d5.size else np.nan
                g5_ok = bool(np.isfinite(mean5) and mean5 >= 0.0)
                table = pd.concat([table, pd.DataFrame([dict(
                    criterion='G5 variant - V0 (the engine default)', n=int(d5.size), value=mean5,
                    bound=np.nan, required=0.0, passed=g5_ok)])], ignore_index=True)
                g5_text = f" G5={mean5:+.4f} ({'ok' if g5_ok else 'failed'})."
        table.insert(0, 'variant', name)
        tables.append(table)
        decisions[name] = text + g5_text
        if text.startswith('NOT DECIDED'):
            undecided = undecided or text
        go[name] = bool(text.startswith('GO') and g5_ok and name != 'V0')
    first = next((v for v in ANS1_ORDER if go.get(v)), None)
    if undecided is not None and first is None:
        decision = f"NOT DECIDED ({undecided[len('NOT DECIDED: '):]})"
    else:
        decision = first if first is not None else 'NONE'
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), decisions, decision


# ver4-51, gate A-NS.2 (blueprint R4): is the robust neural stack at least as good as equal weights?
ANS2_G1_MEAN = -0.002          # G1: mean delta_E >= this ...
ANS2_G1_LOWER = -0.01          # ... and the one-sided 90 % bootstrap lower bound >= this
ANS2_G2_LOSS = -0.02           # G2: a task "loses" when delta_E < this ...
ANS2_G2_MAX_SHARE = 0.10       # ... and at most this share of tasks may lose
ANS2_MIN_TASKS = 20
ANS2_MIN_GROUPS = 2
ANS2_BOOTSTRAPS = 2000
ANS2_BOOT_SEED = 1


def ans2_gate(rows):
    """ver4-51, gate A-NS.2: non-inferiority of the robust neural stack to equal weights, from the
    `replay_neural_stack_variants` rows `rows` (a DataFrame with `variant`, `task`, `r_ns`,
    `r_eq_dep`, `r_naive0`). Zero fits. For each of V5 (nested gate) and V4 (refit swap test only):
    delta_E = test r of the variant - test r of the equal-weight mean of the columns it deploys;
    delta_N = variant - the untuned naive mean (entry 0), both paired by task.

      G1  non-inferiority: mean delta_E >= -0.002 AND the one-sided 90 % bootstrap lower bound
          (2000 resamples, seed 1) >= -0.01;
      G2  tail: the share of tasks with delta_E < -0.02 is <= 0.10;
      G3  default switch: mean delta_N >= 0 (informational for the verdict: it decides whether a
          new-session default is proposed, never the endorsement).

    Needs >= 20 dumps from >= 2 (population, phenotype) groups per variant, else 'NOT DECIDED'.
    Returns `(table, decisions, decision)`: one table row per variant x criterion (`variant,
    criterion, n, value, bound, required, passed`), the text per variant and the text after
    `A-NS.2 DECISION:` - the first endorsed variant in the order V5, V4 with its G3 outcome,
    `NONE`, or `NOT DECIDED (...)`. A V5 that passes with mean delta_E near 0 is the recorded
    conclusion "learnt weights add nothing on this data; the robust stack behaves as equal
    weights"."""
    rows = rows.copy()
    tables, decisions, endorsed = [], {}, {}
    undecided = None
    for name in [v for v in ANS2_ORDER if v in set(rows['variant'])]:
        sub = rows[rows['variant'] == name]
        d_e = (sub['r_ns'].astype(float).to_numpy() - sub['r_eq_dep'].astype(float).to_numpy())
        d_n = (sub['r_ns'].astype(float).to_numpy() - sub['r_naive0'].astype(float).to_numpy())
        d_e, d_n = d_e[np.isfinite(d_e)], d_n[np.isfinite(d_n)]
        boots = (np.random.default_rng(ANS2_BOOT_SEED).choice(
            d_e, size=(ANS2_BOOTSTRAPS, d_e.size), replace=True).mean(axis=1)
            if d_e.size else np.array([np.nan]))
        lower = float(np.percentile(boots, 10.0))
        mean_e = float(d_e.mean()) if d_e.size else np.nan
        g1 = bool(np.isfinite(mean_e) and mean_e >= ANS2_G1_MEAN and lower >= ANS2_G1_LOWER)
        share = float(np.mean(d_e < ANS2_G2_LOSS)) if d_e.size else np.nan
        g2 = bool(np.isfinite(share) and share <= ANS2_G2_MAX_SHARE)
        mean_n = float(d_n.mean()) if d_n.size else np.nan
        g3 = bool(np.isfinite(mean_n) and mean_n >= 0.0)
        table = pd.DataFrame([
            dict(criterion='G1 variant - equal weights (deployed columns)', n=int(d_e.size),
                 value=mean_e, bound=lower, required=ANS2_G1_MEAN, passed=g1),
            dict(criterion=f'G2 share of tasks with delta_E < {ANS2_G2_LOSS:g}', n=int(d_e.size),
                 value=share, bound=np.nan, required=ANS2_G2_MAX_SHARE, passed=g2),
            dict(criterion='G3 variant - untuned naive mean (entry 0)', n=int(d_n.size),
                 value=mean_n, bound=np.nan, required=0.0, passed=g3)])
        table.insert(0, 'variant', name)
        tables.append(table)
        n_tasks = int(len(sub))
        groups = _task_groups(sub['task']) if n_tasks else set()
        if n_tasks < ANS2_MIN_TASKS or len(groups) < ANS2_MIN_GROUPS:
            text = (f"NOT DECIDED: {n_tasks} dump(s) from {len(groups)} (population, phenotype) "
                    f"group(s); A-NS.2 needs >= {ANS2_MIN_TASKS} dumps from >= {ANS2_MIN_GROUPS} "
                    f"groups.")
            undecided = undecided or text
        elif g1 and g2:
            text = (f"ENDORSED: G1={mean_e:+.4f} (lower {lower:+.4f}) G2={share:.3f} "
                    f"G3={mean_n:+.4f} ({'passed' if g3 else 'failed'}).")
            endorsed[name] = g3
        else:
            text = (f"NOT ENDORSED: " + ', '.join(c for c, ok in (('G1', g1), ('G2', g2)) if not ok)
                    + f" failed (G1={mean_e:+.4f} lower {lower:+.4f}, G2={share:.3f}).")
        decisions[name] = text
    first = next((v for v in ANS2_ORDER if v in endorsed), None)
    if first is not None:
        _sub = rows[rows['variant'] == first]
        _mean_gap = float(_sub['r_ns'].astype(float).mean() - _sub['r_eq_dep'].astype(float).mean())
        decision = (f"{first} (G1+G2 pass; G3 "
                    + ("passes: propose it as the new-session default"
                       if endorsed[first] else "fails: keep the preset opt-in")
                    + ("; mean delta_E ~ 0: learnt weights add nothing on this data, the next "
                       "lever is the train+valid refit" if abs(_mean_gap) < 0.002 else "") + ")")
    elif undecided is not None:
        decision = f"NOT DECIDED ({undecided[len('NOT DECIDED: '):]})"
    else:
        decision = 'NONE'
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), decisions, decision


def ns_diag_report(diag):
    """ver4-50, R1: the verdict lines of hypotheses H1-H3 from the `replay_ns_diag.csv` rows `diag`
    (a DataFrame, `NS_DIAG_COLUMNS`); returns a list of strings (also what `run` prints).

    H1   median `h1_mse_share` over the tasks with a positive in-sample gain >= `ANS1_H1_SHARE`:
         the gain of the network over the naive mean is mostly calibration (the `-mse/var` term).
    H1b  Pearson correlation, over all library columns of all tasks, of the V0 weight with
         `1 / oof_slope`; > 0 means heavy columns are the over-dispersed ones.
    H2   mean `flip_test_dr` over the tasks with a gate flip; < 0 means a flip loses on test.
    H3   per model, mean (OOF r - test r) of the entry-0 column: a gap of the same size for every
         model family points to split difficulty, not leakage."""
    lines = []
    task_rows = diag[diag['row_type'] == 'task']
    col_rows = diag[diag['row_type'] == 'column']
    share = task_rows['h1_mse_share'].astype(float).dropna()
    if len(share):
        med = float(share.median())
        lines.append(f"H1  median MSE share of the network's in-sample gain = {med:.3f} over "
                     f"{len(share)} task(s) -> {'CONFIRMED' if med >= ANS1_H1_SHARE else 'not '
                     'confirmed'} (threshold {ANS1_H1_SHARE:g}).")
    else:
        lines.append("H1  no task with a positive in-sample gain (lambda* = infinity everywhere?)"
                     " -> not testable.")
    w = col_rows['w_V0'].astype(float)
    inv = 1.0 / col_rows['oof_slope'].astype(float)
    ok = np.isfinite(w) & np.isfinite(inv)
    if int(ok.sum()) >= 3 and float(w[ok].std()) > 0 and float(inv[ok].std()) > 0:
        rho = float(np.corrcoef(w[ok], inv[ok])[0, 1])
        lines.append(f"H1b corr(weight, 1/slope) = {rho:+.3f} over {int(ok.sum())} column(s) -> "
                     f"{'CONFIRMED' if rho > 0 else 'not confirmed'}.")
    else:
        lines.append("H1b too few columns with a finite slope -> not testable.")
    flip = task_rows['flip_test_dr'].astype(float).dropna()
    if len(flip):
        mean_flip = float(flip.mean())
        lines.append(f"H2  mean test r change of a gate flip = {mean_flip:+.4f} over {len(flip)} "
                     f"task(s) with a flip -> {'CONFIRMED' if mean_flip < 0 else 'not confirmed'}.")
    else:
        lines.append("H2  no gate flip under V0 -> not testable.")
    e0 = col_rows[col_rows['entry'] == 0]
    gap = (e0['oof_r'].astype(float) - e0['test_r'].astype(float)).groupby(e0['model']).mean()
    if len(gap):
        lines.append("H3  mean (OOF r - test r) of entry 0 per model: "
                     + ', '.join(f"{m}={v:+.3f}" for m, v in gap.items())
                     + " (a similar gap for every model = split difficulty, not leakage).")
    return lines


def _simplex_grid(k, steps):
    """Yield blocks (m, k) of every weight vector on the simplex with step 1/steps (stars and
    bars). The number of points is C(steps + k - 1, k - 1); blocks keep the memory bounded."""
    block = []
    for bars in itertools.combinations(range(steps + k - 1), k - 1):
        prev, comp = -1, []
        for b in bars:
            comp.append(b - prev - 1)
            prev = b
        comp.append(steps + k - 2 - prev)
        block.append(comp)
        if len(block) >= 20000:
            yield np.asarray(block, float) / steps
            block = []
    if block:
        yield np.asarray(block, float) / steps


def dp_check(dump, *, steps=DP_GRID_STEPS):
    """ver4-42, R2 (A2.1-A2.2): zero-fit measurement of whether a grid / dynamic-programming
    weight optimiser could help. Returns ONE row dict, or `{}` when the dump has no usable
    entry-0 test columns. Everything is computed on the entry-0 (untuned) columns `P0` and the
    scoring targets; the test rows only evaluate.

      dp_gap_insample  best in-sample `ensemble_score` (pearson - mse/var, the engine's
                       objective) over the simplex grid of step 1/T, MINUS the same score at the
                       exact NNLS weights (`simplex_nnls`), floored at 0. A value above
                       `DP_GAP_MAX` means an exhaustive search finds something the exact convex
                       solver misses (optimisation, not variance, would be limiting).
      optimism_opt     scoring-row r minus test r of the NNLS (in-sample optimum) weights.
      optimism_uniform the same for equal weights.
    T is lowered automatically so the grid stays under `DP_GRID_MAX_POINTS` points (the T used
    is reported as `dp_steps`)."""
    from math import comb
    from models import joint_tuning as jt
    models, lib = dump['models'], dump['lib']
    y, y_test = dump['y_score'], dump['y_test']
    k = len(models)
    if k < 2 or y_test.size < 3:
        return {}
    if not all(dump['test'][m].ndim == 2 and dump['test'][m].shape[1] > 0
               and dump['test'][m].shape[0] == y_test.size for m in models):
        return {}
    P0 = np.column_stack([lib[m][0][1] for m in models])
    T0 = np.column_stack([dump['test'][m][:, 0] for m in models])
    if not (np.all(np.isfinite(P0)) and np.all(np.isfinite(T0))):
        return {}
    t = int(steps)
    while t > 2 and comb(t + k - 1, k - 1) > DP_GRID_MAX_POINTS:
        t -= 1
    var_y = float(np.var(y))
    yc = y - y.mean()
    best_score, best_w = -np.inf, None
    for W in _simplex_grid(k, t):
        C = P0 @ W.T                                   # (n, m) combined predictions
        sd = C.std(axis=0) * float(np.std(y))
        cov = ((C - C.mean(axis=0)) * yc[:, None]).mean(axis=0)
        mse = ((C - y[:, None]) ** 2).mean(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            sc = np.where(sd > 0, cov / sd, np.nan) - mse / var_y
        if np.all(np.isnan(sc)):
            continue
        i = int(np.nanargmax(sc))
        if sc[i] > best_score:
            best_score, best_w = float(sc[i]), W[i].copy()
    w_nnls = simplex_nnls(P0, y)
    s_nnls = jt._ensemble_score_of(P0 @ w_nnls, y)
    uniform = np.full(k, 1.0 / k)
    gap = (max(0.0, best_score - s_nnls) if (s_nnls is not None and best_w is not None)
           else np.nan)
    return dict(
        task=os.path.splitext(os.path.basename(dump['path']))[0], n_models=k, dp_steps=t,
        grid_points=int(comb(t + k - 1, k - 1)), dp_gap_insample=gap,
        score_grid_best=best_score if best_w is not None else np.nan,
        score_nnls=s_nnls if s_nnls is not None else np.nan,
        optimism_opt=_r(P0 @ w_nnls, y) - _r(T0 @ w_nnls, y_test),
        optimism_uniform=_r(P0 @ uniform, y) - _r(T0 @ uniform, y_test))


def dp_decision(df):
    """ver4-42, A2.1-A2.2: the decision of R2 from the per-task `dp_check` rows (DataFrame).
    Returns `(table DataFrame, sentence str)`. A2.1: `dp_gap_insample` <= `DP_GAP_MAX` on >= 95 %
    of the tasks. A2.2: mean(optimism_opt - optimism_uniform) >= `DP_OPTIMISM_MIN`. Both hold ->
    'DP not warranted'."""
    if df.empty:
        return pd.DataFrame(), "NOT DECIDED: no dump with usable test columns."
    gap = df['dp_gap_insample'].astype(float)
    share = float((gap.dropna() <= DP_GAP_MAX).mean()) if gap.notna().any() else np.nan
    opt_diff = float((df['optimism_opt'] - df['optimism_uniform']).mean(skipna=True))
    a21 = bool(np.isfinite(share) and share >= 0.95)
    a22 = bool(np.isfinite(opt_diff) and opt_diff >= DP_OPTIMISM_MIN)
    table = pd.DataFrame([
        dict(check='A2.1 share of tasks with dp_gap_insample <= %g' % DP_GAP_MAX, value=share,
             required='>= 0.95', passed=a21),
        dict(check='A2.2 mean optimism(in-sample optimum) - optimism(uniform)', value=opt_diff,
             required='>= %g' % DP_OPTIMISM_MIN, passed=a22)])
    if a21 and a22:
        text = ("DP not warranted: exact weights already reach the grid optimum (optimisation is "
                "not limiting) and the in-sample optimum is far more optimistic than uniform "
                "(estimation variance is limiting) - record 'no change warranted' for R2.")
    elif not a21:
        bad = df.loc[gap > DP_GAP_MAX, 'task'].tolist()
        text = (f"A2.1 FAILS on {len(bad)} task(s) {bad[:10]}: report them; treat a grid-search "
                f"S1 direction as a follow-up (not bundled).")
    else:
        text = ("A2.2 FAILS: the in-sample optimum is not markedly more optimistic than uniform "
                "on these dumps - variance is not clearly limiting; report the table.")
    return table, text


ns_parity_fn = ns_parity   # `run()` has a parameter of the same name as the function


def run(dump_glob, out_dir, estimators=ESTIMATORS, rules=RULES, bootstraps=200, weight_bags=100,
        seed=1, skip_variants=False, members_per_model=3, wf_rules=WF_DEFAULT_RULES,
        wf_max_candidates=6, wf_dedup_tol=0.05, wf_include_untuned=True, wf_pick='bbc',
        wfcv_direction='analytic', wfcv_alpha_rule='min', wfcv_repeats=4,
        n_valid_frac=A10_VALID_FRACTION, dp_check_flag=False, ns_lambdas=None, ns_rule=None,
        ns_repeats=None, ns_objective='ensemble_score', ns_swap_rule='gate', ns_shrink=0.5,
        ns_deploy='argmax', ns_variants=None, ns_seed=None, ns_parity=False,
        ns_outer_gate='off', ns_outer_folds=5, ns_outer_repeats=2):
    """Replay every dump matching `dump_glob`; write the CSVs; return
    `(task_metrics, paired_table)` DataFrames (empty when nothing matched or with
    `skip_variants`). ver4-38: always computes the R2.a measures and writes
    `Joint_replay_summary.csv` (see the module docstring). ver4-42: with rule `weight_first_cv`
    (and no `skip_variants`) also writes `replay_a10_task_metrics.csv` / `replay_a10.csv` and
    prints the A1.0 DECISION; with `dp_check_flag` writes `replay_dp_check.csv` and prints the
    A2.1-A2.2 decision. ver4-48: with rule `neural_stack` (and no `skip_variants`) writes
    `replay_ans0.csv` and prints the `A-NS.0 DECISION` (`ns_lambdas`, `ns_rule`, `ns_repeats`
    override the engine defaults of `hyper_stack` / `joint_tuning._JOINT_NS_*`). ver4-50: the single
    replay uses `ns_objective` / `ns_swap_rule` / `ns_shrink` / `ns_deploy` (default = the ver4-49
    engine, variant V0); with `ns_variants` (names from `NS_VARIANTS`) every variant is replayed on
    every dump instead and `replay_ans1.csv`, `replay_ans1_gate.csv`, `replay_ns_diag.csv` are
    written and `A-NS.1 DECISION:` is printed (`ns_seed=None`: the CV folds of the lambda choice
    are the engine's own; `ns_parity`: also print the max |W_replay - W_engine| of each dump)."""
    paths = sorted(glob.glob(dump_glob, recursive=True))
    if not paths:
        print(f"[REPLAY] no dump matches {dump_glob!r} - nothing to replay")
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    measure_rows = []
    wfcv_rows = []
    ns_rows = []
    ns1_rows = []
    ns_diag_rows = []
    dp_rows = []
    j_estimator_by_task = {}
    for path in paths:
        try:
            dump = load_dump(path)
        except Exception as exc:  # a corrupt / foreign file must not stop the whole replay
            print(f"[REPLAY] skipped {path}: {type(exc).__name__}: {exc}")
            continue
        try:
            measure_rows.append(ceiling_measures(dump, per_model=members_per_model,
                                                 weight_bags=weight_bags))
        except Exception as exc:
            print(f"[REPLAY] {os.path.basename(path)} measures failed: "
                  f"{type(exc).__name__}: {exc}")
        grid_rules = [r for r in rules
                      if r not in ('weight_first', 'weight_first_cv', 'neural_stack')]
        if dp_check_flag:
            try:
                dp_row = dp_check(dump)
                if dp_row:
                    dp_rows.append(dp_row)
            except Exception as exc:
                print(f"[REPLAY] {os.path.basename(path)} dp-check failed: "
                      f"{type(exc).__name__}: {exc}")
        if not skip_variants and 'weight_first_cv' in rules:
            try:
                wfcv_rows.extend(replay_weight_first_cv(
                    dump, direction=wfcv_direction, alpha_rule=wfcv_alpha_rule,
                    repeats=wfcv_repeats, n_valid_frac=n_valid_frac, bootstraps=bootstraps,
                    seed=seed))
            except Exception as exc:
                print(f"[REPLAY] {os.path.basename(path)} weight_first_cv failed: "
                      f"{type(exc).__name__}: {exc}")
        if not skip_variants and 'neural_stack' in rules:
            try:
                if ns_variants:
                    _vrows, _vdiag = replay_neural_stack_variants(
                        dump, {v: NS_VARIANTS[v] for v in ns_variants}, lambdas=ns_lambdas,
                        rule=ns_rule, repeats=ns_repeats, seed=ns_seed, with_diag=True,
                        outer_folds=ns_outer_folds, outer_repeats=ns_outer_repeats)
                    ns1_rows.extend(_vrows)
                    ns_diag_rows.extend(_vdiag)
                else:
                    ns_rows.extend(replay_neural_stack(
                        dump, lambdas=ns_lambdas, rule=ns_rule, repeats=ns_repeats, seed=ns_seed,
                        objective=ns_objective, swap_rule=ns_swap_rule, shrink=ns_shrink,
                        deploy=ns_deploy, outer_gate=ns_outer_gate, outer_folds=ns_outer_folds,
                        outer_repeats=ns_outer_repeats))
                if ns_parity:
                    _gap, _wm = ns_parity_fn(dump)
                    print(f"[REPLAY] {os.path.basename(path)} neural_stack parity: max |W_replay - "
                          f"W_engine| = " + ('n/a (no neural-stack record or not replayable)'
                                             if not np.isfinite(_gap) else f"{_gap:.3e}"
                                             + ('' if _gap <= 1e-9 else '  <-- ABOVE 1e-9: the '
                                                'replay no longer mirrors the engine')))
            except Exception as exc:
                print(f"[REPLAY] {os.path.basename(path)} neural_stack failed: "
                      f"{type(exc).__name__}: {exc}")
        for est in ([] if skip_variants else estimators):
            for rule in grid_rules:
                try:
                    rows.extend(replay_task(dump, est, rule, bootstraps=bootstraps,
                                            weight_bags=weight_bags, seed=seed))
                except Exception as exc:
                    print(f"[REPLAY] {os.path.basename(path)} {est}/{rule} failed: "
                          f"{type(exc).__name__}: {exc}")
        if not skip_variants and 'weight_first' in rules:
            # ver4-40 R2 (A2.0): the weight-first arm does not depend on the estimator, so it
            # runs once per dump; the J comparator (greedy + the estimator the dump was written
            # with) is added when the estimator loop above did not already produce it.
            j_est = str(dump['meta'].get('weight_estimator', 'legacy'))
            if j_est not in ESTIMATORS:
                j_est = 'legacy'
            j_estimator_by_task[os.path.splitext(os.path.basename(path))[0]] = j_est
            try:
                if not ('greedy' in grid_rules and j_est in estimators):
                    rows.extend(r for r in replay_task(dump, j_est, 'greedy',
                                                       bootstraps=bootstraps,
                                                       weight_bags=weight_bags, seed=seed)
                                if r['gate'] == 'off')
                rows.extend(replay_weight_first(
                    dump, rules=wf_rules, max_n=wf_max_candidates, dedup_tol=wf_dedup_tol,
                    include_untuned=wf_include_untuned, pick=wf_pick, bootstraps=bootstraps,
                    weight_bags=weight_bags, seed=seed))
            except Exception as exc:
                print(f"[REPLAY] {os.path.basename(path)} weight_first failed: "
                      f"{type(exc).__name__}: {exc}")
    summary_df, _notes = write_summary(measure_rows, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if dp_rows:
        dp_df = pd.DataFrame(dp_rows)
        dp_df.to_csv(os.path.join(out_dir, 'replay_dp_check.csv'), index=False)
        dp_table, dp_text = dp_decision(dp_df)
        print(dp_table.to_string(index=False))
        print(f"[REPLAY] A2 DECISION: {dp_text}")
    elif dp_check_flag:
        print("[REPLAY] --dp-check: no dump with usable test columns.")
    if wfcv_rows:
        a10_tm = pd.DataFrame(wfcv_rows)
        a10_tm.to_csv(os.path.join(out_dir, 'replay_a10_task_metrics.csv'), index=False)
        g6 = f_means = None
        if len(summary_df) and 'g6' in summary_df.columns:
            _mean = summary_df[summary_df['task'] == 'MEAN']
            if len(_mean) and pd.notna(_mean['g6'].iloc[0]):
                g6 = float(_mean['g6'].iloc[0])
            if len(_mean):
                f_means = {c[len('r_test__'):]: float(_mean[c].iloc[0]) for c in _mean.columns
                           if c.startswith('r_test__') and pd.notna(_mean[c].iloc[0])}
        a10, a10_decision = a10_gate(a10_tm, np.random.default_rng(0), g6=g6,
                                     f_model_means=f_means or None)
        a10.to_csv(os.path.join(out_dir, 'replay_a10.csv'), index=False)
        print(a10.to_string(index=False) if len(a10) else "[REPLAY] A1.0: no comparable task")
        print(f"[REPLAY] alpha* per task: {a10_tm['alpha'].round(2).tolist()}")
        print(f"[REPLAY] A1.0 DECISION: {a10_decision}")
    elif not skip_variants and 'weight_first_cv' in rules:
        print("[REPLAY] A1.0: no dump could be replayed with weight_first_cv (needs "
              "objective_data='cv_train_valid' dumps with entry-0 test columns).")
    if ns_rows:
        ans0_tm = pd.DataFrame(ns_rows)[ANS0_COLUMNS]
        ans0_tm.to_csv(os.path.join(out_dir, 'replay_ans0.csv'), index=False)
        ans0, ans0_decision = ans0_gate(ans0_tm)
        ans0.to_csv(os.path.join(out_dir, 'replay_ans0_gate.csv'), index=False)
        print(ans0.to_string(index=False) if len(ans0) else "[REPLAY] A-NS.0: no comparable task")
        print(f"[REPLAY] lambda* per task: {ans0_tm['lambda'].round(4).tolist()}")
        print(f"[REPLAY] A-NS.0 DECISION: {ans0_decision}")
    elif not skip_variants and 'neural_stack' in rules and not ns1_rows:
        print("[REPLAY] A-NS.0: no dump could be replayed with neural_stack (needs "
              "objective_data='cv_train_valid' dumps with finite entry-0 test columns).")
    if ns1_rows:
        # ver4-50: gate A-NS.1 (variants V0-V3) and the H1-H3 diagnostics
        ans1_tm = pd.DataFrame(ns1_rows)[ANS1_COLUMNS]
        ans1_tm.to_csv(os.path.join(out_dir, 'replay_ans1.csv'), index=False)
        ans1, ans1_texts, ans1_decision = ans1_gate(ans1_tm)
        ans1.to_csv(os.path.join(out_dir, 'replay_ans1_gate.csv'), index=False)
        print(ans1.to_string(index=False) if len(ans1) else "[REPLAY] A-NS.1: no comparable task")
        for _v, _t in ans1_texts.items():
            print(f"[REPLAY] A-NS.1 {_v}: {_t}")
        for _v in ans1_tm['variant'].unique():
            print(f"[REPLAY] lambda* per task ({_v}): "
                  f"{ans1_tm[ans1_tm['variant'] == _v]['lambda'].round(4).tolist()}")
        print(f"[REPLAY] A-NS.1 DECISION: {ans1_decision}")
        if {'V4', 'V5'} & set(ans1_tm['variant']):
            # ver4-51: gate A-NS.2 (variants V4 / V5 against equal weights)
            ans2_tm = pd.DataFrame(ns1_rows).reindex(columns=ANS2_COLUMNS)
            ans2_tm.to_csv(os.path.join(out_dir, 'replay_ans2.csv'), index=False)
            ans2, ans2_texts, ans2_decision = ans2_gate(ans2_tm)
            ans2.to_csv(os.path.join(out_dir, 'replay_ans2_gate.csv'), index=False)
            print(ans2.to_string(index=False) if len(ans2)
                  else "[REPLAY] A-NS.2: no comparable task")
            for _v, _t in ans2_texts.items():
                print(f"[REPLAY] A-NS.2 {_v}: {_t}")
            print(f"[REPLAY] A-NS.2 DECISION: {ans2_decision}")
    elif not skip_variants and 'neural_stack' in rules and ns_variants:
        print("[REPLAY] A-NS.1: no dump could be replayed with neural_stack (needs "
              "objective_data='cv_train_valid' dumps with finite entry-0 test columns).")
    if ns_diag_rows:
        ns_diag = pd.DataFrame(ns_diag_rows).reindex(columns=NS_DIAG_COLUMNS)
        ns_diag.to_csv(os.path.join(out_dir, 'replay_ns_diag.csv'), index=False)
        for _line in ns_diag_report(ns_diag):
            print(f"[REPLAY] {_line}")
    if skip_variants:
        return pd.DataFrame(), pd.DataFrame()
    tm = pd.DataFrame(rows)
    if tm.empty:
        if not (wfcv_rows or dp_rows or ns_rows or ns1_rows):
            print("[REPLAY] no task could be replayed (missing test columns? a dump written "
                  "with the stub on has none)")
        return tm, pd.DataFrame()
    os.makedirs(out_dir, exist_ok=True)
    tm.to_csv(os.path.join(out_dir, 'replay_task_metrics.csv'), index=False)
    rng = np.random.default_rng(0)
    comps = []
    for (est, rule, gate), g in tm.groupby(['estimator', 'rule', 'gate'], sort=False):
        g = g.set_index('task')
        for ref, label in (('r_naive', 'naive (same columns)'),
                           ('r_naive_untuned', 'naive (untuned columns)'),
                           ('r_best_single', 'best single column (by scoring rows)')):
            comps.append(dict(variant=f'{est}/{rule}/gate-{gate}', versus=label, metric='r',
                              **paired(g['r'], g[ref], rng)))
        comps.append(dict(variant=f'{est}/{rule}/gate-{gate}', versus='naive (same columns), '
                          'MSE (positive = variant better)', metric='mse',
                          **paired(g['mse_naive'], g['mse'], rng)))
    pt = pd.DataFrame(comps)
    pt.to_csv(os.path.join(out_dir, 'replay_paired.csv'), index=False)
    if (tm['rule'] == 'weight_first').any():
        g6 = None
        if len(summary_df) and 'g6' in summary_df.columns:
            _mean = summary_df[summary_df['task'] == 'MEAN']
            g6 = float(_mean['g6'].iloc[0]) if len(_mean) and pd.notna(_mean['g6'].iloc[0]) else None
        a20, decision = a20_gate(tm, j_estimator_by_task, np.random.default_rng(0), g6=g6)
        a20.to_csv(os.path.join(out_dir, 'replay_a20.csv'), index=False)
        print(a20.to_string(index=False) if len(a20) else "[REPLAY] A2.0: no comparable task")
        print(f"[REPLAY] A2.0 DECISION: {decision}")
    print(f"[REPLAY] {tm['task'].nunique()} task(s), {len(tm)} variant row(s)")
    show = pt[(pt.metric == 'r') & (pt.versus == 'naive (same columns)')].sort_values(
        'mean', ascending=False)
    print(show.to_string(index=False))
    best = show.iloc[0] if len(show) else None
    if best is not None and best['mean'] > 0:
        print(f"[REPLAY] top variant: {best['variant']} (delta-r vs naive {best['mean']:+.4f}, "
              f"95% CI [{best['ci_lo']:+.4f}, {best['ci_hi']:+.4f}], one-sided Wilcoxon "
              f"p={best['p_greater']:.4f}"
              + ("" if best['p_greater'] < 0.05 else ", NOT significant")
              + ") - a RANKING on a fixed library only; confirm the top two with real runs "
              "(tools/joint_benchmark.py).")
    else:
        print("[REPLAY] no variant beats the naive mean of its own columns on average - the "
              "replay gives no evidence for any estimator / rule on these dumps.")
    return tm, pt


# ------------------------------------------------------------------------------------------
# ver4-44, R2: paired evaluation of a joint run against an independent reference run (ZERO fits)
# ------------------------------------------------------------------------------------------
#
# Why: in a joint run the per-model rows and the naive 'ensemble' row use the joint-TUNED
# columns, so Metric.csv measures the weighting alone and the tuning effect is hidden from both
# sides. The question "does the joint approach improve prediction?" needs a reference run (same
# config, JOINT_W_OPT off, W_OPT methods on, untuned) on the same splits. The splits come from the
# deterministic `sample` table, which JOINT_W_OPT does not enter, so the two runs pair per task,
# row by row; `evaluate_runs` asserts identical test ids and `actual` and drops a task that fails.
#
#   Delta_total  = r(joint) - r(reference naive)
#                = [r(joint naive) - r(reference naive)]     Delta_tune   (the tuning effect)
#                + [r(joint) - r(joint naive)]               Delta_weight (the weighting effect)
#   H            = r(test-fitted simplex weights on the joint run's columns) - r(joint naive),
#                  an optimistic ceiling of the weighting effect (reported, never a target).
#
# The verdict is RECORDED here and never used inside the engine.

EVAL_KEY = ['population', 'phenotype', 'ratio', 'sample']
EVAL_META = ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')
EVAL_JOINT_COLUMN = 'Joint Bayesian'              # the Prediction_result_* column
EVAL_JOINT_LABEL = 'Joint Bayesian optimisation'  # the Metric.csv / Weight.csv 'model' value
EVAL_NAIVE = 'ensemble'
# Weighted-ensemble prediction columns (diversity_summary.py's `_WOPT_COLUMN_PREFIXES`, copied
# because this tool never imports engine modules that fit). A column equal to a prefix or starting
# with '<prefix>__' is not a single model.
EVAL_WOPT_PREFIXES = ('Linear transformation', 'Nelder-Mead', 'Bayesian', 'Analytic least-squares',
                      'Joint Bayesian')
EVAL_MIN_TASKS = 20            # the verdict needs at least this many paired tasks
EVAL_LOSS_MARGIN = -0.01       # 'share of tasks with delta < -0.01'
EVAL_CHECK_TOL = 1e-9          # A2.1 / A2.2 tolerance
EVAL_SCOPES = ('all', 'excluding_flagged')
EVAL_FORBIDDEN_MODULES = ('rpy2', 'genomic_prediction')   # A2.5: importing these means a fit path


def _eval_is_wopt(name):
    return isinstance(name, str) and any(
        name == p or name.startswith(p + '__') for p in EVAL_WOPT_PREFIXES)


def _eval_classify(pred):
    """`{'joint': column|None, 'naive': column|None, 'wopt': [...], 'base': [...]}` for a
    prediction table. Pure."""
    cols = [c for c in pred.columns if c not in EVAL_META]
    joint = [c for c in cols if c == EVAL_JOINT_COLUMN or str(c).startswith(EVAL_JOINT_COLUMN + '__')]
    naive = [c for c in cols if c == EVAL_NAIVE or str(c).startswith(EVAL_NAIVE + '__')]
    wopt = [c for c in cols if _eval_is_wopt(c) and c not in joint]
    base = [c for c in cols if c not in joint and c not in naive and c not in wopt]
    return {'joint': joint[0] if joint else None, 'naive': naive[0] if naive else None,
            'wopt': wopt, 'base': base}


def _eval_r(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if y.size < 3 or not (np.all(np.isfinite(p)) and np.all(np.isfinite(y))):
        return np.nan
    if p.std() <= 0 or y.std() <= 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def _eval_mse(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    return float(np.mean((y - p) ** 2)) if y.size else np.nan


def _eval_boot_r(y, p, idx):
    """Pearson r of (y, p) on every bootstrap row-index matrix `idx` (B x n); NaN where a resample
    is degenerate. Pure, vectorised."""
    yb = np.asarray(y, float)[idx]
    pb = np.asarray(p, float)[idx]
    yc = yb - yb.mean(axis=1, keepdims=True)
    pc = pb - pb.mean(axis=1, keepdims=True)
    den = np.sqrt((yc ** 2).sum(axis=1) * (pc ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (yc * pc).sum(axis=1) / den, np.nan)


def _eval_groups(df):
    """`{(population, phenotype, ratio, sample) as strings: rows}`; the ratio is a string so a
    tuple-valued ratio compares equal across files. Pure."""
    out = {}
    if df is None or df.empty:
        return out
    t = df.copy()
    for c in EVAL_KEY:
        t[c] = t[c].astype(str)
    for key, g in t.groupby(EVAL_KEY, sort=False):
        out[tuple(str(k) for k in key)] = g
    return out


def _eval_align(gj, gr):
    """Align one task's joint and reference test rows. Returns `(gj, gr, reason)`; `reason` is
    None when the two frames hold the identical test ids and `actual` (A2.4), otherwise a short
    string and the frames are returned unchanged."""
    if 'id' in gj.columns and 'id' in gr.columns:
        ij, ir = gj['id'].astype(str).to_numpy(), gr['id'].astype(str).to_numpy()
        if len(set(ij)) != len(ij) or len(set(ir)) != len(ir):
            return gj, gr, 'duplicate_test_ids'
        if set(ij) != set(ir):
            return gj, gr, 'test_ids_differ'
        gj = gj.iloc[np.argsort(ij, kind='stable')]
        gr = gr.iloc[np.argsort(ir, kind='stable')]
    elif len(gj) != len(gr):
        return gj, gr, 'test_row_counts_differ'
    aj, ar = gj['actual'].to_numpy(float), gr['actual'].to_numpy(float)
    if aj.shape != ar.shape or not np.allclose(aj, ar, rtol=0.0, atol=1e-9, equal_nan=True):
        return gj, gr, 'actual_differs'
    return gj, gr, None


def _eval_metric_lookup(metric):
    """`{(population, phenotype, ratio, sample, model): Pearson r}` from a Metric.csv frame. The
    unweighted column is preferred when the run used BLUE weighting (the prediction files carry
    no weights). Pure."""
    if metric is None or metric.empty or 'Pearson correlation' not in metric.columns:
        return {}
    col = ('Pearson correlation (unweighted)'
           if 'Pearson correlation (unweighted)' in metric.columns else 'Pearson correlation')
    out = {}
    for pop, ph, ra, sa, mo, r in zip(metric['population'].astype(str), metric['phenotype'].astype(str),
                                      metric['ratio'].astype(str), metric['sample'].astype(str),
                                      metric['model'].astype(str), metric[col]):
        out[(pop, ph, ra, sa, mo)] = float(r) if pd.notna(r) else np.nan
    return out


def _eval_weight_map(weight, joint_columns):
    """`{task key: {model column: weight}}` of the joint rows of Weight.csv, restricted to the
    model columns present in the joint run's prediction table. Pure."""
    if weight is None or weight.empty or 'model' not in weight.columns:
        return {}
    w = weight[weight['model'].astype(str) == EVAL_JOINT_LABEL]
    out = {}
    for key, g in _eval_groups(w).items():
        row = g.iloc[-1]
        out[key] = {c: float(row[c]) for c in joint_columns
                    if c in g.columns and pd.notna(row[c])}
    return out


def _eval_flag_map(hp):
    """`{task key: (events list, safety_nets bool | None)}` from the joint run's
    hyperparameter.csv, or None when the file has no `joint_fallbacks` column (a run whose
    HYPERPARAMETERS_JOINT_OPT held no ver4-28+ key writes no provenance columns). Pure."""
    if hp is None or hp.empty or 'joint_fallbacks' not in hp.columns:
        return None
    out = {}
    for key, g in _eval_groups(hp).items():
        events = []
        for cell in g['joint_fallbacks'].fillna('').astype(str):
            events.extend(tok.strip() for tok in cell.split(';') if tok.strip())
        nets = None
        if 'joint_safety_nets' in g.columns:
            vals = g['joint_safety_nets'].dropna().astype(str).str.lower()
            nets = bool((vals == 'true').all()) if len(vals) else None
        out[key] = (sorted(set(events)), nets)
    return out


def _eval_pool(values, rng):
    """Pooled statistics of a per-task value vector: n, mean, SE, bootstrap 95 % CI of the mean,
    one-sided paired Wilcoxon p (> 0), win rate and the share below the loss margin. Pure."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    row = dict(n=int(v.size), mean=np.nan, se=np.nan, ci_lo=np.nan, ci_hi=np.nan,
               p_greater=np.nan, win_rate=np.nan, share_below_m0p01=np.nan)
    if v.size == 0:
        return row
    row['mean'] = float(v.mean())
    row['win_rate'] = float(np.mean(v > 0))
    row['share_below_m0p01'] = float(np.mean(v < EVAL_LOSS_MARGIN))
    if v.size >= 2:
        row['se'] = float(v.std(ddof=1) / np.sqrt(v.size))
        p = paired(v, np.zeros_like(v), rng)
        row['ci_lo'], row['ci_hi'], row['p_greater'] = p['ci_lo'], p['ci_hi'], p['p_greater']
    return row


def evaluate_runs(joint_test, ref_test, *, joint_weight=None, joint_metric=None, ref_metric=None,
                  joint_hp=None, n_boot=500, seed=0, min_tasks=EVAL_MIN_TASKS):
    """ver4-44, R2: pair a joint run with a reference run task by task and measure the joint
    approach with ZERO model fits. Pure (no files, no engine imports except the NNLS helper).

    joint_test / ref_test : `Prediction_result_test` tables of the two runs. The joint run must
        hold a 'Joint Bayesian' column and (for the tuning / weighting split) an 'ensemble'
        column; the reference run must hold an 'ensemble' column and may hold single-model and
        W_OPT columns.
    joint_weight : the joint run's Weight.csv (A2.1 recomputation and the H ceiling's column set).
    joint_metric / ref_metric : Metric.csv frames (A2.1 / A2.2 reproduction checks).
    joint_hp : the joint run's hyperparameter.csv (the `joint_fallbacks` flags).

    Returns `{'tasks', 'summary', 'verdicts', 'checks', 'dropped'}`:
      tasks    one row per task of either run: `status` 'paired' or 'dropped' (+ `drop_reason`),
               r / MSE of the joint row, the joint naive row, the reference naive row, every
               reference single model and W_OPT method, `d_total`, `d_tune`, `d_weight`
               (d_total = d_tune + d_weight), `d_vs_<model>` / `d_vs_wopt_<method>`, paired
               row-bootstrap 95 % CIs of the three main deltas, `H`, the flags
               (`flagged`, `fallbacks`, `safety_nets`) and the per-task A2.1 / A2.2 / A2.3 diffs;
      summary  pooled statistics per scope ('all' / 'excluding_flagged') and measure;
      verdicts `{scope: {'n_tasks', 'improves' (True / False / None), 'criteria', 'text'}}`;
      checks   `{'A2.1': max abs diff or NaN, 'A2.2': ..., 'A2.3': ..., 'A2.4': number dropped
               for a split mismatch}`;
      dropped  the dropped tasks as `[(key, reason)]`.
    Raises KeyError when a prediction table lacks the standard columns."""
    rng = np.random.default_rng(seed)
    gj_all, gr_all = _eval_groups(joint_test), _eval_groups(ref_test)
    cj = _eval_classify(joint_test) if joint_test is not None and not joint_test.empty else \
        {'joint': None, 'naive': None, 'wopt': [], 'base': []}
    cr = _eval_classify(ref_test) if ref_test is not None and not ref_test.empty else \
        {'joint': None, 'naive': None, 'wopt': [], 'base': []}
    weights = _eval_weight_map(joint_weight, cj['base'])
    flags = _eval_flag_map(joint_hp)
    jm, rm = _eval_metric_lookup(joint_metric), _eval_metric_lookup(ref_metric)
    rows, dropped = [], []
    diff_a21, diff_a22, diff_a23 = [], [], []
    for key in list(gj_all) + [k for k in gr_all if k not in gj_all]:
        row = dict(zip(EVAL_KEY, key))
        if key not in gj_all or key not in gr_all:
            reason = 'missing_in_reference' if key not in gr_all else 'missing_in_joint'
            row.update(status='dropped', drop_reason=reason)
            rows.append(row)
            dropped.append((key, reason))
            continue
        gj, gr, reason = _eval_align(gj_all[key], gr_all[key])
        if reason is None and cj['joint'] is None:
            reason = 'no_joint_column'
        if reason is None and cr['naive'] is None:
            reason = 'no_reference_naive_column'
        if reason is not None:
            row.update(status='dropped', drop_reason=reason)
            rows.append(row)
            dropped.append((key, reason))
            continue
        y = gj['actual'].to_numpy(float)
        pj = gj[cj['joint']].to_numpy(float)
        pr = gr[cr['naive']].to_numpy(float)
        pn = gj[cj['naive']].to_numpy(float) if cj['naive'] is not None else None
        r_j, r_r = _eval_r(y, pj), _eval_r(y, pr)
        r_n = _eval_r(y, pn) if pn is not None else np.nan
        row.update(status='paired', drop_reason='', n_test=int(len(y)),
                   r_joint=r_j, r_joint_naive=r_n, r_ref_naive=r_r,
                   mse_joint=_eval_mse(y, pj),
                   mse_joint_naive=_eval_mse(y, pn) if pn is not None else np.nan,
                   mse_ref_naive=_eval_mse(y, pr))
        row['d_total'] = r_j - r_r
        row['d_tune'] = r_n - r_r
        row['d_weight'] = r_j - r_n
        row['a23_abs_diff'] = abs(row['d_total'] - (row['d_tune'] + row['d_weight']))
        if np.isfinite(row['a23_abs_diff']):
            diff_a23.append(row['a23_abs_diff'])
        for m in cr['base']:
            row[f'ref_r__{m}'] = _eval_r(y, gr[m].to_numpy(float))
            row[f'd_vs__{m}'] = r_j - row[f'ref_r__{m}']
        for w in cr['wopt']:
            row[f'ref_r__{w}'] = _eval_r(y, gr[w].to_numpy(float))
            row[f'd_vs_wopt__{w}'] = r_j - row[f'ref_r__{w}']
        # paired row-bootstrap CIs of the three main deltas (the same resampled rows for all)
        idx = rng.integers(0, len(y), size=(int(n_boot), len(y)))
        bj, br = _eval_boot_r(y, pj, idx), _eval_boot_r(y, pr, idx)
        bn = _eval_boot_r(y, pn, idx) if pn is not None else np.full(int(n_boot), np.nan)
        for name, boots in (('d_total', bj - br), ('d_tune', bn - br), ('d_weight', bj - bn)):
            ok = boots[np.isfinite(boots)]
            row[f'{name}_ci_lo'] = float(np.percentile(ok, 2.5)) if ok.size >= 20 else np.nan
            row[f'{name}_ci_hi'] = float(np.percentile(ok, 97.5)) if ok.size >= 20 else np.nan
        # H: the weighting ceiling on the joint run's own columns (test-fitted simplex weights)
        wmap = weights.get(key) or {}
        sel = [c for c in cj['base'] if (not wmap or c in wmap)]
        row['H'] = np.nan
        if sel and np.all(np.isfinite(gj[sel].to_numpy(float))) and np.isfinite(r_n):
            P = gj[sel].to_numpy(float)
            r_or = _eval_r(y, P @ simplex_nnls(P, y))
            row['r_oracle'] = r_or
            row['H'] = r_or - r_n
        # A2.1: the joint row recomputed from Weight.csv x the columns, and Metric.csv's own value
        row['a21_abs_diff'] = np.nan
        if wmap:
            cols_w = [c for c in wmap if c in gj.columns]
            r_w = _eval_r(y, gj[cols_w].to_numpy(float) @ np.array([wmap[c] for c in cols_w]))
            d = [abs(r_w - r_j)] if np.isfinite(r_w) and np.isfinite(r_j) else []
            r_met = jm.get(key + (EVAL_JOINT_LABEL,))
            if r_met is not None and np.isfinite(r_met) and np.isfinite(r_w):
                d.append(abs(r_w - r_met))
            if d:
                row['a21_abs_diff'] = float(max(d))
                diff_a21.append(row['a21_abs_diff'])
        # A2.2: the reference naive row recomputed against the reference Metric.csv value
        row['a22_abs_diff'] = np.nan
        r_met = rm.get(key + (EVAL_NAIVE,))
        if r_met is not None and np.isfinite(r_met) and np.isfinite(r_r):
            row['a22_abs_diff'] = abs(r_met - r_r)
            diff_a22.append(row['a22_abs_diff'])
        # flags from the joint run's hyperparameter.csv
        if flags is None:
            row.update(flagged=False, fallbacks='', safety_nets=np.nan, flags_known=False)
        else:
            events, nets = flags.get(key, ([], None))
            row.update(flagged=bool(events), fallbacks=';'.join(events),
                       safety_nets=(np.nan if nets is None else nets), flags_known=True)
        rows.append(row)
    tasks = pd.DataFrame(rows)
    paired_tasks = tasks[tasks['status'] == 'paired'] if len(tasks) else tasks
    summary_rows, verdicts = [], {}
    measure_cols = [c for c in (paired_tasks.columns if len(paired_tasks) else [])
                    if c in ('d_total', 'd_tune', 'd_weight', 'H') or c.startswith('d_vs__')
                    or c.startswith('d_vs_wopt__')]
    mean_cols = [c for c in (paired_tasks.columns if len(paired_tasks) else [])
                 if c in ('r_joint', 'r_joint_naive', 'r_ref_naive') or c.startswith('ref_r__')]
    for scope in EVAL_SCOPES:
        sub = paired_tasks if scope == 'all' or not len(paired_tasks) else \
            paired_tasks[~paired_tasks['flagged'].astype(bool)]
        for m in measure_cols:
            summary_rows.append(dict(scope=scope, kind='delta', measure=m, **_eval_pool(sub[m], rng)))
        for m in mean_cols:
            v = sub[m].to_numpy(float)
            v = v[np.isfinite(v)]
            summary_rows.append(dict(scope=scope, kind='mean_r', measure=m, n=int(v.size),
                                     mean=float(v.mean()) if v.size else np.nan,
                                     se=float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan))
        verdicts[scope] = _eval_verdict(sub, cr, min_tasks, rng)
    checks = {'A2.1': max(diff_a21) if diff_a21 else np.nan,
              'A2.2': max(diff_a22) if diff_a22 else np.nan,
              'A2.3': max(diff_a23) if diff_a23 else np.nan,
              'A2.4': int(sum(1 for _, r in dropped if r not in ('missing_in_reference',
                                                                  'missing_in_joint')))}
    return {'tasks': tasks, 'summary': pd.DataFrame(summary_rows), 'verdicts': verdicts,
            'checks': checks, 'dropped': dropped}


def _eval_verdict(sub, cr, min_tasks, rng):
    """The recorded verdict of one scope. 'The joint approach improves prediction' iff, over at
    least `min_tasks` paired tasks, the mean Delta_total > 0 against the reference naive with
    p < 0.05, AND the mean joint r is >= the mean r of the best reference single model AND of
    every reference W_OPT method. `improves` is None when there are too few tasks. Pure."""
    n = int(np.isfinite(sub['d_total'].to_numpy(float)).sum()) if len(sub) else 0
    out = {'n_tasks': n, 'improves': None, 'criteria': {}, 'text': ''}
    if n < min_tasks:
        out['text'] = (f"INCONCLUSIVE: {n} paired task(s) < {min_tasks} required; the statistics "
                       f"below are descriptive only.")
        return out
    tot = _eval_pool(sub['d_total'], rng)
    crit = {'mean_d_total_gt_0_p_lt_0.05': bool(tot['mean'] > 0 and tot['p_greater'] < 0.05)}
    mean_joint = float(np.nanmean(sub['r_joint']))
    singles = {m: float(np.nanmean(sub[f'ref_r__{m}'])) for m in cr['base']
               if f'ref_r__{m}' in sub.columns and sub[f'ref_r__{m}'].notna().any()}
    wopts = {w: float(np.nanmean(sub[f'ref_r__{w}'])) for w in cr['wopt']
             if f'ref_r__{w}' in sub.columns and sub[f'ref_r__{w}'].notna().any()}
    if singles:
        best = max(singles, key=singles.get)
        crit[f'mean_r_joint_ge_best_single({best})'] = bool(mean_joint >= singles[best])
    if wopts:
        crit['mean_r_joint_ge_every_wopt'] = bool(all(mean_joint >= v for v in wopts.values()))
    else:
        crit['mean_r_joint_ge_every_wopt'] = False   # no W_OPT arm in the reference: cannot be shown
    out['criteria'] = crit
    out['improves'] = bool(all(crit.values()))
    out['text'] = (f"{'IMPROVES' if out['improves'] else 'DOES NOT IMPROVE'} (n={n}): "
                   f"mean d_total={tot['mean']:+.4f} (SE {tot['se']:.4f}, p={tot['p_greater']:.4f}); "
                   f"mean r joint={mean_joint:.4f}; "
                   + '; '.join(f"{k}={'ok' if v else 'FAIL'}" for k, v in crit.items()))
    return out


def _read_predictions(run_dir, key='result_test'):
    """Read one result file of a finished run from its folder: the combined file
    (`Prediction_result_test.csv`, or `.csv.gz`), else the per-batch files
    (`Prediction_result_test_<idx>.csv[.gz]`, ascending idx) concatenated. Only the standard
    `checkpoint_utils.RESULT_FILE_NAMES` names are used (I8). `run_dir` is a folder, or a
    RESULT_NAME resolved under the engine's Result directory. An absent, empty or truncated
    file gives an empty DataFrame (never raises)."""
    import re
    from checkpoint_utils import RESULT_FILE_NAMES, _INCOMPLETE_FILE_ERRORS
    if key not in RESULT_FILE_NAMES:
        raise KeyError(f"unknown result key {key!r}; known: {', '.join(RESULT_FILE_NAMES)}")
    if not os.path.isdir(run_dir):
        from pipeline_utils import result_dir_path
        run_dir = result_dir_path(run_dir)
    stem, ext = os.path.splitext(RESULT_FILE_NAMES[key])
    folder = os.path.join(run_dir, os.path.dirname(stem)) if os.path.dirname(stem) else run_dir
    stem = os.path.basename(stem)
    if not os.path.isdir(folder):
        return pd.DataFrame()
    names = os.listdir(folder)
    paths = [os.path.join(folder, n) for n in (stem + ext + '.gz', stem + ext) if n in names]
    if not paths:
        pat = re.compile('^' + re.escape(stem) + r'_(\d+)' + re.escape(ext) + r'(\.gz)?$')
        batches = sorted((int(m.group(1)), n) for n in names for m in [pat.match(n)] if m)
        paths = [os.path.join(folder, n) for _, n in batches]
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except _INCOMPLETE_FILE_ERRORS:
            print(f"[EVAL] NOTE: {path} is empty or truncated - ignored")
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def evaluate(joint_dir, reference_dir, out_dir, *, exclude_fallbacks=False, n_boot=500, seed=0,
             min_tasks=EVAL_MIN_TASKS):
    """Read the two runs, run `evaluate_runs`, write `evaluation_tasks.csv` and
    `evaluation_summary.csv` into `out_dir`, print the self-checks and the verdict, and return the
    `evaluate_runs` result (None when a run has no test predictions). `exclude_fallbacks` picks
    which scope the headline VERDICT line states (both are always printed and written)."""
    joint_test = _read_predictions(joint_dir, 'result_test')
    ref_test = _read_predictions(reference_dir, 'result_test')
    for label, frame, path in (('joint', joint_test, joint_dir), ('reference', ref_test, reference_dir)):
        if frame.empty:
            print(f"[EVAL] no Prediction_result_test found for the {label} run {path!r} - nothing "
                  f"to evaluate")
            return None
    res = evaluate_runs(
        joint_test, ref_test,
        joint_weight=_read_predictions(joint_dir, 'weight'),
        joint_metric=_read_predictions(joint_dir, 'record'),
        ref_metric=_read_predictions(reference_dir, 'record'),
        joint_hp=_read_predictions(joint_dir, 'hp_record'),
        n_boot=n_boot, seed=seed, min_tasks=min_tasks)
    os.makedirs(out_dir, exist_ok=True)
    tasks_path = os.path.join(out_dir, 'evaluation_tasks.csv')
    summary_path = os.path.join(out_dir, 'evaluation_summary.csv')
    res['tasks'].to_csv(tasks_path, index=False)
    res['summary'].to_csv(summary_path, index=False)
    tasks = res['tasks']
    n_paired = int((tasks['status'] == 'paired').sum()) if len(tasks) else 0
    print(f"[EVAL] {n_paired} task(s) paired, {len(res['dropped'])} dropped")
    for key, reason in res['dropped']:
        print(f"[EVAL] dropped task {key}: {reason}")
    if len(tasks) and 'flags_known' in tasks.columns and not tasks['flags_known'].fillna(False).any():
        print("[EVAL] NOTE: the joint run's hyperparameter.csv has no `joint_fallbacks` column "
              "(no fallback-flag key in HYPERPARAMETERS_JOINT_OPT), so no task could be flagged; the "
              "'excluding_flagged' scope equals 'all'.")
    elif len(tasks):
        n_flag = int(tasks.loc[tasks['status'] == 'paired', 'flagged'].fillna(False).astype(bool).sum())
        print(f"[EVAL] {n_flag} paired task(s) carry a class C fallback event")
    ck = res['checks']
    for name, label in (('A2.1', 'joint row = Weight.csv x columns = Metric.csv'),
                        ('A2.2', 'reference naive = reference Metric.csv'),
                        ('A2.3', 'd_total = d_tune + d_weight')):
        v = ck[name]
        status = ('n/a' if not np.isfinite(v) else ('ok' if v <= EVAL_CHECK_TOL else 'DIFFERS'))
        print(f"[EVAL] check {name} ({label}): max abs diff "
              f"{'n/a' if not np.isfinite(v) else f'{v:.3g}'} -> {status}"
              + (" (BLUE weighting or a library ensemble can legitimately differ; otherwise "
                 "investigate)" if status == 'DIFFERS' and name != 'A2.3' else ''))
    print(f"[EVAL] check A2.4 (split mismatch): {ck['A2.4']} task(s) dropped for a mismatched split")
    leaked = sorted(m for m in sys.modules
                    if any(m == f or m.startswith(f + '.') for f in EVAL_FORBIDDEN_MODULES))
    print(f"[EVAL] check A2.5 (zero fits): "
          + ("ok (no rpy2 / genomic_prediction module loaded)" if not leaked
             else f"FAILED - loaded {leaked}"))
    summ = res['summary']
    for scope in EVAL_SCOPES:
        sub = summ[(summ['scope'] == scope) & (summ['kind'] == 'delta')]
        print(f"[EVAL] --- pooled deltas, scope={scope} ---")
        show = sub[sub['measure'].isin(['d_total', 'd_tune', 'd_weight', 'H'])]
        if len(show):
            print(show[['measure', 'n', 'mean', 'se', 'ci_lo', 'ci_hi', 'p_greater', 'win_rate',
                        'share_below_m0p01']].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    primary = 'excluding_flagged' if exclude_fallbacks else 'all'
    for scope in EVAL_SCOPES:
        v = res['verdicts'][scope]
        print(f"[EVAL] VERDICT{' (headline)' if scope == primary else ''} [{scope}]: {v['text']}")
    print("[EVAL] The verdict is recorded only; nothing in the engine reads it. "
          f"Wrote {tasks_path} and {summary_path}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dumps', default=None, help="glob of library dumps (.npz); required unless "
                                                  "--evaluate")
    ap.add_argument('--out', default='replay_out')
    ap.add_argument('--bootstraps', type=int, default=200)
    ap.add_argument('--weight-bags', type=int, default=100)
    ap.add_argument('--estimators', default=','.join(ESTIMATORS))
    ap.add_argument('--rules', default=','.join(RULES))
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--skip-variants', action='store_true',
                    help="only compute the R2.a measures and write Joint_replay_summary.csv")
    ap.add_argument('--members-per-model', type=int, default=3,
                    help="library pruning for the ceiling CL (default 3 per model + the selected)")
    ap.add_argument('--wf-rules', default=','.join(WF_DEFAULT_RULES),
                    help="weight-first rules (uniform, analytic, shrunk:<a>, es)")
    ap.add_argument('--wf-max-candidates', type=int, default=6)
    ap.add_argument('--wf-dedup-tol', type=float, default=0.05)
    ap.add_argument('--wf-no-untuned', action='store_true',
                    help="exclude the untuned arms from the weight-first finalists "
                         "(wf_include_untuned=False, the strict policy)")
    ap.add_argument('--wf-pick', choices=('bbc', 'bbc_max', 'max'), default='bbc')
    ap.add_argument('--wfcv-direction', choices=('analytic', 'es'), default='analytic',
                    help="direction of the shrinkage path of the weight_first_cv arm")
    ap.add_argument('--wfcv-alpha-rule', choices=('min', 'one_se'), default='min')
    ap.add_argument('--wfcv-repeats', type=int, default=4, help="CV repeats of the alpha choice")
    ap.add_argument('--n-valid-frac', type=float, default=A10_VALID_FRACTION,
                    help="validation share of the scoring rows used for the W_OPT analogue when "
                         "the dump has no meta['n_valid'] (0.25 = a 60/20/20 split)")
    ap.add_argument('--ns-lambdas', default=None,
                    help="comma list of lambdas of the neural_stack arm (default: the "
                         "engine's joint_tuning._JOINT_NS_LAMBDAS_DEFAULT; lambda = inf is always "
                         "added)")
    ap.add_argument('--ns-rule', choices=('min', 'one_se'), default=None,
                    help="lambda choice rule of the neural_stack arm (default: the "
                         "engine default)")
    ap.add_argument('--ns-repeats', type=int, default=None,
                    help="CV repeats of the neural_stack lambda choice (default: the "
                         "engine default)")
    ap.add_argument('--ns-objective', choices=('ensemble_score', 'pearson'),
                    default='ensemble_score',
                    help="training objective of the single neural_stack replay (default: "
                         "the engine's ensemble_score)")
    ap.add_argument('--ns-swap-rule', choices=('gate', 'paired_se', 'paired_refit'), default='gate',
                    help="hyperparameter-switch rule of the single neural_stack replay")
    ap.add_argument('--ns-shrink', type=float, default=0.5,
                    help="weight shrink towards equal weights of the single replay "
                         "(the engine's ns_weight_shrink, 0-1)")
    ap.add_argument('--ns-deploy', choices=('argmax', 'mixture'), default='argmax',
                    help="deployment of the single neural_stack replay (the engine's "
                         "ns_deploy; the engine default is mixture, the A-NS.1 baseline V0 uses "
                         "argmax)")
    ap.add_argument('--ns-variants', default=None,
                    help="comma list of the A-NS.1 variants to replay on every dump "
                         "(V0 = the engine's default, V1 = pearson, V2 = pearson + paired_se, V3 = "
                         "V2 + mixture); writes replay_ans1.csv, replay_ans1_gate.csv, "
                         "replay_ns_diag.csv and prints 'A-NS.1 DECISION:'. Include V0 and V1 for "
                         "the full diagnostics")
    ap.add_argument('--ns-seed', type=int, default=None,
                    help="seed of the neural_stack lambda-choice folds (default: the "
                         "engine's own seed stored in each dump, so the replay reproduces the "
                         "engine); --seed is NOT used for the neural_stack arm")
    ap.add_argument('--ns-parity', action='store_true',
                    help="for every dump with a neural-stack record print the largest "
                         "|W_replay - W_engine| (AC1.1: it must be below 1e-9)")
    ap.add_argument('--ns-outer-gate', choices=['off', 'nested'], default='off',
                    help="nested outer gate of the single-variant neural_stack replay "
                         "(variants V5 always use it); default off")
    ap.add_argument('--ns-outer-folds', type=int, default=5,
                    help="outer folds of the nested gate (>= 2)")
    ap.add_argument('--ns-outer-repeats', type=int, default=2,
                    help="outer repeats of the nested gate (>= 1)")
    ap.add_argument('--dp-check', action='store_true',
                    help="zero-fit grid / DP measurement (A2.1-A2.2), writes "
                         "replay_dp_check.csv")
    ap.add_argument('--evaluate', action='store_true',
                    help="pair a joint run with a reference run (zero fits); needs "
                         "--joint and --reference, writes evaluation_tasks.csv and "
                         "evaluation_summary.csv into --out and prints the verdict")
    ap.add_argument('--joint', default=None,
                    help="--evaluate: the joint run's Result folder (or its RESULT_NAME)")
    ap.add_argument('--reference', default=None,
                    help="--evaluate: the reference run's Result folder (or RESULT_NAME): the same "
                         "config with JOINT_W_OPT off and the W_OPT methods on, untuned")
    ap.add_argument('--exclude-fallbacks', action='store_true',
                    help="--evaluate: the headline verdict excludes tasks whose joint run logged a "
                         "class C fallback (both scopes are always printed and written)")
    ap.add_argument('--eval-bootstraps', type=int, default=500,
                    help="--evaluate: row-bootstrap resamples per task for the delta CIs")
    args = ap.parse_args()
    if args.evaluate:
        if not args.joint or not args.reference:
            ap.error("--evaluate needs --joint and --reference")
        evaluate(args.joint, args.reference, args.out, exclude_fallbacks=args.exclude_fallbacks,
                 n_boot=args.eval_bootstraps, seed=args.seed)
        return
    if not args.dumps:
        ap.error("--dumps is required unless --evaluate is given")
    ests = [e.strip() for e in args.estimators.split(',') if e.strip()]
    rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    for e in ests:
        if e not in ESTIMATORS:
            ap.error(f"unknown estimator {e!r}; known: {', '.join(ESTIMATORS)}")
    for r in rules:
        if r not in RULE_CHOICES:
            ap.error(f"unknown rule {r!r}; known: {', '.join(RULE_CHOICES)}")
    ns_variants = ([v.strip() for v in args.ns_variants.split(',') if v.strip()]
                   if args.ns_variants is not None else None)
    for v in (ns_variants or []):
        if v not in NS_VARIANTS:
            ap.error(f"unknown --ns-variants entry {v!r}; known: {', '.join(NS_VARIANTS)}")
    if not 0.0 <= args.ns_shrink <= 1.0:
        ap.error("--ns-shrink must be in [0, 1]")
    if args.ns_outer_folds < 2 or args.ns_outer_repeats < 1:
        ap.error("--ns-outer-folds must be >= 2 and --ns-outer-repeats >= 1")
    run(args.dumps, args.out, ests, rules, args.bootstraps, args.weight_bags, args.seed,
        skip_variants=args.skip_variants, members_per_model=args.members_per_model,
        wf_rules=[r.strip() for r in args.wf_rules.split(',') if r.strip()],
        wf_max_candidates=args.wf_max_candidates, wf_dedup_tol=args.wf_dedup_tol,
        wf_include_untuned=not args.wf_no_untuned, wf_pick=args.wf_pick,
        wfcv_direction=args.wfcv_direction, wfcv_alpha_rule=args.wfcv_alpha_rule,
        wfcv_repeats=args.wfcv_repeats, n_valid_frac=args.n_valid_frac,
        dp_check_flag=args.dp_check,
        ns_lambdas=([float(v) for v in args.ns_lambdas.split(',') if v.strip()]
                    if args.ns_lambdas is not None else None),
        ns_rule=args.ns_rule, ns_repeats=args.ns_repeats, ns_objective=args.ns_objective,
        ns_swap_rule=args.ns_swap_rule, ns_shrink=args.ns_shrink, ns_deploy=args.ns_deploy,
        ns_variants=ns_variants, ns_seed=args.ns_seed, ns_parity=args.ns_parity,
        ns_outer_gate=args.ns_outer_gate, ns_outer_folds=args.ns_outer_folds,
        ns_outer_repeats=args.ns_outer_repeats)


if __name__ == '__main__':
    main()
