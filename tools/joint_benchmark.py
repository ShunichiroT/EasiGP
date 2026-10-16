"""joint_benchmark.py - ver4-34, R2.d / ver4-38, R2.a (INVESTIGATE): benchmark of the joint search.

Why: whether the joint-search levers (bagged/cluster or ensemble-selection weights, CV objective,
library strategies, priority search space, racing, ...) really beat the procedures the joint row
competes with must be decided on the user's own data before any preset changes. This script
never fits a model and is never imported by the engine; it (1) writes one engine config per arm
and (2) analyses the finished runs, reading STANDARD outputs only (`Metric`, `Prediction_result_
valid`, `Prediction_result_test`, `Weight`, and the run logs for fit counts). Arms (identical
seeds, so tasks pair across arms):

  B0 ver4-32 defaults | B1 ver4-32 recommended preset | N1 B1 + bagged_cluster
  N2 N1 + cv_train_valid | N3 N2 + parallel_library + priority + freeze_low_weight=0.05
  N3v N3 with objective_data='valid' (speed-oriented variant)

ver4-36 adds the blueprint arms (all keys appended to the above):
  F1 = JOINT_FAST_SETTINGS (+ library_dump) | A1 = F1 + cv_train_valid + ensemble_selection +
  selection_rule 'bbc' + noise_aware_surrogate (+ library_dump) | A2 = A1 with bagged_cluster |
  A3 = A1 scored on 'valid'.
ver4-38: F1 gains the R1 keys of the fast preset (fold racing, value stop rule, search-fidelity
final weights, dead-column guard) and A4 = A1 + library_ensemble is added (the R2.b2 arm).
ver4-40 adds WF = F1 with search_strategy='weight_first' (+ library_dump; R2 arm, A2.6).
ver4-42 adds WFCV = F1 with search_strategy='weight_first_cv' and WFCV-W = WFCV with
wf_cv_tune=False (weights only; A1.5). Neither writes a library dump.
ver4-44 adds the pure-mode arms MONO-PURE (monolithic), WFCV-PURE and WFCV-PURE-BAG (weight_first_cv
with wf_cv_final 'refit' / 'bagged'), all with safety_nets=False, and the reference arm REF
(JOINT_W_OPT off, the three W_OPT methods on, untuned; `make-configs --ref-hp <gui config>` supplies
their settings). REF has no joint row: pair each joint arm with it through
`tools/joint_replay.py --evaluate --joint Result/<base>_<arm> --reference Result/<base>_REF`.
ver4-48 adds the neural-stack arms NS-PURE (search_strategy='neural_stack', safety_nets=False: the
network's own answer, a measurement, never a preset) and NS-NETS (the same with the safety nets on,
i.e. lambda = infinity - the untuned naive ensemble - is a candidate); both score by
objective_data='cv_train_valid'. Judge NS-PURE like the other pure arms (`--evaluate` against REF).
ver4-50 adds NS-P (NS-NETS + ns_objective='pearson': the network is trained on ranking only) and
NS-PS (NS-P + ns_swap_rule='paired_se': a hyperparameter switch needs a paired CV gain above its
standard error). Both write a library dump (`library_dump=True`) so `tools/joint_replay.py
--rules neural_stack --ns-variants V0,V1,V2,V3` can rank the variants offline (gate A-NS.1) before
a real run confirms the best one.
ver4-51 adds NS-R: the GUI preset 'Neural stack (robust)' (ns_objective='pearson', ns_swap_rule=
'paired_refit', ns_outer_gate='nested', ns_deploy='argmax', ns_weight_shrink=0.5) plus
`library_dump=True`; rank it offline with `tools/joint_replay.py --ns-variants V0,V4,V5` (gate
A-NS.2, `A-NS.2 DECISION:` line, replay_ans2.csv).
`--arms B0,F1,A1,A2,A3` restricts `make-configs` / `analyse` to the blueprint set. F1 and A1
write `Result/<name>/Joint_library/*.npz`; rank the variants offline with tools/joint_replay.py
first, then confirm the top two with real runs.

Per task and arm (ver4-38, R2.a) the analysis computes, on the TEST rows:

  J  the joint row's Pearson r              N  the naive 'ensemble' row
  S  the single model with the best VALIDATION r in that task, scored on test (an honest
     selection procedure)
  F  the single model with the highest MEAN test r over all tasks of the arm (best on average)
  O  the per-task test-best single model (an ORACLE - reported, never a target or a gate)

and the paired differences J-N, J-S, J-F (mean, bootstrap 95 % CI, one-sided Wilcoxon) plus the
MSE difference J-N. It also checks that the r it recomputes from the prediction files equals the
`Pearson correlation` column of Metric.csv (max absolute difference reported; they must agree
to 1e-9 when no BLUE weighting is used).

Usage (run from the EasiGP source directory):

  python tools/joint_benchmark.py make-configs --base cfg.json --out arms/
      # run the pipeline once per arms/<arm>_config.json (>= 2 phenotypes x >= 20 replicates,
      # split ratio (0.5, 0.25, 0.25)); each arm writes to RESULT_NAME '<base>_<arm>'
  python tools/joint_benchmark.py analyse --base-name <base> --out bench_out \\
      [--log-glob 'logs/{arm}_*.log'] [--time-target-s 600] [--ceiling-gap-f 0.008]

`--base` is the user's six-model joint config (JOINT settings should hold only n_iter,
init_points and seed). `analyse` writes `Joint_benchmark_summary.csv` (one row per arm: the
mean r of J, N, S, F, O, the three paired comparisons, the MSE difference, effective models,
fits, seconds, the Metric reproduction check and a pass flag per criterion), the supporting
`benchmark_task_baselines.csv`, `benchmark_task_metrics.csv`, `benchmark_arms.csv` and
`benchmark_paired.csv`, and prints the table and the decision. They are tool outputs, never read
by the engine and not part of RESULT_FILE_NAMES.

Pass criteria per arm (planning document R2): mean dr(J-N) > 0 and mean dr(J-S) > 0, each with
one-sided Wilcoxon p < 0.05; mean MSE(J) - MSE(N) < 0; mean dr(J-F) > 0. When the replay
(tools/joint_replay.py) measured the ceiling over F below 0.01, pass `--ceiling-gap-f <gap>`:
the J-F criterion then becomes non-inferiority (mean dr(J-F) >= -0.005) and the report states
the measured ceiling. Among passing arms within the time target the best mean dr(J-N) wins; if
no arm passes the report says so for each arm and NO fallback is added.
"""
import argparse
import copy
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JOINT_LABEL = 'Joint Bayesian optimisation'     # the Metric.csv / Weight.csv 'model' value
JOINT_COLUMN = 'Joint Bayesian'                 # the Prediction_result_* column name
NAIVE_LABEL = 'ensemble'
META_COLS = ('id', 'population', 'ratio', 'phenotype', 'sample', 'actual')
TASK_KEY = ['population', 'phenotype', 'ratio', 'sample']
# Prediction columns written by the weighted-ensemble methods (diversity_summary.py's
# `_WOPT_COLUMN_PREFIXES`, copied because this tool does not import engine modules). A column
# equal to a prefix, or starting with '<prefix>__', is not a single model.
_WOPT_COLUMN_PREFIXES = ('Linear transformation', 'Nelder-Mead', 'Bayesian',
                         'Analytic least-squares', 'Joint Bayesian')
NON_INFERIORITY_MARGIN = -0.005
CEILING_NEGLIGIBLE = 0.01
# Keep in step with main_app.JOINT_RECOMMENDED_SETTINGS (main_app cannot be imported, I1).
_B1 = {'search_strategy': 'block_coordinate', 'freeze_fidelity_fields': True,
       'warm_start': 'individual', 'score_weight_rule': 'shrunk', 'search_patience': 8,
       'gate_fit_cache': True, 'reuse_confirmatory_fit': True, 'parallel_trials': True}
_N1 = {**_B1, 'weight_estimator': 'bagged_cluster'}
_N2 = {**_N1, 'objective_data': 'cv_train_valid'}
_N3 = {**_N2, 'search_strategy': 'parallel_library', 'search_space': 'priority',
       'freeze_low_weight': 0.05}
# ver4-36, R2.d / ver4-38, R1.e: the fast / accuracy arms. Keep in step with
# main_app.JOINT_FAST_SETTINGS (main_app cannot be imported, I1). F1 = the fast preset; A1 = F1 +
# the four accuracy levers; A2 = A1 with 'bagged_cluster' instead of 'ensemble_selection'; A3 =
# A1 scored on 'valid'; A4 = A1 + the library ensemble (R2.b2). F1 and A1 also write the library
# dump (`library_dump`, replay with tools/joint_replay.py).
_F1 = {'search_strategy': 'async_library', 'search_space': 'priority', 'gate_mode': 'bbc',
       'search_test_stub': True, 'search_engine': 'exact_kernel', 'freeze_fidelity_fields': True,
       'reuse_confirmatory_fit': True, 'parallel_trials': True,
       'cv_racing': True, 'library_stop_rule': 'value', 'freeze_low_weight': 0.02,
       'oof_fidelity': 'search', 'dead_column_guard': True}
_WF = {**_F1, 'search_strategy': 'weight_first'}
_WFCV = {**_F1, 'search_strategy': 'weight_first_cv'}
_WFCV_W = {**_WFCV, 'wf_cv_tune': False}
_A1 = {**_F1, 'objective_data': 'cv_train_valid', 'weight_estimator': 'ensemble_selection',
       'selection_rule': 'bbc', 'noise_aware_surrogate': True}
_A2 = {**_A1, 'weight_estimator': 'bagged_cluster'}
_A3 = {**_A1, 'objective_data': 'valid'}
_A4 = {**_A1, 'library_ensemble': True}
# ver4-44, R1/R2/R3: the pure-mode arms (no safety nets, so the joint row is the optimiser's own
# answer) and the reference arm. `REF` is NOT a joint arm: `make_configs` writes it with
# JOINT_W_OPT off and the W_OPT methods on (untuned), the independent reference that
# `tools/joint_replay.py --evaluate` pairs each joint arm against. The pure arms keep only the
# strategy and `safety_nets` (plus `wf_cv_final` for the bagged twin); n_iter / init_points / seed
# come from the base config's JOINT settings so every arm shares them.
_MONO_PURE = {'search_strategy': 'monolithic', 'safety_nets': False}
_WFCV_PURE = {**_F1, 'search_strategy': 'weight_first_cv', 'safety_nets': False,
              'wf_cv_final': 'refit'}
_WFCV_PURE_BAG = {**_WFCV_PURE, 'wf_cv_final': 'bagged'}
# ver4-48, R3: the neural-stack arms. Both need the cross-validation objective (the engine forces it
# and logs N20 otherwise); n_iter / init_points / seed come from the base config as for every arm.
_NS_PURE = {'search_strategy': 'neural_stack', 'safety_nets': False,
            'objective_data': 'cv_train_valid'}
_NS_NETS = {'search_strategy': 'neural_stack', 'objective_data': 'cv_train_valid'}
# ver4-50, R4: the two arms that attribute a gain to each ver4-50 change (never bundled).
_NS_P = {**_NS_NETS, 'ns_objective': 'pearson', 'library_dump': True}
_NS_PS = {**_NS_P, 'ns_swap_rule': 'paired_se'}
# ver4-51, R4: the robust preset of the GUI (`main_app.JOINT_NS_ROBUST_SETTINGS`: pearson objective,
# fair refitted swap test, nested outer gate, argmax deployment, shrink 0.5) with the library dump
# on, so gate A-NS.2 can be measured offline (`tools/joint_replay.py --ns-variants V0,V4,V5`).
_NS_R = {'search_strategy': 'neural_stack', 'safety_nets': True,
         'objective_data': 'cv_train_valid', 'ns_objective': 'pearson',
         'ns_swap_rule': 'paired_refit', 'ns_outer_gate': 'nested', 'ns_deploy': 'argmax',
         'ns_weight_shrink': 0.5, 'library_dump': True}
REF_ARM = 'REF'
REF_WOPT_METHODS = ['Analytic least-squares', 'Nelder Mead', 'Bayesian optimisation']
ARMS = {'B0': {}, 'B1': _B1, 'N1': _N1, 'N2': _N2, 'N3': _N3,
        'N3v': {**_N3, 'objective_data': 'valid'},
        'F1': {**_F1, 'library_dump': True}, 'A1': {**_A1, 'library_dump': True},
        'A2': _A2, 'A3': _A3, 'A4': _A4, 'WF': {**_WF, 'library_dump': True},
        'WFCV': _WFCV, 'WFCV-W': _WFCV_W,
        'MONO-PURE': _MONO_PURE, 'WFCV-PURE': _WFCV_PURE, 'WFCV-PURE-BAG': _WFCV_PURE_BAG,
        'NS-PURE': _NS_PURE, 'NS-NETS': _NS_NETS, 'NS-P': _NS_P, 'NS-PS': _NS_PS,
        'NS-R': _NS_R}
# Arms `analyse` can score as joint arms; REF has no joint row and is evaluated by
# `tools/joint_replay.py --evaluate`, never by `analyse`.
PURE_ARMS = ('MONO-PURE', 'WFCV-PURE', 'WFCV-PURE-BAG', 'NS-PURE')
# The blueprint's own arm set (B0, F1, A1, A2, A3); the ver4-34 arms and A4 stay available.
BLUEPRINT_ARMS = ('B0', 'F1', 'A1', 'A2', 'A3')
# A dump written with the stub on is impossible (the dump forces the stub off with a NOTE), so a
# dump arm runs a little slower than its fast twin - compare F1/A1 times with that in mind.
ACCURACY_ARMS = ('A1', 'A2', 'A3', 'A4')


def _ref_config(base, ref_hp_path=None):
    """The reference arm's config (ver4-44 R2): the base config with JOINT_W_OPT off, every
    model untuned (HP_TUNE None) and the three W_OPT methods on. The methods' own settings
    (`HYPERPARAMETERS_OPT[<method>]`, a positional list the engine indexes directly) cannot be
    invented here, so they come from `ref_hp_path` (a GUI-saved config with those methods ticked)
    or from the base config; returns None, with a message, when either lacks one of them. Pure
    apart from reading `ref_hp_path`."""
    source = base
    if ref_hp_path:
        with open(ref_hp_path, encoding='utf-8') as fh:
            source = json.load(fh)
    hp = source.get('HYPERPARAMETERS_OPT') or {}
    missing = [m for m in REF_WOPT_METHODS if not isinstance(hp.get(m), list)]
    if missing:
        print(f"[BENCH] arm {REF_ARM}: not written - HYPERPARAMETERS_OPT of "
              f"{'--ref-hp' if ref_hp_path else 'the base config'} has no settings for "
              f"{missing}. Tick {', '.join(REF_WOPT_METHODS)} in the GUI (Ensemble tab), save "
              f"that config and pass it as --ref-hp; the reference must use the same data, "
              f"models, RATIO, SAMPLE_NUM and seeds as the joint arms.")
        return None
    cfg = copy.deepcopy(base)
    cfg['JOINT_W_OPT'], cfg['HYPERPARAMETERS_JOINT_OPT'], cfg['HP_TUNE'] = None, None, None
    cfg['W_OPT'] = list(REF_WOPT_METHODS)
    cfg['HYPERPARAMETERS_OPT'] = {m: copy.deepcopy(hp[m]) for m in REF_WOPT_METHODS}
    return cfg


def make_configs(base_path, out_dir, arms=None, ref_hp_path=None):
    """Write `<out_dir>/<arm>_config.json` for every arm; returns the written paths. ver4-44: the
    `REF` arm (the reference run for `tools/joint_replay.py --evaluate`) is written like the
    others when selected (or when `arms` is None) and its W_OPT settings are available (see
    `_ref_config`)."""
    with open(base_path, encoding='utf-8') as fh:
        base = json.load(fh)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for arm, settings in [*ARMS.items(), (REF_ARM, None)]:
        if arms and arm not in arms:
            continue
        if arm == REF_ARM:
            cfg = _ref_config(base, ref_hp_path)
            if cfg is None:
                continue
        else:
            cfg = copy.deepcopy(base)
            cfg['JOINT_W_OPT'] = [JOINT_LABEL]
            cfg['HP_TUNE'], cfg['W_OPT'], cfg['HYPERPARAMETERS_OPT'] = None, None, {}
            cfg['HYPERPARAMETERS_JOINT_OPT'] = {**(base.get('HYPERPARAMETERS_JOINT_OPT') or {}),
                                                **settings}
        cfg['RESULT_NAME'] = f"{base['RESULT_NAME']}_{arm}"
        path = os.path.join(out_dir, f'{arm}_config.json')
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(cfg, fh, indent=2)
        paths.append(path)
        print(f"[BENCH] wrote {path} (RESULT_NAME={cfg['RESULT_NAME']})")
    return paths


# ------------------------------------------------------------------------------------------
# Columns and per-task metrics
# ------------------------------------------------------------------------------------------

def _is_wopt_column(name):
    return isinstance(name, str) and any(
        name == p or name.startswith(p + '__') for p in _WOPT_COLUMN_PREFIXES)


def classify_columns(pred):
    """Split the prediction table's columns into `{'joint': column or None, 'naive': column or
    None, 'base': [single-model columns]}`. The joint prediction column is 'Joint Bayesian'
    (its Metric.csv label is 'Joint Bayesian optimisation'); the naive row is 'ensemble'
    (possibly with a '__<group>' suffix). Pure."""
    joint = [c for c in pred.columns if c == JOINT_COLUMN or str(c).startswith(JOINT_COLUMN + '__')]
    naive = [c for c in pred.columns
             if c == NAIVE_LABEL or str(c).startswith(NAIVE_LABEL + '__')]
    base = [c for c in pred.columns if c not in META_COLS and not _is_wopt_column(c)
            and c not in naive]
    return {'joint': joint[0] if joint else None, 'naive': naive[0] if naive else None,
            'base': base}


def label_of(column, cols):
    """The Metric.csv-style label of prediction column `column`."""
    if column == cols['joint']:
        return JOINT_LABEL
    if column == cols['naive']:
        return NAIVE_LABEL
    return column


def _spearman(a, b):
    with np.errstate(invalid='ignore', divide='ignore'):   # a constant column gives NaN
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def _top_coincidence(actual, pred, frac=0.10):
    k = max(1, int(round(frac * len(actual))))
    top_a = set(np.argsort(-np.asarray(actual))[:k].tolist())
    top_p = set(np.argsort(-np.asarray(pred))[:k].tolist())
    return len(top_a & top_p) / float(k)


def _pearson(y, p):
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    if y.size < 3 or not np.all(np.isfinite(p)) or p.std() <= 0 or y.std() <= 0:
        return np.nan
    return float(np.corrcoef(y, p)[0, 1])


def _task_frame(pred):
    """`pred` with a string 'ratio' (so tuple ratios compare equal across files)."""
    out = pred.copy()
    out['ratio'] = out['ratio'].astype(str)
    return out


def task_metrics(pred, arm):
    """Long table: one row per (task, model) with r, MSE, Spearman, Top-10 coincidence. The
    model label of the joint column is 'Joint Bayesian optimisation' and of the naive column
    'ensemble' (ver4-38: the joint column used to be matched by the wrong name)."""
    rows = []
    cols = classify_columns(pred)
    models = [c for c in pred.columns if c not in META_COLS]
    for (pop, pheno, ratio, sample), g in _task_frame(pred).groupby(TASK_KEY, sort=False):
        y = g['actual'].to_numpy(float)
        for m in models:
            p = g[m].to_numpy(float)
            rows.append(dict(arm=arm, population=pop, phenotype=pheno, ratio=str(ratio),
                             sample=sample, model=label_of(m, cols), r=_pearson(y, p),
                             mse=float(np.mean((y - p) ** 2)), spearman=_spearman(y, p),
                             top10=_top_coincidence(y, p)))
    return pd.DataFrame(rows)


def baselines(test_pred, valid_pred):
    """ver4-38, R2.a: one row per task with the joint row and the four baselines, all scored on
    the TEST rows. Pure (no files).

    Columns: the task key, `r_J`/`mse_J`, `r_N`/`mse_N` (naive row), `S_model` + `r_S`/`mse_S`
    (the model with the best VALIDATION r in that task, ties and NaN -> the first finite one),
    `F_model` + `r_F`/`mse_F` (the model with the highest mean test r over all tasks of this
    table), `O_model` + `r_O` (per-task test-best model: an oracle, report only) and
    `n_models`. A baseline that cannot be formed (no validation rows, no naive column) is NaN.
    Raises KeyError when `test_pred` lacks the standard columns."""
    cols = classify_columns(test_pred)
    base = cols['base']
    if cols['joint'] is None or not base:
        return pd.DataFrame()
    valid = _task_frame(valid_pred) if valid_pred is not None and not valid_pred.empty else None
    valid_groups = ({k: g for k, g in valid.groupby(TASK_KEY, sort=False)}
                    if valid is not None else {})
    rows, per_model_r = [], []
    for key, g in _task_frame(test_pred).groupby(TASK_KEY, sort=False):
        y = g['actual'].to_numpy(float)
        row = dict(zip(TASK_KEY, key))
        r_model = {m: _pearson(y, g[m].to_numpy(float)) for m in base}
        per_model_r.append(r_model)
        row['r_J'] = _pearson(y, g[cols['joint']].to_numpy(float))
        row['mse_J'] = float(np.mean((y - g[cols['joint']].to_numpy(float)) ** 2))
        if cols['naive'] is not None:
            pn = g[cols['naive']].to_numpy(float)
            row['r_N'], row['mse_N'] = _pearson(y, pn), float(np.mean((y - pn) ** 2))
        else:
            row['r_N'] = row['mse_N'] = np.nan
        vg = valid_groups.get(key)
        s_model = None
        if vg is not None and len(vg):
            yv = vg['actual'].to_numpy(float)
            vr = {m: _pearson(yv, vg[m].to_numpy(float)) for m in base if m in vg.columns}
            vr = {m: v for m, v in vr.items() if np.isfinite(v)}
            if vr:
                s_model = max(vr, key=lambda m: (vr[m], -base.index(m)))
        row['S_model'] = s_model
        if s_model is not None:
            ps = g[s_model].to_numpy(float)
            row['r_S'], row['mse_S'] = r_model[s_model], float(np.mean((y - ps) ** 2))
        else:
            row['r_S'] = row['mse_S'] = np.nan
        finite = {m: v for m, v in r_model.items() if np.isfinite(v)}
        row['O_model'] = max(finite, key=lambda m: (finite[m], -base.index(m))) if finite else None
        row['r_O'] = finite[row['O_model']] if finite else np.nan
        row['n_models'] = len(base)
        rows.append(row)
    out = pd.DataFrame(rows)
    means = pd.DataFrame(per_model_r).mean(axis=0, skipna=True)
    f_model = str(means.idxmax()) if means.notna().any() else None
    out['F_model'] = f_model
    if f_model is not None:
        out['r_F'] = [rm[f_model] for rm in per_model_r]
        mse_f = []
        for key, g in _task_frame(test_pred).groupby(TASK_KEY, sort=False):
            mse_f.append(float(np.mean((g['actual'].to_numpy(float)
                                        - g[f_model].to_numpy(float)) ** 2)))
        out['mse_F'] = mse_f
    else:
        out['r_F'] = out['mse_F'] = np.nan
    return out


def metric_reproduction(test_pred, metric, arm):
    """A-2a.2: recompute r from the prediction file for J, N and every single model and compare
    with the `Pearson correlation` column of Metric.csv. Returns `(max_abs_diff, n_compared)`
    (NaN, 0 when Metric.csv is empty or has no matching rows). Pure."""
    if metric is None or metric.empty or 'Pearson correlation' not in metric.columns:
        return np.nan, 0
    cols = classify_columns(test_pred)
    models = [c for c in test_pred.columns if c not in META_COLS]
    mt = metric.copy()
    mt['ratio'] = mt['ratio'].astype(str)
    lookup = {}
    for pop, pheno, ratio, sample, model, r_metric in zip(
            mt['population'], mt['phenotype'], mt['ratio'], mt['sample'], mt['model'],
            mt['Pearson correlation']):
        lookup[(str(pop), str(pheno), str(ratio), str(sample), str(model))] = float(r_metric)
    diffs = []
    for key, g in _task_frame(test_pred).groupby(TASK_KEY, sort=False):
        y = g['actual'].to_numpy(float)
        for m in models:
            ref = lookup.get(tuple(str(k) for k in key) + (label_of(m, cols),))
            mine = _pearson(y, g[m].to_numpy(float))
            if ref is None or not (np.isfinite(ref) and np.isfinite(mine)):
                continue
            diffs.append(abs(ref - mine))
    return (float(max(diffs)) if diffs else np.nan), len(diffs)


def weight_entropy(weight, models):
    """Mean Shannon entropy (nats) of the joint search's normalised weights, per arm."""
    w = weight[weight['model'] == JOINT_LABEL] if not weight.empty else weight
    cols = [m for m in models if not w.empty and m in w.columns]
    if w.empty or not cols:
        return np.nan
    a = w[cols].abs().to_numpy(float)
    a = a / np.where(a.sum(axis=1, keepdims=True) > 0, a.sum(axis=1, keepdims=True), 1.0)
    return float(np.mean(-(np.where(a > 0, a * np.log(np.where(a > 0, a, 1.0)), 0.0)).sum(axis=1)))


def parse_fit_logs(pattern):
    """Mean fits and Step 8.5 seconds per task from '[joint_tuning] fits:' lines."""
    fits, secs = [], []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                m = re.search(r'\[joint_tuning\] fits: .*? total=(\d+) \| secs: search=([\d.]+) '
                              r'gate=([\d.]+) confirm=([\d.]+) oof=([\d.]+)', line)
                if m:
                    fits.append(int(m.group(1)))
                    secs.append(sum(float(m.group(i)) for i in range(2, 6)))
    return (float(np.mean(fits)) if fits else np.nan, float(np.mean(secs)) if secs else np.nan,
            len(fits))


def paired(a, b, rng, n_boot=2000, alternative='greater'):
    """Paired a - b: mean delta, bootstrap 95% CI and the one-sided Wilcoxon p for
    `alternative` ('greater': a > b; 'less': a < b)."""
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
            p = float(wilcoxon(d, alternative=alternative).pvalue)
    except ValueError:  # all differences are zero
        p = float('nan')
    return dict(n=int(d.size), mean=float(d.mean()), ci_lo=float(np.percentile(boots, 2.5)),
                ci_hi=float(np.percentile(boots, 97.5)), p_greater=p)


def arm_summary(arm, bl, rng, ceiling_gap_f=None):
    """One row of Joint_benchmark_summary.csv for a baselines table `bl`: the mean r of J, N, S,
    F, O, the paired comparisons J-N, J-S, J-F (and the MSE difference J-N, one-sided 'less'),
    and the pass flag of every criterion. Pure. `ceiling_gap_f`: the replay's measured ceiling
    over F; below 0.01 the J-F criterion is non-inferiority (>= -0.005)."""
    row = dict(arm=arm, tasks=int(len(bl)),
               F_model=(bl['F_model'].iloc[0] if len(bl) else None))
    for k in ('J', 'N', 'S', 'F', 'O'):
        row[f'r_{k}'] = float(np.nanmean(bl[f'r_{k}'])) if len(bl) and bl[f'r_{k}'].notna().any() \
            else np.nan
    for tag, ref in (('JN', 'N'), ('JS', 'S'), ('JF', 'F')):
        p = paired(bl['r_J'], bl[f'r_{ref}'], rng)
        for key in ('n', 'mean', 'ci_lo', 'ci_hi', 'p_greater'):
            row[f'd_{tag}_{key}'] = p[key]
    pm = paired(bl['mse_J'], bl['mse_N'], rng, alternative='less')
    row['dMSE_JN_mean'], row['dMSE_JN_p_less'] = pm['mean'], pm['p_greater']
    row['pass_JN'] = bool(row['d_JN_mean'] > 0 and row['d_JN_p_greater'] < 0.05)
    row['pass_JS'] = bool(row['d_JS_mean'] > 0 and row['d_JS_p_greater'] < 0.05)
    row['pass_MSE'] = bool(row['dMSE_JN_mean'] < 0)
    non_inf = (ceiling_gap_f is not None and np.isfinite(ceiling_gap_f)
               and ceiling_gap_f < CEILING_NEGLIGIBLE)
    row['JF_criterion'] = (f"non-inferiority (>= {NON_INFERIORITY_MARGIN}); measured ceiling "
                           f"over F = {ceiling_gap_f:.4f}" if non_inf else 'superiority (> 0)')
    row['pass_JF'] = bool((row['d_JF_mean'] >= NON_INFERIORITY_MARGIN) if non_inf
                          else (row['d_JF_mean'] > 0))
    row['pass_all'] = bool(row['pass_JN'] and row['pass_JS'] and row['pass_MSE']
                           and row['pass_JF'])
    return row


def analyse(base_name, out_dir, log_glob, time_target, arms=None, ceiling_gap_f=None):
    from batch_reader import load_combined
    rng = np.random.default_rng(0)
    os.makedirs(out_dir, exist_ok=True)
    tables, base_tables, arm_rows, summary_rows = [], [], [], []
    for arm in ARMS:
        if arms and arm not in arms:
            continue
        name = f'{base_name}_{arm}'
        pred = load_combined(name, 'result_test')
        if pred.empty:
            print(f"[BENCH] arm {arm}: no Prediction_result_test for {name!r} - arm skipped")
            continue
        cols = classify_columns(pred)
        if cols['joint'] is None:
            print(f"[BENCH] arm {arm}: no 'Joint Bayesian' prediction column in {name!r} "
                  f"(JOINT_W_OPT not run?) - arm skipped")
            continue
        if cols['naive'] is None:
            print(f"[BENCH] NOTE: arm {arm}: no naive 'ensemble' prediction column - N, J-N and "
                  f"the MSE criterion are NaN for this arm.")
        t = task_metrics(pred, arm)
        tables.append(t)
        valid = load_combined(name, 'result_valid')
        if valid.empty:
            print(f"[BENCH] NOTE: arm {arm}: no Prediction_result_valid - S (the validation-"
                  f"selected single model) cannot be formed for this arm.")
        bl = baselines(pred, valid)
        bl.insert(0, 'arm', arm)
        base_tables.append(bl)
        diff, n_cmp = metric_reproduction(pred, load_combined(name, 'record'), arm)
        if n_cmp and diff > 1e-9:
            print(f"[BENCH] NOTE: arm {arm}: the r recomputed from the prediction file differs "
                  f"from Metric.csv by up to {diff:.3g} over {n_cmp} value(s) (BLUE weighting "
                  f"scores the Metric rows with weights; otherwise investigate).")
        fits, secs, n_logged = ((np.nan, np.nan, 0) if not log_glob
                                else parse_fit_logs(log_glob.format(arm=arm)))
        models = cols['base']
        wentropy = weight_entropy(load_combined(name, 'weight'), models)
        arm_rows.append(dict(arm=arm, tasks=t[t.model == JOINT_LABEL].shape[0], mean_fits=fits,
                             mean_step85_secs=secs, logged_tasks=n_logged,
                             weight_entropy=wentropy))
        srow = arm_summary(arm, bl, rng, ceiling_gap_f)
        srow.update(effective_models=(float(np.exp(wentropy)) if np.isfinite(wentropy)
                                      else np.nan),
                    mean_fits=fits, mean_step85_secs=secs,
                    metric_max_abs_diff=diff, metric_values_compared=n_cmp)
        summary_rows.append(srow)
    if not tables:
        print("[BENCH] no arm had results - nothing to analyse")
        return None
    allm = pd.concat(tables, ignore_index=True)
    allm.to_csv(os.path.join(out_dir, 'benchmark_task_metrics.csv'), index=False)
    pd.concat(base_tables, ignore_index=True).to_csv(
        os.path.join(out_dir, 'benchmark_task_baselines.csv'), index=False)
    arms_df = pd.DataFrame(arm_rows)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, 'Joint_benchmark_summary.csv'), index=False)
    key = TASK_KEY
    joint_r = allm[allm.model == JOINT_LABEL].pivot_table(
        index=key, columns='arm', values='r', aggfunc='first')
    comps = []
    for arm in arms_df['arm']:
        sub = allm[allm.arm == arm].set_index(key)
        j, n = sub[sub.model == JOINT_LABEL], sub[sub.model == NAIVE_LABEL]
        indiv = sub[~sub.model.isin([JOINT_LABEL, NAIVE_LABEL])]
        best = indiv.groupby('model')['r'].mean().idxmax() if len(indiv) else None
        for metric in ('r', 'spearman', 'top10'):
            comps.append(dict(comparison=f'{arm}: joint - naive', metric=metric,
                              **paired(j[metric].reindex(n.index), n[metric], rng)))
        if best is not None:
            best_r = indiv[indiv.model == best]['r'].reindex(j.index)
            comps.append(dict(comparison=f'{arm}: joint - best-on-average ({best})', metric='r',
                              **paired(j['r'], best_r, rng)))
        mse = paired(n['mse'], j['mse'].reindex(n.index), rng)  # lower is better: naive - joint
        comps.append(dict(comparison=f'{arm}: naive - joint (MSE, positive = joint better)',
                          metric='mse', **mse))
        for ref in ('B0', 'B1'):
            if ref in joint_r.columns and arm != ref and arm in joint_r.columns:
                comps.append(dict(comparison=f'{arm} - {ref}: joint', metric='r',
                                  **paired(joint_r[arm], joint_r[ref], rng)))
    pd.DataFrame(comps).to_csv(os.path.join(out_dir, 'benchmark_paired.csv'), index=False)
    arms_df.to_csv(os.path.join(out_dir, 'benchmark_arms.csv'), index=False)
    show = ['arm', 'tasks', 'r_J', 'r_N', 'r_S', 'r_F', 'r_O', 'd_JN_mean', 'd_JN_p_greater',
            'd_JS_mean', 'd_JS_p_greater', 'd_JF_mean', 'dMSE_JN_mean', 'effective_models',
            'mean_fits', 'mean_step85_secs', 'metric_max_abs_diff']
    print(summary[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print("[BENCH] O is the per-task test-best single model (an oracle) - reported only, never a "
          "target.")
    return decide(summary, time_target, ceiling_gap_f)


def decide(summary, time_target=None, ceiling_gap_f=None):
    """Print the per-arm verdict on every criterion and the decision; return the winning arm
    name or None. Pure apart from `print` (no fallback is ever added when no arm passes)."""
    if ceiling_gap_f is not None and np.isfinite(ceiling_gap_f):
        print(f"[BENCH] measured ceiling over F (from the replay) = {ceiling_gap_f:.4f}: the J-F "
              + ("criterion is NON-INFERIORITY (>= -0.005)." if ceiling_gap_f < CEILING_NEGLIGIBLE
                 else "criterion stays superiority (> 0)."))
    for _, r in summary.iterrows():
        print(f"[BENCH] arm {r['arm']}: J-N {'PASS' if r['pass_JN'] else 'fail'} "
              f"(dr={r['d_JN_mean']:+.4f}, p={r['d_JN_p_greater']:.4f}) | "
              f"J-S {'PASS' if r['pass_JS'] else 'fail'} "
              f"(dr={r['d_JS_mean']:+.4f}, p={r['d_JS_p_greater']:.4f}) | "
              f"MSE(J)-MSE(N) {'PASS' if r['pass_MSE'] else 'fail'} "
              f"({r['dMSE_JN_mean']:+.4f}) | J-F {'PASS' if r['pass_JF'] else 'fail'} "
              f"(dr={r['d_JF_mean']:+.4f}; {r['JF_criterion']})")
    ok = summary[summary['pass_all']]
    if len(ok) and ok['arm'].isin(PURE_ARMS).any():
        print("[BENCH] NOTE: the pure arms (safety_nets=False) are measurements, never "
              "presets; they are excluded from the decision. Judge them with "
              "tools/joint_replay.py --evaluate against the REF arm.")
        ok = ok[~ok['arm'].isin(PURE_ARMS)]
    if time_target:
        ok = ok[ok['mean_step85_secs'] <= time_target]
    if ok.empty:
        print("[BENCH] DECISION: no arm passes every criterion"
              + (" within the time target" if time_target else "")
              + " (see the per-arm lines above). No fallback is added; revisit the levers.")
        return None
    best = ok.sort_values('d_JN_mean', ascending=False).iloc[0]
    print(f"[BENCH] DECISION: arm {best['arm']} passes every criterion (dr vs naive "
          f"{best['d_JN_mean']:+.4f}, 95% CI [{best['d_JN_ci_lo']:+.4f}, "
          f"{best['d_JN_ci_hi']:+.4f}], p={best['d_JN_p_greater']:.4f}) -> becomes "
          + ('JOINT_ACCURACY_SETTINGS' if best['arm'] in ACCURACY_ARMS
             else 'JOINT_RECOMMENDED_SETTINGS')
          + f": {json.dumps(ARMS[best['arm']])}")
    return best['arm']


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    mk = sub.add_parser('make-configs')
    mk.add_argument('--base', required=True)
    mk.add_argument('--out', default='arms')
    mk.add_argument('--arms', default=None, help="comma list, e.g. B0,F1,A1,A2,A3 (default: all)")
    mk.add_argument('--ref-hp', default=None,
                    help="a GUI-saved config with the W_OPT methods ticked; its "
                         "HYPERPARAMETERS_OPT feeds the REF arm (else the base config's is used)")
    an = sub.add_parser('analyse')
    an.add_argument('--base-name', required=True, help="the base config's RESULT_NAME")
    an.add_argument('--out', default='bench_out')
    an.add_argument('--log-glob', default=None, help="e.g. 'logs/{arm}_*.log'")
    an.add_argument('--time-target-s', type=float, default=None)
    an.add_argument('--arms', default=None, help="comma list, e.g. B0,F1,A1,A2,A3 (default: all)")
    an.add_argument('--ceiling-gap-f', type=float, default=None,
                    help="the replay's measured ceiling over F (Joint_replay_summary.csv, "
                         "column ceiling_gap_F); below 0.01 the J-F criterion becomes "
                         "non-inferiority")
    args = ap.parse_args()
    arms = [a.strip() for a in args.arms.split(',')] if args.arms else None
    for a in (arms or []):
        if a not in ARMS and not (args.cmd == 'make-configs' and a == REF_ARM):
            ap.error(f"unknown arm {a!r}; known: {', '.join(ARMS)}"
                     + (f", {REF_ARM}" if args.cmd == 'make-configs' else ''))
    if args.cmd == 'make-configs':
        make_configs(args.base, args.out, arms, args.ref_hp)
    else:
        analyse(args.base_name, args.out, args.log_glob, args.time_target_s, arms,
                args.ceiling_gap_f)


if __name__ == '__main__':
    main()
