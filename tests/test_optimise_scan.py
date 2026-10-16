"""Regression tests for the optimise_scan update (findings F1-F9 of its Change Summary).

Run from the repository root:  python -m pytest tests/test_optimise_scan.py -q
Everything is generated here (no data file, no torch / R needed except where noted)."""
import ast
import collections
import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding='utf-8') as fh:
        return fh.read()


# ----------------------------------------------------------------------------- F1
def _als_inputs(effect_rows):
    """Two models 'KNN', 'RF'; `effect_rows` is a list of (model, [marker effects]) tuples."""
    rng = np.random.default_rng(0)
    n = 12

    def split():
        return pd.DataFrame({'id': [f'g{i}' for i in range(n)], 'actual': rng.normal(size=n),
                             'KNN': rng.normal(size=n), 'RF': rng.normal(size=n)})
    record = pd.DataFrame({'model': ['KNN', 'RF'], 'Pearson correlation': [0.1, 0.2], 'MSE': [1.0, 1.1]})
    cols = ['model', 'population', 'phenotype', 'ratio', 'sample', 'm0', 'm1', 'm2']
    rows = [[m, 'A', 'y', 0.8, 1] + list(v) for m, v in effect_rows]
    effect = pd.DataFrame(rows, columns=cols)
    return split(), split(), split(), record, effect, pd.DataFrame()


def _run_als(effect_rows):
    from models.Analytic_least_squares import Analytic_least_squares
    tr, va, te, rec, eff, wt = _als_inputs(effect_rows)
    return Analytic_least_squares(tr, va, te, rec, eff, wt, ['KNN', 'RF'], [1e-6, 1.0])


def test_f1_first_model_without_effect_rows_no_longer_crashes():
    # KNN (index 0) has no effect rows - this raised UnboundLocalError before the fix.
    _rec, effect, *_ = _run_als([('RF', [1.0, 2.0, 1.0])])
    assert effect['model'].iloc[-1] == 'Analytic least-squares'
    row = effect.iloc[-1][['m0', 'm1', 'm2']].astype(float).to_numpy()
    assert np.all(np.isfinite(row)) and row.sum() > 0


def test_f1_no_effect_rows_at_all_skips_the_ensemble_effect_row():
    _rec, effect, *_ = _run_als([])
    assert effect.shape[0] == 0           # unchanged, and no exception


def test_f1_all_models_with_effects_is_the_weighted_sum():
    _rec, effect, _te, _va, _tr, weight = _run_als([('KNN', [1.0, 1.0, 2.0]), ('RF', [2.0, 1.0, 1.0])])
    w = weight[['KNN', 'RF']].iloc[-1].astype(float).to_numpy()
    w = w / w.sum()
    expect = w[0] * np.array([.25, .25, .5]) + w[1] * np.array([.5, .25, .25])
    got = effect.iloc[-1][['m0', 'm1', 'm2']].astype(float).to_numpy()
    assert np.allclose(got, expect)


@pytest.mark.parametrize('fn', ['models/Analytic_least_squares.py', 'models/Nelder_Mead.py',
                                'models/Bayesian_optimisation.py', 'models/Linear_transformation.py'])
def test_f1_all_four_weight_methods_share_the_fixed_pattern(fn):
    src = _read(fn)
    assert 'effect_weighted = None' in src
    assert 'if effect_weighted is None:' in src
    assert 'if effect_weighted is not None:' in src
    assert not re.search(r'^\s*if i == 0:\s*$', src, re.M)


# ----------------------------------------------------------------------------- F3
def test_f3_defer_result_sort_is_off_by_default_and_opt_in():
    import pipeline_utils as pu
    assert 'DEFER_RESULT_SORT' in pu._ACCELERATION_KEYS
    base = pu.resolve_compute_resources({'N_JOBS': 1})
    assert base['accel_defer_result_sort'] is False
    on = pu.resolve_compute_resources({'N_JOBS': 1, 'ACCELERATION': {'DEFER_RESULT_SORT': True}})
    assert on['accel_defer_result_sort'] is True


def test_f3_stable_single_sort_equals_task_order_within_ties():
    # The deferred path: one stable sort over the task-ordered concatenation keeps tied ids in task order.
    parts = [pd.DataFrame({'id': ['b', 'a', 'b'], 't': [i] * 3}) for i in range(3)]
    got = pd.concat(parts).sort_values(['id'], kind='stable').reset_index(drop=True)
    assert got['id'].tolist() == ['a'] * 3 + ['b'] * 6
    assert got.loc[got['id'] == 'b', 't'].tolist() == [0, 0, 1, 1, 2, 2]


# ----------------------------------------------------------------------------- F4
_MODELS_DIRECT = None


def _calls_by_region():
    tree = ast.parse(_read('genomic_prediction.py'))
    runner = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == 'ModelTrialRunner')
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_process_one_task')
    modules = set(re.findall(r'^from models\.([A-Za-z_]+) import \*', _read('genomic_prediction.py'), re.M))

    def collect(node):
        out = collections.defaultdict(set)
        for c in ast.walk(node):
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in modules:
                out[c.func.id].add((len(c.args), tuple(sorted(k.arg for k in c.keywords if k.arg))))
        return out
    return collect(runner), collect(main)


def test_f4_dual_dispatch_stays_in_lockstep():
    """ModelTrialRunner (tuning / joint / SL) and the main per-task loop are two implementations of
    'how to call a model' (Phase-2 standard: a signature change goes into BOTH). Every model the two
    call directly by name must use the same positional count and keyword names in both places."""
    runner, main = _calls_by_region()
    common = sorted(set(runner) & set(main))
    assert len(common) >= 6, f"guard became vacuous: only {common} are called in both dispatchers"
    drift = {m: (runner[m], main[m]) for m in common if runner[m] != main[m]}
    assert not drift, f"dispatcher signature drift (update BOTH places): {drift}"


# ----------------------------------------------------------------------------- F6
def test_f6_no_defined_name_collision_between_star_imported_model_modules():
    src = _read('genomic_prediction.py')
    modules = re.findall(r'^from (models\.[A-Za-z_]+) import \*', src, re.M)
    defined = collections.defaultdict(set)
    for mod in modules:
        tree = ast.parse(_read(mod.replace('.', '/') + '.py'))
        for n in tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith('_'):
                defined[n.name].add(mod)
    clash = {k: sorted(v) for k, v in defined.items() if len(v) > 1}
    assert not clash, f"later star-import silently wins for: {clash}"


# ----------------------------------------------------------------------------- F7
def test_f7_single_resource_advise_with_the_gui_default_quantile():
    assert not os.path.exists(os.path.join(ROOT, 'resource_advise.py'))
    src = _read('tools/resource_advise.py')
    assert "'mem_quantile', 0.95" in src.replace('"', "'")
    assert '--keep-batch-size' in src


# ----------------------------------------------------------------------------- F2 / F5 / F8
def test_f2_f5_dead_locals_are_gone():
    src = _read('genomic_prediction.py')
    used = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}     # code, not comments
    for dead in ('_asreml_last_per_set_grm_matrices', '_asreml_response_unit_for_log',
                 '_asreml_model_source_for_log', '_asreml_mask_valid_test_for_log',
                 '_asreml_use_entire_dataset_for_log', '_joint_t0', '_asreml_main_grm_diagnostics',
                 '_asreml_unmasked_grm_diag_cache'):
        assert dead not in used, dead
    # the checkpoint-contract keys stay
    assert "'grm_train'" in src and 'asreml_grm_train' in src


def test_f8_removed_helpers_have_no_references():
    names = ['_rank_scaled_alpha', '_circos_gene_count', '_plan_waves', '_join_and', 'group_colour_map',
             'strategy_file_tag', '_apply_joint_accuracy']
    blob = ''
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('__pycache__', 'tests')]
        for f in fn:
            if f.endswith('.py'):
                blob += open(os.path.join(dp, f), encoding='utf-8', errors='ignore').read()
    for n in names:
        assert not re.search(r'\b' + re.escape(n) + r'\b', blob), n
