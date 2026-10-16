"""Tests for ver4-102: abstention (O04), pairwise diversity (D13), class hierarchy (H07 read-out),
class-coverage note (D01 / R2) and the seed bag (P4).

Run from the EasiGP root:  python -m pytest tests/test_ver4_102.py -q
No test needs R, a GPU or a real run; the seed-bag tests use a fake fit function (the torch test
is skipped when torch is not installed).
"""
import io
import os
import sys
import contextlib

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import model_registry                                    # noqa: E402
from models import seed_bag                              # noqa: E402
from tools import candidate_reliability as cr            # noqa: E402
from tools import pairwise_diversity as pdv              # noqa: E402
from tools import class_hierarchy as ch                  # noqa: E402


# ----------------------------------------------------------------------------- fixtures
def _interval(n=60, seed=0, with_ood=True):
    rng = np.random.RandomState(seed)
    k = rng.uniform(0.1, 1.0, n)
    noise = (1.2 - k) * rng.randn(n)                     # related lines are predicted better
    actual = rng.randn(n)
    frame = pd.DataFrame({
        'population': 'P', 'phenotype': 'T', 'ratio': 0.8, 'model': 'Super learner',
        'prediction': actual + noise, 'actual': actual,
        'lower': -1.0 - (1.2 - k), 'upper': 1.0 + (1.2 - k), 'k_train': k})
    if with_ood:
        frame['out_of_domain'] = k < 0.2
    return frame


def _prediction_frame(n_tasks=4, n=40, seed=1, models=('rrBLUP', 'BayesB', 'RF', 'SVR', 'KNN')):
    rng = np.random.RandomState(seed)
    rows = []
    for s in range(n_tasks):
        actual = rng.randn(n)
        common = rng.randn(n)
        part = pd.DataFrame({'population': 'P', 'phenotype': 'T', 'ratio': 0.8, 'sample': s,
                             'actual': actual})
        for j, m in enumerate(models):
            part[m] = 0.5 * actual + 0.3 * common + 0.5 * rng.randn(n) * (1 + 0.1 * j)
        part['ensemble'] = part[list(models)].mean(axis=1)
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------------------- abstention (O04)
def test_abstention_full_fraction_equals_baseline():
    table = cr.abstention_table(_interval())
    full = table[(table['fraction'] == 1.0) & (table['rule'] == 'k_train')].iloc[0]
    assert full['n_kept'] == full['n_total'] == 60
    assert abs(full['d_pearson']) < 1e-12 and abs(full['d_mse']) < 1e-12


def test_abstention_keeps_most_related_and_helps_when_error_tracks_relatedness():
    table = cr.abstention_table(_interval(n=200, seed=3), fractions=(1.0, 0.5))
    half = table[(table['rule'] == 'k_train') & (table['fraction'] == 0.5)].iloc[0]
    assert half['n_kept'] == 100
    assert half['d_mse'] < 0 and half['d_pearson'] > 0      # error shrinks with relatedness here


def test_abstention_out_of_domain_rule_and_min_kept():
    table = cr.abstention_table(_interval())
    ood = table[table['rule'] == 'out_of_domain']
    assert len(ood) == 1 and ood.iloc[0]['n_kept'] < ood.iloc[0]['n_total']
    # min_kept above every kept count removes the row
    assert cr.abstention_table(_interval(), min_kept=1000).empty
    assert cr.abstention_table(_interval(with_ood=False))['rule'].isin(['k_train', 'interval_width']).all()


def test_abstention_rejects_bad_fractions():
    for bad in ((), (0.0,), (1.5,), (-0.1, 1.0)):
        with pytest.raises(ValueError):
            cr.abstention_table(_interval(), fractions=bad)


def test_abstention_empty_input():
    assert cr.abstention_table(_interval().iloc[0:0]).empty


# ----------------------------------------------------------------------------- pairwise diversity (D13)
def test_pair_measures_identical_and_independent():
    rng = np.random.RandomState(0)
    actual = rng.randn(400)
    a = actual + rng.randn(400)
    same = pdv.pair_measures(a, a.copy(), actual)
    assert same['pred_corr'] == pytest.approx(1.0) and same['err_corr'] == pytest.approx(1.0)
    assert same['disagreement'] == 0.0 and same['q_stat'] == pytest.approx(1.0)
    b = actual + rng.randn(400)
    indep = pdv.pair_measures(a, b, actual)
    assert abs(indep['err_corr']) < 0.2 and abs(indep['q_stat']) < 0.4
    assert 0.4 < indep['disagreement'] < 0.6 and 0.15 < indep['double_fault'] < 0.35


def test_pair_measures_rejects_bad_quantile():
    x = np.arange(10.0)
    for bad in (0.0, 1.0, -1, 2):
        with pytest.raises(ValueError):
            pdv.pair_measures(x, x, x, bad)


def test_pairwise_table_and_summary_shape():
    test = _prediction_frame()
    members = pdv.member_columns(list(test.columns))
    assert 'ensemble' not in members and set(members) == {'rrBLUP', 'BayesB', 'RF', 'SVR', 'KNN'}
    table = pdv.pairwise_table(test, members)
    assert len(table) == 4 * 10                              # 4 tasks x C(5,2) pairs
    summary = pdv.summarise(table)
    assert len(summary) == 10 and (summary['n_tasks'] == 4).all()
    assert pdv.pairwise_table(test, members, min_lines=1000).empty


def test_pairwise_cli_round_trip(tmp_path):
    _prediction_frame().to_csv(tmp_path / 'Prediction_result_test.csv', index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        assert pdv.main(['--result', str(tmp_path)]) == pdv.EXIT_OK
    assert (tmp_path / 'Pairwise_diversity.csv').is_file()
    assert (tmp_path / 'Pairwise_diversity_summary.csv').is_file()
    with contextlib.redirect_stdout(io.StringIO()):
        assert pdv.main(['--result', str(tmp_path), '--models', 'RF']) == pdv.EXIT_NOTHING
        assert pdv.main(['--result', str(tmp_path), '--bad-quantile', '1.5']) == pdv.EXIT_ERROR
        assert pdv.main(['--result', str(tmp_path / 'missing')]) == pdv.EXIT_ERROR


# ----------------------------------------------------------------------------- class hierarchy (H07)
def test_hierarchy_weights_sum_to_one_and_levels():
    flat, hier = ch.hierarchy_weights(['rrBLUP', 'GBLUP', 'RF', 'ExtraTrees', 'GBDT', 'KNN'])
    assert flat.sum() == pytest.approx(1.0) and hier.sum() == pytest.approx(1.0)
    # additive, tree, local = 3 classes; rrBLUP shares 1/3 with GBLUP, KNN has the whole 1/3
    assert hier[0] == pytest.approx(1 / 3 / 2) and hier[5] == pytest.approx(1 / 3)
    assert hier[2] == pytest.approx(1 / 3 / 3)
    with pytest.raises(ValueError):
        ch.hierarchy_weights([])


def test_hierarchy_tuning_variants_share_their_base_weight():
    flat, hier = ch.hierarchy_weights(['RF__Grid', 'RF__Bayes', 'SVR'])
    assert hier[0] == pytest.approx(hier[1]) == pytest.approx(0.25)
    assert hier[2] == pytest.approx(0.5)


def test_hierarchy_equals_flat_when_every_class_has_one_model():
    flat, hier = ch.hierarchy_weights(['rrBLUP', 'BayesB', 'RF', 'KNN'])
    assert np.allclose(flat, hier)


def test_hierarchy_table_and_summary():
    test = _prediction_frame()
    members = ch.member_columns(list(test.columns))
    table = ch.hierarchy_table(test, members)
    assert set(table['method']) == {'flat_equal', 'class_hierarchy', 'ensemble'}
    summary = ch.summarise(table)
    row = summary[summary['method'] == 'class_hierarchy'].iloc[0]
    assert row['n_tasks'] == 4 and np.isfinite(row['d_pearson_vs_flat'])
    assert ch.hierarchy_table(test, members, min_lines=1000).empty


def test_hierarchy_cli_round_trip(tmp_path):
    _prediction_frame().to_csv(tmp_path / 'Prediction_result_test.csv', index=False)
    with contextlib.redirect_stdout(io.StringIO()):
        assert ch.main(['--result', str(tmp_path)]) == ch.EXIT_OK
    for name in ('Class_hierarchy_weights.csv', 'Class_hierarchy.csv', 'Class_hierarchy_summary.csv'):
        assert (tmp_path / name).is_file()


# ----------------------------------------------------------------------------- class coverage (D01 / R2)
def test_class_coverage_note():
    assert model_registry.class_coverage_note(['rrBLUP']) is None                # fewer than 2 models
    full = ['rrBLUP', 'BayesB', 'RF', 'SVR', 'MLP']
    assert model_registry.class_coverage_note(full) is None or isinstance(
        model_registry.class_coverage_note(full), str)
    only_trees = model_registry.class_coverage_note(['RF', 'ExtraTrees', 'GBDT'])
    assert isinstance(only_trees, str) and only_trees
    cov = model_registry.class_coverage(['RF', 'ExtraTrees'])
    assert cov['n_models'] == 2 and cov['missing']


# ----------------------------------------------------------------------------- seed bag (P4)
def test_resolve_seed_bag():
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        assert seed_bag.resolve_seed_bag(None) is None
        assert seed_bag.resolve_seed_bag(False) is None
        assert seed_bag.resolve_seed_bag(True) is None
        assert seed_bag.resolve_seed_bag(1) is None
        assert seed_bag.resolve_seed_bag('five') is None
        assert seed_bag.resolve_seed_bag(2.5) is None
        assert seed_bag.resolve_seed_bag(5) == {'n': 5, 'seed': 0}
        assert seed_bag.resolve_seed_bag(5.0) == {'n': 5, 'seed': 0}
        assert seed_bag.resolve_seed_bag({'n': 3, 'seed': 7}) == {'n': 3, 'seed': 7}
        assert seed_bag.resolve_seed_bag({'n': 99}) == {'n': seed_bag.MAX_BAG, 'seed': 0}
        assert seed_bag.resolve_seed_bag({'seed': 3}) is None
        assert seed_bag.resolve_seed_bag({'n': 3, 'bogus': 1}) == {'n': 3, 'seed': 0}
    assert 'WARNING' in quiet.getvalue()
    assert seed_bag.fingerprint_view(None) is None
    assert seed_bag.fingerprint_view({'n': 3, 'seed': 7}) == {'n': 3, 'seed': 7}


def test_member_seed_is_pure_and_distinct():
    seeds = {seed_bag.member_seed(0, t, k) for t in range(20) for k in range(10)}
    assert len(seeds) == 200
    assert seed_bag.member_seed(4, 9, 2) == seed_bag.member_seed(4, 9, 2)
    assert seed_bag.member_seed(0, 9, 2) != seed_bag.member_seed(1, 9, 2)


def test_isolated_rng_restores_global_state():
    import random
    random.seed(11)
    np.random.seed(11)
    before_py, before_np = random.random(), np.random.rand()
    random.seed(11)
    np.random.seed(11)
    with seed_bag.isolated_rng(5):
        a = (random.random(), np.random.rand())
    with seed_bag.isolated_rng(5):
        b = (random.random(), np.random.rand())
    assert a == b                                                  # same seed, same stream
    assert (random.random(), np.random.rand()) == (before_py, before_np)   # prior stream untouched


def test_isolated_rng_restores_on_error():
    np.random.seed(3)
    expected = np.random.rand()
    np.random.seed(3)
    with pytest.raises(RuntimeError):
        with seed_bag.isolated_rng(9):
            raise RuntimeError('boom')
    assert np.random.rand() == expected


def _fake_fit_factory(actual, calls):
    def fit():
        calls.append(1)
        noise = np.random.randn(actual.size)
        pred = actual + noise
        return (0.0, 0.0, 'effect', 'interaction', pred.tolist(), [], (actual + noise).tolist())
    return fit


def test_bag_call_means_and_recomputed_scores():
    actual = np.random.RandomState(0).randn(80)
    calls = []
    out = seed_bag.bag_call(_fake_fit_factory(actual, calls), 'MLP', 5, 0, 12, actual)
    assert len(calls) == 5 and len(out) == 7
    assert out[2] == 'effect' and out[3] == 'interaction'            # member 0's extras
    assert out[5] == []                                              # no validation set stays empty
    pred = np.asarray(out[4])
    assert out[1] == pytest.approx(float(np.mean((pred - actual) ** 2)))
    assert out[0] == pytest.approx(float(np.corrcoef(pred, actual)[0, 1]))
    single = seed_bag.bag_call(_fake_fit_factory(actual, []), 'MLP', 2, 0, 12, actual)
    assert out[1] < single[1] + 1.0
    assert isinstance(out[4], list)                                   # list in, list out


def test_bag_call_is_reproducible_and_reduces_variance():
    actual = np.random.RandomState(1).randn(300)
    one = seed_bag.bag_call(_fake_fit_factory(actual, []), 'MLP', 2, 0, 5, actual)
    two = seed_bag.bag_call(_fake_fit_factory(actual, []), 'MLP', 2, 0, 5, actual)
    assert one[4] == two[4]
    big = seed_bag.bag_call(_fake_fit_factory(actual, []), 'MLP', 20, 0, 5, actual)
    assert big[1] < one[1]                                            # mean MSE falls with the bag


def test_bag_call_layouts_and_errors():
    actual = np.arange(6.0)

    def gat():
        return (0.0, 0.0, 'e', actual.tolist(), [], actual.tolist(), 'attn')
    out = seed_bag.bag_call(gat, 'GAT_fully_connected', 2, 0, 1, actual)
    assert out[6] == 'attn' and out[0] == pytest.approx(1.0) and out[1] == pytest.approx(0.0)
    with pytest.raises(KeyError):
        seed_bag.bag_call(gat, 'RF', 2, 0, 1, actual)

    lengths = iter([6, 5])

    def ragged():
        n = next(lengths)
        return (0.0, 0.0, 'e', [0.0] * n, [], [0.0] * n, 'attn')
    with pytest.raises(ValueError):
        seed_bag.bag_call(ragged, 'GAT_fully_connected', 2, 0, 1, actual)


def test_bag_applies_only_to_the_listed_models():
    for m in ('MLP', 'GAT_infinitesimal', 'GAT_infinitesimal_node_level', 'GAT_fully_connected',
              'GAT_prior_knowledge'):
        assert seed_bag.bag_applies(m)
    for m in ('RF', 'rrBLUP', 'GAT_biological_prior_knowledge', 'MLP__Grid'):
        assert not seed_bag.bag_applies(m)


def test_isolated_rng_isolates_torch():
    torch = pytest.importorskip('torch')
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    with seed_bag.isolated_rng(1):
        a = torch.rand(3)
    with seed_bag.isolated_rng(1):
        b = torch.rand(3)
    assert torch.equal(a, b)
    assert torch.equal(torch.rand(3), expected)


# ----------------------------------------------------------------------------- off == identical (I11)
def test_engine_signature_defaults_are_off():
    """Read from the source (importing the engine needs torch_geometric, which a CI box may lack)."""
    import ast
    tree = ast.parse(open(os.path.join(ROOT, 'genomic_prediction.py'), encoding='utf-8').read())
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_GP_single')
    names = [a.arg for a in func.args.args]
    defaults = dict(zip(names[-len(func.args.defaults):], func.args.defaults))
    assert 'SEED_BAG' in defaults and isinstance(defaults['SEED_BAG'], ast.Constant)
    assert defaults['SEED_BAG'].value is None


def test_seed_bag_off_calls_fit_once_and_untouched():
    """The engine's wrapper with SEED_BAG off is `fit()` itself; this pins that the source does not
    route the off path through bag_call."""
    src = open(os.path.join(ROOT, 'genomic_prediction.py'), encoding='utf-8').read()
    marker = "if _seed_bag_cfg is None or not seed_bag_applies(model_name):\n            return fit()"
    assert marker in src
    assert "_fp_extra['SEED_BAG']" in src and "if _seed_bag_cfg is not None:\n        _fp_extra" in src


def test_every_entry_point_passes_the_key():
    for name in ('run_sequential.py', 'run_step1_batch.py', 'main_app.py'):
        text = open(os.path.join(ROOT, name), encoding='utf-8').read()
        assert "SEED_BAG=cfg.get('SEED_BAG')" in text, name
