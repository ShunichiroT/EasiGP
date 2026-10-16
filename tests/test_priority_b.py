"""Tests for the ver4-98 Priority-B additions (O02, V06, O03, O04, X04).

Run from the repository root:  python -m pytest tests/test_priority_b.py -q
Every fixture is generated here; no data file is needed."""
import gzip
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import candidate_reliability as cr  # noqa: E402
import member_importance as mi  # noqa: E402
import prediction_diagnostics as pdg  # noqa: E402
from model_registry import model_class  # noqa: E402
from models.relatedness import max_kinship, standardise_on  # noqa: E402


# ----------------------------------------------------------------------------- relatedness
def test_max_kinship_duplicate_and_mask_and_loo():
    rng = np.random.default_rng(0)
    G = rng.integers(0, 3, size=(40, 200)).astype(float)
    Zr, Zq = standardise_on(G, G[:5])
    k = max_kinship(Zq, Zr)
    self_rel = np.einsum('ij,ij->i', Zq, Zq[:5].copy()) / Zr.shape[1]
    assert np.allclose(k, np.maximum(self_rel, k))          # the duplicate is the best match
    assert np.allclose(k, (Zq * Zr[:5]).sum(1) / Zr.shape[1], atol=1e-12) or np.all(k >= self_rel - 1e-12)
    mask = np.zeros((5, 40), bool)
    mask[np.arange(5), np.arange(5)] = True                 # forbid the duplicate itself
    k_masked = max_kinship(Zq, Zr, mask=mask)
    sims = Zq @ Zr.T / Zr.shape[1]
    sims[np.arange(5), np.arange(5)] = -np.inf
    assert np.allclose(k_masked, sims.max(1))
    loo = max_kinship(Zr, Zr, exclude_self=True)
    full = Zr @ Zr.T / Zr.shape[1]
    np.fill_diagonal(full, -np.inf)
    assert np.allclose(loo, full.max(1))
    assert np.allclose(max_kinship(Zr, Zr, chunk=7, exclude_self=True), loo)   # chunk invariance


def test_max_kinship_errors_and_all_masked_is_nan():
    Z = np.random.default_rng(1).normal(size=(6, 10))
    with pytest.raises(ValueError):
        max_kinship(Z, Z[:, :5])
    with pytest.raises(ValueError):
        max_kinship(Z, Z, mask=np.zeros((2, 2), bool))
    with pytest.raises(ValueError):
        max_kinship(Z, Z[:4], exclude_self=True)
    bad = Z.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        max_kinship(bad, Z)
    out = max_kinship(Z[:2], Z, mask=np.ones((2, 6), bool))
    assert np.isnan(out).all()


def test_model_class_suffixes():
    assert model_class('rrBLUP') == 'additive'
    assert model_class('RF__Grid') == 'tree'
    assert model_class('GAT_biological_prior_knowledge_3') == 'bio_prior'
    assert model_class('GAT_fully_connected__Bayesian') == 'graph'
    assert model_class('SomethingNew') == 'other'


# ----------------------------------------------------------------------------- O02 / V06
def _pred_table(n=200, seed=0, option=None):
    rng = np.random.default_rng(seed)
    actual = rng.normal(size=n)
    df = pd.DataFrame({'id': [f'L{i}' for i in range(n)], 'population': 'P', 'ratio': '0.8',
                       'phenotype': 'y', 'sample': 0, 'actual': actual})
    df['m_double'] = 2 * actual + rng.normal(scale=0.05, size=n)   # over-dispersed: slope ~ 0.5
    df['m_good'] = actual + rng.normal(scale=0.3, size=n)
    df['m_const'] = 1.0
    if option is not None:
        df.insert(0, 'asreml_option', option)
    return df


def test_ac41_slope_of_doubled_predictions():
    per_task, _ = pdg.run(_pred_table(n=2000))
    row = per_task[per_task['model'] == 'm_double'].iloc[0]
    assert abs(row['slope'] - 0.5) < 0.02


def test_ac42_constant_column_nan_with_note_others_unaffected():
    notes = []
    per_task, _ = pdg.run(_pred_table(), notes=notes)
    const = per_task[per_task['model'] == 'm_const'].iloc[0]
    assert np.isnan(const['slope']) and np.isnan(const['pearson'])
    assert any("'m_const'" in n for n in notes)
    assert np.isfinite(per_task[per_task['model'] == 'm_good'].iloc[0]['slope'])


def test_ac43_asreml_option_is_a_key_and_pearson_matches():
    df = pd.concat([_pred_table(seed=1, option='BLUEs'), _pred_table(seed=2, option='BLUPs')])
    per_task, summary = pdg.run(df)
    assert set(per_task['asreml_option']) == {'BLUEs', 'BLUPs'}
    assert 'asreml_option' in summary.columns
    one = df[df['asreml_option'] == 'BLUEs']
    ref = np.corrcoef(one['actual'], one['m_good'])[0, 1]
    got = per_task[(per_task['asreml_option'] == 'BLUEs') & (per_task['model'] == 'm_good')]['pearson'].iloc[0]
    assert abs(got - ref) < 1e-10


def _grouped(seed=0, n_groups=4, per=40, within_noise=1.0):
    rng = np.random.default_rng(seed)
    rows, gmap = [], {}
    centres = np.linspace(-3, 3, n_groups)
    for g in range(n_groups):
        for i in range(per):
            actual = centres[g] + rng.normal()
            pred = centres[g] + rng.normal(scale=within_noise)   # independent of actual inside the group
            rows.append({'id': f'g{g}_{i}', 'population': 'P', 'ratio': '0.8', 'phenotype': 'y',
                         'sample': 0, 'actual': actual, 'm': pred})
            gmap[f'g{g}_{i}'] = f'G{g}'
    return pd.DataFrame(rows), pd.Series(gmap, name='group')


def test_ac51_within_near_zero_between_near_one():
    df, groups = _grouped(per=200)
    per_task, _ = pdg.run(df, groups=groups)
    row = per_task.iloc[0]
    assert abs(row['within_r_mean']) < 0.15
    assert row['between_r'] > 0.95
    assert row['n_groups'] == 4


def test_ac52_min_size_excludes_small_groups():
    df, groups = _grouped(per=40)
    small = df[df['id'].str.startswith('g3_')].index[:3]   # leave 37 lines in group 3 ...
    df = df.drop(df[df['id'].str.startswith('g3_')].index[3:])   # ... then cut it to 3 lines
    per_task, _ = pdg.run(df, groups=groups, min_group_size=5)
    assert per_task.iloc[0]['n_groups'] == 3
    per_task, _ = pdg.run(df, groups=groups, min_group_size=2)
    assert per_task.iloc[0]['n_groups'] == 4


def test_ac53_missing_ids_use_remaining_rows():
    df, groups = _grouped(per=60)
    keep = groups.index[: int(0.7 * len(groups))]
    per_task, _ = pdg.run(df, groups=groups.loc[keep])
    assert np.isfinite(per_task.iloc[0]['within_r_mean'])


def test_ac54_no_groups_no_group_columns():
    per_task, summary = pdg.run(_pred_table())
    assert not {'within_r_mean', 'between_r', 'n_groups'} & set(per_task.columns)
    assert not any(c.startswith('within_r') for c in summary.columns)


def test_ac55_duplicate_group_ids_exit_1(tmp_path):
    groups = tmp_path / 'g.csv'
    groups.write_text('ID,group\nA,1\nB,1\nA,2\n')
    with pytest.raises(ValueError, match='A'):
        pdg.read_groups(str(groups))
    result = tmp_path / 'run'
    result.mkdir()
    _pred_table().to_csv(result / 'Prediction_result_test.csv', index=False)
    assert pdg.main(['--result', str(result), '--groups', str(groups)]) == 1


def test_cli_end_to_end_gz_and_exit_codes(tmp_path):
    result = tmp_path / 'run'
    result.mkdir()
    assert pdg.main(['--result', str(result)]) == 3              # nothing there
    with gzip.open(result / 'Prediction_result_test.csv.gz', 'wt') as fh:
        _pred_table().to_csv(fh, index=False)
    assert pdg.main(['--result', str(result)]) == 0
    assert (result / 'Prediction_diagnostics.csv').exists()
    assert (result / 'Prediction_diagnostics_summary.csv').exists()
    assert pdg.main(['--result', str(tmp_path / 'missing')]) == 1


# ----------------------------------------------------------------------------- O04 / O03
def _population(n_a=900, n_b=60, m=400, seed=3):
    """Family A: lines drawn from a small set of founders (high relatedness). Family B: lines from
    unrelated founders."""
    rng = np.random.default_rng(seed)
    def family(n, n_founders):
        founders = rng.integers(0, 3, size=(n_founders, m)).astype(float)
        rows = founders[rng.integers(0, n_founders, size=n)].copy()
        flip = rng.random(rows.shape) < 0.03
        rows[flip] = rng.integers(0, 3, size=flip.sum())
        return rows
    GA, GB = family(n_a, 15), family(n_b, 15)
    G = np.vstack([GA, GB])
    ids = [f'A{i}' for i in range(n_a)] + [f'B{i}' for i in range(n_b)]
    beta = rng.normal(scale=0.2, size=m)
    g = (G - G.mean(0)) @ beta
    y = g + rng.normal(scale=g.std(), size=G.shape[0])
    return pd.DataFrame(G, index=ids), pd.Series(y, index=ids)


def test_domain_flag_finds_the_other_family():
    G, y = _population()
    ref = G.loc[[i for i in G.index if i.startswith('A')][:600]].to_numpy()
    own = G.loc[[i for i in G.index if i.startswith('A')][600:750]].to_numpy()
    other = G.loc[[i for i in G.index if i.startswith('B')]].to_numpy()
    res_own = cr.domain_flags(ref, own, 0.05)
    res_other = cr.domain_flags(ref, other, 0.05)
    assert res_own['out_of_domain'].mean() < 0.15
    assert res_other['out_of_domain'].mean() > 0.9
    assert res_other['k_percentile'].mean() < res_own['k_percentile'].mean()


def test_domain_flag_needs_enough_reference_lines():
    G = np.random.default_rng(0).integers(0, 3, size=(5, 20)).astype(float)
    with pytest.raises(ValueError):
        cr.domain_flags(G, G[:2])
    with pytest.raises(ValueError):
        cr.domain_flags(np.tile(G, (4, 1)), G[:2], quantile=1.5)


def test_conformal_marginal_coverage_and_finite_sample_quantile():
    rng = np.random.default_rng(5)
    n_val, n_te, covered = 200, 4000, []
    for rep in range(30):
        pv = rng.normal(size=n_val); av = pv + rng.normal(size=n_val)
        pt = rng.normal(size=n_te); at = pt + rng.normal(size=n_te)
        res = cr.conformal_intervals(pv, av, np.zeros(n_val), pt, np.zeros(n_te), alpha=0.1, n_bins=1)
        covered.append(np.mean((at >= res['lower']) & (at <= res['upper'])))
    assert abs(np.mean(covered) - 0.9) < 0.02
    assert cr._conformal_quantile(np.arange(5.0), 0.1) == float('inf')     # n too small for 90 %
    with pytest.raises(ValueError):
        cr.conformal_intervals([1.0], [np.nan], [0.0], [1.0], [0.0])
    with pytest.raises(ValueError):
        cr.conformal_intervals([1.0], [1.0], [0.0], [1.0], [0.0], alpha=1.5)


def test_conformal_strata_widen_for_distant_lines():
    rng = np.random.default_rng(6)
    n = 800
    k = rng.uniform(0, 1, size=n)
    pred = np.zeros(n)
    actual = rng.normal(scale=0.2 + 2.0 * (1 - k))          # distant (low k) lines are noisier
    res = cr.conformal_intervals(pred, actual, k, np.zeros(2), np.array([0.05, 0.95]),
                                 alpha=0.1, n_bins=2, min_cal=20)
    assert list(res['stratum']) == [0, 1]
    assert res['half_width'][0] > 2 * res['half_width'][1]
    pooled = cr.conformal_intervals(pred, actual, k, np.zeros(2), np.array([0.05, 0.95]),
                                    alpha=0.1, n_bins=1)
    assert pooled['half_width'][0] == pooled['half_width'][1]
    sparse = cr.conformal_intervals(pred[:30], actual[:30], k[:30], np.zeros(2),
                                    np.array([0.05, 0.95]), alpha=0.1, n_bins=2, min_cal=20)
    assert list(sparse['stratum']) == [-1, -1]               # strata too small: pooled width


def _run_folder(tmp_path, G, y, seed=0, n_te_b=40):
    """A result folder as EasiGP writes it: ridge predictions, train 500 / valid 200 / test 200 +
    n_te_b lines of family B."""
    rng = np.random.default_rng(seed)
    a_ids = [i for i in G.index if i.startswith('A')]
    b_ids = [i for i in G.index if i.startswith('B')][:n_te_b]
    rng.shuffle(a_ids)
    tr, va, te = a_ids[:500], a_ids[500:700], a_ids[700:900] + b_ids
    Xtr = G.loc[tr].to_numpy(); mu = Xtr.mean(0)
    lam = 50.0
    A = (Xtr - mu).T @ (Xtr - mu) + lam * np.eye(Xtr.shape[1])
    b = np.linalg.solve(A, (Xtr - mu).T @ (y.loc[tr].to_numpy() - y.loc[tr].mean()))
    def predict(ids):
        return (G.loc[ids].to_numpy() - mu) @ b + y.loc[tr].mean()
    def table(ids):
        p = predict(ids)
        return pd.DataFrame({'id': ids, 'population': 'P', 'ratio': '(0.6, 0.2, 0.2)',
                             'phenotype': 'y', 'sample': 0, 'actual': y.loc[ids].to_numpy(),
                             'ensemble': p, 'Super learner__analytic': p, 'RF': p * 0.5})
    run = tmp_path / 'run'; run.mkdir()
    table(tr).to_csv(run / 'Prediction_result_train.csv', index=False)
    table(va).to_csv(run / 'Prediction_result_valid.csv', index=False)
    table(te).to_csv(run / 'Prediction_result_test.csv', index=False)
    geno = G.copy(); geno.insert(0, 'population', 'P'); geno.insert(0, 'ID', geno.index)
    geno.to_csv(tmp_path / 'geno.csv', index=False)
    return run, tmp_path / 'geno.csv', len(te) - n_te_b


def test_cli_candidate_reliability_end_to_end(tmp_path):
    G, y = _population()
    run, geno, n_own = _run_folder(tmp_path, G, y)
    code = cr.main(['--result', str(run), '--genotypes', str(geno), '--alpha', '0.1'])
    assert code == 0
    dom = pd.read_csv(run / 'Candidate_domain.csv')
    itv = pd.read_csv(run / 'Candidate_interval.csv')
    assert dom.shape[0] == n_own + 40
    flagged_b = dom[dom['id'].str.startswith('B')]['out_of_domain'].mean()
    flagged_a = dom[dom['id'].str.startswith('A')]['out_of_domain'].mean()
    assert flagged_b > 0.9 and flagged_a < 0.15
    assert set(itv['model']) == {'ensemble', 'Super learner__analytic'}   # default model selection; RF excluded
    own = itv[itv['id'].str.startswith('A') & (itv['model'] == 'ensemble')]
    assert 0.8 <= own['covered'].mean() <= 0.97                           # near the nominal 0.9
    assert (itv['upper'] > itv['lower']).all()
    summary = pd.read_csv(run / 'Candidate_reliability_summary.csv')
    assert {'stratum', 'out_of_domain'} == set(summary['by'])
    flag = summary[summary['by'] == 'out_of_domain']
    far = flag[flag['value'].astype(str) == 'True']      # CSV round trip: the value column is text
    near = flag[flag['value'].astype(str) == 'False']
    assert far['mean_abs_error'].mean() > near['mean_abs_error'].mean()    # the flag points at higher error
    assert far['coverage'].mean() < near['coverage'].mean()    # and the calibration set (all in-domain) under-covers it


def test_candidate_reliability_without_validation_gives_domain_only(tmp_path):
    G, y = _population()
    run, geno, _ = _run_folder(tmp_path, G, y)
    os.remove(run / 'Prediction_result_valid.csv')
    assert cr.main(['--result', str(run), '--genotypes', str(geno)]) == 0
    assert pd.read_csv(run / 'Candidate_interval.csv').shape[0] == 0
    assert pd.read_csv(run / 'Candidate_domain.csv').shape[0] > 0


def test_candidate_reliability_input_errors(tmp_path):
    G, y = _population()
    run, geno, _ = _run_folder(tmp_path, G, y)
    assert cr.main(['--result', str(run), '--genotypes', str(tmp_path / 'nope.csv')]) == 1
    bad = tmp_path / 'bad.csv'
    bad.write_text('x,y\n1,2\n')
    assert cr.main(['--result', str(run), '--genotypes', str(bad)]) == 1
    empty = tmp_path / 'empty'; empty.mkdir()
    assert cr.main(['--result', str(empty), '--genotypes', str(geno)]) == 3


# ----------------------------------------------------------------------------- X04
def _weights():
    return pd.DataFrame({
        'population': 'P', 'phenotype': 'y', 'model': 'Super learner__analytic', 'ratio': '0.8',
        'sample': [0, 1, 2, 3],
        'rrBLUP': [0.0, 0.5, 0.0, 0.0], 'GBLUP': [0.9, 0.0, 0.8, 0.1],
        'RF__Grid': [0.05, 0.25, 0.1, 0.45], 'XGBoost__Grid': [0.05, 0.25, 0.1, 0.45]})


def test_class_shares_sum_to_one_and_concentration():
    shares = mi.class_shares(_weights())
    per_row = shares.groupby('sample')['share'].sum()
    assert np.allclose(per_row, 1.0)
    add0 = shares[(shares['sample'] == 0) & (shares['class'] == 'additive')].iloc[0]
    assert add0['share'] == pytest.approx(0.9) and add0['top_member'] == 'GBLUP'
    assert add0['effective_members'] == pytest.approx(1.0)
    add1 = shares[(shares['sample'] == 1) & (shares['class'] == 'additive')].iloc[0]
    assert add1['top_within_class'] == pytest.approx(1.0)      # only rrBLUP has weight there
    tree = shares[(shares['sample'] == 3) & (shares['class'] == 'tree')].iloc[0]
    assert tree['effective_members'] == pytest.approx(2.0) and tree['n_members'] == 2


def test_class_shares_renormalise_and_skip_zero_rows():
    w = _weights()
    w.loc[0, ['rrBLUP', 'GBLUP', 'RF__Grid', 'XGBoost__Grid']] = [0.0, 1.8, 0.1, 0.1]   # sums to 2
    w.loc[3, ['rrBLUP', 'GBLUP', 'RF__Grid', 'XGBoost__Grid']] = 0.0
    shares = mi.class_shares(w)
    assert 3 not in set(shares['sample'])
    assert shares[shares['sample'] == 0]['share'].sum() == pytest.approx(1.0)
    assert shares[shares['sample'] == 0]['weight_sum'].iloc[0] == pytest.approx(2.0)


def test_summary_top_member_consistency_and_cli(tmp_path):
    w = _weights()
    shares = mi.class_shares(w)
    summary = mi.summarise(shares)
    add = summary[summary['class'] == 'additive'].iloc[0]
    assert add['n_tasks'] == 4 and add['top_member_modal'] == 'GBLUP'
    assert add['top_member_consistency'] == pytest.approx(0.75)   # rrBLUP tops the class in 1 of 4
    run = tmp_path / 'run'; run.mkdir()
    w.to_csv(run / 'Weight.csv', index=False)
    assert mi.main(['--result', str(run)]) == 0
    assert (run / 'Member_class_shares_summary.csv').exists()
    assert mi.main(['--result', str(tmp_path)]) == 3
    with pytest.raises(ValueError):
        mi.class_shares(pd.DataFrame({'population': ['P'], 'model': ['x']}))


def test_zero_weight_class_counts_as_zero_not_a_gap():
    w = _weights()
    w['KNN'] = [0.0, 0.0, 0.0, 0.0]
    w['MLP'] = [0.0, 0.0, 0.0, 0.0]
    w.loc[3, 'MLP'] = 0.2
    shares = mi.class_shares(w)
    neural = shares[shares['class'] == 'neural']
    assert len(neural) == 4                              # one row per task, zeros included
    summary = mi.summarise(shares)
    row = summary[summary['class'] == 'neural'].iloc[0]
    assert row['n_tasks'] == 4 and row['share_mean'] == pytest.approx(0.2 / 1.2 / 4)
    local = summary[summary['class'] == 'local'].iloc[0]
    assert local['share_mean'] == 0.0 and local['top_member_modal'] == ''


# ----------------------------------------------------------------------------- O03 flagged lines (ver4-101)
import itertools  # noqa: E402
_COUNTER = itertools.count()


def _reliab(tmp_path, seed, inflate=1.0, alpha=0.1):
    G, y = _population(seed=seed)
    sub = tmp_path / f's{seed}_{next(_COUNTER)}'
    sub.mkdir()
    run, geno, _ = _run_folder(sub, G, y, seed=seed)
    read = lambda n: pd.read_csv(run / f'Prediction_result_{n}.csv')
    notes = []
    dom, itv = cr.reliability(read('test'), read('train'), read('valid'), cr.load_genotypes(str(geno)),
                              None, alpha, 0.05, 2, 20, notes, inflate)
    return dom, itv


def test_inflate_one_is_the_old_interval_and_f_scales_only_flagged_lines(tmp_path):
    _, base = _reliab(tmp_path, 3)
    _, wide = _reliab(tmp_path, 3, inflate=2.0)
    flagged = base['out_of_domain'].to_numpy(dtype=bool)
    assert flagged.any() and (~flagged).any()
    half0 = ((base['upper'] - base['lower']) / 2).to_numpy()
    half1 = ((wide['upper'] - wide['lower']) / 2).to_numpy()
    assert np.allclose(half1[~flagged], half0[~flagged])
    assert np.allclose(half1[flagged], 2 * half0[flagged])
    assert (base['inflation'] == 1.0).all() and set(wide.loc[flagged, 'inflation']) == {2.0}
    assert np.allclose(base['prediction'], wide['prediction'])
    with pytest.raises(ValueError):
        cr.reliability(pd.DataFrame({'id': []}), None, None, pd.DataFrame(), None, 0.1, 0.05, 2, 20,
                       [], 0.5)


def test_needed_factor_reaches_nominal_coverage_on_flagged_lines(tmp_path):
    _, base = _reliab(tmp_path, 3)
    table = cr.inflation_table(base, 0.1, min_out=10)
    row = table[table['model'] == 'ensemble'].iloc[0]
    assert row['coverage_out'] < row['coverage_in'] and row['needed_factor'] > 1.0
    _, fixed = _reliab(tmp_path, 3, inflate=float(row['needed_factor']))
    after = cr.inflation_table(fixed, 0.1, min_out=10)
    row2 = after[after['model'] == 'ensemble'].iloc[0]
    assert row2['coverage_out'] >= 0.9 - 1e-9               # in-sample by construction
    assert row2['coverage_in'] == pytest.approx(row['coverage_in'])      # in-domain lines unchanged
    assert row2['needed_factor'] == pytest.approx(row['needed_factor'])  # applied factor is divided out


def test_factor_measured_on_one_run_helps_a_fresh_one(tmp_path):
    gain = []
    for bench, deploy in ((4, 5), (6, 7), (8, 9)):
        _, b = _reliab(tmp_path, bench)
        f = cr.inflation_table(b, 0.1, 10).query("model == 'ensemble'").iloc[0]['needed_factor']
        assert np.isfinite(f)
        _, d0 = _reliab(tmp_path, deploy)
        _, d1 = _reliab(tmp_path, deploy, inflate=float(f))
        c0 = cr.inflation_table(d0, 0.1, 10).query("model == 'ensemble'").iloc[0]['coverage_out']
        c1 = cr.inflation_table(d1, 0.1, 10).query("model == 'ensemble'").iloc[0]['coverage_out']
        assert c1 >= c0                                       # a wider interval never covers less
        gain.append(c1 - c0)
    assert np.mean(gain) > 0.05                               # and out-of-sample it recovers coverage


def test_cli_writes_inflation_table_and_rejects_factor_below_one(tmp_path):
    G, y = _population()
    run, geno, _ = _run_folder(tmp_path, G, y)
    assert cr.main(['--result', str(run), '--genotypes', str(geno), '--ood-inflate', '0.5']) == 1
    assert cr.main(['--result', str(run), '--genotypes', str(geno), '--ood-inflate', '1.5']) == 0
    table = pd.read_csv(run / 'Candidate_inflation.csv')
    assert {'n_in', 'n_out', 'coverage_in', 'coverage_out', 'needed_factor'} <= set(table.columns)
    itv = pd.read_csv(run / 'Candidate_interval.csv')
    assert set(itv.loc[itv['out_of_domain'], 'inflation']) == {1.5}
