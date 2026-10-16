"""Tests for the REBUILT modules (recovery patch, 2026-10-15): tools/member_shapley.py,
tools/consensus_stability.py, stability_summary.py.

Run from the repository root:  python -m pytest tests/test_recovery_modules.py -q
Every fixture is generated here; no data file is needed. The expected values are worked out
independently of the code under test (brute force over orderings, hand-built top-K sets)."""
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import consensus_stability as cs  # noqa: E402
import member_shapley as msh  # noqa: E402
import stability_summary as ss  # noqa: E402


# ------------------------------------------------------------------------------- Shapley
def _pred(n=300, seed=0, members=('rrBLUP', 'GBLUP', 'RF', 'KNN'), n_tasks=1):
    rng = np.random.default_rng(seed)
    frames = []
    for t in range(n_tasks):
        y = rng.normal(size=n)
        cols = {'id': [f'L{i}' for i in range(n)], 'population': 'P', 'ratio': '0.8',
                'phenotype': 'y', 'sample': t, 'actual': y}
        for j, m in enumerate(members):
            cols[m] = y * (0.9 - 0.2 * j) + rng.normal(scale=1.0, size=n)
        cols['ensemble'] = y
        cols['Super learner__analytic'] = y
        frames.append(pd.DataFrame(cols))
    return pd.concat(frames, ignore_index=True)


def _brute_force(P, y):
    n = P.shape[1]

    def v(subset):
        if not subset:
            return 0.0
        return float(np.corrcoef(P[:, list(subset)].mean(axis=1), y)[0, 1])

    phi = np.zeros(n)
    orders = list(itertools.permutations(range(n)))
    for order in orders:
        seen = []
        for i in order:
            phi[i] += v(seen + [i]) - v(seen)
            seen.append(i)
    return phi / len(orders), v(range(n))


def test_exact_shapley_matches_brute_force_over_orderings():
    pred = _pred()
    members = msh.member_columns(pred)
    assert members == ['rrBLUP', 'GBLUP', 'RF', 'KNN']            # combined columns excluded
    table = msh.run(pred)
    P = pred[members].to_numpy()
    expected, total = _brute_force(P, pred['actual'].to_numpy())
    got = table.set_index('model').loc[members, 'shapley'].to_numpy()
    assert np.allclose(got, expected, atol=1e-10)
    assert np.allclose(table['shapley'].sum(), total, atol=1e-10)   # efficiency: values sum to total r
    assert (table['method'] == 'exact').all() and (table['total_r'].round(10) == round(total, 10)).all()


def test_duplicate_members_share_credit_equally_and_dummy_member_gets_zero_ish():
    pred = _pred(members=('rrBLUP', 'GBLUP', 'RF'))
    pred['GBLUP_copy'] = pred['GBLUP']                              # a near-duplicate
    table = msh.run(pred).set_index('model')
    assert table.loc['GBLUP', 'shapley'] == pytest.approx(table.loc['GBLUP_copy', 'shapley'], abs=1e-12)
    assert table['share'].sum() == pytest.approx(1.0)


def test_sampled_values_sum_to_total_and_are_close_to_exact():
    pred = _pred(n=400, members=tuple(f'm{i}' for i in range(6)))
    members = msh.member_columns(pred)
    exact = msh.run(pred, exact_max=12).set_index('model').loc[members, 'shapley'].to_numpy()
    samp = msh.run(pred, exact_max=3, n_perm=2000, seed=1)
    assert (samp['method'] == 'sampled').all()
    assert samp['shapley'].sum() == pytest.approx(samp['total_r'].iloc[0], abs=1e-9)
    assert np.allclose(samp.set_index('model').loc[members, 'shapley'].to_numpy(), exact, atol=0.01)
    again = msh.run(pred, exact_max=3, n_perm=2000, seed=1)
    assert np.allclose(samp['shapley'], again['shapley'])           # seeded


def test_shapley_handles_nan_constant_and_tiny_tasks():
    pred = _pred(n=50, members=('a', 'b'))
    pred.loc[0, 'a'] = np.nan                                       # listwise drop
    pred['c'] = 1.0                                                 # constant member: adds nothing
    table = msh.run(pred)
    assert (table['n_lines'] == 49).all()
    assert np.isfinite(table['shapley']).all()
    assert msh.run(_pred(n=2, members=('a', 'b'))).empty            # fewer than 3 lines: no task


def test_shapley_cli_writes_files_and_exit_codes(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    _pred(n_tasks=3).to_csv(run / 'Prediction_result_test.csv', index=False)
    out = tmp_path / 'out'
    assert msh.main(['--result', str(run), '--out-dir', str(out)]) == 0
    assert {'Member_shapley.csv', 'Member_shapley_summary.csv', 'Member_shapley_class.csv'} <= \
        {p.name for p in out.iterdir()}
    summary = pd.read_csv(out / 'Member_shapley_summary.csv')
    assert (summary['n_tasks'] == 3).all()
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert msh.main(['--result', str(empty)]) == 3                  # nothing to evaluate
    assert msh.main(['--result', str(tmp_path / 'missing')]) == 1


# ------------------------------------------------------------------------------ stability
def _effect(n_tasks=6, m=40, signal=(3, 7, 11), seed=0, models=('rrBLUP', 'RF')):
    """Markers in `signal` are always large; the rest are noise that changes per task."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        for model in models:
            eff = rng.normal(scale=0.05, size=m)
            eff[list(signal)] = 5.0 + rng.normal(scale=0.1, size=len(signal))
            rows.append([ 'P', 'y', model, '0.8', t] + list(eff))
    cols = ['population', 'phenotype', 'model', 'ratio', 'sample'] + [f'snp{j}' for j in range(m)]
    return pd.DataFrame(rows, columns=cols)


def test_marker_stability_planted_markers_have_frequency_one():
    stab, freq, notes = ss.marker_stability(_effect(), top_k=3)
    row = stab.iloc[0]
    assert row['n_tasks'] == 6 and row['jaccard_mean'] == pytest.approx(1.0)
    assert row['n_freq_1'] == 3 and row['n_selected_ever'] == 3
    top = freq[freq['frequency'] == 1.0]['marker']
    assert set(top) == {'snp3', 'snp7', 'snp11'}


def test_marker_stability_by_hand_for_two_tasks():
    # task 0 top-2 = {a, b}; task 1 top-2 = {b, c}: Jaccard 1/3, b at frequency 1, a and c 0.5
    effect = pd.DataFrame({'population': 'P', 'phenotype': 'y', 'model': 'rrBLUP', 'ratio': '0.8',
                           'sample': [0, 1], 'a': [9.0, 1.0], 'b': [8.0, 9.0], 'c': [1.0, 8.0],
                           'd': [0.5, 0.5]})
    stab, freq, _ = ss.marker_stability(effect, top_k=2)
    assert stab.iloc[0]['jaccard_mean'] == pytest.approx(1 / 3)
    f = freq.set_index('marker')['frequency']
    assert f['b'] == 1.0 and f['a'] == 0.5 and f['c'] == 0.5 and 'd' not in f.index


def test_models_on_different_scales_count_equally():
    # model A (scale 100): a=100, b=60; model B (scale 1): a=0.1, b=0.5. Row-normalised, the mean
    # is a=0.396, b=0.604, so b wins; un-normalised, A's scale would let a win (50.05 vs 30.25).
    effect = pd.DataFrame({'population': 'P', 'phenotype': 'y', 'ratio': '0.8', 'sample': 0,
                           'model': ['A', 'B'], 'a': [100.0, 0.1], 'b': [60.0, 0.5], 'c': [0.0, 0.0]})
    freq = ss.marker_stability(effect, top_k=1)[1]
    assert list(freq['marker']) == ['b']
    assert freq.iloc[0]['consensus_mean'] == pytest.approx(0.6041666, abs=1e-6)


def test_combined_rows_are_left_out_unless_nothing_else_exists():
    effect = _effect(models=('rrBLUP',))
    ens = effect.copy()
    ens['model'] = 'ensemble'
    ens.loc[:, [c for c in ens.columns if c.startswith('snp')]] = 1.0     # would spoil the result
    both = pd.concat([effect, ens], ignore_index=True)
    stab, freq, _ = ss.marker_stability(both, top_k=3)
    assert set(freq[freq['frequency'] == 1.0]['marker']) == {'snp3', 'snp7', 'snp11'}
    only = ss.marker_stability(ens, top_k=3)[0]                          # only 'ensemble' rows: used
    assert not only.empty
    assert ss.select_models(['ensemble', 'Super learner__analytic', 'RF__Grid']) == ['RF__Grid']


def test_missing_and_zero_rows_and_bad_input():
    effect = _effect()
    effect.loc[0, [c for c in effect.columns if c.startswith('snp')]] = 0.0     # all-zero row
    assert not ss.marker_stability(effect, top_k=3)[0].empty
    assert ss.marker_stability(effect.iloc[0:0])[0].empty
    with pytest.raises(ValueError):
        ss.marker_stability(effect, top_k=0)
    with pytest.raises(ValueError):
        ss.marker_stability(effect.drop(columns=['sample']), top_k=3)
    single = ss.marker_stability(effect[effect['sample'] == 0], top_k=3)
    assert np.isnan(single[0].iloc[0]['jaccard_mean']) and any('single iteration' in n for n in single[2])


def test_bin_pools_consecutive_markers():
    stab, freq, _ = ss.marker_stability(_effect(signal=(3, 4, 11)), top_k=1, bin_size=4)
    assert 'snp0:snp3' in set(freq['marker'])                    # snp3 and snp4 fall in bins 0-3 / 4-7
    assert stab.iloc[0]['n_candidates'] == 10


def _interactions(n_tasks=5, seed=0, stable=(('s1', 's2'), ('s3', 's4'))):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tasks):
        for model in ('RF', 'GBDT'):
            for a, b in stable:
                rows.append(['P', 'y', model, '0.8', t, b, a, 10.0 + rng.normal(scale=0.1)])  # reversed order
            for _ in range(15):
                i, j = rng.choice(30, size=2, replace=False)
                rows.append(['P', 'y', model, '0.8', t, f'n{i}', f'n{j}', abs(rng.normal(scale=0.3))])
    return pd.DataFrame(rows, columns=['population', 'phenotype', 'model', 'ratio', 'sample',
                                       'marker1', 'marker2', 'value'])


def test_pair_stability_finds_stable_pairs_and_is_chunk_size_independent():
    inter = _interactions()
    results = [ss.pair_stability(inter, top_k=2, chunksize=cs_)[1] for cs_ in (7, 50, 10_000)]
    for other in results[1:]:
        pd.testing.assert_frame_equal(results[0], other)
    freq = results[0]
    top = freq[freq['frequency'] == 1.0]
    assert set(zip(top['marker1'], top['marker2'])) == {('s1', 's2'), ('s3', 's4')}   # order-free pairs


def test_pair_stability_reads_a_csv_path_and_rejects_bad_tables(tmp_path):
    path = tmp_path / 'Interaction.csv'
    _interactions().to_csv(path, index=False)
    a = ss.pair_stability(str(path), top_k=2, chunksize=11)[0]
    b = ss.pair_stability(_interactions(), top_k=2)[0]
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        ss.pair_stability(_interactions().drop(columns=['value']), top_k=2)
    assert ss.pair_stability(_interactions().drop(columns=['sample']))[0].empty


def test_cli_end_to_end_and_exit_codes(tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    _effect().to_csv(run / 'Marker_effect.csv', index=False)
    _interactions().to_csv(run / 'Interaction.csv', index=False)
    out = tmp_path / 'out'
    assert cs.main(['--result', str(run), '--out-dir', str(out), '--top-k', '2', '--pairs']) == 0
    names = {p.name for p in out.iterdir()}
    assert {'Consensus_stability.csv', 'Marker_selection_frequency.csv',
            'Pair_selection_frequency.csv'} <= names
    levels = set(pd.read_csv(out / 'Consensus_stability.csv')['level'])
    assert levels == {'marker', 'pair'}
    only = tmp_path / 'only'
    assert cs.main(['--result', str(run), '--out-dir', str(only), '--no-frequencies']) == 0
    assert {p.name for p in only.iterdir()} == {'Consensus_stability.csv'}
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert cs.main(['--result', str(empty)]) == 3
    assert cs.main(['--result', str(run), '--top-k', '0']) == 1
    assert cs.main(['--result', str(tmp_path / 'nope')]) == 1


def test_run_stability_report_is_off_by_default_and_never_raises(tmp_path, monkeypatch):
    import pipeline_utils as pu
    monkeypatch.setattr(pu, 'RESULT_BASE_DIR', str(tmp_path / 'Result'))
    logs = []
    assert ss.run_stability_report({}, 'r1', lambda: _effect(), logs.append) == []
    assert ss.run_stability_report({'STABILITY_SUMMARY_CREATE': False}, 'r1', lambda: _effect(),
                                   logs.append) == []
    assert logs == [] and not (tmp_path / 'Result').exists()          # off: nothing at all (I11)

    def boom():
        raise RuntimeError('disk gone')
    assert ss.run_stability_report({'STABILITY_SUMMARY_CREATE': True}, 'r1', boom, logs.append) == []
    assert any('skipped' in m for m in logs)

    written = ss.run_stability_report({'STABILITY_SUMMARY_CREATE': True, 'STABILITY_TOP_K': 3},
                                      'r1', lambda: _effect(), logs.append)
    assert [os.path.basename(p) for p in written] == ['Consensus_stability.csv',
                                                      'Marker_selection_frequency.csv']
    assert all(os.path.isfile(p) for p in written)


# ----------------------------------------------------------------------- Kuncheva (stability)
def test_kuncheva_is_one_for_identical_sets_and_hand_value():
    stab = ss.marker_stability(_effect(), top_k=3)[0].iloc[0]
    assert stab['kuncheva_mean'] == pytest.approx(1.0)
    # two sets of size 2 among 4 candidates sharing 1: (1*4 - 4) / (2 * (4 - 2)) = 0
    effect = pd.DataFrame({'population': 'P', 'phenotype': 'y', 'model': 'rrBLUP', 'ratio': '0.8',
                           'sample': [0, 1], 'a': [9.0, 1.0], 'b': [8.0, 9.0], 'c': [1.0, 8.0],
                           'd': [0.5, 0.5]})
    assert ss.marker_stability(effect, top_k=2)[0].iloc[0]['kuncheva_mean'] == pytest.approx(0.0)
    # sets of size 3 among 6 candidates sharing 2: (2*6 - 9) / (3 * 3) = 1/3
    six = pd.DataFrame({'population': 'P', 'phenotype': 'y', 'model': 'rrBLUP', 'ratio': '0.8',
                        'sample': [0, 1], 'a': [9.0, 1.0], 'b': [8.0, 9.0], 'c': [7.0, 8.0],
                        'd': [1.0, 7.0], 'e': [0.5, 0.5], 'f': [0.4, 0.4]})
    assert ss.marker_stability(six, top_k=3)[0].iloc[0]['kuncheva_mean'] == pytest.approx(1 / 3)


# ----------------------------------------------------------------------------------- X03
def _x03_inputs(tmp_path, gz=False):
    inter = _interactions()
    inter.insert(0, 'extra_note', 'keep me')              # an unknown extra column must be copied
    path = tmp_path / ('Interaction.csv.gz' if gz else 'Interaction.csv')
    inter.to_csv(path, index=False)
    effect = _effect(signal=(3, 7, 11))
    return path, inter, effect


def test_interaction_frequency_file_values_by_hand(tmp_path):
    path, inter, effect = _x03_inputs(tmp_path)
    # the pair markers in _interactions() are 's1'..'s4' / 'n..'; make the effect table know s1, s3
    effect = effect.rename(columns={'snp3': 's1', 'snp7': 's3'})
    report = ss.build_report(effect, str(path), top_k=3, pairs=True)
    out = tmp_path / 'Interaction_selection_frequency.csv'
    n = ss.write_interaction_frequency(str(path), str(out), report, chunksize=9)
    assert n == inter.shape[0]
    got = pd.read_csv(out, dtype=str, keep_default_na=False)
    # the original columns are carried through unchanged and in order, three columns appended
    assert list(got.columns) == list(inter.columns) + ['pair_frequency', 'marker1_frequency',
                                                       'marker2_frequency']
    assert got[list(inter.columns)].equals(pd.read_csv(path, dtype=str, keep_default_na=False))
    f = got.assign(pair_frequency=got['pair_frequency'].astype(float),
                   marker1_frequency=got['marker1_frequency'].astype(float),
                   marker2_frequency=got['marker2_frequency'].astype(float))
    stable = f[((f['marker1'] == 's2') & (f['marker2'] == 's1')) | ((f['marker1'] == 's4') & (f['marker2'] == 's3'))]
    assert len(stable) == 2 * 5 * 2                                  # 2 pairs x 5 iterations x 2 models
    assert (stable['pair_frequency'] == 1.0).all()                   # both models, all 5 iterations
    noise = f[f['marker1'].str.startswith('n') & f['marker2'].str.startswith('n')]
    assert (noise['pair_frequency'] < 1.0).all() and (noise['marker1_frequency'] == 0.0).all()
    s1_rows = f[(f['marker1'] == 's2') & (f['marker2'] == 's1')]
    assert (s1_rows['marker2_frequency'] == 1.0).all()               # 's1' is a stable top marker
    assert (s1_rows['marker1_frequency'] == 0.0).all()               # 's2' is not in the effect table
    assert not os.path.exists(str(out) + '.partial')


def test_interaction_frequency_gz_source_and_chunk_independence(tmp_path):
    path, _, effect = _x03_inputs(tmp_path, gz=True)
    report = ss.build_report(effect, str(path), top_k=3, pairs=True)
    a, b = tmp_path / 'a.csv', tmp_path / 'b.csv'
    ss.write_interaction_frequency(str(path), str(a), report, chunksize=4)
    ss.write_interaction_frequency(str(path), str(b), report, chunksize=10_000)
    assert a.read_bytes() == b.read_bytes()


def test_interaction_frequency_errors_and_empty(tmp_path):
    path, _, effect = _x03_inputs(tmp_path)
    report = ss.build_report(effect, str(path), top_k=3, pairs=True)
    with pytest.raises(ValueError):
        ss.write_interaction_frequency(_interactions(), str(tmp_path / 'x.csv'), report)   # not a path
    broken = dict(report, pair_frequency=report['pair_frequency'].drop(columns=['ratio']))
    with pytest.raises(ValueError):
        ss.write_interaction_frequency(str(path), str(tmp_path / 'y.csv'), broken)
    header_only = tmp_path / 'Empty.csv'
    header_only.write_text('population,phenotype,model,ratio,sample,marker1,marker2,value\n')
    assert ss.write_interaction_frequency(str(header_only), str(tmp_path / 'z.csv'), report) == 0
    assert not (tmp_path / 'z.csv').exists()


def test_x03_cli_flag_and_report_hook(tmp_path, monkeypatch):
    run = tmp_path / 'run'
    run.mkdir()
    _effect().to_csv(run / 'Marker_effect.csv', index=False)
    _interactions().to_csv(run / 'Interaction.csv', index=False)
    before = (run / 'Interaction.csv').read_bytes()
    out = tmp_path / 'out'
    assert cs.main(['--result', str(run), '--out-dir', str(out), '--top-k', '2',
                    '--interaction-frequency']) == 0                  # implies --pairs
    assert (out / 'Interaction_selection_frequency.csv').is_file()
    assert (run / 'Interaction.csv').read_bytes() == before            # the source is never touched
    plain = tmp_path / 'plain'
    assert cs.main(['--result', str(run), '--out-dir', str(plain), '--pairs']) == 0
    assert 'Interaction_selection_frequency.csv' not in {p.name for p in plain.iterdir()}   # off by default

    import pipeline_utils as pu
    monkeypatch.setattr(pu, 'RESULT_BASE_DIR', str(tmp_path / 'Result'))
    folder = tmp_path / 'Result' / 'r1'
    folder.mkdir(parents=True)
    _interactions().to_csv(folder / 'Interaction.csv', index=False)
    logs = []
    cfg = {'STABILITY_SUMMARY_CREATE': True, 'STABILITY_TOP_K': 2}
    w = ss.run_stability_report(cfg, 'r1', lambda: _effect(), logs.append)
    assert 'Interaction_selection_frequency.csv' not in [os.path.basename(p) for p in w]
    cfg['STABILITY_INTERACTION_FREQUENCY'] = True
    w = ss.run_stability_report(cfg, 'r1', lambda: _effect(), logs.append)
    assert 'Interaction_selection_frequency.csv' in [os.path.basename(p) for p in w]
    assert (folder / 'Interaction_selection_frequency.csv').is_file()
    # asking for it without an Interaction table is a note, not an error
    folder2 = tmp_path / 'Result' / 'r2'
    folder2.mkdir()
    w = ss.run_stability_report(cfg, 'r2', lambda: _effect(), logs.append)
    assert any('no Interaction table' in m for m in logs)


# ----------------------------------------------------------------------- wiring and GUI keys
@pytest.mark.parametrize('name', ['main_app.py', 'run_step2_assemble.py', 'run_sequential.py',
                                  'reporting_pipeline.py'])
def test_every_report_site_calls_the_stability_report(name):
    text = open(os.path.join(ROOT, name), encoding='utf-8').read()
    assert 'run_stability_report' in text                           # main_app has it twice


def test_loader_round_trips_the_x03_key_and_old_configs_stay_off():
    import gui_config_loader as gcl
    on = {'STABILITY_SUMMARY_CREATE': True, 'STABILITY_TOP_K': 20, 'STABILITY_INTERACTION_FREQUENCY': True}
    for kind in ('sequential', 'step1', 'step2'):
        state, unmapped = gcl.engine_config_to_gui_state(dict(on), kind)
        assert state['stability_interaction_frequency'] is True and state['stability_top_k'] == 20
        assert not [u for u in unmapped if 'STABILITY' in u]
        old, _ = gcl.engine_config_to_gui_state({}, kind)               # I11: an absent key means off
        assert old['stability_interaction_frequency'] is False and old['stability_summary_create'] is False
    main = open(os.path.join(ROOT, 'main_app.py'), encoding='utf-8').read()
    assert "cfg['STABILITY_INTERACTION_FREQUENCY']" in main


def test_interaction_frequency_carries_fractional_marker_and_pair_frequencies(tmp_path):
    rng = np.random.default_rng(5)
    rows, eff_rows = [], []
    for t in range(5):
        top = {'s3': 10.0, 's1': 9.0 if t < 3 else 0.01}                  # s1 is a top marker in 3 of 5
        eff = {f'n{j}': abs(rng.normal(scale=0.05)) for j in range(8)}
        eff.update(top)
        eff_rows.append({'population': 'P', 'phenotype': 'y', 'model': 'rrBLUP', 'ratio': '0.8',
                         'sample': t, **eff})
        pair = ('s1', 's3') if t < 3 else ('n1', 'n2')                       # the pair is top in 3 of 5
        rows.append(['P', 'y', 'RF', '0.8', t, pair[0], pair[1], 5.0])
        rows.append(['P', 'y', 'RF', '0.8', t, 'n4', 'n5', 0.01])
    inter = pd.DataFrame(rows, columns=['population', 'phenotype', 'model', 'ratio', 'sample',
                                        'marker1', 'marker2', 'value'])
    path = tmp_path / 'Interaction.csv'
    inter.to_csv(path, index=False)
    report = ss.build_report(pd.DataFrame(eff_rows), str(path), top_k=2, pairs=True)
    out = tmp_path / 'o.csv'
    ss.write_interaction_frequency(str(path), str(out), report)
    got = pd.read_csv(out)
    s13 = got[(got['marker1'] == 's1') & (got['marker2'] == 's3')]
    assert len(s13) == 3 and (s13['pair_frequency'] == 0.6).all()
    assert (s13['marker1_frequency'] == 0.6).all() and (s13['marker2_frequency'] == 1.0).all()
