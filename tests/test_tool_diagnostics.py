"""Tests for ver4-101 Tool-mode per-run diagnostics (tool_diagnostics.py + its wiring in
result_comparison.run_tool / validate_tool_config / run_tool.py and the GUI config loader).
Run: python -m pytest tests/test_tool_diagnostics.py -q. Fixtures are generated here."""
import json
import os
import shutil
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import pipeline_utils as pu  # noqa: E402
import result_comparison as rc  # noqa: E402
import gui_config_loader as gcl  # noqa: E402
import tool_diagnostics as td  # noqa: E402
import test_priority_b as tpb  # noqa: E402


@pytest.fixture
def base(tmp_path, monkeypatch):
    """A private Result/ folder holding two runs ('runA', 'runB')."""
    result = tmp_path / 'Result'
    result.mkdir()
    monkeypatch.setattr(pu, 'RESULT_BASE_DIR', str(result))
    G, y = tpb._population()
    work = tmp_path / 'work'
    work.mkdir()
    run, geno, _ = tpb._run_folder(work, G, y)
    for name in ('runA', 'runB'):
        shutil.copytree(run, result / name)
        pd.DataFrame({'population': ['P'], 'phenotype': ['y'], 'model': ['ensemble'],
                      'ratio': ['(0.6, 0.2, 0.2)'], 'sample': [0],
                      'Pearson correlation': [0.5]}).to_csv(result / name / 'Metric.csv', index=False)
        tpb._weights().to_csv(result / name / 'Weight.csv', index=False)
        test = pd.read_csv(result / name / 'Prediction_result_test.csv')
        test['rrBLUP'] = test['ensemble'] * 0.9 + 0.1 * test['actual']        # members Shapley can score
        test['GBLUP'] = test['ensemble'] * 1.1
        test.to_csv(result / name / 'Prediction_result_test.csv', index=False)
    return result, geno


def cfg_for(folders, geno=None, **extra):
    block = {'folders': folders}
    block.update(extra)
    if geno is not None:
        block['genotypes'] = str(geno)
    return {'RUN_KIND': 'tool', 'TOOL': 'performance_comparison', 'RESULT_NAME': 'Diag',
            'TOOL_GATHER_MODE': 'auto', 'TOOL_COMPARISON': {}, 'TOOL_DIAGNOSTICS': block}


def test_absent_block_changes_nothing(base):
    cfg = cfg_for(['runA'])
    del cfg['TOOL_DIAGNOSTICS']
    assert rc.validate_tool_config(cfg) == []
    res = rc.run_tool(cfg, log=lambda m: None)
    assert res['requested'] == [] and res['written'] == [] and res['errors'] == []
    assert not os.path.exists(os.path.join(pu.RESULT_BASE_DIR, 'Diag', 'Diagnostics'))


def test_default_tools_write_per_run_and_never_into_the_run(base):
    result, _ = base
    before = {p.name for p in (result / 'runA').iterdir()}
    cfg = cfg_for(['runA', 'runB'])
    assert rc.validate_tool_config(cfg) == []
    res = rc.run_tool(cfg, log=lambda m: None)
    assert res['errors'] == [] and res['requested'] == ['diagnostics']
    out = result / 'Diag' / 'Diagnostics'
    for run in ('runA', 'runB'):
        names = {p.name for p in (out / run).iterdir()}
        assert {'Prediction_diagnostics.csv', 'Member_class_shares.csv', 'Member_shapley.csv'} <= names
        assert 'Candidate_domain.csv' not in names          # needs a genotype file: not a default tool
    assert {p.name for p in (result / 'runA').iterdir()} == before      # the run itself is untouched
    assert all(os.path.isfile(p) for p in res['written']) and res['written']


def test_candidate_reliability_runs_when_ticked(base):
    result, geno = base
    cfg = cfg_for(['runA'], geno, tools=['candidate_reliability'], alpha=0.2)
    assert rc.validate_tool_config(cfg) == []
    res = rc.run_tool(cfg, log=lambda m: None)
    assert res['errors'] == []
    assert (result / 'Diag' / 'Diagnostics' / 'runA' / 'Candidate_interval.csv').is_file()


def test_nothing_to_evaluate_is_a_note_not_an_error(base):
    result, _ = base
    logs = []
    res = rc.run_tool(cfg_for(['runA'], tools=['consensus_stability']), log=logs.append)   # no Marker_effect
    assert res['errors'] == [] and 'diagnostics' in res['skipped']
    assert any('NOTE' in m and 'consensus_stability' in m for m in logs)


@pytest.mark.parametrize('block,fragment', [
    ({'folders': ['nope']}, 'does not exist'),
    ({'folders': ['runA', 'runA']}, 'twice'),
    ({'folders': ['runA'], 'tools': ['bogus']}, 'unknown tool'),
    ({'folders': ['runA'], 'tools': []}, 'non-empty'),
    ({'folders': ['runA'], 'split': 'train'}, 'split'),
    ({'folders': ['runA'], 'top_k': 0}, 'top_k'),
    ({'folders': ['runA'], 'alpha': 1.5}, 'alpha'),
    ({'folders': ['runA'], 'tools': ['candidate_reliability']}, 'genotype'),
    ({'folders': ['runA'], 'tools': ['candidate_reliability'], 'genotypes': '/no/such.csv'}, 'not found'),
    ({'folders': ['runA'], 'groups': '/no/such.csv'}, 'groups file'),
])
def test_validation_messages(base, block, fragment):
    cfg = cfg_for([])
    cfg['TOOL_DIAGNOSTICS'] = block
    problems = rc.validate_tool_config(cfg)
    assert problems and any(fragment in p for p in problems), problems


def test_output_folder_must_not_be_a_run(base):
    cfg = cfg_for(['runA'])
    cfg['RESULT_NAME'] = 'runA'
    assert any('finished run' in p for p in rc.validate_tool_config(cfg))


def test_empty_folder_list_is_not_requested(base):
    cfg = cfg_for([])
    assert not td.is_requested(cfg) and rc.validate_tool_config(cfg) == []


def test_run_tool_cli_exit_codes(base, tmp_path):
    import run_tool
    path = tmp_path / 'tool_config.json'
    path.write_text(json.dumps(cfg_for(['runA'])))
    assert run_tool.main(['--config', str(path)]) == 0
    only_skipped = cfg_for(['runA'], tools=['consensus_stability'])
    path.write_text(json.dumps(only_skipped))
    assert run_tool.main(['--config', str(path)]) == 1          # everything skipped: nothing written
    nothing = cfg_for([])
    path.write_text(json.dumps(nothing))
    assert run_tool.main(['--config', str(path)]) == 3


def test_loader_restores_the_block_and_old_configs_unchanged(base):
    cfg = cfg_for(['runA', 'runB'], tools=['member_shapley'], split='valid', top_k=20, pairs=True,
                  alpha=0.2, genotypes='g.csv', groups='gr.csv')
    state, unmapped = gcl.tool_config_to_gui_state(dict(cfg))
    assert not [u for u in unmapped if 'DIAGNOSTICS' in u]
    k = gcl.tool_widget_key
    assert state[k('diagnostics', 'create')] is True
    assert state[k('diagnostics', 'folders')] == ['runA', 'runB']
    assert state[k('diagnostics', 'tools')] == ['member_shapley']
    assert state[k('diagnostics', 'split')] == 'valid' and state[k('diagnostics', 'top_k')] == 20
    assert state[k('diagnostics', 'pairs')] is True and state[k('diagnostics', 'alpha')] == 0.2
    assert state[k('diagnostics', 'genotypes')] == 'g.csv' and state[k('diagnostics', 'groups')] == 'gr.csv'
    old = cfg_for([])
    del old['TOOL_DIAGNOSTICS']
    state, unmapped = gcl.tool_config_to_gui_state(old)
    assert state[k('diagnostics', 'create')] is False and not unmapped
