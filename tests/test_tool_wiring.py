"""Tests for ver4-103: class_hierarchy, pairwise_diversity and three options wired into Tool mode
(`TOOL_DIAGNOSTICS`): `abstain`, `ood_inflate`, `interaction_frequency`.

Run: python -m pytest tests/test_tool_wiring.py -q. Fixtures are generated here."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import gui_config_loader as gcl  # noqa: E402
import result_comparison as rc  # noqa: E402
import tool_diagnostics as td  # noqa: E402
from test_tool_diagnostics import base, cfg_for  # noqa: E402,F401  (the run-folder fixture)
import test_recovery_modules as trm  # noqa: E402

OLD_TOOLS = ('prediction_diagnostics', 'member_importance', 'member_shapley',
             'consensus_stability', 'candidate_reliability')


def test_new_tools_are_offered_but_not_ticked_by_default():
    assert list(td.DIAGNOSTIC_TOOLS)[:5] == list(OLD_TOOLS)            # the old order is kept
    assert {'class_hierarchy', 'pairwise_diversity'} <= set(td.DIAGNOSTIC_TOOLS)
    assert td.DEFAULT_TOOLS == ('prediction_diagnostics', 'member_importance', 'member_shapley',
                                'consensus_stability')                   # I11: same default set
    d = td.with_defaults({'folders': ['r']})
    assert d['abstain'] is False and d['ood_inflate'] == 1.0 and d['interaction_frequency'] is False


def test_argv_is_unchanged_at_defaults_and_adds_flags_only_when_set():
    d = td.with_defaults({'folders': ['r'], 'genotypes': 'g.csv'})
    base_args = ['--result', 'R', '--out-dir', 'O']
    assert td._argv('prediction_diagnostics', 'R', 'O', d) == base_args + ['--split', 'test']
    assert td._argv('member_shapley', 'R', 'O', d) == base_args + ['--split', 'test']
    assert td._argv('consensus_stability', 'R', 'O', d) == base_args + ['--top-k', '50']
    assert td._argv('candidate_reliability', 'R', 'O', d) == base_args + ['--genotypes', 'g.csv', '--alpha', '0.1']
    assert td._argv('class_hierarchy', 'R', 'O', d) == base_args
    assert td._argv('pairwise_diversity', 'R', 'O', d) == base_args
    on = td.with_defaults({'folders': ['r'], 'genotypes': 'g.csv', 'abstain': True,
                           'ood_inflate': 1.5, 'interaction_frequency': True})
    assert td._argv('candidate_reliability', 'R', 'O', on)[-3:] == ['--abstain', '--ood-inflate', '1.5']
    assert td._argv('consensus_stability', 'R', 'O', on)[-1] == '--interaction-frequency'
    assert '--abstain' not in td._argv('consensus_stability', 'R', 'O', on)


@pytest.mark.parametrize('block,fragment', [
    ({'folders': ['runA'], 'ood_inflate': 0.5}, 'ood_inflate'),
    ({'folders': ['runA'], 'ood_inflate': 'wide'}, 'ood_inflate'),
    ({'folders': ['runA'], 'ood_inflate': True}, 'ood_inflate'),
    ({'folders': ['runA'], 'abstain': 'yes'}, 'abstain'),
    ({'folders': ['runA'], 'interaction_frequency': 1}, 'interaction_frequency'),
])
def test_validation_of_the_new_options(base, block, fragment):
    cfg = cfg_for([])
    cfg['TOOL_DIAGNOSTICS'] = block
    problems = rc.validate_tool_config(cfg)
    assert any(fragment in p for p in problems), problems
    ok = cfg_for(['runA'], ood_inflate=1.25, abstain=True, interaction_frequency=False)
    assert rc.validate_tool_config(ok) == []


def test_class_hierarchy_and_pairwise_diversity_run_from_tool_mode(base):
    result, _ = base
    cfg = cfg_for(['runA'], tools=['class_hierarchy', 'pairwise_diversity'])
    assert rc.validate_tool_config(cfg) == []
    res = rc.run_tool(cfg, log=lambda m: None)
    assert res['errors'] == [], res['errors']
    names = {p.name for p in (result / 'Diag' / 'Diagnostics' / 'runA').iterdir()}
    assert {'Class_hierarchy.csv', 'Class_hierarchy_summary.csv', 'Class_hierarchy_weights.csv',
            'Pairwise_diversity.csv', 'Pairwise_diversity_summary.csv'} <= names
    assert not any(p.name.startswith('Class_hierarchy') for p in (result / 'runA').iterdir())   # run untouched


def test_candidate_options_reach_the_tool(base):
    result, geno = base
    out = result / 'Diag' / 'Diagnostics' / 'runA'
    plain = rc.run_tool(cfg_for(['runA'], geno, tools=['candidate_reliability']), log=lambda m: None)
    assert plain['errors'] == [] and 'Candidate_abstention.csv' not in {p.name for p in out.iterdir()}
    opt = rc.run_tool(cfg_for(['runA'], geno, tools=['candidate_reliability'], abstain=True,
                              ood_inflate=1.5), log=lambda m: None)
    assert opt['errors'] == [], opt['errors']
    assert 'Candidate_abstention.csv' in {p.name for p in out.iterdir()}
    interval = pd.read_csv(out / 'Candidate_interval.csv')
    assert 'inflation' in interval.columns and set(interval['inflation']) <= {1.0, 1.5}


def test_interaction_frequency_is_written_through_tool_mode(base):
    result, _ = base
    for name in ('runA', 'runB'):
        trm._effect().to_csv(result / name / 'Marker_effect.csv', index=False)
        trm._interactions().to_csv(result / name / 'Interaction.csv', index=False)
    off = rc.run_tool(cfg_for(['runA'], tools=['consensus_stability'], pairs=True), log=lambda m: None)
    out = result / 'Diag' / 'Diagnostics' / 'runA'
    assert off['errors'] == [] and 'Interaction_selection_frequency.csv' not in {p.name for p in out.iterdir()}
    on = rc.run_tool(cfg_for(['runA'], tools=['consensus_stability'], top_k=2, interaction_frequency=True),
                     log=lambda m: None)
    assert on['errors'] == [], on['errors']
    assert (out / 'Interaction_selection_frequency.csv').is_file()
    assert any(p.endswith('Interaction_selection_frequency.csv') for p in on['written'])
    assert (result / 'runA' / 'Interaction.csv').read_bytes() == \
        (result / 'runB' / 'Interaction.csv').read_bytes()                  # sources untouched


def test_loader_round_trip_and_old_configs_reset_the_new_widgets():
    cfg = cfg_for(['runA'], tools=['class_hierarchy'], abstain=True, ood_inflate=1.4,
                  interaction_frequency=True)
    state, unmapped = gcl.tool_config_to_gui_state(json.loads(json.dumps(cfg)))
    k = gcl.tool_widget_key
    assert not [u for u in unmapped if 'DIAGNOSTICS' in u]
    assert state[k('diagnostics', 'abstain')] is True and state[k('diagnostics', 'ood_inflate')] == 1.4
    assert state[k('diagnostics', 'interaction_frequency')] is True
    assert state[k('diagnostics', 'tools')] == ['class_hierarchy']
    old = cfg_for(['runA'])                                              # a ver4-101 block
    state, unmapped = gcl.tool_config_to_gui_state(old)
    assert state[k('diagnostics', 'abstain')] is False and state[k('diagnostics', 'ood_inflate')] == 1.0
    assert state[k('diagnostics', 'interaction_frequency')] is False
    del old['TOOL_DIAGNOSTICS']
    state, unmapped = gcl.tool_config_to_gui_state(old)                  # no block at all
    assert state[k('diagnostics', 'abstain')] is False and not unmapped


# ------------------------------------------------------------------------------- the GUI itself
@pytest.fixture
def clean_gui_state_file():
    """Running the real app script writes `easigp_last_gui_state.json` into the source folder;
    remove it afterwards unless it was already there, so a test run leaves the tree as it was."""
    path = os.path.join(ROOT, 'easigp_last_gui_state.json')
    existed = os.path.exists(path)
    yield
    if not existed and os.path.exists(path):
        os.remove(path)


def _gui_app():
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.environ['EASIGP_TEST_ROOT'])      # from_function copies only the body
    import main_app as m

    class Source:
        def __init__(self, folder):
            self.folder = folder
    m._render_tool_diagnostics_section([Source('runA'), Source('runB')])


def test_the_real_gui_section_shows_the_new_widgets_off_by_default(clean_gui_state_file):
    pytest.importorskip('streamlit')
    pytest.importorskip('torch_geometric')                                # main_app imports the models
    from streamlit.testing.v1 import AppTest
    os.environ['EASIGP_TEST_ROOT'] = ROOT
    at = AppTest.from_function(_gui_app, default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    labels = [c.label for c in at.checkbox]
    assert any('selection frequency' in x for x in labels)
    assert any('withholding unreliable lines' in x for x in labels)
    assert any('widen intervals' in n.label for n in at.number_input)
    tools = [ms for ms in at.multiselect if any('class_hierarchy' in str(o) for o in ms.options)]
    assert tools and any('pairwise_diversity' in str(o) for o in tools[0].options)
    tick = next(c for c in at.checkbox if 'withholding unreliable lines' in c.label)
    assert tick.value is False                                            # default off


def _gui_gather_app():
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.environ['EASIGP_TEST_ROOT'])
    _sys.path.insert(0, _os.path.join(_os.environ['EASIGP_TEST_ROOT'], 'tests'))
    import json as _json
    import shutil
    import pandas as _pd
    import streamlit as st
    import main_app as m
    import pipeline_utils as pu
    import test_priority_b as tpb
    result = _os.path.join(_os.environ['EASIGP_TEST_TMP'], 'Result')
    if not _os.path.isdir(result):
        _os.makedirs(result)
        G, y = tpb._population()
        run, _geno, _ = tpb._run_folder(__import__('pathlib').Path(_os.environ['EASIGP_TEST_TMP']), G, y)
        shutil.copytree(run, _os.path.join(result, 'runA'))
        _pd.DataFrame({'population': ['P'], 'phenotype': ['y'], 'model': ['ensemble'],
                       'ratio': ['(0.6, 0.2, 0.2)'], 'sample': [0],
                       'Pearson correlation': [0.5]}).to_csv(_os.path.join(result, 'runA', 'Metric.csv'), index=False)
    pu.RESULT_BASE_DIR = result
    wk = m.gui_config_loader.tool_widget_key
    d = 'diagnostics'
    st.session_state.setdefault('tool_result_name', 'Out')
    st.session_state[wk(d, 'create')] = True
    st.session_state[wk(d, 'folders')] = ['runA']
    st.session_state[wk(d, 'tools')] = ['class_hierarchy', 'candidate_reliability']
    st.session_state[wk(d, 'genotypes')] = _os.path.join(_os.environ['EASIGP_TEST_TMP'], 'geno.csv')
    st.session_state[wk(d, 'abstain')] = _os.environ.get('EASIGP_TEST_ABSTAIN') == '1'
    st.session_state[wk(d, 'ood_inflate')] = float(_os.environ.get('EASIGP_TEST_INFLATE', '1.0'))
    st.session_state[wk(d, 'interaction_frequency')] = False
    cfg = m.gather_tool_config()
    st.session_state['gathered'] = _json.dumps(cfg['TOOL_DIAGNOSTICS'], sort_keys=True)


def test_the_gui_exports_new_options_only_when_they_are_set(tmp_path, clean_gui_state_file):
    pytest.importorskip('streamlit')
    pytest.importorskip('torch_geometric')
    from streamlit.testing.v1 import AppTest
    os.environ['EASIGP_TEST_ROOT'] = ROOT
    os.environ['EASIGP_TEST_TMP'] = str(tmp_path)
    (tmp_path / 'geno.csv').write_text('ID,population,m1\nL1,P,0\n')
    for flag, inflate in (('0', '1.0'), ('1', '1.5')):
        os.environ['EASIGP_TEST_ABSTAIN'], os.environ['EASIGP_TEST_INFLATE'] = flag, inflate
        at = AppTest.from_function(_gui_gather_app, default_timeout=180)
        at.run()
        assert not at.exception, at.exception
        block = json.loads(at.session_state['gathered'])
        assert block['tools'] == ['class_hierarchy', 'candidate_reliability']
        if flag == '0':
            assert set(block) == {'folders', 'tools', 'split', 'top_k', 'pairs', 'alpha',
                                  'genotypes', 'groups'}                   # exactly the ver4-101 keys
        else:
            assert block['abstain'] is True and block['ood_inflate'] == 1.5
            assert 'interaction_frequency' not in block
