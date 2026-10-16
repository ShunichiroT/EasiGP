"""gui_config_loader.py - ver4-25, R1 ("Load a previous configuration JSON
into the GUI").

Pure module (I2): imports only `hparam_specs`/`pipeline_utils` (neither of
which import `streamlit` themselves), never `streamlit` directly - so it can
be imported/unit-tested outside a live Streamlit session, and so
`main_app.py` (the only caller) stays the single place that actually reads
or writes `st.session_state`.

Three file "kinds" this loader understands, discovered by
`discover_config_files()`/classified by `detect_config_kind()`:

- 'gui_state': the flat `st.session_state` snapshot `main_app.py::
  save_gui_state()` writes to `./easigp_last_gui_state.json` (lower-case
  widget keys) - and, going forward, the per-purpose sidecar
  `Result/<NAME>/<purpose>_gui_state.json` `render_hpc_export_section()`
  now writes alongside every exported `*_config.json` (see
  `sidecar_path()`). Loading one of these is always LOSSLESS (a full
  key-for-key replace) - it's the exact same shape `load_gui_state()`
  already restores from on every app launch, just aimed at a different
  file/moment.
- 'sequential' / 'step1' / 'step2': an ENGINE config (upper-case keys) -
  `gather_config()`'s own output, written by `render_hpc_export_section()`.
  There is no general inverse of `gather_config()` (many widget keys -
  dynamic HPARAM_SPECS fields, ASReml sub-widgets - aren't recoverable
  from the engine config alone), so loading one of these WITHOUT a sidecar
  goes through `engine_config_to_gui_state()`'s best-effort inverse map
  instead, which reports exactly what it could not restore.

See `main_app.py::_render_config_loader_section()`/
`_apply_pending_config_load()` for how this module's output is actually
applied to `st.session_state` (a two-phase apply + `st.rerun()`, since
Streamlit forbids assigning an already-instantiated widget's key).
"""
import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

from hparam_specs import HPARAM_SPECS
import circos_highlight as _hl_mod   # pure (stdlib + NumPy; matplotlib only lazily) - keeps this module streamlit-free
# pipeline_utils.restore_ratio() - see restore_ratio_text()'s own docstring
# below for why this is needed (JSON flattens a tuple-valued ratio entry to
# a plain list - the SAME fix genomic_prediction.py's own headless callers
# already need before a JSON-round-tripped RATIO reaches GP()). Confirmed
# streamlit-free (this module's own I2 constraint) - see this file's own
# module docstring.
from pipeline_utils import ACCELERATION_LEGACY_DEFAULTS, GAT_STABILITY_ENGINE_DEFAULTS
from pipeline_utils import _ITER_ENS_DEFAULTS   # ver4-57: widget defaults, single authority
from pipeline_utils import _SL_DEFAULTS   # ver4-77: Super-learner widget defaults, single authority
from pipeline_utils import restore_ratio as _restore_ratio_shape

#: Options of the ver4-35 selectbox widgets in the GUI's Acceleration section
#: (must match `main_app.py`); a config value outside them loads as legacy and is
#: reported as unmapped instead of crashing the widget.
_ACCEL_SELECT_CHOICES = {
    'gpu_slots': ('legacy', 'auto', '1', '2', '3', '4'),
    'torch_device_policy': ('legacy', 'auto', 'cpu', 'gpu'),
    'bio_prior_cost_guard': ('warn', 'raise', 'off'),
    'merge_shap_device': ('cpu', 'gpu'),
    'baseline_shap_engine': ('legacy', 'fast'),     # ver4-43 (R1)
    'trial_memory_guard': ('auto', 'off'),          # ver4-61 (R2)
}

# The four W_OPT method names main_app.py's own W_OPT_METHODS list offers -
# duplicated here (rather than imported - I2, this file never imports
# main_app.py) purely to restore each method's 'wopt_selected_<method>'
# checkbox; see the 'HYPERPARAMETERS_OPT' restoration block below for how
# each method's own hyperparameter VALUES are restored (no field-type
# table needed there at all - see that block's own comment).
_WOPT_METHODS = ('Nelder Mead', 'Linear transformation', 'Bayesian optimisation', 'Analytic least-squares')

# ver4-26, R1/R4 (design blueprint `EasiGP_ver4-26_Design_Blueprint.md`):
# the joint hyperparameter+ensemble-weight search's own budget field
# keys - main_app.py's own JOINT_HYPERPARAM_OPT_SPECS list, in order,
# each paired with its own 'opt_key' - duplicated here rather than
# imported (I2, exactly like _WOPT_METHODS above). resolve_joint_wopt()
# builds HYPERPARAMETERS_JOINT_OPT as {opt_key: value}, keyed by name
# rather than position (unlike HYPERPARAMETERS_OPT's per-method
# positional lists) - this is the exact inverse mapping, opt_key -> field
# index, used to write each value back to its own 'joint_wopt_<idx>'
# widget key below.
_JOINT_OPT_KEY_TO_IDX = {
    'n_iter': 0, 'init_points': 1, 'seed': 2, 'objective_mode': 3,
    # BUGFIX (weight-collapse report): the joint search's own weight-
    # collapse protection settings, appended to main_app.py's
    # JOINT_HYPERPARAM_OPT_SPECS at indices 4-7 - added here in lockstep
    # (see this dict's own module-level comment) so a config carrying
    # these keys restores into the GUI instead of falling into
    # `unmapped`.
    'diversity_penalty': 4, 'diversity_method': 5, 'shrinkage_alpha': 6, 'analytic_seed': 7,
    # These five were appended to JOINT_HYPERPARAM_OPT_SPECS in later rounds
    # (valid_repeats/untuned_baseline_floor/multi_fidelity in earlier ones,
    # weight_cv/fresh_gate in Round 10) but were never added here, so a
    # config carrying them was reported as `unmapped` instead of restoring
    # its GUI widget. Kept in lockstep with main_app.py's list.
    'valid_repeats': 8, 'untuned_baseline_floor': 9, 'multi_fidelity': 10,
    'weight_cv': 11, 'fresh_gate': 12,
    # ver4-28 (design blueprint EasiGP_ver4-28_Design_Blueprint.md):
    # appended to JOINT_HYPERPARAM_OPT_SPECS at indices 13-22 - kept in
    # lockstep here for the same reason as every entry above.
    'search_strategy': 13, 'block_cycles': 14, 'block_init_points': 15,
    'block_n_iter': 16, 'block_refine_iter': 17, 'block_tol': 18,
    'parallel_trials': 19, 'reuse_confirmatory_fit': 20,
    'weight_source': 21, 'oof_folds': 22,
    # ver4-32 (design blueprint EasiGP_ver4-32_joint-speed-accuracy_Design_Blueprint.md):
    # appended to JOINT_HYPERPARAM_OPT_SPECS at indices 23-28 - kept in
    # lockstep here for the same reason as every entry above.
    'freeze_fidelity_fields': 23, 'warm_start': 24, 'block_warm_n_iter': 25,
    'score_weight_rule': 26, 'search_patience': 27, 'gate_fit_cache': 28,
    # ver4-34 (design blueprint EasiGP_ver4-34_Design_Blueprint.md): appended to
    # JOINT_HYPERPARAM_OPT_SPECS at indices 29-38 - kept in lockstep here for the same
    # reason as every entry above. (`score_bags` is engine-only and has no widget.)
    'weight_estimator': 29, 'weight_bags': 30, 'cluster_corr': 31, 'objective_data': 32,
    'cv_folds': 33, 'search_space': 34, 'freeze_low_weight': 35, 'time_budget_s': 36,
    'library_rounds': 37, 'library_q': 38,
    # ver4-36 (design blueprint EasiGP_ver4-36_Design_Blueprint.md): appended to
    # JOINT_HYPERPARAM_OPT_SPECS at indices 39-45 - kept in lockstep here for the same reason
    # as every entry above. (The remaining ver4-36 keys - bbc_gate_quantile, library_max_evals,
    # library_refine_rounds, es_steps, es_model_fraction, stub_parity_min, engine_parity_min,
    # library_dump (a widget since ver4-51, index 85) - are engine-only and have no widget.)
    'gate_mode': 39, 'bbc_bootstraps': 40, 'search_test_stub': 41, 'search_engine': 42,
    'library_evals_per_dim': 43, 'noise_aware_surrogate': 44, 'selection_rule': 45,
    # ver4-38 (design blueprint EasiGP_ver4-38_Design_Blueprint.md): appended to
    # JOINT_HYPERPARAM_OPT_SPECS at indices 46-51 - kept in lockstep here for the same reason as
    # every entry above. (racing_margin, racing_penalty, library_members_max and
    # library_members_per_model are engine-only and have no widget.)
    'cv_racing': 46, 'racing_folds': 47, 'library_stop_rule': 48, 'oof_fidelity': 49,
    'dead_column_guard': 50, 'library_ensemble': 51,
    # ver4-40 (design blueprint EasiGP_ver4-40_Design_Blueprint.md, S7-S9): appended to
    # JOINT_HYPERPARAM_OPT_SPECS at indices 52-60 (weight-first strategy), same lockstep rule.
    'wf_rules': 52, 'wf_max_candidates': 53, 'wf_dedup_tol': 54, 'wf_rungs': 55,
    'wf_eta': 56, 'wf_min_tune_weight': 57, 'wf_include_untuned': 58, 'wf_pick': 59,
    'wf_candidate_rows': 60,
    # ver4-42 (design blueprint EasiGP_ver4-42_Design_Blueprint.md): appended at indices 61-67
    # ('weight_first_cv' strategy), same lockstep rule.
    'wf_cv_weight_fidelity': 61, 'wf_cv_direction': 62, 'wf_cv_alphas': 63,
    'wf_cv_alpha_rule': 64, 'wf_cv_repeats': 65, 'wf_cv_tune': 66, 'wf_cv_accept': 67,
    # ver4-44 (design blueprint EasiGP_ver4-44_Design_Blueprint.md): appended at indices 68-69
    # (R1 `safety_nets`, R3 `wf_cv_final`), same lockstep rule.
    'safety_nets': 68, 'wf_cv_final': 69,
    # ver4-47 (R5 `wf_cv_reweight`, R6 `wf_cv_members`): appended at indices 70-71, same rule.
    'wf_cv_reweight': 70, 'wf_cv_members': 71,
    # ver4-48 (R3, neural stack): appended at indices 72-78, same rule.
    'ns_configs': 72, 'ns_configs_mcmc': 73, 'ns_lambdas': 74, 'ns_lambda_rule': 75,
    'ns_repeats': 76, 'ns_deploy': 77, 'ns_fidelity': 78,
    # ver4-50 (R4, neural stack): appended at indices 79-81, same rule.
    'ns_objective': 79, 'ns_swap_rule': 80, 'ns_weight_shrink': 81,
    # ver4-51 (R3): appended at indices 82-85, same rule. `library_dump` (85) was engine-only until
    # this version: the GUI can now write the per-task dump that tools/joint_replay.py reads.
    'ns_outer_gate': 82, 'ns_outer_folds': 83, 'ns_outer_repeats': 84, 'library_dump': 85,
}

# ver4-34, R1.b: what each joint key MEANS when a loaded config does not contain it - the
# engine's own `.get(key, <legacy default>)` value (I11), NOT the widget's new-session
# default. Only keys whose widget default can differ from the engine default need an entry
# for correctness (idx 19, 20, 23 now show True in a new session); the ver4-34 keys are
# listed too so an old JSON restores exactly the behaviour it always ran with, whatever the
# widget defaults become later. Keep in step with `joint_tuning._JOINT_V428_KEY_DEFAULTS`.
_JOINT_OPT_ABSENT_LEGACY_VALUES = {
    'parallel_trials': False, 'reuse_confirmatory_fit': False,
    'freeze_fidelity_fields': False,
    'weight_estimator': 'legacy', 'weight_bags': 100, 'cluster_corr': 0.97,
    'objective_data': 'valid', 'cv_folds': 5, 'search_space': 'all',
    'freeze_low_weight': 0.0, 'time_budget_s': 0, 'library_rounds': 4, 'library_q': 3,
    # ver4-36
    'gate_mode': 'fresh_refit', 'bbc_bootstraps': 200, 'search_test_stub': False,
    'search_engine': 'bglr', 'library_evals_per_dim': 6, 'noise_aware_surrogate': False,
    'selection_rule': 'greedy',
    # ver4-38
    'cv_racing': False, 'racing_folds': 2, 'library_stop_rule': 'off', 'oof_fidelity': 'full',
    'dead_column_guard': False, 'library_ensemble': False,
    # ver4-40 (engine defaults; a config written before ver4-40 never selects weight_first)
    'wf_rules': ['uniform', 'analytic', 'shrunk:0.25', 'shrunk:0.5', 'shrunk:0.75', 'es'],
    'wf_max_candidates': 6, 'wf_dedup_tol': 0.05, 'wf_rungs': 3, 'wf_eta': 2,
    'wf_min_tune_weight': 0.05, 'wf_include_untuned': True, 'wf_pick': 'bbc',
    'wf_candidate_rows': 'scoring',
    # ver4-42 (engine defaults; a config written before ver4-42 never selects weight_first_cv)
    'wf_cv_weight_fidelity': 'full', 'wf_cv_direction': 'analytic',
    'wf_cv_alphas': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'wf_cv_alpha_rule': 'min', 'wf_cv_repeats': 4, 'wf_cv_tune': True,
    'wf_cv_accept': 'bbc_mean',
    # ver4-44 (engine defaults: a config written before ver4-44 runs with every safety net on)
    'safety_nets': True, 'wf_cv_final': 'refit',
    # ver4-47 (legacy: a config written before ver4-47 re-weights nothing)
    'wf_cv_reweight': 'off', 'wf_cv_members': False,
    # ver4-48 (engine defaults: a config written before ver4-48 never selects neural_stack, so
    # these are read only by a run that chose it afterwards)
    'ns_configs': 6, 'ns_configs_mcmc': 3, 'ns_lambdas': [0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3],
    'ns_lambda_rule': 'one_se', 'ns_repeats': 4, 'ns_deploy': 'mixture', 'ns_fidelity': 'full',
    # ver4-50 (the engine's own values: a config written before ver4-50 trained on the ensemble
    # score, admitted every gate switch and used the ver4-49 shrink of 0.5)
    'ns_objective': 'ensemble_score', 'ns_swap_rule': 'gate', 'ns_weight_shrink': 0.5,
    # ver4-51 (a config written before ver4-51 ran no nested gate and wrote no library dump)
    'ns_outer_gate': 'off', 'ns_outer_folds': 5, 'ns_outer_repeats': 2, 'library_dump': False,
}

_LD_UNIT_TO_STATE_SUFFIX = {'kb': 'kb', 'cm': 'cm', 'variants': 'variants'}


# --------------------------------------------------------------------------- #
# ver4-54 (R1/R3/R6): "Tool" mode widget-key authority.
#
# Single place that names the Tool-mode widgets, so `main_app.py` (which
# renders them), `result_comparison.py` (which consumes the config built from
# them) and `tool_config_to_gui_state()` below (the inverse map) can never
# disagree on a key. Pure strings/functions - no heavy import (I2).
# --------------------------------------------------------------------------- #

#: Separates a result folder from an ASReml option inside a source id,
#: e.g. 'MaizeRun :: BLUEs'. `result_comparison.discover_result_sources()` skips
#: any folder whose own name contains it.
SOURCE_SEP = ' :: '

#: The three comparison sections of the Tool tab "Performance comparison".
TOOL_SECTIONS = ('pearson_mse', 'adjusted_accuracy', 'spearman')

#: ver4-60: the fourth, plot-only section "Comparison of metrics between results"
#: (matched-task centroid scatter of each metric between the selected results). It is
#: deliberately NOT in `TOOL_SECTIONS`: that tuple drives the violin / summary-workbook
#: widgets and config fields, which this section does not have. It shares the
#: `tool_<section>_<field>` widget-key scheme, so `tool_widget_key` / `tool_label_key`
#: work for it unchanged.
TOOL_SCATTER_SECTION = 'across_results'

#: Every Tool section, in display order.
TOOL_ALL_SECTIONS = TOOL_SECTIONS + (TOOL_SCATTER_SECTION,)

#: ver4-101: the optional per-run diagnostics block of a Tool config (`TOOL_DIAGNOSTICS`). Its
#: widget keys follow the same `tool_<section>_<field>` scheme; it is not a comparison section.
TOOL_DIAGNOSTICS_SECTION = 'diagnostics'

#: `TOOL_DIAGNOSTICS` key -> widget field suffix (`folders` and `tools` are multiselects).
_TOOL_DIAGNOSTICS_FIELDS = (
    ('folders', 'folders'), ('tools', 'tools'), ('split', 'split'), ('top_k', 'top_k'),
    ('pairs', 'pairs'), ('alpha', 'alpha'), ('genotypes', 'genotypes'), ('groups', 'groups'),
    ('abstain', 'abstain'), ('ood_inflate', 'ood_inflate'),                # ver4-103
    ('interaction_frequency', 'interaction_frequency'),
)

#: ver4-103: the options added after ver4-101 and their defaults. A saved config that lacks one
#: must switch the widget back to it (I11), not leave a tick from an earlier load in place.
_TOOL_DIAGNOSTICS_NEW_DEFAULTS = {'abstain': False, 'ood_inflate': 1.0,
                                  'interaction_frequency': False}

#: Tool-config `plot_config` key -> widget field suffix (`tool_{section}_{suffix}`).
_TOOL_PLOT_FIELDS = (
    ('font_size', 'font'), ('fig_size', 'fig'), ('style', 'style'),
    ('min_n_violin', 'min_n_violin'), ('legend', 'legend'),
    ('between_label', 'between_label'), ('cut', 'cut'), ('show_mean_se', 'show_mean_se'),
    ('show_median', 'show_median'),
    ('auto_width', 'auto_width'), ('group_separators', 'group_separators'),
    ('reference', 'reference'), ('x_label_mode', 'x_label_mode'),
)

#: Tool-config section key -> widget field suffix, for the non-plot section fields.
_TOOL_SECTION_FIELDS = (
    ('create_plot', 'create'), ('create_summary', 'summary_create'),
    ('summary_statistic', 'summary_statistic'), ('summary_decimals', 'summary_decimals'),
    ('mark_best', 'mark_best'), ('dpi', 'dpi'),
    ('metrics', 'metrics'),   # ver4-60: only the scatter section carries it
)


def tool_widget_key(section, field):
    """`'tool_<section>_<field>'` - the session_state key of one Tool-section
    widget (`field` e.g. 'sources', 'models', 'create', 'font')."""
    return f'tool_{section}_{field}'


def make_source_id(folder, option=None):
    """A source = (result folder, ASReml option or None). Its id is the folder
    itself for a single-option / non-ASReml result, else `folder + SOURCE_SEP +
    option`."""
    return folder if option is None else f'{folder}{SOURCE_SEP}{option}'


def split_source_id(source_id):
    """Inverse of `make_source_id`: `source_id` -> `(folder, option or None)`.
    A folder name never contains `SOURCE_SEP` (discovery skips such folders), so
    the first separator is the split point."""
    folder, sep, option = str(source_id).partition(SOURCE_SEP)
    return (folder, option) if sep else (folder, None)


def default_source_label(folder, option=None):
    """The default editable name of a source: the folder name, or
    `'folder (option)'` for one option of a multi-option ASReml result."""
    return folder if option is None else f'{folder} ({option})'


def tool_label_key(section, source_id):
    """session_state key of the editable-name text box of one selected source
    in one section."""
    return f'tool_{section}_label{SOURCE_SEP}{source_id}'


@dataclass
class ConfigCandidate:
    path: str
    kind: str  # 'gui_state' | 'sequential' | 'step1' | 'step2' | 'tool' | 'unknown'
    result_name: Optional[str]
    mtime: float


def read_json(path):
    try:
        with open(path, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    except Exception:
        return None


def detect_config_kind(obj, filename):
    """Filename first (exact - matches this codebase's own fixed naming
    convention for every file `discover_config_files()` looks for and
    every file `render_hpc_export_section()`/`save_gui_state()` write),
    then a key-based heuristic as a fallback for a renamed/hand-copied
    file. `obj` must already be the parsed JSON (a dict); anything else
    (parse failure, a JSON array/scalar) is 'unknown'."""
    if not isinstance(obj, dict):
        return 'unknown'
    name = os.path.basename(filename or '')
    if name == 'easigp_last_gui_state.json' or name.endswith('_gui_state.json'):
        return 'gui_state'
    # ver4-54 (R6): a Tool config is recognised by its filename or, for a
    # renamed copy, by RUN_KIND - checked BEFORE the pipeline heuristics below
    # (a tool config carries RESULT_NAME but none of their discriminating keys).
    if name == 'tool_config.json' or obj.get('RUN_KIND') == 'tool':
        return 'tool'
    if name == 'sequential_config.json':
        return 'sequential'
    if name == 'step1_config.json':
        return 'step1'
    if name == 'step2_config.json':
        return 'step2'
    # Heuristic fallback. 'exec_mode'/'r_path_none' only ever appear in a
    # flat save_gui_state() snapshot (raw widget keys) - no engine config
    # key is ever named either of those. Among engine configs: only
    # gather_config()'s Step 1 branch ever sets 'PARALLEL'; only its
    # Step 2 branch ever sets 'ASSEMBLE_MODE'; Sequential sets neither but
    # (like Step 1) always sets 'MODEL'.
    if 'exec_mode' in obj or 'r_path_none' in obj:
        return 'gui_state'
    if 'RESULT_NAME' not in obj:
        return 'unknown'
    if 'PARALLEL' in obj:
        return 'step1'
    if 'ASSEMBLE_MODE' in obj:
        return 'step2'
    if 'MODEL' in obj:
        return 'sequential'
    return 'unknown'


def _candidate_from_path(path, kind_hint=None):
    obj = read_json(path)
    if obj is None:
        return None
    kind = kind_hint or detect_config_kind(obj, os.path.basename(path))
    if kind == 'gui_state':
        result_name = obj.get('result_name') if isinstance(obj, dict) else None
    else:
        result_name = obj.get('RESULT_NAME') if isinstance(obj, dict) else None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return ConfigCandidate(path=path, kind=kind, result_name=result_name, mtime=mtime)


def discover_config_files(root='.'):
    """Every GUI-state/engine-config file this loader could load, newest
    first: the flat GUI-state file at `<root>/easigp_last_gui_state.json`
    (`save_gui_state()`'s own target) plus every `Result/*/{sequential,
    step1,step2}_config.json` under `<root>` (written by
    `render_hpc_export_section()`). A file that fails to parse as JSON is
    skipped silently - discovery is best-effort; `_render_config_loader_
    section()` surfaces a real error only once the person actually tries
    to load a specific file."""
    found = []
    gui_state_path = os.path.join(root, 'easigp_last_gui_state.json')
    if os.path.isfile(gui_state_path):
        candidate = _candidate_from_path(gui_state_path)
        if candidate is not None:
            found.append(candidate)
    for filename in ('sequential_config.json', 'step1_config.json', 'step2_config.json',
                     'tool_config.json'):
        for path in sorted(glob.glob(os.path.join(root, 'Result', '*', filename))):
            candidate = _candidate_from_path(path)
            if candidate is not None:
                found.append(candidate)
    found.sort(key=lambda c: c.mtime, reverse=True)
    return found


def sidecar_path(config_path):
    """'.../step1_config.json' -> '.../step1_gui_state.json' (same
    directory, '_config.json' suffix swapped for '_gui_state.json') - the
    lossless companion file `render_hpc_export_section()` now writes
    alongside every exported engine config (see that function's own
    comment), carrying the exact filtered `st.session_state` snapshot
    `_gui_state_snapshot()` builds. A config file not ending in
    '_config.json' (defensive - every caller of this function only ever
    passes a real `*_config.json` path) falls back to swapping its
    extension instead, so this never raises."""
    directory = os.path.dirname(config_path)
    base = os.path.basename(config_path)
    if base.endswith('_config.json'):
        sidecar_name = base[:-len('_config.json')] + '_gui_state.json'
    else:
        root, _ext = os.path.splitext(base)
        sidecar_name = root + '_gui_state.json'
    return os.path.join(directory, sidecar_name)


# --------------------------------------------------------------------------- #
# Best-effort inverse of gather_config() - engine config -> GUI widget keys.
# --------------------------------------------------------------------------- #

def _restore_hparam_field(state, key, field_spec, value):
    """Inverse of main_app.py's own render_field()/resolve_field() pair for
    ONE HPARAM_SPECS (or HYPERPARAM_OPT_SPECS) field - writes whatever
    session_state key(s) render_field() itself would have written for
    `value`, mirroring resolve_field()'s own per-type branching 1:1 (see
    that function's docstring in main_app.py). A field whose type renders
    nothing ('json') is left alone - nothing to restore into."""
    ftype = field_spec['type']
    if ftype in ('bool', 'int', 'float', 'str', 'file_path'):
        state[key] = value
        return
    if ftype == 'json':
        return
    mode_key, custom_key = key + '__mode', key + '__custom'
    if ftype in ('top_pct', 'int_or_all'):
        if value == 'all':
            state[mode_key] = 'all'
        else:
            state[mode_key] = 'custom number'
            state[custom_key] = value
        return
    if ftype == 'rf_max_features':
        if value in ('sqrt', 'log2'):
            state[mode_key] = value
        elif value is None:
            state[mode_key] = 'None'
        else:
            state[mode_key] = 'custom number'
            state[custom_key] = value
        return
    if ftype == 'int_float_or_none':
        if value is None:
            state[mode_key] = 'None'
        else:
            state[mode_key] = 'custom number'
            state[custom_key] = value
        return
    if ftype == 'svr_gamma':
        if value in ('scale', 'auto'):
            state[mode_key] = value
        else:
            state[mode_key] = 'custom number'
            state[custom_key] = value
        return
    # Unknown/future field type - leave unrestored rather than guess.


def restore_ratio_text(ratio):
    """Inverse of gather_config()'s own RATIO parsing
    (`ast.literal_eval(f'[{ratio_text}]')`) - `ratio_text` is simply the
    list's own `repr()` with the outer brackets stripped, e.g. `[0.8,
    0.65]` -> `'0.8, 0.65'`, `[(0.8, 0.1, 0.1)]` -> `'(0.8, 0.1, 0.1)'` -
    re-parsing either of those (wrapped back in `[...]`) via
    `ast.literal_eval` reproduces `ratio` exactly, so this round-trips by
    construction rather than by re-deriving gather_config()'s own parsing
    logic a second time.

    Named distinctly from (and NOT to be confused with)
    `pipeline_utils.restore_ratio(ratio_list, scenario)` - a pre-existing,
    unrelated function that fixes a DIFFERENT problem (JSON flattening a
    tuple-valued ratio entry to a plain list before GP() itself ever sees
    it). `ratio` here is called AFTER that fix has already been applied
    (see this function's own call site in `engine_config_to_gui_state()`),
    so by the time this function runs, a 3-way split ratio is already a
    genuine `tuple`, not a list - `repr()` renders it with the parentheses
    `ast.literal_eval` needs to read it back as a tuple again."""
    text = repr(list(ratio))
    return text[1:-1] if text.startswith('[') and text.endswith(']') else text


def _base_model_name(name):
    """'GAT_biological_prior_knowledge_3' -> 'GAT_biological_prior_
    knowledge' (strips a trailing '_<k>' repeat-network instance suffix -
    see gather_config()'s own `_bio_instance_names` construction). Every
    other model name is returned unchanged (it never carries this
    suffix)."""
    if name.startswith('GAT_biological_prior_knowledge_'):
        return 'GAT_biological_prior_knowledge'
    return name


def engine_config_to_gui_state(cfg, kind):
    """Best-effort inverse of `gather_config(mode, step)` - the DESIGN
    RECORD'S own R1 touch point. Returns `(state, unmapped_engine_keys)`:
    `state` is a dict of widget-key -> value ready to be merged into
    `st.session_state` (see `_apply_pending_config_load()`); `unmapped_
    engine_keys` lists every `cfg` key (top-level, or `'PARENT.child'` for
    a partially-restored nested dict) this function could not turn back
    into a widget value - most commonly because no widget exists for it
    any more (an R14-pinned fan-out key), the value can only ever come
    from an upload (a bio-prior network file), or the mapping genuinely
    isn't 1:1 (several ASREML fields are DERIVED from more than one
    widget - see `_asreml_build_cfg_from_widgets()`'s own docstring; this
    function restores only the fields that read back 1:1, exactly as the
    design record's own touch-points table specifies).

    `kind` ('sequential'/'step1'/'step2') decides `exec_mode`/
    `parallel_step` and which step-specific keys (PARALLEL, ASSEMBLE_MODE)
    are even looked for - it is never itself reported as unmapped.
    """
    if kind == 'tool':      # ver4-54 (R6): Tool configs have their own, much smaller inverse map
        return tool_config_to_gui_state(cfg)
    state = {}
    unmapped = []
    cfg = dict(cfg)  # local copy - this function never mutates the caller's dict

    def take(key, default=None):
        """Pop `key` out of the working copy (so leftover keys at the end
        are exactly what nothing below claimed) and return its value."""
        return cfg.pop(key, default)

    def simple(engine_key, widget_key, transform=None):
        if engine_key not in cfg:
            return
        value = take(engine_key)
        state[widget_key] = transform(value) if transform else value

    # --- Execution mode --------------------------------------------------
    state['exec_mode'] = 'Sequential' if kind == 'sequential' else 'Parallel'
    if kind in ('step1', 'step2'):
        state['parallel_step'] = 'Step 1' if kind == 'step1' else 'Step 2'

    # --- R_PATH / misc top-level scalars ---------------------------------
    if 'R_PATH' in cfg:
        r_path = take('R_PATH')
        state['r_path_none'] = r_path is None
        if r_path is not None:
            state['r_path'] = r_path
    simple('USE_GPU_SKLEARN', 'use_gpu_sklearn')
    if 'TORCH_DEVICE' in cfg:
        v = take('TORCH_DEVICE')
        state['torch_device'] = v if v else 'auto'
    # ver4-33: ACCELERATION sub-keys -> widget keys. A sub-key (or the whole
    # dict) that is ABSENT restores the LEGACY value, not the GUI's new-config
    # default, so loading an older config never silently turns acceleration on.
    _accel_cfg = take('ACCELERATION', None) if 'ACCELERATION' in cfg else None
    _accel_cfg = dict(_accel_cfg) if isinstance(_accel_cfg, dict) else {}   # local copy: popped below
    for _accel_key, _accel_widget in (
        ('plink_direct_read', 'accel_plink_direct_read'),
        ('preprocess_memo', 'accel_preprocess_memo'),
        ('gat_dense_batching', 'accel_gat_dense_batching'),
        ('gat_batched_ig', 'accel_gat_batched_ig'),
        ('gpu_device_round_robin', 'accel_gpu_device_round_robin'),
        # ver4-35: an absent key loads as its legacy value, like the ver4-33 ones.
        ('gat_factorised_input', 'accel_gat_factorised_input'),
        ('gat_sync_free', 'accel_gat_sync_free'),
        ('worker_thread_limits', 'accel_worker_thread_limits'),
        ('gpu_slots', 'accel_gpu_slots'),
        ('torch_device_policy', 'accel_torch_device_policy'),
        ('async_model_lane', 'accel_async_model_lane'),
        ('ld_cm_banded', 'accel_ld_cm_banded'),
        ('bio_prior_cost_guard', 'accel_bio_prior_cost_guard'),
        ('merge_shap_device', 'accel_merge_shap_device'),
        # ver4-43 (R1): absent -> 'legacy' (ACCELERATION_LEGACY_DEFAULTS).
        ('baseline_shap_engine', 'accel_baseline_shap_engine'),
        # ver4-61 (R2): absent -> 'auto' (the engine's own default for an absent key).
        ('trial_memory_guard', 'accel_trial_memory_guard'),
    ):
        _accel_value = _accel_cfg.get(_accel_key, ACCELERATION_LEGACY_DEFAULTS[_accel_key])
        if _accel_key in _ACCEL_SELECT_CHOICES:
            # The widget is a selectbox, whose value must be one of its options (a
            # number such as gpu_slots=2 is shown as the text '2').
            _accel_text = str(_accel_value).strip().lower()
            if _accel_text not in _ACCEL_SELECT_CHOICES[_accel_key]:
                unmapped.append(f'ACCELERATION.{_accel_key}={_accel_value!r}')
                _accel_text = str(ACCELERATION_LEGACY_DEFAULTS[_accel_key])
            _accel_value = _accel_text
        state[_accel_widget] = _accel_value
    # ver4-103: GAT stability keys. Absent -> the ENGINE value (seeded, single network, no
    # re-scoring), so an older config loads exactly as the engine would run it.
    _gs_seed = _accel_cfg.pop('GAT_SEED', GAT_STABILITY_ENGINE_DEFAULTS['GAT_SEED']) \
        if 'GAT_SEED' in _accel_cfg else GAT_STABILITY_ENGINE_DEFAULTS['GAT_SEED']
    _gs_off = _gs_seed is None or str(_gs_seed).strip().lower() in ('off', 'none', 'false', 'legacy', '')
    state['accel_gat_seed_fixed'] = not _gs_off
    if not _gs_off:
        try:
            state['accel_gat_seed'] = int(_gs_seed)
        except (TypeError, ValueError):
            unmapped.append(f'ACCELERATION.GAT_SEED={_gs_seed!r}')
    for _gs_key, _gs_widget in (('GAT_SEED_ENSEMBLE', 'accel_gat_seed_ensemble'),
                                ('GAT_RESCORE_TOP', 'accel_gat_rescore_top'),
                                ('GAT_RESCORE_SEEDS', 'accel_gat_rescore_seeds')):
        _gs_val = _accel_cfg.pop(_gs_key, GAT_STABILITY_ENGINE_DEFAULTS[_gs_key])
        try:
            state[_gs_widget] = int(_gs_val)
        except (TypeError, ValueError):
            unmapped.append(f'ACCELERATION.{_gs_key}={_gs_val!r}')
    for _engine_key in sorted(set(_accel_cfg) - set(ACCELERATION_LEGACY_DEFAULTS)):
        unmapped.append(f'ACCELERATION.{_engine_key}')
    # ver4-56 (R1/R3): telemetry switch and the advisor block. A config without
    # RESOURCE_TELEMETRY keeps the engine default (on); a config without RESOURCE_ADVISOR
    # was saved before the calibrated advisor existed, so it loads on the legacy engine
    # (new GUI sessions start on 'calibrated'). The stored 'prediction' is an audit record
    # and is recomputed by the panel, so it is consumed here and not turned into a widget.
    simple('RESOURCE_TELEMETRY', 'run_resource_telemetry', bool)
    simple('RESOURCE_TELEMETRY_INTERVAL_S', 'resource_telemetry_interval_s', float)
    if 'RESOURCE_ADVISOR' in cfg:
        _advisor_cfg = take('RESOURCE_ADVISOR', None)
        _advisor_cfg = _advisor_cfg if isinstance(_advisor_cfg, dict) else {}
        _advisor_engine = str(_advisor_cfg.get('engine', 'calibrated')).strip().lower()
        if _advisor_engine not in ('calibrated', 'legacy'):
            unmapped.append(f"RESOURCE_ADVISOR.engine={_advisor_cfg.get('engine')!r}")
            _advisor_engine = 'calibrated'
        state['resource_advisor_engine'] = _advisor_engine
        _advisor_objective = str(_advisor_cfg.get('objective', 'cost')).strip().lower()
        if _advisor_objective not in ('cost', 'turnaround'):
            unmapped.append(f"RESOURCE_ADVISOR.objective={_advisor_cfg.get('objective')!r}")
            _advisor_objective = 'cost'
        state['resource_advisor_objective'] = _advisor_objective
        for _advisor_key, _advisor_widget, _advisor_cast in (
                ('target_walltime_h', 'resource_advisor_target_h', float),
                ('walltime_quantile', 'resource_advisor_walltime_q', float),
                ('mem_quantile', 'resource_advisor_mem_q', float)):
            if _advisor_key in _advisor_cfg:
                try:
                    state[_advisor_widget] = _advisor_cast(_advisor_cfg[_advisor_key])
                except (TypeError, ValueError):
                    unmapped.append(f'RESOURCE_ADVISOR.{_advisor_key}={_advisor_cfg[_advisor_key]!r}')
        _advisor_sources = _advisor_cfg.get('calibration_sources')
        if isinstance(_advisor_sources, (list, tuple)):
            state['resource_advisor_sources'] = '\n'.join(str(x) for x in _advisor_sources)
        for _advisor_extra in sorted(set(_advisor_cfg) - {
                'engine', 'objective', 'target_walltime_h', 'walltime_quantile',
                'mem_quantile', 'calibration_sources', 'prediction'}):
            unmapped.append(f'RESOURCE_ADVISOR.{_advisor_extra}')
    else:
        state['resource_advisor_engine'] = 'legacy'
    # ver4-99 (R2, I11): a saved config that never mentioned the stability summary must keep
    # producing the files it always did, so its absent key switches the checkbox OFF (it is off in
    # a new configuration too, since ver4-100) instead of leaving a previous tick in place.
    if kind in ('sequential', 'step1', 'step2') and 'STABILITY_SUMMARY_CREATE' not in cfg:
        state['stability_summary_create'] = False
    if kind in ('sequential', 'step1', 'step2') and 'STABILITY_INTERACTION_FREQUENCY' not in cfg:
        state['stability_interaction_frequency'] = False         # X03, I11: absent key = off
    for engine_key, widget_key in (
        ('CUDNN_BENCHMARK', 'cudnn_benchmark'), ('USE_AMP', 'use_amp'),
        ('GPU_EVAL_BATCH', 'gpu_eval_batch'), ('GPU_LD_R2', 'gpu_ld_r2'),
        ('GPU_KERNEL_PRECOMPUTE', 'gpu_kernel_precompute'),
        ('HP_TUNE_PARALLEL_TRIALS', 'hp_tune_parallel_trials'),
        ('HP_TUNE_BAYES_BATCH', 'hp_tune_bayes_batch'),
        ('HP_TUNE_BAYES_BATCH_MAX', 'hp_tune_bayes_batch_max'),
        ('HP_TUNE_BAYES_LIAR', 'hp_tune_bayes_liar'),
        ('HP_TUNE_PARALLEL_RESTARTS', 'hp_tune_parallel_restarts'),
        ('HP_TUNE_BAYES_DOMAIN_REDUCTION', 'hp_tune_bayes_domain_reduction'),
        ('HP_TUNE_BAYES_MULTI_FIDELITY', 'hp_tune_bayes_multi_fidelity'),
        ('HP_TUNE_WARM_START', 'hp_tune_warm_start'),
        ('HP_TUNE_SELECTION_MARGIN', 'hp_tune_selection_margin'),
        ('HP_TUNE_VALID_REPEATS', 'hp_tune_valid_repeats'),
        ('HP_TUNE_SCOPE', 'hp_tune_scope'),
        ('W_OPT_ANALYTIC_SEED', 'w_opt_analytic_seed'),
        ('W_OPT_VALIDATION_FLOOR', 'w_opt_validation_floor'),
        ('W_OPT_SIMPLEX_SEARCH', 'w_opt_simplex_search'),
        ('MODEL_AVAILABILITY_STRICT', 'model_availability_strict'),
        ('QTL_WINDOW', 'qtl_window'), ('QTL_WINDOW_MODE', 'qtl_window_mode'),
        ('PLOT_EFFECT_FLOAT32', 'plot_effect_float32'),
        ('PLOT_EXTRA_FORMATS', 'plot_extra_formats'),   # ver4-31-VIS, R15
        ('MODEL_LABEL_STYLE', 'model_label_style'),     # ver4-95
        ('METRIC_PLOT_DPI', 'metric_plot_dpi'), ('SCATTER_PLOT_DPI', 'scatter_plot_dpi'),
        ('CIRCOS_PLOT_DPI', 'circos_plot_dpi'),
        ('SCENARIO', 'scenario'), ('RESULT_NAME', 'result_name'),
        ('RESULT_COMPRESSION', 'result_compression'),
        ('ASREML_STRICT', 'asreml_strict'),
        ('ITER_NUM', 'iter_num'), ('MIN_DATA_POINTS', 'min_data_points'),
        ('OTHER_MODELS_MARKER_SOURCE', 'other_models_marker_source'),
        ('HP_TUNE_ENSEMBLE_MODE', 'hp_tune_ensemble_mode'),
        ('METRIC_PLOT_CREATE', 'metric_plot_create'),
        ('METRIC_SUMMARY_CREATE', 'metric_summary_create'),
        ('METRIC_SUMMARY_DISPERSION', 'metric_summary_dispersion'),   # ver4-100 (O02)
        ('DPT_SUMMARY_CREATE', 'dpt_summary_create'),
        ('STABILITY_SUMMARY_CREATE', 'stability_summary_create'),   # ver4-99 (R2)
        ('STABILITY_TOP_K', 'stability_top_k'),                     # ver4-99 (R2)
        ('STABILITY_INTERACTION_FREQUENCY', 'stability_interaction_frequency'),   # X03
        ('WEIGHT_PLOT_CREATE', 'weight_plot_create'),
        ('WEIGHT_PLOT_DPI', 'weight_plot_dpi'),
        ('SCATTER_CREATE', 'scatter_create'), ('CIRCOS_CREATE', 'circos_create'),
        # Requirements.md item 4
        ('SCATTER_COLOR_BY_POPULATION', 'scatter_color_by_population'),
        # Requirements.md item 5 (this session)
        ('SCATTER_NORMALIZE_MARKER_EFFECT', 'scatter_normalize_marker_effect'),
        ('CHROMOSOME_INFO', 'chrom_info_path'), ('MARKER_INFO', 'marker_info_path'),
        ('END_ADJUST', 'end_adjust'), ('GENE_ADJUST', 'gene_adjust'), ('WINDOW', 'window_size'),
        ('CIRCOS_BROADCAST_POPULATION', 'circos_broadcast_population'),
        ('CIRCOS_HIGHER_ORDER_LAYOUT', 'circos_higher_order_layout'),
        ('GENOTYPE_PLINK_PATH', 'genotype_plink_path'),
        ('PHENOTYPE_FILE_NAME', 'phenotype_path'),
    ):
        simple(engine_key, widget_key)
    # ver4-40, R1: HP_TUNE_FAST ({sub_key: value} dict, or None = legacy single-model tuning).
    # An ABSENT or None key restores the master checkbox OFF, so loading an older config never
    # silently turns the fast mode on. The widget keys are duplicated from main_app.py's
    # HP_TUNE_FAST_WIDGETS (I2: this file never imports main_app.py).
    _hp_fast_cfg = take('HP_TUNE_FAST', None) if 'HP_TUNE_FAST' in cfg else None
    state['hp_tune_fast'] = isinstance(_hp_fast_cfg, dict)
    if isinstance(_hp_fast_cfg, dict):
        _hp_fast_widgets = {
            'freeze_fidelity': 'hp_tune_fast_freeze_fidelity',
            'search_space': 'hp_tune_fast_search_space',
            'test_stub': 'hp_tune_fast_test_stub',
            'stub_parity_min': 'hp_tune_fast_stub_parity_min',
            'search_engine': 'hp_tune_fast_search_engine',
            'engine_parity_min': 'hp_tune_fast_engine_parity_min',
            'patience': 'hp_tune_fast_patience',
            'auto_budget': 'hp_tune_fast_auto_budget',
            'evals_per_dim': 'hp_tune_fast_evals_per_dim',
            'persistent_executor': 'hp_tune_fast_persistent_executor',
            'gat_lean_fit': 'hp_tune_fast_gat_lean_fit',
            'cap_search_cost': 'hp_tune_fast_cap_search_cost',
            'async_batch': 'hp_tune_fast_async_batch',
            'ship_once': 'hp_tune_fast_ship_once',
            'gat_search_early_stop': 'hp_tune_fast_gat_search_early_stop',   # ver4-61 (R4)
        }
        # a sub-key the dict does not name loads as its legacy value (never as a leftover
        # widget value from an earlier session)
        for _fast_key, _legacy in (('freeze_fidelity', False), ('search_space', 'all'),
                                   ('test_stub', False), ('stub_parity_min', 0.995),
                                   ('search_engine', 'bglr'), ('engine_parity_min', 0.98),
                                   ('patience', 0), ('auto_budget', False),
                                   ('evals_per_dim', 6), ('persistent_executor', False),
                                   ('gat_lean_fit', False), ('cap_search_cost', False),
                                   ('async_batch', False), ('ship_once', False),
                                   ('gat_search_early_stop', 0)):
            state[_hp_fast_widgets[_fast_key]] = _legacy
        for _fast_key, _fast_value in _hp_fast_cfg.items():
            if _fast_key in _hp_fast_widgets:
                state[_hp_fast_widgets[_fast_key]] = _fast_value
            else:
                unmapped.append(f'HP_TUNE_FAST.{_fast_key}')
    # ver4-61 (R6): the pilot hyperparameter file; absent / None loads as an empty text box.
    if 'HP_TUNE_PRESET_FILE' in cfg:
        state['hp_tune_preset_file'] = str(take('HP_TUNE_PRESET_FILE') or '')
    if 'GENE_INFO' in cfg:
        v = take('GENE_INFO')
        state['gene_info_path'] = v or ''
    if 'QTL' in cfg:
        v = take('QTL')
        state['qtl_path'] = v or ''

    # --- Phenotype targets / scenario -------------------------------------
    if 'PHENOTYPE' in cfg:
        v = take('PHENOTYPE')
        state['phenotype_targets'] = 'all' if v == 'all' else ', '.join(v)

    # --- Phenotype file mode / population source --------------------------
    if 'PHENOTYPE_FILE_MODE' in cfg:
        mode_val = take('PHENOTYPE_FILE_MODE')
        state['phenotype_file_mode_radio'] = (
            'Raw (plot-level - multiple rows per individual)' if mode_val == 'raw'
            else 'Analysed (one row per individual)'
        )
        trait_cols = take('PHENOTYPE_TRAIT_COLUMNS', None)
        if mode_val == 'raw' and trait_cols:
            state['phenotype_trait_columns'] = list(trait_cols)
    elif 'PHENOTYPE_TRAIT_COLUMNS' in cfg:
        take('PHENOTYPE_TRAIT_COLUMNS')

    if 'POPULATION_SOURCE' in cfg:
        pop_source = take('POPULATION_SOURCE')
        pop_mode = pop_source.get('mode') if isinstance(pop_source, dict) else None
        if pop_mode == 'constant':
            state['population_source_mode'] = 'One constant label (no natural grouping)'
            state['population_source_label'] = pop_source.get('label', 'all_lines')
        elif pop_mode == 'derived':
            state['population_source_mode'] = 'Derived from several column(s)'
            state['population_source_columns'] = list(pop_source.get('columns', []))
        # mode == 'column' (Analysed mode's own fixed default) needs no
        # widget at all - there is nothing to restore.

    # --- Genotype file --------------------------------------------------
    if 'GENOTYPE_FORMAT' in cfg:
        fmt = take('GENOTYPE_FORMAT')
        file_name = take('GENOTYPE_FILE_NAME', '')
        state['genotype_format'] = 'CSV file' if fmt == 'csv' else 'PLINK binary fileset (.bed/.bim/.fam)'
        if fmt == 'csv':
            state['genotype_path'] = file_name
        else:
            state['genotype_plink_stem'] = file_name

    # --- Models -----------------------------------------------------------
    if 'MODEL' in cfg:
        models = take('MODEL')
        for m in models:
            state[f'model_selected_{_base_model_name(m)}'] = True
        if any(m != _base_model_name(m) for m in models):
            unmapped.append(
                'MODEL (repeat GAT_biological_prior_knowledge network instances - the '
                'checkbox itself is restored, but each instance\'s own network/gene-location '
                'file must be re-selected on the Biological Prior Network tab)'
            )

    if 'RATIO' in cfg:
        # A config read from disk has already been through json.dump/
        # json.load once (every caller of engine_config_to_gui_state()
        # reads a real *_config.json file - see this module's own
        # docstring), which silently flattens a tuple-valued 3-way-split
        # entry (e.g. (0.7, 0.15, 0.15)) into a plain list - the exact
        # problem pipeline_utils.restore_ratio() exists to undo before
        # GP() itself ever sees such a value (see that function's own
        # docstring). Applying it here FIRST, before building ratio_text,
        # means a config with a tuple-shaped ratio round-trips back to the
        # GUI as the same tuple shape (rendered with parentheses) rather
        # than silently degrading into a same-looking-but-differently-
        # TYPED list entry - gather_config()'s own ast.literal_eval only
        # ever recovers a tuple from text that actually has parentheses.
        # ('SCENARIO' was already popped into state['scenario'] by the
        # `simple()` pass above, at the top of this function - read it
        # back from there rather than from `cfg`, which no longer has it.
        # pipeline_utils.restore_ratio()'s own `scenario` argument is
        # purely for API symmetry and never actually changes its output
        # - see that function's own docstring - so this is correct
        # regardless of whether 'SCENARIO' was present in `cfg` at all.)
        raw_ratio = _restore_ratio_shape(take('RATIO'), state.get('scenario'))
        state['ratio_text'] = restore_ratio_text(raw_ratio)

    # --- Between-prediction validation set (Requirements.md patch) --------
    # Not routed through `simple()` above: every one of these four keys is
    # ALWAYS present (as None/[]) in a config gathered from a 'within' run
    # (see gather_config()'s own else-branch), and blindly restoring a
    # None into a checkbox/radio's own session_state key risks Streamlit
    # rejecting that widget's default value on the next run. Only ever
    # restored when the loaded value is truthy, exactly like GENE_INFO/QTL
    # above handle their own "usually absent" case.
    if 'BETWEEN_VALIDATION' in cfg:
        _between_validation_loaded = take('BETWEEN_VALIDATION')
        if _between_validation_loaded is not None:
            state['between_use_validation'] = bool(_between_validation_loaded)
    if 'BETWEEN_VALIDATION_SOURCE' in cfg:
        _between_source_loaded = take('BETWEEN_VALIDATION_SOURCE')
        if _between_source_loaded:
            state['between_validation_source'] = _between_source_loaded
    if 'BETWEEN_RATIO' in cfg:
        _between_ratio_loaded = take('BETWEEN_RATIO')
        if _between_ratio_loaded:
            state['between_ratio_text'] = ', '.join(str(x) for x in _between_ratio_loaded)
    if 'EXCLUDED_POPULATION_COMBINATIONS' in cfg:
        _excluded_combos_loaded = take('EXCLUDED_POPULATION_COMBINATIONS')
        if _excluded_combos_loaded:
            state['between_excluded_combinations'] = list(_excluded_combos_loaded)

    # --- Per-model hyperparameters -----------------------------------------
    if 'HPARAMETERS' in cfg:
        hparameters = take('HPARAMETERS')
        for model, values in hparameters.items():
            base_model = _base_model_name(model)
            spec = HPARAM_SPECS.get(base_model)
            if spec is None or not isinstance(values, list):
                unmapped.append(f'HPARAMETERS.{model}')
                continue
            prefix = f'hp_{base_model}'
            for idx, field_spec in enumerate(spec):
                if idx >= len(values):
                    # ver4-38 (I11): a saved list that predates an appended
                    # field must reload with that field's LEGACY value, not
                    # the spec default the widget would otherwise show.
                    # Only matters where the two differ (SVR field 26,
                    # 'Scale epsilon and C by the trait SD': legacy False,
                    # spec default True); every other trailing field
                    # keeps its widget default exactly as before.
                    if ('legacy_default' in field_spec
                            and field_spec['legacy_default'] != field_spec.get('default')):
                        _restore_hparam_field(state, f'{prefix}_{idx}', field_spec,
                                              field_spec['legacy_default'])
                    continue
                _restore_hparam_field(state, f'{prefix}_{idx}', field_spec, values[idx])

    # --- Hyperparameter tuning ----------------------------------------------
    if 'HP_TUNE' in cfg:
        hp_tune = take('HP_TUNE') or {}
        for model, tune_cfg in hp_tune.items():
            base_model = _base_model_name(model)
            prefix = f'hp_tune_{base_model}'
            if not isinstance(tune_cfg, dict):
                unmapped.append(f'HP_TUNE.{model}')
                continue
            state[f'{prefix}_enabled'] = True
            algorithms = list(tune_cfg.get('algorithms', []))
            state[f'{prefix}_algorithms'] = algorithms
            budget = tune_cfg.get('budget', {})
            for algo in algorithms:
                b = budget.get(algo, {})
                if algo == 'Grid':
                    if 'n_points' in b:
                        state[f'{prefix}_budget_{algo}_n_points'] = b['n_points']
                else:
                    if 'n_iter' in b:
                        state[f'{prefix}_budget_{algo}_n_iter'] = b['n_iter']
                    if algo == 'Bayesian' and 'init_points' in b:
                        state[f'{prefix}_budget_{algo}_init_points'] = b['init_points']
            if 'reduced_cost_search' in tune_cfg:
                state[f'{prefix}_reduced_cost'] = tune_cfg['reduced_cost_search']
            # ver4-61 (R3): per-model wall-clock limits in minutes; an absent key loads as 0
            # (= off), never as a leftover widget value from an earlier session.
            for _limit_key in ('time_limit_min', 'trial_time_limit_min'):
                try:
                    state[f'{prefix}_{_limit_key}'] = float(tune_cfg.get(_limit_key) or 0.0)
                except (TypeError, ValueError):
                    state[f'{prefix}_{_limit_key}'] = 0.0
                    unmapped.append(f'HP_TUNE.{model}.{_limit_key}={tune_cfg.get(_limit_key)!r}')

    # --- LD pruning ---------------------------------------------------------
    if 'LD_PRUNE' in cfg:
        ld_prune = take('LD_PRUNE')
        if ld_prune is None:
            state['ld_prune_enabled'] = False
        else:
            state['ld_prune_enabled'] = True
            state['ld_snp_info_path'] = ld_prune.get('snp_info') or ''
            state['ld_window'] = ld_prune.get('window', 50.0)
            state['ld_window_unit'] = ld_prune.get('window_unit', 'kb')
            state['ld_step'] = ld_prune.get('step', 5)
            state['ld_r2_threshold'] = ld_prune.get('r2_threshold', 0.2)
            state['ld_plink_path'] = ld_prune.get('plink_path', 'plink2')
            state['ld_allow_extra_chr'] = ld_prune.get('allow_extra_chr', False)
            chr_set = ld_prune.get('chr_set')
            state['ld_chr_set_enabled'] = chr_set is not None
            if chr_set is not None:
                state['ld_chr_set'] = chr_set
            work_dir = ld_prune.get('work_dir')
            state['ld_work_dir_enabled'] = bool(work_dir)
            if work_dir:
                state['ld_work_dir'] = work_dir
            state['ld_keep_intermediate'] = ld_prune.get('keep_intermediate', False)
            state['ld_round_dosage'] = ld_prune.get('round_dosage', True)
            state['ld_unmapped_strategy'] = ld_prune.get('unmapped_strategy', 'variant_count')
            maf = ld_prune.get('maf_threshold', 0.0)
            state['ld_maf_enabled'] = bool(maf)
            if maf:
                state['ld_maf_threshold'] = maf
            if 'unmapped_window' in ld_prune:
                state['ld_unmapped_window'] = ld_prune['unmapped_window']
            if 'unmapped_step' in ld_prune:
                state['ld_unmapped_step'] = ld_prune['unmapped_step']
            decay_plot = ld_prune.get('decay_plot')
            if isinstance(decay_plot, dict) and decay_plot.get('enabled'):
                state['ld_decay_plot_enabled'] = True
                state['ld_decay_plot_frequency'] = decay_plot.get('frequency', 1)
                state['ld_decay_max_pairs_per_chr'] = decay_plot.get('max_pairs_per_chr', 20000)
                state['ld_decay_keep_log'] = decay_plot.get('keep_log', True)
                for unit in decay_plot.get('window_units', []):
                    per_unit = decay_plot.get(unit, {})
                    suffix = _LD_UNIT_TO_STATE_SUFFIX.get(unit)
                    if suffix and 'max_distance' in per_unit:
                        state[f'ld_decay_max_distance_{suffix}'] = per_unit['max_distance']
                    if suffix and 'bin_width' in per_unit:
                        state[f'ld_decay_bin_width_{suffix}'] = per_unit['bin_width']
            elif isinstance(decay_plot, dict):
                state['ld_decay_plot_enabled'] = False

    # --- RF marker-importance filtering --------------------------------------
    if 'RF_FILTER' in cfg:
        rf_filter = take('RF_FILTER')
        if rf_filter is None:
            state['rf_filter_enabled'] = False
        else:
            state['rf_filter_enabled'] = True
            state['rf_filter_n_estimators'] = rf_filter.get('n_estimators', 200)
            max_depth = rf_filter.get('max_depth')
            state['rf_filter_max_depth_enabled'] = max_depth is not None
            if max_depth is not None:
                state['rf_filter_max_depth'] = max_depth
            state['rf_filter_max_features'] = rf_filter.get('max_features') or 'all'
            state['rf_filter_min_samples_leaf'] = rf_filter.get('min_samples_leaf', 1)
            state['rf_filter_random_state'] = rf_filter.get('random_state', 0)
            if rf_filter.get('mode') == 'count':
                state['rf_filter_mode'] = 'Fixed number of markers (M)'
                state['rf_filter_top_n'] = rf_filter.get('top_n', 100)
            else:
                state['rf_filter_mode'] = 'Percentage of markers'
                state['rf_filter_top_percent'] = rf_filter.get('top_ratio', 0.2) * 100.0

    # --- Ensemble weighting ---------------------------------------------------
    if 'W_OPT' in cfg:
        selected = take('W_OPT') or []
        for m in _WOPT_METHODS:
            state[f'wopt_selected_{m}'] = m in selected

    if 'HYPERPARAMETERS_OPT' in cfg:
        hp_opt = take('HYPERPARAMETERS_OPT') or {}
        # main_app.py's own HYPERPARAM_OPT_SPECS (the four W_OPT methods'
        # own hyperparameter fields) is never imported here (I2 - it lives
        # in main_app.py itself, a Streamlit module this file must never
        # import). Unlike HPARAM_SPECS (per-model fields, some of them
        # composite mode+custom-number widgets), every field across all
        # four HYPERPARAM_OPT_SPECS methods is a plain scalar type
        # ('bool'/'int'/'float'/'str') - confirmed by grepping main_app.py
        # for any of the five composite field types among its own field
        # definitions: none appear - so resolve_wopt()'s own per-field
        # read-back (`resolve_field(key, field)`) always takes
        # resolve_field()'s first, simplest branch
        # (`st.session_state.get(key, default)`), and the exact inverse is
        # simply writing each positional value straight to
        # `wopt_<method>_<idx>` - no field-type table needed at all.
        for method, values in hp_opt.items():
            if not isinstance(values, list):
                unmapped.append(f'HYPERPARAMETERS_OPT.{method}')
                continue
            prefix = f"wopt_{method.replace(' ', '_')}"
            for idx, value in enumerate(values):
                state[f'{prefix}_{idx}'] = value

    # ver4-26, R1/R4: joint hyperparameter+ensemble-weight optimisation -
    # restores Tab 4's own checkbox and, if it was on, the joint search's
    # budget panel fields, from HYPERPARAMETERS_JOINT_OPT's own
    # {opt_key: value} dict (see _JOINT_OPT_KEY_TO_IDX's own comment for
    # why this is a name-keyed, not position-keyed, restoration unlike
    # HYPERPARAMETERS_OPT just above).
    if 'JOINT_W_OPT' in cfg:
        _joint_selected = take('JOINT_W_OPT') or []
        state['joint_w_opt_enabled'] = 'Joint Bayesian optimisation' in (_joint_selected or [])

    # ver4-34, R1.b: a config that ran the joint search (its own settings dict, or just the
    # JOINT_W_OPT switch) ran it with the engine's legacy value for every key it lacks, so
    # the widgets must show that legacy value - not the faster new-session default - or
    # re-running the loaded config from the GUI would silently change its behaviour (I11).
    _joint_config_present = (
        'HYPERPARAMETERS_JOINT_OPT' in cfg
        or bool(state.get('joint_w_opt_enabled', False)))
    if _joint_config_present:
        for opt_key, legacy_value in _JOINT_OPT_ABSENT_LEGACY_VALUES.items():
            state[f'joint_wopt_{_JOINT_OPT_KEY_TO_IDX[opt_key]}'] = legacy_value

    if 'HYPERPARAMETERS_JOINT_OPT' in cfg:
        _joint_opt = take('HYPERPARAMETERS_JOINT_OPT') or {}
        if isinstance(_joint_opt, dict):
            for opt_key, value in _joint_opt.items():
                idx = _JOINT_OPT_KEY_TO_IDX.get(opt_key)
                if idx is None:
                    unmapped.append(f'HYPERPARAMETERS_JOINT_OPT.{opt_key}')
                    continue
                state[f'joint_wopt_{idx}'] = value
        else:
            unmapped.append('HYPERPARAMETERS_JOINT_OPT (not a dict)')

    # ver4-57 (R3): the opt-in Iterative ensemble. None -> checkbox unticked; a dict ->
    # ticked, with every field restored BY NAME (`iter_ens_<key>`); a key the dict lacks loads
    # as the pipeline default, which is what the engine used for it (I11); an unknown sub-key
    # is reported as unmapped; the key absent from the config leaves the widgets untouched.
    if 'ITERATIVE_ENSEMBLE' in cfg:
        _iter_cfg = take('ITERATIVE_ENSEMBLE')
        if _iter_cfg is None:
            state['iter_ens_enabled'] = False
        elif isinstance(_iter_cfg, dict):
            state['iter_ens_enabled'] = True
            for _iter_key, _iter_default in _ITER_ENS_DEFAULTS.items():
                state[f'iter_ens_{_iter_key}'] = _iter_cfg.get(_iter_key, _iter_default)
            for _iter_key in _iter_cfg:
                if _iter_key not in _ITER_ENS_DEFAULTS:
                    unmapped.append(f'ITERATIVE_ENSEMBLE.{_iter_key}')
        else:
            state['iter_ens_enabled'] = False
            unmapped.append('ITERATIVE_ENSEMBLE (not a dict)')

    # The GUI never combines joint optimisation with the Iterative ensemble (the two Tab 4
    # checkboxes lock each other). A configuration that has both loads with the joint search
    # on and the Iterative ensemble off, and says so.
    if state.get('joint_w_opt_enabled') and state.get('iter_ens_enabled'):
        state['iter_ens_enabled'] = False
        unmapped.append('ITERATIVE_ENSEMBLE (not loaded: the GUI cannot combine it with joint '
                        'hyperparameter + ensemble-weight optimisation)')

    # ver4-77 (R2): the opt-in Super learner, restored exactly like the iterative block above:
    # None -> checkbox unticked; a dict -> ticked, with every field restored BY NAME
    # (`sl_<key>`); a key the dict lacks loads as the pipeline default, which is what the engine
    # used for it (I11); an unknown sub-key is reported as unmapped; the key absent from the
    # config (every older configuration) leaves the widgets untouched, i.e. unticked.
    if 'SUPER_LEARNER' in cfg:
        _sl_cfg = take('SUPER_LEARNER')
        if _sl_cfg is None:
            state['sl_enabled'] = False
        elif isinstance(_sl_cfg, dict):
            state['sl_enabled'] = True
            for _sl_key, _sl_default in _SL_DEFAULTS.items():
                _sl_value = _sl_cfg.get(_sl_key, _sl_default)
                state[f'sl_{_sl_key}'] = list(_sl_value) if isinstance(_sl_value, list) else _sl_value
            # ver4-81 (R1): the weight-optimisation methods the Super learner took over from W_OPT
            # (the saved config has W_OPT = None) are ticked again, so saving the GUI state
            # again reproduces the same configuration.
            _sl_taken = _sl_cfg.get('weight_optimisers') or []
            if isinstance(_sl_taken, (list, tuple)):
                for _m in _WOPT_METHODS:
                    if _m in _sl_taken:
                        state[f'wopt_selected_{_m}'] = True
            for _sl_key in _sl_cfg:
                if _sl_key not in _SL_DEFAULTS:
                    unmapped.append(f'SUPER_LEARNER.{_sl_key}')
        else:
            state['sl_enabled'] = False
            unmapped.append('SUPER_LEARNER (not a dict)')

    # --- Plot config dicts -----------------------------------------------
    if 'WEIGHT_PLOT_CONFIG' in cfg:
        wpc = take('WEIGHT_PLOT_CONFIG') or {}
        state['weight_plot_font'] = wpc.get('font_size', 1)
        state['weight_plot_fig'] = wpc.get('fig_size', 5)
    if 'SCATTER_CONFIG' in cfg:
        sc = take('SCATTER_CONFIG') or {}
        state['scatter_font'] = sc.get('font_size', 2)
        state['scatter_fig'] = sc.get('fig_size', 30)
        # ver4-31-VIS, R4-R7: a key the config lacks means the LEGACY value
        # (engine `.get(key, <legacy>)`), so re-running a loaded older config
        # from the GUI reproduces its original render (I11).
        state['scatter_max_points'] = int(sc.get('max_points') or 0)
        state['scatter_density'] = sc.get('density', 'off')
        state['scatter_effect_scale'] = sc.get('effect_scale', 'max')
        state['scatter_size_mode'] = sc.get('size_mode', 'total')
        state['scatter_outer_labels_only'] = bool(sc.get('outer_labels_only', False))
        state['scatter_diag'] = sc.get('diag', 'name')
        state['scatter_pheno_hue'] = sc.get('pheno_hue', 'level')
    if 'METRIC_PLOT_CONFIG' in cfg:
        mc = take('METRIC_PLOT_CONFIG') or {}
        state['metric_font'] = mc.get('font_size', 1)
        state['metric_fig'] = mc.get('fig_size', 5)
        # ver4-31-VIS, R1-R3: absent key -> legacy value (see the scatter note).
        state['metric_style'] = mc.get('style', 'legacy')
        state['metric_min_n_violin'] = int(mc.get('min_n_violin', 5) or 5)
        state['metric_show_mean_se'] = bool(mc.get('show_mean_se', False))
        state['metric_show_median'] = bool(mc.get('show_median', False))
        state['metric_colour_by'] = mc.get('colour_by', 'model')
        state['metric_auto_width'] = bool(mc.get('auto_width', False))
        state['metric_group_separators'] = bool(mc.get('group_separators', False))
        state['metric_reference'] = 'ensemble' if mc.get('reference') == 'ensemble' else 'none'
        state['metric_legend'] = mc.get('legend', 'on')
        state['metric_between_label'] = mc.get('between_label', 'test')
        state['metric_cut'] = 2 if float(mc.get('cut', 0) or 0) > 0 else 0
    if 'ACCURACY_METRICS_PLOT_CONFIG' in cfg:
        amc = take('ACCURACY_METRICS_PLOT_CONFIG') or {}
        # Requirements.md items 4-7 (multi-option update): six independent
        # subsections, each with its own create flag. `pairwise` is the
        # pre-multi-option name of `pairwise_metrics`.
        _legacy_master_on = bool(take('ACCURACY_METRICS_PLOT_CREATE')) if 'ACCURACY_METRICS_PLOT_CREATE' in cfg else False
        for name in ('violin_accuracy', 'violin_spearman', 'coincidence', 'czekanowski',
                     'pairwise_metrics', 'across_options', 'pairwise'):
            sub = amc.get(name)
            widget_name = 'pairwise_metrics' if name == 'pairwise' else name
            if name == 'pairwise' and 'pairwise_metrics' in amc:
                continue
            if not isinstance(sub, dict):
                continue
            if 'create' in sub:
                state[f'accuracy_metrics_plot_{widget_name}_create'] = bool(sub['create'])
            elif widget_name in ('violin_accuracy', 'violin_spearman', 'coincidence',
                                 'pairwise_metrics'):
                # A config saved before the split: the old master flag
                # enabled these four families together.
                state[f'accuracy_metrics_plot_{widget_name}_create'] = _legacy_master_on
            if 'font_size' in sub:
                state[f'accuracy_metrics_plot_{widget_name}_font'] = sub['font_size']
            if 'fig_size' in sub:
                state[f'accuracy_metrics_plot_{widget_name}_fig'] = sub['fig_size']
            if 'dpi' in sub:
                state[f'accuracy_metrics_plot_{widget_name}_dpi'] = sub['dpi']
            if name == 'coincidence':
                # ver4-31-VIS, R8/R9: absent key -> legacy value.
                state['coincidence_baseline'] = bool(sub.get('baseline', False))
                state['coincidence_y'] = sub.get('y', 'percentage')
                state['coincidence_ci'] = sub.get('ci', 'z95')
                state['coincidence_band'] = sub.get('band', 'fill')
                state['coincidence_highlight_top'] = int(sub.get('highlight_top', 0) or 0)
                state['coincidence_legend_sort'] = bool(sub.get('legend_sort', False))
                state['coincidence_write_summary'] = bool(sub.get('write_summary', False))
            if name == 'czekanowski' and sub.get('thresholds'):
                state['accuracy_metrics_plot_czekanowski_thresholds'] = [
                    int(t) for t in sub['thresholds']
                ]
    if 'METRIC_DIRECTIONS' in cfg:
        # Requirements.md item 4: direction widgets now live on Tab 1.
        _dirs = take('METRIC_DIRECTIONS') or {}
        if isinstance(_dirs, dict):
            state['asreml_accuracy_direction_phenotypes'] = list(_dirs.keys())
            for _ph, _val in _dirs.items():
                if _val in ('higher', 'lower'):
                    state[f'asreml_accuracy_direction_value__{_ph}'] = _val
    if 'ACCURACY_BUBBLE_PLOT_CREATE' in cfg:
        # ver4-25, R6: no widget exists for this any more (see
        # gather_config()'s own comment) - nothing to restore into.
        take('ACCURACY_BUBBLE_PLOT_CREATE')
        unmapped.append('ACCURACY_BUBBLE_PLOT_CREATE (widget removed, R6 - always False now)')

    # --- Circos config ------------------------------------------------------
    if 'CIRCOS_CONFIG' in cfg:
        cc = take('CIRCOS_CONFIG') or {}
        _circos_map = {
            'space': 'circos_space', 'start': 'circos_start', 'end': 'circos_end',
            'link_alpha_min': 'circos_link_alpha_min', 'interaction_top': 'circos_topinteraction',
            'interaction_top_mode': 'circos_topinteraction_mode',
            'interaction_top_count': 'circos_topinteraction_count',
            'label_size': 'circos_labelsize', 'scale': 'circos_scale', 'unit': 'circos_unit',
            'ring_label_size': 'circos_ring_label_size', 'ring_label_fit': 'circos_ring_label_fit',
            'ring_layout': 'circos_ring_layout', 'figsize': 'circos_figsize',
            'seam_gap_max': 'circos_seam_gap_max', 'ring_label_max_chars': 'circos_ring_label_max_chars',
        }
        for engine_field, widget_key in _circos_map.items():
            if engine_field in cc and cc[engine_field] is not None:
                state[widget_key] = cc[engine_field]
        # ver4-31-VIS, R11-R14: absent key -> the engine's legacy value, so a
        # loaded older config re-runs unchanged from the GUI (I11).
        state['circos_link_sort'] = cc.get('link_sort', 'none')
        state['circos_link_scale'] = cc.get('link_scale', 'minmax')
        state['circos_link_colour_mode'] = cc.get('link_colour_mode', 'chrom')
        _lw = cc.get('link_lw_range')
        if _lw and len(_lw) == 2:
            state['circos_link_lw_min'], state['circos_link_lw_max'] = float(_lw[0]), float(_lw[1])
        state['circos_consensus_ring'] = bool(cc.get('consensus_ring', False))
        state['circos_consensus_level'] = int(cc.get('consensus_level', 9))
        state['circos_consensus_scope'] = cc.get('consensus_scope', 'base_models')
        state['circos_label_top_markers'] = int(cc.get('label_top_markers', 0) or 0)
        state['circos_label_source'] = cc.get('label_source', 'ensemble')
        state['circos_label_min_sep_deg'] = float(cc.get('label_min_sep_deg', 2.0))
        state['circos_label_min_selection_frequency'] = float(cc.get('label_min_selection_frequency', 0.0) or 0.0)
        state['circos_interaction_min_selection_frequency'] = float(
            cc.get('interaction_min_selection_frequency', 0.0) or 0.0)
        # Region highlighting: absent keys -> highlighting off and widgets at their defaults,
        # so loading an older config re-runs unchanged from the GUI (I11).
        _hl_unit = cc.get('unit', 'bp') or 'bp'
        _hl_rows = _hl_mod.table_rows_from_regions(cc.get('highlight_regions') or [], _hl_unit)
        state['circos_highlight_on'] = bool(_hl_rows)
        state['circos_highlight_regions'] = _hl_rows
        state['circos_highlight_mode'] = cc.get('highlight_mode', 'emphasise')
        state['circos_highlight_match'] = cc.get('highlight_match', 'either')
        state['circos_highlight_colour_by'] = cc.get('highlight_colour_by', 'region')
        state['circos_highlight_colour'] = cc.get('highlight_colour', _hl_mod.DEFAULT_ACCENT)
        state['circos_highlight_dim'] = cc.get('highlight_dim', 'grey')
        state['circos_highlight_dim_opacity'] = float(cc.get('highlight_dim_opacity', _hl_mod.DEFAULT_DIM_OPACITY))
        state['circos_highlight_lw_boost'] = float(cc.get('highlight_lw_boost', _hl_mod.DEFAULT_LW_BOOST))
        state['circos_highlight_band'] = bool(cc.get('highlight_band', True))
        state['circos_highlight_shade'] = bool(cc.get('highlight_shade', True))
        state['circos_highlight_label'] = bool(cc.get('highlight_label', True))
        state['circos_highlight_only_relevant'] = bool(cc.get('highlight_only_relevant', False))
        # `ensemble_hue_rule` has no widget by design (bug-fix rule; hand-edit
        # only) - reported, never guessed at.
        if 'ensemble_hue_rule' in cc:
            unmapped.append('CIRCOS_CONFIG.ensemble_hue_rule (no widget; hand-edited key)')
    if 'CIRCOS_INTERACTION_ORDERS' in cfg:
        orders = take('CIRCOS_INTERACTION_ORDERS') or (2,)
        state['circos_interaction_orders'] = [str(int(o)) for o in orders]
    if 'ASCENDING' in cfg:
        v = take('ASCENDING')
        state['ascending'] = {True: 'True', False: 'False', None: 'None'}.get(v, 'None')
    if 'CYTOBAND_COLORMAP' in cfg:
        state['cytoband_colormap'] = dict(take('CYTOBAND_COLORMAP') or {})

    # --- ASREML: only the fields _asreml_build_cfg_from_widgets() reads
    # 1:1 from a single widget - see this module's own docstring and the
    # design record's own "ASREML sub-tree" scope note. Everything else
    # (genotype_effect_strategy/response_unit/mask_valid_test/blue_
    # refinement/blue_weighting/concurrent_variance_model/wrappers/
    # residual/interaction_groups/nested_groups/accuracy_metrics.direction
    # /grm_source/grm_blend - all DERIVED from more than one widget, or
    # from a widget whose own visibility depends on other choices) is
    # deliberately left unmapped rather than guessed at.
    if 'ASREML' in cfg:
        asreml = take('ASREML')
        if asreml is None:
            state['asreml_enabled'] = False
        else:
            state['asreml_enabled'] = True
            _asreml_simple = {
                'genotype_col': 'asreml_genotype_col', 'fixed': 'asreml_fixed_effects',
                'random': 'asreml_random_effects', 'workspace': 'asreml_workspace',
                'pworkspace': 'asreml_pworkspace', 'max_update_rounds': 'asreml_max_update_rounds',
                'allow_model_fanout': 'asreml_allow_model_fanout',
                'warm_start_per_task': 'asreml_warm_start_per_task',
                'committed_syntax': 'asreml_committed_syntax',
                'syntax_fingerprint': 'asreml_syntax_fingerprint',
                'holdout_committed_syntax': 'asreml_holdout_committed_syntax',
                'holdout_syntax_fingerprint': 'asreml_holdout_syntax_fingerprint',
            }
            _asreml_unmapped_fields = (
                'output_level', 'legacy_masked_holdout', 'genotype_effect',
                'fixed_holdout_mode', 'response_unit', 'concurrent_variance_model',
                'between_shared_genotypes', 'residual',
                'interaction_groups', 'nested_groups', 'wrappers', 'use_grm', 'threads',
            )
            for asreml_field, widget_key in _asreml_simple.items():
                if asreml_field in asreml and asreml[asreml_field] not in (None, ''):
                    state[widget_key] = asreml[asreml_field]
            if asreml.get('threads'):
                state['asreml_threads'] = asreml['threads']
            if asreml.get('on_nonconvergence'):
                state['asreml_on_nonconvergence'] = (
                    'Skip this task' if asreml['on_nonconvergence'] == 'skip'
                    else 'Stop the run (fail)'
                )
            # --- Requirements.md items 1-3 (multi-option update): Genotype
            # effect multiselect, per-option masking/per-set-GRM/BLUE
            # refinement/weighting, per-option committed syntax, and the
            # adjusted-accuracy checkbox. A config saved before the update
            # (single `genotype_effect_strategy`) migrates to a one-element
            # selection.
            _strategy_label = {
                'blue_fixed': 'BLUEs', 'blup_iid': 'BLUPs without GRM',
                'blup_grm': 'BLUPs with GRM',
            }
            _strategies = list(asreml.get('genotype_effect_strategies') or [])
            if not _strategies and asreml.get('genotype_effect_strategy') in _strategy_label:
                _strategies = [asreml['genotype_effect_strategy']]
            _strategies = [k for k in _strategy_label if k in _strategies]   # canonical order
            if 'genotype_effect_strategies' in asreml or 'genotype_effect_strategy' in asreml:
                state['asreml_genotype_strategies_multiselect'] = [
                    _strategy_label[k] for k in _strategies
                ]
            _settings = dict(asreml.get('strategy_settings') or {})
            if not _settings and _strategies:
                # Legacy single-strategy config: its top-level fields ARE the
                # one option's settings.
                _settings = {_strategies[0]: {
                    key: asreml[key] for key in
                    ('mask_valid_test', 'per_set_grm', 'grm_source', 'grm_blend',
                     'blue_refinement', 'blue_weighting') if key in asreml
                }}
            _last_grm_settings = None
            for _key, _opt_settings in _settings.items():
                if _key not in _strategy_label or not isinstance(_opt_settings, dict):
                    continue
                if 'mask_valid_test' in _opt_settings:
                    state[f'asreml_mask_valid_test_checkbox__{_key}'] = bool(_opt_settings['mask_valid_test'])
                if _key == 'blup_grm' and 'per_set_grm' in _opt_settings:
                    state['asreml_per_set_grm_checkbox'] = bool(_opt_settings['per_set_grm'])
                if 'grm_source' in _opt_settings:
                    _last_grm_settings = _opt_settings
                if _key == 'blue_fixed':
                    _refinement = _opt_settings.get('blue_refinement') or {}
                    if isinstance(_refinement, dict):
                        if 'enabled' in _refinement:
                            state['asreml_blue_refinement_enabled'] = bool(_refinement['enabled'])
                        if 'grm_blend' in _refinement:
                            state['asreml_blue_refinement_grm_blend'] = _refinement['grm_blend']
                    _weighting = _opt_settings.get('blue_weighting')
                    _weighting_label = {
                        'none': 'No', 'full': 'full variance-covariance matrix',
                        'diagonal': 'diagonal approximations',
                    }.get(_weighting)
                    if _weighting_label:
                        state['asreml_blue_weighting_radio'] = _weighting_label
            _grm_source_value = (_last_grm_settings or asreml).get('grm_source')
            if _grm_source_value in ('genotype_file', 'task_pool'):
                state['asreml_grm_source'] = (
                    'Genotype file (full marker set)' if _grm_source_value == 'genotype_file'
                    else 'Task pool (whatever other models train on)'
                )
            _grm_blend_value = (_last_grm_settings or asreml).get('grm_blend')
            if _grm_blend_value is not None:
                state['asreml_grm_blend'] = _grm_blend_value
            if 'use_entire_dataset' in asreml:
                state['asreml_use_entire_dataset_checkbox'] = bool(asreml['use_entire_dataset'])
            _committed = dict(asreml.get('committed_syntaxes') or {})
            _fingerprints = dict(asreml.get('syntax_fingerprints') or {})
            if not _committed and asreml.get('committed_syntax') and len(_strategies) == 1:
                _committed = {_strategies[0]: asreml['committed_syntax']}
                if asreml.get('syntax_fingerprint'):
                    _fingerprints = {_strategies[0]: asreml['syntax_fingerprint']}
            if _committed:
                state['asreml_committed_syntaxes'] = _committed
                state['asreml_syntax_fingerprints'] = _fingerprints
                for _key, _text in _committed.items():
                    state[f'asreml_committed_syntax_text__{_key}'] = _text
                    if _fingerprints.get(_key):
                        state[f'_asreml_last_live_fingerprint__{_key}'] = _fingerprints[_key]
            _acc = asreml.get('accuracy_metrics')
            if isinstance(_acc, dict):
                state['asreml_accuracy_metrics_enabled'] = bool(_acc.get('enabled', False))
                state['asreml_accuracy_metrics_use_grm'] = bool(_acc.get('use_grm', False))
            for asreml_field in _asreml_unmapped_fields:
                if asreml_field in asreml:
                    unmapped.append(f'ASREML.{asreml_field}')

    # --- Parallel batch config (Step 1) / assemble mode (Step 2) -----------
    if 'PARALLEL' in cfg:
        parallel = take('PARALLEL')
        state['batch_size'] = parallel.get('batch_size', 3)
        batch_id = parallel.get('batch_id')
        if batch_id is not None:
            state['batch_id_source'] = 'Manual integer'
            state['batch_id_manual'] = batch_id
        else:
            unmapped.append(
                'PARALLEL.batch_id (Slurm/PBS-resolved at run time - which scheduler source '
                'was selected cannot be recovered from the exported config; re-select it on '
                "'1. Data & Setup' if this was a Slurm/PBS array export)"
            )
    if 'ASSEMBLE_MODE' in cfg:
        mode_val = take('ASSEMBLE_MODE')
        _assemble_label_by_mode = {
            'assemble': 'Assemble into combined files',
            'use_preassembled': 'Use existing pre-assembled files',
            'no_assemble': 'Do not assemble (plot directly from batch files)',
        }
        state['step2_assemble_mode_label'] = _assemble_label_by_mode.get(
            mode_val, 'Assemble into combined files'
        )
        state['step2_assemble_mode'] = mode_val
    if 'EXPECTED_BATCHES' in cfg:
        v = take('EXPECTED_BATCHES')
        state['step2_expected_batches'] = v if v else 0

    # --- Everything left in `cfg` at this point had no restoration logic
    # above at all - report every remaining top-level key. Several of
    # these are legitimately unrestorable (R14 pinned fan-out/compute
    # keys with no widget any more, N_JOBS/N_GPU_SLOTS/PLINK_THREADS/
    # GPU_SLOTS_PER_DEVICE added by render_hpc_export_section() itself
    # after gather_config() returns, SEQUENTIAL_INTRA_BATCH) - listing
    # them anyway (rather than silently swallowing them) is what AC1.3
    # asks for: an explicit, honest "not restored" report, never a
    # silent partial restore.
    unmapped.extend(sorted(cfg.keys()))

    return state, unmapped


# --------------------------------------------------------------------------- #
# ver4-54 (R6): inverse map of a Tool config -> Tool widget keys.
# --------------------------------------------------------------------------- #

def tool_config_to_gui_state(cfg):
    """Inverse of `main_app.gather_tool_config()` - a `tool_config.json` ->
    `(state, unmapped)` in the same shape `engine_config_to_gui_state()` returns
    (`state`: widget key -> value for `st.session_state`; `unmapped`: every config
    key nothing was restored from).

    A Tool config is small and every field has exactly one widget, so unlike a
    pipeline config the inverse is complete: with the `<purpose>_gui_state.json`
    sidecar deleted, sources, editable names, model filters and every section
    setting still come back, and `unmapped` is empty for a config this app wrote.
    `RUN_KIND`, `TOOL` and `ENGINE_VERSION` are provenance only (consumed, not
    restored). A section absent from the config was skipped by the run, so its
    two tick boxes are loaded unticked (the ver4-60 scatter section has only the
    one, `create`). `same_as_first` is always loaded False:
    the exported config already holds each section's own full source list."""
    state = {'exec_mode': 'Tool'}
    unmapped = []
    cfg = dict(cfg)
    for provenance_key in ('RUN_KIND', 'TOOL', 'ENGINE_VERSION'):
        cfg.pop(provenance_key, None)
    for engine_key, widget_key in (('RESULT_NAME', 'tool_result_name'),
                                   ('TOOL_GATHER_MODE', 'tool_gather_mode')):
        if engine_key in cfg:
            state[widget_key] = cfg.pop(engine_key)
    if 'PLOT_EXTRA_FORMATS' in cfg:
        state['tool_plot_extra_formats'] = list(cfg.pop('PLOT_EXTRA_FORMATS') or [])
    if 'MODEL_LABEL_STYLE' in cfg:
        state['tool_model_label_style'] = str(cfg.pop('MODEL_LABEL_STYLE') or 'full')     # ver4-95

    comparison = cfg.pop('TOOL_COMPARISON', None)
    comparison = dict(comparison) if isinstance(comparison, dict) else {}
    for section in TOOL_ALL_SECTIONS:
        if section not in comparison:
            state[tool_widget_key(section, 'create')] = False
            if section != TOOL_SCATTER_SECTION:   # the scatter section has no workbook
                state[tool_widget_key(section, 'summary_create')] = False
            state[tool_widget_key(section, 'same_as_first')] = False
            continue
        scfg = dict(comparison.pop(section) or {})
        source_ids = []
        for item in scfg.pop('sources', None) or []:
            source_id = make_source_id(item.get('folder', ''), item.get('option'))
            source_ids.append(source_id)
            label = item.get('label') or default_source_label(item.get('folder', ''), item.get('option'))
            state[tool_label_key(section, source_id)] = label
        state[tool_widget_key(section, 'sources')] = source_ids
        models = scfg.pop('models', None)
        if models is not None:
            state[tool_widget_key(section, 'models')] = list(models)
        for engine_key, suffix in _TOOL_SECTION_FIELDS:
            if engine_key in scfg:
                value = scfg.pop(engine_key)
                state[tool_widget_key(section, suffix)] = (
                    list(value) if engine_key == 'metrics' else value
                )
        plot_cfg = dict(scfg.pop('plot_config', None) or {})
        for engine_key, suffix in _TOOL_PLOT_FIELDS:
            if engine_key in plot_cfg:
                state[tool_widget_key(section, suffix)] = plot_cfg.pop(engine_key)
        unmapped.extend(f'TOOL_COMPARISON.{section}.plot_config.{k}' for k in sorted(plot_cfg))
        unmapped.extend(f'TOOL_COMPARISON.{section}.{k}' for k in sorted(scfg))
        state[tool_widget_key(section, 'same_as_first')] = False
    unmapped.extend(f'TOOL_COMPARISON.{k}' for k in sorted(comparison))
    # ver4-101: the optional per-run diagnostics block (a top-level key, NOT a comparison section)
    diagnostics = cfg.pop('TOOL_DIAGNOSTICS', None)
    diagnostics = dict(diagnostics) if isinstance(diagnostics, dict) else {}
    state[tool_widget_key(TOOL_DIAGNOSTICS_SECTION, 'create')] = bool(diagnostics.get('folders'))
    for engine_key, default in _TOOL_DIAGNOSTICS_NEW_DEFAULTS.items():
        state[tool_widget_key(TOOL_DIAGNOSTICS_SECTION, engine_key)] = default
    for engine_key, suffix in _TOOL_DIAGNOSTICS_FIELDS:
        if engine_key in diagnostics:
            value = diagnostics.pop(engine_key)
            state[tool_widget_key(TOOL_DIAGNOSTICS_SECTION, suffix)] = (
                list(value) if engine_key in ('folders', 'tools') else value
            )
    unmapped.extend(f'TOOL_DIAGNOSTICS.{k}' for k in sorted(diagnostics))
    unmapped.extend(sorted(cfg))
    return state, unmapped
