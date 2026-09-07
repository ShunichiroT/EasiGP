"""
EasiGP - Genomic Prediction Pipeline GUI
--------------------------------------------------------------------------
HOW TO RUN ON THE HPC AND VIEW IT LOCALLY
--------------------------------------------------------------------------
1. On the HPC (e.g. after `ssh` or via your job's shell), start the app on
   a fixed port:

       streamlit run streamlit_app.py --server.port 8501 --server.headless true

   (pick any free port; 8501 is Streamlit's default)

2. Forward that port to your own machine:

   - PuTTY: in the session's connection settings, go to
     Connection > SSH > Tunnels, set "Source port" to 8501 and
     "Destination" to localhost:8501, click Add, then connect as usual.

   - MobaXterm: Tools > MobaSSHTunnel (or the "Tunneling" button), add a
     local port forward: Local port 8501 -> Remote server 127.0.0.1,
     Remote port 8501, via your existing SSH session.

   - Or, from a terminal: `ssh -L 8501:localhost:8501 you@hpc-host`

3. Open http://localhost:8501 in your local browser.

--------------------------------------------------------------------------
"""

import os
import math
import io
import ast
import json
import time
import tempfile
import uuid
import numpy as np
import matplotlib.colors as mcolors
from datetime import datetime
import inspect
import contextlib
import traceback
import re

import streamlit as st

from genomic_prediction import *
from assemble import *
from metric_plot import *
from scatter_plot import *
from circos_plot import *
from circos_plot import _clean_population_label, _broadcast_population_info
# Update ID ver4-6, R2 (blueprint §3.2/§3.4): the same shared geometry
# authority circos_plot.py itself renders through - importing it here is
# a downward import (Layer 6 -> Layer 3c), the same direction as the
# existing `from checkpoint_utils import (...)` below.
import circos_geometry
from attention_histogram import *
from Preprocess.LD_decay_plot import (
    average_and_plot_ld_decay, ld_decay_data_exists, resolve_snp_info_for_decay,
    WINDOW_UNITS as LD_DECAY_WINDOW_UNITS, DEFAULT_MAX_DISTANCE as LD_DECAY_DEFAULT_MAX_DISTANCE,
    DEFAULT_BIN_WIDTH as LD_DECAY_DEFAULT_BIN_WIDTH,
    DEFAULT_MAX_PAIRS_PER_CHR as LD_DECAY_DEFAULT_MAX_PAIRS_PER_CHR,
)
from checkpoint_utils import (
    find_incomplete_batches, sequential_run_status, describe_incomplete_batch, check_batch_status,
    format_batch_id_list,
)

# Short unit label shown inside LD decay plot number_input widgets (e.g.
# 'Max distance shown (kb)') - kept alongside the import above since it's
# purely a display concern for that one section of the GUI.
_UNIT_LABEL_FOR_GUI = {'kb': 'kb', 'cm': 'cM', 'variants': 'markers'}
from pipeline_utils import (
    BATCH_ID_SOURCES, configure_r_environment, init_rpy2_conversion,
    detect_array_job_env, restore_ratio, resolve_batch_id_from_env,
    TimestampedWriter, make_run_log_path, list_phenotype_columns,
    resolve_compute_resources, count_unique_populations, result_dir_path,
)
from resource_profiles import (
    estimate_resources, _GADI_GPU_NCPUS_PER_GPU, _GPU_PREPROCESS_CPU_HEADROOM,
    HPC_RESOURCE_PROFILES, DEFAULT_HPC_PROFILE, resolve_hpc_profile,
    # ver4-4, R1 (blueprint §2.1) - every exact-dividing (batch_size,
    # n_batches) pair for the 'Suggested job-array size' selector.
    enumerate_array_layouts,
)
from metric_summary import write_metric_summary
# Update ID ver4-9, R5.
from diversity_summary import build_dpt_terms, write_dpt_summary, resolve_ensemble_members, dpt_model_order
# Update ID ver4-9, R6.
from weight_plot import weight_plot
# ver4-4 R7.g: the SAME "5 leading metadata columns, everything past that
# is a marker" convention batch_reader.ResultSet.effect() already uses for
# its own float32 downcast - imported here so the Sequential in-process
# block below (which has no ResultSet - `effect` is GP()'s own in-memory
# accumulator) can apply the identical downcast without re-declaring the
# same literal width a second time.
from batch_reader import _EFFECT_METADATA_WIDTH, load_combined

# The biological prior-knowledge GAT model's preprocessing helpers
# (Preprocess/gene_network_prior.py, flash_p_integration.py,
# gene_location_agent.py) are optional add-ons - guarded here so that
# dropping this GUI file in *before* those three files exist still leaves
# the rest of the app fully working (the new tab and model are simply
# hidden/disabled with an explanatory message, rather than the whole app
# failing to start).
try:
    from Preprocess.gene_network_prior import (
        load_network_json, extract_candidate_genes, build_gene_list,
        save_gene_list_csv, build_gene_adjacency, save_adjacency,
        phenotype_matches_network_metadata,
    )
    from Preprocess.flash_p_integration import run_flash_p, locate_network_json, preflight_check
    from Preprocess.gene_location_agent import curate_gene_locations
    BIO_PRIOR_AVAILABLE = True
    BIO_PRIOR_IMPORT_ERROR = None
except ImportError as _bio_prior_exc:
    BIO_PRIOR_AVAILABLE = False
    BIO_PRIOR_IMPORT_ERROR = str(_bio_prior_exc)


st.set_page_config(page_title='EasiGP', layout='wide')

st.markdown(
    """
    <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
        /* Pull a caption closer to the input it's describing, rather than
           leaving the same default gap used between unrelated widgets. */
        div[data-testid="stCaptionContainer"],
        div[data-testid="stCaption"] {
            margin-top: -0.65rem;
            padding-top: 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Remember the user's last-used GUI settings across separate app launches.
#
# The very first time the app is ever launched (no saved file yet), nothing
# below has any effect and every widget uses its normal hard-coded default -
# behaviour is unchanged from before this feature existed.
#
# On every later launch, whatever was last configured (which models are
# selected, every hyperparameter, file paths, ratios, LD-pruning settings,
# etc.) is restored automatically, so re-running with the same setup doesn't
# require re-entering everything.
#
# This works generically off the whole of st.session_state rather than
# enumerating individual fields one by one - HPARAM_SPECS, model selection,
# and the composite field types (e.g. 'int_or_all') together generate
# hundreds of dynamically-named widget keys, so hand-listing them would be
# both a large amount of code and a maintenance trap (easy to forget to add
# a new field here later). Only plain JSON-serialisable values are kept, and
# any single key that can't be saved/restored (there shouldn't be many, if
# any - Streamlit widgets here are all plain numbers/strings/bools/lists) is
# silently skipped rather than breaking the app.
GUI_STATE_FILE = os.path.join('.', 'easigp_last_gui_state.json')

# Requirement 1 (bugfix): the circos GUI fields that auto-fill themselves
# (see _autofill_number_field()'s own docstring, further down) - and their
# '_..._last_autofill' tracking companions - are deliberately excluded from
# save_gui_state()'s persistence entirely (see that function's own comment
# for why). Listed explicitly here, once, rather than inferred by a naming
# pattern, so both save_gui_state() and load_gui_state() stay in sync
# automatically if this list ever changes.
_CIRCOS_AUTOFILL_FIELDS = {
    'circos_space', 'circos_start', 'circos_end', 'circos_link_alpha_min',
    'circos_labelsize', 'circos_scale', 'window_size', 'end_adjust', 'gene_adjust',
    # Update ID ver4-6, R2 (blueprint §3.4 S4): the one further field this
    # update adds that auto-fills the same way as the fields above.
    'circos_ring_label_size',
}
_CIRCOS_AUTOFILL_TRACKING_KEYS = {f'_{_f}_last_autofill' for _f in _CIRCOS_AUTOFILL_FIELDS}

# Requirement 2 (crash fix - StreamlitValueAssignmentNotAllowedError):
# Streamlit refuses to let ANY code (including load_gui_state()'s generic
# session_state.setdefault() loop below) programmatically set the value of
# an `st.file_uploader` / `st.camera_input` widget - an uploaded file is a
# live, non-serialisable browser buffer, not a plain settable value, and
# Streamlit enforces this by raising StreamlitValueAssignmentNotAllowedError.
# The whole-of-session_state persistence mechanism (see GUI_STATE_FILE's own
# comment above) doesn't know about this restriction on its own, so on a
# later app launch it would try to restore a previously-uploaded file's key
# and crash the entire GUI before the rest of gather_config()'s widget
# restoration ever runs - "previously set configurations were not used" is
# the direct, whole-app-crash symptom of this.
#
# Fix, matching the already-existing _CIRCOS_AUTOFILL_FIELDS pattern above:
# every upload-type widget's key is listed here explicitly, ONCE, so
# save_gui_state()/load_gui_state() stay in sync automatically. Every
# `st.file_uploader(`/`st.camera_input(` call site in this file must have
# its `key=` added here (audited as part of Requirement 2 - currently just
# the one bash-file-conversion uploader in render_hpc_export_section()).
_NON_RESTORABLE_WIDGET_KEYS = {
    f'_{_kp}_sh_convert_upload' for _kp in ('sequential', 'step1', 'step2')
}


def _is_non_restorable_widget_key(key):
    """True for any key known (or, via the '_upload'/'_uploader'/'_camera'
    suffix heuristic below, suspected) to belong to an `st.file_uploader` /
    `st.camera_input` widget - see _NON_RESTORABLE_WIDGET_KEYS's own
    comment. The explicit set is the primary check (exact, no false
    positives); the suffix heuristic is a defence-in-depth net for any
    upload-type widget added later whose key wasn't (yet) added to that
    set explicitly - it only matches the conventional naming this codebase
    already uses for such widgets, so it should never trip on an ordinary
    text/number/select field."""
    if key in _NON_RESTORABLE_WIDGET_KEYS:
        return True
    return key.endswith(('_upload', '_uploader', '_camera_input'))


def load_gui_state():
    """Restore last-used settings from disk, if any. Uses setdefault() so it
    only seeds keys that aren't already present in this session - on a
    session's first run this pre-populates every remembered widget value; on
    later reruns within the same session (the user is actively changing
    things) it's a no-op and never overwrites a live edit."""
    if not os.path.isfile(GUI_STATE_FILE):
        return
    try:
        with open(GUI_STATE_FILE, 'r', encoding='utf-8-sig') as f:
            saved = json.load(f)
    except Exception:
        return  # corrupted/unreadable state file - fall back to defaults silently

    for key, value in saved.items():
        if key.startswith('_btn_'):
            continue  # never replay a button click on a later launch - see below
        # Requirement 1 (bugfix, defence-in-depth): also skip these on the
        # READ side, in addition to save_gui_state() no longer WRITING them
        # (see that function's own comment) - a state file saved by an
        # older version of this app, before this exclusion existed, could
        # still have a stale value for one of these sitting in it; skipping
        # it here too means even an old, already-saved file can't
        # reintroduce the same staleness problem this fix exists for.
        if key in _CIRCOS_AUTOFILL_FIELDS or key in _CIRCOS_AUTOFILL_TRACKING_KEYS:
            continue
        # Requirement 2 (crash fix, primary mitigation): never even attempt
        # to restore a known non-restorable (uploader/camera) widget key -
        # see _NON_RESTORABLE_WIDGET_KEYS's own comment. save_gui_state()
        # never writes these going forward, but an OLDER state file saved
        # before this fix existed could still contain one.
        if _is_non_restorable_widget_key(key):
            continue
        try:
            st.session_state.setdefault(key, value)
        except Exception:
            # Requirement 2 (crash fix, defence-in-depth safety net): catch
            # broadly here (Streamlit's own
            # StreamlitValueAssignmentNotAllowedError, plus any other
            # currently-unknown non-restorable widget type added later and
            # not yet covered by _is_non_restorable_widget_key()) rather
            # than a narrow except - this converts what used to be a
            # whole-GUI-crashing exception into a graceful, logged, single-
            # key skip, so the rest of this restoration loop (every other
            # saved field) still runs normally.
            print(f"[EasiGP] load_gui_state(): could not restore session_state "
                  f"key {key!r} (widget type likely doesn't support "
                  f"programmatic restoration) - skipping just this one key.")
            continue


def save_gui_state():
    """Snapshot the current settings to disk so they're remembered next time
    the app is launched. Called once at the very end of every script run, so
    any change the user makes is persisted almost immediately."""
    snapshot = {}
    for key, value in st.session_state.items():
        # Every button in this app is keyed with a '_btn_' prefix specifically
        # so it can be excluded here: a button's session-state value is just
        # "was it clicked this run", and restoring True for one on a later
        # launch would auto-trigger that action (e.g. auto-running the whole
        # pipeline) without the user ever clicking anything - never persist
        # these, regardless of whether Streamlit would otherwise allow it.
        if key.startswith('_btn_'):
            continue
        # Requirement 1 (bugfix - an auto-filled circos field, e.g.
        # GENE_ADJUST, silently never updating again even across an app
        # restart): _autofill_number_field()'s own "has this been manually
        # overridden" check compares a field's CURRENT value against
        # whatever value IT last auto-filled - if the two get persisted to
        # disk in a way that ever lets them drift apart (a value from an
        # older app version before a formula changed, a value saved before
        # its own '_..._last_autofill' companion key existed at all, etc.),
        # that field looks "manually overridden" forever after, even
        # though the person never actually typed anything - exactly what
        # was reported. Never persisting these fields (or their tracking
        # keys) at all sidesteps the whole class of problem: every one
        # always starts genuinely blank on a fresh app launch, so the
        # auto-fill logic runs cleanly from scratch every time, with
        # nothing that can ever go stale. A person's in-session edits
        # (including a deliberate manual override) still work exactly as
        # before for the rest of that session - this only affects what
        # persists to the NEXT app launch.
        if key in _CIRCOS_AUTOFILL_FIELDS or key in _CIRCOS_AUTOFILL_TRACKING_KEYS:
            continue
        # Requirement 2 (crash fix, primary mitigation): never persist an
        # uploader/camera widget's own value - an uploaded file cannot
        # outlive the browser session regardless of what persistence layer
        # tries to hold onto it, and attempting to restore it on the next
        # launch is exactly what raises StreamlitValueAssignmentNotAllowedError
        # (see _NON_RESTORABLE_WIDGET_KEYS's own comment). The *metadata*
        # (last-used filename) each such widget's call site persists
        # instead is a plain string under a different, non-widget key, so
        # it's saved normally by this same generic loop.
        if _is_non_restorable_widget_key(key):
            continue
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue  # not JSON-serialisable - skip rather than fail the save
        snapshot[key] = value
    try:
        with open(GUI_STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f)
    except Exception:
        pass  # best-effort - a failed save should never break the app


load_gui_state()


# --------------------------------------------------------------------------- #
# Static configuration / defaults (mirrors the values that used to be
# hard-coded at the top of the original script)
# --------------------------------------------------------------------------- #

AVAILABLE_MODELS = [
    'rrBLUP', 'GBLUP', 'BayesB', 'RKHS', 'RF', 'ExtraTrees', 'XGBoost', 'EBM',
    'SVR', 'KNN', 'MLP',
    'GAT_infinitesimal', 'GAT_fully_connected', 'GAT_prior_knowledge',
    'GAT_biological_prior_knowledge', 'ensemble'
]

# Update ID ver4-5, R2 Stage 11 (decision D2): Tier 2 models need an
# OPTIONAL third-party package (model_registry.optional_dependency_for())
# - AVAILABLE_MODELS above still lists them unconditionally (so a saved
# GUI state / headless config referencing them keeps resolving to a real
# name), but the model-picker loop below (Tab 3) filters them out of the
# SELECTABLE checkboxes whenever their package isn't importable in this
# environment, replacing the checkbox with an explanatory caption naming
# the missing package - never a silent omission.
TIER_2_MODELS = frozenset({'XGBoost', 'EBM'})

MODEL_DESCRIPTIONS = {
    'rrBLUP': "Ridge regression: assumes every marker has a small effect, and shrinks them all towards zero by a similar amount. A solid, fast general-purpose baseline.",
    'GBLUP': "Similar idea to rrBLUP, but works from overall genetic similarity between individuals rather than marker-by-marker effects directly.",
    'BayesB': "Like rrBLUP, but assumes only a subset of markers have a real effect (sparser) - a good fit when you expect a few large-effect markers rather than many small ones.",
    'RKHS': "Captures non-linear relationships between individuals' genetics and their trait, using a flexible similarity-based (kernel) approach rather than assuming purely additive marker effects.",
    'RF': "Random Forest: an ensemble of decision trees. Handles non-linear effects and marker interactions well, and can rank markers by importance.",
    'ExtraTrees': "Extremely Randomised Trees: a sibling of Random Forest that picks split thresholds at random rather than searching for the best one. Often a genuinely different, sometimes less overfitted, alternative to RF on high-dimensional genotype data.",
    'XGBoost': "A fast, widely-used gradient-boosting implementation. Requires the optional 'xgboost' package.",
    'EBM': "Explainable Boosting Machine: a 'glass-box' model whose pairwise marker interactions are part of the fit itself, not a separate explanation step computed afterwards. Requires the optional 'interpret' package.",
    'SVR': "Support Vector Regression: fits a flexible boundary through the data, with a tunable margin of error. Can capture non-linear patterns via its kernel setting.",
    'KNN': "K-Nearest Neighbours: predicts a trait as an average of the most genetically similar individuals in the training set. Simple and intuitive, but can be slow on large datasets.",
    'MLP': "Multi-Layer Perceptron: a small neural network. Can capture complex, non-linear patterns, but typically needs more data and tuning than the other models to do so reliably.",
    'GAT_infinitesimal': "A graph neural network that treats every marker does not interact with every other marker.",
    'GAT_fully_connected': "A graph neural network that treats every marker interacts with every other marker.",
    'GAT_prior_knowledge': "A graph neural network that first uses a Random Forest to identify likely-interacting marker pairs, and only connects those in the graph - can be faster and more targeted than connecting every marker to every other.",
    'GAT_biological_prior_knowledge': "A graph neural network where each node is a GENE (grouping together every marker that falls inside it), and the graph's edges come from a curated, external gene-interaction network (uploaded, or built with FLASH-P) instead of being learned from the data - see the 'Biological Prior Network' tab to build its inputs first.",
    'ensemble': "Combines the predictions of every other selected model together (a simple average, unless a weight-optimisation method is also chosen in '4. Ensemble'), often giving more robust predictions than any single model alone. Selected on the '4. Ensemble' tab.",
}

W_OPT_METHODS = ['Nelder Mead', 'Linear transformation', 'Bayesian optimisation', 'Analytic least-squares']

W_OPT_METHOD_DESCRIPTIONS = {
    'Nelder Mead': "Searches for the best per-model weights using a direct, gradient-free numerical search. Simple and generally reliable.",
    'Linear transformation': "Learns the per-model weights using a small neural network (see its settings below), rather than a direct numerical search.",
    'Bayesian optimisation': "Searches for the best per-model weights intelligently, using past attempts to decide where to try next - can find good weights in fewer attempts than Nelder Mead, at the cost of more setup overhead per attempt.",
    'Analytic least-squares': "Solves directly (in milliseconds, via scipy) for the exact, closed-form least-squares combination of your selected models - no iterative search at all, so there's nothing to tune and no randomness between runs.",
}

# Each hyperparameter field is a dict:
#   label:    text shown in the GUI
#   type:     int, float, bool, str, top_pct, int_float_or_none,
#             rf_max_features, svr_gamma
#   default:  pre-filled value
#   choices:  (optional) fixed options offered in a selectbox
#   depends_on: (optional) (controller_index, required_value) - this field
#               is greyed out unless the field at controller_index (within
#               the same model's list, 0-indexed) currently equals
#               required_value.
#   visible_when: (optional) (controller_index, required_value) - like
#               depends_on, but stronger: this field's widget(s) are not
#               rendered AT ALL (not just greyed out) unless the field at
#               controller_index currently equals required_value. Used for
#               settings that are meaningless - not merely inactive - under
#               the controller's other values (e.g. RF's friedman_h-only
#               interaction pre-screen settings, irrelevant whenever
#               'Marker interaction method' is 'pairwise_shap'). A field may
#               carry both keys: 'visible_when' governs whether it appears
#               at all, 'depends_on' still governs greying out once it does.
#
# Field order within each model matches the positional order the backend
# expects in HPARAMETERS[model] - do not reorder.
# HPARAM_SPECS moved to hparam_specs.py (shared with the headless hyperparameter
# tuning engine in models/hyperparameter_tuning.py - see that module's docstring).
# render_field()/resolve_field() below are unaffected: they only ever read
# type/label/default/choices/combo_state/depends_on from each field, none of which
# changed - hparam_specs.py's fields are byte-identical to what used to be defined
# inline here, with 'tunable' ranges merged in on top for the fields that affect fit.
from hparam_specs import HPARAM_SPECS
from model_registry import is_available, optional_dependency_for
# 'ensemble' has no hyperparameters of its own (it combines other models' output).

HYPERPARAM_OPT_SPECS = {
    'Linear transformation': [
        {'label': 'Learning rate', 'type': 'float', 'default': 0.005,
         'help': ("How big a step is taken when adjusting each model's weight after each "
                  "round of learning. Larger values learn faster but can overshoot and "
                  "become unstable; smaller values are more stable but slower.")},
        {'label': 'Epochs', 'type': 'int', 'default': 150,
         'help': ("How many times the weight-learning process passes over the validation "
                  "set while working out how much to trust each model. More epochs can find "
                  "better weights but take longer and risk overfitting.")},
        {'label': 'Decay', 'type': 'float', 'default': 0.01,
         'help': ("A small penalty that discourages the learned weights from growing too "
                  "large, which helps avoid over-relying on any single model.")},
        {'label': 'Batch size', 'type': 'int', 'default': 2,
         'help': ("How many individuals are processed together before each weight update. "
                  "Smaller batches update more often (noisier but can generalise better); "
                  "larger batches are steadier but slower per update.")},
        {'label': 'Patience value', 'type': 'int', 'default': 10,
         'help': ("Stops training early if the fit hasn't improved for this many epochs in a "
                  "row, to avoid wasting time (or overfitting) once it's stopped getting "
                  "better.")},
        {'label': 'Number of MLP models', 'type': 'int', 'default': 30,
         'help': ("This method learns the model weights using a small neural network trained "
                  "several times from different random starting points, keeping the best run "
                  "- this sets how many times it tries. More attempts are more likely to find "
                  "a good set of weights, but take longer.")},
        # Phase 2, Requirement 4/5 follow-up (§4.4 of the audit) - appended-
        # only, indices 6-8, read positionally by Linear_transformation.py.
        # Every default below exactly reproduces this method's original
        # (pre-Phase-2) behaviour, so an old saved config round-trips
        # unchanged.
        {'label': 'Diversity penalty (lambda)', 'type': 'float', 'default': 0.0,
         'help': ("Optional extra regularisation: an additional ridge-to-uniform penalty on "
                  "the model's own combining weights, discouraging them from drifting far from "
                  "equal weighting. 0 (the default) leaves this method's original loss "
                  "unchanged. Not structurally required for this method the way it is for "
                  "Nelder-Mead/Bayesian optimisation - see the tooltip on those two.")},
        {'label': 'Use simplex (non-negative, sum-to-1) weights', 'type': 'bool', 'default': False,
         'help': ("By default this method's combining weights are completely unconstrained, "
                  "which allows negative or extrapolating combinations of models. Enabling "
                  "this reparameterises them through a softmax, so they are always "
                  "non-negative and sum to 1 - a more conservative, blend-only combination, "
                  "at the cost of being unable to represent negative-weight corrections.")},
        {'label': 'Naive-shrinkage alpha', 'type': 'float', 'default': 1.0,
         'help': ("Blends this method's own predictions toward the plain equal-weighted "
                  "(naive) ensemble as an extra safety margin against overfitting the "
                  "validation split: 1.0 (the default) uses this method's predictions as-is; "
                  "lower values shrink further toward the naive average; 0.0 reproduces the "
                  "naive ensemble exactly. Combined with a built-in guard that never performs "
                  "worse than the naive ensemble on the validation split.")},
    ],
    'Nelder Mead': [
        {'label': 'Initial value', 'type': 'float', 'default': 0.5,
         'help': ("The starting weight given to every model before the search begins. Since "
                  "all models start equal, the exact value doesn't usually matter much - it "
                  "just needs to be a valid starting point within the boundaries below. This "
                  "equal-weighting point is also always evaluated as one of several restarts "
                  "the search tries (see 'Adaptive' below), and the search is now guaranteed "
                  "to never return something worse than it.")},
        {'label': 'Minimum boundary', 'type': 'float', 'default': 0.0,
         'help': ("The smallest weight any individual model is allowed to be given during "
                  "the search. 0 (the default) lets the search effectively drop a "
                  "consistently weak model from the ensemble entirely, rather than always "
                  "keeping it at some small nonzero share - now safe to allow, since an "
                  "all-zero-weight combination is automatically detected and handled.")},
        {'label': 'Maximum boundary', 'type': 'float', 'default': 1,
         'help': ("The largest weight any individual model is allowed to be given during "
                  "the search.")},
        {'label': 'fatol', 'type': 'float', 'default': 1e-8,
         'help': ("The search stops once further attempts stop improving the fit by more "
                  "than this tiny amount. Smaller values search more thoroughly (slower); "
                  "larger values stop sooner (faster, less precise).")},
        {'label': 'xatol', 'type': 'float', 'default': 1e-8,
         'help': ("Similar to fatol above, but based on how much the weights themselves stop "
                  "changing between attempts, rather than how much the fit improves.")},
        {'label': 'Adaptive', 'type': 'bool', 'default': True,
         'help': ("Automatically adjusts the search's internal step sizes for problems with "
                  "many models being weighted at once - recommended (and now the default) "
                  "whenever more than a handful of models are being combined, which is the "
                  "common case for this method; scipy's own guidance is to enable this "
                  "once the number of models being weighted climbs much past 4-5.")},
        # Phase 2, Requirement 5 - appended-only, indices 6-9, read
        # positionally via ensemble_regularization.read_regularization_
        # settings(start_index=6). Every default below reproduces this
        # method's original (pre-Phase-2) behaviour exactly, so an old
        # saved config round-trips unchanged.
        {'label': 'Diversity penalty (lambda)', 'type': 'float', 'default': 0.0,
         'help': ("Strength of a regulariser added before optimising, needed to fix a "
                  "structural issue in the ORIGINAL 'dpt_ratio' objective below: it always "
                  "pushes toward one model dominating and the rest collapsing to the boundary, "
                  "rather than a genuine blend. Only relevant when Objective (below) is set to "
                  "'dpt_ratio' - the default 'ensemble_mse' objective doesn't need it (it has a "
                  "genuine interior optimum on its own). 0 (default) disables it; a small "
                  "positive value (e.g. 0.01-0.1) helps if you deliberately use 'dpt_ratio' and "
                  "see near-one-hot weights.")},
        {'label': 'Objective', 'type': 'str', 'default': 'ensemble_mse',
         'choices': ['dpt_ratio', 'ensemble_mse'],
         'help': ("'ensemble_mse' (default since ver4-6) directly minimises the combined "
                  "ensemble's own error against the validation target - a naturally "
                  "well-behaved, interior-optimised objective. 'dpt_ratio' is the original "
                  "published Diversity Prediction Theorem objective; its own optimum is "
                  "provably a single-model selection (a simplex vertex), not a genuine blend, "
                  "which measured 43% worse validation error than simple equal weighting in "
                  "this update's own testing - kept selectable to reproduce published DPT "
                  "results, but not recommended for prediction accuracy. Whichever you choose, "
                  "a validation-MSE floor (see the resource-advisor settings) now guarantees "
                  "the returned weights are never worse than equal weighting.")},
        {'label': 'Naive-shrinkage alpha', 'type': 'float', 'default': 1.0,
         'help': ("Blends the search's own weights toward equal (naive) weighting as an extra "
                  "safety margin: 1.0 (default) uses the search's weights as-is; lower values "
                  "shrink further toward equal weighting; 0.0 reproduces equal weighting "
                  "exactly.")},
        {'label': 'Diversity penalty method', 'type': 'str', 'default': 'ridge_to_uniform',
         'choices': ['ridge_to_uniform', 'entropy'],
         'help': ("How the diversity penalty above is computed. 'ridge_to_uniform' (default) "
                  "penalises squared distance from equal weighting. 'entropy' instead "
                  "penalises the shortfall from maximum (uniform) entropy - a softer penalty "
                  "that tolerates moderate imbalance more than ridge-to-uniform.")},
        # ver4-5 R1.d (blueprint §3.6) - appended-only, index 10 (after
        # the Requirement-5 regularisation fields above, which occupy
        # 6-9). Default True reproduces this update's own new default -
        # NOT ver4-4's pre-R1.d behaviour, which had no vectorised path
        # at all; setting this False is the escape hatch back to the
        # exact original floating-point summation order.
        {'label': 'Vectorised objective', 'type': 'bool', 'default': True,
         'help': ("Evaluates the weight-search objective as a single fast matrix computation "
                  "instead of a per-model loop - mathematically the same formula, agreeing "
                  "with the original computation to about 1 part in a trillion, but the tiny "
                  "floating-point rounding differences can occasionally nudge the search onto "
                  "a different (equally valid) answer. Turn off to reproduce the exact original "
                  "computation.")},
    ],
    'Bayesian optimisation': [
        {'label': 'Minimum boundary', 'type': 'float', 'default': 0.0,
         'help': ("The smallest weight any individual model is allowed to be given during "
                  "the search. 0 (the default) lets the search effectively drop a "
                  "consistently weak model from the ensemble entirely, rather than always "
                  "keeping it at some small nonzero share - now safe to allow, since an "
                  "all-zero-weight combination is automatically detected and handled.")},
        {'label': 'Maximum boundary', 'type': 'float', 'default': 10.0,
         'help': ("The largest weight any individual model is allowed to be given during "
                  "the search.")},
        {'label': 'Iterations', 'type': 'int', 'default': 50,
         'help': ("How many rounds of weight combinations this search tries before settling "
                  "on the best one found. More iterations search more thoroughly but take "
                  "longer.")},
        {'label': 'Point numbers', 'type': 'int', 'default': 3,
         'help': ("How many random weight combinations are tried before the search starts "
                  "using what it's learned so far to make smarter guesses (in addition to "
                  "equal weighting, which is always tried first regardless of this setting). "
                  "A handful gives the search a better initial sense of the weight space to "
                  "build on before it starts making targeted guesses.")},
        {'label': 'Allow duplicate points', 'type': 'bool', 'default': True,
         'help': ("Whether the search is allowed to try the exact same weight combination "
                  "more than once. Leaving this on avoids the search getting stuck if it "
                  "runs out of new combinations to try.")},
        # Phase 2, Requirement 5 - appended-only, indices 5-8, read
        # positionally via ensemble_regularization.read_regularization_
        # settings(start_index=5). Same fields/defaults as Nelder Mead
        # above (see its comments) - this method's own original params
        # list is one field shorter (5 vs 6), hence the different
        # start_index.
        {'label': 'Diversity penalty (lambda)', 'type': 'float', 'default': 0.0,
         'help': ("Strength of a regulariser added before optimising, needed to fix a "
                  "structural issue in the ORIGINAL 'dpt_ratio' objective below: it always "
                  "pushes toward one model dominating and the rest collapsing to the boundary, "
                  "rather than a genuine blend. Only relevant when Objective (below) is set to "
                  "'dpt_ratio' - the default 'ensemble_mse' objective doesn't need it (it has a "
                  "genuine interior optimum on its own). 0 (default) disables it; a small "
                  "positive value (e.g. 0.01-0.1) helps if you deliberately use 'dpt_ratio' and "
                  "see near-one-hot weights.")},
        {'label': 'Objective', 'type': 'str', 'default': 'ensemble_mse',
         'choices': ['dpt_ratio', 'ensemble_mse'],
         'help': ("'ensemble_mse' (default since ver4-6) directly minimises the combined "
                  "ensemble's own error against the validation target - a naturally "
                  "well-behaved, interior-optimised objective. 'dpt_ratio' is the original "
                  "published Diversity Prediction Theorem objective; its own optimum is "
                  "provably a single-model selection (a simplex vertex), not a genuine blend, "
                  "which measured 43% worse validation error than simple equal weighting in "
                  "this update's own testing - kept selectable to reproduce published DPT "
                  "results, but not recommended for prediction accuracy. Whichever you choose, "
                  "a validation-MSE floor (see the resource-advisor settings) now guarantees "
                  "the returned weights are never worse than equal weighting.")},
        {'label': 'Naive-shrinkage alpha', 'type': 'float', 'default': 1.0,
         'help': ("Blends the search's own weights toward equal (naive) weighting as an extra "
                  "safety margin: 1.0 (default) uses the search's weights as-is; lower values "
                  "shrink further toward equal weighting; 0.0 reproduces equal weighting "
                  "exactly.")},
        {'label': 'Diversity penalty method', 'type': 'str', 'default': 'ridge_to_uniform',
         'choices': ['ridge_to_uniform', 'entropy'],
         'help': ("How the diversity penalty above is computed. 'ridge_to_uniform' (default) "
                  "penalises squared distance from equal weighting. 'entropy' instead "
                  "penalises the shortfall from maximum (uniform) entropy - a softer penalty "
                  "that tolerates moderate imbalance more than ridge-to-uniform.")},
        # ver4-5 R1.e/R1.f/R1.d (blueprint §3.6) - appended-only, indices
        # 9-12 (after the Requirement-5 regularisation fields above,
        # which occupy 5-8). Every default reproduces this update's own
        # new defaults; W_OPT_BAYES_RESTARTS=1 alone reproduces ver4-4's
        # exact single-search behaviour regardless of the other three.
        {'label': 'Restarts', 'type': 'int', 'default': 1,
         'help': ("Runs this many independent weight searches (from different random seeds) "
                  "and keeps the best - each restart uses its own CPU core when more than one "
                  "is available, so this is the setting that actually uses multiple CPUs for "
                  "weight optimisation. 1 (default) reproduces the original single-search "
                  "behaviour exactly.")},
        {'label': 'Batch suggestion', 'type': 'bool', 'default': True,
         'help': ("Asks for several candidate weight combinations per round instead of one at "
                  "a time, which can reduce how often the search needs to re-fit its own "
                  "internal model as the iteration budget above grows. This objective is cheap "
                  "to evaluate, so the benefit is modest; safe to leave on.")},
        {'label': 'Batch liar strategy', 'type': 'str', 'default': 'max',
         'choices': ['max', 'mean', 'believer'],
         'help': ("How a not-yet-evaluated candidate within the same batch round is "
                  "provisionally scored so the next candidate in that round isn't just a "
                  "repeat of it. 'max' (default, recommended) is the standard, cautious "
                  "choice. Only relevant when Batch suggestion above is on.")},
        {'label': 'Vectorised objective', 'type': 'bool', 'default': True,
         'help': ("Evaluates the weight-search objective as a single fast matrix computation "
                  "instead of a per-model loop - mathematically the same formula, agreeing "
                  "with the original computation to about 1 part in a trillion, but the tiny "
                  "floating-point rounding differences can occasionally nudge the search onto "
                  "a different (equally valid) answer. Turn off to reproduce the exact original "
                  "computation.")},
    ],
    # Requirements.md item 5: the same closed-form least-squares solve
    # previously only reachable as the other three methods' opt-in
    # W_OPT_ANALYTIC_SEED seed/floor candidate (models/ensemble_
    # regularization.py::analytic_simplex_weights()), now offered as its
    # own, independent weight-optimisation method - see models/
    # Analytic_least_squares.py.
    'Analytic least-squares': [
        {'label': 'Ridge regularisation', 'type': 'float', 'default': 1e-6,
         'help': ("A small conditioning term added to the least-squares solve, needed when "
                  "your selected models' predictions are highly correlated - without it, the exact solve can "
                  "become numerically unstable. The default is small enough to leave a "
                  "well-conditioned solve essentially unaffected; only raise it if you see a "
                  "'[ensemble_regularization] NOTE: ... SLSQP solve did not converge' message "
                  "in the run log.")},
        {'label': 'Naive-shrinkage alpha', 'type': 'float', 'default': 1.0,
         'help': ("Blends this method's own exact solution toward the plain equal-weighted "
                  "(naive) ensemble as an extra safety margin against overfitting the "
                  "validation split: 1.0 (the default) uses the exact solution as-is; lower "
                  "values shrink further toward the naive average; 0.0 reproduces the naive "
                  "ensemble exactly. Combined with a built-in guard that never performs worse "
                  "than the naive ensemble on the validation split.")},
    ],
}

# --------------------------------------------------------------------------- #
# Colour-vision-deficiency (CVD) safety - self-contained, no new runtime
# dependency (see the functions below for why - the natural tool for this,
# `colorspacious`, may not be installed in every EasiGP deployment
# environment, and adding it as a new hard requirement risks breaking an
# existing install; every constant/formula here is a well-known, published
# one - the Machado, Oliveira & Fernandes (2009) CVD simulation matrices
# (doi: 10.1109/TVCG.2009.113, used by colorspacious itself and many other
# CVD-simulation tools) and the standard sRGB -> CIELab colour conversion -
# reimplemented directly in plain numpy, verified to match colorspacious's
# own output to within floating-point tolerance during development).
#
# Requirement 9 (colourblind-friendly circos palette) originally applied
# this same check OFFLINE (as a one-off analysis) to redesign the default
# palette; the additional requirement here is to run the SAME check LIVE,
# every time someone adds a new colour to the circos colour map via the
# GUI, so a new colour never silently reintroduces the exact problem the
# default palette was just fixed for.
# --------------------------------------------------------------------------- #

_CVD_MATRICES = {
    # Full-severity (100) dichromacy matrices from Machado et al. (2009) -
    # applied to LINEARISED (gamma-decoded) RGB, not raw sRGB (verified
    # against colorspacious's own reference implementation).
    'protanomaly': np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998],
    ]),
    'deuteranomaly': np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.01182, 0.04294, 0.968881],
    ]),
    'tritanomaly': np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.3039],
    ]),
}

_SRGB_TO_XYZ_MATRIX = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])


def _srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.clip(np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055), 0.0, 1.0)


def _simulate_cvd_rgb1(rgb1, cvd_type):
    """One colour (0-1 RGB array) as it would appear to someone with FULL
    (severity-100) protanopia/deuteranopia/tritanopia - the worst, most
    conservative case, so a colour verified safe here stays reasonably
    distinguishable for the much more common PARTIAL forms too."""
    linear = _srgb_to_linear(np.asarray(rgb1, dtype=float))
    simulated_linear = _CVD_MATRICES[cvd_type] @ linear
    return _linear_to_srgb(simulated_linear)


def _srgb1_to_lab(rgb1):
    linear = _srgb_to_linear(np.asarray(rgb1, dtype=float))
    xyz = _SRGB_TO_XYZ_MATRIX @ linear
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn
    delta = 6 / 29

    def f(t):
        return np.where(t > delta ** 3, np.cbrt(np.clip(t, 0, None)), t / (3 * delta ** 2) + 4 / 29)

    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def _cvd_worst_case_distance(hex_a, hex_b):
    """The smallest perceptual (CIELab) distance between two colours,
    across normal vision AND all three simulated CVD types - i.e. the
    worst-case, most conservative measure of how distinguishable they
    really are. Returns None (rather than raising) if either colour
    isn't a hex string matplotlib/Streamlit's colour_picker can produce -
    color_picker only ever returns real '#rrggbb' hex, so this is mostly
    a defensive fallback."""
    try:
        rgb_a = np.array(mcolors.to_rgb(hex_a))
        rgb_b = np.array(mcolors.to_rgb(hex_b))
    except ValueError:
        return None
    dists = [np.linalg.norm(_srgb1_to_lab(rgb_a) - _srgb1_to_lab(rgb_b))]
    for cvd_type in _CVD_MATRICES:
        sim_a = _simulate_cvd_rgb1(rgb_a, cvd_type)
        sim_b = _simulate_cvd_rgb1(rgb_b, cvd_type)
        dists.append(np.linalg.norm(_srgb1_to_lab(sim_a) - _srgb1_to_lab(sim_b)))
    return float(min(dists))


def _check_new_color_cvd_safety(new_hex, existing_colormap, min_distance=15.0):
    """Requirement 9 (additional): run the SAME worst-case-CVD-distance
    check used to design the default palette itself, live, against a
    NEW colour someone is about to add - so adding one never silently
    reintroduces a near-indistinguishable pair the default palette was
    just fixed to avoid.

    Returns (is_safe, worst_name, worst_distance) - `is_safe` is False if
    ANY existing colour is closer than `min_distance` (15 - noticeably
    stricter than 'literally zero', matching roughly the weakest
    pairwise distance the redesigned default palette itself settled for,
    so this doesn't demand a higher bar than EasiGP's own palette
    already meets); `worst_name`/`worst_distance` identify the closest
    existing colour, for a clear, specific warning message. Skips
    'gpos*'/'gneg'/'gvar'/'acen'/'stalk' (standard UCSC cytoband terms,
    not something a person is likely choosing a colour to visually
    distinguish FROM in the same glance) and blueN/redN (the sequential
    quantile scale - a NEW category colour isn't meant to look like a
    single quantile step of that scale anyway) to keep the comparison
    relevant to the actual qualitative-category colours a new one would
    realistically be confused with."""
    _skip_prefixes = ('gpos', 'gneg', 'gvar', 'acen', 'stalk', 'blue', 'red')
    worst_name, worst_distance = None, float('inf')
    for name, hexval in existing_colormap.items():
        if any(name.startswith(p) for p in _skip_prefixes):
            continue
        d = _cvd_worst_case_distance(new_hex, hexval)
        if d is not None and d < worst_distance:
            worst_distance = d
            worst_name = name
    if worst_name is None:
        return True, None, None
    return worst_distance >= min_distance, worst_name, worst_distance


def _suggest_colorblind_safe_color(existing_colormap):
    """Requirement 9 (additional): suggest a NEW colour that's already
    well-separated (under simulated CVD) from every colour currently in
    the palette - rather than leaving a person to pick one by eye and
    hope for the best. Sweeps a moderate-saturation/lightness grid of
    candidate hues (matching the 'muted, not too strong' character
    Requirement 9 asked for throughout, not maximally-saturated primary
    colours) and keeps whichever candidate maximises its OWN worst-case
    distance to every existing colour - the same principle the default
    palette's own 12 colours were chosen by by, just run live against
    whatever the CURRENT palette (including anything already added)
    happens to contain. Cheap: ~60 candidates x a handful of existing
    colours, well under a second even for a large palette - deliberately
    nowhere near the scale of the 'Top interaction percentage' mistake
    (reading a whole Interaction.csv) this app already learned from."""
    _skip_prefixes = ('gpos', 'gneg', 'gvar', 'acen', 'stalk', 'blue', 'red')
    existing_hexes = [v for k, v in existing_colormap.items() if not any(k.startswith(p) for p in _skip_prefixes)]

    best_hex, best_score = '#888888', -1.0
    for hue_deg in range(0, 360, 15):
        for sat in (0.45, 0.65):
            for val in (0.55, 0.75):
                candidate = mcolors.to_hex(mcolors.hsv_to_rgb((hue_deg / 360.0, sat, val)))
                if not existing_hexes:
                    return candidate
                score = min(_cvd_worst_case_distance(candidate, h) for h in existing_hexes)
                if score > best_score:
                    best_score, best_hex = score, candidate
    return best_hex


def _trigger_suggest_new_color_cb():
    """on_click callback for the 'Suggest' button next to 'Add a new
    colour' - see _autofill_number_field's own docstring for why this
    can only ever set a flag here, not the colour_picker's value
    directly (same Streamlit rule, same reason)."""
    st.session_state['_pending_suggest_new_color'] = True


DEFAULT_CYTOBAND_COLORMAP = {
    "gpos100": "#000000", "gpos": "#000000", "gpos75": "#828282",
    "gpos66": "#A0A0A0", "gpos50": "#C8C8C8", "gpos33": "#D2D2D2",
    "gpos25": "#C8C8C8", "gvar": "#DCDCDC", "gneg": "#FFFFFF",
    "acen": "#D92F27", "stalk": "#647FA4",
    # green/brown/purple/transduction/transduction_clock/clock/
    # photoperiod/autonomous/integrator/integrator_clock/GA/aging: gene-
    # pathway/source category colours - originally picked by eye, with no
    # apparent consideration for colour-vision deficiency (CVD) or for
    # staying visually distinct from one another. Verified this directly
    # with colorspacious's Machado/Oliveira/Fernandes CVD simulation
    # model (simulating protanomaly/deuteranomaly/tritanomaly at full
    # severity, then measuring CIELab distance) rather than assuming: the
    # ORIGINAL 12 colours had multiple pairs as close as ~1-3 units apart
    # under simulated CVD (e.g. 'integrator_clock' vs 'aging' - distance
    # 1.0, essentially indistinguishable), some effectively
    # unrecoverable regardless of severity.
    #
    # Replaced with 12 colours drawn from established, independently
    # validated colourblind-safe categorical palettes (Paul Tol's muted
    # scheme, Okabe-Ito, IBM's colourblind-safe palette) - deliberately
    # NOT picked by eye or invented from scratch, since a hand-picked set
    # is exactly how the original palette ended up this way. Selected via
    # a greedy farthest-point search over that combined pool, maximising
    # the worst-case pairwise CIELab distance under simulated CVD - the
    # chosen 12 have a worst-case distance of ~13-16 (severity 100-70),
    # a roughly 13x improvement over the original's worst pairs, though
    # fitting 12 mutually-distinguishable colours into the space CVD
    # leaves usable is a genuinely hard constraint, so this is a strong
    # improvement rather than a claim of perfect separation. One colour
    # ('photoperiod', a yellow) has lower contrast against a white
    # background on its own - kept anyway since it's specifically the
    # Okabe-Ito yellow, included in that palette because yellow is one of
    # the few hues that stays clearly distinguishable across every CVD
    # type; circos_plot.py's own _visible_edge_color() (its BORDER
    # colour, not the fill) already darkens any low-contrast fill colour
    # just enough to stay visible, the same mechanism already protecting
    # the palest blueX/redX quantile levels.
    "green": "#117733", "brown": "#D55E00", "purple": "#AA4499",
    "transduction": "#88CCEE", "transduction_clock": "#785EF0", "clock": "#332288",
    "photoperiod": "#F0E442", "autonomous": "#999933", "integrator": "#CC79A7",
    "integrator_clock": "#CC6677", "GA": "#882255", "aging": "#E69F00",
    # blueN/redN (quantile-level marker-effect colours, N=0 lowest to
    # N=9 highest): each family is a straight RGB interpolation between
    # its own original level-0 and level-9 anchor colours (unchanged from
    # the original palette, so the overall look/colour family is
    # unchanged) - replacing the original, manually-picked in-between
    # levels 1-8, whose luminance decreased unevenly step to step (e.g.
    # blue's steps ranged from ~10 to ~34 out of 255 between adjacent
    # levels - some levels barely distinguishable from their neighbour,
    # others a big jump). A straight interpolation between the two fixed
    # endpoints instead gives essentially perfectly even luminance steps
    # (std. deviation of ~0.2, versus ~9-11 for the original), so the
    # progression from 0 (lightest) to 9 (darkest) reads as smooth and
    # evenly-spaced rather than clumped. This also interacts more
    # predictably with circos_plot.py's _visible_edge_color() (the
    # marker/gene region BORDER colour, matched exactly to the fill
    # whenever it's already visible enough on its own, and only adjusted
    # for the few lowest, deliberately pale levels where it isn't) -
    # with even steps, that "needs adjustting" cutoff falls at one clean,
    # predictable point in the sequence (blue3->4, red2->3) instead of
    # being at the mercy of wherever the original uneven jumps happened
    # to land.
    "blue0": "#def2ff", "blue1": "#c5e0f1", "blue2": "#adcee3",
    "blue3": "#94bcd5", "blue4": "#7caac7", "blue5": "#6397ba",
    "blue6": "#4b85ac", "blue7": "#32739e", "blue8": "#1a6190",
    "blue9": "#014f82",
    "red0": "#fce1a7", "red1": "#ebc995", "red2": "#dab182",
    "red3": "#c89970", "red4": "#b7815d", "red5": "#a6694b",
    "red6": "#955138", "red7": "#833926", "red8": "#722113",
    "red9": "#610901",
    "centromere": "#333333",
}

def _nice_round_number(x):
    """Round `x` to a visually 'clean' axis interval/bin-size value - the
    nearest number of the form {1, 2, 5} x 10^n (the standard 'nice
    numbers for graph labels' convention: 1, 2, 5, 10, 20, 50, 100, 200,
    500, ...) - used by the circos 'Suggest' buttons (Requirements 13/14)
    so a suggestion is never an awkward exact value like '187.43', which
    would look arbitrary and be hard to reason about on the plot."""
    if x is None or x <= 0:
        return x
    exponent = math.floor(math.log10(x))
    fraction = x / (10 ** exponent)
    if fraction < 1.5:
        nice_fraction = 1
    elif fraction < 3.5:
        nice_fraction = 2
    elif fraction < 7.5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return float(nice_fraction * (10 ** exponent))


def _circos_max_chromosome_length():
    """Longest chromosome's length (end - start), read from the
    'Chromosome info file path' currently set in the GUI - the basis for
    Requirements 13/14's scale/window-size suggestions (both should scale
    with how physically long the chromosomes actually are). Returns None
    if no valid chromosome info file is set (the caller shows a warning
    in that case, rather than this function raising into the page).

    Requirement (bugfix - every circos auto-fill suggestion silently
    returning blank/None, wrongly blamed on decimal coordinates):
    'population' is deliberately NOT in the expected-columns list passed
    to unify_columns_by_position() below, unlike circos_plot.py's own
    data_conversion() - that column is exactly the one the 'Chromosome/
    gene lengths are the same for every population' broadcast checkbox
    (see main_app.py's own broadcast call sites) lets a person omit
    entirely from their RAW file. That checkbox's own broadcasting only
    happens once the pipeline actually runs, though - while the person
    is still filling in the GUI, these suggestion functions read the raw
    file exactly as provided, population column or not. Requiring 4
    columns unconditionally meant every suggestion on this tab silently
    failed (caught by the blanket except below, returning None - blank
    fields, no error shown) for anyone using that checkbox as intended -
    confirmed directly against a real reported case. None of these
    functions ever read the population column's own values anyway (only
    chromosome/start/end), so requiring it here was unnecessary in the
    first place - and this file's own name/colour/source/phenotype
    columns are still required, unaffected by this."""
    chrom_path = st.session_state.get('chrom_info_path', '').strip()
    if not chrom_path or not os.path.isfile(chrom_path):
        return None
    try:
        chrom = pd.read_csv(chrom_path)
        chrom = unify_columns_by_position(
            chrom, ['chromosome', 'start', 'end'], 'chromosome info file'
        )
        return float((chrom['end'] - chrom['start']).max())
    except Exception:
        return None


def _circos_chromosome_count():
    """Number of distinct chromosomes in the 'Chromosome info file path'
    currently set in the GUI - used by the label-font-size suggestion
    (Requirement 15): more chromosomes sharing the circle means less
    room per label. Returns None if no valid file is set."""
    chrom_path = st.session_state.get('chrom_info_path', '').strip()
    if not chrom_path or not os.path.isfile(chrom_path):
        return None
    try:
        chrom = pd.read_csv(chrom_path)
        chrom = unify_columns_by_position(
            chrom, ['chromosome', 'start', 'end'], 'chromosome info file'
        )
        return int(chrom['chromosome'].nunique())
    except Exception:
        return None



def _circos_chrom_lengths_dict():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1): `{chromosome
    name: length}` for every chromosome in the 'Chromosome info file
    path' currently set in the GUI - the per-chromosome basis
    `circos_geometry.suggest_scale()`/`suggest_window()`/
    `suggest_chrom_label_size()` all need (the pre-ver4-6 helpers above
    only ever exposed the LONGEST chromosome's length, the chromosome
    COUNT, or the TOTAL length - never the full per-chromosome
    breakdown, which is what fixes RC-2/RC-3's "only the longest
    chromosome was ever checked" root cause). Returns `{}` if no valid
    chromosome info file is set - every `circos_geometry` function
    handles an empty `chrom_lengths` dict gracefully (never raises)."""
    chrom_path = st.session_state.get('chrom_info_path', '').strip()
    if not chrom_path or not os.path.isfile(chrom_path):
        return {}
    try:
        chrom = pd.read_csv(chrom_path)
        chrom = unify_columns_by_position(
            chrom, ['chromosome', 'start', 'end'], 'chromosome info file'
        )
        chrom = chrom.drop_duplicates(subset=['chromosome'])
        lengths = (chrom['end'] - chrom['start']).astype(float)
        return dict(zip(chrom['chromosome'].astype(str), lengths))
    except Exception:
        return {}


def _circos_total_chromosome_length():
    """Requirement 4: total genome size (sum of every chromosome's own
    length), read from the 'Chromosome info file path' currently set in
    the GUI - used by the marker/gene-region-widening suggestions
    (_widen_suggestion_from_positions()) instead of just the LONGEST
    chromosome's own length. See that function's own docstring for why:
    what actually determines whether a widened region renders visibly is
    how large it is relative to the WHOLE circle's circumference, which
    is proportional to the genome's TOTAL size (every chromosome's arc
    added together, roughly), not any one chromosome's length - basing
    the suggestion on the longest chromosome alone under-widens badly for
    a genome split across many chromosomes (verified empirically: a real
    26-chromosome genome's longest-chromosome-based suggestion rendered
    as completely invisible; a total-genome-based one didn't). Returns
    None if no valid chromosome info file is set."""
    chrom_path = st.session_state.get('chrom_info_path', '').strip()
    if not chrom_path or not os.path.isfile(chrom_path):
        return None
    try:
        chrom = pd.read_csv(chrom_path)
        chrom = unify_columns_by_position(
            chrom, ['chromosome', 'start', 'end'], 'chromosome info file'
        )
        chrom = chrom.drop_duplicates(subset=['chromosome'])
        return float((chrom['end'] - chrom['start']).sum())
    except Exception:
        return None


def _circos_gene_count():
    """Number of genes in the 'Gene info file path' currently set in the
    GUI - used by the gene-region-widening suggestion. Returns None if no
    valid file is set (including when left blank, since gene_info is
    optional)."""
    gene_path = st.session_state.get('gene_info_path', '').strip()
    if not gene_path or not os.path.isfile(gene_path):
        return None
    try:
        gene = pd.read_csv(gene_path)
        gene = unify_columns_by_position(
            gene, ['chromosome', 'start', 'end', 'name', 'colour', 'source', 'phenotype'],
            'gene info file'
        )
        return int(gene.shape[0])
    except Exception:
        return None


def _widen_suggestion_from_positions(position_df, max_chrom_length, file_description, total_chrom_length=None):
    """Shared core of _circos_suggest_end_adjust() (markers) and
    _circos_suggest_gene_adjust() (genes) - both widen a set of
    chromosome-positioned regions the exact same way (subtract from
    start, add to end, clamp to the region's own chromosome - see
    circos_plot.py's data_conversion()/quantile_conversion()), so both
    pick a suggested widening amount with the exact same logic.
    `position_df` needs 'chromosome' and 'start' columns.

    Requirement 4 (correction - gene locations still invisible,
    especially on large chromosomes): two real problems found by
    rendering actual test cases rather than assuming the formula worked:

    1. The chromosome-length-based candidate used to be 0.05% of the
       LONGEST chromosome alone - but what actually determines on-screen
       visibility is how large a region is relative to the WHOLE circle's
       circumference, which is proportional to the genome's TOTAL size
       (every chromosome's arc added together), not any one chromosome's
       length. For a genome split across many chromosomes (e.g. a real
       26-chromosome cotton assembly), that made the suggestion ~18x
       smaller than it needed to be - confirmed by directly rendering
       both and finding the old suggestion completely invisible. Now
       based on total_chrom_length (falls back to max_chrom_length if
       not given, e.g. for a caller that hasn't computed it).

    2. Taking the plain minimum of the chromosome-length-based and
       spacing-based candidates (to avoid neighbouring regions visually
       merging) breaks down badly for real, densely-packed gene data -
       checked directly against a real annotation file with over a
       thousand genes on a single chromosome only ~600bp apart on
       average: the spacing-based candidate came out under 200bp,
       around 0.00002% of the genome - nowhere near large enough to
       render, even with the border-colour safety net. A floor (2% of
       the length-based candidate) now guarantees a minimum visible
       size regardless of how dense the data is - a few nearby genes
       ending up visually merged into one another is a far better
       outcome than being invisible, and the border-colour mechanism
       already makes even a merged cluster distinguishable in colour
       from its neighbours.

    Returns (suggested_value, note) or (None, warning_text)."""
    if max_chrom_length is None or max_chrom_length <= 0:
        return None, "Set a valid 'Chromosome info file path' above first, so chromosome lengths can be read."

    _genome_length = total_chrom_length if total_chrom_length and total_chrom_length > 0 else max_chrom_length
    genome_length_based = _genome_length * 0.0005  # 0.05% of the TOTAL genome size

    gaps = []
    for _, _group in position_df.groupby('chromosome'):
        _starts = sorted(_group['start'].astype(float).tolist())
        if len(_starts) < 2:
            continue
        gaps.extend(b - a for a, b in zip(_starts, _starts[1:]) if b > a)

    if gaps:
        gaps.sort()
        _median_gap = gaps[len(gaps) // 2]
        spacing_based = _median_gap / 3.0
        _min_floor = genome_length_based * 0.02
        suggested = max(_min_floor, min(genome_length_based, spacing_based))
        if suggested == _min_floor and _min_floor > spacing_based:
            winner = f"a minimum visibility floor (the typical gap between neighbouring {file_description} was too small on its own to stay visible)"
        else:
            winner = f"the typical gap between neighbouring {file_description}" if spacing_based < genome_length_based \
                else "the genome's total size"
    else:
        suggested = genome_length_based
        winner = "the genome's total size"

    if suggested <= 0:
        return None, f"{file_description.capitalize()}/chromosome positions don't allow a meaningful suggestion (is everything at position 0?)."

    return _nice_round_number(suggested), winner


def _circos_suggest_end_adjust():
    """Requirement (marker regions too small to see on the circos plot):
    suggest a value for 'Edge location adjustment (END_ADJUST)' - the
    amount (in the SAME coordinate units as the marker/chromosome info
    files, e.g. bp) subtracted from each marker's start and added to its
    end before plotting, widening it into something actually visible.
    circos_plot.py's own quantile_conversion() already clamps the
    widened region to its OWN chromosome's start/end afterwards (see that
    function - this was a real bug, now fixed there, since it used to
    clamp the start side to a hardcoded 0 rather than the chromosome's
    own start), so a generous suggestion here is always safe regardless
    of how it's picked. See _widen_suggestion_from_positions() for the
    actual candidate-value logic, shared with the gene-region version of
    this same suggestion.

    Returns (suggested_value, note) - note is a short string explaining
    which of the two candidates won, or (None, warning_text) if it
    couldn't be computed (no valid marker/chromosome info file set)."""
    marker_path = st.session_state.get('marker_info_path', '').strip()
    if not marker_path or not os.path.isfile(marker_path):
        return None, "set a valid 'Marker info file path' above"
    try:
        marker = pd.read_csv(marker_path)
        marker = unify_columns_by_position(marker, ['chromosome', 'name', 'start', 'end'], 'marker info file')
    except Exception as exc:
        return None, f"could not read the marker info file: {exc}"
    return _widen_suggestion_from_positions(marker, _circos_max_chromosome_length(), "markers", _circos_total_chromosome_length())


def _circos_suggest_gene_adjust():
    """The gene-region counterpart of _circos_suggest_end_adjust() -
    suggests a value for 'Gene location adjustment (GENE_ADJUST)', the
    same kind of start/end widening but applied to gene regions (from the
    'Gene info file path' above) instead of marker regions. See
    _widen_suggestion_from_positions() for the shared candidate-value
    logic, and circos_plot.py's data_conversion() for where this value is
    actually applied (with the matching per-chromosome clamp)."""
    gene_path = st.session_state.get('gene_info_path', '').strip()
    if not gene_path or not os.path.isfile(gene_path):
        return None, "set a valid 'Gene info file path' above (leave blank if you don't use one)"
    try:
        gene = pd.read_csv(gene_path)
        gene = unify_columns_by_position(
            gene, ['chromosome', 'start', 'end', 'name', 'colour', 'source', 'phenotype'],
            'gene info file'
        )
    except Exception as exc:
        return None, f"could not read the gene info file: {exc}"
    return _widen_suggestion_from_positions(gene, _circos_max_chromosome_length(), "genes", _circos_total_chromosome_length())


def _circos_predicted_ring_labels():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1, §3.4 S4): predicts
    the ring-label list `circos_plot.py::plot()` will actually draw for
    the CURRENT GUI configuration - mirrors that module's own `MODEL`
    (`pd.unique(effect['model'])`, which in practice is every selected
    model expanded for multi-algorithm tuning via
    `models.hyperparameter_tuning.expand_model_list()`, plus `'ensemble'`
    plus every selected weighted-ensemble method's own label - see
    `genomic_prediction.py`'s finalisation block) and `gene_source`
    (`pd.unique(pop_source['source'])`, sourced here directly from the
    'Gene info file path' currently set) construction.

    A PREDICTION, not a guarantee - RK-6 (blueprint §8), accepted by
    design: the GUI cannot know in advance exactly which models will end
    up with data for every (phenotype, population) combination
    (`quantile_conversion()`'s own per-combination REMOVE list, see the
    architecture document's own note on that), and a hand-edited config
    could differ from what this function assumes entirely. This is
    exactly why Defence 2 (`circos_plot.py::plot()`'s own per-ring fit
    guard) exists independently - it is radius-and-string exact at
    render time and makes overlap impossible regardless of whether this
    prediction was right (blueprint §3.2: "Defence 1 makes the common
    case look right; Defence 2 makes overlap impossible even when the
    prediction was wrong").

    Cheap by design, like every other suggestion function on this tab:
    reads only the current GUI session state plus, optionally, one small
    CSV (the gene info file) - never genotype/marker/interaction data.

    Returns
    -------
    list of str - possibly empty (e.g. no model selected yet).
    """
    selected_models = [m for m in AVAILABLE_MODELS if st.session_state.get(f'model_selected_{m}', False)]
    if not selected_models:
        return []

    base_models = [m for m in selected_models if m != 'ensemble']
    hp_tune = {}
    for model in base_models:
        model_hp_tune = resolve_hp_tune(model)
        if model_hp_tune is not None:
            hp_tune[model] = model_hp_tune

    try:
        labels = expand_model_list(base_models, hp_tune)
    except Exception:
        # Never let a prediction failure block rendering the tab - fall
        # back to the plain (un-expanded) selection.
        labels = list(base_models)
    if 'ensemble' in selected_models and 'ensemble' not in labels:
        labels.append('ensemble')

    selected_wopt = [m for m in W_OPT_METHODS if st.session_state.get(f'wopt_selected_{m}', False)]
    labels.extend(selected_wopt)

    gene_path = st.session_state.get('gene_info_path', '').strip()
    if gene_path and os.path.isfile(gene_path):
        try:
            gene = pd.read_csv(gene_path)
            gene = unify_columns_by_position(
                gene, ['chromosome', 'start', 'end', 'name', 'colour', 'source', 'phenotype'],
                'gene info file'
            )
            labels.extend(str(s) for s in pd.unique(gene['source']))
        except Exception:
            pass  # Defence 2 still guarantees no overlap without this basis (blueprint §3.8)

    return labels


def _circos_suggest_space():
    """Requirement 5: fills 'Space between rings' automatically. Cheap by
    design (Requirement: must not be computationally heavy like the old
    'Top interaction percentage' approach was) - only reads the
    chromosome info file, the same lightweight read every other circos
    suggestion already does; never touches genotype/marker/interaction
    data. More chromosomes sharing the same 360 degrees need a smaller
    gap each to leave room for the data itself, so this scales inversely
    with chromosome count."""
    _n_chrom = _circos_chromosome_count()
    if _n_chrom is None or _n_chrom < 1:
        return None, "set a valid 'Chromosome info file path' above"
    _suggested = round(max(0.3, min(2.0, 30.0 / _n_chrom)), 2)
    return _suggested, f"for {_n_chrom} chromosome(s)"


def _circos_suggest_start_end_angle():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1, §3.4 S4): fills
    'Start angle'/'End angle' automatically - now a thin wrapper over
    `circos_geometry.seam_gap_for_labels()`, called TWICE and combined
    with `max()`:

      1. once for the chromosome NAME (`r=108`, the same radius
         `circos_plot.py` itself draws it at) - this alone is what the
         pre-ver4-6 formula measured (RC-1's own root cause: "the
         seam-gap formula measures the wrong text" - it measured ONLY
         this, never any ring label);
      2. once for every PREDICTED ring label
         (`_circos_predicted_ring_labels()`), each at its own ring's
         `r_centre` from `circos_geometry.ring_geometry()` - the text
         that actually overlaps on a many-ring plot, and the one the
         pre-ver4-6 formula never read at all.

    Widening the gap alone cannot fix the worst realistic case (blueprint
    §3.1: `GAT_biological_prior_knowledge__Bayesian` on an inner ring can
    need over 70 degrees, above any sane seam-gap cap) - so the result is
    capped at `seam_gap_max` (default 40 degrees) here, and
    `circos_plot.py::plot()`'s own Defence 2 (shrink, then truncate,
    long ring labels at render time) is what actually guarantees no
    overlap when the true requirement exceeds that cap, not this
    suggestion. The note says so explicitly when capping occurs, rather
    than silently returning a gap insufficient for the worst label.

    Returns (start, end, note) or (None, None, warning_text)."""
    _n_chrom = _circos_chromosome_count()
    if _n_chrom is None or _n_chrom < 1:
        return None, None, "set a valid 'Chromosome info file path' above"
    chrom_lengths = _circos_chrom_lengths_dict()
    _label_size, _ = _circos_suggest_label_size()
    if _label_size is None:
        _label_size = 6.0
    _chrom_name_rendered_size = max(5.0, min(8.0, _label_size * 1.8))
    _longest_chrom_name = max(chrom_lengths, key=len) if chrom_lengths else 'Chr01'
    _seam_gap_max = _num_or_default('circos_seam_gap_max', 40.0)

    _chrom_gap, _chrom_capped = circos_geometry.seam_gap_for_labels(
        [_longest_chrom_name], _chrom_name_rendered_size, [108.0], cap_deg=_seam_gap_max,
    )

    _predicted_labels = _circos_predicted_ring_labels()
    if _predicted_labels:
        _ring_layout = st.session_state.get('circos_ring_layout') or 'legacy'
        _ring_label_size = _num_or_default('circos_ring_label_size', 8.0)
        _ring_radii = [g[2] for g in circos_geometry.ring_geometry(len(_predicted_labels), layout=_ring_layout)]
        _ring_gap, _ring_capped = circos_geometry.seam_gap_for_labels(
            _predicted_labels, _ring_label_size, _ring_radii, cap_deg=_seam_gap_max,
        )
    else:
        _ring_gap, _ring_capped = 8.0, False

    _gap = max(_chrom_gap, _ring_gap)
    _start = round(_gap / 2.0, 1)
    _end = round(360.0 - _gap / 2.0, 1)
    _note = (
        f"a {_gap:g} degree seam gap for {_n_chrom} chromosome(s) at {_chrom_name_rendered_size:g}pt "
        f"chromosome-name labels and {len(_predicted_labels)} predicted ring label(s)"
    )
    if _chrom_capped or _ring_capped:
        _note += (
            f" - the true requirement exceeds the {_seam_gap_max:g} degree cap; the renderer will "
            f"shrink, and if needed truncate, the widest ring label(s) instead (see 'Ring label "
            f"sizing' in Advanced settings)"
        )
    return _start, _end, _note


def _circos_suggest_link_alpha_min():
    """Requirement 5: fills 'Minimum link opacity' automatically. Cheap by
    design - only reads the CURRENT 'Top interaction percentage' value
    already sitting in session_state (no file I/O of any kind): a higher
    percentage means more links end up drawn and so more visual clutter/
    overlap, which benefits from a higher opacity floor so even the
    faintest link still stands out from a crowded plot; a lower
    percentage (few, already-distinct links) can afford a lower floor.
    Returns (suggested_value, note) or (None, warning_text) - practically
    never None, since 'Top interaction percentage' always has SOME
    numeric value once its own field has been touched at all, but a
    fallback default is used if it hasn't."""
    _top_pct = st.session_state.get('circos_topinteraction')
    if _top_pct is None:
        _top_pct = 0.01  # this field's own documented default
    _suggested = round(max(0.1, min(0.5, 0.1 + float(_top_pct) / 100.0 * 0.4)), 2)
    return _suggested, f"based on the current 'Top interaction percentage' ({_top_pct:g}%)"



    """Number of marker-pair interactions ALREADY FOUND in this result's
    own, real, completed Interaction.csv. NO LONGER USED by the 'Top
    interaction percentage' suggestion (Requirement 4 - see
    _circos_suggest_interaction_top()'s own docstring for why: reading
    this file turned out to be a real, practical problem, not just a
    theoretical one - Interaction.csv can be large enough that reading
    it caused real memory pressure and multi-minute delays for exactly
    the people trying to use this to tune a GUI field quickly). Kept
    around only because other, unrelated parts of the app may still
    want to know whether a completed result exists."""
    result_name = st.session_state.get('result_name', '').strip()
    if not result_name:
        return None
    path = os.path.join(result_dir_path(result_name), 'Interaction.csv')
    if not os.path.isfile(path):
        return None
    try:
        return int(os.path.getsize(path) > 0) and sum(1 for _ in open(path, 'r', encoding='utf-8', errors='ignore')) - 1
    except Exception:
        return None


def _circos_suggest_interaction_top(final_marker_estimate=None):
    """Requirement 4 (correction of the correction): the PREVIOUS version
    of this suggestion read the result's own Interaction.csv to get an
    exact interaction count - technically accurate, but Interaction.csv
    can be large enough in practice that reading it caused real memory
    pressure and took a very long time, exactly defeating the purpose of
    a quick GUI suggestion. Never reads that file at all now.

    Instead, takes ONE number the person supplies directly (asked for
    via a small form when the button below is clicked - see
    _render_interaction_top_estimate_form()): their own estimate of how
    many markers will remain after LD pruning/RF marker filtering (the
    thing that can't be predicted reliably from config alone - LD/RF
    filtering survival is too data-dependent - so it's simply asked for
    directly instead of guessed at).

    Requirement 3 (correction): an earlier version also asked for an
    estimate of the total number of data points (train+valid+test
    individuals), intended as a light cap on how many links are worth
    displaying - dropped entirely, since it turned out not to actually
    affect the estimate in any way that mattered here (the target link
    count below is a fixed, sensible constant instead) and was just an
    extra number to ask for with no real payoff.

    Combined with RF's own two interaction-search settings (read via
    hparam_field_value(), same as before): 'Max markers considered for
    interaction search' caps the marker count actually used, and 'Output
    only the top N% of interactions' is RF's own additional filter,
    applied on top of that - both already correctly account for by the
    time RF hands anything to circos.

    Returns (suggested_pct, note) or (None, warning_text). Does NOT
    itself decide when to ask for the estimate - see
    _render_interaction_top_estimate_form(), which collects it and
    then calls this."""
    if final_marker_estimate is None:
        return None, "click 'Suggest' and provide your marker estimate"
    if final_marker_estimate < 2:
        return None, "need at least 2 markers to form any pair"

    _interaction_enabled = hparam_field_value('RF', 'Return marker effect for interactions')
    if not _interaction_enabled:
        return None, ("RF's own 'Return marker effect for interactions?' setting (Models tab) is "
                       "off, so no interactions will be found at all regardless of this percentage")

    _max_features_raw = hparam_field_value('RF', 'Max markers considered for interaction search')
    _rf_top_pct_raw = hparam_field_value('RF', 'Output only the top N% of interactions')

    if _max_features_raw in (None, 'all'):
        _n_markers = final_marker_estimate
        _source_note = "your marker estimate (RF's own marker cap is 'all')"
    else:
        try:
            _max_features = int(_max_features_raw)
        except (TypeError, ValueError):
            _max_features = None
        _n_markers = min(_max_features, final_marker_estimate) if _max_features else final_marker_estimate
        _source_note = "the smaller of your marker estimate and RF's own marker cap"

    _candidate_pairs = _n_markers * (_n_markers - 1) / 2
    if _rf_top_pct_raw not in (None, 'all'):
        try:
            _candidate_pairs = max(1.0, _candidate_pairs * (float(_rf_top_pct_raw) / 100.0))
        except (TypeError, ValueError):
            pass

    _target_links = 75.0  # a density that stays readable without hiding most interactions found
    _suggested_pct = round(max(0.0001, min(100.0, _target_links / _candidate_pairs * 100.0)), 4)
    return _suggested_pct, (
        f"based on {_source_note} ({int(_n_markers)} marker(s)), RF's own top-N% interaction "
        f"filter (~{int(_candidate_pairs)} candidate pair(s) left), and a target of "
        f"~{int(_target_links)} displayed link(s)"
    )


@st.dialog("Estimate 'Top interaction percentage'")
def _render_interaction_top_estimate_form():
    """Requirement 4: the small pop-up form the 'Suggest' button opens to
    collect the number _circos_suggest_interaction_top() needs -
    deliberately just one plain number input and a submit button, never
    reading any file, so opening this is always instant regardless of
    how large a real result's output would be."""
    st.write(
        "Your best estimate is enough."
    )
    marker_est = st.number_input(
        "Estimated number of markers remaining after LD pruning / RF marker filtering",
        min_value=2, value=st.session_state.get('_interaction_marker_estimate', 1000), step=100,
    )
    if st.button("Compute suggestion"):
        st.session_state['_interaction_marker_estimate'] = marker_est
        _suggested_pct, _note = _circos_suggest_interaction_top(marker_est)
        if _suggested_pct is None:
            st.session_state['_suggest_msg_interaction_top'] = ('warning', _note)
        else:
            st.session_state['circos_topinteraction'] = _suggested_pct
            st.session_state['_suggest_msg_interaction_top'] = (
                'success', f"Applied {_suggested_pct}% - {_note}."
            )
        st.rerun()


def _circos_marker_positions_by_chrom():
    """`{chromosome: [start positions, ...]}` from the 'Marker info file
    path' currently set in the GUI - the per-chromosome basis
    `circos_geometry.suggest_window()`'s own `marker_positions` argument
    needs (inter-marker gaps are only meaningful WITHIN a chromosome).
    Returns `None` if no valid marker info file is set - callers fall
    back to a length-only window suggestion in that case, exactly as
    before this file's marker-spacing suggestion existed."""
    marker_path = st.session_state.get('marker_info_path', '').strip()
    if not marker_path or not os.path.isfile(marker_path):
        return None
    try:
        marker = pd.read_csv(marker_path)
        marker = unify_columns_by_position(marker, ['chromosome', 'name', 'start', 'end'], 'marker info file')
        positions = {}
        for chrom, group in marker.groupby('chromosome'):
            positions[str(chrom)] = group['start'].astype(float).tolist()
        return positions
    except Exception:
        return None


def _circos_suggest_label_size():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1, §3.4 S4): fills
    'Label font size' (the chromosome NAME font size) automatically -
    now a thin wrapper over `circos_geometry.suggest_chrom_label_size()`
    (replaces the pre-ver4-6 `150 / n_chrom` heuristic, RC-5), which
    fits the longest chromosome NAME against the NARROWEST sector's own
    actual arc length at the current seam gap/inter-sector spacing,
    rather than reading chromosome COUNT alone.

    `start`/`end` are read from whatever is CURRENTLY set (an earlier
    auto-fill, or a manual override), falling back to a full-circle
    placeholder only on this session's very first render, before either
    has a value yet - this label-size suggestion and
    `_circos_suggest_start_end_angle()`'s own seam-gap suggestion are
    mutually dependent (the gap needs a label size to size the
    chromosome-name's own footprint; this needs a gap to know the
    narrowest sector's own arc length) and converge to a stable pair
    within one or two Streamlit reruns, same as any other auto-filling
    field pair in this file.

    Returns (suggested_size, note) or (None, warning_text)."""
    _n_chrom = _circos_chromosome_count()
    if _n_chrom is None or _n_chrom < 1:
        return None, "set a valid 'Chromosome info file path' above"
    chrom_lengths = _circos_chrom_lengths_dict()
    if not chrom_lengths:
        return None, "set a valid 'Chromosome info file path' above"
    _start = _num_or_default('circos_start', 0.0)
    _end = _num_or_default('circos_end', 360.0)
    _space = _num_or_default('circos_space', 1.0)
    _suggested_size = circos_geometry.suggest_chrom_label_size(
        list(chrom_lengths.keys()), chrom_lengths, start=_start, end=_end, space=_space,
    )
    return _suggested_size, f"for {_n_chrom} chromosome(s), fit to the narrowest sector's own arc"


def _circos_suggest_circos_scale():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1, §3.4 S4): fills
    'Scale' automatically - now a thin wrapper over
    `circos_geometry.suggest_scale()`, which checks tick-label fit
    against EVERY chromosome's own sector arc (RC-2's own root cause:
    the pre-ver4-6 formula, `max_len / 10`, read chromosome length alone
    and never checked whether the resulting tick labels could actually
    fit anywhere). Returns (suggested_value, note) or (None,
    warning_text)."""
    chrom_lengths = _circos_chrom_lengths_dict()
    if not chrom_lengths:
        return None, "set a valid 'Chromosome info file path' above"
    _max_len = max(chrom_lengths.values())
    _label_size, _ = _circos_suggest_label_size()
    if _label_size is None:
        _label_size = 6.0
    _tick_label_size = max(5.0, min(8.0, _label_size * 1.8))
    _start = _num_or_default('circos_start', 0.0)
    _end = _num_or_default('circos_end', 360.0)
    _space = _num_or_default('circos_space', 1.0)
    _suggested = circos_geometry.suggest_scale(
        chrom_lengths, start=_start, end=_end, space=_space, tick_label_size_pt=_tick_label_size,
    )
    return _suggested, f"fit to every chromosome's own sector (longest: {_max_len:g})"


def _circos_suggest_window_size():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1, §3.4 S4): fills
    'Averaging window size (WINDOW)' automatically - now a thin wrapper
    over `circos_geometry.suggest_window()`, which fixes RC-3 (the
    pre-ver4-6 ceiling was anchored on the LONGEST chromosome, so a
    chromosome a quarter that length received only ~1 bin at the
    ceiling - the new ceiling is anchored on the SHORTEST chromosome
    instead) and RC-4 (rounds DOWN, never above, its own ceiling - the
    pre-ver4-6 nearest-rounding could round a clamped value up past the
    very ceiling meant to bound it).

    Returns (suggested_value, note) or (None, warning_text)."""
    chrom_lengths = _circos_chrom_lengths_dict()
    if not chrom_lengths:
        return None, "set a valid 'Chromosome info file path' above"
    marker_positions = _circos_marker_positions_by_chrom()
    _window, _note = circos_geometry.suggest_window(chrom_lengths, marker_positions)
    if marker_positions is None:
        _note += (" Set a 'Marker info file path' above too for a suggestion based on actual "
                   "marker spacing instead.")
    return _window, _note


def _circos_suggest_ring_label_size():
    """Update ID ver4-6, R2 (blueprint §3.2 Defence 1, §5.1): fills
    'Ring label font size' automatically - defaults to 8.0 (this
    codebase's own pre-ver4-6 hard-coded ring-label size, so an untouched
    field renders identically to before this option existed) unless the
    predicted ring list, at the current seam-gap cap, would need a
    smaller size to keep every predicted ring label's own required seam
    gap within that cap - in which case this suggests the largest size
    that still manages that, so the seam-gap suggestion above stays
    modest by default instead of climbing toward (or past)
    `seam_gap_max`.

    Purely a cosmetic default: `circos_plot.py::plot()`'s own Defence 2
    guarantees no ring-label overlap regardless of what this suggests,
    or whether a person overrides it."""
    predicted_labels = _circos_predicted_ring_labels()
    if not predicted_labels:
        return 8.0, "no models/gene sources selected yet - defaulting to 8pt"
    _ring_layout = st.session_state.get('circos_ring_layout') or 'legacy'
    _seam_gap_max = _num_or_default('circos_seam_gap_max', 40.0)
    _radii = [g[2] for g in circos_geometry.ring_geometry(len(predicted_labels), layout=_ring_layout)]
    _widest_label = max(predicted_labels, key=len)
    _innermost_radius = _radii[-1]
    _fitted = circos_geometry.fit_label_size(
        _widest_label, _innermost_radius, _seam_gap_max, size_pt=8.0, min_size_pt=4.0,
    )
    return round(_fitted, 1), f"fit to the widest predicted ring label ('{_widest_label}') at the innermost ring"


# --------------------------------------------------------------------
# Additional_requirements.md, Requirements 1/2: 'Suggest' buttons for LD
# pruning's Window size/r^2 threshold, and for the LD decay plot's Max
# distance shown/Distance bin width/Max marker pairs per chromosome.
#
# Deliberately kept as explicit-click BUTTONS (via the same on_click ->
# pending-flag -> apply-before-widget-draw -> _show_suggest_message()
# pattern already used for the colourblind-safe colour suggestion and
# 'Top interaction percentage' above), NOT the auto-fill-on-every-rerun
# pattern the circos tab's own suggestions moved to (_autofill_number_field) -
# unlike those, a real LD-pruning/LD-decay suggestion should only ever be
# computed and applied when the person actually asks for it, never
# silently overwritten on every rerun (e.g. after they've already
# deliberately tuned these against an LD decay plot they generated).
# --------------------------------------------------------------------

def _ld_read_snp_info_cheap():
    """The SNP info file currently configured for LD pruning (Tab 2),
    read via the exact same Preprocess.LD_decay_plot.resolve_snp_info_for_decay()
    reader/column-unification LD pruning and the LD decay plot feature
    themselves already use for this file - reused here rather than
    re-implemented, so a suggestion can never disagree with what CHR/POS/
    CM those two actually see for the same file. Returns None if no
    valid path is currently set."""
    snp_info_path = st.session_state.get('ld_snp_info_path', '').strip()
    if not snp_info_path or not os.path.isfile(snp_info_path):
        return None
    try:
        return resolve_snp_info_for_decay({'snp_info': snp_info_path})
    except Exception:
        return None


def _ld_marker_positions_by_chrom(window_unit):
    """chromosome -> sorted list of marker positions, in whichever unit
    `window_unit` itself measures in ('POS' for 'kb', 'CM' for 'cm') -
    built from the LD-pruning SNP info file (CSV genotype input) or the
    .bim file (PLINK genotype input, via the same
    Preprocess.plink_io.read_bim_marker_info() genomic_prediction.py
    itself calls to get this - never touches the, potentially far
    larger, .bed genotype matrix).

    Returns None if no source is configured yet at all (caller falls
    back to a fixed default with a note explaining what's missing), or
    {} if a source WAS read but doesn't carry the column this
    `window_unit` needs (a .bim file has no genetic-distance equivalent,
    same requirement Preprocess.LD_pruning.ld_prune_snps() itself has for
    'cm' - see its module docstring) - callers treat {} the same way as
    None (fall back), just with a slightly different note."""
    if window_unit not in ('kb', 'cm'):
        # 'variants' windows are defined purely by column order/count, not
        # by any physical/genetic position - there is nothing here to read
        # a spacing-based suggestion FROM (see ld_prune_snps()'s own
        # module docstring). Treated as "no map available" rather than an
        # error, same as every other missing-input case in this function.
        return {}

    genotype_format = st.session_state.get('genotype_format', 'CSV file')
    pos_col = 'CM' if window_unit == 'cm' else 'POS'

    if genotype_format != 'CSV file' and window_unit != 'cm':
        stem = st.session_state.get('genotype_plink_stem', '').strip()
        bim_path = f"{stem}.bim" if stem else ''
        if not stem or not os.path.isfile(bim_path):
            return None
        try:
            bim = read_bim_marker_info(bim_path)  # chromosome, name, start, end
        except Exception:
            return None
        chrom_series, pos_series = bim['chromosome'], bim['start']
    else:
        snp_info = _ld_read_snp_info_cheap()
        if snp_info is None:
            return None
        if 'CHR' not in snp_info.columns or pos_col not in snp_info.columns:
            return {}
        chrom_series, pos_series = snp_info['CHR'], snp_info[pos_col]

    # Vectorised (pandas-level, not a per-row Python loop) build of
    # chromosome -> sorted position array - the same "group, coerce,
    # sort" approach _circos_suggest_window_size() above already uses for
    # its own marker-info file, kept consistent here rather than
    # re-inventing a slower row-by-row version: pd.to_numeric(...,
    # errors='coerce') converts the whole column in one C-level pass,
    # non-numeric/missing positions become NaN and are dropped once,
    # rather than caught one exception at a time.
    frame = pd.DataFrame({
        'chromosome': chrom_series.astype(str).values,
        'position': pd.to_numeric(pos_series, errors='coerce').values,
    }).dropna(subset=['position'])

    by_chrom = {}
    for chrom, group in frame.groupby('chromosome', sort=False):
        by_chrom[chrom] = sorted(group['position'].tolist())
    return by_chrom


def _ld_sample_count():
    """A cheap read of how many individuals are in the currently
    configured dataset - used only to gauge how noisy r^2 estimates from
    this dataset's (eventual) training split are likely to be, for the
    r^2-threshold suggestion below. Never touches the genotype matrix
    itself, and specifically avoids asking pandas to PARSE a wide
    genotype CSV (a real genotype file can have thousands of marker
    columns - even with usecols=[0], the C parser still has to tokenise
    every field on every line to find the column boundaries, so that's
    not actually cheap): the phenotype file (a handful of trait columns)
    is read with pandas when available, since it's genuinely small; the
    genotype CSV, if that's all that's configured, is instead just
    line-counted (no per-field parsing at all) via a plain file
    iteration; PLINK input reads the .fam file's row count the same way,
    via read_fam_iids(). Returns None if nothing usable is configured
    yet."""
    try:
        if st.session_state.get('genotype_format', 'CSV file') == 'CSV file':
            pheno_path = st.session_state.get('phenotype_path', '').strip()
            if pheno_path and os.path.isfile(pheno_path):
                return int(pd.read_csv(pheno_path, usecols=[0]).shape[0])
            geno_path = st.session_state.get('genotype_path', '').strip()
            if not geno_path or not os.path.isfile(geno_path):
                return None
            with open(geno_path, 'r', encoding='utf-8-sig', errors='ignore') as f:
                n_lines = sum(1 for _ in f)
            return max(0, n_lines - 1)  # minus the header row
        else:
            stem = st.session_state.get('genotype_plink_stem', '').strip()
            fam_path = f"{stem}.fam" if stem else ''
            if not stem or not os.path.isfile(fam_path):
                return None
            return len(read_fam_iids(fam_path))
    except Exception:
        return None


def _ld_suggest_window_and_r2():
    """Requirement 1: suggest a Window size and r^2 threshold for LD
    pruning (Tab 2).

    Window size targets a classic ~50 markers per pruning window (the
    long-standing PLINK tutorial default of `--indep-pairwise 50 5 0.2`),
    based on THIS dataset's own median marker spacing - read from the
    already-configured SNP info file ('kb'/'cm') or .bim file ('kb' only,
    PLINK input; see _ld_marker_positions_by_chrom()) - falling back to a
    fixed window (with a note explaining why) when no map is available
    at all: 'variants' windows never need one, and a real map is
    genuinely required for 'kb'/'cm' (same requirement
    Preprocess.LD_pruning.ld_prune_snps() itself has).

    r^2 threshold: unlike a GWAS QC pipeline (which typically prunes
    aggressively, r^2~0.1-0.2, since it only needs a near-independent
    marker set for e.g. population-structure/PCA correction), a genomic-
    prediction marker pool benefits from staying denser - a small-effect
    QTL can still be usefully tagged by a marker a GWAS pipeline would
    happily discard. This suggestion instead scales mainly with how
    RELIABLE an r^2 estimate from this dataset's sample size is likely
    to be: with few training individuals, r^2 between two markers is
    itself a noisy statistic, so pruning on a strict (low) threshold
    risks discarding markers based on sampling noise rather than real
    LD - a more lenient (higher) threshold is suggested instead as the
    individual count drops. Sample size is read from the phenotype/
    genotype CSV or .fam file (_ld_sample_count()) - never the genotype
    matrix itself.

    Returns (window_value, r2_value, note) - note explains what the
    suggestion was based on (or what's missing for a more informed one).
    Always returns real numbers, never None - a documented fallback
    default is used whenever the ideal inputs aren't available yet."""
    window_unit = st.session_state.get('ld_window_unit', 'kb')
    TARGET_MARKERS_PER_WINDOW = 50  # the classic PLINK-tutorial default (50 5 0.2)
    FALLBACK_WINDOW = {'kb': 500.0, 'cm': 5.0, 'variants': 50.0}

    by_chrom = _ld_marker_positions_by_chrom(window_unit)
    note_bits = []
    window_value = None

    if by_chrom:
        gaps = []
        for positions in by_chrom.values():
            gaps.extend(b - a for a, b in zip(positions, positions[1:]) if b > a)
        if gaps:
            gaps.sort()
            median_gap = gaps[len(gaps) // 2]
            window_value = _nice_round_number(median_gap * TARGET_MARKERS_PER_WINDOW)
            note_bits.append(
                f"window \u2248{TARGET_MARKERS_PER_WINDOW} markers wide based on median marker "
                f"spacing ({median_gap:g} {_UNIT_LABEL_FOR_GUI.get(window_unit, window_unit)}) "
                f"across {len(by_chrom)} chromosome(s)"
            )

    if window_value is None:
        window_value = FALLBACK_WINDOW.get(window_unit, 50.0)
        if window_unit == 'variants':
            note_bits.append(f"a fixed {int(window_value)}-variant window (this unit never needs a map)")
        else:
            note_bits.append(
                f"window left at a fixed fallback ({window_value:g} "
                f"{_UNIT_LABEL_FOR_GUI.get(window_unit, window_unit)}) - set a valid SNP info file "
                f"path above for a suggestion based on this dataset's own marker spacing instead"
            )

    n_samples = _ld_sample_count()
    N_NOISY, N_RELIABLE = 100, 1000
    R2_LENIENT, R2_STRICT = 0.6, 0.2
    if n_samples is None:
        r2_value = 0.2
        note_bits.append(
            "r\u00b2 threshold left at the conventional 0.2 - set a genotype/phenotype file path "
            "above for a sample-size-aware suggestion instead"
        )
    else:
        n_clamped = max(N_NOISY, min(N_RELIABLE, n_samples))
        frac = (n_clamped - N_NOISY) / (N_RELIABLE - N_NOISY)
        r2_value = round(R2_LENIENT - frac * (R2_LENIENT - R2_STRICT), 2)
        note_bits.append(
            f"r\u00b2 threshold {r2_value:g} based on {n_samples} individual(s) in the dataset "
            f"(fewer individuals \u2192 noisier r\u00b2 estimates \u2192 a more lenient threshold)"
        )

    return window_value, r2_value, "; ".join(note_bits)


def _trigger_suggest_ld_window_r2_cb():
    """on_click callback for the 'Suggest window size / r\u00b2 threshold'
    button - see _autofill_number_field's docstring for why this can
    only ever set a pending flag here, not the number_input values
    directly (the widgets with those keys have already been instantiated
    earlier in THIS script run by the time the button below them is
    clicked; only a rerun, applying the flag before they're drawn again,
    can update them)."""
    st.session_state['_pending_suggest_ld_window_r2'] = True


def _ld_decay_suggest_params():
    """Requirement 2: suggest 'Max distance shown', 'Distance bin
    width', and 'Max marker pairs per chromosome' for the LD decay plot
    (Tab 2, nested under LD pruning).

    Max distance shown: an LD decay plot's whole purpose is to show
    where r^2 falls off enough to justify the LD-pruning Window size
    above, so this is suggested as a generous multiple (10x) of the
    CURRENTLY CONFIGURED LD-pruning window size - wide enough to see the
    curve flatten out well past the pruning window, not just up to it -
    falling back to Preprocess.LD_decay_plot.DEFAULT_MAX_DISTANCE[unit]
    if no window size is set yet. Capped at this dataset's own longest
    mapped chromosome span (from the same SNP info/.bim source
    _ld_suggest_window_and_r2() reads), so the suggestion is never wider
    than the data can actually support.

    Distance bin width: targets ~60 points across that range - enough to
    see the decay curve's shape without an excessively slow per-scenario
    computation (more bins isn't free: see compute_ld_decay_data()'s own
    max_pairs_per_chr guard).

    Max marker pairs per chromosome: scaled down from the module's own
    default (Preprocess.LD_decay_plot.DEFAULT_MAX_PAIRS_PER_CHR) when
    many chromosomes are present, so the TOTAL pairs sampled across the
    whole dataset (chromosomes x this cap) stays roughly constant rather
    than growing linearly with chromosome count - keeping this
    diagnostic step's total runtime comparable regardless of how many
    chromosomes the dataset happens to have.

    Returns (max_distance, bin_width, max_pairs, note). Always returns
    real numbers, never None - falls back to
    Preprocess.LD_decay_plot's own documented defaults, with the note
    explaining the fallback."""
    window_unit = st.session_state.get('ld_window_unit', 'kb')
    unit_label = _UNIT_LABEL_FOR_GUI.get(window_unit, window_unit)
    note_bits = []

    by_chrom = _ld_marker_positions_by_chrom(window_unit)
    max_extent, n_chrom = None, None
    if by_chrom:
        n_chrom = len(by_chrom)
        extents = [positions[-1] - positions[0] for positions in by_chrom.values() if len(positions) >= 2]
        if extents:
            max_extent = max(extents)

    configured_window = st.session_state.get('ld_window')
    try:
        configured_window = float(configured_window) if configured_window not in (None, '') else None
    except (TypeError, ValueError):
        configured_window = None

    if configured_window:
        max_distance = configured_window * 10.0
        note_bits.append(f"10x the configured LD-pruning window size ({configured_window:g} {unit_label})")
    else:
        max_distance = LD_DECAY_DEFAULT_MAX_DISTANCE.get(window_unit, 1000.0)
        note_bits.append(
            f"the module default ({max_distance:g} {unit_label}) - set (or suggest) a Window size "
            f"above for a suggestion scaled to it instead"
        )

    if max_extent and max_extent > 0 and max_distance > max_extent:
        max_distance = max_extent
        note_bits.append(f"capped to this dataset's longest mapped chromosome span ({max_extent:g} {unit_label})")
    max_distance = _nice_round_number(max_distance)

    TARGET_BINS = 60
    _bin_floor = {'kb': 0.1, 'cm': 0.01, 'variants': 1.0}.get(window_unit, 0.01)
    bin_width = max(_nice_round_number(max_distance / TARGET_BINS), _bin_floor)
    note_bits.append(f"bin width for \u2248{TARGET_BINS} points across that range")

    TARGET_TOTAL_PAIRS = 200000
    if n_chrom:
        max_pairs = int(max(2000, min(
            LD_DECAY_DEFAULT_MAX_PAIRS_PER_CHR, round(TARGET_TOTAL_PAIRS / n_chrom, -2)
        )))
        note_bits.append(
            f"max marker pairs/chromosome scaled for {n_chrom} chromosome(s) "
            f"(targeting \u2248{TARGET_TOTAL_PAIRS:,} pairs total)"
        )
    else:
        max_pairs = LD_DECAY_DEFAULT_MAX_PAIRS_PER_CHR
        note_bits.append(
            f"max marker pairs/chromosome left at the module default "
            f"({LD_DECAY_DEFAULT_MAX_PAIRS_PER_CHR:,}) - set a SNP info/.bim source above for a "
            f"chromosome-count-aware suggestion instead"
        )

    return max_distance, bin_width, max_pairs, "; ".join(note_bits)


def _trigger_suggest_ld_decay_params_cb():
    """on_click callback for the LD decay plot's 'Suggest' button - see
    _trigger_suggest_ld_window_r2_cb() above for why this can only set a
    pending flag, applied on the rerun it triggers, rather than writing
    the number_input values directly."""
    st.session_state['_pending_suggest_ld_decay_params'] = True


def _qtl_suggest_window():
    """ver4-4 R6: suggest a QTL_WINDOW size (Tab 5), preferring a
    biologically-grounded, LD-decay-derived value over a purely
    spacing-derived one - two tiers, best available first (ver4-4 Design
    Blueprint S2.6.2).

    Tier 1 (preferred). If this RESULT_NAME already has a
    Result/<RESULT_NAME>/LD_decay_plots/LD_decay_average_all.csv (written
    by Preprocess.LD_decay_plot.average_and_plot_ld_decay() - i.e. an LD
    decay plot has already been generated for this exact dataset),
    suggest the distance at which the averaged r^2 curve first drops
    below 0.2 (the same classic PLINK-tutorial threshold
    _ld_suggest_window_and_r2() above already anchors on). This is the
    biologically correct answer: a marker within LD-decay distance of a
    QTL is the marker that would actually tag it.

    Only 'kb' (physical-distance) window_unit rows are used -
    marker_info.csv's own start/end columns (what QTL_WINDOW is
    ultimately widened against, via pipeline_utils.markers_in_windows())
    are physical genome positions, so a genetic-distance ('cm') decay
    curve isn't directly comparable and is skipped rather than silently
    mixed in. Every population present is combined into ONE curve
    (n_pairs-weighted - the same convention
    Preprocess.LD_decay_plot.average_ld_decay_data() itself already uses
    to combine separate scenarios), since QTL_WINDOW is a single,
    dataset-wide setting here, not a per-population one.

    Tier 2 (fallback). `k=25` (half the classic 50-marker LD-pruning
    window default - QTL tagging needs a materially smaller neighbourhood
    than a whole pruning window) times this dataset's own median marker
    spacing, read from the SAME `_ld_marker_positions_by_chrom('kb')`
    helper `_ld_suggest_window_and_r2()` already uses - physical/'kb'
    positions only, matching the QTL/marker_info coordinate space.

    A final fixed fallback (500, the same order of magnitude as
    `_ld_suggest_window_and_r2()`'s own 'kb' fallback) is used only when
    NEITHER tier has anything to work with yet (no LD decay data AND no
    marker position map configured) - genuinely uncommon, since a marker
    position source is normally already configured for LD pruning/circos
    by the time someone reaches this tab.

    Returns
    -------
    (window_value, note) : the suggested window and a note stating which
    tier produced it (the same "value + explanatory note" shape every
    other Suggest helper in this file returns). Always returns a real
    number, never None.
    """
    result_name = st.session_state.get('result_name', '').strip()
    R2_THRESHOLD = 0.2

    if result_name:
        decay_csv_path = os.path.join(
            result_dir_path(result_name), 'LD_decay_plots', 'LD_decay_average_all.csv'
        )
        if os.path.isfile(decay_csv_path):
            try:
                decay = pd.read_csv(decay_csv_path)
            except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
                decay = None
            _required_cols = {'window_unit', 'population', 'bin_mid', 'mean_r2', 'n_pairs'}
            if decay is not None and _required_cols.issubset(decay.columns):
                physical = decay[decay['window_unit'] == 'kb']
                physical = physical.dropna(subset=['bin_mid', 'mean_r2', 'n_pairs'])
                physical = physical[physical['n_pairs'] > 0]
                if physical.shape[0] > 0:
                    physical = physical.copy()
                    physical['_weighted_r2'] = physical['mean_r2'] * physical['n_pairs']
                    grouped = physical.groupby('bin_mid', as_index=False).agg(
                        _weighted_r2_sum=('_weighted_r2', 'sum'),
                        n_pairs=('n_pairs', 'sum'),
                    )
                    grouped['mean_r2'] = grouped['_weighted_r2_sum'] / grouped['n_pairs']
                    grouped = grouped.sort_values('bin_mid').reset_index(drop=True)
                    _below_threshold = grouped[grouped['mean_r2'] < R2_THRESHOLD]
                    if _below_threshold.shape[0] > 0:
                        window_value = float(_below_threshold['bin_mid'].iloc[0])
                        _n_pop = physical['population'].nunique()
                        return window_value, (
                            f"LD-decay-derived: averaged r\u00b2 (physical/'kb' distance, "
                            f"combined across {_n_pop} population(s)) first drops below "
                            f"{R2_THRESHOLD:g} at {window_value:g} - from this RESULT_NAME's own "
                            f"LD_decay_average_all.csv (Tab 2)"
                        )

    # Tier 2 - spacing-derived fallback, same helper _ld_suggest_window_and_r2() uses.
    K_MULTIPLIER = 25
    by_chrom = _ld_marker_positions_by_chrom('kb')
    if by_chrom:
        gaps = []
        for positions in by_chrom.values():
            gaps.extend(b - a for a, b in zip(positions, positions[1:]) if b > a)
        if gaps:
            gaps.sort()
            median_gap = gaps[len(gaps) // 2]
            window_value = _nice_round_number(median_gap * K_MULTIPLIER)
            return window_value, (
                f"spacing-derived fallback: no usable LD-decay data found for this RESULT_NAME "
                f"yet (run an LD decay plot - Tab 2 - first for the biologically-grounded "
                f"suggestion instead) - {K_MULTIPLIER}x median marker spacing "
                f"({median_gap:g}) across {len(by_chrom)} chromosome(s)"
            )

    FALLBACK_WINDOW = 500.0
    return FALLBACK_WINDOW, (
        f"fixed fallback ({FALLBACK_WINDOW:g}): neither LD-decay data nor a marker position map "
        f"is available yet - configure a genotype/SNP-info source (Tab 2) or generate an LD "
        f"decay plot first for a data-driven suggestion"
    )


def _trigger_suggest_qtl_window_cb():
    """on_click callback for the 'Suggest QTL window' button - see
    _trigger_suggest_ld_window_r2_cb() above for why this can only set a
    pending flag, applied on the rerun it triggers, rather than writing
    the 'qtl_window' number_input value directly."""
    st.session_state['_pending_suggest_qtl_window'] = True


def _show_suggest_message(state_key):
    """Pop and display the ('warning'|'success', text) message a Suggest
    callback (above) left in session_state, if any - shown ONCE, right
    after that button is clicked, then gone on the next rerun. Kept as
    its own small helper since every Suggest button needs the exact same
    'show it once, then it's gone' display logic."""
    _msg = st.session_state.pop(state_key, None)
    if _msg:
        getattr(st, _msg[0])(_msg[1])


def _autofill_number_field(key, compute_fn):
    """Requirement (auto-fill instead of a Suggest button): fills a
    number_input field (given by `key`, drawn with value=None so it
    starts genuinely blank) with a computed suggestion as soon as enough
    information is available to compute one - replacing the old
    click-a-'Suggest'-button pattern for every circos field EXCEPT 'Top
    interaction percentage' (kept as a button - see
    _circos_suggest_interaction_top()'s own docstring for why that one
    specifically still needs an explicit action rather than running on
    every rerun).

    MUST be called BEFORE the number_input widget with this exact `key`
    is drawn in the SAME script run - like every other 'write to
    session_state for a not-yet-drawn widget' pattern in this file,
    Streamlit forbids doing so afterward.

    Respects a manual edit: if the field currently holds a value that
    does NOT match the value this function itself last wrote there, the
    person has since typed something in themselves - left completely
    alone from that point on, even if the computed suggestion changes
    later (e.g. they raised or lowered chromosome count). A blank (None)
    field, or one still holding exactly what was last auto-filled here,
    stays in sync with the live suggestion on every rerun - which is
    what actually makes this 'automatic': change an upstream input
    (marker/chromosome file, an RF hyperparameter, etc.) and the field
    updates itself on the very next rerun, no click needed.

    Returns (value, note) exactly as `compute_fn` did, so the caller can
    also show an explanatory caption next to the field.
    """
    autofill_key = f'_{key}_last_autofill'
    value, note = compute_fn()
    current = st.session_state.get(key)
    last_autofill = st.session_state.get(autofill_key)

    if current is None or current == last_autofill:
        if value is not None and value != current:
            st.session_state[key] = value
            st.session_state[autofill_key] = value
        elif value is None and current is not None:
            # The inputs needed to compute a suggestion are no longer
            # available (e.g. the chromosome info file path was cleared)
            # - clear the field back to blank too, rather than leaving a
            # stale auto-filled number behind with nothing backing it.
            st.session_state[key] = None
            st.session_state[autofill_key] = None
    return value, note


def _num_or_default(key, default):
    """float(st.session_state[key]) if it holds a real number, else
    float(default) - handles the auto-filled circos fields correctly
    (Requirement 3/5): they're drawn with value=None so they start
    blank, and st.session_state.get(key, default)'s own fallback ONLY
    ever applies when the key is completely ABSENT, not when it's
    PRESENT but None (which is exactly the state of a field nobody
    (person or auto-fill) has put a number into yet) - using that
    pattern directly would try float(None) and crash. Used throughout
    gather_config() for every field that can now be blank."""
    v = st.session_state.get(key)
    return float(v) if v is not None else float(default)


CIRCOS_HELP = {
    'space': (
        'A circos plot draws each chromosome as a ring/track around a circle. '
        'This sets the empty gap between neighbouring rings - larger values spread '
        'them further apart.'
    ),
    'start': (
        'Circos plots are drawn as segments of a circle rather than always a full '
        '360 degrees. This sets where that segment begins (in degrees).'
        'This allows adding labels for each ring.'
    ),
    'end': (
        'Where the plotted segment of the circle ends (in degrees). Together with '
        "'Start angle', this controls how much of a full circle is used - e.g. "
        'leaving a gap for labels.'
    ),
    'link_alpha_min': (
        "Each link's opacity encodes the strength of that marker-pair interaction - a "
        "higher value means a more solid/opaque line, so a HIGHER minimum here puts more "
        "visual emphasis on strong interactions specifically (weak ones fade out more). "
        "The strongest link on a given plot is always fully opaque (1.0); this setting "
        "controls how faint the WEAKEST displayed link is allowed to get, so even it "
        "stays at least somewhat visible rather than disappearing entirely. 0 = the "
        "weakest link can be fully transparent; 1 = every link is drawn fully opaque "
        "regardless of its value (no visual distinction by strength at all)."
    ),
    'interaction_top': (
        'Only the strongest interactions are drawn as links, to keep the plot '
        'readable - this sets what fraction to keep, e.g. 0.1 keeps only the '
        'top 0.1% strongest marker-pair interactions.'
    ),
    'interaction_top_count': (
        'Only the strongest interactions are drawn as links, to keep the plot '
        'readable - this sets exactly how many of the strongest marker-pair '
        'interactions to keep (fewer if fewer candidate pairs exist), instead of '
        'a percentage.'
    ),
    'label_size': 'Font size used for the interval ticks around the plot.',
    'scale': (
        'Adjust the intervals between ticks on a circos plot. '
        'A larger value will make the intervals wider.'
    ),
    'end_adjust': (
        'Widens each marker\'s highlighted region on the plot - subtracted from its '
        'start and added to its end - for when a region is too small to actually see '
        'against the plot\'s scale. 0 (no widening) uses each marker\'s exact position. '
        'Always kept within that marker\'s own chromosome, however large a value is set.'
    ),
    'gene_adjust': (
        'Widens each gene region on the plot (from the "Gene info '
        'file path" above). 0 (no widening) uses each gene\'s '
        'exact position. Only relevant when a gene info file is set.'
    ),
    'window': (
        'Assign 0 if you do not wish to introduce a window to average the effects '
        'in each window interval. Otherwise, assign a window size here. If the '
        'circos plot does not show with WINDOW > 0, the size of the window needs '
        'to be larger.'
    ),
    'ascending': (
        'Determines the order of genomic marker effect mapping when WINDOW = 0. '
        'Marker effects are mapped in order from the start to the end of the '
        'generated marker effect tsv files, so if two marker regions overlap, the '
        'first is overwritten by the second. True = stronger marker effects are '
        'emphasised. False = weaker marker effects are emphasised. '
        'None = the original marker effect order in the generated tsv files is used.'
    ),
    # Update ID ver4-6, R2 (blueprint §5.1).
    'ring_label_size': (
        "Font size for the labels drawn on each marker-effect/gene-source ring itself "
        "(e.g. 'RF', 'ensemble', 'GAT_biological_prior_knowledge') - separate from the "
        "chromosome NAME font size above. Auto-filled once at least one model is "
        "selected; 8pt is this tool's traditional default."
    ),
    'ring_label_fit': (
        "What to do when a ring label doesn't fit within the seam gap at its own "
        "ring's radius (inner rings need more angular room per character than outer "
        "ones for the same text): 'off' never resizes or shortens a label (matches "
        "every version of this tool before this option existed); 'shrink' reduces "
        "just that label's own font size, down to a 4pt floor; 'shrink_then_truncate' "
        "additionally shortens the label with '...' if even the 4pt floor still "
        "doesn't fit - the full name always still appears in the plot's legend."
    ),
    'ring_layout': (
        "'legacy' stacks rings inward at a fixed width (this tool's traditional "
        "behaviour) - beyond about 33 rings this runs out of room and the plot "
        "becomes invalid. 'fit' instead thins the ring width just enough that every "
        "ring, however many are selected, always fits - recommended for a "
        "many-model/many-gene-source comparison."
    ),
    'figsize': (
        "Overall figure size (inches, square) of each saved circos plot PNG. Leave "
        "blank/8.0 for this tool's traditional size; a larger figure gives ring/tick "
        "labels more physical room to be readable, which matters most for a "
        "many-ring plot."
    ),
    'seam_gap_max': (
        "Upper limit (degrees) on the automatically suggested seam gap above, however "
        "wide the widest ring/chromosome label would otherwise need it to be. A wider "
        "gap always fits more comfortably, but also spends more of the circle on "
        "empty space rather than data - a label that still doesn't fit within this "
        "cap is shrunk (and, if enabled, truncated) at render time instead, per "
        "'Ring label sizing' above."
    ),
    'ring_label_max_chars': (
        "Hard cap on ring-label length (characters), applied before any shrink/"
        "truncate-to-fit logic above - 0 (the default) applies no such cap. Rarely "
        "needed; the shrink/truncate setting above already keeps labels from "
        "overlapping."
    ),
}

RATIO_DEFAULTS = {'within': '0.8', 'between': '(0.8,0.1,0.1)'}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def file_status(path):
    if path and os.path.isfile(path):
        st.caption(f'\u2705 Found: `{path}`')
    elif path:
        st.caption(f'\u26a0\ufe0f File not found on this machine: `{path}`')


def plink_fileset_status(stem):
    """Same idea as file_status(), for a PLINK bed/bim/fam fileset sharing
    one stem - reports each of the three files individually, since a
    partially-present fileset (e.g. .bed and .fam present but .bim missing)
    is a common, easy-to-miss mistake worth calling out precisely rather
    than a single pass/fail."""
    if not stem:
        return
    parts = {'bed': f'{stem}.bed', 'bim': f'{stem}.bim', 'fam': f'{stem}.fam'}
    missing = [ext for ext, p in parts.items() if not os.path.isfile(p)]
    if not missing:
        st.caption(f'\u2705 Found: `{stem}.bed`, `.bim`, `.fam`')
    else:
        found = [ext for ext in parts if ext not in missing]
        st.caption(
            f'\u26a0\ufe0f Missing `{stem}.{"`, `.".join(missing)}`'
            + (f' (found `.{"`, `.".join(found)}`)' if found else '')
        )


# ---------------------------------------------------------------------------
# Phase 2, Requirement 3 - phenotype name auto-suggestion.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _cached_list_phenotype_columns(phenotype_path, _mtime):
    """`_mtime` is part of the cache key only (see
    pipeline_utils.phenotype_file_mtime_key()) - unused inside the
    function body itself - so a repeated Streamlit rerun re-reads the
    phenotype file's header only when it has actually changed on disk,
    not on every single rerun this widget participates in."""
    return list_phenotype_columns(phenotype_path)


def _on_phenotype_multiselect_change():
    """Mutual-exclusivity callback (Requirement 3): if 'all' was just
    added to the selection, it replaces everything else; if any named
    trait was just added while 'all' was already present, 'all' is
    dropped instead. Detected by diffing the new selection against the
    last-known one (stored under a separate, non-widget tracking key),
    since Streamlit's multiselect only ever exposes the full new list,
    not which single option changed."""
    key = 'phenotype_targets_multiselect'
    prev_key = f'_{key}_prev'
    new_selection = list(st.session_state.get(key, []))
    prev_selection = list(st.session_state.get(prev_key, []))

    newly_added = [v for v in new_selection if v not in prev_selection]
    resolved = new_selection
    if 'all' in newly_added:
        resolved = ['all']
    elif 'all' in new_selection and any(v != 'all' for v in newly_added):
        resolved = [v for v in new_selection if v != 'all']

    st.session_state[key] = resolved
    st.session_state[prev_key] = resolved
    # Keep the underlying 'phenotype_targets' STRING key (comma-separated,
    # or 'all') in sync automatically, so gather_config() - and every
    # other reader of 'phenotype_targets' elsewhere in this file - needs
    # NO changes at all; this widget is purely an alternate, structured
    # way of writing the exact same config value (Requirement 3: "No
    # change to the JSON config contract").
    st.session_state['phenotype_targets'] = 'all' if resolved == ['all'] else ','.join(resolved)


def _render_phenotype_target_selector(phenotype_path):
    """Requirement 3: replace the free-text 'Target phenotype(s)' input
    with a typo-proof `st.multiselect` of the phenotype file's own
    detected trait columns (header-only read, cheap - see
    pipeline_utils.list_phenotype_columns()) whenever that file can
    actually be read; falls back to the original free-text input,
    completely unchanged, whenever it can't (not yet uploaded, wrong
    path, etc.) - never blocks config building on this feature."""
    detected_traits = []
    if phenotype_path and os.path.isfile(phenotype_path):
        try:
            _mtime = os.path.getmtime(phenotype_path)
        except OSError:
            _mtime = -1.0
        detected_traits = _cached_list_phenotype_columns(phenotype_path, _mtime)

    if not detected_traits:
        st.text_input(
            "Target phenotype(s) - comma separated, or 'all'", value='days2anthesis', key='phenotype_targets',
            help=("Which trait(s) from the phenotype file to predict. Name one or more columns "
                  "(comma separated) exactly as they appear in that file, or type 'all' to "
                  "predict every trait column found there.")
        )
        st.caption(
            "\u26a0\ufe0f Could not detect trait columns from the phenotype file path above (not "
            "found yet, or unreadable) - using free text for now. A typo-proof selector will "
            "appear here automatically once a valid phenotype file path is set."
        )
        return

    options = detected_traits + ['all']
    key = 'phenotype_targets_multiselect'
    if key not in st.session_state:
        # First render this session: seed from whatever 'phenotype_targets'
        # (the plain string) already holds - e.g. restored from a previous
        # launch's saved GUI state (main_app.py's load_gui_state()) - so
        # switching from the free-text fallback to this selector (once a
        # valid file path is set) doesn't silently reset an already-
        # configured target.
        _raw = st.session_state.get('phenotype_targets', '').strip()
        if _raw.lower() == 'all':
            _seed = ['all']
        else:
            _seed = [p.strip() for p in _raw.split(',') if p.strip() and p.strip() in detected_traits]
        st.session_state[key] = _seed
        st.session_state[f'_{key}_prev'] = _seed
        st.session_state['phenotype_targets'] = 'all' if _seed == ['all'] else ','.join(_seed)

    st.multiselect(
        "Target phenotype(s)", options=options, key=key, on_change=_on_phenotype_multiselect_change,
        help=("Select one or more traits detected in the phenotype file above, or 'all' to "
              "predict every trait column found there. Selecting 'all' clears any individual "
              "selections (and vice versa) - typos are no longer possible when using this list.")
    )


def render_field(key, field, disabled=False):
    """Render the widget(s) for a single hyperparameter field. Nothing is
    returned - Streamlit keeps the live value in st.session_state[key]
    (or, for the composite types, in `{key}__mode` / `{key}__custom`)."""
    ftype = field['type']
    label = field['label']
    default = field['default']
    help_text = field.get('help')

    if ftype == 'bool':
        st.checkbox(label, value=bool(default), key=key, disabled=disabled, help=help_text)

    elif ftype == 'int':
        st.number_input(label, value=int(default), step=1, key=key, disabled=disabled, help=help_text)

    elif ftype == 'float':
        st.number_input(label, value=float(default), key=key, disabled=disabled, help=help_text, format="%0.4f")

    elif ftype == 'str' and 'choices' in field:
        choices = field['choices']
        idx = choices.index(default) if default in choices else 0
        # Defensive self-heal: a value already sitting in session_state for
        # this exact key - restored from an OLDER saved GUI state file, or
        # imported from an older *_config.json - can hold a choice that
        # used to be valid but has since been removed from this field's
        # 'choices' (e.g. the 'surrogate_shap' pre-screen mode, removed
        # from RKHS/RF/SVR/KNN's 'Interaction pre-screen' field). Streamlit
        # raises when a selectbox's key-bound session_state value isn't
        # among its current options, which would otherwise crash the whole
        # GUI on load rather than just this one stale field. Falling back
        # to this field's own current default mirrors exactly what a fresh
        # session would show, and only ever fires for a value that no
        # longer exists to select anyway.
        if key in st.session_state and st.session_state[key] not in choices:
            st.session_state[key] = default if default in choices else choices[0]
        st.selectbox(label, options=choices, index=idx, key=key, disabled=disabled, help=help_text)

    elif ftype == 'str':
        st.text_input(label, value=str(default), key=key, disabled=disabled, help=help_text)

    elif ftype == 'file_path':
        st.text_input(label, value=str(default), key=key, disabled=disabled, help=help_text)
        if not disabled:
            file_status(st.session_state.get(key, default))

    elif ftype == 'top_pct':
        mode_key, custom_key = key + '__mode', key + '__custom'
        default_mode = 'all' if default == 'all' else 'custom number'
        st.selectbox(
            label, options=['all', 'custom number'],
            index=0 if default_mode == 'all' else 1, key=mode_key, disabled=disabled, help=help_text
        )
        show_custom = (not disabled) and st.session_state.get(mode_key, default_mode) == 'custom number'
        custom_default = default if isinstance(default, (int, float)) else 1.0
        st.number_input(
            f'{label} - custom value (%)', value=float(custom_default),
            key=custom_key, disabled=(disabled or not show_custom), help=help_text
        )

    elif ftype == 'int_or_all':
        mode_key, custom_key = key + '__mode', key + '__custom'
        default_mode = 'all' if default == 'all' else 'custom number'
        st.selectbox(
            label, options=['all', 'custom number'],
            index=0 if default_mode == 'all' else 1, key=mode_key, disabled=disabled, help=help_text
        )
        show_custom = (not disabled) and st.session_state.get(mode_key, default_mode) == 'custom number'
        custom_default = default if isinstance(default, int) else 500
        st.number_input(
            f'{label} - custom value', value=int(custom_default), step=1, min_value=1,
            key=custom_key, disabled=(disabled or not show_custom), help=help_text
        )

    elif ftype == 'rf_max_features':
        mode_key, custom_key = key + '__mode', key + '__custom'
        options = ['sqrt', 'log2', 'None', 'custom number']
        if default in ('sqrt', 'log2'):
            default_mode = default
        elif default is None:
            default_mode = 'None'
        else:
            default_mode = 'custom number'
        st.selectbox(label, options=options, index=options.index(default_mode), key=mode_key, disabled=disabled, help=help_text)
        show_custom = (not disabled) and st.session_state.get(mode_key, default_mode) == 'custom number'
        custom_default = default if isinstance(default, (int, float)) else 1.0
        st.number_input(
            f'{label} - custom value', value=float(custom_default),
            key=custom_key, disabled=(disabled or not show_custom), help=help_text
        )

    elif ftype == 'int_float_or_none':
        mode_key, custom_key = key + '__mode', key + '__custom'
        default_mode = 'None' if default is None else 'custom number'
        st.selectbox(
            label, options=['None', 'custom number'],
            index=0 if default_mode == 'None' else 1, key=mode_key, disabled=disabled, help=help_text
        )
        show_custom = (not disabled) and st.session_state.get(mode_key, default_mode) == 'custom number'
        custom_default = default if isinstance(default, (int, float)) else 1.0
        st.number_input(
            f'{label} - custom value', value=float(custom_default),
            key=custom_key, disabled=(disabled or not show_custom), help=help_text
        )

    elif ftype == 'svr_gamma':
        mode_key, custom_key = key + '__mode', key + '__custom'
        options = ['scale', 'auto', 'custom number']
        default_mode = default if default in ('scale', 'auto') else 'custom number'
        st.selectbox(label, options=options, index=options.index(default_mode), key=mode_key, disabled=disabled, help=help_text)
        show_custom = (not disabled) and st.session_state.get(mode_key, default_mode) == 'custom number'
        custom_default = default if isinstance(default, (int, float)) else 1.0
        st.number_input(
            f'{label} - custom value', value=float(custom_default),
            key=custom_key, disabled=(disabled or not show_custom), help=help_text
        )

    elif ftype == 'json':
        # Deliberately renders nothing - every field of this type is always
        # listed in HIDDEN_HPARAM_FIELDS (see render_hparam_panel below) and
        # exists purely as a params-list slot that some other part of the
        # GUI (e.g. the 'Biological Prior Network' tab's 'Data-driven prior
        # network' section) fills in programmatically via
        # st.session_state[key] = <dict>, never through a widget a person
        # types into directly - a raw dict has no sensible generic widget
        # anyway. resolve_field below still returns whatever's there (or
        # the field's own static default) exactly like 'file_path' does for
        # its own hidden fields.
        pass


def resolve_field(key, field):
    """Read back the final Python value for a field rendered by render_field.

    Requirement (bugfix): for every composite type below, the 'mode' this
    field is currently in - and, when relevant, the fallback used for its
    custom-number sub-widget - now falls back to what render_field()
    itself would have shown as the DEFAULT (the exact same default_mode/
    custom_default computation render_field() uses), rather than an
    unrelated hardcoded literal (1.0, or 500) if that widget hasn't been
    rendered (and so hasn't written to session_state) yet in this session.
    Once a widget HAS been rendered at least once, its own session_state
    value always takes precedence here exactly as before - this only
    changes what's returned for a field nothing has touched yet, making
    that case correctly match the field's own configured default instead
    of a one-size-fits-all placeholder."""
    ftype = field['type']
    default = field['default']

    if ftype in ('bool', 'int', 'float', 'str', 'file_path', 'json'):
        return st.session_state.get(key, default)

    mode_key, custom_key = key + '__mode', key + '__custom'

    if ftype == 'top_pct':
        default_mode = 'all' if default == 'all' else 'custom number'
        mode = st.session_state.get(mode_key, default_mode)
        if mode == 'all':
            return 'all'
        custom_default = default if isinstance(default, (int, float)) else 1.0
        return float(st.session_state.get(custom_key, custom_default))

    if ftype == 'int_or_all':
        default_mode = 'all' if default == 'all' else 'custom number'
        mode = st.session_state.get(mode_key, default_mode)
        if mode == 'all':
            return 'all'
        custom_default = default if isinstance(default, int) else 500
        return int(st.session_state.get(custom_key, custom_default))

    if ftype == 'rf_max_features':
        default_mode = default if default in ('sqrt', 'log2') else ('None' if default is None else 'custom number')
        mode = st.session_state.get(mode_key, default_mode)
        if mode in ('sqrt', 'log2'):
            return mode
        if mode == 'None':
            return None
        custom_default = default if isinstance(default, (int, float)) else 1.0
        return float(st.session_state.get(custom_key, custom_default))

    if ftype == 'int_float_or_none':
        default_mode = 'None' if default is None else 'custom number'
        mode = st.session_state.get(mode_key, default_mode)
        if mode == 'None':
            return None
        custom_default = default if isinstance(default, (int, float)) else 1.0
        return float(st.session_state.get(custom_key, custom_default))

    if ftype == 'svr_gamma':
        default_mode = default if default in ('scale', 'auto') else 'custom number'
        mode = st.session_state.get(mode_key, default_mode)
        if mode in ('scale', 'auto'):
            return mode
        custom_default = default if isinstance(default, (int, float)) else 1.0
        return float(st.session_state.get(custom_key, custom_default))

    return None


def get_controller_value(prefix, controller_idx, spec):
    """Live value of a controlling field, falling back to its spec default
    on the very first render (before it has ever been instantiated)."""
    controller_field = spec[controller_idx]
    controller_key = f'{prefix}_{controller_idx}'
    return st.session_state.get(controller_key, controller_field['default'])


def hparam_field_key(model, label_substring):
    """Find the hp_<model>_<idx> session-state key of the HPARAM_SPECS field
    for `model` whose label contains `label_substring`. Used by the
    'Biological Prior Network' tab to push freshly-resolved file paths
    straight into the GAT_biological_prior_knowledge hyperparameter panel,
    without hardcoding that field's numeric position (which would silently
    break if HPARAM_SPECS's field order for that model ever changes)."""
    for idx, field in enumerate(HPARAM_SPECS[model]):
        if label_substring in field['label']:
            return f'hp_{model}_{idx}'
    raise KeyError(f"No HPARAM_SPECS field for model {model!r} matching {label_substring!r}")


def hparam_field_value(model, label_substring):
    """Current value of a hyperparameter field, even if its widget has never
    been rendered yet in this session (falls back to the field's own
    HPARAM_SPECS default) - lets the 'Biological Prior Network' tab preview
    a build using whatever the Models tab is currently set to (or would
    default to), without requiring that tab to have been visited first.

    Requirement (bugfix): delegates to resolve_field() - the SAME function
    render_hparam_panel() itself relies on to read back a field's final
    value - rather than reading st.session_state[key] directly. That
    directly-read approach only ever worked for SIMPLE field types
    ('bool'/'int'/'float'/'str'/...), where render_field() really does
    store the live value under exactly that key; for every COMPOSITE type
    ('top_pct', 'int_or_all', 'rf_max_features', 'int_float_or_none',
    'svr_gamma' - anything with an 'all'/'None'/preset-vs-custom-number
    choice, rendered as a mode selectbox plus a conditional custom
    number_input) render_field() never writes to st.session_state[key] at
    all - only to f'{key}__mode' and f'{key}__custom' - so reading
    st.session_state[key] directly always silently returned the field's
    STATIC DEFAULT, no matter what the person actually selected. This is
    exactly why, for example, the circos 'Top interaction percentage'
    suggestion never seemed to react to changing RF's 'Max markers
    considered for interaction search' or 'Output only the top N% of
    interactions' - both are 'int_or_all'/'top_pct' fields."""
    for idx, field in enumerate(HPARAM_SPECS[model]):
        if label_substring in field['label']:
            key = f'hp_{model}_{idx}'
            return resolve_field(key, field)
    raise KeyError(f"No HPARAM_SPECS field for model {model!r} matching {label_substring!r}")


def render_data_driven_merge_section():
    """Requirement 2: after the network above is resolved (uploaded or
    generated by FLASH-P), optionally generate a SECOND, data-driven
    prior-knowledge network from RF filtering + pairwise Shapley scores,
    and merge it into the biological network (requirement 3).

    Shared across every network configured on the 'Biological Prior
    Network' tab, same as 'Gene-to-marker mapping settings' above it - one
    set of widgets/session_state keys, called once from whichever mode
    (per-phenotype or repeat) is currently active, right after that mode's
    own '...settings above... apply here too' reminder, so it reads as the
    next configuration step in the same flow rather than a separate,
    disconnected section. Only one of the two modes ever actually renders
    in a given script run (they're mutually exclusive, driven by the same
    'How many networks do you want to configure?' radio), so reusing the
    same widget keys from both call sites is safe - see
    build_data_driven_merge_config()'s own docstring for why the resulting
    config is single, shared config, not per-phenotype/per-instance."""
    st.markdown('**Data-driven prior network (optional)**')
    st.checkbox(
        'Combine with a data-driven interaction network (RF importance + pairwise '
        'Shapley scores)',
        value=False, key='bio_data_driven_merge_enabled',
        help="When enabled, an RF-selected marker outside every gene's window becomes "
             "its own graph node with real, data-driven edges to genes/other such "
             "markers - instead of being dropped from the graph entirely (the default). "
             "Requires RF Marker Importance "
             "Filtering (below on this same tab) to be enabled first: this feature "
             "reuses that step's own selected markers and fitted forest configuration, "
             "rather than fitting a second, unrelated one."
    )
    if st.session_state.get('bio_data_driven_merge_enabled', False):
        if not st.session_state.get('rf_filter_enabled', False):
            st.error(
                "Enable 'Apply RF-based marker importance filtering' in the 'RF Marker "
                "Importance Filtering' section (below on this tab) first - this feature "
                "reuses that step's selected markers and forest configuration."
            )
        else:
            st.caption(
                "Reuses the RF Marker Importance Filtering section's own marker-count "
                "mode (percentage or fixed M), forest hyperparameters, and (if enabled) "
                "LD pruning - configure those below/above, then just the interaction "
                "threshold below is specific to this feature."
            )
        st.number_input(
            'Top ___ % strongest interactions', min_value=0.1, max_value=100.0,
            value=20.0, step=1.0, key='bio_data_driven_top_rate',
            help="Only the strongest 'top_rate' percent of pairwise Shapley interaction "
                 "scores among the RF-selected markers become data-driven edges - same "
                 "convention as GAT_prior_knowledge's own 'top_rate' setting."
        )
        c_ov1, c_ov2 = st.columns(2)
        with c_ov1:
            st.checkbox(
                'Override number of trees for the interaction search', value=False,
                key='bio_data_driven_n_estimators_override_enabled',
                help="Leave unchecked to reuse the RF Marker Importance Filtering "
                     "section's own 'Number of trees' setting for this step too."
            )
        with c_ov2:
            st.number_input(
                'Number of trees (interaction search)', min_value=1, value=500, step=10,
                key='bio_data_driven_n_estimators_override',
                disabled=not st.session_state.get('bio_data_driven_n_estimators_override_enabled', False),
                help="Only used if the override checkbox above is on. A forest that "
                     "works well for ranking importance across many markers may want a "
                     "different tree count once restricted to the much smaller, "
                     "already-selected marker set used for the interaction search."
            )
    st.divider()


def build_data_driven_merge_config():
    """Requirement 2: build the (shared, single) data-driven prior-network
    merge config dict from the 'Biological Prior Network' tab's own
    'Data-driven prior network' section - {'enabled': False} if that
    section's checkbox is off (or the section was never configured), in
    which case models/GAT_biological_prior_knowledge.py behaves exactly as
    it always has (requirement: "no changes if not selected").

    Shared (not per-phenotype/per-repeat-instance) by design, mirroring the
    'Gene-to-marker mapping settings' block on the same tab, which is
    likewise one shared config applied to every network configured there -
    see that section's own st.caption for the precedent.
    """
    if not st.session_state.get('bio_data_driven_merge_enabled', False):
        return {'enabled': False}

    if not st.session_state.get('rf_filter_enabled', False):
        raise ValueError(
            "Data-driven prior network: 'Apply RF-based marker importance filtering' must be "
            "enabled above (RF Marker Importance Filtering section) before this can be used - "
            "the data-driven network reuses that RF filtering step's own selected markers and "
            "fitted forest configuration."
        )

    rf_filter_cfg = build_rf_filter_config()
    top_rate = float(st.session_state.get('bio_data_driven_top_rate', 20.0))
    if not (0 < top_rate <= 100):
        raise ValueError(
            "Data-driven prior network: 'Top ___ % strongest interactions' must be between 0 "
            "and 100."
        )

    override_enabled = st.session_state.get('bio_data_driven_n_estimators_override_enabled', False)
    n_estimators_override = (
        int(st.session_state.get('bio_data_driven_n_estimators_override', 500))
        if override_enabled else None
    )

    return {
        'enabled': True,
        'top_rate': top_rate,
        'rf_filter': rf_filter_cfg,
        # LD pruning, if enabled on this same tab, narrows the genotype
        # BEFORE RF selection runs in the side pipeline too (requirement:
        # "LD pruning can also be selected to filter genotypes at the
        # beginning") - reusing the exact same LD_PRUNE config already
        # built for the rest of Tab 2, rather than duplicating an entire
        # second LD-pruning settings panel just for this feature.
        'ld_prune': build_ld_prune_config(),
        'shap_n_estimators_override': n_estimators_override,
    }


def build_bio_prior_params_for(network_json_path, gene_location_csv_path, merge_config=None):
    """Build a full GAT_biological_prior_knowledge params list using the
    shared hyperparameters currently set on the Models tab, but with the
    network JSON / gene-location CSV, and the gene-to-marker mapping
    settings (coordinate unit / mediated edges / max hops - configured on
    the 'Biological Prior Network' tab itself, since they're about network
    construction rather than the model's own training), overridden - used
    to construct a per-phenotype HPARAMETERS dict from that tab (see
    gather_config()).

    `merge_config`, if given, overrides the params list's data-driven
    merge-config slot (requirement 2) - defaults to
    build_data_driven_merge_config()'s own current value when omitted, so
    every existing call site (single/shared, per-phenotype, repeat modes)
    automatically picks up the SAME shared merge config without each
    needing to build/pass it explicitly."""
    model = 'GAT_biological_prior_knowledge'
    params = resolve_hparams(model)
    json_idx = int(hparam_field_key(model, 'Network JSON path').rsplit('_', 1)[1])
    gene_idx = int(hparam_field_key(model, 'Gene location CSV path').rsplit('_', 1)[1])
    unit_idx = int(hparam_field_key(model, 'Coordinate unit').rsplit('_', 1)[1])
    mediated_idx = int(hparam_field_key(model, 'Include mediated edges').rsplit('_', 1)[1])
    hops_idx = int(hparam_field_key(model, 'Max hops for mediated edges').rsplit('_', 1)[1])
    merge_idx = int(hparam_field_key(model, 'Data-driven prior network (merge) config').rsplit('_', 1)[1])
    params[json_idx] = network_json_path
    params[gene_idx] = gene_location_csv_path
    params[unit_idx] = st.session_state.get('bio_coordinate_unit', 'bp')
    params[mediated_idx] = st.session_state.get('bio_include_mediated_edges', True)
    params[hops_idx] = int(st.session_state.get('bio_max_hops', 3))
    params[merge_idx] = merge_config if merge_config is not None else build_data_driven_merge_config()
    return params


# Fields that still need to exist at their correct position in a model's
# params list (other code locates them by index via hparam_field_key(), and
# the model function itself unpacks params positionally), but that are no
# longer meaningful for a user to see or edit directly, because something
# else always overwrites whatever's here before it's used. Network JSON
# path / Gene location CSV path are GAT_biological_prior_knowledge's own
# params list slots 7/8 - every configuration path (per-phenotype mode,
# repeat mode) resolves its own network(s) on the 'Biological Prior
# Network' tab and substitutes them in directly (see
# build_bio_prior_params_for()), so whatever a user might type into these
# two fields here is silently discarded before a run ever uses it. Hidden
# rather than removed from HPARAM_SPECS, since resolve_hparams() still
# needs a value (any value - it's always overwritten) at these exact
# positions to keep the params list the right length/shape.
# rrBLUP/BayesB/GBLUP's own "Return marker-pair interactions?
# (approximate)" toggle and its three dependent settings, hidden for these
# three R models. rrBLUP and GBLUP are purely additive models and BayesB is
# an additive variable-selection model, so a surrogate search for NONLINEAR
# marker-pair interactions on top of them was never a meaningful
# diagnostic. Hidden rather than removed from HPARAM_SPECS for the same
# reason as the bio-prior fields above - the field must still exist at its
# correct position for _r_model_interaction_fields()'s positional read in
# genomic_prediction.py, and resolve_hparams() falls back to each field's
# own spec default (False for the toggle) when its widget is never
# rendered, so runs launched from the GUI simply never turn this on for
# any of these three models.
#
# Update (RKHS unhidden again): RKHS was initially left visible (its kernel
# already captures non-additive structure, unlike the three models above),
# then hidden alongside them because the only available route was a
# surrogate fitted to RKHS's OWN predictions, explained via TreeSHAP - a
# diagnostic about the surrogate's structure, once removed from what makes
# RKHS's own kernel actually interesting. RKHS's own interaction route now
# uses Friedman's H-statistic instead (genomic_prediction.py::
# _r_model_surrogate_interaction(), models.interaction_extraction.
# surrogate_h_statistic_interactions()) - still an approximate surrogate
# (RKHS's own kernel machinery still can't be explained pairwise directly
# at any tolerable cost), but this was judged worth surfacing again, so
# RKHS is deliberately excluded from HIDDEN_HPARAM_FIELDS below while
# rrBLUP/BayesB/GBLUP remain hidden, unchanged.
_R_MODEL_INTERACTION_FIELDS = {
    'Return marker-pair interactions?',
    'Max markers considered for interaction search ("all" for every marker)',
    'Number of individuals used to explain the model',
    'Output only the top N% of interactions ("all" for everything)',
}

# GBLUP and RKHS's own "Shapley row offset" / "Shapley row count" fields are
# a purely internal knob EasiGP itself uses to fan a single task's Shapley
# computation out across several worker processes (see
# genomic_prediction.py's Shapley row-range fan-out worker). There is no
# scenario where a user launching a run from the GUI needs to touch these -
# leave-at-defaults (offset 0 / count -1) always reproduces an ordinary,
# non-fanned-out run. Hidden rather than removed from HPARAM_SPECS for the
# same reason as the fields above: GBLUP.R / RKHS.R still unpack their
# params list positionally, so the fields must stay at their exact
# positions, and resolve_hparams() falls back to each field's own spec
# default when its widget is never rendered, so GUI-launched runs simply
# never set these to anything other than 0 / -1.
_SHAPLEY_ROW_FANOUT_FIELDS = {
    'Shapley row offset (advanced - internal parallel fan-out)',
    'Shapley row count (advanced - internal parallel fan-out; -1 = every row)',
}

HIDDEN_HPARAM_FIELDS = {
    'GAT_biological_prior_knowledge': {
        'Network JSON path', 'Gene location CSV path',
        'Coordinate unit', 'Include mediated edges', 'Max hops for mediated edges',
        'Data-driven prior network (merge) config',
    },
    'rrBLUP': set(_R_MODEL_INTERACTION_FIELDS),
    'BayesB': set(_R_MODEL_INTERACTION_FIELDS),
    'GBLUP': set(_R_MODEL_INTERACTION_FIELDS) | set(_SHAPLEY_ROW_FANOUT_FIELDS),
    # RKHS deliberately has NO entry for _R_MODEL_INTERACTION_FIELDS (see
    # the comment above this dict's own definition) - render_hparam_panel()'s
    # `HIDDEN_HPARAM_FIELDS.get(model, set())` falls back to an empty set
    # for any model with no key, so RKHS's own "Return marker-pair
    # interactions? (approximate)" toggle and its three dependent fields
    # render normally, exactly like every other model that was never
    # hidden. Its Shapley row offset/count fields ARE hidden though, for
    # the same internal-fan-out reason as GBLUP above.
    'RKHS': set(_SHAPLEY_ROW_FANOUT_FIELDS),
}


def render_hparam_panel(model):
    spec = HPARAM_SPECS[model]
    prefix = f'hp_{model}'
    hidden_substrings = HIDDEN_HPARAM_FIELDS.get(model, set())
    with st.expander(f'{model} hyperparameters', expanded=True):
        # Requirement 9: 'Automatically tune ### hyperparameters' comes
        # FIRST, before any of the model's own hyperparameter fields
        # (e.g. before 'Iteration number' for rrBLUP) - it governs
        # whether the fixed values below are even used at all (a search
        # replaces them, per task, when this is checked), so it reads
        # more naturally as the first decision to make about this
        # model's hyperparameters, not an afterthought tacked on below
        # the whole panel. hp_tune_supported() is checked here the same
        # way render_hp_tune_panel() itself always has - some models
        # (see build_param_specs()) have no tunable fields at all, and
        # skip the checkbox entirely rather than offering a control that
        # would do nothing.
        if hp_tune_supported(model):
            _tune_prefix = f'hp_tune_{model}'
            st.checkbox(
                f'Automatically tune {model} hyperparameters', value=False, key=f'{_tune_prefix}_enabled',
                help=("Searches for the hyperparameters (among the tunable fields below) that "
                      "maximise Pearson r - (MSE / phenotype variance) on the validation set, per task, instead of always "
                      "using the fixed values below. Only takes effect when a validation set exists "
                      "for that task (a train/valid/test split, or the 'between' scenario) - "
                      "otherwise the fixed values below are used unchanged.")
            )
            # Update ID 3, R5: the tuning settings now render immediately
            # after their own checkbox (and before every ordinary
            # hyperparameter field below), rather than in a separate
            # expander called later from Tab 3 - see render_hp_tune_panel()'s
            # own docstring. Calling forward is legal (Python resolves the
            # name at call time), so render_hp_tune_panel's definition stays
            # below this function's, unmoved.
            render_hp_tune_panel(model)
            st.divider()
        for idx, field in enumerate(spec):
            if any(sub in field['label'] for sub in hidden_substrings):
                continue
            # 'visible_when' is checked before anything else and, unlike
            # 'depends_on' below, skips rendering the field entirely (no
            # widget at all this run) rather than greying it out - see the
            # key's own description in the schema comment above this
            # function's block. get_controller_value() already falls back
            # to the controller field's own spec default when that widget
            # hasn't been rendered yet this session, so this is safe to
            # evaluate unconditionally, even on the very first render.
            vis = field.get('visible_when')
            if vis:
                controller_idx, required = vis
                if get_controller_value(prefix, controller_idx, spec) != required:
                    continue
            key = f'{prefix}_{idx}'
            dep = field.get('depends_on')
            disabled = False
            if dep:
                controller_idx, required = dep
                disabled = get_controller_value(prefix, controller_idx, spec) != required
            # Requirement 5: highlight 'Output only the top N% of
            # interactions' specifically (not its sibling fields, which
            # share the same depends_on controller but are comparatively
            # minor tuning knobs) whenever it's actually active - i.e.
            # 'Return marker effect for interactions?' is checked. This
            # ONE setting has an outsized effect on both how
            # interpretable the returned interactions are (too high a
            # percentage buries genuine signal in noise; too low hides
            # real interactions) and how long the search takes.
            _highlight = (not disabled) and field['label'].startswith('Output only the top N% of interactions')
            if _highlight:
                with st.container(border=True):
                    st.markdown(
                        "⚠️ **This setting significantly changes both how the interactions can be "
                        "interpreted and how long the process of this model takes:**"
                    )
                    render_field(key, field, disabled=disabled)
            else:
                render_field(key, field, disabled=disabled)


def resolve_hparams(model):
    spec = HPARAM_SPECS[model]
    prefix = f'hp_{model}'
    return [resolve_field(f'{prefix}_{idx}', field) for idx, field in enumerate(spec)]


def render_wopt_panel(method):
    spec = HYPERPARAM_OPT_SPECS[method]
    prefix = f"wopt_{method.replace(' ', '_')}"
    with st.expander(f'{method} hyperparameters', expanded=True):
        for idx, field in enumerate(spec):
            render_field(f'{prefix}_{idx}', field, disabled=False)


def resolve_wopt(method):
    spec = HYPERPARAM_OPT_SPECS[method]
    prefix = f"wopt_{method.replace(' ', '_')}"
    return [resolve_field(f'{prefix}_{idx}', field) for idx, field in enumerate(spec)]


# ------------------------- Hyperparameter tuning ------------------------- #
#
# Opt-in, per-model automatic search for the hyperparameters above, using
# models/hyperparameter_tuning.py. Only ever takes effect when a real
# validation split exists for a given task (see that module's docstring).
# GAT_biological_prior_knowledge is fully supported (including numbered
# instances, e.g. GAT_biological_prior_knowledge_2, if configured directly
# via HP_TUNE rather than through this GUI panel - this panel only offers
# the base model) - genomic_prediction.py's _call_model() resolves its own
# marker pool from the same task-level gene-network extraction the main
# dispatch uses, exactly mirroring that branch.
from models.hyperparameter_tuning import ALGORITHMS, build_param_specs, expand_model_list


def hp_tune_supported(model):
    return len(build_param_specs(model, HPARAM_SPECS)) > 0


def _active_tuning_algorithms():
    """Union of every model's own 'Search algorithm(s)' currently selected
    in this session, across every model whose 'Automatically tune ...
    hyperparameters' checkbox is currently checked. Used solely to decide
    which of the GLOBAL, algorithm-specific hyperparameter-tuning settings
    (HP_TUNE_PARALLEL_TRIALS / HP_TUNE_BAYES_* / HP_TUNE_PARALLEL_RESTARTS,
    rendered once at the bottom of the Models tab) are even relevant to
    show right now - a model with tuning enabled but no algorithm chosen
    yet contributes nothing (resolve_hp_tune() already treats that config
    the same as tuning fully off), and hp_tune_supported() already excludes
    models with no tunable fields at all."""
    algos = set()
    for model in AVAILABLE_MODELS:
        if model == 'ensemble' or not hp_tune_supported(model):
            continue
        if not st.session_state.get(f'hp_tune_{model}_enabled', False):
            continue
        algos.update(st.session_state.get(f'hp_tune_{model}_algorithms', []))
    return algos


def render_hp_tune_panel(model):
    """Requirement 9: the 'Automatically tune ### hyperparameters'
    checkbox itself now renders INSIDE render_hparam_panel()'s own
    expander, at the very top (before this model's own hyperparameter
    fields) - see that function's own comment for why. This function
    still owns everything AFTER the checkbox: the separate 'tuning
    settings' expander (search algorithm(s), budget, etc.), shown only
    when that (elsewhere-rendered) checkbox is ticked - reads its
    session_state value the same way as before, just no longer draws
    the checkbox widget itself."""
    if not hp_tune_supported(model):
        return
    prefix = f'hp_tune_{model}'
    if not st.session_state.get(f'{prefix}_enabled', False):
        return
    with st.container(border=True):
        st.markdown(f'**{model} hyperparameter tuning settings**')
        st.multiselect(
            'Search algorithm(s)', options=ALGORITHMS, key=f'{prefix}_algorithms',
            help=("Selecting more than one produces separate tuned variants of this model, "
                  "each suffixed in every output file (e.g. 'RF__Grid', 'RF__Bayesian') so they "
                  "can be compared side by side. Grid is exhaustive but capped for wide search "
                  "spaces; Random/Bayesian/Nelder-Mead/Powell all scale by iteration budget "
                  "instead and generally reach a good result faster on models with several "
                  "tunable fields (e.g. RF, SVR, the GAT variants).")
        )
        algorithms = st.session_state.get(f'{prefix}_algorithms', [])
        for algo in algorithms:
            cols = st.columns(2)
            if algo == 'Grid':
                with cols[0]:
                    st.number_input(
                        'Grid points per numeric dimension', min_value=2, value=5, step=1,
                        key=f'{prefix}_budget_{algo}_n_points',
                        help="Fields with a fixed step (see hparam_specs.py) ignore this and use "
                             "their own step instead. A safety cap prevents an accidentally huge grid."
                    )
            else:
                with cols[0]:
                    st.number_input(f'{algo} iterations', min_value=1, value=30, step=1,
                                     key=f'{prefix}_budget_{algo}_n_iter')
                if algo == 'Bayesian':
                    with cols[1]:
                        st.number_input('Initial random points', min_value=1, value=5, step=1,
                                         key=f'{prefix}_budget_{algo}_init_points')
        if model in ('rrBLUP', 'GBLUP', 'BayesB', 'RKHS'):
            st.checkbox(
                'Use a cheaper MCMC length while searching', value=True, key=f'{prefix}_reduced_cost',
                help=("Each candidate during the search runs a much shorter MCMC chain than your "
                      "'Iteration number'/'Burn-in' settings above (those are only used for the "
                      "final confirmatory fit once the winning hyperparameters are chosen) - makes "
                      "the search dramatically faster for rrBLUP/GBLUP/BayesB/RKHS. Leave this on "
                      "unless you have a specific reason to search at full MCMC length.")
            )


def resolve_hp_tune(model):
    if not hp_tune_supported(model):
        return None
    prefix = f'hp_tune_{model}'
    if not st.session_state.get(f'{prefix}_enabled', False):
        return None
    algorithms = st.session_state.get(f'{prefix}_algorithms', [])
    if not algorithms:
        return None
    budget = {}
    for algo in algorithms:
        if algo == 'Grid':
            budget[algo] = {'n_points': int(st.session_state.get(f'{prefix}_budget_{algo}_n_points', 5))}
        else:
            b = {'n_iter': int(st.session_state.get(f'{prefix}_budget_{algo}_n_iter', 30))}
            if algo == 'Bayesian':
                b['init_points'] = int(st.session_state.get(f'{prefix}_budget_{algo}_init_points', 5))
            budget[algo] = b
    return {
        'enabled': True,
        'algorithms': algorithms,
        'budget': budget,
        'reduced_cost_search': bool(st.session_state.get(f'{prefix}_reduced_cost', True)),
    }


# NCI Gadi's current documented cap on the number of subjobs inside a
# single PBS job array. Kept as a single named constant (rather than
# scattered magic numbers) so it's easy to find and update - VERIFY this
# against NCI's current documentation / `qsub`'s own limits at
# implementation/deployment time; it is deliberately NOT treated as an
# authoritative, permanently-correct value by this codebase.
GADI_MAX_ARRAY_SUBJOBS = 500


def _render_resource_advisor_panel(kp, export_mode, export_step):
    """Phase 2, Requirement 8 - 'Suggested compute resources' panel, shown
    directly above the HPC job resource-request expander it feeds into
    (Requirement 1's per-batch 'CPUs per task'/'Memory' fields, and - for
    a Gadi native job-array export - 'Chunk size'). Entirely best-effort:
    if the pipeline isn't configured enough yet for gather_config() to
    succeed, this silently shows nothing rather than erroring, since this
    panel is a convenience, not a requirement to proceed.

    Recomputed fresh on every render (the underlying cheap-read helpers -
    pipeline_utils.cheap_marker_and_sample_counts() et al. - are each,
    individually, fast enough not to need their own st.cache_data wrapper
    the way the phenotype-column suggester does; this whole panel still
    completes in well under a second for any realistic config, per
    Requirement 8's own acceptance criterion).

    Patch 3 v3, Requirement 1: the job-array-size ("batch size"/
    n_batches) suggestion that used to live inside this panel has been
    extracted to its own function, _render_suggested_job_array_size(),
    now shown separately UNDER 'Parallel batch configuration (PARALLEL)'
    (right after 'Run pipeline', before 'Batch ID source') rather than
    buried inside this CPU/memory/GPU panel - "clearly separate the
    array-related configuration and other resource-related
    configuration." This panel therefore no longer shows array-size
    numbers at all; see that function instead.

    Patch 3 v3, Requirement 3: this panel now reads an 'HPC cluster
    profile' selection and forwards it into estimate_resources() so the
    GPU/CPU-ratio floor and the array-cap referenced in this panel's own
    captions are calibrated for the SELECTED cluster, not always Gadi.
    Patch 3 v4, Requirement 1: that selectbox itself moved even further
    up the page (right under 'Run pipeline', immediately after choosing
    'Option A' - see _render_hpc_cluster_profile_selector()) - this panel
    still only READS it, never draws its own copy.

    ver4-4, R2 (blueprint §2.2) - this panel accreted four independent
    recommendation systems across Update ID 2, Patch 3 v2/v3/v4 and
    Additional Requirements 9 (an opt-in checkbox that additionally
    considered model-level parallelism, plus the per-batch CPU/GPU/
    memory budget fields it revealed; an always-on block giving a
    second, jointly-sized ncpus/worker-count/memory recommendation; and
    a third, opt-in block detailing the resulting task-level/model-level
    split), each layered on rather than replacing the last - and TWO
    separate apply buttons both silently wrote worker-
    count session state, so worker fan-out ended up enabled just by
    pressing a button labelled "Use these values for CPUs per task /
    Memory / GPU". R2 removes all three of those blocks (and the second
    apply button) outright: this panel is now a single, ALWAYS-COMPUTED
    three-metric headline (ncpus / mem / GPU count) plus ONE apply
    button that writes only 'CPUs per task' / 'Memory' / 'GPU' - never
    worker counts. `estimate_resources()` keeps every one of its
    existing parameters and returned keys (nothing downstream breaks);
    this panel simply stops reading the joint-allocation keys
    (`coordinated_*`, `n_cpu_workers_task`, `n_model_workers`, ...),
    which are still computed internally but no longer displayed here.
    `N_CPU_WORKERS`/`N_MODEL_WORKERS` now live in their own 'Advanced
    setting components' expander (see `_render_advanced_setting_
    components()`, rendered inside `render_hpc_export_section()`), both
    defaulting to 1 and never auto-filled by any suggestion - a person
    who wants worker fan-out sets it there, deliberately, rather than as
    a side effect of an unrelated CPU/memory button.
    """
    try:
        cfg = gather_config(export_mode, export_step)
    except ValueError as exc:
        # Update ID 3, R4: the panel must never vanish silently - this is
        # the fix for "estimates are not available": ticking LD pruning
        # leaves ld_window_unit='kb' and ld_snp_info_path='' at their
        # defaults, build_ld_prune_config() raises on exactly that default
        # state, and the bare `return` this replaces made the ENTIRE panel
        # disappear with no message at all. Do NOT relax that validation -
        # this fixes the panel, not the validator - and this now resolves
        # the same "vanishes with no message" failure for every OTHER
        # gather_config() validator too, not just LD pruning's.
        # Requirements.md item 7: no longer collapsed by default (was
        # expanded=False) - and now titled 'Suggested job-array size' to
        # match the requested display text.
        with st.expander('Suggested job-array size', expanded=True):
            st.info(f"No estimate yet - finish configuring the pipeline first: {exc}")
        return

    if 'MODEL' not in cfg:
        return  # Step 2 export - no model selection to size resources from

    # Requirements.md item 7: no longer collapsed by default (was
    # expanded=False) - and now titled 'Suggested job-array size' to match
    # the requested display text (this panel's own content - the
    # ncpus/mem_gb/ngpus heuristic headline - is unchanged; see the
    # docstring above for the SEPARATE, differently-scoped
    # _render_suggested_job_array_size() panel further down the page).
    with st.expander('Suggested compute resources', expanded=True):
        st.caption(
            "\u26a0\ufe0f Heuristic estimate only, computed from cheap config-only inputs plus a "
            "quick header/line-count read of your genotype/phenotype files (never a full data "
            "load) - not a guarantee. Recommend piloting a small batch before committing a "
            "large array submission."
        )

        # Patch 3 v4, Requirement 1/3: 'HPC cluster profile' is now drawn
        # right under 'Run pipeline', immediately after choosing 'Option
        # A' - see _render_hpc_cluster_profile_selector(), called well
        # before this panel is even reached - so its session_state value
        # already exists by the time we get here (same "read an
        # already-drawn widget's session_state" pattern used throughout
        # this section, e.g. 'use_gpu' below).
        _hpc_profile = st.session_state.get(f'{kp}_hpc_profile', DEFAULT_HPC_PROFILE)
        _max_array_override = st.session_state.get(f'{kp}_hpc_max_array_override')
        _ncpus_per_gpu_override = st.session_state.get(f'{kp}_hpc_ncpus_per_gpu_override')

        # Additional Requirements 9, Requirement 4 - "when those pruning
        # approaches are selected, we should ask for their estimated
        # marker numbers after that pruning if they ask for a
        # suggestion." RF filtering's own reduction is derivable EXACTLY
        # from its config (mode='count'/'percent') - no input needed, just
        # shown once the estimate itself is computed below
        # (`n_markers_effective`). LD pruning's reduction genuinely
        # depends on the real linkage-disequilibrium structure of the
        # data (never read in full by this advisor - see
        # resource_profiles.py's own module docstring), so it is
        # offered here as an OPTIONAL figure the person can supply.
        rf_filter_cfg_for_estimate = cfg.get('RF_FILTER')
        ld_prune_estimated_n_markers = None
        if cfg.get('LD_PRUNE') is not None:
            _ld_estimate_input = st.number_input(
                'Estimated markers remaining after LD pruning (optional - improves accuracy)',
                min_value=0, value=0, step=100, key=f'_{kp}_advisor_ld_estimated_markers',
                help="LD pruning is enabled, but how many markers actually survive PLINK's "
                     "--indep-pairwise depends on your data's real linkage-disequilibrium "
                     "structure, which this advisor deliberately never reads in full. Leave at "
                     "0 to use a conservative (likely over-estimating) fallback that assumes LD "
                     "pruning removes nothing; supply your own estimate (e.g. from a previous "
                     "run's Marker_effect.csv column count, or the LD decay plot) for a tighter "
                     "ncpus/memory figure - this affects GAT_fully_connected's memory estimate "
                     "the most, since it scales with the SQUARE of the marker count."
            )
            ld_prune_estimated_n_markers = int(_ld_estimate_input) if _ld_estimate_input else None

        try:
            n_population = count_unique_populations(cfg.get('PHENOTYPE_FILE_NAME')) or 1
            n_phenotype = len(cfg['PHENOTYPE']) if cfg['PHENOTYPE'] != 'all' else max(
                1, len(list_phenotype_columns(cfg.get('PHENOTYPE_FILE_NAME', '')))
            )
            n_ratio = len(cfg.get('RATIO', [1]))
            parallel_cfg = cfg.get('PARALLEL')
            # Additional Requirements 9, Requirement 2 - "no coordination
            # between the suggested job-array size and the suggested
            # compute resources": when the person hasn't explicitly
            # configured a PARALLEL batch size yet, sizing this estimate's
            # per-batch costing off n_batches=1 (i.e. "one batch = the
            # WHOLE run") gives wildly larger ncpus/mem_gb than what the
            # SUGGESTED job-array layout (shown separately, in
            # _render_suggested_job_array_size()) would actually produce
            # per batch - the two panels would visibly disagree before
            # the person has even touched 'Batch size'. Resolved with the
            # SAME two-call pattern estimate_resources() already uses
            # internally for OTHER_MODELS_MARKER_SOURCE-style short-
            # circuits: compute once (n_batches=1) purely to read this
            # SAME call's own `suggested_n_batches` back out, then
            # recompute the REAL per-batch costing against that - a
            # no-op (single call) once a batch size IS configured, since
            # `_batch_size_configured` is then True.
            _batch_size_configured = bool(parallel_cfg and parallel_cfg.get('batch_size'))
            n_batches = 1
            if _batch_size_configured:
                total_within = n_population * n_phenotype * n_ratio * cfg.get('ITER_NUM', 1)
                n_batches = max(1, math.ceil(max(1, total_within) / int(parallel_cfg['batch_size'])))

            _estimate_kwargs = dict(
                scenario=cfg['SCENARIO'], n_population=n_population, n_phenotype=n_phenotype,
                n_ratio=n_ratio, sample_num=cfg.get('ITER_NUM', 1), model_run=cfg['MODEL'],
                ld_prune_enabled=cfg.get('LD_PRUNE') is not None, rf_filter_enabled=cfg.get('RF_FILTER') is not None,
                genotype_format=cfg.get('GENOTYPE_FORMAT', 'csv'), genotype_file_name=cfg.get('GENOTYPE_FILE_NAME', ''),
                phenotype_file_name=cfg.get('PHENOTYPE_FILE_NAME'), w_opt_enabled=cfg.get('W_OPT') is not None,
                # ver4-4, R2: the joint task-level/model-level allocator
                # and its CPU/GPU/memory "budget" concept are no longer
                # exposed in this panel at all (see docstring above) -
                # estimate_resources() is called exactly the way Phase 2
                # always called it, so the three headline metrics below
                # are byte-identical to Phase 2's own output (A2.2).
                model_parallel_enabled=False,
                budget_ncpus=None,
                budget_ngpus=None,
                budget_mem_gb=None,
                hparameters=cfg.get('HPARAMETERS'), hp_tune=cfg.get('HP_TUNE'),
                # Update ID 3, R4: forward the run's actual marker-source
                # short-circuit (arch §8) and the new headline rollback
                # flag - both default to the pre-Update-3 behaviour when
                # absent from an older cfg (I11).
                other_models_marker_source=cfg.get('OTHER_MODELS_MARKER_SOURCE', 'full_or_filtered'),
                pool_cost_in_headline=cfg.get('POOL_COST_IN_HEADLINE', True),
                # Patch 3, R4: GPU-aware ncpus floor/headroom - see
                # resource_profiles.estimate_resources()'s own docstring.
                gpu_cpu_calibration_in_headline=cfg.get('GPU_CPU_CALIBRATION_IN_HEADLINE', True),
                # Patch 3 v3, R3: calibrate against the SELECTED cluster,
                # not always Gadi - see _render_hpc_cluster_profile_selector()
                # (Patch 3 v4: now drawn right under 'Run pipeline').
                hpc_profile=_hpc_profile,
                max_array_subjobs_override=_max_array_override,
                ncpus_per_gpu_override=_ncpus_per_gpu_override,
                # Additional Requirements 9, Requirement 4.
                rf_filter_cfg=rf_filter_cfg_for_estimate,
                ld_prune_estimated_n_markers=ld_prune_estimated_n_markers,
            )
            estimate = estimate_resources(n_batches=n_batches, **_estimate_kwargs)
            if not _batch_size_configured and estimate.get('suggested_n_batches'):
                n_batches = max(1, int(estimate['suggested_n_batches']))
                estimate = estimate_resources(n_batches=n_batches, **_estimate_kwargs)
        except Exception as exc:
            st.caption(f"Could not compute an estimate yet ({exc}).")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric('Suggested ncpus (per batch)', estimate['ncpus'])
        c2.metric('Suggested mem (per batch, GB)', estimate['mem_gb'])
        # Patch 3 v2, Requirement 2: numeric, not a Yes/No flag.
        # resource_profiles.estimate_resources() already returns 'ngpus' as
        # a plain int (0 when no selected model is GPU-capable, otherwise
        # the recommended GPU count) - only the DISPLAY collapsed it into a
        # Yes/No string before this; the underlying recommendation itself
        # is unchanged.
        c3.metric('Suggested GPU(s)', int(estimate['ngpus']))
        st.caption(
            f"Based on ~{estimate['n_markers']:,} marker(s), ~{estimate['n_samples']:,} sample(s), "
            f"~{estimate['total_tasks_per_batch']:,} task(s)/batch across {n_batches} batch(es) "
            f"({estimate['total_tasks_all_batches']:,} task(s) total) - already accounting for "
            f"every target phenotype, selected prediction model, and split ratio configured above."
        )
        # Update ID 3, R4: makes the headline's now-filtering-aware ncpus/
        # mem_gb legible - without this, a person ticking LD pruning/RF
        # filtering and seeing the numbers move has no explanation for why.
        if (cfg.get('LD_PRUNE') is not None or cfg.get('RF_FILTER') is not None) and estimate.get('pool_cost_headline_applied'):
            st.caption(
                f"Includes the cost of LD pruning and RF marker filtering "
                f"(+{estimate.get('pool_cost_increment', 0.0):.1f} cost units/task)."
            )
        # Additional Requirements 9, Requirement 4 - makes the post LD/RF-
        # filter marker estimate actually used for mem_gb legible (see
        # 'Estimated markers remaining after LD pruning' above, and RF
        # filtering's own exact, config-derived reduction).
        #
        # GUI defect fix - "the suggested memory value does not change even
        # after users type an estimated marker value": this used to say
        # "Memory sized for ~X marker(s)..." purely off
        # marker_filter_calibration_applied (i.e. whenever the marker COUNT
        # changed), even on the many small/pilot-scale runs where the
        # recomputed mem_gb comes out identical to the raw figure because
        # both price out at or below the memory floor - the person types a
        # smaller number, watches 'Suggested mem' not move an inch, and
        # (reasonably) concludes the field is broken. It isn't: there is
        # nothing left to shrink once a run is already at the practical
        # minimum. Branching on marker_filter_changed_mem_gb (resource_
        # profiles.estimate_resources()'s own record of whether mem_gb
        # actually moved) says so explicitly instead of silently implying a
        # change that didn't happen.
        if estimate.get('marker_filter_calibration_applied'):
            if estimate.get('marker_filter_changed_mem_gb'):
                st.caption(
                    f"Memory sized for \u2248{estimate['n_markers_effective']:,} marker(s) after LD/RF "
                    f"filtering (from {estimate['n_markers']:,} raw) - "
                    + ("an estimate you supplied above." if not estimate.get('ld_estimate_uncertain')
                       else "RF filtering's own exact reduction only; LD pruning's is not yet estimated "
                            "(see the note below).")
                )
            else:
                st.caption(
                    f"\u2139\ufe0f Marker count for memory sizing: \u2248{estimate['n_markers_effective']:,} "
                    f"after LD/RF filtering (from {estimate['n_markers']:,} raw) - 'Suggested mem' isn't "
                    f"moving because this run is already at the practical memory floor "
                    f"({estimate['mem_gb']:g} GB) at EITHER marker count; only ncpus/GPU-heavy or very "
                    f"large-marker-set runs (GAT_fully_connected especially) see this figure move."
                )
        # Patch 3, R4: makes the GPU-aware ncpus floor/headroom legible -
        # without this, a person selecting a GPU-capable model has no
        # explanation for why ncpus jumped well above what the model-
        # fitting workload alone would suggest. Patch 3 v3, R3: the
        # message itself is now inside estimate['notes'] (profile-aware,
        # see resource_profiles.estimate_resources()), so this just
        # displays that note rather than re-deriving Gadi-specific text
        # here.
        if estimate.get('gpu_cpu_calibration_applied') and estimate.get('gpu_ncpus_increment', 0) > 0:
            st.caption(f"\u2139\ufe0f ncpus adjusted for GPU use - see the notes below for details.")

        for note in estimate['notes']:
            st.caption(f"\u2139\ufe0f {note}")

        # ver4-4, R2: ONE apply button, writing ONLY the plain resource-
        # request fields it is labelled for - 'CPUs per task' / 'Memory' /
        # 'GPU'. It deliberately does NOT touch N_CPU_WORKERS/
        # N_MODEL_WORKERS any more (see docstring above) - those live in
        # their own 'Advanced setting components' expander, both
        # defaulting to 1, and are never auto-filled by this or any other
        # suggestion.
        if st.button('Use these values for CPUs per task / Memory / GPU above', key=f'_btn_{kp}_apply_resource_advisor'):
            st.session_state[f'{kp}_hpc_cpus_per_task'] = int(estimate['ncpus'])
            st.session_state[f'{kp}_hpc_mem'] = f"{max(1, math.ceil(float(estimate['mem_gb'])))}G"
            # Also apply the GPU recommendation, if any - 'ngpus' may be a
            # plain truthy flag or an actual count depending on
            # resource_profiles.estimate_resources()'s current
            # implementation; either way, request at least 1 GPU when it
            # recommends any.
            if estimate.get('ngpus'):
                st.session_state[f'{kp}_hpc_use_gpu'] = True
                try:
                    st.session_state[f'{kp}_hpc_gpus_per_task'] = max(1, int(estimate['ngpus']))
                except (TypeError, ValueError):
                    st.session_state[f'{kp}_hpc_gpus_per_task'] = 1
            st.success(
                "Applied - see 'CPUs per task' / 'Memory' / 'GPU' in the resource-request "
                "section below. Worker counts (N_CPU_WORKERS / N_MODEL_WORKERS) are configured "
                "separately, under 'Advanced setting components' further down - they default to "
                "1 and are left untouched by this button."
            )


def _render_suggested_job_array_size():
    """Patch 3 v3, Requirement 1 - the array/batch-size suggestion,
    extracted out of the (now-generic) 'Suggested compute resources'
    panel and shown separately, under 'Parallel batch configuration
    (PARALLEL)' (right after 'Run pipeline'), BEFORE 'Batch ID source' -
    "clearly separate the array-related configuration and other
    resource-related configuration" rather than leaving both mixed into
    one combined panel inside 'HPC job resource requests' further down.

    Only meaningful for a Parallel/Step 1 export (the only mode with a
    PARALLEL batch_size / job array to size at all) - the caller only
    invokes this when `mode == 'Parallel' and step == 'Step 1'`.

    Entirely best-effort, exactly like _render_resource_advisor_panel():
    if the pipeline isn't configured enough yet for gather_config() to
    succeed, this shows an info message rather than erroring.

    ver4-4, R1 (blueprint §2.1) - this used to show ONE, already-
    collapsed batch-size/n-batches answer (`resource_profiles.
    suggest_array_layout()`'s own single pick). It now shows this run's
    TOTAL task count as its own headline metric first, then - via
    `resource_profiles.enumerate_array_layouts()` - every (batch_size,
    n_batches) pair that divides that total EXACTLY (so every batch in
    the resulting array processes an identical task count, no smaller
    trailing batch at all), for the person to choose between, with the
    workload-sized pick pre-selected but never forced. The 'Update these
    values' button below applies whichever pair is CURRENTLY SELECTED in
    that list, not always the pre-selected one.

    When the total doesn't have more than one workable exact-dividing
    pair under the array-subjob cap (a prime total, or a cap too narrow
    to contain a second option), this falls back to
    `suggest_array_layout()`'s own closest-achievable-balance answer and
    says so explicitly - the same documented degradation the blueprint
    describes, not an error.

    The 'Update these values' button below writes directly into
    st.session_state['batch_size'] (Tab 1's widget) and
    st.session_state['step1_hpc_array_start']/['step1_hpc_array_end']
    (widgets inside render_hpc_export_section()'s 'HPC job resource
    requests' expander, under 'Option A' further down 'Run pipeline').
    This is legal ONLY because this function is called BEFORE any of
    those three widgets are drawn in the same script run (Tab 1's
    'Batch size' widget lives further down THIS SAME relocated block,
    and the array-index widgets live inside 'Option A', called even
    later) - Streamlit raises StreamlitAPIException if a widget's
    session_state is written AFTER that widget has already been
    instantiated in the same rerun. This is exactly why Requirement 1
    asks for this suggestion, and this button, to sit ABOVE 'Batch ID
    source'/'Batch size' in the first place. **Phase 2: do not move this
    function's call site.**

    Reads (does not draw) the 'HPC cluster profile' selection. Patch 3
    v4, Requirement 1: that selector now renders even further UP the
    page than this function - right under 'Run pipeline', immediately
    after choosing 'Option A' (see _render_hpc_cluster_profile_selector(),
    called by the top-level script before this function) - so by the
    time this runs, its session_state value already reflects the
    person's choice. On the very first render this session (before that
    widget has ever been drawn at all) this simply falls back to
    DEFAULT_HPC_PROFILE ('NCI Gadi')."""
    try:
        cfg = gather_config('Parallel', 'Step 1')
    except ValueError as exc:
        st.info(f"No job-array suggestion yet - finish configuring the pipeline first: {exc}")
        return
    if 'MODEL' not in cfg:
        return

    _hpc_profile = st.session_state.get('step1_hpc_profile', DEFAULT_HPC_PROFILE)
    _max_array_override = st.session_state.get('step1_hpc_max_array_override')
    _ncpus_per_gpu_override = st.session_state.get('step1_hpc_ncpus_per_gpu_override')

    try:
        n_population = count_unique_populations(cfg.get('PHENOTYPE_FILE_NAME')) or 1
        n_phenotype = len(cfg['PHENOTYPE']) if cfg['PHENOTYPE'] != 'all' else max(
            1, len(list_phenotype_columns(cfg.get('PHENOTYPE_FILE_NAME', '')))
        )
        n_ratio = len(cfg.get('RATIO', [1]))
        estimate = estimate_resources(
            scenario=cfg['SCENARIO'], n_population=n_population, n_phenotype=n_phenotype,
            n_ratio=n_ratio, sample_num=cfg.get('ITER_NUM', 1), model_run=cfg['MODEL'],
            ld_prune_enabled=cfg.get('LD_PRUNE') is not None, rf_filter_enabled=cfg.get('RF_FILTER') is not None,
            genotype_format=cfg.get('GENOTYPE_FORMAT', 'csv'), genotype_file_name=cfg.get('GENOTYPE_FILE_NAME', ''),
            phenotype_file_name=cfg.get('PHENOTYPE_FILE_NAME'), w_opt_enabled=cfg.get('W_OPT') is not None,
            # n_batches=1 here is fine regardless of what's actually
            # configured - suggested_batch_size/suggested_n_batches
            # describe the run's FULL task count, independent of the
            # n_batches argument (see suggest_array_layout()'s own
            # docstring in resource_profiles.py).
            n_batches=1, model_parallel_enabled=False,
            hparameters=cfg.get('HPARAMETERS'), hp_tune=cfg.get('HP_TUNE'),
            other_models_marker_source=cfg.get('OTHER_MODELS_MARKER_SOURCE', 'full_or_filtered'),
            pool_cost_in_headline=cfg.get('POOL_COST_IN_HEADLINE', True),
            gpu_cpu_calibration_in_headline=cfg.get('GPU_CPU_CALIBRATION_IN_HEADLINE', True),
            hpc_profile=_hpc_profile,
            max_array_subjobs_override=_max_array_override,
            ncpus_per_gpu_override=_ncpus_per_gpu_override,
        )
    except Exception as exc:
        st.caption(f"Could not compute a job-array suggestion yet ({exc}).")
        return

    total_tasks_all = int(estimate['total_tasks_all_batches'])
    max_array_subjobs = int(estimate['max_array_subjobs'])

    st.markdown("**Suggested job-array size**")
    st.metric('Total tasks (this run)', f"{total_tasks_all:,}")
    # ver4-4, R1: 'between' counts population pairs/triples, not
    # ratio x iteration - compute_total_tasks() (resource_profiles.py)
    # already branches on this; only the CAPTION text needs to say so.
    _formula = (
        "every population pair/triple (W_OPT-dependent) \u00d7 target phenotype"
        if cfg.get('SCENARIO') == 'between' else
        "every population \u00d7 phenotype \u00d7 ratio \u00d7 replicate combination"
    )
    st.caption(
        f"{_formula} configured above, never exceeding the **{estimate['hpc_profile']}** "
        f"profile's {max_array_subjobs}-subjob array cap (change this under 'HPC cluster "
        f"profile' above)."
    )

    layouts = enumerate_array_layouts(
        total_tasks_all, max_array_subjobs=max_array_subjobs,
        preferred_batch_size=int(estimate['suggested_batch_size']),
    )

    if len(layouts) < 2:
        # R1 §2.1.3's documented degradation: total_tasks_all is prime
        # (or too few in-range divisors exist under the array cap) to
        # offer a genuine choice - fall back to suggest_array_layout()'s
        # own closest-achievable-balance single answer, exactly as
        # before R1, and say so explicitly rather than silently.
        st.caption(
            f"\u2139\ufe0f {total_tasks_all:,} total task(s) has no second workable exact-dividing "
            f"batch/array pair under the {max_array_subjobs}-subjob cap - showing the closest "
            f"achievable balance instead."
        )
        chosen_batch_size = int(estimate['suggested_batch_size'])
        chosen_n_batches = int(estimate['suggested_n_batches'])
        ac1, ac2 = st.columns(2)
        ac1.metric('Batch size (tasks/batch)', chosen_batch_size)
        ac2.metric('Resulting job-array size (batches)', chosen_n_batches)
        for note in estimate['notes']:
            st.caption(f"\u2139\ufe0f {note}")
    else:
        _labels = [
            f"{p['batch_size']:,} task(s)/batch \u00d7 {p['n_batches']:,} batch(es)"
            + (" (recommended)" if p['is_recommended'] else "")
            for p in layouts
        ]
        _default_idx = next((i for i, p in enumerate(layouts) if p['is_recommended']), 0)
        _choice_idx = st.radio(
            'Batch size / array size (every option divides the total EXACTLY - no smaller '
            'trailing batch, whichever you pick)',
            options=list(range(len(layouts))), format_func=lambda i: _labels[i],
            index=_default_idx, key='_step1_array_layout_choice',
        )
        chosen_batch_size = int(layouts[_choice_idx]['batch_size'])
        chosen_n_batches = int(layouts[_choice_idx]['n_batches'])
        ac1, ac2 = st.columns(2)
        ac1.metric('Batch size (tasks/batch)', chosen_batch_size)
        ac2.metric('Resulting job-array size (batches)', chosen_n_batches)

    if st.button('Update these values (Batch size / Array end index)', key='_btn_step1_apply_array_layout'):
        st.session_state['batch_size'] = max(1, chosen_batch_size)
        st.session_state['step1_hpc_array_start'] = 0
        st.session_state['step1_hpc_array_end'] = max(0, chosen_n_batches - 1)
        st.success(
            "Applied - see 'Batch size' below, and 'Array start index' / 'Array end index' "
            "further down under 'Run pipeline' > 'Option A' > 'HPC job resource requests'."
        )


# Patch 3 v4, Requirement 2 - the mapping this section uses in BOTH
# directions to keep 'Batch ID source', 'Job scheduler', and 'HPC
# cluster profile' calibrated against each other. 'Manual integer' (a
# Batch ID source value) and 'Custom'/'Generic / other HPC' (an HPC
# cluster profile value) deliberately have NO entry here - they are
# legitimate, independent choices that this synchronisation leaves
# alone ("leave it as it is for other choices").
_SCHEDULER_TO_BATCH_ID_SOURCE = {
    'Slurm': 'Slurm (SLURM_ARRAY_TASK_ID)',
    'PBS': 'PBS (PBS_ARRAY_INDEX / PBS_ARRAYID)',
    # 'NCI' is a PBS-family scheduler (its array subjobs also read
    # PBS_ARRAY_INDEX - see render_hpc_export_section()'s own docstring)
    # so it maps to the SAME batch-ID-source label as plain 'PBS'.
    'NCI': 'PBS (PBS_ARRAY_INDEX / PBS_ARRAYID)',
}
_BATCH_ID_SOURCE_TO_SCHEDULER = {
    'Slurm (SLURM_ARRAY_TASK_ID)': 'Slurm',
    'PBS (PBS_ARRAY_INDEX / PBS_ARRAYID)': 'PBS',
}
_HPC_PROFILE_FORCED_SCHEDULER = {
    # Bunya is a Slurm-only cluster - there is no legitimate reason to
    # pick PBS/NCI directives for it.
    'UQ Bunya': 'Slurm',
    # Gadi runs PBS Pro. 'PBS' (rather than the more Gadi-specific 'NCI'
    # native-job-array generator) is what Requirement 2 asks for as the
    # SUGGESTED default here; a person who specifically wants Gadi's
    # native per-batch job-array submission (run_batch_array.pbs +
    # submit_arrays.sh - see render_hpc_export_section()'s own
    # docstring) can still switch 'Job scheduler' to 'NCI' by hand
    # afterwards - this only supplies the starting suggestion, exactly
    # like every other auto-fill in this section, it does not remove
    # 'NCI' as an option.
    'NCI Gadi': 'PBS',
}


def _apply_autofill_unless_overridden(key, suggested_value, autofill_key=None):
    """Shared helper for the "suggest a value, but never clobber an
    explicit manual choice" bookkeeping pattern already used throughout
    this section (e.g. 'CPUs per task', 'Intra-batch worker processes').

    Writes `suggested_value` into st.session_state[key] ONLY if the
    current value is unset, or exactly equals the last value THIS
    helper itself wrote there (i.e. it has never been knowingly
    overridden by the person since). Safe to call before OR after the
    widget with this key has been drawn in the current script run, in
    either order relative to other calls - it never raises, it just may
    be a no-op if the widget was already drawn with a DIFFERENT value in
    THIS SAME rerun (the one genuine Streamlit restriction, unrelated to
    this bookkeeping).
    """
    autofill_key = autofill_key or f'_{key}_last_autofill'
    current = st.session_state.get(key)
    last_autofill = st.session_state.get(autofill_key)
    if current is None or current == last_autofill:
        if current != suggested_value:
            st.session_state[key] = suggested_value
        st.session_state[autofill_key] = suggested_value


def _render_hpc_cluster_profile_selector(kp):
    """Patch 3 v4, Requirement 1 - the 'HPC cluster profile' selectbox,
    now drawn right after the Option A / Option B choice under 'Run
    pipeline' - "the second thing that will be asked" once 'Option A' is
    chosen - rather than buried inside render_hpc_export_section()'s own
    body further down. Every other Option-A widget that used to draw
    this selectbox itself (render_hpc_export_section()) or read it
    (_render_resource_advisor_panel(), _render_suggested_job_array_size(),
    _render_batch_id_source_selector() below) now simply reads this
    selectbox's session_state value instead.

    Patch 3 v4, Requirement 2 - also keeps 'Job scheduler' (drawn much
    further down, inside render_hpc_export_section()'s 'HPC job resource
    requests' expander) calibrated to the selected cluster, via
    _HPC_PROFILE_FORCED_SCHEDULER above: 'UQ Bunya' -> 'Slurm', 'NCI
    Gadi' -> 'PBS'. 'Generic / other HPC' and 'Custom' are left alone
    ("leave it as it is for other choices"). Uses
    _apply_autofill_unless_overridden() (the SAME auto-suggest-unless-
    manually-overridden bookkeeping used everywhere else in this
    section), so a person who deliberately picks something else
    afterwards (e.g. 'NCI', for Gadi's native per-batch job-array
    generator) keeps it - this only sets the starting suggestion, not a
    lock.
    """
    st.selectbox(
        'HPC cluster profile', options=list(HPC_RESOURCE_PROFILES.keys()) + ['Custom'],
        key=f'{kp}_hpc_profile',
        help="Which cluster's known limits the 'Suggested compute resources' estimate below "
             "(and the job-array-size suggestion under 'Parallel batch configuration', for "
             "Step 1) should calibrate against - specifically, the job-array subjob cap and "
             "whether/how many CPUs are required per GPU requested. 'NCI Gadi' matches this "
             "app's original, Gadi-only behaviour. 'UQ Bunya' has no single fixed "
             "CPUs-per-GPU ratio (its GPU nodes vary), so no CPU floor is applied for that "
             "profile. 'Generic / other HPC' is a conservative, no-assumptions fallback. "
             "'Custom' lets you type in your own two numbers directly. Choosing 'UQ Bunya' or "
             "'NCI Gadi' also suggests the matching 'Job scheduler' and 'Batch ID source' "
             "further down - you can still change either afterwards."
    )
    _selected_hpc_profile = st.session_state.get(f'{kp}_hpc_profile', DEFAULT_HPC_PROFILE)
    if _selected_hpc_profile == 'Custom':
        cp1, cp2 = st.columns(2)
        with cp1:
            st.number_input(
                'Custom job-array subjob cap', min_value=1, value=500, step=1,
                key=f'{kp}_hpc_max_array_override',
                help="The maximum number of subjobs your scheduler allows in one job array."
            )
        with cp2:
            st.number_input(
                'Custom CPUs required per GPU (0 = no fixed ratio)', min_value=0, value=0, step=1,
                key=f'{kp}_hpc_ncpus_per_gpu_override',
                help="0 = this cluster has no fixed CPUs-per-GPU rule (no automatic ncpus "
                     "floor is applied for GPU jobs, same as 'UQ Bunya')."
            )
    else:
        # Not 'Custom' - make sure a stale override from a PREVIOUS
        # 'Custom' selection this session can never silently leak into a
        # later named-profile estimate (resolve_hpc_profile() only reads
        # these two keys, so simply not setting/clearing them here is
        # sufficient - but an explicit clear is more robust against a
        # widget that was drawn earlier this session and left a value
        # behind in session_state before 'Custom' was un-selected).
        st.session_state.pop(f'{kp}_hpc_max_array_override', None)
        st.session_state.pop(f'{kp}_hpc_ncpus_per_gpu_override', None)

    # Requirement 2: profile -> scheduler suggestion. Writes into
    # f'{kp}_hpc_scheduler' BEFORE that widget is drawn later inside
    # render_hpc_export_section() - legal, and the same "write into a
    # not-yet-drawn widget's session_state" pattern used throughout this
    # section (e.g. the N_CPU_WORKERS auto-fill below).
    _suggested_scheduler = _HPC_PROFILE_FORCED_SCHEDULER.get(_selected_hpc_profile)
    if _suggested_scheduler:
        _apply_autofill_unless_overridden(f'{kp}_hpc_scheduler', _suggested_scheduler)


def _render_batch_id_source_selector(kp):
    """Patch 3 v4, Requirement 2 - 'Batch ID source', kept calibrated
    against 'Job scheduler' (drawn later, inside
    render_hpc_export_section()'s 'HPC job resource requests' expander)
    in BOTH directions, via _apply_autofill_unless_overridden():
      - on entry, suggests a value matching the CURRENT 'Job scheduler'
        (which, by this point, already reflects either the person's own
        last explicit choice there, or 'HPC cluster profile''s own
        suggestion from _render_hpc_cluster_profile_selector(), called
        earlier in this same script run);
      - after the widget below is drawn, if the person picked something
        that implies a DIFFERENT scheduler family (Slurm vs PBS/NCI),
        suggests that back onto 'Job scheduler' - legal because 'Job
        scheduler' is drawn LATER in the script than this function.
    'Manual integer' has no scheduler equivalent and is left alone in
    both directions - a legitimate, independent choice (e.g. a local
    single-batch test even while otherwise exporting for a real
    scheduler).

    Only meaningful for Parallel/Step 1 with 'Option A' selected - the
    caller only invokes this in that case. 'Option B' forces 'Manual
    integer' unconditionally instead, without drawing this selectbox at
    all - see the caller, under 'Run pipeline'.
    """
    _current_scheduler = st.session_state.get(f'{kp}_hpc_scheduler', 'Slurm')
    _suggested_batch_source = _SCHEDULER_TO_BATCH_ID_SOURCE.get(
        _current_scheduler, 'Slurm (SLURM_ARRAY_TASK_ID)'
    )
    _apply_autofill_unless_overridden(
        'batch_id_source', _suggested_batch_source, autofill_key='_batch_id_source_last_autofill'
    )

    st.selectbox(
        'Batch ID source', options=BATCH_ID_SOURCES, key='batch_id_source',
        help=(
            "'Manual integer' lets you type the batch ID directly - only used "
            "for a local single-batch test run below. "
            "'Slurm' and 'PBS' need no input here: the actual batch ID is "
            "assigned automatically at run time (SLURM_ARRAY_TASK_ID / "
            "PBS_ARRAY_INDEX, set per-task by the scheduler itself). Kept in "
            "sync with 'Job scheduler' below - picking one "
            "here (other than 'Manual integer') suggests the matching value "
            "there, and vice versa; you can still override either by hand."
        ),
    )
    batch_id_source = st.session_state.get('batch_id_source', BATCH_ID_SOURCES[0])

    # Requirement 2, other direction: an explicit Slurm/PBS pick here
    # suggests the matching 'Job scheduler' - legal because that
    # selectbox is drawn LATER in the script (inside
    # render_hpc_export_section()).
    _suggested_scheduler = _BATCH_ID_SOURCE_TO_SCHEDULER.get(batch_id_source)
    if _suggested_scheduler:
        _apply_autofill_unless_overridden(f'{kp}_hpc_scheduler', _suggested_scheduler)

    if batch_id_source == 'Manual integer':
        st.number_input('Batch ID', min_value=0, value=0, step=1, key='batch_id_manual',
                         help="Which batch (0-indexed) this particular run should process.")
    elif batch_id_source.startswith('Slurm'):
        st.caption(
            "No input needed - `run_step1_batch.py` will read `SLURM_ARRAY_TASK_ID` "
            "automatically on each array task."
        )
        env_val = os.environ.get('SLURM_ARRAY_TASK_ID')
        if env_val is not None:
            st.caption(f'\u2705 Detected in this environment right now: SLURM_ARRAY_TASK_ID = {env_val}')
    else:
        st.caption(
            "No input needed - `run_step1_batch.py` will read `PBS_ARRAY_INDEX` "
            "(or `PBS_ARRAYID`) automatically on each array task."
        )
        env_val = os.environ.get('PBS_ARRAY_INDEX', os.environ.get('PBS_ARRAYID'))
        if env_val is not None:
            st.caption(f'\u2705 Detected in this environment right now: PBS array index = {env_val}')


def _render_advanced_setting_components(kp: str) -> None:
    """ver4-4, R2 (blueprint §2.2.2) - the 'Advanced setting components'
    expander, rendered immediately after 'Additional setup lines
    (optional)' inside 'HPC job resource requests'. Hosts 'Intra-batch
    worker processes (N_CPU_WORKERS)' and 'Model-level workers
    (N_MODEL_WORKERS)', the two worker-fan-out widgets relocated out of
    the old 'Pipeline compute settings' block (which used to sit
    directly inside the expander body, always visible, with N_CPU_WORKERS
    silently auto-filled to match 'CPUs per task' the moment that field
    changed - see the R2 root-cause note on `_render_resource_advisor_
    panel()`). Both widgets here default to plain `value=1` and are
    NEVER auto-filled by any suggestion, anywhere - a person who wants
    worker fan-out sets it here, deliberately, rather than as a side
    effect of pressing an unrelated 'Use these values for CPUs per
    task / Memory / GPU' button.

    ver4-4, R3.b/c (Stage 3): also hosts four GLOBAL (not `kp`-scoped)
    widgets for `R_BLAS_THREADS`, `R_BLAS_FOLLOWS_N_JOBS`,
    `TORCH_NUM_THREADS` and `TORCH_DATALOADER_WORKERS`. These are GLOBAL
    session-state keys, not `{kp}_`-prefixed, deliberately mirroring the
    pre-existing `USE_GPU_SKLEARN` pattern (a single hardware setting for
    the whole run, not one that differs by export purpose) - and matching
    the SAME global key names `gather_config()` already reads
    (`r_blas_threads`, `r_blas_follows_n_jobs`, `torch_num_threads`,
    `torch_dataloader_workers`), written there since Stage 2, before any
    widget existed for them. Safe to render as GLOBAL keys from inside a
    function called once per `kp` - `render_hpc_export_section()` (and
    therefore this function) is itself called exactly once per Streamlit
    script run (a single `if is_step1: ... elif is_step2: ... else: ...`
    chain picks exactly one `kp` per run), so there is never more than
    one of these global-keyed widgets on screen at once.

    Streamlit does not allow an st.expander nested inside another
    st.expander (this is called from inside 'HPC job resource
    requests', itself an expander) - uses the same bordered
    st.container fix already established elsewhere in this section
    (e.g. the old 'Advanced settings' sub-section this replaces).

    Draws nothing else beyond the six widgets above - no auto-fill
    bookkeeping, no N_JOBS preview caption, no model-grouping-strategy
    sub-widgets (MODEL_GROUPING / BIO_PRIOR_GROUPING /
    MIN_MODELS_FOR_PARALLEL, previously only ever reachable behind the
    now-deleted 'Also consider model-level parallelism' checkbox, are
    still exported with their existing sensible defaults - see
    render_hpc_export_section()'s export block - just no longer exposed
    as GUI widgets; a config file can still set them by hand if the
    default grouping strategy isn't wanted).

    ver4-4, R3.g (Stage 6): also hosts one further GLOBAL checkbox,
    `sequential_intra_batch` (`SEQUENTIAL_INTRA_BATCH` - see
    run_sequential.py's own SEQUENTIAL_INTRA_BATCH branch), the
    blueprint's own one deliberate "default off" exception (§4a) among
    every ver4-4 config key - a control-flow change, not a numerics one,
    with its own disclosed scope/limitations (see the Change Summary).
    Rendered regardless of `kp` for the same reason the four torch/BLAS
    widgets above are (a single, global setting) - inert for Parallel
    Step 1/Step 2 exports, which never read this key.

    `kp` is the same purpose-scoped key prefix ('sequential' / 'step1' /
    'step2') every other widget in `render_hpc_export_section()` uses,
    so N_CPU_WORKERS/N_MODEL_WORKERS remain independently configurable
    per export context, exactly as before this restructuring. The four
    torch/BLAS widgets below (and the SEQUENTIAL_INTRA_BATCH checkbox at
    the very end) do NOT take a `kp` prefix, by design (see above).
    """
    with st.container(border=True):
        st.markdown(
            "**Advanced setting components** (worker fan-out - most runs never need this; "
            "see 'Suggested compute resources' above for the plain CPU/memory/GPU headline)"
        )
        st.number_input(
            'Intra-batch worker processes (N_CPU_WORKERS)', min_value=1, value=1, step=1,
            key=f'{kp}_n_cpu_workers',
            help="How many of this batch's own tasks run CONCURRENTLY, each in its own worker "
                 "process (see intra_batch_parallel.py). Leave at 1 to keep this batch's tasks "
                 "fully serial (still uses however many CPUs 'CPUs per task' reserves, but only "
                 "for a single task's own model fitting/tuning at a time)."
        )
        st.number_input(
            'Model-level workers (N_MODEL_WORKERS)', min_value=1, value=1, step=1,
            key=f'{kp}_n_model_workers',
            help="How many of ONE task's own models are fit CONCURRENTLY, each in its own "
                 "worker process (see intra_task_parallel.py) - composes with 'Intra-batch "
                 "worker processes' above on a single shared pool sized N_CPU_WORKERS_TASK x "
                 "N_MODEL_WORKERS. Leave at 1 to keep every task's models fit one at a time "
                 "(the default for every config, including every config saved before this "
                 "feature existed)."
        )

        st.divider()
        st.markdown(
            "**CPU threading (torch / R BLAS)** - applies to this whole run, regardless of "
            "which export purpose is being configured above."
        )
        st.number_input(
            'R BLAS threads (R_BLAS_THREADS)', value=None, min_value=1, step=1,
            key='r_blas_threads', placeholder='auto',
            help="Thread count exported as OPENBLAS_NUM_THREADS/OMP_NUM_THREADS/MKL_NUM_THREADS "
                 "before R/BGLR runs (rrBLUP/GBLUP/BayesB/RKHS's own MCMC linear algebra) - only "
                 "takes effect if the R build in use is linked against a multi-threaded BLAS; "
                 "EasiGP cannot force a different BLAS to be linked at runtime, this only forwards "
                 "a thread-count hint to whichever one is already present. Leave blank to use "
                 "N_JOBS instead (see 'Follow N_JOBS' below) or, if that is also off, today's "
                 "single-threaded default."
        )
        st.checkbox(
            'R BLAS threads follow N_JOBS (R_BLAS_FOLLOWS_N_JOBS)', value=True,
            key='r_blas_follows_n_jobs',
            help="When R BLAS threads (above) is left blank, use this run's own N_JOBS as the "
                 "R BLAS thread count instead of leaving BLAS single-threaded. On by default; "
                 "uncheck to keep R's BLAS threading independent of N_JOBS."
        )
        st.number_input(
            'Torch CPU threads (TORCH_NUM_THREADS)', value=None, min_value=1, step=1,
            key='torch_num_threads', placeholder='auto',
            help="torch.set_num_threads() for MLP/GAT model fitting on CPU (never used on a GPU "
                 "device). Leave blank to derive this automatically from N_JOBS/this run's own "
                 "worker-process layout."
        )
        st.number_input(
            'Torch DataLoader workers (TORCH_DATALOADER_WORKERS)', min_value=0, value=0, step=1,
            key='torch_dataloader_workers',
            help="Background worker processes for PyTorch's own DataLoader, used by the GAT/MLP "
                 "models. 0 (the default, and every config saved before this feature existed) "
                 "means data loading happens on the main process/thread, exactly as today."
        )

        st.divider()
        st.markdown(
            "**Sequential intra-batch parallelism** - applies only to Sequential-mode runs "
            "(`run_sequential.py`); inert (harmlessly ignored) for Parallel Step 1/Step 2 "
            "exports."
        )
        st.checkbox(
            'Fan Sequential mode\'s own tasks out across worker processes (SEQUENTIAL_INTRA_BATCH)',
            value=False, key='sequential_intra_batch',
            help="When on, and 'Intra-batch worker processes (N_CPU_WORKERS)' above is set above 1, "
                 "a Sequential run fans its own tasks out across that many worker processes "
                 "instead of GP()'s own fully serial per-task loop - the same mechanism a "
                 "Parallel-mode array-job batch already uses (intra_batch_parallel.py), applied "
                 "here to the WHOLE run as one synthetic batch. Off (the default, and every "
                 "config saved before this feature existed) reproduces today's exact, fully "
                 "serial Sequential behaviour. Leave off unless you have verified this route "
                 "for your own environment first."
        )


def render_hpc_export_section(export_mode, export_step, purpose, headless_script, config_filename, include_array):
    """Render the 'HPC job resource requests' expander and a single
    'Generate and save job files' button that writes the config JSON and
    the matching submission script(s) (for the chosen scheduler only)
    straight to disk - the config into Result/<RESULT_NAME>/, the
    script(s) next to streamlit_app.py. The saved script is also shown
    afterwards for reference/copying.

    Shared by Sequential, Parallel/Step 1, and Parallel/Step 2 so the GUI
    only ever needs to be configured once per run, regardless of mode.

    export_mode / export_step: passed straight to gather_config(...) to
        build the config that gets exported (e.g. 'Sequential'/None,
        'Parallel'/'Step 1', 'Parallel'/'Step 2').
    purpose: short slug ('sequential', 'step1', 'step2') used for default
        job names, widget key namespacing, and generated filenames.
    headless_script: filename of the standalone, non-interactive script the
        job should invoke (e.g. 'run_step1_batch.py').
    config_filename: filename (not full path) for the exported JSON config.
    include_array: whether this is an array job (Step 1, one task per
        batch) or a single job (Sequential / Step 2).

    Three scheduler choices are offered:
    - Slurm: unchanged, `#SBATCH` directives, `sbatch`/array (`-a`).
    - PBS: plain PBS, `#PBS` directives, `qsub`/array (`-J`) - the exact
      directive set/order used here matches a real, working PBS script
      (ncpus/mem as separate `-l` lines, `-P` for the project/account,
      optional `-l storage=...`, `-r y/n` rerunnable) rather than the
      generic `select=nodes:ncpus:mem` syntax some other PBS Pro sites
      use but this one doesn't.
    - NCI: for a single (non-array) job, identical to 'PBS' above - NCI's
      Gadi cluster runs plain PBS Pro for ordinary jobs too. For an ARRAY
      job (Step 1) specifically, `nci-parallel` task-farming (this
      section's previous approach) forces every co-packed batch to share
      ONE job's ncpus/mem, which does not scale once batch count is large
      or any individual batch (e.g. a GAT-heavy scenario) needs its own
      substantial memory. Instead, 'NCI' now generates TWO files
      implementing a **native PBS job array**, submitted as several small
      automated arrays rather than one call to `nci-parallel` and rather
      than one manual `qsub` per batch:

        * `run_batch_array.pbs` - a per-array PBS template using a real
          `#PBS -J` range, with `ncpus`/`mem` sized for exactly ONE batch
          (the same 'CPUs per task'/'Memory' fields used everywhere else
          in this section - for a native array, those already ARE the
          per-subjob request, so no separate 'per-batch' fields are
          needed). Each subjob computes its own EasiGP batch ID as
          `PBS_ARRAY_INDEX + OFFSET`, with `OFFSET` supplied via `-v`.
        * `submit_arrays.sh` - the submission driver: loops over the full
          batch-ID range in steps of 'Chunk size' (`CHUNK_SIZE`, capped at
          Gadi's current max-subjobs-per-array limit), `qsub`-ing
          `run_batch_array.pbs` once per chunk with a different `OFFSET`,
          and appending each submission's `{offset, array_job_id}` to
          `Result/<RESULT_NAME>/hpc_submission_manifest.json` for
          traceability (job IDs only exist once actually submitted, so
          this manifest is written by the submission script at qsub time,
          not by the GUI at export time).

      A config with N batches and a chosen chunk size C therefore results
      in `ceil(N/C)` array submissions when `submit_arrays.sh` is run,
      each independently resourced, with zero manual `qsub` edits
      required. Gadi's queue may still throttle how many jobs can be
      RUNNING at once regardless of how many were submitted - that is a
      scheduler-level policy EasiGP does not control.

    GPU support (all three schedulers): an optional 'Request GPU(s)'
    checkbox adds the scheduler's GPU-request directive to every
    generated script - `--gres=gpu:<type>:<count>` (or `--gres=gpu:<count>`
    if no type is given) for Slurm, `-l ngpus=<count>` for PBS/NCI (NCI
    Gadi's GPU queue is 'gpuvolta'; its ncpus must be a multiple of 12 per
    GPU requested - not enforced automatically here, so size 'CPUs per
    task' accordingly). For a Gadi native job-array, the GPU count is
    per-INDIVIDUAL-batch, exactly like ncpus/mem above.

    Patch 3 v2, Requirement 3 - Slurm QoS: Slurm ALSO gets an optional
    `--qos=...` directive ('QoS (Slurm only, optional)'), auto-suggested
    to 'gpu' the moment 'Request GPU(s)' is checked - some Slurm clusters
    (e.g. UQ RCC's Bunya - see
    https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md)
    gate GPU access behind a specific QoS in addition to the partition
    itself. Not applicable to PBS/NCI, which has no QoS directive anywhere
    in this generator.

    Array-task throttling (more array tasks than available GPUs): for any
    array job (Step 1, or a Gadi native array), 'Max concurrently running
    array tasks' appends a `%<limit>` slot-limit suffix to the array
    directive (`#SBATCH -a start-end%N` / `#PBS -J start-end%N`) - the
    scheduler then never runs more than N tasks of the array at once,
    however many are queued. The moment 'Request GPU(s)' is checked, this
    field is auto-suggested to equal 'GPUs per task' (so the array is
    throttled immediately, not left unthrottled by default) - the person
    exporting the job should then edit it to match the number of GPUs
    ACTUALLY available on the whole partition/queue, e.g. "12 array tasks
    each requesting 1 GPU, but only 4 GPUs exist on the partition" ->
    edit the suggested value up to 4, matching Slurm's own
    `--array=1-12%4`. For a Gadi native array, the same limit is applied
    inside each `submit_arrays.sh`-issued `qsub -J 0-<chunk size - 1>%N`
    call, capped at that call's own chunk size.
    """
    kp = purpose
    default_job_name = f"EasiGP_{st.session_state.get('result_name', 'job')}_{purpose}"
    default_max_index = 0

    # Patch 3 v4, Requirement 1: 'HPC cluster profile' used to be drawn
    # right HERE, as this function's first action. It now renders much
    # further up the page - right under 'Run pipeline', immediately after
    # choosing 'Option A' (see _render_hpc_cluster_profile_selector(),
    # called by the top-level script before this function is ever
    # reached) - so both this function and _render_resource_advisor_panel()
    # below simply READ its session_state value (f'{kp}_hpc_profile' etc.)
    # rather than drawing their own copy. Nothing else in this function
    # changes: every downstream read of f'{kp}_hpc_profile'/
    # f'{kp}_hpc_max_array_override'/f'{kp}_hpc_ncpus_per_gpu_override'
    # below is unchanged.
    _render_resource_advisor_panel(kp, export_mode, export_step)

    with st.expander('HPC job resource requests (customise the generated script)', expanded=False):
        st.caption(
            "'CPUs per task' / 'Memory' / 'GPU' below control BOTH the #SBATCH / #PBS "
            "resource-request directives in the generated script AND (via 'Pipeline compute "
            "settings' further down) the N_CPU_WORKERS/N_GPU_SLOTS/N_JOBS values written into "
            "the saved config JSON - so the scheduler allocation and what GP() actually uses "
            "stay in sync. Everything else in this expander (job name, walltime, partition, "
            "account, ...) affects the generated script only."
        )
        st.selectbox(
            'Job scheduler', options=['Slurm', 'PBS', 'NCI'], key=f'{kp}_hpc_scheduler',
            help=("Only script(s) for the chosen scheduler are generated below. 'NCI' targets "
                  "NCI's Gadi cluster specifically - for an array/batch job it generates a "
                  "native PBS job-array submission (`run_batch_array.pbs` + `submit_arrays.sh`, "
                  "see this section's own help text below) sized per-batch, rather than packing "
                  "every batch into one shared allocation; for a single job it's the same as "
                  "'PBS'. Suggested automatically from 'HPC cluster profile' above ('UQ Bunya' "
                  "-> 'Slurm', 'NCI Gadi' -> 'PBS') and kept in sync with 'Batch ID source' "
                  "under 'Parallel batch configuration', for Step 1 - change "
                  "it here any time, e.g. to 'NCI' for Gadi's native per-batch job-array "
                  "generator.")
        )
        scheduler = st.session_state.get(f'{kp}_hpc_scheduler', 'Slurm')
        is_pbs_family = scheduler in ('PBS', 'NCI')
        uses_gadi_array = (scheduler == 'NCI' and include_array)
        # Read (rather than render) the GPU checkbox's own state here,
        # BEFORE it is actually drawn further down - needed so the
        # 'Partition / queue' autofill below (which is rendered first,
        # inside the c1/c2/c3 columns) can already suggest a GPU queue
        # when the box is checked. Streamlit persists widget state in
        # session_state across reruns, so reading it early is safe -
        # this mirrors the existing 'scheduler'/'is_pbs_family' pattern.
        use_gpu = bool(st.session_state.get(f'{kp}_hpc_use_gpu', False))

        # Patch 3, Requirement 4 ("the suggestion of optimum CPUs needs to
        # be calibrated when using a GPU"): auto-suggest 'CPUs per task'
        # the moment 'Request GPU(s)' is checked, mirroring the SAME
        # auto-fill-unless-overridden pattern already used for 'Partition
        # / queue' below (read the not-yet-drawn GPU widgets' session
        # state early - safe, since Streamlit persists a widget's value
        # from the user's last interaction regardless of render order).
        # Previously this field's help text explicitly warned "this app
        # does not adjust 'CPUs per task' automatically" - this closes
        # that gap using the SAME calibration
        # resource_profiles._apply_gpu_cpu_calibration_to_core() applies
        # to the 'Suggested compute resources' panel above, so the two
        # never disagree. Patch 3 v3, Requirement 3: calibrated against
        # the SELECTED 'HPC cluster profile' (Patch 3 v4: drawn even
        # further up the page now - see _render_hpc_cluster_profile_selector())
        # rather than always Gadi's fixed ratio - a profile with no known
        # CPUs-per-GPU ratio (e.g. UQ
        # Bunya) applies no floor here either, matching the advisor
        # panel's own behaviour exactly.
        _gpus_per_task_now = int(st.session_state.get(f'{kp}_hpc_gpus_per_task', 1) or 1)
        _cpus_key = f'{kp}_hpc_cpus_per_task'
        _cpus_autofill_key = f'_{_cpus_key}_last_autofill'
        _profile_max_array, _profile_ncpus_per_gpu, _profile_note, _ = resolve_hpc_profile(
            st.session_state.get(f'{kp}_hpc_profile', DEFAULT_HPC_PROFILE),
            st.session_state.get(f'{kp}_hpc_max_array_override'),
            st.session_state.get(f'{kp}_hpc_ncpus_per_gpu_override'),
        )
        if use_gpu:
            if _profile_ncpus_per_gpu and _profile_ncpus_per_gpu > 0:
                _suggested_cpus = _profile_ncpus_per_gpu * _gpus_per_task_now
                if st.session_state.get('ld_prune_enabled', False) or st.session_state.get('rf_filter_enabled', False):
                    _suggested_cpus = int(math.ceil(_suggested_cpus * _GPU_PREPROCESS_CPU_HEADROOM))
                    _suggested_cpus = int(math.ceil(_suggested_cpus / (_profile_ncpus_per_gpu * _gpus_per_task_now))
                                           * (_profile_ncpus_per_gpu * _gpus_per_task_now))
            else:
                # No fixed CPUs-per-GPU ratio for this profile - only the
                # LD/RF headroom (if any) applies, no floor/rounding.
                _suggested_cpus = 1
                if st.session_state.get('ld_prune_enabled', False) or st.session_state.get('rf_filter_enabled', False):
                    _suggested_cpus = int(math.ceil(_suggested_cpus * _GPU_PREPROCESS_CPU_HEADROOM))
        else:
            _suggested_cpus = 1
        _current_cpus = st.session_state.get(_cpus_key)
        _last_cpus_autofill = st.session_state.get(_cpus_autofill_key)
        if _current_cpus is None or _current_cpus == _last_cpus_autofill:
            if _current_cpus != _suggested_cpus:
                st.session_state[_cpus_key] = _suggested_cpus
                st.session_state[_cpus_autofill_key] = _suggested_cpus

        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input('Job name', value=default_job_name, key=f'{kp}_hpc_job_name',
                          help="Label shown for this job in the scheduler's queue (e.g. `squeue`/`qstat`).")
            st.number_input('Nodes', min_value=1, value=1, step=1, key=f'{kp}_hpc_nodes',
                             help="How many physical machines to request. Almost always 1 for this pipeline, "
                                  "since it isn't written to split a single run across multiple machines."
                                  + (" Not used for Gadi native job-array jobs below." if uses_gadi_array else ""))
            st.number_input('Tasks per node', min_value=1, value=1, step=1, key=f'{kp}_hpc_ntasks_per_node',
                             help="How many separate processes to run per node. Leave at 1 unless you "
                                  "specifically know you need more."
                                  + (" Not used for Gadi native job-array jobs below." if uses_gadi_array else ""))
        with c2:
            st.number_input(
                'CPUs per task', min_value=1, step=1, key=_cpus_key,
                help="How many CPU cores to reserve for this job (e.g. for Random Forest's "
                     "parallel tree fitting). Auto-suggested to a Gadi-gpuvolta-compliant value "
                     "(a multiple of 12 per GPU, plus headroom if LD pruning/RF filtering is "
                     "also enabled) the moment 'Request GPU(s)' below is checked - type your own "
                     "value to override; it will stick across reruns."
                     + (" For a Gadi native job-array, this is the ncpus reserved for EACH "
                        "individual batch (every PBS subjob gets its own, independently-sized "
                        "allocation - not shared across batches)." if uses_gadi_array else "")
            )
            st.text_input('Memory (e.g. 10G)', value='10G', key=f'{kp}_hpc_mem',
                          help="How much RAM to reserve for this job."
                               + (" For a Gadi native job-array, this is the memory reserved for "
                                  "EACH individual batch (every PBS subjob gets its own allocation)."
                                  if uses_gadi_array else
                                  " Increase this for large genotype files."))
            st.text_input('Walltime (HH:MM:SS)', value='01:00:00', key=f'{kp}_hpc_time',
                          help="Maximum time the job is allowed to run before the scheduler kills it. "
                               "Set this generously - a job that hits this limit is stopped part-way through."
                               + (" For a Gadi native job-array, this is the budget for EACH batch "
                                  "(subjob), not the whole array." if uses_gadi_array else ""))
        with c3:
            # Requirement 7 (bugfix): text_input's own value= is only
            # ever used for a widget's VERY FIRST render - once its
            # session_state key already holds something (as it will
            # after ANY interaction with the page), later value=
            # arguments are silently ignored. That means switching FROM
            # Slurm TO PBS/NCI after the page has already rendered once
            # never actually updated this field's default before this
            # fix - it would stay stuck on whatever scheduler was
            # selected first. Uses the same 'auto-fill unless manually
            # overridden' pattern as _autofill_number_field() elsewhere
            # in this file, adapted for a text field whose default
            # depends on the scheduler rather than always being blank.
            _partition_key = f'{kp}_hpc_partition'
            _partition_autofill_key = f'_{_partition_key}_last_autofill'
            # GPU-aware suggestion: most clusters route GPU jobs through a
            # separate partition/queue from the CPU-only default (NCI
            # Gadi's GPU queue is specifically named 'gpuvolta'; Slurm
            # sites conventionally name theirs 'gpu', matching the
            # reference script this section was updated against - see
            # this function's docstring). Re-evaluated on every render, so
            # ticking/unticking 'Request GPU(s)' below updates this
            # suggestion immediately, same auto-fill-unless-overridden
            # pattern as the rest of this field.
            if is_pbs_family:
                _suggested_partition = 'gpuvolta' if use_gpu else 'normal'
            else:
                _suggested_partition = 'gpu' if use_gpu else 'general'
            _current_partition = st.session_state.get(_partition_key)
            _last_partition_autofill = st.session_state.get(_partition_autofill_key)
            if _current_partition is None or _current_partition == _last_partition_autofill:
                if _current_partition != _suggested_partition:
                    st.session_state[_partition_key] = _suggested_partition
                    st.session_state[_partition_autofill_key] = _suggested_partition
            st.text_input('Partition / queue', key=_partition_key,
                          help="Which partition/queue on your cluster to submit to - check with your "
                               "cluster's documentation or administrator for the available names "
                               "(NCI Gadi's general-purpose queue is 'normal', its GPU queue is "
                               "'gpuvolta'). Defaults to 'normal'/'gpuvolta' for PBS/NCI and "
                               "'general'/'gpu' for Slurm depending on whether 'Request GPU(s)' "
                               "below is checked - type your own value to override.")
            st.text_input('Account / project code (leave blank to omit)', value='', key=f'{kp}_hpc_account',
                          help="Billing/allocation code to charge this job's usage to. Slurm calls this "
                               "the 'account' (`--account`); PBS/NCI call it the 'project' (`-P`) - "
                               "same idea, different name.")
            st.checkbox('Use login shell (--login)', value=True, key=f'{kp}_hpc_login_shell',
                        help="Runs the job in a login shell, which loads your usual environment/module "
                             "setup (e.g. conda, R). Leave checked unless you know you need otherwise.")

        st.markdown("**GPU**")
        cg1, cg2, cg3 = st.columns(3)
        with cg1:
            st.checkbox(
                'Request GPU(s)', value=False, key=f'{kp}_hpc_use_gpu',
                help="Check this if the run needs GPU acceleration (the MLP model and any of the "
                     "four GAT_* models, which run in PyTorch/PyTorch Geometric, benefit most). "
                     "Adds the scheduler's GPU-request directive to the generated script(s) below - "
                     "`--gres=gpu:...` for Slurm, `-l ngpus=...` for PBS/NCI - and, unless you've "
                     "typed your own 'Partition / queue' above, switches the suggested queue to a "
                     "GPU one."
            )
        with cg2:
            st.number_input(
                'GPUs per task', min_value=1, value=1, step=1, key=f'{kp}_hpc_gpus_per_task',
                disabled=not use_gpu,
                help="How many GPUs to reserve for this job."
                     + (" For a Gadi native job-array, this is per INDIVIDUAL batch/subjob - every "
                        "subjob gets its own independent GPU allocation, exactly like 'CPUs per "
                        "task'/'Memory' above." if uses_gadi_array else
                        " For an array job, EVERY array task requests this many GPUs "
                        "independently - see 'Max concurrently running array tasks' below if you "
                        "have more array tasks than GPUs available on the partition."
                        if include_array else "")
            )
        with cg3:
            st.text_input(
                'GPU type (optional)', value='', key=f'{kp}_hpc_gpu_type',
                disabled=not use_gpu,
                help="Slurm only - a specific GPU model to request, e.g. 'a100' or 'v100' "
                     "(generates `--gres=gpu:<type>:<count>`). Leave blank to let the scheduler "
                     "assign any available GPU (`--gres=gpu:<count>`). Not used for PBS/NCI, which "
                     "request a GPU COUNT only (`-l ngpus=<count>`) - on NCI Gadi the GPU-enabled "
                     "queue is 'gpuvolta', where ncpus must be a multiple of 12 per GPU requested; "
                     "this app does not adjust 'CPUs per task' automatically, so size it "
                     "accordingly if you request GPUs there."
            )

        # Patch 3 v2, Requirement 3: "If GPU is set to > 0, Slurm should
        # generate the QoS field" (see
        # https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md).
        # Slurm's Quality-of-Service (`--qos=...`) is a separate concept
        # from 'Partition / queue' and, on some Slurm clusters, is REQUIRED
        # to actually get a GPU allocation - e.g. UQ RCC's Bunya cluster's
        # own documented GPU job template sets `--qos=gpu` alongside
        # `--partition=gpu_cuda`. PBS/NCI has no QoS directive anywhere in
        # this generator, so this is Slurm-only, matching the requirement's
        # own wording. Auto-suggested to 'gpu' the moment 'Request GPU(s)'
        # is checked (same auto-fill-unless-overridden pattern as
        # 'Partition / queue'/'CPUs per task' above) - blank omits `--qos`
        # entirely, which is also today's (pre-Patch-3-v2) behaviour for
        # every non-GPU job.
        if scheduler == 'Slurm':
            _qos_key = f'{kp}_hpc_qos'
            _qos_autofill_key = f'_{_qos_key}_last_autofill'
            _suggested_qos = 'gpu' if use_gpu else ''
            _current_qos = st.session_state.get(_qos_key)
            _last_qos_autofill = st.session_state.get(_qos_autofill_key)
            if _current_qos is None or _current_qos == _last_qos_autofill:
                if _current_qos != _suggested_qos:
                    st.session_state[_qos_key] = _suggested_qos
                    st.session_state[_qos_autofill_key] = _suggested_qos
            st.text_input(
                'QoS (Slurm only, optional - e.g. "gpu" on UQ Bunya)', key=_qos_key,
                help="Slurm's Quality-of-Service label (`--qos=...`). Separate from 'Partition / "
                     "queue' and often REQUIRED to actually obtain a GPU allocation on Slurm "
                     "clusters that gate GPU access behind a specific QoS (UQ RCC's Bunya, for "
                     "example, documents `--qos=gpu` alongside `--partition=gpu_cuda` for its own "
                     "GPU jobs - see "
                     "https://github.com/UQ-RCC/hpc-docs/blob/main/guides/Bunya-User-Guide.md). "
                     "Auto-suggested to 'gpu' the moment 'Request GPU(s)' is checked; leave blank "
                     "to omit `--qos` entirely and use your account's/partition's own default "
                     "(always the case for a non-GPU job)."
            )

        if is_pbs_family:
            c3b, c3c = st.columns(2)
            with c3b:
                st.text_input(
                    'Storage directive (optional, e.g. scratch/ab12+gdata/yz98)', value='',
                    key=f'{kp}_hpc_storage',
                    help=("PBS/NCI only (`-l storage=...`). Required on NCI Gadi whenever the job "
                          "needs to read/write under /scratch/<project>/ or /g/data/<project>/ - "
                          "list every filesystem area you need, joined with '+'. Leave blank to "
                          "omit entirely (e.g. if everything lives under your home directory).")
                )
            with c3c:
                st.checkbox(
                    'Rerunnable job (-r y)', value=True, key=f'{kp}_hpc_rerunnable',
                    help="PBS/NCI only. If checked, the scheduler is allowed to automatically "
                         "restart this job from the beginning if the compute node it's running on "
                         "fails - safe for this pipeline since a restarted batch/run just "
                         "reprocesses its own tasks from scratch (or resumes from a checkpoint, "
                         "if one was saved before the failure - see genomic_prediction.py's "
                         "checkpoint/resume support)."
                )

        c4, c5 = st.columns(2)
        with c4:
            st.text_input(
                'Output file base name', value='', key=f'{kp}_hpc_output_base',
                help=f"Leave blank to default to '{default_job_name}'. "
                     + ("The array-task index placeholder is appended automatically."
                        if include_array else "")
            )
        with c5:
            st.text_input(
                'Error file base name', value='', key=f'{kp}_hpc_error_base',
                help=f"Leave blank to default to '{default_job_name}'."
            )

        if include_array:
            c6, c7 = st.columns(2)
            with c6:
                st.number_input('Array start index', min_value=0, value=0, step=1, key=f'{kp}_hpc_array_start',
                                 help="The first batch ID to submit (usually 0).")
            with c7:
                st.number_input(
                    'Array end index', min_value=0, value=default_max_index, step=1, key=f'{kp}_hpc_array_end',
                    help="The last batch ID to submit (inclusive)."
                )
            st.caption(
                "Set the array index range to match your total number of batches "
                "(e.g. 0-999 for 1000 batches) - also useful for resubmitting only a "
                "subset of failed batches."
                + (" For a Gadi native job-array, this full range is split automatically into "
                   "'Chunk size'-sized PBS arrays by submit_arrays.sh below." if uses_gadi_array else "")
            )

            # Auto-suggest a concurrency limit as soon as 'Request GPU(s)'
            # is ticked - matches the Slurm reference pattern this section
            # was updated against ('--gres=gpu:1' PLUS '--array=...%4' to
            # cap the WHOLE array at 4 GPUs total; without the '%N' suffix,
            # nothing stops the scheduler running all array tasks at once
            # and over-subscribing the partition's GPUs). Same
            # auto-fill-unless-manually-overridden pattern as
            # 'Partition / queue' above: typing a different number here
            # sticks across reruns, and only reverts to auto-suggestion if
            # 'Request GPU(s)' is unticked again (or 'GPUs per task'
            # changes back to what was last auto-applied).
            #
            # Bugfix: this field can already hold a value in
            # session_state (e.g. its own widget default of 0 from a
            # session that started before auto-suggestion existed, or a
            # value restored by load_gui_state()) even though the
            # AUTOFILL mechanism itself has never actually run yet for
            # this session - '_throttle_autofill_key not in
            # st.session_state' is what detects that "never run before"
            # state. Without this check, a pre-existing 0 reads as
            # indistinguishable from "the person deliberately chose 0",
            # so the very first suggestion after GPUs are requested was
            # silently dropped and '-a'/'-J' kept generating with no
            # '%N' suffix even with --gres/-l ngpus correctly set - this
            # is the bug reported after the previous fix.
            #
            # Patch 3 v2, Requirement 5 ("the default of 'Max concurrently
            # running array tasks' should correspond to the number of GPUs
            # allocated; no GPUs should always be set to zero"): this is
            # EXACTLY what the block below already does
            # (`_suggested_throttle = _gpus_per_task_now if use_gpu else 0`)
            # - verified against the requirement text this session, no
            # change was needed here.
            _throttle_key = f'{kp}_hpc_array_throttle'
            _throttle_autofill_key = f'_{_throttle_key}_last_autofill'
            _gpus_per_task_now = int(st.session_state.get(f'{kp}_hpc_gpus_per_task', 1))
            _suggested_throttle = _gpus_per_task_now if use_gpu else 0
            _current_throttle = st.session_state.get(_throttle_key)
            _last_throttle_autofill = st.session_state.get(_throttle_autofill_key)
            _throttle_never_autofilled = _throttle_autofill_key not in st.session_state
            if _current_throttle is None or _current_throttle == _last_throttle_autofill \
                    or _throttle_never_autofilled:
                if _current_throttle != _suggested_throttle:
                    st.session_state[_throttle_key] = _suggested_throttle
                    st.session_state[_throttle_autofill_key] = _suggested_throttle
            st.number_input(
                'Max concurrently running array tasks (optional, 0 = no limit)', min_value=0,
                step=1, key=_throttle_key,
                help=(
                    "Caps how many array tasks the scheduler will run AT ONCE, however many are "
                    "queued - the standard fix when an array has more tasks than a resource each "
                    "task needs, most commonly GPUs. Auto-suggested to match 'GPUs per task' above "
                    "the moment 'Request GPU(s)' is checked, so the array directive is throttled "
                    "immediately - but EDIT THIS to match the number of GPUs actually available on "
                    "your partition/queue AS A WHOLE (e.g. 12 array tasks, 1 GPU each, but only 4 "
                    "GPUs exist on the partition -> set this to 4, matching Slurm's own "
                    "'--array=1-12%4', not left at the auto-suggested 1). Generates a `%<limit>` "
                    "suffix on the array directive (`#SBATCH -a start-end%<limit>` / `#PBS -J "
                    "start-end%<limit>`). Set to 0 to let every task start as soon as the "
                    "scheduler can place it (no GPU-oversubscription protection)."
                    + (" For a Gadi native job-array, the same limit is applied inside EACH "
                       "`submit_arrays.sh`-issued `qsub -J 0-<chunk size - 1>%<limit>` call "
                       "(automatically capped at that call's own 'Chunk size', since a single "
                       "PBS array can't usefully be throttled below its own size)."
                       if uses_gadi_array else "")
                )
            )

            if uses_gadi_array:
                st.markdown("**Gadi native job-array submission**")
                st.caption(
                    "Generates a per-batch-resourced native PBS job array (`#PBS -J`), submitted "
                    "as several small automated arrays via a generated `submit_arrays.sh` driver - "
                    "the scalable pattern recommended for large or memory-heavy batch counts on "
                    "Gadi, in place of packing every batch into one shared `nci-parallel` "
                    "allocation."
                )
                st.number_input(
                    'Chunk size (subjobs per PBS array submission)', min_value=1,
                    max_value=GADI_MAX_ARRAY_SUBJOBS,
                    value=min(10, GADI_MAX_ARRAY_SUBJOBS), step=1,
                    key=f'{kp}_hpc_gadi_chunk_size',
                    help=(
                        f"How many batches go into EACH individual `qsub`-submitted PBS array. "
                        f"`submit_arrays.sh` submits `ceil(N_batches / chunk size)` separate "
                        f"arrays automatically, one after another - capped here at "
                        f"{GADI_MAX_ARRAY_SUBJOBS} (verify NCI Gadi's CURRENT maximum subjob "
                        f"count per array before relying on this value; it may have changed)."
                    )
                )

        st.text_area(
            'Additional setup lines (optional)', value='', key=f'{kp}_hpc_extra_lines', height=100,
            help=(
                "One shell command per line, inserted after the resource directives "
                "and before the pipeline command - e.g. loading modules or activating "
                "a Conda environment:\n"
                "module load R/4.4.0\n"
                "source activate easigp_env"
            ),
        )

        # Requirements.md item 5: 'Advanced setting components' (worker
        # fan-out) is now revealed only after this button is clicked,
        # rather than always rendered - most runs never need it (see
        # _render_advanced_setting_components()'s own docstring), and the
        # plain 'Suggested compute resources' panel above already covers
        # the CPU/memory/GPU headline every run does need. A persistent
        # session_state flag (not the button's own transient click-state)
        # keeps the section visible across reruns once shown, and the
        # button toggles it back off if clicked again.
        _advanced_shown_key = f'{kp}_show_advanced_setting_components'
        if st.button(
            ('Hide' if st.session_state.get(_advanced_shown_key, False) else 'Show')
            + ' advanced setting components (worker fan-out)',
            key=f'_btn_{kp}_toggle_advanced_setting_components',
            help="Worker fan-out (N_CPU_WORKERS/N_MODEL_WORKERS) and CPU-threading settings - "
                 "most runs never need these; see 'Suggested compute resources' above for the "
                 "plain CPU/memory/GPU headline most runs actually want."
        ):
            st.session_state[_advanced_shown_key] = not st.session_state.get(_advanced_shown_key, False)
            st.rerun()

        if st.session_state.get(_advanced_shown_key, False):
            _render_advanced_setting_components(kp)

    if st.button('Generate and save job files', key=f'_btn_{kp}_generate_save'):
        try:
            export_cfg = gather_config(export_mode, export_step)
        except ValueError as exc:
            st.error(str(exc))
            return

        result_dir = result_dir_path(export_cfg['RESULT_NAME'])
        os.makedirs(result_dir, exist_ok=True)
        config_path = os.path.join(result_dir, config_filename)
        logs_dir = os.path.join(result_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        if include_array:
            # batch_id is per-task and must NOT be baked into the shared
            # config - the headless script resolves it at run time from
            # SLURM_ARRAY_TASK_ID / PBS_ARRAY_INDEX (+ OFFSET, for a Gadi
            # native job-array - see pipeline_utils.resolve_batch_id_from_env),
            # or is passed explicitly via --batch-id (used directly by the
            # generated Gadi array script below) for each individual task.
            export_cfg['PARALLEL'] = {'batch_size': export_cfg['PARALLEL']['batch_size']}

        # Phase 2, Requirement 7 (bugfix): the 'Pipeline compute settings'
        # block above (CPUs per task -> N_CPU_WORKERS/N_JOBS, GPUs per task
        # -> N_GPU_SLOTS) previously only ever reached the generated
        # #SBATCH/#PBS script, never this JSON - meaning run_step1_batch.py
        # always saw N_CPU_WORKERS default to 1 (no intra-batch
        # parallelism at all) no matter how many CPUs were requested above.
        # Written here, right before the file is saved, from the SAME
        # session_state values the resource-request fields themselves use,
        # so the two can never drift apart.
        _export_cpus_per_task = int(st.session_state.get(f'{kp}_hpc_cpus_per_task', 1))
        _export_n_cpu_workers = int(st.session_state.get(f'{kp}_n_cpu_workers', _export_cpus_per_task) or 1)
        _export_n_cpu_workers = max(1, min(_export_n_cpu_workers, _export_cpus_per_task))
        # Requirement.md item 1 fix: also cap N_CPU_WORKERS by how many
        # tasks THIS batch will actually contain (export_cfg['PARALLEL']
        # ['batch_size'], just resolved above for an array export) -
        # task-level fan-out can never usefully run more concurrent
        # workers than there are tasks to hand them (intra_batch_
        # parallel.py's own run_batch_with_intra_batch_parallelism()
        # already falls back to a single serial GP() call whenever
        # batch_size < min_tasks_for_parallel, so any N_CPU_WORKERS above
        # batch_size buys nothing at run time). Without this cap, N_JOBS/
        # PLINK_THREADS below get divided by a worker count that will
        # never actually run - e.g. a one-task-per-array-job GPU
        # submission (batch_size=1, the natural shape when each task
        # needs its own GPU - the common case for GAT_biological_prior_
        # knowledge) paired with N_CPU_WORKERS sized for a many-task batch
        # silently divides this run's ENTIRE CPU allocation down to
        # N_JOBS=PLINK_THREADS=1, starving the CPU-bound LD pruning/RF
        # filtering that model's own data-driven merge (architecture doc
        # §12.4) performs over the full marker pool - reported as "LD
        # filtering and RF filtering take too much time when using a GPU
        # and a GAT biological prior knowledge model". Capping here
        # instead redirects that otherwise-wasted worker capacity into
        # N_JOBS/PLINK_THREADS, matching the policy resource_profiles.py's
        # own suggest_compute_layout() already documents ("task-level
        # width is grown FIRST, up to min(total_tasks_per_batch, budget)
        # ... only once every task in the batch already has its own
        # concurrent worker does any REMAINING CPU budget get spent on
        # n_jobs") - that function is unreferenced dead code (see its own
        # docstring), so this is the one place that policy actually needs
        # enforcing. Only applies for an array export, where batch_size is
        # a real per-batch task count (see the `if include_array:` block
        # immediately above); a Sequential export has no such number here.
        if include_array:
            _export_batch_size = max(1, int(export_cfg['PARALLEL'].get('batch_size', 1) or 1))
            _export_n_cpu_workers = max(1, min(_export_n_cpu_workers, _export_batch_size))
        _export_use_gpu = bool(st.session_state.get(f'{kp}_hpc_use_gpu', False))
        _export_gpus_per_task = int(st.session_state.get(f'{kp}_hpc_gpus_per_task', 1))
        export_cfg['N_CPU_WORKERS'] = _export_n_cpu_workers
        export_cfg['N_JOBS'] = max(1, _export_cpus_per_task // _export_n_cpu_workers)
        export_cfg['N_GPU_SLOTS'] = _export_gpus_per_task if _export_use_gpu else 0
        # Bugfix (companion to the N_JOBS fix immediately above): PLINK_THREADS
        # was never written into the exported config at all, so it silently
        # stayed at resolve_compute_resources()'s own default of 1 no matter
        # how many CPUs were reserved above - meaning every plink2 subprocess
        # call inside LD pruning fell back to plink2's own hardware
        # auto-detection (not cgroup-aware) instead of the actual per-job
        # allocation. Most consequential for GPU jobs, which typically
        # request far fewer CPUs than a CPU-only job of the same size: plink2
        # would still try to use threads sized to the whole node, causing
        # severe oversubscription/thrashing rather than the intended
        # single-task allocation. Kept in sync with N_JOBS (same
        # CPUs-per-task / worker-count division), same as every other
        # 'Pipeline compute settings' key on this page.
        export_cfg['PLINK_THREADS'] = max(1, _export_cpus_per_task // _export_n_cpu_workers)

        # Update ID 2 (R1/R2, blueprint T12/T20): the model-level-
        # parallelism counterpart of the N_CPU_WORKERS/N_JOBS/N_GPU_SLOTS
        # block just above - written the SAME way, for the SAME
        # documented reason (existing code behaviour over a literal
        # reading of the blueprint - Phase2_Coding_2.md §1): gather_config()
        # itself does not set any compute-settings key, so these are
        # written here, right before the file is saved, from the SAME
        # session_state value the 'Model-level workers (N_MODEL_WORKERS)'
        # widget uses (now inside `_render_advanced_setting_components()`
        # - ver4-4, R2), mirroring N_CPU_WORKERS/N_JOBS/N_GPU_SLOTS exactly.
        # N_JOBS above is intentionally divided by N_CPU_WORKERS only, NOT
        # also by N_MODEL_WORKERS here - the second division happens at
        # RUN TIME instead (run_step1_batch.py's own orchestration-level
        # call to pipeline_utils.resolve_compute_resources()), so a
        # config with N_MODEL_WORKERS left at its default (1) writes an
        # N_JOBS byte-identical to a config saved before this feature
        # existed (A1.5/A2.2).
        # ver4-4, R2: the widget itself now always renders with
        # value=1 and is never auto-filled by any suggestion (the
        # opt-in checkbox that used to gate whether this widget was
        # even visible is gone - see _render_resource_advisor_panel())
        # - so a session that never opens 'Advanced setting components' at
        # all still exports EXACTLY N_MODEL_WORKERS=1, and reading the
        # widget's own session_state (falling back to 1 if the expander
        # was never opened this session) is now sufficient on its own.
        _export_n_model_workers = int(st.session_state.get(f'{kp}_n_model_workers', 1) or 1)
        export_cfg['N_CPU_WORKERS_TASK'] = _export_n_cpu_workers
        export_cfg['N_MODEL_WORKERS'] = _export_n_model_workers
        export_cfg['MODEL_GROUPING'] = st.session_state.get(f'{kp}_model_grouping', 'cost_balanced')
        export_cfg['BIO_PRIOR_GROUPING'] = st.session_state.get(f'{kp}_bio_prior_grouping', 'affinity')
        export_cfg['MIN_MODELS_FOR_PARALLEL'] = int(st.session_state.get(f'{kp}_min_models_for_parallel', 2) or 2)
        export_cfg['GPU_SLOTS_PER_DEVICE'] = 1

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(export_cfg, f, indent=2)

        scheduler = st.session_state.get(f'{kp}_hpc_scheduler', 'Slurm')
        is_pbs_family = scheduler in ('PBS', 'NCI')
        uses_gadi_array = (scheduler == 'NCI' and include_array)
        job_name = st.session_state.get(f'{kp}_hpc_job_name', '').strip() or default_job_name
        nodes = int(st.session_state.get(f'{kp}_hpc_nodes', 1))
        ntasks_per_node = int(st.session_state.get(f'{kp}_hpc_ntasks_per_node', 1))
        cpus_per_task = int(st.session_state.get(f'{kp}_hpc_cpus_per_task', 1))
        mem = st.session_state.get(f'{kp}_hpc_mem', '10G').strip() or '10G'
        walltime = st.session_state.get(f'{kp}_hpc_time', '01:00:00').strip() or '01:00:00'
        partition = st.session_state.get(f'{kp}_hpc_partition', 'general').strip() or 'general'
        account = st.session_state.get(f'{kp}_hpc_account', '').strip()
        login_shell = bool(st.session_state.get(f'{kp}_hpc_login_shell', True))
        storage = st.session_state.get(f'{kp}_hpc_storage', '').strip()
        rerunnable = bool(st.session_state.get(f'{kp}_hpc_rerunnable', True))
        output_base = st.session_state.get(f'{kp}_hpc_output_base', '').strip() or default_job_name
        error_base = st.session_state.get(f'{kp}_hpc_error_base', '').strip() or default_job_name
        extra_lines = st.session_state.get(f'{kp}_hpc_extra_lines', '').strip()
        extra_block = (extra_lines + '\n') if extra_lines else ''

        # GPU request, read the same way as every other resource field
        # above. gpu_type only ever affects Slurm's --gres syntax; PBS/NCI
        # request a bare GPU count via -l ngpus= (see this function's
        # docstring for why a type isn't requested there).
        use_gpu = bool(st.session_state.get(f'{kp}_hpc_use_gpu', False))
        gpus_per_task = int(st.session_state.get(f'{kp}_hpc_gpus_per_task', 1))
        gpu_type = st.session_state.get(f'{kp}_hpc_gpu_type', '').strip()
        # Patch 3 v2, Requirement 3: Slurm-only QoS (`--qos=...`) - see the
        # widget's own help text above for why (Bunya-style GPU QoS gating).
        # Read regardless of scheduler (mirrors every other field's
        # read-unconditionally-then-branch-on-scheduler pattern below);
        # only ever emitted into the Slurm script text, and only when
        # non-blank.
        qos = st.session_state.get(f'{kp}_hpc_qos', '').strip()

        shebang = '#!/bin/bash --login' if login_shell else '#!/bin/bash'
        account_line_slurm = f'#SBATCH --account={account}\n' if account else ''
        project_line_pbs = f'#PBS -P {account}\n' if account else ''
        storage_line_pbs = f'#PBS -l storage={storage}\n' if storage else ''
        rerunnable_line_pbs = f"#PBS -r {'y' if rerunnable else 'n'}\n"
        if use_gpu:
            gpu_line_slurm = (f'#SBATCH --gres=gpu:{gpu_type}:{gpus_per_task}\n' if gpu_type
                               else f'#SBATCH --gres=gpu:{gpus_per_task}\n')
            gpu_line_pbs = f'#PBS -l ngpus={gpus_per_task}\n'
        else:
            gpu_line_slurm = ''
            gpu_line_pbs = ''
        # Patch 3 v2, Requirement 3: "If GPU is set to > 0, Slurm should
        # generate the QoS field" - emitted only when both `use_gpu` and a
        # non-blank `qos` are set (a person can still clear the field to
        # omit `--qos` even for a GPU job, e.g. on a cluster where QoS
        # isn't gated by GPU access at all).
        qos_line_slurm = f'#SBATCH --qos={qos}\n' if (use_gpu and qos) else ''

        if include_array:
            array_start = int(st.session_state.get(f'{kp}_hpc_array_start', 0))
            array_end = int(st.session_state.get(f'{kp}_hpc_array_end', default_max_index))
            # Optional slot-limit ('%N') on the array directive, so the
            # scheduler never runs more than N tasks of the array at once -
            # the fix for having more array tasks than a shared resource
            # (typically GPUs) can support simultaneously. 0 = no limit.
            array_throttle = max(0, int(st.session_state.get(f'{kp}_hpc_array_throttle', 0)))
        else:
            array_start = array_end = 0
            array_throttle = 0

        project_dir = os.path.dirname(os.path.abspath(__file__))
        # Absolute path to the headless script, used (instead of a bare
        # filename) in every generated script below. Bugfix: the previous
        # generator invoked `python3 {headless_script}` with no path at
        # all, relying entirely on the job actually starting with cwd ==
        # `project_dir` - true only if the scheduler's own submission-
        # directory variable ($PBS_O_WORKDIR / $SLURM_SUBMIT_DIR) happens
        # to match it, which is not guaranteed (e.g. `submit_arrays.sh`
        # run from a different directory than the EasiGP install
        # directory). Every script generated below now BOTH explicitly
        # `cd`s into `project_dir` (a literal absolute path baked in at
        # generation time, not a scheduler-supplied variable) AND invokes
        # the headless script by its full absolute path, so the job runs
        # correctly regardless of where/how it was submitted from.
        headless_script_path = os.path.join(project_dir, headless_script)
        generated_paths = []  # every file written below, for the final success message

        if uses_gadi_array:
            # ---------------------------------------------------------- #
            # Gadi native job-array (Step 1 only, Requirement 1): TWO
            # files - a per-array PBS template sized for exactly ONE
            # batch's ncpus/mem, and a submission driver that qsub's it
            # once per OFFSET, in 'Chunk size'-sized array submissions,
            # covering the whole array_start..array_end range with
            # ceil(N/CHUNK_SIZE) separate qsub calls - no nci-parallel,
            # no per-batch manual qsub.
            # ---------------------------------------------------------- #
            chunk_size = int(st.session_state.get(f'{kp}_hpc_gadi_chunk_size', min(100, GADI_MAX_ARRAY_SUBJOBS)))
            chunk_size = max(1, min(chunk_size, GADI_MAX_ARRAY_SUBJOBS))
            # Slot-limit applied inside EACH qsub call below is capped at
            # that call's own chunk size - a PBS array can't usefully be
            # throttled below its own subjob count, and a limit bigger
            # than the array is simply a no-op, so capping here keeps the
            # generated qsub commands sane regardless of what was typed
            # into the GUI.
            gadi_throttle = min(array_throttle, chunk_size) if array_throttle > 0 else 0

            pbs_out_pattern = os.path.join(logs_dir, f'{output_base}_^array_index^.output')
            pbs_err_pattern = os.path.join(logs_dir, f'{error_base}_^array_index^.error')
            manifest_path = os.path.join(result_dir, 'hpc_submission_manifest.json')

            pbs_script_filename = f'{job_name}_run_batch_array.pbs'
            pbs_script_path = os.path.join(project_dir, pbs_script_filename)
            pbs_script_text = f"""{shebang}
# EasiGP - {purpose} job (Gadi native job-array batch template)
#PBS -l ncpus={cpus_per_task}
#PBS -l mem={mem}
#PBS -N {job_name}
#PBS -l walltime={walltime}
#PBS -q {partition}
{project_line_pbs}{storage_line_pbs}#PBS -o {pbs_out_pattern}
#PBS -e {pbs_err_pattern}
{rerunnable_line_pbs}{gpu_line_pbs}#PBS -l wd

# NOT meant to be submitted directly with `qsub {pbs_script_filename}` -
# submit_arrays.sh (generated alongside this file) submits it repeatedly,
# once per OFFSET, each call covering up to {chunk_size} batch(es) via
# `-J 0-<chunk size - 1>{f'%{gadi_throttle}' if gadi_throttle else ''} -v OFFSET=<n>`.
# ncpus/mem{'/ngpus' if use_gpu else ''} above are sized for exactly ONE
# batch - every PBS subjob gets its own independent allocation, never
# shared with any other batch.
#
# This subjob's actual EasiGP batch ID is PBS_ARRAY_INDEX + OFFSET.

# Bugfix: `cd` explicitly into the EasiGP install directory (a literal
# absolute path baked in at generation time), NOT `$PBS_O_WORKDIR` - that
# variable is set from wherever `qsub`/`submit_arrays.sh` was actually
# invoked, which need not be this directory. Getting cwd wrong here breaks
# more than just finding this script: genomic_prediction.py's own R-model
# sourcing and this run's `Result/` output location are also resolved
# relative to cwd.
cd "{project_dir}"

ACTUAL_BATCH_ID=$((PBS_ARRAY_INDEX + OFFSET))
if [ "$ACTUAL_BATCH_ID" -gt {array_end} ]; then
    echo "EasiGP: ACTUAL_BATCH_ID=$ACTUAL_BATCH_ID exceeds the configured array end index ({array_end}) - this subjob is padding beyond the last real batch; exiting cleanly."
    exit 0
fi

{extra_block}python3 "{headless_script_path}" --config "{config_path}" --batch-id "$ACTUAL_BATCH_ID"
"""
            with open(pbs_script_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(pbs_script_text)
            try:
                os.chmod(pbs_script_path, 0o755)
            except OSError:
                pass
            generated_paths.append(pbs_script_path)

            submit_script_filename = f'{job_name}_submit_arrays.sh'
            submit_script_path = os.path.join(project_dir, submit_script_filename)
            _gadi_throttle_suffix = f'%{gadi_throttle}' if gadi_throttle else ''
            _gadi_throttle_comment = (
                "#\n"
                f"# Each qsub call below is ALSO explicitly capped at {gadi_throttle} concurrently\n"
                f"# running subjob(s) (a '{_gadi_throttle_suffix}' slot limit on '-J') - set from this\n"
                "# section's 'Max concurrently running array tasks' field, most commonly used to\n"
                "# avoid requesting more GPUs at once than the 'gpuvolta' queue can actually give\n"
                "# this job.\n"
            ) if gadi_throttle else ""
            submit_script_text = f"""#!/bin/bash
# EasiGP - submission driver for '{job_name}' (Gadi native job-array).
# Submits batches {array_start}..{array_end} as ceil(N/{chunk_size}) separate
# small PBS job arrays (chunk size {chunk_size}), each independently
# resourced ({cpus_per_task} ncpus, {mem} mem{f', {gpus_per_task} gpu(s)' if use_gpu else ''} PER BATCH) via
# {pbs_script_filename} - no manual qsub editing required. Run this ONCE,
# from the login node (or wherever `qsub` is available):
#
#     bash {submit_script_filename}
#
# If `qsub` rejects a chunk as too large, NCI Gadi's current max-subjobs-
# per-array limit is lower than {chunk_size} - lower 'Chunk size' in the
# GUI and regenerate. The queue may also throttle how many jobs can be
# RUNNING at once regardless of how many are submitted - a scheduler-level
# policy, not something EasiGP controls.
{_gadi_throttle_comment}
set -euo pipefail

PBS_SCRIPT="{pbs_script_path}"
MANIFEST="{manifest_path}"

echo "[" > "$MANIFEST"
first=1
for OFFSET in $(seq {array_start} {chunk_size} {array_end}); do
    END=$((OFFSET + {chunk_size} - 1))
    if [ "$END" -gt {array_end} ]; then
        END={array_end}
    fi
    SIZE=$((END - OFFSET + 1))
    JOB_ID=$(qsub -J 0-$((SIZE - 1)){_gadi_throttle_suffix} -v OFFSET=$OFFSET "$PBS_SCRIPT")
    echo "Submitted batches $OFFSET..$END as array job $JOB_ID"
    if [ "$first" -eq 0 ]; then
        echo "," >> "$MANIFEST"
    fi
    first=0
    printf '  {{"offset": %d, "chunk_size": %d, "array_job_id": "%s"}}' "$OFFSET" "$SIZE" "$JOB_ID" >> "$MANIFEST"
done
echo "" >> "$MANIFEST"
echo "]" >> "$MANIFEST"
echo "Manifest written to $MANIFEST"
"""
            with open(submit_script_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(submit_script_text)
            try:
                os.chmod(submit_script_path, 0o755)
            except OSError:
                pass
            generated_paths.insert(0, submit_script_path)
            script_text = submit_script_text

        else:
            # ---------------------------------------------------------- #
            # Slurm array/single job, OR plain PBS array/single job, OR
            # NCI single (non-array) job (identical to plain PBS - Gadi
            # runs ordinary PBS Pro for non-farmed jobs too).
            # ---------------------------------------------------------- #
            if include_array:
                # '%N' slot-limit suffix (only appended when a positive
                # throttle was set) caps how many array tasks run AT ONCE -
                # e.g. Slurm's own '#SBATCH --array=1-12%4' for 12 tasks
                # sharing only 4 GPUs. See 'Max concurrently running array
                # tasks' above / this function's docstring.
                _throttle_suffix = f'%{array_throttle}' if array_throttle else ''
                slurm_array_line = f'#SBATCH -a {array_start}-{array_end}{_throttle_suffix}\n'
                pbs_array_line = f'#PBS -J {array_start}-{array_end}{_throttle_suffix}\n'
                slurm_out_pattern = os.path.join(logs_dir, f'{output_base}_%A_%a.output')
                slurm_err_pattern = os.path.join(logs_dir, f'{error_base}_%A_%a.error')
                pbs_out_pattern = os.path.join(logs_dir, f'{output_base}_^array_index^.output')
                pbs_err_pattern = os.path.join(logs_dir, f'{error_base}_^array_index^.error')
                index_note = f'each task ({array_start}..{array_end})'
            else:
                slurm_array_line = ''
                pbs_array_line = ''
                slurm_out_pattern = os.path.join(logs_dir, f'{output_base}.output')
                slurm_err_pattern = os.path.join(logs_dir, f'{error_base}.error')
                pbs_out_pattern = os.path.join(logs_dir, f'{output_base}.output')
                pbs_err_pattern = os.path.join(logs_dir, f'{error_base}.error')
                index_note = 'this single job'

            if scheduler == 'Slurm':
                script_text = f"""{shebang}
# EasiGP - {purpose} job
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --partition={partition}
{account_line_slurm}#SBATCH -o {slurm_out_pattern}
#SBATCH -e {slurm_err_pattern}
{slurm_array_line}{gpu_line_slurm}{qos_line_slurm}
# Submit with:  sbatch {job_name}.sh
# Runs {index_note} via {headless_script} - no GUI and no manual
# configuration is needed beyond this one-time export.

# Bugfix: `cd` explicitly into the EasiGP install directory (a literal
# absolute path baked in at generation time) rather than relying on Slurm's
# default of starting in $SLURM_SUBMIT_DIR, which is only correct if this
# script happens to be submitted from that exact directory. Getting cwd
# wrong here breaks more than just finding this script: genomic_
# prediction.py's own R-model sourcing and this run's `Result/` output
# location are also resolved relative to cwd.
cd "{project_dir}"
{extra_block}python3 "{headless_script_path}" --config "{config_path}"
"""
            else:
                # PBS and NCI-non-array: directive set/order matches a real,
                # working PBS Pro script on this cluster - separate `-l
                # ncpus=`/`-l mem=` lines (not `select=nodes:ncpus:mem`),
                # `-P` for the project code, optional `-l storage=`, and
                # `-r y/n` for rerunnable.
                gpu_reminder_pbs = (
                    "# GPU queue reminder: NCI Gadi requires the gpuvolta queue for GPU jobs,\n"
                    "# with ncpus a multiple of 12 per GPU requested - verify this matches your\n"
                    "# 'CPUs per task' setting above.\n"
                ) if (use_gpu and scheduler == 'NCI') else ''
                script_text = f"""{shebang}
# EasiGP - {purpose} job
#PBS -l ncpus={cpus_per_task}
#PBS -l mem={mem}
#PBS -N {job_name}
#PBS -l walltime={walltime}
#PBS -q {partition}
{project_line_pbs}{storage_line_pbs}#PBS -o {pbs_out_pattern}
#PBS -e {pbs_err_pattern}
{rerunnable_line_pbs}{pbs_array_line}{gpu_line_pbs}
# Submit with:  qsub {job_name}.sh
# Runs {index_note} via {headless_script} - no GUI and no manual
# configuration is needed beyond this one-time export. Note: PBS memory
# units are typically lowercase (e.g. 10gb); adjust 'Memory' above if needed.
{gpu_reminder_pbs}

# Bugfix: `cd` explicitly into the EasiGP install directory (a literal
# absolute path baked in at generation time), NOT `$PBS_O_WORKDIR` - that
# variable is set from wherever `qsub` was actually invoked, which need not
# be this directory. Getting cwd wrong here breaks more than just finding
# this script: genomic_prediction.py's own R-model sourcing and this run's
# `Result/` output location are also resolved relative to cwd.
cd "{project_dir}"
{extra_block}python3 "{headless_script_path}" --config "{config_path}"
"""

            script_filename = f'{job_name}.sh'
            script_path = os.path.join(project_dir, script_filename)
            with open(script_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(script_text)
            try:
                os.chmod(script_path, 0o755)
            except OSError:
                pass
            generated_paths.append(script_path)

        st.session_state[f'{kp}_config_path'] = config_path
        st.session_state[f'{kp}_script_path'] = generated_paths[0]
        st.session_state[f'{kp}_script_draft'] = script_text

        _file_list = '\n\n'.join(f'`{p}`' for p in generated_paths)
        st.success(f'Configuration saved to `{config_path}`.\n\nScript file(s) saved to:\n\n{_file_list}')
        st.info(
            f"Make sure `{headless_script}` and `pipeline_utils.py` are also in "
            f"`{project_dir}` (or adjust the paths in the script). Submitting the "
            "saved script runs everything non-interactively - the GUI is only needed "
            "for this one-time configuration step."
            + (" For a Gadi native job-array, run the `submit_arrays.sh` driver shown above - "
               "it calls `qsub` for you, once per chunk." if uses_gadi_array else "")
            + (f" Requesting {gpus_per_task} GPU(s) per task"
               + (f" (type: {gpu_type})" if gpu_type and scheduler == 'Slurm' else "")
               + (f" (QoS: {qos})" if qos and scheduler == 'Slurm' else "") + "."
               if use_gpu else "")
            + (f" Array tasks are capped at {array_throttle} running concurrently - raise or "
               "remove this limit above if you have more GPUs (or other shared resources) "
               "available than that."
               if include_array and array_throttle else "")
        )

    if st.session_state.get(f'{kp}_script_draft'):
        st.markdown(f"**Saved script** - `{st.session_state.get(f'{kp}_script_path', '')}`")
        st.caption(
            "Shown here for reference/copying. To change it, either edit the file on "
            "disk directly, or adjust the settings above and click the button again "
            "to regenerate and overwrite it."
        )
        st.text_area('Script contents', key=f'{kp}_script_draft', height=320,
                     help="The generated submission script shown for reference/editing before saving.")

    with st.expander("Convert an existing bash file's line endings (Windows -> Linux)"):
        # Requirement 12: 'Generate and save job files' (above) always
        # writes with newline='\n' now, so a FRESHLY generated script is
        # never affected by this in the first place - this utility exists
        # for anything generated (or manually edited) some other way: an
        # older script from before this fix, one edited afterward in a
        # Windows text editor, or any other bash file that needs to run
        # on a Linux HPC cluster. Windows-style CRLF line endings commonly
        # break a script there with an error like "bad interpreter:
        # /bin/bash^M: no such file or directory" - the '^M' is the
        # stray carriage return character contaminating the shebang line.
        #
        # Requirement 6: comes AFTER 'Generate and save job files' now
        # (not before it) and, when that button has been used this
        # session, offers to convert THAT exact just-generated script
        # directly - no need to separately re-upload the very file just
        # saved to disk a moment ago.
        _last_script_path = st.session_state.get(f'{kp}_script_path')
        if _last_script_path and os.path.isfile(_last_script_path):
            st.caption(f"Last generated script: `{_last_script_path}`")
            if st.button('Convert the script just generated', key=f'_btn_{kp}_sh_convert_last'):
                with open(_last_script_path, 'r', encoding='utf-8', newline='') as f:
                    _sh_text = f.read()
                _n_crlf = _sh_text.count('\r\n')
                _n_lone_cr = len(_sh_text.replace('\r\n', '\n').split('\r')) - 1
                _converted = _sh_text.replace('\r\n', '\n').replace('\r', '\n')
                if _n_crlf or _n_lone_cr:
                    with open(_last_script_path, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(_converted)
                    st.success(
                        f"Found and converted {_n_crlf + _n_lone_cr} Windows/old-Mac-style line "
                        f"ending(s) to Linux (LF), and saved the fix back to `{_last_script_path}`."
                    )
                else:
                    st.info(
                        "No Windows-style line endings found - this file already looks Linux-ready "
                        "(expected, since it was just generated by EasiGP itself)."
                    )
            st.divider()

        st.caption(
            "Or upload any other bash file to get back a copy with Linux-style (LF-only) line "
            "endings - useful for a script generated or edited on Windows some other way."
        )
        # Requirement 2 (bugfix): st.file_uploader's own session_state
        # value can never be programmatically restored (Streamlit raises
        # StreamlitValueAssignmentNotAllowedError if anything tries) - see
        # _NON_RESTORABLE_WIDGET_KEYS / load_gui_state()'s own comment
        # near the top of this file for the general fix. This uploader's
        # key is registered there; here we only show a small
        # informational hint (the last-used filename, persisted under a
        # plain, non-widget session_state key) instead of attempting to
        # restore the upload itself.
        _upload_key = f'_{kp}_sh_convert_upload'
        _last_filename_key = f'{_upload_key}_last_filename'
        _last_filename = st.session_state.get(_last_filename_key)
        if _last_filename:
            st.caption(f"Last used: `{_last_filename}` - please re-upload (uploaded files can't "
                       "be remembered across app launches).")
        _uploaded_sh = st.file_uploader(
            "Bash file to convert", type=None, key=_upload_key
        )
        if _uploaded_sh is not None:
            st.session_state[_last_filename_key] = _uploaded_sh.name
            _raw_bytes = _uploaded_sh.getvalue()
            try:
                _sh_text = _raw_bytes.decode('utf-8')
            except UnicodeDecodeError:
                _sh_text = _raw_bytes.decode('utf-8', errors='replace')
            _n_crlf = _sh_text.count('\r\n')
            _n_lone_cr = len(_sh_text.replace('\r\n', '\n').split('\r')) - 1
            _converted = _sh_text.replace('\r\n', '\n').replace('\r', '\n')
            if _n_crlf or _n_lone_cr:
                st.success(
                    f"Found and converted {_n_crlf + _n_lone_cr} Windows/old-Mac-style line "
                    f"ending(s) to Linux (LF)."
                )
            else:
                st.info("No Windows-style line endings found - this file already looks Linux-ready.")
            st.download_button(
                "Download Linux version", data=_converted.encode('utf-8'),
                file_name=_uploaded_sh.name, mime='text/plain', key=f'_{kp}_sh_convert_download',
            )


def validate_ratio_for_wopt(ratio, w_opt, hp_tune=None):
    """Split ratio(s) must be (train, validation, test) tuples whenever a
    validation split is actually needed - either because a weight-
    optimisation approach is selected in '4. Ensemble' (needs a
    validation split to optimise the weights), or because hyperparameter
    tuning is enabled for at least one model (needs a validation split to
    score tuning candidates) - and plain numbers (train ratio only)
    otherwise."""
    reasons = []
    if w_opt:
        reasons.append("a weight-optimisation method is selected in '4. Ensemble'")
    if hp_tune:
        reasons.append('hyperparameter tuning is enabled for at least one model')

    if reasons:
        for item in ratio:
            if not (isinstance(item, tuple) and len(item) == 3 and all(isinstance(x, (int, float)) for x in item)):
                reason_text = ' and '.join(reasons)
                reason_text = reason_text[0].upper() + reason_text[1:]
                raise ValueError(
                    f"{reason_text}, so Split ratio(s) must be a list of (train, validation, "
                    f"test) tuples, e.g. [(0.8, 0.1, 0.1)] - not plain numbers."
                )
    else:
        for item in ratio:
            if isinstance(item, tuple):
                raise ValueError(
                    "No weight-optimisation method is selected in '4. Ensemble' and no "
                    "model has hyperparameter tuning enabled, so Split ratio(s) must be a list "
                    "of plain numbers (train ratio only), e.g. [0.8] - not (train, validation, "
                    "test) tuples."
                )
            if not isinstance(item, (int, float)):
                raise ValueError(f'Split ratio entry {item!r} is not a number.')


def on_scenario_change():
    new_scenario = st.session_state['scenario']
    current_ratio = st.session_state.get('ratio_text', '').strip()
    if current_ratio in RATIO_DEFAULTS.values():
        st.session_state['ratio_text'] = RATIO_DEFAULTS[new_scenario]


# --------------------------------------------------------------------------- #
# Gathering the full config from session_state
# --------------------------------------------------------------------------- #

def build_ld_prune_config():
    """LD pruning (Tab 2) config dict, or None if disabled - the exact same
    dict shape Preprocess.LD_pruning.LD_pruning() / genomic_prediction.py's
    GP() already expect. Factored out of gather_config() (which still calls
    this for cfg['LD_PRUNE']) so the data-driven prior-network merge
    feature (build_data_driven_merge_config(), on the 'Biological Prior
    Network' tab) can reuse the SAME already-configured LD pruning settings
    for its own optional "filter genotypes at the beginning" step, instead
    of duplicating an entire second LD-pruning settings panel."""
    if not st.session_state.get('ld_prune_enabled', False):
        return None

    window_unit = st.session_state.get('ld_window_unit', 'kb')
    snp_info_path = st.session_state.get('ld_snp_info_path', '').strip()
    step = int(st.session_state.get('ld_step', 5))

    if window_unit in ('kb', 'cm') and not snp_info_path:
        raise ValueError(
            "LD Pruning: a SNP info file path is required when Window unit is "
            "'kb' or 'cm'."
        )
    if snp_info_path and not os.path.isfile(snp_info_path):
        raise ValueError(f"LD Pruning: SNP info file not found: {snp_info_path}")

    # Requirement (prevent this from happening silently): PLINK's
    # --indep-pairwise rejects a kb-based window whenever the step size
    # (variant count) isn't exactly 1 - passing e.g. window_unit='kb' with
    # step=5 makes PLINK itself error out partway through the run, deep
    # inside _run_plink()/_prune_partition() in Preprocess/LD_pruning.py,
    # with a raw PLINK error message rather than an obvious, EasiGP-level
    # explanation. Caught here (gather_config's own established validation
    # point, raised as ValueError - every existing caller of gather_config()
    # already catches ValueError and shows it via st.error(), the same
    # pattern as the checks just above) so the GUI stops the run up front
    # with a clear, actionable message instead of letting it fail silently/
    # cryptically partway through. Preprocess.LD_pruning.ld_prune_snps()
    # itself also validates this independently (see its own
    # _validate_inputs), since headless/HPC runs bypass this GUI entirely.
    if window_unit == 'kb' and step != 1:
        raise ValueError(
            f"LD Pruning: PLINK's --indep-pairwise requires Step size to be exactly 1 when "
            f"Window unit is 'kb' (got Step size={step}) - PLINK itself errors out on any "
            f"other value for a kb-based window. Either set Step size to 1, or switch Window "
            f"unit to 'variants' (where a step size other than 1 is meaningful and supported)."
        )

    chr_set_enabled = st.session_state.get('ld_chr_set_enabled', False)
    chr_set = int(st.session_state.get('ld_chr_set', 2)) if chr_set_enabled else None

    work_dir_enabled = st.session_state.get('ld_work_dir_enabled', False)
    work_dir = st.session_state.get('ld_work_dir', '').strip() if work_dir_enabled else ''
    if work_dir_enabled and not work_dir:
        raise ValueError("LD Pruning: please provide a working directory, or uncheck 'Use a custom working directory'.")

    ld_prune = {
        'snp_info': snp_info_path or None,
        'window': float(st.session_state.get('ld_window', 50.0)),
        'window_unit': window_unit,
        'step': step,
        'r2_threshold': float(st.session_state.get('ld_r2_threshold', 0.2)),
        'plink_path': st.session_state.get('ld_plink_path', 'plink2').strip() or 'plink2',
        'allow_extra_chr': bool(st.session_state.get('ld_allow_extra_chr', False)),
        'chr_set': chr_set,
        'work_dir': work_dir or None,
        'keep_intermediate': bool(st.session_state.get('ld_keep_intermediate', False)),
        'round_dosage': bool(st.session_state.get('ld_round_dosage', True)),
        'unmapped_strategy': st.session_state.get('ld_unmapped_strategy', 'variant_count'),
        'maf_threshold': float(st.session_state.get('ld_maf_threshold', 0.0)) if st.session_state.get('ld_maf_enabled', False) else 0.0,
    }
    if st.session_state.get('ld_maf_enabled', False) and not (0 < ld_prune['maf_threshold'] <= 0.5):
        raise ValueError(
            f"LD pruning: MAF threshold must be greater than 0 and at most 0.5 (minor allele "
            f"frequency can never exceed 0.5 by definition) - got {ld_prune['maf_threshold']}."
        )
    if ld_prune['unmapped_strategy'] == 'variant_count':
        ld_prune['unmapped_window'] = int(st.session_state.get('ld_unmapped_window', 50))
        ld_prune['unmapped_step'] = int(st.session_state.get('ld_unmapped_step', 5))

    # LD decay plot (Requirements 1-6 of the LD-filtering diagnostic
    # upgrade) - nested inside LD_PRUNE itself (rather than its own top-
    # level cfg key) since it's only ever meaningful alongside LD pruning
    # being enabled - see genomic_prediction.py's GP().
    ld_prune['decay_plot'] = build_ld_decay_plot_config()

    return ld_prune


def build_ld_decay_plot_config():
    """LD decay plot (Tab 2, nested under 'Apply LD pruning...') config
    dict for LD_PRUNE['decay_plot'] - {'enabled': False} if the feature's
    own checkbox is off, or None if LD pruning itself is disabled (exactly
    mirroring build_ld_prune_config()'s own None-when-disabled contract).
    Factored out into its own function (rather than inlined into
    build_ld_prune_config()) purely for readability - it isn't reused
    anywhere else.

    The decay curve is always plotted in the SAME distance unit as
    whatever 'Window unit' LD pruning itself is configured to use above
    (kb/cm/variants) - it isn't a separate, independently-chosen setting,
    since a decay plot's whole purpose is to help judge the window/r\u00b2
    threshold LD pruning is actually using, which only makes sense
    expressed in that same unit. This also means the 'kb'/'cm' SNP-info
    requirement is already guaranteed satisfied by build_ld_prune_config()
    itself (called before this function - see its own validation) by the
    time this runs, since LD pruning couldn't have been configured with
    window_unit='kb'/'cm' without a SNP info file in the first place - no
    separate check needed here."""
    if not st.session_state.get('ld_prune_enabled', False):
        return None
    if not st.session_state.get('ld_decay_plot_enabled', False):
        return {'enabled': False}

    window_unit = st.session_state.get('ld_window_unit', 'kb')

    frequency = int(st.session_state.get('ld_decay_plot_frequency', 1))
    if frequency < 1:
        raise ValueError(
            "LD decay plot: 'Generate a plot every ___ scenario(s)' must be at least 1."
        )

    max_key, bin_key = {
        'kb': ('ld_decay_max_distance_kb', 'ld_decay_bin_width_kb'),
        'cm': ('ld_decay_max_distance_cm', 'ld_decay_bin_width_cm'),
        'variants': ('ld_decay_max_distance_variants', 'ld_decay_bin_width_variants'),
    }[window_unit]
    max_distance = float(st.session_state.get(max_key, LD_DECAY_DEFAULT_MAX_DISTANCE[window_unit]))
    bin_width = float(st.session_state.get(bin_key, LD_DECAY_DEFAULT_BIN_WIDTH[window_unit]))
    if max_distance <= 0 or bin_width <= 0:
        raise ValueError(
            f"LD decay plot: 'Max distance shown' and 'Distance bin width' for the "
            f"{window_unit!r} unit must both be > 0."
        )

    return {
        'enabled': True,
        'frequency': frequency,
        'window_units': [window_unit],
        'max_pairs_per_chr': int(st.session_state.get('ld_decay_max_pairs_per_chr', 20000)),
        'keep_log': bool(st.session_state.get('ld_decay_keep_log', True)),
        window_unit: {'max_distance': max_distance, 'bin_width': bin_width},
    }


def build_rf_filter_config():
    """RF Marker Importance Filtering (Tab 2) config dict, or None if
    disabled - the same dict shape Preprocess.RF_marker_filtering.
    RF_marker_filtering() expects, now carrying either a percentage
    ('mode': 'percent', 'top_ratio') or a fixed marker count ('mode':
    'count', 'top_n') per requirement 1's new choice. Factored out of
    gather_config() for the same reason as build_ld_prune_config() above -
    the data-driven prior-network merge feature reuses this exact
    configuration (same forest hyperparameters, same percent/count choice)
    for its own RF-selection step, per requirement 2's "reuse the trained
    RF used in the RF filtering"."""
    if not st.session_state.get('rf_filter_enabled', False):
        return None

    rf_filter_mode = st.session_state.get('rf_filter_mode', 'Percentage of markers')
    max_features = st.session_state.get('rf_filter_max_features', 'sqrt')
    max_features = None if max_features == 'all' else max_features

    max_depth = (
        int(st.session_state.get('rf_filter_max_depth', 10))
        if st.session_state.get('rf_filter_max_depth_enabled', False) else None
    )

    rf_filter = {
        'n_estimators': int(st.session_state.get('rf_filter_n_estimators', 200)),
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': int(st.session_state.get('rf_filter_min_samples_leaf', 1)),
        'random_state': int(st.session_state.get('rf_filter_random_state', 0)),
    }

    if rf_filter_mode == 'Percentage of markers':
        top_percent = float(st.session_state.get('rf_filter_top_percent', 20.0))
        if not (0 < top_percent <= 100):
            raise ValueError(
                "RF Marker Importance Filtering: 'Keep the top ___ % of markers' must be "
                "between 0 and 100."
            )
        rf_filter['mode'] = 'percent'
        rf_filter['top_ratio'] = top_percent / 100.0  # GUI takes a percentage; the filter takes a ratio
    else:
        top_n = int(st.session_state.get('rf_filter_top_n', 100))
        if top_n < 1:
            raise ValueError(
                "RF Marker Importance Filtering: 'Keep the top ___ markers (M)' must be at "
                "least 1."
            )
        rf_filter['mode'] = 'count'
        rf_filter['top_n'] = top_n

    return rf_filter


def gather_config(mode, step):
    """Build the config dict from whatever widgets are currently on screen.

    `mode` is 'Sequential' or 'Parallel'. `step` is only meaningful when
    mode == 'Parallel' and is either 'Step 1' or 'Step 2'. Which pieces of
    config get gathered/validated depends on which tabs are actually
    visible for the current mode/step combination (see the tab-visibility
    flags near the bottom of the file).
    """
    is_step1 = (mode == 'Parallel' and step == 'Step 1')
    is_step2 = (mode == 'Parallel' and step == 'Step 2')

    cfg = {}

    cfg['R_PATH'] = None if st.session_state.get('r_path_none') else st.session_state.get('r_path', '').strip()

    # Patch 3, Requirement 3: read unconditionally (not gated on is_step1/
    # is_step2/mode) - resolve_compute_resources() consumes this exactly
    # like N_JOBS/TORCH_DEVICE, regardless of execution mode. Absent-key
    # default (an older config, or the checkbox never rendered this
    # session because Tab 2 was never opened) is False - byte-for-byte
    # today's behaviour (see Preprocess/RF_marker_filtering.py's own
    # _random_forest_regressor_cls(), which already treats a missing/False
    # 'use_gpu_sklearn' resource key as "scikit-learn CPU, as before").
    cfg['USE_GPU_SKLEARN'] = bool(st.session_state.get('use_gpu_sklearn', False))

    # ver4-4 (blueprint §4/§4a) - the sixteen new config keys this update
    # introduces, written HERE, unconditionally (same pattern as
    # USE_GPU_SKLEARN just above), regardless of which stage actually
    # adds the GUI widget/consuming engine code for each one. This is
    # deliberate: `gather_config()` only needs to be touched ONCE for
    # the whole update rather than once per later stage, and every
    # default below already reproduces `ver4-3` behaviour exactly
    # (I11) - a widget that doesn't exist yet simply means
    # `st.session_state.get(...)` returns that same default every time,
    # so nothing observable changes until the stage that adds the
    # widget (and, separately, the engine code that reads the key)
    # actually lands. Every widget lives inside `_render_advanced_
    # setting_components()` (R2's new expander, inside 'HPC job
    # resource requests') unless noted otherwise below - see blueprint
    # §4's 'GUI widget' column for the authoritative placement of each.
    #
    # R3.b/c (Stage 3) - torch/BLAS threading plumbing.
    cfg['R_BLAS_THREADS'] = st.session_state.get('r_blas_threads')
    cfg['R_BLAS_FOLLOWS_N_JOBS'] = bool(st.session_state.get('r_blas_follows_n_jobs', True))
    cfg['TORCH_NUM_THREADS'] = st.session_state.get('torch_num_threads')
    cfg['TORCH_DATALOADER_WORKERS'] = int(st.session_state.get('torch_dataloader_workers', 0) or 0)
    # R4.a (Stage 5) - GPU dispatch/AMP. TORCH_DEVICE's widget will be a
    # selectbox offering 'auto' (mapped to None here - resolve_compute_
    # resources() already treats a missing/None TORCH_DEVICE as "auto-
    # detect"), 'cpu', 'cuda', 'cuda:N'.
    _torch_device_raw = st.session_state.get('torch_device')
    cfg['TORCH_DEVICE'] = None if not _torch_device_raw or _torch_device_raw == 'auto' else _torch_device_raw
    cfg['CUDNN_BENCHMARK'] = bool(st.session_state.get('cudnn_benchmark', True))
    cfg['USE_AMP'] = bool(st.session_state.get('use_amp', True))
    cfg['GPU_EVAL_BATCH'] = int(st.session_state.get('gpu_eval_batch', 32) or 32)
    # R4.g (Stage 5) - GPU LD r^2 kernel. Widget: Tab 2 (Preprocessing),
    # alongside the other LD-pruning fields - not 'Advanced setting
    # components' (see blueprint §4's 'GUI widget' column).
    cfg['GPU_LD_R2'] = bool(st.session_state.get('gpu_ld_r2', True))
    # R4.h (Stage 5) - GPU kernel precompute for GBLUP/RKHS.
    cfg['GPU_KERNEL_PRECOMPUTE'] = bool(st.session_state.get('gpu_kernel_precompute', True))
    # R3.f (Stage 6) - parallel Grid/Random hyperparameter-search trials.
    # Widget: Tab 3 (Models & Hyperparameters), not 'Advanced setting
    # components'.
    cfg['HP_TUNE_PARALLEL_TRIALS'] = bool(st.session_state.get('hp_tune_parallel_trials', True))
    # ver4-5 R1 (blueprint §3.6) - four further hyperparameter-tuning
    # settings, same Tab 3 widget group as HP_TUNE_PARALLEL_TRIALS above.
    cfg['HP_TUNE_BAYES_BATCH'] = bool(st.session_state.get('hp_tune_bayes_batch', True))
    cfg['HP_TUNE_BAYES_BATCH_MAX'] = int(st.session_state.get('hp_tune_bayes_batch_max', 8))
    cfg['HP_TUNE_BAYES_LIAR'] = st.session_state.get('hp_tune_bayes_liar', 'max')
    cfg['HP_TUNE_PARALLEL_RESTARTS'] = bool(st.session_state.get('hp_tune_parallel_restarts', True))
    # Update ID ver4-6 (blueprint §4 config schema delta) - five further
    # hyperparameter-tuning settings (R1/R3/R4) and three weight-
    # optimisation settings (R5/R6/R7), same Tab 3 / Tab 4 widget groups
    # as the ver4-5 keys immediately above. Every default matches
    # pipeline_utils.resolve_compute_resources()'s own documented
    # default - see that function's docstring for the full rationale of
    # each; several are DELIBERATE, DISCLOSED non-ver4-5-equivalent
    # defaults (RK-3/RK-4 in the ver4-6 blueprint's risk register), not
    # oversights.
    cfg['HP_TUNE_BAYES_DOMAIN_REDUCTION'] = st.session_state.get('hp_tune_bayes_domain_reduction', 'auto')
    cfg['HP_TUNE_WARM_START'] = bool(st.session_state.get('hp_tune_warm_start', False))
    cfg['HP_TUNE_SELECTION_MARGIN'] = float(st.session_state.get('hp_tune_selection_margin', 0.02) or 0.0)
    cfg['HP_TUNE_VALID_REPEATS'] = int(st.session_state.get('hp_tune_valid_repeats', 1) or 1)
    cfg['HP_TUNE_SCOPE'] = st.session_state.get('hp_tune_scope', 'per_task')
    cfg['W_OPT_ANALYTIC_SEED'] = bool(st.session_state.get('w_opt_analytic_seed', False))
    cfg['W_OPT_VALIDATION_FLOOR'] = bool(st.session_state.get('w_opt_validation_floor', False))
    cfg['W_OPT_SIMPLEX_SEARCH'] = bool(st.session_state.get('w_opt_simplex_search', True))
    # Update ID ver4-5, R2 Stage 11 (decision D2, blueprint §4.6): whether
    # GP() should FAIL FAST (raise, before any task starts) if a selected
    # Tier 2 model's optional package turns out to be unavailable. Widget:
    # Tab 3 (Models & Hyperparameters), alongside the other global
    # tuning-fan-out settings. True (the default) matches the codebase's
    # own general "log negative states unconditionally, fail fast on
    # contract violations" principle (architecture doc §17); the GUI
    # itself already filters unavailable Tier 2 models out of the
    # selectable checkboxes (see TIER_2_MODELS above), so this key
    # mainly matters for a saved/hand-edited config that names one
    # anyway.
    cfg['MODEL_AVAILABILITY_STRICT'] = bool(st.session_state.get('model_availability_strict', True))
    # R3.g (Stage 6, highest risk) - Sequential intra-batch routing.
    # ver4-4 §4a's ONE deliberate "default off" exception among these
    # sixteen keys: this is a control-flow change with a genuine,
    # not-yet-verified unknown (the blueprint's own PC-2 pre-check), not
    # a numerics-only acceleration, so it stays False until that
    # pre-check is confirmed in a real environment - see the Stage 6
    # entry in the hand-off note for what's still outstanding.
    cfg['SEQUENTIAL_INTRA_BATCH'] = bool(st.session_state.get('sequential_intra_batch', False))
    # R6 (Stage 4) - QTL windowing in the scatter-plot matrix. Widget:
    # Tab 5 (Visualisation).
    cfg['QTL_WINDOW'] = float(st.session_state.get('qtl_window', 0.0) or 0.0)
    cfg['QTL_WINDOW_MODE'] = st.session_state.get('qtl_window_mode') or 'all_in_window'
    # R7.g/R7.h (Stage 3) - circos/plotting performance. Widgets: Tab 5/6
    # (Visualisation).
    cfg['PLOT_EFFECT_FLOAT32'] = bool(st.session_state.get('plot_effect_float32', True))
    # R7.h (Stage 3) - image quality (DPI) for the three raster plot
    # outputs, resolved as THREE INDEPENDENT keys - one per plot section
    # (Tab 5's violin/scatter widgets, Tab 6's circos widget) - rather
    # than the single shared 'PLOT_DPI' this app used before. A user
    # publishing a paper figure often wants the circos plot (the primary
    # deliverable - architecture doc S1) at print resolution while
    # leaving the quick-look violin/scatter diagnostics at a smaller,
    # faster-to-save default, so one shared knob forced an unwanted
    # trade-off between "everything crisp and slow" and "everything fast
    # and coarse". Each defaults to 300, matching this app's own previous
    # single-field default, so a config that never touches these widgets
    # renders identically to before.
    cfg['METRIC_PLOT_DPI'] = int(st.session_state.get('metric_plot_dpi', 300) or 300)
    cfg['SCATTER_PLOT_DPI'] = int(st.session_state.get('scatter_plot_dpi', 300) or 300)
    cfg['CIRCOS_PLOT_DPI'] = int(st.session_state.get('circos_plot_dpi', 300) or 300)

    raw_pheno = st.session_state.get('phenotype_targets', '').strip()
    if raw_pheno.lower() == 'all':
        cfg['PHENOTYPE'] = 'all'
    else:
        cfg['PHENOTYPE'] = [p.strip() for p in raw_pheno.split(',') if p.strip()]
        if not cfg['PHENOTYPE']:
            raise ValueError('Please provide at least one target phenotype (or "all").')

    cfg['SCENARIO'] = st.session_state.get('scenario', 'within')

    cfg['RESULT_NAME'] = st.session_state.get('result_name', '').strip()
    if not cfg['RESULT_NAME']:
        raise ValueError('Please provide a result folder name.')

    # Update ID ver4-9, R7: set UNCONDITIONALLY (never inside an
    # is_step1/is_step2 guard) - unlike METRIC_SUMMARY_CREATE/
    # DPT_SUMMARY_CREATE/WEIGHT_PLOT_* (Tab-5-only, plotting-stage
    # settings), RESULT_COMPRESSION is consumed by GP() itself (Step 1 AND
    # Sequential mode both write result files) as well as by assemble()
    # (Step 2) - every execution mode this GUI can configure needs this
    # key present. Default 'gzip' matches GP()'s own default.
    cfg['RESULT_COMPRESSION'] = st.session_state.get('result_compression', 'gzip')

    # Everything below this point (model selection, hyperparameters, ratio,
    # genotype/phenotype files) is only gathered when Tab 2 ("Models &
    # Hyperparameters") is actually shown, i.e. Sequential mode or
    # Parallel / Step 1. In Parallel / Step 2 these values come from the
    # assemble() call instead, so we skip them entirely here.
    if not is_step2:
        selected_models = [m for m in AVAILABLE_MODELS if st.session_state.get(f'model_selected_{m}', False)]
        if not selected_models:
            raise ValueError('Please select at least one model to run.')

        # Requirement: N independent networks for the same phenotype(s)
        # ('Several independent networks...' on the Biological Prior
        # Network tab) become N independent models -
        # GAT_biological_prior_knowledge_1, _2, ... - each with its own
        # HPARAMETERS entry, replacing the single
        # 'GAT_biological_prior_knowledge' entry in MODEL. This is what
        # gives each one its own, distinguishable 'model' column value in
        # every output file - see genomic_prediction.py's
        # _is_bio_prior_model()/dispatch, which already treat any
        # 'GAT_biological_prior_knowledge*' name as an instance of this
        # model. The N networks can either be shared across every phenotype
        # (_bio_repeat_paths: {k: (json, gene_csv)}), or - if the "separate
        # set of N networks for each phenotype" checkbox is on -
        # independent per phenotype too (_bio_repeat_paths_pp:
        # {phenotype: {k: (json, gene_csv)}}), in which case each
        # GAT_biological_prior_knowledge_<k>'s own HPARAMETERS entry becomes
        # a per-phenotype dict itself, below.
        _bio_repeat_active = (
            'GAT_biological_prior_knowledge' in selected_models
            and st.session_state.get('bio_network_mode', '').startswith('Several independent networks')
        )
        _bio_repeat_per_phenotype = _bio_repeat_active and st.session_state.get('bio_repeat_per_phenotype', False)
        if _bio_repeat_active:
            if _bio_repeat_per_phenotype:
                _bio_repeat_paths_pp = st.session_state.get('bio_repeat_paths_per_phenotype')
                if not _bio_repeat_paths_pp:
                    raise ValueError(
                        "Multiple independent per-phenotype biological-prior networks are enabled "
                        "on the 'Biological Prior Network' tab, but you haven't successfully "
                        "clicked 'Preview + use these networks for the model' yet. Do that first."
                    )
                if cfg['PHENOTYPE'] != 'all':
                    _missing_ph = [p for p in cfg['PHENOTYPE'] if p not in _bio_repeat_paths_pp]
                    if _missing_ph:
                        raise ValueError(
                            f"No independent networks configured for phenotype(s) {_missing_ph} "
                            f"on the 'Biological Prior Network' tab."
                        )
                _all_k = sorted({k for by_k in _bio_repeat_paths_pp.values() for k in by_k})
                _bio_instance_names = [f'GAT_biological_prior_knowledge_{k}' for k in _all_k]
            else:
                _bio_repeat_paths = st.session_state.get('bio_repeat_paths')
                if not _bio_repeat_paths:
                    raise ValueError(
                        "Multiple independent biological-prior networks are enabled on the "
                        "'Biological Prior Network' tab, but you haven't successfully clicked "
                        "'Preview + use these networks for the model' yet. Do that first, or switch "
                        "back to 'One network (shared across all phenotypes)' on that tab."
                    )
                _bio_instance_names = [f'GAT_biological_prior_knowledge_{k}' for k in sorted(_bio_repeat_paths)]
            _insert_at = selected_models.index('GAT_biological_prior_knowledge')
            selected_models = (
                selected_models[:_insert_at] + _bio_instance_names + selected_models[_insert_at + 1:]
            )
        cfg['MODEL'] = selected_models

        raw_ratio_text = st.session_state.get('ratio_text', '').strip()
        try:
            # Users no longer need to type the enclosing square brackets
            # themselves (e.g. '0.8, 0.65' or '(0.8,0.1,0.1)' both work) -
            # only add them if not already present, so typing the brackets
            # explicitly (the old style) still works unchanged.
            wrapped_ratio_text = raw_ratio_text if raw_ratio_text.startswith('[') else f'[{raw_ratio_text}]'
            ratio = ast.literal_eval(wrapped_ratio_text)
        except Exception as exc:
            raise ValueError(f'Could not parse split ratio(s): {exc}')
        if not isinstance(ratio, list):
            raise ValueError('Split ratio(s) must be one or more numbers or tuples, e.g. 0.8 or 0.8, 0.65')
        if not ratio:
            raise ValueError('Please provide at least one split ratio.')
        cfg['RATIO'] = ratio

        cfg['ITER_NUM'] = int(st.session_state.get('iter_num', 1))
        cfg['MIN_DATA_POINTS'] = int(st.session_state.get('min_data_points', 100))
        if cfg['MIN_DATA_POINTS'] < 1:
            raise ValueError("'Minimum data points required per iteration' must be at least 1.")

        genotype_format = st.session_state.get('genotype_format', 'CSV file')
        if genotype_format == 'CSV file':
            cfg['GENOTYPE_FORMAT'] = 'csv'
            cfg['GENOTYPE_FILE_NAME'] = st.session_state.get('genotype_path', '').strip()
            cfg['GENOTYPE_PLINK_PATH'] = 'plink2'
            if not os.path.isfile(cfg['GENOTYPE_FILE_NAME']):
                raise ValueError(f"Genotype file not found: {cfg['GENOTYPE_FILE_NAME']}")
        else:
            cfg['GENOTYPE_FORMAT'] = 'plink'
            cfg['GENOTYPE_FILE_NAME'] = st.session_state.get('genotype_plink_stem', '').strip()
            cfg['GENOTYPE_PLINK_PATH'] = st.session_state.get('genotype_plink_path', 'plink2').strip() or 'plink2'
            if not cfg['GENOTYPE_FILE_NAME']:
                raise ValueError("Please provide a PLINK file stem (Data & Setup tab).")
            _missing_plink = [ext for ext in ('bed', 'bim', 'fam')
                               if not os.path.isfile(f"{cfg['GENOTYPE_FILE_NAME']}.{ext}")]
            if _missing_plink:
                raise ValueError(
                    f"Incomplete PLINK fileset for stem '{cfg['GENOTYPE_FILE_NAME']}': missing "
                    f".{', .'.join(_missing_plink)}. All three of <stem>.bed, <stem>.bim, "
                    f"<stem>.fam must exist, sharing the exact same stem."
                )
        cfg['PHENOTYPE_FILE_NAME'] = st.session_state.get('phenotype_path', '').strip()
        if not os.path.isfile(cfg['PHENOTYPE_FILE_NAME']):
            raise ValueError(f"Phenotype file not found: {cfg['PHENOTYPE_FILE_NAME']}")

        hparameters = {}
        for model in selected_models:
            if model == 'ensemble':
                continue
            if model == 'GAT_biological_prior_knowledge' and st.session_state.get('bio_per_phenotype_mode', False):
                per_pheno_paths = st.session_state.get('bio_per_phenotype_paths')
                if not per_pheno_paths:
                    raise ValueError(
                        "Per-phenotype biological-prior networks are enabled on the 'Biological "
                        "Prior Network' tab, but you haven't successfully clicked 'Preview + use "
                        "per-phenotype inputs for the model' yet. Do that first, or turn the "
                        "toggle off on that tab to share a single network across every phenotype."
                    )
                if cfg['PHENOTYPE'] != 'all':
                    missing = [p for p in cfg['PHENOTYPE'] if p not in per_pheno_paths]
                    if missing:
                        raise ValueError(
                            f"No biological-prior network configured for phenotype(s) {missing} "
                            f"on the 'Biological Prior Network' tab."
                        )
                # Built fresh here (not at preview-click time) so shared
                # hyperparameters (marker_info_path, epoch, etc.) reflect
                # whatever is currently set on the Models tab, even if that
                # was filled in after the per-phenotype preview was run.
                hparameters[model] = {
                    ph: build_bio_prior_params_for(json_path, gene_csv)
                    for ph, (json_path, gene_csv) in per_pheno_paths.items()
                }
            elif _bio_repeat_active and model in _bio_instance_names:
                # model is 'GAT_biological_prior_knowledge_<k>' - look up
                # that instance's own (network JSON, gene-location CSV) from
                # bio_repeat_paths, built fresh (same reasoning as
                # per-phenotype mode above) from whichever k this name maps
                # to. Every instance shares the SAME hyperparameter panel
                # (there's only one 'GAT_biological_prior_knowledge' entry
                # on the Models tab - neuron/dropout/epoch/etc. are the same
                # across all N instances by design; only the network
                # differs), same as build_bio_prior_params_for already does
                # for per-phenotype mode.
                _k = int(model.rsplit('_', 1)[1])
                if _bio_repeat_per_phenotype:
                    # model (GAT_biological_prior_knowledge_<k>)'s own
                    # HPARAMETERS entry is itself a per-phenotype dict here -
                    # phenotype k's own network for THIS k, for every
                    # phenotype that has one - exactly mirroring how plain
                    # per-phenotype mode (above) builds its single instance's
                    # dict, just repeated once per k.
                    hparameters[model] = {
                        ph: build_bio_prior_params_for(*by_k[_k])
                        for ph, by_k in _bio_repeat_paths_pp.items() if _k in by_k
                    }
                else:
                    _json_path, _gene_csv = _bio_repeat_paths[_k]
                    hparameters[model] = build_bio_prior_params_for(_json_path, _gene_csv)
            else:
                hparameters[model] = resolve_hparams(model)
        cfg['HPARAMETERS'] = hparameters

        # Requirement: hyperparameter tuning only ever applies to models the
        # user explicitly turned it on for (Tab 3, right below each model's
        # own hyperparameter panel) - never on by default. GAT_biological_
        # prior_knowledge (and any numbered instance of it) never appears
        # here even if selected, since render_hp_tune_panel()/
        # hp_tune_supported() never offer it a panel to enable in the first
        # place.
        hp_tune = {}
        for model in selected_models:
            if model == 'ensemble':
                continue
            model_hp_tune = resolve_hp_tune(model)
            if model_hp_tune is not None:
                hp_tune[model] = model_hp_tune
        cfg['HP_TUNE'] = hp_tune if hp_tune else None

        # LD pruning (Tab 2) - optional pre-processing step passed into GP().
        cfg['LD_PRUNE'] = build_ld_prune_config()

        # RF marker importance filtering (Tab 2) - optional pre-processing
        # step passed into GP(). Runs after LD pruning above when both are
        # enabled (see genomic_prediction.py's per-task ordering).
        cfg['RF_FILTER'] = build_rf_filter_config()

        cfg['OTHER_MODELS_MARKER_SOURCE'] = st.session_state.get('other_models_marker_source', 'full_or_filtered')

    # PARALLEL batch configuration - only relevant/gathered for Step 1.
    # batch_id is only ever something the *user* selects when the source is
    # 'Manual integer'. For Slurm/PBS it is intentionally left as None here -
    # it gets assigned automatically later (by run_step1_batch.py, reading
    # SLURM_ARRAY_TASK_ID / PBS_ARRAY_INDEX from the scheduler at run time),
    # not resolved from whatever environment happens to be running this GUI.
    if is_step1:
        source = st.session_state.get('batch_id_source', BATCH_ID_SOURCES[0])
        if source == 'Manual integer':
            batch_id = int(st.session_state.get('batch_id_manual', 0))
        else:  # Slurm or PBS - resolved automatically at actual run time
            batch_id = None

        batch_size = int(st.session_state.get('batch_size', 3))
        cfg['PARALLEL'] = {'batch_id': batch_id, 'batch_size': batch_size}

    # Patch 3, Requirement 1: which of the three explicit Step 2
    # gather-results options was chosen (see assemble.load_for_step2()'s
    # own docstring for the full decision table) - only meaningful for
    # Step 2. cfg['SKIP_ASSEMBLE']/['SKIP_ASSEMBLE_BATCH_FALLBACK'] are
    # deliberately NOT written by this (current) GUI anymore - ASSEMBLE_MODE
    # is now authoritative for any config this GUI produces; those two
    # legacy keys remain meaningful only as load_for_step2()'s own fallback
    # for a config JSON written by a GUI version that predates this option.
    if is_step2:
        cfg['ASSEMBLE_MODE'] = st.session_state.get('step2_assemble_mode', 'assemble')
        # Requirement: optional total batch count, so assemble() can report
        # a batch missing at the very end (beyond the highest batch number
        # actually found on disk), not just gaps between ones that ran.
        # 0 (the default) means "unknown" - assemble() then only reports
        # gaps within the range it actually found.
        _expected_batches = int(st.session_state.get('step2_expected_batches', 0))
        cfg['EXPECTED_BATCHES'] = _expected_batches if _expected_batches > 0 else None

    # Ensemble weighting (Tab 3) is shown in every mode/step.
    selected_wopt = [m for m in W_OPT_METHODS if st.session_state.get(f'wopt_selected_{m}', False)]
    cfg['W_OPT'] = selected_wopt if selected_wopt else None

    hparameters_opt = {}
    for method in selected_wopt:
        hparameters_opt[method] = resolve_wopt(method)
    cfg['HYPERPARAMETERS_OPT'] = hparameters_opt

    # Requirement: when more than one hyperparameter-tuning algorithm was
    # used for at least one model, the user chooses whether ensembling (
    # naive average and/or any weighted method above) combines models per
    # tuning method or across all of them - see the radio button on this
    # same tab, only shown when it's actually ambiguous. Always set with a
    # safe default even when the radio was never shown (nothing multi-
    # algorithm-tuned, or Step 2, where cfg.get('HP_TUNE') is never set) -
    # models.hyperparameter_tuning.ensemble_groups() collapses to identical
    # behaviour regardless of this value whenever there's nothing to split.
    cfg['HP_TUNE_ENSEMBLE_MODE'] = st.session_state.get('hp_tune_ensemble_mode', 'per_method')

    if 'RATIO' in cfg:
        validate_ratio_for_wopt(cfg['RATIO'], cfg['W_OPT'], cfg.get('HP_TUNE'))

    # Scatter plot (Tab 5) and Circos plot (Tab 6) config is only gathered
    # when those tabs are shown, i.e. Sequential mode or Parallel / Step 2.
    if not is_step1:
        cfg['METRIC_PLOT_CREATE'] = bool(st.session_state.get('metric_plot_create', True))
        # Update ID 3, R3: absent key -> True is the one intentional
        # behaviour change for old configs (I11) - the artefact is purely
        # additive (a new workbook) and overwrites nothing that exists
        # today, so an older step2_config.json/sequential_config.json
        # simply gains it rather than erroring (blueprint §R3.11).
        cfg['METRIC_SUMMARY_CREATE'] = bool(st.session_state.get('metric_summary_create', True))
        # Update ID ver4-9, R5: absent key -> True is the SAME intentional,
        # purely-additive default METRIC_SUMMARY_CREATE itself set as
        # precedent (design blueprint SS2.2.7/I11) - this artefact is new
        # and overwrites nothing that exists today.
        cfg['DPT_SUMMARY_CREATE'] = bool(st.session_state.get('dpt_summary_create', True))
        # Update ID ver4-9, R6: unlike METRIC_SUMMARY_CREATE/DPT_SUMMARY_CREATE
        # above, WEIGHT_PLOT_CREATE's own documented default is False
        # EVERYWHERE (design blueprint SS2.3.7/I11) - the GUI checkbox
        # below also defaults unchecked, so an existing config that never
        # mentions this key and a brand-new run left at its own defaults
        # behave identically (no new PNG files) until a person actively
        # opts in, rather than a new plot type appearing unannounced.
        cfg['WEIGHT_PLOT_CREATE'] = bool(st.session_state.get('weight_plot_create', False))
        cfg['WEIGHT_PLOT_CONFIG'] = {
            'font_size': int(st.session_state.get('weight_plot_font', 1)),
            'fig_size': int(st.session_state.get('weight_plot_fig', 5)),
        }
        cfg['WEIGHT_PLOT_DPI'] = int(st.session_state.get('weight_plot_dpi', 300) or 300)
        cfg['SCATTER_CREATE'] = bool(st.session_state.get('scatter_create', True))
        cfg['CIRCOS_CREATE'] = bool(st.session_state.get('circos_create', True))
        qtl = st.session_state.get('qtl_path', '').strip()
        cfg['QTL'] = qtl if qtl else None
        cfg['SCATTER_CONFIG'] = {
            'font_size': int(st.session_state.get('scatter_font', 2)),
            'fig_size': int(st.session_state.get('scatter_fig', 30)),
        }
        cfg['METRIC_PLOT_CONFIG'] = {
            'font_size': int(st.session_state.get('metric_font', 1)),
            'fig_size': int(st.session_state.get('metric_fig', 5)),
        }

        cfg['CHROMOSOME_INFO'] = st.session_state.get('chrom_info_path', '').strip()
        cfg['MARKER_INFO'] = st.session_state.get('marker_info_path', '').strip()
        gene_info = st.session_state.get('gene_info_path', '').strip()
        cfg['GENE_INFO'] = gene_info if gene_info else None
        # Requirement: these files are only ever actually used by
        # circos_plot() (see its call sites, all already gated on
        # `if cfg['CIRCOS_CREATE']:`) - so only REQUIRE them when circos
        # plots are actually going to be generated. Without this guard, a
        # user who unchecked 'Create circos plot' (and so reasonably left
        # these fields blank) would still be blocked from running anything
        # at all by a validation error for a plot they explicitly opted
        # out of. The keys themselves are still always populated above
        # (even if blank/invalid) so nothing downstream that merely reads
        # cfg['MARKER_INFO'] etc. (without using it) needs special-casing.
        if cfg['CIRCOS_CREATE']:
            if not os.path.isfile(cfg['CHROMOSOME_INFO']):
                raise ValueError(
                    f"Chromosome info file not found: {cfg['CHROMOSOME_INFO']!r}. This is "
                    f"required because 'Create circos plot' is checked - either provide a "
                    f"valid file, or uncheck 'Create circos plot' if you don't need it."
                )
            if not os.path.isfile(cfg['MARKER_INFO']):
                raise ValueError(
                    f"Marker info file not found: {cfg['MARKER_INFO']!r}. This is required "
                    f"because 'Create circos plot' is checked - either provide a valid file, "
                    f"or uncheck 'Create circos plot' if you don't need it."
                )

        cfg['CIRCOS_CONFIG'] = {
            'space': _num_or_default('circos_space', 1),
            'start': _num_or_default('circos_start', 15),
            'end': _num_or_default('circos_end', 345),
            'link_alpha_min': _num_or_default('circos_link_alpha_min', 0.15),
            'interaction_top': _num_or_default('circos_topinteraction', 0.01),
            # Requirement 4: 'percentage' (default, backward-compatible)
            # or 'count' - see circos_plot._select_top_interactions()'s
            # own docstring for exactly how each mode is applied.
            'interaction_top_mode': st.session_state.get('circos_topinteraction_mode') or 'percentage',
            'interaction_top_count': int(_num_or_default('circos_topinteraction_count', 75)),
            'label_size': _num_or_default('circos_labelsize', 6),
            'scale': _num_or_default('circos_scale', 100),
            'unit': st.session_state.get('circos_unit', 'bp'),
            # Update ID ver4-6, R2 (blueprint §5.1): unlike every field
            # above, these six write the NEW, improved values by default
            # for a config THIS GUI builds - circos_plot.py's own
            # `circos_config.get(key, <legacy default>)` reads are what
            # keep an OLDER, already-written config JSON (missing these
            # keys entirely) rendering exactly as it always has (I11).
            'ring_label_size': _num_or_default('circos_ring_label_size', 8.0),
            'ring_label_fit': st.session_state.get('circos_ring_label_fit') or 'shrink_then_truncate',
            'ring_layout': st.session_state.get('circos_ring_layout') or 'fit',
            'figsize': st.session_state.get('circos_figsize'),  # None -> pycirclize's own 8.0 default
            'seam_gap_max': _num_or_default('circos_seam_gap_max', 40.0),
            'ring_label_max_chars': int(_num_or_default('circos_ring_label_max_chars', 0)),
        }
        if not (0.0 <= cfg['CIRCOS_CONFIG']['link_alpha_min'] <= 1.0):
            raise ValueError(
                f"Circos plot: 'Minimum link opacity' must be between 0 and 1 "
                f"(got {cfg['CIRCOS_CONFIG']['link_alpha_min']})."
            )
        cfg['END_ADJUST'] = _num_or_default('end_adjust', 0)
        cfg['GENE_ADJUST'] = _num_or_default('gene_adjust', 0)
        cfg['WINDOW'] = _num_or_default('window_size', 300)
        cfg['CIRCOS_BROADCAST_POPULATION'] = bool(st.session_state.get('circos_broadcast_population', True))

        ascending_raw = st.session_state.get('ascending', 'None')
        cfg['ASCENDING'] = {'True': True, 'False': False, 'None': None}[ascending_raw]

        cfg['CYTOBAND_COLORMAP'] = dict(st.session_state.get('cytoband_colormap', DEFAULT_CYTOBAND_COLORMAP))

    return cfg


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #

st.title('EasiGP')

# --------------------------------------------------------------------------- #
# Execution mode selection - the very first thing the user picks.
# 'Sequential' reproduces the original, single-run behaviour of this app.
# 'Parallel' splits the work into two separately-run steps so that Step 1
# (the model-fitting stage) can be fanned out across many HPC batch jobs,
# and Step 2 (assembling + plotting) is run once afterwards on the
# combined results.
# --------------------------------------------------------------------------- #
st.subheader('Execution mode')
mode = st.radio(
    'How do you want to run this pipeline?',
    options=['Sequential', 'Parallel'],
    key='exec_mode',
    horizontal=True,
    help=(
        "Sequential: run data setup, model fitting, ensemble weighting, "
        "scatter plots and circos plots in a single pass. "
        "Parallel: split the run into Step 1 (fit models, e.g. as many "
        "HPC batch jobs) and Step 2 (assemble all batches and produce "
        "the plots)."
    ),
)

step = None
if mode == 'Parallel':
    step = st.radio(
        'Which step are you running?',
        options=['Step 1', 'Step 2'],
        key='parallel_step',
        horizontal=True,
        help=(
            "Step 1: fit the selected models for one batch of prediction "
            "scenarios (run this once per batch, e.g. once per array-job "
            "index on your HPC scheduler). "
            "Step 2: assemble the results from all batches and generate "
            "the metric, scatter and circos plots."
        ),
    )
    if step == 'Step 2':
        # Patch 3, Requirement 1: restores the three explicit,
        # independently-selectable options a single "Skip assemble"
        # checkbox used to conflate into one binary choice (checked ->
        # "use pre-assembled files, or fall back to per-batch files if
        # they don't exist yet" - two genuinely different behaviours
        # under one label). See assemble.load_for_step2()'s own
        # docstring for exactly what each option below does.
        st.radio(
            'How should results be gathered for plotting?',
            options=['Assemble into combined files', 'Use existing pre-assembled files',
                     'Do not assemble (plot directly from batch files)'],
            key='step2_assemble_mode_label',
            help=(
                "**Assemble into combined files** (the normal choice): merge every batch's "
                "own output into one combined Metric.csv/Prediction_result_*.csv/etc., then "
                "plot from those. Overwrites any combined files from a previous assemble.\n\n"
                "**Use existing pre-assembled files**: skip merging entirely and plot directly "
                "from the combined files a previous 'Assemble into combined files' run already "
                "wrote - fails with a clear error if they don't exist yet. Useful when only the "
                "scatter/circos plotting step needs retrying (e.g. after fixing a QTL file or "
                "circos config issue) and you don't want to pay the merge cost again.\n\n"
                "**Do not assemble**: plot directly from each batch's own per-batch files, "
                "without ever writing (or even looking at) the combined files - even if a "
                "previous 'Assemble' run already produced them. Never writes anything. Useful "
                "for a quick look at partial/updated results, or on a very large run where "
                "writing the combined files themselves isn't needed this time."
            ),
        )
        _assemble_mode_by_label = {
            'Assemble into combined files': 'assemble',
            'Use existing pre-assembled files': 'use_preassembled',
            'Do not assemble (plot directly from batch files)': 'no_assemble',
        }
        st.session_state['step2_assemble_mode'] = _assemble_mode_by_label[
            st.session_state.get('step2_assemble_mode_label', 'Assemble into combined files')
        ]
        if st.session_state['step2_assemble_mode'] == 'assemble':
            st.number_input(
                'Expected total number of batches (optional - 0 = unknown)',
                min_value=0, value=0, step=1, key='step2_expected_batches',
                help=(
                    "If you know how many batches you submitted (e.g. 'Array end index' + 1 "
                    "from Step 1's HPC export), enter it here so Step 2 can report ANY batch "
                    "that never produced output - including ones at the very end - not just "
                    "gaps between batches that did. Leave at 0 to only detect gaps between the "
                    "lowest and highest batch number actually found on disk."
                ),
            )

        # Requirement: let a person see which batches are complete,
        # incomplete, or missing WITHOUT waiting for the actual assemble
        # step - which, for many/large batches, means reading and
        # concatenating every batch's Metric_<idx>.csv/Prediction_result_*
        # etc. check_batch_status() is the fast, read-only half of that
        # same logic (file-existence + small checkpoint-file reads only),
        # so this is safe to click freely, any time, without committing to
        # a full run.
        if st.button('Check batch status (fast - no data assembly)', key='step2_check_status_btn'):
            _status_result_name = st.session_state.get('result_name', '').strip()
            if not _status_result_name:
                st.error("Enter a 'Result folder name' further down this page first.")
            else:
                _status_expected = int(st.session_state.get('step2_expected_batches', 0)) or None
                _status = check_batch_status(_status_result_name, expected_batches=_status_expected)
                n_complete = len(_status['complete'])
                n_incomplete = len(_status['incomplete'])
                n_missing = len(_status['missing'])

                if n_complete == 0 and n_incomplete == 0 and n_missing == 0:
                    st.warning(
                        f"No batch output found yet for '{_status_result_name}' - make sure "
                        f"Step 1 has been run (and at least one batch has produced output)."
                    )
                elif n_incomplete == 0 and n_missing == 0:
                    st.success(f"All {n_complete} batch(es) found are complete and ready to assemble.")
                else:
                    st.info(
                        f"{n_complete} complete, {n_incomplete} incomplete, {n_missing} missing "
                        f"(out of {n_complete + n_incomplete + n_missing} batch ID(s) accounted for)."
                    )

                if _status['complete']:
                    st.write(f"**Complete ({n_complete}):** " + format_batch_id_list(_status['complete']))
                if _status['incomplete']:
                    st.write(f"**Incomplete ({n_incomplete})** - started but hit an error partway "
                             f"through; re-submit these to resume automatically:")
                    st.code('\n'.join(describe_incomplete_batch(b) for b in _status['incomplete']), language='text')
                    # Requirement 1: same compact, comma-only, copy-paste-
                    # ready ID list format as 'Complete'/'Missing' already
                    # have below - the detailed per-batch description
                    # above is useful on its own, but isn't in a form
                    # that pastes directly into a scheduler directive
                    # (e.g. PBS's `-J`) the way format_batch_id_list()'s
                    # output is.
                    st.write(f"**Incomplete ({n_incomplete}) - batch ID list:**")
                    st.code(format_batch_id_list([b['batch_id'] for b in _status['incomplete']]), language='text')
                if _status['missing']:
                    st.write(f"**Missing ({n_missing})** - never produced any output at all; "
                             f"re-run Step 1 for these from scratch:")
                    st.code(format_batch_id_list(_status['missing']), language='text')

scheduler_name, detected_batch_id = detect_array_job_env()
if scheduler_name is not None:
    st.warning(
        f"This process appears to be running inside a {scheduler_name} job array "
        f"(array index `{detected_batch_id}`). **Don't launch the Streamlit GUI itself "
        "as the array-job command** - that would require configuring it once per task, "
        "which is unworkable for large arrays. Instead: configure Step 1 here "
        "*once*, interactively, click **'Save configuration for HPC array job'**, and "
        "point your array job at `run_step1_batch.py` (a plain, non-interactive script) "
        "instead. See the 'Run pipeline' section below for the exact commands."
    )

st.divider()

# Which tabs are relevant for the chosen mode/step:
#   Sequential            -> all 6 tabs
#   Parallel / Step 1     -> Data & Setup, Data preprocessing,
#                            Models & Hyperparameters, Ensemble
#   Parallel / Step 2     -> Data & Setup, Ensemble, Scatter Plot, Circos Plot
show_tab_models = (mode == 'Sequential') or (mode == 'Parallel' and step == 'Step 1')
show_tab_plots = (mode == 'Sequential') or (mode == 'Parallel' and step == 'Step 2')

tab_labels = ['1. Data & Setup']
tab_keys = ['setup']
if show_tab_models:
    tab_labels.append('2. Data Preprocessing')
    tab_keys.append('data_preprocessing')
    tab_labels.append('3. Models & Hyperparameters')
    tab_keys.append('models')
tab_labels.append('4. Ensemble')
tab_keys.append('ensemble')
if show_tab_plots:
    tab_labels.append('5. Violin, Bar & Scatter Plots')
    tab_keys.append('scatter')
    tab_labels.append('6. Circos Plot')
    tab_keys.append('circos')

# Requirements.md item 6: a dedicated 'Run pipeline' tab, holding exactly
# what used to be the standalone 'Run pipeline' section further down this
# script (now moved inside `with tab_map['run_pipeline']:` - see that
# block's own comment). Always present, regardless of mode/step (unlike
# the conditional tabs above), and always LAST in the tab list.
tab_labels.append('7. Run pipeline')
tab_keys.append('run_pipeline')

_tabs = st.tabs(tab_labels)
tab_map = dict(zip(tab_keys, _tabs))

# ----------------------------- Tab 1: Setup ----------------------------- #
with tab_map['setup']:
    st.checkbox('Auto-detect R installation (ignore path below)', value=True, key='r_path_none',
                help=("Lets rpy2 find R automatically using your system's usual R_HOME/PATH "
                      "setup. Uncheck this only if that auto-detection fails and you need to "
                      "point at a specific R installation manually."))
    st.text_input(
        'R installation path', value=r'C:\Program Files\R\R-4.4.0', key='r_path',
        disabled=st.session_state.get('r_path_none', False),
        help="Folder where R is installed (used to run the rrBLUP/GBLUP/BayesB/RKHS models)."
    )

    st.radio(
        'Genotype input format', options=['CSV file', 'PLINK binary fileset (.bed/.bim/.fam)'],
        key='genotype_format', horizontal=True,
        help=("'CSV file': the usual one-row-per-individual marker table. "
              "'PLINK binary fileset': a .bed/.bim/.fam trio sharing one file stem/prefix - "
              "converted to the same structure a genotype CSV would have automatically, "
              "as late and as narrowly (in terms of which markers actually get converted) "
              "as each step of the pipeline allows, to avoid materialising a huge CSV "
              "unnecessarily. See the help text below once selected for exactly when/how "
              "much gets converted.")
    )
    genotype_format = st.session_state.get('genotype_format', 'CSV file')

    if genotype_format == 'CSV file':
        genotype_path = st.text_input(
            'Genotype file path',
            value='./Data/MaizeNAM/MaizeNAM_dataset_genotype_population_1.csv', key='genotype_path',
            help=("The marker data file: one row per individual, with ID, population, and one "
                  "column per genetic marker. See EasiGP's Data format guide if you need to "
                  "convert your data into this layout.")
        )
        file_status(genotype_path)
    else:
        st.text_input(
            'PLINK file stem (path without .bed/.bim/.fam extension)',
            value='', key='genotype_plink_stem',
            help=("E.g. './Data/mydata' if your files are './Data/mydata.bed', "
                  "'./Data/mydata.bim', and './Data/mydata.fam' - all three MUST share this "
                  "exact stem/prefix (PLINK's own convention).")
        )
        plink_fileset_status(st.session_state.get('genotype_plink_stem', '').strip())

        with st.expander('When does this get converted to a CSV-equivalent table?'):
            st.caption(
                "'ID' and 'population' always come from the phenotype file below, never from "
                "the .fam file (which has no 'population' concept) - the .fam file's own IID "
                "column is only used to match samples to the phenotype file's 'ID' column.\n\n"
                "- If **GAT_biological_prior_knowledge** is NOT selected (Models tab): and "
                "**LD pruning** (Data preprocessing tab) is off, the whole fileset converts "
                "once, up front, and everything else proceeds exactly as with a CSV. If LD "
                "pruning is on, PLINK itself prunes directly on the bed/bim/fam fileset first "
                "(per task, using that task's training individuals only) - only the much "
                "smaller *pruned* result is ever converted.\n"
                "- If **GAT_biological_prior_knowledge** IS selected: only the markers that "
                "fall inside a gene's window (from the 'Biological Prior Network' tab's "
                "network) are ever converted for it - never the whole fileset - since it only "
                "ever uses those markers anyway. If other models are selected too, they get "
                "their own conversion (full or LD-pruned, per the point above), independent of "
                "the gene-window one."
            )

        st.text_input(
            'PLINK2 executable', value='plink2', key='genotype_plink_path',
            help="Name/path of the PLINK2 executable used to read this fileset."
        )

    phenotype_path = st.text_input(
        'Phenotype file path',
        value='./Data/MaizeNAM/MaizeNAM_dataset_phenotype_population_1.csv', key='phenotype_path',
        help=("The trait data file: one row per individual (matching the genotype file's IDs), "
              "with ID, population, and one column per trait.")
    )
    file_status(phenotype_path)

    _render_phenotype_target_selector(phenotype_path)
    st.text_input('Result folder name', value='MaizeNAM', key='result_name',
                  help="A name for this run - results are saved under ./Result/<this name>/.")

    # Update ID ver4-9, R7.
    st.selectbox(
        'Result file compression', options=['gzip', 'none'], key='result_compression',
        format_func=lambda v: {
            'gzip': "gzip (.csv.gz) - smaller files on disk, the default",
            'none': "none (.csv) - plain CSV, matches every pre-ver4-9 run",
        }[v],
        help=("Six of the ten result files - Prediction_result_{train,valid,test}.csv, "
              "Marker_effect.csv, Interaction.csv, Attention.csv - are the large, "
              "per-individual/per-marker/per-pair files that dominate a Result folder's disk "
              "usage; 'gzip' compresses those six as they're written (transparently readable "
              "by pandas, and by `zcat`/`gunzip` from the command line). The other four "
              "(Metric.csv, Weight.csv, hyperparameter.csv, Basic_stats.csv) are always left as "
              "plain CSV either way - they're already small, and staying plain keeps a quick "
              "`head`/spreadsheet-double-click workflow available for them. Safe to change "
              "between a run and its own resume - EasiGP checks for both a file's compressed "
              "and uncompressed form when reading."),
    )

    st.selectbox(
        'Prediction scenario', options=['within', 'between'], key='scenario', on_change=on_scenario_change,
        help=("'within': train and test on individuals from the same population(s). "
              "'between': train on one population and test on a different one, to see how "
              "well predictions transfer across populations.")
    )

    if 'ratio_text' not in st.session_state:
        st.session_state['ratio_text'] = RATIO_DEFAULTS[st.session_state.get('scenario', 'within')]

    if st.session_state.get('scenario', 'within') == 'within':
        st.text_input('Split ratio(s)', key='ratio_text',
                       help=("How individuals are randomly divided into train/test or train/validation/test "
                             "sets."))
        st.caption(
            "You can freely choose either a train-only split (plain numbers) or a train/"
            "validation/test split (tuples) - which one you need depends on whether a "
            "validation set is actually required elsewhere. If you select at least one "
            "weight-optimisation approach in '4. Ensemble' and/or enable hyperparameter "
            "tuning for any model (Tab 3) - both need a validation split - type one or more "
            "(train, validation, test) tuples, e.g. (0.8, 0.1, 0.1). Otherwise, type one or "
            "more plain numbers (train ratio only), e.g. 0.8 or 0.8, 0.65."
        )
        st.number_input('Number of random-split iterations', min_value=1, value=1, step=1, key='iter_num',
                         help=("How many times to repeat the whole train/test or train/validation/test split randomly and "
                               "re-run everything for checking consistency in the results."))
        st.number_input(
            'Minimum data points required per iteration', min_value=1, value=100, step=1,
            key='min_data_points',
            help=("After merging the genotype and phenotype datasets for a given prediction "
                  "scenario, if the total number of individuals left (summed across training, "
                  "validation, and test) is below this, that scenario is skipped entirely "
                  "rather than run - predictions from very few data points aren't reliable, "
                  "and this avoids wasting time on them. Applies per scenario (population x "
                  "phenotype x ratio x replicate), after all missing-value/unmatched-ID "
                  "individuals have already been dropped.")
        )
    #else:
    #    st.caption("'between' expects a list of (train, validation, test) tuples, e.g. [(0.8,0.1,0.1)]")

    # Patch 3 v3, Requirement 1: 'Parallel batch configuration (PARALLEL)'
    # used to live here (Tab 1). It has been relocated to right after the
    # 'Run pipeline' header, further down this script, together with a
    # new 'Suggested job-array size' panel + 'Update these values' button
    # shown BEFORE 'Batch ID source' there - "clearly separate the
    # array-related configuration and other resource-related
    # configuration." See _render_parallel_batch_configuration() below.

# ------------------------ Tab 2: Data preprocessing ------------------------ #
if show_tab_models:
    with tab_map['data_preprocessing']:
        st.subheader('LD Pruning')

        _other_models_selected = [
            m for m in AVAILABLE_MODELS
            if m not in ('ensemble', 'GAT_biological_prior_knowledge')
            and st.session_state.get(f'model_selected_{m}', False)
        ]
        _bio_prior_selected = st.session_state.get('model_selected_GAT_biological_prior_knowledge', False)
        if _bio_prior_selected and not _other_models_selected:
            st.info(
                "Only **GAT_biological_prior_knowledge** is currently selected. It always uses "
                "the full, unpruned marker set (it selects its own markers from the "
                "gene-interaction network instead), so LD pruning below will have no effect "
                "until you also select another model."
            )
        elif _bio_prior_selected and _other_models_selected:
            st.caption(
                "**GAT_biological_prior_knowledge** is also selected - LD pruning below applies "
                "only to your other selected model(s) "
                f"({', '.join(_other_models_selected)}); GAT_biological_prior_knowledge always "
                "uses the full, unpruned marker set."
            )

        st.checkbox(
            'Apply LD pruning as a data pre-processing step before model fitting',
            value=False, key='ld_prune_enabled',
            help="When enabled, LD_pruning() is called inside GP() to remove SNPs in "
                 "high linkage disequilibrium before the selected models are fitted. "
                 "GAT_biological_prior_knowledge is never affected by this, regardless of "
                 "whether it's enabled (see the note above)."
        )

        if st.session_state.get('ld_prune_enabled', False):
            st.caption(
                "Genotype data itself is handled automatically (train/validation/test "
                "splits are passed to LD_pruning() internally) - you only need to "
                "configure the pruning behaviour below. "
                "SNPs to be filtered are decided based on the train set."
            )

            st.selectbox(
                'Window unit', options=['kb', 'variants', 'cm'], key='ld_window_unit',
                help="'variants' windows never need SNP coordinates, so all SNPs are "
                     "pruned together regardless of map completeness. 'kb'/'cm' windows "
                     "split mapped and unmapped SNPs into separate passes (see 'Unmapped "
                     "SNPs' below)."
            )
            window_unit = st.session_state.get('ld_window_unit', 'kb')

            # Requirement 1 (Additional_requirements.md): apply a pending
            # Window size/r^2 threshold suggestion BEFORE the 'ld_window'/
            # 'ld_r2_threshold' number_input widgets below are drawn -
            # Streamlit forbids writing to a widget's session_state key
            # after that widget has already been instantiated in the same
            # run (the same rule every other suggestion/auto-fill in this
            # file already works around - see _autofill_number_field's
            # docstring). The flag itself is set by the 'Suggest' button's
            # on_click callback further down, which only triggers a rerun.
            if st.session_state.pop('_pending_suggest_ld_window_r2', False):
                _sugg_window, _sugg_r2, _sugg_note = _ld_suggest_window_and_r2()
                st.session_state['ld_window'] = _sugg_window
                st.session_state['ld_r2_threshold'] = _sugg_r2
                st.session_state['_suggest_msg_ld_window_r2'] = (
                    'success', f"Applied window={_sugg_window:g}, r\u00b2={_sugg_r2:g} - {_sugg_note}."
                )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input(
                    'Window size', min_value=0.0, value=50.0, key='ld_window',
                    help="Units depend on 'Window unit' above: # variants, kilobases, or centimorgans."
                )
            with c2:
                st.number_input('Step size (variant count)', min_value=1, value=1, step=1, key='ld_step',
                                 help="How many SNPs the pruning window slides forward by after each check. "
                                      "Must be exactly 1 when 'Window unit' is 'kb' - PLINK's own "
                                      "--indep-pairwise errors out on any other value for a kb-based window.")
            with c3:
                st.number_input(
                    'r\u00b2 threshold', min_value=0.0, max_value=1.0, value=0.2, key='ld_r2_threshold',
                    help="Unphased hardcall r\u00b2 threshold above which a variant is pruned "
                         "(same meaning as PLINK's --indep-pairwise)."
                )

            st.button(
                'Suggest window size / r\u00b2 threshold', key='_btn_suggest_ld_window_r2',
                on_click=_trigger_suggest_ld_window_r2_cb,
                help=(
                    "Suggests a Window size from this dataset's own median marker spacing (needs a "
                    "valid 'SNP info csv file path' below for 'kb'/'cm', or a .bim file for PLINK "
                    "input - 'variants' never needs one), and an r\u00b2 threshold based on how many "
                    "individuals are in the dataset (fewer individuals \u2192 noisier r\u00b2 estimates "
                    "\u2192 a more lenient threshold is suggested). See the message below once clicked "
                    "for exactly what each value was based on."
                    "Can be used when there is no initial candidate values."
                )
            )
            _show_suggest_message('_suggest_msg_ld_window_r2')

            st.checkbox(
                'Also apply minor allele frequency (MAF) filtering',
                value=False, key='ld_maf_enabled',
                help=("Removes markers with a minor allele frequency below the threshold "
                      "below, BEFORE LD pruning runs - exactly like running plink2 with "
                      "both `--maf` and `--indep-pairwise` together. Decided from the "
                      "training set only, same as LD pruning itself. A marker with no "
                      "genotype calls at all (frequency undefined) is always kept, since "
                      "that's a missing-data issue rather than evidence the marker is rare.")
            )
            if st.session_state.get('ld_maf_enabled', False):
                st.number_input(
                    'MAF threshold', min_value=0.0, max_value=0.5, value=0.05, step=0.01,
                    format='%.3f', key='ld_maf_threshold',
                    help="Markers with a minor allele frequency below this are removed. "
                         "Must be greater than 0 and at most 0.5 (MAF can never exceed 0.5 "
                         "by definition - a threshold of exactly 0.5 would remove every "
                         "marker, since none can have a strictly higher MAF)."
                )

            # Requirement (prevent this from happening silently): shown immediately
            # while configuring, in addition to gather_config()'s own hard block at
            # run time (build_ld_prune_config() raises ValueError for the exact same
            # condition) - this catches it before the user even gets to the "Run
            # pipeline" button, rather than only once they click it.
            if window_unit == 'kb' and int(st.session_state.get('ld_step', 5)) != 1:
                st.error(
                    "PLINK's --indep-pairwise requires **Step size = 1** when **Window unit** "
                    "is 'kb' - PLINK itself errors out on any other value for a kb-based window. "
                    "Set Step size to 1, or switch Window unit to 'variants'."
                )

            snp_info_required = window_unit in ('kb', 'cm')
            snp_info_path = st.text_input(
                f"SNP info csv file path (SNP ID/CHR/POS/CM)"
                f"{' - required for this window unit' if snp_info_required else ' (optional for variant-count windows)'}",
                value='', key='ld_snp_info_path',
                help=("A file with one row per marker and 4 columns: SNP ID (matching your "
                      "genotype file's marker names - used as the row index), CHR (chromosome "
                      "number), POS (physical position in base pairs), and CM (genetic position "
                      "in centimorgans). If you don't have data for either POS or CM, just fill "
                      "that column with 0. Not needed at all if Window unit is 'variants'.")
            )
            if snp_info_path:
                file_status(snp_info_path)

            st.text_input(
                'plink2 executable path', value='plink2', key='ld_plink_path',
                help="Path to the plink2 executable used to compute LD."
            )

            c4, c5 = st.columns(2)
            with c4:
                st.checkbox(
                    'Allow extra/non-standard chromosome names (--allow-extra-chr)',
                    value=True, key='ld_allow_extra_chr',
                    help=("Check this if your chromosome names aren't plain numbers (e.g. "
                          "'scaffold_12'), so plink2 doesn't reject them as invalid.")
                )
            with c5:
                st.checkbox(
                    'Round fractional dosages to hard calls for PED (round_dosage)',
                    value=True, key='ld_round_dosage',
                    help="PED only stores hard genotype calls. If your data has fractional "
                         "dosage values (0.5/1.5), this rounds them to the nearest hardcall "
                         "before writing PED (with a warning) instead of raising an error. "
                         "Doesn't affect the 'cm' path, which uses raw dosages."
                )

            st.checkbox(
                'Non-human genome (set haploid chromosome count)', value=True, key='ld_chr_set_enabled',
                help=("Check this if working with a species that doesn't have the standard "
                      "human 22 autosome + X/Y chromosome layout plink2 assumes by default.")
            )
            st.number_input(
                'Haploid chromosome count (--chr-set)', min_value=1, value=2, step=1,
                key='ld_chr_set', disabled=not st.session_state.get('ld_chr_set_enabled', False),
                help="The number of chromosome pairs for your species (e.g. maize = 10)."
            )

            st.checkbox(
                'Use a custom working directory for intermediate PLINK files',
                value=False, key='ld_work_dir_enabled',
                help="Leave unchecked to use an automatically-created, auto-cleaned temp directory."
            )
            st.text_input(
                'Working directory', value='', key='ld_work_dir',
                disabled=not st.session_state.get('ld_work_dir_enabled', False),
                help="Folder where plink2's intermediate PED/MAP/BED/log files are written."
            )
            st.checkbox(
                'Keep intermediate PED/MAP/BED/log files',
                value=st.session_state.get('ld_work_dir_enabled', False), key='ld_keep_intermediate',
                help="Automatically enabled if a custom working directory is set above."
            )

            st.selectbox(
                'Unmapped-SNP strategy', options=['variant_count', 'skip', 'drop'], key='ld_unmapped_strategy',
                help=(
                    "Only relevant when Window unit is 'kb' or 'cm', for SNPs lacking a "
                    "real map. 'variant_count' prunes them separately using a "
                    "variant-count window based on column order. 'skip' keeps every "
                    "unmapped SNP untouched. 'drop' discards every unmapped SNP outright."
                ),
            )
            if st.session_state.get('ld_unmapped_strategy', 'variant_count') == 'variant_count':
                c6, c7 = st.columns(2)
                with c6:
                    st.number_input(
                        'Unmapped window (variant count)', min_value=1, value=50, step=1, key='ld_unmapped_window',
                        help="Same idea as 'Window size' above, but counted in markers rather than "
                             "kb/cM, since unmapped markers have no known position."
                    )
                with c7:
                    st.number_input(
                        'Unmapped step (variant count)', min_value=1, value=5, step=1, key='ld_unmapped_step',
                        help="Same idea as 'Step size' above, but counted in markers."
                    )

            st.markdown('**LD decay plot**')
            st.checkbox(
                'Generate LD decay plots (population/replicate-level, plus an average per '
                'population)',
                value=False, key='ld_decay_plot_enabled',
                help=("Diagnostic plot of mean r\u00b2 vs. distance between marker pairs, "
                      "computed from a training set's own pre-pruning genotypes - useful for "
                      "judging the window/r\u00b2 threshold settings above. Always plotted in "
                      "the SAME distance unit as 'Window unit' above (kb/cM/variant count), "
                      "since that's the unit those settings are actually expressed in. LD "
                      "decay reflects which INDIVIDUALS are in the training set, which never "
                      "depends on phenotype (only on population/ratio/replicate) - so this is "
                      "sampled at most once per population/ratio/replicate combination, never "
                      "separately for each phenotype that happens to share one. Saved under "
                      "Result/<result name>/LD_decay_plots/, alongside the data used to draw "
                      "each plot (as .csv). Once every prediction scenario in the run has "
                      "finished, an additional averaged plot is produced for each population "
                      "(averaged across replicates, split ratios, and phenotypes).")
            )
            if st.session_state.get('ld_decay_plot_enabled', False):
                _unit_option_labels = {
                    'kb': 'Physical distance (kb)',
                    'cm': 'Genetic distance (cM)',
                    'variants': 'Variant count (markers apart)',
                }
                st.caption(f"Plotted in **{_unit_option_labels.get(window_unit, window_unit)}** - "
                           f"matching 'Window unit' above.")
                st.number_input(
                    'Generate a plot every ___ population/ratio/replicate combination(s)',
                    min_value=1, value=1, step=1, key='ld_decay_plot_frequency',
                    help=("1 = every combination. A real prediction run can have many distinct "
                          "population/ratio/replicate combinations (before accounting for "
                          "phenotype, which never changes the training-set individuals and so "
                          "is never re-sampled - see above), so generating a full plot for "
                          "every single one can still produce a large number of image files - "
                          "increase this to only plot every Nth combination instead. The final "
                          "averaged plot(s) are always built from every combination that was "
                          "plotted, regardless of this setting.")
                )
                st.checkbox(
                    'Keep the per-combination LD decay data (CSV)?',
                    value=True, key='ld_decay_keep_log',
                    help=("On: keep every population/replicate-level .csv file (the numeric "
                          "data behind each plot) after the run finishes - this is the current "
                          "default behaviour. Off: those per-combination .csv files are deleted "
                          "once the per-population average has been computed from them, to save "
                          "disk space on runs with many scenarios. Either way, every PNG plot - "
                          "population/replicate-level AND the per-population average - is "
                          "always kept, and the average is always computed correctly; only the "
                          "per-combination numeric .csv 'log' data is affected by this setting.")
                )
                with st.expander('Advanced LD decay plot settings'):
                    # Requirement 2 (Additional_requirements.md): apply a
                    # pending Max distance/Distance bin width/Max marker
                    # pairs suggestion BEFORE the number_input widgets
                    # below are drawn - same "callback sets a flag, the
                    # flag is applied on the rerun it triggers, before the
                    # target widgets are (re-)instantiated" pattern as the
                    # Window size/r^2 threshold suggestion above (see
                    # _trigger_suggest_ld_window_r2_cb's docstring). The
                    # two distance-unit-dependent keys are suffixed by
                    # `window_unit`, exactly like the widgets themselves,
                    # so a suggestion always lands on the field that's
                    # actually visible for the currently selected unit.
                    if st.session_state.pop('_pending_suggest_ld_decay_params', False):
                        _sugg_maxd, _sugg_binw, _sugg_pairs, _sugg_note = _ld_decay_suggest_params()
                        st.session_state[f'ld_decay_max_distance_{window_unit}'] = _sugg_maxd
                        st.session_state[f'ld_decay_bin_width_{window_unit}'] = _sugg_binw
                        st.session_state['ld_decay_max_pairs_per_chr'] = _sugg_pairs
                        st.session_state['_suggest_msg_ld_decay_params'] = (
                            'success',
                            f"Applied max distance={_sugg_maxd:g}, bin width={_sugg_binw:g}, "
                            f"max pairs/chromosome={_sugg_pairs:,} - {_sugg_note}."
                        )

                    cu1, cu2 = st.columns(2)
                    with cu1:
                        st.number_input(
                            f'Max distance shown ({_UNIT_LABEL_FOR_GUI[window_unit]})',
                            min_value=0.1, value=LD_DECAY_DEFAULT_MAX_DISTANCE[window_unit],
                            key=f'ld_decay_max_distance_{window_unit}',
                            help="Marker pairs further apart than this are not considered at all."
                        )
                    with cu2:
                        st.number_input(
                            f'Distance bin width ({_UNIT_LABEL_FOR_GUI[window_unit]})',
                            min_value=0.01, value=LD_DECAY_DEFAULT_BIN_WIDTH[window_unit],
                            key=f'ld_decay_bin_width_{window_unit}',
                            help="Width of each point/bin along the distance axis."
                        )
                    st.number_input(
                        'Max marker pairs per chromosome', min_value=100, value=20000, step=100,
                        key='ld_decay_max_pairs_per_chr',
                        help=("Caps how many candidate marker pairs are used per "
                              "chromosome, subsampled randomly if exceeded, to keep this "
                              "diagnostic step fast even on datasets with many markers.")
                    )
                    st.button(
                        'Suggest max distance / bin width / max pairs', key='_btn_suggest_ld_decay_params',
                        on_click=_trigger_suggest_ld_decay_params_cb,
                        help=(
                            "Suggests 'Max distance shown' as 10x the configured LD-pruning Window "
                            "size above (capped to this dataset's longest mapped chromosome span, "
                            "when a SNP info/.bim source is available), 'Distance bin width' for "
                            "\u224860 points across that range, and 'Max marker pairs per chromosome' "
                            "scaled so the total pairs sampled stays roughly constant regardless of "
                            "how many chromosomes the dataset has. See the message below once "
                            "clicked for exactly what each value was based on."
                        )
                    )
                    _show_suggest_message('_suggest_msg_ld_decay_params')

        st.divider()
        st.subheader('RF Marker Importance Filtering')

        if _bio_prior_selected and not _other_models_selected:
            st.info(
                "Only **GAT_biological_prior_knowledge** is currently selected. It always uses "
                "the full marker set (it selects its own markers from the gene-interaction "
                "network instead), so RF importance filtering below will have no effect until "
                "you also select another model."
            )
        elif _bio_prior_selected and _other_models_selected:
            st.caption(
                "**GAT_biological_prior_knowledge** is also selected - RF importance filtering "
                "below applies only to your other selected model(s) "
                f"({', '.join(_other_models_selected)}); GAT_biological_prior_knowledge always "
                "uses the full marker set."
            )

        # Patch 3, Requirement 3: an optional cuML GPU backend for the
        # Random Forest fit(s) used to SELECT markers (both this
        # preprocessing step below, and GAT_biological_prior_knowledge's
        # own data-driven-merge side pipeline, Tab 3 - NOT the SHAP
        # pairwise-interaction search, which always stays CPU-only
        # regardless of this setting; see
        # Preprocess/data_driven_prior_network.py's own comment for why).
        # This mirrors models/RF.py's already-existing USE_GPU_SKLEARN
        # switch (Phase 2, Requirement 6) - previously only the model-
        # fitting RF used it; the marker-SELECTION RF (this checkbox) was
        # never wired to it at all, so a GPU job never actually benefited
        # here even when cuML was installed. Auto-suggested ON the moment
        # any GPU-capable model is selected AND at least one 'Request
        # GPU(s)' HPC export purpose has been checked this session -
        # otherwise left at its previous value/default (off), so a
        # CPU-only run's behaviour is completely unaffected.
        _any_gpu_requested = any(
            bool(st.session_state.get(f'{_purpose}_hpc_use_gpu', False))
            for _purpose in ('step1', 'sequential')
        )
        _gpu_sklearn_key = 'use_gpu_sklearn'
        _gpu_sklearn_autofill_key = f'_{_gpu_sklearn_key}_last_autofill'
        _current_gpu_sklearn = st.session_state.get(_gpu_sklearn_key)
        _last_gpu_sklearn_autofill = st.session_state.get(_gpu_sklearn_autofill_key)
        if _current_gpu_sklearn is None or _current_gpu_sklearn == _last_gpu_sklearn_autofill:
            if _current_gpu_sklearn != _any_gpu_requested:
                st.session_state[_gpu_sklearn_key] = _any_gpu_requested
                st.session_state[_gpu_sklearn_autofill_key] = _any_gpu_requested
        st.checkbox(
            'Use GPU-accelerated Random Forest for marker selection when available '
            '(USE_GPU_SKLEARN, experimental)',
            key=_gpu_sklearn_key,
            help="Applies to the Random Forest fit(s) that DECIDE which markers survive "
                 "filtering - RF-based marker importance filtering below, and "
                 "GAT_biological_prior_knowledge's own data-driven-merge feature (Tab 3) - via "
                 "RAPIDS cuML, when a GPU is available and cuML is installed. Never affects the "
                 "SHAP pairwise-interaction search, which always runs on CPU regardless (SHAP's "
                 "TreeExplainer requires a scikit-learn-compatible tree, which cuML's GPU forest "
                 "doesn't provide). If cuML isn't installed, or no GPU is available at run time, "
                 "this silently falls back to today's scikit-learn CPU behaviour - safe to leave "
                 "checked even on a CPU-only node. Auto-suggested on when 'Request GPU(s)' is "
                 "checked elsewhere in this session; untick if you need exact reproducibility "
                 "with a prior CPU-only run (cuML's forest is not guaranteed bit-identical to "
                 "scikit-learn's)."
        )

        st.checkbox(
            'Apply RF-based marker importance filtering as a data pre-processing step',
            value=False, key='rf_filter_enabled',
            help="Fits a Random Forest on each task's training set only (never on validation/"
                 "test data - same principle as LD pruning above), then keeps only the "
                 "top-importance markers before the selected models are fitted. Runs AFTER LD "
                 "pruning above when both are enabled, narrowing whatever LD pruning already "
                 "produced - useful when the marker count is still large even after LD pruning "
                 "alone. GAT_biological_prior_knowledge is never affected by this, regardless "
                 "of whether it's enabled (see the note above)."
        )

        if st.session_state.get('rf_filter_enabled', False):
            st.caption(
                "Markers to keep are decided using the train set only, exactly like LD pruning "
                "above."
            )

            st.radio(
                'How should the number of markers to keep be specified?',
                options=['Percentage of markers', 'Fixed number of markers (M)'],
                key='rf_filter_mode', horizontal=True,
                help="Requirement: choose either a percentage of the current marker pool, or an "
                     "exact count M, and both are ranked by the same "
                     "trained Random Forest feature importance."
            )
            rf_filter_mode = st.session_state.get('rf_filter_mode', 'Percentage of markers')

            if rf_filter_mode == 'Percentage of markers':
                st.number_input(
                    'Keep the top ___ % of markers by RF importance', min_value=0.1, max_value=100.0,
                    value=20.0, step=1.0, key='rf_filter_top_percent',
                    help="Entered as a percentage (e.g. 20 keeps the top 20% of markers, ranked by "
                         "trained Random Forest feature importance); converted to a 0-1 ratio "
                         "internally before being passed to the filter."
                )
            else:
                st.number_input(
                    'Keep the top ___ markers by RF importance (M)', min_value=1, value=100, step=1,
                    key='rf_filter_top_n',
                    help="An exact marker count instead of a percentage - e.g. 100 keeps exactly "
                         "the top 100 markers by trained Random Forest feature importance "
                         "(clamped down to however many markers are actually available, if fewer)."
                )

            with st.expander('Advanced Random Forest settings'):
                # Requirement: pair up logically-related fields on the same
                # row (rather than stacking unrelated widgets unevenly across
                # the two columns), and give every field its own helper text.
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        'Number of trees', min_value=1, value=200, step=10, key='rf_filter_n_estimators',
                        help="More trees give a more stable importance ranking, at the cost of "
                             "longer fitting time."
                    )
                with c2:
                    st.selectbox(
                        'Max features per split', options=['sqrt', 'log2', 'all'], key='rf_filter_max_features',
                        help="How many markers each tree considers at each split. 'all' "
                             "considers every marker (slower, and more prone to overfitting on "
                             "correlated markers)."
                    )

                c3, c4 = st.columns(2)
                with c3:
                    st.checkbox(
                        'Limit max tree depth', value=False, key='rf_filter_max_depth_enabled',
                        help="When checked, caps how many splits deep each tree can grow (set "
                             "with 'Max tree depth' alongside it), which can reduce overfitting "
                             "and speed up fitting. When left unchecked, trees grow until their "
                             "leaves are pure, matching scikit-learn's default behaviour."
                    )
                with c4:
                    st.number_input(
                        'Max tree depth', min_value=1, value=10, step=1, key='rf_filter_max_depth',
                        disabled=not st.session_state.get('rf_filter_max_depth_enabled', False),
                        help="Maximum number of splits from root to leaf, applied only when "
                             "'Limit max tree depth' (alongside it) is checked. Smaller values "
                             "give simpler, faster trees but risk underfitting the importance "
                             "ranking."
                    )

                c5, c6 = st.columns(2)
                with c5:
                    st.number_input(
                        'Min samples per leaf', min_value=1, value=1, step=1, key='rf_filter_min_samples_leaf',
                        help="Minimum number of training samples required at a leaf node. "
                             "Higher values smooth the importance ranking by preventing splits "
                             "that are based on very small, often noisy, groups of samples."
                    )
                with c6:
                    st.number_input(
                        'Random seed', min_value=0, value=0, step=1, key='rf_filter_random_state',
                        help="Fixed for reproducibility - the same seed gives the same importance "
                             "ranking (and hence the same kept markers) for the same data."
                    )

def render_other_models_marker_source_widget():
    """The 'other selected models' marker-source choice - rendered as the
    final step of whichever Biological Prior Network mode is active
    (per-phenotype or repeat), directly after that mode's own 'Preview +
    use these inputs' step, since the two are directly related: this is
    what every OTHER selected model does with the network(s) just
    configured. One shared widget (same key, same value regardless of
    which branch renders it), called once from the end of each mode's own
    branch rather than as a separate section afterwards."""
    st.divider()
    st.caption(
        "**Marker source for other selected models** - only matters if at least one model "
        "*other than* GAT_biological_prior_knowledge is also selected (it always uses its own "
        "gene-network-selected markers regardless of this setting)."
    )
    st.selectbox(
        'Marker source for other selected models',
        options=['full_or_filtered', 'gene_network', 'gene_network_plus_rf'],
        key='other_models_marker_source',
        help=("'full_or_filtered' (default): other models use the full marker set, or "
              "the LD-pruned/RF-importance-filtered pool if either is enabled on this "
              "tab - unchanged from how every other model has always worked. "
              "'gene_network': other models are instead restricted to the SAME markers "
              "GAT_biological_prior_knowledge selected via the gene-interaction network "
              "- LD pruning/RF filtering settings are not applied on top of this, since "
              "it replaces the full/filtered pool rather than narrowing it further. "
              "'gene_network_plus_rf' (requirement 6): only available once the "
              "'Data-driven prior network' section above has been enabled and configured - "
              "other models are restricted to markers included inside gene nodes PLUS "
              "markers selected by RF filtering (the same extended marker set the merged "
              "data-driven graph itself uses), rather than just the plain gene-window set.")
    )


def _default_run_name_for_suffix(suffix):
    """A blank 'Run name' field falls back to run_flash_p()'s own default,
    which is derived ONLY from phenotype - so two network sections for the
    SAME phenotype (e.g. 'Network 1'/'Network 2' in repeat mode, or
    per-phenotype repeats) would silently write to the exact same
    Outcome/Local_FlashP_Outcome/<name>/ folder and overwrite each other.
    Concretely, that means: the earlier of the two runs' full FLASH-P cost
    (literature review through export) is thrown away the moment the next
    one finishes and overwrites it, and worse, BOTH sections end up
    displaying the same last-written network - silently defeating the
    entire point of building independent replicate networks. Returns None
    for suffix='' (the original single-network case - unaffected, and
    matches run_flash_p()'s own phenotype-based default exactly as before),
    otherwise a sanitized, filesystem-safe fragment derived from `suffix`
    so each section's default is naturally distinct.
    """
    if not suffix:
        return None
    return 'network' + re.sub(r'[^A-Za-z0-9_-]+', '_', suffix)


def _make_flashp_log_callback(placeholder, log_lines, log_file_path=None):
    """Requirement ('view FlashP logs in the GUI'): builds the log_callback
    passed to run_flash_p()/preflight_check(). Each call appends one
    timestamped, formatted line (Preprocess.flash_p_integration's own
    _format_stream_event output - assistant messages, tool calls/results,
    the final summary) to `log_lines` and re-renders `placeholder` with the
    accumulated text, so the log grows live in the GUI while FLASH-P is
    still running. If `log_file_path` is given, each line is also appended
    to that file on disk as it arrives - mirroring the TimestampedWriter
    pattern the main pipeline's own run log already uses (pipeline_utils.py)
    - so a FLASH-P run's log is inspectable both live and afterwards, same
    as any other run in this app. A file-write failure is swallowed (never
    allowed to break the FLASH-P run itself over a logging glitch).
    """
    def _callback(line):
        stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        stamped = '\n'.join(f'[{stamp}] {one_line}' for one_line in line.split('\n'))
        log_lines.append(stamped)
        placeholder.code('\n'.join(log_lines), language='text')
        if log_file_path:
            try:
                with open(log_file_path, 'a', encoding='utf-8') as f:
                    f.write(stamped + '\n')
            except Exception:
                pass
    return _callback


def render_bio_prior_network_and_location_section(suffix, species, phenotype_hint=None, show_headers=True):
    """Renders the 'interaction network JSON' + 'gene locations' inputs for
    ONE (json, gene-location-CSV) pair. Every widget key is suffixed with
    `suffix`, so this can be called more than once on the same page (e.g.
    once per phenotype, in per-phenotype mode) without key collisions.
    suffix='' reproduces the original single/shared-mode widget keys
    unchanged, so existing saved GUI state keeps working.

    `species` is shared across every call (species doesn't vary by
    phenotype); `phenotype_hint`, if given, prefills the FLASH-P 'Phenotype'
    field with that exact trait name, but the field stays editable - the
    actual prediction target is always `phenotype_hint` itself (used
    unconditionally for the network/phenotype match check below), so
    freely rewording the FLASH-P search phenotype into something more
    literature-friendly (e.g. 'tillering' instead of a raw column name like
    'days2anthesis') never changes which phenotype this network is used
    for.

    Returns (resolved_json_path, resolved_gene_location_csv_path) - either
    may be '' if not yet resolved.
    """
    if show_headers:
        st.subheader('1. Interaction network (JSON)')
    st.radio(
        'Where does the network JSON come from?',
        options=['Upload an existing network.json', 'Generate with FLASH-P',
                 'Use a previous local FLASH-P run'],
        key=f'bio_json_source{suffix}', horizontal=False,
        help=("A gene-interaction network describing which genes affect the target "
              "trait and how (activation/inhibition), and their interactions with "
              "each other. This can be prepared by hand, produced by any tool that "
              "writes the expected JSON shape, or built automatically with FLASH-P "
              "(https://flash-p.com/, https://github.com/CMits/FlashP), a multi-agent "
              "system that mines primary literature for a given (species, phenotype) "
              "pair.")
    )
    json_source = st.session_state.get(f'bio_json_source{suffix}', 'Upload an existing network.json')

    if json_source == 'Upload an existing network.json':
        json_path = st.text_input(
            'network.json path', value='', key=f'bio_network_json_path{suffix}',
            help="Path to an existing network JSON file (nodes/edges, FLASH-P's own "
                 "output shape)."
        )
        if json_path:
            file_status(json_path)
        if json_path and os.path.isfile(json_path):
            st.session_state[f'bio_network_json_resolved{suffix}'] = json_path

    elif json_source == 'Generate with FLASH-P':
        st.caption(
            "Runs FLASH-P's full literature-mining pipeline via Claude Code, headlessly. "
            "This can take a while (multiple agent steps, each doing real web research) "
            "and needs `npm install -g @anthropic-ai/claude-code` available on this "
            "machine. Configure once here, then click 'Run FLASH-P now'."
        )
        st.text_input(
            'Phenotype', value=(phenotype_hint or 'Days to Flowering'), key=f'bio_phenotype_name{suffix}',
            help=("Passed to FLASH-P as e.g. 'Days to Flowering'. Prefilled from the phenotype "
                  "this network is for, but feel free to change it to a more literature-"
                  "friendly description (e.g. a common/descriptive name instead of a raw "
                  "column name) - FLASH-P's literature search often works better with an "
                  "informative phenotype name. This never changes which phenotype the "
                  "resulting network is actually used for."),
        )
        st.text_input(
            'FLASH-P checkout path', value='./FlashP', key=f'bio_flashp_dir{suffix}',
            help="Local clone of github.com/CMits/FlashP. Cloned automatically here if "
                 "this path doesn't exist yet or is empty."
        )
        with st.expander('Advanced FLASH-P settings'):
            st.selectbox(
                'Pipeline variant', options=['Flash-P_Plant', 'Flash-P_Medical', 'Flash-P_Animal'],
                key=f'bio_flashp_variant{suffix}',
                help="Which FLASH-P version to run - see FLASH-P's README 'Versions' table."
            )
            st.radio(
                'Run mode', options=['Light (recommended)', 'Full'],
                key=f'bio_flashp_light_mode{suffix}', horizontal=True,
                help=("Both options run the exact same pipeline above - only the prompt differs "
                      "(verified against Flash-P_Plant/README.md's own 'FLASH-P Light' guide). "
                      "'Light': keeps literature review to a single agent doing a knowledge-first "
                      "draft verified only via WebSearch - no subagents, no WebFetching full "
                      "papers, 'the two things that blow the token budget' per FLASH-P's own docs. "
                      "'Full': sends the plain instruction with no such constraint, letting the "
                      "pipeline use subagents/WebFetch as it judges necessary - slower and far "
                      "more token-hungry, but potentially more thorough for a niche or "
                      "under-studied trait where the model's own background knowledge is thin.")
            )
            st.text_input('Run name (leave blank for the default)', value='', key=f'bio_flashp_run_name{suffix}',
                          help="Destination folder name under Outcome/Local_FlashP_Outcome/. "
                               "Defaults to '<phenotype_slug_lowercased>_local_network'.")
            st.text_input('claude executable', value='claude', key=f'bio_claude_binary{suffix}',
                          help="Name/path of the Claude Code executable on this machine.")
            st.number_input('Timeout (seconds)', min_value=60, value=3600, step=60, key=f'bio_flashp_timeout{suffix}',
                             help="How long to wait for the FLASH-P run before giving up.")
            st.selectbox(
                'Permission mode', options=['auto', 'dontAsk', 'bypassPermissions', 'acceptEdits', 'default'],
                key=f'bio_permission_mode{suffix}',
                help=("Controls how much Claude Code auto-approves during the run. "
                      "'auto' (default): routes each tool call "
                      "through Claude Code's own classifier instead - it reads FLASH-P's "
                      "CLAUDE.md for context and judges each action's real risk, so it "
                      "doesn't need a hand-maintained tool list at all. Not guaranteed "
                      "available for every account/model (use 'Test Claude Code "
                      "connection' below to check first) and Anthropic's own docs note "
                      "a small added cost/latency per tool call for the classifier "
                      "itself - usually still cheaper overall than a run stalling on a "
                      "missing tool, but try the cheap connection test before committing "
                      "a full run to it. "
                      "'dontAsk': auto-approves everything in this integration's "
                      "own tool list and silently denies anything else - no prompts, no "
                      "dialog, works on any account/plan, but that list has to be kept "
                      "up to date by hand. "
                      "'bypassPermissions': skips every check, but Claude Code refuses to "
                      "start in this mode at all when running as root/sudo, and even "
                      "otherwise needs a one-time interactive dialog accepted on this "
                      "machine first. 'acceptEdits': auto-approves file edits only, "
                      "will still block waiting for approval on Bash/WebSearch/Task calls "
                      "in a headless run. 'default': prompts for everything (interactive "
                      "use only, not recommended here).")
            )
            st.selectbox(
                'Authentication', options=["Use this app's environment (default)",
                                            'API key', 'Subscription OAuth token'],
                key=f'bio_auth_method{suffix}',
                help=("How the claude subprocess authenticates. Default: whatever "
                      "ANTHROPIC_API_KEY / logged-in session is already in this app's "
                      "environment - that's normally all you need. 'API key': pay-per-"
                      "token billing via an Anthropic Console key, entered below. "
                      "'Subscription OAuth token': bills against a Claude Pro/Max/Team/"
                      "Enterprise plan instead - generate one with `claude setup-token` "
                      "(run once, interactively, on a machine where you can log in with "
                      "a browser - this deployment likely can't) and paste it below. Any "
                      "ANTHROPIC_API_KEY already in this app's environment is unset for "
                      "this run so the token isn't silently overridden. Set this if "
                      "'Run FLASH-P now' fails with an authentication/401 error.")
            )
            auth_method = st.session_state.get(f'bio_auth_method{suffix}', "Use this app's environment (default)")
            if auth_method == 'API key':
                st.text_input('API key', value='', type='password', key=f'bio_claude_api_key{suffix}',
                              help="Passed to the claude subprocess as ANTHROPIC_API_KEY. Not saved anywhere else.")
            elif auth_method == 'Subscription OAuth token':
                st.text_input('OAuth token', value='', type='password', key=f'bio_claude_oauth_token{suffix}',
                              help="From `claude setup-token`. Passed to the claude subprocess as "
                                   "CLAUDE_CODE_OAUTH_TOKEN. Not saved anywhere else.")
            st.text_input(
                'Resume session ID (optional)', value='', key=f'bio_resume_session_id{suffix}',
                help=("Leave blank for a normal new run. If a previous run was interrupted "
                      "partway through - most commonly by a rate/session limit ('You've hit "
                      "your session limit...') - its error message includes a "
                      "resume_session_id. Paste it here and click 'Run FLASH-P now' again "
                      "once the limit has reset to continue that SAME run from where it left "
                      "off, instead of restarting FLASH-P from scratch (which repeats, and "
                      "re-pays for, everything already done). Keep Species/Phenotype above "
                      "set to the SAME values as the original run - they're still needed to "
                      "locate the finished network.json afterward.")
            )
            if st.button('Test Claude Code connection (cheap)', key=f'_btn_bio_preflight{suffix}'):
                test_auth_method = st.session_state.get(f'bio_auth_method{suffix}', "Use this app's environment (default)")
                test_extra_env = None
                if test_auth_method == 'API key':
                    test_api_key = st.session_state.get(f'bio_claude_api_key{suffix}', '').strip()
                    if test_api_key:
                        test_extra_env = {'ANTHROPIC_API_KEY': test_api_key}
                elif test_auth_method == 'Subscription OAuth token':
                    test_oauth_token = st.session_state.get(f'bio_claude_oauth_token{suffix}', '').strip()
                    if test_oauth_token:
                        test_extra_env = {'CLAUDE_CODE_OAUTH_TOKEN': test_oauth_token, 'ANTHROPIC_API_KEY': None}
                preflight_log_placeholder = st.empty()
                preflight_log_lines = []
                with st.spinner('Sending one trivial prompt to check auth/permission_mode...'):
                    ok, message = preflight_check(
                        cwd='.',
                        claude_binary=st.session_state.get(f'bio_claude_binary{suffix}', 'claude').strip(),
                        permission_mode=st.session_state.get(f'bio_permission_mode{suffix}', 'dontAsk'),
                        extra_env=test_extra_env,
                        log_callback=_make_flashp_log_callback(preflight_log_placeholder, preflight_log_lines),
                    )
                (st.success if ok else st.error)(message)
                st.caption(
                    "This only checks that Claude Code can authenticate and start with your "
                    "current Authentication/Permission mode settings - a few thousand tokens "
                    "at most. It doesn't touch FLASH-P itself, so passing here doesn't "
                    "guarantee the full run will succeed (e.g. it won't catch a missing "
                    "FLASH-P checkout or a pipeline-specific tool gap), but it does catch the "
                    "most common failures - bad credentials, wrong permission_mode for this "
                    "environment, claude not on PATH - before you pay for a full run to find "
                    "out."
                )

        if st.button('Run FLASH-P now', key=f'_btn_bio_run_flashp{suffix}'):
            # Requirement ('view FlashP logs in the GUI'): the expander/
            # placeholder are created BEFORE the run starts (not after), and
            # the log_callback below writes into them as each line arrives,
            # so the log grows live while FLASH-P is running - not just
            # after it finishes. Also tee'd to a file via make_run_log_path,
            # the same helper the main pipeline's own run log uses, so a
            # FLASH-P run's log is saved to disk exactly like any other run.
            flashp_log_path = make_run_log_path(
                st.session_state.get('result_name', 'FlashP') or 'FlashP', f'flashp{suffix or "_run"}'
            )
            with st.expander('FLASH-P log', expanded=True):
                st.caption(f'Also saved to: {flashp_log_path}')
                log_placeholder = st.empty()
            log_lines = []
            log_callback = _make_flashp_log_callback(log_placeholder, log_lines, flashp_log_path)
            try:
                with st.spinner('Running FLASH-P (this can take a while)... see the log above.'):
                    auth_method = st.session_state.get(f'bio_auth_method{suffix}', "Use this app's environment (default)")
                    extra_env = None
                    if auth_method == 'API key':
                        api_key = st.session_state.get(f'bio_claude_api_key{suffix}', '').strip()
                        if api_key:
                            extra_env = {'ANTHROPIC_API_KEY': api_key}
                    elif auth_method == 'Subscription OAuth token':
                        oauth_token = st.session_state.get(f'bio_claude_oauth_token{suffix}', '').strip()
                        if oauth_token:
                            # Unset ANTHROPIC_API_KEY (None -> removed, see
                            # _run_claude_headless) so it can't silently
                            # override the subscription token.
                            extra_env = {'CLAUDE_CODE_OAUTH_TOKEN': oauth_token, 'ANTHROPIC_API_KEY': None}
                    resolved_path = run_flash_p(
                        species=species.strip(),
                        # Requirement: use whatever the (now always-editable)
                        # 'Phenotype' field actually contains - previously this
                        # ignored the field entirely and silently substituted
                        # phenotype_hint whenever it was set, which is exactly
                        # what made the field impossible to meaningfully
                        # change. phenotype_hint remains the field's initial
                        # prefill (see the text_input above) and is still what
                        # the network/phenotype match check further below
                        # verifies against - only the FLASH-P search query
                        # itself is user-editable.
                        phenotype=st.session_state.get(f'bio_phenotype_name{suffix}', phenotype_hint or '').strip(),
                        flashp_dir=st.session_state.get(f'bio_flashp_dir{suffix}', './FlashP').strip(),
                        pipeline_variant=st.session_state.get(f'bio_flashp_variant{suffix}', 'Flash-P_Plant'),
                        run_name=(st.session_state.get(f'bio_flashp_run_name{suffix}', '').strip()
                                  or _default_run_name_for_suffix(suffix)),
                        claude_binary=st.session_state.get(f'bio_claude_binary{suffix}', 'claude').strip(),
                        timeout_seconds=int(st.session_state.get(f'bio_flashp_timeout{suffix}', 3600)),
                        permission_mode=st.session_state.get(f'bio_permission_mode{suffix}', 'dontAsk'),
                        extra_env=extra_env,
                        resume_session_id=(st.session_state.get(f'bio_resume_session_id{suffix}', '').strip() or None),
                        log_callback=log_callback,
                        light_mode=(st.session_state.get(f'bio_flashp_light_mode{suffix}', 'Light (recommended)')
                                    == 'Light (recommended)'),
                    )
                st.session_state[f'bio_network_json_resolved{suffix}'] = resolved_path
                st.success(f'FLASH-P run complete: `{resolved_path}`')
            except Exception as exc:
                st.error(f'FLASH-P run failed: {exc}')
                st.code(traceback.format_exc(), language='text')

    else:  # 'Use a previous local FLASH-P run'
        c1, c2 = st.columns(2)
        with c1:
            st.text_input('FLASH-P checkout path', value='./FlashP', key=f'bio_flashp_dir{suffix}',
                          help="Local clone of github.com/CMits/FlashP containing "
                               "Outcome/Local_FlashP_Outcome/.")
        with c2:
            st.text_input(
                'Run name (leave blank to auto-pick the most recent)', value='', key=f'bio_flashp_run_name{suffix}',
                help="Folder name under Outcome/Local_FlashP_Outcome/, e.g. "
                     "'shoot_branching_local_network'."
            )
        if st.button('Locate network.json', key=f'_btn_bio_locate_flashp{suffix}'):
            try:
                resolved_path = locate_network_json(
                    st.session_state.get(f'bio_flashp_dir{suffix}', './FlashP').strip(),
                    run_name=(st.session_state.get(f'bio_flashp_run_name{suffix}', '').strip() or None),
                )
                st.session_state[f'bio_network_json_resolved{suffix}'] = resolved_path
                st.success(f'Found: `{resolved_path}`')
            except Exception as exc:
                st.error(f'Could not locate a previous run: {exc}')

    resolved_json = st.session_state.get(f'bio_network_json_resolved{suffix}', '')
    if resolved_json:
        st.caption(f'Using network JSON: `{resolved_json}`')
        # Only meaningful in per-phenotype mode, where phenotype_hint is the
        # one specific trait this section's network is supposed to be for -
        # in shared mode there's no single phenotype to check against (the
        # same network is deliberately used for every phenotype in the run).
        if phenotype_hint is not None:
            try:
                network_for_check = load_network_json(resolved_json)
                matched, declared = phenotype_matches_network_metadata(network_for_check, phenotype_hint)
            except Exception:
                matched, declared = None, None
            if matched is False:
                st.warning(
                    f"This network's own metadata says it's for **{declared!r}**, which doesn't "
                    f"obviously match the phenotype **{phenotype_hint!r}** this section is "
                    f"configuring. This is a heuristic name check, not proof of a mistake - "
                    f"double-check this is the right file for this trait."
                )

    st.divider()
    if show_headers:
        st.subheader('2. Gene locations')
    st.radio(
        'How should each gene in the network be mapped to a chromosome position?',
        options=['I already have a curated CSV', 'Curate automatically with an agent'],
        key=f'bio_gene_location_mode{suffix}',
        help=("Every gene node in the network needs a genomic location (chromosome, "
              "start, end) before its SNPs can be identified. This can be a table you "
              "already prepared (e.g. by hand from https://www.maizegdb.org/ or https://www.arabidopsis.org/), or "
              "built automatically by an agent that researches each gene. "
              "Currently, the token usage for the agent method is high, which needs to be improved in the future.")
    )
    gene_location_mode = st.session_state.get(f'bio_gene_location_mode{suffix}', 'I already have a curated CSV')

    if gene_location_mode == 'I already have a curated CSV':
        gl_path = st.text_input(
            'Gene location lookup CSV path', value='', key=f'bio_gene_location_csv_path{suffix}',
            help="Columns: Gene_Name, Start_bp/Start_cM, End_bp/End_cM, Chromosome, "
                 "and optionally AGI_Locus_ID, Source (the generic name/chromosome/"
                 "start/end schema is also accepted)."
        )
        if gl_path:
            file_status(gl_path)

    else:
        st.caption(
            "Asks an agent to research each candidate gene's real genomic location "
            "and write the curated CSV itself - it's instructed to omit (never guess "
            "at) any gene it can't confidently verify."
        )
        st.selectbox('Agent backend', options=['claude_code', 'crewai'], key=f'bio_curation_backend{suffix}',
                     help="'claude_code': headless Claude Code with a research prompt "
                          "(no extra dependency). 'crewai': a two-agent crew (Curator "
                          "+ independent QC Reviewer) - needs `pip install crewai "
                          "crewai-tools`, and a real web-search tool for reliable "
                          "results (see Preprocess/gene_location_agent.py).")
        if st.session_state.get(f'bio_curation_backend{suffix}', 'claude_code') == 'claude_code':
            st.selectbox(
                'Permission mode', options=['auto', 'dontAsk', 'bypassPermissions', 'acceptEdits', 'default'],
                key=f'bio_curation_permission_mode{suffix}',
                help="Same trade-off as FLASH-P's permission mode above - "
                     "'auto' (default) uses Claude Code's classifier instead of a hand-maintained "
                     "tool list, worth trying if cost is a concern (test with 'Test "
                     "Claude Code connection' first - not every account/model supports "
                     "it); 'dontAsk' runs the curation agent fully unattended "
                     "without needing a permission dialog or non-root privileges; "
                     "'bypassPermissions' refuses to start as root/sudo. Not used "
                     "by the 'crewai' backend."
            )
            st.selectbox(
                'Authentication', options=["Use this app's environment (default)",
                                            'API key', 'Subscription OAuth token'],
                key=f'bio_curation_auth_method{suffix}',
                help="Same choice as FLASH-P's authentication setting above. Set this "
                     "if the curation run fails with an authentication/401 error. Not "
                     "used by the 'crewai' backend."
            )
            curation_auth_method = st.session_state.get(
                f'bio_curation_auth_method{suffix}', "Use this app's environment (default)")
            if curation_auth_method == 'API key':
                st.text_input('API key', value='', type='password', key=f'bio_curation_claude_api_key{suffix}',
                              help="Passed to the claude subprocess as ANTHROPIC_API_KEY. Not saved anywhere else.")
            elif curation_auth_method == 'Subscription OAuth token':
                st.text_input('OAuth token', value='', type='password', key=f'bio_curation_claude_oauth_token{suffix}',
                              help="From `claude setup-token`. Passed to the claude subprocess as "
                                   "CLAUDE_CODE_OAUTH_TOKEN. Not saved anywhere else.")
            st.text_input(
                'claude executable', value='claude', key=f'bio_curation_claude_binary{suffix}',
                help="Name or absolute path of the Claude Code executable. Leave as 'claude' unless "
                     "'Run curation agent now' fails with \"No such file or directory: 'claude'\" - "
                     "that means the bare name isn't on PATH for whatever user runs this app, and "
                     "you need the exact absolute path instead (find it with `which claude` as that "
                     "same user inside the container)."
            )
            st.number_input(
                'Genes per agent session (batch size)', min_value=1, value=5, step=1,
                key=f'bio_curation_batch_size{suffix}',
                help=("Requirement (token efficiency / avoid unfinished runs): candidate genes are "
                      "split into independent batches of this size, each run as its own Claude Code "
                      "session, instead of one huge session covering every candidate gene - the "
                      "previous behaviour, which was the main cause of both very high token usage "
                      "and runs that stalled out before finishing every gene. Lower this further for "
                      "a very large network or a token-constrained plan; one batch failing no longer "
                      "loses any other batch's already-curated results.")
            )
            st.checkbox(
                'Allow WebFetch (full page fetch) as a last resort per gene', value=False,
                key=f'bio_curation_allow_webfetch{suffix}',
                help=("Off (default, recommended): the agent verifies each gene with WebSearch only - "
                      "a search snippet is normally enough to confirm a coordinate, and this is by far "
                      "the largest token saving (the same finding FLASH-P's own 'Light' mode is built "
                      "on - see the 'Run mode' help text above). On: WebFetch is offered back as a "
                      "last resort for a gene a WebSearch snippet alone can't confirm - useful for a "
                      "small, targeted retry batch of just the genes an off run left unresolved, at a "
                      "substantially higher token cost.")
            )
            if st.button('Test Claude Code connection (cheap)', key=f'_btn_bio_curation_preflight{suffix}'):
                test_curation_auth_method = st.session_state.get(
                    f'bio_curation_auth_method{suffix}', "Use this app's environment (default)")
                test_curation_extra_env = None
                if test_curation_auth_method == 'API key':
                    test_curation_api_key = st.session_state.get(f'bio_curation_claude_api_key{suffix}', '').strip()
                    if test_curation_api_key:
                        test_curation_extra_env = {'ANTHROPIC_API_KEY': test_curation_api_key}
                elif test_curation_auth_method == 'Subscription OAuth token':
                    test_curation_oauth_token = st.session_state.get(f'bio_curation_claude_oauth_token{suffix}', '').strip()
                    if test_curation_oauth_token:
                        test_curation_extra_env = {'CLAUDE_CODE_OAUTH_TOKEN': test_curation_oauth_token,
                                                    'ANTHROPIC_API_KEY': None}
                # Mirrors curate_gene_locations_claude_code()'s own cwd resolution
                # (os.path.dirname(output_csv) or '.') - the curation agent's cwd is
                # unrelated to FLASH-P's cwd, so this needs its own preflight rather
                # than reusing the FLASH-P button above.
                test_curation_output_csv = os.path.abspath(
                    st.session_state.get(f'bio_curation_output_csv{suffix}',
                                          './Data/_bio_prior_network/gene_location_lookup.csv').strip()
                    or './Data/_bio_prior_network/gene_location_lookup.csv'
                )
                with st.spinner('Sending one trivial prompt to check auth/permission_mode...'):
                    ok, message = preflight_check(
                        cwd=os.path.dirname(test_curation_output_csv) or '.',
                        claude_binary=st.session_state.get(f'bio_curation_claude_binary{suffix}', 'claude').strip(),
                        permission_mode=st.session_state.get(f'bio_curation_permission_mode{suffix}', 'dontAsk'),
                        extra_env=test_curation_extra_env,
                    )
                (st.success if ok else st.error)(message)
        st.caption(
            f"Coordinate unit: **{st.session_state.get('bio_coordinate_unit', 'bp')}** "
            "(set above)."
        )
        st.text_input(
            'Output CSV path', value='./Data/_bio_prior_network/gene_location_lookup.csv',
            key=f'bio_curation_output_csv{suffix}',
            help="Where the agent should write the curated CSV."
        )
        if st.button('Run curation agent now', key=f'_btn_bio_curate{suffix}'):
            if not resolved_json:
                st.error('Set up the interaction network JSON in section 1 first.')
            else:
                # Requirement ('view FlashP logs in the GUI' - same pattern applied here
                # since this agent is driven the same way): expander/placeholder created
                # BEFORE the run starts, log_callback grows it live as each batch runs.
                curation_log_path = make_run_log_path(
                    st.session_state.get('result_name', 'FlashP') or 'FlashP', f'genecuration{suffix or "_run"}'
                )
                curation_log_lines = []
                curation_log_callback = None
                if st.session_state.get(f'bio_curation_backend{suffix}', 'claude_code') == 'claude_code':
                    with st.expander('Gene-location curation log', expanded=True):
                        st.caption(f'Also saved to: {curation_log_path}')
                        curation_log_placeholder = st.empty()
                    curation_log_callback = _make_flashp_log_callback(
                        curation_log_placeholder, curation_log_lines, curation_log_path
                    )
                try:
                    with st.spinner('Researching gene locations (this can take a while)...'):
                        network = load_network_json(resolved_json)
                        candidate_genes = extract_candidate_genes(network)
                        candidate_ids = [g['id'] for g in candidate_genes]
                        curation_backend = st.session_state.get(f'bio_curation_backend{suffix}', 'claude_code')
                        extra_kwargs = {}
                        if curation_backend == 'claude_code':
                            # 'crewai' backend has no permission_mode/extra_env/claude_binary/
                            # batch_size/allow_web_fetch/log_callback parameters, so these are
                            # only forwarded for the backend that accepts them.
                            extra_kwargs['claude_binary'] = st.session_state.get(
                                f'bio_curation_claude_binary{suffix}', 'claude'
                            ).strip() or 'claude'
                            extra_kwargs['permission_mode'] = st.session_state.get(
                                f'bio_curation_permission_mode{suffix}', 'dontAsk'
                            )
                            extra_kwargs['batch_size'] = int(st.session_state.get(f'bio_curation_batch_size{suffix}', 20))
                            extra_kwargs['allow_web_fetch'] = st.session_state.get(
                                f'bio_curation_allow_webfetch{suffix}', False
                            )
                            extra_kwargs['log_callback'] = curation_log_callback
                            curation_auth_method = st.session_state.get(
                                f'bio_curation_auth_method{suffix}', "Use this app's environment (default)")
                            if curation_auth_method == 'API key':
                                api_key = st.session_state.get(f'bio_curation_claude_api_key{suffix}', '').strip()
                                if api_key:
                                    extra_kwargs['extra_env'] = {'ANTHROPIC_API_KEY': api_key}
                            elif curation_auth_method == 'Subscription OAuth token':
                                oauth_token = st.session_state.get(f'bio_curation_claude_oauth_token{suffix}', '').strip()
                                if oauth_token:
                                    extra_kwargs['extra_env'] = {
                                        'CLAUDE_CODE_OAUTH_TOKEN': oauth_token, 'ANTHROPIC_API_KEY': None
                                    }
                        _, resolved_ids, unresolved_ids = curate_gene_locations(
                            candidate_ids,
                            species.strip(),
                            st.session_state.get(f'bio_curation_output_csv{suffix}', '').strip(),
                            backend=curation_backend,
                            unit=st.session_state.get('bio_coordinate_unit', 'bp'),
                            **extra_kwargs,
                        )
                    st.session_state[f'bio_gene_location_csv_path{suffix}'] = st.session_state.get(f'bio_curation_output_csv{suffix}', '').strip()
                    st.session_state[f'bio_curation_summary{suffix}'] = {
                        'resolved': resolved_ids, 'unresolved': unresolved_ids,
                    }
                    st.success(f'Curated {len(resolved_ids)}/{len(candidate_ids)} candidate gene(s).')
                except Exception as exc:
                    st.error(f'Curation agent failed: {exc}')
                    st.code(traceback.format_exc(), language='text')

        summary = st.session_state.get(f'bio_curation_summary{suffix}')
        if summary:
            st.caption(f"Resolved: {summary['resolved']}")
            if summary['unresolved']:
                st.caption(f"Dropped (could not be verified, not guessed at): {summary['unresolved']}")

    resolved_gene_csv = st.session_state.get(f'bio_gene_location_csv_path{suffix}', '')
    return resolved_json, resolved_gene_csv


# Biological Prior Network - second half of the merged "Data preprocessing"
# tab (continues inside the same `with tab_map['data_preprocessing']:` block
# opened above for LD Pruning). Builds the two curated inputs the
# GAT_biological_prior_knowledge model needs (a network JSON + a
# gene-location CSV) - either a single pair shared across every phenotype,
# or one pair per phenotype (see the toggle below). The model itself builds
# the gene list + gene-gene adjacency graph automatically from these two
# files every time it runs. Only configurable (and only relevant) once that
# model is actually selected on the Models tab.
if show_tab_models:
    with tab_map['data_preprocessing']:
        st.divider()
        st.subheader('Biological Prior Network')

        if not st.session_state.get('model_selected_GAT_biological_prior_knowledge', False):
            st.info(
                "Select **GAT_biological_prior_knowledge** on the 'Models & Hyperparameters' "
                "tab to configure this section. Every other model ignores it."
            )
        elif not BIO_PRIOR_AVAILABLE:
            st.warning(
                "The preprocessing modules for this model "
                "(`Preprocess/gene_network_prior.py`, `flash_p_integration.py`, "
                "`gene_location_agent.py`) weren't found, so this tab is disabled. "
                f"Import error: `{BIO_PRIOR_IMPORT_ERROR}`. Add those three files to your "
                "`Preprocess/` folder to enable it - the rest of the app is unaffected."
            )
        else:
            st.text_input(
                'Species', value='Maize', key='bio_species',
                help="Shared across every phenotype below - passed to FLASH-P and to the "
                     "gene-location curation agent."
            )
            species = st.session_state.get('bio_species', 'Maize')

            st.markdown('**Gene-to-marker mapping settings**')
            st.caption(
                "Shared across every network below, regardless of mode - these control how "
                "genes get matched to markers and to each other, not the model's own training."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                st.selectbox(
                    'Coordinate unit', options=['bp', 'cM'], key='bio_coordinate_unit',
                    help="Must match both the gene-location CSV and the marker info CSV's "
                         "coordinate units."
                )
            with c2:
                st.checkbox(
                    'Include mediated edges', value=True, key='bio_include_mediated_edges',
                    help="Include gene-gene edges mediated through non-gene nodes (e.g. a "
                         "shared hormone/metabolite), not just direct gene-gene edges."
                )
            with c3:
                st.number_input(
                    'Max hops for mediated edges', min_value=1, value=3, step=1,
                    key='bio_max_hops', disabled=not st.session_state.get('bio_include_mediated_edges', True),
                    help="How many non-gene nodes a mediated path is allowed to pass through."
                )
            st.divider()

            st.radio(
                'How many networks do you want to configure?',
                options=[
                    'One network per phenotype',
                    'Several independent networks for the same phenotype(s) (e.g. repeated FlashP runs)',
                ],
                key='bio_network_mode',
                help=("The second option is for FlashP's own non-determinism - running it again "
                      "on the same phenotype produces a genuinely different network each time. "
                      "Configuring N of them here creates N independent "
                      "GAT_biological_prior_knowledge models (named "
                      "GAT_biological_prior_knowledge_1, _2, ... in every output file's 'model' "
                      "column), so you can see how much the specific network affects the result. "
                      "By default the same N networks apply across every phenotype; a checkbox "
                      "inside that option lets you configure a separate set of N networks for "
                      "each phenotype instead (each network is phenotype-specific, e.g. from a "
                      "FlashP run for that one trait).")
            )
            bio_network_mode = st.session_state.get('bio_network_mode', 'One network per phenotype')
            per_phenotype_mode = bio_network_mode == 'One network per phenotype'
            repeat_mode = bio_network_mode.startswith('Several independent networks')
            # Kept in sync in session_state (not just a local variable) since
            # gather_config() reads this directly to decide how to build
            # GAT_biological_prior_knowledge's own HPARAMETERS entry.
            st.session_state['bio_per_phenotype_mode'] = per_phenotype_mode

            if per_phenotype_mode:
                raw_pheno = st.session_state.get('phenotype_targets', '').strip()
                if not raw_pheno or raw_pheno.lower() == 'all':
                    st.warning(
                        "Per-phenotype mode needs explicit phenotype names in Tab 1's 'Target "
                        "phenotype(s)' field (not blank or 'all') so each one can get its own "
                        "network configured below."
                    )
                    phenotype_list = []
                else:
                    phenotype_list = [p.strip() for p in raw_pheno.split(',') if p.strip()]

                resolved_per_phenotype = {}
                for ph in phenotype_list:
                    with st.expander(f'\U0001f9ec {ph}', expanded=False):
                        r_json, r_gene_csv = render_bio_prior_network_and_location_section(
                            suffix=f'__{ph}', species=species, phenotype_hint=ph, show_headers=False
                        )
                        resolved_per_phenotype[ph] = (r_json, r_gene_csv)
                        #if r_json and r_gene_csv:
                        #    st.caption(f"\u2705 Ready: `{r_json}` + `{r_gene_csv}`")
                        #else:
                        #    st.caption("\u26a0\ufe0f Not fully configured yet.")

                st.divider()
                st.caption(
                    "'Gene-to-marker mapping settings' above (shared across every phenotype) "
                    "apply here too, same as in every other mode."
                )
                render_data_driven_merge_section()

                with st.expander('Optional: save preview copies to disk for inspection'):
                    st.checkbox('Also save a preview CSV pair per phenotype', value=False, key='bio_save_preview_multi')
                    st.text_input(
                        'Preview output folder', value='./Data/_bio_prior_network', key='bio_preview_out_dir',
                        disabled=not st.session_state.get('bio_save_preview_multi', False)
                    )

                if st.button('Preview + use per-phenotype inputs for the model',
                             key='_btn_bio_build_per_phenotype', type='primary'):
                    if not phenotype_list:
                        st.error("No phenotypes to configure - set 'Target phenotype(s)' on Tab 1 first.")
                    else:
                        missing = [ph for ph, (j, g) in resolved_per_phenotype.items() if not j or not g]
                        if missing:
                            st.error(f'Finish configuring these phenotype(s) first: {missing}')
                        else:
                            try:
                                unit_value = st.session_state.get('bio_coordinate_unit', 'bp')
                                include_mediated_value = st.session_state.get('bio_include_mediated_edges', True)
                                max_hops_value = int(st.session_state.get('bio_max_hops', 3))

                                # Only the (json, gene_csv) pair is stored per phenotype here -
                                # NOT a full params list. The other, shared hyperparameters
                                # (marker_info_path, epoch, etc.) are read fresh from the Models
                                # tab in gather_config() at run time instead of being snapshotted
                                # now, so setting/changing them AFTER clicking this button (a very
                                # natural order - e.g. filling in marker_info_path afterwards)
                                # still takes effect, rather than silently being ignored.
                                per_phenotype_paths = {}
                                summary_rows = []
                                for ph, (json_path, gene_csv) in resolved_per_phenotype.items():
                                    network = load_network_json(json_path)
                                    candidate_genes = extract_candidate_genes(network)
                                    gene_list, dropped = build_gene_list(candidate_genes, gene_csv, unit=unit_value)
                                    adjacency, edge_list = build_gene_adjacency(
                                        network, gene_list,
                                        include_mediated_edges=include_mediated_value, max_hops=max_hops_value,
                                    )
                                    summary_rows.append((ph, int(gene_list.shape[0]), int(edge_list.shape[0]), list(dropped)))

                                    if st.session_state.get('bio_save_preview_multi', False):
                                        out_dir = st.session_state.get('bio_preview_out_dir', '').strip()
                                        if out_dir:
                                            os.makedirs(out_dir, exist_ok=True)
                                            safe_ph = re.sub(r'[^A-Za-z0-9_.-]+', '_', ph)
                                            save_gene_list_csv(gene_list, os.path.join(out_dir, f'gene_list_{safe_ph}.csv'))
                                            save_adjacency(adjacency, edge_list,
                                                           edge_list_csv_path=os.path.join(out_dir, f'gene_adjacency_{safe_ph}.csv'))

                                    per_phenotype_paths[ph] = (json_path, gene_csv)

                                st.session_state['bio_per_phenotype_paths'] = per_phenotype_paths
                                st.session_state['bio_per_phenotype_summary'] = summary_rows

                                st.success(
                                    f'Configured {len(per_phenotype_paths)} phenotype(s): '
                                    + ', '.join(f'{ph} ({g} genes, {e} edges)' for ph, g, e, _ in summary_rows)
                                )
                            except Exception as exc:
                                st.error(f'Preview failed: {exc}')
                                st.code(traceback.format_exc(), language='text')

                summary_rows = st.session_state.get('bio_per_phenotype_summary')
                if summary_rows:
                    for ph, g, e, dropped in summary_rows:
                        line = f'**{ph}**: {g} gene(s), {e} edge(s)'
                        if dropped:
                            line += f' \u2014 dropped: {dropped}'
                        st.caption(line)

                render_other_models_marker_source_widget()

            elif repeat_mode:
                st.number_input(
                    'How many independent networks?', min_value=2, max_value=20, value=2, step=1,
                    key='bio_repeat_count',
                    help="Each one becomes its own GAT_biological_prior_knowledge_<N> model."
                )
                n_repeats = int(st.session_state.get('bio_repeat_count', 2))

                st.checkbox(
                    'Configure a separate set of N networks for each phenotype?', value=False,
                    key='bio_repeat_per_phenotype',
                    help=("Off (default): the SAME N networks are applied across every phenotype "
                          "in 'Target phenotype(s)' (Tab 1) - useful for comparing the same "
                          "alternative networks against several traits. On: each phenotype gets "
                          "its own independent set of N networks (e.g. N separate FlashP runs "
                          "*per phenotype*, since a network FlashP builds for one phenotype isn't "
                          "meant for a different one) - GAT_biological_prior_knowledge_<k> still "
                          "shows up as one model per k, but internally uses phenotype k's own "
                          "network for each phenotype, the same way 'One network per phenotype' "
                          "already lets a single network vary by phenotype.")
                )
                repeat_per_phenotype = st.session_state.get('bio_repeat_per_phenotype', False)

                if not repeat_per_phenotype:
                    with st.expander('Run FLASH-P now, N times in a row (optional shortcut)'):
                        st.caption(
                            "Runs FLASH-P N times for the same species/phenotype below, filling in "
                            "network 1, 2, ... N's JSON source automatically - equivalent to opening "
                            "each network below and clicking 'Run FLASH-P now' yourself, just faster. "
                            "You can still edit any of them individually afterwards."
                        )
                        st.text_input(
                            'Phenotype (for FLASH-P)', value='Shoot Branching', key='bio_repeat_flashp_phenotype'
                        )
                        st.text_input('FLASH-P checkout path', value='./FlashP', key='bio_repeat_flashp_dir')
                        st.selectbox(
                            'Pipeline variant',
                            options=['Flash-P_Plant', 'Flash-P_Medical', 'Flash-P_Animal'],
                            key='bio_repeat_flashp_variant',
                            help="Same choice as 'Pipeline variant' under a single network's "
                                 "Advanced FLASH-P settings - applied to all N runs here."
                        )
                        st.radio(
                            'Run mode', options=['Light (recommended)', 'Full'],
                            key='bio_repeat_flashp_light_mode', horizontal=True,
                            help="Same choice as 'Run mode' under a single network's Advanced "
                                 "FLASH-P settings - applied to all N runs here."
                        )
                        if st.button(f'Run FLASH-P {n_repeats} times now', key='_btn_bio_repeat_flashp_run'):
                            # Requirement ('view FlashP logs in the GUI'): one running log
                            # covering all N runs, each run's lines prefixed [run k/N] so
                            # they stay distinguishable in a single shared display.
                            repeat_log_path = make_run_log_path(
                                st.session_state.get('result_name', 'FlashP') or 'FlashP', 'flashp_repeat'
                            )
                            with st.expander('FLASH-P log', expanded=True):
                                st.caption(f'Also saved to: {repeat_log_path}')
                                repeat_log_placeholder = st.empty()
                            repeat_log_lines = []
                            base_log_callback = _make_flashp_log_callback(
                                repeat_log_placeholder, repeat_log_lines, repeat_log_path
                            )
                            try:
                                run_results = []
                                with st.spinner(f'Running FLASH-P {n_repeats} times (this can take a while)... see the log above.'):
                                    for k in range(1, n_repeats + 1):
                                        def _prefixed_log_callback(line, _k=k):
                                            base_log_callback('\n'.join(f'[run {_k}/{n_repeats}] {ln}' for ln in line.split('\n')))
                                        resolved_path = run_flash_p(
                                            species=species.strip(),
                                            phenotype=st.session_state.get('bio_repeat_flashp_phenotype', '').strip(),
                                            flashp_dir=st.session_state.get('bio_repeat_flashp_dir', './FlashP').strip(),
                                            pipeline_variant=st.session_state.get('bio_repeat_flashp_variant', 'Flash-P_Plant'),
                                            # A distinct run_name per k is essential here, not
                                            # cosmetic: species+phenotype are identical across
                                            # every iteration of this loop, so run_flash_p()'s own
                                            # default (phenotype-derived only) would give every k
                                            # the SAME Outcome/Local_FlashP_Outcome/<name>/ folder -
                                            # each iteration silently overwriting the previous
                                            # one's full-cost output, and all N "independent"
                                            # network panels ending up showing the same last-
                                            # written network instead of N different ones.
                                            run_name=_default_run_name_for_suffix(f'__rep{k}'),
                                            log_callback=_prefixed_log_callback,
                                            light_mode=(st.session_state.get('bio_repeat_flashp_light_mode', 'Light (recommended)')
                                                        == 'Light (recommended)'),
                                        )
                                        st.session_state[f'bio_network_json_resolved__rep{k}'] = resolved_path
                                        st.session_state[f'bio_json_source__rep{k}'] = 'Upload an existing network.json'
                                        st.session_state[f'bio_network_json_path__rep{k}'] = resolved_path
                                        run_results.append(resolved_path)
                                st.success(f'Ran FLASH-P {n_repeats} times:\n' + '\n'.join(f'- `{p}`' for p in run_results))
                            except Exception as exc:
                                st.error(f'FLASH-P run failed: {exc}')
                                st.code(traceback.format_exc(), language='text')

                    resolved_repeats = {}
                    for k in range(1, n_repeats + 1):
                        with st.expander(f'\U0001f9ec Network {k}', expanded=False):
                            r_json, r_gene_csv = render_bio_prior_network_and_location_section(
                                suffix=f'__rep{k}', species=species, phenotype_hint=None, show_headers=False
                            )
                            resolved_repeats[k] = (r_json, r_gene_csv)
                            if r_json and r_gene_csv:
                                st.caption(f"\u2705 Ready: `{r_json}` + `{r_gene_csv}`")
                            else:
                                st.caption("\u26a0\ufe0f Not fully configured yet.")

                    st.divider()
                    render_data_driven_merge_section()
                    with st.expander('Optional: save preview copies to disk for inspection'):
                        st.checkbox('Also save a preview CSV pair per network', value=False, key='bio_save_preview_repeat')
                        st.text_input(
                            'Preview output folder', value='./Data/_bio_prior_network', key='bio_preview_out_dir_repeat',
                            disabled=not st.session_state.get('bio_save_preview_repeat', False)
                        )

                    if st.button('Preview + use these networks for the model', key='_btn_bio_build_repeat', type='primary'):
                        missing = [k for k, (j, g) in resolved_repeats.items() if not j or not g]
                        if missing:
                            st.error(f'Finish configuring network(s) {missing} first.')
                        else:
                            try:
                                unit_value = st.session_state.get('bio_coordinate_unit', 'bp')
                                include_mediated_value = st.session_state.get('bio_include_mediated_edges', True)
                                max_hops_value = int(st.session_state.get('bio_max_hops', 3))

                                repeat_paths = {}
                                summary_rows = []
                                for k, (json_path, gene_csv) in resolved_repeats.items():
                                    network = load_network_json(json_path)
                                    candidate_genes = extract_candidate_genes(network)
                                    gene_list, dropped = build_gene_list(candidate_genes, gene_csv, unit=unit_value)
                                    adjacency, edge_list = build_gene_adjacency(
                                        network, gene_list,
                                        include_mediated_edges=include_mediated_value, max_hops=max_hops_value,
                                    )
                                    summary_rows.append((k, int(gene_list.shape[0]), int(edge_list.shape[0]), list(dropped)))

                                    if st.session_state.get('bio_save_preview_repeat', False):
                                        out_dir = st.session_state.get('bio_preview_out_dir_repeat', '').strip()
                                        if out_dir:
                                            os.makedirs(out_dir, exist_ok=True)
                                            save_gene_list_csv(gene_list, os.path.join(out_dir, f'gene_list_rep{k}.csv'))
                                            save_adjacency(adjacency, edge_list,
                                                           edge_list_csv_path=os.path.join(out_dir, f'gene_adjacency_rep{k}.csv'))

                                    repeat_paths[k] = (json_path, gene_csv)

                                st.session_state['bio_repeat_paths'] = repeat_paths
                                st.session_state['bio_repeat_summary'] = summary_rows

                                st.success(
                                    f'Configured {len(repeat_paths)} independent network(s): '
                                    + ', '.join(f'#{k} ({g} genes, {e} edges)' for k, g, e, _ in summary_rows)
                                )
                            except Exception as exc:
                                st.error(f'Preview failed: {exc}')
                                st.code(traceback.format_exc(), language='text')

                    summary_rows = st.session_state.get('bio_repeat_summary')
                    if summary_rows:
                        for k, g, e, dropped in summary_rows:
                            line = f'**Network {k}**: {g} gene(s), {e} edge(s)'
                            if dropped:
                                line += f' \u2014 dropped: {dropped}'
                            st.caption(line)

                else:
                    # --- Per-phenotype repeats: each phenotype gets its own N networks ---
                    raw_pheno = st.session_state.get('phenotype_targets', '').strip()
                    if not raw_pheno or raw_pheno.lower() == 'all':
                        st.warning(
                            "This needs explicit phenotype names in Tab 1's 'Target phenotype(s)' "
                            "field (not blank or 'all') so each one can get its own set of N "
                            "networks configured below."
                        )
                        repeat_phenotype_list = []
                    else:
                        repeat_phenotype_list = [p.strip() for p in raw_pheno.split(',') if p.strip()]

                    st.caption(
                        "No N-times-in-a-row FLASH-P shortcut in this mode (it would mean N x "
                        "however many phenotypes you have, run one after another) - open each "
                        "network below and use its own 'Generate with FLASH-P' option instead."
                    )

                    resolved_repeats_pp = {}
                    for ph in repeat_phenotype_list:
                        with st.expander(f'\U0001f9ec {ph}', expanded=False):
                            resolved_repeats_pp[ph] = {}
                            for k in range(1, n_repeats + 1):
                                st.markdown(f'**Network {k}**')
                                r_json, r_gene_csv = render_bio_prior_network_and_location_section(
                                    suffix=f'__ph_{ph}_rep{k}', species=species, phenotype_hint=ph, show_headers=False
                                )
                                resolved_repeats_pp[ph][k] = (r_json, r_gene_csv)
                                if r_json and r_gene_csv:
                                    st.caption(f"\u2705 Ready: `{r_json}` + `{r_gene_csv}`")
                                else:
                                    st.caption("\u26a0\ufe0f Not fully configured yet.")
                                st.divider()

                    render_data_driven_merge_section()
                    with st.expander('Optional: save preview copies to disk for inspection'):
                        st.checkbox('Also save a preview CSV pair per network', value=False, key='bio_save_preview_repeat_pp')
                        st.text_input(
                            'Preview output folder', value='./Data/_bio_prior_network', key='bio_preview_out_dir_repeat_pp',
                            disabled=not st.session_state.get('bio_save_preview_repeat_pp', False)
                        )

                    if st.button('Preview + use these networks for the model', key='_btn_bio_build_repeat_pp', type='primary'):
                        if not repeat_phenotype_list:
                            st.error("No phenotypes to configure - set 'Target phenotype(s)' on Tab 1 first.")
                        else:
                            missing = [
                                (ph, k) for ph, by_k in resolved_repeats_pp.items()
                                for k, (j, g) in by_k.items() if not j or not g
                            ]
                            if missing:
                                st.error(f'Finish configuring: {missing}')
                            else:
                                try:
                                    unit_value = st.session_state.get('bio_coordinate_unit', 'bp')
                                    include_mediated_value = st.session_state.get('bio_include_mediated_edges', True)
                                    max_hops_value = int(st.session_state.get('bio_max_hops', 3))

                                    repeat_paths_pp = {}
                                    summary_rows_pp = []
                                    for ph, by_k in resolved_repeats_pp.items():
                                        repeat_paths_pp[ph] = {}
                                        for k, (json_path, gene_csv) in by_k.items():
                                            network = load_network_json(json_path)
                                            candidate_genes = extract_candidate_genes(network)
                                            gene_list, dropped = build_gene_list(candidate_genes, gene_csv, unit=unit_value)
                                            adjacency, edge_list = build_gene_adjacency(
                                                network, gene_list,
                                                include_mediated_edges=include_mediated_value, max_hops=max_hops_value,
                                            )
                                            summary_rows_pp.append(
                                                (ph, k, int(gene_list.shape[0]), int(edge_list.shape[0]), list(dropped))
                                            )

                                            if st.session_state.get('bio_save_preview_repeat_pp', False):
                                                out_dir = st.session_state.get('bio_preview_out_dir_repeat_pp', '').strip()
                                                if out_dir:
                                                    os.makedirs(out_dir, exist_ok=True)
                                                    safe_ph = re.sub(r'[^A-Za-z0-9_.-]+', '_', ph)
                                                    save_gene_list_csv(gene_list, os.path.join(out_dir, f'gene_list_{safe_ph}_rep{k}.csv'))
                                                    save_adjacency(adjacency, edge_list,
                                                                   edge_list_csv_path=os.path.join(out_dir, f'gene_adjacency_{safe_ph}_rep{k}.csv'))

                                            repeat_paths_pp[ph][k] = (json_path, gene_csv)

                                    st.session_state['bio_repeat_paths_per_phenotype'] = repeat_paths_pp
                                    st.session_state['bio_repeat_summary_pp'] = summary_rows_pp

                                    st.success(
                                        f'Configured {len(repeat_phenotype_list)} phenotype(s) x {n_repeats} network(s) each.'
                                    )
                                except Exception as exc:
                                    st.error(f'Preview failed: {exc}')
                                    st.code(traceback.format_exc(), language='text')

                    summary_rows_pp = st.session_state.get('bio_repeat_summary_pp')
                    if summary_rows_pp:
                        for ph, k, g, e, dropped in summary_rows_pp:
                            line = f'**{ph} / Network {k}**: {g} gene(s), {e} edge(s)'
                            if dropped:
                                line += f' \u2014 dropped: {dropped}'
                            st.caption(line)

                render_other_models_marker_source_widget()

            st.divider()
            if st.button("Copy Circos tab's Marker info path into the model panel", key='_btn_bio_copy_marker_info'):
                marker_info_value = st.session_state.get('marker_info_path', '').strip()
                if not marker_info_value:
                    st.error("The Circos Plot tab's 'Marker info file path' is empty - fill that in first.")
                else:
                    st.session_state[hparam_field_key('GAT_biological_prior_knowledge', 'Marker info CSV path')] = marker_info_value
                    st.success(f'Copied: `{marker_info_value}`')

# ------------------------ Tab 3: Models & hparams ------------------------ #
if show_tab_models:
    with tab_map['models']:
        st.subheader('Select model(s) to run')
        # 'ensemble' itself is chosen on Tab 4 ('4. Ensemble') now, alongside
        # the weighted ensemble methods, rather than here - see Requirement 5.
        _selectable_models = [m for m in AVAILABLE_MODELS if m != 'ensemble']
        # Update ID ver4-5, R2 Stage 11: Tier 2 models (decision D2) are
        # filtered OUT of the selectable checkboxes when their optional
        # package isn't importable in this environment - replaced with an
        # explanatory caption naming the missing package, rather than a
        # checkbox that would only fail later (blueprint §4.6 failure-mode
        # table: "fail fast ... never mid-run after hours of fitting").
        cols = st.columns(3)
        for i, model in enumerate(_selectable_models):
            if model in TIER_2_MODELS and not is_available(model):
                with cols[i % 3]:
                    st.caption(
                        f"\u26a0\ufe0f **{model}** unavailable - install the optional "
                        f"'{optional_dependency_for(model)}' package to enable it "
                        f"(e.g. `pip install {optional_dependency_for(model)}`)."
                    )
                continue
            default_checked = model in ('rrBLUP', 'BayesB', 'RF')
            with cols[i % 3]:
                st.checkbox(model, value=default_checked, key=f'model_selected_{model}',
                            help=MODEL_DESCRIPTIONS.get(model))

        st.divider()
        for model in AVAILABLE_MODELS:
            if model == 'ensemble':
                continue
            if model in TIER_2_MODELS and not is_available(model):
                continue
            if st.session_state.get(f'model_selected_{model}', False):
                render_hparam_panel(model)
                if model == 'GAT_biological_prior_knowledge':
                    st.caption(
                        "\U0001f4a1 The network JSON / gene location CSV are configured entirely "
                        "on the **'Biological Prior Network'** tab (per phenotype, and/or as "
                        "several independent networks) - there's nothing to set for them here."
                    )

        # ver4-4 R3.f (Stage 6, blueprint §2.3.2/§4 config schema table -
        # 'Tab 3 → checkbox'): a single, GLOBAL setting (not per-model,
        # unlike the tuning enable/algorithm checkboxes inside each
        # render_hparam_panel() above) - applies to every model's own
        # hyperparameter search, whenever HP_TUNE is enabled for it.
        #
        # Each of these five GLOBAL settings only does anything for a
        # particular subset of search algorithms (see each one's own help
        # text), so each is now only rendered once at least one currently-
        # selected model+algorithm combination could actually use it -
        # rather than always showing all five regardless of which (if any)
        # tuning algorithms are actually in play. Widgets not currently
        # rendered simply keep whatever value they last had in
        # st.session_state (or their spec default on first run), read back
        # exactly the same way by cfg['HP_TUNE_...'] below - a person never
        # loses a value by an algorithm choice temporarily hiding its widget.
        _active_algos = _active_tuning_algorithms()
        if _active_algos:
            st.divider()
            st.checkbox(
                'Parallelise hyperparameter-search trials (HP_TUNE_PARALLEL_TRIALS)',
                value=True, key='hp_tune_parallel_trials',
                help="When a model's own hyperparameter tuning is enabled, evaluate its trial "
                     "candidates across N_JOBS worker processes instead of one at a time - Grid/"
                     "Random fan out across candidates directly; Bayesian can additionally use "
                     "'Batch suggestion' below to fan out several candidates per round; Nelder-"
                     "Mead/Powell fan out their own independent restarts ('Parallel restarts' "
                     "below). Safe to leave on: if a particular model's own trial-evaluation "
                     "function cannot safely cross a process boundary, this falls back to "
                     "evaluating trials one at a time automatically (with a note in the run log) "
                     "rather than failing the run - it can only help wall-clock time, never change "
                     "which hyperparameters are ultimately selected."
            )
        if {'Nelder-Mead', 'Powell'} & _active_algos:
            st.checkbox(
                'Parallel restarts (HP_TUNE_PARALLEL_RESTARTS)',
                value=True, key='hp_tune_parallel_restarts',
                help="When a model's own tuning uses the Nelder-Mead or Powell search algorithm, "
                     "run its independent multi-start restarts across worker processes instead of "
                     "one after another. Result-preserving either way (each restart is already "
                     "fully independent) - this can only affect wall-clock time."
            )
        # Requirements.md item 4: every Bayesian-tuning-related GLOBAL
        # setting - the original batch-search trio (HP_TUNE_BAYES_BATCH/
        # _BATCH_MAX/_LIAR) AND the further ver4-6 settings that matter for
        # Bayesian search (HP_TUNE_BAYES_DOMAIN_REDUCTION, HP_TUNE_WARM_
        # START, HP_TUNE_SELECTION_MARGIN, HP_TUNE_VALID_REPEATS) - now
        # live together in ONE section, shown only when Bayesian search is
        # actually selected for at least one model. Previously these were
        # split across two separate blocks (one already Bayesian-gated,
        # one shown whenever ANY tuning algorithm was active, Bayesian or
        # not).
        #
        # Requirements.md item 2: 'Tuning scope (HP_TUNE_SCOPE)' used to
        # render here too - it is now hidden entirely (no widget below),
        # simply keeping its own spec default forever (HP_TUNE_SCOPE=
        # 'per_task') via the same st.session_state.get(key, default)
        # fallback every other never-rendered field in this app already
        # relies on (see gather_config() below, and HIDDEN_HPARAM_FIELDS's
        # own docstring for the general pattern this mirrors).
        # ('Repeated-resample validation draws (HP_TUNE_VALID_REPEATS)'
        # was hidden the same way in the same update, then unhidden again
        # and moved into this section per a follow-up request - see the
        # widget below.)
        if 'Bayesian' in _active_algos:
            st.divider()
            st.markdown('**Bayesian hyperparameter-tuning settings**')
            st.checkbox(
                'Batch Bayesian search (HP_TUNE_BAYES_BATCH)',
                value=True, key='hp_tune_bayes_batch',
                help="When a model's own tuning uses the Bayesian search algorithm, suggest and "
                     "evaluate several candidates per round (constant-liar batch mode) instead of "
                     "one at a time, whenever more than one worker process is available. Turn off "
                     "to keep Bayesian search strictly one-candidate-at-a-time regardless of "
                     "N_JOBS."
            )
            st.number_input(
                'Bayesian batch width cap (HP_TUNE_BAYES_BATCH_MAX)',
                min_value=1, max_value=64, value=8, step=1, key='hp_tune_bayes_batch_max',
                help="Upper limit on how many candidates a single batch round (above) ever "
                     "requests at once, even on a very wide node - a very large batch can make "
                     "each round's own guesses less well-informed by earlier results."
            )
            st.selectbox(
                'Batch liar strategy (HP_TUNE_BAYES_LIAR)',
                options=['max', 'mean', 'believer'], index=0, key='hp_tune_bayes_liar',
                help="How a not-yet-evaluated candidate within the same batch round is "
                     "provisionally scored, so the next candidate in that round isn't just a "
                     "repeat of it. 'max' (default, recommended) is the standard, cautious choice."
            )
            st.selectbox(
                'Bayesian domain reduction (HP_TUNE_BAYES_DOMAIN_REDUCTION)',
                options=['auto', 'always', 'never'], index=0, key='hp_tune_bayes_domain_reduction',
                help="Whether the Bayesian search progressively narrows its search box toward "
                     "promising regions. 'auto' (new default) enables this only when the "
                     "model's tunable fields have no drop-down/choice-type dimension (narrowing "
                     "can permanently exclude some choices after a few noisy early results); "
                     "'always' restores the previous (pre-ver4-6) unconditional behaviour; "
                     "'never' disables it outright."
            )
            st.checkbox(
                'Warm-start tuning across replicates (HP_TUNE_WARM_START)',
                value=False, key='hp_tune_warm_start',
                help="Off (new default): every tuned (task, model) search starts from your own "
                     "configured hyperparameters, never a previous replicate's tuned result - this "
                     "fixes a bug where Sequential and Parallel(HPC) runs of the identical config "
                     "could silently produce different numbers. On restores the previous "
                     "(pre-ver4-6) behaviour of carrying a tuned result forward into the next task."
            )
            st.number_input(
                'Tuning selection margin (HP_TUNE_SELECTION_MARGIN)',
                min_value=0.0, max_value=1.0, value=0.02, step=0.01, format='%.3f',
                key='hp_tune_selection_margin',
                help="A tuned search's winning hyperparameters must beat your own configured "
                     "defaults by at least this much (in objective-score units) or the defaults "
                     "are kept instead - guards the 'never worse than untuned' guarantee against "
                     "comparing one noisy default score against the best of many search attempts. "
                     "0.02 (new default) is a modest margin; 0.0 restores the previous (pre-ver4-6) "
                     "bare comparison."
            )
            st.number_input(
                'Repeated-resample validation draws (HP_TUNE_VALID_REPEATS)',
                min_value=1, max_value=10, value=1, step=1, key='hp_tune_valid_repeats',
                help="When greater than 1, each tuning candidate is scored as the average over "
                     "this many independent inner train/validation resamples of the CURRENT "
                     "task's own training data (never touching its real validation/test split), "
                     "reducing the score's own sampling noise at a proportional cost in extra "
                     "model fits. 1 (default) reproduces the previous single-score behaviour."
            )
        st.divider()
        st.checkbox(
            'Fail fast on missing optional model dependencies (MODEL_AVAILABILITY_STRICT)',
            value=True, key='model_availability_strict',
            help="If a selected model (e.g. XGBoost, EBM) needs an optional package that isn't "
                 "installed, stop immediately with an install instruction rather than partway "
                 "through a run. The checkboxes above already hide unavailable Tier 2 models, so "
                 "this mainly matters for a saved or hand-edited config. Turning this off lets "
                 "the run start anyway - the affected model will still fail the first time it's "
                 "actually dispatched."
        )


# ------------------------------ Tab 4: Ensemble ------------------------------ #
with tab_map['ensemble']:
    st.caption(
        "Choose one or more ways to combine the selected models' predictions into an "
        "ensemble - you can select several at once (e.g. the naive ensemble alongside "
        "one or more weight-optimisation methods) to compare their performance. "
        "Selecting any weight-optimisation method here, and/or enabling hyperparameter "
        "tuning for a model (Tab 3), means the Split ratio(s) on Tab 1 must be given as "
        "(train, validation, test) tuples, e.g. (0.8, 0.1, 0.1)."
    )

    st.checkbox(
        'Naive ensemble (equal weight)', value=True, key='model_selected_ensemble',
        help=MODEL_DESCRIPTIONS.get('ensemble'),
    )
    if not show_tab_models:
        st.caption(
            "Only affects model *fitting* (Sequential or Parallel/Step 1) - Step 2 only "
            "assembles and plots whatever was already fitted, so this has no effect here. "
            "Shown for consistency with the weight-optimisation methods below."
        )
    st.divider()

    st.subheader('Weight-optimisation ensemble(s)')
    cols = st.columns(len(W_OPT_METHODS))
    for i, method in enumerate(W_OPT_METHODS):
        with cols[i]:
            st.checkbox(method, value=False, key=f'wopt_selected_{method}',
                        help=W_OPT_METHOD_DESCRIPTIONS.get(method))

    # Update ID ver4-6 (blueprint §4/§10 Stage 6-8) - GLOBAL
    # weight-optimisation setting, shared by every method above (not a
    # per-method HYPERPARAMETERS_OPT field - see pipeline_utils.
    # resolve_compute_resources()'s own docstring). Only rendered once at
    # least one weighted method is actually selected. Two siblings
    # (W_OPT_ANALYTIC_SEED, W_OPT_VALIDATION_FLOOR) used to render here
    # too - both are now hidden (Requirements.md items 5 & 6; see
    # gather_config()'s own now-False fallback defaults for each).
    if any(st.session_state.get(f'wopt_selected_{m}', False) for m in W_OPT_METHODS):
        st.divider()
        st.checkbox(
            'Search the weight simplex directly (W_OPT_SIMPLEX_SEARCH)',
            value=True, key='w_opt_simplex_search',
            help="Nelder Mead and Bayesian optimisation both search directly over valid weight "
                 "combinations (non-negative, summing to 1) instead of an unconstrained box - "
                 "more efficient, since the unconstrained box has an entire dimension that "
                 "cannot affect the result (only the RATIO between weights ever matters). "
                 "'Minimum/Maximum boundary' on each method's own settings below become a "
                 "post-search clip on the resulting shares rather than literal per-weight "
                 "bounds. On by default (new default); off restores the previous (pre-ver4-6) "
                 "unconstrained box search, where negative weights are reachable if 'Minimum "
                 "boundary' is set below 0."
        )

    st.divider()
    for method in W_OPT_METHODS:
        if st.session_state.get(f'wopt_selected_{method}', False):
            render_wopt_panel(method)

    # Requirement: only offer the per-method/across-methods choice when it's
    # actually ambiguous - i.e. at least one selected model has more than
    # one hyperparameter-tuning algorithm turned on (Tab 3), AND ensembling
    # is actually in play (naive 'ensemble' selected, and/or a weighted
    # method selected here).
    _n_multi_tuned = sum(
        1 for m in AVAILABLE_MODELS
        if m != 'ensemble' and st.session_state.get(f'model_selected_{m}', False)
        and st.session_state.get(f'hp_tune_{m}_enabled', False)
        and len(st.session_state.get(f'hp_tune_{m}_algorithms', [])) > 1
    )
    _ensembling_in_play = st.session_state.get('model_selected_ensemble', False) or any(
        st.session_state.get(f'wopt_selected_{m}', False) for m in W_OPT_METHODS
    )
    if _n_multi_tuned > 0 and _ensembling_in_play:
        st.divider()
        st.radio(
            'More than one hyperparameter-tuning algorithm is in use for at least one model - '
            'when ensembling, combine:',
            options=['per_method', 'across_methods'],
            format_func=lambda v: {
                'per_method': ("Per tuning method - one ensemble per algorithm "
                                "(e.g. 'ensemble__Grid', 'ensemble__Bayesian')"),
                'across_methods': ("Across all tuning methods - one combined ensemble "
                                    "using every tuned variant together"),
            }[v],
            key='hp_tune_ensemble_mode',
            help=("Applies to both the naive (equal-weight) ensemble and any weighted method "
                  "selected above. See hyperparameter.csv for which algorithm produced each "
                  "tuned model variant."),
        )

# ------------------------ Tab 5: Violin & Scatter Plots ------------------------ #
if show_tab_plots:
    with tab_map['scatter']:
        st.subheader('Violin plots')
        st.checkbox('Create violin plots (metric comparison across models)?', value=True, key='metric_plot_create',
                    help=("Draws a violin plot comparing Pearson correlation and MSE across every "
                          "selected model (and, if configured, ensemble weighting method)."))
        # Requirement: give the violin plots the same font/figure-size
        # adjustment the scatter plot matrix already has below, for
        # consistency between the two.
        st.number_input('Font size', min_value=1, value=1, step=1, key='metric_font',
                         help="Text size used for labels in the violin plots (passed to seaborn's font_scale).")
        st.number_input('Figure size', min_value=1, value=5, step=1, key='metric_fig',
                         help="Height of each panel in the violin plots, in inches.")
        # R7.h (Stage 3) - per-section image quality. Own field, independent
        # of the scatter-plot-matrix and circos DPI fields below/on Tab 6 -
        # see gather_config()'s cfg['METRIC_PLOT_DPI'] comment for why these
        # are three separate knobs rather than one shared one.
        st.number_input(
            'Plot quality (DPI)', min_value=72, max_value=1200,
            value=int(st.session_state.get('metric_plot_dpi', 300) or 300), step=50,
            key='metric_plot_dpi',
            help=("Resolution, in dots per inch, used when saving the violin plot PNGs "
                  "(Pearson correlation.png, MSE.png, and their '_total' variants). Higher "
                  "values give crisper images for print/publication at the cost of larger "
                  "files and slower saving. 300 is a reasonable default; 600+ is print "
                  "quality. Only affects this section's own plots.")
        )

        # Update ID 3, R3: writes Result/<n>/Metric_summary.xlsx (or, on an
        # openpyxl-free interpreter, three named CSVs) - median/mean
        # Pearson correlation and MSE at three levels of aggregation. Not
        # gated behind the violin plots' own condition - this is its own,
        # independent artefact.
        st.checkbox('Create the metric summary workbook (Metric_summary.xlsx)', value=True,
                    key='metric_summary_create',
                    help=("Writes a workbook whose first sheet ('summary') is a wide table - one "
                          "block for Pearson correlation, one for MSE - with phenotype as rows and "
                          "model as columns, in the SAME order as the violin plots. Three further "
                          "sheets give a median/mean breakdown by phenotype, by "
                          "phenotype+population, and by phenotype+population+ratio. Degrades to "
                          "CSV files (the pivot table as Metric_summary.csv, plus three more) if "
                          "openpyxl isn't installed in this environment (e.g. a stock Linux "
                          "install)."))

        st.divider()

        # Requirement: the Diversity Prediction Theorem-related components
        # (the DPT summary checkbox, the ensemble weight plot checkbox, and
        # that weight plot's own font/figure-size/DPI fields) now live in
        # their own independent section, placed after the violin plots
        # section rather than appended to the end of it.
        st.subheader('Diversity Prediction Theorem')

        # Update ID ver4-9, R5: writes
        # Result/<n>/Diversity_prediction_theorem.xlsx - the four Diversity
        # Prediction Theorem quantities (Page 2018) for
        # every ensemble selected, naive ensemble included as a first-class
        # row. Not gated behind the violin plots' own condition, same as
        # the metric summary checkbox in that section.
        st.checkbox('Create the Diversity Prediction Theorem summary (Diversity_prediction_theorem.xlsx)',
                    value=True, key='dpt_summary_create',
                    help=("Writes a workbook reporting the four Diversity Prediction Theorem terms "
                          "(Page 2018) for every ensemble method "
                          "selected - the naive (equal-weight) ensemble is reported as a first-class "
                          "row, not a special case. Computed entirely from Prediction_result_test.csv "
                          "and Weight.csv; requires at least one ensemble method (naive and/or "
                          "weighted) with 2 or more contributing models to produce any output. "
                          "Degrades to CSV files if openpyxl isn't installed in this environment."))

        # Update ID ver4-9, R6: writes Result/<n>/Weight.png and
        # Weight_total.png - stacked bar plots of the mean optimised
        # ensemble weight each model received. Unchecked by default
        # EVERYWHERE (gather_config()'s own WEIGHT_PLOT_CREATE comment) -
        # a new plot type, opt-in rather than appearing unannounced.
        st.checkbox('Create ensemble weight plots (Weight.png)', value=False, key='weight_plot_create',
                    help=("Draws a stacked bar chart per population x phenotype - one bar per "
                          "ensemble method (the naive, equal-weight ensemble, if selected, then "
                          "every weighted method actually used) - showing the MEAN weight each "
                          "contributing model received, coloured by model family (conventional "
                          "models in blue, machine-learning models in green). Requires at least "
                          "one ensemble method (naive and/or weighted) with 2 or more contributing "
                          "models to produce any output."))
        st.number_input('Font size', min_value=1, value=1, step=1, key='weight_plot_font',
                         help="Text size used for labels in the weight plots (passed to matplotlib's font.size).")
        st.number_input('Figure size', min_value=1, value=5, step=1, key='weight_plot_fig',
                         help="Height of each panel in the weight plots, in inches.")
        st.number_input(
            'Plot quality (DPI)', min_value=72, max_value=1200,
            value=int(st.session_state.get('weight_plot_dpi', 300) or 300), step=50,
            key='weight_plot_dpi',
            help=("Resolution, in dots per inch, used when saving the weight plot PNGs "
                  "(Weight.png, Weight_total.png). Higher values give crisper images for "
                  "print/publication at the cost of larger files and slower saving. 300 is a "
                  "reasonable default; 600+ is print quality. Only affects this section's own "
                  "plots.")
        )

        st.divider()

        st.subheader('Scatter plot matrix')
        st.checkbox('Create scatter plot matrix?', value=True, key='scatter_create',
                    help=("Draws a grid comparing every pair of selected single models at both predicted phenotype and marker effect levels"))
        # ver4-4 R6: re-enabled (was hidden/commented out in ver4-3 - see
        # the ver4-4 Design Blueprint S2.6.1 for why: exact marker-NAME
        # matching, the only mode this widget offered then, essentially
        # never matches a real genotyped marker to a reported QTL
        # position, since the two are rarely at the exact same base
        # pair). R6 fixes the root cause - coordinate-based window
        # matching, below - rather than only re-exposing the same,
        # largely-inert, legacy behaviour. The field is still read by
        # gather_config() below via st.session_state.get('qtl_path', '')
        # exactly as before; a legacy 2-column QTL file still works
        # exactly as it always did (see scatter_plot.py's own dual-schema
        # detection).
        qtl_path = st.text_input(
            'QTL file path (leave blank for None)', value='', key='qtl_path',
            help=("Optional: a csv file listing QTLs, so nearby markers can be highlighted "
                  "separately from other markers in the scatter plots. Two formats are accepted, "
                  "auto-detected by column count:\n"
                  "- Legacy (2 columns): phenotype, marker - exact marker-name match only "
                  "(ver4-3 behaviour).\n"
                  "- Coordinate (5+ columns): chromosome, start, end, name, phenotype[, colour] "
                  "- markers within the window below are flagged, using the marker info file "
                  "(Tab 6) to map QTL positions to nearby markers by genomic distance."))
        if qtl_path:
            file_status(qtl_path)
            # Requirement (widget-ordering constraint - same pattern as the
            # LD pruning/LD decay 'Suggest' buttons above; see
            # _trigger_suggest_ld_window_r2_cb()'s docstring): apply a
            # pending QTL-window suggestion BEFORE the 'qtl_window'
            # number_input below is drawn in this same run.
            if st.session_state.pop('_pending_suggest_qtl_window', False):
                _qtl_sugg_window, _qtl_sugg_note = _qtl_suggest_window()
                st.session_state['qtl_window'] = _qtl_sugg_window
                st.session_state['_suggest_msg_qtl_window'] = (
                    'success', f"Applied window={_qtl_sugg_window:g} - {_qtl_sugg_note}."
                )

            st.number_input(
                "QTL window (coordinate-mode QTL files only; same units as marker_info.csv's "
                "start/end)",
                min_value=0.0, value=float(st.session_state.get('qtl_window', 0.0) or 0.0),
                step=1000.0, key='qtl_window',
                help=("Every QTL region is widened by this amount on each side before markers "
                      "are matched to it. 0 matches only markers that genuinely, exactly overlap "
                      "the QTL region - the same behaviour a legacy 2-column QTL file always has. "
                      "Has no effect for a legacy 2-column (phenotype, marker) QTL file.")
            )
            st.radio(
                'QTL window mode', options=['all_in_window', 'nearest'],
                index=['all_in_window', 'nearest'].index(
                    st.session_state.get('qtl_window_mode') or 'all_in_window'
                ),
                key='qtl_window_mode',
                format_func=lambda v: {
                    'all_in_window': 'Flag every marker within the window',
                    'nearest': 'Flag only the single nearest marker',
                }[v],
                help=("'Flag every marker within the window' highlights every marker whose "
                      "position overlaps the (possibly widened) QTL region - useful when several "
                      "markers plausibly tag the same QTL. 'Flag only the single nearest marker' "
                      "highlights just the closest one - this still finds a marker even with a "
                      "window of 0. Has no effect for a legacy 2-column QTL file.")
            )
            # Requirements.md item 2: hidden for now (kept in place,
            # not deleted, so it's a one-line flip to re-enable later -
            # the pending-suggestion plumbing above (the
            # '_pending_suggest_qtl_window' check before this
            # number_input) and _trigger_suggest_qtl_window_cb() are
            # untouched, they just never get triggered while this is
            # False).
            _SHOW_QTL_WINDOW_SUGGEST_BUTTON = False
            if _SHOW_QTL_WINDOW_SUGGEST_BUTTON:
                st.button(
                    'Suggest QTL window', key='_btn_suggest_qtl_window',
                    on_click=_trigger_suggest_qtl_window_cb,
                    help=("Prefers the LD-decay distance already computed for this exact "
                          "RESULT_NAME (Tab 2's LD decay plot - the distance at which averaged r\u00b2 "
                          "first drops below 0.2), falling back to 25x median marker spacing when no "
                          "LD decay data exists yet for this RESULT_NAME. See the message below once "
                          "clicked for exactly which one was used.")
                )
                _show_suggest_message('_suggest_msg_qtl_window')
        st.number_input('Font size', min_value=1, value=2, step=1, key='scatter_font',
                         help="Text size used for axis labels in the scatter plot matrix.")
        st.number_input('Figure size', min_value=1, value=30, step=1, key='scatter_fig',
                         help="Overall size of the scatter plot matrix image, in inches.")
        # R7.h (Stage 3) - per-section image quality, independent of the
        # violin-plot and circos DPI fields above/on Tab 6.
        st.number_input(
            'Plot quality (DPI)', min_value=72, max_value=1200,
            value=int(st.session_state.get('scatter_plot_dpi', 300) or 300), step=50,
            key='scatter_plot_dpi',
            help=("Resolution, in dots per inch, used when saving the scatter-plot-matrix "
                  "PNGs (one per phenotype) and their shared legend PNG. Higher values give "
                  "crisper images for print/publication at the cost of larger files and "
                  "slower saving. 300 is a reasonable default; 600+ is print quality. Only "
                  "affects this section's own plots.")
        )

    # ------------------------------ Tab 6: Circos ------------------------------ #
    with tab_map['circos']:
        st.checkbox('Create circos plot?', value=True, key='circos_create',
                    help=("Draws the circular genome plot showing marker effects and "
                          "model-interaction/attention links. The settings below are only used "
                          "if this is checked."))

        chrom_path = st.text_input('Chromosome info file path', value='./Data/MaizeNAM/chrom.csv', key='chrom_info_path',
                                    help="A file giving each chromosome's name and length, used to draw the outer ring.")
        file_status(chrom_path)

        marker_path = st.text_input('Marker info file path', value='./Data/MaizeNAM/marker_info.csv', key='marker_info_path',
                                     help="A file mapping each marker in your genotype data to its chromosome and position, so it can be placed correctly on the plot.")
        file_status(marker_path)

        gene_path = st.text_input(
            'Gene info file path (leave blank for None)', value='./Data/MaizeNAM/gene_info.csv', key='gene_info_path',
            help="Optional: a file of known gene locations to annotate on the plot alongside the markers."
        )
        if gene_path:
            file_status(gene_path)

        st.checkbox(
            "Chromosome/gene lengths are the same for every population (and 'all')",
            value=True, key='circos_broadcast_population',
            help=(
                "If every population (and the combined 'all' view circos always "
                "generates in addition) shares the exact same chromosome lengths and gene "
                "locations, check this and provide the 'Chromosome info file path'/'Gene info "
                "file path' files WITHOUT a 'population' column (or with one whose values don't "
                "matter) - just one set of rows. EasiGP duplicates them across every population "
                "actually needed internally, so you don't have to build the file with the same "
                "rows repeated once per population yourself. Leave unchecked if different "
                "populations genuinely have different chromosome/gene layouts - you'll then "
                "need to include every population's own rows (plus an 'all' block) in the files "
                "yourself, exactly as before."
            )
        )
        st.caption("Unchecked (default) keeps the original behaviour - the files must already "
                   "contain one full set of rows per population, plus 'all'.")

        # Requirement 4: 'Top interaction percentage' comes first, right
        # before 'Space between rings' - the one field on this whole tab
        # that still needs a semi-manual step (an estimate typed into a
        # small form), while every field after it now fills itself in
        # automatically - keeping it up front means everything below is
        # what actually behaves 'automatically', with the one exception
        # to that clearly first. Stays a manually-triggered button
        # (never auto-fills on every rerun like the fields below it) - it
        # needs a number only the person can reasonably estimate (final
        # marker count after pruning), collected via a small pop-up form
        # rather than read from any file (Interaction.csv could be large
        # enough to cause real memory/performance problems to read just
        # for this).
        # Requirement 4 (Requirements.md item 4): choose whether the
        # strongest interactions to draw as links are selected by a
        # percentage of candidate marker pairs (the original behaviour,
        # still the default - every config that predates this feature has
        # no 'circos_topinteraction_mode' key at all, and gather_config()
        # below defaults it to 'percentage', reproducing exactly today's
        # behaviour) or by a fixed absolute number (M) of the strongest
        # pairs instead.
        st.radio(
            'Select top interactions by', options=['percentage', 'count'],
            index=['percentage', 'count'].index(st.session_state.get('circos_topinteraction_mode') or 'percentage'),
            key='circos_topinteraction_mode', horizontal=True,
            format_func=lambda v: {'percentage': 'Percentage', 'count': 'Number (M)'}[v],
            help=("'Percentage' keeps the top N% strongest marker-pair interactions (the "
                  "original behaviour). 'Number (M)' instead keeps exactly the M strongest "
                  "interactions, regardless of how many candidate pairs exist.")
        )
        if st.session_state.get('circos_topinteraction_mode', 'percentage') == 'percentage':
            _ci1, _ci2 = st.columns([3, 1])
            with _ci1:
                st.number_input('Top interaction percentage to display', value=0.001, format='%.4f', key='circos_topinteraction',
                                 help=CIRCOS_HELP['interaction_top'])
            with _ci2:
                st.write("")
                st.write("")
                if st.button('Suggest', key='_btn_suggest_interaction_top', help=(
                    "Asks for your estimate of the final marker count (after pruning), then combines "
                    "that with RF's own interaction-search settings - never reads Interaction.csv (it "
                    "can be too large to read quickly)."
                )):
                    _render_interaction_top_estimate_form()
            _show_suggest_message('_suggest_msg_interaction_top')
            st.caption("Lower this if the plot looks too cluttered with links; raise it to surface more interactions.")
        else:
            st.number_input(
                'Top number of interactions to display (M)', min_value=1, value=75, step=1,
                key='circos_topinteraction_count', help=CIRCOS_HELP['interaction_top_count'])
            st.caption("Keeps exactly this many of the strongest marker-pair interactions (fewer if fewer exist).")

        # Requirement 5: 'Space between rings', 'Start angle', 'End
        # angle' auto-fill from a chromosome count (cheap, no genotype/
        # marker/interaction data read at all).
        _autofill_number_field('circos_space', _circos_suggest_space)
        st.number_input('Space between rings', value=None, key='circos_space', placeholder='auto',
                         help=CIRCOS_HELP['space'])
        st.caption("Auto-filled from the number of chromosomes once a 'Chromosome info file' is set "
                   "above; type your own value to override.")

        def _suggest_start():
            _s, _e, _n = _circos_suggest_start_end_angle()
            return _s, _n
        def _suggest_end():
            _s, _e, _n = _circos_suggest_start_end_angle()
            return _e, _n
        _autofill_number_field('circos_start', _suggest_start)
        _autofill_number_field('circos_end', _suggest_end)
        st.number_input('Start angle', value=None, key='circos_start', placeholder='auto', help=CIRCOS_HELP['start'])
        st.caption("0° starts at the top of the circle and increases clockwise.")
        st.number_input('End angle', value=None, key='circos_end', placeholder='auto', help=CIRCOS_HELP['end'])
        st.caption("Auto-filled to leave a seam gap sized for the number of chromosomes; type your own "
                   "values to override either field.")

        _autofill_number_field('circos_link_alpha_min', _circos_suggest_link_alpha_min)
        st.number_input('Minimum link opacity', value=None, min_value=0.0, max_value=1.0, step=0.05,
                         key='circos_link_alpha_min', placeholder='auto', help=CIRCOS_HELP['link_alpha_min'])
        st.caption("The strongest link is always fully opaque; this sets how faint the weakest one gets. "
                   "Auto-filled from the current 'Top interaction percentage' value above.")

        # Requirement 3: label size / scale / window / end_adjust /
        # gene_adjust all auto-fill directly now (no more Suggest
        # buttons) - each field starts blank and fills itself in as soon
        # as its own required file(s) are set, staying in sync with
        # every later change unless manually overridden (see
        # _autofill_number_field's own docstring).
        _autofill_number_field('circos_labelsize', _circos_suggest_label_size)
        st.number_input('Label font size', value=None, key='circos_labelsize', placeholder='auto',
                         help=CIRCOS_HELP['label_size'])
        st.caption("Auto-filled from the number of chromosomes once a 'Chromosome info file' is set "
                   "above; type your own value to override. 6-8 is usually readable.")

        # Update ID ver4-6, R2 (blueprint §5.1): separate from the
        # chromosome-name 'Label font size' above - this is the font size
        # for the labels drawn ON each ring itself (model names/gene
        # sources), which is what R2 actually fixes overlap for.
        _autofill_number_field('circos_ring_label_size', _circos_suggest_ring_label_size)
        st.number_input('Ring label font size', value=None, key='circos_ring_label_size', placeholder='auto',
                         help=CIRCOS_HELP['ring_label_size'])
        st.caption("Auto-filled once at least one model is selected; type your own value to override. "
                   "8pt is this tool's traditional default.")

        st.selectbox(
            'Chromosome coordinate unit', options=['bp', 'cM'], index=0, key='circos_unit',
            help="What the chromosome/marker/gene position numbers in your input files are measured "
                 "in - base pairs (physical distance) or centimorgans (genetic map distance). "
        )
        st.caption("For 'bp', the plot automatically labels itself in bp/kb/Mb/Gb depending on 'Scale' "
                   "below - whichever reads most naturally.")

        _autofill_number_field('circos_scale', _circos_suggest_circos_scale)
        st.number_input('Scale', value=None, key='circos_scale', placeholder='auto', help=CIRCOS_HELP['scale'])
        st.caption("Auto-filled from the longest chromosome's length once a 'Chromosome info file' is "
                   "set above; type your own value to override.")

        _autofill_number_field('window_size', _circos_suggest_window_size)
        st.number_input('Averaging window size (WINDOW)', value=None, key='window_size', placeholder='auto',
                         help=CIRCOS_HELP['window'])
        st.caption("Auto-filled from typical marker spacing once 'Marker info file'/'Chromosome info "
                   "file' are set above (or from chromosome length alone if only the latter is set); "
                   "type your own value to override. 0 disables windowing and uses each marker's raw "
                   "effect directly instead of an average.")

        # Requirement (moved out of 'Advanced settings', per explicit
        # request - this field is important enough that it shouldn't be
        # hidden behind a collapsed expander like the other, more
        # rarely-touched advanced fields below it are): sits directly
        # after 'Averaging window size (WINDOW)' now, in the main,
        # always-visible area of the tab.
        _autofill_number_field('gene_adjust', _circos_suggest_gene_adjust)
        st.number_input('Gene location adjustment (GENE_ADJUST)', value=None, key='gene_adjust', placeholder='auto',
                         help=CIRCOS_HELP['gene_adjust'])
        st.caption("Auto-filled once 'Marker info file'/'Chromosome file'/' Gene info file' are set above, to "
                   "widen key genomic marker regions just enough to stay visible; type your own value to "
                   "override. Extends the start and end location of each marker for clearer "
                   "visualisation - a larger value may not represent the true location of each "
                   "marker.")

        # R7.h (Stage 3) - per-section image quality, independent of the
        # violin-plot and scatter-plot-matrix DPI fields on Tab 5 (see
        # gather_config()'s cfg['CIRCOS_PLOT_DPI'] comment). Circos is the
        # primary deliverable (architecture doc S1), so kept in the main,
        # always-visible area rather than tucked inside 'Advanced settings'
        # below.
        st.number_input(
            'Plot quality (DPI)', min_value=72, max_value=1200,
            value=int(st.session_state.get('circos_plot_dpi', 300) or 300), step=50,
            key='circos_plot_dpi',
            help=("Resolution, in dots per inch, used when saving every circos plot PNG "
                  "(the marker-effect rings, each interaction/attention overlay, and their "
                  "separate legend PNGs). Higher values give crisper images for print/"
                  "publication at the cost of larger files and slower saving. 300 is a "
                  "reasonable default; 600+ is print quality. Only affects this section's "
                  "own plots.")
        )

        with st.expander('Advanced settings'):
            _autofill_number_field('end_adjust', _circos_suggest_end_adjust)
            st.number_input('Edge location adjustment (END_ADJUST)', value=None, key='end_adjust', placeholder='auto',
                             help=CIRCOS_HELP['end_adjust'])
            st.caption("Auto-filled once 'Marker info file'/'Chromosome info file' are set above, to "
                       "widen marker regions just enough to stay visible; type your own value to "
                       "override. Extends the start and end location of each marker for clearer "
                       "visualisation - a larger value may not represent the true location of each "
                       "marker.")

            st.selectbox('Marker effect ordering (ASCENDING)', options=['True', 'False', 'None'], index=2, key='ascending',
                         help=CIRCOS_HELP['ascending'])
            st.caption("Leave as 'None' unless you specifically want overlapping marker regions resolved by effect strength.")

            # Update ID ver4-6, R2 (blueprint §5.1, §3.4 S4): five further
            # advanced fields, all read via `circos_config.get(key,
            # <legacy default>)` in circos_plot.py, so leaving every one
            # of these untouched reproduces a pre-ver4-6 render exactly
            # (I11, AC2.5).
            st.markdown('**Ring label sizing** (fixes ring labels overlapping each other on plots with many models/gene sources)')
            st.selectbox(
                'Ring label fit', options=['off', 'shrink', 'shrink_then_truncate'],
                index=0, key='circos_ring_label_fit', help=CIRCOS_HELP['ring_label_fit'],
            )
            st.caption("'off' matches every version of this tool before this option existed. "
                       "'shrink_then_truncate' (recommended) is what actually guarantees no ring "
                       "label ever overlaps another, however many rings are drawn.")

            st.selectbox(
                'Ring layout', options=['legacy', 'fit'], index=1, key='circos_ring_layout',
                help=CIRCOS_HELP['ring_layout'],
            )
            st.caption("'legacy' matches this tool's traditional ring stacking, but runs out of room "
                       "beyond ~33 rings. Switch to 'fit' for a many-model/many-gene-source comparison.")

            st.number_input(
                'Figure size (inches)', value=None, min_value=4.0, max_value=40.0, step=0.5,
                key='circos_figsize', placeholder='auto (8.0)', help=CIRCOS_HELP['figsize'],
            )
            st.caption("Leave blank for this tool's traditional 8-inch figure. A larger figure gives "
                       "ring/tick labels more physical room to be readable.")

            st.number_input(
                'Maximum auto-suggested seam gap (degrees)', value=None, min_value=8.0, max_value=180.0,
                step=1.0, key='circos_seam_gap_max', placeholder='auto (40.0)', help=CIRCOS_HELP['seam_gap_max'],
            )
            st.caption("Caps how wide 'Start angle'/'End angle' above will be auto-suggested, however "
                       "large a ring label would otherwise need. A label that still doesn't fit within "
                       "this cap is shrunk/truncated at render time instead (see 'Ring label fit' above).")

            st.number_input(
                'Ring label maximum characters (0 = no limit)', value=0, min_value=0, step=1,
                key='circos_ring_label_max_chars', help=CIRCOS_HELP['ring_label_max_chars'],
            )
            st.caption("Rarely needed; the 'Ring label fit' setting above already keeps labels from "
                       "overlapping without a fixed character limit.")

        st.divider()

        if 'cytoband_colormap' not in st.session_state:
            st.session_state['cytoband_colormap'] = dict(DEFAULT_CYTOBAND_COLORMAP)

        # Requirement 9 (additional - new colours must be checked too):
        # a 'Suggest' click can only set st.session_state['cmap_new_color']
        # here, BEFORE that colour_picker widget is drawn further down in
        # this same run - Streamlit forbids doing so afterward (the same
        # rule every other auto-fill/suggestion in this file already
        # works around) - so the button below only sets a lightweight
        # pending flag, and the actual (cheap - see
        # _suggest_colorblind_safe_color's own docstring) suggestion
        # work happens right here instead.
        if st.session_state.pop('_pending_suggest_new_color', False):
            st.session_state['cmap_new_color'] = _suggest_colorblind_safe_color(
                st.session_state['cytoband_colormap']
            )

        with st.expander('Edit cytoband colour map'):
            st.caption('Click a swatch to change that colour, tick "Remove" to drop it, or add a new one below.')
            to_remove = []
            for name in sorted(st.session_state['cytoband_colormap'].keys()):
                c1, c2, c3 = st.columns([3, 2, 1])
                with c1:
                    st.write(name)
                with c2:
                    st.session_state['cytoband_colormap'][name] = st.color_picker(
                        name, value=st.session_state['cytoband_colormap'][name],
                        key=f'cmap_{name}', label_visibility='collapsed',
                        help=f"Colour used to draw the '{name}' cytoband/region on the circos plot."
                    )
                with c3:
                    if st.checkbox('Remove', key=f'cmap_remove_{name}',
                                    help=f"Tick to delete the '{name}' colour entry entirely."):
                        to_remove.append(name)

            if to_remove and st.button('Apply removals', key='_btn_cmap_apply_removals'):
                for name in to_remove:
                    st.session_state['cytoband_colormap'].pop(name, None)
                st.rerun()

            st.markdown('**Add a new colour**')
            new_name = st.text_input('New colour name', key='cmap_new_name',
                                      help="The cytoband/region name this colour will apply to on the circos plot.")
            _cc1, _cc2 = st.columns([3, 1])
            with _cc1:
                new_color_value = st.color_picker('Pick colour', value='#ffffff', key='cmap_new_color',
                                                   help="The colour to draw that cytoband/region in.")
            with _cc2:
                st.write("")
                st.write("")
                st.button(
                    'Suggest', key='_btn_cmap_suggest_color',
                    on_click=_trigger_suggest_new_color_cb,
                    help=(
                        "Requirement 9: picks a colour already well-separated (under simulated colour-"
                        "vision deficiency) from every colour currently in the palette, using the same "
                        "check the default palette itself was designed with - rather than picking one "
                        "by eye and hoping it doesn't end up too close to an existing colour."
                    )
                )
            st.caption(
                "Checked automatically against the existing palette when you click 'Add colour' below "
                "- you'll get a warning (not a block) if it's too close to an existing colour under "
                "simulated colour blindness."
            )
            if st.button('Add colour', key='_btn_cmap_add_colour'):
                if new_name.strip():
                    # Requirement 9 (additional): the SAME worst-case-CVD-
                    # distance check the default palette was designed
                    # with, run live against whatever's actually in the
                    # CURRENT palette (including anything already added
                    # this session) - a warning, not a hard block, since
                    # the person may have a good reason to proceed anyway
                    # (e.g. this specific export is only ever viewed by
                    # people who don't need this consideration) - but
                    # they're never left to find out the hard way later.
                    _is_safe, _worst_name, _worst_dist = _check_new_color_cvd_safety(
                        new_color_value, st.session_state['cytoband_colormap']
                    )
                    if not _is_safe:
                        st.warning(
                            f"This colour is hard to distinguish from '{_worst_name}' "
                            f"({st.session_state['cytoband_colormap'][_worst_name]}) under simulated "
                            f"colour-vision deficiency (distance {_worst_dist:.0f}, well below the "
                            f"~15 the rest of the palette was designed to stay above) - added anyway, "
                            f"but consider using 'Suggest' above, or picking a different colour, if "
                            f"this needs to stay distinguishable for colourblind viewers."
                        )
                    st.session_state['cytoband_colormap'][new_name.strip()] = new_color_value
                    st.rerun()
                else:
                    st.error('Please enter a name for the new colour.')

# --------------------------------------------------------------------------- #
# Run pipeline (Requirements.md item 6: moved into its own tab, below)
# --------------------------------------------------------------------------- #
with tab_map['run_pipeline']:
    # --------------------------------------------------------------------------- #
    # Run pipeline
    # --------------------------------------------------------------------------- #

    #st.divider()
    #st.header('Run pipeline')
    st.subheader('Workflow for this run')

    is_step1 = (mode == 'Parallel' and step == 'Step 1')
    is_step2 = (mode == 'Parallel' and step == 'Step 2')
    is_sequential = (mode == 'Sequential')
    purpose = 'step1' if is_step1 else ('step2' if is_step2 else 'sequential')

    # --------------------------------------------------------------------------- #
    # Patch 3 v4, Requirement 1: choose the workflow FIRST, right under 'Run
    # pipeline', so only the relevant configuration below renders - rather
    # than showing the complete HPC-export machinery (HPC cluster profile,
    # scheduler, resource requests...) to someone who only wants 'Option B:
    # run now (local)', and vice versa hiding Option B's own local
    # 'Batch ID'/'Run now' controls once 'Option A' is chosen.
    # --------------------------------------------------------------------------- #
    _RUN_OPTION_LABELS = {
        'A': 'Option A (recommended for HPC): export config, then submit a job',
        'B': 'Option B: run now (local)',
    }
    st.radio(
        "Choose from the two options below",
        options=list(_RUN_OPTION_LABELS.keys()), format_func=lambda k: _RUN_OPTION_LABELS[k],
        key=f'{purpose}_run_option', horizontal=True,
        help=("Pick ONE workflow - only its own configuration renders below. 'Option A' only "
              "writes a config + submission script for your HPC scheduler (Slurm/PBS) - nothing "
              "runs on this machine. 'Option B' runs the pipeline immediately, right here - "
              "convenient for a quick local test, but not a substitute for Option A once you "
              "have a real-sized job."),
    )
    use_option_a = (st.session_state.get(f'{purpose}_run_option', 'A') == 'A')

    if use_option_a:
        # Requirement 1: "the second thing that will be asked should be 'HPC
        # cluster profile'" - drawn here, immediately after the Option A/B
        # choice, for every purpose (Sequential/Step 1/Step 2). Everything
        # else below (Parallel batch configuration's 'Batch ID source', the
        # 'Suggested compute resources' panel, and render_hpc_export_section()'s
        # own 'Job scheduler') reads this selection rather than drawing its
        # own copy.
        _render_hpc_cluster_profile_selector(purpose)

    # --------------------------------------------------------------------------- #
    # Parallel batch configuration - only shown for Parallel / Step 1.
    #
    # Patch 3 v3, Requirement 1: relocated here, right after 'Run pipeline'
    # (was previously on Tab 1) - and reordered so the 'Suggested job-array
    # size' panel + its new 'Update these values' button come BEFORE 'Batch
    # ID source', "clearly separat[ing] the array-related configuration and
    # other resource-related configuration" (the CPU/memory/GPU advisor
    # further down under 'Option A' > 'HPC job resource requests').
    # --------------------------------------------------------------------------- #
    if is_step1:
        st.divider()
        st.subheader('Parallel batch configuration (PARALLEL)')
        st.caption(
            "All prediction scenarios are split into a number of batches. "
            "Choose how the batch ID should be determined; only 'Manual integer' "
            "requires you to enter anything - for Slurm/PBS it's assigned "
            "automatically and never needs to be typed in here."
        )

        if use_option_a:
            _render_suggested_job_array_size()
            st.divider()
            _render_batch_id_source_selector(purpose)
        else:
            # Patch 3 v4, Requirement 1: "For Option B, 'Batch ID source'
            # should always be 'Manual integers'" - a local test run has no
            # scheduler to read SLURM_ARRAY_TASK_ID/PBS_ARRAY_INDEX from, so
            # the selectbox isn't even drawn; the value is simply forced. Also
            # recorded as its own "last autofill" (the same bookkeeping
            # _apply_autofill_unless_overridden() would set) so that
            # switching BACK to Option A afterwards resumes the normal
            # profile/scheduler-matched suggestion in
            # _render_batch_id_source_selector() rather than leaving 'Manual
            # integer' stuck as if it had been a deliberate manual choice
            # made under Option A itself.
            st.session_state['batch_id_source'] = 'Manual integer'
            st.session_state['_batch_id_source_last_autofill'] = 'Manual integer'
            st.caption(
                "Batch ID source: fixed to **Manual integer** for Option B (local run) - there is "
                "no scheduler here to assign it automatically."
            )
            st.number_input('Batch ID', min_value=0, value=0, step=1, key='batch_id_manual',
                             help="Which batch (0-indexed) this particular run should process.")

        st.number_input('Batch size (number of prediction scenarios per batch)',
                         min_value=1, value=3, step=1, key='batch_size',
                         help=("How many population/phenotype/ratio/replicate combinations each "
                               "array-job task processes. Larger batches mean fewer, longer-running "
                               "tasks; smaller batches mean more, shorter tasks that finish in "
                               "parallel sooner (if your cluster has the capacity to run them "
                               "simultaneously)."))
        if use_option_a:
            st.caption(
                "\U0001f4a1 The batch ID for each individual array-job task is *not* set here - "
                "it's picked up automatically at run time from the scheduler's own environment "
                "variable (`SLURM_ARRAY_TASK_ID` / `PBS_ARRAY_INDEX`) by `run_step1_batch.py`, a "
                "plain script with no GUI. See 'Option A' below."
            )

    if use_option_a:
        st.subheader('Option A (recommended for HPC): export config, then submit a job')
        if is_step1:
            st.caption(
                "Configure everything above **once**, then click the button below to write the "
                "config and submission script straight into this project's folder. Each array "
                "task runs `run_step1_batch.py` directly - a plain script with no Streamlit/GUI - "
                "and picks up its own batch ID automatically from the scheduler. This is the only "
                "workflow that scales to thousands of batches."
            )
            render_hpc_export_section(mode, step, 'step1', 'run_step1_batch.py', 'step1_config.json', include_array=True)
        elif is_step2:
            st.caption(
                "Configure everything above **once**, then click the button below to write the "
                "config and submission script straight into this project's folder. The job runs "
                "`run_step2_assemble.py` directly - a plain script with no Streamlit/GUI - which "
                "assembles all Step 1 batches and produces the metric/scatter/circos plots."
            )
            render_hpc_export_section(mode, step, 'step2', 'run_step2_assemble.py', 'step2_config.json', include_array=False)
        else:
            st.caption(
                "Configure everything above **once**, then click the button below to write the "
                "config and submission script straight into this project's folder. The job runs "
                "`run_sequential.py` directly - a plain script with no Streamlit/GUI - for the "
                "full single-pass pipeline."
            )
            render_hpc_export_section(mode, step, 'sequential', 'run_sequential.py', 'sequential_config.json', include_array=False)
        run_clicked = False
    else:
        st.divider()
        if is_step1:
            st.subheader('Option B: run a single batch now (local test only)')
            st.caption(
                "Uses the batch ID configured above under 'Parallel batch configuration'. "
                "Useful for testing one batch interactively before submitting the full array "
                "job, but not a substitute for Option A when you have many batches."
            )
            run_clicked = st.button('Run single batch now (local test)', key='_btn_run_step1_local')
        elif is_step2:
            st.subheader('Option B: run now (local)')
            st.caption("Assembles all Step 1 batches and generates the plots on this machine, right now.")
            run_clicked = st.button('Run assemble + plots now (local)', key='_btn_run_step2_local')
        else:
            st.subheader('Option B: run now (local)')
            st.caption("Runs the full pipeline on this machine, right now.")
            run_clicked = st.button('Run pipeline now (local)', type='primary', key='_btn_run_sequential_local')

    if run_clicked:
        try:
            cfg = gather_config(mode, step)
        except ValueError as exc:
            st.error(str(exc))
        else:
            result_dir = result_dir_path(cfg['RESULT_NAME'])
            os.makedirs(result_dir, exist_ok=True)

            shows_progress = (
                not (mode == 'Parallel' and step == 'Step 2')
                and 'progress_callback' in inspect.signature(GP).parameters
            ) or (mode == 'Parallel' and step == 'Step 2')
            if shows_progress:
                progress_bar = st.progress(0)
                progress_caption = st.empty()

            # GP() (LD pruning + model fitting) and the post-processing plot
            # phases are tracked on the SAME 0-100% bar, but GP() has no way to
            # know in advance how many plot phases will follow it - so instead of
            # extending GP()'s own total after the fact (which caused a visible
            # backward jump: GP() reports 100% internally, then the total grows
            # and the percentage drops before climbing again), GP()'s progress is
            # rescaled into a fixed share of the bar, and the plot phases fill
            # the remainder. This guarantees the bar only ever moves forward.
            #   Sequential: GP() = 0-85%, plots fill 85-100%
            #   Parallel Step 1: GP() = 0-100% (nothing plotted here - see below)
            #   Parallel Step 2: no GP() call at all; assemble+plots = 0-100%
            if mode == 'Sequential':
                gp_phase_weight = 0.85
            elif mode == 'Parallel' and step == 'Step 1':
                gp_phase_weight = 1.0
            else:
                gp_phase_weight = 0.0

            post_phase_state = {'total_phases': 0, 'completed': 0}

            def _update_progress(completed, total, label=None):
                if not shows_progress:
                    return
                gp_fraction = (completed / total) if total else 0
                overall_fraction = gp_phase_weight * gp_fraction
                progress_bar.progress(min(max(overall_fraction, 0.0), 1.0))
                caption = f'{overall_fraction*100:.0f}%'
                if label:
                    caption += f' - {label}'
                progress_caption.caption(caption)

            def _set_post_phase_count(n):
                """Declare (or revise upward) how many post-processing phases
                will run. Safe to call more than once - e.g. Step 2 doesn't know
                whether the attention-plot phase applies until after
                assemble()/load_assembled() returns, so it's called once with
                the phases known up front, then again with the revised count.
                Only ever increasing the count avoids any backward jump."""
                post_phase_state['total_phases'] = max(post_phase_state['total_phases'], n)

            # Tracks when the previous progress report happened (shared with GP()'s
            # own timing further below via last_report_time), so post-processing
            # phase announcements also show how long the previous step took.
            last_report_time = [time.time()]

            def _advance_progress(label):
                """Tick the bar forward by one post-processing phase (plot
                generation, assemble/load) that has no internal sub-progress of
                its own - called once right before each such phase starts."""
                if not shows_progress:
                    return
                post_phase_state['completed'] += 1
                n = max(post_phase_state['total_phases'], post_phase_state['completed'])
                remaining_weight = 1.0 - gp_phase_weight
                overall_fraction = gp_phase_weight + remaining_weight * (post_phase_state['completed'] / n)
                progress_bar.progress(min(max(overall_fraction, 0.0), 1.0))
                now = time.time()
                elapsed = now - last_report_time[0]
                last_report_time[0] = now
                timestamp = datetime.now().strftime('%H:%M:%S')
                progress_caption.caption(
                    f'{overall_fraction*100:.0f}% - {label} '
                    f'[{timestamp}, previous step took {elapsed:.1f}s]'
                )

            def _log(message):
                """Print a phase-completion message. No need to add a timestamp
                here directly - stdout itself is wrapped in a TimestampedWriter
                below, which timestamps every line uniformly (including this one)."""
                print(message)

            # Only pass progress_callback through to GP() if this installation's
            # genomic_prediction.py actually supports it - avoids
            # "TypeError: GP() got an unexpected keyword argument 'progress_callback'"
            # if the two files are ever out of sync.
            gp_progress_kwargs = (
                {'progress_callback': _update_progress}
                if shows_progress and 'progress_callback' in inspect.signature(GP).parameters
                else {}
            )

            log_buffer = io.StringIO()
            log_label = (
                'sequential_local' if mode == 'Sequential'
                else 'step1_local' if (mode == 'Parallel' and step == 'Step 1')
                else 'step2_local'
            )
            log_file_path = make_run_log_path(cfg['RESULT_NAME'], log_label)
            # Requirement: surfaced to the user (in addition to the pipeline log
            # itself) after a Parallel/Step 2 assemble - see the 'missing batch'
            # st.warning after the try/except below. Initialised here (rather
            # than only inside the Step 2 branch) so it's always defined even
            # if the run fails before reaching that branch, or isn't Step 2 at
            # all.
            missing_batches = None
            incomplete_batches = None
            with st.spinner('Running pipeline... this may take a while.'):
                try:
                    configure_r_environment(cfg.get('R_PATH'), r_max_ppsize=cfg.get('R_MAX_PPSIZE', 500000))
                    init_rpy2_conversion()

                    with open(log_file_path, 'w', encoding='utf-8') as log_file, \
                         contextlib.redirect_stdout(TimestampedWriter(log_buffer, log_file)):

                        if mode == 'Sequential':
                            # ---------------------------------------------- #
                            # Original, single-pass behaviour.
                            # ---------------------------------------------- #
                            (metrics, predicted_result_train, predicted_result_test, effect,
                             interactions, population, phenotype, attention) = GP(
                                cfg['GENOTYPE_FILE_NAME'], cfg['PHENOTYPE_FILE_NAME'], cfg['MODEL'],
                                cfg['PHENOTYPE'], cfg['RATIO'], cfg['ITER_NUM'], cfg['HPARAMETERS'],
                                cfg['R_PATH'], cfg['W_OPT'], cfg['RESULT_NAME'], cfg['HYPERPARAMETERS_OPT'],
                                cfg['SCENARIO'], LD_prune=cfg['LD_PRUNE'], RF_filter=cfg['RF_FILTER'],
                                GENOTYPE_FORMAT=cfg['GENOTYPE_FORMAT'], GENOTYPE_PLINK_PATH=cfg['GENOTYPE_PLINK_PATH'],
                                OTHER_MODELS_MARKER_SOURCE=cfg['OTHER_MODELS_MARKER_SOURCE'],
                                HP_TUNE=cfg.get('HP_TUNE'), HP_TUNE_ENSEMBLE_MODE=cfg.get('HP_TUNE_ENSEMBLE_MODE', 'per_method'),
                                MIN_DATA_POINTS=cfg.get('MIN_DATA_POINTS', 100),
                                # ver4-4 Stage 5 completion: this in-process
                                # Sequential-local call site had the SAME
                                # compute-resource forwarding gap disclosed
                                # since Stage 3 for run_sequential.py itself
                                # (see that script's own identical comment) -
                                # NOT ONE of USE_GPU_SKLEARN/N_JOBS/
                                # PLINK_THREADS/R_BLAS_THREADS/TORCH_DEVICE/
                                # CUDNN_BENCHMARK/USE_AMP/N_CPU_WORKERS/
                                # N_GPU_SLOTS/GPU_SLOTS_PER_DEVICE, nor any of
                                # Stage 5's own new keys, ever reached GP()
                                # from a local "Run pipeline" click in the GUI,
                                # regardless of what gather_config() wrote into
                                # cfg. Every default below matches GP()'s own
                                # parameter default exactly (I11).
                                USE_GPU_SKLEARN=cfg.get('USE_GPU_SKLEARN', False), N_JOBS=cfg.get('N_JOBS', -1),
                                PLINK_THREADS=cfg.get('PLINK_THREADS', 1), R_BLAS_THREADS=cfg.get('R_BLAS_THREADS'),
                                TORCH_DEVICE=cfg.get('TORCH_DEVICE'), CUDNN_BENCHMARK=cfg.get('CUDNN_BENCHMARK', True),
                                USE_AMP=cfg.get('USE_AMP', False), N_CPU_WORKERS=cfg.get('N_CPU_WORKERS', 1),
                                N_GPU_SLOTS=cfg.get('N_GPU_SLOTS'), GPU_SLOTS_PER_DEVICE=cfg.get('GPU_SLOTS_PER_DEVICE', 1),
                                R_BLAS_FOLLOWS_N_JOBS=cfg.get('R_BLAS_FOLLOWS_N_JOBS', True),
                                TORCH_NUM_THREADS=cfg.get('TORCH_NUM_THREADS'),
                                TORCH_DATALOADER_WORKERS=cfg.get('TORCH_DATALOADER_WORKERS', 0),
                                GPU_EVAL_BATCH=cfg.get('GPU_EVAL_BATCH', 32),
                                GPU_LD_R2=cfg.get('GPU_LD_R2', True),
                                GPU_KERNEL_PRECOMPUTE=cfg.get('GPU_KERNEL_PRECOMPUTE', True),
                                # ver4-4 Stage 6: same forwarding gap, same fix,
                                # for this call site's own newest compute-
                                # resource key - see genomic_prediction.py's own
                                # HP_TUNE dispatch site for the disclosed caveat
                                # on what this can currently achieve in
                                # practice. This block deliberately does NOT
                                # gain a SEQUENTIAL_INTRA_BATCH route of its own
                                # - see this update's Change Summary §7 for why
                                # the GUI's local "Run pipeline" button is left
                                # as a disclosed, deliberate scope decision
                                # rather than silently never using intra-batch
                                # parallelism with no explanation.
                                HP_TUNE_PARALLEL_TRIALS=cfg.get('HP_TUNE_PARALLEL_TRIALS', True),
                                # ver4-5 R1 (blueprint §3.6) - same forward-every-new-key
                                # discipline as the line above, same legacy-preserving defaults.
                                HP_TUNE_BAYES_BATCH=cfg.get('HP_TUNE_BAYES_BATCH', True),
                                HP_TUNE_BAYES_BATCH_MAX=cfg.get('HP_TUNE_BAYES_BATCH_MAX', 8),
                                HP_TUNE_BAYES_LIAR=cfg.get('HP_TUNE_BAYES_LIAR', 'max'),
                                HP_TUNE_PARALLEL_RESTARTS=cfg.get('HP_TUNE_PARALLEL_RESTARTS', True),
                                # Update ID ver4-6 (blueprint §4/§10) - eight further keys,
                                # same forward-every-new-key discipline, same legacy-preserving
                                # (or, where §4 documents a deliberate default flip, §4's own
                                # documented new-default) values.
                                HP_TUNE_BAYES_DOMAIN_REDUCTION=cfg.get('HP_TUNE_BAYES_DOMAIN_REDUCTION', 'auto'),
                                HP_TUNE_WARM_START=cfg.get('HP_TUNE_WARM_START', False),
                                HP_TUNE_SELECTION_MARGIN=cfg.get('HP_TUNE_SELECTION_MARGIN', 0.02),
                                HP_TUNE_VALID_REPEATS=cfg.get('HP_TUNE_VALID_REPEATS', 1),
                                HP_TUNE_SCOPE=cfg.get('HP_TUNE_SCOPE', 'per_task'),
                                W_OPT_ANALYTIC_SEED=cfg.get('W_OPT_ANALYTIC_SEED', False),
                                W_OPT_VALIDATION_FLOOR=cfg.get('W_OPT_VALIDATION_FLOOR', False),
                                W_OPT_SIMPLEX_SEARCH=cfg.get('W_OPT_SIMPLEX_SEARCH', True),
                                MODEL_AVAILABILITY_STRICT=cfg.get('MODEL_AVAILABILITY_STRICT', True),
                                # Update ID ver4-9, R7.
                                RESULT_COMPRESSION=cfg.get('RESULT_COMPRESSION', 'gzip'),
                                **gp_progress_kwargs
                            )
                            _log('Genomic prediction finished.')

                            # Update ID 3, R3: same position as every other R3
                            # call site - immediately after the results become
                            # available, before the first plot.
                            #
                            # Requirements.md item 8: model_labels (the SAME
                            # list metric_plot() below passes as its own
                            # 'hue_order', below) is computed HERE, before
                            # write_metric_summary(), and reused there - so
                            # the metric summary's own model column order is
                            # always identical to the violin plots' own model
                            # order, never independently derived.
                            model_labels = cfg['MODEL'] + cfg['W_OPT'] if cfg['W_OPT'] is not None else cfg['MODEL']
                            _t0 = time.time()
                            _summary_paths = write_metric_summary(
                                metrics, cfg['RESULT_NAME'], create=cfg.get('METRIC_SUMMARY_CREATE', True),
                                model_order=model_labels,
                            )
                            if _summary_paths:
                                _log(f"Metric summary written: {', '.join(_summary_paths)} "
                                     f"(took {time.time() - _t0:.1f}s).")

                            # Update ID ver4-9, R5: same position as the metric
                            # summary just above (both are lightweight,
                            # regenerable reporting artefacts derived from
                            # already-written files - neither is tracked as its
                            # own post-GP() progress-bar phase, matching how
                            # write_metric_summary() is already treated here).
                            # GP() itself returns no 'weight' (its own 8-tuple),
                            # so both Prediction_result_test.csv and Weight.csv
                            # are read fresh off disk via load_combined().
                            _t0 = time.time()
                            _prediction_test_for_dpt = load_combined(cfg['RESULT_NAME'], 'result_test')
                            _weight_for_dpt = load_combined(cfg['RESULT_NAME'], 'weight')
                            _dpt_terms = build_dpt_terms(_prediction_test_for_dpt, _weight_for_dpt, metrics)
                            _dpt_paths = write_dpt_summary(
                                _dpt_terms, cfg['RESULT_NAME'], create=cfg.get('DPT_SUMMARY_CREATE', True),
                                # Requirements.md item 1: ensemble labels
                                # only - see diversity_summary.dpt_model_
                                # order()'s own docstring for why the full
                                # model_labels list (as used for the metric
                                # summary/violin plots) would otherwise
                                # reintroduce a permanently-blank column per
                                # single-prediction model.
                                model_order=dpt_model_order(model_labels),
                            )
                            if _dpt_paths:
                                _log(f"Diversity Prediction Theorem summary written: "
                                     f"{', '.join(_dpt_paths)} (took {time.time() - _t0:.1f}s).")

                            # Declare how many plot phases will run so the remaining
                            # 15% of the bar (85-100%) is divided evenly between them.
                            has_attention = 'GAT_fully_connected' in cfg['MODEL'] or 'GAT_prior_knowledge' in cfg['MODEL'] or any(m.startswith('GAT_biological_prior_knowledge') for m in cfg['MODEL'])
                            has_ld_decay = bool(
                                cfg.get('LD_PRUNE') and cfg['LD_PRUNE'].get('decay_plot')
                                and cfg['LD_PRUNE']['decay_plot'].get('enabled')
                            )
                            has_weight_plot = bool(cfg.get('WEIGHT_PLOT_CREATE', False))
                            _set_post_phase_count(
                                (1 if has_attention else 0) + (1 if has_ld_decay else 0)
                                + (1 if cfg['METRIC_PLOT_CREATE'] else 0) + (1 if has_weight_plot else 0)
                                + (1 if cfg['SCATTER_CREATE'] else 0) + (1 if cfg['CIRCOS_CREATE'] else 0)
                            )

                            if has_ld_decay:
                                _advance_progress('Generating average LD decay plots...')
                                _t0 = time.time()
                                _n_ld_decay_plots = average_and_plot_ld_decay(cfg['RESULT_NAME'])
                                _log(f'{_n_ld_decay_plots} average LD decay plot(s) generated '
                                     f'(took {time.time() - _t0:.1f}s).')

                            if has_attention:
                                _advance_progress('Generating attention distribution plots...')
                                _t0 = time.time()
                                attention_distribution(attention, cfg['RESULT_NAME'], 10)
                                _log(f'Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

                            if cfg['METRIC_PLOT_CREATE']:
                                _advance_progress('Generating metric plots...')
                                _t0 = time.time()
                                metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'], cfg['METRIC_PLOT_CONFIG'], PLOT_DPI=cfg['METRIC_PLOT_DPI'])
                                _log(f'Metric plots generated (took {time.time() - _t0:.1f}s).')

                            # Update ID ver4-9, R6: tracked as its own
                            # post-phase (unlike the DPT summary above),
                            # matching the design blueprint's explicit
                            # SS2.3.7 instruction - a genuinely new plot
                            # phase, not a lightweight table.
                            if has_weight_plot:
                                _advance_progress('Generating weight plots...')
                                _t0 = time.time()
                                _members_for_plot = resolve_ensemble_members(
                                    metrics, _prediction_test_for_dpt, _weight_for_dpt,
                                )
                                _naive_for_plot = next(
                                    (info['models'] for info in _members_for_plot.values() if info['is_naive']),
                                    None,
                                )
                                _weight_plot_paths = weight_plot(
                                    _weight_for_dpt, model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'],
                                    cfg['WEIGHT_PLOT_CONFIG'], PLOT_DPI=cfg['WEIGHT_PLOT_DPI'],
                                    naive_models=_naive_for_plot,
                                )
                                if _weight_plot_paths:
                                    _log(f"Weight plot(s) generated: {', '.join(_weight_plot_paths)} "
                                         f"(took {time.time() - _t0:.1f}s).")

                            # ver4-4 R7.g: apply the SAME float32 downcast
                            # batch_reader.ResultSet.effect() offers Parallel
                            # mode's plotting path - Sequential mode has no
                            # ResultSet at all (`effect` is GP()'s own
                            # in-memory accumulator), so this is done directly
                            # here, once, and the SAME downcast frame is reused
                            # by both scatter_plot() and circos_plot() below.
                            # Marker_effect.csv itself (already written to disk
                            # by GP(), long before this point) is completely
                            # unaffected - this only ever touches an in-memory
                            # COPY used for plotting. When PLOT_EFFECT_FLOAT32
                            # is off, `effect_for_plots` is the SAME object as
                            # `effect` (no copy) - byte-identical to this
                            # code path's pre-ver4-4 behaviour.
                            if cfg['PLOT_EFFECT_FLOAT32'] and effect.shape[1] > _EFFECT_METADATA_WIDTH:
                                _marker_cols = effect.columns[_EFFECT_METADATA_WIDTH:]
                                effect_for_plots = effect.copy()
                                effect_for_plots[_marker_cols] = effect_for_plots[_marker_cols].astype('float32')
                            else:
                                effect_for_plots = effect

                            if cfg['SCATTER_CREATE']:
                                _advance_progress('Generating scatter plot matrix...')
                                _t0 = time.time()
                                # ver4-4 R6: cfg['QTL_WINDOW']/cfg['QTL_WINDOW_MODE']
                                # are always populated by gather_config() (Stage 2
                                # of this same update) whenever Tab 5 is shown, so
                                # a direct cfg[...] read is safe here (this block
                                # only ever runs immediately after gather_config()
                                # built cfg fresh in this same session - unlike
                                # run_sequential.py/run_step2_assemble.py, which
                                # may load an OLDER config from disk and use
                                # cfg.get(...) with a legacy-reproducing default
                                # instead).
                                scatter_plot(cfg['MODEL'], phenotype, predicted_result_test, effect_for_plots,
                                             cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'],
                                             MARKER_INFO=cfg['MARKER_INFO'],
                                             QTL_WINDOW=cfg['QTL_WINDOW'],
                                             QTL_WINDOW_MODE=cfg['QTL_WINDOW_MODE'],
                                             PLOT_DPI=cfg['SCATTER_PLOT_DPI'])
                                _log(f'Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

                            if cfg['CIRCOS_CREATE']:
                                _advance_progress('Generating circos plot...')
                                _t0 = time.time()
                                _chrom_info_path, _gene_info_path = cfg['CHROMOSOME_INFO'], cfg['GENE_INFO']
                                # Requirement (diagnostic): logs exactly what
                                # the broadcast step actually sees/does, so a
                                # 'gene ring missing for a real population'
                                # report can be diagnosed directly from the
                                # person's own log output, rather than
                                # guessing blind at what their specific
                                # session/config produced - whether the
                                # checkbox was actually on, what population
                                # list it broadcast across, and whether a
                                # gene info path was even set.
                                _log(f"[circos] Broadcast checkbox: {cfg.get('CIRCOS_BROADCAST_POPULATION')} | "
                                     f"population from results: {list(population)} | "
                                     f"gene info path set: {bool(_gene_info_path)}")
                                if cfg.get('CIRCOS_BROADCAST_POPULATION'):
                                    # Requirement 8: broadcast BEFORE circos_plot()
                                    # ever sees these paths, so its own internals
                                    # (and every downstream call within it) stay
                                    # completely unaware this ever happened - same
                                    # 'each population + all' target list
                                    # circos_plot() itself always uses internally.
                                    # Requirement (bugfix): normalize each population value to its
                                    # clean string form before it becomes the broadcast
                                    # file's own 'population' column - see
                                    # _clean_population_label()'s own docstring (circos_plot.py)
                                    # for why (a float-promoted 1.0 must broadcast under the SAME
                                    # label '1' that circos_plot()'s own now-normalized loop will
                                    # later look for, not '1.0'). For SCENARIO='between', 'population'
                                    # values are combined train->test labels (e.g.
                                    # 'Historical->2014') - split to the test-population half FIRST,
                                    # matching circos_plot()'s own identical split of its POPULATION
                                    # parameter, so the broadcast file's labels and what
                                    # circos_plot()'s loop later looks for are the SAME clean value
                                    # ('2014'), not one arrow-combined and the other not.
                                    _target_pop_source = population
                                    if cfg['SCENARIO'] == 'between':
                                        _target_pop_source = [
                                            p.split('->')[-1] if isinstance(p, str) and '->' in p else p
                                            for p in population
                                        ]
                                    _target_pops = [_clean_population_label(p) for p in _target_pop_source] + ['all']
                                    _chrom_info_path = _broadcast_population_info(_chrom_info_path, _target_pops, 'chrom')
                                    if _gene_info_path:
                                        _gene_info_path = _broadcast_population_info(_gene_info_path, _target_pops, 'gene')
                                    _log(f"[circos] Broadcast target populations: {_target_pops} | "
                                         f"broadcast chrom file: {_chrom_info_path} | "
                                         f"broadcast gene file: {_gene_info_path}")
                                circos_plot(effect_for_plots, interactions, cfg['MARKER_INFO'], _chrom_info_path,
                                            _gene_info_path, population, phenotype, cfg['CIRCOS_CONFIG'],
                                            cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                                            cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'],
                                            gene_adjust=cfg.get('GENE_ADJUST', 0), plot_dpi=cfg['CIRCOS_PLOT_DPI'],
                                            # Performance (multi-CPU circos-plot
                                            # rendering): deliberately NOT wired
                                            # to CIRCOS_PLOT_WORKERS here, unlike
                                            # run_sequential.py/run_step2_
                                            # assemble.py's identical call -
                                            # this is 'Option B: run now
                                            # (local)', executing in-process
                                            # inside the Streamlit app itself.
                                            # circos_plot()'s worker fan-out
                                            # always spawns (never forks - see
                                            # circos_plot.py::_MP_CONTEXT), and
                                            # a `spawn`-started child re-imports
                                            # this process's own __main__ -
                                            # which, for a script launched via
                                            # `streamlit run main_app.py`, is
                                            # Streamlit's own bootstrap, not a
                                            # guarded `if __name__ ==
                                            # "__main__":` block. Hard-coded to
                                            # 1 (today's behaviour) rather than
                                            # risking that interaction for an
                                            # interactive, highly-visible GUI
                                            # path - 'Option A: export config,
                                            # then submit a job' still carries
                                            # CIRCOS_PLOT_WORKERS through to the
                                            # headless runners above correctly.
                                            n_workers=1)
                                _log(f'Circos plot generated (took {time.time() - _t0:.1f}s).')

                        elif mode == 'Parallel' and step == 'Step 1':
                            # ---------------------------------------------- #
                            # Fit the selected models for a single batch of
                            # prediction scenarios. Nothing gets plotted here -
                            # that happens once, in Step 2, after all batches
                            # have completed.
                            # ---------------------------------------------- #
                            parallel = dict(cfg['PARALLEL'])
                            if parallel['batch_id'] is None:
                                # Slurm/PBS was selected as the batch ID source -
                                # resolve it now, from this process's own
                                # environment, only because we're about to run
                                # a single batch locally for testing. (This is
                                # never required just to export a config.)
                                parallel['batch_id'] = resolve_batch_id_from_env()

                            (metrics, predicted_result_train, predicted_result_test, effect,
                             interactions, population, phenotype, attention) = GP(
                                cfg['GENOTYPE_FILE_NAME'], cfg['PHENOTYPE_FILE_NAME'], cfg['MODEL'],
                                cfg['PHENOTYPE'], cfg['RATIO'], cfg['ITER_NUM'], cfg['HPARAMETERS'],
                                cfg['R_PATH'], cfg['W_OPT'], cfg['RESULT_NAME'], cfg['HYPERPARAMETERS_OPT'],
                                cfg['SCENARIO'], parallel, LD_prune=cfg['LD_PRUNE'], RF_filter=cfg['RF_FILTER'],
                                GENOTYPE_FORMAT=cfg['GENOTYPE_FORMAT'], GENOTYPE_PLINK_PATH=cfg['GENOTYPE_PLINK_PATH'],
                                OTHER_MODELS_MARKER_SOURCE=cfg['OTHER_MODELS_MARKER_SOURCE'],
                                HP_TUNE=cfg.get('HP_TUNE'), HP_TUNE_ENSEMBLE_MODE=cfg.get('HP_TUNE_ENSEMBLE_MODE', 'per_method'),
                                MIN_DATA_POINTS=cfg.get('MIN_DATA_POINTS', 100),
                                # ver4-4 Stage 5 completion: this in-process
                                # Step-1-local call site is a SEPARATE, simpler
                                # code path from run_step1_batch.py's own
                                # three-way N_MODEL_WORKERS/N_CPU_WORKERS route
                                # (it always calls GP() directly, with no
                                # intra_batch_parallel/intra_task_parallel
                                # fan-out at all - this is a single local batch
                                # run for testing, per the comment above) - so
                                # it did NOT inherit run_step1_batch.py's own
                                # gp_kwargs forwarding and had the SAME full gap
                                # as the Sequential-local block above. Same full
                                # key set, same per-key defaults (I11), for the
                                # same reasons - see that block's own comment.
                                USE_GPU_SKLEARN=cfg.get('USE_GPU_SKLEARN', False), N_JOBS=cfg.get('N_JOBS', -1),
                                PLINK_THREADS=cfg.get('PLINK_THREADS', 1), R_BLAS_THREADS=cfg.get('R_BLAS_THREADS'),
                                TORCH_DEVICE=cfg.get('TORCH_DEVICE'), CUDNN_BENCHMARK=cfg.get('CUDNN_BENCHMARK', True),
                                USE_AMP=cfg.get('USE_AMP', False), N_CPU_WORKERS=cfg.get('N_CPU_WORKERS', 1),
                                N_GPU_SLOTS=cfg.get('N_GPU_SLOTS'), GPU_SLOTS_PER_DEVICE=cfg.get('GPU_SLOTS_PER_DEVICE', 1),
                                R_BLAS_FOLLOWS_N_JOBS=cfg.get('R_BLAS_FOLLOWS_N_JOBS', True),
                                TORCH_NUM_THREADS=cfg.get('TORCH_NUM_THREADS'),
                                TORCH_DATALOADER_WORKERS=cfg.get('TORCH_DATALOADER_WORKERS', 0),
                                GPU_EVAL_BATCH=cfg.get('GPU_EVAL_BATCH', 32),
                                GPU_LD_R2=cfg.get('GPU_LD_R2', True),
                                GPU_KERNEL_PRECOMPUTE=cfg.get('GPU_KERNEL_PRECOMPUTE', True),
                                # ver4-4 Stage 6: same forwarding gap, same fix,
                                # for this call site's own newest compute-
                                # resource key - see the Sequential-local
                                # block's own identical comment above.
                                HP_TUNE_PARALLEL_TRIALS=cfg.get('HP_TUNE_PARALLEL_TRIALS', True),
                                # ver4-5 R1 (blueprint §3.6) - same forward-every-new-key
                                # discipline as the line above, same legacy-preserving defaults.
                                HP_TUNE_BAYES_BATCH=cfg.get('HP_TUNE_BAYES_BATCH', True),
                                HP_TUNE_BAYES_BATCH_MAX=cfg.get('HP_TUNE_BAYES_BATCH_MAX', 8),
                                HP_TUNE_BAYES_LIAR=cfg.get('HP_TUNE_BAYES_LIAR', 'max'),
                                HP_TUNE_PARALLEL_RESTARTS=cfg.get('HP_TUNE_PARALLEL_RESTARTS', True),
                                # Update ID ver4-6 (blueprint §4/§10) - eight further keys,
                                # same forward-every-new-key discipline, same legacy-preserving
                                # (or, where §4 documents a deliberate default flip, §4's own
                                # documented new-default) values.
                                HP_TUNE_BAYES_DOMAIN_REDUCTION=cfg.get('HP_TUNE_BAYES_DOMAIN_REDUCTION', 'auto'),
                                HP_TUNE_WARM_START=cfg.get('HP_TUNE_WARM_START', False),
                                HP_TUNE_SELECTION_MARGIN=cfg.get('HP_TUNE_SELECTION_MARGIN', 0.02),
                                HP_TUNE_VALID_REPEATS=cfg.get('HP_TUNE_VALID_REPEATS', 1),
                                HP_TUNE_SCOPE=cfg.get('HP_TUNE_SCOPE', 'per_task'),
                                W_OPT_ANALYTIC_SEED=cfg.get('W_OPT_ANALYTIC_SEED', False),
                                W_OPT_VALIDATION_FLOOR=cfg.get('W_OPT_VALIDATION_FLOOR', False),
                                W_OPT_SIMPLEX_SEARCH=cfg.get('W_OPT_SIMPLEX_SEARCH', True),
                                MODEL_AVAILABILITY_STRICT=cfg.get('MODEL_AVAILABILITY_STRICT', True),
                                # Update ID ver4-9, R7.
                                RESULT_COMPRESSION=cfg.get('RESULT_COMPRESSION', 'gzip'),
                                **gp_progress_kwargs
                            )
                            print(f"Genomic prediction finished for batch_id={parallel['batch_id']} "
                                  f"(batch_size={parallel['batch_size']}).")

                        elif mode == 'Parallel' and step == 'Step 2':
                            # ---------------------------------------------- #
                            # Assemble the results from all previously-run
                            # Step 1 batches (or, if requested, reload a
                            # previous assembly instead), then generate the
                            # plots. There's no GP() call here to seed the
                            # progress bar's total, so it's computed directly:
                            # one unit for assemble/load, plus one per plot
                            # phase that will actually run.
                            # ---------------------------------------------- #
                            # Declare the phases known up front (assemble/load, metric?,
                            # scatter?, circos?); attention's applicability isn't known
                            # until after assemble/load returns, so it's added below.
                            _set_post_phase_count(
                                1 + (1 if cfg['METRIC_PLOT_CREATE'] else 0)
                                + (1 if cfg['SCATTER_CREATE'] else 0) + (1 if cfg['CIRCOS_CREATE'] else 0)
                            )

                            # Patch 3, Requirement 1: the single shared entry
                            # point resolving cfg['ASSEMBLE_MODE'] - see
                            # assemble.load_for_step2()'s own docstring for
                            # the full decision table (including the legacy
                            # SKIP_ASSEMBLE-only fallback for any config
                            # written before ASSEMBLE_MODE existed). Exactly
                            # the same function run_step2_assemble.py's own
                            # main() calls, so the headless and in-process
                            # paths can never disagree about what a given
                            # config means.
                            _advance_progress(
                                'Reloading previously assembled results...'
                                if cfg.get('ASSEMBLE_MODE') in ('use_preassembled', 'no_assemble')
                                or (cfg.get('ASSEMBLE_MODE') is None and cfg.get('SKIP_ASSEMBLE'))
                                else 'Assembling results from all batches...'
                            )
                            result, _assemble_message = load_for_step2(cfg['RESULT_NAME'], cfg)
                            missing_batches, incomplete_batches = result.missing_batches, result.incomplete_batches
                            _log(_assemble_message)
                            if missing_batches:
                                _log(f'WARNING: missing batch ID(s) (never produced any output): '
                                     f'{format_batch_id_list(missing_batches)}')
                            if not result.models:
                                raise RuntimeError(
                                    'No usable batch output was found to assemble - see the '
                                    'missing-batch warning above for which batch(es) to re-run.'
                                )
                            # Requirement: make it easy to notice, at a glance, which
                            # batch(es) started but did not finish - logged right after
                            # whichever of the two branches above ran, so it's never
                            # buried further down (see the st.warning surfacing this
                            # again, more prominently, after the run finishes below).
                            if incomplete_batches:
                                _log(f'{len(incomplete_batches)} batch(es) did NOT finish (excluded from the '
                                     f'results above):')
                                for _b in incomplete_batches:
                                    _log(f'  - {describe_incomplete_batch(_b)}')
                                _log(f'  Incomplete batch ID list: '
                                     f'{format_batch_id_list([_b["batch_id"] for _b in incomplete_batches])}')

                            # Update ID 3, R3: the Metric_summary.xlsx workbook
                            # (or its CSV degrade), written immediately once the
                            # results become available and before the first
                            # plot - same position in every one of R3's four
                            # call sites (blueprint §R3.9).
                            #
                            # Requirements.md item 8: model_labels computed
                            # here (moved up from just before metric_plot()
                            # below) and reused there, so the metric summary's
                            # own model column order always matches the violin
                            # plots' own model order exactly.
                            model_labels = (
                                result.models + cfg['W_OPT'] if cfg['W_OPT'] is not None else result.models
                            )
                            _t0 = time.time()
                            _summary_paths = write_metric_summary(
                                result.metric, cfg['RESULT_NAME'], create=cfg.get('METRIC_SUMMARY_CREATE', True),
                                model_order=model_labels,
                            )
                            if _summary_paths:
                                _log(f"Metric summary written: {', '.join(_summary_paths)} "
                                     f"(took {time.time() - _t0:.1f}s).")

                            # Update ID ver4-9, R5: same position as the metric
                            # summary just above, reusing the already-assembled
                            # `result` (source='combined' or 'batches',
                            # whichever load_for_step2() chose above) rather
                            # than re-reading anything from a hardcoded path.
                            _t0 = time.time()
                            _dpt_terms = build_dpt_terms(result.prediction('test'), result.weight(), result.metric)
                            _dpt_paths = write_dpt_summary(
                                _dpt_terms, cfg['RESULT_NAME'], create=cfg.get('DPT_SUMMARY_CREATE', True),
                                # Requirements.md item 1: ensemble labels
                                # only - see diversity_summary.dpt_model_
                                # order()'s own docstring for why the full
                                # model_labels list (as used for the metric
                                # summary/violin plots) would otherwise
                                # reintroduce a permanently-blank column per
                                # single-prediction model.
                                model_order=dpt_model_order(model_labels),
                            )
                            if _dpt_paths:
                                _log(f"Diversity Prediction Theorem summary written: "
                                     f"{', '.join(_dpt_paths)} (took {time.time() - _t0:.1f}s).")

                            has_attention = 'GAT_fully_connected' in result.models or 'GAT_prior_knowledge' in result.models or any(m.startswith('GAT_biological_prior_knowledge') for m in result.models)
                            if has_attention:
                                _set_post_phase_count(post_phase_state['total_phases'] + 1)

                            # Requirement 6, Parallel path: Step 2 never sees Step 1's
                            # LD_PRUNE config (see gather_config()'s is_step2 skip), so
                            # detect whether the feature was used by checking whether
                            # any per-scenario LD decay data actually exists on disk -
                            # each Step 1 batch's own GP() call would have already
                            # written it directly into this same shared Result folder.
                            has_ld_decay = ld_decay_data_exists(cfg['RESULT_NAME'])
                            if has_ld_decay:
                                _set_post_phase_count(post_phase_state['total_phases'] + 1)

                            # Update ID ver4-9, R6: tracked as its own
                            # post-phase, matching the design blueprint's
                            # explicit SS2.3.7 instruction (symmetric with
                            # the Sequential in-process block above).
                            has_weight_plot = bool(cfg.get('WEIGHT_PLOT_CREATE', False))
                            if has_weight_plot:
                                _set_post_phase_count(post_phase_state['total_phases'] + 1)

                            if has_attention:
                                _advance_progress('Generating attention distribution plots...')
                                _t0 = time.time()
                                attention_distribution(result.attention(), cfg['RESULT_NAME'], 10)
                                _log(f'Attention distribution plots generated (took {time.time() - _t0:.1f}s).')
                                # Update ID 3, R1: free the cached attention frame
                                # before the next stage - circos_plot() below
                                # re-reads it fresh if it needs it (R1.4's fix for
                                # allocation A3: the Step-2 peak is one stage's
                                # working set, not the sum of every stage's).
                                result.release('attention')

                            if has_ld_decay:
                                _advance_progress('Generating average LD decay plots...')
                                _t0 = time.time()
                                _n_ld_decay_plots = average_and_plot_ld_decay(cfg['RESULT_NAME'])
                                _log(f'{_n_ld_decay_plots} average LD decay plot(s) generated '
                                     f'(took {time.time() - _t0:.1f}s).')

                            if cfg['METRIC_PLOT_CREATE']:
                                _advance_progress('Generating metric plots...')
                                _t0 = time.time()
                                metric_plot(result.metric.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'], cfg['METRIC_PLOT_CONFIG'], PLOT_DPI=cfg['METRIC_PLOT_DPI'])
                                _log(f'Metric plots generated (took {time.time() - _t0:.1f}s).')

                            if has_weight_plot:
                                _advance_progress('Generating weight plots...')
                                _t0 = time.time()
                                _members_for_plot = resolve_ensemble_members(
                                    result.metric, result.prediction('test'), result.weight(),
                                )
                                _naive_for_plot = next(
                                    (info['models'] for info in _members_for_plot.values() if info['is_naive']),
                                    None,
                                )
                                _weight_plot_paths = weight_plot(
                                    result.weight(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'],
                                    cfg['WEIGHT_PLOT_CONFIG'], PLOT_DPI=cfg['WEIGHT_PLOT_DPI'],
                                    naive_models=_naive_for_plot,
                                )
                                if _weight_plot_paths:
                                    _log(f"Weight plot(s) generated: {', '.join(_weight_plot_paths)} "
                                         f"(took {time.time() - _t0:.1f}s).")

                            if cfg['SCATTER_CREATE']:
                                _advance_progress('Generating scatter plot matrix...')
                                _t0 = time.time()
                                # ver4-4 R6: see the Sequential in-process block's
                                # identical note above - cfg[...] direct read is
                                # safe here (freshly built by gather_config() in
                                # this same session).
                                scatter_plot(result.models, result.phenotype, result.prediction('test'), result.effect(float32=cfg['PLOT_EFFECT_FLOAT32']),
                                             cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'],
                                             MARKER_INFO=cfg['MARKER_INFO'],
                                             QTL_WINDOW=cfg['QTL_WINDOW'],
                                             QTL_WINDOW_MODE=cfg['QTL_WINDOW_MODE'],
                                             PLOT_DPI=cfg['SCATTER_PLOT_DPI'])
                                _log(f'Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')
                                # Update ID 3, R1: the test-prediction frame isn't
                                # needed again; `effect` IS (circos_plot below),
                                # so it's kept resident.
                                result.release('prediction_test')

                            if cfg['CIRCOS_CREATE']:
                                _advance_progress('Generating circos plot...')
                                _t0 = time.time()
                                _chrom_info_path, _gene_info_path = cfg['CHROMOSOME_INFO'], cfg['GENE_INFO']
                                # Requirement (diagnostic): logs exactly what
                                # the broadcast step actually sees/does, so a
                                # 'gene ring missing for a real population'
                                # report can be diagnosed directly from the
                                # person's own log output, rather than
                                # guessing blind at what their specific
                                # session/config produced - whether the
                                # checkbox was actually on, what population
                                # list it broadcast across, and whether a
                                # gene info path was even set.
                                _log(f"[circos] Broadcast checkbox: {cfg.get('CIRCOS_BROADCAST_POPULATION')} | "
                                     f"population from results: {list(result.population)} | "
                                     f"gene info path set: {bool(_gene_info_path)}")
                                if cfg.get('CIRCOS_BROADCAST_POPULATION'):
                                    # Requirement (bugfix): normalize each population value to its
                                    # clean string form before it becomes the broadcast
                                    # file's own 'population' column - see
                                    # _clean_population_label()'s own docstring (circos_plot.py)
                                    # for why (a float-promoted 1.0 must broadcast under the SAME
                                    # label '1' that circos_plot()'s own now-normalized loop will
                                    # later look for, not '1.0'). For SCENARIO='between', 'population'
                                    # values are combined train->test labels (e.g.
                                    # 'Historical->2014') - split to the test-population half FIRST,
                                    # matching circos_plot()'s own identical split of its POPULATION
                                    # parameter, so the broadcast file's labels and what
                                    # circos_plot()'s loop later looks for are the SAME clean value
                                    # ('2014'), not one arrow-combined and the other not.
                                    _target_pop_source = result.population
                                    if cfg['SCENARIO'] == 'between':
                                        _target_pop_source = [
                                            p.split('->')[-1] if isinstance(p, str) and '->' in p else p
                                            for p in result.population
                                        ]
                                    _target_pops = [_clean_population_label(p) for p in _target_pop_source] + ['all']
                                    _chrom_info_path = _broadcast_population_info(_chrom_info_path, _target_pops, 'chrom')
                                    if _gene_info_path:
                                        _gene_info_path = _broadcast_population_info(_gene_info_path, _target_pops, 'gene')
                                    _log(f"[circos] Broadcast target populations: {_target_pops} | "
                                         f"broadcast chrom file: {_chrom_info_path} | "
                                         f"broadcast gene file: {_gene_info_path}")
                                # OOM fix (large Interaction.csv/Attention.csv
                                # files): same rationale as
                                # run_step2_assemble.py's identical call site -
                                # circos_plot()'s interaction()/attention
                                # handling has always reduced these two tables
                                # down to one row per (population, model,
                                # phenotype, marker1, marker2) combination
                                # before doing anything else, so reading the
                                # FULL raw tables here just to immediately
                                # average them away is what could exhaust
                                # memory on a large assembled result opened
                                # in-process from the GUI. See batch_reader.
                                # aggregate_marker_pair_sums()/ResultSet.
                                # interactions_grouped()/attention_grouped()
                                # for why this is the identical final numbers,
                                # not an approximation.
                                circos_plot(result.effect(float32=cfg['PLOT_EFFECT_FLOAT32']), result.interactions_grouped(), cfg['MARKER_INFO'], _chrom_info_path,
                                            _gene_info_path, result.population, result.phenotype, cfg['CIRCOS_CONFIG'],
                                            cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                                            cfg['RESULT_NAME'], result.attention_grouped(), cfg['SCENARIO'], cfg['ASCENDING'],
                                            gene_adjust=cfg.get('GENE_ADJUST', 0), plot_dpi=cfg['CIRCOS_PLOT_DPI'],
                                            # Performance (multi-CPU circos-plot
                                            # rendering): see the Sequential
                                            # in-process call site above for why
                                            # this stays hard-coded to 1 here
                                            # rather than reading
                                            # CIRCOS_PLOT_WORKERS - same
                                            # Streamlit-process/`spawn` concern
                                            # applies identically to this
                                            # in-process 'Option B' Parallel
                                            # Step 2 path.
                                            n_workers=1)
                                _log(f'Circos plot generated (took {time.time() - _t0:.1f}s).')
                                result.release()

                except Exception:
                    tb_text = traceback.format_exc()
                    st.error('Pipeline failed - see the traceback below.')
                    st.code(tb_text, language='text')
                    try:
                        with open(log_file_path, 'a', encoding='utf-8') as log_file:
                            log_file.write(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Pipeline failed:\n')
                            log_file.write(tb_text)
                    except Exception:
                        pass  # best-effort - a logging failure shouldn't mask the real error

                    # Requirement: make it easy to notice that partial progress
                    # was saved and exactly how to resume - checked generically
                    # (rather than branching on which mode was running) since
                    # only the relevant checkpoint file(s) will actually exist
                    # for whatever just failed (Sequential: a single
                    # '.checkpoint.json'; Parallel/Step 1 run locally: that
                    # batch's own '.checkpoint_<id>.json').
                    _seq_status = sequential_run_status(cfg['RESULT_NAME'])
                    _incomplete = find_incomplete_batches(cfg['RESULT_NAME'])
                    if _seq_status is not None:
                        st.warning(
                            f"Partial progress was saved: {_seq_status['completed']}/"
                            f"{_seq_status['total_tasks']} task(s) completed before this failure. "
                            f"Fix the error above and re-run the same job - EasiGP will "
                            f"automatically resume from task {_seq_status['resume_from_task']}/"
                            f"{_seq_status['total_tasks']} instead of starting over."
                        )
                    if _incomplete:
                        st.warning(
                            f"{len(_incomplete)} Parallel batch(es) have partial progress saved - "
                            f"re-submit these (same config, same batch_id) to resume automatically:"
                        )
                        st.code('\n'.join(describe_incomplete_batch(b) for b in _incomplete), language='text')
                        st.write("**Incomplete batch ID list:**")
                        st.code(format_batch_id_list([b['batch_id'] for b in _incomplete]), language='text')
                else:
                    st.success('Pipeline completed successfully.')

            # Requirement: report any missing Step 2 batches somewhere more
            # visible/copy-pasteable than the pipeline log alone (which already
            # has the same information via the '[assemble] WARNING:' line
            # above, from assemble()'s own print()).
            if missing_batches:
                st.warning(
                    f"{len(missing_batches)} batch(es) produced no output and appear to be "
                    f"missing. Re-run Step 1 for exactly these batch ID(s) once the "
                    f"underlying problem is fixed, then re-run Step 2:"
                )
                st.code(format_batch_id_list(missing_batches), language='text')

            # Requirement: make it easy to notice, at a glance, which Parallel
            # batch(es) STARTED but did NOT finish - distinct from
            # missing_batches above (which never produced any output at all).
            # These have partial results on disk (see checkpoint_utils.py) that
            # were deliberately excluded from the assembled results, each with
            # a known resume point.
            if incomplete_batches:
                st.warning(
                    f"{len(incomplete_batches)} batch(es) started but did NOT finish - they hit an "
                    f"error partway through, and their partial results were excluded from the "
                    f"assembled output above. Re-submit each one below (same config, same batch_id) "
                    f"to resume it automatically from where it stopped:"
                )
                st.code('\n'.join(describe_incomplete_batch(b) for b in incomplete_batches), language='text')

            if log_buffer.getvalue().strip():
                with st.expander('Pipeline log', expanded=True):
                    st.caption(f'Also saved to: {log_file_path}')
                    st.code(log_buffer.getvalue(), language='text')

save_gui_state()
