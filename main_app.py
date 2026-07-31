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
import io
import ast
import json
import time
from datetime import datetime
import inspect
import contextlib
import traceback

import streamlit as st

from genomic_prediction import *
from assemble import *
from metric_plot import *
from scatter_plot import *
from circos_plot import *
from attention_histogram import *
from pipeline_utils import (
    BATCH_ID_SOURCES, configure_r_environment, init_rpy2_conversion,
    detect_array_job_env, restore_ratio, resolve_batch_id_from_env,
    TimestampedWriter, make_run_log_path,
)


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


def load_gui_state():
    """Restore last-used settings from disk, if any. Uses setdefault() so it
    only seeds keys that aren't already present in this session - on a
    session's first run this pre-populates every remembered widget value; on
    later reruns within the same session (the user is actively changing
    things) it's a no-op and never overwrites a live edit."""
    if not os.path.isfile(GUI_STATE_FILE):
        return
    try:
        with open(GUI_STATE_FILE, 'r') as f:
            saved = json.load(f)
    except Exception:
        return  # corrupted/unreadable state file - fall back to defaults silently

    for key, value in saved.items():
        if key.startswith('_btn_'):
            continue  # never replay a button click on a later launch - see below
        try:
            st.session_state.setdefault(key, value)
        except Exception:
            continue  # a small number of widget types can't be pre-seeded
                      # this way - just skip those


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
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            continue  # not JSON-serialisable - skip rather than fail the save
        snapshot[key] = value
    try:
        with open(GUI_STATE_FILE, 'w') as f:
            json.dump(snapshot, f)
    except Exception:
        pass  # best-effort - a failed save should never break the app


load_gui_state()


# --------------------------------------------------------------------------- #
# Static configuration / defaults (mirrors the values that used to be
# hard-coded at the top of the original script)
# --------------------------------------------------------------------------- #

AVAILABLE_MODELS = [
    'rrBLUP', 'GBLUP', 'BayesB', 'RKHS', 'RF', 'SVR', 'KNN', 'MLP',
    'GAT_infinitesimal', 'GAT_fully_connected', 'GAT_prior_knowledge', 'ensemble'
]

MODEL_DESCRIPTIONS = {
    'rrBLUP': "Ridge regression: assumes every marker has a small effect, and shrinks them all towards zero by a similar amount. A solid, fast general-purpose baseline.",
    'GBLUP': "Similar idea to rrBLUP, but works from overall genetic similarity between individuals rather than marker-by-marker effects directly.",
    'BayesB': "Like rrBLUP, but assumes only a subset of markers have a real effect (sparser) - a good fit when you expect a few large-effect markers rather than many small ones.",
    'RKHS': "Captures non-linear relationships between individuals' genetics and their trait, using a flexible similarity-based (kernel) approach rather than assuming purely additive marker effects.",
    'RF': "Random Forest: an ensemble of decision trees. Handles non-linear effects and marker interactions well, and can rank markers by importance.",
    'SVR': "Support Vector Regression: fits a flexible boundary through the data, with a tunable margin of error. Can capture non-linear patterns via its kernel setting.",
    'KNN': "K-Nearest Neighbours: predicts a trait as an average of the most genetically similar individuals in the training set. Simple and intuitive, but can be slow on large datasets.",
    'MLP': "Multi-Layer Perceptron: a small neural network. Can capture complex, non-linear patterns, but typically needs more data and tuning than the other models to do so reliably.",
    'GAT_infinitesimal': "A graph neural network that treats every marker does not interact with every other marker.",
    'GAT_fully_connected': "A graph neural network that treats every marker interacts with every other marker.",
    'GAT_prior_knowledge': "A graph neural network that first uses a Random Forest to identify likely-interacting marker pairs, and only connects those in the graph - can be faster and more targeted than connecting every marker to every other.",
    'ensemble': "Combines the predictions of every other selected model together (a simple average, unless a weight-optimisation method is chosen in 'Ensemble Weighting'), often giving more robust predictions than any single model alone.",
}

W_OPT_METHODS = ['Nelder Mead', 'Linear transformation', 'Bayesian optimisation']

W_OPT_METHOD_DESCRIPTIONS = {
    'Nelder Mead': "Searches for the best per-model weights using a direct, gradient-free numerical search. Simple and generally reliable.",
    'Linear transformation': "Learns the per-model weights using a small neural network (see its settings below), rather than a direct numerical search.",
    'Bayesian optimisation': "Searches for the best per-model weights intelligently, using past attempts to decide where to try next - can find good weights in fewer attempts than Nelder Mead, at the cost of more setup overhead per attempt.",
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
#
# Field order within each model matches the positional order the backend
# expects in HPARAMETERS[model] - do not reorder.
HPARAM_SPECS = {
    'rrBLUP': [
        {'label': 'Iteration number', 'type': 'int', 'default': 12000,
         'help': ("How many rounds of Bayesian model fitting (MCMC sampling) to run. More "
                  "iterations generally give a more stable, reliable fit, but take longer. "
                  "12000 is a reasonable starting point for genomic prediction.")},
        {'label': 'Burn-in', 'type': 'int', 'default': 2000,
         'help': ("How many of the initial iterations (above) are discarded before "
                  "averaging, to let the model 'warm up' and stop being influenced by its "
                  "arbitrary starting point. Must be smaller than the iteration number.")},
        {'label': 'Prior degrees of freedom (df0)', 'type': 'int', 'default': 5,
         'help': ("Controls how strongly the prior belief about marker-effect size is "
                  "held before seeing the data. Higher values make the model trust the "
                  "prior more (stronger shrinkage); lower values let the data dominate "
                  "more quickly. 5 is BGLR's own default.")},
        {'label': 'Expected proportion of variance explained (R2)', 'type': 'float', 'default': 0.5,
         'help': ("Your best guess at what fraction of the trait's variance the markers "
                  "explain overall - used to set how much shrinkage is applied to each "
                  "marker effect. Higher R2 = less shrinkage (bigger effects allowed); "
                  "lower R2 = more shrinkage (effects pulled closer to zero). 0.5 is "
                  "BGLR's own default and a reasonable starting point if unsure.")},
    ],
    'BayesB': [
        {'label': 'Iteration number', 'type': 'int', 'default': 12000,
         'help': ("How many rounds of Bayesian model fitting (MCMC sampling) to run. More "
                  "iterations generally give a more stable, reliable fit, but take longer. "
                  "12000 is a reasonable starting point for genomic prediction.")},
        {'label': 'Burn-in', 'type': 'int', 'default': 2000,
         'help': ("How many of the initial iterations (above) are discarded before "
                  "averaging, to let the model 'warm up' and stop being influenced by its "
                  "arbitrary starting point. Must be smaller than the iteration number.")},
        {'label': 'Prior probability of a nonzero effect (probIn)', 'type': 'float', 'default': 0.5,
         'help': ("BayesB's defining setting: the assumed proportion of markers with a "
                  "real, nonzero effect on the trait. Lower values (e.g. 0.05-0.1) assume "
                  "only a few markers matter (sparser, more like Bayesian variable "
                  "selection); higher values behave more like ridge regression, where "
                  "most markers contribute a little. 0.5 is BGLR's own default.")},
        {'label': 'Prior counts (counts)', 'type': 'int', 'default': 10,
         'help': ("How strongly the 'probIn' belief above is held before seeing the data - "
                  "higher values make BGLR trust that prior more strongly; lower values "
                  "let the data override it more easily. 10 is BGLR's own default.")},
    ],
    'GBLUP': [
        {'label': 'Iteration number', 'type': 'int', 'default': 12000,
         'help': ("How many rounds of Bayesian model fitting (MCMC sampling) to run. More "
                  "iterations generally give a more stable, reliable fit, but take longer. "
                  "12000 is a reasonable starting point for genomic prediction.")},
        {'label': 'Burn-in', 'type': 'int', 'default': 2000,
         'help': ("How many of the initial iterations (above) are discarded before "
                  "averaging, to let the model 'warm up' and stop being influenced by its "
                  "arbitrary starting point. Must be smaller than the iteration number.")},
        {'label': 'Return marker effect?', 'type': 'bool', 'default': False,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction (via Shapley scores), in addition to the prediction itself. "
                  "This is slower - the settings below only apply when this is checked.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'depends_on': (2, True),
         'help': ("How many test individuals to compute marker-effect (Shapley) scores for. "
                  "More individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
        {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
         'type': 'int_or_all', 'default': 500, 'depends_on': (2, True),
         'help': ("Only the top markers (ranked by correlation with the trait) are scored; "
                  "every other marker is reported as 0. Fewer markers = much faster. "
                  "'all' scores every marker but can take a very long time on datasets "
                  "with thousands of markers.")},
        {'label': 'Number of MCMC iterations for Shapley scores', 'type': 'int', 'default': 200,
         'depends_on': (2, True),
         'help': ("Each Shapley perturbation test re-fits the whole model from scratch, so this "
                  "controls how many MCMC iterations that re-fit uses - separate from, and much "
                  "smaller than, the main 'Iteration number' above. More iterations = more stable "
                  "scores but much slower, since this re-fit happens many times. 200 is a "
                  "reasonable balance for large datasets.")},
        {'label': 'Burn-in for Shapley scores', 'type': 'int', 'default': 50,
         'depends_on': (2, True),
         'help': ("How many of the Shapley re-fit's MCMC iterations (above) are discarded as "
                  "'warm-up' before averaging - must be smaller than that value. Separate from, "
                  "and much smaller than, the main 'Burn-in' above.")},
    ],
    'RKHS': [
        {'label': 'Iteration number', 'type': 'int', 'default': 12000,
         'help': ("How many rounds of Bayesian model fitting (MCMC sampling) to run. More "
                  "iterations generally give a more stable, reliable fit, but take longer. "
                  "12000 is a reasonable starting point for genomic prediction.")},
        {'label': 'Burn-in', 'type': 'int', 'default': 2000,
         'help': ("How many of the initial iterations (above) are discarded before "
                  "averaging, to let the model 'warm up' and stop being influenced by its "
                  "arbitrary starting point. Must be smaller than the iteration number.")},
        {'label': 'Kernel bandwidth (h)', 'type': 'float', 'default': 1.0,
         'help': ("Controls how quickly similarity between two samples drops off with "
                  "genetic distance. Higher h = only very close samples are treated as "
                  "similar (more locally-focused); lower h = more distant samples are "
                  "still treated as somewhat similar (smoother). 1 is often used as the midpoint.")},
        {'label': 'Return marker effect?', 'type': 'bool', 'default': False,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction (via Shapley scores), in addition to the prediction itself. "
                  "This is slower - the settings below only apply when this is checked.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'depends_on': (3, True),
         'help': ("How many test individuals to compute marker-effect (Shapley) scores for. "
                  "More individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
        {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
         'type': 'int_or_all', 'default': 500, 'depends_on': (3, True),
         'help': ("Only the top markers (ranked by correlation with the trait) are scored; "
                  "every other marker is reported as 0. Fewer markers = much faster. "
                  "'all' scores every marker but can take a very long time on datasets "
                  "with thousands of markers.")},
        {'label': 'Number of MCMC iterations for Shapley scores', 'type': 'int', 'default': 200,
         'depends_on': (3, True),
         'help': ("Each Shapley perturbation test re-fits the whole model from scratch, so this "
                  "controls how many MCMC iterations that re-fit uses - separate from, and much "
                  "smaller than, the main 'Iteration number' above. More iterations = more stable "
                  "scores but much slower, since this re-fit happens many times. 200 is a "
                  "reasonable balance for large datasets.")},
        {'label': 'Burn-in for Shapley scores', 'type': 'int', 'default': 50,
         'depends_on': (3, True),
         'help': ("How many of the Shapley re-fit's MCMC iterations (above) are discarded as "
                  "'warm-up' before averaging - must be smaller than that value. Separate from, "
                  "and much smaller than, the main 'Burn-in' above.")},
    ],
    'RF': [
        {'label': 'Tree number', 'type': 'int', 'default': 1000,
         'help': ("How many individual decision trees to average together. More trees "
                  "usually give steadier, more reliable predictions, at the cost of longer "
                  "runtime - returns diminish well before 1000 for most datasets.")},
        {'label': 'Maximum features per tree', 'type': 'rf_max_features', 'default': '1.0',
         'choices': ['sqrt', 'log2', 'None'], 'combo_state': 'normal',
         'help': ("How many markers each tree is allowed to consider at every split. "
                  "'sqrt' uses the value of the square root of the total marker count, and 'log2' uses "
                  "log base 2 of it - both use only a small random subset per split (more "
                  "diversity between trees, often better for many markers). 'None' or a "
                  "custom number lets each split consider all (or more) markers.")},
        {'label': 'Maximum samples per tree', 'type': 'int_float_or_none', 'default': None,
         'help': ("How many individuals (out of the training set) each tree is trained on, "
                  "drawn with replacement. 'None' (the default) uses as many as there are "
                  "training individuals. Lowering this makes trees more different from one "
                  "another, which can help or hurt depending on the dataset.")},
        {'label': 'Maximum tree depth', 'type': 'int_float_or_none', 'default': None,
         'help': ("Limits how many splits deep each tree can grow. 'None' (the default) lets "
                  "trees grow until every leaf is pure or too small to split further - this can "
                  "overfit on noisy data. A smaller number (e.g. 5-15) gives simpler, more "
                  "regularised trees.")},
        {'label': 'Minimum samples per leaf in each tree', 'type': 'int', 'default': 1,
         'help': ("The smallest number of samples allowed in a leaf node. 1 (the default) lets "
                  "trees fit very fine-grained detail; raising this (e.g. 5-20) smooths "
                  "predictions and reduces overfitting, especially with noisy phenotypes.")},
        {'label': 'Return marker effect for interactions?', 'type': 'bool', 'default': True,
         'help': ("If checked, also searches for pairs of markers that interact with each "
                  "other (beyond what each marker alone explains), in addition to the "
                  "prediction itself. This is slower - the settings below only apply when "
                  "this is checked.")},
        {'label': 'Number of samples for marker effect interactions', 'type': 'int', 'default': 30,
         'depends_on': (5, True),
         'help': ("How many test individuals to search for marker-pair interactions in. "
                  "More individuals give a more representative picture across the "
                  "population, but take longer.")},
        {'label': 'Output only the top N% of interactions ("all" for everything)', 'type': 'top_pct', 'default': 0.01,
         'depends_on': (5, True),
         'help': ("Only keep the strongest interactions found, as a percentage of all pairs "
                  "tested - e.g.0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which "
                  "can be a very large table for datasets with many markers.")},
        {'label': 'Max markers considered for interaction search ("all" for every marker)',
         'type': 'int_or_all', 'default': 500, 'depends_on': (5, True),
         'help': ("Only the top markers (ranked by importance) are checked for pairwise "
                  "interactions; every other marker pair is left out. Fewer markers = much "
                  "faster. 'all' checks every possible pair but can take a very long time on "
                  "datasets with thousands of markers.")},
    ],
    'SVR': [
        {'label': 'Kernel type', 'type': 'str', 'default': 'rbf',
         'choices': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'], 'combo_state': 'readonly',
         'help': ("The shape of similarity function used to compare individuals. 'rbf' "
                  "(the default) works well in most cases and can capture curved, non-linear "
                  "relationships; 'linear' is simpler and assumes marker effects add up "
                  "directly; 'poly'/'sigmoid' are other curved shapes worth trying if 'rbf' "
                  "underperforms.")},
        {'label': 'Epsilon', 'type': 'float', 'default': 0.5,
         'help': ("A margin of error the model is allowed to ignore - predictions within "
                  "epsilon of the true value aren't penalised at all. Larger epsilon gives a "
                  "simpler, less sensitive model; smaller epsilon tries to fit the data more "
                  "closely.")},
        {'label': 'Constraint', 'type': 'float', 'default': 1.0,
         'help': ("Controls the trade-off between fitting the training data closely and "
                  "keeping the model simple. Higher values fit the training data harder (risk "
                  "of overfitting); lower values favour a smoother, more general model.")},
        {'label': 'Dimension for poly kernel', 'type': 'int', 'default': 3,
         'depends_on': (0, 'poly'),
         'help': ("Only used when Kernel type is 'poly'. Higher values allow more complex, "
                  "curvier relationships between markers and the trait, but are more prone to "
                  "overfitting.")},
        {'label': 'Gamma', 'type': 'svr_gamma', 'default': 'scale',
         'choices': ['scale', 'auto'], 'combo_state': 'normal',
         'help': ("How far the influence of a single individual reaches (for the 'rbf', "
                  "'poly', and 'sigmoid' kernels). Higher gamma = each individual only "
                  "influences its closest neighbours (can overfit); lower gamma = influence "
                  "reaches further. 'scale' (the default) picks a sensible value automatically "
                  "based on your data.")},
        {'label': 'Independent term (coef0)', 'type': 'float', 'default': 0.0,
         'help': ("Only used by the 'poly' and 'sigmoid' kernels (ignored for 'rbf'/'linear'). "
                  "Shifts the kernel function by a constant - 0.0 is sklearn's own default.")},
        {'label': 'Return marker effect?', 'type': 'bool', 'default': False,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction (via Shapley scores), in addition to the prediction itself. "
                  "This is slower - the settings below only apply when this is checked.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'depends_on': (6, True),
         'help': ("How many test individuals to compute marker-effect (Shapley) scores for. "
                  "More individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
        {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
         'type': 'int_or_all', 'default': 500, 'depends_on': (6, True),
         'help': ("Only the top markers (ranked by correlation with the trait) are scored; "
                  "every other marker is reported as 0. Fewer markers = much faster. "
                  "'all' scores every marker but can take a very long time on datasets "
                  "with thousands of markers.")},
        {'label': 'Background sample size for Shapley scores', 'type': 'int', 'default': 50,
         'depends_on': (6, True),
         'help': ("A small reference set of samples used as a 'typical' baseline when working "
                  "out each marker's contribution. Smaller = faster; larger = smoother, more "
                  "stable scores but slower. 50 is a good default - you shouldn't normally "
                  "need to raise this much.")},
        {'label': 'Number of coalition samples for Shapley scores (nsamples)', 'type': 'int', 'default': 200,
         'depends_on': (6, True),
         'help': ("How many random combinations of markers are tried out to estimate each "
                  "marker's contribution to a prediction. Higher = more accurate but slower; "
                  "lower = faster but noisier scores. 200 is a reasonable balance for large "
                  "datasets.")},
    ],
    'KNN': [
        {'label': 'Number of neighbours', 'type': 'int', 'default': 5,
         'help': ("How many of the most genetically similar training individuals are "
                  "averaged together to predict each new individual's trait. Fewer neighbours "
                  "can pick up more local detail but are noisier; more neighbours give a "
                  "smoother, more stable prediction but can blur out real differences.")},
        {'label': 'Neighbour weighting', 'type': 'str', 'default': 'uniform',
         'choices': ['uniform', 'distance'], 'combo_state': 'readonly',
         'help': ("'uniform' treats every one of the k nearest neighbours equally when "
                  "averaging their phenotypes. 'distance' weights closer neighbours more "
                  "heavily than farther ones - often a better fit when genetic distance "
                  "varies a lot within the neighbourhood.")},
        {'label': 'Distance metric power (p)', 'type': 'int', 'default': 2,
         'help': ("Which Minkowski distance is used to find neighbours: p=1 is Manhattan "
                  "distance, p=2 (the default) is ordinary Euclidean distance. Higher values "
                  "increasingly emphasise the single largest per-marker difference between "
                  "two samples.")},
        {'label': 'Return marker effect?', 'type': 'bool', 'default': False,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction (via Shapley scores), in addition to the prediction itself. "
                  "This is slower - the settings below only apply when this is checked.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'depends_on': (3, True),
         'help': ("How many test individuals to compute marker-effect (Shapley) scores for. "
                  "More individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
        {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
         'type': 'int_or_all', 'default': 500, 'depends_on': (3, True),
         'help': ("Only the top markers (ranked by correlation with the trait) are scored; "
                  "every other marker is reported as 0. Fewer markers = much faster. "
                  "'all' scores every marker but can take a very long time on datasets "
                  "with thousands of markers.")},
        {'label': 'Background sample size for Shapley scores', 'type': 'int', 'default': 50,
         'depends_on': (3, True),
         'help': ("A small reference set of samples used as a 'typical' baseline when working "
                  "out each marker's contribution. Smaller = faster; larger = smoother, more "
                  "stable scores but slower. 50 is a good default - you shouldn't normally "
                  "need to raise this much.")},
        {'label': 'Number of coalition samples for Shapley scores (nsamples)', 'type': 'int', 'default': 200,
         'depends_on': (3, True),
         'help': ("How many random combinations of markers are tried out to estimate each "
                  "marker's contribution to a prediction. Higher = more accurate but slower; "
                  "lower = faster but noisier scores. 200 is a reasonable balance for large "
                  "datasets.")},
    ],
    'MLP': [
        {'label': 'Neuron numbers', 'type': 'int', 'default': 30,
         'help': ("How many units are in the (first) hidden layer. More neurons let the "
                  "network represent more complex patterns, but need more data to train "
                  "reliably and are more prone to overfitting.")},
        {'label': 'Dropout', 'type': 'float', 'default': 0,
         'help': ("The fraction of hidden-layer connections randomly switched off during "
                  "each training step, as a safeguard against overfitting. 0 disables this; "
                  "typical values are 0.1-0.5 if overfitting is a concern.")},
        {'label': 'Learning rate', 'type': 'float', 'default': 0.0001,
         'help': ("How big a step the network takes when updating its weights after each "
                  "batch. Too high can make training unstable or fail to settle down; too "
                  "low makes training very slow to improve.")},
        {'label': 'Decay', 'type': 'float', 'default': 5e-4,
         'help': ("A small penalty that discourages the network's weights from growing too "
                  "large, as another safeguard against overfitting. 0 disables it; larger "
                  "values regularise more strongly.")},
        {'label': 'Epoch', 'type': 'int', 'default': 200,
         'help': ("How many times the network passes over the entire training set. More "
                  "epochs let it learn more, but too many can start memorising noise in the "
                  "training data rather than the underlying trend.")},
        {'label': 'Batch size', 'type': 'int', 'default': 8,
         'help': ("How many individuals are processed together before each weight update. "
                  "Smaller batches update more often (noisier but sometimes better "
                  "generalisation); larger batches are faster per epoch but update less "
                  "often.")},
        {'label': 'Second hidden layer size', 'type': 'int_float_or_none', 'default': None,
         'help': ("Adds an extra hidden layer (with the same dropout rate as the first) "
                  "before the output. 'None' (the default) keeps the original single-hidden-"
                  "layer network unchanged. A deeper network can capture more complex "
                  "marker interactions, but is slower to train and easier to overfit on "
                  "small datasets.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'help': ("How many test individuals to compute marker-effect (Shapley) scores for. "
                  "More individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
    ],
    'GAT_infinitesimal': [
        {'label': 'Neuron numbers', 'type': 'int', 'default': 20,
         'help': ("How many units are in each graph-attention layer. More neurons let the "
                  "network represent more complex patterns, but need more data to train "
                  "reliably and are more prone to overfitting.")},
        {'label': 'Dropout', 'type': 'float', 'default': 0,
         'help': ("The fraction of attention connections randomly switched off during each "
                  "training step, as a safeguard against overfitting. 0 disables this; "
                  "typical values are 0.1-0.5 if overfitting is a concern.")},
        {'label': 'Learning rate', 'type': 'float', 'default': 0.01,
         'help': ("How big a step the network takes when updating its weights after each "
                  "batch. Too high can make training unstable or fail to settle down; too "
                  "low makes training very slow to improve.")},
        {'label': 'Decay', 'type': 'float', 'default': 5e-4,
         'help': ("A small penalty that discourages the network's weights from growing too "
                  "large, as another safeguard against overfitting. 0 disables it; larger "
                  "values regularise more strongly.")},
        {'label': 'Epoch', 'type': 'int', 'default': 40,
         'help': ("How many times the network passes over the entire training set. More "
                  "epochs let it learn more, but too many can start memorising noise in the "
                  "training data rather than the underlying trend.")},
        {'label': 'Batch size', 'type': 'int', 'default': 8,
         'help': ("How many individuals are processed together before each weight update. "
                  "Smaller batches update more often (noisier but sometimes better "
                  "generalisation); larger batches are faster per epoch but update less "
                  "often.")},
        {'label': 'Number of heads', 'type': 'int', 'default': 1,
         'help': ("How many independent 'attention patterns' the network learns at once - "
                  "each head can focus on a different way markers relate to one another. "
                  "More heads can capture more varied relationships, at the cost of a bigger, "
                  "slower model.")},
        {'label': 'Return marker effect?', 'type': 'bool', 'default': True,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction, in addition to the prediction itself. This is slower - the "
                  "setting below only applies when this is checked.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'depends_on': (7, True),
         'help': ("How many test individuals to compute marker-effect scores for. More "
                  "individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
    ],
    'GAT_fully_connected': [
        {'label': 'Neuron numbers', 'type': 'int', 'default': 20,
         'help': ("How many units are in each graph-attention layer. More neurons let the "
                  "network represent more complex patterns, but need more data to train "
                  "reliably and are more prone to overfitting.")},
        {'label': 'Dropout', 'type': 'float', 'default': 0,
         'help': ("The fraction of attention connections randomly switched off during each "
                  "training step, as a safeguard against overfitting. 0 disables this; "
                  "typical values are 0.1-0.5 if overfitting is a concern.")},
        {'label': 'Learning rate', 'type': 'float', 'default': 0.01,
         'help': ("How big a step the network takes when updating its weights after each "
                  "batch. Too high can make training unstable or fail to settle down; too "
                  "low makes training very slow to improve.")},
        {'label': 'Decay', 'type': 'float', 'default': 5e-4,
         'help': ("A small penalty that discourages the network's weights from growing too "
                  "large, as another safeguard against overfitting. 0 disables it; larger "
                  "values regularise more strongly.")},
        {'label': 'Epoch', 'type': 'int', 'default': 40,
         'help': ("How many times the network passes over the entire training set. More "
                  "epochs let it learn more, but too many can start memorising noise in the "
                  "training data rather than the underlying trend.")},
        {'label': 'Batch size', 'type': 'int', 'default': 8,
         'help': ("How many individuals are processed together before each weight update. "
                  "Smaller batches update more often (noisier but sometimes better "
                  "generalisation); larger batches are faster per epoch but update less "
                  "often.")},
        {'label': 'Number of heads', 'type': 'int', 'default': 1,
         'help': ("How many independent 'attention patterns' the network learns at once - "
                  "each head can focus on a different way markers relate to one another. "
                  "More heads can capture more varied relationships, at the cost of a bigger, "
                  "slower model.")},
        {'label': 'Return marker effect?', 'type': 'bool', 'default': True,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction, in addition to the prediction itself. This is slower - the "
                  "setting below only applies when this is checked.")},
        {'label': 'Number of samples for Shapley scores', 'type': 'int', 'default': 30,
         'depends_on': (7, True),
         'help': ("How many test individuals to compute marker-effect scores for. More "
                  "individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
    ],
    'GAT_prior_knowledge': [
        {'label': 'Neuron numbers', 'type': 'int', 'default': 20,
         'help': ("How many units are in each graph-attention layer. More neurons let the "
                  "network represent more complex patterns, but need more data to train "
                  "reliably and are more prone to overfitting.")},
        {'label': 'Dropout', 'type': 'float', 'default': 0,
         'help': ("The fraction of attention connections randomly switched off during each "
                  "training step, as a safeguard against overfitting. 0 disables this; "
                  "typical values are 0.1-0.5 if overfitting is a concern.")},
        {'label': 'Learning rate', 'type': 'float', 'default': 0.01,
         'help': ("How big a step the network takes when updating its weights after each "
                  "batch. Too high can make training unstable or fail to settle down; too "
                  "low makes training very slow to improve.")},
        {'label': 'Decay', 'type': 'float', 'default': 5e-4,
         'help': ("A small penalty that discourages the network's weights from growing too "
                  "large, as another safeguard against overfitting. 0 disables it; larger "
                  "values regularise more strongly.")},
        {'label': 'Epoch', 'type': 'int', 'default': 40,
         'help': ("How many times the network passes over the entire training set. More "
                  "epochs let it learn more, but too many can start memorising noise in the "
                  "training data rather than the underlying trend.")},
        {'label': 'Batch size', 'type': 'int', 'default': 8,
         'help': ("How many individuals are processed together before each weight update. "
                  "Smaller batches update more often (noisier but sometimes better "
                  "generalisation); larger batches are faster per epoch but update less "
                  "often.")},
        {'label': 'Number of heads', 'type': 'int', 'default': 1,
         'help': ("How many independent 'attention patterns' the network learns at once - "
                  "each head can focus on a different way markers relate to one another. "
                  "More heads can capture more varied relationships, at the cost of a bigger, "
                  "slower model.")},
        {'label': 'Selection rate for edges from RF (e.g. 10 = select the top 10% of the \
         most important edges))', 'type': 'float', 'default': 10,
         'help': ("This model first uses a Random Forest to identify which pairs of markers "
                  "seem to interact, then only connects those pairs in the graph. This "
                  "setting controls what fraction of all possible marker pairs are kept as "
                  "connections - a lower rate keeps only the strongest, most selective set "
                  "of relationships; a higher rate keeps more (noisier) connections.")},
        {'label': 'Return marker effects?', 'type': 'bool', 'default': True,
         'help': ("If checked, also estimates how much each marker contributes to the "
                  "prediction, in addition to the prediction itself. This is slower - the "
                  "settings below only apply when this is checked.")},
        {'label': 'Number of samples for marker effects', 'type': 'int', 'default': 30,
         'depends_on': (8, True),
         'help': ("How many test individuals to compute marker-effect scores for. More "
                  "individuals give a more representative picture of marker importance "
                  "across the population, but take longer.")},
    ],
}
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
    ],
    'Nelder Mead': [
        {'label': 'Initial value', 'type': 'float', 'default': 0.5,
         'help': ("The starting weight given to every model before the search begins. Since "
                  "all models start equal, the exact value doesn't usually matter much - it "
                  "just needs to be a valid starting point within the boundaries below.")},
        {'label': 'Minimum boundary', 'type': 'float', 'default': 0.1,
         'help': ("The smallest weight any individual model is allowed to be given during "
                  "the search.")},
        {'label': 'Maximum boundary', 'type': 'float', 'default': 10,
         'help': ("The largest weight any individual model is allowed to be given during "
                  "the search.")},
        {'label': 'fatol', 'type': 'float', 'default': 1e-8,
         'help': ("The search stops once further attempts stop improving the fit by more "
                  "than this tiny amount. Smaller values search more thoroughly (slower); "
                  "larger values stop sooner (faster, less precise).")},
        {'label': 'xatol', 'type': 'float', 'default': 1e-8,
         'help': ("Similar to fatol above, but based on how much the weights themselves stop "
                  "changing between attempts, rather than how much the fit improves.")},
        {'label': 'Adaptive', 'type': 'bool', 'default': False,
         'help': ("Automatically adjusts the search's internal step sizes for problems with "
                  "many models being weighted at once. Worth trying if the search seems to "
                  "struggle when combining a large number of models.")},
    ],
    'Bayesian optimisation': [
        {'label': 'Minimum boundary', 'type': 'float', 'default': 0.0001,
         'help': ("The smallest weight any individual model is allowed to be given during "
                  "the search.")},
        {'label': 'Maximum boundary', 'type': 'float', 'default': 10.0,
         'help': ("The largest weight any individual model is allowed to be given during "
                  "the search.")},
        {'label': 'Iterations', 'type': 'int', 'default': 50,
         'help': ("How many rounds of weight combinations this search tries before settling "
                  "on the best one found. More iterations search more thoroughly but take "
                  "longer.")},
        {'label': 'Point numbers', 'type': 'int', 'default': 1,
         'help': ("How many random weight combinations are tried before the search starts "
                  "using what it's learned so far to make smarter guesses. Usually fine left "
                  "at the default.")},
        {'label': 'Allow duplicate points', 'type': 'bool', 'default': True,
         'help': ("Whether the search is allowed to try the exact same weight combination "
                  "more than once. Leaving this on avoids the search getting stuck if it "
                  "runs out of new combinations to try.")},
    ],
}

DEFAULT_CYTOBAND_COLORMAP = {
    "gpos100": "#000000", "gpos": "#000000", "gpos75": "#828282",
    "gpos66": "#A0A0A0", "gpos50": "#C8C8C8", "gpos33": "#D2D2D2",
    "gpos25": "#C8C8C8", "gvar": "#DCDCDC", "gneg": "#FFFFFF",
    "acen": "#D92F27", "stalk": "#647FA4", "green": "#47c462",
    "brown": "#e0a22f", "purple": "#a62bcc", "blue0": "#def2ff",
    "blue1": "#c2e5fc", "blue2": "#addeff", "blue3": "#99d6ff",
    "blue4": "#83ccfc", "blue5": "#68c1fc", "blue6": "#45b5ff",
    "blue7": "#14a0fc", "blue8": "#027ac9", "blue9": "#014f82",
    "red0": "#fce1a7", "red1": "#ffd780", "red2": "#ffc954",
    "red3": "#fcba2b", "red4": "#ffaf03", "red5": "#d99502",
    "red6": "#b57c02", "red7": "#8a5e01", "red8": "#874001",
    "red9": "#610901", "transduction": "#02b01c",
    "transduction_clock": "#0afcd0", "clock": "#0389ad",
    "photoperiod": "#947b01", "autonomous": "#ffbc03",
    "integrator": "#91029c", "integrator_clock": "#f990fc",
    "GA": "#a990fc", "aging": "#90a7fc", "centromere": "#333333",
}

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
    'link_width': (
        'How thick the curved lines are that connect interacting marker pairs '
        'across the circle. Thicker lines are easier to see but can overlap and '
        'clutter a busy plot.'
    ),
    'interaction_top': (
        'Only the strongest interactions are drawn as links, to keep the plot '
        'readable - this sets what fraction to keep, e.g. 0.1 keeps only the '
        'top 0.1% strongest marker-pair interactions.'
    ),
    'label_size': 'Font size used for the interval ticks around the plot.',
    'scale': (
        'Adjust the intervals between ticks on a circos plot. '
        'A larger value will make the intervals wider.'
    ),
    'end_adjust': (
        'A small manual nudge to where each marker is positioned on its ring, to '
        'fix minor visual misalignment. 0 (no adjustment) is fine for most data.'
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
        st.number_input(label, value=float(default), key=key, disabled=disabled, help=help_text)

    elif ftype == 'str' and 'choices' in field:
        choices = field['choices']
        idx = choices.index(default) if default in choices else 0
        st.selectbox(label, options=choices, index=idx, key=key, disabled=disabled, help=help_text)

    elif ftype == 'str':
        st.text_input(label, value=str(default), key=key, disabled=disabled, help=help_text)

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


def resolve_field(key, field):
    """Read back the final Python value for a field rendered by render_field."""
    ftype = field['type']

    if ftype in ('bool', 'int', 'float', 'str'):
        return st.session_state.get(key, field['default'])

    mode_key, custom_key = key + '__mode', key + '__custom'
    mode = st.session_state.get(mode_key)

    if ftype == 'top_pct':
        if mode == 'all':
            return 'all'
        return float(st.session_state.get(custom_key, 1.0))

    if ftype == 'int_or_all':
        if mode == 'all':
            return 'all'
        return int(st.session_state.get(custom_key, 500))

    if ftype == 'rf_max_features':
        if mode in ('sqrt', 'log2'):
            return mode
        if mode == 'None':
            return None
        return float(st.session_state.get(custom_key, 1.0))

    if ftype == 'int_float_or_none':
        if mode == 'None':
            return None
        return float(st.session_state.get(custom_key, 1.0))

    if ftype == 'svr_gamma':
        if mode in ('scale', 'auto'):
            return mode
        return float(st.session_state.get(custom_key, 1.0))

    return None


def get_controller_value(prefix, controller_idx, spec):
    """Live value of a controlling field, falling back to its spec default
    on the very first render (before it has ever been instantiated)."""
    controller_field = spec[controller_idx]
    controller_key = f'{prefix}_{controller_idx}'
    return st.session_state.get(controller_key, controller_field['default'])


def render_hparam_panel(model):
    spec = HPARAM_SPECS[model]
    prefix = f'hp_{model}'
    with st.expander(f'{model} hyperparameters', expanded=True):
        for idx, field in enumerate(spec):
            key = f'{prefix}_{idx}'
            dep = field.get('depends_on')
            disabled = False
            if dep:
                controller_idx, required = dep
                disabled = get_controller_value(prefix, controller_idx, spec) != required
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


def render_hpc_export_section(export_mode, export_step, purpose, headless_script, config_filename, include_array):
    """Render the 'HPC job resource requests' expander and a single
    'Generate and save job files' button that writes the config JSON and
    the matching submission script (for the chosen scheduler only) straight
    to disk - the config into Result/<RESULT_NAME>/, the script next to
    streamlit_app.py. The saved script is also shown afterwards for
    reference/copying.

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
    """
    kp = purpose
    default_job_name = f"EasiGP_{st.session_state.get('result_name', 'job')}_{purpose}"
    default_max_index = 0

    with st.expander('HPC job resource requests (customise the generated script)', expanded=False):
        st.caption(
            "These settings control the #SBATCH / #PBS directives (and any extra setup "
            "commands) in the generated script - they don't affect the pipeline "
            "configuration itself."
        )
        st.selectbox(
            'Job scheduler', options=['Slurm', 'PBS'], key=f'{kp}_hpc_scheduler',
            help="Only a script for the chosen scheduler is generated below."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.text_input('Job name', value=default_job_name, key=f'{kp}_hpc_job_name',
                          help="Label shown for this job in the scheduler's queue (e.g. `squeue`/`qstat`).")
            st.number_input('Nodes', min_value=1, value=1, step=1, key=f'{kp}_hpc_nodes',
                             help="How many physical machines to request. Almost always 1 for this pipeline, "
                                  "since it isn't written to split a single run across multiple machines.")
            st.number_input('Tasks per node', min_value=1, value=1, step=1, key=f'{kp}_hpc_ntasks_per_node',
                             help="How many separate processes to run per node. Leave at 1 unless you "
                                  "specifically know you need more.")
        with c2:
            st.number_input('CPUs per task', min_value=1, value=1, step=1, key=f'{kp}_hpc_cpus_per_task',
                             help="How many CPU cores to reserve for this job (e.g. for Random Forest's "
                                  "parallel tree fitting).")
            st.text_input('Memory (e.g. 10G)', value='10G', key=f'{kp}_hpc_mem',
                          help="How much RAM to reserve for this job. Increase this for large genotype files.")
            st.text_input('Walltime (HH:MM:SS)', value='01:00:00', key=f'{kp}_hpc_time',
                          help="Maximum time the job is allowed to run before the scheduler kills it. "
                               "Set this generously - a job that hits this limit is stopped part-way through.")
        with c3:
            st.text_input('Partition / queue', value='general', key=f'{kp}_hpc_partition',
                          help="Which partition/queue on your cluster to submit to - check with your "
                               "cluster's documentation or administrator for the available names.")
            st.text_input('Account (leave blank to omit)', value='', key=f'{kp}_hpc_account',
                          help="Billing/allocation account to charge this job's usage to, if your "
                               "cluster requires one.")
            st.checkbox('Use login shell (--login)', value=True, key=f'{kp}_hpc_login_shell',
                        help="Runs the job in a login shell, which loads your usual environment/module "
                             "setup (e.g. conda, R). Leave checked unless you know you need otherwise.")

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

    if st.button('Generate and save job files', key=f'_btn_{kp}_generate_save'):
        try:
            export_cfg = gather_config(export_mode, export_step)
        except ValueError as exc:
            st.error(str(exc))
            return

        result_dir = os.path.abspath(os.path.join('.', 'Result', export_cfg['RESULT_NAME']))
        os.makedirs(result_dir, exist_ok=True)
        config_path = os.path.join(result_dir, config_filename)
        logs_dir = os.path.join(result_dir, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        if include_array:
            # batch_id is per-task and must NOT be baked into the shared
            # config - the headless script resolves it at run time from
            # SLURM_ARRAY_TASK_ID / PBS_ARRAY_INDEX for each individual task.
            export_cfg['PARALLEL'] = {'batch_size': export_cfg['PARALLEL']['batch_size']}

        with open(config_path, 'w') as f:
            json.dump(export_cfg, f, indent=2)

        scheduler = st.session_state.get(f'{kp}_hpc_scheduler', 'Slurm')
        job_name = st.session_state.get(f'{kp}_hpc_job_name', '').strip() or default_job_name
        nodes = int(st.session_state.get(f'{kp}_hpc_nodes', 1))
        ntasks_per_node = int(st.session_state.get(f'{kp}_hpc_ntasks_per_node', 1))
        cpus_per_task = int(st.session_state.get(f'{kp}_hpc_cpus_per_task', 1))
        mem = st.session_state.get(f'{kp}_hpc_mem', '10G').strip() or '10G'
        walltime = st.session_state.get(f'{kp}_hpc_time', '01:00:00').strip() or '01:00:00'
        partition = st.session_state.get(f'{kp}_hpc_partition', 'general').strip() or 'general'
        account = st.session_state.get(f'{kp}_hpc_account', '').strip()
        login_shell = bool(st.session_state.get(f'{kp}_hpc_login_shell', True))
        output_base = st.session_state.get(f'{kp}_hpc_output_base', '').strip() or default_job_name
        error_base = st.session_state.get(f'{kp}_hpc_error_base', '').strip() or default_job_name
        extra_lines = st.session_state.get(f'{kp}_hpc_extra_lines', '').strip()
        extra_block = (extra_lines + '\n') if extra_lines else ''

        shebang = '#!/bin/bash --login' if login_shell else '#!/bin/bash'
        account_line_slurm = f'#SBATCH --account={account}\n' if account else ''
        account_line_pbs = f'#PBS -A {account}\n' if account else ''

        if include_array:
            array_start = int(st.session_state.get(f'{kp}_hpc_array_start', 0))
            array_end = int(st.session_state.get(f'{kp}_hpc_array_end', default_max_index))
            slurm_array_line = f'#SBATCH -a {array_start}-{array_end}\n'
            pbs_array_line = f'#PBS -J {array_start}-{array_end}\n'
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
{slurm_array_line}
# Submit with:  sbatch {job_name}.sh
# Runs {index_note} via {headless_script} - no GUI and no manual
# configuration is needed beyond this one-time export.

{extra_block}python {headless_script} --config "{config_path}"
"""
        else:
            script_text = f"""{shebang}
# EasiGP - {purpose} job
#PBS -l select={nodes}:ncpus={cpus_per_task}:mem={mem}
#PBS -N {job_name}
#PBS -l walltime={walltime}
#PBS -q {partition}
{account_line_pbs}#PBS -o {pbs_out_pattern}
#PBS -e {pbs_err_pattern}
{pbs_array_line}
# Submit with:  qsub {job_name}.sh
# Runs {index_note} via {headless_script} - no GUI and no manual
# configuration is needed beyond this one-time export. Note: PBS memory
# units are typically lowercase (e.g. 10gb); adjust 'Memory' above if needed.

cd "$PBS_O_WORKDIR"
{extra_block}python {headless_script} --config "{config_path}"
"""

        project_dir = os.path.dirname(os.path.abspath(__file__))
        script_filename = f'{job_name}.sh'
        script_path = os.path.join(project_dir, script_filename)
        with open(script_path, 'w') as f:
            f.write(script_text)
        try:
            os.chmod(script_path, 0o755)
        except OSError:
            pass

        st.session_state[f'{kp}_config_path'] = config_path
        st.session_state[f'{kp}_script_path'] = script_path
        st.session_state[f'{kp}_script_draft'] = script_text

        st.success(f'Configuration saved to `{config_path}`.\n\nScript saved to `{script_path}`.')
        st.info(
            f"Make sure `{headless_script}` and `pipeline_utils.py` are also in "
            f"`{project_dir}` (or adjust the paths in the script). Submitting the "
            "saved script runs everything non-interactively - the GUI is only needed "
            "for this one-time configuration step."
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


def validate_ratio_for_wopt(ratio, w_opt):
    """Split ratio(s) must be (train, validation, test) tuples whenever a
    weight-optimisation approach is selected in 'Ensemble Weighting' (a
    validation split is required to optimise the weights), and plain
    numbers (train ratio only) otherwise."""
    if w_opt:
        for item in ratio:
            if not (isinstance(item, tuple) and len(item) == 3 and all(isinstance(x, (int, float)) for x in item)):
                raise ValueError(
                    "A weight-optimisation method is selected in 'Ensemble Weighting', "
                    "so Split ratio(s) must be a list of (train, validation, test) "
                    "tuples, e.g. [(0.8, 0.1, 0.1)] - not plain numbers."
                )
    else:
        for item in ratio:
            if isinstance(item, tuple):
                raise ValueError(
                    "No weight-optimisation method is selected in 'Ensemble Weighting', "
                    "so Split ratio(s) must be a list of plain numbers (train ratio "
                    "only), e.g. [0.8] - not (train, validation, test) tuples."
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

    # Everything below this point (model selection, hyperparameters, ratio,
    # genotype/phenotype files) is only gathered when Tab 2 ("Models &
    # Hyperparameters") is actually shown, i.e. Sequential mode or
    # Parallel / Step 1. In Parallel / Step 2 these values come from the
    # assemble() call instead, so we skip them entirely here.
    if not is_step2:
        selected_models = [m for m in AVAILABLE_MODELS if st.session_state.get(f'model_selected_{m}', False)]
        if not selected_models:
            raise ValueError('Please select at least one model to run.')
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

        cfg['GENOTYPE_FILE_NAME'] = st.session_state.get('genotype_path', '').strip()
        cfg['PHENOTYPE_FILE_NAME'] = st.session_state.get('phenotype_path', '').strip()
        if not os.path.isfile(cfg['GENOTYPE_FILE_NAME']):
            raise ValueError(f"Genotype file not found: {cfg['GENOTYPE_FILE_NAME']}")
        if not os.path.isfile(cfg['PHENOTYPE_FILE_NAME']):
            raise ValueError(f"Phenotype file not found: {cfg['PHENOTYPE_FILE_NAME']}")

        hparameters = {}
        for model in selected_models:
            if model == 'ensemble':
                continue
            hparameters[model] = resolve_hparams(model)
        cfg['HPARAMETERS'] = hparameters

        # LD pruning (Tab 2) - optional pre-processing step passed into GP().
        if st.session_state.get('ld_prune_enabled', False):
            window_unit = st.session_state.get('ld_window_unit', 'kb')
            snp_info_path = st.session_state.get('ld_snp_info_path', '').strip()

            if window_unit in ('kb', 'cm') and not snp_info_path:
                raise ValueError(
                    "LD Pruning: a SNP info file path is required when Window unit is "
                    "'kb' or 'cm'."
                )
            if snp_info_path and not os.path.isfile(snp_info_path):
                raise ValueError(f"LD Pruning: SNP info file not found: {snp_info_path}")

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
                'step': int(st.session_state.get('ld_step', 5)),
                'r2_threshold': float(st.session_state.get('ld_r2_threshold', 0.2)),
                'plink_path': st.session_state.get('ld_plink_path', 'plink2').strip() or 'plink2',
                'allow_extra_chr': bool(st.session_state.get('ld_allow_extra_chr', False)),
                'chr_set': chr_set,
                'work_dir': work_dir or None,
                'keep_intermediate': bool(st.session_state.get('ld_keep_intermediate', False)),
                'round_dosage': bool(st.session_state.get('ld_round_dosage', True)),
                'unmapped_strategy': st.session_state.get('ld_unmapped_strategy', 'variant_count'),
            }
            if ld_prune['unmapped_strategy'] == 'variant_count':
                ld_prune['unmapped_window'] = int(st.session_state.get('ld_unmapped_window', 50))
                ld_prune['unmapped_step'] = int(st.session_state.get('ld_unmapped_step', 5))

            cfg['LD_PRUNE'] = ld_prune
        else:
            cfg['LD_PRUNE'] = None

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

    # Whether Step 2 should skip re-combining every batch and just reload the
    # already-assembled combined CSVs from a previous run (see load_assembled()
    # in assemble.py) - only meaningful for Step 2.
    if is_step2:
        cfg['SKIP_ASSEMBLE'] = bool(st.session_state.get('step2_skip_assemble', False))

    # Ensemble weighting (Tab 3) is shown in every mode/step.
    selected_wopt = [m for m in W_OPT_METHODS if st.session_state.get(f'wopt_selected_{m}', False)]
    cfg['W_OPT'] = selected_wopt if selected_wopt else None

    hparameters_opt = {}
    for method in selected_wopt:
        hparameters_opt[method] = resolve_wopt(method)
    cfg['HYPERPARAMETERS_OPT'] = hparameters_opt

    if 'RATIO' in cfg:
        validate_ratio_for_wopt(cfg['RATIO'], cfg['W_OPT'])

    # Scatter plot (Tab 4) and Circos plot (Tab 5) config is only gathered
    # when those tabs are shown, i.e. Sequential mode or Parallel / Step 2.
    if not is_step1:
        cfg['SCATTER_CREATE'] = bool(st.session_state.get('scatter_create', True))
        qtl = st.session_state.get('qtl_path', '').strip()
        cfg['QTL'] = qtl if qtl else None
        cfg['SCATTER_CONFIG'] = {
            'font_size': int(st.session_state.get('scatter_font', 2)),
            'fig_size': int(st.session_state.get('scatter_fig', 30)),
        }

        cfg['CHROMOSOME_INFO'] = st.session_state.get('chrom_info_path', '').strip()
        cfg['MARKER_INFO'] = st.session_state.get('marker_info_path', '').strip()
        gene_info = st.session_state.get('gene_info_path', '').strip()
        cfg['GENE_INFO'] = gene_info if gene_info else None
        if not os.path.isfile(cfg['CHROMOSOME_INFO']):
            raise ValueError(f"Chromosome info file not found: {cfg['CHROMOSOME_INFO']}")
        if not os.path.isfile(cfg['MARKER_INFO']):
            raise ValueError(f"Marker info file not found: {cfg['MARKER_INFO']}")

        cfg['CIRCOS_CONFIG'] = {
            'space': float(st.session_state.get('circos_space', 1)),
            'start': float(st.session_state.get('circos_start', 15)),
            'end': float(st.session_state.get('circos_end', 345)),
            'link_width': float(st.session_state.get('circos_linkwidth', 20)),
            'interaction_top': float(st.session_state.get('circos_topinteraction', 0.01)),
            'label_size': float(st.session_state.get('circos_labelsize', 6)),
            'scale': float(st.session_state.get('circos_scale', 100)),
        }
        cfg['END_ADJUST'] = float(st.session_state.get('end_adjust', 0))
        cfg['WINDOW'] = float(st.session_state.get('window_size', 300))

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
        st.checkbox(
            'Skip assemble (reuse previously assembled results)',
            key='step2_skip_assemble',
            value=False,
            help=(
                "If Step 2 has already been run successfully once for this result "
                "name (so Metric.csv, Prediction_result_test.csv, etc. already exist "
                "in its Result folder) and only the scatter/circos plotting step is "
                "failing - e.g. because of a QTL file or circos config issue - check "
                "this to skip re-combining every batch and just reload the "
                "already-assembled files, then retry the plots. Leave unchecked for "
                "a normal Step 2 run, or if this is the first time assembling this "
                "result."
            ),
        )

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
#   Sequential            -> all 5 tabs
#   Parallel / Step 1     -> Data & Setup, Models & Hyperparameters, Ensemble Weighting
#   Parallel / Step 2     -> Data & Setup, Ensemble Weighting, Scatter Plot, Circos Plot
show_tab_models = (mode == 'Sequential') or (mode == 'Parallel' and step == 'Step 1')
show_tab_plots = (mode == 'Sequential') or (mode == 'Parallel' and step == 'Step 2')

tab_labels = ['1. Data & Setup']
tab_keys = ['setup']
if show_tab_models:
    tab_labels.append('2. LD Pruning')
    tab_keys.append('ld_pruning')
    tab_labels.append('3. Models & Hyperparameters')
    tab_keys.append('models')
tab_labels.append('4. Ensemble Weighting')
tab_keys.append('ensemble')
if show_tab_plots:
    tab_labels.append('5. Scatter Plot')
    tab_keys.append('scatter')
    tab_labels.append('6. Circos Plot')
    tab_keys.append('circos')

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

    genotype_path = st.text_input(
        'Genotype file path',
        value='./Data/MaizeNAM/MaizeNAM_dataset_genotype_population_1.csv', key='genotype_path',
        help=("The marker data file: one row per individual, with ID, population, and one "
              "column per genetic marker. See EasiGP's Data format guide if you need to "
              "convert your data into this layout.")
    )
    file_status(genotype_path)

    phenotype_path = st.text_input(
        'Phenotype file path',
        value='./Data/MaizeNAM/MaizeNAM_dataset_phenotype_population_1.csv', key='phenotype_path',
        help=("The trait data file: one row per individual (matching the genotype file's IDs), "
              "with ID, population, and one column per trait.")
    )
    file_status(phenotype_path)

    st.text_input(
        "Target phenotype(s) - comma separated, or 'all'", value='days2anthesis', key='phenotype_targets',
        help=("Which trait(s) from the phenotype file to predict. Name one or more columns "
              "(comma separated) exactly as they appear in that file, or type 'all' to "
              "predict every trait column found there.")
    )
    st.text_input('Result folder name', value='MaizeNAM', key='result_name',
                  help="A name for this run - results are saved under ./Result/<this name>/.")

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
            "If no weight-optimisation approach is selected in '4. Ensemble Weighting', "
            "type one or more plain numbers (train ratio only), e.g. 0.8 or 0.8, 0.65. "
            "If any weight-optimisation approach is selected there, type one or more "
            "(train, validation, test) tuples instead, e.g. (0.8, 0.1, 0.1)."
        )
        st.number_input('Number of random-split iterations', min_value=1, value=1, step=1, key='iter_num',
                         help=("How many times to repeat the whole train/test or train/validation/test split randomly and "
                               "re-run everything for checking consistency in the results."))
    #else:
    #    st.caption("'between' expects a list of (train, validation, test) tuples, e.g. [(0.8,0.1,0.1)]")

    # ------------------------------------------------------------------- #
    # Parallel batch configuration - only shown for Parallel / Step 1.
    # ------------------------------------------------------------------- #
    if mode == 'Parallel' and step == 'Step 1':
        st.divider()
        st.subheader('Parallel batch configuration (PARALLEL)')
        st.caption(
            "All prediction scenarios are split into a number of batches. "
            "Choose how the batch ID should be determined; only 'Manual integer' "
            "requires you to enter anything - for Slurm/PBS it's assigned "
            "automatically and never needs to be typed in here."
        )
        st.selectbox(
            'Batch ID source', options=BATCH_ID_SOURCES, key='batch_id_source',
            help=(
                "'Manual integer' lets you type the batch ID directly - only used "
                "for a local single-batch test run below. "
                "'Slurm' and 'PBS' need no input here: the actual batch ID is "
                "assigned automatically at run time (SLURM_ARRAY_TASK_ID / "
                "PBS_ARRAY_INDEX, set per-task by the scheduler itself)."
            ),
        )
        batch_id_source = st.session_state.get('batch_id_source', BATCH_ID_SOURCES[0])
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

        st.number_input('Batch size (number of prediction scenarios per batch)',
                         min_value=1, value=3, step=1, key='batch_size',
                         help=("How many population/phenotype/ratio/replicate combinations each "
                               "array-job task processes. Larger batches mean fewer, longer-running "
                               "tasks; smaller batches mean more, shorter tasks that finish in "
                               "parallel sooner (if your cluster has the capacity to run them "
                               "simultaneously)."))
        st.caption(
            "\U0001f4a1 This tab is only configured **once**. The batch ID for each "
            "individual array-job task is *not* set here - it's picked up automatically "
            "at run time from the scheduler's own environment variable "
            "(`SLURM_ARRAY_TASK_ID` / `PBS_ARRAY_INDEX`) by `run_step1_batch.py`, a plain "
            "script with no GUI. See 'Run pipeline' below."
        )

# ------------------------------ Tab 2: LD Pruning ------------------------------ #
if show_tab_models:
    with tab_map['ld_pruning']:
        st.checkbox(
            'Apply LD pruning as a data pre-processing step before model fitting',
            value=False, key='ld_prune_enabled',
            help="When enabled, LD_pruning() is called inside GP() to remove SNPs in "
                 "high linkage disequilibrium before the selected models are fitted."
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

            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input(
                    'Window size', min_value=0.0, value=50.0, key='ld_window',
                    help="Units depend on 'Window unit' above: # variants, kilobases, or centimorgans."
                )
            with c2:
                st.number_input('Step size (variant count)', min_value=1, value=5, step=1, key='ld_step',
                                 help="How many SNPs the pruning window slides forward by after each check.")
            with c3:
                st.number_input(
                    'r\u00b2 threshold', min_value=0.0, max_value=1.0, value=0.2, key='ld_r2_threshold',
                    help="Unphased hardcall r\u00b2 threshold above which a variant is pruned "
                         "(same meaning as PLINK's --indep-pairwise)."
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

# ------------------------ Tab 3: Models & hparams ------------------------ #
if show_tab_models:
    with tab_map['models']:
        st.subheader('Select model(s) to run')
        cols = st.columns(3)
        for i, model in enumerate(AVAILABLE_MODELS):
            default_checked = model in ('rrBLUP', 'BayesB', 'RF', 'ensemble')
            with cols[i % 3]:
                st.checkbox(model, value=default_checked, key=f'model_selected_{model}',
                            help=MODEL_DESCRIPTIONS.get(model))

        st.divider()
        for model in AVAILABLE_MODELS:
            if model == 'ensemble':
                continue
            if st.session_state.get(f'model_selected_{model}', False):
                render_hparam_panel(model)

# ------------------------ Tab 4: Ensemble weighting ------------------------ #
with tab_map['ensemble']:
    st.caption(
        "Only used when 'ensemble' is included in Models & Hyperparameters. "
        "Select the weight-optimisation method(s), or leave all unchecked for "
        "a naive (equal-weight) ensemble. Selecting any method here means the "
        "Split ratio(s) on Tab 1 must be given as (train, validation, test) "
        "tuples, e.g. (0.8, 0.1, 0.1)."
    )
    cols = st.columns(len(W_OPT_METHODS))
    for i, method in enumerate(W_OPT_METHODS):
        with cols[i]:
            st.checkbox(method, value=False, key=f'wopt_selected_{method}',
                        help=W_OPT_METHOD_DESCRIPTIONS.get(method))

    st.divider()
    for method in W_OPT_METHODS:
        if st.session_state.get(f'wopt_selected_{method}', False):
            render_wopt_panel(method)

# ------------------------------ Tab 5: Scatter ------------------------------ #
if show_tab_plots:
    with tab_map['scatter']:
        st.checkbox('Create scatter plot matrix?', value=True, key='scatter_create',
                    help=("Draws a grid comparing every pair of selected single models at both predicted phenotype and marker effect levels"))
        qtl_path = st.text_input('QTL file path (leave blank for None)', value='', key='qtl_path',
                                  help=("Optional: a csv file listing markers already known to be real "
                                        "QTLs, so they can be highlighted separately from other "
                                        "markers in the scatter plots."
                                        "each row represents QTL and contains two columns: "
                                        "phenotype|marker name identified as QTL"))
        if qtl_path:
            file_status(qtl_path)
        st.number_input('Font size', min_value=1, value=2, step=1, key='scatter_font',
                         help="Text size used for axis labels in the scatter plot matrix.")
        st.number_input('Figure size', min_value=1, value=30, step=1, key='scatter_fig',
                         help="Overall size of the scatter plot matrix image, in inches.")

    # ------------------------------ Tab 6: Circos ------------------------------ #
    with tab_map['circos']:
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

        st.number_input('Space between rings', value=1.0, key='circos_space', help=CIRCOS_HELP['space'])
        st.caption("A value around 1 suits most chromosome counts - increase it if labels start overlapping.")

        st.number_input('Start angle', value=15.0, key='circos_start', help=CIRCOS_HELP['start'])
        st.caption("0° starts at the top of the circle and increases clockwise.")

        st.number_input('End angle', value=345.0, key='circos_end', help=CIRCOS_HELP['end'])
        st.caption("Leave a small gap (e.g. 15° to 345°) so labels near the seam don't collide.")

        st.number_input('Link width', value=20.0, key='circos_linkwidth', help=CIRCOS_HELP['link_width'])
        st.caption("A higher value (15-25) makes sparse but important interactions easier to spot.")

        st.number_input('Top interaction percentage to display', value=0.001, format='%.4f', key='circos_topinteraction',
                         help=CIRCOS_HELP['interaction_top'])
        st.caption("Lower this if the plot looks too cluttered with links; raise it to surface more interactions.")

        st.number_input('Label font size', value=6.0, key='circos_labelsize', help=CIRCOS_HELP['label_size'])
        st.caption("6-8 is usually readable; go smaller only with many densely-packed markers.")

        st.number_input('Scale', value=100.0, key='circos_scale', help=CIRCOS_HELP['scale'])
        st.caption("100 is the default; increase for wider tick intervals around plot.")

        st.number_input('Averaging window size (WINDOW)', value=300.0, key='window_size', help=CIRCOS_HELP['window'])
        st.caption("0 disables windowing and uses each marker's raw effect directly instead of an average. "
                   "If you cannot see marker effects, you may consider increase the size of the window.")

        with st.expander('Advanced settings'):
            st.number_input('Edge location adjustment (END_ADJUST)', value=0.0, key='end_adjust', help=CIRCOS_HELP['end_adjust'])
            st.caption("Extend the start and end location of each marker for clearer visualisation. "
                       "A larger value may not represent the true location of each marker.")

            st.selectbox('Marker effect ordering (ASCENDING)', options=['True', 'False', 'None'], index=2, key='ascending',
                         help=CIRCOS_HELP['ascending'])
            st.caption("Leave as 'None' unless you specifically want overlapping marker regions resolved by effect strength.")

        st.divider()

        if 'cytoband_colormap' not in st.session_state:
            st.session_state['cytoband_colormap'] = dict(DEFAULT_CYTOBAND_COLORMAP)

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
            new_color_value = st.color_picker('Pick colour', value='#ffffff', key='cmap_new_color',
                                               help="The colour to draw that cytoband/region in.")
            if st.button('Add colour', key='_btn_cmap_add_colour'):
                if new_name.strip():
                    st.session_state['cytoband_colormap'][new_name.strip()] = new_color_value
                    st.rerun()
                else:
                    st.error('Please enter a name for the new colour.')

# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #

st.divider()
st.header('Run pipeline')

is_step1 = (mode == 'Parallel' and step == 'Step 1')
is_step2 = (mode == 'Parallel' and step == 'Step 2')
is_sequential = (mode == 'Sequential')

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
        result_dir = os.path.abspath(os.path.join('.', 'Result', cfg['RESULT_NAME']))
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
        with st.spinner('Running pipeline... this may take a while.'):
            try:
                configure_r_environment(cfg.get('R_PATH'))
                init_rpy2_conversion()

                with open(log_file_path, 'w') as log_file, \
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
                            cfg['SCENARIO'], LD_prune=cfg['LD_PRUNE'], **gp_progress_kwargs
                        )
                        _log('Genomic prediction finished.')

                        # Declare how many plot phases will run so the remaining
                        # 15% of the bar (85-100%) is divided evenly between them.
                        has_attention = 'GAT_fully_connected' in cfg['MODEL'] or 'GAT_prior_knowledge' in cfg['MODEL']
                        _set_post_phase_count((1 if has_attention else 0) + 1 + (1 if cfg['SCATTER_CREATE'] else 0) + 1)

                        if has_attention:
                            _advance_progress('Generating attention distribution plots...')
                            _t0 = time.time()
                            attention_distribution(attention, cfg['RESULT_NAME'], 10)
                            _log(f'Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

                        model_labels = cfg['MODEL'] + cfg['W_OPT'] if cfg['W_OPT'] is not None else cfg['MODEL']
                        _advance_progress('Generating metric plots...')
                        _t0 = time.time()
                        metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'])
                        _log(f'Metric plots generated (took {time.time() - _t0:.1f}s).')

                        if cfg['SCATTER_CREATE']:
                            _advance_progress('Generating scatter plot matrix...')
                            _t0 = time.time()
                            scatter_plot(cfg['MODEL'], phenotype, predicted_result_test, effect,
                                         cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'])
                            _log(f'Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

                        _advance_progress('Generating circos plot...')
                        _t0 = time.time()
                        circos_plot(effect, interactions, cfg['MARKER_INFO'], cfg['CHROMOSOME_INFO'],
                                    cfg['GENE_INFO'], population, phenotype, cfg['CIRCOS_CONFIG'],
                                    cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                                    cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'])
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
                            cfg['SCENARIO'], parallel, LD_prune=cfg['LD_PRUNE'], **gp_progress_kwargs
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
                        # Declare the phases known up front (assemble/load, metric,
                        # scatter?, circos); attention's applicability isn't known
                        # until after assemble/load returns, so it's added below.
                        _set_post_phase_count(1 + 1 + (1 if cfg['SCATTER_CREATE'] else 0) + 1)

                        if cfg['SKIP_ASSEMBLE']:
                            _advance_progress('Reloading previously assembled results...')
                            _t0 = time.time()
                            (metrics, predicted_result_train, predicted_result_test, effect,
                             interactions, attention, population, phenotype, assembled_model) = load_assembled(
                                cfg['RESULT_NAME']
                            )
                            _log(f'Skipped assemble - reloaded previously assembled results for models '
                                 f'{assembled_model} (took {time.time() - _t0:.1f}s).')
                        else:
                            _advance_progress('Assembling results from all batches...')
                            _t0 = time.time()
                            (metrics, predicted_result_train, predicted_result_test, effect,
                             interactions, attention, population, phenotype, assembled_model) = assemble(
                                cfg['RESULT_NAME']
                            )
                            _log(f'Assembled results from all batches for models '
                                 f'{assembled_model} (took {time.time() - _t0:.1f}s).')

                        has_attention = 'GAT_fully_connected' in assembled_model or 'GAT_prior_knowledge' in assembled_model
                        if has_attention:
                            _set_post_phase_count(post_phase_state['total_phases'] + 1)

                        if has_attention:
                            _advance_progress('Generating attention distribution plots...')
                            _t0 = time.time()
                            attention_distribution(attention, cfg['RESULT_NAME'], 10)
                            _log(f'Attention distribution plots generated (took {time.time() - _t0:.1f}s).')

                        model_labels = (
                            assembled_model + cfg['W_OPT'] if cfg['W_OPT'] is not None else assembled_model
                        )
                        _advance_progress('Generating metric plots...')
                        _t0 = time.time()
                        metric_plot(metrics.copy(), model_labels, cfg['RESULT_NAME'], cfg['SCENARIO'])
                        _log(f'Metric plots generated (took {time.time() - _t0:.1f}s).')

                        if cfg['SCATTER_CREATE']:
                            _advance_progress('Generating scatter plot matrix...')
                            _t0 = time.time()
                            scatter_plot(assembled_model, phenotype, predicted_result_test, effect,
                                         cfg['QTL'], cfg['SCATTER_CONFIG'], cfg['RESULT_NAME'])
                            _log(f'Scatter plot matrix generated (took {time.time() - _t0:.1f}s).')

                        _advance_progress('Generating circos plot...')
                        _t0 = time.time()
                        circos_plot(effect, interactions, cfg['MARKER_INFO'], cfg['CHROMOSOME_INFO'],
                                    cfg['GENE_INFO'], population, phenotype, cfg['CIRCOS_CONFIG'],
                                    cfg['END_ADJUST'], cfg['WINDOW'], cfg['CYTOBAND_COLORMAP'],
                                    cfg['RESULT_NAME'], attention, cfg['SCENARIO'], cfg['ASCENDING'])
                        _log(f'Circos plot generated (took {time.time() - _t0:.1f}s).')

            except Exception:
                tb_text = traceback.format_exc()
                st.error('Pipeline failed - see the traceback below.')
                st.code(tb_text, language='text')
                try:
                    with open(log_file_path, 'a') as log_file:
                        log_file.write(f'\n[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Pipeline failed:\n')
                        log_file.write(tb_text)
                except Exception:
                    pass  # best-effort - a logging failure shouldn't mask the real error
            else:
                st.success('Pipeline completed successfully.')

        if log_buffer.getvalue().strip():
            with st.expander('Pipeline log', expanded=True):
                st.caption(f'Also saved to: {log_file_path}')
                st.code(log_buffer.getvalue(), language='text')

save_gui_state()
