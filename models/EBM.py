"""
models/EBM.py
===============
Update ID ver4-5, R2 Stage 11 (blueprint §4.2 Layer 3, "Tier 2" - decision
D2: gated behind an availability probe, never a hard import at module
scope, invariant I12).

``interpret`` (the InterpretML package) is an OPTIONAL dependency. As
with ``models/XGBoost.py``, this module is only ever REACHED after
``model_registry.is_available('EBM')`` has confirmed the package
imports; the import itself is still deferred to inside a function, so
this file remains safely importable without the package installed.

An Explainable Boosting Machine (EBM, GA2M - "generalized additive model
plus pairwise interactions") is structurally different from every other
model in this codebase for R2's purposes: its pairwise interaction terms
ARE part of the fitted model itself, not a post-hoc explanation computed
afterwards (blueprint §4.2 Layer 3: "native pairwise terms are the
model, not a post-hoc explanation"). ``get_interaction`` therefore
controls something no other model's own flag does: whether EBM's own
FIT-TIME interaction search runs at all (``interactions=0`` disables it
outright, saving fit time), not merely whether an already-fitted model
is explained afterwards. When it does run, `max_interaction_features`
(repurposed here as EBM's own ``interactions`` hyperparameter - see
``hparam_specs.py`` for the field's EBM-specific help text) sets HOW
MANY pairwise terms EBM searches for during fitting; ``threshold`` still
applies this module's usual top-N% OUTPUT filter afterwards, exactly
like every other model, on top of whatever EBM itself already found.

Marker effect also comes from EBM's own already-fitted MAIN-effect terms
(``term_features_`` 1-tuples) via the SAME ``term_importances()`` read
``models.interaction_extraction.ebm_pairwise_interactions()`` uses for
the pairwise terms - free, no extra explainer needed, mirroring RF's/
ExtraTrees' own "effect is already sitting in the fitted model" pattern.
"""

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

from pipeline_utils import get_active_compute_resources
from models.interaction_extraction import ebm_pairwise_interactions, top_select


def _ebm_regressor_cls():
    """Lazily import interpret.glassbox.ExplainableBoostingRegressor -
    see models/XGBoost.py's identical helper for the full rationale."""
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        return ExplainableBoostingRegressor
    except ImportError as exc:
        raise ImportError(
            "models/EBM.py: the 'EBM' model was selected but the optional 'interpret' "
            "package is not installed in this environment. Install it with "
            "'pip install interpret', or remove 'EBM' from MODEL. This should have been "
            "caught earlier by GP()'s own config-validation check "
            "(MODEL_AVAILABILITY_STRICT) - seeing this error directly means that check "
            "was bypassed."
        ) from exc


def _ebm_effect(fitted_ebm, marker_names):
    """Read EBM's own already-fitted MAIN-effect term importances into a
    full-width, marker-name-ordered effect vector - the free, no-extra-
    explainer counterpart to RF.py's own feature_importances_ read."""
    marker_names = list(marker_names)
    effect_full = pd.Series(0.0, index=marker_names)
    for term, importance in zip(fitted_ebm.term_features_, fitted_ebm.term_importances()):
        if len(term) == 1:
            effect_full.iloc[term[0]] = float(importance)
    return pd.DataFrame(effect_full).T


def EBM(train, valid, test, params):

    learning_rate = params[0]
    max_leaves = int(params[1])
    min_samples_leaf = int(params[2])
    outer_bags = int(params[3])
    get_interaction = params[4]
    # EBM's own 'interactions' fit-time hyperparameter: HOW MANY pairwise
    # terms it searches for while fitting (see module docstring) - reuses
    # this codebase's usual 'max_interaction_features' field NAME for GUI
    # consistency with every other model, but its MEANING here is "number
    # of pairwise terms", not "size of a marker shortlist" (documented in
    # hparam_specs.py's own EBM-specific help text).
    max_interaction_features = params[5]
    threshold = params[6]

    train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
    test_x, test_y = test.iloc[:, :-1], test.iloc[:, -1]

    _resources = get_active_compute_resources()
    _ebm_cls = _ebm_regressor_cls()

    _interactions_arg = 0
    if get_interaction == True:
        # 'all' has no clean mapping onto EBM's own 'interactions' fit-time
        # hyperparameter (unlike every OTHER model's own 'max_interaction_
        # features', which shortlists a MARKER set, this field sets a
        # COUNT of pairwise TERMS - "every possible pair" would mean
        # searching all C(M,2) candidates, which is exactly the expensive
        # search EBM's own interaction-discovery heuristic exists to
        # avoid). 'all' therefore falls back to EBM's OWN default
        # interaction-count heuristic (by simply not overriding
        # 'interactions' at all - interpret's own default, currently
        # '5x', applies) rather than being interpreted as "unlimited".
        _interactions_arg = 'all_default' if max_interaction_features == 'all' else int(max_interaction_features)

    _ebm_kwargs = dict(
        learning_rate=learning_rate, max_leaves=max_leaves, min_samples_leaf=min_samples_leaf,
        outer_bags=outer_bags, random_state=0,
        n_jobs=(_resources['n_jobs'] if _resources['n_jobs'] not in (None, 0) else 1),
    )
    if _interactions_arg != 'all_default':
        _ebm_kwargs['interactions'] = _interactions_arg
    ebm = _ebm_cls(**_ebm_kwargs)
    ebm.fit(train_x, train_y)

    predicted = np.asarray(ebm.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(ebm.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(ebm.predict(train_x)).ravel()

    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]

    effect = _ebm_effect(ebm, train_x.columns)

    if get_interaction == True and _interactions_arg != 0:
        interaction_sample = ebm_pairwise_interactions(ebm, train_x.columns)
        interaction_sample = top_select(interaction_sample, 'percentage', threshold)
    else:
        interaction_sample = pd.DataFrame()

    return r, mse, effect, interaction_sample, predicted, predicted_valid, predicted_train
