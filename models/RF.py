from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import shap

from pipeline_utils import get_active_compute_resources
from models.interaction_extraction import tree_shap_interactions, h_statistic_interactions, top_select


def _random_forest_regressor_cls(use_gpu):
    """Return (RandomForestRegressor class, is_gpu) - cuML's GPU-backed
    RandomForestRegressor when `use_gpu` is True AND cuML is importable,
    else scikit-learn's own CPU implementation. Never raises: any failure
    importing cuML (not installed, no compatible GPU/driver, etc.) falls
    back to CPU silently, so a CPU-only node behaves exactly as before
    this option existed (Phase 2, Requirement 6)."""
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestRegressor as CumlRF
            return CumlRF, True
        except Exception as exc:
            print(f"[RF] USE_GPU_SKLEARN requested but cuML is unavailable "
                  f"({exc}) - falling back to scikit-learn's CPU RandomForestRegressor.")
    return RandomForestRegressor, False

# Exact pairwise SHAP interaction values require building an M x M matrix
# (M = number of markers) for every sampled test row, and the triu-mask /
# stack() step afterwards is itself an O(M^2) operation. With ~8,000 markers
# that is a 64-million-cell matrix and a ~32-million-row table - this is what
# turns the interaction step into a multi-day run.
#
# Since only the top `threshold`% of interactions are ever kept anyway, by
# default we restrict the *computation* itself to the top-importance markers
# (from the already-fitted forest) before running SHAP, rather than every
# marker in the dataset. This is the standard two-stage strategy for
# large-scale epistasis screening: filter candidates by main-effect
# importance first, then test pairwise interactions only among the
# shortlist. Predictions and per-marker effects (returned separately below)
# are unaffected - they still use the full model trained on every marker;
# only the interaction search is narrowed. The cap itself is user-configurable
# via params[8] (GUI: "Max markers considered for interaction search").


def RF(train, valid, test, params):
    
    estimators = params[0]
    features_max = params[1]
    sample_max = params[2]
    # Tree-complexity controls. max_depth is coerced to int (or None) since
    # the GUI field it comes from resolves to a float, but sklearn requires
    # max_depth to be an int or None specifically.
    max_depth = int(params[3]) if params[3] is not None else None
    min_samples_leaf = params[4]
    get_interaction = params[5]
    shapley_num = params[6]
    threshold = params[7]
    max_interaction_features = params[8]
    # Requirements.md item 3: selectable between the original pairwise-SHAP
    # route (default) and Friedman's H-statistic (model-agnostic - the same
    # route SVR/KNN already use via models.interaction_extraction.
    # h_statistic_interactions()). Appended-only, read defensively so a
    # pre-existing, shorter HPARAMETERS['RF'] list keeps reproducing this
    # file's original behaviour unchanged.
    interaction_method = params[9] if len(params) > 9 else 'pairwise_shap'
    # Update ID ver4-6, R1/R1b (blueprint §2.4/§2.10.4): appended, read
    # defensively (I5) - only actually used in the 'friedman_h' branch
    # below (see that branch's own comment); the default 'pairwise_shap'
    # route is unaffected by these three fields entirely.
    interaction_screen = params[10] if len(params) > 10 else 'off'
    interaction_screen_top = params[11] if len(params) > 11 else 2.0
    interaction_grid_resolution = params[12] if len(params) > 12 else 3
    # Requirements.md item 2 (cross-model H-index result diversity): a
    # DEDICATED background-sample-size field for the 'friedman_h' route,
    # independent of `shapley_num` above (which is sized for the exact
    # 'pairwise_shap' route's own explained-row count, not for an
    # averaged H-statistic - see hparam_specs.py's own new field for the
    # full rationale). Read defensively (I5) - a config predating this
    # field falls back to reusing `shapley_num`, reproducing this file's
    # previous 'friedman_h' behaviour exactly.
    interaction_h_background = params[13] if len(params) > 13 else None
    
    #Split the data sets into x and y here as specified in the original code
    train_x, train_y = train.iloc[:,:-1], train.iloc[:,-1]    
    if valid.shape[0] != 0:
        valid_x, valid_y = valid.iloc[:,:-1], valid.iloc[:,-1]
    test_x, test_y = test.iloc[:,:-1], test.iloc[:,-1]

    # Phase 2, Requirement 6: n_jobs and an optional cuML GPU backend are
    # resolved from the run's shared compute-resource settings
    # (pipeline_utils.resolve_compute_resources()), not hardcoded - a
    # CPU-only node is completely unaffected (falls back to scikit-learn's
    # own multi-threaded n_jobs=-1, exactly as before this option existed).
    _resources = get_active_compute_resources()
    _rf_cls, _is_gpu = _random_forest_regressor_cls(_resources['use_gpu_sklearn'])

    #Develop & evaluate a model here as specified in the original code
    if _is_gpu:
        # cuML's RandomForestRegressor has a narrower constructor surface
        # than scikit-learn's - unsupported CPU-only knobs are simply
        # omitted rather than raising.
        rf = _rf_cls(n_estimators=estimators, random_state=0,
                     max_features=features_max, max_depth=max_depth,
                     min_samples_leaf=min_samples_leaf)
    else:
        rf = _rf_cls(n_estimators=estimators, random_state=0, max_samples=sample_max,
                     max_features=features_max, max_depth=max_depth,
                     min_samples_leaf=min_samples_leaf, n_jobs=_resources['n_jobs'])
    rf.fit(train_x, train_y)
    
    predicted = np.asarray(rf.predict(test_x)).ravel()
    if valid.shape[0] != 0:
        predicted_valid = np.asarray(rf.predict(valid_x)).ravel()
    else:
        predicted_valid = []
    predicted_train = np.asarray(rf.predict(train_x)).ravel()
    _feature_importances = np.asarray(rf.feature_importances_).ravel()

    #Calculate the metrics
    actual_test = test_y.values.tolist()
    mse = mean_squared_error(actual_test, predicted)
    r = pearsonr(actual_test, predicted)[0]
    
    #Extract interactions - the SHAP interaction search always uses a
    # CPU (scikit-learn) forest, regardless of whether the main fit above
    # used a GPU backend: shap.TreeExplainer requires a scikit-learn-
    # compatible tree structure that cuML's GPU forest does not expose.
    if get_interaction == True:
        # Defensive clamp: avoid shap.sample() erroring if shapley_num exceeds
        # the number of available test rows.
        shapley_num = min(shapley_num, test_x.shape[0])

        n_features = train_x.shape[1]
        if max_interaction_features != 'all' and n_features > max_interaction_features:
            # Refit a lightweight forest on just the top-importance markers so
            # the SHAP interaction matrix is max_interaction_features x
            # max_interaction_features instead of n_features x n_features.
            top_features = train_x.columns[
                np.argsort(_feature_importances)[::-1][:max_interaction_features]
            ]
            rf_interaction = RandomForestRegressor(n_estimators = estimators, random_state = 0,
                                                     max_samples=sample_max, max_features=features_max,
                                                     max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                     n_jobs=_resources['n_jobs'])
            rf_interaction.fit(train_x[top_features], train_y)
            interaction_test_x = test_x[top_features]
        elif _is_gpu and interaction_method != 'friedman_h':
            # 'all' markers, but the main fit was GPU-backed - still need a
            # CPU forest for shap.TreeExplainer; refit on every marker.
            # Requirements.md item 3: Friedman's H-statistic is fully
            # model-agnostic (needs only a plain .predict callable - see
            # h_statistic_interactions()'s own docstring), so it has no such
            # requirement and this refit is skipped for it below.
            rf_interaction = RandomForestRegressor(n_estimators=estimators, random_state=0,
                                                    max_samples=sample_max, max_features=features_max,
                                                    max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                    n_jobs=_resources['n_jobs'])
            rf_interaction.fit(train_x, train_y)
            interaction_test_x = test_x
        else:
            # 'all' - use every marker, no cap
            rf_interaction = rf
            interaction_test_x = test_x

        if interaction_method == 'friedman_h':
            # Requirements.md item 3: Friedman's H-statistic (Friedman &
            # Popescu, 2008) via models.interaction_extraction.
            # h_statistic_interactions() - the same model-agnostic route
            # SVR/KNN already use for their own interaction search, reusing
            # RF's own already-fitted forest and importance-based shortlist
            # above (RF has a real importance measure, unlike SVR/KNN, which
            # fall back to a correlation-based proxy).
            interaction_columns = list(interaction_test_x.columns)
            interaction_pairs = [
                (a, b) for a in range(len(interaction_columns)) for b in range(a + 1, len(interaction_columns))
            ]
            # Requirements.md item 2: use the DEDICATED
            # `interaction_h_background` field when this config has it
            # (properly sized to match SVR/KNN/RKHS's own background,
            # default 100 - see hparam_specs.py's own new field), rather
            # than always reusing `shapley_num` (30 by default - tuned
            # for the 'pairwise_shap' route's exact per-row explanation,
            # not for H^2's own averaged estimate). A config predating
            # this field (`interaction_h_background is None`) keeps the
            # exact previous behaviour.
            _h_background = (
                shapley_num if interaction_h_background is None else interaction_h_background
            )
            interaction_sample = h_statistic_interactions(
                rf_interaction.predict, interaction_test_x, interaction_columns,
                pairs=interaction_pairs, n_background=min(_h_background, interaction_test_x.shape[0]),
                n_jobs=_resources['n_jobs'],
                # Update ID ver4-6, R1/R1b: forwarded ONLY here, the
                # 'friedman_h' branch - 'pairwise_shap' below never calls
                # h_statistic_interactions() at all, so these three
                # fields have no meaning for it. See models/SVR.py's own
                # identical note for the percentage -> fraction conversion.
                screen=(interaction_screen if interaction_screen != 'off' else None),
                screen_keep=(float(interaction_screen_top) / 100.0 if interaction_screen_top != 'all' else 1.0),
                grid_resolution=interaction_grid_resolution,
            )
        else:
            # Update ID ver4-5, R2 (blueprint §4.4): the fit/explain/reduce/
            # triangle-select recipe below used to be implemented inline here
            # (and, separately, in models/GAT_prior_knowledge.py and
            # Preprocess/data_driven_prior_network.py) - all three now share
            # one implementation, models.interaction_extraction.
            # tree_shap_interactions(). reduce='abs_then_sum' is this file's
            # OWN pre-existing reduction order, passed explicitly so this
            # redirection is a numeric no-op (see that function's own
            # docstring). The row-fan-out (pipeline_utils.parallel_shap_values)
            # is unchanged - it is called FROM inside tree_shap_interactions()
            # now, not from here directly.
            interaction_sample = tree_shap_interactions(
                rf_interaction, shap.sample(interaction_test_x, shapley_num),
                interaction_test_x.columns, n_jobs=_resources['n_jobs'], reduce='abs_then_sum',
            )

        # RF's own pre-existing top-N% filter - moved, unchanged, into the
        # shared top_select() helper (models.interaction_extraction) so
        # its 'percentage' semantics are defined in exactly one place;
        # this call reproduces the original inline
        # `interaction_sample[interaction_sample['value'] >
        # interaction_sample['value'].quantile((1-(threshold/100)))]` line
        # exactly (strict '>', 'all' = no filtering).
        interaction_sample = top_select(interaction_sample, 'percentage', threshold)
    else:
        interaction_sample = pd.DataFrame()
    
    return r, mse, pd.DataFrame(_feature_importances).T, interaction_sample, predicted, predicted_valid, predicted_train

    
