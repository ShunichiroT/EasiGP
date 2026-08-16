"""
EasiGP - hparam_specs.py
--------------------------------------------------------------------------
Single source of truth for every prediction model's positional
HPARAMETERS[model] layout - moved verbatim out of main_app.py's own
HPARAM_SPECS (see main_app.py: `from hparam_specs import HPARAM_SPECS`
replaces the block that used to be defined inline there), with 'tunable'
ranges merged onto the fields that actually affect model fit.

This is a MECHANICAL extraction, not a retyped reconstruction: every
label/type/default/choices/depends_on/help string below is byte-identical
to what main_app.py's HPARAM_SPECS contained, verified by round-tripping
through Python's exec() + dict comparison rather than hand-copied - the
only thing added on top is the 'tunable' key on specific fields.
main_app.py's render_field()/resolve_field() only ever read
type/label/default/choices/combo_state/depends_on, so this extra key is
invisible to them - nothing about GUI rendering or config resolution
changes because of it.

WHY SOME FIELDS HAVE NO 'tunable' KEY
--------------------------------------------------------------------------
Only fields that change what the model actually FITS are eligible. Every
`depends_on`-gated field (SHAP sample counts, interaction/attention
thresholds, "return marker effect?" checkboxes, R model Shapley-refit MCMC
settings, etc.) controls an explainability side-computation, not the fit
itself, and is deliberately left untunable - tuning candidates always run
with these forced off for speed (see
models/hyperparameter_tuning.py:_disable_explainability), and the final
confirmatory fit restores the user's real settings for them.
GAT_biological_prior_knowledge's file-path/config fields (network JSON,
gene-location CSV, marker-info CSV, coordinate unit, mediated-edge
settings) are also never tunable - they identify WHICH biological network
to use, not a numeric hyperparameter of the fit.

'tunable' shapes, matching each field's existing 'type':
  int / float           -> {'low': ..., 'high': ..., 'step': ... (optional, grid only)}
  bool                   -> choices are always [True, False] (not used here - no
                            bool field is tuned, since every bool field in this
                            schema is itself an explainability toggle)
  str (+ choices)         -> {'choices': [...]} (a subset of, or identical to,
                            the field's own 'choices')
  rf_max_features          -> {'choices': ['sqrt', 'log2', 0.5, 0.8, 1.0]}
  svr_gamma                -> {'choices': ['scale', 'auto']}
  int_float_or_none       -> {'choices': [None, 5, 10, 20, 40]} (explicit
                            discrete candidates - a mixed None/continuous
                            domain has no single (low, high) representation)

MODELS COVERED
--------------------------------------------------------------------------
Every model in main_app.py's AVAILABLE_MODELS except 'ensemble' (which has
no hyperparameters of its own): rrBLUP, GBLUP, BayesB, RKHS, RF, SVR, KNN,
MLP, GAT_infinitesimal, GAT_fully_connected, GAT_prior_knowledge,
GAT_biological_prior_knowledge.

GAT_infinitesimal_node_level is ALSO included below even though it isn't
in AVAILABLE_MODELS (the GUI doesn't currently expose it as a selectable
model) - genomic_prediction.py's GP() dispatch and its own model file both
fully support it, so it remains usable (including for hyperparameter
tuning) if invoked directly (e.g. a hand-written config or HPC script),
just not from this GUI.
"""

HPARAM_SPECS = {'rrBLUP': [{'label': 'Iteration number',
             'type': 'int',
             'default': 12000,
             'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                     'iterations generally give a more stable, reliable fit, but take longer. '
                     '12000 is a reasonable starting point for genomic prediction.',
             'tunable': {'low': 2000, 'high': 20000, 'step': 2000}},
            {'label': 'Burn-in',
             'type': 'int',
             'default': 2000,
             'help': 'How many of the initial iterations (above) are discarded before averaging, '
                     "to let the model 'warm up' and stop being influenced by its arbitrary "
                     'starting point. Must be smaller than the iteration number.',
             'tunable': {'low': 200, 'high': 5000, 'step': 200}},
            {'label': 'Prior degrees of freedom (df0)',
             'type': 'int',
             'default': 5,
             'help': 'Controls how strongly the prior belief about marker-effect size is held '
                     'before seeing the data. Higher values make the model trust the prior more '
                     '(stronger shrinkage); lower values let the data dominate more quickly. 5 is '
                     "BGLR's own default.",
             'tunable': {'low': 1, 'high': 20, 'step': 1}},
            {'label': 'Expected proportion of variance explained (R2)',
             'type': 'float',
             'default': 0.5,
             'help': "Your best guess at what fraction of the trait's variance the markers explain "
                     'overall - used to set how much shrinkage is applied to each marker effect. '
                     'Higher R2 = less shrinkage (bigger effects allowed); lower R2 = more '
                     "shrinkage (effects pulled closer to zero). 0.5 is BGLR's own default and a "
                     'reasonable starting point if unsure.',
             'tunable': {'low': 0.1, 'high': 0.9}}],
 'BayesB': [{'label': 'Iteration number',
             'type': 'int',
             'default': 12000,
             'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                     'iterations generally give a more stable, reliable fit, but take longer. '
                     '12000 is a reasonable starting point for genomic prediction.',
             'tunable': {'low': 2000, 'high': 20000, 'step': 2000}},
            {'label': 'Burn-in',
             'type': 'int',
             'default': 2000,
             'help': 'How many of the initial iterations (above) are discarded before averaging, '
                     "to let the model 'warm up' and stop being influenced by its arbitrary "
                     'starting point. Must be smaller than the iteration number.',
             'tunable': {'low': 200, 'high': 5000, 'step': 200}},
            {'label': 'Prior probability of a nonzero effect (probIn)',
             'type': 'float',
             'default': 0.5,
             'help': "BayesB's defining setting: the assumed proportion of markers with a real, "
                     'nonzero effect on the trait. Lower values (e.g. 0.05-0.1) assume only a few '
                     'markers matter (sparser, more like Bayesian variable selection); higher '
                     'values behave more like ridge regression, where most markers contribute a '
                     "little. 0.5 is BGLR's own default.",
             'tunable': {'low': 0.01, 'high': 0.9}},
            {'label': 'Prior counts (counts)',
             'type': 'int',
             'default': 10,
             'help': "How strongly the 'probIn' belief above is held before seeing the data - "
                     'higher values make BGLR trust that prior more strongly; lower values let the '
                     "data override it more easily. 10 is BGLR's own default.",
             'tunable': {'low': 2, 'high': 100, 'step': 2}}],
 'GBLUP': [{'label': 'Iteration number',
            'type': 'int',
            'default': 12000,
            'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                    'iterations generally give a more stable, reliable fit, but take longer. 12000 '
                    'is a reasonable starting point for genomic prediction.',
            'tunable': {'low': 2000, 'high': 20000, 'step': 2000}},
           {'label': 'Burn-in',
            'type': 'int',
            'default': 2000,
            'help': 'How many of the initial iterations (above) are discarded before averaging, to '
                    "let the model 'warm up' and stop being influenced by its arbitrary starting "
                    'point. Must be smaller than the iteration number.',
            'tunable': {'low': 200, 'high': 5000, 'step': 200}},
           {'label': 'Return marker effect?',
            'type': 'bool',
            'default': False,
            'help': 'If checked, also estimates how much each marker contributes to the prediction '
                    '(via Shapley scores), in addition to the prediction itself. This is slower - '
                    'the settings below only apply when this is checked.'},
           {'label': 'Number of samples for Shapley scores',
            'type': 'int',
            'default': 30,
            'depends_on': (2, True),
            'help': 'How many test individuals to compute marker-effect (Shapley) scores for. More '
                    'individuals give a more representative picture of marker importance across '
                    'the population, but take longer.'},
           {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
            'type': 'int_or_all',
            'default': 500,
            'depends_on': (2, True),
            'help': 'Only the top markers (ranked by correlation with the trait) are scored; every '
                    "other marker is reported as 0. Fewer markers = much faster. 'all' scores "
                    'every marker but can take a very long time on datasets with thousands of '
                    'markers.'},
           {'label': 'Number of iterations for Shapley scores',
            'type': 'int',
            'default': 200,
            'depends_on': (2, True),
            'help': 'Each Shapley perturbation test re-fits the whole model from scratch, so this '
                    'controls how many MCMC iterations that re-fit uses - separate from, and much '
                    "smaller than, the main 'Iteration number' above. More iterations = more "
                    'stable scores but much slower, since this re-fit happens many times. 200 is a '
                    'reasonable balance for large datasets.'},
           {'label': 'Burn-in for Shapley scores',
            'type': 'int',
            'default': 50,
            'depends_on': (2, True),
            'help': "How many of the Shapley re-fit's MCMC iterations (above) are discarded as "
                    "'warm-up' before averaging - must be smaller than that value. Separate from, "
                    "and much smaller than, the main 'Burn-in' above."}],
 'RKHS': [{'label': 'Iteration number',
           'type': 'int',
           'default': 12000,
           'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                   'iterations generally give a more stable, reliable fit, but take longer. 12000 '
                   'is a reasonable starting point for genomic prediction.',
           'tunable': {'low': 2000, 'high': 20000, 'step': 2000}},
          {'label': 'Burn-in',
           'type': 'int',
           'default': 2000,
           'help': 'How many of the initial iterations (above) are discarded before averaging, to '
                   "let the model 'warm up' and stop being influenced by its arbitrary starting "
                   'point. Must be smaller than the iteration number.',
           'tunable': {'low': 200, 'high': 5000, 'step': 200}},
          {'label': 'Kernel bandwidth (h)',
           'type': 'float',
           'default': 1.0,
           'help': 'Controls how quickly similarity between two samples drops off with genetic '
                   'distance. Higher h = only very close samples are treated as similar (more '
                   'locally-focused); lower h = more distant samples are still treated as somewhat '
                   'similar (smoother). 1 is often used as the midpoint.',
           'tunable': {'low': 0.1, 'high': 5.0}},
          {'label': 'Return marker effect?',
           'type': 'bool',
           'default': False,
           'help': 'If checked, also estimates how much each marker contributes to the prediction '
                   '(via Shapley scores), in addition to the prediction itself. This is slower - '
                   'the settings below only apply when this is checked.'},
          {'label': 'Number of samples for Shapley scores',
           'type': 'int',
           'default': 30,
           'depends_on': (3, True),
           'help': 'How many test individuals to compute marker-effect (Shapley) scores for. More '
                   'individuals give a more representative picture of marker importance across the '
                   'population, but take longer.'},
          {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
           'type': 'int_or_all',
           'default': 500,
           'depends_on': (3, True),
           'help': 'Only the top markers (ranked by correlation with the trait) are scored; every '
                   "other marker is reported as 0. Fewer markers = much faster. 'all' scores every "
                   'marker but can take a very long time on datasets with thousands of markers.'},
          {'label': 'Number of iterations for Shapley scores',
           'type': 'int',
           'default': 200,
           'depends_on': (3, True),
           'help': 'Each Shapley perturbation test re-fits the whole model from scratch, so this '
                   'controls how many MCMC iterations that re-fit uses - separate from, and much '
                   "smaller than, the main 'Iteration number' above. More iterations = more stable "
                   'scores but much slower, since this re-fit happens many times. 200 is a '
                   'reasonable balance for large datasets.'},
          {'label': 'Burn-in for Shapley scores',
           'type': 'int',
           'default': 50,
           'depends_on': (3, True),
           'help': "How many of the Shapley re-fit's MCMC iterations (above) are discarded as "
                   "'warm-up' before averaging - must be smaller than that value. Separate from, "
                   "and much smaller than, the main 'Burn-in' above."}],
 'RF': [{'label': 'Tree number',
         'type': 'int',
         'default': 1000,
         'help': 'How many individual decision trees to average together. More trees usually give '
                 'steadier, more reliable predictions, at the cost of longer runtime - returns '
                 'diminish well before 1000 for most datasets.',
         'tunable': {'low': 100, 'high': 2000, 'step': 100}},
        {'label': 'Maximum features per tree',
         'type': 'rf_max_features',
         'default': '1.0',
         'choices': ['sqrt', 'log2', 'None'],
         'combo_state': 'normal',
         'help': "How many markers each tree is allowed to consider at every split. 'sqrt' uses "
                 "the value of the square root of the total marker count, and 'log2' uses log base "
                 '2 of it - both use only a small random subset per split (more diversity between '
                 "trees, often better for many markers). 'None' or a custom number lets each split "
                 'consider all (or more) markers.',
         'tunable': {'choices': ['sqrt', 'log2', 0.5, 0.8, 1.0]}},
        {'label': 'Maximum samples per tree',
         'type': 'int_float_or_none',
         'default': None,
         'help': 'How many individuals (out of the training set) each tree is trained on, drawn '
                 "with replacement. 'None' (the default) uses as many as there are training "
                 'individuals. Lowering this makes trees more different from one another, which '
                 'can help or hurt depending on the dataset.',
         'tunable': {'choices': [None, 0.5, 0.7, 0.9]}},
        {'label': 'Maximum tree depth',
         'type': 'int_float_or_none',
         'default': None,
         'help': "Limits how many splits deep each tree can grow. 'None' (the default) lets trees "
                 'grow until every leaf is pure or too small to split further - this can overfit '
                 'on noisy data. A smaller number (e.g. 5-15) gives simpler, more regularised '
                 'trees.',
         'tunable': {'choices': [None, 5, 10, 15, 25]}},
        {'label': 'Minimum samples per leaf in each tree',
         'type': 'int',
         'default': 1,
         'help': 'The smallest number of samples allowed in a leaf node. 1 (the default) lets '
                 'trees fit very fine-grained detail; raising this (e.g. 5-20) smooths predictions '
                 'and reduces overfitting, especially with noisy phenotypes.',
         'tunable': {'low': 1, 'high': 20, 'step': 1}},
        {'label': 'Return marker effect for interactions?',
         'type': 'bool',
         'default': True,
         'help': 'If checked, also searches for pairs of markers that interact with each other '
                 '(beyond what each marker alone explains), in addition to the prediction itself. '
                 'This is slower - the settings below only apply when this is checked.'},
        {'label': 'Number of samples for marker effect interactions',
         'type': 'int',
         'default': 30,
         'depends_on': (5, True),
         'help': 'How many test individuals to search for marker-pair interactions in. More '
                 'individuals give a more representative picture across the population, but take '
                 'longer.'},
        {'label': 'Output only the top N% of interactions ("all" for everything)',
         'type': 'top_pct',
         'default': 'all',
         'depends_on': (5, True),
         'help': 'Only keep the strongest interactions found, as a percentage of all pairs tested '
                 "- e.g.0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                 'a very large table for datasets with many markers.'},
        {'label': 'Max markers considered for interaction search ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (5, True),
         'help': 'Only the top markers (ranked by importance) are checked for pairwise '
                 'interactions; every other marker pair is left out. Fewer markers = much faster. '
                 "'all' checks every possible pair but can take a very long time on datasets with "
                 'thousands of markers.'}],
 'SVR': [{'label': 'Kernel type',
          'type': 'str',
          'default': 'rbf',
          'choices': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
          'combo_state': 'readonly',
          'help': "The shape of similarity function used to compare individuals. 'rbf' (the "
                  'default) works well in most cases and can capture curved, non-linear '
                  "relationships; 'linear' is simpler and assumes marker effects add up directly; "
                  "'poly'/'sigmoid' are other curved shapes worth trying if 'rbf' underperforms.",
          'tunable': {'choices': ['linear', 'poly', 'rbf', 'sigmoid']}},
         {'label': 'Epsilon',
          'type': 'float',
          'default': 0.5,
          'help': 'A margin of error the model is allowed to ignore - predictions within epsilon '
                  "of the true value aren't penalised at all. Larger epsilon gives a simpler, less "
                  'sensitive model; smaller epsilon tries to fit the data more closely.',
          'tunable': {'low': 0.01, 'high': 2.0}},
         {'label': 'Constraint',
          'type': 'float',
          'default': 1.0,
          'help': 'Controls the trade-off between fitting the training data closely and keeping '
                  'the model simple. Higher values fit the training data harder (risk of '
                  'overfitting); lower values favour a smoother, more general model.',
          'tunable': {'low': 0.01, 'high': 100.0}},
         {'label': 'Dimension for poly kernel',
          'type': 'int',
          'default': 3,
          'depends_on': (0, 'poly'),
          'help': "Only used when Kernel type is 'poly'. Higher values allow more complex, curvier "
                  'relationships between markers and the trait, but are more prone to overfitting.',
          'tunable': {'low': 2, 'high': 5, 'step': 1}},
         {'label': 'Gamma',
          'type': 'svr_gamma',
          'default': 'scale',
          'choices': ['scale', 'auto'],
          'combo_state': 'normal',
          'help': "How far the influence of a single individual reaches (for the 'rbf', 'poly', "
                  "and 'sigmoid' kernels). Higher gamma = each individual only influences its "
                  'closest neighbours (can overfit); lower gamma = influence reaches further. '
                  "'scale' (the default) picks a sensible value automatically based on your data.",
          'tunable': {'choices': ['scale', 'auto']}},
         {'label': 'Independent term (coef0)',
          'type': 'float',
          'default': 0.0,
          'help': "Only used by the 'poly' and 'sigmoid' kernels (ignored for 'rbf'/'linear'). "
                  "Shifts the kernel function by a constant - 0.0 is sklearn's own default.",
          'tunable': {'low': -1.0, 'high': 1.0}},
         {'label': 'Return marker effect?',
          'type': 'bool',
          'default': False,
          'help': 'If checked, also estimates how much each marker contributes to the prediction '
                  '(via Shapley scores), in addition to the prediction itself. This is slower - '
                  'the settings below only apply when this is checked.'},
         {'label': 'Number of samples for Shapley scores',
          'type': 'int',
          'default': 30,
          'depends_on': (6, True),
          'help': 'How many test individuals to compute marker-effect (Shapley) scores for. More '
                  'individuals give a more representative picture of marker importance across the '
                  'population, but take longer.'},
         {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
          'type': 'int_or_all',
          'default': 500,
          'depends_on': (6, True),
          'help': 'Only the top markers (ranked by correlation with the trait) are scored; every '
                  "other marker is reported as 0. Fewer markers = much faster. 'all' scores every "
                  'marker but can take a very long time on datasets with thousands of markers.'},
         {'label': 'Background sample size for Shapley scores',
          'type': 'int',
          'default': 50,
          'depends_on': (6, True),
          'help': "A small reference set of samples used as a 'typical' baseline when working out "
                  "each marker's contribution. Smaller = faster; larger = smoother, more stable "
                  "scores but slower. 50 is a good default - you shouldn't normally need to raise "
                  'this much.'},
         {'label': 'Number of coalition samples for Shapley scores (nsamples)',
          'type': 'int',
          'default': 200,
          'depends_on': (6, True),
          'help': "How many random combinations of markers are tried out to estimate each marker's "
                  'contribution to a prediction. Higher = more accurate but slower; lower = faster '
                  'but noisier scores. 200 is a reasonable balance for large datasets.'}],
 'KNN': [{'label': 'Number of neighbours',
          'type': 'int',
          'default': 5,
          'help': 'How many of the most genetically similar training individuals are averaged '
                  "together to predict each new individual's trait. Fewer neighbours can pick up "
                  'more local detail but are noisier; more neighbours give a smoother, more stable '
                  'prediction but can blur out real differences.',
          'tunable': {'low': 1, 'high': 30, 'step': 1}},
         {'label': 'Neighbour weighting',
          'type': 'str',
          'default': 'uniform',
          'choices': ['uniform', 'distance'],
          'combo_state': 'readonly',
          'help': "'uniform' treats every one of the k nearest neighbours equally when averaging "
                  "their phenotypes. 'distance' weights closer neighbours more heavily than "
                  'farther ones - often a better fit when genetic distance varies a lot within the '
                  'neighbourhood.',
          'tunable': {'choices': ['uniform', 'distance']}},
         {'label': 'Distance metric power (p)',
          'type': 'int',
          'default': 2,
          'help': 'Which Minkowski distance is used to find neighbours: p=1 is Manhattan distance, '
                  'p=2 (the default) is ordinary Euclidean distance. Higher values increasingly '
                  'emphasise the single largest per-marker difference between two samples.',
          'tunable': {'low': 1, 'high': 3, 'step': 1}},
         {'label': 'Return marker effect?',
          'type': 'bool',
          'default': False,
          'help': 'If checked, also estimates how much each marker contributes to the prediction '
                  '(via Shapley scores), in addition to the prediction itself. This is slower - '
                  'the settings below only apply when this is checked.'},
         {'label': 'Number of samples for Shapley scores',
          'type': 'int',
          'default': 30,
          'depends_on': (3, True),
          'help': 'How many test individuals to compute marker-effect (Shapley) scores for. More '
                  'individuals give a more representative picture of marker importance across the '
                  'population, but take longer.'},
         {'label': 'Max markers considered for Shapley scores ("all" for every marker)',
          'type': 'int_or_all',
          'default': 500,
          'depends_on': (3, True),
          'help': 'Only the top markers (ranked by correlation with the trait) are scored; every '
                  "other marker is reported as 0. Fewer markers = much faster. 'all' scores every "
                  'marker but can take a very long time on datasets with thousands of markers.'},
         {'label': 'Background sample size for Shapley scores',
          'type': 'int',
          'default': 50,
          'depends_on': (3, True),
          'help': "A small reference set of samples used as a 'typical' baseline when working out "
                  "each marker's contribution. Smaller = faster; larger = smoother, more stable "
                  "scores but slower. 50 is a good default - you shouldn't normally need to raise "
                  'this much.'},
         {'label': 'Number of coalition samples for Shapley scores (nsamples)',
          'type': 'int',
          'default': 200,
          'depends_on': (3, True),
          'help': "How many random combinations of markers are tried out to estimate each marker's "
                  'contribution to a prediction. Higher = more accurate but slower; lower = faster '
                  'but noisier scores. 200 is a reasonable balance for large datasets.'}],
 'MLP': [{'label': 'Neuron numbers',
          'type': 'int',
          'default': 30,
          'help': 'How many units are in the (first) hidden layer. More neurons let the network '
                  'represent more complex patterns, but need more data to train reliably and are '
                  'more prone to overfitting.',
          'tunable': {'low': 8, 'high': 256, 'step': 8}},
         {'label': 'Dropout',
          'type': 'float',
          'default': 0,
          'help': 'The fraction of hidden-layer connections randomly switched off during each '
                  'training step, as a safeguard against overfitting. 0 disables this; typical '
                  'values are 0.1-0.5 if overfitting is a concern.',
          'tunable': {'low': 0.0, 'high': 0.6}},
         {'label': 'Learning rate',
          'type': 'float',
          'default': 0.0001,
          'help': 'How big a step the network takes when updating its weights after each batch. '
                  'Too high can make training unstable or fail to settle down; too low makes '
                  'training very slow to improve.',
          'tunable': {'low': 0.0001, 'high': 0.1}},
         {'label': 'Decay',
          'type': 'float',
          'default': 0.0005,
          'help': "A small penalty that discourages the network's weights from growing too large, "
                  'as another safeguard against overfitting. 0 disables it; larger values '
                  'regularise more strongly.',
          'tunable': {'low': 0.0, 'high': 0.01}},
         {'label': 'Epoch',
          'type': 'int',
          'default': 200,
          'help': 'How many times the network passes over the entire training set. More epochs let '
                  'it learn more, but too many can start memorising noise in the training data '
                  'rather than the underlying trend.',
          'tunable': {'low': 10, 'high': 200, 'step': 10}},
         {'label': 'Batch size',
          'type': 'int',
          'default': 8,
          'help': 'How many individuals are processed together before each weight update. Smaller '
                  'batches update more often (noisier but sometimes better generalisation); larger '
                  'batches are faster per epoch but update less often.',
          'tunable': {'low': 4, 'high': 64, 'step': 4}},
         {'label': 'Second hidden layer size',
          'type': 'int_float_or_none',
          'default': None,
          'help': 'Adds an extra hidden layer (with the same dropout rate as the first) before the '
                  "output. 'None' (the default) keeps the original single-hidden-layer network "
                  'unchanged. A deeper network can capture more complex marker interactions, but '
                  'is slower to train and easier to overfit on small datasets.',
          'tunable': {'choices': [None, 16, 32, 64]}},
         {'label': 'Number of samples for Shapley scores',
          'type': 'int',
          'default': 30,
          'help': 'How many test individuals to compute marker-effect (Shapley) scores for. More '
                  'individuals give a more representative picture of marker importance across the '
                  'population, but take longer.'}],
 'GAT_infinitesimal': [{'label': 'Neuron numbers',
                        'type': 'int',
                        'default': 20,
                        'help': 'How many units are in each graph-attention layer. More neurons '
                                'let the network represent more complex patterns, but need more '
                                'data to train reliably and are more prone to overfitting.',
                        'tunable': {'low': 8, 'high': 128, 'step': 8}},
                       {'label': 'Dropout',
                        'type': 'float',
                        'default': 0,
                        'help': 'The fraction of attention connections randomly switched off '
                                'during each training step, as a safeguard against overfitting. 0 '
                                'disables this; typical values are 0.1-0.5 if overfitting is a '
                                'concern.',
                        'tunable': {'low': 0.0, 'high': 0.6}},
                       {'label': 'Learning rate',
                        'type': 'float',
                        'default': 0.01,
                        'help': 'How big a step the network takes when updating its weights after '
                                'each batch. Too high can make training unstable or fail to settle '
                                'down; too low makes training very slow to improve.',
                        'tunable': {'low': 0.0001, 'high': 0.1}},
                       {'label': 'Decay',
                        'type': 'float',
                        'default': 0.0005,
                        'help': "A small penalty that discourages the network's weights from "
                                'growing too large, as another safeguard against overfitting. 0 '
                                'disables it; larger values regularise more strongly.',
                        'tunable': {'low': 0.0, 'high': 0.01}},
                       {'label': 'Epoch',
                        'type': 'int',
                        'default': 40,
                        'help': 'How many times the network passes over the entire training set. '
                                'More epochs let it learn more, but too many can start memorising '
                                'noise in the training data rather than the underlying trend.',
                        'tunable': {'low': 10, 'high': 200, 'step': 10}},
                       {'label': 'Batch size',
                        'type': 'int',
                        'default': 8,
                        'help': 'How many individuals are processed together before each weight '
                                'update. Smaller batches update more often (noisier but sometimes '
                                'better generalisation); larger batches are faster per epoch but '
                                'update less often.',
                        'tunable': {'low': 4, 'high': 64, 'step': 4}},
                       {'label': 'Number of heads',
                        'type': 'int',
                        'default': 1,
                        'help': "How many independent 'attention patterns' the network learns at "
                                'once - each head can focus on a different way markers relate to '
                                'one another. More heads can capture more varied relationships, at '
                                'the cost of a bigger, slower model.',
                        'tunable': {'low': 1, 'high': 8, 'step': 1}},
                       {'label': 'Return marker effect?',
                        'type': 'bool',
                        'default': True,
                        'help': 'If checked, also estimates how much each marker contributes to '
                                'the prediction, in addition to the prediction itself. This is '
                                'slower - the setting below only applies when this is checked.'},
                       {'label': 'Number of samples for Shapley scores',
                        'type': 'int',
                        'default': 30,
                        'depends_on': (7, True),
                        'help': 'How many test individuals to compute marker-effect scores for. '
                                'More individuals give a more representative picture of marker '
                                'importance across the population, but take longer.'}],
 'GAT_fully_connected': [{'label': 'Neuron numbers',
                          'type': 'int',
                          'default': 20,
                          'help': 'How many units are in each graph-attention layer. More neurons '
                                  'let the network represent more complex patterns, but need more '
                                  'data to train reliably and are more prone to overfitting.',
                          'tunable': {'low': 8, 'high': 128, 'step': 8}},
                         {'label': 'Dropout',
                          'type': 'float',
                          'default': 0,
                          'help': 'The fraction of attention connections randomly switched off '
                                  'during each training step, as a safeguard against overfitting. '
                                  '0 disables this; typical values are 0.1-0.5 if overfitting is a '
                                  'concern.',
                          'tunable': {'low': 0.0, 'high': 0.6}},
                         {'label': 'Learning rate',
                          'type': 'float',
                          'default': 0.01,
                          'help': 'How big a step the network takes when updating its weights '
                                  'after each batch. Too high can make training unstable or fail '
                                  'to settle down; too low makes training very slow to improve.',
                          'tunable': {'low': 0.0001, 'high': 0.1}},
                         {'label': 'Decay',
                          'type': 'float',
                          'default': 0.0005,
                          'help': "A small penalty that discourages the network's weights from "
                                  'growing too large, as another safeguard against overfitting. 0 '
                                  'disables it; larger values regularise more strongly.',
                          'tunable': {'low': 0.0, 'high': 0.01}},
                         {'label': 'Epoch',
                          'type': 'int',
                          'default': 40,
                          'help': 'How many times the network passes over the entire training set. '
                                  'More epochs let it learn more, but too many can start '
                                  'memorising noise in the training data rather than the '
                                  'underlying trend.',
                          'tunable': {'low': 10, 'high': 200, 'step': 10}},
                         {'label': 'Batch size',
                          'type': 'int',
                          'default': 8,
                          'help': 'How many individuals are processed together before each weight '
                                  'update. Smaller batches update more often (noisier but '
                                  'sometimes better generalisation); larger batches are faster per '
                                  'epoch but update less often.',
                          'tunable': {'low': 4, 'high': 64, 'step': 4}},
                         {'label': 'Number of heads',
                          'type': 'int',
                          'default': 1,
                          'help': "How many independent 'attention patterns' the network learns at "
                                  'once - each head can focus on a different way markers relate to '
                                  'one another. More heads can capture more varied relationships, '
                                  'at the cost of a bigger, slower model.',
                          'tunable': {'low': 1, 'high': 8, 'step': 1}},
                         {'label': 'Return marker effect?',
                          'type': 'bool',
                          'default': True,
                          'help': 'If checked, also estimates how much each marker contributes to '
                                  'the prediction, in addition to the prediction itself. This is '
                                  'slower - the setting below only applies when this is checked.'},
                         {'label': 'Number of samples for Shapley scores',
                          'type': 'int',
                          'default': 30,
                          'depends_on': (7, True),
                          'help': 'How many test individuals to compute marker-effect scores for. '
                                  'More individuals give a more representative picture of marker '
                                  'importance across the population, but take longer.'}],
 'GAT_prior_knowledge': [{'label': 'Neuron numbers',
                          'type': 'int',
                          'default': 20,
                          'help': 'How many units are in each graph-attention layer. More neurons '
                                  'let the network represent more complex patterns, but need more '
                                  'data to train reliably and are more prone to overfitting.',
                          'tunable': {'low': 8, 'high': 128, 'step': 8}},
                         {'label': 'Dropout',
                          'type': 'float',
                          'default': 0,
                          'help': 'The fraction of attention connections randomly switched off '
                                  'during each training step, as a safeguard against overfitting. '
                                  '0 disables this; typical values are 0.1-0.5 if overfitting is a '
                                  'concern.',
                          'tunable': {'low': 0.0, 'high': 0.6}},
                         {'label': 'Learning rate',
                          'type': 'float',
                          'default': 0.01,
                          'help': 'How big a step the network takes when updating its weights '
                                  'after each batch. Too high can make training unstable or fail '
                                  'to settle down; too low makes training very slow to improve.',
                          'tunable': {'low': 0.0001, 'high': 0.1}},
                         {'label': 'Decay',
                          'type': 'float',
                          'default': 0.0005,
                          'help': "A small penalty that discourages the network's weights from "
                                  'growing too large, as another safeguard against overfitting. 0 '
                                  'disables it; larger values regularise more strongly.',
                          'tunable': {'low': 0.0, 'high': 0.01}},
                         {'label': 'Epoch',
                          'type': 'int',
                          'default': 40,
                          'help': 'How many times the network passes over the entire training set. '
                                  'More epochs let it learn more, but too many can start '
                                  'memorising noise in the training data rather than the '
                                  'underlying trend.',
                          'tunable': {'low': 10, 'high': 200, 'step': 10}},
                         {'label': 'Batch size',
                          'type': 'int',
                          'default': 8,
                          'help': 'How many individuals are processed together before each weight '
                                  'update. Smaller batches update more often (noisier but '
                                  'sometimes better generalisation); larger batches are faster per '
                                  'epoch but update less often.',
                          'tunable': {'low': 4, 'high': 64, 'step': 4}},
                         {'label': 'Number of heads',
                          'type': 'int',
                          'default': 1,
                          'help': "How many independent 'attention patterns' the network learns at "
                                  'once - each head can focus on a different way markers relate to '
                                  'one another. More heads can capture more varied relationships, '
                                  'at the cost of a bigger, slower model.',
                          'tunable': {'low': 1, 'high': 8, 'step': 1}},
                         {'label': 'Selection rate for edges from RF (e.g. 10 = select the top 10% '
                                   'of the          most important edges))',
                          'type': 'float',
                          'default': 10,
                          'help': 'This model first uses a Random Forest to identify which pairs '
                                  'of markers seem to interact, then only connects those pairs in '
                                  'the graph. This setting controls what fraction of all possible '
                                  'marker pairs are kept as connections - a lower rate keeps only '
                                  'the strongest, most selective set of relationships; a higher '
                                  'rate keeps more (noisier) connections.'},
                         {'label': 'Return marker effects?',
                          'type': 'bool',
                          'default': True,
                          'help': 'If checked, also estimates how much each marker contributes to '
                                  'the prediction, in addition to the prediction itself. This is '
                                  'slower - the settings below only apply when this is checked.'},
                         {'label': 'Number of samples for marker effects',
                          'type': 'int',
                          'default': 30,
                          'depends_on': (8, True),
                          'help': 'How many test individuals to compute marker-effect scores for. '
                                  'More individuals give a more representative picture of marker '
                                  'importance across the population, but take longer.'}],
 'GAT_biological_prior_knowledge': [{'label': 'Neuron numbers',
                                     'type': 'int',
                                     'default': 20,
                                     'help': 'How many units are in each graph-attention layer. '
                                             'More neurons let the network represent more complex '
                                             'patterns, but need more data to train reliably and '
                                             'are more prone to overfitting.',
                                     'tunable': {'low': 8, 'high': 128, 'step': 8}},
                                    {'label': 'Dropout',
                                     'type': 'float',
                                     'default': 0,
                                     'help': 'The fraction of attention connections randomly '
                                             'switched off during each training step, as a '
                                             'safeguard against overfitting. 0 disables this; '
                                             'typical values are 0.1-0.5 if overfitting is a '
                                             'concern.',
                                     'tunable': {'low': 0.0, 'high': 0.6}},
                                    {'label': 'Learning rate',
                                     'type': 'float',
                                     'default': 0.01,
                                     'help': 'How big a step the network takes when updating its '
                                             'weights after each batch. Too high can make training '
                                             'unstable or fail to settle down; too low makes '
                                             'training very slow to improve.',
                                     'tunable': {'low': 0.0001, 'high': 0.1}},
                                    {'label': 'Decay',
                                     'type': 'float',
                                     'default': 0.0005,
                                     'help': "A small penalty that discourages the network's "
                                             'weights from growing too large, as another safeguard '
                                             'against overfitting. 0 disables it; larger values '
                                             'regularise more strongly.',
                                     'tunable': {'low': 0.0, 'high': 0.01}},
                                    {'label': 'Epoch',
                                     'type': 'int',
                                     'default': 40,
                                     'help': 'How many times the network passes over the entire '
                                             'training set. More epochs let it learn more, but too '
                                             'many can start memorising noise in the training data '
                                             'rather than the underlying trend.',
                                     'tunable': {'low': 10, 'high': 200, 'step': 10}},
                                    {'label': 'Batch size',
                                     'type': 'int',
                                     'default': 8,
                                     'help': 'How many individuals are processed together before '
                                             'each weight update. Smaller batches update more '
                                             'often (noisier but sometimes better generalisation); '
                                             'larger batches are faster per epoch but update less '
                                             'often.',
                                     'tunable': {'low': 4, 'high': 64, 'step': 4}},
                                    {'label': 'Number of heads',
                                     'type': 'int',
                                     'default': 1,
                                     'help': "How many independent 'attention patterns' the "
                                             'network learns at once - each head can focus on a '
                                             'different way genes relate to one another. More '
                                             'heads can capture more varied relationships, at the '
                                             'cost of a bigger, slower model.',
                                     'tunable': {'low': 1, 'high': 8, 'step': 1}},
                                    {'label': 'Network JSON path',
                                     'type': 'file_path',
                                     'default': '',
                                     'help': 'The gene-interaction network JSON (uploaded, or '
                                             "generated with FLASH-P) - set up on the 'Biological "
                                             "Prior Network' tab, which fills this in "
                                             'automatically once resolved there.'},
                                    {'label': 'Gene location CSV path',
                                     'type': 'file_path',
                                     'default': '',
                                     'help': 'The curated gene-location lookup table (columns: '
                                             'Gene_Name, Chromosome, Start_bp/Start_cM, '
                                             'End_bp/End_cM, and optionally AGI_Locus_ID, Source) '
                                             "- also set up on the 'Biological Prior Network' tab. "
                                             'The gene list and gene-gene adjacency graph are '
                                             'built automatically from this file plus the network '
                                             "JSON above, every time this model runs - there's no "
                                             'separate gene list/adjacency file to manage.'},
                                    {'label': 'Marker info CSV path (chromosome, name, start, end)',
                                     'type': 'file_path',
                                     'default': '',
                                     'help': 'Same file structure as Data/MaizeNAM/marker_info.csv '
                                             '- maps every SNP in your genotype file to a '
                                             'chromosome/position, so each SNP can be assigned to '
                                             "whichever gene's window it falls inside. If you're "
                                             'already using the same marker_info.csv on the Circos '
                                             'Plot tab, you can just copy that path here.'},
                                    {'label': 'Coordinate unit',
                                     'type': 'str',
                                     'default': 'bp',
                                     'choices': ['bp', 'cM'],
                                     'combo_state': 'readonly',
                                     'help': 'Must match the unit used in both the gene location '
                                             'CSV and the marker info CSV above.'},
                                    {'label': 'Include mediated edges (through non-gene nodes)',
                                     'type': 'bool',
                                     'default': True,
                                     'help': 'Recovers gene-gene connectivity that only exists via '
                                             'a hormone/metabolite/protein-complex node in between '
                                             '(e.g. GeneA -> Auxin -> GeneB), which is common in '
                                             'literature-mined networks. Unchecked keeps only '
                                             'edges directly stated between two genes in the '
                                             'JSON.'},
                                    {'label': 'Max hops for mediated edges',
                                     'type': 'int',
                                     'default': 3,
                                     'depends_on': (11, True),
                                     'help': 'How many non-gene nodes a mediated path is allowed '
                                             'to pass through.'},
                                    {'label': 'Data-driven prior network (merge) config',
                                     'type': 'json',
                                     'default': {'enabled': False},
                                     'help': "Internal, non-editable slot - always overwritten by "
                                             "the 'Biological Prior Network' tab's 'Data-driven "
                                             "prior network' section (requirement 2), never shown "
                                             "or typed into directly. When left disabled (the "
                                             "default), markers outside every gene's window are "
                                             "simply excluded from the graph (their reported "
                                             "effect is exactly 0) - unchanged from the original "
                                             "behaviour. When enabled, this instead carries the "
                                             "RF-filtering (and optional LD-pruning) config used to "
                                             "select markers for a data-driven interaction network, "
                                             "which is merged into the biological network per "
                                             "requirement 3: an RF-selected marker outside every "
                                             "gene becomes its own graph node with real, "
                                             "data-driven edges (not just a self-loop) - see "
                                             "Preprocess/data_driven_prior_network.py."},
                                    {'label': 'Return marker effects?',
                                     'type': 'bool',
                                     'default': True,
                                     'help': 'If checked, also estimates how much each marker '
                                             'contributes to the prediction, in addition to the '
                                             "prediction itself (each gene's estimated effect is "
                                             'broadcast onto every SNP that falls inside it; SNPs '
                                             'outside every gene in the network are reported as '
                                             '0). This is slower - the setting below only applies '
                                             'when this is checked.'},
                                    {'label': 'Number of samples for marker effects',
                                     'type': 'int',
                                     'default': 30,
                                     'depends_on': (14, True),
                                     'help': 'How many test individuals to compute marker-effect '
                                             'scores for. More individuals give a more '
                                             'representative picture of marker importance across '
                                             'the population, but take longer.'}]}

# GAT_infinitesimal_node_level - confirmed directly from
# models/GAT_infinitesimal_node_level.py's own params unpacking: neuron(0),
# dropout(1), lrate(2), decay(3), epoch(4), bsize(5), heads(6), samples(7),
# marker_effect(8) - note samples/marker_effect are swapped relative to
# GAT_infinitesimal (this is that model's genuine positional order, not a
# typo). Not part of main_app.py's own HPARAM_SPECS (see module docstring).
HPARAM_SPECS['GAT_infinitesimal_node_level'] = [
    {'label': 'Neuron numbers', 'type': 'int', 'default': 20,
     'tunable': {'low': 8, 'high': 128, 'step': 8}},
    {'label': 'Dropout', 'type': 'float', 'default': 0,
     'tunable': {'low': 0.0, 'high': 0.6}},
    {'label': 'Learning rate', 'type': 'float', 'default': 0.01,
     'tunable': {'low': 1e-4, 'high': 0.1}},
    {'label': 'Decay', 'type': 'float', 'default': 5e-4,
     'tunable': {'low': 0.0, 'high': 1e-2}},
    {'label': 'Epoch', 'type': 'int', 'default': 40,
     'tunable': {'low': 10, 'high': 200, 'step': 10}},
    {'label': 'Batch size', 'type': 'int', 'default': 8,
     'tunable': {'low': 4, 'high': 64, 'step': 4}},
    {'label': 'Number of heads', 'type': 'int', 'default': 1,
     'tunable': {'low': 1, 'high': 8, 'step': 1}},
    {'label': 'Number of samples for marker effects', 'type': 'int', 'default': 30,
     'depends_on': (8, True)},
    {'label': 'Return marker effect?', 'type': 'bool', 'default': True},
]
