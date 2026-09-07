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
             'tunable': {'low': 0.1, 'high': 0.9}},
            {'label': 'Return marker-pair interactions?',
             'type': 'bool',
             'default': False,
             'help': 'If checked, also searches for pairs of markers that interact with each '
                     'other. rrBLUP is an additive model with no native pairwise-interaction '
                     "computation, so this fits a lightweight surrogate model to rrBLUP's OWN "
                     'predictions and searches THAT for interactions - a useful hint, but an '
                     'APPROXIMATION of rrBLUP itself, not an exact computation. The settings '
                     'below only apply when this is checked.'},
            {'label': 'Max markers considered for interaction search ("all" for every marker)',
             'type': 'int_or_all',
             'default': 500,
             'depends_on': (4, True),
             'help': 'Only the top markers (ranked by correlation with the trait) are checked for '
                     "pairwise interactions; every other marker pair is left out. 'all' checks "
                     'every possible pair but can take a very long time on datasets with '
                     'thousands of markers.'},
            {'label': 'Number of individuals used to explain the surrogate model',
             'type': 'int',
             'default': 50,
             'depends_on': (4, True),
             'help': 'How many training individuals the surrogate model is explained on, when '
                     'estimating each pair\'s interaction strength. More individuals give a more '
                     'representative picture, but take longer.'},
            {'label': 'Output only the top N% of interactions ("all" for everything)',
             'type': 'top_pct',
             'default': 'all',
             'depends_on': (4, True),
             'help': 'Only keep the strongest interactions found, as a percentage of all pairs '
                     "tested - e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, "
                     'which can be a very large table for datasets with many markers.'}],
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
             'tunable': {'low': 2, 'high': 100, 'step': 2}},
            {'label': 'Return marker-pair interactions?',
             'type': 'bool',
             'default': False,
             'help': 'If checked, also searches for pairs of markers that interact with each '
                     'other, via a surrogate model fitted to this model\'s OWN predictions - see '
                     "rrBLUP's identical field for the full explanation of why this is an "
                     'APPROXIMATION, not an exact computation. The settings below only apply when '
                     'this is checked.'},
            {'label': 'Max markers considered for interaction search ("all" for every marker)',
             'type': 'int_or_all',
             'default': 500,
             'depends_on': (4, True),
             'help': 'Only the top markers (ranked by correlation with the trait) are checked for '
                     "pairwise interactions; every other marker pair is left out. 'all' checks "
                     'every possible pair but can take a very long time on datasets with '
                     'thousands of markers.'},
            {'label': 'Number of individuals used to explain the surrogate model',
             'type': 'int',
             'default': 50,
             'depends_on': (4, True),
             'help': 'How many training individuals the surrogate model is explained on, when '
                     'estimating each pair\'s interaction strength. More individuals give a more '
                     'representative picture, but take longer.'},
            {'label': 'Output only the top N% of interactions ("all" for everything)',
             'type': 'top_pct',
             'default': 'all',
             'depends_on': (4, True),
             'help': 'Only keep the strongest interactions found, as a percentage of all pairs '
                     "tested - e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, "
                     'which can be a very large table for datasets with many markers.'}],
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
                    "and much smaller than, the main 'Burn-in' above."},
           {'label': 'Shapley row offset (advanced - internal parallel fan-out)',
            'type': 'int',
            'default': 0,
            'depends_on': (2, True),
            'help': "Internal/advanced setting, not something you normally need to "
                    "change. EasiGP itself uses this to split one task's Shapley computation "
                    "across several worker processes, each explaining a different row range. "
                    "Leave at 0 for a single, ordinary run."},
           {'label': 'Shapley row count (advanced - internal parallel fan-out; -1 = every row)',
            'type': 'int',
            'default': -1,
            'depends_on': (2, True),
            'help': "Internal/advanced setting, paired with the row offset above. "
                    "-1 (default) explains every row exactly as if this setting didn't exist. "
                    "Leave at -1 unless you are deliberately restricting this call to a sub-range "
                    "of test individuals."},
           {'label': 'Return marker-pair interactions?',
            'type': 'bool',
            'default': False,
            'help': 'If checked, also searches for pairs of markers that interact with each '
                    'other, via a surrogate model fitted to this model\'s OWN predictions - GBLUP '
                    'is a kernel model with no native pairwise-interaction computation, so this '
                    'is an APPROXIMATION, not an exact computation. The settings below only apply '
                    'when this is checked.'},
           {'label': 'Max markers considered for interaction search ("all" for every marker)',
            'type': 'int_or_all',
            'default': 500,
            'depends_on': (9, True),
            'help': 'Only the top markers (ranked by correlation with the trait) are checked for '
                    "pairwise interactions; every other marker pair is left out. 'all' checks "
                    'every possible pair but can take a very long time on datasets with '
                    'thousands of markers.'},
           {'label': 'Number of individuals used to explain the surrogate model',
            'type': 'int',
            'default': 50,
            'depends_on': (9, True),
            'help': 'How many training individuals the surrogate model is explained on, when '
                    'estimating each pair\'s interaction strength. More individuals give a more '
                    'representative picture, but take longer.'},
           {'label': 'Output only the top N% of interactions ("all" for everything)',
            'type': 'top_pct',
            'default': 'all',
            'depends_on': (9, True),
            'help': 'Only keep the strongest interactions found, as a percentage of all pairs '
                    "tested - e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, "
                    'which can be a very large table for datasets with many markers.'}],
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
           # Update ID ver4-6, R1.2(b): log-scaled - see this update's
           # Change Summary §4 for the full rationale (held-out surrogate
           # rank correlation on a raw-unit, multi-order-of-magnitude
           # tunable box was measured at -0.059; unit-cube + log scaling
           # on the affected fields restores it to +0.905, blueprint E1).
           'tunable': {'low': 0.1, 'high': 5.0, 'scale': 'log'}},
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
                   "and much smaller than, the main 'Burn-in' above."},
          {'label': 'Shapley row offset (advanced - internal parallel fan-out)',
           'type': 'int',
           'default': 0,
           'depends_on': (3, True),
           'help': "Internal/advanced setting, not something you normally need to "
                   "change. EasiGP itself uses this to split one task's Shapley computation "
                   "across several worker processes, each explaining a different row range. "
                   "Leave at 0 for a single, ordinary run."},
          {'label': 'Shapley row count (advanced - internal parallel fan-out; -1 = every row)',
           'type': 'int',
           'default': -1,
           'depends_on': (3, True),
           'help': "Internal/advanced setting, paired with the row offset above. "
                   "-1 (default) explains every row exactly as if this setting didn't exist. "
                   "Leave at -1 unless you are deliberately restricting this call to a sub-range "
                   "of test individuals."},
          {'label': 'Return marker-pair interactions?',
           'type': 'bool',
           'default': False,
           'help': 'If checked, also searches for pairs of markers that interact with each '
                   'other, via a surrogate model fitted to this model\'s OWN predictions and '
                   'analysed with Friedman\'s H-statistic. Runs across multiple CPU cores automatically (and a '
                   'GPU, if this run has GPU acceleration enabled) to speed this up. The '
                   'settings below only apply when this is checked.'},
          {'label': 'Max markers considered for interaction search ("all" for every marker)',
           'type': 'int_or_all',
           'default': 500,
           'depends_on': (10, True),
           'help': 'Only the top markers (ranked by correlation with the trait) are checked for '
                   "pairwise interactions; every other marker pair is left out. 'all' checks "
                   'every possible pair but can take a very long time on datasets with '
                   'thousands of markers.'},
          {'label': 'Number of individuals used to explain the model',
           'type': 'int',
           'default': 50,
           'depends_on': (10, True),
           'help': 'How many training individuals the surrogate model\'s predictions are '
                   'averaged over, when estimating each pair\'s interaction strength. More '
                   'individuals give a more representative picture, but take longer.'},
          {'label': 'Output only the top N% of interactions ("all" for everything)',
           'type': 'top_pct',
           'default': 'all',
           'depends_on': (10, True),
           'help': 'Only keep the strongest interactions found, as a percentage of all pairs '
                   "tested - e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, "
                   'which can be a very large table for datasets with many markers.'},
          # Update ID ver4-6, R1/R1b (blueprint §5.2/§2.10.4): appended,
          # never inserted (I5) - verified against this list's own
          # length (14 before this edit, so 14/15/16 are genuinely the
          # next free indices, not assumed). Both R1 fields gate an
          # explainability side-computation (the interaction search
          # itself), so neither is 'tunable' - hyperparameter_tuning.
          # _disable_explainability() already forces the controlling
          # 'Return marker-pair interactions?' toggle off during search
          # regardless. `genomic_prediction.py::_r_model_interaction_
          # fields()`/`_r_model_surrogate_interaction()` read these three
          # and populate `surrogate_h_statistic_interactions()`'s own
          # `surrogate_cfg['screen']`/`['screen_keep']`/`['grid_resolution']`.
          # Update ID ver4-8: 'surrogate_shap' removed from 'choices' below
          # (and from every other model's identical field - RF's
          # friedman_h-only variant, SVR, KNN) together with the help text
          # describing it. It remains fully implemented in
          # models/interaction_extraction.py (_VALID_SCREEN_MODES still
          # includes it) - only the GUI-offered choice is gone, so a
          # headless config that sets this positionally to 'surrogate_shap'
          # still works exactly as before. render_field()'s selectbox
          # branch in main_app.py now self-heals any session_state value
          # left over from an older saved GUI state that still points at
          # the removed choice, so this change is safe against stale state
          # files too.
          {'label': 'Interaction pre-screen (speed vs. completeness)',
           'type': 'str',
           'default': 'off',
           'choices': ['off', 'marginal_pd'],
           'combo_state': 'readonly',
           'depends_on': (10, True),
           'help': "'off' (default) evaluates every candidate marker pair exactly - correct, but "
                   "slow when many markers are shortlisted. 'marginal_pd' (cheap) ranks pairs by "
                   "how much each marker's own effect varies on its own, then only computes the "
                   "full, exact interaction score for the strongest-ranked pairs - much faster, "
                   "but can miss a pair where BOTH markers look unremarkable alone yet interact "
                   "strongly together (a 'pure epistasis' pair). This means the result is an "
                   "APPROXIMATION - unselected pairs are reported as having no interaction, which "
                   "may not be true, they simply were not checked exactly."},
          {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
           'type': 'top_pct',
           'default': 'all',
           'depends_on': (10, True),
           'help': "Only used when the pre-screen above is not 'off'. What fraction of candidate "
                   "pairs the pre-screen shortlists for an exact interaction score - e.g. 2 means "
                   "the top 2% of pairs by the pre-screen's own cheap ranking. Lower = faster but "
                   "more likely to miss a real interaction; higher = slower but more thorough."},
          {'label': 'Interaction grid points per marker (3 = genotype classes)',
           'type': 'int',
           'default': 3,
           'depends_on': (10, True),
           'help': "How many representative values each marker's own interaction grid uses - 3 "
                   "matches this tool's usual 0/1/2 genotype coding exactly and is the right "
                   "choice for ordinary hard-called data. Raising this can help on fractional/"
                   "dosage-coded marker data (e.g. RIL/NAM populations, or PLINK data with some "
                   "missing calls) at higher cost (cost grows with the SQUARE of this number)."}],
 'RF': [{'label': 'Tree number',
         'type': 'int',
         'default': 100,
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
                 'thousands of markers.'},
        {'label': 'Marker interaction method',
         'type': 'str',
         'default': 'pairwise_shap',
         'choices': ['pairwise_shap', 'friedman_h'],
         'depends_on': (5, True),
         'help': "How pairs of markers are scored for interaction strength. 'pairwise_shap' "
                 '(default) uses exact pairwise SHAP interaction values from the fitted forest '
                 "- fast and exact for a tree model, but only available for tree-based models. "
                 "'friedman_h' uses Friedman's H-statistic instead - a model-agnostic measure "
                 "(the same one used for SVR/KNN elsewhere in this pipeline) based on how much "
                 "a pair's joint effect on the prediction departs from the two markers acting "
                 'independently. Both use the same top-importance marker shortlist and top-N% '
                 'output filtering configured above.'},
        # Update ID ver4-6, R1/R1b (blueprint §5.2): appended, never
        # inserted (I5) - this list's own length was 10 (terminal index
        # 9) before this edit, verified rather than assumed. Read
        # defensively by RF.py, and only actually FORWARDED in the
        # 'friedman_h' branch (see that file's own comment) - these three
        # settings have no effect on the exact 'pairwise_shap' route,
        # which never calls h_statistic_interactions() at all. Neither
        # new field is 'tunable' for the same reason as every other
        # interaction-search setting in this schema (an explainability
        # side-computation, forced off during a hyperparameter search).
        #
        # Update ID ver4-7: added 'visible_when' (9, 'friedman_h') to all
        # three fields below - unlike 'depends_on' (which only greys a
        # field out), 'visible_when' makes render_hparam_panel() skip
        # rendering the field's widget(s) entirely whenever the field at
        # index 9 ('Marker interaction method') isn't currently
        # 'friedman_h'. These three are meaningless for the default
        # 'pairwise_shap' route (see the comment above), so there is
        # nothing useful to show or grey out in that case - the
        # field simply doesn't appear. 'depends_on': (5, True) is left in
        # place unchanged, so once 'friedman_h' IS selected these still
        # grey out correctly if 'Return marker effect for interactions?'
        # is unchecked. resolve_hparams() is unaffected: a field that was
        # never rendered this run simply falls back to its own
        # HPARAM_SPECS default via resolve_field()'s existing
        # st.session_state.get(key, default), exactly as already happens
        # for any field whose widget hasn't been drawn yet - and since
        # RF.py never forwards these outside the 'friedman_h' branch
        # anyway, a stale/default value sitting unused behind a hidden
        # widget has no effect. Any value a user previously entered while
        # 'friedman_h' was selected is preserved in st.session_state and
        # reappears unchanged if they switch back to it later.
        # Update ID ver4-8: 'surrogate_shap' removed from 'choices' - see
        # the identical note on RKHS's own copy of this field for the full
        # rationale (applies uniformly to RKHS/RF/SVR/KNN).
        {'label': 'Interaction pre-screen (speed vs. completeness, "friedman_h" method only)',
         'type': 'str',
         'default': 'off',
         'choices': ['off', 'marginal_pd'],
         'combo_state': 'readonly',
         'depends_on': (5, True),
         'visible_when': (9, 'friedman_h'),
         'help': "Only used when 'Marker interaction method' above is 'friedman_h' - has no "
                 "effect for the default 'pairwise_shap' method, which is already exact. 'off' "
                 "(default) evaluates every candidate marker pair exactly - correct, but slow "
                 "when many markers are shortlisted. 'marginal_pd' (cheap) ranks pairs by how "
                 "much each marker's own effect varies on its own, then only computes the full, "
                 "exact interaction score for the strongest-ranked pairs - much faster, but can "
                 "miss a pair where BOTH markers look unremarkable alone yet interact strongly "
                 "together (a 'pure epistasis' pair). This means the result is an "
                 "APPROXIMATION - unselected pairs are reported as having no interaction, which "
                 "may not be true, they simply were not checked exactly."},
        {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
         'type': 'top_pct',
         'default': 'all',
         'depends_on': (5, True),
         'visible_when': (9, 'friedman_h'),
         'help': "Only used when the pre-screen above is not 'off'. What fraction of candidate "
                 "pairs the pre-screen shortlists for an exact interaction score - e.g. 2 means "
                 "the top 2% of pairs by the pre-screen's own cheap ranking. Lower = faster but "
                 "more likely to miss a real interaction; higher = slower but more thorough."},
        {'label': 'Interaction grid points per marker (3 = genotype classes, "friedman_h" method only)',
         'type': 'int',
         'default': 3,
         'depends_on': (5, True),
         'visible_when': (9, 'friedman_h'),
         'help': "Only used when 'Marker interaction method' above is 'friedman_h'. How many "
                 "representative values each marker's own interaction grid uses - 3 matches this "
                 "tool's usual 0/1/2 genotype coding exactly and is the right choice for ordinary "
                 "hard-called data. Raising this can help on fractional/dosage-coded marker data "
                 "(e.g. RIL/NAM populations, or PLINK data with some missing calls) at higher "
                 "cost (cost grows with the SQUARE of this number)."},
        # Update (Requirements.md item 2 - cross-model H-index result
        # diversity): appended, never inserted (I5) - this list's own
        # length was 13 (terminal index 12) before this edit, verified
        # rather than assumed. 'friedman_h' previously reused field 6
        # ('Number of samples for marker effect interactions', default
        # 30) as its OWN background sample size for Friedman's
        # H-statistic - a field whose default of 30 was tuned for the
        # 'pairwise_shap' route's exact, per-row TreeSHAP explanation
        # (cheap enough to explain few rows exactly), not for H^2's own
        # partial-dependence AVERAGE, which needs a background at least
        # as large as SVR/KNN/RKHS's own equivalent fields (each
        # defaulting to 100/100/50) to be comparably stable. A 30-row
        # background gives a visibly noisier H^2 ranking than the same
        # statistic computed on a 100-row background for another model -
        # this was a real, fixable contributor to "the number of
        # extracted interactions among the prediction models are quite
        # diverse even under the same configuration when using H-index"
        # (the shortlist size and top-N% were already reproducible
        # across models - see top_select()'s own fix - only the
        # H-statistic's own precision was not). This new field gives
        # 'friedman_h' its own properly-sized default (100, matching
        # SVR/KNN), independent of `shapley_num`'s own Shapley-era
        # value; RF.py reads it defensively (`len(params) > 13`) and
        # falls back to the OLD `shapley_num`-reuse behaviour for any
        # config predating this field, so nothing already saved changes
        # silently. Not 'tunable' for the same reason as every other
        # interaction-search setting in this schema (an explainability
        # side-computation, forced off during a hyperparameter search).
        {'label': 'Background sample size for Friedman H-statistic interactions ("friedman_h" method only)',
         'type': 'int',
         'default': 100,
         'depends_on': (5, True),
         'visible_when': (9, 'friedman_h'),
         'help': "Only used when 'Marker interaction method' above is 'friedman_h'. How many test "
                 "individuals are used to average out every marker other than the pair currently "
                 "being tested, when estimating each pair's interaction strength - the same role "
                 "'Background sample size for interaction search' plays for SVR/KNN. Larger = more "
                 "stable, more comparable H-statistic estimates but slower; too small (e.g. the "
                 "'pairwise_shap' route's own much smaller sample count) makes the ranking noisier "
                 "than the equivalent SVR/KNN/RKHS computation on the same data."}],
 'ExtraTrees': [{'label': 'Tree number',
         'type': 'int',
         'default': 100,
         'help': 'How many individual decision trees to average together. More trees usually give '
                 'steadier, more reliable predictions, at the cost of longer runtime.',
         'tunable': {'low': 100, 'high': 2000, 'step': 100}},
        {'label': 'Maximum features per tree',
         'type': 'rf_max_features',
         'default': '1.0',
         'choices': ['sqrt', 'log2', 'None'],
         'combo_state': 'normal',
         'help': "How many markers each tree is allowed to consider at every split. Extremely "
                 'Randomised Trees pick the SPLIT THRESHOLD at random rather than searching for '
                 "the best one, so this setting matters somewhat less here than for RF, but still "
                 "controls how much randomness each tree sees.",
         'tunable': {'choices': ['sqrt', 'log2', 0.5, 0.8, 1.0]}},
        {'label': "Bootstrap sample fraction ('None' = ExtraTrees' own default: no resampling)",
         'type': 'int_float_or_none',
         'default': None,
         'help': "'None' (the default) uses Extremely Randomised Trees in their natural, "
                 'textbook form: every tree sees the FULL training set, relying only on random '
                 'split thresholds for diversity between trees - this is what typically makes '
                 'ExtraTrees a genuinely different, often less overfitted, alternative to RF. '
                 'Setting a fraction here instead switches on bootstrap resampling at that '
                 'fraction (like RF does), for anyone who wants that comparison instead.',
         'tunable': {'choices': [None, 0.5, 0.7, 0.9]}},
        {'label': 'Maximum tree depth',
         'type': 'int_float_or_none',
         'default': None,
         'help': "Limits how many splits deep each tree can grow. 'None' (the default) lets trees "
                 'grow until every leaf is pure or too small to split further.',
         'tunable': {'choices': [None, 5, 10, 15, 25]}},
        {'label': 'Minimum samples per leaf in each tree',
         'type': 'int',
         'default': 1,
         'help': 'The smallest number of samples allowed in a leaf node. Raising this (e.g. 5-20) '
                 'smooths predictions and reduces overfitting, especially with noisy phenotypes.',
         'tunable': {'low': 1, 'high': 20, 'step': 1}},
        {'label': 'Return marker effect for interactions?',
         'type': 'bool',
         'default': False,
         'help': 'If checked, also searches for pairs of markers that interact with each other '
                 '(beyond what each marker alone explains), using the same exact TreeSHAP method '
                 "RF uses. This is slower - the settings below only apply when this is checked. "
                 "Defaults to unchecked (unlike RF, whose own default is checked) so an existing "
                 "config that adds this model does not silently pay the extra cost."},
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
                 "- e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                 'a very large table for datasets with many markers.'},
        {'label': 'Max markers considered for interaction search ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (5, True),
         'help': 'Only the top markers (ranked by importance) are checked for pairwise '
                 'interactions; every other marker pair is left out. Fewer markers = much faster. '
                 "'all' checks every possible pair but can take a very long time on datasets with "
                 'thousands of markers.'}],
 'GBDT': [{'label': 'Boosting iterations (max_iter)',
         'type': 'int',
         'default': 100,
         'help': 'How many boosting rounds (trees added one after another, each correcting the '
                 "previous rounds' errors) to run. More iterations can fit more detail but risk "
                 'overfitting - usually tuned together with the learning rate below.',
         'tunable': {'low': 50, 'high': 500, 'step': 50}},
        {'label': 'Learning rate',
         'type': 'float',
         'default': 0.1,
         'help': 'How much each new tree is allowed to correct the previous rounds. Smaller values '
                 'need more boosting iterations but often generalise better; larger values fit '
                 'faster but risk overshooting.',
         # Update ID ver4-6, R1.2(b): log-scaled (see the RKHS Kernel
         # bandwidth field's own comment above for the full rationale).
         'tunable': {'low': 0.01, 'high': 0.3, 'scale': 'log'}},
        {'label': 'Maximum tree depth',
         'type': 'int_float_or_none',
         'default': None,
         'help': "Limits how many splits deep each tree can grow. 'None' (the default) lets depth "
                 'be governed only by the maximum leaf nodes setting below.',
         'tunable': {'choices': [None, 3, 5, 8, 15]}},
        {'label': 'Maximum leaf nodes per tree',
         'type': 'int',
         'default': 30,
         'help': "How many leaf nodes each individual (shallow, boosted) tree is allowed.",
         'tunable': {'low': 8, 'high': 64, 'step': 8}},
        {'label': 'Minimum samples per leaf in each tree',
         'type': 'int',
         'default': 20,
         'help': 'The smallest number of samples allowed in a leaf node. Larger values smooth '
                 'predictions and reduce overfitting, especially with noisy phenotypes.',
         'tunable': {'low': 5, 'high': 50, 'step': 5}},
        {'label': 'L2 regularisation',
         'type': 'float',
         'default': 0.0,
         'help': 'A penalty on large leaf values, discouraging the model from fitting extreme '
                 'per-leaf corrections. 0 (the default) disables it; larger values regularise more '
                 'strongly.',
         'tunable': {'low': 0.0, 'high': 1.0}},
        {'label': 'Return marker effect for interactions?',
         'type': 'bool',
         'default': False,
         'help': 'If checked, also searches for pairs of markers that interact with each other, '
                 "using the same exact TreeSHAP method RF uses. This model has no free, "
                 "already-fitted marker-effect measure the way RF does, so its own marker effect "
                 '(returned regardless of this setting) is read from permutation importance on a '
                 'correlation-shortlisted marker set. This is slower - the settings below only '
                 'apply when this is checked.'},
        {'label': 'Number of samples for marker effect interactions',
         'type': 'int',
         'default': 30,
         'depends_on': (6, True),
         'help': 'How many test individuals to search for marker-pair interactions in. More '
                 'individuals give a more representative picture across the population, but take '
                 'longer.'},
        {'label': 'Output only the top N% of interactions ("all" for everything)',
         'type': 'top_pct',
         'default': 'all',
         'depends_on': (6, True),
         'help': 'Only keep the strongest interactions found, as a percentage of all pairs tested '
                 "- e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                 'a very large table for datasets with many markers.'},
        {'label': 'Max markers considered for interaction search ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (6, True),
         'help': 'Only the top markers are checked for pairwise interactions; every other marker '
                 "pair is left out. Fewer markers = much faster. 'all' checks every possible pair "
                 'but can take a very long time on datasets with thousands of markers.'}],
 'XGBoost': [{'label': 'Tree number (n_estimators)',
         'type': 'int',
         'default': 100,
         'help': 'How many boosting rounds (trees) to build, one after another, each correcting '
                 'the previous rounds\' errors.',
         'tunable': {'low': 50, 'high': 500, 'step': 50}},
        {'label': 'Maximum tree depth',
         'type': 'int_float_or_none',
         'default': 6,
         'help': "How deep each individual boosted tree is allowed to grow. XGBoost's own "
                 'default is 6 - deeper trees can capture more complex interactions but overfit '
                 'more easily.',
         'tunable': {'choices': [3, 4, 6, 8, 10]}},
        {'label': 'Learning rate',
         'type': 'float',
         'default': 0.1,
         'help': 'How much each new tree is allowed to correct the previous rounds. Smaller values '
                 'need more boosting rounds but often generalise better.',
         # Update ID ver4-6, R1.2(b): log-scaled (see the RKHS Kernel
         # bandwidth field's own comment for the full rationale).
         'tunable': {'low': 0.01, 'high': 0.3, 'scale': 'log'}},
        {'label': 'Row subsample fraction',
         'type': 'float',
         'default': 1.0,
         'help': 'The fraction of training individuals randomly sampled to grow each tree. Values '
                 'below 1.0 add randomness between trees, which can reduce overfitting.',
         'tunable': {'low': 0.5, 'high': 1.0}},
        {'label': 'Column subsample fraction per tree',
         'type': 'float',
         'default': 1.0,
         'help': 'The fraction of markers randomly sampled to grow each tree. Values below 1.0 add '
                 'randomness between trees, which can reduce overfitting on high-dimensional '
                 'genotype data.',
         'tunable': {'low': 0.3, 'high': 1.0}},
        {'label': 'L2 regularisation (reg_lambda)',
         'type': 'float',
         'default': 1.0,
         'help': "A penalty on large leaf weights. XGBoost's own default is 1.0; larger values "
                 'regularise more strongly.',
         'tunable': {'low': 0.0, 'high': 5.0}},
        {'label': 'Return marker effect for interactions?',
         'type': 'bool',
         'default': False,
         'help': 'If checked, also searches for pairs of markers that interact with each other, '
                 'using the same exact TreeSHAP method RF uses. This is slower - the settings '
                 'below only apply when this is checked. Requires the optional "xgboost" package '
                 'to be installed.'},
        {'label': 'Number of samples for marker effect interactions',
         'type': 'int',
         'default': 30,
         'depends_on': (6, True),
         'help': 'How many test individuals to search for marker-pair interactions in. More '
                 'individuals give a more representative picture across the population, but take '
                 'longer.'},
        {'label': 'Output only the top N% of interactions ("all" for everything)',
         'type': 'top_pct',
         'default': 'all',
         'depends_on': (6, True),
         'help': 'Only keep the strongest interactions found, as a percentage of all pairs tested '
                 "- e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                 'a very large table for datasets with many markers.'},
        {'label': 'Max markers considered for interaction search ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (6, True),
         'help': 'Only the top markers (ranked by importance) are checked for pairwise '
                 'interactions; every other marker pair is left out. Fewer markers = much faster. '
                 "'all' checks every possible pair but can take a very long time on datasets with "
                 'thousands of markers.'}],
 'EBM': [{'label': 'Learning rate',
         'type': 'float',
         'default': 0.04,
         'help': "How much each boosting round is allowed to correct the previous rounds. This "
                 "estimator's own default (0.04) is deliberately small - EBMs are boosted very "
                 'slowly, over many rounds, to keep each individual feature/pair curve smooth and '
                 'interpretable.',
         # Update ID ver4-6, R1.2(b): log-scaled (see the RKHS Kernel
         # bandwidth field's own comment for the full rationale).
         'tunable': {'low': 0.01, 'high': 0.2, 'scale': 'log'}},
        {'label': 'Maximum leaves per tree',
         'type': 'int',
         'default': 3,
         'help': 'How many leaf nodes each individual boosting-round tree is allowed. EBMs use '
                 'very shallow trees by design, so this is normally left small.',
         'tunable': {'low': 2, 'high': 6, 'step': 1}},
        {'label': 'Minimum samples per leaf',
         'type': 'int',
         'default': 4,
         'help': 'The smallest number of samples allowed in a leaf node. Larger values smooth the '
                 'fitted curves and reduce overfitting.',
         'tunable': {'low': 2, 'high': 20, 'step': 2}},
        {'label': 'Outer bags',
         'type': 'int',
         'default': 8,
         'help': 'How many independent copies of the model are bagged together and averaged - '
                 'more bags give smoother, more stable curves at the cost of longer fitting time.',
         'tunable': {'low': 2, 'high': 16, 'step': 2}},
        {'label': 'Search for marker-pair interactions?',
         'type': 'bool',
         'default': False,
         'help': "If checked, this model searches for pairs of markers that interact with each "
                 "other WHILE FITTING - unlike every other model here, an EBM's pairwise terms "
                 'ARE part of the model itself, not a separate explanation step computed '
                 'afterwards, so this setting also affects the FIT, not only what gets reported. '
                 'Requires the optional "interpret" package to be installed.'},
        {'label': 'Number of interaction terms to search for ("all" = this model\'s own default)',
         'type': 'int_or_all',
         'default': 10,
         'depends_on': (4, True),
         'help': 'How many candidate marker pairs this model searches for and fits as pairwise '
                 "terms while training. This is a COUNT of interaction terms, not a marker "
                 "shortlist size - unlike every other model's own similarly-named field. 'all' "
                 "uses this model's own built-in default heuristic instead of a fixed count."},
        {'label': 'Output only the top N% of interactions ("all" for everything)',
         'type': 'top_pct',
         'default': 'all',
         'depends_on': (4, True),
         'help': 'Only keep the strongest of the interaction terms this model already found while '
                 "fitting, as a percentage - e.g. 0.01 keeps only the top 0.01%. 'all' keeps every "
                 'term this model found.'}],
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
          # Update ID ver4-6, R1.2(b): log-scaled (see the RKHS Kernel
          # bandwidth field's own comment for the full rationale).
          'tunable': {'low': 0.001, 'high': 2.0, 'scale': 'log'}},
         {'label': 'Constraint',
          'type': 'float',
          'default': 1.0,
          'help': 'Controls the trade-off between fitting the training data closely and keeping '
                  'the model simple. Higher values fit the training data harder (risk of '
                  'overfitting); lower values favour a smoother, more general model.',
          # Update ID ver4-6, R1.2(b): log-scaled - this field's own
          # native range spans FOUR orders of magnitude (0.01-100), the
          # single worst offender named in the blueprint's own R1.1
          # "amplifiers" list.
          'tunable': {'low': 0.01, 'high': 100.0, 'scale': 'log'}},
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
                  'but noisier scores. 200 is a reasonable balance for large datasets.'},
         {'label': 'Return marker-pair interactions?',
          'type': 'bool',
          'default': False,
          'help': 'If checked, also searches for pairs of markers that interact with each other, '
                  "using Friedman's H-statistic (SVR has no built-in pairwise-interaction method, "
                  "so this is a model-agnostic ranking rather than an exact computation like RF's "
                  'own TreeSHAP interactions). The settings below only apply when this is checked.'},
         {'label': 'Max markers considered for interaction search ("all" for every marker)',
          'type': 'int_or_all',
          'default': 500,
          'depends_on': (11, True),
          'help': 'Only the top markers (ranked by correlation with the trait) are checked for '
                  "pairwise interactions; every other marker pair is left out. Fewer markers = much "
                  "faster. 'all' checks every possible pair but can take a very long time on "
                  'datasets with thousands of markers.'},
         {'label': 'Background sample size for interaction search',
          'type': 'int',
          'default': 100,
          'depends_on': (11, True),
          'help': 'How many test individuals are used to average out every marker other than '
                  'the pair currently being tested, when estimating each pair\'s interaction '
                  'strength. Larger = more stable estimates but slower.'},
         {'label': 'Output only the top N% of interactions ("all" for everything)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (11, True),
          'help': 'Only keep the strongest interactions found, as a percentage of all pairs tested '
                  "- e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                  'a very large table for datasets with many markers.'},
         # Update ID ver4-6, R1/R1b (blueprint §5.2): appended, never
         # inserted (I5) - this list's own length was 15 (terminal index
         # 14) before this edit, verified rather than assumed.
         # Update ID ver4-8: 'surrogate_shap' removed from 'choices' - see
         # the identical note on RKHS's own copy of this field for the full
         # rationale (applies uniformly to RKHS/RF/SVR/KNN).
         {'label': 'Interaction pre-screen (speed vs. completeness)',
          'type': 'str',
          'default': 'off',
          'choices': ['off', 'marginal_pd'],
          'combo_state': 'readonly',
          'depends_on': (11, True),
          'help': "'off' (default) evaluates every candidate marker pair exactly - correct, but "
                  "slow when many markers are shortlisted (SVR has no native pairwise-interaction "
                  "API, so every pair costs a real prediction call). 'marginal_pd' (cheap) ranks "
                  "pairs by how much each marker's own effect varies on its own, then only "
                  "computes the full, exact interaction score for the strongest-ranked pairs - "
                  "much faster, but can miss a pair where BOTH markers look unremarkable alone "
                  "yet interact strongly together (a 'pure epistasis' pair). This means the "
                  "result is an APPROXIMATION - unselected pairs are reported as having no "
                  "interaction, which may not be true, they simply were not checked exactly."},
         {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (11, True),
          'help': "Only used when the pre-screen above is not 'off'. What fraction of candidate "
                  "pairs the pre-screen shortlists for an exact interaction score - e.g. 2 means "
                  "the top 2% of pairs by the pre-screen's own cheap ranking. Lower = faster but "
                  "more likely to miss a real interaction; higher = slower but more thorough."},
         {'label': 'Interaction grid points per marker (3 = genotype classes)',
          'type': 'int',
          'default': 3,
          'depends_on': (11, True),
          'help': "How many representative values each marker's own interaction grid uses - 3 "
                  "matches this tool's usual 0/1/2 genotype coding exactly and is the right "
                  "choice for ordinary hard-called data. Raising this can help on fractional/"
                  "dosage-coded marker data (e.g. RIL/NAM populations, or PLINK data with some "
                  "missing calls) at higher cost (cost grows with the SQUARE of this number)."}],
 'KNN': [{'label': 'Number of neighbours',
          'type': 'int',
          'default': 5,
          'help': 'How many of the most genetically similar training individuals are averaged '
                  "together to predict each new individual's trait. Fewer neighbours can pick up "
                  'more local detail but are noisier; more neighbours give a smoother, more stable '
                  "prediction but can blur out real differences. If 'Return marker-pair "
                  "interactions?' below is also checked, a small value here (e.g. the default 5) "
                  'also makes KNN\'s own predictions less smooth from one marker value to the '
                  'next, which can inflate the H-statistic interaction search broadly rather than '
                  'just at true interactions - raising this can reduce that effect somewhat.',
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
                  'but noisier scores. 200 is a reasonable balance for large datasets.'},
         {'label': 'Return marker-pair interactions?',
          'type': 'bool',
          'default': False,
          'help': 'If checked, also searches for pairs of markers that interact with each other, '
                  "using Friedman's H-statistic (KNN has no built-in pairwise-interaction method, "
                  'so this is a model-agnostic ranking, not an exact computation). The settings '
                  'below only apply when this is checked. Known limitation: KNN predicts by '
                  "averaging a fixed set of 'nearest' training individuals, which can change "
                  'abruptly as a marker value is varied - unlike a smooth model (e.g. SVR), this '
                  'can make the H-statistic look elevated for MANY marker pairs at once, not just '
                  "truly interacting ones. If KNN's ring looks like a much denser web of links "
                  "than another model's under the same settings, treat the RELATIVE ranking "
                  "within KNN's own results as informative, but be cautious reading its raw "
                  'magnitudes as directly comparable to another model - a larger neighbour count '
                  'above and a larger background sample below both help smooth this out, though '
                  'neither fully removes it.'},
         {'label': 'Max markers considered for interaction search ("all" for every marker)',
          'type': 'int_or_all',
          'default': 500,
          'depends_on': (8, True),
          'help': 'Only the top markers (ranked by correlation with the trait) are checked for '
                  "pairwise interactions; every other marker pair is left out. Fewer markers = much "
                  "faster. 'all' checks every possible pair but can take a very long time on "
                  'datasets with thousands of markers.'},
         {'label': 'Background sample size for interaction search',
          'type': 'int',
          'default': 100,
          'depends_on': (8, True),
          'help': 'How many test individuals are used to average out every marker other than '
                  'the pair currently being tested, when estimating each pair\'s interaction '
                  'strength. Larger = more stable estimates but slower.'},
         {'label': 'Output only the top N% of interactions ("all" for everything)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (8, True),
          'help': 'Only keep the strongest interactions found, as a percentage of all pairs tested '
                  "- e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                  'a very large table for datasets with many markers.'},
         # Update ID ver4-6, R1/R1b (blueprint §5.2): appended, never
         # inserted (I5) - this list's own length was 12 (terminal index
         # 11) before this edit, verified rather than assumed.
         # Update ID ver4-8: 'surrogate_shap' removed from 'choices' - see
         # the identical note on RKHS's own copy of this field for the full
         # rationale (applies uniformly to RKHS/RF/SVR/KNN).
         {'label': 'Interaction pre-screen (speed vs. completeness)',
          'type': 'str',
          'default': 'off',
          'choices': ['off', 'marginal_pd'],
          'combo_state': 'readonly',
          'depends_on': (8, True),
          'help': "'off' (default) evaluates every candidate marker pair exactly - correct, but "
                  "slow when many markers are shortlisted (KNN has no native pairwise-interaction "
                  "API, so every pair costs a real prediction call). 'marginal_pd' (cheap) ranks "
                  "pairs by how much each marker's own effect varies on its own, then only "
                  "computes the full, exact interaction score for the strongest-ranked pairs - "
                  "much faster, but can miss a pair where BOTH markers look unremarkable alone "
                  "yet interact strongly together (a 'pure epistasis' pair). This means the "
                  "result is an APPROXIMATION - unselected pairs are reported as having no "
                  "interaction, which may not be true, they simply were not checked exactly."},
         {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (8, True),
          'help': "Only used when the pre-screen above is not 'off'. What fraction of candidate "
                  "pairs the pre-screen shortlists for an exact interaction score - e.g. 2 means "
                  "the top 2% of pairs by the pre-screen's own cheap ranking. Lower = faster but "
                  "more likely to miss a real interaction; higher = slower but more thorough."},
         {'label': 'Interaction grid points per marker (3 = genotype classes)',
          'type': 'int',
          'default': 3,
          'depends_on': (8, True),
          'help': "How many representative values each marker's own interaction grid uses - 3 "
                  "matches this tool's usual 0/1/2 genotype coding exactly and is the right "
                  "choice for ordinary hard-called data. Raising this can help on fractional/"
                  "dosage-coded marker data (e.g. RIL/NAM populations, or PLINK data with some "
                  "missing calls) at higher cost (cost grows with the SQUARE of this number)."}],
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
          'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log'}},  # ver4-6 R1.2(b)
         {'label': 'Decay',
          'type': 'float',
          'default': 0.0005,
          'help': "A small penalty that discourages the network's weights from growing too large, "
                  'as another safeguard against overfitting. 0 disables it; larger values '
                  'regularise more strongly.',
          'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log'}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
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
                  'population, but take longer.'},
         {'label': 'Return marker-pair interactions?',
          'type': 'bool',
          'default': False,
          'help': 'If checked, also searches for pairs of markers that interact with each other, '
                  "using Neural Interaction Detection (NID) - read directly from this network's "
                  'own trained first-layer weights, so it costs virtually nothing extra beyond the '
                  'training that already happened. The setting below only applies when this is '
                  'checked.'},
         {'label': 'Max markers considered for interaction search ("all" for every marker)',
          'type': 'int_or_all',
          'default': 500,
          'depends_on': (8, True),
          'help': "Only the top markers (ranked by this model's own marker-effect scores) are "
                  "included in the interaction table; every other marker pair is left out. This "
                  'only bounds how big the OUTPUT table is - it does not change how long the '
                  "search itself takes. 'all' includes every marker."},
         {'label': 'Output only the top N% of interactions ("all" for everything)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (8, True),
          'help': 'Only keep the strongest interactions found, as a percentage of all pairs tested '
                  "- e.g. 0.01 keeps only the top 0.01%. 'all' keeps every pair tested, which can be "
                  'a very large table for datasets with many markers.'}],
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
                        'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log'}},  # ver4-6 R1.2(b)
                       {'label': 'Decay',
                        'type': 'float',
                        'default': 0.0005,
                        'help': "A small penalty that discourages the network's weights from "
                                'growing too large, as another safeguard against overfitting. 0 '
                                'disables it; larger values regularise more strongly.',
                        'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log'}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
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
                          'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log'}},  # ver4-6 R1.2(b)
                         {'label': 'Decay',
                          'type': 'float',
                          'default': 0.0005,
                          'help': "A small penalty that discourages the network's weights from "
                                  'growing too large, as another safeguard against overfitting. 0 '
                                  'disables it; larger values regularise more strongly.',
                          'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log'}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
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
                          'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log'}},  # ver4-6 R1.2(b)
                         {'label': 'Decay',
                          'type': 'float',
                          'default': 0.0005,
                          'help': "A small penalty that discourages the network's weights from "
                                  'growing too large, as another safeguard against overfitting. 0 '
                                  'disables it; larger values regularise more strongly.',
                          'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log'}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
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
                                  'importance across the population, but take longer.'},
                         {'label': 'Return marker-pair interactions?',
                          'type': 'bool',
                          'default': False,
                          'help': 'If checked, writes this model\'s own marker-pair interaction '
                                  'strengths to Interaction.csv (one ring per selected model on '
                                  'the circos plot). This model already computes these values '
                                  'internally to decide which marker pairs to connect in its own '
                                  'graph - checking this only changes whether that already-'
                                  'computed table is also reported, not how the model itself '
                                  'trains or predicts.'}],
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
                                     'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log'}},  # ver4-6 R1.2(b)
                                    {'label': 'Decay',
                                     'type': 'float',
                                     'default': 0.0005,
                                     'help': "A small penalty that discourages the network's "
                                             'weights from growing too large, as another safeguard '
                                             'against overfitting. 0 disables it; larger values '
                                             'regularise more strongly.',
                                     'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log'}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
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
     'tunable': {'low': 1e-4, 'high': 0.1, 'scale': 'log'}},  # ver4-6 R1.2(b)
    {'label': 'Decay', 'type': 'float', 'default': 5e-4,
     'tunable': {'low': 1e-6, 'high': 1e-2, 'scale': 'log'}},  # ver4-6 R1.2(b): low raised (log needs low>0)
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
