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
             'tunable': {'low': 2000, 'high': 20000, 'step': 2000, 'floor_at_baseline': True}},
            {'label': 'Burn-in',
             'type': 'int',
             'default': 2000,
             'help': 'How many of the initial iterations (above) are discarded before averaging, '
                     "to let the model 'warm up' and stop being influenced by its arbitrary "
                     'starting point. Must be smaller than the iteration number.',
             'tunable': {'low': 200, 'high': 5000, 'step': 200, 'floor_at_baseline': True}},
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
             'tunable': {'low': 0.1, 'high': 0.9, 'priority': True}},
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
                     'which can be a very large table for datasets with many markers.'},
            {'label': 'Marker effect method',
             'type': 'str',
             'default': 'beta_direct',
             'legacy_default': 'beta_direct',
             'choices': ['beta_direct', 'closed_form'],
             'combo_state': 'readonly',
             'help': ("How each marker's effect is reported.\n"
                      "\n"
                      "- **beta_direct:** the model's own fitted marker effect "
                      "(beta), as it is.\n"
                      "- **closed_form (default):** an exact Shapley value (game-theoretic "
                      "contribution) for each marker, computed directly from the same fitted "
                      "beta with a closed-form formula and no extra refitting. Useful for "
                      "comparing marker importance on the same footing as the Shapley-based "
                      "effects of the other models.")}],
 'BayesB': [{'label': 'Iteration number',
             'type': 'int',
             'default': 12000,
             'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                     'iterations generally give a more stable, reliable fit, but take longer. '
                     '12000 is a reasonable starting point for genomic prediction.',
             'tunable': {'low': 2000, 'high': 20000, 'step': 2000, 'floor_at_baseline': True}},
            {'label': 'Burn-in',
             'type': 'int',
             'default': 2000,
             'help': 'How many of the initial iterations (above) are discarded before averaging, '
                     "to let the model 'warm up' and stop being influenced by its arbitrary "
                     'starting point. Must be smaller than the iteration number.',
             'tunable': {'low': 200, 'high': 5000, 'step': 200, 'floor_at_baseline': True}},
            {'label': 'Prior probability of a nonzero effect (probIn)',
             'type': 'float',
             'default': 0.5,
             'help': "BayesB's defining setting: the assumed proportion of markers with a real, "
                     'nonzero effect on the trait. Lower values (e.g. 0.05-0.1) assume only a few '
                     'markers matter (sparser, more like Bayesian variable selection); higher '
                     'values behave more like ridge regression, where most markers contribute a '
                     "little. 0.5 is BGLR's own default.",
             'tunable': {'low': 0.01, 'high': 0.9, 'priority': True}},
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
                     'which can be a very large table for datasets with many markers.'},
            {'label': 'Marker effect method',
             'type': 'str',
             'default': 'beta_direct',
             'legacy_default': 'beta_direct',
             'choices': ['beta_direct', 'closed_form'],
             'combo_state': 'readonly',
             'help': ("How each marker's effect is reported.\n"
                      "\n"
                      "- **beta_direct:** the model's own fitted marker effect "
                      "(beta), as it is.\n"
                      "- **closed_form (default):** an exact Shapley value (game-theoretic "
                      "contribution) for each marker, computed directly from the same fitted "
                      "beta with a closed-form formula and no extra refitting. Useful for "
                      "comparing marker importance on the same footing as the Shapley-based "
                      "effects of the other models.")}],
 'GBLUP': [{'label': 'Iteration number',
            'type': 'int',
            'default': 12000,
            'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                    'iterations generally give a more stable, reliable fit, but take longer. 12000 '
                    'is a reasonable starting point for genomic prediction.',
            'tunable': {'low': 2000, 'high': 20000, 'step': 2000, 'floor_at_baseline': True}},
           {'label': 'Burn-in',
            'type': 'int',
            'default': 2000,
            'help': 'How many of the initial iterations (above) are discarded before averaging, to '
                    "let the model 'warm up' and stop being influenced by its arbitrary starting "
                    'point. Must be smaller than the iteration number.',
            'tunable': {'low': 200, 'high': 5000, 'step': 200, 'floor_at_baseline': True}},
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
            'help': ("Internal/advanced setting, paired with the row offset above. -1 "
                     "(default) explains every row. Leave at -1 unless you deliberately want "
                     "to restrict this call to a sub-range of test individuals.")},
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
                    'which can be a very large table for datasets with many markers.'},
           # ver4-25, Stage 7 (design blueprint R13, D5): appended, never
           # inserted (I5) - this list's own length was 13 (terminal
           # index 12) before this edit. GBLUP is a kernel model with no
           # native per-marker coefficient (unlike rrBLUP/BayesB, which
           # fit directly on the marker matrix), so its own marker
           # effect has always needed an INDIRECT estimation route.
           # 'iml_shapley' (the legacy default, for every config
           # predating this field) keeps today's perturbation-refit
           # Shapley estimator (GBLUP.R's own `Shapley$new(...)` loop -
           # slow, and only an approximation, since each perturbed
           # coalition needs its own full BGLR MCMC refit).
           # 'closed_form' uses the exact VanRaden (2008) back-solving
           # identity instead (models/kernel_effects.py::
           # gblup_closed_form_effects()) - no refits, no sampling, and
           # exact for GBLUP's own linear kernel - so it is both much
           # faster AND more accurate; it is therefore the new GUI
           # default, while 'iml_shapley' remains available (and is what
           # an old, saved config without this field keeps using,
           # 'legacy_default' below).
           {'label': 'Marker effect method',
            'type': 'str',
            'default': 'closed_form',
            'legacy_default': 'iml_shapley',
            'choices': ['iml_shapley', 'closed_form'],
            'combo_state': 'readonly',
            'depends_on': (2, True),
            'help': ("How the per-marker effect is estimated.\n"
                     "\n"
                     "- **closed_form (default):** converts GBLUP's fitted breeding values "
                     "directly into marker effects with an exact algebraic identity. No "
                     "sampling and no refitting, so it is fast and exact for this linear "
                     "kernel.\n"
                     "- **iml_shapley:** a Shapley-value estimate that refits the whole model "
                     "for every perturbed marker coalition. Much slower and only an "
                     "approximation; kept for comparison.")}],
 'RKHS': [{'label': 'Iteration number',
           'type': 'int',
           'default': 12000,
           'help': 'How many rounds of Bayesian model fitting (MCMC sampling) to run. More '
                   'iterations generally give a more stable, reliable fit, but take longer. 12000 '
                   'is a reasonable starting point for genomic prediction.',
           'tunable': {'low': 2000, 'high': 20000, 'step': 2000, 'floor_at_baseline': True}},
          {'label': 'Burn-in',
           'type': 'int',
           'default': 2000,
           'help': 'How many of the initial iterations (above) are discarded before averaging, to '
                   "let the model 'warm up' and stop being influenced by its arbitrary starting "
                   'point. Must be smaller than the iteration number.',
           'tunable': {'low': 200, 'high': 5000, 'step': 200, 'floor_at_baseline': True}},
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
           'tunable': {'low': 0.1, 'high': 5.0, 'scale': 'log', 'priority': True}},
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
          # Requirements.md item 1: this field and 'Burn-in for Shapley
          # scores' below are ONLY used by the 'iml_shapley' marker-effect
          # route (field 17, "Marker effect method") - they control the
          # cheap re-fit iml::Shapley re-runs the model with for every
          # perturbed marker coalition (see RKHS.R's own "Deliberately
          # using the smaller, fast Shapley_nIter/Shapley_burnIn here"
          # comment, inside its own `!effect_uses_python_reconstruction`
          # block). Neither 'shap_predictor' nor 'shapiq_predictor' ever
          # refits the model at all (both explain this fit's OWN already-
          # completed BGLR result directly via `RKHSPredictor` - see
          # genomic_prediction.py's `_apply_rkhs_shapiq_predictor_effect()`),
          # so neither field does anything under either of those routes -
          # confirmed by reading all three call sites, not assumed.
          # 'visible_when' hides them (rather than just greying out via
          # 'depends_on' alone) whenever 'iml_shapley' isn't the selected
          # marker-effect method, so the panel doesn't show two settings
          # that would silently be ignored. The two fields ABOVE this one
          # ('Number of samples for Shapley scores', 'Max markers
          # considered for Shapley scores') are used by ALL THREE routes
          # (confirmed the same way - RKHS.R's own iml_shapley block AND
          # _apply_rkhs_shapiq_predictor_effect()'s own shared
          # reconstruction/shortlist code both read Shapley_num/
          # max_shap_features), so they keep no 'visible_when' and stay
          # shown regardless of which method is chosen.
          {'label': 'Number of iterations for Shapley scores',
           'type': 'int',
           'default': 200,
           'depends_on': (3, True),
           'visible_when': (17, 'iml_shapley'),
           'help': 'Only used when \'Marker effect method\' below is \'iml_shapley\' (has no '
                   "effect for 'shap_predictor', which does not re-fit the "
                   'model). Each Shapley perturbation test re-fits the whole model from scratch, so '
                   'this controls how many MCMC iterations that re-fit uses - separate from, and '
                   "much smaller than, the main 'Iteration number' above. More iterations = more "
                   'stable scores but much slower, since this re-fit happens many times. 200 is a '
                   'reasonable balance for large datasets.'},
          {'label': 'Burn-in for Shapley scores',
           'type': 'int',
           'default': 50,
           'depends_on': (3, True),
           'visible_when': (17, 'iml_shapley'),
           'help': "Only used when 'Marker effect method' below is 'iml_shapley' (has no effect "
                   "for 'shap_predictor', which does not re-fit the model). "
                   "How many of the Shapley re-fit's MCMC iterations (above) are discarded as "
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
           'help': ("Internal/advanced setting, paired with the row offset above. -1 "
                    "(default) explains every row. Leave at -1 unless you deliberately want "
                    "to restrict this call to a sub-range of test individuals.")},
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
          # Requirements.md item 1: these three fields only affect the
          # 'friedman_h' interaction route (field 18, "Marker interaction
          # method") - meaningless for 'shapiq', which has its own three
          # dedicated settings below instead. 'visible_when' added so the
          # panel shows exactly the settings relevant to whichever
          # interaction method is currently selected, never both groups
          # (or neither) at once.
          {'label': 'Interaction pre-screen (speed vs. completeness)',
           'type': 'str',
           'default': 'off',
           'choices': ['off', 'marginal_pd'],
           'combo_state': 'readonly',
           'depends_on': (10, True),
           # Update (perf): also shown for 'baseline_shap' now that route
           # supports the SAME screen - baseline_shap has no grid concept,
           # so only this field and the one below it (never the grid-
           # points field further down) extend to it.
           'visible_when': (18, ('friedman_h', 'baseline_shap')),
           'help': ("Only used when 'Marker interaction method' below is 'baseline_shap'.\n"
                    "\n"
                    "- **off (default):** evaluates every candidate marker pair exactly. "
                    "Correct, but slow when many markers are shortlisted.\n"
                    "- **marginal_pd:** ranks pairs cheaply by how much each marker's own "
                    "effect varies, then computes the exact interaction score only for the "
                    "strongest-ranked pairs. Much faster, but approximate: a pair whose "
                    "markers both look unremarkable alone yet interact strongly (pure "
                    "epistasis) can be missed, and unselected pairs are reported as having no "
                    "interaction.")},
          {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
           'type': 'top_pct',
           'default': 'all',
           'depends_on': (10, True),
           'visible_when': (18, ('friedman_h', 'baseline_shap')),
           'help': ("Only used when 'Marker interaction method' below is 'baseline_shap' and "
                    "the pre-screen above is not 'off'. The percentage of candidate pairs "
                    "that the pre-screen shortlists for an exact interaction score; for "
                    "example, 2 means the top 2% of pairs. Lower = faster but more likely to "
                    "miss a real interaction; higher = slower but more thorough.")},
          {'label': 'Interaction grid points per marker (3 = genotype classes)',
           'type': 'int',
           'default': 3,
           'depends_on': (10, True),
           'visible_when': (18, 'friedman_h'),
           'help': "Only used when 'Marker interaction method' below is 'friedman_h'. How many "
                   "representative values each marker's own interaction grid uses - 3 "
                   "matches this tool's usual 0/1/2 genotype coding exactly and is the right "
                   "choice for ordinary hard-called data. Raising this can help on fractional/"
                   "dosage-coded marker data (e.g. RIL/NAM populations, or PLINK data with some "
                   "missing calls) at higher cost (cost grows with the SQUARE of this number)."},
          # ver4-25, Stage 7 (design blueprint R13, D5): appended, never
          # inserted (I5) - this list's own length was 17 (terminal index
          # 16) before this edit. Same rationale as GBLUP's own identical
          # field (RKHS is likewise a kernel model with no native
          # per-marker coefficient): both non-'iml_shapley' choices below
          # reconstruct this fit's own kernel regression exactly in NumPy
          # (models/kernel_effects.py::RKHSPredictor) and explain THAT -
          # one fit, no refits - while 'iml_shapley' (kept, and what an
          # old config keeps using) is the original refit-per-coalition
          # Shapley estimator. genomic_prediction.py gates both at
          # runtime (V13.2): if RKHSPredictor's own reconstructed test
          # predictions do not match this task's real BGLR test
          # predictions closely enough, it automatically falls back to
          # 'iml_shapley' with a logged NOTE rather than silently
          # returning a possibly-wrong effect vector.
          #
          # This update (RKHS marker-effect/shapiq comparability fix -
          # see genomic_prediction.py::_apply_rkhs_shapiq_predictor_
          # effect()'s own docstring for the full rationale): this field
          # used to offer only 'iml_shapley'/'shapiq_predictor', and
          # 'shapiq_predictor' actually explained the reconstruction with
          # `shap.KernelExplainer` - the classic `shap` package, never the
          # `shapiq` package, despite the name. That's now split into two
          # choices so the name means what it says: 'shap_predictor' is
          # the exact original implementation, renamed (I11 under its new
          # name - anyone relying on its exact numbers selects it
          # explicitly); 'shapiq_predictor' (still the field's own
          # default - an EXISTING config whose saved value is literally
          # the string 'shapiq_predictor' now gets this genuinely-shapiq
          # behaviour the next time it runs, a deliberate exception to
          # this codebase's usual "a value's meaning never changes"
          # discipline, because the old behaviour never matched what that
          # value's own name always promised) now explains the SAME
          # reconstruction with the real `shapiq` package instead
          # (`pipeline_utils.shapiq_marker_effects()`) - the SAME
          # explainer, and the SAME automatic large-marker-count SPEX
          # escalation (this update's own fix), every other model's own
          # 'shapiq' marker-effect choice (SVR/KNN/RF/GBDT/XGBoost/
          # ExtraTrees) already uses, making RKHS's marker effects
          # genuinely comparable to theirs for the first time. A `shapiq`
          # failure never fails the task either way - 'shapiq_predictor'
          # falls back to 'shap_predictor' (not all the way to
          # 'iml_shapley') with a logged NOTE, the same "try shapiq,
          # catch, fall back to the proven non-shapiq route" pattern used
          # everywhere else 'shapiq' is offered. The two new fields right
          # after this one ('Background sample size'/'Number of
          # coalition samples for marker effect') configure ONLY the new
          # 'shapiq_predictor' route - 'shap_predictor' keeps its own
          # pre-existing hardcoded 50/200, exactly as before this update.
          {'label': 'Marker effect method',
           'type': 'str',
           'default': 'shapiq_predictor',
           'legacy_default': 'iml_shapley',
           'choices': ['iml_shapley', 'shap_predictor', 'shapiq_predictor', 'baseline_shap'],
           'combo_state': 'readonly',
           'depends_on': (3, True),
           'help': ("How the per-marker effect is estimated.\n"
                    "\n"
                    "- **shap_predictor:** reconstructs this fit's kernel "
                    "regression exactly and explains it with the 'shap' package's "
                    "KernelExplainer. No extra model refits.\n"
                    "- **iml_shapley:** a Shapley-value estimate that refits the whole model "
                    "for every perturbed marker coalition. Much slower.\n"
                    "- **baseline_shap (default):** explains the same exact reconstruction by a "
                    "background-referenced knock-out: a marker's effect is the average change "
                    "in the prediction when that marker is swapped to the value of a real "
                    "training individual (the same estimator as the 'baseline_shap' "
                    "interaction method). It scores every marker in the shortlist at any "
                    "marker count, deterministically and in seconds, but its numbers are on a "
                    "different baseline and are not comparable with 'shap_predictor' values.\n"
                    "\n"
                    "If the exact reconstruction cannot be verified against the run's own fit "
                    "closely enough, EasiGP falls back to 'iml_shapley' for that task and "
                    "says so in the run log. Changing this setting starts a fresh run "
                    "(earlier results are not reused).")},
          # ver4-25, Stage 8 (design blueprint R11): appended, never
          # inserted (I5). RKHS's own pairwise-interaction search
          # ('Return marker-pair interactions?' above, field 10) has
          # always used a surrogate model + Friedman's H-statistic
          # ('friedman_h', unchanged default and behaviour for any config
          # predating this field). 'shapiq' fits the SAME kind of
          # lightweight surrogate but explains it with shapiq's own
          # Shapley-INTERACTION indices (k-SII/FSII/STII) instead of
          # Friedman's H-statistic. Both routes remain surrogate-based
          # (a disclosed scope decision - see models.interaction_
          # extraction.surrogate_shapiq_interactions()'s own docstring),
          # so despite the shared 'shapiq' name this is NOT the same
          # exactness as the marker-EFFECT field's own 'shapiq_predictor'
          # choice above (the exact `RKHSPredictor` reconstruction, no
          # surrogate) - the two fields answer different questions
          # (per-marker effect vs. marker-PAIR interaction) and are
          # never compared against each other in code, only documented
          # here so the naming overlap doesn't read as a contradiction.
          # Only pairwise (order-2) results are currently stored even
          # though 'shapiq' can compute higher orders (see the three
          # fields below) - design blueprint R12 (Stage 9)'s own higher-
          # order storage has not landed yet.
          #
          # Requirements.md item 1: choices renamed from
          # 'surrogate_friedman_h'/'surrogate_shapiq' to plain
          # 'friedman_h'/'shapiq' - the "surrogate" prefix added nothing
          # a user could act on (every route on this field was ALREADY
          # surrogate-based, per the paragraph above) and just made this
          # field's dropdown noisier than every other model's identical
          # field (SVR/KNN already spell theirs 'friedman_h'/'shapiq').
          # A GUI session with an older saved 'surrogate_*' value self-
          # heals to its new equivalent via this field's own
          # 'legacy_value_map' (see main_app.py's render_field(), 'str'
          # + 'choices' branch) rather than silently resetting to the
          # unrelated default; a headless/JSON config or checkpoint still
          # holding the OLD spelling is also still accepted directly at
          # every consuming call site (genomic_prediction.py's dispatch,
          # checkpoint_utils.py's _SHAPIQ_INTERACTION_FIELD_INDEX) - I11
          # either way.
          {'label': 'Marker interaction method',
           'type': 'str',
           'default': 'friedman_h',
           'legacy_default': 'friedman_h',
           'choices': ['friedman_h', 'shapiq', 'shapiq_predictor', 'kernel_hessian',
                       'friedman_h_predictor', 'baseline_shap'],
           'legacy_value_map': {'surrogate_friedman_h': 'friedman_h', 'surrogate_shapiq': 'shapiq'},
           'combo_state': 'readonly',
           'depends_on': (10, True),
           'help': ("How marker-pair interactions are searched for.\n"
                    "\n"
                    "- **baseline_shap:** explains this fit's own exact kernel regression "
                    "using only real genotype combinations observed in the training "
                    "population, so linkage disequilibrium is never violated. It needs no "
                    "coalition-sampling budget (a fixed, predictable cost), and its values "
                    "use the same convention as the RF TreeSHAP interaction route, so they "
                    "sit on the same scale for a side-by-side comparison with RF.\n"
                    "\n"
                    "If the exact reconstruction cannot be verified against the run's own fit "
                    "closely enough, or the route fails for any other reason, EasiGP falls "
                    "back to a surrogate-model route for that task and says so in the run "
                    "log. It never silently reports a possibly wrong result.")},
          {'label': 'Interaction orders to search for (shapiq methods only, e.g. "2" or "2,3")',
           'type': 'str',
           'default': '2',
           'legacy_default': '2',
           'depends_on': (10, True),
           'visible_when': (18, ('shapiq', 'shapiq_predictor')),
           'help': "Only used when 'Marker interaction method' above is 'shapiq' or "
                   "'shapiq_predictor'. Which interaction orders (k = how many markers act "
                   'together) to search for, as a comma-separated list - e.g. "2" for pairs only '
                   '(the same as every other method here), or "2,3" to also search for three-way '
                   'interactions. Order-2 (pairwise) results are shown as the usual chord '
                   'diagram; order 3 and above are shown as hub-and-spoke glyphs on the circos '
                   'plot (choose which orders to draw, and whether as separate per-order plots or '
                   "one combined plot, under Tab 6's advanced circos settings)."},
          {'label': 'Shapley interaction index (shapiq methods only)',
           'type': 'str',
           'default': 'k-SII',
           'legacy_default': 'k-SII',
           'choices': ['k-SII', 'FSII', 'STII'],
           'combo_state': 'readonly',
           'depends_on': (10, True),
           'visible_when': (18, ('shapiq', 'shapiq_predictor')),
           'help': "Only used when 'Marker interaction method' above is 'shapiq' or "
                   "'shapiq_predictor'. Which axiomatic Shapley interaction index shapiq "
                   "computes - 'k-SII' (default) is a good general-purpose choice; FSII/STII are "
                   "alternative definitions offered by the shapiq package for anyone who wants to "
                   "compare them directly."},
          {'label': 'Coalition budget per explained row (shapiq methods only)',
           'type': 'int',
           'default': 4096,
           'legacy_default': 4096,
           'depends_on': (10, True),
           'visible_when': (18, ('shapiq', 'shapiq_predictor')),
           'help': "Only used when 'Marker interaction method' above is 'shapiq' or "
                   "'shapiq_predictor'. How many marker coalitions shapiq samples per explained "
                   "row when an exact computation is not feasible (more than ~12 shortlisted "
                   "markers). Higher = more accurate but slower; lower = faster but noisier. "
                   "Exact (no sampling) is used automatically whenever the shortlist is small "
                   "enough."},
          # This update (RKHS marker-effect/shapiq comparability fix):
          # appended, never inserted (I5) - this list's own length was 22
          # (terminal index 21) before this edit, verified rather than
          # assumed (an EARLIER draft of this same fix mistakenly placed
          # these two fields right after field 17 instead, which would
          # have shifted every field from 'Marker interaction method'
          # (18) onward by two positions and broken every params[N] read
          # of them throughout genomic_prediction.py/RKHS.R/
          # checkpoint_utils.py - caught and corrected before shipping).
          # Configure ONLY the new 'shapiq_predictor' route (field 17) -
          # see that field's own comment above for the full rationale;
          # 'shap_predictor' keeps its own pre-existing hardcoded 50/200,
          # exactly as it always has, and 'iml_shapley' ignores both.
          # Mirrors GBDT's/XGBoost's/ExtraTrees's own identically-named
          # 'Background sample size for marker effect (shapiq)'/'Number
          # of coalition samples for marker effect (shapiq)' fields
          # (defaults 50/200) for a consistent, comparable knob across
          # every model that offers a real 'shapiq' marker-effect route.
          {'label': 'Background sample size for marker effect (shapiq_predictor method only)',
           'gui_label': 'Background sample size for marker effect',
           'type': 'int',
           'default': 50,
           'legacy_default': 50,
           'depends_on': (3, True),
           'visible_when': (17, ('shapiq_predictor', 'baseline_shap')),
           'help': ("Only used when 'Marker effect method' above is 'baseline_shap'. A small reference set of training individuals used "
                    "as a 'typical' baseline when working out each marker's contribution; for "
                    "'baseline_shap' these are the real training rows each marker is swapped "
                    "to. Smaller = faster; larger = smoother, more stable scores but slower. "
                    "Has no effect on 'shap_predictor' (which always uses 50) or "
                    "'iml_shapley'.")},
          {'label': 'Number of coalition samples for marker effect (shapiq_predictor method only)',
           'type': 'int',
           'default': 200,
           'legacy_default': 200,
           'depends_on': (3, True),
           'visible_when': (17, 'shapiq_predictor'),
           'help': "Only used when 'Marker effect method' above is 'shapiq_predictor'. How many "
                   "coalitions shapiq samples per explained test individual when an exact "
                   "computation is not feasible. Higher = more accurate but slower. Has no effect "
                   "on 'shap_predictor' (which always uses 200) or 'iml_shapley'."},
          # This update - see RF's/SVR's identical field for the full
          # rationale. Appended, never inserted (I5) - this list's own
          # length was 24 (terminal index 23) before this edit.
          # Eligibility compares this model's own effect cap (field 5,
          # "Max markers considered for Shapley scores" - applies to
          # EVERY marker effect method here, including 'iml_shapley' -
          # see that field's own comment) against the interaction cap
          # (field 11). Consulted by both of this model's own interaction
          # routes (genomic_prediction.py's _r_model_surrogate_
          # interaction() and _apply_rkhs_predictor_interaction()).
          {'label': 'Use marker effects to select markers for interaction search?',
           'type': 'bool',
           'default': True,
           'depends_on': (10, True),
           'help': ("If ticked (default), and 'Max markers considered for interaction search' "
                    "above is 'all' or no larger than 'Max markers considered for Shapley "
                    "scores', the interaction-search shortlist reuses this model's "
                    "marker-effect (Shapley) values instead of a separate trait-correlation "
                    "ranking. Untick to always rank by trait correlation for the interaction "
                    "shortlist.\n"
                    "\n"
                    "This option is switched off automatically (trait correlation is used) "
                    "when the interaction shortlist needs more markers than the marker-effect "
                    "step covers, or when 'Return marker effect?' above is unticked.")}],
 'RF': [{'label': 'Tree number',
         'type': 'int',
         'default': 100,
         'help': 'How many individual decision trees to average together. More trees usually give '
                 'steadier, more reliable predictions, at the cost of longer runtime - returns '
                 'diminish well before 1000 for most datasets.',
         'tunable': {'low': 100, 'high': 2000, 'step': 100, 'floor_at_baseline': True}},
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
         'tunable': {'choices': ['sqrt', 'log2', 0.5, 0.8, 1.0], 'priority': True}},
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
         'tunable': {'low': 1, 'high': 20, 'step': 1, 'priority': True}},
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
         # ver4-25, Stage 8 (design blueprint R11): 'shapiq_tree' added to
         # 'choices' - an existing config's saved 'pairwise_shap'/
         # 'friedman_h' value is untouched (I11), and RF.py's own
         # dispatch (below) only takes the new branch when a user/config
         # explicitly selects it.
         'choices': ['pairwise_shap', 'friedman_h', 'shapiq_tree'],
         'depends_on': (5, True),
         'help': ("How pairs of markers are scored for interaction strength.\n"
                  "\n"
                  "- **pairwise_shap (default):** exact pairwise SHAP interaction values from "
                  "the fitted forest. Fast and exact, for tree-based models only.\n"
                  "- **shapiq_tree:** shapiq's TreeSHAP-IQ Shapley interaction indices on the "
                  "fitted forest. Exact and tree-specific for a 2-way (pairs only) search, "
                  "like 'pairwise_shap', but it can also search for interactions among 3 or "
                  "more markers (see the settings below).\n"
                  "\n"
                  "The exact tree algorithm becomes exponentially more expensive with each "
                  "extra interaction order. A 3-way-or-higher search therefore uses a "
                  "budget-bounded sampling estimate instead (see 'Coalition sample budget for "
                  "3-way-or-higher shapiq_tree search' below): it trades exactness for a "
                  "bounded, predictable runtime. A 2-way-only search remains exact.\n"
                  "\n"
                  "Both methods use the same top-importance marker shortlist and top-N% "
                  "output filtering configured above. Rows with an identical genotype across "
                  "the shortlist (common for closely related or inbred lines) are explained "
                  "only once.")},
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
                 "than the equivalent SVR/KNN/RKHS computation on the same data."},
        # ver4-25, Stage 8 (design blueprint R11): appended, never
        # inserted (I5) - this list's own length was 14 (terminal index
        # 13) before this edit.
        {'label': 'Interaction orders to search for (shapiq_tree method only, e.g. "2" or "2,3")',
         'type': 'str',
         'default': '2',
         'legacy_default': '2',
         'depends_on': (5, True),
         'visible_when': (9, 'shapiq_tree'),
         'help': "Only used when 'Marker interaction method' above is 'shapiq_tree'. Which "
                 "interaction orders (k = how many markers act together) to search for, as a "
                 'comma-separated list - e.g. "2" for pairs only, or "2,3" to also search for '
                 'three-way interactions.'},
        {'label': 'Shapley interaction index (shapiq_tree method only)',
         'type': 'str',
         'default': 'k-SII',
         'legacy_default': 'k-SII',
         'choices': ['k-SII', 'FSII', 'STII'],
         'combo_state': 'readonly',
         'depends_on': (5, True),
         'visible_when': (9, 'shapiq_tree'),
         'help': "Only used when 'Marker interaction method' above is 'shapiq_tree'. Which "
                 "axiomatic Shapley interaction index shapiq computes - 'k-SII' (default) is a "
                 'good general-purpose choice.'},
        # Requirements3.md item 2: RF's own per-marker additive effect
        # (returned regardless of the interaction settings above) has
        # always been the fitted forest's built-in impurity-based
        # `feature_importances_` - fast (already computed as a free
        # by-product of fitting), but only a WITHIN-model ranking, not a
        # per-prediction attribution the way every other model's own
        # marker effect is. 'shap' offers an alternative: exact TreeSHAP
        # values (the SAME `shap.TreeExplainer` route RF's own
        # 'pairwise_shap' interaction method already uses), aggregated
        # the same way every other model's Shapley-based effect is
        # (sum of |value| over the explained test rows) - directly
        # comparable to rrBLUP/BayesB/GBLUP/RKHS/SVR/KNN's own effect
        # numbers, at some extra cost. Appended, never inserted (I5);
        # 'feature_importance' (unchanged default) reproduces every
        # existing config's behaviour exactly.
        # Requirements.md item 2 (this update): 'shapiq' added to 'choices' -
        # a model-agnostic order-1 Shapley marker-effect route
        # (pipeline_utils.shapiq_marker_effects(), the same helper SVR/KNN/
        # RKHS's own 'shapiq' effect route already uses) offered here mainly
        # for consistency/tooling parity with those models, not because it
        # adds new information over the existing exact 'shap' (TreeSHAP)
        # route - order-1 Shapley values are index-independent, so 'shap'
        # already computes the mathematically-identical quantity, exactly
        # rather than by sampling. 'feature_importance'/'shap' (default and
        # existing alternative) are completely unchanged (I11). Requirements.
        # md item 3 (this update): 'Return marker effect?' (new field 18,
        # below) now gates this whole step - added here as this field's own
        # 'depends_on' so it greys out, matching every other model's
        # identical 'Marker effect method' field (RKHS/SVR/KNN).
        {'label': 'Marker effect method',
         'type': 'str',
         'default': 'feature_importance',
         'legacy_default': 'feature_importance',
         'choices': ['feature_importance', 'shap', 'shapiq'],
         'combo_state': 'readonly',
         'depends_on': (18, True),
         'help': ("How the per-marker effect (reported when 'Return marker effect?' above is "
                  "ticked) is computed.\n"
                  "\n"
                  "- **feature_importance:** the fitted forest's built-in "
                  "impurity-based importance. Fast, but a within-model ranking rather than a "
                  "per-prediction attribution.\n"
                  "- **shap (default):** exact TreeSHAP values on a sample of test individuals, summing "
                  "their magnitudes as every other Shapley-based model does. Directly "
                  "comparable with those, at extra cost.\n"
                  "\n"
                  "The settings below only apply when this is 'shap'.")},
        # Requirements.md item 2 (this update): label/gating generalised
        # from "'shap' only" to "'shap' or 'shapiq'" - this field already
        # controlled how many test individuals are explained for the exact
        # TreeSHAP route, and the new 'shapiq' route (immediately below)
        # needs the identical "how many rows to explain" setting, so it is
        # shared rather than duplicated (mirrors SVR's own four Shapley-
        # effect fields, already shared between its 'kernel_shap' and
        # 'shapiq' methods - see hparam_specs.py's SVR block). Renamed to
        # match this update's own new 'Number of individuals used to
        # explain the model' phrasing (Requirements.md item 3) rather than
        # adding a second, separately-labelled row-count field that would
        # do the exact same job. 'depends_on' now points at the new master
        # 'Return marker effect?' toggle (18) instead of the method field
        # itself - matching SVR's identical shared-effect-field convention
        # (its own depends_on points at its master toggle, not at 'Marker
        # effect method') - while 'visible_when' still restricts this to
        # the two methods that actually use a row count ('feature_
        # importance' needs none at all). The (16, ('shap', 'shapiq'))
        # tuple form is the new multi-value 'visible_when'/'depends_on'
        # extension (see main_app.py's schema comment above
        # HPARAM_DISPLAY_ORDER... i.e. above render_hparam_panel).
        {'label': 'Number of individuals used to explain the model (effect method "shap" or "shapiq" only)',
         'type': 'int',
         'default': 30,
         'legacy_default': 30,
         'depends_on': (18, True),
         'visible_when': (16, ('shap', 'shapiq')),
         'help': "Only used when 'Marker effect method' above is 'shap' or 'shapiq'. How many test "
                 "individuals are explained for the marker-effect calculation. More individuals "
                 "give a more representative picture but take longer."},
        # Requirements.md item 3 (this update): the new master toggle for
        # this model's own marker-EFFECT step (separate from 'Return marker "
        # effect for interactions?' above, field 5, which only gates the "
        # marker-PAIR interaction search) - as SVR/KNN/RKHS already have.
        # Appended, never inserted (I5); default True reproduces every
        # existing config's behaviour exactly (this model's marker effect
        # was always computed unconditionally before this field existed).
        {'label': 'Return marker effect?',
         'type': 'bool',
         'default': True,
         'help': ("If ticked (default), also computes each marker's contribution to the "
                  "prediction, in addition to the prediction itself, using the 'Marker effect "
                  "method' selected above. Untick to skip this step and save time when the "
                  "marker effects are not needed.")},
        # Requirements.md item 2 (this update): three fields dedicated to
        # the new 'shapiq' marker-effect method specifically - unlike
        # 'shap' (exact TreeSHAP, needs neither a marker shortlist nor a
        # background/coalition budget), shapiq_marker_effects() is a
        # sampling-based route (pipeline_utils.py's own TabularExplainer
        # wrapper) whose cost scales with all three, exactly like SVR/KNN's
        # analogous Shapley-effect settings. Kept separate from (not reused
        # from) the existing 'Max markers considered for interaction
        # search'/'friedman_h' background fields above, which govern the
        # unrelated marker-PAIR interaction step (field 5's own settings) -
        # per this project's own "respective fields for both marker effects
        # and interactions" convention (mirrors SVR's own fully separate
        # effect-section vs. interaction-section fields). All three are
        # appended, never inserted (I5), and default to reproducing a small,
        # SVR-KNN-matching budget.
        {'label': 'Max markers considered for marker effect (shapiq) ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (18, True),
         'visible_when': (16, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. Only the top markers "
                 "(ranked by this forest's own feature_importances_) are explained; every other "
                 "marker is reported as 0. Fewer markers = much faster. 'all' explains every "
                 "marker but can take a long time on datasets with thousands of markers. Not used "
                 "by 'feature_importance' (already free and full-width) or 'shap' (exact TreeSHAP "
                 "explains every marker directly, no shortlist needed)."},
        {'label': 'Background sample size for marker effect (shapiq)',
         'type': 'int',
         'default': 50,
         'depends_on': (18, True),
         'visible_when': (16, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. A small reference set "
                 "of training individuals used as a 'typical' baseline when working out each "
                 "marker's contribution. Smaller = faster; larger = smoother, more stable scores "
                 "but slower."},
        {'label': 'Number of coalition samples for marker effect (shapiq)',
         'type': 'int',
         'default': 200,
         'depends_on': (18, True),
         'visible_when': (16, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. How many random "
                 "combinations of markers are tried out to estimate each marker's contribution to "
                 "a prediction. Higher = more accurate but slower; lower = faster but noisier "
                 "scores."},
        # This update: "use marker effects to select markers for
        # interaction search?" - appended, never inserted (I5); this
        # list's own length was 22 (terminal index 21) before this edit.
        # Only ever eligible (and only ever rendered - see main_app.py's
        # own dynamic show/hide logic, which calls the SAME
        # models.interaction_extraction.
        # should_use_effect_for_interaction_shortlist() this field's own
        # engine-side check (RF.py) does) when 'Max markers considered
        # for interaction search' above is 'all' or no larger than this
        # model's own marker-effect coverage ('all' for 'feature_
        # importance'/'shap', or 'Max markers considered for marker
        # effect (shapiq)' above for 'shapiq') - whenever that is NOT the
        # case, the checkbox is hidden entirely and this model falls back
        # to its own pre-existing feature_importances_-based shortlist,
        # exactly as before this field existed (I11). Default True (this
        # field's own default, and what any config predating it
        # resolves to) reuses this model's marker-EFFECT values for the
        # interaction shortlist instead of recomputing feature_
        # importances_ separately for that step - both are already the
        # SAME underlying quantity whenever eligible, so this changes
        # nothing observable in that common case; unchecking it restores
        # the previous, always-separate feature_importances_-based
        # ranking.
        {'label': 'Use marker effects to select markers for interaction search?',
         'type': 'bool',
         'default': True,
         'depends_on': (5, True),
         'help': "If checked (default), and 'Max markers considered for interaction search' above "
                 "is 'all' or no larger than this model's own marker-effect coverage, the "
                 "interaction-search shortlist reuses this model's own already-computed "
                 "marker-effect values (from 'Marker effect method' above) instead of a separate "
                 "feature-importance ranking - both usually agree, so this mainly saves a small "
                 "amount of redundant computation. Uncheck to always rank by this forest's own "
                 "feature_importances_ for the interaction shortlist specifically, independent of "
                 "'Marker effect method'. Hidden automatically (falls back to feature_importances_) "
                 "whenever the interaction shortlist would need MORE markers than this model's own "
                 "marker-effect step covers."},
        # ver4-30: exact TreeSHAP-IQ's own cost is EXPONENTIAL in the
        # requested interaction order (O(T*L*D**k)) - confirmed, even
        # after every existing depth/estimator/row-count adaptive-shrink
        # mitigation above, to remain "extremely large"/"cannot be used"
        # for a genuine k>2 ("3-way-or-higher") search on a realistically
        # sized forest/shortlist. A k>2 search now instead explains this
        # task's own real fitted forest through shapiq's model-agnostic,
        # BUDGET-bounded TabularExplainer (the same bounded-cost route
        # RKHS's 'surrogate_shapiq'/SVR's/KNN's own 'shapiq' methods
        # already use) - this field is that route's own coalition-sample
        # ceiling. Appended, never inserted (I5); meaningless (never
        # read) for an order-2-only search, which is unaffected and still
        # uses the exact, calibrated-depth-capped TreeExplainer route.
        # Not 'tunable', for the same reason as every other
        # interaction-search setting in this schema (an explainability
        # side-computation, forced off during a hyperparameter search).
        {'label': 'Coalition sample budget for 3-way-or-higher shapiq_tree search (shapiq_tree method only)',
         'type': 'int',
         'default': 4096,
         'depends_on': (5, True),
         'visible_when': (9, 'shapiq_tree'),
         'help': "Only used when 'Marker interaction method' above is 'shapiq_tree' AND 'Interaction "
                 "orders to search for' above includes an order greater than 2 (a 3-way-or-higher "
                 "search). shapiq's own exact tree-walking algorithm becomes exponentially more "
                 "expensive per extra order, so a 3-way-or-higher search instead uses a "
                 "budget-bounded, sampling-based estimate on this model's own real fitted forest - "
                 "this sets how many marker-coalition samples that estimate is allowed, directly "
                 "trading accuracy for speed (higher = more accurate but slower; lower = faster but "
                 "noisier). Has no effect on a 2-way-only search, which remains exact and unaffected."}],
 'ExtraTrees': [{'label': 'Tree number',
         'type': 'int',
         'default': 100,
         'help': 'How many individual decision trees to average together. More trees usually give '
                 'steadier, more reliable predictions, at the cost of longer runtime.',
         'tunable': {'low': 100, 'high': 2000, 'step': 100, 'floor_at_baseline': True}},
        {'label': 'Maximum features per tree',
         'type': 'rf_max_features',
         'default': '1.0',
         'choices': ['sqrt', 'log2', 'None'],
         'combo_state': 'normal',
         'help': "How many markers each tree is allowed to consider at every split. Extremely "
                 'Randomised Trees pick the SPLIT THRESHOLD at random rather than searching for '
                 "the best one, so this setting matters somewhat less here than for RF, but still "
                 "controls how much randomness each tree sees.",
         'tunable': {'choices': ['sqrt', 'log2', 0.5, 0.8, 1.0], 'priority': True}},
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
         'tunable': {'low': 1, 'high': 20, 'step': 1, 'priority': True}},
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
                 'thousands of markers.'},
        # Requirements3.md item 2 - see RF's identical field for the full
        # rationale. Appended, never inserted (I5) - this list's own
        # length was 9 (terminal index 8) before this edit.
        # Requirements.md items 2/3 (this update) - see RF's identical
        # field for the full rationale: 'shapiq' added to 'choices'
        # (consistency/tooling parity with RF/SVR/KNN/RKHS, though 'shap'
        # above is already exact and usually faster); 'depends_on' added,
        # pointing at the new master 'Return marker effect?' toggle (11,
        # below).
        {'label': 'Marker effect method',
         'type': 'str',
         'default': 'feature_importance',
         'legacy_default': 'feature_importance',
         'choices': ['feature_importance', 'shap', 'shapiq'],
         'combo_state': 'readonly',
         'depends_on': (11, True),
         'help': ("How the per-marker effect (reported when 'Return marker effect?' above is "
                  "ticked) is computed.\n"
                  "\n"
                  "- **feature_importance:** the fitted forest's built-in "
                  "impurity-based importance.\n"
                  "- **shap (default):** exact TreeSHAP values on a sample of test individuals, summing "
                  "their magnitudes as every other Shapley-based model does. Directly "
                  "comparable with those, at extra cost.\n"
                  "\n"
                  "The settings below only apply when this is 'shap'.")},
        # Requirements.md item 2/3 (this update) - see RF's identical field
        # for the full rationale: label/gating generalised from "'shap'
        # only" to "'shap' or 'shapiq'" and reused (not duplicated) as the
        # 'shapiq' route's own row-count setting; renamed to this update's
        # 'Number of individuals used to explain the model' phrasing;
        # 'depends_on' now points at the new master toggle (11) rather than
        # the method field itself, matching RF's identical field.
        {'label': 'Number of individuals used to explain the model (effect method "shap" or "shapiq" only)',
         'type': 'int',
         'default': 30,
         'legacy_default': 30,
         'depends_on': (11, True),
         'visible_when': (9, ('shap', 'shapiq')),
         'help': "Only used when 'Marker effect method' above is 'shap' or 'shapiq'. How many test "
                 "individuals are explained for the marker-effect calculation."},
        # Requirements.md item 3 (this update) - see RF's identical field
        # for the full rationale: the new master toggle for this model's
        # own marker-EFFECT step (separate from 'Return marker effect for
        # interactions?' above, field 5, which only gates the marker-PAIR
        # interaction search). Appended, never inserted (I5); default True
        # reproduces every existing config's behaviour exactly (this
        # model's marker effect was always computed unconditionally before
        # this field existed) - note this default is True even though
        # field 5's own default is False (see that field's own comment):
        # the two toggles gate two independent, pre-existing behaviours
        # that were never symmetric to begin with.
        {'label': 'Return marker effect?',
         'type': 'bool',
         'default': True,
         'help': ("If ticked (default), also computes each marker's contribution to the "
                  "prediction, in addition to the prediction itself, using the 'Marker effect "
                  "method' selected above. Untick to skip this step and save time when the "
                  "marker effects are not needed.")},
        # Requirements.md item 2 (this update) - see RF's identical three
        # fields for the full rationale: dedicated to the new 'shapiq'
        # marker-effect method specifically, kept separate from the
        # existing interaction-section fields above per this project's
        # "respective fields for both marker effects and interactions"
        # convention. Appended, never inserted (I5).
        {'label': 'Max markers considered for marker effect (shapiq) ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (11, True),
         'visible_when': (9, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. Only the top markers "
                 "(ranked by this forest's own feature_importances_) are explained; every other "
                 "marker is reported as 0. Fewer markers = much faster. 'all' explains every "
                 "marker but can take a long time on datasets with thousands of markers."},
        {'label': 'Background sample size for marker effect (shapiq)',
         'type': 'int',
         'default': 50,
         'depends_on': (11, True),
         'visible_when': (9, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. A small reference set "
                 "of training individuals used as a 'typical' baseline when working out each "
                 "marker's contribution. Smaller = faster; larger = smoother, more stable scores "
                 "but slower."},
        {'label': 'Number of coalition samples for marker effect (shapiq)',
         'type': 'int',
         'default': 200,
         'depends_on': (11, True),
         'visible_when': (9, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. How many random "
                 "combinations of markers are tried out to estimate each marker's contribution to "
                 "a prediction. Higher = more accurate but slower; lower = faster but noisier "
                 "scores."},
        # This update - see RF's identical field for the full rationale.
        # Appended, never inserted (I5) - this list's own length was 15
        # (terminal index 14) before this edit.
        {'label': 'Use marker effects to select markers for interaction search?',
         'type': 'bool',
         'default': True,
         'depends_on': (5, True),
         'help': "If checked (default), and 'Max markers considered for interaction search' above "
                 "is 'all' or no larger than this model's own marker-effect coverage, the "
                 "interaction-search shortlist reuses this model's own already-computed "
                 "marker-effect values (from 'Marker effect method' above) instead of a separate "
                 "feature-importance ranking. Uncheck to always rank by this forest's own "
                 "feature_importances_ for the interaction shortlist specifically, independent of "
                 "'Marker effect method'. Hidden automatically (falls back to feature_importances_) "
                 "whenever the interaction shortlist would need MORE markers than this model's own "
                 "marker-effect step covers."}],
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
                 'but can take a very long time on datasets with thousands of markers.'},
        # Requirements.md item 2 (this update): GBDT is engine-only -
        # hidden from AVAILABLE_MODELS (main_app.py), so these fields are
        # not reachable from the GUI panel the way RF/ExtraTrees/XGBoost's
        # identical fields are, only via a headless/JSON config (the same
        # way every one of this model's OTHER hyperparameters already
        # works - see main_app.py's own resolve_hparams() call, which
        # resolves this model's params purely from these defaults since
        # no widget is ever rendered for it). Unlike RF/ExtraTrees/
        # XGBoost, GBDT has no existing per-marker-effect method selector
        # at all (module docstring: no free feature_importances_, so its
        # effect has always come, unconditionally, from permutation_
        # importance on a correlation-shortlisted refit) and no comparable
        # 'Return marker effect for interactions?'-style toggle to reorder
        # around (Requirements.md item 3 is scoped to RF/ExtraTrees/
        # XGBoost only) - so only a method selector is added here, no new
        # master toggle (the effect stays unconditional, exactly as
        # before). 'shapiq' computes a model-agnostic order-1 Shapley
        # marker effect (pipeline_utils.shapiq_marker_effects()) as an
        # alternative to permutation importance - a genuinely different
        # (not merely index-equivalent) quantity, unlike RF/ExtraTrees/
        # XGBoost's own 'shap' vs 'shapiq' (both exactly the same order-1
        # Shapley value there); 'permutation_importance' (default,
        # unchanged) reproduces every existing config's behaviour exactly
        # (I11). Appended, never inserted (I5).
        {'label': 'Marker effect method',
         'type': 'str',
         'default': 'permutation_importance',
         'legacy_default': 'permutation_importance',
         'choices': ['permutation_importance', 'shapiq'],
         'combo_state': 'readonly',
         'help': ("How the per-marker effect (always computed, unlike the optional toggle of "
                  "RF / ExtraTrees / XGBoost) is estimated.\n"
                  "\n"
                  "- **permutation_importance (default):** measures how much shuffling each "
                  "marker degrades the model's predictions, on a correlation-shortlisted "
                  "refit.\n"
                  "- **shapiq:** an order-1 Shapley marker effect computed with the optional "
                  "shapiq package. A genuinely different quantity, directly comparable with "
                  "the other Shapley-based models.\n"
                  "\n"
                  "The settings below only apply when this is 'shapiq'.")},
        {'label': 'Max markers considered for marker effect (shapiq) ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'visible_when': (10, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. Only the top markers "
                 "(by the same correlation shortlist this model's own permutation-importance route "
                 "already uses) are explained; every other marker is reported as 0. Fewer markers "
                 "= much faster."},
        {'label': 'Background sample size for marker effect (shapiq)',
         'type': 'int',
         'default': 50,
         'visible_when': (10, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. A small reference set "
                 "of training individuals used as a 'typical' baseline when working out each "
                 "marker's contribution. Smaller = faster; larger = smoother, more stable scores "
                 "but slower."},
        {'label': 'Number of coalition samples for marker effect (shapiq)',
         'type': 'int',
         'default': 200,
         'visible_when': (10, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. How many random "
                 "combinations of markers are tried out to estimate each marker's contribution to "
                 "a prediction. Higher = more accurate but slower; lower = faster but noisier "
                 "scores."}],
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
         'tunable': {'choices': [3, 4, 6, 8, 10], 'priority': True}},
        {'label': 'Learning rate',
         'type': 'float',
         'default': 0.1,
         'help': 'How much each new tree is allowed to correct the previous rounds. Smaller values '
                 'need more boosting rounds but often generalise better.',
         # Update ID ver4-6, R1.2(b): log-scaled (see the RKHS Kernel
         # bandwidth field's own comment for the full rationale).
         'tunable': {'low': 0.01, 'high': 0.3, 'scale': 'log', 'priority': True}},
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
         'tunable': {'low': 0.0, 'high': 5.0, 'priority': True}},
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
                 'thousands of markers.'},
        # Requirements3.md item 2 - see RF's identical field for the full
        # rationale. Appended, never inserted (I5) - this list's own
        # length was 10 (terminal index 9) before this edit.
        # Requirements.md items 2/3 (this update) - see RF's identical
        # field for the full rationale: 'shapiq' added to 'choices';
        # 'depends_on' added, pointing at the new master 'Return marker
        # effect?' toggle (12, below).
        {'label': 'Marker effect method',
         'type': 'str',
         'default': 'feature_importance',
         'legacy_default': 'feature_importance',
         'choices': ['feature_importance', 'shap', 'shapiq'],
         'combo_state': 'readonly',
         'depends_on': (12, True),
         'help': ("How the per-marker effect (reported when 'Return marker effect?' above is "
                  "ticked) is computed.\n"
                  "\n"
                  "- **feature_importance:** the fitted model's built-in gain-based "
                  "importance.\n"
                  "- **shap (default):** exact TreeSHAP values on a sample of test individuals, summing "
                  "their magnitudes as every other Shapley-based model does. Directly "
                  "comparable with those, at extra cost.\n"
                  "\n"
                  "The settings below only apply when this is 'shap'.")},
        # Requirements.md items 2/3 (this update) - see RF's identical
        # field for the full rationale: label/gating generalised from
        # "'shap' only" to "'shap' or 'shapiq'" and reused (not
        # duplicated) as the 'shapiq' route's own row-count setting;
        # renamed to this update's 'Number of individuals used to explain
        # the model' phrasing; 'depends_on' now points at the new master
        # toggle (12) rather than the method field itself.
        {'label': 'Number of individuals used to explain the model (effect method "shap" or "shapiq" only)',
         'type': 'int',
         'default': 30,
         'legacy_default': 30,
         'depends_on': (12, True),
         'visible_when': (10, ('shap', 'shapiq')),
         'help': "Only used when 'Marker effect method' above is 'shap' or 'shapiq'. How many test "
                 "individuals are explained for the marker-effect calculation."},
        # Requirements.md item 3 (this update) - see RF's identical field
        # for the full rationale: the new master toggle for this model's
        # own marker-EFFECT step (separate from 'Return marker effect for
        # interactions?' above, field 6, which only gates the marker-PAIR
        # interaction search). Appended, never inserted (I5); default True
        # reproduces every existing config's behaviour exactly.
        {'label': 'Return marker effect?',
         'type': 'bool',
         'default': True,
         'help': ("If ticked (default), also computes each marker's contribution to the "
                  "prediction, in addition to the prediction itself, using the 'Marker effect "
                  "method' selected above. Untick to skip this step and save time when the "
                  "marker effects are not needed.")},
        # Requirements.md item 2 (this update) - see RF's identical three
        # fields for the full rationale: dedicated to the new 'shapiq'
        # marker-effect method specifically, kept separate from the
        # existing interaction-section fields above. Appended, never
        # inserted (I5).
        {'label': 'Max markers considered for marker effect (shapiq) ("all" for every marker)',
         'type': 'int_or_all',
         'default': 500,
         'depends_on': (12, True),
         'visible_when': (10, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. Only the top markers "
                 "(ranked by this model's own gain-based importance) are explained; every other "
                 "marker is reported as 0. Fewer markers = much faster. 'all' explains every "
                 "marker but can take a long time on datasets with thousands of markers."},
        {'label': 'Background sample size for marker effect (shapiq)',
         'type': 'int',
         'default': 50,
         'depends_on': (12, True),
         'visible_when': (10, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. A small reference set "
                 "of training individuals used as a 'typical' baseline when working out each "
                 "marker's contribution. Smaller = faster; larger = smoother, more stable scores "
                 "but slower."},
        {'label': 'Number of coalition samples for marker effect (shapiq)',
         'type': 'int',
         'default': 200,
         'depends_on': (12, True),
         'visible_when': (10, 'shapiq'),
         'help': "Only used when 'Marker effect method' above is 'shapiq'. How many random "
                 "combinations of markers are tried out to estimate each marker's contribution to "
                 "a prediction. Higher = more accurate but slower; lower = faster but noisier "
                 "scores."},
        # This update - see RF's identical field for the full rationale.
        # Appended, never inserted (I5) - this list's own length was 16
        # (terminal index 15) before this edit.
        {'label': 'Use marker effects to select markers for interaction search?',
         'type': 'bool',
         'default': True,
         'depends_on': (6, True),
         'help': "If checked (default), and 'Max markers considered for interaction search' above "
                 "is 'all' or no larger than this model's own marker-effect coverage, the "
                 "interaction-search shortlist reuses this model's own already-computed "
                 "marker-effect values (from 'Marker effect method' above) instead of a separate "
                 "importance ranking. Uncheck to always rank by this model's own gain-based "
                 "importance for the interaction shortlist specifically, independent of 'Marker "
                 "effect method'. Hidden automatically (falls back to gain-based importance) "
                 "whenever the interaction shortlist would need MORE markers than this model's own "
                 "marker-effect step covers."}],
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
          'tunable': {'choices': ['linear', 'poly', 'rbf', 'sigmoid'], 'priority': True}},
         {'label': 'Epsilon',
          'type': 'float',
          'default': 0.5,
          'help': 'A margin of error the model is allowed to ignore - predictions within epsilon '
                  "of the true value aren't penalised at all. Larger epsilon gives a simpler, less "
                  'sensitive model; smaller epsilon tries to fit the data more closely.',
          # Update ID ver4-6, R1.2(b): log-scaled (see the RKHS Kernel
          # bandwidth field's own comment for the full rationale).
          'tunable': {'low': 0.001, 'high': 2.0, 'scale': 'log', 'priority': True}},
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
          'tunable': {'low': 0.01, 'high': 100.0, 'scale': 'log', 'priority': True}},
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
         # Requirements.md item 1: these three fields only affect the
         # 'friedman_h' interaction route (field 18, "Marker interaction
         # method", defined below) - meaningless for 'shapiq', which has
         # its own three dedicated settings instead. 'visible_when' added
         # so the panel shows exactly the settings relevant to whichever
         # interaction method is currently selected.
         {'label': 'Interaction pre-screen (speed vs. completeness)',
          'type': 'str',
          'default': 'off',
          'choices': ['off', 'marginal_pd'],
          'combo_state': 'readonly',
          'depends_on': (11, True),
          # Update (perf): also shown for 'baseline_shap' now that route
          # supports the SAME screen - baseline_shap has no grid concept,
          # so only this field and the one below it (never the grid-
          # points field further down) extend to it.
          'visible_when': (18, ('friedman_h', 'baseline_shap')),
          'help': ("Only used when 'Marker interaction method' below is 'baseline_shap'.\n"
                   "\n"
                   "- **off (default):** evaluates every candidate marker pair exactly. "
                   "Correct, but slow when many markers are shortlisted (SVR has no native "
                   "pairwise-interaction method, so every pair costs a real prediction call).\n"
                   "- **marginal_pd:** ranks pairs cheaply by how much each marker's own "
                   "effect varies, then computes the exact interaction score only for the "
                   "strongest-ranked pairs. Much faster, but approximate: a pair whose "
                   "markers both look unremarkable alone yet interact strongly (pure "
                   "epistasis) can be missed, and unselected pairs are reported as having no "
                   "interaction.")},
         {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (11, True),
          'visible_when': (18, ('friedman_h', 'baseline_shap')),
          'help': ("Only used when 'Marker interaction method' below is 'baseline_shap' and "
                   "the pre-screen above is not 'off'. The percentage of candidate pairs that "
                   "the pre-screen shortlists for an exact interaction score; for example, 2 "
                   "means the top 2% of pairs. Lower = faster but more likely to miss a real "
                   "interaction; higher = slower but more thorough.")},
         {'label': 'Interaction grid points per marker (3 = genotype classes)',
          'type': 'int',
          'default': 3,
          'depends_on': (11, True),
          'visible_when': (18, 'friedman_h'),
          'help': "Only used when 'Marker interaction method' below is 'friedman_h'. How many "
                  "representative values each marker's own interaction grid uses - 3 "
                  "matches this tool's usual 0/1/2 genotype coding exactly and is the right "
                  "choice for ordinary hard-called data. Raising this can help on fractional/"
                  "dosage-coded marker data (e.g. RIL/NAM populations, or PLINK data with some "
                  "missing calls) at higher cost (cost grows with the SQUARE of this number)."},
         # ver4-25, Stage 8 (design blueprint R11): appended, never
         # inserted (I5) - this list's own length was 18 (terminal index
         # 17) before this edit. SVR's own pairwise-interaction search has
         # always used Friedman's H-statistic (SVR has no native
         # pairwise-interaction API) - 'friedman_h' (unchanged default)
         # keeps that; 'shapiq' instead explains SVR's own `.predict`
         # directly with shapiq's Shapley interaction indices, and can
         # search for 3-or-more-way interactions (the three settings
         # below), not just pairs.
         {'label': 'Marker interaction method',
          'type': 'str',
          'default': 'friedman_h',
          'legacy_default': 'friedman_h',
          'choices': ['friedman_h', 'shapiq', 'kernel_hessian', 'baseline_shap'],
          'combo_state': 'readonly',
          'depends_on': (11, True),
          'help': ("How marker-pair interactions are searched for.\n"
                   "\n"
                   "- **baseline_shap:** evaluates only real genotype combinations observed "
                   "in the training population, so linkage disequilibrium is never violated, "
                   "and it works for every kernel. It needs no coalition-sampling budget (a "
                   "fixed, predictable cost), and its values use the same convention as the "
                   "RF TreeSHAP interaction route, so they sit on the same scale for a "
                   "side-by-side comparison with RF.\n"
                   "\n"
                   "If this route fails, EasiGP falls back to Friedman's H-statistic for that "
                   "task and says so in the run log. It never silently reports a possibly "
                   "wrong result.")},
         {'label': 'Interaction orders to search for (shapiq method only, e.g. "2" or "2,3")',
          'type': 'str',
          'default': '2',
          'legacy_default': '2',
          'depends_on': (11, True),
          'visible_when': (18, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. Which "
                  "interaction orders (k = how many markers act together) to search for, as a "
                  'comma-separated list - e.g. "2" for pairs only, or "2,3" to also search for '
                  'three-way interactions.'},
         {'label': 'Shapley interaction index (shapiq method only)',
          'type': 'str',
          'default': 'k-SII',
          'legacy_default': 'k-SII',
          'choices': ['k-SII', 'FSII', 'STII'],
          'combo_state': 'readonly',
          'depends_on': (11, True),
          'visible_when': (18, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. Which axiomatic "
                  "Shapley interaction index shapiq computes - 'k-SII' (default) is a good "
                  'general-purpose choice.'},
         {'label': 'Coalition budget per explained row (shapiq method only)',
          'type': 'int',
          'default': 4096,
          'legacy_default': 4096,
          'depends_on': (11, True),
          'visible_when': (18, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. How many marker "
                  "coalitions shapiq samples per explained test individual when an exact "
                  "computation is not feasible. Higher = more accurate but slower."},
         # ver4-25, Stage 7 (design blueprint R13, D5): SVR's own marker
         # EFFECT (not interaction) method - 'kernel_shap' (unchanged
         # default) is the existing shap.KernelExplainer route (fields
         # 7-10 above); 'shapiq' instead uses shapiq's own KernelSHAP/
         # exact-when-small explainer, sharing its background/imputer
         # with the interaction step above when both are set to a shapiq
         # method (models/SVR.py's own cost note).
         {'label': 'Marker effect method',
          'type': 'str',
          'default': 'kernel_shap',
          'legacy_default': 'kernel_shap',
          'choices': ['kernel_shap', 'shapiq', 'baseline_shap'],
          'combo_state': 'readonly',
          'depends_on': (6, True),
          'help': ("How the per-marker effect (Shapley scores, above) is computed.\n"
                   "\n"
                   "- **kernel_shap:** the 'shap' package's KernelExplainer.\n"
                   "- **baseline_shap (default):** scores every marker in the shortlist as the average "
                   "change in the prediction when that marker is swapped to the value of a "
                   "real training individual (the 'Background sample size for Shapley scores' "
                   "rows). It is the same estimator as the 'baseline_shap' interaction "
                   "method, evaluated in closed form for the kernel. Deterministic, preserves "
                   "linkage disequilibrium and is fast at any marker count, but its numbers "
                   "are on a different baseline and are not comparable with 'kernel_shap' "
                   "values.\n"
                   "\n"
                   "Changing this setting starts a fresh run (earlier results are not "
                   "reused).")},
         # ver4-29 (this update): which of shapiq's own approximators the
         # 'shapiq' interaction search (field 18) uses - appended, never
         # inserted (I5); this list's own length was 23 (terminal index
         # 22) before this edit, verified rather than assumed. See
         # pipeline_utils.shapiq_pairwise_interactions()'s own
         # `approximator` docstring for the full rationale: at this
         # model's own default 'Coalition budget per explained row'
         # (4096) against a several-hundred-marker shortlist, 'auto'
         # (unchanged default - a genuine no-op, I11) already resolves to
         # shapiq's own SPEX approximator internally, which this update's
         # own pipeline_utils.py changes now let run directly against the
         # full marker shortlist instead of being pre-narrowed away first
         # - fixing 'shapiq' interactions reporting mostly value=0.0
         # padded rows at this scale. Only 'auto' and 'spex' are offered
         # here (rather than every choice shapiq itself accepts) because
         # they are the two genuinely useful ones for this field's own
         # purpose (let shapiq choose automatically, or force the
         # large-marker-count-efficient approximator explicitly) -
         # 'montecarlo'/'svarm'/'permutation'/'regression' remain
         # reachable by hand-editing HPARAMETERS for a specific
         # comparison, but are not surfaced as GUI choices to keep this
         # field simple for the common case.
         {'label': 'Shapiq approximator (shapiq method only)',
          'type': 'str',
          'default': 'auto',
          'legacy_default': 'auto',
          'choices': ['auto', 'spex'],
          'combo_state': 'readonly',
          'depends_on': (11, True),
          'visible_when': (18, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. Which of "
                  "shapiq's own estimation algorithms searches for interactions. 'auto' "
                  "(default) lets shapiq choose - for a large marker shortlist (roughly 65+ "
                  "markers at pairs-only, fewer at higher orders) it already chooses 'spex', an "
                  "algorithm built specifically for a large number of markers under a limited "
                  "coalition budget, so 'auto' is usually the right choice. 'spex' forces that "
                  "same algorithm even for a smaller marker shortlist, where 'auto' would "
                  "otherwise pick a different (exact-when-small) estimator."},
         {'label': 'Shapiq approximator for marker effects (shapiq method only)',
          'type': 'str',
          'default': 'auto',
          'legacy_default': 'auto',
          'choices': ['auto', 'spex'],
          'combo_state': 'readonly',
          'depends_on': (6, True),
          'visible_when': (22, 'shapiq'),
          'help': ("Only used when 'Marker effect method' above is 'shapiq'. Which shapiq "
                   "estimation algorithm computes each marker's effect score.\n"
                   "\n"
                   "- **auto (default):** chosen by EasiGP. For a marker shortlist beyond 64 "
                   "markers it switches to 'spex' to avoid numerical instability.\n"
                   "- **spex:** forces that algorithm even for a smaller shortlist, where "
                   "'auto' would pick an estimator that is exact when small.")},
         # This update - see RF's identical field for the full rationale.
         # Appended, never inserted (I5) - this list's own length was 25
         # (terminal index 24) before this edit. Eligibility compares
         # this model's own effect cap (field 8, "Max markers considered
         # for Shapley scores" - applies to BOTH 'kernel_shap' and
         # 'shapiq' marker-effect methods here, unlike RF's split between
         # capped/uncapped routes) against the interaction cap (field 12).
         {'label': 'Use marker effects to select markers for interaction search?',
          'type': 'bool',
          'default': True,
          'depends_on': (11, True),
          'help': ("If ticked (default), and 'Max markers considered for interaction search' "
                   "above is 'all' or no larger than 'Max markers considered for Shapley "
                   "scores', the interaction-search shortlist reuses this model's "
                   "marker-effect (Shapley) values instead of a separate trait-correlation "
                   "ranking. Untick to always rank by trait correlation for the interaction "
                   "shortlist.\n"
                   "\n"
                   "This option is switched off automatically (trait correlation is used) "
                   "when the interaction shortlist needs more markers than the marker-effect "
                   "step covers, or when 'Return marker effect?' above is unticked.")},
         # ver4-38, R3 (design blueprint 'R3 - SVR constant predictions'):
         # appended, never inserted (I5) - this list's own length was 26
         # (terminal index 25) before this edit. Epsilon and Constraint (C)
         # are passed to scikit-learn on the RAW phenotype scale, so on a
         # trait whose SD is far below the default epsilon (0.5) every
         # training point sits inside the epsilon-tube, there are no support
         # vectors and the model predicts a constant. Scaling both by the
         # training-trait SD is mathematically the same as fitting the SVR
         # on y / SD (see models/SVR.py), so the fitted object, dual_coef_
         # and every explainer path are unchanged. NOT tunable: it changes
         # the meaning of fields 1 and 2 rather than being a fit-affecting
         # value to search over. The spec default is True (new GUI
         # sessions); a HPARAMETERS['SVR'] list that does NOT have this
         # index is read as False by models/SVR.py, so every pre-ver4-38
         # config still reproduces its old numbers (I11). 'legacy_default'
         # records that absent-value for readers of this file.
         {'label': 'Scale epsilon and C by the trait SD',
          'type': 'bool',
          'default': True,
          'legacy_default': False,
          'help': ("If ticked (recommended), Epsilon and Constraint are multiplied by the "
                   "standard deviation of the training trait, so they mean the same for a "
                   "trait measured in kilograms or in grams. Without this, an Epsilon larger "
                   "than the spread of the trait puts every individual inside the margin of "
                   "error, and the model predicts the same value for everyone (a constant "
                   "prediction, so the correlation with the true values is undefined).")}],
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
          'tunable': {'low': 1, 'high': 30, 'step': 1, 'priority': True}},
         {'label': 'Neighbour weighting',
          'type': 'str',
          'default': 'uniform',
          'choices': ['uniform', 'distance'],
          'combo_state': 'readonly',
          'help': "'uniform' treats every one of the k nearest neighbours equally when averaging "
                  "their phenotypes. 'distance' weights closer neighbours more heavily than "
                  'farther ones - often a better fit when genetic distance varies a lot within the '
                  'neighbourhood.',
          'tunable': {'choices': ['uniform', 'distance'], 'priority': True}},
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
          'help': ("If ticked, also searches for pairs of markers that interact, using the "
                   "'Marker interaction method' selected below. KNN has no built-in "
                   "pairwise-interaction method, so this is a model-agnostic estimate, not an "
                   "exact computation. The settings below only apply when this is ticked.\n"
                   "\n"
                   "**Known limitation:** KNN predicts by averaging a fixed set of nearest "
                   "training individuals, which can change abruptly as a marker value is "
                   "varied. Unlike a smooth model such as SVR, this can make interaction "
                   "scores look elevated for many marker pairs at once. If KNN's ring looks "
                   "like a much denser web of links than another model's, rely on the "
                   "relative ranking within KNN's own results and be cautious comparing raw "
                   "magnitudes across models. A larger neighbour count above and a larger "
                   "background sample below both help, though neither removes the effect "
                   "completely.")},
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
         # Requirements.md item 1: these three fields only affect the
         # 'friedman_h' interaction route (field 15, "Marker interaction
         # method", defined below) - meaningless for 'shapiq', which has
         # its own three dedicated settings instead. 'visible_when' added
         # so the panel shows exactly the settings relevant to whichever
         # interaction method is currently selected.
         {'label': 'Interaction pre-screen (speed vs. completeness)',
          'type': 'str',
          'default': 'off',
          'choices': ['off', 'marginal_pd'],
          'combo_state': 'readonly',
          'depends_on': (8, True),
          # Update (perf): also shown for 'baseline_shap' now that route
          # supports the SAME screen - baseline_shap has no grid concept,
          # so only this field and the one below it (never the grid-
          # points field further down) extend to it.
          'visible_when': (15, ('friedman_h', 'baseline_shap')),
          'help': ("Only used when 'Marker interaction method' below is 'baseline_shap'.\n"
                   "\n"
                   "- **off (default):** evaluates every candidate marker pair exactly. "
                   "Correct, but slow when many markers are shortlisted (KNN has no native "
                   "pairwise-interaction method, so every pair costs a real prediction call).\n"
                   "- **marginal_pd:** ranks pairs cheaply by how much each marker's own "
                   "effect varies, then computes the exact interaction score only for the "
                   "strongest-ranked pairs. Much faster, but approximate: a pair whose "
                   "markers both look unremarkable alone yet interact strongly (pure "
                   "epistasis) can be missed, and unselected pairs are reported as having no "
                   "interaction.")},
         {'label': 'Fraction of pairs scored exactly when pre-screening (%)',
          'type': 'top_pct',
          'default': 'all',
          'depends_on': (8, True),
          'visible_when': (15, ('friedman_h', 'baseline_shap')),
          'help': ("Only used when 'Marker interaction method' below is 'baseline_shap' and "
                   "the pre-screen above is not 'off'. The percentage of candidate pairs that "
                   "the pre-screen shortlists for an exact interaction score; for example, 2 "
                   "means the top 2% of pairs. Lower = faster but more likely to miss a real "
                   "interaction; higher = slower but more thorough.")},
         {'label': 'Interaction grid points per marker (3 = genotype classes)',
          'type': 'int',
          'default': 3,
          'depends_on': (8, True),
          'visible_when': (15, 'friedman_h'),
          'help': "Only used when 'Marker interaction method' below is 'friedman_h'. How many "
                  "representative values each marker's own interaction grid uses - 3 "
                  "matches this tool's usual 0/1/2 genotype coding exactly and is the right "
                  "choice for ordinary hard-called data. Raising this can help on fractional/"
                  "dosage-coded marker data (e.g. RIL/NAM populations, or PLINK data with some "
                  "missing calls) at higher cost (cost grows with the SQUARE of this number)."},
         # ver4-25, Stage 8/Stage 7 (design blueprint R11/R13) - see SVR's
         # identical five fields for the full rationale. Appended, never
         # inserted (I5) - this list's own length was 15 (terminal index
         # 14) before this edit.
         {'label': 'Marker interaction method',
          'type': 'str',
          'default': 'friedman_h',
          'legacy_default': 'friedman_h',
          'choices': ['friedman_h', 'shapiq', 'baseline_shap'],
          'combo_state': 'readonly',
          'depends_on': (8, True),
          'help': ("How marker-pair interactions are searched for.\n"
                   "\n"
                   "- **baseline_shap:** evaluates only real genotype combinations observed "
                   "in the training population, so linkage disequilibrium is never violated. "
                   "It needs no coalition-sampling budget (a fixed, predictable cost), and "
                   "its values use the same convention as the RF TreeSHAP interaction route, "
                   "so they sit on the same scale for a side-by-side comparison with RF.\n"
                   "\n"
                   "If this route fails, EasiGP falls back to Friedman's H-statistic for that "
                   "task and says so in the run log. It never silently reports a possibly "
                   "wrong result.")},
         {'label': 'Interaction orders to search for (shapiq method only, e.g. "2" or "2,3")',
          'type': 'str',
          'default': '2',
          'legacy_default': '2',
          'depends_on': (8, True),
          'visible_when': (15, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. Which "
                  "interaction orders (k = how many markers act together) to search for, as a "
                  'comma-separated list - e.g. "2" for pairs only, or "2,3" to also search for '
                  'three-way interactions.'},
         {'label': 'Shapley interaction index (shapiq method only)',
          'type': 'str',
          'default': 'k-SII',
          'legacy_default': 'k-SII',
          'choices': ['k-SII', 'FSII', 'STII'],
          'combo_state': 'readonly',
          'depends_on': (8, True),
          'visible_when': (15, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. Which axiomatic "
                  "Shapley interaction index shapiq computes - 'k-SII' (default) is a good "
                  'general-purpose choice.'},
         {'label': 'Coalition budget per explained row (shapiq method only)',
          'type': 'int',
          'default': 4096,
          'legacy_default': 4096,
          'depends_on': (8, True),
          'visible_when': (15, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. How many marker "
                  "coalitions shapiq samples per explained test individual when an exact "
                  "computation is not feasible. Higher = more accurate but slower."},
         {'label': 'Marker effect method',
          'type': 'str',
          # ver4-67: new GUI default 'baseline_shap' (true interventional
          # Shapley values - the same game the 'baseline_shap' INTERACTION
          # route uses, so effects and interactions are now consistent).
          # `legacy_default` stays 'kernel_shap' (the Relief-style route), so a
          # saved config that predates this choice reproduces exactly (I11).
          'default': 'baseline_shap',
          'legacy_default': 'kernel_shap',
          'choices': ['kernel_shap', 'shapiq', 'baseline_shap'],
          'combo_state': 'readonly',
          'depends_on': (3, True),
          'help': ("How the per-marker effect (Shapley scores, above) is computed.\n"
                   "\n"
                   "- **baseline_shap (default):** exact-in-expectation interventional "
                   "Shapley values against real training individuals (the same game as the "
                   "'baseline_shap' interaction method, so effects and interactions are "
                   "consistent). A KNN-specific fast algorithm keeps the cost per step "
                   "independent of the number of markers; groups of neighbouring markers are "
                   "refined to single markers only where their Shapley share is statistically "
                   "significant.\n"
                   "- **kernel_shap:** a Relief-style neighbour-distance decomposition. It is "
                   "not a Shapley value.")},
         # This update - see RF's/SVR's identical field for the full
         # rationale. Appended, never inserted (I5) - this list's own
         # length was 20 (terminal index 19) before this edit. Eligibility
         # compares this model's own effect cap (field 5, "Max markers
         # considered for Shapley scores" - applies to both 'kernel_shap'
         # and 'shapiq' marker-effect methods here) against the
         # interaction cap (field 9).
         {'label': 'Use marker effects to select markers for interaction search?',
          'type': 'bool',
          'default': True,
          'depends_on': (8, True),
          'help': ("If ticked (default), and 'Max markers considered for interaction search' "
                   "above is 'all' or no larger than 'Max markers considered for Shapley "
                   "scores', the interaction-search shortlist reuses this model's "
                   "marker-effect (Shapley) values instead of a separate trait-correlation "
                   "ranking. Untick to always rank by trait correlation for the interaction "
                   "shortlist.\n"
                   "\n"
                   "This option is switched off automatically (trait correlation is used) "
                   "when the interaction shortlist needs more markers than the marker-effect "
                   "step covers, or when 'Return marker effect?' above is unticked.")}],
 'MLP': [{'label': 'Neuron numbers',
          'type': 'int',
          'default': 30,
          'help': 'How many units are in the (first) hidden layer. More neurons let the network '
                  'represent more complex patterns, but need more data to train reliably and are '
                  'more prone to overfitting.',
          'tunable': {'low': 8, 'high': 256, 'step': 8, 'priority': True}},
         {'label': 'Dropout',
          'type': 'float',
          'default': 0,
          'help': 'The fraction of hidden-layer connections randomly switched off during each '
                  'training step, as a safeguard against overfitting. 0 disables this; typical '
                  'values are 0.1-0.5 if overfitting is a concern.',
          'tunable': {'low': 0.0, 'high': 0.6, 'priority': True}},
         {'label': 'Learning rate',
          'type': 'float',
          'default': 0.0001,
          'help': 'How big a step the network takes when updating its weights after each batch. '
                  'Too high can make training unstable or fail to settle down; too low makes '
                  'training very slow to improve.',
          'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b)
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
          'help': ("Adds an extra hidden layer (with the same dropout rate as the first) "
                   "before the output. 'None' (default) keeps a single hidden layer. A deeper "
                   "network can capture more complex marker interactions, but is slower to "
                   "train and overfits more easily on small datasets."),
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
          'help': 'If checked, also searches for pairs of markers that interact with each other. '
                  "See 'Marker interaction method' below for the two ways this is computed. The "
                  'settings below only apply when this is checked.'},
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
                  'a very large table for datasets with many markers.'},
         # Requirements.md item 2 (this update) - see RF's identical field
         # for the full rationale. Unlike RF/ExtraTrees/XGBoost, MLP has
         # no existing per-marker-effect method selector at all (its
         # effect has always come, unconditionally, from shap.
         # DeepExplainer) and no comparable 'Return marker effect?'-style
         # toggle to add/reorder around (Requirements.md item 3 is scoped
         # to RF/ExtraTrees/XGBoost only) - so only a method selector is
         # added here, no new master toggle. This model's interaction
         # search (NID, field 8) reuses whichever effect vector this
         # field selects as its own importance-based shortlist ranking
         # (see the module's own comment above 'get_interaction' in
         # models/MLP.py) - unchanged behaviour, just now sourced from
         # 'shapiq' when that is selected instead of always from
         # 'deep_shap'. 'deep_shap' (default, unchanged) reproduces every
         # existing config's behaviour exactly (I11). Appended, never
         # inserted (I5).
         #
         # Unlike every other model's own 'shapiq' marker-effect option,
         # there is deliberately NO marker-shortlist field here: this
         # network's own trained first layer has a FIXED input width (one
         # unit per marker in the original training data - see the MLP
         # class's own __init__), so - unlike RF/SVR/ExtraTrees/XGBoost,
         # which can cheaply refit a SEPARATE, smaller model on a marker
         # subset - there is no lightweight way to explain a narrowed
         # column subset with THIS SAME trained network without retraining
         # it entirely (a much larger change than a shortlist elsewhere in
         # this codebase implies). 'shapiq' therefore always explains
         # every marker here, exactly like 'deep_shap' already
         # unconditionally does - only the ROW count and shapiq's own
         # background/coalition-budget settings below are configurable.
         {'label': 'Marker effect method',
          'type': 'str',
          'default': 'deep_shap',
          'legacy_default': 'deep_shap',
          'choices': ['deep_shap', 'shapiq'],
          'combo_state': 'readonly',
          'help': ("How the per-marker effect (always computed, over every marker, because "
                   "the network's input width is fixed) is estimated.\n"
                   "\n"
                   "- **deep_shap (default):** the shap package's DeepExplainer applied "
                   "directly to the trained network. It uses 'Number of samples for Shapley "
                   "scores' above for both its background and the rows explained.")},
         {'label': 'Background sample size for marker effect (shapiq)',
          'type': 'int',
          'default': 50,
          'visible_when': (11, 'shapiq'),
          'help': "Only used when 'Marker effect method' above is 'shapiq'. A small reference set "
                  "of training individuals used as a 'typical' baseline when working out each "
                  "marker's contribution. Smaller = faster; larger = smoother, more stable scores "
                  "but slower."},
         {'label': 'Number of coalition samples for marker effect (shapiq)',
          'type': 'int',
          'default': 200,
          'visible_when': (11, 'shapiq'),
          'help': "Only used when 'Marker effect method' above is 'shapiq'. How many random "
                  "combinations of markers are tried out to estimate each marker's contribution "
                  "to a prediction. Higher = more accurate but slower; lower = faster but noisier "
                  "scores."},
         # This update: gives MLP a genuine SHAP-based marker-PAIR-
         # interaction route, alongside its existing default (Neural
         # Interaction Detection, NID - pure matrix algebra over the
         # already-trained first-layer weights, free, but not a Shapley-
         # based quantity at all). 'shapiq' wraps this network's own
         # trained forward pass as a plain predict(X) -> array callable
         # (the SAME wrapper this model's own 'Marker effect method' /
         # 'shapiq' choice above already builds for the identical
         # reason) and searches for interactions with the optional
         # shapiq package's own Shapley interaction indices - the SAME
         # model-agnostic route SVR/KNN/RKHS already offer under this
         # same name. Appended, never inserted (I5); 'nid' (default)
         # reproduces every existing config's behaviour exactly (I11).
         {'label': 'Marker interaction method',
          'type': 'str',
          'default': 'nid',
          'legacy_default': 'nid',
          'choices': ['nid', 'shapiq'],
          'combo_state': 'readonly',
          'depends_on': (8, True),
          'help': ("How marker-pair interactions are searched for (only used when 'Return "
                   "marker-pair interactions?' above is ticked).\n"
                   "\n"
                   "- **nid (default):** Neural Interaction Detection, read directly from the "
                   "network's trained first-layer weights. It costs virtually nothing beyond "
                   "the training that already happened, but is an approximate ranking, not a "
                   "Shapley value.")},
         {'label': 'Background sample size for interaction (shapiq)',
          'type': 'int',
          'default': 50,
          'depends_on': (8, True),
          'visible_when': (14, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. A small "
                  "reference set of training individuals used as a 'typical' baseline when "
                  "working out each marker pair's interaction strength. Smaller = faster; larger "
                  "= smoother, more stable scores but slower."},
         {'label': 'Coalition budget per explained row for interaction (shapiq)',
          'type': 'int',
          'default': 200,
          'depends_on': (8, True),
          'visible_when': (14, 'shapiq'),
          'help': "Only used when 'Marker interaction method' above is 'shapiq'. How many marker "
                  "coalitions shapiq samples per explained row when an exact computation is not "
                  "feasible (more than ~12 shortlisted markers). Higher = more accurate but "
                  "slower; lower = faster but noisier. Exact (no sampling) is used automatically "
                  "whenever the shortlist is small enough."},
         # This update - see RF's identical field for the full rationale.
         # Appended, never inserted (I5) - this list's own length was 17
         # (terminal index 16) before this edit. UNLIKE every other model
         # this checkbox was added to, MLP's interaction shortlist
         # ALREADY reuses this model's own marker-effect values by
         # default (see 'Marker interaction method' field's own comment,
         # and models/MLP.py's 'get_interaction' block) - there was no
         # correlation-based route here before this update. This model's
         # own marker-effect step never caps how many markers it covers
         # (its trained input width is fixed - see 'Marker effect
         # method' field's own comment), so this checkbox is never
         # hidden for MLP (main_app.py's own dynamic show/hide check,
         # using the SAME shared eligibility function, always resolves
         # to "eligible" here) - unlike RF/ExtraTrees/XGBoost/SVR/KNN/
         # RKHS, where it can be.
         {'label': 'Use marker effects to select markers for interaction search?',
          'type': 'bool',
          'default': True,
          'depends_on': (8, True),
          'help': "If checked (default - matches this model's existing behaviour), the "
                  "interaction-search shortlist reuses this model's own already-computed "
                  "marker-effect values (from 'Marker effect method' above) - as it always has. "
                  "Uncheck for a NEW alternative: rank the interaction shortlist by trait "
                  "correlation instead, matching RF's/SVR's/KNN's own default convention."}],
 'GAT_infinitesimal': [{'label': 'Neuron numbers',
                        'type': 'int',
                        'default': 20,
                        'help': 'How many units are in each graph-attention layer. More neurons '
                                'let the network represent more complex patterns, but need more '
                                'data to train reliably and are more prone to overfitting.',
                        'tunable': {'low': 8, 'high': 128, 'step': 8, 'search_high': 64, 'priority': True}},
                       {'label': 'Dropout',
                        'type': 'float',
                        'default': 0,
                        'help': 'The fraction of attention connections randomly switched off '
                                'during each training step, as a safeguard against overfitting. 0 '
                                'disables this; typical values are 0.1-0.5 if overfitting is a '
                                'concern.',
                        'tunable': {'low': 0.0, 'high': 0.6, 'priority': True}},
                       {'label': 'Learning rate',
                        'type': 'float',
                        'default': 0.01,
                        'help': 'How big a step the network takes when updating its weights after '
                                'each batch. Too high can make training unstable or fail to settle '
                                'down; too low makes training very slow to improve.',
                        'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b)
                       {'label': 'Decay',
                        'type': 'float',
                        'default': 0.0005,
                        'help': "A small penalty that discourages the network's weights from "
                                'growing too large, as another safeguard against overfitting. 0 '
                                'disables it; larger values regularise more strongly.',
                        'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
                       {'label': 'Epoch',
                        'type': 'int',
                        'default': 40,
                        'help': 'How many times the network passes over the entire training set. '
                                'More epochs let it learn more, but too many can start memorising '
                                'noise in the training data rather than the underlying trend.',
                        'tunable': {'low': 10, 'high': 200, 'step': 10, 'run_length': True}},
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
                        'tunable': {'low': 1, 'high': 8, 'step': 1, 'search_high': 4}},
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
                          'tunable': {'low': 8, 'high': 128, 'step': 8, 'search_high': 64, 'priority': True}},
                         {'label': 'Dropout',
                          'type': 'float',
                          'default': 0,
                          'help': 'The fraction of attention connections randomly switched off '
                                  'during each training step, as a safeguard against overfitting. '
                                  '0 disables this; typical values are 0.1-0.5 if overfitting is a '
                                  'concern.',
                          'tunable': {'low': 0.0, 'high': 0.6, 'priority': True}},
                         {'label': 'Learning rate',
                          'type': 'float',
                          'default': 0.01,
                          'help': 'How big a step the network takes when updating its weights '
                                  'after each batch. Too high can make training unstable or fail '
                                  'to settle down; too low makes training very slow to improve.',
                          'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b)
                         {'label': 'Decay',
                          'type': 'float',
                          'default': 0.0005,
                          'help': "A small penalty that discourages the network's weights from "
                                  'growing too large, as another safeguard against overfitting. 0 '
                                  'disables it; larger values regularise more strongly.',
                          'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
                         {'label': 'Epoch',
                          'type': 'int',
                          'default': 40,
                          'help': 'How many times the network passes over the entire training set. '
                                  'More epochs let it learn more, but too many can start '
                                  'memorising noise in the training data rather than the '
                                  'underlying trend.',
                          'tunable': {'low': 10, 'high': 200, 'step': 10, 'run_length': True}},
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
                          'tunable': {'low': 1, 'high': 8, 'step': 1, 'search_high': 4}},
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
                          'tunable': {'low': 8, 'high': 128, 'step': 8, 'search_high': 64, 'priority': True}},
                         {'label': 'Dropout',
                          'type': 'float',
                          'default': 0,
                          'help': 'The fraction of attention connections randomly switched off '
                                  'during each training step, as a safeguard against overfitting. '
                                  '0 disables this; typical values are 0.1-0.5 if overfitting is a '
                                  'concern.',
                          'tunable': {'low': 0.0, 'high': 0.6, 'priority': True}},
                         {'label': 'Learning rate',
                          'type': 'float',
                          'default': 0.01,
                          'help': 'How big a step the network takes when updating its weights '
                                  'after each batch. Too high can make training unstable or fail '
                                  'to settle down; too low makes training very slow to improve.',
                          'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b)
                         {'label': 'Decay',
                          'type': 'float',
                          'default': 0.0005,
                          'help': "A small penalty that discourages the network's weights from "
                                  'growing too large, as another safeguard against overfitting. 0 '
                                  'disables it; larger values regularise more strongly.',
                          'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
                         {'label': 'Epoch',
                          'type': 'int',
                          'default': 40,
                          'help': 'How many times the network passes over the entire training set. '
                                  'More epochs let it learn more, but too many can start '
                                  'memorising noise in the training data rather than the '
                                  'underlying trend.',
                          'tunable': {'low': 10, 'high': 200, 'step': 10, 'run_length': True}},
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
                          'tunable': {'low': 1, 'high': 8, 'step': 1, 'search_high': 4}},
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
                                     'tunable': {'low': 8, 'high': 128, 'step': 8, 'search_high': 64, 'priority': True}},
                                    {'label': 'Dropout',
                                     'type': 'float',
                                     'default': 0,
                                     'help': 'The fraction of attention connections randomly '
                                             'switched off during each training step, as a '
                                             'safeguard against overfitting. 0 disables this; '
                                             'typical values are 0.1-0.5 if overfitting is a '
                                             'concern.',
                                     'tunable': {'low': 0.0, 'high': 0.6, 'priority': True}},
                                    {'label': 'Learning rate',
                                     'type': 'float',
                                     'default': 0.01,
                                     'help': 'How big a step the network takes when updating its '
                                             'weights after each batch. Too high can make training '
                                             'unstable or fail to settle down; too low makes '
                                             'training very slow to improve.',
                                     'tunable': {'low': 0.0001, 'high': 0.1, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b)
                                    {'label': 'Decay',
                                     'type': 'float',
                                     'default': 0.0005,
                                     'help': "A small penalty that discourages the network's "
                                             'weights from growing too large, as another safeguard '
                                             'against overfitting. 0 disables it; larger values '
                                             'regularise more strongly.',
                                     'tunable': {'low': 1e-6, 'high': 0.01, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b): low raised from 0.0 (log needs low>0)
                                    {'label': 'Epoch',
                                     'type': 'int',
                                     'default': 40,
                                     'help': 'How many times the network passes over the entire '
                                             'training set. More epochs let it learn more, but too '
                                             'many can start memorising noise in the training data '
                                             'rather than the underlying trend.',
                                     'tunable': {'low': 10, 'high': 200, 'step': 10, 'run_length': True}},
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
                                     'tunable': {'low': 1, 'high': 8, 'step': 1, 'search_high': 4}},
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
                                     'help': ("Internal, non-editable slot: it is always set "
                                              "from the 'Data-driven prior network' part of "
                                              "the 'Biological Prior Network' section in '2. "
                                              "Data Preprocessing' and is never shown or "
                                              "typed into directly.\n"
                                              "\n"
                                              "When disabled (default), markers outside every "
                                              "gene's window are excluded from the graph "
                                              "(their reported effect is exactly 0). When "
                                              "enabled, it carries the RF-filtering (and "
                                              "optional LD-pruning) settings used to select "
                                              "markers for a data-driven interaction network, "
                                              "which is merged into the biological network: "
                                              "an RF-selected marker outside every gene "
                                              "becomes its own graph node with real, "
                                              "data-driven edges (not just a self-loop).")},
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
                                             'the population, but take longer.'},
                                    # ver4-61, R4 (appended - I5): index 16. Not tunable: it
                                    # decides when training stops, not what the search
                                    # explores; search fits take their patience from
                                    # HP_TUNE_FAST['gat_search_early_stop'] instead.
                                    {'label': 'Early stopping patience (epochs, 0 = off)',
                                     'type': 'int',
                                     'default': 0,
                                     'legacy_default': 0,
                                     'help': ("Stops training when the validation error has "
                                              "not improved for this many epochs in a row, "
                                              "and keeps the weights from the best epoch.\n"
                                              "\n"
                                              "- 0 (default): train for the full number of "
                                              "epochs.\n"
                                              "- Needs a validation split (a 3-way ratio); "
                                              "without one this setting is ignored.\n"
                                              "- A value of 3-10 is typical; the bigger the "
                                              "model, the more it saves.")}]}

# GAT_infinitesimal_node_level - confirmed directly from
# models/GAT_infinitesimal_node_level.py's own params unpacking: neuron(0),
# dropout(1), lrate(2), decay(3), epoch(4), bsize(5), heads(6), samples(7),
# marker_effect(8) - note samples/marker_effect are swapped relative to
# GAT_infinitesimal (this is that model's genuine positional order, not a
# typo). Not part of main_app.py's own HPARAM_SPECS (see module docstring).
HPARAM_SPECS['GAT_infinitesimal_node_level'] = [
    {'label': 'Neuron numbers', 'type': 'int', 'default': 20,
     'tunable': {'low': 8, 'high': 128, 'step': 8, 'search_high': 64, 'priority': True}},
    {'label': 'Dropout', 'type': 'float', 'default': 0,
     'tunable': {'low': 0.0, 'high': 0.6, 'priority': True}},
    {'label': 'Learning rate', 'type': 'float', 'default': 0.01,
     'tunable': {'low': 1e-4, 'high': 0.1, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b)
    {'label': 'Decay', 'type': 'float', 'default': 5e-4,
     'tunable': {'low': 1e-6, 'high': 1e-2, 'scale': 'log', 'priority': True}},  # ver4-6 R1.2(b): low raised (log needs low>0)
    {'label': 'Epoch', 'type': 'int', 'default': 40,
     'tunable': {'low': 10, 'high': 200, 'step': 10, 'run_length': True}},
    {'label': 'Batch size', 'type': 'int', 'default': 8,
     'tunable': {'low': 4, 'high': 64, 'step': 4}},
    {'label': 'Number of heads', 'type': 'int', 'default': 1,
     'tunable': {'low': 1, 'high': 8, 'step': 1, 'search_high': 4}},
    {'label': 'Number of samples for marker effects', 'type': 'int', 'default': 30,
     'depends_on': (8, True)},
    {'label': 'Return marker effect?', 'type': 'bool', 'default': True},
]


# ---------------------------------------------------------------------------
# ExtraTrees / XGBoost: RF-style 'Marker interaction method' block
# ---------------------------------------------------------------------------
# Until now RF was the only tree model with a selectable marker-pair
# interaction route (pairwise_shap / friedman_h / shapiq_tree); ExtraTrees
# and XGBoost always used exact TreeSHAP. Both now get the SAME eight
# fields RF has, in the SAME relative order and with the SAME semantics,
# built by deep-copying RF's own field definitions (RF fields 9-15 and 23)
# so the three models can never drift apart. APPEND-ONLY (I5): the new
# fields go on the END of each model's existing list, so every existing
# positional index (and every saved config) is unchanged, and GP()'s
# right-padding fills an older/shorter list with these fields' own defaults
# ('pairwise_shap' = exactly the pre-existing behaviour, I11).
#
#   new offset  RF index  field
#   +0          9         Marker interaction method
#   +1          10        Interaction pre-screen ("friedman_h" only)
#   +2          11        Fraction of pairs scored exactly (pre-screen)
#   +3          12        Interaction grid points per marker ("friedman_h")
#   +4          13        Background sample size for H-statistic ("friedman_h")
#   +5          14        Interaction orders to search for ("shapiq_tree")
#   +6          15        Shapley interaction index ("shapiq_tree")
#   +7          23        Coalition sample budget, k>2 ("shapiq_tree")
#
# ExtraTrees: 16 -> 24 fields (method = 16, orders = 21).
# XGBoost:    17 -> 25 fields (method = 17, orders = 22).
import copy as _copy

_RF_INTERACTION_METHOD_SRC_INDICES = (9, 10, 11, 12, 13, 14, 15, 23)


def _append_rf_style_interaction_fields(model, interaction_toggle_idx):
    rf_spec = HPARAM_SPECS['RF']
    base = len(HPARAM_SPECS[model])
    remap = {src: base + k for k, src in enumerate(_RF_INTERACTION_METHOD_SRC_INDICES)}
    for src in _RF_INTERACTION_METHOD_SRC_INDICES:
        field = _copy.deepcopy(rf_spec[src])
        dep = field.get('depends_on')
        if dep:
            # RF's own gate is field 5 ('Return marker effect for
            # interactions?'); this model's equivalent toggle differs.
            field['depends_on'] = (interaction_toggle_idx, dep[1])
        vis = field.get('visible_when')
        if vis:
            field['visible_when'] = (remap[vis[0]], vis[1])
        if 'help' in field:
            # RF's help text says "forest"; ExtraTrees is a forest but
            # XGBoost is a boosted-tree ensemble - "tree ensemble" is
            # accurate for both.
            field['help'] = field['help'].replace('forest', 'tree ensemble')
        HPARAM_SPECS[model].append(field)
    return remap


_ET_INTERACTION_REMAP = _append_rf_style_interaction_fields('ExtraTrees', 5)
_XGB_INTERACTION_REMAP = _append_rf_style_interaction_fields('XGBoost', 6)


# ---------------------------------------------------------------------------
# GUI-hidden choices
# ---------------------------------------------------------------------------
# Some 'choices' entries are deliberately NOT offered in the GUI's drop-down
# (requested UI simplification) while remaining fully supported by the
# engine: the 'choices' list itself, every model file's own branch for the
# hidden value, and every hand-written / HPC config that already carries it
# are all untouched. main_app.render_field()/resolve_field() read the two
# keys this block attaches to a field:
#   'gui_hidden_choices' - values removed from the drop-down; a stale
#                          session/loaded value equal to one of them is
#                          replaced by the field's GUI default.
#   'gui_default'        - the value shown/used in the GUI when the field's
#                          own spec 'default' is one of the hidden choices
#                          (spec 'default' is left alone on purpose: GP()
#                          right-pads short configs with it, so changing it
#                          would silently change old configs' results).
# Keyed by model, then by the field's exact label.
_GUI_HIDDEN_CHOICES = {
    'RKHS': {
        'Marker effect method': ({'shapiq_predictor'}, 'shap_predictor'),
        'Marker interaction method': (
            {'friedman_h', 'shapiq', 'shapiq_predictor', 'kernel_hessian', 'friedman_h_predictor'},
            'baseline_shap'),
    },
    'RF': {
        'Marker effect method': ({'shapiq'}, None),
        'Marker interaction method': ({'friedman_h'}, None),
    },
    'ExtraTrees': {
        'Marker effect method': ({'shapiq'}, None),
        'Marker interaction method': ({'friedman_h'}, None),
    },
    'XGBoost': {
        'Marker effect method': ({'shapiq'}, None),
        'Marker interaction method': ({'friedman_h'}, None),
    },
    'SVR': {
        'Marker effect method': ({'shapiq'}, None),
        'Marker interaction method': ({'friedman_h', 'shapiq', 'kernel_hessian'}, 'baseline_shap'),
    },
    'KNN': {
        'Marker effect method': ({'shapiq'}, None),
        # KNN has no 'kernel_hessian' choice at all (SVR-only), so only
        # these two of the three requested names exist to hide here.
        'Marker interaction method': ({'friedman_h', 'shapiq'}, 'baseline_shap'),
    },
    'MLP': {
        'Marker effect method': ({'shapiq'}, None),
        'Marker interaction method': ({'shapiq'}, None),
    },
}


def _apply_gui_hidden_choices():
    for model, rules in _GUI_HIDDEN_CHOICES.items():
        by_label = {f['label']: f for f in HPARAM_SPECS[model]}
        for label, (hidden, gui_default) in rules.items():
            # Fail loudly at import on a typo: a silent no-op here would
            # just leave the option visible with no error anywhere.
            field = by_label[label]
            choices = field['choices']
            unknown = set(hidden) - set(choices)
            assert not unknown, f"{model}/{label}: hidden choice(s) {unknown} not in {choices}"
            visible = [c for c in choices if c not in hidden]
            assert visible, f"{model}/{label}: every choice would be hidden"
            spec_default = field['default']
            if spec_default in hidden:
                assert gui_default in visible, (
                    f"{model}/{label}: default {spec_default!r} is hidden, so a visible "
                    f"gui_default is required (got {gui_default!r}; visible {visible})")
            else:
                assert gui_default is None or gui_default == spec_default, (
                    f"{model}/{label}: gui_default given but spec default {spec_default!r} is visible")
                gui_default = spec_default
            field['gui_hidden_choices'] = sorted(hidden)
            field['gui_default'] = gui_default
            # The help text of a field describes only the choices the GUI offers
            # (the hidden engine choices are documented in the module notes above).


_apply_gui_hidden_choices()


# GUI-only default choices for the marker-effect and marker-interaction methods: Shapley-based
# methods first, so every model reports effects (and interactions) on a comparable footing.
# Only what a NEW GUI session shows changes: the spec 'default' stays as it is (GP() right-pads
# short or older configs with it, so changing it would change their results), and a config that
# lists the field explicitly is never affected. Keyed by model, then by the field's exact label.
_GUI_DEFAULT_OVERRIDES = {
    'rrBLUP': {'Marker effect method': 'closed_form'},
    'BayesB': {'Marker effect method': 'closed_form'},
    'RKHS': {'Marker effect method': 'baseline_shap'},
    'RF': {'Marker effect method': 'shap'},
    'ExtraTrees': {'Marker effect method': 'shap'},
    'XGBoost': {'Marker effect method': 'shap'},
    'SVR': {'Marker effect method': 'baseline_shap'},
}


def _apply_gui_default_overrides():
    for model, rules in _GUI_DEFAULT_OVERRIDES.items():
        by_label = {f['label']: f for f in HPARAM_SPECS[model]}
        for label, value in rules.items():
            field = by_label[label]
            hidden = set(field.get('gui_hidden_choices') or ())
            assert value in field['choices'] and value not in hidden, (
                f"{model}/{label}: GUI default {value!r} is not an offered choice")
            field['gui_default'] = value


_apply_gui_default_overrides()


def field_index(model, label):
    """Position of the field called `label` in `HPARAM_SPECS[model]`.

    ver4-61, R4: the 'all' substitution for the marker-effect sample count used
    to read `params[-1]`, which silently pointed at the wrong field once a
    field was appended after it. Looking the position up by label keeps
    that substitution correct however the list grows (I5).

    Args:
        model: a key of `HPARAM_SPECS` (a bio-prior instance suffix such as
            `_2` is NOT resolved here; pass the schema key).
        label: the field's exact `label`.

    Returns:
        The zero-based index.

    Raises:
        KeyError: unknown model or label (a programming error, never data).
    """
    for position, field in enumerate(HPARAM_SPECS[model]):
        if field['label'] == label:
            return position
    raise KeyError(f"hparam_specs.field_index: model {model!r} has no field labelled {label!r}")
