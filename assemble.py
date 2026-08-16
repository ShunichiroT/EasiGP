import pandas as pd
import os

import checkpoint_utils as _ckpt
from checkpoint_utils import check_batch_status


# Combine prediction results from all batches
def assemble(RESULT_NAME, expected_batches=None):
    """expected_batches: optional total number of batches submitted (e.g.
    Array end index + 1 from Step 1's HPC export). When given, batch IDs
    are assumed to run 0..expected_batches-1 (the usual Slurm/PBS array-job
    convention), so a batch missing at the very end (beyond the highest
    batch ID actually found on disk) is reported too, not just gaps between
    batches that did produce output. When omitted (None), only gaps between
    the lowest and highest batch ID actually found are reported."""

    prediction_train = pd.DataFrame()
    prediction_valid = pd.DataFrame()
    prediction_test = pd.DataFrame()
    metric = pd.DataFrame()
    marker_effect = pd.DataFrame()
    interaction = pd.DataFrame()
    weight = pd.DataFrame()
    attention = pd.DataFrame()
    hyperparameter = pd.DataFrame()  # store optimised hyperparameters found via HP_TUNE
    stats = pd.DataFrame()           # basic per-scenario statistics (split sizes, marker count)
    
    # Discover the actual batch IDs present on disk, and their status
    # (complete / incomplete-with-a-checkpoint / missing entirely) - via
    # checkpoint_utils.check_batch_status(), the SAME fast, read-only
    # discovery logic a GUI "check status" button can call on its own
    # without paying assemble()'s own (much slower, for many/large
    # batches) data-reading cost below - so the two can never disagree
    # about a given run's status.
    #
    # Requirement: make an INCOMPLETE batch (checkpointing - see
    # checkpoint_utils.py - means a batch that failed partway through now
    # DOES leave a Metric_<idx>.csv behind, just a partial one, unlike
    # before checkpointing existed) easy to notice and impossible to
    # silently fold into the "final" assembled results. These batches are
    # excluded from `batch_ids` below (their partial data is never merged
    # in) and reported separately from fully-missing batches, since the
    # two need different fixes: a missing batch never produced anything at
    # all, while an incomplete one has a specific, known task to resume
    # from.
    _status = check_batch_status(RESULT_NAME, expected_batches=expected_batches)
    batch_ids = _status['complete']
    incomplete_batches = _status['incomplete']
    missing_batches = _status['missing']

    if incomplete_batches:
        print(f"[assemble] WARNING: {len(incomplete_batches)} batch(es) STARTED but did NOT "
              f"finish - they hit an error partway through and only have PARTIAL results saved "
              f"(see checkpoint_utils.py). They are EXCLUDED from this assembly so incomplete "
              f"data doesn't silently end up in your final output:")
        for _b in incomplete_batches:
            print(f"[assemble]   - {_ckpt.describe_incomplete_batch(_b)}")

    if missing_batches:
        print(f"[assemble] WARNING: {len(missing_batches)} batch(es) produced no output "
              f"whatsoever and appear to be missing: {_ckpt.format_batch_id_list(missing_batches)}. "
              f"These never even started (or crashed before completing a single scenario) - "
              f"re-run Step 1 for exactly these batch ID(s) (e.g. as a custom Slurm/PBS array, or "
              f"one at a time) once the underlying problem is fixed, then re-run Step 2.")
    elif not incomplete_batches:
        if batch_ids:
            print(f"[assemble] All {len(batch_ids)} batch(es) found on disk were assembled "
                  f"successfully; no missing or incomplete batches detected"
                  + (f" (checked against an expected total of {expected_batches})." if expected_batches else "."))
        else:
            print(f"[assemble] WARNING: No batch output files (Metric_*.csv) were found in "
                  f"'./Result/{RESULT_NAME}/' - nothing to assemble. Check that Step 1 has actually "
                  f"been run (and completed) for this result name.")

    for i in batch_ids:
        if os.path.isfile('./Result/'+RESULT_NAME+'/Prediction_result_train_'+str(i)+'.csv'):
            try:
                prediction_train = pd.concat([prediction_train,
                                              pd.read_csv('./Result/'+RESULT_NAME+'/Prediction_result_train_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                prediction_train = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Prediction_result_valid_'+str(i)+'.csv'):
            try:
                prediction_valid = pd.concat([prediction_valid,
                                              pd.read_csv('./Result/'+RESULT_NAME+'/Prediction_result_valid_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                prediction_valid = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Prediction_result_test_'+str(i)+'.csv'):
            try:
                prediction_test = pd.concat([prediction_test,
                                             pd.read_csv('./Result/'+RESULT_NAME+'/Prediction_result_test_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                prediction_test = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Metric_'+str(i)+'.csv'):
            try:
                metric = pd.concat([metric,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/Metric_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                # Requirement (bugfix - a batch where every task was
                # legitimately skipped wasn't being recognised as
                # complete): `batch_ids` (the loop this is inside) is
                # `_status['complete']` above - ONLY batches
                # check_batch_status() already confirmed finished every
                # task WITHOUT error (their checkpoint file is gone). An
                # empty Metric_<idx>.csv for one of these can therefore
                # NEVER mean anything is broken - it means every task in
                # that batch was legitimately SKIPPED (e.g. no
                # phenotype/genotype data matched for any of its
                # scenarios, or every one fell below the configured
                # minimum data-point count - see genomic_prediction.py's
                # own per-task skip messages, already in this batch's own
                # run log, for the specific reason). This used to be
                # printed as a WARNING and treated the same as a
                # genuinely missing/broken batch, which then cascaded
                # into the final 'no usable output' check below reporting
                # a hard failure even when every batch had, in fact,
                # completed exactly as configured. A plain, informational
                # log line is what's actually warranted here - nothing
                # about this batch needs fixing, and it is NOT excluded
                # from the assembled totals or treated as missing.
                print(f"[assemble] Batch {i} completed successfully but had nothing to report "
                      f"(Metric_{i}.csv is empty) - every task in it was skipped, most likely "
                      f"because no matching phenotype/genotype data was found for any of its "
                      f"scenarios. See this batch's own run log for the specific reason(s).")
        if os.path.isfile('./Result/'+RESULT_NAME+'/Marker_effect_'+str(i)+'.csv'):
            try:
                marker_effect = pd.concat([marker_effect,
                                           pd.read_csv('./Result/'+RESULT_NAME+'/Marker_effect_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                marker_effect = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Interaction_'+str(i)+'.csv'):
            try:
                interaction = pd.concat([interaction,
                                         pd.read_csv('./Result/'+RESULT_NAME+'/Interaction_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                interaction = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Weight_'+str(i)+'.csv'):
            try:
                weight = pd.concat([weight,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/Weight_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                weight = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/Attention_'+str(i)+'.csv'):
            try:
                attention = pd.concat([attention,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/Attention_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                attention = pd.DataFrame()
        if os.path.isfile('./Result/'+RESULT_NAME+'/hyperparameter_'+str(i)+'.csv'):
            try:
                hyperparameter = pd.concat([hyperparameter,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/hyperparameter_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                hyperparameter = pd.DataFrame()
        # Requirement 6: combine every batch's own Basic_stats_<i>.csv the
        # same way as every other per-batch output above.
        if os.path.isfile('./Result/'+RESULT_NAME+'/Basic_stats_'+str(i)+'.csv'):
            try:
                stats = pd.concat([stats,
                                    pd.read_csv('./Result/'+RESULT_NAME+'/Basic_stats_'+str(i)+'.csv')])
            except pd.errors.EmptyDataError:
                stats = pd.DataFrame()

    prediction_train.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_train.csv',index=False)
    prediction_valid.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_valid.csv',index=False)
    prediction_test.to_csv('./Result/'+RESULT_NAME+'/Prediction_result_test.csv',index=False)
    metric.to_csv('./Result/'+RESULT_NAME+'/Metric.csv',index=False)
    marker_effect.to_csv('./Result/'+RESULT_NAME+'/Marker_effect.csv',index=False)
    interaction.to_csv('./Result/'+RESULT_NAME+'/Interaction.csv',index=False)
    
    if weight.shape[0] != 0: 
        weight.to_csv('./Result/'+RESULT_NAME+'/Weight.csv',index=False)
    if attention.shape[0] != 0:
        attention.to_csv('./Result/'+RESULT_NAME+'/Attention.csv',index=False)
    if hyperparameter.shape[0] != 0:
        hyperparameter.to_csv('./Result/'+RESULT_NAME+'/hyperparameter.csv',index=False)
    if stats.shape[0] != 0:
        stats.to_csv('./Result/'+RESULT_NAME+'/Basic_stats.csv',index=False)

    # Requirement: an empty `metric` (no batch produced any usable Metric_*
    # output) has no 'model'/'population'/'phenotype' columns to read,
    # which would otherwise crash pd.unique(metric['model']) below with a
    # cryptic KeyError - returning empty results here either way is
    # unavoidable and correct regardless of why `metric` is empty. But
    # WHY it's empty matters for what gets logged: `batch_ids` (see
    # above) only ever contains batches check_batch_status() already
    # confirmed complete, so if it's non-empty, every one of those
    # batches genuinely finished - they simply had nothing to report
    # (every task skipped, e.g. no matching phenotype/genotype data - see
    # the per-batch log lines above for specifics). That is a legitimate
    # outcome reflecting the data/configuration, not a pipeline failure,
    # and reporting it plainly here is the fix requested - a real ERROR
    # is only warranted when NO batch completed at all (`batch_ids` is
    # empty - see missing_batches/incomplete_batches above for why).
    if metric.shape[0] == 0:
        if batch_ids:
            print(f"[assemble] All {len(batch_ids)} batch(es) completed successfully, but none "
                  f"had anything to report - every task across every one of them was skipped "
                  f"(most likely no matching phenotype/genotype data for any scenario). "
                  f"Returning empty results; this reflects your data/configuration, not a "
                  f"pipeline failure - see each batch's own run log for the specific reason(s) "
                  f"its tasks were skipped.")
        else:
            print(f"[assemble] ERROR: No usable batch output was found to assemble for "
                  f"'{RESULT_NAME}' - returning empty results. See the missing-batch warning "
                  f"above for which batch(es) to re-run.")
        return (metric, prediction_train, prediction_test, marker_effect, interaction, attention,
                [], [], [], missing_batches, incomplete_batches)

    MODEL = pd.unique(metric['model']).tolist()
    # Strip weighted-ensemble pseudo-models (e.g. 'Nelder Mead') AND their
    # per-hyperparameter-tuning-method suffixed variants (e.g.
    # 'Nelder Mead__Random', 'Nelder Mead__Bayesian') - a plain '=='
    # membership check only catches the unsuffixed form, leaving suffixed
    # variants behind for downstream plotting code that expects MODEL to
    # contain only real prediction models.
    _WOPT_LABEL_PREFIXES = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation')
    MODEL = [m for m in MODEL if not any(m == p or m.startswith(p + '__') for p in _WOPT_LABEL_PREFIXES)]

    return metric, prediction_train, prediction_test, marker_effect, interaction, attention,\
            pd.unique(metric['population']).tolist(), pd.unique(metric['phenotype']).tolist(), MODEL,\
            missing_batches, incomplete_batches


# Skip re-combining every batch and instead load the already-assembled combined
# CSVs that a previous assemble() call wrote to disk. Useful when Step 2 has
# already been run successfully once and only the scatter/circos plotting
# step needs retrying (e.g. after fixing a QTL file or circos config issue) -
# re-running assemble() on every retry is wasteful, and for many batches,
# slow, when the combined files are already sitting there correctly.
def load_assembled(RESULT_NAME):

    result_dir = './Result/'+RESULT_NAME+'/'
    required = ['Metric.csv', 'Prediction_result_train.csv', 'Prediction_result_test.csv',
                'Marker_effect.csv', 'Interaction.csv']
    missing = [f for f in required if not os.path.isfile(result_dir+f)]
    if missing:
        raise FileNotFoundError(
            "Cannot skip assemble: the following previously-assembled file(s) are "
            f"missing from '{result_dir}': {', '.join(missing)}. Run Step 2 at least "
            "once with 'Skip assemble' unchecked first."
        )

    try:
        metric = pd.read_csv(result_dir+'Metric.csv')
    except pd.errors.EmptyDataError:
        # Requirement (bugfix - same distinction as assemble() itself,
        # see that function's own comment): an empty combined Metric.csv
        # here does NOT necessarily mean the previous assemble() call
        # failed - it's exactly what a successful assemble() call would
        # also produce if every batch completed but had nothing to
        # report (every task skipped, e.g. no matching phenotype/
        # genotype data for any scenario). Checking check_batch_status()
        # here makes the same distinction assemble() does: if nothing is
        # missing or incomplete, every batch that exists genuinely
        # finished, and an empty result is a legitimate outcome to
        # return, not a reason to raise.
        _status = check_batch_status(RESULT_NAME)
        if _status['complete'] and not _status['incomplete'] and not _status['missing']:
            print(f"[load_assembled] '{result_dir}Metric.csv' is empty, but every batch found "
                  f"completed successfully - every task across every one of them was skipped "
                  f"(most likely no matching phenotype/genotype data). Loading empty results; "
                  f"this reflects your data/configuration, not a failed assemble() run.")
            # Return the same empty shape assemble() itself returns for
            # this exact situation - metric has no 'model'/'population'/
            # 'phenotype' columns to read (see that function's own
            # comment), so nothing past this point can be computed from
            # it; there's nothing further to load either.
            return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                    pd.DataFrame(), [], [], [], [], _ckpt.find_incomplete_batches(RESULT_NAME))
        else:
            raise FileNotFoundError(
                f"Cannot skip assemble: '{result_dir}Metric.csv' exists but is empty (the previous "
                f"assemble() call likely found no usable batch output). Run Step 2 with 'Skip "
                f"assemble' unchecked to re-assemble from the individual batches instead."
            )

    def _read_or_empty(path):
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    prediction_train = _read_or_empty(result_dir+'Prediction_result_train.csv')
    prediction_test = _read_or_empty(result_dir+'Prediction_result_test.csv')
    marker_effect = _read_or_empty(result_dir+'Marker_effect.csv')
    interaction = _read_or_empty(result_dir+'Interaction.csv')

    attention = pd.DataFrame()
    if os.path.isfile(result_dir+'Attention.csv'):
        try:
            attention = pd.read_csv(result_dir+'Attention.csv')
        except pd.errors.EmptyDataError:
            attention = pd.DataFrame()

    MODEL = pd.unique(metric['model']).tolist()
    # Same prefix-aware cleanup as assemble() - see its comment for why a
    # plain '==' check isn't enough once tuning-method suffixes are in play.
    _WOPT_LABEL_PREFIXES = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation')
    MODEL = [m for m in MODEL if not any(m == p or m.startswith(p + '__') for p in _WOPT_LABEL_PREFIXES)]

    # No batch rescan happens here (that's the point of skipping assemble),
    # so there's nothing new to report about MISSING batches specifically -
    # but checking for batches that are still INCOMPLETE (a leftover
    # checkpoint file - see checkpoint_utils.py) is cheap (just a glob +
    # tiny JSON read, not re-reading every batch's full Metric_<idx>.csv)
    # and worth doing even here: those batches' partial data was already
    # excluded from whatever assemble() call produced the combined files
    # being loaded now, but the user should still be told they exist and
    # haven't been finished/re-assembled yet.
    missing_batches = []
    incomplete_batches = _ckpt.find_incomplete_batches(RESULT_NAME)
    if incomplete_batches:
        print(f"[assemble] NOTE: {len(incomplete_batches)} batch(es) still have an unfinished "
              f"checkpoint (excluded from the combined results being loaded here):")
        for _b in incomplete_batches:
            print(f"[assemble]   - {_ckpt.describe_incomplete_batch(_b)}")

    return metric, prediction_train, prediction_test, marker_effect, interaction, attention,\
            pd.unique(metric['population']).tolist(), pd.unique(metric['phenotype']).tolist(), MODEL,\
            missing_batches, incomplete_batches
