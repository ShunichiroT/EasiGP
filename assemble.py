import os
import time

import pandas as pd

import checkpoint_utils as _ckpt
from checkpoint_utils import check_batch_status
import batch_reader as _reader
from batch_reader import ResultSet


# Streaming-merge keys that are only ever (re-)written when they produced
# >= 1 data row for THIS run - matches today's
# `if x.shape[0] != 0: x.to_csv(...)` guards exactly (Weight/Attention/
# hyperparameter/Basic_stats are all model-selection- or data-driven, not
# something every run necessarily produces - see
# checkpoint_utils.static_write_flags_for()). Every other key is written
# unconditionally, even when empty, exactly as the pre-Update-3 body below
# always has.
_CONDITIONAL_KEYS = ('weight', 'attention_total', 'hp_record', 'stats')


# Combine prediction results from all batches
def assemble(RESULT_NAME, expected_batches=None, *, streaming=True, legacy_return=False,
             RESULT_COMPRESSION='gzip'):
    """Merge every complete batch's per-batch result files into the
    combined `Result/<RESULT_NAME>/*.csv` files GP()'s plotting stage
    consumes.

    expected_batches: optional total number of batches submitted (e.g.
    Array end index + 1 from Step 1's HPC export). When given, batch IDs
    are assumed to run 0..expected_batches-1 (the usual Slurm/PBS array-job
    convention), so a batch missing at the very end (beyond the highest
    batch ID actually found on disk) is reported too, not just gaps between
    batches that did produce output. When omitted (None), only gaps between
    the lowest and highest batch ID actually found are reported.

    streaming (Update ID 3, R1, default True): the merge strategy.
        True  -> a two-pass, raw-CSV streaming merge (batch_reader.py) that
                 never holds more than one batch's rows resident and never
                 calls `pd.concat` in a loop - fixes the O(N^2)-copy peak
                 RSS the pre-Update-3 body below has. Read as
                 `cfg.get('ASSEMBLE_STREAMING', True)` by every caller.
        False -> runs the PRE-UPDATE-3 body completely unchanged (see
                 `_assemble_legacy()`) - the R1 rollback path, byte-for-byte
                 pre-Update-3 output UNDER RESULT_COMPRESSION='none' (see
                 that function's own updated docstring for what changes
                 under 'gzip' - Update ID ver4-9, R7), including that
                 body's own R1.2 accumulator-reset-on-`EmptyDataError`
                 behaviour.
        Either way, the return value is a `batch_reader.ResultSet` (see
        `legacy_return` below) - `streaming` only changes HOW the combined
        files get written, never what kind of object this function hands
        back to an R1/R2-aware caller.

    legacy_return (default False): when True, materialise and return the
        ORIGINAL 11-tuple (`metric, prediction_train, prediction_test,
        marker_effect, interaction, attention, populations, phenotypes,
        MODEL, missing_batches, incomplete_batches`) instead of a
        `ResultSet` - for a caller that has not been updated to the new
        attribute/method access (K3). Every R1/R2-aware call site in this
        codebase (run_step2_assemble.py, main_app.py's Parallel Step-2
        block) leaves this at its default and consumes the `ResultSet`
        directly; `ResultSet.__iter__` also yields this same tuple, so
        `metric, ... = assemble(...)` keeps working even without this flag.

    RESULT_COMPRESSION (Update ID ver4-9, R7, default `'gzip'`): the run's
        own `RESULT_COMPRESSION` config value, forwarded to
        `_assemble_streaming()`/`_assemble_legacy()`. Unlike
        `checkpoint_utils.result_file_paths()`'s own conservative,
        backward-compatible `compression=None` default, this parameter
        defaults to `'gzip'` - `assemble()` has exactly two callers in
        this codebase (run_step2_assemble.py,
        main_app.py's Step-2 in-process block), BOTH updated as part of
        this same R7 change to pass their own resolved
        `cfg.get('RESULT_COMPRESSION', 'gzip')` value explicitly, so there
        is no "untouched external caller" this default needs to protect -
        it simply matches the SAME global default every other new
        RESULT_COMPRESSION-aware entry point in this codebase uses.
    """
    if not streaming:
        # R1 rollback path (ASSEMBLE_STREAMING=False): the pre-Update-3
        # concat-loop body runs completely UNCHANGED below (kept, not
        # deleted) - it writes the SAME ten combined files, at the SAME
        # paths, that the streaming path writes, so wrapping its result in
        # a ResultSet(source='combined') reads back byte-for-byte the same
        # content this call itself just wrote, letting every R1/R2-aware
        # caller keep consuming a ResultSet regardless of which merge mode
        # actually ran.
        legacy_tuple = _assemble_legacy(RESULT_NAME, expected_batches, RESULT_COMPRESSION=RESULT_COMPRESSION)
        if legacy_return:
            return legacy_tuple
        (_metric, _pred_train, _pred_test, _marker_effect, _interaction, _attention,
         _population, _phenotype, _model, missing_batches, incomplete_batches) = legacy_tuple
        return ResultSet(RESULT_NAME, source='combined',
                          missing_batches=missing_batches, incomplete_batches=incomplete_batches)

    batch_ids, missing_batches, incomplete_batches = _assemble_streaming(
        RESULT_NAME, expected_batches, RESULT_COMPRESSION=RESULT_COMPRESSION,
    )
    result = ResultSet(RESULT_NAME, source='combined', batch_ids=batch_ids,
                        missing_batches=missing_batches, incomplete_batches=incomplete_batches)
    if legacy_return:
        return tuple(result)
    return result


def _batch_metric_is_empty(RESULT_NAME, batch_id):
    """True if this batch's own Metric_<batch_id>.csv exists but has zero
    data rows (zero-byte, or a header/blank first line with nothing after
    it). This is the ONE legacy per-batch informational case
    `_assemble_legacy()` printed about explicitly (its `metric` accumulator
    was the only one of the ten given a corrected, non-data-losing
    `EmptyDataError` handler pre-Update-3) - preserved verbatim here even
    though the streaming merge below never raises that exception itself
    (see `_CONDITIONAL_KEYS`'s sibling note in `_assemble_streaming()` for
    why the other nine keys no longer need an equivalent per-batch line:
    their own row COUNT, logged per key below, already shows the truth
    now that an empty batch no longer silently wipes the ones before it)."""
    path = _ckpt.result_file_paths(RESULT_NAME, batch_id, True)['record']
    if not os.path.isfile(path):
        return False
    with open(path, 'r', newline='', encoding='utf-8') as f:
        first_line = f.readline()
        if not first_line:
            return True  # zero-byte
        second_line = f.readline()
        return not second_line  # header-only (or a columnless blank first line) - no data rows


def _assemble_streaming(RESULT_NAME, expected_batches, RESULT_COMPRESSION='gzip'):
    """R1's streaming-merge path. Returns (batch_ids, missing_batches,
    incomplete_batches) - the caller wraps these into a ResultSet.
    Preserves every pre-Update-3 print() verbatim (this function's own
    docstring in `assemble()` above lists which); adds the new,
    unconditional per-key row-count/timing lines arch §17 requires.

    Update ID ver4-9, R7: `RESULT_COMPRESSION` - forwarded to
    `checkpoint_utils.result_file_paths()` (for `final_path` below) and to
    `batch_reader.merge_key_streaming()` (for its own write, since its
    temp `out_path` doesn't carry a `.gz` extension even when the
    eventual final path will - see that function's own docstring)."""
    # Discover the actual batch IDs present on disk, and their status
    # (complete / incomplete-with-a-checkpoint / missing entirely) - via
    # checkpoint_utils.check_batch_status(), the SAME fast, read-only
    # discovery logic a GUI "check status" button can call on its own
    # without paying assemble()'s own (much slower, for many/large
    # batches) data-reading cost below - so the two can never disagree
    # about a given run's status.
    #
    # An INCOMPLETE batch (checkpointing means a batch that failed partway
    # through now DOES leave a Metric_<idx>.csv behind, just a partial
    # one) is easy to notice and impossible to silently fold into the
    # "final" assembled results. These batches are excluded from
    # `batch_ids` below (their partial data is never merged in) and
    # reported separately from fully-missing batches, since the two need
    # different fixes: a missing batch never produced anything at all,
    # while an incomplete one has a specific, known task to resume from.
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

    # Requirement (R1.2 fix - a batch where every task was legitimately
    # skipped, e.g. no matching phenotype/genotype data for any of its
    # scenarios, is not an error): reported per batch, in ascending batch
    # order, exactly like the pre-Update-3 body's own `metric`-only
    # EmptyDataError handler did - see `_batch_metric_is_empty()`. The
    # streaming merge below never loses earlier batches' rows over this
    # (R1.2), so this is purely informational.
    for _b in batch_ids:
        if _batch_metric_is_empty(RESULT_NAME, _b):
            print(f"[assemble] Batch {_b} completed successfully but had nothing to report "
                  f"(Metric_{_b}.csv is empty) - every task in it was skipped, most likely "
                  f"because no matching phenotype/genotype data was found for any of its "
                  f"scenarios. See this batch's own run log for the specific reason(s).")

    result_dir = './Result/' + RESULT_NAME + '/'
    _t_total0 = time.time()
    rows_by_key = {}
    _final_paths = _ckpt.result_file_paths(RESULT_NAME, 0, False, compression=RESULT_COMPRESSION)
    for key, filename in _ckpt.RESULT_FILE_NAMES.items():
        _t_key0 = time.time()
        # Update ID ver4-9, R7: final_path now resolved through
        # checkpoint_utils.result_file_paths() (I8 - the single source of
        # truth for what a combined file is actually named), rather than
        # string-concatenated here - so this key's own '.gz'-or-not
        # decision can never drift from RESULT_FILE_COMPRESSION's own.
        final_path = _final_paths[key]
        tmp_path = final_path + '.tmp_assemble'
        n = _reader.merge_key_streaming(RESULT_NAME, key, batch_ids, out_path=tmp_path,
                                         compression=RESULT_COMPRESSION)
        rows_by_key[key] = n
        if key in _CONDITIONAL_KEYS and n == 0:
            # Matches today's `if x.shape[0] != 0: x.to_csv(...)` guard
            # exactly: when this run's merge produced NO rows for a
            # conditional key, no combined file is (re)written for it at
            # all - any PRE-EXISTING combined file from an earlier run is
            # left untouched, exactly as the old guard's "don't call
            # to_csv()" also left it untouched.
            #
            # Update ID ver4-9, R7: also removes a stale file at the
            # OTHER compression's own path, if one exists - e.g. a
            # previous run of this same RESULT_NAME wrote a real,
            # non-empty Weight.csv.gz under RESULT_COMPRESSION='gzip',
            # and this run (RESULT_COMPRESSION='none', or simply no
            # weighted method selected this time) produces zero weight
            # rows - without this, the OLD '.gz' would still be sitting
            # there, mismatched with (and silently preferred over, by
            # resolve_result_path()'s own "compressed checked first" rule)
            # whatever this run's own state actually is (RK-7, same
            # reasoning as clear_result_files()'s own fix).
            os.remove(tmp_path)
            _other_compression = None if RESULT_COMPRESSION == 'gzip' else 'gzip'
            _stale_path = _ckpt.result_file_paths(RESULT_NAME, 0, False, compression=_other_compression)[key]
            if _stale_path != final_path and os.path.isfile(_stale_path):
                try:
                    os.remove(_stale_path)
                except OSError as exc:
                    print(f"[assemble] WARNING: could not remove stale combined file "
                          f"'{_stale_path}': {exc!r}.")
        else:
            os.replace(tmp_path, final_path)
        # New, unconditional log line (arch §17): per-key rows merged and
        # elapsed time - this is also what "log one informational line
        # instead" (of the nine deleted EmptyDataError-reset handlers)
        # means in practice: the true, un-truncated row count for every
        # key is now always visible here, rather than silently reset to 0
        # by one empty batch's file.
        print(f"[assemble] merged key '{key}' ({filename}): {n} row(s) from "
              f"{len(batch_ids)} batch(es) in {time.time() - _t_key0:.2f}s.")
    print(f"[assemble] streaming merge (ASSEMBLE_STREAMING=True) completed "
          f"{len(_ckpt.RESULT_FILE_NAMES)} file(s) in {time.time() - _t_total0:.2f}s.")

    if rows_by_key['record'] == 0:
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

    return batch_ids, missing_batches, incomplete_batches


def _assemble_legacy(RESULT_NAME, expected_batches=None, RESULT_COMPRESSION='gzip'):
    """R1 rollback path (ASSEMBLE_STREAMING=False) - the pre-Update-3
    `assemble()` body, kept as close to verbatim as Update ID ver4-9, R7
    allows (never deleted) so setting the config flag off restores the
    SAME accumulate-then-concat merge strategy, including this body's own
    R1.2 accumulator-reset-on-`EmptyDataError` behaviour for every
    accumulator except `metric`. See `assemble()`'s own docstring for the
    `expected_batches` parameter.

    Update ID ver4-9, R7: this body's own "byte-for-byte pre-Update-3
    output" promise is now precise only UNDER `RESULT_COMPRESSION='none'`
    - every per-batch READ below is routed through
    `checkpoint_utils.resolve_result_path()` (a compressed-then-
    uncompressed existence probe, RK-6) rather than a hardcoded
    `os.path.isfile('./Result/.../X_<i>.csv')` check, and every combined
    WRITE is routed through `checkpoint_utils.result_file_paths(...,
    compression=RESULT_COMPRESSION)` rather than a hardcoded path string -
    so this path now ALSO reads/writes gzip for the six large keys when
    RESULT_COMPRESSION='gzip' is in effect (this codebase's own default),
    exactly like `_assemble_streaming()` does. What stays genuinely
    unchanged, regardless of RESULT_COMPRESSION, is the MERGE STRATEGY
    itself: `pd.concat()` in a loop, one call per batch per key - the
    property this rollback path exists to restore (e.g. if a future
    regression is ever suspected in the streaming merge's own row-
    realignment logic)."""

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
        # Update ID ver4-9, R7: each `_ckpt.resolve_result_path(RESULT_NAME,
        # i, True, key)` call probes BOTH the compressed and uncompressed
        # path for this batch/key and returns whichever exists (or `None`)
        # - the read-side counterpart to the write-side `compression=
        # RESULT_COMPRESSION` used below. `pd.read_csv()` infers gzip vs
        # plain transparently from the resolved path's own extension, so
        # no explicit `compression=` argument is needed at any read site
        # here.
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'result_train')
        if _path is not None:
            try:
                prediction_train = pd.concat([prediction_train, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                prediction_train = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'result_valid')
        if _path is not None:
            try:
                prediction_valid = pd.concat([prediction_valid, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                prediction_valid = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'result_test')
        if _path is not None:
            try:
                prediction_test = pd.concat([prediction_test, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                prediction_test = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'record')
        if _path is not None:
            try:
                metric = pd.concat([metric, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
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
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'effect')
        if _path is not None:
            try:
                marker_effect = pd.concat([marker_effect, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                marker_effect = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'interactions')
        if _path is not None:
            try:
                interaction = pd.concat([interaction, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                interaction = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'weight')
        if _path is not None:
            try:
                weight = pd.concat([weight, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                weight = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'attention_total')
        if _path is not None:
            try:
                attention = pd.concat([attention, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                attention = pd.DataFrame()
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'hp_record')
        if _path is not None:
            try:
                hyperparameter = pd.concat([hyperparameter, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                hyperparameter = pd.DataFrame()
        # Requirement 6: combine every batch's own Basic_stats_<i>.csv the
        # same way as every other per-batch output above.
        _path = _ckpt.resolve_result_path(RESULT_NAME, i, True, 'stats')
        if _path is not None:
            try:
                stats = pd.concat([stats, pd.read_csv(_path)])
            except _ckpt._INCOMPLETE_FILE_ERRORS:
                stats = pd.DataFrame()

    _out_paths = _ckpt.result_file_paths(RESULT_NAME, 0, False, compression=RESULT_COMPRESSION)

    def _to_csv(df, key):
        path = _out_paths[key]
        df.to_csv(path, index=False, compression='gzip' if path.endswith('.gz') else None)

    _to_csv(prediction_train, 'result_train')
    _to_csv(prediction_valid, 'result_valid')
    _to_csv(prediction_test, 'result_test')
    _to_csv(metric, 'record')
    _to_csv(marker_effect, 'effect')
    _to_csv(interaction, 'interactions')

    if weight.shape[0] != 0:
        _to_csv(weight, 'weight')
    if attention.shape[0] != 0:
        _to_csv(attention, 'attention_total')
    if hyperparameter.shape[0] != 0:
        _to_csv(hyperparameter, 'hp_record')
    if stats.shape[0] != 0:
        _to_csv(stats, 'stats')

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
    _WOPT_LABEL_PREFIXES = ('Linear transformation', 'Nelder Mead', 'Bayesian optimisation', 'Analytic least-squares')
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
#
# Update ID 3, R2: when the combined files DON'T exist (yet, or ever, e.g.
# a fresh Result folder pointed straight at "skip assemble"), this no
# longer just hard-raises - see `allow_batch_fallback` below.
#
# Patch 3, Requirement 1: `force_batch_source` (new) restores the THIRD,
# genuinely distinct "do not assemble" choice that `allow_batch_fallback`
# alone cannot express - see this parameter's own docstring below, and
# `load_for_step2()` further down for the single place that turns a Step
# 2 config's explicit choice among all three options into the right call
# here.
def load_assembled(RESULT_NAME, *, allow_batch_fallback=True, legacy_return=False,
                    force_batch_source=False):
    """Reload previously-assembled combined results without re-running the
    (batch) merge.

    allow_batch_fallback (Update ID 3, R2, default True): when the combined
        `Metric.csv` doesn't exist, and at least one COMPLETE batch does
        (per `checkpoint_utils.check_batch_status()`), return a
        `ResultSet(source='batches')` instead of raising - the SAME frames
        (same columns, same column order, same row order) a Step-2 assemble
        would have produced, just read straight from the per-batch files,
        WITHOUT writing any combined file. Read as
        `cfg.get('SKIP_ASSEMBLE_BATCH_FALLBACK', True)` by every caller;
        `False` restores today's hard `FileNotFoundError` unconditionally
        (the rollback path).

    force_batch_source (Patch 3, Requirement 1, default False): when True,
        ALWAYS build the result directly from the per-batch files
        (`ResultSet(source='batches')`), regardless of whether the
        combined files exist - i.e. genuinely "do not assemble", not
        merely "assemble was never run yet". Takes priority over
        `allow_batch_fallback` (which is about what to do when the
        combined files are ABSENT; this is about never consulting them at
        all, even when present) and never writes anything. This is what
        distinguishes the GUI's third explicit option ("Do not assemble -
        plot directly from batch files") from its second ("Use existing
        pre-assembled files", `allow_batch_fallback=False` below) and from
        the pre-existing hybrid (`allow_batch_fallback=True`, which
        PREFERS the combined files when present and only falls back to
        batches when they're missing). Raises the same `FileNotFoundError`
        as the missing-combined-file fallback when no complete batch
        exists to read from.

    legacy_return (default False): see `assemble()`'s own docstring - same
        meaning here.

    The hard `required` list is narrowed to `['Metric.csv']` (previously
    also `Prediction_result_train.csv`, `Prediction_result_test.csv`,
    `Marker_effect.csv`, `Interaction.csv`). The other four are resolved
    per-file by `ResultSet` itself, exactly the way `Attention.csv` was
    already handled pre-Update-3 (present -> read; absent -> empty frame) -
    this is a WIDENING of what's accepted, never a narrowing, and it fixes
    a second, independent defect as a side effect: `Interaction.csv` being
    hard-required meant a run with no RF model (which legitimately never
    writes that file) could never use this checkbox at all, even after a
    fully successful assemble.
    """
    result_dir = './Result/'+RESULT_NAME+'/'
    metric_path = result_dir + 'Metric.csv'

    if force_batch_source:
        # Unconditionally the per-batch path - reuses
        # _load_assembled_batch_fallback() (already implements exactly
        # this: discover complete batches, build ResultSet(source=
        # 'batches'), write nothing, raise if none are complete) rather
        # than the "only when Metric.csv is absent" gate above it.
        return _load_assembled_batch_fallback(RESULT_NAME, result_dir, metric_path,
                                               True, legacy_return)

    if not os.path.isfile(metric_path):
        return _load_assembled_batch_fallback(RESULT_NAME, result_dir, metric_path,
                                               allow_batch_fallback, legacy_return)

    try:
        # A throwaway read purely to detect the SAME pd.errors.EmptyDataError
        # a truly-empty-or-columnless Metric.csv raises (matches
        # pre-Update-3 behaviour exactly - a header-only file with real
        # column names does NOT raise this, and is read again, properly,
        # by ResultSet below). Metric.csv is small ("already small" -
        # blueprint §R3.1), so this second read costs nothing worth
        # avoiding.
        pd.read_csv(metric_path)
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
            print(f"[load_assembled] '{metric_path}' is empty, but every batch found "
                  f"completed successfully - every task across every one of them was skipped "
                  f"(most likely no matching phenotype/genotype data). Loading empty results; "
                  f"this reflects your data/configuration, not a failed assemble() run.")
            result = ResultSet(RESULT_NAME, source='combined',
                                missing_batches=[], incomplete_batches=_ckpt.find_incomplete_batches(RESULT_NAME))
            return tuple(result) if legacy_return else result
        else:
            raise FileNotFoundError(
                f"Cannot skip assemble: '{metric_path}' exists but is empty (the previous "
                f"assemble() call likely found no usable batch output). Run Step 2 with 'Skip "
                f"assemble' unchecked to re-assemble from the individual batches instead."
            )

    # No batch rescan happens here (that's the point of skipping assemble),
    # so there's nothing new to report about MISSING batches specifically -
    # but checking for batches that are still INCOMPLETE (a leftover
    # checkpoint file - see checkpoint_utils.py) is cheap (just a glob +
    # tiny JSON read, not re-reading every batch's full Metric_<idx>.csv)
    # and worth doing even here: those batches' partial data was already
    # excluded from whatever assemble() call produced the combined files
    # being loaded now, but the user should still be told they exist and
    # haven't been finished/re-assembled yet.
    incomplete_batches = _ckpt.find_incomplete_batches(RESULT_NAME)
    if incomplete_batches:
        print(f"[assemble] NOTE: {len(incomplete_batches)} batch(es) still have an unfinished "
              f"checkpoint (excluded from the combined results being loaded here):")
        for _b in incomplete_batches:
            print(f"[assemble]   - {_ckpt.describe_incomplete_batch(_b)}")

    # The other four ("previously required") files resolve per-file inside
    # ResultSet._read_key() itself, exactly like Attention.csv already did
    # pre-Update-3 (present -> read; absent -> empty frame) - nothing else
    # needs to be done for them here.
    result = ResultSet(RESULT_NAME, source='combined',
                        missing_batches=[], incomplete_batches=incomplete_batches)
    return tuple(result) if legacy_return else result


def _load_assembled_batch_fallback(RESULT_NAME, result_dir, metric_path, allow_batch_fallback, legacy_return):
    """The combined `Metric.csv` doesn't exist (or, when called with
    `allow_batch_fallback=True` from `load_assembled(force_batch_source=
    True)`, is simply never consulted at all) - Update ID 3, R2's decision
    table (blueprint §R2.2), extended by patch 3, Requirement 1's explicit
    'do not assemble' mode:

        allow_batch_fallback AND >=1 complete batch  -> ResultSet(source='batches'); write nothing
        allow_batch_fallback, no complete batch       -> FileNotFoundError (extended message)
        not allow_batch_fallback                      -> today's FileNotFoundError, verbatim
    """
    if not allow_batch_fallback:
        raise FileNotFoundError(
            "Cannot use pre-assembled files: the following combined file(s) are missing from "
            f"'{result_dir}': Metric.csv. Run Step 2 in 'Assemble into combined files' mode at "
            "least once first."
        )

    _status = check_batch_status(RESULT_NAME)
    complete = _status['complete']
    incomplete_batches = _status['incomplete']
    if not complete:
        raise FileNotFoundError(
            "Cannot gather results: the combined file(s) are missing from "
            f"'{result_dir}' (Metric.csv), AND no complete batch was found either to plot "
            "directly from its per-batch files. Check that Step 1 has completed at least one "
            "batch for this result name, or run Step 2 in 'Assemble into combined files' mode "
            "first."
        )

    # Update ID ver4-9, R7: presence is checked via
    # checkpoint_utils.resolve_result_path() (compressed-then-uncompressed
    # probe), not a hardcoded uncompressed filename - otherwise this
    # informational line would wrongly claim every one of these was
    # "missing" whenever RESULT_COMPRESSION='gzip' actually produced them
    # under their '.gz' names.
    _combined_keys = (('Metric.csv', 'record'), ('Prediction_result_train.csv', 'result_train'),
                       ('Prediction_result_test.csv', 'result_test'), ('Marker_effect.csv', 'effect'),
                       ('Interaction.csv', 'interactions'))
    _missing_combined = [label for label, key in _combined_keys
                          if _ckpt.resolve_result_path(RESULT_NAME, 0, False, key) is None]
    print(f"[load_assembled] Plotting directly from {len(complete)} complete per-batch file(s) "
          f"instead of the combined files, and writing NO combined file. Combined file(s) not "
          f"found (or intentionally not consulted): {', '.join(_missing_combined) or 'n/a'}. "
          f"Batch ID(s) read instead: {_ckpt.format_batch_id_list(complete)}.")
    if incomplete_batches:
        print(f"[assemble] NOTE: {len(incomplete_batches)} batch(es) still have an unfinished "
              f"checkpoint (excluded from the per-batch results being read here):")
        for _b in incomplete_batches:
            print(f"[assemble]   - {_ckpt.describe_incomplete_batch(_b)}")

    result = ResultSet(RESULT_NAME, source='batches', batch_ids=complete,
                        missing_batches=_status['missing'], incomplete_batches=incomplete_batches)
    return tuple(result) if legacy_return else result


# ---------------------------------------------------------------------------
# Patch 3, Requirement 1 - "restore the option to select pre-merged output
# files... users should be able to choose one of the three available
# options (assemble to make single output files, use preassembled files,
# and do not assemble files)".
# ---------------------------------------------------------------------------

def load_for_step2(RESULT_NAME, cfg):
    """Resolve and load Step 2's results, per patch 3 Requirement 1's three
    explicit gather-results options - the single place `run_step2_
    assemble.py` and `main_app.py`'s own in-process Step 2 block both
    call, so the two execution paths can never drift out of sync (this
    codebase's established single-source-of-truth pattern - e.g.
    `checkpoint_utils.RESULT_FILE_NAMES`).

    `cfg['ASSEMBLE_MODE']` (new key), when present, selects EXACTLY one of
    three explicit behaviours:

        'assemble'          -> assemble() - merge every batch into the
            combined files (the normal, default behaviour).
        'use_preassembled'  -> load_assembled(allow_batch_fallback=False)
            - ONLY read the combined files a previous assemble() already
            wrote; raise if they don't exist. Never reads per-batch files,
            never writes anything.
        'no_assemble'       -> load_assembled(force_batch_source=True) -
            ALWAYS read directly from the per-batch files, even if
            combined files already exist; never writes anything. This is
            the genuinely new capability this requirement restores -
            distinct from 'use_preassembled' (which REQUIRES the combined
            files) and from the legacy hybrid below (which PREFERS the
            combined files when present).

    `cfg['ASSEMBLE_MODE']` ABSENT (any config written before this option
    existed) falls back EXACTLY to the pre-existing
    `cfg['SKIP_ASSEMBLE']`/`cfg['SKIP_ASSEMBLE_BATCH_FALLBACK']` pair,
    completely unchanged - old configs keep behaving exactly as they did
    before this requirement was implemented (matches this codebase's
    established backward-compatibility discipline, e.g. Update ID 3's own
    §9 "Old `*_config.json` files still load and run"):

        SKIP_ASSEMBLE False (or absent) -> assemble()
        SKIP_ASSEMBLE True              -> load_assembled(
            allow_batch_fallback=cfg.get('SKIP_ASSEMBLE_BATCH_FALLBACK', True))
            - prefers the combined files when present, transparently
            falls back to per-batch files when they're not.

    Returns (result, message) - `message` is a ready-to-print/log summary
    string. This function performs no I/O beyond the assemble/load call
    itself (no print()/st.* calls of its own), so it is equally usable
    from a plain print() context (run_step2_assemble.py) and a Streamlit
    progress-log context (main_app.py's own `_log()`).
    """
    mode = cfg.get('ASSEMBLE_MODE')
    _t0 = time.time()
    if mode == 'no_assemble':
        result = load_assembled(RESULT_NAME, force_batch_source=True)
        message = (f"'Do not assemble' selected - plotting directly from per-batch files for "
                   f"models {result.models} (took {time.time() - _t0:.1f}s). No combined file "
                   f"written.")
    elif mode == 'use_preassembled':
        result = load_assembled(RESULT_NAME, allow_batch_fallback=False)
        message = (f"'Use pre-assembled files' selected - reloaded previously assembled "
                   f"results for models {result.models} (took {time.time() - _t0:.1f}s).")
    elif mode == 'assemble':
        result = assemble(RESULT_NAME, expected_batches=cfg.get('EXPECTED_BATCHES'),
                           streaming=cfg.get('ASSEMBLE_STREAMING', True),
                           RESULT_COMPRESSION=cfg.get('RESULT_COMPRESSION', 'gzip'))
        message = (f"Assembled results from all batches for models {result.models} "
                   f"(took {time.time() - _t0:.1f}s).")
    elif cfg.get('SKIP_ASSEMBLE'):
        # Legacy hybrid (ASSEMBLE_MODE absent, SKIP_ASSEMBLE=True) -
        # byte-for-byte the pre-patch-3 "Skip assemble" checkbox
        # behaviour: prefer the combined files, fall back to per-batch
        # files only when they're missing.
        result = load_assembled(RESULT_NAME, allow_batch_fallback=cfg.get('SKIP_ASSEMBLE_BATCH_FALLBACK', True))
        message = (f"Skipped assemble - reloaded previously assembled results for models "
                   f"{result.models} (took {time.time() - _t0:.1f}s).")
    else:
        # Legacy default (ASSEMBLE_MODE absent, SKIP_ASSEMBLE False/absent).
        result = assemble(RESULT_NAME, expected_batches=cfg.get('EXPECTED_BATCHES'),
                           streaming=cfg.get('ASSEMBLE_STREAMING', True),
                           RESULT_COMPRESSION=cfg.get('RESULT_COMPRESSION', 'gzip'))
        message = (f"Assembled results from all batches for models {result.models} "
                   f"(took {time.time() - _t0:.1f}s).")
    return result, message
