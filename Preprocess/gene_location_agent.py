"""
Preprocess/gene_location_agent.py
===================================

Requirement 1 needs a curator-prepared `location_lookup` table (gene symbol
-> chromosome/start/end) before `Preprocess.gene_network_prior.build_gene_list`
can run - manual curation is correct but slow. This module automates that
curation step with an LLM agent instead of a human, while keeping the same
"never guess, always report and drop" discipline as the rest of this feature:
an agent that cannot find a trustworthy, species-correct, single-locus
genomic coordinate for a gene must say so and leave it out, not fabricate one.

Output schema (canonical - matches a real, hand-verified example table)
--------------------------------------------------------------------------
    Gene_Name, Start_bp (or Start_cM), End_bp (or End_cM), Chromosome,
    AGI_Locus_ID, Source

`AGI_Locus_ID`/`Source` are for provenance and cross-checking, not strictly
required for `build_gene_list`, but every agent backend here is instructed to
fill them in (they are exactly what let a human reviewer sanity-check the
agent's work in seconds, e.g. by looking up the AGI code on
https://www.arabidopsis.org/ directly).

Two backends
------------
- 'claude_code' (default): drives Claude Code headlessly (the same
  `-p`/`--print` mechanism used in Preprocess.flash_p_integration) with a
  detailed research prompt and WebSearch/WebFetch access, asking it to write
  the CSV directly to disk.
- 'crewai' (optional): defines a two-agent CrewAI crew - a Curator that
  researches each gene, and a QC Reviewer that independently cross-checks
  every proposed row and can reject entries the Curator got wrong - for
  cases where that extra adversarial review step is worth the added cost.
  Requires `pip install crewai crewai-tools`; raises a clear ImportError
  (not a silent fallback) if unavailable, since backend='crewai' was an
  explicit choice.

Both backends are validated identically after they run: the resulting CSV is
loaded through the exact same `Preprocess.gene_network_prior._load_location_lookup`
column-resolution logic that a human-authored table would go through, so a
malformed or incomplete agent output fails loudly here rather than silently
downstream in `build_gene_list`.
"""

import json
import os
import re
import warnings

import pandas as pd

from Preprocess.gene_network_prior import _load_location_lookup, build_gene_list
from Preprocess.flash_p_integration import (
    _run_claude_headless, _summarize_permission_denials, _summarize_auto_mode_unavailable,
    _summarize_rate_limit, _claude_binary_not_found_message,
)


CANONICAL_COLUMNS = {
    'bp': ['Gene_Name', 'Start_bp', 'End_bp', 'Chromosome', 'AGI_Locus_ID', 'Source'],
    'cM': ['Gene_Name', 'Start_cM', 'End_cM', 'Chromosome', 'AGI_Locus_ID', 'Source'],
}


# ---------------------------------------------------------------------------
# Shared prompt / task text
# ---------------------------------------------------------------------------

def _curation_instructions(candidate_gene_ids, species, unit, output_csv, allow_web_fetch=False):
    columns = CANONICAL_COLUMNS[unit]
    gene_bullet_list = '\n'.join(f'- {g}' for g in candidate_gene_ids)
    # Requirement (token efficiency for gene-location curation): this
    # "TOKEN HYGIENE" block mirrors the exact finding FLASH-P's own "Light"
    # mode is built on (see Preprocess.flash_p_integration's
    # LIGHT_MODE_PROMPT_SUFFIX comment for the full citation trail against
    # github.com/CMits/Flash-P) - WebFetching a whole gene/genome-database
    # page costs vastly more tokens than a WebSearch snippet, and a snippet
    # is normally enough to confirm a chromosome/start/end coordinate. This
    # was previously the single biggest driver of both "a prohibitive
    # amount of tokens" and unfinished runs: every gene's WebFetch page(s)
    # stayed in this same session's context for every gene looked up after
    # it. allow_web_fetch=True (used for the optional harder-genes retry
    # pass in curate_gene_locations_claude_code) re-enables it for genes a
    # WebSearch-only pass couldn't resolve.
    if allow_web_fetch:
        research_line = (
            "Use WebSearch to verify each one; if a WebSearch snippet alone isn't enough to "
            "confirm a coordinate, WebFetch the single most authoritative page - but only as a "
            "last resort per gene, never routinely."
        )
    else:
        research_line = (
            "Use WebSearch ONLY to verify each one (DOI/coordinate taken straight from the search "
            "result) - do NOT WebFetch full pages; a search snippet is normally enough to confirm "
            "a chromosome/start/end coordinate."
        )
    token_hygiene = f"""
TOKEN HYGIENE (read this before starting):
- Knowledge-first: for each gene, draft your best-guess coordinate from your own training
  knowledge FIRST, then verify or correct it with one targeted search - don't research from a
  blank slate every time.
- {research_line}
- Batch independent WebSearches for different genes together in the same turn where possible,
  rather than one search per turn.
- Do not paste full search results or full page contents into your reply - extract just the
  coordinate/AGI-ID/citation you need and move on.
- This is one batch of {len(candidate_gene_ids)} gene(s) out of a larger list processed in
  separate batches - focus ONLY on the genes listed below; do not try to look up or comment on
  any other gene.
"""
    return f"""You are curating genomic locations for a gene-regulatory-network model.

SPECIES: {species}
CANDIDATE GENE SYMBOLS (from a network-extraction tool's "GENE"-typed nodes -
treat this list as unverified; some entries may not actually be real,
single-locus genes in this species):
{gene_bullet_list}
{token_hygiene}
TASK: for every gene symbol above, find its authoritative genomic location in
{species} (chromosome, start, end in {unit}) using the official reference
genome annotation for this species - e.g. TAIR/Araport11 via
https://www.arabidopsis.org/ for Arabidopsis thaliana, MaizeGDB for Zea mays,
RAP-DB/RGI for Oryza sativa, NCBI Gene/RefSeq as a cross-check for any
species. {research_line} Do not rely on your own memory alone without
verification, and do not report a coordinate you are not confident is
correct for THIS species and THIS reference genome build.

For every gene symbol that you CANNOT confidently resolve to a real,
single-locus gene in {species} with a verifiable coordinate, DO NOT invent or
guess a value - simply omit that gene from the output entirely. In
particular, watch for and exclude:
  - symbols that are not actually single-locus genes (a named pathway, a
    hormone/metabolite class, a protein complex, etc., mistakenly typed as a
    gene by the upstream extraction tool)
  - symbols that belong to a different species than {species} (e.g. an
    "Os"-prefixed rice gene symbol appearing in an Arabidopsis network)
  - symbols that are ambiguous (multiple, materially different genes share
    the symbol) unless you can determine which one this network actually
    means from context
A gene with a genuinely duplicated/synonym relationship to another gene in
the list (e.g. two historical names for the same locus) should still be
reported - with its own real, verified coordinate, which may be identical to
its synonym's - rather than dropped, since both symbols may appear as
distinct nodes in the source network.

OUTPUT: write a CSV file to exactly this path: {output_csv}
with EXACTLY these columns, in this order: {', '.join(columns)}
- One row per gene per locus (a gene with more than one genomic location
  gets more than one row, each sharing the same Gene_Name).
- Chromosome as a bare number/name matching the reference genome's own
  convention (no 'chr' prefix unless that IS the reference's own convention).
- AGI_Locus_ID: the systematic locus identifier if the species has one (e.g.
  Arabidopsis AGI codes like AT3G18550), else leave blank.
- Source: a short, specific citation of where the coordinate came from (e.g.
  "TAIR10 genome annotation", "NCBI RefSeq (GCF_...)", including a note if
  this is a synonym of another gene in the list).
- Do not include any gene you could not confidently resolve - omit its row
  entirely rather than guessing.
- Write ONLY the CSV to that file path. Do not add commentary inside the CSV
  file itself.
"""


def _validate_curated_csv(output_csv, candidate_gene_ids, unit):
    """Load the agent-produced CSV through the same column-resolution logic
    the rest of the pipeline uses, and report which candidate genes were
    (and were not) resolved - never trusting the agent's own claims about
    completeness without checking the file it actually wrote.
    """
    if not os.path.isfile(output_csv):
        raise RuntimeError(f"Expected the agent to write a CSV to '{output_csv}', but no file "
                            f"was found there after the run completed.")

    table = _load_location_lookup(output_csv, unit=unit)
    resolved = sorted(set(table['name']) & set(candidate_gene_ids))
    unresolved = sorted(set(candidate_gene_ids) - set(table['name']))
    unexpected = sorted(set(table['name']) - set(candidate_gene_ids))

    if unexpected:
        warnings.warn(f"The curated CSV contains {len(unexpected)} gene name(s) that were not "
                       f"in the candidate list and will be ignored by build_gene_list(): {unexpected}")

    print(f"[gene_location_agent] Curated {len(resolved)}/{len(candidate_gene_ids)} candidate gene(s). "
          f"Left unresolved (dropped, not guessed): {unresolved}")

    return table, resolved, unresolved


# ---------------------------------------------------------------------------
# Backend 1: Claude Code (headless)
# ---------------------------------------------------------------------------

DEFAULT_GENE_LOCATION_BATCH_SIZE = 20


def _gene_id_batches(candidate_gene_ids, batch_size):
    for i in range(0, len(candidate_gene_ids), batch_size):
        yield candidate_gene_ids[i:i + batch_size]


def curate_gene_locations_claude_code(candidate_gene_ids, species, output_csv, unit='bp',
                                        cwd=None, claude_binary='claude', model=None,
                                        permission_mode='dontAsk',
                                        # Glob/Grep added alongside Read/Write/WebSearch -
                                        # see the DEFAULT_ALLOWED_TOOLS comment in
                                        # flash_p_integration.py for why: they're "no permission
                                        # required" tools an interactive session gets for free, but
                                        # still need to be on this explicit list under 'dontAsk'.
                                        # Bash was added after a live run's permission_denials showed
                                        # 7 denied Bash calls and the agent never wrote output_csv -
                                        # unlike FLASH-P, this task wasn't originally expected to need
                                        # a shell, but evidently does (e.g. for writing/formatting the
                                        # CSV), so the same "dontAsk denies silently instead of
                                        # prompting" failure mode applied here too. WebFetch is
                                        # deliberately NOT in this default list any more - see
                                        # allow_web_fetch below.
                                        allowed_tools=('Read', 'Write', 'WebSearch', 'Glob', 'Grep', 'Bash'),
                                        timeout_seconds=600, extra_env=None,
                                        batch_size=DEFAULT_GENE_LOCATION_BATCH_SIZE,
                                        allow_web_fetch=False, log_callback=None):
    """Curate gene locations by driving Claude Code headlessly with a
    detailed research prompt (see `_curation_instructions`). Mirrors
    `Preprocess.flash_p_integration._run_claude_headless`'s invocation
    pattern (Claude Code's documented `-p`/`--print` mode:
    https://code.claude.com/docs/en/headless), but as a standalone research
    task rather than the full FLASH-P pipeline, so it needs no FLASH-P
    checkout - any working directory the agent can write `output_csv` from
    is enough.

    Requirement (token efficiency / avoid unfinished runs): candidate genes
    are curated in independent batches of `batch_size` genes each (default
    20), run as SEPARATE Claude Code sessions rather than one single session
    covering every candidate gene. This was previously the single biggest
    cause of both "a prohibitive amount of tokens" and "the task is often
    unfinished": a network with, say, 120 candidate genes put every one of
    them, plus every WebSearch/WebFetch result gathered along the way, into
    ONE continuously-growing session context - each new gene's lookup paid
    to re-send the entire transcript of every gene looked up before it
    (session context grows every turn, so total cost scales worse than
    linearly with gene count), and a single dropped connection, rate limit,
    or the fixed overall timeout cut off every gene after that point in the
    list, not just the one being processed. Batching bounds each session's
    context to ~batch_size genes' worth of work, and - just as importantly -
    each batch is now wrapped in its own try/except below: one batch
    failing (timeout, rate limit, a bad run) no longer discards every OTHER
    batch's already-completed, real research - those rows are still merged
    into `output_csv` and reported as resolved, and only the failed batch's
    genes end up in `unresolved_gene_ids`, exactly as if the agent had
    itself failed to resolve them.

    Also see `_curation_instructions`'s "TOKEN HYGIENE" section (same
    WebFetch-avoidance finding FLASH-P's own "Light" mode is built on -
    Preprocess.flash_p_integration.LIGHT_MODE_PROMPT_SUFFIX's comment has
    the full citation trail) and `allow_web_fetch` below.

    Parameters (new/changed from before this feature)
    ----------
    batch_size : int
        How many candidate genes go into each independent Claude Code
        session. Smaller = lower per-session token cost and finer-grained
        failure isolation, at the cost of more separate `claude` process
        launches (each with its own small fixed startup overhead). 20 is a
        reasonable default for a typical gene-network size; lower it
        further for very large networks or a very token-constrained plan.
    allow_web_fetch : bool
        False (default): WebFetch is not offered to the agent at all - see
        `_curation_instructions`. True: WebFetch is added back to
        `allowed_tools` and permitted as a last resort per gene - use this
        for a smaller, targeted retry batch of genes a WebSearch-only pass
        left unresolved, not as the default for a full run.
    timeout_seconds : int
        Per-BATCH timeout (not for the whole candidate list any more, since
        each batch is now its own session) - defaults to 600s (10 min) for
        a batch of up to `batch_size` genes, versus the old 1800s that had
        to cover the entire candidate list in one session.
    log_callback : callable(str), optional
        Forwarded to every batch's `_run_claude_headless` call - see that
        function's docstring (Preprocess/flash_p_integration.py) for the
        live-log-in-the-GUI mechanism this reuses unchanged.

    permission_mode defaults to 'dontAsk' rather than 'bypassPermissions'
    or 'auto':
    'dontAsk' auto-approves everything in `allowed_tools` (already covering
    what this function needs) and silently denies anything outside it,
    without ever showing a permission dialog. 'bypassPermissions'
    (--dangerously-skip-permissions) skips checks entirely but Claude Code
    refuses to start in that mode at all when the process is root/sudo
    ("cannot be used with root/sudo privileges for security reasons") - a
    common situation in a containerized deployment - and even as a
    non-root user it still needs a one-time interactive "I accept" dialog
    before it can run headlessly. 'auto' routes each tool call through
    Claude Code's own classifier instead of `allowed_tools` (see the full
    comparison in flash_p_integration.py's _run_claude_headless) - not
    every account/model supports it, see `_summarize_auto_mode_unavailable`.
    Pass permission_mode='bypassPermissions' explicitly if you specifically
    want that behavior on a non-root machine, or 'acceptEdits'/'default' to
    require manual approval.

    Returns (gene_list_table, resolved_gene_ids, unresolved_gene_ids) - see
    `_validate_curated_csv`. `unresolved_gene_ids` includes both genes an
    agent explicitly couldn't confirm AND genes whose whole batch failed to
    run at all (a warning is printed distinguishing the two).
    """
    output_csv = os.path.abspath(output_csv)
    cwd = cwd or os.path.dirname(output_csv) or '.'
    os.makedirs(cwd, exist_ok=True)

    effective_allowed_tools = tuple(allowed_tools) + (() if not allow_web_fetch or 'WebFetch' in allowed_tools
                                                        else ('WebFetch',))

    batches = list(_gene_id_batches(candidate_gene_ids, max(1, int(batch_size))))
    n_batches = len(batches)
    batch_csv_paths = []
    batch_errors = {}  # batch index -> (gene_ids, error message)

    for batch_idx, batch_gene_ids in enumerate(batches, start=1):
        batch_csv = f"{output_csv}.batch{batch_idx}.csv"
        prompt = _curation_instructions(batch_gene_ids, species, unit, batch_csv,
                                         allow_web_fetch=allow_web_fetch)

        print(f"[gene_location_agent] Batch {batch_idx}/{n_batches}: asking Claude Code to curate "
              f"{len(batch_gene_ids)} gene location(s) for species={species!r} -> '{batch_csv}' (cwd={cwd})")
        try:
            result = _run_claude_headless(
                prompt, cwd=cwd, claude_binary=claude_binary, permission_mode=permission_mode,
                allowed_tools=effective_allowed_tools, model=model, timeout_seconds=timeout_seconds,
                extra_env=extra_env, log_callback=log_callback,
            )
        except FileNotFoundError as e:
            # A missing claude binary affects every remaining batch identically -
            # no point continuing to try (and burning the per-batch timeout on
            # each attempt) once this is confirmed on the first batch.
            raise RuntimeError(_claude_binary_not_found_message(claude_binary, e)) from e

        if result.returncode != 0:
            rate_note = _summarize_rate_limit(result.stdout)
            notes = [n for n in (rate_note, _summarize_permission_denials(result.stdout),
                                  _summarize_auto_mode_unavailable(result.stdout)) if n]
            msg = (f"claude exited with code {result.returncode}."
                   + ("\n" + "\n".join(notes) if notes else "") +
                   f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")
            warnings.warn(f"[gene_location_agent] Batch {batch_idx}/{n_batches} failed - its "
                           f"{len(batch_gene_ids)} gene(s) will be reported as unresolved, but "
                           f"every OTHER batch's results are unaffected: {msg}")
            batch_errors[batch_idx] = (batch_gene_ids, msg)

            if rate_note:
                # A rate/session limit is an ACCOUNT-WIDE ceiling, not a
                # per-batch one - unlike every other failure mode here,
                # trying the remaining batches wouldn't isolate an
                # unrelated problem, it would just repeat the exact same
                # doomed request against the exact same exhausted quota,
                # burning real tokens (each attempt still pays for its own
                # startup/context) for a guaranteed identical failure. Stop
                # here instead, and report every batch that never got a
                # chance to run as unresolved too - but distinctly, since
                # "never attempted" is a different, more actionable state
                # than "attempted and failed" (nothing to inspect, and
                # simply re-running later is enough once the limit clears).
                remaining = batches[batch_idx:]  # batches after this one (0-indexed slice, batch_idx is 1-based)
                if remaining:
                    skipped_genes = [g for b in remaining for g in b]
                    warnings.warn(
                        f"[gene_location_agent] Stopping after batch {batch_idx}/{n_batches}: this was "
                        f"a rate/session limit (account-wide, not specific to this batch), so the "
                        f"remaining {len(remaining)} batch(es) ({len(skipped_genes)} gene(s): "
                        f"{skipped_genes}) would fail identically right now. They were never "
                        f"attempted - simply re-running curate_gene_locations() with just these genes "
                        f"once the limit resets is enough, no special recovery needed."
                    )
                    for skip_idx, skip_batch in enumerate(remaining, start=batch_idx + 1):
                        batch_errors[skip_idx] = (skip_batch, "Skipped - not attempted (see rate-limit warning above).")
                break

            continue

        if not os.path.isfile(batch_csv):
            notes = [n for n in (_summarize_permission_denials(result.stdout),
                                  _summarize_auto_mode_unavailable(result.stdout)) if n]
            agent_result_text = None
            try:
                agent_result_text = json.loads(result.stdout).get('result')
            except (ValueError, TypeError, AttributeError):
                pass
            if agent_result_text and not notes:
                notes = [f"The agent's own final message was: {agent_result_text!r}"]
            msg = "claude exited successfully but did not write the expected batch CSV." \
                  + ("\n" + "\n".join(notes) if notes else "")
            warnings.warn(f"[gene_location_agent] Batch {batch_idx}/{n_batches} failed - its "
                           f"{len(batch_gene_ids)} gene(s) will be reported as unresolved, but "
                           f"every OTHER batch's results are unaffected: {msg}")
            batch_errors[batch_idx] = (batch_gene_ids, msg)
            continue

        batch_csv_paths.append(batch_csv)

    if not batch_csv_paths:
        all_failed_genes = [g for gene_ids, _ in batch_errors.values() for g in gene_ids]
        raise RuntimeError(
            f"Every batch failed ({n_batches}/{n_batches}) - no gene locations were curated for "
            f"{len(all_failed_genes)} candidate gene(s). See the warnings above for each batch's "
            f"individual error."
        )

    # Merge every successful batch's CSV into the single final output_csv
    # the rest of the pipeline expects (same canonical columns in every
    # batch, since every batch shares the same _curation_instructions
    # template) - a plain concat is safe, no per-batch header handling
    # needed since each batch CSV already has its own header row.
    merged = pd.concat([pd.read_csv(p, encoding='utf-8-sig') for p in batch_csv_paths], ignore_index=True)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    merged.to_csv(output_csv, index=False)
    for p in batch_csv_paths:
        try:
            os.remove(p)
        except OSError:
            pass  # best-effort cleanup only - a leftover per-batch CSV never affects correctness

    table, resolved, unresolved = _validate_curated_csv(output_csv, candidate_gene_ids, unit)

    if batch_errors:
        skipped_msg = "Skipped - not attempted (see rate-limit warning above)."
        attempted_failed = {k: v for k, v in batch_errors.items() if v[1] != skipped_msg}
        skipped = {k: v for k, v in batch_errors.items() if v[1] == skipped_msg}

        if attempted_failed:
            failed_genes = sorted({g for gene_ids, _ in attempted_failed.values() for g in gene_ids})
            print(f"[gene_location_agent] {len(attempted_failed)}/{n_batches} batch(es) failed to run "
                  f"at all (as opposed to the agent explicitly declining a gene) - affected gene(s): "
                  f"{failed_genes}. Re-run curate_gene_locations() with just these genes to retry only "
                  f"the failed batches, rather than the full candidate list.")
        if skipped:
            skipped_genes = sorted({g for gene_ids, _ in skipped.values() for g in gene_ids})
            print(f"[gene_location_agent] {len(skipped)}/{n_batches} batch(es) were never attempted "
                  f"(stopped early after a rate/session limit - see the warning above for when it "
                  f"resets) - affected gene(s): {skipped_genes}. Re-run curate_gene_locations() with "
                  f"just these genes once the limit clears.")

    return table, resolved, unresolved


# ---------------------------------------------------------------------------
# Backend 2: CrewAI (optional)
# ---------------------------------------------------------------------------

def _extract_csv_block(text):
    """Agents tend to wrap their final answer in prose and/or a fenced code
    block; pull out just the CSV. Falls back to the raw text if no fence is
    found (e.g. the agent returned bare CSV)."""
    fence = re.search(r"```(?:csv)?\s*\n(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


def curate_gene_locations_crewai(candidate_gene_ids, species, output_csv, unit='bp',
                                   llm=None, search_tool=None, verbose=False):
    """Curate gene locations with a two-agent CrewAI crew: a Curator that
    researches each gene, and an independent QC Reviewer that cross-checks
    and can reject the Curator's entries. Optional - only used if it
    improves curation quality enough to justify the extra agent call; the
    'claude_code' backend above is sufficient on its own for most cases.

    Parameters
    ----------
    llm : a crewai.LLM instance (or provider string CrewAI accepts), or None
        to use CrewAI's own default. Passed straight through to both agents.
    search_tool : a crewai_tools tool instance (e.g. crewai_tools.SerperDevTool(),
        which needs a SERPER_API_KEY), or None. Without a real search tool
        the agents can only draw on the underlying LLM's training data, with
        no way to verify a coordinate against a live database - a warning is
        raised in that case, since it weakens exactly the "never guess"
        guarantee this whole module exists to provide.

    Returns (gene_list_table, resolved_gene_ids, unresolved_gene_ids) - see
    `_validate_curated_csv`.
    """
    try:
        from crewai import Agent, Task, Crew, Process
    except ImportError as e:
        raise ImportError(
            "backend='crewai' requires the crewai package: `pip install crewai crewai-tools`. "
            "Use backend='claude_code' instead if you don't want this extra dependency."
        ) from e

    if search_tool is None:
        warnings.warn(
            "curate_gene_locations_crewai() was called without a search_tool - the crew can only "
            "draw on the LLM's training data, with no way to verify a gene's coordinates against a "
            "live database. Pass e.g. crewai_tools.SerperDevTool() (requires SERPER_API_KEY) for "
            "real literature/database lookups; results without one should be treated as a "
            "first draft, not a final curated table."
        )
    tools = [search_tool] if search_tool is not None else []
    columns = CANONICAL_COLUMNS[unit]

    curator = Agent(
        role='Genomic Location Curator',
        goal=(f'Find a correct, verifiable genomic location (chromosome, start, end in {unit}) '
              f'for each given gene symbol in {species}, using the official reference genome '
              f'annotation for that species.'),
        backstory=('An expert genome curator who never reports a coordinate they have not '
                    'verified against a real annotation source, and who explicitly flags any '
                    'gene symbol that turns out not to be a real single-locus gene in this '
                    'species rather than guessing at one.'),
        tools=tools,
        llm=llm,
        verbose=verbose,
    )
    reviewer = Agent(
        role='Genomic Curation QC Reviewer',
        goal=('Independently cross-check every proposed gene location for species/build '
              'correctness and internal consistency, and reject (never patch with a guess) any '
              'row that is unverifiable, ambiguous, not a single-locus gene, or from the wrong '
              'species.'),
        backstory=('A skeptical second reviewer whose job is to catch the Curator\'s mistakes, '
                    'not to rubber-stamp their work.'),
        tools=tools,
        llm=llm,
        verbose=verbose,
    )

    curate_task = Task(
        description=_curation_instructions(candidate_gene_ids, species, unit, output_csv)
                    + "\n\nReturn the CSV content directly in your final answer (in addition to "
                      "any file you write), so the next reviewer can see exactly what you found.",
        expected_output=f"A CSV with header exactly: {','.join(columns)}",
        agent=curator,
    )
    review_task = Task(
        description=(
            "Review every row the Curator proposed. For each row, independently verify the "
            "chromosome/start/end against the same class of authoritative source the Curator was "
            "asked to use. Drop (do not fix by guessing) any row you cannot verify, any row for a "
            "symbol that is not really a single-locus gene, and any row for the wrong species. "
            f"Return the FINAL, corrected CSV with header exactly: {','.join(columns)}"
        ),
        expected_output=f"A CSV with header exactly: {','.join(columns)}",
        agent=reviewer,
        context=[curate_task],
    )

    crew = Crew(agents=[curator, reviewer], tasks=[curate_task, review_task],
                process=Process.sequential, verbose=verbose)
    result = crew.kickoff()

    csv_text = _extract_csv_block(str(result))
    output_csv = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', encoding='utf-8') as f:
        f.write(csv_text)

    return _validate_curated_csv(output_csv, candidate_gene_ids, unit)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def curate_gene_locations(candidate_gene_ids, species, output_csv, backend='claude_code', **kwargs):
    """Single entry point: backend='claude_code' (default) or 'crewai'.
    `candidate_gene_ids` is typically
    `[g['id'] for g in Preprocess.gene_network_prior.extract_candidate_genes(network)]`.
    """
    if backend == 'claude_code':
        return curate_gene_locations_claude_code(candidate_gene_ids, species, output_csv, **kwargs)
    elif backend == 'crewai':
        return curate_gene_locations_crewai(candidate_gene_ids, species, output_csv, **kwargs)
    else:
        raise ValueError(f"backend must be 'claude_code' or 'crewai', got {backend!r}")


def curate_and_build_gene_list(candidate_genes, species, output_csv, unit='bp',
                                 backend='claude_code', **kwargs):
    """Convenience wrapper chaining curation straight into
    `gene_network_prior.build_gene_list`, for callers who don't need the
    intermediate CSV inspected separately.

    `candidate_genes` here is the list of node dicts (NOT just ids) as
    returned by `gene_network_prior.extract_candidate_genes`, matching
    `build_gene_list`'s own expected input.
    """
    candidate_gene_ids = [g['id'] for g in candidate_genes]
    curate_gene_locations(candidate_gene_ids, species, output_csv, backend=backend,
                           unit=unit, **kwargs)
    return build_gene_list(candidate_genes, output_csv, unit=unit)
