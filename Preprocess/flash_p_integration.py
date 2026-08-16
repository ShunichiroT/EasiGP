"""
Preprocess/flash_p_integration.py
===================================

Requirement 1 said the interaction-network JSON "can be uploaded to EasiGP
by users or can be made using FlashP (https://flash-p.com/)". This module
implements the second path: it drives FLASH-P (https://github.com/CMits/FlashP)
end to end for a given (species, phenotype) pair and hands back the path to
the resulting `network.json`, so the rest of the pipeline
(Preprocess.gene_network_prior -> models.GAT_biological_prior_knowledge)
never needs to know whether that file was uploaded by a user or generated.

What FLASH-P actually is (verified against github.com/CMits/FlashP and
flash-p.com, not assumed)
--------------------------------------------------------------------------
FLASH-P is a multi-agent pipeline - literature reviewer -> judge -> builder
-> judge -> perturbation -> validator -> refiner -> exporter - that mines
primary literature for a (phenotype, species) pair and writes out a
validated causal network. It ships as a set of plain-text agent prompts plus
an orchestrator file (`CLAUDE.md`) and is designed to be driven by a coding
agent that reads that file - Claude Code, Codex CLI, or any AGENTS.md-aware
tool. The officially documented way to run it is:

    npm install -g @anthropic-ai/claude-code
    git clone https://github.com/CMits/FlashP.git
    cd FlashP/Flash-P_Plant/Claude
    claude
    > /run-flashp Shoot Branching in Arabidopsis

That is an *interactive* Claude Code session. This module reproduces the
same run *headlessly* (Claude Code's documented `-p`/`--print` non-interactive
mode: https://code.claude.com/docs/en/headless), so it can be called
programmatically from EasiGP's GUI/pipeline instead of requiring someone to
sit at a terminal.

Per FLASH-P's own `CLAUDE.md`, every run writes into
`networks/<Phenotype_Slug>/` (relative to the Claude Code working directory,
i.e. `<flashp_dir>/Flash-P_Plant/Claude/`), with the network itself at
`networks/<Phenotype_Slug>/network/network.json`. This module then relocates
that self-contained run folder into
`Outcome/Local_FlashP_Outcome/<run_name>/`, exactly mirroring the layout of
FLASH-P's own shipped local-run example
(`Outcome/Local_FlashP_Outcome/shoot_branching_local_network/` in the FLASH-P
repo itself - the same folder that produced the `network.json` used
throughout this EasiGP feature as the worked example), so the final path is:

    Outcome/Local_FlashP_Outcome/<run_name>/network/network.json

which is exactly the location this feature's GUI option should point users
at.

Typical usage
-------------
    from Preprocess.flash_p_integration import run_flash_p

    network_json_path = run_flash_p(
        species='Arabidopsis thaliana',
        phenotype='Shoot Branching',
        flashp_dir='/path/to/FlashP',   # a local clone; cloned automatically if missing
    )
"""

import json
import os
import re
import shutil
import subprocess
import threading
import warnings

DEFAULT_FLASHP_REPO_URL = 'https://github.com/CMits/FlashP.git'

# Tools the FLASH-P pipeline actually needs, per Flash-P_Plant/Claude/CLAUDE.md
# and the individual Agent/*_AGENT.md files: WebSearch for the literature
# review step, Read/Write/Edit/Bash for the file-driven handoffs between
# steps and for running the QA/export scripts, Task because the orchestrator
# dispatches each pipeline stage (builder, judge, validator, ...) as a
# subagent, and Monitor (Claude Code v2.1.98+) because the orchestrator uses
# it to wait on background evidence-verification jobs before dispatching the
# next subagent - confirmed by a live run whose JSON result included
# "permission_denials":[{"tool_name":"Monitor", ...}] and then quietly ended
# its turn ("I'll pause here") instead of continuing, without network.json
# ever being written, since 'dontAsk' denies (rather than prompts for)
# anything outside this list.
#
# Glob and Grep were added after investigating higher-than-expected Claude
# usage running FLASH-P through this integration vs. standalone: Claude
# Code's own tools reference (code.claude.com/docs/en/tools-reference) marks
# Read/Grep/Glob/LSP as "Permission required: No" for in-directory paths -
# i.e. a standalone interactive `claude` session gets Grep/Glob for free,
# with no prompt and no extra cost, but our explicit --allowedTools list
# still gates them under 'dontAsk' regardless of that "no permission
# needed" status, and they weren't on the list. FLASH-P's multi-agent
# pipeline (juggling files across networks/<slug>/, evidence, and agent
# handoffs) very plausibly reaches for them constantly; every silent denial
# either forces a costlier Bash find/grep workaround or risks the same
# stall the Monitor gap caused.
#
# 'Agent' is included alongside the (evidently working, per multi-model
# usage seen in real runs) 'Task' purely defensively: Claude Code's current
# tools reference names the subagent-dispatch tool 'Agent', not 'Task' -
# possibly a version-dependent rename. Keeping both costs nothing if one is
# unused, and guards against silently losing subagent delegation (which
# would be a much larger cost hit than Glob/Grep, since undelegated work
# bloats the main conversation's context instead of running in a subagent's
# own separate context window).
#
# If a future FLASH-P/Claude Code release needs another tool, the same
# failure pattern will recur for that tool - see the permission_denials
# check in run_flash_p() below, which surfaces exactly this without having
# to eyeball raw JSON again.
DEFAULT_ALLOWED_TOOLS = ('Read', 'Write', 'Edit', 'Bash', 'WebSearch', 'WebFetch',
                          'Task', 'Agent', 'Monitor', 'Glob', 'Grep')


def _claude_binary_not_found_message(claude_binary, exc=None):
    """Shared, detailed message for FileNotFoundError when spawning
    `claude_binary` - used by both preflight_check() and run_flash_p()/
    curate_gene_locations_claude_code() so a missing binary always produces
    the same actionable diagnosis instead of a raw traceback in some call
    paths and a clean message in others.

    [Errno 2] No such file or directory on the exact string 'claude_binary'
    means the OS's PATH lookup failed for that process - this is NOT a
    permissions or auth problem. The most common cause in a container is
    that `claude_binary='claude'` (the default) relies on PATH, and the
    user actually running this process can't see wherever `npm install -g
    @anthropic-ai/claude-code` put it - e.g. if that install ran as root at
    build time but the container now runs as a non-root user at runtime
    (see the Dockerfile's step 7 comment on this exact tradeoff).
    """
    return (
        f"Could not start '{claude_binary}' - [Errno 2] No such file or directory"
        + (f" ({exc})" if exc else "") + ". This means the OS couldn't find that "
        f"executable via PATH for whatever user is actually running this process - "
        f"not a permissions or authentication problem. To check: exec into the "
        f"container as the SAME user this app runs as and run `which {claude_binary}` "
        f"and `echo $PATH`. A common cause is `npm install -g @anthropic-ai/claude-code` "
        f"having been run as root at image build time while the container now runs as "
        f"a non-root user at runtime, whose PATH doesn't include wherever that put the "
        f"binary. Either fix that user's PATH, or set the 'Claude binary' setting to "
        f"the exact absolute path (e.g. the output of `which claude` as root: often "
        f"'/usr/local/bin/claude' or '/usr/lib/node_modules/.bin/claude') instead of "
        f"relying on PATH."
    )


def preflight_check(cwd, claude_binary='claude', permission_mode='dontAsk',
                     allowed_tools=DEFAULT_ALLOWED_TOOLS, extra_env=None, timeout_seconds=60):
    """A near-free sanity check (a single trivial prompt, no file/network
    access needed) that Claude Code can actually authenticate and start
    from `cwd` with the given permission_mode/extra_env, BEFORE spending a
    full FLASH-P run (typically tens of minutes and several dollars) to
    discover the same problem. Every config error found so far during this
    integration's setup - the wrong permission_mode for root/sudo, an
    invalid/missing API key, a stale ANTHROPIC_API_KEY silently outranking
    a subscription token, claude not being on PATH for this process's user -
    would have been caught here for a fraction of a cent instead of mid-run.
    Run this once after changing auth/permission_mode/claude_binary, not
    before every single FLASH-P run.

    Returns (ok: bool, message: str). Never raises - a preflight check that
    itself throws defeats the point of a safe, cheap sanity check.
    """
    try:
        result = _run_claude_headless(
            'Reply with exactly one word: OK', cwd=cwd, claude_binary=claude_binary,
            permission_mode=permission_mode, allowed_tools=allowed_tools,
            timeout_seconds=timeout_seconds, extra_env=extra_env,
        )
    except FileNotFoundError as e:
        return False, _claude_binary_not_found_message(claude_binary, e)
    except subprocess.TimeoutExpired:
        return False, f"No response within {timeout_seconds}s - check network access from this container."
    except Exception as e:  # pragma: no cover - defensive, see docstring
        return False, f"Unexpected error launching claude: {e}"

    if result.returncode == 0:
        return True, "Claude Code responded successfully - authentication and permission_mode are working."

    denial_note = _summarize_permission_denials(result.stdout)
    rate_note = _summarize_rate_limit(result.stdout)
    auto_note = _summarize_auto_mode_unavailable(result.stdout)
    detail = "\n".join(n for n in (rate_note, denial_note, auto_note) if n) or result.stdout or result.stderr
    return False, f"claude exited with code {result.returncode}.\n{detail}"


def _phenotype_slug(phenotype):
    """CLAUDE.md: "<Phenotype_Slug> = the phenotype in Title_Case_With_Underscores"
    (e.g. 'Shoot Branching' -> 'Shoot_Branching', matching the shipped
    networks/Shoot_Branching/ example exactly)."""
    words = re.split(r'\s+', phenotype.strip())
    return '_'.join(w[:1].upper() + w[1:] for w in words if w)


def ensure_flash_p_repo(flashp_dir, repo_url=DEFAULT_FLASHP_REPO_URL, git_binary='git'):
    """Clone FLASH-P into `flashp_dir` if it isn't there yet (the README's own
    setup step: `git clone https://github.com/CMits/FlashP.git`). No-ops if
    `flashp_dir` already looks like a FLASH-P checkout."""
    marker = os.path.join(flashp_dir, 'Flash-P_Plant', 'Claude', 'CLAUDE.md')
    if os.path.isfile(marker):
        return flashp_dir

    if os.path.isdir(flashp_dir) and os.listdir(flashp_dir):
        raise FileNotFoundError(
            f"'{flashp_dir}' already exists and is non-empty, but doesn't look like a "
            f"FLASH-P checkout (missing {marker}). Point flashp_dir at an empty/nonexistent "
            f"path to clone into, or at an existing FLASH-P checkout."
        )

    print(f"[flash_p_integration] Cloning FLASH-P from {repo_url} into '{flashp_dir}'...")
    subprocess.run([git_binary, 'clone', repo_url, flashp_dir], check=True)
    return flashp_dir


def _propagate_streamlit_ctx(thread):
    """Best-effort: give a background reader thread (see the streaming path
    in _run_claude_headless below) access to Streamlit's per-script
    context, before starting it. Without this, a `log_callback` that
    updates Streamlit UI (the whole point of it - showing live progress in
    the GUI) triggers a 'missing ScriptRunContext! This warning can be
    ignored when running in bare mode' warning on every single streamed
    event, since a plain threading.Thread has no script context of its
    own - harmless per Streamlit's own message, but noisy enough over a
    long multi-batch run to bury the actually useful log output. Calling
    add_script_run_ctx(thread) with no explicit ctx argument captures the
    CALLING thread's context automatically - safe to call from here since
    _run_claude_headless always runs on the actual Streamlit script thread
    (a button click handler), never from a background thread itself. A
    no-op when Streamlit isn't installed/importable (e.g. this module used
    standalone, outside any Streamlit app) or the internal API it relies
    on isn't present in some Streamlit version - this is a display nicety,
    never something the actual Claude Code run should depend on.
    """
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(thread)
    except Exception:
        pass


def _run_claude_headless(prompt, cwd, claude_binary='claude', permission_mode='dontAsk',
                          allowed_tools=DEFAULT_ALLOWED_TOOLS, model=None, timeout_seconds=3600,
                          extra_env=None, resume_session_id=None, log_callback=None):
    """Invoke Claude Code non-interactively (`claude -p ...`, Claude Code's
    documented headless mode) with `prompt`, from working directory `cwd`.
    Returns an object with the same .returncode/.stdout/.stderr interface
    as subprocess.CompletedProcess, regardless of which code path below
    actually produced it.

    permission_mode defaults to 'dontAsk' rather than 'acceptEdits',
    'bypassPermissions', or 'auto':
      - 'acceptEdits' only auto-approves the Edit/Write tool category -
        FLASH-P also needs Bash (validator/exporter scripts), WebSearch/
        WebFetch (literature review), and Task (subagent dispatch), none of
        which 'acceptEdits' covers, so a headless run would still block on
        a permission prompt for every one of those calls.
      - 'bypassPermissions' (--dangerously-skip-permissions) skips every
        check, but Claude Code refuses to start in this mode at all when
        the process is root/sudo ("cannot be used with root/sudo privileges
        for security reasons") - a common situation for a containerized
        deployment - and even as a non-root user it still requires a
        one-time interactive "I accept" dialog before it can be used
        headlessly.
      - 'dontAsk' auto-approves everything in `allowed_tools` (see
        DEFAULT_ALLOWED_TOOLS above, which already covers what FLASH-P
        needs) and silently denies anything outside that list, rather than
        prompting - no dialog, no root restriction, and Anthropic's own
        headless/CI guidance recommends it over 'bypassPermissions' for
        exactly this reason. If FLASH-P ever needs a tool not in
        `allowed_tools`, that call is denied (not blocked waiting for
        approval) and will show up as a failure/incomplete run rather than
        hanging - widen `allowed_tools` if that happens (this is exactly
        how the 'Monitor' tool gap that once stalled a run was found).
      - 'auto' routes each tool call through Claude Code's own classifier
        (code.claude.com/docs/en/auto-mode-config) instead of a static
        allowlist - it reads FLASH-P's own CLAUDE.md for context and judges
        each action's actual risk, so it doesn't need `allowed_tools` kept
        up to date by hand the way 'dontAsk' does (no more discovering a
        missing tool - Monitor, Glob, Grep, Agent - one stalled run at a
        time). Anthropic's docs note auto mode "may have a small impact on
        token consumption, cost, and latency for tool calls" (a classifier
        call per action) and that it isn't available for every account/
        model - see `_summarize_auto_mode_unavailable`. `allowed_tools` is
        NOT sent when using this mode: auto mode suspends broad tool-
        category grants like the ones in DEFAULT_ALLOWED_TOOLS anyway (only
        narrow, command-scoped rules like 'Bash(npm test)' carry over), so
        sending them would be inert. CLAUDE_CODE_ENABLE_AUTO_MODE=1 is set
        automatically - required on Claude Code v2.1.158-v2.1.206 for some
        session types, harmless no-op otherwise.
    Pass permission_mode='bypassPermissions' explicitly if you specifically
    want the older skip-everything behavior on a non-root machine, or
    'acceptEdits'/'default' to require manual approval.

    resume_session_id : str, optional
        If given, continues that session (`claude -p --resume <id> ...`)
        instead of starting a new one - `prompt` becomes a continuation
        instruction rather than the original task. Session lookup is
        scoped to `cwd`, so this only works from the same working
        directory the original session ran in (run_flash_p always uses
        the same claude_cwd for a given flashp_dir/pipeline_variant, so
        this is satisfied automatically there). Useful for continuing a
        run that was interrupted mid-pipeline - e.g. by a rate/session
        limit (see _summarize_rate_limit) - without repeating, and paying
        for, the work already done.

    log_callback : callable(str), optional
        If given, switches this call from the default single-shot
        `--output-format json` (silent until the whole run finishes, then
        one JSON object) to `--output-format stream-json --verbose`
        (Claude Code's documented live-streaming mode:
        code.claude.com/docs/en/headless - newline-delimited JSON events
        emitted as they happen: session start, each tool call, assistant
        text, then a final result event). `log_callback` is called with a
        short, human-readable status string for each event as it arrives -
        see `_format_stream_event` - so a long run (FLASH-P's full pipeline,
        or a multi-batch gene-location curation run) can show live progress
        somewhere (a GUI log panel, a terminal) instead of going silent for
        however many minutes it takes. A raised exception inside
        `log_callback` itself is swallowed (this is a "nice to have"
        progress display, not something the actual Claude Code run should
        ever be brought down by). Every existing caller (log_callback=None,
        the default) is completely unaffected - same request, same
        blocking subprocess.run-equivalent behavior as before this
        parameter existed. Regardless of which output format was actually
        used, the RETURNED object's .stdout always ends up holding just the
        final result event's JSON (matching exactly what `--output-format
        json` would have returned directly), so every existing caller that
        does `json.loads(result.stdout)` - `_summarize_permission_denials`,
        `_summarize_rate_limit`, `_summarize_auto_mode_unavailable` -
        keeps working unchanged either way.
    """
    streaming = log_callback is not None

    cmd = [claude_binary, '-p', prompt,
           '--permission-mode', permission_mode,
           '--output-format', ('stream-json' if streaming else 'json')]
    if streaming:
        # stream-json is documented as requiring --verbose. Token-level text
        # deltas would additionally need --include-partial-messages, which
        # is deliberately left off: one event per tool call/message is
        # plenty granular for a "batch N/M running..." style progress log,
        # and skips the overhead of streaming every individual token.
        cmd += ['--verbose']
    if permission_mode != 'auto':
        cmd += ['--allowedTools', ','.join(allowed_tools)]
    if resume_session_id:
        cmd += ['--resume', resume_session_id]
    if model:
        cmd += ['--model', model]

    env = os.environ.copy()
    if permission_mode == 'auto':
        env.setdefault('CLAUDE_CODE_ENABLE_AUTO_MODE', '1')
    if extra_env:
        # A value of None explicitly removes that var from the subprocess's
        # environment rather than setting it - e.g. needed to force a
        # subscription CLAUDE_CODE_OAUTH_TOKEN to take priority, since
        # Claude Code always prefers an ANTHROPIC_API_KEY that's already
        # present in the environment over subscription auth in headless mode.
        for k, v in extra_env.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v

    print(f"[flash_p_integration] Running: {' '.join(cmd[:2])} \"...\" (cwd={cwd})")

    if not streaming:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                                 timeout=timeout_seconds, env=env)
        return result

    # --- streaming path -----------------------------------------------
    # subprocess.run(capture_output=True) only returns after the WHOLE
    # process exits, so it can't be used for live progress - reading stdout
    # line by line as stream-json events are produced is what actually
    # makes this "live" rather than "silent for N minutes, then everything
    # at once". stdout and stderr are drained on separate threads (not
    # read sequentially) specifically to avoid the classic subprocess
    # deadlock where one pipe's OS buffer fills up while the code is
    # blocked reading the other - and threads (rather than e.g.
    # select.select on the pipes) keep this working on Windows too, where
    # select() doesn't support pipes.
    process = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                bufsize=1)

    stdout_lines = []
    stderr_chunks = []
    final_result_line = [None]  # mutable cell so the reader thread can write to it

    def _read_stdout():
        for line in process.stdout:
            line = line.rstrip('\n')
            if not line:
                continue
            stdout_lines.append(line)
            try:
                event = json.loads(line)
            except (ValueError, TypeError):
                continue
            if event.get('type') == 'result':
                final_result_line[0] = line
            try:
                log_callback(_format_stream_event(event))
            except Exception:
                pass  # see log_callback's docstring above - never let this break the real run

    def _read_stderr():
        for line in process.stderr:
            stderr_chunks.append(line)

    stdout_thread = threading.Thread(target=_read_stdout, daemon=True)
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    _propagate_streamlit_ctx(stdout_thread)
    _propagate_streamlit_ctx(stderr_thread)
    stdout_thread.start()
    stderr_thread.start()

    try:
        returncode = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        raise subprocess.TimeoutExpired(cmd, timeout_seconds)

    stdout_thread.join(timeout=5)
    stderr_thread.join(timeout=5)

    # The final 'result' event has the exact same fields --output-format
    # json would have returned in one shot - reconstructing .stdout to hold
    # just that (rather than every event) is what keeps every existing
    # json.loads(result.stdout) caller working unchanged. Fall back to all
    # collected lines if a 'result' event was never seen (e.g. the process
    # was killed before completing) so there's still SOMETHING to inspect.
    final_stdout = final_result_line[0] if final_result_line[0] is not None else '\n'.join(stdout_lines)

    return subprocess.CompletedProcess(cmd, returncode, stdout=final_stdout, stderr=''.join(stderr_chunks))


def _format_stream_event(event):
    """Best-effort short, human-readable status string for one stream-json
    event (see _run_claude_headless's log_callback parameter) - not
    exhaustive, just enough for a live progress log to show something
    meaningful (session started, which tool is being used, a snippet of
    assistant text, the final result) instead of raw JSON lines."""
    etype = event.get('type')
    if etype == 'system':
        return "Session started."
    if etype == 'result':
        return f"Done: {event.get('result', '')}"
    if etype == 'assistant':
        for block in (event.get('message') or {}).get('content') or []:
            if block.get('type') == 'tool_use':
                return f"Using tool: {block.get('name', '?')}"
            if block.get('type') == 'text' and block.get('text', '').strip():
                text = block['text'].strip()
                return text if len(text) <= 200 else text[:200] + '...'
    return f"[{etype}]"


def _summarize_auto_mode_unavailable(stdout):
    """Best-effort detection of Claude Code reporting 'auto' permission mode
    as unavailable for this account/model (see
    code.claude.com/docs/en/auto-mode-config for the supported models and
    any organization-level control on Team/Enterprise plans - the exact
    eligibility rules can change, so this checks the actual response rather
    than assuming). Returns None if stdout isn't parseable JSON or doesn't
    look like this specific failure.
    """
    if not stdout:
        return None
    try:
        data = json.loads(stdout)
        text = str(data.get('result', ''))
    except (ValueError, TypeError):
        text = stdout
    lowered = text.lower()
    if 'auto mode' in lowered and ('not available' in lowered or 'unavailable' in lowered
                                    or 'not supported' in lowered):
        return (
            f"Claude Code reported: {text!r}. permission_mode='auto' doesn't appear to be "
            f"available for this account/model (see code.claude.com/docs/en/auto-mode-config "
            f"for current requirements). Switch permission_mode back to 'dontAsk', which works "
            f"on any account."
        )
    return None


def _summarize_rate_limit(stdout):
    """Best-effort detection of a rate/session-limit hit (e.g.
    api_error_status 429, "You've hit your session limit · resets 7:20pm
    (UTC)") in a `claude -p --output-format json` result - a genuine usage
    cap on the account behind these credentials, not a bug - plus the
    session_id needed to resume the SAME run once it clears (see
    run_flash_p's resume_session_id parameter), rather than restarting
    FLASH-P from scratch and re-paying for work already done. Returns None
    if stdout isn't parseable JSON or doesn't show a 429.
    """
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    if data.get('api_error_status') != 429:
        return None
    session_id = data.get('session_id')
    message = data.get('result') or 'a rate/session limit was reached'
    note = (
        f"Hit a usage limit partway through this run: {message!r}. This is a cap on "
        f"the Claude account behind these credentials (subscription session limits "
        f"reset on a schedule; API rate limits reset faster) - the run itself was "
        f"progressing normally up to this point, so it isn't a bug in FLASH-P or "
        f"this integration."
    )
    if session_id:
        note += (
            f" Once the limit resets, continue this SAME run instead of starting a "
            f"new one (which would repeat, and re-pay for, everything already done) "
            f"by calling run_flash_p(..., resume_session_id={session_id!r})."
        )
    return note


def _summarize_permission_denials(stdout):
    """Best-effort extraction of the 'permission_denials' field from a
    `claude -p --output-format json` result. 'dontAsk' (DEFAULT permission
    mode - see DEFAULT_ALLOWED_TOOLS above) denies any tool call outside
    allowed_tools silently rather than prompting for it or failing loudly,
    so a run can exit 0 having quietly given up partway through. Surfacing
    which tool(s) got denied here saves having to manually find that one
    field inside a large raw JSON blob every time this happens - it's how
    the 'Monitor' tool gap that stalled an earlier FLASH-P run got found.
    Returns None if stdout isn't parseable JSON or records no denials.
    """
    try:
        data = json.loads(stdout)
    except (ValueError, TypeError):
        return None
    denials = data.get('permission_denials') or []
    if not denials:
        return None
    denied_tools = sorted({d.get('tool_name', '?') for d in denials})
    return (
        f"Claude Code silently DENIED {len(denials)} tool call(s) during this run "
        f"(tool(s): {', '.join(denied_tools)}) - 'dontAsk' denies anything outside "
        f"allowed_tools rather than prompting for it. This is very likely why the run "
        f"didn't complete: add the denied tool(s) to DEFAULT_ALLOWED_TOOLS (or the "
        f"allowed_tools argument) and re-run."
    )


def _validate_network_schema(flashp_claude_dir, net_dir):
    """Best-effort QC gate: run FLASH-P's own schema validator
    (`Agent/shared/validate_schema.py --network <NET>`, exit 0 = valid) over
    the freshly built network, exactly as the pipeline's own
    `/run-flashp` final-report step recommends. Never raises - a validator
    failure is reported, not fatal, since a human may still want to inspect
    a partially-valid run rather than lose it outright.
    """
    validator = os.path.join(flashp_claude_dir, 'Agent', 'shared', 'validate_schema.py')
    if not os.path.isfile(validator):
        return None
    try:
        result = subprocess.run(['python', validator, '--network', net_dir],
                                 cwd=flashp_claude_dir, capture_output=True, text=True, timeout=300)
        ok = (result.returncode == 0)
        if not ok:
            warnings.warn(
                f"FLASH-P's own schema validator reported issues with the generated network "
                f"(exit code {result.returncode}). stdout/stderr follow - inspect before trusting "
                f"this network.json:\n{result.stdout}\n{result.stderr}"
            )
        return ok
    except Exception as e:  # pragma: no cover - best-effort only
        warnings.warn(f"Could not run FLASH-P's schema validator: {e}")
        return None


DEFAULT_RESUME_PROMPT = (
    "Continue the FLASH-P run from exactly where it left off - check on the status of "
    "any in-progress background jobs (e.g. evidence verification) first, then proceed "
    "through the remaining pipeline steps to completion."
)


def run_flash_p(species, phenotype, flashp_dir, pipeline_variant='Flash-P_Plant',
                 run_name=None, claude_binary='claude', model=None,
                 permission_mode='dontAsk', allowed_tools=DEFAULT_ALLOWED_TOOLS,
                 timeout_seconds=3600, repo_url=DEFAULT_FLASHP_REPO_URL,
                 validate=True, extra_env=None, resume_session_id=None,
                 resume_prompt=DEFAULT_RESUME_PROMPT, log_callback=None):
    """Run FLASH-P end to end for (species, phenotype) and return the path to
    the resulting network.json, relocated to
    Outcome/Local_FlashP_Outcome/<run_name>/network/network.json.

    Parameters
    ----------
    species : str, e.g. 'Arabidopsis thaliana'
    phenotype : str, e.g. 'Shoot Branching'
    flashp_dir : str
        Path to a local clone of https://github.com/CMits/FlashP (cloned
        automatically via `ensure_flash_p_repo` if the path doesn't exist or
        is empty).
    pipeline_variant : {'Flash-P_Plant', 'Flash-P_Medical', 'Flash-P_Animal'}
        Which FLASH-P version to run - see the FLASH-P README's "Versions"
        table. Defaults to the plant pipeline (EasiGP's own use case).
    run_name : str, optional
        Name of the destination folder under Outcome/Local_FlashP_Outcome/.
        Defaults to '<phenotype_slug_lowercased>_local_network', matching
        the naming of FLASH-P's own shipped example
        (Outcome/Local_FlashP_Outcome/shoot_branching_local_network/).
    claude_binary : str
        Name/path of the Claude Code executable (`npm install -g
        @anthropic-ai/claude-code`).
    permission_mode : str
        Defaults to 'dontAsk' so the run completes unattended without
        needing a permission dialog or non-root privileges - see
        `_run_claude_headless` for the full comparison against
        'acceptEdits'/'bypassPermissions' and why 'dontAsk' is the safer,
        more portable choice for a headless run.
    resume_session_id : str, optional
        If given, continues that prior session instead of starting a new
        FLASH-P run - `resume_prompt` is sent instead of the original
        `/run-flashp` prompt. `species`/`phenotype` must still be the SAME
        values as the original run (they're used to locate the finished
        networks/<slug>/network.json afterward, regardless of whether this
        call started or resumed the session) - only the initial prompt text
        is skipped. Get resume_session_id from a previous failed run's
        error message (see `_summarize_rate_limit`) - most useful after a
        rate/session limit interrupted an otherwise-progressing run, so you
        don't repeat (and re-pay for) work already done.
        `flashp_dir`/`pipeline_variant` must also match the original run,
        since Claude Code's session lookup is scoped to the working
        directory the session started in.
    validate : bool
        If True (default), also run FLASH-P's own
        `Agent/shared/validate_schema.py` over the result as a QC gate
        (warns, does not raise, on failure - see `_validate_network_schema`).
    log_callback : callable(str), optional
        Forwarded to `_run_claude_headless` - see its docstring for the
        live-progress mechanism this enables. Particularly useful here
        given how long a full FLASH-P run typically takes (tens of minutes)
        with nothing else to show it's actually progressing.

    Notes
    -----
    FLASH-P is expected to write its output to
    `networks/<Phenotype_Slug>/network/network.json` per its own CLAUDE.md
    convention, but doesn't always name that folder exactly the way this
    module's mechanical Title_Case_With_Underscores slugification of
    `phenotype` would (e.g. singular vs plural - "Rosette_Branch_Number"
    vs "Rosette_Branch_Numbers" - seen in a real run). If the expected path
    isn't found, this falls back to the most recently modified
    networks/<name>/network/network.json instead of failing outright - see
    the inline comment at that fallback for why "most recent" is a safe
    heuristic here.

    Returns
    -------
    str : absolute path to Outcome/Local_FlashP_Outcome/<run_name>/network/network.json
    """
    if pipeline_variant not in ('Flash-P_Plant', 'Flash-P_Medical', 'Flash-P_Animal'):
        raise ValueError(f"pipeline_variant must be one of 'Flash-P_Plant'/'Flash-P_Medical'/"
                          f"'Flash-P_Animal', got {pipeline_variant!r}")

    flashp_dir = ensure_flash_p_repo(flashp_dir, repo_url=repo_url)
    claude_cwd = os.path.join(flashp_dir, pipeline_variant, 'Claude')
    if not os.path.isdir(claude_cwd):
        raise FileNotFoundError(
            f"'{claude_cwd}' does not exist - only the Claude Code variant of each FLASH-P "
            f"pipeline is currently wired up by this module (pipeline_variant={pipeline_variant!r})."
        )

    prompt = resume_prompt if resume_session_id else f"/run-flashp {phenotype} in {species}"
    try:
        result = _run_claude_headless(
            prompt, cwd=claude_cwd, claude_binary=claude_binary, permission_mode=permission_mode,
            allowed_tools=allowed_tools, model=model, timeout_seconds=timeout_seconds, extra_env=extra_env,
            resume_session_id=resume_session_id, log_callback=log_callback,
        )
    except FileNotFoundError as e:
        raise RuntimeError(_claude_binary_not_found_message(claude_binary, e)) from e

    if result.returncode != 0:
        notes = [n for n in (_summarize_rate_limit(result.stdout),
                              _summarize_permission_denials(result.stdout),
                              _summarize_auto_mode_unavailable(result.stdout)) if n]
        raise RuntimeError(
            f"FLASH-P run failed (claude exited with code {result.returncode})."
            + ("\n" + "\n".join(notes) if notes else "") +
            f"\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )

    slug = _phenotype_slug(phenotype)
    source_net_dir = os.path.join(claude_cwd, 'networks', slug)
    source_network_json = os.path.join(source_net_dir, 'network', 'network.json')

    if not os.path.isfile(source_network_json):
        # FLASH-P doesn't always name its own output folder exactly the way
        # our mechanical Title_Case_With_Underscores slugification would -
        # e.g. a "Rosette Branch Numbers" run's own final report said
        # "Artifacts: networks/Rosette_Branch_Number/..." (singular), not
        # "Rosette_Branch_Numbers" (plural) - evidently the agent's own
        # stylistic judgment, not a literal rule we can rely on matching.
        # Rather than fail whenever this happens, fall back to whatever
        # folder FLASH-P actually just created: the most recently modified
        # networks/<name>/network/network.json under this run's own
        # networks/ directory. Since this only runs immediately after a
        # freshly-completed invocation, "most recently modified" reliably
        # means "the one this run just wrote" even if older runs' folders
        # are sitting alongside it - the same trusted pattern already used
        # by locate_network_json() for browsing past runs.
        networks_root = os.path.join(claude_cwd, 'networks')
        fallback_candidates = []
        if os.path.isdir(networks_root):
            for entry in os.listdir(networks_root):
                candidate_json = os.path.join(networks_root, entry, 'network', 'network.json')
                if os.path.isfile(candidate_json):
                    fallback_candidates.append(candidate_json)

        if fallback_candidates:
            fallback_candidates.sort(key=os.path.getmtime, reverse=True)
            source_network_json = fallback_candidates[0]
            source_net_dir = os.path.dirname(os.path.dirname(source_network_json))
            print(f"[flash_p_integration] Expected folder 'networks/{slug}/' wasn't used - "
                  f"FLASH-P named it '{os.path.basename(source_net_dir)}/' instead. Using that "
                  f"folder since it's the most recently modified one with a network.json.")
        else:
            notes = [n for n in (_summarize_rate_limit(result.stdout),
                                  _summarize_permission_denials(result.stdout),
                                  _summarize_auto_mode_unavailable(result.stdout)) if n]
            raise RuntimeError(
                f"FLASH-P's own claude process exited successfully, but no network.json was "
                f"found anywhere under '{networks_root}' - not at the expected "
                f"'networks/{slug}/network/network.json' (CLAUDE.md's own convention), nor "
                f"under any other folder name."
                + ("\n" + "\n".join(notes) if notes else "") +
                f"\nThe run's own final report (claude's stdout, captured below) may explain "
                f"why:\n{result.stdout}"
            )

    if validate:
        _validate_network_schema(claude_cwd, source_net_dir)

    if run_name is None:
        run_name = f"{slug.lower()}_local_network"

    dest_dir = os.path.join(flashp_dir, 'Outcome', 'Local_FlashP_Outcome', run_name)
    os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
    shutil.copytree(source_net_dir, dest_dir, dirs_exist_ok=True)

    dest_network_json = os.path.join(dest_dir, 'network', 'network.json')
    print(f"[flash_p_integration] FLASH-P run complete: '{source_net_dir}' -> '{dest_dir}'. "
          f"network.json at '{dest_network_json}'.")
    return dest_network_json


def locate_network_json(flashp_dir, run_name=None):
    """Find an already-completed run's network.json under
    Outcome/Local_FlashP_Outcome/, without (re-)running FLASH-P. If
    `run_name` is given, looks there directly; otherwise returns the most
    recently modified network.json under Outcome/Local_FlashP_Outcome/**/network/.
    Useful for a GUI that lets a user pick from previous local runs.
    """
    root = os.path.join(flashp_dir, 'Outcome', 'Local_FlashP_Outcome')
    if run_name is not None:
        candidate = os.path.join(root, run_name, 'network', 'network.json')
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"No network.json found at '{candidate}'.")
        return candidate

    candidates = []
    shallow_candidates = []  # <run_name>/network/network.json - the canonical, top-level result
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            if 'network.json' in filenames and os.path.basename(dirpath) == 'network':
                full_path = os.path.join(dirpath, 'network.json')
                candidates.append(full_path)
                # A run can contain nested per-model-variant copies too (e.g.
                # <run_name>/runs/<model>/network/network.json, as FLASH-P's
                # own shipped example does) - those are intermediate/
                # exploratory artefacts, not the canonical result, and can be
                # *more* recently modified than the top-level one, so a plain
                # "most recent mtime" search can silently return the wrong
                # file. Only <run_name>/network/network.json (exactly one
                # directory level below `root`) counts as canonical here.
                rel_parts = os.path.relpath(full_path, root).split(os.sep)
                if len(rel_parts) == 3:  # <run_name>/network/network.json
                    shallow_candidates.append(full_path)

    pool = shallow_candidates if shallow_candidates else candidates
    if not pool:
        raise FileNotFoundError(f"No network.json found anywhere under '{root}'.")

    pool.sort(key=os.path.getmtime, reverse=True)
    return pool[0]


def get_network_json(source, **kwargs):
    """Single entry point for the GUI's "upload a JSON" vs "generate with
    FlashP" choice (requirement 1: "Users can select either approach using
    the GUI").

    source='upload' : kwargs must include uploaded_path (str) - validated
                       and returned as-is.
    source='flashp'  : runs (or locates an already-completed) FLASH-P run.
                       kwargs are forwarded to run_flash_p(), except that
                       passing run_only_locate=True skips re-running FLASH-P
                       and instead calls locate_network_json() with the same
                       flashp_dir/run_name.

    Returns the path to a network.json, ready for
    Preprocess.gene_network_prior.load_network_json().
    """
    if source == 'upload':
        uploaded_path = kwargs.get('uploaded_path')
        if not uploaded_path or not os.path.isfile(uploaded_path):
            raise FileNotFoundError(f"source='upload' requires an existing uploaded_path, got "
                                     f"{uploaded_path!r}.")
        with open(uploaded_path, 'r', encoding='utf-8-sig') as f:
            json.load(f)  # fail fast on malformed JSON before it reaches the rest of the pipeline
        return uploaded_path

    elif source == 'flashp':
        if kwargs.pop('run_only_locate', False):
            return locate_network_json(kwargs['flashp_dir'], run_name=kwargs.get('run_name'))
        return run_flash_p(**kwargs)

    else:
        raise ValueError(f"source must be 'upload' or 'flashp', got {source!r}")
