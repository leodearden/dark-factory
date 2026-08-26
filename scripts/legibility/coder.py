#!/usr/bin/env python3
"""scripts/legibility/coder.py — Haiku trickle coder: confusion digest ->
strict-JSON §7.3 coding record.

Task delta of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.3, contract §7.3, boundary tests §8.1 consumer side + §8.6). Reads ONE
confusion digest (alpha/digest.py output), builds a COMPACT codebook index
(entry ids + titles + one-line causes — NOT the full YAML) from the v2
codebook (gamma/codebook.py), invokes ONE headless
``claude -p --model <model>`` call (default haiku), parses the model's
strict-JSON judgment, and assembles it into a deterministic-header coding
record that codebook.validate_coding_record schema-gates.

Dependency-light library + argparse CLI, a plain-Python scripts/legibility
sibling of digest.py/codebook.py — deliberately does NOT import the
heavyweight async ``shared.cli_invoke`` machinery (usage gates, cost
stores, cap-retry, transcript watchdogs). What it DOES take from ``shared``
is one pure function: ``cap_markers.looks_like_blocking_banner``, the loose
OR-substring DEFER GATE — the same matcher ``census.preflight_headroom``
uses, and explicitly not the strict production cap detector
(``usage_gate.detect_cap_hit``), whose combined prefix-AND-confirm policy is
tuned for account failover. This module has nothing to fail over TO: the
trickle unit runs under an interpreter where the orchestrator config, and
therefore a multi-account ``UsageGate``, is unreachable. A defer gate is
exactly the contract it needs, and ``cap_markers``' own docstring argues for
that split (task 4736). The real LLM call lives behind
exactly one swappable seam, the module-level ``_invoke_cli``, which every
public function accepts as an ``invoke`` override. What no test ever does
is spawn a REAL model — but the seam ITSELF is exercised, so "the LLM is
always mocked" is not the same claim as "``_invoke_cli`` never runs": see
its own docstring for which tests reach it and how they stay free.

Never-fabricate contract (codebook lesson ``one-shot-subagent-contract`` —
the fail-soft fallback that hid a total outage): a CLI-invocation error,
unparseable output, or schema-invalid record is SKIPPED + counted, never
partially applied and never fabricated into an empty verdict — and is
DISTINGUISHED from a legitimately empty-but-schema-valid record
(``{"matches": [], "candidates": []}``), which is a genuine success. A
batch whose failure fraction STRICTLY exceeds 50% (failed/total > 0.5) is a
run-level FAILURE: the CLI then writes ZERO coding records and exits
non-zero. This module never escalates and never writes the codebook —
that is epsilon/gamma's job.

SKIPPED + COUNTED NOW ALSO MEANS ANNOUNCED (task 4511): every per-digest
failure is logged at WARNING on ``legibility.coder`` as it happens, naming
the session and the reason. Counting alone was not enough, because the
count only ever reaches a human through epsilon's storm escalation — and a
SUB-storm batch (failed/total <= 0.5) is ``status="ok"``, so those failures
previously reached no sink whatsoever. Escalation is still not this
module's job; a journal line is not an escalation.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute. An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy of this script running from a worktree would scan cap-banner text using
# the MAIN checkout's marker list rather than its own. Same reasoning and same
# form as census.py:74-88 (itself citing tasks 2881/2882/3329), with
# parents[2] because coder.py sits at the same depth as census.py
# (scripts/legibility/, not scripts/). Unconditional -- deliberately NOT
# inside a `__main__` guard -- because the `shared.cap_markers` import it
# enables is module-level, so it must resolve under pytest and package import
# too.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

import codebook as codebook_mod  # noqa: E402
import yaml  # noqa: E402
from shared.cap_markers import looks_like_blocking_banner  # noqa: E402

logger = logging.getLogger("legibility.coder")


class CoderParseError(Exception):
    """Raised when the coder cannot parse a digest's frontmatter or the
    LLM's raw output into a usable structure. Never silently defaulted —
    callers must treat this as a hard per-digest failure (never-fabricate
    contract)."""


class CoderInvocationError(Exception):
    """Raised when the ``claude -p --model`` subprocess invocation fails —
    non-zero exit or a timeout. Carries a tail of BOTH output streams for
    diagnosis, each labelled. Never silently swallowed: code_digest turns
    this into a per-digest failure, never a fabricated record.

    BOTH streams, not just stderr, because of what happened on 2026-08-24:
    the claude CLI wrote its usage-cap banner to STDOUT and exited 1, and
    this error embedded only ``(proc.stderr or "")[-2000:]``. With stderr
    empty, the reason that reached the journal, the epsilon escalation and
    ``run.failures`` was the bare ``claude CLI exited 1 (model='haiku',
    ...): `` — nothing after the colon — on 17 of 20 digests. The CLI had
    stated exactly what was wrong and the coder discarded it, so a night
    with one plain cause was investigated as twenty causeless failures.
    A diagnostic the process EMITTED must never be dropped on the floor
    because it arrived on the less-expected stream.
    """


class CoderCapExhausted(CoderInvocationError):
    """Raised when the CLI answered with a capacity/auth banner instead of a
    model turn — i.e. there is no headroom left to code this digest.

    A SUBCLASS, not a sibling, and that is load-bearing: three sites already
    catch ``CoderInvocationError`` (``code_digest`` here,
    ``census._build_default_verify_fn`` and ``census.preflight_headroom``)
    and none of them is touched by this task. A sibling type would escape all
    three, turning a typed per-digest failure into an uncaught crash that
    takes down the whole batch.

    **This is a NORMAL operating condition, never a coder defect.** Leo's
    standing directive (sibling task 4503): an all-accounts-capped night is
    expected weather, not an incident. Before this existed, 2026-08-24
    presented as 17 of 20 hard per-digest failures, tripped ``code_digests``'
    >50% storm threshold, and became ``exit_code=1`` plus an ERROR-level
    escalation — an infra page for a condition ruled routine.

    ``marker`` names the banner marker that matched, so a deferral reason can
    quote WHICH signal fired — the difference between an operator reading
    "deferred: weekly limit" and reading "deferred". Mirrors
    ``census.preflight_headroom``'s "...carries a banner marker: {marker!r}".

    Never fabricated into a verdict. A capped digest yields no record at all;
    it is labelled and excluded, exactly as ``evals/runner.py`` excludes a
    ``cap_exhausted:`` cell from a reported mean rather than scoring it 0.0.
    """

    def __init__(self, message: str, *, marker: str) -> None:
        super().__init__(message)
        self.marker = marker


@dataclass
class CodingResult:
    """Outcome of coding one digest.

    ``ok=True`` means ``record`` is a schema-valid §7.3 coding record —
    including a legitimately empty one (``matches=[]``, ``candidates=[]``
    is a genuine finding, not a failure). ``ok=False`` means ``record`` is
    None and ``reason`` explains why: a CLI invocation error, unparseable
    LLM output, or a schema-invalid assembled record are never partially
    applied and never fabricated into a record.

    ``capped=True`` means the CLI answered with a capacity/auth banner
    instead of a model turn. It is a strict REFINEMENT of ``ok=False``,
    never a third success state — ``record`` is still None and the
    never-fabricate contract is untouched. What it records is a fact about
    the ACCOUNT, not a judgment about the digest: this digest was never
    actually coded, so charging it to the coder as a failure of the work is
    simply wrong. Downstream (``code_digests`` tallies it, ``is_cap_deferral``
    reads the tally) that distinction is what separates "there was no
    headroom tonight" from "the coder is broken" — on 2026-08-24 their
    conflation turned expected weather into an ERROR-level infra page.
    """

    ok: bool
    record: dict | None
    reason: str | None = None
    session: str | None = None
    capped: bool = False


# ---------------------------------------------------------------------------
# build_codebook_index — compact codebook index (id + title + one-line cause)
# ---------------------------------------------------------------------------

_INDEX_CAUSE_MAX_LEN = 200
"""Character cap for a codebook entry's one-line cause summary in the
compact index — keeps the prompt's token budget bounded regardless of how
long-winded a real ``cause`` field (multi-paragraph, see e.g.
one-shot-subagent-contract in the live codebook) is."""


def _one_line_cause(cause) -> str:
    """Collapse a (possibly multi-paragraph, possibly absent) cause value
    to a single whitespace-collapsed line, capped to
    ``_INDEX_CAUSE_MAX_LEN`` characters."""
    if not cause:
        return ""
    collapsed = " ".join(str(cause).split())
    if len(collapsed) > _INDEX_CAUSE_MAX_LEN:
        collapsed = collapsed[:_INDEX_CAUSE_MAX_LEN].rstrip() + "..."
    return collapsed


def build_codebook_index(codebook: dict) -> str:
    """Render a COMPACT index of every codebook entry: one line each,
    ``- {id}: {title} — {one-line cause}`` — NOT the full YAML. Heavy
    fields (fix/fix_where/sightings/candidates) are never included.

    ALL entries are included, retired ones too: a census re-observes
    pre-fix traces (PRD §6 floor_days rationale), so a live sighting can
    still match a retired cause — dropping retired entries would force
    spurious candidates.
    """
    entries = (codebook.get("entries") or []) if isinstance(codebook, dict) else []
    lines = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = entry.get("id", "")
        title = entry.get("title", "")
        cause = _one_line_cause(entry.get("cause"))
        if cause:
            lines.append(f"- {entry_id}: {title} — {cause}")
        else:
            lines.append(f"- {entry_id}: {title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parse_frontmatter — digest's leading YAML frontmatter -> meta dict
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)


def parse_frontmatter(digest_text: str) -> dict:
    """Extract and parse a digest's leading ``---``...``---`` YAML
    frontmatter block (PRD §7.2), returning the meta dict (at minimum
    session/date/agent_class — the deterministic coding-record header
    fields). Raises CoderParseError if the delimiters are absent or the
    parsed block isn't a mapping — never silently defaults.
    """
    match = _FRONTMATTER_RE.match(digest_text)
    if not match:
        raise CoderParseError(
            "digest text has no leading '---'...'---' frontmatter block"
        )
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise CoderParseError(f"frontmatter block is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise CoderParseError(
            f"frontmatter block did not parse to a mapping, got {type(meta).__name__}"
        )
    return meta


# ---------------------------------------------------------------------------
# build_prompt — instructions + codebook index + digest, embedded verbatim
# ---------------------------------------------------------------------------

def build_prompt(digest_text: str, codebook_index: str) -> str:
    """Compose the full prompt handed to the trickle coder LLM.

    Embeds *codebook_index* and *digest_text* verbatim (pure data
    plumbing — a broken coder would drop one), the legal phase vocabulary
    (codebook.PHASES, including "unknown"), and the required strict-JSON
    output shape (PRD §7.3).
    """
    phases = ", ".join(codebook_mod.PHASES)
    return (
        "You are the trickle coder for the dark-factory agent-confusion "
        "codebook (plans/confusion-reduction-prd.md §7.3). Read the "
        "session digest below and decide which existing codebook entries "
        "it matches (if any), and whether it reveals any novel confusion "
        "causes not yet in the codebook (candidates).\n\n"
        "Never guess a phase you can't support from the evidence — use "
        '"unknown" instead. Legal phase values: ' + phases + ".\n\n"
        "Respond with STRICT JSON ONLY (no prose, no markdown fences), "
        "exactly this shape:\n"
        '{"matches": [{"entry_id": "...", "origin_phase": "...", '
        '"manifested_phase": "...", "invariant_violated": null, '
        '"note": "..."}], '
        '"candidates": [{"title": "...", "cause": "...", "area": "...", '
        '"origin_phase": "...", "manifested_phase": "...", '
        '"evidence_quote": "..."}]}\n'
        'If nothing matches and nothing is novel, respond with '
        '{"matches": [], "candidates": []}.\n\n'
        "=== CODEBOOK INDEX ===\n" + codebook_index + "\n\n"
        "=== SESSION DIGEST ===\n" + digest_text
    )


# ---------------------------------------------------------------------------
# parse_coder_output — raw LLM stdout -> judgment dict
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def parse_coder_output(raw: str) -> dict:
    """Parse the trickle coder LLM's raw stdout into a judgment dict.

    Tries, in order: (1) the whole string as JSON; (2) stripping a
    ```/```json fence and retrying; (3) — only if neither (1) nor (2)
    parsed as JSON at all — slicing from the first ``{`` to the last
    ``}`` and retrying (the "object embedded in surrounding prose" case).

    A candidate that parses cleanly to a NON-dict JSON value (a top-level
    array or scalar) fails immediately rather than falling through to
    brace-slicing: brace-slicing exists to rescue an object buried in
    prose noise, not to dig a nested object out of an already
    well-formed-but-wrong-shaped JSON value (e.g. slicing the first
    ``{...}`` out of a top-level ``[{...}]`` array would silently accept
    array-shaped output, which must instead raise). Output that never
    parses to a dict at all raises CoderParseError. Never returns a
    fabricated default.
    """
    primary_candidates = [raw]
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        primary_candidates.append(fence_match.group(1))

    for candidate in primary_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise CoderParseError(
            f"coder output parsed as {type(parsed).__name__}, expected a JSON object"
        )

    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        try:
            sliced = json.loads(raw[first_brace : last_brace + 1])
        except json.JSONDecodeError:
            sliced = None
        if isinstance(sliced, dict):
            return sliced

    raise CoderParseError(
        f"could not parse a JSON object from coder output: {raw[:200]!r}"
    )


# ---------------------------------------------------------------------------
# _invoke_cli — the ONE real `claude -p --model` subprocess boundary
# ---------------------------------------------------------------------------

_DEFAULT_INVOKE_TIMEOUT_SECS = 120
"""Default subprocess timeout (seconds) for a single _invoke_cli call."""

_CLAUDE_BIN_ENV_VAR = "LEGIBILITY_CLAUDE_BIN"
"""Env var overriding the `claude` binary path; falls back to the bare
"claude" (PATH-resolved -- /home/leo/.local/bin is on PATH)."""

_ERROR_STREAM_TAIL_CHARS = 2000
"""How much of EACH captured output stream a CoderInvocationError carries.

One constant for both streams, deliberately: this started as a bare
``[-2000:]`` on stderr alone, and the asymmetry that grew beside it (stdout
not bounded because stdout was not carried at all) is exactly the 2026-08-24
diagnostic loss. Bounded because the text lands verbatim in journal lines,
``run.failures`` entries and escalation bodies; the TAIL is kept because a
CLI's last words are its diagnostic ones."""


def _invoke_cli(
    prompt: str,
    model: str,
    *,
    claude_bin: str | None = None,
    timeout: float = _DEFAULT_INVOKE_TIMEOUT_SECS,
    cwd: str | os.PathLike | None = None,
) -> str:
    """Invoke the real headless ``claude -p --model <model>`` CLI exactly
    once, delivering *prompt* via stdin, and return its raw stdout.

    This is the ONE real-subprocess boundary in this module -- every
    public function accepts an ``invoke`` override, and most tests inject
    a fake one. *claude_bin* resolves, in order: the explicit argument,
    the ``LEGIBILITY_CLAUDE_BIN`` env var, else the bare ``"claude"``.

    THIS FUNCTION IS ITSELF UNDER TEST -- it is no longer true that "no
    test ever reaches it", and the resolution order above is exactly what
    keeps those tests free. Two suites reach it, from two modules:
    test_legibility_coder.py points *claude_bin* / ``LEGIBILITY_CLAUDE_BIN``
    at a FAKE ``claude`` script it writes itself (task 4510, argv/stdin
    delivery, non-zero exit, timeout, and the env-var branch), and
    test_legibility_nightly.py replays the 2026-08-18 ENOENT incident end
    to end by pointing ``LEGIBILITY_CLAUDE_BIN`` at a NONEXISTENT path and
    running ``run_nightly`` with no ``invoke=`` override at all (task
    4511). Both scrub PATH of any real ``claude`` first and assert
    ``shutil.which("claude") is None``, because the bare-name fallback at
    the end of the chain would otherwise turn a regression in the env-var
    lookup into genuine, billable model calls inside a unit test. Preserve
    that assertion if you touch the resolution order.

    *cwd*, when given, is the directory the headless CLI process RUNS IN;
    ``None`` (the default) is subprocess's own "inherit the parent's
    working directory", i.e. exactly the behavior every caller had before
    this parameter existed. It is load-bearing, not cosmetic: ``claude -p``
    SANDBOXES its tool access to the cwd tree, so a caller that wants the
    model to read a tree other than the launcher's MUST pass it -- every
    Read/Bash against that other tree is otherwise permission-denied, and
    non-interactively there is no prompt to approve. Proven on 2026-08-03:
    the legibility census verifying a project other than its launcher's cwd
    had every verifier read denied, and since the verify seam fails CLOSED
    per cluster (``census._build_default_verify_fn``) that surfaced as a
    silent mass rejection of every cluster rather than an error.

    Raises CoderInvocationError on a non-zero exit, a timeout, or a
    failure to START the process at all -- never silently swallowed, never
    a fabricated empty stdout. On a non-zero exit the error message carries
    a tail of BOTH output streams, each LABELLED, plus the resolved cwd, so
    a future sandbox/permission failure NAMES the directory the process was
    scoped to instead of leaving it to be inferred.

    Both streams because the CLI does not reliably diagnose itself on
    stderr: on 2026-08-24 it wrote a usage-cap banner to STDOUT and exited
    1, and a stderr-only message reached the journal EMPTY after the colon
    on 17 of 20 digests (see CoderInvocationError). The exit-0 RETURN
    contract is untouched by that -- stdout is still returned raw and
    unbounded there, which census._build_stage_invokes (wiring this
    function as its mining/verify/synthesis primitive) and
    census.preflight_headroom (scanning the returned reply itself) both
    depend on.

    That third case is why the ``OSError`` arm below exists. Passing *cwd*
    hands ``subprocess.run`` a second thing that can be missing besides the
    binary: a cwd that does not exist, is not a directory, or is not
    searchable makes it raise a RAW ``FileNotFoundError`` /
    ``NotADirectoryError`` / ``PermissionError``, which would escape this
    function and falsify the contract above. It also lands badly
    downstream: census's FIRST invoke is the headroom probe, and
    ``census.preflight_headroom`` folds ANY probe exception into
    ``HeadroomResult(ok=False)`` (deliberately fail-safe), so an operator
    typo in ``--project-root`` would read exactly like a usage-limit defer
    -- exit 0, an INFO escalation, and a census that silently never runs
    again. Wrapping it here keeps the failure typed; ``census.main`` also
    rejects a non-directory project_root up front so the loud error names
    the flag rather than the subprocess.
    """
    resolved_bin = claude_bin or os.environ.get(_CLAUDE_BIN_ENV_VAR) or "claude"

    try:
        proc = subprocess.run(
            [resolved_bin, "-p", "--model", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise CoderInvocationError(
            f"claude CLI timed out after {timeout}s (model={model!r}, "
            f"claude_bin={resolved_bin!r}, cwd={cwd!r})"
        ) from exc
    except OSError as exc:
        # The process never started: a missing/non-executable binary, or a
        # cwd that is missing / not a directory / not searchable. Both name
        # themselves in the underlying OSError text, so echo it verbatim
        # alongside BOTH candidates rather than guessing which one the
        # kernel objected to.
        raise CoderInvocationError(
            f"claude CLI could not be started (model={model!r}, "
            f"claude_bin={resolved_bin!r}, cwd={cwd!r}): {exc}"
        ) from exc

    if proc.returncode != 0:
        # BOTH streams, each labelled. The claude CLI does not reliably put
        # its own diagnostics on stderr -- on 2026-08-24 it wrote a usage-cap
        # banner to STDOUT and exited 1 -- so carrying one stream and
        # labelling neither loses both the text and the fact of WHICH stream
        # said it. See CoderInvocationError's docstring for the incident.
        stdout_tail = (proc.stdout or "")[-_ERROR_STREAM_TAIL_CHARS:]
        stderr_tail = (proc.stderr or "")[-_ERROR_STREAM_TAIL_CHARS:]
        message = (
            f"claude CLI exited {proc.returncode} (model={model!r}, "
            f"claude_bin={resolved_bin!r}, cwd={cwd!r}): "
            f"stdout={stdout_tail!r} stderr={stderr_tail!r}"
        )
        # Scan for a cap/auth banner ONLY here, on an already-FAILED
        # invocation. This is census's split-on-parse-success rule, adopted
        # unchanged: a failed invocation is never a verdict, so re-reading it
        # as a banner can only re-LABEL an already-failed digest. An exit-0
        # reply is the opposite -- arbitrary model output, whose JSON
        # `note`/`cause`/`evidence_quote` legitimately QUOTE cap-themed
        # sessions. census._build_default_verify_fn records what happens when
        # that distinction is lost: this repo's codebook is dominated by
        # clusters ABOUT usage and weekly limits, so the loose markers match
        # ordinary healthy content and the census aborted on cap-themed
        # clusters. The exit-0 path below is therefore left alone, which also
        # keeps census.preflight_headroom -- whose whole probe is "call this
        # function and scan what comes BACK" -- working unchanged.
        marker = looks_like_blocking_banner(f"{stdout_tail}\n{stderr_tail}")
        if marker:
            raise CoderCapExhausted(message, marker=marker)
        raise CoderInvocationError(message)

    return proc.stdout


# ---------------------------------------------------------------------------
# code_digest — one digest -> one CodingResult
# ---------------------------------------------------------------------------

def code_digest(
    digest_text: str,
    codebook: dict,
    *,
    project: str,
    model: str = "haiku",
    invoke=None,
) -> CodingResult:
    """Code one digest against one codebook.

    Flow: parse the digest's frontmatter -> build the compact codebook
    index -> build the prompt -> invoke the LLM (*invoke* override, or the
    real ``_invoke_cli`` by default) -> parse its strict-JSON judgment ->
    assemble a §7.3 coding record with a DETERMINISTIC header
    (session/date/agent_class from the digest's own frontmatter; project
    from *project* — the LLM never supplies the header) -> schema-gate it
    via codebook.validate_coding_record.

    Never-fabricate contract: unparseable/malformed digest frontmatter, an
    invocation error, a usage/auth CAP, unparseable LLM output, or a
    schema-invalid assembled record all come back as ``ok=False`` with
    ``record=None`` and ``reason`` set — never partially applied, never
    fabricated. The cap case additionally sets ``capped=True``: it is the
    one cause that says nothing about this digest or this coder, only that
    the account had no headroom left to look (see ``CoderCapExhausted``). A
    legitimately empty judgment (``{"matches": [], "candidates": []}``)
    that passes schema validation is a genuine ``ok=True`` success: "coded
    fine, found nothing" is never conflated with "coding failed" (codebook
    lesson one-shot-subagent-contract).
    """
    try:
        meta = parse_frontmatter(digest_text)
    except CoderParseError as exc:
        return CodingResult(ok=False, record=None, reason=str(exc), session=None)
    session = meta.get("session")

    index = build_codebook_index(codebook)
    prompt = build_prompt(digest_text, index)
    invoke_fn = invoke or _invoke_cli

    try:
        raw = invoke_fn(prompt, model)
    except CoderCapExhausted as exc:
        # ORDERED ABOVE the generic arm below, and that ordering is
        # load-bearing: CoderCapExhausted SUBCLASSES CoderInvocationError, so
        # reversing these two silently routes every cap into the generic arm
        # and the label is never applied.
        return CodingResult(
            ok=False, record=None, reason=str(exc), session=session, capped=True,
        )
    except CoderInvocationError as exc:
        return CodingResult(ok=False, record=None, reason=str(exc), session=session)

    try:
        judgment = parse_coder_output(raw)
    except CoderParseError as exc:
        return CodingResult(ok=False, record=None, reason=str(exc), session=session)

    record = {
        "session": session,
        "date": meta.get("date"),
        "project": project,
        "agent_class": meta.get("agent_class"),
        "matches": judgment.get("matches") or [],
        "candidates": judgment.get("candidates") or [],
    }

    errors = codebook_mod.validate_coding_record(record)
    if errors:
        return CodingResult(
            ok=False, record=None, reason="; ".join(errors), session=session,
        )

    return CodingResult(ok=True, record=record, session=session)


# ---------------------------------------------------------------------------
# code_digests — a batch of digests -> RunResult, with the storm threshold
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Outcome of coding a batch of digests via code_digests().

    ``records`` holds every successful (schema-valid) coding record;
    ``failures`` holds a ``(session, reason)`` pair for every digest that
    could not be coded — never a fabricated record. ``status`` is
    ``"failure"`` when the batch's failure fraction STRICTLY exceeds 0.5
    (``failed/total > 0.5``, PRD §5.3/§6.8's storm threshold); exactly 50%
    failed is NOT a storm and stays ``"ok"``. This function never
    escalates and never touches the codebook — that is epsilon/gamma's
    job; it only returns the tallied result.
    """

    status: str
    records: list
    failures: list
    total: int
    succeeded: int
    failed: int


def code_digests(
    digests,
    codebook: dict,
    *,
    project: str,
    model: str = "haiku",
    invoke=None,
) -> RunResult:
    """Code a batch of digests against one codebook.

    Calls ``code_digest`` once per digest, each wrapped in its own
    isolated try/except so a single unexpected crash (e.g. a digest whose
    frontmatter fails to parse) can't abort the rest of the batch — a
    belt-and-braces layer on top of code_digest's own never-fabricate
    contract. Successes are appended to ``records``; failures to
    ``failures`` as ``(session, reason)`` pairs (session is ``None`` when
    the crash happened before a session could even be determined).

    EACH FAILURE IS ALSO ANNOUNCED AT WARNING AS IT HAPPENS, through ONE
    append+log funnel that both failure paths converge on — the isolating
    ``except`` above and the ``not result.ok`` arm — so neither can drift
    from the other or be forgotten by a later edit.

    That WARNING is the ONLY sink some failures ever reach. A batch whose
    failure fraction does not STRICTLY exceed 0.5 (2 of 4, say) returns
    ``status="ok"``, so epsilon escalates nothing and, before this, those
    failures were invisible everywhere: not the journal, not an escalation,
    nowhere. Per-digest lines also keep 38 identical ENOENTs distinguishable
    from 38 distinct model errors — a distinction epsilon's single joined
    aggregate detail flattens.

    WARNING rather than ERROR, deliberately: one failed digest does not by
    itself fail the run. Only the storm does, and that branch's ERROR is
    emitted by ``nightly.post_escalation``. The reason is logged unbounded;
    ``_invoke_cli`` already tail-bounds the stderr it embeds.

    TWO CONSUMERS, VERY DIFFERENT VOLUMES — and everything above is the
    TRICKLE's argument. ``nightly.run_nightly`` codes exactly ONE small
    batch per night, so its worst case is a handful of lines.
    ``census.run_mining`` calls this once per MINED BATCH, in a loop that
    runs until novelty saturates or the batch source exhausts — and a storm
    batch explicitly does NOT stop mining. So under a SYSTEMIC failure (the
    ENOENT-on-``claude`` shape) a census emits one WARNING per failed digest
    per batch, bounded by nothing but the operator's ``--max-batches``. That
    output is not swallowed: ``nightly._default_census_launcher`` runs
    census.py with no ``capture_output``, so census inherits the trickle
    unit's stderr and the volume lands in the same
    ``journalctl --user -u legibility-trickle@<project>`` an operator reads.

    That volume is ACCEPTED here rather than fixed here, deliberately.
    Bounding it inside this function cannot work: the flood comes from the
    batch COUNT, which only the mining loop knows, and a per-batch cap would
    buy nothing when a batch is already only a handful of digests. The fix,
    if it ever bites, belongs to ``run_mining``, which already computes
    ``BatchStats.failed`` per batch and could surface ONE per-batch line
    naming the DISTINCT reasons — preserving the
    38-ENOENTs-vs-38-model-errors property without a line per digest. Filed
    as a follow-up out of task 4511's review (census.py is outside that
    task's lock). Do NOT instead silence this line or drop it to DEBUG: that
    restores the sub-storm blind spot above for EVERY caller, including the
    trickle, to spare a flood only one of them can produce.

    ``status`` is ``"failure"`` when ``failed/total`` STRICTLY exceeds
    0.5 — a majority-failure storm — else ``"ok"``. Never escalates,
    never writes the codebook.
    """
    records = []
    failures = []

    for digest_text in digests:
        try:
            result = code_digest(
                digest_text, codebook, project=project, model=model, invoke=invoke,
            )
        except Exception as exc:  # isolate: one crash can't abort the batch
            failure = (None, str(exc))
        else:
            if result.ok:
                records.append(result.record)
                continue
            failure = (result.session, result.reason)

        # ONE append+log site for BOTH failure paths, so they cannot drift
        # apart and a later edit cannot silence one of them.
        session, reason = failure
        logger.warning(
            "legibility coder: digest failed (session=%s): %s", session, reason,
        )
        failures.append(failure)

    total = len(digests)
    failed = len(failures)
    succeeded = len(records)
    status = "failure" if total and (failed / total) > 0.5 else "ok"

    return RunResult(
        status=status, records=records, failures=failures,
        total=total, succeeded=succeeded, failed=failed,
    )


# ---------------------------------------------------------------------------
# main(argv) — CLI: digests + codebook -> JSONL of §7.3 coding records.
# Fail-loud: a storm (code_digests status="failure") writes ZERO records and
# returns non-zero, so epsilon can escalate and skip the merge (PRD §8.6).
# ---------------------------------------------------------------------------

def _print_summary(result: RunResult, *, matched: int, candidates: int, file) -> None:
    print(
        f"coder: status={result.status} total={result.total} "
        f"succeeded={result.succeeded} failed={result.failed} "
        f"matched={matched} candidates={candidates}",
        file=file,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: code one or more digests against a codebook.

    Reads each digest file (positional args and/or every file in a
    ``--digests`` directory), loads the codebook via ``codebook.load``,
    and calls ``code_digests``. On a run-level ``"ok"`` status, writes
    every successful record as a JSONL line to ``--out`` (or stdout) and
    returns 0. On a run-level ``"failure"`` status (storm, PRD §8.6),
    writes ZERO coding records — if ``--out`` is given, it is truncated to
    empty so a stale file from a prior successful run is never left
    looking like this run's output — prints a failure summary to stderr,
    and returns 1 (fail-loud), so a driving script (epsilon) can escalate
    and skip the merge. Either way, a one-line status summary is printed
    to stderr.
    """
    parser = argparse.ArgumentParser(
        prog="coder",
        description=(
            "Haiku trickle coder: confusion digest -> strict-JSON section 7.3 "
            "coding record."
        ),
    )
    parser.add_argument(
        "digest_files", nargs="*", metavar="DIGEST",
        help="One or more confusion digest files (alpha/digest.py output)",
    )
    parser.add_argument(
        "--digests", dest="digests_dir", default=None, metavar="DIR",
        help="A directory of confusion digest files, combined with any "
        "positional DIGEST files given",
    )
    parser.add_argument(
        "--codebook", required=True, help="Path to the v2 codebook YAML file",
    )
    parser.add_argument(
        "--project", required=True,
        help="Project id stamped into each coding record's deterministic header",
    )
    parser.add_argument(
        "--model", default="haiku", help="LLM model tier (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Write coding records as JSONL to this file instead of stdout",
    )
    args = parser.parse_args(argv)

    digest_paths = [Path(p) for p in args.digest_files]
    if args.digests_dir:
        # Regular files only -- a bare iterdir() also yields subdirectories
        # and stray non-digest entries (e.g. a nested dir or a .DS_Store),
        # and read_text() on a directory raises IsADirectoryError, aborting
        # the whole run before the storm logic can even run.
        digest_paths.extend(
            sorted(p for p in Path(args.digests_dir).iterdir() if p.is_file())
        )

    if not digest_paths:
        print(
            "coder: no digest files given (positional DIGEST args or --digests DIR)",
            file=sys.stderr,
        )
        return 1

    codebook = codebook_mod.load(args.codebook)
    digests = [p.read_text(encoding="utf-8") for p in digest_paths]

    result = code_digests(digests, codebook, project=args.project, model=args.model)

    matched = sum(len(r.get("matches") or []) for r in result.records)
    candidates = sum(len(r.get("candidates") or []) for r in result.records)

    if result.status == "failure":
        print(
            f"coder: FAILURE - {result.failed}/{result.total} digests failed "
            "coding (storm threshold exceeded) -- zero coding records written",
            file=sys.stderr,
        )
        for session, reason in result.failures:
            print(f"  session={session!r}: {reason}", file=sys.stderr)
        if args.out:
            # Never leave a stale --out from a prior successful run lying
            # around on a storm: a downstream consumer that reads the file
            # instead of gating on the exit code must see this run's true
            # (empty) outcome, not a previous night's records.
            Path(args.out).write_text("", encoding="utf-8")
        _print_summary(result, matched=matched, candidates=candidates, file=sys.stderr)
        return 1

    lines = [json.dumps(record) for record in result.records]
    output = "\n".join(lines)
    if output:
        output += "\n"

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)

    _print_summary(result, matched=matched, candidates=candidates, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
