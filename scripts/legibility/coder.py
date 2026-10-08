#!/usr/bin/env python3
"""scripts/legibility/coder.py — Haiku trickle coder: confusion digest ->
strict-JSON §7.3 coding record.

Task delta of the confusion-reduction PRD (plans/confusion-reduction-prd.md
§5.3, contract §7.3, boundary tests §8.1 consumer side + §8.6). Reads ONE
confusion digest (alpha/digest.py output), builds a COMPACT codebook index
(entry ids + titles + one-line causes — NOT the full YAML) from the v2
codebook (gamma/codebook.py), makes ONE model call (default haiku), parses
the model's strict-JSON judgment, and assembles it into a deterministic-header
coding record that codebook.validate_coding_record schema-gates.

The model call crosses exactly one seam, ``invoke``: a ``(prompt, model) ->
str`` callable that every public function REQUIRES, with no default. The
production one is ``session_runner.SessionRunner.invoker(TRICKLE_CODER_STAGE)``
-- the orchestrator's own session runner over the fleet's account pool, never
a spawn of this module's own (Leo's 2026-09-29 ruling, task 6042). Rotation,
cap detection and credentials are all the runner's; what reaches this module
is a reply, a ``session_runner.NoHeadroom`` (no pool account could take the
call) or an ``InvocationFailed`` (any other call without a reply). No test
reaches a real model: they inject an ``invoke``, or run ``main`` over a fake
CLI.

What this module still takes from ``shared`` directly is one pure function,
``cap_markers.looks_like_blocking_banner``: the loose OR-substring DEFER GATE
``census.preflight_headroom`` also uses, applied on one path only -- a reply
that came back as a SUCCESS yet could not be parsed into a verdict. The runner
puts such a reply to the gate's STRICT detector first
(``TRICKLE_CODER_STAGE.is_usable_reply``), which rotates on a recognised cap;
what still arrives unparseable is a banner the strict detector does not know,
such as an auth banner. A loose false positive there can only re-label a
digest that was failing anyway.

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
import re
import sys
from collections.abc import Sequence
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
# too. orchestrator/src is bound the same way, for the code-quality slicer
# `orchestrator.agents.code_quality` (stdlib-only): under
# `uv run --project shared` nothing else puts it on sys.path.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))
_ORCH_SRC = Path(__file__).resolve().parents[2] / "orchestrator" / "src"
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

# Self-bootstrap for a standalone `python scripts/legibility/coder.py` run --
# must precede the `legibility.*` import below, since a direct script
# invocation puts only scripts/legibility/ (not scripts/) on sys.path. Skipped
# under pytest/package import. Mirrors census.py/nightly.py's identical guard.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codebook as codebook_mod  # noqa: E402
import yaml  # noqa: E402
from legibility import invariants, session_runner  # noqa: E402
from orchestrator.agents import code_quality  # noqa: E402
from shared.cap_markers import looks_like_blocking_banner  # noqa: E402

logger = logging.getLogger("legibility.coder")


class CoderParseError(Exception):
    """Raised when the coder cannot parse a digest's frontmatter or the
    LLM's raw output into a usable structure. Never silently defaulted —
    callers must treat this as a hard per-digest failure (never-fabricate
    contract)."""


@dataclass
class CodingResult:
    """Outcome of coding one digest.

    ``ok=True`` means ``record`` is a schema-valid §7.3 coding record —
    including a legitimately empty one (``matches=[]``, ``candidates=[]``
    is a genuine finding, not a failure). ``ok=False`` means ``record`` is
    None and ``reason`` explains why: a CLI invocation error, unparseable
    LLM output, or a schema-invalid assembled record are never partially
    applied and never fabricated into a record.

    ``capped=True`` means no pool account had headroom to take this digest
    (``session_runner.NoHeadroom``), or the reply was a capacity/auth banner
    rather than a verdict. It is a strict REFINEMENT of ``ok=False``,
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

_DEFINITION_HEADING = "## Definition"


def _quality_definition() -> str:
    """The normative code-quality doc's ``## Definition`` body, read per call
    so a renamed heading fails one digest rather than the import."""
    doc_text = code_quality.NORMATIVE_DOC.read_text(encoding="utf-8")
    return code_quality.section(doc_text, _DEFINITION_HEADING).strip()


def _invariant_slugs_block(invariant_slugs: Sequence[str]) -> str:
    if not invariant_slugs:
        return (
            "This project declares no invariant slugs in "
            f"{invariants.DOC_RELPATH}, so invariant_violated must be null."
        )
    return (
        "invariant_violated must be one of these slugs from this project's "
        f"{invariants.DOC_RELPATH}, or null:\n"
        + "\n".join(f"- {slug}" for slug in invariant_slugs)
    )


def build_prompt(
    digest_text: str, codebook_index: str, *, invariant_slugs: Sequence[str]
) -> str:
    """Compose the full prompt handed to the trickle coder LLM.

    Embeds *codebook_index* and *digest_text* verbatim (pure data
    plumbing — a broken coder would drop one), the legal phase vocabulary
    (codebook.PHASES, including "unknown"), the required strict-JSON
    output shape (PRD §7.3), the code-quality Definition as the test a
    confusion must meet to be minted as a candidate, and the observed
    project's *invariant_slugs* (an empty list is stated, not omitted).
    The slugs are guidance only: the merger is what enforces them.
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
        "=== QUALITY DEFINITION ===\n"
        "Mint a candidate only for a confusion that meets this definition, "
        "that is, one that shows a cost or risk to the next change. "
        "Otherwise do not mint one.\n"
        + _quality_definition() + "\n\n"
        "=== INVARIANT SLUGS ===\n" + _invariant_slugs_block(invariant_slugs) + "\n\n"
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
# TRICKLE_CODER_STAGE — how one trickle digest calls the model
# ---------------------------------------------------------------------------

def reply_parses_as_judgment(reply: str) -> bool:
    """Could ``code_digest`` parse *reply* into a judgment at all?"""
    try:
        parse_coder_output(reply)
    except CoderParseError:
        return False
    return True


TRICKLE_CODER_STAGE = session_runner.StageSpec(
    name="trickle-coder",
    cwd=None,
    timeout_secs=120,
    max_turns=2,
    max_budget_usd=1.0,
    tools=session_runner.CLASSIFIER,
    is_usable_reply=reply_parses_as_judgment,
)
"""A pure classifier: the codebook index and the digest are both in the
prompt, so it needs no tools and no project cwd. 120s, sized for one Haiku
coding call, is the binding bound; turns and budget are ceilings.

A successful reply that does not parse is put to the gate's STRICT cap
detector, so a banner delivered at exit 0 caps that account and the same
digest is retried on the next (task 5637), while a parsed verdict quoting a
banner is never offered. Every real banner fails to parse
(``scripts/tests/test_legibility_coder.py::test_the_trickle_stage_offers_every_real_banner_to_the_pool``).

Known residual, accepted since task 5637: a verdict TRUNCATED mid-JSON after
a cap-quoting ``evidence_quote`` is unparseable AND strictly a cap, so it caps
a healthy account for the night."""


# ---------------------------------------------------------------------------
# code_digest — one digest -> one CodingResult
# ---------------------------------------------------------------------------

def code_digest(
    digest_text: str,
    codebook: dict,
    *,
    project: str,
    model: str = "haiku",
    invariant_slugs: Sequence[str],
    invoke,
) -> CodingResult:
    """Code one digest against one codebook.

    Flow: parse the digest's frontmatter -> build the compact codebook
    index -> build the prompt -> make the model call through *invoke* ->
    parse its strict-JSON judgment ->
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
    no account had headroom left to look (see ``session_runner.NoHeadroom``). A
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
    prompt = build_prompt(digest_text, index, invariant_slugs=invariant_slugs)

    try:
        raw = invoke(prompt, model)
    except session_runner.NoHeadroom as exc:
        # ORDERED ABOVE the generic arm below, and that ordering is
        # load-bearing: NoHeadroom SUBCLASSES InvocationFailed, so
        # reversing these two silently routes every cap into the generic arm
        # and the label is never applied.
        return CodingResult(
            ok=False, record=None, reason=str(exc), session=session, capped=True,
        )
    except session_runner.InvocationFailed as exc:
        return CodingResult(ok=False, record=None, reason=str(exc), session=session)

    try:
        judgment = parse_coder_output(raw)
    except CoderParseError as exc:
        # The coder's one cap scan, and its placement INSIDE this arm is
        # load-bearing. The CLI does not always FAIL when it declines to
        # answer -- it can deliver the banner as an ordinary successful reply.
        # The session runner offers an unparseable reply to the gate's STRICT
        # detector, which rotates on a cap it recognises; a banner it does not
        # recognise (an auth banner, a new wording) still arrives here as
        # prose that could not be parsed into a verdict, and this is what
        # labels it capped.
        #
        # Scanning only AFTER the parse has failed is census's
        # split-on-parse-success rule, adopted unchanged (see
        # ``census._build_default_verify_fn``'s docstring, which records the
        # live defect): scanning arbitrary model output with the loose marker
        # list aborted the census on cap-THEMED clusters, because this repo's
        # codebook is dominated by clusters ABOUT usage and weekly limits, so
        # the markers match ordinary HEALTHY content. The coder is exposed
        # identically -- a judgment's model-authored
        # ``title``/``cause``/``evidence_quote`` legitimately quote capped
        # sessions, since "an agent stalled on a usage limit" is a real
        # confusion worth coding. A pre-parse scan would discard exactly those
        # findings, quietly, and label the loss a deferral. A reply that
        # PARSES into a verdict is a verdict.
        #
        # ONE deliberate divergence from census: no confirmation probe. Census
        # needs one because its scan can strike a reply it would otherwise
        # have ACCEPTED, so a false positive there destroys a good verdict.
        # This scan sits on an already-failed path, where the alternative
        # disposition is already "per-digest failure" -- so a false positive
        # can only re-LABEL a digest that was failing anyway, never launder a
        # genuine verdict into a defer. It also never costs an account: the
        # label stays on this digest, and the account is not reported to the
        # pool.
        marker = looks_like_blocking_banner(raw)
        if marker:
            return CodingResult(
                ok=False, record=None, session=session, capped=True,
                reason=(
                    f"CLI reply is a capacity/auth banner, not a verdict "
                    f"(marker: {marker!r}); {exc}"
                ),
            )
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

    ``capped`` counts how many of those failures were a usage/auth CAP
    rather than a failure of the coding — digests the CLI never actually
    looked at. It REFINES ``failed`` (a capped digest is counted in both),
    and it is deliberately a COUNT plus the ``is_cap_deferral`` predicate
    rather than a third ``status`` value: ``census.py`` computes
    ``saturated = dup_rate >= config.dup_rate and run_result.status !=
    "failure"`` and selects storm batches with ``s.status == "failure"``, so
    a new status value would silently make a capped mining batch count as
    saturated and stop the census early.

    TAINT-AND-EXCLUDE for a sub-storm capped run: a capped digest is
    labelled and left out of ``records``, but the batch's genuinely coded
    records still merge. 2 capped of 20 stays ``status="ok"`` and returns
    all 18 real records — the same contract ``evals/runner.py`` uses when it
    excludes a ``cap_exhausted:`` cell from a reported mean instead of
    scoring it 0.0. Discarding 18 records that cost real tokens because two
    digests found no headroom would be its own kind of fabrication.
    """

    status: str
    records: list
    failures: list
    total: int
    succeeded: int
    failed: int
    capped: int = 0


def code_digests(
    digests,
    codebook: dict,
    *,
    project: str,
    model: str = "haiku",
    invariant_slugs: Sequence[str],
    invoke,
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
    ``session_runner`` already tail-bounds the output streams it embeds.

    TWO CONSUMERS, VERY DIFFERENT VOLUMES — and everything above is the
    TRICKLE's argument. ``nightly.run_nightly`` codes exactly ONE small
    batch per night, so its worst case is a handful of lines.
    ``census.mine_to_saturation`` calls this once per MINED BATCH, in a loop that
    runs until novelty saturates or the batch source exhausts — and a storm
    batch explicitly does NOT stop mining. So under a SYSTEMIC failure (the
    ENOENT-on-``claude`` shape) a census emits one WARNING per failed digest
    per batch, bounded by nothing but the operator's ``--max-batches``. That
    output is not swallowed: ``nightly._default_census_launcher`` runs
    census.py with no ``capture_output``, so census inherits the trickle
    unit's stderr and the volume lands in the same
    ``journalctl --user -u legibility-trickle@<project>`` an operator reads.

    That volume is bounded CALLER-SIDE rather than here, deliberately.
    Bounding it inside this function cannot work: the flood comes from the
    batch COUNT, which only the mining loop knows, and a per-batch cap would
    buy nothing when a batch is already only a handful of digests. The bound
    lives in ``scripts/legibility/census.py::mine_to_saturation`` (see its
    ``_bounded_coder_warnings``). Do NOT silence this line or drop it to
    DEBUG: that restores the sub-storm blind spot above for EVERY caller,
    including the trickle, to spare a flood only one of them can produce.

    ``status`` is ``"failure"`` when ``failed/total`` STRICTLY exceeds
    0.5 — a majority-failure storm — else ``"ok"``. Never escalates,
    never writes the codebook.
    """
    records = []
    failures = []
    capped = 0

    for digest_text in digests:
        try:
            result = code_digest(
                digest_text, codebook, project=project, model=model,
                invariant_slugs=invariant_slugs, invoke=invoke,
            )
        except Exception as exc:  # isolate: one crash can't abort the batch
            # An unexpected crash is never a cap: the cap paths are typed and
            # return a CodingResult, they do not escape as bare exceptions.
            failure, was_capped = (None, str(exc)), False
        else:
            if result.ok:
                records.append(result.record)
                continue
            failure, was_capped = (result.session, result.reason), result.capped

        # ONE append+log site for BOTH failure paths, so they cannot drift
        # apart and a later edit cannot silence one of them. The cap tally is
        # threaded THROUGH this funnel rather than counted at a second site,
        # for the same reason: two sites are two things to forget.
        session, reason = failure
        logger.warning(
            "legibility coder: digest failed (session=%s): %s", session, reason,
        )
        failures.append(failure)
        if was_capped:
            capped += 1

    total = len(digests)
    failed = len(failures)
    succeeded = len(records)
    status = "failure" if total and (failed / total) > 0.5 else "ok"

    return RunResult(
        status=status, records=records, failures=failures,
        total=total, succeeded=succeeded, failed=failed, capped=capped,
    )


def is_cap_deferral(result: RunResult) -> bool:
    """True when a run-level FAILURE is really a capped night — a DEFERRAL
    rather than a coder failure.

    An all-accounts-capped night is a NORMAL operating condition (Leo's
    standing directive; sibling task 4503), not an incident. The coder must
    not fabricate a verdict for a digest it never got to look at, and must
    not present the resulting empty night as an infra failure. Before this
    existed, 2026-08-24 came back as 17 of 20 hard per-digest failures,
    tripped the >50% storm threshold, and became ``exit_code=1`` plus an
    ERROR-level escalation — an operator paged for expected weather.

    The majority rule (``capped * 2 > failed``) deliberately reuses the storm
    threshold's own strictly-greater-than-half shape. A genuine coder
    regression that merely COINCIDES with a cap or two still reads as a
    storm, still exits non-zero, and still gets looked at: the deferral
    branch must not become a place for real bugs to hide.

    A PREDICATE over ``RunResult``, deliberately NOT a third ``status``
    value. ``census.py`` computes ``saturated = dup_rate >= config.dup_rate
    and run_result.status != "failure"`` and selects storm batches with
    ``s.status == "failure"``; adding a status value would silently make a
    capped mining batch count as saturated and stop the census early. One
    policy, one home, two callers (``main`` here and ``nightly.run_nightly``).

    Note the ``status == "failure"`` guard: a sub-storm run with a minority of
    caps is NOT a deferral. Those runs coded most of their digests and their
    records still merge (see ``RunResult``'s taint-and-exclude contract).
    """
    return result.status == "failure" and result.capped * 2 > result.failed


# ---------------------------------------------------------------------------
# main(argv) — CLI: digests + codebook -> JSONL of §7.3 coding records.
# Fail-loud: a storm (code_digests status="failure") writes ZERO records and
# returns non-zero, so epsilon can escalate and skip the merge (PRD §8.6).
# ---------------------------------------------------------------------------

def _print_summary(result: RunResult, *, matched: int, candidates: int, file) -> None:
    print(
        f"coder: status={result.status} total={result.total} "
        f"succeeded={result.succeeded} failed={result.failed} "
        f"capped={result.capped} "
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

    THIRD DISPOSITION — the DEFERRAL (task 4736). When that run-level
    failure is really a capped night (``is_cap_deferral``), the same
    zero-records/truncate discipline applies but the exit code is 0 and the
    banner says ``coder: DEFERRED`` rather than ``coder: FAILURE``. An
    all-accounts-capped night is a normal operating condition, not an
    incident (Leo's directive; sibling task 4503), and a non-zero exit here
    made ``check_trickle_liveness.sh`` report a failure for a timer behaving
    exactly as designed — on 2026-08-24 that became an ERROR-level
    escalation. The shape deliberately mirrors ``census.main``, which prints
    ``census: deferred -- stage=... -- <reason>`` and returns 0 for its own
    headroom defer, so an operator reads the same thing from both legibility
    CLIs. The one-line summary still reports ``status=failure`` with the
    true counts: the exit code changes, the tally never lies.

    Every digest is coded on the fleet's shared account pool, through one
    ``session_runner`` opened for the run and closed when coding ends.
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
        "--project-root", required=True,
        help="The observed project's root, whose design-invariants doc "
        "declares the invariant slugs the prompt lists",
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
    invariant_slugs = invariants.read_slugs(args.project_root)

    with session_runner.open_pooled_runner(label="legibility-coder-cli") as runner:
        result = code_digests(
            digests, codebook, project=args.project, model=args.model,
            invariant_slugs=invariant_slugs,
            invoke=runner.invoker(TRICKLE_CODER_STAGE),
        )

    matched = sum(len(r.get("matches") or []) for r in result.records)
    candidates = sum(len(r.get("candidates") or []) for r in result.records)

    if result.status == "failure":
        deferred = is_cap_deferral(result)
        if deferred:
            print(
                f"coder: DEFERRED - {result.capped}/{result.total} digests hit "
                "a usage/auth cap (no headroom) -- zero coding records "
                "written, nothing fabricated",
                file=sys.stderr,
            )
        else:
            print(
                f"coder: FAILURE - {result.failed}/{result.total} digests failed "
                "coding (storm threshold exceeded) -- zero coding records written",
                file=sys.stderr,
            )
        # The same per-session loop for BOTH arms, so the cap banner the CLI
        # actually printed reaches the operator instead of being summarised
        # away -- the whole point of carrying it this far.
        for session, reason in result.failures:
            print(f"  session={session!r}: {reason}", file=sys.stderr)
        if args.out:
            # Never leave a stale --out from a prior successful run lying
            # around on a storm OR a deferral: a downstream consumer that
            # reads the file instead of gating on the exit code must see this
            # run's true (empty) outcome, not a previous night's records.
            Path(args.out).write_text("", encoding="utf-8")
        # The summary stays honest either way -- status=failure with the true
        # counts. Exit 0 must not launder the tally: "deferred, nothing coded"
        # being readable as "coded fine, found nothing" is the never-fabricate
        # conflation again, at the run level.
        _print_summary(result, matched=matched, candidates=candidates, file=sys.stderr)
        return 0 if deferred else 1

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
