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
stores, cap-retry, transcript watchdogs). The real LLM call lives behind
exactly one swappable seam, the module-level ``_invoke_cli``, which every
public function accepts as an ``invoke`` override — the LLM is ALWAYS
mocked in this module's own test suite; no test ever shells out to a real
model.

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
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import yaml

import codebook as codebook_mod


class CoderParseError(Exception):
    """Raised when the coder cannot parse a digest's frontmatter or the
    LLM's raw output into a usable structure. Never silently defaulted —
    callers must treat this as a hard per-digest failure (never-fabricate
    contract)."""


class CoderInvocationError(Exception):
    """Raised when the ``claude -p --model`` subprocess invocation fails —
    non-zero exit or a timeout. Carries a stderr tail for diagnosis. Never
    silently swallowed: code_digest turns this into a per-digest failure,
    never a fabricated record."""


@dataclass
class CodingResult:
    """Outcome of coding one digest.

    ``ok=True`` means ``record`` is a schema-valid §7.3 coding record —
    including a legitimately empty one (``matches=[]``, ``candidates=[]``
    is a genuine finding, not a failure). ``ok=False`` means ``record`` is
    None and ``reason`` explains why: a CLI invocation error, unparseable
    LLM output, or a schema-invalid assembled record are never partially
    applied and never fabricated into a record.
    """

    ok: bool
    record: dict | None
    reason: str | None = None
    session: str | None = None


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

    Never-fabricate contract: an invocation error, unparseable LLM output,
    or a schema-invalid assembled record comes back as ``ok=False`` with
    ``record=None`` and ``reason`` set — never partially applied, never
    fabricated. A legitimately empty judgment
    (``{"matches": [], "candidates": []}``) that passes schema validation
    is a genuine ``ok=True`` success: "coded fine, found nothing" is never
    conflated with "coding failed" (codebook lesson
    one-shot-subagent-contract).
    """
    meta = parse_frontmatter(digest_text)
    session = meta.get("session")

    index = build_codebook_index(codebook)
    prompt = build_prompt(digest_text, index)
    invoke_fn = invoke or _invoke_cli

    try:
        raw = invoke_fn(prompt, model)
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
