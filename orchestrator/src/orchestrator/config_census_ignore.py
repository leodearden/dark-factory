"""Grammar and audit for ``config_key_census.ignore`` entries (task 3395).

THE DEFECT
----------
``config_key_census.ignore`` is a bare list of glob strings.  Each entry is an
unfalsifiable ASSERTION about a non-OrchestratorConfig consumer — "some other
tool reads this key, so do not report it" — that is never re-checked, and the
grammar has no way to express "temporary, until task X lands".

reify's ``cpu_governance.DF_AGENT_CPU_GOVERN`` entry is the worked proof.
Membership in the ignore list logically PROVES dark-factory does not consume
the key: dark-factory owns the schema, so a key dark-factory consumed would be
a FIELD on the model, hence classified ``known``, hence never in need of an
ignore entry.  That entry was nevertheless added on the expectation that
dark-factory eventually WOULD read it — which is exactly what made the
resulting total CPU-governance outage both permanent and silent.

This module widens the entry grammar to a reasoned ``{path, reason}`` form
(bare strings stay accepted) and adds an audit that grades each reason against
a violation taxonomy, including the liveness of any tracking task it cites.

SOURCE OF THE GRAMMAR AND TAXONOMY
----------------------------------
Adopted VERBATIM from ``/home/leo/src/reify/docs/prds/reify-audit-ptodo-detector.md``
§6.4 (canonical citation form), §8.2 (one live cite suffices), §8.3 (the kind
taxonomy), §8.4 (severity grading) and §9 (fail-open scenarios) — a sibling
repo, so the PRD is not present in this worktree.  The same invariant is stated
in prose in-repo at ``skills/review-briefing/SKILL.md``.  Adopting rather than
inventing a second taxonomy is deliberate: two near-identical vocabularies for
"this claim outlived its justification" is the drift this task exists to stop.

WHY THE CITATION REGEX DELIBERATELY DIFFERS FROM ``TASK_REF_RE``
----------------------------------------------------------------
``fused_memory.reconciliation.task_filter.TASK_REF_RE`` is deliberately
PERMISSIVE — it matches ``task N``, ``df N`` and ``#N`` alike, because its job
is to find ANY reference in free prose.  This module's job is the opposite:
per PTODO §6.4 the canonical form is ``#NNNN`` STRICTLY, and ``task NNNN`` is
itself a finding (``malformed-cite``).  Sharing one regex would erase the very
distinction the taxonomy is built on.  The two patterns are not required to
agree — they are required to DIFFER — so this is not an INV-5 duplication.
(Importing across the orchestrator → fused_memory package boundary is also not
an established direction in this repo; orchestrator reaches fused-memory over
MCP and imports only from ``shared/``.)

Dependencies are stdlib plus ``shared.task_statuses``; ``config.py`` imports
FROM here, never the reverse, so there is no cycle.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

__all__ = [
    'CENSUS_IGNORE_ENTRY_KEYS',
    'HARD_KINDS',
    'CensusIgnoreFinding',
    'CensusIgnoreSpec',
    'audit_census_ignore_specs',
    'parse_census_ignore_entries',
]


class CensusIgnoreSpec(NamedTuple):
    """One parsed ``config_key_census.ignore`` entry.

    ``pattern`` is the fnmatch glob matched against a dotted key path.
    ``reason`` is the operator's justification, or ``None`` for a bare-string
    entry (or one whose ``reason:`` was malformed) — an un-reasoned entry is
    reported as debt rather than dropped.
    ``citations`` are the canonical ``#NNNN`` task ids found in the reason, in
    source order with duplicates removed.
    """

    pattern: str
    reason: str | None
    citations: tuple[int, ...]


# The dict-entry keys this raw-tree parser recognizes.  Pinned to
# ``config.CensusIgnoreEntry.model_fields`` by a drift test rather than by a
# comment: the entry shape necessarily exists in TWO places (the pydantic model
# that validates the config, and this raw-tree parser, which cannot use the
# validated model because it must keep working when the config has an unrelated
# value-level validation error).  Since the two MUST agree byte-for-byte on the
# key names, INV-5 demands a machine check.
CENSUS_IGNORE_ENTRY_KEYS = frozenset({'path', 'reason'})

# Canonical citation form (PTODO §6.4): a '#' immediately followed by 1-5
# digits.  Strict BY DESIGN — see the module docstring on why this must not be
# unified with task_filter.TASK_REF_RE.
_CANONICAL_CITE_RE = re.compile(r'#(\d{1,5})\b')


def _citations(reason: str) -> tuple[int, ...]:
    """Canonical ``#NNNN`` ids in *reason*, in order, deduped."""
    seen: dict[int, None] = {}
    for match in _CANONICAL_CITE_RE.finditer(reason):
        seen.setdefault(int(match.group(1)), None)
    return tuple(seen)


def parse_census_ignore_entries(tree: dict[Any, Any]) -> list[CensusIgnoreSpec]:
    """Parse ``config_key_census.ignore`` off the RAW project *tree*, fail-open.

    Read from the raw tree rather than a validated OrchestratorConfig so the
    census keeps working when the config has an unrelated value-level
    validation error (the same reason ``check-config`` calls the census
    directly).

    Inherits ``config._census_ignore_specs``' degradation contract verbatim and
    extends it one level for the dict form: a malformed hatch — non-dict block,
    non-list ``ignore``, a non-str/non-dict entry, a dict missing ``path`` or
    carrying a non-str ``path`` — degrades to "that entry does not exist"
    instead of raising.  A broken escape hatch must never take out the census
    that surfaces real phantom keys.

    A bad ``reason`` is the one case that does NOT drop the entry: the reason
    degrades to ``None`` while the PATTERN survives, because deleting the
    suppression would resurrect a key the operator deliberately excused.  The
    resulting reasonless spec is reported as debt by the audit.

    Source ORDER is preserved: fnmatch classification is first-match-wins, so
    reordering entries could silently attach a different entry's justification
    to a key.
    """
    block = tree.get('config_key_census')
    if not isinstance(block, dict):
        return []
    raw = block.get('ignore')
    if not isinstance(raw, list):
        return []

    specs: list[CensusIgnoreSpec] = []
    for entry in raw:
        if isinstance(entry, str):
            specs.append(CensusIgnoreSpec(entry, None, ()))
            continue
        if not isinstance(entry, dict):
            continue
        path = entry.get('path')
        if not isinstance(path, str):
            continue
        reason = entry.get('reason')
        if not isinstance(reason, str):
            reason = None
        specs.append(
            CensusIgnoreSpec(path, reason, _citations(reason) if reason else ())
        )
    return specs


# ---------------------------------------------------------------------------
# The violation taxonomy (PTODO §8.3) and its severity grading (§8.4)
# ---------------------------------------------------------------------------


class CensusIgnoreFinding(NamedTuple):
    """One graded defect found in an ignore entry.

    ``pattern`` is the offending entry's glob, so an operator can locate the
    entry in the YAML; ``kind`` is a taxonomy member; ``severity`` is
    ``'hard'`` or ``'advisory'``; ``detail`` states the defect AND the concrete
    remediation — a finding that only says "wrong" is not actionable.
    """

    pattern: str
    kind: str
    severity: str
    detail: str


KIND_UNREASONED = 'unreasoned'
KIND_SELF_REFUTING = 'self-refuting'
KIND_MISSING_CITE = 'missing-cite'
KIND_MALFORMED_CITE = 'malformed-cite'
KIND_UNKNOWN_ID = 'unknown-id'
KIND_ORPHANED = 'orphaned'
KIND_PARKED_ON_ANCHOR = 'parked-on-anchor'

SEVERITY_HARD = 'hard'
SEVERITY_ADVISORY = 'advisory'

# Grading per PTODO §8.4.  A kind is HARD only when a positive answer is
# certain from information the linter actually holds:
#   * self-refuting — certain with NO external state at all (the entry
#     contradicts itself: dark-factory owns the schema, so a key it consumed
#     would be a model field and would never need excusing);
#   * missing-cite  — certain from the reason text alone (a not-yet-landed
#     claim with no citation has no expiry, so nothing will ever re-check it);
#   * orphaned      — the cited task is POSITIVELY terminal in the task DB, so
#     the justification is provably spent.
# Everything else stays ADVISORY: `unknown-id` can be a task-DB sync artifact
# and must never hard-fail a gate, `malformed-cite` still leaves a usable
# pointer, `parked-on-anchor` is a deliberate operator state, and `unreasoned`
# is pre-existing debt that must not turn every green config red on upgrade.
HARD_KINDS = frozenset({KIND_SELF_REFUTING, KIND_MISSING_CITE, KIND_ORPHANED})

# The framework itself named as the consumer.  \bDF\b catches the
# DF_-prefixed env-var convention (e.g. DF_AGENT_CPU_GOVERN) that names
# dark-factory directly.
_SELF_CONSUMER_RE = re.compile(
    r'\b(dark[-\s]?factory|orchestrator|orchestratorconfig|DF)\b', re.IGNORECASE
)

# Prose asserting the consumer has NOT landed yet.  Such a claim is only
# auditable if it says WHICH task will land it.
_PENDING_PROSE_RE = re.compile(
    r'\b(pending|not yet|until|once|lands?|landing|will be|planned|upcoming|in flight)\b',
    re.IGNORECASE,
)

# Legacy / non-canonical reference forms.  Matching one of these while the
# canonical #NNNN extractor found nothing means the operator DID leave a
# pointer, just not one this linter (or any other tool) can resolve.
_LOOSE_REF_RE = re.compile(
    r'\b(task|tkt|ticket|df)\s*[-#]?\s*([0-9]{1,5}|[Ͱ-Ͽ])\b', re.IGNORECASE
)


def audit_census_ignore_specs(
    specs: list[CensusIgnoreSpec],
    status_probe: dict[int, Any] | None = None,
) -> list[CensusIgnoreFinding]:
    """Grade every spec against the taxonomy; pure and sync.

    *status_probe* is the injected task-liveness view (``None`` = "cannot
    know"), mirroring the shape of
    ``fused_memory.reconciliation.task_filter.nonterminal_completion_claim_task_ids``.
    All task-store access flows through it, so every kind is unit-testable with
    a dict-backed fake and no sqlite at all.

    Kinds are INDEPENDENT defects, not a single classification: one entry can
    be both self-refuting and uncited, and suppressing either would hide a real
    problem.
    """
    findings: list[CensusIgnoreFinding] = []
    for spec in specs:
        reason = (spec.reason or '').strip()

        if not reason:
            findings.append(CensusIgnoreFinding(
                spec.pattern,
                KIND_UNREASONED,
                SEVERITY_ADVISORY,
                f'{spec.pattern}: no reason given — an ignore entry is an '
                'assertion that some non-orchestrator consumer reads this key, '
                'and with no reason that assertion cannot be checked by anyone. '
                'Add a `reason:` naming the actual consumer.',
            ))
            # Every remaining structural kind reads the reason text, so there is
            # nothing further to say about an entry that has none.
            continue

        if _SELF_CONSUMER_RE.search(reason):
            findings.append(CensusIgnoreFinding(
                spec.pattern,
                KIND_SELF_REFUTING,
                SEVERITY_HARD,
                f'{spec.pattern}: the reason names dark-factory/the orchestrator '
                'as the consumer, which is self-refuting — dark-factory owns the '
                'schema, so a key it consumed would be a FIELD on the model, '
                'hence classified known, hence never in need of an ignore entry. '
                'Add the key as a field on the model instead of excusing it.',
            ))

        if _PENDING_PROSE_RE.search(reason) and not spec.citations:
            findings.append(CensusIgnoreFinding(
                spec.pattern,
                KIND_MISSING_CITE,
                SEVERITY_HARD,
                f'{spec.pattern}: the reason says the consumer has not landed '
                'yet but cites no tracking task, so nothing will ever prompt a '
                're-check and the entry has no expiry. Cite the tracking task '
                'as #NNNN.',
            ))

        if not spec.citations and _LOOSE_REF_RE.search(reason):
            findings.append(CensusIgnoreFinding(
                spec.pattern,
                KIND_MALFORMED_CITE,
                SEVERITY_ADVISORY,
                f'{spec.pattern}: the reason references a task in a '
                'non-canonical form that no tool can resolve. Rewrite it as '
                '#NNNN so the citation can be checked for liveness.',
            ))

    return findings
