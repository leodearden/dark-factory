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
    'CensusIgnoreSpec',
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
