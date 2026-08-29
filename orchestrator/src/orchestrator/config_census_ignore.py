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

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, NamedTuple

import yaml
from shared.task_statuses import TERMINAL

logger = logging.getLogger(__name__)

__all__ = [
    'CENSUS_IGNORE_ENTRY_KEYS',
    'HARD_KINDS',
    'CensusIgnoreFinding',
    'CensusIgnoreSpec',
    'TaskCiteStatus',
    'audit_census_ignore_entries',
    'audit_census_ignore_specs',
    'parse_census_ignore_entries',
    'read_task_statuses',
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
    status_probe: dict[int, TaskCiteStatus] | None = None,
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

        # Liveness fires ONLY on a positively-terminal / positively-parked
        # answer.  A None probe (task store absent or unreadable) or an id the
        # probe has never heard of can therefore only ever UNDER-fire — the
        # audit is loudest where it knows most, never where it knows least.
        if status_probe is not None:
            findings.extend(_liveness_findings(spec, status_probe))

    return findings


# ---------------------------------------------------------------------------
# Citation liveness: the ONLY I/O in this module
# ---------------------------------------------------------------------------


class TaskCiteStatus(NamedTuple):
    """The two facts the audit needs about a cited task.

    ``status`` is the raw DB string (compared against ``shared.task_statuses``'
    canonical vocabulary rather than a fourth hardcoded ``{'done','cancelled'}``
    copy).  ``do_not_complete`` is the deliberate-park marker read off the
    task's ``metadata`` blob.
    """

    status: str
    do_not_complete: bool


# Per-process dedup for probe-failure breadcrumbs, mirroring
# shared.safe_io._warned_corrupt_paths: a restart re-enables the warning.
_warned_probe_paths: set[str] = set()


def _warn_once(path: str, message: str, *args: Any) -> None:
    """Emit *message* at WARNING at most once per *path* per process."""
    if path in _warned_probe_paths:
        return
    _warned_probe_paths.add(path)
    logger.warning(message, *args)


def tasks_db_path(project_root: Path | str) -> Path:
    """Conventional task-store location for *project_root*."""
    return Path(project_root) / '.taskmaster' / 'tasks' / 'tasks.db'


def read_task_statuses(project_root: Path | str) -> dict[int, TaskCiteStatus] | None:
    """Read ``{id: TaskCiteStatus}`` for the ``master`` tag, or ``None``.

    ``None`` means "cannot know" — NOT "clean".  ``check-config`` is an offline
    operator tool routinely pointed at another project's YAML from a machine
    that does not have that project's ``.taskmaster/tasks/tasks.db`` at all, so
    absence must never be allowed to manufacture findings.  Every failure path
    leaves a breadcrumb naming the path it looked for, so the degradation is
    visible rather than silent (INV-4).

    Opened strictly read-only via the ``mode=ro`` URI, mirroring
    ``sandbox_soak._connect_ro`` and ``b3_gate`` — a predicate must never mutate
    the store it measures.  Deliberately NOT a call into
    ``sandbox_soak.read_task_status``: that one returns status only (we also
    need ``metadata`` for ``do_not_complete``) and RAISES on a missing store,
    whereas this caller must fail open.

    Failure handling uses NARROW typed handlers plus a logged breadcrumb, never
    a broad ``except Exception: return None`` — the latter is what
    ``shared/tests/test_silent_fallthrough_gate.py`` exists to ratchet against.
    """
    db = tasks_db_path(project_root)
    if not db.exists():
        logger.debug('census-ignore audit: no task store at %s', db)
        return None

    try:
        conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    except sqlite3.Error as exc:
        _warn_once(
            str(db),
            'census-ignore audit: cannot open task store %s (%s) — citation '
            'liveness checks are SKIPPED for this config',
            db, exc,
        )
        return None

    try:
        rows = conn.execute(
            'SELECT id, status, metadata FROM tasks WHERE tag = ?', ('master',)
        ).fetchall()
    except sqlite3.Error as exc:
        _warn_once(
            str(db),
            'census-ignore audit: cannot query task store %s (%s) — citation '
            'liveness checks are SKIPPED for this config',
            db, exc,
        )
        return None
    finally:
        conn.close()

    statuses: dict[int, TaskCiteStatus] = {}
    for task_id, status, metadata in rows:
        try:
            key = int(task_id)
        except (TypeError, ValueError):
            continue
        statuses[key] = TaskCiteStatus(
            str(status or ''), _do_not_complete(metadata, db)
        )
    return statuses


def _do_not_complete(metadata: Any, db: Path) -> bool:
    """Read ``metadata.do_not_complete``, defaulting to False.

    Keys on ``do_not_complete`` SPECIFICALLY — not on a bare ``deferred``
    status (an ordinary non-terminal state) and not on ``do_not_dispatch`` (a
    scheduler knob that says nothing about whether the task will ever
    complete).  Both are documented false-positive guards.
    """
    if not isinstance(metadata, str) or not metadata:
        return False
    try:
        parsed = json.loads(metadata)
    except json.JSONDecodeError as exc:
        _warn_once(
            f'{db}:metadata',
            'census-ignore audit: corrupt task metadata JSON in %s (%s) — '
            'treating do_not_complete as unset',
            db, exc,
        )
        return False
    return isinstance(parsed, dict) and parsed.get('do_not_complete') is True


def _liveness_findings(
    spec: CensusIgnoreSpec, status_probe: dict[int, TaskCiteStatus]
) -> list[CensusIgnoreFinding]:
    """Grade *spec*'s citations against the probe (positively-terminal only)."""
    findings: list[CensusIgnoreFinding] = []
    known = {cid: status_probe[cid] for cid in spec.citations if cid in status_probe}

    for cid in spec.citations:
        if cid not in status_probe:
            findings.append(CensusIgnoreFinding(
                spec.pattern,
                KIND_UNKNOWN_ID,
                SEVERITY_ADVISORY,
                f'{spec.pattern}: cited task #{cid} is not present in the task '
                'store. Advisory only — most often a task-DB sync artifact '
                'rather than a real defect. Confirm the id is right.',
            ))

    # PTODO §8.2: one live cite suffices. Only report the entry as orphaned
    # when EVERY id the probe actually knows about is terminal — an entry
    # tracked by any still-open task is still tracked.
    terminal = {
        cid: st.status for cid, st in known.items() if st.status in TERMINAL
    }
    if known and len(terminal) == len(known):
        detail = ', '.join(f'#{cid} ({status})' for cid, status in terminal.items())
        findings.append(CensusIgnoreFinding(
            spec.pattern,
            KIND_ORPHANED,
            SEVERITY_HARD,
            f'{spec.pattern}: every cited task has closed ({detail}), so this '
            "entry's justification is spent — either the consumer landed (make "
            'the key a field on the model) or it never will (delete the entry).',
        ))
        return findings

    for cid, st in known.items():
        if st.do_not_complete:
            findings.append(CensusIgnoreFinding(
                spec.pattern,
                KIND_PARKED_ON_ANCHOR,
                SEVERITY_ADVISORY,
                f'{spec.pattern}: cited task #{cid} is parked '
                '(metadata.do_not_complete), so it will not close on its own '
                'and this entry has no realistic expiry date.',
            ))
    return findings


def audit_census_ignore_entries(config_path: Path | str) -> list[CensusIgnoreFinding]:
    """Audit the ``config_key_census.ignore`` entries of the YAML at *config_path*.

    The operator-facing entry point: parse the entries off the RAW tree, resolve
    the project's task store from that same raw tree, and return the combined
    structural + liveness findings.

    Reads the raw tree rather than a validated ``OrchestratorConfig`` for the
    same reason ``census_config_keys`` does — a lint must still report entry
    debt on a config that currently fails validation for an unrelated
    value-level reason.  This is a deliberate SECOND read of the YAML rather
    than a threading of the parsed tree out of the census: keeping
    ``census_config_keys(config_path)`` a pure function of the file is what
    lets the born-at-L2 stay keyed on the config alone (the L2-decoupling
    decision), so the census signature can never shift because a task's status
    changed.

    Fail-open throughout: an unreadable/malformed/non-dict YAML yields ``[]``,
    and an unresolvable ``project_root`` yields a ``None`` probe — the liveness
    half goes quiet (with a breadcrumb naming the DB path that was looked for)
    while the structural half is still returned in full.
    """
    try:
        with open(config_path) as f:
            tree = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        logger.debug('census-ignore audit: cannot read %s (%s)', config_path, exc)
        return []
    if not isinstance(tree, dict):
        return []

    specs = parse_census_ignore_entries(tree)
    if not specs:
        return []

    project_root = tree.get('project_root')
    probe = (
        read_task_statuses(project_root) if isinstance(project_root, str) else None
    )
    if probe is None and not isinstance(project_root, str):
        logger.debug(
            'census-ignore audit: %s has no usable project_root — citation '
            'liveness checks are SKIPPED', config_path,
        )
    return audit_census_ignore_specs(specs, probe)
