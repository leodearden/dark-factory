"""Escalation filers for leaked MCP envelope markup (tasks 3141, 4458).

A recurring harness serialization bug leaks raw MCP envelope fragments — the
closing/opening tags of the tool-call wire format — into the *payload* of
fused-memory writes. Two observed vectors: memory ``content`` arriving with a
``</content>``/``</invoke>`` tail (permanent specimens now sitting in the mem0
and Graphiti corpora), and task text arriving with a ``<parameter name=``
fragment that the interceptor's description parser then mis-parses *silently*
(reify task 3210 was filed ``priority=high`` and stored as ``medium``).

WHAT THIS MODULE IS NOW
-----------------------
Task 3141 put the containment in a WRITE-TIME tripwire: a pattern matcher
(``find_markup_pattern`` / ``find_markup_violation``), a rejection-block builder
(``build_markup_block``) and a burst counter (``MarkupStormCounter``), called
from a ``_markup_gate`` closure at four tool bodies in ``server/tools.py``.

Task 4458 moved the containment to the DISPATCH BOUNDARY
(:mod:`fused_memory.server.markup_guard`, wrapping ``ToolManager.call_tool``
with the shared :class:`shared.mcp_markup_middleware.MarkupGuardMiddleware`),
which covers every tool rather than four, and repairs a swallowed parameter
before pydantic ever sees the call. The four in-line gates were retired, and
those four symbols went with them: keeping a second, fully working markup
mechanism alive behind its own tests is precisely the standing invitation to
re-introduce a duplicate implementation that INV-5 forbids. Their behavioural
coverage now lives in ``tests/server/test_markup_tripwire_gate.py``, which pins
that an UNGUARDED server no longer refuses a leaked write while a guarded one
does.

What remains here is what the boundary guard's escalation sink CALLS, plus the
re-exports ``server/tools.py`` still imports:

* :func:`emit_markup_storm_escalation` — one deduped record per burst (INV-4).
* :func:`emit_markup_residue_escalation` — one record per unrepairable refusal,
  deliberately NOT deduped, holding that caller's payload verbatim (INV-7).
* ``_MARKUP_STORM_THRESHOLD`` / ``_MARKUP_STORM_WINDOW_SECONDS`` — the ONE
  source for the tuning, read by the guard when it builds the middleware.
* ``MARKUP_OVERRIDE_KEY``, ``markup_override_requested``,
  ``strip_markup_override``, ``MCP_MARKUP_PATTERNS`` — RE-EXPORTS from
  :mod:`shared.toolcall_markup`, which OWNS the literal enumeration and the
  override lifecycle (INV-5, task 3688/3689). Nothing in this package spells
  those literals, and the write-time/read-time predicate calibration is
  documented there rather than restated here.

Root cause, the Qdrant payload text-match read tool, and the retroactive corpus
sweep belong to **DF task 3083**, which is DONE and CLOSED to appends — report
a recurrence against its successor ``plans/toolcall-markup-containment-prd.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

# markup_override_requested / strip_markup_override are RE-EXPORTS, not local
# callers: server/tools.py imports both from this module (the write-path guards
# call them, this module only defines the flag they key on). The noqa is the
# re-export marker — deleting either import as "unused" would break tools.py.
from shared.toolcall_markup import (  # noqa: F401
    MARKUP_OVERRIDE_KEY,
    MCP_MARKUP_PATTERNS,
    markup_override_requested,
    strip_markup_override,
)

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Defensive import of the optional ``escalation`` workspace package, mirroring
# middleware/candidate_key_escalation.py: when it is missing (minimal CI envs,
# deployments that have not installed it) the storm escalation becomes a logged
# no-op. This module sits on the MCP write path, so it must never make a write
# fail — the rejection is already decided by the time escalation is attempted.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)

# MCP_MARKUP_PATTERNS — imported above, NOT spelled here.
#
# The enumeration is owned by shared.toolcall_markup (INV-5). It used to be
# written out in this module while the read-time prefilter list was written out
# in fused_memory/utils/toolcall_xml_leak.py, each documented as the single
# source of truth; they drifted, which is the defect task 3688 repairs.
#
# The matcher OVER it (find_markup_pattern / find_markup_violation) was deleted
# in task 4458 with the write-time gate it served; shared.toolcall_markup.detect
# is the live generalisation, and it is what the boundary guard calls. The
# constant stays re-exported here for the same-file drift guard in
# tests/server/test_markup_tripwire.py, which pins its value and order from this
# import site.

# MARKUP_OVERRIDE_KEY, markup_override_requested, strip_markup_override —
# imported above, NOT defined here.
#
# The enumeration moved to shared.toolcall_markup under task 3688; task 3689
# moved its OVERRIDE LIFECYCLE after it, for the same reason — the middleware
# sits in the base layer, which may not import fused_memory, and a second
# implementation of "is this deliberate?" is exactly the lockstep-duplication
# defect (INV-5) the enumeration move repaired.
#
# Since task 4458 there is only ONE guard that bounces a caller for envelope
# markup (shared.mcp_markup_middleware, at the dispatch boundary), and it is
# what honours the override. server/tools.py still imports strip_markup_override
# from HERE, because the middleware forwards the flag UNCHANGED to any tool
# declaring `metadata`: the tool body remains the party that keeps a write-time
# control flag out of the corpus.
#
# What none of that changed: the semantics are byte-for-byte the ones this
# module has always had — fail-closed on a literal boolean True only,
# dict-or-JSON-string tolerant, never raising, and stripped NON-mutatingly in
# the caller's own shape before dispatch. shared/tests/test_toolcall_markup.py
# pins both the semantics and this re-export.

# Storm thresholds are plain module constants rather than FusedMemoryConfig
# fields, following the _PLACEHOLDER_DROP_STORM_* precedent (harness.py:200-215):
# this leaf owns both the predicate and its only consumer, so a config field
# would add hot-reload tier surface and a schema migration for no operator gain.
_MARKUP_STORM_THRESHOLD = 3
_MARKUP_STORM_WINDOW_SECONDS = 3600.0

# Escalation wiring, copied shape-for-shape from
# middleware/candidate_key_escalation.py. _ANCHOR_TASK_ID is a stable per-project
# anchor (not a real task id) so the resulting ids form one greppable
# ``esc-markup-tripwire-N`` series and the dedup check has something to key on.
_QUEUE_DIRNAME: str = 'data/escalations'
_ANCHOR_TASK_ID: str = 'markup-tripwire'
_AGENT_ROLE: str = 'fused-memory/markup-tripwire'
_CATEGORY: str = 'mcp_markup_write_storm'

# RESIDUE wiring — a different record KIND on the same queue, so it gets its own
# anchor (an ``esc-markup-residue-N`` series) rather than sharing either the
# tripwire's or the boundary guard's storm anchor. A storm record summarises a
# burst and is deduped; a residue record HOLDS a caller's payload and is not.
# The category/level/summary/suggested_action are the middleware's OWN
# (``_ESCALATION_CATEGORY`` = 'mcp_markup_residue', ``_ESCALATION_OWNER`` =
# 'l2-escalation-watcher', ``_ESCALATION_LEVEL`` = 2), read off the record rather
# than re-authored here; these are fallbacks for a record that omits them.
_RESIDUE_ANCHOR_TASK_ID: str = 'markup-residue'
_RESIDUE_AGENT_ROLE: str = 'fused-memory/markup-guard'
_RESIDUE_CATEGORY: str = 'mcp_markup_residue'
_RESIDUE_LEVEL: int = 2


def emit_markup_storm_escalation(
    project_root: str | None,
    storm: dict[str, Any],
    *,
    anchor_task_id: str = _ANCHOR_TASK_ID,
) -> str | None:
    """File an ``mcp_markup_write_storm`` escalation for a rejection burst (INV-4).

    Returns the escalation id — freshly filed, or the id of an already-open
    escalation for this project (dedup) — or ``None`` when filing is impossible
    or fails.

    *storm* is the record its ONE live producer sends —
    :meth:`shared.mcp_markup_middleware.MarkupGuardMiddleware._record_storm` via
    :mod:`fused_memory.server.markup_guard`'s sink — and the keys read off it
    are ``count``, ``threshold``, ``window_seconds`` and ``outcome``. Every one
    is read defensively, because degenerate and legacy shapes (``{}``,
    ``{'count': 9}``) reach this filer from tests and older callers and must
    still produce a routable record.

    ``count`` is ALREADY this project's own number: the producer keys one
    ``StormCounter`` per ``(project, outcome)`` pair, so a window can only ever
    hold one project's events. Do not re-add a "the count may span other
    projects" hedge — that was true of the per-SERVER ``MarkupStormCounter``
    task 4458 retired, and restating it now tells the triager to discount a
    number that is exactly reproducible from their own journal.

    NEVER raises. This is called from an MCP write path whose rejection has
    already been decided, so escalation is purely additive: every failure mode
    degrades to ``None`` plus a log line rather than changing the write's
    outcome. Copied shape-for-shape from
    :func:`middleware.candidate_key_escalation.emit_residual_candidate_key_escalation`,
    adding an early return for a ``None`` *project_root* — ``add_memory`` /
    ``add_episode`` take a ``project_id``, which for an unknown project resolves
    to no root at all, and that must be a quiet no-op rather than a crash.

    The anchor dedup matters beyond the counter's own per-window rate limit: a
    leak running for hours would otherwise file one escalation per window, so
    those collapse into the single open record until an operator resolves it.

    *anchor_task_id* selects WHICH anchor that dedup keys on, and is threaded
    through both the lookup and the filing so a caller can never file under one
    anchor while deduping against another. It exists because the default anchor
    is SHARED: the L1 escalation watcher files its own cluster records under
    ``markup-tripwire`` and squats it — measured, the tripwire filed nothing
    2026-08-16..2026-08-19 while 41 rejections occurred, all 17 records at
    dedupe_count 0 — so a filer deduping against a squatted anchor is suppressed
    indefinitely, which is silence an operator reads as calm.

    This is a PARAMETERISATION of the one filer, not licence to write a second
    one (INV-5). The default is unchanged, so every existing caller behaves
    identically.
    """
    if project_root is None:
        logger.debug(
            'markup_tripwire: no project_root resolved; storm %r will not be escalated',
            storm,
        )
        return None
    if not HAS_ESCALATION:
        logger.debug(
            'markup_tripwire: escalation package unavailable; storm %r in '
            'project_root=%r will not be escalated',
            storm, project_root,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'markup_tripwire: failed to open the escalation queue for '
            'project_root=%r; storm %r not escalated',
            project_root, storm,
        )
        return None

    # Best-effort dedup: a read failure falls THROUGH to filing rather than
    # bailing out — losing duplicate-suppression is strictly better than losing
    # the alarm for an actively running leak.
    try:
        existing = queue.get_by_task(anchor_task_id, status='pending')
    except Exception:
        logger.exception(
            'markup_tripwire: failed to check for an existing open escalation '
            'for project_root=%r; proceeding to file a new one',
            project_root,
        )
        existing = []
    if existing:
        logger.info(
            'markup_tripwire: %s already open for project_root=%r (storm %r now); '
            'not filing a duplicate',
            existing[0].id, project_root, storm,
        )
        return existing[0].id

    count = storm.get('count')
    window_seconds = storm.get('window_seconds')
    detail = '\n'.join([
        f'project_root={project_root!r}',
        # The producer keys one StormCounter per (project, outcome) pair
        # (mcp_markup_middleware.py:648-652), so this window holds ONE project's
        # events for ONE outcome: the count below is this project_root's own
        # contribution, and an operator can reproduce it by grepping this
        # project's own `markup guard: <outcome> tool=... project=...` lines.
        # It was not always so — task 4505 removed a `projects_in_window=` line
        # and a comment hedging that the count "may span more than this
        # project_root", both left over from the per-SERVER MarkupStormCounter
        # task 4458 retired. That pooling is what filed esc-markup-tripwire-6
        # into reify's queue stating 3, for a window reify contributed 1 to.
        f'outcome={storm.get("outcome")!r}',
        f'rejected_writes_in_window={count!r}',
        f'threshold={storm.get("threshold")!r}',
        f'window_seconds={window_seconds!r}',
        '',
        'The fused-memory MCP write tripwire (task 3141) rejected multiple '
        'writes carrying raw MCP envelope markup within one rolling window. A '
        'burst means the upstream harness serialization leak is ACTIVE right '
        'now, not that the tripwire is misfiring — do NOT disable it, or '
        'further specimens will land permanently in the corpus.',
        '',
        'DF task 3083 delivered the root cause and the Qdrant payload '
        'text-match read tool, but it is DONE and CLOSED to appends — nothing '
        'reads what is attached there, and its metadata is APPEND-ONLY: a key '
        'can be added or overwritten, but none can be removed or renamed, '
        'because deletion needs metadata_mode=replace and any faithful replace '
        'payload carries done_provenance, which the write-authority floor '
        'refuses. Its successor plans/toolcall-markup-containment-prd.md '
        'owns the live blast-radius, deterministic-repair and retro-sweep work. '
        'Attach the agent_id, field and matched_pattern from the rejection '
        "responses (grep the server logs for 'markup_tripwire_storm') against "
        "that PRD's open leaves, not against 3083.",
    ])

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(anchor_task_id),
            task_id=anchor_task_id,
            agent_role=_AGENT_ROLE,
            severity='blocking',
            category=_CATEGORY,
            summary=(
                f'{count} MCP write(s) rejected for leaked envelope markup in '
                f'{window_seconds}s — serialization leak active '
                '(see plans/toolcall-markup-containment-prd.md)'
            ),
            detail=detail,
            suggested_action=(
                'identify the leaking caller from the rejection logs and report '
                'it against plans/toolcall-markup-containment-prd.md — DF task '
                '3083 is done and closed to appends'
            ),
            level=1,
        )
        esc_id = queue.submit(esc)
    except Exception:
        # A queue I/O failure must not propagate: the write has already been
        # rejected, and the ERROR log at the call site has already recorded the
        # burst. The operator simply loses the queued heads-up.
        logger.exception(
            'markup_tripwire: failed to submit storm escalation for '
            'project_root=%r (storm %r)',
            project_root, storm,
        )
        return None

    logger.warning(
        'markup_tripwire: queued %s for project_root=%r (storm %r)',
        esc_id, project_root, storm,
    )
    return esc_id


def emit_markup_residue_escalation(
    project_root: str | None,
    record: dict[str, Any],
    *,
    anchor_task_id: str = _RESIDUE_ANCHOR_TASK_ID,
) -> str | None:
    """Preserve the payload of ONE refused-as-unrepairable call (INV-7, B5).

    Returns the filed escalation id, or ``None`` when filing is impossible or
    fails. The id is what makes the caller-facing refusal HONEST: the shared
    middleware folds it into the response as ``escalation_id`` beside a hint
    that says "your full payload is preserved verbatim in the escalation named
    above". Returning ``None`` there leaves that sentence pointing at a record
    that does not exist — and per the boundary specimen table, unrepairable is
    the COMMON outcome for the real corpus shapes, not an edge case.

    A TRANSLATOR, not a second escalation vocabulary (INV-5). The middleware
    already emits an escalation-SHAPED record — ``category``, ``level``,
    ``summary``, ``suggested_action``, plus the flat ``tool``/``field``/
    ``matched_pattern``/``agent_id``/``project``/``raw_value`` fields — because
    the record has to survive being routed to whichever queue a host server
    happens to own. This function moves those fields onto an
    :class:`escalation.models.Escalation` and writes it; it authors no prose of
    its own beyond the detail layout.

    DELIBERATELY NOT DEDUPED, which is the one place it diverges from
    :func:`emit_markup_storm_escalation`. That filer collapses a running leak
    into one open record because every burst summary says the same thing. Here
    each record is the ONLY surviving copy of a DIFFERENT caller payload, so
    folding two together would destroy data — the exact loss this record exists
    to prevent. Volume during a burst is what the storm record is for: it
    summarises the incident once, while these hold the individual payloads.

    NEVER raises, for the same reason as its sibling: the refusal is already
    decided by the time this runs, so every failure mode degrades to ``None``
    plus a log line rather than changing the call's outcome.
    """
    if project_root is None:
        logger.debug(
            'markup_tripwire: no project_root resolved; residue %r will not be filed',
            record.get('tool'),
        )
        return None
    if not HAS_ESCALATION:
        logger.debug(
            'markup_tripwire: escalation package unavailable; residue of %r in '
            'project_root=%r will not be filed',
            record.get('tool'), project_root,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'markup_tripwire: failed to open the escalation queue for '
            'project_root=%r; residue of %r not filed',
            project_root, record.get('tool'),
        )
        return None

    raw_value = record.get('raw_value')
    detail = '\n'.join([
        f'tool={record.get("tool")!r}',
        f'field={record.get("field")!r}',
        f'matched_pattern={record.get("matched_pattern")!r}',
        f'agent_id={record.get("agent_id")!r}',
        f'project={record.get("project")!r}',
        f'owner={record.get("owner")!r}',
        '',
        'The MCP boundary markup guard REFUSED this call: the argument below '
        'absorbed raw MCP tool-call envelope markup whose own boundary cannot '
        'be determined, so no repair was attempted and NOTHING was written. '
        'Guessing a boundary would silently drop whatever arguments hide in the '
        'residue.',
        '',
        'The caller was told its payload is preserved here, so this record is '
        'the only surviving copy: recover it for the caller if it is still '
        'needed, then chase the harness serialization leak that produced it. '
        'Report the recurrence against plans/toolcall-markup-containment-prd.md '
        '(DF 3083 is done and closed to appends, so not against 3083).',
        '',
        'raw_value (VERBATIM, the caller\'s own bytes):',
        str(raw_value),
    ])

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(anchor_task_id),
            task_id=anchor_task_id,
            agent_role=_RESIDUE_AGENT_ROLE,
            severity='blocking',
            category=str(record.get('category') or _RESIDUE_CATEGORY),
            summary=str(
                record.get('summary')
                or f'Unrepairable MCP envelope markup in {record.get("tool")}'
            ),
            detail=detail,
            suggested_action=str(record.get('suggested_action') or ''),
            level=int(record.get('level') or _RESIDUE_LEVEL),
        )
        esc_id = queue.submit(esc)
    except Exception:
        logger.exception(
            'markup_tripwire: failed to submit the residue escalation for '
            'project_root=%r (tool=%r field=%r); the payload survives only in '
            'the caller-facing log line',
            project_root, record.get('tool'), record.get('field'),
        )
        return None

    logger.warning(
        'markup_tripwire: queued %s holding the unrepairable payload of %s.%s '
        '(%d chars) for project_root=%r',
        esc_id, record.get('tool'), record.get('field'),
        len(raw_value or ''), project_root,
    )
    return esc_id
