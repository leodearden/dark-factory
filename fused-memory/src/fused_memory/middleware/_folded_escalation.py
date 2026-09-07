"""The ONE home for the pending-anchor-fold escalation filer skeleton (INV-5).

Seven fused-memory filers had each been carrying this body verbatim: a
defensive optional-``escalation``-package import, a guarded
``EscalationQueue`` construction, a ``get_by_task(anchor, status='pending')``
dedupe fold, and a never-raise ``Escalation(...)`` + ``queue.submit(...)``.
One of them says so in its own source ("Copied shape-for-shape from
``server/markup_tripwire.emit_markup_storm_escalation``").  Seven copies of a
never-raise alarm path means seven places a guard can be forgotten — and
``candidate_key_escalation`` had already forgotten one (see
:func:`file_folded_escalation`'s queue-construction guard).

WHAT LIVES HERE is the SKELETON only.  Each caller keeps its own
``_ANCHOR_TASK_ID`` / ``_AGENT_ROLE`` / ``_CATEGORY`` constants and its own
summary/detail/suggested_action construction in its own module: that is the
caller's content, and hoisting it would make the helper a grab-bag of seven
unrelated alarms.

WHY ``anchor_task_id`` IS REQUIRED, WITH NO DEFAULT.  A filer that dedupes
against an anchor somebody else keeps open never files again, and that silence
is indistinguishable from health.  MEASURED (the anchor-squat incident, pinned
by ``server/write_triage.py``'s comment block and by
``tests/test_folded_escalation.py::TestNoTwoFilersShareAnAnchor``): the L1
escalation watcher squatted the ``markup-tripwire`` anchor, so the tripwire
filed NOTHING from 2026-08-16 to 2026-08-19 while 41 rejections occurred — all
17 records sat at ``dedupe_count`` 0, i.e. the fold was not even folding, it
was simply never firing.  A required keyword-only parameter with no default
makes a shared anchor impossible to introduce by OMISSION: forgetting it is a
``TypeError`` at call time, not a silently disabled alarm.

NOT THE HOME FOR the ``submit_or_dedupe`` content-fingerprint family —
:mod:`fused_memory.middleware.scope_violation_escalator`,
:mod:`fused_memory.middleware.mem0_update_storm_escalator` and
:mod:`fused_memory.middleware.entity_mint_storm_escalator`.  Those do not carry
this skeleton at all: they dedupe on ``compute_content_fingerprint`` +
``DedupeConfig`` through ``escalation.dedupe.submit_or_dedupe`` over a CACHED
per-project queue, at ``severity='info'``.  ``ScopeViolationEscalator._submit``
is already the extracted one-home for that family's three modes, and its
docstring states that keeping the dedup policy in one place is load-bearing
(the task-3119 mislabelling defect).  Folding them in here would replace
content-fingerprint dedupe with pending-anchor dedupe — a behaviour regression,
not a refactor.  Do not "finish the job" by migrating them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Defensive import, lifted verbatim from the seven copies this module replaces:
# the `escalation` workspace package is optional (minimal CI envs, unit tests
# without escalation infra, deployments that have not installed it). When it is
# missing every filer becomes a logged no-op, so each caller's own
# never-fail-the-write guarantee is never at risk.
#
# This is ALSO the monkeypatch seam every migrated caller's tests now target:
# the names live here and NOT in the callers (no compatibility re-export), so a
# stale `setattr(caller_module, 'EscalationQueue', ...)` fails loudly with
# AttributeError instead of silently becoming a no-op that leaves the test
# passing for the wrong reason.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

_QUEUE_DIRNAME: str = 'data/escalations'


def file_folded_escalation(
    project_root: str | None,
    *,
    anchor_task_id: str,
    agent_role: str,
    category: str,
    severity: str,
    summary: str,
    detail: str,
    suggested_action: str,
    logger: logging.Logger,
    log_label: str,
    level: int = 1,
    dedupe: bool = True,
) -> str | None:
    """File one escalation under *anchor_task_id* into *project_root*'s queue.

    Args:
        project_root: The affected project's root; the escalation lands in that
            project's OWN ``data/escalations`` queue. Never defaulted to the
            server cwd, where no operator watches.
        anchor_task_id: The synthetic task id the record is filed under and the
            fold is keyed on. REQUIRED and keyword-only, with NO default — see
            the module docstring's anchor-squat incident. Callers keep this as
            their own module-level constant (or compute it per-ref/per-writer);
            it must never be shared with another filer.
        agent_role: The filer's role string, e.g.
            ``'fused-memory/referent-repair-guard'``.
        category: The escalation category, e.g. ``'referent_repair_storm'``.
        severity: ``'blocking'`` or ``'info'`` — the CALLER's choice. Two of the
            seven filers file at ``info``; the helper never imposes one.
        summary: One-line operator-facing summary.
        detail: The INV-2 structured evidence, already rendered by the caller.
        suggested_action: What a triager should do.
        logger: The CALLER's logger, so every record is attributed to the
            caller's module name rather than to this helper.
        log_label: The caller's grep token (``'write_triage'``,
            ``'markup_tripwire'``, ...), prefixed onto every emitted message.
            Existing detail text tells triagers to grep for these.
        level: Escalation level. Defaults to 1.

            BORN AT L1, NOT L0. The L0-routes-to-the-steward rule governs an
            escalation filed BY A DISPATCHED AGENT about its own task; it does
            not reach here. ``Steward._pick_escalation`` reads
            ``escalation_queue.get_by_task(self.task_id, status='pending',
            level=0)`` (``orchestrator/src/orchestrator/steward.py::Steward``),
            scoped to the REAL task that steward was spawned for. Every filer
            using this helper is a background server process filing under a
            synthetic anchor that is never dispatched and therefore never has a
            steward — so an L0 entry here would have no consumer at all. It
            would be reached only by
            ``HarnessRunner._reap_orphan_l0_escalations``, which promotes
            unclaimed L0s to L1 after ``orphan_l0_timeout_secs``. Filing at L0
            would therefore not route the alarm to a steward; it would merely
            DELAY it by that timeout before landing exactly where L1 puts it
            immediately. For a storm escape whose whole purpose is that a
            regression not be absorbed silently, a built-in delay is the wrong
            default. ``emit_markup_residue_escalation`` overrides this with a
            caller-supplied level.
        dedupe: When true (the default), fold into an already-open escalation
            under *anchor_task_id* instead of filing a duplicate. Set false
            only when each record is the sole surviving copy of a DIFFERENT
            payload — ``emit_markup_residue_escalation`` is the one such caller,
            where folding two records together would destroy the very data the
            record exists to preserve.

    Returns the escalation id — freshly filed, or the id of the already-open
    escalation under this anchor when one exists — or ``None`` when filing was
    not possible. NEVER raises.
    """
    if project_root is None:
        logger.debug(
            '%s: no project_root, so there is no project queue to file into; '
            'nothing escalated', log_label,
        )
        return None

    if not HAS_ESCALATION:
        logger.debug(
            '%s: escalation package unavailable; nothing will be escalated',
            log_label,
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        # Constructing the queue creates its directory; a read-only or missing
        # project_root must not turn an alarm into a crash on a write path.
        logger.exception(
            '%s: could not open the escalation queue at project_root=%r; '
            'nothing escalated', log_label, project_root,
        )
        return None

    # DEDUPE-FOLD. Once a project is storming, EVERY subsequent event breaches
    # the threshold again — the streak only grows until a clean pass resets it.
    # Filing per breach would bury the operator queue under near-identical
    # entries and make the real signal (one project, one regression) harder to
    # see, not easier. `anchor_task_id` is a stable per-caller anchor, so any
    # still-pending escalation under it IS this caller's open alarm.
    #
    # THE ANCHOR MUST STAY PER-CALLER, and it is threaded through BOTH this
    # lookup and `make_id`/`task_id` below from the SAME parameter, so it is
    # structurally impossible to file under one anchor while deduping against
    # another. Two filers sharing an anchor is not a cosmetic collision: the
    # second one goes permanently silent behind the first one's open record,
    # and that silence reads exactly like health. See the module docstring for
    # the measured incident.
    #
    # A read failure falls THROUGH to filing rather than aborting: a possible
    # duplicate is a far cheaper failure than a silenced alarm, and this arm is
    # reached only when the queue directory is already misbehaving. This guard
    # must never `return` — returning early here would convert a transient
    # queue-scan error into exactly the permanent silence the fold's anchor
    # discipline exists to prevent.
    existing = []
    if dedupe:
        try:
            existing = queue.get_by_task(anchor_task_id, status='pending')
        except Exception:
            logger.exception(
                '%s: failed to check for an already-open alarm under anchor %r '
                'in project_root=%r; proceeding to file a new one rather than '
                'silencing the alarm',
                log_label, anchor_task_id, project_root,
            )
            existing = []
    if existing:
        logger.info(
            '%s: %s already open; folding into it rather than filing a duplicate',
            log_label, existing[0].id,
        )
        return existing[0].id

    # `Escalation(...)` is constructed INSIDE the guard deliberately, matching
    # `ScopeViolationEscalator._submit`'s stated reason: a malformed payload
    # must degrade to "no escalation", never to an exception out of the guard.
    # The events these filers complain about have already committed by the time
    # this runs; a queue I/O failure must cost the operator a heads-up, never
    # the write.
    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(anchor_task_id),
            task_id=anchor_task_id,
            agent_role=agent_role,
            severity=severity,
            category=category,
            summary=summary,
            detail=detail,
            suggested_action=suggested_action,
            level=level,
        )
        esc_id = queue.submit(esc)
    except Exception:
        logger.exception(
            '%s: failed to submit the alarm under anchor %r in '
            'project_root=%r', log_label, anchor_task_id, project_root,
        )
        return None

    logger.warning('%s: queued %s', log_label, esc_id)
    return esc_id
