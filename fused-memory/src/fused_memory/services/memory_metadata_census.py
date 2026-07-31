"""Observability half of the Mem0 metadata vocabulary contract (task 3195, leaf β).

Deliberately kept OUT of :mod:`fused_memory.memory_metadata` so the registry
module stays a pure, cheaply-importable vocabulary: leaf ι's prompt-pinning
tests and the drift test import the REGISTRY, and should not have to drag in
the optional ``escalation`` package, process-lifetime mutable state and a
queue writer to do it.  This split also puts the never-raises escalation code
in the services layer, where the
:mod:`fused_memory.middleware.candidate_key_escalation` precedent it is ported
from already lives.

Three pieces, matching PRD ``docs/prds/memory-metadata-vocabulary.md`` V1:

* :func:`emit_schema_warnings` — the grep-anchored census line.  Under the
  shipped warn-mode default NOTHING raises, so this log line is the entire
  observable: if it is wrong, the whole warn tier is silent.
* :class:`UnknownKeyStormDetector` — a per-``(project_id, agent_id)`` rolling
  window, so "a drifting writer flooding unknown keys is heard, not logged
  into oblivion" (INV-4).
* :func:`file_unknown_key_storm_escalation` — the escape hatch, filed once
  per open condition and structurally unable to raise.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from fused_memory.memory_metadata import MetadataViolation

if TYPE_CHECKING:
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped]

# Defensive import of the optional ``escalation`` workspace package, mirroring
# ``middleware/candidate_key_escalation.py``.  When it is missing (minimal CI
# envs, deployments that have not installed it) this module degrades to a
# logged no-op rather than breaking the memory write path that calls it.
try:
    from escalation.models import Escalation  # type: ignore[import-untyped]
    from escalation.queue import EscalationQueue  # type: ignore[import-untyped,no-redef]
    HAS_ESCALATION = True
except ImportError:  # pragma: no cover — exercised only in minimal envs
    HAS_ESCALATION = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# The census line
# ---------------------------------------------------------------------------

#: Rendered in place of a missing ``agent_id``.
#:
#: NOT the bare string ``'None'``: that is indistinguishable from a writer
#: that literally set ``agent_id='None'``, which would silently merge two
#: different writers in the storm detector's per-writer aggregation and in
#: any operator grep.  A token that cannot be a real agent id keeps "unset"
#: separable.
UNSET_AGENT_ID = '<unset>'

#: The census grep anchor.
#:
#: CONTRACT-FIXED — **never rename this token.**  It is what the enforce-gate
#: census greps for in the fused-memory journal (PRD §1/§5), exactly as
#: ``task_metadata.schema_warning`` is for the task-metadata boundary
#: (``backends/sqlite_task_backend.py:853``).  It is deliberately distinct
#: from that token so the two censuses never conflate.
#:
#: Operator recipe — separate enforcement-relevant classes from the
#: unknown-key tail, which is expected to be noisy by design::
#:
#:     grep 'memory_metadata.schema_warning' | grep -v code=unknown_key
CENSUS_ANCHOR = 'memory_metadata.schema_warning'


def emit_schema_warnings(
    violations: Sequence[MetadataViolation],
    *,
    project_id: str,
    agent_id: str | None,
) -> None:
    """Emit one WARNING census line per violation.

    Not scoped to warn-mode: a **fatal** violation censuses too.  Under the
    shipped default (``memory_metadata.enforce = False``) a fatal violation
    does not reject, so if it did not also census, the single most
    enforcement-relevant class of violation would be the one class that left
    no trace at all — the exact silent-fail-soft this census exists to
    prevent.

    The ``code=`` token is the violation's class discriminator
    (``unknown_key`` / ``unknown_kind`` / ``invalid_topic_slug`` / ...), which
    is what lets an operator measure how much of the census is the expected
    long tail versus a real shape problem before flipping ``enforce``.
    """
    writer = agent_id if agent_id else UNSET_AGENT_ID
    for violation in violations:
        logger.warning(
            '%s project_id=%s agent_id=%s code=%s key=%s message=%s',
            CENSUS_ANCHOR,
            project_id,
            writer,
            violation.code,
            violation.key,
            violation.message,
        )


# ---------------------------------------------------------------------------
# The storm detector
# ---------------------------------------------------------------------------

class UnknownKeyStormDetector:
    """Rolling-window unknown-key warn counter, keyed per writer.

    Keyed on ``(project_id, agent_id)`` rather than globally because the
    signal PRD V1 asks for is one writer's RATE, not fleet volume.  With a
    1,627-distinct-key live baseline a global counter would fire permanently
    against healthy traffic and would never identify the culprit — it would
    be the "logged into oblivion" failure wearing an alarm.

    The clock is injectable so window behaviour is testable without sleeping;
    the default is :func:`time.monotonic` rather than wall clock so an NTP
    step cannot corrupt an in-flight window.

    Instance state is process-lifetime: ``MemoryService`` constructs one in
    ``__init__`` so counts survive across writes but never leak between
    processes.
    """

    def __init__(
        self,
        *,
        threshold: int,
        window_seconds: int,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._time_fn = time_fn
        self._warns: dict[tuple[str, str], deque[float]] = {}
        #: Writers already over the line, so the crossing fires exactly once.
        self._firing: set[tuple[str, str]] = set()

    def record(
        self, project_id: str, agent_id: str | None, keys: Iterable[str]
    ) -> bool:
        """Record one warn per key and report whether THIS call crossed.

        Returns ``True`` only on the call that crosses the threshold, never on
        subsequent calls while the writer stays over it.  A detector that
        latched ``True`` would ask the filer to act on every single write
        thereafter; the filer's open-escalation dedup would absorb it, but
        only by doing a queue read per memory write.  The crossing is the
        event — not the state.

        The latch clears once the window drains back below the threshold, so
        a writer that drifts, is fixed, and later drifts again is heard both
        times.
        """
        new_keys = list(keys)
        if not new_keys:
            return False

        writer = (project_id, agent_id if agent_id else UNSET_AGENT_ID)
        now = self._time_fn()
        window = self._warns.setdefault(writer, deque())

        for _ in new_keys:
            window.append(now)

        cutoff = now - self._window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

        if len(window) < self._threshold:
            # Back under the line — re-arm, so a recurrence is heard again.
            self._firing.discard(writer)
            return False

        if writer in self._firing:
            return False
        self._firing.add(writer)
        return True


# ---------------------------------------------------------------------------
# The escalation filer
# ---------------------------------------------------------------------------

_QUEUE_DIRNAME: str = 'data/escalations'

#: Stable anchor task id, so the resulting escalation ids
#: (``esc-memory-metadata-unknown-key-storm-N``) are greppable and distinct
#: from every other fused-memory escalation series.  It is also what the
#: open-escalation dedup below keys on.
_ANCHOR_TASK_ID: str = 'memory-metadata-unknown-key-storm'

_AGENT_ROLE: str = 'fused-memory/memory-metadata-census'
_CATEGORY: str = 'memory_metadata_unknown_key_storm'


def file_unknown_key_storm_escalation(
    project_root: str,
    *,
    project_id: str,
    agent_id: str | None,
    keys: Sequence[str],
) -> str | None:
    """File (or reuse) the unknown-key storm escalation for one writer.

    Returns the escalation id — freshly filed, or the id of an already-open
    escalation for this condition — or ``None`` when neither is possible.

    **NEVER RAISES.**  This is called from the live memory write path.  A
    raise here would fail the write because the *complaint about* the write
    failed, turning a census warning into a lost memory — strictly worse than
    the drift being escalated.  Both the optional-package absence and any
    queue I/O failure therefore degrade to a log line and ``None``.

    Deduping against an already-open escalation under ``_ANCHOR_TASK_ID``
    matters for the same reason it does in the ported precedent: a drift
    condition that outlives a process restart would otherwise mint a
    brand-new escalation every time a fresh detector crossed the threshold,
    flooding the operator queue with near-identical entries.
    """
    writer = agent_id if agent_id else UNSET_AGENT_ID

    if not HAS_ESCALATION:
        logger.debug(
            'memory_metadata_census: escalation package unavailable; '
            'unknown-key storm from project_id=%r agent_id=%r (%d key(s)) '
            'will not be escalated',
            project_id, writer, len(keys),
        )
        return None

    try:
        queue = EscalationQueue(Path(project_root) / _QUEUE_DIRNAME)
    except Exception:
        logger.exception(
            'memory_metadata_census: could not open the escalation queue at '
            'project_root=%r; unknown-key storm from project_id=%r '
            'agent_id=%r not escalated',
            project_root, project_id, writer,
        )
        return None

    # Best-effort dedup: a read failure falls THROUGH to filing rather than
    # suppressing. Failing closed here would let a broken queue read swallow
    # the storm signal entirely, which is the outcome this escape exists to
    # prevent.
    try:
        existing = queue.get_by_task(_ANCHOR_TASK_ID, status='pending')
    except Exception:
        logger.exception(
            'memory_metadata_census: failed to check for an existing open '
            'storm escalation at project_root=%r; proceeding to file a new one',
            project_root,
        )
        existing = []
    if existing:
        logger.info(
            'memory_metadata_census: %s already open; unknown-key storm from '
            'project_id=%r agent_id=%r (%d key(s)) folded into it',
            existing[0].id, project_id, writer, len(keys),
        )
        return existing[0].id

    key_list = ', '.join(repr(k) for k in keys)
    detail = '\n'.join([
        f'project_id={project_id!r}',
        f'agent_id={writer!r}',
        f'unknown_key_count={len(keys)}',
        f'keys={key_list}',
        '',
        'This writer emitted enough unknown-metadata-key census warnings '
        f'({CENSUS_ANCHOR} code=unknown_key) inside the configured rolling '
        'window to look like schema drift rather than the expected long '
        'tail. The keys above belong to no known layer (mem0-managed, '
        'server-stamped, reserved vocabulary, or blessed conventional) and '
        'carry no x_ experimental prefix.',
        '',
        'Resolve by one of: adding the key to the blessed tier in '
        'fused_memory.memory_metadata if it is a deliberate convention; '
        'renaming it with the x_ prefix if it is genuinely experimental; or '
        'fixing the writer if the key is a typo or a leak. The vocabulary '
        'and its tiers are defined in fused_memory.memory_metadata; the '
        'thresholds are memory_metadata.unknown_key_storm_* in the '
        'fused-memory config.',
    ])

    try:
        esc = Escalation(  # type: ignore[possibly-unbound]
            id=queue.make_id(_ANCHOR_TASK_ID),
            task_id=_ANCHOR_TASK_ID,
            agent_role=_AGENT_ROLE,
            severity='info',
            category=_CATEGORY,
            summary=(
                f'unknown metadata-key storm from project_id={project_id} '
                f'agent_id={writer} ({len(keys)} key(s): {key_list})'
            ),
            detail=detail,
            suggested_action=(
                'bless, x_-prefix, or fix the drifting writer'
            ),
            level=1,
        )
        esc_id = queue.submit(esc)
    except Exception:
        logger.exception(
            'memory_metadata_census: failed to submit the unknown-key storm '
            'escalation for project_id=%r agent_id=%r (%d key(s))',
            project_id, writer, len(keys),
        )
        return None

    logger.warning(
        'memory_metadata_census: filed %s for an unknown-key storm from '
        'project_id=%r agent_id=%r (%d key(s): %s)',
        esc_id, project_id, writer, len(keys), key_list,
    )
    return esc_id
