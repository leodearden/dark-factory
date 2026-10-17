"""Observability half of the Mem0 metadata vocabulary contract (task 3195, leaf β).

Deliberately kept OUT of :mod:`fused_memory.memory_metadata` so the registry
module stays a **pure** vocabulary — no process-lifetime mutable state, no
queue writer, and no dependency on the optional ``escalation`` package: leaf
ι's prompt-pinning tests and the drift test import the REGISTRY, not a
running subsystem.  Note this buys purity, NOT import cheapness: the registry
still transitively imports the mem0 SDK via ``backends/mem0_client.py``, the
D12-decided home of ``MEM0_MANAGED_METADATA_KEYS`` (see that module's
docstring for the measured cost).  This split also puts the never-raises
escalation code in the services layer, where the
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
import re
import time
from collections.abc import Callable, Iterable, Sequence

from shared.storm_counter import KeyedStormCounters

from fused_memory.memory_metadata import MetadataViolation
from fused_memory.middleware._folded_escalation import file_folded_escalation

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

#: Recorded events between stale-writer sweeps (the sweep itself is
#: ``shared/src/shared/storm_counter.py::KeyedStormCounters``).  Sized so the
#: O(writers) sweep is negligible against the write path while still bounding
#: the registry long before it is large enough to matter.
DEFAULT_SWEEP_EVERY = 256


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
    processes.  Because it IS process-lifetime, silent writers are swept
    (:attr:`tracked_writers`); without that, a long-running MCP server
    accumulates one entry per distinct writer forever.

    The per-writer keying and sweep ARE
    :class:`shared.storm_counter.KeyedStormCounters` in ``fire_mode='latched'``
    (tasks 4519 and 5102, INV-5).  This class owns only the writer key, the
    multi-key call, and the crossing-is-the-event contract.  Evicting a
    drained writer clears its latch by deleting the counter that holds it,
    which ``StormCounter.prune``'s licence makes behaviour-preserving.
    """

    def __init__(
        self,
        *,
        threshold: int,
        window_seconds: int,
        time_fn: Callable[[], float] = time.monotonic,
        sweep_every: int = DEFAULT_SWEEP_EVERY,
    ) -> None:
        self._threshold = threshold
        self._window_seconds = window_seconds
        self._time_fn = time_fn
        # ``sweep_every`` is injectable for the reason ``time_fn`` is: so the
        # eviction contract is testable without DEFAULT_SWEEP_EVERY records.
        self._warns: KeyedStormCounters[tuple[str, str]] = KeyedStormCounters(
            fire_mode='latched', sweep_every=sweep_every
        )

    @property
    def tracked_writers(self) -> frozenset[tuple[str, str]]:
        """The ``(project_id, agent_id)`` writers currently holding a window.

        ``agent_id`` is free-form per-task text, so evicting silent writers is
        a real memory bound and belongs in the interface, as
        ``BoundaryStormEscape.tracked_keys`` argues.
        """
        return self._warns.tracked_keys

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
        times.  A call carrying N keys decides exactly as N single-key calls
        at the same instant would, returning ``True`` iff one of its keys
        crossed, so the re-arm rule has one home:
        :class:`shared.storm_counter.StormCounter`'s ``fire_mode='latched'``.

        The call's instant is resolved ONCE from ``time_fn`` and threaded
        through every per-key
        :meth:`~shared.storm_counter.KeyedStormCounters.record` as its
        ``now=``, because the body this replaced computed a single ``cutoff``
        for the whole call: a shared instant keeps every key in a call landing
        in the same window, and ages the sweep against that same instant.
        (It is a local, not a parameter of this method — the ``*name*``
        emphasis in this file is reserved for real arguments, as it is on
        :meth:`~shared.storm_counter.StormCounter.record` itself.)
        """
        new_keys = list(keys)
        if not new_keys:
            return False

        writer = (project_id, agent_id if agent_id else UNSET_AGENT_ID)
        now = self._time_fn()

        # An explicit loop, never ``any(self._warns.record(...) for ...)``: a
        # generator short-circuits on the first fire and would skip the
        # remaining keys, silently under-counting a multi-key burst.
        crossed = False
        for _ in new_keys:
            summary = self._warns.record(
                writer,
                threshold=self._threshold,
                window_seconds=self._window_seconds,
                now=now,
            )
            crossed = crossed or summary is not None
        return crossed


# ---------------------------------------------------------------------------
# The escalation filer
# ---------------------------------------------------------------------------

#: Stable PREFIX of the anchor task id, so every escalation in this series
#: (``esc-memory-metadata-unknown-key-storm-<project>-<agent>-N``) is greppable
#: as one family and distinct from every other fused-memory series.  The full
#: anchor is per-writer — see :func:`writer_anchor_task_id`.
_ANCHOR_TASK_ID: str = 'memory-metadata-unknown-key-storm'

#: Cap on each writer component of the anchor.  ``EscalationQueue.make_id``
#: names a durable per-``task_id`` counter file (``queue_dir/esc-{task_id}.seq``),
#: so the anchor becomes a filename — bound it rather than trusting free-form
#: agent ids to stay short.
_ANCHOR_COMPONENT_MAX_LEN: int = 48

_ANCHOR_UNSAFE_RE = re.compile(r'[^a-z0-9]+')

_AGENT_ROLE: str = 'fused-memory/memory-metadata-census'
_CATEGORY: str = 'memory_metadata_unknown_key_storm'


def _anchor_slug(value: str) -> str:
    """Slugify one writer component for safe use inside an anchor task id."""
    slug = _ANCHOR_UNSAFE_RE.sub('-', value.lower()).strip('-')
    slug = slug[:_ANCHOR_COMPONENT_MAX_LEN].strip('-')
    return slug or 'unknown'


def writer_anchor_task_id(project_id: str, agent_id: str | None) -> str:
    """Anchor task id for ONE drifting writer.

    Scoped per ``(project_id, agent_id)`` rather than global.  The escalation's
    entire job is to say WHICH writer is drifting, so a single global anchor
    would fold every later writer's crossing into the first writer's still-open
    escalation (the ``get_by_task`` dedup below) and leave writers B..N visible
    only in an INFO log line: the operator would see one culprit named and get
    no signal at all that others crossed.  Per-writer scoping keeps the dedup
    that matters — one writer still cannot flood the queue — and drops the
    masking that does not.

    The id space this opens is bounded by construction: an anchor is minted
    only when a writer actually crosses the storm threshold, so it is "writers
    that stormed", not the open ``agent_id`` space.  Each such writer is by
    definition one an operator needs to see.
    """
    return (
        f'{_ANCHOR_TASK_ID}'
        f'-{_anchor_slug(project_id)}'
        f'-{_anchor_slug(agent_id or UNSET_AGENT_ID)}'
    )


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

    Deduping against an already-open escalation matters for the same reason
    it does in the ported precedent: a drift condition that outlives a
    process restart would otherwise mint a brand-new escalation every time a
    fresh detector crossed the threshold, flooding the operator queue with
    near-identical entries.  The dedup is keyed on the PER-WRITER anchor
    (:func:`writer_anchor_task_id`), so it suppresses repeats of the same
    writer without masking a second, different writer that also crossed.
    """
    writer = agent_id if agent_id else UNSET_AGENT_ID
    anchor = writer_anchor_task_id(project_id, agent_id)

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

    return file_folded_escalation(
        project_root,
        anchor_task_id=anchor,
        agent_role=_AGENT_ROLE,
        category=_CATEGORY,
        severity='info',
        summary=(
            f'unknown metadata-key storm from project_id={project_id} '
            f'agent_id={writer} ({len(keys)} key(s): {key_list})'
        ),
        detail=detail,
        suggested_action=(
            'bless, x_-prefix, or fix the drifting writer'
        ),
        logger=logger,
        log_label='memory_metadata_census',
        context=(
            f'unknown-key storm from project_id={project_id!r} '
            f'agent_id={writer!r} ({len(keys)} key(s): {key_list})'
        ),
        level=1,
    )
