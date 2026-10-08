"""Backend data layer for the Escalations dashboard section.

The Escalations tab reads the escalation corpus
(:mod:`dashboard.data.escalation_corpus`); this module turns that walk into
the tab's queues and names the task each row's card needs.

- ``build_escalation_queues`` — one subsection per corpus queue: its live-queue
  rows, the files it could not read, level/status summaries and the two named
  views, plus the same rolled up across every queue.
- ``resolve_owning_project`` — maps a reconciliation escalation back to its
  owning project via worktree-prefix matching or an active-row probe.
- ``card_task_refs`` / ``card_datums`` — the task each row's card names, and
  each row's card as a ``Datum`` once those have been looked up.
- ``load_queue_escalations`` — the older root-only ``esc-*.json`` reader of
  one queue directory.  Its one consumer is ``dashboard.data.memory_evals``.
- ``fetch_pins_recovery`` — the one ASYNC function here: fans
  ``get_pending_escalations`` out across every configured escalation MCP and
  returns each project's per-record ``pins_recovery`` annotation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import httpx

from dashboard.data.datum import Datum, unknown_datum
from dashboard.data.escalation_corpus import (
    EscalationCorpus,
    Location,
    QueueKind,
    QueueRef,
    views_over,
)
from dashboard.data.memory import mcp_tool_call
from dashboard.data.task_lookup import FETCHED_ROW_FRESHNESS_BOUND_SECONDS, TaskRef

logger = logging.getLogger(__name__)


def load_queue_escalations(
    esc_dir: Path,
    *,
    skipped: list[dict[str, Any]] | None = None,
) -> list[dict]:
    """Load all escalation JSON files from *esc_dir* (root only, no archive).

    Only ``esc-*.json`` files directly inside *esc_dir* are read — subdirectories
    (including ``archive/YYYY-MM-DD/``) are **not** traversed.  This is an
    intentional divergence from :func:`escalation.queue.iter_all_escalation_paths`,
    which also walks the archive subtree.

    Only ``esc-*.json`` names an escalation.  The queue directory has other
    residents: ``orchestrator/src/orchestrator/b3_gate.py::STATE_REL_PATH``
    keeps ``b3-state.json`` there, and a ``*.json`` glob read it as an id-less
    escalation.  :func:`escalation.queue.iter_all_escalation_paths`, and so the
    corpus walk the Escalations tab reads, globs ``esc-*.json`` for the same
    reason; ``plans/escalation-watcher-queue-ops-hardening-prd.md`` invariant D6
    names protecting that file as why the glob is never widened.

    A missing or non-directory *esc_dir* returns ``[]`` without raising.
    A file that cannot be read or parsed as JSON is skipped with a ``WARNING``
    log — and, if the caller opted in, recorded in *skipped*.

    Skipping is the right behaviour (one corrupt file must not fail a whole
    queue scan), but a skip with no return channel is a silent discard: the
    caller cannot tell "this queue holds nothing" from "this queue holds
    something I could not read".  *skipped* is that channel, and it is what
    lets a consumer report the loss as structured payload rather than leaving
    it in a log line only a human tailing stderr will ever see (INV-2,
    ``structured-facts-at-failure``).

    An out-parameter is a deliberate divergence from the sibling readers in
    this subsystem (``_index_escalations``, ``_read_limits``), which return a
    tuple when they have a second result to report.  It predates the corpus
    walk, when this reader had a second caller that wanted none of this.

    Args:
        esc_dir: Path to the escalation queue root directory.
        skipped: Optional accumulator for the files this call dropped.  When a
            list is passed, one record per skipped file is **appended** to it
            as ``{'path': Path, 'error': str}`` — appended, not assigned, so a
            single list can span several queue dirs across several calls.
            ``path`` stays a ``Path`` (the caller may want ``.name`` or a retry
            read); stringify at the payload boundary, not here.  The default
            ``None`` leaves behaviour verbatim as it was before this parameter
            existed.  The ``WARNING`` log is emitted either way, so it remains
            the signal for a caller that does not opt in.

    Returns:
        List of escalation dicts (fields passed through unchanged).  Unaffected
        by *skipped*: opting in adds a report, it never changes what is read.
    """
    if not esc_dir.is_dir():
        return []

    results: list[dict] = []
    for path in esc_dir.glob('esc-*.json'):
        try:
            results.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning('Failed to load escalation %s: %s', path, exc)
            if skipped is not None:
                skipped.append({'path': path, 'error': str(exc)})
            continue
    return results


Owner = TypeVar('Owner')


def resolve_owning_project(
    esc: Mapping[str, Any],
    candidates: Sequence[tuple[Owner, Path, Sequence[Mapping[str, Any]]]],
) -> Owner | None:
    """Resolve the owning project for a reconciliation escalation.

    Two-pass probe (first match wins in each pass):

    1. **Worktree prefix** — if ``esc['worktree']`` is a path under ``root``
       (i.e. ``Path(worktree).is_relative_to(root)``), that candidate owns it.
       This matches the conventional ``.worktrees/<task_id>`` layout (where
       ``.worktrees/`` lives inside ``root``) as well as any other subdirectory.
    2. **Task-map probe** — if ``str(esc['task_id'])`` matches ``str(t['id'])``
       for any task *t* in a candidate's task rows, that candidate owns it.

    Args:
        esc: Escalation dict (fields ``worktree`` and ``task_id`` are probed).
        candidates: Ordered ``(owner, root_path, task_rows)`` triples.  The
            first matching candidate wins, so callers should put the primary
            root first.

    Returns:
        The matching candidate's *owner*, exactly as passed, or ``None`` if
        neither probe finds a match.  Returning the owner rather than a name
        keeps two roots that share a basename distinct.
    """
    worktree = esc.get('worktree') or ''

    # Pass 1: worktree prefix matching
    # Use Path.is_relative_to (Python ≥ 3.9) rather than raw str.startswith to
    # avoid false positives where a sibling root name is a string-prefix of
    # another (e.g. "workspace" matching ".../workspace-2/...") or where a
    # sibling directory suffix causes a false ".worktrees-archive" → ".worktrees"
    # match.  is_relative_to compares path *components*, not raw characters.
    #
    # .resolve(strict=False) canonicalises the worktree string (e.g. resolves
    # symlink segments) so it matches the resolved roots from DashboardConfig.
    # strict=False avoids FileNotFoundError for worktrees that have been deleted.
    #
    # is_relative_to(root) alone is sufficient: the conventional `.worktrees/`
    # directory lives *inside* root, so any path under root/.worktrees/ is
    # also relative to root.  A separate `or is_relative_to(root / '.worktrees')`
    # arm would be logically redundant (it matches a strict subset of the first arm).
    if worktree:
        wt = Path(worktree).resolve(strict=False)
        for owner, root, _task_rows in candidates:
            if wt.is_relative_to(root):
                return owner

    # Pass 2: task-map probe
    task_id = str(esc.get('task_id', ''))
    if task_id:
        for owner, _root, task_rows in candidates:
            if any(str(t.get('id')) == task_id for t in task_rows):
                return owner

    return None


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

_LEVEL_KEYS = (0, 1, 2)
_STATUS_KEYS = ('pending', 'resolved', 'dismissed')


def _bucket(escalations: list[dict], *, skipped_count: int) -> dict:
    """Compute per-subsection summary counts from a list of escalation dicts.

    Only ``level`` values in ``{0, 1, 2}`` and ``status`` values in
    ``{pending, resolved, dismissed}`` are counted.  Unknown values are
    silently ignored (the escalation is still present in the subsection's
    ``escalations`` list).

    Args:
        escalations: The escalations this subsection actually loaded.
        skipped_count: How many files in this queue could **not** be
            read, and therefore how short the ``by_level``/``by_status`` counts
            beside it may be.  Passed through verbatim so the annotation lives
            in the same dict as the counts it qualifies, read by the same
            consumers at the same nesting.  Required, deliberately: a default of
            ``0`` would let a future call site that forgot the kwarg report a
            clean queue it never checked, which is the silent degradation this
            whole field exists to prevent.  A call site that genuinely has no
            skips says ``skipped_count=0`` and means it.

    Returns:
        ``{"by_level": {0: int, 1: int, 2: int},
           "by_status": {"pending": int, "resolved": int, "dismissed": int},
           "skipped_count": int}``
    """
    by_level: dict = {k: 0 for k in _LEVEL_KEYS}
    by_status: dict = {k: 0 for k in _STATUS_KEYS}
    for esc in escalations:
        lvl = esc.get('level')
        if lvl in by_level:
            by_level[lvl] += 1
        st = esc.get('status')
        if st in by_status:
            by_status[st] += 1
    return {'by_level': by_level, 'by_status': by_status, 'skipped_count': skipped_count}


def _merge_summaries(summaries: list[dict]) -> dict:
    """Merge a list of per-subsection summary dicts into one aggregate.

    ``skipped_count`` aggregates here alongside the level/status counts, so the
    top-level rollup comes free from the one call this function already has —
    there is no second aggregation path to keep in sync (INV-5).  It reports how
    many files across **all** queues could not be read, and therefore
    how short the merged ``by_level``/``by_status`` counts beside it may be.

    ``skipped_count`` is indexed, not ``.get(..., 0)``-ed, to match the adjacent
    ``by_level``/``by_status`` handling: a summary dict missing the key is
    malformed, and defaulting it would turn "this summary was built by a path
    that forgot to count skips" into a confident "0 unreadable" — precisely the
    silent-zero the key exists to eliminate.  Fail loud on a ``KeyError``
    instead (INV-11, ``no-silent-fail-soft``).
    """
    by_level: dict = {k: 0 for k in _LEVEL_KEYS}
    by_status: dict = {k: 0 for k in _STATUS_KEYS}
    skipped_count = 0
    for s in summaries:
        for k in _LEVEL_KEYS:
            by_level[k] += s['by_level'].get(k, 0)
        for k in _STATUS_KEYS:
            by_status[k] += s['by_status'].get(k, 0)
        skipped_count += s['skipped_count']
    return {'by_level': by_level, 'by_status': by_status, 'skipped_count': skipped_count}


def _reconciliation_owners(
    orchestrators: Sequence[QueueRef],
    active_rows: Mapping[str, Sequence[dict]],
) -> Callable[[dict], QueueRef | None]:
    """Resolve a reconciliation row to the orchestrator queue that owns its task, or None.

    The probe population is each root's ACTIVE rows, never its whole tree
    (PRD decision 12): a task active in no root has no owner.
    """
    candidates = [
        (queue, Path(queue.id).resolve(strict=False), active_rows.get(queue.id, ()))
        for queue in orchestrators
    ]
    return lambda esc: resolve_owning_project(esc, candidates)


def build_escalation_queues(
    corpus_datum: Datum[EscalationCorpus],
    *,
    active_rows: Mapping[str, Sequence[dict]],
) -> dict:
    """The Escalations tab's queues, one subsection per corpus queue in corpus order.

    Each subsection::

        {
            "id", "label", "kind",
            "escalations": [row, ...],   # the live queue: ROOT-location records only
            "skipped":     [{"path": str, "error": str, "location": "root"|"archive"}],
            "summary":     {"by_level", "by_status", "skipped_count"},
            "views":       {EscalationView: Datum[int]},
        }

    A row is the escalation's own fields plus ``project`` (the owning root's
    label) and ``project_root`` (its id), both ``None`` when no owner is
    found. An orchestrator queue owns its rows; a reconciliation row is
    resolved by :func:`resolve_owning_project` against *active_rows*, keyed by
    orchestrator queue id.

    The archive is not listed: it feeds the views, not the table.
    """
    corpus = corpus_datum.value
    scans = corpus.scans if corpus is not None else ()
    orchestrators = [scan.queue for scan in scans if scan.queue.kind is QueueKind.ORCHESTRATOR]
    reconciliation_owner = _reconciliation_owners(orchestrators, active_rows)

    subsections: list[dict] = []
    for scan in scans:
        rows: list[dict] = []
        for record in scan.records:
            if record.location is not Location.ROOT:
                continue
            esc = record.escalation.to_dict()
            owner = (
                scan.queue if scan.queue.kind is QueueKind.ORCHESTRATOR
                else reconciliation_owner(esc)
            )
            rows.append({
                **esc,
                'project': None if owner is None else owner.label,
                'project_root': None if owner is None else owner.id,
            })
        skipped = [
            {'path': u.path, 'error': u.error, 'location': u.location.value}
            for u in scan.unreadable
        ]
        subsections.append({
            'id': scan.queue.id,
            'label': scan.queue.label,
            'kind': scan.queue.kind.value,
            'escalations': rows,
            'skipped': skipped,
            'summary': _bucket(rows, skipped_count=len(skipped)),
            'views': views_over(corpus_datum, [scan.queue.id]),
        })

    return {
        'subsections': subsections,
        'summary': _merge_summaries([s['summary'] for s in subsections]),
        'views': views_over(corpus_datum, [scan.queue.id for scan in scans]),
    }


# ---------------------------------------------------------------------------
# Task cards
# ---------------------------------------------------------------------------

_NO_OWNER_REASON = (
    'no owning project (no worktree under a configured root and the task is '
    'active in none)'
)


def _card_ref(row: Mapping[str, Any]) -> TaskRef | str:
    """The task a row's card names, or the reason it names none."""
    task_id = row.get('task_id')
    if task_id is None or str(task_id) == '':
        return 'no task id'
    if not str(task_id).isdecimal():
        return f'task id is not a number: {task_id!r}'
    if row.get('project_root') is None:
        return _NO_OWNER_REASON
    return TaskRef(row['project_root'], int(task_id))


def _rows(queues: Mapping[str, Any]) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for sub in queues['subsections']:
        for row in sub['escalations']:
            yield sub['id'], row


def card_task_refs(queues: Mapping[str, Any]) -> set[TaskRef]:
    """Every task a row of *queues* names that a lookup can answer."""
    return {ref for _sub_id, row in _rows(queues) if isinstance(ref := _card_ref(row), TaskRef)}


def card_datums(
    queues: Mapping[str, Any],
    lookup: Mapping[TaskRef, Datum[dict]],
) -> dict[tuple[str, Any], Datum[dict]]:
    """Each row's task card, keyed by ``(queue id, escalation id)``.

    The lookup's answer when it has one; otherwise an ``unknown`` Datum whose
    reason says why this row has no card.
    """
    cards: dict[tuple[str, Any], Datum[dict]] = {}
    for sub_id, row in _rows(queues):
        ref = _card_ref(row)
        if isinstance(ref, str):
            card = unknown_datum(ref, FETCHED_ROW_FRESHNESS_BOUND_SECONDS)
        elif (found := lookup.get(ref)) is not None:
            card = found
        else:
            card = unknown_datum(
                f'task {ref.task_id} was not looked up', FETCHED_ROW_FRESHNESS_BOUND_SECONDS,
            )
        cards[(sub_id, row['id'])] = card
    return cards


# ---------------------------------------------------------------------------
# pins_recovery fan-out (task 3543, spec S8)
# ---------------------------------------------------------------------------

_PINS_DEFAULT_PER_CALL_TIMEOUT = 2.0


def _pins_map_from_records(records: list, base_url: str) -> dict[str, list[str]]:
    """Project a ``get_pending_escalations`` list into ``{esc_id: pins_recovery}``.

    A record is included ONLY when it carries a usable ``pins_recovery`` list.
    Three kinds of record are dropped, all for the same reason:

    * no ``pins_recovery`` key — a pre-3543 escalation server that never
      computed the annotation, or a server that computed it and deliberately
      OMITTED it because the task-status read was unavailable (the omission
      contract in ``get_pending_escalations``' docstring);
    * a non-list ``pins_recovery`` (including ``None``) — malformed;
    * a non-dict record, or a record with no ``id`` — ragged input.

    Dropping is not the same as defaulting to ``[]``.  An id absent from the
    returned map is UNKNOWN, and every consumer must render it as nothing.
    Defaulting would manufacture a confident "this record pins nothing" out of
    a server that never said so — the exact false negative (esc-3163) the
    escalation side refuses to emit.
    """
    pins: dict[str, list[str]] = {}
    dropped = 0
    for rec in records:
        if not isinstance(rec, dict):
            dropped += 1
            continue
        esc_id = rec.get('id')
        if not isinstance(esc_id, str) or not esc_id:
            dropped += 1
            continue
        if 'pins_recovery' not in rec:
            # Deliberately absent (unknown) — not an error, not a zero.
            continue
        value = rec['pins_recovery']
        if not isinstance(value, list):
            dropped += 1
            continue
        pins[esc_id] = [str(t) for t in value]
    if dropped:
        logger.debug(
            'get_pending_escalations from %s: dropped %d malformed record(s)',
            base_url, dropped,
        )
    return pins


async def _fetch_pins_one(
    client: httpx.AsyncClient,
    base_url: str,
    timeout: float,
) -> dict[str, list[str]] | None:
    """Read one escalation server's pins_recovery annotations.

    Returns ``None`` for UNKNOWN (transport failure, timeout, or a result that
    is not the tool's declared ``list`` shape) and a dict — possibly empty —
    for a successful read.
    """
    try:
        result = await asyncio.wait_for(
            mcp_tool_call(
                client, base_url, 'get_pending_escalations', {'compact': True},
                timeout=timeout,
            ),
            timeout=timeout,
        )
    except (TimeoutError, httpx.HTTPError, OSError, ValueError) as exc:
        logger.debug('get_pending_escalations failed for %s: %s', base_url, exc)
        return None

    if isinstance(result, list):
        return _pins_map_from_records(result, base_url)

    # Not a list.  The one benign case is an EMPTY dict: fastmcp encodes an
    # empty list return as an empty ``content`` array (tools/base.py
    # ``_convert_to_content`` — the ``all(isinstance(...))`` guard is vacuously
    # true for ``[]``), and the shared ``_extract_tool_result`` collapses empty
    # content to ``{}``.  So a project with zero pending escalations arrives
    # here as ``{}`` and is an authoritative empty read, not an unknown one.
    #
    # Caveat, recorded rather than papered over: ``_extract_tool_result`` also
    # returns ``{}`` when the inner text fails to parse.  That path logs its own
    # WARNING from dashboard.data.memory, so the loss is visible; disambiguating
    # it would mean widening that shared helper's return contract, which is out
    # of scope here.
    if isinstance(result, dict) and not result:
        return {}
    logger.debug(
        'get_pending_escalations from %s returned %s, not a list — treating as '
        'unknown', base_url, type(result).__name__,
    )
    return None


async def fetch_pins_recovery(
    client: httpx.AsyncClient,
    escalation_urls: dict[str, str],
    *,
    per_call_timeout: float = _PINS_DEFAULT_PER_CALL_TIMEOUT,
) -> dict[str, dict[str, list[str]] | None]:
    """Fan ``get_pending_escalations`` out to every escalation URL concurrently.

    Returns ``{project_label: {escalation_id: [task_ids]} | None}``, with the
    labels taken verbatim from *escalation_urls* (project basenames, matching
    ``shape_merge_queue`` and ``get_merge_halt_status``).  Every configured
    label is always present in the result: one project's failure never sinks
    another's, because each probe is isolated.

    The value is THREE-state, mirroring
    :attr:`escalation.pins.PinReport.store_unavailable`:

    * ``None`` — this project could not be read (transport error, timeout, a
      non-list result such as an error envelope, or an unanticipated exception
      escaping the probe).  UNKNOWN.
    * ``{}`` — the read succeeded and no record carried an annotation (zero
      pending escalations, or a pre-3543 server).
    * ``{id: [task_ids]}`` — the read succeeded and these records are annotated.

    Callers must not collapse the first into the second.  An empty map reads as
    "nothing pins this", which is precisely the failure (esc-3163) that routes
    a genuinely-pinned strand down the wrong branch.  The same discipline holds
    per RECORD: an id absent from a non-``None`` map is unknown, never "does
    not pin" — see :func:`_pins_map_from_records`.

    ``compact=True`` is requested because ``_COMPACT_PENDING_FIELDS`` on the
    escalation side exists for this caller: it re-adds ``pins_recovery`` on top
    of the shared compact projection so the heavy free-text fields this
    function never reads stay off the wire on every poll.

    The default ``mcp_tool_call`` timeout is 10s — too slow for a polling loop —
    so each call is wrapped in ``asyncio.wait_for`` and worst-case latency stays
    close to *per_call_timeout* even when every orchestrator is down.
    """
    if not escalation_urls:
        return {}
    labels = list(escalation_urls.keys())
    urls = [escalation_urls[lbl] for lbl in labels]
    base_urls = [u.removesuffix('/mcp').rstrip('/') for u in urls]
    # return_exceptions=True is what makes the per-project isolation promised
    # above actually hold.  `_fetch_pins_one` catches the transport family it
    # can anticipate ((TimeoutError, httpx.HTTPError, OSError, ValueError)), but
    # `mcp_tool_call` reaches `McpSession.call_tool`, whose failure modes are
    # not contractually narrowed to those — a RuntimeError from session-state
    # handling, say, or a TypeError while unwrapping an envelope.  With
    # return_exceptions=False such an escape propagates out of the gather and
    # app.py's outer `except Exception` blanks the annotation for EVERY
    # project at once, which is precisely the sinking-the-whole-fleet failure
    # this fan-out exists to prevent.  Mapped to None, it degrades exactly one
    # project to UNKNOWN instead.
    settled = await asyncio.gather(
        *(_fetch_pins_one(client, base, per_call_timeout) for base in base_urls),
        return_exceptions=True,
    )
    results: list[dict[str, list[str]] | None] = []
    for label, outcome in zip(labels, settled, strict=True):
        if isinstance(outcome, BaseException):
            logger.debug(
                'pins_recovery probe for %s raised %s: %s — treating that '
                'project as unknown', label, type(outcome).__name__, outcome,
            )
            results.append(None)
        else:
            results.append(outcome)
    return dict(zip(labels, results, strict=True))
