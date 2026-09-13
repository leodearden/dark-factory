"""Terminal-subject recon-escalation reaper — DETECTION only (task 3052).

THE DEFECT.  ``stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog``
files a ``reconciliation_stale_gate_backlog`` L1 for a gate task, but only
while that task is selected by
``stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids``, which
requires ``task['status'] == 'blocked'``.  The moment the subject goes
terminal — ``done``/``cancelled`` — or disappears from the task store, the
record's whole premise ("a human decision is still awaited on this blocked
gate") is false.  It cannot ever re-file, and nothing closes it, so it sits
pending forever.  The sibling category
``reconciliation_stale_human_operator`` (``::maybe_escalate_stalled_tasks``)
has the identical lifecycle and the identical defect.

WHY THIS MODULE ONLY DETECTS.  The A7b escalation-closure contract, stated
verbatim in the comment above ``reconciliation/harness.py::_RECON_DEDUP_CONFIG``:
"The reconciliation harness NEVER calls queue.resolve() ON THE RECON
ESCALATION QUEUE ... The watcher session (port 8103) is the sole closer of
recon escalations."  So this module computes the reap set and emits a Stage-1
flag naming it; the CLOSE is performed by the watcher session, or by an
operator running ``fused-memory/scripts/derive_orphaned_recon_escalations.py
--apply``.  Keeping detection and repair in different actors is the same
discipline ``cli_stage_runner.py`` applies when it denies Stage 3
``repair_memory_citation``.

WHY NOT THE ORCHESTRATOR'S EXISTING SWEEP.  ``orchestrator/harness.py``
already auto-closes a terminal-subject escalation with
``resolution_class='moot-terminal-subject'`` — but it is structurally blind to
these records on two independent grounds.  It reads the orchestrator's OWN
``<project_root>/data/escalations`` queue, not
``config.escalation_queue_dir``; and it returns early on
``getattr(esc, 'level', None) != 2`` BEFORE
``config.escalation_revalidation_allowlist`` is ever consulted, whereas every
reconciliation-filed stale record is born at ``level=1`` (re-verified live
2026-09-02: 124 of 124 pending ``reconciliation_stale_gate_backlog`` records
are ``level=1``).  Widening that allowlist therefore cannot reach them, which
is why the detection lives here instead.

WHY REAPING IS SANCTIONED, NOT A POLICY CHANGE.
``skills/recon-escalation-watcher/SKILL.md``'s playbook row for this category
makes PARK the default — correctly, because resolving a still-``blocked``
subject's record re-arms the filing rule and produces measured churn
(``esc-650-1`` -> ``esc-650-2`` in ~4h).  But the same row already carves out
the exit: "**Resolve only** when the underlying task will genuinely stop
qualifying for re-selection."  A terminal or absent subject is provably that
case — selection requires ``status == 'blocked'`` — so a reap here cannot
churn.  The gap this module closes is that nobody COMPUTED which pending
records fall in that already-sanctioned branch.

Design decisions (captured in plan.json):

- The classifier is compared against EACH RECORD'S OWN project's task store,
  resolved through ``BaseStage.known_projects``.  A record whose
  ``project_id`` cannot be parsed, or that names a project absent from that
  map, is counted ``unresolvable`` and is NEVER called an orphan — classifying
  a foreign record against the querying project's census would tell the sole
  closer to resolve a record whose subject may still be legitimately
  ``blocked``.
- The per-project census is CROSS-TAG-COMPLETE (``list_tags`` then one
  ``get_statuses_fresh`` per tag, merged).  ``get_statuses_fresh`` defaults to
  a single tag — see
  ``backends/task_backend_protocol.py::list_tags`` — so a single untagged read
  would classify a subject living in another tag as ``missing`` and reap a
  possibly-still-``blocked`` record.
- ``get_statuses_fresh`` rather than ``get_statuses``: it opens its own
  short-lived autocommit connection per call and so can never be pinned to a
  stale WAL read-snapshot (task 2388), the same reason
  ``harness.py::_fetch_task_count_census`` chose it.
- Best-effort, and fail-SAFE in ONE direction: an errored read is never
  evidence of terminality.  A false ``terminal`` hands the sole closer a live
  record; a missed detection merely waits for the next cycle.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# The two recon stale families this reaper covers.  Both are pending L1
# records filed per-subject-task by Stage 1 whose premise a terminal subject
# genuinely moots, both write ``project_id:`` as their first detail line, and
# ``skills/recon-escalation-watcher/SKILL.md`` already treats their playbook
# rows identically ("Same aging/park shape as reconciliation_stale_gate_backlog
# above").  ``reconciliation_stale_human_operator`` has ZERO pending records
# today (live census 2026-09-02), so its inclusion is pure future-proofing —
# which is why an emitted flag always names the record's own category, so the
# watcher lands on the right playbook row rather than assuming gate-backlog.
REAPABLE_STALE_CATEGORIES: frozenset[str] = frozenset({
    'reconciliation_stale_gate_backlog',
    'reconciliation_stale_human_operator',
})

# Subject statuses from which a task can never return to ``blocked`` and so
# can never re-qualify for re-selection by
# ``stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids``.
# ``deferred`` is deliberately NOT here: a deferred task can be un-deferred
# back into ``blocked``, so reaping its record would re-arm the filing rule.
TERMINAL_TASK_STATUSES: frozenset[str] = frozenset({'done', 'cancelled'})

# Stage-1 flag identity.  Together with the str-coerced subject ``task_id``
# these form ``flag_dedup.compute_flag_signature``'s key, so the flag earns a
# ``stage1_flag_marker`` recurrence row and honours explicit suppression
# instead of re-emitting unmarked every cycle.
ORPHANED_ESCALATION_FLAG_TYPE = 'orphaned_recon_escalation'

# A member of ``cli_stage_runner.FINDING_ITEM_SCHEMA``'s nine-value category
# enum.  The defect is a disagreement between two DISTINCT stores — the recon
# escalation queue (a JSON queue directory) and the subject project's task
# store — which is exactly what ``cross_store_inconsistency`` names.  The
# curator-gate sweep's ``task_memory_mismatch`` would be wrong here: no memory
# is involved at all, so copying it would mislead a reader grepping by
# category.
ORPHANED_ESCALATION_FLAG_CATEGORY = 'cross_store_inconsistency'

# The detail-block key both producers write.  Compared case-sensitively and
# anchored to the start of a stripped line so a ``project_id`` mention inside
# a free-text ``description:`` line cannot be mistaken for the field.
_PROJECT_ID_DETAIL_KEY = 'project_id:'


def escalation_project_id(esc):
    """Return the subject ``project_id`` parsed out of *esc*'s detail block.

    DELIBERATE INV-2 EXCEPTION.  ``escalation.models.Escalation`` has no
    ``project_id`` field (verified: zero occurrences in
    ``escalation/src/escalation/models.py``), so there is no structured fact
    to read and the value must be recovered from prose.  Adding the field
    would help only FUTURE records; the entire population this reaper exists
    to clear is the records already on disk, which would still need parsing.
    This function is therefore the SINGLE owner of that parse — the in-cycle
    sweep and ``scripts/derive_orphaned_recon_escalations.py`` both call it,
    so the rule cannot drift into two copies that disagree.

    The line is written by both producers as the FIRST entry of their
    ``detail_parts`` list —
    ``stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog`` and
    ``stage1_stall_detector.py::maybe_escalate_stalled_tasks`` — but position
    is not relied upon here: the first line whose stripped form starts with
    ``project_id:`` wins.  Empirical basis for treating the parse as total: 0
    of 124 live pending records fail it, across both observed detail vintages
    (``age_hours_at_filing:`` and the older ``age_hours:`` shape still carried
    by ``esc-5943-1``).

    Splits on the FIRST colon only, so a value that itself contains a colon is
    returned whole rather than silently truncated (a truncated id would miss
    ``known_projects`` and be counted ``unresolvable`` — fail-safe, but an
    avoidable recall loss).

    Returns ``None`` when *esc* has no readable ``detail``, or its detail
    carries no ``project_id:`` line, or the parsed value is empty.  NEVER
    raises: detail is deserialised from JSON on disk, so a malformed record
    must degrade to ``unresolvable`` rather than abort the sweep for every
    other record.

    Pure: no I/O, no side effects.
    """
    detail = getattr(esc, 'detail', None)
    if not isinstance(detail, str):
        return None
    for raw_line in detail.splitlines():
        line = raw_line.strip()
        if not line.startswith(_PROJECT_ID_DETAIL_KEY):
            continue
        value = line[len(_PROJECT_ID_DETAIL_KEY):].strip()
        return value or None
    return None


def select_reapable_escalations(escalations):
    """Return the elements of *escalations* this reaper may consider, in order.

    All three conditions must hold; widening any of them would hand the sole
    closer records it has no sanction to close:

    - ``category in REAPABLE_STALE_CATEGORIES`` — other categories (notably
      ``recon_integrity_issue``) have a different lifecycle in which a
      terminal subject does not moot the record.
    - ``status == 'pending'`` — an already-resolved or dismissed record is
      closed; re-closing it is at best a no-op and at worst re-archives it.
    - ``level == 1`` — the level every reconciliation-filed stale record is
      born at.  An L2 record of this category would have been promoted by a
      human-facing path (``promote_to_l2``) and belongs to the orchestrator's
      own revalidation sweep, so this reaper stays out of it.

    A non-``Escalation`` element (or one missing any of these attributes) is
    skipped rather than raising — one malformed row must not cost the whole
    selection.

    Pure: no I/O, no side effects.  Empty input returns ``[]``.
    """
    selected = []
    for esc in escalations:
        if getattr(esc, 'category', None) not in REAPABLE_STALE_CATEGORIES:
            continue
        if getattr(esc, 'status', None) != 'pending':
            continue
        if getattr(esc, 'level', None) != 1:
            continue
        selected.append(esc)
    return selected


def _observed_statuses(esc, statuses):
    """Every status *statuses* records for *esc*'s subject, or ``None`` if absent.

    A census value is either a bare status string (the legacy flat shape) or a
    collection of statuses — one per tag that carries this id (see
    ``_project_status_census``).  ``isinstance(value, str)`` is tested FIRST
    because a ``str`` is itself iterable: read as a collection, ``'done'``
    would explode into ``{'d', 'o', 'n', 'e'}`` and render every flat census
    ambiguous, silently reducing recall to zero.

    The id lookup is ``str``-coerced on BOTH sides: censuses come back
    ``{id_str: status}`` but task ids arrive as ints on some paths, and an
    un-coerced lookup would report a ``done`` subject as ``missing`` — the
    reap decision would coincide, but the EVIDENCE handed to the closer would
    be false, which is exactly the failure the sibling sweeps' "not proof"
    discipline exists to prevent.

    Pure: no I/O, no side effects.
    """
    tid = str(getattr(esc, 'task_id', None))
    by_str = {str(k): v for k, v in statuses.items()}
    if tid not in by_str:
        return None
    value = by_str[tid]
    return {value} if isinstance(value, str) else set(value)


def sole_subject_status(esc, statuses):
    """The subject's single observed status, or ``None`` if absent or ambiguous.

    The one owner of "what status do we report as evidence", shared by the
    Stage-1 flag's prose and the operator script's resolution note so the two
    can never describe the same record differently.  ``None`` is returned
    exactly when there is no single status to name — which is why neither
    caller has to know the census's internal value shape.

    Pure: no I/O, no side effects.
    """
    observed = _observed_statuses(esc, statuses)
    if observed is None or len(observed) != 1:
        return None
    return next(iter(observed))


def classify_orphan(esc, statuses):
    """Classify *esc* against *statuses*, its own project's status census.

    Returns one of:

    - ``'terminal'`` — the subject's sole status is in
      ``TERMINAL_TASK_STATUSES``.  The record is moot and provably cannot
      churn if closed: selection requires ``status == 'blocked'``.
    - ``'missing'`` — the subject has no row in the census at all.  The
      caller MUST have built that census cross-tag-complete (see
      ``_project_status_census``), because "absent from a single tag" is not
      "absent from the store", and reaping on the weaker signal could close a
      record whose subject is still ``blocked`` in another tag.
    - ``'ambiguous'`` — the id carries MORE THAN ONE distinct status, i.e. it
      exists in several tags and they disagree.  Ids are per-tag
      (``PRIMARY KEY (tag, id)`` with a per-tag ``id_counters`` high-water
      mark in ``backends/sqlite_task_backend.py``), so every tag numbers from
      1 and a collision is the norm rather than a coincidence; an escalation
      record carries no tag, so the subject cannot be identified.  Two
      DIFFERENT terminal statuses are ambiguous too — ``done`` here and
      ``cancelled`` there are still two different tasks, and naming either as
      the observed status would fabricate the evidence even where the reap
      verdict happened to coincide.
    - ``'live'`` — any other sole status, including ``blocked`` (the state
      that re-qualifies the subject for re-selection) and ``deferred`` (which
      can return to ``blocked``).  A ``'live'`` record is never flagged.

    Only ``'terminal'`` and ``'missing'`` are flaggable.

    *statuses* must be a successfully-read census.  An errored or partial read
    must never reach here: it would render as ``missing`` for every record.
    That guarantee does NOT rest on the backend raising —
    ``backends/sqlite_task_backend.py::get_statuses_fresh`` never does, it
    "fails open to ``{}`` on any error" — so ``_project_status_census``
    additionally rejects an EMPTY read as a failed one, and returns ``None``
    for the whole project rather than a census this function could mistake
    for evidence of absence.

    Pure: no I/O, no side effects.
    """
    observed = _observed_statuses(esc, statuses)
    if not observed:
        return 'missing'
    if len(observed) > 1:
        return 'ambiguous'
    if next(iter(observed)) in TERMINAL_TASK_STATUSES:
        return 'terminal'
    return 'live'


def build_orphaned_escalation_flag(
    esc,
    classification,
    *,
    subject_project_id,
    subject_status,
):
    """Build the Stage-1 flag announcing that *esc* is an orphaned record.

    Stage 1 holds no authority to close a recon escalation — the A7b contract
    above ``reconciliation/harness.py::_RECON_DEDUP_CONFIG`` reserves that for
    the port-8103 watcher session — so this flag's job is to hand that closer
    a re-derivable finding, and to route it to the playbook branch that
    already sanctions the close.

    The description states ONLY what was OBSERVED (mirroring
    ``curator_gate_resolution_sweep.py::build_gate_resolution_flag``'s
    discipline): the record's own fields, plus the subject's status read off
    that project's cross-tag-complete census microseconds earlier in the same
    sweep.  It never asserts that the record has been or will be closed — the
    record is still pending when this flag is written, and the closer is a
    different actor entirely.  ``suggested_action`` therefore carries the
    no-churn argument explicitly: without it, the watcher's documented PARK
    default is the correct reading of its own playbook and the flag would
    rightly be ignored.

    ``flag_type``/``category`` are the module constants and ``task_id`` is the
    ``str``-coerced SUBJECT task id.  Together those are
    ``flag_dedup.compute_flag_signature``'s key, so an un-actioned orphan
    gains a ``stage1_flag_marker`` recurrence row and honours explicit
    suppression instead of re-emitting unmarked every cycle.  Keying on the
    escalation id instead would make every re-file of the same subject look
    like a brand-new finding.

    Args:
        esc: The pending ``Escalation`` being reported.
        classification: ``'terminal'`` or ``'missing'`` — the result of
            ``classify_orphan``.
        subject_project_id: The project whose task store was consulted.  Named
            in the description so "no row" is read relative to a specific
            store rather than as "this task does not exist anywhere".
        subject_status: The observed status for ``'terminal'``; ``None`` for
            ``'missing'``.

    Raises:
        ValueError: for any *classification* other than ``'terminal'``/
            ``'missing'`` — notably ``'live'``.  A live record must never
            reach the closer, since resolving a still-``blocked`` subject's
            record re-arms the filing rule and reproduces the measured
            re-file churn (``esc-650-1`` -> ``esc-650-2`` in ~4h), so a wiring
            mistake is made loud here rather than silently forwarded.

    Pure: no I/O, no side effects.
    """
    if classification not in ('terminal', 'missing'):
        raise ValueError(
            f'build_orphaned_escalation_flag: classification must be '
            f"'terminal' or 'missing', got {classification!r}; a 'live' or "
            'unresolvable record must never be handed to the closer',
        )

    tid = str(getattr(esc, 'task_id', None))
    esc_id = getattr(esc, 'id', None)
    category = getattr(esc, 'category', None)

    if classification == 'terminal':
        observation = (
            f'subject task {tid} is {subject_status} (terminal) in '
            f"{subject_project_id}'s task store"
        )
    else:
        observation = (
            f'subject task {tid} has no row in '
            f"{subject_project_id}'s task store (checked across every tag)"
        )

    description = (
        f'Pending level-1 escalation {esc_id} (category {category}) is '
        f'orphaned: {observation}. That record is filed only while its '
        "subject is status == 'blocked' "
        '(stage1_stall_detector.py::extract_stalled_gate_backlog_task_ids), '
        'so its premise no longer holds and it can never be re-filed, yet '
        'nothing closes it — the reconciliation harness never resolves its '
        'own escalation queue (A7b), and the orchestrator revalidation sweep '
        'reads a different queue and only considers level == 2.'
    )

    return {
        'description': description,
        'severity': 'minor',
        'actionable': False,
        'task_id': tid,
        'flag_type': ORPHANED_ESCALATION_FLAG_TYPE,
        'category': ORPHANED_ESCALATION_FLAG_CATEGORY,
        'suggested_action': (
            f'For the port-8103 watcher session (the sole closer of recon '
            f'escalations): resolve {esc_id} with '
            "resolution_class='moot-terminal-subject'. This is the playbook's "
            'existing "Resolve only when the underlying task will genuinely '
            'stop qualifying for re-selection" branch, not its PARK default: '
            "re-selection requires status == 'blocked', so a terminal or "
            'absent subject provably cannot cause a re-file. First VERIFY the '
            f"observation against {subject_project_id}'s task store; if task "
            f'{tid} is in fact still blocked (or lives in a tag that was not '
            'read), dismiss this flag and leave the record pending — closing '
            'a live record re-arms the filing rule.'
        ),
    }


def _merge_tag_census(census, per_tag):
    """Fold one tag's ``{id: status}`` map into the accumulating census.

    Accumulates a SET per id rather than overwriting, because ids are per-tag:
    ``PRIMARY KEY (tag, id)`` with a per-tag ``id_counters`` high-water mark
    (``backends/sqlite_task_backend.py``) means every tag numbers its tasks
    from 1, so two tags holding the same id is the norm.  A last-tag-wins
    ``dict.update`` would silently resolve that collision — and an escalation
    record carries no tag, so there is nothing to resolve it WITH.
    ``classify_orphan`` reports a multi-status id as ``'ambiguous'`` instead.
    """
    for tid, status in per_tag.items():
        census.setdefault(str(tid), set()).add(status)


async def _project_status_census(taskmaster, project_root, *, log = logger):
    """Return a CROSS-TAG-COMPLETE ``{id: {status, ...}}`` census for *project_root*.

    Underscore-prefixed but deliberately SHARED with
    ``fused-memory/scripts/derive_orphaned_recon_escalations.py``: the
    cross-tag rule below is the same soundness requirement in both call sites,
    and a second copy could drift so that the in-cycle flag and the operator
    reap disagree about which records are safe to close. The name stays
    private because it is not part of the module's Stage-1-facing surface.

    ``taskmaster.get_statuses_fresh`` defaults to a SINGLE tag when none is
    given — stated in
    ``backends/task_backend_protocol.py::list_tags``: "a caller that needs a
    cross-tag-complete view ... must enumerate every tag first via this
    method".  A single untagged read would therefore report a subject living
    in another tag as having no row at all, which ``classify_orphan`` renders
    as ``'missing'`` and this sweep hands to the sole closer as a reap
    instruction — for a task that may still be legitimately ``blocked``.  That
    is the exact false positive the whole design exists to prevent, so the
    tags are enumerated even though every store inspected on 2026-09-02 had
    exactly one (``master``; dark_factory 4958 tasks, reify 7150).

    We deliberately depart here from ``citation_verifier.py``'s "Out of scope"
    note and from ``harness.py::_fetch_task_count_census``'s untagged read:
    for those a missed row merely weakens a check, whereas here it drives an
    IRREVERSIBLE queue mutation by a downstream closer.  Asymmetric cost,
    asymmetric rigour.

    ``ids=`` is never passed, so "absent from the returned map" is an
    unambiguous no-row signal rather than a backend's missing-id convention.
    ``get_statuses_fresh`` is chosen over ``get_statuses`` for the task-2388
    reason ``harness.py::_fetch_task_count_census`` gives: it opens its own
    short-lived autocommit connection per call and so can never be pinned to
    a stale WAL read-snapshot.

    Returns ``None`` when any read failed — the caller must then classify
    NOTHING for that project.  An empty-but-successful tag list falls back to
    one untagged read (the shape a backend with a ``list_tags`` stub
    presents); a ``list_tags`` FAILURE is an error, not a fallback, because
    falling back would perform precisely the single-tag read this helper
    exists to avoid.

    AN EMPTY CENSUS IS ALSO A FAILURE, not an empty store.  The production
    backend never raises: ``backends/sqlite_task_backend.py::get_statuses_fresh``
    "fails open to ``{}`` on any error" — a non-existent DB file, a permission
    error, a disk I/O failure, a corrupt file, an exhausted 5000ms WAL
    ``busy_timeout`` under write contention — and only logs a warning.  The
    except ladders above would therefore never fire for it, and a failed read
    would arrive here as a clean ``{}`` that ``classify_orphan`` renders
    ``'missing'`` for every record.  ``list_tags`` is
    ``SELECT DISTINCT tag FROM tasks``, so every tag it returns has at least
    one row by construction: an empty per-tag map is PROOF of a fail-open
    read.  The emptiness check below is what makes that proof load-bearing.

    We deliberately do NOT try to tell a fail-open apart from the vanishingly
    rare, benign race where a listed tag's last row was deleted between
    ``list_tags`` and the read.  Both resolve to "classify nothing this
    cycle", which costs at most one cycle of detection — the next cycle
    re-checks — whereas the opposite error is an irreversible close of a live
    record by the sole closer.

    ``asyncio.CancelledError``/``KeyboardInterrupt``/``SystemExit`` propagate
    unchanged.
    """
    try:
        tags = await taskmaster.list_tags(project_root)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'orphaned_recon_escalation_sweep: list_tags failed for '
            'project_root=%s — classifying nothing for this project '
            '(a single-tag fallback could reap a live record)',
            project_root,
        )
        return None

    census: dict[str, set[str]] = {}
    try:
        if not tags:
            # No tags reported: one untagged read, held to the same standard.
            # This helper only runs for a project with >=1 pending reapable
            # record naming it, and a store with zero task rows in any tag
            # cannot have produced a gate-backlog escalation naming a
            # `blocked` subject — so an all-empty census here is far likelier
            # an unreadable or not-yet-created DB than ground truth.
            untagged = await taskmaster.get_statuses_fresh(project_root)
            if not untagged:
                log.warning(
                    'orphaned_recon_escalation_sweep: empty untagged census for '
                    'project_root=%s — treating as a FAILED read (get_statuses_fresh '
                    'fails open to {} on any error) and classifying nothing',
                    project_root,
                )
                return None
            _merge_tag_census(census, untagged)
        else:
            for tag in tags:
                per_tag = await taskmaster.get_statuses_fresh(project_root, tag=tag)
                if not per_tag:
                    log.warning(
                        'orphaned_recon_escalation_sweep: empty census for '
                        'project_root=%s tag=%s, which list_tags just reported — '
                        'SELECT DISTINCT tag FROM tasks only yields a tag with >=1 '
                        'row, so this is a fail-open read; classifying nothing for '
                        'this project (a partial census is indistinguishable from a '
                        'failed one)',
                        project_root, tag,
                    )
                    return None
                _merge_tag_census(census, per_tag)
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'orphaned_recon_escalation_sweep: get_statuses_fresh failed for '
            'project_root=%s (tags=%r) — classifying nothing for this project',
            project_root, tags,
        )
        return None

    return census


async def sweep_orphaned_recon_escalations(
    escalation_queue,
    taskmaster,
    known_projects,
    *,
    log = logger,
):
    """Flag every pending recon stale record whose subject went terminal or vanished.

    DETECTION ONLY.  This function never calls ``escalation_queue.resolve()``:
    the A7b contract above
    ``reconciliation/harness.py::_RECON_DEDUP_CONFIG`` makes the port-8103
    watcher session the sole closer of recon escalations.  The emitted flags
    reach that closer through ``skills/recon-escalation-watcher/SKILL.md``;
    an operator can also re-derive and close the same set on demand with
    ``fused-memory/scripts/derive_orphaned_recon_escalations.py --apply``.

    Flow: ``get_pending()`` -> ``select_reapable_escalations`` -> group by
    ``escalation_project_id`` -> one cross-tag-complete census per resolvable
    project -> ``classify_orphan`` per record -> a flag for ``'terminal'`` and
    ``'missing'`` only.

    Args:
        escalation_queue: An ``EscalationQueue`` over the RECON queue dir.
            Only ``get_pending()`` (sync) is called — the queue root, which
            correctly excludes already-archived closed records.
        taskmaster: A ``TaskBackendProtocol`` providing ``list_tags`` and
            ``get_statuses_fresh``.
        known_projects: ``{project_id: project_root}``, i.e.
            ``BaseStage.known_projects``.  A record naming a project absent
            from this map is ``unresolvable``, never an orphan.
        log: Logger to use (default: this module's logger).

    Returns:
        dict with ``flags`` (Stage-1 flag dicts to append to
        ``report.items_flagged``) and int counts ``scanned`` (reapable records
        considered), ``terminal``, ``missing``, ``live``, ``ambiguous``,
        ``unresolvable``, ``errors``.  Every key is always present, so a
        caller never needs a ``.get(..., 0)`` fallback and can read the
        degraded-cycle signature (``errors > 0``) apart from the
        clean-but-empty one directly.

    Best-effort, and fail-SAFE in ONE direction: a queue-read or census-read
    failure is caught, logged, tallied into ``errors``, and NEVER classified
    as ``terminal`` or ``missing``.  The asymmetry matters more here than in
    the sibling sweeps: a false ``terminal`` tells the sole closer to resolve
    a record whose subject is still ``blocked``, re-arming the filing rule and
    producing the measured re-file churn the watcher playbook documents
    (``esc-650-1`` -> ``esc-650-2`` in ~4h), whereas a missed detection is
    simply re-checked next cycle.
    ``asyncio.CancelledError``/``KeyboardInterrupt``/``SystemExit`` are
    re-raised unchanged.

    The per-record loop is sequential rather than an ``asyncio.gather`` so
    per-record error attribution stays exact, mirroring
    ``curator_gate_resolution_sweep.py::sweep_resolved_curator_gates``.
    """
    stats = {
        'flags': [],
        'scanned': 0,
        'terminal': 0,
        'missing': 0,
        'live': 0,
        'ambiguous': 0,
        'unresolvable': 0,
        'errors': 0,
    }

    try:
        pending = escalation_queue.get_pending()
    except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        log.exception(
            'orphaned_recon_escalation_sweep: get_pending failed — '
            'no records classified this cycle',
        )
        stats['errors'] += 1
        return stats

    reapable = select_reapable_escalations(pending)
    if not reapable:
        return stats

    # Census cache keyed by project_id.  A project whose census read failed is
    # cached as None so a second record for the same project neither retries
    # the failing backend nor is silently classified against a partial map.
    censuses: dict[str, dict[str, set[str]] | None] = {}
    unresolved_project_ids: set[str] = set()
    ambiguous_subject_ids: set[str] = set()

    for esc in reapable:
        stats['scanned'] += 1

        project_id = escalation_project_id(esc)
        if project_id is None or project_id not in known_projects:
            # Never an orphan: the subject's own store was never consulted.
            # Folding this into `missing` would reap records on no evidence.
            stats['unresolvable'] += 1
            unresolved_project_ids.add(
                project_id if project_id is not None else '<unparseable>',
            )
            continue

        if project_id not in censuses:
            censuses[project_id] = await _project_status_census(
                taskmaster, known_projects[project_id], log=log,
            )
        census = censuses[project_id]
        if census is None:
            stats['errors'] += 1
            continue

        classification = classify_orphan(esc, census)
        if classification == 'live':
            stats['live'] += 1
            continue
        if classification == 'ambiguous':
            # The id exists in several tags with differing statuses, so the
            # subject cannot be identified from a record that carries no tag.
            # Counted and surfaced below, never flagged.
            stats['ambiguous'] += 1
            ambiguous_subject_ids.add(str(esc.task_id))
            continue

        stats[classification] += 1
        stats['flags'].append(
            build_orphaned_escalation_flag(
                esc,
                classification,
                subject_project_id=project_id,
                subject_status=sole_subject_status(esc, census),
            ),
        )

    if stats['ambiguous']:
        # Sibling of the registry-gap canary below.  An ambiguous record is
        # invisible to the reaper for a reason no operator can see from the
        # stats alone, and the bucket grows the moment a project starts using
        # a second tag — so the colliding subject ids are named here rather
        # than letting recall shrink silently.
        log.warning(
            'orphaned_recon_escalation_sweep: %d of %d reapable record(s) name a '
            'subject id that exists in MORE THAN ONE tag with differing statuses '
            '(subject task ids: %s) — ids are per-tag (PRIMARY KEY (tag, id)) and '
            'a record carries no tag, so the subject cannot be identified; these '
            'were NOT classified and must be resolved by hand',
            stats['ambiguous'], stats['scanned'],
            ', '.join(sorted(ambiguous_subject_ids)),
        )

    if stats['scanned'] and stats['unresolvable']:
        # Registry-gap canary, mirroring sweep_resolved_curator_gates' zero-
        # recall canary.  An unresolvable record is silently invisible to the
        # reaper — its subject is never checked — so a growing bucket is
        # shrinking recall, not a clean result.  The deployed
        # DASHBOARD_KNOWN_PROJECT_ROOTS covers all seven projects present in
        # the live queue and 0 of 124 records fail the detail parse, so a
        # non-empty bucket is a real signal worth grepping for.
        log.warning(
            'orphaned_recon_escalation_sweep: %d of %d reapable record(s) could '
            'not be scoped to a known project and were NOT classified '
            '(project_ids: %s) — a known_projects registry gap or a detail-block '
            'format drift silently shrinks this sweep\'s recall',
            stats['unresolvable'], stats['scanned'],
            ', '.join(sorted(unresolved_project_ids)),
        )

    return stats
