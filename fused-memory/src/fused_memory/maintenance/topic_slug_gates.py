"""The consolidation-gate half of ``scripts/normalize_topic_slugs.py`` (task 4878).

Reads every consolidation gate out of the named task stores through an
injected MCP client, and pairs each planned slug move with the one gate filed
against its legacy slug, so a slug's memories and its gate move as a pair.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fused_memory.maintenance.topic_slug_plan import Rename
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.topic_slug import is_valid_topic_slug


@dataclass(frozen=True)
class GateGroup:
    """Every rename on one slug, plus the gate task that must move with them.

    The unit of atomicity.  Renames are grouped by ``(project_id, old_topic)``
    rather than handled one record at a time because the gate's closure scroll
    addresses a TOPIC, not a record: moving four of a slug's five records
    splits the cluster just as badly as moving none of them.

    Attributes:
        project_id: The corpus this slug lives in.
        old_topic / new_topic: The slug's move, shared by every member.
        renames: The frozen plan rows, in plan order.
        gate_task_id: The consolidation gate filed against *old_topic*, or
            ``None`` — most topics are not gates, and an ungated group needs
            no MCP handshake at all.
        gate_project_root: The gate task's project root (tasks are addressed
            by root, memories by project id — a genuinely different store).
        gate_block: The PATCHED ``x_recon_consolidation_gate`` block: the
            live one with ``topic`` moved and every sibling key carried across
            verbatim.  Built at plan time so the artifact shows exactly what
            ``--apply`` will send.
    """

    project_id: str
    old_topic: str
    new_topic: str
    renames: tuple[Rename, ...]
    gate_task_id: str | None = None
    gate_project_root: str | None = None
    gate_block: dict[str, Any] | None = None


@dataclass(frozen=True)
class GateCensus:
    """Every consolidation gate the task stores hold, and what could not be read.

    A value rather than a bare list because "no gate was found" and "no store
    was read" must never share a representation: an empty list said both, and
    that ambiguity is how the live path came to treat every slug as ungated.

    Attributes:
        roots: The project roots attempted, deduplicated, in the order given.
        tasks: The gate tasks of the covered projects, in the shape
            :func:`pair_gate_blocks` consumes, sorted by ``(project_id, id)``.
        covered_projects: The project ids whose one task store answered.
        failures: One ``gate_census_incomplete`` entry per root that could not
            be read, or whose store contradicted another root's.
    """

    roots: tuple[str, ...]
    tasks: tuple[dict, ...]
    covered_projects: frozenset[str]
    failures: tuple[dict, ...]


def _census_failure(root: str, error: str, **extra) -> dict:
    return {'reason': 'gate_census_incomplete', 'project_root': root, **extra, 'error': error}


def _unreadable_listing(response: object) -> str | None:
    """Why a ``get_tasks`` answer is not a task listing, or ``None`` if it is."""
    if not isinstance(response, dict):
        return f'get_tasks answered a {type(response).__name__}, not a task listing'
    if (
        isinstance(response.get('tasks'), list)
        and isinstance(response.get('project_id'), str) and response['project_id']
        and isinstance(response.get('project_root'), str) and response['project_root']
    ):
        return None
    if response.get('error') or response.get('error_type'):
        return f'{response.get("error_type") or "error"}: {response.get("error")}'
    return 'get_tasks answered without a task list, project_id and project_root'


def _gate_tasks_in(listing: dict) -> list[dict]:
    gates: list[dict] = []
    for task in listing['tasks']:
        metadata = task.get('metadata') if isinstance(task, dict) else None
        block = metadata.get(GATE_METADATA_KEY) if isinstance(metadata, dict) else None
        if not isinstance(block, dict):
            continue
        gates.append({
            'id': str(task.get('id')),
            'project_id': listing['project_id'],
            'project_root': listing['project_root'],
            'metadata': {GATE_METADATA_KEY: block},
        })
    return gates


async def census_consolidation_gates(client, roots) -> GateCensus:
    """Read every consolidation gate out of each root's task store.

    One ``get_tasks`` per DISTINCT root, through the same client that later
    patches the gates, so a run cannot pair against one task store and patch
    another.  Each root's call is individually guarded: a timed-out or refused
    root becomes one failure entry instead of unwinding the census, because
    the census's whole contract is to state what it could NOT see.

    A refusal is detected STRUCTURALLY — a missing or non-list ``tasks``, or
    a missing ``project_id``/``project_root`` — because
    ``FusedMemoryClient.call_tool`` raises only on a JSON-RPC-level error; a
    tool-level refusal arrives as an ordinary dict.

    ``project_id`` and ``project_root`` are taken from the RESPONSE as-is.
    The server derives the id with ``resolve_project_id`` over the
    main-checkout-normalized root, which is the derivation
    ``TaskInterceptor._consolidation_closure_error`` scrolls under, so the
    pairing key and the closure key cannot diverge.  Two roots the server
    normalizes to one store are one store read twice; two DIFFERENT stores
    answering for one project id are a contradiction this census cannot
    settle, so that project is left uncovered.

    Why no paging: ``page_size`` slices a FULL server-side re-read on every
    page, and an offset shifts when a task is removed between pages, so a gate
    could be skipped with no error.  One call is one SQL SELECT, an atomic
    snapshot.  Measured 2026-09-23 through ``FusedMemoryClient``: dark-factory
    5,747 tasks, 32.3 MB, 11.4 s, 17 gates; reify 7,760 tasks, 33.1 MB, 5.8 s,
    40 gates — both inside the client's 30 s timeout, and a timeout degrades
    into a failure entry, never into a silent gap.
    """
    attempted = tuple(dict.fromkeys(roots))
    gates: list[dict] = []
    failures: list[dict] = []
    # project_id -> (the store that answered for it, the root it was asked as)
    stores: dict[str, tuple[str, str]] = {}
    contradicted: set[str] = set()
    for root in attempted:
        try:
            listing = await client.call_tool('get_tasks', {'project_root': root})
        except Exception as exc:
            failures.append(_census_failure(root, f'{type(exc).__name__}: {exc}'))
            continue
        problem = _unreadable_listing(listing)
        if problem is not None:
            failures.append(_census_failure(root, problem))
            continue
        project_id, store = listing['project_id'], listing['project_root']
        first_store, first_root = stores.setdefault(project_id, (store, root))
        if first_store != store:
            contradicted.add(project_id)
            failures.append(_census_failure(
                root,
                f'ambiguous task store: {root!r} answered as project '
                f'{project_id!r} from {store!r}, but {first_root!r} answered '
                f'as {project_id!r} from {first_store!r}',
                project_id=project_id,
            ))
        elif first_root == root:
            gates.extend(_gate_tasks_in(listing))

    covered = frozenset(stores) - contradicted
    return GateCensus(
        roots=attempted,
        tasks=tuple(sorted(
            (gate for gate in gates if gate['project_id'] in covered),
            key=lambda gate: (gate['project_id'], gate['id']),
        )),
        covered_projects=covered,
        failures=tuple(failures),
    )


def gate_topic(task: dict) -> object:
    """The ``topic`` of a gate task's block, or ``None`` when it carries no block."""
    block = (task.get('metadata') or {}).get(GATE_METADATA_KEY)
    return block.get('topic') if isinstance(block, dict) else None


def pair_gate_blocks(renames, gate_tasks) -> tuple[list[GateGroup], list[dict]]:
    """Pair each slug's renames with the consolidation gate filed against it.

    Pure.  Takes the plan rows and the gate tasks as plain dicts (``{'id',
    'project_id', 'project_root', 'metadata'}`` — the shape
    :func:`census_consolidation_gates` produces) and returns
    ``(groups, skips)``.

    ``metadata.x_recon_consolidation_gate`` is a Tier-C block on TASK
    metadata, read by ``reconciliation/consolidation_gate.py`` and enforced on
    the ``done`` transition by ``middleware/task_interceptor.py``.  Its
    ``topic`` is what the closure scroll matches on, so a gate left behind on
    a slug this sweep vacated becomes permanently uncloseable — the trap this
    task exists to remove, re-created at a new address.

    The patch moves ``topic`` and NOTHING else: ``provenance``,
    ``considered_and_kept`` and any future sibling are copied across verbatim.
    A block rebuilt from scratch would silently drop the provenance a curator
    needs to adjudicate the gate.  The source task is never mutated — the
    planner's whole job is to leave the corpus untouched.

    A renamed slug carrying MORE THAN ONE gate is refused as
    ``gate_ambiguous``: no group is built, so neither store moves.  A
    :class:`GateGroup` moves exactly one gate, and
    ``fused-memory/src/fused_memory/maintenance/topic_slug_lockstep.py::rename_group``'s
    commit-point argument — the single gate patch is the one atomic write of
    the pair — depends on that.  Moving one of several gates would strand the
    rest on the vacated slug, the same failure as moving none.  Refusing
    leaves both stores on the legacy slug for hand resolution, the posture
    ``slug_collision`` already takes.  It costs nothing today: no
    non-conforming gate topic in either live store carries more than one gate
    (measured 2026-09-23).

    An unmatched gate on a NON-conforming slug is reported as
    ``orphan_gate_topic``: this sweep found no live record carrying it, so
    renaming would not help and staying silent would hide a gate that is
    already uncloseable.  An unmatched gate on a CONFORMING slug is neither
    work nor an orphan and is reported nowhere — nothing to migrate is not
    the same fact as a dangling gate.
    """
    # (project_id, topic) -> the gate tasks carrying it, lowest id first so
    # every verdict below is a property of the corpus rather than of read order.
    gates_by_topic: dict[tuple[str, str], list[dict]] = {}
    for task in gate_tasks or ():
        topic = gate_topic(task)
        if not isinstance(topic, str) or not topic:
            continue
        key = (str(task.get('project_id') or ''), topic)
        gates_by_topic.setdefault(key, []).append(task)
    for bucket in gates_by_topic.values():
        bucket.sort(key=lambda t: str(t.get('id')))

    by_slug: dict[tuple[str, str], list[Rename]] = {}
    for rename in renames or ():
        by_slug.setdefault((rename.project_id, rename.old_topic), []).append(rename)

    groups: list[GateGroup] = []
    ambiguous: list[dict] = []
    for key, members in sorted(by_slug.items(), key=lambda kv: (kv[0][0], kv[1][0].new_topic)):
        project_id, old_topic = key
        bucket = gates_by_topic.get(key) or []
        if len(bucket) > 1:
            ambiguous.append({
                'reason': 'gate_ambiguous',
                'project_id': project_id,
                'old_topic': old_topic,
                'new_topic': members[0].new_topic,
                'gate_task_ids': [str(task.get('id')) for task in bucket],
                'memory_ids': sorted(r.memory_id for r in members),
                'note': (
                    'more than one consolidation gate is filed against this '
                    'legacy slug, and moving one would strand the others on '
                    'the vacated slug — neither store was touched; resolve the '
                    'duplicate gates by hand before re-running'
                ),
            })
            continue
        gate = bucket[0] if bucket else None
        block: dict[str, Any] | None = None
        if gate is not None:
            live_block = gate['metadata'][GATE_METADATA_KEY]
            # Copy, then move exactly one key: the source task stays as read.
            block = {**live_block, 'topic': members[0].new_topic}
        groups.append(GateGroup(
            project_id=project_id,
            old_topic=old_topic,
            new_topic=members[0].new_topic,
            renames=tuple(members),
            gate_task_id=None if gate is None else str(gate.get('id')),
            gate_project_root=None if gate is None else gate.get('project_root'),
            gate_block=block,
        ))
    groups.sort(key=lambda g: (g.project_id, g.new_topic))

    # A gate on a slug this sweep renames was paired or refused above; only a
    # gate whose slug no rename reached can be an orphan.
    orphans: list[dict] = []
    for (project_id, topic), bucket in gates_by_topic.items():
        for task in bucket:
            if (project_id, topic) in by_slug:
                continue
            if is_valid_topic_slug(topic):
                continue
            orphans.append({
                'reason': 'orphan_gate_topic',
                'project_id': project_id,
                'gate_task_id': str(task.get('id')),
                'gate_project_root': task.get('project_root'),
                'gate_topic': topic,
                'note': (
                    'a consolidation gate carries a non-conforming slug that '
                    'THIS sweep matched to no live record, so normalizing the '
                    'corpus will not close it — the gate needs hand work '
                    '(check the under_enumerated bucket before concluding the '
                    'cluster is genuinely empty)'
                ),
            })
    orphans.sort(key=lambda s: (s['project_id'], s['gate_topic'], s['gate_task_id']))
    return groups, ambiguous + orphans
