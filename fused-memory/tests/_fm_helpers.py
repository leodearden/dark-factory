"""Shared test helpers for fused-memory tests, imported by name.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.  Each subproject exports its helpers under a unique
module name so test files can `from _fm_helpers import X` without
colliding with sibling subprojects' helpers.

An opt-in fixture defined here applies only to the modules that import it
by name (see `lease_dir_fixture`).

Five siblings follow the same convention, each owning one seam: `_fm_polling`
(wait barriers), `_fm_qdrant_live` and `_fm_falkor_live` (live-backend
scaffolding), `_fm_git_repos` (real-git builders) and `_fm_httpx_leaks`
(leaked httpx client reaping).
"""

import asyncio
import contextlib
import functools
import importlib.util
import inspect
import json
import pathlib
import re
import sqlite3
import subprocess
import sys
import threading
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import RateLimitError
from pydantic import BaseModel
from shared.project_layout import tasks_db

from fused_memory.backends.graphiti_client import (
    INCOMPLETE_KINDS,
    PagedRead,
    ReadCompleteness,
)
from fused_memory.services.durable_queue import DurableWriteQueue

# Constants for the process lifetime — lifted out of pydantic_spec (task 1426)
# to avoid re-computing BaseModel reflection on every call.
_BASEMODEL_PROPS: frozenset[str] = frozenset(
    name for name, v in inspect.getmembers(BaseModel) if isinstance(v, property)
)
_BASEMODEL_ATTRS: frozenset[str] = frozenset(dir(BaseModel))


@functools.cache
def pydantic_spec(model: type[BaseModel]) -> type:
    """Return a proxy class exposing ``model``'s fields for ``MagicMock(spec_set=...)``.

    Pydantic v2 hides field names from ``dir()``, so passing a BaseModel subclass
    directly to ``spec_set=`` would only expose BaseModel/BaseSettings methods.
    The returned proxy has each field name as a class attribute, so MagicMock
    sees them and rejects typos on both get and set.

    User-defined ``@property`` descriptors (e.g. ``OrchestratorConfig.overrides_db_path``)
    are also included in the proxy so ``spec_set`` accepts both read and write.
    BaseModel-inherited properties (``model_extra``, ``model_fields_set``, …)
    are excluded to preserve the invariant that BaseModel API surface is NOT
    exposed.

    User-defined regular methods (callables on ``model`` not inherited from
    ``BaseModel``, not dunder names, not ``@property`` descriptors) are also
    included.  The canonical example is ``OrchestratorConfig.for_module``
    (config.py:991): a plain instance method absent from ``model_fields``.
    This removes the need for the ~18 ad-hoc ``_spec.for_module = None``
    patches that previously worked around the spec_set gap.  The method walk
    covers the full MRO down to (but not including) ``BaseModel``, so methods
    inherited from any intermediate base class (e.g. a shared mixin) are also
    included — spec_set should permit access to any non-BaseModel callable the
    real object exposes.

    Pydantic v2 ``PrivateAttr`` members (e.g. ``OrchestratorConfig._module_configs``
    at config.py:971) are also included by walking ``model.__private_attributes__``.
    PrivateAttr names begin with ``_`` and are therefore excluded by the regular
    method walk's underscore filter; this separate walk re-includes them because
    they are legitimate mock assignment targets.

    BaseModel API (``model_dump``, ``model_validate``, ``model_construct``,
    ``model_copy``, ``model_json_schema``, ``model_post_init``, …) remains
    explicitly excluded via the module-level ``_BASEMODEL_ATTRS`` frozenset
    (``frozenset(dir(BaseModel))``, computed once at import) applied to the
    method walk.  This preserves the invariant established by task 1064: writing
    ``mock.model_dump = ...`` must still raise ``AttributeError``.
    """
    members: dict[str, None] = {f: None for f in model.model_fields}
    # @property descriptors declared on the user's class (e.g.
    # OrchestratorConfig.overrides_db_path) — exclude properties inherited
    # from BaseModel (model_extra, model_fields_set, __fields_set__) so the
    # existing "BaseModel API is not exposed" invariant is preserved.
    for name, _ in inspect.getmembers(model, lambda v: isinstance(v, property)):
        if name in _BASEMODEL_PROPS:
            continue
        members[name] = None
    # User-defined regular methods — exclude dunders, BaseModel API surface, and
    # anything already collected as a @property above.
    # The walk covers the *full* MRO down to (but not including) BaseModel, so
    # methods inherited from any intermediate base class (e.g. a shared mixin
    # between BaseModel and the target model) are also included.  This is
    # intentional: spec_set should permit access to any non-BaseModel callable
    # the real object exposes, including helpers inherited from mixins.
    for name, _ in inspect.getmembers(model, callable):
        if name.startswith('_'):
            continue
        if name in _BASEMODEL_ATTRS:
            continue
        if isinstance(getattr(model, name, None), property):
            # Belt-and-braces: plain properties fail callable() and are already
            # excluded by the predicate above.  This guard catches exotic
            # descriptors that subclass property AND implement __call__, which
            # would slip through the callable() filter but belong in the
            # @property walk (already collected), not here.
            continue
        members[name] = None
    # Pydantic v2 PrivateAttr members — stored in __private_attributes__ dict,
    # NOT in model_fields.  The underscore-name filter above intentionally skips
    # these; walk them separately so PrivateAttrs bypass that filter.
    for name in getattr(model, '__private_attributes__', {}):
        members[name] = None
    return type(
        f'_{model.__name__}Spec',
        (),
        members,
    )


@dataclass
class MockNode:
    """Simulates a Graphiti entity node (source/target of an edge)."""

    name: str
    uuid: str = ''
    labels: list[str] = field(default_factory=list)


@dataclass
class MockEdge:
    """Simulates a Graphiti entity edge returned from add_episode or search."""

    fact: str
    uuid: str = ''
    source_node: MockNode | None = None
    target_node: MockNode | None = None
    source_node_uuid: str = ''
    target_node_uuid: str = ''
    episodes: list[str] = field(default_factory=list)
    valid_at: Any = None
    invalid_at: Any = None
    # graphiti_core's EntityEdge carries `expired_at` (graphiti_core/edges.py).
    # A restored edge keeps it with `invalid_at` cleared; the double carries it so
    # tests can express that signature (task 4714).
    expired_at: Any = None
    created_at: Any = None


@dataclass
class MockAddEpisodeResult:
    """Simulates the AddEpisodeResults returned by Graphiti's add_episode.

    The real AddEpisodeResults class uses 'edges' as the field name.
    We keep 'entity_edges' for backward compat with existing tests that
    construct MockAddEpisodeResult(entity_edges=[...]).

    'nodes' mirrors AddEpisodeResults.nodes (graphiti_core/graphiti.py:111) —
    the entity nodes this episode touched — consumed by the post-write
    node-dedup sweep (MemoryService._dedup_episode_nodes).

    'episode' mirrors AddEpisodeResults.episode — the EpisodicNode that
    graphiti_core actually minted and persisted (task 3561). It is the ONLY
    place the real episode uuid is observable: the uuid the caller passes in
    means "LOAD this existing episode", never "create under this uuid", so
    MemoryService._execute_graphiti_write reads registration identity off
    ``result.episode.uuid``. Defaults to None so the ~10 pre-existing users of
    this dataclass are unaffected; tests that care about episode identity set
    it to e.g. ``SimpleNamespace(uuid='real-uuid')``.
    """

    entity_edges: list[MockEdge] = field(default_factory=list)
    edges: list[MockEdge] = field(default_factory=list)
    nodes: list[MockNode] = field(default_factory=list)
    episode: Any = None

    def __post_init__(self) -> None:
        if self.edges == [] and self.entity_edges:
            self.edges = list(self.entity_edges)


def install_identity_mocks(mock_graphiti: MagicMock) -> None:
    """Stub GraphitiBackend's write-time-identity primitives (task 2198/α) onto a mock.

    ``MemoryService._execute_graphiti_write`` (task 2202/β) wraps its
    add_episode + _reconcile_episode_identity critical section in
    ``async with self.graphiti._identity_lock_for(group_id):``. A bare
    ``MagicMock()._identity_lock_for(...)`` returns a non-async-context-manager
    child mock, which breaks that ``async with`` for every test that builds
    ``svc.graphiti = MagicMock()``. Call this immediately after constructing
    such a mock (before returning it from a fixture) to install:

      - ``_identity_lock_for``: a real lazy per-group_id ``asyncio.Lock``
        registry (same group_id -> same Lock instance across calls),
        mirroring the production contract at graphiti_client.py:255. A
        sync callable (matching the real sync accessor) so
        ``async with mock_graphiti._identity_lock_for(gid):`` works.
      - ``_resolve_or_create_entity``: an ``AsyncMock(return_value=None)`` —
        the no-op default (0 exact-name matches), since most callers only
        need the awaited call to not raise. Tests that care about
        resolve/collapse behavior override this per-test.

    Args:
        mock_graphiti: The ``MagicMock()`` standing in for ``svc.graphiti``.
            Mutated in place; nothing is returned.
    """
    locks: dict[str, asyncio.Lock] = {}

    def _lock_for(group_id: str) -> asyncio.Lock:
        lock = locks.get(group_id)
        if lock is None:
            lock = asyncio.Lock()
            locks[group_id] = lock
        return lock

    mock_graphiti._identity_lock_for = MagicMock(side_effect=_lock_for)
    mock_graphiti._resolve_or_create_entity = AsyncMock(return_value=None)


async def assert_ro_query_only(
    backend,
    make_graph_mock_fn,
    rows: list[list],
    method_name: str,
    *args,
    **kwargs,
) -> MagicMock:
    """Assert that a backend method uses ro_query and never calls query.

    Creates a graph mock via *make_graph_mock_fn*, wires it into
    *backend._driver._get_graph*, invokes the named method, then asserts:
      - graph.ro_query was awaited exactly once
      - graph.query was not awaited at all

    Returns the graph mock so callers can add additional assertions.
    """
    graph = make_graph_mock_fn(rows)
    backend._driver._get_graph = MagicMock(return_value=graph)
    await getattr(backend, method_name)(*args, **kwargs)
    graph.ro_query.assert_awaited_once()
    graph.query.assert_not_awaited()
    return graph


_REBUILD_DETAIL_NO_ERROR = object()  # sentinel — distinguishes "error not provided" from None


def make_rebuild_detail(
    uuid: str,
    name: str,
    *,
    old_summary: str = '',
    new_summary: str = '',
    edge_count: int = 0,
    status: str = 'rebuilt',
    error: Any = _REBUILD_DETAIL_NO_ERROR,
) -> dict:
    """Return a rebuild-detail dict for use in rebuild pipeline tests.

    Pass ``error=None`` (or any value) to include an 'error' key in the
    returned dict.  When omitted, 'error' is absent from the dict.
    """
    d: dict = {
        'uuid': uuid,
        'name': name,
        'old_summary': old_summary,
        'new_summary': new_summary,
        'edge_count': edge_count,
        'status': status,
    }
    if error is not _REBUILD_DETAIL_NO_ERROR:
        d['error'] = error
    return d


def extract_cypher(call_args: Any) -> str:
    """Return the Cypher query string from a mock call_args object.

    Checks positional args[0] first, then falls back to the 'query' keyword
    argument. Returns '' if neither is present.
    """
    if call_args.args:
        return call_args.args[0]
    return call_args.kwargs.get('query', '')


def extract_params(call_args: Any) -> dict:
    """Return the Cypher params dict from a mock call_args object.

    Checks positional args[1] first, then falls back to the 'params' keyword
    argument. Returns {} if neither is present.
    """
    if len(call_args.args) > 1:
        return call_args.args[1]
    return call_args.kwargs.get('params', {})


# ---------------------------------------------------------------------------
# PagedRead test doubles (task 4386)
# ---------------------------------------------------------------------------
#
# The paginated whole-graph reads (`enumerate_all_valid_edges`,
# `enumerate_entity_nodes`) return `(collection, PagedRead)`, and the
# consumers wired in task 4386 project that PagedRead's completeness into
# their own per-cycle stats.  Every stub of one of those methods therefore has
# to supply a second tuple element, across five test modules in two
# directories — so the doubles live here rather than being hand-rolled per
# site, where they would be unreadable and would drift.
#
# `rows` is deliberately always empty: these stand in for the SECOND tuple
# element only.  The collection a consumer actually iterates is the stub's
# FIRST element, while `rows_seen`/`expected_rows` are what a consumer
# projects into stats — so the two must be settable independently.
# ---------------------------------------------------------------------------


def complete_paged_read(
    *, rows_seen: int = 0, expected_rows: int | None = None
) -> PagedRead:
    """A PagedRead reporting a PROVEN-complete enumeration.

    `expected_rows` defaults to `rows_seen`, which is what a healthy read
    looks like: the census and the pages agree.  Pass it explicitly only to
    build a shape the real backend would not produce.
    """
    return PagedRead(
        rows=[],
        complete=True,
        rows_seen=rows_seen,
        expected_rows=rows_seen if expected_rows is None else expected_rows,
        reason=None,
        incomplete_kind=None,
    )


def incomplete_paged_read(
    kind: str,
    *,
    rows_seen: int = 0,
    expected_rows: int | None = None,
    reason: str | None = None,
) -> PagedRead:
    """A PagedRead reporting an INCOMPLETE enumeration of the given kind.

    `kind` is validated against `INCOMPLETE_KINDS` rather than
    taken on trust: a typo'd kind string matches no policy branch, so it would
    quietly behave like a complete read and the test would pass for the wrong
    reason.

    `reason` defaults to a diagnostic string naming the kind and both counts,
    mirroring the real backend's shape closely enough that an assertion on the
    reason reaching an operator-facing message is meaningful.
    """
    assert kind in INCOMPLETE_KINDS, (
        f'unknown incomplete_kind {kind!r}; expected one of '
        f'{sorted(INCOMPLETE_KINDS)}'
    )
    if reason is None:
        reason = (
            f'test double: enumeration reported incomplete '
            f'(incomplete_kind={kind}) with rows_seen={rows_seen} '
            f'expected_rows={expected_rows}'
        )
    return PagedRead(
        rows=[],
        complete=False,
        rows_seen=rows_seen,
        expected_rows=expected_rows,
        reason=reason,
        incomplete_kind=kind,
    )


#: The verdict for a proven-complete read, for the StaleSummaryResult fields a
#: test does not care about (task 4914).
COMPLETE_READ = ReadCompleteness(complete=True, incomplete_kind=None)


# ---------------------------------------------------------------------------
# 8df8bdcd regression scenario builder + shared parse helpers
# ---------------------------------------------------------------------------
#
# Cycle 8df8bdcd: tasks 1355/1361/1369 appeared in Stage 1 output each
# carrying the NEXT task's title in the sorted completion sequence.
# The canonical scenario (task 1379) uses non-consecutive ids in completion
# order 1369→1355→1361 (differs from id-sort order 1355<1361<1369).
#
# Centralised here so the four test suites that cover this contract import
# from a single source of truth instead of each keeping a private copy.
# ---------------------------------------------------------------------------

# Active-task rendered line format: "- [<id>] (<status>) <title> deps=[...]"
ID_TITLE_LINE_RE: re.Pattern[str] = re.compile(
    r'^- \[(\d+)\] \([^)]+\) (.+?) deps=',
    re.MULTILINE,
)

# Provenance-section rendered line format:
#   commit branch:  "- [<id>] <title>" (no trailing token, just EOL or continuation)
#   legacy branch:  "- [<id>] <title> — provenance: ..."
# Capture everything up to the first "—" (em-dash) or end-of-line.
# NOTE: task titles that themselves contain an em-dash ("—") are NOT supported.
# The regex would silently truncate at the first "—", producing a misleading
# mismatch failure rather than a clear message.  Future callers must use only
# em-dash-free titles when relying on PROVENANCE_LINE_RE.
PROVENANCE_LINE_RE: re.Pattern[str] = re.compile(
    r'^- \[(\d+)\] (.+?)(?:\s*—|\s*$)',
    re.MULTILINE,
)

# Canonical titles for the three 8df8bdcd scenario tasks (completion-order list).
_8DF8_IDS_IN_COMPLETION_ORDER = [1369, 1355, 1361]
_8DF8_TITLES = {
    1369: 'Refactor event dispatch to async',
    1355: 'Implement rate limiter middleware',
    1361: 'Add retry logic for database connections',
}
# Per-task provenance metadata mirroring test_stages.py _TASKS fixture:
#   1369 — commit branch
#   1355 — note-only branch
#   1361 — legacy/none (no metadata.done_provenance key)
_8DF8_PROVENANCE: dict[int, dict | None] = {
    1369: {'commit': 'abc123deadbeef'},
    1355: {'note': 'Covered by sibling task 1354'},
    1361: None,  # legacy: no done_provenance
}


def make_8df8_scenario(
    *,
    id_type: type = int,
    status: str = 'done',
    with_provenance: bool = False,
) -> tuple[list[dict], dict]:
    """Return the canonical 8df8bdcd scenario as (tasks, title_by_id).

    Args:
        id_type: ``int`` or ``str`` — coerces task ids and title_by_id keys.
        status: Task status string applied to all three tasks.
        with_provenance: When True, attaches metadata.done_provenance to tasks
            that have a provenance fixture (matching test_stages.py's _TASKS).
            Task 1361 (legacy branch) gets no metadata.done_provenance key.

    Returns:
        A 2-tuple ``(tasks, title_by_id)`` where:
        - ``tasks`` is a list of 3 task dicts in completion order (1369→1355→1361).
        - ``title_by_id`` maps id (coerced to id_type) → title string.
    """
    tasks: list[dict] = []
    for raw_id in _8DF8_IDS_IN_COMPLETION_ORDER:
        task: dict = {
            'id': id_type(raw_id),
            'title': _8DF8_TITLES[raw_id],
            'status': status,
            'dependencies': [],
        }
        if with_provenance:
            prov = _8DF8_PROVENANCE[raw_id]
            if prov is not None:
                task['metadata'] = {'done_provenance': prov}
        tasks.append(task)

    title_by_id: dict = {id_type(raw_id): _8DF8_TITLES[raw_id] for raw_id in _8DF8_IDS_IN_COMPLETION_ORDER}
    return tasks, title_by_id


def parse_rendered_id_title_pairs(rendered: str, kind: str) -> dict[int, str]:
    """Extract {id: title} pairs from a rendered task output string.

    Args:
        rendered: The string output of a formatter (format_task_list,
            format_filtered_task_tree, _render_done_provenance_section, …).
        kind: ``'active'`` to parse active-task lines via ID_TITLE_LINE_RE,
              ``'provenance'`` to parse provenance-section lines via
              PROVENANCE_LINE_RE.  Any other value raises ``ValueError``.

    Returns:
        A dict mapping integer task id → stripped title string.
        Returns an empty dict when no lines match (callers must guard
        against vacuity themselves or use assert_id_title_pairing).
    """
    if kind not in ('active', 'provenance'):
        raise ValueError(f'kind must be active or provenance, got {kind!r}')
    pattern = ID_TITLE_LINE_RE if kind == 'active' else PROVENANCE_LINE_RE
    found: dict[int, str] = {}
    for m in pattern.finditer(rendered):
        found[int(m.group(1))] = m.group(2).strip()
    return found


def assert_id_title_pairing(
    rendered: str,
    title_by_id: dict,
    kind: str,
    *,
    expected_ids: set | None = None,
) -> None:
    """Assert that every rendered id pairs with its OWN title from title_by_id.

    Bundles three checks:
    1. Anti-vacuity: the regex found at least one match (zero matches → fail loudly).
    2. Own-title check: for each found id, rendered title == title_by_id[id].
    3. No-neighbor-bleed / completeness check:
       - When ``expected_ids`` is given, asserts ``found.keys() == expected_ids``
         (all expected ids present, no extra ids).
       - When ``expected_ids`` is ``None``, asserts every found id is present in
         ``title_by_id`` — an id absent from the reference map is treated as a
         stray/bleed id and fails loudly.  Callers must either include all
         plausible rendered ids in ``title_by_id``, or pass ``expected_ids`` to
         restrict the check to a known subset.

    Args:
        rendered: The formatter output string.
        title_by_id: Map of id → expected title.  Keys may be int or str;
            they are compared after normalising both sides to int.
        kind: ``'active'`` or ``'provenance'`` (forwarded to parse helper).
        expected_ids: Optional set of int ids expected to appear.  When None,
            every found id must be present in ``title_by_id`` (neighbor-bleed
            protection is still enforced; see check 3 above).

    Raises:
        AssertionError: On zero matches, wrong title, unexpected ids, or a
            found id absent from ``title_by_id`` (when ``expected_ids`` is None).
    """
    found = parse_rendered_id_title_pairs(rendered, kind=kind)

    # 1. Anti-vacuity
    assert found, (
        f'assert_id_title_pairing: regex matched nothing in rendered output '
        f'(kind={kind!r}) — test would be vacuous.\n'
        f'Rendered:\n{rendered}'
    )

    # Normalise title_by_id keys to int for comparison
    norm_title_by_id: dict[int, str] = {int(k): v for k, v in title_by_id.items()}

    # 2. Own-title check
    for tid, rendered_title in found.items():
        # When no explicit expected_ids is given, treat any id absent from the
        # reference map as a potential stray/bleed id (latent footgun if silent).
        if expected_ids is None:
            assert tid in norm_title_by_id, (
                f'assert_id_title_pairing: id={tid} appeared in rendered output '
                f'but is absent from title_by_id — possible stray/bleed id. '
                f'Add it to title_by_id or pass expected_ids to restrict the check.\n'
                f'Rendered:\n{rendered}'
            )
        if tid in norm_title_by_id:
            expected_title = norm_title_by_id[tid]
            assert rendered_title == expected_title, (
                f'id={tid}: rendered title={rendered_title!r}, '
                f'expected own title={expected_title!r}\n'
                f'Rendered:\n{rendered}'
            )

    # 3. Expected-ids check (no-neighbor-bleed and completeness)
    if expected_ids is not None:
        assert set(found.keys()) == expected_ids, (
            f'Expected ids {expected_ids}, got {set(found.keys())}.\n'
            f'Rendered:\n{rendered}'
        )


async def submit_and_resolve(
    interceptor,
    project_root: str,
    *,
    timeout_seconds: float = 30.0,
    **kwargs,
) -> dict:
    """Submit a task ticket and wait for the worker to resolve it.

    Reconstructs the legacy facade result shape from ``result_json`` so that
    migrated test assertions (``result['id']``, ``result['action']``, etc.)
    remain verbatim.  Designed as a mechanical drop-in for the removed
    ``TaskInterceptor.add_task`` facade in test code.

    Returns:
        The parsed ``result_json`` dict on success (keys: ``id``, ``title``,
        ``action``, etc. — the legacy add_task shape).
        When ``submit_task`` rejects the request (e.g. backlog gate, closed
        server), returns the submit-error dict directly so callers can assert
        on ``result.get('error')`` / ``result.get('error_type')``.

    Raises:
        AssertionError: One of three conditions:

            - ``submit_task`` returned a non-dict value (e.g. ``None``).
              Helper is test-only; loud failure is preferred over silent
              pass-through of an invalid contract.
            - The ticket resolved but the worker never wrote a
              ``result_json`` (row is None or result_json is empty).  The
              message names the ticket id and dumps ``resolve_result`` so
              the failure is diagnosable without digging through logs.
            - ``result_json`` exists but is not valid JSON.

    Args:
        interceptor: A ``TaskInterceptor`` instance (or compatible).
        project_root: Absolute path to the project root.
        timeout_seconds: How long to wait for the worker to resolve the ticket.
            Defaults to 30 s — generous enough for heavy-concurrency tests on
            loaded CI without being an indefinite wait.  Pass a smaller value
            for tests that intentionally exercise timeout paths.
        **kwargs: Forwarded verbatim to ``submit_task``.
    """
    submit_result = await interceptor.submit_task(project_root, **kwargs)
    assert isinstance(submit_result, dict), (
        f'submit_and_resolve: submit_task returned non-dict: {submit_result!r}'
    )
    if 'ticket' not in submit_result:
        return submit_result
    ticket = submit_result['ticket']
    resolve_result = await interceptor.resolve_ticket(
        ticket, project_root, timeout_seconds=timeout_seconds,
    )
    # TODO: reaching into interceptor._ticket_store is private-attribute coupling.
    # If TaskInterceptor ever exposes a stable accessor (e.g. get_ticket_result(ticket))
    # or if resolve_ticket starts returning the parsed result_json directly, prefer that
    # and remove the _ticket_store access here.
    row = await interceptor._ticket_store.get(ticket)
    if row is not None and row.get('result_json'):
        try:
            return json.loads(row['result_json'])
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f'submit_and_resolve: malformed result_json for ticket {ticket!r}: '
                f'{row["result_json"]!r}'
            ) from exc
    raise AssertionError(
        f'submit_and_resolve: ticket {ticket!r} resolved with no result_json '
        f'(resolve_result={resolve_result!r})'
    )


#: The documented-safe MCP tool-response size, in serialised chars. Derived in
#: test_get_statuses_pagination.py::test_auto_page_limit_fits_documented_safe_envelope
#: (see also server/tools.py::_STATUSES_AUTO_PAGE_LIMIT).
MCP_SAFE_RESPONSE_CHARS = 62_000


# ---------------------------------------------------------------------------
# Topic-guard note fixtures (task 3054)
# ---------------------------------------------------------------------------
# One note drives both the pure-matcher assertions in test_config_schema.py
# and the whole-tool assertions in server/test_add_memory_near_duplicate_gate.py,
# so an edit to it moves both suites and neither silently tests a stale note.
# ---------------------------------------------------------------------------

STRADDLING_WRITE_FIXTURE = (
    'Warm-lane reseed gotcha: when a task is dispatched into a recycled warm '
    'lane, the in-worktree .task/plan.json can be a DANGLING absolute symlink '
    'whose target worktrees/.task-meta/<lane>/plan.json was never written or '
    'was scrubbed. The architect then sees no plan and may wrongly call '
    'report_task_already_done instead of replanning.'
)

MERGE_BASE_NEGATIVE_CONTROL_NOTE = (
    'Use git merge-base --is-ancestor <sha> <branch> to test whether a '
    'commit is an ancestor of a branch tip before cherry-picking.'
)


# ---------------------------------------------------------------------------
# Shared xdist worker-id resolver
# ---------------------------------------------------------------------------

def resolve_xdist_worker_id(request_or_session: pytest.FixtureRequest | pytest.Session) -> str:
    """Return this worker's id ('gw0', 'gw1', …) or 'master' — without the xdist PLUGIN.

    Backs the session-scoped ``worker_id`` fixture in ``tests/conftest.py``,
    which exists because ``worker_id`` was otherwise supplied *solely* by
    pytest-xdist (``xdist/plugin.py``).  The offline-deep lane's serial confirm
    re-run appends ``-p no:xdist -o addopts=`` (see
    ``orchestrator/src/orchestrator/verify_cmd.py``), and ``-p no:xdist``
    unregisters the plugin along with its FIXTURES — not just its ``-n`` /
    ``--dist`` CLI options.  Every test requesting ``worker_id`` therefore
    ERRORED at setup with ``fixture 'worker_id' not found`` in that re-run, and
    a developer typing ``pytest -p no:xdist`` locally hit the same wall.

    It DELEGATES to xdist's own ``get_xdist_worker_id`` rather than
    reimplementing its three lines of ``workerinput`` logic.  A conftest
    fixture SHADOWS a same-named plugin fixture, so the shim takes over
    ``worker_id`` for every fused-memory test — including healthy
    ``-n auto --dist loadgroup`` runs, where per-worker isolation is the whole
    point.  Any semantic drift from xdist would silently collapse worker
    namespaces and cause the cross-worker collisions those suffixes exist to
    prevent; delegating makes such drift structurally impossible, which is what
    makes the shadowing provably safe.

    The import is function-local and deliberate: ``-p no:xdist`` unregisters the
    PLUGIN but leaves the MODULE importable, so delegation works in exactly the
    case that is otherwise broken.

    There is deliberately no ``except ImportError: return 'master'`` fallback.
    pytest-xdist is a declared dev dependency here and this project's addopts
    hardcode ``-n auto``, so a venv genuinely missing the module is broken
    rather than a supported configuration — and quietly collapsing every worker
    onto one shared 'master' namespace there would reintroduce precisely the
    collisions above, as flaky live-service tests instead of a clear error.

    Args:
        request_or_session: a pytest ``request`` or ``session`` object — the
            same parameter xdist's own function takes.
    """
    from xdist.plugin import get_xdist_worker_id

    return get_xdist_worker_id(request_or_session)


# ---------------------------------------------------------------------------
# Shared durable-queue retry-budget factory (task 4847)
# ---------------------------------------------------------------------------

def make_retry_budget_queue(tmp_path, execute, **overrides) -> DurableWriteQueue:
    """A DurableWriteQueue with plain budget 5, extended transient budget 12,
    sub-second backoff, and no ``transient_error_names=`` override, so the
    shipped DEFAULT_TRANSIENT_ERROR_NAMES decides which budget applies.

    The single definition that keeps task 3585's and task 3586's attempt
    counts comparable. *overrides* replace any constructor keyword.
    """
    # Annotated: without it pyright joins the heterogeneous literal values into
    # a single narrow type and rejects the ** unpacking below.
    kwargs: dict[str, Any] = dict(
        data_dir=tmp_path / 'queue',
        execute_write=execute,
        workers_per_group=1,
        semaphore_limit=5,
        max_attempts=5,
        retry_base_seconds=0.01,
        retry_max_delay_seconds=0.05,
        write_timeout_seconds=2.0,
        transient_max_attempts=12,
    )
    kwargs.update(overrides)
    return DurableWriteQueue(**kwargs)


# ---------------------------------------------------------------------------
# Shared LoopFreedomProbe loop-freedom oracle (task 5920)
# ---------------------------------------------------------------------------

#: Loop turns LoopFreedomProbe.suspend() yields before recording a held loop.
#: One turn suffices on a free loop, because the marker is queued ahead of the
#: stub's own resumption; the slack only absorbs a future reordering of
#: asyncio's ready queue. It counts loop turns, never seconds.
LOOP_FREEDOM_TURN_BUDGET = 10

#: Seconds an off-loop-thread caller waits for the captured loop to run its
#: marker. A CEILING, not a floor: a free loop answers within microseconds, so
#: only a held loop reaches it, and it bounds how long that failure takes.
LOOP_FREEDOM_CEILING_SECONDS = 30.0


class LoopFreedomProbe:
    """Did the test's event loop stay free while the code under test was paused?

    Construct it inside the test coroutine (it captures the running loop and
    its thread). A stubbed await point calls ``await probe.suspend()``; a
    stubbed blocking call calls ``probe.block()``. Then call
    :meth:`assert_loop_stayed_free`. Each call queues a marker on the captured
    loop and records True iff the marker ran:

    * ``suspend()`` on the captured loop yields a bounded number of loop turns,
      so a call site that resumes the coroutine itself reads False.
    * ``block()``, or ``suspend()`` on another thread's loop, waits for the
      marker up to a ceiling, so a call site that holds the loop's thread while
      that work runs elsewhere reads False. ``block()`` on the loop's own
      thread reads False at once, because it is the holder.

    Every verdict is deterministic and none hangs. It replaces a FLOOR on ticker
    wake-ups inside a wall-clock window, which reds under host/GIL contention
    (task 5920; the same defect
    ``orchestrator/tests/test_verify_ruff_config_boundary.py::TestProbeDoesNotBlockTheEventLoop``
    fixed for task 4520 / esc-4520-6).
    """

    def __init__(self, *, ceiling_seconds: float = LOOP_FREEDOM_CEILING_SECONDS) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self._ceiling_seconds = ceiling_seconds
        self._observations: list[bool] = []

    @property
    def observations(self) -> tuple[bool, ...]:
        return tuple(self._observations)

    async def suspend(self) -> None:
        if asyncio.get_running_loop() is not self._loop:
            self.block()
            return

        marker_ran = False

        def _mark() -> None:
            nonlocal marker_ran
            marker_ran = True

        self._loop.call_soon_threadsafe(_mark)
        for _ in range(LOOP_FREEDOM_TURN_BUDGET):
            if marker_ran:
                break
            await asyncio.sleep(0)
        self._observations.append(marker_ran)

    def block(self) -> None:
        if threading.get_ident() == self._loop_thread:
            self._observations.append(False)
            return

        loop_answered = threading.Event()
        self._loop.call_soon_threadsafe(loop_answered.set)
        self._observations.append(loop_answered.wait(self._ceiling_seconds))

    def assert_loop_stayed_free(self) -> None:
        assert self._observations, 'LoopFreedomProbe: the probe was never reached by a stubbed call'
        assert all(self._observations), (
            'LoopFreedomProbe: the event loop was held while the code under test '
            f'was paused (observations={self.observations})'
        )


# ---------------------------------------------------------------------------
# Claimant-less tasks.db builder
# ---------------------------------------------------------------------------

def make_db_without_claimant_columns(db_path: pathlib.Path, *, status: str = 'pending') -> None:
    """Create a tasks.db in the v1 shape but stamped ``user_version = 2`` (columns absent).

    Simulates a connection whose claimant columns never got ALTERed in —
    e.g. a routine orchestrator restart racing ahead of the fused-memory
    deploy that ships this migration. Opening it runs only the v2->v3
    candidate_key step (the v1->v2 claimant ALTER is gated on ``version < 2``
    and is skipped for an already-v2 DB), so the claimant columns stay
    absent, exercising set_task_status's fail-safe (WARNING, no error) path.

    ``status`` seeds the single row's status column (default ``'pending'``,
    matching every pre-existing caller); pass e.g. ``'done'`` to exercise
    the fail-safe path on an already-terminal row.
    """
    import sqlite3
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            tag           TEXT NOT NULL DEFAULT 'master',
            id            INTEGER NOT NULL,
            title         TEXT NOT NULL,
            description   TEXT,
            details       TEXT,
            test_strategy TEXT,
            status        TEXT NOT NULL,
            priority      TEXT,
            metadata      TEXT,
            updated_at    TEXT NOT NULL,
            PRIMARY KEY (tag, id)
        );
        CREATE INDEX IF NOT EXISTS ix_tasks_status ON tasks (tag, status);
        CREATE TABLE IF NOT EXISTS dependencies (
            tag        TEXT NOT NULL DEFAULT 'master',
            task_id    INTEGER NOT NULL,
            depends_on INTEGER NOT NULL,
            PRIMARY KEY (tag, task_id, depends_on)
        );
        CREATE TABLE IF NOT EXISTS id_counters (
            tag    TEXT NOT NULL DEFAULT 'master',
            max_id INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (tag)
        );
    """)
    conn.execute(
        "INSERT INTO tasks (tag, id, title, status, updated_at) "
        "VALUES ('master', 1, 'stranded-shape task', ?, '2026-01-01T00:00:00.000Z')",
        (status,),
    )
    conn.execute("PRAGMA user_version = 2")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Shared quota/rate-limit error builder (task 2448)
# ---------------------------------------------------------------------------
#
# De-duplicates the openai.RateLimitError builder that was previously
# copy-pasted verbatim in test_memory_service.py and
# tests/server/test_get_entity_tool.py (flagged by reviewer_comprehensive
# amendment pass on task 2448) — a single copy avoids the two drifting apart
# if the openai RateLimitError constructor signature ever changes.
# ---------------------------------------------------------------------------


def _make_rate_limit_error(
    message: str = 'Rate limit exceeded.',
    code: str = 'insufficient_quota',
) -> RateLimitError:
    """Build a real openai.RateLimitError (429 response) for quota-exhaustion tests."""
    request = httpx.Request('POST', 'https://api.openai.com/v1/embeddings')
    response = httpx.Response(429, request=request)
    body = {'message': message, 'type': 'insufficient_quota', 'param': None, 'code': code}
    return RateLimitError(message, response=response, body=body)


# ---------------------------------------------------------------------------
# Leaked TaskInterceptor curator-worker reaper (task 2737)
# ---------------------------------------------------------------------------
#
# Mirrors orchestrator/tests/_orch_helpers.py::reap_leaked_aiosqlite_connections
# (task 2413) and orchestrator/tests/conftest.py::_reap_leaked_merge_workers
# (task 1907): a test that starts a TaskInterceptor per-project curator
# worker (via submit_task or _start_worker_if_needed) and then raises before
# its own inline cleanup runs — or relies on interceptor_with_store's
# teardown, which only cancels workers it can enumerate — orphans a live
# worker task onto that test's function-scoped event loop. The worker blocks
# forever on an empty asyncio.Queue.get(), so nothing else ever cancels it.
# Under asyncio_mode="strict" + `-n auto --dist loadgroup`
# (fused-memory/pyproject.toml), pytest-asyncio then tears down the per-test
# loop with the worker still pending; the resulting task-destroyed /
# loop-closed noise (and any in-flight aiosqlite work it was mid-drain on)
# surfaces as an order/xdist-dependent flake attributed to a later, innocent
# test.
# ---------------------------------------------------------------------------


async def reap_leaked_ticket_workers() -> int:
    """Cancel + drain any orphaned TaskInterceptor._curator_worker task.

    Unlike MergeWorker.run (task 1907), _curator_worker has no
    subprocess/child-loop machinery to gracefully unwind — it just blocks on
    queue.get() — so a bare ``task.cancel()`` + bounded await is sufficient,
    and deliberately avoids calling ``TaskInterceptor.close()`` (which would
    also close the still-fixture-owned ticket store out from under its own
    teardown).

    Best-effort and bounded, and a cheap no-op for the (vast majority of)
    tests that leak no worker. Only the ``asyncio.wait_for`` timeout is
    suppressed (``asyncio.TimeoutError``): the drained worker's own
    exceptions are already captured as results by ``return_exceptions=True``
    in the ``gather``, so nothing else needs catching. A genuine bug inside
    the reaper itself — or a ``CancelledError`` targeting the reaper's own
    task rather than the worker task it is draining — is deliberately left
    to propagate instead of being masked by a broad ``Exception`` /
    ``BaseException`` suppress (loud-over-silent-degradation).

    Returns:
        The number of orphaned worker tasks actually cancelled and drained
        (task.done() confirmed post-drain) — a worker that fails to unwind
        within the bounded timeout is not counted, even though it was
        cancel()-requested.
    """
    reaped = 0
    for task in list(asyncio.all_tasks()):
        if task.done():
            continue
        coro = task.get_coro()
        if not getattr(coro, '__qualname__', '').endswith('TaskInterceptor._curator_worker'):
            continue
        task.cancel()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(task, return_exceptions=True), timeout=10.0,
            )
        if task.done():
            reaped += 1
    return reaped


# ---------------------------------------------------------------------------
# Non-package script loading — the single loader for every test module that
# imports a script by file path. Task 3738 hoisted it here out of two
# independent copies; task 3895 migrated the 42 further copies that had
# accumulated since, and tests/test_script_loader_routing_guard.py holds that
# population at zero.
# ---------------------------------------------------------------------------

# sys.modules keys this helper itself installed. Only these may be REPLACED by
# a later load of a different file under the same key -- see the shadowing
# guard in load_script_module().
_LOADED_SCRIPT_MODULE_NAMES: set[str] = set()


def load_script_module(
    script_path: pathlib.Path, mod_name: str | None = None
) -> types.ModuleType:
    """Load a non-package script (e.g. under ``scripts/``) by file path.

    ``scripts/`` is not a package and its modules are not on ``PYTHONPATH``,
    so this is how a script's pure functions get imported into a test without
    ``sys.path`` pollution. *mod_name* defaults to the file stem and is the
    key the module is registered under in ``sys.modules``.

    An already-loaded module for the SAME file is reused rather than
    re-executed: two independent test modules loading the same script under
    the same *mod_name* would otherwise make the second load silently
    replace the first module object in ``sys.modules`` -- two live module
    objects whose identity depends on collection order. Harmless while only
    pure functions are used, but a latent hazard the moment anything holds
    module state.

    A ``sys.modules`` entry this helper did NOT install is never replaced:
    that raises ``ImportError`` instead. The stem default makes accidental
    collisions easy to write (``scripts/config.py``, ``scripts/utils.py``,
    ``scripts/types.py`` are ordinary script names), and a silent replacement
    persists for the rest of the pytest process, so the damage would surface
    as an unrelated test failing far away. Deliberate key SHARING -- two test
    modules loading the same script, or a test loading a different file under
    a key this helper owns -- is unaffected.
    """
    name = mod_name or script_path.stem
    cached = sys.modules.get(name)
    cached_file = getattr(cached, '__file__', None)
    if cached is not None and cached_file is not None and (
        pathlib.Path(cached_file).resolve() == script_path.resolve()
    ):
        return cached
    if cached is not None and name not in _LOADED_SCRIPT_MODULE_NAMES:
        raise ImportError(
            f'refusing to shadow already-imported module {name!r} '
            f'(from {cached_file!r}) with {script_path}; '
            'pass an explicit mod_name that does not collide'
        )
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    _LOADED_SCRIPT_MODULE_NAMES.add(name)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        # The failed load left nothing installed under *name*, so ownership
        # lapses with it: a later real import of that name must be protected
        # by the guard above rather than treated as this helper's to replace.
        sys.modules.pop(name, None)
        _LOADED_SCRIPT_MODULE_NAMES.discard(name)
        raise
    return module


def make_populated_task_store(project_root: pathlib.Path) -> pathlib.Path:
    """Create the tasks.db *project_root* resolves to, holding one task row.

    This is the shape ``target_store_preflight``'s guard reads -- a ``tasks``
    table with at least one row -- NOT the backend schema. The suites using it
    patch ``SqliteTaskBackend`` out, so the guard is the only reader of the
    file; agreement with the real backend is pinned once, in
    tests/test_target_store_preflight.py.
    """
    db = tasks_db(project_root)
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db)
    try:
        connection.execute('CREATE TABLE tasks (id INTEGER PRIMARY KEY)')
        connection.execute('INSERT INTO tasks (id) VALUES (1)')
        connection.commit()
    finally:
        connection.close()
    return db


def make_zero_byte_task_store(project_root: pathlib.Path) -> pathlib.Path:
    """Create the tasks.db *project_root* resolves to as an EMPTY file.

    The residue shape ``target_store_preflight``'s guard must refuse, and the
    counterpart of :func:`make_populated_task_store`.
    """
    db = tasks_db(project_root)
    db.parent.mkdir(parents=True)
    db.touch()
    return db


_CLEANUP_TEST_COLLECTIONS_SCRIPT = (
    pathlib.Path(__file__).parent.parent / 'scripts' / 'cleanup_test_collections.py'
)


@functools.cache
def _lease_dir_env() -> str:
    """The env var name, read from the reaper that defines it.

    Read rather than spelled out here so a rename of the constant cannot
    leave `lease_dir_fixture` setting a variable nothing consults any more —
    which would look exactly like isolation while every test fell through to
    the real machine-global directory.

    Loaded lazily and cached, because conftest.py and most test modules
    import this module and an eager load would charge every pytest session.
    Only the resulting STRING is kept: the modules that import the fixture
    load the reaper through their own loaders, and holding a second reference
    to a module object they may replace is a hazard with nothing to buy it.
    """
    return load_script_module(_CLEANUP_TEST_COLLECTIONS_SCRIPT).LEASE_DIR_ENV


def isolated_lease_dir(monkeypatch, directory: pathlib.Path) -> pathlib.Path:
    """Point the reaper's lease directory at *directory*, through *monkeypatch*.

    The one mechanism behind `lease_dir_fixture` and
    `lease_isolated_monkeypatch`.  *directory* is not created, for the reason
    `lease_dir_fixture` gives.
    """
    monkeypatch.setenv(_lease_dir_env(), str(directory))
    return directory


@contextlib.contextmanager
def lease_isolated_monkeypatch(
    reaper: types.ModuleType, lease_directory: pathlib.Path,
) -> Iterator[tuple[pytest.MonkeyPatch, list[pathlib.Path]]]:
    """A `pytest.MonkeyPatch.context()` for a recording made at module scope.

    A module-scoped fixture is set up BEFORE the autouse `lease_dir` fixture,
    so a recorder that opened a bare context would lease the machine-global
    directory the live cron reads.  This one yields its monkeypatch with the
    lease directory already isolated to *lease_directory*, together with a
    list that gains the directory every `reaper.hold_lease` inside the
    context resolved, so a recording can carry where its leases really went
    rather than only that it asked.

    *reaper* is the module the code under test resolves, passed in for the
    reason `_lease_dir_env` keeps no module reference of its own.
    """
    held_in: list[pathlib.Path] = []
    hold_lease = reaper.hold_lease

    def _observed_hold_lease(*args, directory=None, **kwargs):
        held_in.append(reaper.lease_dir() if directory is None else pathlib.Path(directory))
        return hold_lease(*args, directory=directory, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        isolated_lease_dir(monkeypatch, lease_directory)
        monkeypatch.setattr(reaper, 'hold_lease', _observed_hold_lease)
        yield monkeypatch, held_in


@pytest.fixture(autouse=True, name='lease_dir')
def lease_dir_fixture(tmp_path, monkeypatch):
    """Point ``DF_EPHEMERAL_COLLECTION_LEASE_DIR`` at a per-test directory.

    A hard isolation boundary, not a convenience.  The lease directory
    ``cleanup_test_collections.lease_dir()`` returns by default is a
    HARDCODED machine-global absolute path (that is the property the guard's
    correctness rests on — see the design note on that function), and it is
    the very directory the live 6-hourly cron reads.  A test that wrote a
    lease into it would hold a real sweep off this host; a test that reaped
    it would unlink the lease of a live bake-off running in another checkout.

    Autouse, and applied to EVERY test in an importing module rather than
    only its lease tests, for exactly that reason: a test that forgets to
    request the isolation must not be able to fall through silently to the
    real directory.  Tests that need the path can still request this fixture
    by name; the directory is not created here, because a lease-dir-absent
    case is one of the behaviours under test.

    Activation is by import, per module
    (``from _fm_helpers import lease_dir_fixture  # noqa: F401``), and NEVER
    from conftest.py: an autouse fixture there would apply to every test in
    the package, redirecting the ``-m integration`` lane away from the
    directory the live cron reads and silently disabling the guard task 4775
    built.  The attribute is ``lease_dir_fixture`` while the fixture NAME
    stays ``lease_dir`` because a module-level ``lease_dir`` binding would be
    shadowed by every ``def test_x(self, lease_dir)`` parameter, which ruff
    reports as F811 once per test.
    """
    return isolated_lease_dir(monkeypatch, tmp_path / 'ephemeral-collection-leases')


def as_async_run_git(side_effect):
    """Adapt a `subprocess.run` side_effect to a `shared.git_async.run_git` fake.

    Task 3778 moved the detector's three git probes off blocking
    `subprocess.run` and onto the async `run_git` helper, which invalidated the
    `patch('subprocess.run', ...)` seam this suite was built on. Rather than
    hand-rewrite ~90 canned git responses, every existing side_effect is passed
    through this one adapter, so the *responses* stay byte-for-byte what they
    were and only the seam changes.

    It faithfully reproduces `run_git`'s contract, which differs from
    `subprocess.run`'s in exactly two ways that matter here:

    - stdout/stderr are `.strip()`ed by `run_git` itself, so the adapter strips
      too (a fake that did not would let a test pass against behaviour the real
      helper cannot produce).
    - a TIMEOUT is RETURNED as `timed_out=True` with a non-zero returncode, not
      raised. `subprocess.run` raises `TimeoutExpired`, so any side_effect that
      raises it is converted here. Every other exception (notably `OSError`)
      propagates, because `run_git` propagates it too.

    Accepts the two shapes `unittest.mock` accepts for `side_effect`: a
    callable, or an exception instance/class to raise.
    """
    from shared.git_async import TIMEOUT_RETURNCODE, GitResult

    def _timed_out(timeout) -> GitResult:
        return GitResult(
            returncode=TIMEOUT_RETURNCODE,
            stdout='',
            stderr=f'timed out after {timeout}s',
            timed_out=True,
        )

    async def fake_run_git(cmd, cwd=None, *, input_text=None, timeout=None):
        # mock's own semantics: a bare exception instance/class means "raise".
        if isinstance(side_effect, BaseException) or (
            isinstance(side_effect, type) and issubclass(side_effect, BaseException)
        ):
            if isinstance(side_effect, subprocess.TimeoutExpired) or (
                side_effect is subprocess.TimeoutExpired
            ):
                return _timed_out(timeout)
            raise side_effect

        try:
            completed = side_effect(list(cmd), timeout=timeout)
        except subprocess.TimeoutExpired:
            return _timed_out(timeout)

        return GitResult(
            returncode=completed.returncode,
            stdout=(completed.stdout or '').strip(),
            stderr=(completed.stderr or '').strip(),
        )

    return fake_run_git
# ---------------------------------------------------------------------------
# Cross-run citation repair fixtures (task 3065)
#
# Shared by BOTH tests/reconciliation/test_citation_repair.py and
# tests/server/test_recon_report_citation_repair.py — one definition here rather
# than a copy in each module, so the journal shape the repair path reads and the
# lookup verdicts it branches on cannot drift between the two suites.
# ---------------------------------------------------------------------------


async def build_journal_with_closed_run(
    tmp_path: Any,
    *,
    run_id: str,
    project_id: str = 'reify',
    status: str = 'completed',
    stage: str = 'memory_consolidator',
    findings: list[dict[str, Any]],
    extra_stage_reports: dict[str, Any] | None = None,
):
    """Open a real ``ReconciliationJournal`` holding one run with ``findings``.

    A REAL journal on a tmp_path SQLite file, not a fake: the repair path's whole
    reason to exist is that a closed run's findings live only in the journal's
    ``runs.stage_reports`` blob, so a test that stubbed the round-trip would stop
    exercising the one thing under test (``StageReport`` parse on read,
    ``model_dump(mode='json')`` re-serialize on write).

    ``findings`` becomes ``stage_reports[stage].items_flagged``.
    ``extra_stage_reports`` is merged in verbatim AFTER that entry, so a caller
    can seed a second real stage or a raw non-``StageReport`` entry
    (``{'_error': {...}}``) to prove the stage scan tolerates it.

    Returns the OPEN journal — the caller closes it (``await journal.close()``).
    """
    from datetime import UTC, datetime

    from fused_memory.models.reconciliation import (
        ReconciliationRun,
        RunStatus,
        RunType,
        StageId,
        StageReport,
    )
    from fused_memory.reconciliation.journal import ReconciliationJournal

    journal = ReconciliationJournal(pathlib.Path(tmp_path))
    await journal.initialize()
    now = datetime.now(UTC)
    # ``status``/``stage`` stay plain ``str`` in the signature so a caller writes
    # the same literal the journal row holds; coerced here because the model
    # fields are StrEnums (pydantic would coerce anyway — this just makes the
    # conversion visible to the type checker rather than implicit).
    await journal.start_run(
        ReconciliationRun(
            id=run_id,
            project_id=project_id,
            run_type=RunType.full,
            trigger_reason='test',
            started_at=now,
            completed_at=None if status == 'running' else now,
            status=RunStatus(status),
        )
    )
    reports: dict[str, Any] = {
        stage: StageReport(
            stage=StageId(stage),
            started_at=now,
            completed_at=now,
            items_flagged=findings,
        )
    }
    reports.update(extra_stage_reports or {})
    await journal.update_run_stage_reports(run_id, reports)
    return journal


class FakeMemoryLookup:
    """Async ``get_memory_by_id`` stub with an explicit per-id verdict.

    The map's value decides the branch, so a test states which of the THREE
    outcomes it means without any implicit default:

    - ``dict``      -> the memory resolves (live);
    - ``None``      -> genuinely absent (the only verdict that licenses a repair);
    - ``Exception`` -> the backend RAISED (unknown, never "absent").

    An id missing from the map resolves to ``None``, matching the real service's
    not-found return. ``calls`` records every ``(project_id, memory_id)`` in
    order, so a test can assert a gate fired BEFORE any lookup was attempted.

    Deliberately exposes NO ``get_memory``: ``citation_repair`` reads the
    replacement's fingerprint off the raw Qdrant payload this returns, not
    through ``MemoryService.get_memory`` (for why, see
    ``fused_memory/reconciliation/citation_repair.py::_fingerprint_from_record``).
    A test that needs to pin that non-call subclasses this and adds a
    recording ``get_memory``.
    """

    def __init__(self, memories: dict[str, Any] | None = None):
        self.memories: dict[str, Any] = dict(memories or {})
        self.calls: list[tuple[str, str]] = []

    async def get_memory_by_id(self, project_id: str, memory_id: str) -> Any:
        self.calls.append((project_id, memory_id))
        verdict = self.memories.get(memory_id)
        if isinstance(verdict, BaseException):
            raise verdict
        return verdict


async def seed_resolved_ticket(
    store: Any,
    db_path: pathlib.Path,
    project_id: str,
    *,
    status: str,
    latency_s: float,
    resolved_at: datetime,
    reason: str | None = None,
) -> str:
    """Submit and resolve one ticket, then pin its age and resolve latency.

    The ticket goes through ``TicketStore``'s public writers, which both stamp
    the wall clock; the timestamps are then rewritten through an independent
    connection to *db_path*, treating the db file as the fixture.
    """
    import aiosqlite

    ticket_id = await store.submit(project_id, '{}')
    await store.mark_resolved(ticket_id, status=status, task_id='1', reason=reason)
    created_at = resolved_at - timedelta(seconds=latency_s)
    async with aiosqlite.connect(str(db_path), timeout=5) as db:
        await db.execute(
            'UPDATE tickets SET created_at = ?, resolved_at = ? WHERE ticket_id = ?',
            (created_at.isoformat(), resolved_at.isoformat(), ticket_id),
        )
        await db.commit()
    return ticket_id
