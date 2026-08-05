"""Non-fixture test helpers for fused-memory tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.  Each subproject exports its helpers under a unique
module name so test files can `from _fm_helpers import X` without
colliding with sibling subprojects' helpers.
"""

import ast
import asyncio
import contextlib
import functools
import inspect
import json
import os
import pathlib
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import RateLimitError
from pydantic import BaseModel

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


@dataclass
class MockAddEpisodeResult:
    """Simulates the AddEpisodeResults returned by Graphiti's add_episode.

    The real AddEpisodeResults class uses 'edges' as the field name.
    We keep 'entity_edges' for backward compat with existing tests that
    construct MockAddEpisodeResult(entity_edges=[...]).

    'nodes' mirrors AddEpisodeResults.nodes (graphiti_core/graphiti.py:111) —
    the entity nodes this episode touched — consumed by the post-write
    node-dedup sweep (MemoryService._dedup_episode_nodes).
    """

    entity_edges: list[MockEdge] = field(default_factory=list)
    edges: list[MockEdge] = field(default_factory=list)
    nodes: list[MockNode] = field(default_factory=list)

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


# ---------------------------------------------------------------------------
# Shared Qdrant test scaffolding (task 2277)
# ---------------------------------------------------------------------------
#
# De-duplicates the _qdrant_available()/QDRANT_URL/skip-marker trio that was
# previously copy-pasted verbatim in test_recon_dedup_premise.py and
# test_mem0_qdrant_integration.py (flagged by esc-2221-26).
# ---------------------------------------------------------------------------

QDRANT_URL = 'http://localhost:6333'


def _qdrant_available() -> bool:
    """Probe whether a Qdrant instance is reachable at QDRANT_URL.

    Imports QdrantClient lazily (rather than at module scope) so that
    importing this module — which conftest.py does every test session —
    never pulls in qdrant_client, and so tests can deterministically patch
    ``qdrant_client.QdrantClient``.
    """
    from qdrant_client import QdrantClient

    try:
        client = QdrantClient(url=QDRANT_URL, timeout=2)
    except Exception:
        return False

    try:
        client.get_collections()
        return True
    except Exception:
        return False
    finally:
        with contextlib.suppress(Exception):
            client.close()


def qdrant_skipif(reason: str = 'Qdrant not reachable'):
    """Return a ``pytest.mark.skipif`` marker gated on `_qdrant_available()`.

    A factory rather than a module-level marker constant: _fm_helpers is
    imported by conftest.py on every test session, so evaluating the probe at
    import time would run it even for test sessions that never touch Qdrant.
    Calling this from a module's ``pytestmark`` instead defers the probe to
    that module's own collection time — identical timing to each file's
    previous local copy.
    """
    return pytest.mark.skipif(not _qdrant_available(), reason=reason)


def ensure_fresh_collection(
    client: Any,
    collection_name: str,
    *,
    size: int,
    distance: Any = None,
) -> None:
    """Idempotently (re)create *collection_name* as an empty size-*size* collection.

    Replaces the fragile ``delete_collection(suppressed) + create_collection``
    pair each integration fixture rolled by hand. Under ``-n auto`` xdist load,
    a prior run's teardown can be swallowed (or a concurrent worker can win the
    create race), leaving a stale collection behind that ``collection_exists()``
    may still report absent — so a naive create then 409-conflicts and fails the
    next run. This helper self-heals that case: if the create raises a 409
    ``UnexpectedResponse`` it deletes the stale collection and creates once more.
    Any other ``UnexpectedResponse`` (e.g. a genuine 500) is re-raised — fail
    loud, no retry.

    Imports ``VectorParams``/``Distance``/``UnexpectedResponse`` lazily inside
    the body, mirroring ``_qdrant_available``'s convention (see its docstring):
    conftest.py imports this module every test session, so a module-scope
    qdrant import would pull qdrant_client into every session and break tests
    that monkeypatch ``qdrant_client.QdrantClient``.

    Args:
        client: A ``qdrant_client.QdrantClient`` (or test double) exposing
            ``collection_exists`` / ``delete_collection`` / ``create_collection``.
        collection_name: The collection to (re)create.
        size: Vector dimensionality for the new collection.
        distance: Optional ``qdrant_client.models.Distance``; defaults to
            ``Distance.COSINE`` when None.
    """
    from qdrant_client.http.exceptions import UnexpectedResponse
    from qdrant_client.models import Distance, VectorParams

    vectors_config = VectorParams(size=size, distance=distance or Distance.COSINE)

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    try:
        client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config,
        )
    except UnexpectedResponse as exc:
        if getattr(exc, 'status_code', None) != 409:
            raise
        # Stale collection survived a swallowed teardown / lost create race:
        # delete it and create once more rather than propagating the conflict.
        client.delete_collection(collection_name)
        client.create_collection(
            collection_name=collection_name, vectors_config=vectors_config,
        )


# ---------------------------------------------------------------------------
# Shared FalkorDB test scaffolding (task 3502)
# ---------------------------------------------------------------------------
#
# Connection constants, reachability probe and skip-marker factory for tests
# that drive a live FalkorDB. Placed directly after the Qdrant section above so
# the two reachability-probe pairs (constants → probe → skipif factory) read as
# one convention. tests/test_falkor_probe_routing_guard.py enforces that no
# test module forks this scaffolding back into itself.
# ---------------------------------------------------------------------------

def _falkor_host_from_env() -> str:
    """Derive the FalkorDB host from FALKOR_HOST, defaulting to localhost.

    A named function rather than an expression inlined at the constant's
    assignment, so the derivation is reachable at call time and can be tested
    under a monkeypatched environment without ``importlib.reload``.
    """
    return os.environ.get('FALKOR_HOST', 'localhost')


def _falkor_port_from_env() -> int:
    """Derive the FalkorDB port from FALKOR_PORT, defaulting to 6379.

    The ``int()`` is load-bearing, not cosmetic: ``os.environ`` yields str, and
    ``FalkorDB(port='6379')`` fails or misbehaves on a str port.

    Unset *or empty* takes the default. Empty is what a CI template like
    ``FALKOR_PORT: ${FALKOR_PORT}`` produces when the variable is not set, and
    it is the same shape ``known_project_roots_from_env`` (models/scope.py)
    already treats as unset.

    A non-empty, non-numeric value is an operator error and raises
    ``RuntimeError`` naming the offending value.

    That raise fires at *import* of this module, and conftest.py imports it on
    every session — so a malformed FALKOR_PORT aborts the whole run, including
    the unit lane, which never touches FalkorDB. This is intentional, and is
    faithful to the behaviour it replaces: each of the six migrated modules
    already did a bare ``int(os.environ.get('FALKOR_PORT', 6379))`` at its own
    import, so a malformed value was always fatal. What changes is (a) the
    message — actionable text naming the offending value, rather than a
    ``ValueError`` traceback into a shared helper — and (b) the blast radius,
    which widens from the six FalkorDB modules to every fused-memory module.
    Deferring the raise into ``_falkor_available()`` would narrow (b), at the
    cost of letting a typo'd port read as "FalkorDB not reachable" and silently
    skip the live lane rather than reporting the typo. Failing loudly is the
    better trade.
    """
    raw = os.environ.get('FALKOR_PORT', '').strip()
    if not raw:
        return 6379
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(
            f'FALKOR_PORT must be an integer, got {raw!r}. Unset it to use the '
            f'default (6379), or set it to the port your FalkorDB listens on.'
        ) from None


# Evaluated once at import; consumers import the constants, not the helpers.
FALKOR_HOST: str = _falkor_host_from_env()
FALKOR_PORT: int = _falkor_port_from_env()


def _falkor_available() -> bool:
    """Probe whether a FalkorDB instance is reachable at FALKOR_HOST:FALKOR_PORT.

    Returns False rather than raising for any failure — construct, query or
    close — and is bounded by ``socket_connect_timeout=2``, so an unreachable
    host costs ~2s at most.

    Uses the falkordb-native sync client rather than ``redis``: redis is not a
    declared dependency of fused-memory, only a transitive of
    graphiti-core[falkordb], so a redis-based probe would break silently if
    falkordb ever switched transport.

    Imports FalkorDB lazily so (a) conftest.py's session-wide import of this
    module does not drag falkordb into sessions that never touch it, and (b)
    the name resolves at call time, which is what lets tests patch
    ``falkordb.FalkorDB`` deterministically. Same convention as
    ``_qdrant_available`` above.

    Deliberately NOT cached: each consuming module pays its own probe at its
    own collection time. Memoising would break the call-time tracking
    ``falkor_skipif`` relies on, and test_integration_marker_real_service.py
    budgets for the per-module cost explicitly.
    """
    from falkordb import FalkorDB

    try:
        client = FalkorDB(
            host=FALKOR_HOST, port=FALKOR_PORT, socket_connect_timeout=2
        )
        try:
            client.select_graph('_probe').query('RETURN 1')
        finally:
            with contextlib.suppress(Exception):
                client.close()
        return True
    except Exception:
        return False


def falkor_skipif(reason: str = 'FalkorDB not reachable'):
    """Return a ``pytest.mark.skipif`` marker gated on `_falkor_available()`.

    A factory rather than a module-level marker constant: conftest.py imports
    this module on every session, so evaluating the probe at import time would
    cost a bounded ~2s connection attempt even for sessions that never touch
    FalkorDB. Called from a consuming module's ``pytestmark``, or as a
    class/function decorator, the probe instead fires at that module's own
    collection time.

    Resolves ``_falkor_available`` through the module global at call time
    rather than capturing it, so the marker tracks whatever the probe
    currently reports.
    """
    return pytest.mark.skipif(not _falkor_available(), reason=reason)


def unique_graph_name(slug: str) -> str:
    """Return a fresh per-run FalkorDB graph name, ``_test_<slug>_<8 hex>``.

    A fresh name is minted on every call so that concurrent test runs —
    pytest-xdist ``-n auto``, or a CI matrix pointed at one shared FalkorDB —
    never select the same graph and wipe each other's fixtures.

    The caller still owns the rest of the lifecycle: the best-effort delete of
    a stale graph left by a prior run, and the teardown ``graph.delete()`` /
    ``client.aclose()`` pair.

    Args:
        slug: Should embed the owning task id, by convention (e.g.
            ``2207_merge_entities``), so a graph orphaned by a killed run is
            traceable back to the test module that created it.
    """
    return f'_test_{slug}_{uuid.uuid4().hex[:8]}'


# ---------------------------------------------------------------------------
# Shared poll_until() helper (task 2377)
# ---------------------------------------------------------------------------
#
# Generalizes the local poll-then-assert precedents in test_durable_queue.py
# (_poll_until_dead and the inline deadline loop) into a single shared
# helper. Converts fixed "sleep(N) then assert background work is done"
# sites — fragile under host CPU oversubscription, since the hardcoded
# window may elapse before the background drainer/worker/watchdog is even
# scheduled — into a bounded poll on the same post-condition. A genuinely
# stuck background task still surfaces as a clear AssertionError once the
# (generous) deadline passes.
# ---------------------------------------------------------------------------

async def poll_until(
    predicate: Callable[[], Any],
    *,
    timeout: float = 10.0,
    interval: float = 0.02,
    message: str | None = None,
) -> Any:
    """Poll *predicate* until it is truthy, returning its truthy result verbatim.

    *predicate* may be a plain sync callable or an async/coroutine function —
    either way it is called with no arguments on each iteration; if the
    result is awaitable (``inspect.isawaitable``), it is awaited before the
    truthiness check. This lets callers poll on state that itself requires an
    await (e.g. ``lambda: queue.get_stats()``).

    Checks *before* sleeping, so an already-true predicate returns on the
    first call with no sleep at all. Uses the asyncio event-loop monotonic
    clock (mirrors the ``_poll_until_dead`` precedent in
    test_durable_queue.py) rather than wall-clock/perf_counter, to avoid the
    same class of timing fragility this helper exists to remove.

    Args:
        predicate: Zero-arg sync or async callable evaluated each iteration.
        timeout: Seconds to keep polling before giving up. Defaults to 10s —
            generous enough to absorb host CPU oversubscription while still
            catching a genuinely-stuck post-condition.
        interval: Seconds to sleep between polls. Defaults to 0.02s.
        message: Custom AssertionError message on timeout. When None, a
            default message naming *timeout* is used.

    Returns:
        The predicate's truthy result, returned verbatim (not coerced to
        ``True``) so callers can chain on it (e.g. a stats dict).

    Raises:
        AssertionError: *predicate* never became truthy within *timeout*
            seconds — signals a genuinely-stuck background task, the same
            failure class the fixed-sleep asserts this helper replaces used
            to (accidentally) surface.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        result = predicate()
        if inspect.isawaitable(result):
            result = await result
        if result:
            return result
        if loop.time() >= deadline:
            raise AssertionError(message or f'poll_until: condition not met within {timeout}s')
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Shared poll_until_stable() settle barrier (task 3697)
# ---------------------------------------------------------------------------
#
# The companion to poll_until above, and deliberately placed beside it so the
# two read as a matched PAIR of barrier primitives rather than two unrelated
# utilities. poll_until is a LIVENESS barrier; this is a SETTLE barrier. Picking
# the wrong one is the defect this exists to prevent -- see the docstring's
# decision rule. Public (no leading underscore) for the reason this module
# already documents for poll_until / ensure_fresh_collection: it is imported
# cross-module.
# ---------------------------------------------------------------------------

#: Cap on the observed-value trail carried in the never-settled AssertionError.
#: A pathological sampler must not be able to produce a multi-megabyte message.
_STABLE_TRAIL_LIMIT = 10


async def poll_until_stable(
    sample: Callable[[], Any],
    *,
    settle: float = 0.2,
    timeout: float = 10.0,
    interval: float = 0.02,
    message: str | None = None,
) -> Any:
    """Poll *sample* until its truthy value STOPS CHANGING, then return it verbatim.

    Choosing between this and :func:`poll_until` — the whole point of having
    both:

    * ``poll_until`` is a **liveness** barrier. Correct when the assertion that
      follows is "X happened".
    * ``poll_until_stable`` is a **settle** barrier. **Required** when the
      assertion that follows is an exact count or an "exactly once" claim,
      because a liveness poll returns at the *first* occurrence and therefore
      structurally cannot observe a duplicate that arrives after it.

    The motivating call site is
    ``test_ticket_worker.py::test_worker_created_path_emits_journal_event``,
    whose ``assert len(task_created_events) == 1`` was gated on
    ``poll_until(lambda: any(...))`` — so the assertion ran the instant the
    first event landed. The duplicate that assertion exists to catch is
    concrete, not hypothetical: ``task_created`` is emitted from two distinct
    code paths (``task_interceptor.py`` lines 3786-3795 and 4164-4173), each
    persisting the terminal ticket row *before* emitting — so no ticket-row
    predicate closes the emission window either, and only a settle window does.

    **The flake asymmetry**, which is what makes adding a settle window safe: a
    too-short *settle* can only cause a false PASS (a missed duplicate), never
    a false FAIL, because a stable value stays stable no matter how starved the
    host is. The only false-FAIL surface is the outer *timeout*, which keeps
    ``poll_until``'s already-proven 10s default. Raising *settle* is therefore
    always safe, and is the correct response to a suspected missed duplicate.

    *sample* may be a plain sync callable or an async/coroutine function —
    either way it is called with no arguments each iteration and, if the result
    is awaitable (``inspect.isawaitable``), awaited before use. A falsy result
    means "not ready yet": polling continues and the settle window does not
    start. Change detection uses ``!=``, and the deadline uses the asyncio
    event-loop monotonic clock (same rationale as ``poll_until``), bounding
    total time *including* settle restarts so a forever-changing value cannot
    spin past *timeout*.

    Args:
        sample: Zero-arg sync or async callable evaluated each iteration.
        settle: Seconds the value must be observed unchanged before it is
            returned. Any change restarts this window. Defaults to 0.2s.
        timeout: Total seconds to keep polling before giving up. Defaults to
            10s, matching ``poll_until``.
        interval: Seconds to sleep between polls. Defaults to 0.02s.
        message: Custom AssertionError message used when the value never
            became truthy at all.

    Returns:
        The settled truthy value, returned verbatim (not coerced to ``True``),
        matching ``poll_until``.

    Raises:
        AssertionError: with two deliberately DISTINGUISHABLE messages, because
            "it never happened" and "it kept changing" demand different
            debugging and must not collapse into one string —

            * never became truthy → the caller's *message* (or a default
              naming *timeout*), matching ``poll_until``'s failure text so the
              common case reads identically;
            * became truthy but never settled → a message saying so and
              naming the observed values (capped at the last
              ``_STABLE_TRAIL_LIMIT`` distinct ones).
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    candidate: Any = None
    have_candidate = False
    stable_since = 0.0
    trail: list[Any] = []
    while True:
        result = sample()
        if inspect.isawaitable(result):
            result = await result
        if result:
            now = loop.time()
            if not have_candidate or result != candidate:
                candidate = result
                have_candidate = True
                stable_since = now
                trail.append(result)
                del trail[:-_STABLE_TRAIL_LIMIT]
            elif now - stable_since >= settle:
                return candidate
        else:
            # Not ready yet — a falsy sample does not start the settle window.
            have_candidate = False
        if loop.time() >= deadline:
            if trail:
                raise AssertionError(
                    f'poll_until_stable: value never settled for {settle}s within '
                    f'{timeout}s; observed (last {len(trail)}): {trail!r}'
                )
            raise AssertionError(
                message or f'poll_until_stable: condition not met within {timeout}s'
            )
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Shared FalkorDB index-readiness barrier (task 3377; extracted from task 3334)
# ---------------------------------------------------------------------------
#
# Extracted from test_falkor_fulltext_integration.py so every live-index test
# module routes through ONE implementation instead of re-forking it. Public
# (no leading underscore) because it is now imported cross-module, matching
# poll_until / ensure_fresh_collection / collection_vector_size.
# tests/test_falkor_index_barrier_guard.py enforces that routing.
#
# IndexHeaderError is public for the same reason: it is the blessed name a
# caller catches to tell "FalkorDB changed shape" apart from "the index never
# became ready". Keeping it module-private would document a distinction across
# the module boundary that no caller could actually reach.
# ---------------------------------------------------------------------------

class IndexHeaderError(AssertionError):
    """``CALL db.indexes()`` did not expose a ``status`` column.

    A distinct type from the timeout AssertionError because the two failure
    modes mean different things and call for different operator actions: a
    timeout is a slow or wedged index build, whereas a missing ``status``
    column means FalkorDB changed its result shape and the barrier cannot be
    trusted at all. Collapsing the second into a generic timeout message would
    hide a silent no-op behind a plausible-looking timeout.

    Subclasses ``AssertionError`` so a caller that only cares that the barrier
    failed can still catch the broad type, while a caller that needs to act on
    the difference has a name to catch. Callers asserting on the *timeout* path
    specifically should match its message (``'not OPERATIONAL'``) rather than
    the bare type, which this subclass also satisfies.
    """


async def await_index_operational(graph, timeout_s: float = 10.0) -> None:
    """Block until every index on ``graph`` reports ``OPERATIONAL``.

    FalkorDB builds indices **asynchronously**. Querying before the index is
    ready silently returns success for a query the engine would otherwise
    reject. Measured while characterising task 3334: a known-unparseable
    fulltext query issued immediately after
    ``CALL db.idx.fulltext.createNodeIndex(...)`` across six fresh graphs gave
    FAIL, OK, OK, FAIL, FAIL, FAIL — **2 of 6 runs falsely reported success**.
    Range indices are built asynchronously too (measured on a 200k-node graph:
    ``'[Indexing] 628/200000: UNDER CONSTRUCTION'``, never operational within
    40 polls), so this barrier is not fulltext-specific.

    The ``status`` column is resolved **by name** from ``result.header`` rather
    than hardcoded to its current position (7 of 9), so a FalkorDB column
    reorder fails loudly here instead of silently reading the wrong column and
    turning the barrier into a no-op.

    Readiness is an **exact** match on ``'OPERATIONAL'`` for every row, never a
    substring test for ``'UNDER CONSTRUCTION'``: the live not-ready value
    carries a varying progress prefix (``'[Indexing] N/M: UNDER
    CONSTRUCTION'``), so matching the not-ready side would have to anticipate
    every status string FalkorDB might ever emit and would treat an
    unrecognised future status as ready. Testing the ready side is fail-closed
    — anything unfamiliar blocks and eventually raises. An empty
    ``result_set`` counts as NOT ready, since ``all()`` over an empty sequence
    is vacuously true and would otherwise return success having gated nothing.

    Polling is delegated to :func:`poll_until` (event-loop monotonic clock,
    check-before-sleep, raise-on-timeout) rather than hand-rolling a second
    deadline loop in the module that hosts it. An already-operational index
    therefore costs exactly one ``CALL db.indexes()`` round-trip.

    Raises on timeout — deliberately. A never-operational index must be a loud
    failure; skipping or passing would reintroduce exactly the false-green this
    barrier exists to remove.

    Measured against FalkorDB module v41800 (4.18.0), whose live header is
    ``['label', 'properties', 'types', 'options', 'language', 'stopwords',
    'entitytype', 'status', 'info']``.

    Args:
        graph: An async FalkorDB graph exposing ``await graph.query(...)``.
        timeout_s: Seconds to wait for every index to become operational.
            Callers with a module-wide ``pytest.mark.timeout`` must keep this
            budget inside it.

    Raises:
        IndexHeaderError: ``CALL db.indexes()`` exposed no ``status`` column —
            raised on the first poll, never retried until the deadline.
        AssertionError: Not every index became ``OPERATIONAL`` within
            *timeout_s*; the message carries the last-seen status strings.
    """
    last_statuses: list[str] = []

    async def _all_operational() -> bool:
        nonlocal last_statuses
        result = await graph.query('CALL db.indexes()')
        header_names = [col[1] for col in result.header]
        if 'status' not in header_names:
            raise IndexHeaderError(
                'CALL db.indexes() has no "status" column; FalkorDB changed its '
                f'result shape (header={header_names}). The index-readiness '
                'barrier cannot be trusted until this is re-verified.'
            )
        status_idx = header_names.index('status')
        last_statuses = [row[status_idx] for row in result.result_set]
        return bool(last_statuses) and all(s == 'OPERATIONAL' for s in last_statuses)

    try:
        await poll_until(_all_operational, timeout=timeout_s, interval=0.05)
    except IndexHeaderError:
        # Shape failure — propagate untouched so it is never reported as (or
        # masked by) a timeout.
        raise
    except AssertionError as exc:
        raise AssertionError(
            f'index not OPERATIONAL within {timeout_s}s '
            f'(last statuses={last_statuses!r})'
        ) from exc


async def collection_vector_size(
    client: Any,
    collection: str,
    *,
    timeout: float = 15.0,
    interval: float = 0.05,
) -> int:
    """Read *collection*'s vector dimension, tolerating the transient-404 read race.

    Under ``-n auto`` xdist load a collection created moments earlier can
    transiently 404 (or the HTTP layer can raise ``ResponseHandlingException``)
    before it is durably readable, so a single naive ``get_collection`` flakes.
    This wraps the read in the shared :func:`poll_until` engine so the
    transient window becomes a bounded retry: a 404 ``UnexpectedResponse`` or a
    ``ResponseHandlingException`` maps to "keep polling", while any other
    ``UnexpectedResponse`` (e.g. a genuine 500) is re-raised immediately —
    fail loud, no retry. A collection that never becomes readable surfaces as
    the usual :func:`poll_until` ``AssertionError`` at the deadline.

    Imports the qdrant symbols lazily inside the body, mirroring
    :func:`_qdrant_available` / :func:`ensure_fresh_collection` (see their
    docstrings) so importing this module never pulls in qdrant_client.

    Args:
        client: A ``qdrant_client.QdrantClient`` (or test double) exposing
            ``get_collection``.
        collection: The collection to read.
        timeout: Seconds to keep retrying the transient-404 window before
            raising. Defaults to 15s — generous enough to absorb host CPU
            oversubscription while still catching a genuinely-missing collection.
        interval: Seconds to sleep between probes. Defaults to 0.05s.

    Returns:
        The collection's single-vector dimensionality.

    Raises:
        AssertionError: the collection never became readable within *timeout*.
        qdrant_client.http.exceptions.UnexpectedResponse: a non-404 server
            error on the read (surfaced immediately, not retried).
    """
    from qdrant_client.http.exceptions import (
        ResponseHandlingException,
        UnexpectedResponse,
    )
    from qdrant_client.models import VectorParams

    def _probe():
        try:
            return client.get_collection(collection)
        except UnexpectedResponse as exc:
            if getattr(exc, 'status_code', None) == 404:
                return None  # transient: not yet durably visible — keep polling
            raise  # genuine server error — fail loud, no retry
        except ResponseHandlingException:
            return None  # transient HTTP-layer hiccup — keep polling

    info = await poll_until(
        _probe,
        timeout=timeout,
        interval=interval,
        message=f'collection {collection!r} not readable within {timeout}s',
    )
    vectors = info.config.params.vectors
    assert isinstance(vectors, VectorParams), (
        f'collection {collection!r} vectors config is not a single VectorParams: {vectors!r}'
    )
    return vectors.size


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


def _init_git_repo(path) -> str:
    """Create a minimal git repo at path with one commit; return full SHA.

    Shared by test_task_interceptor.py and test_status_authority_gate.py: a
    ``done_provenance={'kind': 'merged', 'commit': ...}`` write needs a real
    ``main`` branch for the interceptor's ``_verify_commit_on_main`` ancestor
    backstop (``git merge-base --is-ancestor <sha> main``) to resolve against.
    """
    import subprocess

    subprocess.run(['git', 'init', '-q', '-b', 'main', str(path)], check=True)
    subprocess.run(
        ['git', '-C', str(path), 'config', 'user.email', 't@e.example'],
        check=True,
    )
    subprocess.run(
        ['git', '-C', str(path), 'config', 'user.name', 'T'],
        check=True,
    )
    (path / 'seed.txt').write_text('seed\n')
    subprocess.run(['git', '-C', str(path), 'add', '-A'], check=True)
    subprocess.run(
        ['git', '-C', str(path), 'commit', '-q', '-m', 'seed'],
        check=True,
    )
    return subprocess.run(
        ['git', '-C', str(path), 'rev-parse', 'HEAD'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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
# Shared AST migration-guard machinery (task 3502; consumers widened by 3574)
# ---------------------------------------------------------------------------
#
# Migration guards (tests/test_falkor_probe_routing_guard.py;
# tests/test_falkor_index_barrier_guard.py;
# tests/test_gather_idiom_helper_routing.py) assert over the source of a set of
# migrated modules. The target is not always a TEST module: the gather-idiom
# guard asserts over production source under src/fused_memory/, which is why
# the parse is named for Python modules generally rather than for test ones.
# Parsing is shared and memoised here so several guards asserting over
# overlapping module sets pay one parse per file per session rather than one
# each.
# ---------------------------------------------------------------------------


@functools.cache
def parse_python_module(path: pathlib.Path) -> ast.Module:
    """Parse *path* into an ``ast.Module``, memoised per session.

    Accepts any Python module — a test module or production source under
    ``src/fused_memory/``. Sources do not change mid-session, so several guards
    asserting over overlapping module sets can share one parse per file.

    The returned tree is SHARED between callers, so no consumer may mutate it.
    """
    assert path.exists(), f'{path} not found'
    # The read is deliberately not memoised separately: this function is its
    # only caller and is itself cached on the same key, so a source cache would
    # score zero hits while pinning every parsed file's full text for the
    # session on top of its tree. The tree is the one retained artefact.
    return ast.parse(path.read_text(), filename=str(path))


def calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    """Every ``ast.Call`` under *node* whose callee is ``name(...)`` / ``….name(...)``.

    AST, not string grep: prose that merely mentions the name — a docstring
    describing the helper a guard enforces — must not satisfy or trip a check.

    Takes any node, not only a module, so a guard can ask the narrower question
    "is it called *here*" — inside a decorator, or inside the value assigned to
    ``pytestmark`` — rather than only "is it called anywhere in the file".
    """
    found: list[ast.Call] = []
    for descendant in ast.walk(node):
        if not isinstance(descendant, ast.Call):
            continue
        func = descendant.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            found.append(descendant)
    return found


def imported_names_from(node: ast.AST, module: str) -> set[str]:
    """Every name a ``from <module> import …`` under *node* brings in.

    Reports the SOURCE-side name, so ``from m import x as y`` is reported as
    ``x``. Guards compare the result against the name as the helper module
    exports it, so an aliased import must not read as a missing one.

    Only ``from <module> import …`` counts: a plain ``import <module>`` binds
    the module and never a name from it, so it cannot satisfy a guard demanding
    a specific helper be imported. A star import reports the literal ``'*'``
    (that is its ``alias.name``), which likewise satisfies no guard asking for
    a NAMED helper. Returns an empty set — never None — when
    nothing is imported from *module*, which callers rely on both for set
    algebra and to render ``imported or 'nothing'`` in a failure message.

    Takes any node rather than only a module, for symmetry with
    :func:`calls_named`.
    """
    names: set[str] = set()
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.ImportFrom) and descendant.module == module:
            names.update(alias.name for alias in descendant.names)
    return names
