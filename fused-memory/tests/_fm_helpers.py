"""Non-fixture test helpers for fused-memory tests.

Lives outside conftest.py to avoid the `sys.modules['conftest']` collision
that arises when root-level pytest loads multiple subprojects' conftests in
the same process.  Each subproject exports its helpers under a unique
module name so test files can `from _fm_helpers import X` without
colliding with sibling subprojects' helpers.
"""

import functools
import inspect
import json
import re
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

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
    """

    entity_edges: list[MockEdge] = field(default_factory=list)
    edges: list[MockEdge] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.edges == [] and self.entity_edges:
            self.edges = list(self.entity_edges)


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
