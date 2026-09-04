"""Lazy, atomic repair of envelope-markup corruption in a live plan.json.

Task 3692 (PRD ``plans/toolcall-markup-containment-prd.md``, task EPSILON,
boundary row B12). The subject under test is ``orchestrator.mcp.plan_tools``'s
read-time repair surface; the detection/repair mechanism itself is owned by
``shared.toolcall_markup`` (task 3688) and is NOT re-implemented here.

## Sentinel-literal hazard — every fixture is BUILT, never written verbatim

This module describes MCP tool-call envelope markup, so it is exactly the file
that must not contain any of that markup literally. The rationale is the one
recorded at ``shared/src/shared/toolcall_markup.py`` lines 52-62: an agent
editing a file that contains a raw envelope literal has to emit that literal
INSIDE its own tool-call argument, which reproduces the very over-consumption
defect under test — the Write/Edit argument terminates early, truncating this
file and silently dropping that call's sibling arguments.

So every specimen below is assembled at import time from :func:`_close`,
:func:`_open_param` and :data:`_INVOKE_CLOSER`, which build their angle bracket
from ``chr(60)``. The result is byte-identical at runtime and never appears
verbatim in the file text. :func:`_assert_no_raw_sentinels` enforces that on the
module's OWN BYTES at import, so a future editor cannot quietly reintroduce one
(it is a check on this file's source text, not on any docstring's wording).
"""

from __future__ import annotations

import ast
import asyncio
import copy
import errno
import functools
import inspect
import json
import logging
import stat
import tempfile
import textwrap
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
from shared.toolcall_markup import ENVELOPE_LITERALS, detect

from orchestrator.artifacts import (
    PLAN_SCHEMA_VERSION,
    ArtifactWriteError,
    TaskArtifacts,
)
from orchestrator.mcp import plan_tools

# ---------------------------------------------------------------------------
# Sentinel BUILDERS — the only way markup enters this module.
# ---------------------------------------------------------------------------

#: The opening angle bracket, spelled so it never appears verbatim in the file.
_LT = chr(60)


def _close(name: str) -> str:
    """Build the closing tag for *name* (the mis-close shape the harness emits)."""
    return _LT + '/' + name + '>'


def _open_param(name: str) -> str:
    """Build the canonical opening tag for parameter *name*."""
    return _LT + 'parameter name="' + name + '">'


#: The bare invoke closer — the terminator that trails a last-parameter leak.
_INVOKE_CLOSER = _close('invoke')


def _assert_no_raw_sentinels() -> None:
    """Fail at IMPORT if this file's own bytes carry a raw envelope literal.

    Checked against ``shared.toolcall_markup.ENVELOPE_LITERALS`` (the single
    owner of the literal set, INV-5) plus the two structural prefixes every
    built specimen uses, so a builder output spelled out by hand is caught even
    when it is not itself one of the enumerated literals.
    """
    source = Path(__file__).read_text(encoding='utf-8')
    forbidden = (*ENVELOPE_LITERALS, _LT + '/', _LT + 'parameter ')
    for sequence in forbidden:
        if sequence in source:
            raise AssertionError(
                f'{Path(__file__).name} contains a RAW envelope sentinel '
                f'({sequence!r}). Build it from _close()/_open_param() instead '
                '— a verbatim literal here corrupts the tool call that writes '
                'this file. See the module docstring.'
            )


_assert_no_raw_sentinels()


# ---------------------------------------------------------------------------
# The four REAL specimen shapes, measured on the 28 corrupted live plans.
# ---------------------------------------------------------------------------

#: Clean prose used as the intended value of whichever field a specimen poisons.
#: Deliberately free of any markup so ``detect()`` on a repaired prefix is None.
_RATIONALE_PROSE = (
    'Reusing the shared detector keeps one enumeration of the literals, so the '
    'write-time guard and the read-time repair can never drift apart.'
)
_HOW_PROSE = (
    'Imported directly; the helper does the grammar work and owns every '
    'accept/refuse decision, so this module adds no literal enumeration.'
)
_DECISION_PROSE = (
    'Repair the plan lazily on read rather than sweeping the fleet, because a '
    'sweep would have to quiesce every running task first.'
)

#: TRAILING RESIDUE on ``design_decisions[].rationale`` — the DOMINANT live
#: shape (97 of 118 corrupted strings): the parameter was last in the call, so
#: nothing was absorbed and only the mis-close plus the invoke closer trail it.
TRAILING_RATIONALE = _RATIONALE_PROSE + _close('rationale') + '\n' + _INVOKE_CLOSER + '\n'

#: The same trailing-residue shape on ``reuse[].how`` (27 of the 97).
TRAILING_HOW = _HOW_PROSE + _close('how') + '\n' + _INVOKE_CLOSER + '\n'

#: ABSORBED SIBLING on ``design_decisions[].decision``: the parser mis-closed
#: ``decision`` and then swallowed the whole ``rationale`` parameter into it, so
#: the rationale a later reader sees is another field's text (PRD section 2.4).
#: The final opener is UNTERMINATED — its closer was consumed as the terminator.
ABSORBED_RATIONALE = (
    _DECISION_PROSE + _close('decision') + '\n' + _open_param('rationale') + _RATIONALE_PROSE
)

#: PROSE FALSE POSITIVE, measured live in worktree 2939 — a plan ABOUT this leak,
#: whose authored text legitimately QUOTES the sentinels in ordinary sentences.
#: ``detect()`` fires, ``repair()`` correctly declines, and a trailing-only
#: sanitize contract would mutilate it. This is the specimen that makes the
#: repair-or-leave-byte-identical contract load-bearing rather than stylistic.
PROSE_QUOTED = (
    'The harness closes the argument with ' + _close('description') + ' or with '
    + _close('parameter') + ', and then re-opens with ' + _open_param('x')
    + ' before the next value, which is how the sibling arguments get lost.'
)


# ---------------------------------------------------------------------------
# Fixtures and plan factory.
# ---------------------------------------------------------------------------


@pytest.fixture()
def plan_artifacts(tmp_path):
    """TaskArtifacts over a temp worktree — mirrors ``test_plan_tools_server``."""
    a = TaskArtifacts(tmp_path)
    a.init('test-1', 'Test task', 'A test')
    return a


def corrupt_plan(**overrides) -> dict:
    """Return a complete, VALID plan dict whose fields can then be poisoned.

    Every call builds a fresh, independent document (no shared mutable state),
    so a test may poison ``plan['design_decisions'][0]['rationale']`` in place.
    Keyword *overrides* replace whole top-level keys, which is how a test swaps
    in its own collection (e.g. four decisions instead of the default two).

    The default document is entirely CLEAN: nothing here trips ``detect()``, so
    any fact a test observes came from the field it poisoned.
    """
    plan: dict = {
        'task_id': 'test-1',
        'title': 'A test plan',
        'analysis': 'Clean analysis prose describing the approach.',
        'files': ['orchestrator/src/orchestrator/mcp/plan_tools.py'],
        'prerequisites': [
            {
                'id': 'pre-1',
                'description': 'Clean prerequisite prose.',
                'status': 'pending',
                'commit': None,
                'tests': [],
            },
        ],
        'steps': [
            {
                'id': 'step-1',
                'type': 'test',
                'description': 'Clean step prose for the first step.',
                'status': 'pending',
                'commit': None,
            },
            {
                'id': 'step-2',
                'type': 'impl',
                'description': 'Clean step prose for the second step.',
                'status': 'pending',
                'commit': None,
            },
        ],
        'design_decisions': [
            {'decision': _DECISION_PROSE, 'rationale': 'Clean rationale prose.'},
            {'decision': 'A second clean decision.', 'rationale': 'A second clean rationale.'},
        ],
        'reuse': [
            {
                'what': 'The shared detector',
                'where': 'shared/src/shared/toolcall_markup.py',
                'how': 'Clean reuse prose.',
            },
            {
                'what': 'The plan artifact reader',
                'where': 'orchestrator/src/orchestrator/artifacts.py',
                'how': 'A second clean reuse prose.',
            },
        ],
    }
    plan.update(overrides)
    return plan


# ---------------------------------------------------------------------------
# step-1 — the repairable surface is DECLARED and machine-checked (INV-1).
# ---------------------------------------------------------------------------

#: The (collection, field) pairs measured as the corrupted surface across the
#: 28 live plans. ``collection is None`` means a top-level plan key. Held here
#: as the test's own independent statement of the contract — the point of the
#: table is that it cannot drift from this, so the two are deliberately
#: separate spellings rather than one imported from the other.
_EXPECTED_PAIRS = {
    (None, 'title'),
    (None, 'analysis'),
    ('prerequisites', 'description'),
    ('steps', 'description'),
    ('design_decisions', 'decision'),
    ('design_decisions', 'rationale'),
    ('reuse', 'what'),
    ('reuse', 'where'),
    ('reuse', 'how'),
}

#: Which plan-tools entry point AUTHORED each collection, i.e. whose parameter
#: names ``repair()`` must validate a recovery against.
_ORIGINATING_TOOL = {
    None: plan_tools._create_plan,
    'prerequisites': plan_tools._add_prerequisite,
    'steps': plan_tools._add_plan_step,
    'design_decisions': plan_tools._add_design_decision,
    'reuse': plan_tools._add_reuse_item,
}


def _tool_params(fn) -> tuple[str, ...]:
    """The tool's parameter names, minus the leading ``artifacts`` injection."""
    names = tuple(inspect.signature(fn).parameters)
    assert names[0] == 'artifacts', (
        f'{fn.__name__} no longer takes artifacts first — the table\'s '
        'schema_params derivation assumes it does'
    )
    return names[1:]


#: Tool PARAMETERS that name no prose plan key and may therefore NEVER receive a
#: recovered string. Three of them are stored under a DIFFERENT plan key
#: (``prereq_id``/``step_id`` -> ``id``, ``step_type`` -> ``type``), and two are
#: not prose at all (``files`` is a list, ``task_id`` an identifier). Writing a
#: recovered tail into any of them is silent-wrong-value corruption of the
#: artifact that the lock charter and the merge gate both consume.
_NON_PROSE_PARAMS = frozenset({'task_id', 'files', 'prereq_id', 'step_id', 'step_type'})

#: The tool name a fact reports as ``tool`` for each collection — its SCHEMA
#: OWNER, spelled independently of the module's own mapping so the two cannot
#: drift together.
_COLLECTION_SCHEMA_TOOL_NAME = {
    None: 'create_plan',
    'prerequisites': 'add_prerequisite',
    'steps': 'add_plan_step',
    'design_decisions': 'add_design_decision',
    'reuse': 'add_reuse_item',
}

#: The only three ``TaskArtifacts`` methods that mutate plan.json — measured,
#: not assumed: ``write_plan`` (artifacts.py:356), ``update_step_status``
#: (:734) and ``mark_step_committed`` (:753) each write
#: ``self.root / 'plan.json'`` (the latter via a call to ``self.write_plan``).
#: ``_plan_writing_tool_names()`` below keys its live-module walk on this set
#: to derive the candidate set for "could this plan-tools entry point write a
#: ``_REPAIRABLE_PLAN_FIELDS`` cell?".
#:
#: The five ``write_<report-kind>`` methods (``write_blocking_dependency`` /
#: ``write_already_done`` / ``write_ready_to_merge`` / ``write_unactionable_task``
#: / ``write_false_premise``) are deliberately EXCLUDED: each persists to its
#: own separate artifact file and never touches plan.json, so admitting one
#: here would wrongly pull the whole ``report_*`` family into the sweep.
_PLAN_MUTATING_ARTIFACT_METHODS = frozenset({
    'write_plan',
    'update_step_status',
    'mark_step_committed',
})


@functools.lru_cache(maxsize=1)
def _registered_plan_tool_names() -> frozenset[str]:
    """The tool names plan-tools' server actually REGISTERS, read from the server.

    Derived, never restated: builds a real server over a throwaway artifacts
    root and asks it what it registered. Measured against the live tree — 16
    names, the 11 plan writers plus the 5 ``report_*``.

    Cached because :func:`_undeclared_alternates` re-derives the candidate set
    once per row, which would otherwise pay one server build per row for an
    answer that cannot change within a process.

    ``asyncio.run`` is safe here: ``list_tools`` is a coroutine function
    (fastmcp 3.2.2) and this file has no ``async def`` test, so no loop is
    ever already running — ``asyncio_mode = "auto"`` governs async test
    functions only.
    """
    server = plan_tools.create_server(TaskArtifacts(Path(tempfile.mkdtemp())))
    return frozenset(tool.name for tool in asyncio.run(server.list_tools()))


def _plan_writing_tool_names(registered: frozenset[str] | None = None) -> tuple[str, ...]:
    """Derive the plan-writer candidate set from plan_tools' LIVE module surface.

    A plan_tools entry point is a plan writer iff (a) it is a module-level
    function DEFINED IN plan_tools — not an imported helper merely visible in
    its namespace; (b) its name starts with ``_``, the private-impl-behind-an-
    MCP-tool convention this file already keys on via
    ``getattr(plan_tools, '_' + tool_name)`` (see :func:`_alternate_writer_changed_the_cell`);
    (c) its first parameter is ``artifacts``; (d) its OWN BODY calls one of
    :data:`_PLAN_MUTATING_ARTIFACT_METHODS`; and (e) its ``_``-stripped name is
    one the server actually REGISTERED as a tool (*registered*, defaulting to
    :func:`_registered_plan_tool_names`). Returns the surviving names with the
    leading ``_`` stripped, sorted, as a tuple.

    Guard (e) exists because "writes plan.json" was never sufficient to mean
    "is a probeable tool impl". An INTERNAL helper may legitimately call
    ``artifacts.write_plan`` while returning something that is not a status
    envelope — ``_read_plan_repaired`` does exactly that on main, post-task-3957
    (commit c005fabb00), returning the bare ``(plan, facts)`` tuple. Without (e)
    the probe calls it and dies on ``result.get('status')`` with
    ``AttributeError: 'tuple' object has no attribute 'get'``. Deriving the
    registered surface from the server rather than hand-listing it keeps this
    guard from becoming the very kind of unchecked list this helper replaced.

    *registered* is a parameter, not a hardcoded call, so a test can stage a
    writer that a real registration would have made probeable
    (``test_an_undeclared_plan_writer_cannot_escape_the_sweep``).

    Reads each survivor's source via ``inspect.getsource`` off the LIVE module
    OBJECT, never by parsing plan_tools.py as a file. This is required, not
    stylistic: a synthetic writer monkeypatched onto ``plan_tools`` at test
    time (as the completeness test does) is invisible to a file-level parse,
    which would make the completeness property untestable — the one test that
    proves "a new plan writer is caught" could not be written.

    Detects a write with an ``ast.Call``/``ast.Attribute`` walk over each
    survivor's own body, never a substring scan of the source text. A text
    scan would be actively WRONG here: this module's own comments and
    docstrings mention these method names in prose repeatedly — including this
    very docstring. Measured on plan_tools, ``_repair_one_field`` discusses
    ``write_plan`` and ``create_server`` discusses ``mark_step_committed``,
    neither calling the method it names; a substring match would classify both
    as writers. Other guards happen to exclude those two as well, which is
    exactly the point — the AST walk is what makes the classification CORRECT
    rather than accidentally correct, and it is the guard that survives a
    future helper that is artifacts-first, ``_``-prefixed, and merely
    discusses a write.

    The ``report_*`` family (``report_blocking_dependency``,
    ``report_task_already_done``, ``report_ready_to_merge``,
    ``report_unactionable_task``, ``report_false_premise``) never appears in
    the returned tuple — not because it is hand-excluded, but as a
    MACHINE-CHECKED CONSEQUENCE of this derivation: each persists to its own
    separate artifact file (``artifacts.write_blocking_dependency`` /
    ``write_already_done`` / ``write_ready_to_merge`` /
    ``write_unactionable_task`` / ``write_false_premise``,
    plan_tools.py:1339-1431) and so calls none of
    :data:`_PLAN_MUTATING_ARTIFACT_METHODS`. Their absence costs no
    unrecognised-parameter synthesis (``classification``, ``premise``,
    ``evidence``, ...) for a guaranteed no-op — the fragility the module's
    design decisions rejected a full entry-point sweep over.
    """
    if registered is None:
        registered = _registered_plan_tool_names()
    names = []
    for name, obj in vars(plan_tools).items():
        if not inspect.isfunction(obj):
            continue
        if obj.__module__ != plan_tools.__name__:
            continue  # an imported helper, not an entry point defined here
        if not name.startswith('_'):
            continue
        params = tuple(inspect.signature(obj).parameters)
        if params[:1] != ('artifacts',):
            continue
        tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        if called & _PLAN_MUTATING_ARTIFACT_METHODS and name[1:] in registered:
            names.append(name[1:])
    return tuple(sorted(names))


def _seed_plan_through_real_writers(root) -> TaskArtifacts:
    """Build a plan by calling the five REAL writer tools; return its artifacts.

    The single source of the seeding shape. Both :func:`_observed_plan_keys`
    and :func:`_alternate_writer_changed_the_cell` build their fixture plan
    through this helper rather than each restating the same five calls, so
    adding a sixth writer — or changing one of these five signatures — is one
    edit instead of two silently-divergeable ones.
    """
    artifacts = TaskArtifacts(root)
    artifacts.init('test-1', 'Test task', 'A test')
    plan_tools._create_plan(artifacts, 'test-1', 'A title.', 'An analysis.', ['a.py'])
    plan_tools._add_prerequisite(artifacts, 'pre-1', 'A prerequisite.')
    plan_tools._add_plan_step(artifacts, 'step-1', 'test', 'A step.')
    plan_tools._add_design_decision(artifacts, 'A decision.', 'A rationale.')
    plan_tools._add_reuse_item(artifacts, 'A thing', 'somewhere.py', 'By importing it.')
    return artifacts


def _observed_plan_keys(root) -> dict[str | None, set[str]]:
    """Build a plan through the REAL writers; return the keys they actually wrote.

    Keyed by collection (``None`` for the top-level document). This is the
    machine-check that keeps ``target_keys`` from drifting into a vocabulary the
    plan never uses — deriving the allowed key sets from the tools themselves,
    exactly as ``schema_params`` is derived from ``inspect.signature``, rather
    than restating them as a hardcoded literal a refactor could silently orphan.
    """
    plan = _seed_plan_through_real_writers(root).read_plan()

    observed: dict[str | None, set[str]] = {None: set(plan)}
    for collection in ('prerequisites', 'steps', 'design_decisions', 'reuse'):
        items = plan[collection]
        assert items, f'{collection} came back empty — the writers did not run'
        observed[collection] = {key for item in items for key in item}
    return observed


def _alternate_writer_changed_the_cell(
    root, collection: str | None, field: str, tool_name: str
) -> tuple[bool, dict[str, object] | None]:
    """(changed, refusal) for calling *tool_name* on a plan seeded through the
    real writers, addressed at the (*collection*, *field*) cell.

    ``changed`` is True iff the cell's value differs after the call AND its
    item still exists. ``refusal`` is the tool's own status envelope when that
    envelope indicates it declined the call (``None`` on a successful call,
    whether or not the cell changed). Every plan-tools impl returns its
    refusal envelope BEFORE calling
    ``artifacts.write_plan``/``update_step_status``/``mark_step_committed`` on
    every refusal path, so ``refusal is not None`` always implies
    ``changed is False`` — never the reverse. Separating the two lets a caller
    distinguish "genuinely does not write this field" from "the probe called
    it wrong" (e.g. a stale id, an item already ``done``); collapsing both into
    a bare bool made every refusal read as proof of the former.

    THE BEHAVIOURAL PROOF ``also_written_by`` NEEDS. A signature check
    (``field in _tool_params(impl)``) can only see a writer that takes the
    field AS A PARAMETER — false for ``mark_step_committed``, which prepends a
    provenance tag to ``description`` by re-reading and rewriting the plan
    without ever taking a ``description`` argument at all. Proving the write
    behaviourally is the invariant the signature proxy was standing in for.

    Seeds via :func:`_seed_plan_through_real_writers` (same five real writers
    as :func:`_observed_plan_keys`), snapshots the addressed cell, invokes
    *tool_name* with arguments synthesized from ITS OWN live signature — never
    a hardcoded call. A parameter name this synthesis does not recognise
    raises loudly rather than being silently left unset, so a probe can never
    pass by having quietly called the alternate with a hole in its arguments.

    A collection whose items carry no ``id`` (``design_decisions``, ``reuse``)
    is addressed POSITIONALLY — the single seeded item at index 0 — rather
    than by id, since there is no id to look up; probing one of their rows
    used to raise a bare ``KeyError`` before this existed.

    An alternate whose guard depends on state this fixture cannot supply
    (currently only ``mark_step_committed``, via ``_sha_exists_on_branch`` —
    ``root`` here is not a git repository) is the CALLER's responsibility to
    satisfy first, e.g. via ``monkeypatch``.
    """
    artifacts = _seed_plan_through_real_writers(root)

    seeded_id = None
    address_by_index = False
    if collection is not None:
        items = artifacts.read_plan()[collection]
        assert items and isinstance(items[0], dict), (
            f'the seeded plan has no item in {collection!r} to probe'
        )
        if 'id' in items[0]:
            seeded_id = items[0]['id']
        else:
            address_by_index = True

    def _cell() -> tuple[object, bool]:
        """(value, item still present) at the (collection, field) address."""
        plan = artifacts.read_plan()
        if collection is None:
            return plan.get(field), True
        items = plan.get(collection, [])
        if address_by_index:
            if items and isinstance(items[0], dict):
                return items[0].get(field), True
            return None, False
        for item in items:
            if isinstance(item, dict) and item.get('id') == seeded_id:
                return item.get(field), True
        return None, False

    before, _ = _cell()

    impl = getattr(plan_tools, '_' + tool_name)
    kwargs: dict[str, object] = {}
    for name in inspect.signature(impl).parameters:
        if name == 'artifacts':
            continue
        if name in ('step_id', 'prereq_id'):
            kwargs[name] = seeded_id
        elif name == 'step_type':
            kwargs[name] = 'test'
        elif name in ('sha', 'commit_sha'):
            kwargs[name] = 'a' * 40
        elif name == 'files':
            kwargs[name] = ['a.py']
        elif name == 'task_id':
            kwargs[name] = 'test-1'
        elif name in (
            'description', 'analysis', 'title', 'decision', 'rationale',
            'what', 'where', 'how',
        ):
            kwargs[name] = f'Probe marker for {tool_name}.{name}.'
        else:
            raise AssertionError(
                f'_alternate_writer_changed_the_cell does not know how to '
                f'synthesize a value for {impl.__name__}({name!r}) — extend '
                'the probe rather than silently probing nothing, or exclude '
                'it from _PLAN_MUTATING_ARTIFACT_METHODS if it does not '
                'write plan.json'
            )
    result = impl(artifacts, **kwargs)
    refusal = None
    if not (result.get('status') == 'ok' or result.get('ok') is True):
        refusal = result

    after, still_exists = _cell()
    return still_exists and after != before, refusal


def _undeclared_alternates(
    tmp_path, monkeypatch, registered: frozenset[str] | None = None
) -> set[tuple[str, str | None, str]]:
    """Every (tool, collection, field) triple OBSERVED writing a cell while
    declared neither as that row's schema owner nor in its ``also_written_by``.

    Completeness half of ``also_written_by``: the soundness check
    (``test_every_alternate_writer_really_writes_that_field``) only proves
    every DECLARED alternate really writes its field, and passes VACUOUSLY
    when an alternate is silently dropped from the table. This sweeps every
    row of ``_REPAIRABLE_PLAN_FIELDS`` against every name in the DERIVED
    candidate set :func:`_plan_writing_tool_names` returns — never a
    hand-maintained list a new writer could be left out of — and asserts the
    CONVERSE of the soundness check.

    A refused probe call is not itself a finding — most (tool, row) pairs are
    simply not applicable (``mark_step_done`` can never touch ``reuse``), and
    per :func:`_alternate_writer_changed_the_cell`'s contract a refusal is
    conclusive proof the cell was not written; only an OBSERVED, UNDECLARED
    change is accumulated.

    A candidate whose signature the probe cannot synthesize a value for
    raises ``AssertionError`` (propagated unchanged from
    :func:`_alternate_writer_changed_the_cell`) rather than being silently
    skipped — a skip would reopen exactly the completeness hole this sweep
    exists to close.

    *registered* is forwarded to :func:`_plan_writing_tool_names` unchanged;
    see its docstring for why the candidate set is intersected with the
    server's own registered tool surface.
    """
    monkeypatch.setattr(plan_tools, '_sha_exists_on_branch', lambda *_a, **_k: True)
    undeclared: set[tuple[str, str | None, str]] = set()
    for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
        owner = _COLLECTION_SCHEMA_TOOL_NAME[record.collection]
        for tool_name in _plan_writing_tool_names(registered):
            if tool_name == owner:
                continue  # already reported as `tool`, not an "alternate"
            root = tmp_path / f'{record.collection}-{record.field}-{tool_name}'
            changed, _refusal = _alternate_writer_changed_the_cell(
                root, record.collection, record.field, tool_name
            )
            if not changed:
                continue
            if tool_name not in record.also_written_by:
                undeclared.add((tool_name, record.collection, record.field))
    return undeclared


class TestRepairableFieldTable:
    """``_REPAIRABLE_PLAN_FIELDS`` is the single declared repairable surface."""

    def test_table_is_an_immutable_sequence_of_records(self):
        table = plan_tools._REPAIRABLE_PLAN_FIELDS
        assert isinstance(table, tuple), (
            'the table must be immutable — a list could be appended to at '
            'runtime, which would make the declared surface un-auditable'
        )
        assert table, 'the table must not be empty'
        for record in table:
            assert isinstance(record, tuple)
            for attr in ('collection', 'field', 'schema_params', 'target_keys'):
                assert hasattr(record, attr), f'record {record!r} lacks {attr!r}'

    def test_covers_exactly_the_measured_corrupted_surface(self):
        table = plan_tools._REPAIRABLE_PLAN_FIELDS
        pairs = {(r.collection, r.field) for r in table}
        assert pairs == _EXPECTED_PAIRS
        # 'reuse' contributes three of the nine, so a set of pairs alone would
        # not catch a duplicated row: pin the record count too.
        assert len(table) == len(_EXPECTED_PAIRS) == 9

    # There is deliberately NO test pinning ``schema_params`` against
    # ``inspect.signature`` of the originating tool. The table now DERIVES that
    # tuple from the live signature (``plan_tools._params_of``), so such a test
    # would compare a value to itself. The drift it used to police is gone by
    # construction — which is the fix, not the loss of a check. What remains
    # below tests the half that is still hand-declared: ``field`` and
    # ``target_keys``, whose plan-key vocabulary cannot be derived from the tool.

    def test_the_repaired_field_is_itself_a_parameter_of_its_tool(self):
        """The hand-declared ``field`` must name a real parameter of its tool.

        Not vacuous despite the derivation: ``field`` is a literal in the table,
        so renaming a tool parameter without updating the row fails HERE rather
        than silently making the row unmatchable at repair time.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert record.field in record.schema_params, (
                f'{record.field!r} is not a parameter of '
                f'{_ORIGINATING_TOOL[record.collection].__name__}'
            )

    def test_schema_params_are_tuples_of_str(self):
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert isinstance(record.schema_params, tuple)
            assert all(isinstance(name, str) for name in record.schema_params)

    def test_files_is_neither_a_repaired_field_nor_a_recovery_target(self):
        """``files`` entries are paths, already recovered by ``_coerce_files``.

        BOTH directions matter, and only the first was ever asserted. Not being
        a repaired FIELD stops the walk from rewriting the list; not being a
        recovery TARGET stops an absorbed ``files`` tail from being written back
        as a bare ``str``, replacing the list the lock charter reads.
        """
        table = plan_tools._REPAIRABLE_PLAN_FIELDS
        assert 'files' not in {r.field for r in table}
        assert 'files' not in {name for r in table for name in r.target_keys}
        assert 'files' not in {key for r in table for key in r.target_keys.values()}

    def test_every_record_declares_an_immutable_target_keys_mapping(self):
        """A recovery target must be DECLARED, not inferred from the param name.

        ``schema_params`` is the tool's vocabulary; the plan document has its
        own. Conflating them is what let a recovered ``step_type`` land as a
        junk key on a step dict.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            targets = record.target_keys
            assert isinstance(targets, Mapping), (
                f'{record.collection}.{record.field}.target_keys is '
                f'{type(targets).__name__}, not a Mapping'
            )
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in targets.items())
            assert not isinstance(targets, dict), (
                'target_keys must not be a plain dict — the declared surface '
                'would then be mutable at runtime and un-auditable; wrap it in '
                'types.MappingProxyType'
            )
            with pytest.raises(TypeError):
                targets['injected'] = 'anything'  # type: ignore[index]

    def test_every_target_key_names_a_real_parameter_of_its_tool(self):
        """``target_keys`` is hand-declared, so its keys are checked, not derived.

        A key naming no real parameter of the originating tool could never be
        recovered — the row would be dead weight that reads as coverage.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            invented = set(record.target_keys) - set(record.schema_params)
            assert invented == set(), (
                f'{record.collection}.{record.field} declares recovery targets '
                f'for parameters {sorted(invented)} that '
                f'{_ORIGINATING_TOOL[record.collection].__name__} never takes'
            )

    def test_every_target_value_is_a_key_the_real_writers_produce(self, tmp_path):
        """INV-1 again: the mapping is checked against a plan the TOOLS wrote.

        A target value that no writer ever produces is by definition a junk key
        — the exact defect this mapping exists to prevent — so the allowed set
        is derived, never hardcoded.
        """
        observed = _observed_plan_keys(tmp_path)

        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            allowed = observed[record.collection]
            for param, key in record.target_keys.items():
                assert key in allowed, (
                    f'{record.collection}.{record.field} would recover {param!r} '
                    f'into {key!r}, which the writers never produce; real keys '
                    f'are {sorted(allowed)}'
                )

    def test_the_repaired_field_is_a_target_that_maps_to_itself(self):
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert record.field in record.target_keys
            assert record.target_keys[record.field] == record.field

    def test_every_alternate_writer_really_writes_that_field(self, tmp_path, monkeypatch):
        """``also_written_by`` must name real call sites that ACTUALLY write the field.

        The fact's ``tool`` is the collection's SCHEMA OWNER, which is not always
        the call that produced the corruption — ``replace_plan_step`` also writes
        ``description`` on steps AND prerequisites, ``update_plan_metadata`` also
        writes ``analysis``, and ``mark_step_committed`` prepends a provenance tag
        to ``description`` on either collection. An alternate that named no real
        entry point, or one that never actually touches the field, would send a
        triager somewhere the corruption could not have come from: a diagnostic
        that reads as precision while pointing at nothing.

        Checked BEHAVIOURALLY, not by signature. The previous version of this
        check asserted ``record.field in _tool_params(impl)`` — a proxy that can
        only see a writer that takes the field AS A PARAMETER. That is false of
        ``mark_step_committed``: it rewrites ``description`` by re-reading and
        rewriting the plan without ever taking a ``description`` argument at
        all, so the proxy would have REJECTED a real writer. Calling
        :func:`_alternate_writer_changed_the_cell` instead proves the actual
        invariant the proxy was standing in for, and is strictly stronger — it
        catches a declared alternate that names no real entry point (as before)
        AND one that names a real entry point that simply does not write the
        field (which the signature proxy could not).
        """
        monkeypatch.setattr(plan_tools, '_sha_exists_on_branch', lambda *_a, **_k: True)
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            assert isinstance(record.also_written_by, tuple)
            owner = _COLLECTION_SCHEMA_TOOL_NAME[record.collection]
            for name in record.also_written_by:
                impl = getattr(plan_tools, '_' + name, None)
                assert callable(impl), f'{name!r} is not a plan_tools function'
                assert name != owner, (
                    f'{name!r} is the schema owner, already reported as `tool`'
                )
                root = tmp_path / f'{record.collection}-{record.field}-{name}'
                changed, refusal = _alternate_writer_changed_the_cell(
                    root, record.collection, record.field, name
                )
                assert changed, (
                    f'{name!r} refused the probe call: {refusal!r} — the probe '
                    'called it wrong, it is not necessarily a non-writer'
                    if refusal is not None else
                    f'{record.collection}.{record.field} names {name!r} as an '
                    'alternate writer, but the probe observed no change on '
                    'that cell — it cannot actually write that field'
                )

    def test_no_plan_writing_tool_is_an_undeclared_alternate(self, tmp_path, monkeypatch):
        """Completeness half of ``also_written_by``: nothing UNDECLARED writes a cell.

        The soundness check above only proves every DECLARED alternate really
        writes its field, and passes VACUOUSLY when an alternate is silently
        dropped from the table. Mutation-verified in this worktree: dropping
        ``replace_plan_step`` from ``_DESCRIPTION_ALSO``, or
        ``update_plan_metadata`` from ``_UPDATE_METADATA_ALSO``, left every
        test in this file green — this test is what catches both, and the
        next undeclared writer along with them, without re-adding hardcoded
        pins one at a time.

        Sweeps every tool in the DERIVED candidate set
        (:func:`_plan_writing_tool_names`, not a hand-maintained list a new
        writer could be left out of) against every row and asserts the
        CONVERSE of the soundness check: whatever the probe OBSERVES writing
        a cell is either that row's schema owner (already reported as
        ``tool``, so skipped here) or already named in ``also_written_by``.
        A refused probe call is not itself a failure — most (tool, row)
        pairs are simply not applicable (``mark_step_done`` can never touch
        ``reuse``), and per :func:`_alternate_writer_changed_the_cell`'s
        contract a refusal is conclusive proof the cell was not written;
        only an OBSERVED, UNDECLARED change fails this test.
        """
        assert _undeclared_alternates(tmp_path, monkeypatch) == set()

    def test_no_non_prose_parameter_is_ever_a_recovery_target(self):
        """Identifiers, enums and lists may never receive a recovered string.

        ``prereq_id``/``step_id`` are stored under ``id`` and ``step_type``
        under ``type``, so a recovery keyed on the PARAMETER name could only
        ever create a junk key while leaving the real one corrupt.
        """
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            illegal = _NON_PROSE_PARAMS & set(record.target_keys)
            assert illegal == set(), (
                f'{record.collection}.{record.field} declares non-prose '
                f'recovery targets {sorted(illegal)}'
            )

    def test_a_tag_prepending_writer_is_declared_like_any_other(
        self, tmp_path, monkeypatch
    ):
        """S3 (task 3982): ``mark_step_committed`` belongs in ``also_written_by``.

        It prepends a ``[COMMITTED <sha12>]`` provenance tag to ``description``
        on either collection WITHOUT ever taking a ``description`` parameter —
        the exact shape a signature-based check cannot see. PROVE the premise
        first (the probe below), THEN assert the declaration — half (c) is the
        invariant; the table is only a restatement of it.
        """
        monkeypatch.setattr(plan_tools, '_sha_exists_on_branch', lambda *_a, **_k: True)

        for collection in ('steps', 'prerequisites'):
            changed, refusal = _alternate_writer_changed_the_cell(
                tmp_path / collection, collection, 'description', 'mark_step_committed'
            )
            assert changed, (
                f'mark_step_committed refused the probe call: {refusal!r} — the '
                'probe called it wrong, it is not necessarily a non-writer'
                if refusal is not None else
                f'mark_step_committed is supposed to rewrite {collection}.description '
                '(it prepends a [COMMITTED <sha>] tag) but the probe observed no change'
            )

        alternates = {
            (r.collection, r.field): set(r.also_written_by)
            for r in plan_tools._REPAIRABLE_PLAN_FIELDS
        }
        assert 'mark_step_committed' in alternates[('steps', 'description')]
        assert 'mark_step_committed' in alternates[('prerequisites', 'description')]

    def test_plan_mutating_method_names_are_real_task_artifacts_methods(self):
        """Guards the derivation's silent-vacuity failure mode.

        ``_plan_writing_tool_names()`` keys its source-level walk on
        ``_PLAN_MUTATING_ARTIFACT_METHODS`` — the TaskArtifacts method names
        that mutate plan.json. If one of those names ever stopped being a real
        TaskArtifacts method (e.g. a rename), a derivation keyed on the stale
        name would silently shrink the candidate set instead of failing
        loudly, and the completeness sweep would then pass over fewer tools
        while still reporting green — the exact silent-vacuity failure mode
        this guards against.
        """
        assert isinstance(_PLAN_MUTATING_ARTIFACT_METHODS, frozenset)
        assert _PLAN_MUTATING_ARTIFACT_METHODS, (
            'the set of plan-mutating TaskArtifacts methods must not be empty'
        )
        for name in _PLAN_MUTATING_ARTIFACT_METHODS:
            attr = getattr(TaskArtifacts, name, None)
            assert callable(attr), (
                f'{name!r} is not a real callable attribute of TaskArtifacts — '
                'the derivation would silently lose this writer'
            )
        separate_artifact_writers = {
            'write_blocking_dependency',
            'write_already_done',
            'write_ready_to_merge',
            'write_unactionable_task',
            'write_false_premise',
        }
        assert _PLAN_MUTATING_ARTIFACT_METHODS.isdisjoint(separate_artifact_writers), (
            'admitting a report_* separate-artifact writer here would wrongly '
            'pull the whole report_* family into the plan-writer sweep — each '
            'persists to its own artifact file and never touches plan.json'
        )

    def test_the_derived_candidate_set_cannot_silently_collapse(self):
        """Non-vacuity floor for the derivation: production's own claims.

        The floor is computed from the PRODUCTION table itself — every
        collection's schema-owner tool name, plus every name any row already
        declares in ``also_written_by`` — so this check adds no new
        hardcoded list of its own; it merely proves the derivation cannot
        lose a tool the module already claims writes a repairable cell. If
        the derivation silently collapsed (e.g. a broken AST walk), the
        completeness sweep would then pass vacuously over a shrunken
        candidate set while still reporting green — the dangerous failure
        mode a derived set is exposed to that a hardcoded list is not.
        """
        derived = _plan_writing_tool_names()
        assert derived, 'the derived candidate set must not be empty'

        floor = set(plan_tools._COLLECTION_SCHEMA_TOOL.values())
        for record in plan_tools._REPAIRABLE_PLAN_FIELDS:
            floor.update(record.also_written_by)

        assert floor <= set(derived), (
            f'the derived candidate set {sorted(derived)!r} is missing '
            f'{sorted(floor - set(derived))!r} — tool(s) the production table '
            'itself already claims write a repairable cell'
        )

    def test_a_plan_writing_non_tool_is_never_a_sweep_candidate(self, monkeypatch):
        """The derivation admits only REAL registered MCP tool impls.

        "Writes plan.json" was never sufficient to mean "is a probeable tool
        impl": an INTERNAL helper may legitimately call
        ``artifacts.write_plan`` while returning something that is not a
        status envelope, and the probe
        (:func:`_alternate_writer_changed_the_cell`) can only call a real
        tool impl.

        MEASURED PROVENANCE — this is a reproduced divergence, not a
        hypothetical. This branch's base is 83107bfe51; main has since
        advanced via c005fabb00 ("refactor: delete plan-tools' duplicate
        plan.json writer", task 3957 step-8), which DELETED the module-level
        ``_atomic_write_plan`` this branch still has and re-routed
        ``_read_plan_repaired``'s write-back through
        ``artifacts.write_plan(repaired)``. ``git merge-base --is-ancestor
        main HEAD`` reports NO, so this branch has diverged and WILL take
        that change. Reproduced with main's shape monkeypatched onto the live
        module: the derivation returns 12 names including
        ``read_plan_repaired``; the probe then calls
        ``_read_plan_repaired(artifacts)``, which returns the ``(plan,
        facts)`` tuple, and ``result.get('status')`` raises ``AttributeError:
        'tuple' object has no attribute 'get'`` — so BOTH
        ``test_no_plan_writing_tool_is_an_undeclared_alternate`` and
        ``test_an_undeclared_plan_writer_cannot_escape_the_sweep`` ERROR on
        rebase/merge.

        So the candidate set is intersected with the server's own REGISTERED
        tool surface, which is itself derived (:func:`_registered_plan_tool_names`)
        rather than hand-listed.
        """
        registered = _registered_plan_tool_names()
        assert registered, 'the registered tool surface must not be empty'
        assert set(_plan_writing_tool_names()) <= registered, (
            f'the derived candidate set {sorted(_plan_writing_tool_names())!r} '
            f'admits {sorted(set(_plan_writing_tool_names()) - registered)!r}, '
            'which the server never registered as a tool — the probe can only '
            'call a real tool impl'
        )
        assert 'read_plan_repaired' not in registered, (
            "_read_plan_repaired is an internal read-time helper, not a "
            'registered tool — if it ever becomes one, the probe must learn '
            'to call it'
        )

        def _internal_plan_rewriter(artifacts):
            """Exactly main's post-c005fabb00 ``_read_plan_repaired`` shape."""
            plan = artifacts.read_plan()
            artifacts.write_plan(plan)
            return plan, []

        # The same honest-simulation forgery
        # `test_an_undeclared_plan_writer_cannot_escape_the_sweep` performs:
        # without it the derivation's __module__ guard excludes a function
        # defined in this test module, so the regression cannot be staged.
        _internal_plan_rewriter.__module__ = plan_tools.__name__
        monkeypatch.setattr(
            plan_tools, '_internal_plan_rewriter', _internal_plan_rewriter, raising=False
        )
        assert 'internal_plan_rewriter' not in _plan_writing_tool_names(), (
            'a plan-mutating INTERNAL helper was admitted to the sweep — the '
            'probe would call it and crash on its non-envelope return value'
        )

    def test_a_non_envelope_return_is_reported_not_crashed(self, tmp_path, monkeypatch):
        """A mis-derived candidate is reported ACTIONABLY, not crashed opaquely.

        The second, INDEPENDENT half of the registered-surface fix. This test
        calls :func:`_alternate_writer_changed_the_cell` DIRECTLY rather than
        through :func:`_undeclared_alternates`, so it stays valid regardless
        of what the derivation admits — the independence is the point. Even
        if a future derivation change re-admits a non-tool, the failure names
        the offending candidate instead of dying on ``.get``.

        The staged return shape is the exact one main's ``_read_plan_repaired``
        returns post-c005fabb00: the bare ``(plan, facts)`` tuple, never a
        status envelope. Today the probe raises ``AttributeError: 'tuple'
        object has no attribute 'get'``, which names neither the offending
        candidate nor a remedy — what made the reviewer's reproduction so
        hard to read.
        """

        def _tuple_returning_writer(artifacts, description):
            plan = artifacts.read_plan()
            plan['steps'][0]['description'] = description
            artifacts.write_plan(plan)
            return plan, []

        _tuple_returning_writer.__module__ = plan_tools.__name__
        monkeypatch.setattr(
            plan_tools, '_tuple_returning_writer', _tuple_returning_writer, raising=False
        )

        with pytest.raises(AssertionError) as exc_info:
            _alternate_writer_changed_the_cell(
                tmp_path, 'steps', 'description', 'tuple_returning_writer'
            )
        assert 'tuple_returning_writer' in str(exc_info.value)

    def test_an_undeclared_plan_writer_cannot_escape_the_sweep(self, tmp_path, monkeypatch):
        """Pins the contract of ``_undeclared_alternates`` — the property
        nothing checks today, since the completeness sweep's candidate set
        was, until now, a hand-maintained tuple that a new writer could be
        left out of.

        (a) COMPLETENESS: a plan writer that exists but is declared nowhere
        is actually caught.
        (b) LOUD-ON-UNKNOWN-PARAMETER: a parameter the probe cannot
        synthesize halts the sweep with an actionable finding rather than
        silently skipping the tool — a skip would just reopen the same
        completeness hole this task exists to close.
        """

        def _probe_writer(artifacts, description):
            plan = artifacts.read_plan()
            plan['steps'][0]['description'] = description
            artifacts.write_plan(plan)
            return {'status': 'ok'}

        # The honest simulation of "a function defined in plan_tools": the
        # derivation's __module__ guard would otherwise exclude a function
        # defined in this test module.
        _probe_writer.__module__ = plan_tools.__name__
        monkeypatch.setattr(plan_tools, '_probe_writer', _probe_writer, raising=False)

        # The second half of that same honest simulation: a new plan writer
        # only reaches the sweep once its tool is REGISTERED, so staging the
        # regression means staging the registration too. Exactly parallel to
        # the ``__module__`` forgery above — without it the derivation's
        # registered-surface guard filters the synthetic writers straight back
        # out and the property under test cannot be exercised at all.
        staged = _registered_plan_tool_names() | {'probe_writer', 'probe_flavoured'}

        undeclared = _undeclared_alternates(
            tmp_path / 'completeness', monkeypatch, registered=staged
        )
        assert ('probe_writer', 'steps', 'description') in undeclared

        def _probe_flavoured(artifacts, description, flavour):
            plan = artifacts.read_plan()
            plan['steps'][0]['description'] = description
            artifacts.write_plan(plan)
            return {'status': 'ok'}

        _probe_flavoured.__module__ = plan_tools.__name__
        monkeypatch.setattr(plan_tools, '_probe_flavoured', _probe_flavoured, raising=False)

        with pytest.raises(AssertionError) as exc_info:
            _undeclared_alternates(
                tmp_path / 'unknown-param', monkeypatch, registered=staged
            )
        message = str(exc_info.value)
        assert 'probe_flavoured' in message
        assert 'flavour' in message


# ---------------------------------------------------------------------------
# step-3 — the pure repair pass over the DOMINANT trailing-residue shape.
# ---------------------------------------------------------------------------

#: The trailing shape on a step/prerequisite ``description``. Built here rather
#: than in the specimen block because the same field name serves two tools.
_TRAILING_DESCRIPTION = (
    'Clean step prose for the first step.' + _close('description') + '\n' + _INVOKE_CLOSER + '\n'
)
#: The trailing shape on the top-level ``analysis`` key (collection None).
_TRAILING_ANALYSIS = (
    'Clean analysis prose describing the approach.'
    + _close('analysis') + '\n' + _INVOKE_CLOSER + '\n'
)


#: A path to one string inside a plan document: dict keys and list indices.
_Path = tuple[str | int, ...]


def _all_strings(
    value, path: _Path = ()
) -> Iterator[tuple[_Path, str]]:
    """Every str in a nested plan document, keyed by its full path.

    Both the ``path`` parameter and the return are annotated rather than
    inferred. ``value`` is deliberately untyped (a plan document is arbitrary
    nested JSON), which puts this function on pyright's call-site return
    inference path, and the recursive ``yield from`` defeats that inference --
    it falls back to the bare ``()`` default's type, ``tuple[()]``. Callers
    then see an empty-tuple key, making every ``path[0]`` / ``path[-1]`` an
    out-of-range index error. The explicit return type pins the real key type:
    a mix of dict keys (str) and list indices (int).
    """
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _all_strings(item, (*path, key))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _all_strings(item, (*path, i))


class TestRepairPlanFieldsTrailing:
    """The last-parameter / trailing-residue shape — PRD boundary row B4.

    97 of the 118 corrupted strings measured across the 28 live plans have this
    shape: the leaked parameter was LAST in the call, so nothing was absorbed
    and ``repair()`` returns ``recovered == {}``. That is a success, not a
    refusal.
    """

    def test_trailing_rationale_repairs_with_a_full_fact(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE

        repaired, facts = plan_tools._repair_plan_fields(plan)

        decision = repaired['design_decisions'][0]
        assert decision['rationale'] == _RATIONALE_PROSE
        assert detect(decision['rationale']) is None
        # The sibling the tool authored in the same call is untouched.
        assert decision['decision'] == _DECISION_PROSE
        assert facts == [
            {
                'tool': 'add_design_decision',
                # add_design_decision is the ONLY writer of this field, so the
                # fact claims no alternates. Where a field has more than one
                # writer the fact says so rather than overstating its precision.
                'also_written_by': [],
                'param': 'rationale',
                'pattern': _INVOKE_CLOSER,
                'misclose': _close('rationale'),
                'outcome': 'repaired',
                'recovered_params': [],
                'declined_params': [],
                'collection': 'design_decisions',
                'index': 0,
                'field': 'rationale',
            }
        ]

    def test_is_pure__input_plan_is_not_mutated(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE

        repaired, _facts = plan_tools._repair_plan_fields(plan)

        assert plan['design_decisions'][0]['rationale'] == TRAILING_RATIONALE, (
            'the caller\'s dict must not be mutated in place — the write-back '
            'decision belongs to the caller, not to the repair pass'
        )
        assert repaired is not plan
        assert repaired['design_decisions'] is not plan['design_decisions']

    def test_index_locator_is_the_items_real_position(self):
        plan = corrupt_plan()
        plan['reuse'][1]['how'] = TRAILING_HOW

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['reuse'][1]['how'] == _HOW_PROSE
        assert repaired['reuse'][0]['how'] == 'Clean reuse prose.'
        assert len(facts) == 1
        assert facts[0]['collection'] == 'reuse'
        assert facts[0]['index'] == 1
        assert facts[0]['field'] == 'how'
        assert facts[0]['tool'] == 'add_reuse_item'
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == []
        assert facts[0]['misclose'] == _close('how')

    @pytest.mark.parametrize(
        ('collection', 'index', 'field', 'poisoned', 'clean', 'tool'),
        [
            (
                'steps', 0, 'description', _TRAILING_DESCRIPTION,
                'Clean step prose for the first step.', 'add_plan_step',
            ),
            (
                'prerequisites', 0, 'description',
                'Clean prerequisite prose.' + _close('description') + '\n' + _INVOKE_CLOSER + '\n',
                'Clean prerequisite prose.', 'add_prerequisite',
            ),
        ],
    )
    def test_trailing_shape_on_each_collection(
        self, collection, index, field, poisoned, clean, tool
    ):
        plan = corrupt_plan()
        plan[collection][index][field] = poisoned

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired[collection][index][field] == clean
        assert detect(repaired[collection][index][field]) is None
        assert len(facts) == 1
        assert facts[0]['collection'] == collection
        assert facts[0]['index'] == index
        assert facts[0]['field'] == field
        assert facts[0]['tool'] == tool
        assert facts[0]['outcome'] == 'repaired'

    def test_top_level_field_reports_collection_and_index_none(self):
        plan = corrupt_plan(analysis=_TRAILING_ANALYSIS)

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['analysis'] == 'Clean analysis prose describing the approach.'
        assert len(facts) == 1
        assert facts[0]['collection'] is None
        assert facts[0]['index'] is None
        assert facts[0]['field'] == 'analysis'
        assert facts[0]['param'] == 'analysis'
        assert facts[0]['tool'] == 'create_plan'

    def test_four_corrupted_rationales_yield_four_indexed_facts(self):
        decisions = [
            {'decision': f'Decision number {i}.', 'rationale': TRAILING_RATIONALE}
            for i in range(4)
        ]
        plan = corrupt_plan(design_decisions=decisions)
        before = dict(_all_strings(plan))

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert len(facts) == 4
        assert [f['index'] for f in facts] == [0, 1, 2, 3]
        assert {f['outcome'] for f in facts} == {'repaired'}
        for item in repaired['design_decisions']:
            assert item['rationale'] == _RATIONALE_PROSE

        # Every OTHER string in the document is byte-identical.
        after = dict(_all_strings(repaired))
        assert set(after) == set(before)
        for path, value in after.items():
            if path[0] == 'design_decisions' and path[-1] == 'rationale':
                continue
            assert value == before[path], f'{path} changed but was not poisoned'

    def test_clean_plan_is_returned_unchanged_with_no_facts(self):
        plan = corrupt_plan()

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert repaired == plan

    def test_a_clean_plan_is_never_deep_copied(self, monkeypatch):
        """The copy is paid on the corrupted path only, not on every read.

        Every plan-tools call in the fleet goes through this pass, and the
        overwhelming majority of plans are clean. A plan carries tens of KB of
        analysis prose plus every step description, so an unconditional
        ``deepcopy`` before any detection has run would be the single most
        expensive thing this surface does on the path where it does nothing.
        """
        plan = corrupt_plan()
        copies: list = []
        real_deepcopy = copy.deepcopy
        monkeypatch.setattr(
            plan_tools.copy,
            'deepcopy',
            lambda obj, *a, **kw: (copies.append(obj), real_deepcopy(obj, *a, **kw))[1],
        )

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert copies == [], 'a clean plan was deep-copied for nothing'
        assert repaired is plan, (
            'nothing changed, so the same document is handed back — the '
            'no-mutation contract holds because this path writes nothing'
        )

    def test_a_corrupted_plan_is_still_copied_before_it_is_touched(self, monkeypatch):
        """The laziness must not leak into mutating the caller's document."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        copies: list = []
        real_deepcopy = copy.deepcopy
        monkeypatch.setattr(
            plan_tools.copy,
            'deepcopy',
            lambda obj, *a, **kw: (copies.append(obj), real_deepcopy(obj, *a, **kw))[1],
        )

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert len(facts) == 1
        assert copies == [plan]
        assert repaired is not plan
        assert plan['design_decisions'][0]['rationale'] == TRAILING_RATIONALE

    @pytest.mark.parametrize(
        ('collection', 'index', 'field', 'poisoned'),
        [
            ('design_decisions', 0, 'rationale', TRAILING_RATIONALE),
            ('reuse', 1, 'how', TRAILING_HOW),
            ('steps', 0, 'description', _TRAILING_DESCRIPTION),
        ],
    )
    def test_d5_repaired_value_is_always_a_prefix_of_the_original(
        self, collection, index, field, poisoned
    ):
        """INVARIANT D5: recovery only ever SLICES; it never synthesises text."""
        plan = corrupt_plan()
        plan[collection][index][field] = poisoned

        repaired, facts = plan_tools._repair_plan_fields(plan)

        new_value = repaired[collection][index][field]
        assert poisoned.startswith(new_value)
        assert new_value != poisoned
        assert facts[0]['outcome'] == 'repaired'

    def test_fact_tool_names_a_real_plan_tools_entry_point(self):
        """The ``tool`` field is a live function name, not a prose label."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan['reuse'][0]['how'] = TRAILING_HOW
        plan['steps'][0]['description'] = _TRAILING_DESCRIPTION
        plan['analysis'] = _TRAILING_ANALYSIS

        _repaired, facts = plan_tools._repair_plan_fields(plan)

        assert len(facts) == 4
        for fact in facts:
            impl = getattr(plan_tools, '_' + fact['tool'], None)
            assert callable(impl), f'{fact["tool"]!r} is not a plan_tools function'
            assert fact['param'] in _tool_params(impl)


# ---------------------------------------------------------------------------
# step-5 — refusal is byte-identical and LOUD; never a partial or guessed fix.
# ---------------------------------------------------------------------------

#: DOUBLY CORRUPTED ``reuse[].how``: the recovered tail itself carries a second
#: mis-close, so the recovered value's own boundary would be a guess. Alpha's
#: ``_parse_tail`` refuses this by construction (PRD boundary row B5).
_DOUBLY_CORRUPT_HOW = (
    'Reuse prose that was cut short.' + _close('how') + '\n'
    + _open_param('what') + 'the absorbed value' + _close('description')
    + ' and then still more leftover text.'
)


class TestRepairPlanFieldsRefuses:
    """``repair()`` declines -> the field is left EXACTLY as it was.

    This is task 3685's open reject-vs-sanitize question, answered. There is NO
    trailing-only sanitize fallback, because the declining population is
    dominated by plans that legitimately QUOTE the sentinels in prose.
    """

    def test_prose_false_positive_is_left_byte_identical(self):
        """Worktree 2939's shape: a plan ABOUT the leak, quoting the sentinels.

        A trailing-only sanitize contract would mutilate this authored text —
        the same class of silent-wrong-value damage the PRD exists to end.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['design_decisions'][0]['decision'] == PROSE_QUOTED
        # Nothing was shaved off the end: all three quoted sentinels survive.
        for quoted in (_close('description'), _close('parameter'), _open_param('x')):
            assert quoted in repaired['design_decisions'][0]['decision']
        assert len(facts) == 1
        fact = facts[0]
        assert fact['outcome'] == 'unrepairable'
        # The residue stays VISIBLE: the detected pattern is reported.
        assert fact['pattern'] == detect(PROSE_QUOTED) == _close('description')
        assert fact['recovered_params'] == []
        assert fact['misclose'] is None
        assert fact['collection'] == 'design_decisions'
        assert fact['index'] == 0
        assert fact['field'] == 'decision'
        assert fact['tool'] == 'add_design_decision'

    def test_doubly_corrupted_value_is_left_byte_identical(self):
        plan = corrupt_plan()
        plan['reuse'][0]['how'] = _DOUBLY_CORRUPT_HOW

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['reuse'][0]['how'] == _DOUBLY_CORRUPT_HOW
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        # Task 4696: the diagnostic now names the tag it ACTUALLY saw — this
        # value's own \x3c/how> closer, which precedes the canonical opener the
        # blanket predicate used to report. Naming the literal that merely
        # TRAILS a leak is PRD section 2.2's original complaint, so reporting
        # the earlier self-name closer is the improvement, not a changed
        # expectation. The two are asserted DISTINCT so this row keeps proving
        # the widening is what is being observed.
        assert facts[0]['pattern'] == _close('how')
        assert facts[0]['pattern'] != detect(_DOUBLY_CORRUPT_HOW)
        assert facts[0]['recovered_params'] == []
        assert facts[0]['field'] == 'how'

    def test_refusal_never_partially_repairs_the_sibling_fields(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        before = dict(_all_strings(plan))

        repaired, _facts = plan_tools._repair_plan_fields(plan)

        assert dict(_all_strings(repaired)) == before

    def _mixed_plan(self) -> dict:
        """One repairable field and one refusing field, in the same record."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan['reuse'][0]['how'] = _DOUBLY_CORRUPT_HOW
        plan['reuse'][1]['how'] = TRAILING_HOW
        return plan

    def test_is_deterministic_across_repeated_runs(self):
        first_plan, first_facts = plan_tools._repair_plan_fields(self._mixed_plan())
        second_plan, second_facts = plan_tools._repair_plan_fields(self._mixed_plan())

        assert first_plan == second_plan
        assert first_facts == second_facts
        assert {f['outcome'] for f in first_facts} == {'repaired', 'unrepairable'}

    def test_converges__second_pass_repairs_nothing_and_still_refuses(self):
        """Idempotent CONVERGENCE, not oscillation.

        Feeding the result back must leave the already-repaired fields alone
        (zero further 'repaired' facts) while still reporting the residue that
        was, and remains, unrepairable.
        """
        once, first_facts = plan_tools._repair_plan_fields(self._mixed_plan())
        twice, second_facts = plan_tools._repair_plan_fields(once)

        assert twice == once
        assert [f for f in second_facts if f['outcome'] == 'repaired'] == []
        assert (
            [f for f in second_facts if f['outcome'] == 'unrepairable']
            == [f for f in first_facts if f['outcome'] == 'unrepairable']
        )

    @pytest.mark.parametrize('value', [None, 42, 3.5, [], {}, True])
    def test_non_str_values_produce_no_fact_and_no_crash(self, value):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = value

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert repaired['design_decisions'][0]['rationale'] == value

    def test_missing_field_and_non_dict_item_produce_no_fact_and_no_crash(self):
        plan = corrupt_plan()
        del plan['design_decisions'][0]['rationale']
        del plan['analysis']
        plan['reuse'].append('not a dict at all')
        plan['steps'] = 'not a list at all'

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert facts == []
        assert 'rationale' not in repaired['design_decisions'][0]
        assert 'analysis' not in repaired
        assert repaired['reuse'][-1] == 'not a dict at all'
        assert repaired['steps'] == 'not a list at all'


# ---------------------------------------------------------------------------
# step-7 — the ABSORBED-SIBLING shape: fill a HOLE, never overwrite authored text.
# ---------------------------------------------------------------------------

#: A real, later-authored rationale, as the live worktree-3024 plans look after
#: the agent noticed the truncation and retried. At read time this and a
#: recovered tail are indistinguishable — which is precisely why one must win
#: by rule rather than by luck.
_AUTHORED_RATIONALE = (
    'The agent retried the call and this rationale is the one it actually meant '
    'to record, so nothing may clobber it.'
)


def _absorbed_decision(**record) -> dict:
    """A plan whose ``design_decisions[0]`` absorbed its rationale sibling."""
    return corrupt_plan(
        design_decisions=[{'decision': ABSORBED_RATIONALE, **record}],
    )


class TestRepairPlanFieldsAbsorbedSibling:
    """PRD section 2.4's headline damage, restated for the read-time surface.

    The rationale was ABSORBED INTO decision, so the design rationale a future
    architect reads is another field's text. Recovering it is the point of the
    task — but only ever into an EMPTY or ABSENT sibling (boundary row B9).
    """

    def test_recovers_into_an_empty_sibling(self):
        plan = _absorbed_decision(rationale='')

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert detect(item['decision']) is None
        assert item['rationale'] == _RATIONALE_PROSE
        # D5: the recovered text is a VERBATIM substring of the original value,
        # never synthesised.
        assert item['rationale'] in ABSORBED_RATIONALE
        assert ABSORBED_RATIONALE.startswith(item['decision'])
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']
        assert facts[0]['field'] == 'decision'
        assert facts[0]['misclose'] == _close('decision')

    def test_recovers_into_a_missing_sibling__key_is_created(self):
        plan = _absorbed_decision()
        assert 'rationale' not in plan['design_decisions'][0]

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert item['rationale'] == _RATIONALE_PROSE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']

    def test_whitespace_only_sibling_counts_as_a_hole(self):
        plan = _absorbed_decision(rationale='   \n\t  ')

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert item['rationale'] == _RATIONALE_PROSE
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']

    def test_a_tail_redeclaring_the_repaired_field_never_displaces_it(self):
        """The one hole ``repair()``'s B9 check structurally cannot cover.

        ``supplied`` must EXCLUDE the field being repaired — repair() is
        repairing that parameter, so calling it already-supplied would refuse
        every candidate — which means a tail that re-declares that same name
        passes the disjointness test and comes back in ``recovered``. Writing it
        would overwrite the ``clean_value`` just recovered, leaving the STUB
        where the authored prose was: a verbatim slice, so D5 still holds, and
        silent authored-text loss all the same. The authored value wins, and the
        redeclaration is reported rather than dropped in silence.
        """
        authored = 'The authored rationale, all of it.'
        plan = corrupt_plan(
            design_decisions=[{
                'decision': 'A decision.',
                'rationale': (
                    authored + _close('rationale') + '\n'
                    + _open_param('rationale') + 'stub'
                ),
            }],
        )

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['rationale'] == authored, (
            'the stub from the tail displaced the authored prose'
        )
        assert detect(item['rationale']) is None
        assert item['decision'] == 'A decision.'
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'repaired'
        # Nothing was WRITTEN from the tail; the redeclaration is declined, and
        # says so, so the case cannot read as a successful sibling recovery.
        assert facts[0]['recovered_params'] == []
        assert facts[0]['declined_params'] == ['rationale']

    def test_a_redeclaring_tail_wins_when_the_misclose_starts_the_value(self):
        """The MIRROR of the case above, and it must resolve the other way.

        Same shape — a tail re-declaring the field being repaired — but the
        mis-close sits at position 0, so the two sides swap roles:
        ``clean_value`` is the HOLE and the recovered self-value is the authored
        prose. Declining here would blank the field, and because the fact still
        reads ``outcome: 'repaired'``, ``_read_plan_repaired`` persists that
        blank through ``TaskArtifacts.write_plan`` — authored text destroyed on disk,
        unrecoverable on the next read. Which side wins is therefore decided by
        CONTENT, never by position.
        """
        authored = 'The REAL authored rationale, all of it.'
        plan = corrupt_plan(
            design_decisions=[{
                'decision': 'A decision.',
                'rationale': (
                    _close('rationale') + '\n'
                    + _open_param('rationale') + authored
                ),
            }],
        )

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['rationale'] == authored, (
            'the decline blanked the field the recovered tail was holding'
        )
        assert detect(item['rationale']) is None
        assert item['decision'] == 'A decision.'
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'repaired'
        # The mirror of the case above: here the tail is what gets WRITTEN, so
        # nothing is declined and the fact must not read as a refusal.
        assert facts[0]['recovered_params'] == ['rationale']
        assert facts[0]['declined_params'] == []

    def test_a_whitespace_only_clean_value_is_a_hole_the_tail_may_fill(self):
        """``_is_authored`` decides it, so whitespace is a hole here too.

        The near-miss of the case above: the mis-close sits just past a run of
        whitespace rather than at offset 0, so ``clean_value`` is non-empty but
        still holds nothing. Reading emptiness as ``clean_value == ''`` would
        let this one through as a blanking write.
        """
        authored = 'The REAL authored rationale, all of it.'
        plan = corrupt_plan(
            design_decisions=[{
                'decision': 'A decision.',
                'rationale': (
                    '   \n\t  ' + _close('rationale') + '\n'
                    + _open_param('rationale') + authored
                ),
            }],
        )

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['rationale'] == authored
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']
        assert facts[0]['declined_params'] == []

    def test_never_overwrites_a_non_blank_authored_sibling(self):
        """The invariant: fill a hole, NEVER overwrite authored text.

        A recovered tail and a later-retried real value are indistinguishable
        at read time, so clobbering the authored one would be exactly the
        silent-wrong-value failure this surface exists to prevent.
        """
        plan = _absorbed_decision(rationale=_AUTHORED_RATIONALE)

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['rationale'] == _AUTHORED_RATIONALE
        # The WHOLE decision field is left byte-identical — not truncated to
        # its clean prefix with the recovery merely suppressed.
        assert item['decision'] == ABSORBED_RATIONALE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['field'] == 'decision'
        assert facts[0]['recovered_params'] == []

    @pytest.mark.parametrize('sibling', ['', '   \n\t  ', None])
    def test_recovered_value_is_always_a_slice_of_the_original(self, sibling):
        plan = _absorbed_decision(rationale=sibling)

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert facts[0]['outcome'] == 'repaired'
        assert item['rationale'] == _RATIONALE_PROSE
        assert item['decision'] in ABSORBED_RATIONALE
        assert item['rationale'] in ABSORBED_RATIONALE
        assert item['decision'] + item['rationale'] != ABSORBED_RATIONALE, (
            'the two slices must not simply re-concatenate to the input — the '
            'mis-close and the re-opened tag are DROPPED, not re-homed'
        )


# ---------------------------------------------------------------------------
# step-19 — a recovery may only ever land on a REAL prose plan key.
# ---------------------------------------------------------------------------

#: An absorbed ``files`` argument on the top-level ``title``. This is the state
#: the harness leaves behind when it mis-closes ``title`` during ``create_plan``
#: and swallows the ``files`` argument: the plan's ``files`` is the empty list
#: (nothing was ever parsed into it), which is precisely what makes it look like
#: a HOLE a recovery may fill — and ``files`` is a LIST, so filling it with the
#: recovered ``str`` replaces the value the lock charter (``derive_modules`` /
#: ``files_to_modules``) and the merge gate (``plan_files_not_touched``) read.
_ABSORBED_FILES_TITLE = (
    'A real title.' + _close('title') + '\n' + _open_param('files') + 'orchestrator/src/a.py'
)

#: An absorbed ``step_type`` argument on a step ``description``. ``step_type``
#: is the TOOL's parameter name; the plan stores it as ``type``. Keyed on the
#: parameter name, the recovery creates a junk ``step_type`` key and leaves the
#: real ``type`` — the field actually corrupted — untouched.
_ABSORBED_STEP_TYPE = 'Do the thing.' + _close('description') + '\n' + _open_param('step_type') + 'test'

#: The same shape with ``prereq_id``, which the plan stores as ``id``.
_ABSORBED_PREREQ_ID = (
    'A prerequisite.' + _close('description') + '\n' + _open_param('prereq_id') + 'pre-99'
)


class TestRecoveryTargetsAreRealPlanKeys:
    """A recovered value lands on a DECLARED prose key, or nowhere at all.

    Both correctness findings share one root cause: the walk keyed recoveries
    on the TOOL's parameter names, which are not the PLAN's key names. The fix
    is not a new collision check — ``repair()`` still makes every accept/refuse
    decision (INV-5). It is that a parameter with no prose target is supplied
    UNCONDITIONALLY, so ``repair()``'s existing B9 disjointness condition
    refuses any candidate that would recover it.
    """

    def test_an_absorbed_files_argument_never_replaces_the_files_list(self):
        """Measured before the fix: ``plan['files']`` came back a ``str``."""
        plan = corrupt_plan(title=_ABSORBED_FILES_TITLE, files=[])

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert isinstance(repaired['files'], list), (
            'a recovered path string replaced the files LIST — the lock charter '
            'and the merge gate both read this value'
        )
        assert repaired['files'] == []
        assert repaired['title'] == _ABSORBED_FILES_TITLE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['field'] == 'title'
        assert facts[0]['recovered_params'] == []

    def test_a_non_empty_files_list_is_equally_untouched(self):
        original = ['orchestrator/src/orchestrator/mcp/plan_tools.py']
        plan = corrupt_plan(title=_ABSORBED_FILES_TITLE, files=list(original))

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert repaired['files'] == original
        assert repaired['title'] == _ABSORBED_FILES_TITLE
        assert facts[0]['outcome'] == 'unrepairable'

    def test_an_absorbed_step_type_creates_no_junk_key(self):
        """Measured before the fix: the step gained ``step_type: 'test'``.

        ...while ``type``, the field actually corrupted, stayed wrong — so the
        repair reported success having fixed nothing and added a key no reader
        of the plan schema expects.
        """
        step = {
            'id': 'step-1',
            'type': 'impl',
            'description': _ABSORBED_STEP_TYPE,
            'status': 'pending',
            'commit': None,
        }
        plan = corrupt_plan(steps=[step])

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['steps'][0]
        assert 'step_type' not in item, f'junk key written: {sorted(item)}'
        assert item['type'] == 'impl'
        assert item['description'] == _ABSORBED_STEP_TYPE
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'
        assert facts[0]['recovered_params'] == []

    def test_an_absorbed_prereq_id_creates_no_junk_key(self):
        prereq = {
            'id': 'pre-1',
            'description': _ABSORBED_PREREQ_ID,
            'status': 'pending',
            'commit': None,
            'tests': [],
        }
        plan = corrupt_plan(prerequisites=[prereq])

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['prerequisites'][0]
        assert 'prereq_id' not in item, f'junk key written: {sorted(item)}'
        assert item['id'] == 'pre-1'
        assert item['description'] == _ABSORBED_PREREQ_ID
        assert len(facts) == 1
        assert facts[0]['outcome'] == 'unrepairable'

    def test_no_junk_key_survives_a_mixed_plan(self, tmp_path):
        """The invariant, over every specimen at once and against REAL keys.

        The allowed key sets are derived by building a plan through the actual
        writers, exactly as the table test does — a hardcoded list here would
        just be a second thing to keep in sync.
        """
        observed = _observed_plan_keys(tmp_path)
        decisions = [
            {'decision': _DECISION_PROSE, 'rationale': TRAILING_RATIONALE},
            {'decision': 'A clean decision.', 'rationale': 'A clean rationale.'},
        ]
        plan = corrupt_plan(
            title=_ABSORBED_FILES_TITLE,
            files=[],
            steps=[
                {
                    'id': 'step-1',
                    'type': 'impl',
                    'description': _ABSORBED_STEP_TYPE,
                    'status': 'pending',
                    'commit': None,
                },
            ],
            prerequisites=[
                {
                    'id': 'pre-1',
                    'description': _ABSORBED_PREREQ_ID,
                    'status': 'pending',
                    'commit': None,
                    'tests': [],
                },
            ],
            design_decisions=decisions,
        )

        repaired, _facts = plan_tools._repair_plan_fields(plan)

        assert set(repaired) <= observed[None], (
            f'top-level junk keys: {sorted(set(repaired) - observed[None])}'
        )
        for collection in ('prerequisites', 'steps', 'design_decisions', 'reuse'):
            for index, item in enumerate(repaired[collection]):
                junk = set(item) - observed[collection]
                assert junk == set(), f'{collection}[{index}] junk keys: {sorted(junk)}'

    def test_the_dominant_trailing_shapes_still_repair(self):
        """The load-bearing no-over-refusal guard.

        Supplying the non-target parameters unconditionally must not make the
        repair conservative in general: 97 of the 118 live corrupted strings
        have the trailing shape, and every one of them must still repair.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan['reuse'][1]['how'] = TRAILING_HOW
        plan['steps'][0]['description'] = _TRAILING_DESCRIPTION
        plan['analysis'] = _TRAILING_ANALYSIS

        repaired, facts = plan_tools._repair_plan_fields(plan)

        assert [f['outcome'] for f in facts] == ['repaired'] * 4
        assert all(f['recovered_params'] == [] for f in facts)
        assert repaired['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        assert repaired['reuse'][1]['how'] == _HOW_PROSE
        assert repaired['steps'][0]['description'] == 'Clean step prose for the first step.'
        assert repaired['analysis'] == 'Clean analysis prose describing the approach.'

    def test_the_absorbed_sibling_recovery_still_fills_an_empty_sibling(self):
        """``rationale`` IS a declared target, so step-7's recovery is intact."""
        plan = _absorbed_decision(rationale='')

        repaired, facts = plan_tools._repair_plan_fields(plan)

        item = repaired['design_decisions'][0]
        assert item['decision'] == _DECISION_PROSE
        assert item['rationale'] == _RATIONALE_PROSE
        assert facts[0]['outcome'] == 'repaired'
        assert facts[0]['recovered_params'] == ['rationale']


# ---------------------------------------------------------------------------
# step-9 / task 3957 step-8 — the plan.json write contract.
#
# `_atomic_write_plan` and its helpers (`_verify_plan_json`,
# `_target_file_mode`, `PlanWriteError`) are GONE, together with the
# `_AtomicSpies` harness, `TestAtomicWritePlan` and
# `TestAtomicWritePlanFollowsSymlink`.  Their coverage did not disappear — it
# MOVED down to the layer that now owns it, `TaskArtifacts._write_json`, in
# orchestrator/tests/test_artifacts.py:
#
#   torn reads: a racing reader sees the  ->  TestWriteJsonIsAtomic
#     complete old or complete new doc,
#     and the name is never absent
#   mode preservation across the swap,    ->  TestWriteJsonPreservesMode
#     stat failures that must surface
#   symlink write-through, the temp       ->  TestWriteJsonFollowsSymlinks
#     BESIDE the real file, dangling
#     refusal, no temp residue
#   that `_write_json` asks safe_io       ->  TestWriteJsonDelegates
#     for fsync=True / mkdir=True at all      ToSharedSafeIo
#
# One piece landed a layer FURTHER down, since `_write_json` no longer owns
# the mechanism it delegates to: the fsync of the temp before the replace
# (and of the parent directory after it) is pinned by
# shared/tests/test_safe_io.py::TestAtomicWriteDurability.
#
# What stays HERE is the assertion that is genuinely plan-tools-level: the
# repair write-back produces the same bytes the ordinary mutation path does.
# That was the anti-duplication invariant while there were two writers; with
# one writer it becomes the self-parity check that a repair write-back never
# churns a plan's formatting.
# ---------------------------------------------------------------------------


class TestRepairWriteBackByteFormat:
    """A repaired plan differs from its predecessor ONLY where it was repaired."""

    def test_the_write_back_stamps_the_schema_version(self, plan_artifacts):
        _seed_corrupt(plan_artifacts)

        repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert any(f['outcome'] == 'repaired' for f in facts)
        written = _on_disk(plan_artifacts)
        assert written['_schema_version'] == PLAN_SCHEMA_VERSION
        assert written['task_id'] == 'test-1'
        assert written == repaired

    def test_byte_format_parity_with_task_artifacts_write_plan(
        self, plan_artifacts, tmp_path
    ):
        """No reindentation, no key-order noise, no permission drift.

        The repaired document that `_read_plan_repaired` writes back must be
        byte-identical to what `TaskArtifacts.write_plan` produces for the
        same document, so a repaired plan diffs only where it was repaired.
        """
        _seed_corrupt(plan_artifacts)
        repaired, _ = plan_tools._read_plan_repaired(plan_artifacts)
        via_write_back = plan_artifacts.root / 'plan.json'

        reference_worktree = tmp_path / 'reference'
        reference_worktree.mkdir()
        reference = TaskArtifacts(reference_worktree)
        reference.init('test-1', 'Test task', 'A test')
        reference.write_plan(copy.deepcopy(repaired))
        via_write_plan = reference.root / 'plan.json'

        assert via_write_back.read_bytes() == via_write_plan.read_bytes()
        # ...and the same PERMISSIONS, a difference the content comparison
        # above cannot see.  This is what catches a writer that lands its
        # temp's 0600 on the target instead of the umask-derived bits.
        assert (
            stat.S_IMODE(via_write_back.stat().st_mode)
            == stat.S_IMODE(via_write_plan.stat().st_mode)
        )


# ---------------------------------------------------------------------------
# step-13 — lazy write-back on read, its idempotence, and boundary row B12.
# ---------------------------------------------------------------------------

#: The stable event name every structured markup fact is logged under (INV-2).
MARKUP_FACT_EVENT = 'plan_markup_repaired'

#: C2's field names verbatim, plus the plan-specific locators this surface adds.
FACT_LOG_KEYS = frozenset({
    'tool', 'param', 'pattern', 'misclose', 'outcome', 'recovered_params',
    'collection', 'index', 'field', 'path',
    # This surface's own additions: the OTHER tools that write the field (so the
    # single ``tool`` does not overstate its precision) and any recovered name
    # the walk refused to write (so a declined recovery cannot read as a
    # successful one).
    'also_written_by', 'declined_params',
})


def _fact_payloads(caplog) -> list[dict]:
    """Every structured fact plan-tools logged, already parsed.

    Consuming a fact must never require regex-scraping prose out of a log line:
    the WHOLE message is the payload, so ``json.loads`` is the only parser
    needed. A record from this logger that is not parseable JSON fails here,
    which is the point — that is what "structured" has to mean to be usable.
    """
    payloads = []
    for record in caplog.records:
        if record.name != plan_tools.__name__ or record.levelno < logging.WARNING:
            continue
        payloads.append(json.loads(record.getMessage()))
    return payloads


@pytest.fixture(autouse=True)
def _isolate_the_refusal_memo():
    """Clear ``_REPORTED_REFUSALS`` around every test in this module.

    The memo is process-local by design (one plan-tools subprocess per agent
    invocation, so "once per process" is "once per session"), which under pytest
    means one set shared by every test in the run. Clearing it here keeps the
    suite order-independent instead of making a later test depend on whether an
    earlier one happened to report the same locator.
    """
    plan_tools._REPORTED_REFUSALS.clear()
    yield
    plan_tools._REPORTED_REFUSALS.clear()


def _seed_mixed_plan(artifacts) -> dict:
    """Write a plan carrying one REPAIRABLE and one UNREPAIRABLE field."""
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
    plan['design_decisions'][1]['decision'] = PROSE_QUOTED
    artifacts.write_plan(copy.deepcopy(plan))
    return plan


class TestReadPlanRepaired:
    """The lazy write-back: repaired on read, in place, atomically."""

    def test_returns_the_repaired_document(self, plan_artifacts):
        _seed_mixed_plan(plan_artifacts)

        plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        assert detect(plan['design_decisions'][0]['rationale']) is None
        outcomes = sorted(fact['outcome'] for fact in facts)
        assert outcomes == ['repaired', 'unrepairable']

    def test_the_repair_is_persisted_to_disk(self, plan_artifacts):
        """The user-observable signal: the FILE is fixed, not just the reply.

        An in-memory-only repair would leave the next agent — and every later
        reader of this plan — looking at the corrupted rationale again.
        """
        _seed_mixed_plan(plan_artifacts)

        plan_tools._read_plan_repaired(plan_artifacts)

        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text())
        assert on_disk['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        # ...and the refusing field is BYTE-IDENTICAL on disk, not sanitized.
        assert on_disk['design_decisions'][1]['decision'] == PROSE_QUOTED

    def test_write_back_leaves_the_rest_of_the_document_intact(self, plan_artifacts):
        seeded = _seed_mixed_plan(plan_artifacts)

        plan_tools._read_plan_repaired(plan_artifacts)

        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text())
        assert on_disk['_schema_version'] == PLAN_SCHEMA_VERSION
        for key in ('task_id', 'title', 'analysis', 'files', 'prerequisites',
                    'steps', 'reuse'):
            assert on_disk[key] == seeded[key]
        assert on_disk['design_decisions'][0]['decision'] == _DECISION_PROSE
        assert on_disk['design_decisions'][1]['rationale'] == 'A second clean rationale.'

    # -- idempotence ------------------------------------------------------

    def test_second_read_repairs_nothing_and_writes_nothing(
        self, plan_artifacts, monkeypatch
    ):
        """A clean (or already-repaired) plan must NOT be rewritten per call.

        Rewriting on every tool call would churn mtimes under every watcher and
        turn a read into a write for the overwhelmingly common clean path.
        """
        _seed_mixed_plan(plan_artifacts)
        plan_path = plan_artifacts.root / 'plan.json'

        first, _ = plan_tools._read_plan_repaired(plan_artifacts)
        before_bytes = plan_path.read_bytes()
        before_mtime = plan_path.stat().st_mtime_ns

        writes: list = []
        real_write_plan = TaskArtifacts.write_plan

        def spy(self, plan):
            writes.append(plan)
            return real_write_plan(self, plan)

        monkeypatch.setattr(TaskArtifacts, 'write_plan', spy)

        second, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert second == first
        assert [f for f in facts if f['outcome'] == 'repaired'] == []
        assert writes == [], 'an already-repaired plan was rewritten'
        assert plan_path.read_bytes() == before_bytes
        assert plan_path.stat().st_mtime_ns == before_mtime

    def test_a_wholly_clean_plan_is_never_written(self, plan_artifacts, monkeypatch):
        plan_artifacts.write_plan(corrupt_plan())
        plan_path = plan_artifacts.root / 'plan.json'
        before_mtime = plan_path.stat().st_mtime_ns
        writes: list = []
        monkeypatch.setattr(
            TaskArtifacts, 'write_plan', lambda self, plan: writes.append(plan)
        )

        plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert facts == []
        assert writes == []
        assert plan_path.stat().st_mtime_ns == before_mtime

    def test_an_unrepairable_only_plan_is_never_written(
        self, plan_artifacts, monkeypatch
    ):
        """Nothing changed, so nothing is written — the 2939 prose plan stays put."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan_artifacts.write_plan(plan)
        plan_path = plan_artifacts.root / 'plan.json'
        before_bytes = plan_path.read_bytes()
        writes: list = []
        monkeypatch.setattr(
            TaskArtifacts, 'write_plan', lambda self, plan: writes.append(plan)
        )

        _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert [f['outcome'] for f in facts] == ['unrepairable']
        assert writes == []
        assert plan_path.read_bytes() == before_bytes

    # -- a refusal is reported ONCE, not on every call ---------------------

    def test_a_repeat_refusal_is_not_re_reported(self, plan_artifacts, caplog):
        """A refusal never converges, so it must not re-announce itself forever.

        A plan that legitimately QUOTES the sentinels in prose (worktree 2939 —
        a plan ABOUT this leak) refuses on every read, for the life of the plan.
        Left unguarded, an architect making ~50 plan-tool calls against it gets
        50 identical warning lines and 50 identical facts fed back into its
        context, each reading like a NEW detection and none actionable. The
        first report keeps the residue visible; the rest are amplification.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan_artifacts.write_plan(plan)

        with caplog.at_level(logging.DEBUG, logger=plan_tools.__name__):
            _, first = plan_tools._read_plan_repaired(plan_artifacts)
            first_warnings = len(_fact_payloads(caplog))
            _, second = plan_tools._read_plan_repaired(plan_artifacts)

        assert [f['outcome'] for f in first] == ['unrepairable']
        assert first_warnings == 1
        assert second == [], (
            'the repeat refusal must drop out of the response too, so a second '
            'call converges to the clean shape instead of re-reporting'
        )
        # Still exactly one WARNING; the repeat went to DEBUG, not to silence.
        assert len(_fact_payloads(caplog)) == 1
        debug_payloads = [
            json.loads(r.getMessage())
            for r in caplog.records
            if r.name == plan_tools.__name__ and r.levelno == logging.DEBUG
        ]
        assert [p['field'] for p in debug_payloads] == ['decision']

    def test_the_response_omits_markup_repairs_entirely_on_a_repeat(
        self, plan_artifacts
    ):
        """The convergence is user-observable through a real tool call."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan_artifacts.write_plan(plan)

        first = plan_tools._add_design_decision(plan_artifacts, 'D.', 'R.')
        second = plan_tools._add_design_decision(plan_artifacts, 'D2.', 'R2.')

        assert [f['outcome'] for f in first['markup_repairs']] == ['unrepairable']
        assert 'markup_repairs' not in second

    def test_an_edited_refusing_value_is_reported_again(self, plan_artifacts):
        """The memo is keyed on the VALUE, not just the locator.

        Suppressing on the locator alone would silence a genuinely NEW refusal
        the moment it landed in a field that had refused before — a stale key
        hiding a fresh fact.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan_artifacts.write_plan(copy.deepcopy(plan))
        _, first = plan_tools._read_plan_repaired(plan_artifacts)
        _, repeat = plan_tools._read_plan_repaired(plan_artifacts)

        plan['design_decisions'][0]['decision'] = 'Different prose. ' + PROSE_QUOTED
        plan_artifacts.write_plan(plan)
        _, after_edit = plan_tools._read_plan_repaired(plan_artifacts)

        assert [f['outcome'] for f in first] == ['unrepairable']
        assert repeat == []
        assert [f['outcome'] for f in after_edit] == ['unrepairable']

    def test_a_repair_is_never_memoized(self, plan_artifacts, caplog):
        """Only refusals are suppressed — a repair converges on its own.

        The second read finds the field already fixed and emits nothing, so
        there is no repetition to suppress; memoizing repairs too would only
        risk hiding a field that got RE-corrupted between calls.
        """
        _seed_mixed_plan(plan_artifacts)
        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            _, first = plan_tools._read_plan_repaired(plan_artifacts)
        assert sorted(f['outcome'] for f in first) == ['repaired', 'unrepairable']

        # Re-corrupt the same field and read again: reported afresh.
        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text())
        on_disk['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
        plan_artifacts.write_plan(on_disk)

        _, again = plan_tools._read_plan_repaired(plan_artifacts)

        assert [f['outcome'] for f in again] == ['repaired']

    # -- the structured fact (INV-2) --------------------------------------

    def test_every_fact_is_logged_as_a_parseable_structured_payload(
        self, plan_artifacts, caplog
    ):
        _seed_mixed_plan(plan_artifacts)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        payloads = _fact_payloads(caplog)
        assert len(payloads) == len(facts) == 2
        for payload in payloads:
            assert payload['event'] == MARKUP_FACT_EVENT
            assert set(payload) >= FACT_LOG_KEYS
            assert payload['outcome'] in {'repaired', 'unrepairable'}
            assert payload['path'] == str(plan_artifacts.root / 'plan.json')
        assert sorted(p['outcome'] for p in payloads) == ['repaired', 'unrepairable']

    def test_the_logged_payload_carries_the_full_locator(self, plan_artifacts, caplog):
        """Enough to point at the exact field without reopening the plan."""
        plan = corrupt_plan()
        plan['reuse'][1]['how'] = TRAILING_HOW
        plan_artifacts.write_plan(plan)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            plan_tools._read_plan_repaired(plan_artifacts)

        payload = _fact_payloads(caplog)[0]
        assert payload['collection'] == 'reuse'
        assert payload['index'] == 1
        assert payload['field'] == 'how'
        assert payload['param'] == 'how'
        assert payload['tool'] == 'add_reuse_item'
        assert payload['outcome'] == 'repaired'
        assert payload['recovered_params'] == []

    def test_each_message_is_a_single_line(self, plan_artifacts, caplog):
        """One fact per log record — never a multi-line blob to reassemble."""
        _seed_mixed_plan(plan_artifacts)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            plan_tools._read_plan_repaired(plan_artifacts)

        messages = [
            r.getMessage() for r in caplog.records if r.name == plan_tools.__name__
        ]
        assert messages
        for message in messages:
            assert '\n' not in message

    # -- degradation ------------------------------------------------------

    def test_missing_plan_degrades_exactly_as_read_plan_does(
        self, plan_artifacts, monkeypatch
    ):
        """No plan.json -> ``{}``, no facts, no write. No NEW failure mode."""
        plan_path = plan_artifacts.root / 'plan.json'
        plan_path.unlink(missing_ok=True)
        assert plan_artifacts.read_plan() == {}
        writes: list = []
        monkeypatch.setattr(
            TaskArtifacts, 'write_plan', lambda self, plan: writes.append(plan)
        )

        plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan == {}
        assert facts == []
        assert writes == []
        assert not plan_path.exists()

    def test_unparseable_plan_raises_exactly_what_read_plan_raises(
        self, plan_artifacts, monkeypatch
    ):
        plan_path = plan_artifacts.root / 'plan.json'
        plan_path.write_text('{not json at all')
        before_bytes = plan_path.read_bytes()
        with pytest.raises(json.JSONDecodeError):
            plan_artifacts.read_plan()
        writes: list = []
        monkeypatch.setattr(
            TaskArtifacts, 'write_plan', lambda self, plan: writes.append(plan)
        )

        with pytest.raises(json.JSONDecodeError):
            plan_tools._read_plan_repaired(plan_artifacts)

        assert writes == []
        assert plan_path.read_bytes() == before_bytes

    def test_a_failed_write_back_still_serves_the_repaired_plan(
        self, plan_artifacts, monkeypatch, caplog
    ):
        """Refusing to serve a repair we could not persist is strictly worse.

        The caller would then get the CORRUPTED text back — the exact failure
        this surface exists to end — because persistence happened to fail.
        """
        _seed_mixed_plan(plan_artifacts)

        # Patched at the writer plan-tools now delegates to, so the contract
        # is still exercised after the module's own writer was retired.
        def boom(self, plan):
            raise ArtifactWriteError('disk on fire')

        monkeypatch.setattr(TaskArtifacts, 'write_plan', boom)

        with caplog.at_level(logging.WARNING, logger=plan_tools.__name__):
            plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan['design_decisions'][0]['rationale'] == _RATIONALE_PROSE
        assert any(f['outcome'] == 'repaired' for f in facts)
        assert 'disk on fire' in caplog.text, 'the write failure must be LOUD'
        assert plan_tools._MARKUP_WRITE_FAILED_EVENT in caplog.text, (
            'the failure must be a STRUCTURED fact, not just prose'
        )


# ---------------------------------------------------------------------------
# task 3982 S1 — an UNRESOLVABLE locator must be reported every time, never
# memoized under a key that means nothing. Drives ``_report_markup_facts``
# directly: it is module-private and callable, which is how this branch is
# reachable even though ``_read_plan_repaired`` can never produce it today
# (it walks the same document it passes back, so its own locators always
# resolve against it).
# ---------------------------------------------------------------------------


def _unresolvable_refusal_fact(**overrides: object) -> dict[str, object]:
    """An 'unrepairable' fact in the exact shape ``_repair_one_field`` emits.

    Defaults to locators that CANNOT resolve against ``corrupt_plan()``:
    ``collection='steps'`` with ``index=99``, far past that plan's two-item
    ``steps`` list.
    """
    fact: dict[str, object] = {
        'tool': 'add_plan_step',
        'also_written_by': [],
        'param': 'description',
        'pattern': 'a mis-close literal',
        'misclose': None,
        'outcome': 'unrepairable',
        'recovered_params': [],
        'declined_params': [],
        'collection': 'steps',
        'index': 99,
        'field': 'description',
    }
    fact.update(overrides)
    return fact


class TestUnresolvableLocatorIsNeverMemoized:
    """S1: a locator that no longer resolves must be reported every time.

    ``_fact_value``'s except branch already carried a comment claiming this
    ('report it every time rather than suppress on a key that means
    nothing'), but returned plain ``None`` — indistinguishable from a
    RESOLVABLE locator whose field is legitimately absent, so the two shared
    one memo key and a SECOND unrelated unresolvable refusal was silently
    suppressed. The comment disclaimed exactly what the code did.
    """

    def test_an_unresolvable_locator_is_reported_on_every_call(self, tmp_path, caplog):
        plan = corrupt_plan()
        fact = _unresolvable_refusal_fact()
        plan_path = tmp_path / 'plan.json'

        with caplog.at_level(logging.DEBUG, logger=plan_tools.__name__):
            first = plan_tools._report_markup_facts(plan_path, plan, [fact])
            second = plan_tools._report_markup_facts(plan_path, plan, [fact])

        assert first == [fact]
        assert second == [fact], (
            'an unresolvable locator must be reported every time, not '
            'suppressed on the second call'
        )
        assert set() == plan_tools._REPORTED_REFUSALS, (
            'no key that means nothing may ever be recorded'
        )
        assert len(_fact_payloads(caplog)) == 2, 'both calls must log at WARNING'
        debug_records = [
            r for r in caplog.records
            if r.name == plan_tools.__name__ and r.levelno == logging.DEBUG
        ]
        assert debug_records == [], (
            'an unresolvable refusal must never be suppressed to DEBUG'
        )

    def test_a_resolvable_locator_with_an_absent_field_is_still_memoized(
        self, tmp_path, caplog
    ):
        """The boundary that must NOT change: a legitimately absent field.

        A locator that DOES resolve, whose field simply holds nothing (the
        holder has no such key), is a different case from a BROKEN locator
        and keeps the existing memoization behaviour. This guards against
        over-widening the S1 fix into 'never memoize a None'.
        """
        plan = corrupt_plan()
        assert 'not_a_real_field' not in plan['steps'][0], (
            'the fixture must pick a field the holder genuinely lacks'
        )
        fact = _unresolvable_refusal_fact(index=0, field='not_a_real_field')
        plan_path = tmp_path / 'plan.json'

        with caplog.at_level(logging.DEBUG, logger=plan_tools.__name__):
            first = plan_tools._report_markup_facts(plan_path, plan, [fact])
            second = plan_tools._report_markup_facts(plan_path, plan, [fact])

        assert first == [fact]
        assert second == [], (
            'a resolvable locator whose field is legitimately absent must '
            'still converge to silence on repeat, unlike a broken locator'
        )
        assert len(plan_tools._REPORTED_REFUSALS) == 1


# ---------------------------------------------------------------------------
# step-15 — EVERY read path repairs, and the triggering tool reports it.
# ---------------------------------------------------------------------------

_NEW_SHA = 'a' * 40


def _seed_corrupt(artifacts) -> None:
    """Put a corrupted-but-otherwise-valid plan on disk for a tool to open."""
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = TRAILING_RATIONALE
    artifacts.write_plan(plan)


def _on_disk(artifacts) -> dict:
    return json.loads((artifacts.root / 'plan.json').read_text())


def _added_step(plan: dict) -> None:
    assert [s['id'] for s in plan['steps']] == ['step-1', 'step-2', 'step-9']


def _added_prereq(plan: dict) -> None:
    assert [p['id'] for p in plan['prerequisites']] == ['pre-1', 'pre-9']


def _added_decision(plan: dict) -> None:
    assert plan['design_decisions'][2] == {
        'decision': 'A newly added decision.',
        'rationale': 'A newly added rationale.',
    }


def _added_reuse(plan: dict) -> None:
    assert plan['reuse'][2]['what'] == 'A newly reused thing'


def _step_marked_done(plan: dict) -> None:
    assert plan['steps'][0]['status'] == 'done'
    assert plan['steps'][0]['commit'] == _NEW_SHA


def _step_pre_committed(plan: dict) -> None:
    assert plan['steps'][0]['status'] == 'done'
    assert plan['steps'][0]['commit'] == _NEW_SHA
    assert plan['steps'][0]['description'].startswith(f'[COMMITTED {_NEW_SHA[:12]}]')


def _metadata_updated(plan: dict) -> None:
    assert plan['files'] == ['orchestrator/src/orchestrator/artifacts.py']
    assert plan['analysis'] == 'A newly written analysis.'


def _step_removed(plan: dict) -> None:
    assert [s['id'] for s in plan['steps']] == ['step-1']


def _step_replaced(plan: dict) -> None:
    assert plan['steps'][1]['type'] == 'impl'
    assert plan['steps'][1]['description'] == 'A newly written step description.'


def _plan_finalized(plan: dict) -> None:
    assert plan['_finalized_at']
    assert plan['_revalidated_at']


#: ``(id, call, expected-response-subset, on-disk-mutation-check)`` for every
#: plan-tools entry point that OPENS an existing plan. ``_create_plan`` is
#: absent on purpose: it overwrites the document wholesale, so it has no read to
#: repair — guarding its INBOUND arguments belongs to the write-time middleware.
#:
#: THIS TABLE IS THE COVERAGE MECHANISM for read-path repair. Each row is
#: exercised behaviourally by ``TestPlanToolsReadPathsRepair`` — the tool is
#: called against a corrupted plan and the ON-DISK result is asserted — so a
#: read site that skipped the repair would fail here, whatever it looked like in
#: source. WHEN A NEW READ SITE IS ADDED, ADD A ROW HERE; that is the whole of
#: the obligation. (A source-text pin over ``inspect.getsource`` used to sit
#: alongside this and was deleted: it locked spelling rather than behaviour, so
#: aliasing the read or moving it into a helper passed it silently while a
#: harmless refactor failed it.)
READ_PATH_CASES = [
    (
        'add_plan_step',
        lambda a: plan_tools._add_plan_step(a, 'step-9', 'impl', 'A newly added step.'),
        {'status': 'ok', 'step_id': 'step-9', 'total_steps': 3},
        _added_step,
    ),
    (
        'add_prerequisite',
        lambda a: plan_tools._add_prerequisite(a, 'pre-9', 'A newly added prerequisite.'),
        {'status': 'ok', 'prereq_id': 'pre-9'},
        _added_prereq,
    ),
    (
        'add_design_decision',
        lambda a: plan_tools._add_design_decision(
            a, 'A newly added decision.', 'A newly added rationale.'
        ),
        {'status': 'ok', 'total_decisions': 3},
        _added_decision,
    ),
    (
        'add_reuse_item',
        lambda a: plan_tools._add_reuse_item(
            a, 'A newly reused thing', 'somewhere.py', 'By importing it.'
        ),
        {'status': 'ok', 'total_reuse': 3},
        _added_reuse,
    ),
    (
        'mark_step_done',
        lambda a: plan_tools._mark_step_done(a, 'step-1', _NEW_SHA),
        {'status': 'ok', 'step_id': 'step-1', 'new_status': 'done', 'commit': _NEW_SHA},
        _step_marked_done,
    ),
    (
        # Contract A1's DELIBERATELY divergent envelope — ``ok`` plus ``status``
        # meaning the step's NEW status. Adding markup_repairs must not disturb it.
        'mark_step_committed',
        lambda a: plan_tools._mark_step_committed(a, 'step-1', _NEW_SHA),
        {'ok': True, 'step_id': 'step-1', 'status': 'done'},
        _step_pre_committed,
    ),
    (
        'update_plan_metadata',
        lambda a: plan_tools._update_plan_metadata(
            a,
            files=['orchestrator/src/orchestrator/artifacts.py'],
            analysis='A newly written analysis.',
        ),
        {'status': 'ok', 'files': 1},
        _metadata_updated,
    ),
    (
        'remove_plan_step',
        lambda a: plan_tools._remove_plan_step(a, 'step-2'),
        {'status': 'ok', 'removed': 'step-2', 'collection': 'steps'},
        _step_removed,
    ),
    (
        'replace_plan_step',
        lambda a: plan_tools._replace_plan_step(
            a, 'step-2', 'impl', 'A newly written step description.'
        ),
        {'status': 'ok', 'replaced': 'step-2'},
        _step_replaced,
    ),
    (
        'confirm_plan',
        lambda a: plan_tools._confirm_plan(a),
        {'status': 'ok', 'finalized': True, 'steps': 2, 'files': 1},
        _plan_finalized,
    ),
]

READ_PATH_IDS = [case[0] for case in READ_PATH_CASES]


@pytest.fixture()
def on_branch(monkeypatch):
    """``mark_step_committed``'s reachability guard, satisfied.

    Stubbed rather than staged as a real commit: this module tests the repair
    surface, and the guard itself is covered by ``test_plan_tools_server``.
    """
    monkeypatch.setattr(plan_tools, '_sha_exists_on_branch', lambda worktree, sha: True)


@pytest.mark.parametrize(
    ('call', 'expected_response', 'check_mutation'),
    [case[1:] for case in READ_PATH_CASES],
    ids=READ_PATH_IDS,
)
class TestPlanToolsReadPathsRepair:
    """Opening the plan through ANY tool repairs it — no tool is a bypass.

    One unhooked read site is enough to leave a corrupted plan corrupted for
    the whole task, so coverage is asserted entry point by entry point rather
    than inferred from the one helper they are all supposed to share.
    """

    def test_the_documented_envelope_is_unchanged(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        _seed_corrupt(plan_artifacts)

        response = call(plan_artifacts)

        for key, value in expected_response.items():
            assert response[key] == value, f'{key!r} changed shape'

    def test_the_on_disk_plan_is_repaired(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        _seed_corrupt(plan_artifacts)

        call(plan_artifacts)

        rationale = _on_disk(plan_artifacts)['design_decisions'][0]['rationale']
        assert rationale == _RATIONALE_PROSE
        assert detect(rationale) is None

    def test_the_requested_mutation_lands_in_the_same_file(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        """The repair must not eat the write it rode in on.

        A repair write-back followed by the tool's own write is two writes to
        one file; if they raced or ordered wrongly, one of them would vanish.
        """
        _seed_corrupt(plan_artifacts)

        call(plan_artifacts)

        plan = _on_disk(plan_artifacts)
        check_mutation(plan)
        assert plan['design_decisions'][0]['rationale'] == _RATIONALE_PROSE

    def test_the_response_reports_what_was_repaired(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        """The agent that opened the plan must SEE what changed under it."""
        _seed_corrupt(plan_artifacts)

        response = call(plan_artifacts)

        repairs = response['markup_repairs']
        assert isinstance(repairs, list)
        assert len(repairs) == 1
        assert repairs[0]['outcome'] == 'repaired'
        assert repairs[0]['collection'] == 'design_decisions'
        assert repairs[0]['index'] == 0
        assert repairs[0]['field'] == 'rationale'

    def test_a_clean_plan_response_is_byte_identical_to_today(
        self, plan_artifacts, on_branch, call, expected_response, check_mutation
    ):
        """No key at all on the common path — not an empty list, not a null.

        Every existing caller and every existing assertion on these envelopes
        must be unable to tell this change happened.
        """
        plan_artifacts.write_plan(corrupt_plan())

        response = call(plan_artifacts)

        assert 'markup_repairs' not in response


# Deliberately NOT a method of TestPlanToolsReadPathsRepair: that class carries
# a class-level parametrize over READ_PATH_CASES, so a method there would run
# ten identical times with three unused arguments. Same section, same subject —
# the ``_create_plan`` half of the read-path coverage story.
class TestCreatePlanIsDeliberatelyNotHooked:
    """``_create_plan`` does not repair, and the next read does. Both asserted.

    This is a CHARACTERIZATION test of existing behaviour, replacing a deleted
    ``inspect.getsource`` pin that asserted the same claim by grepping the
    function's source text for two call spellings. The claim is real and worth
    keeping; the source-text form of it was not, because renaming a local or
    moving the read into a helper would satisfy it without the behaviour
    holding, and a harmless refactor would break it without the behaviour
    changing.
    """

    def test_inbound_arguments_are_not_repaired_but_the_next_read_repairs_them(
        self, plan_artifacts
    ):
        """create_plan overwrites the document, so it has no read to repair.

        Guarding its INBOUND arguments is the write-time middleware's job. The
        residue therefore survives the call verbatim — and is then cleaned up
        by whichever hooked entry point opens the plan next, which is exactly
        the lazy-repair-on-read contract this task implements.
        """
        response = plan_tools._create_plan(
            plan_artifacts, 'test-1', 'A title.', _TRAILING_ANALYSIS, ['a.py']
        )

        assert response == {'status': 'ok', 'task_id': 'test-1'}
        assert 'markup_repairs' not in response
        assert _on_disk(plan_artifacts)['analysis'] == _TRAILING_ANALYSIS

        followup = plan_tools._add_design_decision(plan_artifacts, 'A decision.', 'A rationale.')

        assert followup['status'] == 'ok'
        assert [f['field'] for f in followup['markup_repairs']] == ['analysis']
        analysis = _on_disk(plan_artifacts)['analysis']
        assert analysis == 'Clean analysis prose describing the approach.'
        assert detect(analysis) is None


# ---------------------------------------------------------------------------
# Task 4696 — the SELF-NAME closer, invisible to the read-repair prefilter.
# ---------------------------------------------------------------------------

#: A rationale mis-closed with its OWN tag and NOTHING else: no invoke closer,
#: no parameter-open token. 296 real plan entries have exactly this shape.
_SELF_NAME_RATIONALE_PROSE = 'Both mechanisms partition rather than race.'
_SELF_NAME_RATIONALE = _SELF_NAME_RATIONALE_PROSE + _close('rationale')

#: The same on the second-largest victim, ``add_reuse_item.how`` (129 entries).
_SELF_NAME_HOW_PROSE = 'Reuse the declared table directly.'
_SELF_NAME_HOW = _SELF_NAME_HOW_PROSE + _close('how')

#: THE 4525 SHAPE, verbatim in structure: the field's own closer, then an
#: invoke closer, then the SAME parameter re-declared. repair() refuses it —
#: the invoke closer leads the tail so no candidate parses, and ``invoke`` does
#: not qualify — so the string must be left byte-identical and merely FLAGGED.
_UNREPAIRABLE_RATIONALE = (
    'See plan_tools.py:65-74.'
    + _close('rationale')
    + '\n'
    + _INVOKE_CLOSER
    + '\n'
    + _open_param('rationale')
    + 'See decision text.'
)


def _seed_self_name_plan(artifacts) -> dict:
    """Write a plan whose ONLY corruption is two self-name closers."""
    plan = corrupt_plan()
    plan['design_decisions'][0]['rationale'] = _SELF_NAME_RATIONALE
    plan['reuse'][0]['how'] = _SELF_NAME_HOW
    artifacts.write_plan(copy.deepcopy(plan))
    return plan


class TestSelfNameCloserIsSeenByTheReadRepair:
    """Epsilon's lazy read-repair was gated on a predicate that could not see it.

    ``_carries_markup`` is the cheap prefilter that decides whether the repair
    pass runs at all, and it asked the param-free ``detect``. A plan whose only
    damage is a field mis-closed with its OWN name-echoing tag therefore looked
    CLEAN: the prefilter returned False, the deep copy never happened, and the
    corruption sat on disk untouched read after read — which is exactly why 296
    ``rationale`` and 129 ``how`` specimens were still there five weeks after
    the read-repair went live.

    The repairer behind that gate was correct for them the whole time. Unlike
    the middleware boundary, this site pays NOTHING to be fully schema-aware:
    the walk already yields the ``_PlanField`` record, so ``record.field`` and
    ``record.schema_params`` are both in hand from the DECLARED table.
    """

    def test_the_specimen_is_invisible_to_the_blanket_predicate(self):
        """Otherwise this class would be re-testing an already-caught dialect."""
        for value in (_SELF_NAME_RATIONALE, _SELF_NAME_HOW):
            assert _INVOKE_CLOSER not in value
            assert _LT + 'parameter ' not in value
            assert detect(value) is None

    def test_carries_markup_sees_the_self_name_closers(self):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _SELF_NAME_RATIONALE
        plan['reuse'][0]['how'] = _SELF_NAME_HOW

        assert plan_tools._carries_markup(plan) is True

    def test_a_genuinely_clean_plan_is_still_not_copied(self):
        """The prefilter's whole purpose survives the widening."""
        assert plan_tools._carries_markup(corrupt_plan()) is False

    def test_both_fields_come_back_repaired(self, plan_artifacts):
        _seed_self_name_plan(plan_artifacts)

        plan, _facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert plan['design_decisions'][0]['rationale'] == _SELF_NAME_RATIONALE_PROSE
        assert plan['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE

    def test_the_repair_is_persisted_to_disk(self, plan_artifacts):
        _seed_self_name_plan(plan_artifacts)

        plan_tools._read_plan_repaired(plan_artifacts)

        on_disk = json.loads((plan_artifacts.root / 'plan.json').read_text(encoding='utf-8'))
        assert on_disk['design_decisions'][0]['rationale'] == _SELF_NAME_RATIONALE_PROSE
        assert on_disk['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE

    def test_the_facts_locate_each_repair_by_collection_index_and_field(
        self, plan_artifacts
    ):
        _seed_self_name_plan(plan_artifacts)

        _plan, facts = plan_tools._read_plan_repaired(plan_artifacts)

        located = {
            (f['collection'], f['index'], f['field']): f
            for f in facts
        }
        assert set(located) == {
            ('design_decisions', 0, 'rationale'),
            ('reuse', 0, 'how'),
        }
        for (_collection, _index, field), fact in located.items():
            assert fact['outcome'] == 'repaired'
            assert fact['param'] == field
            assert fact['misclose'] == _close(field)
            assert fact['recovered_params'] == []

    def test_the_4525_shape_is_flagged_UNREPAIRABLE_and_never_guessed(
        self, plan_artifacts
    ):
        """The task's own specimen: refuse, flag, and change not one byte.

        Its tail leads with an invoke closer, so no candidate parses and the
        only other candidate name — ``invoke`` — does not qualify. There is
        nothing to delete and nothing to preserve separately: the tail's
        re-declaration is of the SAME parameter, INSIDE the one string, so the
        "fabricated sibling" is not a sibling key at all. Visible damage beats
        a guessed repair.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _UNREPAIRABLE_RATIONALE
        plan_artifacts.write_plan(copy.deepcopy(plan))
        before = (plan_artifacts.root / 'plan.json').read_bytes()

        repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        flagged = [f for f in facts if f['field'] == 'rationale']
        assert [f['outcome'] for f in flagged] == ['unrepairable']
        assert repaired['design_decisions'][0]['rationale'] == _UNREPAIRABLE_RATIONALE
        assert sorted(repaired['design_decisions'][0]) == ['decision', 'rationale']
        assert (plan_artifacts.root / 'plan.json').read_bytes() == before, (
            'an unrepairable field must leave the file BYTE-IDENTICAL — a '
            'rewrite here would mean something was guessed'
        )

    def test_the_unrepairable_fact_names_the_tag_it_actually_saw(
        self, plan_artifacts
    ):
        """Not ``None``, and not the invoke closer that merely follows it.

        The diagnostic pattern on the refusal path came from the same blind
        predicate, so before this task it named whatever fixed literal happened
        to trail the leak — PRD section 2.2's original complaint, one layer in.
        """
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _UNREPAIRABLE_RATIONALE
        plan_artifacts.write_plan(copy.deepcopy(plan))

        _repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        flagged = [f for f in facts if f['field'] == 'rationale']
        assert flagged[0]['pattern'] == _close('rationale')


#: A ``rationale`` whose prose legitimately ENDS by quoting a SIBLING field's
#: tag pair. ``decision`` is a real sibling parameter of ``add_design_decision``
#: and its closer is NOT one of the fixed ``ENVELOPE_LITERALS``, so it is a name
#: the task-4696 widening contributed and nothing else.
_QUOTED_SIBLING_PROSE = (
    'The harness emits ' + _LT + 'decision>X' + _close('decision')
)


class TestQuotedSiblingTagIsNeverTruncated:
    """A plan that TALKS ABOUT the markup must not be rewritten by the reader.

    The widening at ``_carries_markup`` / ``_repair_one_field`` added every
    sibling ``record.schema_params`` name to the gate, and ``repair`` accepts an
    EMPTY tail — a candidate closer at end-of-string recovers ``{}`` and still
    returns ``clean_value = value[:candidate.start()]``. Composed, a rationale
    ending in ``\x3c/decision>`` was TRUNCATED, reported ``repaired``, and
    persisted atomically by ``_read_plan_repaired``, with nothing left to
    surface the loss. Pre-4696 the blanket ``detect`` returned None and the
    value was left alone, so the loss surface was introduced by that change.

    This is not hypothetical in a repo whose plans discuss tool-call markup —
    the containment PRD itself quotes these tags. And the widening bought
    nothing measured: the PRD's 2026-08-25 census puts the CROSS-FIELD
    population at ZERO (212/212 invisible specimens are self-name).

    The fix is at the shared ``repair`` chokepoint, so the sweep's own
    sibling-key widening is closed by the same mechanism (INV-5).
    """

    def test_the_specimen_was_invisible_before_the_widening(self):
        """Otherwise this class would be pinning pre-existing behaviour."""
        assert detect(_QUOTED_SIBLING_PROSE) is None

    def test_the_value_comes_back_byte_identical(self, plan_artifacts):
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan_artifacts.write_plan(copy.deepcopy(plan))

        repaired, _facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert repaired['design_decisions'][0]['rationale'] == _QUOTED_SIBLING_PROSE

    def test_no_repaired_fact_is_emitted_for_it(self, plan_artifacts):
        """``repaired`` would be an outright false report: nothing was
        recovered, so the only change would have been text DESTROYED."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan_artifacts.write_plan(copy.deepcopy(plan))

        _repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        flagged = [f for f in facts if f['field'] == 'rationale']
        assert [f['outcome'] for f in flagged] == ['unrepairable']
        assert flagged[0]['recovered_params'] == []

    def test_the_file_is_never_rewritten(self, plan_artifacts):
        """The all-refusals branch of ``_read_plan_repaired`` must hold, or the
        truncation would be durable and the mtime would churn under watchers."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan_artifacts.write_plan(copy.deepcopy(plan))
        before = (plan_artifacts.root / 'plan.json').read_bytes()

        plan_tools._read_plan_repaired(plan_artifacts)

        assert (plan_artifacts.root / 'plan.json').read_bytes() == before

    def test_a_self_name_closer_in_the_same_plan_is_still_repaired(
        self, plan_artifacts
    ):
        """The guard is scoped to ``name != param``, so the dialect this task
        exists to fix is untouched — the fix narrows REPAIR, not DETECTION."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['rationale'] = _QUOTED_SIBLING_PROSE
        plan['reuse'][0]['how'] = _SELF_NAME_HOW
        plan_artifacts.write_plan(copy.deepcopy(plan))

        repaired, _facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert repaired['reuse'][0]['how'] == _SELF_NAME_HOW_PROSE
        assert repaired['design_decisions'][0]['rationale'] == _QUOTED_SIBLING_PROSE


# ---------------------------------------------------------------------------
# task 3957 step-7 — plan-tools persists a repaired plan THROUGH TaskArtifacts,
# and keeps no second plan.json writer of its own.
# ---------------------------------------------------------------------------


class TestRepairWriteBackDelegatesToTaskArtifacts:
    """`TaskArtifacts.write_plan` is the SINGLE owner of the plan.json write.

    plan-tools used to carry `_atomic_write_plan` — a parallel implementation
    of the same byte format and the same durability contract, kept only
    because `TaskArtifacts._write_json` was truncate-then-write.  It is not
    any more (task 3957 steps 1-6), so the duplicate has no reason to exist
    and the write-back goes through the public writer.
    """

    @staticmethod
    def _spy_on_write_plan(monkeypatch) -> list[dict]:
        """A DELEGATING spy: records the call, then performs the real write.

        Delegating rather than stubbing is what keeps the on-disk assertions
        meaningful — a stub would prove the call happened while quietly
        turning every "and the bytes landed" check vacuous.
        """
        calls: list[dict] = []
        real_write_plan = TaskArtifacts.write_plan

        def spy(self, plan):
            calls.append(plan)
            return real_write_plan(self, plan)

        monkeypatch.setattr(TaskArtifacts, 'write_plan', spy)
        return calls

    def test_repair_write_back_goes_through_task_artifacts_write_plan(
        self, plan_artifacts, monkeypatch
    ):
        _seed_mixed_plan(plan_artifacts)
        calls = self._spy_on_write_plan(monkeypatch)

        repaired, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert any(f['outcome'] == 'repaired' for f in facts)
        assert len(calls) == 1, (
            f'expected exactly one write_plan call, got {len(calls)} — the '
            'repair write-back must go through TaskArtifacts'
        )
        assert calls[0] == repaired
        # And the delegation really wrote: the bytes on disk are the repair.
        assert _on_disk(plan_artifacts) == repaired

    def test_a_clean_plan_still_writes_nothing(self, plan_artifacts, monkeypatch):
        """Consolidating the writer must not convert a read into a write."""
        plan_artifacts.write_plan(corrupt_plan())
        calls = self._spy_on_write_plan(monkeypatch)

        _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert facts == []
        assert calls == []

    def test_an_all_refusal_plan_still_writes_nothing(
        self, plan_artifacts, monkeypatch
    ):
        """Nothing changed, so nothing is written — the 2939 prose plan stays."""
        plan = corrupt_plan()
        plan['design_decisions'][0]['decision'] = PROSE_QUOTED
        plan_artifacts.write_plan(plan)
        before_bytes = (plan_artifacts.root / 'plan.json').read_bytes()
        calls = self._spy_on_write_plan(monkeypatch)

        _, facts = plan_tools._read_plan_repaired(plan_artifacts)

        assert [f['outcome'] for f in facts] == ['unrepairable']
        assert calls == []
        assert (plan_artifacts.root / 'plan.json').read_bytes() == before_bytes
