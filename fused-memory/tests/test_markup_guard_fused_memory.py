"""Boundary markup-guard tests for fused-memory's BUNDLED FastMCP (task 4458).

PRD ``plans/toolcall-markup-containment-prd.md``, leaf gamma-3. Task 3689
(beta) delivered :class:`shared.mcp_markup_middleware.MarkupGuardMiddleware`
against the STANDALONE ``fastmcp`` package. fused-memory runs the FastMCP
BUNDLED inside the ``mcp`` SDK, whose ``FastMCP`` has no ``add_middleware`` and
no ``get_tool``, so the middleware cannot be attached the documented way.
:mod:`fused_memory.server.markup_guard` adapts it to the bundled
``mcp._tool_manager.call_tool`` chokepoint; these tests pin that adaptation.

What is asserted here that the retiring in-line ``_markup_gate`` could not do:

* ``add_system_record`` and ``update_memory`` — two write tools that had NO
  gate at all, so a leaked ``content`` that swallowed a trailing OPTIONAL
  parameter was stored with that parameter silently ``None``.
* ``repaired_call`` — the COMPLETE argument map with the absorbed sibling
  recovered verbatim, which a single-literal write-time guard cannot produce.

AUTHORING RULE, binding on this whole file: never write a raw MCP envelope
sentinel literal into a source, test or doc file — a file that contains one
becomes a specimen of the very corruption under test and trips the read-side
prefilter. Specimens are BUILT from ``shared.toolcall_markup``'s own constants
by :func:`_leaked`, exactly as ``shared/tests/test_toolcall_markup.py`` and the
middleware itself already do.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# The consolidate cluster harness, reused verbatim rather than re-modelled: a
# bare AsyncMock leaves every `mem0_update` config leaf a Mock, which the
# fail-closed authz resolver rejects, so a hand-rolled double would make the
# consolidate pin below refuse for an authorization reason and pass for the
# wrong one. Cross-test import per tests/test_halt_visibility.py; imported
# MODULE-QUALIFIED so each borrowed constant names its origin at the use site
# and cannot be confused with this file's own specimen constants.
import test_consolidate_memories_tool as _consolidate
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for

from fused_memory.server.main import _install_safe_tool_wrapper
from fused_memory.server.markup_guard import (
    _resolve_project_root,
    install_markup_guard,
)
from fused_memory.server.tools import create_mcp_server

_PROJECT_ID = 'dark_factory'
_PROJECT_ROOT = '/project'

#: The guard's project-attribution map (bare id -> root), mirroring the
#: ``_known_projects_map`` main.py passes at the real registration site.
_KNOWN_PROJECTS = {_PROJECT_ID: _PROJECT_ROOT}

#: add_system_record is recon-stage-only by agent_id convention (its
#: authorization gate runs FIRST in the tool body), so the specimen's swallowed
#: agent_id is a recon-stage one. That keeps the negative control meaningful:
#: a clean call with this identity actually reaches the service instead of
#: bouncing off the authorization gate before the write.
_AGENT_ID = 'recon-stage-9'
_CLEAN_CONTENT = 'a deterministic cycle summary for the merge lane'


def _leaked(clean: str, param: str, value: str) -> str:
    """Build a specimen: *clean* text that mis-closed and absorbed *param*.

    The corpus shape (PRD section 2.1): the caller's text emission closed the
    wrong tag, so everything the envelope emitted AFTER that point — the next
    parameter's opener, name and value — landed inside this argument's string
    instead of being parsed as its own argument.

    Assembled from ``closer_for`` / ``CANONICAL_OPENER_PREFIX`` so no raw
    sentinel is authored here.
    """
    return (
        clean
        + closer_for('parameter')
        + CANONICAL_OPENER_PREFIX
        + f'"{param}">'
        + value
        + closer_for('parameter')
    )


def _pass_through(mock_service: AsyncMock, method: str) -> None:
    """Give *method*'s return value a real ``model_dump``.

    An unspecced AsyncMock chains AsyncMock all the way down, so
    ``result.model_dump()`` would be an unawaited coroutine unless the return
    value is an explicit MagicMock (mirrors
    tests/server/test_markup_tripwire_gate.py::_pass_through).
    """
    result = MagicMock()
    result.model_dump.return_value = {'id': 'ok'}
    getattr(mock_service, method).return_value = result


def _build_guarded_server(
    *methods: str, known_projects: dict[str, str] | None = None
) -> tuple[Any, AsyncMock]:
    """A real bundled-FastMCP server with the boundary guard installed.

    Shape copied from tests/test_tool_safe_wrapper.py::_build_server_with_tool:
    the guard is installed by ``main.py``, NOT by ``create_mcp_server``, so a
    test must install it explicitly — the same shape
    ``_install_safe_tool_wrapper`` already has.
    """
    mock_service = AsyncMock()
    for method in methods:
        _pass_through(mock_service, method)
    server = create_mcp_server(mock_service)
    install_markup_guard(
        server,
        policy=RepairPolicy.REJECT_WITH_REPAIR,
        known_projects=_KNOWN_PROJECTS if known_projects is None else known_projects,
    )
    return server, mock_service


def _payload(exc_info) -> dict:
    """The guard's structured rejection, parsed out of the raised ToolError.

    The guard RAISES rather than returning a dict (prototype P4: a returned
    dict is destroyed by the output schema of any tool annotated ``-> str``),
    so the payload travels as the exception's ``json.dumps`` message.
    """
    return json.loads(str(exc_info.value))


class TestAddSystemRecordBoundary:
    """add_system_record — the first of the two tools that had NO in-line gate.

    Its ``agent_id`` is OPTIONAL and trailing, which is the shape where the
    swallow is SILENT: pydantic defaults the eaten parameter to ``None`` and
    the write lands with a null identity and an isError=False response.
    """

    @pytest.mark.asyncio
    async def test_leaked_content_is_rejected_with_the_absorbed_agent_id_recovered(self):
        server, mock_service = _build_guarded_server('add_system_record')
        arguments = {
            'content': _leaked(_CLEAN_CONTENT, 'agent_id', _AGENT_ID),
            'project_id': _PROJECT_ID,
            'category': 'observations_and_summaries',
        }

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('add_system_record', arguments)

        payload = _payload(exc_info)
        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['tool'] == 'add_system_record'
        assert payload['field'] == 'content'
        assert payload['recovered_params'] == ['agent_id']
        # The COMPLETE argument map, resubmittable verbatim: content restored
        # to the clean prefix and the swallowed agent_id recovered VERBATIM.
        assert payload['repaired_call'] == {
            'content': _CLEAN_CONTENT,
            'project_id': _PROJECT_ID,
            'category': 'observations_and_summaries',
            'agent_id': _AGENT_ID,
        }

    @pytest.mark.asyncio
    async def test_rejected_write_never_reaches_the_service(self):
        """Nothing written — asserted at the service, not by reading storage."""
        server, mock_service = _build_guarded_server('add_system_record')

        with pytest.raises(ToolError):
            await server._tool_manager.call_tool(
                'add_system_record',
                {
                    'content': _leaked(_CLEAN_CONTENT, 'agent_id', _AGENT_ID),
                    'project_id': _PROJECT_ID,
                    'category': 'observations_and_summaries',
                },
            )

        mock_service.add_system_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_service_unchanged(self):
        """Negative control: the guard sits on EVERY call, so a clean one must
        pass through untouched. Without this the suite above would also pass
        against a guard that rejected everything."""
        server, mock_service = _build_guarded_server('add_system_record')

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _CLEAN_CONTENT,
                'project_id': _PROJECT_ID,
                'category': 'observations_and_summaries',
                'agent_id': _AGENT_ID,
            },
        )

        assert result == {'id': 'ok'}
        mock_service.add_system_record.assert_awaited_once()
        assert mock_service.add_system_record.await_args.kwargs['content'] == _CLEAN_CONTENT
        assert mock_service.add_system_record.await_args.kwargs['agent_id'] == _AGENT_ID


class TestUpdateMemoryBoundary:
    """update_memory — the second tool that had NO in-line gate.

    PRD section 2.1's fourth specimen is a real 2026-08-02 update_memory call
    that LANDED precisely because this boundary was ungated. Both parameters
    exercised here (``agent_id``, ``reason``) are OPTIONAL and trailing, the
    shape where the swallow is silent.
    """

    @staticmethod
    def _build():
        server, mock_service = _build_guarded_server()
        mock_service.update_memory.return_value = {'ok': True}
        return server, mock_service

    @staticmethod
    def _arguments(content: str) -> dict[str, Any]:
        return {
            'memory_id': 'mem-1',
            'store': 'mem0',
            'project_id': _PROJECT_ID,
            'content': content,
        }

    @pytest.mark.parametrize(
        ('absorbed', 'absorbed_value'),
        [
            # PRD section 2.1's fourth specimen: the swallowed identity.
            ('agent_id', 'claude-task-4458'),
            # Found among the 95 unrepairable specimens in the 2026-08-19
            # transcript replay: the swallowed amendment rationale.
            ('reason', 'correcting a stale line anchor'),
        ],
    )
    @pytest.mark.asyncio
    async def test_leaked_content_is_rejected_with_the_absorbed_parameter_recovered(
        self, absorbed, absorbed_value
    ):
        server, mock_service = self._build()
        arguments = self._arguments(_leaked(_CLEAN_CONTENT, absorbed, absorbed_value))

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool('update_memory', arguments)

        payload = _payload(exc_info)
        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['tool'] == 'update_memory'
        assert payload['field'] == 'content'
        assert payload['recovered_params'] == [absorbed]
        assert payload['repaired_call'] == {
            **self._arguments(_CLEAN_CONTENT),
            absorbed: absorbed_value,
        }
        # Nothing amended: the tool body never ran.
        mock_service.update_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clean_content_reaches_the_tool_body(self):
        """Negative control, asserted at the boundary the guard actually owns.

        update_memory carries its own kill switch: ``resolve_mem0_update_enabled``
        reads ``mem0_update.enabled`` live off the service config and denies
        unless the leaf is a real ``bool`` — a Mock attribute denies by design.
        So a clean call here lands on ``Mem0UpdateToolDisabled`` rather than on
        the service.

        That is exactly the assertion worth making: reaching the tool body's OWN
        gate proves the guard passed the call through, which is the guard's whole
        contract. Fabricating an authorized mem0_update config would test the
        authz stack instead, and asserting a service call would make this test
        fail whenever an unrelated gate ahead of it changes.
        """
        server, mock_service = self._build()

        result = await server._tool_manager.call_tool(
            'update_memory', self._arguments(_CLEAN_CONTENT)
        )

        # Past the guard (no ToolError), into the body, refused by the body's
        # own knob — never a markup rejection.
        assert result['error_type'] == 'Mem0UpdateToolDisabled'
        mock_service.update_memory.assert_not_awaited()


class TestDeclaredExemption:
    """scan_memory_content is DECLARED exempt (PRD boundary row B7, INV-1).

    Its whole job is searching the corpus for exactly these substrings, so
    guarding it would make the retroactive-sweep tool unable to look for the
    thing it was built to find — the tool that exists because a semantic probe
    for the corrupted records returned ZERO.

    The exemption is load-bearing on a MEASURED wire shape, not a precaution:
    ``needles`` is declared ``list[str] | None``, and the bundled dispatcher
    ACCEPTS it as a JSON STRING and coerces it (probed: a call passing
    ``json.dumps([...])`` runs and returns normally). A JSON-string ``needles``
    is therefore a scanned STRING argument, so without the exemption a sweep
    for the envelope literals is rejected by the guard.
    """

    @staticmethod
    def _build():
        server, mock_service = _build_guarded_server()
        mock_service.scan_memory_content.return_value = {'matches': [], 'scanned': 0}
        return server, mock_service

    def test_scan_memory_content_is_declared_exempt_by_its_bare_name(self):
        """The declaration itself, machine-checkable (INV-1).

        BARE in-server name: ``context.message.name`` is ``scan_memory_content``,
        not the agent-facing ``mcp__fused-memory__scan_memory_content``, and an
        exemption declared with the prefixed spelling would silently never
        match — a declaration that fails open is worse than none.
        """
        from fused_memory.server.markup_guard import EXEMPT_TOOLS

        assert 'scan_memory_content' in EXEMPT_TOOLS
        assert not any(name.startswith('mcp__') for name in EXEMPT_TOOLS)

    @pytest.mark.asyncio
    async def test_json_string_needles_carrying_a_literal_passes_through(self):
        """The discriminating case: a scanned STRING argument full of literals."""
        server, mock_service = self._build()
        needles = json.dumps([closer_for('content'), closer_for('parameter')])

        result = await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': _PROJECT_ID, 'needles': needles}
        )

        assert result['scanned'] == 0
        mock_service.scan_memory_content.assert_awaited_once()
        # UNMODIFIED: the guard neither rejected nor rewrote the sweep's needles.
        assert mock_service.scan_memory_content.await_args.kwargs['needles'] == json.loads(
            needles
        )

    @pytest.mark.asyncio
    async def test_list_needles_carrying_a_literal_passes_through(self):
        """The primary wire shape. Passes with or without the exemption — the
        guard only scans STRING arguments, so a list is skipped either way —
        and is pinned anyway because it is the shape the sweep actually sends
        and a future widening of the scan to list members must not break it."""
        server, mock_service = self._build()
        needles = [closer_for('content')]

        await server._tool_manager.call_tool(
            'scan_memory_content', {'project_id': _PROJECT_ID, 'needles': needles}
        )

        assert mock_service.scan_memory_content.await_args.kwargs['needles'] == needles


class TestOverrideHatch:
    """The deliberate-quoting override, at a newly covered tool.

    ``MarkupGuardMiddleware._apply_override`` decides from the invoked tool's
    OWN LIVE SCHEMA whether the flag travels onward. add_system_record DECLARES
    a ``metadata`` parameter, so the flag is forwarded UNCHANGED and the tool
    body remains the party that owns the rest of the lifecycle.
    """

    @pytest.mark.asyncio
    async def test_override_lets_a_leaked_write_through_with_the_flag_intact(self):
        server, mock_service = _build_guarded_server('add_system_record')
        leaked = _leaked(_CLEAN_CONTENT, 'session_id', 'sess-1')

        result = await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': leaked,
                'project_id': _PROJECT_ID,
                'category': 'observations_and_summaries',
                'agent_id': _AGENT_ID,
                'metadata': {'allow_mcp_markup': True},
            },
        )

        assert result == {'id': 'ok'}
        mock_service.add_system_record.assert_awaited_once()
        kwargs = mock_service.add_system_record.await_args.kwargs
        # Forwarded VERBATIM — not repaired, not stripped by the guard. The
        # caller declared it is quoting the markup deliberately.
        assert kwargs['content'] == leaked

    @pytest.mark.asyncio
    async def test_a_non_literal_true_flag_does_not_unlock_the_guard(self):
        """The truthiness case. ``markup_override_requested`` requires the value
        to be literally ``True``; a truthy string must NOT open the hatch, or
        the override becomes an accident anyone can trip into."""
        server, mock_service = _build_guarded_server('add_system_record')

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_system_record',
                {
                    'content': _leaked(_CLEAN_CONTENT, 'session_id', 'sess-1'),
                    'project_id': _PROJECT_ID,
                    'category': 'observations_and_summaries',
                    'agent_id': _AGENT_ID,
                    'metadata': {'allow_mcp_markup': 'yes'},
                },
            )

        assert _payload(exc_info)['error_type'] == 'mcp_markup_detected'
        mock_service.add_system_record.assert_not_awaited()


class TestOverrideFlagNeverReachesPersistence:
    """``allow_mcp_markup`` is a WRITE-TIME control, not corpus metadata.

    The lifecycle, which is why this needs a behavioural pin at all:
    ``MarkupGuardMiddleware._apply_override`` decides from the invoked tool's
    OWN LIVE SCHEMA whether the flag travels onward, and deliberately forwards
    ``metadata`` UNCHANGED to any tool DECLARING a ``metadata`` parameter. So
    the boundary guard is NOT the party that strips it — the tool body is, and
    a tool body that forgets writes the flag straight into the corpus, where it
    rides along on every future read of a record that needed it exactly once.

    SCOPE — audited over the tool surface and deliberately narrow. 18 tools
    declare a top-level ``metadata`` parameter; only THREE capture the cleaned
    dict (``causation_id, source, cleaned_meta = _extract_causation(...)``) and
    can therefore persist it. ``add_memory`` already strips. The two pinned
    here are the remaining ones:

    * ``add_system_record`` -> ``memory_service.add_system_record(metadata=...)``
    * ``consolidate_memories`` -> ``canonical_meta`` -> ``add_memory(metadata=...)``

    The other 15 discard it (``causation_id, source, _ = _extract_causation``)
    and cannot leak it, so they are not pinned — a strip there would be noise
    that dilutes the signal that these two sites are load-bearing.

    Every assertion co-passes an ordinary caller key and asserts it SURVIVES,
    so "strip the flag" cannot be satisfied by dropping metadata wholesale.
    One shape, mirroring tests/server/test_markup_tripwire_gate.py's
    ``test_override_flag_is_stripped_before_persistence``.
    """

    @pytest.mark.asyncio
    async def test_add_system_record_strips_the_flag_before_persistence(self):
        server, mock_service = _build_guarded_server('add_system_record')
        leaked = _leaked(_CLEAN_CONTENT, 'session_id', 'sess-1')

        await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': leaked,
                'project_id': _PROJECT_ID,
                'category': 'observations_and_summaries',
                'agent_id': _AGENT_ID,
                'metadata': {'allow_mcp_markup': True, 'kind': 'cycle_summary'},
            },
        )

        persisted = mock_service.add_system_record.await_args.kwargs['metadata'] or {}
        assert 'allow_mcp_markup' not in persisted, (
            f'the write-time control flag must not reach Mem0, got: {persisted!r}'
        )
        assert persisted.get('kind') == 'cycle_summary', (
            f"the caller's own metadata must survive the strip, got: {persisted!r}"
        )
        # The override still forwards the PAYLOAD verbatim — the caller
        # declared it is quoting the markup deliberately. Stripping the flag
        # must not be confused with repairing the content.
        assert mock_service.add_system_record.await_args.kwargs['content'] == leaked

    @pytest.mark.asyncio
    async def test_consolidate_memories_strips_the_flag_before_the_canonical_write(
        self,
    ):
        """The second persisting tool, which has no markup coverage today.

        Its ``cleaned_meta`` is the BASE of ``canonical_meta``, so an unstripped
        flag is written into the one record consolidation creates to outlive
        the whole cluster it folds.

        Harness per tests/test_consolidate_memories_tool.py: a bare AsyncMock
        makes every ``mem0_update`` config leaf a Mock, which the fail-closed
        authz resolver rejects — so this drives that module's ``make_service``
        rather than ``_build_guarded_server``, or the call would refuse for an
        authorization reason and pass for the wrong one.
        """
        svc = _consolidate.make_service()
        server = create_mcp_server(svc)
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects=_KNOWN_PROJECTS,
        )

        await server._tool_manager.call_tool(
            'consolidate_memories',
            {
                'canonical_content': _leaked(_consolidate.CONTENT, 'run_id', _consolidate.RUN_ID),
                'topic': _consolidate.TOPIC,
                'project_id': _PROJECT_ID,
                'supersedes': list(_consolidate.SUPERSEDES),
                'run_id': _consolidate.RUN_ID,
                'agent_id': _consolidate.AGENT,
                'metadata': {'allow_mcp_markup': True, 'kind': 'cycle_summary'},
            },
        )

        svc.add_memory.assert_awaited_once()
        persisted = svc.add_memory.await_args.kwargs['metadata'] or {}
        assert 'allow_mcp_markup' not in persisted, (
            f'the write-time control flag must not reach the canonical record, '
            f'got: {persisted!r}'
        )
        assert persisted.get('kind') == 'cycle_summary', (
            f"the caller's own metadata must survive the strip, got: {persisted!r}"
        )
        # The consolidation keys this op stamps ON TOP of the caller's metadata
        # must be untouched by the strip — they are what makes the record
        # findable as the cluster's canonical.
        assert persisted.get('topic') == _consolidate.TOPIC
        assert persisted.get('canonical') is True
        assert list(persisted.get('supersedes') or []) == list(_consolidate.SUPERSEDES)


# ---------------------------------------------------------------------------
# The storm escape (INV-4), and the two defects this replacement must fix.
# ---------------------------------------------------------------------------

#: The middleware's threshold is 3 in a 3600s window, so three rejections in one
#: test fire exactly one burst. No injected clock is needed: real time cannot
#: leave a 3600s window during three in-process calls, and a fake clock here
#: would test StormCounter (already covered by its own suite) rather than the
#: sink wiring this class is about.
_BURST = 3


def _escalations(root) -> list[dict]:
    """Every escalation record filed under *root*, parsed."""
    return [
        json.loads(path.read_text())
        for path in sorted((root / 'data' / 'escalations').glob('esc-*.json'))
    ]


class TestStormEscape:
    """A BURST means the upstream serialization leak is running right now.

    One corrupted call is routine (the measured rate is 0.26%); a burst is the
    operator-facing event, and it is the only signal that survives a caller
    which simply retries forever.
    """

    @staticmethod
    def _server(known_projects):
        """A guarded server whose project map points at the test's queue dir."""
        return _build_guarded_server('add_system_record', known_projects=known_projects)

    @staticmethod
    async def _burst(server, project_id=_PROJECT_ID, n=_BURST):
        for i in range(n):
            with pytest.raises(ToolError) as exc_info:
                await server._tool_manager.call_tool(
                    'add_system_record',
                    {
                        'content': _leaked(f'{_CLEAN_CONTENT} {i}', 'agent_id', _AGENT_ID),
                        'project_id': project_id,
                        'category': 'observations_and_summaries',
                    },
                )
        return _payload(exc_info)

    @pytest.mark.asyncio
    async def test_a_burst_files_under_an_anchor_the_l1_watcher_does_not_share(
        self, tmp_path
    ):
        """The measured defect (a).

        The L1 escalation watcher files its cluster records under the
        'markup-tripwire' anchor and SQUATS it — measured: the tripwire filed
        nothing 2026-08-16..2026-08-19 while 41 rejections occurred, all 17
        records at dedupe_count 0. A boundary guard inheriting that anchor would
        inherit the silence, which reads as calm rather than as suppression.
        """
        server, _ = self._server({_PROJECT_ID: str(tmp_path)})

        await self._burst(server)

        records = _escalations(tmp_path)
        assert len(records) == 1, f'expected exactly one burst record: {records!r}'
        assert records[0]['task_id'] == 'markup-guard'
        assert records[0]['task_id'] != 'markup-tripwire'

    @pytest.mark.asyncio
    async def test_an_open_markup_tripwire_record_does_not_suppress_the_burst(
        self, tmp_path
    ):
        """The squat, reproduced: an open record on the OLD anchor must not
        silence the new guard."""
        from fused_memory.server.markup_tripwire import emit_markup_storm_escalation

        squatter = emit_markup_storm_escalation(str(tmp_path), {'count': 9})
        assert squatter is not None, 'escalation package unavailable in this environment'
        server, _ = self._server({_PROJECT_ID: str(tmp_path)})

        await self._burst(server)

        anchors = {record['task_id'] for record in _escalations(tmp_path)}
        assert anchors == {'markup-tripwire', 'markup-guard'}

    @pytest.mark.asyncio
    async def test_a_bare_project_id_is_translated_through_known_projects(
        self, tmp_path
    ):
        """The measured defect (b).

        ``MarkupGuardMiddleware._identity`` resolves ``project_root`` FIRST then
        ``project_id``, so a leaked add_system_record — which carries only a
        project_id — yields a bare id like 'dark_factory', not a path. Filing
        against that would try to open a queue at a relative directory named
        after the project. The sink translates it through known_projects, the
        same ``_kp.get(project_id)`` translation the in-line gate does.
        """
        server, _ = self._server({_PROJECT_ID: str(tmp_path)})

        await self._burst(server)

        assert len(_escalations(tmp_path)) == 1
        # And nothing was created at a path named after the bare project id.
        assert not (tmp_path / _PROJECT_ID).exists()

    @pytest.mark.asyncio
    async def test_an_unresolvable_project_files_nothing(self, tmp_path):
        """An escalation filed against a guessed default is worse than one an
        operator has to place by hand, so an unresolvable project files NOTHING
        — but the rejection itself is unaffected."""
        server, _ = self._server({'some_other_project': str(tmp_path)})

        payload = await self._burst(server)

        assert payload['error_type'] == 'mcp_markup_detected'
        assert not (tmp_path / 'data' / 'escalations').exists()

    @pytest.mark.asyncio
    async def test_the_filed_record_routes_at_the_live_prd(self, tmp_path):
        """The measured defect (c): routing text.

        Carries forward the correction commit e0ea6e3fe9 made to the ERROR line
        inside the _markup_gate closure that step-12 deletes. DF 3083 is DONE
        and CLOSED to appends, so a reader sent to report a recurrence there
        reports it nowhere.
        """
        server, _ = self._server({_PROJECT_ID: str(tmp_path)})

        await self._burst(server)

        record = _escalations(tmp_path)[0]
        routing = f'{record["summary"]}\n{record["detail"]}\n{record["suggested_action"]}'
        assert 'toolcall-markup-containment-prd.md' in routing, (
            f'must name the live owner: {routing!r}'
        )
        # 3083 may be NAMED as the closed predecessor — what must not happen is
        # the reader being DIRECTED to report a recurrence against it. That
        # correction is stated explicitly, so assert it explicitly.
        assert 'not against 3083' in routing, (
            f'must send recurrences to the PRD, not the closed predecessor: {routing!r}'
        )

    @pytest.mark.asyncio
    async def test_the_operator_facing_error_line_names_the_prd(self, tmp_path, caplog):
        """The greppable half. The payload folded into the response reaches only
        the leaking caller — the one party that already knows — so the operator's
        copy cannot ride on it."""
        server, _ = self._server({_PROJECT_ID: str(tmp_path)})

        with caplog.at_level('ERROR', logger='fused_memory.server.markup_guard'):
            await self._burst(server)

        mine = [
            r for r in caplog.records
            if r.name == 'fused_memory.server.markup_guard' and r.levelname == 'ERROR'
        ]
        assert mine, f'expected one greppable ERROR line; got {caplog.records!r}'
        text = '\n'.join(r.getMessage() for r in mine)
        assert 'toolcall-markup-containment-prd.md' in text
        assert 'see DF 3083' not in text, (
            f'must not direct the reader at the closed predecessor: {text!r}'
        )
        # The visible-counter half of the task's remedy, without a schema
        # change. Asserted IN CONTEXT: a bare `str(_BURST) in text` is vacuous
        # here, because '3' also appears in 'threshold=3', in '3600.0s' and in
        # the routing text's '3083' — it would pass with the count rendered as
        # None or as the wrong number.
        assert f'markup_guard_storm: {_BURST} rejected outcome(s)' in text, (
            f'must state the observed count where a reader can attribute it: {text!r}'
        )

    @pytest.mark.asyncio
    async def test_a_sink_that_raises_never_changes_the_rejection(
        self, tmp_path, monkeypatch
    ):
        """The measured defect (d), and the reason the emit is wrapped.

        The rejection is already decided by the time the sink runs, so
        escalation is purely additive — the same reasoning the retiring
        _markup_gate applies to its own emitter.
        """
        import fused_memory.server.markup_guard as guard_module

        def _boom(*args, **kwargs):
            raise RuntimeError('queue on fire')

        monkeypatch.setattr(guard_module, 'emit_markup_storm_escalation', _boom)
        server, mock_service = self._server({_PROJECT_ID: str(tmp_path)})

        payload = await self._burst(server)

        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['repaired_call']['agent_id'] == _AGENT_ID
        mock_service.add_system_record.assert_not_awaited()


# ---------------------------------------------------------------------------
# step-7: the guard must be LIVE on the real server, installed in the ONE
# order prototype P4 forces.
# ---------------------------------------------------------------------------


def _leaked_add_system_record_arguments() -> dict[str, Any]:
    """The step-1 specimen, reused so the order tests compare like with like."""
    return {
        'content': _leaked(_CLEAN_CONTENT, 'agent_id', _AGENT_ID),
        'project_id': _PROJECT_ID,
        'category': 'observations_and_summaries',
    }


def _build_server_installed(*, guard_first: bool) -> tuple[Any, AsyncMock]:
    """A real server with BOTH ``call_tool`` wrappers, installed in either order.

    Each wrapper is idempotent against its own sentinel and wraps whatever
    ``call_tool`` it finds, so the install order alone decides which one ends up
    OUTSIDE — which is exactly the variable under test.
    """
    mock_service = AsyncMock()
    _pass_through(mock_service, 'add_system_record')
    server = create_mcp_server(mock_service)

    def _guard() -> None:
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects=_KNOWN_PROJECTS,
        )

    if guard_first:
        _guard()
        _install_safe_tool_wrapper(server)
    else:
        _install_safe_tool_wrapper(server)
        _guard()
    return server, mock_service


class TestInstalledOnTheRealServer:
    """(a) The installer helper does what the real server relies on it doing.

    A guard that only ever runs in its own tests contains nothing, so these
    drive ``_install_tool_dispatch_guards`` — the helper ``run_server`` calls —
    against a real ``create_mcp_server`` product, rather than driving
    ``run_server`` itself, which builds stores, a reconciliation harness and two
    uvicorn servers.

    KNOWN GAP: the last link, that ``run_server`` actually calls this helper, is
    deliberately NOT pinned here. It was, by asserting over
    ``run_server.__code__.co_names``, and that pin was deleted as unsound: a bare
    reference to the name satisfied it without the call being reached
    (fails-open), while moving the call into a nested helper broke it with the
    wiring intact (fails-closed on a benign refactor). Pinning it honestly needs
    the construction half of ``run_server`` extracted the way
    ``_build_recon_report_components`` already was; until then the link is held
    by review, not by a test. No assertion here reads source text.
    """

    def test_the_installer_helper_declares_the_reject_with_repair_tier(self):
        """INV-1: the tier is declared at the interception point, not inferred.

        REJECT_WITH_REPAIR is the PRD's declared tier for fused-memory
        (section 4, C2). It also settles the PRD's open question 2:
        add_system_record and update_memory take this SERVER default rather than
        a per-tool tier, because a per-tool tier would be exactly the tool-name
        inference INV-1 forbids.
        """
        from fused_memory.server import main as main_module

        recorded: dict[str, Any] = {}

        def _record(mcp: Any, **kwargs: Any) -> None:
            recorded['mcp'] = mcp
            recorded.update(kwargs)

        known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        server = create_mcp_server(AsyncMock())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(main_module, 'install_markup_guard', _record)
            main_module._install_tool_dispatch_guards(
                server, known_projects=known_projects
            )

        assert recorded['mcp'] is server
        assert recorded['policy'] is RepairPolicy.REJECT_WITH_REPAIR
        assert recorded['known_projects'] is known_projects

    def test_an_undeliverable_policy_is_refused_at_WIRING_time(self):
        """The declared ValueError, which is a fail-LOUD wiring guard.

        ``MarkupGuardMiddleware._forward`` returns a fastmcp ``ToolResult`` that
        the bundled dispatcher cannot consume, so FORWARD_REPAIR does not merely
        behave differently here — it breaks at the FIRST repair, arbitrarily far
        from the line that declared it. A tier that cannot work has to fail at
        the wiring, and the tool manager must be left unwrapped so the failure
        cannot be half-applied.
        """
        server = create_mcp_server(AsyncMock())
        before = server._tool_manager.call_tool

        with pytest.raises(ValueError) as exc_info:
            install_markup_guard(server, policy=RepairPolicy.FORWARD_REPAIR)

        assert 'FORWARD_REPAIR' in str(exc_info.value) or 'REJECT_WITH_REPAIR' in str(
            exc_info.value
        )
        # ``==`` not ``is``: an UNWRAPPED call_tool is a bound method, and every
        # attribute access mints a fresh bound-method object, so identity would
        # fail here even with nothing installed. Equality compares (func, self),
        # which is exactly the question — and once wrapped the attribute holds a
        # plain closure, which no bound method equals.
        assert server._tool_manager.call_tool == before
        assert getattr(server._tool_manager, '_fused_memory_markup_guarded', False) is False

    def test_installing_twice_does_not_double_wrap(self):
        """The ``_fused_memory_markup_guarded`` sentinel's whole job.

        Without it a second install would chain a second guard around the first:
        every call scanned twice, and a rejection counted twice into the storm
        window, so a burst would fire at half the declared threshold. Re-entry is
        routine — tests install onto a server ``main.py`` may already have
        guarded, mirroring ``_install_safe_tool_wrapper``'s own sentinel.
        """
        server = create_mcp_server(AsyncMock())
        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=_KNOWN_PROJECTS
        )
        wrapped = server._tool_manager.call_tool

        install_markup_guard(
            server, policy=RepairPolicy.REJECT_WITH_REPAIR, known_projects=_KNOWN_PROJECTS
        )

        assert server._tool_manager.call_tool is wrapped

    def test_the_installer_helper_installs_both_wrappers(self):
        """Unpatched, the helper leaves both sentinels set on the one tool
        manager — the guard is added to the defence-in-depth wrapper, never
        substituted for it."""
        from fused_memory.server.main import _install_tool_dispatch_guards

        server = create_mcp_server(AsyncMock())
        _install_tool_dispatch_guards(server, known_projects=_KNOWN_PROJECTS)

        manager = server._tool_manager
        assert getattr(manager, '_fused_memory_safe_wrapped', False) is True
        assert getattr(manager, '_fused_memory_markup_guarded', False) is True


class TestInstallationOrder:
    """(b) The substantive assertion, and the regression it prevents.

    The bundled ``Tool.run`` wraps EVERY tool-body exception into ToolError, so
    ``_safe_call_tool`` cannot be taught to re-raise ToolError without gutting
    the containment tests/test_tool_safe_wrapper.py pins. The guard therefore
    has to sit OUTSIDE it. That is a property of the pair, so it is pinned by a
    test rather than by a comment.
    """

    @pytest.mark.asyncio
    async def test_guard_outside_the_safe_wrapper_delivers_the_payload_intact(self):
        server, mock_service = _build_server_installed(guard_first=False)

        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_system_record', _leaked_add_system_record_arguments()
            )

        payload = _payload(exc_info)
        assert payload['error_type'] == 'mcp_markup_detected'
        assert payload['field'] == 'content'
        assert payload['repaired_call']['content'] == _CLEAN_CONTENT
        assert payload['repaired_call']['agent_id'] == _AGENT_ID
        mock_service.add_system_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_guard_inside_the_safe_wrapper_degrades_the_payload(self):
        """The negative control: reverse the two calls and the diagnosis is lost.

        The rejection still happens — nothing is written either way — but the
        safe wrapper catches the guard's ToolError and flattens it into its own
        ``{'error': str, 'error_type': str}`` shape. ``repaired_call`` stops
        being a key the caller can read and survives only as text inside an
        opaque string, so the agent cannot resubmit the repair. This is why the
        two calls must not be 'tidied' into the other order.
        """
        server, mock_service = _build_server_installed(guard_first=True)

        result = await server._tool_manager.call_tool(
            'add_system_record', _leaked_add_system_record_arguments()
        )

        assert isinstance(result, dict)
        assert result['error_type'] == 'ToolError'
        assert 'repaired_call' not in result
        assert 'error_type' in result and result['error_type'] != 'mcp_markup_detected'
        # The diagnosis is not destroyed, only buried: it is now a substring of
        # an opaque error message instead of a structured field.
        assert 'repaired_call' in result['error']
        mock_service.add_system_record.assert_not_awaited()


class TestReconReportServerIsNotGuarded:
    """(c) The second ``_install_safe_tool_wrapper`` call site stays bare.

    The recon-report server hosts the cite_* / report tools and no write tools,
    so it has no write boundary to guard. Asserting that keeps it from being
    wired by accident when the two call sites are next edited together.
    """

    def _make_config(self) -> Any:
        from fused_memory.config.schema import (
            FusedMemoryConfig,
            ReconciliationConfig,
            ServerConfig,
        )

        return FusedMemoryConfig(
            server=ServerConfig(recon_report_port=8003, host='127.0.0.1'),
            reconciliation=ReconciliationConfig(recon_report_state_ttl_seconds=300),
        )

    def test_recon_report_server_gets_the_safe_wrapper_but_not_the_guard(self):
        from fused_memory.server.main import _build_recon_report_components

        _, mcp, _ = _build_recon_report_components(self._make_config())

        manager = mcp._tool_manager
        assert getattr(manager, '_fused_memory_safe_wrapped', False) is True
        assert getattr(manager, '_fused_memory_markup_guarded', False) is False


# ---------------------------------------------------------------------------
# amend: the OUTER wrapper's own containment, and the registry check on the
# label it files against.
# ---------------------------------------------------------------------------


_CLEAN_ADD_SYSTEM_RECORD = {
    'content': _CLEAN_CONTENT,
    'project_id': _PROJECT_ID,
    'category': 'observations_and_summaries',
    'agent_id': _AGENT_ID,
}


class TestTheGuardContainsItsOwnDefects:
    """Being OUTERMOST puts this layer outside ``_safe_call_tool``'s net.

    The installation order is deliberate and pinned by
    :class:`TestInstallationOrder`: the guard has to sit OUTSIDE the
    defence-in-depth wrapper for its ``ToolError`` to survive. The cost is that
    the guard's own code — ``repair()``, ``json.dumps`` over an exotic argument
    value, the shims — no longer has anything catching it, and an escape from
    here reaches StreamableHTTPSessionManager's shared task group and cascades
    into uvicorn. That is exactly the failure class ``_safe_call_tool`` exists to
    prevent, so the outer layer carries its own containment.

    ToolError's passage is NOT re-asserted here — every rejection test in this
    file already depends on it, and a regression that swallowed it would fail
    them all.
    """

    @staticmethod
    def _break_guard(monkeypatch, impl):
        from shared.mcp_markup_middleware import MarkupGuardMiddleware

        monkeypatch.setattr(MarkupGuardMiddleware, 'on_call_tool', impl)

    @pytest.mark.asyncio
    async def test_a_defect_in_the_guard_fails_OPEN_to_the_contained_dispatcher(
        self, monkeypatch, caplog
    ):
        """A scan that crashed contained nothing anyway.

        So the call is re-dispatched to the inner, still-contained wrapper
        rather than poisoning the task group: the write is no more dangerous
        than it was before this guard existed, and the operator gets a
        traceback saying the call went UNSCANNED.
        """
        async def _boom(self, context, call_next):
            raise RuntimeError('a defect in the guard itself')

        self._break_guard(monkeypatch, _boom)
        server, mock_service = _build_guarded_server('add_system_record')

        with caplog.at_level('ERROR', logger='fused_memory.server.markup_guard'):
            result = await server._tool_manager.call_tool(
                'add_system_record', dict(_CLEAN_ADD_SYSTEM_RECORD)
            )

        assert result == {'id': 'ok'}
        mock_service.add_system_record.assert_awaited_once()
        text = '\n'.join(r.getMessage() for r in caplog.records)
        assert 'NOT scanned' in text, f'the operator must be told: {text!r}'

    @pytest.mark.asyncio
    async def test_a_defect_AFTER_forwarding_never_doubles_the_write(
        self, monkeypatch
    ):
        """Fail-open must not mean fail-twice.

        If the guard raises once the call has already been forwarded, the tool
        body has ALREADY RUN — re-dispatching would write twice, which is worse
        than any containment it buys. The exception is let out instead.
        """
        async def _boom_after(self, context, call_next):
            await call_next(context)
            raise RuntimeError('a defect after the forward')

        self._break_guard(monkeypatch, _boom_after)
        server, mock_service = _build_guarded_server('add_system_record')

        with pytest.raises(RuntimeError):
            await server._tool_manager.call_tool(
                'add_system_record', dict(_CLEAN_ADD_SYSTEM_RECORD)
            )

        assert mock_service.add_system_record.await_count == 1

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates(self, monkeypatch):
        """The one exception the containment must NOT absorb.

        Swallowing CancelledError leaks tasks and breaks structured
        concurrency — the same contract ``_safe_call_tool`` states first.
        """
        async def _cancel(self, context, call_next):
            raise asyncio.CancelledError()

        self._break_guard(monkeypatch, _cancel)
        server, mock_service = _build_guarded_server('add_system_record')

        with pytest.raises(asyncio.CancelledError):
            await server._tool_manager.call_tool(
                'add_system_record', dict(_CLEAN_ADD_SYSTEM_RECORD)
            )

        mock_service.add_system_record.assert_not_awaited()


class TestTheFiledProjectRootIsRegistryVouched:
    """The queue root is never a value the CALLER chose.

    ``MarkupGuardMiddleware._identity`` reads the RAW ``project_root`` argument
    at the dispatch boundary — before ``submit_task``/``update_task`` normalise
    it through ``resolve_main_checkout`` — and ``EscalationQueue.__init__`` does
    ``mkdir(parents=True, exist_ok=True)``. Passing an unvouched absolute string
    through would therefore let three leaked writes create a
    ``<anywhere>/data/escalations/`` tree on this host. The retired
    ``_markup_gate`` never had that exposure: it saw either an already-normalized
    root or a ``_kp.get(project_id)`` lookup.
    """

    def test_a_registered_root_resolves_to_itself(self):
        assert _resolve_project_root('/registered', {'p': '/registered'}) == '/registered'

    def test_an_unvouched_absolute_path_resolves_to_nothing(self):
        assert _resolve_project_root('/anything/at/all', {'p': '/registered'}) is None

    def test_a_bare_id_is_still_translated_through_the_registry(self):
        assert _resolve_project_root('p', {'p': '/registered'}) == '/registered'

    def test_an_unknown_id_and_an_absent_registry_resolve_to_nothing(self):
        assert _resolve_project_root('nope', {'p': '/registered'}) is None
        assert _resolve_project_root('/registered', None) is None
        assert _resolve_project_root('', {'p': '/registered'}) is None
        assert _resolve_project_root(None, {'p': '/registered'}) is None

    @pytest.mark.asyncio
    async def test_a_burst_carrying_a_caller_chosen_root_creates_no_queue_there(
        self, tmp_path, monkeypatch
    ):
        """The behavioural half, at the one tool shape that carries a path.

        submit_task takes ``project_root`` from the caller, so a leaking agent
        picks the label this sink would file under.
        """
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p)
        )
        interceptor = AsyncMock()
        interceptor.submit_task.return_value = {'ticket': 'tkt_x'}
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects={_PROJECT_ID: str(tmp_path / 'registered')},
        )
        chosen = tmp_path / 'chosen-by-the-caller'

        for i in range(_BURST):
            with pytest.raises(ToolError):
                await server._tool_manager.call_tool(
                    'submit_task',
                    {
                        'project_root': str(chosen),
                        'prompt': f'Reconcile task {i}',
                        'description': _leaked(
                            f'Reconcile task {i}', 'agent_id', _AGENT_ID
                        ),
                    },
                )

        assert not chosen.exists(), (
            f'the guard created a queue at a caller-chosen path: '
            f'{sorted(tmp_path.rglob("*"))!r}'
        )
        interceptor.submit_task.assert_not_called()
