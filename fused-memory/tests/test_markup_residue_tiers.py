"""The residue preservation tiers of fused-memory's markup guard (task 4809).

PRD ``plans/toolcall-markup-containment-prd.md`` C2/B5/INV-7: a refusal must
not destroy the payload it refuses. The guard hands an unrepairable payload to
the first tier that takes it: the project's escalation queue, routed by the
call's identity or else re-derived from the residue; then the server's
``MarkupResidueSpool`` floor; then its own ERROR log line.
``get_status.markup_residue`` counts what no tier held (INV-4).

The boundary guard itself is pinned in ``tests/test_markup_guard_fused_memory.py``.
Both files build their servers with ``tests/_markup_guard_harness.py``.

AUTHORING RULE: never write a raw MCP envelope sentinel literal here. Specimens
are BUILT from ``shared.toolcall_markup``'s constants, and
``tests/scripts/test_no_raw_envelope_literal.py`` enforces that for every file
importing that module.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from _markup_guard_harness import (
    _AGENT_ID,
    _CLEAN_CONTENT,
    _PROJECT_ID,
    _build_guarded_server,
    _escalations,
    _payload,
)
from fastmcp.exceptions import ToolError
from shared.mcp_markup_middleware import MARKUP_RECURRENCE_ROUTING, RepairPolicy
from shared.toolcall_markup import CANONICAL_OPENER_PREFIX, closer_for
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

from fused_memory.server.markup_guard import ResidueTier, install_markup_guard
from fused_memory.server.markup_residue_spool import MarkupResidueSpool
from fused_memory.server.tools import create_mcp_server

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_REPO_ROOT = Path(__file__).resolve().parents[2]

_GUARD_LOGGER = 'fused_memory.server.markup_guard'


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _residue_loss_specimens() -> list[tuple[str, str]]:
    """``(raw_value, declared project_id)`` for every measured residue loss.

    The 2026-08-23..27 refusals whose payload the guard destroyed because the
    leak in ``add_memory.content`` ate the routing key. Non-empty is asserted so
    a moved corpus fails loudly; exact counts are not (PRD G6, see
    shared/tests/test_toolcall_markup_corpus.py::TestCommittedCorpus).
    """
    records = _read_jsonl(
        _REPO_ROOT / 'plans' / 'toolcall-markup-containment-prd.residue-loss-2026-08-27.jsonl'
    )
    specimens = [
        (record['raw_value'], record['identity_present_in_payload']['project_id'])
        for record in records
    ]
    assert specimens, 'the residue-loss corpus is empty or has moved'
    return specimens


def _unrepairable_add_memory_specimens() -> list[dict[str, Any]]:
    """Every committed corpus record ``repair()`` refuses for ``add_memory``.

    Keys are the corpus's own: ``value``, ``param``, ``schema_params``,
    ``supplied`` and the fully-qualified ``tool``.
    """
    specimens = [
        record
        for record in _read_jsonl(
            _REPO_ROOT / 'shared' / 'tests' / 'fixtures' / 'toolcall_markup_corpus.jsonl'
        )
        if record['expected_outcome'] == 'unrepairable'
        and record['tool'] == 'mcp__fused-memory__add_memory'
    ]
    assert specimens, 'the unrepairable add_memory corpus slice is empty or has moved'
    return specimens


def _content_that_ate(name: str, value: str) -> str:
    """add_memory ``content`` whose leak ate the *name* parameter and then the
    envelope's own closer, so ``repair()`` cannot attribute it: the measured
    residue-loss shape."""
    return (
        _CLEAN_CONTENT
        + closer_for('content')
        + CANONICAL_OPENER_PREFIX
        + f'"{name}">{value}'
        + closer_for('parameter')
        + closer_for('invoke')
        + '\nthe head of the next tool call'
    )


async def _refused(server, tool: str, arguments: dict[str, Any]) -> dict:
    """Call *tool* and return the unrepairable refusal it must raise."""
    with pytest.raises(ToolError) as exc_info:
        await server._tool_manager.call_tool(tool, arguments)
    payload = _payload(exc_info)
    assert payload['error_type'] == 'mcp_markup_unrepairable', payload
    return payload


def _residue_outcomes(caplog) -> list[dict[str, Any]]:
    """The structured outcome of every guard log record that reports a residue
    tier: its ``extra=`` fields plus the rendered ``message``."""
    return [
        {
            key: value for key, value in record.__dict__.items()
            if key.startswith('residue_')
        } | {'message': record.getMessage()}
        for record in caplog.records
        if record.name == _GUARD_LOGGER and 'residue_tier' in record.__dict__
    ]


def _spooled(payload: dict, spool_dir: Path) -> Any:
    """The ``raw_value`` in the floor file a refusal names, which must sit in
    *spool_dir* and must not pose as an escalation."""
    path = Path(payload['residue_spool_path'])
    assert path.parent == spool_dir, payload
    assert payload['escalation_id'] is None, f'a spool file is not an escalation: {payload!r}'
    return json.loads(path.read_text(encoding='utf-8'))['raw_value']


def _unrepairable(marker: str) -> str:
    """A specimen the repairer cannot attribute — built from the constants."""
    return f'{_CLEAN_CONTENT} {marker}' + closer_for('invoke')


async def _refuse_unrepairable(server, marker: str = 'one') -> dict:
    """The refusal of an ``add_system_record`` whose content is :func:`_unrepairable`."""
    with pytest.raises(ToolError) as exc_info:
        await server._tool_manager.call_tool(
            'add_system_record',
            {
                'content': _unrepairable(marker),
                'project_id': _PROJECT_ID,
                'category': 'observations_and_summaries',
                'agent_id': _AGENT_ID,
            },
        )
    return _payload(exc_info)


def _unwritable_spool(tmp_path) -> MarkupResidueSpool:
    """A spool whose parent is a regular FILE: root-safe, unlike chmod."""
    blocker = tmp_path / 'not-a-directory'
    blocker.write_text('occupied')
    return MarkupResidueSpool(blocker / 'spool')


class TestUnrepairableResidueIsPreserved:
    """A refusal must not destroy the payload it refuses.

    ``\x3c/invoke>`` closes the ENVELOPE rather than a parameter, so
    ``shared.toolcall_markup.repair`` can attribute the residue to nothing and
    refuses under its NO SILENT PARTIAL REPAIR contract. Per the boundary
    specimen table that is the COMMON outcome for the real corpus shapes, not an
    edge case — and the caller is told, in the shared hint, that its full payload
    is "preserved verbatim in the escalation named above". These pin that the
    escalation named there EXISTS.
    """

    @pytest.mark.asyncio
    async def test_the_refusal_names_a_record_that_actually_exists(self, tmp_path):
        """The defect: escalation_id was null while the hint pointed at it."""
        server, mock_service = _build_guarded_server(
            'add_system_record', known_projects={_PROJECT_ID: str(tmp_path)}
        )

        payload = await _refuse_unrepairable(server)

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        assert payload['escalation_id'] is not None, (
            f'the hint promises a record; it must exist: {payload!r}'
        )
        records = _escalations(tmp_path)
        assert [r['id'] for r in records] == [payload['escalation_id']]
        mock_service.add_system_record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_record_holds_the_payload_verbatim(self, tmp_path):
        """"Preserved VERBATIM" is the promise, so assert the caller's own bytes."""
        server, _ = _build_guarded_server(
            'add_system_record', known_projects={_PROJECT_ID: str(tmp_path)}
        )

        await _refuse_unrepairable(server)

        record = _escalations(tmp_path)[0]
        assert _unrepairable('one') in record['detail']
        assert record['task_id'] == 'markup-residue'
        # A residue record is NOT a burst summary: it must not land on either
        # storm anchor, whose records are deduped into one.
        assert record['task_id'] not in {'markup-guard', 'markup-tripwire'}

    @pytest.mark.asyncio
    async def test_two_refusals_are_never_folded_into_one_record(self, tmp_path):
        """The one place this filer must NOT dedupe like its storm sibling.

        Each record is the only surviving copy of a DIFFERENT payload, so
        collapsing them destroys data — the exact loss the record exists to
        prevent.
        """
        server, _ = _build_guarded_server(
            'add_system_record', known_projects={_PROJECT_ID: str(tmp_path)}
        )

        await _refuse_unrepairable(server, marker='first')
        await _refuse_unrepairable(server, marker='second')

        details = '\n'.join(r['detail'] for r in _escalations(tmp_path))
        assert len(_escalations(tmp_path)) == 2
        assert _unrepairable('first') in details
        assert _unrepairable('second') in details

    @pytest.mark.asyncio
    async def test_an_unfilable_residue_falls_back_to_the_spool(self, tmp_path, caplog):
        """No queue resolves, so the floor holds the payload and the caller is
        handed its locator. The payload is not dumped into the log as well."""
        spool = MarkupResidueSpool(tmp_path / 'spool')
        server, _ = _build_guarded_server(
            'add_system_record',
            known_projects={'some_other_project': str(tmp_path)},
            residue_floor=spool.preserve,
        )

        with caplog.at_level('ERROR', logger=_GUARD_LOGGER):
            payload = await _refuse_unrepairable(server)

        assert _spooled(payload, tmp_path / 'spool') == _unrepairable('one')
        assert not (tmp_path / 'data' / 'escalations').exists()
        [outcome] = _residue_outcomes(caplog)
        assert outcome['residue_tier'] == ResidueTier.FLOOR
        assert outcome['residue_locator'] == payload['residue_spool_path'], (
            'the operator still needs the pointer'
        )
        assert _unrepairable('one') not in outcome['message']

    @pytest.mark.asyncio
    async def test_a_filed_residue_does_not_dump_the_payload_into_the_log(
        self, tmp_path, caplog
    ):
        """The log-volume half. Once the payload is queued, repeating it on every
        refusal is maximum log volume exactly when a leak is bursting."""
        server, _ = _build_guarded_server(
            'add_system_record', known_projects={_PROJECT_ID: str(tmp_path)}
        )

        with caplog.at_level('ERROR', logger='fused_memory.server.markup_guard'):
            payload = await _refuse_unrepairable(server)

        text = '\n'.join(
            r.getMessage() for r in caplog.records
            if r.name == 'fused_memory.server.markup_guard'
        )
        assert payload['escalation_id'] in text, (
            f'the operator still needs the pointer: {text!r}'
        )
        assert _unrepairable('one') not in text

    @pytest.mark.asyncio
    async def test_a_residue_filer_that_raises_never_changes_the_refusal(
        self, tmp_path, monkeypatch
    ):
        """Escalation is purely additive: the refusal is already decided."""
        import fused_memory.server.markup_guard as guard_module

        def _boom(*args, **kwargs):
            raise RuntimeError('queue on fire')

        monkeypatch.setattr(guard_module, 'emit_markup_residue_escalation', _boom)
        spool = MarkupResidueSpool(tmp_path / 'spool')
        server, mock_service = _build_guarded_server(
            'add_system_record',
            known_projects={_PROJECT_ID: str(tmp_path)},
            residue_floor=spool.preserve,
        )

        payload = await _refuse_unrepairable(server)

        assert payload['error_type'] == 'mcp_markup_unrepairable'
        mock_service.add_system_record.assert_not_awaited()
        # The floor caught the payload the raising filer dropped.
        assert _spooled(payload, tmp_path / 'spool') == _unrepairable('one')


class TestRoutingIdentityIsRederivedFromTheResidue:
    """add_memory declares ``content`` before ``project_id``, so a leak in
    ``content`` eats the routing key. The id survives only inside the refused
    payload, and is recovered from there for QUEUE PLACEMENT only — every
    scanned token is still vouched for by the registry.

    Each case refuses one add_memory through the guarded server's ``call_tool``
    and asserts which queue the residue landed in, or that only the floor
    holds it.
    """

    @staticmethod
    def _registry(tmp_path) -> dict[str, str]:
        return {'reify': str(tmp_path / 'reify'), 'dark_factory': str(tmp_path / 'df')}

    @staticmethod
    def _echo(name: str, value: str) -> str:
        return '\x3c' + name + '>' + value + closer_for(name)

    @staticmethod
    async def _refuse(tmp_path, content: str, known_projects: dict[str, str] | None) -> dict:
        server = create_mcp_server(AsyncMock())
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects=known_projects,
            residue_floor=MarkupResidueSpool(tmp_path / 'spool').preserve,
        )
        return await _refused(server, 'add_memory', {'content': content})

    @staticmethod
    def _queues_holding_a_record(registry: dict[str, str]) -> list[str]:
        return sorted(name for name, root in registry.items() if _escalations(Path(root)))

    @classmethod
    def _assert_only_the_floor_holds_it(cls, payload: dict, tmp_path, registry) -> None:
        assert cls._queues_holding_a_record(registry) == []
        assert _spooled(payload, tmp_path / 'spool') is not None

    @pytest.mark.parametrize(
        ('raw_value', 'project_id'),
        [
            pytest.param(raw_value, project_id, id=f'{index}-{project_id}')
            for index, (raw_value, project_id) in enumerate(_residue_loss_specimens())
        ],
    )
    @pytest.mark.asyncio
    async def test_every_measured_residue_loss_files_into_its_own_project(
        self, tmp_path, raw_value, project_id
    ):
        registry = self._registry(tmp_path)

        payload = await self._refuse(tmp_path, raw_value, registry)

        assert self._queues_holding_a_record(registry) == [project_id]
        [record] = _escalations(Path(registry[project_id]))
        assert record['id'] == payload['escalation_id']

    @pytest.mark.asyncio
    async def test_an_unregistered_id_files_only_into_the_floor(self, tmp_path):
        registry = self._registry(tmp_path)

        payload = await self._refuse(
            tmp_path, _content_that_ate('project_id', 'not_a_registered_project'), registry
        )

        self._assert_only_the_floor_holds_it(payload, tmp_path, registry)

    @pytest.mark.asyncio
    async def test_an_unvouched_absolute_root_files_only_into_the_floor(self, tmp_path):
        registry = self._registry(tmp_path)
        chosen = tmp_path / 'chosen-by-the-caller'

        payload = await self._refuse(
            tmp_path, _content_that_ate('project_root', str(chosen)), registry
        )

        self._assert_only_the_floor_holds_it(payload, tmp_path, registry)
        assert not chosen.exists()

    @pytest.mark.asyncio
    async def test_a_registered_absolute_root_files_into_its_own_queue(self, tmp_path):
        registry = self._registry(tmp_path)

        await self._refuse(
            tmp_path, _content_that_ate('project_root', registry['reify']), registry
        )

        assert self._queues_holding_a_record(registry) == ['reify']

    @pytest.mark.asyncio
    async def test_two_different_registered_projects_file_only_into_the_floor(self, tmp_path):
        registry = self._registry(tmp_path)
        content = (
            _content_that_ate('project_id', 'reify')
            + self._echo('project_root', registry['dark_factory'])
        )

        payload = await self._refuse(tmp_path, content, registry)

        self._assert_only_the_floor_holds_it(payload, tmp_path, registry)

    @pytest.mark.asyncio
    async def test_one_id_with_two_values_files_only_into_the_floor(self, tmp_path):
        registry = self._registry(tmp_path)
        content = (
            _content_that_ate('project_id', 'reify')
            + self._echo('project_id', 'dark_factory')
        )

        payload = await self._refuse(tmp_path, content, registry)

        self._assert_only_the_floor_holds_it(payload, tmp_path, registry)

    @pytest.mark.asyncio
    async def test_an_id_and_a_root_naming_the_same_project_agree(self, tmp_path):
        registry = self._registry(tmp_path)
        content = (
            _content_that_ate('project_id', 'reify')
            + self._echo('project_root', registry['reify'])
        )

        await self._refuse(tmp_path, content, registry)

        assert self._queues_holding_a_record(registry) == ['reify']

    @pytest.mark.parametrize('known_projects', [None, {}], ids=['no-registry', 'empty-registry'])
    @pytest.mark.asyncio
    async def test_without_a_registry_it_files_only_into_the_floor(
        self, tmp_path, known_projects
    ):
        payload = await self._refuse(
            tmp_path, _content_that_ate('project_id', 'reify'), known_projects
        )

        assert not (tmp_path / 'reify').exists()
        assert _spooled(payload, tmp_path / 'spool') is not None


class TestPreservationIsUnconditional:
    """C2/B5/INV-7: a refused payload is preserved whether or not routing survived.

    add_memory declares ``content`` before ``project_id``, so a forward-consuming
    leak in ``content`` eats the routing key. Before task 4809 that made
    preservation depend on parameter DECLARATION ORDER: the 2026-08-23..27
    refusals of add_memory were all lost.
    """

    #: A registry that resolves none of the corpus payloads.
    _UNRELATED = 'some_other_project'

    #: Values for the arguments a specimen declares it supplied besides ``content``.
    _SUPPLIED_VALUES = {
        'project_id': 'not_a_registered_project',
        'agent_id': 'claude-task-4809-specimen',
        'category': 'observations_and_summaries',
    }

    @classmethod
    def _arguments(cls, specimen: dict[str, Any]) -> dict[str, Any]:
        arguments = {name: cls._SUPPLIED_VALUES[name] for name in specimen['supplied']
                     if name != specimen['param']}
        arguments[specimen['param']] = specimen['value']
        return arguments

    @pytest.mark.asyncio
    async def test_no_corpus_payload_of_add_memory_is_lost(self, tmp_path):
        """Every specimen the LIVE schema still refuses reaches the floor verbatim.

        A specimen recorded against an older add_memory schema can be repaired
        by the live one (measured: one predates ``session_id``), and its caller
        then holds the payload in ``repaired_call``. At least one specimen must
        still be refused, or this would pass without ever reaching the floor.
        """
        reached_the_floor = []
        for index, specimen in enumerate(_unrepairable_add_memory_specimens()):
            spool_dir = tmp_path / f'spool-{index}'
            server, mock_service = _build_guarded_server(
                known_projects={self._UNRELATED: str(tmp_path / 'other')},
                residue_floor=MarkupResidueSpool(spool_dir).preserve,
            )

            with pytest.raises(ToolError) as exc_info:
                await server._tool_manager.call_tool('add_memory', self._arguments(specimen))

            payload = _payload(exc_info)
            mock_service.add_memory.assert_not_awaited()
            if payload['error_type'] == 'mcp_markup_detected':
                assert payload['repaired_call']['content'] in specimen['value']
                continue
            assert payload['error_type'] == 'mcp_markup_unrepairable', payload
            assert _spooled(payload, spool_dir) == specimen['value'], specimen['tool_use_id']
            reached_the_floor.append(specimen['tool_use_id'])

        assert reached_the_floor, (
            'the live add_memory schema repairs every corpus specimen, so none '
            'reached the floor and this test no longer tests it'
        )

    @pytest.mark.asyncio
    async def test_a_registered_project_named_only_in_the_residue_files_into_its_queue(
        self, tmp_path
    ):
        spool_dir = tmp_path / 'spool'
        server, mock_service = _build_guarded_server(
            known_projects={_PROJECT_ID: str(tmp_path)},
            residue_floor=MarkupResidueSpool(spool_dir).preserve,
        )

        payload = await _refused(
            server, 'add_memory', {'content': _content_that_ate('project_id', _PROJECT_ID)}
        )

        records = _escalations(tmp_path)
        assert [record['id'] for record in records] == [payload['escalation_id']]
        assert 'UNVERIFIED' in records[0]['suggested_action']
        assert 'project=None' in records[0]['detail']
        assert not spool_dir.exists(), 'the queue stays primary; the floor is not touched'
        mock_service.add_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_surviving_project_root_files_exactly_as_before(self, tmp_path, monkeypatch):
        """The positive control: submit_task declares ``project_root`` first, so
        its identity survives and the residue takes the unchanged route."""
        monkeypatch.setattr(
            'fused_memory.server.tools.resolve_main_checkout', lambda p: str(p)
        )
        interceptor = AsyncMock()
        server = create_mcp_server(AsyncMock(), task_interceptor=interceptor)
        spool_dir = tmp_path / 'spool'
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects={_PROJECT_ID: str(tmp_path)},
            residue_floor=MarkupResidueSpool(spool_dir).preserve,
        )

        payload = await _refused(
            server,
            'submit_task',
            {
                'project_root': str(tmp_path),
                'prompt': 'Reconcile the queue',
                'description': 'Reconcile the queue' + closer_for('invoke'),
            },
        )

        records = _escalations(tmp_path)
        assert [record['id'] for record in records] == [payload['escalation_id']]
        assert 'UNVERIFIED' not in records[0]['suggested_action']
        assert not spool_dir.exists()
        interceptor.submit_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_when_every_tier_fails_the_caller_is_told_nothing_was_preserved(
        self, tmp_path
    ):
        server, mock_service = _build_guarded_server(
            known_projects={self._UNRELATED: str(tmp_path / 'other')},
            residue_floor=_unwritable_spool(tmp_path).preserve,
        )

        payload = await _refused(
            server,
            'add_memory',
            {'content': _CLEAN_CONTENT + closer_for('invoke'), 'project_id': 'unregistered'},
        )

        assert payload['escalation_id'] is None
        assert 'residue_spool_path' not in payload
        assert 'NOTHING WAS PRESERVED' in payload['hint']
        mock_service.add_memory.assert_not_awaited()


class TestPreservationFailuresAreReadable:
    """INV-4: a payload no tier could hold is counted where an operator looks.

    ``get_status.markup_residue`` reports the floor's own loss count. The floor
    is the last tier, so a failed floor write IS a lost payload; a fallback to
    the floor is not a loss and is not counted.
    """

    @staticmethod
    def _server(
        spool: MarkupResidueSpool | None, known_projects: dict[str, str] | None = None
    ) -> tuple[Any, AsyncMock]:
        mock_service = AsyncMock()
        mock_service.get_status = AsyncMock(side_effect=lambda **_: {'mem0': {'ok': True}})
        if spool is None:
            return create_mcp_server(mock_service), mock_service
        server = create_mcp_server(mock_service, markup_residue_spool=spool)
        install_markup_guard(
            server,
            policy=RepairPolicy.REJECT_WITH_REPAIR,
            known_projects=known_projects or {'some_other_project': '/elsewhere'},
            residue_floor=spool.preserve,
        )
        return server, mock_service

    @staticmethod
    async def _status(server) -> dict[str, Any]:
        return await server._tool_manager.call_tool('get_status', {})

    @staticmethod
    async def _refuse_add_memory(server) -> dict:
        with pytest.raises(ToolError) as exc_info:
            await server._tool_manager.call_tool(
                'add_memory',
                {'content': _CLEAN_CONTENT + closer_for('invoke'), 'project_id': _PROJECT_ID},
            )
        payload = _payload(exc_info)
        assert payload['error_type'] == 'mcp_markup_unrepairable', payload
        return payload

    @pytest.mark.asyncio
    async def test_a_fresh_server_asserts_nothing_was_lost(self, tmp_path):
        spool = MarkupResidueSpool(tmp_path / 'spool')
        server, _ = self._server(spool)

        status = await self._status(server)

        assert status['markup_residue']['preservation_failures'] == 0
        assert status['markup_residue']['spool_dir'] == str(tmp_path / 'spool')

    @pytest.mark.asyncio
    async def test_a_payload_no_tier_held_is_counted(self, tmp_path):
        server, mock_service = self._server(_unwritable_spool(tmp_path))

        payload = await self._refuse_add_memory(server)
        status = await self._status(server)

        assert payload['escalation_id'] is None
        assert 'residue_spool_path' not in payload
        residue = status['markup_residue']
        assert residue['preservation_failures'] == 1
        assert residue['last_failure']['tool'] == 'add_memory'
        assert residue['last_failure']['field'] == 'content'
        mock_service.add_memory.assert_not_awaited()

    @pytest.mark.parametrize('tier', ['queue', 'floor'])
    @pytest.mark.asyncio
    async def test_a_preserved_payload_is_not_counted(self, tmp_path, tier):
        """The counter measures losses, not fallbacks."""
        spool = MarkupResidueSpool(tmp_path / 'spool')
        known_projects = (
            {_PROJECT_ID: str(tmp_path)} if tier == 'queue' else {'some_other_project': '/x'}
        )
        server, _ = self._server(spool, known_projects)

        payload = await self._refuse_add_memory(server)
        status = await self._status(server)

        assert (payload['escalation_id'] is not None) is (tier == 'queue')
        assert ('residue_spool_path' in payload) is (tier == 'floor')
        assert status['markup_residue']['preservation_failures'] == 0

    @pytest.mark.asyncio
    async def test_a_spooled_payload_is_reported_as_waiting(self, tmp_path):
        """An operator sees a growing floor, not only its losses."""
        server, _ = self._server(MarkupResidueSpool(tmp_path / 'spool'))

        payload = await self._refuse_add_memory(server)
        status = await self._status(server)

        residue = status['markup_residue']
        assert residue['spooled_files'] == 1
        assert residue['last_preserved'] == payload['residue_spool_path']

    @pytest.mark.asyncio
    async def test_the_spool_is_listed_off_the_event_loop(self, tmp_path, monkeypatch):
        """get_status is polled, so listing a large spool must not block the loop."""
        spool = MarkupResidueSpool(tmp_path / 'spool')
        server, _ = self._server(spool)
        listing_threads: list[int] = []
        real_status = spool.status

        def _recording_status() -> dict[str, Any]:
            listing_threads.append(threading.get_ident())
            return real_status()

        monkeypatch.setattr(spool, 'status', _recording_status)

        status = await self._status(server)

        assert status['markup_residue']['spooled_files'] == 0
        assert listing_threads
        assert threading.get_ident() not in listing_threads

    @pytest.mark.asyncio
    async def test_an_unwired_server_omits_the_field(self):
        """Unwired is not zero: a zero would assert something nobody measured."""
        server, _ = self._server(None)

        status = await self._status(server)

        assert 'markup_residue' not in status

    @pytest.mark.asyncio
    async def test_the_field_is_top_level_not_under_queue(self, tmp_path):
        server, _ = self._server(MarkupResidueSpool(tmp_path / 'spool'))

        status = await self._status(server)

        assert 'markup_residue' in status
        assert 'markup_residue' not in status['queue']


class TestTheLastTierLogLineIsHonest:
    """When neither the queue nor the floor held a payload, the guard's ERROR
    line is where it survives. Its outcome is logged as data: the tier, and
    whether a floor was wired at all.
    """

    async def _last_tier_outcome(self, caplog, residue_floor: Any = None) -> dict[str, Any]:
        server, _ = _build_guarded_server(
            'add_system_record',
            known_projects={'some_other_project': '/elsewhere'},
            residue_floor=residue_floor,
        )
        with caplog.at_level('ERROR', logger=_GUARD_LOGGER):
            payload = await _refuse_unrepairable(server)
        assert payload['escalation_id'] is None
        assert 'residue_spool_path' not in payload
        [outcome] = _residue_outcomes(caplog)
        assert outcome['residue_tier'] == ResidueTier.UNPRESERVED
        return outcome

    @pytest.mark.asyncio
    async def test_an_absent_floor_is_logged_as_unwired(self, caplog):
        outcome = await self._last_tier_outcome(caplog)

        assert outcome['residue_floor_wired'] is False

    @pytest.mark.asyncio
    async def test_a_failed_floor_is_logged_as_wired(self, caplog, tmp_path):
        outcome = await self._last_tier_outcome(caplog, _unwritable_spool(tmp_path).preserve)

        assert outcome['residue_floor_wired'] is True

    @pytest.mark.asyncio
    async def test_it_still_carries_the_payload(self, caplog):
        outcome = await self._last_tier_outcome(caplog)

        assert _unrepairable('one') in outcome['message']
        assert MARKUP_RECURRENCE_ROUTING in outcome['message']
