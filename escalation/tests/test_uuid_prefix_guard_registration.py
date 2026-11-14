"""UuidPrefixGuardMiddleware registration on the REAL escalation server (task 5324).

PRD ``plans/uuid-prefix-resolution-prd.md`` §4-C3, leaf δ. The guard itself
is proven against toy servers in ``shared``; this file proves it is ATTACHED
to ``create_server``'s server, scoped and classed the way C3 rules for this
server, and that a truncated citation in a real ``escalate_info`` lands
expanded, forwarded or untouched as the resolver's answer dictates.

Every call goes through ``async with Client(server)``. Middleware is BYPASSED
by ``tool.fn(...)`` and ``await tool.run({...})``, so a test written that way
would pass while running none of the guard.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastmcp import Client
from shared.mcp_markup_middleware import MarkupGuardMiddleware
from shared.toolcall_markup import INVOKE_CLOSER, closer_for
from shared.uuid_prefix_guard import (
    FACT_UUID_PREFIX_DETECTED,
    Candidate,
    PrefixResolver,
    Resolution,
    ResolverUnavailable,
    UuidPrefixGuardMiddleware,
)

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation
from escalation.prefix_resolver_client import PrefixResolverClient
from escalation.queue import EscalationQueue
from escalation.server import create_server
from escalation.store_identity import StoreIdentity

PREFIX = 'bff81530'
FULL = 'bff81530-70e5-4ab6-939e-507299d1bcd2'
SIBLING = 'bff81530-0000-4000-8000-000000000002'

#: ``escalate_info``'s four required parameters, as in the markup sibling file.
REQUIRED = {
    'task_id': '3441',
    'agent_role': 'implementer',
    'category': 'cleanup_needed',
    'summary': 'A dead repair is cited by its short id.',
}

CITING_DETAIL = f'see memory {PREFIX} for the dead repair'

UNIQUE = Resolution('unique', (Candidate('mem0', FULL, 'the dead repair'),))
AMBIGUOUS = Resolution('ambiguous', (
    Candidate('mem0', FULL, 'a memory'),
    Candidate('graphiti_node', SIBLING, 'a node'),
))
NONE = Resolution('none', ())

#: UNIQUE as fused-memory's POST /resolve-prefix spells it on the wire.
UNIQUE_BODY = {'outcome': 'unique', 'candidates': [{'namespace': 'mem0', 'id': FULL, 'preview': 'the dead repair'}]}


def http_resolver(*, answers_after: float, **client_kwargs: float) -> PrefixResolver:
    """A REAL PrefixResolverClient whose fused-memory answers UNIQUE after *answers_after* seconds."""

    async def answer(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(answers_after)
        return httpx.Response(200, json=UNIQUE_BODY)

    return PrefixResolverClient(
        base_url='http://fused-memory.test', transport=httpx.MockTransport(answer), **client_kwargs,
    ).resolve_prefix


def project_identity(tmp_path: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='project', queue_dir=tmp_path / 'esc', project_id='reify', project_root=tmp_path,
    )


def reconciliation_identity(tmp_path: Path) -> StoreIdentity:
    return StoreIdentity(
        kind='reconciliation', queue_dir=tmp_path / 'esc', project_id=None, project_root=None,
    )


class RecordingResolver:
    """A ``PrefixResolver`` answering one scripted Resolution, or raising one outage."""

    def __init__(
        self,
        answer: Resolution = NONE,
        *,
        outage: ResolverUnavailable | None = None,
    ) -> None:
        self.answer = answer
        self.outage = outage
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, project_id: str, prefix: str) -> Resolution:
        self.calls.append((project_id, prefix))
        if self.outage is not None:
            raise self.outage
        return self.answer


def build(tmp_path: Path, resolver: Any, identity: StoreIdentity | None = None):
    """The real server, with the sweep off and the prefix guard given *resolver*."""
    identity = identity if identity is not None else project_identity(tmp_path)
    queue = EscalationQueue(tmp_path / 'esc')
    server = create_server(
        queue, startup_sweep=False, store_identity=identity, prefix_resolver=resolver,
    )
    return queue, server


def guards_of(server, kind: type) -> list[Any]:
    return [m for m in getattr(server, 'middleware', []) or [] if isinstance(m, kind)]


def guard_of(server) -> UuidPrefixGuardMiddleware:
    """The registered prefix guard — asserting there is exactly one."""
    guards = guards_of(server, UuidPrefixGuardMiddleware)
    assert len(guards) == 1, (
        f'expected exactly one UuidPrefixGuardMiddleware on the escalation server, '
        f'found {len(guards)}'
    )
    return guards[0]


def prefix_facts(caplog: pytest.LogCaptureFixture) -> list[tuple[int, dict[str, Any]]]:
    """Every ``uuid_prefix_detected`` record the server logged, parsed whole."""
    marker = f'{FACT_UUID_PREFIX_DETECTED} '
    return [
        (record.levelno, json.loads(record.getMessage()[len(marker):]))
        for record in caplog.records
        if record.name == 'escalation.server' and record.getMessage().startswith(marker)
    ]


def meta_of(result) -> dict[str, Any]:
    assert result.meta is not None, 'the call returned no meta at all'
    return result.meta


async def call(server, tool: str, arguments: dict[str, Any]):
    async with Client(server) as client:
        return await client.call_tool(tool, arguments)


def _leak(absorbed: str, param: str, value: str) -> str:
    """A ``detail`` that mis-closed and swallowed exactly one parameter.

    Built from ``shared.toolcall_markup``'s own enumeration, never typed out:
    writing an envelope literal into a source file makes the authoring agent
    emit it inside its own tool call. Mirrors the markup sibling file's helper.
    """
    return (
        absorbed
        + closer_for('detail')
        + '\n'
        + '\x3c' + param + '>'
        + value
        + closer_for(param)
        + INVOKE_CLOSER
    )


class TestRegistration:
    def test_exactly_one_guard_is_registered_after_the_markup_guard(self, tmp_path: Path):
        _, server = build(tmp_path, RecordingResolver())

        prefix_guard = guard_of(server)
        markup_guards = guards_of(server, MarkupGuardMiddleware)
        assert len(markup_guards) == 1
        assert server.middleware.index(prefix_guard) > server.middleware.index(markup_guards[0])

    def test_a_server_built_without_a_prefix_resolver_carries_none(self, tmp_path: Path):
        server = create_server(EscalationQueue(tmp_path / 'esc'), startup_sweep=False)

        assert guards_of(server, UuidPrefixGuardMiddleware) == []

    @pytest.mark.asyncio
    async def test_nothing_is_exempt_and_every_live_tool_forwards_on_ambiguity(self, tmp_path: Path):
        _, server = build(tmp_path, RecordingResolver())
        guard = guard_of(server)

        async with Client(server) as client:
            live = {tool.name for tool in await client.list_tools()}

        assert guard.exempt_tools == frozenset()
        assert guard.forward_on_ambiguity_tools == live


class TestUniqueIsExpanded:
    """The PRD δ signal: a cited short id is filed as the full id."""

    @pytest.mark.asyncio
    async def test_the_filed_detail_carries_the_full_id(self, tmp_path: Path, caplog):
        caplog.set_level(logging.INFO, logger='escalation.server')
        resolver = RecordingResolver(UNIQUE)
        queue, server = build(tmp_path, resolver)

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': CITING_DETAIL})

        assert resolver.calls == [('reify', PREFIX)]
        assert result.data['status'] == 'queued'
        esc = queue.get(result.data['id'])
        assert esc is not None
        assert FULL in esc.detail
        assert f'{PREFIX} ' not in esc.detail
        assert meta_of(result)['uuid_prefix_repair']['substitutions'] == [{
            'field': 'detail', 'path': ['detail'], 'from': PREFIX, 'to': FULL, 'namespace': 'mem0',
        }]
        facts = prefix_facts(caplog)
        assert [(level, fact['outcome'], fact['project']) for level, fact in facts] == [
            (logging.INFO, 'expanded', 'reify'),
        ]


class TestSlowResolverStillResolves:
    """Measured /resolve-prefix latency reaches ~2.4 s under the guard's 8-wide
    fan-out; a fused-memory answering in 1.5 s must still expand the citation."""

    @pytest.mark.asyncio
    async def test_a_resolver_answering_after_1_5s_files_the_full_id(self, tmp_path: Path, caplog):
        caplog.set_level(logging.INFO, logger='escalation.server')
        queue, server = build(tmp_path, http_resolver(answers_after=1.5))

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': CITING_DETAIL})

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert FULL in esc.detail
        repair = meta_of(result)['uuid_prefix_repair']
        assert 'substitutions' in repair
        assert 'resolver' not in repair
        assert [fact['outcome'] for _, fact in prefix_facts(caplog)] == ['expanded']


class TestAmbiguousIsForwarded:
    """Every tool on this server forwards on ambiguity: a lost escalation is worse."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('tool', ['escalate_info', 'escalate_blocker'])
    async def test_the_escalation_is_filed_unchanged_naming_both_candidates(self, tmp_path: Path, tool):
        queue, server = build(tmp_path, RecordingResolver(AMBIGUOUS))

        result = await call(server, tool, {**REQUIRED, 'detail': CITING_DETAIL})

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.detail == CITING_DETAIL
        ambiguous = meta_of(result)['uuid_prefix_repair']['ambiguous']
        assert [candidate['id'] for candidate in ambiguous[0]['candidates']] == [FULL, SIBLING]

    @pytest.mark.asyncio
    async def test_a_non_filing_tool_forwards_too(self, tmp_path: Path):
        queue, server = build(tmp_path, RecordingResolver(AMBIGUOUS))
        filed = await call(server, 'escalate_info', REQUIRED)
        amendment = f'the ruling cites {PREFIX}'

        result = await call(
            server, 'amend_escalation', {'escalation_id': filed.data['id'], 'detail': amendment},
        )

        assert result.data['amendment_recorded'] is True
        esc = queue.get(filed.data['id'])
        assert esc is not None
        assert esc.amendments[-1]['detail'] == amendment
        assert meta_of(result)['uuid_prefix_repair']['ambiguous'][0]['token'] == PREFIX


class TestFailOpen:
    """The half that must not regress: fused-memory stopped never loses an escalation."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('base_url', [
        'http://127.0.0.1:1',
        'http://localhost:abc',
        'http://127.0.0.1:80020',
    ], ids=['refused', 'invalid-port', 'port-out-of-range'])
    async def test_an_unreachable_resolver_files_unchanged_with_a_warning_fact(
        self, tmp_path: Path, caplog, base_url,
    ):
        caplog.set_level(logging.INFO, logger='escalation.server')
        resolver = PrefixResolverClient(base_url=base_url).resolve_prefix
        queue, server = build(tmp_path, resolver)

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': CITING_DETAIL})

        assert result.data['status'] == 'queued'
        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.detail == CITING_DETAIL
        assert meta_of(result)['uuid_prefix_repair'] == {
            'resolver': 'unavailable', 'store': 'fused-memory', 'reason': 'unreachable',
        }
        assert [
            (level, fact['outcome'], fact['project'], fact['token'])
            for level, fact in prefix_facts(caplog)
        ] == [(logging.WARNING, 'resolver_unavailable', 'reify', PREFIX)]

    @pytest.mark.asyncio
    async def test_a_resolver_past_its_deadline_files_unchanged_naming_the_timeout(self, tmp_path: Path):
        queue, server = build(tmp_path, http_resolver(answers_after=30, timeout_seconds=0.05))

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': CITING_DETAIL})

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.detail == CITING_DETAIL
        assert meta_of(result)['uuid_prefix_repair'] == {
            'resolver': 'unavailable', 'store': 'fused-memory', 'reason': 'timed_out',
        }


class TestNoneIsInert:
    @pytest.mark.asyncio
    async def test_an_unmatched_prefix_leaves_no_trace(self, tmp_path: Path, caplog):
        caplog.set_level(logging.INFO, logger='escalation.server')
        queue, server = build(tmp_path, RecordingResolver(NONE))

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': CITING_DETAIL})

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.detail == CITING_DETAIL
        assert 'uuid_prefix_repair' not in (result.meta or {})
        assert prefix_facts(caplog) == []


class TestReconciliationStoreIsUnscoped:
    """The 8103 shape: a multi-project store never guesses a project."""

    @pytest.mark.asyncio
    async def test_a_prefixed_call_is_filed_unchanged_without_resolving(self, tmp_path: Path):
        resolver = RecordingResolver(UNIQUE)
        queue, server = build(tmp_path, resolver, reconciliation_identity(tmp_path))

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': CITING_DETAIL})

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.detail == CITING_DETAIL
        assert meta_of(result)['uuid_prefix_repair'] == {'resolver': 'no_project'}
        assert resolver.calls == []


class TestRecoveredParameterIsScanned:
    """B15: the markup guard runs first, so a parameter it recovers is resolved too."""

    @pytest.mark.asyncio
    async def test_a_swallowed_citation_is_recovered_and_expanded(self, tmp_path: Path):
        queue, server = build(tmp_path, RecordingResolver(UNIQUE))
        detail = _leak('The repair died.', 'suggested_action', f'see {PREFIX}')

        result = await call(server, 'escalate_info', {**REQUIRED, 'detail': detail})

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.suggested_action == f'see {FULL}'
        assert 'suggested_action' in meta_of(result)['markup_repair']['recovered_params']
        assert meta_of(result)['uuid_prefix_repair']['substitutions'][0]['field'] == 'suggested_action'


class TestOverride:
    @pytest.mark.asyncio
    async def test_the_author_override_files_the_bare_token(self, tmp_path: Path):
        resolver = RecordingResolver(UNIQUE)
        queue, server = build(tmp_path, resolver)

        result = await call(server, 'escalate_info', {
            **REQUIRED, 'detail': CITING_DETAIL, 'metadata': {'allow_uuid_prefix': True},
        })

        esc = queue.get(result.data['id'])
        assert esc is not None
        assert esc.detail == CITING_DETAIL
        assert resolver.calls == []
        assert 'uuid_prefix_repair' not in (result.meta or {})


# ---------------------------------------------------------------------------
# INV-4: the burst alarm, on the real server.
# ---------------------------------------------------------------------------

#: The alarm's synthetic anchor, category and filer. Spelled out rather than
#: imported: these tests assert the registration site put this vocabulary on a
#: real record, and importing both sides of a contract lets both drift together.
STORM_ANCHOR = 'uuid-prefix-storm'
STORM_CATEGORY = 'uuid_prefix_storm'
STORM_ROLE = 'harness-uuid-prefix-guard'

#: ``UuidPrefixGuardMiddleware``'s default ``storm_threshold``; the server does
#: not override it.
STORM_THRESHOLD = 3


class TestStormBurstAlarm:
    """A burst of fail-soft outcomes files ONE open alarm, never a refusal."""

    @staticmethod
    async def _burst(server, calls: int = STORM_THRESHOLD) -> list[Any]:
        """Drive *calls* prefixed escalate_info calls; return each result."""
        results = []
        async with Client(server) as client:
            for index in range(calls):
                results.append(await client.call_tool('escalate_info', {
                    **REQUIRED,
                    'summary': f'{REQUIRED["summary"]} (call {index})',
                    'detail': CITING_DETAIL,
                }))
        return results

    @staticmethod
    def _alarms(queue) -> list[Any]:
        """Every pending burst alarm on the queue, in filing order."""
        return sorted(
            (esc for esc in queue.get_pending() if esc.category == STORM_CATEGORY),
            key=lambda esc: esc.id,
        )

    @staticmethod
    def _outage_server(tmp_path: Path):
        return build(tmp_path, RecordingResolver(outage=ResolverUnavailable('mem0')))

    @pytest.mark.asyncio
    async def test_a_burst_files_every_call_and_exactly_one_alarm(self, tmp_path: Path):
        queue, server = self._outage_server(tmp_path)

        results = await self._burst(server)

        assert [result.data['status'] for result in results] == ['queued'] * STORM_THRESHOLD
        storms = [meta_of(result)['uuid_prefix_repair'].get('storm') for result in results]
        assert [storm is not None for storm in storms] == [False, False, True]
        assert storms[-1]['count'] == STORM_THRESHOLD
        assert storms[-1]['outcome'] == 'resolver_unavailable'
        assert storms[-1]['reason'] == 'unreachable'

        alarms = self._alarms(queue)
        assert len(alarms) == 1, f'expected one burst alarm, found {len(alarms)}'
        alarm = alarms[0]
        assert alarm.task_id == STORM_ANCHOR
        assert alarm.agent_role == STORM_ROLE
        assert alarm.level == 1
        assert alarm.severity == 'blocking'
        assert alarm.severity not in BORN_AT_L2_SEVERITIES
        assert str(STORM_THRESHOLD) in alarm.summary
        assert 'resolver_unavailable' in alarm.summary

    @pytest.mark.asyncio
    async def test_a_timed_out_burst_files_one_alarm_naming_the_timeout(self, tmp_path: Path):
        queue, server = build(tmp_path, http_resolver(answers_after=30, timeout_seconds=0.05))

        results = await self._burst(server)

        storms = [meta_of(result)['uuid_prefix_repair'].get('storm') for result in results]
        assert storms[-1]['reason'] == 'timed_out'
        alarms = self._alarms(queue)
        assert len(alarms) == 1, f'expected one burst alarm, found {len(alarms)}'
        assert 'timed_out' in alarms[0].summary

    @pytest.mark.asyncio
    async def test_a_second_burst_folds_into_the_open_alarm(self, tmp_path: Path):
        """Driven through a SECOND server on the same queue: a restart is how
        a second fire reaches the sink, since each guard counts per instance."""
        queue, server = self._outage_server(tmp_path)
        await self._burst(server)
        first = self._alarms(queue)
        assert len(first) == 1

        _, restarted = self._outage_server(tmp_path)
        await self._burst(restarted)

        alarms = self._alarms(queue)
        assert len(alarms) == 1, f'a second burst must fold into the open alarm, found {len(alarms)}'
        assert alarms[0].id == first[0].id
        assert alarms[0].amendments == []

    @pytest.mark.asyncio
    async def test_a_burst_of_another_reason_is_noted_on_the_open_alarm(self, tmp_path: Path):
        """The alarm an operator is reading says "timed out"; when fused-memory
        then goes down, the unreachable burst folds in AS A NOTE, not silently."""
        queue, slow = build(tmp_path, http_resolver(answers_after=30, timeout_seconds=0.05))
        await self._burst(slow)
        _, down = self._outage_server(tmp_path)

        await self._burst(down)

        alarms = self._alarms(queue)
        assert len(alarms) == 1, f'expected the burst folded into one alarm, found {len(alarms)}'
        alarm = alarms[0]
        assert 'resolver_unavailable (timed_out)' in alarm.summary
        assert "reason='timed_out'" in alarm.detail.splitlines()
        notes = [note.get('summary', '') for note in alarm.amendments]
        assert len(notes) == 1, f'expected one note naming the new reason, found {notes}'
        assert 'resolver_unavailable (unreachable)' in notes[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('category', 'agent_role'), [
        ('design_concern', 'escalation-watcher'),
        (STORM_CATEGORY, 'harness-markup-guard'),
    ], ids=['another-category', 'another-filer'])
    async def test_a_record_that_is_not_this_alarm_does_not_suppress_it(
        self, tmp_path: Path, category: str, agent_role: str,
    ):
        """Dedup means "MY alarm is open": category AND filer, never the anchor alone."""
        queue, server = self._outage_server(tmp_path)
        squatter = queue.submit(Escalation(
            id=queue.make_id(STORM_ANCHOR),
            task_id=STORM_ANCHOR,
            agent_role=agent_role,
            severity='blocking',
            category=category,
            summary='Another producer filed on the same synthetic anchor.',
            level=1,
        ))

        await self._burst(server)

        mine = [esc for esc in self._alarms(queue) if esc.id != squatter]
        assert len(mine) == 1, f'the squatting record suppressed the alarm, found {len(mine)}'
        assert mine[0].agent_role == STORM_ROLE

    @pytest.mark.asyncio
    async def test_an_ambiguous_burst_files_one_alarm_naming_its_outcome(self, tmp_path: Path):
        queue, server = build(tmp_path, RecordingResolver(AMBIGUOUS))

        await self._burst(server)

        alarms = self._alarms(queue)
        assert len(alarms) == 1, f'expected one burst alarm, found {len(alarms)}'
        assert 'forwarded_ambiguous' in alarms[0].summary
