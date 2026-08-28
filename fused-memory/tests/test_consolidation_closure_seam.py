"""The close-time refusal at the ``set_task_status`` chokepoint (task 3112).

Defect 2's user-observable half. ``consolidate_memories`` reports closure at
*op* time, but nothing stood between a curator's closure CLAIM and the gate
task actually going ``done``. ``TaskInterceptor._apply_status_transition`` is
the sole seam — ``SqliteTaskBackend`` raises ``StatusWriteAuthorityError`` if
status is written any other way — so the refusal lives there.

Fixture conventions follow ``test_task_interceptor.py``: an ``AsyncMock``
taskmaster whose ``get_task`` returns the ``before`` snapshot, a real
``EventBuffer`` on ``tmp_path``, and ``TaskInterceptor(taskmaster, reconciler,
event_buffer)``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.models.scope import resolve_project_id
from fused_memory.reconciliation.consolidation_gate import GATE_METADATA_KEY
from fused_memory.reconciliation.event_buffer import EventBuffer

_TOPIC = 'seam-demo-topic'
_PROJECT_ROOT = '/tmp/seam-demo'


def _uuid(n):
    return f'00000000-0000-4000-8000-{n:012d}'


def _gate_metadata(**overrides):
    meta = {
        'execution_class': 'operational',
        'operational_mode': 'gate',
        'task_kind': 'deterministic',
        'always_escalates': True,
        GATE_METADATA_KEY: {'topic': _TOPIC},
    }
    meta.update(overrides)
    return meta


def _member(mid, *, canonical=None):
    meta = {'topic': _TOPIC}
    if canonical is not None:
        meta['canonical'] = canonical
    return {'id': mid, 'created_at': '2026-08-24T00:00:00+00:00', 'metadata': meta}


_WELL_FORMED = [_member(_uuid(1), canonical=True), _member(_uuid(2))]
_MALFORMED = [_member(_uuid(1)), _member(_uuid(2))]  # no canonical


def _scroll(members, *, total=None, raises=None, live_ids=None, probe_raises=None):
    """A stand-in for the injected memory scroll.

    *live_ids* (task 4808) arms the third collaborator: an ``exists`` probe
    reporting those ids LIVE and everything else ABSENT.  It is hung off the
    same object as ``count`` so the interceptor's ``getattr(scroll, 'exists',
    None)`` fallback picks it up from ONE bound collaborator, mirroring the
    ``scroll.count`` idiom this file already uses.  Passing no *live_ids* and
    no *probe_raises* leaves ``exists`` unset, which is the DORMANT shape.
    """
    calls = []

    async def scroll(filters, *, limit, project_id):
        calls.append({'filters': filters, 'limit': limit, 'project_id': project_id})
        if raises is not None:
            raise raises
        return list(members)

    async def count(filters, *, project_id):
        calls.append({'count': filters, 'project_id': project_id})
        if raises is not None:
            raise raises
        return len(members) if total is None else total

    scroll.calls = calls
    scroll.count = count

    if live_ids is not None or probe_raises is not None:
        probes = []
        known = {str(i).lower() for i in (live_ids or ())}

        async def exists(memory_id, *, project_id):
            probes.append((memory_id, project_id))
            if probe_raises is not None:
                raise probe_raises
            return str(memory_id).lower() in known

        exists.calls = probes
        scroll.exists = exists
        scroll.probes = probes
    return scroll


def _prov_gate(observed, **overrides):
    """Gate metadata whose inert provenance enumerates *observed*."""
    block = {
        'topic': _TOPIC,
        'provenance': {
            'report_run': 'run-abc',
            'observed_members': list(observed),
            'detector': 'topic-cluster-scan',
            'authoritative': False,
        },
    }
    block.update(overrides)
    return _gate_metadata(**{GATE_METADATA_KEY: block})


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(
        return_value={
            'id': '9001',
            'status': 'pending',
            'title': 'Consolidation gate',
            'metadata': _gate_metadata(),
        }
    )
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.set_status_and_stamp_audit = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': []})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'seam_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


async def _set_done(interceptor, **kwargs):
    return await interceptor.set_task_status(
        '9001', 'done', project_root=_PROJECT_ROOT, **kwargs
    )


class TestSeamRefusal:
    @pytest.mark.asyncio
    async def test_refuses_to_close_over_a_malformed_cluster(
        self, interceptor, taskmaster
    ):
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'
        assert result['topic'] == _TOPIC
        assert [r['code'] for r in result['reasons']] == ['no_canonical']

    @pytest.mark.asyncio
    async def test_a_refusal_must_carry_an_error_key(self, interceptor):
        """The CSV branch computes all_ok from ``result.get('error') is None``,
        so a refusal lacking 'error' would be REPORTED AS SUCCESS."""
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result.get('error') is not None

    @pytest.mark.asyncio
    async def test_a_refusal_mutates_nothing(
        self, interceptor, taskmaster, reconciler, event_buffer
    ):
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        await _set_done(interceptor)
        taskmaster.set_task_status.assert_not_called()
        taskmaster.set_status_and_stamp_audit.assert_not_called()
        reconciler.reconcile_task.assert_not_called()
        stats = await event_buffer.get_buffer_stats(
            resolve_project_id(_PROJECT_ROOT)
        )
        assert stats['size'] == 0

    @pytest.mark.asyncio
    async def test_a_well_formed_cluster_proceeds(self, interceptor, taskmaster):
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED))
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1


class TestSeamFailsClosed:
    @pytest.mark.asyncio
    async def test_a_scroll_that_raises_refuses_rather_than_passing(
        self, interceptor, taskmaster
    ):
        """``get_memories_by_metadata`` propagates a read TimeoutError rather
        than returning [], so this is reachable. A gate whose job is refuting a
        false closure claim must not pass when it cannot see (INV-3)."""
        interceptor.set_consolidation_scroll(
            _scroll(_WELL_FORMED, raises=TimeoutError('qdrant read timeout'))
        )
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'
        taskmaster.set_task_status.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unexpected_exception_also_refuses(self, interceptor):
        interceptor.set_consolidation_scroll(
            _scroll(_WELL_FORMED, raises=RuntimeError('boom'))
        )
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'


class TestSeamDormancy:
    """Nothing on the current corpus can regress: dormancy is STRUCTURAL."""

    @pytest.mark.asyncio
    async def test_dormant_when_no_scroll_is_wired(self, interceptor, taskmaster):
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1

    @pytest.mark.asyncio
    async def test_dormant_without_the_gate_key(self, interceptor, taskmaster):
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': {'execution_class': 'operational', 'operational_mode': 'gate'},
        }
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert scroll.calls == []

    @pytest.mark.asyncio
    async def test_dormant_when_not_a_gate(self, interceptor, taskmaster):
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _gate_metadata(operational_mode='llm'),
        }
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert scroll.calls == []

    @pytest.mark.asyncio
    async def test_dormant_for_a_non_done_transition(self, interceptor, taskmaster):
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await interceptor.set_task_status(
            '9001', 'in-progress', project_root=_PROJECT_ROOT
        )
        assert result.get('error') is None
        assert scroll.calls == []


class TestSeamShapeAndPrecedence:
    @pytest.mark.asyncio
    async def test_metadata_arriving_as_a_json_string_is_still_gated(
        self, interceptor, taskmaster
    ):
        """``before['metadata']`` may be a dict OR a JSON string."""
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': json.dumps(_gate_metadata()),
        }
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result['error'] == 'consolidation_not_closed'

    @pytest.mark.asyncio
    async def test_the_terminal_exit_gate_still_runs_first(
        self, interceptor, taskmaster
    ):
        """Gate precedence is unchanged: a terminal task is rejected by the
        earlier gate and the scroll is never consulted."""
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'cancelled',
            'metadata': _gate_metadata(),
        }
        scroll = _scroll(_MALFORMED)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result['error'] == 'terminal_exit_rejected'
        assert scroll.calls == []

    @pytest.mark.asyncio
    async def test_the_scroll_is_asked_for_the_gates_topic(self, interceptor):
        scroll = _scroll(_WELL_FORMED)
        interceptor.set_consolidation_scroll(scroll)
        await _set_done(interceptor)
        assert scroll.calls
        assert all(
            call.get('filters', call.get('count')) == {'topic': _TOPIC}
            for call in scroll.calls
        )

    @pytest.mark.asyncio
    async def test_the_scroll_is_scoped_to_the_tasks_project(self, interceptor):
        """A cross-project scroll would judge one project's gate against
        another project's memories."""
        scroll = _scroll(_WELL_FORMED)
        interceptor.set_consolidation_scroll(scroll)
        await _set_done(interceptor)
        expected = resolve_project_id(_PROJECT_ROOT)
        assert all(call['project_id'] == expected for call in scroll.calls)


# --------------------------------------------------------------------------- #
# Task 4808 — ACCEPTANCE 1: the seam now DERIVES `unstamped_live_ids` from the
# gate block's inert provenance, so `unstamped_cluster_member` is reachable in
# production for the first time.
# --------------------------------------------------------------------------- #


class TestSeamUnstampedClusterMember:
    """A cluster member the detector observed live but which never got
    stamped into the topic is invisible to the topic scroll, so it can reach
    the predicate ONLY through the gate block's provenance."""

    @pytest.mark.asyncio
    async def test_refuses_and_names_the_unstamped_id(
        self, interceptor, taskmaster
    ):
        stray = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([stray]),
        }
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED, live_ids=[stray]))
        result = await _set_done(interceptor)

        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'
        codes = [r['code'] for r in result['reasons']]
        assert 'unstamped_cluster_member' in codes
        named = [
            r for r in result['reasons'] if r['code'] == 'unstamped_cluster_member'
        ]
        assert named[0]['ids'] == [stray]

    @pytest.mark.asyncio
    async def test_the_refusal_mutates_nothing(self, interceptor, taskmaster):
        stray = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([stray]),
        }
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED, live_ids=[stray]))
        await _set_done(interceptor)
        taskmaster.set_task_status.assert_not_called()
        taskmaster.set_status_and_stamp_audit.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_stamped_observed_member_still_closes(
        self, interceptor, taskmaster
    ):
        """The measured 2026-08-27 corpus shape: every observed member is
        already in the scroll, so nothing is probed and nothing is refused."""
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([_uuid(1), _uuid(2)]),
        }
        scroll = _scroll(_WELL_FORMED, live_ids=[_uuid(1), _uuid(2)])
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1
        assert scroll.probes == []

    @pytest.mark.asyncio
    async def test_the_probe_is_scoped_to_the_gates_project(
        self, interceptor, taskmaster
    ):
        """A cross-project probe would judge one project\'s gate against
        another project\'s memories — the same argument
        ``set_consolidation_scroll`` already makes for the scroll."""
        stray = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([stray]),
        }
        scroll = _scroll(_WELL_FORMED, live_ids=[stray])
        interceptor.set_consolidation_scroll(scroll)
        await _set_done(interceptor)
        assert scroll.probes == [(stray, resolve_project_id(_PROJECT_ROOT))]

    @pytest.mark.asyncio
    async def test_exists_may_be_passed_explicitly(self, interceptor, taskmaster):
        """``exists=`` is a keyword with a default, so existing POSITIONAL
        callers of ``set_consolidation_scroll(scroll)`` / ``(scroll, count)``
        keep working untouched."""
        stray = _uuid(42)
        probes = []

        async def exists(memory_id, *, project_id):
            probes.append((memory_id, project_id))
            return True

        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([stray]),
        }
        scroll = _scroll(_WELL_FORMED)
        interceptor.set_consolidation_scroll(scroll, exists=exists)
        result = await _set_done(interceptor)
        assert result['error'] == 'consolidation_not_closed'
        assert probes == [(stray, resolve_project_id(_PROJECT_ROOT))]


class TestSeamUnstampedEdgePolicies:
    """The policies the minimal wiring is not FORCED to get right.

    Each is asserted through the real ``set_task_status`` chokepoint, because
    the property that matters is what the seam does, not what the predicate
    would do if called correctly.
    """

    @staticmethod
    def _canonical_claiming(absorbed):
        """``_WELL_FORMED``, with the canonical claiming *absorbed* deleted."""
        canonical = {
            'id': _uuid(1),
            'created_at': '2026-08-24T00:00:00+00:00',
            'metadata': {
                'topic': _TOPIC,
                'canonical': True,
                'supersedes': [absorbed],
            },
        }
        return [canonical, _member(_uuid(2))]

    @pytest.mark.asyncio
    async def test_the_delete_arm_still_closes(self, interceptor, taskmaster):
        """ACCEPTANCE 3. An observed id absent from the scroll AND claimed in
        the canonical\'s ``supersedes`` is a correctly absorbed member, not a
        stray. Without this, every correctly executed delete-arm consolidation
        would become permanently uncloseable."""
        absorbed = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([_uuid(1), absorbed]),
        }
        # The probe would report it LIVE if asked — proving the SUPPRESSION is
        # what closes the gate, not a lucky probe result.
        scroll = _scroll(self._canonical_claiming(absorbed), live_ids=[absorbed])
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1
        assert scroll.probes == []

    @pytest.mark.asyncio
    async def test_hard_deleted_but_unclaimed_still_closes(
        self, interceptor, taskmaster
    ):
        """The PROBE, not the claim, is the final discriminator: an id deleted
        by hand without being recorded in ``supersedes`` is still absorbed."""
        gone = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([gone]),
        }
        scroll = _scroll(_WELL_FORMED, live_ids=[])
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1
        assert scroll.probes == [(gone, resolve_project_id(_PROJECT_ROOT))]

    @pytest.mark.asyncio
    async def test_dormant_without_a_probe(self, interceptor, taskmaster):
        """An unwired probe must not manufacture a refusal it cannot
        substantiate — the inertness property
        ``test_consolidation_gate.py::test_provenance_never_grants_a_pass``
        states, in the direction that matters here."""
        stray = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([stray]),
        }
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED))
        with_prov = await _set_done(interceptor)

        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _gate_metadata(),
        }
        taskmaster.set_task_status.reset_mock()
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED))
        without_prov = await _set_done(interceptor)

        assert with_prov == without_prov
        assert with_prov.get('error') is None

    @pytest.mark.asyncio
    async def test_a_raising_probe_fails_closed(self, interceptor, taskmaster):
        """Same direction as ``TestSeamFailsClosed``\'s scroll cases: the probe
        inherits the fail-closed policy by sitting inside the same try."""
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([_uuid(42)]),
        }
        interceptor.set_consolidation_scroll(
            _scroll(_WELL_FORMED, probe_raises=TimeoutError('qdrant point read'))
        )
        result = await _set_done(interceptor)
        assert result['success'] is False
        assert result['error'] == 'consolidation_not_closed'
        taskmaster.set_task_status.assert_not_called()
        taskmaster.set_status_and_stamp_audit.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncation_suppresses_it_at_the_seam(
        self, interceptor, taskmaster
    ):
        """The caller INHERITS ``evaluate_closure``\'s absence-based guard
        rather than re-implementing it: past the cap, \'not stamped\' and
        \'not seen\' are the same fact."""
        stray = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate([stray]),
        }
        interceptor.set_consolidation_scroll(
            _scroll(_WELL_FORMED, total=500, live_ids=[stray])
        )
        result = await _set_done(interceptor)
        codes = [r['code'] for r in result['reasons']]
        assert 'scroll_incomplete' in codes
        assert 'unstamped_cluster_member' not in codes

    @pytest.mark.asyncio
    async def test_a_waiver_reaches_the_derived_id(self, interceptor, taskmaster):
        """The only sanctioned exit for a stray a curator deliberately kept —
        which also proves the derived ids really do land in
        ``evaluate_closure``\'s ``live_universe``."""
        stray = _uuid(42)
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _prov_gate(
                [stray],
                considered_and_kept=[
                    {
                        'id': stray,
                        'note': 'kept deliberately: a separate claim, not residue',
                        'recorded_at': '2026-08-27T12:00:00+00:00',
                        'recorded_by': 'recon-stage-2',
                    }
                ],
            ),
        }
        interceptor.set_consolidation_scroll(_scroll(_WELL_FORMED, live_ids=[stray]))
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1


# --------------------------------------------------------------------------- #
# Task 4808 — the PRODUCTION binding. Without this the whole change ships
# dormant in the live server, which is the exact "both halves exist and
# neither is wired to the other" failure this task exists to fix.
# --------------------------------------------------------------------------- #


class _StubMemoryService:
    """Just ``get_memory_by_id`` — no store, no config, no MemoryService."""

    def __init__(self, *, result=None, raises=None):
        self.result = result
        self.raises = raises
        self.calls = []

    async def get_memory_by_id(self, project_id, memory_id):
        self.calls.append((project_id, memory_id))
        if self.raises is not None:
            raise self.raises
        return self.result


class TestProductionProbeWiring:
    @pytest.mark.asyncio
    async def test_a_payload_dict_reads_as_live(self):
        from fused_memory.server.main import _closure_exists_for

        stub = _StubMemoryService(
            result={'id': _uuid(42), 'content': 'x', 'metadata': {}}
        )
        probe = _closure_exists_for(stub)
        assert await probe(_uuid(42), project_id='dark_factory') is True

    @pytest.mark.asyncio
    async def test_none_reads_as_absent(self):
        """The two outcomes that distinguish live-but-unstamped from absorbed."""
        from fused_memory.server.main import _closure_exists_for

        probe = _closure_exists_for(_StubMemoryService(result=None))
        assert await probe(_uuid(42), project_id='dark_factory') is False

    @pytest.mark.asyncio
    async def test_project_id_is_the_first_positional_argument(self):
        """``MemoryService.get_memory_by_id(self, project_id, memory_id)``.
        An argument-order slip here would probe the wrong scope and silently
        report every candidate as absent."""
        from fused_memory.server.main import _closure_exists_for

        stub = _StubMemoryService(result=None)
        probe = _closure_exists_for(stub)
        await probe(_uuid(42), project_id='dark_factory')
        assert stub.calls == [('dark_factory', _uuid(42))]

    @pytest.mark.asyncio
    async def test_a_timeout_propagates_rather_than_collapsing_to_false(self):
        """``get_memory_by_id``\'s docstring makes this contract explicit: the
        timeout is PROPAGATED, not collapsed into None, precisely so a caller
        can tell "genuinely absent" from "backend timed out". Collapsing it
        here would let an unreadable store read as "no strays"."""
        from fused_memory.server.main import _closure_exists_for

        probe = _closure_exists_for(
            _StubMemoryService(raises=TimeoutError('qdrant point read'))
        )
        with pytest.raises(TimeoutError):
            await probe(_uuid(42), project_id='dark_factory')

    def test_both_production_call_sites_pass_exists(self):
        """A probe wired at only ONE ``set_consolidation_scroll`` site would
        leave the closure gate half-armed depending on config — the same class
        of silent half-wiring this task fixes. The comment above each site
        already records why both exist (defining the collaborators inside the
        enabled arm left the disabled arm raising NameError at startup)."""
        import inspect  # noqa: PLC0415

        from fused_memory.server import main as server_main

        src = inspect.getsource(server_main)
        sites = [
            block
            for block in src.split('task_interceptor.set_consolidation_scroll(')[1:]
        ]
        assert len(sites) == 2
        for site in sites:
            call = site.split(')')[0]
            assert 'exists=' in call, call


class TestSeamFlagsABlockLessGate:
    """The seam FLAGS a block-less gate carrying a hand-rolled member
    enumeration; it does NOT refuse.

    `operational_mode == 'gate'` is a generic human-gate marker
    (`curator_gate_resolution_sweep.py::extract_open_gate_task_ids` selects on
    exactly that value across all 127 gates), so refusing would brick the 123
    that legitimately carry no block. One WARNING converts a silent dormancy
    into a visible one at zero brick risk.
    """

    @staticmethod
    def _blockless(**extra):
        meta = {'execution_class': 'operational', 'operational_mode': 'gate'}
        meta.update(extra)
        return {'id': '9001', 'status': 'pending', 'metadata': meta}

    @pytest.mark.asyncio
    async def test_it_still_closes(self, interceptor, taskmaster, caplog):
        taskmaster.get_task.return_value = self._blockless(
            related_memory_ids=[_uuid(1), _uuid(2)]
        )
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert taskmaster.set_task_status.await_count == 1

    @pytest.mark.asyncio
    async def test_it_emits_exactly_one_warning_naming_the_facts(
        self, interceptor, taskmaster, caplog
    ):
        import logging  # noqa: PLC0415

        caplog.set_level(logging.WARNING)
        taskmaster.get_task.return_value = self._blockless(
            related_memory_ids=[_uuid(1)]
        )
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        await _set_done(interceptor)

        hits = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'related_memory_ids' in r.getMessage()
        ]
        assert len(hits) == 1
        msg = hits[0].getMessage()
        assert '9001' in msg
        assert GATE_METADATA_KEY in msg
        assert _uuid(1) in msg
        # Grep-stable: an operator must be able to find every dormant gate.
        assert 'consolidation gate is DORMANT' in msg

    @pytest.mark.asyncio
    async def test_an_ordinary_block_less_gate_stays_silent(
        self, interceptor, taskmaster, caplog
    ):
        """`TestSeamDormancy::test_dormant_without_the_gate_key`\'s shape — the
        118-task majority. The existing dormancy tests must not be made
        noisy."""
        import logging  # noqa: PLC0415

        caplog.set_level(logging.WARNING)
        taskmaster.get_task.return_value = self._blockless()
        interceptor.set_consolidation_scroll(_scroll(_MALFORMED))
        result = await _set_done(interceptor)
        assert result.get('error') is None
        assert [
            r for r in caplog.records
            if r.levelno == logging.WARNING and 'DORMANT' in r.getMessage()
        ] == []


class TestKnownGoodCorpusShape:
    """ACCEPTANCE 4, as a permanent guard rather than a one-time manual check.

    The SHAPE measured on 2026-08-28 across all four consolidated topics —
    `worktree-stale-base-premise-verification`,
    `gitops-quarantine-rename-worktree-bare-branch-name`,
    `watchdog-clock-gate-test-isolation` and `mem0-tombstone-coverage`, the
    hand-verified regression corpus named in the task's acceptance criteria:

    * every `provenance.observed_members` id IS present in the live topic
      scroll (the curator hand-stamped them on 2026-08-27);
    * the scroll is COMPLETE;
    * exactly one canonical;
    * `canonical.supersedes` is EMPTY — all four were RETAIN-arm
      consolidations.

    The parametrised counts are the real measured numbers: observed 6/3/3/2
    against scroll totals 9/4/4/3. Member CONTENT is deliberately NOT
    invented — only the counts and the stamped-ness relation are
    load-bearing, and it is the SHAPE that is being pinned. This runs without
    a store, so acceptance 4 keeps a guard after the live re-run in step-16
    has passed into history.
    """

    @pytest.mark.parametrize(
        'topic,observed_count,scroll_total',
        [
            ('worktree-stale-base-premise-verification', 6, 9),
            ('gitops-quarantine-rename-worktree-bare-branch-name', 3, 4),
            ('watchdog-clock-gate-test-isolation', 3, 4),
            ('mem0-tombstone-coverage', 2, 3),
        ],
    )
    @pytest.mark.asyncio
    async def test_closes_with_zero_probes(
        self, interceptor, taskmaster, topic, observed_count, scroll_total
    ):
        members = [
            {
                'id': _uuid(i),
                'created_at': '2026-08-27T00:00:00+00:00',
                'metadata': (
                    {'topic': topic, 'canonical': True}
                    if i == 1
                    else {'topic': topic}
                ),
            }
            for i in range(1, scroll_total + 1)
        ]
        observed = [_uuid(i) for i in range(1, observed_count + 1)]
        taskmaster.get_task.return_value = {
            'id': '9001',
            'status': 'pending',
            'metadata': _gate_metadata(
                **{
                    GATE_METADATA_KEY: {
                        'topic': topic,
                        'provenance': {
                            'report_run': 'recon-2026-08-27',
                            'observed_members': observed,
                            'detector': 'topic-cluster-scan',
                            'authoritative': False,
                        },
                    }
                }
            ),
        }
        # Armed to report every id LIVE, so a probe that IS issued would be
        # visible in `probes` rather than accidentally harmless.
        scroll = _scroll(members, live_ids=[m['id'] for m in members] + observed)
        interceptor.set_consolidation_scroll(scroll)
        result = await _set_done(interceptor)

        assert result.get('error') is None
        assert result.get('reasons') is None or result['reasons'] == []
        assert taskmaster.set_task_status.await_count == 1
        # STRONGER than merely asserting closure: a future refactor that
        # probed every observed member unconditionally would still close, but
        # would issue 14 pointless point reads per gate close across the
        # corpus. Zero is the property that makes this change free.
        assert scroll.probes == []
