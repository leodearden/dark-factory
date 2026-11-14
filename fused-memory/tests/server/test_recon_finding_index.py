"""RunFindingIndex unit tests, on local records that satisfy its Protocols, plus
the B9/B10 boundary tests driven through recon_report's MCP tools over a real
ReconReportStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from fused_memory.reconciliation.flag_identity import description_digest
from fused_memory.server.recon_finding_index import (
    FindingKeys,
    PersistedSlice,
    RunFindingIndex,
    cited_task_key,
)

RUN = 'run-1'


@dataclass
class _Rec:
    finding_id: str
    task_id: str | None = None
    flag_type: str | None = None
    description: str = ''
    cited_tasks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _Entry:
    run_id: str
    stage: str
    findings: list[_Rec] = field(default_factory=list)


def _cite(project_id: str, task_id: Any) -> dict[str, Any]:
    return {'project_id': project_id, 'task_id': task_id, 'title': 't'}


def _index_with(stage: str = 'task_knowledge_sync') -> tuple[RunFindingIndex[_Entry], _Entry]:
    idx: RunFindingIndex[_Entry] = RunFindingIndex(RUN)
    entry = _Entry(RUN, stage)
    idx.install(entry)
    idx.activate(stage)
    return idx, entry


def _add(idx: RunFindingIndex[_Entry], entry: _Entry, rec: _Rec) -> FindingKeys:
    entry.findings.append(rec)
    return idx.register(rec, entry)


class TestKeysFor:
    @pytest.mark.parametrize(
        ('rec', 'stage', 'expected'),
        [
            pytest.param(
                _Rec('f', '42', 'X', 'd'),
                'task_knowledge_sync',
                FindingKeys(signature=('42', 'X')),
                id='ordinary-signature',
            ),
            pytest.param(
                _Rec('f', None, None, '  Some   Text '),
                'task_knowledge_sync',
                FindingKeys(desc_hash=description_digest('some text')),
                id='null-null-desc-hash',
            ),
            pytest.param(
                _Rec('f', None, None, '   '),
                'task_knowledge_sync',
                FindingKeys(),
                id='blank-description-no-key',
            ),
            pytest.param(
                _Rec('f', None, 'X', 'd', [_cite('p', '42')]),
                'task_knowledge_sync',
                FindingKeys(
                    signature=(None, 'X'),
                    cited_task='p:42',
                    derived_signature=('42', 'X'),
                ),
                id='null-task-anchor',
            ),
            pytest.param(
                _Rec('f', None, 'X', 'd', [_cite('p', '42')]),
                'memory_consolidator',
                FindingKeys(signature=(None, 'X'), derived_signature=('42', 'X')),
                id='memory-consolidator-carve-out-is-cited-task-only',
            ),
            pytest.param(
                _Rec('f', '41,42', 'X', 'd', [_cite('p', '42')]),
                'task_knowledge_sync',
                FindingKeys(signature=('41,42', 'X'), derived_signature=('42', 'X')),
                id='composite-member-citation',
            ),
            pytest.param(
                _Rec('f', '41,42', 'X', 'd', [_cite('p', '43')]),
                'task_knowledge_sync',
                FindingKeys(signature=('41,42', 'X')),
                id='composite-non-member-citation',
            ),
            pytest.param(
                _Rec('f', '42', None, 'd', [_cite('p', '42')]),
                'task_knowledge_sync',
                FindingKeys(signature=('42', None)),
                id='null-flag-type-no-citation-keys',
            ),
            pytest.param(
                _Rec('f', None, 'X', 'd', [_cite('p', 42)]),
                'task_knowledge_sync',
                FindingKeys(
                    signature=(None, 'X'),
                    cited_task='p:42',
                    derived_signature=('42', 'X'),
                ),
                id='int-cited-task-id',
            ),
        ],
    )
    def test_keys_of_a_registered_finding(self, rec, stage, expected):
        idx, entry = _index_with(stage)
        entry.findings.append(rec)
        idx.register(rec, entry)
        assert idx.keys_for(rec) == expected

    def test_unregistered_finding_has_no_cited_task_key(self):
        idx, _entry = _index_with()
        rec = _Rec('f', None, 'X', 'd', [_cite('p', '42')])
        keys = idx.keys_for(rec)
        assert keys.cited_task is None
        assert keys.derived_signature == ('42', 'X')

    def test_cited_task_key_spelling(self):
        assert cited_task_key('p', '42') == 'p:42'
        assert cited_task_key('p', 42) == 'p:42'


class TestCitationKeys:
    def test_uncited_finding_gets_the_keys_a_primary_citation_would_add(self):
        idx, entry = _index_with()
        rec = _Rec('f', None, 'X', 'd')
        _add(idx, entry, rec)
        keys = idx.citation_keys(rec, project_id='p', task_id='42')
        assert keys.cited_task == 'p:42'
        assert keys.derived_signature == ('42', 'X')

    def test_memory_consolidator_finding_gets_no_cited_task(self):
        idx, entry = _index_with('memory_consolidator')
        rec = _Rec('f', None, 'X', 'd')
        _add(idx, entry, rec)
        keys = idx.citation_keys(rec, project_id='p', task_id='42')
        assert keys.cited_task is None
        assert keys.derived_signature == ('42', 'X')

    def test_already_cited_finding_gets_no_citation_keys(self):
        idx, entry = _index_with()
        rec = _Rec('f', None, 'X', 'd', [_cite('p', '7')])
        _add(idx, entry, rec)
        keys = idx.citation_keys(rec, project_id='p', task_id='42')
        assert keys.cited_task is None
        assert keys.derived_signature is None


class TestRegister:
    def test_register_returns_keys_and_answers_lookups(self):
        idx, entry = _index_with()
        a = _Rec('a', '42', 'X', 'd')
        n = _Rec('n', None, None, 'note')
        assert _add(idx, entry, a) == idx.keys_for(a)
        _add(idx, entry, n)
        assert idx.by_signature(('42', 'X')) == 'a'
        assert idx.by_desc(description_digest('note')) == 'n'
        assert idx.owner_of('a') is entry
        assert idx.violations() == []

    def test_second_registrant_never_steals_a_signature(self):
        idx, entry = _index_with()
        a = _Rec('a', '42', 'X', 'd')
        b = _Rec('b', '42', 'X', 'd2')
        _add(idx, entry, a)
        _add(idx, entry, b)
        assert idx.by_signature(('42', 'X')) == 'a'
        violations = idx.violations()
        assert len(violations) == 1
        assert "'b'" in violations[0]
        assert RUN in violations[0]

    def test_register_citation_registers_the_primary_citation_keys(self):
        idx, entry = _index_with()
        rec = _Rec('f', None, 'X', 'd')
        _add(idx, entry, rec)
        rec.cited_tasks.append(_cite('p', '42'))
        idx.register_citation(rec)
        assert idx.by_cited_task('p:42') == 'f'
        assert idx.by_signature(('42', 'X')) == 'f'
        assert idx.violations() == []

    def test_register_citation_keeps_the_first_derived_signature_owner(self):
        idx, entry = _index_with()
        owner = _Rec('o', '42', 'X', 'd', [_cite('dark_factory', '42')])
        _add(idx, entry, owner)
        late = _Rec('l', None, 'X', 'd2')
        _add(idx, entry, late)
        late.cited_tasks.append(_cite('other', '42'))
        idx.register_citation(late)
        assert idx.by_signature(('42', 'X')) == 'o'
        assert idx.by_cited_task('other:42') == 'l'
        assert idx.violations() == []

    def test_register_of_a_finding_the_entry_does_not_hold_raises(self):
        idx, entry = _index_with('s1')
        with pytest.raises(ValueError, match=r"run-1.*'f'.*'s1'"):
            idx.register(_Rec('f', '42', 'X', 'd'), entry)
        assert idx.owner_of('f') is None
        assert idx.by_signature(('42', 'X')) is None

    def test_register_of_a_finding_another_entry_owns_raises(self):
        idx, first = _index_with('s1')
        rec = _Rec('f', '42', 'X', 'd')
        _add(idx, first, rec)
        second = _Entry(RUN, 's2', [rec])
        idx.install(second)
        with pytest.raises(ValueError, match=r"run-1.*'f'.*'s2'.*'s1'"):
            idx.register(rec, second)
        assert idx.owner_of('f') is first

    def test_register_citation_of_an_unregistered_finding_raises(self):
        idx, entry = _index_with()
        rec = _Rec('f', None, 'X', 'd')
        _add(idx, entry, rec)
        entry.findings.remove(rec)
        idx.unregister(rec)
        rec.cited_tasks.append(_cite('p', '42'))
        with pytest.raises(ValueError, match=r"run-1.*'f'.*not registered"):
            idx.register_citation(rec)
        assert idx.owner_of('f') is None
        assert idx.by_cited_task('p:42') is None
        assert idx.by_signature(('42', 'X')) is None
        assert idx.violations() == []

    def test_register_citation_of_a_finding_its_owner_no_longer_holds_raises(self):
        idx, entry = _index_with('s1')
        rec = _Rec('f', None, 'X', 'd')
        _add(idx, entry, rec)
        entry.findings.remove(rec)
        rec.cited_tasks.append(_cite('p', '42'))
        with pytest.raises(ValueError, match=r"run-1.*'f'.*'s1'"):
            idx.register_citation(rec)
        assert idx.by_cited_task('p:42') is None
        assert idx.by_signature(('42', 'X')) is None


class TestUnregister:
    def test_unregister_pops_only_keys_still_naming_the_finding(self):
        idx, entry = _index_with()
        owner = _Rec('o', '42', 'X', 'd')
        _add(idx, entry, owner)
        late = _Rec('l', None, 'X', 'd2')
        _add(idx, entry, late)
        late.cited_tasks.append(_cite('p', '42'))
        idx.register_citation(late)

        entry.findings.remove(late)
        idx.unregister(late)

        assert idx.by_signature(('42', 'X')) == 'o'
        assert idx.by_signature((None, 'X')) is None
        assert idx.by_cited_task('p:42') is None
        assert idx.owner_of('l') is None
        assert idx.violations() == []

    def test_unregister_of_an_unknown_finding_is_a_no_op(self):
        idx, entry = _index_with()
        a = _Rec('a', '42', 'X', 'd')
        _add(idx, entry, a)
        idx.unregister(_Rec('ghost', '42', 'X', 'd'))
        assert idx.by_signature(('42', 'X')) == 'a'
        assert idx.violations() == []


class TestFlagTypesForTask:
    def test_collects_non_null_flag_types_including_derived_keys(self):
        idx, entry = _index_with()
        _add(idx, entry, _Rec('a', '42', 'X', 'd'))
        _add(idx, entry, _Rec('b', '42', None, 'd2'))
        anchor = _Rec('c', None, 'Y', 'd3')
        _add(idx, entry, anchor)
        anchor.cited_tasks.append(_cite('p', '42'))
        idx.register_citation(anchor)
        assert idx.flag_types_for_task('42') == {'X', 'Y'}
        assert idx.flag_types_for_task('99') == set()


class TestSession:
    def test_install_makes_the_entry_resident_and_reachable(self):
        idx: RunFindingIndex[_Entry] = RunFindingIndex(RUN)
        entry = _Entry(RUN, 's1')
        assert idx.install(entry) is None
        assert idx.resident_entry('s1') is entry
        assert dict(idx.resident_entries) == {'s1': entry}
        assert idx.reachable_entries() == [entry]
        assert idx.active_stage is None
        assert idx.run_id == RUN

    def test_activate_sets_the_active_stage(self):
        idx, _entry = _index_with('s1')
        assert idx.active_stage == 's1'

    def test_activating_a_non_resident_stage_raises(self):
        idx, _entry = _index_with('s1')
        with pytest.raises(ValueError, match=r"run-1.*'s2'"):
            idx.activate('s2')

    def test_evict_keeps_the_entry_reachable_and_clears_its_active_pointer(self):
        idx, entry = _index_with('s1')
        sibling = _Entry(RUN, 's2')
        idx.install(sibling)
        idx.evict('s1')
        assert idx.active_stage is None
        assert idx.resident_entry('s1') is None
        assert dict(idx.resident_entries) == {'s2': sibling}
        assert entry in idx.reachable_entries()
        assert not idx.quiescent
        idx.evict('s2')
        assert idx.quiescent

    def test_evicting_an_inactive_stage_keeps_the_active_pointer(self):
        idx, _entry = _index_with('s1')
        idx.install(_Entry(RUN, 's2'))
        idx.evict('s2')
        assert idx.active_stage == 's1'

    def test_same_stage_install_returns_the_displaced_entry(self):
        idx, old = _index_with('s1')
        idx.evict('s1')
        fresh = _Entry(RUN, 's1')
        assert idx.install(fresh) is old
        assert idx.reachable_entries() == [fresh]

    def test_installing_another_runs_entry_raises(self):
        idx: RunFindingIndex[_Entry] = RunFindingIndex(RUN)
        with pytest.raises(ValueError, match='run-2'):
            idx.install(_Entry('run-2', 's1'))


class TestPersistedSlice:
    def _populated(self) -> tuple[RunFindingIndex[_Entry], _Entry, _Rec, _Rec, _Rec]:
        idx, entry = _index_with()
        ordinary = _Rec('o', '42', 'X', 'd')
        note = _Rec('n', None, None, 'a note')
        anchor = _Rec('a', None, 'Y', 'd3')
        for rec in (ordinary, note, anchor):
            _add(idx, entry, rec)
        anchor.cited_tasks.append(_cite('p', '7'))
        idx.register_citation(anchor)
        return idx, entry, ordinary, note, anchor

    def test_owned_slice_partitions_like_entry_json(self):
        idx, entry, *_ = self._populated()
        assert idx.owned_slice(entry) == PersistedSlice(
            signature_to_finding={('42', 'X'): 'o', (None, 'Y'): 'a'},
            deschash_to_finding={description_digest('a note'): 'n'},
            sig_anchor_slice={('7', 'Y'): 'a'},
            cited_task_slice={'p:7': 'a'},
        )

    def test_absorb_rebuilds_the_same_lookups(self):
        idx, entry, *_ = self._populated()
        persisted = idx.owned_slice(entry)
        fresh: RunFindingIndex[_Entry] = RunFindingIndex(RUN)
        fresh.install(entry)
        fresh.absorb(entry, persisted)
        assert fresh.by_signature(('42', 'X')) == 'o'
        assert fresh.by_signature(('7', 'Y')) == 'a'
        assert fresh.by_desc(description_digest('a note')) == 'n'
        assert fresh.by_cited_task('p:7') == 'a'
        assert fresh.owner_of('n') is entry
        assert fresh.owned_slice(entry) == persisted
        assert fresh.violations() == []

    def test_absorb_ignores_keys_naming_findings_the_entry_does_not_hold(self):
        entry = _Entry(RUN, 's1', [_Rec('o', '42', 'X', 'd')])
        idx: RunFindingIndex[_Entry] = RunFindingIndex(RUN)
        idx.install(entry)
        idx.absorb(
            entry,
            PersistedSlice(
                signature_to_finding={('42', 'X'): 'o', ('9', 'Z'): 'gone'},
                deschash_to_finding={'h': 'gone'},
                sig_anchor_slice={('8', 'Z'): 'gone'},
                cited_task_slice={'p:8': 'gone'},
            ),
        )
        assert idx.by_signature(('42', 'X')) == 'o'
        assert idx.by_signature(('9', 'Z')) is None
        assert idx.by_signature(('8', 'Z')) is None
        assert idx.by_desc('h') is None
        assert idx.by_cited_task('p:8') is None
        assert idx.violations() == []


class TestViolations:
    def test_unregistered_finding_in_an_entry(self):
        idx, entry = _index_with()
        entry.findings.append(_Rec('stray', '42', 'X', 'd'))
        violations = idx.violations()
        assert violations
        assert all(RUN in v for v in violations)
        assert any("'stray'" in v for v in violations)

    def test_registered_finding_removed_without_unregister(self):
        idx, entry = _index_with()
        rec = _Rec('a', '42', 'X', 'd')
        _add(idx, entry, rec)
        entry.findings.remove(rec)
        violations = idx.violations()
        assert violations
        assert all(RUN in v for v in violations)
        assert any("('42', 'X')" in v for v in violations)

    def test_description_mutated_after_register(self):
        idx, entry = _index_with()
        rec = _Rec('n', None, None, 'old text')
        _add(idx, entry, rec)
        rec.description = 'new text'
        violations = idx.violations()
        assert len(violations) == 2
        assert all(RUN in v and "'n'" in v for v in violations)
        assert any(description_digest('old text') in v for v in violations)
        assert any(description_digest('new text') in v for v in violations)

    def test_displaced_twin_findings_left_registered(self):
        idx, twin = _index_with('s1')
        rec = _Rec('t', '42', 'X', 'd')
        _add(idx, twin, rec)
        idx.evict('s1')
        idx.install(_Entry(RUN, 's1'))
        violations = idx.violations()
        assert violations
        assert all(RUN in v for v in violations)
        assert any("'t'" in v for v in violations)

    def test_consistent_states_report_nothing(self):
        idx, entry = _index_with('s1')
        _add(idx, entry, _Rec('a', '42', 'X', 'd'))
        _add(idx, entry, _Rec('n', None, None, 'note'))
        second = _Entry(RUN, 's2')
        idx.install(second)
        idx.activate('s2')
        idx.evict('s1')
        anchor = _Rec('c', None, 'Y', 'd2')
        _add(idx, second, anchor)
        anchor.cited_tasks.append(_cite('p', '5'))
        idx.register_citation(anchor)
        assert idx.violations() == []


class _FakeMemoryService:
    async def get_entity(self, name, project_id):
        return {'nodes': [{'uuid': 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', 'name': name}]}


class _FakeTaskInterceptor:
    async def get_task(self, task_id, project_root, tag=None):
        return {'id': task_id, 'title': f'T-{task_id}'}


class TestRunIndexAcrossRestartViaMcpTools:
    """B9 and B10: the run index survives eviction, folds and restarts with
    no resurrected finding and no duplicate, driven through the MCP tools."""

    _RUN = 'b-run'
    _S1 = 'memory_consolidator'
    _S2 = 'task_knowledge_sync'

    def _open(self, db, clock):
        from fused_memory.server.recon_report import (
            ReconReportState,
            create_recon_report_server,
        )
        from fused_memory.server.recon_report_store import ReconReportStore

        store = ReconReportStore(db)
        store.open()
        state = ReconReportState(
            ttl_seconds=300,
            clock=lambda: clock[0],
            memory_service=_FakeMemoryService(),
            task_interceptor=_FakeTaskInterceptor(),
            store=store,
        )
        state.known_projects = {'dark_factory': '/home/leo/src/dark-factory'}
        state.hydrate_from_store()
        return store, state, create_recon_report_server(state)._tool_manager

    async def _add(self, tools, task_id, flag_type, description='d'):
        return await tools.call_tool('add_finding', {
            'run_id': self._RUN, 'severity': 'moderate', 'category': 'memory_stale',
            'description': description, 'suggested_action': 'a',
            'task_id': task_id, 'flag_type': flag_type,
        })

    @staticmethod
    def _row_finding_ids(store, stage):
        import json

        (row,) = [r for r in store.load_all() if r['stage'] == stage]
        return {f['finding_id'] for f in json.loads(row['entry_json'])['findings']}

    @pytest.mark.asyncio
    async def test_b9_folded_evicted_finding_is_not_resurrected(self, tmp_path):
        db = tmp_path / 'recon_report_state.db'
        clock = [0.0]
        store, state, tools = self._open(db, clock)
        try:
            await tools.call_tool('start_report', {
                'run_id': self._RUN, 'stage': self._S1, 'project_id': 'dark_factory',
            })
            a_id = (await self._add(tools, '42', 'X', 'claim about 42'))['finding_id']
            b_id = (await self._add(tools, None, 'X', 'unanchored claim'))['finding_id']
            await tools.call_tool('complete', {'run_id': self._RUN, 'summary': 's1'})
            await tools.call_tool('start_report', {
                'run_id': self._RUN, 'stage': self._S2, 'project_id': 'dark_factory',
            })
            assert state.violations() == []

            clock[0] = 301.0
            assert state.tick() == 1
            assert state.violations() == []

            folded = await tools.call_tool('cite_task', {
                'run_id': self._RUN, 'finding_id': b_id,
                'project_id': 'dark_factory', 'task_id': '42',
            })
            assert folded['error'] == 'duplicate_finding'
            assert folded['existing_finding_id'] == a_id
            assert self._row_finding_ids(store, self._S1) == {a_id}
            assert state.violations() == []
        finally:
            store.close()

        store, state, tools = self._open(db, [0.0])
        try:
            assert state.violations() == []
            gone = await tools.call_tool('cite_entity', {
                'run_id': self._RUN, 'finding_id': b_id, 'name': 'Foo',
            })
            assert gone['error'] == 'finding_unknown'
            kept = await tools.call_tool('cite_entity', {
                'run_id': self._RUN, 'finding_id': a_id, 'name': 'Foo',
            })
            assert 'error' not in kept, kept
            refiled = await self._add(tools, '42', 'X', 'claim about 42 again')
            assert refiled['error'] == 'duplicate_finding'
            assert refiled['existing_finding_id'] == a_id
            assert state.violations() == []
        finally:
            store.close()

    @pytest.mark.asyncio
    async def test_b10_restart_between_two_filings_dedups(self, tmp_path):
        db = tmp_path / 'recon_report_state.db'
        store, state, tools = self._open(db, [0.0])
        try:
            await tools.call_tool('start_report', {
                'run_id': self._RUN, 'stage': self._S1, 'project_id': 'dark_factory',
            })
            sig_id = (await self._add(tools, '42', 'X', 'first'))['finding_id']
            note_id = (await self._add(tools, None, None, 'An informational note'))[
                'finding_id'
            ]
        finally:
            store.close()

        store, state, tools = self._open(db, [0.0])
        try:
            assert state.violations() == []
            again = await self._add(tools, '42', 'X', 'second, worded differently')
            assert again['error'] == 'duplicate_finding'
            assert again['existing_finding_id'] == sig_id
            note_again = await self._add(tools, None, None, 'an  informational NOTE')
            assert note_again['error'] == 'duplicate_finding'
            assert note_again['existing_finding_id'] == note_id

            report = state.get_assembled_report(self._RUN, self._S1)
            assert report is not None
            identities = [(i['task_id'], i['flag_type']) for i in report['flagged_items']]
            assert sorted(identities, key=repr) == sorted(
                [('42', 'X'), (None, None)], key=repr
            )
            assert state.violations() == []
        finally:
            store.close()
