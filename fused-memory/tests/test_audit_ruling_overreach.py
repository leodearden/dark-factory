"""Tests for scripts/audit_ruling_overreach.py (task 4716, esc-4639-1).

The script is loaded by path, as ``tests/test_audit_wrong_binding_edges.py``
loads its sibling, so ``scripts/`` never lands on ``sys.path``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

from fused_memory.reconciliation import task_filter
from fused_memory.services import completion_claim_gate

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'audit_ruling_overreach.py'

mod = load_script_module(SCRIPT_PATH, mod_name='audit_ruling_overreach')

DECISIONS = 'decisions_and_rationale'


def _edge(uuid: str, episodes: tuple[str, ...], *, graph: str = 'reify',
          invalid_at: str | None = None, expired_at: str | None = None):
    return mod.Edge(
        graph=graph, uuid=uuid, fact=f'fact of {uuid}', source_name='S',
        target_name='T', episodes=episodes, invalid_at=invalid_at,
        expired_at=expired_at,
    )


class TestParseSourceDescription:
    def test_bare_add_memory_description(self) -> None:
        parsed = mod.parse_source_description('add_memory:decisions_and_rationale')
        assert parsed.category == DECISIONS
        assert parsed.tags == frozenset()

    def test_unverified_claim_prefix_is_a_tag(self) -> None:
        parsed = mod.parse_source_description(
            '[unverified_claim] add_memory:decisions_and_rationale'
        )
        assert parsed.category == DECISIONS
        assert parsed.tags == frozenset({'unverified_claim'})

    def test_composed_prefixes_as_graphiti_backend_writes_them(self) -> None:
        """GraphitiBackend.add_episode applies temporal first, unverified_claim outermost."""
        parsed = mod.parse_source_description(
            '[unverified_claim] [temporal:planning] add_memory:temporal_facts'
        )
        assert parsed.category == 'temporal_facts'
        assert parsed.tags == frozenset({'unverified_claim', 'temporal:planning'})

    @pytest.mark.parametrize('raw', ['REFRESH_FAILURE:x', None, '', '[unverified_claim] '])
    def test_non_add_memory_descriptions_have_no_category(self, raw) -> None:
        assert mod.parse_source_description(raw).category is None


class TestEdgeStatus:
    def test_an_edge_with_neither_stamp_is_served_and_live_strict(self) -> None:
        edge = _edge('e', ('A',))
        assert edge.served is True
        assert edge.live_strict is True

    def test_an_expired_only_edge_is_served_but_not_live_strict(self) -> None:
        """The restored shape task 4714 measured: read paths filter invalid_at only."""
        edge = _edge('e', ('A',), expired_at='2026-09-01T00:00:00+00:00')
        assert edge.served is True
        assert edge.live_strict is False

    def test_an_invalidated_edge_is_not_served(self) -> None:
        edge = _edge('e', ('A',), invalid_at='2026-09-01T00:00:00+00:00')
        assert edge.served is False
        assert edge.live_strict is False


class TestAttributeEdges:
    def test_first_episode_mints_and_later_episodes_corroborate(self) -> None:
        edge = _edge('e1', ('A', 'B'))
        attribution = mod.attribute_edges([edge])
        assert attribution.by_episode[('reify', 'A')].minted == (edge,)
        assert attribution.by_episode[('reify', 'A')].corroborated == ()
        assert attribution.by_episode[('reify', 'B')].minted == ()
        assert attribution.by_episode[('reify', 'B')].corroborated == (edge,)
        assert attribution.unattributed == 0

    def test_an_edge_with_no_episodes_is_counted_as_unattributed(self) -> None:
        attribution = mod.attribute_edges([_edge('orphan', ())])
        assert attribution.unattributed == 1
        assert dict(attribution.by_episode) == {}

    def test_a_minting_episode_repeated_later_does_not_also_corroborate(self) -> None:
        edge = _edge('e1', ('A', 'B', 'A'))
        attribution = mod.attribute_edges([edge])
        assert attribution.by_episode[('reify', 'A')].minted == (edge,)
        assert attribution.by_episode[('reify', 'A')].corroborated == ()

    def test_attribution_is_scoped_by_graph(self) -> None:
        df_edge = _edge('e1', ('A',), graph='dark_factory')
        reify_edge = _edge('e2', ('A',), graph='reify')
        attribution = mod.attribute_edges([df_edge, reify_edge])
        assert attribution.by_episode[('dark_factory', 'A')].minted == (df_edge,)
        assert attribution.by_episode[('reify', 'A')].minted == (reify_edge,)

    def test_an_unknown_episode_has_no_edges(self) -> None:
        attribution = mod.attribute_edges([])
        assert attribution.edges_of('reify', 'missing') == mod.EpisodeEdges((), ())


class TestRecordsAreFrozen:
    def test_episode_is_frozen(self) -> None:
        episode = mod.Episode(
            graph='reify', uuid='u', created_at='2026-09-01T00:00:00+00:00',
            source=mod.parse_source_description('add_memory:decisions_and_rationale'),
            content='body',
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            episode.content = 'other'  # type: ignore[misc]

    def test_edge_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _edge('e', ('A',)).fact = 'other'  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Ruling-shape classifiers and strata (step 3)
# --------------------------------------------------------------------------- #

SPECIMEN_HEADS = {
    ('reify', '59d2d750-4042-4e58-a893-798f5c4fd2c1'): (
        'RULING (Leo, 2026-08-17, task 6080 / esc-6080-3): orient_exp and '
        "transform_exp's angular gate narrow to ANGLE ONLY — DIMENSIONLESS is "
        'REJECTED with a spanned diagnostic, NOT tolerated for back-compat.'
    ),
    ('reify', '5c0884a3-1572-4afe-b7bc-b4786b1095cd'): (
        'Q7/AnalysisResult ruling (Leo 2026-08-10, task 6165, posture 2): trait '
        "AnalysisResult's five stress params (von_mises_stress, "
        'principal_stress_1/2/3, max_shear_stress) retype Real → Stress.'
    ),
    ('dark_factory', '9b33077f-03a9-49e1-ac80-09de623c3d1b'): (
        "L2-watcher decision (esc-1907-37, 2026-06-26): when an L2 escalation's "
        "task is being actively driven by a human's live interactive "
        '`/unblock <id>` session, the watcher STANDS DOWN.'
    ),
    ('dark_factory', 'cf03f276-1351-4861-be67-c7799098f4fb'): (
        'Recovery + ordering design decisions for the two-layer merge-queue PRD '
        '(2026-06-23). (1) RECOVERY: "discard the stale branch and re-dispatch '
        'fresh" (lever B from the containment runbook) is expensive.'
    ),
}

NEGATIVE_CONTROLS = (
    'Task 848 (stale merge-lane cleanup) is done as of 2026-04-20 with commit bb7101',
    'BinOp::Implies lowers to kleene_implies in the evaluator',
)

CONTENT_CLASSIFIERS = (
    'header_ruling_paren', 'header_ruling', 'ruling_lexeme_head', 'decision_anchor_head',
)


def _episode(content: str, *, graph: str = 'reify', uuid: str = 'u',
             source: str = 'add_memory:decisions_and_rationale',
             created_at: str = '2026-09-01T00:00:00+00:00'):
    return mod.Episode(
        graph=graph, uuid=uuid, created_at=created_at,
        source=mod.parse_source_description(source), content=content,
    )


def _specimen_episodes():
    return {
        (graph, uuid): _episode(head, graph=graph, uuid=uuid)
        for (graph, uuid), head in SPECIMEN_HEADS.items()
    }


_59D2 = ('reify', '59d2d750-4042-4e58-a893-798f5c4fd2c1')
_5C08 = ('reify', '5c0884a3-1572-4afe-b7bc-b4786b1095cd')
_9B33 = ('dark_factory', '9b33077f-03a9-49e1-ac80-09de623c3d1b')
_CF03 = ('dark_factory', 'cf03f276-1351-4861-be67-c7799098f4fb')

EXPECTED_MATCHES = {
    'category_decisions': {_59D2, _5C08, _9B33, _CF03},
    'header_ruling_paren': {_59D2},
    'header_ruling': {_59D2},
    'ruling_lexeme_head': {_59D2, _5C08},
    'decision_anchor_head': {_59D2, _5C08, _9B33, _CF03},
}


class TestClassifiers:
    def test_the_candidate_set_is_exactly_the_five_named(self) -> None:
        assert set(mod.CLASSIFIERS) == set(EXPECTED_MATCHES)

    def test_the_candidate_set_is_read_only(self) -> None:
        with pytest.raises(TypeError):
            mod.CLASSIFIERS['another'] = lambda episode: True  # type: ignore[index]

    @pytest.mark.parametrize('name', sorted(EXPECTED_MATCHES))
    @pytest.mark.parametrize('key', sorted(SPECIMEN_HEADS))
    def test_truth_table_over_the_specimens(self, name, key) -> None:
        episode = _specimen_episodes()[key]
        assert mod.CLASSIFIERS[name](episode) is (key in EXPECTED_MATCHES[name])

    @pytest.mark.parametrize('name', CONTENT_CLASSIFIERS)
    @pytest.mark.parametrize('content', NEGATIVE_CONTROLS)
    def test_negative_controls_match_no_content_classifier(self, name, content) -> None:
        assert mod.CLASSIFIERS[name](_episode(content)) is False

    @pytest.mark.parametrize('name', ['ruling_lexeme_head', 'decision_anchor_head'])
    def test_a_lexeme_past_the_head_window_does_not_count(self, name) -> None:
        content = 'x' * 250 + ' ruling (Leo, 2026-09-01, esc-1-2)'
        assert mod.HEAD_CHARS == 200
        assert mod.CLASSIFIERS[name](_episode(content)) is False

    def test_category_decisions_reads_the_parsed_category(self) -> None:
        episode = _episode('RULING (Leo, 2026-09-01): x', source='add_memory:temporal_facts')
        assert mod.CLASSIFIERS['category_decisions'](episode) is False


class TestStrata:
    def test_strata_are_named_in_precedence_order(self) -> None:
        assert mod.STRATA == ('ruling_lexeme', 'decision_anchor', 'other_decisions')

    @pytest.mark.parametrize(('key', 'stratum'), [
        (_59D2, 'ruling_lexeme'),
        (_5C08, 'ruling_lexeme'),
        (_9B33, 'decision_anchor'),
        (_CF03, 'decision_anchor'),
    ])
    def test_specimen_strata(self, key, stratum) -> None:
        assert mod.stratum_of(_specimen_episodes()[key]) == stratum

    def test_a_decision_record_with_no_lexeme_is_other_decisions(self) -> None:
        assert mod.stratum_of(_episode(NEGATIVE_CONTROLS[0])) == 'other_decisions'

    def test_a_ruling_outside_the_decisions_category_is_still_ruling_lexeme(self) -> None:
        episode = _episode('RULING (Leo, 2026-09-01): x', source='add_memory:temporal_facts')
        assert mod.stratum_of(episode) == 'ruling_lexeme'

    def test_a_non_decision_record_with_no_lexeme_has_no_stratum(self) -> None:
        episode = _episode(NEGATIVE_CONTROLS[1], source='add_memory:entities_and_relations')
        assert mod.stratum_of(episode) is None


class TestSpecimens:
    def test_the_four_specimens_are_tabled_with_full_uuids(self) -> None:
        assert {(s.graph, s.episode_uuid) for s in mod.SPECIMENS} == set(SPECIMEN_HEADS)
        assert {s.edge_uuid for s in mod.SPECIMENS} == {
            '4f99fbf2-3608-4437-aba5-eeeb99ddf991',
            'c6ac6d99-a98f-4f52-a59d-0bbbbfadd0e1',
            'b2267a98-49db-49ae-862c-75d42439ddd1',
            'df7b7746-066e-4380-b4ea-f684cac1c6d0',
        }
        assert all(s.note.strip() for s in mod.SPECIMENS)

    @pytest.mark.parametrize(('name', 'recall'), [
        ('category_decisions', 1.0),
        ('header_ruling_paren', 0.25),
        ('header_ruling', 0.25),
        ('ruling_lexeme_head', 0.5),
        ('decision_anchor_head', 1.0),
    ])
    def test_specimen_recall(self, name, recall) -> None:
        assert mod.specimen_recall(name, _specimen_episodes()) == recall

    def test_recall_is_over_the_specimens_that_were_read(self) -> None:
        episodes = _specimen_episodes()
        del episodes[_CF03]
        assert mod.specimen_recall('ruling_lexeme_head', episodes) == pytest.approx(2 / 3, abs=1e-4)

    def test_recall_with_no_specimen_read_is_not_computed(self) -> None:
        assert mod.specimen_recall('ruling_lexeme_head', {}) is None


# --------------------------------------------------------------------------- #
# Counterfactual census of the write-time detectors (step 5)
# --------------------------------------------------------------------------- #

# Positive controls, each copied from the detector's OWN suite, so a detector
# that never fires is distinguishable from a mis-invoked one.
POSITIVE_CONTROLS = {
    # tests/test_completion_claim_gate.py::TestClauseBoundaryIsolation::test_plain_completion_claim_still_extracts
    'completion_claim': 'Task 777 has landed.',
    # tests/test_task_filter.py::TestIsProposedResolutionFraming::test_positive_proposed_fix_phrase
    'proposed_resolution': 'Proposed fix: move the transition after the await.',
    # tests/test_task_filter.py::TestIsBatchPlanFraming::test_positive_decompose_and_queue_with_range
    'batch_plan': 'PRD decomposed into tasks 1985-2002 and queued',
}
# tests/test_completion_claim_gate.py::TestClauseBoundaryIsolation::test_plain_pending_statement_is_still_not_a_claim
NEGATIVE_CONTROL = 'Task 888 is still pending.'


class TestDetectorsAreTheSharedOnes:
    def test_framing_predicates_are_imported_not_copied(self) -> None:
        assert mod.is_proposed_resolution_framing is task_filter.is_proposed_resolution_framing
        assert mod.is_batch_plan_framing is task_filter.is_batch_plan_framing

    def test_completion_claim_extraction_is_imported_not_copied(self) -> None:
        assert mod.extract_completion_claims is completion_claim_gate.extract_completion_claims

    def test_the_tag_vocabulary_is_imported_not_copied(self) -> None:
        assert mod.UNVERIFIED_CLAIM_TAG is completion_claim_gate.UNVERIFIED_CLAIM_TAG


class TestDetectorHits:
    def test_detectors_are_named_in_report_order(self) -> None:
        assert mod.DETECTORS == (
            'unverified_claim_tag', 'completion_claim', 'proposed_resolution', 'batch_plan',
        )

    @pytest.mark.parametrize('detector', sorted(POSITIVE_CONTROLS))
    def test_each_content_detector_fires_on_its_own_positive_control(self, detector) -> None:
        hits = mod.detector_hits(_episode(POSITIVE_CONTROLS[detector]))
        assert detector in hits

    def test_the_tag_detector_reads_the_parsed_source_tags(self) -> None:
        episode = _episode(
            NEGATIVE_CONTROL, source='[unverified_claim] add_memory:decisions_and_rationale',
        )
        assert mod.detector_hits(episode) == frozenset({'unverified_claim_tag'})

    def test_nothing_fires_on_the_negative_control(self) -> None:
        assert mod.detector_hits(_episode(NEGATIVE_CONTROL)) == frozenset()

    def test_no_wired_detector_fires_on_the_specimens(self) -> None:
        """4715's gate misses all four known overreach episodes. Unwired batch_plan
        does fire on cf03f276, reading '(2026-06-23)' as a task-id range."""
        for episode in _specimen_episodes().values():
            assert not mod.detector_hits(episode) & mod.WIRED_ON_ADD_MEMORY, episode.uuid

    def test_odd_content_never_raises(self) -> None:
        episode = mod.Episode(
            graph='reify', uuid='u', created_at='2026-09-01T00:00:00+00:00',
            source=mod.parse_source_description(None), content=None,  # type: ignore[arg-type]
        )
        assert mod.detector_hits(episode) == frozenset()

    def test_completion_claims_resolve_against_the_episode_graph(self) -> None:
        hits = mod.detector_hits(
            _episode('reify task 777 has landed.', graph='dark_factory'),
            known_project_ids=frozenset({'dark_factory', 'reify'}),
        )
        assert 'completion_claim' in hits


class TestWiredOnAddMemory:
    def test_only_the_completion_gate_and_its_tag_are_wired_after_4715(self) -> None:
        assert frozenset({'unverified_claim_tag', 'completion_claim'}) == mod.WIRED_ON_ADD_MEMORY

    def test_the_wired_set_is_a_subset_of_the_detectors(self) -> None:
        assert set(mod.DETECTORS) >= mod.WIRED_ON_ADD_MEMORY


class TestDetectorCensus:
    def test_census_counts_hits_and_episodes_per_stratum(self) -> None:
        census = mod.detector_census({
            'ruling_lexeme': [
                _episode(POSITIVE_CONTROLS['completion_claim'], uuid='a'),
                _episode(NEGATIVE_CONTROL, uuid='b'),
            ],
            'other_decisions': [],
        })
        assert census['ruling_lexeme'] == {
            'episodes': 2, 'unverified_claim_tag': 0, 'completion_claim': 1,
            'proposed_resolution': 0, 'batch_plan': 0,
        }
        assert census['other_decisions']['episodes'] == 0


# --------------------------------------------------------------------------- #
# Deterministic stratified out-of-sample sample and worksheet (step 7)
# --------------------------------------------------------------------------- #

WINDOW = {'window_start': '2026-08-25T00:00:00+00:00', 'window_end': '2026-10-01T00:00:00+00:00'}
RULING_HEAD = 'RULING (Leo, 2026-09-01): x'
ANCHOR_HEAD = 'Merge-lane decision (esc-1-2): y'
PLAIN_DECISION = 'We keep the merge lane serial.'


def _sha(graph: str, uuid: str) -> str:
    return hashlib.sha256(f'{graph}:{uuid}'.encode()).hexdigest()


def _population():
    episodes = []
    for graph in ('dark_factory', 'reify'):
        for i in range(5):
            episodes.append(_episode(RULING_HEAD, graph=graph, uuid=f'r{i}'))
            episodes.append(_episode(ANCHOR_HEAD, graph=graph, uuid=f'a{i}'))
        episodes.append(_episode(PLAIN_DECISION, graph=graph, uuid='o0'))
    return episodes


class TestSelectSample:
    def test_input_order_does_not_change_the_output(self) -> None:
        episodes = _population()
        expected = mod.select_sample(episodes, cap=3, **WINDOW)
        for seed in range(5):
            shuffled = list(episodes)
            random.Random(seed).shuffle(shuffled)
            assert mod.select_sample(shuffled, cap=3, **WINDOW) == expected

    def test_takes_the_first_cap_by_hash_within_each_graph_and_stratum(self) -> None:
        sample = mod.select_sample(_population(), cap=3, **WINDOW)
        df_ruling = [s.episode.uuid for s in sample
                     if s.episode.graph == 'dark_factory' and s.stratum == 'ruling_lexeme']
        expected = sorted((f'r{i}' for i in range(5)), key=lambda u: _sha('dark_factory', u))[:3]
        assert df_ruling == expected

    def test_a_group_smaller_than_cap_yields_all_its_members(self) -> None:
        sample = mod.select_sample(_population(), cap=3, **WINDOW)
        other = [s for s in sample if s.stratum == 'other_decisions']
        assert {(s.episode.graph, s.episode.uuid) for s in other} == {
            ('dark_factory', 'o0'), ('reify', 'o0'),
        }

    def test_output_is_ordered_by_graph_stratum_then_hash(self) -> None:
        sample = mod.select_sample(_population(), cap=3, **WINDOW)
        keys = [(s.episode.graph, mod.STRATA.index(s.stratum), s.sample_key) for s in sample]
        assert keys == sorted(keys)
        assert all(s.sample_key == _sha(s.episode.graph, s.episode.uuid) for s in sample)
        assert isinstance(sample, tuple)

    @pytest.mark.parametrize('created_at', [
        '2026-08-24T23:59:59+00:00',
        '2026-10-01T00:00:00+00:00',
    ])
    def test_episodes_outside_the_half_open_window_never_appear(self, created_at) -> None:
        episode = _episode(RULING_HEAD, created_at=created_at)
        assert mod.select_sample([episode], cap=8, **WINDOW) == ()

    def test_the_window_compares_instants_not_strings(self) -> None:
        """Each string sorts on the wrong side of the window end from its instant."""
        episode = _episode(RULING_HEAD, created_at='2026-09-30T23:30:00-02:00')
        assert mod.select_sample([episode], cap=8, **WINDOW) == ()
        inside = _episode(RULING_HEAD, created_at='2026-10-01T01:30:00+02:00')
        assert len(mod.select_sample([inside], cap=8, **WINDOW)) == 1

    def test_episodes_with_no_stratum_never_appear(self) -> None:
        episode = _episode(PLAIN_DECISION, source='add_memory:entities_and_relations')
        assert mod.select_sample([episode], cap=8, **WINDOW) == ()


class TestSampleDefinition:
    def test_round_trips_through_a_dict(self) -> None:
        definition = mod.SampleDefinition(cap=8, **WINDOW)
        assert mod.SampleDefinition.from_dict(definition.to_dict()) == definition

    def test_defaults_are_the_frozen_window(self) -> None:
        definition = mod.SampleDefinition()
        assert definition.window_start == mod.DEFAULT_WINDOW_START == WINDOW['window_start']
        assert definition.window_end == mod.DEFAULT_WINDOW_END == WINDOW['window_end']
        assert definition.cap == mod.DEFAULT_CAP == 8
        assert definition.strata == mod.STRATA
        assert definition.hash_rule == 'sha256(graph:uuid)'

    def test_the_dict_form_is_plain_json(self) -> None:
        blob = json.dumps(mod.SampleDefinition().to_dict())
        assert mod.SampleDefinition.from_dict(json.loads(blob)) == mod.SampleDefinition()

    def test_select_applies_the_definition(self) -> None:
        definition = mod.SampleDefinition(cap=2, **WINDOW)
        assert definition.select(_population()) == mod.select_sample(_population(), cap=2, **WINDOW)


class TestWorksheetRows:
    def test_rows_offer_only_minted_edges_and_count_corroborations(self) -> None:
        ep = _episode(RULING_HEAD, graph='reify', uuid='E')
        minted = _edge('m1', ('E', 'Z'), expired_at='2026-09-02T00:00:00+00:00')
        corroborated = _edge('c1', ('Z', 'E'))
        sample = mod.select_sample([ep], cap=8, **WINDOW)
        attribution = mod.attribute_edges([minted, corroborated])
        (row,) = list(mod.worksheet_rows(sample, attribution))
        assert row == {
            'graph': 'reify', 'uuid': 'E', 'stratum': 'ruling_lexeme',
            'classifiers': list(mod.CLASSIFIERS),
            'created_at': '2026-09-01T00:00:00+00:00',
            'category': 'decisions_and_rationale', 'content': RULING_HEAD,
            'minted': [{
                'edge_uuid': 'm1', 'fact': 'fact of m1', 'source_name': 'S',
                'target_name': 'T', 'served': True, 'live_strict': False,
            }],
            'corroborated_count': 1,
        }

    def test_an_episode_with_no_minted_edges_still_appears(self) -> None:
        ep = _episode(ANCHOR_HEAD, graph='reify', uuid='E')
        sample = mod.select_sample([ep], cap=8, **WINDOW)
        (row,) = list(mod.worksheet_rows(sample, mod.attribute_edges([])))
        assert row['minted'] == []
        assert row['corroborated_count'] == 0

    def test_minted_edges_are_listed_in_uuid_order(self) -> None:
        ep = _episode(ANCHOR_HEAD, graph='reify', uuid='E')
        edges = [_edge('m2', ('E',)), _edge('m1', ('E',))]
        sample = mod.select_sample([ep], cap=8, **WINDOW)
        (row,) = list(mod.worksheet_rows(sample, mod.attribute_edges(edges)))
        assert [m['edge_uuid'] for m in row['minted']] == ['m1', 'm2']


# --------------------------------------------------------------------------- #
# Verdict-file validation, fail closed (step 9)
# --------------------------------------------------------------------------- #

EXPECTED_EDGES = {('reify', 'e1'): 'E', ('reify', 'e2'): 'E', ('dark_factory', 'e3'): 'F'}
EXISTING_EDGES = frozenset(EXPECTED_EDGES) | {('reify', 'outside')}


def _verdict(graph: str, edge_uuid: str, label: str = 'holding', *,
             episode_uuid: str | None = None, rationale: str = 'states the holding "x"'):
    return {
        'graph': graph,
        'episode_uuid': episode_uuid or EXPECTED_EDGES.get((graph, edge_uuid), 'E'),
        'edge_uuid': edge_uuid, 'fact': f'fact of {edge_uuid}', 'label': label,
        'rationale': rationale,
    }


def _file(*verdicts, sample=None):
    return {
        'sample': sample if sample is not None else mod.SampleDefinition().to_dict(),
        'verdicts': list(verdicts) if verdicts else [
            _verdict('reify', 'e1'), _verdict('reify', 'e2', 'overreach'),
            _verdict('dark_factory', 'e3', 'bookkeeping'),
        ],
    }


def _load(obj):
    return mod.load_verdicts(
        obj, expected_edges=EXPECTED_EDGES, existing_edges=EXISTING_EDGES,
        definition=mod.SampleDefinition(),
    )


class TestLoadVerdicts:
    def test_the_label_vocabulary_is_closed(self) -> None:
        assert mod.LABELS == (
            'holding', 'bookkeeping', 'context', 'overreach', 'misbound', 'unjudgeable',
        )

    def test_a_valid_file_loads(self) -> None:
        verdicts = _load(_file())
        assert dict(verdicts.by_edge) == {
            ('reify', 'e1'): 'holding', ('reify', 'e2'): 'overreach',
            ('dark_factory', 'e3'): 'bookkeeping',
        }
        assert verdicts.stale == ()
        assert verdicts.definition == mod.SampleDefinition()

    def test_a_valid_file_round_trips(self) -> None:
        verdicts = _load(_file())
        assert _load(json.loads(json.dumps(verdicts.to_dict()))) == verdicts

    def test_the_dict_form_is_sorted(self) -> None:
        ordered = [(v['graph'], v['episode_uuid'], v['edge_uuid'])
                   for v in _load(_file()).to_dict()['verdicts']]
        assert ordered == sorted(ordered)

    def test_the_verdict_set_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _load(_file()).stale = ()  # type: ignore[misc]

    def test_unknown_label(self) -> None:
        obj = _file(_verdict('reify', 'e1', 'wrong'), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'))
        with pytest.raises(mod.UnknownLabel) as caught:
            _load(obj)
        assert caught.value.edge_uuids == ('e1',)
        assert isinstance(caught.value, mod.VerdictError)

    @pytest.mark.parametrize('rationale', ['', '   '])
    def test_missing_rationale(self, rationale) -> None:
        obj = _file(_verdict('reify', 'e1', rationale=rationale), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'))
        with pytest.raises(mod.MissingRationale) as caught:
            _load(obj)
        assert caught.value.edge_uuids == ('e1',)

    def test_duplicate_verdict(self) -> None:
        obj = _file(_verdict('reify', 'e1'), _verdict('reify', 'e1', 'context'),
                    _verdict('reify', 'e2'), _verdict('dark_factory', 'e3'))
        with pytest.raises(mod.DuplicateVerdict) as caught:
            _load(obj)
        assert caught.value.edge_uuids == ('e1',)

    def test_missing_verdicts_lists_every_missing_edge(self) -> None:
        with pytest.raises(mod.MissingVerdicts) as caught:
            _load(_file(_verdict('reify', 'e1')))
        assert caught.value.edge_uuids == ('e2', 'e3')

    def test_a_verdict_for_a_vanished_edge_is_stale_not_an_error(self) -> None:
        obj = _file(_verdict('reify', 'e1'), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'), _verdict('reify', 'merged-away'))
        verdicts = _load(obj)
        assert [v.edge_uuid for v in verdicts.stale] == ['merged-away']
        assert ('reify', 'merged-away') not in verdicts.by_edge

    def test_a_verdict_for_an_existing_edge_outside_the_sample(self) -> None:
        obj = _file(_verdict('reify', 'e1'), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'), _verdict('reify', 'outside'))
        with pytest.raises(mod.OutOfSampleVerdict) as caught:
            _load(obj)
        assert caught.value.edge_uuids == ('outside',)

    def test_a_verdict_naming_the_wrong_episode_is_out_of_sample(self) -> None:
        obj = _file(_verdict('reify', 'e1', episode_uuid='not-E'), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'))
        with pytest.raises(mod.OutOfSampleVerdict) as caught:
            _load(obj)
        assert caught.value.edge_uuids == ('e1',)

    def test_a_sample_block_disagreeing_with_the_definition(self) -> None:
        sample = mod.SampleDefinition(cap=9).to_dict()
        with pytest.raises(mod.SampleMismatch):
            _load(_file(sample=sample))

    @pytest.mark.parametrize('obj', [
        [], {'verdicts': []}, {'sample': {}, 'verdicts': []},
        {'sample': mod.SampleDefinition().to_dict(), 'verdicts': {}},
        {'sample': mod.SampleDefinition().to_dict(), 'verdicts': [{'graph': 'reify'}]},
    ])
    def test_a_malformed_file(self, obj) -> None:
        with pytest.raises(mod.MalformedVerdicts):
            _load(obj)
