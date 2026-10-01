"""Tests for scripts/audit_ruling_overreach.py (task 4716, esc-4639-1).

The script is loaded by path, as ``tests/test_audit_wrong_binding_edges.py``
loads its sibling, so ``scripts/`` never lands on ``sys.path``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
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
                mod.detector_hits(_episode(POSITIVE_CONTROLS['completion_claim'], uuid='a')),
                mod.detector_hits(_episode(POSITIVE_CONTROLS['batch_plan'], uuid='b')),
                mod.detector_hits(_episode(NEGATIVE_CONTROL, uuid='c')),
            ],
            'other_decisions': [],
        })
        assert census['ruling_lexeme'] == {
            'episodes': 3, 'unverified_claim_tag': 0, 'completion_claim': 1,
            'proposed_resolution': 0, 'batch_plan': 1, 'any_detector': 2, 'any_wired': 1,
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
SAMPLED_EPISODES = frozenset({('reify', 'E'), ('dark_factory', 'F'), ('reify', 'MINTED-NOTHING')})
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
        obj, expected_edges=EXPECTED_EDGES, sampled_episodes=SAMPLED_EPISODES,
        existing_edges=EXISTING_EDGES, definition=mod.SampleDefinition(),
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

    def test_a_vanished_edge_of_a_sampled_episode_that_minted_nothing_is_stale(self) -> None:
        obj = _file(_verdict('reify', 'e1'), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'),
                    _verdict('reify', 'merged-away', episode_uuid='MINTED-NOTHING'))
        assert [v.edge_uuid for v in _load(obj).stale] == ['merged-away']

    def test_a_nonexistent_edge_on_an_unsampled_episode_is_out_of_sample(self) -> None:
        """A mistyped edge uuid must not pass silently as stale."""
        obj = _file(_verdict('reify', 'e1'), _verdict('reify', 'e2'),
                    _verdict('dark_factory', 'e3'),
                    _verdict('reify', 'no-such-edge', episode_uuid='never-sampled'))
        with pytest.raises(mod.OutOfSampleVerdict) as caught:
            _load(obj)
        assert caught.value.edge_uuids == ('no-such-edge',)

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


# --------------------------------------------------------------------------- #
# Rates (step 11)
# --------------------------------------------------------------------------- #

class TestWilsonInterval:
    @pytest.mark.parametrize(('k', 'n', 'interval'), [
        (13, 174, (0.0442, 0.1236)),  # esc-4639-1 '7.5% [4.4-12.4%]'
        (10, 111, (0.0497, 0.1579)),  # esc-4639-1 '9.0% [5.0-15.8%]'
        (0, 20, (0.0, 0.1611)),
    ])
    def test_reproduces_the_published_intervals(self, k, n, interval) -> None:
        assert mod.wilson_interval(k, n) == interval

    def test_an_empty_denominator_is_not_computed(self) -> None:
        assert mod.wilson_interval(0, 0) is None


def _minted(uuid: str, *, served: bool = True, live_strict: bool = True) -> dict:
    return {'edge_uuid': uuid, 'fact': f'fact of {uuid}', 'source_name': 'S',
            'target_name': 'T', 'served': served, 'live_strict': live_strict}


RATE_ROWS = (
    {'graph': 'reify', 'uuid': 'A', 'stratum': 'ruling_lexeme',
     'classifiers': list(mod.CLASSIFIERS), 'created_at': '2026-09-01T00:00:00+00:00',
     'category': DECISIONS, 'content': RULING_HEAD, 'corroborated_count': 0,
     'minted': [_minted('a1'), _minted('a2', live_strict=False), _minted('a3'),
                _minted('a4', served=False, live_strict=False)]},
    {'graph': 'dark_factory', 'uuid': 'B', 'stratum': 'other_decisions',
     'classifiers': ['category_decisions'], 'created_at': '2026-09-01T00:00:00+00:00',
     'category': DECISIONS, 'content': PLAIN_DECISION, 'corroborated_count': 2,
     'minted': [_minted('b1'), _minted('b2'), _minted('b3'), _minted('b4')]},
)
RATE_LABELS = {
    ('reify', 'a1'): 'holding', ('reify', 'a2'): 'overreach',
    ('reify', 'a3'): 'bookkeeping', ('reify', 'a4'): 'overreach',
    ('dark_factory', 'b1'): 'holding', ('dark_factory', 'b2'): 'context',
    ('dark_factory', 'b3'): 'unjudgeable', ('dark_factory', 'b4'): 'misbound',
}


def _rate_verdicts():
    rows = {(r['graph'], m['edge_uuid']): r['uuid'] for r in RATE_ROWS for m in r['minted']}
    obj = {'sample': mod.SampleDefinition().to_dict(), 'verdicts': [
        {'graph': g, 'episode_uuid': rows[(g, e)], 'edge_uuid': e, 'fact': f'fact of {e}',
         'label': label, 'rationale': 'r'} for (g, e), label in RATE_LABELS.items()
    ]}
    return mod.load_verdicts(obj, expected_edges=rows, existing_edges=rows,
                             sampled_episodes={(r['graph'], r['uuid']) for r in RATE_ROWS},
                             definition=mod.SampleDefinition())


def _rate(k: int, n: int) -> dict:
    return {'k': k, 'n': n, 'rate': round(k / n, 4) if n else None,
            'ci': list(mod.wilson_interval(k, n)) if n else None}


class TestAdjudicatedRates:
    def test_ruling_lexeme_stratum(self) -> None:
        rates = mod.adjudicated_rates(RATE_ROWS, _rate_verdicts())['ruling_lexeme']
        assert rates['episodes'] == 1
        assert rates['minted'] == 4
        assert rates['overreach_rate_minted'] == _rate(2, 4)
        assert rates['overreach_rate_substantive'] == _rate(2, 3)
        assert rates['episode_hit_rate'] == _rate(1, 1)
        assert rates['holding_share'] == _rate(1, 4)
        assert rates['holding_share_in_hit_episodes'] == _rate(1, 4)
        assert rates['served_fraction_by_label']['overreach'] == _rate(1, 2)
        assert rates['served_fraction_by_label']['holding'] == _rate(1, 1)
        assert rates['live_strict_fraction_by_label']['overreach'] == _rate(0, 2)
        assert rates['misbound_count'] == 0
        assert rates['label_counts'] == {
            'holding': 1, 'bookkeeping': 1, 'context': 0, 'overreach': 2,
            'misbound': 0, 'unjudgeable': 0,
        }

    def test_misbound_is_counted_and_excluded_from_overreach(self) -> None:
        rates = mod.adjudicated_rates(RATE_ROWS, _rate_verdicts())['other_decisions']
        assert rates['misbound_count'] == 1
        assert rates['overreach_rate_minted'] == _rate(0, 4)
        assert rates['overreach_rate_substantive'] == _rate(0, 3)
        assert rates['episode_hit_rate'] == _rate(0, 1)
        assert rates['holding_share_in_hit_episodes'] == _rate(0, 0)

    def test_the_all_roll_up(self) -> None:
        rates = mod.adjudicated_rates(RATE_ROWS, _rate_verdicts())['all']
        assert rates['episodes'] == 2
        assert rates['minted'] == 8
        assert rates['overreach_rate_minted'] == _rate(2, 8)
        assert rates['overreach_rate_substantive'] == _rate(2, 6)
        assert rates['episode_hit_rate'] == _rate(1, 2)
        assert rates['holding_share'] == _rate(2, 8)

    def test_an_empty_stratum_reports_none_not_zero(self) -> None:
        rates = mod.adjudicated_rates(RATE_ROWS, _rate_verdicts())['decision_anchor']
        assert rates['episodes'] == 0
        assert rates['overreach_rate_minted'] == _rate(0, 0)
        assert rates['overreach_rate_minted']['rate'] is None
        assert rates['served_fraction_by_label']['overreach']['rate'] is None

    def test_every_stratum_and_the_roll_up_are_present(self) -> None:
        assert set(mod.adjudicated_rates(RATE_ROWS, _rate_verdicts())) == {*mod.STRATA, 'all'}


class TestPerClassifierRates:
    def test_rates_are_restricted_to_the_episodes_each_classifier_matches(self) -> None:
        rates = mod.per_classifier_rates(RATE_ROWS, _rate_verdicts())
        assert set(rates) == set(mod.CLASSIFIERS)
        assert rates['header_ruling_paren']['episodes'] == 1
        assert rates['header_ruling_paren']['overreach_rate_minted'] == _rate(2, 4)
        assert rates['category_decisions']['episodes'] == 2
        assert rates['category_decisions']['overreach_rate_minted'] == _rate(2, 8)


class TestWeightedRate:
    def test_equal_weights_reduce_to_the_unweighted_rate(self) -> None:
        weighted = mod.weighted_rate([(3.0, 13, 100), (3.0, 0, 74)])
        assert weighted == {'rate': _rate(13, 174)['rate'], 'ci': _rate(13, 174)['ci'],
                            'effective_n': 174.0}

    def test_unequal_weights_move_the_rate_and_shrink_the_effective_n(self) -> None:
        weighted = mod.weighted_rate([(10.0, 2, 4), (30.0, 0, 4)])
        assert weighted['rate'] == 0.125
        assert weighted['effective_n'] == 6.4

    @pytest.mark.parametrize('strata', [[], [(5.0, 0, 0)], [(0.0, 1, 4)]])
    def test_no_weighted_denominator_is_not_computed(self, strata) -> None:
        assert mod.weighted_rate(strata) == {'rate': None, 'ci': None, 'effective_n': None}


WINDOW_POPULATION = {
    'reify': {'ruling_lexeme': 10, 'decision_anchor': 0, 'other_decisions': 0},
    'dark_factory': {'ruling_lexeme': 0, 'decision_anchor': 0, 'other_decisions': 30},
}


class TestPopulationWeightedRates:
    """RATE_ROWS samples one episode per cell, so the weights are the populations."""

    def _weighted(self) -> dict:
        return mod.population_weighted_rates(
            RATE_ROWS, _rate_verdicts(), WINDOW_POPULATION, window_days=10.0,
        )

    def test_each_cell_is_weighted_by_its_window_population(self) -> None:
        assert self._weighted()['weights'] == {
            'reify': {'ruling_lexeme': 10.0}, 'dark_factory': {'other_decisions': 30.0},
        }

    def test_the_weighted_rate_differs_from_the_sample_mean(self) -> None:
        weighted = self._weighted()
        assert weighted['overreach_rate_minted'] == mod.weighted_rate([(10.0, 2, 4), (30.0, 0, 4)])
        assert weighted['overreach_rate_minted']['rate'] == 0.125
        sample_mean = mod.adjudicated_rates(RATE_ROWS, _rate_verdicts())['all']
        assert sample_mean['overreach_rate_minted']['rate'] == 0.25

    def test_substantive_drops_bookkeeping_and_unjudgeable(self) -> None:
        assert self._weighted()['overreach_rate_substantive'] == mod.weighted_rate(
            [(10.0, 2, 3), (30.0, 0, 3)],
        )

    def test_window_volume_per_day(self) -> None:
        weighted = self._weighted()
        assert weighted['minted_per_day'] == 16.0
        assert weighted['overreach_per_day'] == 2.0


class TestDetectorCatch:
    def test_counts_adjudicated_overreach_by_the_detectors_on_its_episode(self) -> None:
        hits = {('reify', 'A'): frozenset({'batch_plan'}),
                ('dark_factory', 'B'): frozenset({'completion_claim'})}
        catch = mod.detector_catch(RATE_ROWS, _rate_verdicts(), hits)
        assert catch == {
            'overreach_edges': 2,
            'by_detector': {'unverified_claim_tag': 0, 'completion_claim': 0,
                            'proposed_resolution': 0, 'batch_plan': 2},
            'any_detector': 2,
            'any_wired': 0,
        }

    def test_a_wired_detector_on_the_episode_counts_as_wired(self) -> None:
        hits = {('reify', 'A'): frozenset({'unverified_claim_tag'})}
        catch = mod.detector_catch(RATE_ROWS, _rate_verdicts(), hits)
        assert catch['any_wired'] == 2


# --------------------------------------------------------------------------- #
# Reader, report and CLI end to end, against a graph double (step 13)
# --------------------------------------------------------------------------- #

_SKIP_LIMIT_RE = re.compile(r'SKIP\s+(\d+)\s+LIMIT\s+(\d+)', re.IGNORECASE)
_CENSUS_RE = re.compile(r'RETURN\s+count\((\*|r)\)\s*$', re.IGNORECASE)


class _FakeResult:
    def __init__(self, result_set: list[list]):
        self.result_set = result_set


class _FakeGraph:
    """Serves SKIP/LIMIT pages of an episode and an edge corpus, each with a census.

    The census follows FalkorDB: over an edge pattern, ``count(*)`` with ``r``
    unreferenced counts connected (source, target) PAIRS, so multi-edges
    collapse; ``count(r)`` counts edges. ``truncate_at`` makes the pages serve
    only that many rows of a corpus, as a short read would. ``query`` raises:
    the instrument may only ever issue ``ro_query``.
    """

    def __init__(self, episodes: list[list], edges: list[list], *,
                 census_override: dict[str, int] | None = None,
                 truncate_at: dict[str, int] | None = None, cap: int = 5):
        self.corpora = {'episodes': episodes, 'edges': edges}
        self.census_override = census_override or {}
        self.truncate_at = truncate_at or {}
        self.cap = cap
        self.queries: list[str] = []

    def _census(self, which: str, counted: str) -> int:
        corpus = self.corpora[which]
        if which == 'edges' and counted == '*':
            return len({(row[2], row[3]) for row in corpus})
        return len(corpus)

    async def ro_query(self, cypher: str, params: dict | None = None) -> _FakeResult:
        self.queries.append(cypher)
        which = 'episodes' if ':Episodic' in cypher else 'edges'
        if census := _CENSUS_RE.search(cypher.strip()):
            count = self.census_override.get(which, self._census(which, census.group(1)))
            return _FakeResult([[count]])
        match = _SKIP_LIMIT_RE.search(cypher)
        assert match, cypher
        skip, limit = int(match.group(1)), int(match.group(2))
        corpus = self.corpora[which][: self.truncate_at.get(which)]
        return _FakeResult(corpus[skip: skip + limit][: self.cap])

    async def query(self, cypher: str, params: dict | None = None):
        raise AssertionError('the instrument is read-only: it may never issue query()')


DECISION_SOURCE = 'add_memory:decisions_and_rationale'
EPISODE_ROWS = [
    ['E1', DECISION_SOURCE, '2026-09-01T10:00:00+00:00', RULING_HEAD],
    ['E2', DECISION_SOURCE, '2026-09-02T10:00:00+00:00', ANCHOR_HEAD],
    ['E3', DECISION_SOURCE, '2026-09-03T10:00:00+00:00', PLAIN_DECISION],
    ['E4', DECISION_SOURCE, '2026-08-01T10:00:00+00:00', RULING_HEAD],
    ['E5', 'add_memory:entities_and_relations', '2026-09-04T10:00:00+00:00', 'x uses y'],
]
EDGE_ROWS = [  # x1 and x2 are a multi-edge: both A -> B
    ['x1', 'holding fact', 'A', 'B', ['E1'], None, None],
    ['x2', 'overreach fact', 'A', 'B', ['E1', 'E2'], None, '2026-09-05T00:00:00+00:00'],
    ['x3', 'anchor fact', 'D', 'E', ['E2'], '2026-09-06T00:00:00+00:00', None],
    ['x4', 'plain fact', 'F', 'G', ['E3'], None, None],
    ['x5', 'orphan fact', 'H', 'I', [], None, None],
    ['x6', 'old fact', 'J', 'K', ['E4'], None, None],
]


def _graphs(**over):
    return {
        'dark_factory': _FakeGraph(EPISODE_ROWS, EDGE_ROWS, **over),
        'reify': _FakeGraph(EPISODE_ROWS[:2], EDGE_ROWS[:3]),
    }


def _factory(graphs):
    return lambda name: mod.GraphReader(
        graph=graphs[name], graph_name=name, page_size=2, resultset_size=5,
    )


def _run(*argv: str, graphs=None, reader_factory=None) -> int:
    return mod.run(['--graph', 'dark_factory', '--graph', 'reify', *argv],
                   reader_factory=reader_factory or _factory(graphs or _graphs()))


def _worksheet(tmp_path, graphs=None) -> list[dict]:
    path = tmp_path / 'worksheet.jsonl'
    assert _run('--emit-worksheet', str(path), graphs=graphs) == 0
    return [json.loads(line) for line in path.read_text().splitlines()]


def _verdicts_for(rows: list[dict], label: str = 'holding') -> dict:
    return {'sample': mod.SampleDefinition().to_dict(), 'verdicts': [
        {'graph': row['graph'], 'episode_uuid': row['uuid'], 'edge_uuid': m['edge_uuid'],
         'fact': m['fact'], 'label': label, 'rationale': 'quotes "the holding"'}
        for row in rows for m in row['minted']
    ]}


class TestGraphReader:
    @pytest.mark.asyncio
    async def test_reads_every_episode_across_pages(self) -> None:
        graph = _FakeGraph(EPISODE_ROWS, EDGE_ROWS)
        reader = mod.GraphReader(graph=graph, graph_name='dark_factory', page_size=2,
                                 resultset_size=5)
        episodes, read = await reader.fetch_episodes()
        assert [e.uuid for e in episodes] == ['E1', 'E2', 'E3', 'E4', 'E5']
        assert read.complete and read.rows_seen == 5
        assert episodes[0] == mod.Episode(
            graph='dark_factory', uuid='E1', created_at='2026-09-01T10:00:00+00:00',
            source=mod.parse_source_description(DECISION_SOURCE), content=RULING_HEAD,
        )
        assert sum(1 for q in graph.queries if _SKIP_LIMIT_RE.search(q)) >= 3

    @pytest.mark.asyncio
    async def test_reads_every_edge_live_or_not(self) -> None:
        graph = _FakeGraph(EPISODE_ROWS, EDGE_ROWS)
        reader = mod.GraphReader(graph=graph, graph_name='dark_factory', page_size=2,
                                 resultset_size=5)
        edges, read = await reader.fetch_edges()
        assert read.complete and read.rows_seen == 6
        assert edges[1] == mod.Edge(
            graph='dark_factory', uuid='x2', fact='overreach fact', source_name='A',
            target_name='B', episodes=('E1', 'E2'), invalid_at=None,
            expired_at='2026-09-05T00:00:00+00:00',
        )
        assert edges[2].served is False

    @pytest.mark.asyncio
    async def test_a_read_short_by_the_multi_edge_excess_is_incomplete(self) -> None:
        """A node-pair census would equal this short read and pass it as complete."""
        node_pairs = len({(row[2], row[3]) for row in EDGE_ROWS})
        assert node_pairs < len(EDGE_ROWS)
        graph = _FakeGraph(EPISODE_ROWS, EDGE_ROWS, truncate_at={'edges': node_pairs})
        reader = mod.GraphReader(graph=graph, graph_name='dark_factory', page_size=2,
                                 resultset_size=5)
        edges, read = await reader.fetch_edges()
        assert len(edges) == read.rows_seen == node_pairs
        assert read.complete is False
        assert read.expected_rows == len(EDGE_ROWS)

    def test_the_edge_read_never_projects_the_embedding_or_filters_liveness(self) -> None:
        assert 'fact_embedding' not in mod.EDGE_PAGE_CYPHER
        assert 'invalid_at IS NULL' not in mod.EDGE_PAGE_CYPHER
        assert 'ORDER BY r.uuid' in mod.EDGE_PAGE_CYPHER
        assert 'ORDER BY e.uuid' in mod.EPISODE_PAGE_CYPHER


class TestRun:
    def test_an_incomplete_read_exits_one_and_writes_nothing(self, tmp_path, capsys) -> None:
        graphs = _graphs(census_override={'edges': 99})
        worksheet, out_dir = tmp_path / 'ws.jsonl', tmp_path / 'out'
        code = _run('--emit-worksheet', str(worksheet), '--out-dir', str(out_dir), '--json',
                    graphs=graphs)
        assert code == 1
        assert capsys.readouterr().out == ''
        assert not worksheet.exists()
        assert not out_dir.exists()

    def test_a_failing_reader_exits_one_and_writes_nothing(self, tmp_path) -> None:
        def explode(name):
            raise RuntimeError('FalkorDB unreachable')

        out_dir = tmp_path / 'out'
        assert _run('--out-dir', str(out_dir), reader_factory=explode) == 1
        assert not out_dir.exists()

    def test_the_worksheet_is_the_sample_rows_as_jsonl(self, tmp_path) -> None:
        rows = _worksheet(tmp_path)
        assert [(r['graph'], r['uuid']) for r in rows] == [
            ('dark_factory', 'E1'), ('dark_factory', 'E2'), ('dark_factory', 'E3'),
            ('reify', 'E1'), ('reify', 'E2'),
        ]
        df_e1 = rows[0]
        assert [m['edge_uuid'] for m in df_e1['minted']] == ['x1', 'x2']
        assert rows[1]['corroborated_count'] == 1

    def test_a_verdict_run_writes_the_full_report(self, tmp_path) -> None:
        rows = _worksheet(tmp_path)
        verdict_path = tmp_path / 'verdicts.json'
        verdict_path.write_text(json.dumps(_verdicts_for(rows)))
        out_dir = tmp_path / 'out'
        assert _run('--verdicts', str(verdict_path), '--out-dir', str(out_dir)) == 0
        report = json.loads((out_dir / 'report.json').read_text())
        assert set(report) == {
            'swept_at', 'graphs', 'read_population', 'edge_population', 'classifiers',
            'specimens', 'strata', 'detector_census', 'wired_on_add_memory', 'sample',
            'adjudicated', 'prior_measurement', 'caveats',
        }
        adjudicated = report['adjudicated']
        assert adjudicated['rates']['all']['minted'] == 7
        assert adjudicated['rates']['all']['holding_share']['rate'] == 1.0
        assert set(adjudicated) >= {
            'rates', 'population_weighted', 'per_classifier', 'detector_catch', 'stale_verdicts',
        }
        weighted = adjudicated['population_weighted']
        assert weighted['overreach_rate_minted']['rate'] == 0.0
        assert weighted['minted_per_day'] == round(7 / report['sample']['window_days'], 1)
        assert report['read_population']['dark_factory']['edges']['complete'] is True
        assert report['sample']['unattributed_edges'] == 1
        assert report['prior_measurement']['overreach_minted']['ci'] == [0.0442, 0.1236]

    def test_the_edge_population_counts_multi_episode_and_expired_only_edges(
        self, tmp_path,
    ) -> None:
        out_dir = tmp_path / 'out'
        assert _run('--out-dir', str(out_dir)) == 0
        report = json.loads((out_dir / 'report.json').read_text())
        assert report['edge_population'] == {
            'dark_factory': {'edges': 6, 'multi_episode': 1, 'expired_only': 1},
            'reify': {'edges': 3, 'multi_episode': 1, 'expired_only': 1},
        }

    def test_the_detector_census_counts_any_and_wired_per_stratum(self, tmp_path) -> None:
        out_dir = tmp_path / 'out'
        assert _run('--out-dir', str(out_dir)) == 0
        census = json.loads((out_dir / 'report.json').read_text())['detector_census']
        assert set(census['dark_factory']) == set(mod.STRATA)
        assert set(census['dark_factory']['ruling_lexeme']) == {
            'episodes', *mod.DETECTORS, 'any_detector', 'any_wired',
        }
        assert census['dark_factory']['ruling_lexeme']['episodes'] == 2

    def test_without_verdicts_adjudicated_is_null(self, tmp_path) -> None:
        out_dir = tmp_path / 'out'
        assert _run('--out-dir', str(out_dir)) == 0
        assert json.loads((out_dir / 'report.json').read_text())['adjudicated'] is None

    def test_two_runs_are_byte_identical_but_for_swept_at(self, tmp_path) -> None:
        blobs = []
        for i in range(2):
            out_dir = tmp_path / f'out{i}'
            assert _run('--out-dir', str(out_dir)) == 0
            report = json.loads((out_dir / 'report.json').read_text())
            report.pop('swept_at')
            blobs.append(json.dumps(report, sort_keys=True))
        assert blobs[0] == blobs[1]

    def test_the_report_file_has_sorted_keys(self, tmp_path) -> None:
        out_dir = tmp_path / 'out'
        _run('--out-dir', str(out_dir))
        text = (out_dir / 'report.json').read_text()
        assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + '\n'

    def test_a_verdict_error_exits_two_and_writes_no_report(self, tmp_path) -> None:
        rows = _worksheet(tmp_path)
        bad = _verdicts_for(rows)
        bad['verdicts'].pop()
        verdict_path = tmp_path / 'verdicts.json'
        verdict_path.write_text(json.dumps(bad))
        out_dir = tmp_path / 'out'
        assert _run('--verdicts', str(verdict_path), '--out-dir', str(out_dir)) == 2
        assert not out_dir.exists()

    def test_an_unreadable_verdict_file_exits_two(self, tmp_path) -> None:
        verdict_path = tmp_path / 'verdicts.json'
        verdict_path.write_text('{not json')
        assert _run('--verdicts', str(verdict_path)) == 2

    def test_the_cli_offers_no_mutation_flag(self, capsys) -> None:
        with pytest.raises(SystemExit) as exited:
            mod.run(['--help'])
        assert exited.value.code == 0
        flags = set(re.findall(r'--[a-z][a-z-]*', capsys.readouterr().out))
        assert {'--graph', '--verdicts', '--out-dir'} <= flags
        assert not flags & {'--apply', '--invalidate', '--delete', '--repair', '--quarantine'}
