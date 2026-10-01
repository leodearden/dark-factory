"""Tests for scripts/audit_ruling_overreach.py (task 4716, esc-4639-1).

The script is loaded by path, as ``tests/test_audit_wrong_binding_edges.py``
loads its sibling, so ``scripts/`` never lands on ``sys.path``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from _fm_helpers import load_script_module

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
