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
