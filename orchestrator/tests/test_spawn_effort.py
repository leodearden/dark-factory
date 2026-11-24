"""Tests for orchestrator.spawn_effort against T3 Code's ``modelSelection`` and ``orchestrator_capabilities`` shapes.

The capabilities reply below is what the live T3 server (0.1.10) answered a
``search`` for ``fable`` with on 2026-11-24, abridged to two providers.
"""

from __future__ import annotations

from typing import Any

import pytest

from orchestrator.spawn_effort import (
    EffortLevel,
    effort_option,
    in_level_order,
    offered_levels,
    selection_effort,
)

FABLE_LEVELS = ['low', 'medium', 'high', 'xhigh', 'max', 'ultrathink']


def fable(model_id: str, levels: list[Any] = FABLE_LEVELS) -> dict[str, Any]:
    return {
        'id': model_id,
        'options': [
            {'id': 'effort', 'values': levels, 'default': 'medium'},
            {'id': 'ultracode', 'values': [False, True]},
        ],
    }


def capabilities(*providers: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        'providers': [
            {'providerInstanceId': instance_id, 'driverKind': 'claudeAgent', 'models': models}
            for instance_id, models in providers
        ],
    }


LIVE_FABLE_SEARCH = capabilities(
    ('claudeAgent', [fable('claude-fable-5-2')]),
    ('claude-proxy', [fable('claude-fable-5-1'), fable('claude-fable-5-2')]),
)


class TestInLevelOrder:
    def test_levels_run_low_to_max(self) -> None:
        assert in_level_order(['max', 'low', 'xhigh', 'medium', 'high']) == [level.value for level in EffortLevel]

    def test_levels_outside_the_closed_set_come_last_by_name(self) -> None:
        assert in_level_order(['ultrathink', 'max', 'turbo', 'low']) == ['low', 'max', 'turbo', 'ultrathink']


class TestEffortOption:
    @pytest.mark.parametrize('level', list(EffortLevel))
    def test_a_level_is_the_effort_entry_of_a_model_selections_options(self, level: EffortLevel) -> None:
        assert effort_option(level) == {'id': 'effort', 'value': level.value}


class TestSelectionEffort:
    def test_the_effort_entry_among_other_options_names_the_effort(self) -> None:
        selection = {
            'instanceId': 'claude-proxy',
            'model': 'claude-fable-5-2',
            'options': [{'id': 'fastMode', 'value': False}, {'id': 'effort', 'value': 'xhigh'}],
        }

        assert selection_effort(selection) == 'xhigh'

    @pytest.mark.parametrize(
        'selection',
        [
            None,
            ['effort', 'xhigh'],
            {'instanceId': 'claude-proxy', 'model': 'claude-fable-5-2'},
            {'options': {'effort': 'xhigh'}},
            {'options': ['effort', {'id': 'fastMode', 'value': True}]},
            {'options': [{'id': 'effort', 'value': None}]},
        ],
        ids=['no-selection', 'not-a-mapping', 'no-options', 'options-not-a-list', 'no-effort-entry', 'value-not-text'],
    )
    def test_a_selection_naming_no_effort_answers_none(self, selection: object) -> None:
        assert selection_effort(selection) is None


class TestOfferedLevels:
    def test_a_model_listed_by_id_answers_its_effort_values(self) -> None:
        assert offered_levels(LIVE_FABLE_SEARCH, 'claude-proxy', 'claude-fable-5-1') == frozenset(FABLE_LEVELS)

    def test_an_alias_the_search_matched_to_one_model_answers_that_models_values(self) -> None:
        assert offered_levels(LIVE_FABLE_SEARCH, 'claudeAgent', 'fable') == frozenset(FABLE_LEVELS)

    def test_the_values_are_those_of_the_instance_asked_about(self) -> None:
        listed = capabilities(
            ('claudeAgent', [fable('claude-haiku-4-5', ['low'])]),
            ('claude-proxy', [fable('claude-haiku-4-5', ['low', 'high'])]),
        )

        assert offered_levels(listed, 'claude-proxy', 'claude-haiku-4-5') == frozenset({'low', 'high'})

    @pytest.mark.parametrize(
        ('listed', 'instance_id', 'model'),
        [
            (LIVE_FABLE_SEARCH, 'claude-proxy', 'fable'),
            (LIVE_FABLE_SEARCH, 'codex', 'claude-fable-5-2'),
            (capabilities(('claude-proxy', [])), 'claude-proxy', 'claude-fable-5-2'),
            (
                capabilities(('claude-proxy', [{'id': 'claude-fable-5-2', 'options': [{'id': 'fastMode', 'values': [True]}]}])),
                'claude-proxy',
                'claude-fable-5-2',
            ),
            (capabilities(('claude-proxy', [fable('claude-fable-5-2', [])])), 'claude-proxy', 'claude-fable-5-2'),
            (capabilities(('claude-proxy', [fable('claude-fable-5-2', [True, 3])])), 'claude-proxy', 'claude-fable-5-2'),
            ({'providers': {'claude-proxy': []}}, 'claude-proxy', 'claude-fable-5-2'),
            ({'providers': [{'providerInstanceId': 'claude-proxy', 'models': 'claude-fable-5-2'}]}, 'claude-proxy', 'claude-fable-5-2'),
            (None, 'claude-proxy', 'claude-fable-5-2'),
        ],
        ids=[
            'alias-ambiguous',
            'instance-unlisted',
            'model-unlisted',
            'no-effort-option',
            'no-values',
            'values-not-text',
            'providers-not-a-list',
            'models-not-a-list',
            'no-reply',
        ],
    )
    def test_a_catalogue_naming_no_levels_for_the_model_answers_none(
        self, listed: object, instance_id: str, model: str
    ) -> None:
        assert offered_levels(listed, instance_id, model) is None
