"""Tests for orchestrator.spawn_t3_provider against settings.json shapes of t3code v0.1.6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from orchestrator.spawn_t3_provider import ProviderChoice, ProviderSource, resolve_instance

PROJECT_ID = '5a05e1d3-5bd0-40e5-8ce7-24b2cd160300'

PROXY_SETTINGS: dict[str, Any] = {
    'defaultModelSelection': {'instanceId': 'claude-proxy', 'model': 'claude-opus-5-5'},
    'projectSettingsFolded': True,
    'providerInstances': {
        'claudeAgent': {'driver': 'claudeAgent', 'enabled': False, 'config': {}},
        'claude-proxy': {'driver': 'claudeAgent', 'enabled': True, 'config': {}},
    },
}
"""The dark-factory host's settings since the 11-01 CLIProxyAPI evaluation, abridged."""


def userdata(tmp_path: Path, settings: object) -> Path:
    (tmp_path / 'settings.json').write_text(json.dumps(settings))
    return tmp_path


def test_the_projects_default_claude_instance_wins_over_the_fallback(tmp_path: Path) -> None:
    choice = resolve_instance(userdata(tmp_path, PROXY_SETTINGS), PROJECT_ID, None)

    assert choice == ProviderChoice('claude-proxy', ProviderSource.PROJECT, True)


def test_an_explicit_instance_wins_and_is_checked(tmp_path: Path) -> None:
    root = userdata(tmp_path, PROXY_SETTINGS)

    assert resolve_instance(root, PROJECT_ID, 'claudeAgent') == ProviderChoice(
        'claudeAgent', ProviderSource.EXPLICIT, False
    )


def test_a_project_override_beats_the_environment_default(tmp_path: Path) -> None:
    settings = {
        **PROXY_SETTINGS,
        'providerInstances': {
            **PROXY_SETTINGS['providerInstances'],
            'claude-b': {'driver': 'claudeAgent'},
        },
        'projectSettingsOverrides': {
            PROJECT_ID: {
                'defaultModelSelection': {'instanceId': 'claude-b', 'model': 'm'},
                'providerInstances': {'claude-proxy': {'enabled': False}},
            }
        },
    }
    root = userdata(tmp_path, settings)

    assert resolve_instance(root, PROJECT_ID, None).instance_id == 'claude-b'
    assert resolve_instance(root, PROJECT_ID, 'claude-proxy').enabled is False
    assert resolve_instance(root, 'another-project', 'claude-proxy').enabled is True


def test_a_non_claude_default_falls_to_the_only_enabled_claude_instance(tmp_path: Path) -> None:
    settings = {
        'defaultModelSelection': {'instanceId': 'codex', 'model': 'gpt'},
        'providerInstances': {
            'claudeAgent': {'driver': 'claudeAgent', 'config': {'enabled': False}},
            'claude-proxy': {'driver': 'claudeAgent'},
        },
    }

    choice = resolve_instance(userdata(tmp_path, settings), PROJECT_ID, None)

    assert choice == ProviderChoice('claude-proxy', ProviderSource.PROJECT, True)


def test_two_enabled_claude_instances_and_no_claude_default_fall_back(tmp_path: Path) -> None:
    settings = {'providerInstances': {'claude-proxy': {'driver': 'claudeAgent'}}}

    choice = resolve_instance(userdata(tmp_path, settings), PROJECT_ID, None)

    assert choice == ProviderChoice('claudeAgent', ProviderSource.FALLBACK, True)


def test_no_enabled_claude_instance_falls_back_to_a_disabled_claude_agent(tmp_path: Path) -> None:
    settings = {'providerInstances': {'claudeAgent': {'driver': 'claudeAgent', 'enabled': False}}}

    choice = resolve_instance(userdata(tmp_path, settings), PROJECT_ID, None)

    assert choice == ProviderChoice('claudeAgent', ProviderSource.FALLBACK, False)


@pytest.mark.parametrize(
    'content', [None, 'not json', '[1]', json.dumps({'providers': {'claudeAgent': {'enabled': False}}})]
)
def test_unreadable_settings_verify_nothing(tmp_path: Path, content: str | None) -> None:
    if content is not None:
        (tmp_path / 'settings.json').write_text(content)

    assert resolve_instance(tmp_path, PROJECT_ID, None) == ProviderChoice(
        'claudeAgent', ProviderSource.FALLBACK, None
    )
    assert resolve_instance(tmp_path, PROJECT_ID, 'x') == ProviderChoice(
        'x', ProviderSource.EXPLICIT, None
    )


@pytest.mark.parametrize(
    ('instance', 'enabled'),
    [
        ({'driver': 'cursor'}, False),
        ({'driver': 'cursor', 'enabled': True}, True),
        ({'driver': 'claudeAgent', 'enabled': True, 'config': {'enabled': False}}, False),
        ({'driver': 'claudeAgent', 'enabled': 0}, True),
    ],
)
def test_instance_enablement_follows_t3s_rules(
    tmp_path: Path, instance: dict[str, Any], enabled: bool
) -> None:
    root = userdata(tmp_path, {'providerInstances': {'i': instance}})

    assert resolve_instance(root, None, 'i').enabled is enabled


def test_an_unconfigured_instance_is_enabled_only_when_t3_builds_it_in(tmp_path: Path) -> None:
    root = userdata(tmp_path, {})

    assert resolve_instance(root, None, 'claudeAgent').enabled is True
    assert resolve_instance(root, None, 'claude-proxy').enabled is False
