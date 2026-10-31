"""Which environment variables carry secrets, and which an agent child may inherit.

The names below are the external contract, so every one is spelled as a literal
here rather than imported from ``shared.secret_env``.
"""

from __future__ import annotations

import pytest

from shared.secret_env import agent_subprocess_env, is_secret_env_name, secret_env_values

SECRET_NAMES = (
    'CLAUDE_OAUTH_TOKEN_A',
    'CLAUDE_OAUTH_TOKEN_H',
    'CLAUDE_OAUTH_TOKEN_MAX_B',
    'CLAUDE_CODE_OAUTH_TOKEN',
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_OAUTH_TOKEN',
    'OPENAI_API_KEY',
    'GEMINI_API_KEY',
    'PERPLEXITY_API_KEY',
    'RUNPOD_API_KEY',
    'TYPESAFE_API_KEY',
    'HF_DOWNLOAD_TOKEN',
)

ORDINARY_NAMES = (
    'PATH',
    'HOME',
    'CLAUDE_CONFIG_DIR',
    'CLAUDE_SPAWN_ROLE',
    'MCP_TIMEOUT',
    'CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR',
    'ANTHROPIC_BASE_URL',
    'MEM0_TELEMETRY',
    'SSH_AUTH_SOCK',
    'DF_AGENT_CPU_NICE',
)

POOL = {f'CLAUDE_OAUTH_TOKEN_{letter}': f'pool-{letter}-' + 'p' * 40 for letter in 'ABCDEFGH'}

ORDINARY_ENV = {
    'PATH': '/usr/bin:/bin',
    'HOME': '/home/someone',
    'CLAUDE_CONFIG_DIR': '/tmp/claude-config',
    'MCP_TIMEOUT': '30000',
}


def _environ_with_secrets() -> dict[str, str]:
    return {
        **POOL,
        'OPENAI_API_KEY': 'openai-' + 'o' * 40,
        'ANTHROPIC_API_KEY': 'anthropic-' + 'a' * 40,
        'GEMINI_API_KEY': 'gemini-' + 'g' * 40,
        'CLAUDE_CODE_OAUTH_TOKEN': 'session-' + 's' * 40,
        **ORDINARY_ENV,
    }


@pytest.mark.parametrize('name', SECRET_NAMES)
def test_secret_names_are_secret(name):
    assert is_secret_env_name(name) is True


@pytest.mark.parametrize('name', ORDINARY_NAMES)
def test_ordinary_names_are_not_secret(name):
    assert is_secret_env_name(name) is False


class TestAgentSubprocessEnv:
    def test_returns_a_new_dict(self):
        environ = _environ_with_secrets()
        child = agent_subprocess_env(environ)
        assert isinstance(child, dict)
        assert child is not environ

    def test_drops_the_pool_and_every_api_key(self):
        child = agent_subprocess_env(_environ_with_secrets())
        assert [k for k in child if k.startswith('CLAUDE_OAUTH_TOKEN_')] == []
        for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GEMINI_API_KEY'):
            assert name not in child

    def test_keeps_the_session_credential_and_ordinary_vars_unchanged(self):
        environ = _environ_with_secrets()
        child = agent_subprocess_env(environ)
        assert child['CLAUDE_CODE_OAUTH_TOKEN'] == environ['CLAUDE_CODE_OAUTH_TOKEN']
        for name, value in ORDINARY_ENV.items():
            assert child[name] == value

    def test_leaves_the_input_unmutated(self):
        environ = _environ_with_secrets()
        snapshot = dict(environ)
        agent_subprocess_env(environ)
        assert environ == snapshot

    def test_the_only_secret_bearing_key_is_the_session_credential(self):
        child = agent_subprocess_env(_environ_with_secrets())
        assert {k for k in child if is_secret_env_name(k)} == {'CLAUDE_CODE_OAUTH_TOKEN'}

    def test_drops_python_env_selection(self):
        environ = {
            'VIRTUAL_ENV': '/srv/orch/.venv',
            'UV_PROJECT_ENVIRONMENT': '/srv/orch/.venv',
            'UV_RUN_RECURSION_DEPTH': '1',
            'PATH': '/srv/orch/.venv/bin:/usr/bin:/bin',
            'CLAUDE_CODE_OAUTH_TOKEN': 'session-' + 's' * 40,
            'CLAUDE_OAUTH_TOKEN_A': POOL['CLAUDE_OAUTH_TOKEN_A'],
        }
        child = agent_subprocess_env(environ)
        for name in ('VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT', 'UV_RUN_RECURSION_DEPTH'):
            assert name not in child
        assert child['PATH'] == '/usr/bin:/bin'
        assert child['CLAUDE_CODE_OAUTH_TOKEN'] == environ['CLAUDE_CODE_OAUTH_TOKEN']
        assert 'CLAUDE_OAUTH_TOKEN_A' not in child


class TestSecretEnvValues:
    def test_maps_every_secret_name_with_a_value(self):
        environ = _environ_with_secrets()
        expected = {k: v for k, v in environ.items() if k not in ORDINARY_ENV}
        assert secret_env_values(environ) == expected
        assert 'CLAUDE_CODE_OAUTH_TOKEN' in secret_env_values(environ)

    def test_omits_empty_values(self):
        environ = {**_environ_with_secrets(), 'RUNPOD_API_KEY': ''}
        assert 'RUNPOD_API_KEY' not in secret_env_values(environ)

    def test_no_secrets_yields_nothing(self):
        assert secret_env_values(ORDINARY_ENV) == {}
