"""Tests for shared.backends: the backend vocabulary and its one decoder."""

from __future__ import annotations

import pytest

from shared.backends import (
    AGENT_CLI_BACKENDS,
    SDK_BACKENDS,
    Backend,
    UnknownBackend,
    parse_backend,
)


def test_legacy_claude_cli_alias_decodes_to_claude() -> None:
    assert parse_backend('claude_cli') is Backend.CLAUDE


@pytest.mark.parametrize('member', list(Backend))
def test_every_member_round_trips_by_value_and_by_member(member: Backend) -> None:
    assert parse_backend(member.value) is member
    assert parse_backend(member) is member


def test_zcode_and_mimo_decode_to_their_members() -> None:
    assert parse_backend('zcode') is Backend.ZCODE
    assert parse_backend('mimo') is Backend.MIMO


def test_canonical_wire_spellings() -> None:
    assert {b.value for b in Backend} == {
        'claude',
        'codex',
        'gemini',
        'pi',
        'zcode',
        'mimo',
        'anthropic',
        'openai',
    }


def test_unknown_backend_carries_value_and_allowed_values() -> None:
    with pytest.raises(UnknownBackend) as caught:
        parse_backend('Claude')
    exc = caught.value
    assert exc.value == 'Claude'
    assert exc.allowed == tuple(b.value for b in Backend)
    message = str(exc)
    assert 'Claude' in message
    for allowed in exc.allowed:
        assert allowed in message


@pytest.mark.parametrize('value', ['claude-cli', '', ' claude', 'CLAUDE_CLI'])
def test_near_miss_spellings_are_rejected(value: str) -> None:
    with pytest.raises(UnknownBackend) as caught:
        parse_backend(value)
    assert caught.value.value == value
    assert repr(value) in str(caught.value)


@pytest.mark.parametrize('value', [None, 1])
def test_non_str_input_raises_unknown_backend(value: object) -> None:
    with pytest.raises(UnknownBackend) as caught:
        parse_backend(value)
    assert caught.value.value is value


def test_unknown_backend_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        parse_backend('nope')


def test_agent_cli_backends_are_the_six_dispatch_arms() -> None:
    assert {b.value for b in AGENT_CLI_BACKENDS} == {
        'claude',
        'codex',
        'gemini',
        'pi',
        'zcode',
        'mimo',
    }


def test_sdk_backends_are_the_in_process_sdks() -> None:
    assert SDK_BACKENDS == {Backend.ANTHROPIC_API, Backend.OPENAI_API}


def test_every_backend_is_exactly_one_kind() -> None:
    assert AGENT_CLI_BACKENDS.isdisjoint(SDK_BACKENDS)
    assert AGENT_CLI_BACKENDS | SDK_BACKENDS == set(Backend)
