"""Tests for the ensure_entity_node authorization resolver (task 4932).

Structural sibling of ``tests/server/test_update_memory_authz_gate.py``, which
gates the in-place ``update_memory`` tool. Same posture and the same deliberate
inversion of ``server/near_duplicate_guard.py``'s permissive fallbacks: those
resolvers fail OPEN because that is the safe direction for a soft-block guard,
whereas this is a MUTATION-AUTHORIZATION gate whose safe direction is DENY.

Minting an Entity node is a write-time-IDENTITY primitive: a node minted under a
non-canonical name SPLITS a referent rather than resolving it, and nothing
sweeps orphan minted nodes. So the resolver lives in its own module rather than
inline in ``server/tools.py``'s registration closure — a guard body buried there
is neither importable nor independently testable, and ``config/reload.py``'s
reload-safety rule requires a test that proves the consumer re-reads config live
before its leaf may be registered green-tier.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fused_memory.config.schema import EntityMintConfig, FusedMemoryConfig
from fused_memory.server.entity_mint_authz import resolve_entity_mint_authorization


def _service(**entity_mint_kwargs):
    """A minimal stand-in for MemoryService carrying a real config object.

    ``entity_mint`` is ALWAYS rebuilt from an explicit ``EntityMintConfig`` so
    these tests pin schema defaults deterministically — a bare
    ``FusedMemoryConfig()`` is a BaseSettings that loads ``config/config.yaml``
    from the test cwd, and inheriting whatever that file says is how the
    sibling module's TestLiveRead became a 65b011ed8c tripwire casualty."""
    config = FusedMemoryConfig()
    config.entity_mint = EntityMintConfig(**entity_mint_kwargs)
    return SimpleNamespace(config=config)


class TestLiveRead:
    """config/reload.py's precondition for registering the five leaves."""

    def test_reads_allowlist_live_on_every_call(self):
        # 'stranger-' as the not-on-the-bar example, NOT 'curator-', which IS on
        # the shipped default bar — the sibling module learned that the hard way
        # when the esc-3524-1 grant put curator- on the bar and broke its first
        # assertion.
        svc = _service()
        first = resolve_entity_mint_authorization(svc, agent_id='stranger-session')
        assert first.allowed is False, 'stranger- is not on the default bar'

        # Mutate the SHARED config object in place, exactly as reload_config does.
        svc.config.entity_mint.allowed_agent_prefixes.append('stranger-')

        second = resolve_entity_mint_authorization(svc, agent_id='stranger-session')
        assert second.allowed is True, (
            'the resolver must re-read config on every call; a value captured at '
            'import or construction would make the leaf restart-only in disguise'
        )

    def test_kill_switch_flipped_in_place_takes_effect(self):
        svc = _service()
        assert resolve_entity_mint_authorization(
            svc, agent_id='recon-stage-1',
        ).allowed is True

        svc.config.entity_mint.enabled = False

        assert resolve_entity_mint_authorization(
            svc, agent_id='recon-stage-1',
        ).allowed is False, (
            'a restart-only kill switch is no kill switch — flipping enabled in '
            'place must deny the very next call'
        )


class TestFailsClosed:
    """Unlike the near-dup guard, a missing/corrupt leaf must DENY."""

    @pytest.mark.parametrize('svc', [
        SimpleNamespace(),                                  # no config at all
        SimpleNamespace(config=SimpleNamespace()),          # no entity_mint section
        SimpleNamespace(config=SimpleNamespace(entity_mint=None)),
        SimpleNamespace(config=None),
    ])
    def test_missing_config_hop_denies(self, svc):
        decision = resolve_entity_mint_authorization(svc, agent_id='recon-stage-1')
        assert decision.allowed is False, (
            'a mutation-authorization gate must fail CLOSED, not open'
        )

    def test_unspecced_mock_denies_and_does_not_raise(self):
        """An unspecced Mock auto-generates every attribute, so the leaf arrives
        as a Mock rather than a list — strict type checks must reject it."""
        decision = resolve_entity_mint_authorization(
            MagicMock(), agent_id='recon-stage-1',
        )
        assert decision.allowed is False

    @pytest.mark.parametrize('corrupt', ['recon-stage-', 42, None, {'a': 1}])
    def test_non_list_allowlist_denies(self, corrupt):
        """A bare STRING is the load-bearing case: ``'x'.startswith`` still works
        against a str, so a naive implementation would silently treat the whole
        string as one prefix — a mis-typed config value that reads as working
        while gating on something the operator never wrote."""
        svc = _service()
        object.__setattr__(svc.config.entity_mint, 'allowed_agent_prefixes', corrupt)
        assert resolve_entity_mint_authorization(
            svc, agent_id='recon-stage-1',
        ).allowed is False

    def test_non_bool_enabled_denies(self):
        svc = _service()
        object.__setattr__(svc.config.entity_mint, 'enabled', 'yes')
        assert resolve_entity_mint_authorization(
            svc, agent_id='recon-stage-1',
        ).allowed is False


class TestKillSwitchOutranksAgentId:
    def test_disabled_denies_every_caller_with_named_reason(self):
        svc = _service(enabled=False)
        for agent_id in ('recon-stage-memory_consolidator', 'curator-gate', None):
            decision = resolve_entity_mint_authorization(svc, agent_id=agent_id)
            assert decision.allowed is False
            assert decision.error_type == 'EntityMintToolDisabled', (
                f'expected EntityMintToolDisabled, got {decision.error_type!r}'
            )

    def test_disabled_outranks_an_allowlisted_agent(self):
        """Proves the switch is evaluated FIRST: an on-the-bar agent_id would
        otherwise pass, so the disabled error_type is only reachable if the
        kill switch runs before the prefix check."""
        svc = _service(enabled=False)
        decision = resolve_entity_mint_authorization(svc, agent_id='recon-stage-1')
        assert decision.allowed is False
        assert decision.error_type == 'EntityMintToolDisabled'


class TestAgentIdTypes:
    @pytest.mark.parametrize('agent_id', [None, '', 42, b'recon-stage-1',
                                          ['recon-stage-1']])
    def test_non_str_or_empty_agent_id_denies(self, agent_id):
        svc = _service()
        decision = resolve_entity_mint_authorization(svc, agent_id=agent_id)
        assert decision.allowed is False, f'{agent_id!r} must not pass the gate'

    def test_empty_allowlist_denies_everyone(self):
        svc = _service(allowed_agent_prefixes=[])
        assert resolve_entity_mint_authorization(
            svc, agent_id='recon-stage-1',
        ).allowed is False

    def test_allowlisted_prefixes_pass_out_of_the_box(self):
        """The task's stated minimum bar: no operator config required."""
        svc = _service()
        for agent_id in ('recon-stage-memory_consolidator', 'curator-repair'):
            assert resolve_entity_mint_authorization(
                svc, agent_id=agent_id,
            ).allowed is True, f'{agent_id!r} is on the shipped default bar'


class TestDecisionShape:
    def test_denial_is_a_value_carrying_a_structured_reason(self):
        svc = _service()
        decision = resolve_entity_mint_authorization(svc, agent_id='rando')
        assert decision.allowed is False
        assert isinstance(decision.error_type, str) and decision.error_type
        assert isinstance(decision.error, str) and decision.error, (
            'the deny reason must be a caller-facing message, so the tool can '
            'return a structured rejection rather than raising (INV-1)'
        )

    def test_allowed_decision_has_no_error(self):
        svc = _service()
        decision = resolve_entity_mint_authorization(svc, agent_id='recon-stage-1')
        assert decision.allowed is True
        assert decision.error_type is None
        assert decision.error is None

    def test_denial_names_agent_required_prefixes_and_the_config_knob(self):
        """An operator reading the refusal must learn what to widen, and the
        caller must learn which bar it failed — so the message names the
        offending agent_id, the authorized prefixes, and the knob."""
        svc = _service()
        decision = resolve_entity_mint_authorization(svc, agent_id='rando')
        assert decision.error_type == 'EntityMintNotAuthorized', decision
        message = decision.error
        assert isinstance(message, str)
        assert 'rando' in message, (
            f'the message must name the rejected agent_id, got {message!r}'
        )
        assert 'recon-stage-' in message and 'curator-' in message, (
            f'the message must name the required prefixes, got {message!r}'
        )
        assert 'entity_mint.allowed_agent_prefixes' in message, (
            f'the message must name the config knob, got {message!r}'
        )
