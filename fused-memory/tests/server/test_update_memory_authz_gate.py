"""Tests for the in-place update_memory authorization resolver (task 3088).

Modelled on the live-read resolver posture of ``server/near_duplicate_guard.py``
(``resolve_near_dup_guard_enabled`` et al) with ONE deliberate inversion: those
resolvers fall back to PERMISSIVE defaults because fail-open is the safe
direction for a soft-block guard. This is a MUTATION-AUTHORIZATION gate, so the
safe direction is DENY.

Lives in its own module rather than inline in the 5300-line ``server/tools.py``
registration closure so these tests can call the resolver DIRECTLY — a guard
body buried in that closure is neither importable nor independently testable,
and ``config/reload.py``'s reload-safety rule requires a test that proves the
consumer re-reads config live before its leaf may be registered.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from fused_memory.config.schema import FusedMemoryConfig, Mem0UpdateConfig
from fused_memory.server.mem0_update_authz import resolve_mem0_update_authorization


def _service(**mem0_update_kwargs):
    """A minimal stand-in for MemoryService carrying a real config object."""
    config = FusedMemoryConfig()
    if mem0_update_kwargs:
        config.mem0_update = Mem0UpdateConfig(**mem0_update_kwargs)
    return SimpleNamespace(config=config)


class TestLiveRead:
    """config/reload.py's precondition for registering the five leaves."""

    def test_reads_config_live_on_every_call(self):
        svc = _service()
        first = resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=True, metadata_patch=False,
        )
        assert first.allowed is False, 'curator-gate is not on the default bar'

        # Mutate the SHARED config object in place, exactly as reload_config does.
        svc.config.mem0_update.content_amend_allowed_agent_prefixes.append('curator-')

        second = resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=True, metadata_patch=False,
        )
        assert second.allowed is True, (
            'the resolver must re-read config on every call; a value captured at '
            'import or construction would make the leaf restart-only in disguise'
        )

    def test_kill_switch_flipped_in_place_takes_effect(self):
        svc = _service()
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is True

        svc.config.mem0_update.enabled = False

        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is False


class TestFailsClosed:
    """Unlike the near-dup guard, a missing/corrupt leaf must DENY."""

    @pytest.mark.parametrize('svc', [
        SimpleNamespace(),                                  # no config at all
        SimpleNamespace(config=SimpleNamespace()),          # no mem0_update section
        SimpleNamespace(config=SimpleNamespace(mem0_update=None)),
        SimpleNamespace(config=None),
    ])
    def test_missing_config_hop_denies(self, svc):
        decision = resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        )
        assert decision.allowed is False, (
            'a mutation-authorization gate must fail CLOSED, not open'
        )

    def test_unspecced_mock_denies_and_does_not_raise(self):
        """An unspecced Mock auto-generates every attribute, so the leaf arrives
        as a Mock rather than a list — strict type checks must reject it."""
        decision = resolve_mem0_update_authorization(
            MagicMock(), agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        )
        assert decision.allowed is False

    @pytest.mark.parametrize('corrupt', ['recon-stage-', 42, None, {'a': 1}])
    def test_non_list_allowlist_denies(self, corrupt):
        """A bare string is the dangerous case: 'x'.startswith would still work
        against a str, so a naive implementation would silently treat the whole
        string as one prefix."""
        svc = _service()
        object.__setattr__(
            svc.config.mem0_update, 'content_amend_allowed_agent_prefixes', corrupt,
        )
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is False

    def test_non_bool_enabled_denies(self):
        svc = _service()
        object.__setattr__(svc.config.mem0_update, 'enabled', 'yes')
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is False


class TestKillSwitch:
    def test_disabled_denies_every_caller_with_named_reason(self):
        svc = _service(enabled=False)
        for agent_id in ('recon-stage-memory_consolidator', 'anyone-else', None):
            decision = resolve_mem0_update_authorization(
                svc, agent_id=agent_id, content_amend=False, metadata_patch=True,
            )
            assert decision.allowed is False
            assert decision.error_type == 'Mem0UpdateToolDisabled', (
                f'expected Mem0UpdateToolDisabled, got {decision.error_type!r}'
            )

    def test_disabled_outranks_an_authorized_agent(self):
        svc = _service(enabled=False)
        decision = resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=True,
        )
        assert decision.allowed is False
        assert decision.error_type == 'Mem0UpdateToolDisabled'


class TestPerArmDecisions:
    def test_content_amend_checks_only_the_content_list(self):
        svc = _service(
            content_amend_allowed_agent_prefixes=['recon-stage-'],
            metadata_patch_allowed_agent_prefixes=[],
        )
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is True

    def test_metadata_patch_checks_only_the_metadata_list(self):
        svc = _service(
            content_amend_allowed_agent_prefixes=[],
            metadata_patch_allowed_agent_prefixes=['curator-'],
        )
        assert resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=False, metadata_patch=True,
        ).allowed is True

    def test_metadata_bar_alone_does_not_grant_content_amend(self):
        """The concrete operator story the two decoupled lists exist for."""
        svc = _service(
            content_amend_allowed_agent_prefixes=['recon-stage-'],
            metadata_patch_allowed_agent_prefixes=['recon-stage-', 'curator-'],
        )
        assert resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=False, metadata_patch=True,
        ).allowed is True
        assert resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=True, metadata_patch=False,
        ).allowed is False

    def test_both_arms_must_pass_both_lists(self):
        svc = _service(
            content_amend_allowed_agent_prefixes=['recon-stage-'],
            metadata_patch_allowed_agent_prefixes=['curator-'],
        )
        # Authorized for content but not metadata -> denied.
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=True,
        ).allowed is False
        # Authorized for metadata but not content -> denied.
        assert resolve_mem0_update_authorization(
            svc, agent_id='curator-gate', content_amend=True, metadata_patch=True,
        ).allowed is False

    def test_both_arms_allowed_when_on_both_lists(self):
        svc = _service()
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-memory_consolidator',
            content_amend=True, metadata_patch=True,
        ).allowed is True

    def test_no_arm_requested_denies(self):
        """Nothing to authorize is not a licence; arm validation is a separate
        layer and must not be reachable via an empty authorization request."""
        svc = _service()
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=False, metadata_patch=False,
        ).allowed is False


class TestAgentIdTypes:
    @pytest.mark.parametrize('agent_id', [None, 42, b'recon-stage-1', ['recon-stage-1'], ''])
    def test_non_str_or_empty_agent_id_denies(self, agent_id):
        svc = _service()
        decision = resolve_mem0_update_authorization(
            svc, agent_id=agent_id, content_amend=True, metadata_patch=False,
        )
        assert decision.allowed is False, f'{agent_id!r} must not pass the gate'

    def test_empty_allowlist_denies_everyone(self):
        svc = _service(content_amend_allowed_agent_prefixes=[])
        assert resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        ).allowed is False


class TestDecisionShape:
    def test_denial_carries_a_structured_reason_and_never_raises(self):
        svc = _service()
        decision = resolve_mem0_update_authorization(
            svc, agent_id='rando', content_amend=True, metadata_patch=False,
        )
        assert decision.allowed is False
        assert isinstance(decision.error_type, str) and decision.error_type
        assert isinstance(decision.error, str) and decision.error, (
            'the deny reason must be a caller-facing message, so the tool can '
            'return a structured rejection rather than raising (INV-1)'
        )

    def test_allowed_decision_has_no_error(self):
        svc = _service()
        decision = resolve_mem0_update_authorization(
            svc, agent_id='recon-stage-1', content_amend=True, metadata_patch=False,
        )
        assert decision.allowed is True
        assert decision.error_type is None
        assert decision.error is None

    def test_denial_names_the_offending_agent_and_arm(self):
        svc = _service()
        decision = resolve_mem0_update_authorization(
            svc, agent_id='rando', content_amend=True, metadata_patch=False,
        )
        assert 'rando' in decision.error, (
            f'the message must name the rejected agent_id, got {decision.error!r}'
        )
