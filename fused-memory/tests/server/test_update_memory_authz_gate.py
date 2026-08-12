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
    """A minimal stand-in for MemoryService carrying a real config object.

    ``mem0_update`` is ALWAYS rebuilt from an explicit ``Mem0UpdateConfig`` so
    these tests pin schema defaults deterministically — a bare
    ``FusedMemoryConfig()`` is a BaseSettings that loads ``config/config.yaml``
    from the test cwd, and inheriting whatever that file says is how
    TestLiveRead became a 65b011ed8c tripwire casualty."""
    config = FusedMemoryConfig()
    config.mem0_update = Mem0UpdateConfig(**mem0_update_kwargs)
    return SimpleNamespace(config=config)


class TestLiveRead:
    """config/reload.py's precondition for registering the five leaves."""

    def test_reads_config_live_on_every_call(self):
        # 'stranger-' as the not-on-the-bar example, NOT 'curator-': that was
        # this test's original choice, and the esc-3524-1 grant then put
        # curator- on the default bar and broke the first assertion.
        svc = _service()
        first = resolve_mem0_update_authorization(
            svc, agent_id='stranger-session', content_amend=True, metadata_patch=False,
        )
        assert first.allowed is False, 'stranger- is not on the default bar'

        # Mutate the SHARED config object in place, exactly as reload_config does.
        svc.config.mem0_update.content_amend_allowed_agent_prefixes.append('stranger-')

        second = resolve_mem0_update_authorization(
            svc, agent_id='stranger-session', content_amend=True, metadata_patch=False,
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
        assert isinstance(decision.error, str) and 'rando' in decision.error, (
            f'the message must name the rejected agent_id, got {decision.error!r}'
        )


_PROJECT_ID = 'dark_factory'
_MEMORY_ID = '77a3f6bc-0000-0000-0000-000000000000'


def _mock_service(**mem0_update_kwargs):
    """A mock MemoryService whose mem0_update leaves are REAL config values.

    A bare AsyncMock would make every leaf a Mock, which the fail-closed
    resolvers reject — so every gate test would pass for the wrong reason.
    """
    from unittest.mock import AsyncMock

    mock_service = AsyncMock()
    mock_service.config.mem0_update = Mem0UpdateConfig(**mem0_update_kwargs)
    mock_service.update_memory = AsyncMock(return_value={'status': 'updated'})
    return mock_service


async def _call_tool(mock_service, **args):
    from fused_memory.server.tools import create_mcp_server

    server = create_mcp_server(mock_service)
    return await server._tool_manager.call_tool('update_memory', {
        'memory_id': _MEMORY_ID,
        'store': 'mem0',
        'project_id': _PROJECT_ID,
        **args,
    })


class TestUpdateMemoryToolGate:
    """The gate as wired into the MCP tool — modelled on test_add_system_record_gate.

    Same harness and the same ordering rule: an unauthorized caller is rejected
    before any other validation work happens on its behalf, and long before any
    write.
    """

    @pytest.mark.asyncio
    async def test_rejects_unlisted_agent_before_any_write(self):
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service,
            content='rewritten', reason='because', agent_id='claude-interactive',
        )

        assert isinstance(result, dict), f'expected dict, got {type(result)}: {result!r}'
        assert result.get('error_type') == 'Mem0UpdateNotAuthorized', result
        assert isinstance(result.get('error'), str) and result['error']
        mock_service.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_arms_are_gated_independently(self):
        """A wider metadata bar must not become a back door to content amendment."""
        mock_service = _mock_service(
            content_amend_allowed_agent_prefixes=['recon-stage-'],
            metadata_patch_allowed_agent_prefixes=['recon-stage-', 'curator-'],
        )

        ok = await _call_tool(
            mock_service, metadata_patch={'topic': 'x'}, agent_id='curator-gate',
        )
        assert ok.get('error_type') is None, ok
        mock_service.update_memory.assert_called_once()

        mock_service.update_memory.reset_mock()
        denied = await _call_tool(
            mock_service, content='rewritten', reason='because', agent_id='curator-gate',
        )
        assert denied.get('error_type') == 'Mem0UpdateNotAuthorized', denied
        mock_service.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_arms_in_one_call_must_clear_both_bars(self):
        mock_service = _mock_service(
            content_amend_allowed_agent_prefixes=['recon-stage-'],
            metadata_patch_allowed_agent_prefixes=['recon-stage-', 'curator-'],
        )

        result = await _call_tool(
            mock_service,
            content='rewritten', reason='because',
            metadata_patch={'topic': 'x'}, agent_id='curator-gate',
        )

        assert result.get('error_type') == 'Mem0UpdateNotAuthorized', result
        mock_service.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_gate_fires_before_project_and_store_validation(self):
        """An unauthorized caller learns nothing about argument validity.

        Mirrors add_system_record's documented ordering rationale: the gate is
        the whole point of the tool, so rejection precedes any work done on the
        caller's behalf.
        """
        from fused_memory.server.tools import create_mcp_server

        mock_service = _mock_service()
        server = create_mcp_server(mock_service)

        result = await server._tool_manager.call_tool('update_memory', {
            'memory_id': _MEMORY_ID,
            'store': 'not-a-real-store',
            'project_id': 'no such project!!',
            'content': 'rewritten',
            'reason': 'because',
            'agent_id': 'claude-interactive',
        })

        assert result.get('error_type') == 'Mem0UpdateNotAuthorized', (
            f'authz must outrank project/store validation, got: {result!r}'
        )
        mock_service.update_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_kill_switch_denies_even_an_allowlisted_agent(self):
        """One operator knob that reliably stops an in-flight incident."""
        mock_service = _mock_service(enabled=False)

        result = await _call_tool(
            mock_service,
            content='rewritten', reason='because',
            agent_id='recon-stage-memory_consolidator',
        )

        assert result.get('error_type') == 'Mem0UpdateToolDisabled', result
        mock_service.update_memory.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('args', [
        {'content': 'rewritten', 'reason': 'consolidation'},
        {'metadata_patch': {'topic': 'docs-prd-landing'}},
    ], ids=['content_arm', 'metadata_arm'])
    async def test_recon_stage_agent_passes_out_of_the_box(self, args):
        """The task's stated minimum bar: no operator config required."""
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service, agent_id='recon-stage-memory_consolidator', **args,
        )

        assert result.get('error_type') is None, result
        mock_service.update_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_graphiti_store_is_rejected_naming_update_edge(self):
        """No silent fan-out: a wrong store is an actionable error, not a guess."""
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service, store='graphiti',
            content='rewritten', reason='because',
            agent_id='recon-stage-memory_consolidator',
        )

        assert result.get('error_type') == 'ValidationError', result
        assert 'update_edge' in result.get('error', ''), result
        mock_service.update_memory.assert_not_called()
