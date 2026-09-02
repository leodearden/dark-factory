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
from typing import Any
from unittest.mock import MagicMock

import pytest

from fused_memory.config.schema import EntityMintConfig, FusedMemoryConfig
from fused_memory.server.entity_mint_authz import (
    resolve_entity_mint_authorization,
    validate_mint_name,
)


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


class TestValidateMintName:
    """Guard 3: only a CANONICALLY-SPELLED task-shaped name may be minted.

    Every expected classification below was CONFIRMED by running
    ``utils/canonical_labels.py::parse_node_name`` against the current tree —
    none is guessed. That module is the single normative label vocabulary
    (INV-5), so ``validate_mint_name`` calls it rather than carrying a second
    copy of the pattern.
    """

    def test_canonical_task_name_is_accepted_carrying_the_referent(self):
        decision = validate_mint_name('Task 3222')
        assert decision.allowed is True
        assert decision.error_type is None
        assert decision.error is None
        assert decision.referent is not None
        assert decision.referent.number == '3222'
        assert decision.referent.node_name == 'Task 3222', (
            'the decision must carry the parsed referent so the caller need not '
            're-parse the name'
        )

    @pytest.mark.parametrize('name', [
        'task #3222', 'Task: 3222', 'tasks 3222', '  task 3222  ',
    ])
    def test_non_canonical_spellings_are_refused_naming_the_canonical_form(self, name):
        """These all PARSE — to Referent(number='3222') — but do not round-trip.

        Refusing them (rather than silently normalizing) is what makes the
        variants converge on ONE node instead of splitting, exactly as leaf eta
        already does; naming the canonical form is what lets the caller retry.
        """
        decision = validate_mint_name(name)
        assert decision.allowed is False
        assert decision.error_type == 'EntityMintNonCanonicalName', decision
        assert isinstance(decision.error, str)
        assert 'Task 3222' in decision.error, (
            f'the refusal must name the canonical retry form, got {decision.error!r}'
        )

    @pytest.mark.parametrize('name', [
        'Postgres', 'Task 42 orchestrator', 'subtask 5', '',
    ])
    def test_non_task_names_are_refused(self, name):
        """parse_node_name returns None for each of these. Refusing keeps this
        from becoming a general junk-node minter."""
        decision = validate_mint_name(name)
        assert decision.allowed is False
        assert decision.error_type == 'EntityMintNonTaskName', decision
        assert isinstance(decision.error, str) and decision.error
        assert decision.referent is None

    def test_project_qualified_name_is_accepted(self):
        """The cross-project case ensure_entity_node was originally built for
        (task 3335): LLM extraction discards the qualifier and collapses the
        reference onto a bare 'Task N'. The qualifier is never normalized away,
        because that collapse is the bug utils/cross_project_refs.py exists to
        detect."""
        decision = validate_mint_name('reify:132')
        assert decision.allowed is True, decision
        assert decision.referent is not None
        assert decision.referent.project_id == 'reify'
        assert decision.referent.node_name == 'reify:132'

    def test_uppercase_qualifier_is_refused_as_non_canonical(self):
        """'REIFY:132' parses, but canonicalizes its qualifier to lowercase, so
        node_name != name — the NON-CANONICAL arm, not the None arm."""
        decision = validate_mint_name('REIFY:132')
        assert decision.allowed is False
        assert decision.error_type == 'EntityMintNonCanonicalName', decision
        assert 'reify:132' in (decision.error or ''), decision

    def test_non_ascii_digit_is_refused_as_a_non_task_name(self):
        """Pins the interaction with canonical_labels' \\d -> [0-9] narrowing:
        an Arabic-Indic digit now parses to None, so the mint tool can never
        create a 'Task \u0663' node."""
        decision = validate_mint_name('task \u0663')
        assert decision.allowed is False
        assert decision.error_type == 'EntityMintNonTaskName', decision

    @pytest.mark.parametrize('name', [None, 42, b'Task 3222', ['Task 3222'], {'a': 1}])
    def test_non_str_input_returns_a_refusal_rather_than_raising(self, name):
        decision = validate_mint_name(name)
        assert decision.allowed is False, f'{name!r} must not pass'
        assert isinstance(decision.error_type, str) and decision.error_type
        assert isinstance(decision.error, str) and decision.error


# ---------------------------------------------------------------------------
# The tool half: the gate as wired into the MCP surface.
#
# Copies the second half of ``tests/server/test_update_memory_authz_gate.py``
# (its ``_mock_service`` / ``_call_tool`` / ``TestUpdateMemoryToolGate``). The
# resolver tests above prove the DECISION; these prove the WIRING — that the
# tool asks, in the right order, and honours the answer.
# ---------------------------------------------------------------------------

_PROJECT_ID = 'dark_factory'
_CANONICAL_NAME = 'Task 3222'


def _mock_service(**entity_mint_kwargs):
    """A mock MemoryService whose entity_mint leaves are REAL config values.

    A bare AsyncMock would make every leaf a Mock, which the fail-closed
    resolvers reject — so every gate test below would pass for the wrong
    reason (denied because the config was unreadable, not because the gate
    works).
    """
    from unittest.mock import AsyncMock

    mock_service = AsyncMock()
    mock_service.config.entity_mint = EntityMintConfig(**entity_mint_kwargs)
    mock_service.ensure_entity_node = AsyncMock(return_value={
        'status': 'minted', 'store': 'graphiti', 'uuid': 'u1', 'minted': True,
        'name': _CANONICAL_NAME,
    })
    return mock_service


def _server(mock_service, **server_kwargs):
    from fused_memory.server.tools import create_mcp_server

    return create_mcp_server(mock_service, **server_kwargs)


async def _call_on(server, **args):
    return await server._tool_manager.call_tool('ensure_entity_node', {
        'name': _CANONICAL_NAME,
        'project_id': _PROJECT_ID,
        **args,
    })


async def _call_tool(mock_service, _server_kwargs=None, **args):
    return await _call_on(_server(mock_service, **(_server_kwargs or {})), **args)


class TestEnsureEntityNodeToolGate:
    """The authorization gate as the MCP tool actually applies it.

    Same ordering rule the sibling ``update_memory`` gate documents: an
    unauthorized caller is rejected before any other validation work happens on
    its behalf, and long before any write. Note ``update_memory`` has an extra
    arm-PRESENCE check between identity and authorization; that is specific to
    its multi-arm shape and has NO analogue here, so authorization is genuinely
    the second thing this tool does.
    """

    @pytest.mark.asyncio
    async def test_an_allowlisted_agent_mints(self):
        """The task's user-observable POSITIVE signal."""
        mock_service = _mock_service()

        result = await _call_tool(mock_service, agent_id='curator-repair')

        assert result.get('error_type') is None, result
        assert result.get('minted') is True, result
        mock_service.ensure_entity_node.assert_awaited_once()
        kwargs = mock_service.ensure_entity_node.await_args.kwargs
        assert kwargs['name'] == _CANONICAL_NAME
        assert kwargs['project_id'] == _PROJECT_ID
        assert kwargs['agent_id'] == 'curator-repair'

    @pytest.mark.asyncio
    async def test_project_id_reaches_the_service_canonicalized(self):
        """The boundary adapter runs before dispatch, not after."""
        mock_service = _mock_service()

        await _call_tool(
            mock_service, project_id='Dark_Factory', agent_id='curator-repair',
        )

        kwargs = mock_service.ensure_entity_node.await_args.kwargs
        assert kwargs['project_id'] == 'dark_factory', (
            f'the service must receive the canonical form, got {kwargs["project_id"]!r}'
        )

    @pytest.mark.asyncio
    async def test_a_non_allowlisted_agent_is_refused_and_nothing_is_created(self):
        """The task's user-observable NEGATIVE control, same session."""
        mock_service = _mock_service()

        result = await _call_tool(mock_service, agent_id='claude-interactive')

        assert result.get('error_type') == 'EntityMintNotAuthorized', result
        error = result.get('error') or ''
        assert 'recon-stage-' in error and 'curator-' in error, (
            f'the refusal must NAME the required prefixes so the caller can '
            f'act on it, got {error!r}'
        )
        mock_service.ensure_entity_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_the_kill_switch_denies_even_an_allowlisted_agent(self):
        mock_service = _mock_service(enabled=False)

        result = await _call_tool(mock_service, agent_id='recon-stage-1')

        assert result.get('error_type') == 'EntityMintToolDisabled', result
        mock_service.ensure_entity_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_flipping_the_kill_switch_denies_the_very_next_call(self):
        """No restart, no reconstruction — the SAME server object.

        This is the operator story the green-tier registration promises: a
        restart-only kill switch is no kill switch.
        """
        mock_service = _mock_service()
        server = _server(mock_service)

        allowed = await _call_on(server, agent_id='recon-stage-1')
        assert allowed.get('error_type') is None, allowed

        mock_service.config.entity_mint.enabled = False
        denied = await _call_on(server, agent_id='recon-stage-1')

        assert denied.get('error_type') == 'EntityMintToolDisabled', denied
        assert mock_service.ensure_entity_node.await_count == 1, (
            'only the first call may have dispatched'
        )

    @pytest.mark.asyncio
    async def test_authz_outranks_every_other_argument_error(self):
        """An unauthorized caller learns nothing about its other arguments.

        One call carrying THREE independent defects: an unauthorized agent_id,
        a garbage project_id and a junk non-task name. The authz error is the
        one that must come back.
        """
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service,
            name='Postgres',
            project_id='no such project!!',
            agent_id='claude-interactive',
        )

        assert result.get('error_type') == 'EntityMintNotAuthorized', (
            f'authz must outrank project and name validation, got {result!r}'
        )
        mock_service.ensure_entity_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unregistered_project_is_refused_before_the_name_is_parsed(self):
        """`_known_project_gate` is load-bearing, not decorative.

        `_graph_for(group_id)` creates a graph ON DEMAND, so a typo'd project_id
        would otherwise mint into a brand-new graph nobody is watching. Of the
        four existing entity tools only `reassign_edge` calls this gate — the
        others stop at `validate_project_id` — so wiring it here is a real
        correction to the prevailing local pattern.

        The gate delegates to `utils/validation.py::validate_known_project_id`,
        which is PERMISSIVE when the registry is falsy, so a non-empty
        `known_projects` is required or this would pass vacuously.
        """
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service,
            {'known_projects': {'dark_factory': '/tmp/df'}},
            project_id='some_other_project',
            name='Postgres', # also invalid — the project error must win
            agent_id='curator-repair',
        )

        assert result.get('error_type') == 'ValidationError', result
        assert 'not a known project' in (result.get('error') or ''), result
        mock_service.ensure_entity_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_non_canonical_name_is_refused_without_dispatch(self):
        """The variants converge instead of splitting — exactly as leaf eta does."""
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service, name='task #3222', agent_id='curator-repair',
        )

        assert result.get('error_type') == 'EntityMintNonCanonicalName', result
        assert _CANONICAL_NAME in (result.get('error') or ''), (
            'the refusal must name the canonical form so the caller can retry'
        )
        mock_service.ensure_entity_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_non_task_name_is_refused_without_dispatch(self):
        """v1 is not a general junk-node minter."""
        mock_service = _mock_service()

        result = await _call_tool(
            mock_service, name='Postgres', agent_id='curator-repair',
        )

        assert result.get('error_type') == 'EntityMintNonTaskName', result
        mock_service.ensure_entity_node.assert_not_called()


class TestMintNameTaskVerification:
    """Guard 4: the referent must name a task the live registry actually has.

    The THREE-VALUED distinction this needs is exactly what
    ``_claim_task_statuses`` cannot express — its documented contract is "an
    ABSENT key is the unresolvable signal", so "no such task" and "could not
    consult" collapse to the same absence. Guard 4 must refuse on the former
    ONLY: refusing on the latter would make the tool unusable on any deployment
    without the registry.
    """

    @staticmethod
    def _server(mock_service, *, statuses=None, raises=False,
                known_projects=None, task_interceptor=True) -> tuple[Any, Any]:
        from unittest.mock import AsyncMock, MagicMock

        from fused_memory.server.tools import create_mcp_server

        interceptor = None
        if task_interceptor:
            interceptor = MagicMock()
            interceptor.get_statuses = AsyncMock(
                side_effect=RuntimeError('taskmaster down') if raises else None,
                return_value=None if raises else (statuses or {}),
            )
        server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=known_projects if known_projects is not None
            else {'dark_factory': '/tmp/df', 'reify': '/tmp/reify'},
        )
        return server, interceptor

    @pytest.mark.asyncio
    async def test_a_task_the_registry_has_dispatches(self):
        """POSITIVE PRESENT."""
        mock_service = _mock_service()
        server, interceptor = self._server(
            mock_service, statuses={'3222': 'done'},
        )

        result = await _call_on(server, agent_id='curator-repair')

        assert result.get('error_type') is None, result
        mock_service.ensure_entity_node.assert_awaited_once()
        interceptor.get_statuses.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_successfully_consulted_registry_that_lacks_the_task_refuses(self):
        """POSITIVE ABSENT — the registry ANSWERED, and the answer was 'no'.

        This is the leg that distinguishes guard 4 from a probe that only
        tolerates failure: a consulted-and-empty read is a real "no such task",
        and minting a node for a task that does not exist is exactly the junk
        node this gate exists to prevent.
        """
        mock_service = _mock_service()
        server, interceptor = self._server(mock_service, statuses={})

        result = await _call_on(server, agent_id='curator-repair')

        assert result.get('error_type') == 'EntityMintUnknownTask', result
        assert '3222' in (result.get('error') or ''), result
        mock_service.ensure_entity_node.assert_not_called()
        interceptor.get_statuses.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_dict_lacking_the_id_also_refuses(self):
        """A batched read that answered about OTHER ids still answered 'no' here."""
        mock_service = _mock_service()
        server, _interceptor = self._server(
            mock_service, statuses={'9999': 'done'},
        )

        result = await _call_on(server, agent_id='curator-repair')

        assert result.get('error_type') == 'EntityMintUnknownTask', result
        mock_service.ensure_entity_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unconfigured_taskmaster_does_not_refuse(self):
        """UNRESOLVABLE (1) — the tool must not require the registry to exist."""
        mock_service = _mock_service()
        server, _interceptor = self._server(mock_service, task_interceptor=False)

        result = await _call_on(server, agent_id='curator-repair')

        assert result.get('error_type') is None, result
        mock_service.ensure_entity_node.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_unregistered_referent_project_does_not_refuse(self):
        """UNRESOLVABLE (2) — no root for the referent's project, so no answer.

        Note the WRITING project stays registered (`_known_project_gate` already
        passed on it); it is the foreign qualifier that has no root.
        """
        mock_service = _mock_service()
        server, interceptor = self._server(
            mock_service, statuses={}, known_projects={'dark_factory': '/tmp/df'},
        )

        result = await _call_on(
            server, name='reify:132', agent_id='curator-repair',
        )

        assert result.get('error_type') is None, result
        mock_service.ensure_entity_node.assert_awaited_once()
        interceptor.get_statuses.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_raising_status_read_does_not_refuse(self):
        """UNRESOLVABLE (3) — a registry outage must not become a mint refusal."""
        mock_service = _mock_service()
        server, _interceptor = self._server(mock_service, raises=True)

        result = await _call_on(server, agent_id='curator-repair')

        assert result.get('error_type') is None, result
        mock_service.ensure_entity_node.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_foreign_referent_probes_its_own_project_root(self):
        """CROSS-PROJECT ROUTING — the claimed project adjudicates, not the writer.

        Reading the writer's tree for 'does reify task 132 exist' answers a
        question nobody asked, confidently and with the wrong tree — the same
        esc-3085-1 mistake `_group_refs_by_project` exists to avoid.
        """
        mock_service = _mock_service()
        server, interceptor = self._server(
            mock_service, statuses={'132': 'done'},
        )

        result = await _call_on(
            server, name='reify:132', agent_id='curator-repair',
        )

        assert result.get('error_type') is None, result
        interceptor.get_statuses.assert_awaited_once()
        kwargs = interceptor.get_statuses.call_args.kwargs
        assert kwargs.get('project_root') == '/tmp/reify', (
            "the probe must target the REFERENT's project root, not the "
            f'writing project\'s; got {kwargs!r}'
        )
        assert sorted(kwargs.get('ids') or []) == ['132'], kwargs

    @pytest.mark.asyncio
    async def test_verification_runs_after_authz_and_the_name_guards(self):
        """An unauthorized caller is never worth a registry round trip."""
        mock_service = _mock_service()
        server, interceptor = self._server(mock_service, statuses={})

        result = await _call_on(server, agent_id='claude-interactive')

        assert result.get('error_type') == 'EntityMintNotAuthorized', result
        interceptor.get_statuses.assert_not_awaited()


class TestClaimTaskStatusesIsUnchanged:
    """REGRESSION: extracting `_batched_task_statuses` must not move the
    completion-claim gate.

    `_claim_task_statuses` keeps byte-identical external behaviour by DISCARDING
    the new `consulted` set. These legs re-pin its contract through the public
    ingestion path; `tests/server/test_completion_claim_gate_ingestion.py` and
    `tests/test_completion_claim_gate.py` cover it in full and must pass
    UNMODIFIED.
    """

    @staticmethod
    def _episode_server(*, statuses=None, raises=False, known_projects=None,
                        task_interceptor=True) -> tuple[Any, Any, Any]:
        from unittest.mock import AsyncMock, MagicMock

        from fused_memory.server.tools import create_mcp_server

        mock_service = AsyncMock()
        ep_result = MagicMock()
        ep_result.model_dump.return_value = {'id': 'ep'}
        mock_service.add_episode.return_value = ep_result

        interceptor = None
        if task_interceptor:
            interceptor = MagicMock()
            interceptor.get_statuses = AsyncMock(
                side_effect=RuntimeError('taskmaster down') if raises else None,
                return_value=None if raises else (statuses or {}),
            )
            interceptor.get_ticket_row = AsyncMock(return_value=None)
        server = create_mcp_server(
            mock_service,
            task_interceptor=interceptor,
            known_projects=known_projects if known_projects is not None
            else {'dark_factory': '/df-root', 'reify': '/reify-root'},
        )
        return server, mock_service, interceptor

    @pytest.fixture(autouse=True)
    def _no_real_escalations(self, monkeypatch):
        """A tagged ingestion files into `<project_root>/data/escalations`."""
        import fused_memory.server.tools as tools_mod

        monkeypatch.setattr(
            tools_mod, 'emit_unverified_claim_escalation', lambda *a, **k: None,
        )

    @staticmethod
    async def _ingest(server, content, project_id='reify'):
        return await server._tool_manager.call_tool('add_episode', {
            'content': content,
            'agent_id': 'claude-task-5638-implementer',
            'project_id': project_id,
        })

    @pytest.mark.asyncio
    async def test_one_batched_read_per_claimed_project(self):
        server, _svc, interceptor = self._episode_server(
            statuses={'3142': 'in-progress', '5638': 'in-progress'},
        )

        await self._ingest(
            server,
            'dark_factory task 3142 has landed. reify task 5638 has landed',
        )

        reads = {
            call.kwargs.get('project_root'): sorted(call.kwargs.get('ids') or [])
            for call in interceptor.get_statuses.await_args_list
        }
        assert reads == {'/df-root': ['3142'], '/reify-root': ['5638']}, reads

    @pytest.mark.asyncio
    async def test_a_raising_read_still_leaves_the_key_absent(self):
        """Absent, not fabricated — the claim lands UNVERIFIABLE and is tagged."""
        server, mock_service, _interceptor = self._episode_server(raises=True)

        await self._ingest(server, 'reify task 5638 has landed')

        kwargs = mock_service.add_episode.call_args.kwargs
        assert kwargs.get('unverified_claim') is True, (
            f'a raising status read must tag, never pass; got {kwargs!r}'
        )

    @pytest.mark.asyncio
    async def test_an_unconfigured_interceptor_still_leaves_the_key_absent(self):
        server, mock_service, _interceptor = self._episode_server(
            task_interceptor=False,
        )

        await self._ingest(server, 'reify task 5638 has landed')

        kwargs = mock_service.add_episode.call_args.kwargs
        assert kwargs.get('unverified_claim') is True, (
            f'no interceptor must tag, never fabricate a permissive answer; '
            f'got {kwargs!r}'
        )

    @pytest.mark.asyncio
    async def test_an_unregistered_project_still_leaves_the_key_absent(self):
        # An EMPTY registry, because `_known_project_gate` is permissive on a
        # falsy one — that is the only shape where ingestion proceeds and the
        # claimed project still resolves to no root. A non-empty registry
        # missing the writer would be rejected before any claim was read.
        server, mock_service, _interceptor = self._episode_server(
            statuses={'5638': 'done'}, known_projects={},
        )

        await self._ingest(server, 'reify task 5638 has landed')

        kwargs = mock_service.add_episode.call_args.kwargs
        assert kwargs.get('unverified_claim') is True, (
            f'an unregistered project must tag, never pass; got {kwargs!r}'
        )
