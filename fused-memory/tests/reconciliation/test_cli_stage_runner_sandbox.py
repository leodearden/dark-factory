"""Tests for sandbox confinement wiring in cli_stage_runner.run_stage_via_cli (task 1935).

Verifies:
  (a) confinement on → invoke_with_cap_retry receives sandbox_wrap= the sentinel callable
  (b) fail-closed: RemediationSandboxUnavailable → StageResult.success=False, error set,
      and invoke_with_cap_retry NOT called (agent not launched unconfined)
  (c) confinement off (sandbox_recon_agents=False) → sandbox_wrap=None passed (or absent)
  (d) non-mocked integration: resolve_recon_sandbox_wrap is actually importable from the
      test environment and returns a callable (or raises RemediationSandboxUnavailable if
      no backend is available) — catches the case where orchestrator becomes unimportable
  (e) task 4003: the PER-RUN recon ``CLAUDE_CONFIG_DIR`` is inside the sandbox's writable
      set, while the config-dir BASE and every sibling run's dir stay read-only.  From
      task 2744 (2026-07-18) until task 4003 (2026-08-11) the per-run dir was outside the
      writable set, so every recon stage silently wrote zero transcripts.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from shared.cli_invoke import AgentResult
from shared.config_dir import TaskConfigDir

from fused_memory.config.schema import ReconciliationConfig
from fused_memory.reconciliation.cli_stage_runner import (
    recon_config_base_dir,
    run_stage_via_cli,
)
from fused_memory.reconciliation.sandbox_guard import (
    RemediationSandboxUnavailable,
    resolve_recon_sandbox_wrap,
)


def _make_agent_result(**overrides) -> AgentResult:
    """Build a successful AgentResult for mocking invoke_with_cap_retry."""
    defaults = {
        'success': True,
        'output': '{"summary": "ok"}',
        'cost_usd': 0.01,
        'stderr': '',
        'structured_output': {'summary': 'ok', 'flagged_items': []},
    }
    defaults.update(overrides)
    return AgentResult(**defaults)


def _make_config(**overrides) -> ReconciliationConfig:
    """Build a minimal ReconciliationConfig for testing."""
    defaults: dict = {
        'sandbox_recon_agents': True,
        'sandbox_recon_writable_extras': [],
    }
    defaults.update(overrides)
    return ReconciliationConfig(**defaults)


def _make_mcp_config() -> dict:
    return {'mcpServers': {}}


@pytest.mark.asyncio
async def test_confinement_on_passes_wrap_to_invoke(tmp_path):
    """When sandbox_recon_agents=True, invoke_with_cap_retry receives sandbox_wrap= sentinel.

    Confirms the wrap callable produced by resolve_recon_sandbox_wrap is
    forwarded as sandbox_wrap= kwarg into invoke_with_cap_retry (which then
    passes it through its **invoke_kwargs to invoke_claude_agent).
    """
    sentinel_wrap = lambda cmd: ['SENTINEL'] + cmd  # noqa: E731

    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap',
        return_value=sentinel_wrap,
    ) as mock_resolve:
        config = _make_config(sandbox_recon_agents=True)
        await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=tmp_path,
        )

    # resolve_recon_sandbox_wrap should have been called
    assert mock_resolve.called, 'Expected resolve_recon_sandbox_wrap to be called'

    # invoke_with_cap_retry should have been called with sandbox_wrap= sentinel
    assert mock_invoke.called, 'Expected invoke_with_cap_retry to be called'
    _, call_kwargs = mock_invoke.call_args
    assert 'sandbox_wrap' in call_kwargs, (
        f'Expected sandbox_wrap in invoke kwargs; got keys={list(call_kwargs.keys())}'
    )
    assert call_kwargs['sandbox_wrap'] is sentinel_wrap, (
        f'Expected sentinel_wrap to be forwarded; got {call_kwargs["sandbox_wrap"]}'
    )


@pytest.mark.asyncio
async def test_fail_closed_not_launched(tmp_path):
    """When resolve_recon_sandbox_wrap raises RemediationSandboxUnavailable,
    the StageResult carries an error AND invoke_with_cap_retry is NOT called.

    This is the explicit fail-closed leaf: the agent must not be launched
    unconfined when confinement is enabled but no backend is available.
    """
    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap',
        side_effect=RemediationSandboxUnavailable('no sandbox'),
    ):
        config = _make_config(sandbox_recon_agents=True)
        result = await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=tmp_path,
        )

    # Agent must NOT have been launched
    assert not mock_invoke.called, (
        'invoke_with_cap_retry must NOT be called when confinement fails (fail-closed)'
    )
    # StageResult must carry an error
    assert result.success is False, 'StageResult.success must be False on fail-closed'
    assert result.error, f'StageResult.error must be non-empty; got {result.error!r}'


@pytest.mark.asyncio
async def test_confinement_off_no_wrap(tmp_path):
    """When sandbox_recon_agents=False, invoke_with_cap_retry is called with sandbox_wrap=None.

    Operators who explicitly opt out of confinement should get unconfined
    invocations (sandbox_wrap=None, which cli_invoke treats as no-wrap).
    """
    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'fused_memory.reconciliation.cli_stage_runner.resolve_recon_sandbox_wrap',
    ) as mock_resolve:
        config = _make_config(sandbox_recon_agents=False)
        await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=tmp_path,
        )

    # resolve_recon_sandbox_wrap should NOT be called when confinement is off
    assert not mock_resolve.called, (
        'resolve_recon_sandbox_wrap must not be called when sandbox_recon_agents=False'
    )

    # invoke_with_cap_retry should have been called
    assert mock_invoke.called, 'Expected invoke_with_cap_retry to be called'
    _, call_kwargs = mock_invoke.call_args
    # sandbox_wrap should be absent or None
    sandbox_wrap_val = call_kwargs.get('sandbox_wrap')
    assert sandbox_wrap_val is None, (
        f'Expected sandbox_wrap=None when confinement is off; got {sandbox_wrap_val!r}'
    )


def test_resolve_recon_sandbox_wrap_importable_and_functional(tmp_path):
    """Non-mocked: resolve_recon_sandbox_wrap must be importable and return a callable
    (or raise RemediationSandboxUnavailable when no backend is available).

    This test does NOT mock orchestrator — it exercises the real import path to
    catch the case where orchestrator becomes unimportable in the test environment
    (e.g. venv without orchestrator installed).  A missing orchestrator surfaces as
    RemediationSandboxUnavailable, which is the expected fail-closed result; but if
    orchestrator IS importable and Landlock/bwrap is available, a callable is returned.

    Rationale: the mocked wiring tests (above) verify that cli_stage_runner correctly
    routes the result of resolve_recon_sandbox_wrap, but they never exercise the real
    import boundary.  This test guards that boundary.
    """
    try:
        wrap = resolve_recon_sandbox_wrap(tmp_path, writable_extras=[])
    except RemediationSandboxUnavailable:
        # Acceptable: no Landlock/bwrap backend available, or orchestrator not importable.
        # The fail-closed behaviour is correct; cli_stage_runner will refuse to launch.
        return

    # If no exception was raised, we must have a callable back.
    assert callable(wrap), (
        f'resolve_recon_sandbox_wrap should return a Callable; got {type(wrap)}'
    )
    # Sanity-check: wrap must accept a list[str] and return a list[str].
    result = wrap(['claude', '--print'])
    assert isinstance(result, list) and result, (
        f'wrap callable should return a non-empty list; got {result!r}'
    )
    assert all(isinstance(t, str) for t in result), (
        f'wrap callable result must be list[str]; got {result!r}'
    )


def _writable_values(argv: list[str]) -> list[str]:
    """Extract the ``--writable <path>`` values from a wrapped argv.

    Backend-agnostic on purpose: ``build_landlock_command`` emits
    ``--writable <path>`` pairs, so this reads the grant list straight off the
    argv the kernel helper will actually be handed — not off a mock.
    """
    return [argv[i + 1] for i, tok in enumerate(argv) if tok == '--writable']


async def _capture_writables(config, cwd, config_dir):
    """Drive the REAL two-layer path and return the granted ``--writable`` paths.

    ``resolve_recon_sandbox_wrap`` is deliberately NOT mocked: this exercises
    cli_stage_runner → sandbox_guard → build_landlock_command end to end, which
    is the only way to catch a writable-set regression (a mocked wrap would have
    happily passed all through the 2026-07-18 → 2026-08-11 breakage window).
    """
    mock_invoke = AsyncMock(return_value=_make_agent_result())

    with patch(
        'fused_memory.reconciliation.cli_stage_runner.invoke_with_cap_retry',
        mock_invoke,
    ), patch(
        'orchestrator.agents.landlock.is_landlock_available',
        return_value=True,
    ):
        result = await run_stage_via_cli(
            system_prompt='sys',
            payload='pay',
            disallowed_tools=[],
            config=config,
            mcp_config=_make_mcp_config(),
            cwd=cwd,
            session_id='sid',
            config_dir=config_dir,
        )

    assert mock_invoke.called, (
        f'invoke_with_cap_retry must be called (fail-closed path taken?); '
        f'StageResult.error={result.error!r}'
    )
    _, call_kwargs = mock_invoke.call_args
    sandbox_wrap = call_kwargs.get('sandbox_wrap')
    assert callable(sandbox_wrap), (
        f'Expected a sandbox_wrap callable; got {sandbox_wrap!r}'
    )
    wrapped = sandbox_wrap(['claude', '--print'])
    # `call_kwargs.get` is typed `object`; narrow before handing it to a
    # `list[str]` parameter (same idiom as the import-boundary test above).
    assert isinstance(wrapped, list), (
        f'wrap callable must return a list; got {wrapped!r}'
    )
    return _writable_values(wrapped)


@pytest.mark.asyncio
async def test_recon_config_dir_is_landlock_writable(tmp_path):
    """The PER-RUN recon config dir is writable; the base and siblings are NOT (task 4003).

    Regression pin for the 2026-07-18 → 2026-08-11 silent-transcript-loss window.
    Task 2744 redirected recon stages to a per-run ``CLAUDE_CONFIG_DIR`` under
    ``<data_dir>/recon-config/`` — a path that is neither ``/tmp`` nor
    ``<cwd>/.task``, i.e. outside every writable root both sandbox backends
    grant.  The CLI could not write its session JSONL, so
    ``count_transcript_turns`` returned None forever: the liveness watchdog went
    inert and every cap-retry force-freshed instead of resuming, silently, for
    three weeks.

    The two negative assertions are the credential-isolation invariant and must
    never regress: ``recon_config_base_dir(...)`` is the root under which EVERY
    run's ``claude-config-<run_id>/.credentials.json`` lives, so granting the
    base (the naive fix) would hand every recon stage write access to every
    other run's OAuth credentials — a capability that does not exist today.
    """
    base = recon_config_base_dir(tmp_path)
    mine = TaskConfigDir(task_id='run-mine', base_dir=base)
    # A second run's dir must exist on disk, so "not granted" is a real
    # statement about the ruleset rather than an accident of absence.
    sibling = TaskConfigDir(task_id='run-other', base_dir=base)

    writables = await _capture_writables(
        _make_config(sandbox_recon_agents=True), tmp_path, mine,
    )

    assert str(mine.path) in writables, (
        f'The per-run recon config dir must be in the writable set, else the CLI '
        f'cannot write its transcript. Wanted {str(mine.path)!r}; got {writables!r}'
    )
    assert str(base) not in writables, (
        f'The config-dir BASE must NEVER be granted — it would make every run\'s '
        f'.credentials.json writable by every other run. Got {writables!r}'
    )
    assert str(sibling.path) not in writables, (
        f'A sibling run\'s config dir must stay read-only. Got {writables!r}'
    )


@pytest.mark.asyncio
async def test_operator_writable_extras_are_preserved(tmp_path):
    """The computed per-run grant APPENDS to operator extras, never replaces them.

    ``sandbox_recon_writable_extras`` is an operator escape hatch (e.g. a uvx
    cache dir needed by a stdio MCP server); a computed grant that clobbered it
    would break those hosts.
    """
    base = recon_config_base_dir(tmp_path)
    mine = TaskConfigDir(task_id='run-mine', base_dir=base)

    writables = await _capture_writables(
        _make_config(
            sandbox_recon_agents=True,
            sandbox_recon_writable_extras=['/var/tmp/opextra'],
        ),
        tmp_path,
        mine,
    )

    assert '/var/tmp/opextra' in writables, (
        f'Operator-configured extras must survive the computed append; got {writables!r}'
    )
    assert str(mine.path) in writables, (
        f'The per-run config dir must still be granted alongside operator extras; '
        f'got {writables!r}'
    )
