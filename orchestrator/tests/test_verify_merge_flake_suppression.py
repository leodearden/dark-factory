"""Tests for the merge-gate single flake-retry gate (PRD task α).

`plans/cpu-load-robust-verify-prd.md` task α: give role='merge' verify the
per-node-id isolated-rerun-confirm gate that run_main_tip_sweep /
confirm_main_tip_failure_is_real already have, so a CPU-starvation-induced red
on an unrelated test does not block a correct, code-complete task from landing.
Critically, α resolves to a pass/fail verdict INLINE (returns a VerifyResult) —
it MUST NOT raise VerifyInfraError, because the merge path (merge_queue.py) has
no VerifyInfraError handler.

Test structure mirrors the confirm-gate test suite in
test_verify_main_tip_sweep.py (fake on-disk project layout so the node-id ->
subproject existence mapping runs against real files; run_verification
monkeypatched so no real subprocess runs).
"""

from __future__ import annotations

import dataclasses

from orchestrator.verify_cmd import (
    ToolKind,
    parse_config_command,
    render,
    with_pytest_timeout,
)


class TestWithPytestTimeout:
    """with_pytest_timeout(cmd, secs) appends a `--timeout <secs>` flag to a
    structured pytest command's base_flags (PRD task α; INV-5 reuse).

    Copies with_junitxml's structure exactly: same ToolKind.PYTEST +
    cmd.raw is None guard, same `replace(cmd, base_flags=(*cmd.base_flags,
    '--timeout', str(secs)))` shape. A structured command's per-test timeout
    override is REQUIRED because the pyproject `timeout=60` default lives in
    `[tool.pytest.ini_options]`, NOT addopts, so the serial-recovery
    `-o addopts=` does not clear it — without an explicit `--timeout=300` the
    confirm run could itself starve into a false non-suppression.
    """

    def test_structured_single_invocation_appends_timeout_flag(self):
        cmd = parse_config_command('pytest tests/x.py')
        result = render(with_pytest_timeout(cmd, 300))
        assert result == 'pytest --timeout 300 tests/x.py'

    def test_base_flags_end_with_timeout_pair(self):
        cmd = parse_config_command('pytest tests/x.py')
        mutated = with_pytest_timeout(cmd, 300)
        assert mutated.base_flags[-2:] == ('--timeout', '300')
        assert mutated.raw is None

    def test_secs_rendered_as_string(self):
        """secs is an int at the call site but must render as a bare token."""
        cmd = parse_config_command('pytest tests/x.py')
        mutated = with_pytest_timeout(cmd, 300)
        assert '300' in mutated.base_flags
        assert 300 not in mutated.base_flags  # str, never int

    def test_raw_retained_chain_returned_unchanged(self):
        """A recognised-but-unstructurable pytest chain (raw is not None) is
        NEVER regex-rewritten — mirrors with_junitxml's deliberate divergence
        from apply_pytest_numprocesses/serial_pytest. Byte-identical no-op.
        """
        raw = (
            'cd shared && uv run pytest tests/ && '
            'cd ../orchestrator && uv run pytest tests/'
        )
        cmd = parse_config_command(raw)
        assert cmd.tool is ToolKind.PYTEST
        assert cmd.raw == raw

        result = with_pytest_timeout(cmd, 300)
        assert result == cmd
        assert render(result) == raw

    def test_non_pytest_command_returned_unchanged(self):
        cmd = parse_config_command('ruff check .')
        assert with_pytest_timeout(cmd, 300) == cmd

    def test_pyright_command_returned_unchanged(self):
        cmd = parse_config_command('pyright')
        assert with_pytest_timeout(cmd, 300) == cmd

    def test_noop_on_opaque(self):
        cmd = parse_config_command('mypy src/')
        assert cmd.tool is ToolKind.OPAQUE
        assert with_pytest_timeout(cmd, 300) == cmd

    def test_noop_returns_same_object_identity(self):
        """A no-op returns the SAME object (is), so _with_pytest_timeout_str's
        `rewritten is parsed` short-circuit detects it and returns the input
        string byte-identically.
        """
        cmd = parse_config_command('cargo test --workspace')
        assert with_pytest_timeout(cmd, 300) is cmd


class TestWithPytestTimeoutStr:
    """_with_pytest_timeout_str(cmd, secs) — verify.py string wrapper around
    parse_config_command -> with_pytest_timeout -> render (mirrors
    _serial_pytest_str). Returns cmd unchanged when None / non-pytest / OPAQUE.
    """

    def test_roundtrips_real_pytest_command(self):
        from orchestrator.verify import _with_pytest_timeout_str

        cmd = (
            'uv run --project orchestrator --directory orchestrator '
            'pytest tests/ --tb=short -q'
        )
        result = _with_pytest_timeout_str(cmd, 300)
        assert result is not None
        assert '--timeout 300' in result

    def test_none_input_returns_none(self):
        from orchestrator.verify import _with_pytest_timeout_str

        assert _with_pytest_timeout_str(None, 300) is None

    def test_non_pytest_input_returned_unchanged(self):
        from orchestrator.verify import _with_pytest_timeout_str

        cmd = 'cargo test --workspace'
        assert _with_pytest_timeout_str(cmd, 300) == cmd

    def test_opaque_input_returned_unchanged(self):
        from orchestrator.verify import _with_pytest_timeout_str

        cmd = 'mypy src/'
        assert _with_pytest_timeout_str(cmd, 300) == cmd

    def test_composes_with_serial_pytest_str(self):
        """The α confirm command is
        _with_pytest_timeout_str(_serial_pytest_str(scoped)) — both flags
        must survive composition (serial recovery + generous per-test timeout).
        """
        from orchestrator.verify import _serial_pytest_str, _with_pytest_timeout_str

        base = (
            'uv run --project orchestrator --directory orchestrator '
            'pytest tests/ --tb=short -q'
        )
        composed = _with_pytest_timeout_str(_serial_pytest_str(base), 300)
        assert composed is not None
        assert '-p no:xdist' in composed
        assert '-o addopts=' in composed
        assert '--timeout 300' in composed


# Sanity: the module imports even before verify.py gains the new symbols would
# fail — dataclasses is imported here for later steps' VerifyResult construction.
_ = dataclasses
