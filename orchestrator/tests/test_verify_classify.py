"""Tests for orchestrator.verify_classify — the tool-dispatched
``classify_failure(tool, rc, output, timed_out) -> FailureCategory``
(PRD: plans/verify-plan-prd.md task δ; Contract §classify_failure, Invariants C1/C2).

Replaces verify.py's tool-BLIND ``_classify_failure`` regex ladder with a
per-tool-dispatched classifier: a tool-T pattern lives ONLY in tool-T's
table, so a cargo token can no longer swallow a pytest/rustc line by
construction (C1). Where a tool offers structured output, the classifier
parses it directly instead of regex-matching human text (C2).

Test coverage:
  step-1: guards (rc==0 / timed_out) applied uniformly across every ToolKind,
          ahead of any per-tool dispatch, plus the ToolKind.OPAQUE branch —
          the full legacy generic ladder, byte-identical to today's
          _classify_failure — and the FailureCategory return contract.
  step-3: PYTEST tool table + env_transient PYTEST-only scoping (C1 forward).
  step-5: CARGO_TEST/CARGO_CLIPPY + NPX tables — GOLDEN cargo corpus
          (re-grounded from the historical 1103/1109/1116 fix commits) plus
          the headline C1 reverse signal.
  step-7: structured-output parsing (C2) — cargo/pyright/ruff JSON.
  step-9: verify._summarize_checks threads per-check tool identity (the
          config test_cmd/lint_cmd/type_cmd) into classify_failure, instead
          of discarding it — the same C1 signal one layer up the stack, at
          the real run_verification integration point.
"""

from __future__ import annotations

import json

import pytest

from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_cmd import ToolKind

# Every recognised tool identity, including OPAQUE — the guards in step-1
# apply before any per-tool dispatch, so they must hold for all of them.
ALL_TOOL_KINDS = list(ToolKind)


def _classify(tool: ToolKind, output: str, rc: int, timed_out: bool) -> FailureCategory:
    from orchestrator.verify_classify import classify_failure  # noqa: PLC0415

    return classify_failure(tool, rc, output, timed_out)


class TestGuardsApplyToEveryToolKind:
    """(a)/(b): rc==0 -> PASSED and timed_out -> INFRA_TIMEOUT are checked
    up front, before any per-tool dispatch, so they hold for every ToolKind
    regardless of what the per-tool table would otherwise have matched."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_passed_when_rc_zero(self, tool):
        assert _classify(tool, '', 0, False) == FailureCategory.PASSED

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_passed_when_rc_zero_even_with_failed_token_in_output(self, tool):
        """rc==0 short-circuits before any output pattern is even consulted."""
        output = 'FAILED tests/test_foo.py::test_bar - AssertionError\n'
        assert _classify(tool, output, 0, False) == FailureCategory.PASSED

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_infra_timeout_when_timed_out_true_no_output(self, tool):
        assert _classify(tool, '', 1, True) == FailureCategory.INFRA_TIMEOUT

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_infra_timeout_wins_over_matching_output_pattern(self, tool):
        """timed_out=True wins even when output matches a failure pattern —
        the root cause is the wall-clock limit, not the command output."""
        output = 'FAILED tests/test_foo.py::test_bar - AssertionError\n'
        assert _classify(tool, output, 1, True) == FailureCategory.INFRA_TIMEOUT


class TestOpaqueReproducesLegacyGenericLadder:
    """(c): classify_failure(ToolKind.OPAQUE, …) reproduces the full legacy
    _classify_failure generic ladder byte-identically — OPAQUE is the ONLY
    surviving dispatch target for this ladder (see verify_classify module
    docstring); every other ToolKind gets its own narrower table."""

    # compile_error: rustc diagnostic error codes / generic 'compile error'
    def test_compile_error_rustc_code(self):
        output = (
            'Compiling my-crate v0.1.0\nerror[E0308]: mismatched types\n  --> src/lib.rs:10:5\n'
        )
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.COMPILE_ERROR

    def test_compile_error_compile_error_string(self):
        output = 'compile error in foo.py line 5\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.COMPILE_ERROR

    # cargo_cli_error: narrow allowlist of grounded cargo-only prefixes
    def test_cargo_cli_error_exclude_pattern(self):
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: --exclude can only be used together with --workspace\n'
        )
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    def test_cargo_cli_error_no_such_subcommand(self):
        output = 'error: no such subcommand: `tset`\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    def test_cargo_cli_error_failed_to_parse_manifest(self):
        output = 'error: failed to parse manifest at `/x/Cargo.toml`\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    def test_cargo_cli_error_failed_to_compile(self):
        output = 'error: failed to compile `proc-macro-foo`\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    def test_cargo_cli_error_could_not_find(self):
        output = 'error: could not find `Cargo.toml` in `/path` or any parent directory\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    # cargo_cli_error negatives — un-grounded tokens must fall through
    def test_rustc_top_level_diagnostics_not_cargo_cli_error(self):
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: aborting due to previous errors\n'
            'error: could not compile `my-crate` (lib) due to previous error\n'
        )
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_rustc_invalid_diagnostic_not_cargo_cli_error(self):
        output = 'Compiling my-crate v0.1.0\nerror: invalid attribute value\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_failed_to_find_alone_not_cargo_cli_error(self):
        output = 'error: failed to find some-bin\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_package_prefix_not_cargo_cli_error(self):
        output = 'error: package `foo` cannot be found\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    # pytest_internalerror — checked before FAILED so a worker-death run with
    # collateral FAILED lines still classifies as infra, not drift
    def test_pytest_internalerror_wins_over_collateral_failed(self):
        output = (
            'FAILED orchestrator/tests/test_scheduler.py::TestScheduler::test_dispatch - '
            'collected err\n'
            'INTERNALERROR> Traceback (most recent call last):\n'
            'INTERNALERROR>     KeyError: <WorkerController gw3>\n'
        )
        assert _classify(ToolKind.OPAQUE, output, 3, False) == FailureCategory.PYTEST_INTERNALERROR

    # test_failure: rust test runner / pytest FAILED lines
    def test_test_failure_rust_test_runner(self):
        output = 'running 3 tests\ntest my::mod::it FAILED\ntest my::mod::another ... ok\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.TEST_FAILURE

    def test_test_failure_pytest_failed(self):
        output = 'FAILED tests/test_foo.py::test_bar - AssertionError\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.TEST_FAILURE

    # npm_error
    def test_npm_err_exclamation(self):
        output = 'npm ERR! code ELIFECYCLE\nnpm ERR! errno 1\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.NPM_ERROR

    def test_npm_error_lowercase(self):
        output = 'npm error peer dep missing: react@^18\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.NPM_ERROR

    # flock_error
    def test_flock_error_pattern(self):
        output = 'flock: failed to acquire lock on /var/lock/mylock\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.FLOCK_ERROR

    # tree_sitter_generate_error
    def test_tree_sitter_generate_error(self):
        output = 'Running tree-sitter generate\ntree-sitter generate failed: unexpected token\n'
        assert (
            _classify(ToolKind.OPAQUE, output, 1, False)
            == FailureCategory.TREE_SITTER_GENERATE_ERROR
        )

    # unknown_test_failure fallback
    def test_unknown_test_failure_fallback(self):
        output = 'Something went wrong but no recognizable pattern\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_unknown_test_failure_empty_output(self):
        assert _classify(ToolKind.OPAQUE, '', 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE


class TestReturnContract:
    """(d): classify_failure returns a real FailureCategory instance whose
    json.dumps output is byte-identical to the legacy plain-str category —
    FailureCategory is a StrEnum, so it IS its string value on the wire."""

    @pytest.mark.parametrize(
        ('output', 'rc', 'timed_out', 'expected'),
        [
            ('', 0, False, 'passed'),
            ('', 1, True, 'infra_timeout'),
            ('error[E0308]: unresolved import', 1, False, 'compile_error'),
            ('nothing matched', 1, False, 'unknown_test_failure'),
        ],
    )
    def test_isinstance_and_byte_identical_json(self, output, rc, timed_out, expected):
        result = _classify(ToolKind.OPAQUE, output, rc, timed_out)
        assert isinstance(result, FailureCategory)
        assert result == expected
        assert json.dumps({'category': result}) == json.dumps({'category': expected})


# Grounded shared-venv-mutation signatures (task 2048, re-grounded here for
# the PYTEST table): a concurrent `uv sync` from another orchestrator process
# on the shared .venv can transiently remove-then-readd packages WHILE a
# consumer is mid-pytest against it. See test_verify_env_transient.py's
# module docstring for the full task-2045 grounding narrative.
_XDIST_USAGE_ERROR_OUTPUT = (
    'usage: pytest [options] [file_or_dir] [file_or_dir] [...]\n'
    'pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
)
_PIP_ABSENT_RUNPY_OUTPUT = '/home/leo/src/dark-factory/.venv/bin/python3.12: No module named pip\n'
_MODULENOTFOUND_XDIST_OUTPUT = (
    'Traceback (most recent call last):\n'
    '  File "<string>", line 1, in <module>\n'
    "ModuleNotFoundError: No module named 'xdist'\n"
)
_MODULENOTFOUND_PYTEST_XDIST_OUTPUT = (
    'Traceback (most recent call last):\n'
    '  File "<string>", line 1, in <module>\n'
    "ModuleNotFoundError: No module named 'pytest_xdist'\n"
)


class TestPytestTable:
    """step-3: classify_failure(ToolKind.PYTEST, …) — env_transient checked
    FIRST (Invariant C1: consulted ONLY under PYTEST — see
    TestEnvTransientIsPytestScopedC1 below), then INTERNALERROR, then FAILED,
    then the unknown_test_failure fallback (covers pytest rc=5).

    RED today (pre step-4): the PYTEST branch doesn't exist yet, so every
    call here falls through to the shared OPAQUE placeholder in
    classify_failure. The env_transient assertions genuinely fail (the
    OPAQUE ladder never produces env_transient); the INTERNALERROR/FAILED
    assertions may already pass incidentally (their patterns are also part
    of today's OPAQUE placeholder ladder) but are asserted here regardless
    to pin the PYTEST table's own contract once step-4 gives PYTEST its own
    table.
    """

    def test_pytest_internalerror_wins_over_collateral_failed(self):
        output = (
            'FAILED orchestrator/tests/test_scheduler.py::TestScheduler::test_dispatch - '
            'collected err\n'
            'INTERNALERROR> Traceback (most recent call last):\n'
            'INTERNALERROR>     KeyError: <WorkerController gw3>\n'
        )
        assert _classify(ToolKind.PYTEST, output, 3, False) == FailureCategory.PYTEST_INTERNALERROR

    def test_test_failure_failed_line(self):
        output = 'FAILED tests/test_foo.py::test_bar - AssertionError\n'
        assert _classify(ToolKind.PYTEST, output, 1, False) == FailureCategory.TEST_FAILURE

    def test_xdist_usage_error_is_env_transient(self):
        assert (
            _classify(ToolKind.PYTEST, _XDIST_USAGE_ERROR_OUTPUT, 4, False)
            == FailureCategory.ENV_TRANSIENT
        )

    def test_pip_absent_runpy_line_is_env_transient(self):
        assert (
            _classify(ToolKind.PYTEST, _PIP_ABSENT_RUNPY_OUTPUT, 1, False)
            == FailureCategory.ENV_TRANSIENT
        )

    def test_modulenotfounderror_xdist_is_env_transient(self):
        assert (
            _classify(ToolKind.PYTEST, _MODULENOTFOUND_XDIST_OUTPUT, 1, False)
            == FailureCategory.ENV_TRANSIENT
        )

    def test_modulenotfounderror_pytest_xdist_is_env_transient(self):
        assert (
            _classify(ToolKind.PYTEST, _MODULENOTFOUND_PYTEST_XDIST_OUTPUT, 1, False)
            == FailureCategory.ENV_TRANSIENT
        )

    def test_quoted_pip_forms_still_env_transient(self):
        for output in (
            "ModuleNotFoundError: No module named 'pip'\n",
            'ModuleNotFoundError: No module named "pip"\n',
        ):
            assert _classify(ToolKind.PYTEST, output, 1, False) == FailureCategory.ENV_TRANSIENT, (
                output
            )

    def test_rc5_no_tests_ran_is_unknown_test_failure(self):
        """Invariant (b, task 1852): pytest rc=5 must stay RED, not PASSED."""
        output = '===== no tests ran in 0.01s =====\n'
        assert _classify(ToolKind.PYTEST, output, 5, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('module_name', ['pipx', 'pipenv', 'pip_audit', 'pip-tools'])
    def test_pip_prefixed_module_name_not_env_transient(self, module_name):
        """A module name that merely STARTS with 'pip' is a real import/code
        regression, not the pip-absence signature — word-boundary guard."""
        output = f"ModuleNotFoundError: No module named '{module_name}'\n"
        assert _classify(ToolKind.PYTEST, output, 1, False) != FailureCategory.ENV_TRANSIENT

    def test_pipeline_module_not_found_is_unknown_test_failure(self):
        output = "ModuleNotFoundError: No module named 'pipeline'\n"
        assert _classify(ToolKind.PYTEST, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_non_pytest_unrecognized_arguments_not_env_transient(self):
        """A different CLI tool's usage error captured in pytest's own output
        must not be swept into env_transient — anchored to the literal
        'pytest: error:' prefix argparse emits for pytest's own CLI."""
        output = (
            'usage: sometool [options]\n'
            'sometool: error: unrecognized arguments: -n --dist --max-worker-restart=0\n'
        )
        assert _classify(ToolKind.PYTEST, output, 4, False) == FailureCategory.UNKNOWN_TEST_FAILURE


class TestEnvTransientIsPytestScopedC1:
    """CRITICAL C1: env_transient is consulted ONLY under ToolKind.PYTEST —
    the structural win of per-tool dispatch. A pip/xdist-absence signature
    appearing in a NON-pytest tool's output must never classify env_transient,
    even though the exact same text would under ToolKind.PYTEST."""

    def test_cargo_output_with_pip_absence_is_not_env_transient(self):
        output = 'error: could not compile `my-crate`\nNo module named pip\n'
        result = _classify(ToolKind.CARGO_TEST, output, 1, False)
        assert result != FailureCategory.ENV_TRANSIENT, (
            f'cargo output must never classify env_transient (PYTEST-only), got {result!r}'
        )

    def test_pyright_output_with_pip_absence_is_not_env_transient(self):
        output = 'No module named pip\n'
        result = _classify(ToolKind.PYRIGHT, output, 1, False)
        assert result != FailureCategory.ENV_TRANSIENT, (
            f'pyright output must never classify env_transient (PYTEST-only), got {result!r}'
        )


# GOLDEN corpus: re-grounded from the historical cargo_cli_error allowlist
# fix commits 1703f86f95 (drop `package `), 18f57fe922 (drop `invalid `),
# 1aed67cd56 (tighten the allowlist), 264d5b5e8a (drop `find`, ground
# `compile`) — the cargo re-groundings of tasks 1103/1109/1116, already
# encoded as passing cases in test_verify.py's TestClassifyFailure. Re-
# asserted here through the tool-dispatched classifier with the SAME
# expected categories (not invented strings).
_CARGO_TOOL_KINDS = [ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY]


class TestCargoTable:
    """step-5: classify_failure(ToolKind.CARGO_TEST / CARGO_CLIPPY, …) — the
    GOLDEN cargo corpus.

    Most assertions here already hold via the shared OPAQUE placeholder
    fallthrough (step-2), since the placeholder ladder is a superset of the
    eventual cargo table's patterns — genuinely RED only once the cargo
    table stops reaching patterns that don't belong to it (see
    TestCargoTableIsolationC1 below). Kept here regardless to pin the
    CARGO_TEST/CARGO_CLIPPY table's full positive/negative contract once
    step-6 gives it its own table.
    """

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_exclude_pattern(self, tool):
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: --exclude can only be used together with --workspace\n'
        )
        assert _classify(tool, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_no_such_subcommand(self, tool):
        output = 'error: no such subcommand: `tset`\n'
        assert _classify(tool, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_failed_to_parse_manifest(self, tool):
        output = 'error: failed to parse manifest at `/x/Cargo.toml`\n'
        assert _classify(tool, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_failed_to_compile(self, tool):
        output = 'error: failed to compile `proc-macro-foo`\n'
        assert _classify(tool, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_could_not_find(self, tool):
        output = 'error: could not find `Cargo.toml` in `/path` or any parent directory\n'
        assert _classify(tool, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_rustc_top_level_diagnostics_not_cargo_cli_error(self, tool):
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: aborting due to previous errors\n'
            'error: could not compile `my-crate` (lib) due to previous error\n'
        )
        assert _classify(tool, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_rustc_invalid_diagnostic_not_cargo_cli_error(self, tool):
        output = 'Compiling my-crate v0.1.0\nerror: invalid attribute value\n'
        assert _classify(tool, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_failed_to_find_alone_not_cargo_cli_error(self, tool):
        output = 'error: failed to find some-bin\n'
        assert _classify(tool, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_package_prefix_not_cargo_cli_error(self, tool):
        output = 'error: package `foo` cannot be found\n'
        assert _classify(tool, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_compile_error_rustc_code(self, tool):
        output = 'error[E0308]: mismatched types\n  --> src/lib.rs:10:5\n'
        assert _classify(tool, output, 1, False) == FailureCategory.COMPILE_ERROR

    def test_test_failure_rust_test_runner(self):
        output = 'running 3 tests\ntest my::mod::it FAILED\ntest my::mod::another ... ok\n'
        assert _classify(ToolKind.CARGO_TEST, output, 1, False) == FailureCategory.TEST_FAILURE


class TestCargoTableIsolationC1:
    """CRITICAL C1: patterns that belong to OTHER tools' tables (pytest
    INTERNALERROR, npm errors) must not be reachable via ToolKind.CARGO_TEST
    — proving the cargo table is its own narrow list, not a continuation of
    the shared OPAQUE placeholder ladder.

    RED today (pre step-6): CARGO_TEST still falls through to the shared
    OPAQUE placeholder (step-2), which DOES contain these patterns.
    """

    def test_internalerror_not_reachable_via_cargo(self):
        output = 'INTERNALERROR> pytest crashed unexpectedly\n'
        result = _classify(ToolKind.CARGO_TEST, output, 1, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE, (
            f'pytest INTERNALERROR must not leak into the cargo table, got {result!r}'
        )

    def test_npm_error_not_reachable_via_cargo(self):
        output = 'npm ERR! code ELIFECYCLE\n'
        result = _classify(ToolKind.CARGO_TEST, output, 1, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE, (
            f'npm errors must not leak into the cargo table, got {result!r}'
        )


class TestNpxTable:
    """step-5: classify_failure(ToolKind.NPX, …) — npm_error, else fallback."""

    def test_npm_err_exclamation(self):
        output = 'npm ERR! code ELIFECYCLE\nnpm ERR! errno 1\n'
        assert _classify(ToolKind.NPX, output, 1, False) == FailureCategory.NPM_ERROR

    def test_npm_error_lowercase(self):
        output = 'npm error peer dep missing: react@^18\n'
        assert _classify(ToolKind.NPX, output, 1, False) == FailureCategory.NPM_ERROR

    def test_unrelated_output_is_unknown_test_failure(self):
        output = 'Something went wrong but no recognizable pattern\n'
        assert _classify(ToolKind.NPX, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE


class TestNpxTableIsolationC1:
    """CRITICAL C1: a rustc-shaped compile_error pattern must not be
    reachable via ToolKind.NPX — only npm_error/fallback belong there.

    RED today (pre step-6): NPX still falls through to the shared OPAQUE
    placeholder, which DOES contain the rustc error[Exxxx] pattern.
    """

    def test_rustc_style_error_not_reachable_via_npx(self):
        output = 'error[E0308]: mismatched types\n'
        result = _classify(ToolKind.NPX, output, 1, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE, (
            f'a rustc-shaped pattern must not leak into the NPX table, got {result!r}'
        )


class TestHeadlineC1ReverseSignal:
    """The PRD's headline C1 example: a cargo token embedded in PYTEST
    output can no longer swallow a pytest FAILED line, because
    ToolKind.PYTEST dispatches to its own table (step-4) which never
    consults the cargo_cli_error allowlist at all. Already green since
    step-4 — re-asserted here alongside the CARGO_TEST/NPX tables this step
    introduces, for symmetry with TestCargoTableIsolationC1/TestNpxTableIsolationC1.
    """

    def test_cargo_token_in_pytest_output_still_classifies_test_failure(self):
        output = 'error: no such subcommand: `tset`\nFAILED tests/test_x.py::test_y\n'
        result = _classify(ToolKind.PYTEST, output, 1, False)
        assert result == FailureCategory.TEST_FAILURE, (
            f'a cargo CLI token in pytest output must not swallow the FAILED line '
            f'into cargo_cli_error, got {result!r}'
        )


# step-7: structured-output parsing (Invariant C2). Fixtures are built from
# the documented JSON schemas, hand-verified against the pinned tool
# versions in this worktree (cargo 1.96.0 `--message-format json` NDJSON,
# pyright 1.1.408 `--outputjson`, ruff `--output-format json`) rather than
# invented shapes.


def _cargo_ndjson(*records: dict) -> str:
    """Join dict records as cargo-shaped NDJSON (one compact JSON object per line)."""
    return '\n'.join(json.dumps(r) for r in records)


# A `reason: compiler-message` record whose `message.level == 'error'` — but
# whose human-readable `rendered`/`message` text is deliberately chosen to
# match NONE of `_CARGO_PATTERNS` (no `error[E\d+]:` bracket form, no
# cargo-CLI allowlist prefix, no `FAILED`/npm/flock/tree-sitter token). This
# is the genuinely RED fixture: today it falls through the text table to
# unknown_test_failure — only a parser that reads the structured `level`
# field can classify it compile_error, which is the entire point of C2
# (structured output catches what the regex ladder would miss).
_CARGO_JSON_COMPILER_ERROR = _cargo_ndjson(
    {
        'reason': 'compiler-message',
        'package_id': 'path+file:///workspace/my_crate#0.1.0',
        'manifest_path': 'Cargo.toml',
        'target': {
            'kind': ['lib'],
            'crate_types': ['lib'],
            'name': 'my_crate',
            'src_path': 'src/lib.rs',
            'edition': '2021',
            'doc': True,
            'doctest': True,
            'test': True,
        },
        'message': {
            'rendered': 'error: cannot find value `x` in this scope\n --> src/lib.rs:2:5\n',
            '$message_type': 'diagnostic',
            'children': [],
            'level': 'error',
            'message': 'cannot find value `x` in this scope',
            'spans': [
                {
                    'file_name': 'src/lib.rs',
                    'line_start': 2,
                    'line_end': 2,
                    'column_start': 5,
                    'column_end': 6,
                    'is_primary': True,
                    'label': 'not found in this scope',
                }
            ],
            'code': {'code': 'E0425', 'explanation': None},
        },
    },
    {'reason': 'build-finished', 'success': False},
)

# pyright --outputjson (verified schema, pyright 1.1.408): a top-level object
# with a `generalDiagnostics` array; each entry's `severity` is 'error' /
# 'warning' / 'information'.
_PYRIGHT_JSON_ERROR_DIAGNOSTIC = json.dumps(
    {
        'version': '1.1.408',
        'time': '1783914788715',
        'generalDiagnostics': [
            {
                'file': '/workspace/bad.py',
                'severity': 'error',
                'message': (
                    'Type "Literal[\'not an int\']" is not assignable to return type "int"'
                ),
                'range': {
                    'start': {'line': 1, 'character': 11},
                    'end': {'line': 1, 'character': 23},
                },
                'rule': 'reportReturnType',
            }
        ],
        'summary': {
            'filesAnalyzed': 1,
            'errorCount': 1,
            'warningCount': 0,
            'informationCount': 0,
            'timeInSec': 0.5,
        },
    }
)

# ruff --output-format json (verified schema): a top-level ARRAY of violation
# objects (not wrapped in an object) — each entry carries its own 'severity'.
_RUFF_JSON_VIOLATIONS = json.dumps(
    [
        {
            'cell': None,
            'code': 'F821',
            'end_location': {'column': 15, 'row': 4},
            'filename': '/workspace/bad.py',
            'fix': None,
            'location': {'column': 1, 'row': 4},
            'message': 'Undefined name `undefined_name`',
            'noqa_row': 4,
            'severity': 'error',
            'url': 'https://docs.astral.sh/ruff/rules/undefined-name',
        }
    ]
)


class TestStructuredOutputParsingC2:
    """step-7: Invariant C2 — where a tool offers structured output, the
    classifier parses it directly instead of regex-matching human text.

    The genuinely RED assertion (pre step-8) is
    ``test_cargo_json_compiler_message_error_is_compile_error``: its fixture
    is a `reason: compiler-message` / `level: error` record whose human text
    matches none of `_CARGO_PATTERNS`, so today it falls through to
    unknown_test_failure — only a JSON-aware parser that reads the
    structured `level` field can classify it compile_error. The pyright/ruff
    'preserved mapping' assertions and the non-JSON-fallback assertions
    already hold today (PYRIGHT/RUFF have no dedicated text table yet, so
    they fall through the shared OPAQUE placeholder to unknown_test_failure
    regardless of JSON-awareness — see classify_failure's docstring); kept
    here to pin the full JSON-first contract once step-8 adds the parsers,
    per Invariant C2's "falls back to the text table otherwise" / exception-
    safety requirements.
    """

    def test_cargo_json_compiler_message_error_is_compile_error(self):
        assert (
            _classify(ToolKind.CARGO_TEST, _CARGO_JSON_COMPILER_ERROR, 1, False)
            == FailureCategory.COMPILE_ERROR
        )

    def test_pyright_json_error_diagnostic_is_unknown_test_failure(self):
        """Preserved mapping: no dedicated PYRIGHT category exists (out of
        scope for this task) — an error-severity generalDiagnostic still maps
        to unknown_test_failure, matching today's on-the-wire category."""
        assert (
            _classify(ToolKind.PYRIGHT, _PYRIGHT_JSON_ERROR_DIAGNOSTIC, 1, False)
            == FailureCategory.UNKNOWN_TEST_FAILURE
        )

    def test_ruff_json_violation_is_unknown_test_failure(self):
        """Preserved mapping: no dedicated RUFF category exists (out of scope
        for this task) — a violations array still maps to
        unknown_test_failure, matching today's on-the-wire category."""
        assert (
            _classify(ToolKind.RUFF, _RUFF_JSON_VIOLATIONS, 1, False)
            == FailureCategory.UNKNOWN_TEST_FAILURE
        )

    def test_cargo_non_json_output_still_uses_text_table(self):
        """A plain-text (non-JSON) cargo failure must still classify via the
        existing _CARGO_PATTERNS text table — the JSON-first codepath must
        not break the text fallback."""
        output = 'error: no such subcommand: `tset`\n'
        assert _classify(ToolKind.CARGO_TEST, output, 1, False) == FailureCategory.CARGO_CLI_ERROR

    def test_pyright_non_json_output_falls_back_without_crash(self):
        output = (
            '/workspace/bad.py:2:12 - error: Type "Literal[\'not an int\']" is not '
            'assignable to return type "int" (reportReturnType)\n'
            '1 error, 0 warnings, 0 informations\n'
        )
        assert (
            _classify(ToolKind.PYRIGHT, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE
        )

    def test_ruff_non_json_output_falls_back_without_crash(self):
        output = 'bad.py:4:1: F821 Undefined name `undefined_name`\nFound 1 error.\n'
        assert _classify(ToolKind.RUFF, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE

    def test_cargo_malformed_json_falls_back_to_text_table(self):
        """A JSON-shaped-but-invalid cargo output (e.g. truncated mid-stream)
        must not crash the classifier — parsing is best-effort and
        exception-safe (Invariant C2), falling back to the text table."""
        output = (
            '{"reason": "compiler-message", "message": {"level": "error", '
            'THIS IS NOT VALID JSON\n'
            'error: no such subcommand: `tset`\n'
        )
        assert _classify(ToolKind.CARGO_TEST, output, 1, False) == FailureCategory.CARGO_CLI_ERROR


# step-9: verify._summarize_checks must thread the per-check config command
# (test_cmd/lint_cmd/type_cmd) into classify_failure as that check's
# ToolKind, instead of discarding tool identity (today's tool-blind
# `_classify_failure(out, rc, to)` call). RED today: the OLD
# `_summarize_checks` takes exactly 9 positional args (test_rc, test_out,
# test_timed_out, lint_rc, lint_out, lint_timed_out, type_rc, type_out,
# type_timed_out) — no `*_cmd` params — so calling it with the NEW 12-arg
# signature (each check's rc/out/timed_out immediately followed by that
# check's cmd) raises TypeError before the body even runs.


def _summarize(*args):
    from orchestrator.verify import _summarize_checks  # noqa: PLC0415

    return _summarize_checks(*args)


class TestSummarizeChecksThreadsToolIdentity:
    """step-9: `_summarize_checks` (verify.py) must resolve each check's
    ToolKind from its config command (via `_tool_for_cmd`/
    `parse_config_command` — step-10) and thread it into `classify_failure`,
    rather than calling the tool-blind classifier. Proves the CRITICAL C1
    signal one layer up the stack from TestHeadlineC1ReverseSignal: a REAL
    run_verification call site, not just the bare classifier.
    """

    def test_pytest_tool_identity_c1_cargo_token_does_not_swallow_pytest_failure(self):
        """A failing test leg whose output mixes a cargo CLI token and a
        pytest FAILED line must classify as test_failure (NOT
        cargo_cli_error): `_summarize_checks` must resolve `test_cmd` to
        ToolKind.PYTEST and thread it into classify_failure, so the
        cargo_cli_error allowlist (only reachable via CARGO_TEST/CARGO_CLIPPY/
        OPAQUE) is never consulted for this check."""
        test_cmd = 'uv run --project orchestrator --directory orchestrator pytest tests/'
        test_out = 'error: no such subcommand: `tset`\nFAILED tests/test_x.py::test_y\n'
        passed, category, cause_hint, summary = _summarize(
            1, test_out, False, test_cmd,
            0, '', False, None,
            0, '', False, None,
        )
        assert not passed
        assert category == FailureCategory.TEST_FAILURE

    def test_lint_chain_dispatches_opaque_generic_ladder(self):
        """A lint command that is a raw-retained `&&`-chain (neither pytest
        nor a cargo test/clippy chain) resolves to ToolKind.OPAQUE via
        `_tool_for_cmd` — `_summarize_checks` still classifies its failure
        through the full legacy generic ladder, byte-identical to today's
        tool-blind `_classify_failure`."""
        lint_cmd = 'ruff check . && mypy src/'
        lint_out = 'flock: failed to acquire lock on /var/lock/mylock\n'
        passed, category, cause_hint, summary = _summarize(
            0, '', False, None,
            1, lint_out, False, lint_cmd,
            0, '', False, None,
        )
        assert not passed
        assert category == FailureCategory.FLOCK_ERROR
