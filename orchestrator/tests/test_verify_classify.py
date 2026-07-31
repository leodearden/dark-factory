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

from orchestrator import verify_classify
from orchestrator.verify_categories import INFRA_TRANSIENT_CATEGORIES, FailureCategory
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


# ---------------------------------------------------------------------------
# PART 2 (task 2549): host-infrastructure pre-dispatch guard —
# _classify_environmental(output), checked immediately after the
# timed_out->INFRA_TIMEOUT guard and before per-tool dispatch. ENOSPC (incl.
# linker SIGBUS-on-full-disk) and flock/semaphore slot-acquisition timeouts
# previously had no category at all and fell through to whichever code-fault
# category each tool's table happened to match first. Both conditions are
# tool-blind (a full disk means the same thing regardless of which tool hit
# it), so the guard must fire identically for every ToolKind — mirrors
# TestGuardsApplyToEveryToolKind above.
#
# RED today: neither _classify_environmental nor its patterns exist yet, so
# every positive case below falls through to per-tool dispatch and lands on
# whatever category that tool's table matches first (or unknown_test_failure).
# ---------------------------------------------------------------------------

_DISK_FULL_ENOSPC_OUTPUT = 'error: No space left on device (os error 28)\n'

_DISK_FULL_LINKER_SIGBUS_OUTPUT = (
    'error: linking with `cc` failed: exit status: 1\n'
    '  = note: collect2: fatal error: ld terminated with signal 7 [Bus error], core dumped\n'
    '  = note: No space left on device (os error 28)\n'
)

_SEMAPHORE_TIMEOUT_FLOCK_OUTPUT = 'flock: timeout while waiting to acquire lock\n'

_SEMAPHORE_TIMEOUT_SLOT_OUTPUT = 'Timeout waiting for semaphore slot after 300s\n'


class TestEnvironmentalGuardsApplyToEveryToolKind:
    """PART 2 (task 2549): the new tool-blind environmental pre-dispatch
    guard fires identically for every ToolKind — mirrors
    TestGuardsApplyToEveryToolKind's rc==0/timed_out coverage, but for the
    DISK_FULL/SEMAPHORE_TIMEOUT host-infrastructure categories."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_enospc_is_disk_full(self, tool):
        assert _classify(tool, _DISK_FULL_ENOSPC_OUTPUT, 1, False) == FailureCategory.DISK_FULL

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_linker_sigbus_on_full_disk_is_disk_full(self, tool):
        assert (
            _classify(tool, _DISK_FULL_LINKER_SIGBUS_OUTPUT, 1, False)
            == FailureCategory.DISK_FULL
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_flock_timeout_is_semaphore_timeout(self, tool):
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_FLOCK_OUTPUT, 1, False)
            == FailureCategory.SEMAPHORE_TIMEOUT
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_semaphore_slot_timeout_is_semaphore_timeout(self, tool):
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_SLOT_OUTPUT, 1, False)
            == FailureCategory.SEMAPHORE_TIMEOUT
        )


class TestEnvironmentalGuardNegatives:
    """Negatives that must NOT regress once the environmental guard is wired
    in: a plain flock-contention error (no timeout token) still classifies
    FLOCK_ERROR, and a bare 'SIGBUS' mention with no disk/linker context does
    not false-positive into DISK_FULL."""

    def test_flock_failed_to_acquire_lock_stays_flock_error(self):
        """The existing OPAQUE golden — no timeout token present, so the
        narrower SEMAPHORE_TIMEOUT guard must not shadow the pre-existing
        FLOCK_ERROR pattern."""
        output = 'flock: failed to acquire lock on /var/lock/mylock\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.FLOCK_ERROR

    def test_bare_sigbus_mention_without_disk_context_not_disk_full(self):
        output = 'test_signal_handling: caught SIGBUS in signal handler test\n'
        result = _classify(ToolKind.OPAQUE, output, 1, False)
        assert result != FailureCategory.DISK_FULL, (
            f'a bare SIGBUS mention with no disk/linker context must not '
            f'false-positive into disk_full, got {result!r}'
        )


# ---------------------------------------------------------------------------
# task 2748: SEMAPHORE_TIMEOUT deterministic-diagnostic veto. A genuine
# flock/semaphore slot-acquisition timeout is raised by the concurrency-
# limiter WRAPPER (reify lib_slot_acquire.sh / `flock -w`) BEFORE the wrapped
# tool ever runs, so its output cannot ALSO carry a compiler/lint diagnostic.
# But `_LOCK_TOKEN_RE` and `_TIMEOUT_TOKEN_RE` are two independent
# whole-output searches, so a deterministic cargo clippy/rustc diagnostic
# that happens to quote lock/slot/semaphore source and mention "timed out" in
# a note satisfies both and is misclassified SEMAPHORE_TIMEOUT — an
# INFRA_TRANSIENT_CATEGORIES member — which sends a deterministic lint/type
# regression through the bounded infra-retry loop instead of the debugger.
# Mirrors TestEnvironmentalGuardsApplyToEveryToolKind's tool-blind coverage.
#
# RED today: no deterministic-diagnostic veto exists yet, so both goldens
# below still satisfy the loose lock+timeout co-occurrence heuristic and
# classify SEMAPHORE_TIMEOUT (a member of INFRA_TRANSIENT_CATEGORIES).
# ---------------------------------------------------------------------------

_SEMAPHORE_TIMEOUT_CLIPPY_DIAGNOSTIC_OUTPUT = (
    'error: this `.lock()` call is held across an `.await` point\n'
    '  --> src/worker/pool.rs:42:19\n'
    '   |\n'
    '42 |         let _slot = self.semaphore.lock().await;\n'
    '   |                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
    '   |\n'
    "   = note: `-D clippy::await-holding-lock` implied by `-D warnings`\n"
    '   = help: for further information visit '
    'https://rust-lang.github.io/rust-clippy/master/index.html#await_holding_lock\n'
    '   = note: a peer that timed out waiting for the slot may hold the lock '
    'indefinitely\n'
    '\n'
    'error: aborting due to previous error\n'
)

# rustc compile-error golden: starts with a rustc error code (error[E0308]:),
# quotes lock/slot/semaphore source and mentions "timed out" in a note, but
# carries NO `clippy::` token — proves the veto must also recognize a bare
# rustc diagnostic marker, not just clippy's.
_SEMAPHORE_TIMEOUT_RUSTC_DIAGNOSTIC_OUTPUT = (
    'error[E0308]: mismatched types\n'
    '  --> src/worker/pool.rs:42:20\n'
    '   |\n'
    "42 |         let guard: MutexGuard<'_, Slot> = self.semaphore.lock();\n"
    "   |                    ^^^^^^^^^^^^^^^^^^^^ expected `Slot`, found "
    "`MutexGuard<'_, Slot>`\n"
    '   |\n'
    '   = note: a peer that timed out waiting for the lock may have left the '
    'semaphore in an inconsistent state\n'
    '\n'
    'error: aborting due to previous error\n'
)


class TestSemaphoreTimeoutDeterministicDiagnosticVeto:
    """task 2748: a deterministic Rust compiler/clippy diagnostic must veto
    the loose lock+timeout co-occurrence heuristic — the lock/slot/semaphore
    and "timed out" tokens are incidental to a deterministic lint/compile
    failure, not evidence of a genuine slot-acquisition timeout (which is
    raised by the wrapper BEFORE the wrapped tool runs, so it can never carry
    a diagnostic marker)."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_clippy_lint_diagnostic_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        result = _classify(tool, _SEMAPHORE_TIMEOUT_CLIPPY_DIAGNOSTIC_OUTPUT, 1, False)
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a deterministic clippy lint diagnostic (clippy:: marker present) '
            f'with incidental lock/slot/semaphore + "timed out" tokens must not '
            f'classify infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_rustc_compile_error_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        result = _classify(tool, _SEMAPHORE_TIMEOUT_RUSTC_DIAGNOSTIC_OUTPUT, 1, False)
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a deterministic rustc compile-error diagnostic (rustc error-code '
            f'marker present, no clippy:: marker) with incidental '
            f'lock/slot/semaphore + "timed out" tokens must not classify '
            f'infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize(
        'tool', [ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY, ToolKind.OPAQUE]
    )
    def test_rustc_compile_error_with_incidental_lock_and_timeout_tokens_is_compile_error(
        self, tool
    ):
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_RUSTC_DIAGNOSTIC_OUTPUT, 1, False)
            == FailureCategory.COMPILE_ERROR
        )


# ---------------------------------------------------------------------------
# task 2821: SEMAPHORE_TIMEOUT deterministic-TEST-VERDICT veto (gate-hole #1
# of the reify 2026-07-19 red-main incident, /deb deb-reify-964887).
# Generalizes the task-2748 Rust-diagnostic veto directly above: a genuine
# flock/semaphore slot-acquisition timeout is raised by the concurrency-
# limiter WRAPPER (reify lib_slot_acquire.sh / `flock -w`) BEFORE the wrapped
# tool ever runs, so its output cannot ALSO carry a pytest/cargo test-runner
# VERDICT (a "N passed, M failed" count-pair, or a FAILED line). But
# `_LOCK_TOKEN_RE`/`_TIMEOUT_TOKEN_RE` are two independent whole-output
# searches, so a deterministic test failure that happens to quote
# lock/slot/semaphore source and mention "timed out" in a note satisfies both
# and is misclassified SEMAPHORE_TIMEOUT — an INFRA_TRANSIENT_CATEGORIES
# member — sending a deterministic test regression through the bounded
# infra-retry loop (silently re-verified against a deterministically-red
# gate) instead of surfacing to the debugger. Mirrors
# TestSemaphoreTimeoutDeterministicDiagnosticVeto's shape, generalized from
# Rust compiler/clippy diagnostics to any test-runner verdict.
#
# RED today: no test-verdict veto exists yet, so both goldens below still
# satisfy the loose lock+timeout co-occurrence heuristic and classify
# SEMAPHORE_TIMEOUT (a member of INFRA_TRANSIENT_CATEGORIES).
# ---------------------------------------------------------------------------

# pytest-shaped golden: leading-form FAILED line (pytest-report-shaped only),
# a note mentioning the "slot" that "timed out", and pytest's bracketed
# short-summary with the failed-before-passed order pytest actually uses.
_SEMAPHORE_TIMEOUT_PYTEST_VERDICT_OUTPUT = (
    'FAILED tests/test_registry_drift.py::test_registry_pin - AssertionError\n'
    'note: a stale peer that timed out may hold the slot\n'
    '===== 6 failed, 48 passed in 12.34s =====\n'
)

# cargo-shaped golden: trailing-form FAILED line (cargo's own test-runner
# shape), a note mentioning the "semaphore" slot that "timed out", and
# cargo's literal "test result: FAILED. 48 passed; 6 failed; 0 ignored"
# verdict line (passed-before-failed order — the reverse of the pytest
# golden above, so both count-pair orders are grounded, matching the
# incident's literal "48 passed, 6 failed").
_SEMAPHORE_TIMEOUT_CARGO_VERDICT_OUTPUT = (
    'test families::non_geometry_families_disjoint ... FAILED\n'
    'note: a peer holding the semaphore slot had timed out\n'
    'test result: FAILED. 48 passed; 6 failed; 0 ignored\n'
)

# ZERO-PASS pytest shape: when every collected test fails, pytest omits the
# "0 passed" token entirely, so the ONLY verdict is the bracketed
# failure-count short-summary ("===== 6 failed in 1.20s ====="). Deliberately
# carries NO leading/trailing FAILED line and NO "N passed"/"M failed" count
# PAIR (no "passed" token anywhere), so step-2's `_has_deterministic_test_
# verdict` (count-pair + FAILED-line markers only) does not yet recognize it
# as a verdict — RED until step-4 adds a dedicated bracketed-summary marker.
_SEMAPHORE_TIMEOUT_PYTEST_ZERO_PASS_OUTPUT = (
    'note: a stale peer that timed out may hold the slot\n'
    '===== 6 failed in 1.20s =====\n'
)

# Count-pair-ONLY goldens (review follow-up): both grounded verdict goldens
# above ALSO contain a FAILED line, so `_TEST_FAILURE_LEADING_RE`/
# `_TEST_FAILURE_TRAILING_RE` already satisfy `_has_deterministic_test_verdict`
# on their own — `_TEST_VERDICT_PAIR_RE` (the "N passed, M failed" count-pair
# matcher, the flagship marker this task adds) is never independently
# exercised by them; it could be deleted, or either alternation order broken,
# and every existing test would still pass. These two goldens carry ONLY a
# bare count pair — no FAILED line, no `=====` banner (which would itself
# satisfy `_PYTEST_FAILURE_SUMMARY_RE`), no " in <float>s" timing suffix — so
# only `_TEST_VERDICT_PAIR_RE` can account for the veto, covering both
# alternation orders.
_SEMAPHORE_TIMEOUT_VERDICT_PAIR_ONLY_FAILED_FIRST_OUTPUT = (
    'note: a stale peer that timed out may hold the slot\n'
    'summary: 6 failed, 48 passed\n'
)

_SEMAPHORE_TIMEOUT_VERDICT_PAIR_ONLY_PASSED_FIRST_OUTPUT = (
    'note: a peer holding the semaphore slot had timed out\n'
    'test result: FAILED. 48 passed; 6 failed; 0 ignored\n'
)


class TestSemaphoreTimeoutDeterministicTestVerdictVeto:
    """task 2821: a deterministic pytest/cargo test-runner VERDICT must veto
    the loose lock+timeout co-occurrence heuristic exactly like task 2748's
    Rust-diagnostic veto — the lock/slot/semaphore and "timed out" tokens are
    incidental to a deterministic test failure, not evidence of a genuine
    slot-acquisition timeout (which is raised by the wrapper BEFORE the
    wrapped tool runs, so it can never carry a pass/fail verdict)."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_pytest_verdict_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        result = _classify(tool, _SEMAPHORE_TIMEOUT_PYTEST_VERDICT_OUTPUT, 1, False)
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a deterministic pytest test-verdict ("N failed, M passed" summary '
            f'present) with incidental lock/slot/semaphore + "timed out" tokens '
            f'must not classify infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_cargo_verdict_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        result = _classify(tool, _SEMAPHORE_TIMEOUT_CARGO_VERDICT_OUTPUT, 1, False)
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a deterministic cargo test-verdict ("N passed; M failed" summary '
            f'present) with incidental lock/slot/semaphore + "timed out" tokens '
            f'must not classify infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize('tool', [ToolKind.PYTEST, ToolKind.OPAQUE])
    def test_pytest_verdict_leading_failed_line_is_test_failure(self, tool):
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_PYTEST_VERDICT_OUTPUT, 1, False)
            == FailureCategory.TEST_FAILURE
        )

    @pytest.mark.parametrize(
        'tool',
        [ToolKind.PYTEST, ToolKind.OPAQUE, ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY],
    )
    def test_cargo_verdict_trailing_failed_line_is_test_failure(self, tool):
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_CARGO_VERDICT_OUTPUT, 1, False)
            == FailureCategory.TEST_FAILURE
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_pytest_zero_pass_verdict_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        result = _classify(tool, _SEMAPHORE_TIMEOUT_PYTEST_ZERO_PASS_OUTPUT, 1, False)
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a deterministic pytest zero-pass verdict (bracketed "N failed in '
            f'Ts" short-summary present, no "passed" token) with incidental '
            f'lock/slot/semaphore + "timed out" tokens must not classify '
            f'infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize('tool', [ToolKind.PYTEST, ToolKind.OPAQUE])
    def test_pytest_zero_pass_verdict_is_unknown_test_failure(self, tool):
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_PYTEST_ZERO_PASS_OUTPUT, 1, False)
            == FailureCategory.UNKNOWN_TEST_FAILURE
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_verdict_pair_only_failed_first_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        """Isolates `_TEST_VERDICT_PAIR_RE`'s failed-before-passed
        alternation: no FAILED line, no bracketed banner, no timing suffix —
        only the bare count pair can account for the veto."""
        result = _classify(
            tool, _SEMAPHORE_TIMEOUT_VERDICT_PAIR_ONLY_FAILED_FIRST_OUTPUT, 1, False
        )
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a bare "N failed, M passed" count pair (no FAILED line, no '
            f'banner) with incidental lock/slot/semaphore + "timed out" '
            f'tokens must not classify infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_verdict_pair_only_passed_first_with_incidental_lock_and_timeout_tokens_is_not_infra_transient(
        self, tool
    ):
        """Isolates `_TEST_VERDICT_PAIR_RE`'s passed-before-failed
        alternation (cargo's own summary order): no per-test FAILED line, no
        bracketed banner — only the bare count pair can account for the
        veto."""
        result = _classify(
            tool, _SEMAPHORE_TIMEOUT_VERDICT_PAIR_ONLY_PASSED_FIRST_OUTPUT, 1, False
        )
        assert result not in INFRA_TRANSIENT_CATEGORIES, (
            f'a bare "N passed; M failed" count pair (no FAILED line, no '
            f'banner) with incidental lock/slot/semaphore + "timed out" '
            f'tokens must not classify infra-transient, got {result!r}'
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_verdict_pair_only_goldens_are_unknown_test_failure(self, tool):
        """Pins the exact category too, not just "not infra-transient": with
        no FAILED line present, per-tool dispatch has nothing else to match,
        so both isolated-pair goldens fall through to UNKNOWN_TEST_FAILURE."""
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_VERDICT_PAIR_ONLY_FAILED_FIRST_OUTPUT, 1, False)
            == FailureCategory.UNKNOWN_TEST_FAILURE
        )
        assert (
            _classify(tool, _SEMAPHORE_TIMEOUT_VERDICT_PAIR_ONLY_PASSED_FIRST_OUTPUT, 1, False)
            == FailureCategory.UNKNOWN_TEST_FAILURE
        )


class TestSemaphoreTimeoutDeterministicTestVerdictVetoNegatives:
    """Regression negatives: pure-wrapper genuine timeouts carrying no test
    verdict must stay SEMAPHORE_TIMEOUT, both before and after the task-2821
    veto — a veto only ever removes false-positives."""

    def test_flock_output_stays_semaphore_timeout(self):
        assert (
            _classify(ToolKind.OPAQUE, _SEMAPHORE_TIMEOUT_FLOCK_OUTPUT, 1, False)
            == FailureCategory.SEMAPHORE_TIMEOUT
        )

    def test_slot_output_stays_semaphore_timeout(self):
        assert (
            _classify(ToolKind.OPAQUE, _SEMAPHORE_TIMEOUT_SLOT_OUTPUT, 1, False)
            == FailureCategory.SEMAPHORE_TIMEOUT
        )

    def test_stray_number_word_without_verdict_pair_stays_semaphore_timeout(self):
        """A lone 'N <word>' count with no paired failed/error count must not
        false-positive into the new verdict-pair veto."""
        output = (
            'Timeout after 300 seconds; 0 slots passed the wait — semaphore '
            'slot timed out\n'
        )
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.SEMAPHORE_TIMEOUT

    def test_bracketed_banner_without_failure_count_stays_semaphore_timeout(self):
        """A genuine wrapper banner using the same `=====`-bracket punctuation
        as pytest's short-summary, but with NO numeric failure count, must not
        false-positive into the step-4 bracketed-summary veto marker."""
        output = '===== semaphore slot wait timed out after 300s =====\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.SEMAPHORE_TIMEOUT

    def test_bare_errors_in_seconds_wrapper_line_stays_semaphore_timeout(self):
        """Review follow-up: `_PYTEST_FAILURE_SUMMARY_RE`'s bare trailing
        'N (failed|errors) ... in <float>s' fallback (considered and dropped)
        was not anchored to pytest's own `=====` banner, so a wrapper-shaped
        message like this one would have false-positive-vetoed a genuine
        SEMAPHORE_TIMEOUT."""
        output = '2 errors in 30s while waiting for the slot that timed out\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.SEMAPHORE_TIMEOUT

    def test_uppercase_trailing_failed_wrapper_line_without_test_context_stays_semaphore_timeout(
        self,
    ):
        """Review follow-up: `_TEST_FAILURE_TRAILING_RE` matches ANY line
        ending in ' FAILED', including an incidental wrapper log line with no
        test runner involved at all. Since this evidence is consulted inside
        the SEMAPHORE_TIMEOUT arm — i.e. only once lock/slot/semaphore +
        "timed out" tokens are already present — an unanchored match would
        wrongly veto a genuine slot timeout, which would make the "a veto
        only ever removes false-positives" claim above false.
        `_has_deterministic_test_verdict` additionally requires the FAILED
        line to carry a test-runner-shaped 'test'/'::' token, which this
        pure-wrapper line does not."""
        output = 'flock: semaphore slot acquisition timed out; wait FAILED\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.SEMAPHORE_TIMEOUT

    def test_leading_failed_wrapper_line_without_test_context_stays_semaphore_timeout(self):
        """Same as above for the leading-form marker
        (`_TEST_FAILURE_LEADING_RE`, '^FAILED\\s')."""
        output = 'FAILED to release semaphore slot before it timed out\n'
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.SEMAPHORE_TIMEOUT


# ---------------------------------------------------------------------------
# task 2756: broken _merge-verify worktree (unreadable/missing Cargo.lock) is
# a HOST/ENVIRONMENT condition, not a per-tool output pattern — exactly like
# DISK_FULL/SEMAPHORE_TIMEOUT above. Grounded in the reify 5120 RC-2 07-13
# sample: check-manifold-deps.sh (an OPAQUE guard script) failed with
# "could not read manifold-csg-sys version from .../_merge-verify/Cargo.lock"
# because the ephemeral merge-verify worktree's own Cargo.lock was
# unreadable — a broken verify ENVIRONMENT, not a branch fault. Mirrors
# TestEnvironmentalGuardsApplyToEveryToolKind's tool-blind coverage.
#
# RED today: no broken-verify-worktree signature exists yet, so the golden
# below falls through _classify_environmental (no ENOSPC/linker/semaphore
# match) and per-tool dispatch (no table matches this shape either) to
# UNKNOWN_TEST_FAILURE instead of ENV_TRANSIENT.
# ---------------------------------------------------------------------------

_VERIFY_WORKTREE_BROKEN_CARGO_LOCK_OUTPUT = (
    'check-manifold-deps.sh: could not read manifold-csg-sys version from '
    '/work/_merge-verify/Cargo.lock\n'
)


class TestVerifyWorktreeBrokenEnvGuard:
    """task 2756: a broken _merge-verify worktree's unreadable Cargo.lock
    classifies ENV_TRANSIENT — tool-blind, like DISK_FULL/SEMAPHORE_TIMEOUT —
    so the merge_queue's transient-infra hold (INFRA_TRANSIENT_CATEGORIES)
    fires instead of a silent generic block."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_broken_verify_worktree_cargo_lock_is_env_transient(self, tool):
        assert (
            _classify(tool, _VERIFY_WORKTREE_BROKEN_CARGO_LOCK_OUTPUT, 1, False)
            == FailureCategory.ENV_TRANSIENT
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_broken_verify_worktree_is_infra_transient(self, tool):
        """Observable signal #1: the unit-level proxy that this output will
        hit the merge_queue.py:1886 transient-infra gate."""
        result = _classify(tool, _VERIFY_WORKTREE_BROKEN_CARGO_LOCK_OUTPUT, 1, False)
        assert result in INFRA_TRANSIENT_CATEGORIES


class TestVerifyWorktreeBrokenEnvGuardNegatives:
    """Narrowness / ordering regression guards for the broken-verify-worktree
    signature — green both before and after the impl."""

    def test_genuine_cargo_lock_mention_not_env_transient(self):
        """A real rustc compile fault that merely mentions Cargo.lock on a
        line WITHOUT a read-failure verb must not be swallowed by the new
        pattern — proves the pattern does not shadow genuine branch faults."""
        output = (
            'error[E0308]: mismatched types\n'
            '  --> src/lib.rs:10:5\n'
            'note: Cargo.lock is up to date\n'
        )
        assert (
            _classify(ToolKind.CARGO_TEST, output, 1, False) == FailureCategory.COMPILE_ERROR
        )

    def test_read_failure_without_cargo_lock_not_env_transient(self):
        """A read failure that doesn't mention Cargo.lock must not classify
        ENV_TRANSIENT — proves the pattern requires the Cargo.lock lockfile
        token, not any read failure."""
        output = 'could not read /work/_merge-verify/config/settings.toml\n'
        assert (
            _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.UNKNOWN_TEST_FAILURE
        )

    def test_enospc_with_unreadable_lock_stays_disk_full(self):
        """The more specific DISK_FULL root cause wins on co-occurrence —
        the ENOSPC check is ordered first in _classify_environmental."""
        output = _DISK_FULL_ENOSPC_OUTPUT + _VERIFY_WORKTREE_BROKEN_CARGO_LOCK_OUTPUT
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.DISK_FULL


# ---------------------------------------------------------------------------
# task 2831: _merge-verify restart-collateral shapes (reify merge-gate RCA,
# 2026-07-19) — a WIDER class of broken-worktree signal than task 2756's
# single Cargo.lock read-failure above. When the ephemeral _merge-verify
# worktree is removed out from under a running verify, the guard/wrapper/
# shell emits one of five documented collateral shapes, none of which are a
# branch fault — exactly like the task-2756 Cargo.lock signature, this is a
# HOST/ENVIRONMENT condition. Mirrors TestVerifyWorktreeBrokenEnvGuard's
# tool-blind coverage.
#
# RED today: none of the five collateral patterns exist yet, so every
# positive below falls through _classify_environmental (no ENOSPC/linker/
# semaphore/Cargo.lock match) and per-tool dispatch (no table matches this
# shape either) to UNKNOWN_TEST_FAILURE instead of ENV_TRANSIENT.
# ---------------------------------------------------------------------------

# shape 1: a read-failure of some OTHER tracked file (not Cargo.lock) using
# the "couldn't" contraction — the RCA's own grounded phrasing, which the
# existing task-2756 verb alternation (could not/cannot/can't/unable to/
# failed to) does not yet include.
_COLLATERAL_COULDNT_READ_OUTPUT = (
    "error: couldn't read 'crates/core/src/lib.rs': No such file or directory\n"
)

# shape 2: the verify entrypoint script itself is gone (rc=127 lint stage,
# <0.4s) — the worktree was removed before the lint stage's shell could even
# exec verify.sh.
_COLLATERAL_VERIFY_SH_MISSING_OUTPUT = './scripts/verify.sh: No such file or directory\n'

# shape 3: the shell cannot write its own captured output back into the
# (now-removed) _merge-verify worktree.
_COLLATERAL_COULD_NOT_WRITE_OUTPUT = (
    'bash: line 1: could not write output to /work/_merge-verify/target/out.log\n'
)

# shape 4: bash's own shell-init cwd probe fails because its cwd (inside the
# removed worktree) no longer exists.
_COLLATERAL_GETCWD_OUTPUT = (
    'shell-init: error retrieving current directory: getcwd: cannot access parent '
    'directories: No such file or directory\n'
)

# shape 5: the anticipatory marker reify emits once reify 5261 lands — see
# plan.json's design_decisions for why this is landed ahead of reify 5261
# (the classifier capability is delivered entirely by this task; reify 5261
# only governs whether reify emits the marker in production).
_COLLATERAL_INTERRUPTED_MARKER_OUTPUT = '=== INTERRUPTED (worktree removed) ===\n'

_ALL_COLLATERAL_SHAPES = [
    ('couldnt_read', _COLLATERAL_COULDNT_READ_OUTPUT),
    ('verify_sh_missing', _COLLATERAL_VERIFY_SH_MISSING_OUTPUT),
    ('could_not_write', _COLLATERAL_COULD_NOT_WRITE_OUTPUT),
    ('getcwd', _COLLATERAL_GETCWD_OUTPUT),
    ('interrupted_marker', _COLLATERAL_INTERRUPTED_MARKER_OUTPUT),
]


class TestMergeVerifyCollateralEnvGuard:
    """task 2831: each of the five documented merge-verify restart-collateral
    shapes classifies ENV_TRANSIENT — tool-blind, like the task-2756
    Cargo.lock signature — so the merge_queue's transient-infra hold
    (INFRA_TRANSIENT_CATEGORIES) fires instead of hard-blaming an innocent
    branch for host-caused collateral."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    @pytest.mark.parametrize(('shape_name', 'output'), _ALL_COLLATERAL_SHAPES)
    def test_collateral_shape_is_env_transient(self, tool, shape_name, output):
        assert _classify(tool, output, 1, False) == FailureCategory.ENV_TRANSIENT, (
            f'shape {shape_name!r} must classify ENV_TRANSIENT under {tool!r}'
        )

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    @pytest.mark.parametrize(('shape_name', 'output'), _ALL_COLLATERAL_SHAPES)
    def test_collateral_shape_is_infra_transient(self, tool, shape_name, output):
        """Observable signal: the unit-level proxy that this output will hit
        the merge_queue.py:1886 transient-infra gate."""
        result = _classify(tool, output, 1, False)
        assert result in INFRA_TRANSIENT_CATEGORIES, (
            f'shape {shape_name!r} must be infra-transient under {tool!r}, got {result!r}'
        )


# task 2831 review: rustc's OWN verbatim wording for a genuine branch fault
# — a `#[path = "..."]` attribute pointing at a file the branch forgot to
# add — is "couldn't read <path>: No such file or directory", textually
# identical to shape-1's collateral read-failure golden above. Unlike a
# guard/shell script's single-line read failure, rustc always renders its
# stable span-pointer line (" --> path:line:col") alongside a location-tied
# diagnostic, which is what the shape-1 veto (`_RUSTC_DIAGNOSTIC_SPAN_RE`)
# keys on.
_GENUINE_RUSTC_MISSING_PATH_MODULE_FAULT_OUTPUT = (
    "error: couldn't read src/missing_module.rs: No such file or directory (os error 2)\n"
    ' --> src/lib.rs:3:1\n'
    '  |\n'
    '3 | #[path = "missing_module.rs"] mod missing_module;\n'
    '  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
)


class TestMergeVerifyCollateralEnvGuardNegatives:
    """Narrowness / non-regression guards for the new collateral patterns —
    green both before and after the impl."""

    def test_existing_settings_toml_read_failure_stays_non_env_transient(self):
        """The existing DF-2756 negative sample (no ENOENT text) must stay
        non-ENV_TRANSIENT under the new collateral patterns too — proves
        shape-1's extended verb alternation still requires the 'No such file
        or directory' text, not just any read-failure verb."""
        output = 'could not read /work/_merge-verify/config/settings.toml\n'
        result = _classify(ToolKind.OPAQUE, output, 1, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE
        assert result != FailureCategory.ENV_TRANSIENT

    def test_bare_application_file_not_found_without_read_verb_not_env_transient(self):
        """A bare application FileNotFoundError with no adjacent read verb
        must not be swallowed by shape-1 — proves the pattern requires a
        READ verb immediately preceding 'No such file or directory', not a
        bare mention of that text anywhere in the output."""
        output = "FileNotFoundError: [Errno 2] No such file or directory: 'fixture.json'\n"
        result = _classify(ToolKind.PYTEST, output, 1, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE
        assert result != FailureCategory.ENV_TRANSIENT

    def test_genuine_compile_error_mentioning_path_stays_compile_error(self):
        """A real rustc compile fault that merely mentions a source path (the
        same path shape-1's golden uses) must not be swallowed by the new
        collateral patterns — proves they do not shadow genuine branch
        faults."""
        output = (
            'error[E0308]: mismatched types\n'
            '  --> crates/core/src/lib.rs:10:5\n'
            'note: expected struct `Foo`, found struct `Bar`\n'
        )
        assert _classify(ToolKind.CARGO_TEST, output, 1, False) == FailureCategory.COMPILE_ERROR

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_genuine_rustc_missing_path_module_fault_not_env_transient(self, tool):
        """task 2831 review: shape-1's read-failure text is also rustc's OWN
        wording for a genuine `#[path = ...]` branch fault. Proves the
        rustc/clippy diagnostic span-pointer line (' --> path:line:col')
        vetoes the shape-1 match, so this genuine branch fault is left for
        per-tool dispatch (unknown_test_failure today — the same fallback it
        would already have received before this task, since none of the
        existing COMPILE_ERROR patterns match this codeless rustc message
        either — see the module docstring's shape-1 veto / Known gap notes)
        instead of being swallowed as host collateral and granted an
        undeserved infra retry."""
        result = _classify(tool, _GENUINE_RUSTC_MISSING_PATH_MODULE_FAULT_OUTPUT, 1, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE
        assert result != FailureCategory.ENV_TRANSIENT
        assert result not in INFRA_TRANSIENT_CATEGORIES


# ---------------------------------------------------------------------------
# task 2831 (the task SIGNAL): replayed reify 5164/5071 outputs — a wrapper
# lock/timeout line co-occurring with collateral shapes. Dep 2821 (merged,
# ee39b89bd7) tightened the SEMAPHORE_TIMEOUT deterministic-diagnostic/
# test-verdict veto, but that veto only suppresses a genuine compiler/lint/
# test VERDICT — it does not recognize a broken worktree as a MORE-SPECIFIC
# root cause. Today (before the step-4 reorder) the broken-worktree
# ENV_TRANSIENT checks sit AFTER the semaphore lock+timeout arm in
# _classify_environmental, so these replayed outputs still classify
# SEMAPHORE_TIMEOUT — the exact failure mode this task exists to fix, since
# the collateral shapes below carry NO 2821 veto marker (no clippy::, no
# error[E\d+]:, no test-verdict count-pair, no pytest ===== failure banner,
# no test-runner-shaped FAILED line), so they genuinely trip the semaphore
# arm rather than passing incidentally via the veto's None-return.
#
# RED today: the broken-worktree ENV_TRANSIENT checks run AFTER the
# semaphore arm, so both replayed outputs below classify SEMAPHORE_TIMEOUT
# instead of ENV_TRANSIENT.
# ---------------------------------------------------------------------------

# Replayed reify 5164: a wrapper lock/timeout line + a getcwd failure + a
# missing verify.sh entrypoint (two collateral shapes co-occurring).
_REPLAY_REIFY_5164_OUTPUT = (
    'flock: waiting for the merge-verify slot ... a peer timed out holding the lock\n'
    'shell-init: error retrieving current directory: getcwd: cannot access parent '
    'directories: No such file or directory\n'
    './scripts/verify.sh: No such file or directory\n'
)

# Replayed reify 5071: a wrapper lock/timeout line + a couldn't-read failure +
# a could-not-write-to-_merge-verify failure (two collateral shapes
# co-occurring, distinct from the 5164 pair above).
_REPLAY_REIFY_5071_OUTPUT = (
    'note: a peer holding the semaphore slot had timed out\n'
    "error: couldn't read 'crates/core/src/lib.rs': No such file or directory\n"
    'bash: line 1: could not write output to /work/_merge-verify/target/out.log\n'
)


class TestReplayedMergeVerifyCollateralWinsOverSemaphore:
    """task 2831 (the task SIGNAL): a replayed lock+timeout wrapper line
    co-occurring with broken-worktree collateral shapes must classify
    ENV_TRANSIENT, not SEMAPHORE_TIMEOUT (the more-specific root cause wins,
    exactly like DISK_FULL already does on ENOSPC co-occurrence) and not
    TEST_FAILURE (the fall-through this task heads off once dep 2821's
    tightened veto stops handing these outputs a free SEMAPHORE_TIMEOUT
    infra-retry)."""

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_replayed_reify_5164_is_env_transient(self, tool):
        result = _classify(tool, _REPLAY_REIFY_5164_OUTPUT, 1, False)
        assert result == FailureCategory.ENV_TRANSIENT
        assert result != FailureCategory.SEMAPHORE_TIMEOUT
        assert result != FailureCategory.TEST_FAILURE

    @pytest.mark.parametrize('tool', ALL_TOOL_KINDS)
    def test_replayed_reify_5071_is_env_transient(self, tool):
        result = _classify(tool, _REPLAY_REIFY_5071_OUTPUT, 1, False)
        assert result == FailureCategory.ENV_TRANSIENT
        assert result != FailureCategory.SEMAPHORE_TIMEOUT
        assert result != FailureCategory.TEST_FAILURE


class TestBrokenWorktreeOrderingNegatives:
    """Ordering negatives (green throughout, both before and after step-4's
    reorder): the more-specific DISK_FULL root cause still wins on ENOSPC
    co-occurrence, and a genuine wrapper-only timeout with NO collateral
    shape still classifies SEMAPHORE_TIMEOUT."""

    def test_enospc_with_collateral_stays_disk_full(self):
        output = _DISK_FULL_ENOSPC_OUTPUT + _COLLATERAL_GETCWD_OUTPUT
        assert _classify(ToolKind.OPAQUE, output, 1, False) == FailureCategory.DISK_FULL

    def test_genuine_flock_timeout_without_collateral_stays_semaphore_timeout(self):
        assert (
            _classify(ToolKind.OPAQUE, _SEMAPHORE_TIMEOUT_FLOCK_OUTPUT, 1, False)
            == FailureCategory.SEMAPHORE_TIMEOUT
        )

    def test_genuine_slot_timeout_without_collateral_stays_semaphore_timeout(self):
        assert (
            _classify(ToolKind.OPAQUE, _SEMAPHORE_TIMEOUT_SLOT_OUTPUT, 1, False)
            == FailureCategory.SEMAPHORE_TIMEOUT
        )


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


# ---------------------------------------------------------------------------
# step-3367: mis-resolved pyright interpreter (esc-3359-1)
# ---------------------------------------------------------------------------


def _unresolved_import_lines(*modules: str, path: str = 'src/fused_memory/server.py') -> str:
    """Render *modules* in pyright's REAL unresolved-import line shape.

    Captured verbatim during the task-3367 reproduction, e.g.::

        /repo/src/fused_memory/server.py:12:8 - error: Import "pytest" could
        not be resolved (reportMissingImports)
    """
    return '\n'.join(
        f'  /repo/{path}:{i + 8}:8 - error: Import "{mod}" could not be '
        f'resolved (reportMissingImports)'
        for i, mod in enumerate(modules)
    )


# The observed false-red: the baseline sentinel `pytest` is unresolved
# alongside many other workspace third-party packages, because the resolved
# interpreter is not this worktree's .venv at all.
PHANTOM_MISSING_IMPORTS = _unresolved_import_lines(
    'pytest', 'pydantic', 'aiosqlite', 'openai', 'qdrant_client'
)


class TestInterpreterMissingWorkspacePackages:
    """Pins ``is_interpreter_missing_workspace_packages`` — task 3367 / esc-3359-1.

    The predicate recognises an interpreter holding NONE of the workspace's
    third-party packages (pyright resolved an ambient env rather than this
    worktree's ``.venv``), and must NOT fire on a genuine missing-dependency
    regression — a false positive would excuse a real branch defect as
    environmental and let it through the merge gate as an infra hold.
    """

    def test_sentinel_plus_many_distinct_modules_is_environmental(self) -> None:
        """The observed shape: `pytest` unresolved alongside >=5 distinct modules."""
        assert verify_classify.is_interpreter_missing_workspace_packages(
            PHANTOM_MISSING_IMPORTS
        )

    def test_dotted_submodules_collapse_to_their_top_level_module(self) -> None:
        """Submodules of ONE package must not inflate the distinct count alone.

        Both blocks below carry the sentinel. The first names five distinct
        top-level packages via dotted paths and must fire; the second names
        five dotted paths that all belong to two top-level packages and must
        not.
        """
        five_distinct_tops = _unresolved_import_lines(
            'pytest',
            'graphiti_core.nodes',
            'qdrant_client.models',
            'pydantic.fields',
            'openai.types',
        )
        assert verify_classify.is_interpreter_missing_workspace_packages(
            five_distinct_tops
        )

        two_distinct_tops = _unresolved_import_lines(
            'pytest',
            'qdrant_client.models',
            'qdrant_client.http',
            'qdrant_client.http.models',
            'qdrant_client.conversions',
        )
        assert not verify_classify.is_interpreter_missing_workspace_packages(
            two_distinct_tops
        )

    def test_single_genuine_missing_dependency_is_not_environmental(self) -> None:
        """A real branch defect must stay RED, never be excused as environmental."""
        assert not verify_classify.is_interpreter_missing_workspace_packages(
            _unresolved_import_lines('brand_new_lib')
        )

    def test_many_distinct_modules_without_the_sentinel_is_not_environmental(self) -> None:
        """The sentinel is what separates "wrong interpreter" from "wrong dep set".

        A branch that legitimately dropped several dependencies loses those
        imports but never loses ``pytest`` — every workspace member declares
        it as a dev dependency, so a healthy env ALWAYS resolves it.
        """
        assert not verify_classify.is_interpreter_missing_workspace_packages(
            _unresolved_import_lines(
                'pydantic', 'aiosqlite', 'openai', 'qdrant_client', 'graphiti_core'
            )
        )

    def test_empty_output_is_not_environmental(self) -> None:
        assert not verify_classify.is_interpreter_missing_workspace_packages('')

    def test_other_pyright_rules_are_not_environmental(self) -> None:
        """Only reportMissingImports counts — other pyright rules are branch defects."""
        other_rules = (
            '  /repo/src/a.py:3:9 - error: Argument of type "int" cannot be '
            'assigned to parameter "x" (reportArgumentType)\n'
            '  /repo/src/b.py:4:9 - error: Cannot access attribute "foo" for '
            'class "Bar" (reportAttributeAccessIssue)\n'
            '  /repo/src/c.py:5:9 - error: "baz" is not a known attribute of '
            'module "qux" (reportAttributeAccessIssue)\n'
            '  /repo/src/d.py:6:9 - error: Expression of type "None" is '
            'incompatible (reportAssignmentType)\n'
            '  /repo/src/e.py:7:9 - error: No overloads for "get" match '
            '(reportCallIssue)\n'
        )
        assert not verify_classify.is_interpreter_missing_workspace_packages(other_rules)

    @pytest.mark.parametrize(
        'text',
        [
            'Traceback (most recent call last):\n  File "t.py", line 1\nImportError: x\n',
            'src/a.py:1:1: F401 [*] `os` imported but unused\nFound 1 error.\n',
            'error: could not compile `foo` (lib) due to 1 previous error\n',
            '\x00\x01 not even text — Import " could not be resolved (\n',
        ],
    )
    def test_arbitrary_non_pyright_text_returns_false_without_raising(
        self, text: str
    ) -> None:
        """Exception-safe on any input — a classifier must never raise."""
        assert verify_classify.is_interpreter_missing_workspace_packages(text) is False
