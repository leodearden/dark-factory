"""Tool-dispatched failure classifier (PRD: plans/verify-plan-prd.md task δ;
Contract §classify_failure, Invariants C1/C2).

``classify_failure(tool, rc, output, timed_out) -> FailureCategory`` replaces
verify.py's tool-BLIND ``_classify_failure`` regex ladder with a per-tool
dispatch: a tool-T pattern lives ONLY in tool-T's table, so a cargo token can
no longer swallow a pytest/rustc line by construction (C1). Where a tool
offers structured output, the classifier parses it directly instead of
regex-matching human text, falling back to the text table otherwise (C2).

Self-contained: this module defines its own compiled regex copies rather
than importing them from verify.py, both to avoid a verify <->
verify_classify import cycle and to decouple the machine-category
classifier from verify.py's ``_extract_cause_hint`` human-hint ladder — a
separate, deliberately unchanged concern (see that function's docstring in
verify.py).
"""

from __future__ import annotations

import re

from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_cmd import ToolKind

# pytest INTERNALERROR marker — shared by the PYTEST and OPAQUE tables below
# (an internal dedup within this one self-contained module; this module
# avoids importing from verify.py to sidestep a verify <-> verify_classify
# import cycle, not internal sharing between its own two ladders).
_PYTEST_INTERNALERROR_RE = re.compile(r'^INTERNALERROR>.+$', re.MULTILINE)


def classify_failure(tool: ToolKind, rc: int, output: str, timed_out: bool) -> FailureCategory:
    """Classify a command failure into a ``FailureCategory``, dispatched by tool.

    Guards (checked before any per-tool dispatch, identically for every
    ``ToolKind``):
    1. ``rc == 0``   -> ``FailureCategory.PASSED``
    2. ``timed_out`` -> ``FailureCategory.INFRA_TIMEOUT`` (wins over any
       output pattern — the root cause is the wall-clock limit, not the
       command output)

    Then dispatches on *tool* to a per-tool classification table (Invariant
    C1: a tool-T pattern lives ONLY in tool-T's table, so a cargo token can
    no longer swallow a pytest/rustc line by construction). ``ToolKind.PYTEST``
    gets its own table (see ``_classify_pytest``) — notably the env_transient
    shared-venv-mutation signatures, consulted ONLY here. ``ToolKind.OPAQUE``
    carries the FULL legacy generic ladder (moved verbatim from verify.py's
    ``_CLASSIFY_PATTERNS`` — see ``_classify_opaque``) — the ONLY survivor of
    the tool-blind ladder. Every other recognised ``ToolKind`` currently
    falls through to that same generic ladder as a placeholder; dedicated
    tables (CARGO_TEST/CARGO_CLIPPY, NPX, PYRIGHT, RUFF) are added by later
    steps of this task (PRD task δ), each peeling its tool off this
    fallthrough with its own branch ahead of it.

    CLOSED DOMAIN: the return value is always a ``FailureCategory`` member —
    see that enum's docstring for the closed 12-value output domain its
    ``CATEGORY_POLICY`` table enforces exhaustively at import time.
    """
    if rc == 0:
        return FailureCategory.PASSED
    if timed_out:
        return FailureCategory.INFRA_TIMEOUT

    if tool is ToolKind.PYTEST:
        return _classify_pytest(output)

    return _classify_opaque(output)


# ---------------------------------------------------------------------------
# ToolKind.PYTEST — env_transient (shared-venv-mutation signatures, task
# 2048) is consulted FIRST and ONLY here (Invariant C1's structural win: no
# other tool's table even references these patterns), then INTERNALERROR,
# then FAILED lines, then flock (the test leg is flock-admission-wrapped),
# falling through to UNKNOWN_TEST_FAILURE — which also covers pytest rc=5
# ("no tests ran", kept RED per task 1852 — see _classify_opaque's docstring
# for the same rc=5-stays-RED contract, which applies here identically).
# ---------------------------------------------------------------------------

# Shared-venv mutation signatures (task 2048): a concurrent `uv sync` from
# another orchestrator process on the shared .venv can transiently
# remove-then-readd packages WHILE a consumer is mid-pytest against it (a
# non-atomic install window). Grounded in task 2045's observation: an
# identical `pytest -n auto` that had just passed failed with a pytest usage
# error naming -n/--dist/--max-worker-restart (the xdist plugin vanished),
# `python -c "import xdist"` raised ModuleNotFoundError, and `python -m pip`
# reported "No module named pip". A serial run (`-o addopts=""`) passed,
# confirming it was environmental, not a code regression. Consulted ONLY
# under ToolKind.PYTEST (Invariant C1) — a pytest run's OWN captured output
# is the only place these harness-infrastructure-absence signatures can
# legitimately appear; application/cargo/pyright code does not normally
# emit them, so a genuine code failure in another tool can never be silently
# relabelled environmental just because it happens to mention pip/xdist.
_ENV_TRANSIENT_PATTERNS: list[re.Pattern[str]] = [
    # pytest usage error (rc=4) when the xdist plugin vanished mid-run:
    # "pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0"
    # Anchored to the literal "pytest: error: unrecognized arguments:" prefix
    # that argparse emits for pytest's own CLI (prog='pytest') rather than a
    # bare "unrecognized arguments:" substring, so an unrelated tool's usage
    # error that happens to mention -n/--dist/--max-worker-restart cannot
    # false-positive into env_transient — the same inverse-misattribution
    # hazard the pip pattern below is hardened against with its word-boundary
    # lookahead.
    re.compile(
        r'^.*pytest: error: unrecognized arguments:.*(?:-n\b|--dist\b|--max-worker-restart\b).*$',
        re.MULTILINE,
    ),
    # `python -m pip` when pip itself vanished from the venv. The trailing
    # negative lookahead (?![\w-]) requires 'pip' to be followed by a
    # non-word, non-hyphen boundary so a ModuleNotFoundError whose module
    # name merely STARTS with 'pip' (pipeline, pipx, pipenv, pip_audit,
    # pip-tools) does not false-positive into env_transient — that would be
    # the exact inverse misattribution this feature forbids (a genuine
    # import/code regression silently relabelled environmental, auto-retried,
    # and archive-denied). Grounded positives (task 2045's unquoted runpy
    # line '<executable>: No module named pip' and the quoted 'pip'/"pip"
    # forms) still match since the boundary follows the closing quote (or
    # end of line) in each case.
    re.compile(r"""No module named ['"]?pip['"]?(?![\w-])""", re.MULTILINE),
    # `import xdist` / `import pytest_xdist` when the plugin vanished.
    re.compile(
        r"""ModuleNotFoundError: No module named ['"](xdist|pytest_xdist)['"]""",
        re.MULTILINE,
    ),
]

# Compiled regex patterns for the PYTEST table (checked after env_transient
# above). Order matters: INTERNALERROR before FAILED so a worker-death run
# (which has both INTERNALERROR> lines and collateral FAILED lines from the
# dead worker) classifies as pytest_internalerror, not test_failure. Reuses
# the module-level _PYTEST_INTERNALERROR_RE, shared with the OPAQUE ladder.
_PYTEST_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_PYTEST_INTERNALERROR_RE, FailureCategory.PYTEST_INTERNALERROR),
    (re.compile(r'^.+\s+FAILED\s*$', re.MULTILINE), FailureCategory.TEST_FAILURE),
    (re.compile(r'^FAILED\s', re.MULTILINE), FailureCategory.TEST_FAILURE),
    # flock lock failures — the test leg is flock-admission-wrapped.
    (re.compile(r'^flock:', re.MULTILINE), FailureCategory.FLOCK_ERROR),
]


def _classify_pytest(output: str) -> FailureCategory:
    """The PYTEST table: env_transient FIRST (Invariant C1: ONLY here), then
    INTERNALERROR/FAILED/flock, falling through to UNKNOWN_TEST_FAILURE.
    """
    for env_pattern in _ENV_TRANSIENT_PATTERNS:
        if env_pattern.search(output):
            return FailureCategory.ENV_TRANSIENT
    for pattern, category in _PYTEST_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.OPAQUE — the full legacy generic ladder, moved verbatim from
# verify.py's _CLASSIFY_PATTERNS (with its grounding comments). The ONLY
# surviving dispatch target for the tool-blind ladder — every other ToolKind
# gets its own narrower table so a tool-T pattern can only ever match tool-T
# output (Invariant C1).
# ---------------------------------------------------------------------------

# Compiled regex patterns for the OPAQUE generic ladder — hoisted to module
# scope so re.compile() runs once at import time rather than on every call.
# Order matters: rustc diagnostic codes (error[E0308]) appear before plain
# 'error:' so compile errors are distinguished from cargo CLI errors.
_CLASSIFY_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (re.compile(r'error\[E\d+\]:', re.MULTILINE), FailureCategory.COMPILE_ERROR),
    (re.compile(r'compile error', re.MULTILINE | re.IGNORECASE), FailureCategory.COMPILE_ERROR),
    # cargo CLI errors — narrow allowlist of cargo-only prefixes so rustc
    # top-level diagnostics ('error: aborting due to previous errors',
    # 'error: could not compile `…`') fall through to unknown_test_failure.
    # Intentionally conservative: novel cargo CLI messages not listed here
    # (e.g. 'error: unexpected argument', 'error: the manifest-path must be …',
    # 'error: manifest path … does not exist') will fall through to
    # unknown_test_failure until added to the allowlist. Extend when a new
    # cargo CLI failure mode appears in production and needs its own bucket.
    #
    # Each retained token is grounded in a real observed cargo CLI log line:
    #   --              → 'error: --exclude can only be used together with --workspace'
    #   no such subcommand
    #                   → 'error: no such subcommand: `tset`'
    #   failed to (parse|compile|read)
    #                   → 'error: failed to parse manifest at `/path/Cargo.toml`'
    #                     'error: failed to compile `<name>` (lib), intermediates ...'
    #                       (cargo_rustc / compiler job-queue orchestrator)
    #                     'error: failed to read `/path/Cargo.toml`'
    #   could not find  → 'error: could not find `Cargo.toml` in `/path` or any parent directory'
    #
    # Dropped tokens — no grounded cargo CLI sample available:
    #   `invalid `  — see test_rustc_invalid_diagnostic_not_cargo_cli_error.
    #   `package `  — re-add with a tighter suffix (e.g. 'package \`') once a
    #                 real cargo log line is observed.
    #   `find`      — 'failed to find' (without 'could not' prefix) has no verified
    #                 cargo CLI sample; cargo uses 'could not find' for its typical
    #                 find-failure case, covered by the top-level alternative above.
    #                 See test_failed_to_find_alone_not_cargo_cli_error.
    (
        re.compile(
            r'^error: (--|no such subcommand|failed to (parse|compile|read)|could not find)',
            re.MULTILINE,
        ),
        FailureCategory.CARGO_CLI_ERROR,
    ),
    # pytest INTERNALERROR — must be checked BEFORE the FAILED patterns so that
    # a worker-death run (which has both INTERNALERROR> lines and collateral
    # FAILED lines from the dead worker) classifies as pytest_internalerror
    # rather than test_failure.
    (_PYTEST_INTERNALERROR_RE, FailureCategory.PYTEST_INTERNALERROR),
    # Rust test runner / pytest FAILED lines
    (re.compile(r'^.+\s+FAILED\s*$', re.MULTILINE), FailureCategory.TEST_FAILURE),
    (re.compile(r'^FAILED\s', re.MULTILINE), FailureCategory.TEST_FAILURE),
    # npm errors
    (re.compile(r'npm\s+(ERR!|error)', re.MULTILINE), FailureCategory.NPM_ERROR),
    # flock lock failures
    (re.compile(r'^flock:', re.MULTILINE), FailureCategory.FLOCK_ERROR),
    # tree-sitter generate failures
    (re.compile(r'tree-sitter generate', re.MULTILINE), FailureCategory.TREE_SITTER_GENERATE_ERROR),
]


def _classify_opaque(output: str) -> FailureCategory:
    """The full legacy generic ladder (verbatim), used for ``ToolKind.OPAQUE``.

    First match wins; a non-matching, non-zero-rc output falls through to
    ``UNKNOWN_TEST_FAILURE`` — this also covers pytest rc=5 ("no tests ran")
    among other non-zero exits. INVARIANT (task 1852): pytest rc=5 is
    intentionally classified RED here, ranked above ``PASSED`` in
    ``CATEGORY_PRIORITY``. A real test target that unexpectedly collects
    zero tests must stay RED so the merge gate catches "tests vanished"
    regressions — the data-module false-RED is fixed in the SCOPING layer
    (verify.py's ``_is_collectable_test_file``), not here.
    """
    for pattern, category in _CLASSIFY_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE
