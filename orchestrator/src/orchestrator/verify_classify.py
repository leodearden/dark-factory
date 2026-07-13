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
    no longer swallow a pytest/rustc line by construction). ``ToolKind.OPAQUE``
    carries the FULL legacy generic ladder (moved verbatim from verify.py's
    ``_CLASSIFY_PATTERNS`` — see ``_classify_opaque``) — the ONLY survivor of
    the tool-blind ladder. Every other recognised ``ToolKind`` currently
    falls through to that same generic ladder as a placeholder; dedicated
    tables (PYTEST, CARGO_TEST/CARGO_CLIPPY, NPX, PYRIGHT, RUFF) are added by
    later steps of this task (PRD task δ), each peeling its tool off this
    fallthrough with its own branch ahead of it.

    CLOSED DOMAIN: the return value is always a ``FailureCategory`` member —
    see that enum's docstring for the closed 12-value output domain its
    ``CATEGORY_POLICY`` table enforces exhaustively at import time.
    """
    if rc == 0:
        return FailureCategory.PASSED
    if timed_out:
        return FailureCategory.INFRA_TIMEOUT

    return _classify_opaque(output)


# ---------------------------------------------------------------------------
# ToolKind.OPAQUE — the full legacy generic ladder, moved verbatim from
# verify.py's _CLASSIFY_PATTERNS (with its grounding comments). The ONLY
# surviving dispatch target for the tool-blind ladder — every other ToolKind
# gets its own narrower table so a tool-T pattern can only ever match tool-T
# output (Invariant C1).
# ---------------------------------------------------------------------------

# pytest INTERNALERROR marker. A private, self-contained copy — this module
# avoids importing from verify.py to sidestep a verify <-> verify_classify
# import cycle.
_PYTEST_INTERNALERROR_RE = re.compile(r'^INTERNALERROR>.+$', re.MULTILINE)

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
