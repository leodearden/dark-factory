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

import json
import re

from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_cmd import ToolKind

# ---------------------------------------------------------------------------
# Shared regex primitives — patterns that legitimately belong to more than
# one tool's table (e.g. a rustc compile_error diagnostic is meaningful for
# both ToolKind.CARGO_TEST/CARGO_CLIPPY and the ToolKind.OPAQUE generic
# ladder, which must keep classifying byte-identically for any output it
# still receives). Hoisted once here and referenced by every table that
# needs them, rather than duplicated per table — so a future grounding fix
# (in the spirit of the historical cargo_cli_error allowlist tightening
# commits — tasks 1103/1109/1116) lands in one place instead of risking
# re-diverging two copies of "the same" rule, which is the exact
# same-bug-fixed-in-two-places hazard this per-tool-dispatch refactor exists
# to close for the tool-blind ladder itself.
# ---------------------------------------------------------------------------

_PYTEST_INTERNALERROR_RE = re.compile(r'^INTERNALERROR>.+$', re.MULTILINE)
_COMPILE_ERROR_RUSTC_CODE_RE = re.compile(r'error\[E\d+\]:', re.MULTILINE)
_COMPILE_ERROR_STRING_RE = re.compile(r'compile error', re.MULTILINE | re.IGNORECASE)

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
_CARGO_CLI_ERROR_RE = re.compile(
    r'^error: (--|no such subcommand|failed to (parse|compile|read)|could not find)',
    re.MULTILINE,
)

# Rust test runner / pytest FAILED lines — the trailing form ('test NAME ...
# FAILED') is cargo's own test runner shape and also matches a pytest FAILED
# line; the leading form ('FAILED test::name') is pytest-report-shaped only
# and is NOT part of the cargo table (see _CARGO_PATTERNS below).
_TEST_FAILURE_TRAILING_RE = re.compile(r'^.+\s+FAILED\s*$', re.MULTILINE)
_TEST_FAILURE_LEADING_RE = re.compile(r'^FAILED\s', re.MULTILINE)

_NPM_ERROR_RE = re.compile(r'npm\s+(ERR!|error)', re.MULTILINE)
_FLOCK_ERROR_RE = re.compile(r'^flock:', re.MULTILINE)
_TREE_SITTER_GENERATE_RE = re.compile(r'tree-sitter generate', re.MULTILINE)


# ---------------------------------------------------------------------------
# Tool-blind environmental pre-dispatch guard (task 2549) — host-
# infrastructure conditions mean the same thing regardless of which tool's
# command hit them (a full disk is a full disk whether cargo, pytest, or npx
# emitted the line), so DISK_FULL/SEMAPHORE_TIMEOUT are classified in
# `_classify_environmental` BEFORE any per-tool dispatch — mirroring the
# `timed_out` -> `INFRA_TIMEOUT` guard's "the root cause is the environment,
# not the command output" precedent (checked immediately after it, in
# `classify_failure` below). Neither category belongs to any single tool's
# table, so Invariant C1 ("a tool-T pattern lives only in tool-T's table")
# does not apply here.
# ---------------------------------------------------------------------------

# ENOSPC markers reused VERBATIM from merge_queue._ENOSPC_MARKERS (single
# grounded vocabulary shared with merge_queue's _verify_hit_enospc and
# git_ops's disk-pressure detection) — do not invent new ENOSPC strings;
# extend that constant instead if a new grounded sample appears. Matched
# case-insensitively against the whole output, same as _verify_hit_enospc.
_ENOSPC_MARKERS = ('no space left on device', 'os error 28', 'enospc')

# Linker SIGBUS-on-full-disk (git_ops.py: "proceeding into an ENOSPC build
# that SIGBUSes the linker") — a real secondary disk-full mode where the
# human-visible failure is the linker's signal rather than a literal ENOSPC
# string. Requires a linking/collect2 context TOGETHER WITH a signal-7/
# SIGBUS/Bus-error token (both, not either) so a bare 'SIGBUS' mention out of
# context (e.g. a test asserting on signal-handling behavior) cannot
# false-positive into disk_full.
_LINKER_CONTEXT_RE = re.compile(r'linking with|collect2', re.IGNORECASE)
_LINKER_SIGNAL_RE = re.compile(r'signal:?\s*7\b|SIGBUS|Bus error', re.IGNORECASE)

# flock/semaphore slot-acquisition timeout — a lock/slot/semaphore token
# co-occurring with a timeout token. Deliberately NARROWER than
# _FLOCK_ERROR_RE ('^flock:') so a plain flock-contention error with no
# timeout token (e.g. the golden 'flock: failed to acquire lock on
# /var/lock/mylock') still falls through to FLOCK_ERROR unchanged.
_LOCK_TOKEN_RE = re.compile(r'\b(?:flock|lock|slot|semaphore)\b', re.IGNORECASE)
_TIMEOUT_TOKEN_RE = re.compile(r'\btimed?\s*out\b', re.IGNORECASE)

# task 2748: deterministic-diagnostic veto for the SEMAPHORE_TIMEOUT arm. A
# genuine flock/semaphore slot-acquisition timeout is emitted by the
# concurrency-limiter WRAPPER (reify lib_slot_acquire.sh / `flock -w`) BEFORE
# the wrapped tool ever runs, so its output CANNOT also contain a compiler/
# lint diagnostic. `_LOCK_TOKEN_RE`/`_TIMEOUT_TOKEN_RE` are independent
# whole-output searches, so a deterministic `cargo clippy`/rustc diagnostic
# that quotes lock/slot/semaphore source and mentions "timed out" in a note
# satisfies both and is misclassified SEMAPHORE_TIMEOUT. The clippy
# lint-namespace token `clippy::` is present in every denied-lint note (e.g.
# `-D clippy::await-holding-lock`) and a rustc error code (`_COMPILE_ERROR_
# RUSTC_CODE_RE`, reused from the shared primitives above) is present in
# every rustc compile-error diagnostic — either PROVES the wrapped command
# ran, so both are used as the veto marker set rather than inventing a new
# one.
# Plain substring check (not a compiled regex — 'clippy::' has no
# metacharacters to match and this mirrors the _ENOSPC_MARKERS substring
# style used a few lines below).
_CLIPPY_LINT_MARKER = 'clippy::'

# Broken _merge-verify worktree (task 2756) — a guard script (e.g. reify's
# check-manifold-deps.sh) that cannot read the ephemeral merge-verify
# worktree's own Cargo.lock is a broken verify ENVIRONMENT, not a branch
# fault — exactly like DISK_FULL/SEMAPHORE_TIMEOUT above. Line-scoped: the
# read-failure verb and the `Cargo.lock` token must appear on the same line,
# mirroring the single-line grounded sample below.
#
# Grounded in a real observed guard-script failure (reify task 5120 RC-2,
# 2026-07-13):
#   'check-manifold-deps.sh: could not read manifold-csg-sys version from
#    /work/_merge-verify/Cargo.lock'
# Extend the verb alternation only when a new grounded read-failure sample
# appears, mirroring _CARGO_CLI_ERROR_RE's allowlist-tightening discipline.
_VERIFY_ENV_BROKEN_RE = re.compile(
    r"^.*\b(?:could not|cannot|can[’']t|unable to|failed to)\s+read\b.*\bCargo\.lock\b",
    re.MULTILINE | re.IGNORECASE,
)


def _classify_environmental(output: str) -> FailureCategory | None:
    """Tool-blind host-infrastructure guard: DISK_FULL / SEMAPHORE_TIMEOUT /
    ENV_TRANSIENT (broken ``_merge-verify`` worktree).

    Checked in ``classify_failure`` immediately after the ``timed_out`` ->
    ``INFRA_TIMEOUT`` guard and before any per-tool dispatch — these
    conditions are properties of the HOST, not of any one tool, so no
    per-tool table is the right home for them. Returns ``None`` (the caller
    proceeds to per-tool dispatch) when none of these conditions is grounded
    in *output*.

    SEMAPHORE_TIMEOUT deterministic-diagnostic veto (task 2748): a genuine
    slot/lock-acquisition timeout is raised by the concurrency-limiter
    wrapper BEFORE the wrapped tool runs, so its output can never carry a
    Rust compiler/clippy diagnostic. When *output* carries the clippy
    lint-namespace token (``clippy::``) or a rustc error code
    (``error[E\\d+]:``), the lock/slot/semaphore + timeout tokens are
    incidental to a deterministic lint/compile failure, so this returns
    ``None`` instead of ``SEMAPHORE_TIMEOUT`` and defers to per-tool
    dispatch. Applies ONLY to the SEMAPHORE_TIMEOUT arm — DISK_FULL's ENOSPC
    markers are specific enough (and checked first) that a compile/lint
    failure caused by a genuinely full disk must stay DISK_FULL.

    Known gap (out of scope for task 2748): the veto marker set is
    Rust-specific (clippy/rustc diagnostics only), because this guard runs
    tool-blind, BEFORE per-tool dispatch, for every ``ToolKind``. A
    deterministic non-Rust failure whose output incidentally quotes a
    lock/slot/semaphore token together with a "timed out" token — e.g. a
    pytest failure in a test named ``test_lock_timed_out``, or a similarly
    incidental npm/tree-sitter message — still satisfies the loose
    ``_LOCK_TOKEN_RE`` + ``_TIMEOUT_TOKEN_RE`` co-occurrence with no veto
    marker to suppress it, and is still misclassified ``SEMAPHORE_TIMEOUT``.
    That looseness predates task 2748 and is only narrowed for Rust here; a
    future task extending the marker set per-tool, or tightening
    ``_LOCK_TOKEN_RE``/``_TIMEOUT_TOKEN_RE`` to require wrapper-emitted
    context (e.g. an anchored ``lib_slot_acquire.sh``/``flock -w`` marker so
    a genuine slot timeout is matched positively instead of by loose
    co-occurrence), would close it.

    ENV_TRANSIENT — broken verify worktree (task 2756): a malformed
    ``_merge-verify`` worktree (e.g. an unreadable or missing ``Cargo.lock``)
    is a property of the HOST/environment, not of whichever guard script or
    tool happened to trip over it, so it is checked LAST here — after the
    more specific ENOSPC/linker/semaphore checks, so a co-occurring disk-full
    still wins as the more specific root cause (see
    ``test_enospc_with_unreadable_lock_stays_disk_full``) — and wins over
    per-tool dispatch regardless of ``ToolKind``. This source is DISTINCT
    from the pytest-scoped shared-venv ``_ENV_TRANSIENT_PATTERNS`` below
    (different grounded patterns mapping to the same ``ENV_TRANSIENT``
    category, exactly like how ``DISK_FULL`` above is produced by this same
    tool-blind guard rather than any per-tool table) — this does not violate
    Invariant C1, which forbids duplicating the SAME pattern across per-tool
    tables, not a category being reachable from more than one guard.
    """
    lower = output.lower()
    if any(marker in lower for marker in _ENOSPC_MARKERS):
        return FailureCategory.DISK_FULL
    if _LINKER_CONTEXT_RE.search(output) and _LINKER_SIGNAL_RE.search(output):
        return FailureCategory.DISK_FULL
    if (
        _LOCK_TOKEN_RE.search(output)
        and _TIMEOUT_TOKEN_RE.search(output)
        and _CLIPPY_LINT_MARKER not in output
        and not _COMPILE_ERROR_RUSTC_CODE_RE.search(output)
    ):
        return FailureCategory.SEMAPHORE_TIMEOUT
    if _VERIFY_ENV_BROKEN_RE.search(output):
        return FailureCategory.ENV_TRANSIENT
    return None


def classify_failure(tool: ToolKind, rc: int, output: str, timed_out: bool) -> FailureCategory:
    """Classify a command failure into a ``FailureCategory``, dispatched by tool.

    Guards (checked before any per-tool dispatch, identically for every
    ``ToolKind``):
    1. ``rc == 0``   -> ``FailureCategory.PASSED``
    2. ``timed_out`` -> ``FailureCategory.INFRA_TIMEOUT`` (wins over any
       output pattern — the root cause is the wall-clock limit, not the
       command output)
    3. ``_classify_environmental(output)`` -> ``FailureCategory.DISK_FULL``,
       ``FailureCategory.SEMAPHORE_TIMEOUT``, or ``FailureCategory.ENV_TRANSIENT``
       (broken ``_merge-verify`` worktree) when grounded in *output* (wins
       over any per-tool output pattern for the same "root cause is the
       environment, not the command output" reason as guard 2 — a host
       condition like a full disk, a lock/semaphore-slot timeout, or an
       unreadable worktree lockfile is not a property of any one tool).

    Then dispatches on *tool* to a per-tool classification table (Invariant
    C1: a tool-T pattern lives ONLY in tool-T's table, so a cargo token can
    no longer swallow a pytest/rustc line by construction):
    - ``ToolKind.PYTEST`` -> ``_classify_pytest`` (notably the env_transient
      shared-venv-mutation signatures, consulted ONLY here — see the
      "BEHAVIORAL NARROWING" note above ``_ENV_TRANSIENT_PATTERNS`` below for
      the resulting constraint: the caller must resolve a command to
      ``ToolKind.PYTEST`` for env_transient auto-recovery to ever fire).
    - ``ToolKind.CARGO_TEST`` / ``ToolKind.CARGO_CLIPPY`` -> a structured
      NDJSON parse (``_parse_cargo_json``) is attempted FIRST (Invariant C2);
      when *output* isn't detected as cargo's ``--message-format json`` NDJSON
      (or carries no error-level compiler-message), falls back to
      ``_classify_cargo``, the text table.
    - ``ToolKind.NPX`` -> ``_classify_npx``.
    - ``ToolKind.PYRIGHT`` -> a structured ``--outputjson`` parse
      (``_parse_pyright_json``) is attempted FIRST (Invariant C2); otherwise
      falls back to ``_classify_opaque`` (no dedicated PYRIGHT text table —
      preserves today's on-the-wire mapping).
    - ``ToolKind.RUFF`` -> a structured ``--output-format json`` parse
      (``_parse_ruff_json``) is attempted FIRST (Invariant C2); otherwise
      falls back to ``_classify_opaque`` (no dedicated RUFF text table —
      preserves today's on-the-wire mapping).
    - ``ToolKind.OPAQUE`` -> ``_classify_opaque``, the FULL legacy generic
      ladder (moved verbatim from verify.py's ``_CLASSIFY_PATTERNS``) — the
      ONLY survivor of the tool-blind ladder.

    Every structured parser is best-effort and exception-safe: *output* that
    isn't valid JSON (or isn't shaped as that tool's schema) yields ``None``
    rather than raising, so the classifier always falls back to the text
    table (Invariant C2's "falls back … does not crash"). The verify COMMANDS
    themselves are unchanged by this — they still emit human text — so the
    human log is unchanged by construction.

    CLOSED DOMAIN: the return value is always a ``FailureCategory`` member —
    see that enum's docstring for the closed 14-value output domain its
    ``CATEGORY_POLICY`` table enforces exhaustively at import time.
    """
    if rc == 0:
        return FailureCategory.PASSED
    if timed_out:
        return FailureCategory.INFRA_TIMEOUT
    environmental_category = _classify_environmental(output)
    if environmental_category is not None:
        return environmental_category

    if tool is ToolKind.PYTEST:
        return _classify_pytest(output)
    if tool in (ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY):
        # `is not None` (not `or`): FailureCategory.NONE == '' is falsy, so an
        # `or` fallback would wrongly skip a genuine (if currently unused)
        # NONE result from the parser.
        cargo_category = _parse_cargo_json(output)
        return cargo_category if cargo_category is not None else _classify_cargo(output)
    if tool is ToolKind.NPX:
        return _classify_npx(output)
    if tool is ToolKind.PYRIGHT:
        pyright_category = _parse_pyright_json(output)
        return pyright_category if pyright_category is not None else _classify_opaque(output)
    if tool is ToolKind.RUFF:
        ruff_category = _parse_ruff_json(output)
        return ruff_category if ruff_category is not None else _classify_opaque(output)

    return _classify_opaque(output)


# ---------------------------------------------------------------------------
# ToolKind.PYTEST — env_transient (shared-venv-mutation signatures, task
# 2048) is consulted FIRST and ONLY here (Invariant C1's structural win: no
# other tool's table even references these patterns), then INTERNALERROR,
# then FAILED lines, then flock (the test leg is flock-admission-wrapped),
# falling through to UNKNOWN_TEST_FAILURE — which also covers pytest rc=5
# ("no tests ran", kept RED per task 1852 — see _classify_opaque's docstring
# for the same rc=5-stays-RED contract, which applies here identically).
#
# BEHAVIORAL NARROWING vs. the pre-δ tool-blind ladder (intentional, tracked
# here rather than reverted): env_transient auto-recovery is now reachable
# ONLY when the failing check's config command lexically resolves to
# ToolKind.PYTEST via `parse_config_command` (see verify.py's
# `_tool_for_cmd`) — the tool-blind `_classify_failure` used to consult these
# patterns unconditionally, for every check. A test command the parser
# cannot identify as pytest (invoked indirectly through e.g. `make test`, a
# shell wrapper script, or a bare tox/nox target with no literal `pytest`
# token in the config string) resolves to ToolKind.OPAQUE instead, whose
# ladder deliberately does NOT include these patterns (see
# TestEnvTransientIsPytestScopedC1 in test_verify_classify.py) — so a
# genuine mid-run shared-venv mutation behind such a wrapper would classify
# as UNKNOWN_TEST_FAILURE rather than ENV_TRANSIENT, losing run_verification's
# single-serial-retry auto-recovery (verify.py, gated on
# `category == FailureCategory.ENV_TRANSIENT`) and attributing an infra
# transient to a code regression instead. Today every production test_cmd
# (`uv run pytest ...`, `cd sub && uv run pytest ...`, `python -m pytest`)
# contains a literal `pytest` token and correctly resolves to PYTEST, so this
# is a latent constraint on future config-command shapes, not a currently
# active gap; if a wrapped/indirect test command shape is ever introduced,
# either extend `parse_config_command` to see through the wrapper to PYTEST,
# or revisit whether these patterns belong in the OPAQUE ladder too.
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
    re.compile(r'''No module named ['"]?pip['"]?(?![\w-])''', re.MULTILINE),
    # `import xdist` / `import pytest_xdist` when the plugin vanished.
    re.compile(
        r'''ModuleNotFoundError: No module named ['"](xdist|pytest_xdist)['"]''',
        re.MULTILINE,
    ),
]

# Order matters: INTERNALERROR before FAILED so a worker-death run (which has
# both INTERNALERROR> lines and collateral FAILED lines from the dead worker)
# classifies as pytest_internalerror, not test_failure.
_PYTEST_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_PYTEST_INTERNALERROR_RE, FailureCategory.PYTEST_INTERNALERROR),
    (_TEST_FAILURE_TRAILING_RE, FailureCategory.TEST_FAILURE),
    (_TEST_FAILURE_LEADING_RE, FailureCategory.TEST_FAILURE),
    # flock lock failures — the test leg is flock-admission-wrapped.
    (_FLOCK_ERROR_RE, FailureCategory.FLOCK_ERROR),
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
# ToolKind.CARGO_TEST / ToolKind.CARGO_CLIPPY — a structured NDJSON parse
# (Invariant C2) is attempted FIRST; when that finds no error-level
# compiler-message, share one text table: compile_error, then the cargo-CLI
# allowlist, then the (trailing-form only) FAILED line, then tree-sitter
# generate failures, falling through to UNKNOWN_TEST_FAILURE. No
# INTERNALERROR/npm_error/flock_error/leading-FAILED branch here (Invariant
# C1) — those belong to the PYTEST/NPX/OPAQUE tables only, so a stray pytest
# or npm token in cargo output can never be misclassified as a
# cargo-specific category.
# ---------------------------------------------------------------------------


def _parse_cargo_json(output: str) -> FailureCategory | None:
    """Best-effort NDJSON parse of ``cargo … --message-format json`` output.

    cargo's ``--message-format json`` emits one compact JSON object per
    line (NDJSON) — verified against the pinned cargo 1.96.0 binary in this
    worktree, e.g.::

        {"reason":"compiler-message","message":{"level":"error",...}}
        {"reason":"build-finished","success":false}

    Scans line by line for a ``"reason": "compiler-message"`` record whose
    ``message.level == "error"`` -> ``COMPILE_ERROR`` (the structured signal
    Invariant C2 exists to catch — a diagnostic whose rendered human text
    doesn't happen to match any text-table pattern is still correctly
    classified). A line that isn't valid JSON (plain non-JSON text, or a
    JSON-shaped-but-malformed fragment) is skipped rather than raised —
    parsing is best-effort and exception-safe. Returns ``None`` (the caller
    falls back to ``_classify_cargo``, the text table) when no line parses as
    an error-level compiler-message, including when *output* isn't NDJSON at
    all.
    """
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(record, dict) or record.get('reason') != 'compiler-message':
            continue
        message = record.get('message')
        if isinstance(message, dict) and message.get('level') == 'error':
            return FailureCategory.COMPILE_ERROR
    return None


_CARGO_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_COMPILE_ERROR_RUSTC_CODE_RE, FailureCategory.COMPILE_ERROR),
    (_COMPILE_ERROR_STRING_RE, FailureCategory.COMPILE_ERROR),
    (_CARGO_CLI_ERROR_RE, FailureCategory.CARGO_CLI_ERROR),
    (_TEST_FAILURE_TRAILING_RE, FailureCategory.TEST_FAILURE),
    (_TREE_SITTER_GENERATE_RE, FailureCategory.TREE_SITTER_GENERATE_ERROR),
]


def _classify_cargo(output: str) -> FailureCategory:
    """The CARGO_TEST/CARGO_CLIPPY table, falling through to UNKNOWN_TEST_FAILURE."""
    for pattern, category in _CARGO_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.NPX — npm_error only, falling through to UNKNOWN_TEST_FAILURE.
# ---------------------------------------------------------------------------

_NPX_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_NPM_ERROR_RE, FailureCategory.NPM_ERROR),
]


def _classify_npx(output: str) -> FailureCategory:
    """The NPX table: npm_error, falling through to UNKNOWN_TEST_FAILURE."""
    for pattern, category in _NPX_PATTERNS:
        if pattern.search(output):
            return category
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.PYRIGHT — structured JSON only (Invariant C2); no dedicated text
# table exists — non-JSON PYRIGHT output falls through to the OPAQUE
# placeholder ladder (see classify_failure's docstring), preserving today's
# on-the-wire mapping. Adding a dedicated PYRIGHT/type-check FailureCategory
# is out of scope for this task.
# ---------------------------------------------------------------------------


def _parse_pyright_json(output: str) -> FailureCategory | None:
    """Best-effort parse of ``pyright --outputjson`` output.

    pyright's ``--outputjson`` emits one JSON object with a top-level
    ``generalDiagnostics`` array — verified against the pinned pyright
    1.1.408 binary in this worktree, e.g.::

        {"version": "1.1.408", "generalDiagnostics": [
            {"file": "...", "severity": "error", "message": "...", ...}
        ], "summary": {...}}

    A ``generalDiagnostics`` entry with ``severity == "error"`` maps to
    ``UNKNOWN_TEST_FAILURE`` — the preserved on-the-wire category (no
    dedicated PYRIGHT category exists — see module docstring). Exception-safe:
    *output* that isn't valid JSON, or isn't shaped like this schema, returns
    ``None`` rather than raising, so the caller falls back to
    ``_classify_opaque``, the text table.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    diagnostics = payload.get('generalDiagnostics')
    if not isinstance(diagnostics, list):
        return None
    for diagnostic in diagnostics:
        if isinstance(diagnostic, dict) and diagnostic.get('severity') == 'error':
            return FailureCategory.UNKNOWN_TEST_FAILURE
    return None


# ---------------------------------------------------------------------------
# ToolKind.RUFF — structured JSON only (Invariant C2); no dedicated text
# table exists — non-JSON RUFF output falls through to the OPAQUE placeholder
# ladder (see classify_failure's docstring), preserving today's on-the-wire
# mapping. Adding a dedicated RUFF/lint FailureCategory is out of scope for
# this task.
# ---------------------------------------------------------------------------


def _parse_ruff_json(output: str) -> FailureCategory | None:
    """Best-effort parse of ``ruff check --output-format json`` output.

    ruff's ``--output-format json`` emits a top-level JSON ARRAY of
    violation objects (not wrapped in an object) — verified against the
    pinned ruff binary in this worktree, e.g.::

        [{"cell": null, "code": "F821", "filename": "...",
          "message": "...", "severity": "error", ...}]

    A non-empty violations array maps to ``UNKNOWN_TEST_FAILURE`` — the
    preserved on-the-wire category (no dedicated RUFF category exists — see
    module docstring). Exception-safe: *output* that isn't valid JSON, or
    isn't shaped like this schema, returns ``None`` rather than raising, so
    the caller falls back to ``_classify_opaque``, the text table.
    """
    try:
        payload = json.loads(output)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, list) or not payload:
        return None
    if not any(isinstance(item, dict) for item in payload):
        return None
    return FailureCategory.UNKNOWN_TEST_FAILURE


# ---------------------------------------------------------------------------
# ToolKind.OPAQUE — the full legacy generic ladder, moved verbatim from
# verify.py's _CLASSIFY_PATTERNS (with its grounding comments, now hoisted
# above as shared primitives). The ONLY surviving dispatch target for the
# tool-blind ladder — every other ToolKind gets its own narrower table so a
# tool-T pattern can only ever match tool-T output (Invariant C1).
# ---------------------------------------------------------------------------

# Compiled regex patterns for the OPAQUE generic ladder — order matters:
# rustc diagnostic codes (error[E0308]) appear before plain 'error:' so
# compile errors are distinguished from cargo CLI errors; INTERNALERROR
# before FAILED so a worker-death run classifies as pytest_internalerror
# rather than test_failure.
_CLASSIFY_PATTERNS: list[tuple[re.Pattern[str], FailureCategory]] = [
    (_COMPILE_ERROR_RUSTC_CODE_RE, FailureCategory.COMPILE_ERROR),
    (_COMPILE_ERROR_STRING_RE, FailureCategory.COMPILE_ERROR),
    (_CARGO_CLI_ERROR_RE, FailureCategory.CARGO_CLI_ERROR),
    (_PYTEST_INTERNALERROR_RE, FailureCategory.PYTEST_INTERNALERROR),
    (_TEST_FAILURE_TRAILING_RE, FailureCategory.TEST_FAILURE),
    (_TEST_FAILURE_LEADING_RE, FailureCategory.TEST_FAILURE),
    (_NPM_ERROR_RE, FailureCategory.NPM_ERROR),
    (_FLOCK_ERROR_RE, FailureCategory.FLOCK_ERROR),
    (_TREE_SITTER_GENERATE_RE, FailureCategory.TREE_SITTER_GENERATE_ERROR),
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
