"""Test/lint/typecheck runner for verification stages."""

import asyncio
import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from shared.proc_group import terminate_process_group

from orchestrator.cargo_scope import discover_workspace_crates, files_to_crates
from orchestrator.config import ModuleConfig, OrchestratorConfig

logger = logging.getLogger(__name__)


def _scope_command(cmd: str | None, tool_keyword: str, files: list[str]) -> str | None:
    """Narrow *cmd* to operate on *files* instead of whole directories.

    Finds *tool_keyword* in *cmd*, keeps everything up to and including it as
    the prefix, extracts any dash-prefixed flags from the remainder, and
    rebuilds the command as ``'{prefix} {files} {flags}'``.

    Returns:
        ``None`` when *cmd* is ``None`` or *files* is empty.
        The original *cmd* unchanged when *tool_keyword* is not found.
        The scoped command otherwise.
    """
    if cmd is None:
        return None
    if not files:
        return None

    idx = cmd.find(tool_keyword)
    if idx == -1:
        return cmd

    prefix = cmd[: idx + len(tool_keyword)]
    remainder = cmd[idx + len(tool_keyword):]

    # Preserve any dash-prefixed flags from the original remainder
    flags = [tok for tok in remainder.split() if tok.startswith('-')]

    parts = [prefix] + files
    if flags:
        parts += flags
    return ' '.join(parts)


def _is_test_file(path: str) -> bool:
    """Return True for test-tree members (test_*.py, *_test.py, anything under tests/).

    This is the BROAD test-tree-membership predicate: a file that is True here
    means the test tree was touched and tests should run.  It is distinct from
    ``_is_collectable_test_file`` (the narrow pytest-collectability predicate):
    a data module under tests/ is a test-tree member but NOT collectable (passing
    it to pytest produces rc=5 "no tests ran").

    Returns False for conftest.py at any depth — conftest files are not directly
    runnable by pytest, so excluding them here means callers can pass the result
    straight to pytest without a follow-up filter.
    """
    name = path.rsplit('/', 1)[-1]
    return (
        (
            name.startswith('test_')
            or name.endswith('_test.py')
            or '/tests/' in path
            or path.startswith('tests/')
        )
        and not _is_conftest(path)
    )


def _is_collectable_test_file(path: str) -> bool:
    """Return True for files pytest will actually collect when passed as a target.

    Narrow predicate: only ``test_*.py`` and ``*_test.py`` (minus conftest).
    A data module under tests/ (e.g. ``shared/tests/silent_fallthrough_allowlist.py``)
    satisfies ``_is_test_file`` (test-tree membership) but NOT this predicate
    because its filename is not in the ``test_*`` / ``*_test`` convention —
    passing it to pytest produces rc=5 ("no tests ran").

    Distinct from ``_is_test_file`` (test-tree membership, broad).
    Task 1852: scoping builders must use this predicate for pytest targets.
    """
    name = path.rsplit('/', 1)[-1]
    return (name.startswith('test_') or name.endswith('_test.py')) and not _is_conftest(path)


def _is_conftest(path: str) -> bool:
    """Return True when *path* is a ``conftest.py`` file."""
    return path.rsplit('/', 1)[-1] == 'conftest.py'


# Matches a class that inherits from Protocol or TypedDict (as a base class).
# Deliberately a cheap content grep rather than an AST parse: structural files
# almost always use the canonical ``class Foo(Protocol):`` /
# ``class Bar(TypedDict):`` form, and the cost of a rare false positive (an
# extra package-wide pyright run) is far cheaper than missing a cross-file
# invariance break caused by scoping pyright to only the changed file.
_PROTOCOL_RE = re.compile(r'\bclass\s+\w+\s*\([^)]*\bProtocol\b')
_TYPEDDICT_RE = re.compile(r'\bclass\s+\w+\s*\([^)]*\bTypedDict\b')


def _is_structural_python_file(path: str, content: str) -> bool:
    """Return True when *path* defines a Protocol or TypedDict subclass.

    A deliberately cheap content grep (not an AST parse): matches
    ``class Foo(Protocol):`` and ``class Bar(TypedDict):`` patterns including
    multi-base forms such as ``class Foo(Base, Protocol):``.

    Cross-file invariant trap: pyright scoped to a single file cannot verify
    that all implementors of a Protocol still conform after the Protocol's
    definition widens.  Returning True causes ``scope_module_config`` to fall
    back to the full package-wide type-check command so pyright sees the
    complete picture.

    Scope notes:
    - ``.pyi`` stub files are explicitly excluded (``path.endswith('.py')``
      check).  Stub-defined Protocols create the same cross-file invariant
      trap, but ``.pyi`` stubs are filtered from ``scoped`` before this
      helper is reached (see ``scope_module_config``'s ``endswith('.py')``
      filter), so they would never be passed here in normal operation.
    - Type-argument usage such as ``class Foo(Dict[str, Protocol]):`` is a
      known false positive: the regex matches ``Protocol`` inside the base
      list even when it is not a direct base.  The cost is an extra
      package-wide pyright run — acceptable given the alternative is silently
      missing a real invariance break.
    """
    if not path.endswith('.py'):
        return False
    return bool(_PROTOCOL_RE.search(content) or _TYPEDDICT_RE.search(content))


def _strip_directory_flag(cmd: str | None, module_prefix: str) -> str | None:
    """Remove ``--directory <module_prefix>`` from a ``uv run`` command.

    When scoping to worktree-relative file paths, the ``--directory`` flag
    would cause tools to resolve paths relative to the module subdirectory,
    leading to double-prefixed paths that don't exist.  The ``--project``
    flag is kept so ``uv`` still activates the correct venv.
    """
    if cmd is None:
        return None
    cmd = cmd.replace(f'--directory {module_prefix} ', '')
    cmd = cmd.replace(f'--directory={module_prefix} ', '')
    # Handle flag at end of string (no trailing space)
    cmd = cmd.replace(f'--directory {module_prefix}', '')
    cmd = cmd.replace(f'--directory={module_prefix}', '')
    return ' '.join(cmd.split())


def _strip_leading_cd(cmd: str | None) -> str | None:
    """Strip a single leading ``cd <dir> &&`` segment from *cmd*.

    The project's per-subproject lint/type commands begin with
    ``cd <subproject> && <tool> …`` so each tool runs against that subproject's
    config (e.g. ``cd fused-memory && npx pyright``).  In the fallback path
    (:func:`_build_fallback_config`) the touched files live *outside* any
    module — e.g. a root-level ``tests/scripts/test_spawn_claude.py`` — and the
    scoped command targets a worktree-root-relative path.  A leading
    ``cd <subproject> &&`` would then resolve that path inside the wrong
    directory (``cd fused-memory && npx pyright tests/scripts/…`` → "File or
    directory does not exist", rc 4).  Removing the leading ``cd … &&`` makes
    the scoped command run from the worktree root where the path is valid.

    Only the *leading* segment is stripped; any later ``cd`` in the original
    chain has already been dropped by :func:`_scope_command`, which truncates
    the command at the first tool keyword.  Commands without a leading ``cd``
    (the common case) are returned unchanged.
    """
    if cmd is None:
        return None
    return re.sub(r'^\s*cd\s+\S+\s*&&\s*', '', cmd, count=1)


# Cargo subcommands whose ``--workspace`` flag we know how to rewrite.  Other
# cargo subcommands (doc, bench, ...) are left alone to avoid semantic drift.
_CARGO_SUBCMDS = ('test', 'clippy', 'check', 'build', 'run')

# Non-.rs file extensions that are safe to ignore when deciding whether to scope
# cargo commands to individual crates.  These are pure config/data files that
# do not contain executable source code and therefore don't require running a
# non-Rust toolchain alongside cargo.  Any extension NOT in this set (including
# the empty string for files like Dockerfile or LICENSE) triggers a fallthrough
# to ``--workspace`` — the conservative default protects chained commands such
# as ``cargo test --workspace && uv run pytest``.
_CARGO_SCOPE_SAFE_NON_RS_EXTS = frozenset({'.toml', '.yaml', '.yml', '.json', '.md'})

# Rust-specific filenames whose extensions can't be globally whitelisted.
# ``Cargo.lock`` has the ``.lock`` extension, which is also used by non-Rust
# ecosystem lockfiles (``yarn.lock``, ``poetry.lock``, ``uv.lock``), so adding
# ``.lock`` to ``_CARGO_SCOPE_SAFE_NON_RS_EXTS`` would silently admit those
# files and break the polyglot guard for mixed-ecosystem diffs.
# ``rust-toolchain`` is a rustup toolchain pin file with no extension (unlike
# ``rust-toolchain.toml``, which is already handled by the ``.toml`` whitelist).
_CARGO_SCOPE_SAFE_NON_RS_NAMES = frozenset({'Cargo.lock', 'rust-toolchain'})

# Pre-compiled regex that matches ``cargo <subcmd> ...--workspace`` where
# ``...`` does not cross a shell delimiter (``&&``, ``||``, ``;``, ``|``),
# so chained non-cargo commands (like ``cd gui && npm test``) are left alone.
_CARGO_WORKSPACE_RE = re.compile(
    r'(cargo\s+(?:' + '|'.join(_CARGO_SUBCMDS) + r')\b[^&|;]*?)'
    r'\s--workspace\b',
)

# Matches ``--exclude <name>`` inside the same cargo subcommand segment.
# After ``--workspace`` is replaced with ``-p <crate>``, any lingering
# ``--exclude`` flags become invalid (cargo rejects them with "--exclude
# can only be used together with --workspace"), so they must be stripped
# from the rewritten segment.
_CARGO_EXCLUDE_RE = re.compile(
    r'(cargo\s+(?:' + '|'.join(_CARGO_SUBCMDS) + r')\b[^&|;]*?)'
    r'\s--exclude(?:\s+|=)\S+',
)


# Pytest-aware cause-hint patterns. Anchored to whole lines so they don't
# false-match prose. ``_PYTEST_PROGRESS_*`` patterns are used to filter the
# fallback (last-non-blank-line) path so a pytest run killed mid-progress
# doesn't surface "...." dots as the cause hint.
_PYTEST_FAILED_LINE_RE = re.compile(r'^FAILED .+$', re.MULTILINE)
_PYTEST_INTERNALERROR_RE = re.compile(r'^INTERNALERROR>.+$', re.MULTILINE)
_PYTEST_FAILURE_SUMMARY_RE = re.compile(r'^=+ \d+ failed.*=+$', re.MULTILINE)
_PYTEST_TRACEBACK_E_RE = re.compile(r'^E   .+$', re.MULTILINE)
_PYTEST_PROGRESS_BARE_RE = re.compile(r'^[\.FsxXEPp]+(\s+\[\s*\d+%\])?$')
_PYTEST_PROGRESS_FILE_RE = re.compile(r'^\S+\.py [\.FsxXEPp]+(\s+\[\s*\d+%\])?$')


def _extract_cause_hint(output: str) -> str:
    """Extract a one-line failure hint from command output.

    Uses a pattern ladder (first match wins):
    1. ``FAILED test::name`` — pytest failure lines (start of line)
    2. ``INTERNALERROR>`` — pytest collection / plugin errors
    3. ``===== N failed in Xs =====`` — pytest summary line
    4. ``error: …``         — cargo/clippy surface errors
    5. ``… FAILED``         — Rust test runner failure lines
    6. ``Command timed out after Ns: …`` — our own timeout wrapper
    7. ``ERROR: …``         — flock/script wrapper errors
    8. ``… npm (ERR!|error) …`` — npm errors
    9. last ``E   …`` line  — pytest traceback (last match wins; most specific)
    10. fallback: last non-blank line of output, with pytest progress lines
        filtered. If only progress lines remain, returns an opaque-exit message.

    Returns ``''`` for None, empty, or whitespace-only input.
    Result is stripped to a single line and capped at 200 chars.
    """
    if not output or not output.strip():
        return ''

    _HINT_PATTERNS = [
        _PYTEST_FAILED_LINE_RE,
        _PYTEST_INTERNALERROR_RE,
        _PYTEST_FAILURE_SUMMARY_RE,
        re.compile(r'^error: .+$', re.MULTILINE),
        re.compile(r'^.+\s+FAILED$', re.MULTILINE),
        re.compile(r'^Command timed out after \d+s:.+$', re.MULTILINE),
        re.compile(r'^ERROR: .+$', re.MULTILINE),
        re.compile(r'^.*npm (ERR!|error).*$', re.MULTILINE),
    ]

    for pattern in _HINT_PATTERNS:
        m = pattern.search(output)
        if m:
            return m.group(0).strip()[:200]

    # Pytest traceback E-line — capture LAST one (most specific).
    e_matches = _PYTEST_TRACEBACK_E_RE.findall(output)
    if e_matches:
        return e_matches[-1].strip()[:200]

    # Filtered fallback: drop pytest progress lines before the last-non-blank.
    # Without this, a killed-mid-run pytest surfaces lines like
    # "orchestrator/tests/test_scheduler.py .." as the cause hint.
    meaningful = [
        line for line in reversed(output.splitlines())
        if line.strip()
        and not _PYTEST_PROGRESS_BARE_RE.match(line)
        and not _PYTEST_PROGRESS_FILE_RE.match(line)
    ]
    if not meaningful:
        return 'opaque test exit (no failure markers in output)'
    return meaningful[0].strip()[:200]


# Compiled regex patterns for _classify_failure — hoisted to module scope so
# re.compile() runs once at import time rather than on every call.
# Order matters: rustc diagnostic codes (error[E0308]) appear before plain
# 'error:' so compile errors are distinguished from cargo CLI errors.
_CLASSIFY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'error\[E\d+\]:', re.MULTILINE), 'compile_error'),
    (re.compile(r'compile error', re.MULTILINE | re.IGNORECASE), 'compile_error'),
    # cargo CLI errors — narrow allowlist of cargo-only prefixes so rustc
    # top-level diagnostics ('error: aborting due to previous errors',
    # 'error: could not compile `…`') fall through to unknown_test_failure.
    # Intentionally conservative: novel cargo CLI messages not listed here
    # (e.g. 'error: unexpected argument', 'error: the manifest-path must be …',
    # 'error: manifest path … does not exist') will fall through to
    # unknown_test_failure until added to the allowlist.  Extend when a new
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
    (re.compile(
        r'^error: (--|no such subcommand|failed to (parse|compile|read)|could not find)',
        re.MULTILINE,
    ), 'cargo_cli_error'),
    # pytest INTERNALERROR — must be checked BEFORE the FAILED patterns so that
    # a worker-death run (which has both INTERNALERROR> lines and collateral
    # FAILED lines from the dead worker) classifies as pytest_internalerror
    # rather than test_failure.  Reuses _PYTEST_INTERNALERROR_RE (hoisted above).
    (_PYTEST_INTERNALERROR_RE, 'pytest_internalerror'),
    # Rust test runner / pytest FAILED lines
    (re.compile(r'^.+\s+FAILED\s*$', re.MULTILINE), 'test_failure'),
    (re.compile(r'^FAILED\s', re.MULTILINE), 'test_failure'),
    # npm errors
    (re.compile(r'npm\s+(ERR!|error)', re.MULTILINE), 'npm_error'),
    # flock lock failures
    (re.compile(r'^flock:', re.MULTILINE), 'flock_error'),
    # tree-sitter generate failures
    (re.compile(r'tree-sitter generate', re.MULTILINE), 'tree_sitter_generate_error'),
]


def _classify_failure(output: str, rc: int, timed_out: bool) -> str:
    """Classify a command failure into a named category bucket.

    Uses a pattern ladder (first match wins):
    1. ``rc == 0``                              → ``'passed'``
    2. ``timed_out``                            → ``'infra_timeout'``
    3. ``error[E\\d+]:``                        → ``'compile_error'``
    4. ``compile error``                         → ``'compile_error'``
    5. ``error: <cargo CLI prefix>``             → ``'cargo_cli_error'`` (allowlist of cargo CLI prefixes; rustc top-level ``error: aborting…`` / ``error: could not compile`` fall through)
    6. ``INTERNALERROR>``                        → ``'pytest_internalerror'`` (checked BEFORE FAILED so a worker-death run with collateral FAILED lines classifies as infra, not drift)
    7. ``… FAILED``                              → ``'test_failure'``
    8. ``npm (ERR!|error)``                      → ``'npm_error'``
    9. ``flock:``                                → ``'flock_error'``
    10. ``tree-sitter generate``                 → ``'tree_sitter_generate_error'``
    11. fallback (rc != 0)                       → ``'unknown_test_failure'``

    The ``timed_out`` flag wins over any output pattern because the root
    cause is the wall-clock limit, not the command output.
    """
    if rc == 0:
        return 'passed'
    if timed_out:
        return 'infra_timeout'

    for pattern, category in _CLASSIFY_PATTERNS:
        if pattern.search(output):
            return category

    # Fallback — covers pytest rc=5 ("no tests ran") among other non-zero exits.
    # INVARIANT (b, task 1852): pytest rc=5 is intentionally classified RED here
    # ('unknown_test_failure', ranked above 'passed' in _CATEGORY_PRIORITY).
    # A real test target that unexpectedly collects zero tests must stay RED so
    # the merge gate catches "tests vanished" regressions.
    #
    # The data-module false-RED (e.g. shared/tests/silent_fallthrough_allowlist.py
    # producing rc=5) is fixed in the SCOPING layer — scope_module_config and
    # _build_fallback_config use _is_collectable_test_file to exclude non-test
    # data files from pytest targets.  DO NOT map rc=5 → 'passed' here: that
    # would silently green-light a real "tests vanished" regression.
    return 'unknown_test_failure'


# Categories that must NOT be auto-archived even though they end with '_error'.
# compile_error is handled by the debugger (type annotations, missing imports);
# the human triage criterion is "a human, not a debugger, has to look".
# pytest_internalerror is an infra crash (xdist worker killed by os._exit); it is
# non-deterministic and does NOT warrant human triage — the sweep already retries it.
# test_failure is handled by the debugger (self-correcting); compile_error likewise.
_ARCHIVE_DENY_LIST = frozenset({
    'compile_error', 'test_failure', 'infra_timeout', 'passed', '',
    'pytest_internalerror',
})

# Ordered from highest to lowest severity; used by ``_worst_category``.
# Categories absent from this list (e.g. custom ones) sort lower than all listed.
_CATEGORY_PRIORITY: list[str] = [
    'infra_timeout',
    'cargo_cli_error',
    'compile_error',
    'tree_sitter_generate_error',
    'flock_error',
    'npm_error',
    'pytest_internalerror',   # above test_failure: an infra crash, not a test drift
    'test_failure',
    'unknown_test_failure',
    'passed',
    '',
]


# Categories skipped by the preexisting-main-break probe.  Timeouts and
# lock contention are non-deterministic to re-run, so probing main for them
# is neither cheap nor a reliable signal.  All other genuine failure categories
# (compile_error, test_failure, npm_error, unknown_test_failure, …) DO
# reproduce with a stable (category, cause_hint) signature when inherited and
# are safe to re-check.
PREEXISTING_BREAK_SKIP_CATEGORIES: frozenset[str] = frozenset({
    'infra_timeout',
    'flock_error',
    # pytest_internalerror: xdist worker was killed by os._exit — non-deterministic,
    # so re-probing main is not a reliable signal.  The sweep already retries it.
    'pytest_internalerror',
})

# Process-wide cache for main-probe results: avoids redundant worktree-add +
# full-build/test re-runs when the same task retries (helper returned False, debugger
# ran, verify failed again) or when sibling tasks probe the same unchanged main.
# Key: (main_sha, category, normalised_cause_hint); Value: (probe_time, is_preexisting).
_PROBE_CACHE: dict[tuple[str, str, str], tuple[float, bool]] = {}
_PROBE_CACHE_TTL: float = 300.0  # 5 minutes; main_sha changes on every hotfix merge


def _worst_category(categories: list[str]) -> str:
    """Return the highest-severity category from *categories*.

    Priority is defined by ``_CATEGORY_PRIORITY``; a category not in the list
    sorts below all listed entries.  Returns ``''`` when *categories* is empty.
    """
    def _rank(cat: str) -> int:
        try:
            return _CATEGORY_PRIORITY.index(cat)
        except ValueError:
            return len(_CATEGORY_PRIORITY)  # unknown → lowest priority

    return min(categories, key=_rank, default='')


def _should_archive_category(category: str) -> bool:
    """Return True when the failure category warrants durable archival.

    Archival means the log is copied to ``data/verify-logs/<task_id>/`` for
    human triage — categories where the debugger can self-correct (compile
    errors, known test failures, timeouts) are excluded.

    Rule:
    - ``'unknown_test_failure'`` → True  (no pattern matched; human must look)
    - any category ending with ``'_error'`` AND not in deny-list → True
      (cargo CLI bugs, npm install failures, flock contention — infra issues)
    - everything else → False
    """
    if category in _ARCHIVE_DENY_LIST:
        return False
    if category == 'unknown_test_failure':
        return True
    return category.endswith('_error')


def _build_summary_payload(runs: list[dict], category: str, cause_hint: str) -> dict:
    """Build the summary.json payload dict from a list of run dicts.

    Extracted from ``_persist_attempt_logs`` so both the task-path summary
    (written into ``<worktree>/.task/verify/``) and the merge-path summary
    (written directly to the durable archive) share an identical shape.

    Top-level rc/cmd/timed_out/started_at/duration_secs fields come from the
    run with the highest numeric rc (timed_out used as a tiebreaker).  This
    intentionally differs from the 'category' field, which uses _worst_category
    priority semantics.  The rationale: rc is the most unambiguous exit-code
    signal for the outermost process, while category conveys semantic severity
    across tools that may use different rc scales.  Downstream readers should
    treat top-level metadata as "the loudest raw exit code" and 'category' as
    "the highest-severity classification".
    """
    active_runs = [r for r in runs if r.get('cmd') is not None]
    if active_runs:
        worst = max(active_runs, key=lambda r: (r['rc'], r['timed_out']))
    else:
        worst = {'rc': 0, 'timed_out': False, 'cmd': None,
                 'started_at': '', 'duration_secs': 0.0}

    return {
        'category': category,
        'cause_hint': cause_hint,
        'rc': worst['rc'],
        'timed_out': worst['timed_out'],
        'cmd': worst['cmd'],
        'started_at': worst['started_at'],
        'duration_secs': worst['duration_secs'],
        'commands': [
            {
                'label': r['label'],
                'cmd': r['cmd'],
                'rc': r['rc'],
                'timed_out': r['timed_out'],
                'started_at': r['started_at'],
                'duration_secs': r['duration_secs'],
            }
            for r in active_runs
        ],
    }


def _make_infix(module_prefix: 'str | None') -> str:
    """Return the dot-prefixed filename infix for *module_prefix*, or '' if None.

    Sanitizes by replacing ``/`` and spaces with ``_``, mirroring
    :func:`_warm_marker_name`.

    Examples::

        _make_infix('src/my module') -> '.src_my_module'
        _make_infix(None)            -> ''
    """
    if module_prefix is None:
        return ''
    safe = module_prefix.replace('/', '_').replace(' ', '_')
    return f'.{safe}'


def _write_run_log(
    target_dir: Path,
    attempt_id: int,
    infix: str,
    run: dict,
    *,
    ts_suffix: str = '',
    skip_if_exists: bool = False,
    caller: str = '_write_run_log',
) -> 'Path | None':
    """Write a single run's output to a log file; return the path or None on error.

    Filename: ``attempt-{attempt_id}{infix}.{run['label']}{ts_suffix}.log``

    Parameters
    ----------
    target_dir:
        Directory to write into (must already exist).
    attempt_id:
        Attempt counter embedded in the filename stem.
    infix:
        Module-prefix infix (including leading ``.``), or ``''``.  Build via
        :func:`_make_infix`.
    run:
        Run dict with at least ``label`` (str) and optionally ``output`` (str).
    ts_suffix:
        Timestamp suffix to append before ``.log`` (e.g. ``'-20260615T120000_000000Z'``).
        Empty string on the task path (no timestamp); non-empty on the merge path.
    skip_if_exists:
        When True and the target path already exists, return the path without
        rewriting.  Used on the task path where ``_run_cmd`` may have already
        streamed output there.
    caller:
        Name embedded in warning log messages for attribution.

    Returns ``None`` on OSError (logged at warning level).
    """
    log_path = target_dir / f'attempt-{attempt_id}{infix}.{run["label"]}{ts_suffix}.log'
    if skip_if_exists and log_path.exists():
        return log_path
    try:
        log_path.write_text(run.get('output', ''), encoding='utf-8')
        return log_path
    except OSError as exc:
        logger.warning('%s: could not write %s: %s', caller, log_path, exc)
        return None


def _persist_attempt_logs(
    worktree: Path,
    attempt_id: int,
    runs: list[dict],
    category: str,
    cause_hint: str,
    *,
    module_prefix: 'str | None' = None,
) -> list[Path]:
    """Write per-command outputs and a summary JSON to ``<worktree>/.task/verify/``.

    Each *run* dict must have:
        ``label``        — "test", "lint", or "type"
        ``cmd``          — shell command string or ``None`` (skipped check)
        ``rc``           — return code (int)
        ``output``       — combined stdout+stderr (str)
        ``timed_out``    — bool
        ``started_at``   — ISO timestamp string
        ``duration_secs``— elapsed seconds (float)

    No-op (returns ``[]``) when ``(worktree / '.task')`` is absent — review-
    checkpoint and merge-queue paths lack ``.task/`` and must not be created.

    When *module_prefix* is provided it is sanitized (``/`` and spaces →
    ``_``, mirroring :func:`_warm_marker_name`) and inserted as a middle infix:
    ``attempt-{N}.{safe_prefix}.{label}.log`` and
    ``attempt-{N}.{safe_prefix}.summary.json``.  This prevents last-writer-wins
    clobber when ``run_scoped_verification`` gathers multiple concurrent
    :func:`run_verification` calls for different sub-projects into the same
    worktree + attempt_id.

    When *module_prefix* is ``None`` the filenames remain
    ``attempt-{N}.{label}.log`` / ``attempt-{N}.summary.json`` so the single-
    module path is byte-identical to the pre-prefix behaviour.

    Writes:
    - ``attempt-{N}[.{safe_prefix}].{label}.log`` for every run where ``cmd is not None``
    - ``attempt-{N}[.{safe_prefix}].summary.json`` with the summary shape described in the
      task description: top-level keys are from the worst-failing run plus a
      ``commands`` list containing all per-run sub-dicts.

    Returns the list of log paths actually written (summary.json excluded
    so callers can pass the list straight to ``_archive_attempt_log``).
    """
    task_dir = worktree / '.task'
    if not task_dir.is_dir():
        return []

    verify_dir = task_dir / 'verify'
    try:
        verify_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning('_persist_attempt_logs: could not create %s: %s', verify_dir, exc)
        return []

    infix = _make_infix(module_prefix)
    written: list[Path] = []

    # Write per-command log files.  When the file already exists on disk it
    # was streamed there by ``_run_cmd`` (Change 2 — streaming variant): skip
    # rewriting to avoid clobbering streamed-but-truncated output on a kill,
    # but still record the path so summary.json and downstream archival see it.
    for run in runs:
        if run.get('cmd') is None:
            continue
        # No ts_suffix on the task path; skip_if_exists handles streamed files.
        path = _write_run_log(
            verify_dir, attempt_id, infix, run,
            skip_if_exists=True, caller='_persist_attempt_logs',
        )
        if path is not None:
            written.append(path)

    # Build summary.json via the shared helper (same shape as merge-path summary).
    summary_payload = _build_summary_payload(runs, category, cause_hint)

    summary_path = verify_dir / f'attempt-{attempt_id}{infix}.summary.json'
    try:
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
    except OSError as exc:
        logger.warning('_persist_attempt_logs: could not write %s: %s', summary_path, exc)

    return written


def _archive_attempt_log(
    worktree_log_paths: list[Path],
    archive_root: 'Path | None',
    task_id: str,
    attempt_id: int,
    category: str,
) -> list[Path]:
    """Copy worktree logs to the durable archive when ``category`` warrants it.

    Archive target: ``<archive_root>/<task_id>/attempt-{N}-<utc_ts>.log``.

    Early-returns ``[]`` when:
    - ``archive_root`` is ``None``
    - ``_should_archive_category(category)`` is ``False``

    All filesystem errors are caught, logged, and swallowed (best-effort).

    .. note::
        ``_prune_archive`` is intentionally NOT called here.  The caller
        (``run_scoped_verification``) calls it exactly once after all modules
        have been gathered, preventing concurrent per-module prune walks from
        racing on the same ``archive_root`` directory tree.
    """
    if archive_root is None:
        return []
    if not _should_archive_category(category):
        return []

    target_dir = archive_root / task_id
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning('_archive_attempt_log: could not create %s: %s', target_dir, exc)
        return []

    utc_ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    archived: list[Path] = []
    for src in worktree_log_paths:
        src = Path(src)
        # Preserve the source stem to avoid collisions when multiple log files
        # share the same suffix (e.g. attempt-1.test.log and attempt-1.lint.log
        # would both resolve to attempt-1-TS.log without the stem).
        dest = target_dir / f'{src.stem}-{utc_ts}{src.suffix}'
        try:
            shutil.copy2(src, dest)
            archived.append(dest)
        except OSError as exc:
            logger.warning('_archive_attempt_log: could not copy %s → %s: %s', src, dest, exc)

    return archived


def _archive_merge_verify_logs(
    runs: list[dict],
    archive_root: 'Path | None',
    task_id: str,
    attempt_id: int,
    category: str,
    cause_hint: str,
    *,
    module_prefix: 'str | None' = None,
) -> list[Path]:
    """Write merge-verify run outputs + summary DIRECTLY to the durable archive.

    Unlike ``_archive_attempt_log`` (which copies files from the worktree's
    ``.task/verify/``), this function writes directly to
    ``<archive_root>/<task_id>/`` because merge worktrees have ``.task/``
    scrubbed by design (git_ops.py) and there are no intermediate worktree
    log files to copy from.

    Differences from the task-path helpers:
    - **No deny-list check**: on the merge path there is no debugger loop;
      every failure reaches a human.  ``infra_timeout`` and ``test_failure``
      (the exact categories that distinguish timeout-vs-real-failure) are
      archived unconditionally.
    - **Direct-to-archive**: bypasses ``.task/verify/``; the archive is the
      only persistence target that survives merge_wt cleanup.

    Filename convention mirrors ``_archive_attempt_log``:
        ``attempt-{N}[.{safe_prefix}].{label}-{utc_ts}.log``
        ``attempt-{N}[.{safe_prefix}].summary-{utc_ts}.json``

    Returns the list of paths actually written (both .log and .json).
    Returns ``[]`` when ``archive_root`` is ``None``.
    All filesystem errors are caught, logged, and swallowed (best-effort).
    """
    if archive_root is None:
        return []

    target_dir = archive_root / task_id
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning('_archive_merge_verify_logs: could not create %s: %s', target_dir, exc)
        return []

    infix = _make_infix(module_prefix)
    # Use microsecond precision so rapid back-to-back merge-verify retries for
    # the same task (same attempt_id=1 default, same second) never overwrite each
    # other.  The format is still lexicographically sortable.
    utc_ts = datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')
    ts_suffix = f'-{utc_ts}'
    archived: list[Path] = []

    # Write per-command log files directly to the archive.
    for run in runs:
        if run.get('cmd') is None:
            continue
        path = _write_run_log(
            target_dir, attempt_id, infix, run,
            ts_suffix=ts_suffix, caller='_archive_merge_verify_logs',
        )
        if path is not None:
            archived.append(path)

    # Write summary.json using the shared payload builder.
    summary_path = target_dir / f'attempt-{attempt_id}{infix}.summary-{utc_ts}.json'
    try:
        summary_payload = _build_summary_payload(runs, category, cause_hint)
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False),
            encoding='utf-8',
        )
        archived.append(summary_path)
    except OSError as exc:
        logger.warning(
            '_archive_merge_verify_logs: could not write %s: %s', summary_path, exc,
        )

    return archived


_DEFAULT_ARCHIVE_MAX_AGE_DAYS = 30
_DEFAULT_ARCHIVE_MAX_BYTES = 500 * 1024 * 1024
# Process-local throttle: at most one rglob walk per process per 30 min.
# Cross-process redundancy is accepted as a cost-only trade-off; see task 1102.
# Thread-safety: _maybe_prune_archive has no awaits, so concurrent async coroutines
# cannot interleave the check + update. This is NOT safe for multi-threaded callers
# (e.g. asyncio.to_thread) without a threading.Lock — add one if that ever changes.
_PRUNE_THROTTLE_SECS: float = 1800  # 30 minutes
_LAST_PRUNE_AT: float | None = None
# Module-level reference capture for test injection: patch `verify._monotonic`
# instead of `verify.time.monotonic` (which would mutate the stdlib globally).
_monotonic = time.monotonic


def _prune_archive(
    archive_root: Path,
    max_age_days: int = _DEFAULT_ARCHIVE_MAX_AGE_DAYS,
    max_total_bytes: int = _DEFAULT_ARCHIVE_MAX_BYTES,
) -> None:
    """Enforce age + size retention on ``archive_root``.

    Two-pass strategy:
    1. Delete files older than ``max_age_days`` (by mtime).
    2. If aggregate size still exceeds ``max_total_bytes``, delete oldest-first
       until under cap.

    Best-effort: per-file errors are logged and swallowed. Outer FS errors (e.g.
    archive_root.exists() or rglob walk) may raise OSError — callers wishing to
    ignore those should wrap the call (see _maybe_prune_archive).
    """
    if not archive_root.exists():
        return

    # module-level `import time` is sufficient; no local import needed.
    now = time.time()
    cutoff = now - max_age_days * 86_400

    # Single rglob walk — collect all archivable files once, avoiding a second
    # directory scan for the size-cap pass.  Both *.log and *.json are counted
    # because _archive_merge_verify_logs emits summary.json files into the same
    # tree and they would otherwise accumulate unbounded (never counted toward the
    # size budget, never pruned).
    _PRUNE_SUFFIXES = frozenset(('.log', '.json'))
    all_entries: list[tuple[Path, float, int]] = []
    for path in archive_root.rglob('*'):
        if path.suffix not in _PRUNE_SUFFIXES:
            continue
        try:
            st = path.stat()
            if not path.is_file():
                continue
            all_entries.append((path, st.st_mtime, st.st_size))
        except OSError:
            continue

    # Pass 1: age-based deletion; collect survivors for the size-cap pass.
    survivors: list[tuple[Path, float, int]] = []
    for path, mtime, size in all_entries:
        if mtime < cutoff:
            try:
                path.unlink()
            except OSError as exc:
                logger.warning('_prune_archive: could not delete %s: %s', path, exc)
        else:
            survivors.append((path, mtime, size))

    # Pass 2: size cap on survivors (no second rglob needed).
    total = sum(sz for _, _, sz in survivors)
    if total > max_total_bytes:
        # Sort oldest-first.
        survivors.sort(key=lambda t: t[1])
        for path, _mtime, size in survivors:
            if total <= max_total_bytes:
                break
            try:
                path.unlink()
                total -= size
            except OSError as exc:
                logger.warning('_prune_archive: could not delete %s: %s', path, exc)


def _maybe_prune_archive(archive_root: Path | None) -> bool:
    """Thin wrapper around ``_prune_archive`` with None-guard and time throttle.

    Returns True if ``_prune_archive`` was invoked, False otherwise.

    - ``archive_root=None`` short-circuits immediately without updating the
      throttle timestamp (preserves semantics: None means no archival/pruning).
    - First call in a process always fires (``_LAST_PRUNE_AT is None``).
    - Subsequent calls within ``_PRUNE_THROTTLE_SECS`` are skipped.
    - After the window elapses, the next call fires and slides the window forward.
    - If ``_prune_archive`` raises OSError (e.g. ``archive_root.exists()`` or
      ``rglob`` fails on a permission-broken FS), the error is logged at warning
      level and ``_LAST_PRUNE_AT`` still advances — preventing the same exception
      from being raised on every subsequent verification call within the throttle
      window.  Non-OSError exceptions still propagate.
    """
    global _LAST_PRUNE_AT
    if archive_root is None:
        return False
    now = _monotonic()
    if _LAST_PRUNE_AT is not None and now - _LAST_PRUNE_AT < _PRUNE_THROTTLE_SECS:
        logger.debug(
            'skipping prune: %.0fs since last (throttle %ds)',
            now - _LAST_PRUNE_AT,
            _PRUNE_THROTTLE_SECS,
        )
        return False
    try:
        _prune_archive(archive_root)
    except OSError as exc:
        # Deliberate OSError-only scope: _prune_archive uses path.unlink() (not
        # shutil.rmtree), so shutil.Error cannot arise.  PermissionError /
        # FileNotFoundError are OSError subclasses on Python ≥ 3.3.  Programming
        # bugs (RuntimeError, AttributeError, etc.) still propagate uncaught.
        logger.warning(
            '_maybe_prune_archive: prune raised %s; advancing throttle to suppress retry storm',
            exc,
        )
    _LAST_PRUNE_AT = now
    return True


async def _derive_task_files_from_git(
    worktree: Path, config: OrchestratorConfig,
) -> list[str] | None:
    """Derive task file list from ``git diff main...HEAD`` in the worktree.

    Returns ``None`` when:
    - the worktree is on main (no diff to derive)
    - ``git diff`` fails for any reason
    - no files changed
    """
    from orchestrator.git_ops import _run
    try:
        rc, main_sha, _ = await _run(
            ['git', 'rev-parse', '--verify', config.git.main_branch],
            cwd=worktree,
        )
        if rc != 0:
            return None
        rc, head_sha, _ = await _run(
            ['git', 'rev-parse', 'HEAD'], cwd=worktree,
        )
        if rc != 0 or head_sha == main_sha:
            return None
        rc, output, _ = await _run(
            ['git', 'diff', '--name-only',
             f'{config.git.main_branch}...HEAD'],
            cwd=worktree,
        )
        if rc != 0:
            return None
        files = [f for f in output.strip().splitlines() if f.strip()]
        if files:
            logger.info('Derived %d task files from git diff', len(files))
            return files
    except Exception:
        logger.debug(
            'Failed to derive task files from git diff', exc_info=True,
        )
    return None


def _scope_cargo_workspace(cmd: str | None, crates: list[str]) -> str | None:
    """Rewrite ``cargo ... --workspace`` → ``cargo ... -p c1 -p c2 ...``.

    Returns *cmd* unchanged when:
    - *cmd* is ``None``
    - *crates* is empty
    - ``--workspace`` is not present
    - no cargo subcommand we recognise precedes ``--workspace``

    Supported cargo subcommands: test, clippy, check, build, run.

    Flags present between ``cargo <subcmd>`` and ``--workspace`` (e.g.,
    ``--all-targets``, ``--tests``, ``--lib``, ``-F <feature>``) and trailing
    args (``-- --test-threads=1``) are preserved verbatim — the helper only
    substitutes the ``--workspace`` token.  Chained shell commands after
    ``&&``/``||``/``;``/``|`` are untouched.
    """
    if cmd is None or not crates:
        return cmd
    if '--workspace' not in cmd:
        return cmd

    p_flags = ' '.join(f'-p {c}' for c in crates)
    new_cmd = _CARGO_WORKSPACE_RE.sub(
        lambda m: f'{m.group(1)} {p_flags}', cmd,
    )
    # Strip any ``--exclude <name>`` pairs from the rewritten cargo segment;
    # they are only valid with ``--workspace`` and cargo errors out otherwise.
    # Loop until stable to handle multiple ``--exclude`` flags on one cmd.
    prev = None
    while prev != new_cmd:
        prev = new_cmd
        new_cmd = _CARGO_EXCLUDE_RE.sub(lambda m: m.group(1), new_cmd)
    return new_cmd


def _apply_cargo_scope(
    mc: ModuleConfig,
    task_files: list[str],
    project_root: Path,
    scope_cargo_enabled: bool,
) -> ModuleConfig:
    """Return *mc* with cargo ``--workspace`` rewritten to touched crates.

    Guard conditions — returns *mc* unchanged when any fail:
    - ``scope_cargo_enabled`` is False, or ``mc.scope_cargo`` is explicitly False
    - *task_files* is empty
    - *task_files* contains no ``.rs`` files (no Rust source touched)
    - any non-``.rs`` file has an extension outside the safe config/data whitelist
      (``.toml``, ``.yaml``, ``.yml``, ``.json``, ``.md``) AND its basename is not
      in the filename allowlist (``Cargo.lock``, ``rust-toolchain``); this prevents under-protecting
      the non-Rust side of polyglot tasks with chained commands such as
      ``cargo test --workspace && uv run pytest``
    - the workspace has no discoverable crates
    - ``files_to_crates`` returns ``None`` (a file lives outside all crates)
    - the rewritten commands are byte-identical to the originals
    """
    if not scope_cargo_enabled:
        return mc
    if mc.scope_cargo is False:
        return mc
    if not task_files:
        return mc
    # Filter to .rs files for crate mapping — only Rust files determine which
    # crates need testing.
    rs_files = [f for f in task_files if f.endswith('.rs')]
    if not rs_files:
        return mc
    # Polyglot guard: if any non-.rs file has an extension outside the safe
    # config/data whitelist, bail to --workspace.  This protects chained
    # non-Rust commands (e.g. ``cargo test --workspace && uv run pytest``)
    # from being silently skipped when only some crates are scoped.
    non_rs = [f for f in task_files if not f.endswith('.rs')]
    for f in non_rs:
        p = Path(f)
        if (
            p.suffix.lower() not in _CARGO_SCOPE_SAFE_NON_RS_EXTS
            and p.name not in _CARGO_SCOPE_SAFE_NON_RS_NAMES
        ):
            return mc

    crates_map = discover_workspace_crates(project_root)
    if not crates_map:
        return mc
    matched = files_to_crates(rs_files, crates_map)
    if not matched:
        return mc

    new_test = _scope_cargo_workspace(mc.test_command, matched)
    new_lint = _scope_cargo_workspace(mc.lint_command, matched)
    new_type = _scope_cargo_workspace(mc.type_check_command, matched)

    if (new_test, new_lint, new_type) == (
        mc.test_command, mc.lint_command, mc.type_check_command,
    ):
        return mc  # nothing to rewrite — original didn't contain --workspace

    for label, old, new in (
        ('test', mc.test_command, new_test),
        ('lint', mc.lint_command, new_lint),
        ('type', mc.type_check_command, new_type),
    ):
        if old != new:
            logger.info('cargo scope (%s): %r -> %r', label, old, new)

    return ModuleConfig(
        prefix=mc.prefix,
        test_command=new_test,
        lint_command=new_lint,
        type_check_command=new_type,
        lock_depth=mc.lock_depth,
        max_per_module=mc.max_per_module,
        module_overrides=mc.module_overrides,
        verify_command_timeout_secs=mc.verify_command_timeout_secs,
        verify_cold_command_timeout_secs=mc.verify_cold_command_timeout_secs,
        concurrent_verify=mc.concurrent_verify,
        verify_env=mc.verify_env,
        scope_cargo=mc.scope_cargo,
    )


# Source-file extensions that warrant running verify at all. Markdown / YAML /
# TOML / JSON diffs are inert: every existing scoping branch
# (scope_module_config filters .py, _build_fallback_config filters .py,
# _apply_cargo_scope filters .rs) would no-op on them anyway, so the
# previous global-pytest fall-through was the only path that actually fired
# — and it was unsafe (see plans/fix-all-root-causes-humble-dream.md R1/R2).
_SOURCE_EXTENSIONS: frozenset[str] = frozenset({'.py', '.rs'})


def _has_source_files(task_files: list[str]) -> bool:
    """Return True when *task_files* contains at least one .py or .rs path."""
    exts = tuple(_SOURCE_EXTENSIONS)
    return any(f.endswith(exts) for f in task_files)


def _trivial_pass(reason: str) -> 'VerifyResult':
    """Build a VerifyResult that represents 'verify trivially passed'.

    Forward-reference string annotation: ``VerifyResult`` is defined further
    down in this module (no ``from __future__ import annotations`` here).
    """
    return VerifyResult(
        passed=True,
        summary=reason,
        test_output='',
        lint_output='',
        type_output='',
        timed_out=False,
        cause_hint='',
    )


async def _verify_pipeline_guard_requires_full_gate(
    worktree: Path,
    changed_files: list[str],
) -> bool:
    """Return True iff reify's verify-pipeline-guard.sh says the full gate is required.

    Shells out to ``<worktree>/scripts/verify-pipeline-guard.sh requires-full-gate
    <changed_files...>`` and returns ``True`` when the script exits 0 (conventional
    Unix predicate: exit 0 ⟹ condition is true ⟹ full gate is required).

    Fail-open contract — returns False for ANY of:
    - ``scripts/verify-pipeline-guard.sh`` absent in the worktree (backward-compat:
      dark-factory's own merges, pre-4626 reify, non-reify projects).
    - ``changed_files`` is empty.
    - Script exits non-zero (guard says fast-path is safe).
    - Script non-executable, spawn fails, WorktreeMissing, or any other exception
      (guard hiccup must never wedge the merge pipeline → log WARNING, return False).

    Mirrors GitOps._provision_reify_debug_port (the canonical cross-repo seam).
    """
    try:
        script = worktree / 'scripts' / 'verify-pipeline-guard.sh'
        if not script.exists() or not changed_files:
            return False
        from orchestrator.git_ops import _run  # noqa: PLC0415, I001 — lazy, mirrors verify_failure_is_preexisting_on_main
        rc, _out, _err = await _run(
            [str(script), 'requires-full-gate', *changed_files],
            cwd=worktree,
        )
        return rc == 0
    except Exception:
        logger.warning(
            '_verify_pipeline_guard_requires_full_gate: unexpected error for %s',
            worktree, exc_info=True,
        )
        return False


def scope_module_config(
    mc: ModuleConfig,
    task_files: list[str],
    worktree: Path | None = None,
) -> ModuleConfig | None:
    """Narrow *mc*'s commands to the specific *task_files* it covers.

    Filters *task_files* to ``.py`` files matching ``mc.prefix + '/'`` and
    keeps full worktree-relative paths.  The ``--directory`` flag is stripped
    from scoped commands so that tools resolve paths from the worktree root,
    where the full paths are valid.

    When *worktree* is provided, any scoped ``.py`` file that defines a
    ``Protocol`` or ``TypedDict`` subclass causes *mc.type_check_command* to
    be used verbatim (unscoped).  File-scoped pyright cannot verify cross-file
    invariants such as Protocol conformance; the package-wide form is the only
    safe option.  The ``--directory`` flag is preserved (NOT stripped) in the
    unscoped case — it is required for ``uv run`` to resolve ``src/``/``tests/``
    correctly when running from the worktree root.  This mirrors the existing
    ``has_conftest`` branch that sets ``test_cmd = mc.test_command`` verbatim.

    Returns ``None`` when no ``.py`` files from *task_files* fall under the
    prefix — the caller should skip that subproject entirely rather than run
    its full unscoped suite.  Running a subproject's complete test suite for
    a task that touched zero of its files is both wasteful and a source of
    unrelated-flake blockers on the merge-queue path.
    """
    prefix = mc.prefix + '/'
    # Keep full worktree-relative paths, filter to .py files under this module
    scoped: list[str] = []
    for f in task_files:
        if f.startswith(prefix) and f.endswith('.py'):
            scoped.append(f)

    if not scoped:
        return None

    # conftest.py defines fixtures/hooks that affect every test in the directory
    # subtree — the only correct scope is the full unscoped suite expressed by
    # mc.test_command.  Passing conftest.py directly to pytest finds 0 tests
    # (pytest >= 9 exits 1 with "no tests ran").
    has_conftest = any(_is_conftest(f) for f in scoped)
    # Narrow: only test_*.py / *_test.py files pytest will actually collect.
    # A data module under tests/ (e.g. silent_fallthrough_allowlist.py) is in
    # the test tree (_is_test_file) but NOT collectable — passing it to pytest
    # produces rc=5 ("no tests ran") → RED.  Task 1852 fixes this at the
    # scoping layer; the classifier (_classify_failure) is left untouched.
    collectable_tests = [f for f in scoped if _is_collectable_test_file(f)]
    # has_test_data: in-tree (test-tree member) but not collectable — mirrors
    # has_conftest: fall back to the full owning-package suite (mc.test_command).
    has_test_data = any(
        _is_test_file(f) and not _is_collectable_test_file(f) for f in scoped
    )

    # Detect structural files (Protocol/TypedDict definitions) when we have a
    # worktree to read from.  File-scoped pyright misses cross-file invariance
    # breaks; the package-wide command is the only safe scope.
    # Guard: skip the I/O loop entirely when there is no type-check command to
    # widen — has_structural can only affect type_cmd, so reading files is
    # wasted work when mc.type_check_command is None/empty.
    has_structural = False
    structural_trigger: str | None = None
    if worktree is not None and mc.type_check_command:
        for f in scoped:
            full = worktree / f
            if full.is_file():
                try:
                    content = full.read_text(encoding='utf-8', errors='replace')
                except OSError:
                    continue
                if _is_structural_python_file(f, content):
                    has_structural = True
                    structural_trigger = f
                    break

    # Build scoped commands with worktree-relative paths, then strip
    # --directory so tools resolve paths from the worktree root
    lint_cmd = _strip_directory_flag(
        _scope_command(mc.lint_command, 'ruff check', scoped), mc.prefix)
    if has_structural:
        # Verbatim unscoped type check — mirrors the has_conftest test_cmd branch.
        # --directory is intentionally preserved so uv resolves src/ and tests/
        # relative to the module subdirectory (NOT the worktree root).
        type_cmd = mc.type_check_command
        logger.info('pyright unscoped: structural file %s in diff', structural_trigger)
    else:
        type_cmd = _strip_directory_flag(
            _scope_command(mc.type_check_command, 'pyright', scoped), mc.prefix)
    if has_conftest or has_test_data:
        # Full unscoped suite: conftest changes affect everything it shadows;
        # data-module changes (e.g. a σ-allowlist re-baseline) are consumed by
        # tests we cannot enumerate from the path alone, so the full suite is
        # the only safe scope.  Both branches mirror each other (task 1852).
        test_cmd = mc.test_command
    elif collectable_tests:
        test_cmd = _strip_directory_flag(
            _scope_command(mc.test_command, 'pytest', collectable_tests), mc.prefix)
    else:
        test_cmd = None

    return ModuleConfig(
        prefix=mc.prefix,
        lint_command=lint_cmd,
        type_check_command=type_cmd,
        test_command=test_cmd,
        lock_depth=mc.lock_depth,
        max_per_module=mc.max_per_module,
        module_overrides=mc.module_overrides,
    )


def _build_fallback_config(
    task_files: list[str],
    config: OrchestratorConfig | None = None,
) -> ModuleConfig | None:
    """Build a synthetic ModuleConfig from *task_files* when no module configs match.

    Filters to ``.py`` files, classifies into source vs test, and builds
    targeted commands.  When *config* provides non-default commands (e.g.
    ``uv run --extra dev --extra web pytest``), those are used directly so
    that tools not installed globally can still be reached.

    For ``lint_command`` and ``type_check_command``, :func:`_scope_command`
    narrows the configured command to the touched files when the standard tool
    keyword appears (e.g. ``ruff check`` in ``uv run ruff check``), and
    returns the command unchanged when it doesn't (e.g. ``true`` or
    ``mypy``-based type checking).  For ``test_command``, the configured
    command is used as-is when it differs from the bare ``pytest`` default so
    that complex flag sequences like ``-m 'not slow' --ignore=tests/e2e`` are
    not mangled by :func:`_scope_command`'s naive dash-token extraction.

    Returns ``None`` when no ``.py`` files are found.
    """
    py_files = [f for f in task_files if f.endswith('.py')]
    if not py_files:
        return None

    # conftest.py cannot be passed directly to pytest (pytest >= 9 exits 1 with
    # "no tests ran").  The fallback path has no mc.test_command to reuse, so
    # we target the *parent directory* of each conftest instead — that directory
    # contains every test the conftest can affect.  A root-level conftest (no
    # parent) maps to '.' so we never produce 'pytest conftest.py'.  Sorted
    # deduped set gives deterministic output.
    has_conftest = any(_is_conftest(f) for f in py_files)
    # Narrow: only test_*.py / *_test.py that pytest will actually collect.
    # A data module under tests/ (e.g. tests/some_data.py) satisfies
    # _is_test_file but NOT _is_collectable_test_file — passing it to pytest
    # produces rc=5 ("no tests ran") → RED.  Task 1852 fixes at scoping layer.
    collectable_tests = [f for f in py_files if _is_collectable_test_file(f)]
    # has_test_data: in-tree but not collectable (and not conftest).
    # In _build_fallback_config there is no owning module suite to fall back to,
    # so we only propagate has_test_data to the non-default configured command
    # branch (where a real suite exists).  In the bare-pytest branch a lone data
    # module yields test_cmd = None — targeting its parent dir risks rc=5 if
    # that dir holds only fixtures/data with zero tests.
    has_test_data = any(
        _is_test_file(f) and not _is_collectable_test_file(f) for f in py_files
    )

    # Lint and type commands: use configured commands when *config* is provided.
    # _scope_command narrows to the touched files when the standard tool keyword
    # appears in the command, and returns unchanged when it doesn't.
    if config is not None:
        # _strip_leading_cd drops a ``cd <subproject> &&`` prefix that the
        # per-subproject lint/type commands carry: the fallback runs from the
        # worktree root, so a module-cd would misresolve the root-relative file
        # path the scoper just inserted.
        lint_cmd = _strip_leading_cd(
            _scope_command(config.lint_command, 'ruff check', py_files) or config.lint_command
        )
        type_cmd = _strip_leading_cd(
            _scope_command(config.type_check_command, 'pyright', py_files) or config.type_check_command
        )
    else:
        lint_cmd = 'ruff check ' + ' '.join(py_files)
        type_cmd = 'pyright ' + ' '.join(py_files)

    # Test command: when a non-default configured command exists (e.g.
    # `uv run --extra dev --extra web pytest -m 'not slow' --ignore=tests/e2e`),
    # use it as-is to avoid mangling multi-token flags.  The configured command
    # already encodes which extras, markers, and ignores are required.
    # has_test_data is included here: a data module (e.g. σ-allowlist) consumed
    # by real tests warrants running the full suite when a real suite exists —
    # the configured test_command has real tests, so rc≠5 (task 1852).
    if config is not None and config.test_command != 'pytest':
        test_cmd: str | None = config.test_command if (collectable_tests or has_conftest or has_test_data) else None
        return ModuleConfig(
            prefix='__fallback__',
            lint_command=lint_cmd,
            type_check_command=type_cmd,
            test_command=test_cmd,
        )

    if has_conftest:
        conftest_dirs = sorted({
            f.rsplit('/', 1)[0] if '/' in f else '.'
            for f in py_files
            if _is_conftest(f)
        })
        # Also include test files that live *outside* every conftest directory.
        # e.g. ['a/conftest.py', 'b/test_x.py'] → 'pytest a b/test_x.py' so
        # tests in b/ are not silently skipped.  A root-level conftest ('.')
        # shadows everything, so in that case no files are "outside".
        #
        # `collectable_tests` always contains file paths (e.g. 'a/sub/test_x.py'),
        # never bare directory paths — _is_collectable_test_file gates on filename
        # prefixes/suffixes, none of which match a directory entry.  That
        # guarantees `t.startswith(d + '/')` reliably means "t is inside
        # directory d" without false positives from a sibling like 'ab/'.
        if '.' not in conftest_dirs:
            outside = [
                t for t in collectable_tests
                if not any(t.startswith(d + '/') for d in conftest_dirs)
            ]
        else:
            outside = []
        targets = conftest_dirs + outside
        test_cmd = 'pytest ' + ' '.join(targets)
    elif collectable_tests:
        # Only collectable (test_*.py / *_test.py) files are targeted.
        # A lone data module under tests/ yields test_cmd = None (no rc=5).
        test_cmd = 'pytest ' + ' '.join(collectable_tests)
    else:
        test_cmd = None
        if has_test_data:
            # A test-tree data module (e.g. tests/some_data.py) is being skipped:
            # no collectable tests present and no configured suite to fall back
            # to.  The skip avoids a false-RED rc=5, but this change ships
            # UNVALIDATED.  Configure a non-default test_command for this project
            # so data-module changes are validated by a real suite (task 1852).
            _data_files = [
                f for f in py_files
                if _is_test_file(f) and not _is_collectable_test_file(f)
            ]
            logger.warning(
                '_build_fallback_config: test-tree data module(s) %s skipped '
                '— no tests will run for this change; configure a non-default '
                'test_command to validate data-module changes (task 1852)',
                _data_files,
            )

    return ModuleConfig(
        prefix='__fallback__',
        lint_command=lint_cmd,
        type_check_command=type_cmd,
        test_command=test_cmd,
    )


def _verify_duration_secs(runs: list[dict]) -> float:
    """Sum per-command ``duration_secs`` values from a verification runs list.

    Each entry is expected to have a ``duration_secs`` key (float); entries
    that are missing the key contribute 0.0.  Returns 0.0 for an empty list.

    This is the correct wall-clock measure when commands were run **serially**
    (sum of sequential durations).  For the **concurrent** branch (asyncio.gather
    of test/lint/type) the caller should use ``max(...)`` of the individual
    durations — mirroring the multi-module logic in ``_aggregate_results`` — to
    avoid overstating wall-clock by ~3×.
    """
    return sum(r.get('duration_secs', 0.0) for r in runs)


@dataclass
class VerifyResult:
    passed: bool
    test_output: str
    lint_output: str
    type_output: str
    summary: str
    timed_out: bool = False
    cause_hint: str = ''
    category: str = ''
    worktree_log_paths: list[str] = field(default_factory=list)
    archive_log_paths: list[str] = field(default_factory=list)
    # Wall-clock verify cost.  For a single-module run: max(test, lint, type)
    # when the three commands ran concurrently (asyncio.gather), or their sum
    # when run serially.  For a multi-module run: max across child
    # VerifyResults (set by _aggregate_results — modules run concurrently via
    # asyncio.gather so max approximates wall-time).  Defaults to 0.0 for
    # _trivial_pass and mocked results.
    #
    # compare=False: wall-clock duration differs between two independent runs of
    # the same logical verification, so it must NOT participate in __eq__ (else
    # cli_result == local can never hold — see test_cli
    # test_verify_merge_cli_wrapper_transparency). Folded in here to clear a
    # preexisting main red introduced by task 1802.
    duration_secs: float = field(default=0.0, compare=False)

    def failure_report(self) -> str:
        """Format all failures into a single report for the debugger."""
        sections = []
        if self.timed_out:
            # Lead with timeout info so the debugger knows the failure may not
            # be real code — list which commands actually hit the wall clock.
            timed_out_cmds = []
            if self.test_output and 'timed out' in self.test_output.lower():
                timed_out_cmds.append('test')
            if self.lint_output and 'timed out' in self.lint_output.lower():
                timed_out_cmds.append('lint')
            if self.type_output and 'timed out' in self.type_output.lower():
                timed_out_cmds.append('type')
            joined = ', '.join(timed_out_cmds) if timed_out_cmds else 'unknown'
            sections.append(
                f'## Verify Timed Out\n\nCommands that hit the timeout: {joined}.\n'
                f'This may indicate a cold build, resource contention, or a '
                f'genuinely hanging command — inspect the output below before '
                f'treating it as a real failure.'
            )
        if self.cause_hint:
            sections.append(f'## Failure Cause\n\n{self.cause_hint}')
        # ## Verify Logs — list on-disk paths so the reader can `cat` the full evidence.
        # Appears between ## Failure Cause and ## Test Failures.
        if self.worktree_log_paths or self.archive_log_paths:
            log_lines = ['## Verify Logs', '']
            if self.category:
                log_lines.append(f'Category: {self.category}')
                log_lines.append('')
            log_lines.append('Worktree:')
            for p in self.worktree_log_paths:
                log_lines.append(f'- {p}')
            if self.archive_log_paths:
                log_lines.append('')
                log_lines.append('Archive (durable, survives worktree cleanup):')
                for p in self.archive_log_paths:
                    log_lines.append(f'- {p}')
            sections.append('\n'.join(log_lines))
        if self.test_output and 'FAILED' in self.test_output:
            sections.append(f'## Test Failures\n\n```\n{self.test_output[-3000:]}\n```')
        if self.lint_output and self.lint_output.strip():
            sections.append(f'## Lint Issues\n\n```\n{self.lint_output[-2000:]}\n```')
        if self.type_output and 'error' in self.type_output.lower():
            sections.append(f'## Type Errors\n\n```\n{self.type_output[-2000:]}\n```')
        return '\n\n'.join(sections) if sections else self.summary


async def _kill_cgroup_scope(unit: str) -> None:
    """Force-kill a transient systemd ``--user`` scope unit and reap its cgroup.

    Sends SIGKILL to every process in the scope's cgroup, then stops the unit.
    Used when a verify command was spawned inside a transient scope (see
    ``_run_cmd``'s ``use_cgroup_scope`` path): killing the cgroup reaps the
    ENTIRE subtree (bash → cargo → rustc, and any inner ``timeout`` that
    setpgid'd cargo into a separate process group), which a plain ``killpg`` on
    the spawn pgid cannot reach.  Best-effort: every systemctl call is wrapped
    in ``suppress`` and bounded by a short timeout so a hung manager cannot
    block the verify path.
    """
    for action in (['kill', '--signal=SIGKILL', unit], ['stop', unit]):
        with contextlib.suppress(Exception):
            p = await asyncio.create_subprocess_exec(
                'systemctl', '--user', *action,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(p.wait(), 10)


# Environment variables that activate a *specific* Python virtualenv / uv
# project.  The orchestrator runs under `uv run --project orchestrator`, which
# activates dark-factory/.venv and exports these into our process env.  If they
# leak into a TARGET project's verify/build/test subprocess, the target's `uv`
# resolves THIS venv instead of its own and a target `uv sync` writes the
# target's deps into the orchestrator's runtime interpreter — the 2026-05-29
# ghost-venv incident (autopilot-video's torch/insightface stack got synced into
# dark-factory/.venv, flipping it 3.13->3.12 and deleting the live interpreter's
# stdlib out from under the running orchestrator, which then hit FileNotFoundError
# every scheduler cycle and stopped dispatching).
#
# Denylist, NOT allowlist: reify's cargo verify depends on a broad, evolving set
# of toolchain/sccache/jobserver vars (RUSTC_WRAPPER, CARGO_*, the jobserver
# FIFO, ...); an allowlist would silently break Rust verify the first time a new
# var is needed.  We remove ONLY the python-env-selection vars that cause the
# leak and pass everything else through untouched.
_VENV_ISOLATION_KEYS: frozenset[str] = frozenset({
    'VIRTUAL_ENV',
    'UV_PROJECT_ENVIRONMENT',
    'UV_PROJECT',
    'UV_ACTIVE',
    # Fix 3 may run the orchestrator unit with --frozen; UV_FROZEN / UV_NO_SYNC
    # must NOT leak to a target, which has to stay free to sync its own deps.
    'UV_FROZEN',
    'UV_NO_SYNC',
    # `uv run` (our ExecStart) sets UV_RUN_RECURSION_DEPTH in our env to guard
    # against runaway nested invocations.  A target's verify is a logically
    # independent uv invocation tree, so it must start that counter fresh rather
    # than inherit our depth — otherwise a uv-based target (dark-factory /
    # autopilot-video) would `uv run` "pre-loaded" at our depth.
    'UV_RUN_RECURSION_DEPTH',
    'CONDA_PREFIX',
    'CONDA_DEFAULT_ENV',
    'PYTHONHOME',
})


def _strip_venv_bin_from_path(path: str | None, venv: str | None) -> str | None:
    """Drop the active venv's ``bin`` directory from a PATH string.

    ``uv run`` prepends ``$VIRTUAL_ENV/bin`` to PATH in the orchestrator
    process, so removing ``VIRTUAL_ENV`` alone is insufficient: the venv's bin
    dir is still first on PATH and a target's bare ``python`` / ``uv`` / ``pip``
    would still resolve into the orchestrator venv.  Remove exactly that one
    component (matched as ``<venv>/bin`` via normpath); every other PATH entry
    (cargo, sccache, system bins) keeps its order.  Returns *path* unchanged
    when *path* or *venv* is falsy.
    """
    if not path or not venv:
        return path
    venv_bin = os.path.normpath(os.path.join(venv, 'bin'))
    kept = [
        p for p in path.split(os.pathsep)
        if p and os.path.normpath(p) != venv_bin
    ]
    return os.pathsep.join(kept)


def _target_subprocess_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Build the subprocess env for a TARGET project's verify/build/test spawn.

    Starts from ``os.environ`` minus the orchestrator's own venv/uv activation
    vars (``_VENV_ISOLATION_KEYS``) and minus the venv ``bin`` dir on PATH, so
    the target's toolchain resolves the target's OWN .venv.  Then injects
    ``PYTHONUNBUFFERED=1`` (the partial-log invariant — see ``_run_cmd``) and
    finally overlays *extra* (the caller's ``_resolve_verify_env`` result:
    ``DF_VERIFY_ROLE`` plus reify's ``RUSTC_WRAPPER`` / ``CARGO_*`` / jobserver
    vars) LAST, so target-supplied vars always win.
    """
    venv = os.environ.get('VIRTUAL_ENV')
    env = {k: v for k, v in os.environ.items() if k not in _VENV_ISOLATION_KEYS}
    stripped_path = _strip_venv_bin_from_path(env.get('PATH'), venv)
    if stripped_path is not None:
        env['PATH'] = stripped_path
    env['PYTHONUNBUFFERED'] = '1'
    if extra:
        env.update(extra)
    return env


async def _run_cmd(
    cmd: str,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
    log_path: 'Path | None' = None,
    *,
    use_cgroup_scope: bool = False,
) -> tuple[int, str, bool]:
    """Run a shell command, return (returncode, combined output, timed_out).

    When *env* is non-None, it is merged on top of ``os.environ`` and passed
    to the subprocess so callers can inject build accelerators like
    ``RUSTC_WRAPPER=sccache`` without mutating the parent process's env.

    When *use_cgroup_scope* is True and ``systemd-run`` is available, the
    command is launched inside a transient systemd ``--user --scope`` (its own
    cgroup) so a timeout/cancel can reap the WHOLE subtree by cgroup
    (``_kill_cgroup_scope``), regardless of process-group escapes — e.g. an
    inner GNU ``timeout`` that setpgid'd cargo into a separate group, which
    defeats the ``killpg``-on-spawn-pgid fallback and was the leak that let a
    defeated post-merge verify strand live ``cargo`` for up to 30 minutes.
    Falls back to the plain ``start_new_session`` + ``killpg`` path when the
    flag is off or ``systemd-run`` is missing, so the default behaviour and the
    existing test suite are unchanged.

    When *log_path* is provided, subprocess output is streamed (read in 4 KiB
    chunks and flushed) to that file as it arrives, so a timeout-killed child
    leaves the partial buffer on disk instead of producing a 0-byte file.  The
    accumulated buffer is also returned via the second tuple slot, identical
    to the legacy ``proc.communicate()`` contract.  When *log_path* is None
    no file is created.

    ``PYTHONUNBUFFERED=1`` is unconditionally injected into the subprocess env
    so that python children (pytest, ruff, pyright via uv) flush their stdout
    per-line — necessary for the partial-log invariant under heavy buffering.
    """
    # PYTHONUNBUFFERED is the cheap-but-decisive lever: without it pytest's
    # progress dots stay in stdio buffers and never reach our streaming loop,
    # so a hanging subprocess produces an opaque ``Command timed out after …``
    # cause hint with no actionable signal.
    # Build the target subprocess env via the venv-isolation scrub so the
    # TARGET's `uv` resolves the TARGET's .venv, never dark-factory/.venv (the
    # 2026-05-29 ghost-venv coupling).  The scrub strips the orchestrator's
    # venv/uv activation vars + the venv bin dir from PATH, sets
    # PYTHONUNBUFFERED, and reapplies the caller overlay (`env`) LAST so reify's
    # RUSTC_WRAPPER/CARGO_*/jobserver vars and DF_VERIFY_ROLE always win.
    subprocess_env: dict[str, str] = _target_subprocess_env(env)

    proc = None
    pgid: int | None = None
    scope_unit: str | None = None
    if use_cgroup_scope and shutil.which('systemd-run') is not None:
        scope_unit = f'df-verify-{uuid.uuid4().hex[:12]}.scope'
    try:
        if scope_unit is not None:
            # Launch inside a transient --user scope (its own cgroup) so a
            # timeout/cancel can reap the WHOLE subtree by cgroup.  --scope runs
            # bash as a direct child of systemd-run, inheriting our cwd/env and
            # forwarding stdio to our pipe; --collect auto-removes the scope when
            # it exits (no unit leak on the normal-completion path).
            proc = await asyncio.create_subprocess_exec(
                'systemd-run', '--user', '--scope', '--quiet', '--collect',
                f'--unit={scope_unit}',
                '/bin/bash', '-c', cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=subprocess_env,
                start_new_session=True,
            )
        else:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                executable='/bin/bash',
                env=subprocess_env,
                start_new_session=True,
            )
        # Capture pgid at spawn; start_new_session guarantees pgid == pid.
        pgid = proc.pid

        if log_path is None:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            rc = proc.returncode if proc.returncode is not None else 1
            return rc, stdout.decode(errors='replace'), False

        # Streamed path: chunked read + per-chunk flush so the kill on timeout
        # cannot strand the partial output in kernel buffers.  The ``with``
        # block guarantees the FD closes even when wait_for/CancelledError
        # unwinds through the outer try; the close happens before the except
        # handler runs, which is exactly what we want.
        buf = bytearray()
        with open(log_path, 'wb') as log_fh:
            async def _stream() -> None:
                assert proc is not None and proc.stdout is not None
                while True:
                    chunk = await proc.stdout.read(4096)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    log_fh.write(chunk)
                    log_fh.flush()
                await proc.wait()

            await asyncio.wait_for(_stream(), timeout=timeout)
        rc = proc.returncode if proc.returncode is not None else 1
        return rc, buf.decode(errors='replace'), False
    except TimeoutError:
        # cgroup kill (primary, reaps process-group escapes) then killpg backstop.
        if scope_unit is not None:
            await _kill_cgroup_scope(scope_unit)
        if proc is not None and pgid is not None:
            await terminate_process_group(proc, pgid, grace_secs=5.0)
        return 1, f'Command timed out after {timeout}s: {cmd}', True
    except asyncio.CancelledError:
        if scope_unit is not None:
            await _kill_cgroup_scope(scope_unit)
        if proc is not None and pgid is not None:
            await terminate_process_group(proc, pgid, grace_secs=5.0)
        raise
    except Exception as e:
        return 1, f'Command failed: {e}', False


# Marker file that records a worktree has completed at least one non-timeout verify.
_VERIFY_WARM_MARKER = 'verify_warmed'


def _warm_marker_name(module_prefix: str | None) -> str:
    """Return the marker filename for the given module prefix.

    When *module_prefix* is ``None`` the shared worktree marker is used
    (``verify_warmed``).  When a prefix is provided the marker is scoped to
    that subproject (``verify_warmed_<safe_prefix>``), preventing a successful
    subproject A from hiding a cold-build need for concurrently-run subproject B.
    Path separators and spaces are replaced with underscores.
    """
    if module_prefix is None:
        return _VERIFY_WARM_MARKER
    safe = module_prefix.replace('/', '_').replace(' ', '_')
    return f'{_VERIFY_WARM_MARKER}_{safe}'


def _is_verify_cold(worktree: Path, module_prefix: str | None = None) -> bool:
    """Return True when *worktree* has never completed a non-timeout verify.

    A worktree is considered cold when its ``.task/`` scratch directory exists
    but the ``verify_warmed`` marker inside it does not.  Paths without
    ``.task/`` (e.g., the project root used by review checkpoints) are treated
    as warm so that review-checkpoint verifies always use the standard timeout.

    When *module_prefix* is provided the check uses a per-subproject marker
    (``verify_warmed_{prefix}``), so one subproject completing successfully
    does not falsely warm-classify a concurrently-run sibling subproject.
    """
    task_dir = worktree / '.task'
    if not task_dir.is_dir():
        return False
    return not (task_dir / _warm_marker_name(module_prefix)).exists()


def _mark_verify_warm(worktree: Path, module_prefix: str | None = None) -> None:
    """Atomically mark *worktree* as warm by touching the verify_warmed marker.

    No-op when ``.task/`` is absent — we never create the scratch directory
    from within the verify path.  Idempotent (``exist_ok=True``).

    When *module_prefix* is provided the per-subproject marker is touched
    (``verify_warmed_{prefix}``) rather than the shared worktree marker.
    """
    task_dir = worktree / '.task'
    if not task_dir.is_dir():
        return
    marker_path = task_dir / _warm_marker_name(module_prefix)
    marker_path.touch(exist_ok=True)
    logger.debug('verify warm marker set: %s', marker_path)


def _resolve_verify_timeout(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
    *,
    is_cold: bool,
    is_merge_verify: bool = False,
) -> float:
    """Return the effective per-command verify timeout.

    When *is_cold* is False (warm cache), the warm timeout is returned:
    ``module_config.verify_command_timeout_secs`` takes precedence over
    ``config.verify_command_timeout_secs``.

    When *is_cold* is True (first verify in a fresh worktree), the cold
    timeout is resolved via the cascade:
      0. ``config.merge_verify_cold_command_timeout_secs`` (if set AND
         *is_merge_verify* is True) — merge-verify-specific cold budget;
         wins before the per-module and general cold knobs.
      1. ``module_config.verify_cold_command_timeout_secs`` (if set)
      2. ``config.verify_cold_command_timeout_secs`` (if set)
      3. The warm timeout computed above (fallback when cold knob is unset
         at every level — preserves existing behaviour for deployments that
         don't configure the cold window).

    *is_merge_verify* only affects the cold track; for warm resolves (or
    when ``config.merge_verify_cold_command_timeout_secs`` is None) the
    resolver falls through to the existing cascade unchanged.
    """
    # Warm track: module override wins over top-level.
    warm: float
    if module_config is not None and module_config.verify_command_timeout_secs is not None:
        warm = module_config.verify_command_timeout_secs
    else:
        warm = config.verify_command_timeout_secs

    if not is_cold:
        return warm

    # Cold track: merge-verify budget wins first, then cascade module → top → warm.
    if is_merge_verify and config.merge_verify_cold_command_timeout_secs is not None:
        return config.merge_verify_cold_command_timeout_secs
    if module_config is not None and module_config.verify_cold_command_timeout_secs is not None:
        return module_config.verify_cold_command_timeout_secs
    if config.verify_cold_command_timeout_secs is not None:
        return config.verify_cold_command_timeout_secs
    return warm


def _resolve_concurrent_verify(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
) -> bool:
    """Return whether test/lint/type should run concurrently.

    Module override wins over top-level config.
    """
    if module_config is not None and module_config.concurrent_verify is not None:
        return module_config.concurrent_verify
    return config.concurrent_verify


def _resolve_verify_env(
    config: OrchestratorConfig,
    module_config: ModuleConfig | None,
    *,
    role: Literal['merge', 'task'] = 'task',
) -> dict[str, str]:
    """Return the effective env injected into verify commands.

    Merges ``config.verify_env`` with ``module_config.verify_env``; module
    keys override top-level keys.  The orchestrator-supplied *role* is then
    stamped in as ``DF_VERIFY_ROLE`` and is always authoritative — it overrides
    any ``DF_VERIFY_ROLE`` entry that may appear in static config.
    """
    merged: dict[str, str] = {}
    merged.update(config.verify_env or {})
    if module_config is not None and module_config.verify_env:
        merged.update(module_config.verify_env)
    merged['DF_VERIFY_ROLE'] = role
    return merged


def _maybe_govern_merge_cmd(
    cmd: 'str | None',
    config: OrchestratorConfig,
    worktree: 'Path | None',
    role: str,
) -> 'str | None':
    """Wrap *cmd* in a cpu-governed-exec.sh invocation when role=='merge' and governance is enabled.

    Returns *cmd* unchanged when:

    * *cmd* is ``None`` (skip-guard — caller's ``if cmd is None: return`` fires first)
    * *role* is not ``'merge'`` (only merge-verify uses the merge-weighted cgroup scope)
    * ``config.cpu_governance`` is disabled (default; fail-open)
    * ``config.cpu_governance.resolved_exec_path(worktree)`` returns ``None``
      (non-executable, missing, or worktree=None; fail-open)

    Otherwise wraps as::

        <shlex.quote(exec_abs)> --role merge -- /bin/bash -c <shlex.quote(cmd)>

    so that shell operators (``&&``, ``|``, leading env assignments) in *cmd*
    survive intact inside the merge-weighted cgroup scope.  The inner
    ``/bin/bash -c <quoted>`` makes the whole original command a single argv
    payload for ``cpu-governed-exec.sh``.

    Does NOT alter ``_run_cmd``'s signature, the ``use_cgroup_scope`` path, or
    any merge PSI/semaphore bypass.

    **Interaction with verify_use_cgroup_scope**: when both
    ``config.cpu_governance.enabled`` *and* ``config.verify_use_cgroup_scope``
    are ``True``, ``_run_or_skip_timed`` wraps the command here first
    (so ``cpu-governed-exec.sh`` becomes ``argv[0]``), then passes
    ``use_cgroup_scope=True`` to ``_run_cmd``.  ``_run_cmd`` in turn launches
    the already-wrapped command inside a ``systemd-run --user --scope``
    (outer ``df-verify`` scope).  ``cpu-governed-exec.sh``, on its governed
    path, tries to create an *inner* ``systemd-run --user --scope`` scope —
    a nested transient scope inside the outer ``df-verify`` scope.  Nested
    ``--user --scope`` invocations are allowed by systemd (each creates a
    distinct cgroup slice), so this is not a correctness or leak bug; the
    outer scope's cgroup kill still reaps the entire subtree regardless.
    The live reify deployment currently sets ``verify_use_cgroup_scope=False``,
    so this combination does not occur in practice.  ``cpu-governed-exec.sh``
    also has a runtime probe + fail-open, so a nested-scope failure degrades
    gracefully.
    """
    if cmd is None or role != 'merge':
        return cmd
    gov = getattr(config, 'cpu_governance', None)
    if gov is None or not gov.enabled:
        return cmd
    exec_abs = gov.resolved_exec_path(worktree)
    if not exec_abs:
        return cmd
    return f'{shlex.quote(exec_abs)} --role merge -- /bin/bash -c {shlex.quote(cmd)}'


async def run_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_config: ModuleConfig | None = None,
    *,
    allow_cold_cache: bool = True,
    max_retries: int | None = None,
    is_merge_verify: bool = False,
    attempt_id: int | None = None,
    task_id: str | None = None,
    archive_root: Path | None = None,
    role: Literal['merge', 'task'] = 'task',
) -> VerifyResult:
    """Run test suite, linter, and type checker. Return structured result.

    When *module_config* is provided, a ``None`` command means "skip that check"
    (the subproject doesn't define it).  When *module_config* is ``None``,
    global config commands are used for every check.

    If any enabled command times out while the others pass, the whole verify
    is retried up to *max_retries* times (default ``config.verify_timeout_retries``).
    Pass ``max_retries=0`` to disable retries entirely — appropriate for
    merge-queue post-merge verification, where a deterministic hang would
    otherwise triple the queue-wide stall.  A retry that surfaces a genuine
    failure (e.g., a real lint error) is returned immediately instead of
    being retried further.

    When *allow_cold_cache* is ``False`` cold-timeout detection is disabled
    entirely regardless of filesystem state, and the warm timeout is always
    used.  Useful for review-checkpoint or eval callers that pass arbitrary
    paths which may happen to contain a ``.task/`` directory.  Defaults to
    ``True`` (auto-detect from filesystem).

    When *is_merge_verify* is ``True`` the verify is treated as always cold,
    regardless of filesystem state (merge worktrees are freshly created per
    merge — no ``.task/`` dir, but also no warm build cache — and so the
    ``_is_verify_cold`` filesystem heuristic mis-classifies them as warm).
    This bypasses ``_is_verify_cold`` entirely and uses the cold-track timeout
    cascade (``verify_cold_command_timeout_secs``).  Also implies
    ``allow_cold_cache=True`` semantics for the timeout; the warm marker is
    NOT written on success because merge worktrees are ephemeral.  Defaults
    to ``False`` for all non-merge callers so existing cold-detection
    behaviour is preserved.
    """
    if module_config is not None:
        # Scoped: use module command; None → skip
        test_cmd = module_config.test_command
        lint_cmd = module_config.lint_command
        type_cmd = module_config.type_check_command
    else:
        # Global fallback
        test_cmd = config.test_command
        lint_cmd = config.lint_command
        type_cmd = config.type_check_command

    module_prefix = module_config.prefix if module_config is not None else None
    if is_merge_verify:
        # Merge worktrees are freshly created per merge — cargo caches are
        # cold and the ``.task/`` marker is absent — so the filesystem
        # heuristic mis-classifies them as warm.  Force cold semantics.
        is_cold = True
    elif allow_cold_cache:
        is_cold = _is_verify_cold(worktree, module_prefix)
    else:
        is_cold = False
    timeout = _resolve_verify_timeout(config, module_config, is_cold=is_cold, is_merge_verify=is_merge_verify)
    if max_retries is None:
        max_retries = config.verify_timeout_retries

    if is_cold:
        warm_timeout = _resolve_verify_timeout(config, module_config, is_cold=False)
        if timeout != warm_timeout:
            logger.info(
                'Cold-cache verify: using %ds timeout (warm would be %ds)',
                int(timeout), int(warm_timeout),
            )
    concurrent = _resolve_concurrent_verify(config, module_config)
    verify_env = _resolve_verify_env(config, module_config, role=role)

    # DF_VERIFY_ROLE is always present (injected by _resolve_verify_env); log at
    # INFO only when user-configured keys also exist so we don't inflate INFO
    # volume on the hot verify path for plain task verifies.
    user_env_keys = set(verify_env.keys()) - {'DF_VERIFY_ROLE'}
    if user_env_keys:
        logger.info(
            'Verification env (mode=%s): %s',
            'concurrent' if concurrent else 'sequential',
            sorted(verify_env.keys()),
        )
    else:
        logger.debug(
            'Verification mode: %s',
            'concurrent' if concurrent else 'sequential',
        )

    # Resolve the streaming log path for each label.  Identical to the
    # filename computed in ``_persist_attempt_logs`` so that ``_run_cmd``'s
    # streamed file is the same file ``_persist_attempt_logs`` would have
    # written via ``write_text`` — the latter now skips the rewrite when the
    # streamed file already exists on disk.  Returns None when ``.task/`` is
    # absent (review-checkpoint / merge-queue paths), preserving the legacy
    # buffered behaviour for those callers.
    def _stream_log_path(label: str, current_attempt: int) -> 'Path | None':
        if attempt_id is None:
            return None
        task_dir = worktree / '.task'
        if not task_dir.is_dir():
            return None
        verify_dir = task_dir / 'verify'
        try:
            verify_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        if module_prefix is not None:
            safe = module_prefix.replace('/', '_').replace(' ', '_')
            infix = f'.{safe}'
        else:
            infix = ''
        return verify_dir / f'attempt-{current_attempt}{infix}.{label}.log'

    async def _run_or_skip_timed(
        cmd: str | None,
        *,
        label: str,
        current_attempt: int,
    ) -> tuple[int, str, bool, str | None, float]:
        """Like _run_cmd but returns (rc, out, timed_out, started_at_iso, duration_secs).

        When *cmd* is None (skipped check), returns (0, '', False, None, 0.0).
        """
        if cmd is None:
            return 0, '', False, None, 0.0
        # Wrap the command in cpu-governed-exec.sh when role=='merge' and
        # cpu_governance is enabled + exec resolves.  Fail-open: returns cmd
        # unchanged when governance is disabled or the path is non-executable,
        # so a misconfig never makes a verify spawn fail.
        cmd = _maybe_govern_merge_cmd(cmd, config, worktree, role)
        assert cmd is not None  # _maybe_govern_merge_cmd returns None only when cmd is None; guarded above
        started_at = datetime.now(UTC).isoformat()
        t0 = time.monotonic()
        # Pass use_cgroup_scope only when enabled so the default-off call
        # signature stays byte-identical (test doubles stub the legacy kwargs).
        _scope_kw = (
            {'use_cgroup_scope': True} if config.verify_use_cgroup_scope else {}
        )
        rc, out, timed_out_flag = await _run_cmd(
            cmd,
            worktree,
            timeout,
            env=verify_env or None,
            log_path=_stream_log_path(label, current_attempt),
            **_scope_kw,
        )
        return rc, out, timed_out_flag, started_at, time.monotonic() - t0

    # Pre-loop initialisation satisfies static analysis: mypy cannot prove that
    # `while True:` executes the body at least once before a break, so it
    # requires these to be assigned before their first use after the loop.
    # In practice the loop body always overwrites them on the first iteration;
    # these sentinel values are never read by any caller.
    test_started_at: str | None = None
    test_duration: float = 0.0
    lint_started_at: str | None = None
    lint_duration: float = 0.0
    type_started_at: str | None = None
    type_duration: float = 0.0

    attempt = 0
    while True:
        # attempt_id is the persistence ID handed in by the caller (or None for
        # callers that don't persist).  We use it directly as the streaming
        # attempt index so the streamed log path lines up with the path
        # ``_persist_attempt_logs`` computes below; this loop's local
        # ``attempt`` counter is for retry bookkeeping only.
        current_attempt_id = attempt_id if attempt_id is not None else 0
        if concurrent:
            (
                (test_rc, test_out, test_timed_out, test_started_at, test_duration),
                (lint_rc, lint_out, lint_timed_out, lint_started_at, lint_duration),
                (type_rc, type_out, type_timed_out, type_started_at, type_duration),
            ) = await asyncio.gather(
                _run_or_skip_timed(test_cmd, label='test', current_attempt=current_attempt_id),
                _run_or_skip_timed(lint_cmd, label='lint', current_attempt=current_attempt_id),
                _run_or_skip_timed(type_cmd, label='type', current_attempt=current_attempt_id),
            )
        else:
            test_rc, test_out, test_timed_out, test_started_at, test_duration = await _run_or_skip_timed(
                test_cmd, label='test', current_attempt=current_attempt_id,
            )
            lint_rc, lint_out, lint_timed_out, lint_started_at, lint_duration = await _run_or_skip_timed(
                lint_cmd, label='lint', current_attempt=current_attempt_id,
            )
            type_rc, type_out, type_timed_out, type_started_at, type_duration = await _run_or_skip_timed(
                type_cmd, label='type', current_attempt=current_attempt_id,
            )

        passed = test_rc == 0 and lint_rc == 0 and type_rc == 0
        any_timed_out = test_timed_out or lint_timed_out or type_timed_out

        # Check whether every failure is a timeout (no real rc!=0 without
        # timeout).  If so, the failure is a pure timeout and is retryable.
        pure_timeout_failure = (
            not passed
            and any_timed_out
            and (test_rc == 0 or test_timed_out)
            and (lint_rc == 0 or lint_timed_out)
            and (type_rc == 0 or type_timed_out)
        )

        if passed or not pure_timeout_failure or attempt >= max_retries:
            break

        attempt += 1
        timed_out_names = []
        if test_timed_out:
            timed_out_names.append('test')
        if lint_timed_out:
            timed_out_names.append('lint')
        if type_timed_out:
            timed_out_names.append('type')
        logger.warning(
            'Verification hit timeout on %s; retry %d/%d',
            ','.join(timed_out_names), attempt, max_retries,
        )

    # Classify timed_out: true only when the final failure was a pure timeout
    # (no real non-timeout failure mixed in).
    timed_out = (not passed) and pure_timeout_failure

    # Build summary
    if timed_out:
        summary = f'Verification timed out after {max_retries} retries' if max_retries > 0 else 'Verification timed out'
    else:
        parts = []
        if test_rc != 0:
            parts.append('tests failed')
        if lint_rc != 0:
            parts.append('lint issues')
        if type_rc != 0:
            parts.append('type errors')
        summary = 'All checks passed' if passed else f'Failures: {", ".join(parts)}'

    # Build cause_hint from each failing check's output; join with ' | '.
    # Also classify each failing check and pick the worst category.
    if passed:
        cause_hint = ''
        category = 'passed'
    else:
        hint_parts = []
        per_check_categories = []
        for rc, out, to in (
            (test_rc, test_out, test_timed_out),
            (lint_rc, lint_out, lint_timed_out),
            (type_rc, type_out, type_timed_out),
        ):
            if rc != 0:
                h = _extract_cause_hint(out)
                if h:
                    hint_parts.append(h)
                per_check_categories.append(_classify_failure(out, rc, to))
        cause_hint = ' | '.join(hint_parts)
        category = _worst_category(per_check_categories) if per_check_categories else 'unknown_test_failure'

    # Hoist runs list so both the merge-path and task-path branches can use it.
    runs = [
        {
            'label': 'test',
            'cmd': test_cmd,
            'rc': test_rc,
            'output': test_out,
            'timed_out': test_timed_out,
            'started_at': test_started_at or '',
            'duration_secs': test_duration,
        },
        {
            'label': 'lint',
            'cmd': lint_cmd,
            'rc': lint_rc,
            'output': lint_out,
            'timed_out': lint_timed_out,
            'started_at': lint_started_at or '',
            'duration_secs': lint_duration,
        },
        {
            'label': 'type',
            'cmd': type_cmd,
            'rc': type_rc,
            'output': type_out,
            'timed_out': type_timed_out,
            'started_at': type_started_at or '',
            'duration_secs': type_duration,
        },
    ]

    worktree_log_paths: list[str] = []
    archive_log_paths: list[str] = []
    if role == 'merge':
        # Merge worktrees have .task/ scrubbed by design (git_ops.py); there
        # are no worktree log files to copy.  Write directly to the durable
        # archive instead.  No deny-list check: on the merge path there is no
        # debugger loop — every failure goes straight to a human/steward —
        # so infra_timeout and test_failure (the exact categories that
        # distinguish timeout-vs-real-failure) must be archived
        # unconditionally.  archive_root is not None is the discriminator
        # that auto-excludes cold-shadow / drift paths (left at None).
        if archive_root is not None and task_id is not None and not passed:
            try:
                arch_paths = _archive_merge_verify_logs(
                    runs, archive_root, task_id, attempt_id or 1,
                    category, cause_hint, module_prefix=module_prefix,
                )
                archive_log_paths = [str(p) for p in arch_paths]
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    'run_verification: merge archival error (non-fatal): %s', exc,
                )
    elif attempt_id is not None and task_id is not None:
        # Task path: persist to worktree/.task/verify/ then optionally copy
        # to the durable archive when category warrants it.
        try:
            wt_paths = _persist_attempt_logs(
                worktree, attempt_id, runs, category, cause_hint,
                module_prefix=module_prefix,
            )
            worktree_log_paths = [str(p) for p in wt_paths]
            arch_paths = _archive_attempt_log(
                wt_paths, archive_root, task_id, attempt_id, category,
            )
            archive_log_paths = [str(p) for p in arch_paths]
        except Exception as exc:  # noqa: BLE001
            logger.warning('run_verification: persistence error (non-fatal): %s', exc)

    # When the three verify commands ran concurrently (asyncio.gather) the
    # true wall-clock cost is the longest single command, not their sum.
    # Serial mode is rare (legacy / explicit opt-out) and sums correctly.
    if concurrent:
        _wall_secs = max(test_duration, lint_duration, type_duration)
    else:
        _wall_secs = _verify_duration_secs(runs)

    result = VerifyResult(
        passed=passed,
        test_output=test_out,
        lint_output=lint_out if lint_rc != 0 else '',
        type_output=type_out if type_rc != 0 else '',
        summary=summary,
        timed_out=timed_out,
        cause_hint=cause_hint,
        category=category,
        worktree_log_paths=worktree_log_paths,
        archive_log_paths=archive_log_paths,
        duration_secs=_wall_secs,
    )

    # Mark the worktree warm whenever the build completed (no pure timeout),
    # so subsequent verifies use the faster warm timeout.  The marker is
    # per-subproject (keyed by module_prefix) so a concurrent sibling
    # subproject that times out remains cold on the next attempt.
    # Skip the marker for merge-queue verifies: merge worktrees are
    # ephemeral (cleaned up right after), and their `.task/` dir is absent
    # anyway — `_mark_verify_warm` would be a no-op, but the skip keeps the
    # intent explicit.
    if not result.timed_out and not is_merge_verify:
        _mark_verify_warm(worktree, module_prefix)

    if passed:
        logger.info('Verification passed: %s', summary)
    else:
        # Use the richer format when we have a category and a persisted log path —
        # this avoids dumping the raw blob into the orchestrator log.
        if result.category and result.worktree_log_paths:
            hint_part = result.cause_hint or '<no hint>'
            log_ref = result.worktree_log_paths[0]
            log_msg = 'Verification failed: %s — %s (full log: %s)'
            if timed_out:
                logger.warning(log_msg, result.category, hint_part, log_ref)
            else:
                logger.info(log_msg, result.category, hint_part, log_ref)
        else:
            # Legacy format — no log path available (merge-queue, review-checkpoint,
            # or path outside .task/).
            detail_tail = f' — {cause_hint}' if cause_hint else ''
            if timed_out:
                logger.warning('Verification failed: %s%s', summary, detail_tail)
            else:
                logger.info('Verification failed: %s%s', summary, detail_tail)
    return result


def _aggregate_results(results: list[VerifyResult]) -> VerifyResult:
    """Merge per-subproject VerifyResults into one."""
    if len(results) == 1:
        return results[0]

    passed = all(r.passed for r in results)
    test_output = '\n'.join(r.test_output for r in results if r.test_output)
    lint_output = '\n'.join(r.lint_output for r in results if r.lint_output)
    type_output = '\n'.join(r.type_output for r in results if r.type_output)

    # Aggregate timed_out: true only when every failing subproject failed
    # purely due to timeout.  A single real failure poisons the signal.
    failing = [r for r in results if not r.passed]
    timed_out = (not passed) and bool(failing) and all(r.timed_out for r in failing)

    if timed_out:
        summary = 'Verification timed out'
    else:
        parts = []
        if any('tests failed' in r.summary for r in results):
            parts.append('tests failed')
        if any('lint issues' in r.summary for r in results):
            parts.append('lint issues')
        if any('type errors' in r.summary for r in results):
            parts.append('type errors')
        summary = 'All checks passed' if passed else f'Failures: {", ".join(parts)}'

    # Collect cause_hint from failing child results; join with ' | '.
    cause_hint = ' | '.join(r.cause_hint for r in results if r.cause_hint)

    # Pick the worst child category by priority.
    # Empty string is filtered out to avoid pulling the aggregate to '' when
    # legacy callers (no-persistence path) produced results with no category.
    # 'passed' is intentionally included when present — _worst_category correctly
    # orders failures above 'passed', so a mix of passing and failing children
    # still resolves to the worst failure.  If all children pass the aggregate
    # category will be 'passed', which is the correct result.
    child_categories = [r.category for r in results if r.category]
    category = _worst_category(child_categories) if child_categories else ''

    # Flatten per-child log path lists.
    worktree_log_paths: list[str] = []
    archive_log_paths: list[str] = []
    for r in results:
        worktree_log_paths.extend(r.worktree_log_paths)
        archive_log_paths.extend(r.archive_log_paths)

    return VerifyResult(
        passed=passed,
        test_output=test_output,
        lint_output=lint_output,
        type_output=type_output,
        summary=summary,
        timed_out=timed_out,
        cause_hint=cause_hint,
        category=category,
        worktree_log_paths=worktree_log_paths,
        archive_log_paths=archive_log_paths,
        # Wall-clock approximation: modules run concurrently via asyncio.gather
        # so the slowest module dominates the total elapsed time.  Single-module
        # tasks hit the len==1 fast path above and carry the exact value.
        duration_secs=max((r.duration_secs for r in results), default=0.0),
    )


async def run_full_verification(
    project_root: Path,
    config: OrchestratorConfig,
    *,
    force_rediscover: bool = False,
) -> VerifyResult:
    """Run verification for ALL subprojects against the project root.

    Unlike run_scoped_verification, this runs full (unscoped) test suites
    for every subproject that has an orchestrator.yaml. Used by review
    checkpoints to check integration health across the whole codebase.

    Discovery reuse: ``config._module_configs`` uses a sentinel of ``None`` to
    mean "discovery never ran".  When it holds any dict (including ``{}``,
    meaning discovery ran and found no subprojects) and *project_root* resolves
    to the same absolute path as ``config.project_root``, the pre-discovered
    dict is reused directly and no additional filesystem walk is performed.  A
    fresh walk via ``_discover_module_configs`` is retained as a fallback for
    two cases: (1) *project_root* differs from ``config.project_root``; or (2)
    the config was constructed without going through ``load_config`` (e.g.
    direct instantiation in tests) so ``_module_configs`` is still ``None``.

    **Staleness note**: the reuse path reads a load_config-time snapshot, so a
    subproject whose ``orchestrator.yaml`` is added or merged *after* startup
    is absent from full verification for the remainder of the run.  This is an
    intentional trade-off: the snapshot eliminates the redundant filesystem walk
    on the hot production review-checkpoint path, where the vast majority of
    runs see a stable module set.  A mid-run addition is rare; documenting the
    trade-off and providing an explicit escape hatch is the right balance.

    To force fresh discovery (e.g. after a merge that introduces a new module),
    pass ``force_rediscover=True``.  The primary production caller
    ``review_checkpoint.py`` does **not** pass this flag — it keeps the fast
    snapshot path.  The ``force_rediscover`` parameter is keyword-only so call
    sites that opt in must state the intent explicitly.
    """
    from orchestrator.config import _discover_module_configs

    resolved = project_root.resolve()
    if not force_rediscover and config._module_configs is not None and resolved == config.project_root:
        module_configs = config._module_configs
    else:
        module_configs = _discover_module_configs(project_root)
    if not module_configs:
        logger.info('Full verification: no subproject configs — using global')
        return await run_verification(project_root, config)

    logger.info(
        'Full verification: running %d subprojects in parallel',
        len(module_configs),
    )
    results = await asyncio.gather(
        *(run_verification(project_root, config, mc) for mc in module_configs.values())
    )
    return _aggregate_results(list(results))


async def run_scoped_verification(
    worktree: Path,
    config: OrchestratorConfig,
    module_configs: list[ModuleConfig],
    task_files: list[str] | None = None,
    *,
    max_retries: int | None = None,
    is_merge_verify: bool = False,
    attempt_id: int | None = None,
    task_id: str | None = None,
    archive_root: Path | None = None,
    force_workspace: bool = False,
    role: Literal['merge', 'task'] = 'task',
) -> VerifyResult:
    """Run verification scoped to specific subprojects and optionally to task files.

    Scoping modes (in priority order):

    1. **File-scoped within subprojects** — when *module_configs* is non-empty
       and *task_files* is provided, each ModuleConfig's commands are narrowed
       to the specific files via :func:`scope_module_config`.  Subprojects
       with zero matching files are skipped entirely.
    2. **Fallback-scoped** — when *module_configs* is empty and *task_files* is
       provided, a synthetic ModuleConfig is built via
       :func:`_build_fallback_config`, bypassing the global commands entirely.
    3. **Global** — when *task_files* is ``None`` (or falsy) with no
       module_configs, or when fallback returns ``None`` (no .py files).
    4. **Workspace (train-member override)** — when *force_workspace* is
       ``True``, all scoping is bypassed and the project-wide commands from
       *config* (e.g. ``cargo test --workspace``) run against the worktree
       branch tip.  Mirrors the *is_merge_verify* flag-threading pattern
       (PRD §9.5, γ₁).  Non-train callers leave this at its default ``False``
       for byte-identical existing behaviour.

    *max_retries* overrides ``config.verify_timeout_retries`` for this call;
    pass ``0`` from the merge-queue path so a deterministic hang doesn't
    triple the stall.

    *is_merge_verify* is forwarded unchanged to every :func:`run_verification`
    call.  Merge worktrees are freshly created (no warm cargo cache) but lack
    ``.task/``, which would otherwise misclassify them as warm via
    :func:`_is_verify_cold`.  Set this to ``True`` from the merge-queue
    call sites so post-merge verifies get the cold timeout.
    """
    scope_cargo_enabled = config.scope_cargo

    # Bound the per-subproject fan-out so a large (or accidentally polluted)
    # module set can never launch an unbounded number of full builds into one
    # worktree at once.  Created per call; a no-op when only one module runs.
    # (Root cause of the 226-way merge-verify storm: a polluted module set
    # turned the no-match fan-out into 226 concurrent `cargo` pipelines in one
    # `_merge-*` worktree.)
    _fanout_sem = asyncio.Semaphore(max(1, config.max_concurrent_module_verifies))

    async def _verify_module(mc: ModuleConfig) -> 'VerifyResult':
        async with _fanout_sem:
            return await run_verification(
                worktree, config, mc,
                max_retries=max_retries,
                is_merge_verify=is_merge_verify,
                attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                role=role,
            )

    # When the plan didn't provide a file list, try to derive one from git —
    # but only when we're not bypassing scoping entirely (force_workspace=True
    # goes straight to the global run_verification call without needing a file
    # list).
    if task_files is None and not force_workspace:
        task_files = await _derive_task_files_from_git(worktree, config)

    # _prune_archive runs exactly once in the finally block regardless of which
    # branch returns, preventing concurrent per-module prune races and removing
    # the repetitive guard at every return site.
    try:
        # Train-member workspace override: bypass ALL scoping and run the
        # project-wide workspace command verbatim (config.test_command/…).
        # Cargo --workspace→-p rewrites only fire when task_files are passed,
        # so this path runs `cargo test --workspace` (or the configured command)
        # unchanged against the worktree branch tip.
        if force_workspace:
            logger.info(
                'Verification mode: workspace (train member — file-scoping bypassed)'
            )
            return await run_verification(
                worktree, config,
                max_retries=max_retries,
                is_merge_verify=is_merge_verify,
                attempt_id=attempt_id,
                task_id=task_id,
                archive_root=archive_root,
                role=role,
            )
        if module_configs:
            # Apply file-level scoping within each subproject when task_files given
            if task_files:
                # Filter to files that still exist — tasks may delete files as part of their work
                existing_files = [f for f in task_files if (worktree / f).exists()]
                # scope_module_config returns None when no files touch the subproject;
                # those subprojects are skipped rather than running their full suite.
                per_module = [
                    (mc.prefix, scope_module_config(mc, existing_files, worktree=worktree))
                    for mc in module_configs
                ]
                skipped = [prefix for prefix, scoped_mc in per_module if scoped_mc is None]
                scoped = [scoped_mc for _prefix, scoped_mc in per_module if scoped_mc is not None]
                if skipped:
                    logger.info(
                        'Verification scope: skipping %d subproject(s) with no matching files: %s',
                        len(skipped), ', '.join(skipped),
                    )
                if not scoped:
                    # No subproject has matching files. Two sub-cases:
                    #   (a) Diff has no .py/.rs at all (docs, YAML, JSON …) —
                    #       every existing scope branch would no-op anyway; the
                    #       previous global-pytest fall-through was unsafe in
                    #       this layout. Trivially pass — UNLESS the merge-role
                    #       verify-pipeline-guard says a full gate is required
                    #       (e.g. diff touches verify.sh which shifts plan-line
                    #       counts — the drift-ambush class).
                    #   (b) Source files exist but don't fit any prefix
                    #       (e.g. root-level conftest.py + skills/*.md). Fan
                    #       out per-subproject so each runs in its own venv
                    #       with its own pyproject options.
                    if not _has_source_files(existing_files):
                        should_override = (
                            role == 'merge'
                            and await _verify_pipeline_guard_requires_full_gate(
                                worktree, existing_files,
                            )
                        )
                        if should_override:
                            logger.info(
                                'config-only fast-path overridden by verify-pipeline-guard'
                                ' — running full gate (module_configs merge path)',
                            )
                            # Fall through to per-subproject fan-out below.
                        else:
                            logger.info(
                                'Verification mode: trivial pass (no source files in diff)',
                            )
                            return _trivial_pass(
                                'No source files changed — verify trivially passes',
                            )
                    logger.info(
                        'Verification mode: per-subproject fan-out (%d subprojects)',
                        len(module_configs),
                    )
                    results = await asyncio.gather(*(
                        _verify_module(mc) for mc in module_configs
                    ))
                    return _aggregate_results(list(results))
                # Rewrite cargo --workspace → cargo -p <crate> when all task files
                # are .rs and map to known workspace crates.
                scoped = [
                    _apply_cargo_scope(mc, existing_files, worktree, scope_cargo_enabled)
                    for mc in scoped
                ]
                n_files = len(existing_files)
                n_mods = len(scoped)
                logger.info('Verification mode: file-scoped (%d files across %d subprojects)', n_files, n_mods)
            else:
                scoped = module_configs
                logger.info('Verification mode: subproject-scoped (%d subprojects)', len(module_configs))
            results = await asyncio.gather(
                *(_verify_module(mc) for mc in scoped)
            )
            return _aggregate_results(list(results))

        # No module_configs — try fallback or global
        if task_files:
            # Filter to files that still exist — tasks may delete files as part of their work
            existing_files = [f for f in task_files if (worktree / f).exists()]
            # Mirror the same docs-only short-circuit as the module_configs
            # branch: with no .py/.rs files _build_fallback_config would
            # return None and we'd fall through to the unsafe global pytest.
            if not _has_source_files(existing_files):
                should_override = (
                    role == 'merge'
                    and await _verify_pipeline_guard_requires_full_gate(
                        worktree, existing_files,
                    )
                )
                if should_override:
                    logger.info(
                        'config-only fast-path overridden by verify-pipeline-guard'
                        ' — running full gate (no-module_configs merge path)',
                    )
                    # Fall through to the existing global run_verification path.
                else:
                    logger.info(
                        'Verification mode: trivial pass (no source files, no module configs)',
                    )
                    return _trivial_pass(
                        'No source files changed — verify trivially passes',
                    )
            fallback = _build_fallback_config(existing_files, config)
            if fallback is not None:
                fallback = _apply_cargo_scope(
                    fallback, existing_files, worktree, scope_cargo_enabled,
                )
                logger.info('Verification mode: fallback-scoped (%d files)', len(existing_files))
                return await run_verification(
                    worktree, config, fallback, max_retries=max_retries,
                    is_merge_verify=is_merge_verify,
                    attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                    role=role,
                )

            # For Rust projects with no module_configs and no Python fallback
            # (Reify's layout), try to scope the global commands.
            if existing_files and scope_cargo_enabled:
                synthetic = ModuleConfig(
                    prefix='__cargo_scoped__',
                    test_command=config.test_command,
                    lint_command=config.lint_command,
                    type_check_command=config.type_check_command,
                )
                rewritten = _apply_cargo_scope(
                    synthetic, existing_files, worktree, scope_cargo_enabled,
                )
                if rewritten is not synthetic:
                    logger.info(
                        'Verification mode: cargo-scoped (%d .rs files)',
                        len(existing_files),
                    )
                    return await run_verification(
                        worktree, config, rewritten, max_retries=max_retries,
                        is_merge_verify=is_merge_verify,
                        attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                        role=role,
                    )

        logger.info('Verification mode: global (no scope info)')
        return await run_verification(
            worktree, config, max_retries=max_retries,
            is_merge_verify=is_merge_verify,
            attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
            role=role,
        )
    finally:
        _maybe_prune_archive(archive_root)


async def verify_failure_is_preexisting_on_main(
    worktree: Path,
    config: 'OrchestratorConfig',
    module_configs: 'list[ModuleConfig]',
    task_files: 'list[str] | None',
    failing_result: VerifyResult,
    git_ops: object,
) -> tuple[bool, str]:
    """Detect whether *failing_result* is inherited from the current main HEAD.

    Returns:
        ``(True, main_sha)`` iff the same (category, normalised cause_hint) signature
        reproduces on main — the break is preexisting.  The caller can reuse
        *main_sha* for fingerprint composition without a second ``get_main_sha`` call.

        ``(False, '')`` (fail-safe) for any of:
          - Main probe passes (break is task-own).
          - Different signature (break is task-own or different inherited break).
          - ``git_ops.get_main_sha()`` fails or returns empty.
          - ``git worktree add --detach`` fails after retries.
          - Any unexpected exception during probing.

    A process-wide TTL cache (keyed by (main_sha, category, normalised cause_hint))
    avoids redundant probes from the same or sibling tasks against an unchanged main.

    The task worktree is NEVER mutated — all git ops target *config.project_root*
    (for worktree-level commands) and the detached probe path (for probe verify).
    Cleanup (worktree remove --force + shutil.rmtree of the probe dir) always
    runs in a ``finally`` block.  No broad ``git worktree prune`` is issued so
    concurrently-active sibling probes are not disturbed.

    The probe worktree is created under *git_ops.worktree_base* with a
    ``_mainprobe-<id>`` prefix (mirroring ``_create_merge_worktree``'s
    ``_merge-<id>`` scheme).  Placement under worktree_base ensures environment
    parity: upward directory traversal resolves node_modules / repo-root shared
    installs exactly as task worktrees do.  The ``_mainprobe-`` prefix is
    distinct from ``_merge-`` so the disk-pressure prune
    (``prune_stale_merge_worktrees``, targeting ``_merge-*`` only) never
    reclaims the probe mid-run.
    """
    import uuid

    from orchestrator.git_ops import _run

    # Lazy-import normalisation helper from workflow to avoid the
    # verify<->workflow import cycle (workflow imports verify at module level).
    # Using the same normaliser as the loop-guard ensures the comparison is
    # apples-to-apples.
    def _normalize(hint: str | None) -> str:
        try:
            from orchestrator.workflow import _normalize_cause_hint
            return _normalize_cause_hint(hint)
        except Exception:
            return (hint or '').strip().lower()

    # tmp_path: probe worktree path under git_ops.worktree_base.
    # Using worktree_base/<name> (not /tmp) ensures the same upward directory
    # traversal as task worktrees for node_modules / repo-root dependencies.
    # The '_mainprobe-' prefix keeps it distinct from '_merge-*' so the disk-
    # pressure prune (prune_stale_merge_worktrees) never reclaims it mid-run.
    # git worktree add CREATES tmp_path; we must NOT pre-create it or git will
    # reject an already-present directory on strict versions.
    tmp_path: Path | None = None
    worktree_added: bool = False
    try:
        # Resolve the current main SHA.
        try:
            main_sha: str = await git_ops.get_main_sha()  # type: ignore[union-attr]
        except Exception:
            logger.debug('verify_failure_is_preexisting_on_main: get_main_sha failed', exc_info=True)
            return False, ''
        if not main_sha:
            return False, ''

        # Check the process-wide probe cache before paying the worktree-add cost.
        _norm_hint = _normalize(failing_result.cause_hint)
        _cache_key = (main_sha, failing_result.category or '', _norm_hint)
        _now = time.monotonic()
        if _cache_key in _PROBE_CACHE:
            _cached_at, _cached = _PROBE_CACHE[_cache_key]
            if _now - _cached_at < _PROBE_CACHE_TTL:
                logger.debug(
                    'verify_failure_is_preexisting_on_main: cache hit '
                    '(main_sha=%.8s, preexisting=%s)', main_sha, _cached,
                )
                return _cached, (main_sha if _cached else '')

        # Create the probe worktree path under worktree_base so upward directory
        # traversal resolves node_modules / repo-root installs identically to
        # task worktrees.  git worktree add CREATES the path, so we must NOT
        # pre-create it (strict git rejects pre-existing directories).
        base: Path = git_ops.worktree_base  # type: ignore[union-attr]
        base.mkdir(parents=True, exist_ok=True)
        tmp_path = base / f'_mainprobe-{uuid.uuid4().hex[:8]}'

        # Retry worktree add on transient git lock contention (serialised metadata
        # writes mean concurrent sibling probes can hit LOCK_MAX).
        _MAX_ADD_RETRIES = 3
        rc, _, err = 1, '', 'not attempted'
        for _attempt in range(_MAX_ADD_RETRIES):
            rc, _, err = await _run(
                ['git', 'worktree', 'add', '--detach', str(tmp_path), main_sha],
                cwd=config.project_root,
            )
            if rc == 0:
                worktree_added = True
                break
            if _attempt < _MAX_ADD_RETRIES - 1:
                await asyncio.sleep(0.5 * (_attempt + 1))
        if not worktree_added:
            logger.warning(
                'verify_failure_is_preexisting_on_main: worktree add failed after %d retries '
                '(rc=%d): %s — contagion guard disabled for this attempt',
                _MAX_ADD_RETRIES, rc, err,
            )
            return False, ''

        # Probe main with the same scoped commands, no retries.
        try:
            main_result = await run_scoped_verification(
                tmp_path, config, module_configs,
                task_files=task_files,
                max_retries=0,
                role='task',
            )
        except Exception:
            logger.debug(
                'verify_failure_is_preexisting_on_main: probe verify raised', exc_info=True,
            )
            return False, ''

        if main_result.passed:
            # Main is clean — the break is task-own.
            _PROBE_CACHE[_cache_key] = (time.monotonic(), False)
            return False, ''

        # Compare (category, normalised cause_hint).
        branch_sig = (failing_result.category or '', _norm_hint)
        main_sig = (main_result.category or '', _normalize(main_result.cause_hint))
        is_preexisting = branch_sig == main_sig
        _PROBE_CACHE[_cache_key] = (time.monotonic(), is_preexisting)
        return is_preexisting, (main_sha if is_preexisting else '')

    except Exception:
        logger.debug('verify_failure_is_preexisting_on_main: unexpected error', exc_info=True)
        return False, ''
    finally:
        # Scoped cleanup: remove only this specific probe worktree.
        # INTENTIONALLY NO 'git worktree prune' here (DD5 guarantee): a broad prune
        # would deregister ANY concurrently-active sibling probe (other tasks running
        # verify_failure_is_preexisting_on_main in parallel), causing their git
        # worktree add to succeed but probe path to vanish mid-verify.  Scoped
        # 'git worktree remove --force <tmp_path>' deregisters ONLY this probe.
        if worktree_added and tmp_path is not None:
            try:
                await _run(
                    ['git', 'worktree', 'remove', '--force', str(tmp_path)],
                    cwd=config.project_root,
                )
            except Exception:
                logger.debug(
                    'verify_failure_is_preexisting_on_main: worktree remove failed',
                    exc_info=True,
                )
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                # Belt-and-suspenders: rmtree the probe dir in case git worktree
                # remove left an empty skeleton, or the worktree add never ran.
                shutil.rmtree(tmp_path, ignore_errors=True)


async def run_main_tip_sweep(
    config: 'OrchestratorConfig',
    git_ops: object,
    *,
    main_sha: str | None = None,
) -> 'tuple[str, VerifyResult] | None':
    """Run a full unscoped verification sweep against the current main-tip SHA.

    Creates a throwaway detached worktree pinned at *main_sha* under
    ``git_ops.worktree_base`` (``_mainsweep-<hex>`` prefix — distinct from
    ``_merge-``/``_mainprobe-`` so the disk-pressure prune never reclaims it
    mid-run).  Runs ``run_full_verification`` (all subprojects: test + lint +
    typecheck in parallel) and returns ``(main_sha, result)``.

    Args:
        config: Orchestrator configuration.
        git_ops: GitOps instance (needs ``get_main_sha``, ``worktree_base``).
        main_sha: Optional pre-resolved SHA.  When provided the internal
            ``git_ops.get_main_sha()`` call is skipped, eliminating a redundant
            subprocess and closing the TOCTOU window between the harness
            SHA-dedup gate and the worktree pin.  Callers that already resolved
            the SHA (e.g. ``_run_main_tip_sweep`` in harness.py) should pass it.

    Returns:
        ``(main_sha, VerifyResult)`` on success (result.passed may be False).
        ``None`` (fail-safe) when:
          - ``get_main_sha()`` raises or returns an empty string.
          - ``git worktree add --detach`` fails after retries.
          - Any unexpected exception during sweep setup.
        The harness treats ``None`` as "no signal — retry next tick" and does
        NOT mark the SHA as swept, so the same tip is retried on the next interval.

    Cleanup: scoped ``git worktree remove --force <tmp_path>`` + ``shutil.rmtree``
    always runs in a ``finally`` block.  NO broad ``git worktree prune`` (DD5
    guarantee: a broad prune would deregister concurrently-active sibling
    probe/merge worktrees).
    """
    import uuid  # noqa: PLC0415, I001
    from orchestrator.git_ops import _run  # noqa: PLC0415 — lazy, mirrors verify_failure_is_preexisting_on_main

    # git worktree add CREATES tmp_path; do NOT pre-create or strict git rejects it.
    tmp_path: Path | None = None
    worktree_added: bool = False
    try:
        # Resolve the current main SHA unless the caller pre-resolved it.
        # Accepting a pre-resolved value eliminates a redundant git rev-parse
        # subprocess and closes the TOCTOU window between the harness SHA-dedup
        # gate and the worktree pin (both now use the same resolved value).
        if main_sha is None:
            try:
                main_sha = await git_ops.get_main_sha()  # type: ignore[union-attr]
            except Exception:
                logger.debug('run_main_tip_sweep: get_main_sha failed', exc_info=True)
                return None
        if not main_sha:
            return None

        # Build the sweep worktree path under worktree_base.  The '_mainsweep-'
        # prefix is distinct from '_merge-' and '_mainprobe-' so the disk-pressure
        # prune (prune_stale_merge_worktrees, targeting '_merge-*' only) never
        # reclaims the probe mid-run.  Same env-parity reasoning as mainprobe.
        base: Path = git_ops.worktree_base  # type: ignore[union-attr]
        base.mkdir(parents=True, exist_ok=True)
        tmp_path = base / f'_mainsweep-{uuid.uuid4().hex[:8]}'

        # Retry worktree add on transient git lock contention (serialised metadata
        # writes mean concurrent sibling probes can hit LOCK_MAX).
        _MAX_ADD_RETRIES = 3
        rc, _, err = 1, '', 'not attempted'
        for _attempt in range(_MAX_ADD_RETRIES):
            rc, _, err = await _run(
                ['git', 'worktree', 'add', '--detach', str(tmp_path), main_sha],
                cwd=config.project_root,  # type: ignore[union-attr]
            )
            if rc == 0:
                worktree_added = True
                break
            if _attempt < _MAX_ADD_RETRIES - 1:
                await asyncio.sleep(0.5 * (_attempt + 1))
        if not worktree_added:
            logger.warning(
                'run_main_tip_sweep: worktree add failed after %d retries '
                '(rc=%d): %s — sweep skipped for this tick',
                _MAX_ADD_RETRIES, rc, err,
            )
            return None

        # Run full (unscoped) verification — all discovered subprojects, no scope filter.
        result = await run_full_verification(tmp_path, config)  # type: ignore[arg-type]

        # pytest INTERNALERROR means the test infrastructure itself crashed (e.g. an
        # xdist worker was killed by os._exit).  Treat it as an infra failure — return
        # the None sentinel so the harness retries next tick and files no false-positive
        # drift L1.  The finally block's worktree cleanup still runs.
        if result.category == 'pytest_internalerror':
            logger.warning(
                'run_main_tip_sweep: pytest INTERNALERROR in sweep — '
                'treating as infra crash, not drift (retrying next tick); '
                'cause_hint=%r',
                result.cause_hint,
            )
            return None

        return (main_sha, result)

    except Exception:
        logger.debug('run_main_tip_sweep: unexpected error', exc_info=True)
        return None
    finally:
        # Scoped cleanup: remove only this specific sweep worktree.
        # INTENTIONALLY NO 'git worktree prune' (DD5 guarantee).
        if worktree_added and tmp_path is not None:
            try:
                await _run(
                    ['git', 'worktree', 'remove', '--force', str(tmp_path)],
                    cwd=config.project_root,  # type: ignore[union-attr]
                )
            except Exception:
                logger.debug('run_main_tip_sweep: worktree remove failed', exc_info=True)
        if tmp_path is not None:
            with contextlib.suppress(Exception):
                shutil.rmtree(tmp_path, ignore_errors=True)
