"""Test/lint/typecheck runner for verification stages."""

import asyncio
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

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
    """Return True for concrete test files (test_*.py, *_test.py, anything under tests/).

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


def _is_conftest(path: str) -> bool:
    """Return True when *path* is a ``conftest.py`` file."""
    return path.rsplit('/', 1)[-1] == 'conftest.py'


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


def _extract_cause_hint(output: str) -> str:
    """Extract a one-line failure hint from command output.

    Uses a pattern ladder (first match wins):
    1. ``error: …``         — cargo/clippy surface errors
    2. ``… FAILED``         — Rust test runner failure lines
    3. ``Command timed out after Ns: …`` — our own timeout wrapper
    4. ``ERROR: …``         — flock/script wrapper errors
    5. ``… npm (ERR!|error) …`` — npm errors
    6. fallback: last non-blank line of output

    Returns ``''`` for None, empty, or whitespace-only input.
    Result is stripped to a single line and capped at 200 chars.
    """
    if not output or not output.strip():
        return ''

    _HINT_PATTERNS = [
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

    # Fallback: last non-blank line
    last = next(
        (line for line in reversed(output.splitlines()) if line.strip()),
        '',
    )
    return last.strip()[:200]


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
    #   failed to (parse|compile|read|find)
    #                   → 'error: failed to parse manifest at `/path/Cargo.toml`'
    #                     'error: failed to read `/path/Cargo.toml`'
    #   could not find  → 'error: could not find `Cargo.toml` in `/path` or any parent directory'
    #
    # Dropped tokens — no grounded cargo CLI sample available for either:
    #   `invalid `  — see test_rustc_invalid_diagnostic_not_cargo_cli_error.
    #   `package `  — re-add with a tighter suffix (e.g. 'package \`') once a
    #                 real cargo log line is observed.
    (re.compile(
        r'^error: (--|no such subcommand|failed to (parse|compile|read|find)|could not find)',
        re.MULTILINE,
    ), 'cargo_cli_error'),
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
    6. ``… FAILED``                              → ``'test_failure'``
    7. ``npm (ERR!|error)``                      → ``'npm_error'``
    8. ``flock:``                                → ``'flock_error'``
    9. ``tree-sitter generate``                  → ``'tree_sitter_generate_error'``
    10. fallback (rc != 0)                       → ``'unknown_test_failure'``

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

    return 'unknown_test_failure'


# Categories that must NOT be auto-archived even though they end with '_error'.
# compile_error is handled by the debugger (type annotations, missing imports);
# the human triage criterion is "a human, not a debugger, has to look".
_ARCHIVE_DENY_LIST = frozenset({'compile_error', 'test_failure', 'infra_timeout', 'passed', ''})

# Ordered from highest to lowest severity; used by ``_worst_category``.
# Categories absent from this list (e.g. custom ones) sort lower than all listed.
_CATEGORY_PRIORITY: list[str] = [
    'infra_timeout',
    'cargo_cli_error',
    'compile_error',
    'tree_sitter_generate_error',
    'flock_error',
    'npm_error',
    'test_failure',
    'unknown_test_failure',
    'passed',
    '',
]


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

    # Sanitize the module prefix for use in filenames (mirrors _warm_marker_name).
    if module_prefix is not None:
        safe_prefix = module_prefix.replace('/', '_').replace(' ', '_')
        infix = f'.{safe_prefix}'
    else:
        infix = ''

    written: list[Path] = []

    # Write per-command log files.
    for run in runs:
        if run.get('cmd') is None:
            continue
        log_path = verify_dir / f'attempt-{attempt_id}{infix}.{run["label"]}.log'
        try:
            log_path.write_text(run['output'], encoding='utf-8')
            written.append(log_path)
        except OSError as exc:
            logger.warning('_persist_attempt_logs: could not write %s: %s', log_path, exc)

    # Build summary.json.
    # Top-level rc/cmd/timed_out/started_at/duration_secs fields come from the
    # run with the highest numeric rc (timed_out used as a tiebreaker).  This
    # intentionally differs from the 'category' field, which uses _worst_category
    # priority semantics.  The rationale: rc is the most unambiguous exit-code
    # signal for the outermost process, while category conveys semantic severity
    # across tools that may use different rc scales.  Downstream readers should
    # treat top-level metadata as "the loudest raw exit code" and 'category' as
    # "the highest-severity classification".
    active_runs = [r for r in runs if r.get('cmd') is not None]
    if active_runs:
        worst = max(active_runs, key=lambda r: (r['rc'], r['timed_out']))
    else:
        worst = {'rc': 0, 'timed_out': False, 'cmd': None,
                 'started_at': '', 'duration_secs': 0.0}

    summary_payload: dict = {
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


_DEFAULT_ARCHIVE_MAX_AGE_DAYS = 30
_DEFAULT_ARCHIVE_MAX_BYTES = 500 * 1024 * 1024
# Process-local throttle: at most one rglob walk per process per 30 min.
# Cross-process redundancy is accepted as a cost-only trade-off; see task 1102.
# Thread-safety: _maybe_prune_archive has no awaits, so concurrent async coroutines
# cannot interleave the check + update. This is NOT safe for multi-threaded callers
# (e.g. asyncio.to_thread) without a threading.Lock — add one if that ever changes.
_PRUNE_THROTTLE_SECS: float = 1800  # 30 minutes
_LAST_PRUNE_AT: float | None = None


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

    All errors are logged and swallowed; never raises.
    """
    if not archive_root.exists():
        return

    # module-level `import time` is sufficient; no local import needed.
    now = time.time()
    cutoff = now - max_age_days * 86_400

    # Single rglob walk — collect all log files once, avoiding a second
    # directory scan for the size-cap pass.
    all_entries: list[tuple[Path, float, int]] = []
    for path in archive_root.rglob('*.log'):
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
    """
    global _LAST_PRUNE_AT
    if archive_root is None:
        return False
    now = time.monotonic()
    if _LAST_PRUNE_AT is not None and now - _LAST_PRUNE_AT < _PRUNE_THROTTLE_SECS:
        logger.debug(
            'skipping prune: %.0fs since last (throttle %ds)',
            now - _LAST_PRUNE_AT,
            _PRUNE_THROTTLE_SECS,
        )
        return False
    _prune_archive(archive_root)
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


def scope_module_config(mc: ModuleConfig, task_files: list[str]) -> ModuleConfig | None:
    """Narrow *mc*'s commands to the specific *task_files* it covers.

    Filters *task_files* to ``.py`` files matching ``mc.prefix + '/'`` and
    keeps full worktree-relative paths.  The ``--directory`` flag is stripped
    from scoped commands so that tools resolve paths from the worktree root,
    where the full paths are valid.

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
    # conftest.py is already excluded by _is_test_file at any depth;
    # conftest files are handled by the has_conftest branch above.
    test_files = [f for f in scoped if _is_test_file(f)]

    # Build scoped commands with worktree-relative paths, then strip
    # --directory so tools resolve paths from the worktree root
    lint_cmd = _strip_directory_flag(
        _scope_command(mc.lint_command, 'ruff check', scoped), mc.prefix)
    type_cmd = _strip_directory_flag(
        _scope_command(mc.type_check_command, 'pyright', scoped), mc.prefix)
    if has_conftest:
        # Full unscoped suite: conftest changes affect everything it shadows.
        test_cmd = mc.test_command
    elif test_files:
        test_cmd = _strip_directory_flag(
            _scope_command(mc.test_command, 'pytest', test_files), mc.prefix)
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


def _build_fallback_config(task_files: list[str]) -> ModuleConfig | None:
    """Build a synthetic ModuleConfig from *task_files* when no module configs match.

    Filters to ``.py`` files, classifies into source vs test, and builds bare
    ``ruff check``/``pyright``/``pytest`` commands (no ``uv run`` wrapper —
    callers run these in the worktree root where venvs aren't needed for the
    fallback path).

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
    # conftest.py is already excluded by _is_test_file at any depth.
    test_files = [f for f in py_files if _is_test_file(f)]

    lint_cmd = 'ruff check ' + ' '.join(py_files)
    type_cmd = 'pyright ' + ' '.join(py_files)
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
        # `test_files` always contains file paths (e.g. 'a/sub/test_x.py'),
        # never bare directory paths — _is_test_file gates on filename
        # suffixes/prefixes and a /tests/ substring, none of which match a
        # directory entry.  That guarantees `t.startswith(d + '/')` reliably
        # means "t is inside directory d" without false positives from a
        # sibling directory like 'ab/' matching the prefix 'a'.
        if '.' not in conftest_dirs:
            outside = [
                t for t in test_files
                if not any(t.startswith(d + '/') for d in conftest_dirs)
            ]
        else:
            outside = []
        targets = conftest_dirs + outside
        test_cmd = 'pytest ' + ' '.join(targets)
    elif test_files:
        test_cmd = 'pytest ' + ' '.join(test_files)
    else:
        test_cmd = None

    return ModuleConfig(
        prefix='__fallback__',
        lint_command=lint_cmd,
        type_check_command=type_cmd,
        test_command=test_cmd,
    )


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


async def _run_cmd(
    cmd: str,
    cwd: Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> tuple[int, str, bool]:
    """Run a shell command, return (returncode, combined output, timed_out).

    When *env* is non-None, it is merged on top of ``os.environ`` and passed
    to the subprocess so callers can inject build accelerators like
    ``RUSTC_WRAPPER=sccache`` without mutating the parent process's env.
    """
    proc = None
    pgid: int | None = None
    subprocess_env: dict[str, str] | None = None
    if env:
        subprocess_env = {**os.environ, **env}
    try:
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
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        rc = proc.returncode if proc.returncode is not None else 1
        return rc, stdout.decode(), False
    except TimeoutError:
        if proc is not None and pgid is not None:
            await terminate_process_group(proc, pgid, grace_secs=5.0)
        return 1, f'Command timed out after {timeout}s: {cmd}', True
    except asyncio.CancelledError:
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
) -> float:
    """Return the effective per-command verify timeout.

    When *is_cold* is False (warm cache), the warm timeout is returned:
    ``module_config.verify_command_timeout_secs`` takes precedence over
    ``config.verify_command_timeout_secs``.

    When *is_cold* is True (first verify in a fresh worktree), the cold
    timeout is resolved via the cascade:
      1. ``module_config.verify_cold_command_timeout_secs`` (if set)
      2. ``config.verify_cold_command_timeout_secs`` (if set)
      3. The warm timeout computed above (fallback when cold knob is unset
         at every level — preserves existing behaviour for deployments that
         don't configure the cold window).
    """
    # Warm track: module override wins over top-level.
    warm: float
    if module_config is not None and module_config.verify_command_timeout_secs is not None:
        warm = module_config.verify_command_timeout_secs
    else:
        warm = config.verify_command_timeout_secs

    if not is_cold:
        return warm

    # Cold track: cascade module → top → warm fallback.
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
) -> dict[str, str]:
    """Return the effective env injected into verify commands.

    Merges ``config.verify_env`` with ``module_config.verify_env``; module
    keys override top-level keys.
    """
    merged: dict[str, str] = {}
    merged.update(config.verify_env or {})
    if module_config is not None and module_config.verify_env:
        merged.update(module_config.verify_env)
    return merged


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
    timeout = _resolve_verify_timeout(config, module_config, is_cold=is_cold)
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
    verify_env = _resolve_verify_env(config, module_config)

    if verify_env:
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

    async def _run_or_skip_timed(
        cmd: str | None,
    ) -> tuple[int, str, bool, str | None, float]:
        """Like _run_cmd but returns (rc, out, timed_out, started_at_iso, duration_secs).

        When *cmd* is None (skipped check), returns (0, '', False, None, 0.0).
        """
        if cmd is None:
            return 0, '', False, None, 0.0
        started_at = datetime.now(UTC).isoformat()
        t0 = time.monotonic()
        rc, out, timed_out_flag = await _run_cmd(cmd, worktree, timeout, env=verify_env or None)
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
        if concurrent:
            (
                (test_rc, test_out, test_timed_out, test_started_at, test_duration),
                (lint_rc, lint_out, lint_timed_out, lint_started_at, lint_duration),
                (type_rc, type_out, type_timed_out, type_started_at, type_duration),
            ) = await asyncio.gather(
                _run_or_skip_timed(test_cmd),
                _run_or_skip_timed(lint_cmd),
                _run_or_skip_timed(type_cmd),
            )
        else:
            test_rc, test_out, test_timed_out, test_started_at, test_duration = await _run_or_skip_timed(test_cmd)
            lint_rc, lint_out, lint_timed_out, lint_started_at, lint_duration = await _run_or_skip_timed(lint_cmd)
            type_rc, type_out, type_timed_out, type_started_at, type_duration = await _run_or_skip_timed(type_cmd)

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

    # Persist attempt logs when requested (opt-in via attempt_id kwarg).
    # No-op when attempt_id is None (existing callers: merge_queue, review_checkpoint).
    worktree_log_paths: list[str] = []
    archive_log_paths: list[str] = []
    if attempt_id is not None and task_id is not None:
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
    )


async def run_full_verification(
    project_root: Path,
    config: OrchestratorConfig,
) -> VerifyResult:
    """Run verification for ALL subprojects against the project root.

    Unlike run_scoped_verification, this runs full (unscoped) test suites
    for every subproject that has an orchestrator.yaml. Used by review
    checkpoints to check integration health across the whole codebase.
    """
    from orchestrator.config import _discover_module_configs

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

    # When the plan didn't provide a file list, try to derive one from git.
    if task_files is None:
        task_files = await _derive_task_files_from_git(worktree, config)

    # _prune_archive runs exactly once in the finally block regardless of which
    # branch returns, preventing concurrent per-module prune races and removing
    # the repetitive guard at every return site.
    try:
        if module_configs:
            # Apply file-level scoping within each subproject when task_files given
            if task_files:
                # Filter to files that still exist — tasks may delete files as part of their work
                existing_files = [f for f in task_files if (worktree / f).exists()]
                # scope_module_config returns None when no files touch the subproject;
                # those subprojects are skipped rather than running their full suite.
                per_module = [
                    (mc.prefix, scope_module_config(mc, existing_files))
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
                    # No subproject has matching files — fall through to global.
                    # Otherwise we'd gather() zero coroutines and silently pass.
                    logger.info(
                        'Verification mode: global (file-scoped filtered every subproject)',
                    )
                    return await run_verification(
                        worktree, config, max_retries=max_retries,
                        is_merge_verify=is_merge_verify,
                        attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                    )
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
                *(
                    run_verification(
                        worktree, config, mc,
                        max_retries=max_retries,
                        is_merge_verify=is_merge_verify,
                        attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
                    )
                    for mc in scoped
                )
            )
            return _aggregate_results(list(results))

        # No module_configs — try fallback or global
        if task_files:
            # Filter to files that still exist — tasks may delete files as part of their work
            existing_files = [f for f in task_files if (worktree / f).exists()]
            fallback = _build_fallback_config(existing_files)
            if fallback is not None:
                fallback = _apply_cargo_scope(
                    fallback, existing_files, worktree, scope_cargo_enabled,
                )
                logger.info('Verification mode: fallback-scoped (%d files)', len(existing_files))
                return await run_verification(
                    worktree, config, fallback, max_retries=max_retries,
                    is_merge_verify=is_merge_verify,
                    attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
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
                    )

        logger.info('Verification mode: global (no scope info)')
        return await run_verification(
            worktree, config, max_retries=max_retries,
            is_merge_verify=is_merge_verify,
            attempt_id=attempt_id, task_id=task_id, archive_root=archive_root,
        )
    finally:
        _maybe_prune_archive(archive_root)
