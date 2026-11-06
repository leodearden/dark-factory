#!/usr/bin/env python3
"""Run the prepare-sitting mode headless on Fable before the 05:30 render, so the return brief carries prepared judgement.

Why this run exists: the task mandates the prepare-sitting mode "run nightly on
Fable", and a watcher session is not guaranteed to be armed at 05:30. Its
judgement reaches the page only through ``prepare_sitting.py record``, which
writes ``data/sitting/``. The render that follows re-measures every figure
itself, so a failed night costs thinner recommendations, never a wrong page.

Recommend-only regardless of flags. Three layers confine the run:

- Landlock is the write boundary. The CLI runs under
  ``orchestrator/src/orchestrator/agents/sandbox_dispatch.py::wrap_command``
  pinned to Landlock, so the whole tree is read-only except ``data/sitting/``
  (the inbox, the store and the run's ``CLAUDE_CONFIG_DIR``), ``~/.cache/uv``,
  ``/tmp`` and ``<root>/.task/``, which Landlock always grants in its cwd (the
  run's cwd is the root, since ``PREPARE_COMMAND`` is repo-relative). Without
  Landlock the run is refused, never launched unconfined; bwrap is not a
  fallback, since its mount order leaves the host ``/tmp`` read-only.
- The CLI rules are the first gate. It runs under ``--permission-mode dontAsk``
  with ``nightly_allowed_tools`` as the only tools it may use, reading no
  settings file (``setting_sources=[]``), and ``NIGHTLY_DENIED_TOOLS`` naming
  every apply verb. They are not a write boundary: the CLI matches rules
  against raw command text, so a shell-quoted ``--output`` passes the git
  denies (``shared/src/shared/git_read_rules.py``). Through its own tools the
  run Writes JSON batches to a fresh per-run inbox,
  ``data/sitting/inbox/<UTC stamp>/``, and runs ``prepare_sitting.py record
  --from <file>``.
- ``prepare_sitting.py`` confines itself in code. The child env carries
  ``NIGHTLY_CONFINEMENT_ENV``, set to the run's inbox, under which it refuses
  any other subcommand, ``record`` from stdin or from outside that inbox, a
  non-default ``--preparation``, ``--ledger`` and ``--apply-closes``.

The outcome is measured from structured data, not from the agent's say-so:
``recorded`` counts the item keys whose store entry equals a preparation the
run's inbox files offer and changed during the run. A run that records nothing exits 1.

Its MCP servers are exactly ``NIGHTLY_MCP_SERVERS``, copied from the
checkout's ``.mcp.json`` and passed with ``--strict-mcp-config``. The
allowlisted reads therefore never depend on the ambient project config being
approved for a headless run, and no other server starts.

The account comes from the shared pool (``account_pool.build_pool``): one
lease, read and handed straight back before the run, because nothing in the
child can settle a slot. The known cost is that this run's spend is invisible
to the gate. The leased token is written into the run's config dir and
removed on every exit that reaches ``run``'s ``finally`` (a SIGKILL leaves
it). With no lease the run is skipped (exit 1): it never uses the host login.
The child's env is ``shared.secret_env.agent_subprocess_env`` of this
process's, with the lease's ``CLAUDE_CODE_OAUTH_TOKEN`` on top, so it holds no
API key or other secret, no endpoint selection and no Python-environment
selection (OPERATIONS.md §12, "Legibility trickle accounts (03:00)").

Exit codes: 0 the run completed and recorded something; 1 it failed, timed out,
hit a usage limit or recorded nothing; 2 a configuration error. ``main`` never raises,
though the wrapper renders the page whatever this returns.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Bind scripts/, shared/ and orchestrator/ to THIS checkout, never an editable install's (tasks 2881/2882).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / 'orchestrator' / 'src', _REPO_ROOT / 'shared' / 'src', _REPO_ROOT / 'scripts'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
# legibility/coder.py imports its sibling as a bare `codebook`; appended last so no flat legibility module shadows anything.
if str(_REPO_ROOT / 'scripts' / 'legibility') not in sys.path:
    sys.path.append(str(_REPO_ROOT / 'scripts' / 'legibility'))

from legibility import account_pool  # noqa: E402
from orchestrator.agents.sandbox_dispatch import (  # noqa: E402  # debt: task 6760
    Backend,
    SandboxUnavailable,
    resolve_backend_or_refuse,
    set_backend,
    wrap_command,
)
from shared.cap_markers import looks_like_blocking_banner  # noqa: E402
from shared.cli_invoke import build_claude_argv, claude_binary_spec  # noqa: E402  # debt: task 6760
from shared.config_dir import TaskConfigDir  # noqa: E402  # debt: task 6760
from shared.git_read_rules import git_read_rules  # noqa: E402  # debt: task 6760
from shared.secret_env import agent_subprocess_env  # noqa: E402  # debt: task 6760
from sitting import inventory, preparation  # noqa: E402  # debt: task 6760

EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2

DEFAULT_MODEL = 'fable'
DEFAULT_BUDGET_USD = 10.0
DEFAULT_TIMEOUT_SECS = 2700.0
DEFAULT_MAX_TURNS = 200
PERMISSION_MODE = 'dontAsk'
STREAM_TAIL_CHARS = 2000

DEFAULT_MCP_JSON = _REPO_ROOT / '.mcp.json'
NIGHTLY_MCP_SERVERS: tuple[str, ...] = ('escalation', 'fused-memory')

PREPARE_COMMAND = 'uv run --frozen --project shared python scripts/sitting/prepare_sitting.py'
"""The one shell spelling the run may execute; repo-relative, so the child runs in the run's root."""

_GIT_READS = git_read_rules('show', 'log')

NIGHTLY_READ_TOOLS: tuple[str, ...] = (
    'Read',
    'Grep',
    'Glob',
    *_GIT_READS.allow,
    f'Bash({PREPARE_COMMAND} brief --json:*)',
    'mcp__escalation__get_escalation',
    'mcp__escalation__get_pending_escalations',
    'mcp__escalation__get_task_escalations',
    'mcp__escalation__get_task_escalation_history',
    'mcp__fused-memory__get_task',
    'mcp__fused-memory__get_statuses',
    'mcp__fused-memory__search',
    'mcp__fused-memory__search_tasks',
    'mcp__fused-memory__get_memories_by_metadata',
)

TIMESTAMP_FORMAT = '%Y%m%dT%H%M%SZ'

SITTING_DATA_PATH = preparation.DEFAULT_PREPARATION_PATH.parent


def nightly_allowed_tools(inbox: Path) -> tuple[str, ...]:
    """The reads, plus the one write the run has: files in *inbox* (``//`` marks an absolute path; the Write tool is gated by an ``Edit`` rule) and ``record --from``.

    ``record`` is allowed for any path because a Bash rule matches only at a word boundary; that it reads
    only the inbox is ``prepare_sitting.py::nightly_confinement_breach``'s to enforce.
    """
    return (*NIGHTLY_READ_TOOLS, f'Bash({PREPARE_COMMAND} record --from:*)', f'Edit(/{inbox}/**)')


NIGHTLY_DENIED_TOOLS: tuple[str, ...] = (
    'NotebookEdit',
    *_GIT_READS.deny,
    'mcp__escalation__resolve_issue',
    'mcp__escalation__stamp_triage',
    'mcp__escalation__promote_to_l2',
    'mcp__escalation__declare_pin',
    'mcp__escalation__amend_escalation',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__submit_task',
    'mcp__fused-memory__add_memory',
)

NIGHTLY_SYSTEM_PROMPT = (
    'You are the dark-factory sitting preparer, running unattended before dawn. '
    'You investigate the questions awaiting Leo and record prepared judgement for his return. '
    'You never apply a ruling, close an escalation, stamp a task or edit a file outside your inbox; '
    'every tool outside your allowlist is denied without a prompt.'
)


def nightly_prompt(inbox: Path, root: Path) -> str:
    return f"""\
Run the prepare-sitting mode in its nightly form, as `skills/escalation-watcher/SKILL.md` \
section "Sitting preparer (`prepare-sitting` mode)" describes it.

1. Run `{PREPARE_COMMAND} brief --json` to list every open item with its bucket, \
ownership probes and gate facts.
2. For each numbered item without a current preparation, carry out that section's \
investigation duties before recording anything.
3. Record in batches as you go, so a night cut short keeps what it finished. Per batch, \
Write the preparations as a JSON array to a NEW file in `{inbox}/` (`batch-01.json`, then \
`batch-02.json`; never reuse a name), then run \
`{PREPARE_COMMAND} record --from {inbox.relative_to(root)}/<file>`. \
A refused record writes nothing and says why: fix the file with Edit and record it again. \
`record --from -` is refused for this run.

This run is recommend-only: Leo applies rulings when he returns. Finish with one line, \
`recorded <n> of <m> numbered items`.
"""


@dataclass(frozen=True)
class Settings:
    claude_bin: str
    model: str
    budget_usd: float
    timeout_secs: float
    max_turns: int
    mcp_json: Path
    root: Path
    sandbox_backend: Backend = 'landlock'


class McpConfigUnusable(Exception):
    """The MCP config file cannot supply a block for every server in ``NIGHTLY_MCP_SERVERS``."""


@dataclass(frozen=True)
class Outcome:
    """What the run's inbox offered against what the store holds."""

    offered_keys: int
    undecodable: int
    recorded: int


@dataclass(frozen=True)
class Completed:
    """What the CLI left behind; ``returncode`` is None when the timeout killed it."""

    returncode: int | None
    stdout: str
    stderr: str


def run(settings: Settings, *, gate: Any = None) -> int:
    """Lease an account, run the CLI once under the OS sandbox with the prompt on stdin, and judge how it ended."""
    claude = shutil.which(settings.claude_bin)
    if claude is None:
        _log(f'claude binary not found or not executable: {settings.claude_bin!r}')
        return EXIT_CONFIG
    try:
        mcp_config = _mcp_config(settings.mcp_json)
    except McpConfigUnusable as exc:
        _log(f'no usable MCP config: {exc}')
        return EXIT_CONFIG
    set_backend(settings.sandbox_backend)
    try:
        resolve_backend_or_refuse()
    except SandboxUnavailable as exc:
        _log(f'refusing to run unconfined: {exc}')
        return EXIT_CONFIG
    root = settings.root.resolve()
    inbox = _new_inbox(root)
    sitting_data = root / SITTING_DATA_PATH
    config_dir = TaskConfigDir(f'nightly-prepare-{inbox.name}', base_dir=sitting_data / 'config')
    try:
        env = _child_env(gate if gate is not None else account_pool.build_pool(), inbox, config_dir)
        if env is None:
            config_dir.cleanup()
            return EXIT_FAILED
        return _run_confined(
            claude=claude, settings=settings, mcp_config=mcp_config, env=env, root=root, inbox=inbox,
            grants=(sitting_data, _ensure_uv_cache()),
        )
    finally:
        config_dir.remove_credentials()


def _ensure_uv_cache() -> Path:
    """``~/.cache/uv``, created if absent: Landlock can grant only a directory that exists."""
    uv_cache = Path.home() / '.cache' / 'uv'
    uv_cache.mkdir(parents=True, exist_ok=True)
    return uv_cache


def _run_confined(*, claude: str, settings: Settings, mcp_config: dict[str, Any], env: dict[str, str],
                  root: Path, inbox: Path, grants: Sequence[Path]) -> int:
    argv, temp_files = build_claude_argv(
        model=settings.model,
        max_budget_usd=settings.budget_usd,
        system_prompt=NIGHTLY_SYSTEM_PROMPT,
        max_turns=settings.max_turns,
        permission_mode=PERMISSION_MODE,
        allowed_tools=list(nightly_allowed_tools(inbox)),
        disallowed_tools=list(NIGHTLY_DENIED_TOOLS),
        mcp_config=mcp_config,
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=True,
        setting_sources=[],
    )
    argv[0] = claude
    confined = wrap_command(argv, root, [], writable_extras=[str(grant) for grant in grants])
    store_path = root / preparation.DEFAULT_PREPARATION_PATH
    held_before = _load_or_empty(store_path)
    try:
        completed = _run_bounded(confined, env, settings.timeout_secs, cwd=root, prompt=nightly_prompt(inbox, root))
    except OSError as exc:
        _log(f'claude could not be started ({claude!r}): {exc}')
        return EXIT_CONFIG
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)
    return _judge(completed, settings, _measure(inbox, store_path, held_before))


def main(argv: Sequence[str] | None = None, *, gate: Any = None, root: Path = _REPO_ROOT,
         sandbox_backend: Backend = 'landlock') -> int:
    args = _parser().parse_args(argv)
    settings = Settings(args.claude_bin, args.model, args.budget_usd, args.timeout_secs, args.max_turns, args.mcp_json,
                        root, sandbox_backend)
    try:
        return run(settings, gate=gate)
    except Exception as exc:
        _log(f'failed before the run completed: {exc!r}')
        return EXIT_FAILED


def _new_inbox(root: Path) -> Path:
    inbox = root / preparation.DEFAULT_INBOX_PATH / datetime.now(UTC).strftime(TIMESTAMP_FORMAT)
    inbox.mkdir(parents=True)
    return inbox


def _mcp_config(path: Path) -> dict[str, Any]:
    """*path*'s blocks for ``NIGHTLY_MCP_SERVERS`` and no other."""
    try:
        servers = json.loads(path.read_text(encoding='utf-8'))['mcpServers']
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise McpConfigUnusable(f'{path}: {type(exc).__name__}: {exc}') from exc
    if not isinstance(servers, dict):
        raise McpConfigUnusable(f'{path}: mcpServers is not an object')
    missing = [name for name in NIGHTLY_MCP_SERVERS if not isinstance(servers.get(name), dict)]
    if missing:
        raise McpConfigUnusable(f'{path}: no mcpServers block for {", ".join(missing)}')
    return {'mcpServers': {name: servers[name] for name in NIGHTLY_MCP_SERVERS}}


def _child_env(gate: Any, inbox: Path, config_dir: TaskConfigDir) -> dict[str, str] | None:
    """The CLI's env with a leased account written into *config_dir*, or None when no pool account is free."""
    env = agent_subprocess_env(os.environ)
    env[preparation.NIGHTLY_CONFINEMENT_ENV] = str(inbox)
    env['CLAUDE_CONFIG_DIR'] = str(config_dir.path)
    lease = gate.try_lease(reverse=True)
    if lease is None:
        _log('no pool account free; skipping: the nightly never uses the host login')
        return None
    try:
        env['CLAUDE_CODE_OAUTH_TOKEN'] = lease.token
        config_dir.write_credentials(lease.token)
        return env
    finally:
        gate.release_probe_slot(lease.token)


def _run_bounded(argv: list[str], env: dict[str, str], timeout_secs: float, *, cwd: Path, prompt: str) -> Completed:
    """Kill the whole process group on timeout: the CLI's own children would otherwise hold the pipes open."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=cwd, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return Completed(None, stdout, stderr)
    return Completed(proc.returncode, stdout, stderr)


def _load_or_empty(store_path: Path) -> preparation.PreparationStore:
    try:
        return preparation.load(store_path)
    except preparation.PreparationStoreCorrupt:
        return preparation.PreparationStore()


def _measure(inbox: Path, store_path: Path, held_before: preparation.PreparationStore) -> Outcome:
    """The item keys whose store entry is now one of the preparations *inbox*'s files offer and was not that entry before the run.

    Not a before/after diff of the whole store: a watcher may record into it concurrently, which
    can only lower this count, never raise it.
    """
    offered, undecodable = _offered(inbox)
    try:
        held_now = preparation.load(store_path)
    except preparation.PreparationStoreCorrupt as exc:
        _log(f'the preparation store is unreadable, counting 0 recorded: {exc}')
        return Outcome(len(offered), undecodable, 0)
    recorded = sum(
        1 for key, offers in offered.items()
        if held_now.entries.get(key) in offers and held_now.entries.get(key) != held_before.entries.get(key)
    )
    return Outcome(len(offered), undecodable, recorded)


def _offered(inbox: Path) -> tuple[dict[str, list[preparation.Preparation]], int]:
    """Each decodable preparation in the batch files of *inbox* by item key, and how many files or entries were not."""
    offered: dict[str, list[preparation.Preparation]] = {}
    undecodable = 0
    for path in sorted(inbox.iterdir()):
        if not preparation.is_inbox_batch(inbox, path):
            continue
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError, RecursionError):
            undecodable += 1
            continue
        for entry in data if isinstance(data, list) else [data]:
            try:
                prep = preparation.from_json_payload(entry)
            except (TypeError, ValueError):
                undecodable += 1
                continue
            offered.setdefault(inventory.key_str(prep.item_key), []).append(prep)
    return offered, undecodable


def _judge(completed: Completed, settings: Settings, outcome: Outcome) -> int:
    run_label = f'model={settings.model!r}, recorded={outcome.recorded}'
    if completed.returncode is None:
        _log(f'claude timed out after {settings.timeout_secs:g}s ({run_label}): {_tails(completed)}')
        return EXIT_FAILED
    if completed.returncode != 0:
        _log(f'claude exited {completed.returncode} ({run_label}){_banner_clause(completed)}: {_tails(completed)}')
        return EXIT_FAILED
    result = _parse_result(completed.stdout)
    if result is None:
        _log(f'claude exited 0 without a JSON result ({run_label}){_banner_clause(completed)}: {_tails(completed)}')
        return EXIT_FAILED
    if result.get('is_error'):
        _log(f'claude reported an error result (subtype={result.get("subtype")!r}, {run_label}): {_tails(completed)}')
        return EXIT_FAILED
    summary = (f'model={settings.model!r}, turns={result.get("num_turns")}, cost_usd={result.get("total_cost_usd")}, '
               f'recorded={outcome.recorded} of {outcome.offered_keys}, undecodable={outcome.undecodable}, '
               f'permission_denials={len(result.get("permission_denials") or [])}')
    verdict = f'{result.get("result", "")!s:.200}'
    if outcome.recorded == 0:
        _log(f'recorded nothing; {outcome.offered_keys} item key(s) offered ({summary}): {verdict}')
        return EXIT_FAILED
    print(f'nightly_prepare: done ({summary}): {verdict}')
    return EXIT_OK


def _parse_result(stdout: str) -> dict[str, Any] | None:
    """The CLI's ``--output-format json`` result object; a reply that parses is a verdict, never a banner."""
    try:
        result = json.loads(stdout)
    except ValueError:
        return None
    return result if isinstance(result, dict) else None


def _banner_clause(completed: Completed) -> str:
    marker = looks_like_blocking_banner(f'{completed.stdout}\n{completed.stderr}')
    return f', usage-limit banner {marker!r}' if marker else ''


def _tails(completed: Completed) -> str:
    return (f'stdout={completed.stdout[-STREAM_TAIL_CHARS:]!r} '
            f'stderr={completed.stderr[-STREAM_TAIL_CHARS:]!r}')


def _log(message: str) -> None:
    print(f'nightly_prepare: {message}', file=sys.stderr)


def _positive(cast: Callable[[str], float]) -> Callable[[str], Any]:
    def parse(text: str) -> float:
        value = cast(text)
        if value <= 0:
            raise argparse.ArgumentTypeError(f'must be positive, got {text}')
        return value
    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='nightly_prepare.py', description='Run the prepare-sitting mode headless.')
    parser.add_argument('--claude-bin', default=claude_binary_spec(), help='the claude CLI (default: $CLAUDE_BINARY, else claude on PATH)')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--budget-usd', type=_positive(float), default=DEFAULT_BUDGET_USD)
    parser.add_argument('--timeout-secs', type=_positive(float), default=DEFAULT_TIMEOUT_SECS)
    parser.add_argument('--max-turns', type=_positive(int), default=DEFAULT_MAX_TURNS)
    parser.add_argument('--mcp-json', type=Path, default=DEFAULT_MCP_JSON,
                        help="the MCP config whose escalation and fused-memory blocks the run uses (default: this checkout's)")
    return parser


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='nightly_prepare: %(name)s: %(message)s')
    sys.exit(main())
