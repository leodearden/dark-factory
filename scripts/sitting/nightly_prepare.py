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

The account comes from the fleet pool (``account_pool.build_pool``), and the
run goes through the shared session runner,
``shared.cli_invoke.invoke_with_cap_retry``, in roster order like every other
session:

- A seat at its usage limit, or one whose credentials are rejected (401/403),
  hands the run to the next seat; a capped session that had made progress
  resumes there.
- It never parks: ``max_cap_retries`` is one pass over the pool, and a pool
  with no admissible seat ends the run at once.
- ``--timeout-secs`` bounds the whole run, failovers and the cooldowns
  between them included.
- The leased token lives in the run's config dir and is removed on every exit
  that reaches ``run``'s ``finally`` (a SIGKILL leaves it).
- An empty pool skips the night (exit 1): the host login is never a fallback.

The runner builds the child's env as ``shared.secret_env.agent_subprocess_env``
of this process's, with the seat's ``CLAUDE_CODE_OAUTH_TOKEN`` on top, so it
holds no API key or other secret, no endpoint selection and no
Python-environment selection (OPERATIONS.md §12, "Legibility trickle accounts
(03:00)").

Exit codes: 0 the run completed and recorded something; 1 it failed, timed out,
found no seat that could take it or recorded nothing; 2 a configuration error.
``main`` never raises, though the wrapper renders the page whatever this returns.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
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
from shared.cli_invoke import (  # noqa: E402  # debt: task 6760
    AgentResult,
    AllAccountsCappedException,
    claude_binary_spec,
    invoke_with_cap_retry,
)
from shared.config_dir import TaskConfigDir  # noqa: E402  # debt: task 6760
from shared.git_read_rules import git_read_rules  # noqa: E402  # debt: task 6760
from shared.usage_gate import PoolFrozen, UsageGate  # noqa: E402  # debt: task 6760
from sitting import inventory, preparation  # noqa: E402  # debt: task 6760

EXIT_OK, EXIT_FAILED, EXIT_CONFIG = 0, 1, 2

DEFAULT_MODEL = 'fable'
DEFAULT_BUDGET_USD = 10.0
DEFAULT_TIMEOUT_SECS = 2700.0
DEFAULT_MAX_TURNS = 200
PERMISSION_MODE = 'dontAsk'
TAIL_CHARS = 2000

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
class TimedOut:
    """The run's own wall clock ran out; it covers every seat the run failed over to."""

    after_secs: float


@dataclass(frozen=True)
class PoolExhausted:
    """No seat could take the run: every one capped or rejecting its credentials, as the gate held them at the end."""

    capped: tuple[str, ...]
    auth_failed: tuple[str, ...]
    earliest_reset: datetime | None


Ending = AgentResult | TimedOut | PoolExhausted
"""How a run ended: the CLI's own result, or the bound or the pool ending it first."""


def run(settings: Settings, *, gate: Any = None) -> int:
    """Run the CLI on the fleet pool under the OS sandbox with the prompt on stdin, and judge how it ended."""
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
    if gate is None:
        gate = account_pool.build_pool()
    if not gate.account_count:
        _log('no pool account resolved; skipping: the nightly never uses the host login')
        return EXIT_FAILED
    root = settings.root.resolve()
    inbox = _new_inbox(root)
    sitting_data = root / SITTING_DATA_PATH
    config_dir = TaskConfigDir(f'nightly-prepare-{inbox.name}', base_dir=sitting_data / 'config')
    try:
        return _run_confined(
            claude=claude, settings=settings, mcp_config=mcp_config, gate=gate, config_dir=config_dir,
            root=root, inbox=inbox, grants=(sitting_data, _ensure_uv_cache()),
        )
    finally:
        config_dir.remove_credentials()


def _ensure_uv_cache() -> Path:
    """``~/.cache/uv``, created if absent: Landlock can grant only a directory that exists."""
    uv_cache = Path.home() / '.cache' / 'uv'
    uv_cache.mkdir(parents=True, exist_ok=True)
    return uv_cache


def _run_confined(*, claude: str, settings: Settings, mcp_config: dict[str, Any], gate: UsageGate,
                  config_dir: TaskConfigDir, root: Path, inbox: Path, grants: Sequence[Path]) -> int:
    store_path = root / preparation.DEFAULT_PREPARATION_PATH
    held_before = _load_or_empty(store_path)
    try:
        ending = asyncio.run(_invoke_on_pool(
            gate,
            timeout_secs=settings.timeout_secs,
            config_dir=config_dir,
            prompt=nightly_prompt(inbox, root),
            system_prompt=NIGHTLY_SYSTEM_PROMPT,
            cwd=root,
            model=settings.model,
            max_turns=settings.max_turns,
            max_budget_usd=settings.budget_usd,
            permission_mode=PERMISSION_MODE,
            allowed_tools=list(nightly_allowed_tools(inbox)),
            disallowed_tools=list(NIGHTLY_DENIED_TOOLS),
            mcp_config=mcp_config,
            strict_mcp_config=True,
            setting_sources=[],
            env_overrides={preparation.NIGHTLY_CONFINEMENT_ENV: str(inbox)},
            sandbox_wrap=_spawn_command(claude, root, grants),
        ))
    except OSError as exc:
        _log(f'claude could not be started ({claude!r}): {exc}')
        return EXIT_CONFIG
    return _judge(ending, settings, _measure(inbox, store_path, held_before))


async def _invoke_on_pool(gate: UsageGate, *, timeout_secs: float, **invoke_kwargs: Any) -> Ending:
    """One run through the shared runner, failing over across *gate*'s seats, under ONE wall clock.

    The clock covers every attempt and every cap cooldown between them; its expiry kills the CLI's
    process group. ``max_cap_retries`` is one pass over the pool, and a frozen pool raises rather
    than parks. Every gate read happens here, and the gate is shut down on this loop, before it closes.
    """
    deadline = asyncio.timeout(timeout_secs)
    try:
        async with deadline:
            return await invoke_with_cap_retry(
                gate, 'nightly_prepare', max_cap_retries=gate.account_count, park_on_frozen_pool=False,
                **invoke_kwargs,
            )
    except TimeoutError:
        if not deadline.expired():
            raise
        return TimedOut(timeout_secs)
    except (AllAccountsCappedException, PoolFrozen):
        return PoolExhausted(gate.capped_account_names, gate.auth_failed_account_names, gate.soonest_resets_at)
    finally:
        await gate.shutdown()


def _spawn_command(claude: str, root: Path, grants: Sequence[Path]) -> Callable[[list[str]], list[str]]:
    """The command actually spawned for a built argv: *claude* as the binary, confined under the pinned backend."""
    writable = [str(grant) for grant in grants]
    return lambda argv: wrap_command([claude, *argv[1:]], root, [], writable_extras=writable)


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


def _judge(ending: Ending, settings: Settings, outcome: Outcome) -> int:
    account = f', account={ending.account_name!r}' if isinstance(ending, AgentResult) else ''
    run_label = f'model={settings.model!r}{account}, recorded={outcome.recorded}'
    if isinstance(ending, TimedOut):
        _log(f'claude timed out after {ending.after_secs:g}s, failovers included ({run_label})')
        return EXIT_FAILED
    if isinstance(ending, PoolExhausted):
        reset = ending.earliest_reset.isoformat() if ending.earliest_reset else 'unknown'
        _log(f'no pool account could take the run: capped [{", ".join(ending.capped)}], '
             f'auth-failed [{", ".join(ending.auth_failed)}]; earliest reset {reset} ({run_label})')
        return EXIT_FAILED
    if not ending.success:
        _log(f'claude failed (subtype={ending.subtype!r}, api_error_status={ending.api_error_status!r}, '
             f'{run_label}): {_tails(ending)}')
        return EXIT_FAILED
    summary = (f'model={settings.model!r}{account}, turns={ending.turns}, cost_usd={ending.cost_usd}, '
               f'recorded={outcome.recorded} of {outcome.offered_keys}, undecodable={outcome.undecodable}')
    verdict = f'{ending.output:.200}'
    if outcome.recorded == 0:
        _log(f'recorded nothing; {outcome.offered_keys} item key(s) offered ({summary}): {verdict}')
        return EXIT_FAILED
    print(f'nightly_prepare: done ({summary}): {verdict}')
    return EXIT_OK


def _tails(result: AgentResult) -> str:
    return (f'result={result.output[-TAIL_CHARS:]!r} '
            f'stderr={result.stderr[-TAIL_CHARS:]!r}')


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
