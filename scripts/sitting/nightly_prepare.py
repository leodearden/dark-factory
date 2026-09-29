#!/usr/bin/env python3
"""Run the prepare-sitting mode headless on Fable before the 05:30 render, so the return brief carries prepared judgement.

Why this run exists: the task mandates the prepare-sitting mode "run nightly on
Fable", and a watcher session is not guaranteed to be armed at 05:30. Its
judgement reaches the page only through ``prepare_sitting.py record``, which
writes ``data/sitting/``. The render that follows re-measures every figure
itself, so a failed night costs thinner recommendations, never a wrong page.

Recommend-only regardless of flags. The CLI runs under ``--permission-mode
dontAsk`` with ``NIGHTLY_ALLOWED_TOOLS`` as the only tools it may use (the
prepare script only as ``brief --json`` and ``record --from -``) and
``NIGHTLY_DENIED_TOOLS`` naming every apply verb. The child env also carries
``NIGHTLY_CONFINEMENT_ENV``, under which ``prepare_sitting.py`` itself refuses
any other subcommand, a non-default ``--preparation``, ``--ledger`` and
``--apply-closes``. A write outside ``data/sitting/`` is therefore refused in
code rather than merely instructed against.

Its MCP servers are exactly ``NIGHTLY_MCP_SERVERS``, copied from the
checkout's ``.mcp.json`` and passed with ``--strict-mcp-config``. The
allowlisted reads therefore never depend on the ambient project config being
approved for a headless run, and no other server starts.

The account comes from the shared pool through ``account_pool.subprocess_env``
and inherits its known limit: the lease is handed straight back, so this run's
spend is invisible to the gate. With no lease the child inherits this process's
environment. ``ANTHROPIC_API_KEY`` is stripped on both paths (OPERATIONS.md §12,
"Legibility trickle accounts (03:00)").

Exit codes: 0 the run completed; 1 it failed, timed out or hit a usage limit,
with both stream tails logged; 2 a configuration error. ``main`` never raises,
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
from shared.cap_markers import looks_like_blocking_banner  # noqa: E402
from shared.cli_invoke import build_claude_argv  # noqa: E402
from sitting.preparation import NIGHTLY_CONFINEMENT_ENV  # noqa: E402

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
"""The one shell spelling the run may execute; repo-relative, so the child runs in this checkout."""

NIGHTLY_ALLOWED_TOOLS: tuple[str, ...] = (
    'Read',
    'Grep',
    'Glob',
    'Bash(git show:*)',
    'Bash(git log:*)',
    f'Bash({PREPARE_COMMAND} brief --json:*)',
    f'Bash({PREPARE_COMMAND} record --from -:*)',
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

NIGHTLY_DENIED_TOOLS: tuple[str, ...] = (
    'Edit',
    'Write',
    'NotebookEdit',
    'mcp__escalation__resolve_issue',
    'mcp__escalation__stamp_triage',
    'mcp__escalation__promote_to_l2',
    'mcp__escalation__declare_pin',
    'mcp__fused-memory__update_task',
    'mcp__fused-memory__add_dependency',
    'mcp__fused-memory__submit_task',
    'mcp__fused-memory__add_memory',
)

NIGHTLY_SYSTEM_PROMPT = (
    'You are the dark-factory sitting preparer, running unattended before dawn. '
    'You investigate the questions awaiting Leo and record prepared judgement for his return. '
    'You never apply a ruling, close an escalation, stamp a task or edit a file; '
    'every tool outside your allowlist is denied without a prompt.'
)

NIGHTLY_PROMPT = f"""\
Run the prepare-sitting mode in its nightly form, as `skills/escalation-watcher/SKILL.md` \
section "Sitting preparer (`prepare-sitting` mode)" describes it.

1. Run `{PREPARE_COMMAND} brief --json` to list every open item with its bucket, \
ownership probes and gate facts.
2. For each numbered item without a current preparation, carry out that section's \
investigation duties before recording anything.
3. Record with `{PREPARE_COMMAND} record --from -`, passing the preparations JSON on stdin.

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


class McpConfigUnusable(Exception):
    """The MCP config file cannot supply a block for every server in ``NIGHTLY_MCP_SERVERS``."""


@dataclass(frozen=True)
class Completed:
    """What the CLI left behind; ``returncode`` is None when the timeout killed it."""

    returncode: int | None
    stdout: str
    stderr: str


def run(settings: Settings, *, gate: Any = None) -> int:
    """Lease an account, run the CLI once with the prompt on stdin, and judge how it ended."""
    claude = shutil.which(settings.claude_bin)
    if claude is None:
        _log(f'claude binary not found or not executable: {settings.claude_bin!r}')
        return EXIT_CONFIG
    try:
        mcp_config = _mcp_config(settings.mcp_json)
    except McpConfigUnusable as exc:
        _log(f'no usable MCP config: {exc}')
        return EXIT_CONFIG
    env = _child_env(gate if gate is not None else account_pool.build_pool())
    argv, temp_files = build_claude_argv(
        model=settings.model,
        max_budget_usd=settings.budget_usd,
        system_prompt=NIGHTLY_SYSTEM_PROMPT,
        max_turns=settings.max_turns,
        permission_mode=PERMISSION_MODE,
        allowed_tools=list(NIGHTLY_ALLOWED_TOOLS),
        disallowed_tools=list(NIGHTLY_DENIED_TOOLS),
        mcp_config=mcp_config,
        output_schema=None,
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=True,
    )
    argv[0] = claude
    try:
        completed = _run_bounded(argv, env, settings.timeout_secs)
    except OSError as exc:
        _log(f'claude could not be started ({claude!r}): {exc}')
        return EXIT_CONFIG
    finally:
        for path in temp_files:
            Path(path).unlink(missing_ok=True)
    return _judge(completed, settings)


def main(argv: Sequence[str] | None = None, *, gate: Any = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings(args.claude_bin, args.model, args.budget_usd, args.timeout_secs, args.max_turns, args.mcp_json)
    try:
        return run(settings, gate=gate)
    except Exception as exc:
        _log(f'failed before the run completed: {exc!r}')
        return EXIT_FAILED


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


def _child_env(gate: Any) -> dict[str, str]:
    env = account_pool.subprocess_env(gate)
    inherited = dict(os.environ) if env is None else env
    inherited.pop('ANTHROPIC_API_KEY', None)
    inherited[NIGHTLY_CONFINEMENT_ENV] = '1'
    return inherited


def _run_bounded(argv: list[str], env: dict[str, str], timeout_secs: float) -> Completed:
    """Kill the whole process group on timeout: the CLI's own children would otherwise hold the pipes open."""
    proc = subprocess.Popen(
        argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env, cwd=_REPO_ROOT, start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(NIGHTLY_PROMPT, timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        stdout, stderr = proc.communicate()
        return Completed(None, stdout, stderr)
    return Completed(proc.returncode, stdout, stderr)


def _judge(completed: Completed, settings: Settings) -> int:
    run_label = f'model={settings.model!r}'
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
    print(f'nightly_prepare: done ({run_label}, turns={result.get("num_turns")}, '
          f'cost_usd={result.get("total_cost_usd")}): {result.get("result", "")!s:.200}')
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
    parser.add_argument('--claude-bin', default='claude', help='the claude CLI (default: claude on PATH)')
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
