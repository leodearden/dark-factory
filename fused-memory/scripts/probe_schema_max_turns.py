#!/usr/bin/env python3
"""MANUAL diagnostic: measure the ``--max-turns`` x ``--json-schema`` interaction.

Task 3241.  Five separate comments in this repo asserted, in mutually
contradictory ways, what ``max_turns`` does to a ``--json-schema`` invocation —
and none of them could be cheaply re-checked, so the wrong ones survived for
months.  ``agent_loop.py`` passed ``max_turns=1`` on the belief that "schema
tool-use -> JSON response happens within the same turn"; measured, that value
failed 100% of the time and the failure was invisible because ``verify.py``
degrades to ``default_verdict='inconclusive'``.

The behaviour is CLI-VERSION-DEPENDENT and WILL drift again.  This script exists
so the next person does not have to rebuild a probe from scratch to re-check a
patch bump — which is exactly what the 3241 revalidation had to do.

NOT part of the pytest suite, deliberately.  It requires live Claude CLI
credentials and spends real tokens; the suite stays hermetic and offline.  Run
it by hand when a comment citing these numbers needs revalidating.

Baseline to diff against
------------------------
Recon-verify shape (EXPLORE_AGENT_SYSTEM_PROMPT + CLAUDE_CLI_RESPONSE_SCHEMA),
6 repeats per cell:

    Claude CLI 2.1.236:  mt=1 -> 0/6   mt=3 -> 4/6   mt=6 -> 4/6   mt=10 -> 6/6
    Claude CLI 2.1.233:  mt=1 -> 0/6   mt=3 -> 2/6   mt=6 -> 4/5   mt=10 -> 4/5

Every failure was ``subtype='error_max_turns'``, ``is_error=True``, carrying NO
structured payload — so ``schema_salvaged`` was False in all 12 observed
failures and salvage never engaged.  ``mt=1 -> 0/6`` is stable across both CLI
versions; the intermediate rates are not.  The failure is STOCHASTIC, which is
why every cell is repeated: a single run per cell is what produced an earlier,
wrong "max_turns=1 is safe" verdict.

Usage
-----
    # Default matrix (1, 3, 10) x 6 repeats, recon-verify shape.
    python scripts/probe_schema_max_turns.py

    # Cheap smoke run.
    python scripts/probe_schema_max_turns.py --max-turns 1 10 --repeat 2

    # Both shapes, including the judge's.
    python scripts/probe_schema_max_turns.py --shape both

Fidelity notes (all load-bearing — a naive probe gets a wrong answer)
--------------------------------------------------------------------
* argv is built by CALLING ``shared.cli_invoke.build_claude_argv``, never a
  hand-rolled flag list, so the probe measures the invocation production sends.
* The system prompt is the REAL one, captured by driving
  ``CodebaseVerifier.verify()`` up to its ``AgentLoop`` construction.  A toy
  system prompt SUCCEEDS at ``max_turns=1`` and manufactures the opposite
  verdict — this is the single biggest way to get this measurement wrong.
* The env is production's: ``ANTHROPIC_API_KEY`` stripped, per-invocation
  ``CLAUDE_CODE_OAUTH_TOKEN`` injected, per-invocation ``CLAUDE_CONFIG_DIR``.
  Omit it and the CLI returns ``is_error=True, subtype='success', num_turns=1,
  result='Not logged in - Please run /login'``, which is NOT a turn-cap failure
  but looks like a uniform 0/N and is easily misread as one.  Auth and credit
  failures are detected and reported SEPARATELY from ``error_max_turns``.
* No token value is ever printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FM_SRC = _REPO_ROOT / 'fused-memory' / 'src'
_SHARED_SRC = _REPO_ROOT / 'shared' / 'src'
for _p in (_FM_SRC, _SHARED_SRC):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.cli_invoke import (  # noqa: E402
    _REAL_BUILTIN_TOOLS_DENYLIST,
    build_claude_argv,
    no_mcp_servers_config,
)

_DEFAULT_ACCOUNTS_FILE = _REPO_ROOT / 'config' / 'usage-accounts.yaml'

# Substrings that mean "this run never reached the model", NOT "the turn cap bit".
# Conflating the two is the misread this script exists to prevent, so they are
# classified into their own outcome and excluded from the success rate.
_AUTH_FAILURE_MARKERS = (
    'not logged in',
    'please run /login',
    'oauth access token has been revoked',
    'oauth token has expired',
    'invalid api key',
    'authentication_error',
)
_CREDIT_FAILURE_MARKERS = (
    'hit your weekly limit',
    'usage limit reached',
    'out of credit',
    'credit balance is too low',
)

_OUTCOME_OK = 'ok'
_OUTCOME_MAX_TURNS = 'error_max_turns'
_OUTCOME_AUTH = 'auth_failure'
_OUTCOME_CREDIT = 'credit_exhausted'
_OUTCOME_OTHER = 'other_failure'
# Not a CLI error at all: a success whose payload carries an EMPTY tool_calls
# array.  agent_loop's run() reads that as the `no_tool_calls` sentinel and ends
# the turn, so it is a SILENT failure mode distinct from error_max_turns and has
# to be visible here or it gets counted as a win.
_OUTCOME_NO_TOOL_CALLS = 'no_tool_calls'


@dataclass
class Observation:
    """One CLI invocation's outcome."""

    max_turns: int
    shape: str
    outcome: str
    returncode: int
    subtype: str
    is_error: bool
    num_turns: Any
    has_payload: bool
    schema_salvaged: bool
    schema_tool_denied: bool
    tool_calls: int | None
    detail: str = ''

    def line(self) -> str:
        bits = [
            f'rc={self.returncode}',
            f'subtype={self.subtype or "-"!s}',
            f'is_error={self.is_error}',
            f'num_turns={self.num_turns}',
            f'payload={"yes" if self.has_payload else "NO"}',
            f'schema_salvaged={self.schema_salvaged}',
        ]
        if self.schema_tool_denied:
            bits.append('schema_tool_denied=True')
        if self.tool_calls is not None:
            bits.append(f'tool_calls={self.tool_calls}')
        out = f'    [{self.outcome:<16}] ' + '  '.join(bits)
        if self.detail:
            out += f'\n        {self.detail}'
        return out


@dataclass
class Shape:
    """A (system prompt, output schema) pair to probe."""

    name: str
    system_prompt: str
    output_schema: dict
    prompt: str
    # Only agent_loop's schema has a tool_calls array to inspect.
    counts_tool_calls: bool = False
    model: str = 'sonnet'
    argv_extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shape capture — the REAL prompts/schemas, never a re-typed approximation
# ---------------------------------------------------------------------------


class _ShapeCaptured(Exception):
    """Sentinel raised to stop ``verify()`` at its ``AgentLoop`` construction."""


def _capture_agent_shape(codebase_root: Path) -> Shape:
    """Capture the recon-verify system prompt exactly as production builds it.

    Drives the real ``CodebaseVerifier.verify()`` far enough to construct its
    ``AgentLoop`` — which is where verify.py's tool set is handed over — then
    aborts before any CLI call.  Deriving it this way (rather than re-listing
    the tools here) means the probe follows verify.py if its tool set changes.
    """
    from fused_memory.config.schema import ReconciliationConfig
    from fused_memory.reconciliation import verify as verify_mod
    from fused_memory.reconciliation.agent_loop import (
        CLAUDE_CLI_RESPONSE_SCHEMA,
        AgentLoop,
    )

    captured: dict[str, Any] = {}

    class _CapturingAgentLoop(AgentLoop):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            captured['agent'] = self
            raise _ShapeCaptured

    config = ReconciliationConfig(
        agent_llm_provider='claude_cli',
        agent_llm_model='sonnet',
        explore_codebase_root=str(codebase_root),
    )
    verifier = verify_mod.CodebaseVerifier(config=config)
    original = verify_mod.AgentLoop
    verify_mod.AgentLoop = _CapturingAgentLoop  # type: ignore[misc]
    try:
        asyncio.run(
            verifier.verify(
                claim='AgentLoop caps a single assistant round-trip, not the conversation.',
                context='Probe run; the verdict is irrelevant, only the invocation shape matters.',
                scope_hints=['fused-memory/src/fused_memory/reconciliation/agent_loop.py'],
            )
        )
    except _ShapeCaptured:
        pass
    finally:
        verify_mod.AgentLoop = original  # type: ignore[misc]

    agent = captured.get('agent')
    if agent is None:
        raise RuntimeError(
            'Failed to capture the recon-verify shape: CodebaseVerifier.verify() '
            'returned without constructing an AgentLoop. verify.py has changed '
            'shape; update _capture_agent_shape rather than substituting a toy '
            'prompt, which measures the wrong thing.'
        )

    tool_schemas = [t.to_anthropic_schema() for t in agent.tools.values()]
    system_prompt = agent._build_cli_system_prompt(tool_schemas)  # noqa: SLF001
    return Shape(
        name='recon-verify',
        system_prompt=system_prompt,
        output_schema=CLAUDE_CLI_RESPONSE_SCHEMA,
        prompt=(
            'Verify this claim against the codebase: AgentLoop.run() drives multi-turn '
            'externally, one CLI invocation per outer step. Investigate and call '
            '`verification_complete` with your findings.'
        ),
        counts_tool_calls=True,
    )


def _capture_judge_shape() -> Shape:
    from fused_memory.reconciliation.judge import JUDGE_VERDICT_SCHEMA
    from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

    return Shape(
        name='judge',
        system_prompt=JUDGE_SYSTEM_PROMPT,
        output_schema=JUDGE_VERDICT_SCHEMA,
        prompt=(
            'Evaluate this reconciliation run: 3 memories were written, 1 task was '
            'closed, and every claim cited a file path. Produce your verdict.'
        ),
    )


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _pool_token_env_names(accounts_file: Path) -> list[tuple[str, str]]:
    """Read ``(account_name, oauth_token_env)`` pairs from usage-accounts.yaml.

    Read rather than hardcoded: the pool membership changes (accounts get
    retired, re-tested, commented out), and a stale hardcoded list would silently
    probe a dead account and report it as a turn-cap failure.
    """
    if not accounts_file.is_file():
        return []
    try:
        import yaml
    except ImportError:
        print(f'! PyYAML unavailable; cannot read {accounts_file}', file=sys.stderr)
        return []
    data = yaml.safe_load(accounts_file.read_text()) or {}
    pairs: list[tuple[str, str]] = []
    for entry in data.get('accounts') or []:
        if isinstance(entry, dict) and entry.get('oauth_token_env'):
            pairs.append((str(entry.get('name') or '?'), str(entry['oauth_token_env'])))
    return pairs


def _classify_failure_text(text: str) -> str | None:
    lowered = text.lower()
    if any(m in lowered for m in _AUTH_FAILURE_MARKERS):
        return _OUTCOME_AUTH
    if any(m in lowered for m in _CREDIT_FAILURE_MARKERS):
        return _OUTCOME_CREDIT
    return None


def _auth_precheck(token: str, model: str, timeout: float) -> str | None:
    """Return None if the token works, else the failure outcome constant."""
    cmd = ['claude', '--print', '--output-format', 'json', '--model', model, '--max-turns', '1']
    env = _probe_env(token)
    config_dir = Path(tempfile.mkdtemp(prefix='probe_precheck_'))
    env['CLAUDE_CONFIG_DIR'] = str(config_dir)
    try:
        proc = subprocess.run(
            cmd, input=b'Reply with the single word: ok', capture_output=True,
            env=env, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return _OUTCOME_OTHER
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)
    blob = (proc.stdout or b'').decode(errors='replace') + (proc.stderr or b'').decode(errors='replace')
    return _classify_failure_text(blob)


def _resolve_tokens(accounts_file: Path, model: str, timeout: float) -> list[tuple[str, str]]:
    """Return usable ``(account_name, token)`` pairs, skipping ones that fail auth."""
    usable: list[tuple[str, str]] = []
    for name, env_name in _pool_token_env_names(accounts_file):
        token = os.environ.get(env_name)
        if not token:
            print(f'  - {name} ({env_name}): not set in env, skipped')
            continue
        verdict = _auth_precheck(token, model, timeout)
        if verdict is not None:
            print(f'  - {name} ({env_name}): pre-check failed ({verdict}), skipped')
            continue
        print(f'  - {name} ({env_name}): usable')
        usable.append((name, token))
    return usable


def _probe_env(token: str) -> dict[str, str]:
    """Production's env shape (cli_invoke's ``_invoke_claude``): strip the API key
    so the CLI falls back to OAuth, then inject the per-invocation token."""
    env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
    env['CLAUDE_CODE_OAUTH_TOKEN'] = token
    return env


# ---------------------------------------------------------------------------
# One invocation
# ---------------------------------------------------------------------------


def _run_once(shape: Shape, max_turns: int, token: str, cwd: Path, timeout: float) -> Observation:
    cmd, temp_files = build_claude_argv(
        model=shape.model,
        max_budget_usd=1.0,
        system_prompt=shape.system_prompt,
        max_turns=max_turns,
        permission_mode='bypassPermissions',
        allowed_tools=None,
        # Passed VERBATIM as production does. build_claude_argv expands the
        # wildcard into _REAL_BUILTIN_TOOLS_DENYLIST when an output_schema is
        # present, so that StructuredOutput survives; asserting the expansion
        # below (rather than passing a copied list) keeps the probe honest if
        # that central behaviour ever changes.
        disallowed_tools=['*'],
        mcp_config=no_mcp_servers_config(),
        output_schema=shape.output_schema,
        effort=None,
        resume_session_id=None,
        session_id=None,
        strict_mcp_config=True,
    )
    if _REAL_BUILTIN_TOOLS_DENYLIST and _REAL_BUILTIN_TOOLS_DENYLIST[0] not in cmd:
        print(
            '! build_claude_argv did not expand the wildcard deny into the real-builtins '
            'list; StructuredOutput may be blocked and every cell will fail for a reason '
            'that has nothing to do with the turn cap.',
            file=sys.stderr,
        )

    config_dir = Path(tempfile.mkdtemp(prefix='probe_maxturns_'))
    env = _probe_env(token)
    env['CLAUDE_CONFIG_DIR'] = str(config_dir)
    try:
        proc = subprocess.run(
            cmd, input=shape.prompt.encode(), capture_output=True,
            env=env, cwd=str(cwd), timeout=timeout, check=False,
        )
        stdout = (proc.stdout or b'').decode(errors='replace')
        stderr = (proc.stderr or b'').decode(errors='replace')
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        return Observation(
            max_turns=max_turns, shape=shape.name, outcome=_OUTCOME_OTHER,
            returncode=-1, subtype='', is_error=True, num_turns=None,
            has_payload=False, schema_salvaged=False, schema_tool_denied=False,
            tool_calls=None, detail=f'timed out after {timeout}s',
        )
    finally:
        shutil.rmtree(config_dir, ignore_errors=True)
        for path in temp_files:
            Path(path).unlink(missing_ok=True)

    try:
        data = json.loads(stdout) if stdout.strip() else {}
    except json.JSONDecodeError:
        data = {}

    subtype = str(data.get('subtype', ''))
    is_error = bool(data.get('is_error', False))
    structured = data.get('structured_output')
    has_payload = isinstance(structured, dict)
    # Mirrors cli_invoke's derivation exactly, so the reported value is the one
    # production would compute — the point being that it is False on failure.
    schema_salvaged = is_error and has_payload
    denials = data.get('permission_denials')
    schema_tool_denied = (
        not has_payload
        and isinstance(denials, list)
        and any(isinstance(d, dict) and d.get('tool_name') == 'StructuredOutput' for d in denials)
    )

    tool_calls: int | None = None
    if shape.counts_tool_calls and has_payload:
        calls = structured.get('tool_calls') if isinstance(structured, dict) else None
        tool_calls = len(calls) if isinstance(calls, list) else None

    detail = ''
    auth_or_credit = _classify_failure_text(str(data.get('result', '')) + stderr)
    if auth_or_credit is not None:
        outcome = auth_or_credit
        detail = 'NOT a turn-cap failure — this run never reached the model.'
    elif schema_tool_denied:
        outcome = _OUTCOME_OTHER
        detail = 'StructuredOutput was DENIED — a config break, not a turn-cap failure.'
    elif subtype == 'error_max_turns':
        outcome = _OUTCOME_MAX_TURNS
    elif is_error or returncode != 0 or not has_payload:
        outcome = _OUTCOME_OTHER
        detail = (str(data.get('result', '')) or stderr or stdout)[:300].replace('\n', ' ')
    elif tool_calls == 0:
        outcome = _OUTCOME_NO_TOOL_CALLS
        detail = 'Payload returned but tool_calls is EMPTY — run() reads this as the end of the turn.'
    else:
        outcome = _OUTCOME_OK

    return Observation(
        max_turns=max_turns, shape=shape.name, outcome=outcome, returncode=returncode,
        subtype=subtype, is_error=is_error, num_turns=data.get('num_turns'),
        has_payload=has_payload, schema_salvaged=schema_salvaged,
        schema_tool_denied=schema_tool_denied, tool_calls=tool_calls, detail=detail,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _cli_version() -> str:
    try:
        proc = subprocess.run(
            ['claude', '--version'], capture_output=True, timeout=30, check=False,
        )
        return (proc.stdout or b'').decode(errors='replace').strip() or '(unknown)'
    except (OSError, subprocess.TimeoutExpired):
        return '(claude CLI not runnable)'


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Measure the --max-turns x --json-schema interaction (task 3241).',
    )
    parser.add_argument(
        '--max-turns', type=int, nargs='+', default=[1, 3, 10], metavar='N',
        help='Turn caps to probe (default: 1 3 10).',
    )
    parser.add_argument(
        '--repeat', type=int, default=6, metavar='N',
        help='Runs per cell (default: 6). The failure is STOCHASTIC — one run per '
             'cell is what produced the earlier wrong verdict. Do not set this to 1 '
             'and then quote the result as a rate.',
    )
    parser.add_argument(
        '--shape', choices=['agent', 'judge', 'both'], default='agent',
        help="Which prompt/schema shape to probe (default: agent, i.e. recon-verify).",
    )
    parser.add_argument('--model', default='sonnet', help='Model to probe (default: sonnet).')
    parser.add_argument(
        '--codebase-root', default=str(_REPO_ROOT),
        help='cwd for the invocation, mirroring explore_codebase_root.',
    )
    parser.add_argument(
        '--accounts-file', default=str(_DEFAULT_ACCOUNTS_FILE),
        help='usage-accounts.yaml to read oauth_token_env names from.',
    )
    parser.add_argument('--timeout', type=float, default=300.0, help='Per-run timeout seconds.')
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error('--repeat must be >= 1')

    codebase_root = Path(args.codebase_root).resolve()

    print('=' * 78)
    print('probe_schema_max_turns — MANUAL diagnostic, spends real tokens (task 3241)')
    print(f'claude --version : {_cli_version()}')
    print(f'model            : {args.model}')
    print(f'matrix           : max_turns={args.max_turns} x repeat={args.repeat}')
    print(f'cwd              : {codebase_root}')
    print('=' * 78)

    print('\nResolving pool credentials:')
    tokens = _resolve_tokens(Path(args.accounts_file), args.model, args.timeout)
    if not tokens:
        print(
            '\nNo usable account tokens. Aborting rather than reporting a uniform 0/N, '
            'which would look exactly like a turn-cap failure and be misread as one.',
            file=sys.stderr,
        )
        return 2

    shapes: list[Shape] = []
    if args.shape in ('agent', 'both'):
        shapes.append(_capture_agent_shape(codebase_root))
    if args.shape in ('judge', 'both'):
        shapes.append(_capture_judge_shape())
    for shape in shapes:
        shape.model = args.model
        print(f'\nShape {shape.name!r}: system prompt {len(shape.system_prompt)} chars, '
              f'schema keys={sorted((shape.output_schema.get("properties") or {}).keys())}')

    results: dict[tuple[str, int], list[Observation]] = {}
    for shape in shapes:
        for max_turns in args.max_turns:
            print(f'\n--- shape={shape.name} max_turns={max_turns} ---')
            observations: list[Observation] = []
            for i in range(args.repeat):
                _, token = tokens[i % len(tokens)]
                obs = _run_once(shape, max_turns, token, codebase_root, args.timeout)
                observations.append(obs)
                print(obs.line())
            results[(shape.name, max_turns)] = observations

    print('\n' + '=' * 78)
    print('SUMMARY (success rate excludes auth/credit runs, which never reached the model)')
    print('=' * 78)
    exhausted = False
    for (shape_name, max_turns), observations in results.items():
        counts = Counter(o.outcome for o in observations)
        reached = [o for o in observations if o.outcome not in (_OUTCOME_AUTH, _OUTCOME_CREDIT)]
        ok = counts[_OUTCOME_OK]
        rate = f'{ok}/{len(reached)}' if reached else 'n/a (0 runs reached the model)'
        breakdown = ', '.join(f'{k}={v}' for k, v in sorted(counts.items()))
        print(f'  {shape_name:<14} max_turns={max_turns:<3} -> {rate:<10} [{breakdown}]')
        if counts[_OUTCOME_CREDIT]:
            exhausted = True
    salvaged = sum(1 for obs in results.values() for o in obs if o.schema_salvaged)
    failures = sum(1 for obs in results.values() for o in obs if o.outcome != _OUTCOME_OK)
    print(f'\n  schema_salvaged fired on {salvaged} of {failures} non-success runs.')
    print(
        '  (If that is 0, salvage is not a backstop on this path: an error_max_turns\n'
        '   result carries no payload, so there is nothing to salvage.)'
    )
    if exhausted:
        print('\n! Some runs hit a usage cap. Those cells are under-sampled — re-run them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
