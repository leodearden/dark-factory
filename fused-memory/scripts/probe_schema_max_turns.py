#!/usr/bin/env python3
"""MANUAL diagnostic: measure the ``--max-turns`` x ``--json-schema`` interaction.

Task 3241.  This docstring is the single record of the measured numbers behind
the recon-verify invocation SHAPE: every ``max_turns`` floor on a
``--json-schema`` invocation in fused-memory
(``agent_loop.py::_AGENT_CLI_MAX_TURNS``, ``judge.py::_JUDGE_CLI_MAX_TURNS``,
and the curator / path-scope-adjudicator ``ge=3`` floors), and the property set
of ``agent_loop.py::CLAUDE_CLI_RESPONSE_SCHEMA`` (task 6022, below).  Those
sites state the mechanism and point here; rates, sample sizes and CLI versions
live only below.  The behaviour is CLI-VERSION-DEPENDENT: re-run this script
rather than trusting a number copied anywhere else.

NOT part of the pytest suite, deliberately.  It requires live Claude CLI
credentials and spends real tokens; the suite stays hermetic and offline.

Baseline to diff against
------------------------
Recon-verify shape (EXPLORE_AGENT_SYSTEM_PROMPT + CLAUDE_CLI_RESPONSE_SCHEMA),
6 repeats per cell:

    Claude CLI 2.1.236:  mt=1 -> 0/6   mt=3 -> 4/6   mt=6 -> 4/6   mt=10 -> 6/6
    Claude CLI 2.1.233:  mt=1 -> 0/6   mt=3 -> 2/6   mt=6 -> 4/5   mt=10 -> 4/5

The model emits a prose turn before it calls ``StructuredOutput``, and a cap of
1 leaves no room for it.  Every failure was ``subtype='error_max_turns'`` with
NO structured payload, so ``schema_salvaged`` was False every time and salvage
never engaged.  ``mt=1 -> 0/6`` held on both CLI versions; the intermediate
rates moved between them.  The failure is STOCHASTIC, which is why every cell
is repeated: one run per cell is what produced an earlier, wrong "max_turns=1
is safe" verdict.  Six clean runs at mt=10 cannot exclude a residual failure
rate of a few tens of percent.

``num_turns`` is not the counter ``--max-turns`` bounds: every failure reported
``num_turns == max_turns + 1``, but successes at mt=10 reported 9, 11 and 14.

The judge's shape (``--shape judge``) has no recorded baseline yet.

Task 6022: reasoning_extraction refusals
----------------------------------------
Measured on Claude CLI 2.1.285, model alias ``sonnet`` = Sonnet 5.5, one pool
account (max-g; the others were capped or org-disabled at the time).

Provenance: these runs used the pre-task-3995 argv, where
``disallowed_tools=['*']`` with an ``output_schema`` expanded into an
enumerated built-in deny list.  Task 3995 has since replaced that with
``--tools ''``.  Both strip real tools, and the bisected variable (the
"thinking" schema field plus its prompt instruction) is independent of tool
scoping.  Still, record the argv when re-running so results compare like for
like.

* End-to-end ``CodebaseVerifier.verify()`` on the claims of production-refused
  tasks 5546, 4449, 6017 and 5945, with a required "thinking" schema field plus
  the system-prompt line telling the model to use it to "explain your
  reasoning": 4 of 4 refused.  One refusal came at outer step 3 (after two
  tool-call steps); the others at step 1.
* The same 4 tasks with that field and that line removed: 4 of 4 real verdicts
  (3 confirmed, 1 inconclusive), 3 to 4 steps each with real tool calls.
* Single-shot recon-verify shape: baseline 1 of 2 reached runs refused; field
  removed 0 of 2; field renamed to "notes" 0 of 1.
* The refused CLI JSON carries ``stop_reason='refusal'``,
  ``terminal_reason='api_error'``, ``is_error=true``, ``subtype='success'``
  and ``api_error_status=null``.  Its ``result`` text varies by model version
  (Sonnet 5 named ``[reasoning_extraction]``; Sonnet 5.5 said its "safeguards
  flagged this message"), which is why ``classify_agent_failure`` keys
  ``API_REFUSAL`` on ``stop_reason`` and never on the prose.  Such runs are
  reported here as ``api_refusal``.

Hypothesis, not observable from outside the API: a required output field named
"thinking", together with an instruction to explain reasoning in it, reads to
the classifier as an attempt to extract the model's chain of thought.  The
refusal is STOCHASTIC per call, so re-run any schema or prompt change with
repeats rather than trusting a single clean run.

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
* Every run goes through ``shared.cli_invoke.invoke_claude_agent`` — the call
  production's ``invoke_with_cap_retry`` makes — with the same schema, deny
  and MCP kwargs agent_loop and the judge pass.  argv, env handling, the
  default ``max_budget_usd`` and the ``AgentResult`` verdict (``success``,
  ``schema_salvaged``, ``schema_tool_denied``) are production's own.
* Model and per-run timeout come from ``ReconciliationConfig``
  (``agent_cli_timeout_seconds`` / ``judge_cli_timeout_seconds``), so a run
  production would kill is counted as ``timed_out`` here, not as a success.
* The system prompt is the REAL one, captured by driving
  ``CodebaseVerifier.verify()`` up to its ``AgentLoop`` construction.  A toy
  system prompt SUCCEEDS at ``max_turns=1`` and manufactures the opposite
  verdict — this is the single biggest way to get this measurement wrong.
* Each run gets a pool account's OAuth token and a throwaway
  ``CLAUDE_CONFIG_DIR``.  Without credentials the CLI returns
  ``is_error=True, subtype='success', num_turns=1,
  result='Not logged in - Please run /login'``, which is NOT a turn-cap
  failure but looks like a uniform 0/N and is easily misread as one.  Auth and
  credit failures are detected and reported SEPARATELY from ``error_max_turns``.
* No token value is ever printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FM_SRC = _REPO_ROOT / 'fused-memory' / 'src'
_SHARED_SRC = _REPO_ROOT / 'shared' / 'src'
for _p in (_FM_SRC, _SHARED_SRC):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.cli_invoke import (  # noqa: E402
    AgentFailureKind,
    AgentResult,
    classify_agent_failure,
    invoke_claude_agent,
    no_mcp_servers_config,
)

from fused_memory.config.schema import ReconciliationConfig  # noqa: E402

_DEFAULT_ACCOUNTS_FILE = _REPO_ROOT / 'config' / 'usage-accounts.yaml'
_PRECHECK_TIMEOUT_SECS = 120.0
# A run this close to its production timeout is flagged: it passed here, but
# ordinary latency variance would have it killed in production.
_NEAR_TIMEOUT_FRACTION = 0.8

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
    'disabled claude subscription access',
)
_CREDIT_FAILURE_MARKERS = (
    'hit your weekly limit',
    'usage limit reached',
    'out of credit',
    'credit balance is too low',
)

_OUTCOME_OK = 'ok'
_OUTCOME_MAX_TURNS = 'error_max_turns'
_OUTCOME_TIMEOUT = 'timed_out'
_OUTCOME_AUTH = 'auth_failure'
_OUTCOME_CREDIT = 'credit_exhausted'
_OUTCOME_OTHER = 'other_failure'
# The API's usage-policy safeguards refused the call (task 6022).  The run DID
# reach the model, so unlike auth/credit it stays in the success-rate denominator.
_OUTCOME_API_REFUSAL = 'api_refusal'
# Not a CLI error at all: a success whose payload carries an EMPTY tool_calls
# array.  agent_loop's run() reads that as the `no_tool_calls` sentinel and ends
# the turn, so it is a SILENT failure mode distinct from error_max_turns and has
# to be visible here or it gets counted as a win.
_OUTCOME_NO_TOOL_CALLS = 'no_tool_calls'


@dataclass(frozen=True)
class Observation:
    """One CLI invocation's outcome."""

    max_turns: int
    shape: str
    outcome: str
    success: bool
    subtype: str
    num_turns: int
    has_payload: bool
    schema_salvaged: bool
    schema_tool_denied: bool
    tool_calls: int | None
    duration_s: float
    timeout_s: float
    detail: str = ''

    @property
    def near_timeout(self) -> bool:
        return self.duration_s >= _NEAR_TIMEOUT_FRACTION * self.timeout_s

    def line(self) -> str:
        bits = [
            f'success={self.success}',
            f'subtype={self.subtype or "-"!s}',
            f'num_turns={self.num_turns}',
            f'payload={"yes" if self.has_payload else "NO"}',
            f'schema_salvaged={self.schema_salvaged}',
            f'took={self.duration_s:.0f}s/{self.timeout_s:.0f}s',
        ]
        if self.near_timeout:
            bits.append('NEAR-TIMEOUT')
        if self.schema_tool_denied:
            bits.append('schema_tool_denied=True')
        if self.tool_calls is not None:
            bits.append(f'tool_calls={self.tool_calls}')
        out = f'    [{self.outcome:<16}] ' + '  '.join(bits)
        if self.detail:
            out += f'\n        {self.detail}'
        return out


@dataclass(frozen=True)
class Shape:
    """A (system prompt, output schema) pair to probe, with production's limits."""

    name: str
    system_prompt: str
    output_schema: dict
    prompt: str
    model: str
    timeout_seconds: float
    # Only agent_loop's schema has a tool_calls array to inspect.
    counts_tool_calls: bool = False


# ---------------------------------------------------------------------------
# Shape capture — the REAL prompts/schemas, never a re-typed approximation
# ---------------------------------------------------------------------------


class _ShapeCaptured(Exception):
    """Sentinel raised to stop ``verify()`` at its ``AgentLoop`` construction."""


def _capture_agent_shape(config: ReconciliationConfig, codebase_root: Path) -> Shape:
    """Capture the recon-verify system prompt exactly as production builds it.

    Drives the real ``CodebaseVerifier.verify()`` far enough to construct its
    ``AgentLoop`` — which is where verify.py's tool set is handed over — then
    aborts before any CLI call.  Deriving it this way (rather than re-listing
    the tools here) means the probe follows verify.py if its tool set changes.
    """
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

    # verify() takes ``codebase_root`` per call and does not read
    # ``config.explore_codebase_root`` (verify.py::CodebaseVerifier __init__,
    # PRD D3), so the root is passed here, not configured.
    verifier = verify_mod.CodebaseVerifier(config=config)
    original = verify_mod.AgentLoop
    verify_mod.AgentLoop = _CapturingAgentLoop  # type: ignore[misc]
    returned = None
    try:
        returned = asyncio.run(
            verifier.verify(
                claim='AgentLoop caps a single assistant round-trip, not the conversation.',
                context='Probe run; the verdict is irrelevant, only the invocation shape matters.',
                scope_hints=['fused-memory/src/fused_memory/reconciliation/agent_loop.py'],
                codebase_root=codebase_root,
            )
        )
    except _ShapeCaptured:
        pass
    finally:
        verify_mod.AgentLoop = original  # type: ignore[misc]

    agent = captured.get('agent')
    if agent is None:
        raise RuntimeError(_capture_failure_message(returned, codebase_root))

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
        model=config.agent_llm_model,
        timeout_seconds=float(config.agent_cli_timeout_seconds),
        counts_tool_calls=True,
    )


def _capture_failure_message(returned: Any, codebase_root: Path) -> str:
    """Say WHY verify() returned without building an AgentLoop.

    A refused root is an input error the operator fixes with --codebase-root;
    anything else means verify.py changed shape and this script needs updating.
    """
    from fused_memory.reconciliation.verify import CODEBASE_ROOT_UNRESOLVED

    summary = getattr(returned, 'summary', None) or '(no VerificationResult returned)'
    if getattr(returned, 'failure_token', '') == CODEBASE_ROOT_UNRESOLVED:
        return (
            f'verify() refused --codebase-root {codebase_root}: {summary}. '
            'Pass the root of a git checkout.'
        )
    return (
        'Failed to capture the recon-verify shape: CodebaseVerifier.verify() '
        f'returned without constructing an AgentLoop ({summary}). verify.py has '
        'changed shape; update _capture_agent_shape rather than substituting a '
        'toy prompt, which measures the wrong thing.'
    )


def _capture_judge_shape(config: ReconciliationConfig) -> Shape:
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
        model=config.judge_llm_model,
        timeout_seconds=float(config.judge_cli_timeout_seconds),
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


def _auth_precheck(token: str, model: str) -> str | None:
    """Return None only if a trivial call SUCCEEDS, else the failure outcome.

    Any failure disqualifies the account, recognised marker or not: an account
    that cannot answer "ok" would be counted as a 0/N turn-cap result.
    """
    with tempfile.TemporaryDirectory(prefix='probe_precheck_') as config_dir:
        result = asyncio.run(
            invoke_claude_agent(
                prompt='Reply with the single word: ok',
                system_prompt='Connectivity check.',
                cwd=Path(config_dir),
                model=model,
                max_turns=1,
                oauth_token=token,
                config_dir=Path(config_dir),
                timeout_seconds=_PRECHECK_TIMEOUT_SECS,
            )
        )
    if result.success:
        return None
    if result.timed_out:
        return _OUTCOME_TIMEOUT
    return _classify_failure_text(result.output + result.stderr) or _OUTCOME_OTHER


def _resolve_tokens(accounts_file: Path, model: str) -> list[tuple[str, str]]:
    """Return usable ``(account_name, token)`` pairs, skipping ones that fail auth."""
    usable: list[tuple[str, str]] = []
    for name, env_name in _pool_token_env_names(accounts_file):
        token = os.environ.get(env_name)
        if not token:
            print(f'  - {name} ({env_name}): not set in env, skipped')
            continue
        verdict = _auth_precheck(token, model)
        if verdict is not None:
            print(f'  - {name} ({env_name}): pre-check failed ({verdict}), skipped')
            continue
        print(f'  - {name} ({env_name}): usable')
        usable.append((name, token))
    return usable


# ---------------------------------------------------------------------------
# One invocation
# ---------------------------------------------------------------------------


def _run_once(shape: Shape, max_turns: int, token: str, cwd: Path) -> Observation:
    with tempfile.TemporaryDirectory(prefix='probe_maxturns_') as config_dir:
        started = time.monotonic()
        result = asyncio.run(
            invoke_claude_agent(
                prompt=shape.prompt,
                system_prompt=shape.system_prompt,
                cwd=cwd,
                model=shape.model,
                max_turns=max_turns,
                output_schema=shape.output_schema,
                disallowed_tools=['*'],
                mcp_config=no_mcp_servers_config(),
                strict_mcp_config=True,
                oauth_token=token,
                config_dir=Path(config_dir),
                timeout_seconds=shape.timeout_seconds,
            )
        )
        duration_s = time.monotonic() - started

    structured = result.structured_output
    has_payload = isinstance(structured, dict)
    tool_calls: int | None = None
    if shape.counts_tool_calls and isinstance(structured, dict):
        calls = structured.get('tool_calls')
        tool_calls = len(calls) if isinstance(calls, list) else None

    outcome, detail = _classify(result, has_payload, tool_calls, shape.timeout_seconds)
    return Observation(
        max_turns=max_turns, shape=shape.name, outcome=outcome, success=result.success,
        subtype=result.subtype, num_turns=result.turns, has_payload=has_payload,
        schema_salvaged=result.schema_salvaged, schema_tool_denied=result.schema_tool_denied,
        tool_calls=tool_calls, duration_s=duration_s, timeout_s=shape.timeout_seconds,
        detail=detail,
    )


def _classify(
    result: AgentResult, has_payload: bool, tool_calls: int | None, timeout: float,
) -> tuple[str, str]:
    """Map production's ``AgentResult`` verdict onto a probe outcome and detail."""
    if not result.success:
        auth_or_credit = _classify_failure_text(result.output + result.stderr)
        if auth_or_credit is not None:
            return auth_or_credit, 'NOT a turn-cap failure — this run never reached the model.'
        if classify_agent_failure(result).kind == AgentFailureKind.API_REFUSAL:
            return _OUTCOME_API_REFUSAL, (
                'Refused by API usage-policy safeguards (stop_reason=refusal), '
                'NOT a turn-cap failure.'
            )
        if result.schema_tool_denied:
            return _OUTCOME_OTHER, 'StructuredOutput was DENIED — a config break, not a turn-cap failure.'
        if result.timed_out:
            return _OUTCOME_TIMEOUT, f'Killed at the production timeout ({timeout:.0f}s).'
        if result.subtype == 'error_max_turns':
            return _OUTCOME_MAX_TURNS, ''
        return _OUTCOME_OTHER, (result.output or result.stderr)[:300].replace('\n', ' ')
    if not has_payload:
        return _OUTCOME_OTHER, 'success reported but no structured payload came back.'
    if tool_calls == 0:
        return _OUTCOME_NO_TOOL_CALLS, (
            'Payload returned but tool_calls is EMPTY — run() reads this as the end of the turn.'
        )
    return _OUTCOME_OK, ''


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
    parser.add_argument(
        '--model', default=None,
        help="Model to probe (default: each shape's production model from ReconciliationConfig).",
    )
    parser.add_argument(
        '--codebase-root', default=str(_REPO_ROOT),
        help='cwd for the invocation, and the root handed to verify().',
    )
    parser.add_argument(
        '--accounts-file', default=str(_DEFAULT_ACCOUNTS_FILE),
        help='usage-accounts.yaml to read oauth_token_env names from.',
    )
    parser.add_argument(
        '--timeout', type=float, default=None,
        help="Per-run timeout seconds (default: each shape's production value — "
             'agent_cli_timeout_seconds / judge_cli_timeout_seconds). Raising it '
             'hides runs production would kill.',
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error('--repeat must be >= 1')

    codebase_root = Path(args.codebase_root).resolve()
    config = ReconciliationConfig()

    print('=' * 78)
    print('probe_schema_max_turns — MANUAL diagnostic, spends real tokens (task 3241)')
    print(f'claude --version : {_cli_version()}')
    print(f'matrix           : max_turns={args.max_turns} x repeat={args.repeat}')
    print(f'cwd              : {codebase_root}')
    print('=' * 78)

    print('\nResolving pool credentials:')
    tokens = _resolve_tokens(Path(args.accounts_file), args.model or config.agent_llm_model)
    if not tokens:
        print(
            '\nNo usable account tokens. Aborting rather than reporting a uniform 0/N, '
            'which would look exactly like a turn-cap failure and be misread as one.',
            file=sys.stderr,
        )
        return 2

    shapes: list[Shape] = []
    if args.shape in ('agent', 'both'):
        shapes.append(_capture_agent_shape(config, codebase_root))
    if args.shape in ('judge', 'both'):
        shapes.append(_capture_judge_shape(config))
    shapes = [
        replace(
            shape,
            model=args.model or shape.model,
            timeout_seconds=args.timeout or shape.timeout_seconds,
        )
        for shape in shapes
    ]
    for shape in shapes:
        print(f'\nShape {shape.name!r}: model={shape.model} timeout={shape.timeout_seconds:.0f}s '
              f'system prompt {len(shape.system_prompt)} chars, '
              f'schema keys={sorted((shape.output_schema.get("properties") or {}).keys())}')

    results: dict[tuple[str, int], list[Observation]] = {}
    for shape in shapes:
        for max_turns in args.max_turns:
            print(f'\n--- shape={shape.name} max_turns={max_turns} ---')
            observations: list[Observation] = []
            for i in range(args.repeat):
                _, token = tokens[i % len(tokens)]
                obs = _run_once(shape, max_turns, token, codebase_root)
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
        slowest = max(observations, key=lambda o: o.duration_s)
        print(f'  {shape_name:<14} max_turns={max_turns:<3} -> {rate:<10} [{breakdown}] '
              f'slowest={slowest.duration_s:.0f}s/{slowest.timeout_s:.0f}s'
              f'{" NEAR-TIMEOUT" if slowest.near_timeout else ""}')
        if counts[_OUTCOME_CREDIT]:
            exhausted = True
    hit_cap = [o for obs in results.values() for o in obs if o.subtype == 'error_max_turns']
    salvaged = sum(1 for o in hit_cap if o.schema_salvaged)
    print(f'\n  error_max_turns on {len(hit_cap)} run(s); schema salvage rescued {salvaged}.')
    print(
        '  (A salvaged run counts as a success above, as production treats it. If\n'
        '   none were salvaged, salvage is not a backstop on this path: the\n'
        '   error_max_turns results carried no payload to recover.)'
    )
    if exhausted:
        print('\n! Some runs hit a usage cap. Those cells are under-sampled — re-run them.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
