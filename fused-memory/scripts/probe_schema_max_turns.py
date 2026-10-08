#!/usr/bin/env python3
"""MANUAL diagnostic: measure production's ``--json-schema`` CLI invocations.

Tasks 3241 and 4344.  This docstring is the single record of the measured
numbers behind every ``max_turns`` cap on a ``--json-schema`` invocation in
fused-memory (``agent_loop.py::_AGENT_CLI_MAX_TURNS``,
``judge.py::_JUDGE_CLI_MAX_TURNS``, and the curator / path-scope-adjudicator
``ge=3`` floors), and behind the property set of the recon-verify output
schema, ``verify.py``'s ``verification_complete`` parameters (formerly the
retired pseudo-tool response schema; task 6022, below).  Those sites state the
mechanism and point here; rates, sample sizes and CLI versions live only below.
The behaviour is CLI-VERSION-DEPENDENT: re-run this script rather than trusting
a number copied anywhere else.

NOT part of the pytest suite, deliberately.  It requires live Claude CLI
credentials and spends real tokens; the suite stays hermetic and offline.

Current shape (task 4344)
-------------------------
Recon-verify is ONE CLI invocation: ``--tools Read,Grep,Glob``,
``--permission-mode dontAsk`` with no allow rules, ``--setting-sources ''``,
the strict empty MCP config, ``--json-schema`` = the ``verification_complete``
parameters, and ``max_turns=_AGENT_CLI_MAX_TURNS`` (20).

Measured on Claude CLI 2.1.293, alias ``sonnet`` served as
``claude-sonnet-5-5``, 2026-10-07, ``--repeat 12`` at the production cap, one
pool account (max-c; four were usage-capped and two failed auth that day),
BEFORE ``--setting-sources ''``, so with the root's CLAUDE.md loaded:

    recon-verify  max_turns=20 -> 12/12 ok   num_turns 5 on every run
                  slowest 41s of the 180s timeout (others 12-22s)
                  unregistered_tool_calls 0   api_refusal 0   error_max_turns 0

Permission facts
----------------
The verifier's read-only, root-confined contract rests on these.  Re-check them
on every CLI upgrade.

* CLI 2.1.287 (task 4344 architect, 2026-10-02): ``dontAsk`` with no allow
  rules lets in-cwd Read/Grep through and DENIES out-of-cwd Read/Grep.  A bare
  ``'Read'`` allow rule widened reads to ``/etc/hostname``, so the verifier
  passes no allow rules.  ``'Bash(git log:*)'`` under ``dontAsk`` allowed
  ``git log -1 --output=FILE`` to WRITE a file, so a git-scoped Bash is not
  read-only and the CLI path gets no git access.  ``StructuredOutput``
  survives ``--tools <list>`` under ``dontAsk``.
* CLI 2.1.293 (task 4344 implementer, 2026-10-07, sonnet, one ad-hoc
  ``invoke_claude_agent`` call with the production kwargs, a throwaway cwd and
  ``CLAUDE_CONFIG_DIR``): (i) Read of a file in cwd succeeded; (ii) Read of
  ``/etc/hostname`` and Grep of the cwd's parent directory were both DENIED,
  each with "Permission to use <Tool> has been denied because Claude Code is
  running in don't ask mode"; (iii) ``StructuredOutput`` delivered the schema
  payload (accepted in the transcript); (iv) the transcript's tool uses were
  Read, Read, Grep, StructuredOutput — nothing outside the registry.
* CLI 2.1.293 (task 4344 amendment, 2026-10-08, sonnet, ad-hoc
  ``invoke_claude_agent`` calls with the production kwargs, a throwaway cwd
  holding a CLAUDE.md, and a throwaway ``CLAUDE_CONFIG_DIR``): the facts above
  hold only while no settings FILE carries an allow rule.  A
  ``Read(//etc/**)`` allow rule in EITHER the cwd's
  ``.claude/settings.local.json`` OR the config dir's ``settings.json`` let a
  Read of ``/etc/hostname`` through under ``dontAsk``.  With
  ``setting_sources=[]`` (``--setting-sources ''``) and both rules present the
  same Read was DENIED, while in-cwd Read and ``StructuredOutput`` still
  worked; so the verifier passes it.  The flag also stops the CLI loading the
  cwd's CLAUDE.md: a fact stated only there was reported without the flag and
  not with it.  Managed policy settings are not a setting source and would
  still apply.

RETIRED pseudo-tool shape (pre-4344) history
--------------------------------------------
Until task 4344, AgentLoop's claude_cli path listed the in-process tools in the
system prompt and dispatched them as JSON inside a ``tool_calls`` array of the
output schema, one CLI call per outer step threaded with ``--resume``.  The
model kept calling those pseudo-tools NATIVELY, and the CLI rejected each call.
That attractor is why the shape was replaced, not reworded.

Recon-verify pseudo-tool shape, 6 repeats per cell:

    Claude CLI 2.1.236:  mt=1 -> 0/6   mt=3 -> 4/6   mt=6 -> 4/6   mt=10 -> 6/6
    Claude CLI 2.1.233:  mt=1 -> 0/6   mt=3 -> 2/6   mt=6 -> 4/5   mt=10 -> 4/5

The model emits a prose turn before it calls ``StructuredOutput``, and a cap of
1 leaves no room for it.  Every failure was ``subtype='error_max_turns'`` with
NO structured payload, so ``schema_salvaged`` was False every time and salvage
never engaged.  ``mt=1 -> 0/6`` held on both CLI versions; the intermediate
rates moved between them.  The failure is STOCHASTIC, which is why every cell
is repeated: one run per cell is what produced an earlier, wrong "max_turns=1
is safe" verdict.

``num_turns`` is not the counter ``--max-turns`` bounds: every failure reported
``num_turns == max_turns + 1``, but successes at mt=10 reported 9, 11 and 14.

Task 5919, CLI 2.1.283, sonnet (2026-09-26):

* pseudo-tool shape, mt=1: 0/3, every run ending ``error_max_turns`` right
  after a native pseudo-tool call;
* pseudo-tool shape, mt=10: 7/7, but every run first wasted 1-4 turns on
  rejected native pseudo-tool calls;
* prompt-only rewording at mt=1, five variants: 5/20 combined;
* native Read/Grep/Glob plus the verdict schema at mt=20: 5/5, 4 turns each,
  zero rejected calls.

Task 4344 architect STEP 0, CLI 2.1.287, sonnet (2026-10-02):

* pseudo-tool shape at mt=10: 12/12, num_turns 3-5, 5-10s of 180s, zero
  ``error_max_turns``, zero empty ``tool_calls``.  Pooled mt=10 is 28/28 across
  four CLI versions, so the ~20% residual plateau once claimed is refuted for
  the turn-cap question.
* transcript scan of production '## Verification Request' sessions,
  pre-3241: 152 sessions, 83 with native pseudo-tool calls, 100 ending at
  maxTurns 1.
* the same scan post-3241: 15 sessions, 13 of them turn-1 refusals or account
  errors.  Both working sessions (c99dfc96, 88a58609) opened with native
  pseudo-tool calls, so the attractor survived the cap raise.
* the planned native shape at mt=20: 3/3 runs that reached the model, 4-8
  turns, 5-16s, zero rejected calls, zero denials.

The judge's shape (``--shape judge``) has no recorded baseline yet.

Task 6022: reasoning_extraction refusals
----------------------------------------
Measured on Claude CLI 2.1.285, model alias ``sonnet`` = Sonnet 5.5, one pool
account (max-g; the others were capped or org-disabled at the time).  These
runs used the RETIRED pseudo-tool schema.  The current schema carries a
free-text ``summary`` but no reasoning field; any property added to it must be
re-probed here first.

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
    # Each shape at its production cap, 6 repeats, recon-verify shape.
    python scripts/probe_schema_max_turns.py

    # An explicit turn-cap matrix.
    python scripts/probe_schema_max_turns.py --max-turns 1 10 --repeat 2

    # Both shapes, including the judge's.
    python scripts/probe_schema_max_turns.py --shape both

Fidelity notes (all load-bearing — a naive probe gets a wrong answer)
--------------------------------------------------------------------
* Every run goes through ``shared.cli_invoke.invoke_claude_agent`` — the call
  production's ``invoke_with_cap_retry`` makes.  argv, env handling, the
  default ``max_budget_usd`` and the ``AgentResult`` verdict (``success``,
  ``schema_salvaged``, ``schema_tool_denied``) are production's own.
* The recon-verify invocation is CAPTURED, not re-assembled: the real
  ``CodebaseVerifier.verify()`` runs on the claude_cli provider until AgentLoop
  calls ``invoke_with_cap_retry``, and that call's kwargs are replayed — system
  prompt, user prompt, schema, ``available_tools``, permission mode and MCP
  closure alike.  Only ``max_turns``, ``timeout_seconds``, ``model`` and
  ``cwd`` are the probe's own.  A toy system prompt SUCCEEDS at
  ``max_turns=1`` and manufactures the opposite verdict — this is the single
  biggest way to get this measurement wrong.
* Model and per-run timeout come from ``ReconciliationConfig``
  (``agent_cli_timeout_seconds`` / ``judge_cli_timeout_seconds``), so a run
  production would kill is counted as ``timed_out`` here, not as a success.
* Each run gets a pool account's OAuth token and a throwaway
  ``CLAUDE_CONFIG_DIR``.  Without credentials the CLI returns
  ``is_error=True, subtype='success', num_turns=1,
  result='Not logged in - Please run /login'``, which is NOT a turn-cap
  failure but looks like a uniform 0/N and is easily misread as one.  Auth and
  credit failures are detected and reported SEPARATELY from ``error_max_turns``.
* Each run's transcript is read before its ``CLAUDE_CONFIG_DIR`` is deleted.
  ``unregistered_tool_calls`` counts its tool uses outside the shape's
  ``available_tools`` (every tool use, for the judge's empty registry): each is
  a native call the CLI rejected, the attractor that retired the pseudo-tool
  shape.  ``model_id`` is the exact served model read from the transcript.
* No token value is ever printed.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

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
    transcript_evidence_for_session,
    transcript_model_id_for_session,
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
    'hit your session limit',
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

# invoke_claude_agent kwargs the probe sets itself, per run: the matrix sets
# max_turns, --model and --timeout override model and timeout_seconds, and cwd
# is --codebase-root.  Everything else is production's own captured value.
_PROBE_CONTROLLED_KWARGS = frozenset({'max_turns', 'timeout_seconds', 'model', 'cwd'})


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
    # Native tool calls to a tool outside the shape's ``available_tools``: the
    # CLI rejects each one, so this is the native-call attractor signal.  None
    # when the run's transcript could not be read.
    unregistered_tool_calls: int | None
    model_id: str | None
    duration_s: float
    timeout_s: float
    detail: str = ''

    @property
    def near_timeout(self) -> bool:
        return self.duration_s >= _NEAR_TIMEOUT_FRACTION * self.timeout_s

    def line(self) -> str:
        unregistered = '?' if self.unregistered_tool_calls is None else self.unregistered_tool_calls
        bits = [
            f'success={self.success}',
            f'subtype={self.subtype or "-"!s}',
            f'num_turns={self.num_turns}',
            f'payload={"yes" if self.has_payload else "NO"}',
            f'schema_salvaged={self.schema_salvaged}',
            f'unregistered_tool_calls={unregistered}',
            f'model_id={self.model_id or "-"}',
            f'took={self.duration_s:.0f}s/{self.timeout_s:.0f}s',
        ]
        if self.near_timeout:
            bits.append('NEAR-TIMEOUT')
        if self.schema_tool_denied:
            bits.append('schema_tool_denied=True')
        out = f'    [{self.outcome:<16}] ' + '  '.join(bits)
        if self.detail:
            out += f'\n        {self.detail}'
        return out


@dataclass(frozen=True)
class Shape:
    """One production CLI invocation to probe, with production's limits.

    ``cli_kwargs`` are the ``invoke_claude_agent`` kwargs production passes,
    minus :data:`_PROBE_CONTROLLED_KWARGS`.  ``max_turns`` is production's cap,
    the default when ``--max-turns`` is not given.
    """

    name: str
    cli_kwargs: Mapping[str, Any]
    model: str
    timeout_seconds: float
    max_turns: int


# ---------------------------------------------------------------------------
# Shape capture — production's exact call, never a re-typed approximation
# ---------------------------------------------------------------------------


class _ShapeCaptured(Exception):
    """Sentinel raised to stop ``verify()`` at its CLI invocation."""


def _capture_agent_shape(config: ReconciliationConfig, codebase_root: Path) -> Shape:
    """Capture the recon-verify CLI invocation exactly as production makes it.

    Drives the real ``CodebaseVerifier.verify()`` on the claude_cli provider up
    to the moment AgentLoop calls ``invoke_with_cap_retry``, records that
    call's kwargs, and aborts before any CLI call.  The probe therefore follows
    any change to verify.py or agent_loop.py's invocation automatically.
    """
    from fused_memory.reconciliation import agent_loop
    from fused_memory.reconciliation.verify import CodebaseVerifier

    captured: dict[str, Any] = {}

    async def _capture(*_args: Any, **kwargs: Any) -> AgentResult:
        captured.update(kwargs)
        raise _ShapeCaptured

    # verify() takes ``codebase_root`` per call and does not read
    # ``config.explore_codebase_root`` (verify.py::CodebaseVerifier __init__,
    # PRD D3), so the root is passed here, not configured.
    verifier = CodebaseVerifier(
        config.model_copy(update={'agent_llm_provider': 'claude_cli'}),
    )
    returned = None
    with patch.object(agent_loop, 'invoke_with_cap_retry', _capture), suppress(_ShapeCaptured):
        returned = asyncio.run(
            verifier.verify(
                claim='The claude_cli AgentLoop runs a verification as one CLI invocation.',
                context='Probe run; the verdict is irrelevant, only the invocation shape matters.',
                scope_hints=['fused-memory/src/fused_memory/reconciliation/agent_loop.py'],
                codebase_root=codebase_root,
            )
        )

    if not captured:
        raise RuntimeError(_capture_failure_message(returned, codebase_root))
    accepted = set(inspect.signature(invoke_claude_agent).parameters) - _PROBE_CONTROLLED_KWARGS
    return Shape(
        name='recon-verify',
        cli_kwargs={k: v for k, v in captured.items() if k in accepted},
        model=str(captured['model']),
        timeout_seconds=float(captured['timeout_seconds']),
        max_turns=int(captured['max_turns']),
    )


def _capture_failure_message(returned: Any, codebase_root: Path) -> str:
    """Say WHY verify() returned without invoking the CLI.

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
        f'returned without invoking the CLI ({summary}). verify.py or '
        'agent_loop.py has changed shape; update _capture_agent_shape rather than '
        'substituting a toy prompt, which measures the wrong thing.'
    )


def _capture_judge_shape(config: ReconciliationConfig) -> Shape:
    from fused_memory.reconciliation.judge import _JUDGE_CLI_MAX_TURNS, JUDGE_VERDICT_SCHEMA
    from fused_memory.reconciliation.prompts.judge import JUDGE_SYSTEM_PROMPT

    return Shape(
        name='judge',
        cli_kwargs={
            'system_prompt': JUDGE_SYSTEM_PROMPT,
            'prompt': (
                'Evaluate this reconciliation run: 3 memories were written, 1 task was '
                'closed, and every claim cited a file path. Produce your verdict.'
            ),
            'output_schema': JUDGE_VERDICT_SCHEMA,
            'disallowed_tools': ['*'],
            'mcp_config': no_mcp_servers_config(),
            'strict_mcp_config': True,
        },
        model=config.judge_llm_model,
        timeout_seconds=float(config.judge_cli_timeout_seconds),
        max_turns=_JUDGE_CLI_MAX_TURNS,
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
    registered = set(shape.cli_kwargs.get('available_tools') or ())
    with tempfile.TemporaryDirectory(prefix='probe_maxturns_') as config_dir:
        started = time.monotonic()
        result = asyncio.run(
            invoke_claude_agent(
                **shape.cli_kwargs,
                model=shape.model,
                max_turns=max_turns,
                timeout_seconds=shape.timeout_seconds,
                oauth_token=token,
                config_dir=Path(config_dir),
                cwd=cwd,
            )
        )
        duration_s = time.monotonic() - started
        # Read INSIDE the TemporaryDirectory: the transcript dies with it.
        # AgentResult.model_id is filled only for a pre-allocated session id,
        # which production does not pass, so the served id is read here too.
        evidence = model_id = None
        if result.session_id:
            evidence = transcript_evidence_for_session(Path(config_dir), result.session_id)
            model_id = result.model_id or transcript_model_id_for_session(
                Path(config_dir), result.session_id,
            )

    unregistered = (
        None if evidence is None
        else sum(1 for name in evidence.other_tool_uses if name not in registered)
    )
    has_payload = isinstance(result.structured_output, dict)
    outcome, detail = _classify(result, has_payload, shape.timeout_seconds)
    return Observation(
        max_turns=max_turns, shape=shape.name, outcome=outcome, success=result.success,
        subtype=result.subtype, num_turns=result.turns, has_payload=has_payload,
        schema_salvaged=result.schema_salvaged, schema_tool_denied=result.schema_tool_denied,
        unregistered_tool_calls=unregistered, model_id=model_id,
        duration_s=duration_s, timeout_s=shape.timeout_seconds, detail=detail,
    )


def _classify(result: AgentResult, has_payload: bool, timeout: float) -> tuple[str, str]:
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
        description='Measure production --json-schema CLI invocations (tasks 3241, 4344).',
    )
    parser.add_argument(
        '--max-turns', type=int, nargs='+', default=None, metavar='N',
        help="Turn caps to probe (default: each shape's PRODUCTION cap).",
    )
    parser.add_argument(
        '--repeat', type=int, default=6, metavar='N',
        help='Runs per cell (default: 6). The failure is STOCHASTIC — one run per '
             'cell is what produced the earlier wrong verdict. Do not set this to 1 '
             'and then quote the result as a rate.',
    )
    parser.add_argument(
        '--shape', choices=['agent', 'judge', 'both'], default='agent',
        help="Which production invocation to probe (default: agent, i.e. recon-verify).",
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
    print('probe_schema_max_turns — MANUAL diagnostic, spends real tokens (tasks 3241, 4344)')
    print(f'claude --version : {_cli_version()}')
    caps = args.max_turns or "each shape's production cap"
    print(f'matrix           : max_turns={caps} x repeat={args.repeat}')
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
        kw = shape.cli_kwargs
        print(f'\nShape {shape.name!r}: model={shape.model} timeout={shape.timeout_seconds:.0f}s '
              f'production max_turns={shape.max_turns} '
              f'available_tools={kw.get("available_tools")} '
              f'disallowed_tools={kw.get("disallowed_tools")} '
              f'permission_mode={kw.get("permission_mode", "(default)")} '
              f'system prompt {len(kw["system_prompt"])} chars, '
              f'schema keys={sorted((kw["output_schema"].get("properties") or {}).keys())}')

    results: dict[tuple[str, int], list[Observation]] = {}
    for shape in shapes:
        for max_turns in args.max_turns or [shape.max_turns]:
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
        unregistered = sum(o.unregistered_tool_calls or 0 for o in reached)
        unread = sum(1 for o in reached if o.unregistered_tool_calls is None)
        turns = sorted(o.num_turns for o in reached)
        turn_range = f'{turns[0]}-{turns[-1]}' if turns else '-'
        models = sorted({o.model_id for o in reached if o.model_id})
        print(f'  {shape_name:<14} max_turns={max_turns:<3} -> {rate:<10} [{breakdown}] '
              f'num_turns={turn_range} '
              f'slowest={slowest.duration_s:.0f}s/{slowest.timeout_s:.0f}s'
              f'{" NEAR-TIMEOUT" if slowest.near_timeout else ""} '
              f'unregistered_tool_calls={unregistered}'
              f'{f" (transcript unread on {unread} run(s))" if unread else ""} '
              f'model_id={",".join(models) or "-"}')
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
