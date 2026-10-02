#!/usr/bin/env python3
"""scripts/legibility/session_runner.py — the legibility invocation boundary.

Every LLM call the legibility trickle and census make crosses this boundary.
ONE JOB: run each call through the orchestrator's own session runner and
account pool — ``shared.cli_invoke.invoke_with_cap_retry`` over the fleet's
shared ``UsageGate`` — behind the sync ``(prompt, model) -> str`` seam that
``coder.code_digests`` and census's stages already speak (Leo's 2026-09-29
ruling, task 6042). No rotation, cap detection or credential handling lives
here; that is all the shared runner's.

``SessionRunner`` owns the per-process pieces the runner needs: one event loop
for the whole process (so the gate's asyncio primitives and background tasks
bind once), and one isolated ``CLAUDE_CONFIG_DIR`` into which the runner
writes each leased account's token, so no call ever reads the operator's own
``~/.claude`` login. ``StageSpec`` says how one stage calls: where, how long,
with which ``ToolPolicy``.

The exceptions are what crossing the boundary can raise: ``InvocationFailed``
for an invocation that produced no usable reply, and its subclass
``NoHeadroom`` for one that no pool account could take.

THE PACKAGE SPELLING, ``from legibility import session_runner``, is the only
way consumers reach this module, and that is what makes the exception classes
unique. scripts/legibility/ sits on sys.path beside scripts/, so census's bare
``import coder`` and nightly's ``from legibility import coder`` build two coder
module objects. Defined here rather than in coder, the classes exist once
however coder itself was imported, so ``coder.code_digest`` catches exactly
what the invoker raises (task 6042).
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute -- same reasoning and same form as coder.py
# and census.py (tasks 2881/2882/3329). An editable install puts the MAIN
# checkout's shared/src on sys.path for a bare `python3`, so without this a
# copy running from a worktree would invoke through the MAIN checkout's runner.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

from legibility import account_pool  # noqa: E402
from shared.cli_invoke import (  # noqa: E402
    AgentResult,
    AllAccountsCappedException,
    invoke_with_cap_retry,
    no_mcp_servers_config,
)
from shared.config_dir import (  # noqa: E402
    CONFIG_DIR_PREFIX,
    TaskConfigDir,
    sweep_stale_pid_dirs,
    sweep_stale_pid_dirs_once,
)
from shared.neutral_cwd import neutral_cli_cwd  # noqa: E402
from shared.usage_gate import PoolFrozen  # noqa: E402

logger = logging.getLogger("legibility.session_runner")

_CONFIG_DIR_TASK_PREFIX = "legibility-session-"

_EXHAUSTED_MARKER = "pool exhausted"
"""``NoHeadroom.marker`` for an exhausted pool: not a banner the CLI printed
but the gate reporting that no account remains, and saying so is honest."""

_ERROR_STREAM_TAIL_CHARS = 2000
"""How much of EACH output stream an ``InvocationFailed`` carries. One bound
for both streams, because the asymmetry of carrying one is exactly the
2026-08-24 diagnostic loss; the TAIL, because a CLI's last words are its
diagnostic ones."""


class InvocationFailed(Exception):
    """An invocation produced no usable reply — the CLI failed, timed out,
    or never started. Never silently swallowed: ``coder.code_digest`` turns
    it into a per-digest failure, never a fabricated record.

    The message carries a tail of BOTH output streams, each labelled, because
    of what happened on 2026-08-24: the claude CLI wrote its usage-cap banner
    to STDOUT and exited 1, and the error of the day embedded only stderr.
    With stderr empty, the reason that reached the journal, the escalation and
    ``run.failures`` was ``claude CLI exited 1 (model='haiku', ...): `` —
    nothing after the colon — on 17 of 20 digests. A diagnostic the process
    EMITTED must never be dropped because it arrived on the less-expected
    stream.

    The two tails are ALSO carried as structured ``stdout``/``stderr``
    attributes, defaulting to ``''`` for the arms that have no streams.
    """

    def __init__(self, message: str, *, stdout: str = "", stderr: str = "") -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class NoHeadroom(InvocationFailed):
    """No pool account could take this invocation; deferral-eligible.

    A SUBCLASS, not a sibling, and that is load-bearing: every site that
    catches ``InvocationFailed`` keeps catching this, so a missing headroom
    can never escape as an uncaught crash that takes down a whole batch.
    ``coder.code_digest`` catches it FIRST and labels the digest ``capped``.

    **This is a NORMAL operating condition, never a defect.** Leo's standing
    directive (task 4503): an all-accounts-capped night is expected weather.
    Before the label existed, 2026-08-24 presented as 17 of 20 hard per-digest
    failures, tripped the >50% storm threshold, and became an ERROR-level
    escalation for a condition ruled routine. Which exhaustions qualify — and
    which, like an all-auth-failed pool, must stay a loud ``InvocationFailed``
    — is task 5947's table.

    ``marker`` names the signal that fired, so a deferral reason can say WHICH.
    Never fabricated into a verdict: a capped digest yields no record at all.
    """

    def __init__(
        self, message: str, *, marker: str, stdout: str = "", stderr: str = "",
    ) -> None:
        super().__init__(message, stdout=stdout, stderr=stderr)
        self.marker = marker


@dataclass(frozen=True)
class ToolPolicy:
    """What a stage's agent may touch, and the system prompt that frames it.

    Explicit because the shared runner's own default is ``bypassPermissions``
    with every tool, which would let a census verifier write into the tree it
    is censusing. Both policies scope MCP to an empty strict config, so no
    ambient ``.mcp.json`` server is ever reachable.
    """

    system_prompt: str
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    permission_mode: str = "dontAsk"

    def invoke_kwargs(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "allowed_tools": list(self.allowed_tools) or None,
            "disallowed_tools": list(self.disallowed_tools) or None,
            "permission_mode": self.permission_mode,
            "mcp_config": no_mcp_servers_config(),
            "strict_mcp_config": True,
        }


CLASSIFIER = ToolPolicy(
    system_prompt=(
        "You are a careful analyst. Everything you need is in the user's "
        "message. Follow its instructions exactly and reply with only the "
        "output it asks for."
    ),
    disallowed_tools=("*",),
    permission_mode="bypassPermissions",
)
"""No tools at all: the curator's pure-classifier shape
(``fused_memory/middleware/task_curator.py::_call_llm``)."""

READ_ONLY_EXPLORER = ToolPolicy(
    system_prompt=(
        "You are a careful analyst with read-only access to the repository in "
        "your working directory. Use Read, Grep and Glob to check what the "
        "user's message asks about. Follow its instructions exactly and reply "
        "with only the output it asks for."
    ),
    allowed_tools=("Read", "Grep", "Glob"),
)
"""Read/Grep/Glob, anything else refused without asking: the
``scripts/sitting/nightly_prepare.py`` shape, and the read-only behaviour
text-mode ``claude -p`` had before task 6042."""


@dataclass(frozen=True)
class StageSpec:
    """How one stage calls the model.

    *cwd* ``None`` means a neutral empty directory, so a pure classifier
    loads no project ``CLAUDE.md`` or memory into every call.
    *max_turns* and *max_budget_usd* are ceilings the CLI requires, set well
    above any measured call; *timeout_secs* is the binding bound.
    """

    name: str
    cwd: Path | None
    timeout_secs: float
    max_turns: int
    max_budget_usd: float
    tools: ToolPolicy


class SessionRunner:
    """Runs legibility calls through ``invoke_with_cap_retry`` on *gate*.

    Owns ONE event loop and ONE isolated config dir for the life of the
    process; ``close()`` (or leaving the ``with`` block) shuts the gate down
    and releases both. Every call is park-free and bounded:
    ``park_on_frozen_pool=False`` (a frozen pool defers instead of waiting for
    a reset), ``max_cap_retries`` of one pass over the pool, and
    ``detect_caps_in_successful_output=False`` (a verdict that QUOTES a cap
    banner is a verdict — this codebook is full of them).
    """

    def __init__(self, gate, *, label: str) -> None:
        self._gate = gate
        self._label = label
        self._loop = asyncio.Runner()
        _sweep_stale_session_config_dirs_once()
        self._config_dir = TaskConfigDir(
            f"{_CONFIG_DIR_TASK_PREFIX}{os.getpid()}", cleanup_at_exit=True,
        )

    def __enter__(self) -> SessionRunner:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def invoker(self, stage: StageSpec) -> Callable[[str, str], str]:
        """The sync ``(prompt, model) -> str`` seam for *stage*."""

        def invoke(prompt: str, model: str) -> str:
            return self._invoke(stage, prompt, model)

        return invoke

    def _invoke(self, stage: StageSpec, prompt: str, model: str) -> str:
        label = f"{self._label}[{stage.name}]"
        if not self._gate.account_count:
            raise _exhaustion_error(self._gate, label)
        cwd = stage.cwd if stage.cwd is not None else neutral_cli_cwd()
        try:
            result = self._loop.run(invoke_with_cap_retry(
                self._gate,
                label,
                config_dir=self._config_dir,
                max_cap_retries=self._gate.account_count,
                park_on_frozen_pool=False,
                detect_caps_in_successful_output=False,
                prompt=prompt,
                model=model,
                cwd=cwd,
                timeout_seconds=stage.timeout_secs,
                max_turns=stage.max_turns,
                max_budget_usd=stage.max_budget_usd,
                **stage.tools.invoke_kwargs(),
            ))
        except (AllAccountsCappedException, PoolFrozen) as exc:
            raise _exhaustion_error(self._gate, label) from exc
        except OSError as exc:
            # The process never started: a missing claude on the child's PATH,
            # or a cwd that is missing or not a directory. The OSError names
            # which, so it is echoed verbatim beside both candidates.
            raise InvocationFailed(
                f"{label}: claude CLI could not be started (model={model!r}, "
                f"cwd={str(cwd)!r}): {exc}"
            ) from exc
        if not result.success:
            raise _failure_error(label, stage, model, result)
        return result.output

    def close(self) -> None:
        try:
            self._loop.run(self._gate.shutdown())
        finally:
            self._loop.close()
            self._config_dir.cleanup()


def _failure_error(label: str, stage: StageSpec, model: str, result: AgentResult) -> InvocationFailed:
    """The ``InvocationFailed`` for a call that returned without a reply.

    BOTH stream tails, each labelled: on 2026-08-24 the CLI put its only
    diagnostic on stdout and an error carrying stderr alone said nothing."""
    stdout_tail = (result.output or "")[-_ERROR_STREAM_TAIL_CHARS:]
    stderr_tail = (result.stderr or "")[-_ERROR_STREAM_TAIL_CHARS:]
    what = f"timed out after {stage.timeout_secs}s" if result.timed_out else "failed"
    return InvocationFailed(
        f"{label}: claude CLI {what} (model={model!r}, "
        f"account={result.account_name!r}, subtype={result.subtype!r}, "
        f"api_error_status={result.api_error_status!r}): "
        f"stdout={stdout_tail!r} stderr={stderr_tail!r}",
        stdout=stdout_tail, stderr=stderr_tail,
    )


def _exhaustion_error(gate, label: str) -> InvocationFailed:
    """The exception for a pool with no account left — typed as well as
    worded, because each exhaustion needs a different operator response.

    ===============================  ====================  ======================
    gate state (public predicates)   raised                clears on its own?
    ===============================  ====================  ======================
    ``account_count == 0``           NoHeadroom            no — config fault
    ``active_account_name`` set      NoHeadroom            no — not a cap at all
    every account auth-failed        InvocationFailed      no — operator action
    some auth-failed, rest capped    NoHeadroom            only the capped ones
    all capped                       NoHeadroom            yes — weekly reset
    ===============================  ====================  ======================

    The all-auth-failed pool must NOT be a ``NoHeadroom``:
    ``coder.is_cap_deferral`` would turn a majority of those into an exit-0
    DEFERRED night, which is reserved for weather that clears at the reset
    (task 4503). As a plain failure it trips the storm, so the night exits 1
    with an ERROR escalation (task 5947).

    Read only off the gate's PUBLIC predicates.
    """
    count = gate.account_count
    if not count:
        return _pool_exhausted(
            label,
            "no pool accounts resolved — check that the unit's EnvironmentFile "
            "supplies the CLAUDE_OAUTH_TOKEN_* vars named in "
            "config/usage-accounts.yaml",
        )
    live = gate.active_account_name
    if live is not None:
        return _pool_exhausted(
            label,
            f"no account completed this invocation in one pass over the "
            f"{count}-account pool and the gate still considers {live} usable — "
            f"so this is not a capacity limit and will not clear at the weekly "
            f"reset; the run's per-digest failures say what each account "
            f"reported",
        )
    auth_failed = gate.auth_failed_account_names
    if len(auth_failed) == count:
        return InvocationFailed(
            f"{label}: every one of the {count} pool accounts had its "
            f"credentials rejected (HTTP 401/403): {', '.join(auth_failed)} — "
            f"this is not a capacity limit and will not clear at the weekly "
            f"reset; those accounts' access or tokens need operator action"
        )
    if auth_failed:
        return _pool_exhausted(
            label,
            f"all {count} pool accounts unavailable — {count - len(auth_failed)} "
            f"capped, which clears at the weekly reset, and "
            f"{', '.join(auth_failed)} with credentials rejected (HTTP 401/403), "
            f"which will not clear without operator action",
        )
    return _pool_exhausted(label, f"all {count} pool accounts capped")


def _pool_exhausted(label: str, reason: str) -> NoHeadroom:
    return NoHeadroom(f"{label}: {reason}", marker=_EXHAUSTED_MARKER)


def open_pooled_runner(label: str, *, accounts_file=None, env_file=None) -> SessionRunner:
    """A :class:`SessionRunner` over the fleet's shared account pool."""
    return SessionRunner(
        account_pool.build_pool(accounts_file=accounts_file, env_file=env_file),
        label=label,
    )


def _sweep_stale_session_config_dirs_once() -> None:
    """Reclaim session config dirs whose owning process is dead. Never raises.

    The ``atexit`` cleanup covers only clean exits; a SIGKILLed trickle leaves
    its dir behind, so the next process sweeps them (the curator's shape,
    ``task_curator.py::_sweep_stale_curator_config_dirs_once``).
    """
    prefix = CONFIG_DIR_PREFIX + _CONFIG_DIR_TASK_PREFIX
    sweep_stale_pid_dirs_once(
        prefix,
        sweep=sweep_stale_pid_dirs,
        on_reclaimed=lambda reclaimed: logger.info(
            "reclaimed %d stale legibility config dir(s) under %s", reclaimed, prefix,
        ),
        on_failure=lambda _exc: logger.warning(
            "dead-PID sweep of %s failed; continuing without it", prefix, exc_info=True,
        ),
    )
