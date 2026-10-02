#!/usr/bin/env python3
"""scripts/legibility/account_pool.py — the fleet's shared multi-account
``UsageGate``, built for a scripts/ process.

ONE JOB: build the gate from the roster (``config/usage-accounts.yaml``) and
the unit's ``.env`` alone — no orchestrator config is loaded or needed — for
the two scripts/ consumers that draw on the fleet's pool:
``legibility/session_runner.py`` (every trickle and census call, run through
the orchestrator's own ``shared.cli_invoke.invoke_with_cap_retry``) and
``sitting/nightly_prepare.py`` (one lease for its headless run). Rotation,
cap detection and account phase are all ``shared.usage_gate``'s; nothing
here decides policy.

The gate NEVER rides the operator's own ``~/.claude`` login (its
default-credential fallback is disabled) and ``build_pool`` NEVER raises: an
empty or unreadable roster is a loud empty pool, which the session runner
turns into an honest deferral (task 6042, Leo's 2026-09-29 ruling).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute -- same reasoning and same form as
# coder.py and census.py (tasks 2881/2882/3329). An editable install puts the
# MAIN checkout's shared/src on sys.path for a bare `python3`, so without this
# a copy running from a worktree would build the MAIN checkout's gate.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from shared.config_models import UsageCapConfig  # noqa: E402
from shared.usage_gate import UsageGate  # noqa: E402

logger = logging.getLogger("legibility.account_pool")

_REPO_ROOT = Path(__file__).resolve().parents[2]


def default_accounts_file() -> Path:
    """The shared account roster, resolved against THIS checkout.

    ``config/usage-accounts.yaml`` is the fleet's single source of truth for
    the pool (the orchestrator and fused-memory reconciliation both point
    ``usage_cap.accounts_file`` at it). Resolved ``__file__``-relatively and
    never as a hardcoded absolute, so a copy of this script running from a
    worktree reads its own roster — same reasoning as coder.py's ``shared``
    bootstrap (tasks 2881/2882/3329). ABSOLUTE by construction, which
    matters: ``UsageCapConfig``'s validator ``.resolve()``s a relative path
    against the CWD, and the trickle's CWD is the systemd unit's.
    """
    return _REPO_ROOT / "config" / "usage-accounts.yaml"


def build_pool(*, accounts_file=None, env_file=None) -> UsageGate:
    """Construct the shared multi-account ``UsageGate`` for this process.

    No orchestrator config is loaded or needed — only the roster file, which
    is the same two-liner ``fused_memory/config/schema.py`` uses. That is
    what makes the pool reachable from this interpreter at all: the
    ORCHESTRATOR YAML is unreachable here, the gate is not.

    ORDER IS LOAD-BEARING. ``load_dotenv`` runs BEFORE the gate is built,
    because ``UsageGate._init_accounts`` reads ``os.environ`` eagerly at
    construction: a ``.env`` loaded afterwards resolves nothing and every
    account comes back token-less. The gate is built with the
    ``~/.claude/.credentials.json`` fallback DISABLED, so that degrades to an
    empty pool rather than to the operator's own login. The path is
    passed EXPLICITLY rather than letting ``load_dotenv()`` search: the bare
    call is frame-relative and silently switches to the CWD under a
    debugger or an interactive interpreter.

    The ``ANTHROPIC_API_KEY`` pop repairs that load: the ``.env`` defines
    the key, so ``load_dotenv`` would put back what the unit's
    ``UnsetEnvironment=`` removed. Policy and the other two strip points:
    ``OPERATIONS.md`` §"Legibility trickle accounts (03:00)".

    NEVER RAISES: a pool that comes back empty, and a roster that cannot be
    read or parsed at all (task 5635), both degrade LOUDLY to an empty pool.
    Copies ``evals/runner.py::_build_eval_usage_gate``'s warn-rather-than-crash
    shape, for a reason specific to this caller: refusing to start would
    take the whole night down, while a warned empty pool still reaches task
    4736's honest DEFERRED path (``session_runner`` raises ``NoHeadroom``
    naming this exact condition). What must never happen is the quiet
    version.
    """
    load_dotenv(env_file if env_file is not None else _REPO_ROOT / ".env")
    os.environ.pop("ANTHROPIC_API_KEY", None)

    resolved = accounts_file or os.environ.get("USAGE_ACCOUNTS_FILE") or str(
        default_accounts_file()
    )
    try:
        gate = UsageGate(_pool_config(accounts_file=str(Path(resolved).resolve())))
    except Exception as exc:  # noqa: BLE001 — a bad roster defers the night, never crashes it
        logger.warning(
            "legibility account pool could not load its roster %s (%s: %s) — "
            "every invocation will defer for want of an account; there is no "
            "~/.claude fallback.",
            resolved, type(exc).__name__, exc,
        )
        return UsageGate(_pool_config())

    # The roster's names, because the gate publishes only a count: the
    # operator's next move on a short pool is to check which
    # CLAUDE_OAUTH_TOKEN_* the unit is missing, and only the roster names that.
    configured = _roster_names(resolved)
    if gate.account_count == 0:
        logger.warning(
            "legibility account pool resolved NO usable accounts from %s "
            "(configured: %s) — every invocation will defer for want of an "
            "account; there is no ~/.claude fallback. Check the unit's "
            "EnvironmentFile supplies those CLAUDE_OAUTH_TOKEN_* vars.",
            resolved, ", ".join(configured) or "<none>",
        )
    else:
        logger.info(
            "legibility account pool: %d of %d configured accounts resolved "
            "(configured: %s)",
            gate.account_count, len(configured), ", ".join(configured),
        )
    return gate


def _pool_config(**roster) -> UsageCapConfig:
    """Never the operator's own ~/.claude login (fallback off), and no resume
    probes: those reopen a pool for PARKED callers, and the legibility runner
    never parks (task 6042)."""
    return UsageCapConfig(
        fallback_to_default_credential=False, wait_for_reset=False, **roster,
    )


def _roster_names(accounts_file) -> list[str]:
    """Account names the roster FILE declares, whether or not their tokens
    resolved. Read straight back off the YAML because the gate keeps no
    record of an account it skipped, and "which account is missing its
    token" is the only question the warning above is asked to answer."""
    try:
        data = yaml.safe_load(Path(accounts_file).read_text()) or {}
        return [entry.get("name", "?") for entry in data.get("accounts", [])]
    except Exception as exc:  # noqa: BLE001 — a warning's detail must never raise
        logger.warning(
            "could not read the configured account names back from roster %s "
            "(%s: %s)", accounts_file, type(exc).__name__, exc,
        )
        return []
