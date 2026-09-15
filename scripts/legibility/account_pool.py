#!/usr/bin/env python3
"""scripts/legibility/account_pool.py — the shared multi-account
``UsageGate``, wired to the legibility ``invoke=`` seam.

ONE JOB: turn the fleet's account pool into the ``(prompt, model) -> str``
callable that ``coder.code_digests(invoke=...)`` already speaks. Nothing
here decides cap policy, parses a roster or tracks account phase — all of
that is ``shared.usage_gate``'s, consumed wholesale rather than
reimplemented (task 5488: the one thing this module must never become is a
second rotation beside the gate's).

WHY THE TRICKLE NEEDED THIS. Before it, the nightly's ``invoke=`` seam
resolved to ``coder._invoke_cli`` with no token, so every one of the
night's 33 one-shots authenticated as whatever login ``~/.claude``
happened to hold. One capped account therefore deferred the whole night
while six live accounts sat idle — and the deferral was indistinguishable
from a genuinely exhausted fleet.

THE NARROW INTERFACE IT CONSUMES, and the only reason it can live outside
``shared``: six gate members, all synchronous — ``try_lease`` (the sync,
non-blocking selection knob added for exactly this caller),
``account_count`` (how the exhaustion reason says WHICH exhaustion), and
``detect_cap_hit`` / ``confirm_account_ok`` / ``on_agent_complete`` /
``release_probe_slot`` (reached through the real ``InvokeSlot``, whose
constructor is a plain ``def``; ``UsageGate.invoke_slot`` is async only
because it awaits ``before_invoke``). No event loop is created or needed.

WHY IT IS ITS OWN FILE rather than code in coder.py or nightly.py.
``coder.py``'s docstring makes its dependency-light character a
load-bearing argument — importing pydantic, asyncio and the gate's
per-account config-dir machinery into it would falsify that, and would
drag gate machinery into every coder test. ``nightly.py`` is already 1608
lines and a second concern there is a file-size problem (heuristic 14).
Here the heavy import is confined to one small module with one purpose
(heuristics 6, 9, 13), which ``census.py`` can reuse when its own
per-invocation rotation lands.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Bind `shared` to the SAME checkout as this script via a __file__-relative
# path, never a hardcoded absolute -- same reasoning and same form as
# coder.py:73-78 and census.py:74-88 (tasks 2881/2882/3329). An editable
# install puts the MAIN checkout's shared/src on sys.path for a bare
# `python3`, so without this a copy running from a worktree would select
# accounts through the MAIN checkout's gate rather than its own.
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# THE PACKAGE SPELLING, never a bare `import coder`, and the difference is
# load-bearing. scripts/legibility/ sits on sys.path alongside scripts/, so
# the two spellings build two DISTINCT module objects carrying two distinct
# `CoderCapExhausted` classes. `pool_invoke` raises that exception and
# `coder.code_digest` catches it by name, under two except arms with no
# generic `except Exception` beneath them -- so a mismatch would not
# mislabel the deferral, it would let the exception escape `run_nightly`
# entirely and crash the very night task 4736 exists to make exit 0. Any
# future consumer of this module (census.py, which today reaches the coder
# by its bare name) must reach it by this same spelling.
from legibility import coder  # noqa: E402
from shared.config_models import UsageCapConfig  # noqa: E402
from shared.usage_gate import InvokeSlot, UsageGate  # noqa: E402

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
    """Construct the shared multi-account ``UsageGate`` for the trickle.

    No orchestrator config is loaded or needed — only the roster file, which
    is the same two-liner ``fused_memory/config/schema.py`` uses. That is
    what makes the pool reachable from this interpreter at all, and it
    falsifies the old claim in ``coder.py``'s docstring that a multi-account
    gate is unreachable here: the ORCHESTRATOR YAML is unreachable, the gate
    is not.

    ORDER IS LOAD-BEARING. ``load_dotenv`` runs BEFORE the gate is built,
    because ``UsageGate._init_accounts`` reads ``os.environ`` eagerly at
    construction: a ``.env`` loaded afterwards resolves nothing, every
    account comes back token-less, and the gate silently falls back to
    ``~/.claude/.credentials.json`` as an account named 'default' — which is
    exactly the broken behaviour this module exists to end. The path is
    passed EXPLICITLY rather than letting ``load_dotenv()`` search: the bare
    call is frame-relative and silently switches to the CWD under a
    debugger or an interactive interpreter.

    Degrades LOUDLY, never raises, when the pool comes back empty. Copies
    ``evals/runner.py::_build_eval_usage_gate``'s warn-rather-than-crash
    shape, for a reason specific to this caller: refusing to start would
    take the whole night down, while a warned empty pool still reaches task
    4736's honest DEFERRED path (``pool_invoke`` raises
    ``CoderCapExhausted`` naming this exact condition). What must never
    happen is the quiet version.
    """
    load_dotenv(env_file if env_file is not None else _REPO_ROOT / ".env")

    resolved = accounts_file or os.environ.get("USAGE_ACCOUNTS_FILE") or str(
        default_accounts_file()
    )
    gate = UsageGate(UsageCapConfig(accounts_file=str(Path(resolved).resolve())))

    names = [acct.name for acct in gate._accounts]
    if gate.account_count == 0 or names == ["default"]:
        # `default` is the name _init_accounts gives the ~/.claude fallback,
        # so it is indistinguishable from the pre-5488 behaviour and equally
        # useless to fail over with. Name the roster it FAILED to resolve --
        # the operator's next move is to check which CLAUDE_OAUTH_TOKEN_* the
        # unit is missing, and only the roster names that.
        configured = _roster_names(resolved)
        logger.warning(
            "legibility account pool resolved NO usable accounts from %s "
            "(configured: %s) — every invocation would fall back to the "
            "ambient ~/.claude login. Check the unit's EnvironmentFile "
            "supplies those CLAUDE_OAUTH_TOKEN_* vars.",
            resolved, ", ".join(configured) or "<none>",
        )
    else:
        logger.info(
            "legibility account pool: %d accounts — %s", len(names), ", ".join(names),
        )
    return gate


def _roster_names(accounts_file) -> list[str]:
    """Account names the roster FILE declares, whether or not their tokens
    resolved. Read straight back off the YAML because the gate keeps no
    record of an account it skipped, and "which account is missing its
    token" is the only question the warning above is asked to answer."""
    try:
        data = yaml.safe_load(Path(accounts_file).read_text()) or {}
        return [entry.get("name", "?") for entry in data.get("accounts", [])]
    except OSError:
        return []


_DEFAULT_INVOKE = coder._invoke_cli
"""The real subprocess boundary this pool hands tokens to.

Captured as a module constant rather than spelled as a default argument so
the wiring is assertable without calling it -- a test that had to INVOKE
the default to discover it would be spawning a real `claude`."""


_EXHAUSTED_MARKER = "pool exhausted"
"""Marker carried by the pool's own CoderCapExhausted.

``CoderCapExhausted.marker`` names the signal that fired, so a deferral
reason can quote WHICH one. A per-digest cap quotes the banner phrase the
CLI printed; this one is not a banner at all -- it is the gate reporting
that no account remains -- and saying so is the honest spelling."""


def _exhaustion_reason(gate) -> str:
    """Say WHICH exhaustion this is, because the two need different
    operator responses.

    "all N pool accounts capped" self-clears at the weekly reset and is
    expected weather (task 4503). "no pool accounts resolved" is a config
    fault that will never clear on its own -- it is the state
    ``UsageGate._init_accounts`` degrades to when no token env var resolves,
    which is also the state whose silent fallback to ``~/.claude`` this task
    exists to remove. Folding the second into the first would send an
    operator to wait for a reset that never comes.
    """
    count = gate.account_count
    if not count:
        return (
            "no pool accounts resolved — check that the unit's EnvironmentFile "
            "supplies the CLAUDE_OAUTH_TOKEN_* vars named in "
            "config/usage-accounts.yaml"
        )
    return f"all {count} pool accounts capped"


def pool_invoke(gate, *, reverse: bool = True, invoke=_DEFAULT_INVOKE):
    """Return a ``(prompt, model) -> str`` callable that runs each
    invocation as an account leased from *gate*.

    The returned closure is what goes through the legibility ``invoke=``
    seam, so ``coder``'s control flow — the never-fabricate contract, the
    storm threshold, the taint-and-exclude rule, task 4736's whole deferral
    chain — is inherited unchanged. The only new input it ever produces is a
    ``CoderCapExhausted`` that now means the POOL is exhausted rather than
    one login.

    *reverse* defaults to True: the trickle drains the roster h→b so its
    one-shots do not contend with the orchestrator's b→h first-available
    order. The two meet only when the pool is nearly exhausted, which is
    when contention is unavoidable anyway.

    FAILOVER HAPPENS WITHIN A DIGEST, not across digests. A blocking banner
    marks the account capped and the SAME prompt is retried on the next
    account; ``CoderCapExhausted`` escapes only when the whole pool is
    exhausted. That is a reading of the existing code rather than a
    preference: ``CoderCapExhausted`` already means "there is no headroom
    left to code this digest", and both ``coder.is_cap_deferral`` and
    nightly's DEFERRED summary read ``capped`` as "the CLI never looked at
    this digest". If one account's banner set ``capped=True``, that
    predicate would silently weaken to "the account I happened to draw was
    out" — a night with six live accounts could then trip the majority rule
    and read as DEFERRED, making the deferral branch a place real failures
    hide. Rotating here instead also stops burning one digest per burned
    account. The payoff: nightly's long-standing "all accounts capped"
    summary becomes TRUE for the first time.

    TERMINATION IS STRUCTURAL, not a retry budget: every iteration marks
    exactly one account capped, so the admissible set strictly shrinks and
    ``try_lease`` returns None after at most ``account_count`` passes.

    ONLY THE GATE'S STRICT DETECTOR ROTATES. ``coder``'s loose
    OR-substring matcher keeps its own job — labelling an already-FAILED
    invocation as a per-digest defer — while ``slot.detect_cap_hit`` (prefix
    AND confirm) decides whether an ACCOUNT is out. When it disagrees the
    original exception propagates unrotated, so a loose false positive can
    re-label one digest and can never burn the pool. That is exactly the
    split ``shared/src/shared/cap_markers.py``'s docstring argues for, and
    it matters here because this repo's codebook is dominated by clusters
    ABOUT usage limits, so healthy model output quotes banner text.

    LEASE DISCIPLINE mirrors ``UsageGate.invoke_slot``'s, because a leaked
    PROBE_IN_FLIGHT claim is permanent: the account is never admissible
    again for this process, so the pool would silently shrink by one
    account per probe. The ``finally`` hands the claim back on every exit
    path, including an exception raised by the CLI.

    It calls ``release_probe_slot`` UNCONDITIONALLY, which is the idiom
    ``InvokeSlot.report`` documents for itself: the call is a guarded no-op
    whenever there is no claim outstanding (``confirm`` and a True
    ``detect_cap_hit`` each already released it), "which is cheaper than
    re-deriving what the handler just did". The alternative — reading
    ``slot._settled`` the way ``invoke_slot`` does — would reach into
    another module's private state from outside it to recompute an answer
    the gate already guards.
    """

    def invoke_through_pool(prompt: str, model: str) -> str:
        while True:
            lease = gate.try_lease(reverse=reverse)
            if lease is None:
                raise coder.CoderCapExhausted(
                    f"legibility trickle: {_exhaustion_reason(gate)}",
                    marker=_EXHAUSTED_MARKER,
                )
            slot = InvokeSlot(gate, lease)
            try:
                reply = invoke(prompt, model, oauth_token=slot.token)
                slot.confirm()
                return reply
            except coder.CoderCapExhausted as exc:
                # The loose per-digest gate fired. Ask the STRICT detector
                # whether this ACCOUNT is out; it marks the account capped and
                # settles the slot when it agrees.
                if not slot.detect_cap_hit(exc.stderr, exc.stdout):
                    raise
                logger.info(
                    "account %s is capped — retrying this digest on the next "
                    "account in the pool", slot.account_name,
                )
            finally:
                gate.release_probe_slot(slot.token)

    return invoke_through_pool
