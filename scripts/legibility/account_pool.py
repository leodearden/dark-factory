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
``shared``: seven gate members, all synchronous — ``try_lease`` (the sync,
non-blocking selection knob added for exactly this caller),
``account_count`` and ``active_account_name`` (how the exhaustion reason
says WHICH exhaustion), and
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

    The ``ANTHROPIC_API_KEY`` pop repairs that load: the ``.env`` defines
    the key, so ``load_dotenv`` would put back what the unit's
    ``UnsetEnvironment=`` removed. Policy and the other two strip points:
    ``OPERATIONS.md`` §"Legibility trickle accounts (03:00)".

    Degrades LOUDLY, never raises, when the pool comes back empty. Copies
    ``evals/runner.py::_build_eval_usage_gate``'s warn-rather-than-crash
    shape, for a reason specific to this caller: refusing to start would
    take the whole night down, while a warned empty pool still reaches task
    4736's honest DEFERRED path (``pool_invoke`` raises
    ``CoderCapExhausted`` naming this exact condition). What must never
    happen is the quiet version.
    """
    load_dotenv(env_file if env_file is not None else _REPO_ROOT / ".env")
    os.environ.pop("ANTHROPIC_API_KEY", None)

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


def _exhaustion_reason(gate, tried) -> str:
    """Say WHICH exhaustion this is, because the three need different
    operator responses.

    "all N pool accounts capped" self-clears at the weekly reset and is
    expected weather (task 4503). "no pool accounts resolved" is a config
    fault that will never clear on its own -- it is the state
    ``UsageGate._init_accounts`` degrades to when no token env var resolves,
    which is also the state whose silent fallback to ``~/.claude`` this task
    exists to remove. Folding either into the other would send an operator
    to wait for a reset that never comes.

    THE THIRD STATE exists only because termination is enforced by the
    caller's exclusion set rather than by the gate's cap transitions: every
    account refused this digest while the gate still considers one usable.
    Nothing is capped, so "all N pool accounts capped" would be false — and
    false in the costly direction, since it names a weekly reset that will
    never arrive because there is nothing to reset. The account the gate
    still calls usable is named, because that fact is what makes "capped"
    the wrong word.

    WHICH state it is, is read off the gate's PUBLIC predicates.
    ``active_account_name`` is already the gate's answer to "is any account
    still usable" (None iff no non-capped, non-auth-failed account remains);
    recomputing it by walking ``gate._accounts`` would reach past a
    published answer into another module's internals to derive what it
    already says.
    """
    count = gate.account_count
    if not count:
        return (
            "no pool accounts resolved — check that the unit's EnvironmentFile "
            "supplies the CLAUDE_OAUTH_TOKEN_* vars named in "
            "config/usage-accounts.yaml"
        )
    live = gate.active_account_name
    if live is None:
        return f"all {count} pool accounts capped"
    return (
        f"no account in the pool completed this digest ({len(tried)} of "
        f"{count} tried) and the gate still considers {live} usable — so "
        f"this is not a capacity limit and will not clear at the weekly "
        f"reset; the run's per-digest failures say what each account "
        f"reported"
    )


_ROUTE_NONZERO_EXIT = "the CLI exited non-zero with a banner the gate confirmed"
_ROUTE_ZERO_EXIT = "the CLI exited 0 with a banner instead of a verdict"
"""The two ways the CLI declines, named so the journal can tell them apart.

Structured constants rather than a phrase assembled at each call site
(heuristic 12): "which route rotated" is a two-valued fact about the run, and
the only reason it is ever rendered as text is that a journal line is the
medium. The task record's 2026-09-19 measurement could count the non-zero
route (46 firings on one night) but could not establish whether the exit-0
route had ever fired, because both arms said the same thing — so these
spellings are what make the route greppable.
"""


def _log_rotation(slot, route) -> None:
    """Record that *slot*'s account did not complete this digest, naming
    *route*.

    ONE helper called from both arms, not two ``logger.info`` calls, so the
    shared half of the wording cannot drift between the routes (heuristic 11)
    while the route clause remains the only thing that differs. The sentence
    itself is unchanged from the single route-blind line this replaces, so an
    operator's existing greps and the 2026-09-19 measurement's baseline keep
    working.

    INFO, because a rotation is normal operating weather — the pool exists to
    absorb exactly this — and that is how the line it replaces already treated
    it. The escalation-worthy event is the pool running OUT, which
    ``_exhaustion_reason`` reports on a raised ``CoderCapExhausted``.
    """
    logger.info(
        "account %s did not complete this digest and the gate recorded a cap "
        "signal against it (route: %s) — retrying this digest on the next "
        "account in the pool", slot.account_name, route,
    )


def _banner_instead_of_verdict(slot, reply) -> bool:
    """Is this exit-0 *reply* a cap banner rather than a verdict?

    Census's split-on-parse-success rule, adopted unchanged: a reply that
    PARSES into a verdict IS a verdict, so only a reply the coder could not
    use at all is offered to the gate. The rule is already cited at both of
    ``coder``'s scan sites (``_invoke_cli`` and ``code_digest``); this module
    inherits it rather than inventing a third policy.

    HERE IT IS LOAD-BEARING, NOT STYLISTIC, and that is the difference from
    the coder's two sites — both of which sit on already-FAILED paths where a
    false positive can only re-label a digest that was failing anyway. This
    one sits on a SUCCEEDING path, where a false positive costs an account.
    The gate's strict detector matches a cap-hit prefix and a confirm keyword
    ANYWHERE in the text, not anchored at the start: measured 2026-09-20, a
    schema-valid verdict quoting ``REAL_CLI_CAP_HIT_MESSAGES[0]`` in its
    ``evidence_quote`` classifies as CapHit. Since this repo's codebook is
    dominated by usage-limit clusters, an unguarded scan would let one
    cap-themed digest cap every account the rotation walked — for the night.

    THE GATE STILL DECIDES WHETHER THIS IS A CAP. The parse decides only
    whether to ASK, never what the answer is, so ``pool_invoke``'s "do not
    re-classify streams here to decide anything the gate decides" holds: a
    parse attempt answers "could the coder use this reply at all", the same
    question ``code_digest`` asks one layer down.

    AND THE GATE COSTS NO COVERAGE, because a real banner never parses: every
    entry of ``shared.cap_markers.REAL_CLI_CAP_MESSAGES`` — the union, so both
    the cap-hit and the near-cap wordings — raises ``CoderParseError``
    (measured 2026-09-20, and kept true by
    ``scripts/tests/test_legibility_account_pool.py::test_a_zero_exit_banner_rotates_and_the_same_digest_completes_next_door``,
    which parametrizes over that same union: a future corpus entry that
    happened to parse turns it red rather than silently losing this route).

    KNOWN RESIDUAL, pinned rather than closed. "A real banner never parses"
    holds in one direction only, and the converse does not: a reply that fails
    to parse AND quotes a banner is read here as a banner. So a verdict
    TRUNCATED mid-JSON after a cap-quoting ``evidence_quote`` caps a healthy
    account — measured 2026-09-20, the schema-shaped prefix ``{"matches":
    [{"cluster_id": "usage-limit-stall", "evidence_quote": "<banner>"`` with no
    closing brace raises ``CoderParseError`` and classifies CapHit. The cost is
    one account per occurrence, not the pool: the digest still completes on the
    next lease, and the rotation stays bounded by ``tried``. It is left open
    DELIBERATELY, because both remedies would give this module a cap policy of
    its own — a second confirmation probe, or a "this looks like truncated
    JSON" discriminator — and the paragraphs above are the argument that it has
    none. Pinned by ``test_a_truncated_verdict_that_quotes_a_banner_costs_that_account``
    so the disposition is a decision a reader can find, not an oversight.
    """
    try:
        coder.parse_coder_output(reply)
    except coder.CoderParseError:
        return slot.detect_cap_hit("", reply)
    return False


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

    FAILOVER HAPPENS WITHIN A DIGEST. When the CLI fails with a cap banner
    and the gate's strict ``slot.detect_cap_hit`` (prefix AND confirm)
    agrees, the SAME prompt is retried on the next account;
    ``CoderCapExhausted`` escapes only when no account is left, carrying
    ``_exhaustion_reason``'s account of which exhaustion it was. Never
    mark a digest ``capped`` for one account's banner: ``capped`` means "no
    headroom left anywhere", and weakening it would let a night with live
    accounts trip nightly's majority rule and read as DEFERRED. When the
    strict detector disagrees with ``coder``'s loose matcher, the original
    exception propagates unrotated — this repo's codebook quotes banner
    text, so a loose false positive may re-label one digest but must never
    burn the pool.

    BOTH CAP ROUTES ROTATE, and they are the same rotation (task 5637). The
    CLI declines two ways: a non-zero exit whose banner the gate confirms
    (above), and an exit-0 reply that is a banner rather than a verdict. The
    second arm asks the same strict detector — ``slot.detect_cap_hit("",
    reply)``, the reply as the OUTPUT stream and no stderr — takes the same
    next lease, and is bounded the same way. Which route the CLI takes is
    the CLI's choice and not the pool's, so rotating on one and confirming
    on the other left one bannering account able to lose a whole night.
    ``_log_rotation`` names which of the two fired, so the journal can count
    them separately.

    TERMINATION is bounded by the caller's ``tried`` set, passed as
    ``exclude=``, not by the gate's cap transitions: a near-cap verdict
    only annotates an account, so re-asking the gate could return it
    forever. ``tried`` grows by one per pass, so ``try_lease`` returns None
    within ``account_count`` passes. Do not re-classify streams here to
    decide anything the gate decides (heuristic 11).

    The ``finally`` calls ``release_probe_slot`` on every exit path,
    unconditionally (a guarded no-op when ``confirm`` or ``detect_cap_hit``
    already released it): a leaked PROBE_IN_FLIGHT claim makes that account
    inadmissible for the rest of the process.
    """

    def invoke_through_pool(prompt: str, model: str) -> str:
        # Scoped to ONE digest, deliberately: an account that refused this
        # prompt is not thereby done for the night — a near-cap warning is
        # not a cap — so the next digest starts from the full roster again.
        tried: set[str] = set()
        while True:
            lease = gate.try_lease(reverse=reverse, exclude=tried)
            if lease is None:
                raise coder.CoderCapExhausted(
                    f"legibility trickle: {_exhaustion_reason(gate, tried)}",
                    marker=_EXHAUSTED_MARKER,
                )
            tried.add(lease.name)
            slot = InvokeSlot(gate, lease)
            try:
                reply = invoke(prompt, model, oauth_token=slot.token)
            except coder.CoderCapExhausted as exc:
                # The loose per-digest gate fired. Ask the STRICT detector
                # what the GATE makes of it; on any True verdict it settles
                # the slot and releases the probe claim. WHICH transition it
                # took is the gate's business and not knowable from here — a
                # CapHit caps the account, a NearCap only annotates it — which
                # is exactly why the retry is bounded by `tried` rather than
                # by an assumption about what just happened.
                if not slot.detect_cap_hit(exc.stderr, exc.stdout):
                    raise
                _log_rotation(slot, _ROUTE_NONZERO_EXIT)
            else:
                # THE OTHER CAP ROUTE. The CLI can decline by PRINTING its
                # banner and exiting 0, so a cap arrives as a RETURNED reply
                # as readily as a raised one. The SAME strict detector
                # decides, reached through `_banner_instead_of_verdict` so
                # that only a reply the coder cannot parse is put to it. On a
                # True verdict, do NOT confirm (which would clear the gate's
                # own near-cap annotation on an account that just refused to
                # answer) and do NOT return: fall through to the next lease
                # exactly as the arm above does, bounded by the same `tried`.
                if not _banner_instead_of_verdict(slot, reply):
                    slot.confirm()
                    return reply
                _log_rotation(slot, _ROUTE_ZERO_EXIT)
            finally:
                gate.release_probe_slot(slot.token)

    return invoke_through_pool


def subprocess_env(gate, *, reverse = True):
    """The env a CHILD PROCESS needs to run as an account from *gate*, or
    ``None`` to inherit the parent's unchanged.

    The trickle spawns one child it does not drive through the gate: the
    periodic census (``nightly._default_census_launcher`` ->
    ``subprocess.run(census.py)``). That grandchild carried no env of its own,
    so it was authenticated only because a 2026-09-14 stopgap drop-in exported
    ONE account's token into the systemd unit. This is what replaces that pin,
    and it is strictly better than what it replaces: an account the gate
    believes is live AT LAUNCH TIME, rather than a hardcoded max-h that may
    have capped hours earlier.

    THE LEASE IS READ AND HANDED STRAIGHT BACK. Nothing in another process can
    settle a slot, so a retained PROBE_IN_FLIGHT claim would never be released
    and that account would be inadmissible to the trickle's own digests for the
    rest of the night -- the pool silently shrinking by one every time a census
    fired. The cost of handing it back is that the census's spend is invisible
    to the gate; per-invocation rotation INSIDE census.py (its own file lock, a
    mining loop over many batches) is the real fix and is filed as a follow-up.

    DEGRADES TO ``None``, never raises. ``census.preflight_headroom`` folds any
    failure into a fail-SAFE defer, so a census that cannot authenticate
    silently skips instead of erroring: "no account to give it" must therefore
    mean "inherit exactly as before", which is what the census did before this
    existed, and never "fail the census".
    """
    lease = gate.try_lease(reverse=reverse)
    if lease is None:
        logger.info(
            "legibility account pool: no account available for the census "
            "subprocess — it inherits the unit's environment, as before",
        )
        return None
    try:
        return coder.child_env(lease.token)
    finally:
        gate.release_probe_slot(lease.token)
