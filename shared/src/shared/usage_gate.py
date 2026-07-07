"""Usage cap detection, pause gate, and auto-resume for OAuth-based Claude agents.

Supports multiple Claude Max accounts for failover: when one account hits its
usage cap, the gate returns the next available account's token. Only blocks
when *all* accounts are capped.

Cap detection is reactive (stderr pattern matching). Resume is timer-based:
when an account is capped, we sleep until the parsed ``resets_at`` time, then
uncap. If the uncap is premature, the retry loop in ``invoke_with_cap_retry``
re-detects the cap on the next invocation.

Works with 1 or N accounts — there is no separate "single-account" path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from shared.config_dir import TaskConfigDir
from shared.config_models import UsageCapConfig
from shared.proc_group import terminate_process_group

if TYPE_CHECKING:
    from shared.cost_store import CostStore

logger = logging.getLogger(__name__)

__all__ = [
    'UsageGate',
    'InvokeSlot',
    'AccountState',
    'AccountPhase',
    'SessionBudgetExhausted',
    'IllegalTransitionError',
]
# NOTE: AccountLease is intentionally NOT listed here even though it is part
# of this module's public surface (consumed by callers of
# before_invoke()/InvokeSlot.lease). shared/tests/test_public_api.py pins
# this module's __all__ to an exact set (and shared.__all__ to the union of
# every submodule's __all__); AccountLease is out of that pinned set's scope.
# __all__ only governs `from shared.usage_gate import *`; explicit
# `from shared.usage_gate import AccountLease` works regardless of __all__
# membership.

# Patterns that indicate a usage cap has been hit (from Claude Code CLI output)
CAP_HIT_PREFIXES = [
    "You've hit your",
    "You've used",
    "You're out of extra",
    "You're now using extra",
]
# Secondary confirmation — at least one of these keywords must also appear in
# the same text for a CAP_HIT or NEAR_CAP prefix match to be accepted
# (defense-in-depth against ambiguous prefix false positives).
# NOTE: 'upgrade' was narrowed to multi-word phrases because the bare verb is
# too common in unrelated CLI messaging (e.g. 'Upgrade to v2 for more features')
# and would effectively reduce the guard to a near-prefix-only match in those
# cases.  'upgrade your plan' and 'upgrade your subscription' are natural SaaS
# cap-message phrases unlikely to appear in non-cap contexts.  The primary
# defense remains the CAP_HIT_PREFIXES / NEAR_CAP_PREFIXES prefix match.
#
# Known verbatim Claude CLI cap-hit messages that motivated this list
# (update if Claude changes its wording):
#   "You've hit your usage limit for Claude Pro. Your plan resets in 3 hours."
#       → 'usage limit', 'resets'
#   "You've used all available credits. Upgrade your plan for more capacity."
#       → 'upgrade your plan'
#   "You're out of extra usage for this billing period. Your plan resets in 2h."
#       → 'resets'
#   "You're now using extra compute credits. Your plan resets in 1h."
#       → 'resets'
#   "You're close to reaching your usage limit. Your plan resets in 1h."  (near-cap)
#       → 'usage limit', 'resets'
# See also: TestCapDetectionPatterns.test_realistic_cap_messages in
# test_usage_gate_exhaustive.py for the full parametrized fixture set.
CAP_CONFIRM_KEYWORDS = ["resets", "usage limit", "upgrade your plan", "upgrade your subscription"]

# Patterns for near-cap warnings (pause proactively)
NEAR_CAP_PREFIXES = [
    "You're close to",
]

# Codex (OpenAI) cap-hit patterns
CODEX_CAP_PATTERNS = ['usage limit reached', 'rate limit', 'quota exceeded',
                      'insufficient_quota', 'rate_limit_exceeded']

# Gemini (Google) cap-hit patterns
GEMINI_CAP_PATTERNS = ['quota exceeded', 'rate limit', 'resource exhausted',
                       'RESOURCE_EXHAUSTED', 'quota_exceeded']

CREDENTIALS_PATH = Path.home() / '.claude' / '.credentials.json'


def _probe_hit_local_budget_cap(stdout_bytes: bytes) -> bool:
    """Return True iff the probe stdout is a CLI JSON result reporting that
    the local ``--max-budget-usd`` cap was hit.

    This indicates the Anthropic API accepted the request and consumed real
    tokens — the account is NOT capped. Distinct from account-level cap hits
    (which surface as text prefixes in stderr, not JSON subtypes).
    """
    if not stdout_bytes:
        return False
    try:
        obj = json.loads(stdout_bytes.decode(errors='replace'))
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(obj, dict) and obj.get('subtype') == 'error_max_budget_usd'


class AccountPhase(StrEnum):
    """Explicit lifecycle phase for one account (PRD §7.3, task W4-γ).

    Exactly one phase is active per account at any time — this is the single
    source of truth that replaces the old, independently-mutable
    ``capped``/``probing``/``probe_in_flight``/``auth_failed`` booleans on
    :class:`AccountState` (which could — and did — drift out of sync, e.g.
    ``capped`` and ``auth_failed`` both True simultaneously).
    """

    AVAILABLE = 'available'
    PROBING = 'probing'
    PROBE_IN_FLIGHT = 'probe_in_flight'
    CAPPED = 'capped'
    AUTH_FAILED = 'auth_failed'


class IllegalTransitionError(Exception):
    """Raised by :meth:`UsageGate._transition` on a phase edge not present in
    ``_LEGAL_TRANSITIONS``. Account state is left unchanged."""


# Legal phase edges (PRD §7.3, task W4-γ). Any edge not listed here raises
# IllegalTransitionError from _transition unless called with force=True (the
# escape hatch reserved for _on_sighup_async's operator-driven hard reset).
_LEGAL_TRANSITIONS: dict[AccountPhase, frozenset[AccountPhase]] = {
    AccountPhase.AVAILABLE: frozenset({AccountPhase.CAPPED, AccountPhase.AUTH_FAILED}),
    AccountPhase.PROBING: frozenset({
        AccountPhase.AVAILABLE,
        AccountPhase.PROBE_IN_FLIGHT,
        AccountPhase.CAPPED,
        AccountPhase.AUTH_FAILED,
    }),
    AccountPhase.PROBE_IN_FLIGHT: frozenset({
        AccountPhase.AVAILABLE,
        AccountPhase.CAPPED,
        AccountPhase.AUTH_FAILED,
    }),
    AccountPhase.CAPPED: frozenset({AccountPhase.PROBING}),
    AccountPhase.AUTH_FAILED: frozenset({AccountPhase.AVAILABLE, AccountPhase.CAPPED}),
}


@dataclass
class AccountState:
    """Per-account cap tracking."""

    name: str
    token: str | None          # None = default account (no override)
    phase: AccountPhase = AccountPhase.AVAILABLE
    resets_at: datetime | None = None
    pause_started_at: datetime | None = None
    resume_task: asyncio.Task | None = field(default=None, repr=False)
    probe_count: int = 0
    near_cap: bool = False
    auth_failed_at: datetime | None = None
    auth_reprobe_task: asyncio.Task | None = field(default=None, repr=False)
    # Monotonic counter bumped once per successful `_transition` edge (task
    # W4-δ, PRD §7.4). Lets a capturer of an `AccountLease` (see below)
    # detect whether the account has re-transitioned since the lease was
    # taken — see `UsageGate.lease_is_current`.
    generation: int = 0

    # --- Legacy-compat boolean shim -----------------------------------
    #
    # capped/probing/probe_in_flight/auth_failed used to be four independent
    # dataclass fields, mutated directly across ~10 sites in this module and
    # read/written by sibling test files outside this task's edit scope
    # (test_concurrency.py, test_usage_gate_exhaustive.py, test_auth_failed.py,
    # test_probe_loop.py, test_failover_integration.py, and
    # orchestrator/tests/test_usage_gate.py via its re-export shim).
    #
    # Production code (UsageGate._transition) now writes `phase` exclusively.
    # These properties keep the old boolean attribute-access surface working
    # for those out-of-scope callers: getter compares to the matching phase;
    # setter(True) enters that phase, setter(False) reverts to AVAILABLE only
    # if the flag being cleared is the account's *current* phase (clearing a
    # non-current flag is a no-op, matching legacy behavior where the other
    # flags were already False).
    @property
    def capped(self) -> bool:
        return self.phase == AccountPhase.CAPPED

    @capped.setter
    def capped(self, value: bool) -> None:
        if value:
            self.phase = AccountPhase.CAPPED
        elif self.phase == AccountPhase.CAPPED:
            self.phase = AccountPhase.AVAILABLE

    @property
    def probing(self) -> bool:
        return self.phase == AccountPhase.PROBING

    @probing.setter
    def probing(self, value: bool) -> None:
        if value:
            self.phase = AccountPhase.PROBING
        elif self.phase == AccountPhase.PROBING:
            self.phase = AccountPhase.AVAILABLE

    @property
    def probe_in_flight(self) -> bool:
        return self.phase == AccountPhase.PROBE_IN_FLIGHT

    @probe_in_flight.setter
    def probe_in_flight(self, value: bool) -> None:
        if value:
            self.phase = AccountPhase.PROBE_IN_FLIGHT
        elif self.phase == AccountPhase.PROBE_IN_FLIGHT:
            self.phase = AccountPhase.AVAILABLE

    @property
    def auth_failed(self) -> bool:
        return self.phase == AccountPhase.AUTH_FAILED

    @auth_failed.setter
    def auth_failed(self, value: bool) -> None:
        if value:
            self.phase = AccountPhase.AUTH_FAILED
        elif self.phase == AccountPhase.AUTH_FAILED:
            self.phase = AccountPhase.AVAILABLE


@dataclass(frozen=True)
class AccountLease:
    """Frozen snapshot of the account :meth:`UsageGate.before_invoke` selected
    (task W4-δ, PRD §7.4).

    Built IN-LOCK at selection time (after any PROBING -> PROBE_IN_FLIGHT
    claim), so ``name`` and ``token`` always identify the SAME account —
    closing a skew where :class:`InvokeSlot` used to re-derive
    ``account_name`` from :attr:`UsageGate.active_account_name` independently
    of the ``token`` ``before_invoke`` returned, which could name a
    different account (finding 3 / boundary test B5).

    ``generation`` is a snapshot of :attr:`AccountState.generation` at
    capture time — compare it against the account's live value (see
    :meth:`UsageGate.lease_is_current`) to detect a lease gone stale from a
    mid-flight re-transition.
    """

    name: str
    token: str | None
    generation: int


class SessionBudgetExhausted(Exception):
    """Raised when the per-run session budget is exceeded."""

    def __init__(self, cumulative_cost: float):
        self.cumulative_cost = cumulative_cost
        super().__init__(f'Session budget exhausted: ${cumulative_cost:.2f} spent')


class InvokeSlot:
    """Probe-slot guard for one iteration of a cap-retry loop.

    Guarantees that ``release_probe_slot`` is called on any exit path
    (``break``, ``continue``, ``return``, exception) unless the slot was
    explicitly settled by :meth:`detect_cap_hit` (returning True) or
    :meth:`confirm`.

    Use via :meth:`UsageGate.invoke_slot`::

        async with gate.invoke_slot() as slot:
            result = await run_agent(oauth_token=slot.token)
            if slot.detect_cap_hit(result.stderr, result.output):
                continue  # probe slot released by detect_cap_hit
            slot.confirm(result.cost_usd)
            break  # probe slot released by confirm
        # any other exit: __aexit__ calls release_probe_slot
    """

    __slots__ = ('_gate', 'lease', '_settled')

    def __init__(self, gate: UsageGate, lease: AccountLease | None) -> None:
        self._gate = gate
        self.lease = lease
        self._settled = False

    @property
    def token(self) -> str | None:
        """OAuth token of the leased account (task W4-δ, PRD §7.4)."""
        return self.lease.token if self.lease is not None else None

    @property
    def account_name(self) -> str:
        """Name of the leased account — the SAME account ``token`` came from.

        Derived from ``lease`` rather than independently re-resolved (the
        old ``gate.active_account_name`` re-derivation omitted
        ``probe_in_flight`` from its predicate, so it could name a
        *different* account than ``token`` — finding 3 / boundary test B5).
        """
        return self.lease.name if self.lease is not None else ''

    def detect_cap_hit(
        self,
        stderr: str,
        output: str,
        backend: str = 'claude',
    ) -> bool:
        """Proxy to ``UsageGate.detect_cap_hit``; auto-settles on True."""
        hit = self._gate.detect_cap_hit(
            stderr, output, backend, oauth_token=self.token,
        )
        if hit:
            self._settled = True
        return hit

    def confirm(self, cost_usd: float = 0.0) -> None:
        """Mark invocation successful; clears probe state and accumulates cost."""
        self._gate.confirm_account_ok(self.token)
        self._gate.on_agent_complete(cost_usd)
        self._settled = True

    def settle(self) -> None:
        """Mark probe state as externally handled.

        Call this after manually invoking gate methods that clear
        ``probe_in_flight`` (e.g. ``_handle_cap_detected`` in heuristic
        cap detection).
        """
        self._settled = True


class UsageGate:
    """Shared gate that pauses all agent invocations when a usage cap is hit.

    Tracks cap status per account and returns the first available account's
    token from ``before_invoke()``. Only blocks when *all* accounts are
    capped.  Works with 1 or N accounts.

    Phase precedence — CAPPED takes precedence over AUTH_FAILED: an account
    already CAPPED that then also hits an auth failure (e.g. a concurrent
    caller's 403 on a token that gets revoked while the account is capped)
    stays CAPPED rather than demoting to AUTH_FAILED — see
    ``_handle_auth_failure`` and ``_LEGAL_TRANSITIONS`` (CAPPED's only legal
    outbound edge is PROBING; CAPPED -> AUTH_FAILED is not legal). This is a
    deliberate choice, not a defect, but it does mean the two blocked
    reasons are not independently observable while an account is CAPPED.

    Recovery for a genuinely revoked token while CAPPED: if ``resets_at`` is
    known, ``_refresh_capped_accounts``/``before_invoke`` will move the
    account CAPPED -> PROBING -> PROBE_IN_FLIGHT once ``resets_at`` passes
    and hand it a real invocation, which will correctly reclassify it as
    AUTH_FAILED on a genuine 401/403 (the account is no longer CAPPED at
    that point, so the edge is legal). If ``resets_at`` is unknown (unset),
    ``_account_resume_probe_loop`` has no such deadline and will keep
    re-probing indefinitely at the ``max_probe_interval_secs`` backoff
    ceiling — recovery then requires an operator token refresh + SIGHUP
    (``_on_sighup_async`` force-resets every account to AVAILABLE). See
    PRD §7.3 (task W4-γ) for the full transition table.
    """

    def __init__(self, config: UsageCapConfig, *, cost_store: CostStore | None = None):
        self._config: UsageCapConfig = config
        self._open = asyncio.Event()
        self._open.set()  # start open
        self._lock = asyncio.Lock()
        self._cumulative_cost: float = 0.0
        self._paused_reason: str = ''
        self._pause_started_at: datetime | None = None
        self._total_pause_secs: float = 0.0
        self._cost_store: CostStore | None = cost_store
        self._project_id: str | None = None
        self._run_id: str | None = None
        self._last_account_name: str | None = None
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
        self._shutting_down: bool = False

        self._probe_config_dir = TaskConfigDir('usage-gate-probe')
        self._accounts: list[AccountState] = self._init_accounts()
        self._sighup_handler_installed: bool = False
        self.register_signal_handlers()

    def _init_accounts(self) -> list[AccountState]:
        """Resolve account tokens from env vars.

        If no accounts are configured, falls back to reading the default
        credential from ``~/.claude/.credentials.json``.
        """
        accounts: list[AccountState] = []
        for acct_cfg in self._config.accounts:
            token = os.environ.get(acct_cfg.oauth_token_env)
            if not token:
                logger.warning(
                    f'Account {acct_cfg.name!r}: env var {acct_cfg.oauth_token_env} '
                    f'not set — skipping'
                )
                continue
            accounts.append(AccountState(name=acct_cfg.name, token=token))

        if not accounts:
            token = _read_oauth_token()
            if token:
                accounts.append(AccountState(name='default', token=token))
                logger.info('Single-account mode: using default credential')
            else:
                logger.warning('No accounts configured and no default credential found')

        if accounts:
            logger.info(
                f'Failover: {len(accounts)} account(s) active — '
                + ', '.join(a.name for a in accounts)
            )
        return accounts

    def _transition(
        self,
        acct: AccountState,
        new_phase: AccountPhase,
        *,
        resets_at: datetime | None = None,
        reason: str = '',
        force: bool = False,
        clear_near_cap: bool = True,
    ) -> None:
        """Sole writer of ``AccountState.phase`` (PRD §7.3, task W4-γ).

        Writes the new phase, bumps ``acct.generation`` (task W4-δ, PRD
        §7.4 — the sole increment site, so an ``AccountLease`` captured
        before this call can later be checked for staleness via
        ``lease_is_current``), then recomputes the shared ``_open`` event in
        ONE place from the current phase of every account (DD-5 invariant:
        ``_open.is_set() <=> any(a.phase in {AVAILABLE, PROBING})``) —
        replacing the ~10 scattered ``self._open.set()``/``.clear()`` call
        sites this method's callers used to own directly.

        Also owns the per-phase side effects that used to be scattered
        across callers: starting/cancelling the resume and auth-reprobe
        background tasks, firing the ``cap_hit``/``auth_failed`` cost
        events (entering CAPPED/AUTH_FAILED), stamping gate-level
        ``_pause_started_at``/``_paused_reason`` when the edge closes the
        gate for every account — and, symmetrically, consuming the elapsed
        gate-level pause into ``_total_pause_secs`` and clearing
        ``_pause_started_at`` when an edge reopens the gate — plus recovery
        bookkeeping when leaving a blocked phase (probe_count reset,
        per-account ``pause_started_at``/``auth_failed_at`` clearing).
        ``_total_pause_secs`` is a purely gate-level measure: only the
        gate-level ``_pause_started_at`` clock above feeds it, so a pause is
        counted exactly once regardless of how many accounts it affects. The
        ``resumed``/``auth_resumed`` cost events are NOT fired here — they
        stay in the async callers that carry a probe-count label.

        ``clear_near_cap`` defaults to True (the phase write always implies
        a fresh near-cap read for every caller except one): ``release_probe_slot``
        passes ``clear_near_cap=False`` because it fires on an *exception*
        path unrelated to cap status, and must leave a stale ``near_cap``
        warning exactly as ``confirm_account_ok``/``detect_cap_hit`` left it.

        Raises :class:`IllegalTransitionError` — without mutating any state
        — if ``new_phase`` is not a legal edge from ``acct.phase`` per
        ``_LEGAL_TRANSITIONS``, unless ``force=True`` (reserved for
        ``_on_sighup_async``'s operator-driven hard reset to AVAILABLE).
        """
        if not force and new_phase not in _LEGAL_TRANSITIONS.get(acct.phase, frozenset()):
            logger.error(
                'Illegal phase transition for account %r: %s -> %s (reason=%r)',
                acct.name, acct.phase, new_phase, reason,
            )
            raise IllegalTransitionError(
                f'Illegal phase transition for account {acct.name!r}: '
                f'{acct.phase} -> {new_phase}'
            )

        old_phase = acct.phase
        acct.phase = new_phase
        # Bump the monotonic generation counter on every successful edge
        # (including force=True SIGHUP resets) — task W4-δ, PRD §7.4. This
        # is the sole increment site, mirroring _transition's role as the
        # sole writer of `phase`.
        acct.generation += 1
        if clear_near_cap:
            acct.near_cap = False

        # --- Recovery bookkeeping for the edge just taken -----------------
        # NOTE: this per-account clock does NOT feed _total_pause_secs — that
        # would double-count alongside the gate-level consumption below,
        # since the real cap-entry path stamps both clocks at the same
        # instant when this is the sole/last open account (regression fixed
        # at step-29/30). acct.pause_started_at is retained only for
        # per-account diagnostics / field hygiene (e.g. the SIGHUP field-clear
        # assertion).
        if old_phase == AccountPhase.CAPPED and new_phase != AccountPhase.CAPPED:
            acct.pause_started_at = None
        if old_phase == AccountPhase.AUTH_FAILED and new_phase != AccountPhase.AUTH_FAILED:
            acct.auth_failed_at = None
        if (
            (old_phase == AccountPhase.CAPPED and new_phase == AccountPhase.PROBING)
            or (old_phase == AccountPhase.PROBE_IN_FLIGHT and new_phase == AccountPhase.AVAILABLE)
        ):
            acct.probe_count = 0

        # --- Force hard-reset cleanup (operator-driven, SIGHUP only) ------
        # force=True is reserved for _on_sighup_async's per-account hard
        # reset to AVAILABLE — cancel whichever background task was running
        # (the CAPPED/AUTH_FAILED-entry branches below only ever start the
        # OPPOSITE task, so neither would otherwise cancel a stale one here)
        # and clear the fields the two edge-specific bookkeeping blocks
        # above don't cover for a CAPPED/AVAILABLE-self-loop source.
        if force:
            if acct.resume_task is not None and not acct.resume_task.done():
                acct.resume_task.cancel()
            if acct.auth_reprobe_task is not None and not acct.auth_reprobe_task.done():
                acct.auth_reprobe_task.cancel()
            acct.probe_count = 0
            acct.resets_at = None

        # --- Centralized _open recompute (DD-5) ---------------------------
        if any(
            a.phase in (AccountPhase.AVAILABLE, AccountPhase.PROBING)
            for a in self._accounts
        ):
            self._open.set()
            # Symmetric counterpart to the closing branch below: consume the
            # gate-level pause into _total_pause_secs and clear
            # _pause_started_at on reopen. This is the SOLE accumulation site
            # for _total_pause_secs (gate-level-only — see _transition's
            # docstring); the per-account recovery block above intentionally
            # does not also add to it, to avoid double-counting the same
            # elapsed interval. Without this branch, total_pause_secs (which
            # adds "now - _pause_started_at" whenever the latter is truthy)
            # would keep growing forever after the very first full-gate
            # pause, even while the gate sits open and accounts serve
            # traffic.
            if self._pause_started_at is not None:
                self._total_pause_secs += (
                    datetime.now(UTC) - self._pause_started_at
                ).total_seconds()
                self._pause_started_at = None
        else:
            self._open.clear()
            if self._pause_started_at is None:
                self._pause_started_at = datetime.now(UTC)
            if reason:
                self._paused_reason = reason
            # else: a reason-less closing edge (e.g. before_invoke's
            # PROBE_IN_FLIGHT probe-slot claim, reason='' by default) must
            # not clobber a real cap/auth reason already recorded here.
            # SIGHUP is the sole clearer of _paused_reason (see
            # _on_sighup_async).

        # --- Enter-phase side effects: task lifecycle + cost event --------
        if new_phase == AccountPhase.CAPPED:
            if acct.pause_started_at is None:
                acct.pause_started_at = datetime.now(UTC)
            if acct.auth_reprobe_task is not None and not acct.auth_reprobe_task.done():
                acct.auth_reprobe_task.cancel()
            self._start_account_resume_probe(acct)
            if self._cost_store:
                details: dict[str, str] = {'reason': reason}
                if resets_at is not None:
                    details['resets_at'] = resets_at.isoformat()
                self._fire_cost_event(acct.name, 'cap_hit', json.dumps(details))
        elif new_phase == AccountPhase.AUTH_FAILED:
            acct.auth_failed_at = datetime.now(UTC)
            if acct.resume_task is not None and not acct.resume_task.done():
                acct.resume_task.cancel()
            self._start_auth_reprobe(acct)
            if self._cost_store:
                details = {'reason': reason}
                if resets_at is not None:
                    details['resets_at'] = resets_at.isoformat()
                self._fire_cost_event(acct.name, 'auth_failed', json.dumps(details))

    async def check_at_startup(self) -> None:
        """No-op: pre-existing caps are detected reactively on first invocation.

        The usage API (claude.ai/api/oauth/usage) is no longer available.
        If an account is already capped, the first invocation attempt will
        detect it via stderr pattern matching in ``detect_cap_hit()``.
        """
        logger.info(
            'Usage gate startup: %d account(s) configured — '
            'caps will be detected reactively',
            len(self._accounts),
        )

    async def before_invoke(self) -> AccountLease | None:
        """Block until at least one account is available. Return its lease.

        Returns an :class:`AccountLease` snapshotting the selected account's
        name/token/generation (task W4-δ, PRD §7.4) — built IN-LOCK, after
        any PROBING -> PROBE_IN_FLIGHT claim, so the returned lease always
        names the SAME account as its token. Returns ``None`` if no accounts
        are configured (no token override).
        """
        # Session budget check
        if (self._config.session_budget_usd is not None
                and self._cumulative_cost >= self._config.session_budget_usd):
            raise SessionBudgetExhausted(self._cumulative_cost)

        if not self._accounts:
            raise RuntimeError(
                'No OAuth accounts available — configure accounts or provide credentials'
            )

        # Find first non-capped account (works with 1 or N)
        while True:
            async with self._lock:
                for acct in self._accounts:
                    if acct.capped or acct.probe_in_flight or acct.auth_failed:
                        continue
                    if acct.probing:
                        # First task claims the probe slot — others block
                        # until confirm_account_ok() or _handle_cap_detected().
                        # _transition owns: the phase write, probe_count
                        # reset, and the centralized _open recompute.
                        self._transition(acct, AccountPhase.PROBE_IN_FLIGHT)
                        logger.info(
                            f'Account {acct.name}: probe slot claimed — '
                            f'single task testing',
                        )
                    logger.debug(f'Using account {acct.name}')
                    # Failover detection: emit event if account changed.
                    # Update _last_account_name FIRST to close the race window,
                    # then fire the event non-blocking (fire-and-forget).
                    if (
                        self._last_account_name is not None
                        and self._last_account_name != acct.name
                    ):
                        old_name = self._last_account_name
                        self._last_account_name = acct.name
                        if self._cost_store:
                            self._fire_cost_event(
                                acct.name,
                                'failover',
                                json.dumps({'from': old_name, 'to': acct.name}),
                            )
                    else:
                        self._last_account_name = acct.name
                    return AccountLease(
                        name=acct.name, token=acct.token, generation=acct.generation,
                    )

            # All capped — check if any reset times have passed before blocking.
            refreshed = await self._refresh_capped_accounts()
            if refreshed:
                continue  # re-check accounts with updated flags

            # Still all capped after fresh check — wait on global gate.
            # NOTE: this clear() is NOT redundant with _transition's centralized
            # recompute, despite _transition being the sole writer of every
            # PRODUCTION phase change. Out-of-scope sibling test suites (and any
            # other caller of the retained legacy capped/probing/probe_in_flight/
            # auth_failed shims — see AccountState) mutate `phase` directly
            # through those property setters, bypassing _transition and its
            # _open recompute entirely. This clear() is derived independently:
            # the for-loop above just confirmed every account is non-AVAILABLE
            # and non-PROBING, so clearing here is always correct regardless of
            # how _open drifted.
            logger.info('All accounts capped — waiting for any to reopen')
            self._open.clear()
            await self._open.wait()

    @contextlib.asynccontextmanager
    async def invoke_slot(self):
        """Acquire an account slot, releasing the probe lock on any exit path.

        Yields an :class:`InvokeSlot` whose ``token`` and ``account_name``
        are ready to use.  On exit, if neither :meth:`~InvokeSlot.detect_cap_hit`
        (returning True) nor :meth:`~InvokeSlot.confirm` was called,
        ``release_probe_slot`` runs as a safety net.

        Usage::

            while True:
                async with gate.invoke_slot() as slot:
                    result = await run_agent(oauth_token=slot.token)
                    if slot.detect_cap_hit(result.stderr, result.output):
                        continue   # probe settled by cap detection
                    slot.confirm(result.cost_usd)
                    break          # probe settled by confirm
                # any other exit path (continue, exception): auto-released
        """
        lease = await self.before_invoke()
        slot = InvokeSlot(self, lease)
        try:
            yield slot
        finally:
            if not slot._settled:
                self.release_probe_slot(lease.token if lease is not None else None)

    def detect_cap_hit(
        self,
        stderr: str,
        result_text: str,
        backend: str = 'claude',
        oauth_token: str | None = None,
    ) -> bool:
        """Scan stderr and result text for cap-hit patterns.

        Returns True if a cap-hit or near-cap pattern was detected **and** an
        account was successfully resolved and mutated.  Returns False both when
        no pattern matches and when a pattern matches but ``_resolve_account``
        returned None (e.g. explicit unknown token / config drift) — in that
        case no account state changed and the retry loop should not increment
        consecutive_cap_hits or trigger a cooldown, since before_invoke() would
        return the same token on the next iteration.
        """
        combined = f'{stderr}\n{result_text}'

        # Check backend-specific patterns first
        if backend == 'codex':
            for pattern in CODEX_CAP_PATTERNS:
                if pattern.lower() in combined.lower():
                    return self._handle_cap_detected(
                        f'Codex cap hit: {pattern}', None, oauth_token,
                    )
        elif backend == 'gemini':
            for pattern in GEMINI_CAP_PATTERNS:
                if pattern.lower() in combined.lower():
                    return self._handle_cap_detected(
                        f'Gemini cap hit: {pattern}', None, oauth_token,
                    )

        # Claude cap/near-cap detection: require both a prefix match AND a
        # secondary confirmation keyword (defence against false positives on
        # generic prefixes like "You've used" or "You're close to").
        combined_lower = combined.lower()
        has_confirm_keyword = any(kw in combined_lower for kw in CAP_CONFIRM_KEYWORDS)
        if has_confirm_keyword:
            for prefix in CAP_HIT_PREFIXES:
                if prefix.lower() in combined_lower:
                    resets_at = _parse_resets_at(combined)
                    reason = _extract_cap_message(combined, prefix) or f'Cap detected: {prefix}'
                    return self._handle_cap_detected(reason, resets_at, oauth_token)

            for prefix in NEAR_CAP_PREFIXES:
                if prefix.lower() in combined_lower:
                    reason = _extract_cap_message(combined, prefix) or f'Near-cap warning: {prefix}'
                    return self._handle_near_cap_warning(reason, oauth_token)
        else:
            # No confirm keyword — the confirm-keyword guard above would have blocked
            # detection anyway, but if a cap-like prefix IS present, emit a
            # debug breadcrumb so silent false-negatives leave a trace
            # (e.g. stderr truncation or Claude changes its message format).
            for prefix in (*CAP_HIT_PREFIXES, *NEAR_CAP_PREFIXES):
                if prefix.lower() in combined_lower:
                    logger.debug(
                        'Cap-like prefix %r seen but no confirm keyword; ignoring',
                        prefix,
                    )
                    break  # first match is sufficient; avoid log spam

        return False

    def _handle_cap_detected(
        self,
        reason: str,
        resets_at: datetime | None,
        oauth_token: str | None,
    ) -> bool:
        """Mark the matching account as capped.

        Returns True if an account was resolved and mutated; False if
        ``_resolve_account`` returned None (unknown token / all capped).
        """
        acct = self._resolve_account(oauth_token)
        if acct is None:
            logger.warning(f'Cap detected but no matching account: {reason}')
            return False

        # resets_at is refreshed unconditionally (even on a same-phase repeat
        # detection) — it feeds the resume-probe backoff target and
        # _refresh_capped_accounts, independent of whether a phase edge fires.
        acct.resets_at = resets_at
        logger.warning(f'Account {acct.name} CAPPED: {reason}')
        if acct.phase != AccountPhase.CAPPED:
            # _transition owns: the phase write, near_cap clear, the
            # centralized _open recompute, cancelling/starting the
            # opposite/matching background task, the cap_hit cost event, and
            # gate-level pause bookkeeping. A same-phase repeat call (already
            # CAPPED) is a pure no-op here — the old inline mutations had no
            # legality concept and would silently re-run every field write
            # and re-fire the cost event on every repeat detection.
            self._transition(acct, AccountPhase.CAPPED, resets_at=resets_at, reason=reason)
        return True

    def _handle_near_cap_warning(
        self,
        reason: str,
        oauth_token: str | None,
    ) -> bool:
        """Record a near-cap warning without blocking the account.

        Returns True if an account was resolved and mutated; False if
        ``_resolve_account`` returned None (unknown token / all capped).
        """
        acct = self._resolve_account(oauth_token)
        if acct is None:
            logger.warning(f'Near-cap warning but no matching account: {reason}')
            return False

        acct.near_cap = True
        logger.warning(f'Account {acct.name} NEAR CAP: {reason}')
        if self._cost_store:
            self._fire_cost_event(acct.name, 'near_cap', json.dumps({'reason': reason}))
        return True

    def _handle_auth_failure(
        self,
        reason: str,
        oauth_token: str | None,
    ) -> bool:
        """Mark the matching account as auth_failed.

        Returns True if an account was resolved and mutated; False if
        ``_resolve_account`` returned None (unknown token).

        Auth failure is distinct from a usage cap: it indicates the OAuth
        token is no longer accepted (403/401), typically because org access
        was revoked or the token expired. The account is skipped by
        ``before_invoke`` until an explicit re-probe succeeds — usually
        after the operator updates the token in ``.env`` and sends SIGHUP,
        or after ``auth_reprobe_secs`` elapses.

        No-ops (returns True without transitioning) when the account is
        already CAPPED — CAPPED takes precedence over AUTH_FAILED by design;
        see the "Phase precedence" note in :class:`UsageGate`'s docstring
        for the full recovery-semantics writeup and its known gap.
        """
        acct = self._resolve_account(oauth_token)
        if acct is None:
            logger.warning(f'Auth failure but no matching account: {reason}')
            return False

        logger.warning(f'Account {acct.name} AUTH-FAILED: {reason}')
        if acct.phase not in (AccountPhase.AUTH_FAILED, AccountPhase.CAPPED):
            # In production, HTTP 429 with "out of extra usage" carries a
            # "resets ..." phrase in the body — parse and persist it so the
            # dashboard can surface a reset ETA without re-parsing reason
            # strings downstream. Skip the 1h fallback when there's no
            # "resets" hint at all (true 401/403 token revocation).
            resets_at = _parse_resets_at(reason) if 'resets' in reason.lower() else None
            # _transition owns: the phase write, near_cap clear, the
            # centralized _open recompute, cancelling/starting the
            # opposite/matching background task, the auth_failed cost event,
            # and gate-level pause bookkeeping. A same-phase repeat call
            # (already AUTH_FAILED) is a pure no-op here.
            self._transition(acct, AccountPhase.AUTH_FAILED, resets_at=resets_at, reason=reason)
        # else: already AUTH_FAILED (no-op repeat) OR already CAPPED —
        # CAPPED->AUTH_FAILED is not a legal edge (_LEGAL_TRANSITIONS[CAPPED]
        # == {PROBING}), so demoting a time-bounded cap to an operator-gated
        # auth_failed would raise IllegalTransitionError. A concurrent
        # sibling task can cap this account (via _handle_cap_detected) after
        # before_invoke already handed it out AVAILABLE; when the in-flight
        # caller's request then fails with a 403, the cap already makes the
        # account unavailable and self-recovers via resets_at/resume-probe,
        # which takes precedence over auth_failed. Return True unconditionally
        # (below) regardless of which branch ran so cli_invoke.py's
        # invoke_slot() still settles the slot and fails over.
        return True

    def _start_auth_reprobe(self, acct: AccountState) -> None:
        """Schedule a background re-probe loop for an auth_failed account."""
        # getattr default: some test fixtures construct UsageGate via
        # __new__ (bypassing __init__) and predate this field.
        if getattr(self, '_shutting_down', False):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if acct.auth_reprobe_task is None or acct.auth_reprobe_task.done():
            acct.auth_reprobe_task = loop.create_task(
                self._auth_reprobe_loop(acct),
                name=f'usage-gate-auth-reprobe-{acct.name}',
            )

    async def _auth_reprobe_loop(self, acct: AccountState) -> None:
        """Periodically re-probe an auth_failed account.

        Sleeps ``auth_reprobe_secs`` between attempts; each attempt reloads
        ``.env`` and re-reads ``os.environ`` before issuing a minimal CLI
        call. On success (probe returns True), clears ``auth_failed`` and
        re-opens the gate. On failure, keeps waiting for the next cycle.
        """
        interval = max(1, self._config.auth_reprobe_secs)
        while acct.auth_failed:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                return
            if not acct.auth_failed:
                return
            try:
                await self._reprobe_account(acct)
            except Exception:
                logger.warning(
                    f'Account {acct.name}: auth re-probe raised — '
                    f'retrying after interval',
                    exc_info=True,
                )

    async def _reprobe_account(self, acct: AccountState) -> None:
        """Reload env, refresh the account token, and probe once.

        On success: clear ``auth_failed`` + ``auth_failed_at`` and reopen the
        global gate. On failure: leave ``auth_failed`` set; caller retries.
        """
        load_dotenv(override=True)
        token_env = self._token_env_for(acct)
        if token_env:
            fresh = os.environ.get(token_env)
            if fresh and fresh != acct.token:
                logger.info(
                    f'Account {acct.name}: env token changed — refreshing'
                )
                acct.token = fresh

        logger.info(f'Account {acct.name}: firing auth re-probe')
        ok = await self._run_probe(acct)
        if ok:
            if acct.phase == AccountPhase.AUTH_FAILED:
                # _transition owns: the phase write, clearing auth_failed_at,
                # and the centralized _open recompute. A phase mismatch here
                # (e.g. SIGHUP's fan-out reprobing an account that was never
                # auth_failed) is a pure no-op — mirrors the idempotency
                # guards on _handle_cap_detected/_handle_auth_failure. The
                # old unconditional field writes + event fire re-ran (and
                # re-fired 'auth_resumed') on every successful call
                # regardless of the account's current phase.
                self._transition(acct, AccountPhase.AVAILABLE)
                logger.info(f'Account {acct.name} AUTH RESUMED (probe confirmed)')
                if self._cost_store:
                    self._fire_cost_event(
                        acct.name, 'auth_resumed', json.dumps({}),
                    )
        else:
            logger.info(
                f'Account {acct.name}: auth re-probe failed — staying auth_failed',
            )

    def _token_env_for(self, acct: AccountState) -> str | None:
        """Look up the env-var name that sourced *acct*'s token."""
        for cfg in self._config.accounts:
            if cfg.name == acct.name:
                return cfg.oauth_token_env
        return None

    def register_signal_handlers(self) -> None:
        """Install a SIGHUP handler that triggers a token-reload + reprobe.

        Idempotent: safe to call multiple times. Tracks installation via
        ``self._sighup_handler_installed`` so a second call is a no-op.

        Uses ``loop.add_signal_handler`` to avoid the pitfalls of
        ``signal.signal`` inside asyncio (interrupting asyncio internals at
        arbitrary bytecode). When no event loop is running (e.g. when the
        gate is constructed before ``asyncio.run()``), this returns silently
        with a debug breadcrumb; callers that need the handler must invoke
        this method again from inside the loop.
        """
        if self._sighup_handler_installed:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug(
                'SIGHUP handler deferred: no running event loop at __init__; '
                'callers must invoke register_signal_handlers() inside the loop',
            )
            return
        try:
            loop.add_signal_handler(signal.SIGHUP, self._on_sighup)
        except (NotImplementedError, ValueError):
            # NotImplementedError: Windows. ValueError: not main thread.
            logger.debug('SIGHUP handler not installed (unsupported on this platform/thread)')
            return
        self._sighup_handler_installed = True
        logger.info('SIGHUP handler installed for usage gate token reload')

    def _on_sighup(self) -> None:
        """Signal-handler entry point — schedules the async reprobe trigger."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(
            self._on_sighup_async(),
            name='usage-gate-sighup-reprobe',
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _on_sighup_async(self) -> None:
        """Reload tokens + reset cap/auth state for ALL accounts, then probe.

        Treats SIGHUP as an operator-driven "refresh everything" signal:
        - reload .env (override existing env vars)
        - refresh each account's token from its env var if it changed
        - force every account to AVAILABLE so it is probe-worthy
        - reopen the global gate
        - fire a probe per account in parallel

        ``cumulative_cost`` is intentionally NOT reset — it is a budget
        counter, unrelated to token state.
        """
        load_dotenv(override=True)
        for acct in self._accounts:
            token_env = self._token_env_for(acct)
            if token_env:
                fresh = os.environ.get(token_env)
                if fresh and fresh != acct.token:
                    logger.info(
                        f'SIGHUP: account {acct.name} env token changed — refreshing'
                    )
                    acct.token = fresh
            # _transition owns: the phase write, cancelling any in-flight
            # resume/auth-reprobe task, probe_count/resets_at reset,
            # pause-time consumption, auth_failed_at clearing, and the
            # centralized _open recompute. force=True is required because
            # CAPPED/AUTH_FAILED -> AVAILABLE is not a legal edge outside
            # this operator-driven hard reset.
            self._transition(acct, AccountPhase.AVAILABLE, force=True)
        self._paused_reason = ''
        logger.info(
            f'SIGHUP: reloaded {len(self._accounts)} account(s); firing probes'
        )
        await asyncio.gather(
            *(self._reprobe_account(a) for a in self._accounts),
            return_exceptions=True,
        )

    def _find_account_by_token(self, token: str) -> AccountState | None:
        for acct in self._accounts:
            if acct.token == token:
                return acct
        return None

    def _resolve_account(self, oauth_token: str | None) -> AccountState | None:
        """Look up an account by token, with two distinct fallback paths.

        Paths:
        1. If ``oauth_token`` is provided and ``_find_account_by_token`` returns a
           match, that account is returned.
        2. If ``oauth_token`` is provided but *no* match is found (config drift),
           log a DEBUG breadcrumb and return ``None`` — no best-guess fallback
           applies.  The caller logs a WARNING ('no matching account') which is
           the primary user-visible signal; the debug log here avoids duplicate
           WARNING noise for a single event.
        3. If ``oauth_token`` is ``None`` (no identity signal at all), fall back to
           the first uncapped account in ``_accounts``.  Return ``None`` if all
           accounts are capped.

        The distinction matters because silently attributing cap state to an
        unrelated account (old path 2) is a worse failure mode than a logged
        warning with no action.
        """
        if oauth_token:
            acct = self._find_account_by_token(oauth_token)
            if acct is None:
                logger.debug(
                    'oauth_token provided but does not match any configured account;'
                    ' possible config drift'
                )
            return acct
        # oauth_token is None: no identity — use first-available fallback
        for a in self._accounts:
            if not a.capped and not a.auth_failed:
                return a
        return None

    def _start_account_resume_probe(self, acct: AccountState) -> None:
        """Start an async resume probe for a specific account."""
        # getattr default: some test fixtures construct UsageGate via
        # __new__ (bypassing __init__) and predate this field.
        if getattr(self, '_shutting_down', False):
            return
        if not self._config.wait_for_reset:
            return
        try:
            loop = asyncio.get_running_loop()
            if acct.resume_task is None or acct.resume_task.done():
                acct.resume_task = loop.create_task(
                    self._account_resume_probe_loop(acct),
                    name=f'usage-gate-resume-{acct.name}',
                )
        except RuntimeError:
            pass

    async def _write_cost_event(
        self,
        account_name: str,
        event_type: str,
        details: str,
    ) -> None:
        """Write a cost event to CostStore. Silently swallows errors (telemetry only)."""
        if self._cost_store is None:
            return
        try:
            await self._cost_store.save_account_event(
                account_name=account_name,
                event_type=event_type,
                project_id=self._project_id,
                run_id=self._run_id,
                details=details,
                created_at=datetime.now(UTC).isoformat(),
            )
        except Exception as exc:
            logger.warning('CostStore write failed for %s/%s: %s', account_name, event_type, exc)

    def _fire_cost_event(
        self,
        account_name: str,
        event_type: str,
        details: str,
    ) -> None:
        """Fire-and-forget wrapper for _write_cost_event (for use in sync contexts)."""
        if self._cost_store is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.warning(
                'No running event loop for cost event %s/%s', event_type, account_name
            )
            return
        coro = self._write_cost_event(account_name, event_type, details)
        try:
            task = loop.create_task(
                coro,
                name=f'cost-event-{event_type}-{account_name}',
            )
        except RuntimeError as exc:
            coro.close()
            logger.warning(
                'Failed to schedule cost event %s/%s: %s', event_type, account_name, exc
            )
            return
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _refresh_capped_accounts(self) -> bool:
        """Check reset times for all capped accounts. Return True if any uncapped."""
        now = datetime.now(UTC)
        any_uncapped = False
        for acct in self._accounts:
            if not acct.capped:
                continue
            if acct.resets_at is not None and now >= acct.resets_at:
                logger.info(f'Account {acct.name}: reset time passed — uncapping (probing)')
                # _transition owns: the phase write, near_cap clear, probe_count
                # reset, pause-time consumption into _total_pause_secs, and the
                # centralized _open recompute (gate: one task confirms before
                # opening to all — PROBING, not AVAILABLE).
                self._transition(acct, AccountPhase.PROBING)
                any_uncapped = True

        return any_uncapped

    def on_agent_complete(self, cost: float) -> None:
        """Accumulate cost for session budget tracking."""
        self._cumulative_cost += cost

    async def _account_resume_probe_loop(self, acct: AccountState) -> None:
        """Repeatedly probe an account until it uncaps.

        Uses exponential backoff: ``probe_interval_secs * 2^probe_count``,
        capped at ``max_probe_interval_secs``.  The sleep duration is also
        bounded by the time remaining until ``resets_at``.

        Fires a minimal Claude invocation (haiku, 1 turn) to verify the
        account actually has capacity.  Only uncaps the account on success.
        """
        while acct.capped:
            target = acct.resets_at
            if target is None:
                target = datetime.now(UTC) + timedelta(hours=1)
                logger.warning(f'Account {acct.name}: no resets_at — defaulting to 1h')

            base = self._config.probe_interval_secs
            ceiling = self._config.max_probe_interval_secs
            interval = min(base * (2 ** acct.probe_count), ceiling)

            remaining = max(0, (target - datetime.now(UTC)).total_seconds())
            sleep_for = min(interval, remaining) if remaining > 0 else 0

            if sleep_for > 0:
                logger.info(
                    f'Account {acct.name}: sleeping {sleep_for:.0f}s '
                    f'(probe #{acct.probe_count + 1}, resets in {remaining:.0f}s)',
                )
                try:
                    await asyncio.sleep(sleep_for)
                except asyncio.CancelledError:
                    return

            # _refresh_capped_accounts may have already uncapped this account
            if not acct.capped:
                return

            acct.probe_count += 1
            logger.info(
                f'Account {acct.name}: firing probe #{acct.probe_count}',
            )

            ok = await self._run_probe(acct)

            if ok:
                # _refresh_capped_accounts runs from before_invoke OUTSIDE
                # self._lock and targets the same resets_at event, so it can win
                # the CAPPED->PROBING race while we awaited _run_probe above.
                # Only CAPPED->PROBING is a legal edge here, so a raced-away
                # phase (already PROBING/AVAILABLE/PROBE_IN_FLIGHT) makes the
                # success block a pure no-op — mirrors the auth-reprobe guard in
                # _reprobe_account. Without this guard _transition would raise
                # IllegalTransitionError inside this fire-and-forget task. The
                # confirmed probe still ends the loop either way (return below).
                if acct.phase == AccountPhase.CAPPED:
                    # Captured before _transition resets probe_count — the event
                    # label below reports the probe number that confirmed.
                    confirmed_probe_num = acct.probe_count
                    # _transition owns: the phase write (-> PROBING, gate: let one
                    # real task confirm first), near_cap clear, probe_count reset,
                    # pause-time consumption into _total_pause_secs, and the
                    # centralized _open recompute.
                    self._transition(acct, AccountPhase.PROBING)
                    logger.info(f'Account {acct.name} RESUMED (probe confirmed)')
                    if self._cost_store:
                        await self._write_cost_event(
                            acct.name, 'resumed',
                            json.dumps({'label': f'probe #{confirmed_probe_num} confirmed'}),
                        )
                return
            else:
                logger.info(
                    f'Account {acct.name}: probe #{acct.probe_count} failed — '
                    f'retrying after backoff',
                )

    async def _run_probe(self, acct: AccountState) -> bool:
        """Fire a minimal Claude invocation to test if *acct* has capacity.

        Returns ``True`` if the invocation succeeded (no cap hit), ``False``
        otherwise.  Uses haiku to minimise cost (~$0.001 per probe).
        """
        _PROBE_TIMEOUT = 30

        if acct.token is not None:
            self._probe_config_dir.write_credentials(acct.token)

        cmd = [
            'claude', '--print', '--output-format', 'json',
            '--model', 'haiku',
            '--max-turns', '1',
            '--max-budget-usd', '0.01',
            '--permission-mode', 'bypassPermissions',
            '--', 'Say ok',
        ]

        env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
        if acct.token is not None:
            env['CLAUDE_CODE_OAUTH_TOKEN'] = acct.token
        env['CLAUDE_CONFIG_DIR'] = str(self._probe_config_dir.path)

        proc: asyncio.subprocess.Process | None = None
        pgid: int | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                start_new_session=True,
            )
            # Capture pgid at spawn (pgid == pid under start_new_session).
            pgid = proc.pid
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=_PROBE_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(f'Account {acct.name}: probe timed out')
            if proc is not None and pgid is not None:
                await terminate_process_group(proc, pgid, grace_secs=5.0)
            return False
        except asyncio.CancelledError:
            # Shutdown path: reap the subprocess and re-raise so the probe
            # task actually terminates. Swallowing the cancel would leave
            # usage_gate.shutdown() blocked waiting for this task forever.
            if proc is not None and pgid is not None:
                await terminate_process_group(proc, pgid, grace_secs=5.0)
            raise
        except Exception as exc:
            logger.warning(f'Account {acct.name}: probe error: {exc}')
            return False

        combined = (
            (stderr_bytes.decode(errors='replace') if stderr_bytes else '')
            + '\n'
            + (stdout_bytes.decode(errors='replace') if stdout_bytes else '')
        )

        # NOTE — intentional asymmetry with detect_cap_hit:
        # This loop does NOT apply the CAP_CONFIRM_KEYWORDS guard used by
        # detect_cap_hit.  The probe runs only while an account is already
        # blocked (capped or auth_failed); any whiff of a cap prefix in the
        # probe output means the account is still capped and we must NOT
        # unpause it.  Being conservative here avoids the far worse outcome
        # of unpausing a capped account and burning quota on a still-limited
        # account.  See CAP_CONFIRM_KEYWORDS (module top) for the current
        # keyword list.  Do not 'fix' this asymmetry without understanding the
        # safety-margin implications — see
        # test_probe_prefix_only_without_confirm_keyword_still_returns_false.
        #
        # Demote auth_failed → capped on cap-prefix: an auth_failed account
        # whose reprobe shows a cap-prefix is a 429 misclassification we
        # correct in-flight via the single legal AUTH_FAILED -> CAPPED edge
        # (see _LEGAL_TRANSITIONS) rather than leaving it on the
        # longer-cadence auth-reprobe loop.
        for prefixes in (CAP_HIT_PREFIXES, NEAR_CAP_PREFIXES):
            for prefix in prefixes:
                if prefix.lower() in combined.lower():
                    logger.info(
                        f'Account {acct.name}: probe got cap message: {prefix}',
                    )
                    if acct.phase == AccountPhase.AUTH_FAILED:
                        resets_at = _parse_resets_at(combined)
                        reason = (
                            _extract_cap_message(combined, prefix)
                            or f'Cap detected via auth-reprobe: {prefix}'
                        )
                        logger.warning(
                            f'Account {acct.name}: auth-reprobe saw cap '
                            f'message — demoting auth_failed → capped'
                        )
                        # Cancel + clear the stale auth-reprobe task now (the
                        # current call IS that task; cancel() only takes
                        # effect at its next suspension point, so the field
                        # is cleared explicitly here rather than left to
                        # settle asynchronously).
                        if (
                            acct.auth_reprobe_task is not None
                            and not acct.auth_reprobe_task.done()
                        ):
                            acct.auth_reprobe_task.cancel()
                        acct.auth_reprobe_task = None
                        # resets_at is persisted here (mirrors
                        # _handle_cap_detected) — _transition only threads
                        # resets_at into the cap_hit cost-event details, it
                        # does not write acct.resets_at itself.
                        acct.resets_at = resets_at
                        # _transition owns: the phase write, clearing
                        # auth_failed_at, starting the account-resume probe
                        # loop, the cap_hit cost event, and the centralized
                        # _open recompute.
                        self._transition(
                            acct, AccountPhase.CAPPED,
                            resets_at=resets_at, reason=reason,
                        )
                    return False

        if proc.returncode != 0:
            # Distinguish the probe's own $0.01 budget exhaustion from real
            # Anthropic-side failures. A non-zero exit with subtype
            # ``error_max_budget_usd`` means the API accepted the request and
            # consumed real tokens — the account has capacity; the probe
            # simply can't spend more than $0.01 per run. Cache-creation on a
            # fresh session easily pushes total_cost past $0.01, so this is a
            # routine outcome, not a cap hit.
            if _probe_hit_local_budget_cap(stdout_bytes):
                logger.info(
                    f'Account {acct.name}: probe hit local $0.01 budget '
                    f'cap (API accepted request) — treating as success',
                )
                return True
            logger.warning(
                f'Account {acct.name}: probe exited {proc.returncode}',
            )
            return False

        logger.info(f'Account {acct.name}: probe succeeded')
        return True

    async def shutdown(self) -> None:
        """Cancel all resume probe tasks and drain in-flight background cost-event tasks."""
        # Arm the B10 guard FIRST — before cancelling/draining anything —
        # so a _transition racing this teardown (e.g. a probe or cost-event
        # task completing concurrently) cannot start a new resume/reprobe
        # task via _start_account_resume_probe/_start_auth_reprobe, which
        # would otherwise leak a background task past shutdown.
        self._shutting_down = True
        for acct in self._accounts:
            if acct.resume_task and not acct.resume_task.done():
                acct.resume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await acct.resume_task
                acct.resume_task = None
            if acct.auth_reprobe_task and not acct.auth_reprobe_task.done():
                acct.auth_reprobe_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await acct.auth_reprobe_task
                acct.auth_reprobe_task = None

        for task in list(self._background_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._probe_config_dir.cleanup()

    @property
    def is_paused(self) -> bool:
        if not self._accounts:
            return False
        return all(a.capped or a.auth_failed for a in self._accounts)

    async def wait_for_open(self, timeout: float | None = None) -> bool:
        """Block until at least one account is available, or *timeout* elapses.

        Returns True if the gate opened (an account is available, or its reset
        time has passed), False if the timeout elapsed while still all-capped.

        Unlike :meth:`before_invoke`, this does not claim a probe slot or
        return a token — it is intended for callers that want to defer work
        until accounts uncap (e.g. the curator worker after a batch hit
        :class:`AllAccountsCappedException`) and then re-enter their normal
        invocation path. With ``timeout=None``, blocks indefinitely.

        Honours the same reset-time refresh as ``before_invoke``: if any
        account's ``resets_at`` has passed when this is called, returns
        immediately without sleeping.
        """
        if not self._accounts:
            # No-account mode behaves like permanently-open.
            return True
        # Cheap fast-path: check current state before claiming the lock.
        if not self.is_paused:
            return True
        # Give expired-reset accounts a chance to flip before we wait.
        if await self._refresh_capped_accounts():
            return True
        try:
            if timeout is None:
                await self._open.wait()
            else:
                await asyncio.wait_for(self._open.wait(), timeout=timeout)
        except TimeoutError:
            return False
        return True

    @property
    def paused_reason(self) -> str:
        return self._paused_reason

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    @property
    def total_pause_secs(self) -> float:
        if self._pause_started_at:
            return self._total_pause_secs + (
                datetime.now(UTC) - self._pause_started_at
            ).total_seconds()
        return self._total_pause_secs

    @property
    def account_count(self) -> int:
        """Number of configured accounts."""
        return len(self._accounts)

    @property
    def active_account_name(self) -> str | None:
        """Name of the first non-capped, non-auth-failed account, or None."""
        for acct in self._accounts:
            if not acct.capped and not acct.auth_failed:
                return acct.name
        return None

    @property
    def soonest_resets_at(self) -> datetime | None:
        """Earliest ``resets_at`` across currently-capped accounts, or None if unknown.

        Returns None when no account is capped, or when every capped account
        has ``resets_at=None`` (i.e. the reset time is not yet known).
        """
        times = [
            acct.resets_at
            for acct in self._accounts
            if acct.capped and acct.resets_at is not None
        ]
        return min(times) if times else None

    def confirm_account_ok(self, oauth_token: str | None) -> None:
        """Clear near_cap and (if applicable) the probing gate after a successful invocation.

        Called by ``invoke_with_cap_retry`` when an invocation succeeds (no cap
        detected).  Two effects:

        1. **Always** clears any stale ``near_cap`` flag on the matched account.
        2. If ``probe_in_flight`` was set (a probe cycle was in progress), clears
           that flag, resets ``probe_count``, and opens the shared ``_open`` event
           so other tasks may use this account.
        """
        acct = self._find_account_by_token(oauth_token) if oauth_token else None
        if acct is None:
            return
        # A successful invocation clears any stale near_cap flag; it will be
        # re-set on the next near-cap warning if still applicable.
        acct.near_cap = False
        if acct.phase == AccountPhase.PROBE_IN_FLIGHT:
            logger.info(f'Account {acct.name}: probe confirmed OK — opening to all tasks')
            # _transition owns: the phase write, probe_count reset, and the
            # centralized _open recompute.
            self._transition(acct, AccountPhase.AVAILABLE)

    def release_probe_slot(self, oauth_token: str | None) -> None:
        """Release a probe slot claimed by before_invoke() when invoke raises an exception.

        Called in the except handler of invoke_with_cap_retry / _invoke_with_session
        to clean up probe state when the invoke call raises (subprocess failure,
        CancelledError, etc.) before confirm_account_ok() or detect_cap_hit() can run.

        Effects (only when probe_in_flight is True on the matched account):
        - Clears probe_in_flight
        - Resets probe_count to 0
        - Re-opens the shared _open event so other tasks may proceed

        Is a no-op when:
        - oauth_token is None (no-op; we cannot identify the account)
        - oauth_token is unknown (account not found)
        - probe_in_flight is False on the matched account (nothing to release)

        Does NOT touch near_cap or capped — those flags track cap status, which
        is orthogonal to whether an exception occurred during invocation.
        """
        if not oauth_token:
            return
        acct = self._find_account_by_token(oauth_token)
        if acct is None:
            return
        if acct.phase == AccountPhase.PROBE_IN_FLIGHT:
            logger.info(
                f'Account {acct.name}: probe slot released after exception — '
                f'opening to all tasks',
            )
            # _transition owns: the phase write, probe_count reset, and the
            # centralized _open recompute. clear_near_cap=False preserves
            # this method's documented "does NOT touch near_cap" contract —
            # this is an exception path, orthogonal to cap status.
            self._transition(acct, AccountPhase.AVAILABLE, clear_near_cap=False)

    def lease_is_current(self, lease: AccountLease) -> bool:
        """Whether ``lease`` still reflects the live state of its account (task W4-δ).

        Resolves the account by ``lease.name`` and compares its live
        ``generation`` against the snapshot captured in the lease. A
        mismatch — or the account no longer existing (e.g. removed on
        SIGHUP reload) — means the account has transitioned (or vanished)
        since the lease was taken, so the lease is stale.

        This is a pure detectability primitive: it makes no decision about
        what to do with a stale lease. Consumer task W4-ε's
        ``InvokeSlot.report()`` uses this to implement the Q4 log-and-proceed
        fail-safe policy.
        """
        acct = next((a for a in self._accounts if a.name == lease.name), None)
        return acct is not None and acct.generation == lease.generation

    @property
    def project_id(self) -> str | None:
        """Project identifier set by the harness at run start."""
        return self._project_id

    @project_id.setter
    def project_id(self, value: str | None) -> None:
        self._project_id = value

    @property
    def run_id(self) -> str | None:
        """Run identifier set by the harness at run start."""
        return self._run_id

    @run_id.setter
    def run_id(self, value: str | None) -> None:
        self._run_id = value


# --- Helpers ---


def _read_oauth_token() -> str | None:
    """Read the OAuth access token from ~/.claude/.credentials.json."""
    try:
        data = json.loads(CREDENTIALS_PATH.read_text())
        # The credentials file may have different structures
        # Try common patterns
        if isinstance(data, dict):
            # Direct token field
            if 'accessToken' in data:
                return data['accessToken']
            if 'access_token' in data:
                return data['access_token']
            # Nested under a provider key
            for _key, val in data.items():
                if isinstance(val, dict):
                    if 'accessToken' in val:
                        return val['accessToken']
                    if 'access_token' in val:
                        return val['access_token']
        return None
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.debug(f'Cannot read OAuth credentials: {e}')
        return None


_MONTH_ABBR = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
}


def _parse_resets_at(text: str) -> datetime:
    """Parse reset time from cap-hit message text.

    Handles:
    - "resets in 3h" / "resets in 45m" / "resets in 2d"
    - "resets Mar 30, 6pm (Europe/London)" (date + time + tz)
    - "resets 9pm (Europe/London)" / "resets 3:00 AM (US/Pacific)"
    - Falls back to 1 hour from now
    """
    # Relative: "resets in Xh", "resets in Xm", "resets in Xd"
    m = re.search(r'resets\s+in\s+(\d+)\s*([hmd])', text, re.IGNORECASE)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).lower()
        delta = {
            'h': timedelta(hours=amount),
            'm': timedelta(minutes=amount),
            'd': timedelta(days=amount),
        }.get(unit, timedelta(hours=1))
        return datetime.now(UTC) + delta

    # Absolute with date: "resets Mar 30, 6pm (Europe/London)" or
    # "resets June 5, 7pm (Europe/London)". Month accepts 3-9 chars
    # (any abbreviation through full name) and is matched against
    # _MONTH_ABBR by its lowercased first three characters, since every
    # English month is uniquely identified by them.
    m = re.search(
        r'resets\s+([A-Za-z]{3,9})\s+(\d{1,2}),?\s+'
        r'(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            month_str = m.group(1).lower()[:3]
            day = int(m.group(2))
            time_str = m.group(3).strip()
            tz_str = m.group(4).strip()
            tz = zoneinfo.ZoneInfo(tz_str)
            month = _MONTH_ABBR.get(month_str)
            if month is None:
                raise ValueError(f'Unknown month: {month_str}')
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f'Cannot parse time: {time_str}')
            now_in_tz = datetime.now(tz)
            year = now_in_tz.year
            target = now_in_tz.replace(
                year=year, month=month, day=day,
                hour=parsed_time.hour, minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            # If target is in the past, assume next year
            if target <= now_in_tz:
                target = target.replace(year=year + 1)
            return target.astimezone(UTC)
        except Exception:
            pass

    # Absolute: "resets Xpm (TZ)" or "resets X:XX AM (TZ)"
    m = re.search(
        r'resets\s+(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            time_str = m.group(1).strip()
            tz_str = m.group(2).strip()
            tz = zoneinfo.ZoneInfo(tz_str)
            for fmt in ('%I:%M %p', '%I%p', '%I:%M%p', '%I %p'):
                try:
                    parsed_time = datetime.strptime(time_str, fmt).time()
                    break
                except ValueError:
                    continue
            else:
                return datetime.now(UTC) + timedelta(hours=1)

            now_in_tz = datetime.now(tz)
            target = now_in_tz.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0, microsecond=0,
            )
            if target <= now_in_tz:
                target += timedelta(days=1)
            return target.astimezone(UTC)
        except Exception:
            pass

    # Fallback: 1 hour from now
    return datetime.now(UTC) + timedelta(hours=1)


def _extract_cap_message(text: str, prefix: str) -> str:
    """Extract the full sentence containing the cap-hit prefix."""
    lower = text.lower()
    idx = lower.find(prefix.lower())
    if idx == -1:
        return ''
    # Find the end of the sentence
    end = text.find('\n', idx)
    if end == -1:
        end = min(idx + 200, len(text))
    return text[idx:end].strip()
