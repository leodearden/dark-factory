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

from shared.cli_invoke import AgentResult
from shared.config_dir import CONFIG_DIR_PREFIX, TaskConfigDir, sweep_stale_pid_dirs
from shared.config_models import UsageCapConfig
from shared.invocation_outcome import (
    OK,
    AuthFailed,
    CapHit,
    InvocationOutcome,
    NearCap,
    classify_invocation,
)
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

# The cap/near-cap/backend-pattern string tables that used to live here have
# moved to shared.invocation_outcome (task W4-beta single-source collapse);
# this module now consumes them indirectly via classify_invocation.
CREDENTIALS_PATH = Path.home() / '.claude' / '.credentials.json'

# Robustness bound for the scoped no-freeze wait (task 2857 amendment). The
# per-scope waiter Event is set() ONLY by the scope-uncap sweep
# (_refresh_scope_capped), so it never wakes on an ACCOUNT-level transition —
# most notably a PROBE_IN_FLIGHT account finishing its probe and going
# AVAILABLE with fresh scope headroom. Because is_paused ignores
# probe_in_flight, a scoped caller can reach that wait while the only
# otherwise-selectable account is mid-probe; without a bound it would sleep
# toward the full scope deadline (or max_probe_interval_secs, ~1800s) long
# after the probe opened headroom. When such a probe is outstanding, the
# scoped wait re-polls at most this many seconds so selection is re-checked
# promptly (self-correcting). Deliberately a small internal responsiveness
# constant, not an operator knob.
_SCOPE_WAIT_REPOLL_CEIL_SECS = 5.0

# Probe config-dir naming (task 3086). _PROBE_TASK_ID_PREFIX is the task_id
# stem handed to TaskConfigDir; _PROBE_DIR_PREFIX is the resulting on-disk
# prefix the sweep keys off. Both construction sites below build from the
# same constant, so the swept prefix and the created names are provably the
# same string.
_PROBE_TASK_ID_PREFIX = 'usage-gate-probe-'
_PROBE_DIR_PREFIX = CONFIG_DIR_PREFIX + _PROBE_TASK_ID_PREFIX

# Set once the stale-probe-dir sweep has run in this process. The sweep
# reclaims OTHER (dead) processes' leftovers, so it is a process-wide
# one-shot: re-running it per gate would re-scan /tmp for no benefit, and the
# pathological /tmp this bounds has a 40 MB directory inode.
_probe_dir_sweep_done: bool = False


def _sweep_stale_probe_dirs_once() -> int:
    """Reclaim dead-PID probe config dirs left by earlier processes.

    Runs at most once per process. Returns the number of dirs removed (0 when
    already swept this process, or on failure).

    UsageGate.shutdown() already removes this process's own probe dirs on a
    clean exit, so teardown was never the missing piece. What leaks is (a)
    hard kills — the fleet SIGKILLs and restarts every unit roughly 8-hourly
    and no teardown hook survives that — and (b) constructors that never call
    shutdown() at all (orchestrator/evals/runner.py,
    fused_memory/reconciliation/harness.py). Only reclaiming other processes'
    dead-PID leftovers bounds the population, at
    (live processes x accounts).

    Never raises — for ANY exception class, not just OSError: tmp hygiene must
    not be able to fail gate construction, and therefore orchestrator startup.
    """
    global _probe_dir_sweep_done
    if _probe_dir_sweep_done:
        return 0
    # Set BEFORE the call, not after, so a raising sweep still cannot re-run
    # on every subsequent gate construction.
    _probe_dir_sweep_done = True
    try:
        reclaimed = sweep_stale_pid_dirs(_PROBE_DIR_PREFIX)
        if reclaimed:
            # Silent on the zero case so the steady state stays quiet; loud
            # when there is something to say, so an operator can see the /tmp
            # population draining rather than rebuilding.
            logger.info(
                'UsageGate: reclaimed %d stale probe config dir(s) under %s '
                '(dead-PID sweep, task 3086)', reclaimed, _PROBE_DIR_PREFIX,
            )
        return reclaimed
    except Exception:
        # Deliberately broad. sweep_stale_pid_dirs already contains OSError
        # internally, so anything that reaches here is an UNFORESEEN failure —
        # a future bug, a pathological tree, a mocked side effect in a sibling
        # suite. Letting it escape would fail UsageGate.__init__ and therefore
        # orchestrator startup, which is strictly worse than leaving a stale
        # /tmp dir behind. Logged at WARNING with a traceback, never silent.
        logger.warning(
            'UsageGate: stale probe-dir sweep of %s failed — continuing without it '
            '(the next process start retries)', _PROBE_DIR_PREFIX, exc_info=True,
        )
        return 0


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
    AccountPhase.PROBING: frozenset(
        {
            AccountPhase.AVAILABLE,
            AccountPhase.PROBE_IN_FLIGHT,
            AccountPhase.CAPPED,
            AccountPhase.AUTH_FAILED,
        }
    ),
    AccountPhase.PROBE_IN_FLIGHT: frozenset(
        {
            AccountPhase.AVAILABLE,
            AccountPhase.CAPPED,
            AccountPhase.AUTH_FAILED,
        }
    ),
    AccountPhase.CAPPED: frozenset({AccountPhase.PROBING}),
    AccountPhase.AUTH_FAILED: frozenset({AccountPhase.AVAILABLE, AccountPhase.CAPPED}),
}


def scope_for(model: str, config: UsageCapConfig) -> str | None:
    """Derive the cap scope a model belongs to.

    Returns the model string itself when ``model`` is a scoped-cap model (it
    gets its own per-account cap scope), else an explicit ``None`` — the general
    scope, i.e. today's exact paths. With ``config.scoped_cap_models`` empty (the
    kill switch) this always returns ``None``, so every model — including a
    formerly-scoped one — falls back to the general scope (boundary B6).
    """
    if model in config.scoped_cap_models:
        return model
    return None


@dataclass
class ScopeCap:
    """Per-(account, model) cap overlay — a flag+timer snapshot, NOT a phase machine.

    A lightweight mirror of the account-level cap flags/timers scoped to a single
    model (see :func:`scope_for`), lazily populated by later tasks when a scoped
    cap is actually observed. The account-level :class:`AccountPhase` machine is
    untouched — this is a pure overlay (PRD decision 1).
    """

    capped: bool = False
    resets_at: datetime | None = None
    near_cap: bool = False
    capped_at: datetime | None = None


@dataclass
class AccountState:
    """Per-account cap tracking."""

    name: str
    token: str | None  # None = default account (no override)
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

    # Per-(account, model) cap overlay (task 2855 / PRD scope substrate).
    # Lazily populated by later tasks when a scoped cap is observed; keys are a
    # subset of UsageCapConfig.scoped_cap_models. Empty by default, so today's
    # general (unscoped) paths stay byte-identical.
    scope_caps: dict[str, ScopeCap] = field(default_factory=dict)

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

    __slots__ = ('_gate', 'lease', '_settled', 'scope')

    def __init__(
        self,
        gate: UsageGate,
        lease: AccountLease | None,
        scope: str | None = None,
    ) -> None:
        self._gate = gate
        self.lease = lease
        self._settled = False
        # Cap scope for this invocation (PRD task β): the invoked model when it
        # is a scoped-cap model, else None (the general scope). `report` /
        # `detect_cap_hit` forward it to the gate's cap handlers so a scoped
        # CapHit attributes to only this account's model-scope. β does NOT use
        # it for account selection — that is γ (task 2857).
        self.scope = scope

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
        """Proxy to ``UsageGate.detect_cap_hit``; auto-settles on True.

        Forwards ``self.scope`` (PRD task β) so a scoped cap detected here
        attributes to only this account's model-scope.

        Settling suppresses ``invoke_slot()``'s ``__aexit__`` safety net, so
        settling here also releases the PROBE_IN_FLIGHT claim explicitly
        (task 4096) — see the comment below for why it is unconditional.
        """
        hit = self._gate.detect_cap_hit(
            stderr,
            output,
            backend,
            oauth_token=self.token,
            scope=self.scope,
        )
        if hit:
            # Settling below suppresses invoke_slot()'s __aexit__ safety net, so
            # the PROBE_IN_FLIGHT claim must be released here (task 4096) —
            # mirroring report()'s arms. A guarded no-op whenever the gate
            # handler already moved the account off PROBE_IN_FLIGHT (the
            # unscoped CapHit -> CAPPED case), so the general path is unchanged.
            # It is load-bearing for the two handlers that take no phase
            # transition at all: the SCOPED _handle_cap_detected branch
            # (invariant S5) and _handle_near_cap_warning in EITHER scope.
            self._gate.release_probe_slot(self.token)
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

    def report(self, outcome: InvocationOutcome) -> None:
        """Apply *outcome*'s gate transition and settle the slot, atomically.

        Enforces "slot settled iff gate informed" as an invariant rather than
        caller discipline (PRD §7.4, task W4-ε): ``_settled`` is set in a
        ``finally`` so every variant leaves the slot settled exactly once.
        The PROBE_IN_FLIGHT claim taken by ``before_invoke()`` is released on
        every path — OK/AuthFailed and an UNSCOPED CapHit release it as a side
        effect of their phase transition; a SCOPED CapHit (which takes no phase
        transition at all — see below), NearCap, and the no-phase-change
        variants (ZeroOutputWedge/CliLocalError/Failure) release it explicitly
        via ``release_probe_slot`` since they don't otherwise touch phase. The
        CapHit arm calls ``release_probe_slot`` unconditionally: it is a
        guarded no-op once the account is already CAPPED, so one call covers
        both scopes (task 4096).

        Dispatches to the gate's existing handlers — this method owns no
        transition logic of its own:

        - OK -> ``confirm_account_ok`` (PROBE_IN_FLIGHT -> AVAILABLE; clears
          ``near_cap``). Does not accumulate cost — ``OK`` carries none; cost
          stays a caller concern (``confirm()`` / ``on_agent_complete``).
        - CapHit -> ``_handle_cap_detected`` (-> CAPPED), forwarding
          ``self.scope`` (PRD task β), then ``release_probe_slot``: a scoped
          cap attributes to only this account's model-scope and takes no phase
          transition, so the probe claim is released explicitly there (a no-op
          on the unscoped path, where the CAPPED transition already released
          it) and the account is left AVAILABLE.
        - AuthFailed -> ``_handle_auth_failure`` (-> AUTH_FAILED; a no-op if
          already CAPPED — CAPPED takes precedence, per that handler's own
          guard). Scope-blind.
        - NearCap -> ``_handle_near_cap_warning`` (annotation only, forwarding
          ``self.scope``), then ``release_probe_slot``.
        - Anything else (ZeroOutputWedge/CliLocalError/Failure) ->
          ``release_probe_slot`` only; no phase change.

        Q4 stale-lease fail-safe: if ``lease`` no longer reflects the live
        account state (``UsageGate.lease_is_current`` is False — a
        concurrent sibling re-transitioned the same account since this
        lease was taken), logs a warning and fires a ``lease_stale`` cost
        event, then falls through and proceeds with the normal transition
        + settle anyway (log-and-proceed, never raises). Safe because the
        underlying ``_handle_*``/``confirm_account_ok`` handlers are
        already idempotent and guarded against illegal transitions on a
        raced account.
        """
        if self.lease is not None and not self._gate.lease_is_current(self.lease):
            logger.warning(
                f'Account {self.account_name}: stale lease (generation '
                f'drifted since claimed) — reporting {type(outcome).__name__} '
                f'anyway (Q4 log-and-proceed fail-safe)',
            )
            self._gate._fire_cost_event(
                self.account_name,
                'lease_stale',
                json.dumps({'outcome': type(outcome).__name__}),
            )
        token = self.token
        try:
            if isinstance(outcome, OK):
                self._gate.confirm_account_ok(token)
            elif isinstance(outcome, CapHit):
                self._gate._handle_cap_detected(
                    outcome.reason,
                    outcome.resets_at,
                    token,
                    scope=self.scope,
                )
                # Unconditional, mirroring the NearCap arm (task 4096). The
                # general (scope=None) path is unaffected: _handle_cap_detected
                # has already transitioned the account to CAPPED, and
                # release_probe_slot is guarded on `phase == PROBE_IN_FLIGHT`,
                # so this is a no-op there — the exact idempotency pinned by
                # test_usage_gate_exhaustive.py::TestReleaseProbeSlot::
                # test_noop_after_handle_cap_detected_already_cleared. The
                # SCOPED path needs it: _handle_cap_detected's `scope is not
                # None` branch deliberately bypasses _transition (invariant S5)
                # and returns before any phase write, so without this the
                # PROBE_IN_FLIGHT claim taken by before_invoke() is never
                # released, and `finally: _settled = True` below also suppresses
                # invoke_slot()'s __aexit__ safety net — the account is skipped
                # by before_invoke's predicate forever.
                self._gate.release_probe_slot(token)
            elif isinstance(outcome, AuthFailed):
                self._gate._handle_auth_failure(f'HTTP {outcome.status}', token)
            elif isinstance(outcome, NearCap):
                self._gate._handle_near_cap_warning(outcome.reason, token, scope=self.scope)
                self._gate.release_probe_slot(token)
            else:
                self._gate.release_probe_slot(token)
        finally:
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
        # Per-scope wake Events for the S3 scope-wait (task 2857 / γ). Lazily
        # populated by _scope_waiter on scope-is-not-None paths only, so every
        # existing scope=None caller never allocates one (S1 / B6 byte-equiv).
        self._scope_waiters: dict[str, asyncio.Event] = {}
        # Per-scope last-selected-account tracker (task 2857 / γ) — independent
        # of _last_account_name so scoped failover events carry scope without
        # perturbing the general failover path (S1). Only touched on scope-is-
        # not-None paths.
        self._last_scope_account: dict[str, str] = {}
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks
        self._shutting_down: bool = False

        self._accounts: list[AccountState] = self._init_accounts()
        # Reclaim dead-PID probe dirs left behind by earlier processes BEFORE
        # creating this process's own, so our fresh dirs are never sweep
        # candidates (task 3086).
        _sweep_stale_probe_dirs_once()
        # Per-(account, pid) probe config dirs (PRD §6 task θ, finding 5):
        # concurrent probes across the ~6-process fleet, and the SIGHUP
        # parallel all-account probe gather within one process, must not
        # share a single .credentials.json. pid disambiguates cross-process
        # same-account probes; acct.name disambiguates same-process
        # cross-account probes.
        #
        # That per-(account, pid) naming fixed the race but made the dirs
        # unbounded in /tmp, since nothing reclaimed them once their owner
        # died (task 3086). They are now covered from both ends:
        # cleanup_at_exit handles clean exits — including the constructors
        # that never call shutdown() (orchestrator/evals/runner.py,
        # fused_memory/reconciliation/harness.py) — and the sweep above
        # reclaims whatever a SIGKILL left behind, which no hook can.
        self._probe_config_dirs: dict[str, TaskConfigDir] = {
            acct.name: TaskConfigDir(
                f'{_PROBE_TASK_ID_PREFIX}{acct.name}-{os.getpid()}',
                cleanup_at_exit=True,
            )
            for acct in self._accounts
        }
        # Back-compat alias: several sibling test suites (test_probe_loop.py,
        # test_usage_gate_exhaustive.py, orchestrator/tests/_orch_helpers.py)
        # monkeypatch/assert against a single gate._probe_config_dir. In the
        # common single-account case this is the same object as the sole
        # dict entry, so those assertions keep holding.
        self._probe_config_dir = next(
            iter(self._probe_config_dirs.values()), None
        ) or TaskConfigDir(f'{_PROBE_TASK_ID_PREFIX}{os.getpid()}', cleanup_at_exit=True)
        self._sighup_handler_installed: bool = False
        self.register_signal_handlers()

    def _config_dir_for(self, acct: AccountState) -> TaskConfigDir:
        """Return the per-account probe config dir for ``acct``.

        Falls back to the back-compat ``_probe_config_dir`` alias (mirroring
        the file's ``getattr(self, '_shutting_down', False)`` defensive-read
        idiom) so ``__new__``-built test fixtures that set only the alias
        still work.
        """
        return getattr(self, '_probe_config_dirs', {}).get(acct.name) or self._probe_config_dir

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
                acct.name,
                acct.phase,
                new_phase,
                reason,
            )
            raise IllegalTransitionError(
                f'Illegal phase transition for account {acct.name!r}: {acct.phase} -> {new_phase}'
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
        if (old_phase == AccountPhase.CAPPED and new_phase == AccountPhase.PROBING) or (
            old_phase == AccountPhase.PROBE_IN_FLIGHT and new_phase == AccountPhase.AVAILABLE
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
        if any(a.phase in (AccountPhase.AVAILABLE, AccountPhase.PROBING) for a in self._accounts):
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
            'Usage gate startup: %d account(s) configured — caps will be detected reactively',
            len(self._accounts),
        )

    async def before_invoke(self, scope: str | None = None) -> AccountLease | None:
        """Block until at least one account is available. Return its lease.

        Returns an :class:`AccountLease` snapshotting the selected account's
        name/token/generation (task W4-δ, PRD §7.4) — built IN-LOCK, after
        any PROBING -> PROBE_IN_FLIGHT claim, so the returned lease always
        names the SAME account as its token. Returns ``None`` if no accounts
        are configured (no token override).

        *scope* (PRD task γ, task 2857): when non-None (a scoped-cap model, e.g.
        ``claude-fable-5``), selection additionally skips accounts whose
        ``scope_caps[scope]`` is capped with a future uncap-deadline (S2), while
        account-level CAPPED/AUTH_FAILED still dominates for every scope (S4).
        ``scope=None`` (the general scope) is byte-identical to today (S1) —
        the scope predicate and the scope-wait fall-through are both guarded on
        ``scope is not None``.
        """
        # Session budget check
        if (
            self._config.session_budget_usd is not None
            and self._cumulative_cost >= self._config.session_budget_usd
        ):
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
                    if scope is not None and self._scope_capped_at(acct, scope, datetime.now(UTC)):
                        # Scope-capped for this model (S2): skip for this scope
                        # only — the account still serves general work (S1). The
                        # account-level skip above already dominates (S4), and
                        # this predicate is guarded on scope is not None so the
                        # scope=None path stays byte-identical.
                        continue
                    if acct.probing:
                        # First task claims the probe slot — others block
                        # until confirm_account_ok() or _handle_cap_detected().
                        # _transition owns: the phase write, probe_count
                        # reset, and the centralized _open recompute.
                        self._transition(acct, AccountPhase.PROBE_IN_FLIGHT)
                        logger.info(
                            f'Account {acct.name}: probe slot claimed — single task testing',
                        )
                    logger.debug(f'Using account {acct.name}')
                    # Failover detection: emit event if account changed. The
                    # tracker is updated FIRST to close the race window, then the
                    # event fires non-blocking (fire-and-forget). scope=None uses
                    # the general _last_account_name tracker (byte-identical, S1);
                    # a scoped selection uses an INDEPENDENT per-scope tracker so
                    # it never perturbs the general path and the event carries
                    # `scope` (same 'failover' event name, matching β's cap_hit/
                    # near_cap reuse).
                    if scope is None:
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
                    else:
                        scope_last = self._scope_last_account_map()
                        prev = scope_last.get(scope)
                        if prev is not None and prev != acct.name:
                            scope_last[scope] = acct.name
                            if self._cost_store:
                                self._fire_cost_event(
                                    acct.name,
                                    'failover',
                                    json.dumps({'from': prev, 'to': acct.name, 'scope': scope}),
                                )
                        else:
                            scope_last[scope] = acct.name
                    return AccountLease(
                        name=acct.name,
                        token=acct.token,
                        generation=acct.generation,
                    )

            # All capped — check if any reset times have passed before blocking.
            refreshed = await self._refresh_capped_accounts()
            if refreshed:
                continue  # re-check accounts with updated flags

            if scope is not None:
                # Scope-aware no-freeze fall-through (task γ, S3/S6). The select
                # loop found no account with headroom for THIS scope, and the
                # account-level reset check above freed none. Clear the per-scope
                # waiter BEFORE re-checking (clear-before-check) so a concurrent
                # uncap's set() cannot be lost, then optimistically uncap expired
                # scope caps; if that frees one, re-select — this is what returns
                # the optimistically-uncapped account in ONE call (B5-return).
                evt = self._scope_waiter(scope)
                evt.clear()
                if self._refresh_scope_capped(scope):
                    continue
                if not self.is_paused:
                    # Fleet is NOT frozen — at least one account is generally
                    # serviceable, only this scope is exhausted. Park on the
                    # per-scope waiter toward the soonest scope reset (or the
                    # max-probe ceiling when unknown) WITHOUT ever touching _open
                    # or _paused_reason, so scope exhaustion can never freeze the
                    # fleet (S3) nor delay a concurrent scope=None caller. On
                    # timeout the loop re-runs the selection-time sweep. INV-4:
                    # this always eventually returns an account rather than
                    # blocking indefinitely, so the caller's slot.report(CapHit)
                    # re-detection advances consecutive_cap_hits and the existing
                    # max_cap_retries/AllAccountsCappedException bound applies; the
                    # reactive re-cap installs a FUTURE deadline so uncaps are
                    # spaced, not a tight spin.
                    soonest = self._soonest_scope_reset(scope)
                    sleep_for = (
                        max(0.0, (soonest - datetime.now(UTC)).total_seconds())
                        if soonest is not None
                        else float(self._config.max_probe_interval_secs)
                    )
                    # Robustness (task 2857 amendment): the per-scope waiter is
                    # set() ONLY by the scope-uncap sweep, so it does NOT wake on
                    # an account-level transition that opens scope headroom. The
                    # common such case is a PROBE_IN_FLIGHT account finishing its
                    # probe and going AVAILABLE (confirm_account_ok) — is_paused
                    # ignores probe_in_flight, so a scoped caller can park here
                    # while the only otherwise-selectable account is mid-probe.
                    # When a probe is outstanding, cap the park at a small
                    # re-poll ceiling so the selection-time sweep re-runs within a
                    # bounded delay instead of sleeping toward the full scope
                    # deadline / max_probe_interval_secs. This still never touches
                    # _open / _paused_reason (S3) and never tight-spins (the
                    # ceiling is strictly positive); the pure scope-exhaustion
                    # steady state (no probe in flight) is unchanged — it sleeps
                    # toward the real deadline. (A bare `await self._open.wait()`
                    # here would busy-spin: the fleet is NOT frozen precisely
                    # because some account is generally AVAILABLE, so _open is
                    # already set and would return at once.)
                    if any(a.probe_in_flight for a in self._accounts):
                        sleep_for = min(sleep_for, _SCOPE_WAIT_REPOLL_CEIL_SECS)
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(evt.wait(), timeout=sleep_for)
                    continue
                # else: self.is_paused → the fleet is genuinely frozen (every
                # account account-level capped/auth_failed). Fall through to the
                # legacy _open freeze path below — nothing is serviceable.

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
    async def invoke_slot(self, scope: str | None = None):
        """Acquire an account slot, releasing the probe lock on any exit path.

        Yields an :class:`InvokeSlot` whose ``token`` and ``account_name``
        are ready to use.  On exit, if neither :meth:`~InvokeSlot.detect_cap_hit`
        (returning True) nor :meth:`~InvokeSlot.confirm` was called,
        ``release_probe_slot`` runs as a safety net.

        *scope* (PRD task β) is stored on the yielded slot (``slot.scope``) and
        forwarded by ``report`` / ``detect_cap_hit`` into the cap handlers so a
        scoped cap attributes to only this account's model-scope. It is now
        ALSO threaded into ``before_invoke(scope=scope)`` for scope-aware
        account *selection* (PRD task γ, task 2857): the yielded slot leases an
        account with headroom for this scope, skipping scope-capped ones.

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
        lease = await self.before_invoke(scope=scope)
        slot = InvokeSlot(self, lease, scope=scope)
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
        scope: str | None = None,
    ) -> bool:
        """Scan stderr and result text for cap-hit patterns.

        Delegates to ``classify_invocation`` (task W4-beta consumer-rewire):
        builds a synthetic ``AgentResult`` from *stderr*/*result_text*
        (``success=False`` so the OK short-circuit never fires; a real
        ``api_error_status``/``timed_out`` is never available here, so those
        default to None/False and AuthFailed/ZeroOutputWedge can never be
        produced by this call) and classifies it with ``strict_confirm=True``
        — the detect_cap_hit regime, which demands the CAP_CONFIRM_KEYWORDS
        guard (see DD-2 on ``classify_invocation`` for the asymmetry with the
        ``_run_probe`` regime). CliLocalError now outranks CapHit/NearCap
        uniformly here, exactly as it already did at the cli_invoke layer
        (reify-3604).

        Returns True if a cap-hit or near-cap pattern was detected **and** an
        account was successfully resolved and mutated.  Returns False both when
        no pattern matches and when a pattern matches but ``_resolve_account``
        returned None (e.g. explicit unknown token / config drift) — in that
        case no account state changed and the retry loop should not increment
        consecutive_cap_hits or trigger a cooldown, since before_invoke() would
        return the same token on the next iteration.

        *scope* (PRD task β) is forwarded verbatim to the cap/near-cap handler:
        when non-None it attributes the hit to only that account's model-scope
        (``acct.scope_caps``), leaving the account phase machine untouched; the
        cap-like-prefix breadcrumb path and the Failure return are scope-blind.
        """
        result = AgentResult(success=False, output=result_text, stderr=stderr)
        outcome = classify_invocation(result, strict_confirm=True, backend=backend)

        if isinstance(outcome, CapHit):
            return self._handle_cap_detected(
                outcome.reason,
                outcome.resets_at,
                oauth_token,
                scope=scope,
            )
        if isinstance(outcome, NearCap):
            return self._handle_near_cap_warning(outcome.reason, oauth_token, scope=scope)

        # Not a cap/near-cap verdict (Failure, or CliLocalError overriding a
        # cap-like prefix) — if a cap-like prefix IS present, emit a debug
        # breadcrumb so silent false-negatives leave a trace (e.g. stderr
        # truncation, Claude changes its message format, or a CLI-error
        # marker happened to co-occur with cap-like text). Kept as a direct
        # prefix scan since classify_invocation has no "breadcrumb-only" outcome.
        # Local import (rather than a module-top binding): keeps these two
        # names out of this module's own namespace so single-source-ownership
        # holds (see TestSingleSourceOwnership in test_invocation_outcome.py).
        from shared.invocation_outcome import CAP_HIT_PREFIXES, NEAR_CAP_PREFIXES

        combined_lower = f'{stderr}\n{result_text}'.lower()
        for prefix in (*CAP_HIT_PREFIXES, *NEAR_CAP_PREFIXES):
            if prefix.lower() in combined_lower:
                logger.debug(
                    'Cap-like prefix %r seen but no confirm keyword; ignoring',
                    prefix,
                )
                break  # first match is sufficient; avoid log spam

        return False

    def _scope_cap_for(self, acct: AccountState, scope: str) -> ScopeCap:
        """Lazily get-or-create the per-(account, model) cap overlay for *scope*.

        The scope substrate (task 2855 / PRD scope): ``acct.scope_caps`` starts
        empty and is populated on first scoped observation, so today's general
        (unscoped) paths stay byte-identical.
        """
        sc = acct.scope_caps.get(scope)
        if sc is None:
            sc = ScopeCap()
            acct.scope_caps[scope] = sc
        return sc

    def _scope_waiter(self, scope: str) -> asyncio.Event:
        """Get-or-create the per-scope wake ``asyncio.Event`` (S3 scope-wait).

        Lazily populates ``self._scope_waiters`` through a getattr-default read
        (mirroring the ``_probe_config_dirs`` / ``_shutting_down`` idiom) so a
        ``__new__``-built test fixture that never ran ``__init__`` still works.
        Only touched on scope-is-not-None paths, so every existing scope=None
        caller never allocates a waiter (S1 / B6 byte-equivalence).
        """
        waiters = getattr(self, '_scope_waiters', None)
        if waiters is None:
            waiters = self._scope_waiters = {}
        evt = waiters.get(scope)
        if evt is None:
            evt = waiters[scope] = asyncio.Event()
        return evt

    def _scope_last_account_map(self) -> dict[str, str]:
        """Get-or-create the per-scope last-selected-account map (task 2857 / γ).

        getattr-default read (mirroring ``_scope_waiter``) so a ``__new__``-built
        test fixture that never ran ``__init__`` still works. Only touched on
        scope-is-not-None paths, so the general failover path is byte-identical.
        """
        m = getattr(self, '_last_scope_account', None)
        if m is None:
            m = self._last_scope_account = {}
        return m

    def _scope_uncap_deadline(self, sc: ScopeCap) -> datetime | None:
        """The instant a scope cap becomes optimistically uncappable (S6).

        ``resets_at`` when the classifier parsed one, else the conservative
        fixed backoff ``capped_at + max_probe_interval_secs`` — the single
        deadline both the S6 sweep and the S3 wait consult (PRD open-Q2).
        ``capped_at`` is always stamped by β's scoped ``_handle_cap_detected``,
        so the deadline is computable whenever a scope cap is set; returns None
        only for a malformed cap carrying neither field.
        """
        if sc.resets_at is not None:
            return sc.resets_at
        if sc.capped_at is not None:
            return sc.capped_at + timedelta(seconds=self._config.max_probe_interval_secs)
        return None

    def _scope_capped_at(self, acct: AccountState, scope: str, now: datetime) -> bool:
        """True iff *acct* is scope-capped for *scope* with an uncap deadline
        still in the future at *now* — the authoritative admission predicate
        shared by ``before_invoke(scope=)`` selection and
        ``scope_capacity_snapshot`` (invariant S8).

        A scope cap that is absent, already uncapped, or past its deadline is
        NOT capping (the optimistic-uncap contract): such an account is admitted
        for the scope. A capped cap with no computable deadline (malformed —
        never produced by β) fails safe as still-capped.
        """
        sc = acct.scope_caps.get(scope)
        if sc is None or not sc.capped:
            return False
        deadline = self._scope_uncap_deadline(sc)
        if deadline is None:
            return True
        return now < deadline

    def _refresh_scope_capped(self, scope: str) -> bool:
        """Optimistically clear expired *scope* caps at selection time (S6).

        SYNCHRONOUS by design (contains no ``await``): a no-await sweep is
        atomic under asyncio's cooperative scheduler, so it cannot interleave
        with a concurrent scoped ``_handle_cap_detected`` and is safe to call
        outside ``self._lock`` — matching the module's short-critical-section
        discipline (``scope_caps`` mutations are already not lock-protected).
        Mirrors ``_refresh_capped_accounts`` one dimension down (per scope, not
        per account). Wakes the per-scope waiter (S3) whenever it uncaps
        anything so a concurrent scope-wait re-checks its select condition.

        Returns True iff at least one account's scope cap was cleared.
        """
        now = datetime.now(UTC)
        any_uncapped = False
        for acct in self._accounts:
            sc = acct.scope_caps.get(scope)
            if sc is None or not sc.capped:
                continue
            deadline = self._scope_uncap_deadline(sc)
            if deadline is not None and now >= deadline:
                sc.capped = False
                logger.info(
                    f'Account {acct.name}: scope {scope!r} uncap deadline passed '
                    f'— optimistically uncapping (S6)',
                )
                any_uncapped = True
        if any_uncapped:
            self._scope_waiter(scope).set()
        return any_uncapped

    def _soonest_scope_reset(self, scope: str) -> datetime | None:
        """Earliest scope uncap-deadline across accounts capped for *scope*.

        Mirrors ``soonest_resets_at`` for the scope dimension: the min
        ``_scope_uncap_deadline`` across accounts whose ``scope_caps[scope]`` is
        capped, or None when none is capped (or every capped one has no
        computable deadline). Consulted by the S3 scope-wait to bound its sleep.
        """
        deadlines: list[datetime] = []
        for acct in self._accounts:
            sc = acct.scope_caps.get(scope)
            if sc is None or not sc.capped:
                continue
            deadline = self._scope_uncap_deadline(sc)
            if deadline is not None:
                deadlines.append(deadline)
        return min(deadlines) if deadlines else None

    def scope_capacity_snapshot(self) -> dict[str, bool]:
        """Advisory per-scoped-model headroom snapshot (task γ, invariant S8).

        For each model in ``config.scoped_cap_models``, True iff AT LEAST ONE
        account has headroom for that scope, where headroom ==
        ``(not acct.capped and not acct.auth_failed) and not
        _scope_capped_at(acct, m, now)`` — the SAME admission
        ``before_invoke(scope=m)`` applies, so the resolver's advisory snapshot
        (threaded by δ) agrees with the gate's authoritative invoke-time
        predicate; a stale snapshot degrades to a scope-wait/failover rather
        than a wrong decision. Empty when ``scoped_cap_models`` is empty (the
        kill switch); each model False when no account is configured. Pure read
        — no sweep, no mutation (a read caller is never surprised by an
        optimistic-uncap side effect).
        """
        now = datetime.now(UTC)
        return {
            m: any(
                (not acct.capped and not acct.auth_failed)
                and not self._scope_capped_at(acct, m, now)
                for acct in self._accounts
            )
            for m in self._config.scoped_cap_models
        }

    def scope_status(self) -> dict[str, dict[str, dict]]:
        """Per-(account × scope) serialized cap state (task γ) for a future
        digest/dashboard panel (rendering out of scope here).

        Emits ``acct.name -> {model: {'capped', 'resets_at', 'near_cap',
        'capped_at'}}`` for every account with a NON-EMPTY ``scope_caps``;
        accounts that have observed no scoped cap/near-cap are skipped, so the
        map reflects only genuinely-observed scope state (the lazy-population
        contract). Datetimes render as ISO strings (JSON-ready); None passes
        through as None. Pure read — no sweep, no mutation.
        """
        return {
            acct.name: {
                model: {
                    'capped': sc.capped,
                    'resets_at': sc.resets_at.isoformat() if sc.resets_at else None,
                    'near_cap': sc.near_cap,
                    'capped_at': sc.capped_at.isoformat() if sc.capped_at else None,
                }
                for model, sc in acct.scope_caps.items()
            }
            for acct in self._accounts
            if acct.scope_caps
        }

    def _handle_cap_detected(
        self,
        reason: str,
        resets_at: datetime | None,
        oauth_token: str | None,
        scope: str | None = None,
    ) -> bool:
        """Mark the matching account as capped.

        Returns True if an account was resolved and mutated; False if
        ``_resolve_account`` returned None (unknown token / all capped).

        *scope* (PRD task β, invariant S5): when non-None (a scoped-cap model,
        e.g. ``claude-fable-5``), the cap is attributed to ONLY this account's
        model-scope — ``acct.scope_caps[scope]`` is flagged capped and the
        account-level phase machine is left untouched, so the account keeps
        serving general work. Attribution is by invoked model (PRD decision 2),
        never by cap-message text. ``scope=None`` (the general scope) is
        byte-identical to today: the account transitions to CAPPED via
        ``_transition``.
        """
        acct = self._resolve_account(oauth_token)
        if acct is None:
            logger.warning(f'Cap detected but no matching account: {reason}')
            return False

        if scope is not None:
            # Scoped (per-model) cap overlay — invariant S5: cap ONLY this
            # account's model-scope, leaving the account-level phase machine
            # untouched. _transition is both the sole phase writer AND the
            # account-level cap_hit event site, so the scoped path deliberately
            # bypasses it and fires its own cap_hit event (scope detail added)
            # to preserve observability without transitioning the account.
            # Consequence (task 4096): because this branch takes NO phase
            # transition, a caller holding a PROBE_IN_FLIGHT claim owns
            # releasing it — this handler will not do it for them (see
            # InvokeSlot.report / InvokeSlot.detect_cap_hit).
            # resets_at is the value the generic classifier already parsed
            # (decision 2); an unknown (None) is stored verbatim — the
            # None-backoff policy is γ's (task 2857).
            sc = self._scope_cap_for(acct, scope)
            sc.capped = True
            sc.resets_at = resets_at
            sc.capped_at = datetime.now(UTC)
            logger.warning(f'Account {acct.name} scope {scope!r} CAPPED: {reason}')
            if self._cost_store:
                details: dict[str, str] = {'reason': reason, 'scope': scope}
                if resets_at is not None:
                    details['resets_at'] = resets_at.isoformat()
                self._fire_cost_event(acct.name, 'cap_hit', json.dumps(details))
            return True

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
        scope: str | None = None,
    ) -> bool:
        """Record a near-cap warning without blocking the account.

        Returns True if an account was resolved and mutated; False if
        ``_resolve_account`` returned None (unknown token / all capped).

        *scope* (PRD task β, invariant S5): when non-None, the warning is
        annotated on ONLY this account's model-scope (``acct.scope_caps[scope]``)
        and the account-level ``near_cap`` flag stays untouched. ``scope=None``
        is byte-identical to today (sets ``acct.near_cap``).
        """
        acct = self._resolve_account(oauth_token)
        if acct is None:
            logger.warning(f'Near-cap warning but no matching account: {reason}')
            return False

        if scope is not None:
            # Scoped near-cap overlay (invariant S5): annotate only this
            # account's model-scope; the account-level near_cap flag is left put.
            sc = self._scope_cap_for(acct, scope)
            sc.near_cap = True
            logger.warning(f'Account {acct.name} scope {scope!r} NEAR CAP: {reason}')
            if self._cost_store:
                self._fire_cost_event(
                    acct.name,
                    'near_cap',
                    json.dumps({'reason': reason, 'scope': scope}),
                )
            return True

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
                    f'Account {acct.name}: auth re-probe raised — retrying after interval',
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
                logger.info(f'Account {acct.name}: env token changed — refreshing')
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
                        acct.name,
                        'auth_resumed',
                        json.dumps({}),
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
                    logger.info(f'SIGHUP: account {acct.name} env token changed — refreshing')
                    acct.token = fresh
            # _transition owns: the phase write, cancelling any in-flight
            # resume/auth-reprobe task, probe_count/resets_at reset,
            # pause-time consumption, auth_failed_at clearing, and the
            # centralized _open recompute. force=True is required because
            # CAPPED/AUTH_FAILED -> AVAILABLE is not a legal edge outside
            # this operator-driven hard reset.
            self._transition(acct, AccountPhase.AVAILABLE, force=True)
        self._paused_reason = ''
        logger.info(f'SIGHUP: reloaded {len(self._accounts)} account(s); firing probes')
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
            logger.warning('No running event loop for cost event %s/%s', event_type, account_name)
            return
        coro = self._write_cost_event(account_name, event_type, details)
        try:
            task = loop.create_task(
                coro,
                name=f'cost-event-{event_type}-{account_name}',
            )
        except RuntimeError as exc:
            coro.close()
            logger.warning('Failed to schedule cost event %s/%s: %s', event_type, account_name, exc)
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
            interval = min(base * (2**acct.probe_count), ceiling)

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
                            acct.name,
                            'resumed',
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

        config_dir = self._config_dir_for(acct)
        if acct.token is not None:
            config_dir.write_credentials(acct.token)

        cmd = [
            'claude',
            '--print',
            '--output-format',
            'json',
            '--model',
            'haiku',
            '--max-turns',
            '1',
            '--max-budget-usd',
            '0.01',
            '--permission-mode',
            'bypassPermissions',
            '--',
            'Say ok',
        ]

        env = {k: v for k, v in os.environ.items() if k != 'ANTHROPIC_API_KEY'}
        if acct.token is not None:
            env['CLAUDE_CODE_OAUTH_TOKEN'] = acct.token
        env['CLAUDE_CONFIG_DIR'] = str(config_dir.path)

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
                proc.communicate(),
                timeout=_PROBE_TIMEOUT,
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

        stdout_text = stdout_bytes.decode(errors='replace') if stdout_bytes else ''
        stderr_text = stderr_bytes.decode(errors='replace') if stderr_bytes else ''

        # NOTE — intentional asymmetry with detect_cap_hit: strict_confirm=False
        # skips the CAP_CONFIRM_KEYWORDS guard applied by detect_cap_hit's
        # strict_confirm=True regime (see DD-2 on classify_invocation).  The
        # probe runs only while an account is already blocked (capped or
        # auth_failed); any whiff of a cap prefix in the probe output means the
        # account is still capped and we must NOT unpause it.  Being
        # conservative here avoids the far worse outcome of unpausing a capped
        # account and burning quota on a still-limited account.  Do not 'fix'
        # this asymmetry without understanding the safety-margin implications
        # — see test_probe_prefix_only_without_confirm_keyword_still_returns_false.
        result = AgentResult(success=False, output=stdout_text, stderr=stderr_text)
        outcome = classify_invocation(result, strict_confirm=False, backend='claude')
        if isinstance(outcome, (CapHit, NearCap)):
            logger.info(
                f'Account {acct.name}: probe got cap message: {outcome.reason}',
            )
            # Demote auth_failed → capped on cap-prefix: an auth_failed
            # account whose reprobe shows a cap-prefix is a 429
            # misclassification we correct in-flight via the single legal
            # AUTH_FAILED -> CAPPED edge (see _LEGAL_TRANSITIONS) rather than
            # leaving it on the longer-cadence auth-reprobe loop.
            if acct.phase == AccountPhase.AUTH_FAILED:
                resets_at = outcome.resets_at if isinstance(outcome, CapHit) else None
                reason = outcome.reason
                logger.warning(
                    f'Account {acct.name}: auth-reprobe saw cap '
                    f'message — demoting auth_failed → capped'
                )
                # Cancel + clear the stale auth-reprobe task now (the
                # current call IS that task; cancel() only takes
                # effect at its next suspension point, so the field
                # is cleared explicitly here rather than left to
                # settle asynchronously).
                if acct.auth_reprobe_task is not None and not acct.auth_reprobe_task.done():
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
                    acct,
                    AccountPhase.CAPPED,
                    resets_at=resets_at,
                    reason=reason,
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

        # Clean up every per-account probe dir, deduped by object identity so
        # the common single-account case (where the alias IS the sole dict
        # entry) still calls cleanup() exactly once. getattr defaults keep
        # this working on __new__-built fixtures that set only the alias.
        dirs = set(getattr(self, '_probe_config_dirs', {}).values())
        alias = getattr(self, '_probe_config_dir', None)
        if alias is not None:
            dirs.add(alias)
        for d in dirs:
            d.cleanup()

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
            return (
                self._total_pause_secs
                + (datetime.now(UTC) - self._pause_started_at).total_seconds()
            )
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
            acct.resets_at for acct in self._accounts if acct.capped and acct.resets_at is not None
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
        """Release a probe claim taken by before_invoke() on any path that does
        not itself transition phase.

        Three callers, none of which the account's phase machine settles on its
        own (task 4096 widened this from exception-only):

        - **Exception** — the except handler of invoke_with_cap_retry /
          _invoke_with_session, when the invoke call raises (subprocess failure,
          CancelledError, etc.) before confirm_account_ok() or detect_cap_hit()
          can run; and ``invoke_slot()``'s ``__aexit__`` safety net.
        - **Scoped cap** — ``_handle_cap_detected``'s ``scope is not None``
          branch writes ``scope_caps[scope]`` and deliberately bypasses
          ``_transition`` (invariant S5), so the account-level claim is left for
          the caller: see ``InvokeSlot.report`` / ``InvokeSlot.detect_cap_hit``.
        - **Near-cap** — ``_handle_near_cap_warning`` is annotation-only in
          EITHER scope and never transitions phase.

        An UNSCOPED cap needs no call here (``_handle_cap_detected`` already
        transitioned the account to CAPPED), but the callers above make it
        unconditionally: the ``phase == PROBE_IN_FLIGHT`` guard below makes it a
        no-op, which is cheaper than re-deriving what the handler just did.

        Effects (only when probe_in_flight is True on the matched account):
        - Clears probe_in_flight
        - Resets probe_count to 0
        - Re-opens the shared _open event so other tasks may proceed

        Is a no-op when:
        - oauth_token is None (no-op; we cannot identify the account)
        - oauth_token is unknown (account not found)
        - probe_in_flight is False on the matched account (nothing to release)

        Does NOT touch near_cap or capped — those flags track cap status, which
        is orthogonal to why the probe claim is being handed back.
        """
        if not oauth_token:
            return
        acct = self._find_account_by_token(oauth_token)
        if acct is None:
            return
        if acct.phase == AccountPhase.PROBE_IN_FLIGHT:
            logger.info(
                f'Account {acct.name}: probe slot released — opening to all tasks',
            )
            # _transition owns: the phase write, probe_count reset, and the
            # centralized _open recompute. clear_near_cap=False preserves
            # this method's documented "does NOT touch near_cap" contract —
            # releasing the claim is orthogonal to cap status on every caller
            # (exception, scoped cap, near-cap).
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
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12,
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
        text,
        re.IGNORECASE,
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
                year=year,
                month=month,
                day=day,
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
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
        text,
        re.IGNORECASE,
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
                second=0,
                microsecond=0,
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
