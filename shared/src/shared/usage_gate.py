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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

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
    'SessionBudgetExhausted',
]

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


@dataclass
class AccountState:
    """Per-account cap tracking."""

    name: str
    token: str | None          # None = default account (no override)
    capped: bool = False
    resets_at: datetime | None = None
    pause_started_at: datetime | None = None
    resume_task: asyncio.Task | None = field(default=None, repr=False)
    probe_count: int = 0
    near_cap: bool = False
    # Probe lifecycle:
    #   probing=True  → freshly uncapped by probe, first task should claim it
    #   probe_in_flight=True → one task is testing, others must wait
    probing: bool = False
    probe_in_flight: bool = False


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

    __slots__ = ('_gate', 'token', 'account_name', '_settled')

    def __init__(self, gate: UsageGate, token: str | None) -> None:
        self._gate = gate
        self.token = token
        self.account_name = gate.active_account_name or ''
        self._settled = False

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

        self._probe_config_dir = TaskConfigDir('usage-gate-probe')
        self._accounts: list[AccountState] = self._init_accounts()

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

    async def before_invoke(self) -> str | None:
        """Block until at least one account is available. Return its OAuth token.

        Returns ``None`` if no accounts are configured (no token override).
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
                    if acct.capped or acct.probe_in_flight:
                        continue
                    if acct.probing:
                        # First task claims the probe slot — others block
                        # until confirm_account_ok() or _handle_cap_detected().
                        acct.probing = False
                        acct.probe_in_flight = True
                        self._open.clear()
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
                    return acct.token

            # All capped — check if any reset times have passed before blocking.
            refreshed = await self._refresh_capped_accounts()
            if refreshed:
                continue  # re-check accounts with updated flags

            # Still all capped after fresh check — wait on global gate
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
        token = await self.before_invoke()
        slot = InvokeSlot(self, token)
        try:
            yield slot
        finally:
            if not slot._settled:
                self.release_probe_slot(token)

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

        acct.capped = True
        acct.near_cap = False
        acct.probing = False
        acct.probe_in_flight = False
        acct.resets_at = resets_at
        if acct.pause_started_at is None:
            acct.pause_started_at = datetime.now(UTC)
        logger.warning(f'Account {acct.name} CAPPED: {reason}')
        if self._cost_store:
            self._fire_cost_event(acct.name, 'cap_hit', json.dumps({'reason': reason}))
        self._start_account_resume_probe(acct)

        # If all accounts are now capped, close the global gate
        if all(a.capped for a in self._accounts):
            self._open.clear()
            self._paused_reason = f'All accounts capped (last: {reason})'
            if self._pause_started_at is None:
                self._pause_started_at = datetime.now(UTC)
            logger.warning(f'Usage gate PAUSED: {self._paused_reason}')
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
        # oauth_token is None: no identity — use first-uncapped fallback
        for a in self._accounts:
            if not a.capped:
                return a
        return None

    def _start_account_resume_probe(self, acct: AccountState) -> None:
        """Start an async resume probe for a specific account."""
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
                acct.capped = False
                acct.near_cap = False
                acct.probing = True  # gate: one task confirms before opening to all
                if acct.pause_started_at:
                    self._total_pause_secs += (now - acct.pause_started_at).total_seconds()
                acct.pause_started_at = None
                any_uncapped = True

        if any_uncapped:
            self._open.set()
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
                confirmed_probe_num = acct.probe_count
                acct.capped = False
                acct.near_cap = False
                acct.probing = True  # gate: let one real task confirm first
                acct.probe_count = 0
                if acct.pause_started_at:
                    self._total_pause_secs += (
                        datetime.now(UTC) - acct.pause_started_at
                    ).total_seconds()
                acct.pause_started_at = None
                logger.info(f'Account {acct.name} RESUMED (probe confirmed)')
                self._open.set()
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
        # capped; any whiff of a cap prefix in the probe output means the
        # account is still capped and we must NOT unpause it.  Being
        # conservative here avoids the far worse outcome of unpausing a
        # capped account and burning quota on a still-limited account.
        # See CAP_CONFIRM_KEYWORDS (module top) for the current keyword list.
        # Do not 'fix' this asymmetry without understanding the safety-margin
        # implications — see test_probe_prefix_only_without_confirm_keyword_still_returns_false.
        for prefixes in (CAP_HIT_PREFIXES, NEAR_CAP_PREFIXES):
            for prefix in prefixes:
                if prefix.lower() in combined.lower():
                    logger.info(
                        f'Account {acct.name}: probe got cap message: {prefix}',
                    )
                    return False

        if proc.returncode != 0:
            logger.warning(
                f'Account {acct.name}: probe exited {proc.returncode}',
            )
            return False

        logger.info(f'Account {acct.name}: probe succeeded')
        return True

    async def shutdown(self) -> None:
        """Cancel all resume probe tasks and drain in-flight background cost-event tasks."""
        for acct in self._accounts:
            if acct.resume_task and not acct.resume_task.done():
                acct.resume_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await acct.resume_task
                acct.resume_task = None

        for task in list(self._background_tasks):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        self._probe_config_dir.cleanup()

    @property
    def is_paused(self) -> bool:
        if not self._accounts:
            return False
        return all(a.capped for a in self._accounts)

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
        """Name of the first non-capped account, or None."""
        for acct in self._accounts:
            if not acct.capped:
                return acct.name
        return None

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
        if acct.probe_in_flight:
            acct.probe_in_flight = False
            acct.probe_count = 0
            logger.info(f'Account {acct.name}: probe confirmed OK — opening to all tasks')
            self._open.set()

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
        if acct.probe_in_flight:
            acct.probe_in_flight = False
            acct.probe_count = 0
            logger.info(
                f'Account {acct.name}: probe slot released after exception — '
                f'opening to all tasks',
            )
            self._open.set()

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

    # Absolute with date: "resets Mar 30, 6pm (Europe/London)"
    m = re.search(
        r'resets\s+([A-Za-z]{3})\s+(\d{1,2}),?\s+'
        r'(\d{1,2}(?::\d{2})?\s*[ap]m)\s*\(([^)]+)\)',
        text, re.IGNORECASE,
    )
    if m:
        try:
            import zoneinfo
            month_str = m.group(1).lower()
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
