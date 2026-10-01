"""Tests for UsageGate auth_failed lifecycle.

Covers:
- AccountState.auth_failed field + pause_started_at semantics
- _handle_auth_failure marks account and fires cost event
- before_invoke skips auth_failed accounts
- re-probe loop re-reads env via load_dotenv(override=True)
- SIGHUP handler triggers immediate re-probe
- all-auth-failed closes the gate like all-capped
"""

from __future__ import annotations

import ast
import asyncio
import functools
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple
from unittest.mock import AsyncMock, patch

import pytest
from _usage_gate_test_helpers import spawn_fault as _spawn_fault

from shared import invocation_outcome
from shared.cli_invoke import AgentResult
from shared.config_models import AccountConfig, UsageCapConfig
from shared.invocation_outcome import AuthFailed, auth_failure_reason, classify_invocation
from shared.usage_gate import (
    _SPAWN_FAULT_THRESHOLD,
    AccountPhase,
    AccountState,
    UsageGate,
)


def _make_gate(
    account_names: list[str],
    *,
    wait_for_reset: bool = False,
    auth_reprobe_secs: int = 3600,
) -> UsageGate:
    acct_cfgs = []
    env_vars: dict[str, str] = {}
    for name in account_names:
        env_key = f'TEST_AUTH_TOKEN_{name.upper().replace("-", "_")}'
        env_vars[env_key] = f'fake-token-{name}'
        acct_cfgs.append(AccountConfig(name=name, oauth_token_env=env_key))
    config = UsageCapConfig(
        accounts=acct_cfgs,
        wait_for_reset=wait_for_reset,
        auth_reprobe_secs=auth_reprobe_secs,
    )
    with patch.dict(os.environ, env_vars):
        gate = UsageGate(config)
    gate._run_probe = AsyncMock(return_value=True)
    return gate


class TestAccountStateAuthFailedField:

    def test_auth_failed_defaults_false(self):
        acct = AccountState(name='a', token='t')
        assert acct.auth_failed is False
        assert acct.auth_failed_at is None


class TestHandleAuthFailure:

    def test_marks_account_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        marked = gate._handle_auth_failure(
            'HTTP 403: access denied', oauth_token='fake-token-a',
        )
        assert marked is True
        assert gate._accounts[0].auth_failed is True
        assert gate._accounts[0].auth_failed_at is not None
        # Does NOT set capped — auth failure is a separate lifecycle
        assert gate._accounts[0].capped is False
        # Other account untouched
        assert gate._accounts[1].auth_failed is False

    def test_unknown_token_returns_false(self):
        gate = _make_gate(['a'])
        marked = gate._handle_auth_failure('reason', oauth_token='unknown-token')
        assert marked is False
        assert gate._accounts[0].auth_failed is False

    def test_clears_probe_state(self):
        """_handle_auth_failure clears probe_in_flight so _open event doesn't deadlock."""
        gate = _make_gate(['a'])
        gate._accounts[0].probe_in_flight = True
        marked = gate._handle_auth_failure('reason', oauth_token='fake-token-a')
        assert marked is True
        assert gate._accounts[0].probe_in_flight is False


class TestAuthFailedPersistsResetsAt:
    """auth_failed event details carry the parsed resets_at when the reason
    text contains a "resets …" phrase. Source of truth for cap-time parsing
    is shared.invocation_outcome._parse_resets_at; the dashboard reads what we
    persist rather than re-parsing the reason string itself.

    DECIDED SHAPE (task 4042) — do not "helpfully" re-disable this branch.
    Restoring the 401/403 response-body snippet re-armed the ``'resets' in
    reason.lower()`` branch in ``_handle_auth_failure``, which was unreachable
    dead code while the reason was always the bare ``HTTP 401``. Two halves are
    deliberately pinned here:

      1. The branch IS allowed to fire — a 401/403 body carrying a genuinely
         parseable "resets in 3h" persists a REAL ETA. The line predates the
         regression (93baf1193a < b68eea415b), so suppressing it would invent
         new behaviour.
      2. An UNPARSEABLE "resets" hint persists NOTHING rather than a fabricated
         hour. The branch's original parser WAS the forked
         ``usage_gate._parse_resets_at``, which returned ``now + 1h`` on parse
         failure (task 4042 moved the call site off it; task 4357 retired the
         fork itself). The dashboard
         (``dashboard/src/dashboard/data/costs.py::_extract_resets_at``)
         surfaces the persisted value verbatim as a real reset ETA, so a
         fabricated one would put an invented recovery time on a revoked token
         — violating PRD 7.1.a ("an unknown reset time must be reported as
         explicitly unknown, never fabricated").
    """

    def _capture_fired_details(self, gate: UsageGate) -> list[tuple[str, str, str]]:
        """Replace gate._fire_cost_event with a sync recorder. Returns the
        list it appends to so tests can assert on it.
        """
        captured: list[tuple[str, str, str]] = []

        def _record(account_name: str, event_type: str, details: str) -> None:
            captured.append((account_name, event_type, details))

        gate._fire_cost_event = _record  # type: ignore[assignment]
        # Provide a truthy cost_store so the firing branch executes; the
        # recorder above bypasses real persistence.
        gate._cost_store = object()  # type: ignore[assignment]
        return captured

    def test_persists_resets_at_for_relative_reason(self):
        import json
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        gate._handle_auth_failure(
            "HTTP 429: You're out of extra usage · resets in 3h",
            oauth_token='fake-token-a',
        )
        assert len(captured) == 1
        _, event_type, details_json = captured[0]
        assert event_type == 'auth_failed'
        details = json.loads(details_json)
        assert 'resets_at' in details
        # ISO-format string parses; should be ~3h ahead.
        assert isinstance(details['resets_at'], str)
        assert details['resets_at'].endswith('+00:00')

    def test_persists_resets_at_for_date_prefixed_reason(self):
        """Production reason form for multi-day cap windows.

        Regression: prior to the regex widening, the gate's parser
        silently fell through to the 1-hour fallback for any month name
        of length ≥ 4. With the fix, "resets May 2, 6pm (Europe/London)"
        produces a real reset timestamp.
        """
        import json
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        gate._handle_auth_failure(
            "HTTP 429: You're out of extra usage · resets May 2, 6pm "
            "(Europe/London)",
            oauth_token='fake-token-a',
        )
        _, _, details_json = captured[0]
        details = json.loads(details_json)
        assert 'resets_at' in details
        from datetime import datetime as _dt
        parsed = _dt.fromisoformat(details['resets_at'])
        assert parsed.month == 5
        assert parsed.day == 2

    def test_persists_resets_at_for_full_month_name(self):
        """Full English month names (e.g. 'June', 'September') parse —
        not just 3-letter abbreviations.
        """
        import json
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        gate._handle_auth_failure(
            "HTTP 429: You're out of extra usage · resets September 1, "
            "6am (UTC)",
            oauth_token='fake-token-a',
        )
        _, _, details_json = captured[0]
        details = json.loads(details_json)
        assert 'resets_at' in details
        from datetime import datetime as _dt
        parsed = _dt.fromisoformat(details['resets_at'])
        assert parsed.month == 9
        assert parsed.day == 1

    def test_no_resets_at_for_pure_auth_error(self):
        """True OAuth-revocation reasons (no "resets" phrase) must NOT
        synthesize a bogus 1-hour fallback in the persisted details. A
        blank cell in the dashboard is correct here — there is no reset
        time for a permanently-revoked token.
        """
        import json
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        gate._handle_auth_failure(
            'HTTP 403: token has been revoked',
            oauth_token='fake-token-a',
        )
        _, _, details_json = captured[0]
        details = json.loads(details_json)
        assert 'resets_at' not in details
        assert details['reason'] == 'HTTP 403: token has been revoked'

    def test_unparseable_resets_hint_persists_no_resets_at(self):
        """Half 2 of the decided shape: a "resets" substring with no parseable
        phrase must persist NOTHING, not a fabricated now+1h.
        """
        import json
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        reason = 'HTTP 401: your admin resets access quarterly'
        gate._handle_auth_failure(reason, oauth_token='fake-token-a')
        _, _, details_json = captured[0]
        details = json.loads(details_json)
        assert 'resets_at' not in details
        assert details['reason'] == reason

    def test_parseable_resets_in_401_body_persists_real_resets_at(self):
        """Half 1 of the decided shape: the branch is deliberately allowed to
        fire, so a genuinely parseable reset time in a 401 body is surfaced.
        """
        import json
        from datetime import datetime as _dt
        from datetime import timedelta as _td
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        before = _dt.now(UTC)
        gate._handle_auth_failure(
            'HTTP 401: token rejected, quota resets in 3h',
            oauth_token='fake-token-a',
        )
        _, _, details_json = captured[0]
        details = json.loads(details_json)
        assert 'resets_at' in details
        assert details['resets_at'].endswith('+00:00')
        parsed = _dt.fromisoformat(details['resets_at'])
        # Generous window so this cannot flake on the real wall clock.
        assert before + _td(hours=2, minutes=50) <= parsed <= before + _td(hours=3, minutes=10)

    def test_pure_revocation_body_persists_reason_but_no_resets_at(self):
        """End-to-end through the real seam: classifier -> reason renderer ->
        gate. The restored snippet must reach PERSISTED state (the whole point
        of task 4042), and a revocation body carries no reset ETA.
        """
        import json
        gate = _make_gate(['a'])
        captured = self._capture_fired_details(gate)
        result = AgentResult(
            success=False,
            output=(
                '{"type":"error","error":{"type":"authentication_error",'
                '"message":"OAuth token has been revoked"}}'
            ),
            api_error_status=401,
        )
        outcome = classify_invocation(result, strict_confirm=True)
        assert isinstance(outcome, AuthFailed)
        gate._handle_auth_failure(auth_failure_reason(outcome), oauth_token='fake-token-a')
        _, _, details_json = captured[0]
        details = json.loads(details_json)
        assert 'OAuth token has been revoked' in details['reason']
        assert 'resets_at' not in details


#: shared/tests/test_auth_failed.py -> parents[0]=shared/tests, parents[1]=shared,
#: parents[2]=repo root. Mirrors capability_manifest_corpus.REPO_ROOT; correct
#: inside a `.worktrees/<id>` checkout too.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: The ONE module allowed to define — and so to call unqualified — the bare
#: `_parse_resets_at` / `_extract_cap_message` names. Everywhere else the
#: convention (already followed by shared/src/shared/usage_gate.py's
#: `import ... as _parse_resets_at_strict`) is to import under the explicit
#: alias, so a reader can tell at the call site WHICH copy is running.
_STRICT_PARSE_OWNER = 'shared/src/shared/invocation_outcome.py'

#: The names `_STRICT_PARSE_OWNER` single-sources.
_OWNED_NAMES = ('_parse_resets_at', '_extract_cap_message')

#: Roots whose ABSENCE means the walk below is broken rather than the tree
#: merely reorganised: these two held the retired fork and its re-export.
_REQUIRED_SRC_ROOTS = ('shared/src', 'orchestrator/src')


def _production_src_roots() -> list[Path]:
    """Every workspace package's `<pkg>/src` tree.

    DISCOVERED rather than hardcoded: a fixed list covers only the trees
    someone remembered to name, so a re-fork in a package added later — or in
    one simply overlooked, `dashboard/src` being the pointed example, since
    the dashboard is the consumer a fabricated reset time would mislead —
    would sail past a guard still reporting green. Scoped to `<pkg>/src` on
    purpose: `.worktrees/` sits at the REPO root, so unlike a root-level rglob
    (the trap capability_manifest_corpus.py documents) this walk cannot wander
    into a sibling task's checkout.
    """
    roots = sorted(p for p in _REPO_ROOT.glob('*/src') if p.is_dir())
    found = {str(p.relative_to(_REPO_ROOT)) for p in roots}
    # Loud, not silently narrowed: a guard that quietly stops scanning a tree
    # it can no longer find is worse than no guard.
    missing = [rel for rel in _REQUIRED_SRC_ROOTS if rel not in found]
    assert not missing, f'production source root(s) missing under {_REPO_ROOT}: {missing}'
    return roots


class _Site(NamedTuple):
    """One place the scan found an owned name. `detail` is the defined name
    for a definition, the source line for a call."""

    path: str
    lineno: int
    detail: str

    def __str__(self) -> str:
        return f'{self.path}:{self.lineno}: {self.detail}'


class _OwnedNameSites(NamedTuple):
    calls: tuple[_Site, ...]
    definitions: tuple[_Site, ...]


@functools.cache
def _owned_name_sites() -> _OwnedNameSites:
    """Module-level DEFINITIONS and bare-name CALLS of the owned names, across
    every production source tree.

    AST-based, not grep-based: the retired fork and its history are discussed
    at length in comments and docstrings across `usage_gate.py` and
    `invocation_outcome.py`, and a textual scan would count that prose as
    callers. Only `ast.Call` nodes and MODULE-LEVEL `ast.FunctionDef` nodes
    count, so `_parse_resets_at_strict(...)` (a different name) and every
    mention in prose are excluded by construction.

    The substring pre-filter is an optimisation, never a narrowing: every node
    reported here carries an owned name as a literal identifier, so a file
    whose text contains neither name cannot hold one. Measured on this tree,
    it is what makes scanning all 426 production files (0.10s) cheaper than
    parsing the 188 in two trees (1.84s).
    """
    calls: list[_Site] = []
    definitions: list[_Site] = []
    for root in _production_src_roots():
        for path in sorted(root.rglob('*.py')):
            source = path.read_text(encoding='utf-8')
            if not any(name in source for name in _OWNED_NAMES):
                continue
            rel = str(path.relative_to(_REPO_ROOT))
            lines = source.splitlines()
            tree = ast.parse(source, filename=str(path))
            definitions.extend(
                _Site(rel, node.lineno, node.name)
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name in _OWNED_NAMES
            )
            # ast.walk is breadth-first, so a single file's hits are not
            # source-ordered; both lists are sorted before being returned.
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name):
                    called = func.id
                elif isinstance(func, ast.Attribute):
                    called = func.attr
                else:
                    continue
                if called in _OWNED_NAMES:
                    calls.append(_Site(rel, node.lineno, lines[node.lineno - 1].strip()))
    return _OwnedNameSites(tuple(sorted(calls)), tuple(sorted(definitions)))


class TestSingleResetsParserOwnership:
    """`shared.invocation_outcome` is the SINGLE owner of `_parse_resets_at`
    and `_extract_cap_message`.

    WHAT IS MECHANICALLY ENFORCED, stated exactly so the claim and the scan
    cannot drift apart: across EVERY workspace member's `<pkg>/src` tree, no
    module but the owner may define either name at module level or call either
    bare name; and `orchestrator/src/orchestrator/usage_gate.py` — the shim
    that used to re-export both — may not bind either name. Test trees,
    `scripts/` and prose anywhere are deliberately NOT scanned: a copy in one
    of those cannot reach production.

    Task 4042 moved the fabricating `usage_gate._parse_resets_at` fork's ONE
    live call site — `_handle_auth_failure` — onto the non-fabricating
    `shared.invocation_outcome` copy, because restoring the 401/403 body
    snippet re-armed the `'resets' in reason.lower()` branch: a body merely
    CONTAINING the substring "resets" without a parseable phrase would
    otherwise persist an invented `now + 1h` recovery time onto a revoked
    token (PRD 7.1.a — see `TestAuthFailedPersistsResetsAt` above).

    Task 4357 then retired the fork itself, along with its caller-free
    `_extract_cap_message` twin and the orchestrator re-export that kept both
    importable. This class is what keeps the tree converged: with the fork
    gone, the failure mode that would restore the fabrication regression is
    RE-forking — someone re-adding a module-local parser, or re-exporting one
    — and nothing else in the tree catches that.

    Do not relax any assertion here. If a second copy is genuinely wanted,
    that is a design change: argue it, don't let a guard erode.
    """

    def test_scan_finds_the_owner_defining_and_calling_both_names(self):
        # Anti-vacuity: proves the AST walk resolves BOTH node kinds it
        # reports, so a future rename cannot turn the guards below into no-ops
        # that pass because they found nothing at all.
        sites = _owned_name_sites()
        assert sites.calls, 'AST scan found no call of either owned name — guard is vacuous'
        owner_defs = {s.detail for s in sites.definitions if s.path == _STRICT_PARSE_OWNER}
        assert owner_defs == set(_OWNED_NAMES), (
            f'AST scan did not find both owned names defined in {_STRICT_PARSE_OWNER} '
            f'(found {sorted(owner_defs)}) — the definition guard is vacuous.'
        )

    def test_only_the_owner_calls_the_bare_names(self):
        # Exact match on the structured `path`, not a prefix match on a
        # rendered string — and NO assertion on the call COUNT: a legitimate
        # new call inside the owner's own module must not fail this guard.
        offenders = [
            str(site) for site in _owned_name_sites().calls
            if site.path != _STRICT_PARSE_OWNER
        ]
        assert offenders == [], (
            'new production caller(s) of a bare owned name: '
            f'{offenders}. Outside {_STRICT_PARSE_OWNER} the bare names resolve '
            'to nothing at all (NameError) — unless a module-local fork has '
            'been re-introduced, which is the regression this guards: the '
            'retired fork invented `now + 1h` on parse failure. Import the '
            'strict copy as `_parse_resets_at_strict` instead — see '
            'TestAuthFailedPersistsResetsAt for why.'
        )

    def test_handle_auth_failure_uses_the_single_parser(self):
        # Belt-and-braces on the specific regression: the module that owns
        # _handle_auth_failure must reach the strict copy under its alias.
        import shared.usage_gate as usage_gate_module

        assert usage_gate_module._parse_resets_at_strict is invocation_outcome._parse_resets_at

    def test_no_production_module_redefines_the_names(self):
        """No module but the owner may DEFINE either function.

        Scans every production tree rather than only `usage_gate.py` where
        the fork used to live: with the fork gone, a re-fork is as likely to
        appear in whichever module next wants to parse a reset string —
        `dashboard/src/dashboard/data/costs.py`, the consumer a fabricated
        value would mislead, being the pointed candidate.

        An AST `FunctionDef` scan rather than `hasattr`/`vars()`: a module
        legitimately imports the strict parser under an alias, and a name
        bound by `import ... as _parse_resets_at_strict` must not be mistaken
        for a local definition. Scanning defs also catches a re-fork that is
        never imported anywhere — i.e. before it has a caller for
        `test_only_the_owner_calls_the_bare_names` to find.
        """
        refork = [
            str(site) for site in _owned_name_sites().definitions
            if site.path != _STRICT_PARSE_OWNER
        ]
        assert refork == [], (
            f'{refork} re-define(s) a name owned by {_STRICT_PARSE_OWNER}. Both '
            'live ONLY there; a second copy is exactly the un-guarded drift '
            'surface task 4357 removed, and the copy that used to live in '
            'shared/src/shared/usage_gate.py fabricated `now + 1h` on parse '
            'failure.'
        )

    def test_orchestrator_shim_does_not_reexport_the_retired_names(self):
        """The orchestrator shim is public-surface-only.

        `shared.usage_gate.__all__` lists no underscore names, so the star
        import cannot reintroduce these — only an explicit re-export tuple
        could, which is how they used to survive there.

        An AST scan of the FILE rather than `import orchestrator.usage_gate`
        + `hasattr`, for two independent reasons. (1) Layering: `shared/`
        must not import `orchestrator/` (program decision #4 — the same note
        appears in `test_server_error.py` and
        `test_invocation_outcome_boundary.py`). (2) That import is not even
        available here: verify runs `cd shared && uv run pytest` FIRST in its
        chain, which syncs the workspace venv down to `shared`'s own
        dependencies, so `orchestrator` is not installed and the import
        raises `ModuleNotFoundError` — the guard would go red for a reason
        with nothing to do with what it guards, and go green again only if
        some earlier command happened to leave an `--all-packages` venv
        behind. Reading the source keeps the assertion true of the file,
        which is what "does not re-export" actually means.
        """
        path = _REPO_ROOT / 'orchestrator/src/orchestrator/usage_gate.py'
        assert path.is_file(), f'orchestrator usage_gate shim missing: {path}'
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        bound: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                # A `from ... import *` alias is literally named '*' and binds
                # nothing statically — it cannot reintroduce these two, since
                # shared.usage_gate.__all__ lists no underscore names.
                bound.update(alias.asname or alias.name for alias in node.names)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                bound.add(node.name)
            elif isinstance(node, ast.Assign):
                bound.update(t.id for t in node.targets if isinstance(t, ast.Name))
        leaked = sorted(bound & set(_OWNED_NAMES))
        assert leaked == [], (
            f'orchestrator/src/orchestrator/usage_gate.py re-exports {leaked}. '
            f'Both live ONLY in {_STRICT_PARSE_OWNER}; re-exporting them from a '
            'second module is what let the retired fabricating fork stay '
            'importable.'
        )


@pytest.mark.asyncio
class TestBeforeInvokeSkipsAuthFailed:

    async def test_before_invoke_skips_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        lease = await gate.before_invoke()
        assert lease is not None
        assert lease.token == 'fake-token-b'

    async def test_before_invoke_blocks_when_all_auth_failed(self):
        gate = _make_gate(['a'])
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.before_invoke(), timeout=0.1)


class TestIsPausedIncludesAuthFailed:

    def test_is_paused_true_when_all_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        gate._accounts[1].auth_failed = True
        assert gate.is_paused is True

    def test_is_paused_false_when_one_account_healthy(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        # b is healthy
        assert gate.is_paused is False

    def test_is_paused_true_when_mixed_cap_and_auth_failed(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].capped = True
        gate._accounts[1].auth_failed = True
        assert gate.is_paused is True


@pytest.mark.asyncio
class TestAuthReprobeReReadsEnv:
    """Re-probe loop must call load_dotenv(override=True) and re-read
    os.environ so a refreshed .env picks up the new token."""

    async def test_reprobe_updates_token_from_env(self):
        env_var = 'TEST_AUTH_TOKEN_A'
        with patch.dict(os.environ, {env_var: 'old-token'}):
            config = UsageCapConfig(
                accounts=[AccountConfig(name='a', oauth_token_env=env_var)],
                wait_for_reset=False,
                auth_reprobe_secs=0,
            )
            gate = UsageGate(config)

        gate._run_probe = AsyncMock(return_value=True)
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        with (
            patch('shared.usage_gate.load_dotenv') as mock_load,
            patch.dict(os.environ, {env_var: 'new-token'}, clear=False),
        ):
            await gate._reprobe_account(gate._accounts[0])

        mock_load.assert_called_once()
        # load_dotenv is called with override=True
        call_kwargs = mock_load.call_args.kwargs
        assert call_kwargs.get('override') is True
        # Token refreshed from env
        assert gate._accounts[0].token == 'new-token'
        # Probe succeeded → auth_failed cleared
        assert gate._accounts[0].auth_failed is False

    async def test_reprobe_failure_keeps_auth_failed(self):
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        gate._run_probe = AsyncMock(return_value=False)
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._reprobe_account(gate._accounts[0])

        assert gate._accounts[0].auth_failed is True


@pytest.mark.asyncio
class TestAuthReprobeSpawnFault:
    """The auth path must classify a spawn fault exactly as the resume path does.

    Task 4512. `_reprobe_account` had the same blindness `_run_probe` did: on a
    failed probe it logged "auth re-probe failed - staying auth_failed", a
    sentence that ASSERTS a probe ran and came back negative. With `claude`
    unresolvable no probe ran at all, so that line reported evidence about the
    token that had never been gathered — pointing an operator at credentials
    when the actual fault was the host.

    Both probe callers share one accounting surface on purpose: an operator
    debugging a fleet where some accounts are capped and others auth_failed
    should see ONE infrastructure fault, not two unrelated-looking symptoms.
    """

    async def test_spawn_fault_does_not_propagate_and_is_recorded(self, caplog):
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with (
            caplog.at_level(logging.DEBUG, logger='shared.usage_gate'),
            patch('shared.usage_gate.load_dotenv'),
        ):
            await gate._reprobe_account(acct)

        assert acct.probe_spawn_failures == 1
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, 'a spawn fault on the auth path must reach ERROR'
        assert 'claude' in ' '.join(r.getMessage() for r in errors)

    async def test_spawn_fault_does_not_claim_the_probe_ran(self, caplog):
        """The misleading line must not be emitted — and the honest one must be.

        "auth re-probe failed - staying auth_failed" is a claim about the
        token. Emitting it when nothing was spawned is what sends an operator
        to re-issue credentials for a host problem.

        Deliberately a PAIR of assertions (task 4512 review). A negative
        substring check on log prose is unfalsifiable on its own: merely
        REWORDING the line in `_reprobe_account` would satisfy it while the
        behaviour went away. Pairing it with a positive assertion that the
        ERROR-level spawn-fault record WAS emitted means a rewording turns
        the pair red rather than silently green — one of the two substrings
        has to survive.
        """
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with (
            caplog.at_level(logging.DEBUG, logger='shared.usage_gate'),
            patch('shared.usage_gate.load_dotenv'),
        ):
            await gate._reprobe_account(acct)

        assert not [
            r for r in caplog.records if 'staying auth_failed' in r.getMessage()
        ]
        spawn_errors = [
            r for r in caplog.records
            if r.levelno >= logging.ERROR and 'INFRASTRUCTURE FAULT' in r.getMessage()
        ]
        assert len(spawn_errors) == 1, (
            'the probe-ran claim is suppressed, but nothing took its place: '
            'the operator is now told less, not something truer'
        )
        assert acct.probe_spawn_failures == 1

    async def test_spawn_fault_is_not_read_as_recovery(self):
        """No AUTH_FAILED -> AVAILABLE edge, no auth_resumed event.

        An account we could not VERIFY must not be treated as verified. This
        is the fail-visible direction: staying blocked on an unverifiable
        account is recoverable; unblocking it burns real invocations against
        the same broken host.
        """
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with (
            patch('shared.usage_gate.load_dotenv'),
            patch.object(gate, '_fire_cost_event') as mock_fire,
        ):
            await gate._reprobe_account(acct)

        assert acct.phase == AccountPhase.AUTH_FAILED
        assert 'auth_resumed' not in [c[0][1] for c in mock_fire.call_args_list]

    async def test_auth_reprobe_loops_broad_handler_never_sees_it(self, caplog):
        """`_auth_reprobe_loop`'s `except Exception` must not be the catcher.

        That handler logs a WARNING "auth re-probe raised" and retries, which
        would downgrade the one signal saying this is NOT an auth problem back
        into generic noise. Catching in `_reprobe_account` is what keeps the
        classification.

        Pinned on STATE and LEVEL, not only on log prose (task 4512 review).
        `probe_spawn_failures` advancing once per `_reprobe_account` call is
        positive proof the INNER handler ran to completion on each iteration —
        the broad handler catching instead would skip `_note_probe_spawn_failure`
        entirely and leave the count behind. "No WARNING record at all" is the
        level-based twin of the substring check, and unlike the substring it
        survives any rewording of the loop's message.
        """
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        original_sleep = asyncio.sleep
        sleeps = 0

        async def capture_sleep(duration: float) -> None:
            # Clear on the THIRD sleep, not the first: the loop checks
            # auth_failed immediately after sleeping, so clearing on the first
            # would return before _reprobe_account ever runs and make this
            # test vacuously green. Two full iterations (not one) are what
            # make "once per call" an assertion about the per-call handler
            # rather than about a single increment.
            nonlocal sleeps
            sleeps += 1
            if sleeps >= 3:
                acct.auth_failed = False
            await original_sleep(0)

        with (
            caplog.at_level(logging.DEBUG, logger='shared.usage_gate'),
            patch('shared.usage_gate.load_dotenv'),
            patch('asyncio.sleep', side_effect=capture_sleep),
        ):
            await asyncio.wait_for(gate._auth_reprobe_loop(acct), timeout=5)

        assert gate._run_probe.await_count == 2
        assert acct.probe_spawn_failures == 2, (
            'the inner handler in _reprobe_account must run on every '
            'iteration; a count short of the call count means the loop-level '
            "`except Exception` caught one and skipped the fault's accounting"
        )
        assert not [
            r for r in caplog.records if r.levelno == logging.WARNING
        ], (
            'a spawn fault on the auth path must never be narrated at '
            'WARNING — that is the level the broad handler downgrades it to'
        )
        assert not [
            r for r in caplog.records if 'auth re-probe raised' in r.getMessage()
        ]

    async def test_auth_path_latches_the_same_gate_level_fault(self):
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(side_effect=_spawn_fault())

        with patch('shared.usage_gate.load_dotenv'):
            for _ in range(_SPAWN_FAULT_THRESHOLD):
                await gate._reprobe_account(acct)

        assert gate.probe_infra_fault is not None

    # --- CONTROL: a probe that RAN keeps every existing behaviour --------

    async def test_probe_that_ran_and_failed_still_says_staying_auth_failed(self, caplog):
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(return_value=False)

        with (
            caplog.at_level(logging.DEBUG, logger='shared.usage_gate'),
            patch('shared.usage_gate.load_dotenv'),
        ):
            await gate._reprobe_account(acct)

        assert [r for r in caplog.records if 'staying auth_failed' in r.getMessage()]
        assert acct.phase == AccountPhase.AUTH_FAILED
        assert gate.probe_infra_fault is None
        assert acct.probe_spawn_failures == 0

    async def test_probe_that_ran_and_succeeded_still_resumes(self):
        gate = _make_gate(['a'], auth_reprobe_secs=0)
        acct = gate._accounts[0]
        acct.auth_failed = True
        acct.auth_failed_at = datetime.now(UTC)
        gate._cost_store = AsyncMock()
        gate._run_probe = AsyncMock(return_value=True)

        with (
            patch('shared.usage_gate.load_dotenv'),
            patch.object(gate, '_fire_cost_event') as mock_fire,
        ):
            await gate._reprobe_account(acct)

        assert acct.auth_failed is False
        assert 'auth_resumed' in [c[0][1] for c in mock_fire.call_args_list]


@pytest.mark.asyncio
class TestSighupTriggersReprobe:
    """SIGHUP handler reloads tokens and probes ALL accounts (not just auth_failed)."""

    async def test_sighup_handler_reprobes_every_account(self):
        gate = _make_gate(['a', 'b'])
        gate._accounts[0].auth_failed = True
        gate._accounts[0].auth_failed_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(return_value=True)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        # Both accounts probed; auth_failed cleared on the failing one.
        assert gate._run_probe.await_count == 2
        assert gate._accounts[0].auth_failed is False
        assert gate._accounts[1].auth_failed is False

    async def test_sighup_probes_even_when_no_auth_failed(self):
        gate = _make_gate(['a'])
        gate._run_probe = AsyncMock(return_value=True)
        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()
        # New semantics: SIGHUP always probes every account.
        gate._run_probe.assert_awaited_once()


@pytest.mark.asyncio
class TestSighupClearsAllBlockedState:
    """SIGHUP must put every account back in a probe-worthy state."""

    async def test_sighup_clears_capped_state(self):
        gate = _make_gate(['a'])
        gate._accounts[0].capped = True
        gate._accounts[0].resets_at = datetime.now(UTC)
        gate._accounts[0].pause_started_at = datetime.now(UTC)
        gate._run_probe = AsyncMock(return_value=True)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        assert gate._accounts[0].capped is False
        assert gate._accounts[0].resets_at is None
        assert gate._accounts[0].pause_started_at is None
        gate._run_probe.assert_awaited_once()

    async def test_sighup_refreshes_token_from_env(self):
        env_var = 'TEST_AUTH_TOKEN_A'
        with patch.dict(os.environ, {env_var: 'old-token'}):
            config = UsageCapConfig(
                accounts=[AccountConfig(name='a', oauth_token_env=env_var)],
                wait_for_reset=False,
                auth_reprobe_secs=0,
            )
            gate = UsageGate(config)
        gate._run_probe = AsyncMock(return_value=True)
        assert gate._accounts[0].token == 'old-token'

        with (
            patch('shared.usage_gate.load_dotenv'),
            patch.dict(os.environ, {env_var: 'new-token'}, clear=False),
        ):
            await gate._on_sighup_async()

        assert gate._accounts[0].token == 'new-token'

    async def test_sighup_reopens_global_gate(self):
        gate = _make_gate(['a', 'b'])
        gate._open.clear()
        gate._accounts[0].capped = True
        gate._accounts[1].capped = True
        gate._run_probe = AsyncMock(return_value=True)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        assert gate._open.is_set() is True
        assert gate._accounts[0].capped is False
        assert gate._accounts[1].capped is False

    async def test_sighup_clears_probe_lifecycle_state(self):
        gate = _make_gate(['a'])
        gate._accounts[0].probing = True
        gate._accounts[0].probe_in_flight = True
        gate._accounts[0].probe_count = 5
        gate._accounts[0].near_cap = True
        gate._run_probe = AsyncMock(return_value=True)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        assert gate._accounts[0].probing is False
        assert gate._accounts[0].probe_in_flight is False
        assert gate._accounts[0].probe_count == 0
        assert gate._accounts[0].near_cap is False

    async def test_sighup_preserves_cumulative_cost(self):
        """cumulative_cost is a budget counter — SIGHUP must not reset it."""
        gate = _make_gate(['a'])
        gate._cumulative_cost = 12.34
        gate._run_probe = AsyncMock(return_value=True)

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        assert gate._cumulative_cost == 12.34

    async def test_sighup_cancels_in_flight_resume_tasks(self):
        gate = _make_gate(['a'])
        gate._run_probe = AsyncMock(return_value=True)

        async def _hang():
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                raise

        resume_task = asyncio.create_task(_hang())
        reprobe_task = asyncio.create_task(_hang())
        gate._accounts[0].resume_task = resume_task
        gate._accounts[0].auth_reprobe_task = reprobe_task

        with patch('shared.usage_gate.load_dotenv'):
            await gate._on_sighup_async()

        # Allow the cancelled tasks to settle.
        await asyncio.sleep(0)
        assert resume_task.cancelled() or resume_task.done()
        assert reprobe_task.cancelled() or reprobe_task.done()


@pytest.mark.asyncio
class TestRegisterSignalHandlersIdempotent:
    """register_signal_handlers must be safe to call multiple times."""

    async def test_idempotent_double_registration(self):
        gate = _make_gate(['a'])
        # First call: should install (we're inside an asyncio test loop).
        gate.register_signal_handlers()
        assert gate._sighup_handler_installed is True
        # Second call: must be a no-op, no exception.
        gate.register_signal_handlers()
        assert gate._sighup_handler_installed is True
