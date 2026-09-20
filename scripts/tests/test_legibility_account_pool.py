"""Tests for scripts/legibility/account_pool.py — the shared multi-account
UsageGate, wired to the legibility ``invoke=`` seam (task 5488).

WHAT THIS MODULE IS FOR, and therefore what these tests pin: the trickle
used to ride whatever login ``~/.claude`` happened to hold, so one capped
account deferred an entire night while six other accounts sat idle. The
pool turns the gate's roster into the ``(prompt, model) -> str`` callable
the seam already speaks.

THE FAKE GATE IS THE POINT, not a shortcut. ``account_pool`` depends on
seven members of the real 3004-line ``UsageGate`` — ``try_lease``,
``detect_cap_hit``, ``confirm_account_ok``, ``on_agent_complete``,
``release_probe_slot``, ``account_count`` and ``active_account_name`` — and
stating exactly those here is how the test
says what the interface IS rather than reaching through it into gate
internals (docs/code-quality.md: tests that reach a module's internals are
an interface-design smell). The leases it hands out are REAL
``AccountLease`` objects and the slot wrapping them is the REAL
``InvokeSlot``, so the probe-claim discipline under test is the production
one, not a lookalike.

The LLM is ALWAYS mocked here: every test injects an ``invoke`` stub. The
one test that drives a real ``claude`` is marked ``integration`` and is
deselected by default (``addopts = -m 'not integration'``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import account_pool as mod
import pytest

# The `legibility.` spelling, matching account_pool's own import: a bare
# `import coder` is a SECOND module object whose `CoderCapExhausted` is a
# different class from the one the pool raises, so every `pytest.raises`
# below would stop matching what it is supposed to catch.
from legibility import coder as coder_mod
from shared.usage_gate import AccountLease

from shared import cap_markers
from shared import usage_gate as usage_gate_mod


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot and restore ``os.environ`` around every test in this module.

    Every ``build_pool`` call pops ``ANTHROPIC_API_KEY`` and its
    ``load_dotenv`` sets variables, both directly on ``os.environ``;
    ``monkeypatch`` reverses neither, so without this one test's effects
    would reach every later test in the process.
    """
    before = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)


# ---------------------------------------------------------------------------
# The narrow gate interface account_pool consumes, hand-rolled.
# ---------------------------------------------------------------------------


class FakeAccount:
    def __init__(self, name, token, *, capped=False):
        self.name = name
        self.token = token
        self.capped = capped
        # Annotation-only, exactly as on the real AccountState: a near-cap
        # account is NOT capped and keeps serving turns. The two flags are
        # separate here because the whole near-cap defect below lives in the
        # gap between them.
        self.near_cap = False
        self.generation = 0


class FakeGate:
    """Exactly the seven members ``account_pool`` calls, and nothing else.

    ``try_lease`` reproduces the real gate's first-fit walk and its
    ``reverse`` / ``exclude`` knobs; ``detect_cap_hit`` reproduces the STRICT
    detector's contract with a scripted verdict, so a test can drive every
    branch the caller must handle:

    * ``cap_verdict=True``  — a CAP HIT: returns True AND marks the account
      capped, the ``_handle_cap_detected`` route.
    * ``cap_verdict=False`` — no cap: returns False, and the caller must NOT
      rotate.
    * ``cap_verdict='near'`` — a NEAR-CAP warning: returns True and takes NO
      phase transition at all, because ``_handle_near_cap_warning`` is
      annotation-only by design. This is the mode the suite was missing, and
      the gap it left is the whole reason rotation cannot infer its bound
      from the gate's handlers: a True verdict does not imply a shrinking
      admissible set.
    """

    def __init__(self, accounts, *, cap_verdict: bool | Literal['near'] = True):
        self.accounts = list(accounts)
        self.cap_verdict = cap_verdict
        self.lease_calls = []
        self.detect_calls = []
        self.confirmed = []
        self.released = []
        self.costs = []

    # -- selection ---------------------------------------------------------
    def try_lease(self, *, scope=None, reverse=False, exclude=None):
        # SNAPSHOT the exclusion, never the caller's own set: the caller
        # passes a set it goes on mutating, so recording the object itself
        # would make every entry alias the final value and an assertion
        # about GROWTH would read as an assertion about nothing.
        self.lease_calls.append({
            'scope': scope, 'reverse': reverse,
            'exclude': None if exclude is None else set(exclude),
        })
        roster = reversed(self.accounts) if reverse else self.accounts
        for acct in roster:
            if acct.capped:
                continue
            if exclude and acct.name in exclude:
                continue
            return AccountLease(
                name=acct.name, token=acct.token, generation=acct.generation,
            )
        return None

    # -- the strict cap detector ------------------------------------------
    def detect_cap_hit(self, stderr, output, backend='claude', *,
                       oauth_token=None, scope=None):
        self.detect_calls.append({
            'stderr': stderr, 'output': output,
            'oauth_token': oauth_token, 'scope': scope,
        })
        if not self.cap_verdict:
            return False
        for acct in self.accounts:
            if acct.token == oauth_token:
                if self.cap_verdict == 'near':
                    acct.near_cap = True   # annotation only — NOT capped
                else:
                    acct.capped = True
        return True

    @property
    def account_count(self):
        return len(self.accounts)

    @property
    def active_account_name(self):
        """The gate's own "is anyone still usable" predicate — the first
        non-capped, non-auth-failed account, or None.

        Part of the consumed interface because the exhaustion reason has to
        tell "every account is genuinely capped" apart from "every account
        refused this digest while the gate still considers one live". The
        gate already publishes that answer; recomputing it by walking a
        private roster would be the interface smell, not a shortcut.
        """
        for acct in self.accounts:
            if not acct.capped:
                return acct.name
        return None

    # -- settle surface ----------------------------------------------------
    def confirm_account_ok(self, oauth_token):
        self.confirmed.append(oauth_token)

    def on_agent_complete(self, cost_usd):
        self.costs.append(cost_usd)

    def release_probe_slot(self, oauth_token):
        self.released.append(oauth_token)

    # -- test conveniences (NOT part of the consumed interface) ------------
    def account_named(self, name):
        return next(a for a in self.accounts if a.name == name)


def _pool(*specs):
    """FakeGate over (name, capped) pairs, tokens derived from the name."""
    return FakeGate([
        FakeAccount(name, f'tok-{name}', capped=capped) for name, capped in specs
    ])


class _RecordingInvoke:
    """An ``_invoke_cli``-shaped stub that records the token it was handed."""

    def __init__(self, *, replies=None, raises=None, default_reply='{"matches": []}'):
        self.replies = replies or {}
        self.raises = raises or {}
        self.default_reply = default_reply
        self.calls = []

    def __call__(self, prompt, model, *, oauth_token=None, **kwargs):
        self.calls.append({
            'prompt': prompt, 'model': model, 'oauth_token': oauth_token,
            **kwargs,
        })
        if oauth_token in self.raises:
            raise self.raises[oauth_token]
        return self.replies.get(oauth_token, self.default_reply)


# ---------------------------------------------------------------------------
# step-9: the ruling's FIRST verify case — a digest lands on a live account
# even when the accounts ahead of it are capped.
# ---------------------------------------------------------------------------

def test_pool_invoke_lands_the_digest_on_the_uncapped_account():
    gate = _pool(('max-b', True), ('max-c', True), ('max-d', False))
    invoke = _RecordingInvoke(replies={'tok-max-d': '{"matches": [], "candidates": []}'})

    call = mod.pool_invoke(gate, invoke=invoke)
    out = call('the digest prompt', 'haiku')

    assert len(invoke.calls) == 1, (
        f'exactly one CLI invocation for one digest; got {invoke.calls}'
    )
    assert invoke.calls[0]['oauth_token'] == 'tok-max-d', (
        f"the capped accounts must be skipped by the GATE's selection, not "
        f"by the caller; got {invoke.calls[0]['oauth_token']!r}"
    )
    assert out == '{"matches": [], "candidates": []}', (
        f'the CLI reply is returned VERBATIM — the pool is a transport, and '
        f'a transport that reshapes its payload would fabricate; got {out!r}'
    )


def test_pool_invoke_speaks_the_legibility_seams_exact_signature():
    """``(prompt, model) -> str``, positionally — this callable is handed
    straight to ``coder.code_digests(invoke=...)``, which calls it with two
    positional arguments and nothing else."""
    gate = _pool(('max-b', False))
    invoke = _RecordingInvoke()

    call = mod.pool_invoke(gate, invoke=invoke)
    out = call('prompt text', 'haiku')

    assert isinstance(out, str)
    assert invoke.calls[0]['prompt'] == 'prompt text'
    assert invoke.calls[0]['model'] == 'haiku'


def test_pool_invoke_drains_from_the_end_by_default():
    """reverse=True is the DEFAULT for this caller: the trickle drains h→b
    so its 33 one-shots do not contend with the orchestrator's b→h
    first-available order."""
    gate = _pool(('max-b', False), ('max-c', False), ('max-d', False))
    invoke = _RecordingInvoke()

    mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert gate.lease_calls[0]['reverse'] is True, gate.lease_calls
    assert invoke.calls[0]['oauth_token'] == 'tok-max-d'


def test_pool_invoke_reverse_is_overridable():
    gate = _pool(('max-b', False), ('max-c', False))
    invoke = _RecordingInvoke()

    mod.pool_invoke(gate, reverse=False, invoke=invoke)('prompt', 'haiku')

    assert gate.lease_calls[0]['reverse'] is False
    assert invoke.calls[0]['oauth_token'] == 'tok-max-b'


def test_pool_invoke_confirms_the_account_on_success():
    """A successful turn must settle the slot through the gate, or a probe
    slot claimed at selection leaks and that account is never leased
    again — the pool would shrink by one account per probe."""
    gate = _pool(('max-b', False))
    invoke = _RecordingInvoke()

    mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert gate.confirmed == ['tok-max-b'], (
        f'the leased account must be confirmed OK on success; got {gate.confirmed}'
    )
    assert gate.released == ['tok-max-b'], (
        f"the finally must release the probe claim on the success path too; "
        f"got {gate.released}"
    )


def test_pool_invoke_defaults_to_the_real_coder_seam():
    """The default ``invoke`` is coder's own ``_invoke_cli`` — the one real
    subprocess boundary. Pinned by identity rather than by calling it, so
    this test never spawns anything."""
    assert mod.pool_invoke.__defaults__ is None  # keyword-only, by design
    gate = _pool(('max-b', False))
    # Constructing the closure must not invoke anything.
    assert callable(mod.pool_invoke(gate))
    assert mod._DEFAULT_INVOKE is coder_mod._invoke_cli


# ---------------------------------------------------------------------------
# step-11: the ruling's SECOND verify case — a blocking banner MARKS THE
# ACCOUNT CAPPED and the SAME digest completes on the next account.
#
# Within a digest, not across digests, and that is a deliberate reading of
# the existing code rather than of the task's prose: CoderCapExhausted's
# docstring defines capped as "there is no headroom left to code this
# digest", and both coder.is_cap_deferral and nightly's DEFERRED summary
# read it as "the CLI never looked at this digest". If one account's banner
# set capped=True, that predicate would silently weaken to "the account I
# happened to draw was out", and a night with six live accounts could still
# read as DEFERRED — turning the deferral branch into a place real failures
# hide.
#
# Rotation is driven ONLY by the gate's STRICT detector. coder's loose
# OR-substring matcher keeps its own job (labelling an already-failed
# invocation as a per-digest defer); if the strict prefix-AND-confirm policy
# does not agree, the original exception propagates unrotated. A loose false
# positive can therefore re-label one digest and can never burn the pool —
# which is exactly the split shared/src/shared/cap_markers.py argues for.
# ---------------------------------------------------------------------------

def _cap_exhausted(marker='weekly limit', *, stdout='', stderr=''):
    return coder_mod.CoderCapExhausted(
        f'claude CLI exited 1 (...): stdout={stdout!r} stderr={stderr!r}',
        marker=marker, stdout=stdout, stderr=stderr,
    )


def test_a_banner_caps_that_account_and_the_same_digest_completes_next_door():
    banner = 'Claude usage limit reached. Your limit will reset at 3pm.'
    gate = _pool(('max-b', False), ('max-c', False))
    invoke = _RecordingInvoke(
        # reverse=True leases max-c first; it banners, max-b then answers.
        raises={'tok-max-c': _cap_exhausted(stdout=banner, stderr='')},
        replies={'tok-max-b': '{"matches": [], "candidates": []}'},
    )

    out = mod.pool_invoke(gate, invoke=invoke)('the digest prompt', 'haiku')

    # (a) The SAME digest was retried, not abandoned, and it was retried on
    #     the OTHER account.
    assert [c['oauth_token'] for c in invoke.calls] == ['tok-max-c', 'tok-max-b'], (
        f'the same digest must be retried on the next account; got '
        f'{[c["oauth_token"] for c in invoke.calls]}'
    )
    assert [c['prompt'] for c in invoke.calls] == ['the digest prompt'] * 2, (
        'the retry must carry the SAME prompt — a different one would be '
        'coding a different digest'
    )

    # (b) The gate's STRICT detector decided it, and was handed the two
    #     streams SEPARATELY, as structured data.
    assert len(gate.detect_calls) == 1, gate.detect_calls
    assert gate.detect_calls[0]['oauth_token'] == 'tok-max-c', (
        'the verdict must be attributed to the account that banner came '
        'from, never to whichever account is current'
    )
    assert gate.detect_calls[0]['output'] == banner
    assert gate.detect_calls[0]['stderr'] == ''

    # (c) The account is now capped, so the rest of the night skips it.
    assert gate.account_named('max-c').capped is True
    assert gate.account_named('max-b').capped is False

    # (d) Nothing was fabricated: the second account's real reply came back.
    assert out == '{"matches": [], "candidates": []}'

    # (e) every leased account's claim is handed back. A set, not a list:
    #     InvokeSlot.detect_cap_hit also releases max-c on its True verdict,
    #     and how often the gate is told is the gate's business, not this
    #     module's.
    assert set(gate.released) == {'tok-max-c', 'tok-max-b'}, gate.released


def test_a_loose_false_positive_propagates_unrotated():
    """THE guard that keeps a loose matcher from burning the pool.

    coder's loose gate fired (so the exception is a CoderCapExhausted), but
    the gate's strict prefix-AND-confirm policy did NOT verdict a cap. The
    original exception must propagate untouched and NO second account may be
    leased — the loose verdict can re-label this one digest and nothing
    more. This is the case where healthy model output quotes a cap-themed
    session, which in this repo's codebook is common rather than exotic.
    """
    gate = _pool(('max-b', False), ('max-c', False))
    gate.cap_verdict = False          # strict detector disagrees
    original = _cap_exhausted(stdout='a digest that merely QUOTES a limit banner')
    invoke = _RecordingInvoke(raises={'tok-max-c': original})

    with pytest.raises(coder_mod.CoderCapExhausted) as excinfo:
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert excinfo.value is original, (
        'the ORIGINAL exception must propagate — re-wrapping would lose the '
        'marker and the streams the journal reads'
    )
    assert len(invoke.calls) == 1, (
        f'no second account may be leased when the strict detector says this '
        f'was not a cap; got {invoke.calls}'
    )
    assert gate.account_named('max-c').capped is False, (
        'a loose false positive must never mark an account capped — the '
        'consequence there is account failover (cap_markers docstring)'
    )
    assert len(gate.lease_calls) == 1
    assert gate.released == ['tok-max-c'], (
        f'the finally must release the probe claim even when the original '
        f'exception propagates unrotated; got {gate.released}'
    )


def test_an_ordinary_failure_never_consults_the_gate_at_all():
    """A plain CoderInvocationError is not a capacity signal. It propagates
    immediately, without a cap verdict and without rotation: the account is
    fine, this digest is not."""
    gate = _pool(('max-b', False), ('max-c', False))
    boom = coder_mod.CoderInvocationError(
        'claude CLI exited 1: the model backend is down',
        stdout='', stderr='backend down',
    )
    invoke = _RecordingInvoke(raises={'tok-max-c': boom})

    with pytest.raises(coder_mod.CoderInvocationError) as excinfo:
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert excinfo.value is boom
    assert not isinstance(excinfo.value, coder_mod.CoderCapExhausted)
    assert gate.detect_calls == [], (
        f'an ordinary failure must not be offered to the cap detector at '
        f'all; got {gate.detect_calls}'
    )
    assert len(invoke.calls) == 1
    assert gate.account_named('max-c').capped is False
    assert gate.released == ['tok-max-c'], (
        f'the finally must release the probe claim even for a failure the '
        f'gate never saw; got {gate.released}'
    )


def test_a_bare_exception_still_releases_the_probe_claim_and_never_rotates():
    """The `finally` covers EVERY exit path, not only the two coder
    exceptions this module knows how to interpret. A bare ``ValueError`` is
    not caught by ``except coder.CoderCapExhausted`` at all, so this is the
    one case that proves ``finally: gate.release_probe_slot(...)`` itself
    releases the claim rather than one of the `except` arms doing it.
    """
    gate = _pool(('max-b', False), ('max-c', False))
    boom = ValueError('unexpected')
    invoke = _RecordingInvoke(raises={'tok-max-c': boom})

    with pytest.raises(ValueError) as excinfo:
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert excinfo.value is boom
    assert len(invoke.calls) == 1, (
        f'no rotation on an exception the pool does not interpret at all; '
        f'got {invoke.calls}'
    )
    assert gate.released == ['tok-max-c'], (
        f'the leaked-claim invariant holds for ANY exception, not only the '
        f'two coder ones; got {gate.released}'
    )
    assert gate.account_named('max-c').capped is False, (
        'a bare exception is not a cap signal'
    )


def test_rotation_walks_the_whole_pool_before_giving_up():
    """Three banners, three caps, then the fourth account answers. The loop
    is structurally terminating: each iteration marks exactly one account
    capped, so the admissible set strictly shrinks."""
    gate = _pool(
        ('max-b', False), ('max-c', False), ('max-d', False), ('max-e', False),
    )
    banner = 'Claude usage limit reached.'
    invoke = _RecordingInvoke(
        raises={
            f'tok-{n}': _cap_exhausted(stdout=banner)
            for n in ('max-e', 'max-d', 'max-c')
        },
        replies={'tok-max-b': 'the reply'},
    )

    out = mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert out == 'the reply'
    assert [c['oauth_token'] for c in invoke.calls] == [
        'tok-max-e', 'tok-max-d', 'tok-max-c', 'tok-max-b',
    ]
    assert [a.name for a in gate.accounts if a.capped] == ['max-c', 'max-d', 'max-e']


# ---------------------------------------------------------------------------
# step-13: the ruling's THIRD verify case — an exhausted pool still reaches
# task 4736's exit-0 DEFERRED path, and says WHICH exhaustion it was.
#
# This is the one place the 4736 contract is decided. `capped=True` here now
# means what nightly's summary has always claimed and never actually knew:
# every account in the pool is out. The never-fabricate contract is
# untouched — an exhausted pool yields no record at all, never an empty one.
# ---------------------------------------------------------------------------

def test_an_exhausted_pool_raises_cap_exhausted_without_calling_the_cli():
    """try_lease returns None on the FIRST call. The CLI must never be
    invoked — invoking it with no token is precisely the ~/.claude fallback
    this whole task exists to remove."""
    gate = _pool(('max-b', True), ('max-c', True))
    invoke = _RecordingInvoke()

    with pytest.raises(coder_mod.CoderCapExhausted) as excinfo:
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert invoke.calls == [], (
        f'an exhausted pool must never reach the CLI — a token-less spawn '
        f'would silently ride ~/.claude; got {invoke.calls}'
    )
    assert excinfo.value.marker, (
        'the typed marker is what lets the deferral reason say WHY, the same '
        'way a per-digest cap does'
    )


def test_exhaustion_mid_rotation_also_raises_cap_exhausted():
    """The pool empties DURING a digest: both accounts banner, and the third
    lease finds nothing. Same typed outcome, and no extra CLI call."""
    gate = _pool(('max-b', False), ('max-c', False))
    banner = 'Claude usage limit reached.'
    invoke = _RecordingInvoke(raises={
        'tok-max-b': _cap_exhausted(stdout=banner),
        'tok-max-c': _cap_exhausted(stdout=banner),
    })

    with pytest.raises(coder_mod.CoderCapExhausted):
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert len(invoke.calls) == 2, (
        f'each account gets exactly one try; got {invoke.calls}'
    )
    assert all(a.capped for a in gate.accounts)


def test_exhaustion_reason_names_how_many_accounts_were_capped():
    """'all 2 pool accounts capped' — an operator reading the DEFERRED
    escalation learns the pool size, which is what distinguishes a genuinely
    exhausted fleet from a pool that resolved almost empty."""
    gate = _pool(('max-b', True), ('max-c', True))

    with pytest.raises(coder_mod.CoderCapExhausted) as excinfo:
        mod.pool_invoke(gate, invoke=_RecordingInvoke())('prompt', 'haiku')

    message = str(excinfo.value)
    assert '2' in message, message
    assert 'capped' in message.lower(), message


def test_a_pool_that_resolved_NO_accounts_says_so_instead():
    """The other exhaustion, and a DIFFERENT operator response: 'all accounts
    capped' self-clears at the weekly reset, 'no pool accounts resolved' is a
    config fault that will never clear on its own.

    This is the state UsageGate._init_accounts degrades to when no token env
    var resolves — the very condition that would otherwise silently fall back
    to ~/.claude — so it must be LOUD and distinguishable, not folded into
    the routine one.
    """
    gate = _pool()  # zero accounts

    with pytest.raises(coder_mod.CoderCapExhausted) as excinfo:
        mod.pool_invoke(gate, invoke=_RecordingInvoke())('prompt', 'haiku')

    message = str(excinfo.value).lower()
    assert 'no pool accounts' in message, (
        f'a zero-account pool must not read as "all accounts capped" — that '
        f'would send an operator to wait for a reset that never comes; got '
        f'{message!r}'
    )
    assert 'capped' not in message.split('no pool accounts')[0], message


# ---------------------------------------------------------------------------
# task 5488 / step-29: THE NEAR-CAP ROUTE — a True cap verdict that caps
# NOTHING, and the non-terminating rotation it used to cause.
#
# The suite above never caught this because its FakeGate always capped the
# account whenever its verdict was True. The REAL gate does not: a near-cap
# banner reaches `_handle_near_cap_warning`, which sets `acct.near_cap = True`
# and returns True while taking NO phase transition at all. The account is
# still perfectly admissible, so a rotation that merely re-asked the gate was
# handed the SAME account again — forever, at 03:00, with nothing to observe
# but a unit that never finished.
#
# So termination cannot be inferred from the gate's handler semantics. It has
# to be structural on the caller's side: a growing set of already-tried names,
# passed as `exclude=` to the gate's one selection implementation.
# ---------------------------------------------------------------------------

_NEAR_BANNER = "Approaching your Claude usage limit for this week."


def _near_cap_pool(*specs):
    """A pool whose gate verdicts every banner as a NEAR-cap: True, but no
    account is ever capped by it."""
    gate = _pool(*specs)
    gate.cap_verdict = 'near'
    return gate


class _NeverTwice(_RecordingInvoke):
    """An ``_invoke_cli`` stub that fails LOUDLY the moment one account's
    token is handed to it a second time.

    Deliberately an immediate guard rather than an assertion after the fact.
    The defect being pinned is an infinite loop, and an infinite loop is not
    a wrong value: left to run, it reaches no assertion at all — it grows the
    recorded-call lists until the host runs out of memory, taking the rest of
    the suite with it. Catching the repeat where it happens turns that into
    one line naming the account that was served twice. The
    ``@pytest.mark.timeout`` on these tests stays as the backstop for a
    regression that spins WITHOUT repeating a token.
    """

    def __call__(self, prompt, model, *, oauth_token=None, **kwargs):
        already = [c['oauth_token'] for c in self.calls]
        assert oauth_token not in already, (
            f"account token {oauth_token!r} was handed to the CLI twice for "
            f"one digest — the rotation is not bounded by what it has "
            f"already tried, so it never terminates (calls so far: {already})"
        )
        return super().__call__(prompt, model, oauth_token=oauth_token, **kwargs)


@pytest.mark.timeout(15)
def test_a_near_cap_verdict_everywhere_still_terminates():
    """THE termination property, over a pool where NOTHING ever gets capped.

    Every account banners, every verdict is True, and not one account
    transitions — which is precisely the real gate's near-cap behaviour. The
    digest must still give up after exactly one try per account and raise
    CoderCapExhausted, so the night reaches task 4736's exit-0 DEFERRED path
    instead of hanging the 03:00 unit until the weekly reset.
    """
    gate = _near_cap_pool(('max-b', False), ('max-c', False), ('max-d', False))
    invoke = _NeverTwice(raises={
        f'tok-{name}': _cap_exhausted(marker='approaching', stdout=_NEAR_BANNER)
        for name in ('max-b', 'max-c', 'max-d')
    })

    with pytest.raises(coder_mod.CoderCapExhausted):
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert len(invoke.calls) == gate.account_count == 3, (
        f'exactly one try per account, then stop; got '
        f'{[c["oauth_token"] for c in invoke.calls]}'
    )
    assert not any(a.capped for a in gate.accounts), (
        'the premise this test exists for: a near-cap verdict caps NOTHING, '
        'so the admissible set never shrank and the bound cannot have come '
        'from the gate'
    )
    assert all(a.near_cap for a in gate.accounts), (
        'the gate did record the signal — it simply is not a cap'
    )


@pytest.mark.timeout(15)
def test_a_near_cap_account_rotates_and_the_digest_completes_next_door():
    """Bounding the loop must not COST the rotation, which is the trap in the
    obvious fix.

    "Raise as soon as the gate hands back an account already tried" also
    terminates — and would abandon this digest with a live account sitting
    right there, because the gate has no reason to stop offering a near-cap
    account. Only EXCLUDING the tried names keeps the walk moving on to the
    healthy one.
    """
    gate = _near_cap_pool(('max-b', False), ('max-c', False))
    invoke = _NeverTwice(
        # reverse=True leases max-c first; it near-caps, max-b then answers.
        raises={'tok-max-c': _cap_exhausted(marker='approaching',
                                            stdout=_NEAR_BANNER)},
        replies={'tok-max-b': '{"matches": [], "candidates": []}'},
    )

    out = mod.pool_invoke(gate, invoke=invoke)('the digest prompt', 'haiku')

    assert [c['oauth_token'] for c in invoke.calls] == ['tok-max-c', 'tok-max-b']
    assert [c['prompt'] for c in invoke.calls] == ['the digest prompt'] * 2
    assert out == '{"matches": [], "candidates": []}', (
        f'the healthy account\'s real reply, verbatim — nothing fabricated; '
        f'got {out!r}'
    )
    assert gate.account_named('max-c').capped is False, (
        'a near-cap warning must not cap the account: the gate takes no '
        'phase transition, and neither may the caller'
    )
    assert gate.account_named('max-c').near_cap is True


@pytest.mark.timeout(15)
def test_the_growing_exclusion_is_what_bounds_the_rotation():
    """Not just THAT it terminates — WHY. Each pass asks the gate for a lease
    excluding everything already tried, so the admissible set shrinks by
    construction on the CALLER's side no matter what the gate's handlers do
    with the verdict."""
    gate = _near_cap_pool(('max-b', False), ('max-c', False), ('max-d', False))
    invoke = _NeverTwice(raises={
        f'tok-{name}': _cap_exhausted(marker='approaching', stdout=_NEAR_BANNER)
        for name in ('max-b', 'max-c', 'max-d')
    })

    with pytest.raises(coder_mod.CoderCapExhausted):
        mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    # An omitted exclusion and an empty one mean the same thing to the gate,
    # so normalise rather than over-pin which spelling the first pass uses.
    excludes = [set(c['exclude'] or ()) for c in gate.lease_calls]
    assert excludes == [
        set(), {'max-d'}, {'max-d', 'max-c'}, {'max-d', 'max-c', 'max-b'},
    ], f'the exclusion must grow by the account just tried; got {excludes}'
    assert all(
        previous < nxt
        for previous, nxt in zip(excludes, excludes[1:], strict=False)
    ), f'strictly growing, every pass; got {excludes}'


@pytest.mark.timeout(15)
def test_the_genuine_cap_route_is_unchanged_by_the_bound():
    """The bound is ADDITIVE to real-cap rotation, not a replacement for it.

    A true CAP HIT still caps the account through the gate — that is what
    makes the rest of the night skip it, and what makes nightly's "all
    accounts capped" summary true. The exclusion rides alongside as the
    caller's own bookkeeping; neither mechanism is load-bearing for the
    other.
    """
    gate = _pool(('max-b', False), ('max-c', False))  # cap_verdict=True
    invoke = _NeverTwice(raises={
        'tok-max-c': _cap_exhausted(stdout='Claude usage limit reached.'),
    }, replies={'tok-max-b': 'the reply'})

    out = mod.pool_invoke(gate, invoke=invoke)('prompt', 'haiku')

    assert out == 'the reply'
    assert gate.account_named('max-c').capped is True, (
        'a genuine cap hit must still mark the account capped'
    )
    assert set(gate.lease_calls[1]['exclude'] or ()) == {'max-c'}, (
        'and the caller still excludes what it tried, so the two agree '
        'instead of one depending on the other'
    )


# ---------------------------------------------------------------------------
# task 5488 / step-31: THE DEFERRAL REASON STAYS HONEST on the near-cap route.
#
# Bounding the rotation by the caller's tried set creates a state that could
# not happen before: a digest exhausts the pool while NO account is capped.
# The old text — "all N pool accounts capped" — would then be a fresh lie of
# exactly the class test_main_deferral_summary_stays_honest_about_the_tally
# exists to prevent, and a costly one: it sends an operator to wait for a
# weekly reset that will never come, because there is nothing to reset.
#
# Three reasons, three different operator responses:
#   1. no accounts resolved   -> a config fault; the unit is missing token vars
#   2. all accounts capped    -> expected weather; clears at the weekly reset
#   3. all accounts refused   -> NOT a cap; something else is failing every try
#
# Which of 1/2/3 applies is read off the gate's PUBLIC predicates, never off
# its private roster.
# ---------------------------------------------------------------------------

class _OpaqueGate:
    """A gate that publishes its predicates and NOTHING else — no roster
    attribute at all, public or private.

    This is the structural half of "decided by ``active_account_name``":
    a reason computed by walking ``gate._accounts`` would raise
    AttributeError here instead of answering. Reaching past a published
    answer to recompute it from another module's internals is the smell
    this shuts, and an assertion about the resulting TEXT could not have
    caught it — both spellings produce the same words.
    """

    def __init__(self, *, account_count, active_account_name):
        self._count = account_count
        self._active = active_account_name
        self.lease_calls = []

    def try_lease(self, *, scope=None, reverse=False, exclude=None):
        self.lease_calls.append(set(exclude or ()))
        return None

    def release_probe_slot(self, oauth_token):  # pragma: no cover - never leased
        raise AssertionError('nothing was ever leased')

    @property
    def account_count(self):
        return self._count

    @property
    def active_account_name(self):
        return self._active


def _reason_from(gate, invoke=None):
    """The CoderCapExhausted message ``pool_invoke`` produces for *gate*."""
    with pytest.raises(coder_mod.CoderCapExhausted) as excinfo:
        mod.pool_invoke(gate, invoke=invoke or _RecordingInvoke())('prompt', 'haiku')
    return str(excinfo.value)


@pytest.mark.timeout(15)
def test_an_all_refused_pool_never_claims_a_cap_that_did_not_happen():
    """Reason 3, and the whole point of adding it.

    Every account near-capped, so every account is still perfectly usable as
    far as the gate is concerned. Saying "all 2 pool accounts capped" here
    would be false about the fleet's state AND misdirect the operator to the
    weekly reset. The honest reason names what happened — every account
    tried refused this digest — and names an account the gate still
    considers live, which is the fact that makes "capped" the wrong word.
    """
    gate = _near_cap_pool(('max-b', False), ('max-c', False))
    invoke = _NeverTwice(raises={
        f'tok-{name}': _cap_exhausted(marker='approaching', stdout=_NEAR_BANNER)
        for name in ('max-b', 'max-c')
    })

    message = _reason_from(gate, invoke)

    assert gate.active_account_name is not None, (
        'the premise: the gate still considers an account usable, because a '
        'near-cap warning takes no phase transition'
    )
    assert 'capped' not in message.lower(), (
        f'nothing was capped — claiming otherwise sends an operator to wait '
        f'for a reset that will never come; got {message!r}'
    )
    assert gate.active_account_name in message, (
        f'name the account the gate still considers live: that is the fact '
        f'that distinguishes this from an exhausted fleet; got {message!r}'
    )
    assert '2' in message, f'and how many were tried; got {message!r}'
    assert 'no pool accounts' not in message.lower(), (
        f'nor is this the config fault — accounts resolved fine; got {message!r}'
    )


def test_the_all_capped_reason_is_unchanged():
    """Reason 2, byte-identical. An operator (and
    test_exhaustion_reason_names_how_many_accounts_were_capped) already
    reads this wording."""
    gate = _pool(('max-b', True), ('max-c', True))

    assert 'all 2 pool accounts capped' in _reason_from(gate)


def test_the_all_capped_reason_is_decided_by_the_gates_public_predicate():
    """``active_account_name`` is None iff no non-capped, non-auth-failed
    account remains — the gate's own answer to the question the honesty
    check is asking. The opaque gate has no roster to walk, so this passes
    only if that is genuinely where the answer comes from."""
    gate = _OpaqueGate(account_count=7, active_account_name=None)

    assert 'all 7 pool accounts capped' in _reason_from(gate)


def test_a_live_account_the_gate_would_not_lease_is_not_reported_as_capped():
    """The same fork, from the other side and with nothing tried at all: the
    gate declined to lease (every account probe-in-flight, say) while
    reporting one live. Still not a cap, so still not the capped wording."""
    gate = _OpaqueGate(account_count=7, active_account_name='max-c')

    message = _reason_from(gate)

    assert 'capped' not in message.lower(), message
    assert 'max-c' in message, message


def test_a_pool_that_resolved_no_accounts_still_wins_over_the_other_two():
    """Reason 1 dominates: with zero accounts there is nothing to be capped
    and nothing to be live, and the config fault is the only actionable
    thing to say."""
    assert 'no pool accounts' in _reason_from(_pool()).lower()


# ---------------------------------------------------------------------------
# step-31, the end-to-end gate: the near-cap route reaches task 4736's exit-0
# DEFERRED branch rather than hanging the 03:00 unit.
#
# This is the production outcome the whole task exists to guarantee, so it is
# asserted through the REAL code_digests control flow rather than inferred
# from the pieces.
# ---------------------------------------------------------------------------

def _digest_text(session_id):
    return (
        "---\n"
        f'session: "{session_id}"\n'
        'date: "2026-07-14"\n'
        'agent_class: "interactive"\n'
        "---\n\n"
        f"## User Corrections\n- body marker {session_id}\n"
    )


def _codebook():
    return {
        "version": 2,
        "entries": [
            {
                "id": "one-shot-subagent-contract",
                "title": "Silent no-op one-shot subagent contracts",
                "cause": "Sub-agents are given contracts their runtime cannot honor.",
                "severity": "high",
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [],
            },
        ],
        "candidates": [],
    }


@pytest.mark.timeout(60)
def test_a_near_cap_pool_reads_as_a_cap_deferral_end_to_end():
    """Three digests, two accounts, every invocation near-capping.

    The night must END, and end as a DEFERRAL: `capped` for every digest,
    status "failure", is_cap_deferral True — the exact RunResult shape
    nightly's exit-0 branch keys on. Before the caller-side bound this run
    did not produce a wrong answer, it produced no answer at all: the first
    digest looped until something killed the unit.

    Six CLI calls, not three: the tried set is scoped to ONE digest, so each
    digest starts again from the full roster. That is deliberate — a
    near-cap warning is not a cap, and retiring an account for the night on
    one would throw away headroom the gate never said was gone.
    """
    gate = _near_cap_pool(('max-b', False), ('max-c', False))
    invoke = _RecordingInvoke(raises={
        f'tok-{name}': _cap_exhausted(marker='approaching', stdout=_NEAR_BANNER)
        for name in ('max-b', 'max-c')
    })

    result = coder_mod.code_digests(
        [_digest_text(f"batch-sess-{i}") for i in range(3)], _codebook(),
        project="dark_factory", model="haiku",
        invoke=mod.pool_invoke(gate, invoke=invoke),
    )

    assert result.total == 3
    assert result.capped == 3, (
        f'every digest the pool could not code is labelled capped; got '
        f'{result.capped}'
    )
    assert result.records == [], (
        'the never-fabricate contract holds on this route too: a digest the '
        'CLI never completed yields NO record'
    )
    assert result.status == "failure"
    assert coder_mod.is_cap_deferral(result) is True, (
        "this is the input nightly's exit-0 DEFERRED branch keys on — the "
        "night defers instead of hanging until the weekly reset"
    )
    assert len(invoke.calls) == 6, (
        f'each of the 3 digests tries both accounts afresh — the bound is '
        f'per digest, not per night; got {len(invoke.calls)}'
    )
    assert not any(a.capped for a in gate.accounts), (
        'and not one account was capped along the way'
    )


# ---------------------------------------------------------------------------
# task 5637: THE EXIT-0 BANNER ROUTE — the CLI declines by PRINTING the banner
# and exiting 0, so the banner arrives as a RETURNED reply rather than as a
# raised CoderCapExhausted.
#
# The section above covers the route where the CLI exits non-zero; this is the
# other half of the same weather, and before this task it was not rotated at
# all. `coder.code_digest`'s second scan site turned the unparseable banner
# into `CodingResult(capped=True)` with no rotation and no cap recorded on the
# gate — so the account stayed admissible, and because `tried` is scoped to one
# digest and the pool draws reverse=True, the SAME account was drawn first for
# every subsequent digest. One account emitting exit-0 banners therefore still
# lost the whole night, which is the exact failure the pool exists to end.
#
# The banner text is drawn from `shared.cap_markers`' verbatim real-CLI corpus
# rather than invented, so a future CLI rewording turns these tests red in the
# same sweep as `shared/tests/test_cap_markers.py` and
# `scripts/tests/test_legibility_census.py`.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('banner', cap_markers.REAL_CLI_CAP_HIT_MESSAGES)
def test_a_zero_exit_banner_rotates_and_the_same_digest_completes_next_door(
    banner,
):
    """The exit-0 twin of
    ``test_a_banner_caps_that_account_and_the_same_digest_completes_next_door``.

    The ONE difference from that test is that ``_RecordingInvoke`` RETURNS the
    banner instead of raising it — which is precisely the difference the pool
    used to be blind to. Everything else must hold identically: the gate
    decides, the account is capped, the same digest completes next door, and
    nothing is fabricated.
    """
    gate = _pool(('max-b', False), ('max-c', False))
    invoke = _RecordingInvoke(replies={
        # reverse=True leases max-c first; it banners AT EXIT 0, max-b answers.
        'tok-max-c': banner,
        'tok-max-b': '{"matches": [], "candidates": []}',
    })

    out = mod.pool_invoke(gate, invoke=invoke)('the digest prompt', 'haiku')

    # (a) The SAME digest was retried on the OTHER account.
    assert [c['oauth_token'] for c in invoke.calls] == ['tok-max-c', 'tok-max-b'], (
        f'the same digest must be retried on the next account; got '
        f'{[c["oauth_token"] for c in invoke.calls]}'
    )
    assert [c['prompt'] for c in invoke.calls] == ['the digest prompt'] * 2, (
        'the retry must carry the SAME prompt — a different one would be '
        'coding a different digest'
    )

    # (b) The gate's STRICT detector decided it, and the reply went in as the
    #     OUTPUT stream: there is no stderr on an exit-0 route.
    assert len(gate.detect_calls) == 1, gate.detect_calls
    assert gate.detect_calls[0]['output'] == banner
    assert gate.detect_calls[0]['stderr'] == '', (
        f"an exit-0 reply is the OUTPUT stream and nothing else; got "
        f"{gate.detect_calls[0]['stderr']!r}"
    )
    assert gate.detect_calls[0]['oauth_token'] == 'tok-max-c', (
        'the verdict must be attributed to the account that banner came '
        'from, never to whichever account is current'
    )

    # (c) The account is now capped, so the rest of the night skips it.
    assert gate.account_named('max-c').capped is True
    assert gate.account_named('max-b').capped is False

    # (d) The bannering account must NOT be confirmed healthy. That is the
    #     other half of the defect: `confirm_account_ok` is what clears a
    #     near_cap annotation the gate had already recorded, so confirming an
    #     account that just refused to answer erases the gate's own warning.
    assert gate.confirmed == ['tok-max-b'], (
        f'only the account that actually answered may be confirmed; got '
        f'{gate.confirmed}'
    )

    # (e) Nothing was fabricated: the second account's real reply came back.
    assert out == '{"matches": [], "candidates": []}'

    # (f) A set, not a list: InvokeSlot.detect_cap_hit also releases max-c on
    #     its True verdict, and how often the gate is told is the gate's
    #     business, not this module's.
    assert set(gate.released) == {'tok-max-c', 'tok-max-b'}, gate.released


@pytest.mark.timeout(60)
def test_a_zero_exit_banner_night_is_not_lost_end_to_end():
    """Three digests, two accounts, one of them bannering at exit 0.

    The production outcome the task exists to guarantee, asserted through the
    REAL ``code_digests`` control flow rather than inferred from the pieces:
    every digest is CODED, nothing is labelled capped, and the night is not a
    deferral. Before this, all three digests came back
    ``CodingResult(capped=True)`` from ``code_digest``'s second scan site and
    the night read as DEFERRED with a live account sitting right there.

    FOUR CLI calls, not six, and the count is the assertion that says the
    night was not lost: only the FIRST digest pays a rotation. Unlike a
    near-cap verdict (``test_a_near_cap_pool_reads_as_a_cap_deferral_end_to_end``,
    where the tried set resets per digest and every digest pays), a CapHit
    retires max-c for the night, so digests 2 and 3 lease max-b directly.
    """
    gate = _pool(('max-b', False), ('max-c', False))
    invoke = _RecordingInvoke(replies={
        'tok-max-c': cap_markers.REAL_CLI_CAP_HIT_MESSAGES[0],
        'tok-max-b': '{"matches": [], "candidates": []}',
    })

    result = coder_mod.code_digests(
        [_digest_text(f"batch-sess-{i}") for i in range(3)], _codebook(),
        project="dark_factory", model="haiku",
        invoke=mod.pool_invoke(gate, invoke=invoke),
    )

    assert result.total == 3
    assert len(result.records) == 3, (
        f'every digest must be CODED — the pool had a live account for each '
        f'of them; got {len(result.records)} records, failures '
        f'{result.failures}'
    )
    assert result.capped == 0, (
        f'not one digest may be labelled capped: `capped` means "no headroom '
        f'left anywhere", and one account\'s banner is not that; got '
        f'{result.capped}'
    )
    assert result.status == "ok"
    assert coder_mod.is_cap_deferral(result) is False, (
        "the night must not read as a DEFERRAL — that is the branch where a "
        "night with live accounts used to disappear"
    )
    assert len(invoke.calls) == 4, (
        f'only the first digest pays a rotation; a CapHit retires max-c for '
        f'the night, so digests 2 and 3 lease max-b directly; got '
        f'{[c["oauth_token"] for c in invoke.calls]}'
    )
    assert gate.account_named('max-c').capped is True


# ---------------------------------------------------------------------------
# step-15: build_pool() — a REAL UsageGate from nothing but an accounts file.
#
# The trickle's original excuse for riding ~/.claude was that the
# orchestrator config is unreachable from its interpreter. Only the
# orchestrator YAML is: UsageCapConfig(accounts_file=...) needs nothing else,
# which is the same two-liner fused_memory/config/schema.py already uses.
# ---------------------------------------------------------------------------

_ROSTER_YAML = """\
accounts:
  - name: max-b
    oauth_token_env: CLAUDE_OAUTH_TOKEN_B
  - name: max-c
    oauth_token_env: CLAUDE_OAUTH_TOKEN_C
  - name: max-d
    oauth_token_env: CLAUDE_OAUTH_TOKEN_D
"""


@pytest.fixture
def roster_file(tmp_path):
    path = tmp_path / "usage-accounts.yaml"
    path.write_text(_ROSTER_YAML)
    return path


@pytest.fixture
def empty_env_file(tmp_path):
    """A guaranteed-empty ``.env`` for every ``build_pool`` call that is not
    itself testing dotenv loading.

    ``build_pool``'s default ``env_file=None`` resolves to
    ``_REPO_ROOT / ".env"`` -- inert in a worktree (no such file) but, run
    from the main checkout, a file carrying real ``CLAUDE_OAUTH_TOKEN_*``
    and ``ANTHROPIC_API_KEY``. Passing this fixture instead of leaving
    ``env_file`` unset is what keeps a `build_pool()` call hermetic to which
    checkout the suite happens to run from.
    """
    path = tmp_path / "empty.env"
    path.write_text("")
    return path


def _set_pool_tokens(monkeypatch, *letters):
    for letter in letters:
        monkeypatch.setenv(f"CLAUDE_OAUTH_TOKEN_{letter}", f"tok-{letter.lower()}")


def test_build_pool_resolves_the_roster_in_file_order(
    roster_file, monkeypatch, empty_env_file,
):
    _set_pool_tokens(monkeypatch, "B", "C", "D")

    gate = mod.build_pool(accounts_file=str(roster_file), env_file=str(empty_env_file))

    assert gate.account_count == 3
    assert [a.name for a in gate._accounts] == ["max-b", "max-c", "max-d"], (
        "order is the failover order — config/usage-accounts.yaml says so in "
        "its own header, and reverse= depends on it"
    )


def test_default_accounts_file_resolves_relative_to_this_checkout():
    """Never a hardcoded absolute: a copy of this script running from a
    worktree must read ITS OWN roster, the same reason coder.py resolves
    `shared` __file__-relatively (tasks 2881/2882/3329)."""
    repo_root = Path(mod.__file__).resolve().parents[2]

    assert mod.default_accounts_file() == repo_root / "config" / "usage-accounts.yaml"
    assert mod.default_accounts_file().exists(), (
        "the default must point at a roster that actually exists in this "
        "checkout — a missing accounts_file degrades to an EMPTY pool with "
        "only a warning, which is the silent ~/.claude fallback again"
    )


def test_build_pool_actually_uses_the_default_accounts_file_when_nothing_else_is_set(
    monkeypatch, tmp_path, empty_env_file,
):
    """With neither ``accounts_file`` nor ``USAGE_ACCOUNTS_FILE`` set,
    ``build_pool`` reads ``default_accounts_file()`` -- the test above never
    calls ``build_pool``. The roster's one account exists in no other roster, so
    only a pool built from ``default_accounts_file()`` can resolve it -- a
    ``max-*`` name would also resolve from the real
    ``config/usage-accounts.yaml`` and prove nothing.
    """
    roster = tmp_path / "default-roster.yaml"
    roster.write_text(
        "accounts:\n"
        "  - name: only-in-the-default-roster\n"
        "    oauth_token_env: CLAUDE_OAUTH_TOKEN_DEFAULT_ROSTER_PROBE\n"
    )
    monkeypatch.delenv("USAGE_ACCOUNTS_FILE", raising=False)
    monkeypatch.setattr(mod, "default_accounts_file", lambda: roster)
    monkeypatch.setenv("CLAUDE_OAUTH_TOKEN_DEFAULT_ROSTER_PROBE", "tok-probe")

    gate = mod.build_pool(env_file=str(empty_env_file))

    assert [a.name for a in gate._accounts] == ["only-in-the-default-roster"], (
        "build_pool must resolve the roster through default_accounts_file(), "
        "the only source left once accounts_file and USAGE_ACCOUNTS_FILE are "
        "both unset"
    )


def test_build_pool_honours_the_USAGE_ACCOUNTS_FILE_override(
    roster_file, monkeypatch, empty_env_file,
):
    """The fleet convention, shared with fused_memory/config/schema.py and
    scripts/run_vllm_eval.py."""
    _set_pool_tokens(monkeypatch, "B", "C", "D")
    monkeypatch.setenv("USAGE_ACCOUNTS_FILE", str(roster_file))

    gate = mod.build_pool(env_file=str(empty_env_file))

    assert [a.name for a in gate._accounts] == ["max-b", "max-c", "max-d"]


def test_build_pool_hands_the_validator_an_ABSOLUTE_path(
    roster_file, monkeypatch, empty_env_file,
):
    """A relative accounts_file is `.resolve()`d against the CWD by
    UsageCapConfig's validator, and a path that misses degrades to an empty
    pool with only a warning. The trickle's CWD is the systemd unit's, not
    the repo's, so a relative path would resolve somewhere arbitrary."""
    _set_pool_tokens(monkeypatch, "B", "C", "D")

    assert mod.default_accounts_file().is_absolute()

    gate = mod.build_pool(accounts_file=str(roster_file), env_file=str(empty_env_file))
    assert gate.account_count == 3


def test_build_pool_loads_dotenv_before_building_the_gate(tmp_path, monkeypatch):
    """ORDER IS THE WHOLE POINT: _init_accounts reads os.environ EAGERLY at
    construction, so a .env loaded afterwards resolves nothing and the pool
    silently falls back to ~/.claude — today's broken behaviour."""
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CLAUDE_OAUTH_TOKEN_B=tok-from-dotenv-b\n"
        "CLAUDE_OAUTH_TOKEN_C=tok-from-dotenv-c\n"
        "CLAUDE_OAUTH_TOKEN_D=tok-from-dotenv-d\n"
    )
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)

    gate = mod.build_pool(accounts_file=str(roster), env_file=str(env_file))

    assert gate.account_count == 3
    assert gate._accounts[0].token == "tok-from-dotenv-b", (
        "the token must have come from the .env — if the gate were built "
        "first, every account would have resolved token-less"
    )


def test_build_pool_does_not_let_the_dotenv_undo_the_units_api_key_strip(
    tmp_path, monkeypatch,
):
    """The .env that carries the POOL also carries ANTHROPIC_API_KEY.

    `load_dotenv` sets any variable not already present, and "not present"
    is exactly the state `UnsetEnvironment=ANTHROPIC_API_KEY` leaves the
    trickle in — so the load above would hand the key straight back and
    silently defeat the unit's own directive. The strip belongs HERE, at the
    one point that re-introduces it; `coder.child_env`'s removal cannot
    cover this, because it only shapes children handed an explicit env.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=sk-ant-must-not-come-back\n"
        "CLAUDE_OAUTH_TOKEN_B=tok-from-dotenv-b\n"
        "CLAUDE_OAUTH_TOKEN_C=tok-from-dotenv-c\n"
        "CLAUDE_OAUTH_TOKEN_D=tok-from-dotenv-d\n"
    )
    # delenv rather than a bare assertion, and it is what makes the test safe
    # to run anywhere: monkeypatch restores whatever the ambient value was,
    # including after build_pool pops it for real.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)

    gate = mod.build_pool(accounts_file=str(roster), env_file=str(env_file))

    assert "ANTHROPIC_API_KEY" not in os.environ, (
        "the .env load put the API key back into this process — every child "
        "that INHERITS this environment (the census launcher's env=None "
        "path) would authenticate as the key's identity while the pool's "
        "failover still looked like it worked"
    )
    assert gate.account_count == 3, (
        "the strip must remove ONE variable, not disable the dotenv load the "
        "whole pool depends on"
    )


def test_a_capped_pool_never_hands_the_census_an_api_key(tmp_path, monkeypatch):
    """The path the gap actually ran down, composed as a night runs it.

    The REAL `build_pool` runs first, exactly as `run_nightly` calls it — it
    is the .env load inside it that puts the key back — and by census time
    the pool is capped, which is the state that makes `subprocess_env`
    return None. nightly's `_default_census_launcher` then spawns census.py
    with `env=None`, i.e. this process's own environment. So the question is
    not what the returned env holds, it is what the census PROCESS will see;
    that is what this asserts. Getting it wrong is invisible rather than
    loud: `preflight_headroom` would SUCCEED on the key's identity instead
    of fail-safe deferring, so nothing in the journal would say the account
    choice had been bypassed.

    The capped pool is the FakeGate for the same reason the rest of this
    file uses it — "nothing leasable" is a gate state, not a roster one: a
    roster whose tokens are all absent does not produce it, because
    `_init_accounts` degrades to the `~/.claude` 'default' account, which
    still leases.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ANTHROPIC_API_KEY=sk-ant-must-not-reach-the-census\n"
        "CLAUDE_OAUTH_TOKEN_B=tok-from-dotenv-b\n"
        "CLAUDE_OAUTH_TOKEN_C=tok-from-dotenv-c\n"
        "CLAUDE_OAUTH_TOKEN_D=tok-from-dotenv-d\n"
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)

    mod.build_pool(accounts_file=str(roster), env_file=str(env_file))
    census_env = mod.subprocess_env(_pool(("max-b", True), ("max-h", True)))

    assert census_env is None, "a capped pool leases nothing — the premise"
    effective = os.environ if census_env is None else census_env
    assert "ANTHROPIC_API_KEY" not in effective, (
        "an empty pool must leave the census with NO way to authenticate — "
        "that is what makes census.preflight_headroom's fail-safe defer fire "
        "instead of billing an identity the pool never chose"
    )


def test_build_pool_logs_the_resolved_roster_but_never_a_token(
    roster_file, monkeypatch, caplog, empty_env_file,
):
    _set_pool_tokens(monkeypatch, "B", "C", "D")

    with caplog.at_level("INFO", logger="legibility.account_pool"):
        mod.build_pool(accounts_file=str(roster_file), env_file=str(empty_env_file))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "max-b" in logged and "max-d" in logged, logged
    assert "3" in logged, logged
    assert "tok-b" not in logged and "tok-d" not in logged, (
        f"a token must NEVER reach the journal; got {logged!r}"
    )


def _module_warnings(caplog) -> list[str]:
    """WARNING records THIS module actually emitted.

    ``caplog.at_level(logger=...)`` only raises that logger's level; per the
    pytest docs it does NOT scope capture to that logger. ``UsageGate.
    _init_accounts`` logs its own ``Account 'max-b': env var
    CLAUDE_OAUTH_TOKEN_B not set — skipping`` WARNING for every token-less
    account, which alone would satisfy an unfiltered ``caplog.records``
    assertion regardless of whether ``build_pool``'s own warning fired.
    Filtering by ``r.name`` is what proves the warning came from
    ``account_pool.build_pool`` and not merely from the gate underneath it.
    """
    return [
        r.getMessage() for r in caplog.records
        if r.levelname == "WARNING" and r.name == "legibility.account_pool"
    ]


def test_build_pool_warns_LOUDLY_when_it_resolves_no_accounts(
    tmp_path, monkeypatch, caplog, empty_env_file,
):
    """The degradation that must never be silent: no token env var resolves
    AND no ``~/.claude/.credentials.json`` fallback exists either, so the
    gate ends up with literally zero accounts. It has to be VISIBLE, or this
    task's fix silently un-does itself the day a token env var is dropped
    from the unit.

    ``CREDENTIALS_PATH`` is pinned to a path that does not exist so this
    arm's premise (zero accounts, not the OTHER zero-usable-account arm
    below) does not depend on whether the host running the suite happens to
    hold real Claude credentials.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)
    monkeypatch.setattr(
        usage_gate_mod, "CREDENTIALS_PATH", tmp_path / "no-such-credentials.json",
    )

    with caplog.at_level("WARNING", logger="legibility.account_pool"):
        gate = mod.build_pool(accounts_file=str(roster), env_file=str(empty_env_file))

    assert gate.account_count == 0, (
        f"this arm's premise: no token env vars AND no default credential "
        f"on disk; got {gate.account_count} accounts"
    )
    warnings = _module_warnings(caplog)
    assert warnings, (
        "a pool that resolved no real accounts must SAY so — silently "
        "returning the ~/.claude fallback is the defect this task removes"
    )
    assert any("resolved NO usable accounts" in w for w in warnings), (
        f"the module's own distinctive phrase must be present -- the gate "
        f"logs its own per-account skip warning too, which this assertion "
        f"must not be satisfied by; got {warnings}"
    )
    assert any("max-b" in w for w in warnings), (
        f"name the accounts it could not resolve, so an operator knows which "
        f"env var is missing; got {warnings}"
    )


def test_build_pool_warns_LOUDLY_when_it_falls_back_to_the_default_credential(
    tmp_path, monkeypatch, caplog, empty_env_file,
):
    """The OTHER zero-usable-account arm, and the exact pre-5488 defect:
    every ``CLAUDE_OAUTH_TOKEN_*`` is missing but
    ``~/.claude/.credentials.json`` exists, so ``_init_accounts`` resolves
    ONE account named 'default' -- ``build_pool``'s ``names == ["default"]``
    branch. Before this test existed, that branch was exercised only on a
    host that happened to carry real Claude credentials, so it could pass
    CI green while carrying a defect no run of the suite would ever see.
    """
    roster = tmp_path / "roster.yaml"
    roster.write_text(_ROSTER_YAML)
    for letter in ("B", "C", "D"):
        monkeypatch.delenv(f"CLAUDE_OAUTH_TOKEN_{letter}", raising=False)
    cred_file = tmp_path / "credentials.json"
    cred_file.write_text(json.dumps({"accessToken": "tok-default"}))
    monkeypatch.setattr(usage_gate_mod, "CREDENTIALS_PATH", cred_file)

    with caplog.at_level("WARNING", logger="legibility.account_pool"):
        gate = mod.build_pool(accounts_file=str(roster), env_file=str(empty_env_file))

    assert [a.name for a in gate._accounts] == ["default"], (
        f"this arm's premise: the ~/.claude fallback resolved exactly one "
        f"account named 'default'; got "
        f"{[a.name for a in gate._accounts]}"
    )
    warnings = _module_warnings(caplog)
    assert warnings, (
        "the 'default' fallback IS today's broken behaviour and must warn "
        "just as loudly as the zero-account arm"
    )
    assert any("resolved NO usable accounts" in w for w in warnings), warnings
    assert any("max-b" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# task 5488 / step-19: subprocess_env — the census grandchild's account
#
# nightly's census launcher spawns scripts/legibility/census.py with no ``env``
# of its own, so today that grandchild is authenticated only because the
# 2026-09-14 stopgap drop-in exported ONE account's token into the systemd
# unit. This helper is what replaces that pin, and it is strictly better than
# what it replaces: a LIVE pool-chosen account at launch time rather than a
# hardcoded max-h.
#
# It must degrade to None rather than raise or block. ``census.preflight_
# headroom`` fails SAFE, so a census that cannot authenticate silently defers
# the whole run instead of erroring -- "no account to give it" therefore has to
# mean "inherit exactly as before", never "fail the census".
#
# Per-invocation rotation INSIDE census.py (its own file lock, a mining loop
# over many batches) is a separate job and is filed as a follow-up.
# ---------------------------------------------------------------------------

def test_subprocess_env_carries_a_pool_token_and_strips_the_api_key(monkeypatch):
    """The two halves of the env contract, and why the strip is half of it:
    the CLI prefers ANTHROPIC_API_KEY over the OAuth token, so leaving it set
    silently defeats the account choice -- the identical reasoning that put
    the strip in ``_invoke_cli``."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-survive")
    monkeypatch.setenv("LEGIBILITY_UNRELATED_VAR", "kept")
    gate = _pool(("max-b", False), ("max-h", False))

    env = mod.subprocess_env(gate)

    assert env is not None, "two live accounts must yield an env"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-max-h", (
        "the census draws from the END of the roster like the rest of the "
        "trickle, so it does not contend with the orchestrator's b->h order"
    )
    assert "ANTHROPIC_API_KEY" not in env
    assert env["LEGIBILITY_UNRELATED_VAR"] == "kept", (
        "the rest of the parent env rides along -- census.py needs PATH, HOME "
        "and the unit's own vars, so this is an OVERLAY, not a replacement"
    )
    assert gate.lease_calls == [
        # No exclusion: the census takes ONE account and never rotates, so it
        # has no loop of its own to bound.
        {"scope": None, "reverse": True, "exclude": None},
    ]


def test_subprocess_env_skips_a_capped_account():
    gate = _pool(("max-b", False), ("max-g", True), ("max-h", True))

    env = mod.subprocess_env(gate)

    assert env is not None
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "tok-max-b"


def test_subprocess_env_never_keeps_the_probe_claim():
    """The lease is read for its token and handed straight back.

    The census runs in ANOTHER process, so the gate can observe nothing it
    does and nothing will ever settle the slot. Holding a PROBE_IN_FLIGHT
    claim for the census's whole runtime (its stages carry 120/900/1800s
    timeouts) would make that account inadmissible to the trickle's own
    digests for the rest of the night -- the pool would silently shrink by one
    every time a census fired.
    """
    gate = _pool(("max-b", False), ("max-h", False))

    env = mod.subprocess_env(gate)

    assert env is not None
    assert gate.released == [env["CLAUDE_CODE_OAUTH_TOKEN"]]


def test_subprocess_env_returns_none_when_nothing_can_be_leased():
    """An exhausted pool means "inherit as before", not "fail the census".

    Returning None is what keeps the census's best-effort contract intact: a
    census launch must never crash or fail the nightly run, and must never be
    BLOCKED by a pool problem either.
    """
    gate = _pool(("max-b", True), ("max-h", True))

    assert mod.subprocess_env(gate) is None


# ---------------------------------------------------------------------------
# task 5488 / step-26: THE LIVE EXERCISE, as an automatable pin
#
# The ruling asks for one real run with a pool token deliberately absent,
# completing on another account. Marked `integration`, which the root
# pyproject.toml deselects by default (`-m 'not ... integration ...'`), so it
# never runs in the normal suite and never spends tokens there. Run it on
# purpose:
#
#     uv run --frozen --project shared python -m pytest \
#         scripts/tests/test_legibility_account_pool.py -m integration
#
# Everything else in this file drives a fake gate and a stub invoker; this is
# the one case that proves the whole chain is real -- a roster on disk, tokens
# out of the environment, a genuine `claude -p --model haiku`, and a failover
# that costs the night nothing.
# ---------------------------------------------------------------------------

def _roster_token_envs():
    """The `oauth_token_env` names the REAL roster declares, in file order."""
    import yaml

    data = yaml.safe_load(mod.default_accounts_file().read_text()) or {}
    return [entry["oauth_token_env"] for entry in data.get("accounts", [])]


@pytest.mark.integration
def test_live_one_shot_completes_when_a_pool_token_is_missing(monkeypatch, tmp_path):
    """A token absent from the environment must cost the night NOTHING.

    The account removed is the one the trickle would otherwise take FIRST --
    the LAST in the roster, since the trickle drains h->b -- so a pass here
    means failover actually happened rather than the run getting lucky on an
    account that was never at risk.
    """
    token_envs = _roster_token_envs()
    available = [name for name in token_envs if os.environ.get(name)]
    if len(available) < 2:
        pytest.skip(
            f"needs at least 2 pool tokens in the environment, have "
            f"{len(available)} of {len(token_envs)} — source the project .env"
        )

    # Removed from the ENV, and build_pool pointed at an empty env file, so the
    # deletion cannot be undone by the .env load inside build_pool.
    first_choice = available[-1]
    removed_token = os.environ[first_choice]
    monkeypatch.delenv(first_choice)
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("")

    gate = mod.build_pool(env_file=str(empty_env))
    assert gate.account_count == len(available) - 1, (
        f"expected the pool to resolve every token but {first_choice}; "
        f"got {gate.account_count} accounts"
    )

    used_tokens = []

    def _recording_invoke(prompt, model, **kwargs):
        used_tokens.append(kwargs.get("oauth_token"))
        return coder_mod._invoke_cli(prompt, model, **kwargs)

    invoke = mod.pool_invoke(gate, invoke=_recording_invoke)
    try:
        reply = invoke("Reply with exactly the two characters: OK", "haiku")
    except coder_mod.CoderCapExhausted as exc:
        # The same policy shared/tests/_capacity_skip.py applies to every
        # real-CLI test: a genuinely exhausted fleet is not a failure of this
        # code, and cannot be told apart from one by running it.
        pytest.skip(f"the whole pool is capped, nothing to exercise here: {exc}")

    assert reply.strip(), "a live one-shot must come back with a model turn"
    assert used_tokens, "the CLI seam was never reached"
    assert removed_token not in used_tokens, (
        "the run used the account whose token was removed from the "
        "environment — the pool is not reading the env it claims to"
    )
