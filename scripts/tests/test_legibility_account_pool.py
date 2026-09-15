"""Tests for scripts/legibility/account_pool.py — the shared multi-account
UsageGate, wired to the legibility ``invoke=`` seam (task 5488).

WHAT THIS MODULE IS FOR, and therefore what these tests pin: the trickle
used to ride whatever login ``~/.claude`` happened to hold, so one capped
account deferred an entire night while six other accounts sat idle. The
pool turns the gate's roster into the ``(prompt, model) -> str`` callable
the seam already speaks.

THE FAKE GATE IS THE POINT, not a shortcut. ``account_pool`` depends on
five methods of the real 3004-line ``UsageGate`` — ``try_lease``,
``detect_cap_hit``, ``confirm_account_ok``, ``on_agent_complete`` and
``release_probe_slot`` — and stating exactly those here is how the test
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

import account_pool as mod
import coder as coder_mod
import pytest
from shared.usage_gate import AccountLease

# ---------------------------------------------------------------------------
# The narrow gate interface account_pool consumes, hand-rolled.
# ---------------------------------------------------------------------------


class FakeAccount:
    def __init__(self, name, token, *, capped=False):
        self.name = name
        self.token = token
        self.capped = capped
        self.generation = 0


class FakeGate:
    """Exactly the five methods ``account_pool`` calls, and nothing else.

    ``try_lease`` reproduces the real gate's first-fit walk and its
    ``reverse`` knob; ``detect_cap_hit`` reproduces the STRICT detector's
    contract (marks the account capped and returns True only when it
    verdicts a cap) with a scripted verdict, so a test can drive the
    false-verdict branch that must NOT rotate.
    """

    def __init__(self, accounts, *, cap_verdict=True):
        self.accounts = list(accounts)
        self.cap_verdict = cap_verdict
        self.lease_calls = []
        self.detect_calls = []
        self.confirmed = []
        self.released = []
        self.costs = []

    # -- selection ---------------------------------------------------------
    def try_lease(self, *, scope=None, reverse=False):
        self.lease_calls.append({'scope': scope, 'reverse': reverse})
        roster = reversed(self.accounts) if reverse else self.accounts
        for acct in roster:
            if acct.capped:
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
                acct.capped = True
        return True

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
