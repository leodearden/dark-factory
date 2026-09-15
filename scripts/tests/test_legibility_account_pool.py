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
