"""A SUCCESSFUL reply is a result, not a cap — for a caller that opts in (task 6042).

``invoke_with_cap_retry`` offers every result it does not early-exit on to
``slot.detect_cap_hit``, successes included, and ``UsageGate.detect_cap_hit``
classifies a synthetic ``success=False`` result. So a successful reply that
merely QUOTES a cap banner reads as a cap. Measured 2026-10-02: a coder verdict
whose ``evidence_quote`` is ``REAL_CLI_CAP_HIT_MESSAGES[0]`` classifies ``OK()``
as the success it is, and ``CapHit`` once ``success`` is forced False.

The legibility codebook is dominated by usage-limit clusters, so for that
caller one cap-themed verdict would cap a healthy account, and every retry
would cap the next. ``detect_caps_in_successful_output=False`` confines cap
detection to failed results. The default is unchanged for the fleet, and (c)
deliberately does NOT pin the default's false positive — only that it still
returns.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from _usage_gate_test_helpers import make_gate

from shared.cap_markers import REAL_CLI_CAP_HIT_MESSAGES
from shared.cli_invoke import AgentResult, invoke_with_cap_retry

_BOUND_SECS = 2.0
_BANNER = REAL_CLI_CAP_HIT_MESSAGES[0]
_CAP_QUOTING_VERDICT = json.dumps({
    'matches': [{'cluster_id': 'usage-limit-stall', 'evidence_quote': _BANNER}],
    'candidates': [],
})
_PLAIN_VERDICT = json.dumps({'matches': [], 'candidates': []})


def _scripted(by_token):
    calls = []

    async def _invoke(**kwargs):
        calls.append(kwargs['oauth_token'])
        return by_token[kwargs['oauth_token']]

    return _invoke, calls


async def _run(gate, invoke_fn, **knobs):
    with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
        try:
            return await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'lbl',
                    invoke_fn=invoke_fn,
                    prompt='p', system_prompt='s', cwd=Path('/tmp'),
                    **knobs,
                ),
                _BOUND_SECS,
            )
        finally:
            await gate.shutdown()


async def test_a_successful_reply_quoting_a_banner_is_returned_and_caps_nothing():
    gate = make_gate(['a', 'b'])
    invoke_fn, calls = _scripted({
        'fake-token-a': AgentResult(success=True, output=_CAP_QUOTING_VERDICT),
        'fake-token-b': AgentResult(success=True, output=_PLAIN_VERDICT),
    })

    result = await _run(gate, invoke_fn, detect_caps_in_successful_output=False)

    assert result.output == _CAP_QUOTING_VERDICT
    assert calls == ['fake-token-a']
    assert gate.active_account_name == 'a'
    assert gate.soonest_resets_at is None


async def test_a_failed_reply_carrying_a_banner_still_caps_and_fails_over():
    """The knob narrows only the SUCCESS path: a real cap still rotates."""
    gate = make_gate(['a', 'b'])
    invoke_fn, calls = _scripted({
        'fake-token-a': AgentResult(success=False, output=_BANNER),
        'fake-token-b': AgentResult(success=True, output=_PLAIN_VERDICT),
    })

    result = await _run(gate, invoke_fn, detect_caps_in_successful_output=False)

    assert result.output == _PLAIN_VERDICT
    assert calls == ['fake-token-a', 'fake-token-b']
    assert gate.active_account_name == 'b'


async def test_the_default_still_terminates_and_returns():
    gate = make_gate(['a', 'b'])
    invoke_fn, _calls = _scripted({
        'fake-token-a': AgentResult(success=True, output=_CAP_QUOTING_VERDICT),
        'fake-token-b': AgentResult(success=True, output=_PLAIN_VERDICT),
    })

    result = await _run(gate, invoke_fn)

    assert result.success
