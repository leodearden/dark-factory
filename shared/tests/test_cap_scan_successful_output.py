"""Only a FAILED result is offered to the cap detector (task 6213).

``UsageGate.detect_cap_hit`` classifies a synthetic ``success=False`` result, so
a successful reply that merely QUOTES a cap banner would read as a cap, bench a
healthy account and discard the good reply. A genuine cap arrives as a failure
(``shared/src/shared/cli_invoke.py::invoke_with_cap_retry``), so a successful
result is returned unchanged and the detector is not asked, while a failed one
carrying a banner still caps and fails over.
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


async def _run(gate, invoke_fn):
    with patch('shared.cli_invoke.asyncio.sleep', new_callable=AsyncMock):
        try:
            return await asyncio.wait_for(
                invoke_with_cap_retry(
                    gate, 'lbl',
                    invoke_fn=invoke_fn,
                    prompt='p', system_prompt='s', cwd=Path('/tmp'),
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

    with patch.object(gate, 'detect_cap_hit') as detect_cap_hit:
        result = await _run(gate, invoke_fn)

    assert result.output == _CAP_QUOTING_VERDICT
    assert calls == ['fake-token-a']
    assert gate.active_account_name == 'a'
    assert gate.soonest_resets_at is None
    detect_cap_hit.assert_not_called()


async def test_a_successful_reply_that_is_a_bare_banner_is_returned_unchanged():
    gate = make_gate(['a', 'b'])
    invoke_fn, calls = _scripted({
        'fake-token-a': AgentResult(success=True, output=_BANNER),
        'fake-token-b': AgentResult(success=True, output=_PLAIN_VERDICT),
    })

    with patch.object(gate, 'detect_cap_hit') as detect_cap_hit:
        result = await _run(gate, invoke_fn)

    assert result.output == _BANNER
    assert calls == ['fake-token-a']
    detect_cap_hit.assert_not_called()


async def test_a_failed_reply_carrying_a_banner_still_caps_and_fails_over():
    gate = make_gate(['a', 'b'])
    invoke_fn, calls = _scripted({
        'fake-token-a': AgentResult(success=False, output=_BANNER),
        'fake-token-b': AgentResult(success=True, output=_PLAIN_VERDICT),
    })

    result = await _run(gate, invoke_fn)

    assert result.output == _PLAIN_VERDICT
    assert calls == ['fake-token-a', 'fake-token-b']
    assert gate.active_account_name == 'b'
