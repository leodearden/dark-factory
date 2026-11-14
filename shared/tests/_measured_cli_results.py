"""``claude --print --output-format json`` results measured live.

One copy of each measurement for every suite that replays it, so a
re-measurement updates them all at once. Consumers import by bare module name
-- ``conftest.py`` prepends ``shared/tests`` to ``sys.path`` (same convention
as ``_usage_gate_test_helpers.py``). The leading underscore keeps pytest from
collecting this module as a test file.
"""

from __future__ import annotations

import json

ORG_DISABLED_TEXT = (
    'Your organization has disabled Claude subscription access for Claude '
    'Code · Use an Anthropic API key instead, or ask your admin to enable '
    'access'
)

# Measured 2026-09-26 (CLI 2.1.283) for the org-disabled max-b/max-c
# accounts, via the production probe command.
MEASURED_ORG_DISABLED_403 = json.dumps({
    'type': 'result',
    'subtype': 'success',
    'is_error': True,
    'api_error_status': 403,
    'result': ORG_DISABLED_TEXT,
    'num_turns': 1,
    'total_cost_usd': 0,
})
