"""An AgentResult that reports every token count the ``invocations`` ledger records.

Built for the tests of the calls that write that ledger: the recon stages, the
judge, the verify loop and the link adjudicator (task 10414).
``_invocation_ledger.py`` reads the ledger back. A plain module rather than
conftest.py, for the reason ``_fm_helpers.py`` gives.
"""

from __future__ import annotations

from typing import Any

from shared.cli_invoke import AgentResult

LEDGER_TOKENS: dict[str, int] = {
    'input_tokens': 1200,
    'output_tokens': 300,
    'cache_read_tokens': 50000,
    'cache_create_tokens': 4000,
}
"""Each token count, keyed by its AgentResult field, which is also its ledger column."""


def agent_result_with_tokens(*, success: bool, **fields: Any) -> AgentResult:
    """An AgentResult reporting :data:`LEDGER_TOKENS`, plus *fields*."""
    return AgentResult(success=success, output='', **LEDGER_TOKENS, **fields)
