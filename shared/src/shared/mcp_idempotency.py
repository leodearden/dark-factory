"""Client-op-id idempotency helper shared by the orchestrator and dashboard
``McpSession`` twins (task 2712 / task 2766).

Both twins post JSON-RPC ``tools/call`` requests to a fused-memory MCP
Streamable HTTP endpoint and want the same handful of mutating task tools to
carry a client-supplied idempotency key, so a retried write dedupes
server-side instead of double-applying. Task 2712 introduced
:data:`MUTATING_TASK_TOOLS` and :func:`maybe_inject_client_op_id` twice —
verbatim — in ``orchestrator/src/orchestrator/mcp_lifecycle.py`` and
``dashboard/src/dashboard/data/memory.py``. This module hoists the shared
frozenset and injection helper into one place so the two copies cannot drift.

Caveat (unchanged by this hoist): the two call sites differ in what surrounds
this helper. The orchestrator's ``_raw_call`` calls
``maybe_inject_client_op_id`` ONCE before its transport retry loop, so every
retry of a logical call reuses the same key. The dashboard's ``_raw_call`` has
no transport retry loop, so each invocation generates a fresh key when none is
supplied — see the dashboard twin's docstring for the full caveat. Only the
frozenset + injection helper are de-duplicated here; each call site's
surrounding retry behaviour (or lack thereof) is untouched.
"""

from __future__ import annotations

import uuid
from typing import Any

__all__ = ['MUTATING_TASK_TOOLS', 'maybe_inject_client_op_id']

# Mutating task tools that must carry a client-supplied idempotency key so a
# transport-level retry (after an ambiguous timeout/reset) dedupes server-side
# instead of double-applying (task 2712). Reads/initialize are untouched.
MUTATING_TASK_TOOLS = frozenset(
    {
        'update_task',
        'set_task_status',
        'add_dependency',
        'remove_dependency',
    }
)


def maybe_inject_client_op_id(
    method: str,
    params: dict[str, Any] | None,
) -> None:
    """Inject a fresh ``client_op_id`` into a mutating tool call's arguments.

    No-op for non-``tools/call`` methods, read tools, or a caller that already
    supplied a key (forward-compat with higher-level retries that thread a
    stable key). Mutates ``params['arguments']`` in place (callers' ``payload``
    holds ``params`` by reference and is re-serialized each attempt), so the
    key rides along on the posted body.

    Whether repeated calls to this function (e.g. across a transport retry
    loop) observe the *same* key or a fresh one each time depends entirely on
    when and how often the caller invokes it — see each ``McpSession`` twin's
    own docstring for its specific invariant.
    """
    if (
        method == 'tools/call'
        and isinstance(params, dict)
        and params.get('name') in MUTATING_TASK_TOOLS
        and isinstance(params.get('arguments'), dict)
        and 'client_op_id' not in params['arguments']
    ):
        params['arguments']['client_op_id'] = str(uuid.uuid4())
