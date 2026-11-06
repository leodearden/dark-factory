"""Release the async SDK clients that graphiti_core wrappers mint and never close.

Why an unclosed one matters is documented once, at
fused-memory/tests/_fm_helpers.py::reap_leaked_async_httpx_clients.
"""

import inspect
import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)


async def close_wrapped_sdk_clients(wrappers: Iterable[object | None]) -> None:
    """Close each distinct ``wrapper.client`` whose ``close`` is a coroutine function.

    Wrappers without such a client are skipped. A failing close is logged at
    WARNING and the sweep continues, so this never raises.
    """
    seen: set[int] = set()
    for wrapper in wrappers:
        sdk = getattr(wrapper, 'client', None)
        close = getattr(sdk, 'close', None)
        if not inspect.iscoroutinefunction(close) or id(sdk) in seen:
            continue
        seen.add(id(sdk))
        try:
            await close()
        except Exception:
            logger.warning(
                'could not close %s held by %s',
                type(sdk).__name__, type(wrapper).__name__, exc_info=True,
            )
