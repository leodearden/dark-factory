"""The service factory every operator entry point opens its store through.

:func:`maintenance_service`, opened as a :class:`ServiceOpen` mode, constructs,
opens and closes the one MemoryService an operator run uses. Its callers are
the scripts under ``fused-memory/scripts/`` and the repo-root ``scripts/``,
which ``tests/test_operator_service_construction_guard.py`` holds to it, and
the maintenance CLIs beside this module. :func:`override_config_path` scopes the
CONFIG_PATH a config is built under.

The module name's leading underscore does not make this package-private:
callers outside the package import from here, so changing a signature here
changes every operator script.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator, Generator
from enum import StrEnum
from typing import assert_never

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.services import memory_service

logger = logging.getLogger(__name__)


class ServiceOpen(StrEnum):
    """How :func:`maintenance_service` opens the service it yields.

    * ``FULL`` awaits ``initialize()``: every backend, startup maintenance and
      the durable queue.
    * ``READS`` awaits ``initialize_for_reads()``: the stores a read-only run
      needs, and nothing that writes.
    * ``NONE`` awaits nothing; only the backends that connect lazily on first
      use are ever reached.
    """

    FULL = 'full'
    READS = 'reads'
    NONE = 'none'


@contextlib.contextmanager
def override_config_path(config_path: str | None) -> Generator[None, None, None]:
    """Temporarily set CONFIG_PATH in the environment, then restore it.

    When *config_path* is ``None`` the environment is left completely
    untouched — the context manager is a no-op.

    When *config_path* is provided:
    - ``os.environ['CONFIG_PATH']`` is set to *config_path* on entry.
    - On exit (normal **or** exceptional) the original state is restored:
      if CONFIG_PATH was absent before entry it is popped; if it had a
      value that value is reinstated.

    .. warning:: **Not thread-safe.**  This context manager mutates
       ``os.environ`` without any locking.  Concurrent calls within the
       same process will race on the ``CONFIG_PATH`` key.  Use only from
       a single thread (or a single ``asyncio`` event loop with no
       concurrent tasks that also call this function).
    """
    if config_path is None:
        yield
        return

    old_value = os.environ.get('CONFIG_PATH')
    os.environ['CONFIG_PATH'] = config_path
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop('CONFIG_PATH', None)
        else:
            os.environ['CONFIG_PATH'] = old_value


async def _open(service: memory_service.MemoryService, open_as: ServiceOpen) -> None:
    match open_as:
        case ServiceOpen.FULL:
            await service.initialize()
        case ServiceOpen.READS:
            await service.initialize_for_reads()
        case ServiceOpen.NONE:
            pass
        case _:
            assert_never(open_as)


@contextlib.asynccontextmanager
async def maintenance_service(
    config_path: str | None = None,
    *,
    config: FusedMemoryConfig | None = None,
    open_as: ServiceOpen = ServiceOpen.FULL,
) -> AsyncIterator[tuple[FusedMemoryConfig, memory_service.MemoryService]]:
    """Construct, open and always close the one MemoryService an operator run uses.

    Every operator entry point builds its service here, so construction, opening
    and close are one policy rather than one per script.

    The config comes from exactly one source.  With *config_path* (or neither
    argument) a FusedMemoryConfig is built with CONFIG_PATH set to it; when
    *config_path* is ``None`` the environment is left untouched.  With
    *config* the given object is used as it is, for callers that read or
    rewrite their config before the service exists.  Passing both is a
    ValueError, raised before anything is constructed, as is an *open_as* that
    is not a :class:`ServiceOpen`.

    *open_as* picks the opening (see :class:`ServiceOpen`).  The service is
    closed on every exit once it exists, including after a failed opening, so
    a caller never holds a half-open service.  A close error is logged, never
    raised, so it cannot mask the caller's own exception or the opening's.

    The class is looked up on ``fused_memory.services.memory_service`` at call
    time, which is the seam tests replace it on.

    A caller that turns a failed opening into its own error maps the entry
    alone, never the body. Enter the factory on a ``contextlib.AsyncExitStack``
    and wrap only ``stack.enter_async_context(...)`` in the ``try``, as the
    second example shows. A ``try`` around the whole ``async with`` would also
    map every error the run raises, attributing a body failure to the store
    (``shared/src/shared/cli_boundary.py::run_cli`` states the same rule at the
    process boundary).

    Yields:
        ``(config, service)``, with the service already opened.

    Examples::

        async with maintenance_service(config=config, open_as=ServiceOpen.READS) as (_, memory):
            report = await run(memory)

        async with contextlib.AsyncExitStack() as stack:
            try:
                _, memory = await stack.enter_async_context(maintenance_service(config=config))
            except OSError as exc:
                raise StoreUnavailable(f'the memory store could not be opened: {exc}') from exc
            report = await run(memory)
    """
    open_as = ServiceOpen(open_as)
    if config_path is not None and config is not None:
        raise ValueError(
            f'maintenance_service takes one config source: got config_path={config_path!r} '
            f'and a {type(config).__name__} config='
        )
    service: memory_service.MemoryService | None = None
    with override_config_path(config_path):
        try:
            if config is None:
                config = FusedMemoryConfig()
            service = memory_service.MemoryService(config)
            await _open(service, open_as)
            yield config, service
        finally:
            if service is not None:
                try:
                    await service.close()
                except Exception:
                    logger.warning('Error closing service in maintenance_service', exc_info=True)
