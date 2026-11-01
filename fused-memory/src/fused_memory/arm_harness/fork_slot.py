"""Which child process holds a Redis server's single fork slot, read from INFO persistence."""

from collections.abc import Awaitable, Mapping
from enum import StrEnum
from typing import Any, Protocol

PERSISTENCE_SECTION = 'persistence'


class ForkHolder(StrEnum):
    """A child process that can hold the fork; each value is the INFO flag naming it."""

    RDB_SAVE = 'rdb_bgsave_in_progress'
    AOF_REWRITE = 'aof_rewrite_in_progress'
    MODULE_FORK = 'module_fork_in_progress'


class ServerInfo(Protocol):
    """The slice of redis.asyncio.Redis (falkordb.asyncio.FalkorDB.connection) that reads INFO."""

    def info(self, section: str, /) -> Awaitable[Mapping[str, Any]]: ...


class ForkSlotUnreadable(RuntimeError):
    """The INFO reply lacks a flag naming a fork holder, so whether the fork is free is unknown."""

    def __init__(self, missing: tuple[str, ...]) -> None:
        self.missing = missing
        super().__init__(
            f'INFO {PERSISTENCE_SECTION} reported none of {list(missing)}, so which child '
            'holds the fork cannot be read'
        )


async def read_fork_holders(server: ServerInfo) -> frozenset[ForkHolder]:
    reply = await server.info(PERSISTENCE_SECTION)
    missing = tuple(holder.value for holder in ForkHolder if holder.value not in reply)
    if missing:
        raise ForkSlotUnreadable(missing)
    return frozenset(holder for holder in ForkHolder if reply[holder.value])
