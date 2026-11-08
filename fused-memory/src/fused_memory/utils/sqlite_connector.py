"""How a fused-memory SQLite store opens its connection.

A store that takes a ``connect`` parameter types it as :data:`Connector`: given
the database path, it returns the store's open connection.
``shared.async_sqlite_base.connect_daemon`` is the production connector.
"""

from collections.abc import Awaitable, Callable
from typing import TypeAlias

import aiosqlite

Connector: TypeAlias = Callable[[str], Awaitable[aiosqlite.Connection]]
