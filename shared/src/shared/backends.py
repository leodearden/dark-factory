"""The backend vocabulary and its one decoder.

``Backend`` names every way a role can be invoked, and ``parse_backend`` is the
only place a configured spelling becomes a ``Backend``: it accepts the
canonical values plus one legacy alias and raises ``UnknownBackend`` for
anything else, so a typo fails at load rather than at dispatch.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final

__all__ = [
    'AGENT_CLI_BACKENDS',
    'SDK_BACKENDS',
    'Backend',
    'UnknownBackend',
    'parse_backend',
]


class Backend(StrEnum):
    CLAUDE = 'claude'  # the Claude Code CLI
    CODEX = 'codex'
    GEMINI = 'gemini'
    PI = 'pi'
    ZCODE = 'zcode'
    MIMO = 'mimo'
    ANTHROPIC_API = 'anthropic'  # the in-process Anthropic SDK
    OPENAI_API = 'openai'  # the in-process OpenAI SDK


AGENT_CLI_BACKENDS: Final[frozenset[Backend]] = frozenset(
    {Backend.CLAUDE, Backend.CODEX, Backend.GEMINI, Backend.PI, Backend.ZCODE, Backend.MIMO}
)
SDK_BACKENDS: Final[frozenset[Backend]] = frozenset({Backend.ANTHROPIC_API, Backend.OPENAI_API})

_ALIASES: Final[Mapping[str, Backend]] = {'claude_cli': Backend.CLAUDE}


class UnknownBackend(ValueError):
    """A configured backend spelling names no ``Backend``.

    A ``ValueError`` so that a pydantic validator reports it as a
    ``ValidationError`` carrying this message.

    Attributes:
        value: The offending value, verbatim.
        allowed: The canonical backend values, in declaration order.
    """

    def __init__(self, value: object, *, allowed: tuple[str, ...]) -> None:
        self.value = value
        self.allowed = allowed
        super().__init__(f'unknown backend {value!r}; allowed: {", ".join(allowed)}')


def parse_backend(value: object) -> Backend:
    """Decode a configured backend spelling, exactly and case-sensitively."""
    if isinstance(value, Backend):
        return value
    allowed = tuple(b.value for b in Backend)
    if not isinstance(value, str):
        raise UnknownBackend(value, allowed=allowed)
    if value in _ALIASES:
        return _ALIASES[value]
    try:
        return Backend(value)
    except ValueError:
        raise UnknownBackend(value, allowed=allowed) from None
