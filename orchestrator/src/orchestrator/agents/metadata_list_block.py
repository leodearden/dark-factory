"""A briefing block listing one list-valued task-metadata key, failing open visibly.

Nothing renders when the key is absent or empty. A value that is not a list is
reported, never iterated. Entries the parser refuses are counted, never rendered,
so a malformed record shrinks the list without hiding that it did.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

_Entry = TypeVar('_Entry')


@dataclass(frozen=True)
class MetadataListBlock:
    """The fixed wording of one key's block; ``contents`` names its entries in the plural."""

    title: str
    key: str
    contents: str
    intro: str

    @property
    def heading(self) -> str:
        return f'## {self.title} (metadata.{self.key})'

    def render(
        self,
        raw: object,
        parse: Callable[[object], _Entry | None],
        render_entries: Callable[[Sequence[_Entry]], Iterable[str]],
    ) -> str:
        """The block for the stored value *raw*; *parse* returns None for an entry it cannot read."""
        if not raw:
            return ''
        if not isinstance(raw, list):
            return (
                f'{self.heading}\n\n'
                f'⚠ `metadata.{self.key}` is malformed (a {type(raw).__name__}, not a '
                f'list), so its {self.contents} cannot be shown here; read them from the task record.'
            )
        entries = [entry for entry in map(parse, raw) if entry is not None]
        lines = list(render_entries(entries))
        malformed = len(raw) - len(entries)
        if malformed:
            noun = 'entry' if malformed == 1 else 'entries'
            lines.append(
                f'⚠ {malformed} malformed {noun} on `metadata.{self.key}` could not be '
                'rendered here; read the full list from the task record.'
            )
        listed = '\n'.join(lines)
        return f'{self.heading}\n\n{self.intro}\n\n{listed}'
