"""Parse a warm-lane-gc.sh reclaim summary and log the orchestrator arm's per-pass record.

The script's stdout contract (one ``reclaim: name=N …`` line, field set
differing between dark-factory's and reify's copies) is documented in
``orchestrator/scripts/warm-lane/README.md``, Delta 13 item 2 and Delta 14.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SUMMARY_PREFIX = 'reclaim:'
REQUIRED_FIELDS = ('reset', 'removed', 'preserved')

_FIELD_NAME = re.compile(r'[a-z][a-z_]*')
_TOKEN = re.compile(r'([a-z][a-z_]*)=([0-9]+)')
_STDOUT_QUOTE_LIMIT = 200


class ReclaimSummaryError(ValueError):
    """A reclaim summary that cannot be counted."""


@dataclass(frozen=True)
class ReclaimSummary:
    """One reclaim run's counts, keyed by field name, in the order gc.sh emitted them."""

    fields: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _check_fields(self.fields)

    def count(self, name: str) -> int | None:
        return dict(self.fields).get(name)

    @property
    def reset(self) -> int:
        return dict(self.fields)['reset']

    @property
    def removed(self) -> int:
        return dict(self.fields)['removed']

    @property
    def preserved(self) -> int:
        return dict(self.fields)['preserved']

    def render(self) -> str:
        return ' '.join(f'{name}={value}' for name, value in self.fields)


def _check_fields(fields: tuple[tuple[str, int], ...]) -> None:
    seen: set[str] = set()
    for name, value in fields:
        if not _FIELD_NAME.fullmatch(name):
            raise ReclaimSummaryError(
                f'reclaim summary field name {name!r} does not match {_FIELD_NAME.pattern}',
            )
        if name in seen:
            raise ReclaimSummaryError(f'reclaim summary field {name!r} appears more than once')
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ReclaimSummaryError(
                f'reclaim summary field {name!r} has value {value!r}, not a non-negative int',
            )
        seen.add(name)
    missing = [name for name in REQUIRED_FIELDS if name not in seen]
    if missing:
        raise ReclaimSummaryError(f'reclaim summary lacks required field(s) {missing}')


def parse_reclaim_summary(stdout: str) -> ReclaimSummary:
    summaries = [
        line for line in (raw.strip() for raw in stdout.splitlines())
        if line.startswith(SUMMARY_PREFIX)
    ]
    if len(summaries) != 1:
        raise ReclaimSummaryError(
            f'expected exactly one {SUMMARY_PREFIX!r} line on stdout, found '
            f'{len(summaries)} (stdout={stdout[:_STDOUT_QUOTE_LIMIT]!r})',
        )
    body = summaries[0].removeprefix(SUMMARY_PREFIX)
    return ReclaimSummary(fields=tuple(_parse_token(token) for token in body.split()))


def _parse_token(token: str) -> tuple[str, int]:
    match = _TOKEN.fullmatch(token)
    if match is None:
        raise ReclaimSummaryError(f'reclaim summary token {token!r} is not name=count')
    return match.group(1), int(match.group(2))
