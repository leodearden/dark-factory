"""The envelope that places an event in the feed, apart from what the event says (PRD §7.1)."""

from __future__ import annotations

import itertools
from datetime import datetime
from typing import Any, Final, TypedDict

from client_feed.contract import FEED_VERSION, EventHeader, FeedEvent


class Envelope(TypedDict):
    """The envelope fields of one feed line, as keyword arguments to an event model."""

    seq: int
    at: datetime
    project_id: str
    feed_version: str


_HEADER_FIELDS: Final = frozenset(EventHeader.model_fields)
_ENVELOPE_FIELDS: Final = _HEADER_FIELDS - {'type'}
if Envelope.__required_keys__ != _ENVELOPE_FIELDS:
    raise TypeError(f'Envelope keys {sorted(Envelope.__required_keys__)} are not the envelope {sorted(_ENVELOPE_FIELDS)}')


def payload(event: FeedEvent) -> dict[str, Any]:
    """What an event says: its fields other than its type and the envelope that places it in the feed."""
    return {name: getattr(event, name) for name in type(event).model_fields if name not in _HEADER_FIELDS}


class Stamp:
    """The envelopes of one pass: seq allocated in call order from the first free seq."""

    def __init__(self, *, project_id: str, first_seq: int) -> None:
        self._project_id = project_id
        self._seqs = itertools.count(first_seq)

    def __call__(self, at: datetime) -> Envelope:
        return Envelope(
            seq=next(self._seqs), at=at, project_id=self._project_id, feed_version=FEED_VERSION
        )
