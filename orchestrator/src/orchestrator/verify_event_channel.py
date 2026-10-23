"""DF's reader and writer of reify's REIFY_VERIFY_EVENT_LOG verify event channel.

The channel is newline-delimited C2 v1 JSON records. This module writes DF's own
``channel_open`` header and applies C3 rules 1 (presence) and 3 (the failed
run_all members). The contract's home is reify
``docs/prds/v0_6/run-all-runner-infra-harness.md`` §4 C2/C3, as amended by reify
retro-critic ruling R1; reify's reference reader is
``tests/infra/verify_events.py``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

VERIFY_EVENT_LOG_ENV = 'REIFY_VERIFY_EVENT_LOG'
SCHEMA_VERSION = 1
CHANNEL_OPEN_EVENT = 'channel_open'


def encode_channel_open(*, role: str, verify_id: str) -> bytes:
    record = {
        'v': SCHEMA_VERSION,
        'ts_ns': time.time_ns(),
        'pid': os.getpid(),
        'event': CHANNEL_OPEN_EVENT,
        'role': role,
        'verify_id': verify_id,
    }
    return json.dumps(record, separators=(',', ':')).encode() + b'\n'


@dataclass(frozen=True)
class RunFinish:
    failed: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class VerifyEventChannel:
    """The facts one verify leg's channel carried; the default is an absent channel."""

    present: bool = False
    run_starts: int = 0
    run_finishes: tuple[RunFinish, ...] = ()
    fatal_slot_timeouts: int = 0
    soft_slot_timeouts: int = 0
    skipped_lines: int = 0

    def failed_run_all_members(self) -> list[str] | None:
        """C3 rule 3: None = absent (the text parse governs); [] = full suite."""
        if not self.present:
            return None
        if len(self.run_finishes) != self.run_starts:
            return []
        if not all(finish.complete for finish in self.run_finishes):
            return []
        return list(dict.fromkeys(name for finish in self.run_finishes for name in finish.failed))


def _parse_record(raw: bytes) -> dict[str, object] | None:
    try:
        record = json.loads(raw.decode('utf-8', 'surrogateescape'))
    except ValueError:
        return None
    if not isinstance(record, dict):
        return None
    version = record.get('v')
    if type(version) is not int or version != SCHEMA_VERSION:
        return None
    return record


def _run_finish_from(record: dict[str, object]) -> RunFinish:
    failed = record.get('failed')
    if not isinstance(failed, list) or not all(isinstance(name, str) for name in failed):
        return RunFinish(failed=(), complete=False)
    names_omitted = record.get('names_omitted', 0)
    complete = (
        record.get('interrupted') is False
        and type(names_omitted) is int
        and names_omitted == 0
    )
    return RunFinish(failed=tuple(failed), complete=complete)


def _channel_from(
    records: list[dict[str, object]], *, skipped_lines: int, verify_id: str,
) -> VerifyEventChannel:
    def of_event(event: str) -> list[dict[str, object]]:
        return [record for record in records if record.get('event') == event]

    dispositions = [record.get('disposition') for record in of_event('slot_timeout')]
    return VerifyEventChannel(
        present=any(r.get('verify_id') == verify_id for r in of_event(CHANNEL_OPEN_EVENT)),
        run_starts=len(of_event('run_start')),
        run_finishes=tuple(_run_finish_from(r) for r in of_event('run_finish')),
        fatal_slot_timeouts=dispositions.count('fatal'),
        soft_slot_timeouts=dispositions.count('soft'),
        skipped_lines=skipped_lines,
    )


def parse_verify_event_channel(data: bytes, *, verify_id: str) -> VerifyEventChannel:
    """Parse channel bytes; present only with a ``channel_open`` header for ``verify_id``."""
    records: list[dict[str, object]] = []
    skipped_lines = 0
    for raw in data.split(b'\n'):
        if not raw:
            continue
        record = _parse_record(raw)
        if record is None:
            skipped_lines += 1
        else:
            records.append(record)
    return _channel_from(records, skipped_lines=skipped_lines, verify_id=verify_id)


def read_verify_event_channel(path: Path, *, verify_id: str) -> VerifyEventChannel:
    """Read and parse the channel at ``path``; an unreadable file is absent, never raised."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return VerifyEventChannel()
    except OSError as exc:
        logger.warning('verify event channel %s unreadable; treating it as absent: %s', path, exc)
        return VerifyEventChannel()
    return parse_verify_event_channel(data, verify_id=verify_id)
