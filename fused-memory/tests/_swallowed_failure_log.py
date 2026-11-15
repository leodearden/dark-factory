"""Read back what a log-and-continue path recorded about a failure it swallowed."""

from __future__ import annotations

import logging

import pytest


def swallowed_causes(
    caplog: pytest.LogCaptureFixture, *fragments: str,
) -> list[type[BaseException] | None]:
    """The exception type on each WARNING-or-louder record whose message holds every fragment.

    ``None`` stands for a record that names no cause, so a degradation that
    stayed loud but lost its traceback still shows up in the comparison.
    """
    return [
        record.exc_info[0] if record.exc_info else None
        for record in caplog.records
        if record.levelno >= logging.WARNING
        and all(fragment in record.getMessage() for fragment in fragments)
    ]
