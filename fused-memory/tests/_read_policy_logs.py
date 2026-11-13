"""Which logger the checked node read's incompleteness policy warned under.

fused-memory/src/fused_memory/backends/graphiti_client.py::read_entity_nodes_checked
hands its caller's ``log`` to ``apply_incompleteness_policy``, whose EMPIRICAL
warning names the read method. A test that pins where that warning lands
filters ``caplog`` here, so a change to the level or the method token touches
one place.
"""
from __future__ import annotations

import logging

import pytest


def node_read_warnings(
    caplog: pytest.LogCaptureFixture, logger_name: str
) -> list[logging.LogRecord]:
    """The ``enumerate_entity_nodes`` policy WARNINGs logged under ``logger_name``."""
    return [
        r for r in caplog.records
        if r.name == logger_name and r.levelno == logging.WARNING
        and 'enumerate_entity_nodes' in r.getMessage()
    ]
