"""How far the verify-result cache may act: not at all, observe only, or replay."""

from __future__ import annotations

from enum import StrEnum


class CacheMode(StrEnum):
    OFF = 'off'
    LOG = 'log'
    ON = 'on'
