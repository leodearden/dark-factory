"""The verify-result cache's vocabulary: the verify roles it acts for, and how far it may act."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum
from typing import Literal, get_args

VerifyRole = Literal['task', 'merge', 'background']
VERIFY_ROLES: tuple[VerifyRole, ...] = get_args(VerifyRole)

VERIFY_CACHE_ENV = 'DF_VERIFY_CACHE'


class CacheMode(StrEnum):
    OFF = 'off'
    LOG = 'log'
    ON = 'on'


def forced_off(environ: Mapping[str, str] = os.environ) -> bool:
    """Whether *environ* forces every role's mode off; no value can force a mode on."""
    return environ.get(VERIFY_CACHE_ENV, '').strip().lower() == CacheMode.OFF.value
