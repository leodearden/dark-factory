"""The verify-result cache's vocabulary: the verify roles it acts for, and how far it may act."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, get_args

VerifyRole = Literal['task', 'merge', 'background']
VERIFY_ROLES: tuple[VerifyRole, ...] = get_args(VerifyRole)


class CacheMode(StrEnum):
    OFF = 'off'
    LOG = 'log'
    ON = 'on'
