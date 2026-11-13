"""This test process's ProcessIdentity, and variants of it that only the identity tells apart from it.

Every variant names ``os.getpid()``, so a pid-only liveness check reads each
one as alive; ``IdentityVariant.judged_dead`` says which ones
``shared/src/shared/session_liveness.py::judge`` reads as DEAD regardless.
"""

from __future__ import annotations

import dataclasses
import os
from enum import StrEnum

from shared.process_identity import ProcessIdentity, capture

FORGED_BOOT_ID = '00000000-0000-0000-0000-000000000000'
OTHER_HOST = 'other-host.invalid'


class IdentityVariant(StrEnum):
    LIVE = 'live'
    REUSED_PID = 'reused-pid'
    REBOOTED = 'rebooted'
    OTHER_HOST = 'other-host'

    @property
    def judged_dead(self) -> bool:
        return self in (IdentityVariant.REUSED_PID, IdentityVariant.REBOOTED)


def own_identity(variant: IdentityVariant = IdentityVariant.LIVE) -> ProcessIdentity:
    """This test process's identity as ``capture`` reads it now, altered as *variant* names."""
    live = capture(os.getpid())
    assert live is not None, 'the running test process should be capturable'
    if variant is IdentityVariant.REUSED_PID:
        return dataclasses.replace(live, start_ticks=live.start_ticks + 1)
    if variant is IdentityVariant.REBOOTED:
        return dataclasses.replace(live, boot_id=FORGED_BOOT_ID)
    if variant is IdentityVariant.OTHER_HOST:
        return dataclasses.replace(live, host=OTHER_HOST)
    return live
