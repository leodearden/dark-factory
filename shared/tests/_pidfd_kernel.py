"""Shared test kit: skip where the kernel cannot signal a process group through a pidfd.

The probe and its reason live in :mod:`shared.testing_pidfd`.

Consumers import by bare module name (``from _pidfd_kernel import ...``):
``conftest.py`` prepends ``shared/tests`` to ``sys.path``.
"""
from __future__ import annotations

import pytest

from shared.testing_pidfd import group_pidfd_signals_unsupported_reason

_UNSUPPORTED_REASON = group_pidfd_signals_unsupported_reason()

needs_group_pidfd_signals = pytest.mark.skipif(
    _UNSUPPORTED_REASON is not None, reason=_UNSUPPORTED_REASON or '',
)
