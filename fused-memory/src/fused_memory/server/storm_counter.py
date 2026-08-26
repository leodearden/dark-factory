"""Re-export shim — :class:`StormCounter` now lives in :mod:`shared.storm_counter`.

Task 3088 extracted the rolling-window burst-detector body here so a fourth
consumer would reuse rather than re-copy it (INV-5). Task 3689 is that fourth
consumer — ``shared.mcp_markup_middleware``, which keys a burst by
``(project, policy_outcome)`` — and it sits in the base layer, which may not
import ``fused_memory``. So the class moved DOWN to :mod:`shared.storm_counter`
rather than being copied a fourth time, and this module became a shim: same
public name, same contract, one home.

Kept rather than deleted so the existing importers
(:mod:`fused_memory.server.markup_tripwire`,
:mod:`fused_memory.services.memory_service`) and the prose references in
``config/reload.py`` and ``config/schema.py`` need no edit, and so
``tests/server/test_storm_counter.py`` keeps exercising the contract through
this path — which is what pins the shim honest.

Mirrors exactly what task 3688 did to this package's ``MCP_MARKUP_PATTERNS``:
promote to ``shared``, re-export from the old home, public names unchanged.
"""

from __future__ import annotations

from shared.storm_counter import StormCounter

__all__ = ['StormCounter']
