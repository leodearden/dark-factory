"""Pin for ``fused_memory.server.storm_counter`` as a re-export shim.

Task 3689 promoted ``StormCounter`` to ``shared.storm_counter`` (because
``shared.mcp_markup_middleware`` needed it and ``shared`` may not import
``fused_memory``) and turned this module into a pure re-export:
``from shared.storm_counter import StormCounter``. The full contract lives
in ``shared/tests/test_storm_counter.py``.

Task 3964 collapses this suite to the one thing it is uniquely qualified to
pin: that the old import path still resolves to the promoted class. The
assertion here is unconditional — unlike the mirrored check on the other
side, which must be able to skip since ``shared`` may not depend on
``fused_memory`` — because this module has no such constraint
(``fused_memory`` already depends on ``shared``).
"""

from __future__ import annotations

import shared.storm_counter

import fused_memory.server.storm_counter


def test_the_old_import_path_still_resolves_to_the_promoted_class():
    assert fused_memory.server.storm_counter.StormCounter is shared.storm_counter.StormCounter
