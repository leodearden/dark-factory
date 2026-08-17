"""Pin for ``fused_memory.server.storm_counter`` as a re-export shim.

Task 3689 promoted ``StormCounter`` to ``shared.storm_counter`` (because
``shared.mcp_markup_middleware`` needed it and ``shared`` may not import
``fused_memory``) and turned this module into a pure re-export:
``from shared.storm_counter import StormCounter``. The full contract is now
tested once, at the new home, in ``shared/tests/test_storm_counter.py``.

The two suites used to duplicate ~85% of each other's cases despite exercising
the SAME class object — the INV-5 shape task 3689 exists elsewhere to repair.
Task 3964 collapses this suite to the one thing it is uniquely qualified to
pin: that the old import path still resolves to the promoted class.

``shared/tests/test_storm_counter.py::TestReExportShim`` asserts the same
identity from the other side, but only behind ``pytest.importorskip`` — since
``shared`` may not depend on ``fused_memory``, that check must degrade to a
skip when ``fused_memory`` is not installed. This module has no such
constraint (``fused_memory`` already depends on ``shared``), so it is the one
place this identity is checked unconditionally.
"""

from __future__ import annotations

import fused_memory.server.storm_counter
import shared.storm_counter


def test_the_old_import_path_still_resolves_to_the_promoted_class():
    assert fused_memory.server.storm_counter.StormCounter is shared.storm_counter.StormCounter
