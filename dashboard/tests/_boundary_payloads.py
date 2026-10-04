"""The served bodies the boundary suite carries across the wire.

Each scenario drives a REAL dashboard route over a fixture substrate and keeps
the body that route served. ``test_boundary_js.py`` asserts the server-side
rows on those bodies and writes them where ``js/boundary_*.test.mjs`` read
them, so the client half never applies a hand-built payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_all(workdir: Path) -> dict[str, Any]:
    """Every scenario's served body, keyed by the name the node half reads it under.

    *workdir* is an empty directory the scenarios may root their substrates in.
    """
    return {}
