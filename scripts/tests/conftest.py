"""conftest.py for scripts/ tests — inserts scripts/ onto sys.path.

Mirrors the repo root conftest.py sys.path-insertion pattern so that
`import reviewer_redundancy_diagnostic` resolves when pytest collects
scripts/tests/ under importlib import mode.  No package __init__.py is
needed (importlib mode does not require it).

Also inserts scripts/legibility/ so flat modules nested one level down
(e.g. `digest.py`, and its PRD-decomposition siblings — the sampler,
merger, trickle coder, etc. all planned to live under scripts/legibility/)
resolve via a bare `import digest` the same way top-level scripts/*.py
modules do. Without this, scripts/ on sys.path alone only makes
scripts/legibility/ importable as a namespace package (`import legibility`),
not its contents as bare top-level names.
"""
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent  # scripts/tests/../ = scripts/
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

_LEGIBILITY_DIR = _SCRIPTS_DIR / 'legibility'
if str(_LEGIBILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_LEGIBILITY_DIR))
