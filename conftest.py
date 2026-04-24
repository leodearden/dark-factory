"""Root conftest to add all subproject src dirs to sys.path.

Also pre-imports the subproject packages so pytest's importlib-mode collection
does not register them as namespace packages (which would shadow the real
package and break `from <subproject>.foo import ...`).
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
for subproject in ['dashboard', 'escalation', 'fused-memory', 'orchestrator', 'shared']:
    _src = _ROOT / subproject / 'src'
    if _src.exists() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

# Pre-import the real package so pytest's rootdir-relative collection does not
# register the subproject directory (e.g. dashboard/) as a namespace package
# pointing at the project folder instead of its src/<name>/ subtree.
for pkg_name in ['dashboard', 'escalation', 'orchestrator', 'shared']:
    try:
        __import__(pkg_name)
    except ImportError:
        pass
# fused-memory's package is fused_memory (underscore)
try:
    __import__('fused_memory')
except ImportError:
    pass
