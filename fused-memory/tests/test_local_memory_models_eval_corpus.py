"""Tests for scripts/local_memory_models_eval/build_corpus.py — the LME replay corpus.

PRD ``plans/local-memory-models-eval-prd.md`` task δ: a committed, re-derivable,
stratified sample of real ``dark_factory`` episodes that is **never conditioned
on the incumbent pipeline's outcome**. Consumers: ε (replay engine input),
ζ (control replays), θ (full arm replays).

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — the same ``_load_module()`` helper as
test_memory_eval_transcript_corpus.py / test_memory_eval_retrieval_probe.py.

Every test here runs against in-memory record lists or a hand-written store
double, EXCEPT the single ``@pytest.mark.integration`` smoke, which issues one
``GRAPH.RO_QUERY`` against the live graph. Nothing here ever writes: the store
double's tripwires enforce it offline, and ``GRAPH.RO_QUERY`` enforces it
server-side on the live path.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'local_memory_models_eval' / 'build_corpus.py'
)


def _load_module() -> types.ModuleType:
    """Load build_corpus.py from its file path.

    The module is registered in sys.modules under its name BEFORE
    ``exec_module`` so that ``@dataclass`` and other reflection-based
    decorators work correctly (they call ``sys.modules.get(cls.__module__)``),
    and build_corpus.py defines frozen dataclasses. See the note at
    test_memory_eval_retrieval_probe.py's copy of this helper.
    """
    mod_name = 'lme_build_corpus'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
