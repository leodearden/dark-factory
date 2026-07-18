"""Tests for scripts/strip_leaked_control_keys.py (task 2735).

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_cleanup_count_snapshots.py / test_migrate_cross_graph_leak.py.

This suite covers only the script's pure transform, ``correct_task_metadata``.
The live httpx/JSON-RPC plumbing (modeled on
scripts/migrate_metadata_modules_to_files.py) is exercised operationally,
not unit-tested here — see the script's module docstring.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'strip_leaked_control_keys.py'


def _load_module() -> types.ModuleType:
    """Load strip_leaked_control_keys.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly (they call
    sys.modules.get(cls.__module__)).
    """
    mod_name = 'strip_leaked_control_keys'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()
correct_task_metadata = _mod.correct_task_metadata


# A realistic task-2682-shaped metadata dict: legit keys — including the
# non-blessed-but-legitimate `_causation_id` / `gate_escalated_at`
# annotations — plus the errant `append` key that leaked in from the
# Stage-2 recon LLM's call-flag/metadata-field conflation.
_TASK_2682_SHAPED_METADATA = {
    'source': 'reconciliation',
    'prd_path': 'plans/some-prd.md',
    'prd_task_label': 'Some Task',
    'memory_hints': {'entities': ['A'], 'queries': ['q1']},
    '_causation_id': '8d3af293-0000-0000-0000-000000000000',
    'gate_escalated_at': '2026-07-17T00:00:00+00:00',
    'always_escalates': True,
    'task_kind': 'deterministic',
    'append': True,
}


def test_correct_task_metadata_drops_append_preserves_every_other_key():
    """The corrected dict is the input minus `append`; every legit key —
    including non-blessed annotations — is preserved with identical values."""
    cleaned, dropped = correct_task_metadata(_TASK_2682_SHAPED_METADATA)
    expected = {k: v for k, v in _TASK_2682_SHAPED_METADATA.items() if k != 'append'}
    assert cleaned == expected
    assert dropped == ['append']


def test_correct_task_metadata_idempotent_on_already_corrected_dict():
    """Re-running against an already-clean dict is a no-op."""
    cleaned, _ = correct_task_metadata(_TASK_2682_SHAPED_METADATA)
    twice_cleaned, dropped_again = correct_task_metadata(cleaned)
    assert twice_cleaned == cleaned
    assert dropped_again == []
