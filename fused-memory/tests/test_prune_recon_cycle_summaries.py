"""Tests for scripts/prune_recon_cycle_summaries.py.

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in test_cleanup_count_snapshots.py.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'prune_recon_cycle_summaries.py'


def _load_module() -> types.ModuleType:
    """Load prune_recon_cycle_summaries.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators (e.g. @dataclass) work correctly.
    """
    mod_name = 'prune_recon_cycle_summaries'
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


# ===========================================================================
# Tests: carries_remediation_history
# ===========================================================================

class TestCarriesRemediationHistory:
    """Tests for carries_remediation_history(content) -> bool.

    Fail-safe contract: deletion is irreversible, so the predicate must default
    to True (preserve) whenever content is empty, unparseable, or ambiguous.
    It only returns False for content that is CLEARLY pure-quiescent
    boilerplate — a quiescent marker present with no remediation signal.
    """

    def _call(self, content: str) -> bool:
        return _mod.carries_remediation_history(content)

    # --- Pure-quiescent boilerplate -> False ---

    def test_quiescent_cycle_marker_alone(self):
        content = 'Cycle summary: quiescent cycle, nothing to report.'
        assert self._call(content) is False

    def test_zero_new_episodes_and_zero_mutations(self):
        content = 'Stage 1 cycle summary: 0 new episodes, 0 mutations this cycle.'
        assert self._call(content) is False

    def test_no_mutations_phrase(self):
        content = 'Reviewed all recent memories; no mutations were necessary.'
        assert self._call(content) is False

    def test_zero_episodes_multi_sentence_boilerplate(self):
        content = (
            'Cycle summary for memory_consolidator run abc123: '
            '0 new episodes processed. 0 mutations applied. '
            'System is stable and quiescent cycle confirmed.'
        )
        assert self._call(content) is False

    def test_case_insensitive_quiescent_marker(self):
        content = 'CYCLE SUMMARY: QUIESCENT CYCLE. 0 MUTATIONS.'
        assert self._call(content) is False

    # --- Real remediation -> True ---

    def test_deleted_entity_mentioned(self):
        content = 'Cycle summary: deleted entity e47ac10b-58cc-4372-a567-0e02b2c3d479 (duplicate).'
        assert self._call(content) is True

    def test_invalidated_edge_mentioned(self):
        content = 'Invalidated edge abc-123 because it contained stale count-snapshot text.'
        assert self._call(content) is True

    def test_flag_processed_mentioned(self):
        content = 'Cycle summary: 0 new episodes. 1 flag processed for task 1942 escalation.'
        assert self._call(content) is True

    def test_merged_memory_mentioned(self):
        content = 'Cycle summary: merged memory records for duplicate entities.'
        assert self._call(content) is True

    def test_nonzero_mutations_count(self):
        content = 'Cycle summary: 3 mutations applied this cycle.'
        assert self._call(content) is True

    def test_nonzero_deletions_count(self):
        content = 'Cycle summary: 2 deletions performed to clean up stale markers.'
        assert self._call(content) is True

    def test_refreshed_entity_mentioned(self):
        content = 'Cycle summary: 0 new episodes, 0 mutations. Refreshed entity summary for e1 as follow-up.'
        assert self._call(content) is True

    def test_edge_correction_mentioned(self):
        content = 'Cycle summary: 0 mutations logged, but one edge correction applied manually.'
        assert self._call(content) is True

    # --- Fail-safe: empty / unparseable / ambiguous -> True ---

    def test_empty_string(self):
        assert self._call('') is True

    def test_whitespace_only(self):
        assert self._call('   \n\t  ') is True

    def test_ambiguous_text_with_no_markers(self):
        content = 'Unrelated free-form note with no cycle-summary markers at all.'
        assert self._call(content) is True

    def test_ambiguous_short_content(self):
        assert self._call('...') is True
