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


# ===========================================================================
# Tests: classify_pool (pure core)
# ===========================================================================

def _summary(id: str, created_at: str | None, content: str = 'ordinary cycle summary') -> dict:
    """Build a normalized {id, created_at, content, metadata} summary dict."""
    return {'id': id, 'created_at': created_at, 'content': content, 'metadata': {}}


class TestClassifyPool:
    """classify_pool(summaries, keep_recent_n) sorts by created_at descending,
    always keeps the most-recent keep_recent_n, and among the rest keeps only
    those carrying remediation history — the rest are marked for deletion."""

    def test_pool_below_keep_recent_n_all_kept_no_deletes(self):
        summaries = [
            _summary('a', '2026-03-01T00:00:00+00:00'),
            _summary('b', '2026-02-01T00:00:00+00:00'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'a', 'b'}
        assert result.delete_ids == []
        assert result.reasons['a'] == 'recent'
        assert result.reasons['b'] == 'recent'

    def test_pool_smaller_than_keep_recent_n_all_kept(self):
        summaries = [_summary('only', '2026-01-01T00:00:00+00:00')]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert result.keep_ids == ['only']
        assert result.delete_ids == []

    def test_older_quiescent_member_deleted(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary(
                'old-quiescent', '2026-01-01T00:00:00+00:00',
                content='0 new episodes, 0 mutations. Quiescent cycle.',
            ),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'newest1', 'newest2'}
        assert result.delete_ids == ['old-quiescent']
        assert result.reasons['old-quiescent'] == 'quiescent_boilerplate'
        assert result.reasons['newest1'] == 'recent'
        assert result.reasons['newest2'] == 'recent'

    def test_older_remediation_bearing_member_preserved(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary(
                'old-remediation', '2026-01-01T00:00:00+00:00',
                content='deleted entity e47ac10b-58cc-4372-a567-0e02b2c3d479',
            ),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert 'old-remediation' in result.keep_ids
        assert result.reasons['old-remediation'] == 'remediation'
        assert result.delete_ids == []

    def test_sorts_newest_first_deletes_oldest_beyond_cutoff(self):
        """4 datable members, keep_recent_n=2: the 2 newest are kept as
        'recent'; the 2 oldest (quiescent) are deleted — proves descending
        created_at ordering, not insertion order."""
        summaries = [
            _summary('third', '2026-03-01T00:00:00+00:00', content='0 mutations.'),
            _summary('newest', '2026-04-01T00:00:00+00:00', content='0 mutations.'),
            _summary('oldest', '2026-01-01T00:00:00+00:00', content='0 mutations.'),
            _summary('second', '2026-02-01T00:00:00+00:00', content='0 mutations.'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'newest', 'third'}
        assert set(result.delete_ids) == {'second', 'oldest'}

    def test_missing_created_at_sorts_last_but_eligible_for_remediation_preserve(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary('no-date-quiescent', None, content='0 new episodes, 0 mutations.'),
            _summary('no-date-remediation', None, content='deleted entity xyz-123'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert set(result.keep_ids) == {'newest1', 'newest2', 'no-date-remediation'}
        assert result.delete_ids == ['no-date-quiescent']
        assert result.reasons['no-date-remediation'] == 'remediation'
        assert result.reasons['no-date-quiescent'] == 'quiescent_boilerplate'

    def test_unparseable_created_at_treated_same_as_missing(self):
        summaries = [
            _summary('newest1', '2026-05-01T00:00:00+00:00'),
            _summary('newest2', '2026-04-01T00:00:00+00:00'),
            _summary('bad-date-quiescent', 'not-a-real-date', content='0 mutations, no mutations.'),
        ]
        result = _mod.classify_pool(summaries, keep_recent_n=2)

        assert 'bad-date-quiescent' in result.delete_ids
        assert result.reasons['bad-date-quiescent'] == 'quiescent_boilerplate'

    def test_empty_pool_returns_empty_decision(self):
        result = _mod.classify_pool([], keep_recent_n=2)

        assert result.keep_ids == []
        assert result.delete_ids == []
        assert result.reasons == {}
