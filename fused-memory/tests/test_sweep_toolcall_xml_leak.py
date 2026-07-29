"""Tests for scripts/sweep_toolcall_xml_leak.py — the PURE core (task 3083).

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution -- mirrors test_clear_malformed_empty_memory.py, which is
also this script's structural template (pure core / thin async I/O /
orchestration / CLI, dry-run default, loud refusal under --apply).

The async orchestration tests live alongside these in a later step; everything
here is pure — no I/O, no mocks.

## What the classification has to get right

The sweep repairs stored memory text, so its fail-safe direction is the whole
design. Two shapes are recoverable WITHOUT losing a single character of real
content, and only those two are ever auto-repaired:

  * ``repairable_tail``    — the leak fragment runs to end-of-content and
    carries nothing after its own marker (the c759c53b shape). Dropping it
    restores exactly the text that was written.
  * ``repairable_duplicate`` — the text after the marker is a VERBATIM
    duplicate of the text before it (the 9f2d2ae6 shape, where the harness
    re-appended the truncated argument's remainder into the same field).
    Dropping it restores exactly the text that was written.

Anything else carrying a leak is ``manual_review`` and is NEVER mutated,
mirroring ``is_malformed_empty_payload``'s all-conditions-required fail-safe:
the predicate must be structurally incapable of authorizing the destruction of
real content.

## Sentinel-literal hazard (task 3083, pre-1)

Every tool-call sentinel below is spelled with the ``\\x3c`` escape for ``<``.
Writing one verbatim would make THIS file's own authoring tool call terminate
early — reproducing the bug under test. Byte-identical at runtime; leave them
escaped.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'sweep_toolcall_xml_leak.py'


def _load_module() -> types.ModuleType:
    """Load sweep_toolcall_xml_leak.py from its file path."""
    mod_name = 'sweep_toolcall_xml_leak'
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

_BODY = 'The merge worker consumes the stash stack in project_root.'

# c759c53b shape: stray closing tag, real whitespace, bare closing invoke tag,
# nothing after it.
_TAIL_LEAK = _BODY + '\x3c/content>\n\x3c/invoke>'

# 9f2d2ae6 shape: stray closing tag, real whitespace, a serialized opening
# parameter tag, then a VERBATIM duplicate of the body.
_DUPLICATE_LEAK = _BODY + '\x3c/content>\n\x3cparameter name="content">' + _BODY

# Same marker, but the trailing text is NEITHER absent NOR a duplicate — a
# genuinely different second value. Auto-repairing this would destroy content.
_MANUAL_LEAK = (
    _BODY + '\x3c/content>\n\x3cparameter name="content">A different second value entirely.'
)


def _payload(content: str, key: str = 'data') -> dict:
    return {key: content, 'category': 'observations_and_summaries', 'agent_id': 'claude-x'}


class TestClassifyRecord:
    """classify_record(payload) -> 'clean'|'repairable_tail'|'repairable_duplicate'|'manual_review'."""

    def test_clean_payload_is_clean(self):
        assert _mod.classify_record(_payload(_BODY)) == 'clean'

    def test_payload_with_no_content_is_clean(self):
        """An empty/absent content payload has nothing to repair."""
        assert _mod.classify_record({}) == 'clean'

    def test_tail_shape_is_repairable_tail(self):
        assert _mod.classify_record(_payload(_TAIL_LEAK)) == 'repairable_tail'

    def test_duplicate_shape_is_repairable_duplicate(self):
        assert _mod.classify_record(_payload(_DUPLICATE_LEAK)) == 'repairable_duplicate'

    def test_non_duplicate_trailing_text_is_manual_review(self):
        """The fail-safe case: unrecognised trailing text is never auto-dropped."""
        assert _mod.classify_record(_payload(_MANUAL_LEAK)) == 'manual_review'

    def test_leak_at_offset_zero_is_manual_review(self):
        """Removing the fragment would leave nothing, so there is no repair that
        preserves content — refuse rather than write an empty memory."""
        assert _mod.classify_record(_payload('\x3c/content>\n\x3c/invoke>')) == 'manual_review'

    def test_content_is_read_through_the_canonical_key_order(self):
        """'data' is the canonical Qdrant payload key, with 'memory'/'content'
        as fallbacks — judged identically to the existing dedup sweep."""
        for key in ('data', 'memory', 'content'):
            assert _mod.classify_record({key: _TAIL_LEAK}) == 'repairable_tail', key
        assert _mod._CONTENT_KEYS == ('data', 'memory', 'content')


class TestRepairContent:
    """repair_content(text) -> str | None — None means REFUSE, never mutate."""

    def test_tail_shape_is_repaired_to_the_original_body(self):
        assert _mod.repair_content(_TAIL_LEAK) == _BODY

    def test_duplicate_shape_is_repaired_to_a_single_copy_of_the_body(self):
        assert _mod.repair_content(_DUPLICATE_LEAK) == _BODY

    def test_manual_review_shape_returns_none(self):
        """THE safety assertion: a record the classifier cannot vouch for can
        never be auto-mutated, because there is no repaired string to write."""
        assert _mod.repair_content(_MANUAL_LEAK) is None

    def test_clean_text_is_returned_unchanged(self):
        """A no-op repair on clean text is what makes repair idempotent; None is
        reserved exclusively for 'refuse to touch this'."""
        assert _mod.repair_content(_BODY) == _BODY

    @pytest.mark.parametrize('text', [_TAIL_LEAK, _DUPLICATE_LEAK, _BODY])
    def test_repair_is_idempotent(self, text):
        once = _mod.repair_content(text)
        assert _mod.repair_content(once) == once

    @pytest.mark.parametrize('text', [_TAIL_LEAK, _DUPLICATE_LEAK, _BODY])
    def test_repair_never_empties_non_empty_input(self, text):
        """A repair that emptied a memory would be a worse corruption than the
        leak it fixed."""
        assert _mod.repair_content(text)


class TestBuildReport:
    """build_report — pure assembly of already-computed fields, no I/O."""

    def _records(self):
        return [
            {'id': 'a', 'classification': 'clean', 'repaired': False},
            {'id': 'b', 'classification': 'repairable_tail', 'repaired': True},
            {'id': 'c', 'classification': 'manual_review', 'repaired': False},
        ]

    def test_report_carries_scope_mode_and_counts(self):
        report = _mod.build_report(
            project_id='dark_factory',
            collection='fused_dark_factory',
            dry_run=True,
            exhaustive=False,
            scanned=3,
            truncated=False,
            limit=None,
            records=self._records(),
        )

        assert report['project_id'] == 'dark_factory'
        assert report['collection'] == 'fused_dark_factory'
        assert report['dry_run'] is True
        assert report['exhaustive'] is False
        assert report['scanned'] == 3
        assert report['truncated'] is False
        assert report['limit'] is None
        assert report['records'] == self._records()

    def test_counts_are_tallied_per_classification(self):
        report = _mod.build_report(
            project_id='dark_factory',
            collection='fused_dark_factory',
            dry_run=True,
            exhaustive=True,
            scanned=3,
            truncated=False,
            limit=None,
            records=self._records(),
        )

        assert report['counts']['clean'] == 1
        assert report['counts']['repairable_tail'] == 1
        assert report['counts']['repairable_duplicate'] == 0
        assert report['counts']['manual_review'] == 1
        assert report['repaired'] == 1


class TestResolveExitCode:
    """resolve_exit_code — the loud refusal, pure and sync."""

    def _report(self, **overrides):
        base = {
            'dry_run': True,
            'truncated': False,
            'counts': {
                'clean': 1,
                'repairable_tail': 0,
                'repairable_duplicate': 0,
                'manual_review': 0,
            },
        }
        base.update(overrides)
        return base

    def test_clean_corpus_exits_zero(self):
        assert _mod.resolve_exit_code(self._report(dry_run=False)) == 0

    def test_dry_run_always_exits_zero(self):
        """A dry run mutates nothing, so it is never a failure — even when it
        finds manual_review records. The printed report IS the investigation."""
        counts = {
            'clean': 0,
            'repairable_tail': 0,
            'repairable_duplicate': 0,
            'manual_review': 5,
        }
        assert _mod.resolve_exit_code(self._report(counts=counts, truncated=True)) == 0

    def test_apply_with_manual_review_exits_non_zero(self):
        """The confidently-classified records are still repaired, but the
        operator must not be able to mistake a partial sweep for a complete
        one."""
        counts = {
            'clean': 0,
            'repairable_tail': 2,
            'repairable_duplicate': 0,
            'manual_review': 1,
        }
        assert _mod.resolve_exit_code(self._report(dry_run=False, counts=counts)) != 0

    def test_apply_that_was_truncated_exits_non_zero(self):
        """Same rationale as manual_review: a --limit-capped or otherwise
        truncated apply covered an unknown fraction of the corpus, and reporting
        success would make a silently partial sweep look complete."""
        assert _mod.resolve_exit_code(self._report(dry_run=False, truncated=True)) != 0


class TestBuildParser:
    """_build_parser — the CLI surface, testable without any live I/O."""

    def test_dry_run_is_the_default(self):
        args = _mod._build_parser().parse_args([])
        assert args.apply is False
        assert args.project_id == 'dark_factory'
        assert args.exhaustive is False
        assert args.config is None
        assert args.limit is None

    def test_flags_parse(self):
        args = _mod._build_parser().parse_args(
            ['--apply', '--exhaustive', '--project-id', 'reify', '--limit', '50',
             '--config', '/tmp/c.yaml']
        )
        assert args.apply is True
        assert args.exhaustive is True
        assert args.project_id == 'reify'
        assert args.limit == 50
        assert args.config == '/tmp/c.yaml'
