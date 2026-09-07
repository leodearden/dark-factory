"""Tests for scripts/normalize_topic_slugs.py — the corpus-wide topic
normalization migration (task 4878).

Sibling of ``test_retro_stamp_topics.py`` and deliberately built from the same
scaffolding: the importlib-by-path ``_load_module`` (``scripts/`` is not a
package and is not on PYTHONPATH), the autouse fixture neutralising the
store-mutation preflight, and ``AsyncMock`` doubles at the single I/O boundary.

What is DIFFERENT from that suite, and why it matters here: this script's sweep
is corpus-wide rather than id-bounded, so its enumeration, its collision
refusals and its residue verification all have to be pinned against injected
fakes.  No test in this file asserts a LIVE corpus count.  The corpus is live
and moving — the very history this task cites (98 -> 103 distinct non-conforming
values while the validator sat in warn mode) proves the number changes on its
own — so a test pinning it would be a doomed RED.  The baseline is a committed
constant and the tests pin the arithmetic against it, never the live number.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'normalize_topic_slugs.py'


def _load_module() -> types.ModuleType:
    """Load normalize_topic_slugs.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'normalize_topic_slugs'
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


@pytest.fixture(autouse=True)
def _neutralise_store_mutation_preflight(monkeypatch):
    """Keep this MOCK-unit suite independent of the REAL ``~/.mem0``.

    ``run(..., apply=True)`` runs a fail-closed capability preflight before it
    scrolls.  That probe touches the real filesystem, so without this fixture
    every ``--apply`` test would pass or fail according to whether the machine
    running pytest happens to be able to write mem0's history directory — and
    it genuinely cannot inside an agent sandbox, which is the whole reason the
    guard exists.  This suite is deliberately MOCK-unit, so the environment
    must not be an input to it.

    ``TestRunApplyStoreMutationPreflight`` re-rigs this per test — to refuse,
    to record, or to pass — so the guard's own behaviour is still pinned
    explicitly rather than assumed away.

    Deliberately NOT ``raising=False``: if the guard is ever removed from the
    script this fixture must break loudly rather than silently no-op.
    """
    monkeypatch.setattr(_mod, 'assert_store_mutation_allowed', lambda **_kw: None)


# ---------------------------------------------------------------------------
# Record fixtures — the shape ``scroll_all_by_metadata`` actually yields
# ---------------------------------------------------------------------------

def _rec(memory_id: str, topic: object = None, **meta) -> dict:
    """One scroll-shaped record.

    ``topic=None`` means "no ``topic`` key at all", which is a genuinely
    different case from ``topic`` present-and-``None`` — the caller spells the
    latter explicitly via ``metadata``.
    """
    metadata: dict = dict(meta)
    if topic is not None:
        metadata['topic'] = topic
    return {'id': memory_id, 'created_at': '2026-01-01T00:00:00Z', 'metadata': metadata}


def _by_reason(skips: list[dict], reason: str) -> list[dict]:
    return [s for s in skips if s.get('reason') == reason]


# ===========================================================================
# plan_renames — the pure planner
# ===========================================================================

class TestPlanRenames:
    """``plan_renames(records, *, project_id) -> (renames, skips)``.

    Pure: it takes plain scroll-shaped dicts and no service, so a wrong answer
    here can never be masked by a mock.  It decides what the migration WOULD
    do; nothing in it touches a store.
    """

    def test_non_conforming_topic_that_folds_yields_one_rename(self):
        renames, skips = _mod.plan_renames(
            [_rec('m1', 'mem0_tombstone_coverage')], project_id='dark_factory',
        )
        assert skips == []
        assert len(renames) == 1
        (rename,) = renames
        assert rename.project_id == 'dark_factory'
        assert rename.memory_id == 'm1'
        assert rename.old_topic == 'mem0_tombstone_coverage'
        assert rename.new_topic == 'mem0-tombstone-coverage'

    def test_rename_is_frozen(self):
        """A plan row must not be mutated between planning and writing.

        The whole point of planning the corpus before touching it is that the
        artifact and the writes agree; a mutable row lets them diverge.
        """
        (rename,), _ = _mod.plan_renames(
            [_rec('m1', 'a_b')], project_id='dark_factory',
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rename.new_topic = 'something-else'  # type: ignore[misc]

    def test_conforming_topic_is_neither_a_rename_nor_a_skip(self):
        """Already-good is not work, and a skip line would MISREPORT it.

        Most of the corpus conforms.  Filing each conforming record as a skip
        would bury the ~141 records that actually need attention under tens of
        thousands of lines saying "nothing to do" — and would make the skip
        buckets, which exist to surface coverage holes, unreadable.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1', 'already-good'), _rec('m2', 'x1-2y')],
            project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_record_without_a_topic_key_is_ignored_entirely(self):
        """No ``topic`` is retro_stamp_topics' job, not this script's.

        This sweep NORMALIZES the value of a topic that exists; stamping one
        onto a record that carries none is the bounded sibling's job, and
        claiming it here would double-report the same corpus gap in two
        artifacts with two different definitions of it.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1'), _rec('m2', canonical=True)], project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_topic_present_but_none_is_ignored_entirely(self):
        """A null ``topic`` is an absent topic, not a malformed one.

        ``derive_topic_slug(None)`` is ``None``, so without this case a null
        would land in the ``topic_unfoldable`` bucket and read as a value a
        human needs to adjudicate — when in fact there is nothing there.
        """
        renames, skips = _mod.plan_renames(
            [{'id': 'm1', 'created_at': None, 'metadata': {'topic': None}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_unfoldable_topic_is_a_skip_naming_the_raw_value(self):
        """No honest fold exists — report it, never guess one.

        The skip has to carry the RAW value: an operator resolving this by
        hand needs to see what is actually on the record, and the whole
        reason it is here is that the script refused to name a slug for it.
        """
        renames, skips = _mod.plan_renames(
            [_rec('m1', '!!!')], project_id='dark_factory',
        )
        assert renames == []
        entries = _by_reason(skips, 'topic_unfoldable')
        assert len(entries) == 1
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['memory_id'] == 'm1'
        assert entry['topic'] == '!!!'
        assert 'new_topic' not in entry, 'a refusal must not carry a guessed slug'

    def test_over_long_topic_is_unfoldable_not_truncated(self):
        """The cap refuses rather than repairs — pinned at the planner too.

        ``derive_topic_slug`` already returns ``None`` here, but the planner
        is where that ``None`` becomes an operator-visible line rather than a
        silently dropped record.
        """
        over = 'a' * 101
        renames, skips = _mod.plan_renames(
            [_rec('m1', over)], project_id='dark_factory',
        )
        assert renames == []
        assert _by_reason(skips, 'topic_unfoldable')[0]['topic'] == over

    def test_non_str_topic_is_unfoldable(self):
        """Untrusted values come straight off live records."""
        renames, skips = _mod.plan_renames(
            [{'id': 'm1', 'created_at': None, 'metadata': {'topic': 42}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert _by_reason(skips, 'topic_unfoldable')[0]['topic'] == 42

    def test_many_records_sharing_one_legacy_slug_each_get_a_rename(self):
        """One legacy VALUE can sit on many records; each needs its own write.

        The distinct-value count is what the baseline delta grades (item 4);
        the per-record count is what the write loop does.  Conflating them is
        how a migration reports "1 topic normalized" and leaves 7 records
        behind.
        """
        renames, _ = _mod.plan_renames(
            [_rec(f'm{i}', 'shared_legacy') for i in range(8)],
            project_id='dark_factory',
        )
        assert len(renames) == 8
        assert {r.new_topic for r in renames} == {'shared-legacy'}

    def test_output_is_sorted_for_a_clean_diff(self):
        """Sorted by ``(project_id, new_topic, memory_id)``.

        Two dry runs over an unchanged corpus must produce byte-comparable
        artifacts, and scroll order is not stable.  Without this an operator
        diffing consecutive rehearsals sees churn that means nothing.
        """
        renames, _ = _mod.plan_renames(
            [
                _rec('m9', 'zeta_topic'),
                _rec('m2', 'alpha_topic'),
                _rec('m1', 'zeta_topic'),
                _rec('m5', 'alpha_topic'),
            ],
            project_id='dark_factory',
        )
        assert [(r.new_topic, r.memory_id) for r in renames] == [
            ('alpha-topic', 'm2'),
            ('alpha-topic', 'm5'),
            ('zeta-topic', 'm1'),
            ('zeta-topic', 'm9'),
        ]

    def test_skips_are_sorted_too(self):
        """Same argument as the renames: a rehearsal diff must be readable."""
        _, skips = _mod.plan_renames(
            [_rec('m9', '!!!'), _rec('m1', '???')], project_id='dark_factory',
        )
        assert [s['memory_id'] for s in _by_reason(skips, 'topic_unfoldable')] == [
            'm1', 'm9',
        ]

    def test_record_without_an_id_is_skipped_not_planned(self):
        """A row with no id cannot be written to; refuse it loudly.

        ``update_memory`` is addressed by memory id.  A record whose id is
        missing or empty would otherwise reach the write boundary and fail
        there, one row at a time, inside a report that already claimed it as
        planned work.
        """
        renames, skips = _mod.plan_renames(
            [{'id': '', 'created_at': None, 'metadata': {'topic': 'a_b'}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert len(_by_reason(skips, 'record_without_id')) == 1

    def test_empty_input_is_empty_output(self):
        assert _mod.plan_renames([], project_id='dark_factory') == ([], [])
