"""Tests for scripts/measure_plural_enum_guard_recall.py. (task 3949)

Loaded via importlib so the script (not on PYTHONPATH) can be tested
without sys.path pollution — mirrors the pattern in
test_cleanup_count_snapshots.py.

Every test here drives SYNTHETIC facts or a fake edge source. Nothing in
this file touches a live backend: the probe's whole point is that its
pure band (scan / triage / simulate / paginate / render) is checkable
without FalkorDB, so a broken probe fails in CI rather than silently
reporting a zero.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'measure_plural_enum_guard_recall.py'
)


def _load_module() -> types.ModuleType:
    """Load measure_plural_enum_guard_recall.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    mod_name = 'measure_plural_enum_guard_recall'
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
scan_corpus = _mod.scan_corpus


# The five synthetic facts below are the probe's own positive control: the
# headline live result is a ZERO, and a zero produced by broken wiring is
# indistinguishable from a zero produced by a clean corpus. Pinning one fact
# per outcome class means a probe that has silently stopped matching
# anything fails HERE rather than publishing a meaningless report.
_SUBJECT_POSITIVE = 'Tasks 1020 and 1030 are pending.'
_COMPLEMENT_REJECTION = 'Reviews of tasks 1020 and 1030 are pending.'
_PREAMBLE_REJECTION = 'As of 2026-08-09, tasks 1020 and 1030 are pending.'
# Carries the lexical precondition (`tasks <digits>`) but does NOT match
# PLURAL_ENUM_SNAPSHOT_RE — 'related to' is no status marker. This class is
# what separates 'the corpus has no plural-enum shapes at all' from 'the
# corpus has them and the guard eats them', so it gets its own counter.
_LEXICAL_NEAR_MISS = 'Tasks 1752 and 1753 are related to the uptime feed.'
_UNRELATED = 'The merge worker restarted after the fleet redeploy.'

_SYNTHETIC_CORPUS = [
    _SUBJECT_POSITIVE,
    _COMPLEMENT_REJECTION,
    _PREAMBLE_REJECTION,
    _LEXICAL_NEAR_MISS,
    _UNRELATED,
]


def test_scan_corpus_counts_matches_and_guard_rejections():
    """Every count field is asserted independently, not just the headline.

    A single aggregate assertion would pass while two counters were wrong
    in opposite directions. The fields are also deliberately not required
    to sum: a fact with two matches, one rejected and one surviving,
    counts toward BOTH guard_rejected and selected (see the probe's
    docstring for the counting rule).
    """
    result = scan_corpus(_SYNTHETIC_CORPUS)

    assert result.facts_scanned == 5
    assert result.lexical_precondition == 4  # every fact but _UNRELATED
    assert result.regex_matched == 3  # near-miss fails the regex itself
    assert result.guard_rejected == 2  # complement + preamble
    assert result.selected == 1  # the subject-position positive only


def test_scan_corpus_rejections_are_triageable_records():
    """A nonzero future run has to be diagnosable without a re-run.

    'Zero matches' is a point-in-time fact about a corpus that grows every
    cycle. When it stops being zero, whoever reads the report needs the
    offending fact text and the match offset in the artifact itself —
    otherwise the report says only that recall was lost, not where.
    """
    result = scan_corpus(_SYNTHETIC_CORPUS)

    assert len(result.rejections) == 2
    facts = [r.fact for r in result.rejections]
    assert facts == [_COMPLEMENT_REJECTION, _PREAMBLE_REJECTION]

    # The offsets are the matched span's start, so the fact can be split at
    # the exact prefix the guard was handed.
    by_fact = {r.fact: r.match_start for r in result.rejections}
    assert by_fact[_COMPLEMENT_REJECTION] == _COMPLEMENT_REJECTION.index('tasks 1020')
    assert by_fact[_PREAMBLE_REJECTION] == _PREAMBLE_REJECTION.index('tasks 1020')


def test_scan_corpus_of_empty_corpus_is_all_zeroes():
    """An empty corpus must be a clean zero, not a crash or a None.

    The live run enumerates several project graphs and at least one of them
    (knowlive) held zero valid edges at planning time, so this is a real
    input shape rather than a defensive one.
    """
    result = scan_corpus([])

    assert result.facts_scanned == 0
    assert result.lexical_precondition == 0
    assert result.regex_matched == 0
    assert result.guard_rejected == 0
    assert result.selected == 0
    assert result.rejections == []
