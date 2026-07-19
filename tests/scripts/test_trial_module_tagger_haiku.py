"""Tests for scripts/trial_module_tagger_haiku.py — the offline haiku-vs-sonnet
module_tagger replay-agreement trial (task 2540).

The deterministic core (scoring, replay-input builder, adjudication tally,
decision function, report renderer, and the fixture-driven run_trial) is fully
unit-tested here. The live model run + conditional config flip are impl-only.

Resolved via tests/scripts/conftest.py, which puts scripts/ on sys.path.
"""
from __future__ import annotations

import pytest

from orchestrator import module_tagger_prompt

import trial_module_tagger_haiku as mod


# ── set_scores: precision/recall/F1/Jaccard over sanitized file sets ─────────

def test_set_scores_exact_match_is_all_ones():
    s = mod.set_scores(['a.py', 'b.py'], ['a.py', 'b.py'])
    assert set(s) == {'precision', 'recall', 'f1', 'jaccard'}
    assert s['precision'] == pytest.approx(1.0)
    assert s['recall'] == pytest.approx(1.0)
    assert s['f1'] == pytest.approx(1.0)
    assert s['jaccard'] == pytest.approx(1.0)


def test_set_scores_disjoint_is_all_zeros():
    s = mod.set_scores(['a.py'], ['b.py'])
    assert s['precision'] == pytest.approx(0.0)
    assert s['recall'] == pytest.approx(0.0)
    assert s['f1'] == pytest.approx(0.0)
    assert s['jaccard'] == pytest.approx(0.0)


def test_set_scores_partial_overlap_fractions():
    # predicted {a,b}, gold {b,c}: intersection {b}=1, union {a,b,c}=3.
    s = mod.set_scores(['a.py', 'b.py'], ['b.py', 'c.py'])
    assert s['precision'] == pytest.approx(0.5)   # 1/2 predicted correct
    assert s['recall'] == pytest.approx(0.5)      # 1/2 gold found
    assert s['f1'] == pytest.approx(0.5)          # harmonic mean of 0.5/0.5
    assert s['jaccard'] == pytest.approx(1.0 / 3.0)


def test_set_scores_empty_predicted_nonempty_gold_is_zero_no_zerodiv():
    s = mod.set_scores([], ['a.py'])
    assert s['precision'] == pytest.approx(0.0)
    assert s['recall'] == pytest.approx(0.0)
    assert s['f1'] == pytest.approx(0.0)
    assert s['jaccard'] == pytest.approx(0.0)


def test_set_scores_nonempty_predicted_empty_gold_is_zero_no_zerodiv():
    s = mod.set_scores(['a.py'], [])
    assert s['precision'] == pytest.approx(0.0)
    assert s['recall'] == pytest.approx(0.0)
    assert s['f1'] == pytest.approx(0.0)
    assert s['jaccard'] == pytest.approx(0.0)


def test_set_scores_both_empty_is_perfect_no_zerodiv():
    # Both sides predicted nothing: vacuously perfect agreement, defined
    # (no ZeroDivision). run_trial filters empty-gold tasks out, but the
    # primitive must still be total.
    s = mod.set_scores([], [])
    assert s['precision'] == pytest.approx(1.0)
    assert s['recall'] == pytest.approx(1.0)
    assert s['f1'] == pytest.approx(1.0)
    assert s['jaccard'] == pytest.approx(1.0)


def test_set_scores_strips_directory_shaped_entries_before_scoring():
    # 'somedir/' and 'orchestrator' are directory-shaped (no code extension)
    # → sanitize_files_for_persist strips them from BOTH sides before scoring,
    # so this reduces to an exact {a.py} vs {a.py} match.
    s = mod.set_scores(['a.py', 'somedir/', 'orchestrator'], ['a.py', 'pkg/'])
    assert s['precision'] == pytest.approx(1.0)
    assert s['recall'] == pytest.approx(1.0)
    assert s['f1'] == pytest.approx(1.0)
    assert s['jaccard'] == pytest.approx(1.0)


def test_set_scores_all_directory_predicted_scores_zero_against_real_gold():
    # An all-directory prediction sanitizes to [] → precision/recall/f1/jaccard
    # all 0.0 against a real (non-empty) gold set.
    s = mod.set_scores(['orchestrator', 'scripts/'], ['a.py'])
    assert s['precision'] == pytest.approx(0.0)
    assert s['recall'] == pytest.approx(0.0)
    assert s['f1'] == pytest.approx(0.0)
    assert s['jaccard'] == pytest.approx(0.0)


# ── build_replay_input / faithful_prompt_for: faithful "at the call site" ────

def test_build_replay_input_uses_description_when_present():
    task = {'id': 42, 'title': 'Fix foo', 'description': 'the description', 'details': 'the details'}
    summary = mod.build_replay_input(task, ['orchestrator'])
    assert summary == {'id': '42', 'title': 'Fix foo', 'description': 'the description'}


def test_build_replay_input_falls_back_to_details_when_description_empty():
    # Matches harness: description or details or '' (empty description → details).
    task = {'id': 7, 'title': 'Bar', 'description': '', 'details': 'from details'}
    summary = mod.build_replay_input(task, [])
    assert summary['description'] == 'from details'


def test_build_replay_input_empty_when_neither_present():
    task = {'id': 9, 'title': 'Baz'}
    summary = mod.build_replay_input(task, [])
    assert summary == {'id': '9', 'title': 'Baz', 'description': ''}


def test_build_replay_input_id_is_stringified():
    summary = mod.build_replay_input({'id': 123, 'title': 't', 'description': 'd'}, [])
    assert summary['id'] == '123'
    assert isinstance(summary['id'], str)


def test_faithful_prompt_for_delegates_to_shared_production_builder():
    task = {'id': 55, 'title': 'Add widget', 'description': 'wire the widget'}
    dirs = ['orchestrator', 'scripts', 'shared']

    summary = mod.build_replay_input(task, dirs)
    # The replay prompt is EXACTLY the production builder over a single-task
    # summary list — byte-identical inputs "at the call site".
    expected = module_tagger_prompt.build_tagger_prompt(dirs, [summary])
    assert mod.faithful_prompt_for(task, dirs) == expected


# ── frontier adjudication (D-6 shape): prompt + tally ────────────────────────

def test_build_adjudication_prompt_names_both_sets_and_ground_truth():
    task = {'id': 5, 'title': 'Add thing', 'description': 'wire it'}
    haiku = ['a.py', 'b.py']
    sonnet = ['a.py', 'c.py']
    gt = ['a.py', 'b.py']
    prompt = mod.build_adjudication_prompt(task, haiku, sonnet, gt)

    assert isinstance(prompt, str) and prompt.strip()
    # Both candidate file sets appear.
    for f in set(haiku) | set(sonnet):
        assert f in prompt
    # Ground truth appears.
    for f in gt:
        assert f in prompt
    # D-6 framing: which better matches ground truth, constrained answer.
    low = prompt.lower()
    assert 'ground truth' in low
    assert 'haiku' in low
    assert 'sonnet' in low
    assert 'tie' in low


def test_tally_adjudications_counts_and_worse_fraction():
    verdicts = [
        {'winner': 'haiku'},
        {'winner': 'haiku'},
        {'winner': 'sonnet'},
        {'winner': 'tie'},
    ]
    t = mod.tally_adjudications(verdicts)
    assert t['haiku_better'] == 2
    assert t['sonnet_better'] == 1
    assert t['tie'] == 1
    # haiku_worse_fraction = sonnet_better / max(1, total_non_tie) = 1/3.
    assert t['haiku_worse_fraction'] == pytest.approx(1.0 / 3.0)


def test_tally_adjudications_empty_is_zero_no_zerodiv():
    t = mod.tally_adjudications([])
    assert t['haiku_better'] == 0
    assert t['sonnet_better'] == 0
    assert t['tie'] == 0
    assert t['haiku_worse_fraction'] == pytest.approx(0.0)


def test_tally_adjudications_all_ties_worse_fraction_zero():
    t = mod.tally_adjudications([{'winner': 'tie'}, {'winner': 'tie'}])
    assert t['tie'] == 2
    assert t['haiku_worse_fraction'] == pytest.approx(0.0)


def test_tally_adjudications_all_sonnet_worse_fraction_one():
    t = mod.tally_adjudications([{'winner': 'sonnet'}, {'winner': 'sonnet'}])
    assert t['sonnet_better'] == 2
    assert t['haiku_worse_fraction'] == pytest.approx(1.0)
