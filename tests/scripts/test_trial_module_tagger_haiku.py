"""Tests for scripts/trial_module_tagger_haiku.py — the offline haiku-vs-sonnet
module_tagger replay-agreement trial (task 2540).

The deterministic core (scoring, replay-input builder, adjudication tally,
decision function, report renderer, and the fixture-driven run_trial) is fully
unit-tested here. The live model run + conditional config flip are impl-only.

Resolved via tests/scripts/conftest.py, which puts scripts/ on sys.path.
"""
from __future__ import annotations

import json

import pytest

from orchestrator import module_tagger_prompt

import trial_module_tagger_haiku as mod


def _extract_json_block(md: str) -> dict:
    """Parse the first ```json fenced block out of a rendered report."""
    start = md.index('```json')
    rest = md[start + len('```json'):]
    end = rest.index('```')
    return json.loads(rest[:end])


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


# ── decide: pass / marginal / fail against documented thresholds ─────────────

def _summary(*, haiku_f1, sonnet_f1, jaccard, worse, n):
    return {
        'mean_haiku_f1': haiku_f1,
        'mean_sonnet_f1': sonnet_f1,
        'mean_jaccard': jaccard,
        'haiku_worse_fraction': worse,
        'n_samples': n,
    }


def test_decision_thresholds_are_module_level_named_constants():
    assert mod.F1_PARITY_BAND == pytest.approx(0.05)
    assert mod.F1_FAIL_GAP == pytest.approx(0.15)
    assert mod.AGREEMENT_FLOOR == pytest.approx(0.70)
    assert mod.AGREEMENT_FAIL == pytest.approx(0.50)
    assert mod.ADJ_WORSE_FAIL == pytest.approx(0.60)
    assert mod.MIN_SAMPLES == 20


def test_decide_clear_parity_high_agreement_enough_samples_is_pass():
    s = _summary(haiku_f1=0.80, sonnet_f1=0.80, jaccard=0.85, worse=0.2, n=25)
    assert mod.decide(s) == 'pass'


def test_decide_f1_gap_beyond_fail_gap_is_fail():
    # sonnet_f1 - haiku_f1 = 0.30 > F1_FAIL_GAP (0.15).
    s = _summary(haiku_f1=0.50, sonnet_f1=0.80, jaccard=0.85, worse=0.2, n=25)
    assert mod.decide(s) == 'fail'


def test_decide_agreement_below_fail_floor_is_fail():
    s = _summary(haiku_f1=0.80, sonnet_f1=0.80, jaccard=0.40, worse=0.2, n=25)
    assert mod.decide(s) == 'fail'


def test_decide_opus_majority_worse_is_fail():
    # haiku_worse_fraction 0.70 > ADJ_WORSE_FAIL (0.60).
    s = _summary(haiku_f1=0.80, sonnet_f1=0.80, jaccard=0.85, worse=0.70, n=25)
    assert mod.decide(s) == 'fail'


def test_decide_too_few_samples_is_marginal():
    # Otherwise-passing, but N < MIN_SAMPLES → marginal (escalate to human).
    s = _summary(haiku_f1=0.80, sonnet_f1=0.80, jaccard=0.85, worse=0.2, n=10)
    assert mod.decide(s) == 'marginal'


def test_decide_f1_in_between_band_is_marginal():
    # gap 0.10 is between parity (0.05) and fail-gap (0.15) → marginal.
    s = _summary(haiku_f1=0.70, sonnet_f1=0.80, jaccard=0.85, worse=0.2, n=25)
    assert mod.decide(s) == 'marginal'


def test_decide_agreement_in_between_band_is_marginal():
    # jaccard 0.60 between AGREEMENT_FAIL (0.50) and AGREEMENT_FLOOR (0.70).
    s = _summary(haiku_f1=0.80, sonnet_f1=0.80, jaccard=0.60, worse=0.2, n=25)
    assert mod.decide(s) == 'marginal'


def test_decide_adjudication_in_between_band_is_marginal():
    # worse 0.55 is above the 0.5 pass ceiling but not beyond the 0.60 fail
    # floor → marginal (locks decision 5's two adjudication boundaries).
    s = _summary(haiku_f1=0.80, sonnet_f1=0.80, jaccard=0.85, worse=0.55, n=25)
    assert mod.decide(s) == 'marginal'


# ── render_report: markdown leaderboard + machine-readable summary ───────────

def _trial_result(decision='pass'):
    return mod.TrialResult(
        n_samples=22,
        haiku={'precision': 0.80, 'recall': 0.75, 'f1': 0.77},
        sonnet={'precision': 0.82, 'recall': 0.78, 'f1': 0.80},
        agreement={'precision': 0.90, 'recall': 0.88, 'f1': 0.89, 'jaccard': 0.85},
        adjudication={'haiku_better': 3, 'sonnet_better': 2, 'tie': 5, 'haiku_worse_fraction': 0.4},
        decision=decision,
    )


def test_render_report_is_markdown_with_leaderboard_and_verdict():
    md = mod.render_report(_trial_result('pass'))
    assert isinstance(md, str) and md.strip()

    low = md.lower()
    # Leaderboard: both models named + precision/recall/F1 columns.
    assert 'haiku' in low
    assert 'sonnet' in low
    assert 'precision' in low
    assert 'recall' in low
    assert 'f1' in low
    # Mean haiku-vs-sonnet agreement / Jaccard.
    assert 'jaccard' in low
    # Adjudication tally + decision verdict + N.
    assert 'adjudication' in low
    assert 'pass' in low
    assert '22' in md


def test_render_report_lists_threshold_constants_used():
    md = mod.render_report(_trial_result('pass'))
    # The design thresholds the decision was computed against.
    assert 'F1_PARITY_BAND' in md
    assert 'MIN_SAMPLES' in md
    assert '0.05' in md
    assert '20' in md


def test_render_report_embeds_machine_readable_summary_block():
    md = mod.render_report(_trial_result('marginal'))
    payload = _extract_json_block(md)
    # The act step (step 16) parses this block for the decision + numbers.
    assert payload['decision'] == 'marginal'
    assert payload['n_samples'] == 22
    assert payload['haiku']['f1'] == pytest.approx(0.77)
    assert payload['sonnet']['f1'] == pytest.approx(0.80)
    assert payload['agreement']['jaccard'] == pytest.approx(0.85)
    assert payload['adjudication']['haiku_worse_fraction'] == pytest.approx(0.4)
