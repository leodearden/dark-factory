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

import trial_module_tagger_haiku as mod  # pyright: ignore[reportMissingImports]


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


def test_decide_agreement_floor_gated_when_challenger_wins_ground_truth_f1():
    # esc-2540-17 / option d (Leo-ratified, 2026-07-21): the haiku-vs-sonnet
    # agreement floor must NOT hard-fail the trial when the challenger (haiku)
    # BEATS the incumbent on the PRIMARY quality signal — ground-truth mean F1.
    # A low agreement Jaccard there reflects the incumbent (sonnet) being the
    # weaker model, not the challenger being unreliable — the opposite of the
    # divergence the floor was designed to catch. These are the actual recent-30
    # live-run numbers (haiku F1 0.370 > sonnet 0.267; Jaccard 0.368 < AGREEMENT_FAIL).
    s = _summary(haiku_f1=0.370, sonnet_f1=0.267, jaccard=0.368, worse=0.273, n=30)
    assert mod.decide(s) != 'fail'
    # Jaccard is still below AGREEMENT_FLOOR (0.70), so it cannot PASS either →
    # marginal (escalate to a human), never an auto-flip on low agreement alone.
    assert mod.decide(s) == 'marginal'


def test_decide_agreement_floor_still_fails_when_challenger_not_better_on_f1():
    # The gate is CONDITIONAL, not a blanket removal: when the challenger is NOT
    # better than the incumbent on ground-truth F1 (here haiku slightly worse,
    # gap 0.05 within F1_FAIL_GAP so the F1 trigger itself does not fire), a
    # sub-floor agreement Jaccard STILL hard-fails.
    s = _summary(haiku_f1=0.55, sonnet_f1=0.60, jaccard=0.40, worse=0.2, n=25)
    assert mod.decide(s) == 'fail'


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


# ── run_trial: end-to-end deterministic orchestration over fixtures ──────────

class _FakeResult:
    """Minimal AgentResult stand-in that parse_predictions can read.

    Carries the production tagger output_schema shape
    ``{"predictions": [{id, files}]}`` on ``structured_output``.
    """

    def __init__(self, files):
        self.structured_output = {'predictions': [{'id': 'x', 'files': list(files)}]}
        self.output = None


_DIRS = ['orchestrator', 'scripts', 'shared']

# Eligible: done tasks carrying non-empty metadata.files (the ground truth).
_T1 = {'id': 1, 'title': 'exact match', 'description': 'd1',
       'metadata': {'files': ['a.py', 'b.py']}}
_T2 = {'id': 2, 'title': 'haiku misses one', 'description': 'd2',
       'metadata': {'files': ['x.py', 'y.py']}}
_T3 = {'id': 3, 'title': 'haiku over-predicts', 'description': 'd3',
       'metadata': {'files': ['m.py']}}
_T4 = {'id': 4, 'title': 'both partial', 'description': 'd4',
       'metadata': {'files': ['p.py', 'q.py', 'r.py']}}
# Excluded: deterministic (task_kind), empty files, absent files key, no metadata.
_T5_DET = {'id': 5, 'title': 'deterministic', 'description': 'd5',
           'metadata': {'files': ['z.py'], 'task_kind': 'deterministic'}}
_T6_EMPTY = {'id': 6, 'title': 'empty files', 'description': 'd6',
             'metadata': {'files': []}}
_T7_NOFILES = {'id': 7, 'title': 'no files key', 'description': 'd7',
               'metadata': {}}
_T8_NOMETA = {'id': 8, 'title': 'no metadata', 'description': 'd8'}

_ELIGIBLE = (_T1, _T2, _T3, _T4)
_EXCLUDED = (_T5_DET, _T6_EMPTY, _T7_NOFILES, _T8_NOMETA)
_ALL_TASKS = [_T1, _T2, _T3, _T4, _T5_DET, _T6_EMPTY, _T7_NOFILES, _T8_NOMETA]

# (haiku prediction, sonnet prediction) per eligible task id.
_PREDS = {
    '1': (['a.py', 'b.py'], ['a.py', 'b.py']),        # agree → NO adjudication
    '2': (['x.py'], ['x.py', 'y.py']),                # disagree → adjudicate
    '3': (['m.py', 'extra.py'], ['m.py']),            # disagree → adjudicate
    '4': (['p.py', 'q.py'], ['p.py', 'z.py']),        # disagree → adjudicate
}
# Frontier (opus) winners keyed by task id — only disagreements are adjudicated.
_WINNERS = {'2': 'sonnet', '3': 'haiku', '4': 'tie'}


def _make_invoke_fn(calls):
    """Fake invoke_fn(prompt, model) → _FakeResult of canned predictions.

    ``canned`` is keyed on the FAITHFUL prompt (``faithful_prompt_for``), so a
    non-faithful prompt from run_trial raises KeyError — the lookup itself
    asserts prompt fidelity. Records every (prompt, model) call for inspection.
    """
    canned = {}
    for t in _ELIGIBLE:
        prompt = mod.faithful_prompt_for(t, _DIRS)
        haiku_pred, sonnet_pred = _PREDS[str(t['id'])]
        canned[prompt] = {'haiku': haiku_pred, 'sonnet': sonnet_pred}

    def invoke_fn(prompt, model):
        calls.append((prompt, model))
        return _FakeResult(canned[prompt][model])

    return invoke_fn


def _make_adjudicate_fn(calls):
    """Fake adjudicate_fn(prompt) → {'winner': ...} keyed on the task id.

    The task id is read back out of the adjudication prompt. ``_WINNERS`` only
    has entries for disagreeing tasks, so an adjudication of an AGREEING task
    (a bug) would KeyError — guarding "adjudicate only on disagreements".
    """
    import re

    def adjudicate_fn(prompt):
        calls.append(prompt)
        match = re.search(r'^id: (\S+)$', prompt, re.MULTILINE)
        assert match, f'adjudication prompt carries no "id:" line: {prompt!r}'
        tid = match.group(1)
        return {'winner': _WINNERS[tid]}

    return adjudicate_fn


def test_run_trial_invokes_each_eligible_task_per_model_with_faithful_prompt():
    invoke_calls, adj_calls = [], []
    mod.run_trial(_ALL_TASKS, _DIRS, _make_invoke_fn(invoke_calls),
                  _make_adjudicate_fn(adj_calls))

    # 4 eligible tasks × {haiku, sonnet} = 8 invocations; nothing else.
    assert len(invoke_calls) == 8
    for t in _ELIGIBLE:
        prompt = mod.faithful_prompt_for(t, _DIRS)
        assert (prompt, 'haiku') in invoke_calls
        assert (prompt, 'sonnet') in invoke_calls
    # Excluded tasks (deterministic / empty / absent files / no metadata) never invoked.
    for t in _EXCLUDED:
        prompt = mod.faithful_prompt_for(t, _DIRS)
        assert (prompt, 'haiku') not in invoke_calls
        assert (prompt, 'sonnet') not in invoke_calls
    # Exactly the two trial models were used.
    assert {m for _, m in invoke_calls} == {'haiku', 'sonnet'}


def test_run_trial_adjudicates_only_haiku_sonnet_disagreements():
    import re
    invoke_calls, adj_calls = [], []
    mod.run_trial(_ALL_TASKS, _DIRS, _make_invoke_fn(invoke_calls),
                  _make_adjudicate_fn(adj_calls))

    def _task_id(prompt: str) -> str:
        match = re.search(r'^id: (\S+)$', prompt, re.MULTILINE)
        assert match, f'adjudication prompt carries no "id:" line: {prompt!r}'
        return match.group(1)

    # T1 haiku==sonnet (exact agreement) → NOT adjudicated; T2/T3/T4 disagree.
    adjudicated_ids = {_task_id(p) for p in adj_calls}
    assert adjudicated_ids == {'2', '3', '4'}
    assert len(adj_calls) == 3


def test_run_trial_result_aggregates_scores_tally_and_decision():
    invoke_calls, adj_calls = [], []
    result = mod.run_trial(_ALL_TASKS, _DIRS, _make_invoke_fn(invoke_calls),
                           _make_adjudicate_fn(adj_calls))

    assert isinstance(result, mod.TrialResult)
    assert result.n_samples == 4  # only the eligible tasks

    # Per-model mean precision/recall/F1 vs ground truth — computed via the
    # already-tested set_scores primitive so this pins orchestration, not arithmetic.
    def _mean_vs_gold(key, side):
        vals = []
        for t in _ELIGIBLE:
            haiku_pred, sonnet_pred = _PREDS[str(t['id'])]
            pred = haiku_pred if side == 'haiku' else sonnet_pred
            vals.append(mod.set_scores(pred, t['metadata']['files'])[key])
        return sum(vals) / len(vals)

    for key in ('precision', 'recall', 'f1'):
        assert result.haiku[key] == pytest.approx(_mean_vs_gold(key, 'haiku'))
        assert result.sonnet[key] == pytest.approx(_mean_vs_gold(key, 'sonnet'))

    # Mean symmetric haiku-vs-sonnet agreement = mean set_scores(haiku, sonnet).
    def _mean_agreement(key):
        vals = []
        for t in _ELIGIBLE:
            haiku_pred, sonnet_pred = _PREDS[str(t['id'])]
            vals.append(mod.set_scores(haiku_pred, sonnet_pred)[key])
        return sum(vals) / len(vals)

    for key in ('precision', 'recall', 'f1', 'jaccard'):
        assert result.agreement[key] == pytest.approx(_mean_agreement(key))

    # Adjudication tally: T3→haiku, T2→sonnet, T4→tie.
    assert result.adjudication['haiku_better'] == 1
    assert result.adjudication['sonnet_better'] == 1
    assert result.adjudication['tie'] == 1
    assert result.adjudication['haiku_worse_fraction'] == pytest.approx(0.5)

    # N=4 < MIN_SAMPLES → marginal, and it must equal decide() over the same
    # summary the result exposes (decide is wired into the pipeline).
    assert result.decision == 'marginal'
    assert result.decision == mod.decide({
        'mean_haiku_f1': result.haiku['f1'],
        'mean_sonnet_f1': result.sonnet['f1'],
        'mean_jaccard': result.agreement['jaccard'],
        'haiku_worse_fraction': result.adjudication['haiku_worse_fraction'],
        'n_samples': result.n_samples,
    })


# ── select_trial_sample: bounded MOST-RECENT eligible slice (esc-2540-16) ────

def test_default_sample_size_is_above_min_samples():
    # The bounded-recent default must clear the too-few-samples marginal band.
    assert mod.DEFAULT_SAMPLE_SIZE > mod.MIN_SAMPLES


def test_select_trial_sample_filters_to_eligible_only():
    # Deterministic / empty-files / no-files / no-metadata tasks never selected.
    selected = mod.select_trial_sample(_ALL_TASKS, size=None)
    assert {t['id'] for t in selected} == {t['id'] for t in _ELIGIBLE}


def test_select_trial_sample_drops_task_whose_gold_sanitizes_to_empty():
    # reviewer_comprehensive amendment: eligibility keys on the SANITIZED gold
    # set (the same file-level set set_scores runs on), not raw metadata.files.
    # A task whose ground truth is entirely directory-shaped sanitizes to an
    # empty gold set — admitting it would score set_scores([], []) = 1.0 for an
    # empty prediction but 0.0 for any non-empty one, an asymmetric per-task
    # bias in the comparative verdict — so it is dropped, aligned with scoring.
    tasks = [
        {'id': 1, 'title': 'real gold', 'description': 'd',
         'metadata': {'files': ['a.py']}},
        {'id': 2, 'title': 'dir-only gold', 'description': 'd',
         'metadata': {'files': ['orchestrator', 'scripts/']}},
    ]
    selected = mod.select_trial_sample(tasks, size=None)
    assert [t['id'] for t in selected] == [1]


def test_select_trial_sample_orders_most_recent_first():
    # Highest task id (newest) leads, regardless of input order.
    shuffled = [_T3, _T1, _T4, _T2]
    selected = mod.select_trial_sample(shuffled, size=None)
    assert [t['id'] for t in selected] == [4, 3, 2, 1]


def test_select_trial_sample_caps_to_size_taking_the_newest():
    # size=2 keeps the two most-recent eligible tasks, drops the older ones.
    selected = mod.select_trial_sample(_ALL_TASKS, size=2)
    assert [t['id'] for t in selected] == [4, 3]


def test_select_trial_sample_none_size_returns_all_eligible():
    selected = mod.select_trial_sample(_ALL_TASKS, size=None)
    assert len(selected) == len(_ELIGIBLE)


def test_select_trial_sample_handles_numeric_string_ids():
    # Ids arrive as numeric strings from the task backend; ordering is numeric,
    # not lexicographic (so id "10" sorts above "9", not below).
    tasks = [
        {'id': '9', 'title': 't9', 'description': 'd', 'metadata': {'files': ['a.py']}},
        {'id': '10', 'title': 't10', 'description': 'd', 'metadata': {'files': ['b.py']}},
    ]
    selected = mod.select_trial_sample(tasks, size=None)
    assert [t['id'] for t in selected] == ['10', '9']


def test_select_trial_sample_non_numeric_id_sorts_oldest_not_newest():
    # A malformed id fails safe to the oldest position — never spuriously most-recent.
    tasks = [
        {'id': 'bad', 'title': 'tb', 'description': 'd', 'metadata': {'files': ['a.py']}},
        {'id': 2, 'title': 't2', 'description': 'd', 'metadata': {'files': ['b.py']}},
    ]
    selected = mod.select_trial_sample(tasks, size=1)
    assert [t['id'] for t in selected] == [2]
