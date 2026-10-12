#!/usr/bin/env python3
"""What counts as a correct judge verdict, how a run's answers are scored, and how the report reads.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.
Pure: no file, network, store or LLM I/O. ``eval_write_triage_cases.py``,
``eval_write_triage_live_judge.py`` and ``eval_write_triage_judge.py`` import
it, and it imports none of them.
Alpha's label vocabulary is IMPORTED from ``calibrate_write_triage`` rather
than re-spelled (INV-5).

Where the ground truth comes from
---------------------------------
Entirely from alpha's labels, never from a fresh opinion about what the judge
ought to say. Each non-canonical record in the fixture becomes one judge call,
shown its cluster's canonical plus cross-cluster distractors, and the label
the curator assigned names the acceptable answers:

- ``duplicate`` — correct iff the judge ATTACHES, i.e. ``restated`` OR
  ``amended``. Alpha's labels do not separate a verbatim restatement from a
  rediscovery that carries a novel fragment; the curator recorded "same claim
  as the canonical" and stopped there. Scoring one of the two as wrong would
  invent a label nobody assigned and report a fabricated error rate as a
  measured one. The restated/amended split is reported as a DISTRIBUTION.
- ``distinct`` — correct iff ``stored``. Same cluster, same topic,
  curator-ruled not the same claim: any attach destroys a distinction a human
  drew.
- ``pseudo_contradiction`` — correct iff NOT ``contested``. These are
  adjudicated BOTH-CORRECT pairs (esc-5557/esc-5626): "the contradiction was
  an omission, not a disagreement".
- ``distractor`` — a CONTROL class this script adds, since the labels alone do
  not supply one. One case per cluster, whose slate carries no same-cluster
  record at all; correct iff ``stored``. Without it the eval cannot tell a
  judge that classifies from a judge that attaches to whatever it is shown,
  because every labelled case carries the correct target and "always attach"
  scores well on all of them.

THE THING THIS FIXTURE CANNOT MEASURE
-------------------------------------
**There is no positive ``contested`` ground truth anywhere in alpha's
corpus.** All six ``pseudo_contradiction`` records were adjudicated NOT
contradictions, and no record is labelled as a genuine one. So this eval
measures the judge's ``contested`` FALSE-POSITIVE rate and nothing else: there
is no contested recall or precision to compute here, and reporting one would
be a number with no measurement behind it. The report states this in its own
text, under ``contested_ground_truth`` and in the markdown caveats, so the
operator reading it at the 3169 gate is not misled into thinking the
contradiction detector was validated in both directions.
"""
from __future__ import annotations

import importlib.util
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_JUDGE,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

_CALIBRATE_PATH = Path(__file__).resolve().parent / 'calibrate_write_triage.py'


def _load_script(path: Path, mod_name: str) -> types.ModuleType:
    """Load a ``scripts/`` sibling by path, cached in ``sys.modules``."""
    import sys  # noqa: PLC0415

    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_calibrate = _load_script(_CALIBRATE_PATH, 'calibrate_write_triage')


# Alpha's vocabulary, re-exported rather than re-spelled.
LABEL_CANONICAL = _calibrate.LABEL_CANONICAL
LABEL_DUPLICATE = _calibrate.LABEL_DUPLICATE
LABEL_DISTINCT = _calibrate.LABEL_DISTINCT
LABEL_PSEUDO_CONTRADICTION = _calibrate.LABEL_PSEUDO_CONTRADICTION


#: The control class, which is this script's own construct rather than one of
#: alpha's labels — hence a separate constant, so nothing reads it back as a
#: curator adjudication.
CLASS_DISTRACTOR = 'distractor'

#: Every class the report accounts for, in report order. Held as a tuple so
#: ``per_class`` and ``confusion`` are built from ONE list: a class present in
#: one and absent from the other is the shape that makes "not measured" and
#: "measured perfect" indistinguishable.
EVAL_CLASSES: tuple[str, ...] = (
    LABEL_DUPLICATE,
    LABEL_DISTINCT,
    LABEL_PSEUDO_CONTRADICTION,
    CLASS_DISTRACTOR,
)

#: The confusion table's OTHER axis, in report order — the same "one list,
#: two consumers" shape as ``EVAL_CLASSES`` above, for the same reason.
#: DERIVED rather than hand-written, so a fifth triage outcome added to
#: ``write_triage`` joins this report automatically instead of becoming a
#: second list to keep in sync.
#:
#: ``TRIAGE_OUTCOMES`` is a frozenset, whose iteration order is
#: PYTHONHASHSEED-dependent, and this script's output is a COMMITTED artifact
#: read by an operator at the task-3169 flip gate: iterating it directly makes
#: two identical runs produce two differently-ordered reports, and makes the
#: committed markdown stop being provably the render of the committed JSON.
#: Measured 2026-08-27 — the committed pair disagreed on exactly this.
EVAL_OUTCOMES: tuple[str, ...] = tuple(sorted(TRIAGE_OUTCOMES))

#: Every provenance field the report is expected to carry, in report order —
#: the same "one list, two consumers" shape as ``EVAL_CLASSES`` above, and for
#: a sharper version of the same reason. :func:`build_report` backfills from
#: THIS tuple and ``_run`` supplies its own subset of it, so a field added in
#: one place cannot go missing from reports assembled through the other. Two
#: hand-typed lists is exactly how ``candidate_count_min`` and
#: ``distractor_count_requested`` — the pair that discloses a NARROWED slate —
#: came to be absent from every report ``build_report`` backfilled.
#:
#: An ABSENT key cannot be told apart from an artifact predating the field, so
#: an unmeasured one reads ``None`` rather than vanishing.
PROVENANCE_KEYS: tuple[str, ...] = (
    # Supplied by the caller — what was run, against what.
    'fixture_path',
    'judge_provider',
    'judge_model',
    # The system prompt wording the verdicts came from (PRD C2'').
    'judge_system_prompt_sha256',
    'limit',
    'canonical_aliases_path',
    'canonical_aliases_count',
    'cases_path',
    # How the slate reaching the judge was obtained, and how wide each field
    # of it was rendered. Both change what the model saw without changing any
    # other field in this block, so an artifact missing them is unreadable.
    'slate_mode',
    'field_chars',
    # Measured by `run_judge_eval` — the population and the slate it BUILT.
    'record_count',
    'case_count',
    'candidate_count',
    'candidate_count_min',
    'distractor_count',
    'distractor_count_requested',
    # Resolved from config — what the model could actually SEE, and whether
    # it was asked at all. `judge_write` re-trims the slate to
    # `judge_candidate_count`, and returns `stored` on its first line when
    # `judge_enabled` is false.
    'judge_candidate_count',
    'judge_enabled',
    # Measured by the RETRIEVED slate mode only, `None` under `seeded`. The
    # bands and the width are what routed each case; the three counts are the
    # corpus conditions a reader cannot recover from the accuracies.
    'project_id',
    't_high',
    't_low',
    'candidate_k',
    'canonical_absent',
    'degraded_retrievals',
    'self_retrieved',
)

#: The bands :func:`decide_band` routes to, in report order. NOT
#: :data:`EVAL_OUTCOMES`: ``judge`` is a routing decision rather than a triage
#: outcome and is absent from ``TRIAGE_OUTCOMES``, while ``amended`` and
#: ``contested`` are outcomes no band can reach on its own.
EVAL_BANDS: tuple[str, ...] = (OUTCOME_RESTATED, OUTCOME_JUDGE, OUTCOME_STORED)


#: The two ways a case's slate can be obtained. ``seeded`` is this script's
#: original construction; ``retrieved`` is what production would have found.
SLATE_SEEDED = 'seeded'
SLATE_RETRIEVED = 'retrieved'
SLATE_MODES: tuple[str, ...] = (SLATE_SEEDED, SLATE_RETRIEVED)


#: The outcomes under which ``triage_write`` attaches the write at all.
#: ``stored`` is the one that attaches to nothing
#: (``write_triage.py::_apply_judge_verdict`` nulls its ``canonical_id``).
ATTACH_OUTCOMES: frozenset[str] = frozenset({
    OUTCOME_RESTATED, OUTCOME_AMENDED, OUTCOME_CONTESTED,
})

#: Curator label -> the verdicts that count as correct for it. See the module
#: docstring for the rationale behind each entry; every one traces to a human
#: adjudication rather than to an opinion formed here.
#:
#: ``LABEL_CANONICAL`` is deliberately ABSENT rather than mapped to anything:
#: a canonical IS the attach target, so submitting one would score the
#: fixture's construction rather than the judge. :func:`build_judge_cases`
#: skips those records, and a label in neither this table nor
#: ``LABEL_CANONICAL`` RAISES.
ACCEPTABLE_OUTCOMES: dict[str, frozenset[str]] = {
    LABEL_DUPLICATE: frozenset({OUTCOME_RESTATED, OUTCOME_AMENDED}),
    LABEL_DISTINCT: frozenset({OUTCOME_STORED}),
    LABEL_PSEUDO_CONTRADICTION: frozenset(TRIAGE_OUTCOMES) - {OUTCOME_CONTESTED},
}

#: The control's expectation. Not in the table above because it is not a label.
DISTRACTOR_ACCEPTABLE = frozenset({OUTCOME_STORED})


class UnknownLabelError(ValueError):
    """A fixture record carries a label this eval has no expectation for.

    Raised rather than bucketed. A silently-bucketed fifth label would be
    scored against an expectation nobody ever set, producing an accuracy
    figure that looks measured and is not.
    """


def acceptable_outcomes_for(label: str) -> frozenset[str]:
    try:
        return ACCEPTABLE_OUTCOMES[label]
    except KeyError:
        raise UnknownLabelError(
            f'no acceptable-outcome expectation for label {label!r}; '
            f'known labels are '
            f'{sorted([*ACCEPTABLE_OUTCOMES, LABEL_CANONICAL])}. Add an entry '
            f'to ACCEPTABLE_OUTCOMES rather than bucketing it.',
        ) from None


# ---------------------------------------------------------------------------
# One case's answer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeAnswer:
    """How one case resolved, and what the model was shown to resolve it.

    ``outcome`` is what production would ACK; ``verdict`` is what the LLM said,
    and is ``None`` when no call was made — a deterministic band and a
    below-floor band are answered without one. The two are equal in seeded
    mode, where every case is synthesized into the middle band.

    The elision flags and ``usage`` are ``None`` for the same reason: nothing
    was rendered and nothing was spent.

    ``candidate_id`` is the slate id the verdict named, and ``None`` when no
    judge was asked or its verdict named nothing.
    """

    outcome: str
    verdict: str | None = None
    entry_elided: bool | None = None
    candidates_elided: Mapping[str, bool] | None = None
    usage: Mapping[str, int | None] | None = None
    candidate_id: str | None = None

    def __post_init__(self) -> None:
        if self.candidate_id is None:
            return
        if self.verdict is None or self.outcome not in ATTACH_OUTCOMES:
            raise ValueError(
                f'candidate_id {self.candidate_id!r} named with verdict '
                f'{self.verdict!r} and outcome {self.outcome!r}: only a judge '
                f'verdict that attaches (one of {sorted(ATTACH_OUTCOMES)}) names '
                f'a candidate',
            )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _is_acceptable(case: Mapping[str, Any], outcome: str) -> bool:
    """Did *outcome* satisfy the expectation *case* carries? The ONE reading.

    Read by :func:`score_cases` and by the per-case dump, so the middle-band
    accuracy computed from the dump cannot disagree with the headline one.
    """
    return outcome in case['acceptable_outcomes']


def score_cases(
    cases: Sequence[Mapping[str, Any]],
    verdicts: Sequence[str],
) -> dict[str, Any]:
    """Count *verdicts* against the expectations *cases* carry. Pure counting.

    Every judgment call was already made in :data:`ACCEPTABLE_OUTCOMES`;
    nothing here decides what "correct" means.

    *verdicts* is positional — ``verdicts[i]`` answers ``cases[i]``. A length
    mismatch RAISES rather than zipping to the shorter list, which would drop
    cases out of the denominator and report an accuracy over a population
    nobody chose.

    An ``expected_class`` outside :data:`EVAL_CLASSES` and a verdict outside
    :data:`EVAL_OUTCOMES` RAISE for the same reason, and the pre-seeded
    ``per_class``/``confusion`` dicts are therefore never widened here. An
    absorbed row inflates ``case_count`` in :func:`build_report` while
    :func:`render_markdown` iterates only ``EVAL_CLASSES``/``EVAL_OUTCOMES``,
    so it vanishes from the artifact entirely: the denominator moves and
    nothing in the report says so. ``UnknownLabelError`` is reused rather than
    re-invented so this boundary and :func:`acceptable_outcomes_for`'s
    case-construction boundary agree about what an unknown label is.

    Returned shape, all of it JSON-serializable (it is written to disk
    verbatim):

    - ``per_class`` — ``{n, correct, accuracy}`` for all four classes,
      ALWAYS present even at ``n == 0``. ``accuracy`` is ``None`` for an empty
      class, never ``0.0``: zero reads as "measured, and the judge failed
      everything", which is the opposite of "not measured".
    - ``confusion`` — the full 4xN map of expected class to observed outcome.
      A per-class accuracy cannot tell "every duplicate answered ``stored``"
      (a wiring bug) from "duplicates split across the two attaches" (a
      working judge); the shape of this map can.
    - ``duplicate_outcome_split`` — the restated/amended split within the
      duplicate class, reported and deliberately NOT charged as error.
    - ``false_contested`` — every ``contested`` verdict, all of which are
      false positives here because the corpus carries no positive contested
      ground truth. Zero is a finding, not an absence, so it is ``0`` rather
      than ``None``.
    """
    if len(cases) != len(verdicts):
        raise ValueError(
            f'{len(cases)} case(s) but {len(verdicts)} verdict(s): a positional '
            f'mismatch would silently drop cases from the denominator',
        )

    per_class = {name: {'n': 0, 'correct': 0} for name in EVAL_CLASSES}
    confusion = {
        name: dict.fromkeys(EVAL_OUTCOMES, 0) for name in EVAL_CLASSES
    }
    duplicate_split = {OUTCOME_RESTATED: 0, OUTCOME_AMENDED: 0}
    false_contested = 0

    for case, verdict in zip(cases, verdicts, strict=True):
        name = str(case['expected_class'])
        if name not in per_class:
            raise UnknownLabelError(
                f'no report class for expected_class {name!r}; known classes '
                f'are {sorted(EVAL_CLASSES)}. Add it to EVAL_CLASSES rather '
                f'than bucketing it.',
            )
        if verdict not in EVAL_OUTCOMES:
            raise ValueError(
                f'verdict {verdict!r} is outside the closed triage vocabulary '
                f'{sorted(EVAL_OUTCOMES)}: an absorbed verdict grows the '
                f'confusion row a column render_markdown never emits',
            )
        bucket = per_class[name]
        bucket['n'] += 1
        if _is_acceptable(case, verdict):
            bucket['correct'] += 1
        confusion[name][verdict] += 1
        if name == LABEL_DUPLICATE and verdict in duplicate_split:
            duplicate_split[verdict] += 1
        if verdict == OUTCOME_CONTESTED:
            false_contested += 1

    return {
        'per_class': {
            name: {
                'n': entry['n'],
                'correct': entry['correct'],
                'accuracy': (
                    None if entry['n'] == 0
                    else round(entry['correct'] / entry['n'], 4)
                ),
            }
            for name, entry in per_class.items()
        },
        'confusion': confusion,
        'duplicate_outcome_split': {
            'restated': duplicate_split[OUTCOME_RESTATED],
            'amended': duplicate_split[OUTCOME_AMENDED],
        },
        'false_contested': false_contested,
    }


def _rate(hits: int, total: int) -> float | None:
    """*hits*/*total*, or ``None`` when nothing was measured — never ``0.0``."""
    return round(hits / total, 4) if total else None


def score_attachments(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count WHERE each case attached, not merely whether its class was right.

    `score_cases` scores the verdict WORD against the curator label. This
    scores the attach target, so a case can answer `amended` — correct for its
    label — while attaching the write to an unrelated record.

    The target scored is each row's `attach_target_id`, the record production
    files the write against: for a middle-band attach, the candidate the judge
    named, hoisted to its canonical; otherwise the band winner. Rows from a
    judge that named no candidate (before task 5794) carry the band winner,
    which is where production attached them then.

    Pure counting over the per-case dump. Every rule is a field of a row;
    nothing is recomputed from the cases here.
    """
    attached = [r for r in rows if r['outcome'] in ATTACH_OUTCOMES]
    duplicates = [r for r in rows if r['expected_class'] == LABEL_DUPLICATE]
    negatives = [
        r for r in rows
        if r['expected_class'] in (
            LABEL_DISTINCT, LABEL_PSEUDO_CONTRADICTION, CLASS_DISTRACTOR,
        )
    ]
    judged = [r for r in rows if r['verdict'] is not None]
    middle = [r for r in rows if r['band'] == OUTCOME_JUDGE]
    wrong = [r for r in attached if not r['attach_target_is_canonical']]
    in_slate = [r for r in rows if r['canonical_in_slate']]
    absent = [r for r in rows if r['canonical_present'] is False]

    dup_strict = [
        r for r in duplicates
        if r['outcome'] in ATTACH_OUTCOMES and r['attach_target_is_canonical']
    ]
    dup_any = [r for r in duplicates if r['outcome'] in ATTACH_OUTCOMES]
    false_attach = [r for r in negatives if r['outcome'] in ATTACH_OUTCOMES]
    middle_correct = [r for r in middle if r['correct']]

    return {
        'attach_outcomes': sorted(ATTACH_OUTCOMES),
        'duplicate_attach': {
            'n': len(duplicates),
            'strict': len(dup_strict),
            'strict_rate': _rate(len(dup_strict), len(duplicates)),
            'any': len(dup_any),
            'any_rate': _rate(len(dup_any), len(duplicates)),
        },
        'wrong_record_attach': {
            'n': len(wrong),
            'attached': len(attached),
            'rate_of_attaches': _rate(len(wrong), len(attached)),
            'rate_of_cases': _rate(len(wrong), len(rows)),
            'cases': [
                {
                    'memory_id': r['memory_id'],
                    'expected_class': r['expected_class'],
                    'outcome': r['outcome'],
                    'attach_target_id': r['attach_target_id'],
                    'attach_target_cluster_id': r['attach_target_cluster_id'],
                    'attach_target_category': r['attach_target_category'],
                    'attach_target_label': r['attach_target_label'],
                }
                for r in wrong
            ],
        },
        'false_attach': {
            'n': len(false_attach),
            'denominator': len(negatives),
            'rate': _rate(len(false_attach), len(negatives)),
            'classes': [
                LABEL_DISTINCT, LABEL_PSEUDO_CONTRADICTION, CLASS_DISTRACTOR,
            ],
        },
        'band_split': {
            name: _band_counts([r for r in rows if r['expected_class'] == name])
            for name in EVAL_CLASSES
        },
        'band_split_all': _band_counts(rows),
        'middle_band': {
            'n': len(middle),
            'correct': len(middle_correct),
            'accuracy': _rate(len(middle_correct), len(middle)),
        },
        'canonical_in_slate': {
            'n': len(rows),
            'hits': len(in_slate),
            'rate': _rate(len(in_slate), len(rows)),
        },
        'canonical_absent': len(absent),
        'judge_calls': len(judged),
        'judge_spend': _spend(rows),
    }


def _band_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many of *rows* each band answered, in :data:`EVAL_BANDS` order.

    A band outside the vocabulary RAISES rather than widening the row, for the
    same reason :func:`score_cases` refuses an absorbed verdict: the markdown
    renders only the declared columns, so an absorbed band moves a denominator
    and vanishes from the artifact.
    """
    counts = dict.fromkeys(EVAL_BANDS, 0)
    for row in rows:
        band = str(row['band'])
        if band not in counts:
            raise ValueError(
                f'band {band!r} is outside {list(EVAL_BANDS)}: an absorbed band '
                f'grows a row that render_markdown never emits',
            )
        counts[band] += 1
    return counts


def _spend(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tokens billed across the run, or ``None`` where the provider reported none.

    Usage arrives on the shipped judge's verdict (``TriageJudgeVerdict.usage``),
    so every provider arm is priced; a dry run, an injected stub judge, or a
    call whose usage could not be read reports ``None`` rather than zero.
    """
    used = [row['usage'] for row in rows if row['usage']]
    if not used:
        return {'calls_with_usage': 0, 'prompt_tokens': None,
                'completion_tokens': None, 'total_tokens': None}
    return {
        'calls_with_usage': len(used),
        'prompt_tokens': sum(u.get('prompt_tokens') or 0 for u in used),
        'completion_tokens': sum(u.get('completion_tokens') or 0 for u in used),
        'total_tokens': sum(u.get('total_tokens') or 0 for u in used),
    }


# ---------------------------------------------------------------------------
# The per-case dump
# ---------------------------------------------------------------------------

def _attachable_id(record: Mapping[str, Any]) -> str:
    """The id an attach against *record* would name — its canonical, hoisted.

    Live rows carry the `_canonical_id_of` hoist the retrieval edge computed;
    a fixture record is its own attach target.
    """
    return str(record.get('canonical_id') or record['memory_id'])


def _judged_candidate_id(
    case: Mapping[str, Any],
    answer: JudgeAnswer,
    shown: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """The record the verdict named, hoisted as production attaches to it.

    *shown* is the case's own slate by memory id. A named id off it RAISES
    rather than being scored as an attach production never makes, or as the
    `stored` production would fail it open to. Neither shipped edge can reach
    it: each hands `judge_write` exactly these records, and
    `parse_judge_verdict` refuses an id off what it rendered from them. So
    reaching it means a `judge_fn` answered about records it was not shown, a
    defect every later judged attach would repeat; stopping at the first costs
    one call where failing open would buy a whole run of meaningless rows.
    """
    if answer.candidate_id is None:
        return None
    record = shown.get(answer.candidate_id)
    if record is None:
        raise ValueError(
            f'case {case["memory_id"]!r}: the verdict named '
            f'{answer.candidate_id!r}, which is not on its slate {sorted(shown)}',
        )
    return _attachable_id(record)


def case_row(
    index: int,
    case: Mapping[str, Any],
    answer: JudgeAnswer,
    candidate_records: Sequence[Mapping[str, Any]],
    fixture_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Everything about one case that the aggregate numbers cannot be reread from.

    A function of this case and the curator fixture alone: *candidate_records*
    are the rows ITS judge was shown. The attach target is described from its
    fixture record, so a null cluster or label means it is not one, and
    otherwise from its row on this slate.

    The attach target is where production files the write: the candidate the
    verdict named when it named one, otherwise ``band_winner_id``.
    ``verdict_candidate_id`` is the raw slate id the verdict named and
    ``judged_candidate_id`` is that id HOISTED (PRD
    ``plans/write-triage-flip-readiness-prd.md`` §11 C2''), unlike
    ``BandDecision.judged_candidate_id``, which is the raw id.

    JSON-serializable verbatim: this is the line appended to the cases file as
    each case completes, so a run interrupted partway keeps what it paid for.
    """
    shown = {str(record['memory_id']): record for record in candidate_records}
    judged = _judged_candidate_id(case, answer, shown)
    target_id = judged if judged is not None else case['band_winner_id']
    target = (
        (fixture_by_id.get(str(target_id)) or shown.get(str(target_id))) if target_id else None
    )
    cluster_id = case['cluster_id']
    alias_id = case['canonical_alias_id']
    canonical_ids = {str(cluster_id), *([str(alias_id)] if alias_id else [])}
    return {
        'index': index,
        'memory_id': case['memory_id'],
        'cluster_id': cluster_id,
        'canonical_alias_id': alias_id,
        'category': case.get('category'),
        'expected_class': case['expected_class'],
        'acceptable_outcomes': sorted(case['acceptable_outcomes']),
        'candidates': list(case['candidates']),
        'band_winner_id': case['band_winner_id'],
        'verdict_candidate_id': answer.candidate_id,
        'judged_candidate_id': judged,
        'attach_target_id': target_id,
        'attach_target_cluster_id': target.get('cluster_id') if target else None,
        'attach_target_category': target.get('category') if target else None,
        'attach_target_label': target.get('label') if target else None,
        'attach_target_is_canonical': str(target_id) in canonical_ids,
        'canonical_in_slate': any(
            _attachable_id(record) in canonical_ids for record in candidate_records
        ),
        'canonical_present': case['canonical_present'],
        'band': case['band'],
        'similarity': case['similarity'],
        'verdict': answer.verdict,
        'outcome': answer.outcome,
        'correct': _is_acceptable(case, answer.outcome),
        'entry_elided': answer.entry_elided,
        'candidates_elided': (
            dict(answer.candidates_elided)
            if answer.candidates_elided is not None else None
        ),
        'usage': dict(answer.usage) if answer.usage is not None else None,
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

#: Why no contested accuracy appears in the report, in alpha's
#: ``'<code>: <measured detail>'`` reason format so a consumer can branch on
#: the code without parsing prose.
CONTESTED_GROUND_TRUTH_REASON = (
    'no_positive_contested_labels: the fixture carries 6 pseudo_contradiction '
    'records, every one curator-adjudicated NOT a contradiction, and 0 records '
    'labelled as a genuine contradiction. Contested recall and precision are '
    'therefore unmeasurable against this corpus; only the false-positive count '
    'below is a measurement.'
)


#: Prose the operator at the task-3169 flip gate has to read BEFORE acting on
#: any number above it. Held as data rather than inlined into the renderer so
#: the JSON and the markdown cannot drift apart.
#:
#: These three hold whatever the slate came from. The two that do NOT are in
#: :data:`MODE_CAVEATS`: the control class and the population statement are
#: both properties of how the slate was obtained, and the seeded readings of
#: them are FALSE of a retrieved run.
CAVEATS: tuple[str, ...] = (
    'No accuracy floor is asserted anywhere in this script or its tests (PRD '
    'D10). This artifact is evidence for a human decision, not a gate.',
    'false_contested counts EVERY contested verdict, and every one of them is '
    'a false positive — see contested_ground_truth. A judge structurally '
    'incapable of ever answering `contested` would score identically to a '
    'perfect one here, so a low number is not evidence that the contradiction '
    'detector works.',
    'The duplicate class accepts BOTH `restated` and `amended`, because the '
    "curator's labels do not separate a verbatim restatement from a "
    'rediscovery carrying a novel fragment. The split between them is '
    'reported as a distribution and is not scored as error. NOT SCORED IS '
    'NOT THE SAME AS NOT CONSEQUENTIAL: `_TRIAGE_ATTACH_KINDS` in '
    '`server/tools.py` files a `restated` verdict as a SIGHTING, which '
    '`grouped_read` only counts, and an `amended` one as an AMENDMENT, whose '
    "text is digested into the canonical's grouped read. A swing between the "
    'two therefore changes what an operator reads while leaving every '
    'accuracy above unmoved, so read this split as a behaviour selector '
    'rather than as noise.',
)


#: The two caveats that depend on where the slate came from, keyed by mode.
#: An artifact whose provenance names neither reads as ``seeded``, which is
#: the only thing this script could do before the retrieved mode existed.
MODE_CAVEATS: dict[str, tuple[str, ...]] = {
    SLATE_SEEDED: (
        'The distractor class is a control this script constructs, not a curator '
        'label: one case per cluster whose slate carries no correct attach target '
        'at all. It is what distinguishes a judge that classifies from a judge '
        'that attaches to whatever it is shown.',
        'Every accuracy here is measured over the WHOLE labelled corpus, not over '
        'the [t_low, t_high) middle band the production judge is actually '
        'responsible for. `build_judge_cases` emits a case for every non-canonical '
        'record and `run_judge_eval` calls the judge on each one directly — '
        '`decide_band`, `t_high` and `t_low` never enter the picture, and the band '
        'decision handed to `judge_write` is SYNTHESIZED as a middle-band one. So '
        'these figures include records that in production are answered '
        'deterministically without the judge ever seeing them, and whether the '
        'middle band alone would score higher or lower is not measured here. '
        'Filtering the cases to the band would need real per-record similarities '
        'and is deliberately not done.',
    ),
    SLATE_RETRIEVED: (
        'There is NO distractor control class in this mode. A retrieved slate '
        'carrying no correct attach target is the ORDINARY case here rather '
        'than one this script constructs, so the control would measure nothing '
        'the population does not already show. '
        '`production_shape.canonical_in_slate` is the measured equivalent — '
        "the share of cases whose own canonical reached the prompt at all — and "
        '`canonical_absent` counts the cases whose canonical is no longer in '
        'the corpus. Those are KEPT in the population, because production '
        'meets them.',
        'Every figure here is measured over the population production would '
        'actually route: the slate, the band winner and the band all come '
        'from a live retrieval through `retrieve_candidates` / `decide_band` / '
        '`select_judge_candidates` at this config\'s `candidate_k`, `t_high` '
        'and `t_low`, and the judge was asked ONLY for the middle band. So '
        '`per_class` MIXES bands — a deterministic `restated` and a below-floor '
        '`stored` are counted there without an LLM having seen the case — and '
        '`production_shape.middle_band` is the judge\'s own accuracy. '
        'A verdict that is correct for its curator label can still attach to '
        'ANOTHER RECORD, which `per_class` cannot show. '
        '`production_shape.duplicate_attach.strict` and '
        '`production_shape.wrong_record_attach` score `attach_target_id`, the '
        'record production files the write against. For a middle-band attach '
        'that is the candidate the judge NAMED, hoisted to its canonical '
        '(`judged_candidate_id`); otherwise it is the band winner '
        '(`band_winner_id`). So for the middle band `strict` is recall at '
        '`judge_candidate_count`, not at 1. An artifact from a judge that named '
        'no candidate (before task 5794) attached every judged write to the '
        'band winner, and its rows carry no `judged_candidate_id`.',
    ),
}


def caveats_for(slate_mode: Any) -> list[str]:
    """The caveats a report assembled under *slate_mode* must carry."""
    return [*CAVEATS, *MODE_CAVEATS.get(str(slate_mode), MODE_CAVEATS[SLATE_SEEDED])]


def _judged_share(label: str, production_shape: Mapping[str, Any] | None) -> str:
    """How many *label* cases reached the judge, where the run measured a band split."""
    band_split = (production_shape or {}).get('band_split') or {}
    if label not in band_split:
        return ''
    return f' (of the {band_split[label][OUTCOME_JUDGE]} the judge saw)'


def pseudo_contradiction_caveat(
    scored: Mapping[str, Any], production_shape: Mapping[str, Any] | None,
) -> str:
    """How the pseudo_contradiction class read under the shipped `contests` definition.

    Derived from the scored confusion row, and from the band split where the
    run measured one, so the counts have one source. It reports the reading
    and asserts nothing about whether it is right: the records keep the labels
    they were adjudicated under and are still scored against them.
    """
    label = LABEL_PSEUDO_CONTRADICTION
    row = scored['confusion'][label]
    n = scored['per_class'][label]['n']
    breakdown = ', '.join(f'`{outcome}` {row[outcome]}' for outcome in EVAL_OUTCOMES)
    return (
        'Under the shipped `contests` definition — the entry says a candidate '
        "is wrong, outdated or different (Leo's ruling 2026-09-30, "
        "plans/write-triage-flip-readiness-prd.md §11.3 C1'') — the judge "
        f'answered `contested` on {row[OUTCOME_CONTESTED]} of {n} {label} '
        f'cases{_judged_share(label, production_shape)}; all outcomes: '
        f'{breakdown}. These records were '
        'adjudicated NOT contradictions under the EARLIER definition ("cannot '
        'be true at the same time"), and this report still scores `contested` '
        'on them as wrong and counts it in false_contested. The fixture is '
        'deliberately not relabelled, so this reports how the new definition '
        'reads them without asserting which reading is right.'
    )


def duplicate_contested_caveat(
    scored: Mapping[str, Any],
    production_shape: Mapping[str, Any] | None,
    judge_model: Any,
) -> str:
    """How often a duplicate read as `contested`, and on which judge.

    Derived from the scored confusion row like
    :func:`pseudo_contradiction_caveat`. Every one of these is a write the
    run's judge would flag for a human, so this is the false-alarm figure the
    flip operator reads, and it is only as general as the judge that
    produced it.
    """
    label = LABEL_DUPLICATE
    contested = scored['confusion'][label][OUTCOME_CONTESTED]
    n = scored['per_class'][label]['n']
    return (
        f'The judge, `{judge_model}`, answered `contested` on {contested} of '
        f'{n} {label} cases{_judged_share(label, production_shape)}. The '
        'curator labelled each duplicate the same claim as its canonical, so '
        'this report scores `contested` on one as wrong; with triage enabled, '
        'each is a write this judge would flag `contested` for a human to '
        'read. The `contests` definition was settled on a frontier reasoning '
        'judge: its live boundary test, '
        'fused-memory/tests/server/test_write_triage_judge.py::TestTheShippedWordingLive, '
        "pins one, because gpt-4o-mini answers that test's outdated cases "
        '`contested` under the earlier wording too. Where this judge is not '
        'the model that test pins, the figure measures the wording on a judge '
        'it was not settled on: read it beside a run on that model before '
        'deciding which judge the flip ships.'
    )


def build_report(
    *,
    scored: Mapping[str, Any],
    provenance: Mapping[str, Any],
    production_shape: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON-serializable accuracy report.

    Carries ``scored`` verbatim and adds the two things a bare score table
    cannot say for itself: what the fixture is incapable of measuring
    (``contested_ground_truth``) and how to read the numbers that are there
    (``caveats``).

    The provenance keys are ``setdefault``-ed rather than required, the same
    treatment ``run_calibration`` gives its own: a caller assembling a report
    from already-scored cases still produces an artifact whose provenance
    block has every key, with ``None`` where nothing was measured. An ABSENT
    key cannot be told apart from an artifact predating the field.

    The backfill iterates :data:`PROVENANCE_KEYS` rather than a literal, so
    that promise covers the WHOLE vocabulary. It used to name three of its
    members by hand, which left ``candidate_count_min`` and
    ``distractor_count_requested`` — the two fields that disclose a narrowed
    slate — absent from every report assembled here, so a report built from
    already-scored cases read exactly like a full-width run.
    """
    per_class = dict(scored['per_class'])
    run_provenance = dict(provenance)
    run_provenance.setdefault(
        'case_count', sum(entry['n'] for entry in per_class.values()),
    )
    for key in PROVENANCE_KEYS:
        run_provenance.setdefault(key, None)
    # Emitted in the VOCABULARY's order, not in whichever order the layers of
    # caller happened to set the keys. Two runs of the same command must not
    # produce two differently-ordered artifacts — the same property
    # `EVAL_OUTCOMES` is sorted for. Anything outside the vocabulary keeps its
    # own order, after.
    run_provenance = {
        **{key: run_provenance[key] for key in PROVENANCE_KEYS},
        **{k: v for k, v in run_provenance.items() if k not in PROVENANCE_KEYS},
    }

    return {
        'per_class': per_class,
        'confusion': dict(scored['confusion']),
        'duplicate_outcome_split': dict(scored['duplicate_outcome_split']),
        'false_contested': scored['false_contested'],
        # WHERE each case attached and which band answered it — the half of
        # the measurement `per_class` cannot carry, since a verdict word can
        # be right for its label while the attach lands on another record.
        # `None` means the run did not measure it, never that it measured zero.
        'production_shape': dict(production_shape) if production_shape else None,
        'contested_ground_truth': {
            'available': False,
            'reason': CONTESTED_GROUND_TRUTH_REASON,
        },
        'caveats': [
            *caveats_for(run_provenance.get('slate_mode')),
            pseudo_contradiction_caveat(scored, production_shape),
            duplicate_contested_caveat(
                scored, production_shape, run_provenance['judge_model'],
            ),
        ],
        'provenance': run_provenance,
    }


def _render_production_shape(report: Mapping[str, Any]) -> list[str]:
    """The attach/band half of the report.

    An EMPTY section means a run that assembled no attach accounting, and says
    so rather than reading as a section nobody wrote.
    """
    shape = report['production_shape']
    if not shape:
        return ['', '## Where the writes attached', '',
                '- not measured by this run.']
    dup = shape['duplicate_attach']
    wrong = shape['wrong_record_attach']
    false_attach = shape['false_attach']
    middle = shape['middle_band']
    in_slate = shape['canonical_in_slate']
    lines = [
        '', '## Where the writes attached, and which band answered', '',
        f'- duplicates attaching to their OWN canonical (strict): '
        f'{dup["strict"]}/{dup["n"]} = `{dup["strict_rate"]}`',
        f'- duplicates attaching to ANYTHING: {dup["any"]}/{dup["n"]} = '
        f'`{dup["any_rate"]}`',
        f'- attaches landing on a record that is NOT the case\'s canonical: '
        f'{wrong["n"]}/{wrong["attached"]} of all attaches = '
        f'`{wrong["rate_of_attaches"]}`',
        f'- attaches on a `distinct`/`pseudo_contradiction`/`distractor` case: '
        f'{false_attach["n"]}/{false_attach["denominator"]} = '
        f'`{false_attach["rate"]}`',
        f'- the case\'s canonical was ON the slate: {in_slate["hits"]}/'
        f'{in_slate["n"]} = `{in_slate["rate"]}`',
        f'- cases whose canonical is no longer in the corpus: '
        f'`{shape["canonical_absent"]}`',
        f'- judge accuracy restricted to the MIDDLE band: {middle["correct"]}/'
        f'{middle["n"]} = `{middle["accuracy"]}`',
        f'- LLM calls made: `{shape["judge_calls"]}`, tokens: '
        f'`{shape["judge_spend"]["total_tokens"]}`',
        '', '### Band split — expected class by the band that answered', '',
        '| class | ' + ' | '.join(EVAL_BANDS) + ' |',
        '|---' * (len(EVAL_BANDS) + 1) + '|',
    ]
    for name in EVAL_CLASSES:
        row = shape['band_split'][name]
        lines.append(
            f'| {name} | ' + ' | '.join(
                str(row.get(band, 0)) for band in EVAL_BANDS
            ) + ' |',
        )
    return lines


def render_markdown(report: Mapping[str, Any]) -> str:
    """Human-readable sibling. The operator reads THIS, not the JSON."""
    lines = [
        '# Write-triage judge accuracy report',
        '',
        '## Per-class accuracy',
        '',
        '| class | n | correct | accuracy |',
        '|---|---|---|---|',
    ]
    for name in EVAL_CLASSES:
        entry = report['per_class'][name]
        lines.append(
            f'| {name} | {entry["n"]} | {entry["correct"]} | {entry["accuracy"]} |',
        )

    outcomes = list(EVAL_OUTCOMES)
    lines += [
        '', '## Confusion — expected class by observed verdict', '',
        '| class | ' + ' | '.join(outcomes) + ' |',
        '|---' * (len(outcomes) + 1) + '|',
    ]
    for name in EVAL_CLASSES:
        row = report['confusion'][name]
        lines.append(
            f'| {name} | ' + ' | '.join(str(row.get(o, 0)) for o in outcomes) + ' |',
        )

    lines += _render_production_shape(report)

    split = report['duplicate_outcome_split']
    lines += [
        '', '## Duplicate attach split (a distribution, not an error term)', '',
        f'- `restated`: {split["restated"]}',
        f'- `amended`: {split["amended"]}',
        '',
        '## Contested',
        '',
        f'- contested verdicts observed: **{report["false_contested"]}**, all of '
        f'which are FALSE POSITIVES.',
        f'- ground truth available: `{report["contested_ground_truth"]["available"]}`',
        f'- `{report["contested_ground_truth"]["reason"]}`',
        '',
        '## Caveats',
        '',
    ]
    lines += [f'- {caveat}' for caveat in report['caveats']]
    lines += ['', '## Provenance', '']
    for key, value in report['provenance'].items():
        lines.append(f'- `{key}`: `{value}`')
    return '\n'.join(lines) + '\n'
