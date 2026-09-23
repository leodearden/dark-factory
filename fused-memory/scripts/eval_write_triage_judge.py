#!/usr/bin/env python3
"""Measure the write-triage JUDGE's accuracy against leaf alpha's curator labels.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf γ, decision D10.

Sibling of ``calibrate_write_triage.py``. That script measured what the
DETERMINISTIC bands can do — it derived ``t_high``/``t_low`` from observed
cosine distributions. This one measures what happens in the gap between them,
where the bands decline to answer and ``server/write_triage_judge.py`` is
asked instead.

The report this produces is the operator's input at the task-3169 flip gate.
It is a REPORT, not a gate: nothing in this script or its tests asserts an
accuracy floor, because D10 makes the human the decision-maker and a floor
asserted in code would silently become the decision.

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

Structure
---------
Mirrors the sibling script exactly: a pure synchronous core, unit-tested
against an injected ``judge_fn``, with the live LLM edge constructed only
inside the CLI. The whole test suite therefore runs with no
``OPENAI_API_KEY``, no network and no Qdrant.

Alpha's fixture loader, label vocabulary and ``package_relative`` are IMPORTED
from ``calibrate_write_triage`` rather than re-implemented (INV-5) — the
fixture format and the label words have one home.

Usage
-----
  # Prove the pipeline without spending anything. Its fixed answers are
  # redirected away from the committed artifact — see `guard_committed_report`.
  python scripts/eval_write_triage_judge.py --dry-run

  # A cheap live smoke over the first few cases.
  python scripts/eval_write_triage_judge.py --limit 5

  # The full measured pass, written to calibration/.
  python scripts/eval_write_triage_judge.py

  # What production would actually do: real retrieval, real bands, real
  # attach targets, with every case dumped as it completes.
  python scripts/eval_write_triage_judge.py --slate-mode retrieved \
      --report-path /tmp/retrieved.json --cases-path /tmp/retrieved-cases.jsonl
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import types
from collections.abc import Iterator, Mapping, Sequence
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

logger = logging.getLogger(__name__)

#: The fused-memory package root — the anchor for paths RECORDED in the report.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_CALIBRATE_PATH = _PACKAGE_ROOT / 'scripts' / 'calibrate_write_triage.py'

#: The COMMITTED artifact's path — the operator's stated input at the task-3169
#: flip gate. Named so `_run` can tell "writing the committed report" from
#: "writing somewhere else" and warn before a `--limit` smoke overwrites it.
_DEFAULT_REPORT_PATH = str(
    _PACKAGE_ROOT / 'calibration' / 'write_triage_judge_accuracy_report.json'
)


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


def _load_calibrate() -> types.ModuleType:
    """Load leaf alpha's script as a module.

    ``scripts/`` is not an importable package, so the sibling is reached the
    same way its own test suite reaches it. Importing it rather than copying
    its four label constants and its fixture loader is the point: a fixture
    format change or a fifth label lands in ONE place, and this script either
    follows it or fails loudly at ``build_judge_cases``.
    """
    return _load_script(_CALIBRATE_PATH, 'calibrate_write_triage')


_calibrate = _load_calibrate()

_RETRIEVAL_PATH = _PACKAGE_ROOT / 'scripts' / 'eval_write_triage_retrieval.py'


def load_retrieval() -> types.ModuleType:
    """Load the live-store edge, the same by-path way the calibrator loads.

    Deliberately NOT loaded at import: it pulls in the server's retrieval
    stack, and the whole pure core below is testable without it.
    """
    return _load_script(_RETRIEVAL_PATH, 'eval_write_triage_retrieval')

# Alpha's vocabulary and fixture handling, re-exported rather than re-spelled.
LABEL_CANONICAL = _calibrate.LABEL_CANONICAL
LABEL_DUPLICATE = _calibrate.LABEL_DUPLICATE
LABEL_DISTINCT = _calibrate.LABEL_DISTINCT
LABEL_PSEUDO_CONTRADICTION = _calibrate.LABEL_PSEUDO_CONTRADICTION
load_fixture = _calibrate.load_fixture
load_canonical_aliases = _calibrate.load_canonical_aliases
package_relative = _calibrate.package_relative

_NO_ALIASES: Mapping[str, str] = types.MappingProxyType({})

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

#: The outcomes under which ``triage_write`` files the write against
#: ``decision.canonical_id``. ``stored`` is the one that attaches to nothing
#: (write_triage.py: ``canonical_id = None if verdict == OUTCOME_STORED``).
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
_DISTRACTOR_ACCEPTABLE = frozenset({OUTCOME_STORED})


class UnknownLabelError(ValueError):
    """A fixture record carries a label this eval has no expectation for.

    Raised rather than bucketed. A silently-bucketed fifth label would be
    scored against an expectation nobody ever set, producing an accuracy
    figure that looks measured and is not.
    """


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def _acceptable_for(label: str) -> frozenset[str]:
    try:
        return ACCEPTABLE_OUTCOMES[label]
    except KeyError:
        raise UnknownLabelError(
            f'no acceptable-outcome expectation for label {label!r}; '
            f'known labels are '
            f'{sorted([*ACCEPTABLE_OUTCOMES, LABEL_CANONICAL])}. Add an entry '
            f'to ACCEPTABLE_OUTCOMES rather than bucketing it.',
        ) from None


def _distractor_pool(records: Sequence[Mapping[str, Any]], cluster_id: Any) -> list[str]:
    """Every record outside *cluster_id*, ordered by memory_id.

    Sorted rather than shuffled: this script's output is a COMMITTED artifact,
    and a seeded shuffle would make it reproducible only by someone who also
    knew the seed, an unseeded one by nobody at all.
    """
    return sorted(
        str(r['memory_id']) for r in records if r['cluster_id'] != cluster_id
    )


def _rotated(pool: list[str], offset: int, count: int) -> list[str]:
    """*count* entries from *pool*, starting at *offset*, wrapping around.

    The rotation is what stops all N cases being shown the same slate. Taking
    a plain ``pool[:count]`` would be equally deterministic and would measure
    one arbitrary handful of clusters over and over instead of the corpus.
    """
    if count <= 0:
        return []
    if not pool or len(pool) < count:
        # LOUD rather than silent: a short pool narrows every slate, and the
        # report's own width provenance is measured downstream rather than
        # asserted, so the two together are what stop a 1-wide smoke run being
        # read as the 5-wide measurement the operator is deciding on.
        logger.warning(
            'distractor pool holds %d record(s), %d requested — slate narrowed. '
            'Cross-cluster distractors come only from OTHER clusters, so a '
            'single-cluster run (typically --limit slicing one cluster) leaves '
            'nothing to draw from.',
            len(pool), count,
        )
    take = min(count, len(pool))
    if take <= 0:
        return []
    start = offset % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(take)]


def _case(
    record: Mapping[str, Any],
    *,
    candidates: list[str],
    expected_class: str,
    acceptable: frozenset[str],
    attach_target_id: str | None,
    band: str,
    similarity: float | None,
    canonical_present: bool | None,
    canonical_alias_id: str | None,
) -> dict[str, Any]:
    """One judge case: what is submitted, what it is shown, how it routed.

    Every routing field is explicit because the two slate modes disagree about
    all of them, and a defaulted one would let a retrieved case silently
    publish the seeded construction's answer.
    """
    return {
        'memory_id': str(record['memory_id']),
        'content': record['content'],
        'category': record.get('category'),
        'cluster_id': str(record['cluster_id']),
        'candidates': candidates,
        'expected_class': expected_class,
        'acceptable_outcomes': acceptable,
        'attach_target_id': attach_target_id,
        'band': band,
        'similarity': similarity,
        'canonical_present': canonical_present,
        'canonical_alias_id': canonical_alias_id,
    }


def _seeded_case(
    record: Mapping[str, Any],
    slate: list[str],
    expected_class: str,
    acceptable: frozenset[str],
    canonical_alias_id: str | None,
) -> dict[str, Any]:
    """A case whose slate is CONSTRUCTED rather than retrieved.

    The lead record is the attach target and the band is the middle one — the
    only band that reaches a judge — because that is what `build_judge_fn`
    synthesizes. Similarity and canonical liveness are unmeasured here, and
    `None` says so rather than claiming a figure.
    """
    return _case(
        record,
        candidates=slate,
        expected_class=expected_class,
        acceptable=acceptable,
        attach_target_id=slate[0] if slate else None,
        band=OUTCOME_JUDGE,
        similarity=None,
        canonical_present=None,
        canonical_alias_id=canonical_alias_id,
    )


def build_judge_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    distractors: int,
    aliases: Mapping[str, str] = _NO_ALIASES,
) -> list[dict[str, Any]]:
    """One judge call per case, with the ground truth its label implies.

    Two kinds of case come out of here.

    LABELLED — one per NON-canonical record. Its slate is the record's own
    cluster canonical plus *distractors* cross-cluster records, so the attach
    target is always present: a judge shown a slate without it is answering
    about a different memory than the one an attach would touch. Canonical
    records produce no labelled case, because they ARE the target.

    CONTROL (``CLASS_DISTRACTOR``) — one per CLUSTER, capped there on cost,
    since every case is a paid LLM call and one control per cluster already
    answers the question the control asks. Its slate is ``distractors + 1``
    records drawn only from OTHER clusters, so it is exactly as wide as a
    labelled slate and nothing on it is a correct attach.

    Ordering is deterministic and independent of the input order: records are
    sorted by ``memory_id`` before anything is drawn, so the committed report
    is reproducible from the fixture alone.

    An unrecognised label raises :class:`UnknownLabelError`.
    """
    ordered = sorted(records, key=lambda r: str(r['memory_id']))
    cases: list[dict[str, Any]] = []

    for index, record in enumerate(ordered):
        label = str(record['label'])
        if label == LABEL_CANONICAL:
            continue
        acceptable = _acceptable_for(label)
        canonical_id = str(record['cluster_id'])
        pool = _distractor_pool(ordered, record['cluster_id'])
        slate = [canonical_id, *_rotated(pool, index, distractors)]
        cases.append(_seeded_case(
            record, slate, label, acceptable, aliases.get(canonical_id),
        ))

    # The control set, appended after the labelled ones so a `--limit` smoke
    # run covers the classes the labels actually measure first.
    seen_clusters: set[Any] = set()
    for index, record in enumerate(ordered):
        if str(record['label']) == LABEL_CANONICAL:
            continue
        if record['cluster_id'] in seen_clusters:
            continue
        seen_clusters.add(record['cluster_id'])
        pool = _distractor_pool(ordered, record['cluster_id'])
        slate = _rotated(pool, index, distractors + 1)
        cases.append(_seeded_case(
            record, slate, CLASS_DISTRACTOR, _DISTRACTOR_ACCEPTABLE,
            aliases.get(str(record['cluster_id'])),
        ))

    return cases


# ---------------------------------------------------------------------------
# The plan: what will be judged, and what its slates name
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalPlan:
    """The cases of one run, the records their slates name, and how they were built.

    ``records_by_id`` must resolve every id in every case's ``candidates``: a
    judge handed a bare id has no text to compare and is answering noise, and
    an unresolvable id narrows a slate the report claims was wider.

    ``provenance`` is the plan's OWN disclosure — the fields that describe how
    the slates were obtained. It is merged UNDER the caller's, so nothing here
    can overwrite what the operator asked for.

    A case routed to the judge must show it at least one candidate, and that
    is checked at construction, before anything is spent or written.
    """

    cases: tuple[dict[str, Any], ...]
    records_by_id: Mapping[str, Mapping[str, Any]]
    record_count: int
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        unseen = [
            f"{case['memory_id']} ({case['expected_class']})"
            for case in self.cases
            if case['band'] == OUTCOME_JUDGE and not case['candidates']
        ]
        if unseen:
            raise ValueError(
                f'{len(unseen)} judge-band case(s) have an EMPTY slate: '
                f'{", ".join(unseen)}. judge_write answers those `stored` with no '
                'provider call, so the eval would count a verdict nobody gave. A '
                'single-cluster run leaves the distractor control nothing to draw '
                'from: widen --limit or the fixture.',
            )


def seeded_plan(
    records: Sequence[Mapping[str, Any]],
    *,
    distractors: int,
    aliases: Mapping[str, str] = _NO_ALIASES,
) -> EvalPlan:
    """Today's construction: the cluster canonical seeded at slate position 0."""
    return EvalPlan(
        cases=tuple(build_judge_cases(records, distractors=distractors, aliases=aliases)),
        records_by_id={str(r['memory_id']): r for r in records},
        record_count=len(records),
        provenance={
            'slate_mode': SLATE_SEEDED,
            'distractor_count_requested': distractors,
        },
    )


def plan_from_slates(
    records: Sequence[Mapping[str, Any]],
    slates: Sequence[Any],
    *,
    provenance: Mapping[str, Any],
    aliases: Mapping[str, str] = _NO_ALIASES,
) -> EvalPlan:
    """Pair production-shaped slates with the curator labels they answer for.

    One case per NON-canonical record, positional against *slates*. No control
    class is constructed: a retrieved slate that carries no correct attach
    target is the ordinary case here rather than something to build.

    The expected class stays the fixture's label. The slate, the attach target
    and the band come from the retrieval — including for a record whose
    canonical is no longer in the corpus, which is kept in the population and
    flagged rather than dropped.
    """
    labelled = [r for r in records if str(r['label']) != LABEL_CANONICAL]
    if len(labelled) != len(slates):
        raise ValueError(
            f'{len(labelled)} labelled record(s) but {len(slates)} slate(s): a '
            f'positional mismatch would score a case against another record\'s '
            f'retrieval',
        )
    cases: list[dict[str, Any]] = []
    records_by_id: dict[str, Mapping[str, Any]] = {}
    for record, slate in zip(labelled, slates, strict=True):
        for candidate in slate.candidates:
            records_by_id[str(candidate['memory_id'])] = candidate
        cases.append(_case(
            record,
            candidates=[str(c['memory_id']) for c in slate.candidates],
            expected_class=str(record['label']),
            acceptable=_acceptable_for(str(record['label'])),
            attach_target_id=slate.attach_target_id,
            band=slate.band,
            similarity=slate.similarity,
            canonical_present=slate.canonical_present,
            canonical_alias_id=aliases.get(str(record['cluster_id'])),
        ))
    return EvalPlan(
        cases=tuple(cases),
        records_by_id=records_by_id,
        record_count=len(records),
        provenance={'slate_mode': SLATE_RETRIEVED, **dict(provenance)},
    )


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
    """

    outcome: str
    verdict: str | None = None
    entry_elided: bool | None = None
    candidates_elided: Mapping[str, bool] | None = None
    usage: Mapping[str, int] | None = None


def _as_answer(value: Any) -> JudgeAnswer:
    """Accept a bare verdict from a ``judge_fn`` that has nothing else to say."""
    if isinstance(value, JudgeAnswer):
        return value
    return JudgeAnswer(outcome=str(value), verdict=str(value))


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
    re-invented so this boundary and :func:`_acceptable_for`'s
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
    scores the production consequence: `triage_write` files every non-`stored`
    outcome against `decision.canonical_id`, so a case can answer `amended` —
    correct for its label — while attaching the write to an unrelated record.

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
    """Tokens billed across the run, or ``None`` where the client reported none.

    ``judge_write`` discards the provider response's ``usage``, so this is
    populated only when the run installed a recording client; an ``anthropic``
    run and an injected stub judge both report ``None`` rather than zero.
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


def case_row(
    index: int,
    case: Mapping[str, Any],
    answer: JudgeAnswer,
    records_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Everything about one case that the aggregate numbers cannot be reread from.

    JSON-serializable verbatim: this is the line appended to the cases file as
    each case completes, so a run interrupted partway keeps what it paid for.
    """
    target_id = case['attach_target_id']
    target = records_by_id.get(str(target_id)) if target_id else None
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
        'attach_target_id': target_id,
        'attach_target_cluster_id': target.get('cluster_id') if target else None,
        'attach_target_category': target.get('category') if target else None,
        'attach_target_label': target.get('label') if target else None,
        'attach_target_is_canonical': str(target_id) in canonical_ids,
        'canonical_in_slate': any(
            _attachable_id(records_by_id[cid]) in canonical_ids
            for cid in case['candidates']
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


@contextlib.contextmanager
def _case_sink(cases_path: str | Path | None) -> Iterator[Any]:
    """A callable that appends one JSON line per case, flushed as it goes.

    Flushed per line and not per run: a provider 429 partway through is
    exactly when the rows already paid for matter most. ``None`` yields a
    sink that discards, so the caller has no branch.
    """
    if cases_path is None:
        yield lambda row: None
        return
    path = Path(cases_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as handle:
        def emit(row: Mapping[str, Any]) -> None:
            handle.write(json.dumps(row) + '\n')
            handle.flush()

        yield emit
    logger.info('Wrote %s', path)


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
        'actually route: the slate, the attach target and the band all come '
        'from a live retrieval through `retrieve_candidates` / `decide_band` / '
        '`select_judge_candidates` at this config\'s `candidate_k`, `t_high` '
        'and `t_low`, and the judge was asked ONLY for the middle band. So '
        '`per_class` MIXES bands — a deterministic `restated` and a below-floor '
        '`stored` are counted there without an LLM having seen the case — and '
        '`production_shape.middle_band` is the judge\'s own accuracy. '
        'A verdict that is correct for its curator label can still attach to '
        'ANOTHER RECORD: `triage_write` files every non-`stored` outcome '
        'against `decision.canonical_id`, the band\'s argmax, not against '
        'whatever the judge reasoned about. '
        '`production_shape.duplicate_attach.strict` is the figure that '
        'accounts for that and `per_class` is not.',
    ),
}


def caveats_for(slate_mode: Any) -> list[str]:
    """The caveats a report assembled under *slate_mode* must carry."""
    return [*CAVEATS, *MODE_CAVEATS.get(str(slate_mode), MODE_CAVEATS[SLATE_SEEDED])]


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
        'caveats': caveats_for(run_provenance.get('slate_mode')),
        'provenance': run_provenance,
    }


def _render_production_shape(report: Mapping[str, Any]) -> list[str]:
    """The attach/band half of the report, or nothing on an artifact predating it.

    An ABSENT key means the report was assembled before this section existed,
    so the render stays silent and remains the provable render of that JSON. A
    key PRESENT and empty means a run that assembled no attach accounting, and
    says so rather than reading as a section nobody wrote.
    """
    if 'production_shape' not in report:
        return []
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


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def markdown_sibling(report_path: str | Path) -> Path:
    """Where the markdown for *report_path* goes, or ``ValueError`` if nowhere.

    COMPOSED from the stem, not derived by replacing the last suffix.
    ``with_suffix('.md')`` maps ``foo.md`` back to ``foo.md``, so the markdown
    overwrote the JSON that had just been written — every number the run paid
    for, gone, with no error, on a script whose output is a committed
    artifact.

    A FUNCTION OF THE ARGUMENT ALONE, which is why it is one: nothing about
    the collision depends on what the run measures, so it is knowable before
    any work is done. It used to be evaluated at the bottom of
    :func:`run_judge_eval`, after ``build_judge_cases``, after the whole
    per-case ``judge_fn`` loop and after ``score_cases`` — which protected
    the two FILES and nothing else. On a live run ``--report-path foo.md``
    spent every LLM call for the corpus and then raised with no artifact
    written at all, so the operator paid for the run and got nothing.
    Callers evaluate it up front instead: :func:`run_judge_eval` at its first
    statement, and :func:`_run` before it so much as reads the fixture.

    ``guard_committed_report`` does not cover this: that guard addresses
    dry-run/``--limit`` publishing and returns early for any non-committed
    path, so ``--report-path foo.md`` sailed straight through it.
    """
    report_path = Path(report_path)
    sibling = report_path.parent / (report_path.stem + '.md')
    if sibling == report_path:
        raise ValueError(
            f'report_path {str(report_path)!r} composes the same path as its '
            f'markdown sibling {str(sibling)!r}, so the markdown would '
            f'overwrite the JSON report — pass a report_path whose stem+".md" '
            f'differs from it (e.g. a .json suffix)',
        )
    return sibling


def run_judge_eval(
    *,
    plan: EvalPlan,
    judge_fn: Any,
    report_path: str | Path,
    provenance: Mapping[str, Any],
    cases_path: str | Path | None = None,
) -> dict[str, Any]:
    """Ask *judge_fn* once per case of *plan*, score, dump, write both artifacts.

    ``judge_fn(case, candidates)`` receives the case and its slate RESOLVED to
    records — a judge handed bare ids has no text to compare and is answering
    noise, which is the same defect the production seam's own tests pin. It
    returns a verdict word, or a :class:`JudgeAnswer` when it has more to
    report: whether an LLM was called at all, what was elided, what was spent.

    Every case is dumped to *cases_path* AS IT COMPLETES, before the next one
    is asked. A provider failure mid-run therefore costs the remaining cases,
    not the ones already bought.

    Nothing is caught. A judge error propagates exactly as ``run_calibration``
    lets an ``embed_fn`` error propagate: swallowing it would be
    indistinguishable from a genuine misclassification, so it would both
    shrink the measured population and depress the accuracy reported over it.

    A dangling candidate id raises ``KeyError`` for the same reason — a
    silently-skipped candidate narrows a slate the report claims was 5 wide.

    THE MARKDOWN SIBLING NEVER OVERWRITES THE REPORT. :func:`markdown_sibling`
    composes it and RAISES on a *report_path* that composes back to itself —
    resolved as this function's FIRST statement, before a single case is
    built or a single verdict is bought, so the mistake costs nothing rather
    than merely leaving the two files intact after a paid run.
    """
    report_path = Path(report_path)
    markdown_path = markdown_sibling(report_path)

    cases = list(plan.cases)
    logger.info(
        'Judging %d case(s) from %d record(s)', len(cases), plan.record_count,
    )

    rows: list[dict[str, Any]] = []
    with _case_sink(cases_path) as emit:
        for index, case in enumerate(cases, 1):
            candidates = [plan.records_by_id[cid] for cid in case['candidates']]
            answer = _as_answer(judge_fn(case, candidates))
            row = case_row(index, case, answer, plan.records_by_id)
            emit(row)
            rows.append(row)
            if index % 10 == 0:
                logger.info('Judged %d/%d case(s)', index, len(cases))

    scored = score_cases(cases, [row['outcome'] for row in rows])

    run_provenance = {**dict(plan.provenance), **dict(provenance)}
    run_provenance.setdefault('record_count', plan.record_count)
    run_provenance.setdefault('case_count', len(cases))
    _record_slate_widths(run_provenance, cases)
    report = build_report(
        scored=scored,
        provenance=run_provenance,
        production_shape=score_attachments(rows),
    )

    # `markdown_path` was composed (and its collision with `report_path`
    # rejected) at the top of this function — reused here rather than
    # recomposed, so the path that was validated is the path that is written.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    markdown_path.write_text(render_markdown(report))
    logger.info('Wrote %s and %s', report_path, markdown_path)

    return report


def _record_slate_widths(
    run_provenance: dict[str, Any], cases: Sequence[Mapping[str, Any]],
) -> None:
    """Publish the width the run MEASURED, and warn when it is not the ask.

    `_rotated` truncates when the pool is short, so `distractors + 1` is what
    was REQUESTED and can silently exceed what the slates actually carried. A
    run whose slates all came out 1 wide must not publish `candidate_count: 5`.
    """
    widths = [len(case['candidates']) for case in cases]
    run_provenance.setdefault('candidate_count', max(widths) if widths else 0)
    run_provenance.setdefault('candidate_count_min', min(widths) if widths else 0)
    run_provenance.setdefault(
        'distractor_count', (max(widths) - 1) if widths else 0,
    )
    requested = run_provenance.get('distractor_count_requested')
    if widths and isinstance(requested, int) and min(widths) != requested + 1:
        logger.warning(
            'slate widths ran %d..%d against a requested %d — the report '
            'records what was measured, not what was asked for',
            min(widths), max(widths), requested + 1,
        )
    # The OTHER direction, and a different mechanism: `judge_write` re-trims
    # the slate to `judge_candidate_count` before the prompt is built, so a
    # slate built wider than that cap is measured narrower than it is
    # published.
    cap = run_provenance.get('judge_candidate_count')
    if widths and isinstance(cap, int) and max(widths) > cap:
        logger.warning(
            'slate widths ran %d..%d but judge_candidate_count caps the '
            'prompt at %d — the report records what the model was measured '
            'on, not the wider slate that was built',
            min(widths), max(widths), cap,
        )


# ---------------------------------------------------------------------------
# Live edge / CLI
# ---------------------------------------------------------------------------

#: Synthetic per-store cosines stamped on the eval's candidates. The eval
#: SUPPLIES its slate rather than retrieving it, but ``select_judge_candidates``
#: orders by ``metadata['store_score']`` and DROPS anything with no numeric
#: one, so a slate carrying no scores would arrive at the model empty.
#:
#: Descending by slate position, which preserves the order
#: ``build_judge_cases`` chose — the attach target first. These numbers never
#: reach the model: ``build_judge_prompt`` renders id and text only, no
#: metadata at all (PRD C1). They exist solely to survive the selector.
_SYNTHETIC_TOP_SCORE = 0.90
_SYNTHETIC_SCORE_STEP = 0.01

#: Passed to ``judge_write`` to satisfy its signature. It reaches no prompt —
#: C1 forbids repo or task context from reaching the judge, and
#: ``build_judge_prompt`` interpolates no metadata whatsoever.
_EVAL_PROJECT_ID = 'dark_factory'

#: The answer a ``--dry-run`` gives every case. Chosen as `stored` because it
#: is the fail-open verdict: a dry run should look like the judge declining to
#: attach, never like a confident classification.
_DRY_RUN_VERDICT = OUTCOME_STORED

#: Where a ``--dry-run`` writes when it was aimed at the committed artifact.
#: A fixed name rather than ``mkdtemp``: a dry run is a smoke, and telling the
#: operator exactly where its throwaway landed beats a fresh random directory
#: every invocation.
_DRY_RUN_REPORT_NAME = 'write_triage_judge_accuracy_report.dry-run.json'


def _is_committed_report(report_path: str) -> bool:
    """Is ``report_path`` the committed artifact, however it was spelled?

    ``resolve()`` on BOTH sides deliberately: a string ``==`` would call
    ``calibration/write_triage_judge_accuracy_report.json`` (relative, as an
    operator would naturally type it from the package root) a different file
    from the absolute default, and wave through the exact overwrite this
    guard exists to catch.
    """
    try:
        return Path(report_path).resolve() == Path(_DEFAULT_REPORT_PATH).resolve()
    except OSError:  # pragma: no cover - resolve() on a pathological path
        return False


def guard_committed_report(
    report_path: str, *, dry_run: bool, limit: int | None,
) -> str:
    """Keep a non-measurement run from publishing itself as the measurement.

    The committed report is what ``write_triage.judge_accuracy_report_path``
    points at and, per D10, what the operator reads at the task-3169 flip
    gate. Two kinds of run can reach it without having measured what it
    claims, and they are NOT the same hazard, so they get different answers:

    - ``--dry-run`` measures NOTHING. Every number in its report comes from a
      fixed-answer stub, so publishing it would put fabricated figures under
      the operator's nose with no artifact-level tell beyond a
      ``judge_provider`` field nobody is obliged to read. It is REDIRECTED to
      a temp path — the documented "prove the pipeline" invocation keeps
      working and still prints its report, it just cannot overwrite the
      committed one.
    - ``--limit N`` measures REALLY, just partially. Its numbers are the
      judge's own, and ``provenance.limit`` records the truncation in the
      artifact itself, so an operator (and
      ``TestCommittedJudgeAccuracyReportIsTraceable``) can tell. That is a
      loud WARNING and a proceed, not a redirect.

    Returns the path to actually write to.
    """
    if not _is_committed_report(report_path):
        return report_path

    if dry_run:
        import tempfile  # noqa: PLC0415

        redirected = str(Path(tempfile.gettempdir()) / _DRY_RUN_REPORT_NAME)
        logger.warning(
            'a --dry-run scores a FIXED-ANSWER stub, not the judge: it has no '
            'measurement to publish. Redirecting its report away from the '
            'committed artifact at %s and writing %s instead. Pass '
            '--report-path to choose somewhere else.',
            report_path, redirected,
        )
        return redirected

    if limit is not None:
        logger.warning(
            'a --limit run is about to OVERWRITE the committed report at '
            '%s. It is a partial smoke, not the corpus-wide measurement '
            'the task-3169 flip gate reads: provenance.limit=%d records '
            'that. Pass --report-path to keep the committed artifact.',
            report_path, limit,
        )

    return report_path


@contextlib.contextmanager
def field_chars_override(field_chars: int | None) -> Iterator[int]:
    """Run with the judge's per-field prompt budget set to *field_chars*.

    A MODULE-LEVEL override, restored on exit, because the seam admits nothing
    finer: ``_FIELD_CHARS`` is read inside ``_elide``, which ``build_judge_prompt``
    calls with no width parameter, which ``judge_write`` in turn calls with
    none either. Threading one would mean widening three production signatures
    for a knob only this eval turns.

    ``0`` means NO elision and is spelled as an effective cap of ``sys.maxsize``
    rather than 0, which would elide every field to nothing. ``None`` overrides
    nothing and yields the shipped value, so the caller records what was in
    force either way.
    """
    import sys  # noqa: PLC0415

    from fused_memory.server import write_triage_judge  # noqa: PLC0415

    shipped = write_triage_judge._FIELD_CHARS
    if field_chars is None:
        yield shipped
        return
    write_triage_judge._FIELD_CHARS = sys.maxsize if field_chars == 0 else field_chars
    try:
        yield field_chars
    finally:
        write_triage_judge._FIELD_CHARS = shipped


@contextlib.contextmanager
def usage_recording_openai() -> Iterator[list[Any]]:
    """Collect every openai completion's ``usage`` for the duration of the block.

    ``_call_llm`` discards the response's usage, and the operator deciding the
    3169 flip is deciding about a per-write cost. The client is wrapped rather
    than the production call changed: the wrapper adds one append and returns
    the provider's own response untouched.

    Yields the sink. It stays EMPTY on the anthropic arm and for an injected
    stub judge, which is why the spend block reports ``None`` rather than 0.
    """
    import openai  # noqa: PLC0415

    recorded: list[Any] = []
    shipped = openai.AsyncOpenAI

    def factory(**kwargs: Any) -> Any:
        client = shipped(**kwargs)
        create = client.chat.completions.create

        async def recording_create(**call: Any) -> Any:
            response = await create(**call)
            recorded.append(response.usage)
            return response

        client.chat.completions.create = recording_create
        return client

    openai.AsyncOpenAI = factory
    try:
        yield recorded
    finally:
        openai.AsyncOpenAI = shipped


def _elision_flags(
    content: str, slate: Sequence[Any],
) -> tuple[bool, dict[str, bool]]:
    """Which rendered fields the judge's own ``_elide`` cut, at the width in force.

    Asked of the shipped function rather than re-derived from a length
    comparison here, so the flags cannot disagree with the prompt.
    """
    from fused_memory.server.write_triage_judge import _elide  # noqa: PLC0415

    return (
        _elide(content) != content,
        {c.id: _elide(c.content) != c.content for c in slate},
    )


def _usage_of(recorded: Sequence[Any]) -> dict[str, int] | None:
    """The last recorded completion's token counts, or ``None`` if none was."""
    if not recorded:
        return None
    usage = recorded[-1]
    return {
        field: int(getattr(usage, field, 0) or 0)
        for field in ('prompt_tokens', 'completion_tokens', 'total_tokens')
    }


def _ask_judge(
    service: Any,
    content: str,
    slate: Sequence[Any],
    *,
    outcome: str,
    attach_target_id: str | None,
    similarity: float | None,
    recorded: Sequence[Any],
) -> JudgeAnswer:
    """Drive the SHIPPED judge for one case, or restate a band that decided itself.

    Only the middle band reaches an LLM, exactly as ``triage_write`` routes it:
    a deterministic ``restated`` and a below-floor ``stored`` are returned with
    no call, no rendered prompt and no spend, so none of the three is reported.

    ``asyncio.run`` per case, which is why the runner is synchronous:
    ``judge_write`` is the only awaitable in the pipeline and it needs no
    initialized ``MemoryService``.
    """
    import asyncio  # noqa: PLC0415

    from fused_memory.server.write_triage import BandDecision  # noqa: PLC0415
    from fused_memory.server.write_triage_judge import judge_write  # noqa: PLC0415

    if outcome != OUTCOME_JUDGE:
        return JudgeAnswer(outcome=outcome)

    entry_elided, candidates_elided = _elision_flags(content, slate)
    before = len(recorded)
    verdict = asyncio.run(judge_write(
        memory_service=service,
        content=content,
        project_id=_EVAL_PROJECT_ID,
        decision=BandDecision(
            outcome=OUTCOME_JUDGE,
            canonical_id=attach_target_id,
            similarity=similarity,
            t_high=None,
            t_low=None,
        ),
        candidates=list(slate),
    ))
    return JudgeAnswer(
        outcome=verdict,
        verdict=verdict,
        entry_elided=entry_elided,
        candidates_elided=candidates_elided,
        usage=_usage_of(recorded[before:]),
    )


def _as_memory_result(record: Mapping[str, Any], metadata: Mapping[str, Any]) -> Any:
    """One eval record as the ``MemoryResult`` the shipped judge consumes."""
    from fused_memory.models.enums import SourceStore  # noqa: PLC0415
    from fused_memory.models.memory import MemoryResult  # noqa: PLC0415

    return MemoryResult(
        id=str(record['memory_id']),
        content=str(record['content']),
        source_store=SourceStore.mem0,
        metadata=dict(metadata),
    )


def build_judge_fn(config: Any, recorded: Sequence[Any] = ()) -> Any:
    """The SEEDED live edge: drive the SHIPPED judge, not a re-implementation.

    ``server/write_triage_judge.judge_write`` is called with a duck-typed
    service exposing only ``config`` — which is all it reads — so this eval
    measures the production prompt, the production provider fan-out and the
    production parser. Re-implementing any of the three here would measure
    this script instead of the thing the operator is deciding about.

    The band decision is SYNTHESIZED rather than retrieved: the eval supplies
    its slate directly, so there is no retrieval to derive a winner from. It
    names the cluster canonical, which is exactly what a correct retrieval
    would have found, and marks the middle band — the only band that reaches a
    judge at all.
    """
    service = types.SimpleNamespace(config=config)

    def judge_fn(
        case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    ) -> JudgeAnswer:
        slate = [
            _as_memory_result(record, {
                'store_score': _SYNTHETIC_TOP_SCORE - index * _SYNTHETIC_SCORE_STEP,
            })
            for index, record in enumerate(candidates)
        ]
        return _ask_judge(
            service,
            str(case['content']),
            slate,
            outcome=OUTCOME_JUDGE,
            attach_target_id=slate[0].id if slate else None,
            similarity=_SYNTHETIC_TOP_SCORE,
            recorded=recorded,
        )

    return judge_fn


def build_retrieved_judge_fn(config: Any, recorded: Sequence[Any] = ()) -> Any:
    """The RETRIEVED live edge: the case already carries production's routing.

    The slate, the attach target, the band and the similarity all came from
    ``decide_band``/``select_judge_candidates`` over a real retrieval, so this
    only rebuilds the records and defers to the same judge edge. The candidate
    metadata is passed through whole because it carries the cosine the shipped
    selector re-reads.
    """
    service = types.SimpleNamespace(config=config)

    def judge_fn(
        case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    ) -> JudgeAnswer:
        slate = [
            _as_memory_result(record, record.get('metadata') or {})
            for record in candidates
        ]
        return _ask_judge(
            service,
            str(case['content']),
            slate,
            outcome=str(case['band']),
            attach_target_id=case['attach_target_id'],
            similarity=case['similarity'],
            recorded=recorded,
        )

    return judge_fn


def _dry_run_judge_fn() -> Any:
    """A fixed-answer judge that spends nothing. Proves the pipeline."""
    def judge_fn(case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
        return _DRY_RUN_VERDICT

    return judge_fn


def _limited(
    records: Sequence[Mapping[str, Any]], limit: int,
) -> list[Mapping[str, Any]]:
    """*limit* labelled records drawn ROUND-ROBIN across clusters, plus canonicals.

    Truncating RECORDS rather than cases, because a bare head-N slice can cut a
    cluster's canonical while keeping its members and the runner would then
    KeyError on an unresolvable slate partway through a paid run.

    Round-robin and not a head slice: the fixture is grouped by cluster in file
    order, so `records[:5]` is five records of ONE cluster, `_distractor_pool`
    draws only from OTHER clusters and comes back empty, every slate collapses
    to width 1, and the lone control case gets `candidates == []` — which
    `judge_write` short-circuits to `stored` with no provider call at all.

    Only NON-canonical records are drawn, since a canonical produces no
    labelled case; the ones each drawn record's slate needs come along anyway.
    """
    by_cluster: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        if str(record['label']) == LABEL_CANONICAL:
            continue
        by_cluster.setdefault(str(record['cluster_id']), []).append(record)
    kept: list[Mapping[str, Any]] = []
    for depth in range(max((len(v) for v in by_cluster.values()), default=0)):
        if len(kept) >= limit:
            break
        for group in by_cluster.values():
            if depth < len(group):
                kept.append(group[depth])
                if len(kept) >= limit:
                    break

    kept_ids = {id(r) for r in kept}
    wanted = {str(r['cluster_id']) for r in kept}
    canonicals = [
        r for r in records
        if str(r['memory_id']) in wanted and id(r) not in kept_ids
    ]
    logger.info(
        '--limit %d: measuring %d record(s) across %d cluster(s) '
        '(%d canonical(s) pulled in to keep every slate resolvable)',
        limit, len(kept) + len(canonicals), len(wanted), len(canonicals),
    )
    return [*kept, *canonicals]


def _retrieved_plan(
    args: Any,
    config: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    judge_candidate_count: int,
    aliases: Mapping[str, str],
) -> EvalPlan:
    """Retrieve, band and trim every labelled record exactly as production would.

    The retrieval width and both band edges come from the SAME config the
    shipped ``triage_write`` reads, so a re-calibration moves this eval with it.
    """
    import asyncio  # noqa: PLC0415

    from fused_memory.server.write_triage import (  # noqa: PLC0415
        resolve_bands,
        resolve_candidate_k,
    )
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    retrieval = load_retrieval()
    service = types.SimpleNamespace(config=config)
    k = resolve_candidate_k(service)
    t_high, t_low = resolve_bands(service)
    labelled = [r for r in records if str(r['label']) != LABEL_CANONICAL]
    logger.info(
        'Retrieving %d slate(s) from project_id=%s at k=%d, bands t_high=%s '
        't_low=%s', len(labelled), args.project_id, k, t_high, t_low,
    )

    async def prefetch() -> dict[str, dict[str, Any]]:
        memory = MemoryService(config)
        await memory.initialize()
        try:
            return await retrieval.prefetch_retrievals(
                memory, labelled, project_id=args.project_id, k=k,
            )
        finally:
            await memory.close()

    slates = retrieval.retrieved_slates(
        labelled, asyncio.run(prefetch()),
        t_high=t_high, t_low=t_low,
        judge_candidate_count=judge_candidate_count,
    )
    return plan_from_slates(records, slates, provenance={
        'project_id': args.project_id,
        't_high': t_high,
        't_low': t_low,
        'candidate_k': k,
        'canonical_absent': sum(1 for s in slates if not s.canonical_present),
        'degraded_retrievals': sum(1 for s in slates if s.degraded),
        'self_retrieved': sum(1 for s in slates if s.self_retrieved),
    }, aliases=aliases)


def _run(args: Any) -> int:
    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
        resolve_judge_candidate_count,
        resolve_judge_enabled,
        resolve_judge_model,
        resolve_judge_provider,
    )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    # FIRST, ahead of the fixture load and the config: `--report-path foo.md`
    # is a bad ARGUMENT, and whether it collides with its markdown sibling is
    # decided by the string alone. `run_judge_eval` re-resolves this on the
    # path it is finally handed (which `guard_committed_report` may have
    # redirected); rejecting here just means the operator is told now instead
    # of after a corpus-wide run they paid for.
    markdown_sibling(args.report_path)

    # BEFORE any work, and outside the --limit block: a bare --dry-run also
    # defaults to the committed path, and used to rewrite both committed
    # artifacts with fixed-answer numbers.
    report_path = guard_committed_report(
        args.report_path, dry_run=args.dry_run, limit=args.limit,
    )
    aliases = load_canonical_aliases(args.canonical_aliases) if args.canonical_aliases else {}

    config = FusedMemoryConfig()
    # Mutated on the in-memory config only — config.yaml is never touched. The
    # shipped resolvers read live per call, so this is the one place the cap
    # has to move for both the slate this script builds and the re-trim
    # `judge_write` applies to it.
    if args.judge_candidate_count is not None:
        config.write_triage.judge_candidate_count = args.judge_candidate_count
    service = types.SimpleNamespace(config=config)
    provider = resolve_judge_provider(service)
    model = resolve_judge_model(service)
    # Resolved from the SAME config the shipped judge reads, and recorded even
    # on a --dry-run: the two facts the numbers cannot be read without.
    # `judge_candidate_count` is the width `judge_write` trims the slate to,
    # so it — not the width this script builds — is what the model saw.
    # `judge_enabled` false makes `judge_write` return `stored` on its first
    # line for every case, which scores the `distractor` control 1.0 and
    # `duplicate` 0.0 while spending nothing and writing a report that is
    # otherwise indistinguishable from a measurement.
    judge_candidate_count = resolve_judge_candidate_count(service)
    judge_enabled = resolve_judge_enabled(service)

    records = load_fixture(args.fixture)
    logger.info('Loaded %d labeled record(s) from %s', len(records), args.fixture)
    if args.limit is not None:
        records = _limited(records, args.limit)

    with (
        field_chars_override(args.field_chars) as field_chars,
        usage_recording_openai() as recorded,
    ):
        if args.slate_mode == SLATE_RETRIEVED:
            plan = _retrieved_plan(
                args, config, records,
                judge_candidate_count=judge_candidate_count, aliases=aliases,
            )
        else:
            plan = seeded_plan(records, distractors=args.distractors, aliases=aliases)

        if args.dry_run:
            judge_fn = _dry_run_judge_fn()
            provider, model = 'dry-run', f'fixed:{_DRY_RUN_VERDICT}'
            logger.info('--dry-run: no provider call will be made')
        else:
            # Logged BEFORE the first call, so a mis-resolved model is visible
            # while the run is still free to abort.
            logger.info('Judge resolves to provider=%s model=%s', provider, model)
            judge_fn = (
                build_retrieved_judge_fn(config, recorded)
                if args.slate_mode == SLATE_RETRIEVED
                else build_judge_fn(config, recorded)
            )

        report = run_judge_eval(
            plan=plan,
            judge_fn=judge_fn,
            report_path=report_path,
            cases_path=args.cases_path,
            provenance={
                'fixture_path': package_relative(args.fixture),
                'judge_provider': provider,
                'judge_model': model,
                # Present on EVERY run, `None` on a full one. An absent key
                # would be indistinguishable from an artifact predating the
                # field, and this is the one field that says a committed
                # report is a partial smoke rather than the corpus-wide
                # measurement the task-3169 flip gate reads it as.
                'limit': args.limit,
                'canonical_aliases_path': (
                    package_relative(args.canonical_aliases) if args.canonical_aliases else None
                ),
                'canonical_aliases_count': len(aliases),
                'cases_path': package_relative(args.cases_path) if args.cases_path else None,
                'field_chars': field_chars,
                'judge_candidate_count': judge_candidate_count,
                'judge_enabled': judge_enabled,
            },
        )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    import argparse  # noqa: PLC0415

    repo = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default=str(
        repo / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'))
    parser.add_argument('--report-path', dest='report_path',
                        default=_DEFAULT_REPORT_PATH)
    parser.add_argument('--config', default=None,
                        help='Path to fused-memory config file (sets CONFIG_PATH)')
    parser.add_argument('--distractors', type=int, default=4,
                        help="cross-cluster candidates per slate; 4 gives PRD C1's "
                             'top-5 width alongside the attach target (default: 4)')
    parser.add_argument('--limit', type=int, default=None,
                        help='measure only N labelled records, drawn round-robin across '
                             'clusters, plus the canonicals their slates need '
                             '(a cheap live smoke). Recorded in the report as '
                             'provenance.limit')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        help='score a fixed-answer judge; makes no provider call')
    parser.add_argument('--slate-mode', dest='slate_mode', default=SLATE_SEEDED,
                        choices=SLATE_MODES,
                        help='seeded: the cluster canonical is placed at slate '
                             'position 0. retrieved: the slate, the attach '
                             'target and the band all come from a real '
                             'retrieval against the live store, as production '
                             'would produce them (default: seeded)')
    parser.add_argument('--project-id', dest='project_id', default='reify',
                        help='the corpus retrieved against under '
                             '--slate-mode retrieved (default: reify)')
    parser.add_argument('--field-chars', dest='field_chars', type=int, default=None,
                        help="override the judge's per-field prompt budget for "
                             'this run; 0 renders every field UN-elided. '
                             'Recorded as provenance.field_chars (default: the '
                             'shipped _FIELD_CHARS)')
    parser.add_argument('--judge-candidate-count', dest='judge_candidate_count',
                        type=int, default=None,
                        help='override write_triage.judge_candidate_count for '
                             'this run, in memory only — config.yaml is not '
                             'written')
    parser.add_argument('--canonical-aliases', dest='canonical_aliases', default=None,
                        help='the {old_cluster_canonical_id: current_memory_id} sidecar: '
                             "an attach to a rotated canonical's successor counts as an "
                             'attach to the canonical (default: none)')
    parser.add_argument('--cases-path', dest='cases_path', default=None,
                        help='append one JSON line per case as it completes: '
                             'slate, attach target, band, verdict, outcome and '
                             'elision flags (default: no per-case dump)')
    return _run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
