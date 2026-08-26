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
  # Prove the pipeline without spending anything.
  python scripts/eval_write_triage_judge.py --dry-run

  # A cheap live smoke over the first few cases.
  python scripts/eval_write_triage_judge.py --limit 5

  # The full measured pass, written to calibration/.
  python scripts/eval_write_triage_judge.py
"""
from __future__ import annotations

import importlib.util
import json
import logging
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fused_memory.server.write_triage import (
    OUTCOME_AMENDED,
    OUTCOME_CONTESTED,
    OUTCOME_RESTATED,
    OUTCOME_STORED,
    TRIAGE_OUTCOMES,
)

logger = logging.getLogger(__name__)

#: The fused-memory package root — the anchor for paths RECORDED in the report.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

_CALIBRATE_PATH = _PACKAGE_ROOT / 'scripts' / 'calibrate_write_triage.py'


def _load_calibrate() -> types.ModuleType:
    """Load leaf alpha's script as a module.

    ``scripts/`` is not an importable package, so the sibling is reached the
    same way its own test suite reaches it. Importing it rather than copying
    its four label constants and its fixture loader is the point: a fixture
    format change or a fifth label lands in ONE place, and this script either
    follows it or fails loudly at ``build_judge_cases``.
    """
    import sys  # noqa: PLC0415

    mod_name = 'calibrate_write_triage'
    cached = sys.modules.get(mod_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, _CALIBRATE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {_CALIBRATE_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_calibrate = _load_calibrate()

# Alpha's vocabulary and fixture handling, re-exported rather than re-spelled.
LABEL_CANONICAL = _calibrate.LABEL_CANONICAL
LABEL_DUPLICATE = _calibrate.LABEL_DUPLICATE
LABEL_DISTINCT = _calibrate.LABEL_DISTINCT
LABEL_PSEUDO_CONTRADICTION = _calibrate.LABEL_PSEUDO_CONTRADICTION
load_fixture = _calibrate.load_fixture
package_relative = _calibrate.package_relative

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
    if not pool or count <= 0:
        return []
    take = min(count, len(pool))
    start = offset % len(pool)
    return [pool[(start + i) % len(pool)] for i in range(take)]


def build_judge_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    distractors: int,
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
        cases.append({
            'memory_id': str(record['memory_id']),
            'content': record['content'],
            'category': record.get('category'),
            'candidates': slate,
            'expected_class': label,
            'acceptable_outcomes': acceptable,
        })

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
        cases.append({
            'memory_id': str(record['memory_id']),
            'content': record['content'],
            'category': record.get('category'),
            'candidates': slate,
            'expected_class': CLASS_DISTRACTOR,
            'acceptable_outcomes': _DISTRACTOR_ACCEPTABLE,
        })

    return cases


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

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
        name: dict.fromkeys(TRIAGE_OUTCOMES, 0) for name in EVAL_CLASSES
    }
    duplicate_split = {OUTCOME_RESTATED: 0, OUTCOME_AMENDED: 0}
    false_contested = 0

    for case, verdict in zip(cases, verdicts, strict=True):
        name = str(case['expected_class'])
        bucket = per_class.setdefault(name, {'n': 0, 'correct': 0})
        bucket['n'] += 1
        if verdict in case['acceptable_outcomes']:
            bucket['correct'] += 1
        row = confusion.setdefault(name, dict.fromkeys(TRIAGE_OUTCOMES, 0))
        row[verdict] = row.get(verdict, 0) + 1
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
    'reported as a distribution and is not scored as error.',
    'The distractor class is a control this script constructs, not a curator '
    'label: one case per cluster whose slate carries no correct attach target '
    'at all. It is what distinguishes a judge that classifies from a judge '
    'that attaches to whatever it is shown.',
)


def build_report(
    *,
    scored: Mapping[str, Any],
    provenance: Mapping[str, Any],
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
    """
    per_class = dict(scored['per_class'])
    run_provenance = dict(provenance)
    run_provenance.setdefault(
        'case_count', sum(entry['n'] for entry in per_class.values()),
    )
    for key in ('record_count', 'candidate_count', 'distractor_count'):
        run_provenance.setdefault(key, None)

    return {
        'per_class': per_class,
        'confusion': dict(scored['confusion']),
        'duplicate_outcome_split': dict(scored['duplicate_outcome_split']),
        'false_contested': scored['false_contested'],
        'contested_ground_truth': {
            'available': False,
            'reason': CONTESTED_GROUND_TRUTH_REASON,
        },
        'caveats': list(CAVEATS),
        'provenance': run_provenance,
    }


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

    outcomes = list(TRIAGE_OUTCOMES)
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

def run_judge_eval(
    *,
    records: Sequence[Mapping[str, Any]],
    judge_fn: Any,
    report_path: str | Path,
    provenance: Mapping[str, Any],
    distractors: int,
) -> dict[str, Any]:
    """Build cases, ask *judge_fn* once each, score, write both artifacts.

    ``judge_fn(case, candidates)`` receives the case and its slate RESOLVED to
    records — a judge handed bare ids has no text to compare and is answering
    noise, which is the same defect the production seam's own tests pin.

    Nothing is caught. A judge error propagates exactly as ``run_calibration``
    lets an ``embed_fn`` error propagate: swallowing it would be
    indistinguishable from a genuine misclassification, so it would both
    shrink the measured population and depress the accuracy reported over it.

    A dangling candidate id raises ``KeyError`` for the same reason — a
    silently-skipped candidate narrows a slate the report claims was 5 wide.
    """
    cases = build_judge_cases(records, distractors=distractors)
    by_id = {str(r['memory_id']): r for r in records}
    logger.info('Built %d case(s) from %d record(s)', len(cases), len(records))

    verdicts: list[str] = []
    for index, case in enumerate(cases, 1):
        candidates = [by_id[cid] for cid in case['candidates']]
        verdicts.append(judge_fn(case, candidates))
        if index % 10 == 0:
            logger.info('Judged %d/%d case(s)', index, len(cases))

    scored = score_cases(cases, verdicts)

    run_provenance = dict(provenance)
    run_provenance.setdefault('record_count', len(records))
    run_provenance.setdefault('case_count', len(cases))
    run_provenance.setdefault('candidate_count', distractors + 1)
    run_provenance.setdefault('distractor_count', distractors)
    report = build_report(scored=scored, provenance=run_provenance)

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    report_path.with_suffix('.md').write_text(render_markdown(report))
    logger.info('Wrote %s and its .md sibling', report_path)

    return report


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


def build_judge_fn(config: Any) -> Any:
    """The LIVE edge: drive the SHIPPED judge, not a re-implementation.

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

    ``asyncio.run`` per case, which is why ``_run`` below is synchronous:
    ``judge_write`` is the only awaitable in the whole pipeline and it needs
    no initialized ``MemoryService``, so an async ``_run`` would buy nothing
    and would force a nested-loop bridge around every single call.
    """
    import asyncio  # noqa: PLC0415

    from fused_memory.models.enums import SourceStore  # noqa: PLC0415
    from fused_memory.models.memory import MemoryResult  # noqa: PLC0415
    from fused_memory.server.write_triage import OUTCOME_JUDGE, BandDecision  # noqa: PLC0415
    from fused_memory.server.write_triage_judge import judge_write  # noqa: PLC0415

    service = types.SimpleNamespace(config=config)

    def judge_fn(case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
        slate = [
            MemoryResult(
                id=str(record['memory_id']),
                content=str(record['content']),
                source_store=SourceStore.MEM0,
                metadata={
                    'store_score': _SYNTHETIC_TOP_SCORE - index * _SYNTHETIC_SCORE_STEP,
                },
            )
            for index, record in enumerate(candidates)
        ]
        decision = BandDecision(
            outcome=OUTCOME_JUDGE,
            canonical_id=slate[0].id if slate else None,
            similarity=_SYNTHETIC_TOP_SCORE,
            t_high=None,
            t_low=None,
        )
        return asyncio.run(judge_write(
            memory_service=service,
            content=str(case['content']),
            project_id=_EVAL_PROJECT_ID,
            decision=decision,
            candidates=slate,
        ))

    return judge_fn


def _dry_run_judge_fn() -> Any:
    """A fixed-answer judge that spends nothing. Proves the pipeline."""
    def judge_fn(case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
        return _DRY_RUN_VERDICT

    return judge_fn


def _run(args: Any) -> int:
    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
        resolve_judge_model,
        resolve_judge_provider,
    )

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    service = types.SimpleNamespace(config=config)
    provider = resolve_judge_provider(service)
    model = resolve_judge_model(service)

    records = load_fixture(args.fixture)
    logger.info('Loaded %d labeled record(s) from %s', len(records), args.fixture)

    if args.limit:
        # Truncating RECORDS rather than cases keeps every case's slate
        # resolvable — a case whose canonical was cut would raise in the
        # runner, which is the right behaviour there and a useless one here.
        records = records[: args.limit]
        logger.info('--limit %d: measuring %d record(s)', args.limit, len(records))

    if args.dry_run:
        judge_fn = _dry_run_judge_fn()
        provider, model = 'dry-run', f'fixed:{_DRY_RUN_VERDICT}'
        logger.info('--dry-run: no provider call will be made')
    else:
        # Logged BEFORE the first call, so a mis-resolved model is visible
        # while the run is still free to abort.
        logger.info('Judge resolves to provider=%s model=%s', provider, model)
        judge_fn = build_judge_fn(config)

    report = run_judge_eval(
        records=records,
        judge_fn=judge_fn,
        report_path=args.report_path,
        provenance={
            'fixture_path': package_relative(args.fixture),
            'judge_provider': provider,
            'judge_model': model,
        },
        distractors=args.distractors,
    )
    print(json.dumps(report, indent=2))
    return 0


def main() -> int:
    import argparse  # noqa: PLC0415

    repo = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default=str(
        repo / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'))
    parser.add_argument('--report-path', dest='report_path', default=str(
        repo / 'calibration' / 'write_triage_judge_accuracy_report.json'))
    parser.add_argument('--config', default=None,
                        help='Path to fused-memory config file (sets CONFIG_PATH)')
    parser.add_argument('--distractors', type=int, default=4,
                        help="cross-cluster candidates per slate; 4 gives PRD C1's "
                             'top-5 width alongside the attach target (default: 4)')
    parser.add_argument('--limit', type=int, default=None,
                        help='measure only the first N records (a cheap live smoke)')
    parser.add_argument('--dry-run', dest='dry_run', action='store_true',
                        help='score a fixed-answer judge; makes no provider call')
    return _run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
