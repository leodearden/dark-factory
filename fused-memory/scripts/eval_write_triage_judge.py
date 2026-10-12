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

  # A cheap live smoke over a few cases, kept off the committed artifact.
  python scripts/eval_write_triage_judge.py --limit 5 --report-path /tmp/smoke.json

  # The committed arbiter (plans/write-triage-flip-readiness-prd.md C2'):
  # production's retrieval, bands and attach targets, every case dumped as it
  # completes.
  python scripts/eval_write_triage_judge.py --slate-mode retrieved --project-id reify \\
      --canonical-aliases tests/fixtures/write_triage_calibration.canonical_aliases.json \\
      --report-path calibration/write_triage_judge_accuracy_report.json \\
      --cases-path calibration/write_triage_judge_accuracy_report.cases.jsonl
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import logging
import types
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fused_memory.server.write_triage import (
    OUTCOME_JUDGE,
    OUTCOME_STORED,
    JudgeUsage,
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


# Pyright cannot follow the by-path loader, so it alone resolves `_scoring` and
# `_cases` through the scripts/ extraPath; at runtime each is the by-path load.
if TYPE_CHECKING:
    import eval_write_triage_cases as _cases
    import eval_write_triage_scoring as _scoring
else:
    _scoring = _load_script(
        _PACKAGE_ROOT / 'scripts' / 'eval_write_triage_scoring.py', 'eval_write_triage_scoring',
    )
    _cases = _load_script(
        _PACKAGE_ROOT / 'scripts' / 'eval_write_triage_cases.py', 'eval_write_triage_cases',
    )

# PRD C2' and task 5807 call this at eval_write_triage_judge.py::score_attachments.
score_attachments = _scoring.score_attachments


_RETRIEVAL_PATH = _PACKAGE_ROOT / 'scripts' / 'eval_write_triage_retrieval.py'


def load_retrieval() -> types.ModuleType:
    """Load the live-store edge, the same by-path way the calibrator loads.

    Deliberately NOT loaded at import: it pulls in the server's retrieval
    stack, and the whole pure core below is testable without it.
    """
    return _load_script(_RETRIEVAL_PATH, 'eval_write_triage_retrieval')


_WORDING_PATH = _PACKAGE_ROOT / 'scripts' / 'write_triage_judge_wording.py'


def load_wording() -> types.ModuleType:
    """Load the judge-wording knob (PRD §11 D16), the same by-path way.

    Loaded at import, unlike the retrieval edge: it pulls in only the judge
    module, and the guard's default wording is one of its names.
    """
    return _load_script(_WORDING_PATH, 'write_triage_judge_wording')


_wording = load_wording()


# Alpha's fixture handling, re-exported rather than re-spelled.
load_fixture = _calibrate.load_fixture
load_canonical_aliases = _calibrate.load_canonical_aliases
package_relative = _calibrate.package_relative


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


def _as_answer(value: Any) -> _scoring.JudgeAnswer:
    """Accept a bare verdict from a ``judge_fn`` that has nothing else to say."""
    if isinstance(value, _scoring.JudgeAnswer):
        return value
    return _scoring.JudgeAnswer(outcome=str(value), verdict=str(value))


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


def run_judge_eval(
    *,
    plan: _cases.EvalPlan,
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

    A dangling candidate id already raised when *plan* was built: a
    silently-skipped candidate would narrow a slate the report claims was 5 wide.

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
        paired = zip(cases, plan.candidate_records, strict=True)
        for index, (case, records) in enumerate(paired, 1):
            answer = _as_answer(judge_fn(case, list(records)))
            row = _scoring.case_row(index, case, answer, records, plan.fixture_by_id)
            emit(row)
            rows.append(row)
            if index % 10 == 0:
                logger.info('Judged %d/%d case(s)', index, len(cases))

    scored = _scoring.score_cases(cases, [row['outcome'] for row in rows])

    run_provenance = {**dict(plan.provenance), **dict(provenance)}
    run_provenance.setdefault('record_count', plan.record_count)
    run_provenance.setdefault('case_count', len(cases))
    _record_slate_widths(run_provenance, cases)
    report = _scoring.build_report(
        scored=scored,
        provenance=run_provenance,
        production_shape=score_attachments(rows),
    )

    # `markdown_path` was composed (and its collision with `report_path`
    # rejected) at the top of this function — reused here rather than
    # recomposed, so the path that was validated is the path that is written.
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    markdown_path.write_text(_scoring.render_markdown(report))
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
#: ``build_judge_cases`` chose — the band winner first. These numbers never
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
    report_path: str, *, dry_run: bool, limit: int | None, slate_mode: str,
    wording: str = _wording.WORDING_SHIPPED,
) -> str:
    """Keep a non-measurement run from publishing itself as the measurement.

    The committed report is what ``write_triage.judge_accuracy_report_path``
    points at and, per D10, what the operator reads at the task-3169 flip
    gate. It is the RETRIEVED-slate arbiter (flip-readiness PRD C2'), so a
    measurement in any other slate mode aimed at it is refused outright. Two
    more kinds of run can reach it without having measured what it claims,
    and they are NOT the same hazard, so they get different answers:

    - ``--dry-run`` measures NOTHING. Every number in its report comes from a
      fixed-answer stub, so publishing it would put fabricated figures under
      the operator's nose with no artifact-level tell beyond a
      ``judge_provider`` field nobody is obliged to read. It is REDIRECTED to
      a temp path — the documented "prove the pipeline" invocation keeps
      working and still prints its report, it just cannot overwrite the
      committed one.
    - a non-shipped ``--wording`` measures a prompt production does not
      send, so it is refused outright beside the slate-mode refusal.
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

    if slate_mode != _scoring.SLATE_RETRIEVED:
        raise ValueError(
            f"the committed artifact is the retrieved-slate arbiter (PRD C2'); pass "
            f'--report-path for a {slate_mode} run',
        )

    if wording != _wording.WORDING_SHIPPED:
        raise ValueError(
            f'the committed artifact measures the shipped wording; pass '
            f'--report-path for a {wording} run',
        )

    if limit is not None:
        logger.warning(
            'a --limit run is about to OVERWRITE the committed report at '
            '%s. It is a partial smoke, not the corpus-wide measurement '
            'the task-3169 flip gate reads: provenance.limit=%d records '
            'that. Pass --report-path to keep the committed artifact.',
            report_path, limit,
        )

    return report_path


def apply_field_chars(config: Any, requested: int | None) -> int:
    """Set this run's ``write_triage.judge_field_chars``; return the width to record.

    Written on the run's own in-memory config, exactly as
    ``--judge-candidate-count`` is: config.yaml is never written and nothing
    module-global changes, and the shipped judge reads the leaf live per call.

    ``0`` means NO elision. The leaf is a positive cap, so it is spelled there
    as ``sys.maxsize``, while the returned provenance keeps ``0``. ``None``
    writes nothing and returns the width the config already resolves to.

    A negative width is refused: the resolver would silently read the default
    instead, so the provenance would record a width the judge never used.
    """
    import sys  # noqa: PLC0415

    from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
        resolve_judge_field_chars,
    )

    if requested is None:
        return resolve_judge_field_chars(types.SimpleNamespace(config=config))
    if requested < 0:
        raise ValueError(f'--field-chars must be 0 (no elision) or positive, got {requested}')
    config.write_triage.judge_field_chars = sys.maxsize if requested == 0 else requested
    return requested


def _elision_flags(
    content: str, slate: Sequence[Any], field_chars: int,
) -> tuple[bool, dict[str, bool]]:
    """Which rendered fields the judge's own ``_elide`` cut at *field_chars*.

    Asked of the shipped function rather than re-derived from a length
    comparison here, so the flags cannot disagree with the prompt.
    """
    from fused_memory.server.write_triage_judge import _elide  # noqa: PLC0415

    return (
        _elide(content, field_chars) != content,
        {c.id: _elide(c.content, field_chars) != c.content for c in slate},
    )


def _usage_row(usage: JudgeUsage | None) -> dict[str, int] | None:
    """The verdict's usage under the artifact's ESTABLISHED keys, or ``None``.

    The key names are kept because ``_spend`` and
    ``fused-memory/scripts/score_write_triage_pairs.py::JudgedCase.from_row``
    read them, and the row carries nothing they do not read. The mapping is
    exact: output (``completion_tokens``) already includes reasoning on both
    OpenAI APIs.
    """
    if usage is None:
        return None
    return {
        'prompt_tokens': usage.input_tokens,
        'completion_tokens': usage.output_tokens,
        'total_tokens': usage.input_tokens + usage.output_tokens,
    }


def _ask_judge(
    service: Any,
    content: str,
    slate: Sequence[Any],
    *,
    outcome: str,
    band_winner_id: str | None,
    similarity: float | None,
) -> _scoring.JudgeAnswer:
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
    from fused_memory.server.write_triage_judge import (  # noqa: PLC0415
        judge_write,
        resolve_judge_field_chars,
    )

    if outcome != OUTCOME_JUDGE:
        return _scoring.JudgeAnswer(outcome=outcome)

    entry_elided, candidates_elided = _elision_flags(
        content, slate, resolve_judge_field_chars(service),
    )
    verdict = asyncio.run(judge_write(
        memory_service=service,
        content=content,
        project_id=_EVAL_PROJECT_ID,
        decision=BandDecision(
            outcome=OUTCOME_JUDGE,
            canonical_id=band_winner_id,
            similarity=similarity,
            t_high=None,
            t_low=None,
        ),
        candidates=list(slate),
    ))
    return _scoring.JudgeAnswer(
        outcome=verdict.outcome,
        verdict=verdict.outcome,
        entry_elided=entry_elided,
        candidates_elided=candidates_elided,
        usage=_usage_row(verdict.usage),
        candidate_id=verdict.candidate_id,
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


def build_judge_fn(config: Any) -> Any:
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
    ) -> _scoring.JudgeAnswer:
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
            band_winner_id=slate[0].id if slate else None,
            similarity=_SYNTHETIC_TOP_SCORE,
        )

    return judge_fn


def build_retrieved_judge_fn(config: Any) -> Any:
    """The RETRIEVED live edge: the case already carries production's routing.

    The slate, the band winner, the band and the similarity all came from
    ``decide_band``/``select_judge_candidates`` over a real retrieval, so this
    only rebuilds the records and defers to the same judge edge. The candidate
    metadata is passed through whole because it carries the cosine the shipped
    selector re-reads.
    """
    service = types.SimpleNamespace(config=config)

    def judge_fn(
        case: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]],
    ) -> _scoring.JudgeAnswer:
        slate = [
            _as_memory_result(record, record.get('metadata') or {})
            for record in candidates
        ]
        return _ask_judge(
            service,
            str(case['content']),
            slate,
            outcome=str(case['band']),
            band_winner_id=case['band_winner_id'],
            similarity=case['similarity'],
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
    cluster's canonical while keeping its members, whose slates then cannot resolve.

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
        if str(record['label']) == _scoring.LABEL_CANONICAL:
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
    fixture: Sequence[Mapping[str, Any]],
    judge_candidate_count: int,
    aliases: Mapping[str, str],
) -> _cases.EvalPlan:
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
    labelled = [r for r in records if str(r['label']) != _scoring.LABEL_CANONICAL]
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
    return _cases.plan_from_slates(records, slates, provenance={
        'project_id': args.project_id,
        't_high': t_high,
        't_low': t_low,
        'candidate_k': k,
        'canonical_absent': sum(1 for s in slates if not s.canonical_present),
        'degraded_retrievals': sum(1 for s in slates if s.degraded),
        'self_retrieved': sum(1 for s in slates if s.self_retrieved),
    }, aliases=aliases, fixture=fixture)


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
        slate_mode=args.slate_mode, wording=args.wording,
    )
    aliases = load_canonical_aliases(args.canonical_aliases) if args.canonical_aliases else {}

    config = FusedMemoryConfig()
    # Mutated on the in-memory config only — config.yaml is never touched. The
    # shipped resolvers read live per call, so this is the one place the cap
    # has to move for both the slate this script builds and the re-trim
    # `judge_write` applies to it.
    if args.judge_candidate_count is not None:
        config.write_triage.judge_candidate_count = args.judge_candidate_count
    field_chars = apply_field_chars(config, args.field_chars)
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

    fixture = load_fixture(args.fixture)
    logger.info('Loaded %d labeled record(s) from %s', len(fixture), args.fixture)
    records = fixture if args.limit is None else _limited(fixture, args.limit)

    if args.slate_mode == _scoring.SLATE_RETRIEVED:
        plan = _retrieved_plan(
            args, config, records, fixture=fixture,
            judge_candidate_count=judge_candidate_count, aliases=aliases,
        )
    else:
        plan = _cases.seeded_plan(records, distractors=args.distractors, aliases=aliases)

    if args.dry_run:
        judge_fn = _dry_run_judge_fn()
        provider, model = 'dry-run', f'fixed:{_DRY_RUN_VERDICT}'
        logger.info('--dry-run: no provider call will be made')
    else:
        # Logged BEFORE the first call, so a mis-resolved model is visible
        # while the run is still free to abort.
        logger.info('Judge resolves to provider=%s model=%s', provider, model)
        judge_fn = (
            build_retrieved_judge_fn(config)
            if args.slate_mode == _scoring.SLATE_RETRIEVED
            else build_judge_fn(config)
        )

    with _wording.judge_wording(args.wording) as prompt_sha256:
        report = run_judge_eval(
            plan=plan,
            judge_fn=judge_fn,
            report_path=report_path,
            cases_path=args.cases_path,
            provenance={
                'fixture_path': package_relative(args.fixture),
                'judge_provider': provider,
                'judge_model': model,
                'judge_system_prompt_sha256': prompt_sha256,
                # Present on EVERY run, `None` on a full one. An absent key
                # would be indistinguishable from an artifact predating the
                # field, and this is the one field that says a committed
                # report is a partial smoke rather than the corpus-wide
                # measurement the task-3169 flip gate reads it as.
                'limit': args.limit,
                'canonical_aliases_path': (
                    package_relative(args.canonical_aliases)
                    if args.canonical_aliases else None
                ),
                'canonical_aliases_count': len(aliases),
                'cases_path': (
                    package_relative(args.cases_path) if args.cases_path else None
                ),
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
    parser.add_argument('--slate-mode', dest='slate_mode', default=_scoring.SLATE_SEEDED,
                        choices=_scoring.SLATE_MODES,
                        help='seeded: the cluster canonical is placed at slate '
                             'position 0. retrieved: the slate, the band '
                             'winner and the band all come from a real '
                             'retrieval against the live store, as production '
                             'would produce them (default: seeded)')
    parser.add_argument('--project-id', dest='project_id', default='reify',
                        help='the corpus retrieved against under '
                             '--slate-mode retrieved (default: reify)')
    parser.add_argument('--field-chars', dest='field_chars', type=int, default=None,
                        help="override the judge's per-field prompt budget for "
                             'this run; 0 renders every field UN-elided. '
                             'Recorded as provenance.field_chars (default: '
                             'write_triage.judge_field_chars as the config '
                             'resolves it)')
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
                             'slate, band winner, the candidate the verdict '
                             'named, attach target, band, verdict, outcome and '
                             'elision flags (default: no per-case dump)')
    parser.add_argument('--wording', default=_wording.WORDING_SHIPPED,
                        choices=_wording.WORDINGS,
                        help='the judge system prompt the run is made under; '
                             'pre-psi is the pre-psi wording as an eval-side '
                             'override. Recorded as '
                             'provenance.judge_system_prompt_sha256 '
                             '(default: shipped)')
    return _run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
