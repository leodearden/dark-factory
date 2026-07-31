#!/usr/bin/env python3
"""Derive the ``add_memory`` write-triage band thresholds T_high / T_low from
MEASURED similarity distributions over a labeled curator corpus.

PRD ``docs/prds/memory-write-path-convergence.md`` §9 leaf α (contract C1,
decision D1).

The constraint, and why it exists
---------------------------------
**No threshold in this script is chosen a priori.** Both bounds are
*measured order statistics* of the observed distributions, and every
degenerate input returns ``None`` plus a structured reason rather than a
fabricated number. ``None`` means UNCALIBRATED, which the triage router
must read as fail-open to ``stored``.

This is not pedantry. The existing near-duplicate guard's ``0.92`` default
was inherited from a figure cited in Mem0's own docs, and the one genuine
rediscovery pair we have actually measured scores ``0.824`` — so that
guard could never have fired on the very case it exists to catch. A
plausible-looking constant is exactly the failure mode being corrected;
re-introducing one here, even as a fallback, would reproduce it.

Metric-space parity (why the measurement transfers to the live guard)
---------------------------------------------------------------------
The cosine measured here is the *same quantity* as the Qdrant
``relevance_score`` the guard compares against its threshold, because:

- ``backends/mem0_client.py`` pins ``infer=False`` on every add, so Mem0
  stores and embeds content VERBATIM — there is no LLM fact-extraction
  rewrite between the text in the fixture and the vector in the index.
- Mem0's embedder is built from ``config.embedder`` (provider ``openai``,
  model ``text-embedding-3-small``), and Mem0 passes NO custom
  ``dimensions``.

So this script must mirror that call exactly — same model, no
``dimensions`` override. Passing a different dimensionality would silently
move the measurement into a different space and make the derived
thresholds inapplicable to the guard they are meant to configure. The
report records the embedder model and dimensions actually used so a future
reader can tell whether an embedder change invalidates the calibration.

Structure
---------
Mirrors the sweep-script family (``scripts/audit_duplicate_memories.py``):
all computation lives in pure synchronous functions unit-tested against
injected vectors and retrievals; the live embedder and
``MemoryService.search`` are injected only at the CLI boundary. The whole
test suite therefore runs with no ``OPENAI_API_KEY``, no network and no
Qdrant.

Usage
-----
  # Report only (default): measure, derive, write the report, change nothing.
  python scripts/calibrate_write_triage.py --project-id reify

  # Also write the derived thresholds into config.yaml's write_triage block.
  python scripts/calibrate_write_triage.py --project-id reify --write-config
"""
from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Labels the curator assigned. `distinct` and `pseudo_contradiction` are the
# HARD NEGATIVES: same cluster, same topic, but adjudicated not-the-same-claim.
LABEL_CANONICAL = 'canonical'
LABEL_DUPLICATE = 'duplicate'
LABEL_DISTINCT = 'distinct'
LABEL_PSEUDO_CONTRADICTION = 'pseudo_contradiction'

_NEGATIVE_LABELS = frozenset({LABEL_DISTINCT, LABEL_PSEUDO_CONTRADICTION})

# The fused-memory package root — the anchor for paths RECORDED in config.yaml
# and in the report's provenance.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def package_relative(path: str | Path) -> str:
    """Render `path` relative to the fused-memory package root when inside it.

    Paths that get WRITTEN INTO config.yaml and the report provenance must not
    bake in the absolute checkout they happened to be produced in: this script
    runs in a per-task git worktree, so an absolute
    `/.../.worktrees/<n>/fused-memory/calibration/...` string would be a
    dangling reference the moment that worktree is reset or the branch merges.
    Anything outside the package (e.g. an explicit `--report-path /tmp/x.json`)
    is returned unchanged, since there is no stable anchor for it.
    """
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(_PACKAGE_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------

def load_fixture(path: str | Path) -> list[dict[str, Any]]:
    """Read the labeled JSONL fixture strictly.

    A malformed line raises with its 1-based line number rather than being
    skipped: silently dropping a record would shrink the measured
    population without saying so, yielding a report whose thresholds look
    fine but were computed on a subset.
    """
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{lineno}: malformed JSON line: {exc}') from exc
            if not isinstance(record, dict):
                raise ValueError(f'{path}:{lineno}: expected a JSON object, got {type(record).__name__}')
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

def build_pair_sets(records: list[dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Partition every unordered record pair into three disjoint classes.

    Keyed on ``cluster_id`` — the CANONICAL memory UUID, never the gate id.
    Gates esc-5534/5547/5561/5610 each produced two canonicals, so keying by
    gate would fuse two canonicals' member sets into one cluster and inject
    pairs that are not duplicates into the positive class, dragging the
    derived T_high down.

    - ``true_dup_pairs`` — same cluster, both members ``duplicate`` or
      ``canonical``: the curator-confirmed genuine rediscoveries.
    - ``unrelated_pairs`` — different clusters: the measured negative class.
      The corpus is domain-homogeneous, so these scores must be measured
      rather than assumed low.
    - ``hard_negative_pairs`` — same cluster, but at least one member
      labeled ``distinct`` or ``pseudo_contradiction``: same topic,
      curator-ruled NOT duplicates. The hardest negatives for the
      deterministic band.

    The partition is total: every unordered pair lands in exactly one class.
    """
    true_dup: list[dict[str, str]] = []
    unrelated: list[dict[str, str]] = []
    hard_negative: list[dict[str, str]] = []

    n = len(records)
    for i in range(n):
        left = records[i]
        for j in range(i + 1, n):
            right = records[j]
            a, b = sorted((str(left['memory_id']), str(right['memory_id'])))
            pair = {'a': a, 'b': b}
            if left['cluster_id'] != right['cluster_id']:
                unrelated.append(pair)
            elif left['label'] in _NEGATIVE_LABELS or right['label'] in _NEGATIVE_LABELS:
                hard_negative.append(pair)
            else:
                true_dup.append(pair)

    return {
        'true_dup_pairs': true_dup,
        'unrelated_pairs': unrelated,
        'hard_negative_pairs': hard_negative,
    }


# ---------------------------------------------------------------------------
# Similarity + distribution statistics
# ---------------------------------------------------------------------------

def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Raw cosine between two embedding vectors.

    This is deliberately the same quantity Qdrant reports as
    ``relevance_score`` — see the module docstring's metric-space parity
    note. stdlib only; no numpy dependency is added for a dot product.

    A zero-norm or mismatched-length input raises rather than yielding NaN:
    a NaN would propagate silently through the distributions and corrupt
    every statistic derived from them.
    """
    if len(a) != len(b):
        raise ValueError(f'vector length mismatch: {len(a)} != {len(b)}')
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError(
            'cosine_similarity received a zero-norm vector — refusing to return NaN '
            '(a NaN here would silently corrupt the measured distributions)',
        )
    return math.fsum(x * y for x, y in zip(a, b, strict=True)) / (norm_a * norm_b)


def _order_statistic(ordered: list[float], quantile: float) -> float:
    """Nearest-rank quantile: always a value actually present in the sample.

    Interpolating would produce a threshold no measurement supports, which
    is precisely what derive_bands must not do.
    """
    idx = math.ceil(quantile * len(ordered)) - 1
    return ordered[min(max(idx, 0), len(ordered) - 1)]


def summarize_distribution(scores: Sequence[float]) -> dict[str, Any]:
    """Order-statistic summary of a measured score sample.

    An empty sample reports ``n=0`` with every statistic ``None`` rather
    than ``0.0``, so an empty pair class can never be misread as a measured
    zero.
    """
    if not scores:
        return {
            'n': 0, 'min': None, 'max': None, 'mean': None, 'median': None,
            'p05': None, 'p25': None, 'p75': None, 'p95': None,
        }
    ordered = sorted(float(s) for s in scores)
    return {
        'n': len(ordered),
        'min': ordered[0],
        'max': ordered[-1],
        'mean': math.fsum(ordered) / len(ordered),
        'median': _order_statistic(ordered, 0.50),
        'p05': _order_statistic(ordered, 0.05),
        'p25': _order_statistic(ordered, 0.25),
        'p75': _order_statistic(ordered, 0.75),
        'p95': _order_statistic(ordered, 0.95),
    }


# ---------------------------------------------------------------------------
# Band derivation
# ---------------------------------------------------------------------------

# Machine-readable refusal codes. A caller branches on these rather than
# matching prose; each is emitted with the measurements that produced it.
REASON_EMPTY_CLASS = 'empty_class'
REASON_NOT_SEPARABLE = 'not_separable'
REASON_NO_JUDGE_BAND = 'no_judge_band'


def derive_bands(
    dup_scores: Sequence[float],
    negative_scores: Sequence[float],
) -> tuple[float | None, float | None, str | None]:
    """Derive ``(t_high, t_low, reason)`` from two measured score samples.

    Both thresholds are order statistics of the observed duplicate
    distribution — never interpolated, never defaulted:

    - ``t_high`` is the SMALLEST measured duplicate score that strictly
      exceeds every measured negative. So the deterministic restate band
      admits zero measured false positives by construction, and taking the
      smallest such value keeps that band as wide as the evidence allows.
    - ``t_low`` is the duplicate distribution's lower tail (p05), so
      substantially all curator-confirmed rediscoveries reach at least the
      judge band.

    Every degenerate input returns ``None`` plus a reason code instead of a
    fabricated number. ``None`` means UNCALIBRATED and the triage router
    must fail open to ``stored``.
    """
    if not dup_scores or not negative_scores:
        return None, None, (
            f'{REASON_EMPTY_CLASS}: cannot separate two classes when one is empty '
            f'(n_duplicate={len(dup_scores)}, n_negative={len(negative_scores)})'
        )

    max_negative = max(negative_scores)
    ordered_dups = sorted(float(s) for s in dup_scores)
    separating = [s for s in ordered_dups if s > max_negative]
    if not separating:
        return None, None, (
            f'{REASON_NOT_SEPARABLE}: the highest measured negative ({max_negative}) is at or '
            f'above every measured duplicate (max={ordered_dups[-1]}), so no measured value '
            'separates the classes. Refusing to interpolate a threshold the data does not '
            'support — this outcome is itself the calibration finding.'
        )

    t_high = separating[0]
    t_low = _order_statistic(ordered_dups, 0.05)
    if t_low >= t_high:
        # Perfect separation: t_high IS the duplicate class's own lower
        # bound, so no measured duplicate sits strictly below it and no
        # judge band is derivable from this sample.
        return t_high, None, (
            f'{REASON_NO_JUDGE_BAND}: every measured duplicate ({ordered_dups[0]}..'
            f'{ordered_dups[-1]}) already clears every measured negative (max={max_negative}), '
            f'so t_high={t_high} is the duplicate lower bound and no measured value lies '
            'strictly below it. No judge band is derivable from this sample.'
        )

    return t_high, t_low, None


# ---------------------------------------------------------------------------
# Candidate-retrieval recall
# ---------------------------------------------------------------------------

def compute_recall_at_k(
    retrievals: Sequence[dict[str, Any]],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Measure how often the existing search surfaces the ground-truth canonical.

    Each retrieval carries ``memory_id``, ``canonical_id``,
    ``canonical_present`` and the ranked ``candidates`` search returned.

    Retrievals whose canonical is not in the corpus are listed under
    ``canonical_absent`` and dropped from the denominator. The session's
    duplicates were deleted, so an absent canonical is a CORPUS GAP rather
    than a retrieval failure; scoring it as a miss would understate recall
    and could push T_low lower than the evidence supports.

    An empty denominator reports ``recall=None``, not ``0.0`` — no
    measurement is not a measured zero.
    """
    scorable: list[dict[str, Any]] = []
    absent: list[dict[str, str]] = []
    for item in retrievals:
        if item.get('canonical_present'):
            scorable.append(item)
        else:
            absent.append({
                'memory_id': str(item.get('memory_id')),
                'canonical_id': str(item.get('canonical_id')),
            })

    per_k: list[dict[str, Any]] = []
    for k in ks:
        hits = sum(
            1 for item in scorable
            if item['canonical_id'] in list(item.get('candidates') or [])[:k]
        )
        total = len(scorable)
        per_k.append({
            'k': k,
            'hits': hits,
            'total': total,
            'recall': (hits / total) if total else None,
        })

    return {'per_k': per_k, 'canonical_absent': absent}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Pair classes, in report order. The last two are the NEGATIVE classes: a
# pair from either that reaches the deterministic band is a false positive.
PAIR_CLASSES = ('true_dup', 'unrelated', 'hard_negative')
NEGATIVE_PAIR_CLASSES = ('unrelated', 'hard_negative')


def _band_counts(
    scores: Sequence[float],
    t_high: float | None,
    t_low: float | None,
) -> dict[str, int]:
    """Count a class's scores across the three triage bands.

    ``s >= t_high`` deterministic; ``t_low <= s < t_high`` judge;
    ``s < t_low`` store. A ``None`` t_low means no judge band is derivable,
    so everything below t_high falls to store — the counts still sum to the
    class's n either way.
    """
    deterministic = sum(1 for s in scores if t_high is not None and s >= t_high)
    if t_low is None:
        judge = 0
    else:
        judge = sum(
            1 for s in scores
            if s >= t_low and (t_high is None or s < t_high)
        )
    return {
        'deterministic': deterministic,
        'judge': judge,
        'store': len(scores) - deterministic - judge,
    }


def build_report(
    scores_by_class: Mapping[str, Sequence[float]],
    t_high: float | None,
    t_low: float | None,
    reason: str | None,
    recall: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON-serializable calibration report.

    Its job is to make the deterministic band's false-positive risk
    visible: ``deterministic_band_false_positives`` counts the unrelated
    and hard-negative pairs that would be restated WITHOUT a judge under
    the chosen ``t_high``. True duplicates in that band are the band
    working and are never tallied there.

    Tolerates an uncalibrated run (``t_high=None``): the measured
    distributions plus the refusal reason are exactly what justify the
    refusal, so they are still emitted. The false-positive tally is then
    ``None`` rather than ``0`` — with no deterministic band, ``0`` would
    read as "measured, and safe".
    """
    scores = {name: list(scores_by_class.get(name) or []) for name in PAIR_CLASSES}

    per_band = {
        name: _band_counts(values, t_high, t_low) for name, values in scores.items()
    }
    false_positives = (
        None if t_high is None
        else sum(per_band[name]['deterministic'] for name in NEGATIVE_PAIR_CLASSES)
    )

    run_provenance = dict(provenance)
    run_provenance['pair_counts'] = {name: len(values) for name, values in scores.items()}

    return {
        'chosen_t_high': t_high,
        'chosen_t_low': t_low,
        'reason': reason,
        'deterministic_band_false_positives': false_positives,
        'distributions': {
            name: summarize_distribution(values) for name, values in scores.items()
        },
        'per_band': per_band,
        'recall_at_k': recall,
        'provenance': run_provenance,
    }


# ---------------------------------------------------------------------------
# Config write
# ---------------------------------------------------------------------------

_BLOCK_KEY = 'write_triage:'


def write_triage_config_block(
    yaml_text: str,
    t_high: float | None,
    t_low: float | None,
    report_path: str,
) -> str:
    """Return *yaml_text* with only its ``write_triage:`` block replaced.

    Deliberately a line-oriented surgical edit, NOT a ``yaml.safe_dump``
    round-trip: pyyaml is the only YAML dependency (no ruamel), and dumping
    would strip config.yaml's extensive explanatory comments, which are
    load-bearing for operators.

    Raises when either threshold is ``None`` — an uncalibrated run must
    never put a null threshold into config. The input string is never
    mutated, so a refusal leaves no partial block behind.
    """
    if t_high is None or t_low is None:
        raise ValueError(
            f'refusing to write an uncalibrated threshold into config '
            f'(t_high={t_high!r}, t_low={t_low!r}). Leave the section absent so '
            'triage fails open to `stored`.',
        )

    block = (
        f'{_BLOCK_KEY}\n'
        '  # CALIBRATION OUTPUT — do not hand-edit.\n'
        '  # Derived from measured similarity distributions by\n'
        '  # scripts/calibrate_write_triage.py; both values are order statistics of\n'
        '  # the observed curator-labeled corpus, not chosen constants.\n'
        f'  # Report: {report_path}\n'
        "  # -- records the measured distributions and the deterministic band's\n"
        '  # false-positive count. Re-run the script to change these values.\n'
        f'  t_high: {t_high}\n'
        f'  t_low: {t_low}\n'
        f'  calibration_report_path: {report_path}\n'
    )

    lines = yaml_text.splitlines(keepends=True)
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(_BLOCK_KEY)),
        None,
    )
    if start is None:
        separator = '' if yaml_text.endswith('\n\n') or not yaml_text else '\n'
        return yaml_text + separator + block

    # Scan to the next line that opens a new TOP-LEVEL key (column 0, not a
    # comment and not blank). Bounding the span this way is what keeps a
    # section declared after write_triage from being eaten.
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line[0].isspace() and not line.lstrip().startswith('#'):
            end = i
            break

    # Then walk `end` back: a run of COLUMN-0 comment lines, plus any blank
    # lines before it, is the HEADER for the section that FOLLOWS — that is
    # config.yaml's convention throughout (see the 6-line comment block before
    # `summary_rebuild:`). It must survive the replacement rather than being
    # consumed as block-internal. The column-0 test is deliberately
    # `startswith('#')` and NOT `lstrip().startswith('#')`: an INDENTED comment
    # belongs to the block it sits inside, so the walk stops at it.
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].startswith('#')):
        end -= 1

    # No trailing accumulator: `lines[end:]` already emits every reclaimed line
    # verbatim, so collecting them separately would double-emit them.
    return ''.join(lines[:start]) + block + ''.join(lines[end:])


def render_markdown(report: dict[str, Any]) -> str:
    """Human-readable sibling of the JSON report.

    Carries the per-band table so the deterministic band's false-positive
    risk is readable without parsing JSON.
    """
    lines = [
        '# Write-triage calibration report',
        '',
        f'- `t_high`: `{report["chosen_t_high"]}`',
        f'- `t_low`: `{report["chosen_t_low"]}`',
        f'- deterministic-band false positives: '
        f'**{report["deterministic_band_false_positives"]}**',
    ]
    if report.get('reason'):
        lines.append(f'- refusal reason: `{report["reason"]}`')
    lines += ['', '## Measured distributions', '',
              '| pair class | n | min | p25 | median | p75 | max |',
              '|---|---|---|---|---|---|---|']
    for name in PAIR_CLASSES:
        d = report['distributions'][name]
        lines.append(
            f'| {name} | {d["n"]} | {d["min"]} | {d["p25"]} | {d["median"]} '
            f'| {d["p75"]} | {d["max"]} |',
        )
    lines += ['', '## Per-band counts', '',
              '| pair class | deterministic (s>=t_high) | judge | store |',
              '|---|---|---|---|']
    for name in PAIR_CLASSES:
        b = report['per_band'][name]
        lines.append(f'| {name} | {b["deterministic"]} | {b["judge"]} | {b["store"]} |')
    lines += ['', '## Candidate-retrieval recall', '',
              '| k | hits | total | recall |', '|---|---|---|---|']
    for row in report['recall_at_k'].get('per_k', []):
        lines.append(f'| {row["k"]} | {row["hits"]} | {row["total"]} | {row["recall"]} |')
    absent = report['recall_at_k'].get('canonical_absent') or []
    lines += ['', f'Canonicals absent from the corpus (excluded from the denominator): '
                  f'{len(absent)}', '', '## Provenance', '']
    for key, value in report['provenance'].items():
        lines.append(f'- `{key}`: `{value}`')
    return '\n'.join(lines) + '\n'


def run_calibration(
    records: list[dict[str, Any]],
    embed_fn: Any,
    search_fn: Any,
    report_path: str | Path,
    ks: Sequence[int],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Measure, derive, and write the report. Never writes config.

    ``embed_fn(memory_id, content)`` is called exactly ONCE per distinct
    record — embedding per pair would multiply API cost by O(n).
    ``search_fn(record, k)`` returns ``{candidates, canonical_present}``.

    Neither callable's failure is caught. A swallowed embed error would
    silently shrink the measured population, producing a report whose
    thresholds look fine but were computed on a subset; a swallowed search
    error would be indistinguishable from a genuine recall miss.
    """
    vectors = {r['memory_id']: embed_fn(r['memory_id'], r['content']) for r in records}
    logger.info('Embedded %d distinct record(s)', len(vectors))

    pair_sets = build_pair_sets(records)
    scores_by_class = {
        name: [
            cosine_similarity(vectors[p['a']], vectors[p['b']])
            for p in pair_sets[f'{name}_pairs']
        ]
        for name in PAIR_CLASSES
    }

    negatives = [s for name in NEGATIVE_PAIR_CLASSES for s in scores_by_class[name]]
    t_high, t_low, reason = derive_bands(scores_by_class['true_dup'], negatives)
    if reason:
        logger.warning('Band derivation returned no complete calibration: %s', reason)

    max_k = max(ks) if ks else 0
    retrievals = []
    for record in records:
        if record['label'] == LABEL_CANONICAL:
            continue
        hit = search_fn(record, max_k)
        retrievals.append({
            'memory_id': record['memory_id'],
            'canonical_id': record['cluster_id'],
            'canonical_present': bool(hit.get('canonical_present')),
            'candidates': list(hit.get('candidates') or []),
        })
    recall = compute_recall_at_k(retrievals, list(ks))

    run_provenance = dict(provenance)
    run_provenance.setdefault('record_count', len(records))
    run_provenance.setdefault('cluster_count', len({r['cluster_id'] for r in records}))
    report = build_report(
        scores_by_class=scores_by_class, t_high=t_high, t_low=t_low,
        reason=reason, recall=recall, provenance=run_provenance,
    )

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    report_path.with_suffix('.md').write_text(render_markdown(report))
    logger.info('Wrote %s and its .md sibling', report_path)

    return {
        'report': report, 'scores_by_class': scores_by_class,
        't_high': t_high, 't_low': t_low, 'reason': reason,
        'config_written': False,
    }


# ---------------------------------------------------------------------------
# Live edge / CLI
# ---------------------------------------------------------------------------

def build_embed_fn(config: Any) -> Any:
    """Mirror Mem0's own embedder wiring — model + api_key, NO dimensions.

    See the module docstring: passing a custom dimensionality would move the
    measurement into a different metric space than the guard's
    relevance_score, making the derived thresholds inapplicable.
    """
    from openai import OpenAI  # noqa: PLC0415

    embedder = config.embedder
    openai_provider = embedder.providers.openai
    api_key = openai_provider.api_key if openai_provider is not None else None
    client = (
        OpenAI(api_key=api_key, base_url=openai_provider.api_url)
        if openai_provider is not None
        else OpenAI()
    )

    def embed(memory_id: str, content: str) -> list[float]:
        response = client.embeddings.create(model=embedder.model, input=content)
        return list(response.data[0].embedding)

    return embed


async def _run(args: Any) -> int:
    import os  # noqa: PLC0415

    from fused_memory.config.schema import FusedMemoryConfig  # noqa: PLC0415
    from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.config:
        os.environ['CONFIG_PATH'] = str(args.config)

    config = FusedMemoryConfig()
    memory = MemoryService(config)
    await memory.initialize()
    try:
        records = load_fixture(args.fixture)
        logger.info('Loaded %d labeled record(s) from %s', len(records), args.fixture)

        embed_fn = build_embed_fn(config)

        async def _search_async(record: dict, k: int) -> dict:
            # stores=['mem0'] is load-bearing, not a narrowing convenience.
            # An unscoped MemoryService.search runs the read ROUTER first, and
            # for this corpus the router sends every fixture query to graphiti
            # alone — so the top-k comes back full of graphiti edge UUIDs,
            # which can never equal a mem0 memory_id, and recall reads ~0 for
            # reasons that have nothing to do with retrieval quality.
            # Write triage's candidate set is the mem0/Qdrant neighbourhood
            # (near_duplicate_guard searches mem0; the 21 canonicals live in
            # the fused_reify Qdrant collection), so mem0 is the store whose
            # recall the bands actually depend on. Measured on a 6-record
            # probe: 0/6 canonicals in top-10 unscoped vs 4/6 scoped to mem0.
            results = await memory.search(
                query=record['content'], project_id=args.project_id, limit=k,
                stores=['mem0'],
            )
            rows = results.get('results', results) if isinstance(results, dict) else results
            # MemoryService.search returns pydantic MemoryResult rows, not dicts.
            candidates = [
                str(r['id'] if isinstance(r, dict) else r.id) for r in rows or []
            ]
            canonical = await memory.get_memory_by_id(
                args.project_id, record['cluster_id'],
            )
            if canonical is None:
                present = False
            elif isinstance(canonical, dict):
                present = bool(canonical.get('found', True))
            else:
                present = bool(getattr(canonical, 'found', True))
            return {'candidates': candidates, 'canonical_present': present}

        # Pre-resolve retrievals on this loop, then hand run_calibration a
        # plain lookup — keeps the orchestrator fully synchronous and
        # testable while the I/O stays async here.
        max_k = max(args.k)
        prefetched: dict[str, dict] = {}
        for record in records:
            if record['label'] != LABEL_CANONICAL:
                prefetched[record['memory_id']] = await _search_async(record, max_k)

        result = run_calibration(
            records=records,
            embed_fn=embed_fn,
            search_fn=lambda record, k: prefetched[record['memory_id']],
            report_path=args.report_path,
            ks=args.k,
            provenance={
                'fixture_path': package_relative(args.fixture),
                'project_id': args.project_id,
                'embedder_model': config.embedder.model,
                'embedder_dimensions': getattr(config.embedder, 'dimensions', None),
                'search_stores': 'mem0 (MemoryService.search, stores=[mem0])',
                'search_categories': 'all',
            },
        )

        print(json.dumps(result['report'], indent=2))
        logger.info(
            't_high=%s t_low=%s deterministic-band false positives=%s',
            result['t_high'], result['t_low'],
            result['report']['deterministic_band_false_positives'],
        )

        if not args.write_config:
            logger.info('Report only — config unchanged. Use --write-config to commit.')
            return 0
        if result['t_high'] is None or result['t_low'] is None:
            logger.error(
                'ABORT: no complete calibration was derived (%s) — refusing to write '
                'config. Leaving write_triage uncalibrated means triage fails open '
                'to `stored`, which is the correct behaviour for this outcome.',
                result['reason'],
            )
            return 1

        config_path = Path(args.config) if args.config else (
            Path(__file__).parent.parent / 'config' / 'config.yaml'
        )
        updated = write_triage_config_block(
            config_path.read_text(), result['t_high'], result['t_low'],
            package_relative(args.report_path),
        )
        config_path.write_text(updated)
        logger.info('Wrote write_triage block to %s', config_path)
        return 0
    finally:
        await memory.close()


def main() -> int:
    import argparse  # noqa: PLC0415
    import asyncio  # noqa: PLC0415

    repo = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', default=str(
        repo / 'tests' / 'fixtures' / 'write_triage_calibration.jsonl'))
    parser.add_argument('--project-id', dest='project_id', default='reify')
    parser.add_argument('--k', type=int, action='append', default=None,
                        help='recall@k value (repeatable; default 1 3 5 10)')
    parser.add_argument('--report-path', dest='report_path', default=str(
        repo / 'calibration' / 'write_triage_calibration_report.json'))
    parser.add_argument('--config', default=None,
                        help='Path to fused-memory config file (sets CONFIG_PATH)')
    parser.add_argument('--write-config', dest='write_config', action='store_true',
                        help='Write the derived thresholds into config.yaml '
                             '(default: report only)')
    args = parser.parse_args()
    if not args.k:
        args.k = [1, 3, 5, 10]
    return asyncio.run(_run(args))


if __name__ == '__main__':
    raise SystemExit(main())
