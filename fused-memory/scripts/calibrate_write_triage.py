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
