#!/usr/bin/env python3
"""E4 staleness sweep — is the corpus still surfacing things it superseded?

``docs/prds/memory-eval-program.md`` §5 leaf γ. Where leaf β
(``memory_eval_retrieval_probe.py``, E1) asks whether retrieval returns the
RIGHT thing for a committed registry of topics, this leaf asks the corpus
about itself: which of the supersession/parent/correction pointers it carries
still resolve, whether a superseded entry still outranks the entry that
replaced it, and whether entries assert live task state for tasks that have
since gone terminal.

**What it measures** (four metrics across three families, all owned by this
leaf):

======================================  ==========  ==================
metric_id                               kind        direction
======================================  ==========  ==================
``superseded-still-surfacing``          count       higher_is_worse
``dangling-pointers``                   count       higher_is_worse
``successor-pointer-present``           tripwire    (rule (a) is already
                                                    directional)
``task-terminal-staleness``             count       higher_is_worse
======================================  ==========  ==================

``dangling-pointers`` and ``successor-pointer-present`` are the two spellings
leaf β's docstring and leaf α's committed exemplars reserve for THIS leaf, and
they are not redundant with each other: the count feeds α's Poisson
count-shift trend (is the corpus accumulating dangling pointers?), the
tripwire feeds α's grandfathered structural rule with the ratchet (did THIS
supersession edge newly break, or did a previously-broken one get fixed?). One
aggregate count cannot express per-edge grandfathering; one tripwire cannot
express a trend over ``parent_id``/``corrects`` targets that have no stable
per-item identity.

β's ``superseded-above-successor`` is NOT reused here. That metric is
registry-declared-pair shaped and lives under β's ``e1-retrieval-health``;
this leaf's family (1) is corpus-DISCOVERED from live ``supersedes`` metadata.
Different populations, different exposure, different eval_id.

**This script never writes to the live corpus and never evaluates a limit.**

Both halves are load-bearing:

- *Never writes.* There is no ``--apply`` band, no delete/add/update call and
  no write path anywhere in this module. D8's read-only runner pattern
  (``audit_duplicate_memories.py`` ``_run``) is copied minus every mutation.
  The guarantee is asserted as BEHAVIOUR in the tests — the sweep is driven
  against a service double whose every write method raises — rather than
  merely claimed in this docstring.
- *Never evaluates a limit.* Per G6/M2 every threshold, tolerance,
  grandfather set and alarm lives in leaf α's limits evaluator. No pass rate,
  bound or verdict appears in this script or in any of its tests.
  ``--scan-limit`` is a resource cap whose firing is DISCLOSED into the
  artifact, not a tuned bound.

**Schema home (D2).** Every artifact model, validator, path helper, stamp
format and atomic writer comes from :mod:`shared.memory_eval_metrics`. Nothing
in that contract is re-declared here.

**Pointer parsing (D7 / INV-5).** All three keys in :data:`POINTER_KEYS` are
parsed through the ONE imported
:func:`fused_memory.memory_metadata.normalize_supersedes`. ``parent_id`` and
``corrects`` carry the identical ``None``/scalar/list-of-UUIDs ambiguity
``supersedes`` does, and the failure mode is the one task 3112 recorded: a
bare ``for target in value`` over a 36-character UUID *string* iterates it into
36 single characters, none of which resolve, manufacturing a systematic false
dangling-pointer report. A second local normalizer for the other two keys
would be exactly the INV-5 violation D7 forbids while re-introducing the bug
that helper was written to prevent.

**Zero exposure is ABSENT, not zero (D1).** A family that measured nothing
emits NO metric rather than a ``value=0 / n=0`` datapoint — a fabricated
"nothing wrong here" entering leaf α's baseline window is worse than a gap in
it. ``parent_id`` makes this live rather than hypothetical: it is in
``RESERVED_VOCABULARY_KEYS`` but the metadata census measures zero live
records carrying it. The report names every family explicitly so an absence
can never be misread as health.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger('memory_eval_staleness_sweep')

# ---------------------------------------------------------------------------
# Pinned M1 contract vocabulary
#
# These spellings are COPIED from leaf α's committed consumer-side exemplars
# and from leaf β's docstring, which reserves `dangling-pointers` and
# `successor-pointer-present` for this leaf by name. The limits evaluator
# joins a run to its baseline window BY metric_id, so a different spelling
# would not fail loudly — it would make the metric invisible to the evaluator
# and to the dashboard, which is strictly worse than a crash.
# ---------------------------------------------------------------------------

EVAL_ID = 'e4-staleness-sweep'
"""Also the artifact directory segment (``<root>/<eval_id>/metrics-<STAMP>.json``).

This leaf's OWN eval_id, deliberately not β's ``e1-retrieval-health``, even
though the PRD describes E1+E4 as "one scheduled retrieval-health runner".
That is a SCHEDULING statement (leaf ε invokes both), not a one-artifact one:
``write_metric_series`` atomically OVERWRITES
``<root>/<eval_id>/metrics-<STAMP>.json``, and ``RUN_STAMP_ENV_VAR`` exists
precisely so several runners in one logical run share a stamp — so a shared
eval_id would make this leaf silently clobber β's artifact on every scheduled
run, and vice versa. Separate ids also keep α's baseline windows independent,
since the evaluator joins a window BY metric_id WITHIN one eval_id series.
Overridable via ``--eval-id`` (δ's precedent).
"""

METRIC_SUPERSEDED_STILL_SURFACING = 'superseded-still-surfacing'
"""Count of superseded entries that outranked their successor. n = comparable pairs."""

METRIC_DANGLING_POINTERS = 'dangling-pointers'
"""Count of pointer targets that do not resolve. n = pointers examined."""

METRIC_SUCCESSOR_POINTER_PRESENT = 'successor-pointer-present'
"""Tripwire (M2 rule a). One item per ``supersedes`` edge, keyed by content."""

METRIC_TASK_TERMINAL_STALENESS = 'task-terminal-staleness'
"""Count of entries asserting live state for a terminal task. n = entries
referencing a terminal task at all."""

POINTER_KEYS: tuple[str, ...] = ('supersedes', 'parent_id', 'corrects')
"""The metadata keys whose values are memory-id pointers.

All three go through :func:`normalize_supersedes` — see the module docstring
for why a second parser for the other two would re-introduce 3112's bug.
``parent_id`` is reserved vocabulary with zero live population today; it is
swept anyway so the first genuine use is measured rather than discovered.
"""

TRIPWIRE_ITEM_PREFIX = 's-'
"""``TripwireItem.item_key`` shape. A STORED key (α's grandfather set persists
it), not a display string."""

_DEFAULT_METRICS_ROOT = str(Path(__file__).resolve().parent.parent / 'data' / 'memory-evals')
"""``fused-memory/data/memory-evals`` (M1 §3), resolved off THIS file.

Not off the cwd: a scheduled run's working directory is not guaranteed, and a
relative default would scatter artifacts wherever the scheduler happened to
start — invisible to the limits evaluator, which scans one root.
"""

_WHITESPACE_RE = re.compile(r'\s+')
