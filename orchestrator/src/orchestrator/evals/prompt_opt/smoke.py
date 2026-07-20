"""Hermetic end-to-end acceptance smoke for the Tier-1 prompt-opt stack (T7).

See plans/tier1-prompt-optimization-prd.md T7. Proves the full stack —
loader -> reviewer HEURISTICS -> loop engine -> scorer -> report — works
end-to-end on a <=3-diff synthetic fixture corpus WITHOUT a real
($300-800) run, so a green smoke gates the operator's confidence to launch
the real reviewer/curator loops (runbook §8).

Hermetic by dependency injection: run the REAL stack (the real
``REVIEWER_COMPREHENSIVE.prompt_spec`` contract/heuristics split, the real
:func:`run_optimization_loop` + variance gate + splits, the real
:class:`PromptArtifactStore` loader + :func:`compose_prompt`) while faking
ONLY the three LLM-touching seams deterministically — the ``rollout_fn``
(no ``invoke_agent``; asserts ``model == executor_model``), the
:class:`SmokeReviewerScorer` (verdict-vs-gold agreement + a bounded
occurrence-keyed jitter cycle so the variance band is positive), and the
``propose_fn`` (a scripted improving edit). No ``invoke_agent``, no DB, no
network.

'<=3-diff' means <=3 DISTINCT diff archetypes replicated into a >=10-item
corpus: :func:`run_optimization_loop` calls :func:`split_corpus` with the
FIXED default 2:1:7 ratios and RAISES ``ValueError`` before any rollout when
the selection or test split is empty (``selection_n = n // 10``), so a
held-out TEST verdict via the REAL engine is only achievable with >=10 items.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = ['FixtureItem', 'SmokeReviewerScorer', 'build_fixture_corpus']

# Hermetic labels for the two models the loop threads through (never real API
# ids that would bill): the reviewer executor and the frontier optimizer.
_SMOKE_EXECUTOR_MODEL = 'opus'
_SMOKE_OPTIMIZER_MODEL = 'frontier-opt'


# ---------------------------------------------------------------------------
# Fixture corpus: <=3 DISTINCT synthetic diff archetypes, replicated to >=10
# distinct-id items so the REAL engine's fixed 2:1:7 split is non-empty in all
# three partitions. These are inline module constants — no on-disk fixtures.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureItem:
    """One synthetic corpus item: a diff plus its gold reviewer verdict.

    Frozen (so it is hashable and has a deterministic ``repr`` —
    :func:`run_optimization_loop`'s ``_stable_corpus_hash`` hashes
    ``repr(item)``, and :func:`split_corpus` shuffles a list of these).
    ``gold_severity`` is ``None`` exactly when ``gold_verdict == 'PASS'``.
    """

    item_id: str
    diff: str
    gold_verdict: str  # 'PASS' | 'ISSUES_FOUND'
    gold_severity: str | None  # 'blocking' | 'suggestion' | None (None iff PASS)


# Archetype A — a clear blocking bug (a missing `await` makes a coroutine be
# compared to a number and a debit silently never happen) -> ISSUES_FOUND/blocking.
_DIFF_BLOCKING = """\
--- a/svc/payments.py
+++ b/svc/payments.py
@@ -8,9 +8,9 @@ async def charge(account, amount):
-    balance = await account.get_balance()
+    balance = account.get_balance()  # compares a coroutine object to a number
     if balance < amount:
         raise InsufficientFunds()
-    await account.debit(amount)
+    account.debit(amount)  # fire-and-forget: the debit coroutine is never awaited
     return Receipt(account.id, amount)
"""

# Archetype B — a clean, behaviour-preserving docs tweak -> PASS.
_DIFF_CLEAN = """\
--- a/README.md
+++ b/README.md
@@ -1,3 +1,3 @@
 # Widget
-A small widget library.
+A small, well-tested widget library.
"""

# Archetype C — a borderline style/naming diff (terse single-letter name, a
# lingering TODO) that is not broken -> ISSUES_FOUND/suggestion.
_DIFF_SUGGESTION = """\
--- a/util/strings.py
+++ b/util/strings.py
@@ -1,0 +1,4 @@
+def to_snake(s):
+    # TODO: handle unicode; `s` is a terse single-letter parameter name
+    parts = ['_' + c.lower() if c.isupper() else c for c in s]
+    return ''.join(parts).lstrip('_')
"""

# The <=3 distinct archetypes: (diff text, gold verdict, gold severity).
_ARCHETYPES: tuple[tuple[str, str, str | None], ...] = (
    (_DIFF_BLOCKING, 'ISSUES_FOUND', 'blocking'),
    (_DIFF_CLEAN, 'PASS', None),
    (_DIFF_SUGGESTION, 'ISSUES_FOUND', 'suggestion'),
)


def build_fixture_corpus(*, replicas: int = 4) -> list[FixtureItem]:
    """Build the hermetic reviewer fixture corpus for the T7 smoke.

    Replicates the <=3 distinct :data:`_ARCHETYPES` diff texts *replicas*
    times each into distinct-``item_id`` items. With the default
    ``replicas=4`` this yields ``3 x 4 = 12`` items, which the engine's fixed
    2:1:7 split partitions into a non-empty train (``12*2//10 = 2``),
    selection (``12*1//10 = 1``), and test (``12 - 2 - 1 = 9``) — the
    minimum shape that lets the REAL loop produce a held-out TEST verdict
    (a corpus < 10 items has an empty selection split, which
    :func:`run_optimization_loop` rejects before any rollout).

    The item ids are distinct per replica while the diff TEXTS are reused, so
    the corpus honors '<=3 distinct diffs' yet is still loop-runnable.
    """
    items: list[FixtureItem] = []
    for replica in range(replicas):
        for archetype_index, (diff, gold_verdict, gold_severity) in enumerate(_ARCHETYPES):
            items.append(
                FixtureItem(
                    item_id=f'smoke-a{archetype_index}-r{replica}',
                    diff=diff,
                    gold_verdict=gold_verdict,
                    gold_severity=gold_severity,
                )
            )
    return items


# ---------------------------------------------------------------------------
# Hermetic seams: the executor-model-asserting rollout_fn and the
# verdict-vs-gold Scorer. Both are deterministic and touch no invoke_agent /
# DB / network. They mirror test_prompt_opt_engine.py's PROVEN fixture shape
# (a quality mark + a +-0.03 occurrence-keyed jitter cycle) so the acceptance
# arithmetic (baseline 0.50 -> candidate 0.70, band ~0.06, delta ~0.20 > band
# -> accepted) is grounded in an already-passing test, not a guessed threshold.
# ---------------------------------------------------------------------------

# The optimizer emits an improved heuristics block carrying this sentinel; the
# rollout/scorer parse it out of the composed prompt. Default 0.50 when absent,
# so the REAL reviewer baseline (which has no sentinel) reads as baseline quality.
_SMOKE_QUALITY_RE = re.compile(r'SMOKE_QUALITY=([0-9.]+)')
_DEFAULT_QUALITY = 0.50
# +-0.03 jitter cycle (D-5: the reviewer scorer is a noisy matcher) — bounded,
# deterministic, and cyclic so a positive repeatability band is measured.
_JITTER_CYCLE = (0.03, -0.03)
# Scale applied when the reviewer's REPORTED verdict disagrees with gold —
# strictly < 1 so agreement always scores strictly higher at equal quality.
_DISAGREEMENT_SCALE = 0.5
# Parses the reviewer's reported gold token out of a rollout string emitted by
# _smoke_rollout_fn ("...::gold=<verdict>/<severity>::<composed_prompt>").
_ROLLOUT_GOLD_RE = re.compile(r'::gold=(.*?)::')


def _quality_of(text: str, default: float = _DEFAULT_QUALITY) -> float:
    """The SMOKE_QUALITY sentinel embedded in *text*, or *default* when absent."""
    match = _SMOKE_QUALITY_RE.search(text)
    if match is None:
        return default
    return float(match.group(1))


def _clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


async def _smoke_rollout_fn(composed_prompt: str, item: Any, model: str) -> str:
    """The hermetic RolloutFn: always executor_model, echoes (item gold, composed).

    Standing in for a real executor call, the 'rollout' deterministically
    encodes the item's gold verdict (what a perfect reviewer would report) and
    the composed prompt (which carries the heuristics SMOKE_QUALITY) — enough
    for :class:`SmokeReviewerScorer` to recover both the agreement signal and a
    stable per-(item, heuristics) occurrence key. Never touches invoke_agent.

    Asserts ``model == _SMOKE_EXECUTOR_MODEL`` so every acceptance decision is
    structurally guaranteed to be scored on the ACTUAL executor model, never
    the optimizer (the engine always calls this with ``executor_model``).
    """
    assert model == _SMOKE_EXECUTOR_MODEL, (
        f'_smoke_rollout_fn must ALWAYS be called with the executor model '
        f'{_SMOKE_EXECUTOR_MODEL!r}, got {model!r}'
    )
    return (
        f'rollout::{item.item_id}::gold={item.gold_verdict}/{item.gold_severity}'
        f'::{composed_prompt}'
    )


class SmokeReviewerScorer:
    """Deterministic Scorer: verdict-vs-gold agreement scaled by SMOKE_QUALITY
    plus a bounded occurrence-keyed jitter cycle (implements the Scorer Protocol).

    ``score(item, rollout)`` grades the executor OUTPUT the way the real
    reviewer scorer will — did the reviewer's REPORTED verdict (parsed from the
    rollout) match the item's TRUE gold? — scaled by the heuristics quality
    mark, then perturbed by a small reproducible jitter so repeated scoring of
    identical inputs disagrees by a bounded, positive amount (a positive
    repeatability band). Every distinct rollout string gets its own occurrence
    counter, so repeat N of that exact pair always draws the same jitter —
    fully deterministic, no real randomness (mirrors _FakeScorer).
    """

    def __init__(self) -> None:
        self._occurrence_counts: dict[str, int] = {}

    async def score(self, item: Any, rollout: Any) -> float:
        rollout_text = str(rollout)
        # heuristics quality (0.50 default when the sentinel is absent)
        quality = _quality_of(rollout_text)
        # the reviewer's REPORTED verdict vs the item's TRUE gold
        reported = self._reported_gold(rollout_text)
        truth = f'{item.gold_verdict}/{item.gold_severity}'
        agreement_scale = 1.0 if reported == truth else _DISAGREEMENT_SCALE
        base = quality * agreement_scale
        # bounded, reproducible per-(item, heuristics) jitter cycle
        count = self._occurrence_counts.get(rollout_text, 0)
        self._occurrence_counts[rollout_text] = count + 1
        jitter = _JITTER_CYCLE[count % len(_JITTER_CYCLE)]
        return _clamp_unit(base + jitter)

    @staticmethod
    def _reported_gold(rollout_text: str) -> str | None:
        match = _ROLLOUT_GOLD_RE.search(rollout_text)
        return match.group(1) if match is not None else None
