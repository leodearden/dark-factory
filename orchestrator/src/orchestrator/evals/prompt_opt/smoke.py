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

from dataclasses import dataclass

__all__ = ['FixtureItem', 'build_fixture_corpus']


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
