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

import os
import re
from dataclasses import dataclass
from typing import Any

from shared.prompt_artifact import ArtifactProvenance, PromptArtifactStore

from orchestrator.agents.roles import REVIEWER_COMPREHENSIVE
from orchestrator.evals.prompt_opt.engine import run_optimization_loop
from orchestrator.evals.prompt_opt.scorer import ProposeFn, RolloutFn, Scorer
from orchestrator.evals.prompt_opt.variance import AcceptanceRecord

__all__ = [
    'AcceptanceReport',
    'FixtureItem',
    'SmokeReviewerScorer',
    'build_fixture_corpus',
    'format_markdown',
    'run_acceptance_smoke',
]

# The fixed on-disk CONTRACT/heuristics separator — mirrors the private
# shared.prompt_artifact._SEPARATOR (compose_prompt emits contract + this +
# heuristics). Duplicated here so the byte-level CONTRACT-untouched proof can
# assert the exact prefix without reaching into a private name.
_CONTRACT_SEPARATOR = '\n\n---\n\n'

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


# ---------------------------------------------------------------------------
# The end-to-end acceptance harness: run the REAL loop over the fixture corpus
# with the hermetic seams, pin the loop's final heuristics through the REAL
# loader, resolve it back, and prove the CONTRACT prefix is byte-identical.
# ---------------------------------------------------------------------------

# Loop knobs — mirror test_prompt_opt_engine.py's proven hermetic dry-run
# (n_repeats=3, max_steps=2, minibatch_size=2). git_sha/date are caller-supplied
# hermetic labels (the engine never computes them), so the run is deterministic.
_DEFAULT_SPLIT_SEED = 2498
_DEFAULT_N_REPEATS = 3
_DEFAULT_MAX_STEPS = 2
_DEFAULT_MINIBATCH_SIZE = 2
_SMOKE_GIT_SHA = 'smoke'
_SMOKE_DATE = '2026-07-20'

# The bounded improving edit the default propose_fn appends, carrying a
# SMOKE_QUALITY sentinel above the baseline's implicit 0.50 so the paired delta
# (~0.20) strictly exceeds the ~0.06 jitter band and the gate ACCEPTS it.
_IMPROVING_EDIT = '6. Cite an explicit `file:line` location in every issue you report.'
_SMOKE_CANDIDATE_QUALITY = 0.70


async def _default_improving_propose_fn(
    current_heuristics: str,
    scored_minibatch: list[Any],
    rejected_buffer: list[str],
    *,
    max_edits: int,
    optimizer_model: str,
) -> str:
    """The default (benign) scripted ProposeFn: one bounded improving edit.

    Appends a bounded edit plus a ``SMOKE_QUALITY=0.70`` sentinel (vs the
    baseline's implicit 0.50), so the candidate's paired selection-split delta
    (~0.20) strictly exceeds the ~0.06 jitter band — the acceptance gate
    ACCEPTS it. Signature matches :data:`ProposeFn` exactly and returns the
    same heuristics-block text contract :func:`propose_heuristics_edit`
    produces, so it is a drop-in stand-in for the real optimizer (unit-tested
    separately) and the smoke makes no LLM call. Never receives
    ``spec.contract`` (D-3).
    """
    return f'{current_heuristics}\n{_IMPROVING_EDIT}\nSMOKE_QUALITY={_SMOKE_CANDIDATE_QUALITY}'


@dataclass(frozen=True)
class AcceptanceReport:
    """The full result of one :func:`run_acceptance_smoke` run.

    Carries the held-out TEST verdict, the acceptance-gate outcome, the 8-field
    provenance sidecar, the loader-resolved prompt + its source, the byte-level
    CONTRACT-untouched proof, and the pin key an operator (or the round-trip
    test) uses to reproduce the ship -> rollback (unpin) path.
    """

    test_score: float
    baseline_heuristics: str
    final_heuristics: str
    accepted_count: int
    rejected_count: int
    accept_records: list[AcceptanceRecord]
    provenance: ArtifactProvenance
    resolved_text: str
    resolved_source: str
    contract_prefix_intact: bool
    # The exact byte string proven to be the untouched CONTRACT prefix of
    # resolved_text (contract_prefix_intact == resolved_text.startswith(
    # verified_contract_marker + separator)). Recorded explicitly so the proof
    # is self-contained on the report.
    verified_contract_marker: str
    pin_key: tuple[str, str, str]
    executor_model: str
    optimizer_model: str


async def run_acceptance_smoke(
    *,
    store_root: str | os.PathLike[str],
    split_seed: int = _DEFAULT_SPLIT_SEED,
    executor_model: str = _SMOKE_EXECUTOR_MODEL,
    optimizer_model: str = _SMOKE_OPTIMIZER_MODEL,
    n_repeats: int = _DEFAULT_N_REPEATS,
    max_steps: int = _DEFAULT_MAX_STEPS,
    minibatch_size: int = _DEFAULT_MINIBATCH_SIZE,
    git_sha: str = _SMOKE_GIT_SHA,
    date: str = _SMOKE_DATE,
    corpus: list[FixtureItem] | None = None,
    rollout_fn: RolloutFn = _smoke_rollout_fn,
    scorer: Scorer | None = None,
    propose_fn: ProposeFn = _default_improving_propose_fn,
) -> AcceptanceReport:
    """Run the hermetic end-to-end Tier-1 prompt-opt acceptance smoke.

    Uses the REAL ``REVIEWER_COMPREHENSIVE.prompt_spec`` (real reviewer
    CONTRACT/HEURISTICS split) as the starting prompt, runs the REAL
    :func:`run_optimization_loop` with the hermetic seams (the executor-model
    rollout_fn, the verdict-vs-gold :class:`SmokeReviewerScorer`, and the
    default improving propose_fn), pins the loop's ``final_heuristics`` + the
    loop's provenance into a :class:`PromptArtifactStore` at *store_root*,
    resolves it back (``source='artifact'``), and proves the composed prompt's
    CONTRACT prefix is byte-identical to ``spec.contract``.

    The pin is intentionally LEFT in place at *store_root* so a caller (or the
    loader round-trip test) can reproduce the operator ship -> watch ->
    rollback (:meth:`PromptArtifactStore.unpin`) path from the returned
    :class:`AcceptanceReport.pin_key`. *store_root* must therefore be an
    isolated/throwaway root — never a production artifacts root — since the
    pinned heuristics carry the hermetic ``SMOKE_QUALITY`` sentinel.
    """
    role = REVIEWER_COMPREHENSIVE
    spec = role.prompt_spec
    if spec is None:  # pragma: no cover - guards a T2 regression, not a runtime branch
        raise AssertionError('REVIEWER_COMPREHENSIVE.prompt_spec must not be None for the smoke')
    harness_version = role.prompt_harness_version
    if harness_version is None:  # pragma: no cover - same T2 guard
        raise AssertionError('REVIEWER_COMPREHENSIVE.prompt_harness_version must not be None')

    if corpus is None:
        corpus = build_fixture_corpus()
    if scorer is None:
        scorer = SmokeReviewerScorer()

    result = await run_optimization_loop(
        spec,
        corpus,
        scorer,
        executor_model,
        optimizer_model,
        rollout_fn=rollout_fn,
        propose_fn=propose_fn,
        n_repeats=n_repeats,
        max_steps=max_steps,
        split_seed=split_seed,
        minibatch_size=minibatch_size,
        git_sha=git_sha,
        date=date,
        harness_version=harness_version,
    )

    # Pin the loop's final heuristics + provenance, then resolve it back through
    # the REAL loader (the exact operator ship path) and prove CONTRACT untouched.
    store = PromptArtifactStore(store_root)
    store.pin(
        spec.prompt_id,
        executor_model,
        harness_version,
        heuristics=result.final_heuristics,
        provenance=result.provenance,
    )
    resolved = store.resolve(spec, executor_model, harness_version)
    contract_prefix_intact = resolved.text.startswith(spec.contract + _CONTRACT_SEPARATOR)

    accepted = [record for record in result.accept_records if record.accepted]
    rejected = [record for record in result.accept_records if not record.accepted]

    return AcceptanceReport(
        test_score=result.test_score,
        baseline_heuristics=spec.baseline_heuristics,
        final_heuristics=result.final_heuristics,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        accept_records=list(result.accept_records),
        provenance=result.provenance,
        resolved_text=resolved.text,
        resolved_source=resolved.source,
        contract_prefix_intact=contract_prefix_intact,
        verified_contract_marker=spec.contract,
        pin_key=(spec.prompt_id, executor_model, harness_version),
        executor_model=executor_model,
        optimizer_model=optimizer_model,
    )


def format_markdown(report: AcceptanceReport) -> str:
    """Render *report* as an operator-readable markdown block (mirrors
    reviewer_trial/report.py's format_markdown convention)."""
    lines: list[str] = []
    lines.append('# Tier-1 Prompt-Opt Acceptance Smoke\n')
    lines.append(
        'Hermetic end-to-end run of the reviewer prompt-opt stack '
        '(loader -> HEURISTICS -> loop -> scorer -> report) on a <=3-diff '
        'synthetic fixture corpus — no real LLM, DB, or network.\n'
    )

    lines.append('## Held-out TEST verdict\n')
    lines.append(f'- **held_out_TEST_score**: {report.test_score:.4f}')
    lines.append(
        f'- executor_model: `{report.executor_model}`   '
        f'optimizer_model: `{report.optimizer_model}`\n'
    )

    lines.append('## Acceptance gate\n')
    lines.append(
        f'- accepted candidates: **{report.accepted_count}**   '
        f'rejected: **{report.rejected_count}**'
    )
    for step_index, record in enumerate(report.accept_records):
        tag = 'ACCEPT' if record.accepted else 'reject'
        lines.append(
            f'  - step {step_index}: [{tag}] paired_delta={record.paired_delta:.6f} '
            f'band={record.band:.6f}'
        )
    lines.append('')

    prov = report.provenance
    lines.append('## Provenance sidecar (8 fields)\n')
    lines.append(f'- optimizer_model: `{prov.optimizer_model}`')
    lines.append(f'- corpus_hash: `{prov.corpus_hash}`')
    lines.append(f'- split_seed: {prov.split_seed}')
    lines.append(f'- held_out_TEST_score: {prov.held_out_TEST_score:.4f}')
    lines.append(f'- accept_delta: {prov.accept_delta}')
    lines.append(f'- git_sha: `{prov.git_sha}`')
    lines.append(f'- date: {prov.date}')
    lines.append(f'- harness_version: `{prov.harness_version}`\n')

    lines.append('## CONTRACT untouched\n')
    proof = 'PASS' if report.contract_prefix_intact else 'FAIL'
    lines.append(
        f'- contract prefix intact: **{proof}** '
        f'(resolved source = `{report.resolved_source}`)'
    )
    lines.append(
        '- the loader-resolved prompt begins with the reviewer CONTRACT '
        'byte-for-byte, then the separator, then the heuristics block.\n'
    )

    prompt_id, exec_model, harness = report.pin_key
    lines.append('## Rollback lever\n')
    lines.append(f'- pin key: `({prompt_id!r}, {exec_model!r}, {harness!r})`')
    lines.append(
        f'- rollback: `PromptArtifactStore(root).unpin({prompt_id!r}, '
        f'{exec_model!r}, {harness!r})` restores the in-code reviewer baseline.'
    )
    return '\n'.join(lines)
