#!/usr/bin/env python3
"""Offline haiku-vs-sonnet replay-agreement trial for the ``module_tagger`` role.

Task 2540 (Routing κ). A self-contained, decide-and-act pilot: replay historical
DONE tasks (which carry ground-truth ``metadata.files`` = the actual merge-diff
files) through BOTH haiku and a fresh sonnet run at the production call site,
score each model's file predictions vs ground truth (precision/recall/F1) and
haiku-vs-sonnet agreement (Jaccard), frontier-adjudicate (opus) the
disagreements, and compute a pass/marginal/fail verdict via documented
thresholds. On a clear PASS the trial flips ``module_tagger`` to haiku via a
hot-reload of ``dark-factory-orchestrator.yaml``; otherwise it escalates with the
measured numbers. Both branches always commit the markdown report.

The replay reuses byte-identical production inputs by importing the shared
``orchestrator.module_tagger_prompt`` builder used by ``harness._tag_task_modules``
(single source of truth — no drifting hand-copy).

Deterministic core (set_scores, build_replay_input, faithful_prompt_for,
parse_predictions, adjudication, decide, render_report, run_trial) is unit-tested
in tests/scripts/test_trial_module_tagger_haiku.py with a FAKE invoke_fn over
fixtures. The live model run + config flip are impl-only (live models +
conditional config).
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.harness import _extract_tagger_entries
from orchestrator.module_charter import sanitize_files_for_persist
from orchestrator.module_tagger_prompt import (
    TAGGER_SCHEMA,
    TAGGER_SYSTEM_PROMPT,
    build_tagger_prompt,
)


def set_scores(predicted: list, gold: list) -> dict[str, float]:
    """Score a predicted file set against a gold file set.

    Both sides are first passed through ``sanitize_files_for_persist`` (the
    production WRITE-path normalizer), so scoring happens on the same
    file-level sets production actually locks on — directory-shaped entries are
    stripped symmetrically and never count.

    Returns ``{precision, recall, f1, jaccard}`` over the sanitized sets:

    - ``precision`` = |P∩G| / |P|  (fraction of predictions that were correct)
    - ``recall``    = |P∩G| / |G|  (fraction of gold files that were predicted)
    - ``f1``        = harmonic mean of precision and recall
    - ``jaccard``   = |P∩G| / |P∪G|  (symmetric overlap)

    Edge cases are total (no ZeroDivision): when a denominator would be zero,
    the metric is 1.0 iff BOTH sets are empty (vacuously perfect) else 0.0.
    """
    p = set(sanitize_files_for_persist(list(predicted)))
    g = set(sanitize_files_for_persist(list(gold)))
    inter = len(p & g)
    union = len(p | g)

    precision = inter / len(p) if p else (1.0 if not g else 0.0)
    recall = inter / len(g) if g else (1.0 if not p else 0.0)
    jaccard = inter / union if union else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 'jaccard': jaccard}


def build_replay_input(task: dict, top_level_dirs: list) -> dict:
    """Build the per-task summary the tagger replays over.

    Mirrors ``harness._tag_task_modules`` exactly: ``id`` is stringified, and
    ``description`` falls back to ``details`` (then to ``''``) so the tagger
    still has context when the description is empty. *top_level_dirs* is accepted
    for call-signature uniformity with ``faithful_prompt_for`` (the dir listing
    is embedded by the shared prompt builder, not by this per-task summary).
    """
    return {
        'id': str(task.get('id', '')),
        'title': task.get('title', ''),
        'description': task.get('description') or task.get('details') or '',
    }


def faithful_prompt_for(task: dict, top_level_dirs: list) -> str:
    """Build the module-tagger prompt for a SINGLE task, faithfully.

    Delegates to the shared production builder
    (``module_tagger_prompt.build_tagger_prompt``) over a single-element summary
    list, so the replay's prompt is byte-identical to what production would send
    "at the call site" — no drifting hand-copy.
    """
    summary = build_replay_input(task, top_level_dirs)
    return build_tagger_prompt(top_level_dirs, [summary])


def parse_predictions(agent_result: Any) -> list[str]:
    """Extract the predicted file list from a single-task tagger replay result.

    Reads ``AgentResult.structured_output`` first (falling back to
    ``json.loads(result.output)``), then reuses ``harness._extract_tagger_entries``
    to peel known StructuredOutput wrapper keys ('predictions'/legacy 'tasks')
    before flattening the entries' ``files`` into one list. A missing/unparseable
    payload yields ``[]`` (fail-safe, matching the harness bad-output early
    return). The replay invokes one task at a time, so the flattened list is that
    task's prediction.
    """
    payload = getattr(agent_result, 'structured_output', None)
    if not payload:
        output = getattr(agent_result, 'output', None)
        try:
            payload = json.loads(output) if output else None
        except (json.JSONDecodeError, TypeError):
            payload = None
    if not payload:
        return []

    files: list[str] = []
    for entry in _extract_tagger_entries(payload):
        if isinstance(entry, dict):
            files.extend(entry.get('files') or [])
    return files


# ── frontier adjudication (D-6): opus tie-breaks haiku-vs-sonnet disagreements ─

# output_schema for the opus adjudicator: a single constrained winner.
ADJUDICATION_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {
        'winner': {'type': 'string', 'enum': ['haiku', 'sonnet', 'tie']},
        'reason': {'type': 'string'},
    },
    'required': ['winner'],
}


def build_adjudication_prompt(
    task: dict,
    haiku_files: list,
    sonnet_files: list,
    ground_truth: list,
) -> str:
    """Build the frontier-adjudication prompt for one haiku-vs-sonnet disagreement.

    Mirrors the tier1 D-6 shape: present both candidate file sets and the ground
    truth (the files the task actually touched when it landed), and ask which
    candidate better matches. The answer is constrained to haiku/sonnet/tie by
    ``ADJUDICATION_SCHEMA``. Pure string builder — no model call.
    """
    tid = str(task.get('id', ''))
    title = task.get('title', '')
    return f"""\
Two models predicted which files a coding task would create or modify. Judge
which prediction better matches the GROUND TRUTH — the files the task actually
touched when it landed on main.

# Task
id: {tid}
title: {title}

# Ground truth files (what the task actually touched)
{json.dumps(sorted(set(ground_truth)))}

# Candidate "haiku" prediction
{json.dumps(sorted(set(haiku_files)))}

# Candidate "sonnet" prediction
{json.dumps(sorted(set(sonnet_files)))}

Which candidate better matches the ground truth? Weigh BOTH precision (no
spurious files) and recall (no missed files). Answer with a single winner:
"haiku" if the haiku prediction is better, "sonnet" if the sonnet prediction
is better, or "tie" if they are equivalently good (including equally wrong).
"""


def tally_adjudications(verdicts: list) -> dict[str, float]:
    """Tally frontier verdicts into head-to-head counts + a worse-fraction.

    Each verdict is ``{'winner': 'haiku'|'sonnet'|'tie'}``. Returns
    ``{haiku_better, sonnet_better, tie, haiku_worse_fraction}`` where
    ``haiku_worse_fraction = sonnet_better / max(1, total_non_tie)`` — the share
    of DECIDED (non-tie) disagreements the frontier judged haiku worse on. The
    ``max(1, …)`` keeps it total (0.0 when there are no non-tie verdicts).
    """
    haiku_better = sum(1 for v in verdicts if v.get('winner') == 'haiku')
    sonnet_better = sum(1 for v in verdicts if v.get('winner') == 'sonnet')
    tie = sum(1 for v in verdicts if v.get('winner') == 'tie')
    total_non_tie = haiku_better + sonnet_better
    haiku_worse_fraction = sonnet_better / max(1, total_non_tie)
    return {
        'haiku_better': haiku_better,
        'sonnet_better': sonnet_better,
        'tie': tie,
        'haiku_worse_fraction': haiku_worse_fraction,
    }


# ── decision thresholds (design decision 5) ─────────────────────────────────
#
# haiku is ~10x cheaper than sonnet, so F1 parity within a small band is a net
# win. The agreement floor + opus-not-worse guard catch qualitative divergence
# the F1 mean might hide; MIN_SAMPLES guards against deciding on too few tasks.
# These are DESIGN parameters — the marginal band deliberately escalates to a
# human rather than auto-flipping.
F1_PARITY_BAND = 0.05    # PASS: mean(haiku_f1) >= mean(sonnet_f1) - this
F1_FAIL_GAP = 0.15       # FAIL: mean(sonnet_f1) - mean(haiku_f1) > this
AGREEMENT_FLOOR = 0.70   # PASS: mean haiku-vs-sonnet Jaccard >= this
AGREEMENT_FAIL = 0.50    # FAIL: mean Jaccard < this
ADJ_WORSE_PASS = 0.50    # PASS: opus not-majority-worse — haiku_worse_fraction <= this
ADJ_WORSE_FAIL = 0.60    # FAIL: opus majority-worse — haiku_worse_fraction > this
MIN_SAMPLES = 20         # PASS: at least this many done-with-ground-truth tasks

# ── sample-selection policy (esc-2540-16, L1-ratified) ──────────────────────
#
# The live corpus is ~1800 eligible done tasks — replaying all of them is
# cost-prohibitive (2N tagger + up to N opus adjudications). The trial runs a
# bounded sample of the MOST-RECENT eligible done tasks (highest task ids)
# rather than the oldest: build_tagger_prompt embeds the CURRENT top-level-dir
# listing, so recent tasks reflect the codebase structure both models actually
# see; an oldest-N sample predates today's layout and would depress both
# models' F1 vs ground truth, biasing the verdict. DEFAULT_SAMPLE_SIZE sits
# comfortably above MIN_SAMPLES so a healthy run clears the too-few-samples
# marginal band. Pass --all to opt into the full (expensive) corpus.
DEFAULT_SAMPLE_SIZE = 30


def decide(summary: dict) -> str:
    """Return ``'pass'`` / ``'marginal'`` / ``'fail'`` from a trial summary.

    Consumes ``{mean_haiku_f1, mean_sonnet_f1, mean_jaccard,
    haiku_worse_fraction, n_samples}`` and applies design decision 5:

    - FAIL (checked first — any hard-fail trigger dominates): haiku F1 more than
      ``F1_FAIL_GAP`` below sonnet, OR agreement below ``AGREEMENT_FAIL``, OR the
      frontier judged haiku worse on a majority (> ``ADJ_WORSE_FAIL``) of decided
      disagreements.
    - PASS: F1 within ``F1_PARITY_BAND`` of sonnet AND agreement at/above
      ``AGREEMENT_FLOOR`` AND frontier not-majority-worse (<= ``ADJ_WORSE_PASS``)
      AND at least ``MIN_SAMPLES`` tasks.
    - MARGINAL: everything else (too few samples, or any in-between band) —
      escalate to a human.
    """
    haiku_f1 = summary['mean_haiku_f1']
    sonnet_f1 = summary['mean_sonnet_f1']
    jaccard = summary['mean_jaccard']
    worse = summary['haiku_worse_fraction']
    n = summary['n_samples']

    # Positive gap = haiku is worse than sonnet.
    f1_gap = sonnet_f1 - haiku_f1

    # FAIL — any hard-fail trigger.
    if f1_gap > F1_FAIL_GAP or jaccard < AGREEMENT_FAIL or worse > ADJ_WORSE_FAIL:
        return 'fail'

    # PASS — all guards clear AND enough samples.
    if (
        f1_gap <= F1_PARITY_BAND
        and jaccard >= AGREEMENT_FLOOR
        and worse <= ADJ_WORSE_PASS
        and n >= MIN_SAMPLES
    ):
        return 'pass'

    # Everything else → marginal → escalate to a human.
    return 'marginal'


def _thresholds_dict() -> dict[str, float]:
    """The design-decision-5 thresholds the verdict was computed against."""
    return {
        'F1_PARITY_BAND': F1_PARITY_BAND,
        'F1_FAIL_GAP': F1_FAIL_GAP,
        'AGREEMENT_FLOOR': AGREEMENT_FLOOR,
        'AGREEMENT_FAIL': AGREEMENT_FAIL,
        'ADJ_WORSE_PASS': ADJ_WORSE_PASS,
        'ADJ_WORSE_FAIL': ADJ_WORSE_FAIL,
        'MIN_SAMPLES': MIN_SAMPLES,
    }


@dataclass(frozen=True)
class TrialResult:
    """Aggregated trial outcome (what ``run_trial`` returns, what
    ``render_report`` renders).

    ``haiku`` / ``sonnet`` are per-model mean ``{precision, recall, f1}`` vs
    ground truth; ``agreement`` is the mean symmetric haiku-vs-sonnet
    ``{precision, recall, f1, jaccard}``; ``adjudication`` is the opus tally
    (``tally_adjudications`` output); ``decision`` is ``decide()``'s verdict.
    """
    n_samples: int
    haiku: dict
    sonnet: dict
    agreement: dict
    adjudication: dict
    decision: str


def render_report(trial_result: TrialResult) -> str:
    """Render *trial_result* to a committed markdown report.

    Contains a haiku-vs-sonnet leaderboard, the mean agreement/Jaccard, the
    frontier adjudication tally, the decision verdict, N, and the design
    thresholds — plus a fenced ``json`` machine-readable summary block the act
    step (step 16) parses for the decision + numbers.
    """
    tr = trial_result
    adj = tr.adjudication
    summary = {
        'n_samples': tr.n_samples,
        'haiku': tr.haiku,
        'sonnet': tr.sonnet,
        'agreement': tr.agreement,
        'adjudication': tr.adjudication,
        'decision': tr.decision,
        'thresholds': _thresholds_dict(),
    }

    def _row(name: str, m: dict) -> str:
        return f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"

    return f"""\
# module_tagger haiku replay-agreement trial

Task 2540 (Routing κ) — offline haiku-vs-fresh-sonnet replay over {tr.n_samples}
historical done tasks carrying ground-truth `metadata.files`. The pass/fail
verdict is computed by this trial (δ's rollup is only the post-flip watch).

**Decision: {tr.decision.upper()}**  (N = {tr.n_samples})

## Leaderboard — mean precision / recall / F1 vs ground truth

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
{_row('haiku', tr.haiku)}
{_row('sonnet', tr.sonnet)}

## Haiku-vs-sonnet agreement (symmetric, mean)

- Precision: {tr.agreement['precision']:.3f}
- Recall: {tr.agreement['recall']:.3f}
- F1: {tr.agreement['f1']:.3f}
- Jaccard: {tr.agreement['jaccard']:.3f}

## Frontier (opus) adjudication of haiku-vs-sonnet disagreements

- haiku better: {adj['haiku_better']}
- sonnet better: {adj['sonnet_better']}
- tie: {adj['tie']}
- haiku_worse_fraction: {adj['haiku_worse_fraction']:.3f}

## Thresholds (design decision 5)

| Constant | Value |
|----------|-------|
| F1_PARITY_BAND | {F1_PARITY_BAND} |
| F1_FAIL_GAP | {F1_FAIL_GAP} |
| AGREEMENT_FLOOR | {AGREEMENT_FLOOR} |
| AGREEMENT_FAIL | {AGREEMENT_FAIL} |
| ADJ_WORSE_PASS | {ADJ_WORSE_PASS} |
| ADJ_WORSE_FAIL | {ADJ_WORSE_FAIL} |
| MIN_SAMPLES | {MIN_SAMPLES} |

## Machine-readable summary

```json
{json.dumps(summary, indent=2)}
```
"""


# ── run_trial: deterministic end-to-end orchestration (unit-tested core) ─────


def _eligible_for_trial(task: dict) -> bool:
    """Whether *task* is a valid trial sample.

    Eligible iff it carries a non-empty ground-truth ``metadata.files`` set AND
    is NOT a deterministic task. Deterministic tasks carry no worktree and no
    code (CLAUDE.md "Deterministic task kind"), so a file-lock prediction is
    meaningless for them — the exact exclusion ``harness._tag_task_modules``
    applies. Empty-``files`` tasks (docs-only / no-op) give degenerate
    precision/recall and are dropped. Status filtering to ``done`` is the
    loader's job (see ``load_done_tasks``), not this predicate's.
    """
    meta = task.get('metadata') or {}
    if meta.get('task_kind') == 'deterministic':
        return False
    return bool(meta.get('files') or [])


def _task_recency_key(task: dict) -> tuple[int, int]:
    """Sort key ordering tasks newest-first by numeric id.

    Task ids are numeric (int or numeric string). A non-numeric / missing id
    fails safe to the OLDEST position (``(0, 0)``) so it is never spuriously
    treated as most-recent — the sample is drawn descending, so oldest-position
    keys are the first dropped. The second tuple element is unused today (ids
    are unique) and only pins a stable, deterministic order.
    """
    raw = task.get('id')
    try:
        return (int(str(raw)), 0)
    except (TypeError, ValueError):
        return (0, 0)


def select_trial_sample(tasks: list, size: int | None) -> list[dict]:
    """Pick the bounded MOST-RECENT eligible slice to replay (esc-2540-16).

    Filters *tasks* to ``_eligible_for_trial`` members, orders them newest-first
    by task id, and returns the first *size* (all eligible when ``size is None``
    — the explicit ``--all`` full-corpus path). Selecting most-recent rather
    than the first-N-as-loaded (oldest, ascending id) keeps the sample aligned
    with the current top-level-dir structure ``build_tagger_prompt`` embeds; see
    the ``DEFAULT_SAMPLE_SIZE`` note above for the rationale.
    """
    eligible = sorted(
        (t for t in tasks if _eligible_for_trial(t)),
        key=_task_recency_key,
        reverse=True,
    )
    if size is None:
        return eligible
    return eligible[:size]


def _mean_metrics(scores: list, keys: tuple) -> dict[str, float]:
    """Mean of each metric key across a list of ``set_scores`` dicts.

    Empty input yields 0.0 for every key (total — no ZeroDivision). run_trial
    filters to eligible tasks first, so an empty list only arises with zero
    eligible samples, where N < MIN_SAMPLES makes the verdict marginal anyway.
    """
    if not scores:
        return {k: 0.0 for k in keys}
    return {k: sum(s[k] for s in scores) / len(scores) for k in keys}


def run_trial(
    tasks: list,
    top_level_dirs: list,
    invoke_fn,
    adjudicate_fn,
) -> TrialResult:
    """Drive the whole offline trial deterministically over *tasks*.

    *invoke_fn* is ``(faithful_prompt, model) -> AgentResult`` and is called
    once per eligible task for each of ``'haiku'`` and ``'sonnet'`` with the
    byte-identical production prompt (``faithful_prompt_for``). *adjudicate_fn*
    is ``(adjudication_prompt) -> {'winner': 'haiku'|'sonnet'|'tie'}`` and is
    called ONLY on haiku-vs-sonnet disagreements, compared on the sanitized
    (production-lock) file set — the same set scoring runs on.

    Both model seams are injected so this core is unit-testable with fakes;
    ``main()`` wires the live haiku/sonnet/opus ``invoke_agent``. Returns a
    ``TrialResult`` carrying per-model mean precision/recall/F1 vs ground truth,
    the mean symmetric haiku-vs-sonnet agreement, the opus tally, ``n_samples``,
    and ``decide()``'s verdict.
    """
    eligible = [t for t in tasks if _eligible_for_trial(t)]

    haiku_scores: list = []
    sonnet_scores: list = []
    agreement_scores: list = []
    verdicts: list = []

    for task in eligible:
        gold = (task.get('metadata') or {}).get('files') or []
        prompt = faithful_prompt_for(task, top_level_dirs)

        haiku_files = parse_predictions(invoke_fn(prompt, 'haiku'))
        sonnet_files = parse_predictions(invoke_fn(prompt, 'sonnet'))

        # (a) precision/recall/F1 vs ground truth per model; (b) symmetric
        # haiku-vs-sonnet agreement. set_scores sanitizes both sides.
        haiku_scores.append(set_scores(haiku_files, gold))
        sonnet_scores.append(set_scores(sonnet_files, gold))
        agreement_scores.append(set_scores(haiku_files, sonnet_files))

        # Frontier-adjudicate ONLY genuine disagreements — compared on the
        # sanitized (production-lock) set, so a difference that sanitizes away
        # is not counted as a disagreement.
        haiku_set = set(sanitize_files_for_persist(list(haiku_files)))
        sonnet_set = set(sanitize_files_for_persist(list(sonnet_files)))
        if haiku_set != sonnet_set:
            verdicts.append(adjudicate_fn(
                build_adjudication_prompt(task, haiku_files, sonnet_files, gold)
            ))

    n_samples = len(eligible)
    haiku = _mean_metrics(haiku_scores, ('precision', 'recall', 'f1'))
    sonnet = _mean_metrics(sonnet_scores, ('precision', 'recall', 'f1'))
    agreement = _mean_metrics(
        agreement_scores, ('precision', 'recall', 'f1', 'jaccard'),
    )
    adjudication = tally_adjudications(verdicts)

    decision = decide({
        'mean_haiku_f1': haiku['f1'],
        'mean_sonnet_f1': sonnet['f1'],
        'mean_jaccard': agreement['jaccard'],
        'haiku_worse_fraction': adjudication['haiku_worse_fraction'],
        'n_samples': n_samples,
    })

    return TrialResult(
        n_samples=n_samples,
        haiku=haiku,
        sonnet=sonnet,
        agreement=agreement,
        adjudication=adjudication,
        decision=decision,
    )


# ── live wiring (impl-only; not unit-tested — real models + task DB) ─────────
#
# run_trial is SYNC (its unit-tested contract), while invoke_agent and the task
# read are async. main() bridges with ``asyncio.run()`` per call: main() itself
# is sync, so each call spins up and tears down its own loop with no running
# loop to nest inside — the most robust bridge for invoke_agent's subprocess
# transports, and fine for an offline batch. httpx / invoke_agent are imported
# lazily inside these helpers so the tested core keeps a minimal, dependency-
# light import surface (the ``shared`` pytest env need not carry them).

PROJECT_ROOT = Path('/home/leo/src/dark-factory')
FUSED_MEMORY_URL = 'http://127.0.0.1:8002'
REPORT_REL_PATH = 'plans/module-tagger-haiku-trial-report.md'

# Account-pool OAuth env var for live calls (reviewer_trial / run_vllm_eval
# convention: CLAUDE_OAUTH_TOKEN_A..G; _A is the documented eval/frontier
# account). A standalone script does NOT inherit the orchestrator's usage_gate
# slot, so the spawned Claude CLI subprocess would otherwise run with ambient
# credentials — typically "Not logged in" — and every invocation would fail
# fast. Resolve one of the pool tokens and pass it through as invoke_agent's
# oauth_token (which sets CLAUDE_CODE_OAUTH_TOKEN in the subprocess env).
OAUTH_TOKEN_ENV = 'CLAUDE_OAUTH_TOKEN_A'

# Live-call resilience + throughput. Each invoke_agent call is a ~1-min CLI
# subprocess; a 30-sample run is ~2N tagger + up to N opus calls. Retry a
# transient failure a few times before going loud (so one flake never wastes
# the batch), and fan independent calls out under a bounded semaphore against
# the single pool account (see _prefetch_live) so the whole run fits a single
# foreground window instead of ~80 min sequential.
LIVE_RETRIES = 2
LIVE_RETRY_BACKOFF_SECS = 4.0
DEFAULT_CONCURRENCY = 4


def _resolve_oauth_token(env_var: str) -> str | None:
    """Resolve the account-pool OAuth token for live calls (reviewer_trial shape).

    Returns the token from *env_var* (default ``CLAUDE_OAUTH_TOKEN_A``), or
    ``None`` with a loud WARNING when it is unset — deferring to ambient Claude
    CLI credentials, which for a standalone run are usually "Not logged in".
    Mirrors ``evals/reviewer_trial/__main__.py``'s ``--oauth-token-env`` handling.
    """
    import os
    token = os.environ.get(env_var)
    if not token:
        print(f'WARNING: {env_var} not set; falling back to ambient Claude CLI '
              f'credentials (a standalone run is typically "Not logged in", '
              f'which will surface as a loud per-invocation failure below).')
        return None
    return token


def _flatten_tasks(tasks_payload: Any) -> list[dict]:
    """Flatten a get_tasks payload (with nested subtasks) into a flat list."""
    out: list[dict] = []

    def _walk(items: Any) -> None:
        if not isinstance(items, list):
            return
        for t in items:
            if isinstance(t, dict):
                out.append(t)
                _walk(t.get('subtasks') or [])

    if isinstance(tasks_payload, dict):
        _walk(tasks_payload.get('tasks'))
    return out


class _FusedMemoryClient:
    """Minimal read-only HTTP/JSON-RPC client for the fused-memory MCP server.

    Mirrors ``scripts/migrate_metadata_modules_to_files.py``'s client (the
    established script→server read pattern), condensed to the single read-only
    ``get_tasks`` call this trial needs. httpx is imported lazily so importing
    this module (for the unit-tested core) does not require httpx.
    """

    def __init__(self, server_url: str):
        self._url = server_url.rstrip('/')
        self._client: Any = None
        self._session_id: str | None = None

    async def __aenter__(self) -> _FusedMemoryClient:
        import httpx
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._session_id = uuid.uuid4().hex
        await self._post({
            'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
            'params': {
                'protocolVersion': '2024-11-05',
                'clientInfo': {'name': 'module-tagger-trial', 'version': '1.0'},
                'capabilities': {},
            },
        })
        await self._post({
            'jsonrpc': '2.0', 'method': 'notifications/initialized', 'params': {},
        })
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def _post(self, payload: dict) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json, text/event-stream',
            'mcp-session-id': self._session_id or '',
        }
        resp = await self._client.post(
            f'{self._url}/mcp/', json=payload, headers=headers,
        )
        resp.raise_for_status()
        # 202 Accepted (notifications) returns no body.
        if resp.status_code == 202 or not resp.content:
            return {}
        if 'text/event-stream' in resp.headers.get('content-type', ''):
            for line in resp.text.splitlines():
                if line.startswith('data:'):
                    return json.loads(line[5:].strip())
            raise RuntimeError(f'no SSE data line: {resp.text[:200]}')
        return resp.json()

    async def call_tool(self, name: str, arguments: dict) -> dict:
        result = await self._post({
            'jsonrpc': '2.0', 'id': uuid.uuid4().hex, 'method': 'tools/call',
            'params': {'name': name, 'arguments': arguments},
        })
        if 'error' in result:
            raise RuntimeError(f'{name} failed: {result["error"]}')
        content = result.get('result', {})
        if 'structuredContent' in content:
            return content['structuredContent']
        for entry in content.get('content', []) or []:
            if entry.get('type') == 'text':
                try:
                    return json.loads(entry['text'])
                except json.JSONDecodeError:
                    return {'_raw': entry['text']}
        return content


async def load_done_tasks(
    project_root: Path, server_url: str = FUSED_MEMORY_URL,
) -> list[dict]:
    """Read DONE tasks carrying non-empty ground-truth ``metadata.files``.

    Read-only ``get_tasks`` against the running fused-memory server. Ground
    truth is the done task's already-reconciled ``metadata.files`` (the actual
    merge-diff files; ``TaskWorkflow._reconcile_metadata_files_for_done``).
    Deterministic tasks are dropped later by ``run_trial`` (``_eligible_for_trial``).
    """
    async with _FusedMemoryClient(server_url) as client:
        payload = await client.call_tool(
            'get_tasks',
            {'project_root': str(project_root), 'with_subtasks': True},
        )
    done: list[dict] = []
    for t in _flatten_tasks(payload):
        if str(t.get('status', '')) != 'done':
            continue
        meta = t.get('metadata') or {}
        if meta.get('files'):
            done.append(t)
    return done


def current_top_level_dirs(project_root: Path) -> list[str]:
    """Current top-level dir listing — mirrors ``harness._tag_task_modules``."""
    try:
        return sorted(
            p.name for p in project_root.iterdir()
            if p.is_dir() and not p.name.startswith('.')
        )
    except OSError:
        return []


async def _live_invoke(
    prompt: str, model: str, project_root: Path,
    oauth_token: str | None = None,
) -> Any:
    """Live tagger replay for one (prompt, model) via real ``invoke_agent``.

    Faithful "at the call site": byte-identical production prompt + schema +
    system prompt; only the model is forced (haiku vs sonnet head-to-head), and
    ``effort='low'`` is applied SYMMETRICALLY to both so the comparison stays
    fair. ``allowed_tools=[]`` — the tagger only emits structured output.
    *oauth_token* (the account-pool token) is forwarded so the CLI subprocess is
    actually authenticated (see ``OAUTH_TOKEN_ENV``).

    Fails LOUD on an unsuccessful invocation instead of returning a result that
    ``parse_predictions`` would silently flatten to ``[]`` — a failed call
    (auth/cap/timeout) scored as an "empty prediction" would corrupt the
    precision/recall verdict (no-silent-fail-soft / structured-facts-at-failure).
    A transient failure is retried (``LIVE_RETRIES`` attempts, brief backoff)
    before going loud, so one flaky call does not waste the whole batch.
    """
    from orchestrator.agents.invoke import invoke_agent
    result: Any = None
    for attempt in range(LIVE_RETRIES + 1):
        result = await invoke_agent(
            prompt=prompt,
            system_prompt=TAGGER_SYSTEM_PROMPT,
            cwd=project_root,
            model=model,
            output_schema=TAGGER_SCHEMA,
            allowed_tools=[],
            effort='low',
            max_turns=2,
            max_budget_usd=0.50,
            oauth_token=oauth_token,
        )
        if getattr(result, 'success', False):
            return result
        if attempt < LIVE_RETRIES:
            await asyncio.sleep(LIVE_RETRY_BACKOFF_SECS * (attempt + 1))
    raise RuntimeError(
        f'module_tagger replay FAILED after {LIVE_RETRIES + 1} attempts '
        f'(model={model}, success={getattr(result, "success", None)}, '
        f'subtype={getattr(result, "subtype", None)!r}, '
        f'timed_out={getattr(result, "timed_out", None)}, '
        f'account={getattr(result, "account_name", None)!r}): '
        f'{(getattr(result, "output", None) or "")[:200]!r}. Refusing to '
        f'score a failed call as an empty prediction.'
    )


async def _live_adjudicate(
    prompt: str, project_root: Path, oauth_token: str | None = None,
) -> dict:
    """Frontier (opus) adjudication for one haiku-vs-sonnet disagreement.

    Returns ``{'winner': 'haiku'|'sonnet'|'tie'}``. *oauth_token* (the
    account-pool token) authenticates the CLI subprocess. Fails LOUD on an
    unsuccessful invocation (auth/cap/timeout) — a failed opus call must not be
    silently counted as a neutral 'tie', which would understate
    ``haiku_worse_fraction``. The ``'tie'`` fallback is retained ONLY for a
    SUCCESSFUL call whose payload is missing / out-of-enum, so a genuinely
    ambiguous judgement never crashes the run. A transient failure is retried
    (``LIVE_RETRIES`` attempts, brief backoff) before going loud.
    """
    from orchestrator.agents.invoke import invoke_agent
    result: Any = None
    for attempt in range(LIVE_RETRIES + 1):
        result = await invoke_agent(
            prompt=prompt,
            system_prompt=(
                'You are an impartial adjudicator comparing two file-prediction '
                'candidates against ground truth. Answer with a single winner.'
            ),
            cwd=project_root,
            model='opus',
            output_schema=ADJUDICATION_SCHEMA,
            allowed_tools=[],
            effort='low',
            max_turns=2,
            max_budget_usd=1.0,
            oauth_token=oauth_token,
        )
        if getattr(result, 'success', False):
            break
        if attempt < LIVE_RETRIES:
            await asyncio.sleep(LIVE_RETRY_BACKOFF_SECS * (attempt + 1))
    else:
        raise RuntimeError(
            f'opus adjudication FAILED after {LIVE_RETRIES + 1} attempts '
            f'(success={getattr(result, "success", None)}, '
            f'subtype={getattr(result, "subtype", None)!r}, '
            f'timed_out={getattr(result, "timed_out", None)}, '
            f'account={getattr(result, "account_name", None)!r}): '
            f'{(getattr(result, "output", None) or "")[:200]!r}. Refusing to '
            f'count a failed adjudication as a neutral tie.'
        )
    payload = getattr(result, 'structured_output', None) or {}
    winner = payload.get('winner')
    if winner not in ('haiku', 'sonnet', 'tie'):
        winner = 'tie'
    return {'winner': winner}


async def _prefetch_live(
    eligible: list,
    top_level_dirs: list,
    project_root: Path,
    oauth_token: str | None,
    concurrency: int,
) -> tuple[dict, dict]:
    """Concurrently pre-warm tagger + adjudication results (throughput only).

    Returns two caches — ``pred_cache[(prompt, model)] -> AgentResult`` and
    ``adj_cache[adjudication_prompt] -> {'winner': ...}`` — populated by fanning
    the INDEPENDENT live calls out under a ``concurrency``-bounded semaphore
    (bounded so the single pool account is not overrun). The caches are a PURE
    OPTIMIZATION: ``run_trial`` stays the source of truth for which calls matter,
    and ``main``'s ``invoke_fn`` / ``adjudicate_fn`` fall back to a live call on
    any cache miss — so a mis-warmed key degrades to slower, never to wrong.

    Phase 2 selects adjudications with the EXACT predicate ``run_trial`` uses
    (sanitized-set inequality on the same parsed predictions) and builds the
    prompt with the same ``build_adjudication_prompt`` call, so the pre-warmed
    key matches the one ``run_trial`` will request — a cache hit, not a miss.
    """
    sem = asyncio.Semaphore(max(1, concurrency))

    # Phase 1 — tagger predictions (haiku + sonnet) for every eligible task.
    pred_cache: dict = {}

    async def _tag(prompt: str, model: str) -> None:
        async with sem:
            pred_cache[(prompt, model)] = await _live_invoke(
                prompt, model, project_root, oauth_token,
            )

    tagger_jobs = []
    for task in eligible:
        prompt = faithful_prompt_for(task, top_level_dirs)
        tagger_jobs.append(_tag(prompt, 'haiku'))
        tagger_jobs.append(_tag(prompt, 'sonnet'))
    await asyncio.gather(*tagger_jobs)

    # Phase 2 — opus adjudication ONLY for sanitized-set disagreements.
    adj_cache: dict = {}

    async def _adj(prompt: str) -> None:
        async with sem:
            adj_cache[prompt] = await _live_adjudicate(
                prompt, project_root, oauth_token,
            )

    adj_jobs = []
    for task in eligible:
        gold = (task.get('metadata') or {}).get('files') or []
        prompt = faithful_prompt_for(task, top_level_dirs)
        haiku_files = parse_predictions(pred_cache[(prompt, 'haiku')])
        sonnet_files = parse_predictions(pred_cache[(prompt, 'sonnet')])
        haiku_set = set(sanitize_files_for_persist(list(haiku_files)))
        sonnet_set = set(sanitize_files_for_persist(list(sonnet_files)))
        if haiku_set != sonnet_set:
            adj_jobs.append(_adj(
                build_adjudication_prompt(task, haiku_files, sonnet_files, gold)
            ))
    await asyncio.gather(*adj_jobs)

    return pred_cache, adj_cache


def main() -> int:
    """Run the live trial and write the markdown report (impl-only).

    Loads done tasks (read-only), replays haiku vs fresh sonnet at the call
    site, opus-adjudicates disagreements, computes the pass/marginal/fail
    verdict, and writes ``render_report`` to
    ``plans/module-tagger-haiku-trial-report.md``. The conditional config flip /
    escalation is step 16, not this entrypoint.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='module_tagger haiku replay-agreement trial (task 2540)',
    )
    parser.add_argument('--project-root', default=str(PROJECT_ROOT))
    parser.add_argument('--server-url', default=FUSED_MEMORY_URL)
    parser.add_argument('--out', default=None,
                        help='report path (default: <project_root>/' + REPORT_REL_PATH + ')')
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                        help='number of most-recent eligible done tasks to '
                             'replay (default: %(default)s; esc-2540-16 policy)')
    parser.add_argument('--all', action='store_true',
                        help='replay the FULL eligible corpus instead of a '
                             'bounded recent sample (expensive — opt-in)')
    parser.add_argument('--oauth-token-env', default=OAUTH_TOKEN_ENV,
                        help='env var holding the account-pool OAuth token for '
                             'live calls (default: %(default)s; reviewer_trial '
                             'convention — a standalone run has no usage_gate slot)')
    parser.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help='max concurrent live calls when pre-warming the '
                             'sample (default: %(default)s; bounded to spare the '
                             'single pool account)')
    args = parser.parse_args()

    project_root = Path(args.project_root)
    out_path = Path(args.out) if args.out else project_root / REPORT_REL_PATH

    oauth_token = _resolve_oauth_token(args.oauth_token_env)

    tasks = asyncio.run(load_done_tasks(project_root, args.server_url))
    size = None if args.all else args.sample_size
    tasks = select_trial_sample(tasks, size)
    dirs = current_top_level_dirs(project_root)

    # Pre-warm all independent live calls concurrently (throughput only); the
    # caches are best-effort and invoke_fn/adjudicate_fn fall back to a live
    # call on any miss, so run_trial stays the source of truth.
    eligible = [t for t in tasks if _eligible_for_trial(t)]
    print(f'Pre-warming {len(eligible)} eligible tasks '
          f'(~{2 * len(eligible)} tagger calls + adjudications) '
          f'at concurrency={args.concurrency}…')
    pred_cache, adj_cache = asyncio.run(
        _prefetch_live(eligible, dirs, project_root, oauth_token, args.concurrency)
    )

    def invoke_fn(prompt: str, model: str) -> Any:
        cached = pred_cache.get((prompt, model))
        if cached is not None:
            return cached
        return asyncio.run(_live_invoke(prompt, model, project_root, oauth_token))

    def adjudicate_fn(prompt: str) -> dict:
        cached = adj_cache.get(prompt)
        if cached is not None:
            return cached
        return asyncio.run(_live_adjudicate(prompt, project_root, oauth_token))

    result = run_trial(tasks, dirs, invoke_fn, adjudicate_fn)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_report(result))
    print(f'Trial complete: decision={result.decision} '
          f'N={result.n_samples} → {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
