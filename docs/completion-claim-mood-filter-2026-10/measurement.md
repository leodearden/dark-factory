# Completion-claim gate: non-assertive-mood filter, measured

**Task 6677 · swept 2026-10-10T04:06Z · base `d902cb4b5c` → head `13760eef35`**

The filter stops `completion_claim_gate` from reading a mention of a task's landing as a
claim that it landed. It has six rules: quotations, modal forms, non-veridical scopes
(if/whether/unless/in case), imperative heads, present-tense temporal scopes, and
attributive participles that bind forward only. The rules are pinned by
`fused-memory/tests/test_completion_claim_mood.py`. This file measures what the rules do
on the live corpus.

| file | what it is |
|---|---|
| `report-before.json` | the sweep script's stdout, byte for byte, with the BASE gate on `PYTHONPATH` |
| `report-after.json` | the same command, with this branch's gate |
| `claims-diff.json` | the claim-level replay: every claim the filter dropped or added, adjudicated |
| `measurement.md` | this file |

---

## Headline

| measure | before | after |
|---|---|---|
| Episodes scanned | 11323 (dark_factory 5493 + reify 5830) | 11325 (dark_factory 5495 + reify 5830) |
| Episodes carrying ≥1 claim | 1419 | 1398 |
| Raw `mismatch` verdicts | **25** | **20** |
| Raw `unverifiable` verdicts | **12** | **11** |

`mismatch` and `unverifiable` are different facts and are never summed.

The two runs were back to back, about 45 s apart. Two dark_factory episodes were written
between them. That is corpus movement, and it changed no finding (see the next section).

## Findings delta

Keyed on `(record_uuid, claim_kind, subject, ref)`. **6 removed, 0 added.** All six appear
in `claims-diff.json`, so none of them is corpus movement.

| ref | record | before | why it was a false positive |
|---|---|---|---|
| 5471 | `1679751b` (reify) | mismatch (pending) | esc-unverified-claim-5471-1. In "#5471 owns the shipped Bool-balance … examples", the participle is attributive. |
| 5317 | `1eefbd1e` (reify) | mismatch (pending) | Task-4853 investigation Class B residual. "the shipped `reify build` CLI" describes the CLI. |
| 3846 | `448d8400` (dark_factory) | mismatch (deferred) | Task-4853 investigation residual. "a landed-on-main check" names a check. The `filing_dispatch` claim on 3846 is kept. |
| 5455 | `71a366e9` (dark_factory) | mismatch (pending) | esc-unverified-claim-5455-1. The claim is "only 'if #5455 has landed'", and the same sentence says 5455 is pending. |
| 5023 | `b94a4690` (reify) | mismatch (pending) | esc-unverified-claim-5023-2. "the day #5023 is cancelled" is hypothetical. |
| 7407 | `79716ccc` (dark_factory) | unverifiable | This task's own architect memory (written 2026-10-10T03:32:58Z), quoting the esc-unverified-claim-7407-3 sentence. |

The five removed mismatches are exactly the five the architect's prototype predicted, and
each was confirmed as a false positive. The reify copy of the 7407 sentence (`cf752dbf`)
verified at baseline, so dropping it removed a claim but no finding.

## Claim-level replay

The sweep lists only non-verified findings, so recall needs every claim. The in-scope
population was read once (10563 records, at 2026-10-10T04:09:33Z). Both gate versions
then extracted claims from that same dump, so the diff contains no corpus movement.

| measure | before | after |
|---|---|---|
| Claims | 1553 | 1525 |
| Records with claims | 1420 | 1399 |
| Dropped / added | — | **28 / 0** |

Each dropped claim was attributed to a rule by enabling one rule at a time; each drop
needed exactly one rule:

| rule | dropped |
|---|---|
| present-tense temporal scope | 9 |
| quotation | 8 |
| attributive (backward binding refused) | 8 |
| non-veridical scope | 2 |
| modal | 1 |
| imperative head | 0 |

The imperative rule dropped nothing here. The phrasings that motivate it
(esc-unverified-claim-8246-5, -6020-2) are not in the in-scope Episodic corpus, and neither
is the 5471-4 memory text. Unit and server tests pin all three.

**Precision of the filter: 26 of 28 drops are correct suppressions (92.9%).** The other two
are:

- **recall-loss**: `eedb43e9`, task 2633. "task 2633 (a landed clamp breaking two stale
  golden tests)" is an appositive that does assert the landing.
- **debatable**: `672d203f`, task 2519. The quoted unary fact ("Umbrella task 2519 was
  filed and then cancelled …") is cited for being attached to the wrong node, and the
  writer neither asserts nor disputes it.

Both claims verified at baseline, so the filter loses no tag on the live corpus.

**Recall.** The 2633 row is the only recall loss. Every test pinned by task 4853 still passes:
`test_completion_claim_gate.py`, `test_task_filter.py`,
`test_audit_unverified_completion_claims.py`, `server/test_completion_claim_gate_ingestion.py`,
`server/test_recon_premature_completion_gate.py` and `test_task_interceptor.py` give
1040 passed and 1 skipped.

Compared with the architect's prototype (1547 → 1520, 27 dropped), the one extra drop is
`79716ccc`, a record written after that prototype ran.

## Accepted residuals

- **Possessive-'s attributives, single quotes and backticks** are left out by measurement.
  Each one cost real claims in the architect's replay: "task 4105's landed fix", "was left
  `cancelled`", `kind:'merged'`. Pinned as controls in `TestQuotationsAreMentions` and
  `TestAttributiveMarkersBindForwardOnly`.
- **Appositive attributives** ("task N (a landed X)") lose the claim, as in the 2633 row above.
- **Coordinated subjects inside a scope.** A scope ends at the first barrier, and `and` is
  a barrier. So "if task 6 and task 5 have landed, re-run it" still claims task 5: the
  cue's scope stops before the second subject.
- **Counterfactuals.** In "it would have been faster after #5 landed" the modal blanks
  only the word it governs, and the temporal scope is past tense, so task 5 is still
  claimed.

## Review amendment, replayed

The review amendment, an `amend:` commit on this branch after the measured head, changed
the gate in four ways:

- The modal vocabulary covers `can't`, curly-apostrophe and `'ve` forms.
- The filler run includes `fully` and `successfully`.
- The modal and intention-phrase strippers are one regex, built from one modal list.
- Two kinds of double-quoted pair are now read as assertions: a pair that crosses `;` or
  a sentence end, and a key/value literal with whitespace after the `:`.

Replayed over the same population dump, the amended gate's claims match the after gate
exactly: 1525 → 1525, 0 dropped, 0 added. Every number above therefore holds for the
amended head.

```bash
uv run python /tmp/6677-replay/replay.py \
  extract /tmp/6677-replay/population.jsonl /tmp/6677-replay/claims-amended.jsonl
python3 /tmp/6677-replay/replay.py diff /tmp/6677-replay/claims-after.jsonl \
  /tmp/6677-replay/claims-amended.jsonl /tmp/6677-replay/claims-diff-amend.json
```

## Reproducing

From the worktree root. The three env vars are required, as explained in
`docs/unverified-completion-claim-sweep-2026-08-11/investigation.md` §Reproducing.

```bash
BASE=$(git merge-base HEAD main)            # d902cb4b5c26660ad274f0b90fa911f50ec201b2
mkdir -p /tmp/6677-base && git archive "$BASE" fused-memory/src | tar -x -C /tmp/6677-base
cd fused-memory
export PROJECT_ROOT=/home/leo/src/dark-factory \
       DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/reify \
       RECONCILIATION_DATA_DIR=/home/leo/src/dark-factory/data/reconciliation
# confirm which gate each run loads
PYTHONPATH=/tmp/6677-base/fused-memory/src uv run python -c \
  'import fused_memory.services.completion_claim_gate as g; print(g.__file__)'
PYTHONPATH=/tmp/6677-base/fused-memory/src uv run python scripts/audit_unverified_completion_claims.py \
  --project dark_factory --project reify --include-unverifiable \
  > ../docs/completion-claim-mood-filter-2026-10/report-before.json 2> /tmp/6677-sweep-before.err
uv run python scripts/audit_unverified_completion_claims.py \
  --project dark_factory --project reify --include-unverifiable \
  > ../docs/completion-claim-mood-filter-2026-10/report-after.json 2> /tmp/6677-sweep-after.err
```

Both runs exited 0, at 04:05:59Z → 04:06:35Z and 04:06:35Z → 04:07:19Z. Their stderr shows
the population counts quoted above, and that `know_live` / `knowlive` are not registered
(4 and 3 task claims UNVERIFIABLE, identical in both runs).

The claim-level replay. Read-only: `EpisodeReader` issues only `GRAPH.RO_QUERY`.

```bash
uv run python /tmp/6677-replay/replay.py dump /tmp/6677-replay/population.jsonl
PYTHONPATH=/tmp/6677-base/fused-memory/src uv run python /tmp/6677-replay/replay.py \
  extract /tmp/6677-replay/population.jsonl /tmp/6677-replay/claims-before.jsonl
uv run python /tmp/6677-replay/replay.py \
  extract /tmp/6677-replay/population.jsonl /tmp/6677-replay/claims-after.jsonl
python3 /tmp/6677-replay/replay.py diff /tmp/6677-replay/claims-before.jsonl \
  /tmp/6677-replay/claims-after.jsonl /tmp/6677-replay/claims-diff-raw.json
uv run python /tmp/6677-replay/rules.py /tmp/6677-replay/population.jsonl \
  /tmp/6677-replay/claims-diff-raw.json /tmp/6677-replay/rules.json
```

`/tmp/6677-replay/replay.py`:

```python
import asyncio, importlib.util, json, sys

SWEEP = '<worktree>/fused-memory/scripts/audit_unverified_completion_claims.py'
KNOWN = frozenset({'dark_factory', 'reify'})

def _sweep_module():
    spec = importlib.util.spec_from_file_location('audit_sweep', SWEEP)
    module = importlib.util.module_from_spec(spec)
    sys.modules['audit_sweep'] = module  # before exec: slots dataclasses need it
    spec.loader.exec_module(module)
    return module

async def _dump(out_path):
    sweep = _sweep_module()
    args = sweep._build_parser().parse_args(['--project', 'dark_factory', '--project', 'reify'])
    uri = sweep._resolve_uri(args)
    with open(out_path, 'w') as out:
        for graph in ('dark_factory', 'reify'):
            records = await sweep.EpisodeReader(graph_name=graph, uri=uri).fetch_population()
            for r in records:
                if r.category in sweep.IN_SCOPE_CATEGORIES:
                    out.write(json.dumps({'uuid': r.uuid, 'project_id': r.project_id,
                                          'graph_name': r.graph_name,
                                          'created_at': r.created_at, 'text': r.text}) + '\n')

def _extract(in_path, out_path):
    from fused_memory.services import completion_claim_gate as gate
    with open(in_path) as src, open(out_path, 'w') as out:
        for line in src:
            rec = json.loads(line)
            for c in gate.extract_completion_claims(
                    rec['text'], default_project_id=rec['project_id'] or 'dark_factory',
                    known_project_ids=KNOWN):
                out.write(json.dumps({'record_uuid': rec['uuid'], 'kind': c.kind,
                                      'subject': c.subject, 'ref': c.ref,
                                      'project_id': c.project_id,
                                      'clause': rec['text'][c.span[0]:c.span[1]].strip()}) + '\n')

def _diff(before_path, after_path, out_path):
    def load(path):
        rows = [json.loads(line) for line in open(path)]
        return {(r['record_uuid'], r['kind'], r['subject'], r['ref'], r['project_id']): r
                for r in rows}
    before, after = load(before_path), load(after_path)
    json.dump({'claims_before': len(before), 'claims_after': len(after),
               'records_with_claims_before': len({k[0] for k in before}),
               'records_with_claims_after': len({k[0] for k in after}),
               'dropped': [before[k] for k in sorted(set(before) - set(after))],
               'added': [after[k] for k in sorted(set(after) - set(before))]},
              open(out_path, 'w'), indent=2, ensure_ascii=False)

if __name__ == '__main__':
    command, *paths = sys.argv[1:]
    {'dump': lambda *p: asyncio.run(_dump(*p)), 'extract': _extract, 'diff': _diff}[command](*paths)
```

`/tmp/6677-replay/rules.py` takes the after gate and switches on one rule at a time, with
the others disabled. A claim is attributed to every rule that drops it on its own. The
script rebinds the gate's private names by name and by position, so it is pinned to the
internals at `13760eef35`. It records this measurement only: it is not a maintained tool,
and it will misattribute rules if run against a later head.

```python
import json, re, sys
from fused_memory.services import completion_claim_gate as gate

KNOWN = frozenset({'dark_factory', 'reify'})
ORIG = (gate._blank_quotations, gate._EXEMPTION_STRIPPERS, gate._MOOD_SCOPES,
        gate._ATTRIBUTIVE_LEAD_RE)
SCOPE_ROWS = {'non_veridical': gate._NON_VERIDICAL_CUE_RE,
              'imperative': gate._IMPERATIVE_HEAD_RE, 'temporal': gate._TEMPORAL_CUE_RE}

def configure(rule):
    gate._blank_quotations = ORIG[0] if rule == 'quotation' else (lambda t: t)
    gate._EXEMPTION_STRIPPERS = ORIG[1] if rule == 'modal' else ORIG[1][:-1]
    gate._MOOD_SCOPES = tuple(row for row in ORIG[2] if row[0] is SCOPE_ROWS.get(rule))
    gate._ATTRIBUTIVE_LEAD_RE = ORIG[3] if rule == 'attributive' else re.compile(r'(?!)')

population = {json.loads(l)['uuid']: json.loads(l) for l in open(sys.argv[1])}
out = {}
for row in json.load(open(sys.argv[2]))['dropped']:
    rec = population[row['record_uuid']]
    key = (row['kind'], row['subject'], row['ref'], row['project_id'])
    alone = []
    for rule in ('quotation', 'modal', 'non_veridical', 'imperative', 'temporal', 'attributive'):
        configure(rule)
        claims = gate.extract_completion_claims(
            rec['text'], default_project_id=rec['project_id'] or 'dark_factory',
            known_project_ids=KNOWN)
        if key not in {(c.kind, c.subject, c.ref, c.project_id) for c in claims}:
            alone.append(rule)
    out[f"{row['record_uuid']}|{row['kind']}|{row['ref']}"] = alone
json.dump(out, open(sys.argv[3], 'w'), indent=1)
```

`claims-diff.json` joins the raw diff with `rules.json` and `report-before.json`. A
dropped claim's `baseline_verdict` is its `report-before.json` finding status, looked up
on `(record_uuid, claim_kind, subject, ref)`, or `verified` when it is not listed there.
All 28 dropped records were created before the before-sweep, so an unlisted claim means
it verified. The `adjudication` and `note` columns are the hand adjudication. The
population dump is not committed.
