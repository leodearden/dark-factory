# S23 — Sampler collection groups and schema (PRD)

**Status:** active, authored 2026-10-10 by the S23 seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S23, agent mode, wave 1, size S).
Bare B with a contract section and a boundary-test sketch (G5 fired on blast radius: sampler,
dashboard, scripts). Two leaves.

**Code anchors** verified against main `ae3047ec3a` (2026-10-10); the run's anchors were verified at
`6551b75aba` and `git diff 6551b75aba..main` is empty for `sampler/`, `dashboard/src/dashboard/data/load.py`,
`dashboard/tests/test_load_data.py` and `scripts/load-threshold-calibration.py`. Main moves fast — cite by
symbol; re-locate at implementation time.

## Goal

The load sampler's collection group (psi / process / load) becomes one frozen value in one table, and
every other spelling of the sampler's metric vocabulary — the admission sets, the gate-arm table, the
dashboard's served metrics, the calibration script's selectors — is computed from that table. The cgroup
leaf of a per-leaf metric is stored as data in its own column instead of being packed into the metric
string and re-parsed by every reader. The samples schema gets a public home owned by the store, with
an additive migration step, and the dashboard and calibration import it instead of copying it.

What a user observes when this lands:

- The live `data/load-samples.db`, written by the timer-fired sampler, has a `leaf` column; no row's
  `metric` contains `:`; per-cgroup rows read `metric='own_cpu_some10', leaf='orchestrator-…service'`.
- `scripts/load-threshold-calibration.py --arm own_cpu_some_avg10`, run by the system `python3` that
  the ε1/ε2 gates (tasks 5219, 5220) use, reports one row per leaf, and its arm selectors come from the
  sampler's table rather than from a hand-kept copy.
- `/api/load` still serves its 9 metrics, now listed by a projection of the table.
- Adding a metric or a fourth group is one edit in one module. The next agent does not have to know
  that five other spellings and four reconciler tests exist, because they no longer exist.
- A design invariant (`sampler-vocabulary-one-table`, INV-14 — see §Design invariant) states the
  rule and a guard enforces it.

## Background

The `/review-all` run `review-all-dark_factory-20261008` found that the collection group is the
sampler's main axis of variation, and that no single value describes it. Each group's names, window
mode, admission rule and degrade handling are spelled separately in `sampler/src/sampler/metrics.py`,
`sampler/src/sampler/sampler.py` (13 admission literals), `sampler/src/sampler/__main__.py` (one
try/except per group), `metrics.py::ARM_METRIC_STEMS`, `dashboard/src/dashboard/data/load.py` (9 names)
and `scripts/load-threshold-calibration.py::ARM_METRIC_SELECTORS`. Four tests exist only to reconcile
these copies. Per-leaf metrics are encoded as `'<stem>:<leaf>'` by two f-strings and parsed by five
readers. The schema is the private `sampler.store._SCHEMA`, copied verbatim into the dashboard and a
third time into `scripts/tests/test_load_threshold_calibration.py`. The store has no additive
migration step, although its docstring says it mirrors `orchestrator.run_store.RunStore`, which has
two.

Program rulings this PRD implements without relitigating: R21 (S23 approved as proposed), R22
item 7 (the invariant below), R23 (weakened keys re-verified here).

## Finding keys

| Key | Sev | Verdict | Re-verified 2026-10-10 | Closed by |
|---|---|---|---|---|
| `fk-e86a10b9e217` collection group not a value | medium | confirmed | sampler tree unchanged since `6551b75aba`; 13 literals, 3 try blocks, 3 asserts, 4 reconcilers reproduce | β |
| `fk-37f494b26355` whole-tick-fatal `assert` admission | medium | **weakened** | Kept. The three `assert not unexpected_*` lines and the absent handler around `run_tick` are unchanged. The verifier's point stands (fail-loud was deliberate). But under the table the vocabulary error class becomes nearly impossible by construction. Admission is then a backstop, so the uniform per-group degrade costs nothing extra and removes the `-O` hole. | β |
| `fk-ab2c587ff5a9` stem:leaf packed into `metric` | medium | confirmed | 2 encoders and 5 parsers unchanged; live corpus 4,987,894 of 8,830,537 rows encoded | α |
| `fk-d207683eef5a` schema private, copied, no migration | low | confirmed | `_SCHEMA` private; dashboard copy; third copy in calibration tests; 0 ALTERs | α |
| `fk-be1621618dee` 4 cmdline reads per process, silent excepts | low | confirmed | unchanged; psutil 7.2.2 in the sampler env | β |
| `fk-fcefc2576467` host leaf count stated as fact | low | **weakened** | Kept, scoped to the verifier's reading. Three present-tense claims are stale (`metrics.py` module docstring "25 … seven", `__main__` docstring "25 in total … seven", `run_tick` "16 on this host"). The dated measurements are history. Rewriting these docstrings is already part of α/β, so closing it is free. | β (store.py sites in α) |
| `fk-895f269d7bb5` dashboard schema copy + importorskip reconciler (prior) | low | re-observed | `LOAD_SAMPLES_SCHEMA` and `test_load_schema_and_metrics_match_sampler` still present | β (schema half lands in α) |
| `fk-3e77aa904a0a` sampler prose (prior; superseded by `fk-fcefc2576467`) | low | re-observed | Phase-5 mechanical task **6644** (pending) owns it today | β — 6644 cancelled at phase B, see §Mechanical tasks |

## Pre-conditions for activating

- **S25's `scripts/_checkout.py` bootstrap** (G4 table: S25 owns it). β depends on it: the calibration
  script runs under `/usr/bin/python3` 3.12.3 at gate time, and the bootstrap is what puts this
  checkout's `sampler/src` and `shared/src` on its `sys.path`. Measured: with those two paths
  prepended, `/usr/bin/python3 -I` imports `sampler.metrics` cleanly and does not load `psutil`. The
  import is lazy, inside `collect_process_metrics`. `sampler.store` does NOT import under the system
  python (`shared.sqlite_sync_base` → `shared.async_sqlite_base` → `aiosqlite`), so the script must
  never import the store at runtime. Its tests may, through the same bootstrap. **Phase B confirms
  that S25's filed leaf binds every workspace member's `src/`, sampler's included.** If it binds
  only `shared/src`, that is raised with the lead as a seam gap, not patched here.
- α has no upstream dependency.

## Sketch of approach

**α — leaf as a column, public schema, one connection seam.**
`sampler.store` exports `SCHEMA`, the DDL of a fresh store. Its `samples` columns are
`(ts, metric, value, window_mean, window_max, leaf)`; `leaf` is last because SQLite's `ADD COLUMN`
appends, and a fresh file and a migrated file must have the same shape. The single index is
`idx_samples_metric_ts_leaf ON samples(metric, ts, leaf)`. `LoadSampleStore` applies the schema in an
order that works on all three starting states (fresh, legacy pre-leaf, already migrated). It first
creates tables, then runs `_migrate`, then creates the index. `_migrate` follows RunStore's idiom: it
feature-detects with `PRAGMA table_info(samples)`. If `leaf` is absent, it runs one transaction: add
the column; rewrite every legacy row whose `metric` contains `:` by splitting on the first `:` (the
rule `sampler.sampler::unexpected_metric_names` already states); create the new index; drop
`idx_samples_metric_ts`. Collectors emit `SampleKey(metric, leaf)` keys instead of encoded strings.
`write_tick` and the trailing window take `SampleKey`s, and the window reads `WHERE metric = ? AND
leaf IS ?`. Connection policy gets one seam: a `connect` factory injected at construction, which
`maybe_vacuum` uses with an autocommit flag. Tests then observe "one connection per tick" through
the seam instead of patching `sampler.store.sqlite3`. The calibration script reads `(leaf, ts, value)`
by `metric = ?`, keys its series by `(metric, leaf)`, and pairs a value series with its readability
series by `leaf` equality. Its GLOB statement, its 15-line GLOB-vs-LIKE essay and its four `:` parsers
are deleted. A display label may still render `stem:leaf` for the human report, because rendering is
not parsing. The dashboard deletes `LOAD_SAMPLES_SCHEMA`, declares `dark-factory-load-sampler` as a
workspace dependency, and adds `../sampler/src` to its pyright `extraPaths`. Its tests build fixtures
from `sampler.store.SCHEMA` (or a `LoadSampleStore`) and drop the `importorskip`.

**β — one MetricGroup table and its projections; the invariant.**
`sampler.metrics` holds the table (contract below). Collectors take their names from the specs.
`__main__.main` takes the table as an injectable argument, defaulting to `METRIC_GROUPS`, and loops
it with one degrade handler. `run_tick` takes `Mapping[MetricGroup, Samples]`, validates each group
with an explicit check, and drops only the offending group with an ERROR log naming the group and
the unexpected keys. It reads window mode from `group.windowed`. `ARM_METRIC_STEMS` becomes the
derived `ARM_METRICS`. The dashboard's `KNOWN_METRICS` becomes the sorted exact names of the groups
it serves, named by `GroupName`. The calibration script imports `sampler.metrics` through `_checkout`
and derives each `ArmSpec`'s selector and readability, and its tick clock, from the table. It keeps
only its own facts: ladder and unit per arm. The process collector classifies each process once from
`proc.info` into a frozen record. It catches only `psutil.Error` and reads `memory_info` only for
verify trees. Present-tense host counts go. The formula (exact names + 2 per discovered leaf) is
stated once next to the table, where it can be computed. The four-site lockstep edit registers the
invariant.

## Resolved design decisions

- **D1 — one table, collectors inside it.** `MetricGroup` carries `collect`, as the ruled remedy
  says. The table lives at the bottom of `sampler/src/sampler/metrics.py`, beside the collectors and
  below the spec constants they use. A separate table module would need the collectors (cycle) or
  would split the group again into names here and collectors there. `sampler.metrics` stays importable
  under the system python because `psutil` is imported lazily (measured). The module's purpose is
  then one sentence: "the sampler's metric groups: what each collects, what it is called, how it is
  windowed".
- **D2 — `MetricSpec` refines the ruled `(exact, stems)` pair.** A group holds
  `metrics: tuple[MetricSpec, ...]`, and `exact` / `per_leaf` are derived views. `MetricSpec` also
  carries `arm` (the `shared.psi` gate arm it records, or `None`) and `readability` (the spec that
  records whether it was readable, or `None`). Without these, the gate-arm table and the calibration
  readability pairing would stay hand-kept, which R22 item 7 forbids. The arm *names* are owned by
  `shared.psi`. The table maps onto them, and the existing arm-parity test survives as a
  cross-owner check, reshaped to read the table.
- **D3 — leaf in its own nullable column, legacy rows rewritten in place, index `(metric, ts, leaf)`.**
  Measured on a `.backup` copy of the live 1.01 GB / 8,830,537-row corpus (2026-10-10): `ADD COLUMN`
  0.03 s, the split `UPDATE` 22.5 s, `CREATE INDEX (metric, ts, leaf)` 18.1 s, dropping the old index
  2.1 s. The total is about 43 s, once, in one oneshot tick. The unit is `Type=oneshot` with no
  `TimeoutStartSec`, so nothing kills it. Readers proceed under WAL, a killed migration rolls back
  whole, and a raising one fails store construction loudly (`sys.exit(1)`), as today. The file grows
  by the freed old-index pages (1.012 → 1.257 GB, about 19% free). The existing
  `_VACUUM_MIN_RECLAIMABLE_FRACTION` gate (10%) compacts it on its next daily evaluation. The new
  index adds no bytes net: the leaf text moves from `metric` to `leaf`. Query plans on the migrated
  copy: the dashboard's `metric IN (…) AND ts >= anchor` →
  `SEARCH … USING INDEX idx_samples_metric_ts_leaf (metric=? AND ts>?)`, its anchor `MAX(ts)` →
  covering; calibration's `metric = ? ORDER BY ts` → index-ordered, no temp B-tree; the tick-clock
  `COUNT(*)` → covering. Rejected: (a) not rewriting history, which keeps every parser alive for 30
  days and gives the corpus two encodings; (b) index `(metric, leaf, ts)`, which breaks the dashboard's
  `ts` range unless every reader adds `leaf IS NULL`, a dashboard logic change S22 owns; (c) leaf
  outside the index, which makes per-leaf window reads quadratic in leaf count.
- **D4 — admission drops one group, never the tick.** The run-tick check is explicit (not `assert`).
  It logs at ERROR with the group and the unexpected keys, and passes the other groups to
  `write_tick`. This is the same shape `__main__` uses for collection failures. A dropped group's
  absence is visible in the corpus (no rows for that tick) and in the journal.
- **D5 — the dashboard depends on the sampler package.** `load.py` says the dashboard "must not take
  a runtime dependency on the sampler package". That premise was a workspace-membership accident and
  is false today: `uv run --no-sync --project dashboard python -c 'import sampler.store'` succeeds
  against the root `.venv`. The G4 table rules that the dashboard imports the schema. Declaring the
  dependency makes that import non-accidental.
- **D6 — the calibration script's runtime never imports `sampler.store`.** It reads columns by SQL.
  Only its tests import `SCHEMA`, through the `_checkout` bootstrap, in β. Until β, α updates the
  test-local schema copy for `leaf`. Writer-side coverage stays in the sampler suite:
  `test_the_script_reads_a_corpus_THIS_STORE_wrote` drives the script over a DB the real store wrote,
  which exercises behaviour and is kept. Pointed at an unmigrated corpus (no `leaf` column), the
  script reports a named degradation (`corpus_not_migrated`) and exits 0, per its own degradation
  contract. It does not crash the ε gate in the seconds between merge and the next tick.
- **D7 — prose stance.** Every docstring a leaf rewrites follows the Comments stance in
  `docs/code-quality.md`. Benchmarks, rejected alternatives and dated host measurements go to the
  commit message, leaving at most a one-line pointer. PRD-decomposition letters (α, β, ε1) are replaced
  by module or task names. α does `store.py`; β does `metrics.py`, `sampler.py` and `__main__.py`.

## Contract

```python
# sampler/src/sampler/metrics.py — public surface after β (α introduces SampleKey/Samples)
class SampleKey(NamedTuple):
    metric: str
    leaf: str | None = None          # None for host-level metrics; never '' on a per-leaf metric

Samples = Mapping[SampleKey, float]

class GroupName(StrEnum):
    PSI = 'psi'; PROCESS = 'process'; LOAD = 'load'

@dataclass(frozen=True)
class MetricSpec:
    name: str                         # the bare stem; never contains ':'
    per_leaf: bool = False
    arm: str | None = None            # a shared.psi gate-arm name this metric records
    readability: 'MetricSpec | None' = None

@dataclass(frozen=True)
class MetricGroup:
    name: GroupName
    collect: Callable[[], Samples]    # the live, zero-argument collector
    metrics: tuple[MetricSpec, ...]
    windowed: bool                    # False: source is already windowed (PSI avg10) -> NULL window columns
    # derived: exact: frozenset[str]; per_leaf: frozenset[str]; admits(key: SampleKey) -> bool
    #   admits: (key.metric in exact and key.leaf is None) or (key.metric in per_leaf and bool(key.leaf))

METRIC_GROUPS: tuple[MetricGroup, ...]          # the ONE table
ARM_METRICS: Mapping[str, MetricSpec]           # derived from METRIC_GROUPS; replaces ARM_METRIC_STEMS
TICK_CLOCK: MetricSpec                          # the metric written on every completed load collection

# sampler/src/sampler/store.py
SCHEMA: str                                     # fresh-store DDL (public)
class LoadSampleStore:
    def __init__(self, db_path: Path, *, connect: Callable[..., sqlite3.Connection] = ...): ...
    def write_tick(self, ts: int, *, unwindowed: Samples, windowed: Samples) -> None: ...

# sampler/src/sampler/sampler.py
def run_tick(store, now: int, collected: Mapping[MetricGroup, Samples]) -> None: ...

# sampler/src/sampler/__main__.py
def main(groups: Sequence[MetricGroup] = METRIC_GROUPS) -> None: ...
```

Invariants: a fresh store and a migrated store have identical `PRAGMA table_info(samples)` and index
lists. After `_migrate`, no row's `metric` contains `:`. `_migrate` is idempotent and atomic. Every
metric key a collector emits is admitted by its own group. No module outside `sampler.metrics` spells
a metric name (§Design invariant). Exact field names beyond those above, and `run_tick`'s return value,
are tactical.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | store | open a legacy-shaped file (old DDL, `own_cpu_some10:a`, `own_read_ok:a`, PSI rows) | pre-leaf schema, old index | `leaf` column present; `metric='own_cpu_some10', leaf='a'`; PSI rows `leaf IS NULL`; only `idx_samples_metric_ts_leaf`; table_info equals a fresh store's |
| 2 | store | open twice / open migrated / open fresh | each state | no-op second time; identical shape all three |
| 3 | store | migration raises midway (injected `connect` failing the UPDATE) | legacy file | file unchanged (rolled back); constructor raises |
| 4 | store | trailing window per leaf | 2 leaves × 70 ticks | each leaf's window spans only its own rows |
| 5 | calibration | `read_series` over a corpus the real store wrote via `write_tick` with leaf keys | store-written DB | one series per leaf, paired with its own readability series; report rows unchanged vs the pre-change fixture expectations |
| 6 | calibration | unmigrated corpus | pre-leaf DB | exit 0, `corpus_not_migrated` named in report and JSON |
| 7 | calibration | run by a bare interpreter with only the `_checkout` binding (β) | no venv on `sys.path` | imports `sampler.metrics`, exit 0, `psutil` not loaded |
| 8 | dashboard | `/api/load` over a migrated store holding per-leaf rows | fixture via `SCHEMA` | 9 keys, populated; per-leaf rows not served |
| 9 | sampler | one group's collector emits an unadmitted key (β) | injected table | that group writes 0 rows, ERROR names group + key; other groups' rows present |
| 10 | sampler | add a `MetricSpec` to an injected table (β) | — | appears in run_tick admission, in `ARM_METRICS` if it carries an arm, and in the dashboard projection when its group is served |

## Design invariant (β performs the four-site lockstep edit)

Minted here: **INV-14 `sampler-vocabulary-one-table`**. The slug is the stable id. The number is the
next free one at main `ae3047ec3a` (INV-13 is the last). If another stream's INV lands first, β takes
the next free number and keeps the slug. The FILED table records the number that landed.

Sites (`scripts/tests/test_design_invariants_consistency.py` enforces them; see its `PINNED_SITES`):
`docs/legibility/design-invariants.md` (section, header dating paragraph, `## Fixtures` date list);
`docs/legibility/design-invariants-fixtures.md` (`INV-14-PRD`, `INV-14-CODE`, and a dated addendum walk
with its 2-row rehearsal table and the cumulative match count); `skills/prd/references/gates.md` (the
family-inventory row and the `inv-trigger-shapes` span); `CONTRIBUTING.md` gets nothing (it is pinned
as an absence). Draft text for the normative section:

> **Rule**: The load sampler's metric vocabulary has one declaring table, `sampler.metrics.METRIC_GROUPS`.
> The emitted names, the per-group admission sets, the gate-arm table, the calibration selectors and the
> dashboard's served metrics are projections computed from it. No other module spells a metric name, and
> a parity test between two spellings is not compliance.
>
> **Checkable design question(s)**: Does this feature add, rename or read a load-sampler metric, an
> admission rule, an arm-to-metric mapping, or a dashboard-served metric? Is it a projection of
> `METRIC_GROUPS`, or a new spelling that something must reconcile?
>
> **Evidence**: at `6551b75aba` the vocabulary had five homes and four reconciler tests, and the
> dashboard dropped unknown names silently (`fk-e86a10b9e217`, run `review-all-dark_factory-20261008`).
>
> **House pattern**: `METRIC_GROUPS`; the guard in `sampler/tests` (below).
>
> **Family boundary**: INV-5 `no-lockstep-duplication` governs duplicated logic in general and accepts a
> drift test as a house pattern; this invariant governs one named vocabulary and rejects the drift test
> as its enforcement. INV-9 `one-fact-one-home` governs world-facts written to records; this governs a
> code vocabulary.

Trigger shape for gates.md: *a load-sampler metric name, admission set, arm mapping or dashboard-served
metric spelled outside `sampler.metrics.METRIC_GROUPS` (`sampler-vocabulary-one-table`)?*

The guard has two parts. Both are tested against hand-written fixture inputs (INV-10). (1) Behavioural:
boundary row 10. A spec added to an injected table reaches every projection. (2) Structural backstop:
an AST scan of `sampler/src/sampler/*.py` (except the table's definitions), `dashboard/src/dashboard/
data/load.py` and `scripts/load-threshold-calibration.py` for string constants equal to a table metric
name, excluding docstrings. The scan flags a seeded fixture module and passes the post-β tree. Its
exclusions are *derived*: names that are also `shared.psi` arm names, because `runqueue_ratio`
is both. There are no hand-listed exemption entries, so INV-12 has nothing to own.

## Cross-PRD relationship

| Other PRD / stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| S25 scripts instruments | consumes | `scripts/_checkout.py` bootstrap binding `sampler/src` + `shared/src` for the system python | S25 | β depends on S25's leaf (id at phase B) |
| S22 dashboard readers | produces (to) | `sampler.store.SCHEMA` and the `KNOWN_METRICS` projection imported by `dashboard/src/dashboard/data/load.py` | S23 (this) | S22 makes no load.py logic change for this seam. Phase B checks S22's filed leaves for `load.py` / `test_load_data.py` in `metadata.files` and wires same-file serialization edges. |
| `plans/load-throttle-harmonisation-prd.md` (ε1 5219, ε2 5220) | produces (to) | the calibration script's corpus read and report shape | S23 changes the read path; the PRD's gates consume it unchanged | boundary rows 5–7 protect the gates |
| Phase-5 task 6643 (briefing facts, `fk-f5634bfcc909`) | produces (to) | the sampler known_gap "schema duplicated, no sync check" becomes false after α | 6643 | phase B: `add_dependency(id=6643, depends_on=α)` if 6643 is still pending, so the briefing records the post-α state; no briefing edit here |
| Phase-5 task 6644 (`fk-3e77aa904a0a`) | subsumed | sampler prose rehome | S23 β | phase B: cancel 6644 with `x_acceptance_reason` naming β if still pending; if claimed or done, β depends on it instead |

## Decomposition plan

Overlay bands respected: each leaf is in the 300–1,500 LOC target, ≤ 13 declared files. α and β
share `metrics.py`, `sampler.py`, `load.py`, the calibration script and its tests, so they are
serialized by a real edge (β depends on α).

**α — Sampler store: leaf as a column, public schema with additive migration, one connection seam**
(`task_kind=normal`, priority medium, est. ~1,100 LOC)
- Files: `sampler/src/sampler/store.py`, `sampler/src/sampler/metrics.py`, `sampler/src/sampler/sampler.py`,
  `sampler/tests/test_load_store.py`, `sampler/tests/test_load_sampler.py`, `sampler/tests/test_load_metrics.py`,
  `scripts/load-threshold-calibration.py`, `scripts/tests/test_load_threshold_calibration.py`,
  `dashboard/src/dashboard/data/load.py`, `dashboard/tests/test_load_data.py`, `dashboard/pyproject.toml`,
  `uv.lock`.
- Keys: `fk-ab2c587ff5a9`, `fk-d207683eef5a` (contributes the store.py half of `fk-fcefc2576467` /
  `fk-3e77aa904a0a` and the schema half of `fk-895f269d7bb5`; β closes those).
- Deps: none. Seams: produces `sampler.store.SCHEMA` (S23-owned) for S22 and S25 consumers.
- Observable signal: after merge, the live `data/load-samples.db` written by
  `dark-factory-load-sampler.timer` answers `SELECT COUNT(*) FROM samples WHERE metric GLOB '*:*'` = 0,
  and `SELECT COUNT(DISTINCT leaf) FROM samples WHERE metric='own_cpu_some10' AND ts > <merge ts>`
  equals the live `orchestrator-*.service` / `df-*.slice` leaf count. `scripts/load-threshold-calibration.py
  --arm own_cpu_some_avg10 --db data/load-samples.db` (no `--commit`) exits 0 with one per-leaf row per
  `SELECT DISTINCT leaf`. `curl /api/load` returns the 9 known metrics populated. In CI, boundary rows
  1–6 and 8.
- G7 notes: INV-13 (`readers-prove-their-producer`) holds. The read source moves for calibration
  only. Legacy per-leaf rows are *migrated* and host-level rows need no backfill. The producer is the
  live timer tick, and the signal reads its rows. INV-11 holds via D6's named degradation. INV-3:
  the migration acts on a feature-detect read inside its own transaction.

**β — One MetricGroup table, its projections, and INV-14 `sampler-vocabulary-one-table`**
(`task_kind=normal`, priority medium, est. ~1,200 LOC)
- Files: `sampler/src/sampler/metrics.py`, `sampler/src/sampler/sampler.py`, `sampler/src/sampler/__main__.py`,
  `sampler/tests/test_load_metrics.py`, `sampler/tests/test_load_sampler.py`,
  `sampler/tests/test_vocabulary_one_table.py` (new; the guard), `dashboard/src/dashboard/data/load.py`,
  `dashboard/tests/test_load_data.py`, `scripts/load-threshold-calibration.py`,
  `scripts/tests/test_load_threshold_calibration.py`, `docs/legibility/design-invariants.md`,
  `docs/legibility/design-invariants-fixtures.md`, `skills/prd/references/gates.md`.
- Keys: `fk-e86a10b9e217`, `fk-37f494b26355`, `fk-be1621618dee`, `fk-fcefc2576467`,
  `fk-895f269d7bb5`, `fk-3e77aa904a0a`.
- Deps: α; S25's `scripts/_checkout.py` leaf. Seams: consumes S25's bootstrap; produces the
  `METRIC_GROUPS` table (S23-owned) and the `KNOWN_METRICS` projection.
- Observable signal: `/usr/bin/python3 scripts/load-threshold-calibration.py --arm runqueue_ratio
  --db data/load-samples.db` exits 0. This is the ε1 gate's interpreter and invocation shape. Its
  selector and readability come from `sampler.metrics.ARM_METRICS`, and `ARM_METRIC_STEMS` and
  `TestCalibrationScriptArmTableLockstep` no longer exist. A live tick's journal line from
  `python -m sampler` names its groups from the table. `uv run --project shared pytest
  scripts/tests/test_design_invariants_consistency.py` passes with the new INV present in all pinned
  sites. The guard flags its seeded fixture module. Boundary rows 7, 9 and 10 pass in CI.
- G7 notes: INV-10 holds. The structural half is an AST check over code, not prose, and both halves
  are exercised against fixture inputs. INV-12 holds (derived exclusions, no entries). INV-4: the
  per-tick ERROR on a dropped group files nothing and matches the existing per-group degrade logging,
  so no storm escape is needed. INV-11 holds: `except psutil.Error` replaces the four bare
  `except Exception`, and a programming error now propagates to `__main__`'s loud per-group handler
  instead of reading as a fabricated 0.

## Mechanical tasks (contract §8)

- **6644** (`fk-3e77aa904a0a`, pending at authoring): subsumed by β. That key is superseded by
  `fk-fcefc2576467` in the run record, and β rewrites every docstring it names. Phase B: if still
  `pending`, `set_task_status(6644, cancelled)` with `metadata.x_acceptance_reason = "subsumed by
  S23 leaf β (<id>): prose rehome of sampler/src/sampler/*.py, plans/review-all-s23-sampler-metric-groups-prd.md"`;
  if claimed or done, β gains `add_dependency(β, 6644)` instead.
- **6643** (`fk-f5634bfcc909`): not subsumed. Ordered after α as in the seam table.

## Out of scope

- Changing any group's window mode. `own_cpu_some10` is a kernel avg10 recorded windowed today. That
  is a separate decision with no consumer asking for it.
- Dashboard logic beyond the schema import and the `KNOWN_METRICS` projection (S22).
- The `review/briefing.yaml` sampler section (task 6643).
- Retention, VACUUM policy and cleanup cadence (unchanged; D3 relies on the existing VACUUM gate).
- Serving per-leaf metrics on the dashboard.

## Open questions (tactical)

1. **Report label for per-leaf rows.** Keep rendering `own_cpu_some10:<leaf>` in the human report and
   JSON keys, so ε1/ε2 escalation readers see an unchanged format, or switch to a structured
   `{metric, leaf}` JSON entry? **Suggested:** structured in JSON, unchanged human label. Decide in α.
2. **`run_tick` return value** for `__main__`'s tick log (rows per group, or dropped groups).
   Decide in β.
3. **Where the moved measurements land.** Commit message (default) or a `plans/` note with a
   one-line pointer. Decide per leaf.
4. **Interpreter for boundary row 7 in CI.** The verify host's `/usr/bin/python3`, or `sys.executable -I`
   with a scrubbed `sys.path`. Match whatever S25's bootstrap test uses. Decide in β.
