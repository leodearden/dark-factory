# /prd seat brief — S21 Governed ledgers on one mixin and reader (agent mode)

Program: `/home/leo/src/solar-challenge-platform/plans/review-all-program-solar_challenge_platform-2026-10-14.md` — READ IT FIRST, in full (G4 seam map, rulings, shared conventions). Run `review-all-solar_challenge_platform-20261013`, record `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.json`, report `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.md`. Pinned tree `40a65566270dc43caf55e8f82232d61b3b558b0d`; main has moved — FRESHNESS CLAUSE: before building a task on any anchor below, `git diff 40a6556627..main -- <path>` and re-read the symbol.

## Your stream row (program doc)

| S21 | governed-ledgers-on-one-mixin-and-reader | An EffectiveDatedLedgerMixin in models/mixins.py declaring effective_from/effective_to, actor_user_id, note, created_at and building the CHECK and open-row partial index through declared_attr; one ledger reader in config/ (open row, current-at(instant), histor… | 5 keys: fk-87b83d5678da, fk-e20ab45a3ed0, fk-39e0169712da, fk-fc45dc17225a, fk-f0bc61012c6c | agent | M/low | 2 | S9 |

## Ruling (do not relitigate)

(uncontested stream: approved as proposed under R18)

## Remedy (synthesis)

An EffectiveDatedLedgerMixin in models/mixins.py declaring effective_from/effective_to, actor_user_id, note, created_at and building the CHECK and open-row partial index through declared_attr; one ledger reader in config/ (open row, current-at(instant), history) that the three stores and the three GET routes call; the stores check the actor's role so seed and routers stop restating it.

**Deletable:** The three copies of _close_prior_if_present, the open-row select and the half-open as-of predicate, the three router-side 'effective_to is None' derivations, about 35 lines in each of the four ledger models, coverage_store.py's 'apply in BOTH stores' amendment note, seed_golden_fleet's hand-picked actor ids.

**Seam candidates (G4):** config/ ledger reader (config owns; api/routers/config.py, optimiser and billing read)

## Finding keys this stream acts on (full statements)

### `fk-87b83d5678da` — medium — prior — `src/solar_challenge_platform/config/store.py::ConfigGovernanceStore` — tags h11, h10, comments, kind:duplication — class structural — source hotspot-survey-solar_challenge_platform-20261013
- statement: ConfigGovernanceStore (finance and battery), CoverageConfigStore and EndpointStore each define a byte-equivalent _close_prior_if_present: if current is None return, call require_strictly_after_open_row, set current.effective_to = effective_from, flush. Each also repeats the open-row select (effective_to IS NULL) and the half-open as-of predicate (effective_from <= ts AND (effective_to IS NULL OR effective_to > ts)). That makes four tables with three copies. Only the ordering guard was ever consolidated, and it took three fix commits (f290bc4, 29edb3c, aab07f0) to bring each store onto it. CoverageConfigStore's own docstring states the remaining risk: 'the close-and-flush … and the half-open as-of containment are still re-implemented beside ConfigGovernanceStore … a fix to either must be applied in BOTH stores'. It is actually three stores, and the duplication is justified by merge-contention directives ('NEW sibling module … avoid contending'), not by a design reason. The two-flush ordering that keeps the partial-unique open-row index satisfied is a load-bearing invariant restated in prose three times.
- proposal (input, not instruction): Add config/effective_dated.py::EffectiveDatedTable[Row], a small generic over an ORM model and an optional partition column (device_id, property_id or none). It provides append(row, effective_from), which runs the ordering guard, closes the prior row, flushes, inserts and flushes again; open_row(partition); and as_of(ts, partition). The three stores keep only their value validation and their view projection, and delegate. Deletable: the three _close_prior_if_present methods, the three as-of query bodies, the 'apply in BOTH' and 'mirrors ConfigGovernanceStore' docstrings, and the per-store two-flush rationale. The next governed table, such as billing-close config, is then one line, not a fourth copy. This would have prevented the f290bc4/29edb3c/aab07f0 guard-divergence class.

### `fk-e20ab45a3ed0` — medium — prior — `src/solar_challenge_platform/config/coverage_store.py::CoverageConfigStore` — tags h11, comments, kind:cross-module — class structural — source review-solar_challenge_platform-20261013
- statement: Heuristic 11 (SPOT): the effective-dating rules of the governed append-only ledgers still have three homes. This window extracted only the ordering check (config/contract.py::require_strictly_after_open_row). Close-the-open-row-and-flush is re-implemented in three stores (config/store.py:56, coverage_store.py:100, endpoint_store.py:72). The half-open as-of containment query, effective_from <= ts AND (effective_to IS NULL OR effective_to > ts), is re-implemented four times (store.py:207-210 and 251-254, coverage_store.py:180-183, endpoint_store.py:173-176), and so is the open-row query. These rules decide which finance rate, coverage floor and battery spec billing and dispatch use. Comments stance: the CoverageConfigStore docstring (coverage_store.py:89-94), rewritten in this window, says 'a fix to either must be applied in BOTH stores'. That undercounts: EndpointStore carries the same copies. It also justifies the copies by a 'NEW sibling module' directive (coverage_store.py:3-4) to avoid contending with pending P10 edits, and that P10 work landed long ago. Rationale is pointed at .task/plan.json (coverage_store.py:16), which is untracked. (Observed independently by 2 review seats.)
- proposal (input, not instruction): Give contract.py (or a small config/effective_dating.py) the remaining two rules: one close_prior(session, current, effective_from) and one as_of predicate builder over a model's effective_from/effective_to columns. Call them from all three stores and delete the 'apply in BOTH' note. This is mechanical per store but spans three modules.

### `fk-39e0169712da` — low — confirmed — `src/solar_challenge_platform/models/mixins.py` — tags h11, h3, h10 — class structural — source this run
- statement: The effective-dated, append-only governed ledger has one shape: effective_from and effective_to; CHECK effective_to IS NULL OR effective_to > effective_from; a partial unique index on the open row; actor_user_id FK to users; note; created_at defaulting to utcnow. That shape is written out by hand in four models: battery_config.py, finance_config.py, coverage_config.py and device_endpoint.py. Only the board_ref column has a shared home, BoardRefMixin in mixins.py. The store side of the same invariant does have one home, config/contract.py::require_strictly_after_open_row, used by every store. The database side (CHECK plus the open-row index), which is the redundant enforcement, is re-typed per table. The index differs in only one degree of freedom: partitioned by a column (property_id, device_id) or a fleet-wide singleton expression.
- measurement: copies of the governed-ledger shape = 4 models, each carrying all 7 pieces (CHECK text, open-row partial index, effective_from, effective_to, actor_user_id, note, created_at) (rule: per file under models/: grep -c for the CHECK text, sqlite_where=text("effective_to IS NULL"), and the five Mapped column names; counted only files where all 7 occur)
- measurement: shared mechanism for it = 0 in models/ (BoardRefMixin covers board_ref only); 1 in config/contract.py for the store-side ordering rule (rule: classes and functions in models/mixins.py and config/contract.py)
- next change: Add a fifth governed ledger (e.g. a governed tariff or renewal-policy table), or tighten the window rule, e.g. require effective_from to be UTC half-hour aligned. The agent copies about 35 lines from a sibling model and picks the right index form. An omitted CHECK or open-row index leaves only the store-side check, so the database silently accepts two open rows from any non-store writer. Nothing generic tests the shape; per-table tests catch it only if the new table's author writes them.
- proposal (input, not instruction): Add an EffectiveDatedLedgerMixin to mixins.py that declares the columns and builds the CHECK and open-row index through __table_args__ via declared_attr, parameterised by partition column or singleton. The four models then declare only their payload columns.

### `fk-fc45dc17225a` — low — confirmed — `src/solar_challenge_platform/config/store.py::ConfigGovernanceStore` — tags h9, h11 — class structural — source this run
- statement: The governed-ledger stores expose only writes, current_* and *_as_of. ConfigGovernanceStore also exposes audit(). None of them exposes a per-ledger history read.

So api/routers/config.py's three GET routes query the ORM ledgers directly:
- get_finance selects FleetFinanceConfig (:109).
- get_battery selects HomeBatteryConfig (:213).
- get_coverage selects FleetCoverageConfig (:273).

Each route re-derives 'current' inline as the row with effective_to IS NULL, beside the stores' own current_finance, current_battery and current_coverage. Finance and battery history is thus read by two independent paths: these router selects and ConfigGovernanceStore.audit.

The filed fleet-optimiser _finance_config_from_store finding records that the open row can be future-dated. In that case the router's 'current' and the store's finance_as_of(now) name different values, and nothing ties the two definitions together.
- measurement: API routes selecting governed-ledger ORM rows directly = 3 (api/routers/config.py:109, :213, :273) (rule: select( calls on FleetFinanceConfig/HomeBatteryConfig/FleetCoverageConfig in api/routers/config.py)
- measurement: inline 'current = effective_to is None' derivations outside config/ = 3 (one per GET route) (rule: grep 'effective_to is None' in api/routers/config.py)
- measurement: independent history-read paths for finance and battery = 2 (router GETs; ConfigGovernanceStore.audit) (rule: read of config/store.py::audit against api/routers/config.py)
- next change: Effective-dating semantics will change next, for example 'current' meaning effective at now rather than the open row, or the filed fix that consolidates the stores' open-row predicate. That change must find the three router derivations as well as the three stores. If the router copies are missed, the admin console's Current card and the value billing and the optimiser actually apply diverge silently.
- proposal (input, not instruction): Add history() (and current-at(now)) to each governed store, or one shared ledger reader in config/. Have the three routes call it, so the open-row and as-of predicates have only store-side homes.

### `fk-f0bc61012c6c` — low — prior — `src/solar_challenge_platform/runtime/seed.py::seed_golden_fleet` — tags h11, h10, kind:duplication — class structural — source hotspot-survey-solar_challenge_platform-20261013
- statement: The governed stores accept any actor_user_id: set_finance, set_battery, set_coverage and set_endpoint never check the actor's role. The authority rule (ADMIN writes finance, battery and coverage; ADMIN records consent; OPERATOR registers endpoints) exists only as require_role(Role.ADMIN) dependencies in api/routers/config.py and the consent router. It is restated in prose and by hand in runtime/seed.py: seed_users' docstring says 'The ADMIN user actors the governed config writes … the role POST /config/finance … require', and seed_golden_fleet picks admin.id or operator.id for each write. The two copies have already disagreed twice. scp-seed recorded fleet finance and consent grants under the OPERATOR, a role the API refuses, which tasks 543 and 590 fixed by editing the seed rather than the rule.
- proposal (input, not instruction): Add one table, config/authority.py::GOVERNED_WRITE_ROLES, mapping each governed write kind to the roles allowed to author it. Have the store write methods take a Principal (or actor id plus role) and enforce the table at write time, raising an error that names the write, the role and the allowed set. Derive the router dependencies from the same table, e.g. require_role(*GOVERNED_WRITE_ROLES[FINANCE]). The seed then cannot write a row the API would refuse. Deletable: the role prose in seed_users and the per-call actor selection in seed_golden_fleet, which becomes 'the principal the table names'. This would have prevented 543 and 590.

## Mechanical filings rows for these keys (verbatim from the report; re-run contract §8 step 1 before filing regardless)

| key | sources | dedup step | ticket/task | disposition |
|---|---|---|---|---|
| — | — | — | — | none of this stream's keys was filed mechanically |

## Seam rows naming this stream

| config/ ledger reader (config owns; api/routers/config.py, optimiser and billing read) | S21 | per the stream's PRD; every other stream imports downward |

## What you return (structured block, no commits, no edits to the program doc or another stream's PRD)

AUTHOR pass: write `docs/prds/review-all-<slug>.md` and `docs/prds/review-all-<slug>.capability-manifest.yaml` (this project's convention) under the /prd skill's gates G1-G6 with this brief as the folded-in context; return {prd_path, manifest_path, open_questions (only those the brief and rulings cannot answer), seam_dependencies_on_other_streams, cross_reference_lines}. Do not decompose or file anything in the author pass.
DECOMPOSE pass (only when the lead sends the filed ids you depend on): decompose under planning_mode, stamp metadata.x_finding_key (list) / x_finding_run / source=review-all on every task, add your high/critical leaves as dependencies of the trigger-chain gate id the lead gives you, run commit_planning on your own batch (held leaves excluded), write the stamped sidecar, return your FILED row and cross-reference lines.
