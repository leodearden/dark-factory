# /prd seat brief — S18 Declared package strata (agent mode)

Program: `/home/leo/src/solar-challenge-platform/plans/review-all-program-solar_challenge_platform-2026-10-14.md` — READ IT FIRST, in full (G4 seam map, rulings, shared conventions). Run `review-all-solar_challenge_platform-20261013`, record `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.json`, report `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.md`. Pinned tree `40a65566270dc43caf55e8f82232d61b3b558b0d`; main has moved — FRESHNESS CLAUSE: before building a task on any anchor below, `git diff 40a6556627..main -- <path>` and re-read the symbol.

## Your stream row (program doc)

| S18 | declared-package-strata | A tracked doc states the strata (settings/db/models < timeseries/consent/auth/config/integrations < control/ingestion/billing/optimiser/simulator < ops/portal/schemas/messaging < api/runtime < validation/docsgen/ci/instance) and one AST test asserts every pack… | 3 keys: fk-b53afebe6b7c, fk-2628179f5885, fk-eaa6133b6f56 | agent | S/low | 1 | — |

## Ruling (do not relitigate)

R18: Approved as proposed, each as an agent-mode /prd stream: Device last-seen has one reader; Time grid and UTC home; Simulated fleet identity; Dispatch composition out of control/; Settings domains and deployment profile; Governed ledgers on one mixin and reader; Money quantum and wire decimal grammar; Ops reads through public seams; Declared package strata.

Options considered: approve all nine; approve with exclusions
Consequence that decided it: one option each, confirmed findings, size S/M, low/medium risk; the remedies name what becomes deletable.

## Remedy (synthesis)

A tracked doc states the strata (settings/db/models < timeseries/consent/auth/config/integrations < control/ingestion/billing/optimiser/simulator < ops/portal/schemas/messaging < api/runtime < validation/docsgen/ci/instance) and one AST test asserts every package edge points down, with a shrinking allowlist naming the five known violations (settings→consent, settings→control, settings→simulator, control→optimiser, validation→runtime) by finding; BrokerConfig moves to a leaf ingestion module so the simulator and the harness import it without loading db.session or fastapi; the private-names gate is extended to attribute access on objects whose class is defined in src, with the same task-mapped allowlist.

**Deletable:** The 6 prose layering claims in 4 files, each allowlist entry as its stream lands, the simulator's 47 unneeded module loads.

**Seam candidates (G4):** the strata doc and its test (repo owns)

## Finding keys this stream acts on (full statements)

### `fk-b53afebe6b7c` — medium — confirmed — `slug:no-declared-package-strata` — tags h9, h10, comments — class structural — source this run
- statement: No declared order of strata exists for the 24 top-level packages, and nothing checks one. The package graph is therefore two strongly connected components: a 12-package SCC {auth, config, consent, control, db, ingestion, integrations, models, optimiser, settings, simulator, timeseries} and a 3-package SCC {instance, runtime, validation}. Both exist at module scope, with deferred imports adding no package edge.

Five package edges dissolve both: settings→consent, settings→control, settings→simulator, control→optimiser and validation→runtime. Each has been found and filed separately, by hand, in earlier reviews (settings default_optimiser_interval_seconds h9, control/service from_settings h9, validation/container_images h9).

The layering rules that do exist live only as prose: 'control never imports ops' (control/dispatch_liveness.py:16 and :198, ops/contract.py:181 and :363), 'config/* never imports billing/*' (billing/config.py:178) and 'clean layering' (models/control_command_log.py:26). The project's gates (pytest, mypy) contain only two import-boundary tests, each scoped to one module or one package: tests/control/test_provenance.py checks that control.provenance loads no sqlalchemy, and tests/test_entrypoints.py::test_roles_modules_import_only_common_from_their_package covers runtime.roles internals. A new upward edge is therefore caught only if a reviewer happens to look.
- measurement: package SCCs = 2: 12 packages (auth, config, consent, control, db, ingestion, integrations, models, optimiser, settings, simulator, timeseries) and 3 packages (instance, runtime, validation); identical with or without deferred edges (rule: Tarjan SCC over import_graph.json edges collapsed to the first segment under solar_challenge_platform (scratchpad cross/layering/scc.py, modes top and all))
- measurement: minimal edge set dissolving every package SCC = 5 package edges: settings→consent, settings→control, settings→simulator, control→optimiser, validation→runtime (alternatively instance→validation + runtime→validation in place of the last) (rule: Remove candidate upward edges and recompute SCCs until none remain; each removed edge was checked against prior_open.json, and all 5 are already filed individually)
- measurement: prose layering claims in src = 6 in 4 files (control/dispatch_liveness.py x2, ops/contract.py x2, models/control_command_log.py, billing/config.py) (rule: grep -rn 'never imports|must not import|layering' over src/, keeping statements about package dependency direction (excluding 'src never imports tests'))
- measurement: tests or config enforcing inter-package order = 0 (2 import-boundary tests exist, scoped to control.provenance and to runtime.roles internals; no import-linter/tach section in pyproject.toml) (rule: grep of tests for sys.modules/ast import checks and of pyproject.toml for importlinter/tach/layers)
- next change: The next agent adds a Settings default sourced from a subsystem constant, imports a subsystem type into db/ or models/, or composes a peer inside a subsystem's from_settings, as each of the five filed edges did. To get it right the agent must know the intended stratum order, which exists nowhere it can Read. Today nothing catches a wrong version: mypy and pytest pass, and only a later whole-repo review notices the new SCC member.
- proposal (input, not instruction): Write the strata once in a tracked doc, roughly: settings/db/models < timeseries/consent/auth/config/integrations < control/ingestion/billing/optimiser/simulator < ops/portal/schemas/messaging < api/runtime < validation/docsgen/ci/instance. Add one AST test that asserts every package edge points down, reusing test_entrypoints.py's import_targets/package_module_reached helpers. Give it an explicit allowlist for the 5 known violations, each pointing at its finding, so the list can only shrink.

### `fk-2628179f5885` — low — confirmed — `src/solar_challenge_platform/ingestion/service.py::BrokerConfig` — tags h9, h13 — class mechanical — source this run
- statement: BrokerConfig is a five-field frozen value and is also the shared MQTT connection shape. It is defined in ingestion/service.py, the ingestion composition module, which imports consent.enforcement, db.session, timeseries.factory and aiomqtt at module level.

simulator/service.py imports BrokerConfig, and nothing else, from that module at :58. As a result, importing the simulator process loads 65 platform modules across 9 packages. These include consent.gate, which loads fastapi via auth.dependencies, and db.session, whose import builds a SQLAlchemy Engine. The simulator uses neither.

With BrokerConfig served from a leaf module, the same import loads 18 modules across 6 packages, with no db.session and no fastapi. The same misplacement is one reason validation/ingest_session.py restates the broker session parameters instead of sharing the type (validation-harness.4).
- measurement: modules loaded by importing simulator.service = 65 platform modules, 9 packages (auth, consent, control, db, ingestion, models, settings, simulator, timeseries); db.session.engine is an Engine; fastapi loaded (rule: python -I probe with the review tree's src first on sys.path: import solar_challenge_platform.simulator.service, then count sys.modules entries under solar_challenge_platform.)
- measurement: same import with BrokerConfig from a leaf stub = 18 modules, 6 packages; db.session and fastapi not loaded (rule: Same probe with sys.modules['solar_challenge_platform.ingestion.service'] replaced by a module defining only an identical frozen BrokerConfig)
- measurement: names simulator.service takes from ingestion.service = 1 (BrokerConfig) (rule: grep ingestion over simulator/service.py)
- next change: The next change to broker connection parameters, for example the TLS that validation-harness.4 found missing, or a client-id policy, must be made in the ingestion composition module. Moving BrokerConfig to a leaf (e.g. ingestion/broker.py) means updating 2 importers. A wrong version, such as validation adding a third restatement, is not caught by any test.
- proposal (input, not instruction): Move BrokerConfig to a leaf module that imports nothing from the platform. Have ingestion.service, simulator.service and validation/ingest_session.py import it from there.

### `fk-eaa6133b6f56` — low — weakened — `slug:instance-private-reaches-ungated` — tags h7, tests, h9 — class structural — source this run
- statement: tests/test_private_names.py enforces that no module reaches another module's private module-level names; 17 exceptions sit in _AWAITING_PROMOTION, each mapped to a task. It cannot see private instance attributes. An AST scan of tests/ finds 38 `<obj>._name` reaches in 12 files, with the namedtuple API and self/cls excluded. 27 of them, in 8 files, read or call private state of src classes:
- ControlPlaneService._optimiser / _optimiser_interval / _log_dispatch_outcomes: 9 (test_dispatch_binding 4, tests/runtime/test_composition 5)
- Subscriber._consent_guard: 6
- ControlPlane._feedback_sink: 4
- OpsStreamBroadcaster._fan_out / _subscribers: 4
- SimHomeTelemetrySource._topic / _payload: 2
- FakeStripeInvoicing._invoices: 1
- ModbusTcpTransport._client: 1. This one is needed because ModbusTcpTransport.connect discards the client's connect result.

The gate reports 0 of the 27. The seams these tests reach through (what a service plans with, consent drop counts, the broadcaster's fan-out, transport connectivity) have no public read, so the only behavioural check of wiring is a private read that no gate tracks or retires.
- measurement: test reaches into private instance attributes = 38 in 12 files; 27 into src classes in 8 files; 3 into SQLAlchemy pool internals (tests/db/test_session.py); 8 test-local (attributes tests attach to their own doubles or TestClient) (rule: AST walk of tests/**/*.py for ast.Attribute with a leading-underscore attr, excluding self/cls receivers, dunder names and _asdict/_replace/_fields/_make; owner classified by reading each site)
- measurement: of those, caught by tests/test_private_names.py = 0 (rule: The gate's private_reaches resolves only imported module names and dotted strings, per its docstring and _reaches_at)
- measurement: module-level private reaches the gate does track = 17 names awaiting promotion (tasks 228, 242, 285, 351, 354) (rule: Entries of _AWAITING_PROMOTION in tests/test_private_names.py)
- next change: An agent reworking ControlPlaneService, Subscriber or OpsStreamBroadcaster renames a private attribute, and wiring tests break for reasons unrelated to behaviour. An agent adding a wiring test copies the svc._x pattern because no public seam exists, and no gate flags it. A correct refactor is expensive and a pinned-implementation test passes review.
- proposal (input, not instruction): Give each reached seam a public read: what the dispatch service plans with, the ingest consent-drop counts in IngestTick, the broadcaster's subscriber set, and the transport's connect result. Then extend the private-names gate to attribute access on objects whose class is defined in src, with the same task-mapped allowlist.
- WEAKENED by the skeptic: Wrote and ran an AST walk of tests/**/*.py for underscore Attribute nodes (excluding self/cls, dunders and namedtuple API). Opened the owner of each hit: test_publisher.py, forecast_harness.py, test_dd.py, test_p3_sim_conformance.py, test_ops_api.py, test_composition.py, test_dispatch_binding.py. Read tests/test_private_names.py. — The total of 38 reaches in 12 files reproduces, as do 3 pool-internal reaches and 17 _AWAITING_PROMOTION entries (tasks 228, 242, 285, 351, 354). The gate is module-name based and cannot see instance attributes. The src-class split is off, though. The 2 'SimHomeTelemetrySource._topic/_payload' reaches (tests/simulator/test_publisher.py:313-314) are on _FakeTelemetrySource, a test-local fake whose __init__ assigns those attributes (test_publisher.py:179-191). src/simulator has no _topic or _payload. So it is 25 reaches into src classes in 7 files, not 27 in 8, and the test-local set is 10, not 8. The remaining named groups check out: ControlPlaneService 9, Subscriber._consent_guard 6, ControlPlane._feedback_sink 4, OpsStreamBroadcaster 4, FakeStripeInvoicing 1, ModbusTcpTransport._client 1. The finding's thesis survives, with one named example misattributed.

## Mechanical filings rows for these keys (verbatim from the report; re-run contract §8 step 1 before filing regardless)

| key | sources | dedup step | ticket/task | disposition |
|---|---|---|---|---|
| `fk-2628179f5885` | this run | filed (curator combined) | 451 | filed:451 |

## Seam rows naming this stream

| the strata doc and its test (repo owns) | S18 | per the stream's PRD; every other stream imports downward |

## What you return (structured block, no commits, no edits to the program doc or another stream's PRD)

AUTHOR pass: write `docs/prds/review-all-<slug>.md` and `docs/prds/review-all-<slug>.capability-manifest.yaml` (this project's convention) under the /prd skill's gates G1-G6 with this brief as the folded-in context; return {prd_path, manifest_path, open_questions (only those the brief and rulings cannot answer), seam_dependencies_on_other_streams, cross_reference_lines}. Do not decompose or file anything in the author pass.
DECOMPOSE pass (only when the lead sends the filed ids you depend on): decompose under planning_mode, stamp metadata.x_finding_key (list) / x_finding_run / source=review-all on every task, add your high/critical leaves as dependencies of the trigger-chain gate id the lead gives you, run commit_planning on your own batch (held leaves excluded), write the stamped sidecar, return your FILED row and cross-reference lines.
