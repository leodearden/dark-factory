# /prd seat brief — S24 Settings domains and deployment profile (agent mode)

Program: `/home/leo/src/solar-challenge-platform/plans/review-all-program-solar_challenge_platform-2026-10-14.md` — READ IT FIRST, in full (G4 seam map, rulings, shared conventions). Run `review-all-solar_challenge_platform-20261013`, record `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.json`, report `/home/leo/src/solar-challenge-platform/plans/review-all-solar_challenge_platform-20261013.md`. Pinned tree `40a65566270dc43caf55e8f82232d61b3b558b0d`; main has moved — FRESHNESS CLAUSE: before building a task on any anchor below, `git diff 40a6556627..main -- <path>` and re-read the symbol.

## Your stream row (program doc)

| S24 | settings-domains-and-deployment-profile | Every numeric Settings field carries its domain in its type as the one mechanism, with consumers' own checks as the redundant second point; Settings is frozen with validate_assignment and scp-dispatch passes overrides at construction; one table of hermetic-onl… | 6 keys: fk-5e71d4d3f257, fk-fa9f275a8a19, fk-ac72f1158e90, fk-e1a712c02688, fk-7d58a82ae8df, fk-600b3d557b32 | agent | M/medium | 1 | — |

## Ruling (do not relitigate)

(uncontested stream: approved as proposed under R18)

## Remedy (synthesis)

Every numeric Settings field carries its domain in its type as the one mechanism, with consumers' own checks as the redundant second point; Settings is frozen with validate_assignment and scp-dispatch passes overrides at construction; one table of hermetic-only switches with the value production refuses, iterated by a single is_production validator and read by the instance render, which refuses unknown app keys, derives key names from Settings and checks MQTT_HOST/MQTT_PORT against the rendered broker as it already checks DATABASE_URL; the topic filter derives from ingestion.wire's TOPIC_PREFIX.

**Deletable:** run_dispatch's post-construction mutation and its own FEEDBACK_HEARTBEAT cross-check, two of the three enforcement styles, environment.py's pass-through of 6 of 8 app keys, the three broker-port literals, the os.environ reads outside Settings (LOG_LEVEL, SCP_STATIC_DIR, SCP_PORTAL_STATIC_DIR), the restated topic prefix.

**Seam candidates (G4):** the hermetic-switch table (settings owns; instance/render/environment.py reads)

## Finding keys this stream acts on (full statements)

### `fk-5e71d4d3f257` — medium — confirmed — `src/solar_challenge_platform/settings.py::Settings` — tags h10 — class structural — source this run
- statement: Of Settings' 34 numeric fields, 8 carry their domain in the type (PositiveInt/NonNegativeInt), 1 has a field validator (ingestion_subscribe_qos) and 2 have a cross-field validator (the consent TTL pair). sim_seed is legitimately any int. The other 22 are bare int/float: durations, depths, limits, ports and intervals where zero or negative is meaningless. Some consumers re-check: ModbusIoBound refuses a non-positive timeout, Pacing refuses a negative sim_time_scale, and ops/contract.py::dispatch_staleness_bound refuses interval <= 0. Others do not. INGESTION_BUFFER_MAX_DEPTH=0 boots cleanly, and ingestion/buffer.py::StoreForwardBuffer.offer then evicts every row it is offered (`del self._rows[:overflow]`) and counts it dropped. That is total silent telemetry loss on the ingest path, the fail-safe class the platform ranks first. OPS_STREAM_INTERVAL_SECONDS=0 turns the SSE broadcaster into a continuous DB re-query loop. One kind of invariant, a field's domain, is enforced in three styles (type, validator, consumer) and for most fields not at all.
- measurement: numeric Settings fields by enforcement = 34 total; 8 typed bound, 1 field validator, 2 cross-field validator, 1 unbounded-by-design (sim_seed), 22 unbounded (rule: Settings fields annotated int/float/PositiveInt/NonNegativeInt in settings.py (bool excluded), classified by where a bound is stated in settings.py)
- measurement: consumer-side re-checks of an unbounded field = 3 (control/transport/modbus.py::ModbusIoBound timeout, simulator Pacing time_scale, ops/contract.py dispatch_staleness_bound interval) (rule: raise ValueError on a non-positive value in the constructor or function that receives the field, found by tracing each unbounded field's consumers)
- measurement: unbounded fields whose bad value boots and degrades silently = 2 confirmed by reading (ingestion_buffer_max_depth <= 0 drops every row; ops_stream_interval_seconds <= 0 busy-loops) (rule: Traced from the field to its consumer's first use with value 0)
- next change: Add a tuning knob (ingest flush concurrency, a new ops window) or lower a default. The agent picks one of three enforcement styles by local example, and nothing at boot catches a missing bound. The /review fail-safe yardstick only notices once telemetry is already gone.
- proposal (input, not instruction): Put every field's domain in its type (PositiveFloat, PositiveInt, conint(ge=..)) as the one mechanism. Keep the consumers' own checks as the redundant second point (heuristic 10's 'at construction and at use').

### `fk-fa9f275a8a19` — low — prior — `src/solar_challenge_platform/runtime/roles/dispatch.py::run_dispatch` — tags h8, h10, kind:cross-module — class mechanical — source review-solar_challenge_platform-20261013
- statement: Heuristic 8, prefer immutable data: run_dispatch builds a validated Settings and then mutates it before handing it to ControlPlaneService.from_settings: settings.scp_clock_override_iso / scp_allow_clock_override (dispatch.py:104-105) and dispatch_shard_index / dispatch_shard_count (:107-108). Settings' model_config sets no validate_assignment (settings.py model_config), so the assignments bypass its validators (_reject_shard_index_outside_shard_count and the guarded clock-override contract that resolved_clock_override calls 'the single choke point'). It is the only post-construction mutation of Settings in src/; the shard values happen to be pre-validated by FleetShard.parse, so the invariant holds today by a second, separate check rather than by Settings' own (heuristic 10). (Observed independently by 2 review seats.)
- proposal (input, not instruction): Build the overridden Settings as a new validated value (Settings.model_validate({**settings.model_dump(), ...}) or Settings(**overrides)) so the role's flags pass the same validators as the environment. Mechanical.

### `fk-ac72f1158e90` — medium — confirmed — `src/solar_challenge_platform/settings.py::Settings.is_production` — tags h3, h10, h11 — class structural — source this run
- statement: Hermetic versus production is one axis, but it is spelled as independent per-feature switches. The one production predicate refuses only two of them.

Settings defaults are the hermetic values: environment='development', database_url='sqlite:///:memory:', mqtt_host='localhost', mqtt_tls=False, ingestion_store_backend='memory', dispatch_registry_from_db=False and the dev jwt_secret. A production deployment flips them one key at a time.

Settings.is_production is documented as 'the one predicate every production guard keys on'. Yet only two Settings rules key on it: the JWT sentinel/length validator and telemetry_store_refusal (re-checked in timeseries/factory.py). It has one other reader, scp-seed's --yes guard.

The test and validation opt-outs each carry their own bespoke guard or none:
- INGESTION_CONSENT_ENFORCED=false, which tests/conftest.py sets for non-postgres tests, boots a production ingest with no consent or roster check.
- The clock override has a private two-flag guard, and scp-dispatch --clock-override sets both flags in-process (runtime/roles/dispatch.py:104-105).
- Provider fakes have no setting at all (billing.1).

The production profile itself has three homes and no shared mechanism:
- the 2 Settings validators;
- instance/render/environment.py, which checks DATABASE_URL's host and the shard count;
- deploy/instances/example/instance.env, which restates 5 production values by hand (ENVIRONMENT, DATABASE_URL, DISPATCH_REGISTRY_FROM_DB, INGESTION_STORE_BACKEND, MQTT_TLS).
- measurement: Settings fields defaulting to the hermetic value / refused under ENVIRONMENT=production = 7 / 2 (refused: jwt_secret, ingestion_store_backend; not refused: database_url sqlite, mqtt_host localhost, mqtt_tls False, dispatch_registry_from_db False, plus environment itself) (rule: Settings fields in settings.py whose default is a dev/in-process value that deploy/instances/example/instance.env overrides or that a validator refuses; refusal = a validator or factory check that reads is_production)
- measurement: readers of is_production = 3 rules (jwt validator, telemetry_store_refusal via validator and factory, runtime/roles/seed.py --yes guard) (rule: grep -rn 'is_production' over src, excluding the property definition and docstrings)
- measurement: test/validation opt-outs a production Settings accepts = 3 (ingestion_consent_enforced=false; scp_allow_clock_override with scp_clock_override_iso, also set in-process by scp-dispatch --clock-override; provider fakes, which have no setting) (rule: switches documented in settings.py or role docstrings as test/validation-only, checked for any production refusal)
- measurement: homes of the production profile = 3 (settings.py validators: 2 keys; instance/render/environment.py: DATABASE_URL host, DISPATCH_SHARD_COUNT; deploy/instances/example/instance.env: 5 production values restated by hand); 0 shared table (rule: read of settings.py, instance/render/environment.py and the example instance.env at 40a6556)
- next change: The plausible next change is the billing.1 fix, a provider-mode setting, or any new hermetic-only switch. The agent must know to write a bespoke production refusal in settings.py, to teach instance/render/environment.py separately, and to update the example instance.env. Nothing catches an omission: a production Settings with the new switch at its hermetic value constructs cleanly, and the render passes unparsed keys through (operations-assurance.4).
- proposal (input, not instruction): Make the deployment profile a typed axis. Keep one table of hermetic-only switches, each with the value production refuses. A single Settings validator iterates that table under is_production, and the instance render reads the same table, so a new switch costs one table row.

### `fk-e1a712c02688` — low — prior — `src/solar_challenge_platform/settings.py::Settings.model_config` — tags h10, h11, kind:cross-module — class structural — source review-solar_challenge_platform-20261013
- statement: Heuristic 10: the configuration contract between the new instance render and the roles is unenforced. Under req 15, adding a co-op is meant to be configuration alone. Settings uses extra='ignore' (settings.py:82-89), so every role silently drops any environment key it does not define. A probe confirmed it: DISPATCH_SHARD_CONUT=4 left dispatch_shard_count at 1 with no error. scp-deploy render (instance/render/environment.py, task 443, this window) validates DATABASE_URL, the shard count and quoting, but not key names against Settings fields. It also writes two keys into every role's environment that no Settings field defines and no role reads: HEALTH_COMPONENTS (environment.py:25, 99) and INSTANCE_NAME. So a misspelt instance.env key, the operator's only configuration surface, is indistinguishable from a key meant for later and reaches no one.
- proposal (input, not instruction): Have the render check every non-DEPLOY_ instance.env key against Settings.model_fields, plus an explicit set of render-owned keys, refusing an unknown name with its closest field. Or switch Settings to extra='forbid' once HEALTH_COMPONENTS and INSTANCE_NAME are real fields. Which side owns the check is a design choice.

### `fk-7d58a82ae8df` — low — prior — `src/solar_challenge_platform/settings.py::Settings` — tags h11, h12, kind:latent-bug — class structural — source hotspot-survey-solar_challenge_platform-20261013
- statement: Settings.ingestion_topic_filter defaults to the literal 'sc/v1/tel/#'. ingestion/wire.py defines TOPIC_PREFIX = 'sc/v1/tel' separately, and topic parsing requires exactly that 3-segment prefix. The operator can override INGESTION_TOPIC_FILTER, and no validator checks that the filter sits under TOPIC_PREFIX. Setting it to, say, 'sc/v2/tel/#' or '+/v1/tel/#' gives a subscriber that boots cleanly, connects and subscribes, while every message is rejected as RejectReason.BAD_TOPIC and counted rather than refused at startup. That is silent data loss on the ingest path, which the platform's fail-safe priority forbids.
- proposal (input, not instruction): Derive the default from wire.py, as f"{TOPIC_PREFIX}/#". Add a field validator that refuses a filter not rooted at TOPIC_PREFIX, naming both values, or drop the knob and have BrokerConfig/Subscriber take the filter from wire.py. Deletable: the duplicated literal.

### `fk-600b3d557b32` — medium — confirmed — `src/solar_challenge_platform/instance/render/environment.py::InstanceEnvironment` — tags h10, h12, h11 — class structural — source this run
- statement: Heuristic 10 (uniformly enforced) and heuristic 12 (parse at the boundary): the instance file is where an operator's role configuration enters, but InstanceEnvironment.read parses only 2 of the example's 8 app keys (DATABASE_URL and DISPATCH_SHARD_COUNT). It passes every other key through verbatim to a Settings whose model_config has extra='ignore'.

The two pieces of rendered infrastructure the roles connect to are treated differently:
- _database refuses a DATABASE_URL whose host is not ServiceName.TIMESCALE.
- Nothing checks that MQTT_HOST names ServiceName.MOSQUITTO, or that MQTT_PORT equals the broker's container port compose.py::_MQTTS_PORT.

The 8883 broker port is stated three times: spec.py's mqtts_port default, compose.py::_MQTTS_PORT, and the example's MQTT_PORT. A render with MQTT_HOST=localhost and a misspelt MQTT_PROT=8883 succeeds (probed), and ingest would then silently connect to localhost:1883, the Settings default the README itself warns about. Heuristic 11: the render's key names (DATABASE_URL_KEY, SHARD_COUNT_KEY) restate Settings field names as literals, so nothing ties them to Settings.
- measurement: app keys the render parses = 2 of 8 (DATABASE_URL, DISPATCH_SHARD_COUNT; not ENVIRONMENT, DISPATCH_REGISTRY_FROM_DB, INGESTION_STORE_BACKEND, MQTT_HOST, MQTT_PORT, MQTT_TLS) (rule: non-DEPLOY_, non-INSTANCE_NAME keys in deploy/instances/example/instance.env, against the keys environment.py reads by name)
- measurement: probe: misdirected broker config = render succeeded; the rendered instance.env carried MQTT_HOST='localhost' and MQTT_PROT='8883' (rule: render_instance on a scratch copy of the example instance.env with MQTT_PORT renamed MQTT_PROT and MQTT_HOST=localhost, console_scripts from pyproject)
- measurement: broker port literal sites = 3 (spec.py mqtts_port default 8883; compose.py::_MQTTS_PORT; deploy/instances/example/instance.env MQTT_PORT=8883) (rule: grep 8883 over src/solar_challenge_platform/instance and deploy/)
- next change: The plausible change is the pending broker-TLS or reverse-proxy work. It changes compose.py::_MQTTS_PORT or the broker's service name, or adds a broker credential key. The agent must also know that every instance file's MQTT_HOST and MQTT_PORT have to change by hand, and that Settings drops unknown keys. Nothing catches a stale or misspelt key. The render accepts it, and the role falls back to defaults at runtime.
- proposal (input, not instruction): Validate the app half of the instance file against Settings' fields at render time: refuse unknown keys and derive the key names from Settings. Check the broker endpoint the same way as the DB endpoint: MQTT_HOST must name ServiceName.MOSQUITTO, and MQTT_PORT must equal the broker's container port. Better still, have the render write both, as it already writes HEALTH_COMPONENTS.

## Mechanical filings rows for these keys (verbatim from the report; re-run contract §8 step 1 before filing regardless)

| key | sources | dedup step | ticket/task | disposition |
|---|---|---|---|---|
| `fk-fa9f275a8a19` | review | filed | 1051 | filed:1051 |

## Seam rows naming this stream

| the hermetic-switch table (settings owns; instance/render/environment.py reads) | S24 | per the stream's PRD; every other stream imports downward |

## What you return (structured block, no commits, no edits to the program doc or another stream's PRD)

AUTHOR pass: write `docs/prds/review-all-<slug>.md` and `docs/prds/review-all-<slug>.capability-manifest.yaml` (this project's convention) under the /prd skill's gates G1-G6 with this brief as the folded-in context; return {prd_path, manifest_path, open_questions (only those the brief and rulings cannot answer), seam_dependencies_on_other_streams, cross_reference_lines}. Do not decompose or file anything in the author pass.
DECOMPOSE pass (only when the lead sends the filed ids you depend on): decompose under planning_mode, stamp metadata.x_finding_key (list) / x_finding_run / source=review-all on every task, add your high/critical leaves as dependencies of the trigger-chain gate id the lead gives you, run commit_planning on your own batch (held leaves excluded), write the stamped sidecar, return your FILED row and cross-reference lines.
