# Capability manifest — loopback binds for the factory's MCP servers and stores

PRD: `plans/mcp-network-exposure-prd.md` (landed `0fcc13fcd9`; §13 edited at decompose). Decomposed
2026-11-01 by seat PB in `/prd` decompose mode. Substrate re-read on `main` `beb3d02ef7`+, live host
re-probed the same day. Machine-readable twin: `plans/mcp-network-exposure-prd.capability-manifest.yaml`
(stamped by `commit_planning`). Task ids live in the sidecar.

Binding vocabulary is `skills/prd/references/gates.md` → *Capability Manifest*. Every mechanical
`delivered_check` below was linted against `main` with `shared.delivered_check_polarity.lint_delivered_checks`
before filing (result recorded in the hand-back).

## Resolutions made at decompose

1. **γ's `before_done.script` must exist on `main` at filing (producer-absent).** The deterministic guard
   refused the authored wrapper path (`before_done.script 'scripts/deploy/apply-loopback-binds.sh' does not
   exist under project_root.`). Resolution: γ names the committed executable `scripts/restart-fused-memory.sh`
   with `args: ["--rebind-socket"]`; β produces that mode and carries the whole apply sequence (PRD §3, §13).
   The γ → β edge is what makes the flag exist before γ dispatches.
2. **γ's own checks gate nothing downstream** (no batch task depends on γ; 7792 depends on β). They are
   recorded so the dashboard's delivered mark and a later `/review` read them; the deploy's done-provenance
   (`deterministic-deploy`, fresh `MainPID`) is the primary record.
3. **α's table pin is a test, not a `delivered_check`** — a grep cannot assert "table equals compose
   `ports`". The wiring into `setup-host.sh` §11 is the anti-orphan check instead.

## Per-leaf bindings

### α — `scripts/check_network_exposure.py` and its two callers

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| exposure check script exists | producer α; absent on `main` | PASS | path `scripts/check_network_exposure.py` present |
| its test module exists | producer α; absent on `main` | PASS | path `tests/scripts/test_check_network_exposure.py` present |
| wired into the install's health checks (anti-orphan) | producer α edits `scripts/setup-host.sh` §11; today the script is not mentioned there | PASS | grep `check_network_exposure` in `scripts/setup-host.sh` present |
| tagged output, parity-family convention | producer α; `scripts/systemd_unit_parity.py::emit` is the substrate (present) | PASS | grep `\[network_exposure\]` in `scripts/check_network_exposure.py` present |
| structured sources, no `ss` parser | producer α; `systemctl --user list-sockets -o json`, `docker inspect`, `ip -j addr` verified live | PASS | grep `list-sockets` in `scripts/check_network_exposure.py` present |
| on today's host the check is red and names 8002/8003/6379/6333/6334 | behaviour; judged by α's tests with in-process `0.0.0.0` binds | PASS | manual |

### β — loopback in the committed truth, plus `--rebind-socket`

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| socket template binds 8002 on loopback | producer β; today `ListenStream=0.0.0.0:8002` | PASS | grep `^ListenStream=127\.0\.0\.1:8002$` in `scripts/fused-memory.socket.template` present |
| no wildcard bind left in the socket template (rejection) | producer β | PASS | grep `0\.0\.0\.0` in `scripts/fused-memory.socket.template` absent |
| `server.host` loopback in config.yaml (re-binds :8003) | producer β; today `host: "0.0.0.0"`; parity test `test_socket_listens_on_the_configured_endpoints` ties it to the template | PASS | grep `^  host: "127\.0\.0\.1"` in `fused-memory/config/config.yaml` present |
| `ServerConfig.host` default loopback | producer β; today `default='0.0.0.0'` | PASS | grep `^    host: str = Field\(default='127\.0\.0\.1'` in `fused-memory/src/fused_memory/config/schema.py` present (anchored at the field's indent: the unanchored form matched `escalation_host: str = Field(default='127.0.0.1')` already on `main`, and `commit_planning` refused it as `vacuous_present`) |
| compose publishes FalkorDB/Qdrant on loopback only | producer β; today `"6379:6379"` etc. | PASS | grep `127\.0\.0\.1:6379:6379` in `fused-memory/docker/docker-compose.yml` present |
| no unprefixed publish left in compose (rejection; covers 7792's future service) | producer β + compose contract test | PASS | grep `^\s*-\s*"[0-9]+:[0-9]+"` in `fused-memory/docker/docker-compose.yml` absent |
| `--rebind-socket` mode exists on the restart script (γ's `before_done` substrate) | producer β; today the script rejects the flag (usage, exit 1) | PASS | grep `rebind-socket` in `scripts/restart-fused-memory.sh` present |
| rebind mode uses the documented stop-then-start | producer β; `OPERATIONS.md` §"Socket activation" procedure | PASS | grep `start fused-memory\.socket fused-memory` in `scripts/restart-fused-memory.sh` present |
| rebind mode runs the exposure check as its post-condition | producer β consumes α | PASS | grep `check_network_exposure` in `scripts/restart-fused-memory.sh` present |
| runbook rows updated (OPERATIONS §8 socket table) | producer β; today the row says `0.0.0.0:8002` | PASS | grep `` `0\.0\.0\.0:8002` `` in `OPERATIONS.md` absent |
| test legs green with the loopback trio; `test_recon_report.py` untouched | behaviour; merge gate | PASS | manual |

### γ — apply on the workstation (deterministic auto-deploy)

| Capability | Evidence | Verdict | Check |
|---|---|---|---|
| no factory port reachable off loopback on the host; loopback accepts | behaviour proven by α's script after the restart | PASS | script `scripts/check_network_exposure.py`, 60 s |
| fused-memory restarted on the new socket unit | `DeterministicRunner` verifies fresh `MainPID`/`ActiveEnterTimestamp` on `fused-memory.service` (`docs/task-authoring.md` §5) | PASS | manual |
| nine orchestrators still active, dashboard panels green, a recon cycle completes against :8003 | operational observation after the deploy | PASS | manual |

Signal leaf: γ.
