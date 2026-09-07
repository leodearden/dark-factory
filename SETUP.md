# Setup

A fresh-machine walkthrough for dark-factory: prerequisites, host bootstrap,
secrets, backing stores, the shared services, and onboarding your first
project. For day-2 operation (config reload, fleet redeploy, troubleshooting)
see `OPERATIONS.md` once you're up and running.

## 1. Prerequisites

| Requirement | Why |
|---|---|
| Linux with a user `systemd` instance | Production runs the orchestrator, fused-memory, and the dashboard as `systemctl --user` units, supervised by a watchdog. |
| Python 3.13 (pinned in `.python-version`) | The workspace targets `>=3.11,<4`; 3.13 is the pin every subproject is developed/tested against. |
| [`uv`](https://astral.sh/uv/) | Manages the `uv` workspace (`cockpit`, `dashboard`, `escalation`, `fused-memory`, `orchestrator`, `sampler`, `shared`) and per-project virtualenvs. Install: `curl -LsSf https://astral.sh/uv/install.sh \| sh`. |
| Docker + Compose v2 | Runs FalkorDB and Qdrant, the two backing stores. |
| Node 22+ | Needed for the Playwright MCP server and for `npx pyright` (the type-check command dark-factory itself uses). That pyright is version-pinned by the repo-root `package.json` / `package-lock.json`; run `npm ci` once at the repo root so the bare `npx pyright` resolves the pin instead of fetching whatever is current. Every cold verify worktree does that itself, via `verify_cold_preprovision_command`. |
| [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-code) | `npm install -g @anthropic-ai/claude-code`, then `claude` once interactively to complete OAuth login. |
| `bubblewrap` | Optional per-agent sandboxing primitive. Degrades gracefully if absent — see "Status & known limitations" in `README.md`. |

**Accounts and keys:**
- **`ANTHROPIC_API_KEY` is NOT needed.** Agents authenticate as the Claude Code
  CLI, via your OAuth login (`claude` interactively, once) — not an API key.
- **`OPENAI_API_KEY` IS required.** fused-memory uses it for Graphiti/Mem0
  embeddings (`gpt-4o-mini` + `text-embedding-3-small`) — nothing else in the
  stack needs it.
- **Multi-account OAuth pool is entirely optional.** A single logged-in Claude
  Code CLI session is sufficient to run everything in this doc. The pool
  (`config/usage-accounts.yaml` + `CLAUDE_OAUTH_TOKEN_*` env vars) exists to
  spread load across several Claude Max accounts and fail over on rate-limit
  — skip it unless you're running a large concurrent fleet. Detail in
  §4 below.

## 2. Clone with submodules

```bash
git clone --recurse-submodules <this-repo-url> dark-factory
cd dark-factory
```

The repo vendors two submodules — `graphiti` (getzep/graphiti) and `mem0`
(mem0ai/mem0) — which fused-memory imports directly. If you already cloned
without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## 3. Host bootstrap — `scripts/setup-host.sh`

```bash
bash scripts/setup-host.sh
```

The script is idempotent (safe to re-run) and self-reports each step with
`✓`/`!`/`✗`. It works through, in order:

1. **Prerequisite check** — docker, uv, Node ≥22, Claude Code, `curl`/`jq`/`bubblewrap`; installs what it can, warns about what needs a manual step (e.g. Node version management, `docker` group membership requiring re-login).
2. **Backing stores** — `docker compose up -d falkordb qdrant` from `fused-memory/docker/`, then polls both for health.
3. **`uv sync`** across the Python subprojects (`shared`, `escalation`, `fused-memory`, `orchestrator`, `dashboard`), in dependency order.
4. **fused-memory systemd unit** — renders `scripts/fused-memory.service.template` into `~/.config/systemd/user/fused-memory.service` (substituting your real repo path and `uv` binary path), enables it, and starts it only if `fused-memory/.env` already exists (see §4 — you'll usually run this once before secrets exist, then `systemctl --user restart fused-memory` after writing them).
5. **Orchestrator systemd units + watchdog** — see the **Known gap** callout immediately below; this step is the one you must edit for your own project set.
6. **jCodeMunch** and **7. skim** — optional personal productivity tooling (AST-based code retrieval and context compression) the repo's original maintainer uses. Safe to let run; skip/ignore if you don't want them — nothing else in this repo depends on them.
8. **Dashboard systemd units** — renders and enables `dark-factory-dashboard` + its watchdog timer (installed, not started — see §7).
9. **Claude Code skill symlinks** — wires the in-repo `skills/` into your Claude Code install. Covered in full in §8 below.
10. **Git hooks** — runs `hooks/setup.sh`, which points `core.hooksPath` at `hooks/` (pre-commit, pre-merge-commit).
11. **Manual steps reminder** — printed only, for migrating data from another host (`export-data.sh`/`import-data.sh`); irrelevant on a genuinely fresh install.
12. **Health checks** — re-probes FalkorDB, Qdrant, fused-memory, and jCodeMunch, plus a parity check between your installed fused-memory unit and its template.

### Known gap: step 5 hardcodes the maintainer's other projects

`scripts/setup-host.sh`'s orchestrator-units step unconditionally installs
and enables systemd units for **six projects that are not part of this
repo and won't exist on your machine**: `orchestrator-reify.service`,
`orchestrator-autopilot-video.service`,
`orchestrator-my-solar-challenge.service`,
`orchestrator-solar-challenge-platform.service`,
`orchestrator-know-live.service`, and
`orchestrator-pump-web-ui.service`. Each `ExecStart` points at
a `dark-factory-orchestrator.yaml` in a sibling repo that doesn't exist for
you, so once `default.target` restarts them (next login/boot), they'll
crash-loop until `StartLimitBurst` is exhausted. `orchestrator-dark-factory.service`
itself is genuinely yours and is fine to run — but its `After=` line also
lists `reify-jobserver.service pytest-jobserver.service`, two more units it
doesn't actually need that will show up as (harmless) "not found" ordering
warnings in the journal.

**After running `setup-host.sh`, disable the units that aren't yours:**

```bash
systemctl --user disable --now \
  orchestrator-reify.service \
  orchestrator-autopilot-video.service \
  orchestrator-my-solar-challenge.service \
  orchestrator-solar-challenge-platform.service \
  orchestrator-know-live.service \
  orchestrator-pump-web-ui.service
```

Keep `orchestrator-watchdog.timer` enabled — it skips disabled units, so
this is safe. You'll add your own project's orchestrator unit later, via
`/factory-init`'s Stage 8 (§9 below).

**A second, unrelated hardcode in the same step:** the `wait-for-port.py`
helper is installed to `$HOME/bin/wait-for-port.py` for *whichever user runs
the script* — but the committed unit files (`orchestrator-dark-factory.service`,
and `scripts/fused-memory.service.template`'s `ExecStartPre`) reference the
literal path `/home/leo/bin/wait-for-port.py`, not `$HOME`. If your username
isn't `leo`, both units will fail their `ExecStartPre` port-wait. Fix the
installed copies after rendering:

```bash
sed -i "s#/home/leo/bin/wait-for-port.py#$HOME/bin/wait-for-port.py#" \
  ~/.config/systemd/user/fused-memory.service \
  ~/.config/systemd/user/orchestrator-dark-factory.service
systemctl --user daemon-reload
```

Same caveat applies to `Environment=PATH=...` and `ORCH_UNIT=` lines in
`orchestrator-dark-factory.service` that reference `/home/leo/.local/bin`,
`/home/leo/.cargo/bin` — adjust to wherever `uv`/`cargo`/`node` actually
live on your `$PATH` if verify commands fail with "command not found" once
you're running real tasks.

## 4. Secrets

Copy the two example files and fill them in — both are gitignored, so real
secrets never land in git:

```bash
cp .env.example .env
cp fused-memory/.env.example fused-memory/.env
```

| File | Loaded by | Key vars |
|---|---|---|
| `.env` (repo root) | The orchestrator process (`load_dotenv()` at its `WorkingDirectory`, the repo root — every `orchestrator run` and every `orchestrator-*.service` unit) | `OPENAI_API_KEY`; `CLAUDE_OAUTH_TOKEN_*` (optional pool — see below); `DARK_FACTORY_ROOT` if you did not clone to `/home/leo/src/dark-factory` |
| `fused-memory/.env` | The fused-memory systemd unit (`dotenv` from its own `WorkingDirectory`) | `OPENAI_API_KEY` (**required**); `FALKORDB_URI` / `QDRANT_URL` (optional — sensible defaults below); `DARK_FACTORY_ROOT` if applicable |

The two files mirror each other (`.env.example` and
`fused-memory/.env.example` both carry the same core keys, since
fused-memory reads its own copy independently of the repo root one) — keep
them in sync, or symlink one to the other.

- **`OPENAI_API_KEY`** — required. No default; fused-memory refuses useful
  work without it (embeddings + extraction LLM for both Graphiti and Mem0).
- **`MEM0_TELEMETRY=false`** — recommended in both files; disables Mem0's
  telemetry phone-home.
- **`FALKORDB_URI`** — defaults to `redis://localhost:6379`; only set this if
  you're running FalkorDB somewhere other than the bundled `docker-compose`.
- **`QDRANT_URL`** — defaults to `http://localhost:6333`; same caveat.
- **`RECON_ESCALATION_PORT`** — defaults to `8103`; fused-memory's own
  reconciliation-integrity escalation queue port. Leave it alone unless it
  collides with something else on your host.
- **`DARK_FACTORY_ROOT`** — **set this if you cloned dark-factory to
  anywhere other than `/home/leo/src/dark-factory`.** Both
  `fused-memory/config/config.yaml`'s `usage_cap.accounts_file` and
  `orchestrator/src/orchestrator/defaults.yaml`'s equivalent hardcode
  `${DARK_FACTORY_ROOT:/home/leo/src/dark-factory}/config/usage-accounts.yaml`
  as their fallback — without the env var on a different path, they'll try
  to read a file that isn't there (harmless — the pool just falls back to
  no accounts / plain CLI auth — but worth setting to avoid a confusing
  startup warning).
- **Side-tooling keys** (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
  `PERPLEXITY_API_KEY`, `HF_DOWNLOAD_TOKEN`, `RUNPOD_API_KEY`) appear
  commented out in `.env.example` for optional eval harnesses and auxiliary
  scripts only — **not** needed for the core factory (agents authenticate
  via the Claude Code CLI's own OAuth, never `ANTHROPIC_API_KEY`).

### Optional: multi-account OAuth pool

Only relevant if you want several Claude Max accounts sharing load with
automatic failover on a rate limit. `config/usage-accounts.yaml` lists
accounts by name and the **env var name** holding each one's OAuth token
(never the token itself):

```yaml
accounts:
  - name: max-a
    oauth_token_env: CLAUDE_OAUTH_TOKEN_A
  - name: max-b
    oauth_token_env: CLAUDE_OAUTH_TOKEN_B
```

For each account, generate a long-lived OAuth token (`claude setup-token`)
and put it in `.env` as `CLAUDE_OAUTH_TOKEN_A=...`, `CLAUDE_OAUTH_TOKEN_B=...`,
etc. Absent this file (or an absent env var for a listed account), the pool
degrades gracefully to plain Claude Code CLI default auth — this is the
normal, supported single-account setup. Don't configure this unless you
know you need it; see `OPERATIONS.md` for the failover/capacity model.

## 5. Start backing stores

Already done once by `setup-host.sh` (§3.2 above), but the direct form:

```bash
cd fused-memory/docker
docker compose up -d falkordb qdrant
```

- **FalkorDB** (Redis-protocol graph database — not Neo4j; it's the only
  backend Graphiti is configured for here) on `6379`.
- **Qdrant** (vector store for Mem0) on `6333` (HTTP) / `6334` (gRPC).

A third compose service, `fused-mcp`, builds and runs fused-memory itself
inside Docker — it exists but is **not** the normal path; fused-memory runs
as a systemd user service instead (§6). Ignore `fused-mcp` unless you have a
specific reason to containerize the server.

Verify:

```bash
docker compose exec falkordb redis-cli ping   # → PONG
curl -sf http://localhost:6333/readyz && echo ok
```

## 6. fused-memory service

fused-memory is the single shared MCP server behind Graphiti + Mem0 + task
management — one instance serves every project on the machine.

```bash
systemctl --user daemon-reload
systemctl --user enable --now fused-memory
```

(`setup-host.sh` already did this if `fused-memory/.env` existed at the time
it ran; otherwise re-run the two commands above now that you've written it.)

- Config: `fused-memory/config/config.yaml` (`${VAR}` / `${VAR:default}`
  expansion from `fused-memory/.env`).
- `DASHBOARD_KNOWN_PROJECT_ROOTS` — an `Environment=` line on the
  **installed** unit (`~/.config/systemd/user/fused-memory.service`), a
  comma-separated list of every project root fused-memory's reconciliation
  loop is allowed to touch. The committed template defaults to this repo
  only; every additional project you onboard gets appended here (`/factory-init`
  does this for you — see §9).
- Health check:

```bash
curl -sf http://localhost:8002/health && echo healthy
journalctl --user -u fused-memory -f   # tail logs
```

The MCP endpoint itself is `http://127.0.0.1:8002/mcp` (wired into Claude
Code sessions via each project's `.mcp.json`).

## 7. Dashboard + watchdog units

```bash
systemctl --user daemon-reload
systemctl --user enable --now dark-factory-dashboard
systemctl --user enable --now dark-factory-dashboard-watchdog.timer
systemctl --user enable --now orchestrator-watchdog.timer
```

`setup-host.sh` enables these but does not start the dashboard by default —
start it explicitly the first time. Once running:

- Dashboard UI: `http://127.0.0.1:8080` (task/escalation/memory state, polled
  live from fused-memory + each registered orchestrator).
- `dark-factory-dashboard-watchdog.timer` curls `/healthz` every 30s and
  restarts the dashboard if it wedges.
- `orchestrator-watchdog.timer` TCP-probes every *enabled* orchestrator's
  escalation port every 60s and revives a wedged or dead-but-enabled unit;
  it also runs a fleet-wide staleness pass (at most one redeploy per 8h) —
  see `OPERATIONS.md` for the full fleet-redeploy model. It silently skips
  disabled units, which is why disabling the not-yours units in §3 is safe.

## 8. Skill wiring

Dark Factory's operator workflows are Claude Code **skills**, and there are
two independent wiring mechanisms — know both, because `setup-host.sh` only
completes one of them fully:

1. **`~/.claude/commands/*.md` symlinks** — the classic slash-command
   mechanism, one symlink per `SKILL.md` (plus a `*-references` directory
   symlink for skills that have one). `setup-host.sh` §9 wires: `orchestrate`,
   `reflect`, `unblock`, `unblock-low-risk`, `review`, `review-briefing`,
   `escalation-watcher`, `recon-escalation-watcher`, `merge-queue`, `spawn`,
   `study`, `do`, `census`, `warm`. This step is idempotent — safe to re-run
   after pulling skill changes.
2. **`~/.claude/skills/<name>/` symlinks** — the newer, self-contained Skill
   directory mechanism (a `SKILL.md` plus `references/`/`scripts/` living
   together in one folder, matched as a directory rather than a single
   file). `setup-host.sh` wires exactly three this way: `factory-init`,
   `prd`, `hotspot-survey`.

Both are idempotent `ln -sfn` symlinks back into this repo's `skills/`
directory, so a `git pull` here is picked up immediately — no re-run needed
unless a *new* skill is added (then re-run `setup-host.sh`, or symlink it
yourself: `ln -sfn <dark-factory>/skills/<name> ~/.claude/skills/<name>`).
Two roles are deliberately **not** operator-invocable and have no symlink at
all — `escalation-watcher-auto` and `unblock-auto` are sub-agent-only,
launched internally by the orchestrator and by `/escalation-watcher`
respectively.

Confirm the wiring landed:

```bash
ls -la ~/.claude/commands/ | grep -E 'orchestrate|review|unblock|reflect|merge-queue'
ls -la ~/.claude/skills/ | grep -E 'factory-init|prd|hotspot-survey'
```

## 9. Onboard your first project — `/factory-init`

In a Claude Code session (anywhere; it'll confirm the target path with you),
run:

```
/factory-init /path/to/your-project
```

This is the skill that turns an arbitrary git repo into an orchestrator
target. It works through 8 stages:

0. **Preflight** — confirms dark-factory itself is installed and its backing
   services + fused-memory are healthy (it does **not** install dark-factory
   — that's everything above).
1. **Identify the target** — the path you gave, or your cwd; must not be
   dark-factory itself.
2. **Ensure a git repo** — at least one commit on `main` (the orchestrator
   branches task worktrees from `main`).
3. **Choose a `project_id`** — the directory basename, lowercased, hyphens →
   underscores (e.g. `my-solar-challenge` → `my_solar_challenge`), collision-checked
   against existing projects.
4. **Choose an escalation port** — every project shares fused-memory's
   `8002`, but each needs its **own, unique escalation port**. The skill
   runs `skills/factory-init/scripts/find_escalation_port.py`, which reserves
   `8002`/`8103`, scans known project configs for ports already in use, and
   proposes the lowest free port **≥ 8100**.
5. **Write the per-project config** *into the target repo*:
   `dark-factory-orchestrator.yaml`, `.mcp.json`, `.envrc`, `.gitignore`
   additions, a `.claude/settings.json` with `BASH_MAX_TIMEOUT_MS`, and (if
   the directory name has a hyphen) a `CLAUDE.md` project_id pin.
6. **Register with fused-memory** — see the hard warning immediately below.
7. **Route by code presence** — existing code → `/review-briefing` then
   `/review`, then *offers* `/prd`; greenfield → a goals discussion, then
   1–5 `/prd` runs to queue the first task batch.
8. **Optional supervised systemd unit** — only after tasks are pending (see
   §10 below and "First run").

### ⚠️ Hard warning: register before you queue

**Stage 6 (fused-memory registration) must complete — and fused-memory must
be restarted onto it — before Stage 7 files a single task.** fused-memory's
write path accepts any `project_id`, but its reconciliation loop hard-rejects
any id it doesn't recognize (`UnknownProjectError`) and does **not**
quarantine the bad event — an unregistered or mistyped id filed against a
task status-change can **poison the recon event buffer permanently**. Never
queue a task "just to test it" against a project that hasn't been through
Stage 6.

Registration means appending the target's absolute path to
`DASHBOARD_KNOWN_PROJECT_ROOTS=` on the **installed** `fused-memory.service`
(and, for parity, `dark-factory-dashboard.service`), `daemon-reload`, and
`systemctl --user restart fused-memory` — which **severs your current
session's fused-memory MCP connection** (expected; new sessions reconnect
fine).

### The generated `dark-factory-orchestrator.yaml`

Minimal shape (see `skills/factory-init/references/config-templates.md` for
the full annotated version and a verify-command cookbook per stack):

```yaml
project_root: "<TARGET>"

fused_memory:
  url: "http://127.0.0.1:8002"
  project_id: "<PROJECT_ID>"
  server_command: []

escalation:
  queue_dir: "data/escalations"
  port: <PORT>          # >= 8100, unique, must match .mcp.json

test_command: "<TEST_CMD>"
lint_command: "<LINT_CMD>"
type_check_command: "<TYPECHECK_CMD>"

git:
  main_branch: "main"
  branch_prefix: "task/"
  remote: "origin"
  worktree_dir: ".worktrees"
```

Must-set keys: `project_root`, `fused_memory.project_id`, `escalation.port`,
and the three verify commands. Everything else falls back to
`orchestrator/src/orchestrator/defaults.yaml`.

**pip/venv worktree caveat:** the three verify commands run **inside a
fresh git worktree** of the target project — so a project that's
editable-installed into a local `venv/` (gitignored) has no importable
package inside that worktree, and bare `pytest`/`mypy` will fail there with
import errors. Either add a `scripts/verify.sh` that provisions a
per-worktree env before running the tool (point the verify commands at it),
or switch the project to `uv run pytest` / `uv run mypy`, which resolves and
installs into an ephemeral env per invocation. Don't ship verify commands
that silently can't run under the orchestrator.

## 10. First run

**Recommended (production posture): a supervised systemd unit.** This is
`/factory-init`'s Stage 8, detailed in
`skills/factory-init/references/supervised-unit.md`. Only do this **after**
your project has at least one `pending` task — an orchestrator started
against an empty/all-done tree exits immediately with "No pending tasks
found," and under watchdog supervision that becomes a 60-second crash-loop.

```bash
# confirm there's something to do first:
cd dark-factory
uv run --project orchestrator orchestrator status --config /path/to/your-project/dark-factory-orchestrator.yaml
# → should show >= 1 `pending` task

cp scripts/orchestrator-reify.service ~/.config/systemd/user/orchestrator-<name>.service
# edit WorkingDirectory / ExecStart / ExecStartPre paths for your project + host
systemctl --user daemon-reload
systemctl --user enable --now orchestrator-<name>.service
```

**Ad-hoc / debug path:** run it directly in the foreground, useful for a
first smoke-test or when actively debugging:

```bash
cd dark-factory
uv run --project orchestrator orchestrator run --config /path/to/your-project/dark-factory-orchestrator.yaml
```

**There is no auto-discovery, by design.** `orchestrator run` refuses to
start without an explicit `--config` (or `ORCH_CONFIG_PATH` env var) —
a past incident ran an orchestrator against the wrong project for 12
hours and lost work; the hard guard is deliberate. Never set
`ORCH_CONFIG_PATH` globally in your shell profile — that reintroduces the
exact same hazard for whichever project you happen to `cd` into next.

**`direnv` note:** `/factory-init` writes an `.envrc` in the target repo
that exports the right `ORCH_CONFIG_PATH` for that directory. It's inert
until you install [`direnv`](https://direnv.net/) (`sudo apt install
direnv` + the shell hook) and run `direnv allow` once inside the target —
without it, pass `--config` explicitly every time, which is perfectly fine.

## 11. Verify it's alive

```bash
# Task tree + status for a specific project
cd dark-factory
uv run --project orchestrator orchestrator status --config /path/to/your-project/dark-factory-orchestrator.yaml

# Or via the MCP tools inside a Claude Code session:
#   get_tasks(project_root="/path/to/your-project")

# Dashboard — visual view of tasks/escalations/memory across all projects
xdg-open http://127.0.0.1:8080   # or just open it in a browser

# Live logs
journalctl --user -u orchestrator-<name>.service -f
journalctl --user -u fused-memory -f

# Fleet-wide health, read-only (no mutation)
python3 scripts/orchestrator-watchdog.py --report
```

`--report` prints per-unit start time, newest watched commit, verdict, time
since the last verified fleet deploy, and whether a merge is currently
in-flight (which would defer the next drain-aware redeploy) — reach for it
before manually restarting anything.

## 12. Optional: a remote merge-verify host

By default every merge verify runs locally, on the same host as the
orchestrator. A project under enough load to want a second machine's CPU can
instead register a `verify_runners` entry (workstation-side config — see
"What stays on the workstation" below) that dispatches merge verification to
a second host over ssh. This is an optional throughput lever, not a
requirement for a working install — skip this section unless you need it.

### What lives on the remote host

- **The project checkout**, at a fixed path (e.g. `~/src/<project>`), kept in
  sync with the dispatching workstation's `main` (see "Provisioning the
  remote host" below for how it gets refreshed). It is a full clone that the
  remote runs commands directly against, not a worktree.
- **`receive.denyCurrentBranch = updateInstead`**, set on that checkout
  (`git config receive.denyCurrentBranch updateInstead`). This is required
  because the dispatcher's best-effort keep-alive push targets
  `main:refs/heads/main` — the branch the remote checkout has checked out,
  which git refuses to accept a push to by default. `updateInstead` tells
  git to update the working tree in place instead of rejecting the push. The
  **load-bearing** push — the actual commit under verification — goes to a
  separate, namespaced ref instead, `refs/merge-verify/<request-id>`, which
  is never checked out and so needs no such configuration.
- **A PATH wrapper** for the orchestrator CLI (e.g.
  `/usr/local/bin/orchestrator`), needed because a non-interactive
  `ssh host cmd` does not source `~/.bashrc`/`~/.profile`, so a bare
  `orchestrator` would not otherwise resolve on PATH. The wrapper only needs
  to export the directories the verify subprocesses require (toolchain
  bins, package-manager shims, …) and `exec` the checkout's own venv entry
  point:
  ```bash
  #!/bin/bash
  export PATH="<toolchain-bin-dirs>:$PATH"
  exec <path-to-checkout>/.venv/bin/orchestrator "$@"
  ```
  Keep the wrapper bare — the liveness probe the dispatcher uses
  (`orchestrator verify-merge --help`, unqualified) is deliberately
  unqualified so it exercises the exact same PATH resolution a real dispatch
  does; wrapping the probe in `uv run` or an absolute `.venv/bin` path would
  test something weaker than what actually runs.

**A Dark-Factory code checkout on a shared verify host backs every project
whose `verify_runners` entry points at it.** Refreshing the checkout (the
next subsection) upgrades all of them at once, atomically, whether they are
ready for it or not — and must never be done while a verify is in flight on
that host, since the checkout is the code a live verify is currently
executing.

### Provisioning the remote host

After fast-forwarding the checkout to the workstation's `main` (a plain
`git fetch && git reset --hard origin/main` — never `git clean`; leave
whatever untracked paths are already on that host alone), re-provision the
environment at the new commit:

```bash
<uv path> sync --all-packages
npm ci
```

- **Always `--all-packages`, never a bare `uv sync`.** A Dark-Factory
  checkout is a **uv workspace**, and a bare `uv sync` exits `0` while
  *pruning the other members' console scripts* — including the
  `orchestrator` entry point this host exists to expose (task 4539). Losing
  it turns every subsequent dispatch into a silent `rc=127`.
- **`npm ci`** installs the pinned `npx pyright` (and anything else the
  repo-root `package-lock.json` carries) — needed the same way it is on the
  workstation (§1).
- **Non-login-ssh PATH trap:** a plain `ssh <host> '<uv path> sync ...'` can
  fail `rc=127` for a reason that has nothing to do with the command itself
  — a non-interactive, non-login ssh session does not source
  `~/.bashrc`/`~/.profile`, so a tool installed under, e.g., `~/.local/bin`
  is simply not on `PATH` yet. Invoke it by absolute path (as above) or
  export `PATH` first; don't mistake the resulting `127` for a broken
  install. `node`/`npm` are commonly already on the default ssh `PATH` via
  `/usr/bin`, so this trap tends to bite `uv` (or `cargo`, `~/.local/bin`
  toolchains generally) more often than the Node half of this step.

### Checking the host answers

Two probes, both run exactly as the dispatcher itself runs them —
`orchestrator/src/orchestrator/verify_runner.py`:

```bash
ssh -o BatchMode=yes <host> 'orchestrator verify-merge --help'   # REMOTE_LIVENESS_CMD
ssh -o BatchMode=yes <host> true                                  # RemoteRunner.health()
```

- The CLI probe must be spelled **bare** — `orchestrator verify-merge
  --help`, not `uv run orchestrator ...` and not an absolute `.venv/bin`
  path. It is bare on purpose so the probe exercises the exact same PATH
  resolution (through the host's `orchestrator` wrapper — previous
  subsection) that a real dispatch does; a qualified spelling can pass while
  the real dispatch still fails `rc=127`.
- Assert exit code `0` on both.

**A passing probe proves the CLI is reachable, not that the checkout is
current.** The CLI probe already exits `0` against a stale checkout — it
only executes `--help`, which needs no correct code, just a resolvable
entry point. The currency check is separate: `git rev-parse HEAD` on the
remote must equal the dispatching workstation's `origin/main`. This
distinction is not cosmetic — under `merge_verify_breadth=full` the module
set is *discovered from the remote tree itself* (task 4536), so a stale
remote checkout silently **narrows** the gate rather than failing loudly.
Verify currency, every time, not just reachability.

### The per-host YAML

The remote-side config conventionally lives at
`~/.config/orchestrator/<project>-<host>.yaml`, but that path is **pure
operator convention — nothing in the code discovers it.** It is threaded
through explicitly: the workstation's `verify_runners[].config_path` names
it, passes it as `orchestrator verify-merge --config <path>`, and the
ordinary `load_config` resolution on the remote takes it from there
(`--config` wins over `ORCH_CONFIG_PATH`, which is the fallback when
`config_path` is left `None`). Name it however you like, or omit it and
rely on the remote's own `ORCH_CONFIG_PATH`.

**Load-bearing vs. inert.** `verify-merge` only reads `project_root` and
the `git.*` block from this file — scheduler, routing, budget, and
escalation sections are inert for this subcommand and can be left at
whatever value (or omitted) without effect. Don't spend effort keeping
those sections in sync with the workstation; only `project_root` and
`git.*` need to be right.

**The rule that decides whether verify commands belong in this file:**
whether the project registers any per-module verify configs at all.

- A project with **registered module configs** (module-level
  `test_command`/`lint_command`/`type_check_command` entries) has those
  commands travel inside the `MergeVerifySpec` the workstation sends with
  every dispatch, reconstructed remotely by
  `run_merge_verify_on_worktree`. Restating `test_command`/`lint_command`/
  `type_check_command`/`verify_env` in this file would create a **second
  copy, free to drift** from the workstation's — omit them here.
- A project with **zero** registered module configs ships a spec with an
  empty module-config list, and the remote has nothing to reconstruct from
  — it falls back to *this file's own* `test_command`/`lint_command`/
  `type_check_command`/`verify_env`. Here, omitting them would leave the
  remote with no verify commands at all — set them here.

Get this backwards in either direction and the failure is silent: a
restated command quietly drifts from the one the workstation actually
runs, while a missing fallback command means the remote verify silently
skips (or errors on) whatever step has no command.

### Warm vs. ephemeral verify worktrees

`git.persistent_merge_worktree` controls what a `verify-merge` invocation
materialises its detached checkout into. **Off** (the default): every
invocation gets its own fresh `<worktree_dir>/_merge-<uuid>`, used once and
discarded. **On:** invocations on that host instead reuse one fixed-path
warm worktree, `<worktree_dir>/_merge-verify`, resetting it in place each
time — trading the cost of materialising a worktree from scratch for
reuse of whatever build caches already live inside it.

Two things must both hold before turning it on for a project, not just one:

1. **The dispatcher-side serial-lane guard must be live** — which means
   the knob is set on **both** the dispatching workstation and the remote
   host, not the remote alone. A shared warm worktree is only safe with at
   most one verify in flight against it per host at a time; the
   workstation-startup guard that enforces that serial lane reads the
   *workstation's* copy of this knob. Flip it on the remote only, and the
   host gets the shared warm worktree with **nothing enforcing** the
   serial-lane invariant that worktree depends on.
2. **The project's cold-preprovision cost must be high enough to make
   reuse worth wanting.** If a project pays a full dependency
   install/build on every cold verify (a `verify_cold_preprovision_command`
   with no in-process memoisation across invocations — which is exactly
   what a stateless `ssh host orchestrator verify-merge ...` dispatch is),
   warm reuse is what makes that cost bearable. A project with a cheap cold
   path gets little from the knob and takes on the concurrency constraint
   for nothing.

**Failure mode if you set it on the remote alone:** nothing stops a second,
concurrent verify from landing on the same host while the first is still
running. Both share the one warm worktree, and concurrent use of it is
exactly the hazard invariant 4 (PRD §A) exists to prevent — a race over the
same `target`/build directory. Set the knob on both sides together, or not
at all.

**Dark Factory's answer, as a worked example (decided 2026-09-07, task
5051):** `false` — deliberately mirroring the workstation's own setting.
The dispatcher-side guard is not the blocker here (dark-factory has none
of the arbitration machinery this would need to make safe today); the
decisive fact is that flipping only the laptop half would hand out a
shared warm worktree with the workstation guard reading its own `false`
and enforcing nothing. Revisit after measuring the actual remote-verify
cold-preprovision cost in practice, and only by flipping both sides
together in the same change.

---

For hot-reloading config without a restart, understanding the fleet-redeploy
mechanisms, model routing, and troubleshooting a stuck or misbehaving
orchestrator, see `OPERATIONS.md`.
