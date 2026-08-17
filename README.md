# Dark Factory

Dark Factory is a software factory: it turns a PRD into decomposed tasks,
runs those tasks through a concurrent TDD agent pipeline (plan → implement →
verify → review → merge), and keeps a human in the loop only where it
matters. Three subsystems — **Graphiti** (a temporal knowledge graph),
**Mem0** (a vector memory store), and **Taskmaster AI** (task management) —
are unified behind one MCP server, **fused-memory**, with a reconciliation
loop keeping memory and task state consistent with each other. Agents, the
orchestrator, and a web dashboard all read and write through that same
source of truth. When something goes wrong, a three-tier escalation ladder
(per-task steward → automated triage → human) decides whether it can be
handled without you.

This repo is both the factory itself (the orchestrator, fused-memory, the
dashboard, the escalation server) and the first project it runs against —
its own codebase is developed by the same pipeline it ships.

## Skills you'll use most

Dark Factory's workflows are Claude Code skills (slash commands), not
scripts you invoke directly. The core loop:

| Skill | Use it to |
|---|---|
| `/factory-init` | Onboard a project (new or existing) as an orchestrator target |
| `/prd` | Author a PRD, then gate-decompose it into a queued task batch |
| `/orchestrate` | Run the pipeline, check status, resume, resolve blocks |
| `/escalation-watcher` | Babysit a running orchestrator's L2 (human-tier) escalations |
| `/unblock <task>` | Drive a specific blocked/stuck/escalated task to resolution |
| `/review` | Deep post-hoc audit — does the landed code actually work end-to-end |

Full skill catalog and wiring: `SETUP.md` §"Skill wiring".

## How it works in 60 seconds

1. **Author a PRD** with `/prd` — conversational authoring, then a
   gated decomposition into tasks (each gate checks the task tree is
   complete and coherent enough to hand to agents unsupervised).
2. **Queue** — decomposition writes tasks into fused-memory's task store,
   dependencies and all.
3. **The orchestrator runs the pipeline** per task, up to
   `max_concurrent_tasks` at once, each in its own git worktree and branch:
   `PLAN` (architect) → `EXECUTE` (implementer, TDD) → `VERIFY`
   (test/lint/typecheck) → `DEBUG` (up to 5 cycles, only on verify failure)
   → `REVIEW` (one comprehensive reviewer) → `MERGE` (to `main`, with
   post-merge re-verification and auto-revert on failure).
4. **Escalation ladder** catches anything the pipeline can't resolve itself:
   **L0** a per-task steward session handles it inline; unresolved, it
   escalates to **L1** an automated triage rotation; still unresolved (or
   born critical/urgent), it reaches **L2** a human — via `/escalation-watcher`
   or the dashboard.
5. **`/review`** periodically audits the landed result end-to-end
   (does it actually run, is it wired together, not just "tests green") and
   files follow-up tasks for what it finds — closing the loop back to `/prd`.

See `ARCHITECTURE.md` for the full process topology, port map, and the
escalation ladder's producer/consumer contract.

## Requirements (summary)

- Linux with a user `systemd` instance (production runs the orchestrator,
  fused-memory, and the dashboard as systemd user units).
- Python 3.13 (pinned via `.python-version`), managed by [`uv`](https://astral.sh/uv/).
- Docker + Compose v2 (backing stores).
- Node 22+ (Playwright MCP, `pyright` type-checking).
- [Claude Code CLI](https://www.npmjs.com/package/@anthropic-ai/claude-code)
  with an authenticated Claude subscription (OAuth login — no API key
  needed for the agents themselves).
- `OPENAI_API_KEY` — required for Graphiti/Mem0 embeddings.
- FalkorDB (graph store) + Qdrant (vector store), both via
  `docker-compose`.

Full versions, install commands, and rationale: **SETUP.md**.

## Quickstart

```bash
# 1. Clone with submodules (graphiti, mem0)
git clone --recurse-submodules <this-repo-url> dark-factory
cd dark-factory

# 2. Host bootstrap — see SETUP.md for the full walkthrough and what to
#    adapt on a non-Leo machine
bash scripts/setup-host.sh

# 3. Onboard your own project as an orchestrator target
#    (in a Claude Code session, in dark-factory or the target repo)
/factory-init /path/to/your-project

# 4. Write and queue a PRD against it
/prd

# 5. Run it
cd dark-factory
uv run --project orchestrator orchestrator run --config /path/to/your-project/dark-factory-orchestrator.yaml
```

Step 5 is the ad-hoc/debug invocation. The recommended way to run a project
long-term is a supervised systemd unit (watchdog-supervised, drain-aware
restarts) — `/factory-init`'s Stage 8, or
`skills/factory-init/references/supervised-unit.md`.
Detail on every step: **SETUP.md**.

## Repo map

```
<pkg>/src/<pkg>/     # package double-nesting convention, e.g.
                     #   orchestrator/src/orchestrator/
                     #   fused-memory/src/fused_memory/
                     #   escalation/src/escalation/
                     #   shared/src/shared/
skills/              # in-repo skill source (distinct from ~/.claude/skills —
                     # see SETUP.md "Skill wiring")
dashboard/           # web UI for task/escalation/memory state
scripts/             # operator + CI helper scripts, systemd unit templates
hooks/               # git hooks (pre-commit, pre-merge-commit)
plans/               # design docs and PRDs, in-flight and historical
docs/                # reference docs; docs/legibility/ holds the
                     # design-invariants gate checklist (see below)
graphiti/, mem0/     # git submodules (upstream Graphiti, Mem0)
```

## Documentation index

| Doc | Covers |
|---|---|
| `SETUP.md` | Fresh-machine walkthrough: prerequisites → clone → bootstrap → secrets → backing stores → services → onboard your first project → first run |
| `OPERATIONS.md` | Day-2 runbook: config hot-reload, model routing, fleet redeploy, troubleshooting |
| `ARCHITECTURE.md` | Process topology (what runs where), the escalation ladder, worktree/sandbox model, port map |
| `CONTRIBUTING.md` | Repo conventions (package double-nesting), PRD gates, git workflow, pre-commit |
| `docs/task-authoring.md` | Task metadata: cross-project deps, delivered-check gates, deterministic tasks, milestones, model pins |
| `DESIGN.md` | Fused Memory system design — Graphiti + Mem0 dual-store, write routing, category taxonomy |
| `RECONCILIATION_PLAN.md` | Three-stage sleep-mode reconciliation between memory and tasks |
| `docs/legibility/design-invariants.md` | Checkable design invariants gating `/prd` decompose and `/review` phase 2 |

## Status & known limitations

- **Single-host, personal-fleet heritage.** This repo grew up running one
  maintainer's project fleet on one machine — some scripts and unit files
  still carry hardcoded paths and project lists from that history.
  `SETUP.md` flags each one as a **Known gap** with what to change.
- **Sandboxing is off by default.** Per-task agent sandboxing (`bwrap` /
  Landlock) is implemented but disabled in the shipped defaults for task
  agents — worktree isolation is git-level, not OS-level, today.
- **No auto-discovery, by design.** The orchestrator refuses to run without
  an explicit `--config` (or `ORCH_CONFIG_PATH`) — a past incident ran an
  orchestrator against the wrong project for 12 hours. This is a safety
  feature, not an oversight.
- Troubleshooting and known issues: `OPERATIONS.md`.

## License

[AGPL-3.0](LICENSE).
