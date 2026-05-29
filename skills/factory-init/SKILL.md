---
name: factory-init
description: "Onboard a software project — brand-new or with existing code — so the dark-factory orchestrator can develop it. Use this skill whenever the user wants to 'factory-init' a repo, 'set up a project for the orchestrator', 'make this repo a dark-factory target', 'register a new project', 'onboard <repo> into the factory', or points at a directory and asks to get it ready for autonomous development — even if they don't say 'factory-init'. It ensures dark-factory is installed (installing it is out of scope), ensures a git repo, picks a hyphen-free canonical project_id and an unused escalation port, writes the per-project config (orchestrator.yaml, .mcp.json, .envrc, .gitignore, CLAUDE.md pin), registers the project with fused-memory reconciliation, then routes to the right next skill: review-briefing → review for existing code, or a goals discussion → /prd batch for a greenfield project. NOT for: installing/bootstrapping dark-factory itself (a separate skill), running the orchestrator (/orchestrate), authoring a single PRD (/prd), or unblocking tasks (/unblock)."
---

# Factory Init

Turn a repository — greenfield or one with an existing codebase — into a first-class **dark-factory orchestrator target**: pick its identity and port, write its config, register it with reconciliation, and hand off to the skill that makes sense for its state. The goal is that when you finish, the user can run `/orchestrate` against the project and it Just Works, with reconciliation and the dashboard recognising it.

**Installing dark-factory itself is out of scope.** This skill assumes a working dark-factory (services up, fused-memory live) and only *adds a project to it*. If dark-factory isn't installed, stop and say so — a separate skill will own that.

## What "done" looks like

Hold this end-state in mind; every stage moves toward it:

1. dark-factory is present and its backing services are healthy.
2. The target is a git repo (the orchestrator branches task worktrees from `main` — no repo, no worktrees).
3. A canonical `project_id` is chosen — lowercase, **no hyphens** — and pinned where the directory name would otherwise disagree with it.
4. A unique escalation port is chosen and recorded.
5. The target repo contains `orchestrator.yaml`, `.mcp.json`, `.envrc`, a `.gitignore` covering orchestrator scratch dirs, and (where needed) a `CLAUDE.md` project_id pin.
6. The project root is registered in `DASHBOARD_KNOWN_PROJECT_ROOTS` and fused-memory has been restarted and verified — **before any task is queued**.
7. The right next step is launched: existing code → `/spawn /review-briefing` then `/spawn /review` then *offer* `/spawn /prd`; greenfield → a goals discussion, then `/spawn /prd` run 1–5× serially to queue the first task batch.

## Operating principles

- **This skill mutates a live, shared system.** Writing config files *into the target repo* is reversible — just do it. But restarting fused-memory and spawning sessions are not things to do silently: confirm before them. The user has authorised "full auto with confirm" — honour the confirm.
- **Derive, don't guess, and show your work.** The `project_id` and the escalation port are both mechanical. Compute them and show the evidence (the derivation rule, the table of ports already in use) so the user can sanity-check rather than trust.
- **Register before you queue — this one is load-bearing.** fused-memory's write path accepts *any* `project_id`, but the reconciliation loop hard-rejects ids it doesn't know and does **not** quarantine them. A single task status-change on an unregistered or mistyped id poisons the recon event buffer permanently. So registration + restart (Stage 6) must complete *before* any `/prd` run files a task (Stage 7). Getting the order wrong is not a cosmetic bug.

## Stage 0 — Preflight: is dark-factory installed and healthy?

Installing dark-factory is out of scope; you are only confirming it's there to build on.

- **Locate the dark-factory root.** Default `/home/leo/src/dark-factory`; honour `$DARK_FACTORY_ROOT` if set; or derive it from this skill's own path (`…/skills/factory-init` → repo root). Confirm `orchestrator/` and `scripts/fused-memory.service.template` exist.
- **Check the backing services** without starting or restarting anything:
  - FalkorDB on `6379` and Qdrant on `6333` — probe with `ss -ltn` or `nc`.
  - fused-memory on `8002` — probe its `/health`. **curl is broken on this host** (libcurl leaks even under `env -i`); use python `urllib.request` instead.
- If dark-factory or any service is missing → **stop**. Tell the user dark-factory isn't fully installed, that installing it is out of scope for this skill, and don't attempt to install or start anything. Point them at the (future) install skill.

## Stage 1 — Identify the target project

The target is the path the user named (an argument like `../my-solar-challenge`) or, failing that, the current working directory. Resolve it to an absolute path and confirm it with the user. It must not be the dark-factory repo itself.

## Stage 2 — Ensure a git repo

`git -C <target> rev-parse --is-inside-work-tree`. If it isn't a repo, offer `git init` plus an initial commit of whatever's there. The orchestrator creates each task's worktree by branching from `main`, so a repo with at least one commit on `main` is a hard prerequisite.

## Stage 3 — Choose the project_id (no hyphens)

The canonical id is the directory basename, lowercased, with hyphens replaced by underscores — this is exactly what `resolve_project_id` does, so matching it avoids a write/read mismatch (`my-solar-challenge` → `my_solar_challenge`).

- **Collision check.** The id must not already belong to another project. Scan the existing project configs and `DASHBOARD_KNOWN_PROJECT_ROOTS` (Stage 4's script surfaces these). If it's taken, pick a distinct name with the user.
- **Hyphen handling.** When the directory name contains hyphens, the dashboard label (hyphenated) will differ from the canonical underscore id, and an interactive session left to its own devices may improvise the wrong form. Pin the canonical id in the target's `CLAUDE.md` (Stage 5) so spawned sessions and MCP writes always use the underscore form. For a *brand-new* project, prefer a hyphen-free directory name from the outset and sidestep the whole issue.

## Stage 4 — Choose an unused escalation port

All projects share the single fused-memory server on `8002`; only the **escalation** port is unique per project. Run the bundled finder:

```bash
python3 <skill>/scripts/find_escalation_port.py --exclude-root <target>
```

It reads the known project roots (from the live fused-memory unit and `/home/leo/src/*`), parses each project's `escalation.port`, reserves `8002` (fused-memory) and `8103` (shared reconciliation queue), checks what's actually bound, and prints an evidence table plus the lowest free port ≥ `8100`. Show the table to the user and use the chosen port.

## Stage 5 — Write the per-project config (in the TARGET repo)

Exact templates, the verify-command cookbook, and merge-don't-clobber guidance live in **`references/config-templates.md`** — read it before writing. In brief, create/merge in the target repo:

- **`orchestrator.yaml`** — `project_root`, `fused_memory{project_id, url: http://127.0.0.1:8002}`, `escalation{queue_dir: data/escalations, port: <chosen>}`, a `git` block, and `test_command`/`lint_command`/`type_check_command` matched to the detected stack. Detect the stack from the repo (pyproject → pytest/mypy/ruff; Cargo.toml → cargo; package.json → npm/eslint/tsc) and **confirm the commands with the user** — especially when a linter or type-checker isn't configured. Prefer a documented no-op (`"true"`) over a command that will always fail, and say why in a comment.
- **`.mcp.json`** — `fused-memory` → `8002`, `escalation` → the chosen port. The escalation URL **must** match `orchestrator.yaml` so interactive sessions and the orchestrator share one server. Add `playwright` if the project has a browser surface.
- **`.envrc`** — `export ORCH_CONFIG_PATH="<absolute path to orchestrator.yaml>"`. direnv may not be installed (don't auto-install — offer it); without direnv this file is inert and the orchestrator needs `--config` explicitly.
- **`.gitignore`** — ensure `.worktrees/`, `.task/`, `.taskmaster/`, and `data/escalations/` are ignored. `.task/` is critical: if it ever lands on `main` it contaminates every future worktree. Merge into any existing `.gitignore` rather than overwriting.
- **`CLAUDE.md`** — if a project_id pin is needed (hyphenated dir), add it plus the dark-factory routing conventions (route all task ops through fused-memory MCP with `project_root=<target>`; write-tag with `project_id=<id>`). Merge into an existing `CLAUDE.md`, never clobber it.
- `mkdir -p <target>/data/escalations`.

**Validate before registering.** From the dark-factory repo, run a status check against the new config:

```bash
cd <dark-factory> && uv run --project orchestrator orchestrator status --config <target>/orchestrator.yaml
```

Expect "No tasks found." (or an empty tree) — *not* a config error. This confirms the YAML parses and satisfies the loader before you touch the shared service.

## Stage 6 — Register with reconciliation (DESTRUCTIVE — confirm first)

Full procedure and rationale in **`references/recon-registration.md`**. The shape:

1. Append the target's absolute path to the `DASHBOARD_KNOWN_PROJECT_ROOTS=` line in `<dark-factory>/scripts/fused-memory.service.template` — the source of truth that survives future re-renders.
2. Render it into the live unit the same way `setup-host.sh` does (`sed 's|__REPO_ROOT__|<dark-factory>|g'` → `~/.config/systemd/user/fused-memory.service`), then `systemctl --user daemon-reload`.
3. **Confirm with the user, then `systemctl --user restart fused-memory`.** ⚠️ This severs *this* session's fused-memory MCP tools — they do not reconnect. That's acceptable: nothing after this point needs MCP in this session, and the sessions you spawn in Stage 7 get fresh connections to the restarted server.
4. Verify: a python-urllib probe to `8002/health` returns healthy, and `journalctl --user -u fused-memory` shows the new project recognised. Registration detection is heuristic, so *confirm in the log* rather than assuming.

## Stage 7 — Route by code presence

Decide whether the repo already has substantive source (real modules under `src/`/packages, not just README + config). Spawn sessions with the bundled `/spawn` helper (`<dark-factory>/skills/spawn/spawn-claude.sh <cwd> <skip_perms> '<title>' '<prompt>'`, run in background — it blocks until the spawned session exits, so the background task's completion is your "done" signal). Use `cwd=<target>`.

**Existing code:**
1. `/spawn /review-briefing` — builds `review/briefing.yaml`, the durable project context `/review` consumes.
2. When it completes → `/spawn /review` — surfaces the real state of the existing code and files tasks for the gaps it finds.
3. Then **offer** (don't auto-run) `/spawn /prd` for net-new feature work.

**Greenfield (no code yet):**
1. Discuss with the user to get a crisp outline of the project's goals and a rough split into 1–5 PRD-sized slices.
2. For each slice, `/spawn /prd "<framing of this slice>"` **serially** — wait for each to finish before starting the next. Serial because PRDs share seams (the /prd skill's G4 cross-PRD ownership gate), so later PRDs reference what earlier ones established.
3. The result is an initial batch of queued tasks, ready for `/orchestrate`.

## Offer anything else

Surface these as options once the core onboarding is done; don't do them unprompted:

- **Commit the new config** to the repo so it's tracked (worth doing — untracked config referenced by tooling is a known foot-gun).
- **direnv install** (`sudo apt install direnv` + shell hook) — one-time, makes `.envrc` live so you don't need `--config`. Confirm before any sudo.
- **A supervised systemd orchestrator unit** for always-on projects, modelled on `scripts/orchestrator-reify.service`.
- **A PRD project overlay** at `<target>/.claude/skills/prd/project.md` if the user wants project-specific G2/G3 vocabulary for decomposition.

## Pitfalls (most are in memory for a reason)

- **Recon storm** — register the project_id (Stage 6) *before* any task is queued (Stage 7); an unknown/mistyped id poisons the recon buffer for good.
- **MCP death on restart** — the Stage 6 restart kills this session's fused-memory tools; that's expected and fine here.
- **curl is broken on this host** — use python `urllib` for all HTTP probes.
- **Hyphen → underscore** — pin the canonical id in CLAUDE.md whenever the directory name has a hyphen.
- **`.task/` contamination** — must be gitignored; never let it reach `main`.
- **No global `--config` default** — the orchestrator refuses to run without `--config`/`ORCH_CONFIG_PATH`; that's a safety feature (it once ran 12h on the wrong project). The `.envrc` is the ergonomic fix.
- **venv-in-worktree (Python)** — a project editable-installed into a local `venv/` won't have that env inside a task worktree (it's gitignored), so bare `pytest`/`mypy` verify commands fail there. Flag this and propose a worktree-safe strategy (a `verify.sh` that provisions a per-worktree env, or `uv run`) — don't ship verify commands that silently can't work under the orchestrator.
