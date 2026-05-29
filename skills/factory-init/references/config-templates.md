# Config Templates

The exact files Stage 5 writes into the **target repo**. Substitute the angle-bracket placeholders. Where a file already exists, **merge** — never clobber a `.gitignore` or `CLAUDE.md` the user already maintains.

Placeholders:
- `<TARGET>` — absolute path to the target repo, e.g. `/home/leo/src/my-solar-challenge`
- `<PROJECT_ID>` — canonical id, lowercase, no hyphens, e.g. `my_solar_challenge`
- `<PORT>` — chosen escalation port, e.g. `8104`

---

## orchestrator.yaml

Modelled on the existing project configs (reify, dark-factory). Only `project_root`, `fused_memory.project_id`, the verify commands, and `escalation.port` are strictly required; the rest are sensible, explicit defaults worth writing so the file is self-documenting.

```yaml
# <PROJECT_ID> — dark-factory orchestrator configuration.
# Package defaults live in dark-factory/orchestrator/defaults.yaml; this file
# overrides only what is project-specific.

project_root: "<TARGET>"

# Shared fused-memory HTTP server (one server, all projects). Only project_id
# is per-project; the URL is always 8002.
fused_memory:
  url: "http://127.0.0.1:8002"
  project_id: "<PROJECT_ID>"
  server_command: []

# Escalation MCP — unique port per project; MUST match .mcp.json so interactive
# sessions and the orchestrator share one escalation server.
escalation:
  queue_dir: "data/escalations"
  port: <PORT>

# Verification commands — see the cookbook below. Match the project's stack.
test_command: "<TEST_CMD>"
lint_command: "<LINT_CMD>"
type_check_command: "<TYPECHECK_CMD>"

git:
  main_branch: "main"
  branch_prefix: "task/"
  remote: "origin"
  worktree_dir: ".worktrees"
```

Add concurrency / model overrides (`max_concurrent_tasks`, `lock_depth`, `max_turns`, …) only if the project needs them — the package defaults cover most cases. The three existing configs (`reify/orchestrator.yaml`, `dark-factory/orchestrator/config.yaml`, `autopilot-video/orchestrator-config.yaml`) are worked examples.

### Verify-command cookbook

The orchestrator runs three commands per task: tests, lint, type-check. They run **inside a fresh git worktree** of the project, so they must work without anything that isn't committed.

| Stack | test_command | lint_command | type_check_command |
|-------|--------------|--------------|--------------------|
| Python + uv | `uv run pytest` | `uv run ruff check` | `uv run pyright` (or `mypy <pkg>`) |
| Python + pip/venv | see **worktree-env caveat** | `ruff check` or `"true"` | `mypy <pkg>` |
| Rust | `cargo test` | `cargo clippy --all-targets -- -D warnings` | `"true"` (clippy is a superset of `cargo check`) |
| Node/TS | `npm test` | `npm run lint` (eslint) | `npx tsc --noEmit` |

- **No linter/type-checker configured?** Use a documented no-op `"true"` rather than a command that always fails (a failing verify command blocks every task). Leave a comment saying why, e.g. `# no linter configured yet`.
- **worktree-env caveat (Python + pip/venv):** a project editable-installed into a local `venv/` (gitignored) has no importable package inside a task worktree, so bare `pytest`/`mypy src/<pkg>` fail there with import errors. Options, best first:
  1. Add a `scripts/verify.sh` that provisions a per-worktree env (`python -m venv .venv && .venv/bin/pip install -e ".[dev]"`) then runs the tool, and point the verify commands at it. This is what reify does (its `scripts/verify.sh` is the single source of truth shared with the git hooks).
  2. Switch the project to `uv` (`uv run pytest`), which resolves and installs into an ephemeral env per invocation.
  Surface this to the user during onboarding — don't ship verify commands that silently can't run under the orchestrator.

---

## .mcp.json

```json
{
  "mcpServers": {
    "fused-memory": {
      "type": "http",
      "url": "http://127.0.0.1:8002/mcp"
    },
    "escalation": {
      "type": "http",
      "url": "http://127.0.0.1:<PORT>/mcp"
    }
  }
}
```

Add `playwright` (copy the stdio block from `dark-factory/.mcp.json`) only if the project has a browser/UI surface worth driving. If a `.mcp.json` already exists, merge these two server entries into its `mcpServers` map.

---

## .envrc

```bash
# Auto-loaded by direnv when entering this directory. Sets ORCH_CONFIG_PATH so
# the dark-factory orchestrator targets this project.
#
# Without direnv installed, this file is inert — pass
# --config <TARGET>/orchestrator.yaml explicitly instead.
export ORCH_CONFIG_PATH="<TARGET>/orchestrator.yaml"
```

Always the absolute path. After writing, if direnv is installed, `cd <TARGET> && direnv allow`. **Never** set `ORCH_CONFIG_PATH` globally in `~/.bashrc` — that re-introduces the silent cross-project bug. Per-directory (direnv) or per-invocation (`--config`) only.

---

## .gitignore additions

Ensure these lines are present (append any that are missing; don't duplicate):

```gitignore
# dark-factory orchestrator scratch — must never reach main
.worktrees/
.task/
.taskmaster/
data/escalations/
```

`.task/` is the load-bearing one: the orchestrator's per-worktree scratch dir. If it's ever committed to `main`, every future worktree inherits it and the orchestrator's `.task/`-presence gates trip.

---

## CLAUDE.md (project_id pin + routing)

Only when the directory name has a hyphen (canonical id differs from the dashboard label), or when the repo has no CLAUDE.md guidance for dark-factory at all. **Merge** into an existing CLAUDE.md — append this section, don't replace the file.

```markdown
## Dark Factory

This project is a dark-factory orchestrator target.

- **Canonical `project_id`: `<PROJECT_ID>`** (the directory name is hyphenated;
  the canonical id uses underscores). Always use this exact id for fused-memory
  writes and task operations — the dashboard may display the hyphenated form.
- Route all task operations through the **fused-memory MCP** with
  `project_root: "<TARGET>"` — never edit task state directly.
- Write-tag memory operations with `project_id: "<PROJECT_ID>"` and a
  descriptive `agent_id`.
- Config: `orchestrator.yaml` (+ `.mcp.json`, `.envrc`) at the repo root.
```
