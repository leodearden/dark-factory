# Project Setup for Orchestrator Targeting

Read this when:
- Setting up a new project for orchestrator runs
- The orchestrator errors with "`--config` is required (or set `ORCH_CONFIG_PATH`)"
- A user asks how to configure direnv / `.envrc` for orchestrator ergonomics
- You need the schema for an `orchestrator.yaml` config file

## Why `--config` is required

On 2026-04-06 night, `/orchestrate` invoked from `~/src/reify` silently ran on dark-factory tasks: the skill `cd`'d to dark-factory, the orchestrator's auto-discovery walked cwd and found dark-factory's own `orchestrator/config.yaml`, and the orchestrator spent 12 hours executing dark-factory tasks instead of reify tasks. A full session of uncommitted dark-factory WIP was also lost via the merge-queue stash/sync path.

To prevent any recurrence, the orchestrator binary now refuses to start without an explicit target. There is no auto-discovery from cwd. There is no fallback to package defaults. Every invocation must name its target project — so a model, script, or human cannot accidentally run the wrong project's tasks.

The skill prompting also enforces this (see SKILL.md "Critical: identify the target project FIRST"), but skill prompting is a soft guard. The binary check is the hard guard.

## The contract

Every orchestrator invocation **must** specify a target project via one of:
1. `--config <path>` — explicit flag, takes precedence
2. `ORCH_CONFIG_PATH=<path>` — environment variable

Both forms behave identically. When both are set, `--config` wins. If neither is set, the orchestrator exits with an educational error.

## The three current projects

| Project | Config file | `project_id` | `project_root` |
|---------|------------|--------------|----------------|
| dark-factory | `/home/leo/src/dark-factory/orchestrator/config.yaml` | `dark_factory` | `/home/leo/src/dark-factory` |
| reify | `/home/leo/src/reify/orchestrator.yaml` | `reify` | `/home/leo/src/reify` |
| autopilot-video | `/home/leo/src/autopilot-video/orchestrator-config.yaml` | `autopilot_video` | `/home/leo/src/autopilot-video` |

Note the inconsistent filenames — each project chose its own. There is no convention enforced; the `--config` flag (or `ORCH_CONFIG_PATH`) names the file explicitly so the inconsistency doesn't matter operationally.

## Setting up a new project

1. **Pick a config filename.** Anything works — `orchestrator.yaml`, `orchestrator-config.yaml`, `config.yaml`, etc. — since the orchestrator never auto-discovers it, the name has no constraint. Convention: `orchestrator.yaml` at the repo root.

2. **Write the minimum required keys**:
   ```yaml
   project_root: "/absolute/path/to/your/project"

   fused_memory:
     project_id: "your_project_id"  # used for memory namespace and reconciliation
     url: "http://127.0.0.1:8002"   # shared fused-memory HTTP server

   # Verification commands appropriate to the project's tech stack
   test_command: "pytest"               # or "cargo test", "npm test", etc.
   lint_command: "ruff check"           # or "cargo clippy", "eslint", etc.
   type_check_command: "pyright"        # or "cargo check", "tsc --noEmit", etc.
   ```

3. **Add concurrency / model overrides if needed.** The package defaults at `dark-factory/orchestrator/defaults.yaml` cover most cases. Override anything project-specific (e.g. `max_concurrent_tasks`, `lock_depth`, per-role models). See the three existing configs as worked examples.

4. **Create `.taskmaster/tasks/tasks.json` in the project root** if it doesn't exist — the orchestrator reads tasks from there.

5. **Create an `.envrc` in the project root** for direnv ergonomics (see next section).

6. **Test the setup**:
   ```bash
   cd /home/leo/src/dark-factory
   uv run --project orchestrator orchestrator status \
       --config /absolute/path/to/your/project/orchestrator.yaml
   ```
   Expect to see the project's task tree (or "No tasks found." if `.taskmaster/tasks/tasks.json` is empty).

## Per-project `.envrc` for ergonomics

Each of the three projects has a `.envrc` file at its repo root that exports `ORCH_CONFIG_PATH` to the absolute path of that project's config. When direnv is hooked into the shell, entering the directory automatically sets the env var, so subsequent orchestrator invocations from that shell don't need `--config`.

Example contents:

```bash
# /home/leo/src/reify/.envrc
export ORCH_CONFIG_PATH="/home/leo/src/reify/orchestrator.yaml"
```

To create one for a new project, copy this pattern and adjust the path. Always use the absolute path — direnv runs the file from the directory, but other tools that re-source it may have a different cwd.

## Installing direnv

direnv is **not currently installed** on this machine. Without it, `.envrc` files are inert (just plain bash text that nothing reads). To activate them:

```bash
# 1. Install direnv (Ubuntu/Debian)
sudo apt install direnv

# 2. Hook direnv into your shell — add to ~/.bashrc:
eval "$(direnv hook bash)"
# (or for zsh: eval "$(direnv hook zsh)" in ~/.zshrc)

# 3. Reload your shell
source ~/.bashrc

# 4. Allow direnv in each project root, once per .envrc edit:
cd /home/leo/src/dark-factory && direnv allow
cd /home/leo/src/reify && direnv allow
cd /home/leo/src/autopilot-video && direnv allow
```

After this, `cd /home/leo/src/reify` automatically exports `ORCH_CONFIG_PATH=/home/leo/src/reify/orchestrator.yaml`, and the orchestrator started from any cwd in that shell will target reify without needing `--config`.

The install is a one-time, sudo action. **Do not install direnv automatically** — confirm with the user first.

## Running without direnv

Two options:

1. **Pass `--config` explicitly every time**:
   ```bash
   cd /home/leo/src/dark-factory
   uv run --project orchestrator orchestrator run \
       --config /home/leo/src/reify/orchestrator.yaml
   ```

2. **Set `ORCH_CONFIG_PATH` for the duration of one shell session**:
   ```bash
   export ORCH_CONFIG_PATH=/home/leo/src/reify/orchestrator.yaml
   # all subsequent orchestrator invocations in this shell target reify
   ```

There is **no global default**. Setting `ORCH_CONFIG_PATH` permanently in `~/.bashrc` would re-introduce the silent-cross-project bug — when you `cd` into a different project, the env var would still point at the old one. Use direnv (per-directory) or pass `--config` explicitly per invocation.

## Verifying setup

After creating an `.envrc` and reloading direnv (or sourcing it manually):

```bash
cd /home/leo/src/<project>
echo "$ORCH_CONFIG_PATH"
# Expect: /home/leo/src/<project>/<config-file>

# Run a status check from inside dark-factory's orchestrator package
cd /home/leo/src/dark-factory
uv run --project orchestrator orchestrator status
# Expect: tasks belonging to <project>, NOT to dark-factory
# (because ORCH_CONFIG_PATH was inherited from your previous shell state)
```

If the tasks shown belong to the wrong project, your `ORCH_CONFIG_PATH` is wrong — `cd` back into the right project directory and try again.
