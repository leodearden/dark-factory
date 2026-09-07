"""Shared jcodemunch-mcp stdio launch contract — JCODEMUNCH_COMMAND / JCODEMUNCH_ENV.

Single source of truth for how the jcodemunch MCP server is launched, for
every project and every agent kind.  Relocated out of
orchestrator/mcp_lifecycle.py so fused_memory — which declares
``dark-factory-shared`` but NOT ``orchestrator`` as a dependency — can
consume it statically, exactly as ``shared/branch_names.py`` was relocated
for escalation's server.py.

Consumers:

- ``orchestrator.mcp_lifecycle.McpLifecycle.mcp_config_json`` — the
  per-agent config written for every orchestrator-dispatched agent.
- ``fused_memory.reconciliation.stages.base.BaseStage._build_mcp_config`` —
  the reconciliation stage agents.

Why the recon site matters, and why it is not redundant with the
orchestrator one: reconciliation stage agents run at the CANONICAL
``project_root``, never a linked worktree.  Traced
``base.py::BaseStage.run`` -> ``cli_stage_runner.py::run_stage_via_cli``
-> ``shared/cli_invoke.py::_run_subprocess``, with ``explore_codebase_root``
= ``${PROJECT_ROOT:.}``; the sole per-project override draws from
``DASHBOARD_KNOWN_PROJECT_ROOTS``, a fixed list of canonical repo roots,
and ``fused_memory/models/scope.py::resolve_main_checkout`` actively
normalizes worktrees away.  That does NOT make the identity lever pointless
there — it inverts the severity.  Because the recon agent runs at a real
git root with a parseable ``origin``, it is the launch site most likely to
MINT the shared ``<owner>/<repo>`` index, and per the adoption precondition
documented below a pre-existing git-root index makes this lever INERT for
every orchestrator agent worktree thereafter.  The recon site is therefore
a potential CAUSE of index collapse rather than a victim of it.
"""

from __future__ import annotations

__all__ = ['JCODEMUNCH_COMMAND', 'JCODEMUNCH_ENV']

# Prebuilt, version-pinned launcher on PATH (installed via
# `uv tool install --python 3.13 jcodemunch-mcp==<pin>`; see
# reify scripts/setup-dev.sh).  Named constants centralise the launch contract
# so every project's mcp_config_json() injection references one definition and
# a regression test can lock this against reverting to the unpinned uvx form.
# Invoking the installed launcher on PATH avoids `uvx`'s per-launch re-resolve
# + from-source build of tree-sitter C-extension sdists, which under host load
# stalled agent startup past the 1200s wall — the 0-turn MCP-startup wedge
# (reify esc-4415-232). Missing prebuild now fails fast instead of hanging.
# JCODEMUNCH_NO_VERSION_HINT silences the stderr drift note.
#
# JCODEMUNCH_GIT_ROOT_IDENTITY=0: jcodemunch ships `git_root_identity: True`
# as a DEFAULT, so any checkout with a `.git` and a parseable `origin`
# resolves to the repo's single `<owner>/<repo>` index identity. A linked
# worktree's `.git` FILE is treated like a directory and reads the SHARED
# config, so every DF agent worktree of a repo collapses onto ONE identity —
# the first writer claims that index and the upstream collision guard then
# refuses all others, leaving them unindexed while the shared index reflects
# whichever branch the first worktree happened to be on. '0' selects
# per-worktree `local/<basename>-<sha1[:8]>` identity instead. The shared
# jcodemunch-watcher systemd unit adopted this same lever on 2026-06-11 for
# the same reason; this is the per-agent stdio path catching up.
# Deprecated upstream ("will be removed in v2.0. Use config.jsonc instead."),
# verified working at the installed 1.108.55 — a pin bump past v2.0 must
# re-establish this lever via `"git_root_identity": false` in config.jsonc.
# Note `"identity_mode": "local"` is NOT a substitute at 1.108.55: that key is
# absent from the package's CONFIG_TYPES and is discarded silently despite
# the shipped config template advertising it.
# Shadowing risk: the env fallback is skipped for any key explicitly set in
# ~/.code-index/config.jsonc. That host file does not currently set
# git_root_identity, so this env var wins today — but a future host-config
# edit could silently shadow this fix without any test going red.
# Adoption precondition: this only stops NEW git-root indexes — jcodemunch's
# resolve_index_identity (git_root.py) still returns a PRE-EXISTING legacy
# "<owner>/<repo>" index if one is already on disk (e.g. reify's
# ~/.code-index/leodearden-reify.db), making this env var INERT there until
# `jcodemunch-mcp delete-index <owner>/<repo>` is run — see
# scripts/jcodemunch-watcher.service.template's identical caveat. Dark
# Factory has no such legacy index today, so the fix is effective here now.
# Known gap: two sibling launch sites skip this lever and re-spell
# command/args instead of importing these constants —
# fused_memory/reconciliation/stages/base.py's recon-stage launch and
# scripts/setup-host.sh's user-scope `claude mcp add` registration. Either
# creating a legacy git-root index would silently re-collapse that repo's
# worktrees per the precondition above, undermining this fix. Out of this
# task's two-file scope; tracked at tkt_0RSRDYZ75MPVYJP76093P7PYWQ
# (fused-memory site) and tkt_0RSY9MNEMSYK2GBPQ0MJDTYKCQ (setup-host.sh site
# + this compounding risk).
JCODEMUNCH_COMMAND: str = 'jcodemunch-mcp'
JCODEMUNCH_ENV: dict[str, str] = {
    'JCODEMUNCH_NO_VERSION_HINT': '1',
    'JCODEMUNCH_GIT_ROOT_IDENTITY': '0',
}
