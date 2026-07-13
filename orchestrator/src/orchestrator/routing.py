"""Model routing: allowlist + fail-fast validation + per-account availability probe.

Task beta (Phase-1 substrate of plans/adaptive-model-routing-prd.md). This
module is the PRD-named "allowlist home": ``DEFAULT_ALLOWED_MODELS`` is the
source of truth for ``OrchestratorConfig.routing``'s default (see
``config.py``'s ``RoutingConfig`` submodel and its
``_validate_models_in_allowlist`` cross-field validator).

This module will also host the per-account model-availability probe
(``probe_models``) and its rendered artifact format
(``render_probe_artifact``), consumed by the ``orchestrator probe-models`` CLI
subcommand (``orchestrator/cli.py``) — added by later steps in this task's
plan, not yet present here.

Kept import-light at module top (stdlib only) so ``config.py`` can
``from orchestrator.routing import DEFAULT_ALLOWED_MODELS`` with no circular
import; heavier imports (e.g. ``shared.cli_invoke.invoke_claude_agent``) are
deferred to inside the functions that need them.
"""

from __future__ import annotations

# Claude-backend model aliases admitted by default (task beta). claude-fable-5
# is deliberately NOT included here — task xi admits it to the runtime
# allowlist once probe_models confirms availability across every pool account
# (see FABLE_CANDIDATE_MODEL below).
DEFAULT_ALLOWED_MODELS: tuple[str, ...] = ('haiku', 'sonnet', 'opus')

# Candidate model probed for availability even though it is not yet admitted
# to the runtime allowlist — beta is the G3 gate that produces the
# per-account fable-availability data task xi's admission gate consumes.
FABLE_CANDIDATE_MODEL: str = 'claude-fable-5'

# Default path for the committed probe-models artifact, sibling of
# config/usage-accounts.yaml (the account-pool source of truth).
DEFAULT_PROBE_ARTIFACT_PATH: str = 'config/model-availability.yaml'
