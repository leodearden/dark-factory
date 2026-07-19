"""μ OFAT→matrix→confirm driver in evals/runner.py (task 2478).

Hermetic driver tests. The both-live end-to-end path and the three fan-out
stages compose the EXISTING run_eval / run_architect_eval / run_end_to_end
executors, so every test monkeypatches those executors (the test_runner_matrix
pattern) or mocks build_workflow+collect_metrics (the test_eval_architect
pattern) — no live worktree, no LLM, no cloud call.

Step map:
  step-03/04  build_eval_orch_config(architect_config=...) both-live override
  step-05/06  run_end_to_end (architect + implementer both LIVE)
  step-07/08  run_ofat_stage (role-dispatching fan-out)
  step-09/10  run_matrix_stage (architect×implementer cross product)
  step-11/12  run_confirm_stage (single winning combo × N trials)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.config import load_config
from orchestrator.evals.configs import EvalConfig


def _base_config(tmp_path: Path):
    """A deterministic pure-code-default base config via the REAL load_config().

    Mirrors test_eval_boundary_suite._load_default_config: write a minimal YAML
    setting only project_root so load_config layers it over the packaged
    defaults.yaml — every leaf resolves to its code default through the real
    production config-load entry point (never a hand-built OrchestratorConfig).
    """
    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    return load_config(cfg_path)


def _impl_cfg() -> EvalConfig:
    return EvalConfig('claude-sonnet-max', 'claude', 'sonnet', 'max')


def _arch_cfg() -> EvalConfig:
    # model!='opus' and effort!='high' so both diverge from the hardcoded pin.
    return EvalConfig('architect-sonnet-max', 'claude', 'sonnet', 'max', role='architect')


# ---------------------------------------------------------------------------
# step-03/04 — build_eval_orch_config gains an optional architect_config param.
#
# Default None keeps the current opus/claude/high architect pin byte-identical
# (every existing caller + the P1/B1 parity tripwire stay intact); a supplied
# architect_config derives models/backends/effort.architect from the candidate
# for the both-live end-to-end run, leaving implementer/reviewer unchanged.
# ---------------------------------------------------------------------------

class TestBuildEvalOrchConfigArchitectOverride:
    def test_default_none_keeps_opus_architect_pin(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base, architect_config=None)

        # Current pin, unchanged: architect stays opus/claude/high.
        assert cfg.models.architect == 'opus'
        assert cfg.backends.architect == 'claude'
        assert cfg.effort.architect == 'high'
        # Implementer still driven by the eval config under test.
        assert cfg.models.implementer == 'sonnet'
        assert cfg.backends.implementer == 'claude'
        # Reviewer still the 1× opus comprehensive reviewer.
        assert cfg.models.reviewer == 'opus'

    def test_architect_config_overrides_architect_fields(self, tmp_path: Path):
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        impl, arch = _impl_cfg(), _arch_cfg()
        cfg = build_eval_orch_config(impl, {}, base, architect_config=arch)

        # Architect now derives from the candidate (both-live end-to-end run).
        assert cfg.models.architect == arch.model        # 'sonnet' (was 'opus')
        assert cfg.backends.architect == arch.backend     # 'claude'
        assert cfg.effort.architect == arch.effort         # 'max' (was 'high')

        # Implementer / reviewer fields are untouched by the architect override.
        assert cfg.models.implementer == impl.model        # 'sonnet'
        assert cfg.backends.implementer == impl.backend
        assert cfg.models.reviewer == 'opus'
        assert cfg.backends.reviewer == 'claude'

    def test_architect_config_none_is_backward_compatible_positionally(self, tmp_path: Path):
        # The new param must be keyword-optional with a None default so every
        # existing positional caller (run_eval / run_architect_eval) is intact.
        from orchestrator.evals.runner import build_eval_orch_config

        base = _base_config(tmp_path)
        cfg = build_eval_orch_config(_impl_cfg(), {}, base)
        assert cfg.models.architect == 'opus'
