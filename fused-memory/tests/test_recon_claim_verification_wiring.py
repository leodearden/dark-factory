"""Tests for CuratorConfig + TaskCurator wiring of the recon claim-verification
guard (task 2438).

Mirrors TestCuratorReconPremiseRegistryConfig (test_task_curator.py) for the
config field, and _maybe_premise_refuted_drop's curator-hook tests for the
TaskCurator._maybe_flag_unverified_claims wiring — same shapes, new
(advisory, not a drop) guard. See recon_claim_verification_guard.py's module
docstring for the motivating task-2433 incident.
"""

from __future__ import annotations

import logging

import pytest

# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-07 RED: TestCuratorReconClaimVerificationConfig
# ──────────────────────────────────────────────────────────────────────────────


class TestCuratorReconClaimVerificationConfig:
    """Tests that CuratorConfig has recon_claim_verification_enabled field,
    defaulting to False, and that FusedMemoryConfig round-trips it via YAML.

    Mirrors TestCuratorReconPremiseRegistryConfig in test_task_curator.py —
    same shape, new field.
    """

    def test_curator_config_has_field_default_false(self):
        """CuratorConfig has recon_claim_verification_enabled field, default False."""
        from fused_memory.config.schema import CuratorConfig

        cfg = CuratorConfig()
        assert hasattr(cfg, "recon_claim_verification_enabled")
        assert cfg.recon_claim_verification_enabled is False

    def test_curator_config_accepts_true(self):
        """CuratorConfig accepts recon_claim_verification_enabled=True."""
        from fused_memory.config.schema import CuratorConfig

        cfg = CuratorConfig(recon_claim_verification_enabled=True)
        assert cfg.recon_claim_verification_enabled is True

    def test_fused_memory_config_roundtrips_via_yaml(self, tmp_path, monkeypatch):
        """FusedMemoryConfig round-trips recon_claim_verification_enabled via YAML."""
        import yaml

        from fused_memory.config.schema import FusedMemoryConfig

        raw = {"curator": {"recon_claim_verification_enabled": True}}
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(yaml.dump(raw), encoding="utf-8")

        monkeypatch.setenv("CONFIG_PATH", str(yaml_path))
        cfg = FusedMemoryConfig()
        assert cfg.curator.recon_claim_verification_enabled is True


# ──────────────────────────────────────────────────────────────────────────────
# task-2438 step-09 RED: TestMaybeFlagUnverifiedClaims
# ──────────────────────────────────────────────────────────────────────────────


def _make_config(recon_claim_verification_enabled: bool = False):
    from fused_memory.config.schema import CuratorConfig, FusedMemoryConfig

    cfg = FusedMemoryConfig()
    cfg.curator = CuratorConfig(
        recon_claim_verification_enabled=recon_claim_verification_enabled,
    )
    return cfg


# The verbatim task-2433 sentence: a code-level claim (metadata.
# done_provenance_invalidated=true) attributed to a completed task/ACTION
# (per task 2372 ACTION #5) — exactly the shape extract_attributed_claims
# is built to catch.
_INCIDENT_DESCRIPTION = (
    "This is the same site that stamps "
    "metadata.done_provenance_invalidated=true per task 2372 ACTION #5 "
    "on task reopen."
)


@pytest.mark.asyncio
class TestMaybeFlagUnverifiedClaims:
    """Tests for TaskCurator._maybe_flag_unverified_claims(candidate, probe=None).

    Unlike _maybe_premise_refuted_drop, this hook is advisory: it never
    returns/mutates a CuratorDecision and never drops the candidate — it
    only surfaces unverified claims via a WARNING census line.
    """

    async def test_disabled_returns_empty(self, tmp_path):
        """(a) returns [] when recon_claim_verification_enabled is False —
        even though the probe below would flag every token as absent, proving
        the config flag actually gates the check rather than the result
        coincidentally being empty."""
        from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

        config = _make_config(recon_claim_verification_enabled=False)
        curator = TaskCurator(config=config, taskmaster=None, cwd=tmp_path)
        candidate = CandidateTask(title="T", description=_INCIDENT_DESCRIPTION)

        result = await curator._maybe_flag_unverified_claims(
            candidate, probe=lambda token: False,
        )

        assert result == []

    async def test_enabled_unverified_token_returns_claim_and_warns(self, tmp_path, caplog):
        """(b) enabled + probe reports the token absent -> returns the
        AttributedClaim AND emits a grep-stable WARNING naming
        recon_claim_verification.unverified + the token."""
        from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

        config = _make_config(recon_claim_verification_enabled=True)
        curator = TaskCurator(config=config, taskmaster=None, cwd=tmp_path)
        candidate = CandidateTask(title="T", description=_INCIDENT_DESCRIPTION)

        with caplog.at_level(logging.WARNING):
            result = await curator._maybe_flag_unverified_claims(
                candidate, probe=lambda token: False,
            )

        assert len(result) == 1
        assert result[0].token == "done_provenance_invalidated"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "recon_claim_verification.unverified" in r.getMessage()
            and "done_provenance_invalidated" in r.getMessage()
            for r in warnings
        )

    async def test_enabled_verified_token_returns_empty_no_warning(self, tmp_path, caplog):
        """(c) enabled + probe reports the token present -> [] and no
        recon_claim_verification WARNING (self-correcting)."""
        from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

        config = _make_config(recon_claim_verification_enabled=True)
        curator = TaskCurator(config=config, taskmaster=None, cwd=tmp_path)
        candidate = CandidateTask(title="T", description=_INCIDENT_DESCRIPTION)

        with caplog.at_level(logging.WARNING):
            result = await curator._maybe_flag_unverified_claims(
                candidate, probe=lambda token: True,
            )

        assert result == []
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert not any("recon_claim_verification" in r.getMessage() for r in warnings)

    async def test_cwd_none_fails_open_empty(self):
        """(d) returns [] fail-open when self._cwd is None, even when enabled
        (no probe injected — proves the cwd-None guard short-circuits before
        ever needing to build/call a probe)."""
        from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

        config = _make_config(recon_claim_verification_enabled=True)
        curator = TaskCurator(config=config, taskmaster=None, cwd=None)
        candidate = CandidateTask(title="T", description=_INCIDENT_DESCRIPTION)

        result = await curator._maybe_flag_unverified_claims(candidate)

        assert result == []

    async def test_hook_never_returns_decision_and_never_mutates_candidate(self, tmp_path):
        """(e) The hook returns a plain list of AttributedClaim (never a
        CuratorDecision / drop) and never mutates the candidate."""
        from fused_memory.middleware.recon_claim_verification_guard import AttributedClaim
        from fused_memory.middleware.task_curator import (
            CandidateTask,
            CuratorDecision,
            TaskCurator,
        )

        config = _make_config(recon_claim_verification_enabled=True)
        curator = TaskCurator(config=config, taskmaster=None, cwd=tmp_path)
        candidate = CandidateTask(title="T", description=_INCIDENT_DESCRIPTION)
        before = (candidate.title, candidate.description, candidate.details)

        result = await curator._maybe_flag_unverified_claims(
            candidate, probe=lambda token: False,
        )

        assert isinstance(result, list)
        assert all(isinstance(c, AttributedClaim) for c in result)
        assert not isinstance(result, CuratorDecision)
        assert (candidate.title, candidate.description, candidate.details) == before

    async def test_probe_runs_off_event_loop(self, tmp_path):
        """Remediation RED for the reviewer's performance-async-blocking
        finding: probe verification must run off the event-loop thread.

        make_source_and_history_probe's probe() runs blocking git
        subprocesses (git grep, and — on the exact fabricated-token path
        this guard targets — a full-history `git log --all -S` pickaxe
        too), so calling it synchronously inside this async hook stalls
        the curator/reconciliation event loop for every other coroutine
        sharing it. Proves the offload by recording the thread each probe
        call actually executes on and comparing it to the event loop's
        own thread.
        """
        import threading

        from fused_memory.middleware.task_curator import CandidateTask, TaskCurator

        loop_thread_id = threading.get_ident()
        probe_thread_ids: list[int] = []

        def recording_probe(token: str) -> bool:
            probe_thread_ids.append(threading.get_ident())
            return False

        config = _make_config(recon_claim_verification_enabled=True)
        curator = TaskCurator(config=config, taskmaster=None, cwd=tmp_path)
        candidate = CandidateTask(title="T", description=_INCIDENT_DESCRIPTION)

        result = await curator._maybe_flag_unverified_claims(
            candidate, probe=recording_probe,
        )

        assert len(probe_thread_ids) >= 1
        assert all(tid != loop_thread_id for tid in probe_thread_ids)
        assert len(result) == 1
        assert result[0].token == "done_provenance_invalidated"
