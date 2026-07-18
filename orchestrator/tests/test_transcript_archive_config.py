"""Config coverage for the transcript_archive block (task 2742, PRD α).

Fixtures are kept module-local (no conftest.py) — a conftest.py edit trips
verify.py's has_conftest and forces the merge-time verify to run the full
owning-package suite instead of a scoped subset (mirrors the rationale in
test_config_verify_admission_reload.py).
"""

from __future__ import annotations

from orchestrator.config import (
    RELOADABLE_FIELDS,
    OrchestratorConfig,
    apply_reload,
)


class TestTranscriptArchiveDefaults:
    """The block is fully-formed from Pydantic field defaults (no defaults.yaml)."""

    def test_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        ta = cfg.transcript_archive
        assert ta.enabled is True
        assert ta.root == 'data/orchestrator/agent-transcripts'
        assert ta.retention.max_age_days == 90
        assert ta.retention.max_task_dirs == 5000


class TestTranscriptArchiveReloadable:
    """Every transcript_archive.* leaf is green-tier (whole-submodel group);
    retention is compared as one atomic BaseModel leaf.
    """

    def test_leaves_registered_green_tier(self):
        assert 'transcript_archive.enabled' in RELOADABLE_FIELDS
        assert 'transcript_archive.root' in RELOADABLE_FIELDS
        assert 'transcript_archive.retention' in RELOADABLE_FIELDS

    def test_enabled_flip_hot_applies(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        fresh = OrchestratorConfig(transcript_archive={'enabled': False})
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert 'transcript_archive.enabled' in report['applied']
        assert report['applied']['transcript_archive.enabled'] == {
            'old': True,
            'new': False,
        }
        assert live.transcript_archive.enabled is False
