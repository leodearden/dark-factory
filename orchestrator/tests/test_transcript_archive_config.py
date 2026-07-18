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
    RetentionConfig,
    TranscriptArchiveConfig,
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
        fresh = OrchestratorConfig(
            transcript_archive=TranscriptArchiveConfig(enabled=False)
        )
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        assert 'transcript_archive.enabled' in report['applied']
        assert report['applied']['transcript_archive.enabled'] == {
            'old': True,
            'new': False,
        }
        assert live.transcript_archive.enabled is False

    def test_retention_edit_hot_applies_as_whole_submodel_swap(
        self, monkeypatch, tmp_path
    ):
        """A retention.* edit reloads as the claimed atomic whole-retention swap.

        _iter_leaves descends exactly one level, so transcript_archive.retention
        is one atomic BaseModel leaf: changing a single nested knob
        (max_age_days) swaps the whole RetentionConfig, carrying the untouched
        sibling (max_task_dirs) along. Guards against a regression in
        _iter_leaves/_set_leaf descent depth for this nested submodel.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        live = OrchestratorConfig()
        assert live.transcript_archive.retention.max_age_days == 90
        fresh = OrchestratorConfig(
            transcript_archive=TranscriptArchiveConfig(
                retention=RetentionConfig(max_age_days=30)
            )
        )
        report = apply_reload(live, fresh)
        assert report['reloaded'] is True
        # The whole retention submodel is the reloaded leaf (green-tier).
        assert 'transcript_archive.retention' in report['applied']
        # The nested edit is live, and the untouched sibling rode along in the
        # whole-submodel swap (retention.max_task_dirs still at its default).
        assert live.transcript_archive.retention.max_age_days == 30
        assert live.transcript_archive.retention.max_task_dirs == 5000
