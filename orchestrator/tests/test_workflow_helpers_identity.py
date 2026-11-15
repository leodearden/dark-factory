"""Anti-duplication identity guards for every promoted ``_workflow_helpers`` harness.

``_workflow_helpers`` holds test-harness factories promoted out of divergent
per-suite copies (task 3903 onward; Groups A-D, then Group E by task 4384).
Each group pairs a smoke test in ``test_workflow_helpers.py``, which drives the
promoted factories, with an identity guard here, which pins that every consumer
suite is still bound to *those* objects rather than to re-grown local copies.
These are the regression guards for the defect the promotions exist to close --
divergent copies of one harness -- so they assert object IDENTITY, not
equivalent behaviour. A new promotion group puts its guard here.

Import NO ``orchestrator.*`` lane module here -- only ``_workflow_helpers`` and
the consumer suites, by bare test name. A lane import (any module in
``scripts/merge_lane_metrics.py::CLUSTER_PATHS``) enrols this file in the
merge-lane ratchet sweep, which would count every ``<consumer>._<helper>``
assertion below as a private read of lane internals.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Group A: merge-provenance factories
# Group B: warm-lane workflow factory
# Group C: harness factories
# Group D: e2e factories
# ---------------------------------------------------------------------------


def test_merge_provenance_factories_identity() -> None:
    """Anti-duplication guard: the producer re-exports the SAME objects as the shared module."""
    import test_workflow_merge_provenance as mp  # noqa: PLC0415
    from _workflow_helpers import _bind_landed_row, _Fixture, _make  # noqa: PLC0415

    assert mp._make is _make
    assert mp._bind_landed_row is _bind_landed_row
    assert mp._Fixture is _Fixture


def test_warmlane_workflow_factory_identity() -> None:
    """Anti-duplication guard: the producer re-imports the SAME object under its old local name."""
    import test_workflow_warm_lane_requeue as wr  # noqa: PLC0415
    from _workflow_helpers import _make_warmlane_workflow  # noqa: PLC0415

    assert wr._make_workflow is _make_warmlane_workflow


def test_harness_factories_identity() -> None:
    """Anti-duplication guard: the producer re-exports the SAME objects as the shared module."""
    import test_harness_warm_lane_wiring as hw  # noqa: PLC0415
    from _workflow_helpers import _build_harness, _init_git_repo  # noqa: PLC0415

    assert hw._build_harness is _build_harness
    assert hw._init_git_repo is _init_git_repo


def test_e2e_factories_identity() -> None:
    """Anti-duplication guard: the producer re-exports the SAME objects as the shared module.

    Does NOT `from _workflow_helpers import _derive_meta_root_like_production` here —
    that would pull the autouse-fixture NAME into this module's namespace and
    activate it in every test in this file. Instead the fixture is referenced via
    module-attribute access (`_workflow_helpers._derive_meta_root_like_production`),
    which is inert.
    """
    import _workflow_helpers  # noqa: PLC0415
    import test_workflow_e2e as e2e  # noqa: PLC0415
    from _workflow_helpers import (  # noqa: PLC0415
        PLAN,
        AgentStub,
        _build_workflow,
        _build_workflow_with_escalation,
        _init_repo,
        _make_review,
    )

    assert e2e.AgentStub is AgentStub
    assert e2e.PLAN is PLAN
    assert e2e._make_review is _make_review
    assert e2e._build_workflow is _build_workflow
    assert e2e._build_workflow_with_escalation is _build_workflow_with_escalation
    assert e2e._init_repo is _init_repo
    assert e2e._derive_meta_root_like_production is _workflow_helpers._derive_meta_root_like_production


# ---------------------------------------------------------------------------
# Group E: transcript-archival harness (task 4384)
# ---------------------------------------------------------------------------


def test_transcript_archival_factories_identity() -> None:
    """Anti-duplication guard: the producers re-export the SAME objects as the shared module."""
    import test_transcript_archival_boundary_gate as bg  # noqa: PLC0415
    import test_transcript_archive_backstop as bs  # noqa: PLC0415
    import test_transcript_archive_producer_hook as ph  # noqa: PLC0415
    from _workflow_helpers import (  # noqa: PLC0415
        ENC,
        _archive_root,
        _archived,
        _config,
        _config_dir,
        _init_transcript_repo,
        _make_git_ops,
        _make_transcript_workflow,
        _write_transcript,
    )

    # alpha, the producer suite.
    assert ph.ENC is ENC
    assert ph._config is _config
    assert ph._make_transcript_workflow is _make_transcript_workflow
    assert ph._make_git_ops is _make_git_ops
    assert ph._archive_root is _archive_root
    assert ph._archived is _archived

    # beta, the teardown-backstop suite.
    assert bs._make_git_ops is _make_git_ops
    assert bs._write_transcript is _write_transcript
    assert bs._archive_root is _archive_root
    assert bs._archived is _archived

    # epsilon, the B+H boundary gate that had ported the fixtures from both.
    assert bg.ENC is ENC
    assert bg._archive_root is _archive_root
    assert bg._archived is _archived
    assert bg._config is _config
    assert bg._config_dir is _config_dir
    assert bg._make_git_ops is _make_git_ops
    assert bg._make_transcript_workflow is _make_transcript_workflow
    assert bg._write_transcript is _write_transcript

    # All three share ONE real-git seeder object (their local copies hashed
    # identically, md5 217898b53fcac640e02c5374ca2d4001, before promotion).
    assert ph._init_transcript_repo is _init_transcript_repo
    assert bs._init_transcript_repo is _init_transcript_repo
    assert bg._init_transcript_repo is _init_transcript_repo
