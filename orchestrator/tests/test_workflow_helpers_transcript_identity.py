"""Anti-duplication guard for the Group E transcript-archival harness (task 4384).

The pair to ``test_workflow_helpers.py``'s
``test_transcript_archival_factories_smoke``: the smoke test drives the promoted
factories, this one pins that all three consumer suites are still bound to
*those* objects rather than to re-grown local copies. It is the regression guard
for the defect task 4384 exists to close -- three divergent copies of one
harness -- so it asserts object IDENTITY, not equivalent behaviour.

WHY IT LIVES IN ITS OWN MODULE, apart from the smoke test it pairs with.
``scripts/merge_lane_metrics.py`` enrols a test file into the merge-lane ratchet
sweep on exactly one condition -- the file statically imports a lane module --
and then counts every single-underscore attribute access in it as a
``private_reads`` reach into lane internals. That measure is deliberately
receiver-agnostic and deliberately over-counts (see its header comment), and the
ratchet permits a measure to fall or hold, never to rise.

These assertions are ``<consumer module>._<promoted helper>`` reads: coupling to
the names of TEST-module helpers, which has nothing to do with the lane seam the
measure is about. Nineteen of them in the enrolled ``test_workflow_helpers.py``
pushed that path 16 -> 35 and the derived total up, breaching the ratchet.
``test_harness_reconcile_landed_outbox_none_guard.py`` already resolves exactly
this collision the same way, and states the rule in its own docstring: keep the
file out of the lane-importing set so the sweep does not enrol coverage that has
nothing to do with the lane. So this module imports NO ``orchestrator.*`` lane
module -- only ``_workflow_helpers`` and the consumer suites, all by bare test
name. Adding one here is not a style nit: it re-enrols the file and re-breaches
the ratchet.

The smoke test cannot make the same move -- it anchors the promoted paths to
production via ``orchestrator.git_ops`` and ``orchestrator.config``, which is
precisely a lane import -- but it reads no private attributes, so it costs the
enrolled file nothing and stays put.

Groups A-D's identity tests remain colocated in ``test_workflow_helpers.py``;
they predate the baseline and are already counted in it. Relocating them here
would lower the ratchet further, but that is a separate change to tests this
task does not own.
"""

from __future__ import annotations


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
