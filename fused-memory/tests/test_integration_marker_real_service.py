"""Behavioral guards for the `integration` marker on real-service probes (task 3020).

Follow-up to task 3019 (which marks only test_recon_dedup_premise.py). Several
fused-memory test modules interleave fast MagicMock/AsyncMock unit tests with a
trailing class (or, for test_list_indices_integration.py, a whole module) that
round-trips a *real* Qdrant / FalkorDB backend, gated only by a
qdrant_skipif() / _falkor_available() skip helper. Under the merge lane's
post-merge full-repo sweep those live probes are collected and — with a
service reachable but the box CPU-oversubscribed — thrash the backend into the
"node down" failures 3019 fixes for the recon-dedup probe. This module marks
each such live class/module @pytest.mark.integration so fused-memory's
`-m 'not integration'` addopts (pyproject.toml, task 2736) deselects it from
the default suite, while it stays selectable via `-m integration`
(offline-lane / integration tier, tasks 2790/2791). No assertion is weakened;
the skip guards are retained belt-and-braces for the addopts-cleared serial
rerun path.

Each guard proves the deselection wiring BEHAVIORALLY, not by string-matching
config: it runs a real `pytest --collect-only` subprocess bound to
fused-memory's ACTUAL pyproject.toml (via -c), so the real addopts govern
collection. Collection imports qdrant_client/falkordb but never connects, and
skipif does not prevent collection, so the guards are hermetic and need no
running service.

This is a NEW module rather than an extension of
test_integration_marker_config.py: sibling task 3019 edits that file, and 3020
runs in parallel — keeping the file sets disjoint avoids a concurrency-lock /
merge collision between them.

RED before each corresponding impl step: the live class/module is still tagged
only with its skip guard (not @pytest.mark.integration), so it is collected by
default (the "deselected" assertion fails) and is NOT selected under
`-m integration` (the "selectable" anti-vacuity assertion fails).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
FUSED_MEMORY_DIR = TESTS_DIR.parent
FUSED_MEMORY_PYPROJECT = FUSED_MEMORY_DIR / 'pyproject.toml'


def _collect(module_path: Path, *extra_args: str) -> str:
    """Run `pytest --collect-only` on a real module bound to fused-memory's pyproject.toml.

    Returns combined stdout+stderr. `-n0` overrides fused-memory's `-n auto`
    addopts to keep collection serial without disabling the xdist plugin
    outright (`-p no:xdist` would conflict with the surviving `-n auto` and
    make pytest exit with "unrecognized arguments: -n"). Binding `-c` to the
    real pyproject.toml means the real `-m 'not integration'` addopts govern
    the default collection.

    Collecting these real modules imports mem0/qdrant_client/falkordb (never
    connecting), so the caller gets generous timeout headroom via
    @pytest.mark.timeout(120) over each file's own default budget.
    """
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '--collect-only',
            '-q',
            '-p',
            'no:cacheprovider',
            '-n0',
            '-c',
            str(FUSED_MEMORY_PYPROJECT),
            *extra_args,
            str(module_path),
        ],
        capture_output=True,
        text=True,
        timeout=90,
        cwd=str(FUSED_MEMORY_DIR),
    )
    return result.stdout + result.stderr


@pytest.mark.timeout(120)
def test_mem0_client_qdrant_probe_gated() -> None:
    """The real-Qdrant round-trip class must be gated; its mock siblings must not be.

    TestMem0BackendAddSystemRecordIntegration constructs a real
    QdrantClient(url=QDRANT_URL) and must be deselected by default, while the
    fast mock class TestMem0BackendSearch (control) stays collected. Under
    `-m integration` the live class must become selectable (anti-vacuity).
    """
    module = TESTS_DIR / 'test_mem0_client.py'
    live_class = 'TestMem0BackendAddSystemRecordIntegration'
    mock_anchor = 'TestMem0BackendSearch'

    default_output = _collect(module)
    assert live_class not in default_output, (
        f'{live_class} was collected by default -- it must be marked '
        f"@pytest.mark.integration (task 3020) so fused-memory's -m 'not integration' "
        f'addopts deselects the real-Qdrant round-trip. Output:\n{default_output}'
    )
    assert mock_anchor in default_output, (
        f'expected the mock sibling class {mock_anchor} to remain collected by '
        f'default (fast unit coverage must not be dropped). Output:\n{default_output}'
    )

    override_output = _collect(module, '-m', 'integration')
    assert live_class in override_output, (
        f'expected {live_class} to be collected under -m integration (it must '
        f'remain selectable as an explicit opt-in). Output:\n{override_output}'
    )


@pytest.mark.timeout(120)
def test_list_indices_integration_module_gated() -> None:
    """The whole live-FalkorDB module must be deselected by default, selectable via marker.

    test_list_indices_integration.py is 100% live FalkorDB (both classes
    round-trip a real backend), so its module-level pytestmark carries the
    integration marker. Both classes must be deselected by default and both
    selectable under `-m integration` (anti-vacuity).
    """
    module = TESTS_DIR / 'test_list_indices_integration.py'
    live_classes = ('TestCallDbIndexesOverRoQuery', 'TestBackendListIndicesLive')

    default_output = _collect(module)
    for cls in live_classes:
        assert cls not in default_output, (
            f'{cls} was collected by default -- the whole live-FalkorDB module '
            f'must be marked @pytest.mark.integration (task 3020) so the '
            f"-m 'not integration' addopts deselects it. Output:\n{default_output}"
        )

    override_output = _collect(module, '-m', 'integration')
    for cls in live_classes:
        assert cls in override_output, (
            f'expected {cls} to be collected under -m integration (the module '
            f'must remain selectable as an explicit opt-in). Output:\n{override_output}'
        )


@pytest.mark.timeout(120)
def test_startup_identity_scan_live_classes_gated() -> None:
    """Both live-FalkorDB scan classes gated; the fast mock scan class stays collected.

    TestScanDuplicateEntityNamesLiveFalkorDB and
    TestRepairDuplicateEdgeUuidsLiveFalkorDB each round-trip a real FalkorDB
    and must be deselected by default, while the mock class
    TestScanDuplicateEntityNames (control) stays collected. The mock anchor is
    matched with a trailing '::' delimiter so it cannot be satisfied vacuously
    by the live class name TestScanDuplicateEntityNamesLiveFalkorDB, of which
    it is a prefix. Both live classes must be selectable under `-m integration`.
    """
    module = TESTS_DIR / 'test_startup_identity_scan.py'
    live_classes = (
        'TestScanDuplicateEntityNamesLiveFalkorDB',
        'TestRepairDuplicateEdgeUuidsLiveFalkorDB',
    )
    mock_anchor = 'TestScanDuplicateEntityNames::'

    default_output = _collect(module)
    for cls in live_classes:
        assert cls not in default_output, (
            f'{cls} was collected by default -- it must be marked '
            f"@pytest.mark.integration (task 3020) so the -m 'not integration' "
            f'addopts deselects the real-FalkorDB round-trip. Output:\n{default_output}'
        )
    assert mock_anchor in default_output, (
        f'expected the mock class {mock_anchor} to remain collected by default '
        f'(fast unit coverage must not be dropped). Output:\n{default_output}'
    )

    override_output = _collect(module, '-m', 'integration')
    for cls in live_classes:
        assert cls in override_output, (
            f'expected {cls} to be collected under -m integration (it must '
            f'remain selectable as an explicit opt-in). Output:\n{override_output}'
        )
