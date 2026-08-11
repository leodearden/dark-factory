"""Tests for the fetch-replay cache in bake_off_storage_shape.py (task 4004).

`fetch_arm` is the only part of the E2 bake-off that costs an embedder call
and a live Qdrant collection; everything downstream of it (`read_path`,
`measure_arm`, `rescore`, `build_report`, `render_markdown`) is already pure.
Dumping its return value and replaying it makes every read-side variant free
and — the point of this module — makes the metric code unit-testable against
REAL rankings rather than only hand-built ones.

The script is loaded via importlib so it can be tested without sys.path
pollution — the loader is copied verbatim from
``test_bake_off_storage_shape.py:48-73`` and is invoked lazily (via ``_mod()``)
rather than bound at import time.

LANE DISCIPLINE — READ BEFORE ADDING A TEST
-------------------------------------------
Every test in this file must be free of network, Qdrant and OPENAI_API_KEY
**except a live end-to-end test**, which carries its markers PER-TEST::

    @pytest.mark.integration
    @pytest.mark.timeout(600)
    @qdrant_skipif()
    @pytest.mark.skipif(not os.environ.get('OPENAI_API_KEY'), ...)

Never via a module-level ``pytestmark``.  ``fused-memory/pyproject.toml``
sets ``addopts = "-n auto --dist loadgroup -m 'not integration'"``, so a
module-level integration marker would deselect every pure test in this file
from the merge lane too — see the same warning at
``test_bake_off_storage_shape.py:9-24``.

This file does NOT extend ``test_bake_off_storage_shape.py``: task 3560 is
in-progress and claims that module.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'bake_off_storage_shape.py'

FIXTURES_DIR = Path(__file__).parent / 'fixtures'

#: MEASUREMENT ANCHOR — recorded BEFORE a line of task-4004 code was written,
#: so a later reader can tell "the cache replays a different corpus" from "the
#: cache is truncated" without re-deriving either.
#:
#: Anchor commit: ff303320c7c3d90b093076965992dac246db062a
#: Live-run environment confirmed available at that commit:
#:   - Qdrant  http://localhost:6333/collections -> HTTP 200
#:   - OPENAI_API_KEY set
#: If either is unavailable when the measurement run happens, the run
#: ESCALATES (category='infra_issue').  No number is ever estimated, fabricated
#: or hand-edited into the report artifacts: every measured cell comes from a
#: real run, or renders as the no-measurement em dash.
#:
#: sha256 of the five committed E2 fixtures at the anchor commit.  These are
#: the inputs `materialize_arm` is deterministic over, so a fetch cache dumped
#: against them is replayable exactly as long as they hash to these values.
#: Asserted here rather than only inside the cache so a fixture edit that
#: silently invalidates the committed cache fails a PURE test in the merge
#: lane, not a live run nobody reruns.
ANCHOR_COMMIT = 'ff303320c7c3d90b093076965992dac246db062a'

ANCHOR_FIXTURE_SHA256: dict[str, str] = {
    'write_triage_calibration.jsonl':
        'fa5958f3634ace98b846ac398cdfe28f2e105a746f0348fe48fb5ed08cd03fe3',
    'memory_eval_topic_registry.json':
        '23b5ba77d59b10854a000fe57c2ef4766033bedfd51335de45bcec467ae3ae30',
    'e2_arm_claims.jsonl':
        '0b09c7de1c30c38570543f1705f01c5b4ac5970618f64545facb486e6991c257',
    'e2_query_set.jsonl':
        'c0c4872d2bb76e5e28a3e6660cf80d4b838712fdbf938624d02a0217e12c26d0',
    'e2_distractor_slab.jsonl':
        '8663a11024d14fb7201591a191f33a628d26f44d2449298db9182fef66b57e57',
}


def _load_module() -> types.ModuleType:
    """Load bake_off_storage_shape.py from its file path.

    The module is registered in sys.modules under its bare name so that
    @dataclass and other reflection-based decorators work correctly (they
    call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'bake_off_storage_shape'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


@functools.cache
def _mod() -> types.ModuleType:
    return _load_module()


# ===========================================================================
# pre-2 — the measurement anchor, asserted rather than merely commented
# ===========================================================================

import hashlib  # noqa: E402

import pytest  # noqa: E402


class TestMeasurementAnchor:
    """The committed fetch cache is only replayable against THESE fixtures.

    `materialize_arm` is deterministic over the five committed fixtures, and
    the fetch cache stores `(shape, query_id) -> [(record_id, score)]` keyed
    on the uuid5 `record_id` those fixtures derive.  Edit a fixture and the
    cache still LOADS — it just describes a corpus that no longer exists, and
    the replayed report would publish a stale ranking as a fresh measurement.

    This is the cheap, pure, merge-lane half of that guard: it fails the
    moment a fixture moves, in a test that actually runs, instead of only
    inside a live run nobody reruns.  The expensive half — the per-shape
    corpus fingerprint carried in the dump — is step-3.
    """

    @pytest.mark.parametrize('name', sorted(ANCHOR_FIXTURE_SHA256))
    def test_fixture_still_hashes_to_the_anchor(self, name: str) -> None:
        path = FIXTURES_DIR / name
        assert path.exists(), f'{name} vanished since {ANCHOR_COMMIT}'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == ANCHOR_FIXTURE_SHA256[name], (
            f'{name} changed since the task-4004 measurement anchor '
            f'{ANCHOR_COMMIT}. The committed fetch cache '
            f'(tests/fixtures/e2_fetch_cache.json) was dumped against the old '
            f'bytes, so replaying it now would measure a corpus that no longer '
            f'exists. Re-run the seeding pass with --dump-fetches and update '
            f'this anchor in the same commit — do NOT just edit the expected '
            f'digest.'
        )

    def test_anchor_covers_every_fixture_the_bake_off_defaults_to(self) -> None:
        """No fixture may drift out of the anchor's coverage unnoticed."""
        mod = _mod()
        defaults = [
            mod.DEFAULT_ALPHA_FIXTURE_PATH,
            mod.DEFAULT_REGISTRY_PATH,
            mod.DEFAULT_ARM_CLAIMS_PATH,
            mod.DEFAULT_QUERY_SET_PATH,
            mod.DEFAULT_DISTRACTOR_SLAB_PATH,
        ]
        assert {Path(p).name for p in defaults} == set(ANCHOR_FIXTURE_SHA256)
