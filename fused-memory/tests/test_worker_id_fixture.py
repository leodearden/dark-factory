"""Guards for the `worker_id` fixture every per-worker-isolated test here needs.

The failure this file exists to prevent: the offline-deep lane re-runs a red
lane command through the serial-pytest helper, which appends
`-p no:xdist -o addopts=` (orchestrator/src/orchestrator/verify_cmd.py).
`-p no:xdist` unregisters the pytest-xdist plugin — and that takes the
plugin's FIXTURES with it, not just its `-n`/`--dist` CLI options.  Since
`worker_id` was supplied *solely* by pytest-xdist (xdist/plugin.py:391),
every fused-memory test requesting it — directly or through a per-worker
isolation fixture — ERRORED at setup with `fixture 'worker_id' not found`
in that serial confirm re-run, and the lane reported those setup ERRORs
through the same channel as assertion FAILUREs, mis-attributing them to an
unrelated merge commit.

A developer typing `pytest -p no:xdist` locally hits the identical wall, so
this is a genuine latent defect in the suite rather than a lane quirk.
"""

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import resolve_xdist_worker_id

_FUSED_MEMORY_DIR = Path(__file__).resolve().parents[1]
_CONTRACT_NODE_ID = 'tests/test_worker_id_fixture.py::test_worker_id_matches_xdist_semantics'


def _fake_request(workerinput: dict[str, str] | None) -> Any:
    """A duck-typed stand-in for pytest's `FixtureRequest`/`Session`.

    Returned as `Any` deliberately.  `resolve_xdist_worker_id` names the real
    pytest types in its signature (mirroring xdist's own), but the contract
    under test is purely structural: `hasattr(x.config, 'workerinput')` then
    `x.config.workerinput['workerid']`, else `'master'`.  Exercising it with a
    `SimpleNamespace` pins that observable contract without standing up a real
    pytest session — so these tests survive a future reimplementation that
    stops delegating to xdist.
    """
    config = types.SimpleNamespace()
    if workerinput is not None:
        config.workerinput = workerinput
    return types.SimpleNamespace(config=config)


def test_resolve_xdist_worker_id_reads_workerinput():
    """Under a real xdist worker, the id comes from `config.workerinput`."""
    assert resolve_xdist_worker_id(_fake_request({'workerid': 'gw7'})) == 'gw7'


def test_resolve_xdist_worker_id_falls_back_to_master():
    """With no `workerinput` on the config — the `-p no:xdist` case — it is 'master'."""
    assert resolve_xdist_worker_id(_fake_request(None)) == 'master'


def test_resolve_xdist_worker_id_always_returns_str():
    """Consumers suffix namespaces with the result, so it must always be a str."""
    assert isinstance(resolve_xdist_worker_id(_fake_request({'workerid': 'gw7'})), str)
    assert isinstance(resolve_xdist_worker_id(_fake_request(None)), str)


def test_worker_id_matches_xdist_semantics(worker_id, pytestconfig):
    """The `worker_id` fixture agrees with xdist's own value for this session.

    Also the payload the wiring test below re-runs in a nested serial pytest —
    which is why it must depend on nothing but the fixture itself.  In the
    OUTER run this passes today (xdist supplies `worker_id`), so it is not the
    red signal; registration is what the wiring test proves.
    """
    assert isinstance(worker_id, str) and worker_id
    workerinput = getattr(pytestconfig, 'workerinput', None)
    expected = workerinput['workerid'] if workerinput is not None else 'master'
    assert worker_id == expected


# `timeout(180)` is load-bearing, not padding: this project's ini sets
# `timeout = 60` with `timeout_method = "thread"`, and pytest-timeout's thread
# handler ends in `os._exit(1)` — firing it under `-n auto --dist loadgroup`
# kills the whole xdist worker.  Measured wall for the nested run is ~13s.
@pytest.mark.timeout(180)
def test_worker_id_survives_the_lanes_serial_confirm_rerun():
    """A nested pytest under the lane's exact confirm-re-run flags must succeed.

    This is the test that actually catches the defect.  The unit tests above
    prove `resolve_xdist_worker_id`'s semantics but say nothing about whether
    conftest REGISTERS a `worker_id` fixture — and registration is the entire
    bug.  Nothing in-process can prove it, because the outer run already has
    xdist's own fixture, so an in-process assertion is green before the fix.

    Selecting one explicit node id means the child does NOT collect this test,
    so there is no recursion.  Both `rc == 0` and `'1 passed'` are asserted so
    an exit-5 "no tests ran" cannot masquerade as success.  Deliberately NOT
    marked `integration`: it needs no external service and must run on the
    ordinary verify hot path, which is where it guards the lane.
    """
    # Scrub PYTEST_ADDOPTS/PYTEST_CURRENT_TEST so the outer pytest run's
    # options/state do not leak into the child invocation.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ('PYTEST_ADDOPTS', 'PYTEST_CURRENT_TEST')
    }
    proc = subprocess.run(
        [
            sys.executable, '-m', 'pytest',
            _CONTRACT_NODE_ID,
            '-p', 'no:xdist', '-o', 'addopts=',
            '-p', 'no:cacheprovider', '-q',
        ],
        cwd=str(_FUSED_MEMORY_DIR),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        'The serial confirm re-run (`-p no:xdist -o addopts=`) failed — the '
        "`worker_id` fixture is unavailable without the xdist plugin.\n"
        f'rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}'
    )
    assert '1 passed' in proc.stdout, (
        f'Expected exactly the contract test to run and pass.\n{proc.stdout}\n{proc.stderr}'
    )
