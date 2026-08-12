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

import types
from typing import Any

from _fm_helpers import resolve_xdist_worker_id


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
