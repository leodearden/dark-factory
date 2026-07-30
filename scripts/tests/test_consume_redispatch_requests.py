"""Tests for scripts/consume_redispatch_requests.py (task 3102).

The consumer drains the request files emitted by reify's
``scripts/deterministic-gate-closure-staleness-sweep.sh --emit-requests``.
That script is the NORMATIVE contract; this module's fixtures render request
files byte-faithfully to its ``_write_request`` so the parser under test is
exercised against the real on-disk shape rather than a paraphrase of it.

``scripts/tests/conftest.py`` already inserts ``scripts/`` onto ``sys.path``,
so ``import consume_redispatch_requests`` resolves with no packaging work.
"""
import json
import os

import pytest

# The fixed class -> action mapping, transcribed from the reify sweep
# (_classify_gate_closure -> close, _classify_merge_verify_red -> reverify,
# _classify_unmet_dependency -> redispatch). Tests assert the consumer agrees
# with THIS table; the consumer must never infer an action from a class it
# does not recognise.
CLASS_ACTION = {
    'gate_closure': 'close',
    'merge_verify_red': 'reverify',
    'unmet_dependency': 'redispatch',
}


def request_body(task_id, cls, *, verdict='STALE', action=None,
                 evidence='dependency roll-up satisfied', schema_version=1,
                 main_ref_sha='0' * 40, emitted_by='sweep@host'):
    """Build a request body dict with the sweep's exact key set.

    Defaults produce a VALID request; every field is overridable so a test can
    corrupt exactly one thing at a time.
    """
    return {
        'schema_version': schema_version,
        'task_id': task_id,
        'class': cls,
        'verdict': verdict,
        'action': CLASS_ACTION.get(cls, 'close') if action is None else action,
        'evidence': evidence,
        'main_ref_sha': main_ref_sha,
        'emitted_by': emitted_by,
    }


def write_request(requests_dir, task_id, cls, *, mtime=None, name=None,
                  raw=None, **body_kwargs):
    """Write one request file byte-faithfully to the sweep's ``_write_request``.

    ``json.dump(..., indent=2, sort_keys=True)`` followed by a single trailing
    newline, named ``redispatch-<task_id>-<class>.json`` -- exactly what the
    reify script emits (via its embedded python3 renderer).

    ``mtime`` sets the file's modification time, which is the ONLY recency
    signal the contract provides (the body deliberately carries no wall-clock
    field, so re-emission is byte-idempotent). The compare-and-swap guard reads
    it, so tests must be able to drive it both ways.

    ``raw`` writes the given text verbatim instead of rendering a body, for
    the unparseable-JSON cases. ``name`` overrides the filename, for the
    discovery-predicate cases.
    """
    requests_dir = str(requests_dir)
    fname = name if name is not None else f'redispatch-{task_id}-{cls}.json'
    path = os.path.join(requests_dir, fname)
    if raw is not None:
        text = raw
    else:
        body = request_body(task_id, cls, **body_kwargs)
        text = json.dumps(body, indent=2, sort_keys=True) + '\n'
    with open(path, 'w') as fh:
        fh.write(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def requests_dir(tmp_path):
    """An empty requests directory, as the sweep would create it on demand."""
    d = tmp_path / 'redispatch-requests'
    d.mkdir()
    return d


def test_write_request_is_byte_faithful_to_the_sweep(requests_dir):
    """The fixture renders what the sweep's _write_request renders.

    Pins the helper itself: sorted keys, two-space indent, trailing newline,
    integer task_id, and the sweep's exact eight-key body. If reify's renderer
    ever changes shape, this is the test that should fail first.
    """
    path = write_request(requests_dir, 5321, 'unmet_dependency',
                         evidence='all deps terminal',
                         main_ref_sha='abc123', emitted_by='sweep@box')
    assert os.path.basename(path) == 'redispatch-5321-unmet_dependency.json'
    text = open(path).read()
    assert text == json.dumps({
        'action': 'redispatch',
        'class': 'unmet_dependency',
        'emitted_by': 'sweep@box',
        'evidence': 'all deps terminal',
        'main_ref_sha': 'abc123',
        'schema_version': 1,
        'task_id': 5321,
        'verdict': 'STALE',
    }, indent=2, sort_keys=True) + '\n'
    body = json.loads(text)
    assert set(body) == {
        'schema_version', 'task_id', 'class', 'verdict', 'action',
        'evidence', 'main_ref_sha', 'emitted_by',
    }
    assert body['task_id'] == 5321 and isinstance(body['task_id'], int)
    assert body['verdict'] == 'STALE'
    assert body['action'] == 'redispatch'


def test_write_request_mtime_is_settable(requests_dir):
    """The mtime seam the compare-and-swap guard depends on actually works."""
    path = write_request(requests_dir, 42, 'gate_closure', mtime=1_000_000)
    assert os.path.getmtime(path) == pytest.approx(1_000_000, abs=1)
