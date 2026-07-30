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

import consume_redispatch_requests as crr

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


# ── step-1: request-file discovery and validation ────────────────────────────
#
# The consumer's whole safety posture rests on it acting ONLY on files the
# sweep actually emitted, and never guessing at a body it does not fully
# recognise. These tests pin both halves: the filename predicate (what is even
# looked at) and the strict validator (what is accepted once read).


def _discovered(requests_dir):
    """Basenames the consumer would consider, in whatever order it scans."""
    return sorted(os.path.basename(p) for p in crr.discover_requests(requests_dir))


def test_discovery_picks_up_only_well_formed_request_names(requests_dir):
    write_request(requests_dir, 5321, 'unmet_dependency')
    write_request(requests_dir, 77, 'gate_closure')
    write_request(requests_dir, 88, 'merge_verify_red')
    assert _discovered(requests_dir) == [
        'redispatch-5321-unmet_dependency.json',
        'redispatch-77-gate_closure.json',
        'redispatch-88-merge_verify_red.json',
    ]


def test_discovery_ignores_sweep_tmp_intermediates(requests_dir):
    """redispatch-tmp-XXXXXX is the sweep's mktemp intermediate, mid-write.

    Acting on one would be reading a partial file the atomic-emit contract
    exists to prevent a consumer from ever seeing.
    """
    write_request(requests_dir, 1, 'gate_closure', name='redispatch-tmp-AbCdEf')
    write_request(requests_dir, 1, 'gate_closure', name='redispatch-tmp-XXXXXX.json')
    assert _discovered(requests_dir) == []


def test_discovery_ignores_the_consumed_subdirectory(requests_dir):
    """The consumer's own archive must not be re-consumed on the next run."""
    consumed = requests_dir / 'consumed'
    consumed.mkdir()
    write_request(consumed, 5321, 'gate_closure')
    assert _discovered(requests_dir) == []


def test_discovery_ignores_unrelated_and_malformed_names(requests_dir):
    for name in (
        'README.md',
        'redispatch.json',
        'redispatch-5321.json',                      # no class
        'redispatch--gate_closure.json',             # no id
        'redispatch-abc-gate_closure.json',          # non-numeric id
        'redispatch-5321-unknown_class.json',        # class not in the table
        'redispatch-5321-gate_closure.json.bak',     # not .json
        'sweep-report.txt',
    ):
        write_request(requests_dir, 1, 'gate_closure', name=name)
    assert _discovered(requests_dir) == []


def test_missing_requests_dir_discovers_nothing(tmp_path):
    """An absent dir is a clean zero-request scan, not an error."""
    assert crr.discover_requests(tmp_path / 'nope') == []


def _load(path):
    return crr.load_request(path)


def test_valid_request_loads_into_a_typed_record(requests_dir):
    path = write_request(requests_dir, 5321, 'unmet_dependency',
                         evidence='all deps terminal')
    req = _load(path)
    assert req.skip_reason is None
    assert req.task_id == 5321
    assert req.cls == 'unmet_dependency'
    assert req.action == 'redispatch'
    assert req.verdict == 'STALE'
    assert req.path == path
    assert req.mtime == pytest.approx(os.path.getmtime(path), abs=1)


@pytest.mark.parametrize('cls,action', sorted(CLASS_ACTION.items()))
def test_every_known_class_loads_with_its_fixed_action(requests_dir, cls, action):
    req = _load(write_request(requests_dir, 9, cls))
    assert req.skip_reason is None and req.action == action


def test_wrong_schema_version_is_skipped_not_guessed(requests_dir):
    for bad in (0, 2, '1', None):
        req = _load(write_request(requests_dir, 5, 'gate_closure',
                                  schema_version=bad))
        assert req.skip_reason is not None
        assert 'schema_version' in req.skip_reason


def test_unknown_class_in_body_is_skipped(requests_dir):
    """A body class the consumer has no action table entry for is never acted on."""
    path = write_request(requests_dir, 5, 'gate_closure')
    body = json.load(open(path))
    body['class'] = 'some_future_class'
    open(path, 'w').write(json.dumps(body, indent=2, sort_keys=True) + '\n')
    req = _load(path)
    assert req.skip_reason is not None and 'class' in req.skip_reason


def test_non_integer_task_id_is_skipped(requests_dir):
    for bad in ('5321', 5321.5, None, [5321]):
        req = _load(write_request(requests_dir, bad, 'gate_closure',
                                  name='redispatch-5321-gate_closure.json'))
        assert req.skip_reason is not None
        assert 'task_id' in req.skip_reason


def test_body_task_id_disagreeing_with_the_filename_is_skipped(requests_dir):
    """The name and the body are two statements of the same fact; a mismatch
    means one of them is wrong and there is no basis for choosing."""
    req = _load(write_request(requests_dir, 999, 'gate_closure',
                              name='redispatch-5321-gate_closure.json'))
    assert req.skip_reason is not None and 'task_id' in req.skip_reason


def test_body_class_disagreeing_with_the_filename_is_skipped(requests_dir):
    path = write_request(requests_dir, 5321, 'gate_closure')
    body = json.load(open(path))
    body['class'] = 'unmet_dependency'
    body['action'] = 'redispatch'
    open(path, 'w').write(json.dumps(body, indent=2, sort_keys=True) + '\n')
    req = _load(path)
    assert req.skip_reason is not None and 'class' in req.skip_reason


def test_unparseable_json_is_skipped(requests_dir):
    for raw in ('{not json', '', '[]', 'null', '"a string"'):
        req = _load(write_request(requests_dir, 5, 'gate_closure', raw=raw))
        assert req.skip_reason is not None


def test_missing_required_field_is_skipped(requests_dir):
    for field in ('schema_version', 'task_id', 'class', 'verdict', 'action'):
        path = write_request(requests_dir, 5, 'gate_closure')
        body = json.load(open(path))
        del body[field]
        open(path, 'w').write(json.dumps(body, indent=2, sort_keys=True) + '\n')
        req = _load(path)
        assert req.skip_reason is not None, field


@pytest.mark.parametrize('cls,wrong_action', [
    ('gate_closure', 'redispatch'),
    ('gate_closure', 'reverify'),
    ('merge_verify_red', 'close'),
    ('unmet_dependency', 'close'),
    ('unmet_dependency', 'reverify'),
    ('gate_closure', 'human_gate'),
    ('gate_closure', 'none'),
])
def test_class_action_disagreement_is_skipped(requests_dir, cls, wrong_action):
    """The mapping is FIXED in the sweep. A file that disagrees with it is not
    a request in a dialect we understand -- it is skipped, never reconciled."""
    req = _load(write_request(requests_dir, 5, cls, action=wrong_action))
    assert req.skip_reason is not None and 'action' in req.skip_reason


@pytest.mark.parametrize('verdict', [
    'CORRUPT-HOLD', 'LIVE', 'GATED', 'UNRESOLVED', 'NO-CLASS', 'unknown',
    'stale', 'STALE ',
])
def test_non_stale_verdict_is_a_hard_skip(requests_dir, verdict):
    """The sweep emits ONLY on verdict=STALE, CORRUPT-HOLD included (its
    invariant L5). Anything else in a file is a contract violation upstream,
    so the consumer refuses it rather than inferring intent."""
    req = _load(write_request(requests_dir, 5, 'gate_closure', verdict=verdict))
    assert req.skip_reason is not None and 'verdict' in req.skip_reason
