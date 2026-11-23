"""Tests for orchestrator.fleet_files, the file primitives the fleet-root registries share.

Covers fleet-root resolution, the atomic writer's on-disk semantics and
encoding, and the stable-sidecar exclusive lock.  The bare-shell import
contract of the module lives with the other entrypoints in
test_session_registry.py::TestStdlibOnlySelfContainment.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _utf8_codec import names_the_utf8_codec

from orchestrator import fleet_files

#: The directory that plays the role of ``orchestrator/src``, resolved from
#: THIS FILE so the encoding child below is CWD-independent.
_PKG_ROOT = Path(__file__).resolve().parents[1] / 'src'


def test_fleet_root_defaults_to_dot_claude_fleet(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('CLAUDE_FLEET_ROOT', raising=False)
    monkeypatch.setenv('HOME', '/home/fakeuser')
    assert fleet_files.fleet_root() == Path('/home/fakeuser/.claude/fleet')


def test_fleet_root_honors_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', str(tmp_path))
    assert fleet_files.fleet_root() == tmp_path


def test_fleet_root_explicit_root_overrides_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv('CLAUDE_FLEET_ROOT', '/should/not/be/used')
    other = tmp_path / 'other'
    assert fleet_files.fleet_root(root=other) == other


def test_exclusive_sidecar_lock_creates_its_parent_and_a_sidecar_that_outlives_release(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / 'missing' / 'parent' / 'thing.json.lock'

    with fleet_files.exclusive_sidecar_lock(lock_path):
        assert lock_path.is_file()

    assert lock_path.is_file(), 'the sidecar is a stable inode, never removed'


def _utf8_child_env(**overrides: str) -> dict[str, str]:
    """Scrubbed env for the ``atomic_write_text`` encoding child process.

    Built from scratch rather than from ``os.environ`` so a developer's shell
    (``LANG``, ``PYTHONUTF8``, ``PYTHONWARNDEFAULTENCODING``, ...) cannot
    silently change what these pins measure, and so ``overrides`` is the ONLY
    source of each child's encoding-relevant knobs.

    ``PYTHONPATH`` is derived from ``_PKG_ROOT`` rather than from the CWD, so
    these behave identically under the canonical ``cd orchestrator && pytest
    tests/`` and under an ad-hoc run from the repo root.  The scrub set mirrors
    ``test_session_registry.py::_row_env``'s discipline for the same reasons
    documented there: an explicit ``PATH``/``HOME``, no user site-packages, and
    no bytecode written back into the source tree.

    ``fleet_files`` is stdlib-only by design (see
    ``test_session_registry.py::TestStdlibOnlySelfContainment``), so
    ``sys.executable`` needs nothing installed here and a non-venv interpreter
    is not required — these pins measure write ENCODING, not importability
    without a venv.
    """
    env = {
        'PATH': '/usr/bin:/bin',
        'HOME': os.environ.get('HOME', '/tmp'),
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONNOUSERSITE': '1',
        'PYTHONPATH': str(_PKG_ROOT),
    }
    env.update(overrides)
    return env


#: The non-ASCII JSON payload the child script below writes.  It reaches the
#: child as a ``\\u00e9`` ESCAPE (via ``json.dumps``, whose output is a valid
#: Python string literal for this content) so the generated script file is pure
#: ASCII and the payload rides in the FILE rather than in argv.  Keep it that
#: way if this pin is ever re-parameterised: MEASURED while building it, a
#: literal non-ASCII character cannot cross a process boundary in argv under a
#: C locale at all — ``python3 -c`` with one dies decoding argv before Python
#: starts — so an ASCII-only child source keeps a RED here meaning "the
#: contract broke" rather than "the harness misfired".
_UTF8_PIN_PAYLOAD = '{"note": "café"}'

_ENCODING_WARNING_CHILD = '''\
"""Write one record via fleet_files with implicit encodings made fatal."""
import pathlib
import sys
import warnings

from orchestrator import fleet_files

target = pathlib.Path(sys.argv[1])

# SCOPED to this one call, deliberately.  Promoting EncodingWarning for the
# whole interpreter (``-W error::EncodingWarning``, which this child used to
# rely on) makes ANY unrelated implicit-encoding text open anywhere on the
# ``from orchestrator import fleet_files`` path fail this test with a message
# about task 3387 -- a false RED pointing at the wrong line.  The import is
# therefore left outside the block; only the write under test is strict.
with warnings.catch_warnings():
    warnings.simplefilter('error', EncodingWarning)
    fleet_files.atomic_write_text(target, %(payload)s)
'''


class TestAtomicWriteSemantics:
    """``atomic_write_text``'s on-disk semantics, pinned WITHOUT a delegation seam.

    Task 3223 consolidated the repo's tmp+rename writers into
    ``shared.safe_io.atomic_write_text``, but this module is the deliberate
    exception: it is stdlib-only so ``skills/spawn/spawn-claude.sh`` can run
    session_registry.py, which imports it, with no venv (see
    ``test_session_registry.py::TestStdlibOnlySelfContainment`` and
    ``_ALLOWED_RENAMERS`` in ``tests/scripts/test_atomic_write_regrowth.py``).

    So these assert the OBSERVABLE result on disk rather than that a particular
    helper was called. That is the more durable pin anyway: it holds whether
    the body is inlined or delegated, and it does not go stale the next time
    the implementation choice is revisited.
    """

    def test_writes_0600_and_leaves_no_temp_residue(self, tmp_path):
        """The record is created 0600 (mkstemp's mode), with no tmp left behind."""
        target = tmp_path / 'record.json'

        fleet_files.atomic_write_text(target, '{"a": 1}')

        assert target.read_text() == '{"a": 1}'
        assert target.stat().st_mode & 0o777 == 0o600, (
            'session records hold session state and must stay owner-only'
        )
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    def test_creates_missing_parent_dirs(self, tmp_path):
        """The parent chain is created rather than surfacing FileNotFoundError."""
        target = tmp_path / 'nested' / 'deeper' / 'record.json'

        fleet_files.atomic_write_text(target, '{"a": 1}')

        assert target.read_text() == '{"a": 1}'

    def test_overwrites_atomically_leaving_no_residue(self, tmp_path):
        """A rewrite replaces the contents wholesale and leaves no tmp file."""
        target = tmp_path / 'record.json'

        fleet_files.atomic_write_text(target, '{"a": 1}')
        fleet_files.atomic_write_text(target, '{"b": 2}')

        assert target.read_text() == '{"b": 2}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    def test_failed_write_leaves_original_intact_and_no_residue(
        self, tmp_path, monkeypatch
    ):
        """On a mid-write failure the original survives and the tmp is cleaned up."""
        target = tmp_path / 'record.json'
        fleet_files.atomic_write_text(target, '{"original": true}')

        def _boom(*_args, **_kwargs):
            raise OSError('disk full')

        monkeypatch.setattr(fleet_files.os, 'replace', _boom)

        with pytest.raises(OSError, match='disk full'):
            fleet_files.atomic_write_text(target, '{"replacement": true}')

        assert target.read_text() == '{"original": true}'
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    # -- task 3387: the payload is JSON, and RFC 8259 requires JSON on disk to
    # be UTF-8.  Under a non-UTF-8 locale the old bare ``os.fdopen(fd, 'w')``
    # wrote bytes that the registries' own readers then reject —
    # ``session_registry.py::read_record`` raises ``CorruptSessionRecord``,
    # ``decision_registry.py::list_decisions`` and
    # ``session_leases.py::reap_stale_leases`` log and skip the file — i.e.
    # silent data loss.  (NOT
    # ``shared.safe_io.load_json_or_warn``, which an earlier version of this
    # comment named: that helper never reads session, decision or lease
    # records.)  Two pins, deliberately different in KIND, because neither is
    # trustworthy alone:
    #
    #   1. a boundary spy   — fast, deterministic, host-independent; localises
    #      the defect to the exact argument
    #   2. an EncodingWarning child — the interpreter itself names the defect,
    #      and byte-asserts the result, so the BYTES on disk are pinned too
    #
    # A third pin, a genuine ``LC_ALL=C`` child, was written, mutation-verified
    # and then REMOVED as disproportionate: it pinned the same contract as 2
    # while being the only one that can go vacuous (it self-skips whenever the
    # host coerces the C locale to UTF-8), and it cost a second subprocess in
    # an already subprocess-heavy suite.  Do not re-add it without a failure
    # mode 1 and 2 provably cannot catch; if you do, all four of LC_ALL=C,
    # LANG=C, PYTHONUTF8=0 and PYTHONCOERCECLOCALE=0 are REQUIRED (measured:
    # bare LC_ALL=C LANG=C still reports 'utf-8' under PEP 540/538 coercion,
    # so a version missing the last two is silently vacuous), plus a
    # self-check that the child really landed in a non-UTF-8 env.
    #
    # WHAT THIS REPLACED, so it is not reintroduced: the first version of this
    # pin monkeypatched ``locale.getpreferredencoding`` and asserted the bytes
    # decoded as utf-8.  It was VACUOUS — it passed against the unfixed source.
    # Two independent reasons, both measured: (a) that attribute is never
    # consulted here, because ``os.fdopen`` resolves its default text encoding
    # through the C-level locale, not through the Python ``locale`` module
    # (patched to 'ascii', ``os.fdopen(fd, 'w').encoding`` is still 'UTF-8'
    # and 'café' still writes as b'caf\xc3\xa9'); and (b) the ambient test
    # locale is UTF-8 anyway, so the write succeeded either way.  An in-process
    # monkeypatch CANNOT reach this code path.  That is why 2 and 3 are child
    # processes: only a real interpreter env can vary what ``os.fdopen``
    # defaults to.  (``lane_lifecycle``'s twin pin is NOT this shape — that
    # site calls ``locale.getpreferredencoding(False)`` explicitly at the
    # Python level, so a monkeypatch does reach it and no child is warranted.)
    #
    # MUTATION GATE.  Run with BOTH sites reverted at once — ``os.fdopen(fd,
    # 'w')`` here and ``encoding=locale.getpreferredencoding(False)`` in
    # lane_lifecycle — both pins below were observed FAILING, each in its own
    # way, before the fix was accepted:
    #
    #   spy    -> AssertionError: no encoding= kwarg  (assert None is not None)
    #   strict -> child exit 1, EncodingWarning at this module's fdopen line
    #
    # Re-run that gate before trusting any future edit to these three, in the
    # vocabulary test_session_registry.py uses for its bare-shell rows: a guard
    # must be observed FAILING rather than merely observed green.  A pin not
    # observed failing is not a pin — which is how the version these replaced
    # shipped green and pinned nothing.

    def test_passes_an_explicit_utf8_encoding_to_fdopen(self, tmp_path, monkeypatch):
        """Task 3387: the ``os.fdopen`` boundary is handed an explicit utf-8.

        The cheaper of the two pins and the one that localises the defect:
        it fails on EVERY host regardless of locale, because against the
        unfixed source no ``encoding`` kwarg is passed AT ALL — there is no
        spelling for it to accidentally agree with.

        The spy DELEGATES to the real ``os.fdopen`` rather than standing in for
        it, so the tmp+rename still completes and the residue/bytes assertions
        below stay meaningful.
        """
        seen: list[dict] = []
        real_fdopen = os.fdopen

        def _spy(fd, *args, **kwargs):
            # Record only the write-mode open.  ``monkeypatch.setattr`` on
            # ``fleet_files.os`` patches the global ``os`` module, so an unrelated
            # in-process ``os.fdopen`` would otherwise land in ``seen`` and
            # make the count assertion flaky.  If the source is ever changed to
            # pass the mode as a KWARG this filter records nothing and the test
            # fails loudly on the count — a false RED, never a false green.
            if (kwargs.get('mode', args[0] if args else 'r')) == 'w':
                seen.append(dict(kwargs))
            return real_fdopen(fd, *args, **kwargs)

        monkeypatch.setattr(fleet_files.os, 'fdopen', _spy)

        target = tmp_path / 'record.json'
        fleet_files.atomic_write_text(target, _UTF8_PIN_PAYLOAD)

        assert len(seen) == 1, (
            f'expected exactly one write-mode os.fdopen, recorded {seen}'
        )
        encoding = seen[0].get('encoding')
        assert encoding is not None, (
            'os.fdopen was given no encoding= kwarg, so it falls back to the '
            'ambient locale encoding — the task 3387 bug verbatim'
        )
        assert names_the_utf8_codec(encoding), (
            f'os.fdopen was given encoding={encoding!r}, which is not the utf-8 '
            'codec; JSON on disk must be utf-8 (RFC 8259, task 3387)'
        )
        # The delegated write really did complete, so this pin also covers the
        # semantics it could otherwise have silently disabled.
        assert target.read_bytes() == _UTF8_PIN_PAYLOAD.encode('utf-8')
        assert sorted(p.name for p in tmp_path.iterdir()) == ['record.json']

    def test_names_no_default_encoding_in_a_strict_child(self, tmp_path):
        """Task 3387: an interpreter told to reject implicit encodings runs clean.

        The end-to-end pin.  ``PYTHONWARNDEFAULTENCODING=1`` makes every
        implicit-locale-encoding text open emit an ``EncodingWarning`` naming
        the exact file and line, and the child promotes that warning to an
        error AROUND THE WRITE UNDER TEST ONLY.

        The narrow scope is the point.  Promoting it for the whole interpreter
        (``-W error::EncodingWarning``, the first shape of this pin) meant any
        unrelated implicit-encoding open on the ``from orchestrator import
        fleet_files`` path would fail this test with a
        message about task 3387 — a false RED aimed at the wrong line.  The
        import now happens outside the strict block.

        Preferred over a locale-based pin because it cannot be quietly
        neutralised: it keys on the MISSING ``encoding=`` argument itself, so
        neither PEP 540/538 C-locale coercion nor a host's choice of UTF-8
        alias can make it pass vacuously.  The trailing byte assertion keeps
        the on-disk BYTES pinned as well, which is the actual contract (RFC
        8259) — that is why removing the retired C-locale child cost no
        coverage.

        MEASURED both ways, with the filter scoped as it now is.  Against the
        unfixed ``os.fdopen(fd, 'w')``: exit 1, ``EncodingWarning: 'encoding'
        argument not specified``, pointing at the writer's fdopen line.
        Against the fix: exit 0, clean.
        """
        script = tmp_path / 'child_encoding_warning.py'
        script.write_text(
            _ENCODING_WARNING_CHILD % {'payload': json.dumps(_UTF8_PIN_PAYLOAD)},
            encoding='utf-8',
        )
        target = tmp_path / 'record.json'

        result = subprocess.run(
            [sys.executable, str(script), str(target)],
            env=_utf8_child_env(PYTHONWARNDEFAULTENCODING='1'),
            # A neutral cwd, for the reasons ``_run_row`` documents at length:
            # the process's own cwd goes on sys.path, and the suite's cwd would
            # change what the child resolves.
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            'fleet_files opened a text file without naming an encoding, so '
            'it writes JSON in whatever the ambient locale happens to be '
            f'(task 3387).\nstderr:\n{result.stderr}'
        )
        assert target.read_bytes() == _UTF8_PIN_PAYLOAD.encode('utf-8')
