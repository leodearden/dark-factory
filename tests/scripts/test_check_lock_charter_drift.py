"""Contract for ``scripts/check_lock_charter_drift.py`` — the OUT-OF-SUITE cross-repo checker.

WHY THIS SCRIPT EXISTS AT ALL, restated here because it is what every
assertion below is shaped by.  On 2026-07-29 reify widened its lock-charter
allowlist 36 -> 58 entries.  dark-factory's in-suite Tier-2 drift guard shells
out to reify's ``scripts/lock-charter-guard.sh`` LIVE, so every dark-factory
worktree's full suite went red at once, and task 3118's ``post_merge_verify``
failed on a 7-file orchestrator-only diff that could not possibly have caused
it.  A sibling repo's mutable working state was sitting on this repo's
merge-verify critical path.  The remedy this task lands is a checker that is
EXPLICITLY INVOKED — by an operator or a scheduled job — rather than collected
by pytest, so a reify commit reports drift where drift belongs instead of
blocking an unrelated dark-factory merge.

THE OUTCOME CONTRACT IS THE WHOLE POINT, and it is why there is no skip state.
The in-suite guards degrade to a pytest SKIP when reify is not checked out, and
a skip is indistinguishable from a pass to anything reading an exit code — the
precise conflation this task's other half fixes inside verify.  So this script
has exactly three outcomes, ``EXIT_OK`` / ``EXIT_DRIFT`` / ``EXIT_UNAVAILABLE``,
and "I could not look" is reported as loudly as "I looked and found drift".
The assertions below therefore pin ``!= EXIT_OK`` explicitly on every
could-not-look path, not merely ``== EXIT_UNAVAILABLE``: the failure being
guarded against is a future refactor quietly folding unavailable into success,
and an equality assertion against a renamed-but-still-zero constant would not
catch it.

HERMETICITY IS A HARD RULE HERE (task 3226 pre-3), not a style preference.
reify's live vector may legitimately differ from dark-factory's at any instant —
that divergence is precisely what this script exists to REPORT, and mirroring it
is a different task's job.  A test that executed the real ``/home/leo/src/reify``
checkout and asserted the two real vectors are equal would be RED from the
moment it was written and un-GREENable here, and would re-import into this
suite the exact disease the script treats.  Every case below therefore drives a
FAKE ``scripts/lock-charter-guard.sh`` planted under pytest's ``tmp_path``, and
the autouse fixture scrubs ``REIFY_ROOT`` so the ambient machine cannot steer a
result either.

The module under test is imported as a bare ``import check_lock_charter_drift``:
``tests/scripts/conftest.py`` inserts ``scripts/`` onto ``sys.path`` (needed
because ``--import-mode=importlib`` deliberately does not).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import check_lock_charter_drift as drift
import pytest
from shared.locking import EXTENSIONLESS_FILENAMES, FILE_EXTENSIONS

# The two vectors dark-factory declares, in the exact form the guard script
# emits them (one per line, sorted).  Read from `shared.locking` — never from
# `fused_memory.middleware.lock_charter_guard`, which is not importable under
# this directory's `uv run --project shared` environment.  shared's copy is
# independently pinned to the same canonical vector by
# shared/tests/test_locking.py::TestFileExtensionsDriftGuard, so the script and
# the in-suite Tier-1 guard cannot disagree about what dark-factory declares.
DF_EXTENSIONS = sorted(FILE_EXTENSIONS)
DF_EXTENSIONLESS = sorted(EXTENSIONLESS_FILENAMES)


@pytest.fixture(autouse=True)
def _no_ambient_reify_root(monkeypatch):
    """Scrub REIFY_ROOT for every case in this module.

    An operator (or an orchestrator role) may have REIFY_ROOT exported.  Any
    case that reached the real checkout through it would assert about a repo
    this suite does not control, and would flip red on a reify commit — the
    cross-repo coupling this whole file exists to remove.
    """
    monkeypatch.delenv('REIFY_ROOT', raising=False)


def _plant_guard(
    repo: Path,
    *,
    extensions: list[str] | None = None,
    extensionless: list[str] | None = None,
    rc: int = 0,
    sleep_secs: float | None = None,
) -> Path:
    """Write a FAKE reify ``scripts/lock-charter-guard.sh`` under *repo* and return it.

    Mimics the real script's flag surface — ``--list-extensions`` and
    ``--list-extensionless`` each print their vector one entry per line, and an
    unrecognised flag prints usage to stderr and exits 2 (measured 2026-08-09 on
    the live script, and recorded on ``_reify_guard_vector`` in the two in-suite
    copies).  *rc* and *sleep_secs* exist to drive the BROKEN-emitter arms: a
    script that is present and then fails, and one that hangs.
    """
    scripts_dir = repo / 'scripts'
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / 'lock-charter-guard.sh'

    def _emit(values: list[str] | None) -> str:
        if not values:
            # `printf '%s\n'` with no operands still prints one empty line, so
            # an empty vector must emit nothing at all rather than a blank.
            return ':'
        joined = ' '.join(f"'{v}'" for v in values)
        return f"printf '%s\\n' {joined}"

    body = ['#!/usr/bin/env bash']
    if sleep_secs is not None:
        body.append(f'sleep {sleep_secs}')
    if rc != 0:
        body.append(f'echo "boom" >&2; exit {rc}')
    body += [
        'case "$1" in',
        f'  --list-extensions) {_emit(extensions)} ;;',
        f'  --list-extensionless) {_emit(extensionless)} ;;',
        '  *) echo "usage: lock-charter-guard.sh --list-extensions" >&2; exit 2 ;;',
        'esac',
    ]
    script.write_text('\n'.join(body) + '\n')
    script.chmod(0o755)
    return script


def _agreeing_repo(tmp_path: Path) -> Path:
    """A planted reify checkout whose guard emits EXACTLY dark-factory's vectors."""
    repo = tmp_path / 'reify'
    _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)
    return repo


class TestExitCodeContract:
    """The three outcomes are distinct ints, and 'could not look' is not zero."""

    def test_exit_codes_are_zero_one_two_and_pairwise_distinct(self):
        """A caller's ``rc != 0`` and ``rc == 1`` tests must never collide.

        Pinned as literals rather than merely "distinct": a scheduled job wiring
        this script into a cron or a CI lane will branch on the numbers, and
        renumbering them is a breaking change that must fail HERE rather than
        silently re-classify an unavailable run as a drift report (or worse, as
        a pass).
        """
        assert drift.EXIT_OK == 0
        assert drift.EXIT_DRIFT == 1
        assert drift.EXIT_UNAVAILABLE == 2
        codes = [drift.EXIT_OK, drift.EXIT_DRIFT, drift.EXIT_UNAVAILABLE]
        assert all(isinstance(c, int) for c in codes)
        assert len(set(codes)) == 3

    def test_there_is_no_skip_state(self):
        """No fourth outcome may be added that reads as 'we did not check, but fine'.

        The whole remedy is that absence is loud.  A module-level ``EXIT_SKIP``
        — however spelled — would reintroduce the pytest-skip semantics this
        script exists to escape.
        """
        skipish = [n for n in dir(drift) if n.startswith('EXIT_')]
        assert sorted(skipish) == ['EXIT_DRIFT', 'EXIT_OK', 'EXIT_UNAVAILABLE']


class TestCheckVectors:
    """``check_vectors(reify_repo)`` — the α/γ allowlist comparison."""

    def test_identical_vectors_report_ok(self, tmp_path):
        report = drift.check_vectors(_agreeing_repo(tmp_path))
        assert report['exit_code'] == drift.EXIT_OK
        assert report['status'] == 'ok'

    def test_one_extra_and_one_missing_extension_is_drift_in_both_directions(self, tmp_path):
        """The canonical drift shape: reify has one we lack, and lacks one we have.

        Both directions are asserted from ONE fixture because a checker that
        computed only ``reify - dark_factory`` would still look correct on a
        widening (the 2026-07-29 incident's direction) while being blind to a
        NARROWING — reify dropping an extension dark-factory still declares,
        which is the direction that makes dark-factory's own declarations
        unenforceable on the α side.
        """
        repo = tmp_path / 'reify'
        reify_exts = [e for e in DF_EXTENSIONS if e != 'csv'] + ['protobuf']
        _plant_guard(repo, extensions=sorted(reify_exts), extensionless=DF_EXTENSIONLESS)

        report = drift.check_vectors(repo)

        assert report['exit_code'] == drift.EXIT_DRIFT
        assert report['status'] == 'drift'
        ext = report['vectors']['extensions']
        assert ext['only_in_reify'] == ['protobuf']
        assert ext['only_in_dark_factory'] == ['csv']

    def test_reify_only_extension_does_not_manufacture_a_dark_factory_only_entry(self, tmp_path):
        """One-direction drift must populate exactly one list and leave the other empty."""
        repo = tmp_path / 'reify'
        _plant_guard(
            repo,
            extensions=sorted([*DF_EXTENSIONS, 'protobuf']),
            extensionless=DF_EXTENSIONLESS,
        )

        report = drift.check_vectors(repo)

        assert report['exit_code'] == drift.EXIT_DRIFT
        ext = report['vectors']['extensions']
        assert ext['only_in_reify'] == ['protobuf']
        assert ext['only_in_dark_factory'] == []

    def test_dark_factory_only_extension_does_not_manufacture_a_reify_only_entry(self, tmp_path):
        repo = tmp_path / 'reify'
        _plant_guard(
            repo,
            extensions=[e for e in DF_EXTENSIONS if e != 'csv'],
            extensionless=DF_EXTENSIONLESS,
        )

        report = drift.check_vectors(repo)

        assert report['exit_code'] == drift.EXIT_DRIFT
        ext = report['vectors']['extensions']
        assert ext['only_in_reify'] == []
        assert ext['only_in_dark_factory'] == ['csv']

    def test_extensionless_drift_alone_is_detected_when_extensions_agree(self, tmp_path):
        """The two vectors are structurally different and one must not mask the other.

        ``FILE_EXTENSIONS`` and ``EXTENSIONLESS_FILENAMES`` are separate inputs
        to ``is_file_path`` — the second exists precisely because those names
        would be rejected by the extension test.  A checker that compared only
        the first would report OK while ``Dockerfile`` silently became
        undeclarable on one side.
        """
        repo = tmp_path / 'reify'
        _plant_guard(
            repo,
            extensions=DF_EXTENSIONS,
            extensionless=sorted([*DF_EXTENSIONLESS, 'Makefile']),
        )

        report = drift.check_vectors(repo)

        assert report['exit_code'] == drift.EXIT_DRIFT
        assert report['status'] == 'drift'
        assert report['vectors']['extensions']['only_in_reify'] == []
        assert report['vectors']['extensions']['only_in_dark_factory'] == []
        assert report['vectors']['extensionless']['only_in_reify'] == ['Makefile']
        assert report['vectors']['extensionless']['only_in_dark_factory'] == []

    def test_drift_detail_disowns_the_dark_factory_branch_under_test(self, tmp_path):
        """The message must say WHOSE fault this is, in terms.

        The 2026-07-29 incident's real cost was misattribution: an
        orchestrator-only diff was blamed for a sibling repo's commit.  The
        detail string is where a reader of a scheduled run learns that a drift
        report is a cross-repo MIRROR gap owned by a mirror task, and not a
        defect in any dark-factory branch.
        """
        repo = tmp_path / 'reify'
        _plant_guard(
            repo,
            extensions=sorted([*DF_EXTENSIONS, 'protobuf']),
            extensionless=DF_EXTENSIONLESS,
        )

        detail = drift.check_vectors(repo)['detail'].lower()

        assert 'mirror' in detail
        assert 'not' in detail and 'dark-factory' in detail

    def test_absent_guard_script_is_unavailable_and_never_ok(self, tmp_path):
        """A reify directory carrying no guard script did not answer the question.

        This is the arm the in-suite guards turn into a pytest SKIP.  Here it
        must be an outcome a caller can see, because a scheduled job whose
        sibling checkout has gone missing is exactly the case where a silent
        green is most dangerous: it would read as "the mirrors agree" forever.
        """
        repo = tmp_path / 'reify'
        repo.mkdir()

        report = drift.check_vectors(repo)

        assert report['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['exit_code'] != drift.EXIT_OK
        assert report['status'] == 'unavailable'

    def test_nonexistent_repo_dir_is_unavailable(self, tmp_path):
        report = drift.check_vectors(tmp_path / 'nope')
        assert report['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['status'] == 'unavailable'

    def test_present_but_failing_guard_is_unavailable_not_ok_and_not_drift(self, tmp_path):
        """A broken emitter is not agreement and it is not disagreement.

        Mirrors ``_reify_guard_vector``'s loud-on-broken rationale: an
        unrecognised flag on the live script exits 2, so a checker that
        swallowed a non-zero rc into "no output, therefore both vectors empty,
        therefore drift" — or worse, into OK — would report a confident answer
        it never obtained.
        """
        repo = tmp_path / 'reify'
        _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS, rc=3)

        report = drift.check_vectors(repo)

        assert report['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['exit_code'] not in (drift.EXIT_OK, drift.EXIT_DRIFT)
        assert report['status'] == 'unavailable'

    def test_hanging_guard_is_bounded_and_reports_unavailable(self, tmp_path):
        """The subprocess call is time-bounded, and a timeout is unavailable.

        Without a bound, a hung sibling script would hang the scheduled job
        indefinitely — the same never-answers failure as a missing script, but
        with no output at all to diagnose it from.
        """
        repo = tmp_path / 'reify'
        _plant_guard(
            repo,
            extensions=DF_EXTENSIONS,
            extensionless=DF_EXTENSIONLESS,
            sleep_secs=30,
        )

        report = drift.check_vectors(repo, timeout=0.5)

        assert report['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['exit_code'] not in (drift.EXIT_OK, drift.EXIT_DRIFT)
        assert report['status'] == 'unavailable'

    def test_unavailable_detail_names_the_script_it_could_not_run(self, tmp_path):
        repo = tmp_path / 'reify'
        repo.mkdir()
        assert 'lock-charter-guard.sh' in drift.check_vectors(repo)['detail']


class TestGuardVectorHelper:
    """``_guard_vector`` — the bounded, loud-on-broken subprocess wrapper."""

    def test_raises_guard_unavailable_on_nonzero_rc(self, tmp_path):
        repo = tmp_path / 'reify'
        script = _plant_guard(repo, extensions=DF_EXTENSIONS, rc=3)
        with pytest.raises(drift.GuardUnavailable):
            drift._guard_vector(script, '--list-extensions')

    def test_raises_guard_unavailable_on_timeout(self, tmp_path):
        repo = tmp_path / 'reify'
        script = _plant_guard(repo, extensions=DF_EXTENSIONS, sleep_secs=30)
        with pytest.raises(drift.GuardUnavailable):
            drift._guard_vector(script, '--list-extensions', timeout=0.5)

    def test_raises_guard_unavailable_on_oserror(self, tmp_path):
        """An OSError (no bash, permissions, ENOENT) is unavailable, never a crash.

        The script's whole value is that its three outcomes are total: an
        uncaught OSError would exit with Python's own code 1, which a caller
        would read as DRIFT — a false accusation against reify.
        """
        with pytest.raises(drift.GuardUnavailable):
            drift._guard_vector(tmp_path / 'absent' / 'guard.sh', '--list-extensions')

    def test_returns_the_sorted_stripped_vector(self, tmp_path):
        repo = tmp_path / 'reify'
        script = _plant_guard(repo, extensions=['zz', 'aa', 'mm'])
        assert drift._guard_vector(script, '--list-extensions') == ['aa', 'mm', 'zz']

    def test_guard_unavailable_is_not_a_bare_exception(self):
        """Callers must be able to catch this WITHOUT swallowing real bugs."""
        assert issubclass(drift.GuardUnavailable, Exception)
        assert drift.GuardUnavailable is not Exception
        assert not issubclass(drift.GuardUnavailable, (KeyboardInterrupt, SystemExit))


class TestImportIsHermetic:
    """Importing the module must not touch a real checkout (task 3226 pre-3)."""

    def test_module_import_resolves_no_checkout_at_module_scope(self, tmp_path, monkeypatch):
        """A fresh interpreter importing the module must not shell out or resolve reify.

        Module-level resolution is what makes the in-suite guards' constants
        evaluate at IMPORT time and therefore un-steerable by ``monkeypatch``.
        Here it would also mean a scheduled job's ``--help`` silently depended on
        the ambient ``REIFY_ROOT``.
        """
        src = Path(drift.__file__).read_text()
        # A module-scope call to either resolver would appear at column 0.
        offenders = [
            line
            for line in src.splitlines()
            if line and not line[0].isspace() and 'resolve_reify_checkout(' in line
        ]
        assert offenders == [], f'module-scope reify resolution: {offenders!r}'

        proc = subprocess.run(
            [sys.executable, '-c', 'import check_lock_charter_drift'],
            capture_output=True,
            text=True,
            cwd=str(Path(drift.__file__).parent),
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
