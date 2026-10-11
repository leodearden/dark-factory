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

import json
import os
import subprocess
import sys
from pathlib import Path

import check_lock_charter_drift as drift
import pytest
import shared.reify_checkout as reify_checkout
from shared.locking import EXTENSIONLESS_FILENAMES, FILE_EXTENSIONS, is_file_path

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

    def test_importing_the_module_neither_resolves_reify_nor_spawns_a_process(self):
        """A fresh interpreter whose resolver and process spawners all raise
        must still import the module.

        Module-level resolution is what makes the in-suite guards' constants
        evaluate at IMPORT time and therefore un-steerable by ``monkeypatch``;
        here it would also make a scheduled job's ``--help`` depend on the
        ambient ``REIFY_ROOT``.  Patching the seams themselves, rather than
        steering ``REIFY_ROOT``, catches a module-scope call whether it would
        have succeeded or failed.
        """
        forbid_then_import = (
            'import subprocess\n'
            'import shared.reify_checkout as reify_checkout\n'
            'def _forbidden(*args, **kwargs):\n'
            '    raise SystemExit(f"reached at import time: {args!r}")\n'
            'reify_checkout.resolve_reify_checkout = _forbidden\n'
            'reify_checkout.reify_skip_reason = _forbidden\n'
            'subprocess.Popen = _forbidden\n'
            'subprocess.run = _forbidden\n'
            'import check_lock_charter_drift\n'
        )
        proc = subprocess.run(
            [sys.executable, '-c', forbid_then_import],
            capture_output=True,
            text=True,
            cwd=str(Path(drift.__file__).parent),
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# The CORPUS sweep: corpus -> allowlist, the direction the originating
# 2026-07-28 incident actually failed in.
#
# `tests/infra/run-all-classification.manifest` was a genuinely TRACKED file
# whose extension was simply absent from the allowlist, so `is_file_path`
# classified it as a directory and `update_task` rejected the declaration with
# a LockCharterViolation.  The vector comparison above structurally cannot
# catch that: both repos can agree perfectly on a vector that is missing the
# same extension.  Only sweeping the real tracked corpus finds the 23rd
# omission.
#
# Every case plants a REAL `git init` checkout under tmp_path — no commit is
# needed, `git ls-files` reads the index — and uses `shared.locking.
# is_file_path` ITSELF as the oracle, so an assertion here cannot drift from
# the predicate it is auditing.
# ---------------------------------------------------------------------------


def _plant_repo(root: Path, paths: list[str], *, gitlinks: list[str] | None = None) -> Path:
    """`git init` a checkout at *root* with *paths* tracked, plus any *gitlinks*.

    Gitlinks are planted with ``git update-index --add --cacheinfo 160000,...``
    rather than a real submodule: that is what actually produces the mode column
    the sweep filters on, without needing a second repo on disk or network.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'init', '-q', str(root)], check=True, capture_output=True)
    for rel in paths:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('x\n')
    if paths:
        subprocess.run(
            ['git', '-C', str(root), 'add', '--', *paths], check=True, capture_output=True
        )
    for rel in gitlinks or []:
        subprocess.run(
            [
                'git', '-C', str(root), 'update-index', '--add',
                '--cacheinfo', f'160000,{"0" * 39}1,{rel}',
            ],
            check=True,
            capture_output=True,
        )
    return root


def _track_non_utf8_path(repo: Path) -> None:
    """Track a file in *repo* whose NAME is bytes that are not valid UTF-8.

    ``.py`` is an allowlisted extension, so only a decode problem, never a
    finding, can change the outcome of a sweep over this repo.
    """
    name = b'caf\xe9.py'  # 0xE9 is Latin-1 é; a lone 0xE9 is invalid UTF-8
    with open(os.fsencode(repo) + b'/' + name, 'wb') as fh:
        fh.write(b'x\n')
    # `ls-files -z` suppresses git's quoting, so these raw bytes reach the decoder.
    subprocess.run(['git', '-C', str(repo), 'add', '--', name], check=True, capture_output=True)


class TestCheckCorpus:
    """``check_corpus(reify_repo)`` — every tracked path must classify as a file."""

    def test_clean_corpus_reports_ok_with_no_findings(self, tmp_path):
        """A corpus every one of whose paths the predicate accepts is not drift."""
        paths = ['a/b.py', 'docs/readme.md', 'ops/Dockerfile']
        for rel in paths:
            assert is_file_path(rel), f'oracle disagrees: {rel} must be accepted'
        repo = _plant_repo(tmp_path / 'reify', paths)

        report = drift.check_corpus(repo)

        assert report['findings'] == {}
        assert report['exit_code'] == drift.EXIT_OK
        assert report['status'] == 'ok'

    def test_tracked_path_with_unallowlisted_extension_is_drift(self, tmp_path):
        """The 2026-07-28 incident's exact shape, reproduced structurally.

        The finding must name BOTH the extension (what to add) and a
        representative tracked path (the evidence that a real file is affected)
        — an extension alone reads as a style opinion, while a path alone leaves
        the reader to re-derive which token to allowlist.
        """
        assert not is_file_path('x/y.proto'), (
            "this case needs an extension the allowlist REJECTS; if 'proto' has "
            'since been allowlisted, pick another absent one'
        )
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py', 'x/y.proto'])

        report = drift.check_corpus(repo)

        assert report['exit_code'] == drift.EXIT_DRIFT
        assert report['status'] == 'drift'
        assert report['findings'] == {'proto': 'x/y.proto'}
        assert 'proto' in report['detail']
        assert 'x/y.proto' in report['detail']

    def test_gitlink_is_excluded_from_the_sweep(self, tmp_path):
        """A mode-160000 submodule mount point is not a file in THIS repo.

        Load-bearing rather than tidiness: admitting a mount point would let a
        whole vendored subtree be declared as a lock charter.  The gitlink here
        carries a deliberately unallowlisted extension, so if the mode filter
        were dropped it would surface as a finding and this case would fail —
        the exclusion is proven to be doing real work, not merely asserted.
        """
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'], gitlinks=['vendor/sub.proto'])

        report = drift.check_corpus(repo)

        assert report['findings'] == {}
        assert report['exit_code'] == drift.EXIT_OK

    def test_extensionless_tracked_path_is_not_a_finding(self, tmp_path):
        """No-dot segments have no extension to report, so they cannot be findings.

        ``README`` (no dot, not in EXTENSIONLESS_FILENAMES) IS rejected by the
        predicate, but there is no token to allowlist — reporting it under a key
        of ``''`` would tell a reader to add an empty extension.  Widening
        EXTENSIONLESS_FILENAMES is a separate, name-by-name judgement, so the
        extension sweep stays scoped to dotted segments.
        """
        assert not is_file_path('README')
        repo = _plant_repo(tmp_path / 'reify', ['README', 'a/b.py'])

        assert drift.check_corpus(repo)['findings'] == {}

    def test_non_git_directory_is_unavailable_and_never_ok(self, tmp_path):
        """`git ls-files` failed, so the sweep did not happen — that is not a pass."""
        plain = tmp_path / 'not-a-checkout'
        plain.mkdir()

        report = drift.check_corpus(plain)

        assert report['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['exit_code'] != drift.EXIT_OK
        assert report['status'] == 'unavailable'
        assert report['findings'] == {}

    def test_first_offending_path_is_recorded_per_extension(self, tmp_path):
        """One representative path per extension, not an unbounded list.

        A reify-scale corpus can carry hundreds of files behind a single missing
        extension; the actionable unit is the extension, and one witness is
        enough to prove it is real.
        """
        repo = _plant_repo(tmp_path / 'reify', ['a/one.proto', 'b/two.proto', 'c/three.sql'])

        findings = drift.check_corpus(repo)['findings']

        assert set(findings) == {'proto', 'sql'}
        assert findings['proto'] in ('a/one.proto', 'b/two.proto')
        assert findings['sql'] == 'c/three.sql'


class TestTrackedPaths:
    """``tracked_paths`` — one parse of the corpus, split by git's own mode column."""

    def test_partitions_on_the_mode_column(self, tmp_path):
        """The gitlink set is DERIVED from mode 160000, never from hardcoded names.

        Matching the rationale already recorded on ``_tracked_entries`` in the
        in-suite copies: a name list would silently stop excluding the day reify
        renames or adds a submodule.
        """
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'], gitlinks=['vendor/sub.proto'])

        paths, gitlinks = drift.tracked_paths(repo)

        assert paths == ['a/b.py']
        assert gitlinks == ['vendor/sub.proto']

    def test_raises_guard_unavailable_when_not_a_checkout(self, tmp_path):
        plain = tmp_path / 'plain'
        plain.mkdir()
        with pytest.raises(drift.GuardUnavailable):
            drift.tracked_paths(plain)

    def test_raises_guard_unavailable_when_git_is_not_on_path(self, tmp_path, monkeypatch):
        """No ``git`` is "could not look": an escaping FileNotFoundError would
        exit 1, which a caller reads as DRIFT — a false accusation."""
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))

        with pytest.raises(drift.GuardUnavailable):
            drift.tracked_paths(repo)

    def test_raises_guard_unavailable_when_git_hangs(self, tmp_path, monkeypatch):
        """A ``git`` stuck on a lock or a network filesystem is bounded, not waited on."""
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        fake_bin = tmp_path / 'fake-bin'
        fake_bin.mkdir()
        fake_git = fake_bin / 'git'
        fake_git.write_text('#!/bin/sh\nexec sleep 30\n')
        fake_git.chmod(0o755)
        monkeypatch.setenv('PATH', f'{fake_bin}{os.pathsep}{os.environ["PATH"]}')

        with pytest.raises(drift.GuardUnavailable):
            drift.tracked_paths(repo, timeout=0.5)

    def test_missing_git_makes_the_corpus_sweep_unavailable_not_drift(
        self, tmp_path, monkeypatch,
    ):
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        empty_bin = tmp_path / 'empty-bin'
        empty_bin.mkdir()
        monkeypatch.setenv('PATH', str(empty_bin))

        report = drift.check_corpus(repo)

        assert report['exit_code'] == drift.EXIT_UNAVAILABLE, report

    def test_non_utf8_tracked_path_raises_guard_unavailable(self, tmp_path):
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        _track_non_utf8_path(repo)

        with pytest.raises(drift.GuardUnavailable):
            drift.tracked_paths(repo)

    def test_non_utf8_tracked_path_exits_unavailable_never_drift(self, tmp_path):
        """An escaping UnicodeDecodeError exits 1, which this script's own contract
        reads as DRIFT: could-not-look reported as a finding."""
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        _track_non_utf8_path(repo)
        _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)

        env = {k: v for k, v in os.environ.items() if k != 'REIFY_ROOT'}
        proc = subprocess.run(
            [sys.executable, drift.__file__, '--reify-repo', str(repo)],
            capture_output=True,
            text=True,
            cwd=str(Path(drift.__file__).parent),
            env=env,
            timeout=60,
        )

        assert proc.returncode == drift.EXIT_UNAVAILABLE, proc.stderr
        assert proc.returncode != drift.EXIT_DRIFT
        assert 'Traceback' not in proc.stderr


class TestCheckComposition:
    """``check`` — the worst of its parts, so an unavailable half is never masked."""

    def test_ok_when_both_halves_are_ok(self, tmp_path):
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)

        report = drift.check(repo)

        assert report['exit_code'] == drift.EXIT_OK
        assert report['status'] == 'ok'
        assert set(report['parts']) == {'vectors', 'corpus'}
        assert report['parts']['vectors']['exit_code'] == drift.EXIT_OK
        assert report['parts']['corpus']['exit_code'] == drift.EXIT_OK

    def test_drift_in_either_half_alone_surfaces(self, tmp_path):
        repo = _plant_repo(tmp_path / 'reify', ['x/y.proto'])
        _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)

        report = drift.check(repo)

        assert report['exit_code'] == drift.EXIT_DRIFT
        assert report['parts']['vectors']['exit_code'] == drift.EXIT_OK
        assert report['parts']['corpus']['exit_code'] == drift.EXIT_DRIFT

    def test_unavailable_half_is_never_masked_by_an_ok_half(self, tmp_path):
        """The composition-level restatement of 'absence is never a pass'.

        The corpus half sweeps a real checkout while the vector half needs the
        guard script; a reify checkout can easily have one and not the other.
        Reporting OK because the half that COULD run agreed would be the same
        silent green as a pytest skip, one level up.
        """
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])  # no guard script planted

        report = drift.check(repo)

        assert report['parts']['corpus']['exit_code'] == drift.EXIT_OK
        assert report['parts']['vectors']['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['status'] == 'unavailable'

    def test_unavailable_outranks_drift(self, tmp_path):
        """Ordering is UNAVAILABLE > DRIFT > OK, not last-writer-wins.

        A drifting half plus an unavailable half is not a complete drift report:
        the unexamined half may hold worse. The exit code must say so.
        """
        repo = _plant_repo(tmp_path / 'reify', ['x/y.proto'])  # corpus drifts
        _plant_guard(repo, extensions=DF_EXTENSIONS, rc=3)  # vectors unavailable

        report = drift.check(repo)

        assert report['parts']['corpus']['exit_code'] == drift.EXIT_DRIFT
        assert report['parts']['vectors']['exit_code'] == drift.EXIT_UNAVAILABLE
        assert report['exit_code'] == drift.EXIT_UNAVAILABLE


# ---------------------------------------------------------------------------
# RESOLUTION and the CLI.
#
# DIVISION OF OWNERSHIP, stated so this file does not grow a duplicate of a
# matrix another suite already owns.  `shared/tests/test_reify_checkout.py`
# owns the RESOLVER's semantics — the bare-vs-worktree-vs-extra-nesting layout
# matrix, nearest-ancestor precedence, marker-less-dir shadowing, and the whole
# REIFY_ROOT override contract — in one place, 40-plus cases.  Restating it here
# would mean two suites to edit for one semantic change, which is the exact
# duplication the promotion of that module to shared/ existed to remove (see
# `TestReifyCheckoutAdapter`'s note in fused-memory's copy).
#
# What THIS file pins is only what is specific to THIS call site: that it
# DELEGATES rather than keeping a private copy of the walk, that it binds the
# right marker and starts at its own tree, that an explicit flag out-ranks the
# env var, that the answer does not vary with the caller's cwd, that no
# `parents[N]` index arithmetic has crept back in, and that provenance is never
# re-derived from an ambient REIFY_ROOT.
# ---------------------------------------------------------------------------


class TestResolve:
    """``resolve(explicit=None)`` — precedence, delegation, layout independence."""

    def test_delegates_to_the_shared_single_source(self, tmp_path, monkeypatch):
        """The ancestor walk must live in ONE place — shared.reify_checkout.

        Deliberately white-box, and deliberately the only genuinely-RED
        assertion available for this property: a black-box equivalence check
        cannot express it, because a faithful local COPY of the walk would
        satisfy every one of them while defeating the single-sourcing.  What
        this pin buys is that a second copy cannot quietly reappear here — which
        is precisely what the stale WIP this task discarded had done.
        """
        sentinel = tmp_path / 'sentinel-reify'
        seen = {}

        def _stub(marker, *, start):
            seen['marker'] = Path(marker)
            seen['start'] = Path(start)
            return reify_checkout.ReifyCheckout(sentinel, False)

        monkeypatch.setattr(reify_checkout, 'resolve_reify_checkout', _stub)

        assert drift.resolve().root == sentinel
        assert seen['marker'] == Path('scripts') / 'lock-charter-guard.sh'
        assert seen['start'] == Path(drift.__file__), (
            'the walk must START at this script, not at shared/src/shared/ — a '
            "shared-relative start would walk the WRONG tree's ancestors"
        )

    def test_explicit_repo_beats_reify_root(self, tmp_path, monkeypatch):
        """``--reify-repo`` is the operator saying "this one, now"."""
        monkeypatch.setenv('REIFY_ROOT', str(tmp_path / 'from-env'))
        explicit = tmp_path / 'from-flag'

        checkout = drift.resolve(str(explicit))

        assert checkout.root == explicit.resolve()
        assert checkout.named_by_env is False, (
            'a flag-named root did not come from the env var; claiming it did '
            'would blame the operator for a path they did not set'
        )

    def test_explicit_repo_is_expanded_and_resolved(self, monkeypatch, tmp_path):
        monkeypatch.setenv('HOME', str(tmp_path))
        assert drift.resolve('~/somewhere').root == (tmp_path / 'somewhere').resolve()

    def test_reify_root_beats_discovery_and_is_marked_as_such(self, tmp_path, monkeypatch):
        """Provenance travels WITH the root, so a later message cannot misattribute it."""
        monkeypatch.setenv('REIFY_ROOT', str(tmp_path / 'from-env'))

        checkout = drift.resolve()

        assert checkout.root == (tmp_path / 'from-env').resolve()
        assert checkout.named_by_env is True

    @pytest.mark.parametrize(
        'layout',
        ['dark-factory/scripts', 'dark-factory/.worktrees/3226/scripts'],
        ids=['bare-checkout', 'worktree'],
    )
    def test_resolution_does_not_vary_with_the_callers_cwd(self, tmp_path, monkeypatch, layout):
        """A bare-checkout cwd and a ``.worktrees/<n>`` cwd must resolve identically.

        The two shapes differ by exactly the two segments that made a fixed
        ``parents[N]`` index wrong in one of them.  Resolution keys off the
        SCRIPT's own location, never the caller's, so an operator running this
        from inside a worktree and a cron running it from the main checkout get
        the same answer about the same sibling.
        """
        cwd = tmp_path / layout
        cwd.mkdir(parents=True)
        seen = []

        def _stub(marker, *, start):
            seen.append(Path(start))
            return reify_checkout.ReifyCheckout(tmp_path / 'sentinel-reify', False)

        monkeypatch.setattr(reify_checkout, 'resolve_reify_checkout', _stub)
        monkeypatch.chdir(cwd)

        assert drift.resolve().root == tmp_path / 'sentinel-reify'
        assert seen == [Path(drift.__file__)], (
            f'start varied with cwd {cwd} — resolution must key off the script, '
            f'not the caller'
        )

    def test_ambient_reify_root_neither_replaces_nor_reattributes_a_discovered_root(
        self, tmp_path, monkeypatch,
    ):
        """A root the resolver DISCOVERED stays discovered, whatever REIFY_ROOT holds now.

        Re-reading the env var here would credit the operator with a path they
        never set; see ``resolve``'s docstring.
        """
        discovered = tmp_path / 'discovered-reify'
        monkeypatch.setattr(
            reify_checkout,
            'resolve_reify_checkout',
            lambda marker, *, start: reify_checkout.ReifyCheckout(discovered, False),
        )
        monkeypatch.setenv('REIFY_ROOT', str(tmp_path / 'operator-set-elsewhere'))

        assert drift.resolve() == reify_checkout.ReifyCheckout(discovered, False)


class TestMainCli:
    """``main(argv)`` — exit code, stream routing, and single-sourced wording."""

    def test_returns_the_reports_exit_code_on_ok(self, tmp_path, capsys):
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)

        rc = drift.main(['--reify-repo', str(repo)])

        assert rc == drift.EXIT_OK
        captured = capsys.readouterr()
        assert captured.out.strip(), 'an OK run must say so on stdout'
        assert captured.err == '', 'an OK run must not write to stderr'

    def test_drift_goes_to_stderr_not_stdout(self, tmp_path, capsys):
        """A drift report must survive ``>/dev/null`` in whatever wraps this.

        A scheduled job that discards stdout and mails stderr is the ordinary
        shape; a drift report on stdout would be silently thrown away by it.
        """
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        _plant_guard(
            repo,
            extensions=sorted([*DF_EXTENSIONS, 'protobuf']),
            extensionless=DF_EXTENSIONLESS,
        )

        rc = drift.main(['--reify-repo', str(repo)])

        assert rc == drift.EXIT_DRIFT
        captured = capsys.readouterr()
        assert 'protobuf' in captured.err
        assert captured.out == ''

    def test_unavailable_goes_to_stderr(self, tmp_path, capsys):
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])  # no guard script

        rc = drift.main(['--reify-repo', str(repo)])

        assert rc == drift.EXIT_UNAVAILABLE
        captured = capsys.readouterr()
        assert 'lock-charter-guard.sh' in captured.err
        assert captured.out == ''

    def test_json_emits_one_parseable_object_on_stdout(self, tmp_path, capsys):
        """``--json`` is the MACHINE channel and is on stdout for every status.

        A consumer that piped ``--json`` and got nothing on a drift run — the
        run it most needs to parse — would have to fall back to scraping the
        human text.
        """
        repo = _plant_repo(tmp_path / 'reify', ['x/y.proto'])
        _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)

        rc = drift.main(['--reify-repo', str(repo), '--json'])

        payload = json.loads(capsys.readouterr().out)
        assert payload['status'] == 'drift'
        assert payload['exit_code'] == rc == drift.EXIT_DRIFT

    def test_json_is_emitted_for_an_unavailable_run_too(self, tmp_path, capsys):
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])

        rc = drift.main(['--reify-repo', str(repo), '--json'])

        payload = json.loads(capsys.readouterr().out)
        assert payload['status'] == 'unavailable'
        assert payload['exit_code'] == rc == drift.EXIT_UNAVAILABLE

    def test_discovery_miss_message_is_single_sourced(self, monkeypatch, capsys):
        """The miss wording must be EXACTLY what the in-suite gates say.

        Two call sites describing the same condition in two hand-rolled strings
        is how "reify is not checked out" and "your REIFY_ROOT is wrong" get
        conflated — the distinction `reify_skip_reason` exists to keep.
        """
        monkeypatch.setattr(
            reify_checkout,
            'resolve_reify_checkout',
            lambda marker, *, start: reify_checkout.ReifyCheckout(None, False),
        )

        rc = drift.main([])

        assert rc == drift.EXIT_UNAVAILABLE
        expected = reify_checkout.reify_skip_reason(
            Path('scripts') / 'lock-charter-guard.sh', None, named_by_env=False
        )
        assert expected is not None
        assert expected in capsys.readouterr().err

    def test_bad_reify_root_names_the_path_the_operator_named(self, tmp_path, capsys, monkeypatch):
        """A typo'd override must be self-evident, not silently fall back to discovery."""
        bogus = tmp_path / 'typo'
        bogus.mkdir()
        monkeypatch.setenv('REIFY_ROOT', str(bogus))

        rc = drift.main([])

        assert rc == drift.EXIT_UNAVAILABLE
        err = capsys.readouterr().err
        assert str(bogus) in err
        assert 'REIFY_ROOT' in err

    def test_message_about_a_discovered_root_never_blames_reify_root(
        self, tmp_path, monkeypatch, capsys,
    ):
        """A DISCOVERED root that cannot serve must not be reported as the operator's typo.

        Blaming REIFY_ROOT here would send the operator hunting for a path they
        never set; see ``resolve``'s docstring.
        """
        discovered = tmp_path / 'discovered-reify'
        discovered.mkdir()  # a checkout WITHOUT the guard script, so main() explains why
        elsewhere = tmp_path / 'operator-set-elsewhere'
        monkeypatch.setattr(
            reify_checkout,
            'resolve_reify_checkout',
            lambda marker, *, start: reify_checkout.ReifyCheckout(discovered, False),
        )
        monkeypatch.setenv('REIFY_ROOT', str(elsewhere))

        rc = drift.main([])

        assert rc == drift.EXIT_UNAVAILABLE
        assert rc != drift.EXIT_OK
        err = capsys.readouterr().err
        expected = reify_checkout.reify_skip_reason(
            drift.GUARD_RELPATH, discovered, named_by_env=False
        )
        assert expected is not None
        assert expected in err
        assert 'REIFY_ROOT' not in err
        assert str(elsewhere) not in err

    def test_module_is_executable(self):
        """A script an operator is told to run must be runnable."""
        assert os.access(drift.__file__, os.X_OK), (
            f'{drift.__file__} is not executable — chmod +x it'
        )

    @pytest.mark.parametrize(
        ('plant_guard', 'expected'),
        [(True, drift.EXIT_OK), (False, drift.EXIT_UNAVAILABLE)],
        ids=['ok', 'unavailable'],
    )
    def test_process_exit_status_carries_the_outcome(self, tmp_path, plant_guard, expected):
        """Run the script as a PROCESS and read the status a shell would read.

        Everything above calls ``main()`` in-process, which cannot observe the
        ``sys.exit(main(...))`` wiring at the bottom of the module.  A scheduled
        job or CI lane branches on the numeric exit STATUS, so the could-not-look
        arm is asserted ``!= 0`` as well as ``== EXIT_UNAVAILABLE``: the failure
        being guarded against is a future refactor that swallows the return value
        (``main(...)`` called bare, or wrapped in a ``try``) and lets an
        unavailable run exit 0 — a skip reported as a pass, which is exactly the
        conflation this script exists to end.
        """
        repo = _plant_repo(tmp_path / 'reify', ['a/b.py'])
        if plant_guard:
            _plant_guard(repo, extensions=DF_EXTENSIONS, extensionless=DF_EXTENSIONLESS)

        env = {k: v for k, v in os.environ.items() if k != 'REIFY_ROOT'}
        proc = subprocess.run(
            [sys.executable, drift.__file__, '--reify-repo', str(repo)],
            capture_output=True,
            text=True,
            cwd=str(Path(drift.__file__).parent),
            env=env,
            timeout=60,
        )

        assert proc.returncode == expected, proc.stderr
        if expected != drift.EXIT_OK:
            assert proc.returncode != drift.EXIT_OK
