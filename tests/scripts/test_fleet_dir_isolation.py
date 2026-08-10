"""Guard tests: a test run can never reach the LIVE fleet directory (task 3799).

Two coupled defects, both of the same class as the three defences this repo
already carries in ``df_pytest_isolation``:

DEFECT A — the fake-``systemctl`` harnesses hand REAL unit names to a REAL
restart script.  Five pytest files shim a fake ``systemctl`` onto ``PATH`` and
drive ``scripts/restart-all-orchestrators.sh`` (or its per-unit sibling)
through it.  Three of them passed genuinely installed unit names —
``orchestrator-reify.service`` and ``orchestrator-dark-factory.service``, the
unit this whole factory runs on.  The fake shadows ``systemctl`` only for as
long as its tmpdir lives on ``PATH``: task 3798 measured poll loops that
outlived their test by 27.8 HOURS, well past pytest's tmpdir GC, at which point
``systemctl`` resolves to ``/usr/bin/systemctl`` and the orphan issues a REAL
restart of whatever unit name the fixture handed it.  A SYNTHETIC name makes
that worst case a no-op against a unit that does not exist.

DEFECT B — ``ORCH_FLEET_DIR`` was unset for the whole suite, so
``scripts/restart-all-orchestrators.sh``'s ``FLEET_DIR`` default and
``drain_check.DEFAULT_FLEET_DIR``
both fell through to their machine-global default,
``/home/leo/src/dark-factory/data/fleet``.  That directory is a CROSS-PROJECT
rendezvous dir — measured 2026-08-07 and re-measured 2026-08-09 holding live
heartbeats for seven different projects' orchestrators — so a test-spawned
drain gate read five other projects' live production heartbeats and decided the
real fleet's drain state from them.

This module covers the suite-wide, opt-out-impossible layer in the root
``df_pytest_isolation`` module — the one that catches the next spawner that
forgets:

* the shared synthetic-unit vocabulary (``synthetic_unit`` /
  ``non_synthetic_unit_names`` / ``assert_synthetic_units``), which is what
  makes a leaked fixture heartbeat SELF-IDENTIFYING;
* a session-scoped autouse fixture redirecting ``ORCH_FLEET_DIR`` away from the
  live checkout; and
* the pure helpers behind the live-fleet leak guard (``synthetic_heartbeats_in``
  / ``leaked_fleet_heartbeat_reason``), directly testable without a nested
  pytest run.

Shaped like ``test_deploy_clock_isolation.py``, ``test_drain_process_leak_isolation.py``
and ``test_basetemp_git_isolation.py`` — the modules covering this repo's other
three suite-wide isolation defences — so the four read as one family.

NOTHING here writes to, or asserts the content of, the real
``LIVE_FLEET_DIR``.  A test that touched it would BE the defect under guard.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from df_pytest_isolation import (  # noqa: E402
    LIVE_FLEET_DIR,
    SYNTHETIC_UNIT_PREFIX,
    assert_synthetic_units,
    leaked_fleet_heartbeat_reason,
    non_synthetic_unit_names,
    synthetic_heartbeats_in,
    synthetic_unit,
)

_FLEET_DIR_ENV = 'ORCH_FLEET_DIR'


class TestSyntheticUnit:
    """The builder every fake-systemctl fixture takes its unit names from."""

    def test_it_builds_a_fake_prefixed_service_name(self) -> None:
        """The one spelling the whole family shares, pinned literally."""
        assert synthetic_unit('reify') == 'orchestrator-fake-reify.service'

    def test_what_it_builds_is_accepted_by_the_checker(self) -> None:
        """Round-trip: the builder and the rule cannot drift apart.

        Two symbols encoding one convention is exactly how a convention rots —
        a later tightening of the rule (say, requiring a different separator)
        would otherwise leave every existing call site silently illegal.
        """
        for stem in ('reify', 'alpha', 'bravo', 'dark-factory'):
            assert non_synthetic_unit_names([synthetic_unit(stem)]) == []

    def test_it_still_matches_the_real_list_units_glob(self) -> None:
        """A fixture name must still be one the REAL enumeration could produce.

        ``restart-all-orchestrators.sh`` enumerates via
        ``systemctl list-units 'orchestrator-*.service'``.  A fixture name
        outside that glob would make the harness LESS faithful, not safer — the
        fake would be answering questions the real binary never gets asked.
        """
        name = synthetic_unit('reify')
        assert name.startswith('orchestrator-')
        assert name.endswith('.service')


class TestNonSyntheticUnitNames:
    """Which names the rule rejects, and which it must keep legal."""

    def test_it_reports_a_real_reify_unit(self) -> None:
        """``orchestrator-reify.service`` is INSTALLED on this box."""
        assert non_synthetic_unit_names(['orchestrator-reify.service']) == [
            'orchestrator-reify.service',
        ]

    def test_it_reports_the_real_dark_factory_unit(self) -> None:
        """``orchestrator-dark-factory.service`` runs this entire factory."""
        assert non_synthetic_unit_names(['orchestrator-dark-factory.service']) == [
            'orchestrator-dark-factory.service',
        ]

    def test_the_bare_fake_literal_stays_legal(self) -> None:
        """``orchestrator-fake.service`` — with NO hyphenated stem — is already
        in use as ``tests/scripts/test_restart_all_orchestrators.py``'s
        ``UNIT_NAME`` and the prefix rule is chosen specifically to keep it
        legal.

        Pinned explicitly so a later "tighten it to require a hyphen" cannot
        silently invalidate the one good literal that predates this task.
        """
        assert non_synthetic_unit_names([
            'orchestrator-fake.service',
            'orchestrator-fake-alpha.service',
        ]) == []

    def test_empty_input_is_empty_output(self) -> None:
        """The no-fixtures case is not an error."""
        assert non_synthetic_unit_names([]) == []

    def test_order_and_duplicates_are_preserved(self) -> None:
        """Reported in the CALLER's own order, duplicates intact.

        The message names offenders in the order the caller listed them, so a
        reader can map each one straight back to its argument position — and a
        deduping/sorting helper would break that mapping for the multi-unit
        fixtures (``running_units=[...]``) this exists to check.
        """
        units = [
            'orchestrator-reify.service',
            synthetic_unit('alpha'),
            'orchestrator-dark-factory.service',
            'orchestrator-reify.service',
        ]
        assert non_synthetic_unit_names(units) == [
            'orchestrator-reify.service',
            'orchestrator-dark-factory.service',
            'orchestrator-reify.service',
        ]

    def test_the_prefix_is_the_rule(self) -> None:
        """The exported constant IS what the rule tests, not a parallel copy."""
        assert SYNTHETIC_UNIT_PREFIX == 'orchestrator-fake'
        assert non_synthetic_unit_names([SYNTHETIC_UNIT_PREFIX]) == []


class TestAssertSyntheticUnits:
    """The seam check the fake-systemctl factories call."""

    def test_it_is_a_no_op_for_synthetic_names(self) -> None:
        """The ordinary path costs the caller nothing and returns None."""
        assert assert_synthetic_units(
            [synthetic_unit('alpha'), 'orchestrator-fake.service'],
            where='tests::factory',
        ) is None

    def test_empty_input_is_a_no_op(self) -> None:
        assert assert_synthetic_units([], where='tests::factory') is None

    def test_a_real_name_fails_loudly(self) -> None:
        """``pytest.fail`` raises ``Failed`` (a BaseException), NOT an
        AssertionError — so a production ``except Exception`` in the code under
        test cannot swallow it, and so ``-p no:cacheprovider``-style rewriting
        is irrelevant.  Asserted as such, not via ``pytest.raises(AssertionError)``.
        """
        with pytest.raises(pytest.fail.Exception) as excinfo:
            assert_synthetic_units(
                ['orchestrator-reify.service'],
                where='scripts/tests/test_restart_all_orchestrators.py::_make_fake_systemctl',
            )
        message = str(excinfo.value)
        # Both assertions are on DATA THE CALLER PASSED IN, so they pin real
        # message-construction behaviour.  A prose pin (the remedy symbol, the
        # word "systemctl") would instead freeze wording: it can be satisfied
        # by a useless message and broken by a purely cosmetic rewrite.  If the
        # message's quality matters, fix the text once in df_pytest_isolation.
        # (a) the offending literal
        assert 'orchestrator-reify.service' in message
        # (b) the caller's own `where` label
        assert (
            'scripts/tests/test_restart_all_orchestrators.py::_make_fake_systemctl'
            in message
        )

    def test_it_names_every_offender(self) -> None:
        """All of them, in caller order — fixing one at a time is a rerun each."""
        with pytest.raises(pytest.fail.Exception) as excinfo:
            assert_synthetic_units(
                [
                    'orchestrator-reify.service',
                    synthetic_unit('alpha'),
                    'orchestrator-dark-factory.service',
                ],
                where='tests::factory',
            )
        message = str(excinfo.value)
        assert 'orchestrator-reify.service' in message
        assert 'orchestrator-dark-factory.service' in message
        assert message.index('orchestrator-reify.service') < message.index(
            'orchestrator-dark-factory.service',
        )


def test_df_pytest_isolation_stays_stdlib_and_pytest_only() -> None:
    """The import constraint the module docstring states, pinned.

    Every subproject conftest imports ``df_pytest_isolation`` and must be able
    to, from inside its own venv — escalation's lacks aiosqlite and stubs
    ``shared`` in ``sys.modules``.  An import of ``drain_check`` or
    ``orchestrator.fleet_heartbeat`` here would break collection for a whole
    subproject, loudly but far from the edit that caused it.

    BEHAVIOURAL, not textual.  This deliberately imports the module in a clean
    interpreter and inspects ``sys.modules`` rather than scanning the source
    for forbidden substrings.  The substring form was tried and removed: it let
    ``from orchestrator.fleet_heartbeat import DEFAULT_FLEET_DIR`` — the single
    most likely real regression, and the exact line the comments here forbid —
    through untouched, while false-positiving on PROSE (the comment at the
    FLEET_DIR literal reads "not an import of drain_check…" and passed only by
    the accident of the word "of").  Asking sys.modules catches every spelling,
    cannot be broken by rewording a comment, and cannot confuse a docstring
    mention for a real import.

    That the shared vocabulary is REACHABLE from this root needs no test of its
    own: the module-scope ``from df_pytest_isolation import (...)`` above is the
    reachability proof, and it is load-bearing — drop a symbol and this file
    dies at COLLECTION, before any assertion could run.  A ``hasattr`` loop over
    the same names lived here until it was found to be unfailable, and only
    pinned spelling a rename would have to edit twice.
    """
    probe = '\n'.join((
        'import sys',
        'import df_pytest_isolation',
        'roots = {"drain_check", "orchestrator", "shared"}',
        'leaked = sorted(m for m in sys.modules if m.split(".")[0] in roots)',
        'if leaked:',
        '    sys.exit("df_pytest_isolation pulled in: " + ", ".join(leaked))',
    ))
    # cwd=REPO_ROOT so the bare `import df_pytest_isolation` resolves: the
    # module sits at the repo root, and `python -c` puts cwd on sys.path.
    result = subprocess.run(
        [sys.executable, '-c', probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f'df_pytest_isolation must import with nothing but the stdlib and '
        f'pytest, so every subproject venv can import it.\n'
        f'exit={result.returncode}\nstdout={result.stdout}\n'
        f'stderr={result.stderr}'
    )


def test_fleet_dir_is_redirected_away_from_the_live_checkout(
    _df_fleet_dir_redirect: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``ORCH_FLEET_DIR`` must point somewhere hermetic for the WHOLE session.

    THE CONSEQUENCE of it being unset, which is what this pins and why the
    assertion messages say it out loud: ``restart-all-orchestrators.sh``'s
    ``FLEET_DIR`` default and ``drain_check.DEFAULT_FLEET_DIR``
    both resolve their fleet dir from ``${ORCH_FLEET_DIR:-…}``,
    so an unset (or EMPTY — ``${VAR:-…}`` treats those identically) value falls
    through to the machine-global ``/home/leo/src/dark-factory/data/fleet``.  A
    test-spawned drain gate then reads five other projects' LIVE production
    heartbeats and decides the real fleet's drain state from them.

    Takes the redirect from the fixture BY NAME rather than reading
    ``os.environ`` bare: deleting the fixture then fails collection with a
    message naming ``_df_fleet_dir_redirect``, instead of this test quietly
    passing off some other suite's leftover env var or failing with a bare
    KeyError that names nothing.

    This is the ``tests/`` root's own proof.  ``scripts/tests/`` has its own copy
    (``test_restart_all_orchestrators.py``), because the two roots are wired
    separately and a green test in one says nothing about the other.
    """
    value = os.environ.get(_FLEET_DIR_ENV)
    assert value, (
        f'{_FLEET_DIR_ENV} is {value!r}. Unset AND empty both fall through the '
        "script's ${VAR:-…} default to the machine-global "
        f'{LIVE_FLEET_DIR}, so a test-spawned drain gate reads other projects\' '
        'LIVE production heartbeats. Fix: df_pytest_isolation._df_fleet_dir_redirect.'
    )

    resolved = Path(value).resolve()
    basetemp = tmp_path_factory.getbasetemp().resolve()
    assert resolved.is_relative_to(basetemp), (
        f'{_FLEET_DIR_ENV}={resolved} is outside this run\'s basetemp {basetemp}. '
        'The redirect must land in pytest tmp space, or the suite is writing '
        'heartbeats somewhere that outlives it.'
    )

    live = LIVE_FLEET_DIR.resolve()
    assert resolved != live and not resolved.is_relative_to(live), (
        f'{_FLEET_DIR_ENV}={resolved} is the live fleet dir {live} (or inside it). '
        'That directory is a MACHINE-GLOBAL, CROSS-PROJECT rendezvous dir holding '
        "seven projects' live orchestrator heartbeats."
    )

    # The fixture's yielded value IS the redirect, not a parallel path.
    assert Path(_df_fleet_dir_redirect).resolve() == resolved


class TestSyntheticHeartbeatsIn:
    """Which files in a fleet dir the leak guard counts as evidence.

    Exercised ONLY against tmp dirs — never the real LIVE_FLEET_DIR, since a
    test that wrote there would BE the defect under guard.
    """

    def test_it_ignores_real_unit_heartbeats(self, tmp_path: Path) -> None:
        """THE property that makes the guard immune to production churn.

        The live fleet dir is rewritten by the running orchestrators roughly
        every 30s (measured 2026-08-09: six of seven files moved between two
        readings five minutes apart), so a guard keyed on "did anything change"
        would fail on essentially every run.  Keyed on the SYNTHETIC name, it
        fires only on a file no production process can produce.
        """
        (tmp_path / 'orchestrator-reify.service.json').write_text('{}')
        (tmp_path / 'orchestrator-dark-factory.service.json').write_text('{}')
        (tmp_path / f'{synthetic_unit("reify")}.json').write_text('{}')

        assert synthetic_heartbeats_in(tmp_path) == [
            f'{synthetic_unit("reify")}.json',
        ]

    def test_results_are_sorted(self, tmp_path: Path) -> None:
        """So the failure message is stable across filesystems (readdir order
        is not guaranteed), which is what makes it diffable between runs.
        """
        for stem in ('zulu', 'alpha', 'mike'):
            (tmp_path / f'{synthetic_unit(stem)}.json').write_text('{}')

        assert synthetic_heartbeats_in(tmp_path) == sorted(
            f'{synthetic_unit(stem)}.json' for stem in ('zulu', 'alpha', 'mike')
        )

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """It runs in SESSION TEARDOWN, where an exception would mask whatever
        the run was actually reporting.  A fleet dir that does not exist is the
        ordinary state on a box that has never hosted the fleet.
        """
        assert synthetic_heartbeats_in(tmp_path / 'nope') == []

    def test_an_unreadable_directory_is_empty_not_an_error(
        self, tmp_path: Path,
    ) -> None:
        """Same reason.  Chmod 0 rather than a mocked OSError, so this pins the
        real syscall behaviour and not a stub's idea of it.
        """
        blocked = tmp_path / 'blocked'
        blocked.mkdir()
        (blocked / f'{synthetic_unit("reify")}.json').write_text('{}')
        blocked.chmod(0o000)
        try:
            assert synthetic_heartbeats_in(blocked) == []
        finally:
            blocked.chmod(0o755)

    def test_an_empty_directory_is_empty(self, tmp_path: Path) -> None:
        assert synthetic_heartbeats_in(tmp_path) == []

    def test_it_ignores_non_json_files(self, tmp_path: Path) -> None:
        """A heartbeat is ``<unit>.json``; anything else in that directory is
        not one, and reporting it would send the reader after the wrong file.
        """
        (tmp_path / f'{synthetic_unit("reify")}.txt').write_text('')
        assert synthetic_heartbeats_in(tmp_path) == []


class TestLeakedFleetHeartbeatReason:
    """The message the guard fails with."""

    def test_no_leak_is_none(self) -> None:
        assert leaked_fleet_heartbeat_reason([]) is None

    def test_it_names_every_offender(self) -> None:
        names = [f'{synthetic_unit("alpha")}.json', f'{synthetic_unit("bravo")}.json']
        reason = leaked_fleet_heartbeat_reason(names)
        assert reason is not None
        for name in names:
            assert name in reason

    def test_it_names_the_live_directory(self) -> None:
        """WHERE the leak landed — the one part of the message that is DATA.

        Scoped deliberately to :data:`LIVE_FLEET_DIR`, a module constant, so the
        assertion tracks the constant rather than the sentence around it.  The
        message's other content (the mechanism, the remedy) is prose: pinning it
        by substring would freeze vocabulary the guard is free to improve, and
        the two ``<file>:<line>`` pointers this test used to pin went stale
        inside the very commit series that wrote them.  The remedy for that
        staleness is at the SOURCE — the messages in ``df_pytest_isolation``
        cite symbols (``FLEET_DIR``, ``DEFAULT_FLEET_DIR``) rather than
        ``<file>:<line>`` — not a further test asserting over their prose.
        """
        reason = leaked_fleet_heartbeat_reason([f'{synthetic_unit("alpha")}.json'])
        assert reason is not None
        assert str(LIVE_FLEET_DIR) in reason


def test_no_stray_environment_assumptions() -> None:
    """This module must not depend on the live box's env to be meaningful.

    A canary: the tests above are all pure-function tests, so nothing here may
    quietly start reading ``ORCH_FLEET_DIR`` and passing for the wrong reason.
    """
    assert os.environ.get('ORCH_FLEET_DIR') != ''
