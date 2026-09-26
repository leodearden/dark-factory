"""pytest-xdist worker-crash output specimens shared by the verify tests.

Lives in ``_xdist_crash_fixtures.py`` (not ``conftest.py``) to follow the
``_hold_history_fixtures.py`` pattern: importable from any test file without
``sys.modules['conftest']`` collisions across subprojects.

Each constant transcribes a shape pytest / pytest-xdist actually prints, kept
as a plain string so a consumer's marker profile stays readable at its
assertion site. Consumers: test_verify_env_transient.py (the
``_is_bare_xdist_worker_crash`` routing tests) and test_verify.py (the cause
hint, leg summary, aggregation and failure report).
"""

# task 2365: bare pytest-xdist worker-crash signature. Grounded in
# config.yaml's task-2361 comment (under host CPU oversubscription a starved
# xdist worker crosses the per-test wall-clock ceiling, gets os._exit()'d by
# pytest-timeout's thread method, and --max-worker-restart=0 turns that into
# a false-failing per-test "node down" on whatever test happens to be
# running) and test_cli.py's grounded wording:
# ``[gwN] node down: Not properly terminated``. NO ^E  /^FAILED /failed-summary
# lines — a hard os._exit() worker kill produces no assertion traceback.
XDIST_IN_FLIGHT_NODEID = 'orchestrator/tests/test_config.py::TestFoo::test_bar'
XDIST_WORKER_CRASH_OUTPUT = (
    'orchestrator/tests/test_config.py ....\n'
    '[gw3] node down: Not properly terminated\n'
    f"worker gw3 crashed while running '{XDIST_IN_FLIGHT_NODEID}'\n"
)

# task 5082: xdist's BAILOUT marker. `xdist/dsession.py::DSession.worker_errordown`
# prints it only once the `--max-worker-restart` cap is exceeded (a cap of 0
# here, dark-factory's own setting) and then calls `triggershutdown()`,
# abandoning every queued test; its terminal summary re-emits it in this
# ``=``-barred form.
XDIST_BAILOUT_LINE = (
    '=========== xdist: worker gw3 crashed and worker restarting disabled ===========\n'
)

# The tally pytest prints after that bailout counts only the tests that had
# already run, so it is PARTIAL. Modelled on esc-4176-6: this truncated tally
# versus ``19622 passed, 17 skipped`` on a clean re-run of the same command.
XDIST_PARTIAL_TALLY = '1 failed, 728 passed, 1 skipped in 209.67s\n'

# A worker death that truncated the session, with no FAILED line at all.
XDIST_SESSION_ABORTED_OUTPUT = XDIST_WORKER_CRASH_OUTPUT + XDIST_BAILOUT_LINE + XDIST_PARTIAL_TALLY

# The non-zero-cap spelling of the same bailout: same `triggershutdown()`,
# same truncation.
XDIST_MAX_WORKERS_REACHED_OUTPUT = (
    XDIST_WORKER_CRASH_OUTPUT
    + '=========== xdist: maximum crashed workers reached: 2 ===========\n'
    + '3 failed, 402 passed in 88.10s\n'
)

# The SIBLING branch: below the cap xdist prints ``replacing crashed worker
# gwN``, clones the node, and the session runs to COMPLETION. It carries the
# identical crash signature as the bailout fixtures above — which is why the
# abort label must key on the bailout literal, never on the crash signature.
XDIST_WORKER_REPLACED_OUTPUT = (
    XDIST_WORKER_CRASH_OUTPUT
    + 'replacing crashed worker gw3\n'
    + '19622 passed, 17 skipped in 953.70s\n'
)

# task 5082: the esc-4292-3 shape, and the truncated-session specimen. xdist's
# `handle_crashitem` FABRICATED an ``outcome="failed"`` / ``when="???"`` report
# for the test the dead worker had in flight, and pytest printed an
# ordinary-looking FAILED line for it — naming the SAME node-id as the crash
# notice, by construction. esc-4292-3 measured that test passing in isolation.
#
# The FAILED line deliberately carries NO `` - worker 'gwN' crashed ...``
# suffix: pytest renders it through `_pytest/terminal.py::_format_trimmed`,
# which drops it at a default terminal width, so this pins the HARD case in
# which only the untrimmed crash notice above can attribute the line.
XDIST_CRASH_ATTRIBUTED_FAILED_LINE = f'FAILED {XDIST_IN_FLIGHT_NODEID}\n'
XDIST_CRASH_ATTRIBUTED_FAILED_OUTPUT = (
    XDIST_WORKER_CRASH_OUTPUT
    + XDIST_CRASH_ATTRIBUTED_FAILED_LINE
    + XDIST_BAILOUT_LINE
    + XDIST_PARTIAL_TALLY
)

# task 5082: the in-flight test FAILED in its call phase and THEN its worker
# died in teardown. xdist had already forwarded the genuine ``when="call"``
# report, and `handle_crashitem` then added its synthesized one for the same
# node-id — so the crash notice names the test AND it has two FAILED lines,
# only one of which can be xdist's artefact.
XDIST_FAILED_THEN_CRASHED_OUTPUT = (
    XDIST_WORKER_CRASH_OUTPUT
    + 'E   AssertionError: expected 3, got 4\n'
    + f'FAILED {XDIST_IN_FLIGHT_NODEID} - AssertionError\n'
    + XDIST_CRASH_ATTRIBUTED_FAILED_LINE
    + XDIST_BAILOUT_LINE
    + '2 failed, 728 passed, 1 skipped in 209.67s\n'
)
