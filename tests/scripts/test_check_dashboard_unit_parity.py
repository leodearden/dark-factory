"""Tests for scripts/check_dashboard_unit_parity.py.

All drift-logic tests run against inline fixture strings and tmp_path
directories — NEVER the host's real ~/.config/systemd/user/ — mirroring the
rule tests/scripts/test_check_fused_memory_unit_parity.py states in its own
docstring.

That rule is load-bearing here for a specific reason, not just portability:
as measured on 2026-08-01 the installed dark-factory-dashboard-watchdog.service
is still the pre-incident inline-shell copy, so the checker exits 1 against the
live host today. That is the CORRECT signal — installing the post-3308 units
belongs to task 3289 — but a test asserting parity against the live host would
be red on landing, and one asserting drift would flip red the moment 3289 fixes
it. Either encodes host state rather than checker behaviour.

The only real-tree reads are REPO-side (the committed dashboard/*.service and
*.timer files), used by the registry staleness guard.

Module loading: scripts/ is not a package / not on sys.path, so the checker is
loaded via importlib.util.spec_from_file_location, mirroring
tests/scripts/test_check_fused_memory_unit_parity.py::_load_checker.
"""

import importlib.util
import pathlib
import types

REPO_ROOT = pathlib.Path(__file__).parents[2]
CHECKER_PATH = REPO_ROOT / "scripts" / "check_dashboard_unit_parity.py"


def _load_checker() -> types.ModuleType:
    """Load scripts/check_dashboard_unit_parity.py by file path."""
    spec = importlib.util.spec_from_file_location(
        "check_dashboard_unit_parity", CHECKER_PATH
    )
    assert spec is not None, f"Could not build spec from {CHECKER_PATH}"
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# parse_unit_directives  (step-1 / step-2)
# ---------------------------------------------------------------------------

# Deliberately exercises every rule at once: a pre-header directive, both
# comment spellings, a blank line, a repeated key, and a four-physical-line
# backslash continuation shaped like the real dashboard ExecStart.
_SAMPLE_UNIT = """\
Type=simple
[Unit]
Description=Test Service
# hash comment
; semicolon comment

[Service]
Type=oneshot
Environment=A=1
Environment=B=2
ExecStart=/home/leo/.local/bin/uv run --project dashboard \\
  python -m uvicorn dashboard.app:app \\
  --host 127.0.0.1 --port 8080 \\
  --timeout-graceful-shutdown 8

[Install]
WantedBy=default.target
"""


def test_parse_unit_directives_returns_section_key_values_mapping():
    """Returns section → key → list-of-values, keyed by section name."""
    mod = _load_checker()
    parsed = mod.parse_unit_directives(_SAMPLE_UNIT)

    assert set(parsed) == {"Unit", "Service", "Install"}
    assert parsed["Service"]["Type"] == ["oneshot"]
    assert parsed["Install"]["WantedBy"] == ["default.target"]
    assert parsed["Unit"]["Description"] == ["Test Service"]


def test_parse_unit_directives_skips_comments_and_blanks():
    """'#' and ';' comment lines and blank lines contribute no directives."""
    mod = _load_checker()
    parsed = mod.parse_unit_directives(_SAMPLE_UNIT)

    unit_keys = set(parsed["Unit"])
    assert unit_keys == {"Description"}, (
        f"Comment lines leaked into [Unit] as directives: {unit_keys}"
    )
    # No key or value anywhere retains a comment marker.
    for section, directives in parsed.items():
        for key, values in directives.items():
            assert not key.startswith(("#", ";")), f"comment key in [{section}]: {key!r}"
            for value in values:
                assert "hash comment" not in value
                assert "semicolon comment" not in value


def test_parse_unit_directives_accumulates_repeated_keys():
    """A repeated directive accumulates into a list, in file order.

    This is the first of two deliberate divergences from the precedent's
    parse_unit_sections, whose flat line-list cannot express "the values of
    key K" — which is exactly what a per-directive comparison needs, and what
    makes the two Environment= lines of the real dashboard unit addressable.
    """
    mod = _load_checker()
    parsed = mod.parse_unit_directives(_SAMPLE_UNIT)

    assert parsed["Service"]["Environment"] == ["A=1", "B=2"]


def test_parse_unit_directives_joins_backslash_continuations():
    """A backslash continuation becomes ONE logical value.

    The second deliberate divergence. parse_unit_sections explicitly documents
    that it does NOT join continuations; without joining, every uvicorn flag
    living on a continuation line is invisible to the checker — which would
    silently defeat the whole point of comparing what 3306 added.
    """
    mod = _load_checker()
    parsed = mod.parse_unit_directives(_SAMPLE_UNIT)

    exec_values = parsed["Service"]["ExecStart"]
    assert len(exec_values) == 1, (
        f"ExecStart should join to a single logical value; got {exec_values!r}"
    )
    command = exec_values[0]
    assert "--timeout-graceful-shutdown 8" in command
    assert "--host 127.0.0.1" in command
    assert "--port 8080" in command
    assert "uvicorn dashboard.app:app" in command
    assert "\\" not in command, "Continuation backslashes must be stripped, not kept."


def test_parse_unit_directives_drops_lines_before_first_section_header():
    """A directive before any [Section] header is dropped, not misattributed."""
    mod = _load_checker()
    parsed = mod.parse_unit_directives(_SAMPLE_UNIT)

    # The leading `Type=simple` precedes [Unit]; the only Type= that survives
    # is [Service]'s `oneshot`.
    assert "Type" not in parsed["Unit"]
    assert parsed["Service"]["Type"] == ["oneshot"]
    assert all("simple" not in v for vs in parsed["Service"].values() for v in vs)


def test_parse_unit_directives_splits_on_first_equals_only():
    """Only the FIRST '=' separates key from value.

    Environment=A=1 must yield key 'Environment', value 'A=1' — splitting on
    every '=' would destroy the variable assignment the Environment= branch
    later needs to read.
    """
    mod = _load_checker()
    parsed = mod.parse_unit_directives("[Service]\nEnvironment=DASH_ROOTS=/a,/b\n")

    assert parsed["Service"]["Environment"] == ["DASH_ROOTS=/a,/b"]


# ---------------------------------------------------------------------------
# compare_unit — value-compared directives  (step-3 / step-4)
# ---------------------------------------------------------------------------


def _value_spec(mod: types.ModuleType, *compared: tuple[str, str]):
    """A minimal UnitSpec that value-compares exactly *compared*."""
    return mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        compared=compared,
    )


def test_compare_unit_identical_values_yield_no_drift():
    """Equal values on a compared directive are parity."""
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "TimeoutStopSec"))
    text = "[Service]\nTimeoutStopSec=15\n"

    assert mod.compare_unit(spec, text, text) == []


def test_compare_unit_differing_value_yields_one_drift():
    """A compared directive whose values disagree yields exactly one Drift."""
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "TimeoutStopSec"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nTimeoutStopSec=15\n",
        "[Service]\nTimeoutStopSec=30\n",
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert drift.unit == "fixture.service"
    assert drift.section == "Service"
    assert drift.key == "TimeoutStopSec"
    assert "15" in drift.repo_value
    assert "30" in drift.installed_value
    assert drift.reason, "Every Drift must carry a non-empty reason."


def test_compare_unit_directive_missing_from_installed_is_drift():
    """Declared in the repo copy, absent from the installed copy → drift."""
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "Restart"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nRestart=on-failure\n",
        "[Service]\nType=simple\n",
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert drift.key == "Restart"
    assert drift.installed_value == mod._ABSENT
    assert "on-failure" in drift.repo_value


def test_compare_unit_comparison_is_symmetric():
    """A directive present ONLY on the installed side is drift too.

    Asymmetric comparison would silently bless anything hand-added to the
    installed unit — e.g. an installed `Restart=always` overriding the repo's
    deliberate on-failure policy would read as parity, which is precisely the
    class of divergence this checker exists to surface.
    """
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "Restart"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nType=simple\n",
        "[Service]\nRestart=always\n",
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert drift.key == "Restart"
    assert drift.repo_value == mod._ABSENT
    assert "always" in drift.installed_value


def test_compare_unit_ignores_keys_not_on_the_spec():
    """A differing directive NOT on the compared list is ignored.

    The check is deliberately BOUNDED to a curated registry. An unbounded diff
    would fire on Description, After and every comment reflow, and a gate that
    cries wolf gets disabled within a week.
    """
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "TimeoutStopSec"))

    drifts = mod.compare_unit(
        spec,
        "[Unit]\nDescription=Repo wording\n[Service]\nTimeoutStopSec=15\n",
        "[Unit]\nDescription=Totally different wording\n[Service]\nTimeoutStopSec=15\n",
    )

    assert drifts == []


def test_compare_unit_compares_the_whole_values_list():
    """A repeated compared directive that gained an occurrence is drift.

    Comparing only the first value would miss a second, contradicting
    occurrence — systemd applies both, so the checker must see both.
    """
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "ExecStartPre"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nExecStartPre=/bin/true\n",
        "[Service]\nExecStartPre=/bin/true\nExecStartPre=/bin/false\n",
    )

    assert len(drifts) == 1, drifts
    assert "/bin/false" in drifts[0].installed_value


def test_compare_unit_key_absent_from_both_sides_is_not_drift():
    """A compared key neither copy declares is parity, not a phantom drift."""
    mod = _load_checker()
    spec = _value_spec(mod, ("Service", "RestartMaxDelaySec"))

    assert mod.compare_unit(spec, "[Service]\nType=simple\n", "[Service]\nType=simple\n") == []


# ---------------------------------------------------------------------------
# compare_unit — presence-only directives  (step-5 / step-6)
# ---------------------------------------------------------------------------


def _presence_spec(mod: types.ModuleType, *present_only: tuple[str, str]):
    """A minimal UnitSpec that presence-checks exactly *present_only*."""
    return mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        present_only=present_only,
    )


def test_presence_only_ignores_differing_host_paths():
    """A host-specific value difference is NOT drift.

    WorkingDirectory carries an absolute path that legitimately differs per
    machine. Value-comparing it would make the checker fire on every host that
    is not this one — the fastest possible route to the gate being disabled.
    """
    mod = _load_checker()
    spec = _presence_spec(mod, ("Service", "WorkingDirectory"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nWorkingDirectory=/home/leo/src/dark-factory\n",
        "[Service]\nWorkingDirectory=/opt/df\n",
    )

    assert drifts == []


def test_presence_only_missing_from_installed_is_drift():
    """A presence-only directive absent from the installed copy IS drift."""
    mod = _load_checker()
    spec = _presence_spec(mod, ("Service", "ExecStart"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nExecStart=/usr/bin/true\n",
        "[Service]\nType=oneshot\n",
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert drift.key == "ExecStart"
    assert drift.installed_value == mod._ABSENT
    assert "absent" in drift.reason, (
        "The reason must distinguish 'required directive absent' from a value "
        f"mismatch; got {drift.reason!r}"
    )


def test_presence_only_missing_from_repo_is_drift():
    """Presence checking is symmetric too."""
    mod = _load_checker()
    spec = _presence_spec(mod, ("Service", "ExecStart"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nType=oneshot\n",
        "[Service]\nExecStart=/usr/bin/true\n",
    )

    assert len(drifts) == 1, drifts
    assert drifts[0].repo_value == mod._ABSENT


def test_presence_only_absent_from_both_is_not_drift():
    """A presence-only key neither copy declares is parity — nothing to propagate."""
    mod = _load_checker()
    spec = _presence_spec(mod, ("Service", "Documentation"))

    drifts = mod.compare_unit(
        spec,
        "[Service]\nType=oneshot\n",
        "[Service]\nType=oneshot\n",
    )

    assert drifts == []


def test_presence_only_and_value_compared_reasons_differ():
    """The two drift classes are distinguishable in the report.

    An operator reading 'value differs' reaches for a diff; one reading
    'absent from the installed copy' reaches for the installer. Collapsing
    both into one message costs that distinction.
    """
    mod = _load_checker()
    spec = mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        compared=(("Service", "TimeoutStopSec"),),
        present_only=(("Service", "ExecStart"),),
    )

    drifts = mod.compare_unit(
        spec,
        "[Service]\nTimeoutStopSec=15\nExecStart=/usr/bin/true\n",
        "[Service]\nTimeoutStopSec=30\n",
    )

    by_key = {d.key: d for d in drifts}
    assert set(by_key) == {"TimeoutStopSec", "ExecStart"}, drifts
    assert by_key["TimeoutStopSec"].reason != by_key["ExecStart"].reason


# ---------------------------------------------------------------------------
# compare_unit — Environment= name-set + allowlisted value divergence
# (step-7 / step-8)
# ---------------------------------------------------------------------------

# The real measured divergence, 2026-08-01: the installed dashboard unit
# carries nine aggregation roots, the committed one carries this repo only.
# The repo unit's own comment declares this deliberate — "additional project
# roots are LOCAL settings, added to the installed unit, not committed here".
_NINE_ROOTS = (
    "/home/leo/src/dark-factory,/home/leo/src/a,/home/leo/src/b,/home/leo/src/c,"
    "/home/leo/src/d,/home/leo/src/e,/home/leo/src/f,/home/leo/src/g,/home/leo/src/h"
)


def _env_spec(mod: types.ModuleType):
    """A minimal UnitSpec that name-set-compares [Service] Environment=."""
    return mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        environment_section="Service",
    )


def test_environment_allowlisted_variable_may_diverge_in_value():
    """THE REAL MEASURED DIVERGENCE: nine installed roots vs one committed → parity.

    This single case decides whether the gate survives. A value-equality check
    on DASHBOARD_KNOWN_PROJECT_ROOTS would fire on literally every run of a
    correctly-configured host, and a gate that is always red is a gate someone
    deletes — leaving the accidental drift it exists to catch unwatched too.
    """
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory\n",
        f"[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS={_NINE_ROOTS}\n",
    )

    assert drifts == [], (
        "DASHBOARD_KNOWN_PROJECT_ROOTS is on DIVERGENCE_ALLOWLIST; its value "
        f"is expected to differ per host. Got: {drifts}"
    )


def test_environment_allowlist_does_not_bless_the_variable_disappearing():
    """Allowlisting a VALUE must not allowlist the variable vanishing.

    The allowlist says "this variable's value is a local setting", not "this
    variable is optional". An installed unit that dropped it entirely would
    silently lose every aggregation root, which is a real regression.
    """
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory\n",
        "[Service]\nType=simple\n",
    )

    assert len(drifts) == 1, drifts
    assert "DASHBOARD_KNOWN_PROJECT_ROOTS" in drifts[0].key
    assert drifts[0].installed_value == mod._ABSENT


def test_environment_non_allowlisted_value_difference_is_drift():
    """A variable NOT on the allowlist must match by value."""
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=DASHBOARD_DB_PATH=/var/lib/df/db\n",
        "[Service]\nEnvironment=DASHBOARD_DB_PATH=/tmp/other.db\n",
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert "DASHBOARD_DB_PATH" in drift.key, (
        f"The drift must name the VARIABLE, not just 'Environment'; got {drift.key!r}"
    )
    assert "/var/lib/df/db" in drift.repo_value
    assert "/tmp/other.db" in drift.installed_value


def test_environment_extra_installed_variable_is_drift():
    """A non-allowlisted variable declared only on the installed side is drift."""
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nType=simple\n",
        "[Service]\nEnvironment=DASHBOARD_DEBUG=1\n",
    )

    assert len(drifts) == 1, drifts
    assert "DASHBOARD_DEBUG" in drifts[0].key
    assert drifts[0].repo_value == mod._ABSENT


def test_environment_name_set_compared_across_several_variables():
    """Names are compared as a SET; matching values on shared names are parity."""
    mod = _load_checker()
    spec = _env_spec(mod)

    same = "[Service]\nEnvironment=A=1\nEnvironment=B=2\n"
    assert mod.compare_unit(spec, same, same) == []

    # Order must not matter — systemd applies both regardless of file order.
    reordered = "[Service]\nEnvironment=B=2\nEnvironment=A=1\n"
    assert mod.compare_unit(spec, same, reordered) == []


def test_divergence_allowlist_entries_are_documented():
    """Every allowlist entry carries a non-empty documented reason.

    An allowlist is a hole in the gate. Requiring prose per entry means a hole
    cannot be widened silently — the next person to add one has to say why in
    a place a reviewer will read.
    """
    mod = _load_checker()

    assert mod.DIVERGENCE_ALLOWLIST, "The allowlist should not be empty."
    for name, reason in mod.DIVERGENCE_ALLOWLIST.items():
        assert isinstance(reason, str) and reason.strip(), (
            f"DIVERGENCE_ALLOWLIST[{name!r}] must carry a documented reason."
        )

    assert "DASHBOARD_KNOWN_PROJECT_ROOTS" in mod.DIVERGENCE_ALLOWLIST


def test_environment_not_compared_when_spec_omits_the_section():
    """A spec with no environment_section does no Environment= comparison.

    The watchdog units declare no Environment= at all; the branch must stay
    off for them rather than inventing comparisons.
    """
    mod = _load_checker()
    spec = mod.UnitSpec(name="fixture.timer", repo_relpath="dashboard/fixture.timer")

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=A=1\n",
        "[Service]\nEnvironment=A=2\nEnvironment=B=3\n",
    )

    assert drifts == []


# ---------------------------------------------------------------------------
# compare_unit — uvicorn ExecStart flag comparison  (step-9 / step-10)
# ---------------------------------------------------------------------------


def _exec_unit(uv_bin: str, repo_root: str, tail: str) -> str:
    """A dashboard-service fixture whose ExecStart is a real 4-line continuation.

    The explanatory comment above ExecStart names --timeout-keep-alive in
    prose, exactly as both real dashboard units do — see the case (e) test.
    """
    return f"""\
[Service]
Type=simple
WorkingDirectory={repo_root}
# --timeout-keep-alive 5 is uvicorn's own default, pinned deliberately.
# --timeout-graceful-shutdown 8 bounds the connection drain.
ExecStart={uv_bin} run --project dashboard \\
  python -m uvicorn dashboard.app:app \\
  --host 127.0.0.1 --port 8080 \\
  {tail}
"""


def _flag_spec(mod: types.ModuleType, *flags: str):
    """A minimal UnitSpec that compares exactly *flags* inside ExecStart."""
    return mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        exec_start_flags=flags,
    )


_BOTH_FLAGS = "--timeout-graceful-shutdown 8 \\\n  --timeout-keep-alive 5"


def test_exec_start_flags_ignore_host_specific_prefixes():
    """Differing uv path and repo root are NOT drift when the flags agree.

    ExecStart is presence-only precisely because of these prefixes; the flags
    are how its meaningful content is still compared.
    """
    mod = _load_checker()
    spec = _flag_spec(mod, "timeout-graceful-shutdown", "timeout-keep-alive")

    drifts = mod.compare_unit(
        spec,
        _exec_unit("/home/leo/.local/bin/uv", "/home/leo/src/dark-factory", _BOTH_FLAGS),
        _exec_unit("/usr/local/bin/uv", "/opt/df", _BOTH_FLAGS),
    )

    assert drifts == [], f"Host prefixes must be ignored by construction; got {drifts}"


def test_exec_start_flag_value_difference_is_drift():
    """A flag whose value differs yields one Drift naming that flag."""
    mod = _load_checker()
    spec = _flag_spec(mod, "timeout-graceful-shutdown", "timeout-keep-alive")

    drifts = mod.compare_unit(
        spec,
        _exec_unit("/usr/bin/uv", "/opt/df", _BOTH_FLAGS),
        _exec_unit(
            "/usr/bin/uv",
            "/opt/df",
            "--timeout-graceful-shutdown 20 \\\n  --timeout-keep-alive 5",
        ),
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert "timeout-graceful-shutdown" in drift.key
    assert drift.repo_value == "8"
    assert drift.installed_value == "20"


def test_exec_start_flag_missing_from_installed_is_drift():
    """The exact pre-3306 shape: the installed command lacks the flag entirely."""
    mod = _load_checker()
    spec = _flag_spec(mod, "timeout-graceful-shutdown", "timeout-keep-alive")

    drifts = mod.compare_unit(
        spec,
        _exec_unit("/usr/bin/uv", "/opt/df", _BOTH_FLAGS),
        _exec_unit("/usr/bin/uv", "/opt/df", "--timeout-graceful-shutdown 8"),
    )

    assert len(drifts) == 1, drifts
    (drift,) = drifts
    assert "timeout-keep-alive" in drift.key
    assert drift.installed_value == mod._ABSENT
    assert drift.repo_value == "5"


def test_exec_start_flag_accepts_the_equals_spelling():
    """`--flag=value` equals `--flag value`.

    uvicorn's parser is click-based, so both spellings behave identically.
    Accepting only the space-separated form would report an '=' -form edit as
    a missing flag while the flag is plainly there.
    """
    mod = _load_checker()
    spec = _flag_spec(mod, "timeout-keep-alive")

    drifts = mod.compare_unit(
        spec,
        _exec_unit("/usr/bin/uv", "/opt/df", "--timeout-keep-alive 5"),
        _exec_unit("/usr/bin/uv", "/opt/df", "--timeout-keep-alive=5"),
    )

    assert drifts == []


def test_exec_start_flag_lookup_ignores_the_comment_block():
    """A flag named only in the comment above ExecStart is NOT on the command.

    Both real dashboard units discuss these flags in prose right above
    ExecStart, so a whole-file regex would keep reporting a value after the
    flag had actually been deleted from the command — reporting parity on a
    unit that lost the very flag being checked. The lookup is scoped to the
    parsed ExecStart value, from which comments are already gone.
    """
    mod = _load_checker()
    spec = _flag_spec(mod, "timeout-keep-alive")

    # The repo copy really runs the flag; the installed copy only MENTIONS it
    # in the comment block, having dropped it from the command.
    installed = _exec_unit("/usr/bin/uv", "/opt/df", "--log-level info")
    assert "--timeout-keep-alive" in installed, "fixture must retain the prose mention"

    drifts = mod.compare_unit(
        spec,
        _exec_unit("/usr/bin/uv", "/opt/df", "--timeout-keep-alive 5"),
        installed,
    )

    assert len(drifts) == 1, (
        "The comment's mention of --timeout-keep-alive must not be mistaken "
        f"for the flag being present on the command. Got: {drifts}"
    )
    assert drifts[0].installed_value == mod._ABSENT


def test_exec_start_flags_absent_from_both_is_not_drift():
    """A compared flag neither command carries is parity, not a phantom drift."""
    mod = _load_checker()
    spec = _flag_spec(mod, "workers")

    text = _exec_unit("/usr/bin/uv", "/opt/df", _BOTH_FLAGS)
    assert mod.compare_unit(spec, text, text) == []


def test_exec_start_flag_helper_reads_the_parsed_value():
    """_exec_start_flag returns the raw token, or None when the flag is absent.

    Raw token, not int: --host 127.0.0.1 must work as well as --port 8080.
    """
    mod = _load_checker()
    parsed = mod.parse_unit_directives(
        _exec_unit("/usr/bin/uv", "/opt/df", _BOTH_FLAGS)
    )["Service"]

    assert mod._exec_start_flag(parsed, "host") == "127.0.0.1"
    assert mod._exec_start_flag(parsed, "port") == "8080"
    assert mod._exec_start_flag(parsed, "timeout-keep-alive") == "5"
    assert mod._exec_start_flag(parsed, "workers") is None
    assert mod._exec_start_flag({}, "host") is None


# ---------------------------------------------------------------------------
# The UNITS registry and its staleness guard  (step-11 / step-12)
# ---------------------------------------------------------------------------
#
# These read REPO-side files only — never ~/.config/systemd/user/ — so they
# stay green on a host whose installed units are drifted (which this one is,
# deliberately, until task 3289 lands) and on CI, which has no installed units
# at all.

_DASHBOARD_SERVICE = "dark-factory-dashboard.service"
_WATCHDOG_SERVICE = "dark-factory-dashboard-watchdog.service"
_WATCHDOG_TIMER = "dark-factory-dashboard-watchdog.timer"


def test_units_registry_covers_exactly_the_three_dashboard_units():
    """The registry names the three units the task enumerates, and no others."""
    mod = _load_checker()

    assert set(mod.UNITS) == {_DASHBOARD_SERVICE, _WATCHDOG_SERVICE, _WATCHDOG_TIMER}
    for name, spec in mod.UNITS.items():
        assert spec.name == name, f"UNITS key {name!r} disagrees with spec.name {spec.name!r}"


def test_units_registry_repo_paths_all_exist():
    """Every repo_relpath resolves to a real committed file."""
    mod = _load_checker()

    for name, spec in mod.UNITS.items():
        path = REPO_ROOT / spec.repo_relpath
        assert path.is_file(), f"{name}: repo_relpath {spec.repo_relpath} does not exist"


def test_registry_keys_are_all_declared_in_the_committed_units():
    """STALENESS GUARD: every registered key really exists in the repo unit.

    Without this, a typo'd or obsoleted key silently checks NOTHING — the
    directive is absent from both copies, so it compares equal forever and the
    gate rots into a no-op that still reports green. That failure mode is
    invisible by construction, which is exactly why it needs its own test.

    Note this guards KEYS, not values. Expected values are read from the repo
    unit at run time by design, so there is no third copy of them to go stale.
    """
    mod = _load_checker()

    for name, spec in mod.UNITS.items():
        parsed = mod.parse_unit_directives(
            (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
        )
        for section, key in tuple(spec.compared) + tuple(spec.present_only):
            assert key in parsed.get(section, {}), (
                f"{name}: registry lists [{section}] {key}, but the committed "
                f"unit {spec.repo_relpath} does not declare it — the entry "
                "checks nothing. Fix the registry or the unit."
            )


def test_registry_exec_start_flags_are_really_on_the_committed_command():
    """STALENESS GUARD, flag edition: each registered flag is on the real command."""
    mod = _load_checker()

    for name, spec in mod.UNITS.items():
        if not spec.exec_start_flags:
            continue
        parsed = mod.parse_unit_directives(
            (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
        )
        for flag in spec.exec_start_flags:
            assert mod._exec_start_flag(parsed.get("Service", {}), flag) is not None, (
                f"{name}: registry compares --{flag}, but it is absent from the "
                f"committed ExecStart in {spec.repo_relpath} — the entry checks nothing."
            )


def test_registry_environment_sections_really_declare_environment():
    """STALENESS GUARD, Environment edition."""
    mod = _load_checker()

    for name, spec in mod.UNITS.items():
        if spec.environment_section is None:
            continue
        parsed = mod.parse_unit_directives(
            (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
        )
        assert "Environment" in parsed.get(spec.environment_section, {}), (
            f"{name}: registry compares Environment= in "
            f"[{spec.environment_section}], but the committed unit declares none."
        )


def test_dashboard_service_spec_pins_the_tasks_minimum_coverage():
    """The dashboard service compares the restart directives and 3306's flags."""
    mod = _load_checker()
    spec = mod.UNITS[_DASHBOARD_SERVICE]

    compared_keys = {key for _section, key in spec.compared}
    assert {"Restart", "RestartSec", "TimeoutStopSec"} <= compared_keys, compared_keys
    assert {"timeout-graceful-shutdown", "timeout-keep-alive"} <= set(spec.exec_start_flags)


def test_watchdog_service_spec_compares_the_whole_tick_bound():
    """TimeoutStartSec is 3308's bound on the whole watchdog tick."""
    mod = _load_checker()
    spec = mod.UNITS[_WATCHDOG_SERVICE]

    assert ("Service", "TimeoutStartSec") in spec.compared


def test_watchdog_timer_spec_compares_the_cadence():
    """The timer cadence is load-bearing, not a free knob.

    The watchdog requires FAIL_STREAK (=3) consecutive failed probes, so
    3 x OnUnitActiveSec sets the ~90s sustained-outage detection latency. An
    installed timer that drifted to a different interval would silently change
    that latency in the same proportion.
    """
    mod = _load_checker()
    spec = mod.UNITS[_WATCHDOG_TIMER]

    compared_keys = {key for _section, key in spec.compared}
    assert {"OnBootSec", "OnUnitActiveSec"} <= compared_keys, compared_keys


def test_repo_units_are_at_parity_with_themselves():
    """Every spec compares a unit to itself with zero drift.

    A sanity check on the registry as a whole: if any branch mis-handled a
    real unit's shape, comparing a file to itself would still report drift.
    """
    mod = _load_checker()

    for name, spec in mod.UNITS.items():
        text = (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
        assert mod.compare_unit(spec, text, text) == [], f"{name} drifts against itself"


# ---------------------------------------------------------------------------
# main(argv) exit codes and the report  (step-13 / step-14)
# ---------------------------------------------------------------------------
#
# BOTH sides are tmp_path fixtures — the repo root as well as the installed
# dir. Pointing --repo-root at the real tree would work today but would couple
# these assertions to the committed units' current contents.


def _fake_repo(tmp_path: pathlib.Path, mod: types.ModuleType) -> pathlib.Path:
    """Build a fake repo root holding a copy of each committed unit."""
    root = tmp_path / "repo"
    for spec in mod.UNITS.values():
        dest = root / spec.repo_relpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return root


def _installed_from(
    tmp_path: pathlib.Path,
    mod: types.ModuleType,
    repo_root: pathlib.Path,
    *,
    omit: tuple[str, ...] = (),
    edits: dict[str, tuple[str, str]] | None = None,
) -> pathlib.Path:
    """Build a fake installed dir from *repo_root*, optionally omitting/editing units."""
    installed = tmp_path / "installed"
    installed.mkdir(parents=True, exist_ok=True)
    for name, spec in mod.UNITS.items():
        if name in omit:
            continue
        text = (repo_root / spec.repo_relpath).read_text(encoding="utf-8")
        if edits and name in edits:
            old, new = edits[name]
            assert old in text, f"fixture edit target {old!r} not found in {name}"
            text = text.replace(old, new)
        (installed / name).write_text(text, encoding="utf-8")
    return installed


def test_main_returns_0_on_full_parity(tmp_path: pathlib.Path, capsys):
    """All three installed copies match their repo copies → exit 0."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])

    assert rc == 0, capsys.readouterr().out


def test_main_returns_1_and_names_the_directive_and_path(tmp_path: pathlib.Path, capsys):
    """Drift → exit 1, and the report names the directive AND the file.

    A report that says only "drift detected" forces a hand diff, which is the
    work the checker exists to remove.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "TimeoutStopSec" in out
    assert "15" in out and "30" in out
    assert str(installed / _DASHBOARD_SERVICE) in out, (
        f"The report must name the offending installed file path. Got:\n{out}"
    )


def test_main_returns_2_when_an_installed_unit_is_absent(tmp_path: pathlib.Path, capsys):
    """Missing installed unit, nothing else drifting → exit 2, naming the path."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo, omit=(_WATCHDOG_TIMER,))

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 2, out
    assert str(installed / _WATCHDOG_TIMER) in out


def test_main_drift_dominates_absence(tmp_path: pathlib.Path, capsys):
    """PRECEDENCE: one unit absent AND another drifting → exit 1, not 2.

    setup-host.sh treats 2 as a benign "not installed here, skipping" and only
    1 as something to act on, so reporting 2 would downgrade a real finding to
    a shrug and let an unrelated uninstalled unit mask it.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        omit=(_WATCHDOG_TIMER,),
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 1, f"Drift must dominate absence. Got {rc}:\n{out}"
    # The absent unit is still REPORTED — dominated, not hidden.
    assert str(installed / _WATCHDOG_TIMER) in out


def test_main_unit_flag_restricts_the_run(tmp_path: pathlib.Path, capsys):
    """--unit narrows the run; a drifting unit outside the selection is not reported."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )

    rc = mod.main(
        [
            "--repo-root",
            str(repo),
            "--installed-dir",
            str(installed),
            "--unit",
            _WATCHDOG_TIMER,
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0, out
    assert "TimeoutStopSec" not in out
    assert _DASHBOARD_SERVICE not in out


def test_main_every_emitted_line_carries_the_log_tag(tmp_path: pathlib.Path, capsys):
    """Every printed line is prefixed with the log tag, so the report is greppable."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        omit=(_WATCHDOG_TIMER,),
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )

    mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    captured = capsys.readouterr()

    assert captured.out.strip(), "The checker must report something."
    for line in (captured.out + captured.err).splitlines():
        if not line.strip():
            continue
        assert line.startswith(f"[{mod.LOG_TAG}]"), f"Untagged output line: {line!r}"


def test_main_report_points_at_the_remediation_command(tmp_path: pathlib.Path, capsys):
    """The drift report names how to fix it, since there is no --fix."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )

    mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert "setup-host.sh" in out, (
        f"A checker with no --fix must say what to run instead. Got:\n{out}"
    )


def test_main_reports_the_watchdog_pre_incident_drift(tmp_path: pathlib.Path, capsys):
    """The real measured host drift: the pre-incident inline-shell watchdog.

    Reproduced as a FIXTURE rather than read from ~/.config/systemd/user/, so
    the assertion survives task 3289 installing the post-3308 units.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo, omit=(_WATCHDOG_SERVICE,))
    (installed / _WATCHDOG_SERVICE).write_text(
        "[Unit]\n"
        "Description=Dashboard availability watchdog\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        "ExecStart=/bin/sh -c 'curl -sf --max-time 5 "
        "http://127.0.0.1:8080/healthz "
        "|| systemctl --user restart dark-factory-dashboard.service'\n",
        encoding="utf-8",
    )

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "TimeoutStartSec" in out, (
        f"3308's whole-tick bound is missing from the installed copy: {out}"
    )
    assert _WATCHDOG_SERVICE in out
