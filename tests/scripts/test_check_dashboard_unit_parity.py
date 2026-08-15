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
import os
import pathlib
import subprocess
import sys
import types

import pytest

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

    # The precondition of the case below, asserted where it is exercised.
    assert "DASHBOARD_KNOWN_PROJECT_ROOTS" in mod.DIVERGENCE_ALLOWLIST

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory\n",
        f"[Service]\nEnvironment=DASHBOARD_KNOWN_PROJECT_ROOTS={_NINE_ROOTS}\n",
    )

    assert drifts == [], (
        "DASHBOARD_KNOWN_PROJECT_ROOTS is on DIVERGENCE_ALLOWLIST; its value "
        f"is expected to differ per host. Got: {drifts}"
    )


def test_environment_multi_assignment_line_equals_the_one_per_line_spelling():
    """systemd accepts several assignments on one Environment= line; so must this.

    Measured against the pre-shlex parser: ``Environment=A=1 B=2`` parsed as
    the single variable ``A`` with value ``1 B=2``, so comparing it against the
    one-per-line spelling reported TWO bogus drifts (a value difference on A,
    plus B as installed-only). A pure reformat is not drift, and an
    unexplainable red is how a warn-only gate loses the credibility that is
    its whole value.
    """
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=A=1 B=2\n",
        "[Service]\nEnvironment=A=1\nEnvironment=B=2\n",
    )

    assert drifts == [], (
        "Both spellings declare A=1 and B=2; systemd reads them identically. "
        f"Got: {drifts}"
    )


def test_environment_quoted_assignments_parse_as_their_unquoted_form():
    """Quoted values are unwrapped, not folded into the variable NAME.

    Measured against the pre-shlex parser: ``Environment="A=1"`` produced a
    variable literally named ``"A`` — which then compared unequal to ``A`` on
    the other side, inventing drift in both directions at once.
    """
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        '[Service]\nEnvironment="A=1" "B=2"\n',
        "[Service]\nEnvironment=A=1\nEnvironment=B=2\n",
    )

    assert drifts == [], f"Quoting must not change what is declared. Got: {drifts}"


def test_environment_multi_assignment_still_detects_a_real_difference():
    """TEETH: the forgiving parse must not forgive an actual value change.

    Without this, the two tests above would be satisfied by a parser that
    returned nothing at all for multi-assignment lines.
    """
    mod = _load_checker()
    spec = _env_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nEnvironment=A=1 B=2\n",
        "[Service]\nEnvironment=A=1 B=99\n",
    )

    assert [d.key for d in drifts] == ["Environment=B"], drifts


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
# compare_unit — intra-copy Environment=/directive agreement
#
# DASHBOARD_PROJECT_ROOT's value embeds the repo root, so it cannot be
# value-compared across copies; the contract is split instead — PRESENCE across
# copies via the Environment= name-set branch above, VALUE within each copy
# against that copy's WorkingDirectory=.  The rationale lives on the checker's
# UnitSpec.env_matches_directive; what follows tests it.
#
# Assertions below are scoped to the drifts the NEW branch emits (keyed
# _RELATION_KEY) rather than to the whole list, so each test measures the branch
# it names and nothing else.  That scoping matters concretely: until
# DASHBOARD_PROJECT_ROOT joins DIVERGENCE_ALLOWLIST, the name-set branch ALSO
# fires on a cross-host value difference, and an unscoped assertion here would
# be measuring the allowlist rather than this relation.  The one test that is
# genuinely about the interaction between the two branches
# (..._missing_variable_is_not_double_reported) asserts on the full list, by
# design.
# ---------------------------------------------------------------------------

# The key _compare_env_matches_directive gives its Drift records. Named rather
# than restated per test so that renaming it fails once, loudly, instead of
# silently reducing every filtered assertion below to an empty-list tautology —
# which is exactly how this whole section could rot into a no-op.
_RELATION_KEY = "Environment=DASHBOARD_PROJECT_ROOT vs WorkingDirectory"


def _relation_drifts(drifts):
    """Return only the drifts emitted by the intra-copy relation branch."""
    return [d for d in drifts if d.key == _RELATION_KEY]


def _env_match_spec(mod: types.ModuleType):
    """A minimal UnitSpec relating DASHBOARD_PROJECT_ROOT to WorkingDirectory=.

    ``environment_section`` is registered as well, mirroring the real
    dark-factory-dashboard.service entry: both halves of the split contract are
    live on the real unit, so a fixture that switched one off would not exercise
    the interaction the double-report test below turns on.
    """
    return mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        environment_section="Service",
        env_matches_directive=(("DASHBOARD_PROJECT_ROOT", "WorkingDirectory"),),
    )


def _root_unit(working_directory: str | None, project_root: str | None) -> str:
    """A synthetic [Service] copy declaring either, both or neither directive.

    Pass None to OMIT that line entirely; pass "" to declare it with an EMPTY
    value, which is what setup-host.sh's sed renders from an unset $REPO_ROOT
    and is a distinct case from omission.  Every fixture in this section is an
    in-memory string: the module docstring's standing rule is that drift-logic
    tests never read the host's real ~/.config/systemd/user/, and it binds
    doubly here — the installed dashboard unit legitimately lacks
    DASHBOARD_PROJECT_ROOT until setup-host.sh is re-run, so a host-touching
    test would encode that transient state as if it were checker behaviour.
    """
    lines = ["[Service]"]
    if working_directory is not None:
        lines.append(f"WorkingDirectory={working_directory}")
    if project_root is not None:
        lines.append(f"Environment=DASHBOARD_PROJECT_ROOT={project_root}")
    return "\n".join(lines) + "\n"


def test_env_matches_directive_agreeing_on_both_copies_is_not_drift():
    """Both copies self-consistent at the same root → the relation holds."""
    mod = _load_checker()
    spec = _env_match_spec(mod)
    root = "/home/leo/src/dark-factory"

    drifts = mod.compare_unit(spec, _root_unit(root, root), _root_unit(root, root))

    assert _relation_drifts(drifts) == [], (
        f"Both copies agree internally at {root}; the relation holds. Got: {drifts}"
    )


def test_env_matches_directive_different_host_paths_are_not_drift():
    """THE CROSS-HOST FALSE-POSITIVE GUARD: each copy self-consistent, at its own root.

    This is the case that decides the SHAPE of the check.  A plain cross-copy
    value compare on DASHBOARD_PROJECT_ROOT reports drift here — on a host that
    is configured perfectly correctly, on every run — and the checker's own
    module docstring spends two paragraphs on where that ends: a gate that is
    always red gets switched off, taking the accidental drift it exists to catch
    with it.  The intra-copy relation is host-invariant by construction, so it
    stays silent here while still having teeth in the two tests below.

    Without this test, a later "simplification" of
    _compare_env_matches_directive into a value compare would look correct and
    pass every other case in this section.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)
    leo = "/home/leo/src/dark-factory"
    alice = "/home/alice/src/dark-factory"

    drifts = mod.compare_unit(spec, _root_unit(leo, leo), _root_unit(alice, alice))

    assert _relation_drifts(drifts) == [], (
        "Each copy is internally consistent; only the HOST differs, which is "
        f"exactly what setup-host.sh renders. Got: {drifts}"
    )


def test_env_matches_directive_mismatch_on_the_installed_copy_is_drift():
    """TEETH: an installed copy whose data root disagrees with its cwd is drift.

    DASHBOARD_PROJECT_ROOT=/tmp/wrong is the failure the allowlist alone would
    wave through, and it is not a cosmetic one: that value IS the dashboard's
    entire data root, so the databases would be read from somewhere other than
    the checkout `uv run --project dashboard` resolves against.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)
    alice = "/home/alice/src/dark-factory"

    drifts = _relation_drifts(
        mod.compare_unit(
            spec,
            _root_unit("/home/leo/src/dark-factory", "/home/leo/src/dark-factory"),
            _root_unit(alice, "/tmp/wrong"),
        )
    )

    assert len(drifts) == 1, f"Expected exactly one relation drift. Got: {drifts}"
    drift = drifts[0]
    # Keyed to the RELATION, not to either directive alone: naming only
    # 'Environment=DASHBOARD_PROJECT_ROOT' would leave the operator to work out
    # what it was supposed to agree with.
    assert drift.key == _RELATION_KEY
    assert drift.unit == "fixture.service"
    assert drift.section == "Service"
    # Both halves as OBSERVED on each side, so the report is readable without a
    # manual diff of the two files.
    assert "/tmp/wrong" in drift.installed_value
    assert alice in drift.installed_value
    assert "installed" in drift.reason, (
        f"The reason must name the side that disagrees. Got: {drift.reason!r}"
    )
    assert "DASHBOARD_PROJECT_ROOT" in drift.reason
    assert "WorkingDirectory" in drift.reason


def test_env_matches_directive_mismatch_on_the_repo_copy_is_drift():
    """The symmetric case: the committed copy is not silently trusted.

    It is the source of truth for VALUES, but not exempt from a relation it is
    itself supposed to satisfy — a committed unit whose two directives disagree
    is a defect that would otherwise be rendered straight onto every host by
    setup-host.sh, with this checker reporting parity on the result.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)
    leo = "/home/leo/src/dark-factory"
    alice = "/home/alice/src/dark-factory"

    drifts = _relation_drifts(
        mod.compare_unit(spec, _root_unit(leo, "/tmp/wrong"), _root_unit(alice, alice))
    )

    assert len(drifts) == 1, f"Expected exactly one relation drift. Got: {drifts}"
    drift = drifts[0]
    assert drift.key == _RELATION_KEY
    assert "/tmp/wrong" in drift.repo_value
    assert leo in drift.repo_value
    assert "repo" in drift.reason, (
        f"The reason must name the side that disagrees. Got: {drift.reason!r}"
    )


def test_env_matches_directive_missing_variable_is_not_double_reported():
    """A copy that omits the variable gets ONE report, from the name-set branch.

    Asserts on the FULL drift list, unlike its neighbours, because the property
    IS the interaction between the two branches.  Absence is already reported by
    _compare_environment with a reason worded for exactly that case; the new
    branch firing as well would print two differently-keyed lines for one
    defect, sending the operator looking for two problems.  The relation branch
    therefore speaks only to the value, and only when there is a value to
    relate.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)
    leo = "/home/leo/src/dark-factory"
    alice = "/home/alice/src/dark-factory"

    drifts = mod.compare_unit(
        spec,
        _root_unit(leo, leo),
        _root_unit(alice, None),  # installed copy never declares the variable
    )

    assert len(drifts) == 1, (
        "A variable absent from the installed copy is ONE defect and must get "
        f"one report line. Got: {drifts}"
    )
    assert drifts[0].key == "Environment=DASHBOARD_PROJECT_ROOT", (
        "The single report must come from the Environment= name-set branch, "
        f"whose wording is written for absence. Got: {drifts[0]}"
    )
    assert _relation_drifts(drifts) == []


def test_env_matches_directive_missing_directive_is_drift():
    """The variable present but WorkingDirectory= absent is drift, not silence.

    The relation cannot be established at all in that copy, and staying quiet
    would read as agreement — the one failure direction this branch cannot
    afford, since it is what the value half of the contract rests on.  (It is
    also not double-reported: WorkingDirectory is on present_only, which fires
    only when the two copies DISAGREE about presence; here both would have to
    lose it for that branch to stay silent, and this fixture is exactly that
    case.)
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)
    leo = "/home/leo/src/dark-factory"

    drifts = _relation_drifts(
        mod.compare_unit(spec, _root_unit(leo, leo), _root_unit(None, leo))
    )

    assert len(drifts) == 1, (
        f"An unestablishable relation must be reported. Got: {drifts}"
    )
    assert drifts[0].key == _RELATION_KEY
    assert "WorkingDirectory" in drifts[0].reason
    assert mod._ABSENT in drifts[0].installed_value, (
        "The absent half must be rendered as _ABSENT so the report shows which "
        f"of the two is missing. Got: {drifts[0].installed_value!r}"
    )


def test_env_matches_directive_mismatch_on_both_copies_names_both_sides():
    """BOTH copies internally inconsistent is ONE drift naming both sides.

    The only path that joins two side names into the reason, and the case a
    per-side Drift record would report twice for what is one contract being
    broken.  Reachable in practice: the committed unit is what setup-host.sh
    renders FROM, so a repo-side repoint is normally reproduced on the installed
    side the moment the host is re-provisioned — the copies are not independent.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)

    drifts = _relation_drifts(
        mod.compare_unit(
            spec,
            _root_unit("/home/leo/src/dark-factory", "/tmp/wrong"),
            _root_unit("/home/alice/src/dark-factory", "/tmp/also-wrong"),
        )
    )

    assert len(drifts) == 1, (
        f"One broken relation is one report line, not two. Got: {drifts}"
    )
    assert "repo and installed" in drifts[0].reason, (
        "Both offending sides must be named — an operator told only 'installed' "
        f"would fix the host and re-render the same defect. Got: {drifts[0].reason!r}"
    )


def test_env_matches_directive_empty_on_both_copies_is_drift():
    """Two EMPTY halves must not pass by comparing equal to each other.

    Equality is not the property being checked, usability is — and this is not
    a theoretical input.  setup-host.sh renders the installed unit with
    `sed 's|__REPO_ROOT__|$REPO_ROOT|g'`, so an unset or empty $REPO_ROOT
    empties WorkingDirectory= and Environment=DASHBOARD_PROJECT_ROOT= in the
    SAME pass.  A bare `directive != env[var]` test then reports parity on a
    unit whose data root is unusable — the gate's single worst outcome, since
    green here is what licenses believing every other line of the report.

    The repo-side helper
    (test_dashboard_service_template.py::_assert_project_root_env_matches_working_directory)
    asserts non-empty for exactly this reason; the checker is the half that sees
    RENDERED output, so it needs the rule more, not less.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)

    drifts = _relation_drifts(mod.compare_unit(spec, _root_unit("", ""), _root_unit("", "")))

    assert len(drifts) == 1, (
        f"Empty-vs-empty is drift, not parity. Got: {drifts}"
    )
    assert "EMPTY" in drifts[0].reason, (
        "The reason must say the value is empty rather than merely mismatched — "
        f"the two send an operator to different places. Got: {drifts[0].reason!r}"
    )
    assert "repo and installed" in drifts[0].reason, (
        f"Both empty sides must be named. Got: {drifts[0].reason!r}"
    )


def test_env_matches_directive_absent_from_both_copies_is_silent_here():
    """Neither branch fires when NO copy declares the variable — by design.

    compare_unit reports parity for this input, and that is correct at its
    altitude: it compares two copies, and two copies that agree about declaring
    nothing have not DRIFTED.  Recording the layering explicitly because the
    silence looks like a gap: what catches a variable the registry names and no
    unit declares is
    test_registry_env_matches_directive_entries_are_declared_in_the_committed_units,
    a staleness guard on the REGISTRY, not a drift check.  Deleting that guard
    would leave this case unowned, and this test is where that shows up.
    """
    mod = _load_checker()
    spec = _env_match_spec(mod)

    drifts = mod.compare_unit(
        spec,
        _root_unit("/home/leo/src/dark-factory", None),
        _root_unit("/home/alice/src/dark-factory", None),
    )

    assert drifts == [], (
        "Two copies that both omit the variable have not drifted from each "
        f"other; the registry staleness guard owns this case. Got: {drifts}"
    )


def test_env_matches_directive_without_an_environment_section_is_rejected():
    """The combination that would silently check nothing raises at construction.

    Both halves of the pair are read out of ``environment_section``, so a spec
    registering a pair without it runs the branch over an empty env map and
    reports parity forever while claiming the value is checked.  That is
    unobservable from the report — the run is green and says so — which is why
    it is rejected at import time on the registry rather than left to a test to
    notice later.
    """
    mod = _load_checker()

    with pytest.raises(ValueError) as excinfo:
        mod.UnitSpec(
            name="fixture.service",
            repo_relpath="dashboard/fixture.service",
            environment_section=None,
            env_matches_directive=(("DASHBOARD_PROJECT_ROOT", "WorkingDirectory"),),
        )

    assert "environment_section" in str(excinfo.value), (
        f"The error must name the field to set. Got: {excinfo.value!r}"
    )


def test_env_matches_directive_not_compared_when_spec_omits_it():
    """A spec with no env_matches_directive runs the branch not at all.

    Mirrors test_environment_not_compared_when_spec_omits_the_section above:
    the two watchdog units declare no Environment= at all, so the branch must
    stay off for them rather than inventing comparisons.
    """
    mod = _load_checker()
    spec = mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        environment_section="Service",
    )

    drifts = _relation_drifts(
        mod.compare_unit(
            spec,
            _root_unit("/home/leo/src/dark-factory", "/tmp/wrong"),
            _root_unit("/home/alice/src/dark-factory", "/tmp/also-wrong"),
        )
    )

    assert drifts == [], (
        f"env_matches_directive is empty; the branch must not run. Got: {drifts}"
    )


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


def test_registry_env_matches_directive_entries_are_declared_in_the_committed_units():
    """STALENESS GUARD, intra-copy-relation edition.

    Both halves of each registered pair must really exist in the committed unit.
    A misspelled variable or directive name would compare absent-to-absent
    forever — or, worse, be skipped entirely by the absent-variable branch —
    while the gate kept reporting green, which is the invisible-by-construction
    failure the sibling guards above were written for.

    Also asserts environment_section is set, since the variable is read through
    it: a pair registered on a spec with no section could never resolve.
    """
    mod = _load_checker()

    for name, spec in mod.UNITS.items():
        if not spec.env_matches_directive:
            continue
        assert spec.environment_section is not None, (
            f"{name}: registry declares env_matches_directive but no "
            "environment_section, so the variable can never be read."
        )
        parsed = mod.parse_unit_directives(
            (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
        )
        env = mod._environment_map(parsed, spec.environment_section)
        for var, key in spec.env_matches_directive:
            assert var in env, (
                f"{name}: registry relates Environment={var} to {key}, but the "
                f"committed unit {spec.repo_relpath} declares no such variable "
                "— the entry checks nothing. Fix the registry or the unit."
            )
            assert key in parsed.get(spec.environment_section, {}), (
                f"{name}: registry relates Environment={var} to "
                f"[{spec.environment_section}] {key}, but the committed unit "
                f"{spec.repo_relpath} does not declare that directive."
            )


def test_divergence_allowlist_names_are_declared_in_a_committed_unit():
    """STALENESS GUARD, allowlist edition: no waiver for a variable nobody sets.

    DIVERGENCE_ALLOWLIST is the one deliberate hole in this gate, and its own
    comment says to keep it small and every reason checkable.  An entry naming a
    variable no committed unit declares any more is worse than useless: it is a
    standing, invisible waiver that nothing will ever remove, because nothing
    reports it.  If the variable comes back later — reused for something else —
    the waiver is already there, blessing a value comparison nobody re-examined.
    """
    mod = _load_checker()

    declared: set[str] = set()
    for spec in mod.UNITS.values():
        if spec.environment_section is None:
            continue
        parsed = mod.parse_unit_directives(
            (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
        )
        declared |= set(mod._environment_map(parsed, spec.environment_section))

    for var, reason in mod.DIVERGENCE_ALLOWLIST.items():
        assert var in declared, (
            f"DIVERGENCE_ALLOWLIST waives value comparison for {var}, but no "
            f"committed unit in the registry declares it. Declared: "
            f"{sorted(declared)}. Remove the entry or restore the variable."
        )
        assert reason.strip(), (
            f"DIVERGENCE_ALLOWLIST entry {var} has an empty reason. Every hole "
            "in this gate must state one specific enough for a reviewer to check."
        )


def test_dashboard_service_spec_pins_the_tasks_minimum_coverage():
    """The dashboard service compares the restart directives and 3306's flags."""
    mod = _load_checker()
    spec = mod.UNITS[_DASHBOARD_SERVICE]

    compared_keys = {key for _section, key in spec.compared}
    assert {"Restart", "RestartSec", "TimeoutStopSec"} <= compared_keys, compared_keys
    assert {"timeout-graceful-shutdown", "timeout-keep-alive"} <= set(spec.exec_start_flags)

    # The cap and the directive that makes it effective must be compared
    # TOGETHER. systemd discards RestartMaxDelaySec= outright on a unit with no
    # RestartSteps=, so an installed copy missing only that line matches every
    # other compared key and reports parity while running with no growing
    # backoff at all. The repo-file sweep in test_systemd_restart_backoff.py
    # cannot see this: it reads committed files, never the installed unit.
    if "RestartMaxDelaySec" in compared_keys:
        assert "RestartSteps" in compared_keys, (
            "the dashboard spec compares RestartMaxDelaySec= but not "
            "RestartSteps=, so an installed unit whose cap systemd is silently "
            f"ignoring would be reported as parity. compared: {compared_keys}"
        )


def test_dashboard_service_spec_pins_the_project_root_env_contract():
    """The dashboard service relates DASHBOARD_PROJECT_ROOT to WorkingDirectory=.

    This is the VALUE half of the contract task 3572 made explicit.  Without the
    registry entry the variable is PRESENCE-checked only — the name-set branch
    would still catch it disappearing, but ``DASHBOARD_PROJECT_ROOT=/tmp/wrong``
    on the installed copy would be reported as parity.  That value is the
    dashboard's entire data root (config.py's project_root falls back to
    Path.cwd() when it is unset), so "present, and pointing anywhere at all"
    is not the claim this gate should be making about it.
    """
    mod = _load_checker()
    spec = mod.UNITS[_DASHBOARD_SERVICE]

    assert ("DASHBOARD_PROJECT_ROOT", "WorkingDirectory") in spec.env_matches_directive, (
        "the dashboard spec does not relate DASHBOARD_PROJECT_ROOT to "
        "WorkingDirectory=, so its value is unchecked. "
        f"env_matches_directive: {spec.env_matches_directive}"
    )


def test_project_root_is_allowlisted_for_cross_host_value_divergence():
    """...and it is allowlisted out of the CROSS-COPY value comparison.

    The pairing is deliberate, and the two entries must land together.  The
    allowlist waives a comparison that would otherwise fire on every host whose
    checkout is not /home/leo/src/dark-factory: the committed unit hardcodes
    that path while setup-host.sh renders the installed copy from
    scripts/dashboard.service.template with the host's real $REPO_ROOT.

    Unpaired, either entry alone is a defect.  The allowlist alone would leave
    the data root genuinely unchecked — the hole its own preamble warns about.
    The relation alone would leave the gate permanently red off this host, which
    the module docstring records as how a gate gets switched off entirely.
    Together, presence is checked cross-copy, value is checked intra-copy, and
    nothing is waived that is not checked another way — so this is the one
    allowlist entry that is not a hole.
    """
    mod = _load_checker()

    assert "DASHBOARD_PROJECT_ROOT" in mod.DIVERGENCE_ALLOWLIST, (
        "DASHBOARD_PROJECT_ROOT is not allowlisted, so its value is compared "
        "across copies — that fires on every correctly-configured host whose "
        "repo does not live at /home/leo/src/dark-factory."
    )
    assert mod.DIVERGENCE_ALLOWLIST["DASHBOARD_PROJECT_ROOT"].strip(), (
        "every allowlist entry must state its reason; this one must also name "
        "what closes the hole (env_matches_directive), so the preamble's "
        "'THIS IS A HOLE IN THE GATE' stays honest."
    )


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
# Third-site lockstep: the templated dashboard unit
# ---------------------------------------------------------------------------
#
# setup-host.sh does NOT cp dark-factory-dashboard.service into
# ~/.config/systemd/user/. It RENDERS it from scripts/dashboard.service.template
# (scripts/setup-host.sh:362-367, substituting __REPO_ROOT__ and __UV_PATH__),
# and only cp's the two watchdog units verbatim. So the registry's repo_relpath
# for this one unit names a file that is NOT the source of the copy being
# checked: a third site that must agree.
#
# That pair is ALREADY guarded, and more strongly than anything this module
# could add:
#
#     tests/scripts/test_dashboard_service_template.py::
#         test_template_renders_to_hardcoded_file
#
# renders the template with setup-host.sh's own substitutions and asserts
# BYTE-FOR-BYTE equality with dashboard/dark-factory-dashboard.service. A
# registry-scoped version of that check would compare only the curated keys, so
# it would MISS a divergence in any directive nobody thought to register --
# strictly weaker, and a second guard on the same invariant that fails second.
# The checker's module docstring and its UNITS entry both point at the
# byte-for-byte test instead.

# ---------------------------------------------------------------------------
# Override mechanisms: the unit file is not necessarily what takes effect
# ---------------------------------------------------------------------------
#
# Comparing directives only proves something if the installed unit FILE is what
# systemd actually runs. Two standard mechanisms make that false without
# touching a single compared byte:
#
#   EnvironmentFile=  — pulls values from a file off this tree
#   <unit>.d/*.conf   — merged OVER the unit at load time; what
#                       `systemctl --user edit` writes
#
# Neither is present for the dashboard units today, so neither can misfire; the
# tests below are what keeps the blind spot closed. Drop-ins are NOT
# hypothetical in this environment — ~/.config/systemd/user/
# orchestrator-reify.service.d/ exists, so the mechanism is already in live use
# on this host, just not on these units.


def _override_spec(mod: types.ModuleType):
    """A minimal UnitSpec registering EnvironmentFile= as an override directive."""
    return mod.UnitSpec(
        name="fixture.service",
        repo_relpath="dashboard/fixture.service",
        compared=(("Service", "Type"),),
        override_directives=(("Service", "EnvironmentFile"),),
    )


def test_environment_file_added_only_to_the_installed_copy_is_drift():
    """The reviewer's measured hole: every compared directive matches, yet drift.

    An EnvironmentFile= on the installed side alone can set anything at all
    from a path off this tree. Before it was registered, this exact input
    produced zero drifts and exit 0 — "parity" over a unit whose effective
    configuration was unknown.
    """
    mod = _load_checker()
    spec = _override_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nType=simple\n",
        "[Service]\nType=simple\nEnvironmentFile=/tmp/evil.env\n",
    )

    assert [d.key for d in drifts] == ["EnvironmentFile"], drifts
    assert drifts[0].installed_value == "/tmp/evil.env"
    assert drifts[0].repo_value == mod._ABSENT


def test_environment_file_on_both_copies_is_not_drift():
    """Registered as presence-SYMMETRIC, so a legitimately adopted one agrees.

    This is why the field needs no staleness guard of its own: if the committed
    unit ever gains an EnvironmentFile, the installed copy gains it too and the
    pair simply matches.
    """
    mod = _load_checker()
    spec = _override_spec(mod)

    unit = "[Service]\nType=simple\nEnvironmentFile=/etc/dashboard.env\n"

    assert mod.compare_unit(spec, unit, unit) == []


def test_environment_file_dropped_from_the_installed_copy_is_drift():
    """Symmetric in the other direction: a committed one that never landed."""
    mod = _load_checker()
    spec = _override_spec(mod)

    drifts = mod.compare_unit(
        spec,
        "[Service]\nType=simple\nEnvironmentFile=/etc/dashboard.env\n",
        "[Service]\nType=simple\n",
    )

    assert [d.key for d in drifts] == ["EnvironmentFile"], drifts


def test_registry_registers_environment_file_on_both_service_units():
    """The two service units must actually carry the override registration.

    A helper-only test would pass while the real registry left the hole open —
    the same rot the key-staleness tests exist to prevent one level down.
    """
    mod = _load_checker()

    for name in (_DASHBOARD_SERVICE, _WATCHDOG_SERVICE):
        assert ("Service", "EnvironmentFile") in mod.UNITS[name].override_directives, (
            f"{name} does not register EnvironmentFile=, so one added locally "
            "would leave the checker reporting parity over an unknown "
            "effective configuration."
        )


def test_watchdog_environment_injected_on_the_installed_copy_is_drift():
    """The measured hole: an inline Environment= on the installed copy alone.

    scripts/dashboard-watchdog.py reads nine env knobs — PROBE_URL,
    PROBE_TIMEOUT, GRACE_SECS, FAIL_STREAK, MAX_RESTARTS, RATE_WINDOW_SECS,
    STATE_PATH, ESCALATION_QUEUE_DIR, UV_BIN — and those knobs ARE the
    hysteresis/grace/rate-ceiling supervision policy. An installed copy that
    picked up ``Environment=DASHBOARD_WATCHDOG_FAIL_STREAK=99`` outside the
    repo (``systemctl --user edit`` writing a bare Environment= line, or a
    hand-edit of the installed file) means ~99 consecutive failed probes
    before any restart — supervision effectively off. Measured: with
    ``environment_section`` unset on the watchdog spec, this exact input
    produces zero drift lines — ``compare_unit`` returns ``[]``, a clean
    "[ok] parity" over a unit whose effective failure tolerance was silently
    disabled.

    Uses the REAL registry spec and the REAL committed unit, not a fixture
    UnitSpec — a fixture-only test would pass while the registry left the hole
    open, the same reason
    test_registry_registers_environment_file_on_both_service_units states in
    its own docstring. The installed text is the repo text plus one injected
    line, so this is a REPO-side-only read and stays green on CI and on a host
    whose installed units are drifted.
    """
    mod = _load_checker()
    spec = mod.UNITS[_WATCHDOG_SERVICE]
    repo_text = (REPO_ROOT / spec.repo_relpath).read_text(encoding="utf-8")
    installed_text = repo_text + "Environment=DASHBOARD_WATCHDOG_FAIL_STREAK=99\n"

    drifts = mod.compare_unit(spec, repo_text, installed_text)

    assert [d.key for d in drifts] == ["Environment=DASHBOARD_WATCHDOG_FAIL_STREAK"], drifts
    assert drifts[0].section == "Service"
    assert drifts[0].installed_value == "99"
    assert drifts[0].repo_value == mod._ABSENT
    assert "installed copy" in drifts[0].reason


def test_registry_environment_section_registered_on_both_service_units():
    """Registry pin mirroring the EnvironmentFile one, so a revert is loud.

    Without this, an edit that removes ``environment_section`` from the
    watchdog spec would silently reopen the hole
    test_watchdog_environment_injected_on_the_installed_copy_is_drift
    measures, with no assertion naming the registry entry itself — the same
    "helper passes, registry regresses" failure mode
    test_registry_registers_environment_file_on_both_service_units guards
    against for override_directives.

    The timer is pinned the OTHER way, and the message states why: the
    committed dark-factory-dashboard-watchdog.timer parses to sections Unit /
    Timer / Install — there is no [Service] section, and Environment= is not a
    valid directive in a timer unit. Registering a section there would compare
    absent-to-absent forever, which is exactly the rot the staleness guards
    elsewhere in this module exist to prevent.
    """
    mod = _load_checker()

    for name in (_DASHBOARD_SERVICE, _WATCHDOG_SERVICE):
        assert mod.UNITS[name].environment_section == "Service", (
            f"{name} does not register environment_section='Service', so an "
            "Environment= line added to the installed copy alone would leave "
            "the checker reporting parity over an unknown effective "
            "configuration."
        )

    assert mod.UNITS[_WATCHDOG_TIMER].environment_section is None, (
        f"{_WATCHDOG_TIMER} has no [Service] section (it parses to Unit / "
        "Timer / Install) and Environment= is not valid in a timer unit — "
        "registering environment_section here would compare absent-to-absent "
        "forever."
    )


def test_find_dropins_returns_nothing_when_no_dropin_dir_exists(tmp_path: pathlib.Path):
    """The overwhelmingly common case must be silent."""
    mod = _load_checker()
    installed = tmp_path / "installed"
    installed.mkdir()
    (installed / _DASHBOARD_SERVICE).write_text("[Service]\n", encoding="utf-8")

    assert mod.find_dropins(installed, _DASHBOARD_SERVICE) == []


def test_find_dropins_reports_conf_files_and_ignores_the_rest(tmp_path: pathlib.Path):
    """Counts exactly what systemd would load: *.conf, nothing else.

    systemd ignores non-.conf files in a drop-in directory, so reporting them
    would raise an alarm over a stray editor backup — a false positive on the
    one gate whose value is being believed when it fires.
    """
    mod = _load_checker()
    installed = tmp_path / "installed"
    installed.mkdir()
    dropin_dir = installed / f"{_DASHBOARD_SERVICE}.d"
    dropin_dir.mkdir()
    (dropin_dir / "override.conf").write_text("[Service]\nRestart=always\n")
    (dropin_dir / "10-limits.conf").write_text("[Service]\nTimeoutStopSec=90\n")
    (dropin_dir / "override.conf.bak").write_text("ignored\n")

    found = mod.find_dropins(installed, _DASHBOARD_SERVICE)

    assert [p.name for p in found] == ["10-limits.conf", "override.conf"], found


def test_main_reports_a_dropin_and_refuses_to_call_it_parity(
    tmp_path: pathlib.Path, capsys
):
    """Unit files identical + a drop-in present → exit 1, not 0.

    The drop-in sets TimeoutStopSec=90 over a committed 15. Every compared
    directive still matches character for character, because the drop-in is a
    separate file systemd merges at load time — so before this, the run
    reported "[ok] parity" over a unit whose effective shutdown bound was six
    times the committed one.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)
    dropin_dir = installed / f"{_DASHBOARD_SERVICE}.d"
    dropin_dir.mkdir()
    (dropin_dir / "override.conf").write_text(
        "[Service]\nTimeoutStopSec=90\n", encoding="utf-8"
    )

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 1, f"A drop-in leaves the effective config unverified. Got {rc}:\n{out}"
    assert "[override]" in out, out
    assert str(dropin_dir / "override.conf") in out, (
        f"The report must name the drop-in file, or the operator cannot find "
        f"it. Got:\n{out}"
    )
    assert "[ok] parity" not in out, (
        f"A run that could not verify the effective config must not claim "
        f"parity. Got:\n{out}"
    )


def test_main_dropin_report_is_worded_apart_from_drift(tmp_path: pathlib.Path, capsys):
    """Same exit code as drift, DIFFERENT wording — they need different actions.

    A drift is a directive diff to propagate with setup-host.sh; a drop-in is a
    separate file to inspect or remove. Collapsing them would send the operator
    hunting for a diff that does not exist, which is the same care the
    [vanished] block takes.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)
    dropin_dir = installed / f"{_WATCHDOG_TIMER}.d"
    dropin_dir.mkdir()
    (dropin_dir / "override.conf").write_text(
        "[Timer]\nOnUnitActiveSec=600\n", encoding="utf-8"
    )

    mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert "[drift]" not in out, (
        f"No compared directive differs; only a drop-in exists. Got:\n{out}"
    )
    assert "systemctl --user cat" in out, (
        f"The override block must name how to inspect the merged unit. Got:\n{out}"
    )


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


# ---------------------------------------------------------------------------
# The vanished committed unit  (step-19 / step-20)
# ---------------------------------------------------------------------------
#
# The committed unit is this checker's source of truth. When it is absent the
# run verified NOTHING for that unit, so "parity" is the one verdict that must
# be impossible. Reproduced against the checker as it stood before these tests:
#
#     $ python3 scripts/check_dashboard_unit_parity.py \
#           --repo-root /tmp/emptyrepo --installed-dir /tmp/emptyrepo
#     [dashboard_unit_parity] [skip] dark-factory-dashboard.service: committed
#         unit not found at /tmp/emptyrepo/dashboard/dark-factory-dashboard.service
#     ... (x3)
#     [dashboard_unit_parity] [ok] parity — 3 unit(s) match their committed copies.
#     $ echo $?
#     0
#
# Zero units compared, a count of three reported, exit 0 — so setup-host.sh
# printed `ok`. A typo'd --repo-root, a renamed unit file, or a `git mv` of
# dashboard/*.service silently disarmed the entire gate. That is the same
# failure class the checker itself exists to expose: green while verifying
# nothing.
#
# A vanished committed unit reuses exit 1 rather than minting a new code:
# setup-host.sh already branches on the 0/1/2 vocabulary and reads 2 as the
# benign "not installed on this host". A missing source of truth is the
# opposite of benign, so it belongs with drift.


def test_main_no_committed_units_is_not_parity(tmp_path: pathlib.Path, capsys):
    """A --repo-root holding no committed units must not report parity.

    The typo'd-path case, reproduced: the installed copies are all present and
    correct, but the tree we are comparing them AGAINST is empty.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)
    typo_root = tmp_path / "typo-repo-root"
    typo_root.mkdir()

    rc = mod.main(["--repo-root", str(typo_root), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc != 0, (
        "A run that compared zero units has proved nothing and must never "
        f"report parity. Got {rc}:\n{out}"
    )


def test_main_success_line_counts_only_units_actually_compared(
    tmp_path: pathlib.Path, capsys
):
    """COUNT HONESTY: the success line names what was compared, not what was selected.

    Only the positive direction is asserted, and deliberately so. The obvious
    negative half — unlink a committed unit, then assert the count is no longer
    three — is VACUOUS: that run takes the `vanished` path and returns 1
    without ever reaching the success line, so "3 unit(s)" is absent for a
    reason having nothing to do with counting. What the partial run actually
    prints is asserted by test_main_names_the_vanished_committed_unit below.

    On the success path `len(compared) == len(selected)` always holds: a
    vanished committed unit returns 1 and an absent installed unit returns 2,
    both before the success line. The count is therefore honest by
    construction here, and this test pins the line that reports it rather than
    a divergence no input can produce.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)

    assert mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)]) == 0
    full_out = capsys.readouterr().out
    assert "3 unit(s)" in full_out, (
        f"With all three compared, the count must say so. Got:\n{full_out}"
    )

    # The count tracks the run, not a constant: restricting to one unit must
    # say one. This is the assertion the vacuous negative half was reaching
    # for, on a path that actually reaches the success line.
    assert (
        mod.main(
            [
                "--repo-root",
                str(repo),
                "--installed-dir",
                str(installed),
                "--unit",
                _WATCHDOG_TIMER,
            ]
        )
        == 0
    )
    single_out = capsys.readouterr().out
    assert "1 unit(s)" in single_out, (
        f"One unit was compared; the line must say one. Got:\n{single_out}"
    )
    assert "3 unit(s)" not in single_out, (
        "Claiming three when one was compared overstates what the gate "
        f"verified. Got:\n{single_out}"
    )


def test_main_names_the_vanished_committed_unit(tmp_path: pathlib.Path, capsys):
    """The report names the vanished committed path, distinctly from a drift.

    An operator who reads setup-host.sh's warn line must not be sent hunting
    for a directive diff that does not exist.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)
    vanished = repo / mod.UNITS[_WATCHDOG_TIMER].repo_relpath
    vanished.unlink()

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert str(vanished) in out, (
        f"The report must name the committed path that vanished. Got:\n{out}"
    )
    assert _WATCHDOG_TIMER in out
    assert "directive(s) differ" not in out, (
        "Nothing drifted — the two other units are at parity. Reporting a "
        f"directive diff would misdirect the operator. Got:\n{out}"
    )
    assert "[ok] parity" not in out, (
        "The other two units DID match, but the run could not verify the "
        "third, so no parity line may be printed at all — a partial success "
        f"line is what let a zero-unit run report '3 unit(s) match'. Got:\n{out}"
    )


def test_main_vanished_committed_unit_does_not_mask_drift(
    tmp_path: pathlib.Path, capsys
):
    """PRECEDENCE: a vanished committed unit AND a drifting one → still 1, both reported.

    Both findings are actionable and neither may mask the other.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )
    vanished = repo / mod.UNITS[_WATCHDOG_TIMER].repo_relpath
    vanished.unlink()

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "TimeoutStopSec" in out, f"The drift must still be reported:\n{out}"
    assert str(vanished) in out, f"The vanished unit must still be reported:\n{out}"


def test_main_empty_installed_dir_still_returns_exactly_2(
    tmp_path: pathlib.Path, capsys
):
    """REGRESSION FLOOR: committed units present, nothing installed → still exactly 2.

    setup-host.sh reads 2 as "not installed on this host, skipping" — a benign
    state on any machine that does not run the dashboard. The zero-compared
    guard must not upgrade it to an alarm.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = tmp_path / "installed-empty"
    installed.mkdir()

    rc = mod.main(["--repo-root", str(repo), "--installed-dir", str(installed)])
    out = capsys.readouterr().out

    assert rc == 2, f"Expected exactly 2 (benign, not installed here). Got {rc}:\n{out}"


# ---------------------------------------------------------------------------
# CLI subprocess boundary  (step-15 / step-16)
# ---------------------------------------------------------------------------


def _run_checker(repo: pathlib.Path, installed: pathlib.Path, *extra: str):
    """Invoke the checker as a real subprocess, as setup-host.sh does."""
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER_PATH),
            "--repo-root",
            str(repo),
            "--installed-dir",
            str(installed),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def test_checker_subprocess_exit_0_on_parity(tmp_path: pathlib.Path):
    """The standalone CLI exits 0 on parity.

    Asserts on real exit codes and real stdout rather than in-process return
    values, because the subprocess IS the interface setup-host.sh and the
    operator use. Mirrors test_parity_checker_callable_as_subprocess in the
    fused-memory test module.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)

    result = _run_checker(repo, installed)

    assert result.returncode == 0, (
        f"Expected 0; got {result.returncode}.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_checker_subprocess_exit_1_on_the_measured_watchdog_drift(tmp_path: pathlib.Path):
    """The real host drift, through the real CLI → exit 1, reported on STDOUT.

    setup-host.sh branches on this exit code, and an operator reads this
    stdout; both are asserted here rather than assumed.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo, omit=(_WATCHDOG_SERVICE,))
    installed_watchdog = installed / _WATCHDOG_SERVICE
    installed_watchdog.write_text(
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

    result = _run_checker(repo, installed)

    assert result.returncode == 1, (
        f"Expected 1; got {result.returncode}.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "TimeoutStartSec" in result.stdout
    assert _WATCHDOG_SERVICE in result.stdout
    assert str(installed_watchdog) in result.stdout


def test_checker_subprocess_exit_2_on_empty_installed_dir(tmp_path: pathlib.Path):
    """No installed units at all → exit 2 (setup-host.sh's benign skip)."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    empty = tmp_path / "empty"
    empty.mkdir()

    result = _run_checker(repo, empty)

    assert result.returncode == 2, (
        f"Expected 2; got {result.returncode}.\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_checker_subprocess_runs_without_third_party_imports(tmp_path: pathlib.Path):
    """The checker runs under a bare `python3` with an empty PYTHONPATH.

    setup-host.sh invokes it as plain `python3`, on a host that may not have
    the project venv active — a third-party import would make it die there
    while passing in this suite's environment.
    """
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(tmp_path, mod, repo)

    result = subprocess.run(
        [
            "python3",
            str(CHECKER_PATH),
            "--repo-root",
            str(repo),
            "--installed-dir",
            str(installed),
        ],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
    )

    assert result.returncode == 0, (
        f"Bare python3 run failed ({result.returncode}).\nstdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )


def test_checker_subprocess_unit_flag_narrows_the_run(tmp_path: pathlib.Path):
    """--unit works across the CLI boundary too."""
    mod = _load_checker()
    repo = _fake_repo(tmp_path, mod)
    installed = _installed_from(
        tmp_path,
        mod,
        repo,
        edits={_DASHBOARD_SERVICE: ("TimeoutStopSec=15", "TimeoutStopSec=30")},
    )

    narrowed = _run_checker(repo, installed, "--unit", _WATCHDOG_TIMER)
    assert narrowed.returncode == 0, narrowed.stdout

    full = _run_checker(repo, installed)
    assert full.returncode == 1, full.stdout
