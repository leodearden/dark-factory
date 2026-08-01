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
