"""The run_parity driver contract, read at its own seam.

Every run here uses real tmp_path files, an ``io.StringIO`` sink, a trivial
``compare`` (identical texts match; anything else is one Drift) and report
wording made of distinctive markers, so each assertion reads the DRIVER's
layout and ladder rather than any checker's prose. The checkers' own output is
pinned separately, byte for byte, by test_unit_parity_cli_contract.py.
"""

import io
import os
import pathlib

import pytest
import systemd_unit_parity as parity

TOOL = "t"
VERDICT = "[t] verdict "
UNDECODABLE = b"[Service]\nType=\xff\xfe\n"


def _report(**overrides) -> "parity.ReportText":
    fields = {
        "drift": parity.BlockText("drift", "{count} DRIFT-HEADER:", ("DRIFT-TAIL",)),
        "skip": "SKIP {name} at {path}",
        "vanished": parity.BlockText("vanished", "{count} VANISHED-HEADER:", ("VANISHED-TAIL",)),
        "override": parity.BlockText("override", "{count} OVERRIDE-HEADER:", ("OVERRIDE-TAIL",)),
    }
    fields.update(overrides)
    return parity.ReportText(**fields)


def _write(path: pathlib.Path, content: str | bytes | None) -> None:
    if content is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _unit(
    tmp_path: pathlib.Path,
    name: str,
    *,
    repo: str | bytes | None = "A\n",
    installed: str | bytes | None = "A\n",
) -> "parity.UnitRef":
    """A unit whose committed copy and installed copy hold the given content (None: absent)."""
    unit = parity.UnitRef(name, tmp_path / "repo" / name, tmp_path / "installed" / name)
    _write(unit.repo_path, repo)
    _write(unit.installed_path, installed)
    return unit


def _dropin(unit: "parity.UnitRef", filename: str = "override.conf") -> pathlib.Path:
    path = unit.installed_path.parent / f"{unit.name}.d" / filename
    _write(path, "[Service]\nRestartSec=1\n")
    return path


def _compare(unit: "parity.UnitRef", repo_text: str, installed_text: str) -> list:
    if repo_text == installed_text:
        return []
    return [
        parity.Drift(
            unit.name, "Service", "Type", repo_text.strip(), installed_text.strip(), "REASON"
        )
    ]


def _never_compare(unit, repo_text, installed_text):
    raise AssertionError(f"compare must not run for {unit.name}")


def _run(units, *, compare=_compare, report=None, probes=(), print_verdicts=False):
    out = io.StringIO()
    rc = parity.run_parity(
        parity.ParityRun(
            tool=TOOL,
            units=tuple(units),
            compare=compare,
            report=report if report is not None else _report(),
            probes=tuple(probes),
            print_verdicts=print_verdicts,
        ),
        out=out,
    )
    return rc, out.getvalue().splitlines()


def _probe(outcome_for, blocks):
    """A Probe whose run is ``outcome_for(unit, committed_text)``, recording every call."""
    calls: list[tuple[str, str]] = []

    def run(unit, committed_text):
        calls.append((unit.name, committed_text))
        return outcome_for(unit, committed_text)

    return parity.Probe(blocks=tuple(blocks), run=run), calls


def _drift_lines(unit: "parity.UnitRef", repo_label: str = "repo") -> list[str]:
    return [
        f"[t]   {unit.name} [Service] Type",
        "[t]       REASON",
        f"[t]       {repo_label.ljust(9)} {unit.repo_path}: A",
        f"[t]       installed {unit.installed_path}: B",
    ]


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_exit_codes_are_the_frozen_ladder():
    assert [int(code) for code in parity.ParityExit] == [0, 1, 2]
    assert parity.ParityExit.PARITY == 0
    assert parity.ParityExit.FINDING == 1
    assert parity.ParityExit.NOT_INSTALLED == 2


def test_identical_units_are_parity_and_say_how_many_were_compared(tmp_path):
    rc, lines = _run([_unit(tmp_path, "a"), _unit(tmp_path, "b")])

    assert rc == 0
    assert lines == ["[t] [ok] parity — 2 unit(s) match their committed copies."]


def test_a_unit_not_installed_is_skipped_with_the_report_skip_wording(tmp_path):
    unit = _unit(tmp_path, "a", installed=None)

    rc, lines = _run([unit])

    assert rc == 2
    assert lines == [f"[t] [skip] SKIP a at {unit.installed_path}"]


def test_drift_dominates_absence_and_both_are_reported(tmp_path):
    drifted = _unit(tmp_path, "a", installed="B\n")
    absent = _unit(tmp_path, "b", installed=None)

    rc, lines = _run([drifted, absent])

    assert rc == 1
    assert lines == [
        f"[t] [skip] SKIP b at {absent.installed_path}",
        "[t] [drift] 1 DRIFT-HEADER:",
        *_drift_lines(drifted),
        "[t] [drift] DRIFT-TAIL",
    ]


def test_a_vanished_committed_copy_is_a_finding_never_parity(tmp_path):
    unit = _unit(tmp_path, "a", repo=None)
    report = parity.ReportText(drift=_report().drift)

    rc, lines = _run([unit], report=report)

    assert rc == 1
    assert lines == [
        "[t] [vanished] 1 committed unit(s) not found — nothing was verified for them:",
        f"[t]   a: expected committed copy at {unit.repo_path}",
        *[f"[t] [vanished] {line}" for line in report.vanished.tail],
    ]


def test_a_run_that_compared_nothing_never_reports_parity():
    rc, lines = _run(())

    assert rc == 1
    assert lines == ["[t] [error] no units were compared — nothing was verified."]


def test_a_dropin_over_identical_copies_is_an_override_finding(tmp_path):
    unit = _unit(tmp_path, "a")
    dropin = _dropin(unit)

    rc, lines = _run([unit])

    assert rc == 1
    assert lines == [
        "[t] [override] 1 OVERRIDE-HEADER:",
        f"[t]   a: {dropin}",
        "[t] [override] OVERRIDE-TAIL",
    ]


@pytest.mark.parametrize("side", ["installed", "repo"])
def test_an_undecodable_unit_is_reported_unreadable_naming_the_side_that_failed(
    tmp_path, side
):
    unit = _unit(tmp_path, "a", **{side: UNDECODABLE})
    failed = unit.installed_path if side == "installed" else unit.repo_path

    rc, lines = _run([unit], compare=_never_compare)

    assert rc == 1
    assert lines[0] == (
        "[t] [unreadable] 1 unit file(s) could not be read — nothing was verified for them:"
    )
    assert lines[1].startswith(f"[t]   a: {failed}: ")
    assert "can't decode" in lines[1]
    assert all(line.startswith("[t] [unreadable] ") for line in lines[2:])
    assert not any("[ok]" in line for line in lines)


# ---------------------------------------------------------------------------
# Report layout
# ---------------------------------------------------------------------------


def test_blocks_print_in_the_fixed_order_whatever_the_unit_order(tmp_path):
    finding_block = parity.BlockText("probed", "{count} PROBE-HEADER:", ("PROBE-TAIL",))
    probe, _ = _probe(
        lambda unit, text: parity.ProbeOutcome(
            findings=(parity.ProbeFinding(finding_block, (f"  {unit.name}",)),)
            if unit.name == "f"
            else ()
        ),
        [finding_block],
    )
    units = [
        _unit(tmp_path, "f"),
        _unit(tmp_path, "e", installed="B\n"),
        _unit(tmp_path, "d"),
        _unit(tmp_path, "c", installed=UNDECODABLE),
        _unit(tmp_path, "b", repo=None),
        _unit(tmp_path, "a", installed=None),
    ]
    _dropin(units[2])

    rc, lines = _run(units, probes=[probe])

    tags = ["[skip]", "[vanished]", "[unreadable]", "[override]", "[drift]", "[probed]"]
    first = [next(i for i, line in enumerate(lines) if line.startswith(f"[t] {tag}")) for tag in tags]
    assert rc == 1
    assert first == sorted(first)


@pytest.mark.parametrize("repo_label", ["repo", "committed"])
def test_a_drift_entry_names_unit_section_key_reason_and_both_sides(tmp_path, repo_label):
    unit = _unit(tmp_path, "a", installed="B\n")
    report = _report() if repo_label == "repo" else _report(repo_label=repo_label)

    rc, lines = _run([unit], report=report)

    assert rc == 1
    assert lines[1:5] == _drift_lines(unit, repo_label)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def test_probe_findings_render_as_their_own_blocks_in_declared_order_after_drift(tmp_path):
    first = parity.BlockText("first", "{count} FIRST-HEADER:", ("FIRST-TAIL",))
    second = parity.BlockText("second", "{count} SECOND-HEADER:", ("SECOND-TAIL-1", "SECOND-TAIL-2"))
    probe, _ = _probe(
        lambda unit, text: parity.ProbeOutcome(
            findings=(
                parity.ProbeFinding(second, (f"  {unit.name}: two",)),
                parity.ProbeFinding(first, (f"  {unit.name}", "      detail")),
            )
        ),
        [first, second],
    )
    drifted = _unit(tmp_path, "a", installed="B\n")

    rc, lines = _run([drifted], probes=[probe])

    assert rc == 1
    assert lines[lines.index("[t] [drift] DRIFT-TAIL") + 1 :] == [
        "[t] [first] 1 FIRST-HEADER:",
        "[t]   a",
        "[t]       detail",
        "[t] [first] FIRST-TAIL",
        "[t] [second] 1 SECOND-HEADER:",
        "[t]   a: two",
        "[t] [second] SECOND-TAIL-1",
        "[t] [second] SECOND-TAIL-2",
    ]


def test_a_probe_finding_alone_is_a_finding(tmp_path):
    block = parity.BlockText("probed", "{count} PROBE-HEADER:", ("PROBE-TAIL",))
    probe, _ = _probe(
        lambda unit, text: parity.ProbeOutcome(findings=(parity.ProbeFinding(block, ("  a",)),)),
        [block],
    )

    rc, lines = _run([_unit(tmp_path, "a")], probes=[probe])

    assert rc == 1
    assert lines == ["[t] [probed] 1 PROBE-HEADER:", "[t]   a", "[t] [probed] PROBE-TAIL"]


def test_the_probe_runs_only_for_compared_units_and_receives_the_committed_text(tmp_path):
    probe, calls = _probe(lambda unit, text: parity.ProbeOutcome(), [])
    units = [
        _unit(tmp_path, "vanished", repo=None),
        _unit(tmp_path, "missing", installed=None),
        _unit(tmp_path, "unreadable", installed=UNDECODABLE),
        _unit(tmp_path, "compared", repo="COMMITTED\n", installed="COMMITTED\n"),
    ]

    _run(units, probes=[probe])

    assert calls == [("compared", "COMMITTED\n")]


def test_a_probe_dropin_naming_a_globbed_file_by_another_spelling_is_listed_once(tmp_path):
    real_dir = tmp_path / "installed"
    link_dir = tmp_path / "link"
    real_dir.mkdir()
    os.symlink(real_dir, link_dir)
    unit = parity.UnitRef("a", tmp_path / "repo" / "a", link_dir / "a")
    _write(unit.repo_path, "A\n")
    _write(real_dir / "a", "A\n")
    real_dropin = real_dir / "a.d" / "10-x.conf"
    _write(real_dropin, "[Service]\n")
    probe, _ = _probe(lambda u, text: parity.ProbeOutcome(dropins=(real_dropin,)), [])

    rc, lines = _run([unit], probes=[probe])

    assert rc == 1
    assert [line for line in lines if line.startswith("[t]   ")] == [
        f"[t]   a: {link_dir / 'a.d' / '10-x.conf'}"
    ]


def test_a_probe_only_dropin_creates_the_units_override_entry(tmp_path):
    elsewhere = pathlib.Path("/etc/systemd/user/a.d/50-site.conf")
    probe, _ = _probe(lambda unit, text: parity.ProbeOutcome(dropins=(elsewhere,)), [])

    rc, lines = _run([_unit(tmp_path, "a")], probes=[probe])

    assert rc == 1
    assert lines == [
        "[t] [override] 1 OVERRIDE-HEADER:",
        f"[t]   a: {elsewhere}",
        "[t] [override] OVERRIDE-TAIL",
    ]


def test_a_confirmation_prints_just_before_ok_and_only_on_parity(tmp_path):
    probe, _ = _probe(
        lambda unit, text: parity.ProbeOutcome(confirmation=f"CONFIRMED {unit.name}"), []
    )

    rc, lines = _run([_unit(tmp_path / "clean", "a")], probes=[probe])
    drift_rc, drift_lines = _run([_unit(tmp_path / "drifted", "a", installed="B\n")], probes=[probe])

    assert (rc, lines) == (
        0,
        ["[t] CONFIRMED a", "[t] [ok] parity — 1 unit(s) match their committed copies."],
    )
    assert drift_rc == 1
    assert not any("CONFIRMED" in line for line in drift_lines)


def test_a_probe_finding_in_an_undeclared_block_is_rejected_by_tag(tmp_path):
    declared = parity.BlockText("declared", "{count} D:", ())
    rogue = parity.BlockText("rogue", "{count} R:", ())
    probe, _ = _probe(
        lambda unit, text: parity.ProbeOutcome(findings=(parity.ProbeFinding(rogue, ("  a",)),)),
        [declared],
    )

    with pytest.raises(ValueError, match="rogue"):
        _run([_unit(tmp_path, "a")], probes=[probe])


@pytest.mark.parametrize("kind", parity.VERDICT_KINDS)
def test_a_probe_block_reusing_a_built_in_verdict_kind_is_rejected_when_declared(kind):
    clashing = parity.BlockText(kind, "{count} C:", ())

    with pytest.raises(ValueError, match=kind):
        parity.Probe(blocks=(clashing,), run=lambda unit, text: parity.ProbeOutcome())


# ---------------------------------------------------------------------------
# Verdict lines
# ---------------------------------------------------------------------------


def _verdict_units(tmp_path):
    drifted = _unit(tmp_path, "c-drift", installed="B\n")
    _dropin(drifted)
    return [drifted, _unit(tmp_path, "b-absent", installed=None), _unit(tmp_path, "a-clean")]


def test_verdict_lines_are_sorted_and_precede_every_human_block(tmp_path):
    rc, lines = _run(_verdict_units(tmp_path), print_verdicts=True)

    assert rc == 1
    assert lines[:3] == [
        "[t] verdict a-clean clean",
        "[t] verdict b-absent absent",
        "[t] verdict c-drift drift,override",
    ]
    assert not any(line.startswith(VERDICT) for line in lines[3:])


def test_verdict_lines_are_purely_additive(tmp_path):
    with_verdicts = _run(_verdict_units(tmp_path / "on"), print_verdicts=True)
    without = _run(_verdict_units(tmp_path / "off"))

    def normalised(result, root):
        rc, lines = result
        return rc, [
            line.replace(str(root), "<ROOT>") for line in lines if not line.startswith(VERDICT)
        ]

    assert not any(line.startswith(VERDICT) for line in without[1])
    assert normalised(with_verdicts, tmp_path / "on") == normalised(without, tmp_path / "off")


# ---------------------------------------------------------------------------
# The tagger and the block renderer
# ---------------------------------------------------------------------------


def test_emit_tags_every_physical_line():
    out = io.StringIO()

    parity.emit("t", "one\ntwo\n  three", out=out)

    assert out.getvalue() == "[t] one\n[t] two\n[t]   three\n"


def test_emit_block_renders_header_entries_then_each_tail_line():
    out = io.StringIO()

    parity.emit_block("t", "tag", "HEADER", ["  e1", "  e2"], ("T1", "T2"), out=out)

    assert out.getvalue().splitlines() == [
        "[t] [tag] HEADER",
        "[t]   e1",
        "[t]   e2",
        "[t] [tag] T1",
        "[t] [tag] T2",
    ]


# ---------------------------------------------------------------------------
# The Environment= policy table
# ---------------------------------------------------------------------------


_DASH, _FUSED = parity.UnitFamily.DASHBOARD, parity.UnitFamily.FUSED_MEMORY


def test_a_host_local_row_that_is_also_required_is_rejected_naming_row_and_fields():
    with pytest.raises(ValueError) as excinfo:
        parity.EnvPolicy(_FUSED, "X", host_local=True, required_value="v")

    message = str(excinfo.value)
    assert _FUSED.value in message
    assert "'X'" in message
    assert "host_local" in message
    assert "required_value" in message


def test_a_host_local_row_whose_value_is_compared_cross_copy_is_rejected_naming_the_row():
    with pytest.raises(ValueError) as excinfo:
        parity.EnvPolicy(_DASH, "X", host_local=True, compared_cross_copy=True)

    message = str(excinfo.value)
    assert _DASH.value in message
    assert "'X'" in message
    assert "compared_cross_copy" in message


@pytest.mark.parametrize(
    ("family", "column", "value"),
    [(_FUSED, "compared_cross_copy", False), (_DASH, "required_value", "v")],
)
def test_a_column_the_familys_checker_never_reads_is_rejected(family, column, value):
    with pytest.raises(ValueError) as excinfo:
        parity.EnvPolicy(family, "X", **{column: value})

    message = str(excinfo.value)
    assert family.value in message
    assert "'X'" in message
    assert column in message


@pytest.mark.parametrize("name", ["", "A=B", "A B", "A\tB", "A\nB"])
def test_a_name_that_cannot_be_an_environment_variable_is_rejected(name):
    with pytest.raises(ValueError, match="name"):
        parity.EnvPolicy(_FUSED, name)


def test_a_required_value_spanning_lines_is_rejected():
    with pytest.raises(ValueError, match="required_value"):
        parity.EnvPolicy(_FUSED, "X", required_value="a\nb")


def test_a_table_declaring_one_family_and_name_twice_is_rejected_naming_it():
    first = parity.EnvPolicy(_FUSED, "DUP")
    second = parity.EnvPolicy(_FUSED, "DUP", host_local=True)

    with pytest.raises(ValueError) as excinfo:
        parity.EnvPolicyTable((first, second))

    message = str(excinfo.value)
    assert _FUSED.value in message
    assert "'DUP'" in message


def test_one_name_in_two_families_is_two_rows():
    table = parity.EnvPolicyTable((parity.EnvPolicy(_DASH, "SAME"), parity.EnvPolicy(_FUSED, "SAME")))

    assert len(table.rows) == 2


def test_a_row_setting_no_column_is_the_policy_of_a_name_with_no_row():
    table = parity.EnvPolicyTable(tuple(parity.EnvPolicy(f, "X") for f in parity.UnitFamily))

    for family in parity.UnitFamily:
        assert table.host_local_names(family) == ()
        assert table.divergence_allowlist(family) == frozenset()
        assert table.required_directives(family) == ()


def _synthetic_table() -> "parity.EnvPolicyTable":
    return parity.EnvPolicyTable(
        (
            parity.EnvPolicy(_DASH, "D_LOCAL_B", host_local=True, compared_cross_copy=False),
            parity.EnvPolicy(_FUSED, "F_REQUIRED_B", required_value="2"),
            parity.EnvPolicy(_DASH, "D_BLESSED", compared_cross_copy=False),
            parity.EnvPolicy(_FUSED, "F_LOCAL", host_local=True),
            parity.EnvPolicy(_DASH, "D_LOCAL_A", host_local=True, compared_cross_copy=False),
            parity.EnvPolicy(_FUSED, "F_REQUIRED_A", required_value="a b=c"),
            parity.EnvPolicy(_DASH, "D_COMPARED"),
        )
    )


def test_host_local_names_are_that_familys_in_table_order():
    table = _synthetic_table()

    assert table.host_local_names(_DASH) == ("D_LOCAL_B", "D_LOCAL_A")
    assert table.host_local_names(_FUSED) == ("F_LOCAL",)


def test_divergence_allowlist_is_the_familys_names_not_compared_cross_copy():
    allowlist = _synthetic_table().divergence_allowlist(_DASH)

    assert isinstance(allowlist, frozenset)
    assert allowlist == frozenset({"D_LOCAL_B", "D_BLESSED", "D_LOCAL_A"})


def test_required_directives_are_the_familys_required_rows_as_environment_lines():
    assert _synthetic_table().required_directives(_FUSED) == (
        "Environment=F_REQUIRED_B=2",
        "Environment=F_REQUIRED_A=a b=c",
    )


def test_the_live_table_gives_every_consumer_rows_to_project():
    """No consumer's derivation test can pass by comparing two empty projections."""
    for family in parity.UnitFamily:
        assert parity.ENV_POLICY.host_local_names(family), family
    assert parity.ENV_POLICY.divergence_allowlist(_DASH)
    assert parity.ENV_POLICY.required_directives(_FUSED)
