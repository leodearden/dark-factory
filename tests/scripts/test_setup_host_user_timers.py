"""setup-host.sh §7b: arm the expected user timers, believing the status only with the [user_timers] tag."""

from __future__ import annotations

import json
import pathlib
import subprocess

from setup_host_sections import checker_repo, run_section, slice_section, usage_error_checker

SCRIPT_NAME = "user_timers.py"
REPORT_TAG = "[user_timers]"
_SITE_START = "_user_timers_script="
_SITE_END = "\nfi\n"
_ARGV_RECORD = "user-timers-argv.json"


def _stub_checker(tmp_path: pathlib.Path, report: str, status: int) -> str:
    return (
        "import json, sys\n"
        f"open({str(tmp_path / _ARGV_RECORD)!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
        f"print({report!r})\n"
        f"sys.exit({status})\n"
    )


def _run_site(
    tmp_path: pathlib.Path, *, body: str | None, with_checker: bool = True
) -> subprocess.CompletedProcess:
    repo = checker_repo(tmp_path, SCRIPT_NAME, body=body, with_checker=with_checker)
    result = run_section(
        tmp_path, slice_section(_SITE_START, _SITE_END), repo_root=repo, unit_dir=tmp_path / "units"
    )
    assert result.returncode == 0, (
        f"fail() only printfs; the arming site must not abort setup-host.\n{result.stdout}\n{result.stderr}"
    )
    return result


def test_site_reports_ok_when_every_timer_is_armed(tmp_path: pathlib.Path):
    result = _run_site(tmp_path, body=_stub_checker(tmp_path, f"{REPORT_TAG} 15 of 15 expected user timers armed", 0))

    assert "OK " in result.stdout, result.stdout
    assert "WARN " not in result.stdout, result.stdout
    assert "FAIL " not in result.stdout, result.stdout


def test_site_warns_on_a_tagged_unarmed_report_and_reprints_it(tmp_path: pathlib.Path):
    report = f"{REPORT_TAG} [masked] return-brief.timer: a mask is a deliberate off"
    result = _run_site(tmp_path, body=_stub_checker(tmp_path, report, 1))

    assert report in result.stdout, result.stdout
    assert any(
        line.startswith("WARN ") and f"{REPORT_TAG} report" in line for line in result.stdout.splitlines()
    ), result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_site_does_not_believe_an_untagged_report(tmp_path: pathlib.Path):
    result = _run_site(tmp_path, body=_stub_checker(tmp_path, "15 of 15 expected user timers armed", 0))

    assert any(
        line.startswith("FAIL ") and f"produced no {REPORT_TAG} report" in line for line in result.stdout.splitlines()
    ), result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_site_fails_on_a_usage_error(tmp_path: pathlib.Path):
    result = _run_site(tmp_path, body=usage_error_checker(SCRIPT_NAME, "[-h] {check,arm}", "--bogus"))

    assert f"produced no {REPORT_TAG} report" in result.stdout, result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_site_fails_when_the_script_is_missing(tmp_path: pathlib.Path):
    result = _run_site(tmp_path, body=None, with_checker=False)

    missing = tmp_path / "repo" / "scripts" / SCRIPT_NAME
    assert any(
        line.startswith("FAIL ") and str(missing) in line for line in result.stdout.splitlines()
    ), result.stdout
    assert "OK " not in result.stdout, result.stdout


def test_site_runs_arm_against_this_repo_root(tmp_path: pathlib.Path):
    _run_site(tmp_path, body=_stub_checker(tmp_path, f"{REPORT_TAG} 15 of 15 expected user timers armed", 0))

    repo = tmp_path / "repo"
    assert json.loads((tmp_path / _ARGV_RECORD).read_text()) == ["arm", "--repo-root", str(repo)]
