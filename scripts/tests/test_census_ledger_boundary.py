"""Boundary tests for one census run against the processed-session ledger
(plans/census-incremental-prd.md §4.2-4.3, §4.8 rows 1, 2 and 4).

Real transcripts and a real sqlite ledger in tmp, mined through a real
``census_window.WindowBatchSource``; only the LLM, MCP and git seams are
fakes, passed through ``run_census``'s own parameters.
"""
from __future__ import annotations

import json
import logging
import random
import sqlite3
from contextlib import closing
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import census as mod
import pytest
from legibility import census_window, config, inventory, session_ledger, trickle_state, unlanded
from legibility.session_ledger import CodedBy, LedgerRow, Outcome

_NOW = datetime(2026, 10, 6, 12, 0, tzinfo=UTC)
_DATE = "2026-10-06"
_RUN_ID = "census-p-20261006"
_EMPTY_JUDGMENT = json.dumps({"matches": [], "candidates": []})


@pytest.fixture(autouse=True)
def _isolate_state_root(tmp_path, monkeypatch):
    monkeypatch.setenv(trickle_state.STATE_ROOT_ENV, str(tmp_path / "state"))


def _write_session(directory: Path, cwd: str, sid: str, *, day: date, signal: bool) -> None:
    base = {"cwd": cwd, "sessionId": sid, "timestamp": f"{day.isoformat()}T10:00:00.000Z"}
    records: list[dict[str, Any]] = [{
        **base, "type": "user", "isSidechain": False, "isMeta": False,
        "message": {"role": "user", "content": f"Please look at {sid}."},
    }]
    if signal:
        records.append({
            **base, "type": "user",
            "message": {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "t1", "is_error": True,
                "content": "cat: /x: No such file or directory",
            }]},
        })
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{sid}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )


class _Recorder:
    """A seam fake that records its keyword calls and returns *result*."""

    def __init__(self, result=None, *, raises: Exception | None = None, effect=None):
        self.calls: list[dict[str, Any]] = []
        self._result = result
        self._raises = raises
        self._effect = effect

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        if self._effect is not None:
            self._effect()
        if self._raises is not None:
            raise self._raises
        return self._result


class _Invoke:
    """The miner/probe seam: every prompt gets an empty judgment."""

    def __init__(self, reply: str = _EMPTY_JUDGMENT):
        self.prompts: list[str] = []
        self._reply = reply

    def __call__(self, prompt, model):
        self.prompts.append(prompt)
        return self._reply

    def mined(self, sessions) -> list[str]:
        return sorted(sid for sid in sessions for prompt in self.prompts if sid in prompt)


class _Census:
    def __init__(self, tmp_path: Path):
        self.root = tmp_path / "project"
        (self.root / "plans").mkdir(parents=True)
        self.cwd = str(self.root)
        self.projects_root = tmp_path / "projects"
        self.cfg = config.LegibilityConfig(
            project_id="p",
            project_root=self.cwd,
            escalation_port=8103,
            cwd_prefixes=[self.cwd],
        )
        self.ledger_path = session_ledger.ledger_path("p")
        self.codebook_path = self.root / "docs" / "legibility" / "confusion-codebook.yaml"
        self.state_path = self.root / "docs" / "legibility" / "census-state.json"

    def session(self, sid: str, *, signal: bool = True, days_ago: int = 3) -> None:
        _write_session(
            self.projects_root / inventory.encode_cwd(self.cwd), self.cwd, sid,
            day=_NOW.date() - timedelta(days=days_ago), signal=signal,
        )

    def trickle_row(self, sid: str) -> None:
        session_ledger.record_codings(self.ledger_path, [LedgerRow(
            session=sid, instrument_version=5, coded_by=CodedBy.TRICKLE,
            run_ref="trickle-p-20261005", outcome=Outcome.EMPTY, coded_at=_NOW,
        )])

    def source(self, *, batch_size: int = census_window.DEFAULT_BATCH_SIZE):
        return census_window.WindowBatchSource(
            self.cfg,
            projects_root=self.projects_root,
            now=_NOW,
            ledger_path=self.ledger_path,
            last_census_at=None,
            batch_size=batch_size,
            rng=random.Random(0),
        )

    def report_path(self, basename: str = f"confusion-census-{_DATE}") -> Path:
        return self.root / "plans" / f"{basename}.md"

    def kwargs(self, **overrides) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            batch_source=self.source(),
            invoke=_Invoke(),
            verify_fn=lambda clusters, *, model: {"verified": [], "rejected": [], "fixed": []},
            synthesize_fn=lambda verified, *, model: "No novel clusters this census.",
            submit_fn=_Recorder({"ticket": "tkt_1"}),
            escalate_fn=_Recorder(),
            status_fetcher=lambda: {"statuses": {}},
            commit=_Recorder(),
            roll_back=_Recorder(unlanded.Rollback(quarantine_dir=None, paths=())),
            codebook_dict={"version": 2, "entries": [], "candidates": []},
            config=self.cfg,
            project_root=self.cwd,
            project_id="p",
            codebook_path=self.codebook_path,
            census_state_path=self.state_path,
            report_path=self.report_path(),
            date=_DATE,
            run_id=_RUN_ID,
            as_of_sha="a" * 40,
            since=None,
        )
        kwargs.update(overrides)
        return kwargs

    def ledger_rows(self) -> list[tuple[str, str, str]]:
        with closing(sqlite3.connect(self.ledger_path)) as conn:
            return conn.execute(
                "SELECT session, coded_by, run_ref FROM coded_sessions ORDER BY session"
            ).fetchall()


@pytest.fixture
def census(tmp_path):
    return _Census(tmp_path)


def _record_of(report_path: Path) -> tuple[str, dict[str, Any]]:
    text = report_path.with_suffix(".json").read_text(encoding="utf-8")
    return text, json.loads(text)


def test_the_run_writes_its_record_first_and_renders_the_report_from_it(census):
    census.session("S-tango")
    kwargs = census.kwargs()

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "done"
    report_path = kwargs["report_path"]
    record_path = report_path.with_suffix(".json")
    assert outcome.record_path == str(record_path)
    text, record = _record_of(report_path)
    assert report_path.read_text(encoding="utf-8") == mod.render_report(json.loads(text))
    assert record["run_id"] == _RUN_ID
    [commit] = kwargs["commit"].calls
    assert set(commit["paths"]) >= {
        str(report_path), str(record_path), str(census.codebook_path), str(census.state_path),
    }


def test_the_method_records_the_selection_and_the_state_its_watermark(census):
    census.session("S-tango")
    source = census.source()

    mod.run_census(**census.kwargs(batch_source=source))

    _, record = _record_of(census.report_path())
    selection = source.selection
    assert selection is not None
    evidence = record["method"]["evidence"]
    assert evidence == {
        "window": selection.window.to_record(),
        "sessions_enumerated": 1,
        "skipped_coded": 0,
        "skipped_zero_signal": 0,
        "mined": 1,
        "ledger_rows": 0,
    }
    assert record["method"]["extra"]["ledger_created_this_run"] is True
    state = json.loads(census.state_path.read_text(encoding="utf-8"))
    assert state["session_watermark"] == selection.window.end.isoformat()


def test_row1_a_trickle_coded_session_is_skipped_and_the_rest_ledgered(census):
    census.session("S-sierra")
    census.session("S-tango")
    census.trickle_row("S-sierra")
    invoke = _Invoke()

    outcome = mod.run_census(**census.kwargs(invoke=invoke))

    _, record = _record_of(census.report_path())
    assert record["method"]["evidence"]["skipped_coded"] == 1
    assert invoke.mined(["S-sierra", "S-tango"]) == ["S-tango"]
    assert census.ledger_rows() == [
        ("S-sierra", "trickle", "trickle-p-20261005"),
        ("S-tango", "census", _RUN_ID),
    ]
    assert outcome.ledger_rows_written == 1
    assert outcome.ledger_write_error is None


def test_row2_a_capped_run_is_resumed_by_the_next(census):
    sessions = ["S-alpha", "S-bravo", "S-charlie", "S-delta"]
    for sid in sessions:
        census.session(sid)

    first = mod.run_census(**census.kwargs(
        batch_source=census.source(batch_size=2), max_batches=1,
    ))

    assert first.stop_reason == "capped"
    first_rows = census.ledger_rows()
    assert [row[1:] for row in first_rows] == [("census", _RUN_ID)] * 2
    assert "\n- RESUMED NEXT RUN:" in census.report_path().read_text(encoding="utf-8")

    second_invoke = _Invoke()
    second_report = census.report_path(f"confusion-census-{_DATE}-2")
    mod.run_census(**census.kwargs(
        batch_source=census.source(batch_size=2),
        invoke=second_invoke,
        report_path=second_report,
        run_id=f"{_RUN_ID}-2",
    ))

    first_mined = sorted(row[0] for row in first_rows)
    assert second_invoke.mined(sessions) == sorted(set(sessions) - set(first_mined))
    _, record = _record_of(second_report)
    assert record["method"]["evidence"]["skipped_coded"] == 2
    final = census.ledger_rows()
    assert sorted(row[0] for row in final) == sessions


def test_row4_a_zero_signal_session_is_never_mined(census):
    census.session("S-tango")
    census.session("S-zulu", signal=False)
    invoke = _Invoke()

    mod.run_census(**census.kwargs(invoke=invoke))

    assert invoke.mined(["S-tango", "S-zulu"]) == ["S-tango"]
    _, record = _record_of(census.report_path())
    assert record["method"]["evidence"]["skipped_zero_signal"] == 1


def test_an_unlanded_commit_rolls_back_the_record_and_ledgers_nothing(census):
    census.session("S-tango")
    roll_back = _Recorder(unlanded.Rollback(quarantine_dir=None, paths=()))
    kwargs = census.kwargs(
        commit=_Recorder(raises=RuntimeError("pre-commit refused")), roll_back=roll_back,
    )

    outcome = mod.run_census(**kwargs)

    assert outcome.status == "unlanded"
    assert outcome.record_path is None
    [call] = roll_back.calls
    assert str(kwargs["report_path"].with_suffix(".json")) in call["paths"]
    assert census.ledger_rows() == []


def test_a_preflight_defer_ledgers_nothing(census):
    census.session("S-tango")

    outcome = mod.run_census(**census.kwargs(
        invoke=_Invoke("You have reached your usage limit for this period."),
    ))

    assert outcome.status == "deferred"
    assert session_ledger.read_ledger(census.ledger_path).total_rows is None


def test_a_ledger_write_failure_after_the_commit_is_reported_not_raised(census, caplog):
    census.session("S-tango")

    def clobber_ledger():
        census.ledger_path.write_bytes(b"this is not a sqlite database")

    with caplog.at_level(logging.WARNING):
        outcome = mod.run_census(**census.kwargs(commit=_Recorder(effect=clobber_ledger)))

    assert outcome.status == "done"
    assert outcome.ledger_rows_written is None
    assert outcome.ledger_write_error is not None
    assert str(census.ledger_path) in outcome.ledger_write_error
    ledger_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and str(census.ledger_path) in r.getMessage()
    ]
    assert len(ledger_warnings) == 1
