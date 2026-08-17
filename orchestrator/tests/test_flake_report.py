"""Tests for the flake-ledger operator report (plans/flake-ledger-prd.md, task ι).

DB-backed fixtures seed ``flake_occurrence`` / ``flake_debt`` with
``flake_ledger.ensure_schema`` plus raw ``sqlite3`` INSERTs, never through
``record_flake_occurrence`` / ``open_debt`` / ``resolve_debt``.  That follows
``test_flake_ledger.py``'s stated convention of bypassing the module under test for
on-disk truth, and it additionally keeps ι's suite decoupled from the two write paths
tasks ζ and η are concurrently rewriting.

STRUCTURAL CONSTRAINT — sync and async tests live in strictly separate classes.
``orchestrator/pyproject.toml`` sets no ``asyncio_mode`` (pytest-asyncio STRICT) and
promotes the mark-mismatch warning to an ERROR-level filterwarning.  ι's report path is
ENTIRELY synchronous, so every test here is a plain sync ``def test_`` and no class in
this file carries ``@pytest.mark.asyncio``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from orchestrator.flake_report import _parse_stamp, format_age


class TestStampAndAge:
    """The two rendering primitives every other section leans on."""

    def test_canonical_utc_stamp_parses_to_aware_utc(self):
        parsed = _parse_stamp('2026-08-08T12:00:00+00:00')
        assert parsed is not None
        assert parsed.tzinfo is not None, 'a parsed stamp must be timezone-AWARE'
        assert parsed.utcoffset() == timedelta(0)
        assert parsed == datetime(2026, 8, 8, 12, 0, 0, tzinfo=UTC)

    def test_naive_stamp_is_assumed_utc(self):
        # Mirrors flake_ledger._canonicalize_utc: a naive value gets UTC ATTACHED,
        # never .astimezone(), which would apply the HOST's local offset and silently
        # shift the stamp by the dispatcher's timezone.
        parsed = _parse_stamp('2026-08-08T12:00:00')
        assert parsed is not None
        assert parsed.tzinfo is not None
        assert parsed == datetime(2026, 8, 8, 12, 0, 0, tzinfo=UTC)

    def test_z_and_offset_spellings_are_the_same_instant(self):
        # flake_ledger stores canonicalised '+00:00', but a hand-built stamp may
        # arrive as '…Z'.  Both spellings must name one instant, or an age computed
        # from one and a window bound from the other disagree.
        assert _parse_stamp('2026-08-08T12:00:00Z') == _parse_stamp('2026-08-08T12:00:00+00:00')

    def test_malformed_stamp_returns_none_and_does_not_raise(self):
        # One bad row must not take down the whole report: a read path an operator
        # cannot rely on is worse than a row rendered 'unknown'.
        assert _parse_stamp('not-a-date') is None

    def test_none_stamp_returns_none(self):
        assert _parse_stamp(None) is None

    def test_format_age_renders_days_and_hours(self):
        assert format_age(timedelta(days=3, hours=4)) == '3d 4h'

    def test_format_age_renders_zero_days_explicitly(self):
        assert format_age(timedelta(hours=5)) == '0d 5h'

    def test_format_age_of_none_says_unknown_not_zero(self):
        # The dangerous direction: an unparseable opened_at rendered as '0d 0h' makes a
        # stale debt row look brand new and suppresses it from the age backstop.
        rendered = format_age(None)
        assert 'unknown' in rendered, rendered
        assert rendered != '0d 0h'
