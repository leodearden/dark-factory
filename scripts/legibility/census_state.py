"""scripts/legibility/census_state.py — the census-state.json document
(plans/census-incremental-prd.md §4.2): its validated reader, :func:`load`,
and its one writer, :func:`advance`.

One other reader remains: ``trickle_state.read_last_census_at`` reads the
raw ``last_census_at`` of the same file, quietly, for the census-fire
streak's exact-equality check. A key renamed or retyped here must change
there too.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

# Import state_docs by the same spelling this module was imported by, so a
# caller matching on Missing / Malformed / Ok sees the classes this module
# returns: one source file loaded as two modules has two copies of each.
if __package__:
    from . import state_docs
else:
    import state_docs


def _as_census_date(name: str, raw: object) -> date:
    """The written ``YYYY-MM-DD``, or a legacy ISO datetime read as its UTC date."""
    try:
        return state_docs.as_date(name, raw)
    except ValueError:
        pass
    try:
        return state_docs.as_datetime(name, raw).astimezone(UTC).date()
    except ValueError:
        raise ValueError(
            f'{name}: expected an ISO date (or a legacy ISO datetime), got {raw!r}'
        ) from None


def _as_text(name: str, raw: object) -> str:
    text = state_docs.as_str(name, raw)
    if not text:
        raise ValueError(f'{name}: expected a non-empty string, got {raw!r}')
    return text


@dataclass(frozen=True)
class CensusState:
    """What the last census run recorded. Every field is ``None`` until a
    run has written it."""

    last_census_at: date | None
    last_census_run_id: str | None
    last_census_report: str | None
    last_census_as_of_sha: str | None
    session_watermark: datetime | None

    @classmethod
    def from_json(cls, raw: object) -> CensusState:
        """Strict about every field present; the legacy
        ``last_census_done_count`` and unknown keys are ignored."""
        doc = state_docs.json_object(raw, 'census state')
        defaulted = state_docs.defaulted
        return cls(
            last_census_at=defaulted(doc, 'last_census_at', _as_census_date, None),
            last_census_run_id=defaulted(doc, 'last_census_run_id', _as_text, None),
            last_census_report=defaulted(doc, 'last_census_report', _as_text, None),
            last_census_as_of_sha=defaulted(doc, 'last_census_as_of_sha', _as_text, None),
            session_watermark=defaulted(
                doc, 'session_watermark', state_docs.as_datetime, None
            ),
        )

    def to_json(self) -> dict[str, object]:
        return {
            'last_census_at': _isoformat(self.last_census_at),
            'last_census_run_id': self.last_census_run_id,
            'last_census_report': self.last_census_report,
            'last_census_as_of_sha': self.last_census_as_of_sha,
            'session_watermark': _isoformat(self.session_watermark),
        }


def _isoformat(value: date | None) -> str | None:
    return None if value is None else value.isoformat()


def load(path: str | Path) -> state_docs.StateRead[CensusState]:
    """Missing, Malformed (with the reason, one WARNING) or Ok. Never raises."""
    return state_docs.read_state(Path(path), CensusState.from_json)


def advance(
    path: str | Path,
    *,
    census_at: date,
    run_id: str,
    report_path: str,
    as_of_sha: str,
    session_watermark: datetime | None,
) -> CensusState:
    """Atomically replace *path* with the state a finished run leaves.

    *report_path* is repo-relative; *session_watermark* is the exclusive end
    of the mined window, ``None`` when the run had none. Every key is
    written, a ``None`` as null. Creates *path*'s parent directories."""
    state = CensusState(
        last_census_at=census_at,
        last_census_run_id=run_id,
        last_census_report=report_path,
        last_census_as_of_sha=as_of_sha,
        session_watermark=session_watermark,
    )
    state_docs.write_state(Path(path), state)
    return state
