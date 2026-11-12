"""Scaffolding shared by the suites that drive ``census_run.run_census``.

A plain module rather than a conftest fixture, the same convention as
``legibility_task_store_fake``: conftest appends this directory to
``sys.path``, so the test files import it by bare name.

Nothing here imports a census module. The outcome and effects classes come
from ``legibility.census_run``; a bare ``import census_run`` would load a
second module object with a second set of classes, so each helper takes the
caller's ``CensusEffects`` instance or outcome class instead.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

Outcome = TypeVar("Outcome")
Effects = TypeVar("Effects", bound="DataclassInstance")

PERSISTING_SEAMS = {
    "submit_fn": "file tickets",
    "dump_codebook": "dump codebook",
    "advance_state": "advance state",
    "gate": "gate",
    "commit": "commit",
    "roll_back": "roll back",
    "record_codings": "ledger",
}
"""The journal label of each persisting ``CensusEffects`` seam other than
``write_text``, whose label depends on the path it writes."""


def narrowing(variant: type[Outcome]) -> Callable[[object], Outcome]:
    """A check that a run's outcome is *variant*, returning it narrowed."""
    def narrowed(outcome: object) -> Outcome:
        assert isinstance(outcome, variant), outcome
        return outcome

    return narrowed


def with_seams(effects: Effects, overrides: Mapping[str, Any]) -> tuple[Effects, dict[str, Any]]:
    """*effects* with every override named after one of its fields, and the
    overrides left over."""
    names = {seam.name for seam in dataclasses.fields(effects)}
    seams = {name: value for name, value in overrides.items() if name in names}
    rest = {name: value for name, value in overrides.items() if name not in names}
    return (dataclasses.replace(effects, **seams) if seams else effects), rest


def journaled(effects: Effects, journal: list[str], *, labels: Mapping[Path, str]) -> Effects:
    """*effects* with every persisting seam appending its label to *journal*
    before it delegates, so the real writes still happen. ``write_text`` is
    labelled by the path it writes, from *labels*."""
    def logged(name: str, label_of: Callable[..., str]) -> Callable[..., Any]:
        seam = getattr(effects, name)

        def call(*args: Any, **kwargs: Any) -> Any:
            journal.append(label_of(*args))
            return seam(*args, **kwargs)

        return call

    def constant(label: str) -> Callable[..., str]:
        return lambda *args: label

    return dataclasses.replace(
        effects,
        write_text=logged("write_text", lambda path, text: labels[Path(path)]),
        **{name: logged(name, constant(label)) for name, label in PERSISTING_SEAMS.items()},
    )
