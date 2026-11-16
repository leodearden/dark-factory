"""The D5 structural invariants every ``shared.toolcall_markup.repair`` result must satisfy.

Test-support module, NOT production code. Consumed by
``shared/tests/test_toolcall_markup.py``'s hand-authored specimens, by
``shared/tests/test_toolcall_markup_corpus.py``'s committed-corpus replay, and
by the report-specimen gate in
``tests/repo_guards/test_upstream_parser_report.py``. All three must run this
SAME check, so it lives in one place rather than in any one of their test
modules. Importable bare (``from toolcall_repair_invariants import ...``)
because ``shared/tests/conftest.py`` and ``tests/repo_guards/conftest.py`` both
insert this ``cross_suite_helpers/`` directory onto ``sys.path``.
"""

from __future__ import annotations

from shared.toolcall_markup import Repair, detect


def assert_repair_invariants(value: object, result: Repair | None) -> None:
    """Assert PRD section 4 C1's D5 structural invariants for one repair.

    NON-CIRCULAR BY CONSTRUCTION — which is why the committed-corpus replay
    (``test_toolcall_markup_corpus.py``) imports this exact function rather
    than re-deriving it. Both properties hold or fail independently of what any
    expectation column says, so unlike the machine-generated ``expected_outcome``
    they can actually falsify the repairer:

    * ``clean_value`` is a PREFIX of the input — the repairer never invents or
      reorders caller text;
    * every recovered value is a VERBATIM SUBSTRING of the input — the repairer
      never synthesises a value;
    * ``misclose`` sits exactly at the prefix boundary, so the three fields
      describe one consistent cut of the input rather than three guesses;
    * ``clean_value`` is ENVELOPE-FREE — ``detect(clean_value) is None``. A
      repair whose own output still trips the detector is a SILENT PARTIAL
      repair: contract C2's middleware forwards ``clean_value`` as the repaired
      argument, so a residual envelope there would re-trip the write-time
      tripwire downstream AND leave the arguments hiding in the residue
      permanently dropped, with no diagnostic. Refusing outright is the only
      honest answer; this clause is what makes that hold corpus-wide rather
      than for one hand-authored specimen.

    ``None`` (unrepairable) satisfies all four trivially and is accepted, so a
    caller can pass any result through unconditionally.
    """
    if result is None:
        return
    assert isinstance(value, str), 'a Repair can only come from a str input'
    assert value.startswith(result.clean_value), (
        'D5 violated: clean_value is not a prefix of the input'
    )
    for name, recovered_value in result.recovered.items():
        assert recovered_value in value, (
            f'D5 violated: recovered {name!r} is not a verbatim substring of the input'
        )
    assert value[len(result.clean_value):].startswith(result.misclose), (
        'D5 violated: misclose does not sit at the clean_value boundary'
    )
    residual = detect(result.clean_value)
    assert residual is None, (
        f'SILENT PARTIAL REPAIR: clean_value still carries {residual!r}; '
        'repair() must refuse rather than hand the middleware a value that '
        'still trips detect() and still swallows the caller arguments hiding '
        'in the residue'
    )
