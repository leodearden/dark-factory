"""Tests for ``scripts/inline_suppression_consumers.py``: which tool honours a marker.

One subject: D8's consumer model — whether pyright, ruff (through the nearest
``pyproject.toml``), a first-party checker, or nothing at all reads a given
suppression.  ``plans/inv12-exceptions-owned-or-ratified-prd.md``, D8.

ONE SUBPROCESS, deliberately.  The first-party code drift guard runs
``fused-memory/scripts/check_bare_magicmock_config.py`` as a subprocess and
reads the violation messages it EMITS, rather than importing its private
``_RULE_A_CODE`` / ``_RULE_B_CODE`` constants — ``docs/code-quality.md``'s
Tests stance (a test that reads another module's private attributes pins
implementation rather than behaviour).  Every other test here is in-process,
over a plain tree of files from ``inline_suppression_fixtures.write_files``.
"""

import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
from inline_suppression_consumers import FIRST_PARTY_CODES, Consumer, ConsumerModel
from inline_suppression_fixtures import RUFF_CONFIG, SUBPROCESS_TIMEOUT_SECS, write_files
from inline_suppression_kinds import KIND_SPECS, Kind, Site
from inline_suppression_refusal import InstrumentFailure

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# D8 — the consumer model: which tool, if any, actually reads this marker.

def _consumer_of(
    root: Path,
    *,
    path: str = 'pkg/mod.py',
    kind=None,
    codes: tuple[str, ...] = (),
):
    """The consumer *root*'s model resolves for one synthetic site."""
    site = Site(
        path=path,
        line=1,
        kind=kind if kind is not None else Kind.NOQA,
        codes=codes,
        text='x = 1',
    )
    return ConsumerModel(root).consumer_for(site)


def test_type_ignore_and_pyright_ignore_resolve_to_pyright_whatever_the_code(tmp_path: Path):
    """No config is consulted for these two: pyright runs over every package as
    a declared gate, so the marker is read wherever it sits."""
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n'})

    for kind in (
        Kind.TYPE_IGNORE,
        Kind.PYRIGHT_IGNORE,
    ):
        for codes in ((), ('arg-type',), ('reportArgumentType',), ('not-a-real-code',)):
            assert _consumer_of(tmp_path, kind=kind, codes=codes) is (
                Consumer.PYRIGHT
            ), (kind, codes)


def test_every_kind_the_scanner_scans_resolves_to_a_consumer(tmp_path: Path):
    """The totality that ``consumer_for``'s guard defends, asserted one stage
    earlier than the guard can fire.

    The consumer table is deliberately PARTIAL — ``noqa`` has no row because its
    answer depends on the code and the config — so a sixth kind added to
    ``KIND_SPECS`` and forgotten there falls through to the noqa path and is
    resolved against a ruff config that has never heard of it.  The guard turns
    that into a loud exit 2 for whoever next runs a scan; this turns it into a
    red for the author who added the kind, while they still hold the context to
    fix it.  Both, because the two catch it at different moments (heuristic 10).

    DRIVEN OFF ``KIND_SPECS`` AND THE PUBLIC RESOLVER, never by patching the
    private table by dotted path: that is the one shape ``docs/code-quality.md``
    names outright, and the first-party drift guard
    (``test_the_first_party_table_matches_the_codes_that_checker_actually_emits``)
    turns down the same shortcut for the first-party code table.  A behavioural assertion is also
    strictly stronger here — it fails for a kind whose row exists but resolves
    by accident, which a patched-out table could not detect.
    """
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    for kind in KIND_SPECS:
        assert isinstance(
            _consumer_of(tmp_path, kind=kind, codes=('E402',)),
            Consumer,
        ), kind


def test_pragma_no_cover_and_nosec_resolve_to_no_consumer(tmp_path: Path):
    """D8's finding: nothing in this repository reads either one today.

    No coverage gate runs ``# pragma: no cover``, and bandit is not installed —
    the live ``nosec`` count is zero, which is what makes that kind's row a
    statement about tools rather than about code.
    """
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    for kind in (
        Kind.PRAGMA_NO_COVER,
        Kind.NOSEC,
    ):
        assert _consumer_of(tmp_path, kind=kind) is Consumer.NONE, kind


def test_a_noqa_code_the_nearest_config_selects_is_consumed_by_ruff(tmp_path: Path):
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    for code in ('E402', 'F401', 'B006', 'SIM102', 'I001', 'UP038'):
        assert _consumer_of(tmp_path, codes=(code,)) is Consumer.RUFF, code


def test_a_noqa_code_the_nearest_config_does_not_select_has_no_consumer(tmp_path: Path):
    """D8's whole point, and its largest single inflow.

    ``PLC0415`` alone accounts for 963 of this tree's markers and no
    ``pyproject.toml`` here selects ``PL``; every one of them is dead.
    Reporting them as ruff-consumed would leave the biggest source of new
    markers entirely unpoliced.
    """
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    for code in ('PLC0415', 'ANN001', 'ARG002', 'A002', 'N802'):
        assert _consumer_of(tmp_path, codes=(code,)) is Consumer.NONE, code


def test_a_selector_matches_by_linter_and_number_never_by_string_prefix(tmp_path: Path):
    """THE LOAD-BEARING CASE — ``select = ["B"]`` does NOT select ``BLE001``.

    Measured against real ruff, not assumed: ``ruff check --isolated --select
    B`` does not flag ``BLE001`` while ``--select BLE`` does.  A selector
    resolves to a (linter, code-prefix) PAIR, so linter ``B``
    (flake8-bugbear) never reaches linter ``BLE`` (flake8-blind-except).  This
    tree carries 202 ``# noqa: BLE001`` markers — the second-largest
    population — and a naive ``code.startswith(selector)`` would silently
    report every one of them as ruff-consumed, defeating D8 for them.

    Parsing both sides at the boundary into a typed ``(linter, number)`` pair
    is also exactly what heuristic 12 prescribes, so the correct behaviour and
    the cited heuristic coincide here.
    """
    write_files(
        tmp_path,
        {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': '[tool.ruff.lint]\nselect = ["B"]\n'},
    )

    assert _consumer_of(tmp_path, codes=('B006',)) is Consumer.RUFF
    assert _consumer_of(tmp_path, codes=('BLE001',)) is Consumer.NONE


def test_a_partial_selector_matches_on_the_number_prefix(tmp_path: Path):
    """Within one linter, a selector IS a numeric prefix: ``E4`` selects E402
    and ``E5`` does not."""
    write_files(
        tmp_path,
        {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': '[tool.ruff.lint]\nselect = ["E4"]\n'},
    )

    assert _consumer_of(tmp_path, codes=('E402',)) is Consumer.RUFF
    assert _consumer_of(tmp_path, codes=('E501',)) is Consumer.NONE


def test_an_ignored_code_has_no_consumer_even_though_a_selector_matches(tmp_path: Path):
    """All eight pyprojects here set ``ignore = ["E501"]``, so ruff provably
    never emits E501 and every ``# noqa: E501`` in the tree is dead.  Reading
    ``ignore`` as well as ``select`` is the same tomllib read and is strictly
    more honest."""
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    assert _consumer_of(tmp_path, codes=('E501',)) is Consumer.NONE


def test_ignore_is_read_from_its_own_table_when_select_is_declared_elsewhere(tmp_path: Path):
    """Ruff honours a DEPRECATED top-level lint key unless ``[tool.ruff.lint]``
    declares that same key, so the two resolve independently.

    Taking both off whichever table happened to declare ``select`` errs in the
    lenient direction and so is not a gate hole — but it defeats precisely the
    case reading ``ignore`` exists for, telling an author to write a disposition
    for a marker no tool will ever read.  No pyproject in this repository splits
    the keys today, which is why the behaviour needs a fixture rather than the
    live tree to hold it.

    The second assertion is the other half: resolving ``ignore`` independently
    must not cost the ``select`` beside it.
    """
    write_files(
        tmp_path,
        {
            'pkg/mod.py': 'x = 1\n',
            'pyproject.toml': '[tool.ruff]\nignore = ["E402"]\n[tool.ruff.lint]\nselect = ["E"]\n',
        },
    )

    assert _consumer_of(tmp_path, codes=('E402',)) is Consumer.NONE
    assert _consumer_of(tmp_path, codes=('E731',)) is Consumer.RUFF


def test_a_bare_noqa_is_consumed_when_the_nearest_config_selects_anything(tmp_path: Path):
    """A bare ``# noqa`` silences whatever ruff would have said, so it is
    consumed exactly when ruff has something to say at all."""
    write_files(
        tmp_path,
        {
            'pkg/mod.py': 'x = 1\n',
            'pyproject.toml': RUFF_CONFIG,
            'bare/mod.py': 'x = 1\n',
            'bare/pyproject.toml': '[tool.ruff.lint]\nselect = []\n',
        },
    )

    assert _consumer_of(tmp_path) is Consumer.RUFF
    assert _consumer_of(tmp_path, path='bare/mod.py') is Consumer.NONE


def test_the_nearest_pyproject_wins_and_is_not_merged_with_the_root(tmp_path: Path):
    """Ruff takes the NEAREST applicable config without merging — stated in
    this repository's own root ``pyproject.toml``, and modelled here."""
    write_files(
        tmp_path,
        {
            'pyproject.toml': RUFF_CONFIG,
            'member/pyproject.toml': '[tool.ruff.lint]\nselect = ["ANN"]\n',
            'member/mod.py': 'x = 1\n',
            'top.py': 'x = 1\n',
        },
    )

    assert _consumer_of(tmp_path, path='member/mod.py', codes=('ANN001',)) is (
        Consumer.RUFF
    )
    assert _consumer_of(tmp_path, path='member/mod.py', codes=('E402',)) is (
        Consumer.NONE
    )
    assert _consumer_of(tmp_path, path='top.py', codes=('E402',)) is (
        Consumer.RUFF
    )


def test_a_pyproject_with_no_ruff_section_is_skipped_and_the_walk_continues(tmp_path: Path):
    """Ruff skips a ``pyproject.toml`` carrying no ``[tool.ruff]`` at all, so a
    packaging-only manifest must not shadow the config above it."""
    write_files(
        tmp_path,
        {
            'pyproject.toml': RUFF_CONFIG,
            'member/pyproject.toml': '[project]\nname = "member"\nversion = "0"\n',
            'member/mod.py': 'x = 1\n',
        },
    )

    assert _consumer_of(tmp_path, path='member/mod.py', codes=('E402',)) is (
        Consumer.RUFF
    )


def test_a_file_with_no_pyproject_above_it_has_no_ruff_consumer(tmp_path: Path):
    """The walk stops at the scan ROOT, never climbing out of the tree under
    measurement — otherwise a scan of a fixture tree would silently read this
    repository's own config."""
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n'})

    assert _consumer_of(tmp_path, codes=('E402',)) is Consumer.NONE


@pytest.mark.parametrize(
    ('key', 'config'),
    [
        ('extend-select', '[tool.ruff.lint]\nselect = ["E"]\nextend-select = ["ANN"]\n'),
        ('extend', '[tool.ruff]\nextend = "../shared-ruff.toml"\n[tool.ruff.lint]\nselect = ["E"]\n'),
    ],
)
def test_a_config_key_that_could_widen_the_selected_set_is_an_instrument_failure(
    tmp_path: Path, key: str, config: str
):
    """THE SPLIT IS BY DIRECTION OF ERROR, which is the only thing that matters
    for a gate.

    Under-reading the selected set makes the scanner reject a marker ruff
    genuinely honours — a false red on a legitimate suppression, the expensive
    failure — so it refuses to guess rather than proceeding on a config it does
    not fully model.  Exit 2, naming the file AND the key.

    ``extend`` is the INHERITANCE case, and it widens by the same arithmetic
    from another file: the inherited config carries its own ``select`` /
    ``extend-select``, so the list read here is a subset of the rules ruff
    actually runs.  Each key is written in the table it may legally appear in,
    and the refusal names that table.
    """
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': config})

    with pytest.raises(InstrumentFailure) as caught:
        _consumer_of(tmp_path, codes=('E402',))

    assert 'pyproject.toml' in str(caught.value)
    assert repr(key) in str(caught.value)


def test_config_keys_that_only_ever_subtract_are_tolerated(tmp_path: Path):
    """Over-reading the selected set only grandfathers a dead marker — the
    cheap failure — so it is tolerated with the reason recorded rather than
    modelled.

    ``per-file-ignores`` is the concrete case: present only in
    ``orchestrator/pyproject.toml`` (``tests/**/*.py: ["F811"]``), affecting 13
    grandfathered markers, and modelling it would need path-glob machinery for
    no change in the direction that can hurt.
    """
    write_files(
        tmp_path,
        {
            'pkg/mod.py': 'x = 1\n',
            'pyproject.toml': (
                '[tool.ruff.lint]\n'
                'select = ["E", "F"]\n'
                'extend-ignore = ["E731"]\n'
                '[tool.ruff.lint.per-file-ignores]\n'
                '"tests/**/*.py" = ["F811"]\n'
            ),
        },
    )

    assert _consumer_of(tmp_path, codes=('E402',)) is Consumer.RUFF


def test_a_ruff_section_that_declares_no_select_is_an_instrument_failure(tmp_path: Path):
    """The same direction-of-error rule as ``extend-select``, applied to an
    ABSENT key rather than an unmodelled one.

    A ``[tool.ruff]`` section with no ``select`` anywhere does not mean "ruff
    checks nothing" — it means ruff applies its BUILT-IN default rule set,
    which is wider than the nothing this model would otherwise infer and which
    drifts with the ruff version.  Reading it as empty is precisely the
    under-read that rejects a marker ruff genuinely honours.  This repository's
    own root ``pyproject.toml`` records having been in that state, with two
    gated directories reporting "All checks passed!" while running a rule set
    nobody chose.

    Nothing in this tree reaches the refusal: all eight pyprojects declare
    ``select`` explicitly.
    """
    write_files(
        tmp_path,
        {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': '[tool.ruff]\nline-length = 100\n'},
    )

    with pytest.raises(InstrumentFailure) as caught:
        _consumer_of(tmp_path, codes=('E402',))

    assert 'pyproject.toml' in str(caught.value)
    assert 'select' in str(caught.value)


# ---------------------------------------------------------------------------
# D8, first-party half — the codes no ruff selector can ever match.

#: The checker whose private rule-code constants this scanner necessarily
#: copies.  Its path is named once here, and the guard below runs it.
_FIRST_PARTY_CHECKER = REPO_ROOT / 'fused-memory' / 'scripts' / 'check_bare_magicmock_config.py'

#: One fixture per rule, each written to trigger exactly one of them, so the
#: guard below sees every code the checker can name.  Kept as SOURCE rather
#: than as a list of expected codes: the point is to read the codes back out of
#: what the checker emits, never to restate them.
_FIRST_PARTY_TRIGGERS: Mapping[str, str] = {
    'rule_a.py': 'from unittest.mock import MagicMock\n\nconfig = MagicMock()\n',
    'rule_b.py': (
        'from unittest.mock import MagicMock\n\ndouble = MagicMock(passed=True, summary=\'ok\')\n'
    ),
    'rule_c.py': (
        'import asyncio\n\n\nasync def run(req):\n'
        '    return await asyncio.wait_for(req.result, timeout=5)\n'
    ),
}


def test_a_first_party_code_resolves_to_the_first_party_consumer(tmp_path: Path):
    """These are not ruff codes at all, so no selector can ever match them —
    and the ``(linter, number)`` split must tolerate a kebab-case code, which
    has no numeric run."""
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    for code in FIRST_PARTY_CODES:
        assert _consumer_of(tmp_path, codes=(code,)) is (
            Consumer.FIRST_PARTY
        ), code


def test_an_invented_neighbouring_code_has_no_consumer(tmp_path: Path):
    """The table is a closed set, not a kebab-case shape test: a code that
    merely LOOKS like a first-party one is read by nobody."""
    write_files(tmp_path, {'pkg/mod.py': 'x = 1\n', 'pyproject.toml': RUFF_CONFIG})

    assert _consumer_of(tmp_path, codes=('bare-something-else',)) is (
        Consumer.NONE
    )


def test_the_first_party_table_matches_the_codes_that_checker_actually_emits(tmp_path: Path):
    """THE DRIFT GUARD, and it is BEHAVIOURAL IN BOTH DIRECTIONS.

    The source of truth is a private constant in another package's
    stdlib-only script, so there is no public seam to import and a second copy
    is unavoidable.  Heuristic 11 then asks that the copy's drift be made loud
    rather than that the copy be hidden.

    It asserts on the violation messages the checker EMITS, never on its
    private ``_RULE_A_CODE`` / ``_RULE_B_CODE`` / ``_RULE_C_CODE`` — reading
    those would pin implementation rather than behaviour
    (``docs/code-quality.md``, Tests stance), and the emitted message is
    strictly the stronger assertion: a checker that renamed its constant while
    still emitting the old code would remain correctly modelled.

    SET EQUALITY, not containment, and that direction is load-bearing.  A
    containment check ("every code in our table is real") passes happily while
    the table MISSES a code the checker honours — which is exactly the defect
    this test found: ``wall-clock-deadline`` (Rule C, task 4246) is a live
    first-party code that both the PRD's D8 prose and this task's plan omit.
    Missing a code is the expensive direction: the scanner would tell an author
    to delete a marker a live checker genuinely reads.
    """
    write_files(tmp_path, _FIRST_PARTY_TRIGGERS)

    emitted = set()
    for name in _FIRST_PARTY_TRIGGERS:
        completed = subprocess.run(
            [sys.executable, str(_FIRST_PARTY_CHECKER), str(tmp_path / name)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECS,
        )
        emitted.update(re.findall(r'#\s*noqa:\s*([a-z0-9]+(?:-[a-z0-9]+)+)', completed.stdout))

    assert emitted, 'the checker emitted no suppression remedy at all — fixtures stopped triggering'
    assert emitted == set(FIRST_PARTY_CODES)
