"""A write is near-verbatim to a reference only when it adds nothing to it.

Near-verbatim means the same words and symbols in the same order: only case and
whitespace may differ. A high character similarity is not the test. One changed
number, a dropped ``not``, a flipped operator or two swapped numbers leave a
long text almost character-identical, and each changes what the text says.
Every text here is synthetic.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator

import pytest
from _write_triage_calibration_fixtures import CALIBRATION_FIXTURES, FIXTURES_DIR

from fused_memory.server.near_verbatim import is_near_verbatim

_LONG_ENTRY = (
    'The nightly verify of the ledger service finishes in 14m22s on the shared '
    'build host when the warm cache is present, and the cold path takes roughly '
    'twice as long because every wheel is rebuilt from source. The retry loop '
    'backs off while x > 5, where x is the depth of the pending queue, and it '
    'resumes as soon as the queue drains below that mark. The clock skew '
    'correction applies an offset of -5 seconds to every journal timestamp '
    'before the merge worker compares it with the remote head. The pre-commit '
    'hook does not run the module checks on docs-only commits, so a commit that '
    'touches only markdown lands in a few seconds. Operators who need to rerun '
    'the gate by hand should invoke the screen script from the checkout root, '
    'never from a package directory, since the script resolves its config '
    'relative to the root. A failed run leaves its log under the task scratch '
    'directory, and the next run overwrites it. The uploader retries 3 times '
    'with a 5 second wait, and on the release branch the merge worker runs '
    'before the hook.'
)
_PREPENDED_CORRECTION = 'Correction to the earlier entry: the flag is no longer unset. '


def _with(old: str, new: str) -> str:
    assert _LONG_ENTRY.count(old) == 1, f'{old!r} must occur exactly once'
    return _LONG_ENTRY.replace(old, new)


class TestAnUnchangedTextIsNearVerbatim:

    def test_an_identical_text_is_near_verbatim(self) -> None:
        assert is_near_verbatim(_LONG_ENTRY, _LONG_ENTRY)

    def test_case_and_whitespace_runs_do_not_count_as_changes(self) -> None:
        reflowed = _LONG_ENTRY.upper().replace('. ', '.\n\n').replace(', ', ',   ')

        assert is_near_verbatim(reflowed, _LONG_ENTRY)

    def test_spacing_beside_a_symbol_does_not_count_as_a_change(self) -> None:
        assert is_near_verbatim(_with('x > 5', 'x>5'), _LONG_ENTRY)


class TestATextThatSaysSomethingNewIsNotNearVerbatim:

    def test_a_changed_measurement_is_not_near_verbatim(self) -> None:
        assert not is_near_verbatim(_with('14m22s', '10m57s'), _LONG_ENTRY)

    def test_a_prepended_correction_is_not_near_verbatim(self) -> None:
        corrected = _PREPENDED_CORRECTION + _LONG_ENTRY

        assert not is_near_verbatim(corrected, _LONG_ENTRY)

    @pytest.mark.parametrize(
        ('old', 'new'),
        [
            pytest.param('does not run', 'does run', id='dropped-not'),
            pytest.param('x > 5', 'x < 5', id='flipped-operator'),
            pytest.param('offset of -5', 'offset of 5', id='dropped-sign'),
        ],
    )
    def test_a_one_token_change_of_meaning_is_not_near_verbatim(
        self, old: str, new: str,
    ) -> None:
        assert not is_near_verbatim(_with(old, new), _LONG_ENTRY)

    @pytest.mark.parametrize(
        ('old', 'new'),
        [
            pytest.param(
                'retries 3 times with a 5 second wait',
                'retries 5 times with a 3 second wait',
                id='swapped-numbers',
            ),
            pytest.param(
                'the merge worker runs before the hook',
                'the hook runs before the merge worker',
                id='swapped-actors',
            ),
        ],
    )
    def test_a_reordering_of_the_same_tokens_is_not_near_verbatim(
        self, old: str, new: str,
    ) -> None:
        assert not is_near_verbatim(_with(old, new), _LONG_ENTRY)

    def test_a_long_text_with_its_words_shuffled_is_not_near_verbatim(self) -> None:
        words = (_LONG_ENTRY + ' ') * 40
        shuffled = words.split()
        random.Random(7155).shuffle(shuffled)

        assert not is_near_verbatim(' '.join(shuffled), words)

    def test_a_paraphrase_is_not_near_verbatim(self) -> None:
        paraphrase = (
            'On the shared host the ledger nightly verify takes about fourteen '
            'minutes with a warm cache and twice that when it has to rebuild.'
        )

        assert not is_near_verbatim(paraphrase, _LONG_ENTRY)


def _calibration_pairs() -> Iterator[tuple[str, str, str]]:
    """Every non-canonical record with its own cluster's canonical text."""
    for project, path in CALIBRATION_FIXTURES.items():
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        canonical_text = {
            record['cluster_id']: record['content']
            for record in records if record['label'] == 'canonical'
        }
        for record in records:
            if record['label'] != 'canonical':
                yield (
                    f"{project}:{record['memory_id']}",
                    record['content'],
                    canonical_text[record['cluster_id']],
                )


def _rated_seed_pairs() -> list[tuple[str, str, str]]:
    """The rated seed pairs whose two records both have committed texts."""
    text_by_id = {}
    for path in CALIBRATION_FIXTURES.values():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                text_by_id[record['memory_id']] = record['content']
    seed_path = FIXTURES_DIR / 'write_triage_pair_verdicts_seed.jsonl'
    rated = [json.loads(line) for line in seed_path.read_text().splitlines() if line.strip()]
    return [
        (f"{row['entry_id']}->{row['target_id']}", text_by_id[row['entry_id']], text_by_id[row['target_id']])
        for row in rated
        if row['entry_id'] in text_by_id and row['target_id'] in text_by_id
    ]


class TestNoCommittedDuplicateIsNearVerbatim:

    def test_no_committed_corpus_pair_or_rated_pair_is_near_verbatim(self) -> None:
        calibration_pairs = list(_calibration_pairs())
        rated_pairs = _rated_seed_pairs()

        assert calibration_pairs, 'no calibration corpus pair was read'
        assert rated_pairs, 'no rated seed pair joined to a committed text'
        near_verbatim = [
            pair_id
            for pair_id, text, reference in calibration_pairs + rated_pairs
            if is_near_verbatim(text, reference)
        ]
        assert near_verbatim == []
