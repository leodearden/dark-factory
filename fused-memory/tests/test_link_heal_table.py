"""The H1 action table and its write images, pure (task 6181).

Contract: plans/write-triage-link-healing-prd.md H1. Rows are evaluated top to
bottom and the first match wins; the regression class at the bottom pins that
the table reproduces PRD §6's counts over the committed hand-link corpus.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fused_memory.maintenance.link_heal import (
    BasisSource,
    CorpusFormatError,
    HealAction,
    KindClass,
    LinkBasis,
    LinkImage,
    LinkState,
    ParentPresence,
    Report,
    Verdict,
    change_for,
    decide,
    journal_reason,
    kind_class,
    load_corpus_bases,
    post_image_for,
    undo_changes,
)
from fused_memory.maintenance.link_heal_store import LiveRecord, MetadataChange, text_sha256
from fused_memory.server.grouped_read import (
    AMENDMENT_KIND,
    CONTESTED_METADATA_KEY,
    PARENT_ID_KEY,
    SIGHTING_KIND,
)

CORPUS = Path(__file__).resolve().parent.parent / 'calibration' / 'hand_link_verdicts.jsonl'

PROJECT = 'dark_factory'
CHILD = '11111111-1111-1111-1111-111111111111'
PARENT = '22222222-2222-2222-2222-222222222222'
CHILD_TEXT = 'the child note'
PARENT_TEXT = 'the parent note'


def _state(
    kind: str | None = AMENDMENT_KIND,
    *,
    contested: Any = None,
    presence: ParentPresence = ParentPresence.FOUND,
    parent_meta: dict[str, Any] | None = None,
    has_children: bool = False,
    child_text: str = CHILD_TEXT,
    parent_text: str = PARENT_TEXT,
) -> LinkState:
    meta: dict[str, Any] = {PARENT_ID_KEY: PARENT}
    if kind is not None:
        meta['kind'] = kind
    if contested is not None:
        meta[CONTESTED_METADATA_KEY] = contested
    parent = (
        LiveRecord(memory_id=PARENT, text=parent_text, metadata=parent_meta or {})
        if presence is ParentPresence.FOUND
        else None
    )
    return LinkState(
        project_id=PROJECT,
        child=LiveRecord(memory_id=CHILD, text=child_text, metadata=meta),
        parent_presence=presence,
        parent=parent,
        has_children=has_children,
    )


def _basis(
    verdict: str,
    *,
    child_text: str = CHILD_TEXT,
    parent_text: str = PARENT_TEXT,
    matches: bool = True,
) -> LinkBasis:
    return LinkBasis(
        project_id=PROJECT,
        child_id=CHILD,
        parent_id=PARENT,
        verdict=Verdict(verdict),
        child_sha256=text_sha256(child_text),
        parent_sha256=text_sha256(parent_text),
        source=BasisSource.CORPUS,
        key='H000',
        rated_text_matches_live=matches,
    )


class TestDecideRowByRow:
    def test_a_stale_basis_is_stale_rating(self):
        assert decide(_state(), _basis('RELATED', matches=False)).outcome is Report.STALE_RATING

    def test_a_contested_child_is_reported(self):
        assert decide(_state(contested=True), _basis('EXTENDS')).outcome is (
            Report.CONTESTED_REPORTED
        )

    def test_a_parent_absent_everywhere_is_a_deterministic_detach(self):
        decision = decide(_state(presence=ParentPresence.ABSENT), None)
        assert decision.outcome is HealAction.DETACH
        assert decision.basis_source is BasisSource.DETERMINISTIC

    def test_a_parent_only_in_another_run_project_is_reported(self):
        decision = decide(_state(presence=ParentPresence.OTHER_PROJECT), _basis('RELATED'))
        assert decision.outcome is Report.CROSS_PROJECT_REPORTED

    def test_a_parent_that_is_itself_a_link_is_a_chain(self):
        state = _state(parent_meta={PARENT_ID_KEY: '33333333-3333-3333-3333-333333333333'})
        assert decide(state, _basis('EXTENDS')).outcome is Report.CHAIN_REPORTED

    @pytest.mark.parametrize('verdict', ['RELATED', 'UNRELATED'])
    @pytest.mark.parametrize(
        'kind', [SIGHTING_KIND, AMENDMENT_KIND, 'peer', None, 'correction'],
    )
    def test_a_misfile_verdict_detaches_every_kind_class(self, kind, verdict):
        decision = decide(_state(kind), _basis(verdict))
        assert decision.outcome is HealAction.DETACH
        assert decision.basis_source is BasisSource.CORPUS

    def test_unclear_is_reported(self):
        assert decide(_state(), _basis('UNCLEAR')).outcome is Report.UNCLEAR

    @pytest.mark.parametrize(
        ('kind', 'verdict', 'outcome'),
        [
            (SIGHTING_KIND, 'EXTENDS', HealAction.RELABEL),
            (SIGHTING_KIND, 'CORRECTS', HealAction.RELABEL_FLAG),
            (SIGHTING_KIND, 'SAME', Report.NO_ACTION),
            (SIGHTING_KIND, 'SUBSUMED', Report.NO_ACTION),
            (AMENDMENT_KIND, 'CORRECTS', HealAction.FLAG),
            (AMENDMENT_KIND, 'SAME', Report.NO_ACTION),
            (AMENDMENT_KIND, 'SUBSUMED', Report.NO_ACTION),
            (AMENDMENT_KIND, 'EXTENDS', Report.NO_ACTION),
            ('peer', 'SAME', Report.PEER_REPORTED),
            ('peer', 'EXTENDS', Report.PEER_REPORTED),
            ('peer', 'SUBSUMED', Report.PEER_REPORTED),
            ('peer', 'CORRECTS', Report.PEER_REPORTED),
            (None, 'EXTENDS', HealAction.COMPLETE_AMENDMENT),
            (None, 'CORRECTS', HealAction.COMPLETE_AMENDMENT_FLAG),
            (None, 'SAME', HealAction.COMPLETE_SIGHTING),
            (None, 'SUBSUMED', HealAction.COMPLETE_SIGHTING),
        ],
    )
    def test_the_kind_by_verdict_rows(self, kind, verdict, outcome):
        assert decide(_state(kind), _basis(verdict)).outcome is outcome

    @pytest.mark.parametrize('verdict', ['SAME', 'EXTENDS', 'SUBSUMED', 'CORRECTS'])
    def test_a_half_link_with_children_is_reported(self, verdict):
        decision = decide(_state(None, has_children=True), _basis(verdict))
        assert decision.outcome is Report.HAS_CHILDREN

    def test_a_link_reaching_the_verdict_rows_unrated_is_unexamined(self):
        assert decide(_state(), None).outcome is Report.UNEXAMINED


class TestDecidePrecedence:
    def test_contested_outranks_a_misfile(self):
        assert decide(_state(contested=True), _basis('RELATED')).outcome is (
            Report.CONTESTED_REPORTED
        )

    def test_a_chain_outranks_a_misfile(self):
        state = _state(parent_meta={PARENT_ID_KEY: '33333333-3333-3333-3333-333333333333'})
        assert decide(state, _basis('RELATED')).outcome is Report.CHAIN_REPORTED

    def test_a_misfile_outranks_having_children(self):
        decision = decide(_state(None, has_children=True), _basis('RELATED'))
        assert decision.outcome is HealAction.DETACH

    @pytest.mark.parametrize('kind', ['correction', 'extension', 'child_amendment', None])
    def test_agent_invented_kinds_class_as_half_links(self, kind):
        assert kind_class(kind) is KindClass.HALF_LINK

    def test_peer_classes_as_peer(self):
        assert kind_class('peer') is KindClass.PEER


class TestStaleness:
    def test_an_edited_child_is_stale(self):
        state = _state(child_text='edited after rating')
        assert decide(state, _basis('RELATED')).outcome is Report.STALE_RATING

    def test_an_edited_parent_is_stale(self):
        state = _state(parent_text='edited after rating')
        assert decide(state, _basis('RELATED')).outcome is Report.STALE_RATING

    def test_an_absent_parent_with_a_current_child_is_a_detach_not_stale(self):
        decision = decide(_state(presence=ParentPresence.ABSENT), _basis('EXTENDS'))
        assert decision.outcome is HealAction.DETACH
        assert decision.basis_source is BasisSource.DETERMINISTIC


def _image(kind: str | None, *, contested: Any = None, parent: str | None = PARENT) -> LinkImage:
    return LinkImage(parent_id=parent, kind=kind, contested=contested)


class TestChangeFor:
    @pytest.mark.parametrize('kind', [AMENDMENT_KIND, SIGHTING_KIND])
    def test_detaching_a_child_kind_deletes_parent_and_kind(self, kind):
        assert change_for(HealAction.DETACH, _image(kind)) == MetadataChange.delete_only(
            [PARENT_ID_KEY, 'kind'],
        )

    @pytest.mark.parametrize('kind', ['extension', None])
    def test_detaching_a_half_link_keeps_its_kind(self, kind):
        assert change_for(HealAction.DETACH, _image(kind)) == MetadataChange.delete_only(
            [PARENT_ID_KEY],
        )

    @pytest.mark.parametrize(
        ('action', 'pre_kind', 'patch'),
        [
            (HealAction.RELABEL, SIGHTING_KIND, {'kind': AMENDMENT_KIND}),
            (
                HealAction.RELABEL_FLAG, SIGHTING_KIND,
                {'kind': AMENDMENT_KIND, CONTESTED_METADATA_KEY: True},
            ),
            (
                HealAction.COMPLETE_AMENDMENT_FLAG, None,
                {'kind': AMENDMENT_KIND, CONTESTED_METADATA_KEY: True},
            ),
            (HealAction.FLAG, AMENDMENT_KIND, {CONTESTED_METADATA_KEY: True}),
            (HealAction.COMPLETE_AMENDMENT, 'correction', {'kind': AMENDMENT_KIND}),
            (HealAction.COMPLETE_SIGHTING, None, {'kind': SIGHTING_KIND}),
        ],
    )
    def test_every_other_action_is_patch_only(self, action, pre_kind, patch):
        assert change_for(action, _image(pre_kind)) == MetadataChange.patch_only(patch)

    def test_a_post_image_is_the_pre_image_with_the_change_applied(self):
        assert post_image_for(HealAction.RELABEL_FLAG, _image(SIGHTING_KIND)) == _image(
            AMENDMENT_KIND, contested=True,
        )

    def test_keys_absent_from_a_post_image_are_absent_never_none(self):
        post = post_image_for(HealAction.DETACH, _image(AMENDMENT_KIND))
        assert post == LinkImage(parent_id=None, kind=None, contested=None)
        assert post.as_dict() == {}

    def test_a_detached_half_link_keeps_its_kind_in_the_post_image(self):
        post = post_image_for(HealAction.DETACH, _image('extension'))
        assert post.as_dict() == {'kind': 'extension'}


class TestUndoChanges:
    def test_undoing_a_detach_restores_parent_and_kind_in_one_patch(self):
        pre = _image(AMENDMENT_KIND)
        post = post_image_for(HealAction.DETACH, pre)
        assert undo_changes(post, pre) == [
            MetadataChange.patch_only({PARENT_ID_KEY: PARENT, 'kind': AMENDMENT_KIND}),
        ]

    def test_undoing_a_no_kind_completion_deletes_the_kind(self):
        pre = _image(None)
        post = post_image_for(HealAction.COMPLETE_SIGHTING, pre)
        assert undo_changes(post, pre) == [MetadataChange.delete_only(['kind'])]

    def test_undoing_a_relabel_and_flag_deletes_first_then_patches(self):
        pre = _image(SIGHTING_KIND)
        post = post_image_for(HealAction.RELABEL_FLAG, pre)
        assert undo_changes(post, pre) == [
            MetadataChange.delete_only([CONTESTED_METADATA_KEY]),
            MetadataChange.patch_only({'kind': SIGHTING_KIND}),
        ]

    @pytest.mark.parametrize('action', list(HealAction))
    @pytest.mark.parametrize('pre_kind', [AMENDMENT_KIND, SIGHTING_KIND, None, 'correction'])
    def test_no_change_ever_carries_a_none_value(self, action, pre_kind):
        pre = _image(pre_kind)
        changes = [change_for(action, pre), *undo_changes(post_image_for(action, pre), pre)]
        for change in changes:
            assert None not in (change.patch or {}).values()


class TestJournalReason:
    def test_reads_as_a_pointer(self):
        reason = journal_reason('abcd1234', HealAction.DETACH, _image(AMENDMENT_KIND))
        assert reason == (
            f'link-heal r=abcd1234 a=detach prev_parent={PARENT} prev_kind=amendment'
        )

    def test_an_absent_kind_reads_none(self):
        reason = journal_reason('abcd1234', HealAction.COMPLETE_SIGHTING, _image(None))
        assert reason.endswith('prev_kind=none')

    def test_stays_under_200_characters_for_an_invented_500_char_kind(self):
        reason = journal_reason(
            'abcd1234', HealAction.COMPLETE_AMENDMENT_FLAG, _image('k' * 500),
        )
        assert len(reason) < 200
        assert reason.startswith('link-heal r=abcd1234 a=complete_amendment_flag')


class TestLoadCorpusBases:
    def test_loads_every_committed_row_as_a_corpus_basis(self):
        bases = load_corpus_bases(CORPUS)

        assert len(bases) == 359
        assert {basis.source for basis in bases} == {BasisSource.CORPUS}
        first = json.loads(CORPUS.read_text().splitlines()[0])
        basis = next(b for b in bases if b.key == first['item_id'])
        assert basis == LinkBasis(
            project_id=first['project'],
            child_id=first['entry_id'],
            parent_id=first['target_id'],
            verdict=Verdict(first['verdict']),
            child_sha256=first['child_sha256'],
            parent_sha256=first['parent_sha256'],
            source=BasisSource.CORPUS,
            key=first['item_id'],
            rated_text_matches_live=first['rated_text_matches_live'],
        )

    def _write_rows(self, tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
        path = tmp_path / 'corpus.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def _rows(self, n: int) -> list[dict[str, Any]]:
        return [json.loads(line) for line in CORPUS.read_text().splitlines()[:n]]

    def test_an_unknown_verdict_word_names_the_row(self, tmp_path):
        rows = self._rows(2)
        rows[1]['verdict'] = 'BELONGS'
        with pytest.raises(CorpusFormatError, match=rows[1]['item_id']):
            load_corpus_bases(self._write_rows(tmp_path, rows))

    def test_a_missing_hash_names_the_row(self, tmp_path):
        rows = self._rows(2)
        del rows[1]['parent_sha256']
        with pytest.raises(CorpusFormatError, match=rows[1]['item_id']):
            load_corpus_bases(self._write_rows(tmp_path, rows))

    def test_a_duplicate_link_names_the_row(self, tmp_path):
        rows = self._rows(2)
        rows[1] = {**rows[0], 'item_id': 'H999'}
        with pytest.raises(CorpusFormatError, match='H999'):
            load_corpus_bases(self._write_rows(tmp_path, rows))


class TestCorpusRegressionAgainstPrdSection6:
    """Applying the table to the corpus at its export state reproduces PRD §6.

    Each row's export-time state comes from its own fields: its live kind and
    contested flag, its parent present, no chain, no children, and texts that
    match its basis hashes (synthetic texts, with the basis hashes rebased onto
    them, so the rating is current exactly as it was at export).
    """

    EXPECTED = {
        HealAction.DETACH: 15,
        HealAction.RELABEL: 66,
        HealAction.RELABEL_FLAG: 14,
        HealAction.FLAG: 64,
        HealAction.COMPLETE_AMENDMENT: 32,
        HealAction.COMPLETE_AMENDMENT_FLAG: 31,
        HealAction.COMPLETE_SIGHTING: 11,
        Report.PEER_REPORTED: 4,
        Report.CONTESTED_REPORTED: 1,
        Report.NO_ACTION: 121,
    }

    def _export_state(self, row: dict[str, Any], basis: LinkBasis) -> tuple[LinkState, LinkBasis]:
        child_text, parent_text = f'child {basis.key}', f'parent {basis.key}'
        meta: dict[str, Any] = {PARENT_ID_KEY: row['live_parent_id']}
        if row['live_kind'] is not None:
            meta['kind'] = row['live_kind']
        if row['live_contested']:
            meta[CONTESTED_METADATA_KEY] = True
        state = LinkState(
            project_id=row['project'],
            child=LiveRecord(memory_id=row['entry_id'], text=child_text, metadata=meta),
            parent_presence=ParentPresence.FOUND,
            parent=LiveRecord(memory_id=row['target_id'], text=parent_text, metadata={}),
            has_children=False,
        )
        current = replace(
            basis,
            child_sha256=text_sha256(child_text),
            parent_sha256=text_sha256(parent_text),
        )
        return state, current

    def test_reproduces_the_prd_counts(self):
        rows = {
            row['item_id']: row
            for row in (json.loads(line) for line in CORPUS.read_text().splitlines())
        }
        counts: Counter = Counter()
        for basis in load_corpus_bases(CORPUS):
            state, current = self._export_state(rows[basis.key], basis)
            counts[decide(state, current).outcome] += 1

        assert dict(counts) == self.EXPECTED
        assert sum(counts.values()) == 359
