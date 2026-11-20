"""A Claude Code sub-agent transcript is its own legibility session.

Claude Code persists an Agent-tool sub-agent at
``<enc>/<parent sid>/subagents/agent-<agentId>.jsonl`` beside a
``.meta.json`` sibling. Every record in it is ``isSidechain: true`` and
carries ``agentId`` plus its PARENT's ``sessionId``; its first record is a
``user`` turn whose text is the parent agent's prompt. The parent transcript
embeds none of those records. The fixture builder below writes exactly that
measured shape.

The contract spans inventory (enumeration, identity), sampling (the session
key the ledger skip compares), detectors (agent class, first turn) and digest
(frontmatter, counted signals), so it is read here in one place, through each
module's public seam over real tmp-dir trees.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import date as dt_date
from pathlib import Path
from typing import Any

import pytest
import yaml
from legibility import detectors, digest, inventory, reingestion, sampling

MAIN_CWD = '/home/leo/src/dark-factory'
MAIN_ENC = '-home-leo-src-dark-factory'
WORKTREE_CWD = '/home/leo/src/dark-factory/.worktrees/2573'
WORKTREE_ENC = '-home-leo-src-dark-factory--worktrees-2573'
COCKPIT_CWD = '/home/leo/src/dark-factory-cockpit'

TARGET_DATE = dt_date(2026, 7, 13)
TARGET_TS = '2026-07-13T10:00:00.000Z'
OTHER_DAY_TS = '2026-07-12T10:00:00.000Z'

PARENT_SID = '298d816e-484a-4675-ad8d-9549c8b89b08'
AGENT_ID = 'a85ff34487a904824'
DEFAULT_PROMPT = 'Find every caller of resolve_route and report their files.'


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    return path


def _write_parent_and_subagent(
    enc_dir: Path,
    *,
    parent_sid: str = PARENT_SID,
    agent_id: str = AGENT_ID,
    cwd: str = MAIN_CWD,
    timestamp: str = TARGET_TS,
    prompt: str = DEFAULT_PROMPT,
    subagent_records: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    """Write a parent transcript and one of its sub-agents; return ``(parent, subagent)``.

    *subagent_records* are the sub-agent's records AFTER its prompt record;
    each is stamped with the sidechain, agentId, sessionId, cwd and timestamp
    every live sub-agent record carries, unless it sets them itself.
    """
    parent_path = _write_jsonl(
        enc_dir / f'{parent_sid}.jsonl',
        [
            {
                'type': 'user', 'isSidechain': False, 'sessionId': parent_sid,
                'cwd': cwd, 'timestamp': timestamp, 'parentUuid': None,
                'message': {'role': 'user', 'content': 'please spawn a helper'},
            },
            {
                'type': 'assistant', 'isSidechain': False, 'sessionId': parent_sid,
                'cwd': cwd, 'timestamp': timestamp,
                'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'on it'}]},
            },
        ],
    )

    shared = {
        'isSidechain': True, 'agentId': agent_id, 'sessionId': parent_sid,
        'cwd': cwd, 'timestamp': timestamp,
    }
    prompt_record = {
        'type': 'user', 'parentUuid': None, **shared,
        'message': {'role': 'user', 'content': prompt},
    }
    later = subagent_records if subagent_records is not None else [
        {
            'type': 'assistant',
            'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'found two callers'}]},
        },
    ]
    subagents_dir = enc_dir / parent_sid / 'subagents'
    subagent_path = _write_jsonl(
        subagents_dir / f'agent-{agent_id}.jsonl',
        [prompt_record, *({**shared, **record} for record in later)],
    )
    (subagents_dir / f'agent-{agent_id}.meta.json').write_text(
        json.dumps({'agentType': 'general-purpose', 'description': 'x'})
    )
    return parent_path, subagent_path


def _tool_error_record() -> dict[str, Any]:
    return {
        'type': 'user',
        'message': {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'is_error': True, 'content': 'boom'},
        ]},
    }


def _subagent_key(parent_sid: str = PARENT_SID, agent_id: str = AGENT_ID) -> str:
    return f'{parent_sid}/subagents/agent-{agent_id}'


def _record_for(records: list[inventory.SessionRecord], path: Path) -> inventory.SessionRecord:
    return next(record for record in records if record.path == path)


class TestTranscriptIdentity:
    def test_session_transcript_is_keyed_on_its_session_id(self):
        identity = inventory.transcript_identity({'sessionId': 'p'})
        assert identity is not None
        assert identity.key == 'p'
        assert identity.parent_session is None

    def test_subagent_transcript_is_keyed_under_its_parent(self):
        identity = inventory.transcript_identity({'sessionId': 'p', 'agentId': 'a1'})
        assert identity is not None
        assert identity.key == 'p/subagents/agent-a1'
        assert identity.parent_session == 'p'

    @pytest.mark.parametrize('record', [{}, {'sessionId': ''}, {'agentId': 'a1'}, {'sessionId': 7}])
    def test_record_without_a_session_id_has_no_identity(self, record):
        assert inventory.transcript_identity(record) is None

    def test_identity_is_frozen(self):
        identity = inventory.TranscriptIdentity(session_id='p')
        field = 'session_id'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(identity, field, 'q')


class TestProjectsRootEnumeratesSubagents:
    def _tree(self, tmp_path: Path, **kwargs: Any) -> tuple[Path, Path, Path]:
        projects_root = tmp_path / 'projects'
        parent, subagent = _write_parent_and_subagent(projects_root / MAIN_ENC, **kwargs)
        return projects_root, parent, subagent

    def test_enumerates_both_the_parent_and_its_subagent(self, tmp_path):
        projects_root, parent, subagent = self._tree(tmp_path)
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        assert {record.path for record in records} == {parent, subagent}

    def test_subagent_record_names_its_encoded_dir_cwd_and_parent(self, tmp_path):
        projects_root, _, subagent = self._tree(tmp_path)
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        record = _record_for(records, subagent)
        assert record.encoded_dir == MAIN_ENC
        assert record.cwd == MAIN_CWD
        assert record.identity is not None
        assert record.identity.parent_session == PARENT_SID
        assert record.identity.key == _subagent_key()

    def test_parent_record_is_keyed_on_its_own_session(self, tmp_path):
        projects_root, parent, _ = self._tree(tmp_path)
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        record = _record_for(records, parent)
        assert record.identity is not None
        assert record.identity.key == PARENT_SID
        assert record.identity.parent_session is None

    def test_meta_json_sibling_is_not_enumerated(self, tmp_path):
        projects_root, _, _ = self._tree(tmp_path)
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        assert not [record for record in records if record.path.suffix != '.jsonl']

    def test_non_member_subagent_under_a_candidate_dir_is_excluded(self, tmp_path):
        projects_root, _, _ = self._tree(tmp_path)
        _, foreign = _write_parent_and_subagent(
            projects_root / f'{MAIN_ENC}-cockpit', parent_sid='cockpit-parent', cwd=COCKPIT_CWD
        )
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        assert foreign not in {record.path for record in records}

    def test_subagent_on_another_date_is_excluded(self, tmp_path):
        projects_root, _, _ = self._tree(tmp_path)
        _, other_day = _write_parent_and_subagent(
            projects_root / MAIN_ENC, parent_sid='yesterday-parent', timestamp=OTHER_DAY_TS
        )
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        assert other_day not in {record.path for record in records}

    def test_range_enumeration_includes_the_subagent(self, tmp_path):
        projects_root, _, subagent = self._tree(tmp_path)
        records = inventory.enumerate_sessions_in_range(
            projects_root, [MAIN_CWD], dt_date(2026, 7, 10), dt_date(2026, 7, 14)
        )
        assert subagent in {record.path for record in records}


class TestArchivedSubagentIdentity:
    def test_archived_subagent_names_its_parent(self, tmp_path):
        archive = tmp_path / 'archive'
        _, subagent = _write_parent_and_subagent(
            archive / '2573' / WORKTREE_ENC, parent_sid='sid', agent_id='x', cwd=WORKTREE_CWD
        )
        records = inventory.enumerate_sessions(
            tmp_path / 'no-projects', [MAIN_CWD], TARGET_DATE, agent_transcript_roots=[archive]
        )
        record = _record_for(records, subagent)
        assert record.identity is not None
        assert record.identity.parent_session == 'sid'
        assert record.encoded_dir == WORKTREE_ENC


class TestSubagentSessionKey:
    """The session string the sampler reports, and the ledger skip compares, is the identity key."""

    def test_peek_reports_the_subagent_key(self, tmp_path):
        _, subagent = _write_parent_and_subagent(tmp_path / MAIN_ENC)
        assert sampling.peek_session_id(subagent) == _subagent_key()

    def test_scan_agrees_with_peek(self, tmp_path):
        _, subagent = _write_parent_and_subagent(tmp_path / MAIN_ENC)
        assert sampling.scan_transcript(subagent).session_id == _subagent_key()

    def test_parent_keeps_its_own_session_id(self, tmp_path):
        parent, _ = _write_parent_and_subagent(tmp_path / MAIN_ENC)
        assert sampling.peek_session_id(parent) == PARENT_SID

    def test_a_coded_parent_does_not_hide_its_subagent(self, tmp_path):
        _, subagent = _write_parent_and_subagent(tmp_path / MAIN_ENC)
        coded_sessions = frozenset({PARENT_SID})
        assert sampling.peek_session_id(subagent) not in coded_sessions

    def test_inventory_and_sampler_derive_one_key(self, tmp_path):
        projects_root = tmp_path / 'projects'
        _, subagent = _write_parent_and_subagent(projects_root / MAIN_ENC)
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        record = _record_for(records, subagent)
        assert record.identity is not None
        assert sampling.score_session(record).session_id == record.identity.key


class TestSubagentAgentClassAndFirstTurn:
    """A sub-agent's agent class and first turn come from its own encoded dir and prompt."""

    def test_worktree_subagent_is_orchestrated_like_its_parent(self, tmp_path):
        parent, subagent = _write_parent_and_subagent(
            tmp_path / 'projects' / WORKTREE_ENC, cwd=WORKTREE_CWD
        )
        for path in (parent, subagent):
            agent_class = detectors.classify_agent_class(path, inventory.iter_json_lines(path))
            assert agent_class is detectors.AgentClass.ORCHESTRATED_TASK, path

    def test_subagent_prompt_is_its_first_turn_for_the_classifier(self, tmp_path):
        prompt = f'{reingestion.TRICKLE_CODER_PROMPT_OPENER}. Code the digest below.'
        _, subagent = _write_parent_and_subagent(tmp_path / MAIN_ENC, prompt=prompt)
        agent_class = detectors.classify_agent_class(subagent, inventory.iter_json_lines(subagent))
        assert agent_class is detectors.AgentClass.LEGIBILITY_INSTRUMENT

    def test_scan_reports_the_subagent_prompt_as_first_turn(self, tmp_path):
        _, subagent = _write_parent_and_subagent(tmp_path / MAIN_ENC)
        assert sampling.scan_transcript(subagent).first_turn_text == DEFAULT_PROMPT

    def test_same_shape_subagents_with_different_prompts_both_survive_dedupe(self, tmp_path):
        projects_root = tmp_path / 'projects'
        _, first = _write_parent_and_subagent(
            projects_root / MAIN_ENC, agent_id='a1',
            prompt='Find every caller of resolve_route and report their files.',
            subagent_records=[_tool_error_record()],
        )
        _, second = _write_parent_and_subagent(
            projects_root / MAIN_ENC, agent_id='a2',
            prompt='Summarise why the merge queue halted overnight.',
            subagent_records=[_tool_error_record()],
        )
        records = inventory.enumerate_sessions(projects_root, [MAIN_CWD], TARGET_DATE)
        scored = [sampling.score_session(_record_for(records, path)) for path in (first, second)]
        survivors = sampling.dedupe_shapes(scored)
        assert {record.path for record in survivors} == {first, second}

    def test_sidechain_is_primary_only_in_a_subagent_transcript(self):
        sidechain = {'type': 'user', 'isSidechain': True}
        main_chain = {'type': 'user', 'isSidechain': False}
        assert detectors.is_primary_user_turn(sidechain) is False
        assert detectors.is_primary_user_turn(sidechain, subagent_transcript=True) is True
        assert detectors.is_primary_user_turn(main_chain, subagent_transcript=True) is False


def _frontmatter(rendered: str) -> dict[str, Any]:
    _, block, _ = rendered.split('---\n', 2)
    return yaml.safe_load(block)


class TestSubagentDigest:
    """A sub-agent's digest names it and its parent, and counts its own turns only."""

    def _tree(self, tmp_path: Path) -> tuple[Path, Path]:
        return _write_parent_and_subagent(
            tmp_path / MAIN_ENC,
            subagent_records=[
                {
                    'type': 'assistant',
                    'message': {'role': 'assistant', 'content': [
                        {'type': 'text', 'text': 'my mistake, the caller lives in routing.py'},
                    ]},
                },
                _tool_error_record(),
            ],
        )

    def test_frontmatter_names_the_subagent_and_its_parent(self, tmp_path):
        _, subagent = self._tree(tmp_path)
        front = _frontmatter(digest.build_digest(subagent))
        assert front['session'] == _subagent_key()
        assert front['parent_session'] == PARENT_SID
        assert front['encoded_dir'] == MAIN_ENC

    def test_counts_the_subagents_own_signals(self, tmp_path):
        _, subagent = self._tree(tmp_path)
        counts = _frontmatter(digest.build_digest(subagent))['signal_counts']
        assert counts['tool_error'] == 1
        assert counts['self_correct'] == 1

    def test_parent_agents_prompt_is_not_a_gold_turn(self, tmp_path):
        _, subagent = self._tree(tmp_path)
        rendered = digest.build_digest(subagent)
        assert _frontmatter(rendered)['n_user_turns'] == 0
        assert '## User Corrections' not in rendered

    def test_parent_digest_names_no_parent_and_holds_none_of_the_subagents_hits(self, tmp_path):
        parent, _ = self._tree(tmp_path)
        front = _frontmatter(digest.build_digest(parent))
        assert front['session'] == PARENT_SID
        assert front['parent_session'] == ''
        assert not any(front['signal_counts'].values())
