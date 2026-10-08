"""Tests for write_triage_judge_wording.py — the eval-side judge wording knob (task 6151).

Every override is observed where it matters: on the request the SHIPPED
``_call_llm`` sends, through a faked ``openai.AsyncOpenAI``. No network.
"""
from __future__ import annotations

import functools
import hashlib
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _fm_helpers import load_script_module

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.server import write_triage_judge

SCRIPTS = Path(__file__).parent.parent / 'scripts'

#: sha256 of ``JUDGE_SYSTEM_PROMPT`` at 16794cdd1d^1, the commit before ψ.
PRE_PSI_SHA256 = '49be71b5f521643bebf434145a0a300bbf4a1d1b0a3a64faf1e19c4a20df208e'


@functools.cache
def _mod() -> types.ModuleType:
    return load_script_module(
        SCRIPTS / 'write_triage_judge_wording.py', 'write_triage_judge_wording',
    )


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _openai_client() -> MagicMock:
    """A fake ``AsyncOpenAI`` that is its own async context manager, as the SDK is."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.responses.create = AsyncMock(return_value=types.SimpleNamespace(
        output_text='{"verdict": "distinct"}',
        usage=types.SimpleNamespace(
            input_tokens=10, output_tokens=7, output_tokens_details=None,
        ),
        status='completed',
        incomplete_details=None,
    ))
    return client


async def _call_shipped_llm() -> None:
    await write_triage_judge._call_llm(
        provider='openai',
        model='gpt-4o-mini',
        prompt='p',
        memory_service=types.SimpleNamespace(config=FusedMemoryConfig()),
        timeout=5.0,
        reasoning_effort=None,
    )


class TestWordings:
    def test_the_vocabulary(self) -> None:
        assert _mod().WORDING_SHIPPED == 'shipped'
        assert _mod().WORDING_PRE_PSI == 'pre-psi'
        assert _mod().WORDINGS == ('shipped', 'pre-psi')

    def test_the_pre_psi_prompt_is_the_historical_one_byte_for_byte(self) -> None:
        assert _sha(_mod().PRE_PSI_JUDGE_SYSTEM_PROMPT) == PRE_PSI_SHA256

    def test_shipped_is_the_live_module_prompt_and_differs_from_pre_psi(self) -> None:
        shipped = _mod().system_prompt(_mod().WORDING_SHIPPED)
        assert shipped is write_triage_judge.JUDGE_SYSTEM_PROMPT
        assert _mod().system_prompt(_mod().WORDING_PRE_PSI) == _mod().PRE_PSI_JUDGE_SYSTEM_PROMPT
        assert shipped != _mod().PRE_PSI_JUDGE_SYSTEM_PROMPT

    def test_an_unknown_wording_is_refused_naming_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match='shipped') as excinfo:
            _mod().system_prompt('post-psi')
        assert 'pre-psi' in str(excinfo.value)


class TestJudgeWordingOverride:
    @pytest.mark.asyncio
    async def test_the_shipped_call_path_sends_the_overridden_prompt_then_the_shipped_one(
        self,
    ) -> None:
        client = _openai_client()
        with patch('openai.AsyncOpenAI', return_value=client):
            with _mod().judge_wording(_mod().WORDING_PRE_PSI) as sha:
                await _call_shipped_llm()
            inside = client.responses.create.call_args.kwargs['instructions']
            await _call_shipped_llm()
            after = client.responses.create.call_args.kwargs['instructions']

        assert inside == _mod().PRE_PSI_JUDGE_SYSTEM_PROMPT
        assert sha == _sha(_mod().PRE_PSI_JUDGE_SYSTEM_PROMPT)
        assert after == _mod().system_prompt(_mod().WORDING_SHIPPED)

    def test_the_shipped_wording_yields_the_shipped_sha(self) -> None:
        with _mod().judge_wording(_mod().WORDING_SHIPPED) as sha:
            assert sha == _sha(_mod().system_prompt(_mod().WORDING_SHIPPED))

    def test_the_shipped_prompt_is_restored_when_the_body_raises(self) -> None:
        shipped = _mod().system_prompt(_mod().WORDING_SHIPPED)
        with pytest.raises(KeyError), _mod().judge_wording(_mod().WORDING_PRE_PSI):
            raise KeyError('boom')
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is shipped

    def test_nesting_a_different_wording_is_refused_naming_both(self) -> None:
        with _mod().judge_wording(_mod().WORDING_PRE_PSI):
            with (
                pytest.raises(RuntimeError, match='shipped') as excinfo,
                _mod().judge_wording(_mod().WORDING_SHIPPED),
            ):
                pass  # pragma: no cover - the enter refuses
            assert 'pre-psi' in str(excinfo.value)
            assert write_triage_judge.JUDGE_SYSTEM_PROMPT == _mod().PRE_PSI_JUDGE_SYSTEM_PROMPT
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is _mod().system_prompt(
            _mod().WORDING_SHIPPED,
        )

    @pytest.mark.parametrize('wording', ['shipped', 'pre-psi'])
    def test_re_entering_the_same_wording_is_refused(self, wording: str) -> None:
        with (
            _mod().judge_wording(wording),
            pytest.raises(RuntimeError),
            _mod().judge_wording(wording),
        ):
            pass  # pragma: no cover - the enter refuses
        assert write_triage_judge.JUDGE_SYSTEM_PROMPT is _mod().system_prompt(
            _mod().WORDING_SHIPPED,
        )
