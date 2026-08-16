"""Tests for lms_healthcheck (task 3713, LME-alpha).

PRD-MARKER:local-memory-models-eval serving

Part 1 (step 11): request construction, response verdicts, model identity and
transport failures for the LLM axis.
Part 2 (step 13): the embedding axis.

Every assertion here is really one assertion: *a broken arm must be
DETECTED, not absorbed*.  The PRD's boundary row for this task demands a
deliberately-invalid response be caught, because the whole eval downstream
(eta screening, theta full runs) attributes numbers to arms on the strength
of this check.  Three failure modes are specifically load-bearing:

1.  The probe schema must be NESTED -- `model_json_schema()` emitting
    `$defs`/`$ref` -- because that is the shape graphiti really emits and the
    exact shape llama.cpp silently mishandles by falling back to
    unconstrained output (ggml-org/llama.cpp#21228).  A flat stand-in schema
    would pass on an arm that cannot do the job the eval needs.

2.  The `json_object`-only MoE arm gets the SAME client-side validation as
    the schema-constrained arms.  Without it, an unconstrained fallback
    returning prose is indistinguishable from a pass.

3.  A completion only counts once `/v1/models` lists the arm's
    `served_model_name`.  A `/health` 200 on a colliding port let a DIFFERENT
    model answer and mis-attributed an entire eval run on 2026-04-08
    (scripts/run_vllm_eval.py:541-553).  In a rig that starts and stops units
    repeatedly on a fixed port block, that is the expected failure.

No network is touched: every HTTP call goes through the shared
`install_fake_httpx` fixture (scripts/tests/conftest.py), which exposes only
`post`/`get` and turns any other attribute access into a loud `pytest.fail`.
"""
from __future__ import annotations

import datetime as _datetime
import inspect
import json
from pathlib import Path

import lms_ctl
import lms_healthcheck
import lms_manifest
import lms_vram
import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _arm(**overrides) -> lms_manifest.ArmEntry:
    fields = {
        'arm_id': 'qwen3.5-9b',
        'axis': 'llm',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.26.0',
        'model_ref': 'QuantTrio/Qwen3.5-9B-AWQ',
        'quant': 'awq',
        'port': 8410,
        'served_model_name': 'qwen3.5-9b',
        'reasoning': 'off',
        'structured_output_mode': 'json_schema',
        'est_vram_gib': 6.0,
    }
    fields.update(overrides)
    return lms_manifest.ArmEntry(**fields)


def _moe_arm(**overrides) -> lms_manifest.ArmEntry:
    fields = {
        'arm_id': 'moe-stretch',
        'axis': 'llm',
        'stack': 'llamacpp',
        'image': 'ghcr.io/ggml-org/llama.cpp:server-cuda',
        'model_ref': 'unsloth/Qwen3.6-35B-A3B-GGUF',
        'quant': 'iq4_xs',
        'port': 8413,
        'served_model_name': 'moe-stretch',
        'reasoning': 'off',
        'structured_output_mode': 'json_object',
        'est_vram_gib': 15.0,
    }
    fields.update(overrides)
    return lms_manifest.ArmEntry(**fields)


class _Resp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


_UNSET = object()


def _completion(
    content: str | None, finish_reason: str = 'stop', usage=_UNSET
) -> dict:
    """An OpenAI chat-completions response body carrying *content*.

    `None` is a REAL shape, not a test convenience: vLLM returns
    `content: null` when a reasoning parser consumes the whole generation,
    where llama.cpp returns `''`. Typing this `str` would make the null case
    unexpressible — and that case is exactly what made COMPLETION_TRUNCATED
    unreachable on the vLLM stack.

    `usage` is likewise absent-by-default rather than defaulting to `None`,
    because "no usage block at all" and "a usage block whose
    prompt_tokens_details is null" are two distinct real shapes (task 3781):
    the first is what a minimal server returns, the second is what vLLM
    returns, and only the second proves the reader tolerates a null.
    """
    body: dict = {
        'id': 'cmpl-1',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': finish_reason,
            }
        ],
    }
    if usage is not _UNSET:
        body['usage'] = usage
    return body


def _models_payload(*names: str) -> dict:
    return {'object': 'list', 'data': [{'id': n, 'object': 'model'} for n in names]}


def _valid_probe_json() -> str:
    """A conforming completion — which now means one that actually EXTRACTED.

    This fixture used to name a single entity.  It was widened to the full set
    when the extraction floor landed, and the widening is the point rather than
    a chore: a one-entity stand-in was letting the suite call "conforming"
    something no working arm produces.  Both models measured on 2026-08-06
    return exactly these four whenever they are able to extract at all.
    """
    return json.dumps(
        {
            'entities': [
                {
                    'name': 'Leo',
                    'entity_type': 'Person',
                    'attributes': [{'name': 'role', 'value': 'runs Dark Factory'}],
                },
                {
                    'name': 'Dark Factory',
                    'entity_type': 'Organization',
                    'attributes': [{'name': 'kind', 'value': 'software factory'}],
                },
                {
                    'name': 'Graphiti',
                    'entity_type': 'System',
                    'attributes': [{'name': 'role', 'value': 'temporal knowledge graph'}],
                },
                {
                    'name': 'FalkorDB',
                    'entity_type': 'Database',
                    'attributes': [{'name': 'role', 'value': 'backs Graphiti'}],
                },
            ],
            'summary': 'Four entities extracted from the probe text.',
        }
    )


def _empty_extraction_json() -> str:
    """Schema-valid and empty — verbatim what qwen3.5-9b returned on 2026-08-06
    in every configuration where it could not reason, and what the committed
    README recorded as a PASS."""
    return json.dumps(
        {'entities': [], 'summary': 'No entities extracted from the provided text.'}
    )


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def test_json_schema_arm_sends_response_format_json_schema():
    body = lms_healthcheck.build_llm_probe_request(_arm())

    assert body['model'] == 'qwen3.5-9b'
    assert body['response_format']['type'] == 'json_schema'
    schema = body['response_format']['json_schema']['schema']
    assert schema == lms_healthcheck.ProbeExtraction.model_json_schema()


def test_probe_schema_is_nested_and_carries_defs_and_ref():
    """The whole point of the probe: exercise the shape graphiti really emits.

    A flat schema would pass on an arm that cannot honour `$ref`/`$defs`
    (ggml-org/llama.cpp#21228), which is precisely the capability the eval
    depends on.
    """
    schema = lms_healthcheck.ProbeExtraction.model_json_schema()

    assert '$defs' in schema
    assert '"$ref"' in json.dumps(schema)

    body = lms_healthcheck.build_llm_probe_request(_arm())
    sent = json.dumps(body['response_format']['json_schema']['schema'])
    assert '$defs' in sent and '$ref' in sent


def test_json_object_arm_sends_json_object_and_never_a_json_schema():
    """llama.cpp is `json_object`-only here, and the request must say so.

    Sending a `json_schema` response_format to that arm would be answered
    with an unconstrained completion while LOOKING constrained -- the exact
    silent degradation #21228 produces.
    """
    body = lms_healthcheck.build_llm_probe_request(_moe_arm())

    assert body['response_format'] == {'type': 'json_object'}
    assert 'json_schema' not in json.dumps(body)


def test_json_object_arm_carries_the_schema_in_the_prompt_instead():
    """An unconstrained arm can only comply if it is TOLD the shape."""
    body = lms_healthcheck.build_llm_probe_request(_moe_arm())

    prompt = json.dumps(body['messages'])
    for field in ('entities', 'entity_type', 'attributes', 'summary'):
        assert field in prompt


def test_probe_request_is_deterministic():
    """Temperature 0 and a bounded completion: the probe measures capability,
    not sampling luck, and must not hang on a runaway generation."""
    body = lms_healthcheck.build_llm_probe_request(_arm())

    assert body['temperature'] == 0
    assert isinstance(body['max_tokens'], int) and body['max_tokens'] > 0


def test_building_an_llm_probe_for_an_embedding_arm_is_a_typed_error():
    embedding_arm = _arm(
        arm_id='qwen3-embedding-0.6b',
        axis='embedding',
        served_model_name='qwen3-embedding-0.6b',
        structured_output_mode='none',
        port=8414,
        dims=1024,
    )

    with pytest.raises(lms_healthcheck.HealthcheckError):
        lms_healthcheck.build_llm_probe_request(embedding_arm)


# ---------------------------------------------------------------------------
# The warm-up request (task 3781)
#
# The measured probe is meant to be ENGINE-warm and PREFIX-COLD -- the state a
# production request actually arrives in.  That is only achievable if the
# discarded warm-up carries a DIFFERENT prompt: warming with the same text
# populates the very prefix cache the measured probe must find cold.  Measured
# 2026-08-06 on moe-stretch, llama.cpp served the warm run with 338 of its 343
# prompt tokens from cache, so a same-prompt warm-up does not merely blunt the
# measurement, it inverts what the number means.
# ---------------------------------------------------------------------------


def _user_content(body: dict) -> str:
    """The user message's text, which is where the probe passage lives."""
    return next(m['content'] for m in body['messages'] if m['role'] == 'user')


def test_the_llm_warmup_request_does_not_reuse_the_measured_probe_text():
    warm = lms_healthcheck.build_llm_probe_request(_arm(), warmup=True)
    measured = lms_healthcheck.build_llm_probe_request(_arm(), warmup=False)

    warm_sent = json.dumps(warm['messages'])
    measured_sent = json.dumps(measured['messages'])

    assert lms_healthcheck.PROBE_TEXT not in warm_sent
    assert lms_healthcheck.WARMUP_PROBE_TEXT in warm_sent
    assert lms_healthcheck.PROBE_TEXT in measured_sent
    assert lms_healthcheck.WARMUP_PROBE_TEXT not in measured_sent


def test_the_warmup_text_diverges_at_the_first_user_token():
    """The load-bearing property, not a cosmetic one.

    A prefix cache is a PREFIX cache: it is the shared LEADING tokens that get
    reused, so a warm-up sharing a long opening with the measured probe warms
    exactly the thing the measured probe is supposed to find cold.  The probe
    text is placed first in the user message precisely so a different passage
    diverges at the very first user token, leaving only the short system
    message cacheable.
    """
    warm = _user_content(lms_healthcheck.build_llm_probe_request(_arm(), warmup=True))
    measured = _user_content(lms_healthcheck.build_llm_probe_request(_arm()))

    shared = 0
    # strict=False deliberately: the two passages are different lengths, and
    # the shorter one running out IS the end of any shared prefix.
    for warm_char, measured_char in zip(warm, measured, strict=False):
        if warm_char != measured_char:
            break
        shared += 1

    assert shared < 8, (
        f'the warm-up and measured user messages share a {shared}-character '
        'leading prefix; that prefix is exactly what a prefix cache reuses'
    )


@pytest.mark.parametrize('make_arm', [_arm, _moe_arm], ids=['vllm', 'llamacpp'])
def test_the_warmup_request_is_otherwise_shape_identical(make_arm):
    """Same code path, different text.

    The warm-up's whole job is to warm what the measured probe will exercise --
    grammar compilation, sampler setup, CUDA graph capture.  A warm-up that
    differed in `response_format` or `chat_template_kwargs` would warm a
    DIFFERENT path and leave the measured probe paying the cold cost anyway.
    """
    warm = lms_healthcheck.build_llm_probe_request(make_arm(), warmup=True)
    measured = lms_healthcheck.build_llm_probe_request(make_arm())

    for key in ('model', 'temperature', 'max_tokens', 'response_format'):
        assert warm[key] == measured[key], key
    assert warm.get('chat_template_kwargs') == measured.get('chat_template_kwargs')
    assert set(warm) == set(measured)
    assert [m['role'] for m in warm['messages']] == [
        m['role'] for m in measured['messages']
    ]
    # The system message is the one part that may legitimately stay cached.
    warm_system = next(m['content'] for m in warm['messages'] if m['role'] == 'system')
    measured_system = next(
        m['content'] for m in measured['messages'] if m['role'] == 'system'
    )
    assert warm_system == measured_system


def test_warmup_defaults_to_false_so_the_measured_probe_is_the_default():
    """Every existing caller asks for the measured probe by asking for nothing."""
    assert lms_healthcheck.build_llm_probe_request(
        _arm()
    ) == lms_healthcheck.build_llm_probe_request(_arm(), warmup=False)
    assert lms_healthcheck.build_embedding_probe_request(
        _embedding_arm()
    ) == lms_healthcheck.build_embedding_probe_request(_embedding_arm(), warmup=False)


# ---------------------------------------------------------------------------
# Response verdicts -- the same probe model judges every arm
# ---------------------------------------------------------------------------


def test_conforming_json_completion_passes():
    result = lms_healthcheck.verify_llm_response(_arm(), _completion(_valid_probe_json()))

    assert result.verdict == 'PASS'
    assert result.reason == lms_healthcheck.Reason.OK


def test_prose_completion_fails():
    result = lms_healthcheck.verify_llm_response(
        _arm(), _completion('Sure! The endpoint is healthy and ready to serve.')
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.NOT_JSON


def test_json_missing_a_required_field_fails():
    payload = json.dumps({'entities': []})  # no `summary`

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(payload))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.SCHEMA_MISSING_FIELD
    assert 'summary' in result.detail


def test_json_with_a_wrong_typed_field_fails():
    payload = json.dumps({'entities': [], 'summary': 123})

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(payload))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.SCHEMA_WRONG_TYPE
    assert 'summary' in result.detail


def test_wrong_type_in_a_nested_ref_fails():
    """The nested leg specifically: a `$ref`-reached field must be checked too,
    or an arm that flattens the schema would pass."""
    payload = json.dumps(
        {
            'entities': [
                {'name': 'Graphiti', 'entity_type': 'System', 'attributes': 'none'}
            ],
            'summary': 'x',
        }
    )

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(payload))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.SCHEMA_WRONG_TYPE
    assert 'attributes' in result.detail


def test_markdown_fenced_json_fails_with_its_own_reason_code():
    """A DELIBERATE choice, not an oversight: fenced output means the server
    did not honour the structured-output contract, which is exactly the signal
    the eval needs.  Tolerating the fence here would launder an unconstrained
    arm into a PASS and hide the capability gap this task exists to measure.
    Its own reason code keeps it diagnosable rather than lumped in with prose.
    """
    payload = f'```json\n{_valid_probe_json()}\n```'

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(payload))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.MARKDOWN_FENCED_JSON


def test_empty_completion_fails():
    result = lms_healthcheck.verify_llm_response(_arm(), _completion(''))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.EMPTY_COMPLETION


def test_a_completion_cut_off_by_the_token_cap_is_not_reported_as_empty():
    """The moe-stretch defect measured in step 23: gemma-4-26B-A4B is a
    reasoning arm, its thinking spent the whole 512-token budget, and the
    content field came back empty with finish_reason `length`.  Calling that
    `empty_completion` blames the model for the harness's cap."""
    result = lms_healthcheck.verify_llm_response(
        _moe_arm(), _completion('', finish_reason='length')
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.COMPLETION_TRUNCATED
    assert result.reason != lms_healthcheck.Reason.EMPTY_COMPLETION


def test_a_completion_cut_off_mid_json_is_not_reported_as_unparseable():
    """A fragment that does not parse BECAUSE it was cut off is a truncation,
    not a malformed answer — the arm never got to finish one."""
    result = lms_healthcheck.verify_llm_response(
        _moe_arm(), _completion('{"entities": [{"name": "Gra', finish_reason='length')
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.COMPLETION_TRUNCATED


def test_an_empty_completion_that_stopped_normally_is_still_empty_not_truncated():
    """The truncation code must not swallow the emptiness code: an arm that
    chose to say nothing is a different defect from one that was cut off."""
    result = lms_healthcheck.verify_llm_response(_arm(), _completion(''))

    assert result.reason == lms_healthcheck.Reason.EMPTY_COMPLETION


def test_prose_that_hit_the_cap_is_still_truncation_not_a_pass():
    """Truncation is never a softer verdict — it is FAIL with a better name."""
    result = lms_healthcheck.verify_llm_response(
        _moe_arm(), _completion('Let me think about this', finish_reason='length')
    )

    assert result.verdict == 'FAIL'


def test_valid_json_that_happened_to_hit_the_cap_still_passes():
    """finish_reason alone does not condemn an arm.  If the content the arm
    DID emit validates, it answered — the cap is then irrelevant."""
    result = lms_healthcheck.verify_llm_response(
        _moe_arm(), _completion(_valid_probe_json(), finish_reason='length')
    )

    assert result.verdict == 'PASS'


def test_the_probe_token_cap_is_uniform_across_every_llm_arm():
    """Steward ruling on esc-3713-8: one shared cap, no per-arm and no
    per-structured_output_mode budget.  A cap that varied by arm would hand
    eta a probe-shaped confound between the vLLM arms and the MoE arm."""
    body_dense = lms_healthcheck.build_llm_probe_request(_arm())
    body_moe = lms_healthcheck.build_llm_probe_request(_moe_arm())

    assert body_dense['max_tokens'] == body_moe['max_tokens']
    assert body_dense['max_tokens'] == lms_healthcheck.PROBE_MAX_TOKENS


def test_a_response_body_without_choices_fails_rather_than_raising():
    result = lms_healthcheck.verify_llm_response(_arm(), {'error': 'model not found'})

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.MALFORMED_RESPONSE


def test_every_llm_failure_reason_is_distinct():
    """Distinct machine-readable codes, so a report says WHICH way an arm broke."""
    reasons = {
        lms_healthcheck.verify_llm_response(_arm(), _completion(body)).reason
        for body in (
            'Sure! All healthy.',
            json.dumps({'entities': []}),
            json.dumps({'entities': [], 'summary': 123}),
            f'```json\n{_valid_probe_json()}\n```',
            '',
        )
    }

    assert len(reasons) == 5
    assert lms_healthcheck.Reason.OK not in reasons


@pytest.mark.parametrize(
    'bad_content',
    [
        'The arm is up and running.',
        json.dumps({'entities': []}),
        json.dumps({'entities': [], 'summary': 123}),
    ],
)
def test_the_json_object_arm_gets_the_same_client_side_validation(bad_content):
    """The ONLY thing standing between an unconstrained fallback and a false
    PASS.  llama.cpp cannot enforce the schema (#21228), so the client must."""
    result = lms_healthcheck.verify_llm_response(_moe_arm(), _completion(bad_content))

    assert result.verdict == 'FAIL'


def test_the_json_object_arm_passes_on_conforming_output():
    result = lms_healthcheck.verify_llm_response(
        _moe_arm(), _completion(_valid_probe_json())
    )

    assert result.verdict == 'PASS'


# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------


def test_identity_passes_when_models_lists_the_served_model_name():
    result = lms_healthcheck.check_model_identity(_arm(), _models_payload('qwen3.5-9b'))

    assert result.verdict == 'PASS'


def test_identity_fails_when_a_different_model_answers():
    result = lms_healthcheck.check_model_identity(
        _arm(), _models_payload('mistral-small-3.2-24b')
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.IDENTITY_MISMATCH
    assert 'mistral-small-3.2-24b' in result.detail


def test_identity_fails_on_an_empty_or_malformed_models_body():
    for body in ({'object': 'list', 'data': []}, {}, {'data': 'nope'}):
        result = lms_healthcheck.check_model_identity(_arm(), body)
        assert result.verdict == 'FAIL'
        assert result.reason == lms_healthcheck.Reason.IDENTITY_MISMATCH


def test_a_valid_completion_from_the_wrong_model_still_fails(install_fake_httpx):
    """Identity is checked BEFORE a completion counts.  Otherwise the worst
    outcome is silent: eta/theta attribute a whole arm's metrics to the wrong
    model (the 2026-04-08 404 bug, scripts/run_vllm_eval.py:541-553)."""
    def fake_get(url, **kwargs):
        return _Resp(200, _models_payload('some-other-model'))

    def fake_post(url, **kwargs):
        return _Resp(200, _completion(_valid_probe_json()))

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.IDENTITY_MISMATCH


# ---------------------------------------------------------------------------
# End-to-end probe + transport failures
# ---------------------------------------------------------------------------


def test_probe_llm_arm_passes_against_a_healthy_arm(install_fake_httpx):
    seen = {}

    def fake_get(url, **kwargs):
        seen['get'] = url
        return _Resp(200, _models_payload('qwen3.5-9b'))

    def fake_post(url, **kwargs):
        seen['post'] = url
        seen['body'] = kwargs.get('json')
        seen['timeout'] = kwargs.get('timeout')
        return _Resp(200, _completion(_valid_probe_json()))

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'PASS'
    assert result.reason == lms_healthcheck.Reason.OK
    assert seen['get'] == 'http://127.0.0.1:8410/v1/models'
    assert seen['post'] == 'http://127.0.0.1:8410/v1/chat/completions'
    # 127.0.0.1 explicitly, never `localhost`: the latter can resolve to ::1
    # while the server listens on IPv4 only (scripts/run_vllm_eval.py:505-512).
    assert 'localhost' not in seen['get'] and 'localhost' not in seen['post']
    assert seen['body']['model'] == 'qwen3.5-9b'
    # A plain float, never an httpx.Timeout object -- the shared fake exposes
    # neither, and reaching for one would be a loud fixture miss.
    assert isinstance(seen['timeout'], float)
    assert result.latency_ms >= 0


def test_non_200_completion_fails_with_the_status_reason(install_fake_httpx):
    def fake_get(url, **kwargs):
        return _Resp(200, _models_payload('qwen3.5-9b'))

    def fake_post(url, **kwargs):
        return _Resp(500, {'error': 'engine dead'})

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.HTTP_STATUS
    assert '500' in result.detail


def test_connection_error_fails_rather_than_raising(install_fake_httpx):
    def fake_get(url, **kwargs):
        raise OSError('[Errno 111] Connection refused')

    install_fake_httpx(post=None, get=fake_get)

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.TRANSPORT_ERROR
    assert 'Connection refused' in result.detail


def test_a_timeout_fails_rather_than_raising(install_fake_httpx):
    class _ReadTimeout(Exception):
        pass

    def fake_get(url, **kwargs):
        return _Resp(200, _models_payload('qwen3.5-9b'))

    def fake_post(url, **kwargs):
        raise _ReadTimeout('timed out after 120s')

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.TRANSPORT_ERROR
    # The exception TYPE survives into the report: "it timed out" and "it
    # refused the connection" are different operational problems.
    assert '_ReadTimeout' in result.detail


def test_an_unparseable_completion_body_fails_rather_than_raising(install_fake_httpx):
    class _NotJson(_Resp):
        def json(self):
            raise ValueError('Expecting value: line 1 column 1 (char 0)')

    def fake_get(url, **kwargs):
        return _Resp(200, _models_payload('qwen3.5-9b'))

    def fake_post(url, **kwargs):
        return _NotJson(200)

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'FAIL'
    assert result.reason in (
        lms_healthcheck.Reason.MALFORMED_RESPONSE,
        lms_healthcheck.Reason.TRANSPORT_ERROR,
    )


def test_a_placeholder_arm_is_refused_before_any_request(install_fake_httpx):
    """An arm whose model_ref is still `TBD-Q3` has nothing to probe.  Issuing
    the request anyway would report the resulting 404 as an ARM failure and
    bury the real cause -- an unresolved PRD Open Question -- in a stack of
    identical transport errors."""
    def _boom(url, **kwargs):
        raise AssertionError('no request may be issued for a placeholder arm')

    install_fake_httpx(post=_boom, get=_boom)

    placeholder = _moe_arm(model_ref='TBD-Q3-pick-a-gguf', image='TBD-Q3', quant='TBD-Q3')
    assert placeholder.is_placeholder is True

    result = lms_healthcheck.probe_llm_arm(placeholder)

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.PLACEHOLDER_ARM


# ---------------------------------------------------------------------------
# The cached-prompt-token diagnostic (task 3781)
#
# This is the only DIRECT evidence in the artifact that the measured probe
# found the prefix cache COLD -- the claim the whole cold/warm split rests on.
# Without it, "engine-warm, prefix-cold" is an assertion about the instrument
# that the instrument's own output cannot support.
#
# The two stacks answer differently and the asymmetry is not hidden: llama.cpp
# fills `usage.prompt_tokens_details.cached_tokens`, so a near-zero count there
# proves it; vLLM returns `prompt_tokens_details: null`, so on that stack the
# claim rests on latency alone.  REPORTED and NON-GATING, exactly like
# `top_level_entities_named` -- it must never decide a verdict.
# ---------------------------------------------------------------------------


def _healthy_llm_wire(install_fake_httpx, body, model='qwen3.5-9b'):
    """Install a fake httpx serving *body* from a correctly-identified arm."""
    install_fake_httpx(
        get=lambda url, **kw: _Resp(200, _models_payload(model)),
        post=lambda url, **kw: _Resp(200, body),
    )


def test_a_llamacpp_shaped_response_records_its_cached_prompt_tokens(
    install_fake_httpx,
):
    """The llama.cpp shape.  5 of 343 cached is what a genuinely cold prefix
    looks like; 338 of 343 is the poisoned same-prompt warm-up this task's
    design exists to avoid."""
    _healthy_llm_wire(
        install_fake_httpx,
        _completion(
            _valid_probe_json(),
            usage={'prompt_tokens': 343, 'prompt_tokens_details': {'cached_tokens': 5}},
        ),
    )

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'PASS'
    assert result.cached_prompt_tokens == 5


def test_a_vllm_shaped_null_prompt_tokens_details_still_produces_both_latencies(
    install_fake_httpx,
):
    """The measured vLLM shape.  On that stack the cold/warm split has to be
    visible through LATENCY ALONE, so a null here must never degrade a field,
    move a verdict, or raise."""
    _healthy_llm_wire(
        install_fake_httpx,
        _completion(
            _valid_probe_json(),
            usage={'prompt_tokens': 343, 'prompt_tokens_details': None},
        ),
    )

    report = lms_healthcheck.run_healthcheck(
        [_arm()], gpu_probe=lambda: _snapshot(), baseline=_baseline(),
    )
    row = report.arms[0]

    assert row.verdict == 'PASS'
    assert row.first_probe_ms > 0
    assert row.latency_ms > 0
    assert row.measured_cached_prompt_tokens is None


def test_a_response_with_no_usage_block_at_all_records_no_cached_count(
    install_fake_httpx,
):
    _healthy_llm_wire(install_fake_httpx, _completion(_valid_probe_json()))

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'PASS'
    assert result.cached_prompt_tokens is None


@pytest.mark.parametrize(
    'usage',
    [
        'nope',
        {'prompt_tokens_details': []},
        {'prompt_tokens_details': {'cached_tokens': 'many'}},
        {'prompt_tokens_details': {'cached_tokens': True}},
        {'prompt_tokens_details': {}},
        None,
    ],
    ids=['string', 'list-details', 'string-count', 'bool-count', 'empty', 'null'],
)
def test_a_malformed_usage_block_is_tolerated_not_raised(install_fake_httpx, usage):
    """Read defensively, exactly as `_extract_content` and `_was_truncated`
    already read bodies: an unreadable diagnostic must never abort a sweep or
    change a verdict.  `True` is rejected on its own account -- Python's
    `isinstance(True, int)` hole, closed here as `verify_embedding_response`
    already closes it for vector values."""
    _healthy_llm_wire(
        install_fake_httpx, _completion(_valid_probe_json(), usage=usage)
    )

    result = lms_healthcheck.probe_llm_arm(_arm())

    assert result.verdict == 'PASS'
    assert result.cached_prompt_tokens is None


def test_an_embedding_row_records_no_cached_prompt_tokens():
    """The question does not apply to an embedding arm, so the answer is None
    rather than a 0 that reads like a measured cold cache."""
    row = _report(arms=[_unprefixed_arm()]).arms[0]

    assert row.measured_cached_prompt_tokens is None


# ===========================================================================
# Part 2 (step 13) -- the embedding axis.
#
# An embedding arm fails QUIETLY in a way an LLM arm does not: it returns a
# vector of plausible-looking floats no matter what.  A wrong-length vector, a
# NaN, or an all-zero degenerate output all still LOOK like an embedding, and
# every one of them would silently corrupt the retrieval numbers iota reports
# rather than crashing anything.  So the checks below are the only place those
# failures can be caught at all.
# ===========================================================================


PROBE_DIMS = 1024
R2_DIMS = 768


def _embedding_arm(**overrides) -> lms_manifest.ArmEntry:
    fields = {
        'arm_id': 'qwen3-embedding-0.6b',
        'axis': 'embedding',
        'stack': 'vllm',
        'image': 'vllm/vllm-openai:v0.26.0',
        'model_ref': 'Qwen/Qwen3-Embedding-0.6B',
        'quant': 'none',
        'port': 8414,
        'served_model_name': 'qwen3-embedding-0.6b',
        'structured_output_mode': 'none',
        'est_vram_gib': 2.0,
        'dims': PROBE_DIMS,
        'query_prefix': (
            'Instruct: Given a search query, retrieve relevant memory records '
            'that answer the query\nQuery: '
        ),
    }
    fields.update(overrides)
    return lms_manifest.ArmEntry(**fields)


def _unprefixed_arm(**overrides) -> lms_manifest.ArmEntry:
    return _embedding_arm(
        arm_id='granite-embedding-english-r2',
        model_ref='ibm-granite/granite-embedding-english-r2',
        port=8415,
        served_model_name='granite-embedding-english-r2',
        est_vram_gib=1.0,
        dims=R2_DIMS,
        query_prefix=None,
        **overrides,
    )


def _embedding_payload(vector, model='qwen3-embedding-0.6b') -> dict:
    return {
        'object': 'list',
        'model': model,
        'data': [{'object': 'embedding', 'index': 0, 'embedding': vector}],
    }


# ---------------------------------------------------------------------------
# Request construction
# ---------------------------------------------------------------------------


def test_embedding_request_targets_the_served_model_name():
    body = lms_healthcheck.build_embedding_probe_request(_unprefixed_arm())

    assert body['model'] == 'granite-embedding-english-r2'
    assert body['input'] == [lms_healthcheck.EMBEDDING_PROBE_QUERY]


def test_the_declared_query_prefix_is_applied():
    """The Qwen3-Embedding family REQUIRES a query-side instruct prefix (PRD
    line 134).  Dropping it does not error -- it quietly degrades every
    retrieval number iota later reports, which is a far worse outcome than a
    crash because nothing downstream would ever notice."""
    arm = _embedding_arm()
    prefix = arm.query_prefix
    assert prefix is not None

    body = lms_healthcheck.build_embedding_probe_request(arm)

    assert body['input'] == [prefix + lms_healthcheck.EMBEDDING_PROBE_QUERY]
    assert body['input'][0].startswith('Instruct:')


def test_an_arm_without_a_declared_prefix_gets_none_invented():
    body = lms_healthcheck.build_embedding_probe_request(_unprefixed_arm())

    assert 'Instruct:' not in body['input'][0]


def test_the_embedding_warmup_query_differs_from_the_measured_one():
    """The embedding axis needs its own distinct warm-up text, for the same
    reason the LLM axis does -- and the declared `query_prefix` must be applied
    to BOTH.  An unprefixed warm-up would warm the document-side path of an
    asymmetric model, which is not the path the measured probe takes.
    """
    arm = _embedding_arm()
    prefix = arm.query_prefix
    assert prefix is not None

    warm = lms_healthcheck.build_embedding_probe_request(arm, warmup=True)
    measured = lms_healthcheck.build_embedding_probe_request(arm)

    assert warm['input'][0] != measured['input'][0]
    assert lms_healthcheck.EMBEDDING_PROBE_QUERY not in warm['input'][0]
    assert warm['input'] == [prefix + lms_healthcheck.WARMUP_EMBEDDING_QUERY]
    assert warm['input'][0].startswith('Instruct:')
    assert warm['model'] == measured['model']
    assert warm['encoding_format'] == measured['encoding_format']


def test_building_an_embedding_probe_for_an_llm_arm_is_a_typed_error():
    with pytest.raises(lms_healthcheck.HealthcheckError):
        lms_healthcheck.build_embedding_probe_request(_arm())


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_a_well_formed_vector_of_the_declared_dims_passes():
    arm = _embedding_arm()
    payload = _embedding_payload([0.01 * i for i in range(PROBE_DIMS)])

    result = lms_healthcheck.verify_embedding_response(arm, payload)

    assert result.verdict == 'PASS'
    assert result.reason == lms_healthcheck.Reason.OK


def test_empty_data_fails():
    payload = {'object': 'list', 'model': 'qwen3-embedding-0.6b', 'data': []}

    result = lms_healthcheck.verify_embedding_response(_embedding_arm(), payload)

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.EMPTY_EMBEDDING_DATA


def test_wrong_dimensionality_fails_naming_both_numbers():
    """The manifest/model mismatch that would break iota's comparison: two
    arms on this slate legitimately share 768 dims, so a stale unit on a
    colliding port can return a vector that is the right SHAPE for the wrong
    model."""
    arm = _embedding_arm()  # declares 1024
    payload = _embedding_payload([0.1] * R2_DIMS)

    result = lms_healthcheck.verify_embedding_response(arm, payload)

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.DIMS_MISMATCH
    assert '1024' in result.detail and '768' in result.detail


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_a_non_finite_value_anywhere_in_the_vector_fails(bad):
    arm = _embedding_arm()
    vector = [0.1] * PROBE_DIMS
    vector[512] = bad

    result = lms_healthcheck.verify_embedding_response(arm, _embedding_payload(vector))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.NON_FINITE_EMBEDDING
    assert '512' in result.detail


def test_an_all_zero_vector_fails():
    """A degenerate output a length check ALONE would pass.  An all-zero
    vector has undefined cosine similarity against everything, so iota's
    retrieval scores would be noise rather than an error."""
    arm = _embedding_arm()

    result = lms_healthcheck.verify_embedding_response(
        arm, _embedding_payload([0.0] * PROBE_DIMS)
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.DEGENERATE_EMBEDDING


def test_a_non_numeric_value_in_the_vector_fails_rather_than_raising():
    arm = _embedding_arm()
    vector: list = [0.1] * PROBE_DIMS
    vector[7] = 'nope'

    result = lms_healthcheck.verify_embedding_response(arm, _embedding_payload(vector))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.NON_FINITE_EMBEDDING


def test_a_body_without_a_data_list_fails_rather_than_raising():
    for payload in ({'error': 'model not found'}, {'data': 'nope'}, {}):
        result = lms_healthcheck.verify_embedding_response(_embedding_arm(), payload)
        assert result.verdict == 'FAIL'
        assert result.reason == lms_healthcheck.Reason.MALFORMED_RESPONSE


def test_a_vector_that_is_not_a_list_fails_rather_than_raising():
    payload = {
        'object': 'list',
        'model': 'qwen3-embedding-0.6b',
        'data': [{'object': 'embedding', 'index': 0, 'embedding': 'base64-blob'}],
    }

    result = lms_healthcheck.verify_embedding_response(_embedding_arm(), payload)

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.MALFORMED_RESPONSE


def test_a_response_echoing_a_different_model_fails():
    """Defence in depth behind the /v1/models gate: the OpenAI embeddings
    response echoes the model that answered, and a mismatch there is the same
    stale-unit-on-a-colliding-port hazard as the 2026-04-08 404 bug."""
    arm = _embedding_arm()
    payload = _embedding_payload([0.1] * PROBE_DIMS, model='qwen3-embedding-4b')

    result = lms_healthcheck.verify_embedding_response(arm, payload)

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.IDENTITY_MISMATCH
    assert 'qwen3-embedding-4b' in result.detail


def test_every_embedding_failure_reason_is_distinct():
    arm = _embedding_arm()
    bodies = [
        {'object': 'list', 'model': arm.served_model_name, 'data': []},
        _embedding_payload([0.1] * R2_DIMS),
        _embedding_payload([0.0] * PROBE_DIMS),
        {'error': 'nope'},
    ]
    vector = [0.1] * PROBE_DIMS
    vector[0] = float('nan')
    bodies.append(_embedding_payload(vector))

    reasons = {
        lms_healthcheck.verify_embedding_response(arm, body).reason for body in bodies
    }

    assert len(reasons) == 5
    assert lms_healthcheck.Reason.OK not in reasons


# ---------------------------------------------------------------------------
# End-to-end probe + transport
# ---------------------------------------------------------------------------


def test_probe_embedding_arm_passes_and_puts_the_prefix_on_the_wire(
    install_fake_httpx,
):
    arm = _embedding_arm()
    seen = {}

    def fake_get(url, **kwargs):
        seen['get'] = url
        return _Resp(200, _models_payload('qwen3-embedding-0.6b'))

    def fake_post(url, **kwargs):
        seen['post'] = url
        seen['body'] = kwargs.get('json')
        seen['timeout'] = kwargs.get('timeout')
        return _Resp(200, _embedding_payload([0.01 * i for i in range(PROBE_DIMS)]))

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_embedding_arm(arm)

    assert result.verdict == 'PASS'
    assert seen['get'] == 'http://127.0.0.1:8414/v1/models'
    assert seen['post'] == 'http://127.0.0.1:8414/v1/embeddings'
    assert 'localhost' not in seen['post']
    # The prefix must reach the WIRE, not merely exist in the manifest.
    assert seen['body']['input'][0].startswith('Instruct:')
    assert isinstance(seen['timeout'], float)


def test_probe_embedding_arm_checks_identity_before_the_vector(install_fake_httpx):
    def fake_get(url, **kwargs):
        return _Resp(200, _models_payload('gte-modernbert-base'))

    def fake_post(url, **kwargs):
        raise AssertionError('identity must be checked before the embeddings call')

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_embedding_arm(_embedding_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.IDENTITY_MISMATCH


def test_probe_embedding_arm_fails_on_a_transport_error(install_fake_httpx):
    def fake_get(url, **kwargs):
        raise OSError('[Errno 111] Connection refused')

    install_fake_httpx(post=None, get=fake_get)

    result = lms_healthcheck.probe_embedding_arm(_embedding_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.TRANSPORT_ERROR


def test_probe_embedding_arm_fails_on_a_non_200(install_fake_httpx):
    def fake_get(url, **kwargs):
        return _Resp(200, _models_payload('qwen3-embedding-0.6b'))

    def fake_post(url, **kwargs):
        return _Resp(503, {'error': 'loading'})

    install_fake_httpx(post=fake_post, get=fake_get)

    result = lms_healthcheck.probe_embedding_arm(_embedding_arm())

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.HTTP_STATUS
    assert '503' in result.detail


# ===========================================================================
# Part 3 (step 15) -- report assembly, table rendering, CLI and exit codes.
#
# This is the layer an operator and a downstream task actually consume, and it
# has one failure mode worse than any wrong verdict: a report that LOOKS
# complete while being partial or stale.  Three properties defend against it.
#
# 1.  The human-readable table is rendered FROM the report object and takes
#     nothing else, so the text and the JSON cannot drift apart.  A table that
#     recomputed anything could show PASS beside a FAIL row.
#
# 2.  A broken GPU probe raises rather than degrading.  `used_mib = 0` off a
#     missing nvidia-smi would render a PASSING vram block with maximal
#     headroom -- the single most trustworthy-looking wrong answer this rig
#     can produce.
#
# 3.  The exit code distinguishes "an arm is broken" from "the budget is
#     blown": they have different fixes, and a caller that only sees non-zero
#     has to re-diagnose from scratch.  `--active` with nothing running is its
#     own outcome too, so a sweep that measured NOTHING can never be read as a
#     sweep where everything passed.
# ===========================================================================


MEASURED_TOTAL_MIB = 24576
MEASURED_USED_MIB = 7362
MEASURED_FREE_MIB = 16761


#: The arm's OWN container, as nvidia-smi sees it: arms run as docker
#: containers and nvidia-smi reports HOST pids, so a containerised vLLM appears
#: as an ordinary `python` compute app indistinguishable from any other.  That
#: is why probe-time pollution cannot be judged by the baseline's allowlist.
ARM_CONSUMER = lms_vram.GpuConsumer(
    pid=910001, process_name='python', used_mib=MEASURED_USED_MIB - 3312,
)

#: ollama as measured on this host 2026-08-06, qwen3:14b resident on
#: keep_alive.  The reading that motivated the pollution guard.
OLLAMA_CONSUMER = lms_vram.GpuConsumer(
    pid=905936, process_name='/usr/local/lib/ollama/llama-server', used_mib=10314,
)

#: whisper-writer, as measured on this host: the ONE compute app PRD D10
#: requires resident before an arm starts, and the whole of EXPECTED_CONSUMERS.
WHISPER_CONSUMER = lms_vram.GpuConsumer(
    pid=7575, process_name='python', used_mib=4050,
)


def _snapshot(
    used_mib=MEASURED_USED_MIB,
    total_mib=MEASURED_TOTAL_MIB,
    free_mib=MEASURED_FREE_MIB,
    consumers=None,
) -> lms_vram.GpuSnapshot:
    """The measured host reading, as one injected GPU snapshot.

    The default inventory is the arm's own container and nothing else: the
    clean case, and the one every pre-existing test in this file assumes.
    """
    return lms_vram.GpuSnapshot(
        identity=lms_vram.GpuIdentity(
            name='NVIDIA GeForce RTX 3090', driver_version='580.159.04',
        ),
        reading=lms_vram.GpuReading(
            total_mib=total_mib, used_mib=used_mib, free_mib=free_mib,
        ),
        consumers=[ARM_CONSUMER] if consumers is None else consumers,
    )


#: The card immediately BEFORE the arm started: the KDE/X11 desktop and nothing
#: else.  Every budget verdict subtracts this (esc-3713-6), so the default
#: report below describes an arm that took 7362 - 3312 = 4050 MiB.
BASELINE_USED_MIB = 3312
BASELINE_FREE_MIB = 20811
MEASURED_FOOTPRINT_MIB = MEASURED_USED_MIB - BASELINE_USED_MIB


#: A fixed stamp, so the fixture describes one specific past moment rather than
#: drifting with the clock.  Aware UTC for the same reason `measured_at` is.
BASELINE_MEASURED_AT = _datetime.datetime(
    2026, 8, 6, 9, 30, tzinfo=_datetime.UTC,
)


def _baseline(
    used_mib=BASELINE_USED_MIB, free_mib=BASELINE_FREE_MIB, consumers=None,
    coresident_arms=(),
) -> lms_vram.GpuBaseline:
    """The pre-start card, as one RECORD: the reading and the inventory.

    The default inventory is EMPTY, and that is the measured truth rather than
    a convenience: 3312 MiB of KDE/X11 graphics contexts hold this card at
    baseline and NONE of them are CUDA compute applications, so
    `--query-compute-apps` lists nothing at all.  The default fixture is
    therefore itself an instance of CONSUMER_INVENTORY_NOTE's point -- a
    non-zero reading beside an empty inventory.
    """
    return lms_vram.GpuBaseline(
        reading=lms_vram.GpuReading(
            total_mib=MEASURED_TOTAL_MIB, used_mib=used_mib, free_mib=free_mib,
        ),
        consumers=[] if consumers is None else consumers,
        measured_at=BASELINE_MEASURED_AT,
        coresident_arms=list(coresident_arms),
    )


def _over_budget_snapshot() -> lms_vram.GpuSnapshot:
    """24400 MiB used against a 3312 MiB baseline: the ARM took 21088 MiB,
    more than the 20811 MiB that was free before it started."""
    return _snapshot(used_mib=24400, free_mib=MEASURED_TOTAL_MIB - 24400)


#: Both fakes ACCEPT `warmup` rather than ignoring it (task 3781).  A fake with
#: a bare `(arm)` signature could not be called with the kwarg at all, so it
#: would turn "the warm-up was never fired" into a TypeError somewhere else
#: instead of a legible assertion here.
def _passing_probe(arm, *, warmup: bool = False):
    return lms_healthcheck.ProbeResult(
        verdict='PASS', reason=lms_healthcheck.Reason.OK, detail='ok', latency_ms=12.5,
    )


def _failing_probe(arm, *, warmup: bool = False):
    return lms_healthcheck.ProbeResult(
        verdict='FAIL',
        reason=lms_healthcheck.Reason.IDENTITY_MISMATCH,
        detail='port 8410 serves something else entirely',
        latency_ms=3.0,
    )


def _report(arms=None, probe=_passing_probe, snapshot=None, baseline=None, **kwargs):
    return lms_healthcheck.run_healthcheck(
        arms if arms is not None else [_arm()],
        gpu_probe=lambda: snapshot if snapshot is not None else _snapshot(),
        probe=probe,
        baseline=baseline if baseline is not None else _baseline(),
        **kwargs,
    )


def _recording_probe(result=None):
    """A prober that logs `(arm_id, warmup)` per call.  Returns (fn, calls)."""
    calls: list[tuple[str, bool]] = []

    def probe(arm, *, warmup: bool = False):
        calls.append((arm.arm_id, warmup))
        return (result or _passing_probe)(arm, warmup=warmup)

    return probe, calls


# ---------------------------------------------------------------------------
# The discarded warm-up (task 3781)
#
# `run_healthcheck` fires TWO probes per arm and keeps one.  The first is
# thrown away entirely -- it exists only to warm CUDA graphs, the allocator,
# kernel autotune and grammar compilation, so the SECOND one measures a warm
# engine hitting a cold prefix cache, which is the state production is in.
#
# "Discarded" has to be load-bearing in both directions, and the two tests
# below pin both: a failing warm-up must not fail a healthy arm, and a passing
# warm-up must not launder a broken one.
# ---------------------------------------------------------------------------


def test_the_warmup_probe_is_fired_before_the_measured_one():
    probe, calls = _recording_probe()

    _report(probe=probe)

    assert calls == [('qwen3.5-9b', True), ('qwen3.5-9b', False)]


def test_a_failing_warmup_never_reaches_the_row():
    """The first request after an arm reaches ready is the one most likely to
    look bad for reasons that are not the arm's fault.  Its verdict is thrown
    away, not merged in."""
    def probe(arm, *, warmup: bool = False):
        return _failing_probe(arm) if warmup else _passing_probe(arm)

    report = _report(probe=probe)

    assert report.arms[0].verdict == 'PASS'
    assert report.arms[0].reason == lms_healthcheck.Reason.OK
    assert report.overall == 'PASS'


def test_a_passing_warmup_never_launders_a_failing_measured_probe():
    """The inverse, and the one that matters more: without it, "discard the
    first result" is a hole a broken arm could walk through."""
    def probe(arm, *, warmup: bool = False):
        return _passing_probe(arm) if warmup else _failing_probe(arm)

    report = _report(probe=probe)

    assert report.arms[0].verdict == 'FAIL'
    assert report.arms[0].reason == lms_healthcheck.Reason.IDENTITY_MISMATCH
    assert report.overall == 'FAIL'
    assert lms_healthcheck.exit_code_for(report) == lms_healthcheck.EXIT_ARM_FAILED


def test_the_warmup_is_fired_once_per_arm_not_once_per_sweep():
    """A per-sweep warm-up warms NOTHING for arms two onward: each arm is a
    separate server process with its own CUDA context, so being warmed by a
    different arm's process is the same as not being warmed at all."""
    probe, calls = _recording_probe()

    _report(arms=[_arm(), _unprefixed_arm()], probe=probe)

    assert calls == [
        ('qwen3.5-9b', True),
        ('qwen3.5-9b', False),
        ('granite-embedding-english-r2', True),
        ('granite-embedding-english-r2', False),
    ]


def test_a_placeholder_arm_still_issues_no_request_at_all(install_fake_httpx,
                                                          monkeypatch, tmp_path):
    """The warm-up must not re-open the refusal path.

    A TBD arm has nothing to probe, and issuing the request anyway would report
    a 404 on a literal `TBD-Q3` model id as an arm failure -- burying an
    unresolved PRD Open Question under a transport error like every other one.
    Doubling the requests per arm is exactly the change that could reintroduce
    it, so this drives the REAL prober through the sweep.
    """
    def _boom(url, **kwargs):
        raise AssertionError('no request may be issued for a placeholder arm')

    install_fake_httpx(post=_boom, get=_boom)
    placeholder = _moe_arm(
        model_ref='TBD-Q3-pick-a-gguf', image='TBD-Q3', quant='TBD-Q3'
    )
    assert placeholder.is_placeholder is True

    report = lms_healthcheck.run_healthcheck(
        [placeholder],
        gpu_probe=lambda: _snapshot(),
        baseline=_baseline(),
    )

    assert report.arms[0].reason == lms_healthcheck.Reason.PLACEHOLDER_ARM
    assert report.arms[0].verdict == 'FAIL'


# ---------------------------------------------------------------------------
# Both numbers, unambiguously named (task 3781)
#
# Before this task the row carried ONE latency under a name that did not say
# which of the two very different things it was.  It was in fact the cold one:
# the first request after the arm reached ready.  Both are now on the row under
# names that mean what they say.
#
# Every latency below is INJECTED.  Nothing here asserts cold > warm -- see
# test_a_row_carries_both_the_cold_and_the_warm_latency for why that ordering
# is deliberately not a property of this instrument.
# ---------------------------------------------------------------------------


def _latency_probe(cold: float, warm: float):
    def probe(arm, *, warmup: bool = False):
        return lms_healthcheck.ProbeResult(
            verdict='PASS',
            reason=lms_healthcheck.Reason.OK,
            detail='ok',
            latency_ms=cold if warmup else warm,
        )

    return probe


def test_a_row_carries_both_the_cold_and_the_warm_latency():
    """The measured qwen3.5-9b shape: ~12x between the two.

    Deliberately NOT asserted here, or anywhere: that cold > warm in general.
    qwen3.5-9b at `reasoning: on` measured 43.5 s cold against 41.0 s warm --
    a generation-dominated arm where load cost is a rounding error against the
    generation itself and the ordering sits inside the noise.  A gate on it
    would fail an arm that is serving correctly.
    """
    report = _report(probe=_latency_probe(cold=4249.7, warm=359.1))
    row = report.arms[0]

    assert row.first_probe_ms == 4249.7
    assert row.latency_ms == 359.1
    assert row.first_probe_ms != row.latency_ms
    assert row.first_probe_ms > 0 and row.latency_ms > 0


def test_latency_ms_is_the_measured_probe_not_the_first_one():
    """The anti-regression assertion.

    Before this task `latency_ms` WAS the first probe after ready.  Pinning
    which call it comes from means a later refactor that swaps the two turns
    this red instead of silently restoring a cold number under a warm name.
    """
    report = _report(probe=_latency_probe(cold=4249.7, warm=359.1))

    assert report.arms[0].latency_ms == 359.1
    assert report.arms[0].latency_ms != 4249.7


def test_the_two_latency_fields_are_named_for_what_they_measure():
    """Both present on the dumped row, so the artifact never carries one number
    whose meaning depends on knowing the run order that produced it."""
    dumped = _report().arms[0].model_dump(mode='json')

    assert 'first_probe_ms' in dumped
    assert 'latency_ms' in dumped


def test_a_placeholder_row_reports_both_fields_without_inventing_a_measurement():
    """A refused arm measured nothing.  Both fields are 0.0 rather than either
    being omitted -- an absent field reads as an older schema, and a nonzero
    one would be a measurement nobody took."""
    placeholder = _moe_arm(
        model_ref='TBD-Q3-pick-a-gguf', image='TBD-Q3', quant='TBD-Q3'
    )

    def probe(arm, *, warmup: bool = False):
        return lms_healthcheck.ProbeResult(
            verdict='FAIL',
            reason=lms_healthcheck.Reason.PLACEHOLDER_ARM,
            detail='nothing to probe',
        )

    row = _report(arms=[placeholder], probe=probe).arms[0]

    assert row.first_probe_ms == 0.0
    assert row.latency_ms == 0.0


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------


def test_the_report_carries_a_schema_version():
    """Step 21's verification test and every downstream consumer key off this.

    Without it, a later shape change silently reinterprets an old artifact
    rather than rejecting it.
    """
    report = _report()

    assert report.schema_version == lms_healthcheck.REPORT_SCHEMA_VERSION
    assert isinstance(report.schema_version, int)


def test_the_report_is_stamped_with_an_aware_utc_timestamp():
    """A naive timestamp would make a stale artifact indistinguishable from a
    fresh one across a timezone change -- and this artifact's whole job is to
    prove a live run happened."""
    report = _report()

    stamped = _datetime.datetime.fromisoformat(report.measured_at)

    assert stamped.tzinfo is not None
    assert stamped.utcoffset() == _datetime.timedelta(0)


def test_the_report_carries_a_gpu_identity_block():
    """Which card, which driver.  An arm's numbers are meaningless without it:
    the same manifest on a different GPU produces different verdicts, and the
    artifact has to say which host it was measured on."""
    report = _report()

    assert report.gpu.name == 'NVIDIA GeForce RTX 3090'
    assert report.gpu.driver_version == '580.159.04'
    assert report.gpu.total_mib == MEASURED_TOTAL_MIB


def test_there_is_one_row_per_arm_carrying_the_contract_fields():
    arms = [_arm(), _unprefixed_arm()]

    report = _report(arms=arms)

    assert [row.arm_id for row in report.arms] == ['qwen3.5-9b',
                                                   'granite-embedding-english-r2']
    row = report.arms[0]
    assert row.axis == 'llm'
    assert row.stack == 'vllm'
    assert row.served_model_name == 'qwen3.5-9b'
    assert row.verdict == 'PASS'
    assert row.reason == lms_healthcheck.Reason.OK
    assert row.latency_ms == 12.5


def test_a_row_endpoint_is_the_arms_loopback_base_url():
    """127.0.0.1, never `localhost`: the latter can resolve to ::1 while the
    server listens on IPv4 only, which presents as a dead arm
    (scripts/run_vllm_eval.py:505-512)."""
    report = _report()

    assert report.arms[0].endpoint == 'http://127.0.0.1:8410'
    assert 'localhost' not in report.arms[0].endpoint


def test_the_vram_block_reports_both_budget_figures_and_the_free_reading():
    """PRD D10's nominal ceiling AND the measured operating budget travel
    together, because this host's real budget (~16.4 GiB) is smaller than the
    PRD assumed and a report showing only one of the two figures either hides
    the finding or asserts capacity that does not exist."""
    report = _report()

    vram = report.vram
    assert vram.total_mib == MEASURED_TOTAL_MIB
    assert vram.used_mib == MEASURED_USED_MIB
    assert vram.free_mib == MEASURED_FREE_MIB
    assert vram.nominal_ceiling_gib == lms_vram.NOMINAL_CEILING_GIB
    assert vram.operating_budget_gib == lms_vram.MEASURED_OPERATING_BUDGET_GIB
    assert vram.nominal_ceiling_gib != vram.operating_budget_gib
    assert vram.headroom_gib > 0
    assert vram.verdict == 'PASS'


def test_the_vram_block_fails_when_usage_exceeds_the_nominal_ceiling():
    report = _report(snapshot=_over_budget_snapshot())

    assert report.vram.verdict == 'FAIL'
    assert report.vram.headroom_gib < 0


def test_overall_is_pass_only_when_every_row_and_the_vram_block_pass():
    report = _report(arms=[_arm(), _unprefixed_arm()])

    assert all(row.verdict == 'PASS' for row in report.arms)
    assert report.vram.verdict == 'PASS'
    assert report.overall == 'PASS'


def test_overall_is_fail_when_a_single_arm_fails():
    def probe(arm, *, warmup: bool = False):
        return (
            _failing_probe(arm, warmup=warmup) if arm.arm_id == 'qwen3.5-9b'
            else _passing_probe(arm, warmup=warmup)
        )

    report = _report(arms=[_arm(), _unprefixed_arm()], probe=probe)

    assert report.vram.verdict == 'PASS'
    assert report.overall == 'FAIL'


def test_overall_is_fail_when_only_the_vram_block_fails():
    """Every arm answering correctly while the card is over budget is still a
    failed run: the PRD's user-observable signal is nvidia-smi WITHIN the
    budget, and an overall PASS here would certify a state that evicts
    whisper-writer."""
    report = _report(snapshot=_over_budget_snapshot())

    assert all(row.verdict == 'PASS' for row in report.arms)
    assert report.overall == 'FAIL'


def test_a_dead_arm_does_not_abort_the_sweep(install_fake_httpx):
    """Measured verdicts for the other arms must survive the first dead one.

    Otherwise the report is both incomplete AND silent about being
    incomplete -- it would simply be missing rows nobody asked after.
    """
    dead, alive = _arm(), _unprefixed_arm()

    def fake_get(url, **kwargs):
        if ':8410' in url:
            raise OSError('[Errno 111] Connection refused')
        return _Resp(200, _models_payload('granite-embedding-english-r2'))

    def fake_post(url, **kwargs):
        return _Resp(
            200,
            _embedding_payload(
                [0.01 * i for i in range(R2_DIMS)],
                model='granite-embedding-english-r2',
            ),
        )

    install_fake_httpx(post=fake_post, get=fake_get)

    report = lms_healthcheck.run_healthcheck(
        [dead, alive], gpu_probe=lambda: _snapshot(), baseline=_baseline(),
    )

    assert [row.arm_id for row in report.arms] == [
        'qwen3.5-9b', 'granite-embedding-english-r2',
    ]
    assert report.arms[0].verdict == 'FAIL'
    assert report.arms[0].reason == lms_healthcheck.Reason.TRANSPORT_ERROR
    assert report.arms[1].verdict == 'PASS'
    assert report.overall == 'FAIL'


def test_an_unparseable_gpu_probe_propagates_the_typed_error(install_fake_httpx):
    """No report at all beats a report with a passing VRAM block.

    A swallowed probe failure would render `used 0 MiB, headroom 19.5 GiB` --
    the most trustworthy-looking wrong answer this rig can produce, and the
    one an operator is least likely to question.
    """
    def exploding_probe():
        raise lms_vram.VramProbeError('nvidia-smi returned no memory rows')

    with pytest.raises(lms_vram.VramProbeError):
        lms_healthcheck.run_healthcheck(
            [_arm()], gpu_probe=exploding_probe, probe=_passing_probe,
            baseline=_baseline(),
        )


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------


def test_exit_code_is_zero_only_when_every_row_and_the_vram_block_pass():
    assert lms_healthcheck.exit_code_for(_report()) == 0


def test_an_arm_failure_and_a_vram_failure_have_distinct_exit_codes():
    """Different diagnoses, different fixes.  Collapsing both to 1 costs the
    caller the whole diagnosis again."""
    arm_failed = _report(probe=_failing_probe)
    vram_failed = _report(snapshot=_over_budget_snapshot())

    arm_code = lms_healthcheck.exit_code_for(arm_failed)
    vram_code = lms_healthcheck.exit_code_for(vram_failed)

    assert arm_code != 0
    assert vram_code != 0
    assert arm_code != vram_code
    assert arm_code == lms_healthcheck.EXIT_ARM_FAILED
    assert vram_code == lms_healthcheck.EXIT_VRAM_FAILED


def test_an_arm_failure_dominates_a_simultaneous_vram_failure():
    report = _report(probe=_failing_probe, snapshot=_over_budget_snapshot())

    assert lms_healthcheck.exit_code_for(report) == lms_healthcheck.EXIT_ARM_FAILED


def _polluted_report(probe=_passing_probe, used_mib=MEASURED_USED_MIB,
                     free_mib=MEASURED_FREE_MIB):
    """A run ollama gatecrashed: the numbers exist but mean nothing."""
    return _report(
        probe=probe,
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(used_mib=used_mib, free_mib=free_mib,
                           consumers=[WHISPER_CONSUMER, ARM_CONSUMER,
                                      OLLAMA_CONSUMER]),
    )


def test_a_polluted_measurement_has_its_own_exit_code():
    """"Over budget" and "unmeasurable" send an operator in opposite directions.

    EXIT_VRAM_FAILED means the arm is genuinely too big and something should be
    stopped. A polluted run says nothing about the arm at all -- stopping an arm
    in response would be acting on a number nobody measured.
    """
    report = _polluted_report()

    code = lms_healthcheck.exit_code_for(report)

    assert code == lms_healthcheck.EXIT_VRAM_POLLUTED
    assert code != lms_healthcheck.EXIT_VRAM_FAILED
    assert code != lms_healthcheck.EXIT_ARM_FAILED
    assert code != lms_healthcheck.EXIT_OK


def test_a_polluted_measurement_is_never_a_pass():
    """The block's own verdict may well say PASS -- computed from arithmetic
    that is void.  The exit code is the only thing a CI caller reads."""
    report = _polluted_report()

    assert report.vram.verdict == 'PASS'
    assert lms_healthcheck.exit_code_for(report) != lms_healthcheck.EXIT_OK


def test_pollution_outranks_a_plain_budget_failure():
    """A budget FAIL measured on a contended card is not a budget FAIL.

    Reporting code 3 here would send an operator to shrink an arm that may fit
    perfectly well once ollama releases the card.
    """
    report = _polluted_report(used_mib=24400,
                              free_mib=MEASURED_TOTAL_MIB - 24400)

    assert report.vram.verdict == 'FAIL'
    assert lms_healthcheck.exit_code_for(report) == (
        lms_healthcheck.EXIT_VRAM_POLLUTED
    )


def test_an_arm_failure_still_dominates_a_polluted_measurement():
    """Unchanged precedence, for the reason already documented: an arm that
    answered WRONGLY is the more actionable finding, and pollution cannot
    explain a wrong answer -- it only voids the VRAM arithmetic, which is right
    there in the artifact either way."""
    report = _polluted_report(probe=_failing_probe)

    assert report.vram.pollution == lms_vram.PollutionState.POLLUTED
    assert lms_healthcheck.exit_code_for(report) == lms_healthcheck.EXIT_ARM_FAILED


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def test_the_table_takes_only_the_report_so_text_and_json_cannot_disagree():
    """A structural guarantee, not a hopeful one: given no arms, no HTTP and
    no GPU, `render_table` has nothing left to recompute."""
    signature = inspect.signature(lms_healthcheck.render_table)

    assert list(signature.parameters) == ['report']


def test_the_table_shows_every_row_with_its_verdict_and_reason():
    def probe(arm, *, warmup: bool = False):
        return (
            _failing_probe(arm, warmup=warmup) if arm.arm_id == 'qwen3.5-9b'
            else _passing_probe(arm, warmup=warmup)
        )

    report = _report(arms=[_arm(), _unprefixed_arm()], probe=probe)

    table = lms_healthcheck.render_table(report)

    for row in report.arms:
        assert row.arm_id in table
        assert row.verdict in table
    assert 'identity_mismatch' in table
    assert 'FAIL' in table


def test_the_table_follows_the_report_when_a_verdict_changes():
    """The anti-drift check: edit the structure, the text must move with it."""
    report = _report()
    assert 'FAIL' not in lms_healthcheck.render_table(report)

    flipped = report.model_copy(
        update={
            'arms': [
                report.arms[0].model_copy(
                    update={
                        'verdict': 'FAIL',
                        'reason': lms_healthcheck.Reason.EMPTY_COMPLETION,
                    }
                )
            ],
            'overall': 'FAIL',
        }
    )

    table = lms_healthcheck.render_table(flipped)

    assert 'FAIL' in table
    assert 'empty_completion' in table


def test_the_table_shows_both_vram_figures():
    table = lms_healthcheck.render_table(_report())

    assert str(lms_vram.NOMINAL_CEILING_GIB) in table
    assert str(lms_vram.MEASURED_OPERATING_BUDGET_GIB) in table


def test_the_table_lists_who_else_held_the_card_at_each_reading():
    """The operator reading the terminal must see what the JSON carries.

    Otherwise the inventory is present in the artifact and absent from the only
    output a human actually looks at, and "check nvidia-smi first" stays an
    unwritten discipline that eta/theta/iota have no way to inherit.
    """
    table = lms_healthcheck.render_table(_report(
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(consumers=[WHISPER_CONSUMER, ARM_CONSUMER]),
    ))

    # Two SECTIONS, not one merged list: whisper-writer held the card at both
    # readings and so appears twice, the arm only at the probe.  A structural
    # count rather than a prose match -- `'baseline' in table` was already true
    # before this section existed, from the footprint line's "7362 used - 3312
    # baseline".
    assert table.count(str(WHISPER_CONSUMER.pid)) == 2
    assert table.count(str(ARM_CONSUMER.pid)) == 1
    for consumer in (WHISPER_CONSUMER, ARM_CONSUMER):
        assert consumer.process_name in table
        assert str(consumer.pid) in table
        assert str(consumer.used_mib) in table


def test_the_table_shows_an_empty_inventory_as_a_measured_fact():
    """`(none)` and not a blank line.

    A silently absent section is indistinguishable from a section this build
    does not render -- and an empty compute-app list beside a 3312 MiB baseline
    is a real, explainable reading, not a missing one.
    """
    table = lms_healthcheck.render_table(_report(
        baseline=_baseline(consumers=[]),
        snapshot=_snapshot(consumers=[ARM_CONSUMER]),
    ))

    assert '(none)' in table


def test_the_table_banners_a_polluted_run_and_names_the_intruder():
    """The loudest thing in the output, because it is the only thing that
    matters: every VRAM number above it is void."""
    table = lms_healthcheck.render_table(_polluted_report())

    assert 'POLLUTED' in table
    assert OLLAMA_CONSUMER.process_name in table
    assert str(OLLAMA_CONSUMER.pid) in table


def test_a_clean_run_shows_no_pollution_banner():
    """The banner must mean something.  Printed on every run it would be
    ignored on the one run it matters."""
    table = lms_healthcheck.render_table(_report())

    assert 'POLLUTED' not in table


def test_the_table_follows_the_report_when_the_pollution_state_changes():
    """The anti-drift property, extended to the new field.

    `render_table` takes the report and nothing else, so editing the STRUCTURE
    must move the text -- proving the banner is rendered from the block and not
    recomputed from the consumer lists beside it.
    """
    report = _report()
    assert 'POLLUTED' not in lms_healthcheck.render_table(report)

    flipped = report.model_copy(
        update={
            'vram': report.vram.model_copy(
                update={
                    'pollution': lms_vram.PollutionState.POLLUTED,
                    'pollution_reason': 'pid 4242 /usr/local/lib/ollama/'
                                        'llama-server arrived mid-run',
                }
            )
        }
    )

    table = lms_healthcheck.render_table(flipped)

    assert 'POLLUTED' in table
    assert '4242' in table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Patch the CLI's four seams: the GPU, the baselines, the prober, systemd.

    The baseline store is a real directory with real files, populated as
    `lms_ctl.start` would: the CLI has no baseline parameter on purpose, so the
    only way a report gets one is that something actually started the arm.
    """
    calls = {'probed': [], 'calls': []}

    def probe(arm, *, warmup: bool = False):
        # `probed` stays the MEASURED sweep, so every pre-3781 assertion about
        # which arms were covered keeps meaning what it said.  `calls` is the
        # full wire sequence, warm-ups included, for the tests that care.
        calls['calls'].append((arm.arm_id, warmup))
        if not warmup:
            calls['probed'].append(arm.arm_id)
        return _passing_probe(arm, warmup=warmup)

    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path / 'baselines'))
    for arm_id in lms_manifest.load_arms().arm_ids():
        record = _baseline()
        lms_vram.record_baseline(
            arm_id, record.reading, consumers=record.consumers,
        )
    monkeypatch.setattr(lms_vram, 'probe_gpu_snapshot', lambda *a, **k: _snapshot())
    monkeypatch.setattr(lms_healthcheck, 'probe_arm', probe)
    monkeypatch.setattr(lms_ctl, 'active_arms', lambda: set())
    return calls


def test_cli_all_covers_every_arm_in_the_committed_manifest(cli_env, capsys):
    expected = lms_manifest.load_arms().arm_ids()

    code = lms_healthcheck.main(['--all'])

    assert code == 0
    assert cli_env['probed'] == expected
    assert len(expected) == 7


def test_cli_arm_selects_exactly_one(cli_env, capsys):
    code = lms_healthcheck.main(['--arm', 'qwen3.5-9b'])

    assert code == 0
    assert cli_env['probed'] == ['qwen3.5-9b']


def test_cli_arm_rejects_an_unknown_id_loudly(cli_env, capsys):
    code = lms_healthcheck.main(['--arm', 'no-such-arm'])

    assert code == lms_healthcheck.EXIT_MANIFEST_ERROR
    assert cli_env['probed'] == []
    assert 'no-such-arm' in capsys.readouterr().err


def test_cli_active_covers_only_the_running_arms(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(lms_ctl, 'active_arms', lambda: {'phi-4-14b'})

    code = lms_healthcheck.main(['--active'])

    assert code == 0
    assert cli_env['probed'] == ['phi-4-14b']


def test_cli_active_with_nothing_running_is_a_distinct_non_crash_outcome(
    cli_env, capsys, tmp_path,
):
    """A sweep that measured NOTHING must never exit 0.

    `--active` is the natural thing to put in a wrapper script, and an empty
    unit list returning success would certify a slate nobody probed.
    """
    out_path = tmp_path / 'health-report.json'

    code = lms_healthcheck.main(['--active', '--output', str(out_path)])

    assert code == lms_healthcheck.EXIT_NO_ACTIVE_ARMS
    assert code != 0
    assert 'no active arms' in capsys.readouterr().out.lower()
    assert not out_path.exists()


def test_cli_requires_a_selector(cli_env):
    with pytest.raises(SystemExit):
        lms_healthcheck.main([])


def test_cli_output_writes_the_json_artifact_step_21_validates(cli_env, tmp_path):
    out_path = tmp_path / 'verification' / 'health-report.json'

    code = lms_healthcheck.main(['--all', '--output', str(out_path)])

    assert code == 0
    written = json.loads(out_path.read_text())

    assert written['schema_version'] == lms_healthcheck.REPORT_SCHEMA_VERSION
    assert written['overall'] == 'PASS'
    assert written['gpu']['name'] == 'NVIDIA GeForce RTX 3090'
    assert {row['arm_id'] for row in written['arms']} == set(
        lms_manifest.load_arms().arm_ids()
    )
    assert all(row['verdict'] == 'PASS' for row in written['arms'])
    for key in (
        'total_mib', 'used_mib', 'free_mib', 'baseline_mib', 'budget_mib',
        'arm_footprint_mib', 'nominal_ceiling_gib', 'operating_budget_gib',
        'headroom_gib', 'verdict',
    ):
        assert key in written['vram']
    for key in ('arm_id', 'axis', 'stack', 'endpoint', 'served_model_name',
                'verdict', 'reason', 'latency_ms', 'first_probe_ms'):
        assert key in written['arms'][0]


# ---------------------------------------------------------------------------
# --repeat N (task 3781)
#
# A pure OBSERVABILITY knob.  It shows an operator a spread and must never move
# a verdict -- an operator who could turn a row green by running it again would
# have the cheapest possible way to fake this artifact, which `merge_reports`
# already refuses for exactly the same reason.
#
# Samples after the FIRST are prefix-cache WARM: the first measured probe
# populates the cache the rest are served from.  Measured 2026-08-06 on
# moe-stretch, 338 of 343 prompt tokens came from cache on a repeat, and the
# cache state even changed the OUTPUT at temperature 0 (276 completion tokens
# cold-cache vs 236 warm).  Repeated identical probes are NOT independent
# samples, which is why latency_ms stays pinned to the only prefix-cold one.
# ---------------------------------------------------------------------------


def test_repeat_fires_the_measured_probe_n_times_and_the_warmup_once(cli_env):
    """Exactly ONE warm-up regardless of N.  The engine is warm after the first
    request; re-warming would only spend GPU time to learn nothing."""
    code = lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--repeat', '3'])

    assert code == 0
    assert cli_env['calls'] == [
        ('qwen3.5-9b', True),
        ('qwen3.5-9b', False),
        ('qwen3.5-9b', False),
        ('qwen3.5-9b', False),
    ]


def test_repeat_records_every_measured_sample():
    seen = {'n': 0}

    def probe(arm, *, warmup: bool = False):
        if warmup:
            return lms_healthcheck.ProbeResult(
                verdict='PASS', reason=lms_healthcheck.Reason.OK, latency_ms=4249.7,
            )
        seen['n'] += 1
        return lms_healthcheck.ProbeResult(
            verdict='PASS',
            reason=lms_healthcheck.Reason.OK,
            latency_ms=float(350 + seen['n']),
        )

    row = _report(probe=probe, repeat=3).arms[0]

    assert row.repeat_latencies_ms == [351.0, 352.0, 353.0]
    assert 4249.7 not in row.repeat_latencies_ms
    assert row.first_probe_ms == 4249.7


def test_latency_ms_is_the_first_measured_sample():
    """Not a mean.  Only the FIRST measured probe is prefix-cold; samples 2..N
    are served from the cache it populated, so averaging them would quietly
    redefine the artifact's headline number into a mean of incomparable
    things."""
    seen = {'n': 0}

    def probe(arm, *, warmup: bool = False):
        if warmup:
            return _passing_probe(arm, warmup=True)
        seen['n'] += 1
        return lms_healthcheck.ProbeResult(
            verdict='PASS',
            reason=lms_healthcheck.Reason.OK,
            latency_ms=float(1000 // seen['n']),
        )

    row = _report(probe=probe, repeat=3).arms[0]

    assert row.repeat_latencies_ms is not None
    assert row.latency_ms == row.repeat_latencies_ms[0]
    assert row.latency_ms == 1000.0


def test_repeat_never_changes_a_verdict():
    """Observability, never adjudication.  The FIRST measured probe alone
    supplies the verdict, so a later sample cannot move it in either
    direction."""
    seen = {'n': 0}

    def probe(arm, *, warmup: bool = False):
        if warmup:
            return _passing_probe(arm, warmup=True)
        seen['n'] += 1
        return _failing_probe(arm) if seen['n'] == 3 else _passing_probe(arm)

    report = _report(probe=probe, repeat=3)

    assert report.arms[0].verdict == 'PASS'
    assert report.arms[0].reason == lms_healthcheck.Reason.OK
    assert report.overall == 'PASS'
    assert lms_healthcheck.exit_code_for(report) == lms_healthcheck.EXIT_OK


def test_without_repeat_the_spread_field_is_absent():
    """None, not a one-element list.  "The question was not asked" and "a
    spread of one" are different statements, the same distinction
    top_level_entities_named already draws."""
    assert _report().arms[0].repeat_latencies_ms is None


@pytest.mark.parametrize('count', ['0', '-1'])
def test_repeat_rejects_a_non_positive_count(cli_env, count):
    """Silently probing zero times would write a report with no measurement in
    it that still reads as a completed run."""
    with pytest.raises(SystemExit) as excinfo:
        lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--repeat', count])

    assert excinfo.value.code != 0


def test_run_healthcheck_rejects_a_non_positive_repeat_as_a_caller_error():
    """A HealthcheckError, never a FAIL row: this is the harness being asked
    something incoherent, and blaming the arm for it is this module's stated
    contract violation."""
    with pytest.raises(lms_healthcheck.HealthcheckError):
        _report(repeat=0)


@pytest.mark.parametrize('selector', [['--arm', 'qwen3.5-9b'], ['--all'], ['--active']])
def test_repeat_is_not_part_of_the_selector_group(cli_env, selector):
    """A plain option, not a member of the required mutually-exclusive group,
    so it composes with every way of choosing arms."""
    code = lms_healthcheck.main([*selector, '--repeat', '2'])

    # --active finds nothing under cli_env, which is its own non-zero code.
    assert code in (0, lms_healthcheck.EXIT_NO_ACTIVE_ARMS)


def test_cli_written_artifact_is_pure_json_with_no_enum_repr(cli_env, tmp_path):
    """`Reason` is a StrEnum: dumped in python mode it would serialise as an
    object repr that no downstream JSON consumer can match on."""
    out_path = tmp_path / 'health-report.json'
    lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(out_path)])

    raw = out_path.read_text()

    assert 'Reason.' not in raw
    assert json.loads(raw)['arms'][0]['reason'] == 'ok'


def test_cli_exit_code_reflects_a_failing_arm(cli_env, monkeypatch, tmp_path):
    monkeypatch.setattr(lms_healthcheck, 'probe_arm', _failing_probe)
    out_path = tmp_path / 'health-report.json'

    code = lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(out_path)])

    assert code == lms_healthcheck.EXIT_ARM_FAILED
    # The artifact is still written: a failing run's evidence is the point.
    assert json.loads(out_path.read_text())['overall'] == 'FAIL'


def test_cli_reports_a_broken_gpu_probe_and_writes_no_artifact(cli_env, monkeypatch,
                                                               tmp_path, capsys):
    def exploding(*args, **kwargs):
        raise lms_vram.VramProbeError('nvidia-smi: command not found')

    monkeypatch.setattr(lms_vram, 'probe_gpu_snapshot', exploding)
    out_path = tmp_path / 'health-report.json'

    code = lms_healthcheck.main(['--all', '--output', str(out_path)])

    assert code == lms_healthcheck.EXIT_PROBE_ERROR
    assert 'nvidia-smi' in capsys.readouterr().err
    assert not out_path.exists()


def test_cli_refuses_a_polluted_baseline_and_writes_no_artifact(
    cli_env, tmp_path, capsys,
):
    """The baseline on disk was taken while ollama held 10314 MiB.

    `lms_ctl.start` refuses to write such a file, so this one predates the
    guard or was written by hand -- and either way its number is what every
    footprint gets subtracted from. No artifact is written even though
    `--output` was passed: a file on disk would later read as "the slate was
    checked", which is the same reasoning as the merge and no-active-arms paths.
    """
    path = lms_vram.baseline_path('qwen3.5-9b')
    payload = json.loads(path.read_text())
    payload['consumers'] = [OLLAMA_CONSUMER.model_dump(mode='json')]
    path.write_text(json.dumps(payload))
    out_path = tmp_path / 'health-report.json'

    code = lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(out_path)])

    assert code == lms_healthcheck.EXIT_VRAM_POLLUTED
    # PollutedBaselineError SUBCLASSES VramProbeError, so a handler written in
    # the obvious order would swallow it into EXIT_PROBE_ERROR and report
    # "nvidia-smi is broken" for a perfectly healthy GPU.
    assert code != lms_healthcheck.EXIT_PROBE_ERROR
    err = capsys.readouterr().err
    assert str(OLLAMA_CONSUMER.pid) in err
    assert OLLAMA_CONSUMER.process_name in err
    assert not out_path.exists()


def test_cli_reports_probe_time_pollution_but_still_writes_the_artifact(
    cli_env, monkeypatch, tmp_path,
):
    """The other pollution route, and it is NOT a refusal.

    A baseline that was clean when taken and a card that got crowded afterwards
    still produced an honest record of what was seen -- so the artifact is
    written, carrying the POLLUTED state, and the refusal is in the exit code.
    Suppressing it would destroy the only evidence that ollama was there.
    """
    monkeypatch.setattr(
        lms_vram, 'probe_gpu_snapshot',
        lambda *a, **k: _snapshot(consumers=[ARM_CONSUMER, OLLAMA_CONSUMER]),
    )
    out_path = tmp_path / 'health-report.json'

    code = lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(out_path)])

    assert code == lms_healthcheck.EXIT_VRAM_POLLUTED
    written = json.loads(out_path.read_text())
    assert written['vram']['pollution'] == 'POLLUTED'
    assert str(OLLAMA_CONSUMER.pid) in written['vram']['pollution_reason']


@pytest.mark.parametrize('mutate', ['delete', 'strip_consumers'])
def test_cli_sends_a_stale_baseline_to_lms_ctl_not_to_nvidia_smi(
    cli_env, tmp_path, capsys, mutate,
):
    """A baseline that is absent, or predates the inventory, is NOT a probe
    failure.

    Both spellings of "nothing usable was recorded" used to raise a plain
    VramProbeError and land in the generic branch, printed as "GPU probe
    failed" -- the operator-misdirection class exit 7 exists to end.  The
    stripped-consumers case is not exotic: it is what every operator with a
    pre-guard baseline still in $XDG_RUNTIME_DIR from this boot meets the first
    time they run this after the guard lands, and the fix is `lms_ctl start`.
    """
    path = lms_vram.baseline_path('qwen3.5-9b')
    if mutate == 'delete':
        path.unlink()
    else:
        payload = json.loads(path.read_text())
        del payload['consumers']
        path.write_text(json.dumps(payload))
    out_path = tmp_path / 'health-report.json'

    code = lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(out_path)])

    assert code == lms_healthcheck.EXIT_STALE_BASELINE
    assert code != lms_healthcheck.EXIT_PROBE_ERROR
    err = capsys.readouterr().err
    # The two things an operator needs: WHICH file, and WHAT to run.
    assert 'lms_ctl start qwen3.5-9b' in err
    assert 'GPU probe failed' not in err
    if mutate == 'strip_consumers':
        assert str(path) in err
    assert not out_path.exists()


def test_every_cli_exit_code_is_distinct():
    codes = [
        lms_healthcheck.EXIT_OK,
        lms_healthcheck.EXIT_ARM_FAILED,
        lms_healthcheck.EXIT_MANIFEST_ERROR,
        lms_healthcheck.EXIT_VRAM_FAILED,
        lms_healthcheck.EXIT_PROBE_ERROR,
        lms_healthcheck.EXIT_NO_ACTIVE_ARMS,
        lms_healthcheck.EXIT_MERGE_ERROR,
        lms_healthcheck.EXIT_VRAM_POLLUTED,
        lms_healthcheck.EXIT_STALE_BASELINE,
    ]

    assert len(set(codes)) == len(codes)
    assert lms_healthcheck.EXIT_OK == 0


# ---------------------------------------------------------------------------
# The budget verdict's subject (esc-3713-6)
# ---------------------------------------------------------------------------


def test_the_vram_block_charges_the_arm_only_for_what_it_took():
    """The correction, at report level.

    Under the old subject the block compared TOTAL card usage with PRD D10's
    nominal 19.5 GiB, which charged every arm a second time for the desktop
    baseline D10 had already subtracted.
    """
    vram = _report().vram

    assert vram.baseline_mib == BASELINE_USED_MIB
    assert vram.budget_mib == BASELINE_FREE_MIB
    assert vram.arm_footprint_mib == MEASURED_FOOTPRINT_MIB
    assert vram.used_mib - vram.baseline_mib == vram.arm_footprint_mib
    assert vram.verdict == 'PASS'


def test_the_nominal_ceiling_is_reported_but_no_longer_gates():
    """PRD D10's figure stays legible in the artifact and stops deciding.

    An arm at 21.75 GiB total -- the real qwen3.5-9b measurement -- used to FAIL
    while serving correctly.  It passes now, and the ceiling it exceeds is still
    right there in the block for a reader to see.
    """
    report = _report(snapshot=_snapshot(used_mib=22271, free_mib=2305),
                     baseline=_baseline(used_mib=7310, free_mib=16813))

    assert report.vram.used_gib > report.vram.nominal_ceiling_gib
    assert report.vram.arm_footprint_mib == 14961
    assert report.vram.verdict == 'PASS'


def test_a_missing_baseline_produces_no_report_at_all(monkeypatch, tmp_path):
    """Same stance as a dead GPU probe: no artifact beats a wrong one.

    Defaulting to lms_vram.MEASURED_BASELINE_GIB here would silently restore
    the frozen baseline esc-3713-6 ruled out, and the artifact would look
    identical either way.
    """
    monkeypatch.setenv(lms_vram.BASELINE_DIR_ENV, str(tmp_path / 'empty'))

    with pytest.raises(lms_vram.VramProbeError, match='no baseline'):
        lms_healthcheck.run_healthcheck(
            [_arm()], gpu_probe=lambda: _snapshot(), probe=_passing_probe,
        )


def test_every_row_carries_its_own_measurement_time_and_footprint():
    """The merged slate artifact keeps ONE vram block, so without these the
    other seven arms' measurements would leave the artifact entirely."""
    row = _report().arms[0]

    assert _datetime.datetime.fromisoformat(row.measured_at).tzinfo is not None
    assert row.arm_footprint_mib == MEASURED_FOOTPRINT_MIB


def test_the_table_shows_the_arm_footprint_and_the_budget_it_was_judged_against():
    table = lms_healthcheck.render_table(_report())

    assert str(MEASURED_FOOTPRINT_MIB) in table
    assert 'baseline' in table.lower()


# ---------------------------------------------------------------------------
# Who ELSE held the card (task 3755)
#
# `used - baseline` is only the arm's footprint if nothing else moved on the
# card between the two readings.  Nothing in the v4 artifact recorded whether
# that held, so a run with ollama resident produced numbers indistinguishable
# from a clean one.  These tests pin the evidence into the block.
# ---------------------------------------------------------------------------


def test_the_vram_block_lists_the_consumers_seen_at_each_reading():
    """Both inventories, not just one.

    A single "who is on the card now" list cannot answer the question that
    matters -- whether the SAME processes were there before -- and the drift
    between the two readings is the whole pollution signal.
    """
    report = _report(
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(consumers=[WHISPER_CONSUMER, ARM_CONSUMER]),
    )

    assert report.vram.baseline_consumers == [WHISPER_CONSUMER]
    assert report.vram.probe_consumers == [WHISPER_CONSUMER, ARM_CONSUMER]


def test_the_vram_block_labels_the_inventory_as_not_an_accounting():
    """The list does NOT sum to memory.used, and the artifact has to say so.

    `--query-compute-apps` omits every graphics context, which is exactly why
    this host's operating budget is ~16.4 GiB and not 19.5.  A reader who added
    the entries up and found a shortfall would conclude the reading was wrong.
    """
    note = _report().vram.consumer_inventory_note

    assert note == lms_vram.CONSUMER_INVENTORY_NOTE


def test_the_default_fixture_shows_why_the_inventory_is_not_an_accounting():
    """The clean baseline holds 3312 MiB with ZERO compute apps listed.

    Not an artefact of the fixture: those are KDE/X11 graphics contexts, which
    `--query-compute-apps` cannot see.  If an empty inventory were ever read as
    "the card was empty", this is the reading that would prove it wrong.
    """
    vram = _report().vram

    assert vram.baseline_consumers == []
    assert vram.baseline_mib == BASELINE_USED_MIB > 0


def test_a_clean_run_is_recorded_as_clean_with_no_reason():
    """The arm's own container is a NEW consumer at probe time and that is
    normal: arms are docker containers and nvidia-smi reports host pids, so a
    containerised vLLM is just another `python`.  Marking that POLLUTED would
    fail every healthy run."""
    report = _report(
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(consumers=[WHISPER_CONSUMER, ARM_CONSUMER]),
    )

    assert report.vram.pollution == lms_vram.PollutionState.CLEAN
    assert report.vram.pollution_reason == ''
    assert report.vram.verdict == 'PASS'


def test_an_ollama_newcomer_at_probe_time_marks_the_block_polluted():
    """The exact measured hazard: ollama holding qwen3:14b on keep_alive.

    Its 10314 MiB lands inside the probe reading and is charged straight to the
    arm by `used - baseline`.  The block has to say so, and has to name the
    process an operator would have to deal with.
    """
    report = _report(
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(consumers=[WHISPER_CONSUMER, ARM_CONSUMER,
                                      OLLAMA_CONSUMER]),
    )

    assert report.vram.pollution == lms_vram.PollutionState.POLLUTED
    reason = report.vram.pollution_reason
    assert str(OLLAMA_CONSUMER.pid) in reason
    assert OLLAMA_CONSUMER.process_name in reason
    assert str(OLLAMA_CONSUMER.used_mib) in reason
    # And the evidence is still in the block, not only in the prose.
    assert OLLAMA_CONSUMER in report.vram.probe_consumers
    assert OLLAMA_CONSUMER not in report.vram.baseline_consumers


def test_a_baseline_consumer_that_shrank_is_polluted_in_the_report_too():
    """The FLATTERING direction, end to end.

    whisper-writer releasing the card mid-run leaves `used - baseline` smaller
    than the arm truly took.  That is the direction a fabricated artifact wants,
    so it must not reach the report as a clean PASS.
    """
    shrunk = lms_vram.GpuConsumer(
        pid=WHISPER_CONSUMER.pid, process_name=WHISPER_CONSUMER.process_name,
        used_mib=100,
    )
    report = _report(
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(consumers=[shrunk, ARM_CONSUMER]),
    )

    assert report.vram.pollution == lms_vram.PollutionState.POLLUTED


def test_this_producer_never_emits_the_unmeasured_sentinel():
    """The defaults exist ONLY so a pre-v5 artifact still parses.

    A report this code writes always looked, so it must never carry the value
    that means "nobody looked" -- otherwise UNMEASURED becomes indistinguishable
    from a real measurement and the sentinel is worthless.
    """
    clean = _report(baseline=_baseline(consumers=[WHISPER_CONSUMER]),
                    snapshot=_snapshot(consumers=[WHISPER_CONSUMER,
                                                  ARM_CONSUMER]))
    polluted = _report(baseline=_baseline(consumers=[WHISPER_CONSUMER]),
                       snapshot=_snapshot(consumers=[WHISPER_CONSUMER,
                                                     OLLAMA_CONSUMER]))

    for report in (clean, polluted):
        assert report.vram.pollution != lms_vram.PollutionState.UNMEASURED
        assert report.vram.consumer_inventory_note != ''

    # The sentinel does exist, though -- that is what lets a v4 artifact parse.
    assert lms_vram.PollutionState.UNMEASURED == 'UNMEASURED'


def test_the_report_schema_version_records_the_added_evidence():
    """v5.  A v4 file read by a v5-aware consumer would show a `pollution` it
    never measured, which is precisely the misreading this constant exists to
    prevent."""
    assert lms_healthcheck.REPORT_SCHEMA_VERSION == 5
    assert _report().schema_version == 5


def test_a_polluted_recorded_baseline_produces_no_report_at_all():
    """Same stance as a dead GPU probe and a missing baseline.

    A baseline taken while ollama held 10314 MiB has that memory built into the
    number every footprint is measured against.  Swallowing it would emit a
    report whose arithmetic is void but whose shape is indistinguishable from a
    good one -- so it propagates instead.
    """
    with pytest.raises(lms_vram.PollutedBaselineError,
                       match=str(OLLAMA_CONSUMER.pid)):
        lms_healthcheck.run_healthcheck(
            [_arm()],
            gpu_probe=lambda: _snapshot(),
            probe=_passing_probe,
            baseline=_baseline(consumers=[WHISPER_CONSUMER, OLLAMA_CONSUMER]),
        )


def test_a_vanished_baseline_consumer_reports_pollution_not_a_broken_probe():
    """whisper-writer exits mid-run and the arm takes less than it released, so
    `used` lands BELOW `baseline`.

    `evaluate_budget` raises for that reading, and evaluated first it reached
    the CLI's generic branch as "the GPU probe failed" (exit 4) -- blaming
    nvidia-smi for a probe that worked perfectly on a card that was polluted
    (exit 7). The shrink diagnosis was already computed and never printed.
    """
    baseline = _baseline(used_mib=7362, consumers=[WHISPER_CONSUMER])
    # whisper gone; the arm holds 3000 MiB, so used < baseline.
    after = _snapshot(
        used_mib=6312, free_mib=MEASURED_TOTAL_MIB - 6312,
        consumers=[lms_vram.GpuConsumer(
            pid=41001, process_name='python', used_mib=3000,
        )],
    )

    with pytest.raises(lms_vram.PollutedMeasurementError) as excinfo:
        lms_healthcheck.run_healthcheck(
            [_arm()], gpu_probe=lambda: after,
            probe=_passing_probe, baseline=baseline,
        )

    message = str(excinfo.value)
    # The operator must be told the card moved, not that the tool is broken.
    assert str(WHISPER_CONSUMER.pid) in message
    assert 'GPU probe' not in message or 'probe itself is fine' in message
    assert not isinstance(excinfo.value, lms_vram.PollutedBaselineError)
    # Still a VramProbeError, so no pre-existing handler can let it escape.
    assert isinstance(excinfo.value, lms_vram.VramProbeError)


def test_a_baseline_recorded_with_a_declared_coresident_arm_still_reports():
    """`lms_ctl start --no-exclusive` legitimately records another arm's
    container in the inventory.  Re-applying the STRICT rule at report time
    would refuse to report on a run that was never polluted -- and the flag
    would be unusable end to end even once `start` accepted it."""
    arm_container = lms_vram.GpuConsumer(
        pid=41001, process_name='python', used_mib=9000,
    )

    report = lms_healthcheck.run_healthcheck(
        [_arm()],
        gpu_probe=lambda: _snapshot(),
        probe=_passing_probe,
        baseline=_baseline(
            consumers=[WHISPER_CONSUMER, arm_container],
            coresident_arms=['phi-4-14b'],
        ),
    )

    # Recorded, not silently dropped: the inventory is the audit trail that
    # pays for the relaxation.
    assert arm_container in report.vram.baseline_consumers


def test_a_declared_coresident_arm_does_not_excuse_ollama_at_report_time():
    with pytest.raises(lms_vram.PollutedBaselineError,
                       match=str(OLLAMA_CONSUMER.pid)):
        lms_healthcheck.run_healthcheck(
            [_arm()],
            gpu_probe=lambda: _snapshot(),
            probe=_passing_probe,
            baseline=_baseline(
                consumers=[WHISPER_CONSUMER, OLLAMA_CONSUMER],
                coresident_arms=['phi-4-14b'],
            ),
        )


# ---------------------------------------------------------------------------
# merge_reports — the slate is measured one arm at a time
# ---------------------------------------------------------------------------


def _single(arm, snapshot=None, probe=_passing_probe, baseline=None):
    return _report(arms=[arm], probe=probe, snapshot=snapshot, baseline=baseline)


def test_merging_per_arm_runs_yields_one_row_per_arm():
    """This card cannot hold the slate at once and the PRD's funnel does not
    ask it to, so the committed artifact is necessarily assembled."""
    first = _single(_arm())
    second = _single(_arm(arm_id='phi-4-14b', served_model_name='phi-4-14b',
                          port=8412))

    merged = lms_healthcheck.merge_reports([first, second])

    assert [row.arm_id for row in merged.arms] == ['qwen3.5-9b', 'phi-4-14b']
    assert merged.overall == 'PASS'
    assert merged.schema_version == lms_healthcheck.REPORT_SCHEMA_VERSION


def test_merging_refuses_two_runs_of_the_same_arm():
    """The cheapest way to fake this artifact is to re-run a failing arm and
    append the good result; silently keeping one of the two would allow it."""
    with pytest.raises(lms_healthcheck.ReportMergeError, match='qwen3.5-9b'):
        lms_healthcheck.merge_reports([_single(_arm()), _single(_arm())])


def test_merging_refuses_reports_from_different_gpus():
    other_card = lms_vram.GpuSnapshot(
        identity=lms_vram.GpuIdentity(name='NVIDIA A100', driver_version='999'),
        reading=_snapshot().reading,
        consumers=_snapshot().consumers,
    )
    second = _single(_arm(arm_id='phi-4-14b', served_model_name='phi-4-14b',
                          port=8412), snapshot=other_card)

    with pytest.raises(lms_healthcheck.ReportMergeError, match='different GPUs'):
        lms_healthcheck.merge_reports([_single(_arm()), second])


def test_merging_nothing_is_an_error_not_an_empty_pass():
    with pytest.raises(lms_healthcheck.ReportMergeError):
        lms_healthcheck.merge_reports([])


def test_the_merged_block_is_the_binding_measurement():
    """The arm that came closest to its budget is the one worth reporting."""
    small = _single(_arm())
    big = _single(
        _arm(arm_id='phi-4-14b', served_model_name='phi-4-14b', port=8412),
        snapshot=_snapshot(used_mib=20000, free_mib=4576),
    )

    merged = lms_healthcheck.merge_reports([small, big])

    assert merged.vram.arm_footprint_mib == 20000 - BASELINE_USED_MIB
    assert merged.vram.arm_footprint_mib > small.vram.arm_footprint_mib


def test_a_failing_budget_survives_the_merge():
    """Keeping a passing block while an input failed would put `overall` out of
    reach of its own evidence -- a green light no one would re-check."""
    ok = _single(_arm())
    over = _single(
        _arm(arm_id='phi-4-14b', served_model_name='phi-4-14b', port=8412),
        snapshot=_over_budget_snapshot(),
    )

    merged = lms_healthcheck.merge_reports([ok, over])

    assert merged.vram.verdict == 'FAIL'
    assert merged.overall == 'FAIL'


def test_a_failing_arm_row_survives_the_merge():
    merged = lms_healthcheck.merge_reports([
        _single(_arm()),
        _single(_arm(arm_id='phi-4-14b', served_model_name='phi-4-14b', port=8412),
                probe=_failing_probe),
    ])

    assert merged.overall == 'FAIL'
    assert lms_healthcheck.exit_code_for(merged) == lms_healthcheck.EXIT_ARM_FAILED


# --- pollution must survive the merge (task 3755) --------------------------
#
# The slate is measured one arm at a time over ~39 minutes, so the ONLY place
# an operator meets the pollution evidence is the merged artifact.  Binding
# selection was `failing[0] if any FAIL else max(arm_footprint_mib)` and
# pollution was not part of it, so a POLLUTED run that lost the footprint
# contest was silently laundered into a clean slate.


#: 8312 used against the 3312 MiB baseline: a 5000 MiB footprint.
SMALL_USED_MIB = 8312
#: 12312 used: a 9000 MiB footprint, still well inside the 20811 MiB budget,
#: so BOTH sizes below are budget PASSes and only the footprint differs.
LARGE_USED_MIB = 12312


def _clean_single(arm, used_mib=MEASURED_USED_MIB):
    """One arm's run on a card nobody else touched."""
    return _single(
        arm,
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(
            used_mib=used_mib, free_mib=MEASURED_TOTAL_MIB - used_mib,
            consumers=[WHISPER_CONSUMER, ARM_CONSUMER],
        ),
    )


def _polluted_single(arm, used_mib=MEASURED_USED_MIB):
    """One arm's run that ollama gatecrashed: the numbers exist, and mean
    nothing.  The measured 2026-08-06 scenario, one arm at a time."""
    return _single(
        arm,
        baseline=_baseline(consumers=[WHISPER_CONSUMER]),
        snapshot=_snapshot(
            used_mib=used_mib, free_mib=MEASURED_TOTAL_MIB - used_mib,
            consumers=[WHISPER_CONSUMER, ARM_CONSUMER, OLLAMA_CONSUMER],
        ),
    )


def _other_arm(arm_id='phi-4-14b', port=8412):
    return _arm(arm_id=arm_id, served_model_name=arm_id, port=port)


def _assert_names_the_intruder(text):
    assert str(OLLAMA_CONSUMER.pid) in text
    assert OLLAMA_CONSUMER.process_name in text
    assert str(OLLAMA_CONSUMER.used_mib) in text


@pytest.mark.parametrize(
    'build', [_clean_single, _polluted_single], ids=['clean', 'polluted'],
)
def test_merging_one_report_carries_its_pollution_through_unchanged(build):
    """THE CLASS-KILLING INVARIANT: a one-report merge is a no-op.

    Any binding rule that can drop, invent or rewrite pollution shows up here
    first, whatever the arithmetic of the multi-report cases happens to be.
    """
    single = build(_arm())

    merged = lms_healthcheck.merge_reports([single])

    assert merged.vram.pollution == single.vram.pollution
    assert merged.vram.pollution_reason == single.vram.pollution_reason
    assert (
        lms_healthcheck.exit_code_for(merged)
        == lms_healthcheck.exit_code_for(single)
    )


def test_a_polluted_run_survives_a_merge_it_loses_on_footprint():
    """The flattering direction, and the one the old rule dropped.

    A polluted arm that took LESS than a clean one loses the largest-footprint
    contest, so its evidence left the artifact entirely: exit 0, no banner, and
    a slate that reads as measured on an empty card.
    """
    polluted = _polluted_single(_arm(), used_mib=SMALL_USED_MIB)
    clean = _clean_single(_other_arm(), used_mib=LARGE_USED_MIB)
    assert polluted.vram.arm_footprint_mib < clean.vram.arm_footprint_mib

    merged = lms_healthcheck.merge_reports([polluted, clean])

    assert merged.vram.pollution == lms_vram.PollutionState.POLLUTED
    _assert_names_the_intruder(merged.vram.pollution_reason)
    assert 'qwen3.5-9b' in merged.vram.pollution_reason
    assert (
        lms_healthcheck.exit_code_for(merged)
        == lms_healthcheck.EXIT_VRAM_POLLUTED
    )
    assert 'POLLUTED' in lms_healthcheck.render_table(merged)
    # `overall` is what the ARMS earned, exactly as in the single-report case.
    assert merged.overall == 'PASS'


def test_a_polluted_run_survives_a_merge_it_wins_on_footprint():
    """The growth direction passes today only by luck -- the polluted arm
    happens to win the footprint contest.  Pinned so it cannot regress."""
    polluted = _polluted_single(_arm(), used_mib=LARGE_USED_MIB)
    clean = _clean_single(_other_arm(), used_mib=SMALL_USED_MIB)
    assert polluted.vram.arm_footprint_mib > clean.vram.arm_footprint_mib

    merged = lms_healthcheck.merge_reports([polluted, clean])

    assert merged.vram.pollution == lms_vram.PollutionState.POLLUTED
    _assert_names_the_intruder(merged.vram.pollution_reason)
    assert 'qwen3.5-9b' in merged.vram.pollution_reason
    assert (
        lms_healthcheck.exit_code_for(merged)
        == lms_healthcheck.EXIT_VRAM_POLLUTED
    )
    assert 'POLLUTED' in lms_healthcheck.render_table(merged)


def test_a_failing_budget_still_outranks_pollution_for_the_binding_block():
    """Binding precedence is failing > polluted > largest footprint.

    Failing stays FIRST because `overall` is computed from the surviving
    block: promoting a POLLUTED-but-PASS block over a budget FAIL would flip a
    failing slate green.  Pollution still propagates alongside it.
    """
    over = _single(_arm(), snapshot=_over_budget_snapshot())
    polluted = _polluted_single(_other_arm(), used_mib=SMALL_USED_MIB)

    merged = lms_healthcheck.merge_reports([over, polluted])

    assert merged.vram.verdict == 'FAIL'
    assert merged.overall == 'FAIL'
    assert merged.vram.pollution == lms_vram.PollutionState.POLLUTED
    assert 'phi-4-14b' in merged.vram.pollution_reason
    assert (
        lms_healthcheck.exit_code_for(merged)
        == lms_healthcheck.EXIT_VRAM_POLLUTED
    )


def test_with_nothing_failing_the_polluted_block_binds_over_a_bigger_footprint():
    """The surviving block is ONE input's real measurement, so which input it
    is decides whose consumer lists an operator gets to see.  On a contended
    card the largest footprint is the least meaningful number in the file."""
    polluted = _polluted_single(_arm(), used_mib=SMALL_USED_MIB)
    clean = _clean_single(_other_arm(), used_mib=LARGE_USED_MIB)

    merged = lms_healthcheck.merge_reports([clean, polluted])

    assert merged.vram.arm_footprint_mib == polluted.vram.arm_footprint_mib
    assert OLLAMA_CONSUMER in merged.vram.probe_consumers


def test_two_polluted_runs_union_into_one_reason_naming_both_arms():
    """The merged block belongs to ONE arm, so the reason is the only place the
    others' pollution can be recorded at all."""
    first = _polluted_single(_arm())
    second = _polluted_single(_other_arm(), used_mib=SMALL_USED_MIB)

    merged = lms_healthcheck.merge_reports([first, second])

    assert merged.vram.pollution == lms_vram.PollutionState.POLLUTED
    assert 'qwen3.5-9b' in merged.vram.pollution_reason
    assert 'phi-4-14b' in merged.vram.pollution_reason
    assert (
        lms_healthcheck.exit_code_for(merged)
        == lms_healthcheck.EXIT_VRAM_POLLUTED
    )


def _never_looked(report):
    """A pre-v5 report as `--merge` reads it back off disk.

    The five consumer fields are absent from the payload, so they default --
    and `pollution` defaults to UNMEASURED precisely so this state is legible.
    This producer never emits it; reading an old file is the only way in.
    """
    return report.model_copy(update={
        'vram': report.vram.model_copy(update={
            'baseline_consumers': [],
            'probe_consumers': [],
            'consumer_inventory_note': '',
            'pollution': lms_vram.PollutionState.UNMEASURED,
            'pollution_reason': '',
        }),
    })


def test_merging_refuses_a_report_that_never_looked_at_the_card():
    """UNMEASURED is not a measurement, so it cannot be unioned into one.

    A v5-shaped slate assembled partly from runs that never looked genuinely IS
    an assembly defect, which is what EXIT_MERGE_ERROR already means -- and
    refusing avoids inventing an eighth exit code for a state the producer
    cannot emit.  POLLUTED propagates instead, because it IS a measurement.
    """
    blind = _never_looked(_single(_arm()))

    with pytest.raises(lms_healthcheck.ReportMergeError, match='qwen3.5-9b'):
        lms_healthcheck.merge_reports([blind, _single(_other_arm())])


def test_cli_merge_refuses_a_pre_v5_slate_and_writes_nothing(cli_env, tmp_path):
    """The one path by which UNMEASURED reaches `merge_reports`: files written
    before this producer recorded who else held the card."""
    parts = []
    for arm_id in lms_healthcheck.load_arms().arm_ids():
        path = tmp_path / f'{arm_id}.json'
        assert lms_healthcheck.main(['--arm', arm_id, '--output', str(path)]) == 0
        payload = json.loads(path.read_text())
        payload['schema_version'] = 4
        for key in ('baseline_consumers', 'probe_consumers',
                    'consumer_inventory_note', 'pollution', 'pollution_reason'):
            del payload['vram'][key]
        path.write_text(json.dumps(payload))
        parts.append(str(path))

    out = tmp_path / 'merged.json'
    code = lms_healthcheck.main(['--merge', *parts, '--output', str(out)])

    assert code == lms_healthcheck.EXIT_MERGE_ERROR
    assert not out.exists()


def test_the_report_carries_the_delivered_check_marker():
    """JSON carries no comments, so the marker must live in a real field."""
    assert _report().prd_marker == 'PRD-MARKER:local-memory-models-eval serving'


def test_cli_merge_writes_the_combined_artifact(cli_env, tmp_path):
    """The whole slate merges. Every arm the manifest declares, because the CLI
    now enforces coverage — see the partial-set test below."""
    expected = lms_healthcheck.load_arms().arm_ids()
    parts = []
    for arm_id in expected:
        path = tmp_path / f'{arm_id}.json'
        assert lms_healthcheck.main(['--arm', arm_id, '--output', str(path)]) == 0
        parts.append(str(path))

    out = tmp_path / 'health-report.json'
    code = lms_healthcheck.main(['--merge', *parts, '--output', str(out)])

    assert code == 0
    written = json.loads(out.read_text())
    assert {row['arm_id'] for row in written['arms']} == set(expected)
    assert written['prd_marker'] == 'PRD-MARKER:local-memory-models-eval serving'


def test_cli_merge_refuses_a_partial_slate_and_writes_nothing(cli_env, tmp_path):
    """An arm that never became ready leaves NO per-arm file, so a partial merge
    is what a failed slate run actually looks like — and it must not produce an
    artifact that reads as complete. Measured 2026-08-06: a 7-row artifact came
    out of an 8-arm manifest and nothing objected."""
    path = tmp_path / 'one.json'
    assert lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(path)]) == 0
    out = tmp_path / 'merged.json'

    code = lms_healthcheck.main(['--merge', str(path), '--output', str(out)])

    assert code == lms_healthcheck.EXIT_MERGE_ERROR
    assert not out.exists()


def test_cli_merge_reports_a_refusal_loudly_and_writes_nothing(cli_env, tmp_path):
    path = tmp_path / 'one.json'
    assert lms_healthcheck.main(['--arm', 'qwen3.5-9b', '--output', str(path)]) == 0
    out = tmp_path / 'merged.json'

    code = lms_healthcheck.main(['--merge', str(path), str(path), '--output', str(out)])

    assert code == lms_healthcheck.EXIT_MERGE_ERROR
    assert not out.exists()


# ---------------------------------------------------------------------------
# The extraction floor, and the reasoning-mode contract (esc-3713-10).
#
# Everything below exists because the two checks above it were each passing
# something they should not: schema validity passed an empty extraction, and
# the truncation branch was unreachable on the stack that needed it most.
# ---------------------------------------------------------------------------


def _reasoning_completion(
    content, reasoning: str, *, key: str = 'reasoning_content',
    finish_reason: str = 'stop',
) -> dict:
    """A response carrying a thought channel. *content* may be None.

    `key` selects the stack's spelling: llama.cpp says `reasoning_content`,
    vLLM's parsers say `reasoning`.
    """
    return {
        'id': 'cmpl-1',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content, key: reasoning},
                'finish_reason': finish_reason,
            }
        ],
    }


def test_a_schema_valid_but_empty_extraction_is_not_a_pass():
    """The measured qwen3.5-9b defect: 18 tokens, perfectly typed, no entities.

    `ProbeExtraction` accepts it because an empty list is a valid
    `list[ProbeEntity]`. If this test ever goes green as a PASS, the health
    check has gone back to certifying endpoints that answer with nothing.
    """
    result = lms_healthcheck.verify_llm_response(
        _arm(), _completion(_empty_extraction_json())
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.EXTRACTION_MISSED_ENTITIES
    assert 'none' in result.detail


def test_a_below_floor_extraction_names_exactly_what_it_missed():
    """A reason code without the missing names costs the operator the diagnosis
    again — which is the standard every other code in this module is held to."""
    partial = json.dumps(
        {
            'entities': [
                {'name': 'Graphiti', 'entity_type': 'System', 'attributes': []},
            ],
            'summary': 'Partial extraction.',
        }
    )

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(partial))

    assert result.reason == lms_healthcheck.Reason.EXTRACTION_MISSED_ENTITIES
    assert 'Leo' in result.detail
    assert 'FalkorDB' in result.detail


def test_entity_names_carrying_extra_words_still_count_toward_the_floor():
    """The floor asks whether the arm extracted, not how it spells.

    Both measured models qualified their names differently run to run
    ('Dark Factory' vs 'Dark Factory (software factory)'), and failing an arm
    over that would be the floor manufacturing a defect.
    """
    qualified = json.dumps(
        {
            'entities': [
                {'name': 'Leo (operator)', 'entity_type': 'Person', 'attributes': []},
                {
                    'name': 'Dark Factory (software factory)',
                    'entity_type': 'Organization',
                    'attributes': [],
                },
                {'name': 'FalkorDB database', 'entity_type': 'Database', 'attributes': []},
            ],
            'summary': 'Qualified names.',
        }
    )

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(qualified))

    assert result.verdict == 'PASS'


def test_a_bare_substring_does_not_count_as_finding_a_known_entity():
    """Containment is one-directional on purpose. 'Factory' is not
    'Dark Factory', and a floor loose in both directions would let a
    one-character entity name satisfy every known entity at once."""
    loose = json.dumps(
        {
            'entities': [
                {'name': 'Factory', 'entity_type': 'Thing', 'attributes': []},
                {'name': 'L', 'entity_type': 'Thing', 'attributes': []},
                {'name': 'DB', 'entity_type': 'Thing', 'attributes': []},
            ],
            'summary': 'Loose names.',
        }
    )

    result = lms_healthcheck.verify_llm_response(_arm(), _completion(loose))

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.EXTRACTION_MISSED_ENTITIES


def test_a_null_content_cut_off_by_the_cap_is_truncation_not_malformed():
    """Measured on qwen3.5-9b with --reasoning-parser, 2026-08-06.

    vLLM returns `content: null` when a reasoning parser consumes the whole
    generation; llama.cpp returns `''`. Only the empty string reached the
    truncation branch, so COMPLETION_TRUNCATED — the code written for exactly
    this failure — was unreachable on vLLM and it reported `malformed_response`
    instead, blaming the harness's own cap on a broken response body.
    """
    result = lms_healthcheck.verify_llm_response(
        _arm(),
        _reasoning_completion(
            None, 'thinking ' * 200, key='reasoning', finish_reason='length'
        ),
    )

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.COMPLETION_TRUNCATED
    assert result.reason != lms_healthcheck.Reason.MALFORMED_RESPONSE


def test_a_null_content_with_reasoning_and_a_clean_stop_is_empty_not_malformed():
    """An arm that thought and then chose to say nothing answered badly; it was
    not cut off, and it did not return an unreadable body. Three distinct
    failures, three distinct codes."""
    result = lms_healthcheck.verify_llm_response(
        _arm(), _reasoning_completion(None, 'I decline.', key='reasoning')
    )

    assert result.reason == lms_healthcheck.Reason.EMPTY_COMPLETION


def test_a_null_content_with_no_reasoning_at_all_is_still_malformed():
    """The malformed path must survive: a body with no content and no thought
    channel genuinely is unreadable, and softening that would hide real
    transport-level breakage behind a friendlier code."""
    result = lms_healthcheck.verify_llm_response(_arm(), _completion(None))

    assert result.reason == lms_healthcheck.Reason.MALFORMED_RESPONSE


def test_reasoning_is_read_under_both_stack_spellings():
    """A helper that knew only one spelling would report 'no reasoning' for half
    the slate — and silently route those rows to the wrong reason code."""
    for key in ('reasoning_content', 'reasoning'):
        body = _reasoning_completion(None, 'thought', key=key, finish_reason='length')
        result = lms_healthcheck.verify_llm_response(_arm(), body)
        assert result.reason == lms_healthcheck.Reason.COMPLETION_TRUNCATED, key


def test_a_reasoning_off_arm_is_sent_the_explicit_suppression_kwarg():
    """Sending nothing means 'whatever this server defaults to', and the two
    stocks disagree: llama.cpp forces gemma-4's thinking ON against its own
    template default, vLLM leaves Qwen3.5's ON and then grammar-blocks it. The
    probe has to state the mode rather than inherit one."""
    body = lms_healthcheck.build_llm_probe_request(_arm(reasoning='off'))

    assert body['chat_template_kwargs'] == {'enable_thinking': False}


def test_a_reasoning_on_arm_is_sent_no_thinking_kwarg():
    """`enable_thinking: true` is a measured NO-OP on Qwen3.5 — its template
    tests only for the literal `false`. Omission is the only spelling of 'on'
    that means the same thing on both stacks."""
    body = lms_healthcheck.build_llm_probe_request(
        _arm(reasoning='on', reasoning_parser='qwen3')
    )

    assert 'chat_template_kwargs' not in body


def test_the_probe_cap_is_the_products_own_output_budget():
    """PROBE_MAX_TOKENS is DERIVED from fused-memory's `llm.max_tokens`, not
    chosen here. That is what stops it being a knob: it cannot be raised to
    turn a row green without moving the production budget it mirrors.

    Read from the config file rather than hardcoded, so the two cannot drift
    apart silently — which is the entire point of sourcing it.
    """
    config_path = (
        Path(__file__).resolve().parents[2] / 'fused-memory' / 'config' / 'config.yaml'
    )
    config = yaml.safe_load(config_path.read_text())

    assert config['llm']['max_tokens'] == lms_healthcheck.PROBE_MAX_TOKENS


def test_merge_refuses_a_set_that_does_not_cover_the_manifest():
    """Absence is the one failure a report cannot describe.

    An arm that never becomes ready produces NO per-arm report, so it goes
    missing from the merge rather than red in it. Measured 2026-08-06: a 7-row
    artifact came out of an 8-arm manifest because mistral-small-3.2-24b never
    started, and nothing in merge_reports noticed.
    """
    report = _report([_arm(arm_id='arm-a', served_model_name='arm-a')])

    with pytest.raises(lms_healthcheck.ReportMergeError) as excinfo:
        lms_healthcheck.merge_reports([report], expected_arm_ids=['arm-a', 'arm-b'])

    assert 'arm-b' in str(excinfo.value)
    assert 'NARROWER' in str(excinfo.value)


def test_merge_accepts_a_set_that_covers_the_manifest():
    """The check must not fire on a complete slate."""
    merged = lms_healthcheck.merge_reports(
        [
            _report([_arm(arm_id='arm-a', served_model_name='arm-a')]),
            _report([_arm(arm_id='arm-b', served_model_name='arm-b', port=8411)]),
        ],
        expected_arm_ids=['arm-a', 'arm-b'],
    )

    assert {row.arm_id for row in merged.arms} == {'arm-a', 'arm-b'}


def test_merge_without_an_expected_set_still_merges():
    """`expected_arm_ids=None` keeps the old behaviour for callers with no
    manifest in hand — the CLI passes one, library callers need not."""
    merged = lms_healthcheck.merge_reports(
        [_report([_arm(arm_id='arm-a', served_model_name='arm-a')])]
    )

    assert [row.arm_id for row in merged.arms] == ['arm-a']


def _extraction(entities) -> str:
    return json.dumps({'entities': entities, 'summary': 'A summary.'})


def test_an_entity_captured_as_an_attribute_value_counts_toward_the_floor():
    """Measured on phi-4-14b, 2026-08-06: it returned `Dark Factory` and
    `Graphiti` as top-level entities and `FalkorDB` as an attribute value
    (`{"name": "Backend", "value": "FalkorDB"}`). That is a nested, relational
    representation, not a refusal — and scoring it 2/4 made the floor adjudicate
    representation QUALITY, which is theta's graph-sameness metric, not
    alpha's question."""
    body = _completion(_extraction([
        {'name': 'Dark Factory', 'entity_type': 'Organization', 'attributes': []},
        {
            'name': 'Graphiti',
            'entity_type': 'System',
            'attributes': [{'name': 'Backend', 'value': 'FalkorDB'}],
        },
    ]))

    result = lms_healthcheck.verify_llm_response(_arm(), body)

    assert result.verdict == 'PASS'


def test_the_summary_is_not_scanned_for_known_entities():
    """Counting prose would let an arm that returned ZERO entities pass on the
    strength of a well-written sentence — precisely the refusal this floor
    exists to catch."""
    body = _completion(json.dumps({
        'entities': [],
        'summary': 'Leo runs Dark Factory, which uses Graphiti backed by FalkorDB.',
    }))

    result = lms_healthcheck.verify_llm_response(_arm(), body)

    assert result.verdict == 'FAIL'
    assert result.reason == lms_healthcheck.Reason.EXTRACTION_MISSED_ENTITIES


def test_a_passing_row_reports_its_top_level_count_separately():
    """Reported, NON-gating. The gap between the floor and this number is the
    representation signal alpha declines to judge and eta needs to see."""
    result = lms_healthcheck.verify_llm_response(
        _arm(),
        _completion(_extraction([
            {'name': 'Leo', 'entity_type': 'Person', 'attributes': []},
            {'name': 'Dark Factory', 'entity_type': 'Org', 'attributes': []},
            {
                'name': 'Graphiti',
                'entity_type': 'System',
                'attributes': [{'name': 'Backend', 'value': 'FalkorDB'}],
            },
        ])),
    )

    assert result.verdict == 'PASS'
    # four captured, but only three promoted to entities
    assert result.top_level_entities_named == 3


def test_top_level_count_is_none_when_the_question_does_not_apply():
    """An unparseable response yields no count rather than a misleading zero."""
    result = lms_healthcheck.verify_llm_response(_arm(), _completion('not json'))

    assert result.reason == lms_healthcheck.Reason.NOT_JSON
    assert result.top_level_entities_named is None


# ---------------------------------------------------------------------------
# Schema v5 and the not-comparable caveat (task 3781)
#
# Two things moved at once, and the version exists to stop a consumer papering
# over either.  The ROW SHAPE gained `first_probe_ms`, `repeat_latencies_ms`
# and `measured_cached_prompt_tokens` -- but more importantly `latency_ms`
# CHANGED MEANING: up to v4 it was the first request after the arm reached
# ready (engine-cold AND prefix-cold), and from v5 it is the engine-warm,
# prefix-COLD measured run.  A consumer that read a v4 `latency_ms` as the same
# quantity would be wrong, which is precisely what a version bump prevents.
#
# The CAVEAT is the load-bearing other half.  A corrected number without the
# sentence saying what it is not re-creates the same false comparability the
# corrected number was supposed to retire -- and eta (3720) and theta (3721)
# read the JSON artifact, not the README, so the sentence has to live in a real
# FIELD.  `prd_marker` already exists for exactly this reason.
# ---------------------------------------------------------------------------


def test_the_report_schema_version_is_five():
    """The shape moved AND `latency_ms` changed meaning between v4 and v5."""
    assert lms_healthcheck.REPORT_SCHEMA_VERSION == 5


def test_the_report_carries_the_not_comparable_caveat_in_a_field():
    """A contract test on an artifact FIELD, not a pin on prose: it checks the
    three load-bearing claims are stated, not how they are worded."""
    caveat = lms_healthcheck.LATENCY_CAVEAT

    assert isinstance(caveat, str)
    assert caveat.strip()
    assert _report().latency_caveat == caveat

    lowered = caveat.lower()
    # (1) single-sample, (2) not the p95-under-load envelope metric zeta owns,
    # (3) not a cross-arm ranking.
    assert 'single-sample' in lowered
    assert 'p95' in lowered
    assert 'rank' in lowered


def test_the_written_artifact_carries_the_caveat_and_both_latencies(cli_env, tmp_path):
    """JSON carries no comments, which is exactly why this has to be a field --
    the same reason `prd_marker` is one."""
    out_path = tmp_path / 'health-report.json'

    assert lms_healthcheck.main(['--all', '--output', str(out_path)]) == 0
    written = json.loads(out_path.read_text())

    assert written['latency_caveat'] == lms_healthcheck.LATENCY_CAVEAT
    assert written['arms']
    for row in written['arms']:
        assert 'first_probe_ms' in row
        assert 'latency_ms' in row


def test_the_table_states_the_caveat_and_shows_both_numbers():
    """An operator reading ONE ms column would re-create the same false
    comparability the JSON just fixed, so the table splits it too."""
    def probe(arm, *, warmup: bool = False):
        return lms_healthcheck.ProbeResult(
            verdict='PASS',
            reason=lms_healthcheck.Reason.OK,
            detail='ok',
            latency_ms=4249.7 if warmup else 359.1,
        )

    table = lms_healthcheck.render_table(_report(probe=probe))

    assert 'COLD-MS' in table
    row_line = next(
        line for line in table.splitlines() if line.startswith('qwen3.5-9b')
    )
    assert '4250' in row_line
    assert '359' in row_line
    assert lms_healthcheck.LATENCY_CAVEAT in table


def test_merging_preserves_the_caveat_and_both_latencies():
    """The caveat travels from the BINDING input -- the same report the
    surviving vram block comes from -- so the merged artifact cannot state a
    caveat no input ever made."""
    small = _single(_arm())
    big = _single(
        _arm(arm_id='phi-4-14b', served_model_name='phi-4-14b', port=8412),
        snapshot=_snapshot(used_mib=20000, free_mib=4576),
    )

    merged = lms_healthcheck.merge_reports([small, big])

    assert merged.latency_caveat == lms_healthcheck.LATENCY_CAVEAT
    assert all(row.first_probe_ms > 0 for row in merged.arms)

    stamped = big.model_copy(update={'latency_caveat': 'from the binding run'})
    assert lms_healthcheck.merge_reports(
        [small, stamped]
    ).latency_caveat == 'from the binding run'


def test_a_v4_artifact_no_longer_reads_as_current():
    """`latency_ms` changed meaning at v5, so a stale v4 part must be REFUSED
    rather than reinterpreted alongside v5 rows."""
    stale = _single(_arm()).model_copy(update={'schema_version': 4})
    current = _single(
        _arm(arm_id='phi-4-14b', served_model_name='phi-4-14b', port=8412)
    )

    with pytest.raises(
        lms_healthcheck.ReportMergeError, match='mixed schema versions'
    ):
        lms_healthcheck.merge_reports([stale, current])
