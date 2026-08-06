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

import pytest

import lms_ctl
import lms_healthcheck
import lms_manifest
import lms_vram

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


def _completion(content: str) -> dict:
    """An OpenAI chat-completions response body carrying *content*."""
    return {
        'id': 'cmpl-1',
        'object': 'chat.completion',
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            }
        ],
    }


def _models_payload(*names: str) -> dict:
    return {'object': 'list', 'data': [{'id': n, 'object': 'model'} for n in names]}


def _valid_probe_json() -> str:
    return json.dumps(
        {
            'entities': [
                {
                    'name': 'Graphiti',
                    'entity_type': 'System',
                    'attributes': [{'name': 'role', 'value': 'temporal knowledge graph'}],
                }
            ],
            'summary': 'One entity extracted from the probe text.',
        }
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


def _snapshot(
    used_mib=MEASURED_USED_MIB,
    total_mib=MEASURED_TOTAL_MIB,
    free_mib=MEASURED_FREE_MIB,
) -> lms_vram.GpuSnapshot:
    """The measured host reading, as one injected GPU snapshot."""
    return lms_vram.GpuSnapshot(
        identity=lms_vram.GpuIdentity(
            name='NVIDIA GeForce RTX 3090', driver_version='580.159.04',
        ),
        reading=lms_vram.GpuReading(
            total_mib=total_mib, used_mib=used_mib, free_mib=free_mib,
        ),
    )


def _over_budget_snapshot() -> lms_vram.GpuSnapshot:
    """21000 MiB used -- past PRD D10's 19.5 GiB nominal ceiling."""
    return _snapshot(used_mib=21000, free_mib=MEASURED_TOTAL_MIB - 21000)


def _passing_probe(arm):
    return lms_healthcheck.ProbeResult(
        verdict='PASS', reason=lms_healthcheck.Reason.OK, detail='ok', latency_ms=12.5,
    )


def _failing_probe(arm):
    return lms_healthcheck.ProbeResult(
        verdict='FAIL',
        reason=lms_healthcheck.Reason.IDENTITY_MISMATCH,
        detail='port 8410 serves something else entirely',
        latency_ms=3.0,
    )


def _report(arms=None, probe=_passing_probe, snapshot=None):
    return lms_healthcheck.run_healthcheck(
        arms if arms is not None else [_arm()],
        gpu_probe=lambda: snapshot if snapshot is not None else _snapshot(),
        probe=probe,
    )


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
    def probe(arm):
        return _failing_probe(arm) if arm.arm_id == 'qwen3.5-9b' else _passing_probe(arm)

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
        [dead, alive], gpu_probe=lambda: _snapshot(),
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


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def test_the_table_takes_only_the_report_so_text_and_json_cannot_disagree():
    """A structural guarantee, not a hopeful one: given no arms, no HTTP and
    no GPU, `render_table` has nothing left to recompute."""
    signature = inspect.signature(lms_healthcheck.render_table)

    assert list(signature.parameters) == ['report']


def test_the_table_shows_every_row_with_its_verdict_and_reason():
    def probe(arm):
        return _failing_probe(arm) if arm.arm_id == 'qwen3.5-9b' else _passing_probe(arm)

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch):
    """Patch the CLI's three seams: the GPU, the arm prober, the unit manager."""
    calls = {'probed': []}

    def probe(arm):
        calls['probed'].append(arm.arm_id)
        return _passing_probe(arm)

    monkeypatch.setattr(lms_vram, 'probe_gpu_snapshot', lambda *a, **k: _snapshot())
    monkeypatch.setattr(lms_healthcheck, 'probe_arm', probe)
    monkeypatch.setattr(lms_ctl, 'active_arms', lambda: set())
    return calls


def test_cli_all_covers_every_arm_in_the_committed_manifest(cli_env, capsys):
    expected = lms_manifest.load_arms().arm_ids()

    code = lms_healthcheck.main(['--all'])

    assert code == 0
    assert cli_env['probed'] == expected
    assert len(expected) == 8


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
        'total_mib', 'used_mib', 'free_mib', 'nominal_ceiling_gib',
        'operating_budget_gib', 'headroom_gib', 'verdict',
    ):
        assert key in written['vram']
    for key in ('arm_id', 'axis', 'stack', 'endpoint', 'served_model_name',
                'verdict', 'reason', 'latency_ms'):
        assert key in written['arms'][0]


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


def test_every_cli_exit_code_is_distinct():
    codes = [
        lms_healthcheck.EXIT_OK,
        lms_healthcheck.EXIT_ARM_FAILED,
        lms_healthcheck.EXIT_MANIFEST_ERROR,
        lms_healthcheck.EXIT_VRAM_FAILED,
        lms_healthcheck.EXIT_PROBE_ERROR,
        lms_healthcheck.EXIT_NO_ACTIVE_ARMS,
    ]

    assert len(set(codes)) == len(codes)
    assert lms_healthcheck.EXIT_OK == 0
