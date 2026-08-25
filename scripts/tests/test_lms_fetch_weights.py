"""Tests for lms_fetch_weights (task 3713, LME-alpha).

PRD-MARKER:local-memory-models-eval serving

PRD hazard 5 -- "long runs, weight downloads included, go in transient
`systemd --user` units, never bare background shells" -- is enforced here as a
CHECKED PROPERTY of the built argv rather than a convention in a docstring.

Two measured facts drive the rest:

  * `systemd-run --user` propagates NONE of the caller's environment
    (orchestrator/src/orchestrator/proc_supervision.py:91-92).  So the token
    and the working directory must be passed explicitly, or the unit runs
    unauthenticated from $HOME.
  * On this host `HF_DOWNLOAD_TOKEN` is set while `HF_TOKEN` and
    `HUGGING_FACE_HUB_TOKEN` are NOT.  `huggingface_hub` reads the latter
    names, so a naive `hf download` runs ANONYMOUSLY and fails only later, only
    on gated repos -- which the Mistral and Gemma families are.
"""
import lms_fetch_weights
import lms_manifest
import pytest

_BACKGROUND_SHELL_TOKENS = ('nohup', 'setsid', 'disown', '&')


def _arm(**overrides):
    fields = {
        'arm_id': 'qwen3.5-9b',
        'axis': 'llm',
        'reasoning': 'off',
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


def _gguf_arm(**overrides):
    return _arm(**{
        'arm_id': 'moe-stretch',
        'stack': 'llamacpp',
        'image': 'ghcr.io/ggml-org/llama.cpp:server-cuda',
        'model_ref': 'unsloth/Qwen3.6-35B-A3B-GGUF',
        'quant': 'iq4_xs',
        'port': 8413,
        'served_model_name': 'moe-stretch',
        'structured_output_mode': 'json_object',
        'est_vram_gib': 14.0,
        'gguf_file': 'Qwen3.6-35B-A3B-IQ4_XS.gguf',
        **overrides,
    })


TOKEN_ENV = {'HF_DOWNLOAD_TOKEN': 'hf_measured_value'}


def _flag(argv, prefix):
    matches = [a for a in argv if a.startswith(prefix)]
    assert matches, f'{prefix} missing from {argv}'
    return matches[0]


# ---------------------------------------------------------------------------
# PRD hazard 5 — transient unit, never a background shell
# ---------------------------------------------------------------------------


def test_fetch_argv_is_a_transient_systemd_user_unit():
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    assert argv[:2] == ['systemd-run', '--user']
    # --collect so a completed/failed unit does not linger in the user manager.
    assert '--collect' in argv
    assert _flag(argv, '--unit=') == '--unit=lms-fetch-qwen3.5-9b'


def test_fetch_argv_sets_the_working_directory_explicitly():
    """systemd --user otherwise runs from $HOME, so a relative path in the
    payload would resolve somewhere nobody reviewed."""
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    assert _flag(argv, '--working-directory=') == (
        f'--working-directory={lms_fetch_weights.REPO_ROOT}'
    )


@pytest.mark.parametrize('token', _BACKGROUND_SHELL_TOKENS)
def test_fetch_argv_never_backgrounds_through_a_shell(token):
    """PRD hazard 5. A bare background shell is exactly what is forbidden:
    it is unsupervised, unloggable, and dies with the invoking session."""
    argv = lms_fetch_weights.fetch_argv(_gguf_arm(), TOKEN_ENV)

    for element in argv:
        assert token not in element, f'{token!r} appears in {element!r}'


def test_fetch_argv_payload_downloads_the_manifest_model_ref():
    arm = _arm()

    argv = lms_fetch_weights.fetch_argv(arm, TOKEN_ENV)

    assert 'uvx' in argv
    assert '--from' in argv
    assert argv[argv.index('--from') + 1] == 'huggingface_hub'
    assert 'hf' in argv
    assert 'download' in argv
    assert arm.model_ref in argv
    # The payload follows `--`, so systemd-run never reinterprets its flags.
    assert '--' in argv
    assert argv.index('--') < argv.index('uvx')


# ---------------------------------------------------------------------------
# token mapping — measured host state
# ---------------------------------------------------------------------------


def test_hf_download_token_is_mapped_onto_hf_token():
    argv = lms_fetch_weights.fetch_argv(_arm(), {'HF_DOWNLOAD_TOKEN': 'hf_abc'})

    assert '--setenv=HF_TOKEN=hf_abc' in argv


def test_an_existing_hf_token_is_passed_through():
    argv = lms_fetch_weights.fetch_argv(_arm(), {'HF_TOKEN': 'hf_direct'})

    assert '--setenv=HF_TOKEN=hf_direct' in argv


def test_hf_token_wins_when_both_are_set():
    argv = lms_fetch_weights.fetch_argv(
        _arm(), {'HF_TOKEN': 'hf_direct', 'HF_DOWNLOAD_TOKEN': 'hf_abc'},
    )

    assert '--setenv=HF_TOKEN=hf_direct' in argv
    assert '--setenv=HF_TOKEN=hf_abc' not in argv


def test_no_token_at_all_raises_rather_than_downloading_anonymously():
    """An anonymous download succeeds on ungated repos and fails only later,
    only on the gated ones -- so the failure would arrive hours in, attributed
    to the wrong arm."""
    with pytest.raises(lms_fetch_weights.WeightFetchError) as excinfo:
        lms_fetch_weights.fetch_argv(_arm(), {})

    message = str(excinfo.value)
    assert 'HF_TOKEN' in message
    assert 'HF_DOWNLOAD_TOKEN' in message


def test_the_token_is_never_echoed_by_the_reported_unit_name():
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    assert lms_fetch_weights.fetch_unit_name(_arm()) == 'lms-fetch-qwen3.5-9b'
    assert 'hf_measured_value' not in lms_fetch_weights.fetch_unit_name(_arm())
    assert any('hf_measured_value' in a for a in argv)  # it IS passed, just not named


# ---------------------------------------------------------------------------
# the echoed command line must not carry the token
#
# Found in step 20's live run: `_submit` echoed the argv it was about to run,
# which put the real `hf_...` secret on the operator's terminal, into the
# orchestrator transcript that captured it, and into any log the operator
# piped it to.  The token still has to REACH systemd-run, so redaction belongs
# in the echo path only -- which is exactly what makes it worth a test: the
# redacted and executed argv are deliberately different objects, and nothing
# else would notice if they silently converged.
# ---------------------------------------------------------------------------


def test_redact_argv_masks_the_token_value_but_keeps_the_flag_visible():
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    redacted = lms_fetch_weights.redact_argv(argv)

    assert not any('hf_measured_value' in element for element in redacted)
    # The flag itself must still show: an operator debugging an anonymous
    # download needs to see THAT a token was passed, just not which one.
    assert any(element.startswith('--setenv=HF_TOKEN=') for element in redacted)


def test_redact_argv_changes_nothing_else():
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    redacted = lms_fetch_weights.redact_argv(argv)

    assert len(redacted) == len(argv)
    # strict=True restates the length assertion above at the zip itself, so a
    # future divergence fails here rather than silently truncating the sweep.
    for original, shown in zip(argv, redacted, strict=True):
        if original.startswith('--setenv=HF_TOKEN='):
            continue
        assert shown == original


def test_redact_argv_does_not_mutate_the_argv_that_gets_executed():
    """The executed argv must keep the real token -- redacting in place would
    turn every download anonymous, which succeeds until it meets a gated repo."""
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    lms_fetch_weights.redact_argv(argv)

    assert '--setenv=HF_TOKEN=hf_measured_value' in argv


def test_submitting_a_fetch_echoes_the_redacted_form(capsys, monkeypatch):
    """End to end through the CLI: the secret must not reach stdout/stderr.

    `--dry-run` is what makes this offline -- it echoes without submitting --
    and the echo is precisely the leak, so the dry-run path is the honest place
    to assert it.
    """
    monkeypatch.setenv('HF_DOWNLOAD_TOKEN', 'hf_measured_value')
    monkeypatch.delenv('HF_TOKEN', raising=False)
    monkeypatch.setattr(lms_fetch_weights, 'load_arms', lambda: _FakeManifest(_arm()))

    exit_code = lms_fetch_weights.main(
        ['--arm', 'qwen3.5-9b', '--weights-only', '--dry-run']
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert 'hf_measured_value' not in captured.out + captured.err
    assert 'lms-fetch-qwen3.5-9b' in captured.out


class _FakeManifest:
    def __init__(self, *arms):
        self.arms = list(arms)

    def by_id(self, arm_id):
        return next(a for a in self.arms if a.arm_id == arm_id)


# ---------------------------------------------------------------------------
# GGUF arms fetch ONE quant, not a whole repo
# ---------------------------------------------------------------------------


def test_gguf_arm_includes_only_the_pinned_quant_file():
    """A GGUF repo carries every quant; without --include this is ~200 GB
    instead of ~10 GB."""
    arm = _gguf_arm()

    argv = lms_fetch_weights.fetch_argv(arm, TOKEN_ENV)

    assert '--include' in argv
    assert argv[argv.index('--include') + 1] == arm.gguf_file
    assert '--local-dir' in argv
    assert argv[argv.index('--local-dir') + 1].endswith(f'/{arm.arm_id}')


def test_non_gguf_arm_fetches_into_the_hf_cache_without_an_include_filter():
    argv = lms_fetch_weights.fetch_argv(_arm(), TOKEN_ENV)

    assert '--include' not in argv
    assert '--local-dir' not in argv
    assert any(a.startswith('--setenv=HF_HOME=') for a in argv)


def test_gguf_arm_without_a_pinned_file_raises():
    with pytest.raises(lms_fetch_weights.WeightFetchError) as excinfo:
        lms_fetch_weights.fetch_argv(_gguf_arm(gguf_file=None), TOKEN_ENV)
    assert 'gguf_file' in str(excinfo.value)


def test_placeholder_arm_raises_rather_than_fetching_a_literal_tbd():
    with pytest.raises(lms_fetch_weights.WeightFetchError) as excinfo:
        lms_fetch_weights.fetch_argv(
            _gguf_arm(model_ref='TBD-Q3-pick-a-repo'), TOKEN_ENV,
        )
    assert 'TBD' in str(excinfo.value)


# ---------------------------------------------------------------------------
# image pulls take the same shape
# ---------------------------------------------------------------------------


def test_pull_image_argv_uses_the_same_transient_unit_shape():
    arm = _arm()

    argv = lms_fetch_weights.pull_image_argv(arm)

    assert argv[:2] == ['systemd-run', '--user']
    assert '--collect' in argv
    assert _flag(argv, '--unit=') == '--unit=lms-pull-qwen3.5-9b'
    assert argv[-3:] == ['docker', 'pull', arm.image]


@pytest.mark.parametrize('token', _BACKGROUND_SHELL_TOKENS)
def test_pull_image_argv_never_backgrounds_through_a_shell(token):
    argv = lms_fetch_weights.pull_image_argv(_arm())

    for element in argv:
        assert token not in element


def test_pull_image_argv_prefers_a_pinned_digest():
    arm = _arm(image_digest='sha256:' + 'b' * 64)

    argv = lms_fetch_weights.pull_image_argv(arm)

    assert argv[-1] == f'vllm/vllm-openai@sha256:{"b" * 64}'


# ---------------------------------------------------------------------------
# the committed slate
# ---------------------------------------------------------------------------


def test_every_committed_non_placeholder_arm_builds_a_fetch_argv():
    manifest = lms_manifest.load_arms()

    built = [
        arm.arm_id for arm in manifest.arms
        if not arm.is_placeholder
        and lms_fetch_weights.fetch_argv(arm, TOKEN_ENV)[:2] == ['systemd-run', '--user']
    ]

    # Derived, never a literal count -- see the twin assertion in
    # test_lms_serve.py.  Step 22's Open Q3 resolution made the old `== 7` red.
    assert built == [a.arm_id for a in manifest.arms if not a.is_placeholder]
