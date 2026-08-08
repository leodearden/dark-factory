"""Fetch arm weights and container images inside transient systemd --user units.

PRD-MARKER:local-memory-models-eval serving

Task 3713 (LME-alpha) of `plans/local-memory-models-eval-prd.md`.

PRD hazard 5 is binding: "long runs, weight downloads included, go in transient
`systemd --user` units, never bare background shells".  A ~46 GB fetch through
`nohup ... &` is unsupervised, unloggable, and dies with the invoking session;
a transient unit is followed with `journalctl --user -u <unit>` and survives.

TWO MEASURED FACTS shape the argv:

1. `systemd-run --user` propagates NONE of the caller's environment
   (measured and recorded at
   orchestrator/src/orchestrator/proc_supervision.py:91-92).  So the token,
   the HF cache location and the working directory are all passed explicitly.
2. On this host `HF_DOWNLOAD_TOKEN` is set (shell env and
   `/home/leo/src/dark-factory/.env`) while `HF_TOKEN` and
   `HUGGING_FACE_HUB_TOKEN` are NOT.  `huggingface_hub` reads the latter
   names, so an unmapped token means a silent ANONYMOUS download -- which
   succeeds on ungated repos and fails hours later on the gated ones (the
   Mistral and Gemma families), attributed to the wrong arm.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from lms_manifest import ArmEntry, ArmManifestError, load_arms
from lms_serve import HOST_HF_CACHE, REPO_ROOT, image_ref, host_gguf_dir


class WeightFetchError(Exception):
    """This arm's weights cannot be fetched as specified."""


def fetch_unit_name(arm: ArmEntry) -> str:
    return f'lms-fetch-{arm.arm_id}'


def pull_unit_name(arm: ArmEntry) -> str:
    return f'lms-pull-{arm.arm_id}'


def resolve_hf_token(env: dict[str, str]) -> str:
    """Map whatever token this host has onto the name huggingface_hub reads.

    Raises rather than returning ''.  An anonymous download is the failure
    mode that hides: it works until it meets a gated repo.
    """
    token = env.get('HF_TOKEN') or env.get('HF_DOWNLOAD_TOKEN')
    if not token:
        raise WeightFetchError(
            'no HuggingFace token in the environment: neither HF_TOKEN nor '
            'HF_DOWNLOAD_TOKEN is set. Refusing to launch an anonymous '
            'download, which would succeed on ungated repos and fail only '
            'later on the gated ones (Mistral, Gemma)'
        )
    return token


def _transient_unit_prefix(unit: str, setenv: list[str]) -> list[str]:
    return [
        'systemd-run', '--user',
        # Reap the unit once it finishes so a re-run is not blocked by the
        # corpse of the previous one.
        '--collect',
        f'--unit={unit}',
        # Explicit: systemd --user would otherwise run from $HOME.
        f'--working-directory={REPO_ROOT}',
        *setenv,
        '--',
    ]


def fetch_argv(arm: ArmEntry, env: dict[str, str] | None = None) -> list[str]:
    """The transient-unit argv that downloads one arm's weights."""
    environment = dict(os.environ) if env is None else env
    if arm.is_placeholder:
        raise WeightFetchError(
            f'arm {arm.arm_id!r} still carries TBD placeholders '
            f'(model_ref={arm.model_ref!r}); resolve the PRD open question '
            'that owns it before fetching'
        )

    token = resolve_hf_token(environment)
    setenv = [f'--setenv=HF_TOKEN={token}']

    payload = ['uvx', '--from', 'huggingface_hub', 'hf', 'download', arm.model_ref]

    if arm.stack == 'llamacpp':
        if not arm.gguf_file:
            raise WeightFetchError(
                f'arm {arm.arm_id!r} is stack=llamacpp but pins no gguf_file. A '
                'GGUF repo carries every quant, so fetching without --include '
                'is ~200 GB instead of ~10 GB'
            )
        payload += [
            '--include', arm.gguf_file,
            '--local-dir', str(host_gguf_dir(arm)),
        ]
    else:
        setenv.append(f'--setenv=HF_HOME={HOST_HF_CACHE}')

    return _transient_unit_prefix(fetch_unit_name(arm), setenv) + payload


def pull_image_argv(arm: ArmEntry) -> list[str]:
    """The transient-unit argv that pulls one arm's serving image."""
    return _transient_unit_prefix(pull_unit_name(arm), []) + [
        'docker', 'pull', image_ref(arm),
    ]


#: What replaces the token in the ECHOED argv.  The flag itself stays visible:
#: an operator debugging an unexpectedly anonymous download needs to see THAT a
#: token was passed without the secret landing in their scrollback.
TOKEN_REDACTION = '<redacted>'

_TOKEN_FLAG = '--setenv=HF_TOKEN='


def redact_argv(argv: list[str]) -> list[str]:
    """A copy of `argv` safe to print, with the HF token masked.

    Returns a NEW list and never mutates its argument: the executed argv must
    keep the real token, or every download silently goes anonymous -- which
    succeeds on ungated repos and fails only later, only on the gated ones.
    """
    return [
        f'{_TOKEN_FLAG}{TOKEN_REDACTION}' if element.startswith(_TOKEN_FLAG) else element
        for element in argv
    ]


def _submit(argv: list[str], *, dry_run: bool) -> int:
    # Echo the redacted copy, run the real one.  Discovered in step 20's live
    # smoke: the unredacted echo put the secret on the terminal and into the
    # transcript, and `systemd-run --setenv` would have put it in the journal's
    # unit properties besides.
    print(' '.join(redact_argv(argv)), flush=True)
    if dry_run:
        return 0
    return subprocess.run(argv, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='lms_fetch_weights',
        description='Fetch arm weights/images in transient systemd --user units.',
    )
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument('--arm', help='one arm_id from arms.yaml')
    selector.add_argument('--all', action='store_true', help='every non-placeholder arm')
    parser.add_argument('--images-only', action='store_true')
    parser.add_argument('--weights-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    try:
        manifest = load_arms()
        arms = manifest.arms if args.all else [manifest.by_id(args.arm)]
    except ArmManifestError as exc:
        print(f'lms_fetch_weights: {exc}', file=sys.stderr)
        return 2

    failures = 0
    submitted: list[str] = []
    for arm in arms:
        if arm.is_placeholder:
            # Loud, but not fatal for --all: the rest of the slate is fetchable
            # and a silent skip is what would hide the gap.
            print(
                f'lms_fetch_weights: SKIPPING {arm.arm_id} — unresolved TBD '
                f'placeholder (model_ref={arm.model_ref!r})',
                file=sys.stderr,
            )
            failures += 1
            continue
        try:
            if not args.weights_only:
                if _submit(pull_image_argv(arm), dry_run=args.dry_run) == 0:
                    submitted.append(pull_unit_name(arm))
            if not args.images_only:
                if _submit(fetch_argv(arm), dry_run=args.dry_run) == 0:
                    submitted.append(fetch_unit_name(arm))
        except WeightFetchError as exc:
            print(f'lms_fetch_weights: {arm.arm_id}: {exc}', file=sys.stderr)
            failures += 1

    if submitted:
        print('\nfollow these transient units with:')
        for unit in submitted:
            print(f'    journalctl --user -u {unit} -f')
    return 1 if failures else 0


if __name__ == '__main__':  # pragma: no cover - process entry point
    raise SystemExit(main())
