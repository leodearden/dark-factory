"""run_eval_matrix.sh routes a config to the B200 retry/fallback arm by EXACT membership.

scripts/run_eval_matrix.sh names B200_CONFIGS as the configs that need the
retry+fallback logic, so the launch if/else must ask "is $cfg one of those
names?". An approximation of that question misroutes real configs: `grep -w`
treats `-` as a word boundary, so a hyphen-bounded FRAGMENT of a B200 name
matched too.

Read through the REAL shipped text: each test slices the script's own defaults
block and launch if/else by CODE anchors and runs them against a stub launcher,
so a test asserts on the argv the script builds, never on how the membership
test is spelled. The construct-level ban on `producer | grep -q` in this
script lives in test_quiet_grep_sweep.py.
"""

from __future__ import annotations

from shell_sections import (
    REPO_ROOT,
    run_with_preamble,
    slice_section,
    stub_bin_dir,
    write_stub,
)

RUN_EVAL_MATRIX_PATH = REPO_ROOT / "scripts" / "run_eval_matrix.sh"

# The launch if/else, sliced from its LAUNCH echo to the 4-space `fi` that
# closes it, with the SHIPPED defaults block (CONFIGS through B200_CONFIGS)
# ahead of it, so the membership list under test is the script's own. Both
# arms background the launcher with `&`, hence the appended `wait`.
_MATRIX_DEFAULTS_START = 'CONFIGS="${CONFIGS:-'
_MATRIX_DEFAULTS_END = "B200_CONFIGS="
_MATRIX_LAUNCH_START = "LAUNCH $cfg"
_MATRIX_LAUNCH_END = "\n    fi\n"

# Only the B200 arm passes it.
_B200_ARM_FLAG = "--gpu-retry-minutes"


def _matrix_defaults():
    """`set -uo pipefail` plus run_eval_matrix.sh's own defaults block, verbatim."""
    return "set -uo pipefail\n" + slice_section(
        RUN_EVAL_MATRIX_PATH, _MATRIX_DEFAULTS_START, _MATRIX_DEFAULTS_END
    )


def _launcher_argv(tmp_path, cfg):
    """The argv the launch if/else hands the launcher for *cfg*.

    The shipped PYTHON is an absolute host-venv path, so it is REBOUND to a
    stub that records its argv, not shadowed on PATH.
    """
    argv_file = tmp_path / "launcher-argv"
    python = write_stub(
        stub_bin_dir(tmp_path), "stub-python", f'printf \'%s\\n\' "$@" > {argv_file}\n'
    )
    preamble = _matrix_defaults() + (
        f'PYTHON="{python}"\n'
        "LAUNCHER=run_vllm_eval.py\n"
        "PORT=8200\n"
        f'LOG="{tmp_path / "matrix.log"}"\n'
        f'cfg="{cfg}"\n'
    )
    result = run_with_preamble(
        tmp_path,
        preamble,
        slice_section(RUN_EVAL_MATRIX_PATH, _MATRIX_LAUNCH_START, _MATRIX_LAUNCH_END)
        + "wait\n",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    argv = argv_file.read_text(encoding="utf-8").splitlines()
    assert argv[argv.index("--config") + 1] == cfg, argv
    return argv


def test_matrix_routes_every_b200_config_to_the_retry_arm(tmp_path):
    """Characterization: each name in the shipped B200_CONFIGS gets the retry/fallback flags."""
    listed = run_with_preamble(
        tmp_path, _matrix_defaults(), "printf '%s\\n' $B200_CONFIGS\n"
    )
    b200_configs = listed.stdout.splitlines()
    assert b200_configs, listed.stdout + listed.stderr

    for cfg in b200_configs:
        run_dir = tmp_path / cfg
        run_dir.mkdir()
        assert _B200_ARM_FLAG in _launcher_argv(run_dir, cfg), cfg


def test_matrix_routes_a_default_non_b200_config_to_the_plain_arm(tmp_path):
    """Characterization: a shipped CONFIGS default that is not a B200 config launches plainly."""
    assert _B200_ARM_FLAG not in _launcher_argv(tmp_path, "minimax-m25-nvfp4-new")


def test_matrix_routes_a_fragment_of_a_b200_name_to_the_plain_arm(tmp_path):
    """`reap-172b-nvfp4` is a 2xH200 config, not the B200 `final-reap-172b-nvfp4-gb10`.

    `grep -w` treats `-` as a word boundary, so the fragment matched and the
    2xH200 config was launched with B200 GPU-retry and H200-fallback flags.
    """
    assert _B200_ARM_FLAG not in _launcher_argv(tmp_path, "reap-172b-nvfp4")
