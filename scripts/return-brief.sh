#!/usr/bin/env bash
# return-brief.sh -- the 05:30 cross-project return brief (task 5376). Invoked
# by scripts/return-brief.service's ExecStart, on scripts/return-brief.timer.
# Shaped after scripts/memory-metadata-coverage-census.sh, minus its commit
# step: data/return-brief.md is gitignored (/data/ is root-anchored).
#
# TWO STEPS, IN SEQUENCE.
#   1. scripts/sitting/nightly_prepare.py -- the prepare-sitting mode, headless
#      on Fable and read-only; it records judgement under data/sitting/.
#   2. scripts/sitting/return_brief.py -- the deterministic render, which
#      measures every figure itself and writes data/return-brief.md.
# The render runs whatever prepare returned: a capped or timed-out night
# still produces a page, with unprepared items marked "awaiting preparation".
#
# SEAMS. PREPARE_CMD and RETURN_BRIEF_CMD override the two interpreter
# prefixes (tests inject a fake recorder); REPO overrides the checkout;
# RETURN_BRIEF_SKIP_PREPARE=1 skips step 1 for a no-cost re-render.
#
# ALWAYS EXITS 0 (OPERATIONS.md §12): a recurring oneshot that exits non-zero
# enters systemd `failed` state and stays there, silently ending the nightly
# job. Both exit codes are narrated in the closing `done (...)` line instead.
#
# No git and no .env sourcing: account_pool.build_pool loads the repo .env
# itself, after the unit has unset ANTHROPIC_API_KEY.
set -uo pipefail

REPO="${REPO:-/home/leo/src/dark-factory}"

# shellcheck disable=SC2206
PREPARE_CMD=(${PREPARE_CMD:-uv run --frozen --project "$REPO/shared" python})
# shellcheck disable=SC2206
RETURN_BRIEF_CMD=(${RETURN_BRIEF_CMD:-uv run --frozen --project "$REPO/shared" python})

if [ "${RETURN_BRIEF_SKIP_PREPARE:-0}" = "1" ]; then
    prepare_rc=skipped
    echo "return-brief: RETURN_BRIEF_SKIP_PREPARE=1 — rendering without a Fable prepare run"
else
    echo "return-brief: preparing the sitting headless on Fable (read-only)"
    "${PREPARE_CMD[@]}" "$REPO/scripts/sitting/nightly_prepare.py"
    prepare_rc=$?
    if [ "$prepare_rc" -ne 0 ]; then
        echo "return-brief: prepare exited $prepare_rc — rendering anyway;" \
             "unprepared items show as awaiting preparation" >&2
    fi
fi

echo "return-brief: rendering $REPO/data/return-brief.md"
"${RETURN_BRIEF_CMD[@]}" "$REPO/scripts/sitting/return_brief.py" --output "$REPO/data/return-brief.md"
brief_rc=$?
if [ "$brief_rc" -ne 0 ]; then
    echo "return-brief: render exited $brief_rc" >&2
fi

echo "return-brief: done (prepare=$prepare_rc brief=$brief_rc)"
exit 0
