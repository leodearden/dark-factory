#!/usr/bin/env bash
# setup-host.sh — idempotent bootstrap for a dark-factory development host.
# Assumes the repo is already cloned (with --recurse-submodules).
# Run from anywhere: bash /path/to/dark-factory/scripts/setup-host.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/fused-memory/docker/docker-compose.yml"

info()  { printf '\033[1;34m==> %s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  ✓ %s\033[0m\n' "$*"; }
warn()  { printf '\033[1;33m  ! %s\033[0m\n' "$*"; }
fail()  { printf '\033[1;31m  ✗ %s\033[0m\n' "$*"; }

# Classify what a parity checker just told us. Defined HERE, once, immediately
# below the log shims and above every parity call site — the five sites share
# one shell scope, and every sliced block is executed by the test harness under
# a preamble that provides this helper (tests/scripts/setup_host_sections.py
# slices it live out of this file, so there is no second copy to drift).
#
#   _parity_verdict <captured_output> <exit_status> <bracketed_tag>
#
# echoes exactly one of:
#   unreported — the tag is ABSENT, so the checker did not report. Its status
#                says NOTHING about this host: exit 2 in particular is also what
#                python3 returns for a script it cannot open and what argparse
#                returns for a rejected flag. Checked FIRST and outranking the
#                status entirely, which is the whole point of the guard.
#   parity     — it reported, and found what it was looking for (0).
#   absent     — it reported that the thing is not installed here (2).
#   finding    — it reported something actionable (1, or any other status: 127
#                and friends are not benign just because they are undocumented).
#
# CLASSIFICATION ONLY. Each call site keeps its own wording, severity and side
# effects, because those legitimately differ — `absent` is `info` at the
# orchestrator, dashboard pre-install and lms sites but `warn` at the
# fused-memory and dashboard post-install ones; the orchestrator gate may
# `fail` and sets _orch_install_blocked, while the lms gate is forbidden from
# ever calling `fail` (test_setup_host_lms_parity_gate.py::test_the_lms_gate_is_warn_only).
# A helper that also emitted messages would have to collapse those distinctions
# or take five message parameters — either way re-creating the coupling the
# extraction removes.
#
# The tag is matched with bash's own `[[ ]]`, never `printf | grep -q`. grep
# exits the instant it matches and the tag is on line 1, so once the report
# exceeds the pipe buffer (~64KB) the printf dies of SIGPIPE, `pipefail` makes
# the pipeline return 141, and that becomes "it did not run" on a report that
# plainly carries the tag — a verdict manufactured by the mechanism rather than
# read from the checker, which is the exact class of failure this guard exists
# to remove. `[[ ]]` forks nothing and cannot be signalled. (Measured: the pipe
# form flips at ~82KB of tagged output; this does not.)
_parity_verdict() {
  local _out="$1" _status="$2" _tag="$3"
  if [[ "$_out" != *"$_tag"* ]]; then
    printf '%s\n' unreported
  elif [ "$_status" -eq 0 ]; then
    printf '%s\n' parity
  elif [ "$_status" -eq 2 ]; then
    printf '%s\n' absent
  else
    printf '%s\n' finding
  fi
}

# Does FalkorDB answer? ONE probe, two callers — the section-2 wait loop and the
# section-12 health check ask exactly the same question, and a copy at each site
# is how the two drift apart.
#
# The verdict is read from the captured REPLY, not from a pipeline's exit
# status. `... ping | grep -q PONG` answers with the PRODUCER's status, which is
# a different question and gets this wrong two ways. `grep -q` exits on its
# first match and closes the read end, so a producer still writing dies of
# SIGPIPE and `pipefail` hands the caller that 141; and the same conflation
# misreads any producer that emits PONG and then exits non-zero for reasons of
# its own (an exec whose status covers the whole run, not the one line asked
# about). Either way a live FalkorDB is reported as down. (Measured: with 256KiB
# of trailing output the pipe form reports no PONG 30/30; this form reports it.)
#
# `|| true` is load-bearing: without it the assignment is a simple command and
# `set -e` kills the bootstrap the moment docker is unavailable, where the old
# pipeline merely took the else branch. It also must not be `|| out=""` — that
# throws away a reply the producer did write, preserving the bug. A producer
# that wrote nothing still yields no match, which is the not-answering verdict
# both callers already gave.
#
# `local` confines the reply to the call, so neither caller can ever read a
# verdict the other left behind in the shared shell scope.
falkordb_pings() {
  local out
  out="$(docker compose -f "$COMPOSE_FILE" exec -T falkordb redis-cli ping 2>/dev/null)" || true
  [[ "$out" == *PONG* ]]
}

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
info "Checking prerequisites"

# Docker
if command -v docker &>/dev/null && docker compose version &>/dev/null; then
  ok "Docker + Compose v2"
else
  warn "Docker not found — installing via get.docker.com"
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  warn "Added $USER to docker group — you may need to log out and back in"
fi

# uv
if command -v uv &>/dev/null; then
  ok "uv ($(uv --version))"
else
  warn "uv not found — installing"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  ok "uv installed ($(uv --version))"
fi

# Node 22+
if command -v node &>/dev/null; then
  NODE_MAJOR=$(node --version | sed 's/v\([0-9]*\).*/\1/')
  if [ "$NODE_MAJOR" -ge 22 ]; then
    ok "Node $(node --version)"
  else
    warn "Node $(node --version) found but >= 22 required"
    warn "Install Node 22 via your preferred method (nvm, fnm, nodesource)"
  fi
else
  warn "Node not found — install Node 22 via nvm, fnm, or nodesource"
fi

# Claude Code
if command -v claude &>/dev/null; then
  ok "Claude Code ($(claude --version 2>/dev/null || echo 'installed'))"
else
  warn "Claude Code not found — install with: npm install -g @anthropic-ai/claude-code"
fi

# System packages
for pkg in curl jq bubblewrap; do
  if command -v "$pkg" &>/dev/null; then
    ok "$pkg"
  else
    warn "$pkg not found — installing"
    sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg"
  fi
done

# ---------------------------------------------------------------------------
# 2. Docker Compose — start backing stores
# ---------------------------------------------------------------------------
info "Starting backing stores (FalkorDB + Qdrant)"

mkdir -p "$REPO_ROOT/fused-memory/data/falkordb"
mkdir -p "$REPO_ROOT/fused-memory/data/qdrant"

docker compose -f "$COMPOSE_FILE" up -d falkordb qdrant

# Wait for healthy
for i in $(seq 1 30); do
  # Matched in BASH, not through `| grep -q` — see falkordb_pings above for why
  # that pipeline can report a live server as never healthy.
  if falkordb_pings; then
    ok "FalkorDB healthy"
    break
  fi
  [ "$i" -eq 30 ] && fail "FalkorDB did not become healthy in 30s"
  sleep 1
done

for i in $(seq 1 30); do
  if curl -sf http://localhost:6333/readyz &>/dev/null; then
    ok "Qdrant healthy"
    break
  fi
  [ "$i" -eq 30 ] && fail "Qdrant did not become healthy in 30s"
  sleep 1
done

# ---------------------------------------------------------------------------
# 3. Python subprojects — uv sync (dependency order)
# ---------------------------------------------------------------------------
info "Syncing Python subprojects"

for proj in shared escalation fused-memory orchestrator dashboard; do
  (cd "$REPO_ROOT/$proj" && uv sync --quiet)
  ok "$proj"
done

# ---------------------------------------------------------------------------
# 4. Systemd user unit for fused-memory
# ---------------------------------------------------------------------------
info "Installing fused-memory systemd unit"

UNIT_DIR="$HOME/.config/systemd/user"
mkdir -p "$UNIT_DIR"

UV_PATH="$(command -v uv)"

# THE RENDER. No longer `sed ... > "$UNIT_DIR/fused-memory.service"`, and the
# difference is not stylistic (task 4796; same fix as section 8, task 4793).
#
# scripts/fused-memory.service.template declares
# `Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=__REPO_ROOT__`, which renders to a
# SINGLE root. Further project roots are host-LOCAL settings, appended to the
# INSTALLED unit and deliberately not committed — the template's own comment
# says so. A truncating redirect destroyed them on every re-run.
#
# AND ON THIS UNIT THAT IS WORSE THAN THE DASHBOARD CASE. Here the variable is
# not a view setting: fused_memory/models/scope.py reads it as
# KNOWN_PROJECT_ROOTS_ENV, and reconciliation/harness.py raises
# UnknownProjectError for a project outside the resulting set. Collapsing it
# de-registers every OTHER project from RECONCILIATION — and invisibly, because
# the post-install parity gate (section 12) checks only host-invariant safety
# directives and is structurally incapable of seeing this variable's value.
#
# THE RENDERER OWNS THE DESTINATION rather than being redirected into it.
# `python3 render_dashboard_unit.py ... > "$UNIT_DIR/<unit>"` would be the same
# defect one level up: bash truncates the destination before python ever opens
# it, so the installed value would be gone before it could be read and the tool
# would preserve nothing while reporting success. --output is read FIRST as the
# installed copy, then replaced atomically.
#
# AND THERE IS DELIBERATELY NO sed FALLBACK. Rendering "the old way" when the
# renderer is missing would reinstate the exact clobber it replaced, on the one
# path where nobody is left watching for it. A missing renderer therefore leaves
# the unit ALONE and says so, which is the recoverable direction: stale but
# intact is fixable on the next run; de-registered from reconciliation is not
# noticed until projects start failing with UnknownProjectError.
#
# KEEP IN STEP WITH SECTION 8's `_dash_render_script=` BLOCK. The two are
# deliberately parallel: same hoisted `_<x>_render_script=` anchor, same
# `_<x>_rendered=0` flag, same missing-renderer / rendered / refused three-way,
# same "daemon-reload always, enable on the unit EXISTING, the destructive step
# on the flag" gate split. They are not factored into one helper because every
# message differs (this unit governs reconciliation; that one governs a view)
# and a helper would have to signal WHICH failure occurred back to the call site
# through an exit code, reproducing the three-way at both ends. The cost of that
# choice is drift, so: a change to the control flow or the failure modes here
# belongs in BOTH sites. The one INTENTIONAL divergence is documented at the
# `restart` gate below — section 8 has no equivalent because the dashboard is
# started by hand and never restarted by this script.
_fm_render_script="$REPO_ROOT/scripts/render_dashboard_unit.py"

# Set to 1 only by the branch that actually rendered. `fail` here is a printf,
# not an exit, so without this flag every degraded path still reached the
# `restart` below and the green "installed and started" line under it — bouncing
# the server that backs the orchestrators, the dashboard and this session's own
# MCP tooling on the strength of an install that did not happen.
_fm_rendered=0

if [ ! -f "$_fm_render_script" ]; then
  fail "fused-memory unit renderer missing: $_fm_render_script"
  fail "  NOT rendering it the old way — a plain template render would strip"
  fail "  this host's local DASHBOARD_KNOWN_PROJECT_ROOTS entries, which is what"
  fail "  registers other projects with RECONCILIATION, and the post-install"
  fail "  parity check cannot see that variable's value."
  fail "  $UNIT_DIR/fused-memory.service is left AS-IS. The sections below still run."
elif python3 "$_fm_render_script" \
       --unit      fused-memory \
       --template  "$REPO_ROOT/scripts/fused-memory.service.template" \
       --repo-root "$REPO_ROOT" \
       --uv-path   "$UV_PATH" \
       --output    "$UNIT_DIR/fused-memory.service"; then
  _fm_rendered=1
else
  # The renderer RAN and refused. Without this branch the `elif` chain falls
  # through with status 0 and says nothing about the unit that did not get
  # written — reports-green-because-it-never-ran, one construct over. Leaving
  # `_fm_rendered` at 0 is the other half: it keeps the systemctl calls and the
  # section's closing line from claiming an install that did not happen.
  fail "fused-memory unit render FAILED — see the [fused_memory_unit_render]"
  fail "  report above for which step refused and why."
  fail "  $UNIT_DIR/fused-memory.service was left UNTOUCHED: this host's local"
  fail "  Environment= values (DASHBOARD_KNOWN_PROJECT_ROOTS) survived, and the"
  fail "  unit is at worst STALE — which the parity check reports on the next"
  fail "  run. Re-run this script once the cause is fixed."
fi

# UNCONDITIONAL, exactly as before this task and exactly as in section 8. It is
# a no-op when nothing changed, and skipping it on a degraded path would leave
# systemd reading a stale generation of whatever unit IS on disk.
systemctl --user daemon-reload

# GUARDED on the unit EXISTING, not on `_fm_rendered` — the same split section 8
# makes, for the same reasons. (1) A failed render on a host that already HAS the
# unit must still leave it enabled: stale but supervised is the recoverable
# direction this whole construct chooses, and `enable` is idempotent and cheap.
# Before this gate existed the pre-4796 code enabled unconditionally, so gating
# `enable` on `_fm_rendered` would have been a silent REGRESSION on exactly the
# render-refused path (renderer missing, or apply_preserved refusing a value that
# cannot round-trip through one Environment= line). (2) The combination the guard
# genuinely exists for is a BARE host plus a failed render — there
# `systemctl --user enable` on a unit that does not exist exits non-zero, and
# under this file's `set -e` that aborts the entire installer before every later
# section. Both FAIL branches above promise "the sections below still run"; this
# is what makes that promise true rather than true-only-when-a-unit-was-there.
if [ -f "$UNIT_DIR/fused-memory.service" ]; then
  systemctl --user enable fused-memory
else
  fail "fused-memory NOT enabled: no unit file in $UNIT_DIR."
  fail "  The render above did not happen and this host had no previous copy,"
  fail "  so there is nothing to enable. The sections below still run."
fi

# THE RESTART, and ONLY the restart, stays gated on `_fm_rendered`. This is the
# one deliberate divergence from section 8 (which has no restart at all — the
# dashboard is started by hand). `enable` is idempotent bookkeeping; `restart`
# bounces the server backing the orchestrators, the dashboard and this session's
# own MCP tooling. On a path where nothing was written that outage buys exactly
# nothing: the on-disk unit is unchanged, so the running process already matches
# it. The closing `ok` is inside the gate for the same reason it is in section 8
# — a green "installed and started" would assert precisely what the FAIL lines
# above it had just denied.
if [ "$_fm_rendered" = "1" ]; then
  # Only start if .env exists (needs secrets)
  if [ -f "$REPO_ROOT/fused-memory/.env" ]; then
    systemctl --user restart fused-memory
    ok "fused-memory unit installed and started (host-local Environment= values preserved — see the [fused_memory_unit_render] lines above)"
  else
    warn "fused-memory unit installed but NOT started (fused-memory/.env missing)"
  fi
fi

# ---------------------------------------------------------------------------
# 5. Orchestrator systemd units + watchdog
# ---------------------------------------------------------------------------
# The install is gated PER UNIT (task 4198). Each unit is judged on its own
# parity verdict, so a finding on one no longer declines the install of all
# nine. The POLICY is unchanged and still ratified — a unit that did not clear
# is never overwritten without DF_INSTALL_ORCH_UNITS=1, because a difference
# does not tell you which side is stale — only the blast radius shrinks.
#
# Installs and enables (kept in step with scripts/orchestrator-*.service by
# test_setup_host_installs_every_orchestrator_unit — every template must be
# copied here, and every template with an [Install] section must be enabled):
#   - /home/leo/bin/wait-for-port.py        (port-wait helper used by ExecStartPre)
#   - orchestrator-reify.service            (reify orchestrator, escalation 8100)
#   - orchestrator-dark-factory.service     (dark-factory orchestrator, escalation 8102)
#   - orchestrator-know-live.service        (know-live orchestrator, escalation 8105)
#   - orchestrator-my-solar-challenge.service (my-solar-challenge, escalation 8106)
#   - orchestrator-solar-challenge-platform.service (platform, escalation 8107)
#   - orchestrator-pump-web-ui.service      (pump-web-ui orchestrator, escalation 8108)
#   - orchestrator-autopilot-video.service  (autopilot-video orchestrator, escalation 8101)
#   - orchestrator-watchdog.service/.timer  (60s liveness probe + dead-enabled revival)
#     The .service is static (no [Install]) — the .timer carries the install, so
#     only the timer is enabled below. `systemctl enable` on it would error.
#     This pair is WHY the gate is per-unit. Under the old all-or-nothing gate
#     the deliberate, permanent orchestrator-reify.service.d/warm-lane.conf
#     drop-in on this host (owned by the reify repo, installed by its
#     install-warm-lane-units.sh — NOT drift, and not going away) declined the
#     install of every unit including these two, so a plain re-run could never
#     reinstall or re-enable the supervision safety net. On 2026-08-10 that
#     left the fleet 31.8h stale. It now blocks reify alone, and a plain
#     `bash scripts/setup-host.sh` is the natural repair path for the pair.
#
# Drift direction (task 3641): know-live and pump-web-ui were transcribed from
# the running host into the repo, i.e. committed-follows-installed. BOTH have
# since been reconciled back in the installer's direction — know-live by task
# 3642 (2026-08-06, reinstall + daemon-reload + restart) and pump-web-ui by
# task 3763 (2026-08-08, reinstall from the committed template + daemon-reload;
# no restart was needed, the reloaded config takes effect at the next restart
# scheduling) — so this installer run is now a genuine no-op for BOTH
# transcribed units, with no committed line left ahead of the host in either.
# pump-web-ui's RestartSteps=4 was the last such line: without it systemd
# ignores that unit's RestartMaxDelaySec= cap and its advertised 10s->60s
# backoff never engages. It is now installed and live (`systemctl --user show
# orchestrator-pump-web-ui.service -p RestartSteps` reports 4, and
# `systemd-analyze --user verify` no longer warns for this unit). That
# template's own header — the forward-fix note owned by task 3424 — is
# deliberately retained as the standing instruction to any future parity pass
# NOT to reconcile RestartSteps=4 back out, since a rebuilt host or a stale
# re-install would reopen exactly the gap it warns about.
#
# The orchestrator units run `uv run --frozen ...`, so process start never
# implicitly re-syncs the shared dark-factory/.venv. After any dependency change
# (or a fresh checkout) run scripts/sync-orchestrator-env.sh once to materialize
# the runtime venv on the .python-version pin — the watchdog only port-probes and
# will NOT repair a missing/stale venv (a frozen-start failure that exhausts
# StartLimitBurst is left stopped for operator attention).
#
# The unit files reference /home/leo/bin/wait-for-port.py from ExecStartPre,
# so the helper lives under ~/bin (stable absolute path across repo moves).
# curl is broken on this host (libcurl.so.4 load failure), so all port probes
# use python urllib / raw sockets.
info "Installing orchestrator units + watchdog"

mkdir -p "$HOME/bin"
install -m 0755 "$REPO_ROOT/scripts/wait-for-port.py" "$HOME/bin/wait-for-port.py"

# Pre-install parity gate. Runs BEFORE the cp block below, and that ordering
# is the whole point: once these units have been overwritten there is no drift
# left to observe, so a post-install-only check would report green on exactly
# the divergence this exists to surface. (No post-install re-check either —
# unlike the dashboard's template-rendered unit, these are plain cp targets, so
# a second run would only restate what the copy just did.)
#
# NON-FATAL, but not merely advisory: a finding never aborts this script (the
# sections below still run, and five of the seven registered units are KNOWN
# RED on this host until the follow-up task lands) — it makes the install of
# THAT UNIT opt-in instead. A bare warning would not be an intervention point
# in a non-interactive `set -e` script: it scrolls past and the next line
# overwrites the units anyway, so the operator is told to check the direction
# at the one moment they can no longer act on it.
#
# Mechanically: the gate runs ONCE and reports a verdict per unit; the
# per-unit install decision is taken from those verdicts further down. See the
# section header for the policy that shape implements.
#
# NOTE the report does not mean "the installed copy is stale". Measured
# 2026-08-02, the direction varies per unit: the repo copy is correct for
# RestartSteps=4, but the INSTALLED copy is correct for the ExecStart --config
# path (two committed units name config files that do not exist). That is why
# the skip is the default and DF_INSTALL_ORCH_UNITS=1 is the override, rather
# than the reverse.
# The exit code alone is NOT trusted, because 2 is overloaded three ways:
# the checker's "not installed on this host" (benign), `python3` refusing to
# open a missing script file, and argparse rejecting an unknown flag. Renaming
# the checker or one of its flags would therefore make this block print a
# reassuring "installing below" and copy the units anyway — a gate reporting
# green because it never ran, which is exactly the silent-drift failure the
# checker exists to catch, reproduced one level up in its own wiring. So the
# invocation is guarded by an existence check, and NO exit code is believed
# unless the checker's own [orchestrator_unit_parity] tag appears in the
# output it produced.
#
# HARNESS CONSTRAINT, stated once, HERE, because this next line is what anchors
# it. tests/scripts/test_check_orchestrator_unit_parity.py slices this file
# from the `_orch_parity_script=` ASSIGNMENT below through the install
# construct's closing `fi`, and EXECUTES that slice under bash.
#
# The anchor is CODE and unique to this site — the same line the structural
# sweep in test_check_dashboard_unit_parity.py discovers, and the shared slicer
# (tests/scripts/setup_host_sections.py) skips comment lines, so prose that
# merely QUOTES the anchor cannot move the slice. That retires the rule this
# comment used to state: naming the checker's file up in the section header no
# longer drags the slice's start upward over the `install -m 0755 ...
# "$HOME/bin/..."` above, and two tests now hold that shut rather than a
# request to remember it.
#
# The other consequence still holds and is NOT enforced by a test: anything the
# sliced code needs (the unit array, the skip-reason helper) must be declared
# BELOW this line, or it is unbound at run time and kills the run under
# `set -u`.
_orch_parity_script="$REPO_ROOT/scripts/check_orchestrator_unit_parity.py"

# The units this section installs, declared ONCE. Both loops below iterate this
# array, and check_orchestrator_unit_parity.py's UNITS registry is checked
# against it by tests/scripts/test_check_orchestrator_unit_parity.py — so
# adding a unit is a one-line edit here instead of a `cp` line plus an
# `enable` line kept in step by hand.
#
# Declared below the parity-script assignment rather than up beside the section
# header — see the harness constraint stated there.
#
# ENABLE POLICY. Every project orchestrator is enabled by default, to match the
# running production stack — they coexist on separate escalation ports, noted
# per entry below. Disable selectively (see SETUP.md's `systemctl --user
# disable --now` block) if a host should not be part of the unattended
# workload. This script marks real exclusions with an explicit "Deliberately
# NOT wired" note; an absent note means the unit belongs here.
_orch_units=(
  orchestrator-dark-factory.service              # this repo's own orchestrator, escalation 8102
  orchestrator-reify.service                     # reify, escalation 8100
  orchestrator-autopilot-video.service           # joined 2026-05-29, escalation 8101 (separate target, selected purely via --config)
  orchestrator-my-solar-challenge.service        # joined 2026-05-31, escalation 8106
  orchestrator-solar-challenge-platform.service  # joined 2026-06-21, escalation 8107
  orchestrator-know-live.service                 # escalation 8105; committed since e5273d8623, wired in by task 3641 (its absence was an omission, not a policy)
  orchestrator-pump-web-ui.service               # joined 2026-07-17, escalation 8108; ran on the host with no committed template until task 3641 transcribed it
  orchestrator-watchdog.service                  # static (no [Install]) — pulled in by the .timer, so the enable loop below skips it
  orchestrator-watchdog.timer                    # 60s liveness probe + dead-enabled revival: the safety net that revives an orchestrator killed by e.g. a boot-race dependency cancel
)

# 1 => the run as a WHOLE reported something unverifiable. Still used for the
# operator-facing summary below; the install decision itself is per-unit.
_orch_install_blocked=0

# unit -> the comma-joined verdict kinds the gate reported for it. A unit
# ABSENT from this map has no verdict and is treated as `unverified` below,
# which is BLOCKING: the states that produce no verdict line (checker missing,
# renamed, run without --print-verdicts, or simply not knowing that unit) all
# mean "nothing was checked", and installing on the strength of nothing is the
# silent-drift failure this gate exists to catch.
declare -A _orch_verdict=()
# Initialised before the branch so the parse below is safe under `set -u` even
# when the checker was missing and the else-branch never ran.
_orch_parity_out=""

if [ ! -f "$_orch_parity_script" ]; then
  fail "Orchestrator parity gate missing: $_orch_parity_script"
  fail "  Not treating that as 'nothing to check' — it is 'nothing checked'."
  _orch_install_blocked=1
else
  _orch_parity_out="$(python3 "$_orch_parity_script" \
       --installed-dir "$UNIT_DIR" \
       --repo-root     "$REPO_ROOT" \
       --print-verdicts 2>&1)" && _orch_parity_exit=0 || _orch_parity_exit=$?
  printf '%s\n' "$_orch_parity_out"

  # Classified by the shared helper at the top of this file — see its comment
  # for why the tag outranks the status and why it is matched with `[[ ]]`.
  _orch_parity_verdict="$(_parity_verdict "$_orch_parity_out" \
       "$_orch_parity_exit" '[orchestrator_unit_parity]')"
  case "$_orch_parity_verdict" in
  unreported)
    fail "Orchestrator parity gate produced no [orchestrator_unit_parity] report"
    fail "  (status $_orch_parity_exit) — it did not run, so its status says"
    fail "  nothing about this host. Check the script path and its flags."
    _orch_install_blocked=1
    ;;
  parity)
    ok "Orchestrator units: parity with committed copies"
    ;;
  absent)
    info "Orchestrator units: not yet installed in $UNIT_DIR (installing below)"
    ;;
  finding | *)
    # `*` folded in with `finding` deliberately: the helper is total over four
    # tokens today, and a fifth added later must land on the LOUD arm rather
    # than fall through a `case` that silently matches nothing — the same
    # "benign because undocumented" reading the helper itself refuses.
    #
    # A finding is "drift OR unverifiable" — it also covers a vanished
    # committed unit, an unreadable unit file and a drop-in override, which the
    # checker words apart so the operator is not sent hunting for a directive
    # diff that does not exist.
    warn "Orchestrator units: drift or unverifiable state — see the"
    warn "  [orchestrator_unit_parity] report above. Note a drop-in override"
    warn "  needs manual removal."
    _orch_install_blocked=1
    ;;
  esac
fi

# A unit that did not clear is SKIPPED rather than warned about, because a
# warning scrolling past in a non-interactive `set -e` script is not an
# intervention point: the very next line would overwrite the installed unit.
# A finding does not mean the installed copy is the stale one — measured
# 2026-08-02, two COMMITTED units name --config paths that do not exist on this
# host, so copying them would break those orchestrators on their next restart.
# The gate stays non-fatal (it never aborts the run; sections below still
# execute), it just declines to act on an unverified diff without being told.
#
# Parse the machine-readable verdict lines the gate just printed.
#
# `while read` fed by a HERE-STRING, never `... | while read`: a pipeline runs
# the loop body in a SUBSHELL, so every _orch_verdict[...] assignment would be
# discarded when it exits and every unit would silently fall through to
# `unverified`. That is a fail-safe-SHAPED bug — it installs nothing and looks
# exactly like a gate finding — so it would survive a casual read of the output.
# `|| true` keeps a grep that matches nothing from tripping set -e/pipefail.
while read -r _tag _kw _unit _kinds; do
  [ "$_kw" = verdict ] || continue
  _orch_verdict["$_unit"]="$_kinds"
done <<< "$(printf '%s\n' "$_orch_parity_out" \
            | grep -F '[orchestrator_unit_parity] verdict ' || true)"

# Operator-facing phrasing for each BLOCKING verdict kind. Every kind names its
# own REMEDY, because the remedies are genuinely different and the whole point
# of the per-unit channel is that the operator is told what to do about THIS
# unit: byte-drift is reconciled by editing a directive (after deciding which
# side is stale), a drop-in is removed with `systemctl --user edit`, and a
# vanished template is a repo problem, not a host one.
#
# The kinds come from VERDICT_KINDS in check_orchestrator_unit_parity.py, and a
# cross-artifact test asserts every member has an arm here.
#
# ARM ORDER IS PRECEDENCE, most-blocking first: the two kinds that mean "the
# file itself is unusable" (vanished, unreadable) outrank a content finding.
# EVERY combined arm must precede both of its single arms, or a unit with both
# findings is reported as having only one — an incomplete remedy, which is
# worse than none because it looks like progress.
#
# The checker's control flow admits exactly two combinations, and both have an
# arm: `drift,override` (a compared unit that differs AND is drop-in'd) and
# `override,unreadable` (the drop-in check runs BEFORE the read, so an
# undecodable unit can still be known to carry one). `vanished` and `absent`
# take a `continue` before anything else is consulted, and `clean` asserts the
# absence of every other finding, so none of the three can combine at all.
#
# Defined here rather than near the top of this script — see the harness
# constraint at the parity-script assignment above.
_orch_skip_reason() {
  case ",$1," in
    *,vanished,*)
      printf '%s' "there is no committed copy to install FROM (see the [vanished] report)" ;;
    # The SECOND producible combination, and it needs its own arm for the same
    # reason drift+override does: the checker marks `override` before it ever
    # attempts the read, so an undecodable unit that also carries a drop-in
    # renders `override,unreadable`. Falling through to the single unreadable
    # arm below would send the operator to fix the file encoding, re-run, and
    # be skipped again for a drop-in nobody mentioned. Alternated for the same
    # contiguity reason as the drift+override arm.
    *,override,unreadable,*|*,override,*,unreadable,*)
      printf '%s' "a drop-in override AND a unit file that could not be read or decoded — BOTH must be resolved; inspect the drop-in with: systemctl --user cat $2" ;;
    *,unreadable,*)
      printf '%s' "its unit file exists but could not be read or decoded (see the [unreadable] report)" ;;
    # ALTERNATED, and the first branch is the one that actually fires today.
    # `*,drift,*,override,*` alone can NEVER match ",drift,override," — the
    # comma that `,drift,` consumes is the same one `,override,` needs — so it
    # silently lost every match to the drift-only arm below, telling an
    # operator with both problems about one of them. The kinds are adjacent in
    # VERDICT_KINDS, so contiguity holds today; the second branch keeps this
    # arm firing if a future kind is ever ordered between them.
    *,drift,override,*|*,drift,*,override,*)
      printf '%s' "byte-drift against the committed copy AND a drop-in override — BOTH must be resolved" ;;
    *,drift,*)
      printf '%s' "byte-drift against the committed copy — reconcile the directive, checking WHICH side is correct" ;;
    *,override,*)
      printf '%s' "a drop-in override; inspect with: systemctl --user cat $2" ;;
    *,unverified,*)
      printf '%s' "the parity gate returned no verdict for this unit — it did not run, or does not know it" ;;
    *)
      # LOUD, and self-describing down to the remedy. Reaching this arm means
      # the checker's vocabulary grew without this script following — so the
      # message names the offending kind VERBATIM and both files that have to
      # change, rather than degrading to a generic "skipped" that leaves the
      # operator diffing units looking for a problem the gate already named.
      printf '%s' "the gate reported verdict kind '$1', which this installer has no case arm for, so it declined to act rather than guess. FIX: add an arm to _orch_skip_reason in scripts/setup-host.sh for every kind in VERDICT_KINDS (scripts/check_orchestrator_unit_parity.py)" ;;
  esac
}

# The per-unit install decision — the gate the section header describes, here
# in code: each unit is judged on ITS OWN verdict.
#
# `clean` and `absent` are the two install-eligible kinds, and the checker
# guarantees neither ever appears alongside a blocking one, so an exact match
# on the whole kind string is the right test.
_orch_install_units=()
for _unit in "${_orch_units[@]}"; do
  _kinds="${_orch_verdict[$_unit]:-unverified}"

  # Checked FIRST, ahead of the override, because it is physics rather than
  # judgement: DF_INSTALL_ORCH_UNITS=1 says "install over the reported
  # finding", and no amount of operator intent creates a source file. A bare
  # `cp` from a missing template exits non-zero and, under `set -euo pipefail`,
  # aborts this whole script — so every section BELOW would silently never run
  # because one unit's template was deleted. Measured: rc=1 at `cp: cannot
  # stat`, no daemon-reload, no enables.
  #
  # Deliberately a file test rather than a fourth `vanished` case arm: it must
  # also cover a unit the gate never reported on. `_orch_units` and the
  # checker's registry are kept in step by a test, but under the override a
  # unit with no verdict is installed on trust, and trust does not create a
  # file.
  #
  # It covers EXISTENCE only, and deliberately so — a source that exists can
  # still be uncopyable (mode 000, or an installed copy this user cannot
  # overwrite). That half is caught by the install loop's own failure handling
  # below rather than by a pre-flight test, because only the copy itself knows.
  if [ ! -f "$REPO_ROOT/scripts/$_unit" ]; then
    warn "SKIPPING $_unit — $(_orch_skip_reason vanished "$_unit"); its installed copy is UNCHANGED"
    continue
  fi

  if [ "${DF_INSTALL_ORCH_UNITS:-0}" = "1" ] || [ "$_kinds" = clean ] || [ "$_kinds" = absent ]; then
    _orch_install_units+=("$_unit")
  else
    warn "SKIPPING $_unit — $(_orch_skip_reason "$_kinds" "$_unit"); its installed copy is UNCHANGED"
  fi
done

if [ "${#_orch_install_units[@]}" -eq 0 ]; then
  warn "SKIPPING the orchestrator unit install — NO unit cleared the gate, so"
  warn "  nothing was copied, nothing enabled, the installed units are"
  warn "  UNCHANGED. Read the report above and reconcile whichever side is"
  warn "  wrong, then re-run. To install anyway:"
  warn "    DF_INSTALL_ORCH_UNITS=1 bash scripts/setup-host.sh"
else
  if [ "$_orch_install_blocked" -eq 1 ] && [ "${DF_INSTALL_ORCH_UNITS:-0}" = "1" ]; then
    warn "DF_INSTALL_ORCH_UNITS=1 — installing over the reported drift"
  fi

  # Each copy is INDIVIDUALLY fault-tolerant, for the same reason the source
  # existence test above exists: under `set -euo pipefail` one failing `cp`
  # aborts this script outright, taking daemon-reload, every enable, and every
  # LATER section of the host installer (jCodeMunch, Claude config, ...) with
  # it. A unit is a bad reason to abandon a host setup.
  #
  # The existence test cannot cover this: the checker's `unreadable` is raised
  # on (OSError, UnicodeDecodeError), and the OSError half is exactly a file
  # `cp` cannot touch. Measured with DF_INSTALL_ORCH_UNITS=1 against a mode-000
  # installed unit: `cp` exits 1 at "cannot create regular file: Permission
  # denied". (The UnicodeDecodeError half really is harmless — `cp` copies
  # bytes and never decodes.)
  #
  # A unit whose copy FAILED is dropped from _orch_installed_units, so it is
  # neither enabled (enabling it would act on bytes nobody managed to write)
  # nor counted in the success line below.
  _orch_installed_units=()
  for _unit in "${_orch_install_units[@]}"; do
    if cp "$REPO_ROOT/scripts/$_unit" "$UNIT_DIR/"; then
      _orch_installed_units+=("$_unit")
    else
      warn "FAILED to install $_unit — its installed copy is UNCHANGED; check permissions on $REPO_ROOT/scripts/$_unit and $UNIT_DIR/$_unit"
    fi
  done

  # ONCE, between the copies and the enables: systemd must not be asked to
  # enable a unit it has not re-read, and re-running it per unit would only
  # repeat work the first call already did.
  systemctl --user daemon-reload

  # Enable exactly what was INSTALLED — _orch_installed_units, never
  # _orch_units. A unit whose install the gate declined, or whose copy failed,
  # must not be enabled either: enabling it acts on the very state the skip (or
  # the failure) left unwritten, and on a first run it would enable a unit that
  # is not on disk at all.
  #
  # Within that set the obligation is DERIVED from each unit's own [Install]
  # section, not from a hand-listed exception for the static watchdog service.
  # Two reasons, both load-bearing:
  #   - `systemctl enable` on a unit with no [Install] is an ERROR, not a
  #     no-op, so under `set -e` a hand-list that fell out of step with the
  #     units would abort the installer outright.
  #   - a hand-maintained exception list has to be edited for every future
  #     unit, and a unit nobody remembers to add is precisely the class of bug
  #     the array above exists to close.
  # This is the same rule tests/scripts/test_orchestrator_service_files.py's
  # _unit_has_install_section predicate expresses in Python.
  for _unit in "${_orch_installed_units[@]}"; do
    if grep -q '^\[Install\]' "$REPO_ROOT/scripts/$_unit"; then
      systemctl --user enable "$_unit"
    fi
  done

  # Reports what was ACTUALLY done, not what was attempted: with a per-unit
  # gate a partial install is now a normal outcome, and an unqualified success
  # line would read as "all nine" on a run that installed one. Counted from the
  # units that COPIED, so a failed `cp` is never reported as an install.
  if [ "${#_orch_installed_units[@]}" -eq 0 ]; then
    warn "NO orchestrator unit was installed — every copy that cleared the gate"
    warn "  FAILED; the installed units are UNCHANGED. See the FAILED lines above."
  else
    ok "orchestrator units + watchdog installed and enabled (${#_orch_installed_units[@]}/${#_orch_units[@]})"
  fi
fi

# ---------------------------------------------------------------------------
# 6. jCodeMunch — structured code retrieval for coding agents
# ---------------------------------------------------------------------------
info "Installing jCodeMunch (AST-based code indexing)"

# Global config (ignore patterns shared across all projects)
CODE_INDEX_DIR="$HOME/.code-index"
mkdir -p "$CODE_INDEX_DIR"

if [ ! -f "$CODE_INDEX_DIR/config.jsonc" ]; then
  cat > "$CODE_INDEX_DIR/config.jsonc" << 'JCEOF'
{
  "max_folder_files": 10000,
  "extra_ignore_patterns": [
    ".worktrees/",
    ".eval-worktrees/",
    "*-eval-worktrees/",
    ".claude/worktrees/",
    ".taskmaster/",
    ".playwright-mcp/",
    "node_modules/",
    "target/",
    "__pycache__/",
    ".venv/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".pytest_cache/",
    "*.png",
    "uv.lock",
    "Cargo.lock",
    "pnpm-lock.yaml",
    "package-lock.json"
  ],
  "staleness_days": 3
}
JCEOF
  ok "Global config written to $CODE_INDEX_DIR/config.jsonc"
else
  ok "Global config already exists"
fi

# Project-level config
if [ ! -f "$REPO_ROOT/.jcodemunch.jsonc" ]; then
  cat > "$REPO_ROOT/.jcodemunch.jsonc" << 'JCEOF'
{
  // dark-factory: Python monorepo (fused-memory, orchestrator, escalation, shared)
  "languages": ["python"],
  "max_folder_files": 5000,
  "disabled_tools": ["search_columns"],
  "staleness_days": 3
}
JCEOF
  ok "Project config written"
else
  ok "Project config already exists"
fi

# Add jcodemunch MCP to user-level Claude config (idempotent)
if command -v claude &>/dev/null; then
  # Matched in BASH, not through `| grep -q` — see falkordb_pings above for why
  # that pipeline can report an installed server as absent.
  # Here the cost is re-running `claude mcp add` on a server already registered.
  # The capture stays INSIDE the `command -v claude` guard: hoisting it would
  # run `claude mcp list` on hosts with no claude installed.
  _jcodemunch_mcp_out="$(claude mcp list --scope user 2>/dev/null)" || true
  if [[ "$_jcodemunch_mcp_out" == *jcodemunch* ]]; then
    ok "jcodemunch MCP already in user config"
  else
    claude mcp add --scope user jcodemunch -- uvx --python 3.12 jcodemunch-mcp
    ok "jcodemunch MCP added to user config"
  fi
fi

# Systemd watcher unit
sed \
  -e "s|__REPO_ROOT__|$REPO_ROOT|g" \
  -e "s|__UV_PATH__|$UV_PATH|g" \
  "$REPO_ROOT/scripts/jcodemunch-watcher.service.template" \
  > "$UNIT_DIR/jcodemunch-watcher.service"

systemctl --user daemon-reload
systemctl --user enable jcodemunch-watcher
systemctl --user restart jcodemunch-watcher
ok "jcodemunch-watcher unit installed and started"

# ---------------------------------------------------------------------------
# 7. Skim — context compression for coding agents
# ---------------------------------------------------------------------------
info "Installing skim (context compression)"

if command -v skim &>/dev/null; then
  ok "skim already installed ($(skim --version 2>/dev/null))"
else
  # Find cargo: may be on PATH, in ~/.cargo/bin, or only in a rustup toolchain
  CARGO=""
  if command -v cargo &>/dev/null; then
    CARGO="cargo"
  elif [ -x "$HOME/.cargo/bin/cargo" ]; then
    CARGO="$HOME/.cargo/bin/cargo"
  else
    # Fall back to rustup stable toolchain
    RUSTUP_CARGO="$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo"
    [ -x "$RUSTUP_CARGO" ] && CARGO="$RUSTUP_CARGO"
  fi

  if [ -n "$CARGO" ]; then
    $CARGO install rskim --quiet
    ok "skim installed via cargo ($CARGO)"
  else
    warn "cargo not found — install Rust (rustup.rs) then: cargo install rskim"
  fi
fi

# Install global Claude Code hook (idempotent — skim init checks existing state)
if command -v skim &>/dev/null && command -v claude &>/dev/null; then
  if [ -f "$HOME/.claude/hooks/skim-rewrite.sh" ]; then
    ok "skim hook already installed"
  else
    skim init --yes
    ok "skim hook installed for Claude Code"
  fi

  # The hook rewrites commands to bare `skim`, which must be on PATH for all
  # shell types (login, interactive, non-interactive bash -c).  ~/.cargo/bin
  # is only added by profile/bashrc sourcing — symlink into /usr/local/bin
  # so it's on the base OS PATH unconditionally.
  SKIM_BIN="$HOME/.cargo/bin/skim"
  if [ ! -e /usr/local/bin/skim ]; then
    if [ -x "$SKIM_BIN" ]; then
      sudo ln -s "$SKIM_BIN" /usr/local/bin/skim
      ok "symlinked skim → /usr/local/bin/skim"
    fi
  else
    ok "skim already on system PATH (/usr/local/bin/skim)"
  fi
fi

# Verify skim is on PATH for all shell types an agent session might use.
# The hook rewrites commands to bare `skim`, so it must be findable without
# inheriting a profile-enhanced PATH.  Non-login, non-interactive shells
# (gnome-terminal -- bash -c '...', systemd ExecStart=, asyncio subprocesses)
# only get the base OS PATH unless ~/.cargo/env is sourced outside an
# interactivity guard.
if command -v skim &>/dev/null; then
  info "Checking skim PATH visibility across shell types"

  BASE_PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

  # Login shell (sources ~/.profile → ~/.cargo/env)
  if env -i HOME="$HOME" TERM="$TERM" bash --login -c 'command -v skim' &>/dev/null; then
    ok "skim on PATH: login shell"
  else
    fail "skim NOT on PATH: login shell"
  fi

  # Interactive shell (sources ~/.bashrc — needs to pass interactivity guard)
  # stderr suppressed: bash -ic warns about missing terminal/job-control
  if env -i HOME="$HOME" TERM="$TERM" bash -ic 'command -v skim' >/dev/null 2>&1; then
    ok "skim on PATH: interactive shell"
  else
    fail "skim NOT on PATH: interactive shell"
  fi

  # Non-interactive, non-login shell with base OS PATH only.
  # This simulates: gnome-terminal -- bash -c '...' when the parent env
  # was not profile-initialised, or asyncio.create_subprocess_exec with a
  # stripped env, or a systemd unit without Environment=PATH additions.
  if env -i HOME="$HOME" PATH="$BASE_PATH" bash -c 'command -v skim' &>/dev/null; then
    ok "skim on PATH: non-login non-interactive shell (base PATH)"
  else
    fail "skim NOT on PATH: non-login non-interactive shell (base PATH)"
    warn "  Agents spawned without profile init will fail on skim-rewritten commands"
    warn "  Fix: sudo ln -s $HOME/.cargo/bin/skim /usr/local/bin/skim"
  fi
fi

# ---------------------------------------------------------------------------
# 8. Dashboard systemd units
# ---------------------------------------------------------------------------
info "Installing dashboard systemd units"

# Parity check, BEFORE the install below overwrites its evidence.
#
# THIS IS THE REAL GATE. The install that follows unconditionally re-renders
# dark-factory-dashboard.service and cp's both watchdog units, so any drift
# between the running system and the committed units is erased a few lines
# from here. Checking afterwards could only ever report a failed copy — never
# the case this checker was built around ("the installed watchdog is still the
# pre-incident inline-shell copy"), because by then it isn't.
#
# Warn-only: drift never aborts the install, and the install below is itself
# the remediation. What this buys is a RECORD — the operator sees what was
# silently stale on this host before it got fixed, which is the difference
# between "setup ran" and "setup corrected a supervision gap that had been
# open since April".
#
# Deliberately no --fix in the checker (see its module docstring): re-running
# this installer is the propagation path, and re-ARMING the watchdog timer
# belongs to task 3289.
#
# The gate distinguishes "ran and found drift" from "did not run at all", and
# the install proceeds either way. Exit code alone is NOT trusted, because 2 is
# overloaded three ways: the checker's "not yet installed" (benign), `python3`
# refusing to open a missing script file, and argparse rejecting an unknown
# flag. Renaming the checker or one of its flags would otherwise print a
# reassuring "installing below" on a host whose units are installed AND
# drifted — a gate reporting green because it never ran, which is the silent-
# drift failure the checker exists to catch, reproduced in its own wiring.
# So no status is believed unless the checker's own [dashboard_unit_parity] tag
# appears in the output it produced. That tag is on EVERY line it emits, which
# test_main_every_emitted_line_carries_the_log_tag pins, so its absence is
# conclusive rather than a heuristic.
#
# UNLIKE the orchestrator gate above, a bad verdict does NOT make the install
# opt-in, and must not be changed to: the install is itself the remediation,
# the checker's own report tells the operator to run this script, and the
# incident it guards has the INSTALLED side stale. See
# test_section_8_installs_even_when_the_gate_did_not_run for the full argument.
# What changes on a gate that did not run is only the EPISTEMICS — the operator
# is no longer told a check passed when none ran.
_dash_parity_script="$REPO_ROOT/scripts/check_dashboard_unit_parity.py"

if [ ! -f "$_dash_parity_script" ]; then
  fail "Dashboard parity gate missing: $_dash_parity_script"
  fail "  Not treating that as 'nothing to check' — it is 'nothing checked'."
  fail "  The install below still runs; it simply gathered no evidence first."
else
  # The `&& x=0 || x=$?` idiom is what keeps `set -e` from aborting here.
  _dash_parity_out="$(python3 "$_dash_parity_script" \
       --installed-dir "$UNIT_DIR" \
       --repo-root     "$REPO_ROOT" 2>&1)" && _dash_parity_exit=0 || _dash_parity_exit=$?
  printf '%s\n' "$_dash_parity_out"

  # Classified by the shared helper at the top of this file.
  _dash_parity_verdict="$(_parity_verdict "$_dash_parity_out" \
       "$_dash_parity_exit" '[dashboard_unit_parity]')"
  case "$_dash_parity_verdict" in
  unreported)
    fail "Dashboard parity gate produced no [dashboard_unit_parity] report"
    fail "  (status $_dash_parity_exit) — it did not run, so its status says"
    fail "  nothing about this host. Check the script path and its flags."
    fail "  The install below still runs; it simply gathered no evidence first."
    ;;
  parity)
    ok "Dashboard units: already at parity with the committed copies"
    ;;
  absent)
    info "Dashboard units: not yet installed in $UNIT_DIR (installing below)"
    ;;
  finding | *)
    # A finding is "drift OR unverifiable" — it also covers a vanished
    # committed unit and a drop-in override, which the checker deliberately
    # words apart so the operator is not sent hunting for a directive diff that
    # does not exist. Naming only DRIFT here would collapse that distinction
    # back. (`*` is folded in for the reason given at the orchestrator gate.)
    warn "Dashboard units: pre-existing drift or unverifiable state — see the"
    warn "  [dashboard_unit_parity] report above. The install below propagates"
    warn "  the committed units; a drop-in override needs manual removal."
    ;;
  esac
fi

# THE RENDER. No longer `sed ... > "$UNIT_DIR/<unit>"`, and the difference is
# not stylistic (task 4793).
#
# scripts/dashboard.service.template declares
# `Environment=DASHBOARD_KNOWN_PROJECT_ROOTS=__REPO_ROOT__`, which renders to a
# SINGLE root. Further aggregation roots are host-LOCAL settings, added to the
# installed unit and deliberately not committed — the committed unit's own
# comment says so, and nine of them were measured on this host on 2026-08-01. A
# truncating redirect destroyed eight of those nine on every re-run, and did it
# INVISIBLY: that variable is on the parity checker's DIVERGENCE_ALLOWLIST
# (compared by NAME, value blessed), so the post-install check in section 12
# reported parity afterwards — and the gate above tells the operator to run this
# script, so following the advice was what caused the loss.
#
# THE RENDERER OWNS THE DESTINATION rather than being redirected into it.
# `python3 render_dashboard_unit.py ... > "$UNIT_DIR/<unit>"` would be the same
# defect one level up: bash truncates the destination before python ever opens
# it, so the installed value would be gone before it could be read and the tool
# would preserve nothing while reporting success. --output is read FIRST as the
# installed copy, then replaced atomically.
#
# AND THERE IS DELIBERATELY NO sed FALLBACK. Rendering "the old way" when the
# renderer is missing would reinstate the exact clobber it replaced, on the one
# path where nobody is left watching for it — the post-install gate cannot see
# this variable's value. A missing renderer therefore leaves the unit ALONE and
# says so, which is the recoverable direction: stale but intact.
#
# KEEP IN STEP WITH SECTION 4's `_fm_render_script=` BLOCK (task 4796), which is
# the same construct for fused-memory.service: same anchor shape, same
# `_<x>_rendered=0` flag, same three-way, same "daemon-reload always, enable on
# the unit EXISTING, the rest on the flag" gate split. Deliberately two copies
# rather than one helper — every message differs and a helper would have to
# signal WHICH failure occurred back through an exit code — so a change to the
# control flow or the failure modes here belongs in BOTH sites. Section 4's
# extra `restart` gate has no counterpart here on purpose: the dashboard is
# started by hand, never restarted by this script.
_dash_render_script="$REPO_ROOT/scripts/render_dashboard_unit.py"

# Set to 1 only by the branch that actually rendered. The section's closing line
# and the enable below are worded off it rather than printed unconditionally:
# `fail` here is a printf, not an exit, so without this every degraded path
# still reached a green "Dashboard units installed" — a line asserting exactly
# what the FAIL lines above it had just denied.
_dash_rendered=0

if [ ! -f "$_dash_render_script" ]; then
  fail "Dashboard unit renderer missing: $_dash_render_script"
  fail "  NOT rendering it the old way — a plain template render would strip"
  fail "  this host's local DASHBOARD_KNOWN_PROJECT_ROOTS entries, and the"
  fail "  post-install parity check cannot see that variable's value."
  fail "  $UNIT_DIR/dark-factory-dashboard.service is left AS-IS. The watchdog"
  fail "  units below still install."
elif python3 "$_dash_render_script" \
       --template  "$REPO_ROOT/scripts/dashboard.service.template" \
       --repo-root "$REPO_ROOT" \
       --uv-path   "$UV_PATH" \
       --output    "$UNIT_DIR/dark-factory-dashboard.service"; then
  _dash_rendered=1
  ok "Dashboard service unit rendered (host-local Environment= values preserved — see the [dashboard_unit_render] lines above)"
else
  # The renderer RAN and refused. Without this branch the `elif` chain simply
  # falls through with status 0 and says nothing about the unit that did not get
  # written — the same reports-green-because-it-never-ran failure the parity
  # gate above exists to remove, one construct over. Leaving `_dash_rendered` at
  # 0 is the other half: it is what keeps the section's closing line from
  # claiming the install happened.
  fail "Dashboard service unit render FAILED — see the [dashboard_unit_render]"
  fail "  report above for which step refused and why."
  fail "  $UNIT_DIR/dark-factory-dashboard.service was left UNTOUCHED: this"
  fail "  host's local Environment= values (DASHBOARD_KNOWN_PROJECT_ROOTS)"
  fail "  survived, and the unit is at worst STALE — which the pre-install"
  fail "  parity gate reports on the next run. Re-run this script once the"
  fail "  cause is fixed. The watchdog units below still install: one"
  fail "  un-renderable service unit must not take its supervision with it."
fi

# Watchdog service + timer (no templating needed — no repo-specific paths)
cp "$REPO_ROOT/dashboard/dark-factory-dashboard-watchdog.service" "$UNIT_DIR/"
cp "$REPO_ROOT/dashboard/dark-factory-dashboard-watchdog.timer" "$UNIT_DIR/"

systemctl --user daemon-reload
# GUARDED on the unit existing, not on `_dash_rendered`: a failed render on a
# host that already HAS the unit must still leave it enabled (stale but
# supervised is the recoverable direction this whole construct chooses). The
# combination the guard exists for is a BARE host plus a failed render — there
# `systemctl --user enable` on a unit that does not exist is a non-zero exit,
# and under this file's `set -e` that aborts the entire installer before the
# watchdog TIMER is enabled and before every later section. The two branches
# above promise "the watchdog units below still install"; this is what makes
# that promise true rather than true-only-when-a-unit-was-already-there.
if [ -f "$UNIT_DIR/dark-factory-dashboard.service" ]; then
  systemctl --user enable dark-factory-dashboard
else
  fail "dark-factory-dashboard NOT enabled: no unit file in $UNIT_DIR."
  fail "  The render above did not happen and this host had no previous copy,"
  fail "  so there is nothing to enable. The watchdog timer below still is."
fi
systemctl --user enable dark-factory-dashboard-watchdog.timer
if [ "$_dash_rendered" = "1" ]; then
  ok "Dashboard units installed (start manually when ready: systemctl --user start dark-factory-dashboard)"
else
  warn "Dashboard watchdog units installed; the dashboard SERVICE unit was NOT"
  warn "  rendered — see the FAIL lines above. Whatever copy this host already"
  warn "  had is untouched and still enabled; a bare host has none."
fi

# ---------------------------------------------------------------------------
# 9. Claude Code skill symlinks
# ---------------------------------------------------------------------------
info "Creating Claude Code skill symlinks"

COMMANDS_DIR="$HOME/.claude/commands"
mkdir -p "$COMMANDS_DIR"

declare -A SKILLS=(
  ["orchestrate.md"]="$REPO_ROOT/skills/orchestrate/SKILL.md"
  ["orchestrate-references"]="$REPO_ROOT/skills/orchestrate/references"
  ["reflect.md"]="$REPO_ROOT/skills/reflect/SKILL.md"
  ["unblock.md"]="$REPO_ROOT/skills/unblock/SKILL.md"
  ["unblock-low-risk.md"]="$REPO_ROOT/skills/unblock-low-risk/SKILL.md"
  ["review.md"]="$REPO_ROOT/skills/review/SKILL.md"
  ["review-references"]="$REPO_ROOT/skills/review/references"
  ["review-briefing.md"]="$REPO_ROOT/skills/review-briefing/SKILL.md"
  ["review-briefing-references"]="$REPO_ROOT/skills/review-briefing/references"
  ["escalation-watcher.md"]="$REPO_ROOT/skills/escalation-watcher/SKILL.md"
  ["recon-escalation-watcher.md"]="$REPO_ROOT/skills/recon-escalation-watcher/SKILL.md"
  ["merge-queue.md"]="$REPO_ROOT/skills/merge-queue/SKILL.md"
  ["merge-queue-references"]="$REPO_ROOT/skills/merge-queue/references"
  ["spawn.md"]="$REPO_ROOT/skills/spawn/SKILL.md"
  ["study.md"]="$REPO_ROOT/skills/study/SKILL.md"
  ["do.md"]="$REPO_ROOT/skills/do/SKILL.md"
  ["census.md"]="$REPO_ROOT/skills/census/SKILL.md"
  ["warm.md"]="$REPO_ROOT/skills/warm/SKILL.md"
)
# Deliberately NOT wired: escalation-watcher-auto and unblock-auto are
# sub-agent-only skills loaded programmatically by the orchestrator/watcher —
# they are never invoked as slash commands by an operator.

for name in "${!SKILLS[@]}"; do
  target="${SKILLS[$name]}"
  link="$COMMANDS_DIR/$name"
  if [ -e "$target" ] || [ -d "$target" ]; then
    ln -sfn "$target" "$link"
    ok "$name -> $(basename "$target")"
  else
    warn "Skipping $name — target does not exist: $target"
  fi
done

# Directory-form skills (the newer Claude Code Skill mechanism): these three
# are wired as whole-directory symlinks under ~/.claude/skills/<name> so their
# references/ and scripts/ travel with them (convention documented in
# skills/prd/references/project-overlay.md).
SKILLS_DIR="$HOME/.claude/skills"
mkdir -p "$SKILLS_DIR"
for name in factory-init prd hotspot-survey; do
  target="$REPO_ROOT/skills/$name"
  link="$SKILLS_DIR/$name"
  if [ -d "$target" ]; then
    ln -sfn "$target" "$link"
    ok "skills/$name -> ~/.claude/skills/$name"
  else
    warn "Skipping skills/$name — target does not exist: $target"
  fi
done

# ---------------------------------------------------------------------------
# 10. Git hooks
# ---------------------------------------------------------------------------
info "Setting up git hooks"

if [ -x "$REPO_ROOT/hooks/setup.sh" ]; then
  (cd "$REPO_ROOT" && bash hooks/setup.sh)
  ok "Git hooks configured"
else
  warn "hooks/setup.sh not found or not executable"
fi

# ---------------------------------------------------------------------------
# 11. Manual steps reminder
# ---------------------------------------------------------------------------
info "Manual steps (if migrating from another host)"
echo ""
echo "  On the SOURCE host, run:"
echo "    bash ~/src/dark-factory/scripts/export-data.sh"
echo ""
echo "  This exports fused-memory data, pushes all branches to remote,"
echo "  and prints rsync commands for transferring repos + data + credentials."
echo ""
echo "  On THIS host, after rsync completes:"
echo "    bash $REPO_ROOT/scripts/import-data.sh ~/dark-factory-export"
echo ""

# ---------------------------------------------------------------------------
# 12. Health checks
# ---------------------------------------------------------------------------
info "Health checks"

# FalkorDB
# The same probe the section-2 wait loop runs, so it is the same function and
# not a second copy of it — see falkordb_pings for why the verdict is read from
# the reply rather than from `| grep -q`.
if falkordb_pings; then
  ok "FalkorDB: PONG"
else
  fail "FalkorDB: not responding"
fi

# Qdrant
if curl -sf http://localhost:6333/readyz &>/dev/null; then
  col_count=$(curl -s http://localhost:6333/collections | jq '.result.collections | length' 2>/dev/null || echo "?")
  ok "Qdrant: ready ($col_count collections)"
else
  fail "Qdrant: not responding"
fi

# Fused-memory
if curl -sf http://localhost:8002/mcp 2>/dev/null; then
  ok "Fused-memory: healthy"
elif systemctl --user is-active fused-memory &>/dev/null; then
  warn "Fused-memory: unit active but health check failed (may still be starting)"
else
  warn "Fused-memory: not running (check: journalctl --user -u fused-memory)"
fi

# Fused-memory unit parity check — guard host-invariant safety switches
# (Environment=MEM0_TELEMETRY=false, WatchdogSec=120).  Warn-only: drift does
# not abort the install; re-run with --fix to correct in-place without clobbering
# host-specific lines (e.g. extra DASHBOARD_KNOWN_PROJECT_ROOTS entries).
#
# POST-INSTALL HEALTH CHECK, not a gate on anything: the fused-memory unit is
# rendered and installed back in section 4, and nothing installs after this
# block, so there is no action here to make conditional. What it owes the
# operator is an honest verdict.
#
# The exit code alone is NOT trusted, because 2 is overloaded three ways: the
# checker's "not installed on this host" (benign), `python3` refusing to open a
# missing script file, and argparse rejecting an unknown flag. Renaming the
# checker or one of its flags would therefore print a reassuring "skipping
# parity check" — a check reporting green because it never ran, which is the
# silent-drift failure the checker exists to catch, reproduced one level up in
# its own wiring.
#
# So no status is believed unless the checker's own [fused_memory_unit_parity]
# tag appears in the output it produced. That tag is on EVERY line it emits,
# which test_main_every_emitted_line_carries_the_log_tag pins, so its absence is
# conclusive rather than a heuristic. The other half — that all three real exit
# paths emit it and NEITHER collision source does — is pinned by
# test_gate_tag_appears_on_every_real_exit_path_and_neither_collision, both in
# tests/scripts/test_check_fused_memory_unit_parity.py.
_fm_parity_script="$REPO_ROOT/scripts/check_fused_memory_unit_parity.py"

if [ ! -f "$_fm_parity_script" ]; then
  fail "Fused-memory parity check missing: $_fm_parity_script"
  fail "  Not treating that as 'nothing to check' — it is 'nothing checked'."
else
  # 2>&1 is load-bearing: the [skip] line and the drift trailer go to stderr.
  # The `&& x=0 || x=$?` idiom is what keeps `set -e` from aborting here.
  _fm_parity_out="$(python3 "$_fm_parity_script" \
       --installed "$UNIT_DIR/fused-memory.service" \
       --template  "$REPO_ROOT/scripts/fused-memory.service.template" 2>&1)" \
       && _fm_parity_exit=0 || _fm_parity_exit=$?
  printf '%s\n' "$_fm_parity_out"

  # Classified by the shared helper at the top of this file.
  _fm_parity_verdict="$(_parity_verdict "$_fm_parity_out" \
       "$_fm_parity_exit" '[fused_memory_unit_parity]')"
  case "$_fm_parity_verdict" in
  unreported)
    fail "Fused-memory parity check produced no recognizable report"
    fail "  (status $_fm_parity_exit) — it did not run, so its status says"
    fail "  nothing about this host. Check the script path and its flags."
    ;;
  parity)
    ok "Fused-memory unit: parity with template (all safety directives present)"
    ;;
  absent)
    warn "Fused-memory unit: not installed at $UNIT_DIR/fused-memory.service (skipping parity check)"
    ;;
  finding | *)
    # `*` folded in for the reason given at the orchestrator gate.
    warn "Fused-memory unit: DRIFT detected — run: python3 $_fm_parity_script --fix"
    ;;
  esac
fi

# Dashboard unit parity — POST-INSTALL SANITY CHECK ONLY.
#
# The gate that can actually observe drift runs in section 8, BEFORE the units
# are re-rendered and copied; see the long comment there. By this point the
# installer has already overwritten every installed copy, so a mismatch here
# does not mean "the host drifted" — it means the install itself did not take:
# a failed write, a template that no longer renders to the committed unit, or a
# drop-in override that survives reinstallation because setup-host.sh does not
# touch <unit>.d/ directories.
#
# Warn-only, like the pre-install check.
#
# Same exit-2 overloading as the section-8 gate, and here the false green is
# the strongest of the three sites: this is the LAST word the operator reads
# about whether the install took, and its exit-2 wording is not merely
# reassuring but already a diagnosis ("section 8 did not run?"). A renamed
# checker would send them to investigate an install that in fact completed,
# while the thing that actually failed — the check itself — went unreported.
# So the script path is guarded and no status is believed without the
# checker's own [dashboard_unit_parity] tag.
#
# Distinct variable names from section 8's on purpose: both blocks share one
# shell scope, so under `set -u` a stale _dash_parity_out/_exit from section 8
# would still be readable here and a check that never ran could silently
# inherit the earlier block's verdict.
_dash_post_parity_script="$REPO_ROOT/scripts/check_dashboard_unit_parity.py"

if [ ! -f "$_dash_post_parity_script" ]; then
  fail "Dashboard post-install check missing: $_dash_post_parity_script"
  fail "  Not treating that as 'nothing to check' — it is 'nothing checked'."
  fail "  The install above is therefore UNVERIFIED, not verified."
else
  _dash_post_parity_out="$(python3 "$_dash_post_parity_script" \
       --installed-dir "$UNIT_DIR" \
       --repo-root     "$REPO_ROOT" 2>&1)" \
       && _dash_post_parity_exit=0 || _dash_post_parity_exit=$?
  printf '%s\n' "$_dash_post_parity_out"

  # Classified by the shared helper at the top of this file. Distinct verdict
  # variable from section 8's for the same reason the _out/_exit pair above is
  # distinct: both blocks share one shell scope.
  _dash_post_parity_verdict="$(_parity_verdict "$_dash_post_parity_out" \
       "$_dash_post_parity_exit" '[dashboard_unit_parity]')"
  case "$_dash_post_parity_verdict" in
  unreported)
    fail "Dashboard post-install check produced no [dashboard_unit_parity] report"
    fail "  (status $_dash_post_parity_exit) — it did not run, so its status says"
    fail "  nothing about this host. Check the script path and its flags."
    fail "  The install above is therefore UNVERIFIED, not verified."
    ;;
  parity)
    ok "Dashboard units: install verified (installed copies match the committed ones)"
    ;;
  absent)
    warn "Dashboard units: not installed in $UNIT_DIR (section 8 did not run?)"
    ;;
  finding | *)
    # `*` folded in for the reason given at the orchestrator gate.
    warn "Dashboard units: still not at parity AFTER installing — the install"
    warn "  did not take. See the [dashboard_unit_parity] report above; a"
    warn "  [override] drop-in survives reinstallation and must be removed by"
    warn "  hand (systemctl --user cat <unit> shows the merged result)."
    ;;
  esac
fi

# lms-arm@ unit: REPORT ONLY. setup-host.sh does not install the arms and must
# not start: they hold the whole GPU, and which arm runs when is an operator
# decision per the unit template's own rationale, never a side effect of
# running setup. That is also why exit 2 is INFO rather than a warning — on a
# host that never installed the arms it is the correct and expected state, and
# warning every run is how operators learn to ignore a gate.
#
# This exists because an installer-only check leaves an override invisible
# BETWEEN installs, which is when it bites: a landed worktree stays on disk for
# months, so a drop-in pinning WorkingDirectory at one can outlive its worktree
# with nothing reporting it.
#
# Warn-only, never fail: this task deliberately REFUSES to remove a drop-in
# (it can be load-bearing — task 3750), so a hard failure here would brick host
# bring-up on precisely the state we chose not to auto-fix. Contrast the
# orchestrator gate above, which may fail — that one guards an install this
# script actually performs.
# Same exit-2 overloading as the gates above: 2 is the checker's own benign
# "not installed", AND python3 refusing to open a renamed script, AND argparse
# rejecting a renamed flag. Only the first carries the checker's own
# [lms_unit_parity] tag, so the status is not read as a verdict without one.
# The tag is on every emitted line by construction (LOG_TAG in
# check_lms_unit_parity.py), so its absence is conclusive rather than a
# heuristic. Warn-only here too: a gate that did not run must be loud, but it
# still must not brick bring-up.
_lms_parity_script="$REPO_ROOT/scripts/check_lms_unit_parity.py"

if [ ! -f "$_lms_parity_script" ] \
   || [ ! -f "$REPO_ROOT/scripts/local-model-serving/lms-arm@.service" ]; then
  # A checkout without the checker is not a host with a drop-in. Reporting it
  # as one is the same credibility leak as warning on exit 2.
  info "lms-arm@ unit: parity checker not present in this checkout — skipped"
else
  # 2>&1 keeps a report written to stderr out of the operator's blind spot.
  # The `&& x=0 || x=$?` idiom is what keeps `set -e` from aborting here.
  _lms_parity_out="$(python3 "$_lms_parity_script" \
       --installed-dir "$UNIT_DIR" \
       --repo-root     "$REPO_ROOT" 2>&1)" && _lms_parity_exit=0 || _lms_parity_exit=$?
  printf '%s\n' "$_lms_parity_out"

  # Classified by the shared helper at the top of this file. Note this gate
  # answers `unreported` with warn(), not fail(): it is warn-only by charter
  # (test_setup_host_lms_parity_gate.py::test_the_lms_gate_is_warn_only), which
  # is exactly the per-site severity difference the helper does not collapse.
  _lms_parity_verdict="$(_parity_verdict "$_lms_parity_out" \
       "$_lms_parity_exit" '[lms_unit_parity]')"
  case "$_lms_parity_verdict" in
  unreported)
    warn "lms-arm@ unit: parity gate produced no [lms_unit_parity] report"
    warn "  (status $_lms_parity_exit) — it did not run, so its status says"
    warn "  nothing about this host. Check the script path and its flags."
    ;;
  parity)
    ok "lms-arm@ unit: parity with the committed template (effective configuration verified)"
    ;;
  absent)
    info "lms-arm@ unit: not installed on this host (install with scripts/local-model-serving/install-lms-units.sh)"
    ;;
  finding | *)
    # A finding is "drift OR unverifiable" — it also covers a drop-in override
    # and an effective configuration that disagrees, which the checker
    # deliberately words apart so the operator is not sent hunting for a
    # directive diff that does not exist. Naming only DRIFT here would collapse
    # that distinction. (`*` is folded in for the reason given at the
    # orchestrator gate.)
    warn "lms-arm@ unit: drift, a drop-in override, or an unverifiable effective"
    warn "  configuration — see the [lms_unit_parity] report above."
    warn "  A drop-in SURVIVES reinstallation and is NOT removed automatically."
    warn "  Inspect:  systemctl --user cat lms-arm@<arm>.service"
    warn "  Remove:   scripts/remove-lms-arm-worktree-dropin.sh (checks its"
    warn "            safety preconditions first)"
    ;;
  esac
fi

# jCodeMunch watcher
if systemctl --user is-active jcodemunch-watcher &>/dev/null; then
  ok "jCodeMunch watcher: running"
else
  warn "jCodeMunch watcher: not running (check: journalctl --user -u jcodemunch-watcher)"
fi

echo ""
info "Setup complete"
