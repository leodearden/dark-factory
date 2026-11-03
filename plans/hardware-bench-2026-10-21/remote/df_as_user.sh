#!/usr/bin/env bash
# df_as_user.sh <concurrency>...: the DF merge gate as an unprivileged user (run as
# root on a bench pod after phase 2). As root, ~70 DF tests built on permission
# fixtures fail fast, so a root gate does a different amount of work than the
# laptop's; this measures the same gate as the laptop's own user would run it.
# Same spec (spec-deselect.json), wrapper (eatmydata) and gate.sh as phase 2;
# TZ=Europe/London like the laptop. Results land in $BENCH/results/user-gates/.
set -euo pipefail

U=leo
UHOME=/home/$U
BENCH=/root/df-hwbench
VIEW=$UHOME/dfbench
DF_SHA=${DF_SHA:-71bc561086e5116c5ef65bc91d5b4e44a77ef211}
log() { echo "[df-user $(date -u +%T)] $*"; }

id -u "$U" >/dev/null 2>&1 || useradd -M -d "$UHOME" -s /bin/bash "$U"

log "user-level toolchains and caches for $U"
for d in .local/bin .local/share/uv .local/node .cache/uv .npm .gitconfig .ssh; do
	[[ -e /root/$d ]] || continue
	rm -rf "${UHOME:?}/$d"
	mkdir -p "$(dirname "$UHOME/$d")"
	cp -a "/root/$d" "$UHOME/$d"
done

chmod 755 /root
rm -rf "$VIEW"
mkdir -p "$VIEW/results"
ln -s "$BENCH/df-0" "$VIEW/df-0"
cp "$BENCH/spec-deselect.json" "$VIEW/"
git clone -q --bare "$BENCH/df.git" "$VIEW/df.git"
chown -R "$U:" "$UHOME"
log "clearing root's /tmp leftovers"
find /tmp -mindepth 1 -maxdepth 1 -user root ! -path /tmp/reify-agent-cargo-home -exec rm -rf {} +

for concurrency in "$@"; do
	log "$concurrency gate(s) as $U"
	runuser -u "$U" -- env -i HOME="$UHOME" USER="$U" LOGNAME="$U" LANG=C.UTF-8 TZ=Europe/London \
		PATH="$UHOME/.local/bin:$UHOME/.local/node/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
		BENCH="$VIEW" RESULTS="$VIEW/results" DF_SHA="$DF_SHA" \
		GATE_SPEC="$VIEW/spec-deselect.json" GATE_WRAPPER=eatmydata GATE_LABEL=user-eatmydata-deselect \
		bash "$BENCH/bin/gate.sh" "$concurrency"
done
mkdir -p "$BENCH/results/user-gates"
cp -a "$VIEW/results/." "$BENCH/results/user-gates/"
log "done"
