#!/usr/bin/env bash
# Governor (as the trial's gov2.sh): freeze the run's unit whenever a production verify on THIS host
# is waiting at verify.sh's CPU-pressure gates; thaw it otherwise.  gov.sh <unit> <logfile>
UNIT="$1"; LOG="$2"
state=thawed
trap 'systemctl --user thaw "$UNIT" 2>/dev/null' EXIT
sleep 5
while systemctl --user -q is-active "$UNIT"; do
    if pgrep -f 'verify[.]sh (psi|compile)-gate' >/dev/null; then want=frozen; else want=thawed; fi
    if [ "$want" != "$state" ]; then
        if [ "$want" = frozen ]; then systemctl --user freeze "$UNIT"; else systemctl --user thaw "$UNIT"; fi
        state=$want
        printf '%s %s\n' "$(date +%s.%N)" "$state" >>"$LOG"
    fi
    sleep 10
done
