#!/usr/bin/env bash
# Start / check / stop the grounding service in the BACKGROUND.
#
# The service has to be its own process -- it lives in perception_env
# (Python 3.10, SAM 3) while the skills run in the frankapy venv
# (Python 3.8) -- but it does NOT need its own terminal. Backgrounding it here
# means one terminal for the whole session.
#
#   source scripts/env.sh
#   scripts/grounding.sh start     # or restart / status / stop / log
#   python scripts/run_experiment.py --skill ...
#
# `start` is idempotent: if a healthy service is already up it says so and
# leaves it alone. Exemplar matching (DINOv2 + a second SAM) is off: it
# stacked on top of SAM 3 and exhausted RAM on this host (2026-09-25). The
# printed scoop is the text prompt "white plastic tool". Pass --exemplars
# exemplars on the python command, not through this script, to turn it back on.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

URL="${GROUNDING_URL:-http://127.0.0.1:5005}"
LOG="$ROOT/diag_out/grounding_service.log"
PIDFILE="$ROOT/diag_out/grounding_service.pid"
PY="$ROOT/perception_env/bin/python"
ARGS=(--backend sam3 --model "$ROOT/weights/sam3.pt" --no-exemplars)

health() { curl -s --max-time 5 "$URL/health" 2>/dev/null; }

running_pid() {
    pgrep -f "grounding_service.py" | while read -r p; do
        [ "$(tr -d '\0' < "/proc/$p/cmdline" 2>/dev/null | grep -c python)" -gt 0 ] && echo "$p"
    done | head -1
}

status() {
    local h pid
    h="$(health)"; pid="$(running_pid)"
    if [ -z "$h" ]; then
        echo "grounding: NOT reachable at $URL${pid:+ (pid $pid is up but not answering)}"
        return 1
    fi
    echo "grounding: up at $URL  (pid ${pid:-?})"
    echo "  $h"
    return 0
}

start() {
    if health >/dev/null && [ -n "$(health)" ]; then
        echo "grounding: already up; leaving it alone"; status; return 0
    fi
    mkdir -p "$(dirname "$LOG")"
    echo "grounding: starting in the background -> $LOG"
    nohup "$PY" -u perception_service/grounding_service.py "${ARGS[@]}" >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 60); do
        sleep 2
        [ -n "$(health)" ] && { status; return 0; }
    done
    echo "grounding: did not come up within 120s; last log lines:"; tail -20 "$LOG"
    return 1
}

stop() {
    local pid; pid="$(running_pid)"
    [ -z "$pid" ] && { echo "grounding: not running"; return 0; }
    echo "grounding: stopping pid $pid"; kill "$pid"
    for _ in $(seq 1 15); do sleep 1; [ -z "$(running_pid)" ] && break; done
    [ -n "$(running_pid)" ] && { echo "  still up, sending KILL"; kill -9 "$(running_pid)"; }
    echo "grounding: stopped"
}

case "${1:-status}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    log)     tail -f "$LOG" ;;
    *) echo "usage: $0 {start|stop|restart|status|log}"; exit 2 ;;
esac
