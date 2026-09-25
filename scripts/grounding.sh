#!/usr/bin/env bash
# Start / check / stop the grounding service in the BACKGROUND.
#
# The service has to be its own process -- it lives in perception_env
# (Python 3.10, SAM 3 + DINOv2) while the skills run in the frankapy venv
# (Python 3.8) -- but it does NOT need its own terminal. Backgrounding it here
# means one terminal for the whole session.
#
#   source scripts/env.sh
#   scripts/grounding.sh start     # or restart / status / stop / log
#   python scripts/run_experiment.py --skill ...
#
# `start` is idempotent: if a healthy service is already up it says so and
# leaves it alone. Use `restart` after registering an exemplar -- the library
# is read once at startup, and a service that predates exemplars/ answers
# registered names by text prompting instead, which looks exactly like the
# object not being there (2026-09-25: a service left running since Sep 9
# reported `scoop -> no masks` on three cameras while exemplars/scoop.npz sat
# unread on disk).

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

URL="${GROUNDING_URL:-http://127.0.0.1:5005}"
LOG="$ROOT/diag_out/grounding_service.log"
PIDFILE="$ROOT/diag_out/grounding_service.pid"
PY="$ROOT/perception_env/bin/python"
ARGS=(--backend sam3 --model "$ROOT/weights/sam3.pt")

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
    # A service older than the exemplar library answers registered names by
    # text prompt, which reads as "object not found" rather than as staleness.
    if [ -d exemplars ] && [ -n "$pid" ]; then
        local newer
        newer="$(find exemplars -name '*.npz' -newermt "@$(stat -c %X "/proc/$pid" 2>/dev/null || echo 0)" 2>/dev/null | head -3)"
        if [ -n "$newer" ]; then
            echo "  WARNING: these exemplars are newer than the running service:"
            echo "$newer" | sed 's/^/    /'
            echo "  It has not loaded them. Run: scripts/grounding.sh restart"
        fi
    fi
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
