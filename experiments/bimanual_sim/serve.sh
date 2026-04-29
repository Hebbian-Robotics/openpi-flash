#!/bin/bash
# Lifecycle helper for the runner. Runs wherever the file lives — no hardcoded
# paths. Usage:
#   ./serve.sh start [scene]   # scene defaults to mobile_aloha_ur10e_server_swap
#   ./serve.sh stop
#   ./serve.sh status
#   ./serve.sh logs [N]
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PATH="$HOME/.local/bin:$PATH"

LOG="$SCRIPT_DIR/runner.log"
PIDFILE="$SCRIPT_DIR/runner.pid"

case "${1:-status}" in
  start)
    SCENE="${2:-mobile_aloha_ur10e_server_swap}"
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "already running (pid $(cat $PIDFILE))"; exit 0
    fi
    pkill -f 'runner.py' 2>/dev/null || true
    sleep 1
    nohup uv run python "$SCRIPT_DIR/runner.py" \
      --scene "$SCENE" --host 127.0.0.1 --port 8080 \
      > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    sleep 6
    if kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
      echo "started pid=$(cat $PIDFILE), scene=$SCENE, log=$LOG"
      tail -8 "$LOG"
    else
      echo "failed to start; see $LOG"; tail -20 "$LOG"; exit 1
    fi
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then kill "$(cat $PIDFILE)" 2>/dev/null || true; fi
    pkill -f 'runner.py' 2>/dev/null || true
    rm -f "$PIDFILE"
    echo stopped
    ;;
  status)
    if pgrep -af runner.py >/dev/null; then
      echo running:; pgrep -af runner.py
      ss -tlnp 2>/dev/null | grep :8080 || true
    else
      echo stopped
    fi
    ;;
  logs)
    tail -n "${2:-50}" "$LOG"
    ;;
  *)
    echo "usage: serve.sh {start [scene]|stop|status|logs [N]}"; exit 2
    ;;
esac
