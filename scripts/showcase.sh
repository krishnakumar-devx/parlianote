#!/usr/bin/env bash
# Showcase days: run Parlianote on this laptop for a booth screen, with AI answers off and the Mac kept awake.
#
# On the day:    scripts/showcase.sh                          (Ctrl+C stops it)
# Booth screen:  open http://localhost:8510/?kiosk=1           (the idle tour, and it resets itself between visitors)
#
# Everything runs on this laptop; nothing is published online. Keep the charger in and the lid open: a closed lid still
# puts the Mac to sleep.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PORT:-8510}"   # its own port, so it never clashes with a copy you are developing on 8501
export AI_ANSWERS=0     # never offer AI-written answers on a public screen: anyone at the booth could spend your API credits

[ -d app/data ] || { echo "app/data is missing: see 'Quick start' in README.md"; exit 1; }

caffeinate -dimsu -w $$ &   # keep the Mac (and its display) awake for as long as this script runs

.venv/bin/python -m streamlit run app/app.py --server.port "$PORT" --server.headless true &
APP=$!
trap 'kill "$APP" 2>/dev/null || true' EXIT
trap 'exit 130' INT TERM  # Ctrl+C: without this trap, bash would keep waiting for the app below, which ignores Ctrl+C

echo "Starting Parlianote on http://localhost:$PORT ..."
for _ in $(seq 1 90); do
  curl -sf "http://localhost:$PORT/_stcore/health" >/dev/null && break
  sleep 1
done
curl -sf "http://localhost:$PORT/_stcore/health" >/dev/null || { echo "The app did not start; see the messages above."; exit 1; }

echo "Ready. Booth screen: http://localhost:$PORT/?kiosk=1   (Ctrl+C to stop)"
wait "$APP"
