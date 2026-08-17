#!/bin/bash
# Strategy Lab Launcher
# =====================
# Starts the Streamlit server and opens it in a browser window (app mode).

set -u

# 1. Project root = directory containing this script (portable, no hardcoded path)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Adjust PATH for macOS background processes (launched from Finder/Automator)
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

# PYTHONPATH workaround: Python 3.13 skips .pth files with the UF_HIDDEN flag
# (set by uv/hatchling). Setting PYTHONPATH makes strategy_lab importable anyway.
export PYTHONPATH="$PROJECT_ROOT/src"

PORT=8501
LOG_FILE="/tmp/streamlit_backtest.log"
PID_FILE="/tmp/streamlit_backtest.pid"

# 2. Shut down any existing server on the port
if [ -f "$PID_FILE" ]; then
    kill -9 "$(cat "$PID_FILE")" 2>/dev/null
    rm -f "$PID_FILE"
fi
lsof -ti :$PORT | xargs kill -9 2>/dev/null
pkill -f "streamlit run app/main.py" 2>/dev/null
sleep 1

# 3. Start Streamlit
# Prefer the project venv; fall back to `uv run` if it is not built yet.
if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PY=("$PROJECT_ROOT/.venv/bin/python" -m streamlit)
else
    PY=(uv run streamlit)
fi

nohup "${PY[@]}" run app/main.py \
    --server.port $PORT \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.address 127.0.0.1 > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"

# 4. Wait for the server to answer
for _ in {1..30}; do
    curl -s "http://127.0.0.1:$PORT" > /dev/null && break
    sleep 0.5
done

# 5. Open in Chrome (app mode) if available, otherwise the default browser
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
    "$CHROME" --app="http://127.0.0.1:$PORT" &
else
    open "http://127.0.0.1:$PORT" 2>/dev/null || xdg-open "http://127.0.0.1:$PORT"
fi
