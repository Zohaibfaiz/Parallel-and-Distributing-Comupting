#!/usr/bin/env bash
# Start the GPU worker daemon (Linux/macOS).  Usage: ./scripts/run_server.sh [extra daemon args]
cd "$(dirname "$0")/.." || exit 1
export OFFLOAD_TOKEN="${OFFLOAD_TOKEN:-offload-secret}"
exec python3 server/daemon.py --host 0.0.0.0 --port 5050 "$@"
