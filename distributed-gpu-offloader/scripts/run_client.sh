#!/usr/bin/env bash
# Launch the desktop GUI client (Linux/macOS)
cd "$(dirname "$0")/.." || exit 1
exec python3 client/app.py "$@"
