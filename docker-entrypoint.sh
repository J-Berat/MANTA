#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ $# -eq 0 ]]; then
  exec ./carta
fi

case "$1" in
  bash|sh|julia|xvfb-run)
    exec "$@"
    ;;
  *)
    if [[ "$1" != -* ]] && command -v "$1" >/dev/null 2>&1; then
      exec "$@"
    fi
    exec ./carta "$@"
    ;;
esac
