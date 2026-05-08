#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ $# -eq 0 ]]; then
  exec ./manta
fi

case "$1" in
  xvfb-run)
    "$@"
    exit $?
    ;;
  bash|sh|julia)
    exec "$@"
    ;;
  *)
    if [[ "$1" != -* ]] && command -v "$1" >/dev/null 2>&1; then
      exec "$@"
    fi
    exec ./manta "$@"
    ;;
esac
