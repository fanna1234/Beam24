#!/usr/bin/env bash
set -euo pipefail
beam24_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$beam24_root/artifact/reproduce.sh" "$@"
