#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/artifact/scripts/runtime_env.sh"
case "$TARGET" in acoular|spib|locata) ;; *) echo "usage: $0 {acoular|spib|locata}" >&2; exit 2 ;; esac
DATA_ROOT="${BEAM24_DATA_ROOT:-$ROOT/work/datasets}"
mkdir -p "$DATA_ROOT"

md5_value() {
  if command -v md5sum >/dev/null; then md5sum "$1" | awk '{print $1}'; else md5 -q "$1"; fi
}

fetch() {
  local url="$1" output="$2" expected="${3:-}"
  if [[ ! -f "$output" ]]; then
    curl -L --fail --show-error --retry 3 --continue-at - --progress-bar "$url" -o "$output.part"
    actual="$(md5_value "$output.part")"
    [[ "$actual" == "$expected" ]] || { echo "checksum mismatch: $output.part" >&2; exit 6; }
    mv "$output.part" "$output"
  fi
  if [[ -n "$expected" ]]; then
    local actual
    actual="$(md5_value "$output")"
    [[ "$actual" == "$expected" ]] || { echo "checksum mismatch: $output" >&2; exit 6; }
  fi
}

case "$TARGET" in
  acoular)
    dir="$DATA_ROOT/acoular64"; mkdir -p "$dir"
    fetch "https://zenodo.org/api/records/5809069/files/three_sources.h5/content" "$dir/three_sources.h5" ca8efe2834a63dda373ee67ce6b0678f
    fetch "https://zenodo.org/api/records/5809069/files/array_64.xml/content" "$dir/array_64.xml" 42fa95ba46ddd940548e66e41b1737a4
    ;;
  spib)
    dir="$DATA_ROOT/spib"; mkdir -p "$dir"
    fetch "https://spib.linse.ufsc.br/data/array/A2601_1_10.zip" "$dir/A2601_1_10.zip" b05df6154890cc4fc64c044cc61a9f2c
    fetch "https://spib.linse.ufsc.br/data/array/SACLANT_sens.dat" "$dir/SACLANT_sens.dat" 126b4ae6e007030ffb510dd7d720f4bc
    [[ -d "$dir/A2601_1_10" ]] || unzip -q "$dir/A2601_1_10.zip" -d "$dir/A2601_1_10"
    "$PYTHON_BIN" "$ROOT/scripts/check_spib_extraction.py" "$dir/A2601_1_10.zip" "$dir/A2601_1_10"
    ;;
  locata)
    dir="$DATA_ROOT/locata"; mkdir -p "$dir"
    fetch "https://zenodo.org/api/records/3630471/files/dev.zip/content" "$dir/dev.zip" d5a5417c3f6b2ed0e43581dd06f504e6
    if [[ ! -d "$dir/subset/dev" ]]; then
      mkdir -p "$dir/subset"
      unzip -q "$dir/dev.zip" 'dev/task1/recording*/eigenmike/*' 'dev/task3/recording*/eigenmike/*' -d "$dir/subset"
    fi
    ;;
  *) echo "usage: $0 {acoular|spib|locata}" >&2; exit 2 ;;
esac

echo "$dir"
