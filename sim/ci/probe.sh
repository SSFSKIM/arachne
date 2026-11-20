#!/usr/bin/env bash
# sim-probe.yml's job on one system: builds the probe with the pinned toolchain, writes its
# output to the file named, and prints the probe's import list for the record.
#   probe.sh <system> <output file, relative to sim/>
set -euo pipefail
system=$1
output=$2
source "$(dirname "$0")/toolchain.sh"
python=$(command -v python3 || command -v python)
cargo build --release --locked --example probe
binary=target/release/examples/probe
if [[ -f "$binary.exe" ]]; then
  binary=$binary.exe
  rustup component add llvm-tools --toolchain "$channel"
fi
"$binary" > "$output"
cat "$output"
"$python" ci/imports.py --list "$binary" | tee "imports-$system.txt"
