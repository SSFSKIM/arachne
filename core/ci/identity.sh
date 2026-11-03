#!/usr/bin/env bash
# Builds the identity example twice, for baseline x86-64 and for x86-64-v3 (AVX2 and FMA
# available to the compiler), runs both on the same inputs and requires byte-identical output: the
# core's results may not depend on the machine's vector instructions.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
for flag in avx2 fma bmi2; do
  if ! grep -qw "$flag" /proc/cpuinfo; then
    echo "identity: this runner lacks $flag, so it cannot run an x86-64-v3 build"
    exit 1
  fi
done

out=$(mktemp -d)
for level in x86-64 x86-64-v3; do
  RUSTFLAGS="-C target-cpu=$level" cargo run --release --locked --quiet --example identity \
    --target-dir "target/identity-$level" -- "$out/$level"
done
(cd "$out/x86-64" && sha256sum -- *)
if diff -r "$out/x86-64" "$out/x86-64-v3"; then
  echo "identity: $(ls "$out/x86-64" | wc -l) outputs byte-identical across x86-64 and x86-64-v3"
else
  echo "identity: the x86-64 and x86-64-v3 builds differ"
  exit 1
fi
