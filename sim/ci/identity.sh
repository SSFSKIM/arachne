#!/usr/bin/env bash
# Builds the probe twice, for baseline x86-64 and for x86-64-v3 (AVX2 and FMA available to the
# compiler), and requires: byte-identical output from the two builds (the results may not
# depend on the vector instructions the compiler was allowed); the same output from a second,
# fresh process; and the output committed for this architecture.
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
  RUSTFLAGS="-C target-cpu=$level" cargo build --release --locked --quiet --example probe \
    --target-dir "target/identity-$level"
  "target/identity-$level/release/examples/probe" > "$out/$level"
done
"target/identity-x86-64/release/examples/probe" > "$out/again"
cat "$out/x86-64"

status=0
if diff "$out/x86-64" "$out/x86-64-v3"; then
  echo "identity: $(wc -l < "$out/x86-64") outputs byte-identical across x86-64 and x86-64-v3"
else
  echo "identity: the x86-64 and x86-64-v3 builds differ"
  status=1
fi
if diff "$out/x86-64" "$out/again"; then
  echo "identity: a second process prints the same"
else
  echo "identity: a second process printed something else"
  status=1
fi
if diff ci/probe-x86_64.txt "$out/x86-64"; then
  echo "probe: $(awk '/^world 2d/ {print $3}' "$out/x86-64") 2d," \
    "$(awk '/^world 3d/ {print $3}' "$out/x86-64") 3d, as committed for x86_64"
else
  echo "probe: the output differs from ci/probe-x86_64.txt (above, committed first)"
  status=1
fi
exit $status
