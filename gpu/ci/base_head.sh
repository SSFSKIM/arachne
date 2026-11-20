#!/usr/bin/env bash
# A change to when or where old scenes are drawn and written leaves them as they were: the scenes
# `cpu_bytes.sh` draws are rendered, and drawn through a session's `frame`, at a base commit and
# at the head, each into fresh roots, once into an empty cache and once again into the cache that
# run left. The raw frames, the partial-movie keys, the plays served from the cache and the files a
# render publishes must be equal at the two commits, in both passes.
#
#   bash gpu/ci/base_head.sh <base commit>
set -euo pipefail
base=${1:-}
if [ -z "$base" ]; then
  echo "usage: base_head.sh <base commit> (a pull request's run passes its base)" >&2
  exit 2
fi
root=$(git rev-parse --show-toplevel)
cd "$root"
command -v uv >/dev/null || pipx install uv==0.12.3
head=$(git rev-parse HEAD)
base=$(git rev-parse "$base^{commit}")
work=$(mktemp -d)
git worktree add --quiet --detach "$work/base" "$base"
trap 'git worktree remove --force "$work/base" >/dev/null 2>&1 || true' EXIT
echo "base: $base"
echo "head: $head"

for arm in base head; do
  tree=$root
  [ "$arm" = base ] && tree=$work/base
  (cd "$tree" && UV_PROJECT_ENVIRONMENT="$work/$arm-env" uv sync --all-extras --locked --quiet)
done
# Each arm draws with its own interpreter from its own tree: the head's script and scene list,
# the arm's engine.
# Every pass hashes strings under one seed: a partial-movie key that reads an unordered container
# otherwise takes another value in another process, at either commit (ThreeDCamera's key moves so).
export PYTHONHASHSEED=0
for arm in base head; do
  tree=$root
  [ "$arm" = base ] && tree=$work/base
  for pass in cold warm; do
    echo "--- $arm, $pass"
    (cd "$tree" && "$work/$arm-env/bin/python" "$root/gpu/ci/base_head_draw.py" \
      "$tree" "$work/$arm-roots" "$work/$arm-$pass.json")
  done
done
python3 gpu/ci/base_head_compare.py "$work" "$base" "$head"
