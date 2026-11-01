#!/usr/bin/env bash
# The CI environment beyond the packages .github/ci-kit.yml declares:
# - Cairo, built at the version the control frames were rendered with;
# - on macOS, TeX through TinyTeX with the packages .github/manimdependency.json lists.
# Runs on every gate job; idempotent.
set -euo pipefail

if [ ! -d third_party/cairo ]; then
  python3 .github/scripts/ci_build_cairo.py
fi
python3 .github/scripts/ci_build_cairo.py --set-env-vars

if [ "$(uname)" = Darwin ]; then
  tinytex="$HOME/.TinyTeX"
  if [ ! -d "$tinytex" ]; then
    curl -fsSL --retry 5 -o "$RUNNER_TEMP/TinyTeX.tar.xz" \
      "https://github.com/rstudio/tinytex-releases/releases/download/daily/TinyTeX-1-darwin.tar.xz"
    tar -xJf "$RUNNER_TEMP/TinyTeX.tar.xz" -C "$HOME"
    mv "$HOME/TinyTeX" "$tinytex"
    packages=$(python3 -c "import json; print(' '.join(json.load(open('.github/manimdependency.json'))['macos']['tinytex']))")
    "$tinytex/bin/universal-darwin/tlmgr" update --self
    # shellcheck disable=SC2086
    "$tinytex/bin/universal-darwin/tlmgr" install $packages
  fi
  echo "$tinytex/bin/universal-darwin" >> "$GITHUB_PATH"
fi
