#!/usr/bin/env bash
# Installs the Vulkan loader and Mesa's software adapter (lavapipe), the adapter CI draws on, with
# no recommended packages. Each step has a bounded wait and retries, and the whole install is tried
# up to three times: a stalled or refusing mirror otherwise fails or holds the job.
set -euo pipefail
apt=(sudo -n apt-get -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 -o Acquire::Retries=3)
for attempt in 1 2 3; do
  if timeout 300 "${apt[@]}" update -qq &&
    timeout 300 "${apt[@]}" install -y -qq --no-install-recommends libvulkan1 mesa-vulkan-drivers; then
    ls /usr/share/vulkan/icd.d/
    exit 0
  fi
  echo "software_vulkan: attempt $attempt failed"
  sleep $((attempt * 20))
done
echo "FAIL: libvulkan1 and mesa-vulkan-drivers could not be installed"
exit 1
