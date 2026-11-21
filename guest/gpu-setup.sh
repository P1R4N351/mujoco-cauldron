#!/usr/bin/env bash
# gpu-setup.sh — install the NVIDIA user-space libraries that match the host's kernel driver.
# Run inside the container as root (cauldron gpu-setup).
#
# A container shares the host's kernel driver, and the libraries inside must be the SAME version or
# the GPU is unusable ("driver/library version mismatch"). The version is read from the host's driver
# when the container is on a GPU host; on a host with no GPU, name it: gpu-setup.sh --version 610.43.02
# The packages come from NVIDIA's own Debian 12 repository, the source the workhorse container uses.
set -euo pipefail

readonly REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64"
readonly KEYRING=/usr/share/keyrings/cuda-archive-keyring.gpg
readonly LIST=/etc/apt/sources.list.d/cuda-debian12.list
readonly PACKAGES=(libnvidia-ml1 libcuda1)

log() { printf '[gpu-setup %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

host_version() {
  sed -n 's/.*Kernel Module[[:space:]]\+\([0-9][0-9.]*\).*/\1/p' /proc/driver/nvidia/version 2>/dev/null | head -n 1
}

main() {
  local version=""
  [[ "$(id -u)" == "0" ]] || die "run as root"
  if [[ "${1:-}" == "--version" && $# -eq 2 ]]; then
    version="$2"
  elif [[ $# -eq 0 ]]; then
    version="$(host_version)"
    [[ -n "$version" ]] || die "no NVIDIA driver on this host; on a host with no GPU pass --version <driver version>"
  else
    die "usage: gpu-setup.sh [--version <driver version>]"
  fi
  [[ "$version" =~ ^[0-9]+(\.[0-9]+)+$ ]] || die "not a driver version: ${version}"
  export DEBIAN_FRONTEND=noninteractive

  if [[ ! -s "$KEYRING" ]]; then
    log "adding NVIDIA's repository key"
    curl -fsSL --max-time 60 "${REPO_URL}/3bf863cc.pub" | gpg --dearmor -o "${KEYRING}.new"
    [[ -s "${KEYRING}.new" ]] || die "could not fetch the repository key"
    mv "${KEYRING}.new" "$KEYRING"
  fi
  printf 'deb [signed-by=%s] %s/ /\n' "$KEYRING" "$REPO_URL" > "$LIST"
  apt-get update -qq

  local pins=() pkg
  for pkg in "${PACKAGES[@]}"; do pins+=("${pkg}=${version}-1"); done
  log "installing ${pins[*]}"
  apt-get install -y -qq --no-install-recommends --allow-downgrades "${pins[@]}" >/dev/null

  for pkg in "${PACKAGES[@]}"; do
    [[ "$(dpkg-query -W -f='${Version}' "$pkg" 2>/dev/null)" == "${version}-1" ]] || die "${pkg} is not at ${version}-1 after install"
  done
  log "libraries at ${version}; deciding GPU or CPU again"
  python3 /opt/mujoco-cauldron/bin/detect.py | sed -n 's/^ *"\(mode\|reason\)": /\1: /p'
}

main "$@"
