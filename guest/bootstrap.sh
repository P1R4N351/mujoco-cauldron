#!/usr/bin/env bash
# bootstrap.sh — provision a fresh Debian 12 container as mujoco-cauldron.
# Run inside the container, as root, from the directory holding this file's siblings:
#   bash bootstrap.sh            install everything (idempotent; safe to run again)
#   bash bootstrap.sh --check    report what is and is not in place; changes nothing
set -euo pipefail

readonly BASE=/opt/mujoco-cauldron
readonly VENV="${BASE}/venv"
readonly SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly FILES=(detect.py cauldron selftest.py gpu-setup.sh)
readonly APT_PACKAGES=(python3 python3-venv ca-certificates curl gnupg git libegl1 libgl1 libosmesa6 libglfw3 procps)
readonly PIP_PACKAGES=(mjlab gymnasium onnx onnxruntime)

log() { printf '[bootstrap %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

check_only() {
  local missing=0 f
  for f in "${FILES[@]}"; do
    cmp -s "${SRC}/${f}" "${BASE}/bin/${f}" || { log "differs or missing: ${BASE}/bin/${f}"; missing=1; }
  done
  [[ -x "${VENV}/bin/python" ]] || { log "missing: ${VENV}"; missing=1; }
  [[ "$(readlink -f /usr/bin/cauldron 2>/dev/null)" == "${BASE}/bin/cauldron" ]] || { log "the cauldron command is not on the default PATH"; missing=1; }
  systemctl is-enabled --quiet mujoco-cauldron-detect.service 2>/dev/null || { log "boot detection is not enabled"; missing=1; }
  [[ -s /etc/mujoco-cauldron/mode.env ]] || { log "no recorded decision: /etc/mujoco-cauldron/mode.env"; missing=1; }
  (( missing == 0 )) && log "everything is in place"
  return "$missing"
}

install_system() {
  export DEBIAN_FRONTEND=noninteractive
  log "apt: ${APT_PACKAGES[*]}"
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends "${APT_PACKAGES[@]}" >/dev/null
}

install_files() {
  local f
  install -d -m 0755 "${BASE}/bin" /etc/mujoco-cauldron
  for f in "${FILES[@]}"; do
    [[ -f "${SRC}/${f}" ]] || die "missing beside bootstrap.sh: ${f}"
    install -m 0755 "${SRC}/${f}" "${BASE}/bin/${f}"
  done
  # /usr/bin, not /usr/local/bin: `pct exec` runs with a PATH that leaves /usr/local/bin out
  # (seen 2026-11-14: "cauldron: command not found" from the node).
  ln -sfn "${BASE}/bin/cauldron" /usr/bin/cauldron
  rm -f /usr/local/bin/cauldron
  install -m 0644 "${SRC}/mujoco-cauldron-detect.service" /etc/systemd/system/mujoco-cauldron-detect.service
  # Login shells get the decision too. The file is rewritten at every boot, so it is read, never copied.
  printf '%s\n' '[ -r /etc/mujoco-cauldron/mode.env ] && { set -a; . /etc/mujoco-cauldron/mode.env; set +a; }' \
    > /etc/profile.d/mujoco-cauldron.sh
  systemctl daemon-reload
  systemctl enable --quiet mujoco-cauldron-detect.service
}

install_python() {
  if [[ ! -x "${VENV}/bin/python" ]]; then
    log "creating ${VENV}"
    python3 -m venv "${VENV}"
  fi
  "${VENV}/bin/python" -m pip install --quiet --upgrade pip
  log "pip: ${PIP_PACKAGES[*]} (the first run downloads several GB)"
  "${VENV}/bin/python" -m pip install --quiet "${PIP_PACKAGES[@]}"
}

main() {
  [[ "$(id -u)" == "0" ]] || die "run as root"
  if [[ "${1:-}" == "--check" ]]; then check_only; return $?; fi
  [[ $# -eq 0 ]] || die "usage: bootstrap.sh [--check]"
  grep -q '^ID=debian' /etc/os-release || die "this script is written for Debian"
  install_system
  install_files
  install_python
  log "deciding GPU or CPU"
  python3 "${BASE}/bin/detect.py" >/dev/null
  check_only || die "bootstrap finished but something is not in place"
  log "done: run 'cauldron check'"
}

main "$@"
