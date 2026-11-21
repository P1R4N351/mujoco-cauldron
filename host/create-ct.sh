#!/usr/bin/env bash
# Create the mujoco-cauldron container (id VMID below) on node-a. Run on admin-node (it reaches every node by cluster ssh).
# Idempotent up to the create: refuses if the id or the address is already taken.
set -euo pipefail

readonly VMID=930
readonly NODE=node-a
readonly IP=192.0.2.32
readonly TEMPLATE=debian-12-standard_12.12-1_amd64.tar.zst
readonly SNIPPET=nvidia-cgroup-guard.sh
readonly SSH_OPTS=(-o ConnectTimeout=10 -o BatchMode=yes)

# This script is read from stdin (bash -s), so a remote command must not inherit stdin: it would eat the rest of the script.
on_node() { ssh -n "${SSH_OPTS[@]}" "$NODE" "$@"; }
on_node_stdin() { ssh "${SSH_OPTS[@]}" "$NODE" "$@"; }   # only where stdin is a heredoc or a pipe

echo "== preconditions"
if ls /etc/pve/nodes/*/lxc/${VMID}.conf /etc/pve/nodes/*/qemu-server/${VMID}.conf >/dev/null 2>&1; then
  echo "REFUSED: guest ${VMID} already exists"; exit 3
fi
if grep -l "ip=${IP}/" /etc/pve/nodes/*/lxc/*.conf /etc/pve/nodes/*/qemu-server/*.conf >/dev/null 2>&1; then
  echo "REFUSED: ${IP} is already assigned to a guest"; exit 3
fi
if ping -c1 -W1 "$IP" >/dev/null 2>&1; then
  echo "REFUSED: ${IP} answers ping"; exit 3
fi
[[ -f "/var/lib/vz/template/cache/${TEMPLATE}" ]] || { echo "REFUSED: template missing on this node"; exit 3; }

echo "== template onto ${NODE} (copied from this node, not downloaded)"
want_t="$(sha256sum "/var/lib/vz/template/cache/${TEMPLATE}" | cut -d' ' -f1)"
have_t="$(on_node "sha256sum /var/lib/vz/template/cache/${TEMPLATE} 2>/dev/null | cut -d' ' -f1" || true)"
if [[ "$have_t" != "$want_t" ]]; then
  scp -q "${SSH_OPTS[@]}" "/var/lib/vz/template/cache/${TEMPLATE}" "${NODE}:/var/lib/vz/template/cache/${TEMPLATE}" < /dev/null
  have_t="$(on_node "sha256sum /var/lib/vz/template/cache/${TEMPLATE} | cut -d' ' -f1")"
fi
[[ "$have_t" == "$want_t" ]] || { echo "FAILED: template checksum differs after copy"; exit 4; }
echo "template sha256 ${want_t:0:16}… present on ${NODE}"

echo "== GPU guard hookscript onto ${NODE} (byte-identical copy of gpu-node's)"
want_s="$(ssh -n "${SSH_OPTS[@]}" gpu-node "sha256sum /var/lib/vz/snippets/${SNIPPET} | cut -d' ' -f1")"
[[ ${#want_s} -eq 64 ]] || { echo "FAILED: could not read the hookscript on gpu-node"; exit 4; }
have_s="$(on_node "sha256sum /var/lib/vz/snippets/${SNIPPET} 2>/dev/null | cut -d' ' -f1" || true)"
if [[ "$have_s" != "$want_s" ]]; then
  ssh -n "${SSH_OPTS[@]}" gpu-node "cat /var/lib/vz/snippets/${SNIPPET}" | on_node_stdin "mkdir -p /var/lib/vz/snippets && cat > /var/lib/vz/snippets/${SNIPPET}.new && chmod 0755 /var/lib/vz/snippets/${SNIPPET}.new && mv /var/lib/vz/snippets/${SNIPPET}.new /var/lib/vz/snippets/${SNIPPET}"
  have_s="$(on_node "sha256sum /var/lib/vz/snippets/${SNIPPET} | cut -d' ' -f1")"
fi
[[ "$have_s" == "$want_s" ]] || { echo "FAILED: hookscript checksum differs after copy"; exit 4; }
echo "hookscript sha256 ${want_s:0:16}… present on ${NODE}"

echo "== create"
desc="mujoco-cauldron: MuJoCo simulation and training for Vector and other robots. Detects at boot whether it has an NVIDIA GPU (gpu-node) or not and configures itself for GPU or CPU; see /opt/mujoco-cauldron. Built 2026-11-14."
on_node pct create "$VMID" "local:vztmpl/${TEMPLATE}" \
  --hostname mujoco-cauldron --ostype debian --arch amd64 --unprivileged 1 --features nesting=1 \
  --cores 8 --memory 6144 --swap 2048 --rootfs local-zfs:32 \
  --net0 "name=eth0,bridge=vmbr1,ip=${IP}/24,gw=192.0.2.1" --nameserver 1.1.1.1 \
  --onboot 1
# The description is text, so it is quoted mechanically for the remote shell, never by hand.
on_node "pct set ${VMID} --description $(printf '%q' "$desc")"

echo "== GPU passthrough lines: every mount is optional, so the container starts on a node with no GPU"
on_node_stdin "cat >> /etc/pve/lxc/${VMID}.conf" <<'EOF'
hookscript: local:snippets/nvidia-cgroup-guard.sh
lxc.cgroup2.devices.allow: c 195:* rwm
lxc.mount.entry: /dev/nvidia0 dev/nvidia0 none bind,optional,create=file
lxc.mount.entry: /dev/nvidiactl dev/nvidiactl none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm dev/nvidia-uvm none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-uvm-tools dev/nvidia-uvm-tools none bind,optional,create=file
lxc.mount.entry: /dev/nvidia-modeset dev/nvidia-modeset none bind,optional,create=file
EOF

echo "== start"
on_node pct start "$VMID"
on_node "pct status ${VMID}; pct config ${VMID} | grep -v '^description' | sed -E 's/hwaddr=[0-9A-Fa-f:]+/hwaddr=<mac>/'"
echo "== inside"
on_node "pct exec ${VMID} -- sh -c 'hostname; . /etc/os-release; echo \$PRETTY_NAME; nproc; ip -br a show eth0 | cut -c1-50; ls -la /dev/nvidia* 2>&1 | head -6; ping -c1 -W2 1.1.1.1 >/dev/null 2>&1 && echo net-ok || echo NO-NET'"
