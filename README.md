---
name: mujoco-cauldron
description: CT 930, a container for MuJoCo simulation and policy training that works on any cluster node. At every boot it detects whether the host gives it a usable NVIDIA GPU and configures itself for GPU or CPU.
---

# mujoco-cauldron

Authored-by: CPCS. Built 2026-11-14 at the operator's request, for training robot
policies in simulation (Vector first) and carrying them to the real robot.

| | |
|---|---|
| Guest | CT 930 `mujoco-cauldron`, Debian 12, unprivileged |
| Created on | `node-a` (8 threads, no GPU) |
| Address | `192.0.2.32/24` on `vmbr1` |
| Size | 8 cores, 6 GB memory, 2 GB swap, 32 GB root on `local-zfs` |
| Reach it | from `admin-node`: `ssh node-a 'pct exec 930 -- bash -s' < script` |

## What it does at boot

`mujoco-cauldron-detect.service` runs `/opt/mujoco-cauldron/bin/detect.py`,
which decides **gpu** or **cpu** and records the decision in
`/etc/mujoco-cauldron/mode.env` and `/run/mujoco-cauldron/state.json`.

- **gpu**: `/dev/nvidia0`, `/dev/nvidiactl` and `/dev/nvidia-uvm` are real
  character devices and the driver library reports at least one GPU.
- **cpu**: anything else, with the reason. A GPU that is present but unusable
  (library missing, or library and host driver at different versions) is `cpu`
  with a reason naming the fault. It is never reported as `gpu`.

The decision sets the device (`cuda:0` or `cpu`), the renderer (`egl` or
`osmesa`) and a starting number of parallel simulations: the thread count on
CPU, 64, 256 or 1024 by free GPU memory on GPU. Those GPU numbers are
starting values, not measured limits.

One thing that is easy to get wrong: the container's GPU mounts are
`optional,create=file`. On a node with no GPU the mounts are skipped but the
paths still appear, as empty regular files. A path existing means nothing.

## Use

```bash
cauldron status       # the decision and what is installed
cauldron check        # decide again, then run the self-test in that mode
cauldron python x.py  # the simulation environment's Python, with the decision in its environment
cauldron gpu-setup    # on a GPU host: install the NVIDIA libraries matching its driver
```

A program reads `MUJOCO_CAULDRON_MODE`, `MUJOCO_CAULDRON_DEVICE`,
`MUJOCO_CAULDRON_NUM_ENVS` and `MUJOCO_GL` from its environment.

The environment is `/opt/mujoco-cauldron/venv` (Python 3.11): `mjlab`, which
brings MuJoCo, MuJoCo Warp, Warp, PyTorch and `rsl-rl`; plus `gymnasium`,
`onnx` and `onnxruntime`. It is one environment for both modes.

## Moving it to the GPU

Only `gpu-node` has an NVIDIA GPU (RTX 3060 Ti, 8 GB), shared with the model
`ollama` holds in CT 902. Its `local-zfs` had 4.5 GB free on 2026-11-14 and
its `tank` thin pool 68 GB, already above 92%. To run there:

```bash
pct migrate 930 gpu-node --restart --target-storage tank   # from the node that holds it
```

then, inside, `cauldron gpu-setup` if the library version differs from the
host driver, and `cauldron check`. Check `tank`'s free space first: a full
thin pool damages every volume on it.

The container's configuration carries the hookscript
`local:snippets/nvidia-cgroup-guard.sh`. A node without that file cannot start
the container. It is on `gpu-node` and `node-a`; copy it to any
other node before migrating there (`host/create-ct.sh` shows how).

## Rebuild

`host/create-ct.sh`, run on `admin-node`, creates the container.
Then copy `guest/` into it and run `bash bootstrap.sh`; `bootstrap.sh --check`
reports what is in place without changing anything.

## Evidence (2026-11-14)

- `tests/test_detect.py`: 14 tests. `tests/mutate.py`: 12 of 12 injected
  faults caught, each by the test for that property; a wording-only change
  survived.
- The detector run read-only (`--print`) inside CT 902 on `gpu-node`, against
  the real GPU: `mode gpu`, RTX 3060 Ti, 2473 of 8192 MiB free, host driver
  610.43.02. It wrote nothing there.
- On `node-a` the five device paths were present as 0-byte regular
  files, and the detector decided `cpu`.
- `cauldron check` in the container on `node-a`: `SELFTEST PASS in
  cpu mode`. MuJoCo 3.11.0 loaded and stepped a small model, two runs gave
  identical states; 198,199 steps/s on one process and 1,148,000 steps/s on
  four. PyTorch 2.15.0 computed on the CPU. Warp 1.19.0 initialised with the
  CPU device only. `mjlab` 1.6.0 and `rsl-rl-lib` 5.4.2 are installed.
- After `pct reboot 930` the boot unit ran by itself: active, result success,
  decision recorded 5 seconds after boot.
- `p10-scan --fail-on warn --fail-on-unscannable skills/mujoco-cauldron`:
  errors 0, warns 0.

## Not done

- GPU mode has not been run in this container: it has never been on
  `gpu-node`. The GPU half of `cauldron check` (PyTorch and Warp on the GPU)
  is therefore untested.
- No HA rule and no replica. On a node failure it stays down until moved.
- No robot model or training task is installed yet. This is the place to run
  them.
- `selftest.py` uses at most four worker processes for about a second each.
  It is a function check, not a benchmark of the node.
