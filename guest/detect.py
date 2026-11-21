#!/usr/bin/env python3
"""detect — decide, where this container is running now, whether it has a usable NVIDIA GPU.

Standard library only; runs at boot and on demand.

The container's configuration bind-mounts the host's NVIDIA devices with
`optional,create=file`. On a node with no GPU the mounts are skipped but the
targets still appear, as EMPTY REGULAR FILES (seen on node-a,
2026-11-14). So a path existing means nothing: a device counts only if it is a
character device, and the GPU counts only if the driver library can talk to it.

Modes: "gpu" when every device is real and NVML reports at least one GPU;
"cpu" otherwise, with the reason. A GPU that is present but unusable is "cpu"
with a reason that names the fault, never "gpu".
"""
import ctypes
import json
import os
import re
import shlex
import stat
import sys
import time

DEVICES = ("/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm")
DRIVER_PROC = "/proc/driver/nvidia/version"
NVML_LIB = "libnvidia-ml.so.1"
ENV_PATH = "/etc/mujoco-cauldron/mode.env"
STATE_PATH = "/run/mujoco-cauldron/state.json"
MAX_GPUS = 8
MAX_NAME = 96
# Starting values for the number of parallel simulations, by free GPU memory. Not measured limits.
GPU_ENVS_BY_FREE_MIB = ((6000, 1024), (2000, 256), (0, 64))


class Probes:
    """Everything detect() reads from the machine, so tests can stand in for it."""

    def is_chardev(self, path):
        """True only for a character device; an empty placeholder file is not one."""
        assert isinstance(path, str) and path
        try:
            return stat.S_ISCHR(os.stat(path).st_mode)
        except OSError:
            return False

    def exists(self, path):
        """True when something is at the path, whatever it is."""
        assert isinstance(path, str) and path
        return os.path.lexists(path)

    def read_text(self, path):
        """File contents, or None."""
        assert isinstance(path, str) and path
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                return handle.read(65536)
        except OSError:
            return None

    def cpu_threads(self):
        """CPU threads this process may use."""
        assert hasattr(os, "cpu_count")
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            return os.cpu_count() or 1

    def query_gpus(self):
        """(gpus, error). gpus is a list of {name, mem_total_mib, mem_free_mib}; error is None, 'missing' or a message."""
        assert NVML_LIB.endswith(".so.1")
        try:
            lib = ctypes.CDLL(NVML_LIB)
        except OSError:
            return ([], "missing")
        return _nvml_gpus(lib)


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_gpus(lib):
    """Ask NVML for the GPUs. Any failure is returned as an error string, never raised."""
    assert lib is not None
    lib.nvmlErrorString.restype = ctypes.c_char_p
    rc = lib.nvmlInit_v2()
    if rc != 0:
        return ([], "nvml init failed: %s" % (lib.nvmlErrorString(rc) or b"?").decode("ascii", "replace"))
    gpus = []
    try:
        count = ctypes.c_uint(0)
        rc = lib.nvmlDeviceGetCount_v2(ctypes.byref(count))
        if rc != 0:
            return ([], "nvml device count failed (%d)" % rc)
        for index in range(min(count.value, MAX_GPUS)):
            handle, name, mem = ctypes.c_void_p(), ctypes.create_string_buffer(MAX_NAME), _NvmlMemory()
            if lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(handle)) != 0:
                return ([], "nvml handle %d failed" % index)
            if lib.nvmlDeviceGetName(handle, name, MAX_NAME) != 0 or lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem)) != 0:
                return ([], "nvml query %d failed" % index)
            gpus.append({"name": name.value.decode("ascii", "replace"), "mem_total_mib": mem.total >> 20, "mem_free_mib": mem.free >> 20})
    finally:
        lib.nvmlShutdown()
    return (gpus, None)


def host_driver_version(text):
    """The NVIDIA kernel module version from /proc/driver/nvidia/version, or None."""
    assert text is None or isinstance(text, str)
    match = re.search(r"Kernel Module\s+(\d+(?:\.\d+)+)", text or "")
    return match.group(1) if match else None


def mem_limit_mib(probes):
    """This container's memory limit in MiB, or None when there is none to read."""
    assert probes is not None
    raw = (probes.read_text("/sys/fs/cgroup/memory.max") or "").strip()
    if raw.isdigit():
        return int(raw) >> 20
    match = re.search(r"MemTotal:\s+(\d+) kB", probes.read_text("/proc/meminfo") or "")
    return int(match.group(1)) >> 10 if match else None


def gpu_envs(free_mib):
    """A starting number of parallel simulations for this much free GPU memory."""
    assert isinstance(free_mib, int) and free_mib >= 0
    for floor, envs in GPU_ENVS_BY_FREE_MIB:
        if free_mib >= floor:
            return envs
    return GPU_ENVS_BY_FREE_MIB[-1][1]


def _gpu_verdict(probes):
    """(mode, reason, gpus): the GPU half of the decision."""
    assert probes is not None
    real = [d for d in DEVICES if probes.is_chardev(d)]
    placeholders = [d for d in DEVICES if d not in real and probes.exists(d)]
    if not real:
        why = "no NVIDIA devices on this host"
        return ("cpu", why + (" (placeholder files only)" if placeholders else ""), [])
    if len(real) < len(DEVICES):
        return ("cpu", "NVIDIA devices incomplete: missing %s" % ", ".join(d for d in DEVICES if d not in real), [])
    gpus, err = probes.query_gpus()
    if err == "missing":
        return ("cpu", "GPU present but the driver library %s is not installed; run: cauldron gpu-setup" % NVML_LIB, [])
    if err:
        return ("cpu", "GPU present but unusable: %s; run: cauldron gpu-setup" % err, [])
    if not gpus:
        return ("cpu", "driver library loaded but reports no GPU", [])
    return ("gpu", "NVIDIA GPU usable", gpus)


def detect(probes=None, now=None):
    """The decision and everything it rests on, as one flat record."""
    probes = probes or Probes()
    mode, reason, gpus = _gpu_verdict(probes)
    assert mode in ("gpu", "cpu") and reason
    threads = max(1, int(probes.cpu_threads()))
    best = max(gpus, key=lambda g: g["mem_free_mib"]) if gpus else None
    cpuinfo = probes.read_text("/proc/cpuinfo") or ""
    model = re.search(r"model name\s*:\s*(.+)", cpuinfo)
    return {
        "mode": mode,
        "reason": reason,
        "device": "cuda:0" if mode == "gpu" else "cpu",
        "num_envs": gpu_envs(best["mem_free_mib"]) if best else threads,
        "mujoco_gl": "egl" if mode == "gpu" else "osmesa",
        "gpu_name": best["name"] if best else "",
        "gpu_mem_total_mib": best["mem_total_mib"] if best else 0,
        "gpu_mem_free_mib": best["mem_free_mib"] if best else 0,
        "gpu_count": len(gpus),
        "host_driver": host_driver_version(probes.read_text(DRIVER_PROC)) or "",
        "cpu_threads": threads,
        "cpu_model": model.group(1).strip() if model else "",
        "mem_limit_mib": mem_limit_mib(probes) or 0,
        "detected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now if now is not None else time.time())),
    }


def env_text(record):
    """The record as shell assignments, every value quoted mechanically."""
    assert isinstance(record, dict) and record.get("mode") in ("gpu", "cpu")
    lines = ["# Written by /opt/mujoco-cauldron/bin/detect.py. Do not edit: it is rewritten at every boot."]
    for key in sorted(record):
        lines.append("MUJOCO_CAULDRON_%s=%s" % (key.upper(), shlex.quote(str(record[key]))))
    lines.append("MUJOCO_GL=%s" % shlex.quote(record["mujoco_gl"]))
    return "\n".join(lines) + "\n"


def _write_atomic(path, text):
    """Write a file so a reader never sees half of it."""
    assert isinstance(path, str) and isinstance(text, str)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def main(argv):
    """detect.py [--print]: decide, and unless --print, record the decision."""
    assert isinstance(argv, list)
    if any(a not in ("--print",) for a in argv[1:]):
        sys.stderr.write("usage: detect.py [--print]\n")
        return 2
    record = detect()
    if "--print" not in argv:
        _write_atomic(ENV_PATH, env_text(record))
        _write_atomic(STATE_PATH, json.dumps(record, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
