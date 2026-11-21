#!/usr/bin/env python3
"""Tests for the GPU-or-CPU decision. Standard library only; no GPU, no container needed.

Run:  python3 -m unittest discover -s skills/mujoco-cauldron/tests -v

Every case stands in for a machine through FakeProbes. The cases named for hosts
copy what was observed there on 2026-11-14: node-a (no GPU, placeholder
files where the devices would be) and gpu_node (RTX 3060 Ti, driver 610.43.02,
2.8 GB free beside another container's model).
"""
import importlib.util
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GUEST = Path(__file__).resolve().parent.parent / "guest"
SPEC = importlib.util.spec_from_file_location("cauldron_detect", GUEST / "detect.py")
detect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detect)

DRIVER_TEXT = "NVRM version: NVIDIA UNIX x86_64 Kernel Module  610.43.02  Tue May 19 11:18:18 UTC 2026\nGCC version:\n"
GPU_NODE_GPU = [{"name": "NVIDIA GeForce RTX 3060 Ti", "mem_total_mib": 8192, "mem_free_mib": 2835}]


class FakeProbes(detect.Probes):
    """A machine described by what its device paths are and what the driver library says."""

    def __init__(self, chardevs=(), placeholders=(), gpus=None, gpu_error=None, driver=None, threads=8, mem_max="6442450944"):
        self.chardevs, self.placeholders = set(chardevs), set(placeholders)
        self.gpus, self.gpu_error, self.driver = gpus or [], gpu_error, driver
        self.threads, self.mem_max, self.gpu_queries = threads, mem_max, 0

    def is_chardev(self, path):
        return path in self.chardevs

    def exists(self, path):
        return path in self.chardevs or path in self.placeholders

    def read_text(self, path):
        return {detect.DRIVER_PROC: self.driver, "/sys/fs/cgroup/memory.max": self.mem_max,
                "/proc/cpuinfo": "processor\t: 0\nmodel name\t: AMD Ryzen Embedded R2544 with Radeon Graphics\n",
                "/proc/meminfo": "MemTotal:       16000000 kB\n"}.get(path)

    def cpu_threads(self):
        return self.threads

    def query_gpus(self):
        self.gpu_queries += 1
        return (self.gpus, self.gpu_error)


def branch5():
    return FakeProbes(placeholders=detect.DEVICES)


def gpu_node(**changes):
    args = {"chardevs": detect.DEVICES, "gpus": GPU_NODE_GPU, "driver": DRIVER_TEXT, "threads": 24}
    args.update(changes)
    return FakeProbes(**args)


class Decision(unittest.TestCase):
    def test_no_gpu_host_with_placeholder_files_is_cpu(self):
        probes = branch5()
        rec = detect.detect(probes, now=0)
        self.assertEqual((rec["mode"], rec["device"], rec["num_envs"], rec["mujoco_gl"]), ("cpu", "cpu", 8, "osmesa"))
        self.assertIn("placeholder files only", rec["reason"])
        self.assertEqual(probes.gpu_queries, 0, "the driver library must not be asked when there is no device")
        self.assertEqual((rec["gpu_name"], rec["gpu_count"], rec["host_driver"]), ("", 0, ""))

    def test_bare_host_is_cpu(self):
        rec = detect.detect(FakeProbes(threads=32), now=0)
        self.assertEqual((rec["mode"], rec["num_envs"]), ("cpu", 32))
        self.assertNotIn("placeholder", rec["reason"])

    def test_gpu_host_is_gpu(self):
        rec = detect.detect(gpu_node(), now=0)
        self.assertEqual((rec["mode"], rec["device"], rec["mujoco_gl"]), ("gpu", "cuda:0", "egl"))
        self.assertEqual((rec["gpu_name"], rec["gpu_mem_free_mib"], rec["gpu_mem_total_mib"], rec["host_driver"]),
                         ("NVIDIA GeForce RTX 3060 Ti", 2835, 8192, "610.43.02"))
        self.assertEqual(rec["num_envs"], 256)

    def test_one_missing_device_is_cpu(self):
        for missing in detect.DEVICES:
            probes = gpu_node(chardevs=[d for d in detect.DEVICES if d != missing])
            rec = detect.detect(probes, now=0)
            self.assertEqual(rec["mode"], "cpu", missing)
            self.assertIn(missing, rec["reason"])
            self.assertEqual(probes.gpu_queries, 0)

    def test_devices_without_the_driver_library_is_cpu_and_says_how_to_fix_it(self):
        rec = detect.detect(gpu_node(gpus=[], gpu_error="missing"), now=0)
        self.assertEqual(rec["mode"], "cpu")
        self.assertIn("is not installed", rec["reason"])
        self.assertIn("cauldron gpu-setup", rec["reason"])
        self.assertEqual(rec["host_driver"], "610.43.02")

    def test_an_unusable_gpu_is_cpu_never_gpu(self):
        for error in ("nvml init failed: Driver/library version mismatch", "nvml device count failed (9)"):
            rec = detect.detect(gpu_node(gpus=GPU_NODE_GPU, gpu_error=error), now=0)
            self.assertEqual((rec["mode"], rec["device"], rec["gpu_name"]), ("cpu", "cpu", ""), error)
            self.assertIn(error, rec["reason"])

    def test_a_library_that_reports_no_gpu_is_cpu(self):
        rec = detect.detect(gpu_node(gpus=[]), now=0)
        self.assertEqual(rec["mode"], "cpu")

    def test_the_gpu_with_most_free_memory_is_reported(self):
        gpus = [{"name": "small", "mem_total_mib": 8192, "mem_free_mib": 500},
                {"name": "roomy", "mem_total_mib": 24576, "mem_free_mib": 20000}]
        rec = detect.detect(gpu_node(gpus=gpus), now=0)
        self.assertEqual((rec["gpu_name"], rec["gpu_count"], rec["num_envs"]), ("roomy", 2, 1024))

    def test_parallel_count_follows_free_gpu_memory(self):
        self.assertEqual([detect.gpu_envs(m) for m in (0, 1999, 2000, 5999, 6000, 24000)], [64, 64, 256, 256, 1024, 1024])

    def test_memory_limit(self):
        self.assertEqual(detect.detect(branch5(), now=0)["mem_limit_mib"], 6144)
        self.assertEqual(detect.detect(FakeProbes(mem_max="max"), now=0)["mem_limit_mib"], 15625)

    def test_driver_version_parsing(self):
        self.assertEqual(detect.host_driver_version(DRIVER_TEXT), "610.43.02")
        for text in (None, "", "no driver here"):
            self.assertIsNone(detect.host_driver_version(text))


class RealFiles(unittest.TestCase):
    """The real probes against real files: an empty regular file is not a device."""

    def test_an_empty_regular_file_is_not_a_device_and_dev_null_is(self):
        probes = detect.Probes()
        with tempfile.NamedTemporaryFile() as placeholder:
            self.assertTrue(probes.exists(placeholder.name))
            self.assertFalse(probes.is_chardev(placeholder.name))
        self.assertTrue(probes.is_chardev("/dev/null"))
        self.assertFalse(probes.is_chardev("/nonexistent/nvidia0"))


class Recording(unittest.TestCase):
    def test_env_file_is_valid_shell_and_carries_the_decision(self):
        rec = detect.detect(gpu_node(), now=0)
        rec["reason"] = "it's \"quoted\"; $(touch /tmp/x) `id`"
        text = detect.env_text(rec)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mode.env")
            Path(path).write_text(text)
            out = subprocess.run(["bash", "-c", '. "$1"; printf "%s\\n%s\\n%s\\n%s" "$MUJOCO_CAULDRON_MODE" "$MUJOCO_CAULDRON_DEVICE" "$MUJOCO_GL" "$MUJOCO_CAULDRON_REASON"', "x", path],
                                 capture_output=True, text=True, timeout=20, cwd=tmp)
            self.assertEqual(out.returncode, 0, out.stderr)
            self.assertEqual(out.stdout.split("\n"), ["gpu", "cuda:0", "egl", rec["reason"]])
            self.assertEqual(os.listdir(tmp), ["mode.env"], "sourcing the file must not run anything in it")
        for line in text.splitlines()[1:]:
            key, _, value = line.partition("=")
            self.assertRegex(key, r"^[A-Z_]+$")
            self.assertEqual(len(shlex.split(value)), 1, line)

    def test_cli_print_writes_nothing_and_rejects_other_arguments(self):
        out = subprocess.run([sys.executable, str(GUEST / "detect.py"), "--print"], capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('"mode":', out.stdout)
        bad = subprocess.run([sys.executable, str(GUEST / "detect.py"), "--write-somewhere"], capture_output=True, text=True, timeout=60)
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
