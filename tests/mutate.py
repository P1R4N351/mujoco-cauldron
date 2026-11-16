#!/usr/bin/env python3
"""Break the GPU-or-CPU detector on purpose and check its tests notice.

Authored-by: CPCS. Run:  python3 skills/mujoco-cauldron/tests/mutate.py
Each arm changes one line of guest/detect.py in a scratch copy, runs the suite and restores the
file. CAUGHT means the suite failed; the failing tests are printed. A fault of this harness (a needle
that does not occur exactly once, an unchanged file, a control that fails) is SETUP-ERROR, never a
verdict about the detector. Exit 0: control passes, every arm caught, neutral arm survives.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
ARMS = [
    ("a placeholder file counts as a device", "            return stat.S_ISCHR(os.stat(path).st_mode)", "            return os.path.exists(path)"),
    ("no devices still asks the driver", "    if not real:\n        why =", "    if False:\n        why ="),
    ("an incomplete device set is accepted", "    if len(real) < len(DEVICES):", "    if False:"),
    ("a missing library is not told apart", '    if err == "missing":', "    if False:"),
    ("an unusable GPU is reported as GPU", "    if err:\n        return (\"cpu\", \"GPU present but unusable", "    if False:\n        return (\"cpu\", \"GPU present but unusable"),
    ("a library with no GPU is reported as GPU", "    if not gpus:\n        return (\"cpu\", \"driver library loaded", "    if False:\n        return (\"cpu\", \"driver library loaded"),
    ("GPU mode names the cpu device", '        "device": "cuda:0" if mode == "gpu" else "cpu",', '        "device": "cpu",'),
    ("the first GPU is taken, not the roomiest", '    best = max(gpus, key=lambda g: g["mem_free_mib"]) if gpus else None', "    best = gpus[0] if gpus else None"),
    ("parallel count ignores free memory", "        if free_mib >= floor:\n            return envs", "        if True:\n            return envs"),
    ("values written to the env file unquoted", "shlex.quote(str(record[key]))", "str(record[key])"),
    ("--print still writes", '    if "--print" not in argv:', "    if True:"),
    ("memory limit ignores the container's limit", "    if raw.isdigit():\n        return int(raw) >> 20", "    if False:\n        return int(raw) >> 20"),
]
NEUTRAL = ("neutral wording change", '    """The NVIDIA kernel module version from /proc/driver/nvidia/version, or None."""',
           '    """The NVIDIA kernel module version from /proc/driver/nvidia/version, or None. """')


def run_suite(work):
    """(passed, tests run, failing test names)."""
    assert os.path.isdir(work)
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=work, capture_output=True, text=True, timeout=300)
    ran = re.search(r"^Ran (\d+) tests", proc.stderr, re.M)
    return (proc.returncode == 0, int(ran.group(1)) if ran else 0, sorted(set(re.findall(r"^(?:FAIL|ERROR): (\w+)", proc.stderr, re.M))))


def run_arm(work, arm):
    """Verdict and failing tests for one arm."""
    assert len(arm) == 3
    _label, old, new = arm
    target = Path(work) / "guest" / "detect.py"
    pristine = target.read_text()
    if old == new or pristine.count(old) != 1:
        return ("SETUP-ERROR: needle occurs %d times" % pristine.count(old), [])
    target.write_text(pristine.replace(old, new))
    try:
        passed, ran, failing = run_suite(work)
    finally:
        target.write_text(pristine)
    if ran == 0:
        return ("SETUP-ERROR: the suite did not run", failing)
    return ("SURVIVED" if passed else "CAUGHT", failing)


def main():
    """Control, every arm, the neutral arm, then the restored control."""
    work = tempfile.mkdtemp(prefix="mujoco-cauldron-mutate-")
    try:
        for name in ("guest", "tests"):
            shutil.copytree(SKILL / name, Path(work) / name, ignore=shutil.ignore_patterns("__pycache__"))
        passed, total, failing = run_suite(work)
        print("control: %s, %d tests %s" % ("pass" if passed else "FAIL", total, "" if passed else failing))
        if not passed or total == 0:
            print("SETUP-ERROR: the unmodified copy does not pass, so no arm means anything")
            return 3
        caught = setup = 0
        for arm in ARMS:
            verdict, failing = run_arm(work, arm)
            caught += verdict == "CAUGHT"
            setup += verdict.startswith("SETUP-ERROR")
            print("%-46s %-9s %s" % (arm[0], verdict, ", ".join(failing)[:110]))
        neutral, _f = run_arm(work, NEUTRAL)
        print("%-46s %s (must survive)" % (NEUTRAL[0], neutral))
        again, total_again, _f = run_suite(work)
        print("restored: %s, %d tests | caught %d of %d" % ("pass" if again else "FAIL", total_again, caught, len(ARMS)))
        if setup or neutral != "SURVIVED" or not again or total_again != total:
            return 3
        return 0 if caught == len(ARMS) else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
