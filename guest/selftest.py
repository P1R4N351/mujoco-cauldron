#!/usr/bin/env python3
"""selftest — prove the mode the container decided on, by running it.

Authored-by: CPCS. Runs inside the simulation environment (cauldron check).

Both modes: MuJoCo loads a small model and steps it, on one process and on a few
in parallel, and the steps match a second run exactly.
GPU mode also: PyTorch must see the GPU and compute on it, and Warp must open it.
A GPU mode in which either fails is a FAILED self-test, not a quiet fall back to CPU.

The parallel part is deliberately small (at most half the threads, about a second
each): it is a function check, not a load test.
"""
import concurrent.futures
import json
import os
import sys
import time

MODEL_XML = """
<mujoco model="selftest">
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="cart" pos="0 0 0.05">
      <joint name="slide" type="slide" axis="1 0 0" damping="0.5"/>
      <geom type="box" size="0.05 0.03 0.02" mass="0.2"/>
      <body name="pole" pos="0 0 0.02">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.008" mass="0.05"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slide" gear="1" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""
STEPS = 5000
PARALLEL_SECONDS = 1.0
MAX_PARALLEL_STEPS = 2_000_000


def rollout(steps):
    """Step the model `steps` times under a fixed control pattern; return (final state, seconds)."""
    import mujoco
    assert 0 < steps <= MAX_PARALLEL_STEPS
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    data.qpos[1] = 0.1
    start = time.perf_counter()
    for index in range(steps):
        data.ctrl[0] = 0.5 if (index // 250) % 2 == 0 else -0.5
        mujoco.mj_step(model, data)
    return ([float(v) for v in data.qpos] + [float(v) for v in data.qvel], time.perf_counter() - start)


def timed_worker(seconds):
    """Step for about `seconds`; return the number of steps taken."""
    import mujoco
    assert 0 < seconds <= 10
    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)
    deadline, done = time.perf_counter() + seconds, 0
    for _ in range(MAX_PARALLEL_STEPS // 1000):
        for _inner in range(1000):
            mujoco.mj_step(model, data)
        done += 1000
        if time.perf_counter() >= deadline:
            break
    return done


def check_mujoco(threads):
    """MuJoCo works, is repeatable, and runs in parallel."""
    assert threads >= 1
    import mujoco
    first, seconds = rollout(STEPS)
    second, _ = rollout(STEPS)
    workers = max(1, min(4, threads // 2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        counts = list(pool.map(timed_worker, [PARALLEL_SECONDS] * workers))
    moved = abs(first[0]) > 1e-6 and all(v == v for v in first)
    return {"ok": moved and first == second, "version": mujoco.__version__, "repeatable": first == second,
            "state_changed": moved, "single_steps_per_s": round(STEPS / seconds),
            "parallel_workers": workers, "parallel_steps_per_s": round(sum(counts) / PARALLEL_SECONDS)}


def check_torch(mode):
    """PyTorch computes; in GPU mode it must compute on the GPU."""
    assert mode in ("gpu", "cpu")
    import torch
    out = {"version": torch.__version__, "cuda_available": bool(torch.cuda.is_available())}
    device = "cuda:0" if mode == "gpu" else "cpu"
    if mode == "gpu" and not out["cuda_available"]:
        return dict(out, ok=False, why="GPU mode, but PyTorch cannot see a GPU")
    x = torch.arange(1000, dtype=torch.float32, device=device)
    out.update({"ok": float((x * 2).sum().item()) == 999000.0, "device": str(x.device)})
    if mode == "gpu":
        out["gpu_name"] = torch.cuda.get_device_name(0)
    return out


def check_warp(mode):
    """Warp imports; in GPU mode it must open the GPU. Required only in GPU mode."""
    assert mode in ("gpu", "cpu")
    try:
        import warp as wp
        import mujoco_warp  # noqa: F401 - importing it is the check
    except ImportError as exc:
        return {"ok": mode == "cpu", "installed": False, "why": "%s" % exc}
    wp.config.quiet = True
    wp.init()
    devices = [str(d) for d in wp.get_devices()]
    has_cuda = any(d.startswith("cuda") for d in devices)
    return {"ok": has_cuda or mode == "cpu", "installed": True, "version": wp.__version__, "devices": devices,
            "why": "" if (has_cuda or mode == "cpu") else "GPU mode, but Warp found no CUDA device"}


def main():
    """Run the checks for the decided mode; exit 0 only if every required one passes."""
    mode = os.environ.get("MUJOCO_CAULDRON_MODE", "")
    if mode not in ("gpu", "cpu"):
        sys.stderr.write("MUJOCO_CAULDRON_MODE is not set; run this through: cauldron check\n")
        return 2
    threads = int(os.environ.get("MUJOCO_CAULDRON_CPU_THREADS", "1") or "1")
    report = {"mode": mode, "reason": os.environ.get("MUJOCO_CAULDRON_REASON", "")}
    for name, check in (("mujoco", lambda: check_mujoco(threads)), ("torch", lambda: check_torch(mode)), ("warp", lambda: check_warp(mode))):
        try:
            report[name] = check()
        except Exception as exc:  # noqa: BLE001 - a crash in a check is that check failing, and is reported as such
            report[name] = {"ok": False, "why": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    report["ok"] = all(report[name].get("ok") for name in ("mujoco", "torch", "warp"))
    print(json.dumps(report, indent=1))
    print("SELFTEST %s in %s mode" % ("PASS" if report["ok"] else "FAIL", mode))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
