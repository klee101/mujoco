#!/usr/bin/env python3
"""Reference async mjb integration example.

This file is the recommended starting point for integrating the mjb Python
module into an application. It intentionally avoids benchmark plotting, stubs,
and compatibility code. For stress testing and profiling, use
`mjbatch/tests/test_readback.py` instead.

Core mjb call sequence:
  1. Create BatchRendererConfig.
  2. Create BatchRenderer(model_ptrs, config).
  3. Start the readback thread.
  4. Set an optional readback callback.
  5. record_next_nowait(shm_ptr, max_geoms, max_lights) -> slot.
  6. submit_next(slot).
  7. Poll is_slot_ready(slot) and release the input buffer.
  8. Stop the readback thread during shutdown.
"""

from __future__ import annotations

import argparse
import ctypes
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from multiprocessing import shared_memory


BUF_FREE = 0
BUF_WRITING = 1
BUF_READY = 2
BUF_READING = 3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal async mjb integration example")
    parser.add_argument("--build-dir", type=Path,
                        default=repo_root() / "build")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--num-buffers", type=int, default=4)
    parser.add_argument("--max-geoms", type=int, default=1000)
    parser.add_argument("--max-lights", type=int, default=10)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--debug-mjb", action="store_true",
                        help="Enable verbose C++ mjb pipeline logging")
    parser.add_argument("--readback-timeout", type=float, default=5.0)
    parser.add_argument("--poll-interval", type=float, default=0.00005)
    return parser.parse_args()


def add_mjb_to_python_path(build_dir: Path) -> None:
    lib_dir = build_dir / "lib"
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))


def get_mujoco_address(model_obj) -> int:
    if hasattr(model_obj, "get_model"):
        model_obj = model_obj.get_model()
    if hasattr(model_obj, "_model"):
        model_obj = model_obj._model
    if hasattr(model_obj, "_address"):
        return int(model_obj._address)
    if hasattr(model_obj, "_model_ptr"):
        ptr = model_obj._model_ptr
        return int(ptr) if isinstance(ptr, int) else ctypes.cast(
            ptr, ctypes.c_void_p).value
    raise TypeError("Could not extract mjModel pointer from model object")


def create_model_pointers(robosuite, num_envs: int):
    model_ptrs = []
    keepalive_envs = []
    for _ in range(num_envs):
        env = robosuite.make(
            env_name="TwoArmLift",
            robots=["UR5e", "UR5e"],
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
        )
        keepalive_envs.append(env)
        model_ptrs.append(get_mujoco_address(env.sim.model))
    return model_ptrs, keepalive_envs


def worker_fn(env_idx, step_counter, stop_flag, buffer_state,
              buffer_done_count, buffer_cond, cfg, error_queue):
    env = None
    try:
        import numpy as np
        import robosuite

        env = robosuite.make(
            env_name="TwoArmLift",
            robots=["UR5e", "UR5e"],
            camera_names=["frontview", "robot0_eye_in_hand",
                          "robot1_eye_in_hand"],
            camera_heights=cfg["height"],
            camera_widths=cfg["width"],
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=False,
            shared_memory_name=cfg["shm_name"],
            env_index=env_idx,
            num_buffers=cfg["num_buffers"],
            total_envs=cfg["num_envs"],
            max_geoms=cfg["max_geoms"],
        )
        env.reset()
        low, high = env.action_spec
        local_last_step = -1

        try:
            while not stop_flag.value:
                target_step = local_last_step + 1
                target_buffer = target_step % cfg["num_buffers"]

                with buffer_cond:
                    while not stop_flag.value:
                        if (step_counter.value >= target_step and
                                buffer_state[target_buffer] == BUF_WRITING):
                            break
                        buffer_cond.wait(timeout=0.001)

                if stop_flag.value:
                    break

                env.sim._render_context_offscreen.set_buffer_id(target_buffer)
                env.step(np.random.uniform(low, high))

                with buffer_done_count.get_lock():
                    buffer_done_count[target_buffer] += 1
                    done_count = buffer_done_count[target_buffer]

                if done_count == cfg["num_envs"]:
                    buffer_state[target_buffer] = BUF_READY
                    with buffer_cond:
                        buffer_cond.notify_all()

                local_last_step = target_step
        finally:
            if env is not None:
                env.close()

    except BaseException:
        error_queue.put(traceback.format_exc())
        stop_flag.value = 1
        with buffer_cond:
            buffer_cond.notify_all()


def raise_worker_error_if_any(error_queue) -> None:
    try:
        message = error_queue.get_nowait()
    except queue.Empty:
        return
    raise RuntimeError(f"worker process failed:\n{message}")


def main() -> int:
    args = parse_args()
    add_mjb_to_python_path(args.build_dir)

    import mjb
    import numpy as np
    import robosuite
    from robosuite.utils.binding_utils_mjb import get_render_packet_dtype

    mp.set_start_method("spawn", force=True)

    packet_dtype = get_render_packet_dtype(args.max_geoms, args.max_lights)
    packet_size = packet_dtype.itemsize
    shm_name = f"mjb_async_example_{os.getpid()}"
    shm_size = args.num_buffers * args.num_envs * packet_size

    shm = shared_memory.SharedMemory(create=True, size=shm_size, name=shm_name)
    keepalive_envs = []
    workers = []

    stop_flag = mp.Value("i", 0)
    step_counter = mp.Value("i", 0)
    buffer_state = mp.Array("i", [BUF_FREE] * args.num_buffers)
    buffer_done_count = mp.Array("i", [0] * args.num_buffers)
    buffer_cond = mp.Condition()
    error_queue = mp.Queue()
    buffer_state[0] = BUF_WRITING

    buffer_addresses = []
    for buffer_idx in range(args.num_buffers):
        offset = buffer_idx * args.num_envs * packet_size
        view = np.ndarray((args.num_envs, packet_size), dtype=np.uint8,
                          buffer=shm.buf, offset=offset)
        buffer_addresses.append(view.ctypes.data)

    readback_steps = set()
    readback_frames = 0
    readback_lock = threading.Lock()

    def on_readback_done(step_id: int, frames) -> None:
        nonlocal readback_frames
        with readback_lock:
            readback_steps.add(step_id)
            readback_frames += len(frames)
        # Copy frames here if they need to outlive this callback.

    try:
        model_ptrs, keepalive_envs = create_model_pointers(
            robosuite, args.num_envs)

        renderer_cfg = mjb.BatchRendererConfig()
        renderer_cfg.batch_size = args.num_envs
        renderer_cfg.frame_width = args.width
        renderer_cfg.frame_height = args.height
        renderer_cfg.gpu_id = args.gpu_id
        renderer_cfg.enable_validation = False
        renderer_cfg.debug_logging = args.debug_mjb

        renderer = mjb.BatchRenderer(model_ptrs, renderer_cfg)
        renderer.set_readback_callback(on_readback_done)
        renderer.start_readback_thread()

        worker_cfg = {
            "num_envs": args.num_envs,
            "width": args.width,
            "height": args.height,
            "num_buffers": args.num_buffers,
            "max_geoms": args.max_geoms,
            "shm_name": shm_name,
        }

        for env_idx in range(args.num_envs):
            process = mp.Process(
                target=worker_fn,
                args=(env_idx, step_counter, stop_flag, buffer_state,
                      buffer_done_count, buffer_cond, worker_cfg,
                      error_queue),
            )
            process.start()
            workers.append(process)

        rendered = 0
        recorded = 0
        dispatched = 0

        current_slot = -1
        current_buffer = -1

        pending_slot = -1
        pending_buffer = -1

        t0 = time.perf_counter()

        while rendered < args.steps or current_slot >= 0 or pending_slot >= 0:
            raise_worker_error_if_any(error_queue)
            did_work = False

            if current_slot >= 0 and pending_slot < 0:
                renderer.submit_next(current_slot)
                pending_slot = current_slot
                pending_buffer = current_buffer
                current_slot = -1
                current_buffer = -1
                did_work = True

            if current_slot < 0 and recorded < args.steps:
                read_buffer = -1
                for buffer_idx in range(args.num_buffers):
                    if buffer_state[buffer_idx] == BUF_READY:
                        buffer_state[buffer_idx] = BUF_READING
                        read_buffer = buffer_idx
                        break

                if read_buffer >= 0:
                    slot = -1
                    while slot < 0:
                        raise_worker_error_if_any(error_queue)
                        slot = renderer.record_next_nowait(
                            buffer_addresses[read_buffer],
                            args.max_geoms,
                            args.max_lights,
                        )
                        if slot < 0:
                            time.sleep(args.poll_interval)

                    recorded += 1
                    current_slot = slot
                    current_buffer = read_buffer
                    did_work = True

            if pending_slot >= 0 and renderer.is_slot_ready(pending_slot):
                buffer_state[pending_buffer] = BUF_FREE
                rendered += 1

                with buffer_cond:
                    buffer_cond.notify_all()

                pending_slot = -1
                pending_buffer = -1
                did_work = True

            next_step = dispatched + 1
            next_buffer = next_step % args.num_buffers
            if (next_step < args.steps and
                    buffer_state[next_buffer] == BUF_FREE):
                buffer_done_count[next_buffer] = 0
                buffer_state[next_buffer] = BUF_WRITING
                dispatched = next_step
                step_counter.value = dispatched
                with buffer_cond:
                    buffer_cond.notify_all()
                did_work = True

            if not did_work:
                time.sleep(args.poll_interval)

        deadline = time.perf_counter() + args.readback_timeout
        while len(readback_steps) < args.steps and time.perf_counter() < deadline:
            raise_worker_error_if_any(error_queue)
            time.sleep(args.poll_interval)

        elapsed = time.perf_counter() - t0
        with readback_lock:
            completed_readbacks = len(readback_steps)
            completed_frames = readback_frames

        print(
            "mjb async example completed: "
            f"rendered_steps={rendered}, "
            f"readback_steps={completed_readbacks}, "
            f"readback_frames={completed_frames}, "
            f"elapsed_sec={elapsed:.3f}"
        )
        return 0

    finally:
        stop_flag.value = 1
        with buffer_cond:
            buffer_cond.notify_all()

        for process in workers:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join()

        if "renderer" in locals():
            renderer.stop_readback_thread()
            renderer.set_readback_callback(None)

        for env in keepalive_envs:
            env.close()

        shm.close()
        shm.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
