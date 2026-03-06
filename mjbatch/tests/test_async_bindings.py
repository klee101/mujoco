#!/usr/bin/env python3
"""
Feasibility Test for Async Pipeline Bindings with Valid MuJoCo Models

This test verifies that the new async interfaces work correctly with valid
MuJoCo model pointers from Robosuite.
"""

import os
import sys
import time
import threading
import queue
import numpy as np
import argparse
import multiprocessing
from multiprocessing import shared_memory, Value, Condition, Array

BUILD_LIB_PATH = "/home/hpf/project/vulkan/mujoco/mujoco/build/lib"
if os.path.exists(BUILD_LIB_PATH):
    sys.path.append(BUILD_LIB_PATH)

try:
    import mjb
    MJB_AVAILABLE = True
except ImportError:
    print("[Test] Warning: mjb library not found!")
    MJB_AVAILABLE = False

import robosuite
from robosuite.utils.binding_utils_mjb import get_render_packet_dtype


def get_mujoco_address(model_obj):
    import ctypes
    if hasattr(model_obj, "get_model"):
        model_obj = model_obj.get_model()
    if hasattr(model_obj, "_model"):
        model_obj = model_obj._model
    if hasattr(model_obj, "_address"):
        return model_obj._address
    if hasattr(model_obj, "_model_ptr"):
        ptr = model_obj._model_ptr
        return int(ptr) if isinstance(ptr, int) else ctypes.cast(ptr, ctypes.c_void_p).value
    raise ValueError("Could not extract mjModel address")


class AsyncPipeline:
    def __init__(self, renderer, shm_ptr, max_geom, max_light):
        self.renderer = renderer
        self.shm_ptr = shm_ptr
        self.max_geom = max_geom
        self.max_light = max_light
        self.running = False
        self.result_queue = queue.Queue(maxsize=2)
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def _run(self):
        pending_slot = None

        while self.running:
            try:
                self.renderer.update_async(self.shm_ptr, self.max_geom, self.max_light)
                current_slot = self.renderer.record_next(self.shm_ptr)
                self.renderer.submit_next()

                # 等上一帧渲染完（不等 readback），GPU 同时跑当前帧
                if pending_slot is not None:
                    self.renderer.wait_slot(pending_slot)
                    try:
                        # 把 slot 传出去，消费者调用 get_image(slot,...) 时才等 readback
                        self.result_queue.put(pending_slot, timeout=0.1)
                    except queue.Full:
                        pass

                pending_slot = current_slot

            except Exception as e:
                print(f"[AsyncPipeline] Error: {e}")
                break

        if pending_slot is not None:
            try:
                self.renderer.wait_slot(pending_slot)
                self.result_queue.put(pending_slot, timeout=0.5)
            except Exception:
                pass

    def get_result(self, timeout=1.0):
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None


def test_basic_rendering(num_envs, width, height):
    """Test basic synchronous rendering with valid models"""
    print(f"\n{'='*60}")
    print("TEST 1: Basic Synchronous Rendering")
    print(f"{'='*60}")
    print(f"Config: {num_envs} envs, {width}x{height}")
    
    # Create valid model pointers using Robosuite
    print("[Test] Creating Robosuite environments to get valid model pointers...")
    model_ptrs = []
    kept_envs = []
    
    for i in range(num_envs):
        dummy = robosuite.make(
            env_name='TwoArmLift', robots=['UR5e','UR5e'],
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        kept_envs.append(dummy)
        ptr = get_mujoco_address(dummy.sim.model)
        model_ptrs.append(ptr)
    
    print(f"[Test] Got {len(model_ptrs)} valid model pointers")
    
    # Create renderer
    cfg = mjb.BatchRendererConfig()
    cfg.batch_size = num_envs
    cfg.frame_width = width
    cfg.frame_height = height
    cfg.gpu_id = 0
    cfg.enable_validation = False  # Disable validation for testing
    
    print("[Test] Creating BatchRenderer...")
    renderer = mjb.BatchRenderer(model_ptrs, cfg)
    print("[Test] BatchRenderer created successfully!")
    
    # Create shared memory for rendering
    MAX_GEOMS = 1000
    MAX_LIGHTS = 10
    NUM_BUFFERS = 3
    
    SHM_NAME = "test_async_shm"
    try:
        old_shm = shared_memory.SharedMemory(name=SHM_NAME)
        old_shm.close()
        old_shm.unlink()
    except:
        pass
    
    packet_dtype = get_render_packet_dtype(MAX_GEOMS, MAX_LIGHTS)
    packet_size = packet_dtype.itemsize
    total_shm_size = NUM_BUFFERS * num_envs * packet_size
    
    shm = shared_memory.SharedMemory(create=True, size=total_shm_size, name=SHM_NAME)
    
    buf_addrs = []
    for buf_idx in range(NUM_BUFFERS):
        offset = buf_idx * num_envs * packet_size
        buf_arr = np.ndarray((num_envs, packet_dtype.itemsize),
                            dtype=np.uint8, buffer=shm.buf, offset=offset)
        buf_addrs.append(buf_arr.ctypes.data)
    
    # Run basic render test
    print("[Test] Running render_from_shm...")
    start = time.perf_counter()
    result = renderer.render_from_shm(buf_addrs[0], 0, MAX_GEOMS, MAX_LIGHTS)
    elapsed = time.perf_counter() - start
    
    print(f"[Test] render_from_shm completed in {elapsed*1000:.2f}ms, result={result}")
    
    # Test get_image
    print("[Test] Testing get_image...")
    try:
        img = renderer.get_image(0, 0)
        print(f"[Test]   get_image(0,0): shape={img.shape}, size={img.nbytes} bytes")
    except Exception as e:
        print(f"[Test]   get_image failed: {e}")
    
    return renderer, shm, buf_addrs, kept_envs


# test_async_bindings 函数修改
def test_async_bindings(renderer, shm_ptr, max_geom=1000, max_light=10):
    print(f"\n{'='*60}")
    print("TEST 2: Async Bindings")
    print(f"{'='*60}")
    
    # ✅ 关键修复1: 启动 readback thread，否则 slot.state 永远不归 FREE
    renderer.start_readback_thread()
    
    print("[Test] Testing update_async()...")
    start = time.perf_counter()
    result = renderer.update_async(shm_ptr, max_geom, max_light)
    elapsed = time.perf_counter() - start
    print(f"[Test]   update_async() returned {result} in {elapsed*1000:.2f}ms")
    
    print("[Test] Testing record_next()...")
    start = time.perf_counter()
    slot = renderer.record_next(shm_ptr)
    elapsed = time.perf_counter() - start
    print(f"[Test]   record_next() returned slot={slot} in {elapsed*1000:.2f}ms")
    
    print("[Test] Testing submit_next()...")
    start = time.perf_counter()
    result = renderer.submit_next()
    elapsed = time.perf_counter() - start
    print(f"[Test]   submit_next() returned {result} in {elapsed*1000:.2f}ms")
    
    print("[Test] Testing wait_slot()...")
    start = time.perf_counter()
    # ✅ 关键修复2: 用 record_next 返回的实际 slot，不要硬编码 0
    result = renderer.wait_slot(slot)
    elapsed = time.perf_counter() - start
    print(f"[Test]   wait_slot({slot}) returned {result} in {elapsed*1000:.2f}ms")
    
    renderer.stop_readback_thread()
    return True


# test_async_pipeline 函数修改
def test_async_pipeline(renderer, shm_ptr, num_iterations=5, max_geom=1000, max_light=10):
    print(f"\n{'='*60}")
    print(f"TEST 3: Async Pipeline ({num_iterations} iterations)")
    print(f"{'='*60}")
    
    # ✅ 关键修复: 启动 readback thread
    renderer.start_readback_thread()
    
    times = []
    for i in range(num_iterations):
        print(f"[Test] Iteration {i+1}/{num_iterations}...")
        start = time.perf_counter()
        
        renderer.update_async(shm_ptr, max_geom, max_light)
        slot = renderer.record_next(shm_ptr)   # ✅ 拿到真实 slot
        renderer.submit_next()
        renderer.wait_slot(slot)               # ✅ 等真实 slot
        
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"[Test]   Iteration {i+1} completed in {elapsed*1000:.2f}ms, slot={slot}")
    
    renderer.stop_readback_thread()
    
    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f"\n[Test] Async Pipeline Results:")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  Estimated FPS: {fps:.2f}")
    return avg_time, fps


def test_render_and_save_images(renderer, shm_ptr, num_envs, num_frames=3,
                                 output_dir="/tmp/mujoco_test_frames",
                                 max_geom=1000, max_light=10):
    os.makedirs(output_dir, exist_ok=True)
    renderer.start_readback_thread()

    saved_count = 0
    pending = []  # list of (frame_idx, slot)

    for frame_idx in range(num_frames):
        print(f"[Test] Submitting frame {frame_idx+1}/{num_frames}...")
        renderer.update_async(shm_ptr, max_geom, max_light)
        slot = renderer.record_next(shm_ptr)
        renderer.submit_next()

        # wait_slot 只等渲染完，slot 可复用，GPU 继续跑下一帧
        renderer.wait_slot(slot)
        pending.append((frame_idx, slot))

    # 所有帧提交完后，按顺序等 readback 并保存
    # 此时 GPU 早已完成所有渲染，readback 也大概率已完成
    for frame_idx, slot in pending:
        print(f"[Test] Saving frame {frame_idx+1}/{num_frames} (slot={slot})...")
        for env_idx in range(num_envs):
            try:
                # get_image 内部等 readback 完成
                img = renderer.get_image(slot, env_idx, 0)

                if img is None or img.size == 0:
                    print(f"[Test]   env{env_idx}: get_image returned empty")
                    continue

                img_path = os.path.join(output_dir,
                    f"frame{frame_idx:02d}_env{env_idx:02d}.png")

                from PIL import Image
                if img.shape[2] == 4:
                    img_rgba = Image.fromarray(img)
                    img_rgb = Image.new("RGB", img_rgba.size)
                    img_rgb.paste(img_rgba, mask=img_rgba.split()[3])
                    img_rgb.save(img_path)
                else:
                    Image.fromarray(img).save(img_path)

                print(f"[Test]   env{env_idx}: saved {img_path} ({img.shape})")
                saved_count += 1

            except Exception as e:
                print(f"[Test]   env{env_idx}: failed - {e}")

    renderer.stop_readback_thread()
    print(f"\n[Test] Saved {saved_count} images to {output_dir}")
    return saved_count > 0

def test_async_threaded_pipeline(renderer, shm_ptr, num_frames=5, max_geom=1000, max_light=10):
    """Test async pipeline with background thread"""
    print(f"\n{'='*60}")
    print(f"TEST 4: Async Threaded Pipeline ({num_frames} frames)")
    print(f"{'='*60}")
    
    renderer.start_readback_thread()
    
    pipeline = AsyncPipeline(renderer, shm_ptr, max_geom, max_light)
    pipeline.start()
    
    print("[Test] Started async pipeline thread")
    
    completed = 0
    timeout = 10.0  # seconds
    start_time = time.perf_counter()
    
    while completed < num_frames and (time.perf_counter() - start_time) < timeout:
        slot = pipeline.get_result(timeout=0.5)
        if slot is not None:
            completed += 1
            print(f"[Test] Frame {completed} completed, slot={slot}")
    
    pipeline.stop()
    renderer.stop_readback_thread()
    print(f"[Test] Completed {completed}/{num_frames} frames")
    
    return completed == num_frames


def main():
    parser = argparse.ArgumentParser(description="Async Bindings Feasibility Test")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of environments")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--iterations", type=int, default=5, help="Async pipeline iterations")
    parser.add_argument("--output_dir", type=str, default="/tmp/mujoco_test_frames", help="Output directory for rendered images")
    args = parser.parse_args()
    
    print("="*60)
    print("Async Bindings Feasibility Test")
    print("="*60)
    print(f"Configuration:")
    print(f"  Environments: {args.num_envs}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Iterations: {args.iterations}")
    print(f"  mjb Available: {MJB_AVAILABLE}")
    
    if not MJB_AVAILABLE:
        print("\n[ERROR] mjb library not available. Please build it first.")
        return 1
    
    results = {}
    shm = None
    
    try:
        # Test 1: Basic rendering with valid models
        renderer, shm, buf_addrs, kept_envs = test_basic_rendering(
            args.num_envs, args.width, args.height)
        results['basic'] = True
        
        # Test: Render and save images to verify rendering works
        results['render_images'] = test_render_and_save_images(
            renderer, buf_addrs[0], args.num_envs, num_frames=3, output_dir=args.output_dir)
        
        # Test 2: Async bindings
        results['async_bindings'] = test_async_bindings(
            renderer, buf_addrs[0], 1000, 10)
        
        # Test 3: Async pipeline (synchronous)
        avg_time, fps = test_async_pipeline(
            renderer, buf_addrs[0], args.iterations, 1000, 10)
        results['async_pipeline'] = True
        
        # Test 4: Async threaded pipeline
        results['async_threaded'] = test_async_threaded_pipeline(
            renderer, buf_addrs[0], 5, 1000, 10)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['error'] = str(e)
    
    finally:
        if shm:
            shm.close()
            shm.unlink()
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
        print(f"  {test_name}: {status}")
    
    print()
    if all_passed:
        print("[SUCCESS] All tests passed!")
    else:
        print("[FAILURE] Some tests failed")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
