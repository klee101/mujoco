import time
import threading
import queue
import multiprocessing
from multiprocessing import shared_memory, Value, Condition, Array
import os
import sys
import ctypes
import gc
import numpy as np
import argparse

BUILD_LIB_PATH = "/home/hpf/projects/mujoco/mujoco/build/lib"
if os.path.exists(BUILD_LIB_PATH):
    sys.path.append(BUILD_LIB_PATH)

try:
    import mjb
except ImportError:
    print("[Warning] mjb library not found, using stub")
    class mjb_stub:
        class BatchRendererConfig: pass
        class BatchRenderer:
            def __init__(self, *args, **kwargs): pass
            def render_from_shm(self, *args): time.sleep(0.005)
            def update_async(self, *args): time.sleep(0.001)
            def record_next(self, *args): return 0
            def submit_next(self): time.sleep(0.003)
            def wait_slot(self, *args): time.sleep(0.001)
            def get_image(self, *args): return np.zeros((480, 640, 4), dtype=np.uint8)
    mjb = mjb_stub()

import robosuite
from robosuite.utils.binding_utils_mjb import get_render_packet_dtype

BUF_FREE = 0
BUF_WRITING = 1
BUF_READY = 2
BUF_READING = 3

CMD_RING_SIZE = 4
SWAP_COUNT = 3


class AsyncPipelineController:
    def __init__(self, renderer, num_buffers=3):
        self.renderer = renderer
        self.num_buffers = num_buffers
        self.running = False
        
        self.record_queue = queue.Queue(maxsize=CMD_RING_SIZE)
        self.render_done_event = threading.Event()
        
        self.producer_thread = None
        self.consumer_thread = None
        self.current_slot = 0
    
    def start(self, shm_ptr, max_geom, max_light):
        self.running = True
        self.shm_ptr = shm_ptr
        self.max_geom = max_geom
        self.max_light = max_light
        
        self.producer_thread = threading.Thread(target=self._producer_fn, daemon=True)
        self.consumer_thread = threading.Thread(target=self._consumer_fn, daemon=True)
        
        self.producer_thread.start()
        self.consumer_thread.start()
    
    def stop(self):
        self.running = False
        if self.producer_thread:
            self.producer_thread.join(timeout=1.0)
        if self.consumer_thread:
            self.consumer_thread.join(timeout=1.0)
    
    def _producer_fn(self):
        while self.running:
            try:
                self.renderer.update_async(self.shm_ptr, self.max_geom, self.max_light)
                slot = self.renderer.record_next(self.shm_ptr)
                if slot >= 0:
                    self.record_queue.put(slot, timeout=0.1)
            except queue.Full:
                pass
            except Exception as e:
                print(f"[Producer] Error: {e}")
    
    def _consumer_fn(self):
        while self.running:
            try:
                slot = self.record_queue.get(timeout=0.1)
                self.renderer.submit_next()
                self.renderer.wait_slot(slot)
                self.render_done_event.set()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Consumer] Error: {e}")
    
    def wait_for_frame(self, timeout=None):
        self.render_done_event.wait(timeout=timeout)
        self.render_done_event.clear()
    
    def get_image(self, batch_idx, cam_idx):
        return self.renderer.get_image(batch_idx, cam_idx)


class EventLogger:
    def __init__(self, filename):
        self.filename = filename
        self.lock = threading.Lock()
    
    def init_file(self):
        with open(self.filename, 'w') as f:
            f.write("process,event,start_time,end_time,step,buffer_id\n")
    
    def log(self, process_name, event_name, start, end, step=-1, buffer_id=-1):
        with self.lock:
            with open(self.filename, 'a') as f:
                f.write(f"{process_name},{event_name},{start:.9f},{end:.9f},{step},{buffer_id}\n")


def allocate_worker_cores(env_id, total_workers, start_core, end_core):
    available_cores = end_core - start_core + 1
    if available_cores <= 0:
        return set()
    cores_per_worker = 1
    assigned_core = start_core + (env_id % available_cores)
    return {assigned_core}


def get_mujoco_address(model_obj):
    if hasattr(model_obj, "get_model"):
        model_obj = model_obj.get_model()
    if hasattr(model_obj, "_model"):
        model_obj = model_obj._model
    if hasattr(model_obj, "_address"):
        return model_obj._address
    if hasattr(model_obj, "_model_ptr"):
        ptr = model_obj._model_ptr
        return int(ptr) if isinstance(ptr, int) else ctypes.cast(ptr, ctypes.c_void_p).value
    raise ValueError("Could not extract address")


def worker_fn(env_idx, log_lock, step_counter, buffer_state, buffer_done_count, 
              stop_flag, buffer_cond, cfg):
    
    total_envs = cfg['total_envs']
    frame_width = cfg['frame_width']
    frame_height = cfg['frame_height']
    shm_name = cfg['shm_name']
    num_buffers = cfg['num_buffers']
    max_geoms = cfg['max_geoms']
    log_file = cfg['log_file']
    cpu_start = cfg['worker_cpu_start']
    cpu_end = cfg['worker_cpu_end']
    cold_start = cfg['cold_start_steps']

    os.environ["OMP_NUM_THREADS"] = "1"
    
    worker_cores = allocate_worker_cores(env_idx, total_envs, cpu_start, cpu_end)
    try:
        if worker_cores:
            os.sched_setaffinity(0, worker_cores)
    except Exception:
        pass

    gc.disable()
    logger = EventLogger(log_file)
    
    env = None
    try:
        env = robosuite.make(
            env_name='TwoArmLift', robots=['UR5e','UR5e'],
            camera_names=['frontview','robot0_eye_in_hand','robot1_eye_in_hand'],
            camera_heights=frame_height, camera_widths=frame_width,
            has_offscreen_renderer=True, ignore_done=True, use_camera_obs=False,
            shared_memory_name=shm_name, env_index=env_idx,
            num_buffers=num_buffers, total_envs=total_envs, max_geoms=max_geoms
        )
        env.reset()
        low, high = env.action_spec
    except Exception as e:
        print(f"[Worker {env_idx}] Failed to create env: {e}")
        return

    local_last_step = -1

    try:
        while not stop_flag.value:
            target_step = local_last_step + 1
            target_buf_idx = target_step % num_buffers

            with buffer_cond:
                while True:
                    if stop_flag.value:
                        return
                    
                    if (step_counter.value >= target_step and 
                        buffer_state[target_buf_idx] == BUF_WRITING):
                        break
                    
                    buffer_cond.wait(timeout=0.001)

            env.sim._render_context_offscreen.set_buffer_id(target_buf_idx)

            t_phys_start = time.perf_counter()
            action = np.random.uniform(low, high)
            env.step(action)
            t_phys_end = time.perf_counter()

            if target_step > cold_start:
                logger.log(f"Worker_{env_idx}", "Physics", t_phys_start, t_phys_end, target_step, target_buf_idx)

            with buffer_done_count.get_lock():
                buffer_done_count[target_buf_idx] += 1
                done = buffer_done_count[target_buf_idx]

            if done == total_envs:
                buffer_state[target_buf_idx] = BUF_READY
                with buffer_cond:
                    buffer_cond.notify_all()
            
            local_last_step = target_step

    finally:
        if env:
            env.close()


def main():
    parser = argparse.ArgumentParser(description="Async Pipeline Benchmark")
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cold_start", type=int, default=5)
    parser.add_argument("--log_file", type=str, default="benchmark_async_pipeline.csv")
    parser.add_argument("--use_async", action="store_true", default=True, help="Use async pipeline")
    args = parser.parse_args()
    
    TOTAL_ENVS = args.num_envs
    FRAME_WIDTH = args.width
    FRAME_HEIGHT = args.height
    PROFILING_STEPS = args.steps
    COLD_START_STEPS = args.cold_start
    
    folder_name = f"results_async_env{TOTAL_ENVS}_{FRAME_WIDTH}x{FRAME_HEIGHT}"
    os.makedirs(folder_name, exist_ok=True)
    LOG_FILE = os.path.join(folder_name, f"benchmark_async_pipeline.csv")
    OUTPUT_IMAGE = os.path.join(folder_name, f"timeline_async_env{TOTAL_ENVS}_{FRAME_WIDTH}x{FRAME_HEIGHT}.png")
    
    MAX_GEOMS = 1000
    MAX_LIGHTS = 10
    NUM_BUFFERS = 3
    SHM_NAME = "robosuite_async_pipeline"

    TOTAL_SYSTEM_CORES = multiprocessing.cpu_count()
    MAIN_CPU_CORES = {x for x in range(max(0, TOTAL_SYSTEM_CORES - 4), TOTAL_SYSTEM_CORES)}
    WORKER_CPU_START = 0
    WORKER_CPU_END = max(0, TOTAL_SYSTEM_CORES - 5)

    cfg = {
        'total_envs': TOTAL_ENVS,
        'frame_width': FRAME_WIDTH,
        'frame_height': FRAME_HEIGHT,
        'shm_name': SHM_NAME,
        'num_buffers': NUM_BUFFERS,
        'max_geoms': MAX_GEOMS,
        'log_file': LOG_FILE,
        'worker_cpu_start': WORKER_CPU_START,
        'worker_cpu_end': WORKER_CPU_END,
        'cold_start_steps': COLD_START_STEPS
    }

    multiprocessing.set_start_method("spawn", force=True)
    os.sched_setaffinity(0, MAIN_CPU_CORES)

    log_lock = multiprocessing.Lock()
    EventLogger(LOG_FILE).init_file()

    step_counter = multiprocessing.Value('i', 0)
    stop_flag = multiprocessing.Value('i', 0)

    buffer_state = Array('i', [BUF_FREE] * NUM_BUFFERS)
    buffer_done_count = Array('i', [0] * NUM_BUFFERS)
    buffer_cond = Condition()
    buffer_state[0] = BUF_WRITING

    packet_dtype = get_render_packet_dtype(MAX_GEOMS, MAX_LIGHTS)
    packet_size = packet_dtype.itemsize
    total_shm_size = NUM_BUFFERS * TOTAL_ENVS * packet_size

    try:
        old = shared_memory.SharedMemory(name=SHM_NAME)
        old.close()
        old.unlink()
    except:
        pass

    shm = shared_memory.SharedMemory(create=True, size=total_shm_size, name=SHM_NAME)
    
    buf_addrs = []
    for buf_idx in range(NUM_BUFFERS):
        offset = buf_idx * TOTAL_ENVS * packet_size
        buf_arr = np.ndarray((TOTAL_ENVS, packet_dtype.itemsize),
                             dtype=np.uint8, buffer=shm.buf, offset=offset)
        buf_addrs.append(buf_arr.ctypes.data)

    model_ptrs = []
    kept_envs = []
    for i in range(TOTAL_ENVS):
        dummy = robosuite.make(
            env_name='TwoArmLift', robots=['UR5e','UR5e'],
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        kept_envs.append(dummy)
        ptr = get_mujoco_address(dummy.sim.model)
        model_ptrs.append(ptr)

    mjb_cfg = mjb.BatchRendererConfig()
    mjb_cfg.batch_size = TOTAL_ENVS
    mjb_cfg.frame_width = FRAME_WIDTH
    mjb_cfg.frame_height = FRAME_HEIGHT
    mjb_cfg.gpu_id = 0
    
    print(f"[Main] Creating BatchRenderer with {TOTAL_ENVS} environments...")
    renderer = mjb.BatchRenderer(model_ptrs, mjb_cfg)
    print(f"[Main] BatchRenderer created successfully")

    if args.use_async:
        print("[Main] Using ASYNC pipeline with producer/consumer threads")
        pipeline = AsyncPipelineController(renderer, NUM_BUFFERS)
        pipeline.start(buf_addrs[0], MAX_GEOMS, MAX_LIGHTS)
    else:
        print("[Main] Using SYNC pipeline (original)")
        pipeline = None

    workers = []
    for i in range(TOTAL_ENVS):
        p = multiprocessing.Process(
            target=worker_fn,
            args=(i, log_lock, step_counter, 
                  buffer_state, buffer_done_count, stop_flag, buffer_cond, cfg)
        )
        p.start()
        workers.append(p)

    logger = EventLogger(LOG_FILE)
    
    profiling_start_time = None
    
    try:
        current_step_dispatched = 0 
        rendered_step = -1
        
        while rendered_step < PROFILING_STEPS:
            did_work = False
            
            read_buf = -1
            for i in range(NUM_BUFFERS):
                if buffer_state[i] == BUF_READY:
                    buffer_state[i] = BUF_READING
                    read_buf = i
                    break
            
            if read_buf != -1:
                t_rend_start = time.perf_counter()
                
                if args.use_async:
                    pipeline.wait_for_frame(timeout=5.0)
                else:
                    renderer.render_from_shm(buf_addrs[read_buf], 0, MAX_GEOMS, MAX_LIGHTS)
                
                t_rend_end = time.perf_counter()
                
                if current_step_dispatched > COLD_START_STEPS:
                    logger.log("Main", "Render_Batch", t_rend_start, t_rend_end, rendered_step, read_buf)
                
                buffer_state[read_buf] = BUF_FREE
                rendered_step += 1
                
                if rendered_step == COLD_START_STEPS:
                    profiling_start_time = time.perf_counter()
                    print(f"[Main] Cold start finished at step {rendered_step}. Profiling started.")

                with buffer_cond:
                    buffer_cond.notify_all()
                did_work = True

            next_step = current_step_dispatched + 1
            next_buf_idx = next_step % NUM_BUFFERS

            if next_step <= PROFILING_STEPS:
                if buffer_state[next_buf_idx] == BUF_FREE:
                    buffer_done_count[next_buf_idx] = 0
                    buffer_state[next_buf_idx] = BUF_WRITING
                    
                    current_step_dispatched = next_step
                    step_counter.value = current_step_dispatched
                    
                    with buffer_cond:
                        buffer_cond.notify_all()
                    did_work = True

            if not did_work:
                time.sleep(0.0001)

        profiling_end_time = time.perf_counter()
        
        if profiling_start_time is not None:
            total_duration = profiling_end_time - profiling_start_time
            valid_steps = PROFILING_STEPS - COLD_START_STEPS
            if valid_steps > 0:
                avg_time = total_duration / valid_steps
                fps = valid_steps / total_duration
                result_str = f"[Result] Steps: {valid_steps} | Duration: {total_duration:.4f}s | Avg Time: {avg_time:.6f}s | FPS: {fps:.2f}"
                print(result_str)
                
                summary_file = os.path.join(folder_name, "result.txt")
                with open(summary_file, "w") as f:
                    f.write(result_str + "\n")
                    f.write(f"Config: Envs={TOTAL_ENVS}, Res={FRAME_WIDTH}x{FRAME_HEIGHT}, Steps={PROFILING_STEPS}\n")
                    f.write(f"Pipeline: {'Async' if args.use_async else 'Sync'}\n")

    finally:
        if pipeline:
            pipeline.stop()
        
        stop_flag.value = 1
        for p in workers:
            p.terminate()
            p.join()
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
