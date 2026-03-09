"""
Async Pipeline Architecture Benchmark
======================================
对应新架构（update_async / record_next_nowait / submit_next / is_slot_ready）的性能测试。

新架构并行结构：
  Main Thread
  ├─ [Physics Dispatch]   通过 buffer_cond 通知 Workers 写 shm
  ├─ [update_async()]     从 shm 读取场景数据，上传至 GPU（transfer queue，非阻塞）
  ├─ [record_next_nowait()] 录制 GPU 命令并分配渲染 slot（不阻塞等前帧 fence）
  ├─ [submit_next()]      提交 GPU 任务（非阻塞）
  └─ [is_slot_ready()]    非阻塞轮询 GPU 渲染完成

  GPU渲染(N帧) 与 CPU录制(N+1帧) 真正并行！

  Readback Thread（内部）
  └─ 独立线程异步回读 GPU 结果 → slot.state 归 FREE

  Worker Processes（每个 env 一个进程）
  └─ Physics Step → 写入 shm[buf_idx] → 通知 Main
"""

import time
import multiprocessing
from multiprocessing import shared_memory, Value, Array, Condition
import os
import sys
import ctypes
import gc
import numpy as np
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- NVTX ---
try:
    import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False

class NvtxAnnotate:
    def __init__(self, message, color="green"):
        self.message = message
        self.color = color
    def __enter__(self):
        if NVTX_AVAILABLE:
            self.range = nvtx.start_range(message=self.message, color=self.color)
    def __exit__(self, *a):
        if NVTX_AVAILABLE:
            nvtx.end_range(self.range)

# --- Library ---
BUILD_LIB_PATH = "/home/hpf/projects/mujoco/mujoco/build/lib"
if os.path.exists(BUILD_LIB_PATH):
    sys.path.append(BUILD_LIB_PATH)

try:
    import mjb
    MJB_AVAILABLE = True
except ImportError:
    MJB_AVAILABLE = False
    print("[Benchmark] Warning: mjb not found, using stub.")
    class _MjbStub:
        class BatchRendererConfig:
            batch_size = 1; frame_width = 64; frame_height = 64; gpu_id = 0
        class BatchRenderer:
            def __init__(self, *a, **kw):
                self._slot = 0
            def start_readback_thread(self): pass
            def stop_readback_thread(self): pass
            def update_async(self, *a): time.sleep(0.001); return True
            def record_next_nowait(self, *a):
                s = self._slot; self._slot = (self._slot + 1) % 3; return s
            def submit_next(self): time.sleep(0.0005); return True
            def is_slot_ready(self, s): time.sleep(0.003); return True
            def wait_slot(self, s): return True
            def get_image(self, *a): return np.zeros((64,64,4), dtype=np.uint8)
    mjb = _MjbStub()

import robosuite
from robosuite.utils.binding_utils_mjb import get_render_packet_dtype

# --- Buffer States ---
BUF_FREE    = 0
BUF_WRITING = 1
BUF_READY   = 2
BUF_READING = 3

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
class EventLogger:
    def __init__(self, filename, lock):
        self.filename = filename
        self.lock = lock

    def init_file(self):
        with self.lock:
            with open(self.filename, "w") as f:
                f.write("process,event,start_time,end_time,step,buffer_id,slot\n")

    def log(self, process_name, event_name, start, end, step=-1, buffer_id=-1, slot=-1):
        with self.lock:
            with open(self.filename, "a") as f:
                f.write(f"{process_name},{event_name},{start:.9f},{end:.9f},{step},{buffer_id},{slot}\n")

# ─────────────────────────────────────────────────────────────
# CPU Affinity
# ─────────────────────────────────────────────────────────────
def allocate_worker_cores(env_id, total_workers, start_core, end_core):
    available = end_core - start_core + 1
    if available <= 0:
        return set()
    assigned = start_core + (env_id % available)
    return {assigned}

def get_mujoco_address(model_obj):
    if hasattr(model_obj, "get_model"):  model_obj = model_obj.get_model()
    if hasattr(model_obj, "_model"):     model_obj = model_obj._model
    if hasattr(model_obj, "_address"):   return model_obj._address
    if hasattr(model_obj, "_model_ptr"):
        ptr = model_obj._model_ptr
        return int(ptr) if isinstance(ptr, int) else ctypes.cast(ptr, ctypes.c_void_p).value
    raise ValueError("Could not extract address")

# ─────────────────────────────────────────────────────────────
# Worker Process
# ─────────────────────────────────────────────────────────────
def worker_fn(env_idx, log_lock, step_counter,
              buffer_state, buffer_done_count, stop_flag, buffer_cond,
              cfg):
    total_envs   = cfg["total_envs"]
    frame_width  = cfg["frame_width"]
    frame_height = cfg["frame_height"]
    shm_name     = cfg["shm_name"]
    num_buffers  = cfg["num_buffers"]
    max_geoms    = cfg["max_geoms"]
    log_file     = cfg["log_file"]
    cpu_start    = cfg["worker_cpu_start"]
    cpu_end      = cfg["worker_cpu_end"]
    cold_start   = cfg["cold_start_steps"]

    os.environ["OMP_NUM_THREADS"] = "1"
    cores = allocate_worker_cores(env_idx, total_envs, cpu_start, cpu_end)
    try:
        if cores:
            os.sched_setaffinity(0, cores)
    except Exception:
        pass

    gc.disable()
    logger = EventLogger(log_file, log_lock)

    env = robosuite.make(
        env_name="TwoArmLift", robots=["UR5e","UR5e"],
        camera_names=["frontview","robot0_eye_in_hand","robot1_eye_in_hand"],
        camera_heights=frame_height, camera_widths=frame_width,
        has_offscreen_renderer=True, ignore_done=True, use_camera_obs=False,
        shared_memory_name=shm_name, env_index=env_idx,
        num_buffers=num_buffers, total_envs=total_envs, max_geoms=max_geoms,
    )
    env.reset()
    low, high = env.action_spec
    local_last_step = -1

    try:
        while not stop_flag.value:
            target_step    = local_last_step + 1
            target_buf_idx = target_step % num_buffers

            with buffer_cond:
                while True:
                    if stop_flag.value: return
                    if (step_counter.value >= target_step and
                            buffer_state[target_buf_idx] == BUF_WRITING):
                        break
                    buffer_cond.wait(timeout=0.001)

            env.sim._render_context_offscreen.set_buffer_id(target_buf_idx)

            t0 = time.perf_counter()
            with NvtxAnnotate(f"[W{env_idx}] Physics", color="green"):
                env.step(np.random.uniform(low, high))
            t1 = time.perf_counter()

            if target_step > cold_start:
                logger.log(f"Worker_{env_idx}", "Physics", t0, t1, target_step, target_buf_idx)

            with buffer_done_count.get_lock():
                buffer_done_count[target_buf_idx] += 1
                done = buffer_done_count[target_buf_idx]

            if done == total_envs:
                buffer_state[target_buf_idx] = BUF_READY
                with buffer_cond:
                    buffer_cond.notify_all()

            local_last_step = target_step
    finally:
        env.close()

# ─────────────────────────────────────────────────────────────
# Timeline Plot  —  CPU/GPU 双泳道并行可视化版本
# ─────────────────────────────────────────────────────────────
def plot_async_timeline(csv_file, output_image="async_timeline.png",
                        step_lo=None, step_hi=None):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[Plot] Cannot read {csv_file}: {e}")
        return

    if df.empty:
        print("[Plot] Empty log, skip.")
        return

    # ── 步骤筛选 ──────────────────────────────────────────────
    if step_lo is not None and step_hi is not None:
        df = df[df["step"].between(step_lo, step_hi)].copy()
    if df.empty:
        print("[Plot] No data in selected step range.")
        return

    # 时间归零
    t0_abs = df["start_time"].min()
    df["start_time"] -= t0_abs
    df["end_time"]   -= t0_abs

    # ── 颜色方案 ──────────────────────────────────────────────
    STAGE_COLORS = {
        "update_async" : "#4FC3F7",   # 浅蓝
        "record_next"  : "#81C784",   # 绿
        "submit_next"  : "#FFB74D",   # 橙
        "wait_slot"    : "#E57373",   # 红（轮询检测到完成的时刻点）
        "gpu_render"   : "#CE93D8",   # 紫：GPU执行窗口（submit→is_slot_ready）
        "Dispatch"     : "#B0BEC5",   # 灰
    }
    WORKER_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", -1: "#95a5a6"}

    # ── 泳道排序 ──────────────────────────────────────────────
    # 期望从上到下：Main_GPU → Main_CPU → Worker_N … Worker_0
    LANE_ORDER = {"Main_GPU": 0, "Main_CPU": 1}
    def sort_key(name):
        if name in LANE_ORDER:
            return LANE_ORDER[name]
        if name.startswith("Worker_"):
            return 2 + int(name.split("_")[1])
        return 999

    procs_asc = sorted(df["process"].unique(), key=sort_key)
    procs_display = list(reversed(procs_asc))   # 图上从上往下：GPU在最上

    fig, ax = plt.subplots(figsize=(24, max(6, len(procs_display) * 0.7 + 2)))
    plt.style.use("seaborn-v0_8-whitegrid")

    legend_handles = {}

    for row_i, proc in enumerate(procs_display):
        sub = df[df["process"] == proc]
        for _, row in sub.iterrows():
            evt    = row["event"]
            buf_id = int(row["buffer_id"]) if pd.notna(row.get("buffer_id", float("nan"))) else -1
            slot   = int(row["slot"])       if pd.notna(row.get("slot",      float("nan"))) else -1
            dur    = row["end_time"] - row["start_time"]
            if dur <= 0:
                continue

            # ── 颜色 & 样式 ───────────────────────────────────
            if proc in ("Main_CPU", "Main_GPU"):
                c     = STAGE_COLORS.get(evt, "#90A4AE")
                alpha = 0.55 if proc == "Main_GPU" else 0.92
                elw   = 1.2  if proc == "Main_GPU" else 0.4
            else:
                c     = WORKER_COLORS.get(buf_id % 4, "#95a5a6")
                alpha = 0.88
                elw   = 0.4

            ax.broken_barh(
                [(row["start_time"], dur)],
                (row_i - 0.4, 0.8),
                facecolors=c, edgecolor="white",
                linewidth=elw, alpha=alpha,
            )

            # ── 文字标注 ──────────────────────────────────────
            if proc == "Main_GPU" and evt == "gpu_render" and dur > 0.002:
                ax.text(row["start_time"] + dur / 2, row_i,
                        f"GPU s{slot}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold")

            if proc == "Main_CPU" and evt == "record_next" and slot >= 0 and dur > 0.0005:
                ax.text(row["start_time"] + dur / 2, row_i,
                        f"rec s{slot}", ha="center", va="center",
                        fontsize=5.5, color="white")

            if proc == "Main_CPU" and evt == "wait_slot" and dur > 0.0002:
                ax.text(row["start_time"] + dur / 2, row_i,
                        "✓", ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

            key = evt if proc in ("Main_CPU", "Main_GPU") else f"Worker buf={buf_id % 4}"
            legend_handles.setdefault(key, c)

    # ── 垂直虚线：每帧 submit 时刻 ───────────────────────────
    sub_df = df[(df["process"] == "Main_CPU") & (df["event"] == "submit_next")]
    for _, row in sub_df.iterrows():
        ax.axvline(x=row["end_time"], color="#FFB74D",
                   linewidth=0.7, linestyle=":", alpha=0.55)

    # ── 高亮 CPU‖GPU 重叠区域 ─────────────────────────────────
    gpu_df = df[(df["process"] == "Main_GPU") & (df["event"] == "gpu_render")]
    rec_df = df[(df["process"] == "Main_CPU") & (df["event"] == "record_next")]

    overlap_labeled = False
    for _, gpu_row in gpu_df.iterrows():
        for _, rec_row in rec_df.iterrows():
            # record_next 的 step 应比 gpu_render 的 step 大 1
            if int(rec_row["step"]) - int(gpu_row["step"]) != 1:
                continue
            ov_s = max(gpu_row["start_time"], rec_row["start_time"])
            ov_e = min(gpu_row["end_time"],   rec_row["end_time"])
            if ov_e > ov_s:
                ax.axvspan(ov_s, ov_e, alpha=0.13, color="lime")
                ax.text((ov_s + ov_e) / 2, len(procs_display) - 0.05,
                        "‖ parallel ‖",
                        ha="center", va="bottom",
                        fontsize=6, color="darkgreen", alpha=0.85)
                overlap_labeled = True

    # ── 轴 & 图例 ─────────────────────────────────────────────
    ax.set_yticks(range(len(procs_display)))
    ax.set_yticklabels(procs_display, fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title(
        f"Async Pipeline Timeline — CPU/GPU Overlap View  (steps {step_lo}–{step_hi})\n"
        f"Main_GPU（紫，半透明）: GPU渲染窗口  "
        f"Main_CPU: CPU操作序列  "
        f"绿色区域: CPU∥GPU真实并行重叠",
        fontsize=10,
    )

    patches = [mpatches.Patch(color=c, label=lbl) for lbl, c in legend_handles.items()]
    if overlap_labeled:
        patches.append(mpatches.Patch(color="lime", alpha=0.4, label="CPU‖GPU overlap"))
    ax.legend(handles=patches, loc="upper right", fontsize=7, ncol=3)

    plt.tight_layout()
    plt.savefig(output_image, dpi=130)
    print(f"[Plot] Saved → {output_image}")


# ─────────────────────────────────────────────────────────────
# Pipeline Overlap 示意图（静态架构图）
# ─────────────────────────────────────────────────────────────
def plot_pipeline_diagram(output_image="pipeline_diagram.png"):
    fig, axes = plt.subplots(2, 1, figsize=(18, 8))

    stages_old = [
        (0, 0, 3, "Physics\n(Workers)", "#2ecc71"),
        (0, 3, 3, "render_from_shm\n(sync GPU)", "#e74c3c"),
        (0, 6, 3, "Physics", "#2ecc71"),
        (0, 9, 3, "render_from_shm", "#e74c3c"),
        (0,12, 3, "Physics", "#2ecc71"),
        (0,15, 3, "render_from_shm", "#e74c3c"),
    ]

    stages_new = [
        # Workers
        (0, 0, 3, "Physics\n(Workers)", "#2ecc71"),
        (0, 3, 3, "Physics", "#2ecc71"),
        (0, 6, 3, "Physics", "#2ecc71"),
        (0, 9, 3, "Physics", "#2ecc71"),
        # Main CPU async stages
        (1, 0, 1, "update\nasync", "#4FC3F7"),
        (1, 1, 1, "record\nnext",  "#81C784"),
        (1, 2, 1, "submit\nnext",  "#FFB74D"),
        (1, 3, 1, "update\nasync", "#4FC3F7"),
        (1, 4, 1, "record\nnext",  "#81C784"),
        (1, 5, 1, "submit\nnext",  "#FFB74D"),
        (1, 6, 1, "update\nasync", "#4FC3F7"),
        (1, 7, 1, "record\nnext",  "#81C784"),
        (1, 8, 1, "submit\nnext",  "#FFB74D"),
        # GPU render（与下一帧CPU并行）
        (2, 2, 2, "GPU render\n(slot 0)", "#CE93D8"),
        (2, 5, 2, "GPU render\n(slot 1)", "#CE93D8"),
        (2, 8, 2, "GPU render\n(slot 2)", "#CE93D8"),
        # Readback thread
        (3, 4, 2, "Readback\n(slot 0)", "#9b59b6"),
        (3, 7, 2, "Readback\n(slot 1)", "#9b59b6"),
        (3,10, 2, "Readback\n(slot 2)", "#9b59b6"),
    ]

    lane_labels_old = {0: "Main / Workers"}
    lane_labels_new = {
        0: "Workers (Physics)",
        1: "Main_CPU (Async Stages)",
        2: "Main_GPU (GPU Render)",
        3: "Readback Thread",
    }

    for ax, stages, lane_labels, title in [
        (axes[0], stages_old, lane_labels_old, "旧架构：同步渲染（Physics → GPU 串行）"),
        (axes[1], stages_new, lane_labels_new, "新架构：细粒度异步流水线（CPU录制 ‖ GPU渲染）"),
    ]:
        for (lane, xs, w, lbl, clr) in stages:
            is_gpu = (clr == "#CE93D8")
            ax.broken_barh([(xs, w - 0.08)], (lane - 0.4, 0.8),
                           facecolors=clr, edgecolor="white",
                           linewidth=0.5, alpha=0.55 if is_gpu else 0.88)
            if w >= 0.8:
                ax.text(xs + w/2, lane, lbl, ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

        # 在新架构图上标注并行重叠区
        if len(lane_labels) == 4:
            for overlap_x, overlap_w in [(3, 1), (6, 1), (9, 1)]:
                ax.axvspan(overlap_x, overlap_x + overlap_w,
                           alpha=0.12, color="lime", ymin=0.25, ymax=0.75)

        ax.set_yticks(sorted(lane_labels.keys()))
        ax.set_yticklabels([lane_labels[k] for k in sorted(lane_labels.keys())], fontsize=9)
        ax.set_xlim(0, 18)
        ax.set_xlabel("Time Slots →", fontsize=9)
        ax.set_title(title, fontsize=11, pad=6)
        ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout(pad=2)
    plt.savefig(output_image, dpi=130)
    print(f"[Plot] Architecture diagram saved → {output_image}")

# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Async Pipeline Benchmark (fine-grained parallel)")
    p.add_argument("--num_envs",    type=int,  default=64)
    p.add_argument("--width",       type=int,  default=1024)
    p.add_argument("--height",      type=int,  default=1024)
    p.add_argument("--steps",       type=int,  default=200,  help="Total render steps")
    p.add_argument("--cold_start",  type=int,  default=10,   help="Warm-up steps ignored in stats")
    p.add_argument("--num_buffers", type=int,  default=3)
    p.add_argument("--max_geoms",   type=int,  default=1000)
    p.add_argument("--plot",        action="store_true", help="Generate timeline & architecture plots")
    p.add_argument("--output_dir",  type=str,  default="")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    TOTAL_ENVS    = args.num_envs
    FRAME_W       = args.width
    FRAME_H       = args.height
    TOTAL_STEPS   = args.steps
    COLD_START    = args.cold_start
    NUM_BUFFERS   = args.num_buffers
    MAX_GEOMS     = args.max_geoms
    MAX_LIGHTS    = 10
    SHM_NAME      = "async_pipe_bench_shm"

    folder = args.output_dir or f"results_async_env{TOTAL_ENVS}_{FRAME_W}x{FRAME_H}"
    os.makedirs(folder, exist_ok=True)
    LOG_FILE     = os.path.join(folder, "benchmark_async.csv")
    TIMELINE_IMG = os.path.join(folder, f"timeline_async_env{TOTAL_ENVS}_{FRAME_W}x{FRAME_H}.png")
    DIAG_IMG     = os.path.join(folder, "pipeline_diagram.png")
    RESULT_TXT   = os.path.join(folder, "result.txt")

    TOTAL_CORES  = multiprocessing.cpu_count()
    MAIN_CORES   = set(range(max(0, TOTAL_CORES - 4), TOTAL_CORES))
    WORKER_CPU_S = 0
    WORKER_CPU_E = max(0, TOTAL_CORES - 5)

    cfg = dict(
        total_envs=TOTAL_ENVS, frame_width=FRAME_W, frame_height=FRAME_H,
        shm_name=SHM_NAME, num_buffers=NUM_BUFFERS, max_geoms=MAX_GEOMS,
        log_file=LOG_FILE, worker_cpu_start=WORKER_CPU_S, worker_cpu_end=WORKER_CPU_E,
        cold_start_steps=COLD_START,
    )

    multiprocessing.set_start_method("spawn", force=True)
    print(f"[Bench] Envs={TOTAL_ENVS}  Res={FRAME_W}x{FRAME_H}  Steps={TOTAL_STEPS}  "
          f"Buffers={NUM_BUFFERS}  ColdStart={COLD_START}")

    os.sched_setaffinity(0, MAIN_CORES)

    # ── Shared primitives ───────────────────────────────────
    log_lock        = multiprocessing.Lock()
    step_counter    = multiprocessing.Value("i", 0)
    stop_flag       = multiprocessing.Value("i", 0)
    buffer_state    = Array("i", [BUF_FREE] * NUM_BUFFERS)
    buffer_done_cnt = Array("i", [0]        * NUM_BUFFERS)
    buffer_cond     = Condition()
    buffer_state[0] = BUF_WRITING

    EventLogger(LOG_FILE, log_lock).init_file()

    # ── Shared Memory ───────────────────────────────────────
    packet_dtype = get_render_packet_dtype(MAX_GEOMS, MAX_LIGHTS)
    pkt_size     = packet_dtype.itemsize
    total_shm    = NUM_BUFFERS * TOTAL_ENVS * pkt_size

    try:
        old = shared_memory.SharedMemory(name=SHM_NAME)
        old.close(); old.unlink()
    except Exception:
        pass

    shm = shared_memory.SharedMemory(create=True, size=total_shm, name=SHM_NAME)

    buf_addrs = []
    for bi in range(NUM_BUFFERS):
        offset = bi * TOTAL_ENVS * pkt_size
        arr    = np.ndarray((TOTAL_ENVS, pkt_size), dtype=np.uint8,
                            buffer=shm.buf, offset=offset)
        buf_addrs.append(arr.ctypes.data)

    # ── Model Pointers ──────────────────────────────────────
    print("[Bench] Initializing environments for model pointers...")
    model_ptrs = []
    kept_envs  = []
    for i in range(TOTAL_ENVS):
        dummy = robosuite.make(
            env_name="TwoArmLift", robots=["UR5e","UR5e"],
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        )
        kept_envs.append(dummy)
        model_ptrs.append(get_mujoco_address(dummy.sim.model))

    # ── BatchRenderer ───────────────────────────────────────
    mjb_cfg = mjb.BatchRendererConfig()
    mjb_cfg.batch_size   = TOTAL_ENVS
    mjb_cfg.frame_width  = FRAME_W
    mjb_cfg.frame_height = FRAME_H
    mjb_cfg.gpu_id       = 0
    renderer = mjb.BatchRenderer(model_ptrs, mjb_cfg)
    renderer.start_readback_thread()

    # ── Spawn Workers ───────────────────────────────────────
    workers = []
    for i in range(TOTAL_ENVS):
        p = multiprocessing.Process(
            target=worker_fn,
            args=(i, log_lock, step_counter,
                  buffer_state, buffer_done_cnt, stop_flag, buffer_cond,
                  cfg),
        )
        p.start()
        workers.append(p)

    logger = EventLogger(LOG_FILE, log_lock)

    # ─────────────────────────────────────────────────────────
    # Main Loop  —  细粒度并行版本
    #
    # 流水线状态变量：
    #   pending_slot      : 已 submit、GPU 正在渲染的 slot 索引
    #   pending_step      : pending_slot 对应的渲染步骤编号
    #   pending_buf       : pending_slot 对应的 shm buffer 索引
    #   pending_gpu_start : submit_next 完成时刻（GPU 开始渲染的近似时间）
    #
    #   cur_slot          : 已 record、待 submit 的 slot 索引
    #   cur_read_buf      : cur_slot 对应的 shm buffer 索引
    #   cur_step          : cur_slot 对应的渲染步骤编号
    #
    # 每轮循环尝试推进以下四个阶段（顺序不阻塞彼此）：
    #   A. 若 cur_slot 就绪且无 pending → submit_next()
    #   B. 若 cur_slot 为空且 shm 有 READY buf → update_async + record_next_nowait
    #   C. 若有 pending_slot → is_slot_ready() 非阻塞轮询，完成则释放 buffer
    #   D. 若下一个 buf 为 FREE → 派发物理步骤给 Workers
    # ─────────────────────────────────────────────────────────
    prof_start = None
    rendered   = -1       # 已完成 GPU 渲染的帧数（is_slot_ready 确认后计数）
    dispatched = 0        # 已派发给 Workers 的物理步骤数

    step_latencies = []   # 逐帧端到端延迟（update_async 开始 → is_slot_ready 完成）

    # 流水线状态
    pending_slot      = -1
    pending_step      = -1
    pending_buf       = -1
    pending_gpu_start = 0.0   # submit_next 完成时刻，作为 GPU 开始时间的近似

    cur_slot     = -1
    cur_read_buf = -1
    cur_step     = -1
    cur_frame_t0 = 0.0        # 当前帧 update_async 开始时刻（用于端到端延迟）

    try:
        while rendered < TOTAL_STEPS:
            did_work = False

            # ── 阶段 A：提交已录制的帧 ──────────────────────────
            # 条件：有已录制的 slot，且 GPU 当前无正在飞行的帧（避免超过 SWAP_COUNT）
            if cur_slot >= 0 and pending_slot < 0:
                t0 = time.perf_counter()
                with NvtxAnnotate("submit_next", color="orange"):
                    renderer.submit_next()
                t1 = time.perf_counter()

                if cur_step > COLD_START:
                    logger.log("Main_CPU", "submit_next",
                               t0, t1, cur_step, cur_read_buf, cur_slot)

                # 转移到 pending 状态，记录 GPU 开始时刻
                pending_slot      = cur_slot
                pending_step      = cur_step
                pending_buf       = cur_read_buf
                pending_gpu_start = t1     # GPU 实际开始略晚，这里用 submit 完成时刻近似

                cur_slot     = -1
                cur_read_buf = -1
                cur_step     = -1

                did_work = True

            # ── 阶段 B：录制下一帧（与 GPU 渲染并行！）────────────
            # 条件：当前没有待 submit 的帧，且 shm 中有 READY 的 buffer
            if cur_slot < 0:
                read_buf = -1
                for bi in range(NUM_BUFFERS):
                    if buffer_state[bi] == BUF_READY:
                        buffer_state[bi] = BUF_READING
                        read_buf = bi
                        break

                if read_buf != -1:
                    # step 编号：若有 pending，则当前录制的是 pending+1 帧
                    this_step = rendered + (2 if pending_slot >= 0 else 1)
                    frame_t0  = time.perf_counter()

                    # ① update_async：上传 UBO 到 GPU（transfer queue，信号量同步，非阻塞返回）
                    t0 = time.perf_counter()
                    with NvtxAnnotate("update_async", color="cyan"):
                        renderer.update_async(buf_addrs[read_buf], MAX_GEOMS, MAX_LIGHTS)
                    t1 = time.perf_counter()
                    if dispatched > COLD_START:
                        logger.log("Main_CPU", "update_async",
                                   t0, t1, this_step, read_buf)

                    # ② record_next_nowait：非阻塞录制；slot 未就绪时返回 -1，短暂自旋等待
                    slot = -1
                    while slot < 0:
                        t0 = time.perf_counter()
                        with NvtxAnnotate("record_next_nowait", color="green"):
                            slot = renderer.record_next_nowait(buf_addrs[read_buf])
                        if slot < 0:
                            # 所有 swap slot 仍被 GPU 占用，让出 50μs 后重试
                            time.sleep(0.00005)
                    t1 = time.perf_counter()
                    if dispatched > COLD_START:
                        logger.log("Main_CPU", "record_next",
                                   t0, t1, this_step, read_buf, slot)

                    cur_slot     = slot
                    cur_read_buf = read_buf
                    cur_step     = this_step
                    cur_frame_t0 = frame_t0

                    did_work = True

            # ── 阶段 C：非阻塞轮询上一帧 GPU 是否完成 ──────────
            if pending_slot >= 0:
                if renderer.is_slot_ready(pending_slot):
                    gpu_done_t = time.perf_counter()

                    # 记录 GPU 渲染窗口（submit完成 → is_slot_ready 确认）
                    if pending_step > COLD_START:
                        logger.log("Main_GPU", "gpu_render",
                                   pending_gpu_start, gpu_done_t,
                                   pending_step, pending_buf, pending_slot)
                        # wait_slot 事件：用极短的时间点表示"轮询确认"时刻
                        logger.log("Main_CPU", "wait_slot",
                                   gpu_done_t, gpu_done_t + 1e-6,
                                   pending_step, pending_buf, pending_slot)

                    # 释放 shm buffer，通知 Workers 可以写下一步
                    buffer_state[pending_buf] = BUF_FREE
                    rendered += 1

                    if rendered == COLD_START:
                        prof_start = time.perf_counter()
                        print(f"[Bench] Cold start done at step {rendered}. Profiling started.")

                    if rendered > COLD_START:
                        # 端到端延迟：从当前帧 update_async 开始到 GPU 完成
                        # 注意：cur_frame_t0 是"当前正在录制帧"的 t0，
                        # pending 帧的 t0 需要在 submit 前保存，这里用 pending_gpu_start 近似
                        step_latencies.append(gpu_done_t - pending_gpu_start)

                    with buffer_cond:
                        buffer_cond.notify_all()

                    pending_slot      = -1
                    pending_buf       = -1
                    pending_step      = -1
                    pending_gpu_start = 0.0

                    did_work = True

            # ── 阶段 D：派发下一步物理给 Workers ──────────────
            nxt     = dispatched + 1
            nxt_buf = nxt % NUM_BUFFERS
            if nxt <= TOTAL_STEPS and buffer_state[nxt_buf] == BUF_FREE:
                t0 = time.perf_counter()
                buffer_done_cnt[nxt_buf] = 0
                buffer_state[nxt_buf]    = BUF_WRITING
                dispatched               = nxt
                step_counter.value       = dispatched
                with buffer_cond:
                    buffer_cond.notify_all()
                t1 = time.perf_counter()
                if dispatched > COLD_START:
                    logger.log("Main_CPU", "Dispatch", t0, t1, dispatched, nxt_buf)
                did_work = True

            if not did_work:
                time.sleep(0.00005)   # 50μs 空转，避免 busy-wait 占满 CPU

    finally:
        # 安全退出：确保 pending slot 的 GPU 工作也完成
        if pending_slot >= 0:
            renderer.wait_slot(pending_slot)
        renderer.stop_readback_thread()
        stop_flag.value = 1
        with buffer_cond:
            buffer_cond.notify_all()
        for p in workers:
            p.terminate()
            p.join()
        shm.close()
        shm.unlink()

    # ─────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────
    prof_end = time.perf_counter()

    if prof_start is not None and step_latencies:
        total_dur   = prof_end - prof_start
        valid_steps = len(step_latencies)
        avg_lat     = np.mean(step_latencies)
        p50         = np.percentile(step_latencies, 50)
        p95         = np.percentile(step_latencies, 95)
        p99         = np.percentile(step_latencies, 99)
        fps         = valid_steps / total_dur

        lines = [
            "=" * 60,
            f"  Fine-Grained Async Pipeline Benchmark Results",
            "=" * 60,
            f"  Config : Envs={TOTAL_ENVS}  Res={FRAME_W}x{FRAME_H}  Buffers={NUM_BUFFERS}",
            f"  Valid Steps       : {valid_steps}",
            f"  Total Duration    : {total_dur:.4f} s",
            f"  FPS               : {fps:.2f}",
            f"  Avg Step Latency  : {avg_lat*1000:.3f} ms",
            f"  P50 Latency       : {p50*1000:.3f} ms",
            f"  P95 Latency       : {p95*1000:.3f} ms",
            f"  P99 Latency       : {p99*1000:.3f} ms",
            "=" * 60,
        ]
        output = "\n".join(lines)
        print(output)
        with open(RESULT_TXT, "w") as f:
            f.write(output + "\n")

        # ── Per-stage breakdown ──────────────────────────────
        try:
            df = pd.read_csv(LOG_FILE)

            # Main_CPU 各阶段耗时
            cpu_df = df[df["process"] == "Main_CPU"].copy()
            cpu_df["dur"] = cpu_df["end_time"] - cpu_df["start_time"]
            cpu_bd = cpu_df.groupby("event")["dur"].agg(["mean","std","max"]) * 1000
            cpu_bd.columns = ["mean_ms","std_ms","max_ms"]

            # Main_GPU GPU 渲染窗口耗时
            gpu_df = df[df["process"] == "Main_GPU"].copy()
            gpu_df["dur"] = gpu_df["end_time"] - gpu_df["start_time"]
            gpu_bd = gpu_df.groupby("event")["dur"].agg(["mean","std","max"]) * 1000
            gpu_bd.columns = ["mean_ms","std_ms","max_ms"]

            summary_str = (
                "\nPer-Stage Breakdown — Main_CPU (ms):\n" + cpu_bd.to_string() +
                "\n\nGPU Render Window — Main_GPU (ms):\n" + gpu_bd.to_string()
            )
            print(summary_str)
            with open(RESULT_TXT, "a") as f:
                f.write(summary_str + "\n")

            # ── 并行重叠率统计 ────────────────────────────────
            # 计算 gpu_render 与 record_next 的实际重叠时间占 GPU 时间的比例
            try:
                gr = df[(df["process"] == "Main_GPU") & (df["event"] == "gpu_render")]
                rn = df[(df["process"] == "Main_CPU") & (df["event"] == "record_next")]
                total_gpu_time = 0.0
                total_overlap  = 0.0
                for _, grow in gr.iterrows():
                    total_gpu_time += grow["end_time"] - grow["start_time"]
                    for _, rrow in rn.iterrows():
                        if int(rrow["step"]) - int(grow["step"]) == 1:
                            ov = min(grow["end_time"], rrow["end_time"]) \
                               - max(grow["start_time"], rrow["start_time"])
                            if ov > 0:
                                total_overlap += ov
                if total_gpu_time > 0:
                    overlap_pct = 100.0 * total_overlap / total_gpu_time
                    ol_str = f"\nCPU‖GPU Overlap Ratio: {overlap_pct:.1f}% of GPU time"
                    print(ol_str)
                    with open(RESULT_TXT, "a") as f:
                        f.write(ol_str + "\n")
            except Exception as e:
                print(f"[Bench] Overlap ratio failed: {e}")

        except Exception as e:
            print(f"[Bench] Stage breakdown failed: {e}")
    else:
        print("[Bench] Not enough data for statistics.")

    # ─────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────
    if args.plot:
        plot_async_timeline(
            LOG_FILE, TIMELINE_IMG,
            step_lo=COLD_START + 5,
            step_hi=COLD_START + 35,
        )
        plot_pipeline_diagram(DIAG_IMG)

    print(f"\n[Bench] All outputs saved to: {folder}/")


if __name__ == "__main__":
    main()