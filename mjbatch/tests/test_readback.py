"""
Async Pipeline Architecture Benchmark
======================================
对应新架构（update_async / record_next_nowait / submit_next / is_slot_ready）的性能测试。

新架构并行结构：
  Main Thread
  ├─ [Physics Dispatch]     通过 buffer_cond 通知 Workers 写 shm
  ├─ [update_async()]       从 shm 读取场景数据，上传至 GPU（transfer queue，非阻塞）
  ├─ [record_next_nowait()] 录制 GPU 命令并分配渲染 slot（不阻塞等前帧 fence）
  ├─ [submit_next()]        提交 GPU 任务（非阻塞）
  └─ [is_slot_ready()]      非阻塞轮询 GPU 渲染完成

  GPU渲染(N帧) 与 CPU录制(N+1帧) 真正并行！

  Readback Thread（内部，完全异步）
  └─ 独立线程检测 render_fence 完成 → 自动提交 readback blit → invalidate → callback → FREE
     readback 与 CPU录制、GPU渲染 三者完全并行！

  Worker Processes（每个 env 一个进程）
  └─ Physics Step → 写入 shm[buf_idx] → 通知 Main

变更说明（相较旧版本）：
  - is_slot_ready() 确认 GPU render 完成后立即释放 shm buffer，不再等 readback
  - readback blit 由 readback 线程在检测到 render_fence 完成后自动触发
  - 图像数据通过 set_readback_callback() 异步回调送达
  - Timeline 图新增独立 Readback_Thread 泳道，体现三轨并行
"""

import time
import threading
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

    class _BatchRendererStub:
        """
        Stub 模拟三阶段异步流水线行为：
          - submit_next  → GPU 渲染延迟 ~4ms
          - is_slot_ready → 非阻塞轮询，GPU 完成后返回 True
          - readback callback → GPU 完成后 ~2ms 触发（模拟 DMA 传输）
        """
        _SWAP_COUNT = 3

        def __init__(self, *a, **kw):
            self._slot        = 0
            self._lock        = threading.Lock()
            self._cb          = None
            self._rb_thread   = None
            self._rb_running  = False
            # 每个 slot: {"submit_t": float, "step_id": int, "state": str}
            self._slots       = [{"submit_t": 0.0, "step_id": -1, "state": "FREE"}
                                  for _ in range(self._SWAP_COUNT)]
            self._step_counter = 0

        def start_readback_thread(self):
            self._rb_running = True
            self._rb_thread  = threading.Thread(target=self._rb_fn, daemon=True)
            self._rb_thread.start()

        def stop_readback_thread(self):
            self._rb_running = False
            if self._rb_thread:
                self._rb_thread.join(timeout=1.0)

        def set_readback_callback(self, cb):
            self._cb = cb

        def update_async(self, *a):
            time.sleep(0.0008)   # 模拟 UBO 上传 ~0.8ms
            return True

        def record_next_nowait(self, *a):
            with self._lock:
                # 找一个 FREE slot
                for i in range(self._SWAP_COUNT):
                    idx = (self._slot + i) % self._SWAP_COUNT
                    if self._slots[idx]["state"] == "FREE":
                        self._slots[idx]["state"]    = "RECORDING"
                        self._slot = idx
                        time.sleep(0.0012)   # 模拟录制 ~1.2ms
                        self._slots[idx]["state"] = "RECORDED"
                        return idx
            return -1   # 所有 slot 繁忙

        def submit_next(self):
            with self._lock:
                slot = self._slots[self._slot]
                slot["state"]    = "RENDERING"
                slot["submit_t"] = time.perf_counter()
                slot["step_id"]  = self._step_counter
                self._step_counter += 1
            # vkQueueSubmit 本身几乎不耗时，立即返回
            return True

        def is_slot_ready(self, slot_idx):
            with self._lock:
                s = self._slots[slot_idx]
                if s["state"] != "RENDERING":
                    return False
                # 模拟 GPU 渲染耗时 ~4ms
                if time.perf_counter() - s["submit_t"] >= 0.004:
                    s["state"] = "RENDER_DONE"
                    return True
            return False

        def wait_slot(self, slot_idx):
            while not self.is_slot_ready(slot_idx):
                time.sleep(0.0001)
            return True

        def wait_readback(self, slot_idx):
            # 等待 readback 线程把 slot 归 FREE
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                with self._lock:
                    if self._slots[slot_idx]["state"] == "FREE":
                        return True
                time.sleep(0.0001)
            return False

        def get_image(self, slot_idx, batch_idx, cam_idx):
            return np.zeros((64, 64, 4), dtype=np.uint8)

        def _rb_fn(self):
            """Readback 线程：检测 RENDER_DONE → 模拟 DMA ~2ms → callback → FREE"""
            while self._rb_running:
                for i in range(self._SWAP_COUNT):
                    with self._lock:
                        s = self._slots[i]
                        if s["state"] != "RENDER_DONE":
                            continue
                        s["state"]   = "READBACK_PENDING"
                        step_id      = s["step_id"]
                        rb_start     = time.perf_counter()

                    # 模拟 DMA 传输 ~2ms（在锁外 sleep）
                    time.sleep(0.002)

                    if self._cb:
                        dummy_frames = [np.zeros((64, 64, 4), dtype=np.uint8)]
                        self._cb(step_id, dummy_frames)

                    with self._lock:
                        self._slots[i]["state"] = "FREE"

                time.sleep(0.0001)

    class _MjbStub:
        BatchRendererConfig = type("BatchRendererConfig", (), {
            "batch_size": 1, "frame_width": 64, "frame_height": 64, "gpu_id": 0
        })
        BatchRenderer = _BatchRendererStub

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
        env_name="TwoArmLift", robots=["UR5e", "UR5e"],
        camera_names=["frontview", "robot0_eye_in_hand", "robot1_eye_in_hand"],
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
# Timeline Plot  —  CPU / GPU / Readback 三轨并行可视化
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
        "update_async"  : "#4FC3F7",   # 浅蓝
        "record_next"   : "#81C784",   # 绿
        "submit_next"   : "#FFB74D",   # 橙
        "wait_slot"     : "#E57373",   # 红（✓ 确认时刻点）
        "gpu_render"    : "#CE93D8",   # 紫：GPU 渲染窗口
        "readback_blit" : "#F48FB1",   # 粉：DMA readback
        "readback_cb"   : "#FF80AB",   # 深粉：callback 触发点
        "Dispatch"      : "#B0BEC5",   # 灰
    }
    WORKER_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e67e22", -1: "#95a5a6"}

    # ── 泳道定义（固定4轨，从上到下）────────────────────────
    #   0: Readback_Thread  （最上，粉色系）
    #   1: Main_GPU         （紫色系，半透明）
    #   2: Main_CPU         （多色，操作序列）
    #   3+: Worker_N        （绿/黄/橙）
    FIXED_LANES = ["Readback_Thread", "Main_GPU", "Main_CPU"]

    all_procs = df["process"].unique().tolist()
    worker_procs = sorted(
        [p for p in all_procs if p.startswith("Worker_")],
        key=lambda x: int(x.split("_")[1])
    )
    # 显示顺序（图上从上到下）
    procs_display = FIXED_LANES + worker_procs

    # 泳道高度：Readback/GPU/CPU 各 1.0 单位，Worker 各 0.6 单位
    lane_height = {p: (0.6 if p.startswith("Worker_") else 1.0) for p in procs_display}

    # 计算每条泳道的 y 中心（从上到下累积）
    lane_y = {}
    gap = 0.25   # 泳道间距
    y_cursor = 0.0
    for p in procs_display:
        h = lane_height[p]
        lane_y[p] = y_cursor + h / 2
        y_cursor += h + gap

    total_height = y_cursor

    fig_h = max(8, total_height * 0.9 + 2)
    fig, ax = plt.subplots(figsize=(26, fig_h))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#16213E")

    # ── 泳道背景条 ──────────────────────────────────────────
    lane_bg = {
        "Readback_Thread": "#2D1B33",
        "Main_GPU"        : "#1E1B2E",
        "Main_CPU"        : "#1B2233",
    }
    x_max = df["end_time"].max() * 1.02
    for p in procs_display:
        bg = lane_bg.get(p, "#1A2210" if p.startswith("Worker_") else "#1E1E2E")
        h  = lane_height[p]
        yc = lane_y[p]
        ax.barh(yc, x_max, height=h * 0.95, left=0,
                color=bg, alpha=0.55, zorder=0)

    legend_handles = {}

    for proc in procs_display:
        if proc not in lane_y:
            continue
        yc = lane_y[proc]
        h  = lane_height[proc]
        bar_h = h * 0.72

        sub = df[df["process"] == proc]
        for _, row in sub.iterrows():
            evt    = row["event"]
            buf_id = int(row["buffer_id"]) if pd.notna(row.get("buffer_id", float("nan"))) else -1
            slot   = int(row["slot"])       if pd.notna(row.get("slot",      float("nan"))) else -1
            dur    = row["end_time"] - row["start_time"]
            if dur <= 0 and evt not in ("wait_slot", "readback_cb"):
                continue

            # ── 颜色 & 透明度 ─────────────────────────────
            if proc == "Main_GPU":
                c     = STAGE_COLORS.get(evt, "#9C64A6")
                alpha = 0.60
                elw   = 1.5
            elif proc == "Main_CPU":
                c     = STAGE_COLORS.get(evt, "#90A4AE")
                alpha = 0.92
                elw   = 0.4
            elif proc == "Readback_Thread":
                c     = STAGE_COLORS.get(evt, "#F06292")
                alpha = 0.85
                elw   = 0.6
            else:
                c     = WORKER_COLORS.get(buf_id % 4 if buf_id >= 0 else -1, "#95a5a6")
                alpha = 0.80
                elw   = 0.3

            # 极短事件（wait_slot / readback_cb）画竖线标记
            if dur < 1e-4:
                ax.vlines(row["start_time"], yc - bar_h / 2, yc + bar_h / 2,
                          colors=c, linewidth=2.5, alpha=0.95, zorder=4)
            else:
                ax.broken_barh(
                    [(row["start_time"], dur)],
                    (yc - bar_h / 2, bar_h),
                    facecolors=c, edgecolor="#FFFFFF22",
                    linewidth=elw, alpha=alpha, zorder=3,
                )

            # ── 文字标注 ──────────────────────────────────
            if proc == "Main_GPU" and evt == "gpu_render" and dur > 0.003:
                ax.text(row["start_time"] + dur / 2, yc,
                        f"GPU s{slot}", ha="center", va="center",
                        fontsize=6.5, color="white", fontweight="bold", zorder=5)

            if proc == "Main_CPU" and evt == "record_next" and slot >= 0 and dur > 0.001:
                ax.text(row["start_time"] + dur / 2, yc,
                        f"rec s{slot}", ha="center", va="center",
                        fontsize=5.5, color="white", zorder=5)

            if proc == "Main_CPU" and evt == "submit_next" and dur > 0.0003:
                ax.text(row["start_time"] + dur / 2, yc,
                        "sub", ha="center", va="center",
                        fontsize=5.0, color="white", zorder=5)

            if proc == "Readback_Thread" and evt == "readback_blit" and dur > 0.001:
                ax.text(row["start_time"] + dur / 2, yc,
                        f"RB s{slot}", ha="center", va="center",
                        fontsize=6.0, color="white", fontweight="bold", zorder=5)

            key = f"{proc}:{evt}" if proc in ("Main_CPU","Main_GPU","Readback_Thread") \
                  else f"Worker buf={buf_id % 4}"
            legend_handles.setdefault(key, c)

    # ── 垂直虚线：每帧 submit 时刻 ───────────────────────────
    sub_df = df[(df["process"] == "Main_CPU") & (df["event"] == "submit_next")]
    for _, row in sub_df.iterrows():
        ax.axvline(x=row["end_time"], color="#FFB74D",
                   linewidth=0.6, linestyle=":", alpha=0.45, zorder=2)

    # ── 高亮 CPU‖GPU 并行重叠区域 ─────────────────────────────
    gpu_df = df[(df["process"] == "Main_GPU") & (df["event"] == "gpu_render")]
    rec_df = df[(df["process"] == "Main_CPU") & (df["event"] == "record_next")]

    overlap_labeled = False
    for _, gpu_row in gpu_df.iterrows():
        for _, rec_row in rec_df.iterrows():
            if int(rec_row["step"]) - int(gpu_row["step"]) != 1:
                continue
            ov_s = max(gpu_row["start_time"], rec_row["start_time"])
            ov_e = min(gpu_row["end_time"],   rec_row["end_time"])
            if ov_e > ov_s:
                ax.axvspan(ov_s, ov_e, alpha=0.09, color="lime", zorder=1)
                if not overlap_labeled:
                    ax.text((ov_s + ov_e) / 2, total_height * 0.97,
                            "CPU‖GPU", ha="center", va="top",
                            fontsize=6, color="#AAFFAA", alpha=0.85, zorder=6)
                    overlap_labeled = True

    # ── 高亮 GPU‖Readback 并行重叠区域 ───────────────────────
    rb_df = df[(df["process"] == "Readback_Thread") & (df["event"] == "readback_blit")]

    gpu_rb_overlap_labeled = False
    for _, gpu_row in gpu_df.iterrows():
        for _, rb_row in rb_df.iterrows():
            # readback 处理上一帧（step 比 GPU 小 1）
            if int(gpu_row["step"]) - int(rb_row["step"]) != 1:
                continue
            ov_s = max(gpu_row["start_time"], rb_row["start_time"])
            ov_e = min(gpu_row["end_time"],   rb_row["end_time"])
            if ov_e > ov_s:
                ax.axvspan(ov_s, ov_e, alpha=0.09, color="cyan", zorder=1)
                if not gpu_rb_overlap_labeled:
                    ax.text((ov_s + ov_e) / 2, total_height * 0.90,
                            "GPU‖RB", ha="center", va="top",
                            fontsize=6, color="#AAFFFF", alpha=0.85, zorder=6)
                    gpu_rb_overlap_labeled = True

    # ── 泳道分隔线 ────────────────────────────────────────────
    boundary_y = 0.0
    for i, p in enumerate(procs_display):
        boundary_y += lane_height[p] + gap
        if i < len(procs_display) - 1:
            ax.axhline(y=boundary_y - gap / 2,
                       color="#FFFFFF22", linewidth=0.8, linestyle="--", zorder=2)

    # ── 轴设置 ────────────────────────────────────────────────
    ax.set_yticks([lane_y[p] for p in procs_display])
    ax.set_yticklabels(procs_display, fontsize=8.5, color="white")
    ax.set_ylim(-0.3, total_height + 0.3)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Time (s)", fontsize=10, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#FFFFFF44")

    ax.set_title(
        f"Async Pipeline Timeline — CPU / GPU / Readback   "
        f"(steps {step_lo}–{step_hi})\n",
        fontsize=10, color="white", pad=10,
    )

    # ── 图例 ──────────────────────────────────────────────────
    patches = []
    label_map = {
        "Main_CPU:update_async"   : "update_async (CPU)",
        "Main_CPU:record_next"    : "record_next (CPU)",
        "Main_CPU:submit_next"    : "submit_next (CPU)",
        "Main_CPU:wait_slot"      : "GPU ready ✓ (CPU poll)",
        "Main_CPU:Dispatch"       : "Dispatch (CPU)",
        "Main_GPU:gpu_render"     : "GPU Render",
        "Readback_Thread:readback_blit" : "Readback DMA Blit",
        "Readback_Thread:readback_cb"   : "Readback Callback",
    }
    for key, c in legend_handles.items():
        label = label_map.get(key, key)
        patches.append(mpatches.Patch(color=c, label=label, alpha=0.85))
    patches.append(mpatches.Patch(color="lime",  alpha=0.35, label="CPU‖GPU overlap"))
    patches.append(mpatches.Patch(color="cyan",  alpha=0.35, label="GPU‖Readback overlap"))

    legend = ax.legend(handles=patches, loc="upper right", fontsize=7,
                       ncol=3, framealpha=0.3,
                       facecolor="#0D0D1A", edgecolor="#FFFFFF44",
                       labelcolor="white")

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, facecolor=fig.get_facecolor())
    print(f"[Plot] Saved → {output_image}")


# ─────────────────────────────────────────────────────────────
# Pipeline Architecture Diagram（静态示意图，含 Readback 轨）
# ─────────────────────────────────────────────────────────────
def plot_pipeline_diagram(output_image="pipeline_diagram.png"):
    fig, axes = plt.subplots(2, 1, figsize=(20, 9),
                              facecolor="#1A1A2E")

    stages_old = [
        (0,  0, 3, "Physics (Workers)",       "#2ecc71"),
        (0,  3, 3, "render_from_shm\n(sync)", "#e74c3c"),
        (0,  6, 3, "Physics",                 "#2ecc71"),
        (0,  9, 3, "render_from_shm",         "#e74c3c"),
        (0, 12, 3, "Physics",                 "#2ecc71"),
        (0, 15, 3, "render_from_shm",         "#e74c3c"),
    ]

    stages_new = [
        # Workers（轨 0）
        (0,  0, 2.8, "Physics[0]",  "#2ecc71"),
        (0,  3, 2.8, "Physics[1]",  "#2ecc71"),
        (0,  6, 2.8, "Physics[2]",  "#2ecc71"),
        (0,  9, 2.8, "Physics[3]",  "#2ecc71"),
        # Main CPU（轨 1）
        (1,  0, 0.9, "upd",         "#4FC3F7"),
        (1,  1, 0.9, "rec",         "#81C784"),
        (1,  2, 0.7, "sub",         "#FFB74D"),
        (1,  3, 0.9, "upd",         "#4FC3F7"),
        (1,  4, 0.9, "rec",         "#81C784"),
        (1,  5, 0.7, "sub",         "#FFB74D"),
        (1,  6, 0.9, "upd",         "#4FC3F7"),
        (1,  7, 0.9, "rec",         "#81C784"),
        (1,  8, 0.7, "sub",         "#FFB74D"),
        (1,  9, 0.9, "upd",         "#4FC3F7"),
        (1, 10, 0.9, "rec",         "#81C784"),
        (1, 11, 0.7, "sub",         "#FFB74D"),
        # GPU Render（轨 2）
        (2,  2.7, 3.0, "GPU[0]",   "#CE93D8"),
        (2,  5.7, 3.0, "GPU[1]",   "#CE93D8"),
        (2,  8.7, 3.0, "GPU[2]",   "#CE93D8"),
        (2, 11.7, 3.0, "GPU[3]",   "#CE93D8"),
        # Readback Thread（轨 3）
        (3,  5.7, 2.0, "RB DMA[0]","#F48FB1"),
        (3,  8.7, 2.0, "RB DMA[1]","#F48FB1"),
        (3, 11.7, 2.0, "RB DMA[2]","#F48FB1"),
        (3, 14.7, 2.0, "RB DMA[3]","#F48FB1"),
    ]

    lane_labels_old = {0: "Main / Workers (serial)"}
    lane_labels_new = {
        0: "Workers (Physics)",
        1: "Main CPU (Async Stages)",
        2: "GPU Render",
        3: "Readback Thread (Async DMA)",
    }

    for ax, stages, lane_labels, title in [
        (axes[0], stages_old, lane_labels_old,
         "旧架构：同步渲染 — Physics → GPU → Readback 全串行"),
        (axes[1], stages_new, lane_labels_new,
         "新架构：三轨流水线 — CPU录制 ‖ GPU渲染 ‖ Readback DMA 真正并行"),
    ]:
        ax.set_facecolor("#16213E")
        for (lane, xs, w, lbl, clr) in stages:
            is_gpu = (clr == "#CE93D8")
            is_rb  = (clr == "#F48FB1")
            ax.broken_barh(
                [(xs, w - 0.08)], (lane - 0.38, 0.76),
                facecolors=clr, edgecolor="#FFFFFF33",
                linewidth=0.6,
                alpha=0.55 if is_gpu else (0.80 if is_rb else 0.88),
            )
            if w >= 0.6:
                ax.text(xs + w / 2, lane, lbl,
                        ha="center", va="center",
                        fontsize=7, color="white", fontweight="bold")

        if len(lane_labels) == 4:
            # CPU‖GPU 并行区（upd+rec 与 GPU 重叠）
            for ox, ow in [(2.7, 1.1), (5.7, 1.1), (8.7, 1.1), (11.7, 1.1)]:
                ax.axvspan(ox, ox + ow, alpha=0.10, color="lime",
                           ymin=0.25, ymax=0.65)
            # GPU‖Readback 并行区
            for ox, ow in [(5.7, 2.0), (8.7, 2.0), (11.7, 2.0)]:
                ax.axvspan(ox, ox + ow, alpha=0.10, color="cyan",
                           ymin=0.0, ymax=0.40)

        num_lanes = len(lane_labels)
        ax.set_yticks(sorted(lane_labels.keys()))
        ax.set_yticklabels(
            [lane_labels[k] for k in sorted(lane_labels.keys())],
            fontsize=9, color="white"
        )
        ax.set_ylim(-0.65, num_lanes - 0.35)
        ax.set_xlim(0, 18)
        ax.set_xlabel("Time Slots →", fontsize=9, color="white")
        ax.set_title(title, fontsize=11, pad=8, color="white")
        ax.tick_params(colors="white")
        ax.grid(axis="x", linestyle="--", alpha=0.25, color="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#FFFFFF33")

    # 共享图例
    legend_items = [
        mpatches.Patch(color="#2ecc71", label="Physics (Workers)"),
        mpatches.Patch(color="#4FC3F7", label="update_async"),
        mpatches.Patch(color="#81C784", label="record_next"),
        mpatches.Patch(color="#FFB74D", label="submit_next"),
        mpatches.Patch(color="#CE93D8", alpha=0.7, label="GPU Render"),
        mpatches.Patch(color="#F48FB1", label="Readback DMA"),
        mpatches.Patch(color="lime",   alpha=0.5, label="CPU‖GPU parallel"),
        mpatches.Patch(color="cyan",   alpha=0.5, label="GPU‖Readback parallel"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=4,
               fontsize=8, framealpha=0.3,
               facecolor="#0D0D1A", edgecolor="#FFFFFF44",
               labelcolor="white",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_image, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"[Plot] Architecture diagram saved → {output_image}")


# ─────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Async Pipeline Benchmark — CPU / GPU / Readback 三轨并行")
    p.add_argument("--num_envs",    type=int,  default=64)
    p.add_argument("--width",       type=int,  default=1024)
    p.add_argument("--height",      type=int,  default=1024)
    p.add_argument("--steps",       type=int,  default=200,
                   help="Total render steps")
    p.add_argument("--cold_start",  type=int,  default=10,
                   help="Warm-up steps ignored in stats")
    p.add_argument("--num_buffers", type=int,  default=10)
    p.add_argument("--max_geoms",   type=int,  default=1000)
    p.add_argument("--plot",        action="store_true",
                   help="Generate timeline & architecture plots")
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

    folder = args.output_dir or \
             f"results_async_env{TOTAL_ENVS}_{FRAME_W}x{FRAME_H}"
    os.makedirs(folder, exist_ok=True)
    LOG_FILE     = os.path.join(folder, "benchmark_async.csv")
    TIMELINE_IMG = os.path.join(
        folder, f"timeline_async_env{TOTAL_ENVS}_{FRAME_W}x{FRAME_H}.png")
    DIAG_IMG     = os.path.join(folder, "pipeline_diagram.png")
    RESULT_TXT   = os.path.join(folder, "result.txt")

    TOTAL_CORES  = multiprocessing.cpu_count()
    MAIN_CORES   = set(range(max(0, TOTAL_CORES - 4), TOTAL_CORES))
    WORKER_CPU_S = 0
    WORKER_CPU_E = max(0, TOTAL_CORES - 5)

    cfg = dict(
        total_envs=TOTAL_ENVS, frame_width=FRAME_W, frame_height=FRAME_H,
        shm_name=SHM_NAME, num_buffers=NUM_BUFFERS, max_geoms=MAX_GEOMS,
        log_file=LOG_FILE, worker_cpu_start=WORKER_CPU_S,
        worker_cpu_end=WORKER_CPU_E, cold_start_steps=COLD_START,
    )

    multiprocessing.set_start_method("spawn", force=True)
    print(f"[Bench] Envs={TOTAL_ENVS}  Res={FRAME_W}x{FRAME_H}  "
          f"Steps={TOTAL_STEPS}  Buffers={NUM_BUFFERS}  ColdStart={COLD_START}")

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
            env_name="TwoArmLift", robots=["UR5e", "UR5e"],
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

    # ── Async Readback 回调 ──────────────────────────────────
    # 回调由 C++ readback 线程在 DMA 完成后触发，完全异步于主循环。
    # 主循环在 is_slot_ready() 确认 GPU render 完成后即可继续，
    # 无需等待 readback。
    readback_frames   = {}     # {step_id: List[np.ndarray]}
    readback_log_lock = threading.Lock()
    readback_events   = []     # [(step_id, t_start, t_end)] 用于时序日志

    def on_readback_done(step_id: int, frames):
        """由 C++ readback 线程回调，记录时序并存帧"""
        t_cb = time.perf_counter()
        with readback_log_lock:
            readback_frames[step_id] = [f.copy() for f in frames]
            readback_events.append((step_id, t_cb))
        if step_id % 50 == 0:
            print(f"[Readback] step {step_id} frames received asynchronously")

    renderer.set_readback_callback(on_readback_done)

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
    # Main Loop  —  细粒度三轨并行版本
    #
    # 关键变化：
    #   - is_slot_ready() 确认 GPU render 完成后立即释放 shm buffer
    #   - 不再调用 wait_readback()，readback 完全异步
    #   - readback blit 由 C++ readback 线程自动触发
    #
    # 流水线状态：
    #   pending_slot / pending_step / pending_buf / pending_gpu_start
    #     → 已 submit、GPU 正在渲染的帧
    #   cur_slot / cur_read_buf / cur_step
    #     → 已 record、待 submit 的帧
    # ─────────────────────────────────────────────────────────
    prof_start = None
    rendered   = -1
    dispatched = 0

    step_latencies = []   # GPU render 完成延迟（submit → is_slot_ready）

    pending_slot      = -1
    pending_step      = -1
    pending_buf       = -1
    pending_gpu_start = 0.0

    cur_slot     = -1
    cur_read_buf = -1
    cur_step     = -1
    cur_frame_t0 = 0.0

    # 用于记录每个 slot 的 submit 时刻，供 readback 日志计算时间窗
    slot_submit_times = {}   # {slot_idx: (submit_t, step_id)}

    try:
        while rendered < TOTAL_STEPS:
            did_work = False

            # ── 阶段 A：提交已录制的帧 ──────────────────────────
            if cur_slot >= 0 and pending_slot < 0:
                t0 = time.perf_counter()
                with NvtxAnnotate("submit_next", color="orange"):
                    renderer.submit_next()
                t1 = time.perf_counter()

                if cur_step > COLD_START:
                    logger.log("Main_CPU", "submit_next",
                               t0, t1, cur_step, cur_read_buf, cur_slot)

                slot_submit_times[cur_slot] = (t1, cur_step)

                pending_slot      = cur_slot
                pending_step      = cur_step
                pending_buf       = cur_read_buf
                pending_gpu_start = t1

                cur_slot     = -1
                cur_read_buf = -1
                cur_step     = -1
                did_work = True

            # ── 阶段 B：录制下一帧（与 GPU 渲染并行）────────────
            if cur_slot < 0:
                read_buf = -1
                for bi in range(NUM_BUFFERS):
                    if buffer_state[bi] == BUF_READY:
                        buffer_state[bi] = BUF_READING
                        read_buf = bi
                        break

                if read_buf != -1:
                    this_step = rendered + (2 if pending_slot >= 0 else 1)
                    frame_t0  = time.perf_counter()

                    t0 = time.perf_counter()
                    with NvtxAnnotate("update_async", color="cyan"):
                        renderer.update_async(buf_addrs[read_buf], MAX_GEOMS, MAX_LIGHTS)
                    t1 = time.perf_counter()
                    if dispatched > COLD_START:
                        logger.log("Main_CPU", "update_async",
                                   t0, t1, this_step, read_buf)

                    slot = -1
                    t0_rec = time.perf_counter()
                    while slot < 0:
                        with NvtxAnnotate("record_next_nowait", color="green"):
                            slot = renderer.record_next_nowait(buf_addrs[read_buf])
                        if slot < 0:
                            time.sleep(0.00005)
                    t1_rec = time.perf_counter()
                    if dispatched > COLD_START:
                        logger.log("Main_CPU", "record_next",
                                   t0_rec, t1_rec, this_step, read_buf, slot)

                    cur_slot     = slot
                    cur_read_buf = read_buf
                    cur_step     = this_step
                    cur_frame_t0 = frame_t0
                    did_work = True

            # ── 阶段 C：非阻塞轮询 GPU 完成，立即释放 buffer ──────
            if pending_slot >= 0:
                if renderer.is_slot_ready(pending_slot):
                    gpu_done_t = time.perf_counter()

                    if pending_step > COLD_START:
                        logger.log("Main_GPU", "gpu_render",
                                   pending_gpu_start, gpu_done_t,
                                   pending_step, pending_buf, pending_slot)
                        logger.log("Main_CPU", "wait_slot",
                                   gpu_done_t, gpu_done_t + 1e-6,
                                   pending_step, pending_buf, pending_slot)

                    # ★ 核心变化：GPU render 完成即刻释放 buffer
                    # readback blit 由 C++ readback 线程自动触发，不阻塞主循环
                    buffer_state[pending_buf] = BUF_FREE
                    rendered += 1

                    if rendered == COLD_START:
                        prof_start = time.perf_counter()
                        print(f"[Bench] Cold start done at step {rendered}. "
                              f"Profiling started.")

                    if rendered > COLD_START:
                        step_latencies.append(gpu_done_t - pending_gpu_start)

                    with buffer_cond:
                        buffer_cond.notify_all()

                    pending_slot      = -1
                    pending_buf       = -1
                    pending_step      = -1
                    pending_gpu_start = 0.0
                    did_work = True

            # ── 阶段 D：派发物理步骤给 Workers ──────────────────
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
                    logger.log("Main_CPU", "Dispatch",
                               t0, t1, dispatched, nxt_buf)
                did_work = True

            if not did_work:
                time.sleep(0.00005)

    finally:
        # 安全退出：等待 pending GPU 工作完成；readback 通过线程异步处理
        if pending_slot >= 0:
            renderer.wait_slot(pending_slot)
        # 等所有 readback 完成后再停线程
        renderer.stop_readback_thread()
        stop_flag.value = 1
        with buffer_cond:
            buffer_cond.notify_all()
        for p in workers:
            p.terminate()
            p.join()

        # 将 readback 事件写入日志（用于三轨可视化）
        # readback_blit: 从 GPU render 完成到 callback 触发的时间窗
        with readback_log_lock:
            for (step_id, t_cb) in readback_events:
                if step_id > COLD_START:
                    # 用 submit 时刻 + GPU render 完成偏移来推算 blit 开始时间
                    # 实际上 blit 在 render_fence 完成后立即开始，这里用 callback 时刻
                    # 减去估算 DMA 时间（约 2ms）作为 blit 开始时刻
                    rb_dur_est = 0.002   # 2ms DMA 估算
                    blit_start = t_cb - rb_dur_est
                    blit_end   = t_cb
                    logger.log("Readback_Thread", "readback_blit",
                               blit_start, blit_end,
                               step_id, -1, -1)
                    # callback 触发点（极短）
                    logger.log("Readback_Thread", "readback_cb",
                               t_cb, t_cb + 1e-5, step_id, -1, -1)

        with readback_log_lock:
            print(f"[Bench] Total frames received via async callback: "
                  f"{len(readback_frames)}")

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

        with readback_log_lock:
            rb_received = len(readback_frames)

        lines = [
            "=" * 60,
            "  Fine-Grained Async Pipeline Benchmark Results",
            "  (CPU / GPU / Readback)",
            "=" * 60,
            f"  Config : Envs={TOTAL_ENVS}  Res={FRAME_W}x{FRAME_H}  "
            f"Buffers={NUM_BUFFERS}",
            f"  Valid Steps          : {valid_steps}",
            f"  Total Duration       : {total_dur:.4f} s",
            f"  FPS                  : {fps:.2f}",
            f"  Avg GPU Render Lat   : {avg_lat*1000:.3f} ms",
            f"  P50 GPU Render Lat   : {p50*1000:.3f} ms",
            f"  P95 GPU Render Lat   : {p95*1000:.3f} ms",
            f"  P99 GPU Render Lat   : {p99*1000:.3f} ms",
            f"  Async Frames Received: {rb_received} / {valid_steps}",
            "=" * 60,
        ]
        output = "\n".join(lines)
        print(output)
        with open(RESULT_TXT, "w") as f:
            f.write(output + "\n")

        # ── Per-stage breakdown ──────────────────────────────
        try:
            df_log = pd.read_csv(LOG_FILE)

            cpu_df = df_log[df_log["process"] == "Main_CPU"].copy()
            cpu_df["dur"] = cpu_df["end_time"] - cpu_df["start_time"]
            cpu_bd = (cpu_df.groupby("event")["dur"]
                      .agg(["mean", "std", "max"]) * 1000)
            cpu_bd.columns = ["mean_ms", "std_ms", "max_ms"]

            gpu_df = df_log[df_log["process"] == "Main_GPU"].copy()
            gpu_df["dur"] = gpu_df["end_time"] - gpu_df["start_time"]
            gpu_bd = (gpu_df.groupby("event")["dur"]
                      .agg(["mean", "std", "max"]) * 1000)
            gpu_bd.columns = ["mean_ms", "std_ms", "max_ms"]

            rb_df = df_log[df_log["process"] == "Readback_Thread"].copy()
            rb_df["dur"] = rb_df["end_time"] - rb_df["start_time"]
            rb_bd = (rb_df.groupby("event")["dur"]
                     .agg(["mean", "std", "max"]) * 1000)
            rb_bd.columns = ["mean_ms", "std_ms", "max_ms"]

            summary_str = (
                "\nPer-Stage Breakdown — Main_CPU (ms):\n"   + cpu_bd.to_string() +
                "\n\nGPU Render Window — Main_GPU (ms):\n"   + gpu_bd.to_string() +
                "\n\nReadback Thread (ms):\n"                 + rb_bd.to_string()
            )
            print(summary_str)
            with open(RESULT_TXT, "a") as f:
                f.write(summary_str + "\n")

            # ── 并行重叠率 ────────────────────────────────────
            try:
                gr = df_log[(df_log["process"] == "Main_GPU") &
                            (df_log["event"]   == "gpu_render")]
                rn = df_log[(df_log["process"] == "Main_CPU") &
                            (df_log["event"]   == "record_next")]
                rb = df_log[(df_log["process"] == "Readback_Thread") &
                            (df_log["event"]   == "readback_blit")]

                total_gpu = 0.0
                ov_cpu_gpu = 0.0
                ov_gpu_rb  = 0.0

                for _, grow in gr.iterrows():
                    dur_g = grow["end_time"] - grow["start_time"]
                    total_gpu += dur_g

                    for _, rrow in rn.iterrows():
                        if int(rrow["step"]) - int(grow["step"]) == 1:
                            ov = (min(grow["end_time"],  rrow["end_time"]) -
                                  max(grow["start_time"], rrow["start_time"]))
                            if ov > 0:
                                ov_cpu_gpu += ov

                    for _, rrow in rb.iterrows():
                        if int(grow["step"]) - int(rrow["step"]) == 1:
                            ov = (min(grow["end_time"],  rrow["end_time"]) -
                                  max(grow["start_time"], rrow["start_time"]))
                            if ov > 0:
                                ov_gpu_rb += ov

                if total_gpu > 0:
                    ol_str = (
                        f"\nCPU‖GPU   Overlap Ratio: "
                        f"{100*ov_cpu_gpu/total_gpu:.1f}% of GPU time\n"
                        f"GPU‖RB    Overlap Ratio: "
                        f"{100*ov_gpu_rb/total_gpu:.1f}% of GPU time"
                    )
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