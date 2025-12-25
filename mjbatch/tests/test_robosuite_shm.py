import time
import ctypes
import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import cv2  # pip install opencv-python

# 1. Import your compiled extensions
import mjb             # The Consumer (Renderer)
import test_shm_ext    # The Producer (Binder)

# 2. Import Simulation Env
import robosuite as suite
import mujoco

# --- Constants ---
# [Fix 1] 增加缓冲区大小，防止 Robosuite 场景（包含大量 site/contact）导致扩容崩溃
MAX_GEOM = 10000 
MAX_LIGHT = 100
WIDTH = 640
HEIGHT = 480

# Ctypes Structure for the Header (Matches shared_protocol.h)
class SharedSceneHeader(ctypes.Structure):
    _fields_ = [
        ("frame_counter", ctypes.c_uint32),
        ("ngeom", ctypes.c_uint32),
        ("nlight", ctypes.c_uint32),
        ("cam_pos", ctypes.c_float * 3),
        ("cam_mat", ctypes.c_float * 9),
        ("cam_fovy", ctypes.c_float),
        ("pad", ctypes.c_uint32 * 8), # 保留 Padding 以确保内存对齐
    ]

# -----------------------------------------------------------------------------
# Process 1: Rendering Worker (Consumer)
# -----------------------------------------------------------------------------
def renderer_process(shm_name, xml_path):
    """
    Runs in a separate process.
    Reads from Shared Memory -> Renders -> Displays with OpenCV
    """
    print(f"[Renderer] Initializing BatchRenderer with {xml_path}...")
    
    # 1. Attach to Shared Memory
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        print("[Renderer] Error: Shared memory not found.")
        return

    # 2. Configure Renderer
    cfg = mjb.BatchRendererConfig()
    cfg.batch_size = 1
    cfg.frame_width = WIDTH
    cfg.frame_height = HEIGHT
    cfg.gpu_id = 0
    cfg.enable_validation = False 

    # 3. Create Renderer
    try:
        renderer = mjb.BatchRenderer(xml_path, cfg)
    except Exception as e:
        print(f"[Renderer] Failed to create renderer: {e}")
        return
    
    # Map header for reading frame counters
    header = SharedSceneHeader.from_buffer(shm.buf)
    last_counter = 0

    print("[Renderer] Starting Loop...")
    
    while True:
        # Spin-wait
        current_counter = header.frame_counter
        
        if current_counter > last_counter:
            # --- Render Step ---
            # Pass the raw memory address to C++
            addr = ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))
            renderer.render_from_shm(addr, 0, MAX_GEOM, MAX_LIGHT)
            
            # --- Display Step ---
            img = renderer.get_image(0)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            cv2.putText(img_bgr, f"Frame: {current_counter} | Geoms: {header.ngeom}", 
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Vulkan Batch Renderer", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            last_counter = current_counter
        else:
            time.sleep(0.001)

    print("[Renderer] Closing...")
    shm.close()
    cv2.destroyAllWindows()

# -----------------------------------------------------------------------------
# Process 2: Main Simulation (Producer)
# -----------------------------------------------------------------------------
def main():
    # 1. Setup Robosuite Environment
    print("[Main] Creating Robosuite Environment...")
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,       # Headless
        use_camera_obs=False,     # We render manually
        has_offscreen_renderer=False,
        control_freq=20,
    )
    
    # [Fix 2] 修复 XML 保存方式，适配 Robosuite Wrapper
    xml_path = "/tmp/robosuite_scene.xml"
    with open(xml_path, "w") as f:
        f.write(env.sim.model.get_xml())
    print(f"[Main] Scene saved to {xml_path}")

    # 2. Setup Shared Memory
    header_size = ctypes.sizeof(SharedSceneHeader)
    # [Fix 1] 重新计算大小以匹配 50000 Geoms
    total_size = header_size + (MAX_GEOM * 256) + (MAX_LIGHT * 128) 
    print(f"[Main] Allocating Shared Memory: {total_size/1024/1024:.2f} MB")
    
    shm = shared_memory.SharedMemory(create=True, size=total_size)
    
    # 3. Setup Binder
    binder = test_shm_ext.SharedSceneBinder()
    binder.bind(shm.buf, MAX_GEOM) 

    # 4. Start Renderer Process
    p_render = multiprocessing.Process(target=renderer_process, args=(shm.name, xml_path))
    p_render.start()

    header = SharedSceneHeader.from_buffer(shm.buf)
    
    # [Fix 3] 核心修复：指针获取
    # 剥离 Wrapper 并使用 ctypes.cast 确保拿到完整的 64 位地址
    raw_model = env.sim.model._model 
    raw_data = env.sim.data._data
    
    m_ptr = ctypes.cast(raw_model._address, ctypes.c_void_p).value
    d_ptr = ctypes.cast(raw_data._address, ctypes.c_void_p).value
    
    print(f"[Main] Ptrs (Checked) -> Model: {hex(m_ptr)}, Data: {hex(d_ptr)}")

    print("[Main] Starting Simulation Loop...")
    try:
        steps = 0
        while True:
            # --- A. Physics Step ---
            action = np.random.randn(env.action_dim)
            env.step(action)
            
            # --- B. Update Shared Memory ---
            # Robosuite camera sync (Optional implementation in your binder)
            # cam_id = env.sim.model.camera_name2id("frontview")
            
            # C++ Update
            binder.update(m_ptr, d_ptr)
            
            # Frame Counter
            header.frame_counter += 1
            
            steps += 1
            if steps % 100 == 0:
                print(f"[Main] Simulating step {steps}...")
            
            time.sleep(0.016) 

    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        p_render.terminate()
        p_render.join()

        # [Fix 4] 必须先删除 binder 释放 buffer 引用，否则 shm.close() 会报错
        del binder
        
        shm.close()
        shm.unlink()
        env.close()

if __name__ == "__main__":
    main()