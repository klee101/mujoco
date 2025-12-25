import time
import struct
import mujoco
import multiprocessing
from multiprocessing import shared_memory
import ctypes
import sys
import test_shm_ext 

# 定义 mjvGeom 的 Python 解包格式
# i=int, f=float, 100s=char[100], b=byte
# 注意：最后可能需要根据 sizeof(mjvGeom) 调整 padding
# 假设是标准布局：
GEOM_FMT = "8i 3f 3f 9f 4f 4f 100s 2f b 3x" 
# 解释：
# 8i: type, dataid, objtype, objid, category, matid, texcoord, segid
# 3f: size
# 3f: pos
# 9f: mat (rotation)
# 4f: rgba
# 4f: emission, specular, shininess, reflectance
# 100s: label
# 2f: camdist, modelrbound
# b: transparent
# 3x: padding (align to 4 bytes)

# 计算该格式的大小，用于验证是否与 C++ sizeof 一致
try:
    GEOM_SIZE = struct.calcsize(GEOM_FMT)
except:
    GEOM_SIZE = 0

def debug_print_geom(shm_buf, header_size, geom_idx=0):
    """
    读取并打印指定索引的 Geom 信息
    """
    offset = header_size + geom_idx * GEOM_SIZE
    
    # 截取这段内存
    data = shm_buf[offset : offset + GEOM_SIZE]
    
    try:
        unpacked = struct.unpack(GEOM_FMT, data)
        
        # 提取关键字段
        g_type = unpacked[0]      # type
        g_objid = unpacked[3]     # objid (对应 xml 里的 ID)
        g_pos = unpacked[11:14]   # pos (x, y, z)
        g_rgba = unpacked[23:27]  # rgba
        g_label = unpacked[31].decode('utf-8', errors='ignore').strip('\x00')
        
        print(f"\n--- [DEBUG] Geom {geom_idx} Analysis ---")
        print(f"  Type ID:    {g_type} (3=Capsule, 5=Sphere, 6=Box, etc.)")
        print(f"  Obj ID:     {g_objid}")
        print(f"  Position:   x={g_pos[0]:.2f}, y={g_pos[1]:.2f}, z={g_pos[2]:.2f}")
        print(f"  Color:      r={g_rgba[0]:.2f}, g={g_rgba[1]:.2f}, b={g_rgba[2]:.2f}, a={g_rgba[3]:.2f}")
        if g_label:
            print(f"  Label:      '{g_label}'")
        else:
            print(f"  Label:      (None)")
            
        # 简单逻辑检查
        if g_rgba[3] == 0.0:
            print("  ⚠️ Warning: Alpha is 0.0, this geom might be invisible!")
        if abs(g_pos[0]) > 100 or abs(g_pos[1]) > 100:
            print("  ⚠️ Warning: Position coordinates seem very large/invalid.")
            
    except struct.error as e:
        print(f"  ❌ Struct Unpack Error: {e}. Size mismatch? Python calc size: {GEOM_SIZE}")

xml = """
<mujoco>
  <statistic center="0 0 0.7"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <worldbody>
    <light pos="0 0 2" mode="trackcom"/>
    <body name="torso" pos="0 0 1">
      <freejoint/>
      <geom type="capsule" size="0.07 0.15" rgba="0.8 0.6 0.4 1"/>
      <geom type="sphere" size="0.09" pos="0 0 0.2" rgba="0.8 0.6 0.4 1"/>
    </body>
  </worldbody>
</mujoco>
"""

def render_worker(shm_name, stop_event):
    print("[Worker] Starting...")
    shm = shared_memory.SharedMemory(name=shm_name)
    
    try:
        while not stop_event.is_set():
            # 1. 读取 Header
            ngeom = struct.unpack('i', shm.buf[:4])[0]
            nlight = struct.unpack('i', shm.buf[4:8])[0]
            
            if ngeom > 0:
                if int(time.time()) % 2 == 0: 
                    debug_print_geom(shm.buf, header_size=8, geom_idx=0)
                    time.sleep(1.0)
            
            time.sleep(0.016)
            
    except Exception as e:
        print(f"\n[Worker] Error: {e}")
    finally:
        shm.close()
        print("\n[Worker] Stopped cleanly.")

def main():
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    MAX_GEOM = 1000
    required_size = test_shm_ext.SharedSceneBinder.get_required_size(MAX_GEOM)
    
    print(f"[Main] Allocating SHM: {required_size} bytes")
    shm = shared_memory.SharedMemory(create=True, size=required_size)
    
    binder = test_shm_ext.SharedSceneBinder()
    binder.bind(shm.buf, MAX_GEOM)

    stop_event = multiprocessing.Event()
    
    p = multiprocessing.Process(target=render_worker, args=(shm.name, stop_event))
    p.start()

    print("[Main] Simulation Started. Press Ctrl+C to stop.")
    try:
        frames = 0
        while True:
            mujoco.mj_step(m, d)
            
            m_ptr = int(m._address)
            d_ptr = int(d._address)
            
            binder.update(m_ptr, d_ptr)
            
            frames += 1
            if frames % 60 == 0:
                pass 

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C detected. Stopping worker...")
        stop_event.set()
        
    finally:
        p.join(timeout=2.0)
        if p.is_alive():
            print("[Main] Worker stuck, forcing kill...")
            p.terminate()
            
        shm.close()
        shm.unlink()
        print("[Main] Shared memory unlinked. Bye!")

if __name__ == "__main__":
    main()