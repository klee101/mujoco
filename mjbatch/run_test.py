#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    # 切换到构建目录（如果需要）
    build_dir = "/home/hpf/project/vulkan/mujoco/mujoco/build"
    original_dir = os.getcwd()
    
    try:
        print("步骤1: 构建MuJoCo项目...")
        subprocess.run([
            "/usr/bin/cmake", "--build", build_dir,
            "--config", "Debug", "--target", "mjbatch_smoke", "--parallel"
        ], check=True)

        print("步骤2: 编译着色器...")
        subprocess.run([
            "/usr/bin/cmake", "--build", build_dir,
            "--config", "Debug", "--target", "CompileShaders", "--parallel"
        ], check=True)
        
        print("\n步骤3: 运行mjbatch_smoke...")
        os.chdir(build_dir)
        subprocess.run(["./bin/mjbatch_smoke"], check=True)
        print("\n步骤3: 转换图像格式...")
        subprocess.run([
            "python3", "-c", 
            "from PIL import Image; Image.open('merged_result.ppm').save('merged_result.png')"
        ], check=True)
        
        print("\n✅ 所有任务完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 任务失败: {e}")
        sys.exit(1)
    finally:
        # 确保切换回原始目录
        os.chdir(original_dir)

if __name__ == "__main__":
    main()