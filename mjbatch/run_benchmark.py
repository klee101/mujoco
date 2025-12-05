#!/usr/bin/env python3

import subprocess
import sys
import os
import shutil

def main():
    # 路径配置
    # 注意：根据您的实际情况，这里应该是项目根目录
    project_root = "/home/hpf/project/vulkan/mujoco/mujoco" 
    build_dir = os.path.join(project_root, "build")
    original_dir = os.getcwd()
    
    target_name = "benchmark_test"
    output_ppm = "benchmark_sample_0.ppm"
    output_png = "benchmark_sample_0.png"

    cmake_args = [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON", 
        "-DMUJOCO_ENABLE_AVX=ON",
        "-DMUJOCO_ENABLE_AVX_INTRINSICS=ON",

        "-DCMAKE_CXX_FLAGS=-march=native -O3 -funroll-loops"
    ]

    try:
        print(f"========================================")
        print(f"  准备运行 Benchmark: {target_name}")
        print(f"========================================")
        
        # Step 1: 配置项目 (Configuration Step) - 
        print(f"\n[1/4] 配置 CMake (应用优化标志)...")
        # 命令格式: cmake -S <源码目录> -B <构建目录> [ARGS...]
        config_cmd = ["/usr/bin/cmake", "-S", project_root, "-B", build_dir] + cmake_args
        
        # 打印一下命令，方便调试
        print("执行命令:", " ".join(config_cmd))
        subprocess.run(config_cmd, check=True)

        # Step 2: 编译 C++ Benchmark 程序 (Build Step)
        print(f"\n[2/4] 编译目标: {target_name}...")

        subprocess.run([
            "/usr/bin/cmake", "--build", build_dir,
            "--config", "Release", 
            "--target", target_name, 
            "--parallel"  # 利用多核编译
        ], check=True)

        # Step 3: 编译着色器 (Shader)
        print("\n[3/4] 编译着色器...")
        subprocess.run([
            "/usr/bin/cmake", "--build", build_dir,
            "--config", "Release", 
            "--target", "CompileShaders", 
            "--parallel"
        ], check=True)
        
        print(f"\n[4/4] 运行 {target_name}...")
    
        os.chdir(build_dir)
        
        executable_path = f"./bin/{target_name}"
        if not os.path.exists(executable_path):
            if os.path.exists(f"./{target_name}"):
                executable_path = f"./{target_name}"
            else:
                raise FileNotFoundError(f"找不到可执行文件")

        subprocess.run([executable_path], check=True)

        print("\n[Post] 转换结果图像格式...")
        if os.path.exists(output_ppm):
            subprocess.run([
                "python3", "-c", 
                f"from PIL import Image; Image.open('{output_ppm}').save('{output_png}')"
            ], check=True)
            print(f"✅ 转换成功: {os.path.join(build_dir, output_png)}")
        else:
            print(f"⚠️ 警告: 未找到输出文件 {output_ppm}")
        
        print("\n🎉 Benchmark 流程全部完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    main()