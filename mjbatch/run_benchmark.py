#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    # 路径配置：保持与你的环境一致
    build_dir = "/home/hpf/project/vulkan/mujoco/mujoco/build"
    original_dir = os.getcwd()
    
    # 目标名称：对应 CMakeLists.txt 中的 add_executable(benchmark_test ...)
    target_name = "benchmark_test"
    # 输出文件名：对应 C++ 代码中 write_ppm 的文件名
    output_ppm = "benchmark_sample_0.ppm"
    output_png = "benchmark_sample_0.png"

    try:
        print(f"========================================")
        print(f"🚀 准备运行 Benchmark: {target_name}")
        print(f"========================================")

        # 步骤 1: 编译 C++ Benchmark 程序
        print(f"\n[1/4] 编译目标: {target_name}...")
        subprocess.run([
            "/usr/bin/cmake", "--build", build_dir,
            "--config", "Debug", "--target", target_name, "--parallel"
        ], check=True)

        # 步骤 2: 编译着色器 (确保 Shader 是最新的)
        print("\n[2/4] 编译着色器 (CompileShaders)...")
        subprocess.run([
            "/usr/bin/cmake", "--build", build_dir,
            "--config", "Debug", "--target", "CompileShaders", "--parallel"
        ], check=True)
        
        # 步骤 3: 运行 Benchmark 可执行文件
        print(f"\n[3/4] 运行 {target_name}...")
        os.chdir(build_dir)
        
        # 假设生成的可执行文件在 bin 目录下
        executable_path = f"./bin/{target_name}"
        if not os.path.exists(executable_path):
            # 某些构建系统可能直接生成在 build 根目录，这里做个兼容检查
            if os.path.exists(f"./{target_name}"):
                executable_path = f"./{target_name}"
            else:
                raise FileNotFoundError(f"找不到可执行文件: {executable_path}")

        subprocess.run([executable_path], check=True)

        # 步骤 4: 转换结果图像 (PPM -> PNG)
        print("\n[4/4] 转换结果图像格式...")
        if os.path.exists(output_ppm):
            subprocess.run([
                "python3", "-c", 
                f"from PIL import Image; Image.open('{output_ppm}').save('{output_png}')"
            ], check=True)
            print(f"✅ 转换成功: {os.path.join(build_dir, output_png)}")
        else:
            print(f"⚠️ 警告: 未找到输出文件 {output_ppm}，可能是 Benchmark 运行失败或未保存图像。")
        
        print("\n🎉 Benchmark 流程全部完成!")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 构建或运行失败: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ 文件错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        sys.exit(1)
    finally:
        # 确保无论成败都切换回原始目录，以免影响下一次运行
        os.chdir(original_dir)

if __name__ == "__main__":
    main()