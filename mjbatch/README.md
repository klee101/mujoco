##  benchmark 构建与编译

本项目采用了编译优化选项以确保 benchmark 的性能。

### 1. 配置项目 

在项目根目录(/mujoco)下执行以下命令进行 CMake 配置：

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DMUJOCO_ENABLE_AVX=ON \
    -DMUJOCO_ENABLE_AVX_INTRINSICS=ON \
    -DCMAKE_CXX_FLAGS="-march=native -O3 -funroll-loops"
````

### 2\. 编译目标 
vulkan程序的运行依赖于shader的spv文件，这里通过CompileShaders自动编译并生成spv文件

编译 C++ Benchmark 程序及着色器：
cd
```bash
# 编译主程序
cmake --build build --config Release --target benchmark_test --parallel

# 编译 Shader
cmake --build build --config Release --target CompileShaders --parallel
```
### 3\. 执行 

在mujoco/build路径下：
```bash
./bin/benchmark_test
```


### 4\. 一键运行脚本

为了简化流程，可以直接运行提供的 Python 脚本，它会直接执行上面的流程并且把程序输出的PPM文件转化为可视的png图像（需要下载图像转换对应的库）

```bash
python run_benchmark.py
```

