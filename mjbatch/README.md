## vulkan 下载
下载 vulkan 1.4.328.1

配置类似的 ~/.bashrc
```bash
export VULKAN_SDK=~/vulkan/1.4.328.1/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export PKG_CONFIG_PATH=$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH
````

## 子模块更新

```bash
git submodule update --init --recursive
````

## dxc解压
.so库需要手动解压
```bash
tar zxvf linux_dxc_2025_07_14.x86_64.tar.gz 
````

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

### 4\. 路径问题

项目中tests/benckmark_test.cc中使用的mjcf model是使用robosuite生成的xml，带有绝对路径来索引.stl和纹理图片等等。因此正常运行暂时需要手动提取robosuite的运行时xml。


