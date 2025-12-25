
## Vulkan/Mujoco 项目环境配置与 Benchmark 流程

### I. 环境配置与依赖准备

#### 1\. Vulkan SDK 下载与配置

下载 Vulkan SDK **1.4.328.1** 版本。

**配置 `$HOME/.bashrc`：**
将以下环境变量添加到您的 shell 配置文件（例如 `~/.bashrc`），然后运行 `source ~/.bashrc` 使其生效。

```bash
export VULKAN_SDK=~/vulkan/1.4.328.1/x86_64
export PATH=$VULKAN_SDK/bin:$PATH
export LD_LIBRARY_PATH=$VULKAN_SDK/lib:$LD_LIBRARY_PATH
export VK_LAYER_PATH=$VULKAN_SDK/share/vulkan/explicit_layer.d
export PKG_CONFIG_PATH=$VULKAN_SDK/lib/pkgconfig:$PKG_CONFIG_PATH
```

#### 2\. Submodule 更新

在项目根目录 (`/mujoco`) 下执行命令，拉取并初始化所有子模块依赖：

```bash
git submodule update --init --recursive
```

#### 3\. 依赖库手动处理

由于网络和编译环境限制，部分外部依赖需要手动下载或解压。

  * **DXC 库解压 (.so):**
    将下载的 DXC 压缩包解压，确保动态链接库（`.so` 文件）可用。

    ```bash
    tar zxvf linux_dxc_2025_07_14.x86_64.tar.gz 
    ```

  * **Madrona Toolchain 手动下载 (解决网络问题):**
    如果 CMake 自动下载失败，请手动下载压缩包，并放置到指定的构建源路径。

    1.  **目标路径：**
        ```
        mujoco/external/madrona/external/madrona-toolchain/cmake-tmp/madronabundledtoolchain-subbuild/madronabundledtoolchain-populate-prefix/src/
        ```
    2.  **下载命令：**
        ```bash
        wget https://github.com/shacklettbp/madrona-toolchain/releases/download/8c0b55b/madrona-toolchain-8c0b55b-linux-x86_64.tar.xz
        ```

#### 4\. GLFW/图形后端配置

根据您的设备环境（服务器只有 X11），需要在 CMake 配置中**显式禁用 Wayland** 支持，只使用 X11 后端。

  * **重要设置：** 确保在配置时传入以下选项：
    ```
    -DGLFW_BUILD_WAYLAND=OFF
    ```

### II. Benchmark 构建与编译

本项目采用编译优化选项，以确保 Benchmark 的最高性能。

#### 1\. CMake 配置项目

在项目根目录（`/mujoco`）下执行以下命令进行 CMake 配置：

```bash
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
    -DMUJOCO_ENABLE_AVX=ON \
    -DMUJOCO_ENABLE_AVX_INTRINSICS=ON \
    -DGLFW_BUILD_WAYLAND=OFF \
    -DCMAKE_CXX_FLAGS="-march=native -O3 -funroll-loops"

cmake ..     -DPython_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Debug
```

> **提示:** `-march=native` 会针对当前机器的 CPU 架构进行最大优化。

#### 2\. 编译目标

Vulkan 程序的运行依赖于 **Shader SPV 文件**。这里通过 `CompileShaders` 目标自动编译并生成这些文件。

进入构建目录并执行编译：

```bash
cd /path/to/mujoco
# 编译 C++ Benchmark 主程序
cmake --build build --config Release --target benchmark_test --parallel

# 编译 Shaders（生成 .spv 文件）
cmake --build build --config Release --target CompileShaders --parallel
```

### III. 执行 Benchmark

在 `mujoco/build` 路径下执行编译好的程序：

```bash
cd build
./bin/benchmark_test
```

### IV. 已知路径问题

本项目中 `tests/benckmark_test.cc` 使用的 **mjcf model** 是通过 Robosuite 生成的 XML 文件，其中包含**绝对路径**来索引 `.stl` 和纹理图片等资源。

**解决方案：** 正常运行 Benchmark 暂时需要手动提取 Robosuite 的运行时 XML 文件，并确保所有资源路径都是正确的相对路径或可访问的绝对路径。