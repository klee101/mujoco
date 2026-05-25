## mjbatch

`mjbatch` 是 MuJoCo 的 Vulkan batch renderer 实验模块，提供 Python 模块 `mjb`，用于从 shared memory 中读取 robosuite 写入的场景数据，并通过异步 GPU 渲染和异步 readback 获得批量图像结果。

### 目录说明

```text
mjbatch/
  examples/mjb_async_pipeline.py   # 推荐的 mjb 接入范式
  scripts/build.sh                 # 一键配置和构建脚本
  tests/test_readback.py           # benchmark / stress test
  tests/benchmark.sh               # benchmark 运行脚本
  tests/verify_shm.py              # shared-memory binder 检查脚本
```

### 环境要求

1. Vulkan SDK

当前开发环境使用 Vulkan SDK `1.4.328.1`。请确保 `VULKAN_SDK`、`PATH`、`LD_LIBRARY_PATH`、`VK_LAYER_PATH` 和 `PKG_CONFIG_PATH` 已正确配置。

2. 子模块

在仓库根目录执行：

```bash
git submodule update --init --recursive
```

3. DXC

确认 DXC 可执行文件存在：

```bash
ls mjbatch/dxc/bin/dxc
```

如果 DXC 是手动下载的压缩包，先在 `mjbatch/dxc` 目录下解压。

4. robosuite

Python 示例和 benchmark 依赖 robosuite 侧写入 shared memory 的实现。这部分在自定义 robosuite 中，不在普通上游 robosuite 中。

自定义仓库地址：

```text
git@github.com:klee101/robosuite
```

5. Python 包版本

MuJoCo Python 包要求：

```text
mujoco==3.3.7
```

### 快速开始

激活虚拟环境，在仓库根目录执行：

```bash
./mjbatch/scripts/build.sh
python mjbatch/examples/mjb_async_pipeline.py --num-envs 4 --steps 20
```

`build.sh` 默认执行 Release 配置，并构建：

```text
mjb
CompileShaders
```

### 构建

默认构建：

```bash
./mjbatch/scripts/build.sh
```

Debug 构建：

```bash
./mjbatch/scripts/build.sh --debug
```

只构建指定目标：

```bash
./mjbatch/scripts/build.sh --target mjb
./mjbatch/scripts/build.sh --target CompileShaders
./mjbatch/scripts/build.sh --target test_shm_ext
```

覆盖构建目录、Python 或编译器：

```bash
BUILD_DIR=/tmp/mujoco-build \
PYTHON_EXECUTABLE=$(which python) \
C_COMPILER=gcc-11 \
CXX_COMPILER=g++-11 \
./mjbatch/scripts/build.sh
```

传入额外 CMake 参数：

```bash
EXTRA_CMAKE_ARGS="-DGLFW_BUILD_X11=ON" ./mjbatch/scripts/build.sh
```

查看脚本选项：

```bash
./mjbatch/scripts/build.sh --help
```

### mjb 接入范式

推荐从这个示例开始。首先激活虚拟环境：

```bash
mjbatch/examples/mjb_async_pipeline.py
```

它展示的是应用接入 `mjb` 的标准流程，而不是 benchmark：

1. 创建 `BatchRendererConfig`
2. 构造 `BatchRenderer`
3. 启动 `start_readback_thread()`
4. 设置 `set_readback_callback()`
5. 调用 `record_next_nowait(shm_ptr, max_geoms, max_lights)` 录制下一帧
6. 调用 `submit_next(slot)` 提交 GPU work
7. 用 `is_slot_ready(slot)` 非阻塞轮询 GPU 完成
8. 在 shutdown 时停止 readback thread

运行示例：

```bash
python mjbatch/examples/mjb_async_pipeline.py --num-envs 4 --steps 20
```

如果回调中的图像需要在 callback 返回后继续使用，需要在 callback 内复制 frame 数据。

默认情况下，`mjb` 会关闭 C++ 侧的流水线调试日志。需要排查 slot、submit 或 readback 状态时，可以打开：

```bash
python mjbatch/examples/mjb_async_pipeline.py --debug-mjb
```

在代码中对应配置是：

```python
renderer_cfg.debug_logging = True
```

### Benchmark

`mjbatch/tests/test_readback.py` 保留为 benchmark / stress test，不作为接口范式使用。它包含更重的统计、日志和可视化逻辑。

运行 benchmark：

```bash
./mjbatch/tests/benchmark.sh
```

直接运行 benchmark 并打开 C++ 侧调试日志：

```bash
python mjbatch/tests/test_readback.py --debug_mjb
```

日志默认写入：

```text
mjbatch/tests/debug_logs/
```

### 常见问题

1. `import mjb` 失败

先构建 `mjb`：

```bash
./mjbatch/scripts/build.sh --target mjb
```

示例脚本默认从 `build/lib` 加载 `mjb`。

2. robosuite 找不到 shared-memory 相关参数或接口

确认当前 Python 环境使用的是自定义 robosuite：

```text
git@github.com:klee101/robosuite
```

3. MuJoCo 版本不一致

确认 Python 环境中使用 `mujoco==3.3.7`。
