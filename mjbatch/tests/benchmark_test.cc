#include "renderer.h"
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <cmath>
#include <numeric>
#include <algorithm>

// ---- Benchmark 配置 ----
const int BATCH_SIZE = 8;        // ⚠️ 在这里修改测试规模 (例如 64, 128, 512)
const int FRAME_WIDTH = 640;       // VLA 常用分辨率
const int FRAME_HEIGHT = 480;
const int BENCHMARK_STEPS = 10;  // 测试总帧数
const int WARMUP_STEPS = 5;       // 热身帧数 (不计入统计)
const std::string MODEL_XML = "./model/lift.xml"; // 测试用的模型

// PPM 写入函数 (用于验证渲染结果是否正确)
static bool write_ppm(const std::string& path, const unsigned char* data, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P6\n" << w << " " << h << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(data), std::streamsize(w*h*3));
    return ofs.good();
}

int main() {
    printf("======================================\n");
    printf("⚡ Vulkan Renderer Benchmark (Current)\n");
    printf("   Batch Size: %d\n", BATCH_SIZE);
    printf("   Resolution: %dx%d\n", FRAME_WIDTH, FRAME_HEIGHT);
    printf("======================================\n");

    // ---- 1) 准备模型 (模拟 N 个环境) ----
    char error[1024] = {0};
    mjModel* base_model = mj_loadXML(MODEL_XML.c_str(), nullptr, error, sizeof(error));
    if (!base_model) {
        std::fprintf(stderr, "[Fatal] mj_loadXML failed: %s\n", error);
        return 1;
    }

    std::vector<mjModel*> models;
    models.reserve(BATCH_SIZE);
    
    // 我们复制 base_model N 次，模拟 N 个独立的环境
    // 注意：在实际内存中，这会占用 N 份模型内存。
    models.push_back(base_model);
    for (int i = 1; i < BATCH_SIZE; ++i) {
        models.push_back(mj_copyModel(nullptr, base_model));
    }
    printf("[Init] Loaded %d models.\n", BATCH_SIZE);

    // ---- 2) 初始化渲染器 ----
    BatchRendererConfig cfg;
    cfg.batch_size = BATCH_SIZE;
    cfg.frame_width = FRAME_WIDTH;
    cfg.frame_height = FRAME_HEIGHT;
    cfg.enable_depth = true;
    cfg.enable_validation = false; // Benchmark 时关闭验证层以获得最大性能

    auto start_init = std::chrono::high_resolution_clock::now();
    auto renderer = BatchRenderer::Create(models, cfg);
    auto end_init = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(end_init - start_init).count();

    if (!renderer || !renderer->IsValid()) {
        std::fprintf(stderr, "[Fatal] BatchRenderer::Create failed\n");
        return 2;
    }
    printf("[Init] Renderer initialized in %.2f ms\n", init_ms);

    // ---- 3) 准备数据 (mjData) ----
    std::vector<mjData*> datas(BATCH_SIZE, nullptr);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        datas[i] = mj_makeData(models[i]);
        mj_forward(models[i], datas[i]);
    }

    // ---- 4) 热身 (Warmup) ----
    // 让 GPU 流水线填满，让驱动编译完 Shader
    printf("[Run] Warming up (%d frames)...\n", WARMUP_STEPS);
    for (int i = 0; i < WARMUP_STEPS; ++i) {
        // 稍微动一下物体
        datas[0]->qpos[0] = 0.1f * i; 
        mj_forward(models[0], datas[0]);
        renderer->Render(datas.data(), nullptr);
    }

    // ---- 5) 正式测试循环 ----
    printf("[Run] Running Benchmark (%d frames)...\n", BENCHMARK_STEPS);
    
    std::vector<double> frame_times;
    frame_times.reserve(BENCHMARK_STEPS);

    for (int step = 0; step < BENCHMARK_STEPS; ++step) {
        // A. 更新物理状态 (模拟真实训练中的 step)
        // 我们让所有机器人的关节都在动，确保 Render 需要上传新的 Transform 数据
        double time_val = step * 0.05;
        for (int i = 0; i < BATCH_SIZE; ++i) {
            if (models[i]->nq > 0) {
                // 简单的正弦运动
                datas[i]->qpos[0] = std::sin(time_val + i * 0.01);
            }
            mj_forward(models[i], datas[i]);
        }

        // B. 渲染并计时
        auto t1 = std::chrono::high_resolution_clock::now();
        
        auto result = renderer->Render(datas.data(), nullptr);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        double dt_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        frame_times.push_back(dt_ms);

        if (!result) {
            std::fprintf(stderr, "[Error] Render failed at step %d\n", step);
            break;
        }

        // 简单的进度条
        if (step % 100 == 0) {
            printf("\r   Step %d/%d | Last Frame: %.2f ms", step, BENCHMARK_STEPS, dt_ms);
            fflush(stdout);
        }
    }
    printf("\n");

    // ---- 6) 统计分析 ----
    if (!frame_times.empty()) {
        double total_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0);
        double avg_time = total_time / frame_times.size();
        double min_time = *std::min_element(frame_times.begin(), frame_times.end());
        double max_time = *std::max_element(frame_times.begin(), frame_times.end());
        
        // P99 计算
        std::vector<double> sorted_times = frame_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        double p99_time = sorted_times[static_cast<size_t>(sorted_times.size() * 0.99)];
        
        double fps = 1000.0 / avg_time;

        printf("\n============ RESULTS ============\n");
        printf("Backend:      Current System\n");
        printf("Frames:       %d\n", (int)frame_times.size());
        printf("Throughput:   \033[1;32m%.2f FPS\033[0m\n", fps);
        printf("Latency (Avg): %.3f ms\n", avg_time);
        printf("Latency (Min): %.3f ms\n", min_time);
        printf("Latency (Max): %.3f ms\n", max_time);
        printf("Latency (P99): %.3f ms\n", p99_time);
        printf("=================================\n");
    }

    // ---- 7) 验证输出 (只保存第 0 个环境的图) ----
    const unsigned char* img = renderer->GetRGBFrame(0);
    if (img) {
        std::string filename = "benchmark_sample_0.ppm";
        if (write_ppm(filename, img, FRAME_WIDTH, FRAME_HEIGHT)) {
            printf("[Info] Saved sample image to %s\n", filename.c_str());
        }
    }

    // ---- 8) 清理资源 ----
    // 释放 datas
    for (auto* d : datas) mj_deleteData(d);
    
    // 释放 renderer (这里 reset 会触发 renderer 析构，释放 vulkan 资源)
    renderer.reset();

    // 释放 models (必须在 renderer 释放后释放，如果 renderer 不持有 model 所有权)
    for (auto* m : models) mj_deleteModel(m);

    return 0;
}