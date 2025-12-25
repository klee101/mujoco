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
#include <iomanip> // for std::setprecision

// ---- Benchmark Config ----
const int BATCH_SIZE = 16;
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int BENCHMARK_STEPS = 50;
const int WARMUP_STEPS = 10;
const std::string MODEL_XML = "/home/hpf/project/vulkan/mujoco/mujoco/build/model/lift.xml";


// ---- NVTX 辅助函数 (解决 C++ 严格语法报错) ----
void PushRange(const char* name, uint32_t color) {
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = name;
    nvtxRangePushEx(&eventAttrib);
}

struct StepTimeline {
    double p_start; 
    double p_end;   
    double r_start; 
    double r_end;   
};

void analyze_timeline(const std::vector<StepTimeline>& logs) {
    printf("\n%-8s | %-12s | %-12s | %-12s | %-10s | %-10s\n", 
           "Step", "Phys Start", "Phys End", "Render End", "Phys Dur", "Ren Dur");
    printf("------------------------------------------------------------------------------------\n");

    for (int i = 0; i < std::min((int)logs.size(), 10); ++i) {
        const auto& s = logs[i];
        printf("Step %-3d | +%-10.3fms | +%-10.3fms | +%-10.3fms | %-8.3fms | %-8.3fms\n", 
               i, s.p_start, s.p_end, s.r_end, s.p_end - s.p_start, s.r_end - s.r_start);
    }
    
    double avg_p = 0, avg_r = 0;
    for(const auto& s : logs) {
        avg_p += (s.p_end - s.p_start);
        avg_r += (s.r_end - s.r_start);
    }
    printf("------------------------------------------------------------------------------------\n");
    printf("Average (over %zu steps): Physics: %.3f ms | Render: %.3f ms\n", logs.size(), avg_p/logs.size(), avg_r/logs.size());
}

int main() {
    // 1. 初始化模型
    char error[1024];
    mjModel* m = mj_loadXML(MODEL_XML.c_str(), nullptr, error, sizeof(error));
    if (!m) { printf("Load error: %s\n", error); return 1; }

    std::vector<mjModel*> models(BATCH_SIZE, m);
    for(int i = 1; i < BATCH_SIZE; ++i) models[i] = mj_copyModel(nullptr, m);
    
    std::vector<mjData*> datas(BATCH_SIZE);
    for(int i = 0; i < BATCH_SIZE; ++i) datas[i] = mj_makeData(models[i]);

    // 2. 初始化环境
    int hw_threads = std::thread::hardware_concurrency();
    ThreadPool pool(hw_threads);
    
    BatchRendererConfig cfg;
    cfg.batch_size = BATCH_SIZE;
    cfg.frame_width = FRAME_WIDTH;
    cfg.frame_height = FRAME_HEIGHT;
    cfg.enable_validation = false;
    auto renderer = BatchRenderer::Create(models, cfg);

    // 3. Warmup
    for (int i = 0; i < WARMUP_STEPS; ++i) {
        pool.ParallelFor(BATCH_SIZE, [&](int s, int e) {
            for(int j=s; j<e; ++j) mj_step(models[j], datas[j]);
        });
        renderer->Render(datas.data(), nullptr);
    }

    // 4. 主循环与记录
    std::vector<StepTimeline> timeline_logs;
    timeline_logs.reserve(BENCHMARK_STEPS);

    auto global_t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < BENCHMARK_STEPS; ++step) {
        StepTimeline ts;

        // --- Physics ---
        auto t_p_start = std::chrono::high_resolution_clock::now();
        ts.p_start = std::chrono::duration<double, std::milli>(t_p_start - global_t0).count();

        PushRange("Physics", COLOR_PHYSICS);
        pool.ParallelFor(BATCH_SIZE, [&](int s, int e) {
            for(int i = s; i < e; ++i) mj_step(models[i], datas[i]);
        });
        nvtxRangePop();

        auto t_p_end = std::chrono::high_resolution_clock::now();
        ts.p_end = std::chrono::duration<double, std::milli>(t_p_end - global_t0).count();

        // --- Render ---
        ts.r_start = ts.p_end; 
        
        PushRange("Render", COLOR_RENDER);
        renderer->Render(datas.data(), nullptr);
        nvtxRangePop();

        auto t_r_end = std::chrono::high_resolution_clock::now();
        ts.r_end = std::chrono::duration<double, std::milli>(t_r_end - global_t0).count();

        timeline_logs.push_back(ts);
    }

    // 5. 输出
    analyze_timeline(timeline_logs);

    // 6. 清理
    for(auto* d : datas) mj_deleteData(d);
    for(int i = 1; i < BATCH_SIZE; ++i) mj_deleteModel(models[i]);
    mj_deleteModel(m);

    return 0;
}