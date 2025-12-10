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
#include <vector> 

// ---- Benchmark config ----
const int BATCH_SIZE = 32;      
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
const int BENCHMARK_STEPS = 100; 
const int WARMUP_STEPS = 10;

// hard code the model path
const std::string MODEL_XML = "/home/hpf/project/vulkan/mujoco/mujoco/build/model/lift.xml";

// Input: rgba_data from BatchRenderer::GetRGBFrame()
// Output: standard PPM file
static bool write_ppm(const std::string& path, const unsigned char* rgba_data, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P6\n" << w << " " << h << "\n255\n";

    std::vector<unsigned char> row_buffer(w * 3);

    for (int y = 0; y < h; ++y) {
        const unsigned char* src_row = rgba_data + (size_t)y * w * 4; // 源指针 (RGBA)
        unsigned char* dst_row = row_buffer.data();                   // 目标指针 (RGB)

        for (int x = 0; x < w; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 4 + 0]; // R
            dst_row[x * 3 + 1] = src_row[x * 4 + 1]; // G
            dst_row[x * 3 + 2] = src_row[x * 4 + 2]; // B
            // src_row[x * 4 + 3] (Alpha) ignored
        }

        ofs.write(reinterpret_cast<const char*>(row_buffer.data()), w * 3);
    }

    return ofs.good();
}

int main() {
    nvtxNameOsThreadA(pthread_self(), "Main-Thread");

    int hw_threads = std::thread::hardware_concurrency();
    int pool_size = (hw_threads > 1) ? hw_threads : 1;

    printf("======================================\n");
    printf("   Vulkan Renderer Benchmark (NVTX Instrumented)\n");
    printf("   Batch Size:  %d\n", BATCH_SIZE);
    printf("   Pool Size:   %d Threads\n", pool_size);
    printf("======================================\n");

    ThreadPool pool(pool_size);

    char error[1024] = {0};
    mjModel* base_model = mj_loadXML(MODEL_XML.c_str(), nullptr, error, sizeof(error));
    printf("Loaded model from '%s'\n", MODEL_XML.c_str());
    if (!base_model) return 1;

    std::vector<mjModel*> models;
    models.reserve(BATCH_SIZE);
    models.push_back(base_model);
    for (int i = 1; i < BATCH_SIZE; ++i) {
        models.push_back(mj_copyModel(nullptr, base_model));
    }

    BatchRendererConfig cfg;
    cfg.batch_size = BATCH_SIZE;
    cfg.frame_width = FRAME_WIDTH;
    cfg.frame_height = FRAME_HEIGHT;
    cfg.enable_depth = true;
    cfg.enable_validation = false; 

    auto renderer = BatchRenderer::Create(models, cfg);
    if (!renderer || !renderer->IsValid()) return 2;

    std::vector<mjData*> datas(BATCH_SIZE, nullptr);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        datas[i] = mj_makeData(models[i]);
        mj_forward(models[i], datas[i]);
    }

    // Warmup
    for (int i = 0; i < WARMUP_STEPS; ++i) {
        pool.ParallelFor(BATCH_SIZE, [&](int start, int end) {
            for (int j = start; j < end; ++j) {
                datas[j]->qpos[0] = 0.1f * i;
                mj_forward(models[j], datas[j]);
            }
        });
        renderer->Render(datas.data(), nullptr);
    }

    printf("[Run] Running Benchmark (%d frames)...\n", BENCHMARK_STEPS);
    
    std::vector<double> render_times;     
    std::vector<double> total_loop_times; 
    render_times.reserve(BENCHMARK_STEPS);
    total_loop_times.reserve(BENCHMARK_STEPS);

    for (int step = 0; step < BENCHMARK_STEPS; ++step) {
        // [NVTX] 3. sign frame loop (Blue)
        ScopedNvtxRange frameRange("Frame_Loop", COLOR_LOOP);

        auto t_loop_start = std::chrono::high_resolution_clock::now();

        double time_val = step * 0.05;
        
        {
            // [NVTX] 4. sign physics (Green)
            ScopedNvtxRange physRange("Physics_Step_Total", COLOR_PHYSICS);

            pool.ParallelFor(BATCH_SIZE, [&](int start, int end) {
                for (int i = start; i < end; ++i) {
                    if (models[i]->nq > 0) {
                        datas[i]->qpos[0] = std::sin(time_val + i * 0.01);
                    }
                    mj_forward(models[i], datas[i]);
                }
            });
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        RenderResult result;
        {
            // [NVTX] 5. sign the render_submit (Red)
            ScopedNvtxRange renderRange("Render_Submit", COLOR_RENDER);
            result = renderer->Render(datas.data(), nullptr);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        
        double render_dt = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double loop_dt = std::chrono::duration<double, std::milli>(t2 - t_loop_start).count();
        
        render_times.push_back(render_dt);
        total_loop_times.push_back(loop_dt);

        if (!result) break;

        if (step % 20 == 0) {
            printf("\r   Step %d/%d | Render: %.2f ms | Loop: %.2f ms", step, BENCHMARK_STEPS, render_dt, loop_dt);
            fflush(stdout);
        }
    }
    printf("\n");

    if (!render_times.empty()) {
        auto calc_stats = [](const std::vector<double>& times, const char* label) {
            double total = std::accumulate(times.begin(), times.end(), 0.0);
            double avg = total / times.size();
            printf("--- %s ---\n  Avg: %.3f ms\n", label, avg);
        };
        printf("\n============ RESULTS ============\n");
        calc_stats(render_times, "Render Only");
        calc_stats(total_loop_times, "Total Step");
    }

    const unsigned char* img = renderer->GetRGBFrame(0);
    if (img) write_ppm("benchmark_sample_0.ppm", img, FRAME_WIDTH, FRAME_HEIGHT);

    for (auto* d : datas) mj_deleteData(d);
    renderer.reset(); 
    for (auto* m : models) mj_deleteModel(m);

    return 0;
}

