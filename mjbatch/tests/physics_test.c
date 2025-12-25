#include <mujoco/mujoco.h>
#include <cstdio>
#include <vector>
#include <thread>
#include <chrono>
#include <cmath>
#include <string>
#include <algorithm>
#include <queue>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <iomanip>

// 

// ---- 配置参数 ----
const int BATCH_SIZE = 32;           
const int BENCHMARK_STEPS = 1000;    
const int WARMUP_STEPS = 100;
const std::string MODEL_XML = "/home/hpf/project/vulkan/mujoco/mujoco/build/model/lift.xml";

// ==========================================
// 工具类：模拟旧架构的线程池 (用于 Mode A)
// ==========================================
class SyncThreadPool {
public:
    SyncThreadPool(int num_threads) : stop(false) {
        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                    // 任务完成，原子计数减一并通知主线程
                    tasks_remaining--;
                    done_condition.notify_one();
                }
            });
        }
    }

    ~SyncThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) worker.join();
    }

    // 模拟 ParallelFor：主线程必须在这里阻塞，直到所有子任务完成
    void ParallelFor(int count, std::function<void(int, int)> func) {
        int num_workers = workers.size();
        int items_per_worker = count / num_workers;
        int remainder = count % num_workers;

        tasks_remaining = num_workers; // 重置计数器

        int current_idx = 0;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            for (int t = 0; t < num_workers; ++t) {
                int end = current_idx + items_per_worker + (t < remainder ? 1 : 0);
                // 推送任务
                tasks.emplace([func, current_idx, end]() {
                    func(current_idx, end);
                });
                current_idx = end;
            }
        }
        condition.notify_all(); // 唤醒所有线程

        // --- 关键瓶颈 ---
        // 主线程必须在这里死等所有线程做完这一步，才能继续
        std::unique_lock<std::mutex> lock(done_mutex);
        done_condition.wait(lock, [this] { return tasks_remaining == 0; });
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

    // 同步完成机制
    std::mutex done_mutex;
    std::condition_variable done_condition;
    std::atomic<int> tasks_remaining;
};

// ==========================================
// 核心逻辑
// ==========================================

void setup_scene(std::vector<mjModel*>& models, std::vector<mjData*>& datas, mjModel* base_model) {
    models[0] = base_model;
    datas[0] = mj_makeData(base_model);
    for (int i = 1; i < BATCH_SIZE; ++i) {
        models[i] = mj_copyModel(nullptr, base_model);
        datas[i] = mj_makeData(models[i]);
    }
    // Initial Forward
    for (int i = 0; i < BATCH_SIZE; ++i) mj_forward(models[i], datas[i]);
}

// -------------------------------------------------
// Mode A: 细粒度同步 (模拟之前的 C++ 实现)
// 结构： Loop { Barrier -> Parallel Compute -> Barrier }
// -------------------------------------------------
double run_mode_sync(SyncThreadPool& pool, 
                     const std::vector<mjModel*>& models, 
                     const std::vector<mjData*>& datas) {
    
    auto t_start = std::chrono::high_resolution_clock::now();

    // 循环在主线程！
    for (int step = 0; step < BENCHMARK_STEPS; ++step) {
        double time_val = step * 0.002;

        // 每一帧都要调用一次 ParallelFor，产生巨大的同步开销
        pool.ParallelFor(BATCH_SIZE, [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                if (models[i]->nu > 0) datas[i]->ctrl[0] = std::sin(time_val);
                mj_step(models[i], datas[i]);
            }
        });
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t_end - t_start).count();
}

// -------------------------------------------------
// Mode B: 粗粒度并行 (模拟 Python Multiprocessing)
// 结构： Thread { Loop -> Compute } -> Join
// -------------------------------------------------
void mode_b_worker(int start, int end, 
                   const std::vector<mjModel*>& models, 
                   const std::vector<mjData*>& datas) {
    double time_val = 0.0;
    // 循环在线程内部！没有 Barrier！
    for (int step = 0; step < BENCHMARK_STEPS; ++step) {
        time_val += 0.002;
        for (int i = start; i < end; ++i) {
            if (models[i]->nu > 0) datas[i]->ctrl[0] = std::sin(time_val);
            mj_step(models[i], datas[i]);
        }
    }
}

double run_mode_async(int num_threads, 
                      const std::vector<mjModel*>& models, 
                      const std::vector<mjData*>& datas) {
    
    std::vector<std::thread> workers;
    int items_per_thread = BATCH_SIZE / num_threads;
    int remainder = BATCH_SIZE % num_threads;

    auto t_start = std::chrono::high_resolution_clock::now();

    int current_idx = 0;
    for (int t = 0; t < num_threads; ++t) {
        int end = current_idx + items_per_thread + (t < remainder ? 1 : 0);
        workers.emplace_back(mode_b_worker, current_idx, end, std::ref(models), std::ref(datas));
        current_idx = end;
    }

    for (auto& w : workers) w.join();

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(t_end - t_start).count();
}

// ==========================================
// Main
// ==========================================
int main() {
    int hw_threads = std::thread::hardware_concurrency();
    int num_threads = std::min(hw_threads, BATCH_SIZE);

    printf("========================================================\n");
    printf("   MuJoCo Physics: Sync vs Async Architecture Benchmark\n");
    printf("========================================================\n");
    printf("Batch Size:    %d\n", BATCH_SIZE);
    printf("Steps:         %d\n", BENCHMARK_STEPS);
    printf("Threads:       %d\n", num_threads);
    
    // Load Model
    char error[1024] = {0};
    mjModel* base_model = mj_loadXML(MODEL_XML.c_str(), nullptr, error, sizeof(error));
    if (!base_model) return 1;

    std::vector<mjModel*> models(BATCH_SIZE);
    std::vector<mjData*> datas(BATCH_SIZE);
    setup_scene(models, datas, base_model);

    // Warmup
    printf("Warming up...\n");
    for(int i=0; i<BATCH_SIZE; ++i) mj_step(models[i], datas[i]);

    // ---- Test Mode A: Synchronous (Old C++ Style) ----
    printf("\n[Test A] Running Synchronous Mode (Thread-inside-Loop)...\n");
    printf("   Logic: Main Loop -> Wake Threads -> Wait Threads -> Next Step\n");
    
    SyncThreadPool pool(num_threads); // 创建持久化线程池
    double time_sync = run_mode_sync(pool, models, datas);
    
    double fps_sync = (double)(BATCH_SIZE * BENCHMARK_STEPS) / time_sync;
    printf("   -> Time: %.4f s | Throughput: %.2f Steps/s\n", time_sync, fps_sync);


    // Reset Data for fairness
    for(int i=0; i<BATCH_SIZE; ++i) mj_resetData(models[i], datas[i]);


    // ---- Test Mode B: Asynchronous (Python Style) ----
    printf("\n[Test B] Running Asynchronous Mode (Loop-inside-Thread)...\n");
    printf("   Logic: Launch Threads -> Threads Run All Steps -> Join\n");

    double time_async = run_mode_async(num_threads, models, datas);

    double fps_async = (double)(BATCH_SIZE * BENCHMARK_STEPS) / time_async;
    printf("   -> Time: %.4f s | Throughput: %.2f Steps/s\n", time_async, fps_async);


    // ---- Summary ----
    printf("\n================ SUMMARY ================\n");
    printf("Mode A (Sync Barrier):  %9.2f Steps/s\n", fps_sync);
    printf("Mode B (Async Free):    %9.2f Steps/s\n", fps_async);
    printf("-----------------------------------------\n");
    
    double speedup = fps_async / fps_sync;
    if (speedup > 1.0) {
        printf("Performance Gain:       \033[92m%.2fx FASTER\033[0m\n", speedup);
        printf("Conclusion: Synchronization overhead was the bottleneck.\n");
    } else {
        printf("Performance Gain:       %.2fx (Negligible)\n", speedup);
        printf("Conclusion: Compute bound, overhead is not the issue.\n");
    }
    printf("=========================================\n");

    // Cleanup
    for (auto* d : datas) mj_deleteData(d);
    for (auto* m : models) mj_deleteModel(m);

    return 0;
}