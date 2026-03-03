// ============================================================
// profiler.h
// Drop-in CPU profiler for BatchRenderer
// ============================================================

#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
#include <functional>

struct RenderProfiler
{
    using Clock = std::chrono::high_resolution_clock;

    struct Stat
    {
        double ms = 0.0;
    };

    std::unordered_map<std::string, Stat> stats;

    // Geometry statistics
    uint64_t total_geoms = 0;
    uint64_t visible_geoms = 0;
    uint64_t culled_geoms = 0;

    void Reset()
    {
        stats.clear();
        total_geoms = 0;
        visible_geoms = 0;
        culled_geoms = 0;
    }

    void AddTime(const std::string& name, double ms)
    {
        stats[name].ms += ms;
    }

    void Print()
    {
        std::cout << "\n================ Frame Timing Breakdown (Tree View) ================\n";

        // 1. 识别顶层节点 (1., 2., 3., 4.)
        std::vector<std::string> top_levels = {"1. Update_From_SHM", "2. Record_CommandBuffers", "3. Submit_And_Wait", "4. Readback"};
        double total_frame_ms = 0;
        for(const auto& tl : top_levels) if(stats.count(tl)) total_frame_ms += stats[tl].ms;

        // 辅助 lambda：递归打印树形结构
        std::function<void(const std::string&, int)> print_node = [&](const std::string& name, int indent) {
            if (stats.find(name) == stats.end()) return;

            double current_ms = stats[name].ms;
            double pct = (total_frame_ms > 0) ? (current_ms / total_frame_ms * 100.0) : 0;

            // 打印当前行
            std::string prefix = "";
            for(int i=0; i<indent; ++i) prefix += "  ";
            if(indent > 0) prefix += "└─ ";

            std::cout << std::setw(45) << std::left << (prefix + name)
                    << std::setw(10) << std::right << std::fixed << std::setprecision(3) << current_ms << " ms"
                    << std::setw(8) << std::right << std::setprecision(1) << pct << "%\n";

            // 查找所有子节点 (例如名称以 "name." 开头的)
            std::vector<std::pair<double, std::string>> children;
            double children_sum = 0;
            for (auto& kv : stats) {
                const std::string& k = kv.first;
                // 确保是直接子节点：前缀匹配且只有一个额外的点
                if (k.size() > name.size() + 1 && k.substr(0, name.size() + 1) == (name + ".")) {
                    // 进一步检查是否是“深层”子节点（跳过孙子辈，留给递归处理）
                    if (k.find('.', name.size() + 1) == std::string::npos) {
                        children.push_back({kv.second.ms, k});
                        children_sum += kv.second.ms;
                    }
                }
            }

            // 按耗时对子节点排序
            std::sort(children.rbegin(), children.rend());

            for (auto& child : children) {
                print_node(child.second, indent + 1);
            }

            // 【关键】显示未统计到的差值 (Gap Analysis)
            if (!children.empty() && (current_ms - children_sum) > 0.01) { // 忽略微小误差
                double gap = current_ms - children_sum;
                std::string gap_prefix = "";
                for(int i=0; i<=indent; ++i) gap_prefix += "  ";
                gap_prefix += "└─ [Unaccounted/Overhead]";
                
                std::cout << std::setw(45) << std::left << gap_prefix
                        << std::setw(10) << std::right << std::fixed << std::setprecision(3) << gap << " ms"
                        << std::setw(8) << std::right << std::setprecision(1) << (gap/total_frame_ms*100.0) << "%\n";
            }
        };

        // 依次从顶层开始递归
        for (const auto& tl : top_levels) {
            print_node(tl, 0);
        }

        if (total_geoms > 0)
        {
            std::cout << "Total Geoms:  " << total_geoms << "\n";
            std::cout << "Visible:      " << visible_geoms << "\n";
            std::cout << "Culled:       " << culled_geoms << "\n";
            std::cout << "Cull Ratio:   "
                    << (double)culled_geoms / (double)total_geoms * 100.0
                    << "%\n";
        }

        std::cout << "========================================================\n\n";
    }
};

struct ScopeTimer
{
    RenderProfiler& profiler;
    std::string name;
    RenderProfiler::Clock::time_point start;

    ScopeTimer(RenderProfiler& p, const std::string& n)
        : profiler(p), name(n), start(RenderProfiler::Clock::now()) {}

    ~ScopeTimer()
    {
        auto end = RenderProfiler::Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        profiler.AddTime(name, ms);
    }
};