/*
g++ -shared -fPIC -O2 -o libvkstats.so vk_stats_advanced.cpp \
    -I/home/hpf/vulkan/1.4.328.1/x86_64/include \
    -L/home/hpf/vulkan/1.4.328.1/x86_64/lib \
    -ldl -lvulkan
    
cd /home/hpf/project/vulkan/mujoco/mujoco/build
LD_PRELOAD=/home/hpf/project/vulkan/mujoco/mujoco/mjbatch/tests/libvkstats.so ./bin/benchmark_test
*/
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <vulkan/vulkan.h>
#include <map>
#include <mutex>
#include <atomic>
#include <string.h>

// =============================================================
// 1. 统计逻辑 (与之前相同)
// =============================================================
struct MemoryStats {
    std::atomic<size_t> current_allocated_bytes{0};
    std::atomic<size_t> total_allocated_bytes{0};
    std::atomic<size_t> allocation_count{0};
};

static MemoryStats g_stats;
static std::mutex g_map_lock;
static std::map<VkDeviceMemory, VkDeviceSize> g_alloc_map;

// =============================================================
// 2. 函数指针定义
// =============================================================
// 原始的 dlsym
typedef void* (*PFN_dlsym)(void* handle, const char* symbol);
static PFN_dlsym real_dlsym_ptr = nullptr;

// 原始的 Vulkan 函数
static PFN_vkAllocateMemory real_vkAllocateMemory = nullptr;
static PFN_vkFreeMemory     real_vkFreeMemory     = nullptr;
static PFN_vkGetInstanceProcAddr real_vkGetInstanceProcAddr = nullptr;
static PFN_vkGetDeviceProcAddr   real_vkGetDeviceProcAddr   = nullptr;

// =============================================================
// 3. 我们的 Hook 函数实现
// =============================================================

// 拦截分配
VKAPI_ATTR VkResult VKAPI_CALL Hooked_vkAllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory) 
{
    // 确保有原始函数 (防止直接调用 crashing)
    if (!real_vkAllocateMemory) {
        // 紧急恢复：通常会在 GetProcAddr 阶段拿到，但以防万一
        fprintf(stderr, "[vk_stats] FATAL: real_vkAllocateMemory not loaded!\n"); 
        return VK_ERROR_UNKNOWN;
    }

    VkResult result = real_vkAllocateMemory(device, pAllocateInfo, pAllocator, pMemory);

    if (result == VK_SUCCESS && pAllocateInfo && pMemory) {
        VkDeviceSize size = pAllocateInfo->allocationSize;
        g_stats.current_allocated_bytes += size;
        g_stats.total_allocated_bytes += size;
        g_stats.allocation_count++;

        {
            std::lock_guard<std::mutex> lock(g_map_lock);
            g_alloc_map[*pMemory] = size;
        }

        printf("[vk_stats] ALLOC: %6.2f MB | Total: %6.2f MB\n",
               (double)size / (1024*1024),
               (double)g_stats.current_allocated_bytes / (1024*1024));
    }
    return result;
}

// 拦截释放
VKAPI_ATTR void VKAPI_CALL Hooked_vkFreeMemory(
    VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator) 
{
    if (memory != VK_NULL_HANDLE) {
        std::lock_guard<std::mutex> lock(g_map_lock);
        auto it = g_alloc_map.find(memory);
        if (it != g_alloc_map.end()) {
            size_t size = it->second;
            if (g_stats.current_allocated_bytes >= size)
                g_stats.current_allocated_bytes -= size;
            g_alloc_map.erase(it);
            printf("[vk_stats] FREE : %6.2f MB | Total: %6.2f MB\n",
                   (double)size / (1024*1024),
                   (double)g_stats.current_allocated_bytes / (1024*1024));
        }
    }
    if (real_vkFreeMemory) real_vkFreeMemory(device, memory, pAllocator);
}

// -------------------------------------------------------------
// 核心：拦截 GetInstanceProcAddr 和 GetDeviceProcAddr
// -------------------------------------------------------------
// Vulkan 加载器通过这个函数来查找所有其他函数。我们要在这里把 result 替换成我们的 Hook。

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetInstanceProcAddr(VkInstance instance, const char* pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetDeviceProcAddr(VkDevice device, const char* pName);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    // 1. 获取真正的函数地址
    PFN_vkVoidFunction func = real_vkGetInstanceProcAddr(instance, pName);
    if (!func) return nullptr;

    // 2. 如果请求的是我们要监控的函数，返回我们的 Hook
    if (strcmp(pName, "vkAllocateMemory") == 0) {
        real_vkAllocateMemory = (PFN_vkAllocateMemory)func; // 保存真地址
        return (PFN_vkVoidFunction)Hooked_vkAllocateMemory; // 返回假地址
    }
    if (strcmp(pName, "vkFreeMemory") == 0) {
        real_vkFreeMemory = (PFN_vkFreeMemory)func;
        return (PFN_vkVoidFunction)Hooked_vkFreeMemory;
    }
    
    // 3. 也要拦截获取地址的函数本身（递归链）
    if (strcmp(pName, "vkGetInstanceProcAddr") == 0) return (PFN_vkVoidFunction)Hooked_vkGetInstanceProcAddr;
    if (strcmp(pName, "vkGetDeviceProcAddr") == 0) {
        real_vkGetDeviceProcAddr = (PFN_vkGetDeviceProcAddr)func;
        return (PFN_vkVoidFunction)Hooked_vkGetDeviceProcAddr;
    }

    return func;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetDeviceProcAddr(VkDevice device, const char* pName) {
    PFN_vkVoidFunction func = real_vkGetDeviceProcAddr(device, pName);
    if (!func) return nullptr;

    if (strcmp(pName, "vkAllocateMemory") == 0) {
        real_vkAllocateMemory = (PFN_vkAllocateMemory)func;
        return (PFN_vkVoidFunction)Hooked_vkAllocateMemory;
    }
    if (strcmp(pName, "vkFreeMemory") == 0) {
        real_vkFreeMemory = (PFN_vkFreeMemory)func;
        return (PFN_vkVoidFunction)Hooked_vkFreeMemory;
    }
    
    return func;
}

// =============================================================
// 4. 拦截 dlsym (终极入口)
// =============================================================
// MuJoCo 调用 dlsym(lib, "vkGetInstanceProcAddr") 时会进入这里
extern "C" void* dlsym(void* handle, const char* symbol) {
    // 1. 获取真正的 dlsym (如果还没获取)
    if (!real_dlsym_ptr) {
        // 使用 dlvsym 查找 GLIBC 版本的 dlsym，避免递归死循环
        // "GLIBC_2.2.5" 是 x86_64 Linux 上最常见的版本，如果报错可以尝试 dlvsym(RTLD_NEXT, "dlsym", NULL)
        real_dlsym_ptr = (PFN_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym_ptr) {
             fprintf(stderr, "[vk_stats] Error: failed to find real dlsym\n");
             exit(1);
        }
    }

    // 2. 调用真正的 dlsym 获取结果
    void* result = real_dlsym_ptr(handle, symbol);

    // 3. 检查是否是 Vulkan 的入口函数
    if (symbol && strcmp(symbol, "vkGetInstanceProcAddr") == 0) {
        printf("[vk_stats] Intercepted dlsym(vkGetInstanceProcAddr)!\n");
        // 保存真正的入口地址
        real_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)result;
        // 返回我们的入口地址
        return (void*)Hooked_vkGetInstanceProcAddr;
    }

    return result;
}