/*
cd /home/hpf/project/vulkan/mujoco/mujoco/mjbatch/tests
g++ -shared -fPIC -O2 -o libvkstats.so vkstats.cpp \
    -I/home/hpf/vulkan/1.4.328.1/x86_64/include \
    -L/home/hpf/vulkan/1.4.328.1/x86_64/lib \
    -ldl -lvulkan

cd /home/hpf/project/vulkan/mujoco/mujoco/build
LD_PRELOAD=/home/hpf/project/vulkan/mujoco/mujoco/mjbatch/tests/libvkstats.so ./bin/benchmark_test

// use addr2line to find the code line of the addresses printed in the call stack 
addr2line -e ./bin/benchmark_test 0x418c0 0xf033 0x11ad3 0xc6df

*/
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <vulkan/vulkan.h>
#include <map>
#include <mutex>
#include <atomic>
#include <string.h>

#include <execinfo.h>
#include <cxxabi.h> 

struct MemoryStats {
    std::atomic<size_t> current_allocated_bytes{0};
    std::atomic<size_t> total_allocated_bytes{0};
    std::atomic<size_t> allocation_count{0};
};

static MemoryStats g_stats;
static std::mutex g_map_lock;
static std::map<VkDeviceMemory, VkDeviceSize> g_alloc_map;


typedef void* (*PFN_dlsym)(void* handle, const char* symbol);
static PFN_dlsym real_dlsym_ptr = nullptr;

static PFN_vkAllocateMemory real_vkAllocateMemory = nullptr;
static PFN_vkFreeMemory     real_vkFreeMemory     = nullptr;
static PFN_vkGetInstanceProcAddr real_vkGetInstanceProcAddr = nullptr;
static PFN_vkGetDeviceProcAddr   real_vkGetDeviceProcAddr   = nullptr;

/*
 * used to print call stack 
*/
void print_callstack() {
    void* callstack[128];
    int frames = backtrace(callstack, 128);
    char** strs = backtrace_symbols(callstack, frames);

    if (strs == nullptr) {
        printf("  [Backtrace failed]\n");
        return;
    }

    printf("  [Call Stack]:\n");
    for (int i = 0; i < frames; ++i) {
        char* name_start = NULL;
        char* name_end = NULL;
        
        for (char* p = strs[i]; *p; ++p) {
            if (*p == '(') name_start = p;
            else if (*p == '+') name_end = p;
        }

        if (name_start && name_end && name_start < name_end) {
            *name_start = 0; 
            *name_end = 0;
            
            char* mangled_name = name_start + 1;
            int status;

            char* real_name = abi::__cxa_demangle(mangled_name, 0, 0, &status);
            
            *name_start = '(';
            *name_end = '+';

            if (status == 0) {
                printf("    #%d %s : %s\n", i, strs[i], real_name);
                free(real_name);
            } else {
                printf("    #%d %s\n", i, strs[i]);
            }
        } else {
            printf("    #%d %s\n", i, strs[i]);
        }
    }
    free(strs);
}


VKAPI_ATTR VkResult VKAPI_CALL Hooked_vkAllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo,
    const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory) 
{
    if (!real_vkAllocateMemory) return VK_ERROR_UNKNOWN;

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

        // Only print >10MB allocations
        if (size >= 10 * 1024 * 1024) { 
            printf("\n[vk_stats] >>> LARGE ALLOC DETECTED: %6.2f MB <<<\n", (double)size / (1024*1024));
            printf("[vk_stats] Total Usage: %6.2f MB\n", (double)g_stats.current_allocated_bytes / (1024*1024));
            print_callstack();
            printf("----------------------------------------------------------\n\n");
        }
    }
    return result;
}

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
        }
    }
    if (real_vkFreeMemory) real_vkFreeMemory(device, memory, pAllocator);
}


VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetInstanceProcAddr(VkInstance instance, const char* pName);
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetDeviceProcAddr(VkDevice device, const char* pName);

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL Hooked_vkGetInstanceProcAddr(VkInstance instance, const char* pName) {
    PFN_vkVoidFunction func = real_vkGetInstanceProcAddr(instance, pName);
    if (!func) return nullptr;
    if (strcmp(pName, "vkAllocateMemory") == 0) {
        real_vkAllocateMemory = (PFN_vkAllocateMemory)func;
        return (PFN_vkVoidFunction)Hooked_vkAllocateMemory;
    }
    if (strcmp(pName, "vkFreeMemory") == 0) {
        real_vkFreeMemory = (PFN_vkFreeMemory)func;
        return (PFN_vkVoidFunction)Hooked_vkFreeMemory;
    }
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

extern "C" void* dlsym(void* handle, const char* symbol) {
    if (!real_dlsym_ptr) {
        real_dlsym_ptr = (PFN_dlsym)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym_ptr) exit(1);
    }
    void* result = real_dlsym_ptr(handle, symbol);
    if (symbol && strcmp(symbol, "vkGetInstanceProcAddr") == 0) {
        real_vkGetInstanceProcAddr = (PFN_vkGetInstanceProcAddr)result;
        return (void*)Hooked_vkGetInstanceProcAddr;
    }
    return result;
}