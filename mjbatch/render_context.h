#pragma once 

#include "device.h"
#include "memory.h"
#include "backend.h"
#include <vulkan/vulkan.h>

namespace mujoco{
namespace mjbatch {

// RenderContext：a global context all scenes need to be batch rendered 

struct RenderContext
{
    Device &device;
    Backend &backend;
    MemoryAllocator allocator;

    VkQueue renderQueue = VK_NULL_HANDLE;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkRenderPass shadowPass = VK_NULL_HANDLE;

    VkCommandPool load_cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer load_cmd_ = VK_NULL_HANDLE;
    VkFence load_fence_ = VK_NULL_HANDLE;

    uint32_t num_worlds_ = 0;

    uint32_t per_width_ = 0;
    uint32_t per_height_ = 0;

    // Constructor - requires device and backend references
    RenderContext(Device &dev, Backend &be) 
        : device(dev), backend(be), allocator(dev,backend) {}
};

// Initialize RenderContext with basic Vulkan resources for batch rendering
bool initRenderContext(
    RenderContext &ctx,
    Device &device,
    Backend &backend,
    uint32_t num_worlds,
    uint32_t per_width,
    uint32_t per_height);

// Cleanup RenderContext resources
void cleanupRenderContext(RenderContext &ctx);

}} // namespace mujoco::mjbatch
