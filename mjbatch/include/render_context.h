#pragma once 

#include "device.h"
#include "memory.h"
#include "backend.h"
#include <vulkan/vulkan.h>
#include <mutex>
#include <condition_variable>

namespace mujoco{
namespace mjbatch {

struct RenderContext
{
    Device &device;
    Backend &backend;
    MemoryAllocator allocator;

    VkQueue renderQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;

    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkRenderPass shadowPass = VK_NULL_HANDLE;

    VkCommandPool load_cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer load_cmd_ = VK_NULL_HANDLE;
    VkFence load_fence_ = VK_NULL_HANDLE;

    VkCommandPool transfer_cmd_pool_ = VK_NULL_HANDLE;
    uint32_t transferQF = VK_QUEUE_FAMILY_IGNORED;

    VkCommandPool ring_cmd_pool_ = VK_NULL_HANDLE;

    uint32_t num_worlds_ = 0;

    uint32_t per_width_ = 0;
    uint32_t per_height_ = 0;

    RenderContext(Device &dev, Backend &be) 
        : device(dev), backend(be), allocator(dev,backend) {}

    
};

bool initRenderContext(
    RenderContext &ctx,
    Device &device,
    Backend &backend,
    uint32_t num_worlds,
    uint32_t per_width,
    uint32_t per_height);

void cleanupRenderContext(RenderContext &ctx);

}} // namespace mujoco::mjbatch
