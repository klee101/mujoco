#include "render_context.h"
#include "vkutils.h"

#include <cstring>
#include <array>

namespace mujoco {
namespace mjbatch {

// Initialize RenderContext with basic Vulkan resources for batch rendering
bool initRenderContext(
    RenderContext &ctx,
    Device &device,
    Backend &backend,
    uint32_t num_worlds,
    uint32_t per_width,
    uint32_t per_height)
{
    // Note: device and backend are already set via constructor
    // Just set the batch rendering parameters
    ctx.num_worlds_ = num_worlds;
    ctx.per_width_ = per_width;
    ctx.per_height_ = per_height;

    // Get the graphics queue
    Device &dev = ctx.device;
    ctx.renderQueue = makeQueue(dev, dev.gfxQF, 0);

    // Create main render pass (color + depth)
    {
        VkAttachmentDescription color_attachment{};
        color_attachment.format = VK_FORMAT_R8G8B8A8_UNORM;
        color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depth_attachment{};
        depth_attachment.format = VK_FORMAT_D32_SFLOAT;
        depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference color_ref{};
        color_ref.attachment = 0;
        color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 1;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_ref;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription, 2> attachments = {
            color_attachment, depth_attachment
        };

        VkRenderPassCreateInfo pass_info{};
        pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        pass_info.attachmentCount = attachments.size();
        pass_info.pAttachments = attachments.data();
        pass_info.subpassCount = 1;
        pass_info.pSubpasses = &subpass;
        pass_info.dependencyCount = 1;
        pass_info.pDependencies = &dependency;

        REQ_VK(dev.dt.createRenderPass(
            dev.hdl, &pass_info, nullptr, &ctx.renderPass));
    }

    // Create shadow render pass (depth only)
    {
        VkAttachmentDescription depth_attachment{};
        depth_attachment.format = VK_FORMAT_D32_SFLOAT;
        depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 0;
        subpass.pColorAttachments = nullptr;
        subpass.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                   VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;

        VkRenderPassCreateInfo pass_info{};
        pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        pass_info.attachmentCount = 1;
        pass_info.pAttachments = &depth_attachment;
        pass_info.subpassCount = 1;
        pass_info.pSubpasses = &subpass;
        pass_info.dependencyCount = 1;
        pass_info.pDependencies = &dependency;

        REQ_VK(dev.dt.createRenderPass(
            dev.hdl, &pass_info, nullptr, &ctx.shadowPass));
    }

    // Create command pool for loading operations
    ctx.load_cmd_pool_ = makeCmdPool(dev, dev.gfxQF);

    // Allocate command buffer for loading
    {
        VkCommandBufferAllocateInfo alloc_info{};
        alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc_info.commandPool = ctx.load_cmd_pool_;
        alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc_info.commandBufferCount = 1;

        REQ_VK(dev.dt.allocateCommandBuffers(
            dev.hdl, &alloc_info, &ctx.load_cmd_));
    }

    // Create fence for loading operations
    ctx.load_fence_ = makeFence(dev, false);

    return true;
}

// Cleanup RenderContext resources
void cleanupRenderContext(RenderContext &ctx)
{
    Device &dev = ctx.device;

    // Wait for device to be idle before cleanup
    dev.dt.deviceWaitIdle(dev.hdl);

    // Destroy render passes
    if (ctx.renderPass != VK_NULL_HANDLE) {
        dev.dt.destroyRenderPass(dev.hdl, ctx.renderPass, nullptr);
        ctx.renderPass = VK_NULL_HANDLE;
    }

    if (ctx.shadowPass != VK_NULL_HANDLE) {
        dev.dt.destroyRenderPass(dev.hdl, ctx.shadowPass, nullptr);
        ctx.shadowPass = VK_NULL_HANDLE;
    }

    // Free command buffer
    if (ctx.load_cmd_ != VK_NULL_HANDLE && ctx.load_cmd_pool_ != VK_NULL_HANDLE) {
        dev.dt.freeCommandBuffers(
            dev.hdl, ctx.load_cmd_pool_, 1, &ctx.load_cmd_);
        ctx.load_cmd_ = VK_NULL_HANDLE;
    }

    // Destroy command pool
    if (ctx.load_cmd_pool_ != VK_NULL_HANDLE) {
        dev.dt.destroyCommandPool(dev.hdl, ctx.load_cmd_pool_, nullptr);
        ctx.load_cmd_pool_ = VK_NULL_HANDLE;
    }

    // Destroy fence
    if (ctx.load_fence_ != VK_NULL_HANDLE) {
        dev.dt.destroyFence(dev.hdl, ctx.load_fence_, nullptr);
        ctx.load_fence_ = VK_NULL_HANDLE;
    }

    // MemoryAllocator will clean itself up via destructor
}



}} // namespace mujoco::mjbatch

