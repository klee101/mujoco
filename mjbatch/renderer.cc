// renderer.cc
// Vulkan Batch Renderer Implementation

#include "renderer.h"
#include "vkutils.h"
#include "backend.h"
#include "scene.h"
#include <chrono>
#include <fstream>
#include <cstring>
#include <cassert>
#include <algorithm>

using namespace mujoco::mjbatch;

namespace {

inline void DefaultLog(const std::string &s) {
    fprintf(stderr, "[BatchRenderer] %s\n", s.c_str());
}

} // namespace

// --------------------------- Helpers ----------------------------------------
#define LOG(cfg, msg) do { \
    if ((cfg).log_callback) (cfg).log_callback(msg); \
    else DefaultLog(msg); \
} while(0)

static RenderResult MakeError(RenderError e, const std::string &msg, int idx = -1) {
    RenderResult r;
    r.error = e;
    r.message = msg;
    r.failed_batch_idx = idx;
    return r;
}

// 轻量工具：FNV-1a 哈希 + 字节预览 + 非零计数
static uint32_t Hash32(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

static size_t CountNonZero(const void* data, size_t n) {
    const uint8_t* p = (const uint8_t*)data;
    size_t c = 0;
    for (size_t i = 0; i < n; ++i) c += (p[i] != 0);
    return c;
}

static std::string HexPreview(const void* data, size_t n, size_t maxbytes = 16) {
    const uint8_t* p = (const uint8_t*)data;
    std::ostringstream oss;
    size_t m = std::min(n, maxbytes);
    for (size_t i = 0; i < m; ++i) {
        oss << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
            << (int)p[i] << (i + 1 < m ? " " : "");
    }
    return oss.str();
}

// --------------------------- Factory / ctor ---------------------------------
std::unique_ptr<BatchRenderer> BatchRenderer::Create(
    const mjModel* m,
    const BatchRendererConfig& config)
{
    if (!m) {
        return nullptr;
    }

    auto renderer = std::unique_ptr<BatchRenderer>(new BatchRenderer(m, config));
    if (!renderer->Initialize()) {
        renderer->Cleanup();
        return nullptr;
    }
    return renderer;
}

BatchRenderer::BatchRenderer(const mjModel* m, const BatchRendererConfig& config)
    : model_(m), config_(config)
{
    // Reserve environment resources vector
    env_resources_.resize(config_.batch_size);
    // Preallocate output buffers on CPU
    rgb_buffer_.resize((size_t)config_.batch_size * config_.frame_width * config_.frame_height * 3);
    if (config_.enable_depth) {
        depth_buffer_.resize((size_t)config_.batch_size * config_.frame_width * config_.frame_height);
    }
}

BatchRenderer::~BatchRenderer() {
    Cleanup();
}

BatchRenderer::BatchRenderer(BatchRenderer&& other) noexcept {
    *this = std::move(other);
}

BatchRenderer& BatchRenderer::operator=(BatchRenderer&& other) noexcept {
    if (this != &other) {
        Cleanup();
        model_ = other.model_;
        config_ = other.config_;
        env_resources_ = std::move(other.env_resources_);
        backend_ = std::move(other.backend_);
        device_ = std::move(other.device_);
        render_context_ = std::move(other.render_context_);
        graphics_pipeline_ = other.graphics_pipeline_;
        pipeline_layout_ = other.pipeline_layout_;
        descriptor_set_layout_ = other.descriptor_set_layout_;
        descriptor_pool_ = other.descriptor_pool_;
        descriptor_sets_ = std::move(other.descriptor_sets_);
        vert_shader_module_ = other.vert_shader_module_;
        frag_shader_module_ = other.frag_shader_module_;
        framebuffers_ = std::move(other.framebuffers_);
        color_images_ = std::move(other.color_images_);
        depth_images_ = std::move(other.depth_images_);
        color_image_views_ = std::move(other.color_image_views_);
        depth_image_views_ = std::move(other.depth_image_views_);
        command_pool_ = other.command_pool_;
        command_buffers_ = std::move(other.command_buffers_);
        render_fences_ = std::move(other.render_fences_);
        staging_buffers_ = std::move(other.staging_buffers_);
        vertex_buffers_ = std::move(other.vertex_buffers_);
        index_buffers_ = std::move(other.index_buffers_);
        vertex_counts_ = std::move(other.vertex_counts_);
        index_counts_ = std::move(other.index_counts_);
        vertex_buffer_sizes_ = std::move(other.vertex_buffer_sizes_);
        index_buffer_sizes_ = std::move(other.index_buffer_sizes_);
        rgb_buffer_ = std::move(other.rgb_buffer_);
        depth_buffer_ = std::move(other.depth_buffer_);
        last_stats_ = other.last_stats_;
        initialized_ = other.initialized_;
        frame_counter_ = other.frame_counter_;

        // Invalidate other's handles
        other.graphics_pipeline_ = VK_NULL_HANDLE;
        other.pipeline_layout_ = VK_NULL_HANDLE;
        other.descriptor_set_layout_ = VK_NULL_HANDLE;
        other.descriptor_pool_ = VK_NULL_HANDLE;
        other.vert_shader_module_ = VK_NULL_HANDLE;
        other.frag_shader_module_ = VK_NULL_HANDLE;
        other.command_pool_ = VK_NULL_HANDLE;
        other.initialized_ = false;
    }
    return *this;
}

// --------------------------- Shader Loading ---------------------------------
std::vector<uint32_t> BatchRenderer::readSPIRV(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        FATAL("Failed to open shader file: %s", filename.c_str());
    }

    size_t fileSize = (size_t)file.tellg();
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read((char*)buffer.data(), fileSize);
    file.close();

    return buffer;
}

VkShaderModule BatchRenderer::loadShaderModule(const std::string& shader_path) {
    auto code = readSPIRV(shader_path);
    
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size() * sizeof(uint32_t);
    createInfo.pCode = code.data();

    VkShaderModule shaderModule;
    REQ_VK(device_->dt.createShaderModule(device_->hdl, &createInfo, nullptr, &shaderModule));
    
    return shaderModule;
}

// --------------------------- Initialize / Cleanup ----------------------------
bool BatchRenderer::Initialize() {
    LOG(config_, "Initialize(): starting");

    if (!model_) {
        LOG(config_, "Initialize(): model_ is null");
        return false;
    }

    // Phase 1: Create Backend -> Device -> RenderContext
    if (!CreateVulkanInstance()) {
        LOG(config_, "Initialize(): CreateVulkanInstance failed");
        return false;
    }
    if (!SelectPhysicalDevice()) {
        LOG(config_, "Initialize(): SelectPhysicalDevice failed");
        return false;
    }
    if (!CreateLogicalDevice()) {
        LOG(config_, "Initialize(): CreateLogicalDevice failed");
        return false;
    }

    // Create RenderContext
    render_context_ = std::make_unique<RenderContext>(*device_, *backend_);
    if (!initRenderContext(*render_context_,
                        *device_,
                        *backend_,
                        config_.batch_size,
                        config_.frame_width,
                        config_.frame_height)) {
        LOG(config_, "Initialize(): initRenderContext failed");
        return false;
    }

    // Create pipeline and resources
    if (!CreatePipeline()) {
        LOG(config_, "Initialize(): CreatePipeline failed");
        return false;
    }
    if (!CreateFramebuffers()) {
        LOG(config_, "Initialize(): CreateFramebuffers failed");
        return false;
    }
    if (!CreateBuffers()) {
        LOG(config_, "Initialize(): CreateBuffers failed");
        return false;
    }

    // Initialize per-environment MuJoCo visualization structs
    for (int i = 0; i < config_.batch_size; ++i) {
        PerEnvResources &res = env_resources_[i];
        mjv_defaultCamera(&res.camera);
        mjv_defaultOption(&res.options);
        mjv_defaultScene(&res.scene);
        mjv_makeScene(model_, &res.scene, 2000);

        mjv_defaultFreeCamera(model_, &res.camera);
        // Scene will be created in UpdateScenes when we have actual data
    }

    initialized_ = true;
    LOG(config_, "Initialize(): success");
    return true;
}

void BatchRenderer::Cleanup() {
    if (!initialized_) {
        return;
    }

    Device &dev = *device_;

    // Wait for device to be idle
    dev.dt.deviceWaitIdle(dev.hdl);

    // Free MuJoCo scenes
    for (auto &res : env_resources_) {
        mjv_freeScene(&res.scene);
    }
    env_resources_.clear();

    // Destroy Vulkan resources
    DestroyVulkanResources();

    // Cleanup RenderContext
    if (render_context_) {
        cleanupRenderContext(*render_context_);
        render_context_.reset();
    }

    // Cleanup device and backend
    device_.reset();
    backend_.reset();

    rgb_buffer_.clear();
    depth_buffer_.clear();

    initialized_ = false;
    LOG(config_, "Cleanup(): done");
}

// --------------------------- Public Render API -------------------------------
RenderResult BatchRenderer::Render(mjData** data_array, const int* camera_ids) {
    if (!initialized_) return MakeError(RenderError::INVALID_CONFIG, "Renderer not initialized");
    if (!data_array) return MakeError(RenderError::INVALID_DATA, "data_array is null");

    const int count = config_.batch_size;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (!UpdateScenes(data_array, count)) {
        return MakeError(RenderError::MUJOCO_ERROR, "UpdateScenes failed");
    }

    if (!RecordCommandBuffers(count)) {
        return MakeError(RenderError::VULKAN_ERROR, "RecordCommandBuffers failed");
    }

    if (!SubmitAndWait()) {
        return MakeError(RenderError::VULKAN_ERROR, "SubmitAndWait failed");
    }

    if (!ReadbackResults()) {
        return MakeError(RenderError::VULKAN_ERROR, "ReadbackResults failed");
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    last_stats_.cpu_time_ms = (float)ms;
    last_stats_.frame_number = ++frame_counter_;

    return RenderResult(); // success
}

// --------------------------- Internal Stages ---------------------------------
// TODO: solve the Material issues, upload as a decriptor set per-scene
bool BatchRenderer::UpdateScenes(mjData** data_array, int count) {
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;
    
    // 用于批量上传的staging buffers和拷贝信息
    struct CopyInfo {
        VkBuffer src;
        VkBuffer dst;
        VkDeviceSize size;
    };
    std::vector<HostBuffer> staging_buffers;
    std::vector<CopyInfo> copy_ops;
    staging_buffers.reserve(count); // Only camera updates now
    copy_ops.reserve(count);
    
    // Update MuJoCo scenes and prepare buffers
    for (int i = 0; i < count; ++i) {
        if (!data_array[i]) {
            LOG(config_, "UpdateScenes(): null mjData in array");
            return false;
        }
        
        PerEnvResources &res = env_resources_[i];
        
        // Update MuJoCo visualization scene
        mjv_updateScene(model_, data_array[i], &res.options, nullptr, 
                    &res.camera, mjCAT_ALL, &res.scene);
        
        // First time: create scene with geometry
        if (!res.render_scene) {
            res.render_scene = std::make_unique<mujoco::mjbatch::Scene>(model_, &res.scene);
            
            // First-time geometry buffer creation
            std::vector<mujoco::mjbatch::Vertex> vertices;
            std::vector<uint32_t> indices;
            res.render_scene->GetCombinedBuffers(vertices, indices);
            
            vertex_counts_[i] = vertices.size();
            index_counts_[i] = indices.size();
            
            // Allocate and upload vertex buffer (one-time)
            size_t vertex_buffer_size = vertices.size() * sizeof(mujoco::mjbatch::Vertex);
            if (vertex_buffer_size > 0) {
                auto vertex_buf = allocator.makeLocalBuffer(
                    vertex_buffer_size, 
                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                vertex_buffers_[i] = std::move(*vertex_buf);
                vertex_buffer_sizes_[i] = vertex_buffer_size;
                
                auto staging = allocator.makeStagingBuffer(vertex_buffer_size);
                std::memcpy(staging.ptr, vertices.data(), vertex_buffer_size);
                staging.flush(dev);
                copy_ops.push_back({staging.buffer, vertex_buffers_[i].buffer, vertex_buffer_size});
                staging_buffers.push_back(std::move(staging));
            }
            
            // Allocate and upload index buffer (one-time)
            size_t index_buffer_size = indices.size() * sizeof(uint32_t);
            if (index_buffer_size > 0) {
                auto index_buf = allocator.makeLocalBuffer(
                    index_buffer_size, 
                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                index_buffers_[i] = std::move(*index_buf);
                index_buffer_sizes_[i] = index_buffer_size;
                
                auto staging = allocator.makeStagingBuffer(index_buffer_size);
                std::memcpy(staging.ptr, indices.data(), index_buffer_size);
                staging.flush(dev);
                copy_ops.push_back({staging.buffer, index_buffers_[i].buffer, index_buffer_size});
                staging_buffers.push_back(std::move(staging));
            }
        } else {
            // Subsequent updates: only update transforms in Scene
             res.render_scene->Update(model_, &res.scene);
         }
        
        // Update camera uniform buffer
        const auto& camera_info = res.render_scene->GetCamera();
        CameraUBO camera_ubo{};
        camera_ubo.view_proj = camera_info.view_proj_matrix;
        camera_ubo.position = camera_info.position;
        camera_ubo.forward = camera_info.forward;
        camera_ubo.up = camera_info.up;
        camera_ubo.near_plane = camera_info.near_plane;
        camera_ubo.far_plane = camera_info.far_plane;
        camera_ubo.fov = camera_info.fov;
        
        std::memcpy(camera_staging_buffers_[i].ptr, &camera_ubo, sizeof(CameraUBO));
        camera_staging_buffers_[i].flush(dev);
        copy_ops.push_back({camera_staging_buffers_[i].buffer, 
                           camera_uniform_buffers_[i].buffer, 
                           sizeof(CameraUBO)});
     
    }
    
    // Batch upload all copy operations
    if (!copy_ops.empty()) {
        VkCommandBuffer cmd = render_context_->load_cmd_;
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
        
        for (const auto& copy : copy_ops) {
            VkBufferCopy copyRegion{};
            copyRegion.size = copy.size;
            dev.dt.cmdCopyBuffer(cmd, copy.src, copy.dst, 1, &copyRegion);
        }
        
        REQ_VK(dev.dt.endCommandBuffer(cmd));
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, 
                                &submitInfo, render_context_->load_fence_));
        
        waitForFenceInfinitely(dev, render_context_->load_fence_);
        resetFence(dev, render_context_->load_fence_);
    }

    
    return true;
}

bool BatchRenderer::RecordCommandBuffers(int count) {
    Device &dev = *device_;
    
    for (int i = 0; i < count; ++i) {
        VkCommandBuffer cmd = command_buffers_[i];
        
        // Begin recording (beginCommandBuffer implicitly resets if needed)
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
        
        // Begin render pass
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = render_context_->renderPass;
        renderPassInfo.framebuffer = framebuffers_[i];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height};
        
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();
        
        dev.dt.cmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        // Bind pipeline
        dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
        
        // Bind descriptor set (camera + material)
        dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline_layout_, 0, 1, &descriptor_sets_[i], 0, nullptr);
  
        // Set viewport and scissor
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)config_.frame_width;
        viewport.height = (float)config_.frame_height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        dev.dt.cmdSetViewport(cmd, 0, 1, &viewport);
        
        VkRect2D scissor{};
        scissor.offset = {0, 0};
        scissor.extent = {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height};
        dev.dt.cmdSetScissor(cmd, 0, 1, &scissor);
        
        // Bind vertex and index buffers
        if (i < vertex_buffers_.size() && i < index_buffers_.size() &&
            vertex_buffers_[i].buffer != VK_NULL_HANDLE && 
            index_buffers_[i].buffer != VK_NULL_HANDLE &&
            index_counts_[i] > 0) {
            VkBuffer vertexBuffer = vertex_buffers_[i].buffer;
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, offsets);
            
            vkCmdBindIndexBuffer(cmd, index_buffers_[i].buffer, 0, VK_INDEX_TYPE_UINT32);
            
            // Draw each drawable with its transform as push constant
            const auto& drawables = env_resources_[i].render_scene->GetDrawables();
            uint32_t index_offset = 0;
            int32_t vertex_offset = 0;
            
            for (const auto& drawable : drawables) {
                if (!drawable.visible) {
                    index_offset += drawable.geometry.GetIndexCount();
                    vertex_offset += drawable.geometry.GetVertexCount();
                    continue;
                };
                PushConstants pushConstants{};

                pushConstants.model = drawable.transform;
                pushConstants.rgba = drawable.material.rgba;
                pushConstants.specular = drawable.material.specular;
                pushConstants.emission = drawable.material.emission;
                pushConstants.shininess = drawable.material.shininess;
                pushConstants.texture_id = drawable.material.texture_id;
                
                dev.dt.cmdPushConstants(cmd, pipeline_layout_, 
                                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                      0, sizeof(PushConstants), &pushConstants);
                // Draw this drawable
                uint32_t index_count = drawable.geometry.GetIndexCount();
                vkCmdDrawIndexed(cmd, index_count, 1, index_offset, vertex_offset, 0);
                
                index_offset += index_count;
                vertex_offset += drawable.geometry.GetVertexCount();
            }  
        }
        
        // End render pass
        dev.dt.cmdEndRenderPass(cmd);
        
        // End recording
        REQ_VK(dev.dt.endCommandBuffer(cmd));

        {
            char b[256];
            snprintf(b, sizeof(b),
                "[RecordCommandBuffers] env=%d vtx=%zu idx=%zu fb=%p",
                i, vertex_counts_[i], index_counts_[i], (void*)framebuffers_[i]);
            LOG(config_, b);
        }
    }
    
    return true;
}


bool BatchRenderer::SubmitAndWait() {
    Device &dev = *device_;
    VkQueue queue = render_context_->renderQueue;
    
    const uint64_t TIMEOUT_NS = 5000000000ULL; // 5秒超时
    const uint64_t QUICK_CHECK_NS = 0;         // 0秒超时用于快速检测
    char log_buf[256];
    
    for (size_t i = 0; i < command_buffers_.size(); ++i) {
        snprintf(log_buf, sizeof(log_buf), "Processing fence %zu...", i);
        LOG(config_, log_buf);
        
        // 使用0超时快速检查fence是否已signaled
        VkResult quickCheck = dev.dt.waitForFences(
            dev.hdl,
            1,
            &render_fences_[i],
            VK_TRUE,
            QUICK_CHECK_NS
        );
        
        if (quickCheck == VK_SUCCESS) {
            // Fence已signaled，正常重置
            snprintf(log_buf, sizeof(log_buf), "  Fence %zu already signaled, resetting", i);
            LOG(config_, log_buf);
            resetFence(dev, render_fences_[i]);
        } else if (quickCheck == VK_TIMEOUT) {
            // Fence未signaled（首次使用或有问题）
            snprintf(log_buf, sizeof(log_buf), "  Fence %zu not signaled (first use), resetting", i);
            LOG(config_, log_buf);
            resetFence(dev, render_fences_[i]);
        } else {
            // 错误
            snprintf(log_buf, sizeof(log_buf), "ERROR: Fence %zu check failed: %d", i, quickCheck);
            LOG(config_, log_buf);
            return false;
        }
        
        // 提交命令
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &command_buffers_[i];
        
        VkResult submitResult = dev.dt.queueSubmit(queue, 1, &submitInfo, render_fences_[i]);
        if (submitResult != VK_SUCCESS) {
            snprintf(log_buf, sizeof(log_buf), "ERROR: queueSubmit %zu failed: %d", i, submitResult);
            LOG(config_, log_buf);
            return false;
        }
        
        snprintf(log_buf, sizeof(log_buf), "  Fence %zu submitted", i);
        LOG(config_, log_buf);
    }
    
    LOG(config_, "All submissions complete, waiting for completion...");
    
    // 等待所有fence完成
    for (size_t i = 0; i < render_fences_.size(); ++i) {
        VkResult result = dev.dt.waitForFences(
            dev.hdl,
            1,
            &render_fences_[i],
            VK_TRUE,
            TIMEOUT_NS
        );
        
        if (result == VK_TIMEOUT) {
            snprintf(log_buf, sizeof(log_buf), "ERROR: Final fence %zu timeout! GPU may have hung.", i);
            LOG(config_, log_buf);
            return false;
        } else if (result != VK_SUCCESS) {
            snprintf(log_buf, sizeof(log_buf), "ERROR: Final fence %zu wait failed: %d", i, result);
            LOG(config_, log_buf);
            return false;
        }
    }
    
    LOG(config_, "All fences completed successfully");
    return true;
}

bool BatchRenderer::ReadbackResults() {
    Device &dev = *device_;
    
    VkCommandBuffer cmd = render_context_->load_cmd_;
    
    for (int i = 0; i < config_.batch_size; ++i) {
        // Transition color image to transfer src
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = color_images_[i].image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
        
        dev.dt.cmdPipelineBarrier(cmd,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        // Copy image to staging buffer
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height, 1};
        
        dev.dt.cmdCopyImageToBuffer(cmd, color_images_[i].image, 
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    staging_buffers_[i].buffer, 1, &region);
        
        // Transition back
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dev.dt.cmdPipelineBarrier(cmd,
                                VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                0, 0, nullptr, 0, nullptr, 1, &barrier);
        
        REQ_VK(dev.dt.endCommandBuffer(cmd));
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, render_context_->load_fence_));
        waitForFenceInfinitely(dev, render_context_->load_fence_);
        resetFence(dev, render_context_->load_fence_);
        
        // Invalidate staging buffer to ensure CPU can read it
        staging_buffers_[i].invalidate(dev);

        {
            size_t bytes = (size_t)config_.frame_width * config_.frame_height * 4;
            uint32_t h = Hash32(staging_buffers_[i].ptr, bytes);
            size_t nz = CountNonZero(staging_buffers_[i].ptr, std::min(bytes, (size_t)4096));
            std::string pv = HexPreview(staging_buffers_[i].ptr, bytes);
            char b[256];
            snprintf(b, sizeof(b),
                "[ReadbackResults] env=%d RGBA bytes=%zu hash=0x%08X nz(first4K)=%zu preview=[%s]",
                i, bytes, h, nz, pv.c_str());
            LOG(config_, b);
        }
        
        // Copy from staging buffer to output
        size_t frame_size = (size_t)config_.frame_width * config_.frame_height;
        size_t rgb_offset = (size_t)i * frame_size * 3;
        
        // Convert RGBA to RGB
        const unsigned char* src = static_cast<const unsigned char*>(staging_buffers_[i].ptr);
        unsigned char* dst = rgb_buffer_.data() + rgb_offset;
        for (size_t j = 0; j < frame_size; ++j) {
            dst[j * 3 + 0] = src[j * 4 + 0];
            dst[j * 3 + 1] = src[j * 4 + 1];
            dst[j * 3 + 2] = src[j * 4 + 2];
        }

        // Debug log
        {
            size_t frame_size = (size_t)config_.frame_width * config_.frame_height;
            size_t rgb_offset = (size_t)i * frame_size * 3;
            const unsigned char* dst = rgb_buffer_.data() + rgb_offset;
            uint32_t h = Hash32(dst, frame_size * 3);
            size_t nz = CountNonZero(dst, std::min(frame_size * 3, (size_t)4096));
            std::string pv = HexPreview(dst, frame_size * 3);
            char b[256];
            snprintf(b, sizeof(b),
                "[ReadbackResults] env=%d RGB bytes=%zu hash=0x%08X nz(first4K)=%zu preview=[%s]",
                i, frame_size * 3, h, nz, pv.c_str());
            LOG(config_, b);
        }
        
        // TODO: Readback depth if enabled
        if (config_.enable_depth) {
            // Similar process for depth image
            // For now, fill with placeholder
            size_t depth_offset = (size_t)i * frame_size;
            std::fill(depth_buffer_.begin() + depth_offset, 
                    depth_buffer_.begin() + depth_offset + frame_size, 1.0f);
        }
    }
    
    return true;
}

// --------------------------- Vulkan Resource Creation -----------------------
bool BatchRenderer::CreateVulkanInstance() {
    // Create backend (which creates the instance)
    // Load Vulkan loader library
    LoaderLib *loader = LoaderLib::load();
    if (!loader) {
        LOG(config_, "CreateVulkanInstance(): failed to load Vulkan loader");
        return false;
    }
    
    backend_ = std::make_unique<Backend>(
        loader->getEntryFn(),
        config_.enable_validation,
        false  // no present surface needed for offscreen rendering
    );
    
    LOG(config_, "CreateVulkanInstance(): instance created");
    return true;
}

bool BatchRenderer::SelectPhysicalDevice() {
    // Device selection is done in CreateLogicalDevice
    LOG(config_, "SelectPhysicalDevice(): will select in CreateLogicalDevice");
    return true;
}

bool BatchRenderer::CreateLogicalDevice() {
    // Create device using backend
    device_ = std::unique_ptr<Device>(backend_->makeDevice(config_.gpu_id, {}));
    
    if (!device_) {
        LOG(config_, "CreateLogicalDevice(): failed to create device");
        return false;
    }
    
    LOG(config_, "CreateLogicalDevice(): device created");
    return true;
}

bool BatchRenderer::CreateRenderPass() {
    // Render pass is created in RenderContext, so this is a no-op
    LOG(config_, "CreateRenderPass(): using RenderContext render pass");
    return true;
}

bool BatchRenderer::CreatePipeline() {
    Device &dev = *device_;
    
    // Load shaders (assuming they're compiled to .spv files)
    // In a real implementation, you'd get the shader path from build system
    std::string shader_dir = "shaders_spv/";  // This should come from build config
    vert_shader_module_ = loadShaderModule(shader_dir + "mujoco_vs.spv");
    frag_shader_module_ = loadShaderModule(shader_dir + "mujoco_ps.spv");
    
    // Create shader stages
    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vert_shader_module_;
    vertShaderStageInfo.pName = "VSMain";
    
    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = frag_shader_module_;
    fragShaderStageInfo.pName = "PSMain";
    
    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
    // Vertex input
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(float) * (3 + 3 + 2 + 4); // pos + normal + texcoord + color
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    
    std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = 0;
    
    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = sizeof(float) * 3;
    
    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = sizeof(float) * 6;
    
    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[3].offset = sizeof(float) * 8;
    
    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount = attributeDescriptions.size();
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();
    
    // Input assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;
    
    // Viewport and scissor (dynamic)
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    
    // Rasterization
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    
    // Multisampling
    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    
    // Depth stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;
    
    // Color blending
    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | 
                                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;
    
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    
    // Dynamic state
    std::vector<VkDynamicState> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();

    // Descriptor set layout (BUGFIX!!! now this part not work!!!)
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0; // Camera UBO
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    bindings[1].binding = 1; // Material UBO
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = 2;
    dslInfo.pBindings = bindings;
    REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &dslInfo, nullptr, &descriptor_set_layout_));
    //-------------------------------------------------------------------------//
    
    // Pipeline layout with push constants for per-drawable transform
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float) * 16; // view-projection matrix (mat4)
    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptor_set_layout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &pipelineLayoutInfo, nullptr, &pipeline_layout_));
    
    // Create graphics pipeline
    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipeline_layout_;
    pipelineInfo.renderPass = render_context_->renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    
    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphics_pipeline_));
    
    LOG(config_, "CreatePipeline(): pipeline created");
    return true;
}

bool BatchRenderer::CreateFramebuffers() {
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;
    
    framebuffers_.reserve(config_.batch_size);
    color_images_.reserve(config_.batch_size);
    depth_images_.reserve(config_.batch_size);
    color_image_views_.reserve(config_.batch_size);
    depth_image_views_.reserve(config_.batch_size);
    
    for (int i = 0; i < config_.batch_size; ++i) {
        // Create color image
        auto color_img = allocator.makeColorAttachment(
            config_.frame_width, config_.frame_height, 1, VK_FORMAT_R8G8B8A8_UNORM);
        color_images_.push_back(std::move(color_img));
        
        // Create depth image
        if (config_.enable_depth) {
            auto depth_img = allocator.makeDepthAttachment(
                config_.frame_width, config_.frame_height, 1, VK_FORMAT_D32_SFLOAT);
            depth_images_.push_back(std::move(depth_img));
        }
        
        // Create image views
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = color_images_[i].image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;
        VkImageView colorView;
        REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfo, nullptr, &colorView));
        color_image_views_.push_back(colorView);
        
        VkImageView depthView = VK_NULL_HANDLE;
        if (config_.enable_depth) {
            viewInfo.image = depth_images_[i].image;
            viewInfo.format = VK_FORMAT_D32_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfo, nullptr, &depthView));
        }
        depth_image_views_.push_back(depthView);
        
        // Create framebuffer
        std::vector<VkImageView> attachments = {color_image_views_[i]};
        if (config_.enable_depth && depth_image_views_[i] != VK_NULL_HANDLE) {
            attachments.push_back(depth_image_views_[i]);
        }
        
        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = render_context_->renderPass;
        framebufferInfo.attachmentCount = attachments.size();
        framebufferInfo.pAttachments = attachments.data();
        framebufferInfo.width = config_.frame_width;
        framebufferInfo.height = config_.frame_height;
        framebufferInfo.layers = 1;
        
        REQ_VK(dev.dt.createFramebuffer(dev.hdl, &framebufferInfo, nullptr, &framebuffers_[i]));
    }
    
    LOG(config_, "CreateFramebuffers(): framebuffers created");
    return true;
}


// TODO: Now the Material UBO actually invalid, use push constants instead
bool BatchRenderer::CreateBuffers() {
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;
    
    // Create command pool
    command_pool_ = makeCmdPool(dev, dev.gfxQF);
    
    // Allocate command buffers
    command_buffers_.resize(config_.batch_size);
    VkCommandBufferAllocateInfo allocCmdInfo{};
    allocCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCmdInfo.commandPool = command_pool_;
    allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmdInfo.commandBufferCount = command_buffers_.size();
    REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &allocCmdInfo, command_buffers_.data()));
    
    // Create fences
    render_fences_.resize(config_.batch_size);
    for (auto &fence : render_fences_) {
        fence = makeFence(dev, false);
    }

    // Create descriptor pool
    VkDescriptorPoolSize poolSizes[2];
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = config_.batch_size; // Camera UBOs
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = config_.batch_size; // Material UBOs
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    poolInfo.maxSets = config_.batch_size;
    REQ_VK(dev.dt.createDescriptorPool(dev.hdl, &poolInfo, nullptr, &descriptor_pool_));
    
    // Allocate descriptor sets
    descriptor_sets_.resize(config_.batch_size);
    std::vector<VkDescriptorSetLayout> layouts(config_.batch_size, descriptor_set_layout_);
    VkDescriptorSetAllocateInfo allocDesInfo{};
    allocDesInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocDesInfo.descriptorPool = descriptor_pool_;
    allocDesInfo.descriptorSetCount = config_.batch_size;
    allocDesInfo.pSetLayouts = layouts.data();
    REQ_VK(dev.dt.allocateDescriptorSets(dev.hdl, &allocDesInfo, descriptor_sets_.data()));
    
    // Create uniform buffers
    camera_uniform_buffers_.clear();
    material_uniform_buffers_.clear();
    camera_staging_buffers_.clear();
    
    for (int i = 0; i < config_.batch_size; ++i) {
        // Camera UBO
        auto camera_ubo = allocator.makeLocalBuffer(
            sizeof(CameraUBO), 
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        camera_uniform_buffers_.emplace_back(std::move(*camera_ubo));
        
        // Camera staging buffer for updates
        camera_staging_buffers_.emplace_back(
            allocator.makeStagingBuffer(sizeof(CameraUBO)));
        
        // Material UBO
        auto material_ubo = allocator.makeLocalBuffer(
            sizeof(MaterialUBO), 
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        material_uniform_buffers_.emplace_back(std::move(*material_ubo));
        
        // Update descriptor sets
        VkDescriptorBufferInfo cameraBufferInfo{};
        cameraBufferInfo.buffer = camera_uniform_buffers_[i].buffer;
        cameraBufferInfo.offset = 0;
        cameraBufferInfo.range = sizeof(CameraUBO);
        
        VkDescriptorBufferInfo materialBufferInfo{};
        materialBufferInfo.buffer = material_uniform_buffers_[i].buffer;
        materialBufferInfo.offset = 0;
        materialBufferInfo.range = sizeof(MaterialUBO);
        
        VkWriteDescriptorSet writes[2]{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptor_sets_[i];
        writes[0].dstBinding = 0;
        writes[0].dstArrayElement = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &cameraBufferInfo;
        
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = descriptor_sets_[i];
        writes[1].dstBinding = 1;
        writes[1].dstArrayElement = 0;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &materialBufferInfo;
        
        dev.dt.updateDescriptorSets(dev.hdl, 2, writes, 0, nullptr);
    }


    // Initialize vertex/index buffer vectors - pre-allocate with dummy buffers
    vertex_buffers_.clear();
    index_buffers_.clear();
    for (int i = 0; i < config_.batch_size; ++i) {
        // Create minimal dummy buffers that will be replaced in UpdateScenes
        auto vertex_buffer = allocator.makeLocalBuffer(1, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        if (!vertex_buffer) {
            FATAL("Failed to allocate dummy vertex buffer");
        }
        vertex_buffers_.emplace_back(std::move(*vertex_buffer));
        
        // index buffer
        auto index_buffer = allocator.makeLocalBuffer(1, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        if (!index_buffer) {
            FATAL("Failed to allocate dummy index buffer");
        }
        index_buffers_.emplace_back(std::move(*index_buffer));
    }
    vertex_counts_.resize(config_.batch_size, 0);
    index_counts_.resize(config_.batch_size, 0);
    vertex_buffer_sizes_.resize(config_.batch_size, 0);
    index_buffer_sizes_.resize(config_.batch_size, 0);
    
    // Create staging buffers for readback
    staging_buffers_.clear();
    staging_buffers_.reserve(config_.batch_size);
    for (size_t i = 0; i < config_.batch_size; ++i) {
        size_t bufferSize = config_.frame_width * config_.frame_height * 4;
        staging_buffers_.emplace_back(
            render_context_->allocator.makeStagingBuffer(bufferSize)
        );
    }
    
    LOG(config_, "CreateBuffers(): buffers created");
    return true;
}

void BatchRenderer::DestroyVulkanResources() {
    if (!device_) return;
    
    Device &dev = *device_;
    dev.dt.deviceWaitIdle(dev.hdl);
    
    // Destroy shader modules
    if (vert_shader_module_ != VK_NULL_HANDLE) {
        dev.dt.destroyShaderModule(dev.hdl, vert_shader_module_, nullptr);
        vert_shader_module_ = VK_NULL_HANDLE;
    }
    if (frag_shader_module_ != VK_NULL_HANDLE) {
        dev.dt.destroyShaderModule(dev.hdl, frag_shader_module_, nullptr);
        frag_shader_module_ = VK_NULL_HANDLE;
    }
    
    // Destroy pipeline
    if (graphics_pipeline_ != VK_NULL_HANDLE) {
        dev.dt.destroyPipeline(dev.hdl, graphics_pipeline_, nullptr);
        graphics_pipeline_ = VK_NULL_HANDLE;
    }
    
    // Destroy pipeline layout
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        dev.dt.destroyPipelineLayout(dev.hdl, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    
    // Destroy framebuffers and image views
    for (auto &fb : framebuffers_) {
        if (fb != VK_NULL_HANDLE) {
            dev.dt.destroyFramebuffer(dev.hdl, fb, nullptr);
        }
    }
    framebuffers_.clear();
    
    for (auto &view : color_image_views_) {
        if (view != VK_NULL_HANDLE) {
            dev.dt.destroyImageView(dev.hdl, view, nullptr);
        }
    }
    color_image_views_.clear();
    
    for (auto &view : depth_image_views_) {
        if (view != VK_NULL_HANDLE) {
            dev.dt.destroyImageView(dev.hdl, view, nullptr);
        }
    }
    depth_image_views_.clear();
    
    // Images are destroyed via LocalImage destructors
    color_images_.clear();
    depth_images_.clear();
    
    // Destroy command pool (command buffers are freed automatically)
    if (command_pool_ != VK_NULL_HANDLE) {
        dev.dt.destroyCommandPool(dev.hdl, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    
    // Destroy fences
    for (auto &fence : render_fences_) {
        if (fence != VK_NULL_HANDLE) {
            dev.dt.destroyFence(dev.hdl, fence, nullptr);
        }
    }
    render_fences_.clear();
    
    // Staging buffers are destroyed via HostBuffer destructors
    staging_buffers_.clear();
    
    LOG(config_, "DestroyVulkanResources(): resources released");
}

// --------------------------- Simple accessors --------------------------------
const unsigned char* BatchRenderer::GetRGBFrame(int batch_idx) const {
    if (batch_idx < 0 || batch_idx >= config_.batch_size) return nullptr;
    size_t stride = (size_t)config_.frame_width * config_.frame_height * 3;
    return rgb_buffer_.data() + (size_t)batch_idx * stride;
}

const float* BatchRenderer::GetDepthFrame(int batch_idx) const {
    if (!config_.enable_depth) return nullptr;
    if (batch_idx < 0 || batch_idx >= config_.batch_size) return nullptr;
    size_t stride = (size_t)config_.frame_width * config_.frame_height;
    return depth_buffer_.data() + (size_t)batch_idx * stride;
}

// --------------------------- Utility functions --------------------------------
std::vector<std::string> EnumerateGPUs() {
    // TODO: query Vulkan enumerations and return device names
    return { "VulkanDevice0 (placeholder)" };
}

bool IsGPUSupported(int gpu_id) {
    // TODO: check whether gpu_id exists and supports necessary features
    (void)gpu_id;
    return true;
}

