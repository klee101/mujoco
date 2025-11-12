// -----------------------------------------------------------------------------
// Vulkan Batch Renderer - Simplified Development Header (Stage 1 Ready)
// -----------------------------------------------------------------------------
// Goal: minimal working pipeline for MuJoCo → Vulkan batch rendering
// -----------------------------------------------------------------------------
#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <cstdint>
#include "device.h"


// -----------------------------------------------------------------------------
// Error Handling
// -----------------------------------------------------------------------------
enum class RenderError {
    SUCCESS = 0,
    INVALID_CONFIG,
    INVALID_DATA,
    GPU_DEVICE_NOT_FOUND,
    GPU_OUT_OF_MEMORY,
    DEVICE_LOST,
    SHADER_COMPILATION_FAILED,
    TIMEOUT,
    VULKAN_ERROR,
    MUJOCO_ERROR
};

struct RenderResult {
    VkDevice dev;
    // dev.dt.destroyDescriptorPool(dev.hdl, pool_state.pool, nullptr);
    RenderError error = RenderError::SUCCESS;
    std::string message;
    int failed_batch_idx = -1;
    bool IsSuccess() const { return error == RenderError::SUCCESS; }
    explicit operator bool() const { return IsSuccess(); }
};

// -----------------------------------------------------------------------------
// Configuration (simplified)
// -----------------------------------------------------------------------------
struct BatchRendererConfig {
    int gpu_id = 0;
    int batch_size = 1;
    int frame_width = 640;
    int frame_height = 480;
    bool enable_depth = true;

    // Vulkan setup
    bool enable_validation = true;   // useful for debugging
    int max_frames_in_flight = 1;    // simplified (no multi-frame sync)

    // Logging
    std::function<void(const std::string&)> log_callback = nullptr;
};

// -----------------------------------------------------------------------------
// Statistics (kept minimal)
// -----------------------------------------------------------------------------
struct RenderStats {
    float cpu_time_ms = 0.0f;
    float gpu_time_ms = 0.0f;
    size_t total_vertices = 0;
    size_t total_triangles = 0;
    uint64_t frame_number = 0;
};

// -----------------------------------------------------------------------------
// Main Batch Renderer Class - Early Stage
// -----------------------------------------------------------------------------
class BatchRenderer {
public:
    static std::unique_ptr<BatchRenderer> Create(
        const mjModel* m, 
        const BatchRendererConfig& config = BatchRendererConfig());

    ~BatchRenderer();

    BatchRenderer(const BatchRenderer&) = delete;
    BatchRenderer& operator=(const BatchRenderer&) = delete;
    BatchRenderer(BatchRenderer&&) noexcept;
    BatchRenderer& operator=(BatchRenderer&&) noexcept;

    // -------------------------------------------------------------------------
    // Core Rendering Interface
    // -------------------------------------------------------------------------
    RenderResult Render(mjData** data_array, const int* camera_ids = nullptr);

    const std::vector<unsigned char>& GetRGBBuffer() const { return rgb_buffer_; }
    const std::vector<float>& GetDepthBuffer() const { return depth_buffer_; }

    const unsigned char* GetRGBFrame(int batch_idx) const;
    const float* GetDepthFrame(int batch_idx) const;

    const RenderStats& GetLastStats() const { return last_stats_; }

    bool IsValid() const { return initialized_; }

private:
    BatchRenderer(const mjModel* m, const BatchRendererConfig& config);

    bool Initialize();
    void Cleanup();

    // simplified rendering stages
    bool UpdateScenes(mjData** data_array, int count);
    bool RecordCommandBuffers(int count);
    bool SubmitAndWait();
    bool ReadbackResults();

    // Vulkan resource management
    bool CreateVulkanInstance();
    bool SelectPhysicalDevice();
    bool CreateLogicalDevice();
    bool CreateRenderPass();
    bool CreatePipeline();
    bool CreateFramebuffers();
    bool CreateBuffers();
    void DestroyVulkanResources();

private:
    const mjModel* model_;
    BatchRendererConfig config_;

    struct PerEnvResources {
        mjvScene scene;
        mjvCamera camera;
        mjvOption options;
        mjrContext mjr_context;  // optional: OpenGL compatibility context
    };
    std::vector<PerEnvResources> env_resources_;

    // Minimal Vulkan resource set
    struct VulkanResources {
        void* instance;
        void* physical_device;
        void* device;
        void* graphics_queue;
        void* command_pool;
        void* command_buffers;
        void* render_pass;
        void* pipeline;
        void* framebuffers;
        void* color_images;
        void* depth_images;
        void* staging_buffers;
        void* render_fences;
    } vk_;

    std::vector<unsigned char> rgb_buffer_;
    std::vector<float> depth_buffer_;

    RenderStats last_stats_;
    // may need accumulative stats as well
    bool initialized_ = false;
    uint64_t frame_counter_ = 0;
};

// -----------------------------------------------------------------------------
// Optional Utility (kept simple for now)
// -----------------------------------------------------------------------------
std::vector<std::string> EnumerateGPUs();
bool IsGPUSupported(int gpu_id);

// -----------------------------------------------------------------------------
// Development Plan (Phased Implementation Roadmap)
// -----------------------------------------------------------------------------
//
// PHASE 1 — Minimum Viable Renderer (Target: 1–2 weeks)
// ----------------------------------------------------
// Goal: Get MuJoCo → Vulkan → CPU RGB frame working (single environment)
// - Implement CreateVulkanInstance / SelectPhysicalDevice / CreateDevice
// - Create simple RenderPass + Framebuffer (no MSAA, no compute)
// - Extract mjvScene geometry from MuJoCo
// - Push static vertices to Vulkan vertex buffer
// - Render one color image to CPU via staging buffer
// - Verify image correctness using stbi_write_png
//
// PHASE 2 — Batch Rendering Support (Target: 2–3 weeks)
// ----------------------------------------------------
// - Support multiple mjData environments (loop or multiview)
// - Add per-env framebuffers and descriptor sets
// - Parallelize scene update via std::thread or Vulkan subpasses
// - Implement depth buffer readback
// - Maintain one vk::Fence per environment
//
// PHASE 3 — Resource Management & Optimization
// --------------------------------------------
// - Integrate VMA allocator
// - Persistent mapped staging buffers
// - Indirect rendering (vkCmdDrawIndirect)
// - GPU-side instancing of static geometry
//
// PHASE 4 — Advanced Features (optional, later stage)
// ---------------------------------------------------
// - Async rendering + RenderFence API
// - Profiling via Vulkan timestamp queries
// - MSAA / anisotropy / mipmaps
// - Compute shader path (for deferred shading or visibility)
// - Integration with MuJoCo's native offscreen rendering (mjrContext)
//
// Notes:
// - Keep CPU readback path simple until phase 2.
// - Avoid descriptor set complexity early on.
// - Test each Vulkan step with validation layers enabled.
//
// -----------------------------------------------------------------------------
// Need Class
// 1. Descriptors.hpp/cpp
// 2. Devices.hpp/cpp
// 3. Memory.hpp/cpp
// 4. Shaders.hpp/cpp
// 5. utils.hpp/cpp
// 6. dispatch.cpp/hpp/template