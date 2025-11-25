// -----------------------------------------------------------------------------
// Vulkan Batch Renderer - Main Rendering Logic
// -----------------------------------------------------------------------------
#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "device.h"
#include "backend.h"
#include "render_context.h"
#include "memory.h"
#include "scene.h"

#define kMaxBindlessTextures 32
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
    RenderError error = RenderError::SUCCESS;
    std::string message;
    int failed_batch_idx = -1;
    bool IsSuccess() const { return error == RenderError::SUCCESS; }
    explicit operator bool() const { return IsSuccess(); }
};

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------
struct BatchRendererConfig {
    int gpu_id = 0;
    int batch_size = 1;
    int frame_width = 640;
    int frame_height = 480;
    bool enable_depth = true;

    // Vulkan setup
    bool enable_validation = true;
    int max_frames_in_flight = 1;

    // Logging
    std::function<void(const std::string&)> log_callback = nullptr;
};

// -----------------------------------------------------------------------------
// Statistics
// -----------------------------------------------------------------------------
struct RenderStats {
    float cpu_time_ms = 0.0f;
    float gpu_time_ms = 0.0f;
    size_t total_vertices = 0;
    size_t total_triangles = 0;
    uint64_t frame_number = 0;
};


struct LightUBO {
    mujoco::mjbatch::LightInfo lights[10];
    uint32_t lightCount;
    uint32_t pad[3];
};

struct MaterialTexture {
    MaterialTexture(mujoco::mjbatch::LocalTexture &&src_image,
                    VkImageView src_view,
                    VkDeviceMemory src_backing)
        : image(std::move(src_image)), view(src_view), backing(src_backing)
    {
    }

    mujoco::mjbatch::LocalTexture image;
    VkImageView view;
    VkDeviceMemory backing;
};

struct TextureMapping {
    int type; // 0 = None, 1 = 2D, 2 = Cube
    int index_in_array; // 在对应的 vector<MaterialTexture> 中的下标
};

// 函数的返回结果结构体
struct LoadedTextureResources {
    std::vector<MaterialTexture> textures_2d;
    std::vector<MaterialTexture> textures_cube;
    std::vector<TextureMapping> global_texture_lookup;
    std::vector<mujoco::mjbatch::HostBuffer> host_buffers;
};

// Push constants: model transform + material data
struct PushConstants {
    glm::mat4 model;          // 64 bytes
    glm::vec4 rgba;           // 16 bytes
    glm::vec3 specular;       // 12 bytes
    float emission;           // 4 bytes
    float shininess;          // 4 bytes
    float reflectance;        // 4 bytes
    int texture_index;          // 原 texture_id (表示在对应数组中的下标)
    int texture_type;           // 原 _pad1 (-1: None, 0: 2D, 1: Cube)
};
// -----------------------------------------------------------------------------
// Main Batch Renderer Class
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

    // Core Rendering Interface
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

    // Rendering stages
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

    // Helper functions
    VkShaderModule loadShaderModule(const std::string& shader_path);
    LoadedTextureResources LoadMaterialTextures();

    std::vector<uint32_t> readSPIRV(const std::string& filename);

private:
    const mjModel* model_;
    BatchRendererConfig config_;

    // Backend and device
    std::unique_ptr<mujoco::mjbatch::Backend> backend_;
    std::unique_ptr<mujoco::mjbatch::Device> device_;
    std::unique_ptr<mujoco::mjbatch::RenderContext> render_context_;

    struct PerEnvResources {
        mjvScene scene;
        mjvCamera camera;
        mjvOption options;
        std::unique_ptr<mujoco::mjbatch::Scene> render_scene;  // Extracted render-ready scene
    };
    std::vector<PerEnvResources> env_resources_;

    // Vulkan resources
    VkPipeline graphics_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptor_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptor_sets_;

    VkDescriptorSetLayout global_texture_set_layout = VK_NULL_HANDLE;  
    VkDescriptorPool global_texture_descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet global_texture_descriptor_set;

    VkShaderModule vert_shader_module_ = VK_NULL_HANDLE;
    VkShaderModule frag_shader_module_ = VK_NULL_HANDLE;

    // Framebuffers and images for batch rendering
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<mujoco::mjbatch::LocalImage> color_images_;
    std::vector<mujoco::mjbatch::LocalImage> depth_images_;
    std::vector<VkImageView> color_image_views_;
    std::vector<VkImageView> depth_image_views_;

    // Command buffers and synchronization
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;
    std::vector<VkFence> render_fences_;

    // Staging buffers for readback
    std::vector<mujoco::mjbatch::HostBuffer> staging_buffers_;

    // Vertex/index buffers per environment
    std::vector<mujoco::mjbatch::LocalBuffer> vertex_buffers_;
    std::vector<mujoco::mjbatch::LocalBuffer> index_buffers_;

    // 这一部分内容在目前的程序中并没有被使用到
    std::vector<size_t> vertex_counts_;  // Number of vertices per environment
    std::vector<size_t> index_counts_;  // Number of indices per environment
    std::vector<size_t> vertex_buffer_sizes_;  // Buffer sizes in bytes
    std::vector<size_t> index_buffer_sizes_;  // Buffer sizes in bytes

    // Uniform buffers
    std::vector<mujoco::mjbatch::LocalBuffer> camera_uniform_buffers_;
    std::vector<mujoco::mjbatch::HostBuffer> camera_staging_buffers_;
    
    std::vector<mujoco::mjbatch::LocalBuffer> light_uniform_buffers_;
    std::vector<mujoco::mjbatch::HostBuffer> light_staging_buffers_;

    // feat: texture
    LoadedTextureResources material_textures_;
    VkSampler texture_sampler_;

    // Output buffers
    std::vector<unsigned char> rgb_buffer_;
    std::vector<float> depth_buffer_;

    RenderStats last_stats_;
    bool initialized_ = false;
    uint64_t frame_counter_ = 0;
};

// -----------------------------------------------------------------------------
// Utility Functions
// -----------------------------------------------------------------------------
std::vector<std::string> EnumerateGPUs();
bool IsGPUSupported(int gpu_id);

