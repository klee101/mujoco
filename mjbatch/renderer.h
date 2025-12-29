// -----------------------------------------------------------------------------
// Vulkan Batch Renderer - Main Rendering Logic
// -----------------------------------------------------------------------------
#pragma once

#include <mujoco/mujoco.h>
#include <vector>
#include <memory>
#include <string>
#include <thread>
#include <condition_variable>
#include <future>
#include <functional>
#include <cstdint>
#include <mutex>
#include <queue>
#include <functional>
#include <vulkan/vulkan.h>
#include <cstring>
#include "memcpy_avx.h"
#include "device.h"
#include "backend.h"
#include "render_context.h"
#include "memory.h"
#include "scene.h"
#include "shared_protocol.h"

#include <nvtx3/nvToolsExt.h> 

struct ScopedNvtxRange {
    ScopedNvtxRange(const char* name, uint32_t color_argb = 0xFFFFFFFF) {
        nvtxEventAttributes_t eventAttrib = {0};
        eventAttrib.version = NVTX_VERSION;
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        eventAttrib.colorType = NVTX_COLOR_ARGB;
        eventAttrib.color = color_argb;
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
        eventAttrib.message.ascii = name;
        nvtxRangePushEx(&eventAttrib);
    }
    ~ScopedNvtxRange() {
        nvtxRangePop();
    }
};

const uint32_t COLOR_PHYSICS = 0xFF00FF00; // 绿色
const uint32_t COLOR_RENDER  = 0xFFFF0000; // 红色
const uint32_t COLOR_WORKER  = 0xFFFFFF00; // 黄色
const uint32_t COLOR_LOOP    = 0xFF00FFFF; // 青色
const uint32_t C_UPDATE = 0xFF00BFFF; // Deep Sky Blue (CPU 数据准备)
const uint32_t C_RECORD = 0xFFFFA500; // Orange (Vulkan 指令录制)
const uint32_t C_SUBMIT = 0xFF8A2BE2; // Blue Violet (提交与等待 GPU)
const uint32_t C_READ   = 0xFF20B2AA; // Light Sea Green (回读数据)


class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for(size_t i = 0; i < threads; ++i)
            workers.emplace_back([this, i] { 
                std::string thread_name = "Worker-Thread-" + std::to_string(i);
                nvtxNameOsThreadA(pthread_self(), thread_name.c_str());

                for(;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
            
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if(stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    void ParallelFor(int count, const std::function<void(int start, int end)>& func) {
        int num_workers = workers.size();
        if (num_workers == 0) num_workers = 1;

        int chunk_size = (count + num_workers - 1) / num_workers;
        std::vector<std::future<void>> futures;
        futures.reserve(num_workers);

        for (int i = 0; i < num_workers; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, count);
            if (start >= end) break; 

            futures.emplace_back(enqueue([func, start, end]() {
                ScopedNvtxRange range("Physics_Worker_Job", COLOR_WORKER);
                func(start, end);
            }));
        }

        for (auto& f : futures) {
            f.get();
        }
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

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
    bool enable_validation = false;
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


struct LoadedTextureResources {
    std::vector<MaterialTexture> textures_2d;
    std::vector<TextureMapping> global_texture_lookup;
};

// Push constants: model transform + material data
struct PushConstants {
    glm::mat4 model;          // 64 bytes
    glm::vec4 rgba;           // 16 bytes
    float specular;           // 4 bytes
    float emission;           // 4 bytes
    float shininess;          // 4 bytes
    float reflectance;        // 4 bytes
    int texture_index;          // 原 texture_id (表示在对应数组中的下标)
    int texture_type;           // 原 _pad1 (-1: None, 0: 2D, 1: Cube)
    int pad[2];               // Padding to 16 bytes
};

struct MeshEntry {
    uint32_t vertex_offset;
    uint32_t index_offset;
    uint32_t vertex_count;
    uint32_t index_count;
};

// feat: zero-copy, direct return the staging buffer pointer
struct FrameObservation {
    const uint8_t* data;      // Staging Buffer Pointer
    uint32_t width;           // image width
    uint32_t height;          // image height
    uint32_t stride_bytes;    // Row Pitch
    size_t total_bytes;      
};


// -----------------------------------------------------------------------------
// Main Batch Renderer Class
// -----------------------------------------------------------------------------
class BatchRenderer {
public:
    static std::unique_ptr<BatchRenderer> Create(
        std::vector<mjModel*> models, 
        const BatchRendererConfig& config = BatchRendererConfig());

    ~BatchRenderer();

    BatchRenderer(const BatchRenderer&) = delete;
    BatchRenderer& operator=(const BatchRenderer&) = delete;
    BatchRenderer& operator=(BatchRenderer&&) noexcept;

    // Core Rendering Interface
    RenderResult Render(mjData** data_array, const int* camera_ids = nullptr);

    // [ADD] Shared Memory Interface (New)
    bool UpdateScenesFromMemory(const uint8_t* ptr, int batch_idx, int max_geom, int max_light);
    RenderResult RenderFromMemory(const uint8_t* shared_memory_ptr, int batch_idx, int max_geom, int max_light);
    bool RecordCommandBuffersFromMemory(const uint8_t* ptr, int count);

    const std::vector<unsigned char>& GetRGBBuffer() const { return rgb_buffer_; }
    const std::vector<float>& GetDepthBuffer() const { return depth_buffer_; }

    const unsigned char* GetRGBFrame(int batch_idx) const;
    const float* GetDepthFrame(int batch_idx) const;

    const RenderStats& GetLastStats() const { return last_stats_; }

    const BatchRendererConfig& GetConfig() const { return config_; }

    bool IsValid() const { return initialized_; }

private:
    BatchRenderer(std::vector<mjModel*> models, const BatchRendererConfig& config);

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

    // feat: mesh deduplication
    void InitGlobalGeometry();


private:
    std::vector<mjModel*> models_;

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
    // TODO: use the meta Framebuffer 
    // TODO: fill the FrameBuffer with scenes * views (batch_size * 3) 
    std::vector<VkFramebuffer> framebuffers_;
    std::vector<mujoco::mjbatch::LocalImage> color_images_;
    std::vector<mujoco::mjbatch::LocalImage> depth_images_;
    std::vector<VkImageView> color_image_views_;
    std::vector<VkImageView> depth_image_views_;

    // Command buffers and synchronization
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> command_buffers_;
    VkFence render_fence_;

    // Staging buffers for readback
    std::vector<mujoco::mjbatch::HostBuffer> staging_buffers_;

    std::optional<mujoco::mjbatch::LocalBuffer> global_vertex_buffer_;
    std::optional<mujoco::mjbatch::LocalBuffer> global_index_buffer_;
    std::unordered_map<std::string, MeshEntry> global_mesh_cache_;

    // Uniform buffers
    std::vector<mujoco::mjbatch::LocalBuffer> camera_uniform_buffers_;
    std::vector<mujoco::mjbatch::HostBuffer> camera_staging_buffers_;
    
    std::vector<mujoco::mjbatch::LocalBuffer> light_uniform_buffers_;
    std::vector<mujoco::mjbatch::HostBuffer> light_staging_buffers_;

    // feat: texture
    LoadedTextureResources material_textures_;
    VkSampler texture_sampler_;
    std::vector<int> texture_offsets_;

    // Output buffers
    std::vector<FrameObservation> frames;

    std::vector<unsigned char> rgb_buffer_;
    std::vector<float> depth_buffer_;

    RenderStats last_stats_;
    bool initialized_ = false;
    uint64_t frame_counter_ = 0;

    ThreadPool pool;
};


