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

std::string glm_vec3_to_string(const glm::vec3& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) 
        << "(" << v.x << ", " << v.y << ", " << v.z << ")";
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

static std::vector<TextureInfo> GetExtractTextures(const mjModel* model) {
    std::vector<TextureInfo> tmpTextures_;
    tmpTextures_.reserve(model->ntex);

    for (int i = 0; i < model->ntex; ++i) {
        TextureInfo tex_info;
        tex_info.texture_id = i;
        tex_info.type = model->tex_type[i];
        tex_info.width = model->tex_width[i];
        tex_info.height = model->tex_height[i];
        int nchannel = model->tex_nchannel[i];

        // -----------------------------------------------------------
        // [MODIFIED Strategy] 为了 Vulkan 兼容性 (对齐)，统一转换为 RGBA (4通道)
        // 即使是 CubeMap，如果源是 RGB，也转为 RGBA。
        // -----------------------------------------------------------
        tex_info.channels = 4;

        int tex_data_adr = model->tex_adr[i];
        if (tex_data_adr < 0 || tex_data_adr >= model->ntexdata) continue;

        // 计算总像素数
        // 注意：MuJoCo 的 CubeMap 如果是标准条带，height 已经是 width 的 6 倍(或反之)，
        // pixel_count 自动包含了所有面。
        size_t pixel_count = static_cast<size_t>(tex_info.width) * tex_info.height;
        size_t data_size = pixel_count * tex_info.channels; // always * 4
        tex_info.data.resize(data_size);

        const unsigned char* tex_data = model->tex_data + tex_data_adr;

        // 执行转换 (RGB/Gray -> RGBA)
        if (nchannel == 3) {
            for (size_t j = 0; j < pixel_count; ++j) {
                tex_info.data[j * 4 + 0] = tex_data[j * 3 + 0];
                tex_info.data[j * 4 + 1] = tex_data[j * 3 + 1];
                tex_info.data[j * 4 + 2] = tex_data[j * 3 + 2];
                tex_info.data[j * 4 + 3] = 255; 
            }
        } else if (nchannel == 4) {
            std::memcpy(tex_info.data.data(), tex_data, pixel_count * 4);
        } else if (nchannel == 1) {
            for (size_t j = 0; j < pixel_count; ++j) {
                unsigned char gray = tex_data[j];
                tex_info.data[j * 4 + 0] = gray;
                tex_info.data[j * 4 + 1] = gray;
                tex_info.data[j * 4 + 2] = gray;
                tex_info.data[j * 4 + 3] = 255;
            }
        }

        tmpTextures_.push_back(std::move(tex_info));
        printf("Extracted texture ID %d: %dx%d, Channels=%d, Type=%d\n",
               i, tex_info.width, tex_info.height, tex_info.channels, tex_info.type);
    }
    printf("Extracted %zu textures from model.\n", tmpTextures_.size());
    return tmpTextures_;
}
// --------------------------- Texture Loading ---------------------------------
// NOTE: Now it is only a simple version to test this feature
// In the future, we may need to put this and the whole texture
// System into RenderContext for better management
// NOTE: here all the texture will be store as a 2D texture 
// which means that all the cube map will not be processed correctly
// TODO：divide the Cube and 2d textures

LoadedTextureResources BatchRenderer::LoadMaterialTextures()
{
    LoadedTextureResources result;
    
    Device &dev = *device_;
    MemoryAllocator &alloc = render_context_->allocator;
    VkQueue queue = render_context_->renderQueue; // 确保你有这个队列

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);
    
    auto sources = GetExtractTextures(model_);

    for (size_t i = 0; i < sources.size(); ++i)
    {
        const TextureInfo &tx = sources[i];
        
        // 1. 基础属性判断
        bool is_cube_type = (tx.type == mjTEXTURE_CUBE || tx.type == mjTEXTURE_SKYBOX);
        
        // 检查是否为单面 CubeMap (Source is Square)
        // 标准 MuJoCo CubeMap 数据通常 height = 6 * width (垂直条带) 
        // 或者是 width = 6 * height (水平条带，较少见)
        // 如果 width == height 且是 Cube 类型，则视为 "Single Face Repeat"
        bool is_single_face_cube = is_cube_type && (tx.width == tx.height);

        uint32_t width = tx.width;
        uint32_t height = tx.height; // 如果是 standard cube，这个 height 可能是 6*w
        
        // 真正的单面尺寸 (Face Size)
        uint32_t face_width = width;

        uint32_t layers = is_cube_type ? 6 : 1;

        LocalTexture texture;
        TextureRequirements texture_reqs;

        // 2. 创建 GPU Image 资源 (Target)
        // 无论是单面重复还是标准条带，GPU 端都需要 6 个 layer 的 CubeImage
        if (is_cube_type) {
            auto res = alloc.makeTextureCube(face_width, 1, VK_FORMAT_R8G8B8A8_SRGB);
            texture = res.first;
            texture_reqs = res.second;
        } else {
            auto res = alloc.makeTexture2D(width, height, 1, VK_FORMAT_R8G8B8A8_SRGB);
            texture = res.first;
            texture_reqs = res.second;
        }

        // 3. 准备 Staging Buffer
        // 关键点：如果是 is_single_face_cube，我们只需要上传 1 个面的数据到 Buffer
        VkDeviceSize staging_size = tx.data.size(); 
        
        // 如果是单面重复，GetExtractTextures 返回的数据大小本身就是 width*height*4
        // 如果是标准条带，返回的数据大小是 width*(width*6)*4
        // 所以直接用 tx.data.size() 是安全的
        
        HostBuffer texture_hb_staging = alloc.makeStagingBuffer(staging_size);
        memcpy(texture_hb_staging.ptr, tx.data.data(), staging_size);
        texture_hb_staging.flush(dev);

        // 4. 分配 GPU 显存并绑定
        std::optional<VkDeviceMemory> texture_backing = alloc.alloc(texture_reqs.size);
        assert(texture_backing.has_value());
        dev.dt.bindImageMemory(dev.hdl, texture.image, texture_backing.value(), 0);

        // 5. Barrier: Undefined -> Transfer Dst
        VkImageMemoryBarrier copy_prepare = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        copy_prepare.srcAccessMask = 0;
        copy_prepare.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copy_prepare.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        copy_prepare.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        copy_prepare.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_prepare.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        copy_prepare.image = texture.image;
        copy_prepare.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_prepare.subresourceRange.baseMipLevel = 0;
        copy_prepare.subresourceRange.levelCount = 1;
        copy_prepare.subresourceRange.baseArrayLayer = 0;
        // 这里的 layerCount 必须覆盖所有层，以便一次性转换整个 Image
        copy_prepare.subresourceRange.layerCount = layers; 

        dev.dt.cmdPipelineBarrier(cmdbuf,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &copy_prepare);

        // 6. 执行 Copy (核心修改部分)
        std::vector<VkBufferImageCopy> regions;

        if (is_cube_type) {
            uint32_t face_size_bytes = face_width * face_width * 4; // RGBA

            for (uint32_t face = 0; face < 6; face++) {
                VkBufferImageCopy region = {};
                
                // [SPECIAL LOGIC] 
                if (is_single_face_cube) {
                    // 情况 A: 单面重复。
                    // 所有的 GPU Layer (0-5) 都从 Buffer 的 0 偏移处读取同一份数据
                    region.bufferOffset = 0;
                } else {
                    // 情况 B: 标准条带 (Vertical Strip)。
                    // 假设数据在 Buffer 中是 +X, -X, +Y, -Y, +Z, -Z 顺序排列
                    region.bufferOffset = face * face_size_bytes;
                }

                region.bufferRowLength = 0; // Tightly packed
                region.bufferImageHeight = 0;

                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = face; // 目标层索引
                region.imageSubresource.layerCount = 1;

                region.imageExtent.width = face_width;
                region.imageExtent.height = face_width; // Cube 面是正方形
                region.imageExtent.depth = 1;

                regions.push_back(region);
            }
        } else {
            // 2D Texture
            VkBufferImageCopy region = {};
            region.bufferOffset = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { width, height, 1 };
            regions.push_back(region);
        }

        dev.dt.cmdCopyBufferToImage(cmdbuf, texture_hb_staging.buffer,
                texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                static_cast<uint32_t>(regions.size()), regions.data());

        // 7. Barrier: Transfer Dst -> Shader Read Only
        VkImageMemoryBarrier finish_prepare = copy_prepare;
        finish_prepare.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        finish_prepare.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        finish_prepare.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        finish_prepare.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        // layerCount 依然是 layers (1 或 6)

        dev.dt.cmdPipelineBarrier(cmdbuf,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
                0, 0, nullptr, 0, nullptr, 1, &finish_prepare);

        // 8. Create View & Store
        VkImageViewCreateInfo view_info = {};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.viewType = is_cube_type ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        view_info.image = texture.image;
        view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = layers;

        VkImageView view;
        REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view)); // TODO: Add error handling macro

        // Store Resources
        result.host_buffers.emplace_back(std::move(texture_hb_staging));
        MaterialTexture mat_tex(std::move(texture), view, texture_backing.value());

        if (is_cube_type) {
            TextureMapping mapping = { 1, (int)result.textures_cube.size() };
            result.global_texture_lookup.push_back(mapping);
            result.textures_cube.emplace_back(std::move(mat_tex));
        } else {
            TextureMapping mapping = { 0, (int)result.textures_2d.size() };
            result.global_texture_lookup.push_back(mapping);
            result.textures_2d.emplace_back(std::move(mat_tex));
        }
    }

    // End & Submit
    dev.dt.endCommandBuffer(cmdbuf);
    VkSubmitInfo submit_info = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &cmdbuf;

    dev.dt.queueSubmit(queue, 1, &submit_info, VK_NULL_HANDLE);
    dev.dt.deviceWaitIdle(dev.hdl);
    
    dev.dt.freeCommandBuffers(dev.hdl, tmp_pool, 1, &cmdbuf);
    dev.dt.destroyCommandPool(dev.hdl, tmp_pool, nullptr);

    return result;
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
    // NOTE: a temporary sampler for all textures
    texture_sampler_ = makeImmutableSampler(*device_, VK_SAMPLER_ADDRESS_MODE_REPEAT);
    material_textures_= LoadMaterialTextures();

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
            // Update the transforms 
             res.render_scene->Update(model_, &res.scene, data_array[i]);
         }
        
        // Choose the view, update the camera Info
        int cam_id = res.render_scene->FindCameraID(model_, "frontview");
        // 1.77 = 16:9 aspect ratio
        res.render_scene->UpdateCameraFromSimulation(model_, data_array[i], cam_id, 1.77);
        //Update camera uniform buffer
        const CameraUBO& camera_ubo = res.render_scene->GetCameraUBO();

        LOG(config_, "Scene " + std::to_string(i) + 
            " Camera view_proj=" + HexPreview(&camera_ubo.view_proj, sizeof(camera_ubo.view_proj)) +
            ", position=" + std::to_string(camera_ubo.position.x) +", "+
            std::to_string(camera_ubo.position.y) +", "+
            std::to_string(camera_ubo.position.z) +", "+
            ", forward=" + HexPreview(&camera_ubo.forward, sizeof(camera_ubo.forward)) +
            ", up=" + std::to_string(camera_ubo.up.x) +", "+
            std::to_string(camera_ubo.up.y) +", "+
            std::to_string(camera_ubo.up.z) +", "+
            ", near_plane=" + std::to_string(camera_ubo.near_plane) +
            ", far_plane=" + std::to_string(camera_ubo.far_plane) +
            ", fov=" + std::to_string(camera_ubo.fov));
        
        std::memcpy(camera_staging_buffers_[i].ptr, &camera_ubo, sizeof(CameraUBO));
        camera_staging_buffers_[i].flush(dev);
        copy_ops.push_back({camera_staging_buffers_[i].buffer, 
                           camera_uniform_buffers_[i].buffer, 
                           sizeof(CameraUBO)});

        // Update light uniform buffer
        const auto& lights = res.render_scene->GetLights();
        LOG(config_, "Scene " + std::to_string(i) + " has " + std::to_string(lights.size()) + " lights.");
        LightUBO light_ubo{};
        size_t light_count = std::min(lights.size(), size_t(10));
        for (size_t j = 0; j < light_count; ++j) {
            light_ubo.lights[j] = lights[j];
            std::string type_name;
            switch (lights[j].type) {
                case 0: type_name = "Spot"; break;
                case 1: type_name = "Directional"; break;
                case 2: type_name = "Point"; break;
                default: type_name = "Unknown (" + std::to_string(lights[j].type) + ")"; break;
            }
            LOG(config_, " Light " + std::to_string(j) + 
                        ": pos=" + glm_vec3_to_string(lights[j].position) +
                        ", direction=" + glm_vec3_to_string(lights[j].direction) +
                        ", type=" + type_name + 
                        ", ambient=" + glm_vec3_to_string(lights[j].ambient) +
                        ", diffuse=" + glm_vec3_to_string(lights[j].diffuse) +
                        ", specular=" + glm_vec3_to_string(lights[j].specular) +
                        ", attenuation=" + glm_vec3_to_string(lights[j].attenuation));
        }
        light_ubo.lightCount = static_cast<uint32_t>(light_count);
        std::memcpy(light_staging_buffers_[i].ptr, &light_ubo, sizeof(LightUBO));
        light_staging_buffers_[i].flush(dev);
        copy_ops.push_back({light_staging_buffers_[i].buffer, 
                           light_uniform_buffers_[i].buffer, 
                           sizeof(LightUBO)});

        printf("size of LightUBO: %zu bytes, non-zero bytes: %zu\n", 
            sizeof(LightUBO), CountNonZero(&light_ubo, sizeof(LightUBO)));

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
        
        // Bind descriptor set (camera + light)
        dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline_layout_, 0, 1, &descriptor_sets_[i], 0, nullptr);
        // Bind global texture descriptor set
        dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   pipeline_layout_, 1, 1, &global_texture_descriptor_set, 0, nullptr);

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
                    index_offset += drawable.geometry->GetIndexCount();
                    vertex_offset += drawable.geometry->GetVertexCount();
                    continue;
                };
                PushConstants pushConstants{};

                pushConstants.model = drawable.transform;
                pushConstants.rgba = drawable.material.rgba;
                pushConstants.specular = drawable.material.specular;
                pushConstants.emission = drawable.material.emission;
                pushConstants.shininess = drawable.material.shininess;
                pushConstants.reflectance = drawable.material.reflectance;

                // here, the cube texture id is also < 0, find why 
                bool has_valid_texture = (drawable.material.texture_id >= 0);
                bool index_in_bounds = false;

                if (has_valid_texture) {
                    // 2. 安全检查：确保 ID 没有超出 lookup 表的大小
                    if (drawable.material.texture_id < material_textures_.global_texture_lookup.size()) {
                        index_in_bounds = true;
                        auto& lookup_info = material_textures_.global_texture_lookup[drawable.material.texture_id];
                        
                        pushConstants.texture_type = lookup_info.type;
                        pushConstants.texture_index = lookup_info.index_in_array;
                    } else {
                        // 虽然 ID >= 0，但超出了 lookup table 的范围 (异常情况)
                        printf("[ERROR] Texture ID %d is out of bounds (Lookup Size: %zu)\n", 
                            drawable.material.texture_id, material_textures_.global_texture_lookup.size());
                        pushConstants.texture_type = -1;
                        pushConstants.texture_index = -1;
                    }
                } else {
                    // ID < 0，表示无纹理
                    pushConstants.texture_type = -1;
                    pushConstants.texture_index = -1; // 给一个安全的默认值，防止 Shader 读取垃圾数据
                }
                if(pushConstants.texture_type == -1) 
                {
                    printf("\n=== Drawable Status Report [GeomID: %d] ===\n", drawable.geom_id);
                    
                    printf("  > Geometry Info:\n");
                    printf("    Vertex Count : %d\n", int(drawable.geometry->GetVertexCount()));
                    printf("    Index Count  : %d\n", int(drawable.geometry->GetIndexCount()));

                    // [新增] 打印变换矩阵 (按行打印，方便阅读)
                    printf("  > Transform Matrix (Model):\n");
                    // GLM 是列主序 (Column-Major)，但 printf 通常按内存顺序打印
                    // 这里我们按直观的 4x4 矩阵格式打印 (Row-by-Row for readability)
                    // 注意：glm::mat4[col][row]，所以 matrix[0][0] 是第一列第一行，matrix[3][0] 是第四列第一行 (Tx)
                    for (int row = 0; row < 4; ++row) {
                        printf("    | %6.2f %6.2f %6.2f %6.2f |\n", 
                            pushConstants.model[0][row], 
                            pushConstants.model[1][row], 
                            pushConstants.model[2][row], 
                            pushConstants.model[3][row]);
                    }
                    
                    // 检查缩放是否异常 (矩阵的前3列向量长度)
                    float scaleX = glm::length(glm::vec3(pushConstants.model[0]));
                    float scaleY = glm::length(glm::vec3(pushConstants.model[1]));
                    float scaleZ = glm::length(glm::vec3(pushConstants.model[2]));
                    if (scaleX < 1e-6 || scaleY < 1e-6 || scaleZ < 1e-6) {
                        printf("    [WARNING] Zero or near-zero scale detected! (%.4f, %.4f, %.4f)\n", scaleX, scaleY, scaleZ);
                    }

                    printf("  > Material Props:\n");
                    printf("    RGBA        : (%.2f, %.2f, %.2f, %.2f)\n", 
                        pushConstants.rgba.r, pushConstants.rgba.g, pushConstants.rgba.b, pushConstants.rgba.a);
                    printf("    Specular    : (%.2f, %.2f, %.2f) | Shininess: %.1f\n", 
                        pushConstants.specular.x, pushConstants.specular.y, pushConstants.specular.z, pushConstants.shininess);
                    printf("    Emission    : %.2f | Reflectance: %.2f\n", 
                        pushConstants.emission, pushConstants.reflectance);

                    printf("  > Texture Pipeline:\n");
                    printf("    [1] Raw ID from Material : %d\n", drawable.material.texture_id);
                    
                    if (!has_valid_texture) {
                        printf("    [2] Lookup Status        : SKIPPED (Raw ID < 0)\n");
                        printf("    [3] Reason Check         : Material has no texture assigned in XML?\n");
                    } else if (!index_in_bounds) {
                        printf("    [2] Lookup Status        : FAILED (Out of Bounds)\n");
                    } else {
                        auto& info = material_textures_.global_texture_lookup[drawable.material.texture_id];
                        printf("    [2] Lookup Table Entry   : { Type: %d, Index: %d }\n", info.type, info.index_in_array);
                    }

                    // 注意：这里修正了之前的逻辑，你的 TextureMapping 中 2 通常代表 Cube
                    // 之前代码里三目运算符写的是 (pushConstants.texture_type == 1 ? "Cube" : "2D")，请确认你的 Enum 定义
                    // 假设 0=None/Error, 1=2D, 2=Cube
                    const char* typeName = "Unknown";
                    if (pushConstants.texture_type == -1) typeName = "None";
                    else if (pushConstants.texture_type == 0) typeName = "2D";
                    else if (pushConstants.texture_type == 1) typeName = "Cube";

                    printf("    [4] Final PushConstants  : Type = %d (%s), Index = %d\n", 
                        pushConstants.texture_type, typeName, pushConstants.texture_index);
                    printf("===============================================\n");
                }

                dev.dt.cmdPushConstants(cmd, pipeline_layout_, 
                                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                      0, sizeof(PushConstants), &pushConstants);
                // Draw this drawable
                uint32_t index_count = drawable.geometry->GetIndexCount();
                vkCmdDrawIndexed(cmd, index_count, 1, index_offset, vertex_offset, 0);
                
                index_offset += index_count;
                vertex_offset += drawable.geometry->GetVertexCount();
                // LOG(config_, "Draw drawable with "
                //     + std::to_string(index_count) + " indices at offset "
                //     + std::to_string(index_offset) + ", vertex offset "
                //     + std::to_string(vertex_offset));
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
    rasterizer.cullMode = VK_CULL_MODE_NONE;
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
    // feature 11/19: add texture support 
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0; // Camera UBO
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    bindings[1].binding = 1; // Lighting UBO
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo dslInfo{};
    dslInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    dslInfo.pBindings = bindings.data();
    REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &dslInfo, nullptr, &descriptor_set_layout_));
    

    // 1. Define the bindings
    VkDescriptorSetLayoutBinding texbindings[] = {
        // Binding 0: 2D Textures Array
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = kMaxBindlessTextures, // 例如 1024
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        // Binding 1: Cube Textures Array
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, // Cube 也使用 SAMPLED_IMAGE
            .descriptorCount = kMaxBindlessTextures, // 或者设置一个较小的上限，如 kMaxCubeTextures
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        // Binding 2: Shared Sampler (Immutable)
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = &texture_sampler_
        }
    };

    // 2. Define Binding Flags (Partially Bound for arrays)
    VkDescriptorBindingFlags binding_flags[] = { 
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, // For Binding 0 (2D)
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, // For Binding 1 (Cube)
        0                                            // For Binding 2 (Sampler)
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = 3,
        .pBindingFlags = binding_flags,
    };

    // 3. Create the layout
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = &flag_info;
    layout_info.bindingCount = 3;
    layout_info.pBindings = texbindings;

    REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &layout_info, nullptr, &global_texture_set_layout));
    
    //-------------------------------------------------------------------------//
    
    // Pipeline layout with push constants for per-drawable transform
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(float) * 16; // view-projection matrix (mat4)
    
    std::vector<VkDescriptorSetLayout> setLayouts = {
        descriptor_set_layout_,      // index 0 -> Set 0 (Camera/Light UBOs)
        global_texture_set_layout    // index 1 -> Set 1 (Bindless Textures)
    };

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
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
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = config_.batch_size; // Camera UBOs
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[1].descriptorCount = config_.batch_size; // Light UBOs
    
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
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
    {
        // Create global texture descriptor set for bindless textures
        // here use a independent pool and set for global textures
        // 1. Define the pool size (count needs to cover all descriptors)
        std::array<VkDescriptorPoolSize, 2> pool_sizes{};
        // Binding 0: 图片数组
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // 注意：不再是 COMBINED_IMAGE_SAMPLER
        pool_sizes[0].descriptorCount = kMaxBindlessTextures;  // 必须足够大！

        // Binding 1: 采样器
        pool_sizes[1].type = VK_DESCRIPTOR_TYPE_SAMPLER;
        pool_sizes[1].descriptorCount = 1;

        // 2. Create the pool
        VkDescriptorPoolCreateInfo pool_info = {};
        pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
        pool_info.pPoolSizes = pool_sizes.data();
        pool_info.maxSets = 1;

        REQ_VK(dev.dt.createDescriptorPool(dev.hdl, &pool_info, nullptr, &global_texture_descriptor_pool));
        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = global_texture_descriptor_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &global_texture_set_layout;

        REQ_VK(dev.dt.allocateDescriptorSets(dev.hdl, &alloc_info, &global_texture_descriptor_set));   
    }

    // Create uniform buffers
    camera_uniform_buffers_.clear();
    camera_staging_buffers_.clear();

    light_uniform_buffers_.clear();
    light_staging_buffers_.clear();
    for (int i = 0; i < config_.batch_size; ++i) {
        // Camera UBO
        auto camera_ubo = allocator.makeLocalBuffer(
            sizeof(CameraUBO), 
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        camera_uniform_buffers_.emplace_back(std::move(*camera_ubo));
        
        // Camera staging buffer for updates
        camera_staging_buffers_.emplace_back(
            allocator.makeStagingBuffer(sizeof(CameraUBO)));
        
        // light UBO
        auto light_ubo = allocator.makeLocalBuffer(
            sizeof(LightUBO), 
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        light_uniform_buffers_.emplace_back(std::move(*light_ubo));

        light_staging_buffers_.emplace_back(
            allocator.makeStagingBuffer(sizeof(LightUBO)));
        
        // Update descriptor sets
        VkDescriptorBufferInfo cameraBufferInfo{};
        cameraBufferInfo.buffer = camera_uniform_buffers_[i].buffer;
        cameraBufferInfo.offset = 0;
        cameraBufferInfo.range = sizeof(CameraUBO);
        
        VkDescriptorBufferInfo lightBufferInfo{};
        lightBufferInfo.buffer = light_uniform_buffers_[i].buffer;
        lightBufferInfo.offset = 0;
        lightBufferInfo.range = sizeof(LightUBO);
        
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
        writes[1].pBufferInfo = &lightBufferInfo;
        
        // TODO: Add the texture sampler descriptor
        dev.dt.updateDescriptorSets(dev.hdl, 2, writes, 0, nullptr);
    }

    // update global texture descriptor set for bindless textures
    const auto& loaded_resources = LoadMaterialTextures();
    const auto& textures_2d = loaded_resources.textures_2d;
    const auto& textures_cube = loaded_resources.textures_cube;

    std::vector<VkDescriptorImageInfo> image_infos_2d;
    std::vector<VkDescriptorImageInfo> image_infos_cube;
    
    image_infos_2d.reserve(textures_2d.size());
    image_infos_cube.reserve(textures_cube.size());

    // 2. 填充 2D Image Infos
    for (const auto& tex : textures_2d) {
        VkDescriptorImageInfo info = {};
        info.imageView = tex.view; // 这里的 view 必须是 VK_IMAGE_VIEW_TYPE_2D
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info.sampler = VK_NULL_HANDLE; // 使用 Immutable Sampler，这里填 null
        image_infos_2d.push_back(info);
    }

    // 3. 填充 Cube Image Infos
    for (const auto& tex : textures_cube) {
        VkDescriptorImageInfo info = {};
        info.imageView = tex.view; // 这里的 view 必须是 VK_IMAGE_VIEW_TYPE_CUBE
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info.sampler = VK_NULL_HANDLE; 
        image_infos_cube.push_back(info);
    }

    // 4. 构建 Writes
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(textures_2d.size() + textures_cube.size());

    // 生成 2D 纹理的 Writes (Binding 0)
    for (size_t i = 0; i < textures_2d.size(); ++i) {
        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = global_texture_descriptor_set;
        write.dstBinding = 0; // Binding 0
        write.dstArrayElement = static_cast<uint32_t>(i);
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &image_infos_2d[i]; // 指向稳定的 vector 元素地址

        writes.push_back(write);
    }

    // 生成 Cube 纹理的 Writes (Binding 1)
    for (size_t i = 0; i < textures_cube.size(); ++i) {
        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = global_texture_descriptor_set;
        write.dstBinding = 1; // Binding 1
        write.dstArrayElement = static_cast<uint32_t>(i);
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; // Cube 也是 SAMPLED_IMAGE
        write.pImageInfo = &image_infos_cube[i]; // 指向稳定的 vector 元素地址

        writes.push_back(write);
    }

    // 5. 执行批量更新
    if (!writes.empty()) {
        dev.dt.updateDescriptorSets(dev.hdl, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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

