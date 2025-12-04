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


std::string glm_vec3_to_string(const glm::vec3& v) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) 
        << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return oss.str();
}

// --------------------------- Factory / ctor ---------------------------------
std::unique_ptr<BatchRenderer> BatchRenderer::Create(
    std::vector<mjModel*> models,
    const BatchRendererConfig& config)
{
    if (models.empty()) {
        return nullptr;
    }

    auto renderer = std::unique_ptr<BatchRenderer>(new BatchRenderer(models, config));
    if (!renderer->Initialize()) {
        renderer->Cleanup();
        return nullptr;
    }
    return renderer;
}

BatchRenderer::BatchRenderer(std::vector<mjModel*> models, const BatchRendererConfig& config)
    : models_(models), config_(config), pool(config.batch_size)
{
    // Reserve environment resources vector
    env_resources_.resize(config_.batch_size);
    // Preallocate output buffers on CPU
    rgb_buffer_.resize((size_t)config_.batch_size * config_.frame_width * config_.frame_height * 4);
    if (config_.enable_depth) {
        depth_buffer_.resize((size_t)config_.batch_size * config_.frame_width * config_.frame_height);
    }

}

BatchRenderer::~BatchRenderer() {
    Cleanup();
}

// BatchRenderer::BatchRenderer(BatchRenderer&& other) noexcept {
//     pool = std::move(other.pool);
//     *this = std::move(other);
// }

BatchRenderer& BatchRenderer::operator=(BatchRenderer&& other) noexcept {
    if (this != &other) {
        Cleanup();
        models_ = other.models_;
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
        render_fence_ = std::move(other.render_fence_);
        staging_buffers_ = std::move(other.staging_buffers_);
        global_vertex_buffer_ = std::move(other.global_vertex_buffer_);
        global_index_buffer_ = std::move(other.global_index_buffer_);
        global_mesh_cache_ = std::move(other.global_mesh_cache_);
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
        // print the name
        auto texadr = model->name_texadr[i];
        std::string tex_name = std::string(model->names + texadr);
        if(tex_name.empty() && tex_info.type == mjTEXTURE_SKYBOX) {
            tex_name = "skybox-" + std::to_string(i);
        }
        tex_info.name = tex_name;

        // printf("Texture ID %d Name: %s\n", i, tex_name.c_str());

        if (tex_info.type == mjTEXTURE_CUBE || tex_info.type == mjTEXTURE_SKYBOX) {
            // for those builtin cubemap, transform it to standard 2d and process in shader
            if (tex_info.height == tex_info.width * 6) {
                tex_info.height = tex_info.height / 6;
            }
        }
        // -----------------------------------------------------------
        // [MODIFIED Strategy] 为了 Vulkan 兼容性 (对齐)，统一转换为 RGBA (4通道)
        // 即使是 CubeMap，如果源是 RGB，也转为 RGBA。
        // -----------------------------------------------------------
        tex_info.channels = 4;

        int tex_data_adr = model->tex_adr[i];
        if (tex_data_adr < 0 || tex_data_adr >= model->ntexdata) continue;

        // 计算总像素数
        // 注意：对于所有的1*6 CubeMap，只取第一个面
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
        // printf("Extracted texture ID %d: %dx%d, Channels=%d, Type=%d\n",
        //        i, tex_info.width, tex_info.height, tex_info.channels, tex_info.type);
    }
    // printf("Extracted %zu textures from model.\n", tmpTextures_.size());
    return tmpTextures_;
}
// --------------------------- Texture Loading ---------------------------------
// NOTE: Now it is only a simple version to test this feature
// In the future, we may need to put this and the whole texture
// System into RenderContext for better management
// NOTE: here all the texture will be store as a 2D texture 
// which means that all the cube map will not be processed correctly

LoadedTextureResources BatchRenderer::LoadMaterialTextures()
{
    LoadedTextureResources result;
    std::vector<HostBuffer> host_buffers; 
    
    Device &dev = *device_;
    MemoryAllocator &alloc = render_context_->allocator;
    VkQueue queue = render_context_->renderQueue; // 确保你有这个队列

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);

    // [新增] 全局纹理名称缓存: Name -> Index in result.textures_2d
    std::unordered_map<std::string, int> global_tex_cache;

    for (const mjModel* current_model : models_) {
        // here, record the offset before loading this model's textures
        texture_offsets_.push_back((int)result.global_texture_lookup.size());

        if (!current_model) continue;
        auto sources = GetExtractTextures(current_model);

        for (size_t i = 0; i < sources.size(); ++i)
        {
            const TextureInfo &tx = sources[i];
            std::string unique_name = tx.name;

            // [新增] 1. 检查缓存 (Deduplication)
            if (global_tex_cache.find(unique_name) != global_tex_cache.end()) {
                int cached_idx = global_tex_cache[unique_name];
                
                // 复用已存在的纹理 ID
                // Mapping Type 0 (2D), Index = cached_idx
                TextureMapping mapping = { 0, cached_idx }; 
                result.global_texture_lookup.push_back(mapping);
                
                // printf("Texture Deduplicated: %s -> ID %d\n", unique_name.c_str(), cached_idx);
                continue; // 跳过后续上传步骤
            }

            uint32_t width = tx.width;
            uint32_t height = tx.height;

            LocalTexture texture;
            TextureRequirements texture_reqs;

            // 在 GetExtractTextures 把所有纹理处理为 1*1 2D 
            auto res = alloc.makeTexture2D(width, height, 1, VK_FORMAT_R8G8B8A8_SRGB);
            texture = res.first;
            texture_reqs = res.second;

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
            // MAIN GPU!!!! 绝大部分的显存占用都是在这里分配的！！！！
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
            // Always 1
            copy_prepare.subresourceRange.layerCount = 1; 

            dev.dt.cmdPipelineBarrier(cmdbuf,
                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &copy_prepare);

            // [MODIFIED] 6. 执行 Copy (简单直接，不再需要 Region Loop)
            VkBufferImageCopy region = {};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { width, height, 1 };

            dev.dt.cmdCopyBufferToImage(cmdbuf, texture_hb_staging.buffer,
                    texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, &region);

            // 7. Barrier: Transfer Dst -> Shader Read Only
            VkImageMemoryBarrier finish_prepare = copy_prepare;
            finish_prepare.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            finish_prepare.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            finish_prepare.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            finish_prepare.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            dev.dt.cmdPipelineBarrier(cmdbuf,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 
                    0, 0, nullptr, 0, nullptr, 1, &finish_prepare);

            // 8. Create View & Store
            VkImageViewCreateInfo view_info = {};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.image = texture.image;
            view_info.format = VK_FORMAT_R8G8B8A8_SRGB;
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = 0;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;

            VkImageView view;
            REQ_VK(dev.dt.createImageView(dev.hdl, &view_info, nullptr, &view)); // TODO: Add error handling macro

            // Store Resources
            host_buffers.emplace_back(std::move(texture_hb_staging));
            MaterialTexture mat_tex(std::move(texture), view, texture_backing.value());

            // [MODIFIED] 9. 存入结果并更新缓存
            int new_idx = (int)result.textures_2d.size();
            
            // 存入 textures_2d (不再存入 textures_cube)
            result.textures_2d.emplace_back(std::move(mat_tex));
            TextureMapping mapping;
            // 记录到 Lookup (Type always 0)
            if(tx.type == mjTEXTURE_CUBE || tx.type == mjTEXTURE_SKYBOX) {
                mapping = { 1, new_idx };
            }
            else {
                mapping = { 0, new_idx };
            }

            result.global_texture_lookup.push_back(mapping);
            
            // 更新缓存
            global_tex_cache[unique_name] = new_idx;
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

// TODO: finish the DedupMeshData function
// TODO: use two mjmodel to test firstly
// TODO: use the megaVertex/index buffer
// ---------------------------- Load and Deduplicate Vertices ------------------------
void BatchRenderer::InitGlobalGeometry() {
    LOG(config_, "InitGlobalGeometry(): Starting mesh deduplication and upload...");

    // 1. 准备临时主机内存 (Host Memory)
    std::vector<Vertex> global_vertices;
    std::vector<uint32_t> global_indices;

    // 预估大小以减少 resize 开销 (假设平均每个 Mesh 1000 顶点)
    size_t estimated_meshes = 0;
    for(auto model_ : models_) {
        estimated_meshes += (model_ ? model_->nmesh : 0) + 10; // +10 for primitives
    }
    global_vertices.reserve(estimated_meshes * 10000);
    global_indices.reserve(estimated_meshes * 10000); // 假设平均每个 Mesh 3000 索引

    // --- 辅助 Lambda: 缓存几何体逻辑 ---
    // generator: 一个函数对象，调用它会返回 GeometryBuffers
    auto ProcessGeometry = [&](const std::string& unique_name, std::function<GeometryBuffers()> generator) {
        // Deduplication Check: 如果名字已存在，直接跳过
        if (global_mesh_cache_.find(unique_name) != global_mesh_cache_.end()) {
            return; 
        }

        // 生成几何数据
        GeometryBuffers buffers = generator();

        if (buffers.vertices.empty()) {
            LOG(config_, "Warning: Empty geometry generated for " + unique_name);
            return;
        }

        // 记录缓存条目
        MeshEntry entry;
        entry.vertex_offset = static_cast<uint32_t>(global_vertices.size());
        entry.index_offset  = static_cast<uint32_t>(global_indices.size());
        entry.vertex_count  = static_cast<uint32_t>(buffers.vertices.size());
        entry.index_count   = static_cast<uint32_t>(buffers.indices.size());

        // Merge 到全局大数组
        // 顶点直接追加
        global_vertices.insert(global_vertices.end(), buffers.vertices.begin(), buffers.vertices.end());
        
        // 索引需要加上当前的 vertex_offset (Base Vertex) 吗？
        // 保持 indices 原样 (0,1,2...)。渲染时 vkCmdDrawIndexed 的 vertexOffset 参数填 entry.vertex_offset
        global_indices.insert(global_indices.end(), buffers.indices.begin(), buffers.indices.end());

        // 存入 Map
        global_mesh_cache_[unique_name] = entry;

        // Debug Log (Optional)
        // std::string log_msg = "Cached: " + unique_name + 
        //                       " (V:" + std::to_string(entry.vertex_count) + 
        //                       ", I:" + std::to_string(entry.index_count) + ")";
        // LOG(config_, log_msg);
    };

    // 2. 处理 mjModel 中的 Mesh
    for (int i_model = 0; i_model < models_.size(); ++i_model) {
        for (int i = 0; i < (2 * models_[i_model]->nmesh); i+=2) {
            std::string name;
            // 获取 Mesh 名称，必须与 Scene::ExtractGeometries 逻辑一致
            if (models_[i_model]->names && models_[i_model]->name_meshadr[i/2] >= 0) {
                name = std::string(models_[i_model]->names + models_[i_model]->name_meshadr[i/2]);
                // LOG(config_, "Processing mesh: " + name);
            } else {
                name = "mesh_" + std::to_string(i/2);
                // LOG(config_, "Processing unnamed mesh ID: " + std::to_string(i/2));
            }

            // 调用 GeometryBuilder::BuildMesh
            ProcessGeometry(name, [this, i, i_model]() {
                return GeometryBuilder::BuildMesh(models_[i_model], i/2);
            });
        }
    }

    // 3. 处理 Built-in Primitives (基础几何体)
    // here set all builtin geom the same size params 
    ProcessGeometry("__builtin_box",      [](){ return GeometryBuilder::BuildBox(24); });
    ProcessGeometry("__builtin_sphere",   [](){ return GeometryBuilder::BuildSphere(16, 16); }); // 16 stacks/slices
    ProcessGeometry("__builtin_capsule",  [](){ return GeometryBuilder::BuildCapsule(16, 16); });
    ProcessGeometry("__builtin_cylinder", [](){ return GeometryBuilder::BuildCylinder(100, 100); });
    ProcessGeometry("__builtin_plane",    [](){ return GeometryBuilder::BuildPlane(10); }); // Simple quad

    // 4. 上传到 GPU
    if (global_vertices.empty() || global_indices.empty()) {
        LOG(config_, "InitGlobalGeometry(): No geometry to upload.");
        return;
    }

    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;
    VkCommandBuffer cmd = render_context_->load_cmd_;

    // 计算总大小
    VkDeviceSize v_size = global_vertices.size() * sizeof(Vertex);
    VkDeviceSize i_size = global_indices.size() * sizeof(uint32_t);

    // A. 创建 Staging Buffers (Host Visible)
    // MAIN GPU ！！！ 也有25.67 MB的显存占用！！！！
    // 25.67 * 1024 * 1024 = 48 * 560808 合理
    auto v_staging = allocator.makeStagingBuffer(v_size);
    auto i_staging = allocator.makeStagingBuffer(i_size);

    // Memcpy 数据
    std::memcpy(v_staging.ptr, global_vertices.data(), v_size);
    std::memcpy(i_staging.ptr, global_indices.data(), i_size);
    
    // Flush (确保 CPU 写完)
    v_staging.flush(dev);
    i_staging.flush(dev);

    // B. 创建 GPU Buffers (Device Local)
    // 注意：Transfer Dst 用于拷贝接收
    // MAIN GPU ！！！ 也有25.67 MB的显存占用！！！！
    // 25.67 * 1024 * 1024 = 48 * 560808 合理
    auto v_local = allocator.makeLocalBuffer(v_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto i_local = allocator.makeLocalBuffer(i_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // 移动所有权到成员变量
    global_vertex_buffer_ = std::move(*v_local);
    global_index_buffer_ = std::move(*i_local);

    // C. 录制拷贝命令
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));

    // Copy Vertex
    VkBufferCopy v_copy{};
    v_copy.size = v_size;
    dev.dt.cmdCopyBuffer(cmd, v_staging.buffer, global_vertex_buffer_->buffer, 1, &v_copy);

    // Copy Index
    VkBufferCopy i_copy{};
    i_copy.size = i_size;
    dev.dt.cmdCopyBuffer(cmd, i_staging.buffer, global_index_buffer_->buffer, 1, &i_copy);

    // 插入 Barrier 确保 Copy 完成后再被 Vertex Input 读取 (Optional but recommended)
    VkBufferMemoryBarrier barriers[2] = {};
    // Vertex Buffer Barrier
    barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[0].buffer = global_vertex_buffer_->buffer;
    barriers[0].offset = 0;
    barriers[0].size = VK_WHOLE_SIZE;
    // Index Buffer Barrier
    barriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barriers[1].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
    barriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barriers[1].buffer = global_index_buffer_->buffer;
    barriers[1].offset = 0;
    barriers[1].size = VK_WHOLE_SIZE;

    dev.dt.cmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0,
        0, nullptr,
        2, barriers,
        0, nullptr
    );

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    // D. 提交并等待
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    // 确保 load_fence_ 处于非 signaled 状态 (reset)
    resetFence(dev, render_context_->load_fence_);

    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, render_context_->load_fence_));

    // 等待上传完成
    waitForFenceInfinitely(dev, render_context_->load_fence_);
    
    // Staging Buffers 在此处析构释放
    LOG(config_, "InitGlobalGeometry(): Upload complete. "
                 "Total Verts: " + std::to_string(global_vertices.size()) + 
                 ", Total Idx: " + std::to_string(global_indices.size()));
}

// --------------------------- Initialize / Cleanup ----------------------------
bool BatchRenderer::Initialize() {
    LOG(config_, "Initialize(): starting");

    if (models_.empty()) {
        LOG(config_, "Initialize(): models_ is null");
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

    // here load all the textures from models
    // WARNING：这个函数进去之后会占用大量显存！！！！！
    // 6张图片的lift，每个scene的texture约会占用960MB的显存！！！！
    // 因此，显存的优化核心，在于如何减少texture的占用！！！！
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
    // if (!CreateBuffers()) {
    //     LOG(config_, "Initialize(): CreateBuffers failed");
    //     return false;
    // }

    InitGlobalGeometry();

    // Create buffers (Uniform, Command, etc.) - 此时不再分配 dummy vertex buffers
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
        mjv_makeScene(models_[i], &res.scene, 2000);

        mjv_defaultFreeCamera(models_[i], &res.camera);
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

// -------------------------------------------------------
    // Phase 1: Update Scenes (CPU 计算密集型)
    // 这里的耗时代表：从 mjData 提取骨骼变换并拷贝到 Staging Buffer 的时间
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("1. Update_Scenes (CPU)", C_UPDATE);
        if (!UpdateScenes(data_array, count)) {
            return MakeError(RenderError::MUJOCO_ERROR, "UpdateScenes failed");
        }
    }

    // -------------------------------------------------------
    // Phase 2: Record Command Buffers (CPU/Driver 开销)
    // 这里的耗时代表：生成 Vulkan 渲染指令的时间
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("2. Record_Cmds (Driver)", C_RECORD);
        if (!RecordCommandBuffers(count)) {
            return MakeError(RenderError::VULKAN_ERROR, "RecordCommandBuffers failed");
        }
    }

    // -------------------------------------------------------
    // Phase 3: Submit & Wait (CPU 阻塞 / GPU 执行)
    // 这里的耗时代表：vkQueueSubmit + vkWaitForFences
    // 在 nsys 中，这段时间 CPU 处于 Wait 状态，而 GPU 处于 Active 状态
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("3. Submit_Wait (GPU)", C_SUBMIT);
        if (!SubmitAndWait()) {
            return MakeError(RenderError::VULKAN_ERROR, "SubmitAndWait failed");
        }
    }

    // -------------------------------------------------------
    // !!!! 主要耗时在这里 !!!!
    // Phase 4: Readback (PCIe 传输)
    // 这里的耗时代表：vkCmdCopyImageToBuffer (GPU端) 完成后，
    // CPU 映射内存并将像素数据从显存/Staging拷贝出来的开销
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("4. Readback_Pixels (PCIe)", C_READ);
        if (!ReadbackResults()) {
            return MakeError(RenderError::VULKAN_ERROR, "ReadbackResults failed");
        }
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

    struct CopyInfo {
        VkBuffer src;
        VkBuffer dst;
        VkDeviceSize size;
    };
    std::vector<HostBuffer> staging_buffers;
    std::vector<CopyInfo> copy_ops;
    // NOTE: here count is enough? maybe 2 * count and 4 * count 
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
        mjv_updateScene(models_[i], data_array[i], &res.options, nullptr, 
                    &res.camera, mjCAT_ALL, &res.scene);
        
        // First time: create scene with geometry
        if (!res.render_scene) {
            res.render_scene = std::make_unique<mujoco::mjbatch::Scene>(models_[i], &res.scene);
        } else {
            // Update the transforms 
             res.render_scene->Update(models_[i], &res.scene, data_array[i]);
         }
        
        // Choose the view, update the camera Info
        int cam_id = res.render_scene->FindCameraID(models_[i], "frontview");
        // 1.77 = 16:9 aspect ratio
        res.render_scene->UpdateCameraFromSimulation(models_[i], data_array[i], cam_id, 1.77);
        //Update camera uniform buffer
        const CameraUBO& camera_ubo = res.render_scene->GetCameraUBO();
        
        std::memcpy(camera_staging_buffers_[i].ptr, &camera_ubo, sizeof(CameraUBO));
        camera_staging_buffers_[i].flush(dev);
        copy_ops.push_back({camera_staging_buffers_[i].buffer, 
                           camera_uniform_buffers_[i].buffer, 
                           sizeof(CameraUBO)});

        // Update light uniform buffer
        const auto& lights = res.render_scene->GetLights();
        // LOG(config_, "Scene " + std::to_string(i) + " has " + std::to_string(lights.size()) + " lights.");
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
            // LOG(config_, " Light " + std::to_string(j) + 
            //             ": pos=" + glm_vec3_to_string(lights[j].position) +
            //             ", direction=" + glm_vec3_to_string(lights[j].direction) +
            //             ", type=" + type_name + 
            //             ", ambient=" + glm_vec3_to_string(lights[j].ambient) +
            //             ", diffuse=" + glm_vec3_to_string(lights[j].diffuse) +
            //             ", specular=" + glm_vec3_to_string(lights[j].specular) +
            //             ", attenuation=" + glm_vec3_to_string(lights[j].attenuation));
        }
        light_ubo.lightCount = static_cast<uint32_t>(light_count);
        std::memcpy(light_staging_buffers_[i].ptr, &light_ubo, sizeof(LightUBO));
        light_staging_buffers_[i].flush(dev);
        copy_ops.push_back({light_staging_buffers_[i].buffer, 
                           light_uniform_buffers_[i].buffer, 
                           sizeof(LightUBO)});

        // printf("size of LightUBO: %zu bytes, non-zero bytes: %zu\n", 
        //     sizeof(LightUBO), CountNonZero(&light_ubo, sizeof(LightUBO)));

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
        PerEnvResources &res = env_resources_[i];
        
        // --- 1. Begin Recording & Render Pass ---
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
        
        VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassInfo.renderPass = render_context_->renderPass;
        renderPassInfo.framebuffer = framebuffers_[i];
        renderPassInfo.renderArea.extent = {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height};
        
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; // Dark gray bg
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();
        
        dev.dt.cmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        
        // --- 2. Bind Pipeline & Global State ---
        dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
        
        // Set dynamic state
        VkViewport viewport = {0.0f, 0.0f, (float)config_.frame_width, (float)config_.frame_height, 0.0f, 1.0f};
        dev.dt.cmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor = {{0, 0}, {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height}};
        dev.dt.cmdSetScissor(cmd, 0, 1, &scissor);

        // Bind Descriptor Sets (Camera/Light + Textures)
        std::vector<VkDescriptorSet> sets = { descriptor_sets_[i], global_texture_descriptor_set };
        dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 
                                     0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

        // --- 3. Bind Global Geometry Buffers  ---
        if (global_vertex_buffer_->buffer != VK_NULL_HANDLE && global_index_buffer_->buffer != VK_NULL_HANDLE) {
            VkBuffer vbs[] = { global_vertex_buffer_->buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
            vkCmdBindIndexBuffer(cmd, global_index_buffer_->buffer, 0, VK_INDEX_TYPE_UINT32);

            // --- 4. Draw Loop ---
            const auto& drawables = res.render_scene->GetDrawables();
            
            for (const auto& drawable : drawables) {
                if (!drawable.visible) continue;

                // [LOOKUP] Find geometry offsets in global cache
                auto it = global_mesh_cache_.find(drawable.global_mesh_name);
                if (it == global_mesh_cache_.end()) {
                    LOG(config_, "Warning: Mesh not found in cache: " + drawable.global_mesh_name);
                    continue;
                }
                const MeshEntry& entry = it->second;

                // Setup Push Constants (Transform + Material + Texture)
                PushConstants pushConstants{};
                pushConstants.model = drawable.transform;
                pushConstants.rgba = drawable.material.rgba;
                pushConstants.specular = drawable.material.specular;
                pushConstants.emission = drawable.material.emission;
                pushConstants.shininess = drawable.material.shininess;
                pushConstants.reflectance = drawable.material.reflectance;


                // Texture Lookup Logic (Flat index + Cache lookup)
                bool has_texture = (drawable.material.texture_id >= 0);

                int current_model_tex_offset = 0;
                // 安全检查，防止 models 数量和 batch_size 不一致
                if (i < texture_offsets_.size()) {
                    current_model_tex_offset = texture_offsets_[i];
                }
                int global_tex_id = current_model_tex_offset + drawable.material.texture_id;

                if (has_texture && global_tex_id < material_textures_.global_texture_lookup.size()) {
                    const auto& tex_map = material_textures_.global_texture_lookup[global_tex_id];
                    pushConstants.texture_type = tex_map.type;
                    pushConstants.texture_index = tex_map.index_in_array;
                } else {
                    pushConstants.texture_type = -1;
                    pushConstants.texture_index = -1;
                }

                // ===========================================================
                // [DEBUG LOG] 纹理索引映射检查
                // ===========================================================
                // 限制日志输出频率：仅打印前 100 次 draw call 的信息
                // 如果需要持续调试，可以移除 static counter 限制，但会导致 Log 刷屏
                static int s_tex_debug_count = 0;
                if (s_tex_debug_count < 100 && pushConstants.texture_type != -1) {
                    s_tex_debug_count++;

                    std::string status_msg;
                    bool is_out_of_bounds = has_texture && (global_tex_id >= material_textures_.global_texture_lookup.size());

                    if (!has_texture) {
                        status_msg = "SKIP (Raw ID < 0)";
                    } else if (is_out_of_bounds) {
                        status_msg = "ERROR (Out of Bounds)";
                    } else {
                        status_msg = "OK";
                    }

                    // 格式化输出字符串
                    // char buf[512];
                    // snprintf(buf, sizeof(buf), 
                    //     "[TexMapping] Batch=%d | MeshName=%s | RawID=%d | LookupSize=%zu | %s -> {Type=%d, Index=%d}", 
                    //     i,                                              // 当前 Batch 索引
                    //     drawable.global_mesh_name.c_str(),              // Mesh 名字 (用于确认是哪个物体)
                    //     drawable.material.texture_id,                   // 原始材质中的 ID
                    //     material_textures_.global_texture_lookup.size(),// 全局查找表大小
                    //     status_msg.c_str(),                             // 状态
                    //     pushConstants.texture_type,                     // 最终传给 Shader 的类型
                    //     pushConstants.texture_index                     // 最终传给 Shader 的数组下标
                    // );
                    
                    // LOG(config_, std::string(buf));

                    // 如果发现越界，额外打印一条显眼的警告
                    if (is_out_of_bounds) {
                        LOG(config_, "  !!! CRITICAL WARNING: Texture ID " + 
                            std::to_string(drawable.material.texture_id) + 
                            " exceeds global lookup size!");
                    }
                }

                dev.dt.cmdPushConstants(cmd, pipeline_layout_, 
                                      VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                      0, sizeof(PushConstants), &pushConstants);

                // [DRAW] 使用 Global Buffer 的偏移量
                // indexCount: entry.index_count
                // instanceCount: 1
                // firstIndex: entry.index_offset (全局索引缓冲中的起始位置)
                // vertexOffset: entry.vertex_offset (全局顶点缓冲中的起始位置，会被加到索引值上)
                // firstInstance: 0
                vkCmdDrawIndexed(cmd, 
                               entry.index_count, 
                               1, 
                               entry.index_offset, 
                               entry.vertex_offset, 
                               0);
            }
        }

        dev.dt.cmdEndRenderPass(cmd);
        REQ_VK(dev.dt.endCommandBuffer(cmd));
    }
    
    return true;
}


bool BatchRenderer::SubmitAndWait() {
    Device &dev = *device_;
    VkQueue queue = render_context_->renderQueue;
    const uint64_t TIMEOUT_NS = 10000000000ULL; // 10s timeout 
    // --- 1. 准备阶段：重置 Fence ---
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &render_fence_));

    // --- 2. 准备提交信息 (Batching) ---
    // 这里的关键是：将所有 command buffers 填入同一个 VkSubmitInfo
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = (uint32_t)command_buffers_.size();
    submitInfo.pCommandBuffers = command_buffers_.data(); // 指向 vector 的数据首地址
    

    // --- 3. 一次性提交 (One Submit) ---
    // 这是一个原子操作，驱动会把这一堆 CmdBuffer 一口气喂给 GPU
    VkResult submitResult = dev.dt.queueSubmit(queue, 1, &submitInfo, render_fence_);
    
    if (submitResult != VK_SUCCESS) {
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), "ERROR: Batch queueSubmit failed: %d", submitResult);
        LOG(config_, log_buf);
        return false;
    }

    // --- 4. 一次性等待 (One Wait) ---
    {
        // 加上 NVTX 标记方便你在 Nsight Systems 里验证优化效果
        ScopedNvtxRange range("CPU_Wait_Fence", 0xFF800000); 
        
        VkResult waitResult = dev.dt.waitForFences(
            dev.hdl,
            1,
            &render_fence_,
            VK_TRUE, // Wait All (虽然只有一个)
            TIMEOUT_NS
        );

        if (waitResult != VK_SUCCESS) {
            char log_buf[256];
            snprintf(log_buf, sizeof(log_buf), "ERROR: Batch fence wait failed: %d", waitResult);
            LOG(config_, log_buf);
            return false;
        }
    }
    
    return true;
}

bool BatchRenderer::ReadbackResults() {
    Device &dev = *device_;
    VkCommandBuffer cmd = render_context_->load_cmd_;
    
    // --- 1. Begin Recording ---
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));

    // --- 2. Record Commands for ALL batches ---
    for (int i = 0; i < config_.batch_size; ++i) {
        // A. Barrier: Color Attachment -> Transfer Src
        VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
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

        dev.dt.cmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);

        // B. Copy Image -> Staging Buffer
        VkBufferImageCopy region = {};
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

        // C. Barrier: Transfer Src -> Color Attachment (Restore layout for next frame)
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        dev.dt.cmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            0, 0, nullptr, 0, nullptr, 1, &barrier);
    }
    // --- 3. End & Submit ---
    REQ_VK(dev.dt.endCommandBuffer(cmd));

    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    resetFence(dev, render_context_->load_fence_);
    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, render_context_->load_fence_));

    // --- 4. Wait for GPU (绝对必须) ---
    {
        ScopedNvtxRange range("GPU_Readback_Wait", 0xFF808080);
        // 这一步之后，Staging Buffer 的内容在物理内存中已经是新的了
        waitForFenceInfinitely(dev, render_context_->load_fence_);
    }

    // --- 5. Zero-Copy 处理 (关键修改) ---
    frames.resize(config_.batch_size);

    {
        ScopedNvtxRange range("CPU_Invalidate_Cache", 0xFF4682B4);
        
        // A. 准备 Invalidate Ranges
        // 如果你的 staging buffer 是分开分配的内存，需要为每个 buffer 准备一个 range
        // 如果它们是大块内存分配出来的 sub-allocation，处理方式会有所不同，这里假设是独立的 Memory
        std::vector<VkMappedMemoryRange> ranges;
        ranges.reserve(config_.batch_size);

        for (int i = 0; i < config_.batch_size; ++i) {
            VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
            range.memory = staging_buffers_[i].getMemHdl(); // 必须有对应的 VkDeviceMemory 句柄
            range.offset = 0; // 或者 staging_buffers_[i].offset
            range.size = VK_WHOLE_SIZE; // 或者 staging_buffers_[i].size
            ranges.push_back(range);
        }

        // B. 执行 Cache Invalidation
        // 这一步非常快，它告诉 CPU：“不要信你的 L1/L2/L3 缓存了，去读主内存！”
        // 只有 HOST_COHERENT 内存不需要这一步，但 CACHED 内存通常不是 COHERENT 的。
        // 即使是 COHERENT，调用一下也不会错，且开销很小。
        if (!ranges.empty()) {
            REQ_VK(dev.dt.flushMappedMemoryRanges(dev.hdl, 0, nullptr)); // 通常用于写，这里我们需要 Invalidate
            REQ_VK(dev.dt.invalidateMappedMemoryRanges(dev.hdl, (uint32_t)ranges.size(), ranges.data()));
        }

        // C. 构造返回结果 (直接赋值指针)
        size_t pixel_size = 4; // RGBA8
        uint32_t width = (uint32_t)config_.frame_width;
        uint32_t height = (uint32_t)config_.frame_height;

        for (int i = 0; i < config_.batch_size; ++i) {
            frames[i].data = (const uint8_t*)staging_buffers_[i].ptr; // 原始映射指针
            frames[i].width = width;
            frames[i].height = height;
            // 假设 copy 时 bufferRowLength=0，则 stride 就是 width * pixel_size
            // 如果你有特定的对齐要求（如256字节对齐），这里要填对齐后的值
            frames[i].stride_bytes = width * pixel_size; 
            frames[i].total_bytes = width * height * pixel_size;
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
    uint32_t bindCount = 2;
    VkDescriptorSetLayoutBinding texbindings[] = {
        // Binding 0: 2D Textures Array
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = kMaxBindlessTextures, // 例如 1024
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = nullptr,
        },
        // Binding 1: Shared Sampler (Immutable)
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = &texture_sampler_
        }
    };

    // 2. Define Binding Flags (Partially Bound for arrays)
    VkDescriptorBindingFlags binding_flags[] = { 
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, // For Binding 0 (2D)
        0                                            // For Binding 1 (Sampler)
    };

    VkDescriptorSetLayoutBindingFlagsCreateInfo flag_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = bindCount,
        .pBindingFlags = binding_flags,
    };

    // 3. Create the layout
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = &flag_info;
    layout_info.bindingCount = bindCount;
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
    render_fence_ = makeFence(dev, false);

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
    const auto& loaded_resources = material_textures_;
    const auto& textures_2d = loaded_resources.textures_2d;

    std::vector<VkDescriptorImageInfo> image_infos_2d;
    std::vector<VkDescriptorImageInfo> image_infos_cube;
    
    image_infos_2d.reserve(textures_2d.size());

    // 2. 填充 2D Image Infos
    for (const auto& tex : textures_2d) {
        VkDescriptorImageInfo info = {};
        info.imageView = tex.view; // 这里的 view 必须是 VK_IMAGE_VIEW_TYPE_2D
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info.sampler = VK_NULL_HANDLE; // 使用 Immutable Sampler，这里填 null
        image_infos_2d.push_back(info);
    }


    // 3. 构建 Writes
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(textures_2d.size());

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

    // 5. 执行批量更新
    if (!writes.empty()) {
        dev.dt.updateDescriptorSets(dev.hdl, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    
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
    
    // Destroy textures
    for (auto &tx : material_textures_.textures_2d) {
        dev.dt.destroyImageView(dev.hdl, tx.view, nullptr);
        dev.dt.destroyImage(dev.hdl, tx.image.image, nullptr);
        dev.dt.freeMemory(dev.hdl, tx.backing, nullptr);
    }

    material_textures_.textures_2d.clear();

    dev.dt.destroySampler(dev.hdl, texture_sampler_, nullptr);

    // global vertex/index buffers
    global_vertex_buffer_.reset();
    global_index_buffer_.reset();
    // camera ubo
    camera_staging_buffers_.clear();
    camera_uniform_buffers_.clear();
    // light ubo
    light_staging_buffers_.clear();
    light_uniform_buffers_.clear();

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
    if(render_fence_ != VK_NULL_HANDLE) {
        dev.dt.destroyFence(dev.hdl, render_fence_, nullptr);
        render_fence_ = VK_NULL_HANDLE;
    }
    
    // Staging buffers are destroyed via HostBuffer destructors
    staging_buffers_.clear();
    
    LOG(config_, "DestroyVulkanResources(): resources released");
}

// --------------------------- Simple accessors --------------------------------
const unsigned char* BatchRenderer::GetRGBFrame(int batch_idx) const {
    // 1. 安全检查
    // 注意：这里检查 frames_.size() 比检查 config_.batch_size 更安全，
    // 因为它能确保 ReadbackResults 至少被调用过一次并填充了数据。
    if (batch_idx < 0 || batch_idx >= (int)frames.size()) {
        return nullptr;
    }
    // 2. 直接返回 Zero-Copy 指针
    // frames_[batch_idx].data 指向的是 Staging Buffer (系统内存)
    return frames[batch_idx].data;
}

// TODO: this func is now invalid because we do zero-copy readback
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

