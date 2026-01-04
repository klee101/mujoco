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

// Gribb-Hartmann Plane Extraction
// Planes are stored as (nx, ny, nz, d) where dot(n, p) + d > 0 is inside.
Frustum ExtractFrustum(const glm::mat4& viewProj) {
    Frustum f;
    const float* m = (const float*)&viewProj;

    // Left
    f.planes[0].x = m[3] + m[0];
    f.planes[0].y = m[7] + m[4];
    f.planes[0].z = m[11] + m[8];
    f.planes[0].w = m[15] + m[12];

    // Right
    f.planes[1].x = m[3] - m[0];
    f.planes[1].y = m[7] - m[4];
    f.planes[1].z = m[11] - m[8];
    f.planes[1].w = m[15] - m[12];

    // Bottom
    f.planes[2].x = m[3] + m[1];
    f.planes[2].y = m[7] + m[5];
    f.planes[2].z = m[11] + m[9];
    f.planes[2].w = m[15] + m[13];

    // Top
    f.planes[3].x = m[3] - m[1];
    f.planes[3].y = m[7] - m[5];
    f.planes[3].z = m[11] - m[9];
    f.planes[3].w = m[15] - m[13];

    // Near
    f.planes[4].x = m[3] + m[2];
    f.planes[4].y = m[7] + m[6];
    f.planes[4].z = m[11] + m[10];
    f.planes[4].w = m[15] + m[14];

    // Far
    f.planes[5].x = m[3] - m[2];
    f.planes[5].y = m[7] - m[6];
    f.planes[5].z = m[11] - m[10];
    f.planes[5].w = m[15] - m[14];

    // Normalize planes (optional but recommended for correct distance check)
    for (int i = 0; i < 6; i++) {
        float length = glm::length(glm::vec3(f.planes[i]));
        f.planes[i] /= length;
    }

    return f;
}

// Check if a World Space AABB is visible
bool IsAABBVisible(const Frustum& frustum, const glm::vec3& min_p, const glm::vec3& max_p) {
    // Check box against all 6 planes
    for (int i = 0; i < 6; i++) {
        // Find the point on the AABB furthest in the direction of the normal (n-vertex)
        // If this point is behind the plane, the whole box is outside.
        glm::vec3 p_n;
        p_n.x = (frustum.planes[i].x > 0) ? max_p.x : min_p.x;
        p_n.y = (frustum.planes[i].y > 0) ? max_p.y : min_p.y;
        p_n.z = (frustum.planes[i].z > 0) ? max_p.z : min_p.z;

        if (glm::dot(glm::vec3(frustum.planes[i]), p_n) + frustum.planes[i].w < 0) {
            return false; // Outside
        }
    }
    return true;
}

/*
* ResolveTextureFromMaterial - Given a material ID, find the associated texture ID
* NOTE: Utility function to map material to texture when get Scenes info from Robosuite
*/
int ResolveTextureFromMaterial(const mjModel* m, int matid) {
    if (matid < 0 || matid >= m->nmat) return -1;

    // Iterate through texture roles to find the Diffuse (RGB/RGBA) texture
    // MuJoCo defines roles: 0=RGB, 1=RGBA, 2=Spectral, 3=Normal...
    // We strictly look for color textures (RGB or RGBA).
    for (int role = 0; role < mjNTEXROLE; ++role) {
        int texid_idx = matid * mjNTEXROLE + role;
        
        // Safety check for array bounds
        if (m->mat_texid && texid_idx < m->nmat * mjNTEXROLE) {
            int texid = m->mat_texid[texid_idx];
            
            // If we found a valid texture ID
            if (texid >= 0 && texid < m->ntex) {
                // Priority: RGB (0) or RGBA (1)
                if (role == mjTEXROLE_RGB || role == mjTEXROLE_RGBA) {
                    return texid;
                }
            }
        }
    }
    return -1;
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

}

BatchRenderer::~BatchRenderer() {
    Cleanup();
}

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
        global_aabb_cache_ = std::move(other.global_aabb_cache_);
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

/*
* GetExtractTextures - Extract texture data from mjModel
* NOTE: now only support 2D textures, cube map will be treated as 2D with 6 times height
*/
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
        } else if (tex_name.empty()) {
            tex_name = "unnamed_tex_" + std::to_string(i);
        }
        tex_info.name = tex_name;

        if (tex_info.type == mjTEXTURE_CUBE || tex_info.type == mjTEXTURE_SKYBOX) {
            // for those builtin cubemap, transform it to standard 2d and process in shader
            if (tex_info.height == tex_info.width * 6) {
                tex_info.height = tex_info.height / 6;
            }
        }

        // Always convert to RGBA
        tex_info.channels = 4;
        int tex_data_adr = model->tex_adr[i];
        // Safety Check 1: Valid Address
        if (tex_data_adr < 0 || tex_data_adr >= model->ntexdata) {
            continue;
        }

        // Safety Check 2: Zero Dimensions
        if (tex_info.width == 0 || tex_info.height == 0) {
            continue;
        }

        // NOTE: for 1*6 CubeMap，only get the first face data heres
        size_t pixel_count = static_cast<size_t>(tex_info.width) * tex_info.height;
        size_t data_size = pixel_count * tex_info.channels; // always * 4
        
        // Safety Check 3: Data Size
        if (data_size == 0) {
             continue;
        }
        tex_info.data.resize(data_size);
        const unsigned char* tex_data = model->tex_data + tex_data_adr;
        
        // RGB/Gray -> RGBA
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
    }
    return tmpTextures_;
}

LoadedTextureResources BatchRenderer::LoadMaterialTextures()
{
    LoadedTextureResources result;
    std::vector<HostBuffer> host_buffers; 
    
    Device &dev = *device_;
    MemoryAllocator &alloc = render_context_->allocator;
    VkQueue queue = render_context_->renderQueue; 

    VkCommandPool tmp_pool = makeCmdPool(dev, dev.gfxQF);
    VkCommandBuffer cmdbuf = makeCmdBuffer(dev, tmp_pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
    
    VkCommandBufferBeginInfo begin_info = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    
    dev.dt.beginCommandBuffer(cmdbuf, &begin_info);

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

            // check cache
            if (global_tex_cache.find(unique_name) != global_tex_cache.end()) {
                int cached_idx = global_tex_cache[unique_name];
                
                // Mapping Type 0 (2D), Index = cached_idx
                TextureMapping mapping = { 0, cached_idx }; 
                result.global_texture_lookup.push_back(mapping);
                
                continue; 
            }
            // 3. Create Staging Buffer & Upload Data
            VkDeviceSize staging_size = tx.data.size(); 
            uint32_t width = tx.width;
            uint32_t height = tx.height;

            if (width == 0 || height == 0 || staging_size == 0) {
                
                // Push a placeholder/invalid mapping to maintain index alignment if needed, 
                // OR just skip. Skipping usually safer but might shift indices if logic relies on i.
                // Here we skip adding to 'textures_2d', but we MUST handle the lookup index.
                
                // Option A: Point to a fallback texture (e.g. index 0 if it exists) or -1
                TextureMapping mapping = { -1, -1 }; 
                result.global_texture_lookup.push_back(mapping);
                continue; 
            }


            LocalTexture texture;
            TextureRequirements texture_reqs;

            auto res = alloc.makeTexture2D(width, height, 1, VK_FORMAT_R8G8B8A8_SRGB);
            texture = res.first;
            texture_reqs = res.second;

            
            HostBuffer texture_hb_staging = alloc.makeStagingBuffer(staging_size);
            memcpy(texture_hb_staging.ptr, tx.data.data(), staging_size);
            texture_hb_staging.flush(dev);

            // 4. Allocate & Bind Image Memory
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

            // [MODIFIED] 6. copy Buffer -> Image
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

            int new_idx = (int)result.textures_2d.size();
            
            result.textures_2d.emplace_back(std::move(mat_tex));
            TextureMapping mapping;
            if(tx.type == mjTEXTURE_CUBE || tx.type == mjTEXTURE_SKYBOX) {
                mapping = { 1, new_idx };
            }
            else {
                mapping = { 0, new_idx };
            }

            result.global_texture_lookup.push_back(mapping);
            
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

void BatchRenderer::InitGlobalGeometry() {
    LOG(config_, "InitGlobalGeometry(): Starting mesh deduplication and upload...");

    std::vector<Vertex> global_vertices;
    std::vector<uint32_t> global_indices;

    //  here assume each mesh has about 10k vertices/indices
    size_t estimated_meshes = 0;
    for(auto model_ : models_) {
        estimated_meshes += (model_ ? model_->nmesh : 0) + 10; // +10 for primitives
    }
    global_vertices.reserve(estimated_meshes * 10000);
    global_indices.reserve(estimated_meshes * 10000); 

    // ---  Lambda: cache the geometry ---
    // generator: return GeometryBuffers (vertex + index)
    auto ProcessGeometry = [&](const std::string& unique_name, std::function<GeometryAABB()> generator) {
        // Deduplication Check: 如果名字已存在，直接跳过
        if (global_mesh_cache_.find(unique_name) != global_mesh_cache_.end()) {
            return; 
        }

        GeometryBuffer buffers = generator().buffers;
        AABB aabb = generator().aabb;

        if (buffers.vertices.empty()) {
            LOG(config_, "Warning: Empty geometry generated for " + unique_name);
            return;
        }

        MeshEntry entry;
        entry.vertex_offset = static_cast<uint32_t>(global_vertices.size());
        entry.index_offset  = static_cast<uint32_t>(global_indices.size());
        entry.vertex_count  = static_cast<uint32_t>(buffers.vertices.size());
        entry.index_count   = static_cast<uint32_t>(buffers.indices.size());

        global_vertices.insert(global_vertices.end(), buffers.vertices.begin(), buffers.vertices.end());
        
        // here store the relative indices
        global_indices.insert(global_indices.end(), buffers.indices.begin(), buffers.indices.end());

        global_mesh_cache_[unique_name] = entry;
        global_aabb_cache_[unique_name] = aabb;

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

    // 3. process Built-in Primitives 
    // here set all builtin geom the same size params 
    ProcessGeometry("__builtin_box",      [](){ return GeometryBuilder::BuildBox(24); });
    ProcessGeometry("__builtin_sphere",   [](){ return GeometryBuilder::BuildSphere(16, 16); }); // 16 stacks/slices
    ProcessGeometry("__builtin_capsule",  [](){ return GeometryBuilder::BuildCapsule(16, 16); });
    ProcessGeometry("__builtin_cylinder", [](){ return GeometryBuilder::BuildCylinder(16, 16); });
    ProcessGeometry("__builtin_ellipsoid", [](){ return GeometryBuilder::BuildSphere(16, 16); }); // Reuse sphere
    ProcessGeometry("__builtin_plane",    [](){ return GeometryBuilder::BuildPlane(10); }); // Simple quad

    if (global_vertices.empty() || global_indices.empty()) {
        LOG(config_, "InitGlobalGeometry(): No geometry to upload.");
        return;
    }

    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;
    VkCommandBuffer cmd = render_context_->load_cmd_;

    VkDeviceSize v_size = global_vertices.size() * sizeof(Vertex);
    VkDeviceSize i_size = global_indices.size() * sizeof(uint32_t);

    auto v_staging = allocator.makeStagingBuffer(v_size);
    auto i_staging = allocator.makeStagingBuffer(i_size);

    std::memcpy(v_staging.ptr, global_vertices.data(), v_size);
    std::memcpy(i_staging.ptr, global_indices.data(), i_size);
    
    v_staging.flush(dev);
    i_staging.flush(dev);

    // Create VertexBuffer and IndexBuffer on GPU (Device Local)
    auto v_local = allocator.makeLocalBuffer(v_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto i_local = allocator.makeLocalBuffer(i_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    global_vertex_buffer_ = std::move(*v_local);
    global_index_buffer_ = std::move(*i_local);

    // record Copy Commands
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

    // ensure Copy is finished before using as vertex/index buffer
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

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    resetFence(dev, render_context_->load_fence_);

    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, render_context_->load_fence_));

    waitForFenceInfinitely(dev, render_context_->load_fence_);
    
}

bool BatchRenderer::Initialize() {
    LOG(config_, "Initialize(): starting");

    if (models_.empty()) {
        LOG(config_, "Initialize(): models_ is null");
        return false;
    }

    // Create Backend -> Device -> RenderContext
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

    InitGlobalGeometry();

    // Create buffers (Uniform, Command, etc.) 
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

    camera_cull_info_.resize(config_.batch_size * SHM_NUM_CAMERAS);

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

    initialized_ = false;
    LOG(config_, "Cleanup(): done");
}

RenderResult BatchRenderer::Render(mjData** data_array, const int* camera_ids) {
    if (!initialized_) return MakeError(RenderError::INVALID_CONFIG, "Renderer not initialized");
    if (!data_array) return MakeError(RenderError::INVALID_DATA, "data_array is null");

    const int count = config_.batch_size;

    auto t0 = std::chrono::high_resolution_clock::now();

// -------------------------------------------------------
    // Phase 1: Update Scenes (CPU work)
    // TODO: multi-threading
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("1. Update_Scenes (CPU)", C_UPDATE);
        if (!UpdateScenes(data_array, count)) {
            return MakeError(RenderError::MUJOCO_ERROR, "UpdateScenes failed");
        }
    }

    // -------------------------------------------------------
    // Phase 2: Record Command Buffers (CPU/Driver)
    // TODO: check the pushConstants performance impact
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("2. Record_Cmds (Driver)", C_RECORD);
        if (!RecordCommandBuffers(count)) {
            return MakeError(RenderError::VULKAN_ERROR, "RecordCommandBuffers failed");
        }
    }

    // -------------------------------------------------------
    // Phase 3: Submit & Wait (CPU block / GPU render)
    // 这里的耗时代表：vkQueueSubmit + vkWaitForFences
    // -------------------------------------------------------
    {
        ScopedNvtxRange range("3. Submit_Wait (GPU)", C_SUBMIT);
        if (!SubmitAndWait()) {
            return MakeError(RenderError::VULKAN_ERROR, "SubmitAndWait failed");
        }
    }

    // -------------------------------------------------------
    // Phase 4: Readback (PCIe 传输)
    // vkCmdCopyImageToBuffer (GPU -> CPU)
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

bool BatchRenderer::UpdateScenes(mjData** data_array, int count) {
    Device &dev = *device_;

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
        mjv_updateScene(models_[i], data_array[i], &res.options, nullptr, 
                    &res.camera, mjCAT_ALL, &res.scene);
        
        // First time: create scene with geometry
        if (!res.render_scene) {
            res.render_scene = std::make_unique<mujoco::mjbatch::Scene>(models_[i], &res.scene);
        } else {
            // Update the transforms 
             res.render_scene->Update(models_[i], &res.scene);
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
        LightUBO light_ubo{};
        size_t light_count = std::min(lights.size(), size_t(10));
        light_ubo.lightCount = static_cast<uint32_t>(light_count);
        std::memcpy(light_staging_buffers_[i].ptr, &light_ubo, sizeof(LightUBO));
        light_staging_buffers_[i].flush(dev);
        copy_ops.push_back({light_staging_buffers_[i].buffer, 
                           light_uniform_buffers_[i].buffer, 
                           sizeof(LightUBO)});
        printf("Env %d: Updated %zu lights\n", i, light_count);
        // print the detailed light info
        for (size_t j = 0; j < light_count; ++j) {
            const LightInfo& light = lights[j];
            printf("  Light %zu: pos=(%.2f, %.2f, %.2f), diffuse=(%.2f, %.2f, %.2f), intensity=%.2f\n",
                   j,
                   light.position[0], light.position[1], light.position[2],
                   light.diffuse[0], light.diffuse[1], light.diffuse[2],
                   light.intensity);
        }

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
        // change: one time submit for better performance
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
        
        VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassInfo.renderPass = render_context_->renderPass;
        renderPassInfo.framebuffer = framebuffers_[i];
        renderPassInfo.renderArea.extent = {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height};
        
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; 
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

RenderResult BatchRenderer::RenderFromMemory(const uint8_t* shared_memory_ptr, int batch_idx, int max_geom, int max_light) {
    if (!initialized_) return MakeError(RenderError::INVALID_CONFIG, "Renderer not initialized");
    
    // We ignore 'batch_idx' here assuming the ptr covers the whole batch 
    // OR the logic inside UpdateScenes handles offsets. 
    // Assuming shared_memory_ptr is the BASE of the shm block.

    // 1. Phase 1: Update Uniform Buffers (Camera/Light) directly from SHM
    {
        ScopedNvtxRange range("1. Update_From_SHM", C_UPDATE);
        if (!UpdateScenesFromMemory(shared_memory_ptr, 0, max_geom, max_light)) {
             return MakeError(RenderError::MUJOCO_ERROR, "UpdateScenesFromMemory failed");
        }
    }

    // 2. Phase 2: Record Commands (Iterating SHM Geoms directly)
    {
        ScopedNvtxRange range("2. Record_Cmds", C_RECORD);
        // We pass the pointer down so Record functions can read the geoms
        if (!RecordCommandBuffersFromMemory(shared_memory_ptr, config_.batch_size)) { 
             return MakeError(RenderError::VULKAN_ERROR, "RecordCommandBuffers failed");
        }
    }

    // 3. Phase 3: Submit (Reused)
    {
        ScopedNvtxRange range("3. Submit_Wait", C_SUBMIT);
        if (!SubmitAndWait()) {
             return MakeError(RenderError::VULKAN_ERROR, "SubmitAndWait failed");
        }
    }

    // 4. Phase 4: Readback (Reused)
    {
        ScopedNvtxRange range("4. Readback", C_READ);
        if (!ReadbackResults()) {
             return MakeError(RenderError::VULKAN_ERROR, "ReadbackResults failed");
        }
    }
    
    return RenderResult();
}

bool BatchRenderer::UpdateScenesFromMemory(const uint8_t* ptr, int start_idx, int max_geom, int max_light) {
    Device &dev = *device_;
    const EnvRenderSlot* slots = reinterpret_cast<const EnvRenderSlot*>(ptr);

    // We only update UBOs here (Camera & Light). 
    // Geometry is read directly in the Record phase.

    std::vector<VkBufferCopy> copy_regions; 
    // Using a separate staging loop might be cleaner, but reusing existing infrastructure:
    
    VkCommandBuffer cmd = render_context_->load_cmd_;
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));

    for(int i = 0; i < config_.batch_size; ++i) {
        const EnvRenderSlot& slot = slots[i];
        
        for (int c = 0; c < SHM_NUM_CAMERAS; ++c) {
            int resource_idx = i * SHM_NUM_CAMERAS  + c;
            // --- 1. Update Camera UBO --
            const ShmCamera& src_cam = slot.cameras[c]; 

            // Convert Row-Major to GLM Mat4
            glm::mat4 view = RowMajorToGLM(src_cam.view);
            glm::mat4 proj = RowMajorToGLM(src_cam.proj);

            glm::mat4 view_proj = proj * view;
            glm::vec3 cam_pos = glm::vec3(src_cam.pos[0], src_cam.pos[1], src_cam.pos[2]);

            // --- SAVE INFO FOR CULLING ---
            camera_cull_info_[resource_idx].view_proj = view_proj;
            camera_cull_info_[resource_idx].pos = cam_pos;
            camera_cull_info_[resource_idx].frustum = ExtractFrustum(view_proj);

            CameraUBO cam_ubo{};
            cam_ubo.view_proj = view_proj;
            cam_ubo.position = cam_pos;

            std::memcpy(camera_staging_buffers_[resource_idx].ptr, &cam_ubo, sizeof(CameraUBO));
            camera_staging_buffers_[resource_idx].flush(dev);

            VkBufferCopy camCopy{};
            camCopy.size = sizeof(CameraUBO);
            dev.dt.cmdCopyBuffer(cmd, camera_staging_buffers_[resource_idx].buffer, camera_uniform_buffers_[resource_idx].buffer, 1, &camCopy);


            // --- 2. Update Light UBO (Headlamp Mode) ---
            // Since SHM has no lights, we create a light at the camera position
            LightUBO light_ubo{};
            light_ubo.lightCount = 1;
            
            // Light 0
            light_ubo.lights[0].position[0] = src_cam.pos[0];
            light_ubo.lights[0].position[1] = src_cam.pos[1];
            light_ubo.lights[0].position[2] = src_cam.pos[2];
            
            light_ubo.lights[0].diffuse[0] = 0.8f; 
            light_ubo.lights[0].diffuse[1] = 0.8f; 
            light_ubo.lights[0].diffuse[2] = 0.8f;
            
            light_ubo.lights[0].specular[0] = 0.5f; 
            light_ubo.lights[0].specular[1] = 0.5f; 
            light_ubo.lights[0].specular[2] = 0.5f;
            
            light_ubo.lights[0].attenuation[0] = 1.0f; // Constant
            light_ubo.lights[0].attenuation[1] = 0.0f; // Linear
            light_ubo.lights[0].attenuation[2] = 0.0f; // Quadratic

            std::memcpy(light_staging_buffers_[resource_idx].ptr, &light_ubo, sizeof(LightUBO));
            light_staging_buffers_[resource_idx].flush(dev);

            VkBufferCopy lightCopy{};
            lightCopy.size = sizeof(LightUBO);
            dev.dt.cmdCopyBuffer(cmd, light_staging_buffers_[resource_idx].buffer, light_uniform_buffers_[resource_idx].buffer, 1, &lightCopy);
        }
    }

    REQ_VK(dev.dt.endCommandBuffer(cmd));
    
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    
    // Use renderQueue to submit memory transfers (simple sync)
    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, VK_NULL_HANDLE));
    dev.dt.deviceWaitIdle(dev.hdl); 
    
    return true;
}

bool BatchRenderer::RecordCommandBuffersFromMemory(const uint8_t* ptr, int count) {
    Device &dev = *device_;
    const EnvRenderSlot* slots = reinterpret_cast<const EnvRenderSlot*>(ptr);
    
    // --- STATISTICS COUNTERS ---
    int total_instances = 0;
    int drawn_instances = 0;
    int culled_instances = 0;
    // ---------------------------

    for (int i = 0; i < count; ++i) {

        const EnvRenderSlot& slot = slots[i];
        for (int c = 0; c < SHM_NUM_CAMERAS; ++c) {
            int resource_idx = i * SHM_NUM_CAMERAS + c;

            // Retrieve Culling Info
            const auto& cull_info = camera_cull_info_[resource_idx];

            VkCommandBuffer cmd = command_buffers_[resource_idx];
            // --- 1. Begin Recording & Render Pass ---
            VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
            
            VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            renderPassInfo.renderPass = render_context_->renderPass;
            renderPassInfo.framebuffer = framebuffers_[resource_idx];
            renderPassInfo.renderArea.extent = {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height};
            
            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; 
            clearValues[1].depthStencil = {1.0f, 0};
            renderPassInfo.clearValueCount = clearValues.size();
            renderPassInfo.pClearValues = clearValues.data();
            
            dev.dt.cmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            
            // --- 2. Bind Pipeline & Global State ---
            dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
            
            VkViewport viewport = {0.0f, 0.0f, (float)config_.frame_width, (float)config_.frame_height, 0.0f, 1.0f};
            dev.dt.cmdSetViewport(cmd, 0, 1, &viewport);
            VkRect2D scissor = {{0, 0}, {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height}};
            dev.dt.cmdSetScissor(cmd, 0, 1, &scissor);

            // Bind Descriptor Sets (UBOs + Textures)
            std::vector<VkDescriptorSet> sets = { descriptor_sets_[resource_idx], global_texture_descriptor_set };
            dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 
                                         0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);

            // --- 3. Bind Global Vertex/Index Buffers ---
            if (global_vertex_buffer_->buffer != VK_NULL_HANDLE) {
                VkBuffer vbs[] = { global_vertex_buffer_->buffer };
                VkDeviceSize offsets[] = { 0 };
                vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                vkCmdBindIndexBuffer(cmd, global_index_buffer_->buffer, 0, VK_INDEX_TYPE_UINT32);

                // --- 4. Iteration over SHM Geoms ---
                int32_t active_geoms = std::min(slot.num_geoms, (int32_t)SHM_MAX_GEOMS);
                
                for (int g = 0; g < active_geoms; ++g) {
                    const ShmGeom& geom = slot.geoms[g];

                    total_instances++;
                    
                    // [A] Determine Mesh Name from Type/DataID
                    std::string mesh_name;
                    if (geom.type == 0) { // mjGEOM_PLANE
                        mesh_name = "__builtin_plane";
                    } else if (geom.type == 2) { // mjGEOM_SPHERE
                        mesh_name = "__builtin_sphere";
                    } else if (geom.type == 3) { // mjGEOM_CAPSULE
                        mesh_name = "__builtin_capsule";
                    } else if (geom.type == 4) { // mjGEOM_ELLIPSOID
                        mesh_name = "__builtin_ellipsoid"; 
                    } else if (geom.type == 5) { // mjGEOM_CYLINDER
                        mesh_name = "__builtin_cylinder";
                    } else if (geom.type == 6) { // mjGEOM_BOX
                            mesh_name = "__builtin_box";
                    } else if (geom.type == 7) { // mjGEOM_MESH
                        mesh_name = (models_[i]->names) ? 
                            std::string(models_[i]->names + models_[i]->name_meshadr[geom.dataid/2]) : 
                            "mesh_" + std::to_string(geom.dataid);
                    } else {
                        continue; // Skip unsupported geoms
                    }

                    // [B] Lookup in Cache
                    auto it = global_mesh_cache_.find(mesh_name);
                    if (it == global_mesh_cache_.end()) continue;
                    const MeshEntry& entry = it->second;

                    auto it2 = global_aabb_cache_.find(mesh_name);
                    if (it2 == global_aabb_cache_.end()) continue;
                    const AABB& local_aabb = it2->second; // This is the AABB in Local Space (cached)


                    // [C] Construct Model Matrix
                    // SHM provides 3x3 Rotation (row-major 9 floats) and Pos (3 floats)
                    glm::mat4 model_mat(1.0f);
                    
                    // Copy rotation (converting Row-Major SHM to Column-Major GLM)
                    // geom.mat is [r00, r01, r02, r10, r11, r12, r20, r21, r22]
                    model_mat[0][0] = geom.mat[0]; model_mat[1][0] = geom.mat[1]; model_mat[2][0] = geom.mat[2];
                    model_mat[0][1] = geom.mat[3]; model_mat[1][1] = geom.mat[4]; model_mat[2][1] = geom.mat[5];
                    model_mat[0][2] = geom.mat[6]; model_mat[1][2] = geom.mat[7]; model_mat[2][2] = geom.mat[8];

                    // Set position
                    model_mat[3][0] = geom.pos[0];
                    model_mat[3][1] = geom.pos[1];
                    model_mat[3][2] = geom.pos[2];

                    // Apply Scale based on Type
                    // Primitives are usually unit-sized in cache, so we scale them.
                    if (geom.type == 7) { // Mesh
                        // Meshes are usually pre-baked or scale is Identity
                        // If MuJoCo resizes mesh, apply scale here.
                        // Typically 'geom.size' is BBox, not transform scale for meshes.
                    } else if (geom.type == 0) { // Plane
                        // Scale x/y by size[0], size[1]
                        float sx = (geom.size[0] > 0) ? geom.size[0] : 1000.0f;
                        float sy = (geom.size[1] > 0) ? geom.size[1] : 1000.0f;
                        model_mat = glm::scale(model_mat, glm::vec3(sx, sy, 1.0f));
                        
                        // Z scale 1 for plane
                    } else if (geom.type == 2) { // Sphere
                        model_mat = glm::scale(model_mat, glm::vec3(geom.size[0]));
                    } else {
                        // Box, etc: Full 3D scale
                        model_mat = glm::scale(model_mat, glm::vec3(geom.size[0], geom.size[1], geom.size[2]));
                    }

                    // --- CULLING LOGIC ---
                    
                    // 1. Calculate World Space AABB
                    // Transform Center
                    glm::vec3 local_center = (glm::vec3(local_aabb.min.x, local_aabb.min.y, local_aabb.min.z) + 
                                              glm::vec3(local_aabb.max.x, local_aabb.max.y, local_aabb.max.z)) * 0.5f;
                    glm::vec3 local_extent = (glm::vec3(local_aabb.max.x, local_aabb.max.y, local_aabb.max.z) - 
                                              glm::vec3(local_aabb.min.x, local_aabb.min.y, local_aabb.min.z)) * 0.5f;
                                              
                    glm::vec3 world_center = glm::vec3(model_mat * glm::vec4(local_center, 1.0f));
                    
                    // Transform Extent (using absolute rotation matrix to bound the rotated box)
                    glm::vec3 world_extent;
                    for (int k = 0; k < 3; k++) {
                        world_extent[k] = 
                            std::abs(model_mat[0][k]) * local_extent.x +
                            std::abs(model_mat[1][k]) * local_extent.y +
                            std::abs(model_mat[2][k]) * local_extent.z;
                    }

                    glm::vec3 world_min = world_center - world_extent;
                    glm::vec3 world_max = world_center + world_extent;

                    // 2. Perform Frustum Check
                    // 
                    if (!IsAABBVisible(cull_info.frustum, world_min, world_max)) {
                        culled_instances++; // Log: It was culled
                        continue; // SKIP DRAW CALL
                    }
                    // ---------------------

                    // If we get here, it will be drawn
                    drawn_instances++;
                    
                    // [D] Push Constants
                    PushConstants pc{};
                    pc.model = model_mat;
                    pc.rgba = glm::vec4(geom.rgba[0], geom.rgba[1], geom.rgba[2], geom.rgba[3]);
                    pc.specular = geom.specular;
                    pc.emission = geom.emission;
                    pc.shininess = geom.shininess;
                    pc.reflectance = geom.reflectance;

                    // [MODIFIED] Map matid -> texid -> global lookup
                    int resolved_tex_id = -1;
                    
                    // 1. Resolve matid -> texid
                    if (geom.matid >= 0) {
                        resolved_tex_id = ResolveTextureFromMaterial(models_[i], geom.matid);
                    }
                    // 2. Global Lookup
                    if (resolved_tex_id >= 0 && i < texture_offsets_.size()) {
                        int global_tex_id = texture_offsets_[i] + resolved_tex_id;
                        
                        if (global_tex_id < material_textures_.global_texture_lookup.size()) {
                            const auto& tex_map = material_textures_.global_texture_lookup[global_tex_id];
                            
                            // Valid texture found?
                            if (tex_map.index_in_array >= 0) {
                                pc.texture_type = tex_map.type;
                                pc.texture_index = tex_map.index_in_array;
                            } else {
                                // Fallback if texture failed to load (index -1)
                                pc.texture_type = -1;
                                pc.texture_index = -1;
                            }
                        } else {
                            pc.texture_type = -1;
                            pc.texture_index = -1;
                        }
                    } else {
                        pc.texture_type = -1;
                        pc.texture_index = -1;
                    }

                    dev.dt.cmdPushConstants(cmd, pipeline_layout_, 
                                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                        0, sizeof(PushConstants), &pc);

                    vkCmdDrawIndexed(cmd, entry.index_count, 1, entry.index_offset, entry.vertex_offset, 0);
                }
            }

            dev.dt.cmdEndRenderPass(cmd);
            REQ_VK(dev.dt.endCommandBuffer(cmd));
        }
    }

    // --- PRINT STATISTICS ---
    // NOTE: This will print every time Record is called. For high FPS, consider wrapping this in a timer.
    if (total_instances > 0) {
        float cull_percentage = (static_cast<float>(culled_instances) / total_instances) * 100.0f;
        std::cout << "[BatchRenderer] Culling Stats: "
                  << "Total: " << total_instances << " | "
                  << "Drawn: " << drawn_instances << " | "
                  << "Culled: " << culled_instances << " ("
                  << cull_percentage << "%)" << std::endl;
    }
    // ------------------------
    
    return true;
}

bool BatchRenderer::SubmitAndWait() {
    Device &dev = *device_;
    VkQueue queue = render_context_->renderQueue;
    const uint64_t TIMEOUT_NS = 10000000000ULL; // 10s timeout 
    // --- 1. reset Fence ---
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &render_fence_));

    // --- 2. prepare Batching submitinfo ---
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.commandBufferCount = (uint32_t)command_buffers_.size();
    submitInfo.pCommandBuffers = command_buffers_.data(); 
    

    // --- 3. One Submit ---
    VkResult submitResult = dev.dt.queueSubmit(queue, 1, &submitInfo, render_fence_);
    
    if (submitResult != VK_SUCCESS) {
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), "ERROR: Batch queueSubmit failed: %d", submitResult);
        LOG(config_, log_buf);
        return false;
    }

    // --- 4. One Wait ---
    {
        ScopedNvtxRange range("CPU_Wait_Fence", 0xFF800000); 
        
        VkResult waitResult = dev.dt.waitForFences(
            dev.hdl,
            1,
            &render_fence_,
            VK_TRUE, 
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

    size_t total_slots = config_.batch_size * SHM_NUM_CAMERAS;
    
    // --- 1. Begin Recording ---
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));

    // --- 2. Record Commands for ALL batches ---
    for (int i = 0; i < total_slots; ++i) {
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

    // --- 4. Wait for GPU  ---
    {
        ScopedNvtxRange range("GPU_Readback_Wait", 0xFF808080);
        waitForFenceInfinitely(dev, render_context_->load_fence_);
    }

    // --- 5. Zero-Copy  ---
    frames.resize(total_slots);

    {
        ScopedNvtxRange range("CPU_Invalidate_Cache", 0xFF4682B4);
        
        // prepare Invalidate Ranges
        std::vector<VkMappedMemoryRange> ranges;
        ranges.reserve(total_slots);

        for (int i = 0; i < total_slots; ++i) {
            VkMappedMemoryRange range = { VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE };
            range.memory = staging_buffers_[i].getMemHdl(); 
            range.offset = 0; 
            range.size = VK_WHOLE_SIZE; 
            ranges.push_back(range);
        }

        // B. Cache Invalidation
        if (!ranges.empty()) {
            REQ_VK(dev.dt.invalidateMappedMemoryRanges(dev.hdl, (uint32_t)ranges.size(), ranges.data()));
        }

        // directly point frame data to staging buffer memory
        size_t pixel_size = 4; // RGBA8
        uint32_t width = (uint32_t)config_.frame_width;
        uint32_t height = (uint32_t)config_.frame_height;

        for (int i = 0; i < total_slots; ++i) {
            frames[i].data = (const uint8_t*)staging_buffers_[i].ptr;
            frames[i].width = width;
            frames[i].height = height;
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
    std::string shader_dir = "/home/hpf/project/vulkan/mujoco/mujoco/build/shaders_spv/";  // This should come from build config
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

    // Camera and Light descriptor set layout
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
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstants); // view-projection matrix (mat4)
    
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

    size_t total_slots = config_.batch_size * SHM_NUM_CAMERAS;
    
    framebuffers_.reserve(total_slots);
    color_images_.reserve(total_slots);
    depth_images_.reserve(total_slots);
    color_image_views_.reserve(total_slots);
    depth_image_views_.reserve(total_slots);
    
    for (int i = 0; i < total_slots; ++i) {
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

bool BatchRenderer::CreateBuffers() {
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;

    size_t total_slots = config_.batch_size * SHM_NUM_CAMERAS;
    
    // Create command pool
    command_pool_ = makeCmdPool(dev, dev.gfxQF);
    
    // Allocate command buffers
    command_buffers_.resize(total_slots);
    VkCommandBufferAllocateInfo allocCmdInfo{};
    allocCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocCmdInfo.commandPool = command_pool_;
    allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocCmdInfo.commandBufferCount = command_buffers_.size();
    REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &allocCmdInfo, command_buffers_.data()));
    
    // Create fences
    render_fence_ = makeFence(dev, false);
    {
        // Create descriptor pool
        std::array<VkDescriptorPoolSize, 2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = total_slots; // Camera UBOs
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = total_slots; // Light UBOs
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = total_slots;
        REQ_VK(dev.dt.createDescriptorPool(dev.hdl, &poolInfo, nullptr, &descriptor_pool_));
        
        // Allocate descriptor sets
        descriptor_sets_.resize(total_slots);
        std::vector<VkDescriptorSetLayout> layouts(total_slots, descriptor_set_layout_);
        VkDescriptorSetAllocateInfo allocDesInfo{};
        allocDesInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocDesInfo.descriptorPool = descriptor_pool_;
        allocDesInfo.descriptorSetCount = total_slots;
        allocDesInfo.pSetLayouts = layouts.data();
        REQ_VK(dev.dt.allocateDescriptorSets(dev.hdl, &allocDesInfo, descriptor_sets_.data()));
    }
    {
        // Create global texture descriptor set for bindless textures
        // here use a independent pool and set for global textures
        // 1. Define the pool size (count needs to cover all descriptors)
        std::array<VkDescriptorPoolSize, 2> pool_sizes{};
        // Binding 0: texture 2d array
        pool_sizes[0].type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; 
        pool_sizes[0].descriptorCount = kMaxBindlessTextures;  

        // Binding 1: sampler
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
    for (int i = 0; i < total_slots; ++i) {
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
        
        dev.dt.updateDescriptorSets(dev.hdl, 2, writes, 0, nullptr);
    }

    // update global texture descriptor set for bindless textures
    const auto& loaded_resources = material_textures_;
    const auto& textures_2d = loaded_resources.textures_2d;

    std::vector<VkDescriptorImageInfo> image_infos_2d;
    std::vector<VkDescriptorImageInfo> image_infos_cube;
    
    image_infos_2d.reserve(textures_2d.size());

    // fill 2D Image Infos
    for (const auto& tex : textures_2d) {
        VkDescriptorImageInfo info = {};
        info.imageView = tex.view; 
        info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        info.sampler = VK_NULL_HANDLE; // when use Immutable Sampler，fill null
        image_infos_2d.push_back(info);
    }


    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(textures_2d.size());

    // generate 2D texture  Writes (Binding 0)
    for (size_t i = 0; i < textures_2d.size(); ++i) {
        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet = global_texture_descriptor_set;
        write.dstBinding = 0; // Binding 0
        write.dstArrayElement = static_cast<uint32_t>(i);
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        write.pImageInfo = &image_infos_2d[i]; 

        writes.push_back(write);
    }

    // 5. update the texture
    if (!writes.empty()) {
        dev.dt.updateDescriptorSets(dev.hdl, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    
    // Create staging buffers for readback
    // BUGFIX: with the zero-copy logic, this staging buffer need to be changed
    staging_buffers_.clear();
    staging_buffers_.reserve(total_slots);
    for (size_t i = 0; i < total_slots; ++i) {
        size_t bufferSize = config_.frame_width * config_.frame_height * 4;
        staging_buffers_.emplace_back(
            render_context_->allocator.makeStagingBuffer2(bufferSize)
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

    global_vertex_buffer_.reset();
    global_index_buffer_.reset();

    camera_staging_buffers_.clear();
    camera_uniform_buffers_.clear();

    light_staging_buffers_.clear();
    light_uniform_buffers_.clear();

    // Add these to DestroyVulkanResources:
    if (global_texture_set_layout != VK_NULL_HANDLE) {
        dev.dt.destroyDescriptorSetLayout(dev.hdl, global_texture_set_layout, nullptr);
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        dev.dt.destroyDescriptorSetLayout(dev.hdl, descriptor_set_layout_, nullptr);
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        dev.dt.destroyDescriptorPool(dev.hdl, descriptor_pool_, nullptr);
    }
    if (global_texture_descriptor_pool != VK_NULL_HANDLE) {
        dev.dt.destroyDescriptorPool(dev.hdl, global_texture_descriptor_pool, nullptr);
    }


    if (vert_shader_module_ != VK_NULL_HANDLE) {
        dev.dt.destroyShaderModule(dev.hdl, vert_shader_module_, nullptr);
        vert_shader_module_ = VK_NULL_HANDLE;
    }
    if (frag_shader_module_ != VK_NULL_HANDLE) {
        dev.dt.destroyShaderModule(dev.hdl, frag_shader_module_, nullptr);
        frag_shader_module_ = VK_NULL_HANDLE;
    }
    

    if (graphics_pipeline_ != VK_NULL_HANDLE) {
        dev.dt.destroyPipeline(dev.hdl, graphics_pipeline_, nullptr);
        graphics_pipeline_ = VK_NULL_HANDLE;
    }
    

    if (pipeline_layout_ != VK_NULL_HANDLE) {
        dev.dt.destroyPipelineLayout(dev.hdl, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    
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

const unsigned char* BatchRenderer::GetRGBFrame(int batch_idx) const {
    if (batch_idx < 0 || batch_idx >= (int)frames.size()) {
        return nullptr;
    }
    return frames[batch_idx].data;
}

const float* BatchRenderer::GetDepthFrame(int batch_idx) const {
    return nullptr;
}




