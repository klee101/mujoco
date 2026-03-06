// renderer.cc
// Vulkan Batch Renderer Implementation

#include "renderer.h"
#include "profiler.h"
#include "vkutils.h"
#include "backend.h"
#include "scene.h"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <filesystem>
#include <unistd.h>
#include <limits.h>
#include "shared_protocol.h"
#include <dlfcn.h>

using namespace mujoco::mjbatch;

namespace {

inline void DefaultLog(const std::string &s) {
    fprintf(stderr, "[BatchRenderer] %s\n", s.c_str());
}

} // namespace



std::filesystem::path getLibraryDir()
{
    Dl_info info;
    if (dladdr((void*)&getLibraryDir, &info) && info.dli_fname)
    {
        return std::filesystem::path(info.dli_fname).parent_path();
    }
    return {};
}

std::filesystem::path getAssetsDir() {
    auto libDir = getLibraryDir();
    if (libDir.empty()) return {};
    // LibraryDir -> build/lib -> ../.. -> mujoco root -> mjbatch/assets
    return libDir.parent_path().parent_path() / "mjbatch" / "assets";
}


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

void BatchRenderer::InitRenderDoc() {
    // 1. 先尝试检查是否已经注入 (NOLOAD)
    void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    
    // 2. 如果没有注入，尝试硬加载 (去掉 NOLOAD)
    if (!mod) {
        mod = dlopen("librenderdoc.so", RTLD_NOW);
    }

    if (mod) {
        pRENDERDOC_GetAPI R_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        if (R_GetAPI && R_GetAPI(eRENDERDOC_API_Version_1_4_1, (void **)&rdoc_api_) == 1) {
            // 成功加载
            rdoc_api_->SetCaptureFilePathTemplate("./mjb_capture");
            printf("[BatchRenderer] RenderDoc API connected.\n");
        }
    } else {
        // 打印具体错误原因，这非常重要！
        printf("[BatchRenderer] RenderDoc NOT detected. dlerror: %s\n", dlerror());
    }
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


void PrintMatrix(const char* name, const glm::mat4& m) {
    printf("--- %s ---\n", name);
    for (int i = 0; i < 4; i++) {
        printf("  [ %.4f, %.4f, %.4f, %.4f ]\n", 
               m[0][i], m[1][i], m[2][i], m[3][i]);
    }
}

glm::mat4 ComputeLightViewProj(
    const glm::vec3& lightPos,
    const glm::vec3& lightDir)
{

    glm::vec3 lightDirNorm = glm::normalize(lightDir);

    // up向量
    glm::vec3 up = glm::abs(lightDirNorm.y) < 0.99f ? glm::vec3(0,1,0) : glm::vec3(1,0,0);

    // 光空间视图矩阵
    glm::vec3 target = lightPos + lightDirNorm;
    glm::mat4 lightView = glm::lookAt(lightPos, target, up);

    // 正交投影矩阵：覆盖 [-4,4] x [-4,4]，深度0.1~10
    float orthoWidth = 4.0f;
    float orthoHeight = 4.0f;
    float nearPlane = 0.1f;
    float farPlane = 10.0f;
    glm::mat4 lightProj = glm::orthoRH_ZO(
        -orthoWidth / 2.0f, orthoWidth / 2.0f,
        -orthoHeight / 2.0f, orthoHeight / 2.0f,
        nearPlane, farPlane
    );

    return lightProj * lightView;
}

glm::mat4 ComputeModelMatrix(const ShmGeom& geom) {
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

        return model_mat;
}

std::string GetMeshName(const ShmGeom &geom, const mjModel* m) {
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
            mesh_name = (m->names) ? 
                std::string(m->names + m->name_meshadr[geom.dataid/2]) : 
                "mesh_" + std::to_string(geom.dataid);
        } else {
            return "unknown_geom";
        }

        return mesh_name;
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

bool BatchRenderer::LoadEnvironmentMap() {
    LOG(config_, "LoadEnvironmentMap(): Starting...");
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;

    // A. Load HDR File (CPU)
    std::filesystem::path hdrPath = getAssetsDir() / "default.hdr";
    if (!std::filesystem::exists(hdrPath)) {
        LOG(config_, "Error: default.hdr not found at " + hdrPath.string());
        return false;
    }

    int width, height, nrComponents;
    float *data = stbi_loadf(hdrPath.string().c_str(), &width, &height, &nrComponents, 4); // Force 4 channels
    if (!data) {
        LOG(config_, "Error: Failed to load HDR image.");
        return false;
    }

    // B. Upload to 2D Texture (Intermediate)
    VkDeviceSize imageSize = width * height * 4 * sizeof(float);
    HostBuffer staging = allocator.makeStagingBuffer(imageSize);
    std::memcpy(staging.ptr, data, imageSize);
    staging.flush(dev);
    stbi_image_free(data); // Free CPU memory

    auto tex2d_res = allocator.makeTextureCubeMap( width, width, 1, VK_FORMAT_R32G32B32A32_SFLOAT );
    env_map_.env_2d_texture = std::move(tex2d_res.first);
    
    // Allocate 2D memory
    auto backing2d = allocator.alloc(tex2d_res.second.size);
    if (!backing2d) return false;
    dev.dt.bindImageMemory(dev.hdl, env_map_.env_2d_texture.image, backing2d.value(), 0);
    env_map_.memory_2d = backing2d.value();

    // Create 2D View (for Compute Shader reading)
    VkImageViewCreateInfo viewInfo2D{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo2D.image = env_map_.env_2d_texture.image;
    viewInfo2D.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo2D.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfo2D.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfo2D, nullptr, &env_map_.view_2d));

    // Execute Copy (Staging -> 2D)
    VkCommandBuffer cmd = render_context_->load_cmd_;
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));

    // Barrier: Undefined -> Transfer Dst
    VkImageMemoryBarrier barrier0{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier0.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier0.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier0.srcAccessMask = 0;
    barrier0.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier0.image = env_map_.env_2d_texture.image;
    barrier0.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier0);

    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {(uint32_t)width, (uint32_t)height, 1};
    dev.dt.cmdCopyBufferToImage(cmd, staging.buffer, env_map_.env_2d_texture.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Barrier: Transfer Dst -> Shader Read (General for Compute)
    VkImageMemoryBarrier barrier1 = barrier0;
    barrier1.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier1.newLayout = VK_IMAGE_LAYOUT_GENERAL; 
    barrier1.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier1.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier1);

    REQ_VK(dev.dt.endCommandBuffer(cmd));
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cmd, 0, nullptr};
    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, VK_NULL_HANDLE));
    dev.dt.deviceWaitIdle(dev.hdl); // Wait for upload

    // C. Create Irradiance Cubemap Image (Target)
    // 32x32 per face, only 1 mip level (no mipmap needed for irradiance)
    uint32_t cubeDim = 32;
    uint32_t mipLevels = 1;

    // 参数 1: size (只需要一个，因为是正方形)
    // 参数 2: mip_levels
    // 参数 3: format
    // 参数 4: usage (STORAGE_BIT for Compute Shader, SAMPLED_BIT for sampling)
    auto cube_res = allocator.makeTextureCube(
        cubeDim, 
        mipLevels, 
        VK_FORMAT_R32G32B32A32_SFLOAT, 
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
   env_map_.irradiance_texture = std::move(cube_res.first);
     
    auto backingCube = allocator.alloc(cube_res.second.size);
    if (!backingCube) return false;
    dev.dt.bindImageMemory(dev.hdl, env_map_.irradiance_texture.image, backingCube.value(), 0);
    env_map_.memory_irradiance = backingCube.value();

    // Create Cube View (For Shader Sampling)
    VkImageViewCreateInfo viewInfoCube{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfoCube.image = env_map_.irradiance_texture.image;
    viewInfoCube.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    viewInfoCube.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    viewInfoCube.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, mipLevels, 0, 6};
    REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfoCube, nullptr, &env_map_.view_irradiance));

    // D. Run Compute Shader (Equirectangular -> Cubemap)
    // We assume the shader converts to the base mip level (0)
    
    // Create view for Mip 0 (Storage Image)
    VkImageViewCreateInfo storageViewInfo = viewInfoCube;
    storageViewInfo.subresourceRange.levelCount = 1; 
    VkImageView cubeMip0View;
    REQ_VK(dev.dt.createImageView(dev.hdl, &storageViewInfo, nullptr, &cubeMip0View));

    // Load Shader
    std::filesystem::path csPath = getLibraryDir() / ".." / "shaders_spv" / "irradiance_cs.spv";
    if (!std::filesystem::exists(csPath)) {
        LOG(config_, "Error: equirect_2_cube.spv not found. Skipping IBL generation.");
        return false;
    }
    VkShaderModule compShader = loadShaderModule(csPath.string());

    // Create Pipeline (1 Combined Sampler for input, 1 Storage Image for output)
    VkDescriptorSetLayoutBinding bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, // Input 2D
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}          // Output Cube
    };
    VkDescriptorSetLayoutCreateInfo descLayoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 2, bindings};
    VkDescriptorSetLayout compDescLayout;
    REQ_VK(dev.dt.createDescriptorSetLayout(dev.hdl, &descLayoutInfo, nullptr, &compDescLayout));

    VkPipelineLayoutCreateInfo plLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &compDescLayout, 0, nullptr};
    VkPipelineLayout compPipelineLayout;
    REQ_VK(dev.dt.createPipelineLayout(dev.hdl, &plLayoutInfo, nullptr, &compPipelineLayout));

    VkPipelineShaderStageCreateInfo shaderStage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, compShader, "main", nullptr};
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, shaderStage, compPipelineLayout, VK_NULL_HANDLE, 0};
    VkPipeline compPipeline;
    REQ_VK(dev.dt.createComputePipelines(dev.hdl, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &compPipeline));

    // Allocate & Update Sets
    VkDescriptorPoolSize poolSizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 2, poolSizes};
    VkDescriptorPool compPool;
    REQ_VK(dev.dt.createDescriptorPool(dev.hdl, &poolInfo, nullptr, &compPool));

    VkDescriptorSetAllocateInfo setAllocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, compPool, 1, &compDescLayout};
    VkDescriptorSet compSet;
    REQ_VK(dev.dt.allocateDescriptorSets(dev.hdl, &setAllocInfo, &compSet));

    // Create a temporary sampler for the equirectangular map
    VkSampler equirectSampler = makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    VkDescriptorImageInfo inputInfo{equirectSampler, env_map_.view_2d, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo outputInfo{VK_NULL_HANDLE, cubeMip0View, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet writes[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &inputInfo, nullptr, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, compSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr}
    };
    dev.dt.updateDescriptorSets(dev.hdl, 2, writes, 0, nullptr);

    // Dispatch Irradiance Compute
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));
    
    // Transition to General for Compute write
    VkImageMemoryBarrier barrierCompute{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrierCompute.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrierCompute.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierCompute.srcAccessMask = 0;
    barrierCompute.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrierCompute.image = env_map_.irradiance_texture.image;
    barrierCompute.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrierCompute);

    dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipeline);
    dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipelineLayout, 0, 1, &compSet, 0, nullptr);
    // Dispatch: (32/8, 32/8, 6 faces) local size 8x8
    dev.dt.cmdDispatch(cmd, cubeDim / 8, cubeDim / 8, 6);

    // Transition to Shader Read Only (no mipmap needed for irradiance)
    VkImageMemoryBarrier barrierRead{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrierRead.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrierRead.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrierRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrierRead.image = env_map_.irradiance_texture.image;
    barrierRead.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6};
    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrierRead);

    REQ_VK(dev.dt.endCommandBuffer(cmd));
    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &submitInfo, render_context_->load_fence_));
    waitForFenceInfinitely(dev, render_context_->load_fence_);

    // F. Cleanup
    dev.dt.destroyImageView(dev.hdl, cubeMip0View, nullptr);
    dev.dt.destroySampler(dev.hdl, equirectSampler, nullptr);
    dev.dt.destroyDescriptorPool(dev.hdl, compPool, nullptr);
    dev.dt.destroyDescriptorSetLayout(dev.hdl, compDescLayout, nullptr);
    dev.dt.destroyPipeline(dev.hdl, compPipeline, nullptr);
    dev.dt.destroyPipelineLayout(dev.hdl, compPipelineLayout, nullptr);
    dev.dt.destroyShaderModule(dev.hdl, compShader, nullptr);

    // Create Final Sampler for Cubemap
    env_map_.sampler = makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);

    LOG(config_, "LoadEnvironmentMap(): Done.");
    return true;
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
                return GeometryBuilder::BuildMesh_MujocoStyle(models_[i_model], i/2);
            });
        }
    }

    // 3. process Built-in Primitives 
    // here set all builtin geom the same size params 
    ProcessGeometry("__builtin_box",      [](){ return GeometryBuilder::BuildBox(12); });
    ProcessGeometry("__builtin_sphere",   [](){ return GeometryBuilder::BuildSphere(16, 16); }); // 16 stacks/slices
    ProcessGeometry("__builtin_capsule",  [](){ return GeometryBuilder::BuildCapsule(16, 16); });
    ProcessGeometry("__builtin_cylinder", [](){ return GeometryBuilder::BuildCylinder(32, 64); });
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

    InitRenderDoc();
    // LOG(config_, "Initialize(): RenderDoc initialized");

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

    if(!LoadEnvironmentMap()){
        LOG(config_, "Initialize(): LoadEnvironmentMap failed");
        return false;
    }

    if (!CreatePipeline()) {
        LOG(config_, "Initialize(): CreatePipeline failed");
        return false;
    }
    if (!CreateFramebuffers()) {
        LOG(config_, "Initialize(): CreateFramebuffers failed");
        return false;
    }

    InitGlobalGeometry();

        // Create pipeline and resources
    if(!CreateShadowResources()) {
        LOG(config_, "Initialize(): CreateShadowResources failed");
        return false;
    }

    if(!CreateShadowPipeline()){
        LOG(config_, "Initialize(): CreateShadowPipeline failed");
        return false;
    }

    // Create buffers (Uniform, Command, etc.) 
    if (!CreateBuffers()) { 
        LOG(config_, "Initialize(): CreateBuffers failed");
        return false; 
    }

    // Initialize per-environment MuJoCo visualization structs
    // NOTE: when render from shared memory, here the mjvScene are not used
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

/** @deprecated Use RecordCommandBuffersFromMemory instead.
 *  NOTE: now invalid for have not done adjust for shadow and framebuffer
 */
[[deprecated("Use RecordCommandBuffersFromMemory instead")]]
bool BatchRenderer::RecordCommandBuffers(int count) {
    LOG(config_, "RecordCommandBuffers() is deprecated. Use RecordCommandBuffersFromMemory() instead.");
    return false;
}

RenderResult BatchRenderer::RenderFromMemory(const uint8_t* shared_memory_ptr, int batch_idx, int max_geom, int max_light) {
    if (!initialized_) return MakeError(RenderError::INVALID_CONFIG, "Renderer not initialized");
    
    profiler_.Reset();
    int current_swap = swap_write_idx_;

    // Phase 1: Async UBO upload via transfer queue
    {   
        ScopeTimer t(profiler_, "1.UpdateAsync");
        ScopedNvtxRange range("1. UpdateAsync", C_UPDATE);
        if (!UpdateScenesFromMemory(shared_memory_ptr, 0, max_geom, max_light)) {
             return MakeError(RenderError::MUJOCO_ERROR, "UpdateScenesFromMemory failed");
        }
    }

    LOG(config_, "UpdateScenesFromMemory succeed");

    // Phase 2: Record with current swap slot's framebuffer
    {
        ScopedNvtxRange range("2. Record", C_RECORD);
        ScopeTimer t(profiler_, "2.Record");
        if (!RecordCommandBuffersFromMemory(shared_memory_ptr, config_.batch_size)) { 
             return MakeError(RenderError::VULKAN_ERROR, "RecordCommandBuffers failed");
        }
    }

    LOG(config_, "RecordCommandBuffersFromMemory succeed");

    // Phase 3: Submit and wait
    {
        ScopeTimer t(profiler_, "3.Submit");
        ScopedNvtxRange range("3. Submit_Wait", C_SUBMIT);
        if (!SubmitAndWait()) {
             return MakeError(RenderError::VULKAN_ERROR, "SubmitAndWait failed");
        }
    }

    LOG(config_, "SubmitAndWait succeed");

    // Phase 4: Async Readback (non-blocking)
    {
        ScopeTimer t(profiler_, "4.ReadbackAsync");
        ScopedNvtxRange range("4. ReadbackAsync", C_READ);
        SubmitReadbackAsync(current_swap, frame_counter_);
    }

    LOG(config_, "SubmitReadback succeed");

    // Rotate write slot
    swap_write_idx_ = (swap_write_idx_ + 1) % SWAP_COUNT;
    frame_counter_++;

    profiler_.Print();
    
    return RenderResult();
}

bool BatchRenderer::UpdateScenesFromMemory(const uint8_t* ptr, int start_idx, int max_geom, int max_light) {
    Device &dev = *device_;
    const EnvRenderSlot* slots = reinterpret_cast<const EnvRenderSlot*>(ptr);

    // We only update UBOs here (Camera & Light). 
    // Geometry is read directly in the Record phase.

    /**
     * FIXME: between 1 and 2 , here exist a fence problem that case the process hang permanently
     */

    std::vector<VkBufferCopy> copy_regions;

    // Use transfer queue for async UBO updates
    SwapSlot& wslot = swap_slots_[swap_write_idx_];
    VkCommandBuffer cmd = wslot.transfer_cmd;

    // Wait for previous transfer to complete
    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, &wslot.transfer_fence, VK_TRUE, UINT64_MAX));
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &wslot.transfer_fence));

    REQ_VK(dev.dt.resetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));


    for(int i = 0; i < config_.batch_size; ++i) {
        const EnvRenderSlot& slot = slots[i];

        // --- 1. Prepare Light Data (Shared for all 3 cameras in this Env) ---
        LightUBO env_light_ubo{};
        env_light_ubo.lightCount = slot.num_lights;

        // Safety clamp just in case shared memory has garbage data
        if (env_light_ubo.lightCount > 10) env_light_ubo.lightCount = 10;

        for (int l = 0; l < env_light_ubo.lightCount; ++l) {
            const ShmLight& src = slot.lights[l];
            auto& dst = env_light_ubo.lights[l];

            // Direct mapping from Shared Memory struct to UBO struct
            dst.position    = glm::vec3(src.pos[0], src.pos[1], src.pos[2]);
            dst.direction   = glm::vec3(src.dir[0], src.dir[1], src.dir[2]);
            
            // Attenuation: x=Constant, y=Linear, z=Quadratic
            dst.attenuation = glm::vec3(src.attenuation[0], src.attenuation[1], src.attenuation[2]);
            dst.exponent    = src.exponent;
            dst.cutoff      = src.cutoff;
            
            dst.ambient     = glm::vec3(src.ambient[0], src.ambient[1], src.ambient[2]);
            dst.diffuse     = glm::vec3(src.diffuse[0], src.diffuse[1], src.diffuse[2]);
            dst.specular    = glm::vec3(src.specular[0], src.specular[1], src.specular[2]);
            
            dst.castShadow  = (uint32_t)src.castshadow;
            dst.type = src.type;


            dst.view_proj = ComputeLightViewProj(dst.position,dst.direction); 

        }

        cached_shadow_matrices_[i] = env_light_ubo.lights[1].view_proj;// Assuming light 1 is the shadow caster
        
        for (int c = 0; c < SHM_NUM_CAMERAS; ++c) {
            int resource_idx = i * SHM_NUM_CAMERAS  + c;

            // --- 1. Update Camera UBO --
            const ShmCamera& src_cam = slot.cameras[c]; 

            // Convert Row-Major to GLM Mat4
            glm::mat4 view = RowMajorToGLM(src_cam.view);
            glm::mat4 proj = RowMajorToGLM(src_cam.proj);

            // PrintMatrix("Camera View", view);
            // PrintMatrix("Camera Proj", proj);

            glm::mat4 view_proj = proj * view;
            glm::vec3 cam_pos = glm::vec3(src_cam.pos[0], src_cam.pos[1], src_cam.pos[2]);
            // PrintMatrix("Camera ViewProj", view_proj);

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


            // --- Light UBO Update (Copy the env_light_ubo we made earlier) ---
            std::memcpy(light_staging_buffers_[resource_idx].ptr, &env_light_ubo, sizeof(LightUBO));
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
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &wslot.transfer_semaphore;

    // Submit to transfer queue, signal semaphore when done
    REQ_VK(dev.dt.queueSubmit(render_context_->transferQueue, 1, &submitInfo, wslot.transfer_fence));

    // No longer wait idle - render queue will wait on semaphore
    
    return true;
}

bool BatchRenderer::RecordCommandBuffersFromMemory(const uint8_t* ptr, int count) {
    Device &dev = *device_;
   
    const EnvRenderSlot* slots = reinterpret_cast<const EnvRenderSlot*>(ptr);
    
    SwapSlot& wslot = swap_slots_[swap_write_idx_];

    // Wait for GPU to finish previous use of this slot (3 frames ago), then reset fence
    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, &wslot.render_fence, VK_TRUE, UINT64_MAX));
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &wslot.render_fence));

    {
        std::unique_lock<std::mutex> lk(swap_mutex_);
        swap_cv_.wait(lk, [&] { return wslot.state == SwapSlot::State::FREE; });
        wslot.state = SwapSlot::State::RENDERING;
    }

    REQ_VK(dev.dt.resetCommandBuffer(wslot.render_cmd, 0));
    VkCommandBuffer cmd = wslot.render_cmd;
        // --- 1. Begin Recording & Render Pass ---
    VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &beginInfo));

    // =========================================================================
    // STEP A: SHADOW PASS (Single Atlas Pass)
    // =========================================================================
    {
        ScopeTimer t(profiler_, "2. Record_CommandBuffers.ShadowPass");
        // 1. Calculate Shadow Atlas Layout
        // Must match logic in CreateShadowResources
        int shadow_cols = std::ceil(std::sqrt((float)config_.batch_size));
        int shadow_rows = std::ceil((float)config_.batch_size / shadow_cols);
        uint32_t atlas_w = shadow_cols * SHADOW_MAP_DIM;
        uint32_t atlas_h = shadow_rows * SHADOW_MAP_DIM;

        VkRenderPassBeginInfo shadowPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        shadowPassInfo.renderPass = render_context_->shadowPass;
        shadowPassInfo.framebuffer = shadow_framebuffer_; // [CHANGE] Single Atlas FB
        shadowPassInfo.renderArea.extent = {atlas_w, atlas_h};

        VkClearValue clearDepth = {.depthStencil = {1.0f, 0}};
        shadowPassInfo.clearValueCount = 1;
        shadowPassInfo.pClearValues = &clearDepth;

        // Start Pass ONCE
        dev.dt.cmdBeginRenderPass(cmd, &shadowPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);

        // Bind Global Vertex Buffers ONCE
        if (global_vertex_buffer_->buffer != VK_NULL_HANDLE) {
            VkBuffer vbs[] = { global_vertex_buffer_->buffer };
            VkDeviceSize offsets[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
            vkCmdBindIndexBuffer(cmd, global_index_buffer_->buffer, 0, VK_INDEX_TYPE_UINT32);

            // Loop over environments inside the pass
            for (int i = 0; i < count; ++i) {
                const EnvRenderSlot& slot = slots[i];

                // Calculate Viewport Offset
                int grid_x = i % shadow_cols;
                int grid_y = i / shadow_cols;
                float vp_x = grid_x * (float)SHADOW_MAP_DIM;
                float vp_y = grid_y * (float)SHADOW_MAP_DIM;

                VkViewport vp = {vp_x, vp_y, (float)SHADOW_MAP_DIM, (float)SHADOW_MAP_DIM, 0.0f, 1.0f};
                dev.dt.cmdSetViewport(cmd, 0, 1, &vp);
                VkRect2D sc = {{ (int32_t)vp_x, (int32_t)vp_y }, {SHADOW_MAP_DIM, SHADOW_MAP_DIM}};
                dev.dt.cmdSetScissor(cmd, 0, 1, &sc);

                // Draw Geoms for this shadow map
                int32_t active_geoms = std::min(slot.num_geoms, (int32_t)SHM_MAX_GEOMS);
                glm::mat4 lightViewProj = cached_shadow_matrices_[i];

                for (int g = 0; g < active_geoms; ++g) {
                    const ShmGeom& geom = slot.geoms[g];
                    glm::mat4 model_mat = ComputeModelMatrix(geom);

                    PushConstantsShadow pc_shadow{};
                    pc_shadow.mvp = lightViewProj * model_mat; 

                    dev.dt.cmdPushConstants(cmd, pipeline_layout_, 
                        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantsShadow), &pc_shadow);

                    auto it = global_mesh_cache_.find(GetMeshName(geom, models_[i]));
                    if (it != global_mesh_cache_.end()) {
                        vkCmdDrawIndexed(cmd, it->second.index_count, 1, it->second.index_offset, it->second.vertex_offset, 0);
                    }
                }
            }
        }
        dev.dt.cmdEndRenderPass(cmd);
    }

        // STEP B: BARRIER (Wait for Shadow Map Write to Finish)
        {
            VkImageMemoryBarrier barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL; // Or Undefined if using LOAD_OP_CLEAR and not caring
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // Ready for sampler
            barrier.image = shadow_image_->image;
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;

            dev.dt.cmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &barrier);
        }

        // =========================================================================
        // STEP C: MAIN RENDER PASS (Single Atlas Pass)
        // =========================================================================
        {
            ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass");
            // 1. Calculate Main Atlas Layout
            int total_views = config_.batch_size * SHM_NUM_CAMERAS;
            int main_cols = std::ceil(std::sqrt((float)total_views));
            int main_rows = std::ceil((float)total_views / main_cols);
            uint32_t atlas_w = main_cols * config_.frame_width;
            uint32_t atlas_h = main_rows * config_.frame_height;
        
            VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
            renderPassInfo.renderPass = render_context_->renderPass;
            renderPassInfo.framebuffer = wslot.framebuffer;
            renderPassInfo.renderArea.extent = {atlas_w, atlas_h};
            
            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = {{0.1f, 0.1f, 0.1f, 1.0f}}; 
            clearValues[1].depthStencil = {1.0f, 0};
            renderPassInfo.clearValueCount = clearValues.size();
            renderPassInfo.pClearValues = clearValues.data();
            
            {
                ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.BeginRenderPass");
                dev.dt.cmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
            }
            // --- 2. Bind Pipeline & Global State ---
            {
                ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.BindPipeline");
                dev.dt.cmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
            }
            // Bind Vertices ONCE
            if (global_vertex_buffer_->buffer != VK_NULL_HANDLE) {
                {
                    ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.BindVertexIndex");
                    VkBuffer vbs[] = { global_vertex_buffer_->buffer };
                    VkDeviceSize offsets[] = { 0 };
                    vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                    vkCmdBindIndexBuffer(cmd, global_index_buffer_->buffer, 0, VK_INDEX_TYPE_UINT32);
                }
                // Loop Envs
                for (int i = 0; i < count; ++i) {
                    ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops");
                    const EnvRenderSlot& slot = slots[i];

                    // Loop Cameras
                    for (int c = 0; c < SHM_NUM_CAMERAS; ++c) {
                        int resource_idx = i * SHM_NUM_CAMERAS + c;

                        // Calculate Viewport Offset for this Camera
                        int grid_x = resource_idx % main_cols;
                        int grid_y = resource_idx / main_cols;
                        float vp_x = grid_x * (float)config_.frame_width;
                        float vp_y = grid_y * (float)config_.frame_height;
                        {
                            //ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.SetView");
                            VkViewport viewport = {vp_x, vp_y, (float)config_.frame_width, (float)config_.frame_height, 0.0f, 1.0f};
                            dev.dt.cmdSetViewport(cmd, 0, 1, &viewport);
                            VkRect2D scissor = {{ (int32_t)vp_x, (int32_t)vp_y }, {(uint32_t)config_.frame_width, (uint32_t)config_.frame_height}};
                            dev.dt.cmdSetScissor(cmd, 0, 1, &scissor);
                        }
                        // Re-bind Descriptor Sets for this camera (UBOs are per-camera/slot)
                        {
                            //ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.BindDescriptor");
                            std::vector<VkDescriptorSet> sets = { descriptor_sets_[resource_idx], global_texture_descriptor_set };
                            dev.dt.cmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 
                                                    0, static_cast<uint32_t>(sets.size()), sets.data(), 0, nullptr);
                        }
                        // Draw Geometry (with Culling)
                        // const auto& cull_info = camera_cull_info_[resource_idx];
                        int32_t active_geoms = std::min(slot.num_geoms, (int32_t)SHM_MAX_GEOMS);

                        for (int g = 0; g < active_geoms; ++g)
                        {
                            //ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops");
                            //profiler_.total_geoms++;

                            const ShmGeom& geom = slot.geoms[g];

                            // --------------------------------------------------
                            // Mesh Name + Mesh Cache Lookup
                            // --------------------------------------------------
                            std::string mesh_name;
                            const MeshEntry* entry = nullptr;

                            {
                               ////ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.MeshLookup");

                                mesh_name = GetMeshName(geom, models_[i]);
                                auto it = global_mesh_cache_.find(mesh_name);
                                if (it == global_mesh_cache_.end())
                                    continue;

                                entry = &it->second;
                            }

                            // --------------------------------------------------
                            // Compute Model Matrix
                            // --------------------------------------------------
                            glm::mat4 model_mat;

                            {
                                //ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.ComputeModelMatrix");
                                model_mat = ComputeModelMatrix(geom);
                            }
                            #if 0
                            {
                                // --------------------------------------------------
                                // AABB Lookup
                                // --------------------------------------------------
                                const AABB* local_aabb_ptr = nullptr;

                                {
                                    ////ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.AABBLookup");

                                    auto it2 = global_aabb_cache_.find(mesh_name);
                                    if (it2 == global_aabb_cache_.end())
                                        continue;

                                    local_aabb_ptr = &it2->second;
                                }

                                const AABB& local_aabb = *local_aabb_ptr;

                                //--------------------------------------------------
                                //World AABB Transform
                                //--------------------------------------------------
                                glm::vec3 world_min, world_max;

                                {
                                    //ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.WorldAABB");

                                    glm::vec3 local_center =
                                        (glm::vec3(local_aabb.min.x, local_aabb.min.y, local_aabb.min.z) +
                                        glm::vec3(local_aabb.max.x, local_aabb.max.y, local_aabb.max.z)) * 0.5f;

                                    glm::vec3 local_extent =
                                        (glm::vec3(local_aabb.max.x, local_aabb.max.y, local_aabb.max.z) -
                                        glm::vec3(local_aabb.min.x, local_aabb.min.y, local_aabb.max.z)) * 0.5f;

                                    glm::vec3 world_center =
                                        glm::vec3(model_mat * glm::vec4(local_center, 1.0f));

                                    glm::vec3 world_extent;

                                    for (int k = 0; k < 3; k++)
                                    {
                                        world_extent[k] =
                                            std::abs(model_mat[0][k]) * local_extent.x +
                                            std::abs(model_mat[1][k]) * local_extent.y +
                                            std::abs(model_mat[2][k]) * local_extent.z;
                                    }

                                    world_min = world_center - world_extent;
                                    world_max = world_center + world_extent;
                                }

                                // --------------------------------------------------
                                // Frustum Culling
                                // --------------------------------------------------
                                bool visible;

                                {
                                    ////ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.FrustumTest");
                                    visible = IsAABBVisible(cull_info.frustum, world_min, world_max);
                                }

                                if (!visible)
                                {
                                    // profiler_.culled_geoms++;
                                    // continue;
                                }

                                profiler_.visible_geoms++;

                            }
                            #endif

                            // --------------------------------------------------
                            // Texture Resolve
                            // --------------------------------------------------
                            int texture_type = -1;
                            int texture_index = -1;

                            {
                                ////ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.TextureResolve");

                                int resolved_tex_id = -1;
                                if (geom.matid >= 0)
                                    resolved_tex_id = ResolveTextureFromMaterial(models_[i], geom.matid);

                                if (resolved_tex_id >= 0 && i < texture_offsets_.size())
                                {
                                    int global_tex_id = texture_offsets_[i] + resolved_tex_id;

                                    if (global_tex_id < material_textures_.global_texture_lookup.size())
                                    {
                                        const auto& tex_map =
                                            material_textures_.global_texture_lookup[global_tex_id];

                                        texture_type =
                                            tex_map.index_in_array >= 0 ? tex_map.type : -1;

                                        texture_index = tex_map.index_in_array;
                                    }
                                }
                            }

                            // --------------------------------------------------
                            // Push Constants + Draw
                            // --------------------------------------------------
                            {
                                ////ScopeTimer t(profiler_, "2. Record_CommandBuffers.MainPass.Envloops.Gloops.DrawRecord");

                                int shadow_cols =
                                    std::ceil(std::sqrt((float)config_.batch_size));

                                int shadow_row_idx = i / shadow_cols;
                                int shadow_col_idx = i % shadow_cols;

                                float shadow_scale = 1.0f / (float)shadow_cols;
                                float shadow_offset_x = shadow_col_idx * shadow_scale;
                                float shadow_offset_y = shadow_row_idx * shadow_scale;

                                PushConstants pc{};
                                pc.model = model_mat;
                                pc.rgba = glm::vec4(
                                    geom.rgba[0],
                                    geom.rgba[1],
                                    geom.rgba[2],
                                    geom.rgba[3]);

                                pc.specular = geom.specular;
                                pc.emission = geom.emission;
                                pc.shininess = geom.shininess;
                                pc.reflectance = geom.reflectance;
                                pc.shadow_atlas_params =
                                    glm::vec4(shadow_offset_x, shadow_offset_y, shadow_scale, 0.0f);

                                pc.texture_type = texture_type;
                                pc.texture_index = texture_index;

                                dev.dt.cmdPushConstants(
                                    cmd,
                                    pipeline_layout_,
                                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                    0,
                                    sizeof(PushConstants),
                                    &pc);

                                vkCmdDrawIndexed(
                                    cmd,
                                    entry->index_count,
                                    1,
                                    entry->index_offset,
                                    entry->vertex_offset,
                                    0);
                            }
                        }
                    }
                }
            }
            dev.dt.cmdEndRenderPass(cmd);
        }
    REQ_VK(dev.dt.endCommandBuffer(cmd));
    
    return true;
}

bool BatchRenderer::SubmitAndWait() {
    Device &dev = *device_;
    VkQueue queue = render_context_->renderQueue;
    int slot_idx = swap_write_idx_;
    SwapSlot& slot = swap_slots_[slot_idx];

    // --- 2. Wait for transfer semaphore (UBO updates must be complete) ---
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    // --- 3. prepare Batching submitinfo ---
    VkSubmitInfo submitInfo = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &slot.transfer_semaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &slot.render_cmd;
    // Signal render complete semaphore for readback to wait on
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &slot.render_complete_semaphore;

    // --- 4. Submit (non-blocking: signal slot.render_fence, return immediately) ---
    VkResult submitResult = dev.dt.queueSubmit(queue, 1, &submitInfo, slot.render_fence);

    if (submitResult != VK_SUCCESS) {
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), "ERROR: Batch queueSubmit failed: %d", submitResult);
        LOG(config_, log_buf);
        return false;
    }

    return true;
}

/** @deprecated Use SubmitReadbackAsync with readback thread instead.
 *  Readback results using legacy color_image_, which is now removed.
 */
[[deprecated("Use SubmitReadbackAsync instead")]]
bool BatchRenderer::ReadbackResults() {
    LOG(config_, "ReadbackResults() is deprecated. Use SubmitReadbackAsync() instead.");
    return false;
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
    std::filesystem::path shaderDir = getLibraryDir() / ".." / "shaders_spv";

    vert_shader_module_ = loadShaderModule((shaderDir / "mujoco_vs.spv").string());

    frag_shader_module_ = loadShaderModule((shaderDir / "mujoco_ps.spv").string());
    
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
    std::array<VkDescriptorSetLayoutBinding, 5> bindings{};
    bindings[0].binding = 0; // Camera UBO
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    
    bindings[1].binding = 1; // Lighting UBO
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    bindings[2].binding = 2; // Shadow Map (if used)
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[2].pImmutableSamplers = nullptr;

    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[3].pImmutableSamplers = nullptr;

    // Binding 4: Environment Cubemap
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[4].pImmutableSamplers = nullptr;

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

    int total_views = config_.batch_size * SHM_NUM_CAMERAS;
    int cols = std::ceil(std::sqrt((float)total_views));
    int rows = std::ceil((float)total_views / cols);
    uint32_t total_width = cols * config_.frame_width;
    uint32_t total_height = rows * config_.frame_height;

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = render_context_->ring_cmd_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = SWAP_COUNT;

    std::array<VkCommandBuffer, SWAP_COUNT> rb_cmds;
    REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &alloc_info, rb_cmds.data()));

    // Allocate per-slot transfer command buffers
    std::array<VkCommandBuffer, SWAP_COUNT> transfer_cmds;
    {
        VkCommandBufferAllocateInfo ta{};
        ta.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ta.commandPool = render_context_->transfer_cmd_pool_;
        ta.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ta.commandBufferCount = SWAP_COUNT;
        REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &ta, transfer_cmds.data()));
    }

    for (int s = 0; s < SWAP_COUNT; ++s) {
        auto& slot = swap_slots_[s];

        slot.color_image.emplace(allocator.makeColorAttachment(
            total_width, total_height, 1, VK_FORMAT_R8G8B8A8_UNORM));

        if (config_.enable_depth) {
            slot.depth_image.emplace(allocator.makeDepthAttachment(
                total_width, total_height, 1, VK_FORMAT_D32_SFLOAT));
        }

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        viewInfo.image = slot.color_image->image;
        viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfo, nullptr, &slot.color_view));

        if (config_.enable_depth && slot.depth_image) {
            viewInfo.image = slot.depth_image->image;
            viewInfo.format = VK_FORMAT_D32_SFLOAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfo, nullptr, &slot.depth_view));
        }

        std::vector<VkImageView> attachments = {slot.color_view};
        if (config_.enable_depth && slot.depth_view != VK_NULL_HANDLE) {
            attachments.push_back(slot.depth_view);
        }

        VkFramebufferCreateInfo fbInfo{};
        fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fbInfo.renderPass = render_context_->renderPass;
        fbInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        fbInfo.pAttachments = attachments.data();
        fbInfo.width = total_width;
        fbInfo.height = total_height;
        fbInfo.layers = 1;
        REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fbInfo, nullptr, &slot.framebuffer));

        size_t per_frame = config_.frame_width * config_.frame_height * 4;
        slot.staging_bufs.reserve(total_views);
        for (size_t i = 0; i < total_views; ++i) {
            slot.staging_bufs.emplace_back(allocator.makeStagingBuffer2(per_frame));
        }

        slot.readback_cmd = rb_cmds[s];
        slot.readback_fence = makeFence(dev, true);
        slot.render_complete_semaphore = makeBinarySemaphore(dev);
        slot.transfer_cmd = transfer_cmds[s];
        slot.transfer_fence = makeFence(dev, true);
        slot.transfer_semaphore = makeBinarySemaphore(dev);
        slot.render_fence = makeFence(dev, true);
        slot.state = SwapSlot::State::FREE;
    }

    return true;
}

bool BatchRenderer::CreateBuffers() {
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;

    size_t total_slots = config_.batch_size * SHM_NUM_CAMERAS;
    
    VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    poolInfo.queueFamilyIndex = dev.gfxQF;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    REQ_VK(dev.dt.createCommandPool(dev.hdl, &poolInfo, nullptr, &command_pool_));

    // Allocate per-slot render command buffers
    {
        VkCommandBufferAllocateInfo allocCmdInfo{};
        allocCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocCmdInfo.commandPool = command_pool_;
        allocCmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocCmdInfo.commandBufferCount = SWAP_COUNT;
        std::array<VkCommandBuffer, SWAP_COUNT> render_cmds;
        REQ_VK(dev.dt.allocateCommandBuffers(dev.hdl, &allocCmdInfo, render_cmds.data()));
        for (int s = 0; s < SWAP_COUNT; ++s)
            swap_slots_[s].render_cmd = render_cmds[s];
    }

    // Create fences
    {
        // Create descriptor pool
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = total_slots; // Camera UBOs
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[1].descriptorCount = total_slots; // Light UBOs
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[2].descriptorCount = total_slots * 3; // Shadow Map + BRDF LUT
        
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
        
        std::array<VkWriteDescriptorSet, 4> writes{};

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

        VkDescriptorImageInfo shadowInfo{};
        shadowInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        shadowInfo.imageView = shadow_image_view_; // Use the env's shadow map
        shadowInfo.sampler = shadow_sampler_;

        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstSet = descriptor_sets_[i];
        writes[2].dstBinding = 2; // [IMPORTANT] Ensure Layout has binding 2 added!
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &shadowInfo;

        // Write 3: Irradiance Cubemap (Diffuse IBL)
        VkDescriptorImageInfo irrInfo{};
        irrInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        irrInfo.imageView = env_map_.view_irradiance;
        irrInfo.sampler = env_map_.sampler;
        
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstSet = descriptor_sets_[i];
        writes[3].dstBinding = 3;
        writes[3].dstArrayElement = 0;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo = &irrInfo;
        
        dev.dt.updateDescriptorSets(dev.hdl, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
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
    
    LOG(config_, "CreateBuffers(): buffers created");
    return true;
}

// [NEW] Add this function implementation
bool BatchRenderer::CreateShadowResources() {
    Device &dev = *device_;
    MemoryAllocator &allocator = render_context_->allocator;
    // We strictly assume one shadow map per environment (batch_size)
    size_t count = config_.batch_size; 
    cached_shadow_matrices_.resize(count);

    // 1. Calculate Shadow Atlas Dimensions
    int cols = std::ceil(std::sqrt((float)count));
    int rows = std::ceil((float)count / cols);
    
    uint32_t total_width = cols * SHADOW_MAP_DIM;
    uint32_t total_height = rows * SHADOW_MAP_DIM;

    // 1. Create Shadow Render Pass
    VkAttachmentDescription attachmentDescription{};
    attachmentDescription.format = VK_FORMAT_D32_SFLOAT;
    attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
    attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE; // We need to read it later
    attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; // Ready for sampling

    VkAttachmentReference depthReference = { 0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0; // Depth only
    subpass.pDepthStencilAttachment = &depthReference;

    // Dependency to ensure write finishes before read
    std::array<VkSubpassDependency, 2> dependencies;
    
    // Transition 1: Undefined -> Depth Write
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    // Transition 2: Depth Write -> Shader Read
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &attachmentDescription;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
    renderPassInfo.pDependencies = dependencies.data();

    REQ_VK(dev.dt.createRenderPass(dev.hdl, &renderPassInfo, nullptr, &render_context_->shadowPass));

    // 2. Create Sampler (Shadow Sampler with PCF support usually requires logic in shader, here strictly linear/nearest)
    shadow_sampler_ = makeImmutableSampler(dev, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER);

    // 4. Create Single Large Shadow Image
    shadow_image_.emplace(allocator.makeDepthAttachment(total_width, total_height, 1, VK_FORMAT_D32_SFLOAT));

    // 5. Create View
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = shadow_image_->image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_D32_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    REQ_VK(dev.dt.createImageView(dev.hdl, &viewInfo, nullptr, &shadow_image_view_));

    // 6. Create Single Large Framebuffer
    VkFramebufferCreateInfo fbInfo{};
    fbInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fbInfo.renderPass = render_context_->shadowPass;
    fbInfo.attachmentCount = 1;
    fbInfo.pAttachments = &shadow_image_view_;
    fbInfo.width = total_width;
    fbInfo.height = total_height;
    fbInfo.layers = 1;

    REQ_VK(dev.dt.createFramebuffer(dev.hdl, &fbInfo, nullptr, &shadow_framebuffer_));
    return true;
}

// [NEW] Function to create Shadow Pipeline (Vertex Only)
bool BatchRenderer::CreateShadowPipeline() {
    Device &dev = *device_;
        std::filesystem::path shaderPath =
    getLibraryDir() / ".." / "shaders_spv" / "shadow_vs.spv";
    
    // Reuse existing VS or create a specialized one "shadow_vs.spv"
    // Shadow VS only needs: gl_Position = light_proj * light_view * model * pos;
    // For now, assuming you compile a new shader 'shadow_vs.spv'
    VkShaderModule shadow_vs = loadShaderModule(shaderPath);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = shadow_vs;
    vertStage.pName = "VSMain"; // Entry point

    // Vertex Input (Same as main pipeline to reuse buffers)
    VkVertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(float) * (3 + 3 + 2 + 4); 
    bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Only Position is needed
    VkVertexInputAttributeDescription posAttr{};
    posAttr.binding = 0;
    posAttr.location = 0; 
    posAttr.format = VK_FORMAT_R32G32B32_SFLOAT;
    posAttr.offset = 0;

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &bindingDescription;
    vertexInput.vertexAttributeDescriptionCount = 1;
    vertexInput.pVertexAttributeDescriptions = &posAttr;

    // Input Assembly
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport (Dynamic)
    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_NONE; // Often better to draw backfaces or none for shadows
    rasterizer.depthBiasEnable = VK_TRUE;      // [IMPORTANT] Shadow bias
    rasterizer.depthBiasConstantFactor = 0.002f;
    rasterizer.depthBiasSlopeFactor = 1.5f;


    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; 
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    // Depth Stencil
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    // No Color Blend (Depth only)
    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = 0;

    // Dynamic State
    std::vector<VkDynamicState> dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = dynamicStates.size();
    dynamicState.pDynamicStates = dynamicStates.data();

    // Pipeline Layout (Push Constant for MVP)
    // We reuse the main pipeline layout but we only use the PushConstant range 
    // You might want a separate layout if the sets are different, but reusing is okay if sets are compatible.
    // Shadow pass usually only needs PushConstant (Model) + PushConstant (LightViewProj) OR a UBO.
    // Let's assume we pass LightViewProj via PushConstant offset or a specific UBO.
    // For simplicity: reuse pipeline_layout_ and pass Model via PushConstant, LightMatrix via logic? 
    // Actually, usually Shadow Pass needs [LightMatrix * ModelMatrix]. 
    // Let's assume the Shadow VS takes the SAME PushConstants as main, but ignores material stuff.

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 1;
    pipelineInfo.pStages = &vertStage;
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = pipeline_layout_; // Reuse
    pipelineInfo.renderPass = render_context_->shadowPass;
    pipelineInfo.subpass = 0;

    REQ_VK(dev.dt.createGraphicsPipelines(dev.hdl, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadow_pipeline_));
    
    dev.dt.destroyShaderModule(dev.hdl, shadow_vs, nullptr);
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

    // Irradiance Cubemap resources
    if (env_map_.view_irradiance != VK_NULL_HANDLE) dev.dt.destroyImageView(dev.hdl, env_map_.view_irradiance, nullptr);
    if (env_map_.view_2d != VK_NULL_HANDLE) dev.dt.destroyImageView(dev.hdl, env_map_.view_2d, nullptr);
    if (env_map_.irradiance_texture.image != VK_NULL_HANDLE) dev.dt.destroyImage(dev.hdl, env_map_.irradiance_texture.image, nullptr);
    if (env_map_.env_2d_texture.image != VK_NULL_HANDLE) dev.dt.destroyImage(dev.hdl, env_map_.env_2d_texture.image, nullptr);
    if (env_map_.memory_irradiance != VK_NULL_HANDLE) dev.dt.freeMemory(dev.hdl, env_map_.memory_irradiance, nullptr);
    if (env_map_.memory_2d != VK_NULL_HANDLE) dev.dt.freeMemory(dev.hdl, env_map_.memory_2d, nullptr);
    if (env_map_.sampler != VK_NULL_HANDLE) dev.dt.destroySampler(dev.hdl, env_map_.sampler, nullptr);

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

    // Destroy swap slots resources
    for (int s = 0; s < SWAP_COUNT; ++s) {
        SwapSlot& slot = swap_slots_[s];
        
        if (slot.framebuffer != VK_NULL_HANDLE) {
            dev.dt.destroyFramebuffer(dev.hdl, slot.framebuffer, nullptr);
            slot.framebuffer = VK_NULL_HANDLE;
        }
        
        if (slot.color_view != VK_NULL_HANDLE) {
            dev.dt.destroyImageView(dev.hdl, slot.color_view, nullptr);
            slot.color_view = VK_NULL_HANDLE;
        }
        
        if (slot.depth_view != VK_NULL_HANDLE) {
            dev.dt.destroyImageView(dev.hdl, slot.depth_view, nullptr);
            slot.depth_view = VK_NULL_HANDLE;
        }
        
        if (slot.readback_fence != VK_NULL_HANDLE) {
            dev.dt.destroyFence(dev.hdl, slot.readback_fence, nullptr);
            slot.readback_fence = VK_NULL_HANDLE;
        }
        
        if (slot.render_complete_semaphore != VK_NULL_HANDLE) {
            dev.dt.destroySemaphore(dev.hdl, slot.render_complete_semaphore, nullptr);
            slot.render_complete_semaphore = VK_NULL_HANDLE;
        }

        if (slot.transfer_semaphore != VK_NULL_HANDLE) {
            dev.dt.destroySemaphore(dev.hdl, slot.transfer_semaphore, nullptr);
            slot.transfer_semaphore = VK_NULL_HANDLE;
        }

        if (slot.transfer_fence != VK_NULL_HANDLE) {
            dev.dt.destroyFence(dev.hdl, slot.transfer_fence, nullptr);
            slot.transfer_fence = VK_NULL_HANDLE;
        }

        if (slot.render_fence != VK_NULL_HANDLE) {
            dev.dt.destroyFence(dev.hdl, slot.render_fence, nullptr);
            slot.render_fence = VK_NULL_HANDLE;
        }

        slot.staging_bufs.clear();
        if (slot.color_image) slot.color_image.reset();
        if (slot.depth_image) slot.depth_image.reset();
    }

    // Destroy shadow resources
    if (shadow_pipeline_ != VK_NULL_HANDLE) dev.dt.destroyPipeline(dev.hdl, shadow_pipeline_, nullptr);
    
    dev.dt.destroySampler(dev.hdl, shadow_sampler_, nullptr);

    // [New] Clean up Shadow Pass Resources (Atlas)
    if (shadow_framebuffer_ != VK_NULL_HANDLE) {
        dev.dt.destroyFramebuffer(dev.hdl, shadow_framebuffer_, nullptr);
        shadow_framebuffer_ = VK_NULL_HANDLE;
    }

    if (shadow_image_view_ != VK_NULL_HANDLE) {
        dev.dt.destroyImageView(dev.hdl, shadow_image_view_, nullptr);
        shadow_image_view_ = VK_NULL_HANDLE;
    }

    if (shadow_image_.has_value()) {
        shadow_image_.reset();
    }
    
    
    LOG(config_, "DestroyVulkanResources(): resources released");
}

bool BatchRenderer::UpdateAsync(const uint8_t* shm_ptr, int max_geom, int max_light) {
    return UpdateScenesFromMemory(shm_ptr, 0, max_geom, max_light);
}

int BatchRenderer::RecordNext(const uint8_t* shm_ptr) {
    if (!RecordCommandBuffersFromMemory(shm_ptr, config_.batch_size)) {
        return -1;
    }
    return swap_write_idx_;
}

bool BatchRenderer::SubmitNext() {
    if (!SubmitAndWait()) {
        return false;
    }

    int slot_idx = swap_write_idx_;
    SubmitReadbackAsync(slot_idx, frame_counter_);

    swap_write_idx_ = (swap_write_idx_ + 1) % SWAP_COUNT;
    frame_counter_++;

    return true;
}

bool BatchRenderer::WaitSlot(int slot_idx) {
    Device &dev = *device_;
    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, &swap_slots_[slot_idx].render_fence, VK_TRUE, UINT64_MAX));
    return true;
}

bool BatchRenderer::WaitReadback(int slot_idx) {
    Device &dev = *device_;
    SwapSlot& slot = swap_slots_[slot_idx];

    // 等 readback GPU cmd 完成
    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, 
        &slot.readback_fence, VK_TRUE, UINT64_MAX));
    
    // 等 ReadbackThreadFn 处理完（invalidate + callback + state=FREE）
    std::unique_lock<std::mutex> lk(swap_mutex_);
    swap_cv_.wait(lk, [&] {
        return slot.state == SwapSlot::State::FREE;
    });
    return true;
}

void BatchRenderer::CopyFrameFromStaging(int slot_idx, int resource_idx,
                                          uint8_t* dst, size_t size) {
    SwapSlot& slot = swap_slots_[slot_idx];
    
    // 检查1: staging buffer 指针是否有效
    if (!slot.staging_bufs[resource_idx].ptr) {
        printf("[CopyFrame] ERROR: staging buf ptr is null, slot=%d res=%d\n",
               slot_idx, resource_idx);
        return;
    }
    
    const uint8_t* src = (const uint8_t*)slot.staging_bufs[resource_idx].ptr;
    
    // 检查2: 前16个像素是否全0
    bool all_zero = true;
    for (int i = 0; i < 64; i++) {
        if (src[i] != 0) { all_zero = false; break; }
    }
    printf("[CopyFrame] slot=%d res=%d all_zero=%d first_bytes=[%d,%d,%d,%d]\n",
           slot_idx, resource_idx, all_zero,
           src[0], src[1], src[2], src[3]);
    
    std::memcpy(dst, src, size);
}

bool BatchRenderer::SubmitReadbackAsync(int swap_idx, int step_id) {
    Device &dev = *device_;
    SwapSlot& slot = swap_slots_[swap_idx];

    REQ_VK(dev.dt.waitForFences(dev.hdl, 1, &slot.readback_fence, VK_TRUE, UINT64_MAX));
    REQ_VK(dev.dt.resetFences(dev.hdl, 1, &slot.readback_fence));

    VkCommandBuffer cmd = slot.readback_cmd;
    REQ_VK(dev.dt.resetCommandBuffer(cmd, 0));

    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    REQ_VK(dev.dt.beginCommandBuffer(cmd, &bi));

    int total_views = config_.batch_size * SHM_NUM_CAMERAS;
    int cols = std::ceil(std::sqrt((float)total_views));

    VkImageMemoryBarrier barrier1{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier1.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier1.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier1.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier1.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier1.image = slot.color_image->image;
    barrier1.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier1);

    for (int i = 0; i < total_views; ++i) {
        int gx = i % cols, gy = i / cols;
        VkBufferImageCopy region{};
        region.imageOffset = {(int32_t)(gx * config_.frame_width),
                             (int32_t)(gy * config_.frame_height), 0};
        region.imageExtent = {(uint32_t)config_.frame_width,
                             (uint32_t)config_.frame_height, 1};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        dev.dt.cmdCopyImageToBuffer(cmd, slot.color_image->image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, slot.staging_bufs[i].buffer, 1, &region);
    }

    VkImageMemoryBarrier barrier2{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier2.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier2.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier2.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    barrier2.image = slot.color_image->image;
    barrier2.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    dev.dt.cmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier2);

    REQ_VK(dev.dt.endCommandBuffer(cmd));

    slot.step_id = step_id;
    slot.state = SwapSlot::State::READBACK_PENDING;

    VkPipelineStageFlags waitStageReadback = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &slot.render_complete_semaphore;
    si.pWaitDstStageMask = &waitStageReadback;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    REQ_VK(dev.dt.queueSubmit(render_context_->renderQueue, 1, &si, slot.readback_fence));

    return true;
}

void BatchRenderer::ReadbackThreadFn() {
    while (readback_running_) {
        for (int s = 0; s < SWAP_COUNT; ++s) {
            SwapSlot& slot = swap_slots_[s];
            if (slot.state != SwapSlot::State::READBACK_PENDING) continue;

            VkResult r = device_->dt.waitForFences(device_->hdl, 1,
                &slot.readback_fence, VK_TRUE, 1000000);
            if (r != VK_SUCCESS) continue;

            int total_views = config_.batch_size * SHM_NUM_CAMERAS;
            std::vector<VkMappedMemoryRange> ranges;
            for (int i = 0; i < total_views; ++i) {
                VkMappedMemoryRange mr{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
                mr.memory = slot.staging_bufs[i].getMemHdl();
                mr.size = VK_WHOLE_SIZE;
                ranges.push_back(mr);
            }
            device_->dt.invalidateMappedMemoryRanges(
                device_->hdl, (uint32_t)ranges.size(), ranges.data());

            std::vector<FrameObservation> obs(total_views);
            for (int i = 0; i < total_views; ++i) {
                obs[i] = {
                    .data = (const uint8_t*)slot.staging_bufs[i].ptr,
                    .width = (uint32_t)config_.frame_width,
                    .height = (uint32_t)config_.frame_height,
                    .stride_bytes = (uint32_t)(config_.frame_width * 4),
                    .total_bytes = (size_t)(config_.frame_width * config_.frame_height * 4)
                };
            }

            if (readback_callback_) {
                readback_callback_(slot.step_id, obs);
            }

            {
                std::lock_guard<std::mutex> lk(swap_mutex_);
                slot.state = SwapSlot::State::FREE;
            }
            swap_cv_.notify_all();
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void BatchRenderer::StartReadbackThread() {
    readback_running_ = true;
    readback_thread_ = std::thread(&BatchRenderer::ReadbackThreadFn, this);
}

void BatchRenderer::StopReadbackThread() {
    readback_running_ = false;
    if (readback_thread_.joinable()) {
        readback_thread_.join();
    }
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




