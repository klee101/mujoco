#include "scene.h"
#include <cmath>
#include <cstring>

namespace mujoco {
namespace mjbatch {

// ============================================================================
// Scene Implementation
// ============================================================================

Scene::Scene(const mjModel* model, const mjvScene* scene)
    : model_(model) {
    object_manager_ = std::make_unique<ObjectManager>(model, nullptr);
    ExtractGeometries(model, scene);
    ExtractCamera(scene);
    ExtractMaterials(model, scene);
    ExtractTextures(model);
}

Scene::~Scene() = default;

// Optimized Update - only updates transforms
void Scene::Update(const mjModel* model, const mjvScene* scene) {
    model_ = model;
    
    // Only update transforms for existing geometries
    int geom_idx = 0;
    for (int i = 0; i < scene->ngeom && geom_idx < drawables_.size(); ++i) {
        const mjvGeom* geom = scene->geoms + i;
        
        // Skip flex and skin geometries (same as in ExtractGeometries)
        if (geom->type == mjGEOM_FLEX || geom->type == mjGEOM_SKIN) {
            continue;
        }
        
        // Update only the transform
        drawables_[geom_idx].transform = BuildTransform(geom);
        
        // Apply size scaling to transform
        float3 size = ReadFloat3(geom->size);
        drawables_[geom_idx].transform = drawables_[geom_idx].transform * scaling(size);
        
        geom_idx++;
    }
}

// Optimized ExtractGeometries - skip geometry buffer operations during update
void Scene::ExtractGeometries(const mjModel* model, const mjvScene* scene) {
    drawables_.clear();
    drawables_.reserve(scene->ngeom);
    
    for (int i = 0; i < scene->ngeom; ++i) {
        const mjvGeom* geom = scene->geoms + i;
        
        // Skip flex and skin geometries for now
        if (geom->type == mjGEOM_FLEX || geom->type == mjGEOM_SKIN) {
            continue;
        }
        
        Drawable drawable;
        drawable.geom_id = i;
        drawable.visible = true;
        drawable.transform = BuildTransform(geom);
        drawable.material = ExtractMaterialFromGeom(model, geom);
        
        // Get geometry buffers based on type
        if (geom->type == mjGEOM_MESH) {
            int mesh_id = geom->dataid;
            if (mesh_id >= 0 && mesh_id < model->nmesh) {
                const GeometryBuffers* mesh_buf = object_manager_->GetMeshBuffer(mesh_id * 2);
                if (mesh_buf) {
                    drawable.geometry = *mesh_buf;
                } else {
                    drawable.geometry = GeometryBuilder::BuildMesh(model, mesh_id);
                }
            }
        } else if (geom->type == mjGEOM_HFIELD) {
            int hfield_id = geom->dataid;
            if (hfield_id >= 0 && hfield_id < model->nhfield) {
                const GeometryBuffers* hfield_buf = object_manager_->GetHeightFieldBuffer(hfield_id);
                if (hfield_buf) {
                    drawable.geometry = *hfield_buf;
                } else {
                    drawable.geometry = GeometryBuilder::BuildHeightField(model, hfield_id);
                }
            }
        } else {
            drawable.geometry = GeometryBuilder::BuildFromType(geom->type, model);
        }
        
        // Apply size scaling to transform
        float3 size = ReadFloat3(geom->size);
        drawable.transform = drawable.transform * scaling(size);
        
        drawables_.push_back(std::move(drawable));
    }
}

void Scene::ExtractCamera(const mjvScene* scene) {
    // Use mjvGLCamera from scene (average left and right for mono, or use left)
    const mjvGLCamera* glcam = &scene->camera[0];
    
    // Extract camera position and orientation
    float3 pos = ReadFloat3(glcam->pos);
    float3 forward = ReadFloat3(glcam->forward);
    float3 up = ReadFloat3(glcam->up);
    
    camera_info_.position = pos;
    camera_info_.forward = forward;
    camera_info_.up = up;
    
    // Extract frustum parameters
    camera_info_.near_plane = glcam->frustum_near;
    camera_info_.far_plane = glcam->frustum_far;
    
    // Calculate FOV from frustum
    float frustum_height = glcam->frustum_top - glcam->frustum_bottom;
    camera_info_.fov = 2.0f * std::atan(frustum_height / (2.0f * glcam->frustum_near)) * 180.0f / M_PI;
    
    // Build view matrix (look-at)
    float3 target = pos + forward;
    camera_info_.view_matrix = glm::lookAt(pos, target, up);
    
    // Build projection matrix
    float halfwidth = glcam->frustum_width ? glcam->frustum_width 
                    : 0.5f * (glcam->frustum_top - glcam->frustum_bottom);
    
    if (glcam->orthographic) {
        // Orthographic projection
        camera_info_.proj_matrix = glm::ortho(
            glcam->frustum_center - halfwidth,
            glcam->frustum_center + halfwidth,
            glcam->frustum_bottom,
            glcam->frustum_top,
            glcam->frustum_near,
            glcam->frustum_far
        );
    } else {
        // Perspective projection
        camera_info_.proj_matrix = glm::frustum(
            glcam->frustum_center - halfwidth,
            glcam->frustum_center + halfwidth,
            glcam->frustum_bottom,
            glcam->frustum_top,
            glcam->frustum_near,
            glcam->frustum_far
        );
    }
    // FIX: Flip Y-axis for Vulkan (OpenGL has Y-up, Vulkan has Y-down in NDC)
    camera_info_.proj_matrix[1][1] *= -1.0f;

    // Combined view-projection
    camera_info_.view_proj_matrix = camera_info_.proj_matrix * camera_info_.view_matrix;
}

void Scene::ExtractMaterials(const mjModel* model, const mjvScene* scene) {
    // Materials are extracted per-geom in ExtractGeometries
    // This function can be used for global material extraction if needed
}

Material Scene::ExtractMaterialFromGeom(const mjModel* model, const mjvGeom* geom) {
    Material mat;
    
    // Extract RGBA color directly from geom
    mat.rgba = ReadFloat4(geom->rgba);
    
    // Extract material properties directly from geom (mjvGeom has these directly)
    mat.emission = geom->emission;
    mat.shininess = geom->shininess;
    
    // Specular is a single value in mjvGeom, convert to RGB
    float specular_val = geom->specular;
    mat.specular = float3(specular_val, specular_val, specular_val);
    
    // Check if geom references a material in the model
    mat.texture_id = -1;  // Default: no texture
    if (geom->matid >= 0 && geom->matid < model->nmat) {
        // Get texture ID from material (mjNTEXROLE is the number of texture roles)
        // For now, use the first texture role (usually diffuse)
        int texid_idx = geom->matid * mjNTEXROLE;
        if (model->mat_texid && texid_idx >= 0 && texid_idx < model->nmat * mjNTEXROLE) {
            int texid = model->mat_texid[texid_idx];
            if (texid >= 0 && texid < model->ntex) {
                mat.texture_id = texid;
            }
        }
    }
    
    return mat;
}

void Scene::ExtractTextures(const mjModel* model) {
    textures_.clear();
    textures_.reserve(model->ntex);
    
    for (int i = 0; i < model->ntex; ++i) {
        TextureInfo tex_info;
        tex_info.texture_id = i;
        
        // Get texture dimensions from model arrays
        tex_info.width = model->tex_width[i];
        tex_info.height = model->tex_height[i];
        int nchannel = model->tex_nchannel[i];
        tex_info.channels = 4;  // We'll convert to RGBA
        
        // Get texture data address
        int tex_data_adr = model->tex_adr[i];
        if (tex_data_adr < 0 || tex_data_adr >= model->ntexdata) {
            // No texture data, skip
            continue;
        }
        
        // Calculate data size
        size_t pixel_count = static_cast<size_t>(tex_info.width) * tex_info.height;
        size_t data_size = pixel_count * tex_info.channels;
        tex_info.data.resize(data_size);
        
        // Copy and convert texture data
        const unsigned char* tex_data = model->tex_data + tex_data_adr;
        
        if (nchannel == 3) {
            // RGB -> RGBA
            for (size_t j = 0; j < pixel_count; ++j) {
                tex_info.data[j * 4 + 0] = tex_data[j * 3 + 0];
                tex_info.data[j * 4 + 1] = tex_data[j * 3 + 1];
                tex_info.data[j * 4 + 2] = tex_data[j * 3 + 2];
                tex_info.data[j * 4 + 3] = 255;  // Alpha
            }
        } else if (nchannel == 4) {
            // RGBA -> RGBA (direct copy)
            std::memcpy(tex_info.data.data(), tex_data, pixel_count * 4);
        } else if (nchannel == 1) {
            // Grayscale -> RGBA
            for (size_t j = 0; j < pixel_count; ++j) {
                unsigned char gray = tex_data[j];
                tex_info.data[j * 4 + 0] = gray;
                tex_info.data[j * 4 + 1] = gray;
                tex_info.data[j * 4 + 2] = gray;
                tex_info.data[j * 4 + 3] = 255;
            }
        }
        
        textures_.push_back(std::move(tex_info));
    }
}

mat4 Scene::BuildTransform(const mjvGeom* geom) {
    mat3 rotation = ReadMat3(geom->mat);
    float3 translation = ReadFloat3(geom->pos);
    return fromRotationTranslation(rotation, translation);
}

size_t Scene::GetTotalVertexCount() const {
    size_t count = 0;
    for (const auto& drawable : drawables_) {
        count += drawable.geometry.GetVertexCount();
    }
    return count;
}

size_t Scene::GetTotalIndexCount() const {
    size_t count = 0;
    for (const auto& drawable : drawables_) {
        count += drawable.geometry.GetIndexCount();
    }
    return count;
}

void Scene::GetCombinedBuffers(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) const {
    vertices.clear();
    indices.clear();
    
    vertices.reserve(GetTotalVertexCount());
    indices.reserve(GetTotalIndexCount());
    
    uint32_t vertex_offset = 0;
    
    for (const auto& drawable : drawables_) {
        if (!drawable.visible) continue;
        
        // Add vertices with material color applied
        for (const auto& vertex : drawable.geometry.vertices) {
            Vertex v = vertex;
            // Apply material color
            v.color = drawable.material.rgba;
            vertices.push_back(v);
        }
        
        // Add indices with offset
        for (uint32_t idx : drawable.geometry.indices) {
            indices.push_back(idx);
        }
        
        vertex_offset += static_cast<uint32_t>(drawable.geometry.vertices.size());

        printf("Drawable geom_id=%d: vertices=%zu, indices=%zu\n",
               drawable.geom_id,
               drawable.geometry.GetVertexCount(),
               drawable.geometry.GetIndexCount());
    }
}

}}  // namespace mujoco::mjbatch
