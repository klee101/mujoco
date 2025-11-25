#include "scene.h"
#include <cmath>
#include <cstring>
#include <iostream>

namespace mujoco {
namespace mjbatch {
namespace {

// 1. 将 MuJoCo 的 geom->type 转换为 ObjectManager 的 ShapeType
ShapeType GetBatchShapeType(int mj_geom_type) {
    switch (mj_geom_type) {
        case mjGEOM_PLANE:    return kPlane;
        case mjGEOM_HFIELD:   return kNumShapes; // 特殊处理
        case mjGEOM_SPHERE:   return kSphere;
        case mjGEOM_CAPSULE:  return kCapsule;      // MuJoCo Capsule 通常用 Tube + 半圆盖模拟，或者用胶囊体
        case mjGEOM_ELLIPSOID:return kSphere;    // 椭球通过缩放球体实现
        case mjGEOM_CYLINDER: return kCylinder;      // 这里假设 kTube 对应圆柱体
        case mjGEOM_BOX:      return kBox;
        case mjGEOM_MESH:     return kBox; // 特殊处理
        default:              return kBox;       // 默认回退
    }
}

// 2. 根据几何类型解析 geom->size 并计算缩放向量
// 假设 float3 是你们 utils.h 中定义的结构体
float3 GetGeomScale(int type, const float* size) {
    // size 数组在 MuJoCo 中有不同的含义：
    switch (type) {
        case mjGEOM_BOX:
            // Box: size[0]=x半长, size[1]=y半长, size[2]=z半长
            return float3{size[0], size[1], size[2]};
        
        case mjGEOM_SPHERE:
            // Sphere: size[0]=半径. 缩放 xyz 均为半径
            return float3{size[0], size[1], size[2]};
            
        case mjGEOM_ELLIPSOID:
            // Ellipsoid: size[0], size[1], size[2] 分别为 x,y,z 半径
            return float3{size[0], size[1], size[2]};

        case mjGEOM_CAPSULE:
        case mjGEOM_CYLINDER:
            return float3{size[0], size[1], size[2]}; 
            
        case mjGEOM_PLANE:
            // Plane: size[0]=X范围, size[1]=Y范围. 如果 size[0]>0 为有限平面
            if (size[0] > 0) return float3{size[0], size[1], size[2]};
            return float3{1000.0f, 1000.0f, size[2]}; // 无限平面给一个大数值

        default:
            return float3{size[0], size[1], size[2]};
    }
}

} // namespace anonymous
// ============================================================================
// Scene Implementation
// ============================================================================

Scene::Scene(const mjModel* model, const mjvScene* scene)
    : model_(model) {
    object_manager_ = std::make_unique<ObjectManager>(model, nullptr);
    ExtractGeometries(model, scene);
    ExtractLight(scene);
    ExtractMaterials(model, scene);
}

Scene::~Scene() = default;

// Optimized Update - only updates transforms
void Scene::Update(const mjModel* model, const mjvScene* scene, const mjData* data) {
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

        if(geom->type == mjGEOM_MESH || geom->type == mjGEOM_HFIELD) {
            geom_idx++;
            continue;
        }
        
        // Apply size scaling to transform
        float3 size = ReadFloat3(geom->size);
        size = GetGeomScale(geom->type, geom->size);
        drawables_[geom_idx].transform = drawables_[geom_idx].transform * scaling(size);
        
        geom_idx++;
    }

}

// Optimized ExtractGeometries - skip geometry buffer operations during update
// Helper to read quaternion from MuJoCo array to glm::quat
glm::quat ReadQuat(const double* q) {
    // MuJoCo quat is [w, x, y, z]
    // GLM constructor is (w, x, y, z)
    return glm::quat((float)q[0], (float)q[1], (float)q[2], (float)q[3]);
}

void Scene::ExtractGeometries(const mjModel* model, const mjvScene* scene) {
    drawables_.clear();
    drawables_.reserve(scene->ngeom);
    
    for (int i = 0; i < scene->ngeom; ++i) {
        const mjvGeom* geom = scene->geoms + i;
        // 1. Skip unsupported types / Invisible logic
        if (geom->type == mjGEOM_FLEX || geom->type == mjGEOM_SKIN || 
            geom->type == mjGEOM_ARROW || geom->type == mjGEOM_LABEL) continue;

        Material mat = ExtractMaterialFromGeom(model, geom);
        if (mat.rgba.a < 0.01f) continue; 

        Drawable drawable;
        drawable.geom_id = i;
        drawable.material = mat;
        drawable.visible = true;
        
        // [Step A] 构建基础的世界变换矩阵 (Geom Transform)
        // 这代表了物体在世界坐标系中的位置和姿态
        drawable.transform = BuildTransform(geom); 
        
        // 2. Geometry Handling & Asset Transformation
        if (geom->type == mjGEOM_MESH) {
            int mesh_id = geom->dataid;
            // FIX: mesh_id dont need to *2 !!!
            const GeometryBuffers* mesh_ptr = object_manager_->GetMeshBuffer(mesh_id);
            
            if (mesh_ptr) {
                drawable.geometry = mesh_ptr; 
            } else {
                continue; // Mesh data missing
            }
        } 
        else if (geom->type == mjGEOM_HFIELD) {
             int hfield_id = geom->dataid;
             const GeometryBuffers* hfield_ptr = object_manager_->GetHeightFieldBuffer(hfield_id);
             if (hfield_ptr) {
                 drawable.geometry = hfield_ptr;
                 // HeightField 通常不需要 mesh_pos/quat 这种资源变换，
                 // 它的空间位置由 geom->pos 决定，尺寸由 model->hfield_size 决定(需在 GeometryBuilder 处理或在此缩放)
             } else {
                 continue;
             }
        } 
        else {
            // --- Primitives (Box, Sphere, etc.) ---
            mjbatch::ShapeType shape_type = GetBatchShapeType(geom->type);
            const GeometryBuffers* shape_ptr = object_manager_->GetShapeBuffer(shape_type);
            
            if (shape_ptr) {
                drawable.geometry = shape_ptr;
                
                // 基础几何体只有 Scale (size) 变换，没有 Asset Offset
                float3 scale_vec = GetGeomScale(geom->type, geom->size);
                if (scale_vec.x > 1e-6f) {
                    drawable.transform = drawable.transform * scaling(scale_vec);
                }
            } else {
                continue;
            }
        }
        
        drawables_.push_back(std::move(drawable));
    }
}

// Helper: Find camera ID by name
int Scene::FindCameraID(const mjModel* m, const char* cam_name) {
    int id = mj_name2id(m, mjOBJ_CAMERA, cam_name);
    if (id == -1) {
        std::cerr << "Warning: Camera '" << cam_name << "' not found." << std::endl;
        std::cerr << "Available cameras:" << std::endl;
        for (int i = 0; i < m->ncam; ++i) {
            std::cerr << "  - " << &m->names[m->name_camadr[i]] << std::endl;
        }
    }
    return id;
}

// Main Function: Update camera info from simulation state
void Scene::UpdateCameraFromSimulation(const mjModel* m, const mjData* d, int cam_id, float aspect_ratio) {
    if (cam_id < 0 || cam_id >= m->ncam) return;

    // 1. Basic Info
    camera_info_.id = cam_id;
    // 获取名称 (MuJoCo name string array logic)
    if (m->names) {
        camera_info_.name = std::string(&m->names[m->name_camadr[cam_id]]);
    }

    // 2. Extract Dynamic Transform from mjData (Crucial for attached cameras!)
    // d->cam_xpos 是 3 * ncam 的数组
    // d->cam_xmat 是 9 * ncam 的数组 (3x3 旋转矩阵)
    
    // Position
    camera_info_.position = glm::vec3(
        d->cam_xpos[3 * cam_id + 0],
        d->cam_xpos[3 * cam_id + 1],
        d->cam_xpos[3 * cam_id + 2]
    );

    // Rotation Matrix (MuJoCo is Row-Major packed in a 1D array for xmat?)
    // No, mjData matrices are strictly 9 numbers per object.
    // Constructing Basis Vectors from Rotation Matrix
    // X-axis (Right)
    camera_info_.right = glm::vec3(
        d->cam_xmat[9 * cam_id + 0],
        d->cam_xmat[9 * cam_id + 3],
        d->cam_xmat[9 * cam_id + 6]
    );
    // Y-axis (Up) -> This is the local Up
    camera_info_.up = glm::vec3(
        d->cam_xmat[9 * cam_id + 1],
        d->cam_xmat[9 * cam_id + 4],
        d->cam_xmat[9 * cam_id + 7]
    );
    // Z-axis (Backwards) -> Camera looks towards -Z
    glm::vec3 backward = glm::vec3(
        d->cam_xmat[9 * cam_id + 2],
        d->cam_xmat[9 * cam_id + 5],
        d->cam_xmat[9 * cam_id + 8]
    );
    camera_info_.forward = -backward;

    // 3. Projection Parameters from mjModel
    // mjModel stores FOV in degrees (Y-axis usually)
    camera_info_.fov_y_degrees = m->cam_fovy[cam_id];
    
    // Near/Far are typically generic in MuJoCo visual settings, 
    // usually retrieved from m->vis.map.znear / zfar if not manually managed.
    camera_info_.near_plane = m->vis.map.znear; 
    camera_info_.far_plane = m->vis.map.zfar;
    camera_info_.aspect_ratio = aspect_ratio;

    // 4. Build View Matrix
    // glm::lookAt(eye, center, up)
    // center = eye + forward
    camera_info_.view_matrix = glm::lookAt(
        camera_info_.position,
        camera_info_.position + camera_info_.forward,
        camera_info_.up
    );

    // 5. Build Projection Matrix
    camera_info_.proj_matrix = glm::perspective(
        glm::radians(camera_info_.fov_y_degrees),
        aspect_ratio,
        camera_info_.near_plane,
        camera_info_.far_plane
    );

    // 6. Vulkan Coordinate Fix
    // OpenGL: Y-up, Z-backward (Clip: Z -1 to 1)
    // Vulkan: Y-down, Z-forward  (Clip: Z 0 to 1)
    // GLM is designed for OpenGL. We need to flip the Y axis in the projection matrix.
    camera_info_.proj_matrix[1][1] *= -1.0f;

    // Optional: If you use standard GLM perspective, it maps Z to [-1, 1].
    // Ideally for Vulkan you want [0, 1]. 
    // You can usually fix this in the pipeline creation (minDepth/maxDepth) 
    // or by pre-multiplying a correction matrix, but generally [1][1] *= -1 is the most critical fix.

    // Combined
    camera_info_.view_proj_matrix = camera_info_.proj_matrix * camera_info_.view_matrix;
}

void Scene::ExtractLight(const mjvScene* scene){
    int nlight = scene->nlight;
    for (int i = 0; i < nlight; ++i) {
        LightInfo curlight = LightInfo::fromMjvLight(scene->lights[i]);
        lights_.emplace_back(curlight);
    }
}

void Scene::ExtractMaterials(const mjModel* model, const mjvScene* scene) {
    // Materials are extracted per-geom in ExtractGeometries
    // This function can be used for global material extraction if needed
}

Material Scene::ExtractMaterialFromGeom(const mjModel* model, const mjvGeom* geom) {
Material mat;
    
    // 1. 提取基础属性
    mat.rgba = ReadFloat4(geom->rgba);
    mat.emission = geom->emission;
    mat.shininess = geom->shininess;
    mat.reflectance = geom->reflectance;
    float specular_val = geom->specular;
    mat.specular = float3(specular_val, specular_val, specular_val);
    
    // A. 排除装饰物 (mjCAT_DECOR)
    // Decor 通常是接触点、力箭头等，它们在 model->geom_group 中没有对应数据
    // B. 确认对象类型是 mjOBJ_GEOM
    // 只有 mjOBJ_GEOM 类型的 objid 才对应 model->geom_xxx 数组
    bool is_model_geometry = (geom->category != mjCAT_DECOR) && 
                             (geom->objtype == mjOBJ_GEOM);

    if (is_model_geometry && geom->objid >= 0 && geom->objid < model->ngeom) {
        
        // 安全地获取 Group ID
        int group_id = model->geom_group[geom->objid];

        // printf("Geom ID %d has Group ID %d\n", geom->objid, group_id);
        // 逻辑：Group 0 是碰撞体，将其设为完全透明 (Alpha = 0)
        // 渲染器后续可以根据 Alpha=0 剔除此物体
        if (group_id == 0) {
            mat.rgba.a = 0.0f;
            // printf("Hiding Collision Geom: objid %d\n", geom->objid);
        }
    }
    // =========================================================

    // 2. 初始化纹理字段
    mat.texture_id = -1; 

    // 如果材质已透明，直接返回（性能优化）
    if (mat.rgba.a <= 0.001f) {
        return mat; 
    }

    // printf("--- Geom ID %d (MatID: %d) ---\n", geom->objid, geom->matid); 

    // 3. 检查材质 ID 是否有效
    if (geom->matid >= 0 && geom->matid < model->nmat) {
        
        // 4. 遍历所有纹理角色，查找基色纹理 (RGB/RGBA)
        for (int role_idx = 0; role_idx < mjNTEXROLE; ++role_idx) {
            
            int texid_idx = geom->matid * mjNTEXROLE + role_idx;
            
            // 检查数组边界
            if (model->mat_texid && texid_idx >= 0 && texid_idx < model->nmat * mjNTEXROLE) {
                int texid = model->mat_texid[texid_idx];
                      
                // 5. 如果找到有效 ID
                if (texid >= 0 && texid < model->ntex) {
                    
                    // 6. 优先选择 RGB/RGBA 作为基色纹理
                    if (role_idx == mjTEXROLE_RGB || role_idx == mjTEXROLE_RGBA) {
                        mat.texture_id = texid;
                        
                        // int type = model->tex_type[texid];
                        // printf("  --> FOUND Base Texture! Role: %d, Type: %d (ID=%d)\n", role_idx, type, mat.texture_id);
                        
                        goto end_of_search; 
                    }
                }
            }
        }
    }
    
end_of_search:
    // printf("--- Result: Final texture_id=%d ---\n", mat.texture_id);
    return mat;
}

void Scene::ExtractTextures(const mjModel* model) {
    textures_.clear();
    textures_.reserve(model->ntex);
    
    for (int i = 0; i < model->ntex; ++i) {
        TextureInfo tex_info;
        tex_info.texture_id = i;
        
        // 注意：MuJoCo 的 tex_type 可能不是 1/2，您应该使用 mjTEXTURE_2D, mjTEXTURE_CUBE, mjTEXTURE_SKYBOX
        // 但根据您的 Log，这里暂时沿用 Type 1/2
        tex_info.type = model->tex_type[i];
        
        // Get texture dimensions from model arrays
        tex_info.width = model->tex_width[i];
        tex_info.height = model->tex_height[i];
        int nchannel = model->tex_nchannel[i];
        
        // [HARD FIX] 针对 Cube Map，保持原始通道数 (3)；其他类型转换为 RGBA (4)
        if (tex_info.type == mjTEXTURE_CUBE || tex_info.type == mjTEXTURE_SKYBOX) {
            // Cube Map (或 Skybox) 必须保持原始通道数 (通常为 3)
            // 否则在 CreateCubeTexture 中处理时会出错
            tex_info.channels = nchannel; 
        } else {
            // 2D 纹理统一转换为 RGBA (4) 以简化渲染器处理
            printf("orig tex id %d: nchannel=%d converted to RGBA(4)\n", i, nchannel);
            tex_info.channels = 4;
        }
        
        // Get texture data address
        int tex_data_adr = model->tex_adr[i];
        if (tex_data_adr < 0 || tex_data_adr >= model->ntexdata) {
            continue;
        }
        
        // Calculate data size based on the final planned channel count
        size_t pixel_count = static_cast<size_t>(tex_info.width) * tex_info.height;
        size_t data_size = pixel_count * tex_info.channels;
        tex_info.data.resize(data_size);
        
        // Copy and convert texture data
        const unsigned char* tex_data = model->tex_data + tex_data_adr;
        
        // ***************************************************************
        // [LOGIC FIX] 仅在目标是 RGBA (4通道) 且源不是 4通道时才执行转换
        // ***************************************************************
        if (tex_info.channels == 4) { 
            // 目标是 RGBA (用于 2D 纹理)
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
        } else if (tex_info.channels == 3) {
            // 目标是 RGB (用于 Cube Map)
             if (nchannel == 3) {
                 // RGB -> RGB (direct copy)
                 std::memcpy(tex_info.data.data(), tex_data, pixel_count * 3);
             } else {
                 // 如果 Cube Map 不是 RGB (3 通道)，这可能是一个 MuJoCo/XML 错误
                 // 在此打印警告或抛出错误
                 printf("Warning: Cube Map ID %d has %d channels, but 3 are required. Skipping or data will be corrupted.\n", i, nchannel);
                 continue;
             }
        }
        
        textures_.push_back(std::move(tex_info));
    }

    printf("Extracted %zu textures from model.\n", textures_.size());
    for(const auto& tex : textures_) {
        printf(" Texture ID %d: %dx%d, Channels=%d, Type=%d\n",
               tex.texture_id, tex.width, tex.height, tex.channels, tex.type);
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
        count += drawable.geometry->GetVertexCount();
    }
    return count;
}

size_t Scene::GetTotalIndexCount() const {
    size_t count = 0;
    for (const auto& drawable : drawables_) {
        count += drawable.geometry->GetIndexCount();
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
        for (const auto& vertex : drawable.geometry->vertices) {
            Vertex v = vertex;
            // Apply material color
            v.color = drawable.material.rgba;
            vertices.push_back(v);
        }
        
        // Add indices with offset
        for (uint32_t idx : drawable.geometry->indices) {
            indices.push_back(idx);
        }
        
        vertex_offset += static_cast<uint32_t>(drawable.geometry->vertices.size());
        
        // examine drawable vertex index tex and material
        // printf("Drawable geom_id=%d: vtx=%zu idx=%zu tex_id=%d rgba=(%.2f, %.2f, %.2f, %.2f)\n",
        //        drawable.geom_id,
        //        drawable.geometry.GetVertexCount(),
        //        drawable.geometry.GetIndexCount(),
        //        drawable.material.texture_id,
        //        drawable.material.rgba.x,
        //        drawable.material.rgba.y,
        //        drawable.material.rgba.z,
        //        drawable.material.rgba.w);
    }
}

}}  // namespace mujoco::mjbatch
