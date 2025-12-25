#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp> // 支持 glm::translate, glm::scale
#include <glm/gtc/quaternion.hpp>     // 支持 glm::quat, glm::mat4_cast
#include <glm/gtc/type_ptr.hpp>       
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "builtin.h"
#include "utils.h"

namespace mujoco {
namespace mjbatch {

using float3 = glm::vec3;
using float4 = glm::vec4;
using mat4 = glm::mat4;

// Material properties for rendering
struct Material {
    float4 rgba;           // Base color (RGBA)
    float specular;       // Specular color
    float shininess;       // Shininess factor
    float emission;        // Emission factor
    float reflectance;     // Reflectance factor
    int texture_id;        // Texture ID (-1 if no texture)
    
    Material() : rgba(1, 1, 1, 1), specular(0.5f), 
                 shininess(32.0f), emission(0.0f),reflectance(0.0f), texture_id(-1) {}
};

// Camera/view information
struct CameraInfo {
    // 基础属性
    std::string name;       // 摄像机名称 (from xml)
    int id = -1;            // MuJoCo 中的 ID
    
    // 物理属性 (World Space)
    glm::vec3 position;     // 世界坐标位置
    glm::vec3 forward;      // 前向向量 (-Z in camera space)
    glm::vec3 up;           // 上向量 (+Y in camera space)
    glm::vec3 right;        // 右向量 (+X in camera space)

    // 透视属性
    float fov_y_degrees;    // 垂直视场角
    float near_plane;       // 近裁剪面
    float far_plane;        // 远裁剪面
    float aspect_ratio;     // 宽高比 (W/H)

    // 渲染矩阵 (Vulkan ready)
    glm::mat4 view_matrix;
    glm::mat4 proj_matrix;
    glm::mat4 view_proj_matrix;
};
// -----------------------------------------------------------------------------
// Uniform buffer structures
// -----------------------------------------------------------------------------
struct CameraUBO {
    glm::mat4 view_proj;
    glm::vec3 position;
    float padding1;  // Padding to align to 16 bytes
};

struct LightInfo {
    glm::vec3 position;
    uint32_t type;  // 0: spot, 1: directional, 2: point
    
    glm::vec3 direction;
    float range;
    
    glm::vec3 ambient;
    float cutoff;
    
    glm::vec3 diffuse;
    float exponent;
    
    glm::vec3 specular;
    float bulbRadius;
    
    glm::vec3 attenuation; // x: constant, y: linear, z: quadratic
    float intensity;
    
    uint32_t castShadow;
    float padding[3];  // 调整为2个float来补齐
    
    // 转换函数
    static LightInfo fromMjvLight(const mjvLight& mjLight) {
        LightInfo light;
        
        light.position = glm::vec3(mjLight.pos[0], mjLight.pos[1], mjLight.pos[2]);
        light.direction = glm::vec3(mjLight.dir[0], mjLight.dir[1], mjLight.dir[2]);
        
        // 光照类型转换
        switch(mjLight.type) {
            case 0: light.type = 0; break; // spot
            case 1: light.type = 1; break; // directional  
            case 2: light.type = 2; break; // point
            default: light.type = 1; break;
        }
        
        // 颜色属性
        light.ambient = glm::vec3(mjLight.ambient[0], mjLight.ambient[1], mjLight.ambient[2]);
        light.diffuse = glm::vec3(mjLight.diffuse[0], mjLight.diffuse[1], mjLight.diffuse[2]);
        light.specular = glm::vec3(mjLight.specular[0], mjLight.specular[1], mjLight.specular[2]);
        
        // quadratic model
        light.attenuation = glm::vec3(mjLight.attenuation[0], mjLight.attenuation[1], mjLight.attenuation[2]);
        light.range = mjLight.range;
        
        // 聚光灯参数
        light.cutoff = mjLight.cutoff; 
        light.exponent = mjLight.exponent;
        
        // 阴影参数
        light.bulbRadius = mjLight.bulbradius;
        light.intensity = mjLight.intensity;
        light.castShadow = mjLight.castshadow;
        
        return light;
    }
};

// [MODIFIED] Render-ready drawable object
// 不再持有几何数据指针，而是持有Key(name)用于全局查找
struct Drawable {
    // Identify the geometry in the Global Buffer
    // 对于 Mesh，这是 xml 中的 mesh name   
    // 对于 Builtin，这是预定义的名称 (e.g., "__builtin_box")
    std::string global_mesh_name; 

    mat4 transform;        // Model matrix (World Space)
    Material material;     // Material properties
    
    // Debug / Logic info
    int geom_id;           // Original MuJoCo geom ID
    int32_t mesh_id;       // -1 if not a mesh (primitive)
    bool visible;          // Visibility flag
};



// Texture information
struct TextureInfo {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;  // Raw texture data (RGBA)
    int texture_id;              // MuJoCo texture ID
    int type; // 0：2d 1：cube 2：skybox
    std::string name; // texture name
};

// Main Scene class - provides render-ready data
class Scene {
public:
    Scene(const mjModel* model, const mjvScene* scene);
    ~Scene();

    // Update scene from MuJoCo data
    void Update(const mjModel* model, const mjvScene* scene);

    // Get render-ready data
// Accessors
    const std::vector<Drawable>& GetDrawables() const { return drawables_; }
    const CameraInfo& GetCamera() const { return camera_info_; }
    const std::vector<LightInfo>& GetLights() const { return lights_; }
    // 纹理通常在 Init 阶段由 BatchRenderer 统一处理，Scene 中保留它是为了方便查找
    const std::vector<TextureInfo>& GetTextures() const { return textures_; }

    void UpdateCameraFromSimulation(const mjModel* m, const mjData* d, int cam_id, float aspect_ratio);
    int FindCameraID(const mjModel* m, const char* cam_name);
    CameraUBO GetCameraUBO() const {
        CameraUBO ubo = {}; // 初始化为 0

        // 1. 矩阵直接拷贝 (GLM 默认列主序，符合 HLSL/GLSL 默认行为)
        ubo.view_proj = camera_info_.view_proj_matrix;

        // 2. 向量拷贝并填充 Padding
        ubo.position = camera_info_.position;

        return ubo;
    }
private:
    const mjModel* model_;
    
    // Render-ready data
    std::vector<Drawable> drawables_;
    CameraInfo camera_info_;
    std::vector<LightInfo> lights_;
    std::vector<TextureInfo> textures_;
    
    
    // Internal extraction functions
    void ExtractGeometries(const mjModel* model, const mjvScene* scene);
    void ExtractLight(const mjvScene* scene);
    void ExtractMaterials(const mjModel* model, const mjvScene* scene);
    void ExtractTextures(const mjModel* model);
    
    // Helper to extract material from geom
    Material ExtractMaterialFromGeom(const mjModel* model, const mjvGeom* geom);
    
    // Helper to build transform matrix
    mat4 BuildTransform(const mjvGeom* geom);


};

}}  // namespace mujoco::mjbatch
