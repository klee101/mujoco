#pragma once

#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "object_manager.h"
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
    float3 specular;       // Specular color
    float shininess;       // Shininess factor
    float emission;        // Emission factor
    int texture_id;        // Texture ID (-1 if no texture)
    
    Material() : rgba(1, 1, 1, 1), specular(0.5f, 0.5f, 0.5f), 
                 shininess(32.0f), emission(0.0f), texture_id(-1) {}
};

// Camera/view information
struct CameraInfo {
    mat4 view_matrix;       // View matrix
    mat4 proj_matrix;      // Projection matrix
    mat4 view_proj_matrix; // Combined view-projection
    float3 position;       // Camera position
    float3 forward;        // Forward direction
    float3 up;             // Up direction
    float fov;             // Field of view (degrees)
    float near_plane;      // Near clipping plane
    float far_plane;       // Far clipping plane
};

// TODO: extract the lighting information from scene

// Render-ready drawable object
struct Drawable {
    GeometryBuffers geometry;  // Vertex/index data
    mat4 transform;            // Model matrix
    Material material;         // Material properties
    int geom_id;               // Original MuJoCo geom ID
    bool visible;              // Visibility flag
};

// Texture information
struct TextureInfo {
    int width;
    int height;
    int channels;
    std::vector<uint8_t> data;  // Raw texture data (RGBA)
    int texture_id;              // MuJoCo texture ID
};

// Main Scene class - provides render-ready data
class Scene {
public:
    Scene(const mjModel* model, const mjvScene* scene);
    ~Scene();

    // Update scene from MuJoCo data
    void Update(const mjModel* model, const mjvScene* scene);

    // Get render-ready data
    const std::vector<Drawable>& GetDrawables() const { return drawables_; }
    const CameraInfo& GetCamera() const { return camera_info_; }
    const std::vector<TextureInfo>& GetTextures() const { return textures_; }
    
    // Get total vertex/index counts for buffer allocation
    size_t GetTotalVertexCount() const;
    size_t GetTotalIndexCount() const;
    
    // Get combined vertex/index buffers (for single buffer rendering)
    void GetCombinedBuffers(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices) const;

private:
    const mjModel* model_;
    
    // Render-ready data
    std::vector<Drawable> drawables_;
    CameraInfo camera_info_;
    std::vector<TextureInfo> textures_;
    
    // Object manager for geometry caching
    std::unique_ptr<ObjectManager> object_manager_;
    
    // Internal extraction functions
    void ExtractGeometries(const mjModel* model, const mjvScene* scene);
    void ExtractCamera(const mjvScene* scene);
    void ExtractMaterials(const mjModel* model, const mjvScene* scene);
    void ExtractTextures(const mjModel* model);
    
    // Helper to extract material from geom
    Material ExtractMaterialFromGeom(const mjModel* model, const mjvGeom* geom);
    
    // Helper to build transform matrix
    mat4 BuildTransform(const mjvGeom* geom);
};

}}  // namespace mujoco::mjbatch
