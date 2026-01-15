// [NEW FILE] shared_protocol.h
#pragma once
#include <cstdint>
#include <glm/glm.hpp>

namespace mujoco {
namespace mjbatch {

inline glm::mat4 RowMajorToGLM(const float m[16]) {
    glm::mat4 r;
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            r[col][row] = m[row * 4 + col];
        }
    }
    return r;
}


#pragma pack(push, 1) // Force tight packing to match NumPy

struct ShmCamera {
    float view[16];   // 4x4
    float proj[16];   // 4x4
    float pos[3];     // 3
};

struct ShmLight {
    int32_t type;
    float pos[3];
    float dir[3];
    float attenuation[3];
    float cutoff;
    float exponent;
    float ambient[3];
    float diffuse[3];
    float specular[3];
    int32_t castshadow;
};

struct ShmGeom {
    // Integer IDs
    int32_t type;
    int32_t dataid;
    int32_t objtype;
    int32_t objid;
    int32_t category;
    int32_t matid;
    int32_t texcoord;
    int32_t segid;

    // Spatial Transform
    float pos[3];
    float mat[9];     // 3x3 Flattened Rotation
    float size[3];

    // Material Properties
    float rgba[4];
    float emission;
    float specular;
    float shininess;
    float reflectance;
};

// Defined Constants based on Python config
#define SHM_MAX_GEOMS 1000 
#define SHM_NUM_CAMERAS 3
#define SHM_MAX_LIGHTS 10

struct EnvRenderSlot {
    ShmCamera cameras[SHM_NUM_CAMERAS];
    
    int32_t num_lights;
    ShmLight lights[SHM_MAX_LIGHTS];

    int32_t num_geoms;
    ShmGeom geoms[SHM_MAX_GEOMS];
};
#pragma pack(pop)

} // namespace mjbatch
} // namespace mujoco