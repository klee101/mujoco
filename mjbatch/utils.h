#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace mujoco {
  namespace mjbatch{
using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;
using mat3 = glm::mat3;
using mat4 = glm::mat4;

// Arrow geometry constants
constexpr float kArrowScale = 0.5f;
constexpr float kArrowHeadSize = 2.0f;

// ============================================================================
// Read utilities for MuJoCo arrays
// ============================================================================

// Reads a float2 from an array buffer in the model/scene.
template <typename T>
inline float2 ReadFloat2(const T* arr, int index = 0) {
    const T* ptr = arr + (2 * index);
    return float2(ptr[0], ptr[1]);
}

// Reads a float3 from an array buffer in the model/scene.
template <typename T>
inline float3 ReadFloat3(const T* arr, int index = 0) {
    const T* ptr = arr + (3 * index);
    return float3(ptr[0], ptr[1], ptr[2]);
}

// Reads a float4 from an array buffer in the model/scene.
template <typename T>
inline float4 ReadFloat4(const T* arr, int index = 0) {
    const T* ptr = arr + (4 * index);
    return float4(ptr[0], ptr[1], ptr[2], ptr[3]);
}

// Reads a mat3 from an array buffer in the model/scene.
// MuJoCo stores matrices in column-major order
template <typename T>
inline mat3 ReadMat3(const T* arr, int index = 0) {
    const T* ptr = arr + (9 * index);
    // Column-major layout: each column is stored consecutively
    return mat3(
        ptr[0], ptr[3], ptr[6],  // First column
        ptr[1], ptr[4], ptr[7],  // Second column
        ptr[2], ptr[5], ptr[8]   // Third column
    );
}

// ============================================================================
// mat4 helper functions
// ============================================================================


// Create a translation matrix
inline glm::mat4 translation(const float3& v) {
    return glm::translate(glm::mat4(1.0f), v);
}

// Create a scaling matrix
inline glm::mat4 scaling(const float3& s) {
    return glm::scale(glm::mat4(1.0f), s);
}

// Create a uniform scaling matrix
inline glm::mat4 scaling(float s) {
    return glm::scale(glm::mat4(1.0f), float3(s, s, s));
}

// Create a rotation matrix around an arbitrary axis
// angle: rotation angle in radians
// axis: rotation axis (should be normalized)
inline glm::mat4 rotation(float angle, const float3& axis) {
    return glm::rotate(glm::mat4(1.0f), angle, axis);
}

// Create a rotation matrix around X axis
inline glm::mat4 rotationX(float angle) {
    return glm::rotate(glm::mat4(1.0f), angle, float3(1.0f, 0.0f, 0.0f));
}

// Create a rotation matrix around Y axis
inline glm::mat4 rotationY(float angle) {
    return glm::rotate(glm::mat4(1.0f), angle, float3(0.0f, 1.0f, 0.0f));
}

// Create a rotation matrix around Z axis
inline glm::mat4 rotationZ(float angle) {
    return glm::rotate(glm::mat4(1.0f), angle, float3(0.0f, 0.0f, 1.0f));
}

// Create a mat4 from mat3 rotation and float3 translation
inline glm::mat4 fromRotationTranslation(const mat3& rot, const float3& trans) {
   glm::mat4 result(1.0f);
    // Set rotation part (upper-left 3x3)
    result[0] = float4(rot[0], 0.0f);
    result[1] = float4(rot[1], 0.0f);
    result[2] = float4(rot[2], 0.0f);
    // Set translation part (last column)
    result[3] = float4(trans, 1.0f);
    return result;
}

// Extract translation from mat4
inline float3 getTranslation(const glm::mat4& m) {
    return float3(m[3]);
}

// Extract scale from mat4 (assuming no shear)
inline float3 getScale(const glm::mat4& m) {
    return float3(
        glm::length(float3(m[0])),
        glm::length(float3(m[1])),
        glm::length(float3(m[2]))
    );
}

// Extract rotation matrix from mat4 (removes scale)
inline mat3 getRotation(const glm::mat4& m) {
    float3 scale = getScale(m);
    return mat3(
        float3(m[0]) / scale.x,
        float3(m[1]) / scale.y,
        float3(m[2]) / scale.z
    );
}

  }}