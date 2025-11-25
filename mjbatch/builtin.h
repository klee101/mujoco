// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include <glm/glm.hpp>

namespace mujoco {
  namespace mjbatch {

using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

static constexpr size_t kNumIndicesPerQuad = 6;

// Vertex structure for rendering (position, normal, texcoord, color)
struct Vertex {
  float3 position;
  float3 normal;
  float2 texcoord;
  float4 color;
  
  Vertex() : position{0, 0, 0}, normal{0, 0, 1}, texcoord{0, 0}, color{1, 1, 1, 1} {}
  Vertex(float3 pos, float3 norm, float2 uv, float4 col) 
    : position(pos), normal(norm), texcoord(uv), color(col) {}
};

// Geometry buffer structure containing raw vertex and index data
struct GeometryBuffers {
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;  // Changed to uint32_t for better compatibility
  
  size_t GetVertexCount() const { return vertices.size(); }
  size_t GetIndexCount() const { return indices.size(); }
  size_t GetVertexBufferSize() const { return vertices.size() * sizeof(Vertex); }
  size_t GetIndexBufferSize() const { return indices.size() * sizeof(uint32_t); }
};

// ============================================================================
// Unified Geometry Builder
// ============================================================================

class GeometryBuilder {
public:
  // Build geometry from MuJoCo geometry type
  static GeometryBuffers BuildFromType(int geom_type, const mjModel* model);
  
  // Build specific primitive shapes
  static GeometryBuffers BuildLine();
  static GeometryBuffers BuildPlane(int num_quads_per_axis = 10);
  static GeometryBuffers BuildLineBox();
  static GeometryBuffers BuildBox(int num_quads_per_axis = 10);
  static GeometryBuffers BuildSphere(int num_stacks = 20, int num_slices = 20);
  static GeometryBuffers BuildEllipsoid(int num_stacks = 20, int num_slices = 20);
  static GeometryBuffers BuildCone(int num_stacks = 10, int num_slices = 20);
  static GeometryBuffers BuildDisk(int num_slices = 20);
  static GeometryBuffers BuildDome(int num_stacks = 10, int num_slices = 20);
  static GeometryBuffers BuildTube(int num_stacks = 10, int num_slices = 20);
  static GeometryBuffers BuildCylinder(int num_stacks = 10, int num_slices = 20);
  static GeometryBuffers BuildCapsule(int num_stacks = 10, int num_slices = 20);
  
  // Build mesh from MuJoCo mesh data
  static GeometryBuffers BuildMesh(const mjModel* model, int mesh_id);
  static GeometryBuffers BuildConvexHull(const mjModel* model, int mesh_id);
  // Build height field
  static GeometryBuffers BuildHeightField(const mjModel* model, int hfield_id);

private:
  // Helper functions
  static float4 CalculateOrientation(float3 normal);
  static int AppendQuadIndices(uint32_t* ptr, int idx, uint32_t a, uint32_t b, uint32_t c, uint32_t d);
  static std::size_t NumVerticesPerSide(int num_quads_per_axis);
  static std::size_t NumIndicesPerSide(int num_quads_per_axis);
};

  } // namespace mjbatch
} // namespace mujoco
