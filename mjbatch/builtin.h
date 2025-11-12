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
  namespace mjbatch{

using float2 = glm::vec2;
using float3 = glm::vec3;
using float4 = glm::vec4;

static constexpr size_t kNumIndicesPerQuad = 6;

// Simple vertex structure without UV coordinates
struct VertexNoUv {
  float3 position;
  float4 orientation;
  
  VertexNoUv() : position{0, 0, 0}, orientation{0, 0, 0, 1} {}
  VertexNoUv(float3 pos, float4 orient) : position(pos), orientation(orient) {}
};

// Calculate orientation (quaternion) from normal vector
float4 CalculateOrientation(float3 normal);

inline int AppendQuadIndices(uint16_t* ptr, int idx, uint16_t a, uint16_t b,
                             uint16_t c, uint16_t d) {
  ptr[idx++] = a;
  ptr[idx++] = b;
  ptr[idx++] = c;
  ptr[idx++] = a;
  ptr[idx++] = c;
  ptr[idx++] = d;
  return idx;
}

std::size_t NumVerticesPerSide(int num_quads_per_axis);
std::size_t NumIndicesPerSide(int num_quads_per_axis);

// Geometry buffer structure containing raw vertex and index data
struct GeometryBuffers {
  std::vector<VertexNoUv> vertices;
  std::vector<uint16_t> indices;
};

// ============================================================================
// Geometry Builders
// ============================================================================

class LineBuilder {
 public:
  static GeometryBuffers Build();
};

class PlaneBuilder {
 public:
  static GeometryBuffers Build(int num_quads_per_axis);
};

class LineBoxBuilder {
 public:
  static GeometryBuffers Build();
};

class BoxBuilder {
 public:
  static GeometryBuffers Build(int num_quads_per_axis);
};

class SphereBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices);
};

class TubeBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices);
};

class DiskBuilder {
 public:
  static GeometryBuffers Build(int num_slices);
};

class DomeBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices);
};

class ConeBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices);
};

// ============================================================================
// Public API Functions (Declarations only)
// ============================================================================

GeometryBuffers CreateGeometryFromType(int geom_type, const mjModel* model);

// Helper functions for composite geometries
GeometryBuffers CreateLine(const mjModel* model);
GeometryBuffers CreatePlane(const mjModel* model);
GeometryBuffers CreateLineBox(const mjModel* model);
GeometryBuffers CreateBox(const mjModel* model);
GeometryBuffers CreateSphere(const mjModel* model);
GeometryBuffers CreateDome(const mjModel* model);
GeometryBuffers CreateDisk(const mjModel* model);
GeometryBuffers CreateCone(const mjModel* model);
GeometryBuffers CreateTube(const mjModel* model);

  } // namespace mjbatch
} // namespace mujoco