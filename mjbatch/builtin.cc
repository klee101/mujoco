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
static float4 CalculateOrientation(float3 normal) {
  // Normalize the normal
  float len = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
  if (len < 1e-6f) {
    return float4{0, 0, 0, 1};
  }
  normal = normal / len;
  
  // Create a quaternion from the normal (simplified version)
  // This assumes the normal represents the Z-axis direction
  float3 up{0, 0, 1};
  float3 axis = cross(up, normal);
  float axis_len = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
  
  if (axis_len < 1e-6f) {
    // Normal is parallel to up vector
    if (normal.z > 0) {
      return float4{0, 0, 0, 1};
    } else {
      return float4{1, 0, 0, 0};
    }
  }
  
  axis = axis / axis_len;
  float angle = std::acos(dot(up, normal));
  float half_angle = angle * 0.5f;
  float sin_half = std::sin(half_angle);
  
  return float4{
    axis.x * sin_half,
    axis.y * sin_half,
    axis.z * sin_half,
    std::cos(half_angle)
  };
}

inline static int AppendQuadIndices(uint16_t* ptr, int idx, uint16_t a, uint16_t b,
                             uint16_t c, uint16_t d) {
  ptr[idx++] = a;
  ptr[idx++] = b;
  ptr[idx++] = c;
  ptr[idx++] = a;
  ptr[idx++] = c;
  ptr[idx++] = d;
  return idx;
}

std::size_t NumVerticesPerSide(int num_quads_per_axis) {
  return (num_quads_per_axis + 1) * (num_quads_per_axis + 1);
}

std::size_t NumIndicesPerSide(int num_quads_per_axis) {
  return kNumIndicesPerQuad * num_quads_per_axis * num_quads_per_axis;
}

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
  static GeometryBuffers Build() {
    GeometryBuffers result;
    
    constexpr float4 kOrientation = {0, 0, 0, 1};
    result.vertices = {
      VertexNoUv({0, 0, 0}, kOrientation),
      VertexNoUv({0, 0, 1}, kOrientation)
    };
    
    result.indices = {0, 1};
    return result;
  }
};

class PlaneBuilder {
 public:
  static GeometryBuffers Build(int num_quads_per_axis) {
    GeometryBuffers result;
    
    const float delta = 2.0f / num_quads_per_axis;
    const float4 orientation = CalculateOrientation({0, 0, 1});
    
    // Generate vertices
    for (int x = 0; x <= num_quads_per_axis; ++x) {
      for (int y = 0; y <= num_quads_per_axis; ++y) {
        const float dx = delta * static_cast<float>(x);
        const float dy = delta * static_cast<float>(y);
        result.vertices.push_back(VertexNoUv({dx - 1.0f, dy - 1.0f, 0}, orientation));
      }
    }
    
    // Generate indices
    for (int x = 0; x < num_quads_per_axis; ++x) {
      for (int y = 0; y < num_quads_per_axis; ++y) {
        const int base_idx = x * (num_quads_per_axis + 1) + y;
        const int i0 = base_idx + 0;
        const int i1 = base_idx + 1;
        const int i2 = base_idx + num_quads_per_axis + 2;
        const int i3 = base_idx + num_quads_per_axis + 1;
        
        result.indices.push_back(i0);
        result.indices.push_back(i1);
        result.indices.push_back(i2);
        result.indices.push_back(i0);
        result.indices.push_back(i2);
        result.indices.push_back(i3);
      }
    }
    
    return result;
  }
};

class LineBoxBuilder {
 public:
  static GeometryBuffers Build() {
    GeometryBuffers result;
    
    constexpr float4 kOrientation = {0, 0, 0, 1};
    result.vertices = {
      VertexNoUv({-1.0f, -1.0f, -1.0f}, kOrientation),
      VertexNoUv({ 1.0f, -1.0f, -1.0f}, kOrientation),
      VertexNoUv({-1.0f,  1.0f, -1.0f}, kOrientation),
      VertexNoUv({ 1.0f,  1.0f, -1.0f}, kOrientation),
      VertexNoUv({-1.0f, -1.0f,  1.0f}, kOrientation),
      VertexNoUv({ 1.0f, -1.0f,  1.0f}, kOrientation),
      VertexNoUv({-1.0f,  1.0f,  1.0f}, kOrientation),
      VertexNoUv({ 1.0f,  1.0f,  1.0f}, kOrientation)
    };
    
    result.indices = {
      0, 1,  1, 3,  3, 2,  2, 0,  // Bottom square
      4, 5,  5, 7,  7, 6,  6, 4,  // Top square
      2, 6,  3, 7,  0, 4,  1, 5   // Connecting edges
    };
    
    return result;
  }
};

class BoxBuilder {
 public:
  static GeometryBuffers Build(int num_quads_per_axis) {
    GeometryBuffers result;
    
    const float quad_size = 2.0f / static_cast<float>(num_quads_per_axis);
    const int vertices_per_side = NumVerticesPerSide(num_quads_per_axis);
    
    // Generate vertices for all 6 sides
    auto generate_side = [&](float3 normal, auto pt_gen) {
      float4 orientation = CalculateOrientation(normal);
      for (int x = 0; x <= num_quads_per_axis; ++x) {
        for (int y = 0; y <= num_quads_per_axis; ++y) {
          const float dx = -1.0f + (quad_size * static_cast<float>(x));
          const float dy = -1.0f + (quad_size * static_cast<float>(y));
          const float3 position = pt_gen(float2{dx, dy});
          result.vertices.push_back(VertexNoUv(position, orientation));
        }
      }
    };
    
    generate_side({0, 1, 0}, [](float2 pt) { return float3{pt.x, 1.0f, pt.y}; });
    generate_side({0, -1, 0}, [](float2 pt) { return float3{pt.x, -1.0f, pt.y}; });
    generate_side({1, 0, 0}, [](float2 pt) { return float3{1.0f, pt.x, pt.y}; });
    generate_side({-1, 0, 0}, [](float2 pt) { return float3{-1.0f, pt.x, pt.y}; });
    generate_side({0, 0, 1}, [](float2 pt) { return float3{pt.x, pt.y, 1.0f}; });
    generate_side({0, 0, -1}, [](float2 pt) { return float3{pt.x, pt.y, -1.0f}; });
    
    // Generate indices for all 6 sides
    for (int side = 0; side < 6; ++side) {
      for (int x = 0; x < num_quads_per_axis; ++x) {
        for (int y = 0; y < num_quads_per_axis; ++y) {
          const int base_idx = (side * vertices_per_side) + (x * (num_quads_per_axis + 1)) + y;
          const int i0 = base_idx + 0;
          const int i1 = base_idx + 1;
          const int i2 = base_idx + num_quads_per_axis + 2;
          const int i3 = base_idx + num_quads_per_axis + 1;
          
          result.indices.push_back(i0);
          result.indices.push_back(i1);
          result.indices.push_back(i2);
          result.indices.push_back(i0);
          result.indices.push_back(i2);
          result.indices.push_back(i3);
        }
      }
    }
    
    return result;
  }
};

class SphereBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices) {
    GeometryBuffers result;
    
    const float lat_angle_delta = M_PI / static_cast<float>(num_stacks + 1);
    const float lon_angle_delta = 2.0 * M_PI / static_cast<float>(num_slices);
    
    auto make_vert = [](float x, float y, float z) {
      const float3 pt{x, y, z};
      return VertexNoUv(pt, CalculateOrientation(pt));
    };
    
    // Add poles
    result.vertices.push_back(make_vert(0, 0, 1));   // North pole (index 0)
    result.vertices.push_back(make_vert(0, 0, -1));  // South pole (index 1)
    
    // Generate vertices by latitude
    for (int lat = 0; lat < num_stacks; ++lat) {
      const float lat_angle = static_cast<float>(lat + 1) * lat_angle_delta;
      const float cos_lat = std::cos(lat_angle);
      const float sin_lat = std::sin(lat_angle);
      const float z = cos_lat;
      
      for (int lon = 0; lon < num_slices; ++lon) {
        const float lon_angle = static_cast<float>(lon) * lon_angle_delta;
        const float x = sin_lat * std::cos(lon_angle);
        const float y = sin_lat * std::sin(lon_angle);
        result.vertices.push_back(make_vert(x, y, z));
      }
    }
    
    // Generate indices
    uint16_t row_start = 2;  // First row starts after the two poles
    
    // North polar cap
    for (int lon = 0; lon < num_slices; ++lon) {
      const int next = lon < (num_slices - 1) ? lon + 1 : 0;
      result.indices.push_back(0);  // North pole
      result.indices.push_back(row_start + next);
      result.indices.push_back(row_start + lon);
    }
    
    // Latitudinal triangle strips
    for (int lat = 0; lat < num_stacks - 1; ++lat) {
      const uint16_t north_start = row_start;
      const uint16_t south_start = row_start + num_slices;
      
      for (int lon = 0; lon < num_slices; ++lon) {
        const int adjacent = lon < (num_slices - 1) ? lon + 1 : 0;
        
        result.indices.push_back(north_start + lon);
        result.indices.push_back(south_start + lon);
        result.indices.push_back(south_start + adjacent);
        result.indices.push_back(north_start + lon);
        result.indices.push_back(south_start + adjacent);
        result.indices.push_back(north_start + adjacent);
      }
      row_start += num_slices;
    }
    
    // South polar cap
    for (int lon = 0; lon < num_slices; ++lon) {
      const int adjacent = lon < (num_slices - 1) ? lon + 1 : 0;
      result.indices.push_back(1);  // South pole
      result.indices.push_back(row_start + lon);
      result.indices.push_back(row_start + adjacent);
    }
    
    return result;
  }
};

class TubeBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices) {
    GeometryBuffers result;
    
    const float delta_angle = 2.f * M_PI / static_cast<float>(num_slices);
    const float delta_stack = 2.f / static_cast<float>(num_stacks);
    
    // Generate vertices
    for (int i = 0; i < num_slices; ++i) {
      const float angle = static_cast<float>(i) * delta_angle;
      const float2 pt{std::cos(angle), std::sin(angle)};
      const float4 orientation = CalculateOrientation({pt.x, pt.y, 0});
      
      for (int j = 0; j <= num_stacks; ++j) {
        const float z = -1.0f + (static_cast<float>(j) * delta_stack);
        result.vertices.push_back(VertexNoUv({pt.x, pt.y, z}, orientation));
      }
    }
    
    // Generate indices
    const int num_vertices = result.vertices.size();
    const int num_vertices_in_spine = num_stacks + 1;
    
    for (int i = 0; i < num_slices; ++i) {
      for (int j = 0; j < num_stacks; ++j) {
        const int base_idx = (i * num_vertices_in_spine) + j;
        const int i0 = base_idx + 0;
        const int i1 = base_idx + 1;
        const int i2 = (base_idx + num_stacks + 2) % num_vertices;
        const int i3 = (base_idx + num_stacks + 1) % num_vertices;
        
        result.indices.push_back(i0);
        result.indices.push_back(i1);
        result.indices.push_back(i2);
        result.indices.push_back(i0);
        result.indices.push_back(i2);
        result.indices.push_back(i3);
      }
    }
    
    return result;
  }
};

class DiskBuilder {
 public:
  static GeometryBuffers Build(int num_slices) {
    GeometryBuffers result;
    
    const float delta_angle = 2.0 * M_PI / static_cast<float>(num_slices);
    const float4 orientation = CalculateOrientation({0, 0, 1});
    
    // Center vertex
    result.vertices.push_back(VertexNoUv(float3{0, 0, 0}, orientation));
    
    // Perimeter vertices
    for (int i = 0; i < num_slices; ++i) {
      const float angle = static_cast<float>(i) * delta_angle;
      const float x = std::cos(angle);
      const float y = std::sin(angle);
      result.vertices.push_back(VertexNoUv(float3{x, y, 0}, orientation));
    }
    
    // Generate indices
    for (int i = 0; i < num_slices; ++i) {
      const int next = i < (num_slices - 1) ? i + 1 : 0;
      result.indices.push_back(0);
      result.indices.push_back(1 + i);
      result.indices.push_back(1 + next);
    }
    
    return result;
  }
};

class DomeBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices) {
    GeometryBuffers result;
    
    const float lat_angle_delta = 0.5 * M_PI / static_cast<float>(num_stacks);
    const float lon_angle_delta = 2.0 * M_PI / static_cast<float>(num_slices);
    
    auto make_vert = [](float x, float y, float z) {
      const float3 pt{x, y, z};
      return VertexNoUv(pt, CalculateOrientation(pt));
    };
    
    // Add pole
    result.vertices.push_back(make_vert(0, 0, 1));
    
    // Generate vertices by latitude
    for (int lat = 0; lat < num_stacks; ++lat) {
      const float lat_angle = static_cast<float>(lat + 1) * lat_angle_delta;
      const float cos_lat = std::cos(lat_angle);
      const float sin_lat = std::sin(lat_angle);
      const float z = cos_lat;
      
      for (int lon = 0; lon < num_slices; ++lon) {
        const float lon_angle = static_cast<float>(lon) * lon_angle_delta;
        const float x = sin_lat * std::cos(lon_angle);
        const float y = sin_lat * std::sin(lon_angle);
        result.vertices.push_back(make_vert(x, y, z));
      }
    }
    
    // Generate indices
    uint16_t row_start = 1;
    
    // Polar cap
    for (int lon = 0; lon < num_slices; ++lon) {
      const int next = lon < (num_slices - 1) ? lon + 1 : 0;
      result.indices.push_back(0);
      result.indices.push_back(row_start + next);
      result.indices.push_back(row_start + lon);
    }
    
    // Latitudinal quad strips
    for (int lat = 0; lat < num_stacks - 1; ++lat) {
      const int north_start = row_start;
      const int south_start = row_start + num_slices;
      
      for (int lon = 0; lon < num_slices; ++lon) {
        const int adjacent = lon < (num_slices - 1) ? lon + 1 : 0;
        
        result.indices.push_back(north_start + lon);
        result.indices.push_back(south_start + lon);
        result.indices.push_back(south_start + adjacent);
        result.indices.push_back(north_start + lon);
        result.indices.push_back(south_start + adjacent);
        result.indices.push_back(north_start + adjacent);
      }
      row_start += num_slices;
    }
    
    return result;
  }
};

class ConeBuilder {
 public:
  static GeometryBuffers Build(int num_stacks, int num_slices) {
    GeometryBuffers result;
    
    const float delta_angle = 2.0 * M_PI / static_cast<float>(num_slices);
    const float delta_radius = 1.0f / static_cast<float>(num_stacks);
    
    auto make_vert = [](float theta, float radius) -> VertexNoUv {
      static constexpr float kNormalScale = 0.70710678118f;
      const float cz = std::cos(theta);
      const float sz = std::sin(theta);
      const float3 pt{cz * radius, sz * radius, 1.f - radius};
      const float3 n{cz * kNormalScale, sz * kNormalScale, kNormalScale};
      return VertexNoUv(pt, CalculateOrientation(n));
    };
    
    // Pole: use triangles
    for (int j = 0; j < num_slices; ++j) {
      const float angle1 = static_cast<float>(j) * delta_angle;
      const float angle2 = static_cast<float>(j + 1) * delta_angle;
      
      result.vertices.push_back(make_vert(angle1, delta_radius));
      result.vertices.push_back(make_vert(angle2, delta_radius));
      
      VertexNoUv v3;
      v3.position = {0, 0, 1};
      v3.orientation = CalculateOrientation(v3.position);
      result.vertices.push_back(v3);
    }
    
    // The rest: use quads
    for (int i = 1; i < num_stacks; ++i) {
      const float radius1 = delta_radius * static_cast<float>(i);
      const float radius2 = delta_radius * static_cast<float>(i + 1);
      
      for (int j = 0; j < num_slices; ++j) {
        const float angle1 = static_cast<float>(j) * delta_angle;
        const float angle2 = static_cast<float>(j + 1) * delta_angle;
        
        result.vertices.push_back(make_vert(angle1, radius2));
        result.vertices.push_back(make_vert(angle2, radius2));
        result.vertices.push_back(make_vert(angle2, radius1));
        result.vertices.push_back(make_vert(angle1, radius1));
      }
    }
    
    // Generate indices
    // Triangle indices for pole
    for (int j = 0; j < num_slices * 3; ++j) {
      result.indices.push_back(j);
    }
    
    // Quad indices for body
    int quad_idx = num_slices * 3;
    for (int i = 1; i < num_stacks; ++i) {
      for (int j = 0; j < num_slices; ++j) {
        result.indices.push_back(quad_idx + 0);
        result.indices.push_back(quad_idx + 1);
        result.indices.push_back(quad_idx + 2);
        result.indices.push_back(quad_idx + 0);
        result.indices.push_back(quad_idx + 2);
        result.indices.push_back(quad_idx + 3);
        quad_idx += 4;
      }
    }
    
    return result;
  }
};

// ============================================================================
// Public API Functions
// ============================================================================

GeometryBuffers CreateGeometryFromType(int geom_type, const mjModel* model) {
  const int num_quads = model->vis.quality.numquads;
  const int num_stacks = model->vis.quality.numstacks;
  const int num_slices = model->vis.quality.numslices;
  
  switch (geom_type) {
    case mjGEOM_PLANE:
      return PlaneBuilder::Build(num_quads);
    case mjGEOM_SPHERE:
    case mjGEOM_ELLIPSOID:
      return SphereBuilder::Build(num_stacks, num_slices);
    case mjGEOM_BOX:
      return BoxBuilder::Build(num_quads);
    case mjGEOM_CAPSULE:
      // For capsule, return tube (caller should also create two domes)
      return TubeBuilder::Build(num_stacks, num_slices);
    case mjGEOM_CYLINDER:
      // For cylinder, return tube (caller should also create two disks)
      return TubeBuilder::Build(num_stacks, num_slices);
    case mjGEOM_LINE:
      return LineBuilder::Build();
    case mjGEOM_LINEBOX:
      return LineBoxBuilder::Build();
    default:
      return GeometryBuffers();
  }
}

// Helper functions for composite geometries
GeometryBuffers CreateLine(const mjModel* model) {
  return LineBuilder::Build();
}

GeometryBuffers CreatePlane(const mjModel* model) {
  const int num_quads = model->vis.quality.numquads;
  return PlaneBuilder::Build(num_quads);
}

GeometryBuffers CreateLineBox(const mjModel* model) {
  return LineBoxBuilder::Build();
}

GeometryBuffers CreateBox(const mjModel* model) {
  const int num_quads = model->vis.quality.numquads;
  return BoxBuilder::Build(num_quads);
}
GeometryBuffers CreateSphere(const mjModel* model) {
  const int num_stacks = model->vis.quality.numstacks;
  const int num_slices = model->vis.quality.numslices;
  return SphereBuilder::Build(num_stacks, num_slices);
}

GeometryBuffers CreateDome(const mjModel* model) {
  const int num_stacks = model->vis.quality.numstacks / 2;
  const int num_slices = model->vis.quality.numslices;
  return DomeBuilder::Build(num_stacks, num_slices);
}

GeometryBuffers CreateDisk(const mjModel* model) {
  const int num_slices = model->vis.quality.numslices;
  return DiskBuilder::Build(num_slices);
}

GeometryBuffers CreateCone(const mjModel* model) {
  const int num_stacks = model->vis.quality.numstacks;
  const int num_slices = model->vis.quality.numslices;
  return ConeBuilder::Build(num_stacks, num_slices);
}

GeometryBuffers CreateTube(const mjModel* model) {
  const int num_stacks = model->vis.quality.numstacks;
  const int num_slices = model->vis.quality.numslices;
  return TubeBuilder::Build(num_stacks, num_slices);
}
  }
    }