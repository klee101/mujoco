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

#include "builtin.h"
#include <cmath>
#include <algorithm>
#include <limits>
#include <vector>
#include <unordered_map>

namespace mujoco {
namespace mjbatch {

// ============================================================================
// Helper Functions
// ============================================================================

namespace {


// 顶点唯一键：由 (position index, normal index, uv index) 组成
struct VertexKey {
    int p;
    int n;
    int uv;

    bool operator==(const VertexKey& other) const {
        return p == other.p && n == other.n && uv == other.uv;
    }
};

struct VertexKeyHash {
    size_t operator()(const VertexKey& k) const {
        size_t h1 = std::hash<int>{}(k.p);
        size_t h2 = std::hash<int>{}(k.n);
        size_t h3 = std::hash<int>{}(k.uv);
        // 一个简单稳定的组合
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};


// Helper to calculate AABB from a set of vertices (used for Mesh/HField)
AABB ComputeAABB(const std::vector<Vertex>& vertices) {
    float3 min_bound = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float3 max_bound = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};

    if (vertices.empty()) {
        return { {0,0,0}, {0,0,0} };
    }

    for (const auto& v : vertices) {
        // Assuming Vertex.position is accessible via [] operator or .x .y .z
        // Based on BuildMesh usage: vertex.position[0]
        min_bound.x = std::min(min_bound.x, v.position[0]);
        min_bound.y = std::min(min_bound.y, v.position[1]);
        min_bound.z = std::min(min_bound.z, v.position[2]);

        max_bound.x = std::max(max_bound.x, v.position[0]);
        max_bound.y = std::max(max_bound.y, v.position[1]);
        max_bound.z = std::max(max_bound.z, v.position[2]);
    }

    return {min_bound, max_bound};
}

} // namespace

float4 GeometryBuilder::CalculateOrientation(float3 normal) {
    float len = std::sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (len < 1e-6f) {
        return float4{0, 0, 0, 1};
    }
    normal = normal / len;
    
    float3 up{0, 0, 1};
    float3 axis = glm::cross(up, normal);
    float axis_len = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    
    if (axis_len < 1e-6f) {
        if (normal.z > 0) {
            return float4{0, 0, 0, 1};
        } else {
            return float4{1, 0, 0, 0};
        }
    }
    
    axis = axis / axis_len;
    float angle = std::acos(glm::dot(up, normal));
    float half_angle = angle * 0.5f;
    float sin_half = std::sin(half_angle);
    
    return float4{axis.x * sin_half, axis.y * sin_half, axis.z * sin_half, std::cos(half_angle)};
}

int GeometryBuilder::AppendQuadIndices(uint32_t* ptr, int idx, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    ptr[idx++] = a;
    ptr[idx++] = b;
    ptr[idx++] = c;
    ptr[idx++] = a;
    ptr[idx++] = c;
    ptr[idx++] = d;
    return idx;
}

// Fixed geometry builder functions for Vulkan rendering
// Key fixes:
// 1. Consistent vertex winding order (CCW when viewed from outside)
// 2. Proper normal calculations
// 3. Correct vertex ordering in quads

// Helper to append quad with proper winding (CCW from outside)
static void AppendQuadToVector(std::vector<uint32_t>& indices, 
                               uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    // Quad layout:
    // a --- b
    // |     |
    // d --- c
    // Triangles: (a,b,c) and (a,c,d) - CCW when viewed from front
    indices.push_back(a);
    indices.push_back(b);
    indices.push_back(c);
    indices.push_back(a);
    indices.push_back(c);
    indices.push_back(d);
}

std::size_t GeometryBuilder::NumVerticesPerSide(int num_quads_per_axis) {
    return (num_quads_per_axis + 1) * (num_quads_per_axis + 1);
}

std::size_t GeometryBuilder::NumIndicesPerSide(int num_quads_per_axis) {
    return kNumIndicesPerQuad * num_quads_per_axis * num_quads_per_axis;
}

// ============================================================================
// Primitive Builders
// ============================================================================

GeometryAABB GeometryBuilder::BuildLine() {
    GeometryBuffer result;
    // Line goes from (0,0,0) to (0,0,1)
    AABB aabb = { {0, 0, 0}, {0, 0, 1} };
    
    result.vertices = {
        Vertex({0, 0, 0}, {0, 0, 1}, {0, 0}, {1, 1, 1, 1}),
        Vertex({0, 0, 1}, {0, 0, 1}, {1, 0}, {1, 1, 1, 1})
    };
    result.indices = {0, 1};
    return {result, aabb};
}

// FIXED: BuildPlane with correct winding
GeometryAABB GeometryBuilder::BuildPlane(int num_quads_per_axis) {
    GeometryBuffer result;
    // Plane is generated on XY from -1 to 1, Z is 0
    AABB aabb = { {-1, -1, 0}, {1, 1, 0} };

    // 网格的半边长是 1.0，所以总宽度是 2.0。
    const float total_width = 2.0f; 
    // 每个小格的边长
    const float delta = total_width / num_quads_per_axis; 
    const float3 normal{0, 0, 1}; // 平面法线指向 +Z

    for (int y = 0; y < num_quads_per_axis; ++y) {
        for (int x = 0; x < num_quads_per_axis; ++x) {
            
            const float dx_start = -1.0f + delta * static_cast<float>(x);
            const float dy_start = -1.0f + delta * static_cast<float>(y);
            
            const uint32_t base_index = result.vertices.size(); 

            // 顶点 A (左下角): 局部 UV (0, 0)
            result.vertices.push_back(Vertex(
                {dx_start, dy_start, 0}, 
                normal, 
                {0.0f, 0.0f}, 
                {1, 1, 1, 1}
            ));
            
            result.vertices.push_back(Vertex(
                {dx_start + delta, dy_start, 0}, 
                normal, 
                {1.0f, 0.0f}, 
                {1, 1, 1, 1}
            ));

            result.vertices.push_back(Vertex(
                {dx_start + delta, dy_start + delta, 0}, 
                normal, 
                {1.0f, 1.0f}, 
                {1, 1, 1, 1}
            ));
            
            // 顶点 D (左上角): 局部 UV (0, 1)
            result.vertices.push_back(Vertex(
                {dx_start, dy_start + delta, 0}, 
                normal, 
                {0.0f, 1.0f}, 
                {1, 1, 1, 1}
            ));


            const uint32_t a = base_index + 0;
            const uint32_t b = base_index + 1;
            const uint32_t c = base_index + 2;
            const uint32_t d = base_index + 3;
            
            result.indices.push_back(a);
            result.indices.push_back(d);
            result.indices.push_back(c);
            
            result.indices.push_back(a);
            result.indices.push_back(b);
            result.indices.push_back(c);
        }
    }
    
    return {result, aabb};
}

GeometryAABB GeometryBuilder::BuildLineBox() {
    GeometryBuffer result;
    // Box corners are -1 to 1
    AABB aabb = { {-1, -1, -1}, {1, 1, 1} };

    const float3 normal{0, 0, 1};
    result.vertices = {
        Vertex({-1, -1, -1}, normal, {0, 0}, {1, 1, 1, 1}),
        Vertex({ 1, -1, -1}, normal, {1, 0}, {1, 1, 1, 1}),
        Vertex({-1,  1, -1}, normal, {0, 1}, {1, 1, 1, 1}),
        Vertex({ 1,  1, -1}, normal, {1, 1}, {1, 1, 1, 1}),
        Vertex({-1, -1,  1}, normal, {0, 0}, {1, 1, 1, 1}),
        Vertex({ 1, -1,  1}, normal, {1, 0}, {1, 1, 1, 1}),
        Vertex({-1,  1,  1}, normal, {0, 1}, {1, 1, 1, 1}),
        Vertex({ 1,  1,  1}, normal, {1, 1}, {1, 1, 1, 1})
    };
    result.indices = {
        0, 1,  1, 3,  3, 2,  2, 0,  // Bottom
        4, 5,  5, 7,  7, 6,  6, 4,  // Top
        2, 6,  3, 7,  0, 4,  1, 5   // Edges
    };
    return {result, aabb};
}

// FIXED: BuildBox with correct vertex ordering
GeometryAABB GeometryBuilder::BuildBox(int num_quads_per_axis) {
    GeometryBuffer result;
    // Box faces are at -1 and 1
    AABB aabb = { {-1, -1, -1}, {1, 1, 1} };

    const float quad_size = 2.0f / static_cast<float>(num_quads_per_axis);
    const int vertices_per_side = NumVerticesPerSide(num_quads_per_axis);
    
    // Generate 6 sides with correct normals pointing OUTWARD
    auto generate_side = [&](float3 normal, auto pos_gen) {
        for (int y = 0; y <= num_quads_per_axis; ++y) {
            for (int x = 0; x <= num_quads_per_axis; ++x) {
                const float dx = -1.0f + (quad_size * static_cast<float>(x));
                const float dy = -1.0f + (quad_size * static_cast<float>(y));
                const float3 pos = pos_gen(float2{dx, dy});
                result.vertices.push_back(Vertex(
                    pos, normal, 
                    {static_cast<float>(x) / num_quads_per_axis, 
                     static_cast<float>(y) / num_quads_per_axis}, 
                    {1, 1, 1, 1}
                ));
            }
        }
    };
    
    // +Y face (right) - looking at +Y direction
    generate_side({0, 1, 0}, [](float2 pt) { return float3{pt.x, 1.0f, pt.y}; });
    // -Y face (left) - looking at -Y direction  
    generate_side({0, -1, 0}, [](float2 pt) { return float3{-pt.x, -1.0f, pt.y}; });
    // +X face (front) - looking at +X direction
    generate_side({1, 0, 0}, [](float2 pt) { return float3{1.0f, -pt.x, pt.y}; });
    // -X face (back) - looking at -X direction
    generate_side({-1, 0, 0}, [](float2 pt) { return float3{-1.0f, pt.x, pt.y}; });
    // +Z face (top) - looking down at +Z
    generate_side({0, 0, 1}, [](float2 pt) { return float3{pt.x, pt.y, 1.0f}; });
    // -Z face (bottom) - looking up at -Z
    generate_side({0, 0, -1}, [](float2 pt) { return float3{pt.x, -pt.y, -1.0f}; });
    
    // Generate indices for all sides with proper winding
    for (int side = 0; side < 6; ++side) {
        const int base_offset = side * vertices_per_side;
        for (int y = 0; y < num_quads_per_axis; ++y) {
            for (int x = 0; x < num_quads_per_axis; ++x) {
                const int base = base_offset + (y * (num_quads_per_axis + 1)) + x;
                // Vertices are laid out in rows, left to right, bottom to top
                const uint32_t a = base;
                const uint32_t b = base + 1;
                const uint32_t c = base + (num_quads_per_axis + 1) + 1;
                const uint32_t d = base + (num_quads_per_axis + 1);
                AppendQuadToVector(result.indices, a, b, c, d);
            }
        }
    }
    
    return {result, aabb};
}

// FIXED: BuildSphere with consistent winding
GeometryAABB GeometryBuilder::BuildSphere(int num_stacks, int num_slices) {
    GeometryBuffer result;
    // Unit sphere
    AABB aabb = { {-1, -1, -1}, {1, 1, 1} };

    const float lat_delta = M_PI / static_cast<float>(num_stacks + 1);
    const float lon_delta = 2.0f * M_PI / static_cast<float>(num_slices);
    
    // Top pole (0,0,1)
    result.vertices.push_back(Vertex({0, 0, 1}, {0, 0, 1}, {0.5f, 0}, {1, 1, 1, 1}));
    
    // Generate vertices from top to bottom
    for (int lat = 0; lat < num_stacks; ++lat) {
        const float lat_angle = static_cast<float>(lat + 1) * lat_delta;
        const float cos_lat = std::cos(lat_angle);
        const float sin_lat = std::sin(lat_angle);
        
        for (int lon = 0; lon < num_slices; ++lon) {
            const float lon_angle = static_cast<float>(lon) * lon_delta;
            const float x = sin_lat * std::cos(lon_angle);
            const float y = sin_lat * std::sin(lon_angle);
            const float z = cos_lat;
            float3 normal{x, y, z};
            result.vertices.push_back(Vertex(
                normal, normal,
                {static_cast<float>(lon) / num_slices, 
                 static_cast<float>(lat + 1) / (num_stacks + 1)},
                {1, 1, 1, 1}
            ));
        }
    }
    
    // Bottom pole (0,0,-1)
    result.vertices.push_back(Vertex({0, 0, -1}, {0, 0, -1}, {0.5f, 1}, {1, 1, 1, 1}));
    
    // Top cap triangles (connecting to north pole)
    const uint32_t first_ring = 1;
    for (int lon = 0; lon < num_slices; ++lon) {
        const int next = (lon + 1) % num_slices;
        result.indices.push_back(0);
        result.indices.push_back(first_ring + lon);
        result.indices.push_back(first_ring + next);
    }
    
    // Middle quads
    for (int lat = 0; lat < num_stacks - 1; ++lat) {
        const uint32_t current_ring = 1 + lat * num_slices;
        const uint32_t next_ring = current_ring + num_slices;
        
        for (int lon = 0; lon < num_slices; ++lon) {
            const int next = (lon + 1) % num_slices;
            const uint32_t a = current_ring + lon;
            const uint32_t b = current_ring + next;
            const uint32_t c = next_ring + next;
            const uint32_t d = next_ring + lon;
            AppendQuadToVector(result.indices, a, b, c, d);
        }
    }
    
    // Bottom cap triangles (connecting to south pole)
    const uint32_t last_ring = 1 + (num_stacks - 1) * num_slices;
    const uint32_t south_pole = 1 + num_stacks * num_slices;
    for (int lon = 0; lon < num_slices; ++lon) {
        const int next = (lon + 1) % num_slices;
        result.indices.push_back(south_pole);
        result.indices.push_back(last_ring + next);
        result.indices.push_back(last_ring + lon);
    }
    
    return {result, aabb};
}

GeometryAABB GeometryBuilder::BuildCone(int num_stacks, int num_slices) {
    GeometryBuffer result;
    // Radius 1, Z goes from 0 to 1
    AABB aabb = { {-1, -1, 0}, {1, 1, 1} };

    const float delta_angle = 2.0f * M_PI / static_cast<float>(num_slices);
    const float delta_radius = 1.0f / static_cast<float>(num_stacks);
    
    // Generate vertices
    for (int i = 0; i <= num_stacks; ++i) {
        const float radius = 1.0f - (delta_radius * static_cast<float>(i));
        for (int j = 0; j < num_slices; ++j) {
            const float angle = static_cast<float>(j) * delta_angle;
            const float x = std::cos(angle) * radius;
            const float y = std::sin(angle) * radius;
            const float z = static_cast<float>(i) * delta_radius;
            float3 pos{x, y, z};
            float3 normal = glm::normalize(float3{x, y, 0.5f});
            result.vertices.push_back(Vertex(
                pos, normal,
                {static_cast<float>(j) / num_slices, static_cast<float>(i) / num_stacks},
                {1, 1, 1, 1}
            ));
        }
    }
    
    // Generate indices
    for (int i = 0; i < num_stacks; ++i) {
        for (int j = 0; j < num_slices; ++j) {
            const int base = i * num_slices + j;
            const int next_j = (j + 1) % num_slices;
            const int next_base = base + num_slices;
            AppendQuadToVector(result.indices,
                base, base + next_j, next_base + next_j, next_base);        }
    }
    
    return {result, aabb};
}

GeometryAABB GeometryBuilder::BuildDisk(int num_slices) {
    GeometryBuffer result;
    // Radius 1, Z is 0
    AABB aabb = { {-1, -1, 0}, {1, 1, 0} };

    const float delta_angle = 2.0f * M_PI / static_cast<float>(num_slices);
    const float3 normal{0, 0, 1};
    
    result.vertices.push_back(Vertex({0, 0, 0}, normal, {0.5f, 0.5f}, {1, 1, 1, 1}));
    
    for (int i = 0; i < num_slices; ++i) {
        const float angle = static_cast<float>(i) * delta_angle;
        const float x = std::cos(angle);
        const float y = std::sin(angle);
        result.vertices.push_back(Vertex(
            {x, y, 0}, normal,
            {x * 0.5f + 0.5f, y * 0.5f + 0.5f},
            {1, 1, 1, 1}
        ));
    }
    
    for (int i = 0; i < num_slices; ++i) {
        const int next = (i + 1) % num_slices;
        result.indices.push_back(0);
        result.indices.push_back(1 + i);
        result.indices.push_back(1 + next);
    }
    
    return {result, aabb};
}

GeometryAABB GeometryBuilder::BuildDome(int num_stacks, int num_slices) {
    GeometryBuffer result;
    // Hemisphere: Radius 1, Z from 0 to 1
    AABB aabb = { {-1, -1, 0}, {1, 1, 1} };

    const float lat_delta = 0.5f * M_PI / static_cast<float>(num_stacks);
    const float lon_delta = 2.0f * M_PI / static_cast<float>(num_slices);
    
    result.vertices.push_back(Vertex({0, 0, 1}, {0, 0, 1}, {0.5f, 0}, {1, 1, 1, 1}));
    
    for (int lat = 0; lat < num_stacks; ++lat) {
        const float lat_angle = static_cast<float>(lat + 1) * lat_delta;
        const float cos_lat = std::cos(lat_angle);
        const float sin_lat = std::sin(lat_angle);
        
        for (int lon = 0; lon < num_slices; ++lon) {
            const float lon_angle = static_cast<float>(lon) * lon_delta;
            const float x = sin_lat * std::cos(lon_angle);
            const float y = sin_lat * std::sin(lon_angle);
            const float z = cos_lat;
            float3 normal{x, y, z};
            result.vertices.push_back(Vertex(
                normal, normal,
                {static_cast<float>(lon) / num_slices, static_cast<float>(lat + 1) / num_stacks},
                {1, 1, 1, 1}
            ));
        }
    }
    
    uint32_t row_start = 1;
    for (int lon = 0; lon < num_slices; ++lon) {
        const int next = (lon + 1) % num_slices;
        result.indices.push_back(0);
        result.indices.push_back(row_start + next);
        result.indices.push_back(row_start + lon);
    }
    
    for (int lat = 0; lat < num_stacks - 1; ++lat) {
        const uint32_t north = row_start;
        const uint32_t south = row_start + num_slices;
        for (int lon = 0; lon < num_slices; ++lon) {
            const int next = (lon + 1) % num_slices;
            AppendQuadToVector(result.indices,
                            north + lon, south + lon, south + next, north + next);
        }
        row_start += num_slices;
    }
    
    return {result, aabb};
}

GeometryAABB GeometryBuilder::BuildTube(int num_stacks, int num_slices) {
    GeometryBuffer result;
    // Radius 1, Z from -1 to 1
    AABB aabb = { {-1, -1, -1}, {1, 1, 1} };

    const float delta_angle = 2.0f * M_PI / static_cast<float>(num_slices);
    const float delta_z = 2.0f / static_cast<float>(num_stacks);

    // FIXED: Iterate <= num_slices to create a duplicate vertex row at the seam
    // This allows UVs to go from 0 to 1 cleanly.
    for (int i = 0; i <= num_slices; ++i) {
        const float angle = static_cast<float>(i) * delta_angle;
        
        // Note: For a tube, normals point outward (cos, sin, 0)
        const float3 normal{std::cos(angle), std::sin(angle), 0};

        for (int j = 0; j <= num_stacks; ++j) {
            const float z = -1.0f + (static_cast<float>(j) * delta_z);
            
            // Position matches Normal (Radius = 1) for X and Y
            const float3 pos{normal.x, normal.y, z};

            result.vertices.push_back(Vertex(
                pos, 
                normal,
                // U goes from 0.0 to 1.0. 
                // When i = num_slices, U = 1.0 (Seam fixed)
                {static_cast<float>(i) / num_slices, static_cast<float>(j) / num_stacks},
                {1, 1, 1, 1}
            ));
        }
    }

    // Indices generation
    // We loop < num_slices because we are connecting strip i to strip i+1
    for (int i = 0; i < num_slices; ++i) {
        for (int j = 0; j < num_stacks; ++j) {
            // Because we have (num_stacks + 1) vertices per column
            const int stride = num_stacks + 1;
            
            const int base = i * stride + j;
            const int next_base = (i + 1) * stride + j; // No modulo needed now

            // Standard Quad triangulation
            result.indices.push_back(base);
            result.indices.push_back(next_base);
            result.indices.push_back(next_base + 1);

            result.indices.push_back(base);
            result.indices.push_back(next_base + 1);
            result.indices.push_back(base + 1);
        }
    }

    return {result, aabb};
}

// ============================================================================
// Composite Geometry Builders
// ============================================================================

// Build capsule: tube + 2 domes
GeometryAABB GeometryBuilder::BuildCapsule(int num_stacks, int num_slices) {
    GeometryBuffer result;
    // Tube is [-1, 1] on Z.
    // Top Dome is added at Z+1 (range [1, 2])
    // Bottom Dome is added at Z-1 (range [-2, -1])
    // Radius is 1.
    AABB aabb = { {-1, -1, -2}, {1, 1, 2} };
    
    // Build tube (middle part) - add first
    GeometryAABB tube = BuildTube(num_stacks, num_slices);
    for (const auto& v : tube.buffers.vertices) {
        result.vertices.push_back(v);
    }
    for (uint32_t idx : tube.buffers.indices) {
        result.indices.push_back(idx);
    }
    
    // Build top dome
    GeometryAABB top_dome = BuildDome(num_stacks / 2, num_slices);
    uint32_t top_vertex_offset = static_cast<uint32_t>(tube.buffers.vertices.size());
    for (auto& v : top_dome.buffers.vertices) {
        v.position.z += 1.0f;  // Translate up
        result.vertices.push_back(v);
    }
    // Add top dome indices with offset
    for (uint32_t idx : top_dome.buffers.indices) {
        result.indices.push_back(top_vertex_offset + idx);
    }
    
    // Build bottom dome (flipped)
    GeometryAABB bottom_dome = BuildDome(num_stacks / 2, num_slices);
    uint32_t bottom_vertex_offset = static_cast<uint32_t>(tube.buffers.vertices.size() + top_dome.buffers.vertices.size());
    for (auto& v : bottom_dome.buffers.vertices) {
        v.position.z -= 1.0f;  // Translate down
        v.position.z = -v.position.z;  // Flip
        v.normal.z = -v.normal.z;  // Flip normal
        result.vertices.push_back(v);
    }
    // Add bottom dome indices with offset (reverse winding for flip)
    for (size_t i = 0; i < bottom_dome.buffers.indices.size(); i += 3) {
        result.indices.push_back(bottom_vertex_offset + bottom_dome.buffers.indices[i + 0]);
        result.indices.push_back(bottom_vertex_offset + bottom_dome.buffers.indices[i + 2]);
        result.indices.push_back(bottom_vertex_offset + bottom_dome.buffers.indices[i + 1]);
    }
    
    return {result, aabb};
}

GeometryAABB GeometryBuilder::BuildCylinder(int num_stacks, int num_slices) {
    GeometryBuffer result;
    AABB aabb = { {-1, -1, -1}, {1, 1, 1} };

    // 1. Build the side walls (Tube)
    GeometryAABB tube = BuildTube(num_stacks, num_slices);
    result.vertices = std::move(tube.buffers.vertices);
    result.indices = std::move(tube.buffers.indices);

    // 2. Build Top Disk (+Z)
    // Vertices must be duplicated here because the Normal is different (0,0,1) vs (x,y,0)
    // This creates a "hard edge" which is correct for a cylinder.
    GeometryAABB top_disk = BuildDisk(num_slices);
    uint32_t top_offset = static_cast<uint32_t>(result.vertices.size());
    
    for (auto& v : top_disk.buffers.vertices) {
        v.position.z = 1.0f; // Move disk to top
        // Normal is already {0,0,1} from BuildDisk
        result.vertices.push_back(v);
    }
    for (uint32_t idx : top_disk.buffers.indices) {
        result.indices.push_back(top_offset + idx);
    }

    // 3. Build Bottom Disk (-Z)
    GeometryAABB bottom_disk = BuildDisk(num_slices);
    uint32_t bottom_offset = static_cast<uint32_t>(result.vertices.size());
    
    for (auto& v : bottom_disk.buffers.vertices) {
        v.position.z = -1.0f; // Move disk to bottom
        v.normal = {0, 0, -1}; // Flip normal to point down
        result.vertices.push_back(v);
    }
    
    // Reverse winding for bottom face so it faces outward (down)
    for (size_t i = 0; i < bottom_disk.buffers.indices.size(); i += 3) {
        result.indices.push_back(bottom_offset + bottom_disk.buffers.indices[i]);     // 0
        result.indices.push_back(bottom_offset + bottom_disk.buffers.indices[i + 2]); // 2 (Swap)
        result.indices.push_back(bottom_offset + bottom_disk.buffers.indices[i + 1]); // 1 (Swap)
    }

    return {result, aabb};
}

// Build ellipsoid: sphere with non-uniform scaling (handled by transform, but we build unit sphere)
GeometryAABB GeometryBuilder::BuildEllipsoid(int num_stacks, int num_slices) {
    // Ellipsoid is just a sphere - scaling is applied via transform matrix
    return BuildSphere(num_stacks, num_slices);
}

GeometryAABB GeometryBuilder::BuildFromType(int geom_type, const mjModel* model) {
    const int num_quads = model->vis.quality.numquads;
    const int num_stacks = model->vis.quality.numstacks;
    const int num_slices = model->vis.quality.numslices;
    
    switch (geom_type) {
        case mjGEOM_PLANE:
            return GeometryBuilder::BuildPlane(num_quads);
        case mjGEOM_SPHERE:
            return GeometryBuilder::BuildSphere(num_stacks, num_slices);
        case mjGEOM_ELLIPSOID:
            return GeometryBuilder::BuildEllipsoid(num_stacks, num_slices);
        case mjGEOM_BOX:
            return GeometryBuilder::BuildBox(num_quads);
        case mjGEOM_CYLINDER:
            return GeometryBuilder::BuildCylinder(num_stacks, num_slices);
        case mjGEOM_CAPSULE:
            return GeometryBuilder::BuildCapsule(num_stacks, num_slices);
        case mjGEOM_LINE:
            return GeometryBuilder::BuildLine();
        case mjGEOM_LINEBOX:
            return GeometryBuilder::BuildLineBox();
        default:
            return GeometryAABB();
    }
}

GeometryAABB GeometryBuilder::BuildMesh(const mjModel* model, int mesh_id) {
    GeometryBuffer result;

    if (mesh_id < 0 || mesh_id >= model->nmesh) {
        return {result, AABB{{0,0,0}, {0,0,0}}};
    }

    // 1. 获取所有数据块的起始地址
    int vertadr = model->mesh_vertadr[mesh_id];
    int faceadr = model->mesh_faceadr[mesh_id];
    int facenum = model->mesh_facenum[mesh_id];
    
    int normaladr   = model->mesh_normaladr[mesh_id];
    int texcoordadr = model->mesh_texcoordadr[mesh_id];
    bool has_texcoord = (texcoordadr >= 0);

    // 预分配内存（最大可能数量）
    result.vertices.reserve(facenum * 3);
    result.indices.reserve(facenum * 3);

    // +++ 新增：顶点缓存表（用于去重共享）
    std::unordered_map<VertexKey, uint32_t, VertexKeyHash> vertex_map;
    vertex_map.reserve(facenum * 3);

    // 2. 遍历每一个面
    for (int i = 0; i < facenum; ++i) {
        int global_face_idx = faceadr + i;
        
        int p_idx[3];
        int n_idx[3];
        int uv_idx[3];

        // 位置索引
        p_idx[0] = model->mesh_face[3 * global_face_idx + 0];
        p_idx[1] = model->mesh_face[3 * global_face_idx + 1];
        p_idx[2] = model->mesh_face[3 * global_face_idx + 2];

        // 法线索引
        if (normaladr >= 0 && model->mesh_facenormal) {
            n_idx[0] = model->mesh_facenormal[3 * global_face_idx + 0];
            n_idx[1] = model->mesh_facenormal[3 * global_face_idx + 1];
            n_idx[2] = model->mesh_facenormal[3 * global_face_idx + 2];
        } else {
            // fallback：平滑假设
            n_idx[0] = p_idx[0];
            n_idx[1] = p_idx[1];
            n_idx[2] = p_idx[2];
        }

        // UV 索引
        if (has_texcoord && model->mesh_facetexcoord) {
            uv_idx[0] = model->mesh_facetexcoord[3 * global_face_idx + 0];
            uv_idx[1] = model->mesh_facetexcoord[3 * global_face_idx + 1];
            uv_idx[2] = model->mesh_facetexcoord[3 * global_face_idx + 2];
        } else {
            uv_idx[0] = -1;
            uv_idx[1] = -1;
            uv_idx[2] = -1;
        }

        // 3. 构建三角形三个顶点
        for (int v = 0; v < 3; ++v) {

            // +++ 构造唯一键
            VertexKey key;
            key.p  = p_idx[v];
            key.n  = n_idx[v];
            key.uv = uv_idx[v];

            auto it = vertex_map.find(key);
            if (it != vertex_map.end()) {
                // 已存在，直接复用 index
                result.indices.push_back(it->second);
                continue;
            }

            // --- 构造新顶点 ---
            Vertex vertex;

            // A. Position
            const float* p_ptr = model->mesh_vert + 3 * (vertadr + p_idx[v]);
            vertex.position[0] = p_ptr[0];
            vertex.position[1] = p_ptr[1];
            vertex.position[2] = p_ptr[2];

            // B. Normal
            if (normaladr >= 0) {
                const float* n_ptr = model->mesh_normal + 3 * (normaladr + n_idx[v]);
                vertex.normal[0] = n_ptr[0];
                vertex.normal[1] = n_ptr[1];
                vertex.normal[2] = n_ptr[2];
            } else {
                vertex.normal[0] = 0.0f;
                vertex.normal[1] = 0.0f;
                vertex.normal[2] = 1.0f;
            }

            // C. UV
            if (has_texcoord && uv_idx[v] >= 0) {
                const float* uv_ptr = model->mesh_texcoord + 2 * (texcoordadr + uv_idx[v]);
                vertex.texcoord[0] = uv_ptr[0];
                vertex.texcoord[1] = uv_ptr[1];
            } else {
                vertex.texcoord[0] = 0.0f;
                vertex.texcoord[1] = 0.0f;
            }

            // D. 推入顶点并记录索引
            uint32_t new_index = static_cast<uint32_t>(result.vertices.size());
            result.vertices.push_back(vertex);
            vertex_map[key] = new_index;
            result.indices.push_back(new_index);
        }
    }

    return {result, ComputeAABB(result.vertices)};
}

GeometryAABB GeometryBuilder::BuildMeshSmooth(const mjModel* model, int mesh_id, float normalAngleDeg = 30.0f) {
    GeometryBuffer result;

    if (mesh_id < 0 || mesh_id >= model->nmesh) {
        return {result, AABB{{0,0,0}, {0,0,0}}};
    }

    int vertadr = model->mesh_vertadr[mesh_id];
    int faceadr = model->mesh_faceadr[mesh_id];
    int facenum = model->mesh_facenum[mesh_id];

    int normaladr   = model->mesh_normaladr[mesh_id];
    int texcoordadr = model->mesh_texcoordadr[mesh_id];
    bool has_texcoord = (texcoordadr >= 0);

    result.vertices.reserve(facenum * 3);
    result.indices.reserve(facenum * 3);

    // 顶点去重 map，只按位置 + UV
    struct VertexKeyPos {
        int p;
        int uv;
        bool operator==(const VertexKeyPos& other) const {
            return p == other.p && uv == other.uv;
        }
    };
    struct VertexKeyPosHash {
        size_t operator()(const VertexKeyPos& k) const {
            return std::hash<int>()(k.p) ^ (std::hash<int>()(k.uv) << 1);
        }
    };
    std::unordered_map<VertexKeyPos, uint32_t, VertexKeyPosHash> vertex_map;
    std::vector<glm::vec3> normal_accum; // 累加法线
    std::vector<int> normal_count;

    float normalThreshold = std::cos(glm::radians(normalAngleDeg));

    for (int i = 0; i < facenum; ++i) {
        int global_face_idx = faceadr + i;

        int p_idx[3];
        int n_idx[3];
        int uv_idx[3];

        p_idx[0] = model->mesh_face[3 * global_face_idx + 0];
        p_idx[1] = model->mesh_face[3 * global_face_idx + 1];
        p_idx[2] = model->mesh_face[3 * global_face_idx + 2];

        if (normaladr >= 0 && model->mesh_facenormal) {
            n_idx[0] = model->mesh_facenormal[3 * global_face_idx + 0];
            n_idx[1] = model->mesh_facenormal[3 * global_face_idx + 1];
            n_idx[2] = model->mesh_facenormal[3 * global_face_idx + 2];
        } else {
            n_idx[0] = p_idx[0];
            n_idx[1] = p_idx[1];
            n_idx[2] = p_idx[2];
        }

        if (has_texcoord && model->mesh_facetexcoord) {
            uv_idx[0] = model->mesh_facetexcoord[3 * global_face_idx + 0];
            uv_idx[1] = model->mesh_facetexcoord[3 * global_face_idx + 1];
            uv_idx[2] = model->mesh_facetexcoord[3 * global_face_idx + 2];
        } else {
            uv_idx[0] = -1;
            uv_idx[1] = -1;
            uv_idx[2] = -1;
        }

        for (int v = 0; v < 3; ++v) {
            Vertex vertex;

            // Position
            const float* p_ptr = model->mesh_vert + 3 * (vertadr + p_idx[v]);
            vertex.position[0] = p_ptr[0];
            vertex.position[1] = p_ptr[1];
            vertex.position[2] = p_ptr[2];

            // Normal
            if (normaladr >= 0) {
                const float* n_ptr = model->mesh_normal + 3 * (normaladr + n_idx[v]);
                vertex.normal[0] = n_ptr[0];
                vertex.normal[1] = n_ptr[1];
                vertex.normal[2] = n_ptr[2];
            } else {
                vertex.normal[0] = 0.0f;
                vertex.normal[1] = 0.0f;
                vertex.normal[2] = 1.0f;
            }

            // UV
            if (has_texcoord && uv_idx[v] >= 0) {
                const float* uv_ptr = model->mesh_texcoord + 2 * (texcoordadr + uv_idx[v]);
                vertex.texcoord[0] = uv_ptr[0];
                vertex.texcoord[1] = uv_ptr[1];
            } else {
                vertex.texcoord[0] = 0.0f;
                vertex.texcoord[1] = 0.0f;
            }

            // --- 平滑法线合并 ---
            VertexKeyPos key{p_idx[v], uv_idx[v]};
            auto it = vertex_map.find(key);
            if (it != vertex_map.end()) {
                uint32_t idx = it->second;
                glm::vec3 avgNormal = normal_accum[idx] / float(normal_count[idx]);
                float cosAngle = glm::dot(glm::normalize(avgNormal), glm::vec3(vertex.normal[0], vertex.normal[1], vertex.normal[2]));
                if (cosAngle >= normalThreshold) {
                    // 合并：累加法线
                    normal_accum[idx] += glm::vec3(vertex.normal[0], vertex.normal[1], vertex.normal[2]);
                    normal_count[idx] += 1;
                    result.indices.push_back(idx);
                    continue;
                }
            }

            // 新顶点
            uint32_t new_index = static_cast<uint32_t>(result.vertices.size());
            result.vertices.push_back(vertex);
            vertex_map[key] = new_index;
            normal_accum.push_back(glm::vec3(vertex.normal[0], vertex.normal[1], vertex.normal[2]));
            normal_count.push_back(1);
            result.indices.push_back(new_index);
        }
    }

    // --- 法线归一化 ---
    for (size_t i = 0; i < result.vertices.size(); ++i) {
        glm::vec3 n = normal_accum[i] / float(normal_count[i]);
        n = glm::normalize(n);
        result.vertices[i].normal[0] = n.x;
        result.vertices[i].normal[1] = n.y;
        result.vertices[i].normal[2] = n.z;
    }

    return {result, ComputeAABB(result.vertices)};
}


GeometryAABB GeometryBuilder::BuildConvexHull(const mjModel* model, int mesh_id) {
    return BuildMesh(model, mesh_id);
}

GeometryAABB GeometryBuilder::BuildHeightField(const mjModel* model, int hfield_id) {
    GeometryBuffer result;
    
    if (hfield_id < 0 || hfield_id >= model->nhfield) {
        return {result, AABB{{0,0,0}, {0,0,0}}};
    }
    
    const int nrow = model->hfield_nrow[hfield_id];
    const int ncol = model->hfield_ncol[hfield_id];
    const int dataadr = model->hfield_adr[hfield_id];
    const float* data = model->hfield_data + dataadr;
    
    // Generate vertices
    for (int row = 0; row < nrow; ++row) {
        for (int col = 0; col < ncol; ++col) {
            const float x = static_cast<float>(col) / (ncol - 1) * 2.0f - 1.0f;
            const float y = static_cast<float>(row) / (nrow - 1) * 2.0f - 1.0f;
            const float z = data[row * ncol + col];
            
            // Calculate normal (simplified)
            float3 normal{0, 0, 1};
            if (row > 0 && row < nrow - 1 && col > 0 && col < ncol - 1) {
                const float dzdx = (data[row * ncol + col + 1] - data[row * ncol + col - 1]) * 0.5f;
                const float dzdy = (data[(row + 1) * ncol + col] - data[(row - 1) * ncol + col]) * 0.5f;
                normal = glm::normalize(float3{-dzdx, -dzdy, 1.0f});
            }
            
            result.vertices.push_back(Vertex(
                {x, y, z}, normal,
                {static_cast<float>(col) / (ncol - 1), static_cast<float>(row) / (nrow - 1)},
                {1, 1, 1, 1}
            ));
        }
    }
    
    // Generate indices
    for (int row = 0; row < nrow - 1; ++row) {
        for (int col = 0; col < ncol - 1; ++col) {
            const int base = row * ncol + col;
            AppendQuadToVector(result.indices,
                            base, base + 1, base + ncol + 1, base + ncol);
        }
    }
    
    return {result, ComputeAABB(result.vertices)};
}

}} // namespace mujoco::mjbatch