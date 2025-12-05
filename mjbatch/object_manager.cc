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

#include "object_manager.h"
#include "builtin.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>


namespace mujoco{
  namespace mjbatch{

// Loads binary data from a file using mjBatchConfig callbacks.
struct Asset {
  Asset(const char* filename, const mjBatchConfig* config) {
    const int error = config->load_asset(filename, config->load_asset_user_data,
                                         &payload, &size);
    if (error) {
      mju_error("Failed to load file: %s (error: %d)", filename, error);
    }
  }

  ~Asset() {
    if (payload) {
      free(payload);
      payload = nullptr;
    }
  }

  uint64_t size = 0;
  unsigned char* payload = nullptr;

  Asset(const Asset&) = delete;
  Asset& operator=(const Asset&) = delete;
};


ObjectManager::ObjectManager(const mjModel* model,
                             const mjBatchConfig* config)
    : model_(model), config_(config) {
  // Use unified GeometryBuilder interface
  shapes_[kPlane] = GeometryBuilder::BuildPlane(model->vis.quality.numquads);
  shapes_[kBox] = GeometryBuilder::BuildBox(model->vis.quality.numquads);
  shapes_[kSphere] = GeometryBuilder::BuildSphere(model->vis.quality.numstacks, model->vis.quality.numslices);
  shapes_[kCapsule] = GeometryBuilder::BuildCapsule(model->vis.quality.numstacks, model->vis.quality.numslices);
  shapes_[kEllipsoid] = GeometryBuilder::BuildSphere(model->vis.quality.numstacks, model->vis.quality.numslices);
  shapes_[kCylinder] = GeometryBuilder::BuildCylinder(model->vis.quality.numstacks, model->vis.quality.numslices);
  

}

ObjectManager::~ObjectManager() {

}




const GeometryBuffers* ObjectManager::GetMeshBuffer(int data_id) const {
  
  const int mesh_id = data_id / 2;
  const bool is_convex = (data_id % 2 != 0);

  if (!is_convex) {
    // ---------------- 处理普通 Mesh ----------------
    auto it = meshes_.find(mesh_id);
    if (it != meshes_.end()) {
      return &it->second;
    }
    
    // 边界检查
    if (mesh_id >= 0 && mesh_id < model_->nmesh) {
      // [FIX] 调用我们在 builtin.cc 中实现的 BuildMesh
      GeometryBuffers mesh = GeometryBuilder::BuildMesh(model_, mesh_id);
      
      // 插入 Map 并返回引用
      // 使用 emplace 避免拷贝
      auto inserted = meshes_.emplace(mesh_id, std::move(mesh));
      return &inserted.first->second;
    }
    return nullptr;

  } else {
    // ---------------- 处理 Convex Hull ----------------
    auto it = convex_hulls_.find(mesh_id);
    if (it != convex_hulls_.end()) {
      return &it->second;
    }

    if (mesh_id >= 0 && mesh_id < model_->nmesh) {
        // 通常凸包的几何数据可以用原始 Mesh 近似，或者 MuJoCo 有专门的 graph
        // 这里我们复用 BuildConvexHull (在 builtin 中实现为调用 BuildMesh)
        GeometryBuffers hull = GeometryBuilder::BuildConvexHull(model_, mesh_id);
        
        auto inserted = convex_hulls_.emplace(mesh_id, std::move(hull));
        return &inserted.first->second;
    }
    return nullptr;
  }
}

const GeometryBuffers* ObjectManager::GetHeightFieldBuffer(
    int hfield_id) const {
  auto it = height_fields_.find(hfield_id);
  if (it != height_fields_.end()) {
    return &it->second;
  }
  // Build height field on demand
  if (hfield_id >= 0 && hfield_id < model_->nhfield) {
    GeometryBuffers hfield = GeometryBuilder::BuildHeightField(model_, hfield_id);
    height_fields_[hfield_id] = std::move(hfield);
    return &height_fields_[hfield_id];
  }
  return nullptr;
}

const GeometryBuffers* ObjectManager::GetShapeBuffer(ShapeType shape) const {
  if (shape < 0 || shape >= kNumShapes) {
    mju_error("Invalid shape type: %d", shape);
  }
  return &shapes_[shape];
}

}}