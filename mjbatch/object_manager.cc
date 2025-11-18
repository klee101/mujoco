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
  shapes_[kLine] = GeometryBuilder::BuildLine();
  shapes_[kBox] = GeometryBuilder::BuildBox(model->vis.quality.numquads);
  shapes_[kLineBox] = GeometryBuilder::BuildLineBox();
  shapes_[kCone] = GeometryBuilder::BuildCone(model->vis.quality.numstacks, model->vis.quality.numslices);
  shapes_[kDisk] = GeometryBuilder::BuildDisk(model->vis.quality.numslices);
  shapes_[kDome] = GeometryBuilder::BuildDome(model->vis.quality.numstacks / 2, model->vis.quality.numslices);
  shapes_[kTube] = GeometryBuilder::BuildTube(model->vis.quality.numstacks, model->vis.quality.numslices);
  shapes_[kPlane] = GeometryBuilder::BuildPlane(model->vis.quality.numquads);
  shapes_[kSphere] = GeometryBuilder::BuildSphere(model->vis.quality.numstacks, model->vis.quality.numslices);
  
  // TODO: Load meshes and height fields on demand
}

ObjectManager::~ObjectManager() {

}

// TODO: Implement Mesh Hfield and Texture uploads



const GeometryBuffers* ObjectManager::GetMeshBuffer(int data_id) const {
  // As defined by mjv_updateScene:
  //   original mesh: mesh_id * 2
  //   convex hull: (mesh_id * 2) + 1
  const int mesh_id = data_id / 2;
  if (data_id % 2 == 0) {
    auto it = meshes_.find(mesh_id);
    if (it != meshes_.end()) {
      return &it->second;
    }
    // Build mesh on demand
    if (mesh_id >= 0 && mesh_id < model_->nmesh) {
      GeometryBuffers mesh = GeometryBuilder::BuildMesh(model_, mesh_id);
      meshes_[mesh_id] = std::move(mesh);
      return &meshes_[mesh_id];
    }
    return nullptr;
  } else {
    auto it = convex_hulls_.find(mesh_id);
    if (it != convex_hulls_.end()) {
      return &it->second;
    }
    // TODO: Build convex hull
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