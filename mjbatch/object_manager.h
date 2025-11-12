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

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>
#include "utils.h"
#include "builtin.h"

#include <mujoco/mjmodel.h>

namespace mujoco{
  namespace mjbatch{
struct mjBatchConfig {
  // "Loads" an asset, returning its contents in `out` and size in `out_size`.
  // Returns 0 on success and non-zero to indicate an error. The caller must
  // free `out`.
  typedef int (*load_asset_fn)(const char* path, void* user_data,
                              unsigned char** out, uint64_t* out_size);

  // Used to load filament assets (e.g. materials, image-based lights, etc.).
  load_asset_fn load_asset;
  void* load_asset_user_data;

  // The native window handle into which we can render directly.
  void* native_window;

  // The backend graphics API to use.
  int graphics_api;

  // Whether or not to enable GUI rendering.
  bool enable_gui;
};

enum ShapeType {
    kLine,
    kLineBox,
    kPlane,
    kBox,
    kSphere,
    kCone,
    kDisk,
    kDome,
    kTube,
    kNumShapes,
  };

// Creates and owns various filament objects based on the data in a mjrContext.
class ObjectManager {
 public:
  ObjectManager(const mjModel* model,
                const mjBatchConfig* config);
  ~ObjectManager();

  using SphericalHarmonics = float3[9];

  // void UploadMesh(const mjModel* model, int id);

  // void UploadTexture(const mjModel* model, int id);

  // void UploadHeightField(const mjModel* model, int id);

  // void UploadFont(const uint8_t* pixels, int width, int height, int id);


  // Returns the cached instance of a filament object created from the mjModel.
  const GeometryBuffers* GetMeshBuffer(int data_id) const;
  const GeometryBuffers* GetShapeBuffer(ShapeType shape) const;
  const GeometryBuffers* GetHeightFieldBuffer(int hfield_id) const;
  // Creates and returns a new instance of a filament object. The objects are
  // owned by the ObjectManager and will be deleted in the destructor.

  const mjModel* GetModel() const { return model_; }

  ObjectManager(const ObjectManager&) = delete;
  ObjectManager& operator=(const ObjectManager&) = delete;

 private:
  const mjModel* model_ = nullptr;
  const mjBatchConfig* config_;

  std::array<GeometryBuffers, kNumShapes> shapes_;
  std::unordered_map<int, GeometryBuffers> meshes_;
  std::unordered_map<int, GeometryBuffers> convex_hulls_;
  std::unordered_map<int, GeometryBuffers> height_fields_;
  std::unordered_map<int, SphericalHarmonics> spherical_harmonics_;

};


  }}