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
using mat4 = glm::mat4;

struct ShapeGeometry {
    ShapeType type;
    mat4 transform;
};

class Geom {
private:
    void BuildShapes(const mjvGeom& geom, std::vector<ShapeGeometry>& out_shapes);

public:
    std::vector<ShapeGeometry> shapes;
    int geom_type;

    explicit Geom(const mjvGeom& geom);
    void Update(const mjvGeom& geom);
};

class Scene {
public:
    std::vector<Geom> geoms;
    std::vector<GeometryBuffers> geometryBuffers;
    std::vector<mat4> transforms;

    std::unique_ptr<ObjectManager> object_manager;

    Scene(const mjModel* model, const mjvScene* scene);

    void createGeometryBuffers();
    void UpdateScene(const mjModel* model, const mjvScene* scene);
};

}}  // namespace mujoco::mjbatch
