#include "scene.h"

namespace mujoco {
namespace mjbatch {

// ---------------- Geom Implementation ----------------

void Geom::BuildShapes(const mjvGeom& geom, std::vector<ShapeGeometry>& out_shapes) {
    if (geom.type == mjGEOM_FLEX || geom.type == mjGEOM_SKIN) return;

    const mat4 base = fromRotationTranslation(ReadMat3(geom.mat), ReadFloat3(geom.pos));
    float3 size = ReadFloat3(geom.size);

    if (geom.type == mjGEOM_MESH) {
    } else if (geom.type == mjGEOM_HFIELD) {
    } else if (geom.type == mjGEOM_PLANE) {
        out_shapes.push_back({kPlane, base});
    } else if (geom.type == mjGEOM_SPHERE) {
        out_shapes.push_back({kSphere, base});
    } else if (geom.type == mjGEOM_ELLIPSOID) {
        out_shapes.push_back({kSphere, base});
    } else if (geom.type == mjGEOM_BOX) {
        out_shapes.push_back({kBox, base});
    } else if (geom.type == mjGEOM_CAPSULE) {
        const float xz_size = 0.5f * (size.x + size.y);
        out_shapes.push_back({kTube, base * scaling(size)});
        mat4 top = base * translation(float3{0, 0, size.z}) *
                   scaling(float3{1, 1, xz_size / size.z});
        mat4 bottom = base * translation(float3{0, 0, -size.z}) *
                      rotation(M_PI, float3{1, 0, 0}) *
                      scaling(float3{1, 1, xz_size / size.z});
        out_shapes.push_back({kDome, top});
        out_shapes.push_back({kDome, bottom});
    } else if (geom.type == mjGEOM_CYLINDER) {
        mat4 tube = base * scaling(size);
        mat4 top = base * translation(float3{0, 0, size.z});
        mat4 bottom = base * translation(float3{0, 0, -size.z}) *
                      rotation(M_PI, float3{1, 0, 0});
        out_shapes.push_back({kTube, tube});
        out_shapes.push_back({kDisk, top});
        out_shapes.push_back({kDisk, bottom});
    } else if (geom.type == mjGEOM_ARROW) {
        mat4 scaled = base * scaling(float3{1, 1, kArrowScale}) *
                      translation(float3{0, 0, kArrowScale});
        mat4 tube = scaled * scaling(size);
        mat4 cone = scaled * translation(float3{0, 0, size.z}) *
                    scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
        mat4 coneDisk = scaled * translation(float3{0, 0, size.z}) *
                        rotation(M_PI, float3{1, 0, 0}) *
                        scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
        mat4 bottomDisk = scaled * translation(float3{0, 0, -size.z}) *
                          rotation(M_PI, float3{1, 0, 0});
        out_shapes.push_back({kTube, tube});
        out_shapes.push_back({kCone, cone});
        out_shapes.push_back({kDisk, coneDisk});
        out_shapes.push_back({kDisk, bottomDisk});
    } else if (geom.type == mjGEOM_ARROW1) {
        mat4 scaled = base * scaling(float3{1, 1, kArrowScale}) *
                      translation(float3{0, 0, kArrowScale});
        mat4 tube = scaled * scaling(size);
        mat4 cone = scaled * translation(float3{0, 0, size.z});
        mat4 bottomDisk = scaled * translation(float3{0, 0, -size.z}) *
                          rotation(M_PI, float3{1, 0, 0});
        out_shapes.push_back({kTube, tube});
        out_shapes.push_back({kCone, cone});
        out_shapes.push_back({kDisk, bottomDisk});
    } else if (geom.type == mjGEOM_ARROW2) {
        mat4 scaled = base * scaling(float3{1, 1, kArrowScale}) *
                      translation(float3{0, 0, kArrowScale});
        mat4 tube = scaled * scaling(size);
        mat4 topCone = scaled * translation(float3{0, 0, size.z}) *
                       scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
        mat4 bottomCone = scaled * translation(float3{0, 0, -size.z}) *
                          rotation(M_PI, float3{1, 0, 0}) *
                          scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
        mat4 topDisk = scaled * translation(float3{0, 0, size.z}) *
                       rotation(M_PI, float3{1, 0, 0}) *
                       scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
        mat4 bottomDisk = scaled * translation(float3{0, 0, -size.z}) *
                          scaling(float3{kArrowHeadSize, kArrowHeadSize, 1.0f});
        out_shapes.push_back({kTube, tube});
        out_shapes.push_back({kCone, topCone});
        out_shapes.push_back({kCone, bottomCone});
        out_shapes.push_back({kDisk, topDisk});
        out_shapes.push_back({kDisk, bottomDisk});
    } else if (geom.type == mjGEOM_LINE) {
        out_shapes.push_back({kLine, base});
    } else if (geom.type == mjGEOM_LINEBOX) {
        out_shapes.push_back({kLineBox, base});
    } else {
        mju_warning("Unsupported geom type: %d", geom.type);
    }
}

Geom::Geom(const mjvGeom& geom)
    : geom_type(geom.type) {
    shapes.reserve(8);
    BuildShapes(geom, shapes);
}

void Geom::Update(const mjvGeom& geom) {
    geom_type = geom.type;
    std::vector<ShapeGeometry> new_shapes;
    new_shapes.reserve(shapes.capacity());
    BuildShapes(geom, new_shapes);
    shapes.swap(new_shapes);
}

// ---------------- Scene Implementation ----------------

Scene::Scene(const mjModel* model, const mjvScene* scene) {
    object_manager = std::make_unique<ObjectManager>(model, nullptr);
    for (int i = 0; i < scene->ngeom; ++i) {
        const mjvGeom* geom = scene->geoms + i;
        geoms.emplace_back(*geom);
    }
    createGeometryBuffers();
}

void Scene::createGeometryBuffers() {
    geometryBuffers.clear();
    transforms.clear();

    for (const auto& geom : geoms) {
        for (const auto& shape : geom.shapes) {
            const GeometryBuffers* buffers = object_manager->GetShapeBuffer(shape.type);
            if (buffers) {
                geometryBuffers.push_back(*buffers);
                transforms.push_back(shape.transform);
            }
        }
    }
}

void Scene::UpdateScene(const mjModel* model, const mjvScene* scene) {
    if (scene->ngeom != static_cast<int>(geoms.size())) {
        geoms.clear();
        for (int i = 0; i < scene->ngeom; ++i) {
            const mjvGeom* geom = scene->geoms + i;
            geoms.emplace_back(*geom);
        }
        createGeometryBuffers();
        return;
    }

    bool type_changed = false;
    for (int i = 0; i < scene->ngeom; ++i) {
        const mjvGeom* geom = scene->geoms + i;
        if (geom->type != geoms[i].geom_type) {
            type_changed = true;
        }
        geoms[i].Update(*geom);
    }

    if (type_changed) {
        createGeometryBuffers();
        return;
    }

    size_t transform_index = 0;
    for (const auto& geom : geoms) {
        for (const auto& shape : geom.shapes) {
            if (transform_index < transforms.size()) {
                transforms[transform_index] = shape.transform;
                transform_index++;
            }
        }
    }
}

}}  // namespace mujoco::mjbatch
