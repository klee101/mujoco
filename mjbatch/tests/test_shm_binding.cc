#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mujoco/mujoco.h>
#include <iostream>
#include <vector>
#include <cstring> 

namespace py = pybind11;

class SharedSceneBinder {
public:
    mjvScene scene;
    mjvOption opt;
    mjvCamera cam;
    std::vector<int> local_geomorder;

    int* shm_ngeom_ptr = nullptr;
    int* shm_nlight_ptr = nullptr;
    mjvLight* shm_lights_ptr = nullptr;

    SharedSceneBinder() {
        mjv_defaultScene(&scene);
        mjv_defaultOption(&opt);
        mjv_defaultCamera(&cam);
    }

    void bind(py::buffer buffer, int maxgeom) {
        py::buffer_info info = buffer.request();
        char* ptr = static_cast<char*>(info.ptr);

        shm_ngeom_ptr = reinterpret_cast<int*>(ptr);
        shm_nlight_ptr = reinterpret_cast<int*>(ptr + sizeof(int));

        size_t offset = 2 * sizeof(int);
        scene.maxgeom = maxgeom;
        scene.geoms = reinterpret_cast<mjvGeom*>(ptr + offset);
        
        local_geomorder.resize(maxgeom);
        scene.geomorder = local_geomorder.data();

        offset += maxgeom * sizeof(mjvGeom);
        shm_lights_ptr = reinterpret_cast<mjvLight*>(ptr + offset);

        // Set the scene scale manually
        scene.scale = 1.0f; 

        std::cout << "[C++] Bind successful. scale initialized to 1.0" << std::endl;
    }

    void update(uintptr_t m_ptr, uintptr_t d_ptr) {
        const mjModel* m = reinterpret_cast<const mjModel*>(m_ptr);
        mjData* d = reinterpret_cast<mjData*>(d_ptr);

        if (!m || !d) return;

        {
            py::gil_scoped_release release;
            mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scene);
        }

        if (shm_ngeom_ptr) *shm_ngeom_ptr = scene.ngeom;
        if (shm_nlight_ptr) *shm_nlight_ptr = scene.nlight;

        if (shm_lights_ptr && scene.nlight > 0) {
            std::memcpy(shm_lights_ptr, scene.lights, scene.nlight * sizeof(mjvLight));
        }
    }

    static size_t get_required_size(int maxgeom) {
        return 2 * sizeof(int) + maxgeom * sizeof(mjvGeom) + 20 * sizeof(mjvLight);
    }
    
    static py::dict get_offsets() {
        py::dict d;
        d["header_size"] = 2 * sizeof(int);
        return d;
    }
};

PYBIND11_MODULE(test_shm_ext, m) {
    py::class_<SharedSceneBinder>(m, "SharedSceneBinder")
        .def(py::init<>())
        .def("bind", &SharedSceneBinder::bind)
        .def("update", &SharedSceneBinder::update)
        .def_static("get_required_size", &SharedSceneBinder::get_required_size)
        .def_static("get_offsets", &SharedSceneBinder::get_offsets);
}