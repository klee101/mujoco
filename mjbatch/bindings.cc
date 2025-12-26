#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "renderer.h"

namespace py = pybind11;

PYBIND11_MODULE(mjb, m) {
    // Expose Config
    py::class_<BatchRendererConfig>(m, "BatchRendererConfig")
        .def(py::init<>())
        .def_readwrite("batch_size", &BatchRendererConfig::batch_size)
        .def_readwrite("frame_width", &BatchRendererConfig::frame_width)
        .def_readwrite("frame_height", &BatchRendererConfig::frame_height)
        .def_readwrite("gpu_id", &BatchRendererConfig::gpu_id)
        .def_readwrite("enable_validation", &BatchRendererConfig::enable_validation);

    // Expose Renderer
    py::class_<BatchRenderer>(m, "BatchRenderer")
        
        // [UPDATED] Constructor: Accepts list of Python Objects (mjModel wrappers)
        // We use py::object and reinterpret_cast/capsule logic to be robust
        .def(py::init([](std::vector<py::object> py_models, BatchRendererConfig config) {
            
            std::vector<mjModel*> models;
            models.reserve(py_models.size());

            for (auto& obj : py_models) {
                // Method A: If user passed an integer address (fallback)
                if (py::isinstance<py::int_>(obj)) {
                    models.push_back(reinterpret_cast<mjModel*>(obj.cast<uintptr_t>()));
                } 
                // Method B: DeepMind MuJoCo Binding (pybind11 wrapped)
                // If we linked against mujoco, pybind11 might cast it automatically.
                // However, without the type definition exposed here, we might need a hack.
                // 
                // TRYING DIRECT CAST:
                else {
                    // This relies on the fact that your C++ code includes mujoco.h
                    // and pybind11 might be able to find the pointer. 
                    // If this fails, we will need the user to use a specific property.
                    try {
                        // Attempt to extract pointer from capsule if it's mujoco-py
                        if (py::hasattr(obj, "_model_ptr")) {
                            py::object ptr = obj.attr("_model_ptr");
                            models.push_back(reinterpret_cast<mjModel*>(ptr.cast<uintptr_t>()));
                        }
                        // Attempt to extract from DeepMind MuJoCo (often difficult to cast directly)
                        // Hack: Get address of the pybind11 instance? 
                        // SAFE FIX: Ask Python to give us the address using ctypes in the script,
                        // OR just cast assuming we know the layout.
                        //
                        // Let's use the standard Cast. If this crashes, we use Method A in python.
                        models.push_back(obj.cast<mjModel*>());
                    } catch (...) {
                        throw std::runtime_error("Could not extract mjModel* from Python object. Please pass int(address) using ctypes helper.");
                    }
                }
            }
            
            return BatchRenderer::Create(models, config);
        }))

        // ... [Rest remains same] ...
        .def("render_from_shm", [](BatchRenderer& self, uintptr_t address, int batch_idx, int max_geom, int max_light) {
            return self.RenderFromMemory(reinterpret_cast<uint8_t*>(address), batch_idx, max_geom, max_light).IsSuccess();
        })
        .def("get_image", [](BatchRenderer& self, int batch_idx, int cam_idx) {
             int flat_idx = batch_idx * 3 + cam_idx; 
            
             const uint8_t* ptr = self.GetRGBFrame(flat_idx);
             if (!ptr) throw std::runtime_error("Invalid batch index or uninitialized");
             
             size_t h = self.GetConfig().frame_height; 
             size_t w = self.GetConfig().frame_width;
             
             return py::array_t<uint8_t>(
                {h, w, 4ul},
                {w * 4, 4ul, 1ul}, 
                ptr,
                py::cast(&self)
             );
        });
}