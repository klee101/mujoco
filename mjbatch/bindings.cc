// [NEW FILE] bindings.cc
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
        .def(py::init([](std::string xml_path, BatchRendererConfig config) {
            // Load Model (Helper for Python convenience)
            char error[1000];
            mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, 1000);
            if (!m) throw std::runtime_error(error);
            
            // We transfer ownership of mjModel to a vector for the Create function
            // Note: In real app, manage lifecycle carefully.
            std::vector<mjModel*> models(config.batch_size, m); 
            return BatchRenderer::Create(models, config);
        }))
        // The Main Shared Memory Interface
        .def("render_from_shm", [](BatchRenderer& self, uintptr_t address, int batch_idx, int max_geom, int max_light) {
            return self.RenderFromMemory(reinterpret_cast<uint8_t*>(address), batch_idx, max_geom, max_light).IsSuccess();
        })
        // Zero-Copy Image Access
        .def("get_image", [](BatchRenderer& self, int batch_idx) {
             auto* ptr = self.GetRGBFrame(batch_idx);
             if (!ptr) throw std::runtime_error("Invalid batch index or uninitialized");
             
             // Return numpy array sharing memory with C++ buffer
             size_t h = self.GetConfig().frame_height; // Need to expose GetConfig or store width/height
             size_t w = self.GetConfig().frame_width;
             
             return py::array_t<uint8_t>(
                {h, w, 4ul}, // Shape
                {w * 4, 4ul, 1ul}, // Strides
                ptr, // Data pointer
                py::cast(&self) // Keep renderer alive while array exists
             );
        });
}