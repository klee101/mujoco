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
        
        // Constructor: Accepts list of Python Objects (mjModel wrappers)
        .def(py::init([](std::vector<py::object> py_models, BatchRendererConfig config) {
            
            std::vector<mjModel*> models;
            models.reserve(py_models.size());

            for (auto& obj : py_models) {
                // Method A: If user passed an integer address (fallback)
                if (py::isinstance<py::int_>(obj)) {
                    models.push_back(reinterpret_cast<mjModel*>(obj.cast<uintptr_t>()));
                } 
                else {
                    try {
                        if (py::hasattr(obj, "_model_ptr")) {
                            py::object ptr = obj.attr("_model_ptr");
                            models.push_back(reinterpret_cast<mjModel*>(ptr.cast<uintptr_t>()));
                        }
                        models.push_back(obj.cast<mjModel*>());
                    } catch (...) {
                        throw std::runtime_error("Could not extract mjModel* from Python object. Please pass int(address) using ctypes helper.");
                    }
                }
            }
            
            return BatchRenderer::Create(models, config);
        }))

        // ── RenderDoc ──────────────────────────────────────────────────────────
        .def("start_capture", &BatchRenderer::StartCapture,
             "Manually start RenderDoc frame capture")
        .def("end_capture", &BatchRenderer::EndCapture,
             "Manually end and save RenderDoc frame capture")

        // ── Synchronous render (legacy) ────────────────────────────────────────
        .def("render_from_shm", [](BatchRenderer& self, uintptr_t address,
                                    int batch_idx, int max_geom, int max_light) {
            return self.RenderFromMemory(
                reinterpret_cast<uint8_t*>(address),
                batch_idx, max_geom, max_light).IsSuccess();
        }, py::arg("shm_ptr"), py::arg("batch_idx"),
           py::arg("max_geom"), py::arg("max_light"))

        // ── Async pipeline interfaces ──────────────────────────────────────────

        // Phase 1: Upload UBOs via transfer queue (non-blocking, signals transfer_semaphore)
        .def("update_async", [](BatchRenderer& self, uintptr_t address,
                                 int max_geom, int max_light) {
            return self.UpdateAsync(
                reinterpret_cast<uint8_t*>(address), max_geom, max_light);
        }, py::arg("shm_ptr"), py::arg("max_geom"), py::arg("max_light"),
           "Upload camera/light UBOs asynchronously via transfer queue.\n"
           "Returns True on success. Must be called before record_next().")

        // Phase 2: Record draw commands into current swap slot
        // Returns the slot index that was recorded (pass to wait_slot later)
        .def("record_next", [](BatchRenderer& self, uintptr_t address) {
            return self.RecordNext(reinterpret_cast<const uint8_t*>(address));
        }, py::arg("shm_ptr"),
           "Record draw commands for the current swap slot.\n"
           "Returns slot_idx (int) to be passed to wait_slot(), or -1 on error.")

        // Phase 3: Submit recorded commands to GPU + kick off async readback
        .def("submit_next", &BatchRenderer::SubmitNext,
             "Submit the recorded command buffer to the GPU render queue.\n"
             "Also submits the readback blit for the current slot.\n"
             "Returns True on success.")

        // Phase 4: Wait for a specific slot's readback fence to complete
        // This is the CPU sync point; call with the slot returned by record_next()
        .def("wait_slot", &BatchRenderer::WaitSlot, py::arg("slot_idx"),
             "Block until the readback for slot_idx is GPU-complete.\n"
             "After this returns, get_image() is valid for that slot's frames.")

        // ── Readback thread control ────────────────────────────────────────────
        // The readback thread watches for READBACK_PENDING slots and transitions
        // them back to FREE after the readback fence is signaled.
        // You MUST call start_readback_thread() before using the async pipeline,
        // otherwise slots will never return to FREE and the pipeline will deadlock.
        .def("start_readback_thread", &BatchRenderer::StartReadbackThread,
             "Start the background readback thread.\n"
             "REQUIRED before using update_async/record_next/submit_next/wait_slot.\n"
             "The thread transitions slot state READBACK_PENDING -> FREE after GPU readback.")

        .def("stop_readback_thread", &BatchRenderer::StopReadbackThread,
             "Stop the background readback thread.\n"
             "Call this before destroying the renderer or at end of test.")

        // ── Set readback callback (optional) ──────────────────────────────────
        // Callback signature: fn(step_id: int, frames: list[np.ndarray])
        // Called from the readback thread when a slot's pixel data is ready.
        .def("set_readback_callback", [](BatchRenderer& self, py::object callback) {
            if (callback.is_none()) {
                self.SetReadbackCallback(nullptr);
                return;
            }
            // Capture callback; note this is called from the readback thread,
            // so acquire the GIL before invoking Python.
            self.SetReadbackCallback(
                [callback](int step_id, const std::vector<FrameObservation>& obs) {
                    py::gil_scoped_acquire gil;
                    py::list frames;
                    for (const auto& f : obs) {
                        // Zero-copy view into the staging buffer (valid until next WaitSlot)
                        py::array_t<uint8_t> arr(
                            {(ssize_t)f.height, (ssize_t)f.width, (ssize_t)4},
                            {(ssize_t)f.stride_bytes, (ssize_t)4, (ssize_t)1},
                            f.data,
                            py::none()  // no base: caller must not hold beyond callback
                        );
                        frames.append(arr);
                    }
                    callback(step_id, frames);
                });
        }, py::arg("callback"),
           "Set a Python callback invoked from the readback thread when frames are ready.\n"
           "Signature: callback(step_id: int, frames: List[np.ndarray[H,W,4]])\n"
           "Pass None to clear the callback.")

        // ── Image retrieval ────────────────────────────────────────────────────
        // get_image reads from the latest completed readback staging buffer.
        // Only valid after wait_slot() has returned for the corresponding slot.
        .def("get_image", [](BatchRenderer& self, int batch_idx, int cam_idx) {
            int total_cams = 3; // SHM_NUM_CAMERAS
            int flat_idx = batch_idx * total_cams + cam_idx;

            const uint8_t* ptr = self.GetRGBFrame(flat_idx);
            if (!ptr) {
                throw std::runtime_error(
                    "Invalid batch index or uninitialized. "
                    "Ensure wait_slot() has completed for this frame.");
            }

            size_t h = self.GetConfig().frame_height;
            size_t w = self.GetConfig().frame_width;

            // Copy into a new numpy array so lifetime is independent of renderer
            py::array_t<uint8_t> result({h, w, (size_t)4});
            std::memcpy(result.mutable_data(), ptr, h * w * 4);
            return result;
        }, py::arg("batch_idx"), py::arg("cam_idx"),
           "Get rendered frame as HxWx4 numpy array (RGBA, uint8).\n"
           "Only valid after wait_slot() has returned for the corresponding slot.\n"
           "cam_idx: 0..2 (SHM_NUM_CAMERAS-1)")

        // get_image_view: zero-copy view (unsafe: only valid until next render cycle)
        .def("get_image_view", [](BatchRenderer& self, int batch_idx, int cam_idx) {
            int total_cams = 3; // SHM_NUM_CAMERAS
            int flat_idx = batch_idx * total_cams + cam_idx;

            const uint8_t* ptr = self.GetRGBFrame(flat_idx);
            if (!ptr) {
                throw std::runtime_error(
                    "Invalid batch index or uninitialized. "
                    "Ensure wait_slot() has completed for this frame.");
            }

            size_t h = self.GetConfig().frame_height;
            size_t w = self.GetConfig().frame_width;

            // Zero-copy: caller must not use after next render cycle
            return py::array_t<uint8_t>(
                {h, w, (size_t)4},
                {w * 4, (size_t)4, (size_t)1},
                ptr,
                py::cast(&self)  // keep renderer alive as long as array lives
            );
        }, py::arg("batch_idx"), py::arg("cam_idx"),
           "Zero-copy view of rendered frame as HxWx4 numpy array (RGBA, uint8).\n"
           "WARNING: data is invalid after the next render cycle. Use get_image() for safety.")

        // ── Stats / Config ─────────────────────────────────────────────────────
        .def("get_config", &BatchRenderer::GetConfig,
             py::return_value_policy::reference_internal);
}