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
        .def_readwrite("enable_validation", &BatchRendererConfig::enable_validation)
        .def_readwrite("debug_logging", &BatchRendererConfig::debug_logging);

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

        .def("record_next_nowait", [](BatchRenderer& self, uintptr_t address, 
                                int max_geom, int max_light) {
            return self.RecordNextNoWait(
                reinterpret_cast<const uint8_t*>(address), max_geom, max_light);
        }, py::arg("shm_ptr"), py::arg("max_geom"), py::arg("max_light"),
        "Record + upload UBO for next frame. Returns slot_idx or -1 if no slot free.")

        .def("submit_next", [](BatchRenderer& self, int slot_idx) {
            return self.SubmitNext(slot_idx);
        }, py::arg("slot_idx"),
        "Submit recorded commands for slot_idx to GPU.")

        // Phase 4: Wait for a specific slot's readback fence to complete
        // This is the CPU sync point; call with the slot returned by record_next()
        .def("wait_slot", &BatchRenderer::WaitSlot, py::arg("slot_idx"),
            "Block until GPU rendering for slot_idx is complete (NOT readback).\n"
            "After this returns, the slot can be reused for the next frame.\n"
            "For pixel data, call wait_readback(slot_idx) or get_image(slot_idx, ...).")

        .def("is_slot_ready", &BatchRenderer::IsSlotReady, py::arg("slot_idx"),
            "Non-blocking check whether GPU rendering for slot_idx is complete.\n"
            "Returns True if done, False if still in flight.\n"
            "Use this instead of wait_slot() in a pipelined loop to avoid blocking.")

        .def("wait_readback", &BatchRenderer::WaitReadback, py::arg("slot_idx"),
            "Block until readback (PCIe transfer) for slot_idx is complete.\n"
            "Must be called before get_image() for the corresponding slot.")

        .def("submit_readback", &BatchRenderer::SubmitReadbackForSlot,
            py::arg("slot_idx"),
            "Explicitly submit readback blit for slot_idx after GPU render completes.\n"
            "Usually not needed: the readback thread triggers this automatically.\n"
            "Returns True if submitted, False if GPU render not yet complete (retry later).")


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
            py::call_guard<py::gil_scoped_release>(),
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
        .def("get_image", [](BatchRenderer& self, int slot_idx, int batch_idx, int cam_idx) {
            // 等该 slot 的 readback 完成
            self.WaitReadback(slot_idx);

            size_t h = self.GetConfig().frame_height;
            size_t w = self.GetConfig().frame_width;
            int total_cams = 3; // SHM_NUM_CAMERAS
            int resource_idx = batch_idx * total_cams + cam_idx;

            py::array_t<uint8_t> result({h, w, (size_t)4});
            // 从 staging buffer 直接读（WaitReadback 已保证数据就绪）
            self.CopyFrameFromStaging(slot_idx, resource_idx, result.mutable_data(), h * w * 4);
            return result;
        }, py::arg("slot_idx"), py::arg("batch_idx"), py::arg("cam_idx"),
        "Get rendered frame as HxWx4 numpy array (RGBA, uint8).\n"
        "Internally calls wait_readback(slot_idx) to ensure data is ready.\n"
        "slot_idx: returned by record_next(); batch_idx: env index; cam_idx: 0..2")

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
