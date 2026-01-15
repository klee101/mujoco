#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <span>
#include "device.h"
#include "dispatch.hpp"

using DeviceID = std::array<uint8_t, VK_UUID_SIZE>;

namespace mujoco::mjbatch {

class LoaderLib {
public:
    LoaderLib(const LoaderLib &) = delete;
    ~LoaderLib();

    static LoaderLib * load();
    static LoaderLib * external(void (*entry_fn)());

    inline void (*getEntryFn() const)() { return entry_fn_; }

private:
    LoaderLib(void *lib, void (*entry_fn)(), const char *env_str);
    void *lib_;
    void (*entry_fn_)();

    const char *env_str_;
};

class Backend {
public:
    Backend(void (*vk_entry_fn)(),
            bool enable_validation,
            bool enable_present,
            std::span<const char *const> extra_vk_exts = {});

    Backend(const Backend &) = delete;
    Backend(Backend &&);
    ~Backend();

    Device * makeDevice(
        int64_t gpu_idx, std::span<const VkSurfaceKHR> present_surfaces = {});

    Device * makeDevice(
        const DeviceID &gpu_id, std::span<const VkSurfaceKHR> present_surfaces = {});

    VkInstance hdl;
    InstanceDispatch dt;

private:
    Device * makeDevice(
        VkPhysicalDevice phy, std::span<const VkSurfaceKHR> present_surfaces);

    struct Init;
    inline Backend(Init init, bool enable_present);

    const VkDebugUtilsMessengerEXT debug_;

    VkPhysicalDevice findPhysicalDevice(const DeviceID &id) const;
};

}