#pragma once

#include <vulkan/vulkan.h>

namespace mujoco {
    namespace mjbatch {
struct InstanceDispatch {
#include "dispatch/dispatch_instance_impl.hpp"

    InstanceDispatch(VkInstance inst, PFN_vkGetInstanceProcAddr get_inst_addr,
                     bool need_present);
};

struct DeviceDispatch {
#include "dispatch/dispatch_device_impl.hpp"

    DeviceDispatch(VkDevice dev, PFN_vkGetDeviceProcAddr get_dev_addr,
                   bool need_present, bool need_rt, bool support_mem_export);
};
    }}
     