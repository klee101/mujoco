# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.16)

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

# This line has to appear before 'PROJECT' in order to be able to disable incremental linking
set(MSVC_INCREMENTAL_DEFAULT ON)

project(
  mujoco_samples
  VERSION 3.3.8
  DESCRIPTION "MuJoCo samples binaries"
  HOMEPAGE_URL "https://mujoco.org"
)

enable_language(C)
enable_language(CXX)


# Check if we are building as standalone project.

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(SampleOptions)
include(SampleDependencies)

set(MUJOCO_SAMPLE_COMPILE_OPTIONS "${AVX_COMPILE_OPTIONS}" "${EXTRA_COMPILE_OPTIONS}")
set(MUJOCO_SAMPLE_LINK_OPTIONS "${EXTRA_LINK_OPTIONS}")

# Enable hardening options by default
# for testing, set MUJOCO_HARDEN to OFF, ignore the unused func and variable warning
set(MUJOCO_HARDEN OFF)

if(MUJOCO_HARDEN)
  if(WIN32)
    set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -Wl,/DYNAMICBASE)
  else()
    set(MUJOCO_SAMPLE_COMPILE_OPTIONS "${MUJOCO_SAMPLE_COMPILE_OPTIONS}" -fPIE)
    if(APPLE)
      set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -Wl,-pie)
    else()
      set(MUJOCO_SAMPLE_LINK_OPTIONS "${MUJOCO_SAMPLE_LINK_OPTIONS}" -pie)
    endif()
  endif()
endif()

# Hack to prefer linking against vulkan sdk
IF (DEFINED ENV{VULKAN_SDK})
    IF (DEFINED CMAKE_PREFIX_PATH)
        set(ORIG_PREFIX_PATH CMAKE_PREFIX_PATH)
    ENDIF()
    set(CMAKE_PREFIX_PATH "$ENV{VULKAN_SDK}")
ENDIF()

find_package(Vulkan REQUIRED)

IF (DEFINED ENV{VULKAN_SDK})
    IF (DEFINED ORIG_PREFIX_PATH)
        set(CMAKE_PREFIX_PATH ORIG_PREFIX_PATH)
    ELSE()
        unset(CMAKE_PREFIX_PATH)
    endif()
ENDIF()

# Build vulkan dispatch table
get_directory_property(DIR_DEFINES DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS )
set(DEFINE_STR "")
if (FRONTEND_MSVC)
    set(DEFINE_PREFIX "/D")
else()
    set(DEFINE_PREFIX "-D")
endif()
foreach (f ${DIR_DEFINES})
    list(APPEND DEFINE_STR "${DEFINE_PREFIX}${f}")
endforeach()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/dispatch")

if (FRONTEND_MSVC)
    set(CXX_PREPROCESSOR_CMD
        ${CMAKE_CXX_COMPILER} /nologo /TP /P /EP /C /X ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" /Fidispatch/dispatch.template)
else()
    set(CXX_PREPROCESSOR_CMD
        ${CMAKE_CXX_COMPILER} -x c -E -P -nostdinc ${DEFINE_STR} "${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template" -o dispatch/dispatch.template)
endif()

add_custom_command(
    OUTPUT dispatch/dispatch.template
    COMMAND ${CXX_PREPROCESSOR_CMD} DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch.template
)

if(NOT Python_EXECUTABLE)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(Python_EXECUTABLE ${Python3_EXECUTABLE})
endif()

add_custom_command(
    OUTPUT dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
           dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_dispatch.py
            ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template dispatch
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dispatch/dispatch.template
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/gen_dispatch.py
)

add_custom_target(generate_dispatch DEPENDS
    dispatch/dispatch_device_impl.hpp dispatch/dispatch_device_impl.cpp
    dispatch/dispatch_instance_impl.hpp dispatch/dispatch_instance_impl.cpp
)


# Build renderer library.
add_library(renderer STATIC
  renderer.h           renderer.cc
  render_context.h     render_context.cc
  device.h             device.cc
  memory.h             memory.cc
  backend.h            backend.cc
  dispatch.hpp         dispatch.cpp
  scene.h              scene.cc
  object_manager.h     object_manager.cc
  builtin.h            builtin.cc
  utils.h              
  vkutils.h            vkutils.cc           vkutils.inl
)
target_compile_options(renderer PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_include_directories(renderer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

target_link_libraries(renderer PUBLIC
  mujoco::mujoco
  glfw
  Threads::Threads
  Vulkan::Vulkan
  glm
)

add_dependencies(renderer generate_dispatch)

# Build samples that require GLFW.
add_executable(record_vk record_vk.cc array_safety.h)
target_compile_options(record_vk PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
  record_vk
  renderer
)
target_link_options(record_vk PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

add_executable(vertex_check vertex_check.cc array_safety.h)
target_compile_options(vertex_check PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
  vertex_check
  renderer
)
target_link_options(vertex_check PRIVATE ${MUJOCO_SAMPLE_LINK_OPTIONS})

set(DXC_EXECUTABLE "${CMAKE_SOURCE_DIR}/mjbatch/dxc/bin/dxc")
add_subdirectory(shaders)

add_executable(mjbatch_smoke tests/smoke_test_main.cc)
target_compile_options(mjbatch_smoke PUBLIC ${MUJOCO_SAMPLE_COMPILE_OPTIONS})
target_link_libraries(
    mjbatch_smoke
    renderer
)

