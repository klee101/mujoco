cmake_minimum_required(VERSION 3.14)
project(HLSLReflectorTest)

set(CMAKE_CXX_STANDARD 17)

# 1. Find Vulkan
find_package(Vulkan REQUIRED)

# 2. Find Shaderc
find_library(SHADERC_LIB shaderc_combined 
    HINTS $ENV{VULKAN_SDK}/lib
)
find_path(SHADERC_INCLUDE_DIR shaderc/shaderc.hpp
    HINTS $ENV{VULKAN_SDK}/include
)

# 3. Find SPIRV-Cross
# 修正点：必须显式查找 spirv_cross_glsl，且顺序很重要！
find_package(spirv_cross_core CONFIG REQUIRED)
find_package(spirv_cross_glsl CONFIG REQUIRED) # <--- 新增这一行
find_package(spirv_cross_cpp CONFIG REQUIRED)

add_executable(ReflectorTest main.cpp)

target_include_directories(ReflectorTest PRIVATE ${Vulkan_INCLUDE_DIRS} ${SHADERC_INCLUDE_DIR})

# 修正点：链接库中也加上 spirv-cross-glsl
target_link_libraries(ReflectorTest PRIVATE 
    ${Vulkan_LIBRARIES}
    ${SHADERC_LIB}
    spirv-cross-core
    spirv-cross-glsl  # <--- 新增这一行
    spirv-cross-cpp
)

# Copy shader file to build dir
configure_file(simple.hlsl simple.hlsl COPYONLY)