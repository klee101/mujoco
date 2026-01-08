cmake_minimum_required(VERSION 3.14)
project(HLSLCompilerTest)

set(CMAKE_CXX_STANDARD 17)

# ==============================================================================
# 1. 验证 SDK 环境变量
# ==============================================================================
if(NOT DEFINED ENV{VULKAN_SDK})
    message(FATAL_ERROR "VULKAN_SDK not set!")
else()
    message(STATUS "SDK Path: $ENV{VULKAN_SDK}")
    include_directories("$ENV{VULKAN_SDK}/include")
    link_directories("$ENV{VULKAN_SDK}/lib")
endif()

find_package(Vulkan REQUIRED)

# ==============================================================================
# 2. 查找库文件 (显式指定静态库名称)
# ==============================================================================

# 查找 shaderc_combined (通常是静态库) 或 shaderc_shared
find_library(SHADERC_LIB 
    NAMES shaderc_combined shaderc_shared 
    HINTS "$ENV{VULKAN_SDK}/lib"
    NO_DEFAULT_PATH
)

# 查找 SPIRV-Cross 静态库
# 注意：find_library 在 Linux 下会自动补全 "lib" 前缀和 ".a" 后缀
# 所以 NAMES 写 "spirv-cross-core" 就能找到 "libspirv-cross-core.a"
find_library(SPIRV_CROSS_CORE_LIB NAMES spirv-cross-core HINTS "$ENV{VULKAN_SDK}/lib" NO_DEFAULT_PATH)
find_library(SPIRV_CROSS_GLSL_LIB NAMES spirv-cross-glsl HINTS "$ENV{VULKAN_SDK}/lib" NO_DEFAULT_PATH)
find_library(SPIRV_CROSS_CPP_LIB  NAMES spirv-cross-cpp  HINTS "$ENV{VULKAN_SDK}/lib" NO_DEFAULT_PATH)

# ==============================================================================
# 3. 调试打印 (确认到底找到了没)
# ==============================================================================
message(STATUS "-------------------------------------------------------------")
message(STATUS "  Shaderc Lib : ${SHADERC_LIB}")
message(STATUS "  Cross Core  : ${SPIRV_CROSS_CORE_LIB}")
message(STATUS "  Cross GLSL  : ${SPIRV_CROSS_GLSL_LIB}")
message(STATUS "  Cross CPP   : ${SPIRV_CROSS_CPP_LIB}")
message(STATUS "-------------------------------------------------------------")

if(NOT (SPIRV_CROSS_CORE_LIB AND SPIRV_CROSS_GLSL_LIB AND SPIRV_CROSS_CPP_LIB))
    message(FATAL_ERROR "Failed to find SPIRV-Cross libraries! Please check the paths above.")
endif()

# ==============================================================================
# 4. 构建目标
# ==============================================================================
add_executable(CompilerTest 
    test_compiler.cc 
    hlsl_compiler.cc 
    hlsl_compiler.h
)

# ==============================================================================
# 5. 链接 (注意静态库顺序！)
# ==============================================================================
# 静态库链接顺序原则：依赖别人的放在前面。
# CPP 依赖 GLSL，GLSL 依赖 Core。
target_link_libraries(CompilerTest PRIVATE 
    ${Vulkan_LIBRARIES}
    ${SHADERC_LIB}
    ${SPIRV_CROSS_CPP_LIB}   # 1. CPP (最上层)
    ${SPIRV_CROSS_GLSL_LIB}  # 2. GLSL (中间层)
    ${SPIRV_CROSS_CORE_LIB}  # 3. Core (底层)
)

# 静态链接时通常需要系统库
target_link_libraries(CompilerTest PRIVATE pthread dl)

# 拷贝 HLSL 文件
configure_file(test_shader.hlsl test_shader.hlsl COPYONLY)