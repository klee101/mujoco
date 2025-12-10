cmake_minimum_required(VERSION 3.16)
project(HLSLCompiler)

set(CMAKE_CXX_STANDARD 17)

# DXC 路径
set(DXC_ROOT ${CMAKE_SOURCE_DIR}/mjbatch/dxc)
set(DXC_INCLUDE ${DXC_ROOT}/include)
set(DXC_LIB ${DXC_ROOT}/lib)

# Shader 源目录和输出目录
set(SHADER_SOURCES ${CMAKE_SOURCE_DIR}/mjbatch/shaders)
set(SHADER_OUTPUT ${CMAKE_BINARY_DIR}/shaders_spv)

file(MAKE_DIRECTORY ${SHADER_OUTPUT})

# 查找所有 .hlsl 文件
file(GLOB_RECURSE HLSL_FILES "${SHADER_SOURCES}/*.hlsl")

# 编译工具
add_executable(hlsl_compile compile.cpp)
target_include_directories(hlsl_compile PRIVATE ${DXC_INCLUDE})
target_link_libraries(hlsl_compile PRIVATE
    ${DXC_LIB}/libdxcompiler.so
    dl
)

set(SPV_FILES)

foreach(HLSL_FILE ${HLSL_FILES})
    get_filename_component(FILE_NAME ${HLSL_FILE} NAME_WE)
    get_filename_component(FILE_EXT ${HLSL_FILE} EXT)
    get_filename_component(FILE_PATH ${HLSL_FILE} PATH)

    # 输出 SPV 文件路径
    set(OUTPUT_SPV "${SHADER_OUTPUT}/${FILE_NAME}.spv")

    # 文件名包含 "vs" -> Vertex Shader，"ps" -> Pixel Shader，"cs" -> Compute Shader
    string(TOLOWER ${FILE_NAME} FILE_NAME_LOWER)
    if(FILE_NAME_LOWER MATCHES ".*vs.*")
        set(TARGET "vs")
        set(ENTRY "VSMain")
    elseif(FILE_NAME_LOWER MATCHES ".*ps.*")
        set(TARGET "ps")
        set(ENTRY "PSMain")
    elseif(FILE_NAME_LOWER MATCHES ".*cs.*")
        set(TARGET "cs")
        set(ENTRY "CSMain")
    else()
        # 默认使用 Vertex Shader
        set(TARGET "vs")
        set(ENTRY "VSMain")
    endif()

    # 添加自定义命令
    add_custom_command(
        OUTPUT ${OUTPUT_SPV}
        COMMAND hlsl_compile ${HLSL_FILE} ${TARGET} ${ENTRY} ${OUTPUT_SPV}
        DEPENDS hlsl_compile ${HLSL_FILE}
        COMMENT "Compiling shader ${HLSL_FILE} -> ${OUTPUT_SPV}"
        VERBATIM
    )

    list(APPEND SPV_FILES ${OUTPUT_SPV})
endforeach()

add_custom_target(CompileShaders ALL DEPENDS ${SPV_FILES})

# feat: shader_compiler for reflection
add_library(shader_compiler SHARED
    reflection.h
    shader_compiler.h shader_compiler.cpp
)
target_include_directories(shader_compiler PRIVATE ${DXC_INCLUDE})

target_link_libraries(shader_compiler PRIVATE
    ${DXC_LIB}/libdxcompiler.so
    spv_reflect
)

target_compile_features(shader_compiler PUBLIC cxx_std_20)