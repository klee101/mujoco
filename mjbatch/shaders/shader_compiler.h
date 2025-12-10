#pragma once

#include <memory>
#include <vector>
#include <span>
#include "reflection.h"
#define COMPILER_FUNCTION_NAME __PRETTY_FUNCTION__
#define CountT int64_t

namespace mujoco::mjbatch {

struct CrashInfo {
    const char *file;
    int line;
    const char *funcname;
    const char *msg;
};

[[noreturn]] void fatal(const char *file, int line,
    const char *funcname, const char *fmt, ...);
[[noreturn]] void fatal(const CrashInfo &crash);

#define FATAL(fmt, ...) \
    ::mujoco::mjbatch::fatal(__FILE__, __LINE__, COMPILER_FUNCTION_NAME, fmt __VA_OPT__(,) __VA_ARGS__)

struct SPIRVShader {
    std::vector<uint32_t> bytecode;
    refl::SPIRV reflectionInfo;
};


class ShaderCompiler {
public:
    ShaderCompiler();
    ~ShaderCompiler();

    struct MacroDefn {
        const char *name;
        const char *value;
    };

    struct EntryConfig {
        const char *func;
        ShaderStage stage;
    };

    // If entry is default / not provided, SPIRVShader will
    // have multiple entry points.
    SPIRVShader compileHLSLFileToSPV(
        const char *path,
        std::span<const char *const> include_dirs,
        std::span<const MacroDefn> macro_defns,
        EntryConfig entry = { nullptr, ShaderStage {}});


private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
