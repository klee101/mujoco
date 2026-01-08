#pragma once

#include <filesystem>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <vulkan/vulkan.hpp>

namespace mujoco::mjbatch::shader {

namespace fs = std::filesystem;

class HLSLCompiler {
public:
    /**
     * @brief 编译 HLSL 文件并缓存 SPIR-V 结果
     * @param stage 着色器阶段 (Vertex, Fragment 等)
     * @param filepath HLSL 文件路径
     * @param entryPoint 入口函数名 (默认为 "main", 但通常建议显式指定如 "VSMain"/"PSMain")
     */
    static std::vector<std::uint32_t> const &
    compileHlslFileCached(vk::ShaderStageFlagBits stage,
                          fs::path const &filepath,
                          std::string const &entryPoint = "main");

    /**
     * @brief 递归加载 HLSL 代码，并处理 #include 指令
     * @return tuple<合并后的代码, 行号调试映射表>
     */
    static std::tuple<std::string, std::vector<std::tuple<std::string, int>>>
    loadHlslCodeWithDebugInfo(fs::path const &filepath);

    /**
     * @brief 将 HLSL 源码编译为 SPIR-V
     */
    static std::vector<std::uint32_t>
    compileToSpirv(vk::ShaderStageFlagBits shaderStage,
                   std::string const &hlslCode,
                   std::string const &entryPoint = "main",
                   std::vector<std::tuple<std::string, int>> const &debugInfo = {});

    // 全局初始化/清理 (虽然 shaderc 不需要像 glslang 那样显式初始化，但为了接口一致性保留)
    static void InitializeProcess() {}
    static void FinalizeProcess() {}
};

} // namespace mujoco::mjbatch