#include "hlsl_compiler.h"
#include <shaderc/shaderc.hpp>
#include <fstream>
#include <sstream>
#include <mutex>
#include <regex>
#include <iostream>

namespace mujoco::mjbatch::shader {

static std::vector<char> readFile(std::filesystem::path const &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file: " + filename.string());
    }
    size_t fileSize = (size_t)file.tellg();
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    file.close();
    return buffer;
}

// 缓存：Key 是 "filepath:entryPoint" (因为同一个文件可能有不同的入口)
static std::mutex g_shaderMutex;
static std::unordered_map<std::string, std::vector<std::uint32_t>> g_shaderCodeCache;

std::vector<std::uint32_t> const &
HLSLCompiler::compileHlslFileCached(vk::ShaderStageFlagBits stage, 
                                    fs::path const &filepath,
                                    std::string const &entryPoint) {
    std::string pathKey = fs::canonical(filepath).string() + ":" + entryPoint;
    
    std::lock_guard<std::mutex> guard(g_shaderMutex);
    if (g_shaderCodeCache.find(pathKey) != g_shaderCodeCache.end()) {
        return g_shaderCodeCache[pathKey];
    }

    auto [code, debugInfo] = HLSLCompiler::loadHlslCodeWithDebugInfo(filepath);
    return g_shaderCodeCache[pathKey] = HLSLCompiler::compileToSpirv(stage, code, entryPoint, debugInfo);
}

std::tuple<std::string, std::vector<std::tuple<std::string, int>>>
HLSLCompiler::loadHlslCodeWithDebugInfo(fs::path const &filepath) {
    std::vector<char> charCode = readFile(filepath);
    std::string code{charCode.begin(), charCode.end()};
    std::istringstream iss(code);
    std::string result;

    std::vector<std::tuple<std::string, int>> lineInfo;
    int lineNum = 1;

    for (std::string line; std::getline(iss, line); ++lineNum) {
        // remove CR for windows
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        std::string lineTrimmed = line;
        // left trim
        lineTrimmed.erase(lineTrimmed.begin(), 
            std::find_if(lineTrimmed.begin(), lineTrimmed.end(), [](unsigned char ch) { return !std::isspace(ch); }));

        if (lineTrimmed.rfind("#include", 0) == 0) {
            // Check if there is space after #include
            if (lineTrimmed.length() > 8 && std::isspace(lineTrimmed[8])) {
                std::string content = lineTrimmed.substr(8);
                
                // trim content
                content.erase(content.begin(), std::find_if(content.begin(), content.end(), [](unsigned char ch) { return !std::isspace(ch); }));
                content.erase(std::find_if(content.rbegin(), content.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), content.end());

                if (content.size() >= 2 && content.front() == '"' && content.back() == '"') {
                    std::string filename = content.substr(1, content.size() - 2);
                    auto includePath = filepath.parent_path() / filename;
                    
                    // 递归加载
                    auto [includeCode, includeInfo] = loadHlslCodeWithDebugInfo(includePath);

                    lineInfo.insert(lineInfo.end(), includeInfo.begin(), includeInfo.end());
                    result += includeCode + "\n"; // Add newline to be safe
                } else {
                    throw std::runtime_error("invalid include directive: " + line + " in " + filepath.string());
                }
            } else {
                // #includeSomething (not an include directive)
                lineInfo.push_back({filepath.string(), lineNum});
                result += line + "\n";
            }
        } else {
            lineInfo.push_back({filepath.string(), lineNum});
            result += line + "\n";
        }
    }
    return {result, lineInfo};
}

// 辅助：将 Vulkan Stage 转换为 shaderc shader kind
static shaderc_shader_kind GetShadercKind(vk::ShaderStageFlagBits stage) {
    switch (stage) {
    case vk::ShaderStageFlagBits::eVertex: return shaderc_vertex_shader;
    case vk::ShaderStageFlagBits::eFragment: return shaderc_fragment_shader;
    case vk::ShaderStageFlagBits::eGeometry: return shaderc_geometry_shader;
    case vk::ShaderStageFlagBits::eCompute: return shaderc_compute_shader;
    case vk::ShaderStageFlagBits::eTessellationControl: return shaderc_tess_control_shader;
    case vk::ShaderStageFlagBits::eTessellationEvaluation: return shaderc_tess_evaluation_shader;
    // Ray Tracing stages need shaderc 2020+
    case vk::ShaderStageFlagBits::eRaygenKHR: return shaderc_raygen_shader;
    case vk::ShaderStageFlagBits::eAnyHitKHR: return shaderc_anyhit_shader;
    case vk::ShaderStageFlagBits::eClosestHitKHR: return shaderc_closesthit_shader;
    case vk::ShaderStageFlagBits::eMissKHR: return shaderc_miss_shader;
    case vk::ShaderStageFlagBits::eIntersectionKHR: return shaderc_intersection_shader;
    case vk::ShaderStageFlagBits::eCallableKHR: return shaderc_callable_shader;
    default: throw std::runtime_error("Unsupported shader stage for shaderc compilation");
    }
}

std::vector<std::uint32_t>
HLSLCompiler::compileToSpirv(vk::ShaderStageFlagBits shaderStage, 
                             std::string const &hlslCode,
                             std::string const &entryPoint,
                             std::vector<std::tuple<std::string, int>> const &debugInfo) {
    
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // 1. 设置源语言为 HLSL
    options.SetSourceLanguage(shaderc_source_language_hlsl);
    
    // 2. 设置 Vulkan 目标环境 (Vulkan 1.1 / SPIR-V 1.3 是目前的通用基准)
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    
    // 3. 自动绑定 (这对 HLSL 很重要，自动处理 register 到 binding 的映射)
    // 注意：如果你希望完全手动控制 binding，可以关掉这个，但通常开启比较方便
    options.SetAutoMapLocations(true); 
    options.SetAutoBindUniforms(true); 

    // 4. 生成调试信息 (为了后续反射能读出变量名)
    options.SetGenerateDebugInfo();
    // options.SetOptimizationLevel(shaderc_optimization_level_performance); // 生产环境开启

    // 编译
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
        hlslCode, 
        GetShadercKind(shaderStage), 
        "input.hlsl", // 虚拟文件名，因为我们手动处理了 include
        entryPoint.c_str(), 
        options
    );

    // 错误处理与行号映射
    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::string rawMsg = module.GetErrorMessage();
        std::stringstream ss(rawMsg);
        std::string line;
        
        // Shaderc 错误格式通常为: input.hlsl:<line>: error: <msg>
        // 我们需要把 <line> 映射回原始文件的行号
        while (std::getline(ss, line, '\n')) {
            std::regex pattern("^input\\.hlsl:([0-9]+): (.*)$");
            std::smatch sm;
            if (std::regex_search(line, sm, pattern) && sm.size() == 3) {
                uint32_t globalLine = std::stoi(sm[1]);
                std::string reason = sm[2];
                
                // 映射回原始文件
                // 注意：shaderc 报错行号可能因为 define 略有偏差，但通常 globalLine-1 是对应的
                if (globalLine > 0 && globalLine <= debugInfo.size()) {
                    auto [origFile, origLine] = debugInfo[globalLine - 1];
                    // Logger 替换为你的日志系统
                    std::cerr << "Shader Error [" << entryPoint << "] " 
                              << origFile << ":" << origLine << ": " << reason << std::endl;
                } else {
                    std::cerr << line << std::endl;
                }
            } else {
                std::cerr << line << std::endl;
            }
        }
        throw std::runtime_error("Failed to compile HLSL shader: " + entryPoint);
    }

    return {module.cbegin(), module.cend()};
}

} // namespace mujoco::mjbatch