#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <iomanip>

// Shaderc (Compiler)
#include <shaderc/shaderc.hpp>

// SPIRV-Cross (Reflection)
#include <spirv_cross/spirv_cross.hpp>

// 辅助：读取文件
std::string readFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// 辅助：将 SPIRV-Cross 类型转换为字符串
std::string getTypeName(spirv_cross::SPIRType::BaseType type) {
    using namespace spirv_cross;
    switch (type) {
        case SPIRType::Float: return "Float";
        case SPIRType::Int:   return "Int";
        case SPIRType::UInt:  return "UInt";
        case SPIRType::Boolean: return "Bool";
        case SPIRType::Struct: return "Struct";
        case SPIRType::Image: return "Texture";
        case SPIRType::SampledImage: return "CombinedTextureSampler";
        case SPIRType::Sampler: return "Sampler";
        default: return "Unknown";
    }
}

// ============================================================================
// 1. 编译阶段: 使用 shaderc 将 HLSL 编译为 SPIR-V
// ============================================================================
std::vector<uint32_t> compileHLSLToSPV(const std::string& sourceName, 
                                       const std::string& source, 
                                       shaderc_shader_kind kind,
                                       const std::string& entryPoint) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // 关键：告诉编译器这是 HLSL
    options.SetSourceLanguage(shaderc_source_language_hlsl);
    // 优化级别 (为了调试反射信息，可以设为 Zero，但通常 Performance 也不影响反射)
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    // 生成调试信息 (保留变量名)
    options.SetGenerateDebugInfo();

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
        source, kind, sourceName.c_str(), entryPoint.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << "Shader Compilation Error: " << module.GetErrorMessage() << std::endl;
        return {};
    }

    return {module.cbegin(), module.cend()};
}

// ============================================================================
// 2. 反射阶段: 使用 SPIRV-Cross 分析资源
// ============================================================================
struct ResourceInfo {
    uint32_t set;
    uint32_t binding;
    std::string name;
    std::string typeName;
    uint32_t arraySize; // 0 or 1 means scalar/single, >1 means array
};

void reflectShader(const std::vector<uint32_t>& spvData) {

    spirv_cross::Compiler compiler(spvData);

    // 获取所有着色器资源
    spirv_cross::ShaderResources resources = compiler.get_shader_resources();

    std::vector<ResourceInfo> foundResources;

    // 定义一个 lambda 来处理不同类型的资源列表
    auto processResources = [&](const spirv_cross::SmallVector<spirv_cross::Resource>& resourceList, const std::string& tag) {
        for (const auto& resource : resourceList) {
            ResourceInfo info;
            
            // 获取 ID 和 类型
            info.name = resource.name; // 这是 HLSL 里的变量名 (e.g., CameraBuffer, g_AlbedoMap)
            
            // 获取 Set 和 Binding (Decoration)
            info.set = compiler.get_decoration(resource.id, spv::DecorationDescriptorSet);
            info.binding = compiler.get_decoration(resource.id, spv::DecorationBinding);

            // 获取详细类型信息
            const auto& type = compiler.get_type(resource.type_id);
            info.typeName = getTypeName(type.basetype);
            
            // 处理数组
            if (!type.array.empty()) {
                info.arraySize = type.array[0]; // 只取第一维，多维数组很少用于 Descriptor
            } else {
                info.arraySize = 1;
            }

            // 修正：如果名字是空的（有时 Block 名字会被优化），尝试获取 Block 的名字
            if (info.name.empty()) {
                info.name = compiler.get_name(resource.id);
            }

            foundResources.push_back(info);
        }
    };

    // 我们关心的资源类型
    // Uniform Buffers (cbuffer)
    processResources(resources.uniform_buffers, "Uniform Buffer");
    // Separate Images (Texture2D)
    processResources(resources.separate_images, "Texture");
    // Separate Samplers (SamplerState)
    processResources(resources.separate_samplers, "Sampler");
    // Sampled Images (Combined, rare in HLSL unless specified)
    processResources(resources.sampled_images, "Combined Image Sampler");

    

    // ========================================================================
    // 3. 打印结果
    // ========================================================================
    std::cout << "\n===== Reflection Results =====\n";
    std::cout << std::left << std::setw(10) << "Set" 
              << std::setw(10) << "Binding" 
              << std::setw(25) << "Name" 
              << std::setw(15) << "Type" 
              << "Array Size" << std::endl;
    std::cout << std::string(75, '-') << std::endl;

    // 简单的排序，按 Set 然后 Binding
    std::sort(foundResources.begin(), foundResources.end(), [](const ResourceInfo& a, const ResourceInfo& b) {
        if (a.set != b.set) return a.set < b.set;
        return a.binding < b.binding;
    });

    for (const auto& res : foundResources) {
        std::cout << std::left << std::setw(10) << res.set 
                  << std::setw(10) << res.binding 
                  << std::setw(25) << res.name 
                  << std::setw(15) << res.typeName 
                  << res.arraySize << std::endl;
    }

    // ========================================================================
    // [新增] 2. Push Constants 反射
    // ========================================================================
    std::cout << "\n[Push Constants]\n";
    for (const auto& resource : resources.push_constant_buffers) {
        std::string name = resource.name;
        // 获取该 PushConstant 块的总大小
        const auto& type = compiler.get_type(resource.type_id);
        size_t size = compiler.get_declared_struct_size(type);
        
        std::cout << "  Name: " << name << " | Size: " << size << " bytes" << std::endl;
        
        // 打印内部成员 (例如 Model Matrix)
        for (uint32_t i = 0; i < type.member_types.size(); i++) {
            std::string memberName = compiler.get_member_name(type.self, i);
            size_t memberSize = compiler.get_declared_struct_member_size(type, i);
            size_t memberOffset = compiler.type_struct_member_offset(type, i);
            std::cout << "    - Member: " << memberName 
                      << " (Offset: " << memberOffset << ", Size: " << memberSize << ")" << std::endl;
        }
    }

    // ========================================================================
    // [新增] 3. UBO 结构体成员与大小 (解决对齐问题)
    // ========================================================================
    std::cout << "\n[Uniform Buffers Detail]\n";
    for (const auto& resource : resources.uniform_buffers) {
        std::string name = resource.name.empty() ? compiler.get_name(resource.id) : resource.name;
        const auto& type = compiler.get_type(resource.type_id);
        size_t totalSize = compiler.get_declared_struct_size(type);

        std::cout << "  Block: " << name << " | Total Size: " << totalSize << " bytes" << std::endl;
        
        // 遍历结构体成员
        for (uint32_t i = 0; i < type.member_types.size(); i++) {
            std::string memberName = compiler.get_member_name(type.self, i);
            size_t offset = compiler.type_struct_member_offset(type, i);
            size_t size = compiler.get_declared_struct_member_size(type, i);
            
            std::cout << "    - " << std::left << std::setw(20) << memberName 
                      << " Offset: " << std::setw(4) << offset 
                      << " Size: " << size << std::endl;
        }
    }
}

int main() {
    try {
        std::string shaderPath = "mujoco_ps.hlsl";
        std::cout << "Loading shader: " << shaderPath << "..." << std::endl;
        std::string hlslSource = readFile(shaderPath);

        std::cout << "Compiling HLSL to SPIR-V..." << std::endl;
         
        std::vector<uint32_t> spv = compileHLSLToSPV(shaderPath, hlslSource, shaderc_vertex_shader, "PSMain");

        if (spv.empty()) {
            return -1;
        }
        std::cout << "Compilation successful. SPIR-V size: " << spv.size() * 4 << " bytes." << std::endl;

        std::cout << "Reflecting resources..." << std::endl;
        reflectShader(spv);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}