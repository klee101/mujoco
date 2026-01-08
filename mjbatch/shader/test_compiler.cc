#include "hlsl_compiler.h"
#include <iostream>
#include <iomanip>
#include <vector>

// 引入 SPIRV-Cross 用于验证编译结果是否正确 (是否包含反射信息)
#include <spirv_cross/spirv_cross.hpp>

using namespace mujoco::mjbatch::shader;

// 辅助函数：打印反射信息
void printReflection(const std::vector<uint32_t>& spv, const std::string& stageName) {
    if (spv.empty()) {
        std::cerr << "Error: " << stageName << " SPIR-V is empty!" << std::endl;
        return;
    }

    std::cout << "\n>>> Reflecting " << stageName << " (" << spv.size() * 4 << " bytes) <<<" << std::endl;

    spirv_cross::Compiler compiler(spv);
    auto resources = compiler.get_shader_resources();

    // 1. 打印 Descriptor Sets
    auto printResource = [&](const auto& list, const char* type) {
        for (const auto& res : list) {
            uint32_t set = compiler.get_decoration(res.id, spv::DecorationDescriptorSet);
            uint32_t binding = compiler.get_decoration(res.id, spv::DecorationBinding);
            std::cout << "  [Set " << set << ", Binding " << binding << "] " 
                      << std::left << std::setw(15) << type << " : " << res.name << std::endl;
        }
    };

    printResource(resources.uniform_buffers, "UniformBuffer");
    printResource(resources.separate_images, "Texture");
    printResource(resources.separate_samplers, "Sampler");

    // 2. 打印 Push Constants
    for (const auto& res : resources.push_constant_buffers) {
        std::cout << "  [PushConstant]  Size: " 
                  << compiler.get_declared_struct_size(compiler.get_type(res.type_id)) 
                  << " bytes : " << res.name << std::endl;
    }
}

int main() {
    try {
        std::cout << "===========================================" << std::endl;
        std::cout << "      HLSLCompiler Integration Test        " << std::endl;
        std::cout << "===========================================" << std::endl;

        std::filesystem::path shaderPath = "test_shader.hlsl";
        
        // --- 测试 1: 编译 Vertex Shader ---
        std::cout << "\n1. Compiling Vertex Shader (VSMain)..." << std::endl;
        auto vsSpv = HLSLCompiler::compileHlslFileCached(
            vk::ShaderStageFlagBits::eVertex, 
            shaderPath, 
            "VSMain"
        );
        std::cout << "   Success! Cached code size: " << vsSpv.size() << " words." << std::endl;
        
        // 验证反射信息 (VS应该包含 CameraBuffer 和 PushConstant)
        printReflection(vsSpv, "Vertex Shader");


        // --- 测试 2: 编译 Pixel Shader ---
        std::cout << "\n-------------------------------------------" << std::endl;
        std::cout << "\n2. Compiling Pixel Shader (PSMain)..." << std::endl;
        auto psSpv = HLSLCompiler::compileHlslFileCached(
            vk::ShaderStageFlagBits::eFragment, 
            shaderPath, 
            "PSMain"
        );
        std::cout << "   Success! Cached code size: " << psSpv.size() << " words." << std::endl;

        // 验证反射信息 (PS应该包含 Texture, Sampler 和 CameraBuffer)
        printReflection(psSpv, "Pixel Shader");


        // --- 测试 3: 缓存机制验证 ---
        std::cout << "\n-------------------------------------------" << std::endl;
        std::cout << "\n3. Testing Cache Mechanism..." << std::endl;
        
        // [FIX] 使用 const auto& 避免拷贝，直接引用缓存中的内存
        const auto& vsSpvRef1 = HLSLCompiler::compileHlslFileCached(
            vk::ShaderStageFlagBits::eVertex, 
            shaderPath, 
            "VSMain"
        );
        
        const auto& vsSpvRef2 = HLSLCompiler::compileHlslFileCached(
            vk::ShaderStageFlagBits::eVertex, 
            shaderPath, 
            "VSMain"
        );
        
        // 打印地址以供调试
        std::cout << "   Ptr1: " << vsSpvRef1.data() << std::endl;
        std::cout << "   Ptr2: " << vsSpvRef2.data() << std::endl;

        if (vsSpvRef1.data() == vsSpvRef2.data()) {
            std::cout << "   [PASS] Pointers match. Cache is working." << std::endl;
        } else {
            std::cerr << "   [FAIL] Pointers differ. Cache miss occurred." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL ERROR]: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}