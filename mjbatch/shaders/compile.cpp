#include "compile.h"
#include <iostream>
#include <vector>
#include <string>
#include <codecvt>
#include <locale>
#include <dxc/dxcapi.h>
#include <cassert>
#include <cstdlib>

static std::wstring toWString(const std::string &str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
}


static std::string ensureShaderModel67(const std::string& target) {

    if (target.find('_') != std::string::npos) {
        return target;
    }
    
    return target + "_6_7";
}


static std::wstring getStageTypeString(const std::string& entry, const std::string& target) {
    // 如果入口函数为空，认为是库着色器
    if (entry.empty()) {
        return L"lib_6_7";
    }
    
    // 从目标字符串提取着色器阶段前缀
    std::string prefix;
    size_t underscore_pos = target.find('_');
    if (underscore_pos != std::string::npos) {
        prefix = target.substr(0, underscore_pos);
    } else {
        prefix = target;
    }
    
    // 映射到对应的着色器阶段字符串
    if (prefix == "vs" || prefix == "vertex") {
        return L"vs_6_7";
    } else if (prefix == "ps" || prefix == "pixel" || prefix == "frag" || prefix == "fragment") {
        return L"ps_6_7";
    } else if (prefix == "cs" || prefix == "compute") {
        return L"cs_6_7";
    } else if (prefix == "ms" || prefix == "mesh") {
        return L"ms_6_7";
    } else if (prefix == "as" || prefix == "amplification") {
        return L"as_6_7";
    } else if (prefix == "lib" || prefix == "library") {
        return L"lib_6_7";
    } else {
        std::cerr << "Warning: Unknown shader target '" << target << "', defaulting to lib_6_7\n";
        return L"lib_6_7";
    }
}

bool CompileHLSLtoSPV(const std::string &input,
                      const std::string &target,
                      const std::string &entry,
                      const std::string &output,
                      const CompileOptions& options)
{
    IDxcUtils* pUtils = nullptr;
    IDxcCompiler3* pCompiler = nullptr;
    IDxcIncludeHandler* pIncludeHandler = nullptr;

    if (FAILED(DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&pUtils)))) {
        std::cerr << "Failed to create DXC Utils\n";
        return false;
    }

    if (FAILED(DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&pCompiler)))) {
        std::cerr << "Failed to create DXC Compiler\n";
        pUtils->Release();
        return false;
    }

    if (FAILED(pUtils->CreateDefaultIncludeHandler(&pIncludeHandler))) {
        std::cerr << "Failed to create include handler\n";
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    // Load shader source
    uint32_t codePage = CP_UTF8;
    IDxcBlobEncoding* pSource = nullptr;
    if (FAILED(pUtils->LoadFile(toWString(input).c_str(), &codePage, &pSource))) {
        std::cerr << "Failed to load shader: " << input << "\n";
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    DxcBuffer SourceBuffer;
    SourceBuffer.Ptr = pSource->GetBufferPointer();
    SourceBuffer.Size = pSource->GetBufferSize();
    SourceBuffer.Encoding = DXC_CP_ACP;

    // 自动处理着色器模型版本
    std::string finalTarget = options.forceShaderModel67 ? 
                              ensureShaderModel67(target) : target;
    
    std::wstring wTarget = toWString(finalTarget);
    std::wstring wEntry = toWString(entry);

    // 构建编译参数
    std::vector<LPCWSTR> args = {
        L"-spirv",
        L"-fvk-use-scalar-layout",
        L"-fspv-reflect",
        L"-fspv-target-env=vulkan1.2",
        L"-T", wTarget.c_str(),
        L"-E", wEntry.c_str()
    };

    if (options.optimizationLevel > 0) {
        switch (options.optimizationLevel) {
            case 1: args.push_back(L"-O1"); break;
            case 2: args.push_back(L"-O2"); break;
            case 3: args.push_back(L"-O3"); break;
            default: args.push_back(L"-Od"); break; 
        }
    }


    if (options.includeDebugInfo) {
        args.push_back(L"-Zi"); 
        args.push_back(L"-Qembed_debug"); 
    }

    for (const auto& extraArg : options.extraArgs) {
        args.push_back(toWString(extraArg).c_str());
    }

    IDxcResult* pResults = nullptr;
    HRESULT hr = pCompiler->Compile(
        &SourceBuffer,
        args.data(),
        static_cast<UINT>(args.size()),
        pIncludeHandler,
        IID_PPV_ARGS(&pResults));

    if (FAILED(hr)) {
        std::cerr << "Compilation failed.\n";
        pSource->Release();
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    // Check compilation status
    HRESULT hrStatus;
    pResults->GetStatus(&hrStatus);
    
    // Error messages
    IDxcBlobUtf8* pErrors = nullptr;
    pResults->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&pErrors), nullptr);
    if (pErrors && pErrors->GetStringLength() > 0) {
        std::cerr << "Shader compile errors:\n"
                  << pErrors->GetStringPointer() << std::endl;
    }

    if (FAILED(hrStatus)) {
        std::cerr << "Compilation failed with status: " << hrStatus << "\n";
        if (pErrors) pErrors->Release();
        pResults->Release();
        pSource->Release();
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    IDxcBlob* pShader = nullptr;
    hr = pResults->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&pShader), nullptr);
    if (FAILED(hr) || !pShader) {
        std::cerr << "Failed to retrieve compiled shader.\n";
        if (pErrors) pErrors->Release();
        pResults->Release();
        pSource->Release();
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    if (pShader->GetBufferSize() == 0) {
        std::cerr << "Compiled shader is empty!\n";
        pShader->Release();
        if (pErrors) pErrors->Release();
        pResults->Release();
        pSource->Release();
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    if (options.outputReflection) {
        IDxcBlob* pReflection = nullptr;
        if (SUCCEEDED(pResults->GetOutput(DXC_OUT_REFLECTION, IID_PPV_ARGS(&pReflection), nullptr))) {
            std::string reflectionFile = output + ".refl";
            FILE* reflFile = nullptr;
#ifdef _WIN32
            _wfopen_s(&reflFile, toWString(reflectionFile).c_str(), L"wb");
#else
            reflFile = fopen(reflectionFile.c_str(), "wb");
#endif
            if (reflFile) {
                fwrite(pReflection->GetBufferPointer(), 1, pReflection->GetBufferSize(), reflFile);
                fclose(reflFile);
                std::cout << "Reflection data written to: " << reflectionFile << std::endl;
            }
            pReflection->Release();
        }
    }

    // Manually write output file
    FILE* file = nullptr;
#ifdef _WIN32
    _wfopen_s(&file, toWString(output).c_str(), L"wb");
#else
    file = fopen(output.c_str(), "wb");
#endif

    if (!file) {
        std::cerr << "Failed to open output file: " << output << "\n";
        pShader->Release();
        if (pErrors) pErrors->Release();
        pResults->Release();
        pSource->Release();
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    size_t written = fwrite(pShader->GetBufferPointer(), 
                            1, 
                            pShader->GetBufferSize(), 
                            file);
    fclose(file);

    if (written != pShader->GetBufferSize()) {
        std::cerr << "Failed to write complete shader data\n";
        pShader->Release();
        if (pErrors) pErrors->Release();
        pResults->Release();
        pSource->Release();
        pIncludeHandler->Release();
        pCompiler->Release();
        pUtils->Release();
        return false;
    }

    std::cout << "Successfully compiled " << input << " -> " << output 
              << " (Target: " << finalTarget << ", Size: " << pShader->GetBufferSize() << " bytes)\n";

    // Release all
    pShader->Release();
    if (pErrors) pErrors->Release();
    pResults->Release();
    pSource->Release();
    pIncludeHandler->Release();
    pCompiler->Release();
    pUtils->Release();

    return true;
}


bool CompileHLSLtoSPV(const std::string &input,
                      const std::string &target,
                      const std::string &entry,
                      const std::string &output)
{
    CompileOptions options;
    options.forceShaderModel67 = true; // 默认启用 Shader Model 6.7
    return CompileHLSLtoSPV(input, target, entry, output, options);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::cerr << "Usage: hlsl_compile <input.hlsl> <target> <entry> <output.spv> [options]\n";
        std::cerr << "Examples:\n";
        std::cerr << "  hlsl_compile shader.hlsl vs VSMain shader.vert.spv\n";
        std::cerr << "  hlsl_compile shader.hlsl ps PSMain shader.frag.spv\n";
        std::cerr << "  hlsl_compile shader.hlsl cs CSMain shader.comp.spv\n";
        std::cerr << "\nTargets (automatically upgraded to _6_7):\n";
        std::cerr << "  vs, ps, cs, ms, as, lib\n";
        std::cerr << "  or explicitly specify: vs_6_0, ps_6_5, etc.\n";
        return 1;
    }

    std::string input = argv[1];
    std::string target = argv[2];  // e.g. vs, ps, cs, etc.
    std::string entry = argv[3];   // e.g. VSMain, PSMain
    std::string output = argv[4];

    CompileOptions options;
    options.forceShaderModel67 = true; // 默认强制使用 6.7
    
    for (int i = 5; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-O0") options.optimizationLevel = 0;
        else if (arg == "-O1") options.optimizationLevel = 1;
        else if (arg == "-O2") options.optimizationLevel = 2;
        else if (arg == "-O3") options.optimizationLevel = 3;
        else if (arg == "-g") options.includeDebugInfo = true;
        else if (arg == "-refl") options.outputReflection = true;
        else if (arg == "-no-sm67") options.forceShaderModel67 = false;
        else options.extraArgs.push_back(arg);
    }

    if (!CompileHLSLtoSPV(input, target, entry, output, options))
        return 1;
    return 0;
}