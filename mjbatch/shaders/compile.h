// compile.h
#ifndef COMPILE_H
#define COMPILE_H

#include <string>
#include <vector>

struct CompileOptions {
    bool forceShaderModel67 = true;     // 默认强制使用 Shader Model 6.7
    int optimizationLevel = 0;          // 优化级别 0-3
    bool includeDebugInfo = false;      // 包含调试信息
    bool outputReflection = false;      // 输出反射信息
    std::vector<std::string> extraArgs; // 额外编译参数
};

bool CompileHLSLtoSPV(const std::string &input,
                      const std::string &target,
                      const std::string &entry,
                      const std::string &output,
                      const CompileOptions& options);

bool CompileHLSLtoSPV(const std::string &input,
                      const std::string &target,
                      const std::string &entry,
                      const std::string &output);

#endif // COMPILE_H