#pragma once

#include <string>
#include <vector>
#include <memory>
#include <filesystem>
#include <unordered_map>
#include "base_parser.h"

namespace mujoco::mjbatch::shader {
    struct ShaderConfig;      // 用于存储生成的 DescriptorLayout 描述
    class BasePass;           // 所有 Pass 的基类
    class GBufferPass;        // G-Buffer 几何 Pass
    class ShadowPass;         // 阴影 Pass
    class DeferredPass;       // 延迟光照 Pass
    class PostProcessPass;    // 后处理 Pass (ToneMapping, etc.)
}

namespace mujoco::mjbatch::shader {

/**
 * @brief ShaderPack 负责管理一个特定渲染风格的所有 Shader 文件。
 * * 它的职责：
 * 1. 扫描指定目录。
 * 2. 识别并加载不同阶段的 Shader (GBuffer, Shadow, Deferred)。
 * 3. 使用反射 (Reflection) 分析资源需求。
 * 4. 生成统一的 Pipeline Layout 描述。
 */
class ShaderPack {
public:
    /**
     * @brief 构造函数
     * @param shaderDir Shader 文件所在的根目录 (例如 "assets/shaders/pbr_standard")
     */
    explicit ShaderPack(const std::filesystem::path& shaderDir);
    ~ShaderPack() = default;

    // 禁止拷贝，允许移动 (资源管理类的常见做法)
    ShaderPack(const ShaderPack&) = delete;
    ShaderPack& operator=(const ShaderPack&) = delete;
    ShaderPack(ShaderPack&&) = default;
    ShaderPack& operator=(ShaderPack&&) = default;

    // ========================================================================
    // 核心接口：获取不同阶段的 Pass
    // ========================================================================
    
    // 获取 G-Buffer Pass (可能有多个，处理不同材质或透明度)
    [[nodiscard]] const std::vector<std::shared_ptr<GBufferPass>>& getGBufferPasses() const { return m_gbufferPasses; }
    
    // 获取阴影 Pass (Directional, Point, Spot)
    [[nodiscard]] const std::vector<std::shared_ptr<ShadowPass>>& getShadowPasses() const { return m_shadowPasses; }
    
    // 获取延迟光照 Pass (通常只有一个，负责计算 PBR)
    [[nodiscard]] std::shared_ptr<DeferredPass> getDeferredPass() const { return m_deferredPass; }

    // 获取后处理 Pass 链
    [[nodiscard]] const std::vector<std::shared_ptr<PostProcessPass>>& getPostProcessPasses() const { return m_postPasses; }

    // ========================================================================
    // 资源布局接口
    // ========================================================================
    
    /** 
     * @brief 获取通过反射生成的全局资源布局配置。
     * NOTE: Renderer 会根据这个 Config 来创建 VkDescriptorSetLayout。
     */
    [[nodiscard]] std::shared_ptr<ShaderConfig> getShaderConfig() const { return m_shaderConfig; }

    // 检查是否包含特定功能（用于 Renderer 决定是否开启某些 RenderPass）
    [[nodiscard]] bool hasShadows() const { return !m_shadowPasses.empty(); }
    [[nodiscard]] bool hasDeferred() const { return m_deferredPass != nullptr; }

private:
    /**
     * @brief 核心初始化函数：扫描目录并编译/反射 Shader
     */
    void loadShaders(const std::filesystem::path& dir);

    /**
     * @brief 根据所有 Pass 的反射信息，合并生成全局统一的 Layout
     */
    void generateGlobalLayouts();

private:
    std::filesystem::path m_shaderDir;

    // Pass 集合
    std::vector<std::shared_ptr<GBufferPass>> m_gbufferPasses;
    std::vector<std::shared_ptr<ShadowPass>> m_shadowPasses;
    std::shared_ptr<DeferredPass> m_deferredPass;
    std::vector<std::shared_ptr<PostProcessPass>> m_postPasses;

    // 全局资源配置 (反射结果的汇总)
    std::shared_ptr<ShaderConfig> m_shaderConfig;
};

} // namespace mujoco::mjbatch::shader