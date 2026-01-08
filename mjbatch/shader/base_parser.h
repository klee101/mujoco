#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <future>
#include <vulkan/vulkan.hpp>

// Forward declaration for SPIRV-Cross
namespace spirv_cross {
    class Compiler;
}

namespace mujoco::mjbatch::shader {

/**
 * @brief 逻辑绑定类型。
 * 这对应于你的 HLSL 中 `register(spaceN)` 的逻辑分组。
 * 例如: space0 -> Camera, space1 -> Object, space2 -> Scene
 */
enum class BindingType {
    Camera,   // set = 0
    Object,   // set = 1
    Scene,    // set = 2
    Material, // set = 3
    Textures, // Used for deferred lighting input textures
    Light,    // Shadow/Light buffers
    Unknown
};

/**
 * @brief 描述单个 Descriptor Binding (例如: layout(binding=0) uniform ...)
 */
struct DescriptorBinding {
    std::string name;
    vk::DescriptorType type;
    uint32_t bindingPoint;  // HLSL register(bX)
    uint32_t arraySize;     // 1 for normal, >1 for arrays
    vk::ShaderStageFlags stageFlags; // Vertex, Fragment, or Both

    // 如果是 Image/Sampler，可能需要额外的格式信息
    vk::Format imageFormat = vk::Format::eUndefined;
};

/**
 * @brief 描述整个 Descriptor Set (例如: layout(set=0) ...)
 */
struct DescriptorSetLayoutData {
    uint32_t setIndex;      // HLSL register(spaceX)
    BindingType type = BindingType::Unknown;
    std::map<uint32_t, DescriptorBinding> bindings; // Binding Point -> Binding Data

    // 合并另一个阶段的 Set (例如把 Vert 和 Frag 的 Set 0 合并)
    void merge(const DescriptorSetLayoutData& other);
};

/**
 * @brief 所有 Render Pass 的基类。
 * 负责：
 * 1. 加载和编译 HLSL Shader。
 * 2. 使用 SPIRV-Cross 反射资源布局。
 * 3. 创建 Vulkan Pipeline Layout。
 */
class BasePass {
public:
    virtual ~BasePass() = default;

    /**
     * @brief 加载 HLSL 文件并编译为 SPIR-V
     * @param vertFile .hlsl 顶点着色器路径
     * @param fragFile .hlsl 片元着色器路径 (Shadow Pass 可能为空)
     */
    void loadShaderFiles(const std::string& vertFile, const std::string& fragFile);

    // 异步加载版本 (推荐用于加快启动速度)
    std::future<void> loadShaderFilesAsync(const std::string& vertFile, const std::string& fragFile);

    // ========================================================================
    // 反射接口 (Reflection)
    // ========================================================================
    
    /**
     * @brief 获取该 Pass 所需的所有 Descriptor Sets 的描述。
     * 这是自动生成 PipelineLayout 的依据。
     */
    virtual std::vector<DescriptorSetLayoutData> reflectResources();

    /**
     * @brief 获取延迟渲染所需的输入纹理名称 (G-Buffer Input)。
     * 仅用于 Deferred Pass，分析 `Texture2D g_texture` 等变量。
     */
    virtual std::vector<std::string> getInputTextureNames() const { return {}; }

    // ========================================================================
    // Vulkan 资源创建接口 (Factory Methods)
    // ========================================================================

    /**
     * @brief 创建 Pipeline Layout
     * @param layouts 从 ShaderPack 获取的全局 descriptor layouts
     */
    vk::UniquePipelineLayout createPipelineLayout(vk::Device device, 
                                                  const std::vector<vk::DescriptorSetLayout>& layouts) const;

    /**
     * @brief [纯虚函数] 创建 Render Pass
     * 子类需根据自己是 GBuffer 还是 Shadow 来决定 Attachment 的配置。
     */
    virtual vk::UniqueRenderPass createRenderPass(vk::Device device, 
                                                  vk::Format colorFormat, 
                                                  vk::Format depthFormat) const = 0;

    /**
     * @brief [纯虚函数] 创建 Graphics Pipeline
     */
    virtual vk::UniquePipeline createPipeline(vk::Device device, 
                                              vk::PipelineLayout layout, 
                                              vk::RenderPass renderPass,
                                              vk::CullModeFlags cullMode, 
                                              vk::FrontFace frontFace) const = 0;

    // Getters
    const std::string& getName() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

protected:
    // 内部辅助函数：编译 HLSL -> SPIRV (实现将放在 .cpp 中，调用 shaderc/dxc)
    static std::vector<uint32_t> compileHLSL(const std::string& filepath, vk::ShaderStageFlagBits stage);
    
    // 内部辅助函数：反射单个 Shader Stage
    void reflectStage(const std::vector<uint32_t>& spv, 
                      vk::ShaderStageFlagBits stage, 
                      std::map<uint32_t, DescriptorSetLayoutData>& outSets);

protected:
    std::string m_name;
    
    // 编译后的 SPIR-V 二进制数据
    std::vector<uint32_t> m_vertSPV;
    std::vector<uint32_t> m_fragSPV;
    // 如果有 Geometry Shader，可以在此添加 m_geomSPV
};

} // namespace mujoco::mjbatch::shader