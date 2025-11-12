#pragma once

#include <vulkan/vulkan.h>

#define mjNAUX          10        // number of auxiliary buffers
#define mjMAXTEXTURE    1000      // maximum number of textures
#define mjMAXMATERIAL   1000      // maximum number of materials with textures
#define mjNTEXROLE     3         // number of texture roles per material
enum {
  mjrSPHERE = 0,
  mjrSPHERETOP,
  mjrSPHEREBOTTOM,
  mjrCYLINDER,
  mjrCYLINDEROPEN,
  mjrHAZE,
  mjrBOX,
  mjrCONE,
  mjrNUM
};

// Vulkan Context (correspond to mjrContext)
typedef struct {
  //---------------------- Vulkan core ----------------------
  VkInstance instance;
  VkPhysicalDevice physicalDevice;
  VkDevice device;
  VkQueue graphicsQueue;
  VkQueue presentQueue;
  
  uint32_t graphicsQueueFamily;
  uint32_t presentQueueFamily;
  
  VkCommandPool commandPool;
  VkDescriptorPool descriptorPool;
  
  //---------------------- SwapChain  ----------------------
  VkSwapchainKHR swapchain;
  VkFormat swapchainFormat;
  VkExtent2D swapchainExtent;
  VkImage* swapchainImages;
  VkImageView* swapchainViews;
  VkFramebuffer* swapchainFramebuffers;
  uint32_t swapchainImageCount;
  
  //---------------------- offscreen rendering (correspond to offscreen FBO) ----------------------
  struct {
    VkImage colorImage;
    VkDeviceMemory colorMemory;
    VkImageView colorView;
    
    VkImage depthImage;
    VkDeviceMemory depthMemory;
    VkImageView depthView;
    
    VkFramebuffer framebuffer;
    VkRenderPass renderPass;
    
    int width;
    int height;
    int samples;  // MSAA
  } offscreen;
  
  //---------------------- ShadowMap (correspond to shadow FBO) ----------------------
  struct {
    VkImage depthImage;
    VkDeviceMemory depthMemory;
    VkImageView depthView;
    VkSampler sampler;
    
    VkFramebuffer framebuffer;
    VkRenderPass renderPass;
    
    int size;
  } shadow;
  
  //---------------------- auxMap (correspond to aux buffers) ----------------------
  struct {
    VkImage colorImage;
    VkDeviceMemory colorMemory;
    VkImageView colorView;
    VkFramebuffer framebuffer;
    int width;
    int height;
    int samples;
  } aux[mjNAUX];
  
  //---------------------- geom cache (correspond to Display Lists) ----------------------
  // planes
  struct {
    VkBuffer* vertexBuffers;      // 对应 basePlane 显示列表
    VkBuffer* indexBuffers;
    VkDeviceMemory* vertexMemories;
    VkDeviceMemory* indexMemories;
    uint32_t* indexCounts;
    int count;                     // 对应 rangePlane
  } planes;
  
  // meshes
  struct {
    VkBuffer* vertexBuffers;      // 对应 baseMesh 显示列表
    VkBuffer* indexBuffers;
    VkDeviceMemory* vertexMemories;
    VkDeviceMemory* indexMemories;
    uint32_t* indexCounts;
    int count;                     // nmesh * 2 (原始+凸包)
  } meshes;
  
  // hfields
  struct {
    VkBuffer* vertexBuffers;      // 对应 baseHField 显示列表
    VkBuffer* indexBuffers;
    VkDeviceMemory* vertexMemories;
    VkDeviceMemory* indexMemories;
    uint32_t* indexCounts;
    int count;
  } hfields;
  
  // built-in geoms(sphere, box, cylinder, capsule, ellipsoid, cone, plane, torus)
  struct {
    VkBuffer vertexBuffers[mjrNUM];   // correspond to baseBuiltin
    VkBuffer indexBuffers[mjrNUM];
    VkDeviceMemory vertexMemories[mjrNUM];
    VkDeviceMemory indexMemories[mjrNUM];
    uint32_t indexCounts[mjrNUM];
  } builtins;
  
  // skin 
  struct {
    VkBuffer* vertexBuffers;      //  skinvertVBO
    VkBuffer* normalBuffers;      //  skinnormalVBO
    VkBuffer* texcoordBuffers;    //  skintexcoordVBO
    VkBuffer* indexBuffers;       //  skinfaceVBO
    VkDeviceMemory* vertexMemories;
    VkDeviceMemory* normalMemories;
    VkDeviceMemory* texcoordMemories;
    VkDeviceMemory* indexMemories;
    int count;
  } skins;
  
  //---------------------- texture resourses ----------------------
  struct {
    VkImage* images;
    VkDeviceMemory* memories;
    VkImageView* views;
    VkSampler* samplers;
    int* types;                    // 2D or Cubemap
    int count;
  } textures;
  
  //---------------------- materials (Descriptor Sets) ----------------------
  struct {
    VkDescriptorSetLayout layout;
    VkDescriptorSet sets[mjMAXMATERIAL];
    int texids[mjMAXMATERIAL * mjNTEXROLE];
    float texrepeat[mjMAXMATERIAL * 2];
    int texuniform[mjMAXMATERIAL];
  } materials;
  
  //---------------------- renderPipeline ----------------------
  VkPipelineLayout pipelineLayout;
  VkPipeline pipelines[16];         // different pipelines
  VkRenderPass mainRenderPass;
  
  //---------------------- 字体渲染 (需要重新设计) ----------------------
  struct {
    // Vulkan 不支持位图字体，需要使用纹理图集 + 实例化渲染
    VkImage atlasImage;
    VkDeviceMemory atlasMemory;
    VkImageView atlasView;
    VkSampler atlasSampler;
    
    VkBuffer glyphBuffer;          // 字形元数据
    VkDeviceMemory glyphMemory;
    
    int charWidth[128];
    int charHeight;
    int fontScale;
  } font;
  
  //---------------------- synchronize ----------------------
  VkSemaphore* imageAvailableSemaphores;
  VkSemaphore* renderFinishedSemaphores;
  VkFence* inFlightFences;
  uint32_t currentFrame;
  uint32_t maxFramesInFlight;
  
  //---------------------- commandBuffer ----------------------
  VkCommandBuffer* commandBuffers;
  
  //---------------------- 能力标志 ----------------------
  int glInitialized;              // -> vulkanInitialized
  int windowAvailable;
  int windowSamples;
  int windowStereo;
  int windowDoublebuffer;         // Vulkan always double buffered
  
  //---------------------- renderParams ----------------------
  float fogStart;
  float fogEnd;
  float fogRGBA[4];
  float lineWidth;
  float shadowClip;
  float shadowScale;
  
  int currentBuffer;
  int readPixelFormat;            // -> VkFormat
  int readDepthMap;
  
} mjrContextVk;


//---------------------- 顶点格式定义 ----------------------
// 标准顶点格式 (位置 + 法线 + 纹理坐标)
typedef struct {
  float position[3];
  float normal[3];
  float texcoord[2];
} VertexPNT;

// 获取顶点绑定描述
static inline VkVertexInputBindingDescription getVertexBindingDescription() {
  VkVertexInputBindingDescription desc = {
    .binding = 0,
    .stride = sizeof(VertexPNT),
    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX
  };
  return desc;
}

// 获取顶点属性描述
static inline void getVertexAttributeDescriptions(
    VkVertexInputAttributeDescription attrs[3]) {
  // 位置
  attrs[0].binding = 0;
  attrs[0].location = 0;
  attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrs[0].offset = offsetof(VertexPNT, position);
  
  // 法线
  attrs[1].binding = 0;
  attrs[1].location = 1;
  attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
  attrs[1].offset = offsetof(VertexPNT, normal);
  
  // 纹理坐标
  attrs[2].binding = 0;
  attrs[2].location = 2;
  attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
  attrs[2].offset = offsetof(VertexPNT, texcoord);
}