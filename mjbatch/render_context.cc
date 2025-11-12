// #include "render_context.h"
// #include <vector>
// #include <memory>
// #include <string>
// #include <functional>
// #include <cstdint>
// #include <mujoco/mujoco.h>
// #include <vulkan/vulkan.h>
// #include <cstdlib>
// #include <cstring>
// #include <cmath>

// #ifndef M_PI
// #define M_PI 3.14159265358979323846
// #endif

// // 前向声明
// static bool isExtensionAvailable(const char* name, const std::vector<VkExtensionProperties>& exts);
// static bool findGraphicsQueueFamily(VkPhysicalDevice phys, uint32_t* outIndex);
// static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties);

// //=============================================================================
// // Vulkan 上下文创建主流程
// //=============================================================================

// VkResult mjr_makeContextVk(const mjModel* m, mjrContextVk* con, int fontscale) {
//   VkResult result;
  
//   // 1. 初始化 Vulkan 实例
//   result = createVulkanInstance(con);
//   if (result != VK_SUCCESS) return result;
  
//   // 2. 选择物理设备
//   result = selectPhysicalDevice(con);
//   if (result != VK_SUCCESS) return result;
  
//   // 3. 创建逻辑设备和队列
//   result = createLogicalDevice(con);
//   if (result != VK_SUCCESS) return result;
  
//   // 4. 创建命令池
//   result = createCommandPool(con);
//   if (result != VK_SUCCESS) return result;
  
//   // 5. 创建描述符池
//   result = createDescriptorPool(con);
//   if (result != VK_SUCCESS) return result;
  
//   if (m == nullptr) {
//     // 仅创建离屏渲染和字体
//     result = createOffscreenResources(con, 800, 600);
//     if (result != VK_SUCCESS) return result;
    
//     result = createFontAtlas(con, fontscale);
//     return result;
//   }
  
//   // 6. 创建渲染目标
//   result = createOffscreenResources(con, m->vis.global.offwidth, m->vis.global.offheight);
//   if (result != VK_SUCCESS) return result;
  
//   result = createShadowResources(con, m->vis.quality.shadowsize);
//   if (result != VK_SUCCESS) return result;
  
//   // 7. 上传纹理
//   result = uploadTextures(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   // 8. 创建材质描述符集
//   result = createMaterialDescriptors(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   // 9. 上传几何体
//   result = uploadGeometry(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   // 10. 创建渲染管线
//   result = createRenderPipelines(con);
//   if (result != VK_SUCCESS) return result;
  
//   // 11. 创建同步对象
//   result = createSyncObjects(con);
//   if (result != VK_SUCCESS) return result;
  
//   return VK_SUCCESS;
// }

// //=============================================================================
// // 几何体上传流程
// //=============================================================================

// VkResult uploadGeometry(const mjModel* m, mjrContextVk* con) {
//   VkResult result;
  
//   result = uploadPlanes(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   result = uploadMeshes(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   result = uploadHeightFields(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   result = createBuiltinGeometry(m, con);
//   if (result != VK_SUCCESS) return result;
  
//   result = uploadSkins(m, con);
//   return result;
// }

// VkResult uploadMeshes(const mjModel* m, mjrContextVk* con) {
//   con->meshes.count = 2 * m->nmesh;
  
//   con->meshes.vertexBuffers = static_cast<VkBuffer*>(std::malloc(con->meshes.count * sizeof(VkBuffer)));
//   con->meshes.indexBuffers = static_cast<VkBuffer*>(std::malloc(con->meshes.count * sizeof(VkBuffer)));
//   con->meshes.vertexMemories = static_cast<VkDeviceMemory*>(std::malloc(con->meshes.count * sizeof(VkDeviceMemory)));
//   con->meshes.indexMemories = static_cast<VkDeviceMemory*>(std::malloc(con->meshes.count * sizeof(VkDeviceMemory)));
//   con->meshes.indexCounts = static_cast<uint32_t*>(std::malloc(con->meshes.count * sizeof(uint32_t)));
  
//   for (int i = 0; i < m->nmesh; i++) {
//     VertexPNT* vertices = nullptr;
//     uint32_t* indices = nullptr;
//     uint32_t vertexCount, indexCount;
    
//     convertMeshToVertexBuffer(m, i, &vertices, &vertexCount, &indices, &indexCount);
    
//     createAndUploadBuffer(con, vertices, static_cast<VkDeviceSize>(vertexCount * sizeof(VertexPNT)),
//                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
//                          &con->meshes.vertexBuffers[2*i],
//                          &con->meshes.vertexMemories[2*i]);
    
//     createAndUploadBuffer(con, indices, static_cast<VkDeviceSize>(indexCount * sizeof(uint32_t)),
//                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
//                          &con->meshes.indexBuffers[2*i],
//                          &con->meshes.indexMemories[2*i]);
    
//     con->meshes.indexCounts[2*i] = indexCount;
    
//     std::free(vertices);
//     std::free(indices);
    
//     // 处理凸包
//     if (m->mesh_graphadr[i] >= 0) {
//       convertConvexHullToVertexBuffer(m, i, &vertices, &vertexCount, &indices, &indexCount);
      
//       createAndUploadBuffer(con, vertices, static_cast<VkDeviceSize>(vertexCount * sizeof(VertexPNT)),
//                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
//                            &con->meshes.vertexBuffers[2*i+1],
//                            &con->meshes.vertexMemories[2*i+1]);
      
//       createAndUploadBuffer(con, indices, static_cast<VkDeviceSize>(indexCount * sizeof(uint32_t)),
//                            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
//                            &con->meshes.indexBuffers[2*i+1],
//                            &con->meshes.indexMemories[2*i+1]);
      
//       con->meshes.indexCounts[2*i+1] = indexCount;
      
//       std::free(vertices);
//       std::free(indices);
//     }
//   }
  
//   return VK_SUCCESS;
// }

// void convertMeshToVertexBuffer(const mjModel* m, int meshid,
//                               VertexPNT** vertices, uint32_t* vertexCount,
//                               uint32_t** indices, uint32_t* indexCount) {
//   int vertadr = m->mesh_vertadr[meshid];
//   int normaladr = m->mesh_normaladr[meshid];
//   int texcoordadr = m->mesh_texcoordadr[meshid];
//   int facenum = m->mesh_facenum[meshid];
  
//   *vertexCount = facenum * 3;
//   *vertices = static_cast<VertexPNT*>(std::malloc((*vertexCount) * sizeof(VertexPNT)));
//   *indexCount = facenum * 3;
//   *indices = static_cast<uint32_t*>(std::malloc((*indexCount) * sizeof(uint32_t)));
  
//   for (int face = 0; face < facenum; face++) {
//     int faceadr = m->mesh_faceadr[meshid] + face;
    
//     for (int v = 0; v < 3; v++) {
//       int idx = face * 3 + v;
//       VertexPNT* vert = &(*vertices)[idx];
      
//       const float* v_src = m->mesh_vert + 3*(m->mesh_face[3*faceadr+v] + vertadr);
//       std::memcpy(vert->position, v_src, 3 * sizeof(float));
      
//       const float* n_src = m->mesh_normal + 3*(m->mesh_facenormal[3*faceadr+v] + normaladr);
//       std::memcpy(vert->normal, n_src, 3 * sizeof(float));
      
//       if (texcoordadr >= 0) {
//         const float* t_src = m->mesh_texcoord + 2*(m->mesh_facetexcoord[3*faceadr+v] + texcoordadr);
//         std::memcpy(vert->texcoord, t_src, 2 * sizeof(float));
//       } else {
//         vert->texcoord[0] = 0.0f;
//         vert->texcoord[1] = 0.0f;
//       }
      
//       (*indices)[idx] = static_cast<uint32_t>(idx);
//     }
//   }
// }

// //=============================================================================
// // 内置几何体生成
// //=============================================================================

// VkResult createBuiltinGeometry(const mjModel* m, mjrContextVk* con) {
//   int numslices = m->vis.quality.numslices;
//   int numstacks = m->vis.quality.numstacks;
  
//   generateSphere(numslices, numstacks, 
//                 &con->builtins.vertexBuffers[mjrSPHERE],
//                 &con->builtins.indexBuffers[mjrSPHERE],
//                 &con->builtins.vertexMemories[mjrSPHERE],
//                 &con->builtins.indexMemories[mjrSPHERE],
//                 &con->builtins.indexCounts[mjrSPHERE],
//                 con);
  
//   generateCylinder(numslices, numstacks,
//                   &con->builtins.vertexBuffers[mjrCYLINDER],
//                   &con->builtins.indexBuffers[mjrCYLINDER],
//                   &con->builtins.vertexMemories[mjrCYLINDER],
//                   &con->builtins.indexMemories[mjrCYLINDER],
//                   &con->builtins.indexCounts[mjrCYLINDER],
//                   con);
  
//   return VK_SUCCESS;
// }

// void generateSphere(int slices, int stacks,
//                    VkBuffer* vertexBuffer, VkBuffer* indexBuffer,
//                    VkDeviceMemory* vertexMemory, VkDeviceMemory* indexMemory,
//                    uint32_t* indexCount, mjrContextVk* con) {
//   int vertexCount = (stacks + 1) * (slices + 1);
//   VertexPNT* vertices = static_cast<VertexPNT*>(std::malloc(vertexCount * sizeof(VertexPNT)));
  
//   int idx = 0;
//   for (int i = 0; i <= stacks; i++) {
//     float phi = static_cast<float>(M_PI) * i / static_cast<float>(stacks);
//     for (int j = 0; j <= slices; j++) {
//       float theta = 2.0f * static_cast<float>(M_PI) * j / static_cast<float>(slices);
      
//       vertices[idx].position[0] = std::sin(phi) * std::cos(theta);
//       vertices[idx].position[1] = std::sin(phi) * std::sin(theta);
//       vertices[idx].position[2] = std::cos(phi);
      
//       vertices[idx].normal[0] = vertices[idx].position[0];
//       vertices[idx].normal[1] = vertices[idx].position[1];
//       vertices[idx].normal[2] = vertices[idx].position[2];
      
//       vertices[idx].texcoord[0] = static_cast<float>(j) / static_cast<float>(slices);
//       vertices[idx].texcoord[1] = static_cast<float>(i) / static_cast<float>(stacks);
      
//       idx++;
//     }
//   }
  
//   *indexCount = static_cast<uint32_t>(stacks * slices * 6);
//   uint32_t* indices = static_cast<uint32_t*>(std::malloc((*indexCount) * sizeof(uint32_t)));
  
//   idx = 0;
//   for (int i = 0; i < stacks; i++) {
//     for (int j = 0; j < slices; j++) {
//       int p0 = i * (slices + 1) + j;
//       int p1 = p0 + slices + 1;
      
//       indices[idx++] = static_cast<uint32_t>(p0);
//       indices[idx++] = static_cast<uint32_t>(p1);
//       indices[idx++] = static_cast<uint32_t>(p0 + 1);
      
//       indices[idx++] = static_cast<uint32_t>(p0 + 1);
//       indices[idx++] = static_cast<uint32_t>(p1);
//       indices[idx++] = static_cast<uint32_t>(p1 + 1);
//     }
//   }
  
//   createAndUploadBuffer(con, vertices, static_cast<VkDeviceSize>(vertexCount * sizeof(VertexPNT)),
//                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexMemory);
  
//   createAndUploadBuffer(con, indices, static_cast<VkDeviceSize>(*indexCount * sizeof(uint32_t)),
//                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexMemory);
  
//   std::free(vertices);
//   std::free(indices);
// }

// void generateCylinder(int slices, int stacks,
//                      VkBuffer* vertexBuffer, VkBuffer* indexBuffer,
//                      VkDeviceMemory* vertexMemory, VkDeviceMemory* indexMemory,
//                      uint32_t* indexCount, mjrContextVk* con) {
//   if (slices < 3) slices = 3;
//   if (stacks < 1) stacks = 1;

//   const int ringVerts = slices + 1;
//   const int vertexCount = (stacks + 1) * ringVerts;
//   VertexPNT* vertices = static_cast<VertexPNT*>(std::malloc(sizeof(VertexPNT) * vertexCount));

//   int idx = 0;
//   for (int i = 0; i <= stacks; ++i) {
//     float v = static_cast<float>(i) / static_cast<float>(stacks);
//     float y = -0.5f + v;
//     for (int j = 0; j <= slices; ++j) {
//       float u = static_cast<float>(j) / static_cast<float>(slices);
//       float theta = 2.0f * static_cast<float>(M_PI) * u;
//       float cx = std::cos(theta);
//       float cz = std::sin(theta);

//       vertices[idx].position[0] = cx;
//       vertices[idx].position[1] = y;
//       vertices[idx].position[2] = cz;

//       vertices[idx].normal[0] = cx;
//       vertices[idx].normal[1] = 0.0f;
//       vertices[idx].normal[2] = cz;

//       vertices[idx].texcoord[0] = u;
//       vertices[idx].texcoord[1] = v;

//       ++idx;
//     }
//   }

//   *indexCount = static_cast<uint32_t>(stacks * slices * 6);
//   uint32_t* indices = static_cast<uint32_t*>(std::malloc(sizeof(uint32_t) * (*indexCount)));

//   idx = 0;
//   for (int i = 0; i < stacks; ++i) {
//     for (int j = 0; j < slices; ++j) {
//       uint32_t p0 = static_cast<uint32_t>(i * ringVerts + j);
//       uint32_t p1 = static_cast<uint32_t>((i + 1) * ringVerts + j);
      
//       indices[idx++] = p0;
//       indices[idx++] = p1;
//       indices[idx++] = p0 + 1;
      
//       indices[idx++] = p0 + 1;
//       indices[idx++] = p1;
//       indices[idx++] = p1 + 1;
//     }
//   }

//   createAndUploadBuffer(con, vertices, static_cast<VkDeviceSize>(sizeof(VertexPNT) * vertexCount),
//                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertexBuffer, vertexMemory);
//   createAndUploadBuffer(con, indices, static_cast<VkDeviceSize>(sizeof(uint32_t) * (*indexCount)),
//                        VK_BUFFER_USAGE_INDEX_BUFFER_BIT, indexBuffer, indexMemory);

//   std::free(vertices);
//   std::free(indices);
// }

// //=============================================================================
// // Vulkan 辅助函数
// //=============================================================================

// static VkResult createBuffer(mjrContextVk* ctx, VkDeviceSize size,
//                             VkBufferUsageFlags usage, VkMemoryPropertyFlags properties,
//                             VkBuffer* buffer, VkDeviceMemory* memory) {
//   VkBufferCreateInfo bufferInfo = {
//     .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
//     .size = size,
//     .usage = usage,
//     .sharingMode = VK_SHARING_MODE_EXCLUSIVE
//   };
  
//   if (vkCreateBuffer(ctx->device, &bufferInfo, NULL, buffer) != VK_SUCCESS) {
//     return VK_ERROR_INITIALIZATION_FAILED;
//   }
  
//   VkMemoryRequirements memRequirements;
//   vkGetBufferMemoryRequirements(ctx->device, *buffer, &memRequirements);
  
//   VkMemoryAllocateInfo allocInfo = {
//     .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
//     .allocationSize = memRequirements.size,
//     .memoryTypeIndex = findMemoryType(ctx->physicalDevice, memRequirements.memoryTypeBits, properties)
//   };
  
//   if (vkAllocateMemory(ctx->device, &allocInfo, NULL, memory) != VK_SUCCESS) {
//     vkDestroyBuffer(ctx->device, *buffer, NULL);
//     return VK_ERROR_OUT_OF_DEVICE_MEMORY;
//   }
  
//   vkBindBufferMemory(ctx->device, *buffer, *memory, 0);
//   return VK_SUCCESS;
// }

// static VkCommandBuffer beginSingleTimeCommands(mjrContextVk* ctx) {
//   VkCommandBufferAllocateInfo allocInfo = {
//     .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
//     .commandPool = ctx->commandPool,
//     .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
//     .commandBufferCount = 1
//   };
  
//   VkCommandBuffer commandBuffer;
//   vkAllocateCommandBuffers(ctx->device, &allocInfo, &commandBuffer);
  
//   VkCommandBufferBeginInfo beginInfo = {
//     .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
//     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
//   };
  
//   vkBeginCommandBuffer(commandBuffer, &beginInfo);
//   return commandBuffer;
// }

// static void endSingleTimeCommands(mjrContextVk* ctx, VkCommandBuffer commandBuffer) {
//   vkEndCommandBuffer(commandBuffer);
  
//   VkSubmitInfo submitInfo = {
//     .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
//     .commandBufferCount = 1,
//     .pCommandBuffers = &commandBuffer
//   };
  
//   vkQueueSubmit(ctx->graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
//   vkQueueWaitIdle(ctx->graphicsQueue);
//   vkFreeCommandBuffers(ctx->device, ctx->commandPool, 1, &commandBuffer);
// }

// static void copyBuffer(mjrContextVk* ctx, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {
//   VkCommandBuffer commandBuffer = beginSingleTimeCommands(ctx);
  
//   VkBufferCopy copyRegion = {
//     .srcOffset = 0,
//     .dstOffset = 0,
//     .size = size
//   };
  
//   vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
//   endSingleTimeCommands(ctx, commandBuffer);
// }

// VkResult createAndUploadBuffer(mjrContextVk* con, const void* data, 
//                               VkDeviceSize size, VkBufferUsageFlags usage,
//                               VkBuffer* buffer, VkDeviceMemory* memory) {
//   // 创建暂存缓冲
//   VkBuffer stagingBuffer;
//   VkDeviceMemory stagingMemory;
  
//   VkResult res = createBuffer(con, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
//                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
//                              &stagingBuffer, &stagingMemory);
//   if (res != VK_SUCCESS) return res;
  
//   // 映射并拷贝数据
//   void* mapped;
//   res = vkMapMemory(con->device, stagingMemory, 0, size, 0, &mapped);
//   if (res != VK_SUCCESS) {
//     vkDestroyBuffer(con->device, stagingBuffer, NULL);
//     vkFreeMemory(con->device, stagingMemory, NULL);
//     return res;
//   }
//   memcpy(mapped, data, (size_t)size);
//   vkUnmapMemory(con->device, stagingMemory);
  
//   // 创建设备本地缓冲
//   res = createBuffer(con, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
//                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, buffer, memory);
//   if (res != VK_SUCCESS) {
//     vkDestroyBuffer(con->device, stagingBuffer, NULL);
//     vkFreeMemory(con->device, stagingMemory, NULL);
//     return res;
//   }
  
//   // 拷贝数据
//   copyBuffer(con, stagingBuffer, *buffer, size);
  
//   // 清理暂存缓冲
//   vkDestroyBuffer(con->device, stagingBuffer, NULL);
//   vkFreeMemory(con->device, stagingMemory, NULL);
  
//   return VK_SUCCESS;
// }

// //=============================================================================
// // 核心 Vulkan 初始化函数
// //=============================================================================

// VkResult createVulkanInstance(mjrContextVk* con) {
//   uint32_t extCount = 0;
//   VkResult res = vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
//   if (res != VK_SUCCESS) return res;
  
//   std::vector<VkExtensionProperties> exts(extCount);
//   res = vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());
//   if (res != VK_SUCCESS) return res;

//   std::vector<const char*> enabledExts;
//   VkInstanceCreateFlags flags = 0;

//   if (isExtensionAvailable(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME, exts)) {
//     enabledExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
//     flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
//   }

//   VkApplicationInfo app = {
//     .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
//     .pApplicationName = "mjbatch",
//     .pEngineName = "mjbatch", 
//     .apiVersion = VK_API_VERSION_1_2
//   };

//   VkInstanceCreateInfo ci = {
//     .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
//     .flags = flags,
//     .pApplicationInfo = &app,
//     .enabledExtensionCount = (uint32_t)enabledExts.size(),
//     .ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data()
//   };

//   return vkCreateInstance(&ci, nullptr, &con->instance);
// }

// VkResult selectPhysicalDevice(mjrContextVk* con) {
//   uint32_t count = 0;
//   VkResult res = vkEnumeratePhysicalDevices(con->instance, &count, nullptr);
//   if (res != VK_SUCCESS) return res;
//   if (count == 0) return VK_ERROR_INITIALIZATION_FAILED;

//   std::vector<VkPhysicalDevice> phys(count);
//   res = vkEnumeratePhysicalDevices(con->instance, &count, phys.data());
//   if (res != VK_SUCCESS) return res;

//   for (auto p : phys) {
//     uint32_t graphicsIndex = 0;
//     if (findGraphicsQueueFamily(p, &graphicsIndex)) {
//       con->physicalDevice = p;
//       return VK_SUCCESS;
//     }
//   }
//   return VK_ERROR_INITIALIZATION_FAILED;
// }

// VkResult createLogicalDevice(mjrContextVk* ctx) {
//   uint32_t graphicsIndex = 0;
//   if (!findGraphicsQueueFamily(ctx->physicalDevice, &graphicsIndex)) {
//     return VK_ERROR_INITIALIZATION_FAILED;
//   }

//   float queuePriority = 1.0f;
//   VkDeviceQueueCreateInfo queueInfo = {
//     .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
//     .queueFamilyIndex = graphicsIndex,
//     .queueCount = 1,
//     .pQueuePriorities = &queuePriority
//   };

//   std::vector<const char*> enabledExts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

//   VkPhysicalDeviceFeatures deviceFeatures = { .samplerAnisotropy = VK_TRUE };

//   VkDeviceCreateInfo createInfo = {
//     .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
//     .queueCreateInfoCount = 1,
//     .pQueueCreateInfos = &queueInfo,
//     .enabledExtensionCount = (uint32_t)enabledExts.size(),
//     .ppEnabledExtensionNames = enabledExts.data(),
//     .pEnabledFeatures = &deviceFeatures
//   };

//   VkResult result = vkCreateDevice(ctx->physicalDevice, &createInfo, nullptr, &ctx->device);
//   if (result != VK_SUCCESS) return result;

//   vkGetDeviceQueue(ctx->device, graphicsIndex, 0, &ctx->graphicsQueue);
//   return VK_SUCCESS;
// }

// VkResult createCommandPool(mjrContextVk* con) {
//   uint32_t graphicsIndex = 0;
//   if (!findGraphicsQueueFamily(con->physicalDevice, &graphicsIndex)) {
//     return VK_ERROR_INITIALIZATION_FAILED;
//   }

//   VkCommandPoolCreateInfo ci = {
//     .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
//     .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
//     .queueFamilyIndex = graphicsIndex
//   };

//   return vkCreateCommandPool(con->device, &ci, nullptr, &con->commandPool);
// }

// VkResult createDescriptorPool(mjrContextVk* con) {
//   VkDescriptorPoolSize sizes[] = {
//     { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 },
//     { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64 },
//     { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
//   };

//   VkDescriptorPoolCreateInfo ci = {
//     .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
//     .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
//     .maxSets = 128,
//     .poolSizeCount = (uint32_t)(sizeof(sizes) / sizeof(sizes[0])),
//     .pPoolSizes = sizes
//   };

//   return vkCreateDescriptorPool(con->device, &ci, nullptr, &con->descriptorPool);
// }

// //=============================================================================
// // 工具函数
// //=============================================================================

// static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, 
//                               uint32_t typeFilter, 
//                               VkMemoryPropertyFlags properties) {
//   VkPhysicalDeviceMemoryProperties memProperties;
//   vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
  
//   for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
//     if ((typeFilter & (1 << i)) && 
//         (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
//       return i;
//     }
//   }
  
//   fprintf(stderr, "Failed to find suitable memory type!\n");
//   return 0;
// }

// static bool isExtensionAvailable(const char* name, const std::vector<VkExtensionProperties>& exts) {
//   for (const auto& e : exts) {
//     if (strcmp(name, e.extensionName) == 0) return true;
//   }
//   return false;
// }

// static bool findGraphicsQueueFamily(VkPhysicalDevice phys, uint32_t* outIndex) {
//   uint32_t count = 0;
//   vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, nullptr);
//   if (count == 0) return false;
  
//   std::vector<VkQueueFamilyProperties> props(count);
//   vkGetPhysicalDeviceQueueFamilyProperties(phys, &count, props.data());
  
//   for (uint32_t i = 0; i < count; ++i) {
//     if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
//       *outIndex = i;
//       return true;
//     }
//   }
//   return false;
// }

// //=============================================================================
// // TODO: 待实现函数
// //=============================================================================

// VkResult uploadPlanes(const mjModel* m, mjrContextVk* con) {
//   // TODO: 生成平面网格数据并上传
//   (void)m; (void)con;
//   return VK_SUCCESS;
// }

// VkResult uploadHeightFields(const mjModel* m, mjrContextVk* con) {
//   // TODO: 生成高度场网格并上传
//   (void)m; (void)con;
//   return VK_SUCCESS;
// }

// VkResult uploadSkins(const mjModel* m, mjrContextVk* con) {
//   // TODO: 生成皮肤蒙皮网格并上传
//   (void)m; (void)con;
//   return VK_SUCCESS;
// }

// void convertConvexHullToVertexBuffer(const mjModel* m, int meshid,
//                                     VertexPNT** vertices, uint32_t* vertexCount,
//                                     uint32_t** indices, uint32_t* indexCount) {
//   // TODO: 从凸包拓扑生成三角面数据
//   (void)m; (void)meshid;
//   *vertices = nullptr;
//   *indices = nullptr;
//   *vertexCount = 0;
//   *indexCount = 0;
// }

// VkResult createOffscreenResources(mjrContextVk* con, int width, int height) {
//   // TODO: 创建离屏渲染资源
//   (void)con; (void)width; (void)height;
//   return VK_SUCCESS;
// }

// VkResult createFontAtlas(mjrContextVk* con, int fontscale) {
//   // TODO: 上传字体图集
//   (void)con; (void)fontscale;
//   return VK_SUCCESS;
// }

// VkResult createShadowResources(mjrContextVk* con, int shadowsize) {
//   // TODO: 创建阴影贴图资源
//   (void)con; (void)shadowsize;
//   return VK_SUCCESS;
// }

// VkResult uploadTextures(const mjModel* m, mjrContextVk* con) {
//   // TODO: 上传纹理数据
//   (void)m; (void)con;
//   return VK_SUCCESS;
// }

// VkResult createMaterialDescriptors(const mjModel* m, mjrContextVk* con) {
//   // TODO: 创建材质描述符
//   (void)m; (void)con;
//   return VK_SUCCESS;
// }

// VkResult createRenderPipelines(mjrContextVk* con) {
//   // TODO: 创建渲染管线
//   (void)con;
//   return VK_SUCCESS;
// }

// VkResult createSyncObjects(mjrContextVk* con) {
//   // TODO: 创建同步对象
//   (void)con;
//   return VK_SUCCESS;
// }