#include "renderer.h"
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>
#include <cstring> // 必须包含，用于 memcpy

// PPM 写入函数保持不变
static bool write_ppm(const std::string& path, const unsigned char* data, int w, int h) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs << "P6\n" << w << " " << h << "\n255\n";
  ofs.write(reinterpret_cast<const char*>(data), std::streamsize(w*h*3));
  return ofs.good();
}

int main() {
  // ---- 1) 加载模型 ----
  const std::string xml_nut = "./model/lnuts.xml";
  const std::string xml_lift = "./model/lift.xml";

  char error[1024] = {0};
  
  // 加载模型 1
  mjModel* m1 = mj_loadXML(xml_nut.c_str(), nullptr, error, sizeof(error));
  if (!m1) {
    std::fprintf(stderr, "[Error] mj_loadXML failed for nuts: %s\n", error);
    return 3;
  }

  // 加载模型 2
  mjModel* m2 = mj_loadXML(xml_lift.c_str(), nullptr, error, sizeof(error));
  if (!m2) {
    std::fprintf(stderr, "[Error] mj_loadXML failed for lift: %s\n", error);
    mj_deleteModel(m1);
    return 3;
  }

  // 构建模型列表
  std::vector<mjModel*> models;
  models.push_back(m1);
  models.push_back(m2);

  // ---- 2) 创建 BatchRenderer ----
  BatchRendererConfig cfg;
  cfg.batch_size = 2;          // 对应两个模型
  cfg.frame_width = 1920;
  cfg.frame_height = 1080;
  cfg.enable_depth = true;
  cfg.enable_validation = false;

  auto renderer = BatchRenderer::Create(models, cfg);
  if (!renderer || !renderer->IsValid()) {
    std::fprintf(stderr, "[Error] BatchRenderer::Create failed\n");
    mj_deleteModel(m1);
    mj_deleteModel(m2);
    return 4;
  }

  // ---- 3) 创建数据 mjData ----
  std::vector<mjData*> datas(cfg.batch_size, nullptr);
  for (int i = 0; i < cfg.batch_size; ++i) {
    datas[i] = mj_makeData(models[i]);
    mj_forward(models[i], datas[i]);
  }

  // 物理仿真 (让物体动起来一点)
  for(int step=0; step<0; step++) {
    for (int i = 0; i < cfg.batch_size; ++i) {
      mj_step(models[i], datas[i]);
    }
  }

  // ---- 4) Render ----
  auto result = renderer->Render(datas.data(), nullptr);
  if (!result) {
    std::fprintf(stderr, "[Error] Render failed: %s\n", result.message.c_str());
    return 6;
  }

  // ---- 5) 图像合并逻辑 (Side-by-Side) ----
  // 获取两个 batch 的图像指针
  const unsigned char* img0 = renderer->GetRGBFrame(0); // lnuts
  const unsigned char* img1 = renderer->GetRGBFrame(1); // lift

  if (img0 && img1) {
      int w = cfg.frame_width;
      int h = cfg.frame_height;
      int combined_w = w * 2; // 宽度翻倍
      
      // 分配大缓冲区 (RGB 3通道)
      std::vector<unsigned char> combined_buffer(combined_w * h * 3);

      // 按行拷贝数据
      for (int y = 0; y < h; ++y) {
          // 计算源和目标的行偏移
          size_t src_offset = (size_t)y * w * 3;
          size_t dst_offset = (size_t)y * combined_w * 3;

          // 拷贝左图 (Image 0)
          std::memcpy(&combined_buffer[dst_offset], 
                      &img0[src_offset], 
                      w * 3);

          // 拷贝右图 (Image 1)，注意目标偏移要加上左图的宽度
          std::memcpy(&combined_buffer[dst_offset + w * 3], 
                      &img1[src_offset], 
                      w * 3);
      }

      // 写入合并后的文件
      if (write_ppm("merged_result.ppm", combined_buffer.data(), combined_w, h)) {
          std::printf("[Success] Wrote merged_result.ppm (Size: %dx%d)\n", combined_w, h);
      } else {
          std::fprintf(stderr, "[Error] Failed to write PPM file\n");
      }
  } else {
      std::fprintf(stderr, "[Error] Failed to retrieve frames from renderer\n");
  }

  // ---- 6) 清理 ----
  for (auto* d : datas) mj_deleteData(d);
  // 注意：Renderer 析构时会自动清理 Vulkan 资源，但 mjModel 需要手动释放
  // Renderer 对 mjModel 是弱引用
  renderer.reset(); // 先释放 Renderer，确保它不再使用 model
  mj_deleteModel(m1);
  mj_deleteModel(m2);

  return 0;
}