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

#include <vector> // 确保包含 vector

// 输入: rgba_data (指向 RGBA 数据，每像素4字节)
// 输出: 标准 PPM 文件 (每像素3字节，丢弃 Alpha)
static bool write_ppm(const std::string& path, const unsigned char* rgba_data, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;

    // 1. 写入 P6 头 (P6 代表二进制 RGB)
    ofs << "P6\n" << w << " " << h << "\n255\n";

    // 2. 创建一行 RGB 数据的缓存 (减少磁盘 I/O 次数)
    std::vector<unsigned char> row_buffer(w * 3);

    // 3. 逐行转换并写入
    for (int y = 0; y < h; ++y) {
        const unsigned char* src_row = rgba_data + (size_t)y * w * 4; // 源指针 (RGBA)
        unsigned char* dst_row = row_buffer.data();                   // 目标指针 (RGB)

        for (int x = 0; x < w; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 4 + 0]; // R
            dst_row[x * 3 + 1] = src_row[x * 4 + 1]; // G
            dst_row[x * 3 + 2] = src_row[x * 4 + 2]; // B
            // src_row[x * 4 + 3] (Alpha) 被忽略
        }

        // 将这一行 RGB 数据写入文件
        ofs.write(reinterpret_cast<const char*>(row_buffer.data()), w * 3);
    }

    return ofs.good();
}

int main() {
  // ---- 1) Load the model ----
  const std::string xml_nut = "./model/lnuts.xml";
  const std::string xml_lift = "./model/lift.xml";

  char error[1024] = {0};

  mjModel* m1 = mj_loadXML(xml_nut.c_str(), nullptr, error, sizeof(error));
  if (!m1) {
    std::fprintf(stderr, "[Error] mj_loadXML failed for nuts: %s\n", error);
    return 3;
  }

  mjModel* m2 = mj_loadXML(xml_lift.c_str(), nullptr, error, sizeof(error));
  if (!m2) {
    std::fprintf(stderr, "[Error] mj_loadXML failed for lift: %s\n", error);
    mj_deleteModel(m1);
    return 3;
  }

  std::vector<mjModel*> models;
  models.push_back(m1);
  models.push_back(m2);

  BatchRendererConfig cfg;
  cfg.batch_size = 2;          
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

  // ---- 2) create mjData ----
  std::vector<mjData*> datas(cfg.batch_size, nullptr);
  for (int i = 0; i < cfg.batch_size; ++i) {
    datas[i] = mj_makeData(models[i]);
    mj_forward(models[i], datas[i]);
  }

  // physics step
  for(int step=0; step<100; step++) {
    for (int i = 0; i < cfg.batch_size; ++i) {
      mj_step(models[i], datas[i]);
    }
  }

  // ---- 3) Render ----
  auto result = renderer->Render(datas.data(), nullptr);
  if (!result) {
    std::fprintf(stderr, "[Error] Render failed: %s\n", result.message.c_str());
    return 6;
  }

  // ---- 4) merge two images ----
  const unsigned char* img0 = renderer->GetRGBFrame(0); // lnuts
  const unsigned char* img1 = renderer->GetRGBFrame(1); // lift

  if (img0 && img1) {
      int w = cfg.frame_width;
      int h = cfg.frame_height;
      int combined_w = w * 2;

      std::vector<unsigned char> combined_buffer(combined_w * h * 4);

      for (int y = 0; y < h; ++y) {
          size_t src_offset = (size_t)y * w * 4;
          size_t dst_offset = (size_t)y * combined_w * 4;

          // Image 0
          std::memcpy(&combined_buffer[dst_offset], 
                      &img0[src_offset], 
                      w * 4);

          // Image 1
          std::memcpy(&combined_buffer[dst_offset + w * 4], 
                      &img1[src_offset], 
                      w * 4);
      }

      if (write_ppm("merged_result.ppm", combined_buffer.data(), combined_w, h)) {
          std::printf("[Success] Wrote merged_result.ppm (Size: %dx%d)\n", combined_w, h);
      } else {
          std::fprintf(stderr, "[Error] Failed to write PPM file\n");
      }
  } else {
      std::fprintf(stderr, "[Error] Failed to retrieve frames from renderer\n");
  }

  // ---- 5) clear ----
  for (auto* d : datas) mj_deleteData(d);
  renderer.reset(); 
  mj_deleteModel(m1);
  mj_deleteModel(m2);

  return 0;
}