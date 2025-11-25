#include "renderer.h"
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <string>

static bool write_ppm(const std::string& path, const unsigned char* data, int w, int h) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) return false;
  ofs << "P6\n" << w << " " << h << "\n255\n";
  ofs.write(reinterpret_cast<const char*>(data), std::streamsize(w*h*3));
  return ofs.good();
}

int main() {
  // ---- 1) 加载 humanoid.xml ----

  const std::string xml_path = "./model/lift.xml";

  char error[1024] = {0};
  const mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  if (!m) {
    std::fprintf(stderr, "[humanoid] mj_loadXML failed: %s\n", error);
    return 3;
  }

  // ---- 2) 创建 BatchRenderer ----
  BatchRendererConfig cfg;
  cfg.batch_size = 1;
  cfg.frame_width = 1920;
  cfg.frame_height = 1080;
  cfg.enable_depth = true;
  cfg.enable_validation = false;

  auto renderer = BatchRenderer::Create(m, cfg);
  if (!renderer || !renderer->IsValid()) {
    std::fprintf(stderr, "[humanoid] BatchRenderer::Create failed\n");
    mj_deleteModel((mjModel*)m);
    return 4;
  }

  // ---- 3) 创建数据 mjData ----
  std::vector<mjData*> datas(cfg.batch_size, nullptr);
  for (int i = 0; i < cfg.batch_size; ++i) {
    datas[i] = mj_makeData(m);
    mj_forward(m, datas[i]);
  }
  // 物理仿真
  for(int step=0; step<600; step++) {
    for (int i = 0; i < cfg.batch_size; ++i) {
      mj_step(m, datas[i]);
    }
  }

  // ---- 4) Render 1帧 ----
  auto result = renderer->Render(datas.data(), nullptr);
  if (!result) {
    std::fprintf(stderr, "[humanoid] Render failed: %s\n", result.message.c_str());
    return 6;
  }

  const unsigned char* px = renderer->GetRGBFrame(0);
  const auto& rgb = renderer->GetRGBBuffer();
  if (!px) px = rgb.data();

  write_ppm("humanoid.ppm", px, cfg.frame_width, cfg.frame_height);
  std::printf("[humanoid] wrote humanoid.ppm\n");

  // ---- 5) 清理 ----
  for (auto* d : datas) mj_deleteData(d);
  mj_deleteModel((mjModel*)m);

  return 0;
}
