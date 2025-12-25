#include "renderer.h"
#include <mujoco/mujoco.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstring> 

// [RenderDoc] 1. 头文件
#include <dlfcn.h> 
#include "renderdoc_app.h" 

RENDERDOC_API_1_1_2 *rdoc_api = NULL;

// ... (write_ppm 函数保持不变) ...
static bool write_ppm(const std::string& path, const unsigned char* rgba_data, int w, int h) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs << "P6\n" << w << " " << h << "\n255\n";
    std::vector<unsigned char> row_buffer(w * 3);
    for (int y = 0; y < h; ++y) {
        const unsigned char* src_row = rgba_data + (size_t)y * w * 4;
        unsigned char* dst_row = row_buffer.data();
        for (int x = 0; x < w; ++x) {
            dst_row[x * 3 + 0] = src_row[x * 4 + 0];
            dst_row[x * 3 + 1] = src_row[x * 4 + 1];
            dst_row[x * 3 + 2] = src_row[x * 4 + 2];
        }
        ofs.write(reinterpret_cast<const char*>(row_buffer.data()), w * 3);
    }
    return ofs.good();
}

int main() {
    // [RenderDoc] 2. 加载 RenderDoc 库
    void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD);
    if (!mod) {
        mod = dlopen("librenderdoc.so", RTLD_NOW);
    }
    
    if (mod) {
        pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
        int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
        
        if (ret == 1) {
            printf("[RenderDoc] API loaded.\n");
            
            // [关键修改] 设置捕获文件保存路径模板
            // 不需要加后缀，RenderDoc 会自动变为 mujoco_capture_frame1.rdc
            rdoc_api->SetCaptureFilePathTemplate("mujoco_capture");
            
            // 开启所有资源的引用（防止资源未被保存）
            rdoc_api->SetCaptureOptionU32(eRENDERDOC_Option_RefAllResources, 1);
            rdoc_api->SetCaptureOptionU32(eRENDERDOC_Option_CaptureCallstacks, 1);
        }
    } else {
        printf("[RenderDoc] librenderdoc.so NOT loaded. check LD_LIBRARY_PATH.\n");
    }

    // ---- Load Models (保持不变) ----
    const std::string xml_nut = "./model/lnuts.xml";
    const std::string xml_lift = "./model/lift.xml";
    char error[1024] = {0};

    mjModel* m1 = mj_loadXML(xml_nut.c_str(), nullptr, error, sizeof(error));
    if (!m1) return 3;
    mjModel* m2 = mj_loadXML(xml_lift.c_str(), nullptr, error, sizeof(error));
    if (!m2) return 3;

    std::vector<mjModel*> models = {m1, m2};
    BatchRendererConfig cfg;
    cfg.batch_size = 2;          
    cfg.frame_width = 1920;
    cfg.frame_height = 1080;
    cfg.enable_depth = true;
    cfg.enable_validation = false; // 抓帧时建议关闭 Validation Layer 避免干扰

    auto renderer = BatchRenderer::Create(models, cfg);
    if (!renderer) return 4;

    std::vector<mjData*> datas(cfg.batch_size, nullptr);
    for (int i = 0; i < cfg.batch_size; ++i) {
        datas[i] = mj_makeData(models[i]);
        mj_forward(models[i], datas[i]);
    }

    for(int step=0; step<100; step++) {
        for (int i = 0; i < cfg.batch_size; ++i) mj_step(models[i], datas[i]);
    }

    // ---- [RenderDoc] 3. 触发捕获 ----
    if (rdoc_api) {
        printf("[RenderDoc] Starting Capture...\n");
        rdoc_api->StartFrameCapture(NULL, NULL); 
    }

    auto result = renderer->Render(datas.data(), nullptr);

    if (rdoc_api) {
        rdoc_api->EndFrameCapture(NULL, NULL); 
        printf("[RenderDoc] Capture Saved: ./mujoco_capture_frame1.rdc\n");
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