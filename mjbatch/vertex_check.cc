#include <mujoco/mujoco.h>
#include <iostream>
#include <memory>
#include "object_manager.h"
#include "builtin.h"
#include "utils.h"

#include "scene.h" 
using namespace mujoco::mjbatch;

int main(int argc, const char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: test_scene model.xml\n";
        return 1;
    }

    // 加载 MuJoCo 模型
    char error[1000] = "Could not load model";
    mjModel* m = mj_loadXML(argv[1], 0, error, 1000);
    if (!m) {
        std::cerr << "Model load error: " << error << std::endl;
        return 1;
    }

    // 创建 data 并运行一次 forward 初始化
    mjData* d = mj_makeData(m);
    mj_forward(m, d);

    // 创建场景对象（mjvScene）
    mjvScene scn;
    mjvCamera cam;
    mjvOption opt;
    mjv_defaultScene(&scn);
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);

    mjv_makeScene(m, &scn, 2000);
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);

    Scene testScene(m, &scn);
    std::cout << "Initial Scene created." << std::endl;
    std::cout << "Geom count: " << scn.ngeom << std::endl;
    std::cout << "Shape count: " << testScene.transforms.size() << std::endl;

    mj_step(m, d);
    mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
    testScene.UpdateScene(m, &scn);
    std::cout << "Scene updated successfully." << std::endl;
    std::cout << "Updated shape count: " << testScene.transforms.size() << std::endl;

    mjv_freeScene(&scn);
    mj_deleteData(d);
    mj_deleteModel(m);

    return 0;
}
