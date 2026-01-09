// mjbatch/stb_impl.cc

// ----------------------------------------------------------------------------
// GCC / Clang: 暂时禁用 implicit-fallthrough 警告
// ----------------------------------------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wimplicit-fallthrough"
    // 有些旧版本 GCC 可能还需要忽略这个
    #pragma GCC diagnostic ignored "-Wunused-function" 
#endif

// 定义此宏以启用函数实现
#define STB_IMAGE_IMPLEMENTATION

// 包含头文件
#include "stb_image.h"

// 恢复警告设置
#if defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic pop
#endif