// ============================================================================
// Irradiance Cubemap Generation Compute Shader
// 输入: Equirectangular HDR 2D Texture
// 输出: Irradiance Cubemap (32x32, 漫反射卷积)
// ============================================================================

Texture2D<float4>       inputEquirect  : register(t0);
SamplerState            linearSampler  : register(s0);
RWTexture2DArray<float4> outputIrradiance : register(u1);

static const float PI = 3.14159265359;

// 将 Cubemap 像素坐标转换为方向向量
float3 GetCubeDir(uint3 id, float width, float height) {
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    uv = uv * 2.0 - 1.0; // [-1, 1]

    float3 dir;
    // Layer order: +X, -X, +Y, -Y, +Z, -Z
    switch(id.z) {
        case 0: dir = float3( 1.0,  uv.y, -uv.x); break; // +X
        case 1: dir = float3(-1.0,  uv.y,  uv.x); break; // -X
        case 2: dir = float3( uv.x,  1.0, -uv.y); break; // +Y
        case 3: dir = float3( uv.x, -1.0,  uv.y); break; // -Y
        case 4: dir = float3( uv.x,  uv.y,  1.0); break; // +Z
        case 5: dir = float3(-uv.x,  uv.y, -1.0); break; // -Z
    }
    return normalize(dir);
}

// 方向向量转 Equirectangular UV
float2 DirToEquirectUV(float3 v) {
    float2 uv = float2(atan2(v.z, v.x), asin(clamp(v.y, -1.0, 1.0)));
    uv *= float2(0.1591549, 0.3183099); // 1/(2PI), 1/PI
    uv += 0.5;
    return uv;
}

// ============================================================================
// Main: 对法线 N 的半球做黎曼积分求 Irradiance
// sampleDelta 越小质量越高，代价是更多采样
// ============================================================================
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float width, height, elements;
    outputIrradiance.GetDimensions(width, height, elements);
    if (id.x >= (uint)width || id.y >= (uint)height) return;

    float3 N = GetCubeDir(id, width, height);

    // 构建 TBN，用于在法线所在半球采样
    float3 up    = abs(N.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(up, N));
    up           = cross(N, right);

    float3 irradiance = float3(0, 0, 0);
    float  sampleDelta = 0.025; // 步长 (弧度)，约 5050 次采样
    float  nrSamples   = 0.0;

    for (float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta) {
        for (float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta) {
            // 球坐标 -> 切线空间方向
            float3 tangentSample = float3(
                sin(theta) * cos(phi),
                sin(theta) * sin(phi),
                cos(theta)
            );

            // 切线空间 -> 世界空间
            float3 sampleVec = tangentSample.x * right
                             + tangentSample.y * up
                             + tangentSample.z * N;

            float2 uv = DirToEquirectUV(normalize(sampleVec));
            float3 hdrSample = inputEquirect.SampleLevel(linearSampler, uv, 0).rgb;

            // 漫反射积分权重
            irradiance += hdrSample * cos(theta) * sin(theta);
            nrSamples  += 1.0;
        }
    }

    irradiance = PI * irradiance / nrSamples;
    outputIrradiance[id] = float4(irradiance, 1.0);
}
