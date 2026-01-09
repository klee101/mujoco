// Input: Equirectangular 2D HDR map
Texture2D<float4> inputTexture : register(t0);
SamplerState defaultSampler : register(s0);

// Output: Cubemap face (Mip 0)
// 注意：HLSL CS 写 Cubemap 通常作为 RWTexture2DArray 处理
RWTexture2DArray<float4> outputCube : register(u1);

static const float PI = 3.14159265359;

float3 GetCubeDir(uint3 id, float width, float height) {
    float2 uv = (float2(id.xy) + 0.5) / float2(width, height);
    uv = uv * 2.0 - 1.0; // [-1, 1]

    float3 dir;
    // Layer order: +X, -X, +Y, -Y, +Z, -Z
    switch(id.z) {
        case 0: dir = float3(1.0,  uv.y, -uv.x); break; // +X
        case 1: dir = float3(-1.0, uv.y,  uv.x); break; // -X
        case 2: dir = float3(uv.x, 1.0, -uv.y); break; // +Y
        case 3: dir = float3(uv.x, -1.0, uv.y); break; // -Y
        case 4: dir = float3(uv.x, uv.y, 1.0); break; // +Z
        case 5: dir = float3(-uv.x, uv.y, -1.0); break; // -Z
    }
    return normalize(dir);
}

float2 DirToUV(float3 v) {
    float2 uv = float2(atan2(v.z, v.x), asin(v.y));
    uv *= float2(0.1591, 0.3183); // inv(2*PI), inv(PI)
    uv += 0.5;
    return uv;
}

[numthreads(32, 32, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float width, height, elements;
    outputCube.GetDimensions(width, height, elements);

    if (id.x >= width || id.y >= height) return;

    float3 dir = GetCubeDir(id, width, height);
    
    // Sample spherical map
    // Note: Y might need flip depending on coordinate system
    float2 uv = DirToUV(normalize(dir)); 
    
    // Bilinear sample
    float4 color = inputTexture.SampleLevel(defaultSampler, uv, 0);

    outputCube[id] = color;
}