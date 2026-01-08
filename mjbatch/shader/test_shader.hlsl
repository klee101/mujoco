// test_shader.hlsl

// [测试点1] Push Constants
struct PushConsts {
    float4x4 modelMatrix;
    float time;
};
[[vk::push_constant]] ConstantBuffer<PushConsts> pc;

// [测试点2] Set 0 - Camera Uniform
struct CameraData {
    float4x4 viewProj;
    float3 camPos;
};
cbuffer CameraBuffer : register(b0, space0) {
    CameraData camera;
};

// [测试点3] Set 1 - Texture & Sampler
Texture2D g_AlbedoMap : register(t0, space1);
SamplerState g_Sampler : register(s0, space1);

struct VSInput {
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct PSInput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD;
};

// [测试点4] Vertex Shader Entry
PSInput VSMain(VSInput input) {
    PSInput output;
    // 使用资源防止被优化
    float4 worldPos = mul(pc.modelMatrix, float4(input.pos, 1.0));
    output.pos = mul(camera.viewProj, worldPos);
    output.uv = input.uv;
    return output;
}

// [测试点5] Pixel Shader Entry
float4 PSMain(PSInput input) : SV_Target {
    // 使用纹理防止被优化
    float4 color = g_AlbedoMap.Sample(g_Sampler, input.uv);
    return color + float4(camera.camPos * 0.0001, 1.0); // 强行引用 camera 防止优化
}