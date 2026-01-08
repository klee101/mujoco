// simple.hlsl

// Set 0: Camera Data
struct CameraData {
    float4x4 viewProj;
    float3 camPos;
};
cbuffer CameraBuffer : register(b0, space0) {
    CameraData camera;
};

// Set 1: Object Data
struct ObjectData {
    float4x4 model;
    float4 color;
};
cbuffer ObjectBuffer : register(b0, space1) {
    ObjectData object;
};

// Binding 1: Texture
Texture2D g_AlbedoMap : register(t1, space1);
// Binding 2: Sampler
SamplerState g_Sampler : register(s2, space1);

// Set 2: Material Data (Binding 0 - Array)
Texture2D g_Textures[4] : register(t0, space2);

struct PushConsts {
    float4x4 modelMatrix;
    float time;
};
[[vk::push_constant]] ConstantBuffer<PushConsts> pc;

struct VSInput {
    float3 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct VSOutput {
    float4 pos : SV_Position;
    float2 uv : TEXCOORD;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    
    // 正常计算
    float4 worldPos = mul(object.model, float4(input.pos, 1.0));
    output.pos = mul(camera.viewProj, worldPos);
    output.uv = input.uv;

    // [新增] 强制使用资源，防止被编译器优化掉 (Dead Code Elimination)
    // 注意：在 Vertex Shader 中采样纹理必须用 SampleLevel (因为没有导数)
    float4 dummyColor = g_AlbedoMap.SampleLevel(g_Sampler, input.uv, 0);
    
    // 读取数组纹理
    float4 dummyArray = g_Textures[0].SampleLevel(g_Sampler, input.uv, 0);

    // 将结果微小地叠加到输出上，让编译器认为这个计算是"有意义"的
    output.pos += (dummyColor + dummyArray) * 0.000001;

    output.pos += pc.time * 0.001;

    return output;
}