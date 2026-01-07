// Shadow.hlsl

// Must match the PushConstantsShadow struct in C++
struct PushConstantsShadow {
    float4x4 mvp; // LightViewProj * Model
};

[[vk::push_constant]]
ConstantBuffer<PushConstantsShadow> pushConstShadow;

struct VSInput {
    float3 position : POSITION;
    // We don't need Normal/Color/UV for depth-only rendering
};

// Returns position in Light Clip Space
float4 VSMain(VSInput input) : SV_Position {
    return mul(pushConstShadow.mvp, float4(input.position, 1.0));
}