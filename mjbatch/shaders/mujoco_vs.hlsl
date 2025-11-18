cbuffer CameraData : register(b0) {
    float4x4 view_proj;
    float3 camera_position;
    float _pad1;
    float3 camera_forward;
    float _pad2;
    float3 camera_up;
    float _pad3;
    float near_plane;
    float far_plane;
    float fov;
    float _pad4;
};

// Push constant for per-drawable data (model + material)
struct PushConstants {
    float4x4 model;           // offset 0, 64 bytes
    float4 material_rgba;     // offset 64, 16 bytes
    float3 material_specular; // offset 80, 12 bytes
    float material_emission;  // offset 92, 4 bytes
    float material_shininess; // offset 96, 4 bytes
    int texture_id;           // offset 100, 4 bytes
    float _pad1, _pad2;       // offset 104, 8 bytes
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConst;

struct VSInput {
    float3 position : POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
    float3 world_pos : WORLD_POS;
    float3 view_dir : VIEW_DIR;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    
    // Transform to world space using model matrix from push constant
    float4 world_pos = mul(pushConst.model, float4(input.position, 1.0));
    output.world_pos = world_pos.xyz;
    
    // Transform to clip space
    output.position = mul(view_proj, world_pos);
    
    // Transform normal to world space (assuming uniform scaling)
    output.normal = normalize(mul((float3x3)pushConst.model, input.normal));
   
    output.texcoord = input.texcoord;
    output.color = input.color;
    
    // Calculate view direction
    output.view_dir = normalize(camera_position - world_pos.xyz);
     
    return output;
}