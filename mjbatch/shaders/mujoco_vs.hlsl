cbuffer CameraData : register(b0, space0) {
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

struct PushConstants {
    float4x4 model;
    float4 material_rgba;
    float3 material_specular;
    float material_emission;
    float material_shininess;
    float material_reflectance;
    
    int texture_index; // 在对应数组(2D或Cube)中的索引
    int texture_type;  // 0: Unlit/Color, 1: 2D Texture, 2: Cube Texture
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
    float3 sample_vec : SAMPLE_VEC;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    
    // Transform to world space
    float4 world_pos = mul(pushConst.model, float4(input.position, 1.0));
    output.world_pos = world_pos.xyz;
    
    // Transform to clip space
    output.position = mul(view_proj, world_pos);
    
    // Transform normal with inverse transpose for non-uniform scaling
    // For uniform scaling, this simplifies to (float3x3)model
    float3x3 normal_matrix = (float3x3)pushConst.model;
    output.normal = normalize(mul(normal_matrix, input.normal));
    
    // Pass through texture coordinates and vertex color
    output.texcoord = input.texcoord;
    output.color = input.color;
    
    // Calculate view direction in world space
    output.view_dir = normalize(camera_position - world_pos.xyz);
    
    output.sample_vec = input.position; // For environment mapping

    
    return output;
}