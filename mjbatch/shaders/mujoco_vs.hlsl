cbuffer CameraData : register(b0, space0) {
    float4x4 view_proj;
    float3 camera_position;
    float padding0;
};

struct LightInfo {
    float3 position; 
    uint type; // 0: spot, 1: directional, 2: point

    float3 direction;
    float range;
    
    float3 ambient;
    float cutoff;
    
    float3 diffuse;
    float exponent;
    
    float3 specular;
    float bulbRadius;
    
    float3 attenuation; 
    float intensity;    

    float3 padding; 
    uint castShadow;

    float4x4 view_proj;
};

cbuffer LightData : register(b1, space0) {
    LightInfo input_lights[10];
    uint input_lightCount;
    float3 pad;
};

struct PushConstants {
    float4x4 model;
    float4 material_rgba;
    float material_specular;
    float material_emission;
    float material_shininess;
    float material_reflectance;
    
    int texture_index;
    int texture_type;
    float4 shadow_atlas_params;
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
    float4 shadow_coord : SHADOW_COORD;
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
    output.shadow_coord = mul(input_lights[1].view_proj, world_pos);

    
    return output;
}