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

struct PSInput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
    float3 world_pos : WORLD_POS;
    float3 view_dir : VIEW_DIR;
};

float4 PSMain(PSInput input) : SV_Target {
    // Use material color or vertex color
    float4 base_color = pushConst.material_rgba * input.color;
    
    // Simple directional light
    float3 normal = normalize(input.normal);
    float3 light_dir = normalize(float3(1.0, 1.0, 1.0));
    
    // Ambient
    float3 ambient = base_color.rgb * 0.2;
    
    // Diffuse
    float ndotl = max(dot(normal, light_dir), 0.0);
    float3 diffuse = base_color.rgb * ndotl * 0.6;
    
    // Specular (Blinn-Phong)
    float3 view_dir = normalize(input.view_dir);
    float3 half_dir = normalize(light_dir + view_dir);
    float spec = pow(max(dot(normal, half_dir), 0.0), pushConst.material_shininess);
    float3 specular = pushConst.material_specular * spec * 0.3;
    
    // Emission
    float3 emission = base_color.rgb * pushConst.material_emission;
    
    float3 final_color = ambient + diffuse + specular + emission;
    return float4(final_color, base_color.a);
}
