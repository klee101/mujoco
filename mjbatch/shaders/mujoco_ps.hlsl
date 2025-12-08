// ============================================================================
// RESOURCES
// ============================================================================

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

struct LightInfo {
    // Offset 0: position and type (padding to float4)
    float3 position; 
    uint32_t type; // 0: spot, 1: directional, 2: point

    float3 direction;
    float range;
    
    float3 ambient;
    float cutoff;
    
    float3 diffuse;
    float exponent;
    
    float3 specular;
    float bulbRadius;
    
    float3 attenuation; // x: constant, y: linear, z: quadratic
    float intensity;    // NOTE: This will be read but is noted as unused for intensity scaling.

    uint32_t castShadow;
    float padding[3]; 
};

cbuffer LightData : register(b1, space0) {
    LightInfo lights[10];
    uint32_t lightCount;
    float pad[3];
};

struct PushConstants {
    float4x4 model;
    float4 material_rgba;
    float3 material_specular;
    float material_emission;
    float material_shininess;
    float material_reflectance;
    
    int texture_index; // 在对应数组(2D或Cube)中的索引
    int texture_type;  // -1: Unlit/Color, 0: 2D Texture, 1: Cube Texture
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConst;

// ============================================================================
// TEXTURE RESOURCES (BINDING = 2 used for descriptor set 1, assuming space1)
// ============================================================================

// Binding 0: 2D Texture Array
Texture2D g_textures[] : register(t0, space1);
// Binding 2: Sampler
SamplerState g_sampler : register(s1, space1);

struct PSInput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
    float3 world_pos : WORLD_POS;
    float3 view_dir : VIEW_DIR;
    float3 sample_vec : SAMPLE_VEC;
};

// ============================================================================
// DEBUG MODES
// ============================================================================
// 0: Off
// 1: Normals (World Space)
// 2: Reflectance Heatmap (Blue=Low, Red=High)
// 3: Diffuse Only (White Material)
// 4: Specular Only
// 5: Lighting Complexity (Heatmap of light count)


#ifndef DEBUG_VIEW
  #define DEBUG_VIEW 0
#endif

// Helper for debug colors
static float3 HeatMap(float t) {
    float3 cold = float3(0, 0, 1);
    float3 mid = float3(0, 1, 0);
    float3 hot = float3(1, 0, 0);
    return t < 0.5 ? lerp(cold, mid, t * 2.0) : lerp(mid, hot, (t - 0.5) * 2.0);
}

// ============================================================================
// ACES TONE MAPPING (更美观的电影级色调)
// ============================================================================
float3 ACESToneMapping(float3 color) {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;
    return saturate((color * (A * color + B)) / (color * (C * color + D) + E));
}

// ============================================================================
// [NEW] CUBE MAP PROJECTION HELPER
// ============================================================================
// 手动计算 CubeMap 投影。
float2 CalculateCubeUV(float3 v) {
    float3 vAbs = abs(v);
    float ma; // Major Axis Magnitude
    float2 uv;
    
    if(vAbs.z >= vAbs.x && vAbs.z >= vAbs.y) {
        // Front / Back Face
        ma = vAbs.z;
        uv = float2(v.x, -v.y); 
    } else if(vAbs.y >= vAbs.x) {
        // Top / Bottom Face
        ma = vAbs.y;
        uv = float2(v.x, v.z);
    } else {
        // Left / Right Face
        ma = vAbs.x;
        uv = float2(v.z, -v.y);
    }

    return (uv / ma) * 0.5 + 0.5;
}
// ============================================================================
// LIGHTING CALCULATION
// ============================================================================
void CalculateLightContribution(
    LightInfo light, 
    float3 worldPos,
    float3 N,           // Normal
    float3 V,           // View Dir
    float3 matDiffuse,  // Base Color
    float3 matSpecular, // Specular Color
    float matShininess, // Shininess
    float matReflectance, // [New] Reflectance factor
    inout float3 outDiffuse,
    inout float3 outSpecular,
    inout float3 outAmbient
) {
    // 1. 计算光照向量 L 和 衰减 Attenuation
    float3 L;
    float attenuation = 1.0;

    if (light.type == 1) { // Directional
        L = normalize(-light.direction);
        // Directional lights usually don't have distance attenuation in standard pipelines
        attenuation = 1.0; 
    } else { // Point (2) or Spot (0)
        float3 lightVec = light.position - worldPos;
        float d = length(lightVec);
        L = lightVec / d;
        
        // 避免除零
        float denom = light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d;
        if (denom < 0.0001) denom = 1.0;
        attenuation = 1.0 / denom;

        // Spot Light Cutoff
        if (light.type == 0) {
            float theta = dot(-L, normalize(light.direction));
            float outerCutoff = cos(radians(light.cutoff));
            float epsilon = 0.1; // Soft edge
            float spotIntensity = smoothstep(outerCutoff, outerCutoff + epsilon, theta);
            attenuation *= pow(spotIntensity, light.exponent);
        }
    }

    if (attenuation < 0.001) return;

    // 2. Diffuse (Lambert)
    float NdotL = max(dot(N, L), 0.0);
    float3 lightDiffuse = light.diffuse * attenuation;
    outDiffuse += lightDiffuse * matDiffuse * NdotL;

    // 3. Ambient
    // Ambient usually isn't attenuated by distance/angle in simple models, 
    // but here we simply add it weighted by material color.
    outAmbient += light.ambient * matDiffuse;

    // 4. Specular (Blinn-Phong with Fresnel-ish Reflectance)
    if (NdotL > 0.0) {
        float3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        
        // [Reflectance Logic]
        // 使用 Reflectance 来调制高光强度。
        // 添加简单的 Fresnel 近似：视线越平行于表面，反射越强 (Schlick approximation idea)
        // F = R + (1-R) * (1 - dot(H,V))^5
        float baseF = matReflectance; 
        float fresnel = baseF + (1.0 - baseF) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
        
        float specPower = pow(NdotH, matShininess);
        
        // 最终高光 = 光源高光色 * 材质高光色 * 几何衰减 * 高光指数 * (反射率/Fresnel系数)
        float3 lightSpecular = light.specular * attenuation;
        outSpecular += lightSpecular * matSpecular * specPower * fresnel;
    }
}

// ============================================================================
// MAIN SHADER
// ============================================================================
float4 PSMain(PSInput input) : SV_Target {
    // 1. Prepare Vectors
    float3 N = normalize(input.normal);
    float3 V = normalize(input.view_dir);
    // Double sided lighting fix (optional): if normal points away from camera, flip it
    // if (dot(N, V) < 0) N = -N; 

    // 2. Sample Texture / Base Color
    float4 base_color = pushConst.material_rgba * input.color;
    
    // Texture Logic
    if (pushConst.texture_index >= 0) {
        if (pushConst.texture_type == 0) {
            // [CASE 0] Standard 2D Texture
            float4 tex = g_textures[pushConst.texture_index].Sample(g_sampler, input.texcoord);
            base_color *= tex;
        } 
        else if (pushConst.texture_type == 1) {
            // [CASE 1] Simulated Cube Texture (using 2D array)
            // 原逻辑: g_cube_textures[...].Sample(..., uvw);
            // 新逻辑: 手动计算投影 UV -> 采样 2D 数组
            
            float3 uvw = normalize(input.sample_vec); // 或者是 input.world_pos - camera_pos，取决于你的顶点着色器传参
            
            // [+] 调用手动计算函数
            float2 cube_uv = CalculateCubeUV(uvw);
            
            // [+] 使用计算出的 UV 采样 2D 纹理数组
            // 注意：texture_index 现在指向的是 stored in textures_2d 的那个"原本是cube"的纹理
            float4 tex = g_textures[pushConst.texture_index].Sample(g_sampler, cube_uv);
            
            base_color *= tex;
        }
    }

    // Unlit Logic check (if texture_type is -1 and emission is super high, maybe unlit?)
    // For now, we assume standard lighting path unless explicit unlit flag exists.

    // 3. Prepare Accumulators
    float3 totalDiffuse = float3(0, 0, 0);
    float3 totalSpecular = float3(0, 0, 0);
    float3 totalAmbient = float3(0, 0, 0);

    // 4. Lighting Loop
    for (uint32_t i = 0; i < lightCount; ++i) {
        CalculateLightContribution(
            lights[i], 
            input.world_pos, 
            N, V, 
            base_color.rgb, 
            pushConst.material_specular, 
            pushConst.material_shininess,
            pushConst.material_reflectance, // Pass reflectance
            totalDiffuse,   // ref out
            totalSpecular,  // ref out
            totalAmbient    // ref out
        );
    }

    // 5. Combine Components
    float3 final_color = totalAmbient + totalDiffuse + totalSpecular;
    
    // Add Emission
    final_color += base_color.rgb * pushConst.material_emission;

    // ========================================================================
    // DEBUG VIEWS (Use these to check your data)
    // ========================================================================
    #if DEBUG_VIEW == 1
        // View Normals (Remapped -1..1 to 0..1)
        return float4(N * 0.5 + 0.5, 1.0);
    #elif DEBUG_VIEW == 2
        // View Reflectance Heatmap
        return float4(HeatMap(pushConst.material_reflectance), 1.0);
    #elif DEBUG_VIEW == 3
        // View Diffuse Lighting Only (No Texture, White Base)
        return float4(totalDiffuse / (max(base_color.rgb, 0.001)), 1.0); 
    #elif DEBUG_VIEW == 4
        // View Specular Only
        return float4(totalSpecular, 1.0);
    #elif DEBUG_VIEW == 5
        // View Linear Final Color (Before Tone Mapping)
        return float4(final_color, 1.0);
    #endif

    // ========================================================================
    // POST PROCESSING
    // ========================================================================
    
    // ACES Tone Mapping (More filmic than simple Reinhard)
    final_color = ACESToneMapping(final_color);

    // Gamma Correction (Linear -> sRGB)
    final_color = pow(final_color, 1.0 / 2.2);

    return float4(final_color, base_color.a);
}