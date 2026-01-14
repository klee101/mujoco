// ============================================================================
// RESOURCES
// ============================================================================

cbuffer CameraData : register(b0, space0) {
    float4x4 view_proj;
    float3 camera_position;
    float padding;
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

    float3 padding_l; 
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
    float2 padding_pc;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConst;

// ============================================================================
// TEXTURE RESOURCES (Bindless & IBL)
// ============================================================================
Texture2D g_textures[] : register(t0, space1);
SamplerState g_sampler : register(s1, space1);

Texture2D g_shadowMap : register(t2, space0);
SamplerState g_shadowSampler : register(s2, space0);

// [NEW] BRDF LUT for Split-Sum Approximation
// This matches the C++ descriptor set binding 3
Texture2D g_brdfLUT : register(t3, space0); 
SamplerState g_brdfSampler : register(s3, space0); // Typically linear clamp

TextureCube g_envMap : register(t4, space0);
SamplerState g_envSampler : register(s4, space0);

struct PSInput {
    float4 position : SV_Position;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
    float4 color : COLOR;
    float3 world_pos : WORLD_POS;
    float3 view_dir : VIEW_DIR;
    float3 sample_vec : SAMPLE_VEC;
    float4 shadow_coord : SHADOW_COORD;
};

// ============================================================================
// DEBUG MODES
// ============================================================================
#ifndef DEBUG_VIEW
  #define DEBUG_VIEW 0
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================
static const float IBL_INTENSITY = 0.0; 
static const float EXPOSURE = 1.0;      // 曝光值，配合 Tone Mapping 使用

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static float3 HeatMap(float t) {
    float3 cold = float3(0, 0, 1);
    float3 mid = float3(0, 1, 0);
    float3 hot = float3(1, 0, 0);
    return t < 0.5 ? lerp(cold, mid, t * 2.0) : lerp(mid, hot, (t - 0.5) * 2.0);
}

float3 ACESToneMapping(float3 color) {
    const float A = 2.51f;
    const float B = 0.03f;
    const float C = 2.43f;
    const float D = 0.59f;
    const float E = 0.14f;
    return saturate((color * (A * color + B)) / (color * (C * color + D) + E));
}

// Spherical Equirectangular Mapping
float2 CalculateSphericalUV(float3 v) {
    float phi = atan2(v.x, v.z); 
    float theta = asin(clamp(v.y, -1.0, 1.0));
    const float PI = 3.14159265359;
    float u = (phi / (2.0 * PI)) + 0.5;
    float v_coord = (theta / PI) + 0.5; 
    return float2(u, 1.0 - v_coord); 
}

// [NEW] Fresnel Schlick approximation
float3 fresnelSchlickRoughness(float cosTheta, float3 F0, float roughness) {
    return F0 + (max(float3(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// [NEW] IBL Contribution Function
// Calculates Diffuse Irradiance + Specular Image Based Lighting
float3 IBL_Contribution(
    float3 N, 
    float3 V, 
    float3 R, 
    float3 albedo, 
    float3 F0, 
    float roughness, 
    float metallic,
    int envMapIndex
) {
    // 1. Diffuse IBL (Irradiance)
    // We approximate irradiance by sampling the environment map at a very high mip level.
    // This blurs the details, leaving average color.
    
    float3 kS = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    float3 kD = 1.0 - kS;
    kD *= (1.0 - metallic); // Metals have no diffuse

    // [FIX] Cubemap 采样: 直接使用法线 N 作为方向
    // 使用高 Mip Level (6.0) 模拟辐照度模糊
    float3 irradiance = g_envMap.SampleLevel(g_envSampler, N, 6.0).rgb; 
    float3 diffuse = irradiance * albedo;

    // 2. Specular IBL
    // A. Prefiltered Map: 直接使用反射向量 R 采样 Cubemap
    const float MAX_REFLECTION_LOD = 8.0; 
    float lod = roughness * MAX_REFLECTION_LOD;
    float3 prefilteredColor = g_envMap.SampleLevel(g_envSampler, R, lod).rgb;

    // B. BRDF LUT (保持不变)
    float NdotV = max(dot(N, V), 0.0);
    float2 envBRDF = g_brdfLUT.Sample(g_brdfSampler, float2(NdotV, roughness)).rg;

    float3 specular = prefilteredColor * (F0 * envBRDF.x + envBRDF.y);

    return (kD * diffuse + specular); 
}


// ============================================================================
// LIGHTING CALCULATION (Direct Light)
// ============================================================================
void CalculateLightContribution(
    LightInfo light, 
    float3 worldPos,
    float3 N, 
    float3 V, 
    float3 matDiffuse, 
    float matSpecular, 
    float matShininess, 
    float matReflectance, 
    inout float3 outDiffuse,
    inout float3 outSpecular,
    inout float3 outAmbient,
    float visibility
) {
    float3 L;
    float attenuation = 1.0;

    if (light.type == 1) { // Directional
        L = normalize(-light.direction);
    } else { // Point or Spot
        float3 lightVec = light.position - worldPos;
        float d = length(lightVec);
        L = lightVec / max(d, 0.0001);
        
        float denom = light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d;
        attenuation = 1.0 / max(denom, 0.0001);

        if (light.type == 0) { // Spot Cutoff
            float theta = dot(-L, normalize(light.direction));
            float outerCutoff = cos(radians(light.cutoff));
            float epsilon = 0.1;
            float spotIntensity = smoothstep(outerCutoff, outerCutoff + epsilon, theta);
            attenuation *= pow(spotIntensity, light.exponent);
        }
    }

    if (attenuation < 0.0001) return;

    // --- Diffuse ---
    float NdotL = max(dot(N, L), 0.0);
    outDiffuse += light.diffuse * matDiffuse * NdotL * attenuation * visibility;

    // --- Ambient ---
    outAmbient += light.ambient * matDiffuse; // This is direct light ambient

    // --- Specular (Schlick Fresnel + Blinn-Phong) ---
    if (NdotL > 0.0) {
        float3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        float baseF   = matReflectance; 
        float fresnel = baseF + (1.0 - baseF) * pow(1.0 - HdotV, 5.0);

        float specPower = pow(NdotH, matShininess);
        float specIntensity = matSpecular * specPower * fresnel * attenuation * visibility;
        
        outSpecular += light.specular * specIntensity;
    }
}

// Helper to sample shadow map with PCF
// 在文件顶部定义泊松分布点 (16个采样点)
static const float2 poissonDisk[16] = {
   float2( -0.94201624, -0.39906216 ), float2( 0.94558609, -0.76890725 ),
   float2( -0.09418410, -0.92938870 ), float2( 0.34495938, 0.29387760 ),
   float2( -0.91588581, 0.45771432 ), float2( -0.81544232, -0.87912464 ),
   float2( -0.38277543, 0.27676845 ), float2( 0.97484398, 0.75648379 ),
   float2( 0.44323325, -0.97511554 ), float2( 0.53742981, -0.47371076 ),
   float2( -0.26496911, -0.41893023 ), float2( 0.79197514, 0.19090188 ),
   float2( -0.24188840, 0.99706507 ), float2( -0.81409955, 0.91437590 ),
   float2( 0.19984126, 0.78641367 ), float2( 0.14383161, -0.14100790 )
};

// [MODIFIED] Shadow Calculation utilizing Poisson Sampling
float ShadowCalculation(float4 fragPosLightSpace, float3 normal, float3 lightDir) {
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = projCoords.y * 0.5 + 0.5;
    
    // 边界检查
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;

    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);
    float shadow = 0.0;
    
    // [FIX] 不要硬编码尺寸，使用 GetDimensions
    float width, height;
    g_shadowMap.GetDimensions(width, height);
    float2 texelSize = 1.0 / float2(width, height);

    // 泊松采样循环
    // "diskRadius" 控制阴影软边缘的宽度。值越大越软，但也可能越“飘”
    float diskRadius = 3.0; 
    
    for(int i = 0; i < 16; ++i) {
        // 使用 poissonDisk 偏移采样
        float pcfDepth = g_shadowMap.Sample(g_shadowSampler, projCoords.xy + poissonDisk[i] * texelSize * diskRadius).r; 
        shadow += (projCoords.z - bias > pcfDepth ? 1.0 : 0.0);
    }
    
    return shadow / 16.0;
}

// ============================================================================
// PIXEL SHADER MAIN
// ============================================================================
float4 PSMain(PSInput input) : SV_Target {
    float3 N = normalize(input.normal);
    float3 V = normalize(input.view_dir);
    float3 R = reflect(-V, N);

    // Prepare Base Material Params
    float4 base_color = pushConst.material_rgba * input.color;
    
    // Apply Texture (Base Color / Albedo)
    if (pushConst.texture_index >= 0) {
        float4 texColor = float4(1,1,1,1);
        if (pushConst.texture_type == 0) { // 2D Albedo
             texColor = g_textures[pushConst.texture_index].Sample(g_sampler, input.texcoord);
        } else if (pushConst.texture_type == 1) { // Env Map (Used as color? Unusual but kept for compatibility)
             float2 spherical_uv = CalculateSphericalUV(normalize(input.sample_vec));
             texColor = g_textures[pushConst.texture_index].Sample(g_sampler, spherical_uv);
        }
        base_color *= texColor;
    }

    // --------------------------------------------------------
    // PBR / IBL Parameter Derivation (MuJoCo -> PBR approximation)
    // --------------------------------------------------------
    // 1. Convert Blinn-Phong Shininess to Roughness
    //    shininess in [0,1] mapped to [0,128] for conversion 
    float shininess = clamp(pushConst.material_shininess * 128.0, 0.0, 128.0);

    // Industry standard approximation
    float roughness = sqrt(2.0 / (shininess + 2.0));
    roughness = clamp(roughness, 0.04, 1.0);
    // 2. Reflectance (F0)
    //    Assumes dielectric unless specified. F0 is typically 0.04 for plastics.
    //    MuJoCo's 'reflectance' parameter maps reasonably well to F0 magnitude.
    float3 F0 = float3(pushConst.material_reflectance, pushConst.material_reflectance, pushConst.material_reflectance);
    float metallic = 0.0; // Default to dielectric

    // --------------------------------------------------------
    // Direct Lighting Loop
    // --------------------------------------------------------
    float3 totalDiffuse = float3(0, 0, 0);
    float3 totalSpecular = float3(0, 0, 0);
    float3 totalAmbient = float3(0, 0, 0);

    uint safeLightCount = min(input_lightCount, 10);

    for(uint i = 0; i < safeLightCount; ++i)
    {
        float visibility = 1.0;
        // Shadow logic (Light 1 only for this demo)
        if (i == 1) { 
            float3 L;
            if (input_lights[i].type == 1) L = normalize(-input_lights[i].direction);
            else L = normalize(input_lights[i].position - input.world_pos);
            visibility = 1.0 - ShadowCalculation(input.shadow_coord, N, L);
        }
        
        CalculateLightContribution(
            input_lights[i], input.world_pos, N, V, 
            base_color.rgb, 
            pushConst.material_specular, 
            pushConst.material_shininess, 
            pushConst.material_reflectance,
            totalDiffuse, totalSpecular, totalAmbient,
            visibility
        );
    }

    // --------------------------------------------------------
    // IBL Contribution
    // --------------------------------------------------------
    // float3 ambientIBL = float3(0,0,0);
    
    // // Determine which environment map to use. 
    // // Logic: If the object has a texture of type 1 (EnvMap), use it for IBL.
    // // Ideally, you should pass a 'GlobalSkyboxIndex' in PushConstants if no local env map exists.
    // int envMapIdx = (pushConst.texture_type == 1) ? pushConst.texture_index : 0;
    
    // if (envMapIdx >= 0) {
    //     ambientIBL = IBL_Contribution(N, V, R, base_color.rgb, F0, roughness, metallic, envMapIdx);
    //     ambientIBL *= IBL_INTENSITY;
    // } else {
    //     ambientIBL = totalAmbient; 
    // }

    // --------------------------------------------------------
    // Final Combination
    // --------------------------------------------------------
    float3 final_color = totalDiffuse + totalSpecular;
    
    // Replace constant ambient with IBL if available
    final_color += totalAmbient;

    final_color += base_color.rgb * pushConst.material_emission;

    #if DEBUG_VIEW == 1
        return float4(N * 0.5 + 0.5, 1.0); 
    #elif DEBUG_VIEW == 2
        return float4(HeatMap(roughness), 1.0); 
    #elif DEBUG_VIEW == 3
        return float4(ambientIBL, 1.0); 
    #elif DEBUG_VIEW == 4
        return float4(totalSpecular, 1.0); 
    #endif

    // Tone Mapping & Gamma
    final_color = ACESToneMapping(final_color);
    final_color = pow(final_color, 1.0 / 2.2);
    final_color *= EXPOSURE;

    return float4(final_color, base_color.a);
}