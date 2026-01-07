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

    uint castShadow;
    float padding[3]; 

    float4x4 view_proj;
};

cbuffer LightData : register(b1, space0) {
    LightInfo input_lights[10];
    uint input_lightCount;
    float pad[3];
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
    int padding[2];
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConst;

// ============================================================================
// TEXTURE RESOURCES (Bindless)
// ============================================================================
Texture2D g_textures[] : register(t0, space1);
SamplerState g_sampler : register(s1, space1);

Texture2D g_shadowMap : register(t2, space0);
SamplerState g_shadowSampler : register(s2, space0);

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

// [NEW] Spherical Equirectangular Mapping
// Maps a 3D vector to a 2D texture coordinate smoothly (like a world map)
float2 CalculateSphericalUV(float3 v) {
    // 1. Calculate the angle in the XZ plane (Horizontal)
    // atan2(x, z) returns range [-PI, PI]
    float phi = atan2(v.x, v.z); 
    
    // 2. Calculate the elevation angle (Vertical)
    // asin(y) returns range [-PI/2, PI/2] (Assuming normalized v)
    float theta = asin(clamp(v.y, -1.0, 1.0));

    // 3. Map to [0, 1] UV space
    const float PI = 3.14159265359;
    float u = (phi / (2.0 * PI)) + 0.5;
    float v_coord = (theta / PI) + 0.5; 

    // Optional: Flip V if texture is upside down
    return float2(u, 1.0 - v_coord); 
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
    float matSpecular,  // Specular Color
    float matShininess, // Shininess
    float matReflectance, 
    inout float3 outDiffuse,
    inout float3 outSpecular,
    inout float3 outAmbient,
    float visibility
) {
    float3 L;
    float attenuation = 1.0;

    // --- Light Vector Setup ---
    // Note: MuJoCo directional lights have 'direction' pointing along the light ray.
    // For lighting calculations, L needs to point TOWARDS the light source.
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
    outAmbient += light.ambient * matDiffuse;

    // --- Specular (Schlick Fresnel + Blinn-Phong) ---
    if (NdotL > 0.0) {
        float3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float HdotV = max(dot(H, V), 0.0);

        // Fresnel (scalar)
        float baseF   = matReflectance; 
        float fresnel = baseF + (1.0 - baseF) * pow(1.0 - HdotV, 5.0);

        float specPower = pow(NdotH, matShininess);

        // matSpecular 
        float specIntensity = matSpecular * specPower * fresnel * attenuation * visibility;
        
        outSpecular += light.specular * specIntensity;
    }
}

// [+] NEW: Helper to sample shadow map with PCF
float ShadowCalculation(float4 fragPosLightSpace, float3 normal, float3 lightDir) {
    // 1. Perspective Divide
    float3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;

    // 2. Transform from NDC [-1,1] to Texture [0,1]
    // Vulkan Y is flipped compared to OpenGL, but if we used standard projection:
    projCoords.x = projCoords.x * 0.5 + 0.5;
    projCoords.y = projCoords.y * 0.5 + 0.5;
    
    // Check if outside map
    if (projCoords.z > 1.0 || projCoords.x < 0.0 || projCoords.x > 1.0 || projCoords.y < 0.0 || projCoords.y > 1.0)
        return 0.0;

    // 3. Bias (Slope Scale based)
    float bias = max(0.005 * (1.0 - dot(normal, lightDir)), 0.0005);
    
    // 4. PCF (Percentage Closer Filtering) 3x3
    float shadow = 0.0;
    float2 texelSize = 1.0 / 2048.0; // Hardcoded size, ideally pass via CBuffer
    
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcfDepth = g_shadowMap.Sample(g_shadowSampler, projCoords.xy + float2(x, y) * texelSize).r; 
            // If current depth > stored depth + bias, it is in shadow
            shadow += (projCoords.z - bias > pcfDepth ? 1.0 : 0.0);        
        }    
    }
    shadow /= 9.0;
    
    return shadow;
}

// ============================================================================
// PIXEL SHADER MAIN
// ============================================================================
float4 PSMain(PSInput input) : SV_Target {
    // 1. Prepare Geometry Vectors
    float3 N = normalize(input.normal);
    float3 V = normalize(input.view_dir);

    // 2. Prepare Material Color (Base Color * Texture)
    float4 base_color = pushConst.material_rgba * input.color;
    
    // Texture Logic
    if (pushConst.texture_index >= 0) {
        float4 texColor = float4(1,1,1,1);
        
        if (pushConst.texture_type == 0) { // 2D Texture
            texColor = g_textures[NonUniformResourceIndex(pushConst.texture_index)].Sample(g_sampler, input.texcoord);
        } 
        else if (pushConst.texture_type == 1) { // Simulated Environment Map
            float3 uvw = normalize(input.sample_vec);
            
            // [CHANGE] Use Spherical instead of Cube to fix the "Prism" look
            float2 spherical_uv = CalculateSphericalUV(uvw);
            
            texColor = g_textures[NonUniformResourceIndex(pushConst.texture_index)].Sample(g_sampler, spherical_uv);
        }
        
        base_color *= texColor;
    }

    // 3. Prepare Lighting Accumulators
    float3 totalDiffuse = float3(0, 0, 0);
    float3 totalSpecular = float3(0, 0, 0);
    float3 totalAmbient = float3(0, 0, 0);

    // Clamp to max 10 to prevent infinite loops if memory is garbage
    uint safeLightCount = min(input_lightCount, 10);



    for(uint i = 0; i < safeLightCount; ++i)
    {
        float visibility = 1.0;
        // Only Light 0 casts shadows in this implementation
        if (i == 1) { 
            float3 L;
            if (input_lights[i].type == 1) L = normalize(-input_lights[i].direction);
            else L = normalize(input_lights[i].position - input.world_pos);
            
            // Returns 1.0 if in shadow, so we subtract
            visibility = 1.0 - ShadowCalculation(input.shadow_coord, N, L);
        }
        CalculateLightContribution(
            input_lights[i], 
            input.world_pos, 
            N, 
            V, 
            base_color.rgb, 
            pushConst.material_specular, 
            pushConst.material_shininess, 
            pushConst.material_reflectance,
            totalDiffuse, 
            totalSpecular, 
            totalAmbient,
            visibility
        );
    }

    // 4. Combine Lighting
    float3 final_color = totalAmbient + totalDiffuse + totalSpecular;
    
    // Add Emission
    final_color += base_color.rgb * pushConst.material_emission;

    // ========================================================================
    // DEBUG VIEWS
    // ========================================================================
    #if DEBUG_VIEW == 1
        return float4(N * 0.5 + 0.5, 1.0); // Normals
    #elif DEBUG_VIEW == 2
        return float4(HeatMap(pushConst.material_reflectance), 1.0); // Reflectance
    #elif DEBUG_VIEW == 3
        return float4(totalDiffuse + totalAmbient, 1.0); // Diffuse Only
    #elif DEBUG_VIEW == 4
        return float4(totalSpecular, 1.0); // Specular Only
    #elif DEBUG_VIEW == 5
        return float4(input.texcoord, 0.0, 1.0); // UVs
    #endif

    // ========================================================================
    // POST PROCESSING
    // ========================================================================
    
    // Tone Mapping
    final_color = ACESToneMapping(final_color);
    
    // Gamma Correction
    final_color = pow(final_color, 1.0 / 2.2);

    return float4(final_color, base_color.a);
}