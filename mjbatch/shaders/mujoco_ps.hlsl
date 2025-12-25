// ============================================================================
// RESOURCES
// ============================================================================

cbuffer CameraData : register(b0, space0) {
    float4x4 view_proj;
    float3 camera_position;
    float padding; // Padding to align to 16 bytes
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
    
    float3 attenuation; // x: constant, y: linear, z: quadratic
    float intensity;    

    uint castShadow;
    float padding[3]; 
};

// Even if we don't use the input data, we MUST keep the definition
// so the C++ Descriptor Set binding doesn't crash.
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
    int texture_type;  // -1: Unlit/Color, 0: 2D Texture, 1: Cube Texture
    int padding[2];
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pushConst;

// ============================================================================
// TEXTURE RESOURCES (Bindless)
// ============================================================================
Texture2D g_textures[] : register(t0, space1);
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
// 0: Off (Final Render)
// 1: Normals
// 2: Reflectance Heatmap
// 3: Lighting Only (White Material)
// 4: Specular Only
// 5: UV Coords

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

// Manually calculate Cube Map UVs for 2D Texture Arrays
float2 CalculateCubeUV(float3 v) {
    float3 vAbs = abs(v);
    float ma;
    float2 uv;
    
    if(vAbs.z >= vAbs.x && vAbs.z >= vAbs.y) {
        ma = vAbs.z;
        uv = float2(v.x, -v.y); 
    } else if(vAbs.y >= vAbs.x) {
        ma = vAbs.y;
        uv = float2(v.x, v.z);
    } else {
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
    float matSpecular, // Specular Color
    float matShininess, // Shininess
    float matReflectance, 
    inout float3 outDiffuse,
    inout float3 outSpecular,
    inout float3 outAmbient
) {
    float3 L;
    float attenuation = 1.0;

    // --- Light Vector Setup ---
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
    outDiffuse += light.diffuse * matDiffuse * NdotL * attenuation;

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
        float specIntensity = matSpecular * specPower * fresnel * attenuation;
        
        outSpecular += light.specular * specIntensity;
}

}

// ============================================================================
// PIXEL SHADER MAIN
// ============================================================================
float4 PSMain(PSInput input) : SV_Target {
    // 1. Prepare Geometry Vectors
    float3 N = normalize(input.normal);
    float3 V = normalize(input.view_dir);
    // Optional: Double-sided fix
    // if (dot(N, V) < 0) N = -N; 

    // 2. Prepare Material Color (Base Color * Texture)
    float4 base_color = pushConst.material_rgba * input.color;
    
    // Texture Logic (NonUniformResourceIndex handles bindless arrays safely)
    if (pushConst.texture_index >= 0) {
        float4 texColor = float4(1,1,1,1);
        
        if (pushConst.texture_type == 0) { 
            // 2D Texture
            texColor = g_textures[NonUniformResourceIndex(pushConst.texture_index)].Sample(g_sampler, input.texcoord);
        } 
        else if (pushConst.texture_type == 1) { 
            // Simulated Cube Map
            float3 uvw = normalize(input.sample_vec);
            float2 cube_uv = CalculateCubeUV(uvw);
            texColor = g_textures[NonUniformResourceIndex(pushConst.texture_index)].Sample(g_sampler, cube_uv);
        }
        
        base_color *= texColor;
    }

    // 3. Prepare Lighting Accumulators
    float3 totalDiffuse = float3(0, 0, 0);
    float3 totalSpecular = float3(0, 0, 0);
    float3 totalAmbient = float3(0, 0, 0);

    // ========================================================================
    // [LOGIC] DEFAULT STUDIO LIGHTING SETUP
    // Ignores input_lights[] from UBO entirely.
    // ========================================================================
    
    // Light 1: Key Light (Warm Sun from Top-Right)
    LightInfo keyLight;
    keyLight.type = 1; // Directional
    keyLight.direction = normalize(float3(-1.0, -2.0, -1.0)); // Coming from Top-Right-Front
    keyLight.diffuse = float3(1.0, 0.95, 0.9); // Warm Light
    keyLight.specular = float3(1.0, 1.0, 1.0);
    keyLight.ambient = float3(0.05, 0.05, 0.08); // Slight blue ambient
    keyLight.attenuation = float3(1,0,0); // Unused for Directional
    keyLight.position = float3(0,0,0);
    keyLight.range = 0;
    keyLight.cutoff = 0;
    keyLight.exponent = 0;
    keyLight.bulbRadius = 0;
    keyLight.intensity = 1.0;
    keyLight.castShadow = 0;

    CalculateLightContribution(
        keyLight, input.world_pos, N, V, 
        base_color.rgb, 
        pushConst.material_specular, 
        pushConst.material_shininess, 
        pushConst.material_reflectance,
        totalDiffuse, totalSpecular, totalAmbient
    );

    // Light 2: Fill Light (Headlight / Camera Light)
    // Ensures nothing is ever pitch black by lighting from the view direction
    LightInfo fillLight;
    fillLight.type = 1; // Directional
    fillLight.direction = -V; // Parallel to View Direction (Headlight)
    fillLight.diffuse = float3(0.3, 0.3, 0.35); // Cool, dimmer light
    fillLight.specular = float3(0.2, 0.2, 0.2);
    fillLight.ambient = float3(0.0, 0.0, 0.0); // Ambient handled by Key Light
    fillLight.attenuation = float3(1,0,0);
    // Fill dummy values for struct
    fillLight.position = float3(0,0,0);
    fillLight.range = 0; fillLight.cutoff = 0; fillLight.exponent = 0;
    fillLight.bulbRadius = 0; fillLight.intensity = 1.0; fillLight.castShadow = 0;

    CalculateLightContribution(
        fillLight, input.world_pos, N, V, 
        base_color.rgb, 
        pushConst.material_specular, 
        pushConst.material_shininess, 
        pushConst.material_reflectance,
        totalDiffuse, totalSpecular, totalAmbient
    );

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
        // Diffuse Lighting Only (Grey Clay Mode)
        return float4(totalDiffuse + totalAmbient, 1.0);
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