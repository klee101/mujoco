// brdf_lut.hlsl
#define PI 3.14159265359

RWTexture2D<float2> outLUT : register(u0);

// Radical Inverse VdC
float radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

float2 Hammersley(uint i, uint N) {
    return float2(float(i)/float(N), radicalInverse_VdC(i));
}

float3 ImportanceSampleGGX(float2 Xi, float Roughness, float3 N) {
    float a = Roughness * Roughness;
    float Phi = 2.0 * PI * Xi.x;
    float CosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    float SinTheta = sqrt(1.0 - CosTheta * CosTheta);

    float3 H;
    H.x = SinTheta * cos(Phi);
    H.y = SinTheta * sin(Phi);
    H.z = CosTheta;

    float3 UpVector = abs(N.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 TangentX = normalize(cross(UpVector, N));
    float3 TangentY = cross(N, TangentX);

    return TangentX * H.x + TangentY * H.y + N * H.z;
}

float2 IntegrateBRDF(float Roughness, float NoV) {
    float3 V;
    V.x = sqrt(1.0 - NoV * NoV);
    V.y = 0.0;
    V.z = NoV;

    float A = 0.0;
    float B = 0.0;
    float3 N = float3(0.0, 0.0, 1.0);
    float k = Roughness * Roughness / 2.0; // IBL roughness remap

    const uint samples = 1024u;
    for(uint i = 0u; i < samples; ++i) {
        float2 Xi = Hammersley(i, samples);
        float3 H = ImportanceSampleGGX(Xi, Roughness, N);
        float3 L = normalize(2.0 * dot(V, H) * H - V);

        float NoL = max(L.z, 0.0);
        float NoH = max(H.z, 0.0);
        float VoH = max(dot(V, H), 0.0);

        if(NoL > 0.0) {
            float G = NoL / (lerp(NoV, 1.0, k) * lerp(NoL, 1.0, k)); // Schlick-GGX
            float G_Vis = (G * VoH) / (NoH * NoV);
            float Fc = pow(1.0 - VoH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return float2(A, B) / float(samples);
}

[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float width, height;
    outLUT.GetDimensions(width, height);

    if (id.x >= width || id.y >= height) return;

    // UV mapping: x -> NdotV, y -> Roughness
    float NoV = (float(id.x) + 0.5) / width;
    float roughness = (float(id.y) + 0.5) / height;

    // 为了更好的采样，通常会对 roughness 进行翻转或者非线性映射，这里保持标准线性
    // 注意：有些引擎是 1.0 - roughness，取决于你的主 Shader 怎么采

    outLUT[id.xy] = IntegrateBRDF(roughness, NoV);
}