struct VSInput {
    float3 pos : POSITION;
    float3 col : COLOR0;
};

struct VSOutput {
    float4 pos : SV_POSITION;
    float3 col : COLOR0;
};

VSOutput VSMain(VSInput input) {
    VSOutput output;
    output.pos = float4(input.pos, 1.0);
    output.col = input.col;
    return output;
}
