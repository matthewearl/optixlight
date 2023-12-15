struct Params
{
    OptixTraversableHandle handle;
    unsigned int *counts;
    unsigned int *output;
    unsigned int seed;
    float3 light_origin;
};

struct HitData
{
    float4 m0;
    float4 m1;
    unsigned int idx;
};
