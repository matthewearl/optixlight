struct Face
{
    float4 m0;
    float4 m1;
    unsigned int lm_width;
    unsigned int lm_height;
    unsigned int lm_offset;
};

struct Params
{
    OptixTraversableHandle handle;
    unsigned int *counts;
    unsigned int *output;
    Face *faces;
    unsigned int seed;
    float3 light_origin;
};

struct HitData
{
    int face_idx;
};
