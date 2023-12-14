struct Params
{
    unsigned int seed;
    unsigned int *counts;
    unsigned int width;

    float3       light_origin;

    float pad;

    OptixTraversableHandle handle;
};
