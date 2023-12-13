struct Params
{
    unsigned int seed;
    unsigned int *counts;
    unsigned int width;

    float3       light_origin;

    OptixTraversableHandle handle;
};
