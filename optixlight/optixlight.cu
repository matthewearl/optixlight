#include <optix.h>
#include "optixlight.h"
#include <cuda/random.h>
#include <sutil/vec_math.h>


extern "C" {
__constant__ Params params;
}


#define SAMPLE_SOURCE


#ifdef SAMPLE_SOURCE
static __forceinline__ __device__ void sample_source(unsigned int &seed)
{
    Face *face;
    SourceEntry *se = NULL;
    int s, t, i;
    float3 *output_el;
    float scale;

    // Binary search for a random source luxel.
    {
        unsigned int v = ((lcg(seed) & 0xffffff) << 8) | (lcg(seed) & 0xff);
        int lo, hi, mid;
        lo = -1;
        hi = params.num_source_entries - 1;
        while (lo != hi - 1)
        {
            mid = (lo + hi) >> 1;
            if (v < params.source_cdf[mid])
                hi = mid;
            else
                lo = mid;
        }
        se = &params.source_entries[lo + 1];
    }

    face = &params.faces[se->face_idx];
    s = se->tc.x;
    t = se->tc.y;

    scale = 1.0f / (se->p * params.rays_per_thread);
    atomicAdd(
        &params.output[face->lm_offset + 3 * (s + t * face->lm_width) + 0],
        se->color.x * scale
    );
    atomicAdd(
        &params.output[face->lm_offset + 3 * (s + t * face->lm_width) + 1],
        se->color.y * scale
    );
    atomicAdd(
        &params.output[face->lm_offset + 3 * (s + t * face->lm_width) + 2],
        se->color.z * scale
    );
}
#else
static __forceinline__ __device__ void sample_sphere(const float u1, const float u2, float3& p)
{
    const float theta = 2.0f * M_PIf * u1;
    const float phi = acosf(2.0f * u2 - 1.0f);

    p.x = sinf(phi) * cosf(theta);
    p.y = sinf(phi) * sinf(theta);
    p.z = cosf(phi);
}
#endif

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    unsigned int seed = tea<4>(idx.x, params.seed);

#ifdef SAMPLE_SOURCE
    for (int i = 0; i < params.rays_per_thread; i++)
        sample_source(seed);
#else
    // Trace the ray against our scene hierarchy
    const float3 ray_origin = params.light_origin;
    float3 ray_direction;
    unsigned int p0;

    for (int i = 0; i < params.rays_per_thread; i++)
    {
        sample_sphere(rnd(seed), rnd(seed), ray_direction);

        optixTrace(
                params.handle,
                ray_origin,
                ray_direction,
                0.0f,                // Min intersection distance
                1e16f,               // Max intersection distance
                0.0f,                // rayTime -- used for motion blur
                OptixVisibilityMask(255), // Specify always visible
                OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES,
                0,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                p0);
        atomicAdd(&params.counts[p0], 1);
    }
#endif
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0);
}


extern "C" __global__ void __closesthit__ch()
{
    HitData* rt_data  = reinterpret_cast<HitData*>( optixGetSbtDataPointer() );
    Face* face = &params.faces[rt_data->face_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 poi = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    const float4 poi4 = make_float4(poi.x, poi.y, poi.z, 1);
    int s = static_cast<int>(dot(poi4, face->m0));
    int t = static_cast<int>(dot(poi4, face->m1));

    s = max(0, min(face->lm_width - 1, s));
    t = max(0, min(face->lm_height - 1, t));
    atomicAdd(&params.output[face->lm_offset + s + t * face->lm_width], 1);

    optixSetPayload_0(1 + rt_data->face_idx);
}
