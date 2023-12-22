#include <curand_kernel.h>
#include <optix.h>
#include "optixlight.h"
#include <cuda/random.h>
#include <sutil/vec_math.h>


extern "C" {
__constant__ Params params;
}


static __forceinline__ __device__ void cosine_sample_hemisphere(const float u1, const float u2, float3& p)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  p.x = r * cosf( phi );
  p.y = r * sinf( phi );

  // Project up to hemisphere.
  p.z = sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) );
}


static __forceinline__ __device__ void sample_source(
    curandState &rng_state,
    float3 &ray_origin,
    float3 &ray_direction,
    float3 &color)
{
    Face *face;
    SourceEntry *se = NULL;
    int s, t;
    float scale;

    // Binary search for a random source luxel.
    {
        unsigned int v = curand(&rng_state);
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
    s = se->tc.x + curand_uniform(&rng_state);
    t = se->tc.y + curand_uniform(&rng_state);

    float3 tc = make_float3(s, t, 1);
    float3 ray_dir_local;

    ray_origin.x = dot(tc, face->tc_to_world_0);
    ray_origin.y = dot(tc, face->tc_to_world_1);
    ray_origin.z = dot(tc, face->tc_to_world_2);

    cosine_sample_hemisphere(curand_uniform(&rng_state),
                             curand_uniform(&rng_state), ray_dir_local);
    ray_direction = ray_dir_local.x * face->tangent1;
    ray_direction += ray_dir_local.y * face->tangent2;
    ray_direction += ray_dir_local.z * face->normal;

    scale = 1.0f / (se->p * params.rays_per_thread);
    color.x = scale * (float)se->color.x;
    color.y = scale * (float)se->color.y;
    color.z = scale * (float)se->color.z;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    float3 ray_origin, ray_direction, color;
    unsigned int texel_idx;
    curandState rng_state;

    curand_init(0, idx.x, 0, &rng_state);

    for (int i = 0; i < params.rays_per_thread; i++)
    {
        sample_source(rng_state, ray_origin, ray_direction, color);

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
                texel_idx);

        if (texel_idx != 0xffffffff)
        {
            atomicAdd(&params.output[texel_idx + 0], color.x);
            atomicAdd(&params.output[texel_idx + 1], color.y);
            atomicAdd(&params.output[texel_idx + 2], color.z);
        }
    }
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0xffffffff);
}


extern "C" __global__ void __closesthit__ch()
{
    HitData* rt_data  = reinterpret_cast<HitData*>( optixGetSbtDataPointer() );
    Face* face = &params.faces[rt_data->face_idx];
    const float3 ray_dir = optixGetWorldRayDirection();
    const float3 poi = optixGetWorldRayOrigin() + optixGetRayTmax() * ray_dir;
    const float4 poi4 = make_float4(poi.x, poi.y, poi.z, 1);
    int s = static_cast<int>(dot(poi4, face->world_to_tc_0));
    int t = static_cast<int>(dot(poi4, face->world_to_tc_1));

    s = max(0, min(face->lm_width - 1, s));
    t = max(0, min(face->lm_height - 1, t));

    optixSetPayload_0(face->lm_offset + 3 * (s + t * face->lm_width));
}
